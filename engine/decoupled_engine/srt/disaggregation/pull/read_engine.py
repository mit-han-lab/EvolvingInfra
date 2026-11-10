"""ReadEngine implementations (api.ReadEngine).

``MooncakeReadEngine`` wraps the shared MooncakeTransferEngine and its new
``batch_transfer_sync_read`` passthrough. ``FakeReadEngine`` is an in-process
memcpy engine over fake pointers for CPU tests. ``NixlReadEngine`` is a stub.

Registration: every worker already registers its whole KV
pool + aux + state with the transfer engine at startup
(mooncake/conn.py MooncakeKVManager.register_buffer_to_engine, deduped in
_registerable_regions). The read engine must not register those regions a
second time (double registration fails in the engine), so it is seeded with the
already-registered (ptr, nbytes) set and ``register`` is a no-op for them.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from decoupled_engine.srt.disaggregation.pull.api import ReadNotQuiescentError

logger = logging.getLogger(__name__)

KIND_VRAM = "VRAM"
KIND_DRAM = "DRAM"


def registered_regions_of_kv_args(kv_args: Any) -> List[Tuple[int, int]]:
    """(ptr, nbytes) regions register_buffer_to_engine registers for ``kv_args``,
    same dedupe as mooncake/conn.py MooncakeKVManager._registerable_regions."""
    regions: List[Tuple[int, int]] = []
    seen: set = set()

    def add(ptrs: Sequence[int], lens: Sequence[int]) -> None:
        for ptr, length in zip(ptrs or [], lens or []):
            key = (int(ptr), int(length))
            if key not in seen:
                seen.add(key)
                regions.append(key)

    add(kv_args.kv_data_ptrs, kv_args.kv_data_lens)
    add(kv_args.aux_data_ptrs, kv_args.aux_data_lens)
    for ptrs, lens in zip(kv_args.state_data_ptrs, kv_args.state_data_lens):
        add(ptrs, lens)
    return regions


class MooncakeReadEngine:
    """api.ReadEngine over the shared MooncakeTransferEngine.

    ``session`` in batch_read is the owner's Mooncake session id
    ("host:rpc_port", MooncakeTransferEngine.get_session_id), carried in the
    owner's REGISTER blob (protocol.Registration.session_id).
    """

    def __init__(
        self,
        *,
        te: Any,
        preregistered: Iterable[Tuple[int, int]] = (),
    ) -> None:
        self._te = te
        self._registered: set = {(int(p), int(n)) for p, n in preregistered}
        self._lock = threading.Lock()

    @classmethod
    def from_kv_args(cls, *, te: Any, kv_args: Any) -> "MooncakeReadEngine":
        """Seed with the regions the PD conn registered at startup."""
        return cls(te=te, preregistered=registered_regions_of_kv_args(kv_args))

    @property
    def session_id(self) -> str:
        return self._te.get_session_id()

    def register(self, ptr: int, nbytes: int, kind: str) -> None:
        key = (int(ptr), int(nbytes))
        with self._lock:
            if key in self._registered:
                return
            # Not part of the startup registration (e.g. a scratch buffer). The
            # generic wrapper only logs failures; a READ target that failed to
            # register must never be advertised, so fail loudly here.
            rc = self._te.batch_register([key[0]], [key[1]])
            if rc != 0:
                raise RuntimeError(
                    f"pull read engine: {kind} registration failed ptr=0x{key[0]:x} "
                    f"nbytes={key[1]} rc={rc}"
                )
            self._registered.add(key)

    def batch_read(
        self,
        session: str,
        local_ptrs: Sequence[int],
        remote_ptrs: Sequence[int],
        lengths: Sequence[int],
    ) -> int:
        if not lengths:
            return 0
        # Blocking one-sided READ; called from the importer's transfer thread only.
        try:
            rc = int(
                self._te.batch_transfer_sync_read(
                    session, list(local_ptrs), list(remote_ptrs), list(lengths)
                )
            )
        except Exception as exc:
            raise ReadNotQuiescentError(f"Mooncake READ raised: {exc}") from exc
        if rc != 0:
            # The wrapper exposes no per-descriptor drain/cancel fence on
            # failure. A negative return must not authorize page recycling.
            raise ReadNotQuiescentError(f"Mooncake READ failed: rc={rc}")
        return rc


class FakeMemory:
    """Fake address space: regions of bytes keyed by a fake base pointer."""

    _next_base = 0x1000_0000
    _base_lock = threading.Lock()

    def __init__(self) -> None:
        self._regions: Dict[int, memoryview] = {}

    @classmethod
    def _fresh_base(cls, nbytes: int) -> int:
        with cls._base_lock:
            base = cls._next_base
            # 4 KiB aligned, with a guard gap so out-of-range reads never land
            # in a neighbouring region.
            cls._next_base += ((nbytes + 0x1FFF) // 0x1000) * 0x1000
            return base

    def alloc(self, nbytes: int, *, fill: int = 0) -> int:
        buf = bytearray([fill]) * nbytes
        base = self._fresh_base(nbytes)
        self._regions[base] = memoryview(buf)
        return base

    def attach(self, buf: Any) -> int:
        """Attach a writable bytes-like object (bytearray, numpy uint8 array,
        or a torch CPU tensor via .numpy()) and return its fake base pointer."""
        mv = memoryview(buf).cast("B")
        base = self._fresh_base(len(mv))
        self._regions[base] = mv
        return base

    def locate(self, ptr: int, nbytes: int) -> Tuple[memoryview, int]:
        for base, mv in self._regions.items():
            if base <= ptr and ptr + nbytes <= base + len(mv):
                return mv, ptr - base
        raise KeyError(f"fake ptr 0x{ptr:x}+{nbytes} is not inside any region")

    def view(self, ptr: int, nbytes: int) -> memoryview:
        mv, off = self.locate(ptr, nbytes)
        return mv[off : off + nbytes]

    def contains(self, ptr: int, nbytes: int) -> bool:
        try:
            self.locate(ptr, nbytes)
            return True
        except KeyError:
            return False


class FakeReadEngine:
    """api.ReadEngine doing memcpy between FakeMemory address spaces.

    Every engine is reachable by a session id (its fake "host:port"), so a
    reader engine can READ from an owner engine in the same process exactly
    like Mooncake would across nodes.
    """

    _sessions: Dict[str, "FakeReadEngine"] = {}
    _sessions_lock = threading.Lock()

    def __init__(self, *, session_id: str, memory: Optional[FakeMemory] = None) -> None:
        self.session_id = session_id
        self.memory = memory if memory is not None else FakeMemory()
        self.registered: List[Tuple[int, int, str]] = []
        self.reads: List[Tuple[str, List[int], List[int], List[int]]] = []
        self.fail_next: int = 0  # >0: fail that many batch_read calls
        with self._sessions_lock:
            self._sessions[session_id] = self

    def close(self) -> None:
        with self._sessions_lock:
            if self._sessions.get(self.session_id) is self:
                del self._sessions[self.session_id]

    def register(self, ptr: int, nbytes: int, kind: str) -> None:
        if not self.memory.contains(ptr, nbytes):
            raise RuntimeError(f"fake register of unknown region 0x{ptr:x}+{nbytes}")
        self.registered.append((int(ptr), int(nbytes), kind))

    def batch_read(
        self,
        session: str,
        local_ptrs: Sequence[int],
        remote_ptrs: Sequence[int],
        lengths: Sequence[int],
    ) -> int:
        self.reads.append((session, list(local_ptrs), list(remote_ptrs), list(lengths)))
        if self.fail_next > 0:
            self.fail_next -= 1
            return -1
        with self._sessions_lock:
            remote = self._sessions.get(session)
        if remote is None:
            return -2
        try:
            for lp, rp, n in zip(local_ptrs, remote_ptrs, lengths):
                src = remote.memory.view(int(rp), int(n))
                dst = self.memory.view(int(lp), int(n))
                dst[:] = src
        except KeyError:
            logger.warning(
                "FakeReadEngine: read outside registered regions", exc_info=True
            )
            return -3
        return 0


class NixlReadEngine:
    """Placeholder for the NIXL backend (Mooncake first, NIXL second).

    Shape to copy: vLLM nixl/pull_worker.py:545, which issues
    ``make_prepped_xfer("READ", local_xfer_side, local_block_descs_ids,
    remote_xfer_side, remote_block_descs_ids, notif_msg)`` against the remote's
    pre-registered VRAM descriptors; this engine's nixl/conn.py registers those
    descriptors already but only has WRITE opcodes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("NixlReadEngine: NIXL READ path not implemented yet")

    def register(self, ptr: int, nbytes: int, kind: str) -> None:
        raise NotImplementedError

    def batch_read(
        self,
        session: str,
        local_ptrs: Sequence[int],
        remote_ptrs: Sequence[int],
        lengths: Sequence[int],
    ) -> int:
        raise NotImplementedError
