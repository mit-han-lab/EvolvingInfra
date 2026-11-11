"""C2g plumbing: the copy-engine peer fan-out
(``SGLANG_DSA_HOST_DEDUP_FANOUT=ce_peer``).

The dedup source (TP0) loads from its own private host pool with copy
engines; each follower pulls the same rows from TP0's device buffers with
copy-engine peer copies. This module sets up, once at init:

- the source's device KV / indexer / MTP-draft layer buffers, exported as
  CUDA IPC handles and mapped into every follower (pointers only: the
  followers' copy engines read them, nothing else touches them);
- a one-page shared-memory progress block (memfd, mapped by every rank) with
  one 32-byte slot per rank: a rank that feeds others publishes
  ``[0] = seq << 24 | chunks done``, ``[1] = seq << 24 | chunks in op``,
  ``[2] = error``; a rank that pulls polls its source's slot before each
  chunk. Cross-rank ordering is CPU-only: no stream ever waits on another
  process's GPU work (cross-process stream waits can hang the engine).

Topology (``SGLANG_DSA_HOST_DEDUP_PEER_TOPOLOGY``): ``star`` (every follower
pulls from the dedup source), ``chain`` (rank i pulls from rank i-1) or
``tree`` (rank i pulls from (i-1)//2), in TP-group order; non-star topologies
spread the NVLink egress (and the reads' contention with the forward's
all-reduce) over the ranks.
"""

from __future__ import annotations

import ctypes
import logging
import mmap
import os
from typing import Callable, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

PAGE_BYTES = 4096
SLOT_BYTES = 32
TOPOLOGIES = ("star", "chain", "tree")


def source_index(index: int, topology: str) -> int:
    """TP-group index this group index pulls from (-1 for the dedup source)."""
    if index == 0:
        return -1
    if topology == "star":
        return 0
    if topology == "chain":
        return index - 1
    if topology == "tree":
        return (index - 1) // 2
    raise ValueError(f"unknown C2g topology {topology!r} (have {TOPOLOGIES})")


def _progress_map(fd: int):
    return mmap.mmap(
        fd, PAGE_BYTES, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
    )


def _addr(mm) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(mm))


def _ipc_export(t: torch.Tensor) -> dict:
    from decoupled_engine.srt.mem_cache.dedup_ipc_fanout import export_tensor

    return export_tensor(t)


class _IpcOpener:
    def __init__(self, device_index: int):
        self.device_index = device_index
        self._opened: Dict[bytes, int] = {}

    def __call__(self, spec: dict) -> int:
        raw = spec["handle"]
        base = self._opened.get(raw)
        if base is None:
            from decoupled_engine.srt.distributed.device_communicators.cuda_wrapper import (
                CudaRTLibrary,
                cudaIpcMemHandle_t,
            )

            handle = cudaIpcMemHandle_t()
            ctypes.memmove(ctypes.addressof(handle), raw, len(raw))
            with torch.cuda.device(self.device_index):
                ptr = CudaRTLibrary().cudaIpcOpenMemHandle(handle)
            base = int(ptr.value if hasattr(ptr, "value") else ptr)
            self._opened[raw] = base
        return base + int(spec["offset"])


# cudaPointerGetAttributes writes sizeof(cudaPointerAttributes) bytes, which
# grew across CUDA versions: 24 B in 12.x, 88 B in 13.0 (``long reserved[8]``
# after hostPointer). A 24-byte ctypes struct let the CUDA 13 runtime write 64
# bytes past it into the Python heap (on the first C2g follower load: SIGSEGV
# in the next allocation). Use a buffer far larger than
# any known layout and read only the leading, stable fields.
PTR_ATTR_BUF_BYTES = 1024
_PTR_ATTR_DEVICE_OFFSET = 4  # enum cudaMemoryType type; int device; ...


def pointer_device(ptr: int, *, lib=None) -> int:
    """CUDA device ordinal (in this process) that owns ``ptr``."""
    if lib is None:
        from decoupled_engine.srt.mem_cache.ce_2d_load import _cudart

        lib = _cudart()
    fn = lib.cudaPointerGetAttributes
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    buf = (ctypes.c_char * PTR_ATTR_BUF_BYTES)()
    rc = fn(ctypes.addressof(buf), ctypes.c_void_p(ptr))
    if rc:
        raise RuntimeError(f"cudaPointerGetAttributes failed rc={rc}")
    return int(ctypes.c_int.from_buffer(buf, _PTR_ATTR_DEVICE_OFFSET).value)


class CePeer:
    def __init__(
        self,
        *,
        group,
        src_global_rank: int,
        is_src: bool,
        local_layers: Dict[Tuple[str, int], torch.Tensor],
        exporter: Callable[[torch.Tensor], dict] = _ipc_export,
        opener: Optional[Callable[[dict], int]] = None,
        device_index: Optional[int] = None,
        topology: str = "star",
    ):
        self.is_src = is_src
        self.group = group
        self.topology = topology
        self.ptrs: Dict[Tuple[str, int], int] = {}
        self.src_device: Optional[int] = None
        self._mm = None
        self._fd = None
        ranks = torch.distributed.get_process_group_ranks(group)
        world = len(ranks)
        index = ranks.index(torch.distributed.get_rank())
        if is_src != (index == 0) or ranks[0] != src_global_rank:
            raise RuntimeError(
                "C2g: the dedup source must be the first rank of the group"
            )
        if world * SLOT_BYTES > PAGE_BYTES:
            raise RuntimeError(f"C2g: {world} ranks do not fit one progress page")
        self.index = index
        self.source = source_index(index, topology)
        feeds = any(source_index(i, topology) == index for i in range(world))
        live = {k: t for k, t in local_layers.items() if t.numel() > 0}
        # Every rank that feeds others exports its device layers.
        mine = dict(
            layers={k: exporter(t) for k, t in live.items()} if feeds else None,
            sizes={k: t.numel() * t.element_size() for k, t in live.items()},
        )
        if is_src:
            self._fd = os.memfd_create("sglang-c2g-progress", os.MFD_CLOEXEC)
            os.ftruncate(self._fd, PAGE_BYTES)
            self._mm = _progress_map(self._fd)
            mine.update(pid=os.getpid(), fd=self._fd)
        everyone = [None] * world
        torch.distributed.all_gather_object(everyone, mine, group=group)
        if not is_src:
            page = everyone[0]
            fd = os.open(
                f"/proc/{page['pid']}/fd/{page['fd']}", os.O_RDWR | os.O_CLOEXEC
            )
            try:
                self._mm = _progress_map(fd)
            finally:
                os.close(fd)
            if opener is None:
                opener = _IpcOpener(
                    device_index
                    if device_index is not None
                    else torch.cuda.current_device()
                )
            src = everyone[self.source]
            missing = set(live) ^ set(src["layers"])
            if missing:
                raise RuntimeError(
                    f"C2g: source and follower layer sets differ: {sorted(missing)[:4]}"
                )
            for key, spec in src["layers"].items():
                if src["sizes"][key] != live[key].numel() * live[key].element_size():
                    raise RuntimeError(
                        f"C2g: layer {key} size differs between source and follower"
                    )
                self.ptrs[key] = opener(spec)
        torch.distributed.barrier(group=group)
        self.addr = _addr(self._mm)
        # Own slot (published when this rank feeds others) and source slot.
        self.pub_addr = self.addr + SLOT_BYTES * index if feeds else 0
        self.gate_addr = self.addr + SLOT_BYTES * self.source if self.source >= 0 else 0
        logger.info(
            "C2g: rank index %d topology=%s source=%s feeds=%s mapped %d source layer buffers",
            index,
            topology,
            self.source if self.source >= 0 else "host",
            feeds,
            len(self.ptrs),
        )

    def resolve_src_device(self) -> int:
        """The source device's ordinal in this process (for batch copy hints)."""
        if self.src_device is None and self.ptrs:
            self.src_device = pointer_device(next(iter(self.ptrs.values())))
        return self.src_device if self.src_device is not None else 0

    def set_error(self) -> None:
        """Mark this rank's slot failed (ranks pulling from it fail too)."""
        if self._mm is not None:
            words = (ctypes.c_uint64 * 3).from_buffer(self._mm, SLOT_BYTES * self.index)
            words[2] = 1
