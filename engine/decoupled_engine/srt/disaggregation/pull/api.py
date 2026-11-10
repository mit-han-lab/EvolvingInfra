"""Interface between the scheduler-side symmetric mode (disaggregation/symmetric/) and the
pull data plane (disaggregation/pull/). Both sides code against these types only.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence


class TransferState(enum.IntEnum):
    WAITING = 0  # import created, no chunk published yet
    TRANSFERRING = 1  # at least one chunk READ in flight or done, FINAL not yet applied
    READY = 2  # all chunks + aux read and verified; request may be prebuilt
    FAILED = 3  # unrecoverable (owner gone, generation mismatch, timeout)
    ABORTED = 4  # reader or owner aborted


@dataclass
class PoolPages:
    """Page indices of one KV pool (e.g. 'full', 'swa', 'c4', 'c128', 'state') for a
    token range, with the owner's per-page generation for eviction detection."""

    pool: str
    page_indices: List[int]
    generations: List[int]


@dataclass
class ChunkReady:
    transfer_id: int
    chunk_seq: int
    token_start: int
    token_end: int
    pools: List[PoolPages]
    is_last: bool = False
    tier: str = "gpu"  # "gpu" | "host": which owner tier the pages live in


@dataclass
class FinalMeta:
    """OWNER-side completion record (wire message FINAL).

    ``aux_index`` and ``state_indices`` are the OWNER's slots: the aux slot in the
    owner's disagg metadata buffers holding the first token, and the owner's
    per-state-type item slots (keys match ``ImportMark.local_state_indices``).
    The reader READs them into the local slots the scheduler put on the
    ImportMark; ``ReaderPlane.final_meta`` returns this owner-side record as
    received, never the reader's own indices.
    """

    transfer_id: int
    total_tokens: int
    aux_index: int  # slot in the OWNER's disagg metadata buffers
    state_indices: Dict[str, List[int]] = field(default_factory=dict)  # OWNER's slots
    # Base "host:port" of the owner's PeerKVServicer (rank 0; rank i serves at
    # port + i). A DSv4 window-residency reader receives SWA only for the
    # prompt's trailing window and backfills the SWA below it host->host from
    # here once the owner publishes the prompt. None: the owner runs no
    # servicer, or predates the field (the wire key is optional).
    owner_peer_kv_addr: Optional[str] = None
    # Opaque owner cache generation captured at export completion. A host
    # return echoes it so a flush, detach or restart invalidates old hints.
    # None preserves legacy owners and imports without a return directive.
    owner_cache_epoch: Optional[str] = None


def validate_attempt_id(attempt_id: Optional[int]) -> None:
    if attempt_id is not None and (
        isinstance(attempt_id, bool)
        or not isinstance(attempt_id, int)
        or not 1 <= attempt_id < (1 << 64)
    ):
        raise ValueError("attempt_id must be a nonzero uint64 integer")


def _host_return_endpoint(value: str) -> tuple[str, int]:
    """Validate a rank-0 peer servicer address, without accepting URL syntax."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("host return servicer address must be host:port")
    if value.startswith("["):
        raise ValueError("host return servicer addresses do not support IPv6")
    match = re.fullmatch(r"([^:]+):([0-9]+)", value)
    if match is None:
        raise ValueError("host return servicer address must be host:port")
    host, port_text = match[1].lower().removesuffix("."), match[2]
    if len(host) > 253 or not all(
        re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
        for label in host.split(".")
    ):
        raise ValueError("host return servicer address has invalid host")
    if len(port_text) > 5 or not 1 <= int(port_text) <= 65535:
        raise ValueError("host return servicer port must be in 1..65535")
    return host, int(port_text)


@dataclass(frozen=True)
class HostReturn:
    """Router-selected host transfer after the existing decode backup ACK.

    Both endpoints name rank 0. This is a transfer directive, not an engine
    placement policy; omitting it preserves the ordinary import lifecycle.
    """

    destination_servicer_addr: str
    source_servicer_addr: str
    timeout_ms: int
    max_tokens: int

    def __post_init__(self) -> None:
        destination = _host_return_endpoint(self.destination_servicer_addr)
        source = _host_return_endpoint(self.source_servicer_addr)
        if source == destination:
            raise ValueError("host return source and destination must differ")
        for name, upper in (("timeout_ms", 60000), ("max_tokens", 1048576)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"host return {name} must be an integer in 1..{upper}")


@dataclass
class ExportCandidate:
    """Late D stage 2 (``choose``): one candidate reader of a P-leg. The
    router sends this candidate's D-leg with ``kv_import.attempt_id`` =
    ``attempt_id``; the export binds to the candidate whose BEGIN comes
    first (only the committed winner can begin)."""

    cand_idx: int
    reader_worker: str
    reader_addr: str  # zmq endpoint of the reader's per-rank channel
    attempt_id: int

    def __post_init__(self) -> None:
        if self.attempt_id is None:
            raise ValueError("attempt_id must be a nonzero uint64 integer")
        validate_attempt_id(self.attempt_id)


@dataclass
class ExportMark:
    """Router-set mark on the P-leg request (request field `kv_export`).

    One of two forms: the scalar reader fields (one reader), or, in late D
    stage 2, ``candidates`` without them. A candidate export has no reader
    until its first BEGIN binds one (OwnerPlane): ``reader_worker``,
    ``reader_addr`` and ``attempt_id`` are then set to the winner's."""

    transfer_id: int
    reader_worker: Optional[str] = None
    reader_addr: Optional[str] = None  # zmq endpoint of the reader's per-rank channel
    # Shared by both router-set marks and fresh for each retry, including a
    # reused transfer_id. None preserves legacy owner allocation on both legs.
    attempt_id: Optional[int] = None
    candidates: Optional[List[ExportCandidate]] = None

    def __post_init__(self) -> None:
        validate_attempt_id(self.attempt_id)

    @property
    def is_candidate_export(self) -> bool:
        return bool(self.candidates)


@dataclass
class ImportMark:
    """Router-set mark on the D-leg request (request field `kv_import`).

    ``local_aux_index`` / ``local_state_indices`` are NOT router-set: the
    scheduler (symmetric/import_queue.py) fills them at request arrival when it
    pre-allocates the reader's aux slot and per-state-type slots, before
    ``ReaderPlane.begin_import`` (Import allocation happens
    at arrival). The reader READs FINAL's owner slots into these. Keys of
    ``local_state_indices`` match ``FinalMeta.state_indices``.
    """

    transfer_id: int
    owner_worker: str
    owner_addr: str  # zmq endpoint of the owner's per-rank channel
    expected_tokens: Optional[int] = None
    # Router hint for what the reader does with the imported prompt KV once
    # the request finishes (symmetric P/D routing keeps every engine able to take
    # either leg, so this is decided per request, not per engine):
    #   "retain"   - insert into the radix tree and follow the engine's hicache
    #                write policy (host backup); default, the M2 free-role case
    #   "gpu_only" - insert into the radix tree, skip the host write-through
    #                (what upstream PD decode does); for D-role engines that
    #                never take a P leg
    #   "drop"     - do not insert; free the pages at finish
    cache_policy: str = "retain"
    local_aux_index: Optional[int] = None
    local_state_indices: Dict[str, List[int]] = field(default_factory=dict)
    attempt_id: Optional[int] = None
    # Scheduler-owned, overwritten after preallocation/prefix locking; not a router hint.
    local_reuse_prefix_tokens: int = 0
    host_return: Optional[HostReturn] = None

    def __post_init__(self) -> None:
        validate_attempt_id(self.attempt_id)
        if self.host_return is not None:
            if not isinstance(self.host_return, HostReturn):
                raise ValueError("host_return must be a HostReturn directive")
            if self.attempt_id is None:
                raise ValueError("host_return requires a nonzero attempt_id")
            if self.cache_policy != "retain":
                raise ValueError("host_return requires cache_policy=retain")
            if type(self.expected_tokens) is not int or self.expected_tokens <= 0:
                raise ValueError("host_return requires positive expected_tokens")


IMPORT_CACHE_POLICIES = ("retain", "gpu_only", "drop")


class OwnerPlane(Protocol):
    """Owner side (runs on the P-leg worker). Implemented in disaggregation/pull/owner.py."""

    def begin_export(self, mark: ExportMark, req_id: str) -> None: ...

    def publish_chunk(self, chunk: ChunkReady) -> None:
        """Called by the scheduler after a prefill chunk's KV is written and its pages are
        covered by the Lease. Sends CHUNK_READY to the reader."""

    def publish_final(self, final: FinalMeta) -> None: ...

    def poll_released(self) -> List[int]:
        """transfer_ids whose reader sent DONE/ABORT or whose Lease expired; the scheduler
        frees the request's pages for these."""

    def abort_export(self, transfer_id: int, reason: str) -> None: ...


class ReaderPlane(Protocol):
    """Reader side (runs on the D-leg worker). Implemented in disaggregation/pull/importer.py."""

    def begin_import(
        self,
        mark: ImportMark,
        req_id: str,
        dst_pages: Callable[[ChunkReady], Dict[str, Sequence[int]]],
    ) -> None:
        """dst_pages(chunk) returns the reader's local page indices per pool for that
        chunk's token range (allocated up front by the scheduler at request arrival).
        ``mark.local_aux_index`` must already be set (and ``mark.local_state_indices``
        for every state type the owner will report in FINAL)."""

    def poll(self, transfer_id: int) -> TransferState: ...

    def final_meta(self, transfer_id: int) -> Optional[FinalMeta]:
        """The OWNER's FinalMeta as received on the wire, once READY; None before.
        The reader's own slots are on the ImportMark the scheduler passed in."""

    def failure_reason(self, transfer_id: int) -> Optional[str]: ...

    def abort_import(self, transfer_id: int, reason: str) -> None: ...

    def abort_pending(self, mark: ImportMark, reason: str) -> None:
        """Cancel before destination allocation, notifying the source owner."""

    def release_pending(self, mark: ImportMark, reason: str) -> None:
        """Late D: end before destination allocation without notifying the
        owner (a released candidate, or the owner already aborted)."""


class ReadNotQuiescentError(RuntimeError):
    """A transport error without proof that submitted memory accesses stopped."""


class ReadEngine(Protocol):
    """Transport abstraction. Mooncake first, NIXL second."""

    def register(self, ptr: int, nbytes: int, kind: str) -> None: ...  # kind: VRAM|DRAM

    def batch_read(
        self,
        session: str,
        local_ptrs: Sequence[int],
        remote_ptrs: Sequence[int],
        lengths: Sequence[int],
    ) -> int:
        """Blocking one-sided READ of remote memory into local memory. Returns 0 on
        success, nonzero on a quiescent failure. If an error or timeout cannot
        establish that all accesses stopped, raise ReadNotQuiescentError;
        neither source nor destination pages may then be recycled."""
