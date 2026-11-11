"""PeerKVServicer — owner-side lease servicer for cross-engine host-tier KV.

Implements the owner half of the kv-p2p wire protocol v1 (LOOKUP / RELEASE /
PING over ZMQ REP, msgspec msgpack dicts), plus LOOKUP_SWA_RANGE (capability
``swa_range_lookup_v1``), which leases only the SWA-anchored pools of a page
range for the host->host SWA backfill of DSv4 window residency. A peer engine's storage backend
LOOKUPs page hashes; the servicer confirms the contiguous host-backed prefix,
pins those host pages in the owning allocators, leases them, and returns
per-pool host-buffer (ptr, nbytes) segments for one-sided RDMA READ via the
owner's mooncake TransferEngine registration.

Owner-side independence: the servicer runs on its OWN daemon thread and
answers LOOKUPs from a shadow servable-index fed by an in-process delta queue
pushed at the host-state mutation points on the scheduler thread. LOOKUP and
deferred host-return notifications do not wait for a scheduler handshake.
Incoming return commands are queued back for TP-consistent application at the
existing HiCache scheduler agreement point.

Rank consensus: local servicer state must never directly determine
a structural tree decision — TP ranks can have different remote traffic and
its own wall clock, so any gate on lease state diverges the ranks. Eviction is
therefore lease-blind. Data safety comes from the host pools' PageSlotAllocator:
a lease pins pages BY SLOT + GENERATION on the allocator-owning anchor (KV
logical pool for kv/c4/c4_indexer/c128, SWA pool for swa/*_state); a pinned
page that the tree frees is withheld from the free list until the lease is
released (or expires + cools down), and the allocator keeps ``available_size``
rank-uniform through a fixed per-rank reserve. Pins are accepted only while the
page is still allocated with the generation recorded at store time, so a stale
shadow entry (page evicted and reused before the LOOKUP) is simply refused.

Shadow-index keys wrap the engine's chained page hashes in the RadixKey
namespace (cache_salt and extra_key). Publications, removals and reader
lookups must all use peer_page_keys; unsalted, unqualified keys stay raw.
This isolates both page ownership and SWA coverage across namespaces.

Retraction rule: leases are inserted under the mutex, then
the delta queue is drained once more; any eviction delta covering a just-leased
page retracts that page and everything after it from the reply before it is
sent (precision only — the pinned bytes would have stayed intact regardless).
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
import uuid
import zlib
from array import array
from collections import OrderedDict, deque
from queue import Empty, Full, Queue
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import msgspec
import torch
import zmq

from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.mem_cache.peer_kv_demote import (
    LAG_EDGES_S,
    NOTICE_MAX_AGE_S,
    NOTICE_MAX_KEYS,
    DemoteOutboxEntry,
    lag_bucket,
)
from decoupled_engine.srt.mem_cache.pool_host.common import host_buffer_registration_ranges

logger = logging.getLogger(__name__)

_DELTA_STORE = 0
_DELTA_REMOVE = 1
_DELTA_REMOVE_SWA = 2
_DELTA_CLEAR = 3
# SWA-only pages of a request's spilled prefix (window residency, before its
# tree insert): kv_indices / kv_gen stay None, so only LOOKUP_SWA_RANGE serves
# them; a full STORE of the same hash replaces them.
_DELTA_STORE_SWA = 4
# SGLANG_PEER_KV_PUSH_FAST: a whole node's pages as one index range (the
# scheduler thread computes generations with one vectorized call); the
# servicer thread expands it to the per-page payload of the kind above and
# applies it unchanged (_expand_range_delta).
_DELTA_STORE_RANGE = 5
_DELTA_REMOVE_RANGE = 6
_DELTA_REMOVE_SWA_RANGE = 7

_LEASE_TTL_DEFAULT_MS = 30_000
_LEASE_TTL_MAX_MS = 60_000
_POLL_INTERVAL_MS = 50
_EXACT_LOOKUP_MAX_RETRIES = 3
_CACHE_EVENT_CAPACITY = 4096
# Bounds "peer_transfer_complete" page keys; reserved for the transfer-recency
# change, which shares this event pipe. Nothing in this module produces it yet.
_CACHE_PAGE_KEY_CAPACITY = 1024 * 1024
_CACHE_COMMAND_CAPACITY = 64
_CACHE_COMMAND_TOKEN_CAPACITY = 4 * 1024 * 1024
_CACHE_COMMAND_TYPES = frozenset(("host_return_request", "host_return_done"))

# Capability advertised in the PING reply: the owner answers
# ``lookup_swa_range`` (host->host SWA backfill of DSv4 window residency).
SWA_RANGE_CAPABILITY = "swa_range_lookup_v1"

# Poison-on-release debug pattern (shared with the reader's detector).
POISON_BYTE = 0xDE
POISON_LEN = 64


class PeerLookupTraceRecord(msgspec.Struct, gc=False):
    op: str  # lookup | lookup_swa_range | release | expire
    ts: float
    dur_us: float
    num_hashes: int
    num_pages_confirmed: int
    num_pages_retracted: int
    lease_id: str
    leased_bytes_total: int
    refused_reason: Optional[str] = None  # null | budget | no_pages | pin | reserve
    num_pages_partial: int = 0  # shadow-confirmed pages not pinned (stale/reserve)
    swa_pages_pinned: int = 0
    # Opaque request context plus counts already observed by this LOOKUP.
    req_id: Optional[str] = None
    prefetch_attempt_id: Optional[str] = None
    endpoint_round: Optional[int] = None
    initial_query_pages: Optional[int] = None
    exact_endpoint: bool = False
    full_shadow_pages: Optional[int] = None
    full_pinned_pages: Optional[int] = None
    # op=release: the reader read every leased page (RELEASE complete=true).
    complete: Optional[bool] = None


class PeerLookupTraceRecordV4(PeerLookupTraceRecord, gc=False):
    """NOTICE or COUNTERS on: a LOOKUP record also logs the
    owner's confirmed pages; num_pages_confirmed is the grant (0 on a
    refusal). A new field in a new record type, so old readers stay valid
    and the record is unchanged with the flags off."""

    owner_confirmed: Optional[int] = None


_trace_encoder = msgspec.json.Encoder()
_msg_decoder = msgspec.msgpack.Decoder()
_msg_encoder = msgspec.msgpack.Encoder()


class _PageEntry:
    """Servable page: anchor host indices (values, not views into live state)
    plus, per anchor allocator page it spans, the generation at store time.
    An SWA-only entry (``kv_indices is None``) serves LOOKUP_SWA_RANGE only."""

    __slots__ = ("kv_indices", "swa_indices", "kv_gen", "swa_gen")

    def __init__(
        self,
        kv_indices: Any,
        swa_indices: Optional[Any],
        kv_gen: List[int],
        swa_gen: Optional[List[int]],
    ) -> None:
        self.kv_indices = kv_indices
        self.swa_indices = swa_indices
        self.kv_gen = kv_gen
        self.swa_gen = swa_gen


class _Lease:
    __slots__ = (
        "hashes",
        "expiry_mono",
        "nbytes",
        "kv_idx",
        "swa_idx",
        "trace_metadata",
        "lookup_tick",
    )

    def __init__(
        self,
        hashes: Tuple[str, ...],
        expiry_mono: float,
        nbytes: int,
        kv_idx: List[Any],
        swa_idx: List[Tuple[int, Any]],
        trace_metadata: Optional[dict] = None,
        lookup_tick: Optional[float] = None,
    ):
        self.hashes = hashes
        # --peer-kv-demote-source: the tree's access tick at LOOKUP start.
        self.lookup_tick = lookup_tick
        self.expiry_mono = expiry_mono
        self.nbytes = nbytes
        # Anchor slot tensors held pinned by this lease: kv_idx[i] for lease
        # page i; swa_idx = [(page index, slots), ...] for the trailing window.
        self.kv_idx = kv_idx
        self.swa_idx = swa_idx
        self.trace_metadata = dict(trace_metadata or {})


def _request_trace_metadata(msg: dict) -> dict:
    """Bounded scalar attribution, never a condition for serving a request."""
    raw = msg.get("request_metadata")
    if not isinstance(raw, dict):
        return {}
    result = {}
    for name, limit in (("req_id", 256), ("prefetch_attempt_id", 64)):
        value = raw.get(name)
        if isinstance(value, str) and 0 < len(value) <= limit:
            result[name] = value
    for name in ("endpoint_round", "initial_query_pages"):
        value = raw.get(name)
        if type(value) is int and 0 <= value <= 2**63 - 1:
            result[name] = value
    return result


def _page_generations(
    slots: Any, pages: Sequence[Optional[Any]]
) -> List[Optional[List[int]]]:
    """``slots.page_generation(p)`` for each page, in one allocator read.

    The caller holds the pool lock. A per-page read costs about ten small
    tensor ops, which dominated host eviction on the scheduler thread for
    nodes with many pages. Every page is whole allocator pages (granules), so
    the generations of the concatenation split back at page boundaries and
    the result equals the per-page reads. ``None`` pages stay ``None``.
    SGLANG_PEER_KV_PUSH_FAST only."""
    out: List[Optional[List[int]]] = [None] * len(pages)
    live = [i for i, p in enumerate(pages) if p is not None]
    granule = getattr(slots, "granule", None)
    counts = []
    if len(live) > 1 and granule:
        for i in live:
            n = pages[i].numel()
            if n % granule:
                break  # misaligned: the per-page reads below raise as before
            counts.append(n // granule)
    if len(counts) != len(live) or len(live) < 2:
        for i in live:
            out[i] = slots.page_generation(pages[i])
        return out
    gens = slots.page_generation(torch.cat([pages[i] for i in live]))
    pos = 0
    for i, k in zip(live, counts):
        out[i] = gens[pos : pos + k]
        pos += k
    return out


def _pool_backing_buffers(pool: Any) -> List[Any]:
    """Physical tensors backing a host pool; [] for logical anchors.

    Mirrors mooncake_store's probe: hybrid pools expose
    get_hybrid_pool_buffer(); plain pools expose kv_buffer; DeepSeek V4's
    "kv" anchor entry owns indices but no bytes (kv_buffer is None) — the
    bytes live in the sidecar sub-pools.
    """
    try:
        buffers = list(pool.get_hybrid_pool_buffer())
    except AttributeError:
        buffers = [pool.kv_buffer]
    return [buf for buf in buffers if buf is not None]


def build_owner_transfer_engine(
    host_pool_group: Any,
    gpu_id: int,
    ib_device: Optional[str] = None,
    registered_engine: Any = None,
) -> Any:
    """Create/reuse the process TransferEngine and register every host pool
    buffer remote-accessible. Returns the wrapper (session_id = te_endpoint).

    Mirrors mooncake_store's registration: hybrid pools expose
    get_hybrid_pool_buffer(); plain pools expose kv_buffer.

    ``ib_device`` restricts mooncake's NIC set (comma-separated, same format
    as --mooncake-ib-device); None keeps auto-topology, which on multi-fabric
    hosts can enroll off-fabric HCAs and stall cross-node reads in retry loops.
    """
    from decoupled_engine.srt.distributed.device_communicators.mooncake_transfer_engine import (
        init_mooncake_transfer_engine,
    )
    from decoupled_engine.srt.utils.network import get_local_ip_auto

    # Bound the per-slice transfer timeout explicitly; must stay well below
    # the lease TTL so a stuck reader cannot outlive its lease.
    os.environ.setdefault("MC_SLICE_TIMEOUT", "10000")

    te = registered_engine or init_mooncake_transfer_engine(
        hostname=get_local_ip_auto(), gpu_id=gpu_id, ib_device=ib_device
    )
    # The process engine may already hold these spans: the peer_kv reader
    # shares it and registers the same host pool spans first
    # (PeerKVBackend._shared_transfer_engine). Mooncake rejects an overlapping
    # second registration, so go through the engine's shared record.
    register_once = (
        getattr(te, "register_memory_once", None) if registered_engine is None else None
    )
    registered = 0
    for entry in host_pool_group.entries:
        for buf in _pool_backing_buffers(entry.host_pool):
            for ptr, nbytes in host_buffer_registration_ranges(buf):
                if registered_engine is not None:
                    # Use the reader's exact spans to avoid re-registering them.
                    te.register(ptr, nbytes)
                else:
                    if register_once is not None:
                        result = register_once(ptr, nbytes)
                    else:
                        result = te.engine.register_memory(ptr, nbytes)
                    if result != 0:
                        raise RuntimeError(
                            f"peer_kv owner registration failed: bytes={nbytes}, rc={result}"
                        )
                registered += 1
    logger.info(
        "PeerKVServicer: registered %d host ranges with TransferEngine %s "
        "(devices %r)",
        registered,
        te.get_session_id(),
        getattr(te, "ib_device", None),
    )
    return te


class _GpuLease:
    __slots__ = ("transfer_id", "expiry_mono", "pages", "locks", "cool_until")

    def __init__(self, transfer_id: int, expiry_mono: float) -> None:
        self.transfer_id = transfer_id
        self.expiry_mono = expiry_mono
        # {pool: {page_index: generation at pin time}}
        self.pages: Dict[str, Dict[int, int]] = {}
        # (node_ref, dec_params) per inc_lock_ref taken, released in reverse.
        self.locks: List[Tuple[Any, Any]] = []
        # None while live; set at expiry to expiry + cooling.
        self.cool_until: Optional[float] = None


class GpuLeaseTable:
    """Owner Lease over GPU radix pages.

    Additive counterpart of the host try_pin path above: a transfer pins the
    radix nodes covering its published pages with ``inc_lock_ref`` (the tree's
    own eviction refuses locked nodes; see peer_gpu_export_cache.py which holds
    the same lock during D2H staging) and records (slot, generation) per pool
    so a page that was evicted and reused between publication and FINAL is
    detectable. Host pages keep using the servicer's slot pins; one transfer
    may hold both (mixed lease).

    Threading: ``pin``, ``release`` and ``drain_expired`` mutate the radix tree
    and must run on the scheduler thread (OwnerPlane calls them from
    publish_chunk / poll_released). ``expire`` only flags and may run on the
    servicer thread (_tick). Expiry keeps the locks for one cooling window,
    like the host pins: a READ that started before the TTL ran out may still
    be in flight.

    ``locker`` is anything with ``inc_lock_ref(node_ref) -> IncLockRefResult``
    and ``dec_lock_ref(node_ref, params)``: RadixCache (node objects) or
    UnifiedTreeCore (node ids). ``generation_of(pool, page_indices)`` returns the
    owner's current generation per page; without it, generations are recorded
    as published and cannot be re-verified.
    """

    def __init__(
        self,
        *,
        locker: Any = None,
        ttl_ms: Optional[int] = _LEASE_TTL_DEFAULT_MS,
        expiry_cooling_s: float = 12.0,
        generation_of: Optional[Any] = None,
    ) -> None:
        self._locker = locker
        # Request-held exports have no unilateral expiry: a reader may still
        # be waiting for destination capacity. Only its completion permits free.
        self._ttl_s = (
            float("inf")
            if ttl_ms is None
            else min(int(ttl_ms), _LEASE_TTL_MAX_MS) / 1e3
        )
        self._cooling_s = float(expiry_cooling_s)
        self._generation_of = generation_of
        self._mutex = threading.Lock()
        self._leases: Dict[int, _GpuLease] = {}
        self.stats_refused_generation = 0
        self.stats_expired = 0
        self.stats_released = 0

    def active_count(self) -> int:
        """Leases currently held (including those in the expiry cooling window)."""
        with self._mutex:
            return len(self._leases)

    def wrap_generation_of(self, wrap: Any) -> bool:
        """Replace ``generation_of`` by ``wrap(generation_of)``. Returns False
        (and changes nothing) when there is none. Used once at loop start by
        SGLANG_SYMMETRIC_PUBLISH_ASYNC_READBACK
        (disaggregation/symmetric/publish_readback.py)."""
        if self._generation_of is None:
            return False
        self._generation_of = wrap(self._generation_of)
        return True

    def pin(
        self,
        *,
        transfer_id: int,
        pages: Mapping[str, Tuple[Sequence[int], Sequence[int]]],
        node_refs: Sequence[Any] = (),
        now_mono: Optional[float] = None,
    ) -> Optional[str]:
        """Pin ``pages`` = {pool: (page_indices, generations)} for ``transfer_id``
        and lock ``node_refs``; extends the lease TTL. Chunked publication
        calls this once per chunk. Returns None or a refusal reason; on refusal
        nothing of this call is recorded."""
        now = time.monotonic() if now_mono is None else now_mono
        for pool, (idx, gens) in pages.items():
            if len(idx) != len(gens):
                raise ValueError(
                    f"GpuLeaseTable: pool {pool}: {len(idx)} pages but {len(gens)} generations"
                )
        # Placeholder pages (-1: "no page on this side", see
        # disaggregation/symmetric/export_table.py pool_pages_for_range) are not
        # allocator indices: they are neither leased nor generation-checked.
        pages = {
            pool: tuple(
                zip(*[(int(p), int(g)) for p, g in zip(idx, gens) if int(p) >= 0])
            )
            or ((), ())
            for pool, (idx, gens) in pages.items()
        }
        for pool, (idx, gens) in pages.items():
            if self._generation_of is not None:
                current = self._generation_of(pool, list(idx))
                if [int(g) for g in current] != [int(g) for g in gens]:
                    self.stats_refused_generation += 1
                    return "generation_mismatch"
        locks: List[Tuple[Any, Any]] = []
        if self._locker is not None:
            for ref in node_refs:
                locks.append((ref, self._locker.inc_lock_ref(ref).to_dec_params()))
        with self._mutex:
            lease = self._leases.get(int(transfer_id))
            if lease is None or lease.cool_until is not None:
                lease = _GpuLease(int(transfer_id), now + self._ttl_s)
                self._leases[int(transfer_id)] = lease
            lease.expiry_mono = now + self._ttl_s
            for pool, (idx, gens) in pages.items():
                lease.pages.setdefault(pool, {}).update(
                    {int(p): int(g) for p, g in zip(idx, gens)}
                )
            lease.locks.extend(locks)
        return None

    def check(
        self,
        *,
        transfer_id: int,
        pages: Mapping[str, Tuple[Sequence[int], Sequence[int]]],
    ) -> bool:
        """True iff every (pool, page) is leased with the given generation and,
        when ``generation_of`` is available, still carries it now."""
        with self._mutex:
            lease = self._leases.get(int(transfer_id))
            if lease is None:
                return False
            for pool, (idx, gens) in pages.items():
                rec = lease.pages.get(pool, {})
                if any(rec.get(int(p)) != int(g) for p, g in zip(idx, gens)):
                    return False
        if self._generation_of is not None:
            for pool, (idx, gens) in pages.items():
                current = self._generation_of(pool, list(idx))
                if [int(g) for g in current] != [int(g) for g in gens]:
                    return False
        return True

    def current_generations(self, transfer_id: int) -> Dict[str, Dict[int, int]]:
        """{pool: {page: generation}} as the owner sees it now (for FINAL)."""
        with self._mutex:
            lease = self._leases.get(int(transfer_id))
            recorded = (
                {} if lease is None else {k: dict(v) for k, v in lease.pages.items()}
            )
        if self._generation_of is None:
            return recorded
        out: Dict[str, Dict[int, int]] = {}
        for pool, m in recorded.items():
            idx = list(m.keys())
            out[pool] = {
                int(p): int(g) for p, g in zip(idx, self._generation_of(pool, idx))
            }
        return out

    def is_leased(self, transfer_id: int) -> bool:
        with self._mutex:
            lease = self._leases.get(int(transfer_id))
            return lease is not None and lease.cool_until is None

    def release(self, *, transfer_id: int) -> bool:
        """DONE/ABORT path: unlock immediately (the reader's reads completed or
        it gave up before reading). Scheduler thread."""
        with self._mutex:
            lease = self._leases.pop(int(transfer_id), None)
        if lease is None:
            return False
        self._unlock(lease)
        self.stats_released += 1
        return True

    def expire(self, now_mono: Optional[float] = None) -> List[int]:
        """Flag leases past their TTL (any thread; no tree access)."""
        now = time.monotonic() if now_mono is None else now_mono
        newly: List[int] = []
        with self._mutex:
            for tid, lease in self._leases.items():
                if lease.cool_until is None and lease.expiry_mono <= now:
                    lease.cool_until = now + self._cooling_s
                    newly.append(tid)
        self.stats_expired += len(newly)
        return newly

    def drain_expired(self, now_mono: Optional[float] = None) -> List[int]:
        """Unlock expired leases whose cooling window has passed and return
        their transfer_ids (scheduler thread)."""
        now = time.monotonic() if now_mono is None else now_mono
        self.expire(now)
        with self._mutex:
            done = [
                tid
                for tid, lease in self._leases.items()
                if lease.cool_until is not None and lease.cool_until <= now
            ]
            leases = [self._leases.pop(tid) for tid in done]
        for lease in leases:
            self._unlock(lease)
        return done

    def _unlock(self, lease: _GpuLease) -> None:
        if self._locker is None:
            return
        for ref, params in reversed(lease.locks):
            try:
                self._locker.dec_lock_ref(ref, params)
            except Exception:
                logger.exception("GpuLeaseTable: dec_lock_ref failed (lock leak)")
        lease.locks.clear()

    def stats(self) -> Dict[str, int]:
        with self._mutex:
            live = sum(1 for ls in self._leases.values() if ls.cool_until is None)
            cooling = len(self._leases) - live
            pages = sum(
                len(m) for ls in self._leases.values() for m in ls.pages.values()
            )
        return {
            "gpu_leases": live,
            "gpu_leases_cooling": cooling,
            "gpu_pages_leased": pages,
            "gpu_refused_generation": self.stats_refused_generation,
            "gpu_expired": self.stats_expired,
            "gpu_released": self.stats_released,
        }


class PeerKVServicer:
    """Owner-side lease servicer. One instance per TP-rank scheduler process."""

    _DEMOTE_OUTBOX_MAX = 8192
    _TOUCH_OUTBOX_MAX = 1024
    # F1: request ids of recent LOOKUPs, kept for the lease TTL + 60 s.
    _TOUCH_LOOKUP_RIDS_MAX = 65536
    _TOUCH_LOOKUP_GRACE_S = 60.0
    # SGLANG_PEER_KV_DEMOTE_AFTER_COMMIT: leases read in full whose reader
    # has not yet sent DEMOTE_OK (lease id -> keys, lookup tick, rid, expiry).
    _DEMOTE_PENDING_MAX = 8192

    def __init__(
        self,
        *,
        host_pool_group: Any,
        page_size: int,
        bind_port: int,
        max_leased_bytes: int,
        trace_dir: Optional[str],
        dp_rank: int,
        pp_rank: int,
        attn_tp_rank: int,
        te_wrapper: Any,
        swa_trailing_pages: int = 0,
        expiry_cooling_s: float = 12.0,
        verify_checksums: bool = False,
        gpu_lease_table: Optional[GpuLeaseTable] = None,
        swa_host_complete: bool = False,
        demote_tick_source: Optional[Callable[[], float]] = None,
        touch_outbox: bool = False,
        demote_role: Optional[str] = None,
    ) -> None:
        # GPU-page half of the Lease; None keeps the host-only servicer.
        self.gpu_leases = gpu_lease_table
        # --peer-kv-demote-source, request-broadcast source rank only: a
        # complete RELEASE queues (leased keys, lookup tick) for the scheduler
        # thread (UnifiedRadixCache.collect_peer_kv_demote). The tick source
        # only reads the tree's access counter.
        self._demote_tick = demote_tick_source
        # Owner role check: the launch role (--disaggregation-mode). A decode
        # owner ignores every mark message (RELEASE-time mark, DEMOTE_OK,
        # NOTICE), counted. No flag: no traffic sends marks to a D today.
        self._demote_ignores_marks = demote_role == "decode"
        self.demote_ignored_role = 0
        self._demote_outbox: Optional[deque] = (
            deque(maxlen=self._DEMOTE_OUTBOX_MAX)
            if demote_tick_source is not None
            else None
        )
        # SGLANG_PEER_KV_TOUCH_DEFERRED_PULLS (mem_cache/peer_kv_touch.py),
        # request-broadcast source rank only: TOUCH queues the confirmed keys
        # of a deferred pull's span for the scheduler thread
        # (UnifiedRadixCache.collect_peer_kv_touch). None: TOUCH is an
        # unknown op, as before.
        self._touch_outbox: Optional[deque] = (
            deque(maxlen=self._TOUCH_OUTBOX_MAX) if touch_outbox else None
        )
        self._touch_received = 0
        self._touch_queued = 0
        # A TOUCH refreshes the owner only before the LOOKUP of its own pull
        # (_touch_note_lookup). Servicer thread only: the request ids of
        # recent LOOKUPs -> expiry (monotonic), in insertion order. The lock
        # guards the outbox, which the scheduler thread drains.
        self._touch_lock = threading.Lock()
        self._touch_lookup_rids: OrderedDict[str, float] = OrderedDict()
        self.touch_after_lookup = 0
        self.touch_dropped_by_lookup = 0
        self._touch_lookup_no_rid = 0
        # The end of the access ticks that the last drain reserved for its
        # TOUCHes; a lookup tick is never below it (_lookup_tick).
        self._touch_reserved_end = 0.0
        # Lever-1 hardening (peer_kv_demote.py). Servicer thread only, read
        # by the scheduler thread for logs. COUNTERS adds the pulling request
        # id to each demote outbox entry and the counters to PING.
        self._demote_counters = bool(envs.SGLANG_PEER_KV_DEMOTE_COUNTERS.get())
        self._demote_pending: OrderedDict[str, tuple] = OrderedDict()
        self._demote_pending_ttl_s = float(
            envs.SGLANG_PEER_KV_DEMOTE_PENDING_TTL_S.get()
        )
        self.demote_outbox_drops = 0
        self.demote_touch_outbox_drops = 0
        self.demote_commit_pending = 0
        self.demote_commit_ok = 0
        self.demote_commit_expired = 0
        self.demote_commit_unknown = 0
        # COUNTERS: RELEASE -> DEMOTE_OK lag of the confirmed
        # commits (histogram over LAG_EDGES_S, and the max); in PING and in
        # the resolver's "PeerKV demote:" line.
        self.demote_commit_lag = [0] * (len(LAG_EDGES_S) + 1)
        self.demote_commit_lag_max_ms = 0
        self._init_notice()
        self.trace_dir = trace_dir
        self._host_pool_group = host_pool_group
        self._page_size = page_size
        self._max_leased_bytes = max_leased_bytes
        self._te_endpoint: str = te_wrapper.get_session_id()
        self._swa_trailing_pages = int(swa_trailing_pages)
        # SGLANG_PEER_KV_PUSH_FAST: per-page pushes read generations (and
        # snapshot removals) once per call instead of once per page.
        self._push_fast = bool(envs.SGLANG_PEER_KV_PUSH_FAST.get())
        self._expiry_cooling_s = float(expiry_cooling_s)
        self._verify_checksums = bool(verify_checksums)
        # The owner's host SWA covers every published page (DSv4 uniform
        # lifecycle: window or uniform-full residency). Only such an owner can
        # grant LOOKUP_SWA_RANGE below a node's trailing window; a legacy
        # (non-uniform) owner publishes trailing-window SWA only, so it
        # neither advertises swa_range_lookup_v1 nor serves the op.
        self._swa_host_complete = bool(swa_host_complete)

        # Anchors: the allocator-owning entries whose indices address served
        # bytes. Derived from the group (PoolEntry.indices_from_pool), not from
        # a static table, so any hybrid layout maps itself.
        from decoupled_engine.srt.mem_cache.hicache_storage import PoolName

        anchor_entries = host_pool_group.allocator_entries()
        anchor_names = {str(e.name) for e in anchor_entries}
        self._kv_anchor = host_pool_group.anchor_of(PoolName.KV).host_pool
        self._swa_anchor = (
            host_pool_group.anchor_of(PoolName.SWA).host_pool
            if PoolName.SWA in host_pool_group.entry_map
            else None
        )
        for label, pool in (("kv", self._kv_anchor), ("swa", self._swa_anchor)):
            if pool is None:
                continue
            slots = getattr(pool, "slot_allocator", None)
            if slots is None:
                raise ValueError(
                    "--enable-peer-kv-servicer requires host pools backed by "
                    f"PageSlotAllocator; the {label} anchor {type(pool).__name__} "
                    "has not enabled peer serving."
                )
            if slots.reserve_pages == 0:
                raise ValueError(
                    f"PeerKVServicer: the {label} anchor has reserve_pages=0 — set "
                    f"--peer-kv-servicer-reserve-ratio-{label} > 0 (pins need a "
                    "rank-uniform reserve to park evicted-but-leased pages)."
                )

        # Pool plan: (pool_name, host_pool, anchored_on_swa, has_bytes) for
        # present pools. has_bytes=False marks logical anchors (e.g. the
        # DeepSeek V4 "kv" entry): the page is attested with EMPTY segments
        # on the wire, and its bytes travel via the sidecar pools. Detection
        # matches TE registration exactly, so "servable" can never diverge
        # from "RDMA-readable".
        self._pool_plan: List[Tuple[str, Any, bool, bool]] = []
        for entry in host_pool_group.entries:
            name = str(entry.name)
            anchor = host_pool_group.anchor_of(entry.name)
            has_bytes = len(_pool_backing_buffers(entry.host_pool)) > 0
            if anchor.name == PoolName.KV:
                self._pool_plan.append((name, entry.host_pool, False, has_bytes))
            elif anchor.name == PoolName.SWA:
                self._pool_plan.append((name, entry.host_pool, True, has_bytes))
            else:
                # Pools anchored elsewhere (mamba/draft/...) are not served.
                if has_bytes:
                    logger.info(
                        "PeerKVServicer: host pool %s (anchor %s) is not served",
                        name,
                        anchor.name,
                    )
        logger.info(
            "PeerKVServicer: serving %s; anchors %s; kv reserve %d pages, swa reserve %s; "
            "swa trailing pages %d; expiry cooling %.1fs; verify_checksums=%s",
            [p[0] for p in self._pool_plan],
            sorted(anchor_names),
            self._kv_anchor.slot_allocator.reserve_pages,
            (
                self._swa_anchor.slot_allocator.reserve_pages
                if self._swa_anchor is not None
                else "n/a"
            ),
            self._swa_trailing_pages,
            self._expiry_cooling_s,
            self._verify_checksums,
        )

        self._poison_installed = False
        if envs.SGLANG_PEER_KV_POISON_ON_RELEASE.get():
            self._install_poison_hooks()

        # Shadow servable-index + delta queue (scheduler thread -> servicer).
        self._shadow: Dict[str, _PageEntry] = {}
        self._deltas: Queue[Tuple[int, Any]] = Queue()
        # IO completion/control messages travel back as content identities only.
        # The scheduler must replicate an ordered batch across TP before applying
        # these events; the servicer never mutates a radix tree or its LRU.
        self._cache_events: Queue[dict] = Queue(maxsize=_CACHE_EVENT_CAPACITY)
        self._cache_event_lock = threading.Lock()
        self._cache_commands_pending = 0
        self._cache_command_tokens_pending = 0
        self._cache_page_keys_pending = 0
        self.cache_events_dropped = 0

        # Reuse this servicer's IO loop for one-shot control notifications.
        # Scheduler calls only enqueue; even an unreachable P never blocks D.
        self._outbox: Queue = Queue(maxsize=_CACHE_COMMAND_CAPACITY)
        self._outbox_lock = threading.Lock()
        self._outbox_count = 0
        self._outbox_tokens = 0
        self.control_notifications_failed = 0
        self._wake_read, self._wake_write = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)

        # Lease table (servicer thread; the mutex only guards PING reads and
        # the tests' direct access). Byte budget is advisory backpressure.
        self._lease_mutex = threading.Lock()
        self._leases: Dict[str, _Lease] = {}
        self._leased_bytes_total = 0
        self._pin_leaks = 0
        # F4: legacy LOOKUPs over the byte budget get the prefix that fits.
        self._budget_partial_grant = bool(
            envs.SGLANG_PEER_KV_BUDGET_PARTIAL_GRANT.get()
        )
        self._budget_partial_min_pages = max(
            1, int(envs.SGLANG_PEER_KV_BUDGET_PARTIAL_MIN_PAGES.get())
        )
        self._budget_partial_grants = 0
        self._budget_trimmed_pages = 0
        # F5: LOOKUPs past the reader's reply deadline (+ slack) are refused
        # ("late") at the start and again just before the grant.
        self._lookup_deadline = bool(envs.SGLANG_PEER_KV_LOOKUP_DEADLINE.get())
        self._lookup_deadline_slack_s = max(
            0.0, float(envs.SGLANG_PEER_KV_LOOKUP_DEADLINE_SLACK_S.get())
        )
        self._late_at_start = 0
        self._late_at_grant = 0
        # F6: a legacy LOOKUP checks the budget before it pins, with the
        # smallest KV page bytes of an earlier grant (None until one).
        self._budget_precheck = bool(envs.SGLANG_PEER_KV_BUDGET_PRECHECK.get())
        self._page_nbytes_floor: Optional[int] = None
        self._budget_precheck_refusals = 0
        # The counters of the pull-miss flags (F1, F4, F5, F6, demote
        # COUNTERS) are in PING only; with any of them on, the servicer thread
        # also logs them, at most once per _COUNTERS_LOG_S and only on change.
        self._counters_logged_at = time.monotonic()
        self._counters_last_line: Optional[str] = None

        self._trace_file = None
        if trace_dir is not None:
            os.makedirs(trace_dir, exist_ok=True)
            suffix = f"dp{dp_rank}_pp{pp_rank}_tp{attn_tp_rank}.pid{os.getpid()}"
            self._trace_file = open(
                os.path.join(trace_dir, f"servicer_trace.{suffix}.jsonl"),
                "ab",
                buffering=0,
            )

        self._stop_event = threading.Event()
        self._bind_port = bind_port
        self._startup_done = threading.Event()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._serve_loop, daemon=True, name="peer-kv-servicer"
        )
        self._thread.start()
        # Do not advertise an engine whose peer endpoint failed to bind.
        if not self._startup_done.wait(timeout=30):
            self._stop_event.set()
            self._thread.join(timeout=5)
            raise RuntimeError(f"PeerKVServicer startup timed out on port {bind_port}")
        if self._startup_error is not None:
            self._thread.join(timeout=5)
            if self._trace_file is not None:
                self._trace_file.close()
            raise RuntimeError(
                f"PeerKVServicer failed to listen on port {bind_port}"
            ) from self._startup_error

    def _install_poison_hooks(self) -> None:
        """Debug detector (SGLANG_PEER_KV_POISON_ON_RELEASE): overwrite the
        first 64 B of every byte-bearing segment of a page the moment it enters
        an anchor's free list. Pinned/withheld pages are never touched, so a
        reader that pulls the pattern proves a page was reused under its read."""
        for anchor, on_swa in ((self._kv_anchor, False), (self._swa_anchor, True)):
            if anchor is None:
                continue
            byte_pools = [
                pool
                for _, pool, swa, has_bytes in self._pool_plan
                if swa == on_swa and has_bytes
            ]
            if not byte_pools:
                continue

            def hook(slots: torch.Tensor, pools=tuple(byte_pools)) -> None:
                for pool in pools:
                    ptrs, lens = pool.get_page_buffer_meta(slots)
                    for p, n in zip(ptrs, lens):
                        ctypes.memset(int(p), POISON_BYTE, min(POISON_LEN, int(n)))

            with anchor.lock:
                anchor.slot_allocator.poison_hook = hook
        self._poison_installed = True
        logger.warning(
            "PeerKVServicer: poison-on-release ENABLED (debug; corrupts freed host pages)"
        )

    def _remove_poison_hooks(self) -> None:
        for anchor in (self._kv_anchor, self._swa_anchor):
            if anchor is None:
                continue
            with anchor.lock:
                anchor.slot_allocator.poison_hook = None
        self._poison_installed = False
        logger.warning("PeerKVServicer: poison-on-release disabled")

    # ------------------------------------------------------------------
    # Scheduler-thread API (push sites). NO reads of servicer state.
    # ------------------------------------------------------------------

    def enqueue_cache_event(self, event: dict) -> bool:
        """Queue one scheduler event without blocking IO; return whether accepted.

        Input is a content-keyed completion or router control command. The
        caller transfers ownership of the dictionary and must not mutate it.
        Commands and completion page keys have separate aggregate bounds because
        a single event may carry a long prompt or many content identities.
        """
        command = event.get("type") in _CACHE_COMMAND_TYPES
        tokens = event.get("token_ids", ()) if command else ()
        if not isinstance(tokens, (list, tuple)):
            return False
        token_count = len(tokens)
        page_keys = (
            event.get("page_keys", ())
            if event.get("type") == "peer_transfer_complete"
            else ()
        )
        if not isinstance(page_keys, (list, tuple)):
            return False
        page_count = len(page_keys)
        with self._cache_event_lock:
            if command and (
                self._cache_commands_pending >= _CACHE_COMMAND_CAPACITY
                or self._cache_command_tokens_pending + token_count
                > _CACHE_COMMAND_TOKEN_CAPACITY
            ):
                return False
            if self._cache_page_keys_pending + page_count > _CACHE_PAGE_KEY_CAPACITY:
                self.cache_events_dropped += 1
                return False
            try:
                self._cache_events.put_nowait(event)
            except Full:
                self.cache_events_dropped += 1
                return False
            self._cache_commands_pending += int(command)
            self._cache_command_tokens_pending += token_count
            self._cache_page_keys_pending += page_count
        return True

    def has_cache_events(self) -> bool:
        """Return queue readiness only; the scheduler must TP-sync this signal."""
        return not self._cache_events.empty()

    def drain_cache_events(self, limit: int = 128) -> list[dict]:
        """Return at most limit queued events for a subsequent TP broadcast.

        This method never applies events locally: rank-local arrival order must
        not determine the shared cache tree or its eviction order.
        """
        events = []
        with self._cache_event_lock:
            for _ in range(max(0, limit)):
                try:
                    event = self._cache_events.get_nowait()
                except Empty:
                    break
                self._cache_commands_pending -= int(
                    event.get("type") in _CACHE_COMMAND_TYPES
                )
                if event.get("type") in _CACHE_COMMAND_TYPES:
                    self._cache_command_tokens_pending -= len(
                        event.get("token_ids", ())
                    )
                if event.get("type") == "peer_transfer_complete":
                    self._cache_page_keys_pending -= len(event.get("page_keys", ()))
                events.append(event)
        return events

    def notify_peer(self, endpoint: str, message: dict) -> bool:
        """Enqueue a one-shot CPU control message; return False on backpressure.

        Inputs are a validated rank-0 host:port and an immutable message. The
        existing servicer thread owns sockets/serialization. The budget covers
        queued AND in-flight messages; no network wait runs in a GPU step.
        """
        tokens = len(message.get("token_ids", ()))
        with self._outbox_lock:
            if (
                self._stop_event.is_set()
                or self._outbox_count >= _CACHE_COMMAND_CAPACITY
                or self._outbox_tokens + tokens > _CACHE_COMMAND_TOKEN_CAPACITY
            ):
                self.control_notifications_failed += 1
                return False
            self._outbox.put_nowait((endpoint, message, tokens))
            self._outbox_count += 1
            self._outbox_tokens += tokens
        # A full pipe already wakes the IO loop. This write is nonblocking and
        # avoids adding the normal 50 ms lease-maintenance poll to the return.
        try:
            os.write(self._wake_write, b"\x01")
        except (BlockingIOError, OSError):
            pass
        return True

    def _retire_notification(self, tokens: int, success: bool) -> None:
        """Release one notification's bounded budget after ACK, timeout or stop."""
        with self._outbox_lock:
            self._outbox_count -= 1
            self._outbox_tokens -= tokens
            self.control_notifications_failed += int(not success)

    def _start_notifications(self, ctx, poller, pending: dict) -> None:
        """Start at most eight messages before servicing incoming peer requests."""
        for _ in range(8):
            try:
                endpoint, message, tokens = self._outbox.get_nowait()
            except Empty:
                return
            sock = ctx.socket(zmq.REQ)
            try:
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.SNDHWM, 1)
                if endpoint.startswith("["):
                    sock.setsockopt(zmq.IPV6, 1)
                sock.connect(f"tcp://{endpoint}")
                wire_message = message
                if isinstance(message.get("token_ids"), array):
                    # RadixKey owns array('q'). Only the IO thread materializes
                    # its wire list; neither mutate it nor copy a long prompt
                    # in the GPU scheduler's completion path.
                    wire_message = dict(message)
                    wire_message["token_ids"] = message["token_ids"].tolist()
                sock.send(_msg_encoder.encode(wire_message), flags=zmq.NOBLOCK)
                poller.register(sock, zmq.POLLIN)
                pending[sock] = (time.monotonic() + 2.0, tokens)
            except (zmq.ZMQError, TypeError, ValueError):
                sock.close(linger=0)
                self._retire_notification(tokens, False)

    def push_store(
        self,
        hashes: Sequence[str],
        kv_indices_pages: Sequence[Any],
        swa_indices_pages: Sequence[Optional[Any]],
    ) -> None:
        """A node's pages became host-backed (D2H ack or prefetch commit).

        Records each page's anchor allocator generation so a later LOOKUP can
        refuse pages that were evicted and reused in the meantime."""
        kv_pages = list(kv_indices_pages)
        swa_pages = list(swa_indices_pages)
        kv_slots = self._kv_anchor.slot_allocator
        fast = getattr(self, "_push_fast", False)
        with self._kv_anchor.lock:
            if fast:
                kv_gens = _page_generations(kv_slots, kv_pages)
            else:
                kv_gens = [kv_slots.page_generation(p) for p in kv_pages]
        swa_gens: List[Optional[List[int]]] = [None] * len(swa_pages)
        if self._swa_anchor is not None and any(p is not None for p in swa_pages):
            swa_slots = self._swa_anchor.slot_allocator
            with self._swa_anchor.lock:
                if fast:
                    swa_gens = _page_generations(swa_slots, swa_pages)
                else:
                    for i, p in enumerate(swa_pages):
                        if p is not None:
                            swa_gens[i] = swa_slots.page_generation(p)
        self._deltas.put(
            (_DELTA_STORE, (list(hashes), kv_pages, swa_pages, kv_gens, swa_gens))
        )

    def push_swa_store(
        self, hashes: Sequence[str], swa_indices_pages: Sequence[Any]
    ) -> None:
        """A request's spilled SWA pages became host-backed (spill ack) before
        its tree insert: publish them SWA-only, for LOOKUP_SWA_RANGE. A page
        already published with its FULL backing keeps that entry."""
        if self._swa_anchor is None or not hashes:
            return
        if len(hashes) != len(swa_indices_pages):
            raise ValueError("SWA store hashes and physical pages must align")
        swa_pages = list(swa_indices_pages)
        swa_slots = self._swa_anchor.slot_allocator
        with self._swa_anchor.lock:
            if getattr(self, "_push_fast", False):
                swa_gens = _page_generations(swa_slots, swa_pages)
            else:
                swa_gens = [swa_slots.page_generation(p) for p in swa_pages]
        self._deltas.put((_DELTA_STORE_SWA, (list(hashes), swa_pages, swa_gens)))

    @staticmethod
    def _snapshot_remove_pages(
        anchor: Any, indices_pages: Sequence[Any], fast: bool = False
    ):
        """Copy scheduler-owned CPU identities before their host slots are freed."""
        if fast:
            return PeerKVServicer._snapshot_remove_pages_batched(anchor, indices_pages)
        pages = [None if p is None else p.clone() for p in indices_pages]
        generations = [None] * len(pages)
        if any(p is not None for p in pages):
            if anchor is None:
                raise ValueError("retiring host pages require their allocator anchor")
            with anchor.lock:
                for i, p in enumerate(pages):
                    if p is not None:
                        generations[i] = anchor.slot_allocator.page_generation(p)
        return pages, generations

    @staticmethod
    def _snapshot_remove_pages_batched(anchor: Any, indices_pages: Sequence[Any]):
        """SGLANG_PEER_KV_PUSH_FAST variant of _snapshot_remove_pages.

        One copy for the whole call (torch.cat allocates), split back into
        per-page views of that private copy, then one generation read."""
        pages: List[Optional[Any]] = [None] * len(indices_pages)
        live = [i for i, p in enumerate(indices_pages) if p is not None]
        if len(live) == 1:
            pages[live[0]] = indices_pages[live[0]].clone()
        elif live:
            flat = torch.cat([indices_pages[i] for i in live])
            views = flat.split([len(indices_pages[i]) for i in live])
            for i, view in zip(live, views):
                pages[i] = view
        generations: List[Optional[List[int]]] = [None] * len(pages)
        if live:
            if anchor is None:
                raise ValueError("retiring host pages require their allocator anchor")
            with anchor.lock:
                generations = _page_generations(anchor.slot_allocator, pages)
        return pages, generations

    def push_removed(
        self, hashes: Sequence[str], kv_indices_pages: Sequence[Any]
    ) -> None:
        """Retire this FULL backing, without removing a newer STORE of the hash."""
        if len(hashes) != len(kv_indices_pages):
            raise ValueError("FULL removal hashes and physical pages must align")
        pages, generations = self._snapshot_remove_pages(
            self._kv_anchor, kv_indices_pages, getattr(self, "_push_fast", False)
        )
        self._deltas.put((_DELTA_REMOVE, (list(hashes), pages, generations)))

    def push_swa_removed(
        self, hashes: Sequence[str], swa_indices_pages: Sequence[Optional[Any]]
    ) -> None:
        """Retire only matching SWA slices; pages outside the tail have no slice."""
        if len(hashes) != len(swa_indices_pages):
            raise ValueError("SWA removal hashes and physical pages must align")
        pages, generations = self._snapshot_remove_pages(
            self._swa_anchor, swa_indices_pages, getattr(self, "_push_fast", False)
        )
        self._deltas.put((_DELTA_REMOVE_SWA, (list(hashes), pages, generations)))

    # ---- SGLANG_PEER_KV_PUSH_FAST: whole-node range pushes -------------
    # One node = ``len(hashes)`` consecutive pages of ``page_size`` slots in
    # one 1-D index tensor. The scheduler thread does O(1) torch ops per node
    # (one generation gather, one clone on removal); the servicer thread
    # splits the range into the per-page payload of the per-page API, so the
    # shadow index ends up exactly as the per-page calls would leave it.
    # A geometry the vectorized gather cannot express (page_size not a
    # multiple of the anchor granule) falls back to the per-page API.

    @staticmethod
    def _range_ok(anchor: Any, page_size: int) -> bool:
        granule = int(anchor.slot_allocator.granule)
        return page_size > 0 and page_size % granule == 0

    @staticmethod
    def _range_pages(indices: Any, num_pages: int, page_size: int) -> list:
        return [indices[i * page_size : (i + 1) * page_size] for i in range(num_pages)]

    @staticmethod
    def _check_range(indices: Any, num_pages: int, page_size: int, what: str) -> None:
        if indices.dim() != 1 or int(indices.numel()) != num_pages * page_size:
            raise ValueError(
                f"{what}: {tuple(indices.shape)} indices for {num_pages} pages "
                f"of {page_size}"
            )

    def push_store_range(
        self,
        hashes: Sequence[str],
        kv_indices: Any,
        swa_indices: Optional[Any],
        swa_first_page: int,
        page_size: int,
    ) -> None:
        """push_store of a whole node. ``kv_indices``: the node's pages
        (``len(hashes) * page_size`` slots). ``swa_indices``: pages
        ``[swa_first_page, len(hashes))`` (the trailing window), None when
        the node has no SWA slice; earlier pages publish no SWA slice."""
        hashes = list(hashes)
        n = len(hashes)
        m = 0 if swa_indices is None else n - int(swa_first_page)
        if m <= 0:
            swa_indices = None
        if not self._range_ok(self._kv_anchor, page_size) or (
            swa_indices is not None
            and self._swa_anchor is not None
            and not self._range_ok(self._swa_anchor, page_size)
        ):
            swa_pages: list = [None] * n
            if swa_indices is not None:
                swa_pages[n - m :] = self._range_pages(swa_indices, m, page_size)
            self.push_store(
                hashes, self._range_pages(kv_indices, n, page_size), swa_pages
            )
            return
        self._check_range(kv_indices, n, page_size, "store kv")
        with self._kv_anchor.lock:
            kv_gens = self._kv_anchor.slot_allocator.page_generation(kv_indices)
        swa_gens = None
        if swa_indices is not None:
            self._check_range(swa_indices, m, page_size, "store swa")
            if self._swa_anchor is not None:
                with self._swa_anchor.lock:
                    swa_gens = self._swa_anchor.slot_allocator.page_generation(
                        swa_indices
                    )
        self._deltas.put(
            (
                _DELTA_STORE_RANGE,
                (hashes, kv_indices, swa_indices, n - m, page_size, kv_gens, swa_gens),
            )
        )

    def _push_removed_range(
        self, kind: int, anchor: Any, hashes: list, indices: Any, first: int, ps: int
    ) -> None:
        n = len(hashes)
        m = n - first
        self._check_range(indices, m, ps, "remove")
        if anchor is None:
            raise ValueError("retiring host pages require their allocator anchor")
        # Copy the identity (slots, generations) before the caller frees it.
        snapshot = indices.clone()
        with anchor.lock:
            gens = anchor.slot_allocator.page_generation(snapshot)
        self._deltas.put((kind, (hashes, snapshot, first, ps, gens)))

    def push_removed_range(
        self, hashes: Sequence[str], kv_indices: Any, page_size: int
    ) -> None:
        """push_removed of a whole node (``len(hashes) * page_size`` slots).
        Snapshots slots and generations before returning, as push_removed."""
        hashes = list(hashes)
        if not self._range_ok(self._kv_anchor, page_size):
            self.push_removed(
                hashes, self._range_pages(kv_indices, len(hashes), page_size)
            )
            return
        self._push_removed_range(
            _DELTA_REMOVE_RANGE, self._kv_anchor, hashes, kv_indices, 0, page_size
        )

    def push_swa_removed_range(
        self,
        hashes: Sequence[str],
        swa_indices: Any,
        swa_first_page: int,
        page_size: int,
    ) -> None:
        """push_swa_removed of a whole node: ``swa_indices`` holds pages
        ``[swa_first_page, len(hashes))``; earlier pages have no slice."""
        hashes = list(hashes)
        n = len(hashes)
        m = n - int(swa_first_page)
        if m <= 0:
            return  # no page has a slice: the per-page delta is a no-op
        if self._swa_anchor is None or not self._range_ok(self._swa_anchor, page_size):
            swa_pages: list = [None] * n
            swa_pages[n - m :] = self._range_pages(swa_indices, m, page_size)
            self.push_swa_removed(hashes, swa_pages)
            return
        self._push_removed_range(
            _DELTA_REMOVE_SWA_RANGE,
            self._swa_anchor,
            hashes,
            swa_indices,
            n - m,
            page_size,
        )

    @staticmethod
    def _split_gens(gens: Optional[list], num_pages: int) -> list:
        if gens is None:
            return [None] * num_pages
        per = len(gens) // num_pages if num_pages else 0
        if per == 1:
            return [[g] for g in gens]
        return [gens[i * per : (i + 1) * per] for i in range(num_pages)]

    @staticmethod
    def _split_rows(indices: Any, num_pages: int, page_size: int) -> list:
        if num_pages == 0:
            return []
        return list(indices.reshape(num_pages, page_size).unbind(0))

    def _expand_range_delta(self, kind: int, payload: Any) -> Tuple[int, Any]:
        """A range delta -> (per-page kind, per-page payload). Servicer thread."""
        if kind == _DELTA_STORE_RANGE:
            hashes, kv_idx, swa_idx, first, ps, kv_gens, swa_gens = payload
            n = len(hashes)
            m = n - first
            swa_pages: list = [None] * n
            swa_gen_pages: list = [None] * n
            if swa_idx is not None:
                swa_pages[first:] = self._split_rows(swa_idx, m, ps)
                swa_gen_pages[first:] = self._split_gens(swa_gens, m)
            return _DELTA_STORE, (
                hashes,
                self._split_rows(kv_idx, n, ps),
                swa_pages,
                self._split_gens(kv_gens, n),
                swa_gen_pages,
            )
        hashes, idx, first, ps, gens = payload
        n = len(hashes)
        m = n - first
        pages: list = [None] * n
        gen_pages: list = [None] * n
        pages[first:] = self._split_rows(idx, m, ps)
        gen_pages[first:] = self._split_gens(gens, m)
        per_page = _DELTA_REMOVE if kind == _DELTA_REMOVE_RANGE else _DELTA_REMOVE_SWA
        return per_page, (hashes, pages, gen_pages)

    def push_clear(self) -> None:
        """The whole tree is being reset (flush). NOTICEs held for their
        grace period name nodes of the old tree: drop them (counted)."""
        if getattr(self, "_notice_held", None):
            with self._notice_lock:
                self.notice_held_dropped += len(self._notice_held)
                self._notice_held.clear()
        self._deltas.put((_DELTA_CLEAR, None))

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)
        if self._trace_file is not None:
            self._trace_file.close()

    # ------------------------------------------------------------------
    # Servicer thread
    # ------------------------------------------------------------------

    def _serve_loop(self) -> None:
        ctx = zmq.Context.instance()
        socket = ctx.socket(zmq.REP)
        try:
            socket.bind(f"tcp://0.0.0.0:{self._bind_port}")
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            poller.register(self._wake_read, zmq.POLLIN)
        except Exception as exc:
            self._startup_error = exc
            socket.close(linger=0)
            os.close(self._wake_read)
            os.close(self._wake_write)
            self._startup_done.set()
            return
        self._startup_done.set()
        logger.info("PeerKVServicer listening on tcp://0.0.0.0:%d", self._bind_port)
        pending = {}
        try:
            while not self._stop_event.is_set():
                self._drain_deltas()
                self._tick(time.monotonic())
                self._start_notifications(ctx, poller, pending)
                # Give incoming LOOKUP/RELEASE a turn between bounded encoding
                # batches, without adding 50 ms to a nonempty control outbox.
                ready = dict(
                    poller.poll(0 if not self._outbox.empty() else _POLL_INTERVAL_MS)
                )
                if self._wake_read in ready:
                    try:
                        while os.read(self._wake_read, 4096):
                            pass
                    except BlockingIOError:
                        pass
                now = time.monotonic()
                for peer, (deadline, tokens) in list(pending.items()):
                    if peer not in ready and now < deadline:
                        continue
                    success = False
                    if peer in ready:
                        try:
                            success = (
                                _msg_decoder.decode(peer.recv(zmq.NOBLOCK)).get("ok")
                                is True
                            )
                        except (zmq.ZMQError, msgspec.DecodeError, AttributeError):
                            pass
                    poller.unregister(peer)
                    peer.close(linger=0)
                    del pending[peer]
                    self._retire_notification(tokens, success)
                if socket in ready:
                    raw = socket.recv()
                    try:
                        reply = self._handle_message(_msg_decoder.decode(raw))
                    except Exception as e:  # protocol-level fail-open
                        logger.warning("PeerKVServicer request failed: %s", e)
                        reply = {"ok": False, "error": str(e)}
                    socket.send(_msg_encoder.encode(reply))
        finally:
            for peer, (_, tokens) in pending.items():
                peer.close(linger=0)
                self._retire_notification(tokens, False)
            while True:
                try:
                    _, _, tokens = self._outbox.get_nowait()
                except Empty:
                    break
                self._retire_notification(tokens, False)
            socket.close(linger=0)
            os.close(self._wake_read)
            os.close(self._wake_write)

    _COUNTERS_LOG_S = 60.0

    def _counters_line(self) -> Optional[str]:
        """The PING counters of the pull-miss flags that are on, as one line
        (None when none is on)."""
        parts: List[str] = []
        if self._budget_partial_grant:
            parts += [
                f"budget_partial_grants={self._budget_partial_grants}",
                f"budget_trimmed_pages={self._budget_trimmed_pages}",
            ]
        if self._lookup_deadline:
            parts += [
                f"lookup_late_at_start={self._late_at_start}",
                f"lookup_late_at_grant={self._late_at_grant}",
            ]
        if self._budget_precheck:
            parts += [
                f"budget_precheck_refusals={self._budget_precheck_refusals}",
                f"page_nbytes_floor={self._page_nbytes_floor}",
            ]
        if self._touch_outbox is not None:
            parts += [
                f"touch_received={self._touch_received}",
                f"touch_queued={self._touch_queued}",
                f"touch_after_lookup={self.touch_after_lookup}",
                f"touch_dropped_by_lookup={self.touch_dropped_by_lookup}",
                f"touch_lookup_no_rid={self._touch_lookup_no_rid}",
            ]
        if self._demote_counters:
            parts += [
                f"demote_outbox_drops={self.demote_outbox_drops}",
                f"demote_touch_outbox_drops={self.demote_touch_outbox_drops}",
                f"demote_commit_pending={self.demote_commit_pending}",
                f"demote_commit_ok={self.demote_commit_ok}",
                f"demote_commit_expired={self.demote_commit_expired}",
                f"demote_commit_unknown={self.demote_commit_unknown}",
            ]
        return " ".join(parts) if parts else None

    def _maybe_log_counters(self, now_mono: float) -> None:
        if now_mono - self._counters_logged_at < self._COUNTERS_LOG_S:
            return
        self._counters_logged_at = now_mono
        line = self._counters_line()
        if line is not None and line != self._counters_last_line:
            self._counters_last_line = line
            logger.info("PeerKV servicer counters: port=%d %s", self._bind_port, line)

    def _tick(self, now_mono: float) -> None:
        """Expire leases (unpin with cooling) and drain cooled pages back to
        the anchors' free lists."""
        self._maybe_log_counters(now_mono)
        self._expire_leases(now_mono)
        if self.gpu_leases is not None:
            self.gpu_leases.expire(now_mono)
        for pool in (self._kv_anchor, self._swa_anchor):
            if pool is None:
                continue
            with pool.lock:
                pool.slot_allocator.drain_cooling(now_mono)

    def _handle_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        op = msg["op"]
        if op in ("prepare_gpu", "prepare_gpu_status", "prepare_gpu_release"):
            broker = getattr(self, "gpu_export_broker", None)
            if broker is None:
                return {"ok": False, "reason": "gpu_export_unsupported"}
            if op == "prepare_gpu":
                return broker.request(msg.get("hashes"), msg.get("waiter"))
            if op == "prepare_gpu_status":
                return broker.status(msg.get("export_id"), msg.get("waiter"))
            return broker.release(msg.get("export_id"), msg.get("waiter"))
        if op == "lookup":
            return self._handle_lookup(msg)
        if op == "lookup_swa_range":
            return self._handle_swa_range_lookup(msg)
        if op in _CACHE_COMMAND_TYPES:
            operation_id = msg.get("operation_id")
            if not isinstance(operation_id, str) or not operation_id:
                return {"ok": False, "reason": "invalid_operation_id"}
            event = dict(msg)
            event["type"] = event.pop("op")
            accepted = self.enqueue_cache_event(event)
            return {"ok": accepted, "reason": None if accepted else "cache_queue_full"}
        if op == "release":
            pages = msg.get("pages")
            self._release_lease(
                msg["lease_id"],
                reason="release",
                complete=msg.get("complete") is True,
                pages=pages if type(pages) is int and pages >= 0 else None,
                after_commit=msg.get("demote_after_commit") is True,
            )
            return {"ok": True}
        if op == "demote_ok" and self._demote_outbox is not None:
            return self._handle_demote_ok(msg)
        if op == "notice":
            return self._handle_notice(msg)
        if op == "ping":
            with self._lease_mutex:
                leases = len(self._leases)
                leased_bytes = self._leased_bytes_total
            reply = {
                "ok": True,
                "pages_indexed": len(self._shadow),
                "leases": leases,
                "leased_bytes": leased_bytes,
                "pin_leaks": self._pin_leaks,
                **(
                    {
                        "budget_partial_grants": self._budget_partial_grants,
                        "budget_trimmed_pages": self._budget_trimmed_pages,
                    }
                    if self._budget_partial_grant
                    else {}
                ),
                **(
                    {
                        "lookup_late_at_start": self._late_at_start,
                        "lookup_late_at_grant": self._late_at_grant,
                    }
                    if self._lookup_deadline
                    else {}
                ),
                **(
                    {
                        "budget_precheck_refusals": self._budget_precheck_refusals,
                        "page_nbytes_floor": self._page_nbytes_floor,
                    }
                    if self._budget_precheck
                    else {}
                ),
                **(
                    {
                        "touch_received": self._touch_received,
                        "touch_queued": self._touch_queued,
                        "touch_after_lookup": self.touch_after_lookup,
                        "touch_dropped_by_lookup": self.touch_dropped_by_lookup,
                        "touch_lookup_no_rid": self._touch_lookup_no_rid,
                    }
                    if self._touch_outbox is not None
                    else {}
                ),
                **(
                    {
                        "demote_outbox_drops": self.demote_outbox_drops,
                        "demote_touch_outbox_drops": self.demote_touch_outbox_drops,
                        "demote_commit_pending": self.demote_commit_pending,
                        "demote_commit_ok": self.demote_commit_ok,
                        "demote_commit_expired": self.demote_commit_expired,
                        "demote_commit_unknown": self.demote_commit_unknown,
                        "demote_commit_lag": list(self.demote_commit_lag),
                        "demote_commit_lag_max_ms": self.demote_commit_lag_max_ms,
                        "demote_ignored_role": self.demote_ignored_role,
                        **{
                            f"notice_{k}": getattr(self, f"notice_{k}")
                            for k in self.NOTICE_COUNTERS
                        },
                    }
                    if self._demote_counters
                    else {}
                ),
                "verify_checksums": self._verify_checksums,
                "poison_on_release": self._poison_installed,
                "cache_events_dropped": self.cache_events_dropped,
                "control_notifications_failed": self.control_notifications_failed,
                # Readers negotiate optional ops here; a legacy owner lacks
                # the key and is treated as incapable. Ranges need a complete
                # host SWA, not just an SWA pool.
                "capabilities": (
                    ["host_return_v1"]
                    + ([SWA_RANGE_CAPABILITY] if self._serves_swa_ranges() else [])
                ),
                "page_size": self._page_size,
            }
            for label, pool in (("kv", self._kv_anchor), ("swa", self._swa_anchor)):
                if pool is None:
                    continue
                with pool.lock:
                    reply[f"{label}_slots"] = pool.slot_allocator.stats()
            if self.gpu_leases is not None:
                reply["gpu_leases"] = self.gpu_leases.stats()
            return reply
        if op == "touch" and self._touch_outbox is not None:
            return self._handle_touch(msg)
        if op == "debug":
            # Runtime toggles for the canary detectors (pod specs are immutable
            # in place; the image swap keeps the launch flags). Rank-local,
            # servicer-thread only: no structural effect.
            if "verify_checksums" in msg:
                self._verify_checksums = bool(msg["verify_checksums"])
            if "poison" in msg:
                if msg["poison"]:
                    self._install_poison_hooks()
                else:
                    self._remove_poison_hooks()
            return {
                "ok": True,
                "verify_checksums": self._verify_checksums,
                "poison_on_release": self._poison_installed,
            }
        return {"ok": False, "error": f"unknown op {op!r}"}

    # ------------------------------------------------------------------
    # LOOKUP
    # ------------------------------------------------------------------

    def _past_deadline(self, msg: Dict[str, Any]) -> bool:
        """F5: the reader's reply deadline (its wall clock at the send + its
        RPC timeout) + the slack has passed: the reader has given up and its
        socket drops the reply. A LOOKUP without a deadline (a legacy reader,
        or the flag off at the reader) never is."""
        if not self._lookup_deadline:
            return False
        deadline = msg.get("deadline_unix_s")
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
            return False
        return time.time() > deadline + self._lookup_deadline_slack_s

    def _lookup_tick(self) -> Optional[float]:
        if self._demote_tick is None:
            return None
        tick = self._demote_tick()
        if self._touch_outbox is not None:
            # F1: TOUCHes drained before this point are applied with the ticks
            # reserved at the drain, all below this end.
            tick = max(tick, self._touch_reserved_end)
        return tick

    def drain_demote_outbox(self, limit: int = 4096) -> List[Tuple]:
        """Scheduler thread: up to ``limit`` (leased keys, lookup tick); then
        the held NOTICEs that are due."""
        out: List[Tuple] = []
        outbox = self._demote_outbox
        while outbox and len(out) < limit:
            try:
                out.append(outbox.popleft())
            except IndexError:
                break
        if getattr(self, "_notice_held", None):
            now = self._now()
            with self._notice_lock:
                keep = []
                for item in self._notice_held:
                    due, keys, tick, rid = item
                    if due <= now and len(out) < limit:
                        entry = DemoteOutboxEntry((keys, tick, rid), "notice")
                        entry.held = True
                        out.append(entry)
                    else:
                        keep.append(item)
                self._notice_held[:] = keep
        return out

    def drain_touch_outbox(
        self, limit: int = 1024, tick: Optional[float] = None
    ) -> List[Tuple]:
        """Scheduler thread: up to ``limit`` (confirmed keys, from the first
        page) of TOUCHes. ``tick``: the next access tick now. The drained
        TOUCHes own the ticks [tick, tick + their number); a LOOKUP from now on
        takes its lookup tick at or after their end (``_lookup_tick``)."""
        out: List[Tuple] = []
        outbox = self._touch_outbox
        if not outbox:
            return out
        with self._touch_lock:
            while outbox and len(out) < limit:
                keys, from_root, _rid = outbox.popleft()
                out.append((keys, from_root))
            if tick is not None and out:
                self._touch_reserved_end = max(
                    self._touch_reserved_end, float(tick) + len(out)
                )
        return out

    def _handle_touch(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """TOUCH (SGLANG_PEER_KV_TOUCH_DEFERRED_PULLS): a destination keeps the
        span of a deferred pull warm. Confirm the published prefix of the keys
        (as LOOKUP does, but no pin and no lease) and queue it for the
        scheduler thread, which refreshes the tree on every rank. Not after
        the LOOKUP of the same request (``_touch_note_lookup``)."""
        hashes = msg.get("hashes")
        if not isinstance(hashes, list):
            return {"ok": False, "error": "touch needs a hashes list"}
        self._touch_received += 1
        rid = msg.get("rid")
        rid = rid if isinstance(rid, str) and 0 < len(rid) <= 256 else ""
        if rid and self._touch_after_lookup(rid):
            self.touch_after_lookup += 1
            return {"ok": True, "pages": 0, "after_lookup": True}
        self._drain_deltas()
        confirmed = 0
        for h in hashes:
            entry = self._shadow.get(h) if isinstance(h, str) else None
            if entry is None or entry.kv_indices is None:  # absent or SWA-only
                break
            confirmed += 1
        if confirmed:
            with self._touch_lock:
                if len(self._touch_outbox) >= self._touch_outbox.maxlen:
                    self.demote_touch_outbox_drops += 1  # the oldest goes
                self._touch_outbox.append(
                    (hashes[:confirmed], msg.get("first_page") == 0, rid)
                )
            self._touch_queued += 1
        return {"ok": True, "pages": confirmed}

    def _touch_after_lookup(self, rid: str) -> bool:
        expiry = self._touch_lookup_rids.get(rid)
        if expiry is None:
            return False
        if expiry > time.monotonic():
            return True
        del self._touch_lookup_rids[rid]
        return False

    def _touch_note_lookup(self, msg: Dict[str, Any]) -> None:
        """F1, every LOOKUP kind, before it takes its lookup tick: the TOUCHes
        of the request stop here. Drop the queued ones and remember the
        request for the lease TTL + 60 s, so a late TOUCH of it (sent before
        the issue, delivered after) is not queued. A TOUCH drained before
        this point refreshes with a tick reserved at its drain, which is below
        the lookup tick (``_lookup_tick``). So the complete RELEASE of this
        lease demotes the path as without F1."""
        if self._touch_outbox is None:
            return
        rid = _request_trace_metadata(msg).get("req_id")
        if not rid:
            self._touch_lookup_no_rid += 1
            return
        try:
            ttl_ms = min(
                int(msg.get("lease_ttl_ms", _LEASE_TTL_DEFAULT_MS)), _LEASE_TTL_MAX_MS
            )
        except (TypeError, ValueError):
            ttl_ms = _LEASE_TTL_DEFAULT_MS
        now = time.monotonic()
        rids = self._touch_lookup_rids
        # A later LOOKUP of the request (retry, re-pull) never shortens the
        # block, also with a shorter lease TTL.
        expiry = now + max(0, ttl_ms) / 1e3 + self._TOUCH_LOOKUP_GRACE_S
        rids[rid] = max(expiry, rids.get(rid, 0.0))
        rids.move_to_end(rid)
        while rids:
            oldest, expiry = next(iter(rids.items()))
            if expiry > now and len(rids) <= self._TOUCH_LOOKUP_RIDS_MAX:
                break
            del rids[oldest]
        with self._touch_lock:
            outbox = self._touch_outbox
            if any(entry[2] == rid for entry in outbox):
                kept = [entry for entry in outbox if entry[2] != rid]
                self.touch_dropped_by_lookup += len(outbox) - len(kept)
                outbox.clear()
                outbox.extend(kept)

    def _handle_lookup(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        self._touch_note_lookup(msg)
        if msg.get("require_complete_endpoint") is True:
            return self._handle_complete_lookup(msg)
        t0 = time.perf_counter()
        lookup_tick = self._lookup_tick()
        hashes: List[str] = msg["hashes"]
        trace_metadata = _request_trace_metadata(msg)
        ttl_ms = min(
            int(msg.get("lease_ttl_ms", _LEASE_TTL_DEFAULT_MS)), _LEASE_TTL_MAX_MS
        )
        if self._past_deadline(msg):
            # F5: the reader gave up before this LOOKUP started: no pins.
            self._late_at_start += 1
            return self._lookup_reply(
                t0,
                hashes,
                0,
                0,
                "",
                "late",
                trace_metadata=trace_metadata,
                confirmed=0,
                lookup_tick=lookup_tick,
            )

        self._drain_deltas()
        confirmed = 0
        for h in hashes:
            entry = self._shadow.get(h)
            if entry is None or entry.kv_indices is None:  # absent or SWA-only
                break
            confirmed += 1
        if confirmed == 0:
            return self._lookup_reply(
                t0,
                hashes,
                0,
                0,
                "",
                "no_pages",
                trace_metadata=trace_metadata,
                confirmed=0,
                lookup_tick=lookup_tick,
            )
        if self._budget_precheck and self._page_nbytes_floor is not None:
            # F6: refuse before the pins when even the smallest grant that the
            # check after the pins could make (all confirmed pages; with F4
            # its minimum) does not fit. Leases change on this thread only.
            fewest = (
                min(confirmed, self._budget_partial_min_pages)
                if self._budget_partial_grant
                else confirmed
            )
            with self._lease_mutex:
                room = self._max_leased_bytes - self._leased_bytes_total
            if fewest * self._page_nbytes_floor > room:
                self._budget_precheck_refusals += 1
                return self._lookup_reply(
                    t0,
                    hashes,
                    0,
                    0,
                    "",
                    "budget",
                    trace_metadata=trace_metadata,
                    confirmed=confirmed,
                    lookup_tick=lookup_tick,
                )

        entries = [self._shadow[h] for h in hashes[:confirmed]]

        # --- pin the KV-anchor pages (slot + generation, prefix semantics) ---
        kv_slots = self._kv_anchor.slot_allocator
        num_pages = self._pin_entries(
            self._kv_anchor,
            [e.kv_indices for e in entries],
            [e.kv_gen for e in entries],
        )
        if num_pages == 0:
            with self._kv_anchor.lock:
                exhausted = kv_slots.reserve_pages - kv_slots.num_protected <= 0
            return self._lookup_reply(
                t0,
                hashes,
                0,
                0,
                "",
                "reserve" if exhausted else "pin",
                num_pages_partial=confirmed,
                trace_metadata=trace_metadata,
                confirmed=confirmed,
                lookup_tick=lookup_tick,
            )
        pinned_kv = [e.kv_indices for e in entries[:num_pages]]
        entries = entries[:num_pages]

        # --- SWA anchor: trailing window only (the only pages transferred) ---
        swa_pinned: List[Tuple[int, Any]] = []
        swa_ok_from = num_pages  # entry pages >= this index carry SWA indices
        if self._swa_anchor is not None and self._swa_trailing_pages > 0:
            start = max(0, num_pages - self._swa_trailing_pages)
            cand = [
                (i, e.swa_indices, e.swa_gen)
                for i, e in enumerate(entries)
                if i >= start and e.swa_indices is not None and e.swa_gen is not None
            ]
            if cand:
                full = self._pin_entries(
                    self._swa_anchor, [c[1] for c in cand], [c[2] for c in cand]
                )
                if full == len(cand):
                    swa_pinned = [(c[0], c[1]) for c in cand]
                    swa_ok_from = cand[0][0]
                elif full > 0:
                    # A prefix of the trailing window is useless to the reader
                    # (it needs the LAST pages): give the pins back, KV only.
                    with self._swa_anchor.lock:
                        self._swa_anchor.slot_allocator.unpin(
                            torch.cat([c[1] for c in cand[:full]]), 0.0
                        )

        pools, page_nbytes = self._build_pool_addresses(entries, swa_ok_from)
        total_bytes = sum(page_nbytes)
        if self._budget_precheck and swa_ok_from > 0:
            # F6: the pages before swa_ok_from carry KV bytes only (the same
            # for every page of this owner); the floor is never above them.
            floor = min(page_nbytes[:swa_ok_from])
            if self._page_nbytes_floor is None or floor < self._page_nbytes_floor:
                self._page_nbytes_floor = floor
        if self._past_deadline(msg):
            # F5: the reader gave up while this LOOKUP pinned and built its
            # addresses (LVR1-F-1 67caf1bb: it started 0.2 s before the
            # deadline and granted 0.2 s after it, and the lease held 39 GB
            # of the budget until its TTL). Give the pins back now.
            self._late_at_grant += 1
            self._unpin(pinned_kv, swa_pinned, cool_s=0.0)
            return self._lookup_reply(
                t0,
                hashes,
                0,
                0,
                "",
                "late",
                num_pages_partial=confirmed,
                trace_metadata=trace_metadata,
                confirmed=confirmed,
                lookup_tick=lookup_tick,
            )

        refused: Optional[str] = None
        lease_id = ""
        trimmed_kv: List[Any] = []
        with self._lease_mutex:
            if self._leased_bytes_total + total_bytes > self._max_leased_bytes:
                keep = self._budget_fit(page_nbytes) if not swa_pinned else 0
                if keep:
                    # F4: lease the prefix that fits; give the tail back.
                    trimmed_kv = pinned_kv[keep:]
                    pinned_kv = pinned_kv[:keep]
                    page_nbytes = page_nbytes[:keep]
                    total_bytes = sum(page_nbytes)
                    self._budget_partial_grants += 1
                    self._budget_trimmed_pages += num_pages - keep
                    num_pages = keep
                else:
                    refused = "budget"
            if refused is None:
                lease_id = uuid.uuid4().hex
                self._leases[lease_id] = _Lease(
                    tuple(hashes[:num_pages]),
                    time.monotonic() + ttl_ms / 1e3,
                    total_bytes,
                    pinned_kv,
                    swa_pinned,
                    trace_metadata=trace_metadata,
                    lookup_tick=lookup_tick,
                )
                self._leased_bytes_total += total_bytes
        if trimmed_kv:
            self._unpin(trimmed_kv, [], cool_s=0.0)
        if refused == "budget":
            self._unpin(pinned_kv, swa_pinned, cool_s=0.0)
            return self._lookup_reply(
                t0,
                hashes,
                0,
                0,
                "",
                refused,
                trace_metadata=trace_metadata,
                confirmed=confirmed,
                lookup_tick=lookup_tick,
            )

        # Retraction pass: a delta that raced the pin may have evicted a
        # just-leased page (its bytes stay intact — it is withheld — but the
        # shadow no longer advertises it): drain and truncate for precision.
        removed = self._drain_deltas()
        retracted = 0
        if removed:
            keep = num_pages
            for i in range(num_pages):
                if hashes[i] in removed:
                    keep = i
                    break
            if keep < num_pages:
                retracted = num_pages - keep
                self._shrink_lease(lease_id, keep, page_nbytes)
                num_pages = keep
                if num_pages == 0:
                    lease_id = ""

        reply = self._lookup_reply(
            t0,
            hashes,
            num_pages,
            retracted,
            lease_id,
            None,
            num_pages_partial=confirmed - num_pages - retracted,
            swa_pages_pinned=len(swa_pinned),
            trace_metadata=trace_metadata,
            confirmed=confirmed,
            lookup_tick=lookup_tick,
        )
        if num_pages > 0:
            reply["pools"] = {name: pages[:num_pages] for name, pages in pools.items()}
            if self._verify_checksums:
                reply["sums"] = self._checksums(reply["pools"])
            if self._poison_installed:
                # Tells the reader to look for the poison pattern after reads.
                reply["poison"] = True
        return reply

    def _budget_fit(self, page_nbytes: List[int]) -> int:
        """F4 (under ``_lease_mutex``): the number of leading pages whose
        bytes fit the lease budget that is left, or 0 when partial grants are
        off or fewer than the minimum fit (the caller then refuses as before).
        Only the legacy KV path uses it: a trailing SWA window pinned for the
        full span would not cover a shorter end."""
        if not self._budget_partial_grant:
            return 0
        room = self._max_leased_bytes - self._leased_bytes_total
        keep = 0
        used = 0
        for nbytes in page_nbytes:
            if used + nbytes > room:
                break
            used += nbytes
            keep += 1
        return keep if keep >= self._budget_partial_min_pages else 0

    def _handle_complete_lookup(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Return one completely pinned endpoint, never a FULL-only maximum.

        Endpoint validity is not prefix-monotone: a shorter endpoint needs
        different trailing slots. Scan downwards and acquire its entire tail
        before publishing a grant. The reader still verifies its own required
        pool descriptors and negotiates the same endpoint across TP ranks.
        Legacy LOOKUPs deliberately retain their existing wire/trim semantics.
        """
        t0 = time.perf_counter()
        lookup_tick = self._lookup_tick()
        hashes = msg["hashes"]
        trace_metadata = _request_trace_metadata(msg)
        ttl_ms = min(
            int(msg.get("lease_ttl_ms", _LEASE_TTL_DEFAULT_MS)), _LEASE_TTL_MAX_MS
        )
        refused = "no_pages"
        confirmed = 0
        partial = 0
        # None means this attempt never scanned/pinned FULL; do not invent zero.
        full_shadow_pages = None
        full_pinned_pages = None
        on_swa_names = {name for name, _, on_swa, _ in self._pool_plan if on_swa}
        full_swa_prefix = msg.get("require_full_swa_prefix") is True

        def reply(num_pages=0, lease_id="", pools=None, swa_pages=0):
            result = self._lookup_reply(
                t0,
                hashes,
                num_pages,
                0,
                lease_id,
                None if num_pages else refused,
                num_pages_partial=partial,
                swa_pages_pinned=swa_pages,
                trace_metadata=trace_metadata,
                exact_endpoint=True,
                full_shadow_pages=full_shadow_pages,
                full_pinned_pages=full_pinned_pages,
                confirmed=full_shadow_pages or 0,
                lookup_tick=lookup_tick,
            )
            result.update(
                exact_endpoint=True,
                lease_ttl_ms=ttl_ms,
                refused_reason=None if num_pages else refused,
            )
            if pools is not None:
                result["pools"] = pools
                if self._verify_checksums:
                    result["sums"] = self._checksums(pools)
                if self._poison_installed:
                    result["poison"] = True
            return result

        if self._past_deadline(msg):
            # F5: as in _handle_lookup.
            self._late_at_start += 1
            refused = "late"
            return reply()

        if on_swa_names and (self._swa_anchor is None or self._swa_trailing_pages <= 0):
            refused = "unsupported_tail_contract"
            return reply()

        for _ in range(_EXACT_LOOKUP_MAX_RETRIES):
            self._drain_deltas()
            source_entries = []
            for h in hashes:
                entry = self._shadow.get(h)
                if entry is None or entry.kv_indices is None:  # SWA-only
                    break
                source_entries.append(entry)
            confirmed = len(source_entries)
            full_shadow_pages = confirmed
            full_pinned_pages = None
            partial = 0
            if not confirmed:
                refused = "no_pages"
                return reply()

            # REMOVE_SWA mutates shadow entries in place. Retain the pinned
            # snapshot separately; a later drain must not rewrite its tail.
            entries = [
                _PageEntry(e.kv_indices, e.swa_indices, e.kv_gen, e.swa_gen)
                for e in source_entries
            ]
            pinned_kv = []
            pinned_swa = []
            lease_id = ""
            try:
                count = self._pin_entries(
                    self._kv_anchor,
                    [e.kv_indices for e in entries],
                    [e.kv_gen for e in entries],
                )
                pinned_kv = [e.kv_indices for e in entries[:count]]
                full_pinned_pages = count
                partial = confirmed - count
                if not count:
                    refused = "pin"
                    return reply()
                entries = entries[:count]
                # Compute ALL-pool descriptors once. Only the small candidate
                # tail changes during the descending search (no quadratic
                # reconstruction of FULL addresses under a byte budget).
                full_pools, full_bytes = self._build_pool_addresses(
                    entries, count, require_complete=True
                )
                prefix_bytes = [0]
                for nbytes in full_bytes:
                    prefix_bytes.append(prefix_bytes[-1] + nbytes)
                refused = "no_complete_endpoint"
                uniform_pools = None
                uniform_bytes = None
                candidates = range(count, 0, -1)
                if full_swa_prefix and on_swa_names:
                    # Unlike a trailing checkpoint, complete-prefix coverage is
                    # prefix-monotone. Pin once and account bytes once; repeatedly
                    # rebuilding N, N-1, ... prefixes under pressure is quadratic.
                    complete = next(
                        (
                            i
                            for i, e in enumerate(entries)
                            if e.swa_indices is None or e.swa_gen is None
                        ),
                        count,
                    )
                    if not complete:
                        return reply()
                    got = self._pin_entries(
                        self._swa_anchor,
                        [e.swa_indices for e in entries[:complete]],
                        [e.swa_gen for e in entries[:complete]],
                    )
                    pinned_swa = [
                        (i, e.swa_indices) for i, e in enumerate(entries[:got])
                    ]
                    if not got:
                        return reply()
                    uniform_pools, _ = self._build_pool_addresses(
                        entries[:got], 0, require_complete=True
                    )
                    uniform_bytes = [0]
                    for i in range(got):
                        uniform_bytes.append(
                            uniform_bytes[-1]
                            + sum(
                                n
                                for name in on_swa_names
                                for _, n in uniform_pools[name][i]
                            )
                        )
                    with self._lease_mutex:
                        budget = self._max_leased_bytes - self._leased_bytes_total
                    endpoint = got
                    while (
                        endpoint
                        and prefix_bytes[endpoint] + uniform_bytes[endpoint] > budget
                    ):
                        endpoint -= 1
                    if not endpoint:
                        refused = "budget"
                        return reply()
                    self._unpin([], pinned_swa[endpoint:], cool_s=0.0)
                    pinned_swa = pinned_swa[:endpoint]
                    candidates = (endpoint,)
                for endpoint in candidates:
                    start = (
                        (
                            0
                            if full_swa_prefix
                            else max(0, endpoint - self._swa_trailing_pages)
                        )
                        if on_swa_names
                        else endpoint
                    )
                    tail = entries[start:endpoint]
                    if any(e.swa_indices is None or e.swa_gen is None for e in tail):
                        continue
                    if tail and uniform_pools is None:
                        got = self._pin_entries(
                            self._swa_anchor,
                            [e.swa_indices for e in tail],
                            [e.swa_gen for e in tail],
                        )
                        pinned_swa = [
                            (start + i, e.swa_indices) for i, e in enumerate(tail[:got])
                        ]
                        if got != len(tail):
                            self._unpin([], pinned_swa, cool_s=0.0)
                            pinned_swa = []
                            continue
                    if uniform_pools is not None:
                        tail_pools = {
                            name: pages[:endpoint]
                            for name, pages in uniform_pools.items()
                        }
                        tail_bytes = uniform_bytes[endpoint]
                    else:
                        tail_pools, _ = self._build_pool_addresses(
                            tail, 0, require_complete=True
                        )
                        tail_bytes = sum(
                            n
                            for name in on_swa_names
                            for segments in tail_pools[name]
                            for _, n in segments
                        )
                    total_bytes = prefix_bytes[endpoint] + tail_bytes
                    with self._lease_mutex:
                        over_budget = (
                            self._leased_bytes_total + total_bytes
                            > self._max_leased_bytes
                        )
                    if over_budget:
                        refused = "budget"
                        self._unpin([], pinned_swa, cool_s=0.0)
                        pinned_swa = []
                        continue
                    if self._past_deadline(msg):
                        # F5: as in _handle_lookup; the finally block gives
                        # the pins back.
                        self._late_at_grant += 1
                        refused = "late"
                        return reply()

                    pools = {
                        name: (
                            ([None] * start + tail_pools[name])
                            if name in on_swa_names
                            else pages[:endpoint]
                        )
                        for name, pages in full_pools.items()
                    }
                    self._unpin(pinned_kv[endpoint:], [], cool_s=0.0)
                    pinned_kv = pinned_kv[:endpoint]
                    lease_id = uuid.uuid4().hex
                    with self._lease_mutex:
                        self._leases[lease_id] = _Lease(
                            tuple(hashes[:endpoint]),
                            time.monotonic() + ttl_ms / 1e3,
                            total_bytes,
                            pinned_kv,
                            pinned_swa,
                            trace_metadata={**trace_metadata, "exact_endpoint": True},
                            lookup_tick=lookup_tick,
                        )
                        self._leased_bytes_total += total_bytes
                    # Ownership of all pins transfers to the lease here.
                    pinned_kv, pinned_swa = [], []
                    self._drain_deltas()
                    unchanged = all(
                        self._shadow.get(hashes[i]) is source_entries[i]
                        and (
                            i < start
                            or source_entries[i].swa_indices is entries[i].swa_indices
                        )
                        for i in range(endpoint)
                    )
                    if unchanged:
                        result = reply(endpoint, lease_id, pools, endpoint - start)
                        lease_id = ""  # Successful grant now belongs to reader.
                        return result
                    # Precision retraction must not leave a new endpoint with
                    # the old endpoint's tail. Release everything before retry,
                    # including reserve held by the rejected FULL suffix.
                    self._release_lease(lease_id, reason="release")
                    lease_id = ""
                    refused = "endpoint_changed"
                    break
                else:
                    return reply()
            finally:
                if lease_id:
                    self._release_lease(lease_id, reason="release")
                self._unpin(pinned_kv, pinned_swa, cool_s=0.0)
        return reply()

    def _serves_swa_ranges(self) -> bool:
        """LOOKUP_SWA_RANGE needs an SWA host pool that holds every published
        page's SWA (``swa_host_complete``)."""
        return self._swa_anchor is not None and self._swa_host_complete

    def _handle_swa_range_lookup(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """LOOKUP of SWA-anchored pages only (DSv4 window residency backfill).

        A reader that received only the trailing SWA of a prefix (PD decode,
        symmetric import) pulls the SWA of pages [a, b) host->host from the
        owner. Chain hashes identify pages by position, so a range that does
        not start at page 0 is well defined. Unlike a complete LOOKUP this
        pins no KV-anchor page and grants the maximal contiguous prefix of the
        range whose SWA slices are published with a matching generation, cut
        to what the lease byte budget allows. The reply carries the
        SWA-anchored pools only (SWA itself, c4_state, c4_indexer_state, ...).
        """
        self._touch_note_lookup(msg)
        t0 = time.perf_counter()
        hashes = list(msg["hashes"])
        trace_metadata = _request_trace_metadata(msg)
        ttl_ms = min(
            int(msg.get("lease_ttl_ms", _LEASE_TTL_DEFAULT_MS)), _LEASE_TTL_MAX_MS
        )
        on_swa = [
            (name, pool, has_bytes)
            for name, pool, anchored_on_swa, has_bytes in self._pool_plan
            if anchored_on_swa
        ]
        refused = "no_pages"
        partial = 0

        def reply(num_pages=0, lease_id="", pools=None):
            result = self._lookup_reply(
                t0,
                hashes,
                num_pages,
                0,
                lease_id,
                None if num_pages else refused,
                num_pages_partial=partial,
                swa_pages_pinned=num_pages,
                trace_metadata=trace_metadata,
                op="lookup_swa_range",
            )
            result.update(
                swa_range=True,
                lease_ttl_ms=ttl_ms,
                page_size=self._page_size,
                refused_reason=None if num_pages else refused,
            )
            if pools is not None:
                result["pools"] = pools
                if self._verify_checksums:
                    result["sums"] = self._checksums(pools)
                if self._poison_installed:
                    result["poison"] = True
            return result

        if not on_swa or not self._serves_swa_ranges():
            # No SWA pool, or host SWA only for trailing windows (legacy
            # lifecycle): permanent, the reader turns backfill off.
            refused = "unsupported"
            return reply()

        for _ in range(_EXACT_LOOKUP_MAX_RETRIES):
            self._drain_deltas()
            source_entries = []
            for h in hashes:
                entry = self._shadow.get(h)
                if entry is None or entry.swa_indices is None or entry.swa_gen is None:
                    break
                source_entries.append(entry)
            if not source_entries:
                refused = "no_pages"
                return reply()
            # REMOVE_SWA rewrites shadow entries in place: pin a snapshot.
            entries = [
                _PageEntry(e.kv_indices, e.swa_indices, e.kv_gen, e.swa_gen)
                for e in source_entries
            ]
            pinned: List[Tuple[int, Any]] = []
            lease_id = ""
            try:
                got = self._pin_entries(
                    self._swa_anchor,
                    [e.swa_indices for e in entries],
                    [e.swa_gen for e in entries],
                )
                pinned = [(i, e.swa_indices) for i, e in enumerate(entries[:got])]
                partial = len(entries) - got
                if not got:
                    slots = self._swa_anchor.slot_allocator
                    with self._swa_anchor.lock:
                        exhausted = slots.reserve_pages - slots.num_protected <= 0
                    refused = "reserve" if exhausted else "pin"
                    return reply()
                pools, page_nbytes = self._build_swa_range_addresses(
                    on_swa, entries[:got]
                )
                with self._lease_mutex:
                    budget = self._max_leased_bytes - self._leased_bytes_total
                count = 0
                total_bytes = 0
                while count < got and total_bytes + page_nbytes[count] <= budget:
                    total_bytes += page_nbytes[count]
                    count += 1
                if not count:
                    refused = "budget"
                    return reply()
                self._unpin([], pinned[count:], cool_s=0.0)
                pinned = pinned[:count]
                page_nbytes = page_nbytes[:count]
                lease_id = uuid.uuid4().hex
                with self._lease_mutex:
                    self._leases[lease_id] = _Lease(
                        tuple(hashes[:count]),
                        time.monotonic() + ttl_ms / 1e3,
                        total_bytes,
                        [],
                        pinned,
                        trace_metadata={**trace_metadata, "exact_endpoint": False},
                    )
                    self._leased_bytes_total += total_bytes
                # The lease owns the pins from here on.
                pinned = []
                # Precision retraction: a delta that raced the pin keeps the
                # pinned bytes intact, but the owner no longer advertises the
                # page. Range grants are prefix-monotone, so cut to the
                # unchanged prefix.
                self._drain_deltas()
                keep = count
                for i in range(count):
                    if (
                        self._shadow.get(hashes[i]) is not source_entries[i]
                        or source_entries[i].swa_indices is not entries[i].swa_indices
                    ):
                        keep = i
                        break
                if keep < count:
                    self._shrink_lease(lease_id, keep, page_nbytes)
                    if not keep:
                        lease_id = ""
                        refused = "endpoint_changed"
                        continue
                    count = keep
                result = reply(
                    count,
                    lease_id,
                    {name: pages[:count] for name, pages in pools.items()},
                )
                lease_id = ""  # The grant now belongs to the reader.
                return result
            finally:
                if lease_id:
                    self._release_lease(lease_id, reason="release")
                self._unpin([], pinned, cool_s=0.0)
        return reply()

    def _build_swa_range_addresses(
        self,
        on_swa: List[Tuple[str, Any, bool]],
        entries: List[_PageEntry],
    ) -> Tuple[Dict[str, List[List[List[int]]]], List[int]]:
        """Per SWA-anchored pool, per page [(ptr, nbytes), ...] segments, and
        the SWA-anchored bytes of each page (the lease budget counts only
        these). Logical pools attest pages with empty segments."""
        pools: Dict[str, List[List[List[int]]]] = {name: [] for name, _, _ in on_swa}
        page_nbytes: List[int] = []
        for entry in entries:
            nbytes = 0
            for name, pool, has_bytes in on_swa:
                if not has_bytes:
                    pools[name].append([])
                    continue
                ptrs, lens = pool.get_page_buffer_meta(entry.swa_indices)
                if (
                    not len(ptrs)
                    or len(ptrs) != len(lens)
                    or any(int(p) <= 0 or int(n) <= 0 for p, n in zip(ptrs, lens))
                ):
                    raise ValueError(
                        f"PeerKVServicer: invalid byte segments for {name}"
                    )
                pools[name].append([[int(p), int(n)] for p, n in zip(ptrs, lens)])
                nbytes += sum(int(n) for n in lens)
            page_nbytes.append(nbytes)
        return pools, page_nbytes

    @staticmethod
    def _pin_entries(
        pool: Any, idx_tensors: List[Any], gens_lists: List[List[int]]
    ) -> int:
        """Pin whole entry pages (each may span several allocator pages) as a
        prefix; a partially pinned trailing entry is unpinned again. Returns
        the number of entry pages fully pinned."""
        slots = pool.slot_allocator
        if not idx_tensors:
            return 0
        flat_gens: List[int] = []
        per_entry: List[int] = []
        for t, g in zip(idx_tensors, gens_lists):
            n = t.numel() // slots.granule
            if len(g) != n:
                raise ValueError(
                    f"PeerKVServicer: {n} allocator pages but {len(g)} generations"
                )
            flat_gens.extend(g)
            per_entry.append(n)
        with pool.lock:
            k = slots.try_pin(torch.cat(idx_tensors), flat_gens)
            full = 0
            acc = 0
            for n in per_entry:
                if acc + n <= k:
                    acc += n
                    full += 1
                else:
                    break
            if acc < k:
                # Partial trailing entry: release the extra allocator pages.
                slots.unpin(
                    torch.cat(idx_tensors)[acc * slots.granule : k * slots.granule], 0.0
                )
        return full

    def _lookup_reply(
        self,
        t0: float,
        hashes: List[str],
        num_pages: int,
        retracted: int,
        lease_id: str,
        refused: Optional[str],
        *,
        num_pages_partial: int = 0,
        swa_pages_pinned: int = 0,
        trace_metadata: Optional[dict] = None,
        exact_endpoint: bool = False,
        full_shadow_pages: Optional[int] = None,
        full_pinned_pages: Optional[int] = None,
        op: str = "lookup",
        confirmed: Optional[int] = None,
        lookup_tick: Optional[float] = None,
    ) -> Dict[str, Any]:
        """``confirmed``: the owner's pages on the chain (shadow-confirmed,
        before pins and budget) and ``lookup_tick``: with NOTICE or COUNTERS
        on the reply carries them and refused_reason, and the
        trace logs the true confirmed (owner_confirmed)."""
        fields = confirmed is not None and getattr(self, "_notice_fields", False)
        with self._lease_mutex:
            leased_bytes = self._leased_bytes_total
        self._trace(
            (PeerLookupTraceRecordV4 if fields else PeerLookupTraceRecord)(
                op=op,
                ts=time.time(),
                dur_us=(time.perf_counter() - t0) * 1e6,
                num_hashes=len(hashes),
                num_pages_confirmed=num_pages,
                num_pages_retracted=retracted,
                lease_id=lease_id,
                leased_bytes_total=leased_bytes,
                refused_reason=refused,
                num_pages_partial=max(0, num_pages_partial),
                swa_pages_pinned=swa_pages_pinned,
                exact_endpoint=exact_endpoint,
                full_shadow_pages=full_shadow_pages,
                full_pinned_pages=full_pinned_pages,
                **({"owner_confirmed": int(confirmed)} if fields else {}),
                **(trace_metadata or {}),
            )
        )
        reply = {
            "ok": True,
            "lease_id": lease_id,
            "num_pages": num_pages,
            "te_endpoint": self._te_endpoint,
            "pools": {},
            "gpu_export": bool(getattr(self, "gpu_export_enabled", False)),
            "gpu_export_pin_capacity_pages": (
                min(
                    pool.slot_allocator.reserve_slots // self._page_size
                    for pool in (self._kv_anchor, self._swa_anchor)
                    if pool is not None
                )
                if getattr(self, "gpu_export_enabled", False)
                else None
            ),
        }
        if fields:
            reply["confirmed"] = int(confirmed)
            reply["refused_reason"] = refused
            # The tick counter is a numpy float: msgpack needs a Python one.
            reply["lookup_tick"] = None if lookup_tick is None else float(lookup_tick)
        return reply

    def _build_pool_addresses(
        self,
        entries: List[_PageEntry],
        swa_ok_from: int,
        *,
        require_complete: bool = False,
    ) -> Tuple[Dict[str, List[Optional[List[List[int]]]]], List[int]]:
        """Per-pool, per-page [(ptr, nbytes), ...] segments + per-page bytes.
        SWA-anchored pools get None for pages before ``swa_ok_from`` (not
        pinned / not held)."""
        pools: Dict[str, List[Optional[List[List[int]]]]] = {
            name: [] for name, _, _, _ in self._pool_plan
        }
        page_nbytes: List[int] = []
        for i, entry in enumerate(entries):
            nbytes = 0
            for name, pool, on_swa, has_bytes in self._pool_plan:
                if on_swa:
                    indices = entry.swa_indices if i >= swa_ok_from else None
                else:
                    indices = entry.kv_indices
                if indices is None:
                    pools[name].append(None)
                    continue
                if not has_bytes:
                    # Logical anchor: page present, zero bytes ([] != None —
                    # None means "this pool has no such page").
                    pools[name].append([])
                    continue
                ptrs, lens = pool.get_page_buffer_meta(indices)
                if require_complete and (
                    not len(ptrs)
                    or len(ptrs) != len(lens)
                    or any(int(p) <= 0 or int(n) <= 0 for p, n in zip(ptrs, lens))
                ):
                    raise ValueError(
                        f"PeerKVServicer: invalid byte segments for {name}"
                    )
                segments = [[int(p), int(l)] for p, l in zip(ptrs, lens)]
                pools[name].append(segments)
                nbytes += sum(lens)
            page_nbytes.append(nbytes)
        return pools, page_nbytes

    @staticmethod
    def _checksums(
        pools: Dict[str, List[Optional[List[List[int]]]]],
    ) -> Dict[str, list]:
        """crc32 over each page's segments (pinned pages are immutable for
        the lease lifetime, so the reader can compare after its RDMA read)."""
        sums: Dict[str, list] = {}
        for name, pages in pools.items():
            out: list = []
            for segs in pages:
                if segs is None or len(segs) == 0:
                    out.append(None)
                    continue
                crc = 0
                for ptr, n in segs:
                    crc = zlib.crc32(ctypes.string_at(ptr, n), crc)
                out.append(crc & 0xFFFFFFFF)
            sums[name] = out
        return sums

    # ------------------------------------------------------------------
    # Lease internals (servicer thread)
    # ------------------------------------------------------------------

    def _unpin(
        self, kv_idx: List[Any], swa_idx: List[Tuple[int, Any]], cool_s: float
    ) -> None:
        try:
            if kv_idx:
                with self._kv_anchor.lock:
                    self._kv_anchor.slot_allocator.unpin(torch.cat(kv_idx), cool_s)
            if swa_idx and self._swa_anchor is not None:
                with self._swa_anchor.lock:
                    self._swa_anchor.slot_allocator.unpin(
                        torch.cat([t for _, t in swa_idx]), cool_s
                    )
        except Exception:
            self._pin_leaks += 1
            logger.exception("PeerKVServicer: unpin failed (pin leak)")

    def _shrink_lease(self, lease_id: str, keep: int, page_nbytes: List[int]) -> None:
        with self._lease_mutex:
            lease = self._leases.get(lease_id)
            if lease is None:
                return
            freed = sum(page_nbytes[keep:])
            lease.nbytes -= freed
            self._leased_bytes_total -= freed
            drop_kv = lease.kv_idx[keep:]
            lease.kv_idx = lease.kv_idx[:keep]
            drop_swa = [(i, t) for i, t in lease.swa_idx if i >= keep]
            lease.swa_idx = [(i, t) for i, t in lease.swa_idx if i < keep]
            lease.hashes = lease.hashes[:keep]
            if keep == 0:
                del self._leases[lease_id]
        self._unpin(drop_kv, drop_swa, cool_s=0.0)

    def _release_lease(
        self,
        lease_id: str,
        reason: str,
        complete: bool = False,
        pages: Optional[int] = None,
        after_commit: bool = False,
    ) -> None:
        with self._lease_mutex:
            lease = self._leases.pop(lease_id, None)
            if lease is None:
                return
            self._leased_bytes_total -= lease.nbytes
            leased_bytes = self._leased_bytes_total
        # RELEASE is sent strictly after the reader's reads completed: no cooling.
        self._unpin(lease.kv_idx, lease.swa_idx, cool_s=0.0)
        if (
            (complete or after_commit)
            and self._demote_outbox is not None
            and lease.lookup_tick is not None
        ):
            # Only the pages the reader read (it may have trimmed the lease).
            n = len(lease.hashes) if pages is None else min(pages, len(lease.hashes))
            if n > 0 and complete and self._demote_ignores_marks:
                self.demote_ignored_role += 1  # a D owner
            elif n > 0 and complete:
                self._demote_put(
                    list(lease.hashes[:n]),
                    lease.lookup_tick,
                    lease.trace_metadata.get("req_id", ""),
                    "release",
                )
            elif n > 0:
                # SGLANG_PEER_KV_DEMOTE_AFTER_COMMIT (reader): demote only
                # after the reader committed the pages (DEMOTE_OK).
                self._demote_pending_put(lease_id, lease, n)
        self._trace(
            PeerLookupTraceRecord(
                op=reason,
                ts=time.time(),
                dur_us=0.0,
                num_hashes=len(lease.hashes),
                num_pages_confirmed=0,
                num_pages_retracted=0,
                lease_id=lease_id,
                leased_bytes_total=leased_bytes,
                complete=True if complete else None,
                **lease.trace_metadata,
            )
        )

    def _demote_put(self, keys: List[str], tick: float, rid: str, trigger: str) -> None:
        """Queue one demote for the scheduler thread (the outbox is bounded:
        a full outbox drops its oldest entry, counted). With COUNTERS the
        entry carries the request id and the trigger (release / commit)."""
        outbox = self._demote_outbox
        if len(outbox) >= outbox.maxlen:
            self.demote_outbox_drops += 1
        outbox.append(
            DemoteOutboxEntry((keys, tick, rid), trigger)
            if self._demote_counters or trigger == "notice"
            else (keys, tick)
        )

    # ------------------------------------------------------------------
    # NOTICE
    # ------------------------------------------------------------------

    NOTICE_COUNTERS = (
        "received",
        "queued",
        "unknown",
        "dropped_no_age",
        "dropped_age",
        "dropped_bad",
        "held",
        "held_total",
        "held_dropped",
    )
    _NOTICE_HELD_MAX = 4096

    def _init_notice(self) -> None:
        """SGLANG_PEER_KV_DEMOTE_NOTICE, owner side. The op acts only on the
        demote source (the rank with the outbox) with NOTICE on; the LOOKUP
        reply fields also come with COUNTERS. Servicer-thread counters, read
        by the scheduler thread for logs."""
        self._notice_on = bool(envs.SGLANG_PEER_KV_DEMOTE_NOTICE.get())
        self._notice_fields = self._notice_on or bool(
            getattr(self, "_demote_counters", False)
        )
        self._notice_grace_s = max(
            0.0, float(envs.SGLANG_PEER_KV_DEMOTE_NOTICE_GRACE_S.get())
        )
        # The grace hold list [(due, keys, lookup tick, rid)]: the servicer
        # thread appends, the scheduler thread drains (drain_demote_outbox)
        # and clears it at a flush (push_clear).
        self._notice_lock = threading.Lock()
        self._notice_held: List[tuple] = []
        self.notice_received = 0
        self.notice_queued = 0
        self.notice_unknown = 0
        self.notice_dropped_no_age = 0
        self.notice_dropped_age = 0
        self.notice_dropped_bad = 0
        self.notice_held_total = 0
        self.notice_held_dropped = 0

    @property
    def notice_held(self) -> int:
        """Held NOTICEs now (gauge)."""
        held = getattr(self, "_notice_held", None)
        return len(held) if held else 0

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def _handle_notice(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """NOTICE {rid, keys, lookup_tick, age_ms} from a reader's TP0: the
        reader finished its prefill of a chain that this owner confirmed but
        the reader did not commit from here; it now holds the newer copy, so
        this copy changes class as at a commit. The walk starts at the
        deepest node end among the keys (select_peer_demote_nodes). With the
        grace, hold it until G s after the reader's prefill end."""
        self.notice_received += 1
        if self._demote_outbox is None or not getattr(self, "_notice_on", False):
            # No switch, a rank that is not the demote source, or NOTICE off.
            self.notice_unknown += 1
            return {"ok": False, "error": "unknown op 'notice'"}
        age_ms = msg.get("age_ms")
        if isinstance(age_ms, bool) or not isinstance(age_ms, (int, float)):
            self.notice_dropped_no_age += 1
            return {"ok": True, "dropped": "no_age"}
        if age_ms > NOTICE_MAX_AGE_S * 1000.0:
            self.notice_dropped_age += 1
            return {"ok": True, "dropped": "age"}
        keys = msg.get("keys")
        tick = msg.get("lookup_tick")
        if (
            not isinstance(keys, list)
            or not keys
            or len(keys) > NOTICE_MAX_KEYS
            or not all(isinstance(k, str) for k in keys)
            or isinstance(tick, bool)
            or not isinstance(tick, (int, float))
        ):
            self.notice_dropped_bad += 1
            return {"ok": True, "dropped": "bad"}
        if getattr(self, "_demote_ignores_marks", False):
            self.demote_ignored_role += 1  # a D owner
            return {"ok": True, "dropped": "role"}
        rid = msg.get("rid")
        rid = rid if isinstance(rid, str) else ""
        if self._notice_grace_s > 0:
            due = self._now() + max(0.0, self._notice_grace_s - age_ms / 1000.0)
            with self._notice_lock:
                if len(self._notice_held) >= self._NOTICE_HELD_MAX:
                    del self._notice_held[0]
                    self.notice_held_dropped += 1
                self._notice_held.append((due, keys, float(tick), rid))
            self.notice_held_total += 1
            return {"ok": True, "held": True}
        self._demote_put(keys, float(tick), rid, "notice")
        self.notice_queued += 1
        return {"ok": True, "queued": True}

    def _demote_pending_put(self, lease_id: str, lease: Any, n: int) -> None:
        now = time.monotonic()
        pending = self._demote_pending
        while pending:
            oldest, entry = next(iter(pending.items()))
            if entry[3] > now and len(pending) < self._DEMOTE_PENDING_MAX:
                break
            del pending[oldest]
            self.demote_commit_expired += 1
        pending[lease_id] = (
            list(lease.hashes[:n]),
            lease.lookup_tick,
            lease.trace_metadata.get("req_id", ""),
            now + self._demote_pending_ttl_s,
        )
        self.demote_commit_pending += 1

    def _handle_demote_ok(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """DEMOTE_OK (SGLANG_PEER_KV_DEMOTE_AFTER_COMMIT): the reader committed
        the pages of a lease that it read in full; demote them now, as a
        complete RELEASE does. ``pages``: the committed pages (at most the
        read ones)."""
        lease_id = msg.get("lease_id")
        entry = (
            self._demote_pending.pop(lease_id, None)
            if isinstance(lease_id, str)
            else None
        )
        if entry is None:
            self.demote_commit_unknown += 1
            return {"ok": True, "demoted": False}
        keys, tick, rid, expiry = entry
        now = time.monotonic()
        if expiry <= now:
            self.demote_commit_expired += 1
            return {"ok": True, "demoted": False}
        pages = msg.get("pages")
        n = min(pages, len(keys)) if type(pages) is int and pages >= 0 else len(keys)
        if n <= 0:
            return {"ok": True, "demoted": False}
        if getattr(self, "_demote_ignores_marks", False):
            self.demote_ignored_role += 1  # a D owner
            return {"ok": True, "demoted": False, "ignored": "role"}
        self._demote_put(keys[:n], tick, rid, "commit")
        self.demote_commit_ok += 1
        if self._demote_counters:
            lag = max(0.0, now - (expiry - self._demote_pending_ttl_s))
            self.demote_commit_lag[lag_bucket(lag)] += 1
            self.demote_commit_lag_max_ms = max(
                self.demote_commit_lag_max_ms, int(lag * 1000)
            )
        return {"ok": True, "demoted": True}

    def _expire_leases(self, now_mono: float) -> None:
        with self._lease_mutex:
            expired = [
                (lease_id, lease)
                for lease_id, lease in self._leases.items()
                if lease.expiry_mono <= now_mono
            ]
            for lease_id, lease in expired:
                del self._leases[lease_id]
                self._leased_bytes_total -= lease.nbytes
            leased_bytes = self._leased_bytes_total
        for lease_id, lease in expired:
            # Expiry (no RELEASE): a read that started before the TTL ran out
            # may still be in flight — keep the pages out of the free list for
            # one cooling window (>= the transfer engine slice timeout).
            self._unpin(lease.kv_idx, lease.swa_idx, cool_s=self._expiry_cooling_s)
            self._trace(
                PeerLookupTraceRecord(
                    op="expire",
                    ts=time.time(),
                    dur_us=0.0,
                    num_hashes=len(lease.hashes),
                    num_pages_confirmed=0,
                    num_pages_retracted=0,
                    lease_id=lease_id,
                    leased_bytes_total=leased_bytes,
                    **lease.trace_metadata,
                )
            )

    def _drain_deltas(self) -> set:
        """Apply pending deltas to the shadow index; return removed hashes."""
        removed: set = set()
        while True:
            try:
                kind, payload = self._deltas.get_nowait()
            except Empty:
                return removed
            if kind >= _DELTA_STORE_RANGE:
                kind, payload = self._expand_range_delta(kind, payload)
            if kind == _DELTA_STORE:
                hashes, kv_pages, swa_pages, kv_gens, swa_gens = payload
                for i, h in enumerate(hashes):
                    self._shadow[h] = _PageEntry(
                        kv_pages[i], swa_pages[i], kv_gens[i], swa_gens[i]
                    )
            elif kind == _DELTA_STORE_SWA:
                hashes, swa_pages, swa_gens = payload
                for i, h in enumerate(hashes):
                    entry = self._shadow.get(h)
                    if entry is not None and entry.kv_indices is not None:
                        continue  # the FULL-backed entry wins
                    self._shadow[h] = _PageEntry(None, swa_pages[i], None, swa_gens[i])
            elif kind == _DELTA_REMOVE:
                hashes, pages, generations = payload
                for h, indices, generation in zip(hashes, pages, generations):
                    entry = self._shadow.get(h)
                    if (
                        entry is not None
                        and indices is not None
                        and entry.kv_indices is not None
                        and entry.kv_gen == generation
                        and torch.equal(entry.kv_indices, indices)
                    ):
                        del self._shadow[h]
                        # Stale removals must not retract a newer pinned entry.
                        removed.add(h)
            elif kind == _DELTA_REMOVE_SWA:
                hashes, pages, generations = payload
                for h, indices, generation in zip(hashes, pages, generations):
                    entry = self._shadow.get(h)
                    if (
                        entry is not None
                        and entry.swa_indices is not None
                        and indices is not None
                        and entry.swa_gen == generation
                        and torch.equal(entry.swa_indices, indices)
                    ):
                        entry.swa_indices = None
                        entry.swa_gen = None
                        if entry.kv_indices is None:
                            del self._shadow[h]  # an SWA-only entry is now empty
            elif kind == _DELTA_CLEAR:
                removed.update(self._shadow.keys())
                self._shadow.clear()

    def _trace(self, record: PeerLookupTraceRecord) -> None:
        if self._trace_file is not None:
            self._trace_file.write(_trace_encoder.encode(record) + b"\n")
