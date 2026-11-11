"""Host->host SWA backfill for DSv4 window SWA residency.

A window worker that received only the trailing SWA of a prefix (a PD decode
worker, a symmetric import) holds that prefix's FULL KV but no SWA for
positions [start, end). Before its finished request can be inserted with a
complete host copy (invariant I1), the SWA-anchored pools of those pages are
pulled host->host from the owner's PeerKVServicer (LOOKUP_SWA_RANGE) into
request-owned host slots and committed as an acked ``SwaSegment`` of the
request's ``SwaHostLedger``. Anything not committed in time is a coverage gap,
and the insert is truncated there.

Rank rules:

* ``submit``, ``cancel`` and ``poll`` run on the scheduler thread at lockstep
  points, so every TP rank holds the same job list in the same order.
* Network IO (capability probe, lookup, RDMA read, release) runs on a worker
  thread; ``poll`` never blocks on it.
* Every decision that changes host or ledger state is taken from one MIN
  all-reduce per ``poll`` over each job's IO status, grant, deadline and
  backoff flags. Host slots are allocated only for the MIN grant. Whether
  that allocation succeeded is rank-local (reclaim, a skewed host pool), so
  it is reduced too: a rank whose allocation failed skips its read and
  reports ``alloc_ok=0`` at the next poll, and every rank then frees its
  slots and backs off. Rank-local clocks only feed the reduced flags.
* The collective's shape depends on the job list. ``lockstep_digest`` folds
  the list (request ids, ranges, phases) into one value: it prefixes every
  poll vector, and ``UnifiedRadixCache._sync_hicache_ready_counts`` checks it
  every scheduler step (which catches a rank-local submit before the job
  counts can differ in ``poll``). A mismatch raises instead of hanging or
  reducing misaligned rows.
* Under host dedup only the source rank performs IO; the allocator-only
  followers report the upper bound, so the MIN is the source's result.

Lease freshness. The job deadline (``--dsv4-swa-backfill-timeout-s``, 60 s by
default) may exceed the owner's lease TTL (30 s) plus its expiry cooling
(12 s): each round takes a fresh grant, and a grant is used only while it is
young. A rank reports a grant older than half its TTL as 0 pages, so the MIN
refuses it before any host slot is allocated, and ``read_swa_range`` refuses
to start (or to accept) a read near or past the lease expiry. A read of pages
the owner already reused can therefore not be committed.

Host slots whose read may still be running are freed only once the read has
finished on every rank (a cancelled job becomes an orphan until then).
"""

from __future__ import annotations

import logging
import os
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

import torch

from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.mem_cache.hicache_storage import PoolName
from decoupled_engine.srt.mem_cache.peer_kv_namespace import peer_page_keys
from decoupled_engine.srt.mem_cache.unified_cache.components.tree_component import (
    ComponentType,
)
from decoupled_engine.srt.mem_cache.unified_cache.swa_ledger import SwaHostLedger, SwaSegment

logger = logging.getLogger(__name__)

# Job phases.
QUEUED = "queued"  # waiting for the rate limit (or ready to retry)
LOOKUP = "lookup"  # LOOKUP_SWA_RANGE in flight
READ = "read"  # host slots allocated, RDMA read in flight
BACKOFF = "backoff"  # a round got nothing; retry after a delay
DONE = "done"

# Final results (the ``result`` label of dsv4_swa_backfill_pages_total).
RESULT_OK = "ok"
RESULT_PARTIAL = "partial"
RESULT_TIMEOUT = "timeout"
RESULT_NO_CAPABILITY = "no_capability"
RESULT_CANCELLED = "cancelled"
# Metric labels for the pages a timed-out job left missing, by the cause of
# its last failed round (the job itself ends timeout or partial).
LABEL_REFUSED = "refused"
LABEL_OWNER_EVICTED = "owner_evicted"
# Owner-side refusals of LOOKUP_SWA_RANGE (peer_kv_servicer). "unsupported"
# is permanent and ends the job as no_capability instead.
_OWNER_REFUSALS = ("no_pages", "reserve", "pin", "budget", "endpoint_changed")
# The owner had granted pages of this job before and no longer has them.
_EVICTION_REFUSALS = ("no_pages", "endpoint_changed")

_BACKOFF_INITIAL_S = 0.02
_BACKOFF_MAX_S = 1.0
# A grant is used only while younger than this fraction of its lease TTL.
_GRANT_FRESH_FRACTION = 0.5
# Per-job fields of the per-poll MIN all-reduce.
_IO_DONE, _CAPABLE, _VALUE, _NOT_EXPIRED, _READY, _ALLOC_OK = range(6)
_NUM_FIELDS = 6
# The poll vector starts with [digest, -digest] of the job list.
_HEADER_FIELDS = 2
_DIGEST_MASK = (1 << 42) - 1
_PHASE_CODES = {QUEUED: 1, LOOKUP: 2, READ: 3, BACKOFF: 4, DONE: 5}


def swa_backfill_page_keys(radix_key: Any, num_pages: int, page_size: int) -> list[str]:
    """Peer page keys of the first ``num_pages`` pages of ``radix_key``: the
    tree's chain hashes (``get_hash_str`` from the root, as a node's
    ``hash_value``) wrapped in the key's (cache_salt, extra_key) namespace,
    exactly as the owner publishes them (UnifiedTreeCore._peer_push_servable_node).
    A backfill of [start, end) uses keys[start // page_size : end // page_size]."""
    from decoupled_engine.srt.mem_cache.utils import get_hash_str

    if num_pages <= 0:
        return []
    hashes = get_hash_str(radix_key[: num_pages * page_size], None, page_size=page_size)
    return peer_page_keys(
        hashes,
        getattr(radix_key, "cache_salt", None),
        getattr(radix_key, "extra_key", None),
    )


def swa_anchored_pool_names(host_pool_group: Any) -> list:
    """Host pools addressed by SWA-pool indices (SWA itself first): the pools
    a backfill reads and the servicer's range grant carries."""
    if host_pool_group is None or PoolName.SWA not in host_pool_group.entry_map:
        return []
    names = [PoolName.SWA]
    for entry in host_pool_group.entries:
        if entry.name == PoolName.SWA:
            continue
        if host_pool_group.anchor_of(entry.name).name == PoolName.SWA:
            names.append(entry.name)
    return names


def swa_anchored_bytes_per_page(host_pool_group: Any, pool_names: Sequence) -> int:
    """Host bytes one page of the SWA-anchored pools holds. It comes from pool
    geometry (layers x bytes per page row), which dedup followers share, so it
    is identical on every rank."""
    total = 0
    for name in pool_names:
        pool = host_pool_group.get_pool(name)
        total += int(getattr(pool, "layer_num", 0) or 0) * int(
            getattr(pool, "size_per_token", 0) or 0
        )
    return total


@dataclass
class _LookupResult:
    capable: bool
    pages: int
    grant: Optional[dict] = None
    refused_reason: Optional[str] = None
    # Manager-clock time the LOOKUP RPC started and the grant's lease TTL;
    # None (dedup followers, no grant) skips the freshness check.
    started_at: Optional[float] = None
    ttl_s: Optional[float] = None


class _Done:
    """A completed future (dedup followers, no-backend jobs)."""

    def __init__(self, value: Any):
        self._value = value

    def done(self) -> bool:
        return True

    def result(self) -> Any:
        return self._value


def _set_current_thread_nice(nice: int, sched_idle: bool = False) -> None:
    """Lower this thread's CPU priority (Linux: nice and policy are per
    thread): nice value ``nice``, SCHED_IDLE when ``sched_idle``."""
    tid = threading.get_native_id()
    if nice:
        try:
            os.setpriority(os.PRIO_PROCESS, tid, nice)
        except OSError:  # not permitted (e.g. lowering below the current value)
            logger.warning("SWA backfill IO thread: cannot set nice %d", nice)
    if sched_idle and hasattr(os, "SCHED_IDLE"):
        try:
            os.sched_setscheduler(tid, os.SCHED_IDLE, os.sched_param(0))
        except OSError:
            logger.warning("SWA backfill IO thread: cannot set SCHED_IDLE")


class ThreadedBackfillIO:
    """Runs backfill network IO on one worker thread, in FIFO order.

    The thread is a ThreadPoolExecutor worker (not a daemon: interpreter exit
    joins it), so ``shutdown`` cancels the queued calls; a call already
    running is bounded by its RPC or transfer timeout. The executor is
    recreated on the next ``submit`` after a shutdown (cache reset).

    The RDMA read busy-polls a CPU until it completes, and every TP rank
    reads at the same time; ``nice`` (SGLANG_DSV4_SWA_BACKFILL_IO_NICE) and
    ``sched_idle`` (SGLANG_BUSY_POLL_SCHED_IDLE) let the scheduler threads
    preempt it."""

    def __init__(
        self, nice: Optional[int] = None, sched_idle: Optional[bool] = None
    ) -> None:
        self._pool: Optional[ThreadPoolExecutor] = None
        self._nice = (
            envs.SGLANG_DSV4_SWA_BACKFILL_IO_NICE.get() if nice is None else int(nice)
        )
        self._sched_idle = (
            envs.SGLANG_BUSY_POLL_SCHED_IDLE.get()
            if sched_idle is None
            else bool(sched_idle)
        )

    def submit(self, kind: str, fn: Callable, *args: Any):
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="swa-backfill-io",
                initializer=_set_current_thread_nice,
                initargs=(self._nice, self._sched_idle),
            )
        return self._pool.submit(fn, *args)

    def shutdown(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


@dataclass(eq=False)
class _Job:
    req: Any
    ledger: SwaHostLedger
    start: int
    end: int
    # None: this rank found no usable owner (``local_reason`` says why). The
    # job still exists on every rank, so the job list stays rank-uniform; its
    # lookup reports capable=0 and the MIN ends it no_capability everywhere.
    source: Optional[str]
    page_keys: List[str]
    submitted_at: float
    deadline: float
    next_start: int = 0
    phase: str = QUEUED
    request_pages: int = 0
    agreed_pages: int = 0
    host_idx: Optional[torch.Tensor] = None
    grant: Optional[dict] = None
    future: Any = None
    retry_at: float = 0.0
    backoff_s: float = _BACKOFF_INITIAL_S
    cancelled: bool = False
    committed_pages: int = 0
    rounds: int = 0
    last_refusal: Optional[str] = None
    # Cause of the last failed round (an owner refusal, "stale_grant",
    # "host_full", "read_failed", ...): the metric label of missing pages.
    last_failure: Optional[str] = None
    # A positive grant was agreed at least once (owner_evicted vs refused).
    granted: bool = False
    # This rank allocated the host slots of the agreed grant (READ phase).
    alloc_ok: bool = True
    # Filled by poll from the local future; kept until the reduced decision.
    local: Any = field(default=None, repr=False)
    resolved: bool = False
    local_stale: bool = False
    # Rank-local verdict when ``source`` is None (e.g. no_servicer,
    # tp_mismatch); kept out of the lockstep digest.
    local_reason: Optional[str] = None


class SwaBackfillManager:
    """Request-owned host->host SWA backfills of one window-residency rank.

    ``backend`` is the reader-side peer backend (``PeerKVBackend`` exposes
    ``swa_range_capable``, ``lookup_swa_range``, ``read_swa_range`` and
    ``release_swa_range``); None resolves the cache controller's storage
    backend at each lookup. ``io`` runs the network calls (a
    ``ThreadedBackfillIO`` by default); tests pass a controllable executor.
    """

    STAT_KEYS = (
        "submitted_pages",
        "committed_pages",
        "rounds",
        "refused_rounds",
        "read_failures",
        "host_full",
        "stale_grants",
    ) + (
        RESULT_OK,
        RESULT_PARTIAL,
        RESULT_TIMEOUT,
        RESULT_NO_CAPABILITY,
        RESULT_CANCELLED,
    )

    def __init__(
        self,
        cache: Any,
        backend: Any = None,
        *,
        timeout_s: float = 60.0,
        max_inflight_bytes: int = 16 << 30,
        bytes_per_page: Optional[int] = None,
        clock: Callable[[], float] = time.monotonic,
        is_io_rank: Optional[bool] = None,
        io: Any = None,
        metrics: Any = None,
    ) -> None:
        self._cache = cache
        self._backend = backend
        self._timeout_s = float(timeout_s)
        self._clock = clock
        self._page_size = int(cache.page_size)
        group = cache.host_pool_group
        self._pool_names = swa_anchored_pool_names(group)
        if bytes_per_page is None:
            bytes_per_page = swa_anchored_bytes_per_page(group, self._pool_names)
        self._bytes_per_page = max(1, int(bytes_per_page))
        self._max_inflight_pages = max(
            1, int(max_inflight_bytes) // self._bytes_per_page
        )
        if is_io_rank is None:
            dedup = getattr(group, "host_dedup", None)
            is_io_rank = not (dedup is not None and dedup.is_dummy)
        self._is_io_rank = bool(is_io_rank)
        self._io = io if io is not None else ThreadedBackfillIO()
        self._metrics = metrics
        self._jobs: list[_Job] = []
        self._by_req: dict[int, _Job] = {}
        self.stats = dict.fromkeys(self.STAT_KEYS, 0)

    # ------------------------------------------------------------------
    # Scheduler-thread API (lockstep points only)
    # ------------------------------------------------------------------

    def submit(
        self,
        req: Any,
        *,
        start: int,
        end: int,
        source: Optional[str],
        page_hashes: Sequence[str],
        local_reason: Optional[str] = None,
    ) -> bool:
        """Backfill request positions [start, end) (page aligned) from the
        owner servicer at ``source`` (its base "host:port"; each rank reads
        from its own owner rank). ``page_hashes`` are the peer page keys of
        exactly those pages (``swa_backfill_page_keys``). The request must hold
        a ledger whose coverage the committed slices extend. Returns False,
        with nothing queued, for an empty range.

        ``source`` may be None when this rank's own view of the owner (e.g.
        rank-local bootstrap info) has no usable servicer; ``local_reason``
        says why. Callers submit anyway, so whether a job exists never
        depends on rank-local state: the job's lookup reports capable=0 and
        the poll's MIN ends it no_capability on every rank."""
        page = self._page_size
        assert start % page == 0 and end % page == 0, (start, end)
        assert len(page_hashes) == (end - start) // page, (
            len(page_hashes),
            start,
            end,
        )
        ledger = getattr(req, "swa_ledger", None)
        assert ledger is not None, "open the request's SWA ledger before a backfill"
        assert id(req) not in self._by_req, f"backfill already pending for {req.rid}"
        if end <= start:
            return False
        now = self._clock()
        job = _Job(
            req=req,
            ledger=ledger,
            start=start,
            end=end,
            source=source,
            page_keys=list(page_hashes),
            submitted_at=now,
            deadline=now + self._timeout_s,
            next_start=start,
            local_reason=None if source is not None else (local_reason or "no_source"),
        )
        self._jobs.append(job)
        self._by_req[id(req)] = job
        req.swa_backfill_state = QUEUED
        self.stats["submitted_pages"] += (end - start) // page
        self._launch_queued()
        self._report()
        return True

    def cancel(self, req: Any) -> None:
        """Drop the request's backfill (finish, abort, release). A read that may
        still write the job's host slots keeps them until it has finished on
        every rank; nothing is committed after this call."""
        job = self._by_req.pop(id(req), None)
        if job is None:
            return
        job.cancelled = True
        req.swa_backfill_state = RESULT_CANCELLED
        if job.phase in (QUEUED, BACKOFF):
            self._finish(job, RESULT_CANCELLED)
            self._jobs.remove(job)
            self._launch_queued()
        # LOOKUP / READ: settled by poll once the IO is done everywhere.
        self._report()

    def poll(self) -> None:
        """Advance every job once. One MIN all-reduce when any job exists (the
        job list is identical on every rank). The vector starts with the job
        list's digest pair, so ranks whose lists have the same length but
        differ raise here before any state changes; a length difference is
        caught by the per-step check in ``_sync_hicache_ready_counts``."""
        if not self._jobs:
            return
        now = self._clock()
        digest = self.lockstep_digest()
        fields: list[int] = [digest, -digest]
        for job in self._jobs:
            fields.extend(self._local_status(job, now))
        status = torch.tensor(fields, dtype=torch.int64)
        self._cache._all_reduce(status, torch.distributed.ReduceOp.MIN)
        values = status.tolist()
        self.check_lockstep_digest(values[0], values[1], where="poll")
        reduced = [
            values[i : i + _NUM_FIELDS]
            for i in range(_HEADER_FIELDS, len(values), _NUM_FIELDS)
        ]
        for job, flags in zip(list(self._jobs), reduced):
            self._advance(job, flags, now)
        self._jobs = [job for job in self._jobs if job.phase != DONE]
        self._launch_queued()
        self._report()

    # SGLANG_HICACHE_PIPELINED_SYNC: poll() split around the cache's step-sync
    # vector, which carries these rows and is applied one sync step later.

    def lockstep_rows(self, num_slots: int) -> tuple[list[int], list]:
        """This rank's backfill fields of the pipelined step-sync vector: the
        job-list digest pair, then the status row of each of the first
        ``num_slots`` jobs, zero-padded to ``num_slots`` rows. The size is
        fixed, so ranks whose job lists diverged reduce equal-size vectors and
        raise at the apply instead of hanging; padded rows are never applied.
        Returns (fields, snapshot), the snapshot naming each row's job and the
        (phase, round) it was taken in."""
        now = self._clock()
        digest = self.lockstep_digest()
        fields: list[int] = [digest, -digest]
        snapshot = []
        for job in self._jobs[:num_slots]:
            fields.extend(self._local_status(job, now))
            snapshot.append((job, job.phase, job.rounds))
        fields.extend([0] * (_NUM_FIELDS * (num_slots - len(snapshot))))
        return fields, snapshot

    def apply_lockstep_rows(self, snapshot: list, values: Sequence[int]) -> None:
        """Advance the jobs of a lockstep_rows snapshot by their MIN-reduced
        rows. A job that moved since the snapshot (a lookup launched by a
        submit or cancel, a finished job) skips this round and is covered by
        the next snapshot; those moves run at lockstep points, so every rank
        skips the same rows."""
        self.check_lockstep_digest(values[0], values[1], where="pipelined poll")
        if not snapshot:
            return
        now = self._clock()
        for i, (job, phase, rounds) in enumerate(snapshot):
            if job.phase != phase or job.rounds != rounds:
                continue
            row = _HEADER_FIELDS + i * _NUM_FIELDS
            self._advance(job, list(values[row : row + _NUM_FIELDS]), now)
        self._jobs = [job for job in self._jobs if job.phase != DONE]
        self._launch_queued()
        self._report()

    def lockstep_digest(self) -> int:
        """Rank-uniform fingerprint of the job list: count, then per job the
        request id, range, progress and phase. Rank-local fields (futures,
        grants, clocks, host indices) are left out."""
        d = len(self._jobs) + 1
        for job in self._jobs:
            rid = zlib.crc32(str(getattr(job.req, "rid", "")).encode())
            for v in (
                rid,
                job.start,
                job.end,
                job.next_start,
                _PHASE_CODES[job.phase],
                job.request_pages,
                job.agreed_pages,
                job.committed_pages,
                int(job.cancelled),
            ):
                d = (d * 1000003 + (int(v) & _DIGEST_MASK) + 1) & _DIGEST_MASK
        return d

    def check_lockstep_digest(
        self, reduced_min: int, reduced_neg_max: int, where: str
    ) -> None:
        """Raise when a MIN-reduced ``[digest, -digest]`` pair shows that the
        ranks' job lists differ. Every rank sees the same reduced pair, so
        every rank raises; the alternative is a hang or a reduce over
        misaligned rows."""
        if reduced_min == -reduced_neg_max:
            return
        # ``source`` / ``local_reason`` are rank-local and stay out of the
        # digest; listing them names a rank-local submit verdict as the cause.
        jobs = [
            (
                getattr(j.req, "rid", "?"),
                j.start,
                j.end,
                j.next_start,
                j.phase,
                "source" if j.source is not None else f"no_source:{j.local_reason}",
            )
            for j in self._jobs
        ]
        raise RuntimeError(
            f"SWA backfill job lists diverged across TP ranks at {where}: "
            f"digest min {reduced_min} != max {-reduced_neg_max}; this rank has "
            f"{len(self._jobs)} jobs {jobs[:8]} (digest {self.lockstep_digest()}). "
            "submit/cancel must run at lockstep points with identical arguments "
            "on every rank."
        )

    def state_of(self, req: Any) -> Optional[str]:
        """The job phase while pending, then the final result."""
        job = self._by_req.get(id(req))
        if job is not None:
            return job.phase
        return getattr(req, "swa_backfill_state", None)

    def pending_count(self) -> int:
        """Jobs not yet settled, orphans included (the idle predicate)."""
        return len(self._jobs)

    def inflight_pages(self) -> int:
        return sum(self._reserved_pages(job) for job in self._jobs)

    def inflight_bytes(self) -> int:
        return self.inflight_pages() * self._bytes_per_page

    def host_tokens(self) -> int:
        """Host SWA tokens held by jobs (allocated, not yet committed)."""
        return sum(len(job.host_idx) for job in self._jobs if job.host_idx is not None)

    def close(self) -> None:
        shutdown = getattr(self._io, "shutdown", None)
        if shutdown is not None:
            shutdown()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reserved_pages(self, job: _Job) -> int:
        if job.phase == LOOKUP:
            return job.request_pages
        if job.phase == READ:
            return job.agreed_pages
        return 0

    def _resolve_backend(self) -> Any:
        if self._backend is not None:
            return self._backend
        controller = getattr(self._cache, "cache_controller", None)
        backend = getattr(controller, "storage_backend", None)
        if backend is None or not hasattr(backend, "lookup_swa_range"):
            return None
        return backend

    def _launch_queued(self) -> None:
        """Start lookups in submit order while the in-flight budget allows. A
        job is never overtaken, so a large one cannot starve."""
        budget = self._max_inflight_pages - self.inflight_pages()
        for job in self._jobs:
            if job.phase != QUEUED:
                continue
            if budget <= 0:
                break
            remaining = (job.end - job.next_start) // self._page_size
            job.request_pages = min(remaining, budget)
            budget -= job.request_pages
            self._launch_lookup(job)

    def _launch_lookup(self, job: _Job) -> None:
        offset = (job.next_start - job.start) // self._page_size
        keys = job.page_keys[offset : offset + job.request_pages]
        job.phase = LOOKUP
        job.req.swa_backfill_state = LOOKUP
        job.rounds += 1
        job.alloc_ok = True
        self.stats["rounds"] += 1
        if job.source is None:
            # No usable owner on this rank (checked before the dedup-follower
            # upper bound, so a follower's verdict counts too): capable=0.
            job.future = _Done(_LookupResult(False, 0, None, job.local_reason))
            return
        if not self._is_io_rank:
            # Allocator-only follower: report the upper bound; the MIN picks
            # the source rank's grant.
            job.future = _Done(_LookupResult(True, job.request_pages))
            return
        backend = self._resolve_backend()
        if backend is None:
            job.future = _Done(_LookupResult(False, 0))
            return
        metadata = {"req_id": str(getattr(job.req, "rid", ""))}
        job.future = self._io.submit(
            "lookup", _lookup_io, backend, job.source, keys, metadata, self._clock
        )

    def _launch_read(self, job: _Job) -> None:
        job.phase = READ
        job.req.swa_backfill_state = READ
        if not job.alloc_ok:
            # This rank has no slots for the agreed grant: no read; the next
            # poll reduces alloc_ok=0 and every rank backs off.
            job.future = _Done(False)
            return
        if not self._is_io_rank:
            job.future = _Done(True)
            return
        grant = job.grant
        backend = self._resolve_backend() if grant is not None else None
        if grant is None or backend is None:
            job.future = _Done(False)
            return
        job.future = self._io.submit(
            "read",
            _read_io,
            backend,
            grant,
            job.agreed_pages,
            list(self._pool_names),
            job.host_idx,
        )

    def _local_status(self, job: _Job, now: float) -> list[int]:
        status = [1, 1, 0, int(now < job.deadline), int(now >= job.retry_at), 1]
        if job.phase not in (LOOKUP, READ):
            return status
        if not job.resolved:
            future = job.future
            if future is None or not future.done():
                status[_IO_DONE] = 0
                return status
            try:
                job.local = future.result()
            except Exception:  # noqa: BLE001 - IO errors retry, never raise here
                # Logged once: the value is kept until every rank's IO is done.
                logger.warning("SWA backfill %s IO raised", job.phase, exc_info=True)
                job.local = (
                    _LookupResult(True, 0, refused_reason="io_error")
                    if job.phase == LOOKUP
                    else False
                )
            job.resolved = True
        value = job.local
        if job.phase == LOOKUP:
            status[_CAPABLE] = int(bool(value.capable))
            pages = max(0, min(int(value.pages), job.request_pages))
            # Re-evaluated at every poll while other ranks' lookups run: a
            # grant that aged past the freshness bound is reported as 0, so
            # the MIN refuses it before any host slot is allocated.
            job.local_stale = bool(pages) and self._grant_is_stale(value, now)
            status[_VALUE] = 0 if job.local_stale else pages
        else:
            status[_VALUE] = int(bool(value))
            status[_ALLOC_OK] = int(job.alloc_ok)
        return status

    @staticmethod
    def _grant_is_stale(result: _LookupResult, now: float) -> bool:
        if result.started_at is None or not result.ttl_s:
            return False
        return now - result.started_at > result.ttl_s * _GRANT_FRESH_FRACTION

    @staticmethod
    def _clear_local(job: _Job) -> None:
        job.local = None
        job.resolved = False
        job.local_stale = False

    def _advance(self, job: _Job, flags: list[int], now: float) -> None:
        io_done, capable, value, not_expired, ready, alloc_ok = flags
        if job.phase == LOOKUP:
            if not io_done:
                return
            local, stale = job.local, job.local_stale
            self._clear_local(job)
            job.grant = local.grant if local is not None else None
            if local is not None and local.refused_reason:
                job.last_refusal = local.refused_reason
            if job.cancelled:
                self._release_grant(job)
                self._finish(job, RESULT_CANCELLED)
            elif not capable:
                self._release_grant(job)
                self._finish(job, RESULT_NO_CAPABILITY)
            elif value <= 0:
                self._release_grant(job)
                self.stats["refused_rounds"] += 1
                if stale:
                    self.stats["stale_grants"] += 1
                    job.last_failure = "stale_grant"
                else:
                    job.last_failure = (
                        local.refused_reason if local is not None else None
                    )
                self._retry_or_expire(job, not_expired, now)
            elif not not_expired:
                self._release_grant(job)
                self._finish_expired(job)
            else:
                # Allocate for the MIN grant. The outcome is rank-local
                # (reclaim, skewed pools); a failed rank skips its read and
                # the READ poll reduces alloc_ok, so every rank backs off.
                job.granted = True
                job.agreed_pages = value
                job.host_idx = self._alloc_host(value * self._page_size)
                job.alloc_ok = job.host_idx is not None
                self._launch_read(job)
            return
        if job.phase == READ:
            if not io_done:
                return
            self._clear_local(job)
            self._release_grant(job)
            ok = bool(value) and bool(alloc_ok)
            if job.cancelled or not ok:
                # The read finished on every rank: its slots can go.
                self._free_host(job.host_idx)
                job.host_idx = None
                job.alloc_ok = True
                if job.cancelled:
                    self._finish(job, RESULT_CANCELLED)
                else:
                    if not alloc_ok:
                        self.stats["host_full"] += 1
                        job.last_failure = "host_full"
                    else:
                        self.stats["read_failures"] += 1
                        job.last_failure = "read_failed"
                    self._retry_or_expire(job, not_expired, now)
                return
            self._commit(job)
            if job.phase == DONE:
                return
            if job.next_start >= job.end:
                self._finish(job, RESULT_OK)
            elif not not_expired:
                self._finish_expired(job)
            else:
                # More to fetch: look the remainder up right away.
                job.phase = QUEUED
                job.req.swa_backfill_state = QUEUED
                job.backoff_s = _BACKOFF_INITIAL_S
            return
        # QUEUED (rate limited) or BACKOFF: nothing in flight.
        if not not_expired:
            self._finish_expired(job)
        elif job.phase == BACKOFF and ready:
            job.phase = QUEUED
            job.req.swa_backfill_state = QUEUED

    def _retry_or_expire(self, job: _Job, not_expired: int, now: float) -> None:
        if not not_expired:
            self._finish_expired(job)
            return
        job.phase = BACKOFF
        job.req.swa_backfill_state = BACKOFF
        job.retry_at = now + job.backoff_s
        job.backoff_s = min(job.backoff_s * 2, _BACKOFF_MAX_S)

    def _commit(self, job: _Job) -> None:
        """Append the read slices to the request's ledger (lockstep point)."""
        ledger = getattr(job.req, "swa_ledger", None)
        n_tokens = job.agreed_pages * self._page_size
        host, job.host_idx = job.host_idx, None
        if ledger is not job.ledger:
            # The ledger was released or replaced without a cancel.
            self._free_host(host)
            self._finish(job, RESULT_CANCELLED)
            return
        ledger.add_segment(SwaSegment(job.next_start, job.next_start + n_tokens, host))
        job.next_start += n_tokens
        job.committed_pages += job.agreed_pages
        job.agreed_pages = 0
        job.last_failure = None
        self.stats["committed_pages"] += n_tokens // self._page_size
        job.req.swa_backfill_pages = (
            getattr(job.req, "swa_backfill_pages", 0) + n_tokens // self._page_size
        )

    def _finish_expired(self, job: _Job) -> None:
        self._finish(job, RESULT_PARTIAL if job.committed_pages else RESULT_TIMEOUT)

    def _finish(self, job: _Job, result: str) -> None:
        job.phase = DONE
        current = self._by_req.get(id(job.req))
        if current is job:
            del self._by_req[id(job.req)]
        if current is None or current is job:
            # A cancelled orphan must not overwrite a newer job's state.
            job.req.swa_backfill_state = result
        self.stats[result] += 1
        total_pages = (job.end - job.start) // self._page_size
        metrics = self._metrics
        if metrics is not None:
            labels = metrics.labels
            if job.committed_pages:
                metrics.backfill_pages.labels(
                    **labels,
                    result=RESULT_OK if result == RESULT_OK else RESULT_PARTIAL,
                ).inc(job.committed_pages)
            missing = total_pages - job.committed_pages
            if missing > 0 and result != RESULT_OK:
                metrics.backfill_pages.labels(
                    **labels, result=self.missing_pages_label(job, result)
                ).inc(missing)
            if result == RESULT_OK:
                metrics.backfill_seconds.labels(**labels).observe(
                    max(0.0, self._clock() - job.submitted_at)
                )
        if result != RESULT_OK:
            logger.info(
                "SWA backfill of %s [%d, %d) from %s ended %s: %d of %d pages "
                "committed in %d rounds (last refusal %s, last failure %s)",
                getattr(job.req, "rid", "?"),
                job.start,
                job.end,
                job.source if job.source is not None else f"<none: {job.local_reason}>",
                result,
                job.committed_pages,
                total_pages,
                job.rounds,
                job.last_refusal,
                job.last_failure,
            )

    @staticmethod
    def missing_pages_label(job: _Job, result: str) -> str:
        """``result`` label of the pages a job did not commit. A deadline
        (timeout / partial) is attributed to its last failed round: an owner
        refusal is ``refused``, or ``owner_evicted`` when the owner had
        granted pages of this job before and then no longer had them; slow or
        failed IO stays ``timeout``."""
        if result not in (RESULT_TIMEOUT, RESULT_PARTIAL):
            return result
        reason = job.last_failure
        if reason in _OWNER_REFUSALS:
            if reason in _EVICTION_REFUSALS and job.granted:
                return LABEL_OWNER_EVICTED
            return LABEL_REFUSED
        return RESULT_TIMEOUT

    def _release_grant(self, job: _Job) -> None:
        grant, job.grant = job.grant, None
        if grant is None or not self._is_io_rank:
            return
        backend = self._resolve_backend()
        if backend is None:
            return
        try:
            backend.release_swa_range(grant)
        except Exception:  # noqa: BLE001 - the owner expires it on TTL
            logger.warning("SWA backfill lease release failed", exc_info=True)

    def _alloc_host(self, n_tokens: int) -> Optional[torch.Tensor]:
        cache = self._cache
        return cache.host_pool_group.alloc(
            n_tokens,
            pool=PoolName.SWA,
            reclaim=lambda n: cache.evict_host(n, ComponentType.SWA),
        )

    def _free_host(self, host: Optional[torch.Tensor]) -> None:
        if host is not None and host.numel():
            self._cache.components[ComponentType.SWA].free_host_values([host])

    def _report(self) -> None:
        metrics = self._metrics
        if metrics is not None:
            metrics.backfill_inflight_bytes.labels(**metrics.labels).set(
                self.inflight_bytes()
            )


def _lookup_io(
    backend: Any,
    source: str,
    keys: list,
    metadata: dict,
    clock: Callable[[], float] = time.monotonic,
) -> _LookupResult:
    if not backend.swa_range_capable(source):
        return _LookupResult(False, 0)
    started = clock()
    grant = backend.lookup_swa_range(source, keys, request_metadata=metadata)
    if grant is None:
        return _LookupResult(True, 0, None, "rpc_error")
    refused = grant.get("refused_reason")
    if refused == "unsupported":
        # The owner cannot serve ranges at all (no complete host SWA): the
        # same as a missing capability, not a retryable refusal.
        return _LookupResult(False, 0, None, refused)
    pages = int(grant.get("num_pages", 0))
    ttl_ms = grant.get("lease_ttl_ms")
    return _LookupResult(
        True,
        pages,
        grant if grant.get("lease_id") else None,
        refused,
        started_at=started,
        ttl_s=float(ttl_ms) / 1e3 if ttl_ms else None,
    )


def _read_io(
    backend: Any, grant: dict, num_pages: int, pool_names: list, host_idx: torch.Tensor
) -> bool:
    return bool(backend.read_swa_range(grant, num_pages, pool_names, host_idx))
