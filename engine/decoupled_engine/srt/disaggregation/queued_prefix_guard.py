"""Queued-prefix host protection for PD prefill workers (opt-in).

Stage 2 of queued-prefix protection: the host lock on queued prefixes and
the pulled-page lock.

In a long prefill queue a request waits for minutes. Write-through backups and
peer pulls of other requests evict the host copy of its cached prefix, and the
request recomputes it. ``--hicache-queued-prefix-protect`` puts one host lock
(``inc_host_lock_ref``) on the deepest backed-up node of a queued request's
path until the request is admitted. Every host-freeing path skips a node with
a host lock or with a child (``_is_host_leaf``, duplicate reclaim, unbacked
drop, tombstone cleanup), so one lock keeps the whole path on host.

Sources (one lock per request; the newest lock replaces the older one):
- ``pull``: S-PULL / S-REPULL. When a peer pull of a queued request lands
  (``UnifiedRadixCache._handle_prefetch_result``), the inserted host node is
  locked before the prefetch anchor lock is released (no unlocked window).
- ``all``: ``pull`` + S-ARRIVE. At arrival the deepest backed-up node of the
  request's match is locked (the guard matches a request that the arrival path
  did not match, e.g. a hint-less request on the hint-only peer_kv backend). A
  request that gets no lock stays a candidate: each pass re-matches up to
  ``MAX_EXAMINE_PER_PASS`` candidates, oldest first, at most once per
  ``RETRY_AFTER_PASSES`` passes, while budget is free.

Budget: the sum of lock costs (token depth of the locked node) stays at most
min(``--hicache-queued-prefix-max-tokens``, ``--hicache-queued-prefix-max-
fraction`` x the host pool's usable tokens). Requests get budget in arrival
order (``Req.qp_seq``): an arrival never takes budget from an older lock; a
landing may exceed it, and ``step`` then releases the newest locks.

Release: ``admitted`` (in the batch formed this pass; admission holds its own
device lock), ``gone`` (in neither queue: abort, bootstrap failure, waiting
timeout), ``ttl`` (lock age >= the TTL; MAX all-reduce every
``TTL_CHECK_EVERY_PASSES`` passes), ``budget``, ``pressure`` (below) and
``reset`` (cache flush; the idle check refuses a flush while locks exist).

Never block, never fail: when ``evict_host`` frees less than it
was asked, the guard releases its newest locks (never the waiting-queue head's,
never the lock of the request whose pull allocation asked) and evicts again,
up to ``PRESSURE_RELEASE_FACTOR`` x the shortfall. A backup or a pull
allocation that still cannot get space falls back exactly as without the
guard. A request released for pressure gets no S-ARRIVE lock for
``PRESSURE_NO_ARRIVE_PASSES`` passes.

TP lockstep: every lock and release is keyed by tree node id and request id
and decided from replicated state (queue membership and order, ``qp_seq``,
the radix tree, host ``available_size``, pass counts); the hooks run at
replicated points (arrival, the landing drain, ``evict_host`` callers, the
step after batch formation). The only collective is the TTL MAX all-reduce,
entered by every rank at the same pass with the same lock set. Host dedup:
each rank keeps its own tree and allocator, every rank takes the same lock,
and the budget counts the one pool once.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from decoupled_engine.srt.disaggregation.utils import DisaggregationMode
from decoupled_engine.srt.runtime_context import get_memory

if TYPE_CHECKING:
    from decoupled_engine.srt.managers.schedule_batch import Req, ScheduleBatch
    from decoupled_engine.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

QP_MODES = ("off", "pull", "all")
# Candidates re-matched per pass (each costs one radix match), as BWP.
MAX_EXAMINE_PER_PASS = 16
# A candidate that got no lock is examined again after this many passes.
RETRY_AFTER_PASSES = 64
# No S-ARRIVE lock for this many passes after a pressure release.
PRESSURE_NO_ARRIVE_PASSES = 256
# evict_host releases at most this many times the shortfall.
PRESSURE_RELEASE_FACTOR = 2
# TTL flags are MAX-reduced every this many passes while locks exist, as BWP.
TTL_CHECK_EVERY_PASSES = 32
# A lock released by its TTL (a leak bound) is not taken again for its request.
TTL_NO_ARRIVE_PASSES = 1 << 62
SUMMARY_INTERVAL_S = 60.0


@dataclasses.dataclass
class _Lock:
    req: Any
    node_id: Any
    params: Any
    cost: int
    seq: int
    source: str
    created: float  # time.monotonic(); TTL only, MAX-reduced across ranks


def create_queued_prefix_guard(scheduler: Scheduler) -> Optional[QueuedPrefixGuard]:
    """Build the guard for ``Scheduler.init_disaggregation``.

    None when ``--hicache-queued-prefix-protect`` is off (the scheduler is not
    read) and on a decode worker (P and D may share one launch config, so there
    it is logged as inactive). Every other unsupported setup is refused."""
    mode = get_memory().hicache_queued_prefix_protect
    if mode == "off":
        return None
    flag = f"--hicache-queued-prefix-protect {mode}"
    if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
        logger.info(
            "%s: inactive on a decode worker (it applies to prefill only).", flag
        )
        return None
    problems = _unsupported(scheduler, mode)
    if problems:
        raise ValueError(f"{flag} requires " + ", ".join(problems))
    ttl = float(get_memory().hicache_queued_prefix_ttl)
    if ttl == 0:
        from decoupled_engine.srt.environ import envs

        # The lock TTL defaults to the waiting timeout.
        ttl = float(envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get())
    return QueuedPrefixGuard(
        scheduler,
        mode=mode,
        max_tokens=get_memory().hicache_queued_prefix_max_tokens,
        max_fraction=get_memory().hicache_queued_prefix_max_fraction,
        min_tokens=get_memory().hicache_queued_prefix_min_tokens,
        ttl_s=ttl,
    )


def _unsupported(scheduler: Scheduler, mode: str) -> List[str]:
    from decoupled_engine.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    problems = []
    if mode not in QP_MODES:
        problems.append("--hicache-queued-prefix-protect off, pull or all")
    if scheduler.disaggregation_mode != DisaggregationMode.PREFILL:
        # Symmetric workers are refused until the late-D stage.
        problems.append("--disaggregation-mode prefill")
    if not scheduler.enable_hierarchical_cache:
        problems.append("--enable-hierarchical-cache")
    if not isinstance(scheduler.tree_cache, UnifiedRadixCache):
        problems.append("the unified radix cache")
    if get_memory().hicache_host_memory_mode != "cache":
        problems.append("--hicache-host-memory-mode cache")
    if get_memory().hicache_write_policy == "write_back":
        problems.append("a write-through --hicache-write-policy")
    if scheduler.is_hybrid_ssm or scheduler.is_hybrid_swa:
        # One host lock protects a FULL path; SWA / mamba host state is not.
        problems.append("a FULL-only model (no SWA, no mamba)")
    if scheduler.ps.pp_size != 1:
        problems.append("pp_size == 1")
    if scheduler.ps.attn_cp_size != 1:
        problems.append("attn_cp_size == 1")
    if scheduler.enable_dp_attention:
        problems.append("no DP attention")
    if get_memory().hicache_queued_prefix_max_tokens < 1:
        problems.append("--hicache-queued-prefix-max-tokens >= 1")
    if not 0 < get_memory().hicache_queued_prefix_max_fraction <= 1:
        problems.append("0 < --hicache-queued-prefix-max-fraction <= 1")
    if get_memory().hicache_queued_prefix_min_tokens < 0:
        problems.append("--hicache-queued-prefix-min-tokens >= 0")
    if get_memory().hicache_queued_prefix_ttl < 0:
        problems.append("--hicache-queued-prefix-ttl >= 0")
    return problems


class QueuedPrefixGuard:
    """Host locks on the cached prefixes of queued PD prefill requests."""

    def __init__(
        self,
        scheduler: Scheduler,
        *,
        mode: str,
        max_tokens: int = 3_000_000,
        max_fraction: float = 0.30,
        min_tokens: int = 8192,
        ttl_s: float = 900.0,
    ):
        self.scheduler = scheduler
        self.cache = scheduler.tree_cache
        self.mode = mode
        self.min_tokens = int(min_tokens)
        self.ttl_s = float(ttl_s)
        usable = self._host_usable_tokens()
        self.budget = int(max_tokens)
        if usable is not None:
            self.budget = min(self.budget, int(max_fraction * usable))
        # rid -> the request's one lock, in grant order.
        self.locks: Dict[str, _Lock] = {}
        self.locked_tokens = 0
        self._seq = 0
        self._pass = 0
        self._retry: Dict[str, int] = {}  # S-ARRIVE candidates: rid -> next pass
        self._no_arrive: Dict[str, int] = {}  # pressure-released: rid -> pass
        self.stats: Counter = Counter()
        self._log = scheduler.ps.attn_tp_rank == 0
        self._last_summary = time.monotonic()
        self.cache.queued_prefix_guard = self
        if self._log:
            logger.info(
                "Queued-prefix guard enabled: mode=%s budget=%d (max_tokens=%d "
                "max_fraction=%.2f usable=%s) min_tokens=%d ttl_s=%.0f",
                mode,
                self.budget,
                int(max_tokens),
                float(max_fraction),
                usable,
                self.min_tokens,
                self.ttl_s,
            )

    # ------------------------------------------------------------ hooks
    def on_arrival(self, req: Req) -> None:
        """PD prefill arrival, after the arrival pull (or its (b1) deferral):
        number the request; with ``all``, lock its local prefix (S-ARRIVE)."""
        self._seq += 1
        req.qp_seq = self._seq
        if self.mode != "all":
            return
        # The arrival pull path matched a hinted request already; match the
        # others here (hint-less requests on the hint-only backend, (b1)
        # deferred requests): one radix match per arrival.
        self._try_arrive(req, match=req.last_host_node is None)

    def on_landed(self, rid: str, node_id: Any) -> None:
        """A peer pull of ``rid`` landed and inserted host node ``node_id``
        (called before the prefetch anchor lock is released): lock it
        (S-PULL, or S-REPULL for a (b1)/(b2) pull) and drop the request's older
        lock, if any. A landing may exceed the budget; ``step`` trims it."""
        req = self._find_queued(rid)
        if req is None:
            self.stats["landed_not_queued"] += 1
            return
        node, cost = self._lock_target(node_id)
        if node is None or cost < self.min_tokens:
            self.stats["skip_short"] += 1
            return
        source = "repull" if getattr(req, "jit_pull_issued", None) else "pull"
        old = self.locks.pop(rid, None)
        if old is not None and old.node_id == node.id:
            self.locks[rid] = old
            return
        self._grant(req, node.id, cost, source)
        if old is not None:
            self._unlock(old)
            self.stats["reanchored"] += 1

    def step(self, batch: Optional[ScheduleBatch]) -> None:
        """Once per pass, after batch formation (next to the BWP step)."""
        self._pass += 1
        s = self.scheduler
        boot = s.disagg_prefill_bootstrap_queue.queue
        admitted = {id(r) for r in batch.reqs} if batch is not None else set()
        queued = {id(r) for r in boot}
        queued.update(id(r) for r in s.waiting_queue)
        for lock in sorted(self.locks.values(), key=lambda lk: lk.seq):
            if id(lock.req) in admitted:
                self._release(lock, "admitted")
            elif id(lock.req) not in queued:
                self._release(lock, "gone")
        if self._retry or self._no_arrive:
            live = {r.rid for r in boot}
            live.update(r.rid for r in s.waiting_queue)
            for table in (self._retry, self._no_arrive):
                for rid in [rid for rid in table if rid not in live]:
                    del table[rid]
        if self.ttl_s > 0 and self.locks and self._pass % TTL_CHECK_EVERY_PASSES == 0:
            self._release_expired()
        while self.locked_tokens > self.budget and self.locks:
            self._release(max(self.locks.values(), key=lambda lk: lk.seq), "budget")
        if self.mode == "all" and self.budget - self.locked_tokens >= self.min_tokens:
            self._examine_candidates(boot)
        if self._log and time.monotonic() - self._last_summary >= SUMMARY_INTERVAL_S:
            self._log_summary()

    def relieve_host_pressure(
        self, num_tokens: int, freed: int, exclude_rid: Optional[str], component_type
    ) -> int:
        """Called by ``evict_host`` when it freed less than ``num_tokens``:
        release the newest locks and evict again. Returns the tokens freed."""
        limit = PRESSURE_RELEASE_FACTOR * (num_tokens - freed)
        released = 0
        while freed < num_tokens and released < limit:
            cost = self.release_newest("pressure", exclude_rid)
            if cost == 0:
                break
            released += cost
            result = self.cache.tree_core.drive_host_eviction(
                component_type, num_tokens - freed
            )
            self.cache._free_values(result.device_frees, result.host_frees)
            freed += result.tracker.get(component_type, 0)
        if released:
            self.stats["pressure_events"] += 1
            self.stats["pressure_released_tokens"] += released
        if freed < num_tokens:
            self.stats["pressure_short_events"] += 1
        return freed

    def release_newest(self, reason: str, exclude_rid: Optional[str] = None) -> int:
        """Release the lock with the highest ``qp_seq``, except the waiting
        queue head's and ``exclude_rid``'s. Returns its cost (0: none)."""
        waiting = self.scheduler.waiting_queue
        head = waiting[0] if waiting else None
        for lock in sorted(self.locks.values(), key=lambda lk: -lk.seq):
            if lock.req is head or lock.req.rid == exclude_rid:
                continue
            self._release(lock, reason)
            return lock.cost
        return 0

    def has_locks(self) -> bool:
        return bool(self.locks)

    def reset(self) -> None:
        """The cache was flushed (only when idle, so normally no lock is left):
        forget every lock and candidate; the old node ids are gone."""
        self.stats["released_reset"] += len(self.locks)
        self.locks.clear()
        self.locked_tokens = 0
        self._retry.clear()
        self._no_arrive.clear()

    # --------------------------------------------------------- internals
    def _host_usable_tokens(self) -> Optional[int]:
        controller = getattr(self.cache, "cache_controller", None)
        pool = getattr(controller, "mem_pool_host", None)
        try:
            # Empty at startup, net of the servicer reserve; equal on all ranks.
            return int(pool.available_size())
        except Exception:
            return None

    def _find_queued(self, rid: str) -> Optional[Req]:
        s = self.scheduler
        for req in s.waiting_queue:
            if req.rid == rid:
                return req
        for req in s.disagg_prefill_bootstrap_queue.queue:
            if req.rid == rid:
                return req
        return None

    def _lock_target(self, node_id: Any) -> Tuple[Any, int]:
        """The deepest node at or above ``node_id`` with a FULL host copy (a
        host lock on a node without one does nothing under write-through), and
        its token depth (the lock's cost)."""
        from decoupled_engine.srt.mem_cache.unified_cache.component_type import (
            BASE_COMPONENT_TYPE,
        )

        core = self.cache.tree_core
        if node_id is None:
            return None, 0
        try:
            node = core.node_by_id(node_id)
        except KeyError:
            return None, 0
        root = core.root_node
        while node is not None and node is not root:
            if node.component_data[BASE_COMPONENT_TYPE].host_value is not None:
                break
            node = node.parent
        if node is None or node is root:
            return None, 0
        cost, cur = 0, node
        while cur is not None and cur is not root:
            cost += len(cur.key)
            cur = cur.parent
        return node, cost

    def _try_arrive(self, req: Req, match: bool) -> None:
        rid = req.rid
        if rid in self.locks:
            return
        self._retry[rid] = self._pass + RETRY_AFTER_PASSES
        if self._no_arrive.get(rid, -1) > self._pass:
            self.stats["skip_suppressed"] += 1
            return
        if rid in self.cache.ongoing_prefetch:
            # Its prefetch anchor lock holds the local prefix; S-PULL follows.
            self.stats["skip_inflight"] += 1
            return
        if match:
            req.init_next_round_input(self.cache, cow_mamba=False)
            self.stats["arrive_matches"] += 1
        node, cost = self._lock_target(req.best_match_node)
        if node is None or cost < self.min_tokens:
            self.stats["skip_short"] += 1
            return
        if self.locked_tokens + cost > self.budget:
            self.stats["skip_budget"] += 1
            self.stats["skip_budget_tokens"] += cost
            return
        self._grant(req, node.id, cost, "arrive")
        del self._retry[rid]

    def _examine_candidates(self, boot) -> None:
        s = self.scheduler
        queued = sorted(
            [*s.waiting_queue, *boot], key=lambda r: getattr(r, "qp_seq", 0)
        )
        examined = 0
        for req in queued:
            if examined >= MAX_EXAMINE_PER_PASS:
                break
            if self.budget - self.locked_tokens < self.min_tokens:
                break
            rid = req.rid
            if rid in self.locks or self._retry.get(rid, self._pass) > self._pass:
                continue
            if req.finished():
                continue
            examined += 1
            self._try_arrive(req, match=True)

    def _grant(self, req: Req, node_id: Any, cost: int, source: str) -> None:
        params = self.cache.inc_host_lock_ref(node_id).to_dec_params()
        self.locks[req.rid] = _Lock(
            req=req,
            node_id=node_id,
            params=params,
            cost=cost,
            seq=getattr(req, "qp_seq", 0),
            source=source,
            created=time.monotonic(),
        )
        self.locked_tokens += cost
        self.stats[f"lock_{source}"] += 1
        self.stats[f"lock_{source}_tokens"] += cost

    def _unlock(self, lock: _Lock) -> None:
        self.locked_tokens -= lock.cost
        try:
            self.cache.dec_host_lock_ref(lock.node_id, lock.params)
        except KeyError:
            # The node left the tree although it was host-locked (no known
            # path does this): nothing to release; never fail the pass.
            self.stats["unlock_missing"] += 1

    def _release(self, lock: _Lock, reason: str) -> None:
        rid = lock.req.rid
        del self.locks[rid]
        self._unlock(lock)
        self.stats[f"released_{reason}"] += 1
        self.stats[f"released_{reason}_tokens"] += lock.cost
        # When a still-queued request may take an S-ARRIVE lock again.
        if reason == "budget":
            self._retry[rid] = self._pass + RETRY_AFTER_PASSES
        elif reason == "pressure":
            self._no_arrive[rid] = self._pass + PRESSURE_NO_ARRIVE_PASSES
        elif reason == "ttl":
            self._no_arrive[rid] = self._pass + TTL_NO_ARRIVE_PASSES

    def _release_expired(self) -> None:
        locks = sorted(self.locks.values(), key=lambda lk: lk.seq)
        now = time.monotonic()
        flags = torch.tensor(
            [int(now - lk.created >= self.ttl_s) for lk in locks],
            dtype=torch.int64,
            device="cpu",
        )
        # Wall clocks differ across ranks: any rank's expiry expires the lock.
        self.cache._all_reduce(flags, torch.distributed.ReduceOp.MAX)
        for lock, expired in zip(locks, flags.tolist()):
            if expired:
                self._release(lock, "ttl")

    def _log_summary(self) -> None:
        self._last_summary = time.monotonic()
        logger.info(
            "Queued-prefix guard summary: mode=%s locks=%d locked_tokens=%d "
            "budget=%d %s",
            self.mode,
            len(self.locks),
            self.locked_tokens,
            self.budget,
            " ".join(f"{k}={v}" for k, v in sorted(self.stats.items())),
        )
