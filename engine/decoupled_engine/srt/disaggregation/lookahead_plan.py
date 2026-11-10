"""Read-only, replicated prefill forecast for compute-first late binding.

The forecast is built after processing the previous forward result. It never
matches through the cache API: matching can split nodes and change LRU ticks.
Only integer queue and budget inputs affect the forecast. Wall-clock timing is
reported separately and must not select requests or change cache state.
"""
from __future__ import annotations

import logging
import math
import os
import time
from array import array
from dataclasses import dataclass, field, replace
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrefixPeek:
    device_tokens: int = 0
    host_tokens: int = 0
    # Node references are used only to detect an eviction or split, not ordered.
    nodes: tuple = field(default=(), compare=False, repr=False)
    stamps: tuple = field(default=(), compare=False, repr=False)

    def valid(self) -> bool:
        return self.stamps == tuple(_node_stamp(n) for n in self.nodes)


def _node_stamp(node):
    return (
        bool(getattr(node, "deleted", False)),
        bool(getattr(node, "evicted", False)),
        bool(getattr(node, "backuped", False)),
        len(node.key),
        getattr(getattr(node, "parent", None), "id", None),
    )


def peek_prefix(tree_cache: Any, req: Any) -> PrefixPeek:
    """Return device and host prefix lengths without changing any tree state."""
    from decoupled_engine.srt.mem_cache.radix_cache import RadixKey

    if tree_cache is None or getattr(tree_cache, "disable", False):
        return PrefixPeek()
    core = getattr(tree_cache, "tree_core", tree_cache)
    node = getattr(core, "root_node", None)
    if node is None or not hasattr(node, "children"):
        return PrefixPeek()
    ids = req.origin_input_ids
    ids = ids if isinstance(ids, array) else array("q", ids)
    limit = max(0, len(ids) - 1)
    if not limit:
        return PrefixPeek()
    page = max(1, int(getattr(core, "page_size", 1)))
    key = RadixKey(
        token_ids=ids, extra_key=getattr(req, "extra_key", None), limit=limit,
        cache_salt=getattr(req, "cache_salt", None),
    )
    key, _ = key.maybe_to_bigram_view(bool(getattr(core, "is_eagle", False)))
    key = key.page_aligned(page)
    device = host = 0
    nodes = []
    host_suffix = False
    while len(key):
        child = node.children.get(key.child_key(page))
        if child is None:
            break
        nodes.append(child)
        evicted = bool(getattr(child, "evicted", False))
        if evicted and not getattr(child, "backuped", False):
            break
        n = child.key.match(key, page_size=page)
        host_suffix = host_suffix or evicted
        if host_suffix:
            host += n
        else:
            device += n
        if n < len(child.key):
            break
        node, key = child, key[n:]
    return PrefixPeek(device, host, tuple(nodes), tuple(_node_stamp(n) for n in nodes))


@dataclass(frozen=True)
class PlanCandidate:
    rid: str
    uncached_tokens: int
    host_tokens: int = 0
    device_tokens: int = 0
    pull_tokens: int = 0
    ready_pass: int = 1
    skip_reason: Optional[str] = None
    gated: bool = False
    bound: bool = True

    @property
    def uncached(self):
        return self.uncached_tokens


@dataclass(frozen=True)
class PlanEntry:
    rid: str
    start_pass: Optional[int]
    end_pass: Optional[int]
    gate_open_start_pass: Optional[int]
    uncached_tokens: int
    host_tokens: int
    device_tokens: int
    pull_tokens: int
    uncached_ahead: int
    host_demand: int
    skip_reason: Optional[str]
    gated: bool
    order: int

    @property
    def gate_open_start(self):
        return self.gate_open_start_pass

    @property
    def uncached(self):
        return self.uncached_tokens


@dataclass(frozen=True)
class LookaheadPlan:
    entries: dict[str, PlanEntry]
    order: tuple[str, ...]
    horizon: int
    pass_index: int = 0
    tail_reached: bool = False
    reserve_bound: int = 0
    # Integer inputs permit T0 to append a new request without mutating the plan.
    candidates: tuple[PlanCandidate, ...] = field(default=(), repr=False)
    partial: Optional[PlanCandidate] = field(default=None, repr=False)
    options: dict = field(default_factory=dict, repr=False, compare=False)

    def tail_candidate(self, candidate: PlanCandidate) -> Optional[PlanEntry]:
        if not self.tail_reached:
            return None
        return build_plan(
            self.candidates + (candidate,), partial=self.partial, **self.options
        ).entries.get(candidate.rid)


def _ceil_page(tokens, page):
    return ((max(0, tokens) + page - 1) // page) * page


def _simulate(candidates, partial, *, chunk_tokens, page_size, max_input_tokens,
              request_slots, horizon, gate_open):
    """Forecast ordinary chunked prefill without memory side effects.

    Forecast memory checks deliberately remain approximate. A future bind is not
    invented: gated rows have no normal start until the gate clears in a later
    real snapshot. The gate-open run supplies the binder's alternative forecast.
    """
    waiting = list(candidates)
    remaining = {c.rid: max(1, c.uncached_tokens) for c in waiting}
    if partial is not None:
        remaining[partial.rid] = max(1, partial.uncached_tokens)
    starts, ends, batches = {}, {}, []
    if partial is not None:
        starts[partial.rid] = 1
    active = partial
    # Starts stop at the horizon; an already-started long partial must retain its
    # true end estimate (the binder works backwards from that end).
    for k in range(1, horizon + 1):
        budget = min(chunk_tokens, max_input_tokens)
        batch = []
        cap = request_slots
        if active:
            take = min(remaining[active.rid], budget)
            remaining[active.rid] -= take
            budget -= _ceil_page(take, page_size)
            batch.append((active.rid, take))
            if remaining[active.rid] <= 0:
                ends[active.rid] = k
                active = None
        for c in list(waiting):
            if budget <= 0 or len(batch) >= cap:
                break
            blocked = c.skip_reason not in (None, "pull") or (c.gated and not gate_open)
            if blocked or c.ready_pass > k:
                continue
            need = remaining[c.rid]
            if active is not None:
                # One running partial consumed the chunk budget above.
                continue
            take = min(need, budget)
            starts[c.rid] = k
            remaining[c.rid] -= take
            budget -= _ceil_page(take, page_size)
            batch.append((c.rid, take))
            waiting.remove(c)
            if remaining[c.rid] <= 0:
                ends[c.rid] = k
            else:
                active = c
                break
        batches.append(tuple(batch))
        if not waiting and active is None:
            break
        if k >= horizon:
            if active is not None:
                # No new starts beyond the horizon. Complete the one partial
                # arithmetically, rather than running a long Python pass loop.
                per_pass = min(chunk_tokens, max_input_tokens)
                later = math.ceil(remaining[active.rid] / per_pass)
                ends[active.rid] = k + later
            break
        if not batch and all(c.ready_pass <= k for c in waiting):
            break
    return starts, ends, batches, not waiting


def build_plan(candidates, *, partial=None, chunk_tokens=8192, page_size=64,
               max_input_tokens=None, request_slots=32, horizon=9,
               pass_index=0, reserve_bound=0):
    """Build a forecast from immutable, replicated queue snapshots."""
    options = dict(chunk_tokens=max(1, chunk_tokens), page_size=max(1, page_size),
                   max_input_tokens=max_input_tokens or chunk_tokens,
                   request_slots=max(1, request_slots), horizon=max(1, horizon))
    normal = _simulate(candidates, partial, gate_open=False, **options)
    opened = _simulate(candidates, partial, gate_open=True, **options)
    starts, ends, batches, tail_reached = normal
    rows = list(candidates)
    if partial is not None:
        rows.insert(0, partial)
    # Preserve first-launch order, with stable queue-order ties.
    original = {c.rid: i for i, c in enumerate(rows)}
    first_order = {}
    for batch in batches:
        for rid, _ in batch:
            first_order.setdefault(rid, len(first_order))
    rows.sort(key=lambda c: (starts.get(c.rid, opened[0].get(c.rid, horizon + 1)), first_order.get(c.rid, len(first_order) + original[c.rid])))
    entries = {}
    host_demand = uncached_ahead = 0
    for c in rows:
        p = starts.get(c.rid)
        p0 = opened[0].get(c.rid)
        # Skipped rows contribute host demand but cannot become due issuers.
        host_demand += max(0, c.host_tokens) + max(0, c.pull_tokens)
        entries[c.rid] = PlanEntry(c.rid, p, ends.get(c.rid), p0,
                                  c.uncached_tokens, c.host_tokens, c.device_tokens,
                                  c.pull_tokens, uncached_ahead, host_demand,
                                  c.skip_reason, c.gated, len(entries))
        uncached_ahead += max(0, c.uncached_tokens)
    return LookaheadPlan(entries, tuple(entries), horizon, pass_index, tail_reached,
                         reserve_bound, tuple(candidates),
                         partial, {**options, "pass_index": pass_index,
                                   "reserve_bound": reserve_bound})


@dataclass(frozen=True)
class GateFreeView:
    """A plan's rows with every late-D admission blocker cleared."""

    starts: dict
    order: tuple
    position: dict


LATE_D_SKIPS = ("hold", "intent")


def gate_free_view(plan) -> GateFreeView:
    """The forecast of ``plan``'s rows with every late-D admission blocker
    cleared: the park gate (``gated``) and an unbound row's hold / intent
    skip. Pure: it reruns ``_simulate`` on the plan's replicated integer
    inputs (candidates, partial, options); the plan is not changed. Order:
    by gate-free start, then first-launch order, then queue order (as
    ``build_plan`` orders the real plan). Without late-D blockers the starts
    are the plan's gate-open starts. BWP v2 C6: the prefetch timing input."""
    cleared = tuple(
        replace(
            c,
            gated=False,
            skip_reason=None if c.skip_reason in LATE_D_SKIPS else c.skip_reason,
        )
        for c in plan.candidates
    )
    keys = ("chunk_tokens", "page_size", "max_input_tokens", "request_slots", "horizon")
    options = {k: plan.options[k] for k in keys}
    starts, _, batches, _ = _simulate(cleared, plan.partial, gate_open=True, **options)
    rows = list(cleared)
    if plan.partial is not None:
        rows.insert(0, plan.partial)
    original = {c.rid: i for i, c in enumerate(rows)}
    first_order = {}
    for batch in batches:
        for rid, _ in batch:
            first_order.setdefault(rid, len(first_order))
    horizon = options["horizon"]
    rows.sort(
        key=lambda c: (
            starts.get(c.rid, horizon + 1),
            first_order.get(c.rid, len(first_order) + original[c.rid]),
        )
    )
    order = tuple(c.rid for c in rows)
    return GateFreeView(starts, order, {rid: i for i, rid in enumerate(order)})


def partial_candidate(req) -> PlanCandidate:
    """Remaining work AFTER the batch already launched in this loop pass."""
    end = max(len(req.prefix_indices), int(getattr(getattr(req, "extend_range", None), "end", 0)))
    need = max(0, len(req.full_untruncated_fill_ids) - end)
    return PlanCandidate(req.rid, need, bound=not getattr(req, "pending_bootstrap", False))


class SchedulerLookahead:
    """Scheduler adapter. ``update`` must run at one fixed post-result point.

    ``log`` False: no PREFILL_PLAN_ENTRY / PREFILL_PLAN lines (the private
    plan of the symmetric routing snapshot, prefill_routing_snapshot.py)."""
    def __init__(self, scheduler, log=True):
        self.scheduler = scheduler
        self.log = log
        self.max_reqs = max(1, int(os.getenv("SGLANG_LATE_D_PLAN_MAX_REQS", "32")))
        self.rematch_passes = max(1, int(os.getenv("SGLANG_LATE_D_PLAN_REMATCH_PASSES", "4")))
        self.pass_index = 0
        self._peeks = {}
        self.plan = build_plan([], chunk_tokens=scheduler.chunked_prefill_size,
                               page_size=scheduler.page_size)
        self.cpu_seconds = 0.0
        self._logged_entries = set()

    def candidate(self, req, *, cache=False, ignore_intent=False) -> PlanCandidate:
        # The binder calls this on rank zero only. Its fresh read-only peek
        # must never alter the cache used by the all-rank update cadence.
        cached = self._peeks.get(req.rid) if cache else None
        if cached is None or self.pass_index - cached[0] >= self.rematch_passes or not cached[1].valid():
            peek = peek_prefix(self.scheduler.tree_cache, req)
            if cache:
                self._peeks[req.rid] = (self.pass_index, peek)
        else:
            peek = cached[1]
        uncached = max(1, len(req.origin_input_ids) - peek.device_tokens - peek.host_tokens)
        hint = getattr(req, "kv_pull_hint", None)
        pull = max(0, int(hint.get("expected_hit_tokens", 0)) - peek.device_tokens - peek.host_tokens) if isinstance(hint, dict) else 0
        pending_pull = (req.rid in getattr(self.scheduler.tree_cache, "ongoing_prefetch", {})
                        or bool(getattr(req, "jit_pull_pending", False)))
        # An expected pull is a forecast input, not a cache hit at admission.
        # Its rows move from compute demand to host demand after the ETA.
        if pending_pull:
            uncached = max(1, uncached - pull)
        bound = not getattr(req, "pending_bootstrap", False)
        skip = None
        if not bound and getattr(req, "late_d_hold", False):
            skip = "hold"
        elif not ignore_intent and not bound and uncached > self.scheduler.chunked_prefill_size and not getattr(req, "launched_unbound", False) and not getattr(req, "late_d_start_ok", False):
            skip = "intent"
        elif pending_pull:
            skip = "pull"
        # Deterministic nominal estimate only; no rank-local completion clock.
        nominal = max(0.001, float(os.getenv("SGLANG_JIT_PULL_NOMINAL_PASS_S", "0.38")))
        pull_passes = math.ceil((float(os.getenv("SGLANG_JIT_PULL_SPAN_FIXED_S", "0.78")) + pull / 100000 * float(os.getenv("SGLANG_JIT_PULL_SPAN_S_PER_100K", "0.22"))) / nominal) if pending_pull else 0
        return PlanCandidate(req.rid, uncached, peek.host_tokens, peek.device_tokens,
                             pull if pending_pull else 0, 1 + pull_passes, skip,
                             not bound and bool(getattr(req, "late_d_gate_blocked", False)), bound)

    def update(self):
        start = time.perf_counter()
        self.pass_index += 1
        s = self.scheduler
        queue = [r for r in s.waiting_queue if not r.finished()]
        all_count = len(queue)
        queue = queue[:self.max_reqs]
        candidates = tuple(self.candidate(r, cache=True) for r in queue)
        partial = None
        if s.chunked_req is not None:
            partial = partial_candidate(s.chunked_req)
            if partial.uncached_tokens <= 0:
                partial = None
        pool = int(getattr(s, "max_total_num_tokens", 0) or s.token_to_kv_pool_allocator.size)
        rate = max(1, int(os.getenv("SGLANG_BWP_LEAD_TOKENS_PER_PASS", "131072")))
        margin = max(0, int(os.getenv("SGLANG_BWP_LEAD_MARGIN", "1")))
        horizon = 1 + margin + math.ceil(float(os.getenv("SGLANG_BWP_CHUNKED_HEAD_FRACTION", "0.4")) * pool / rate)
        horizon += max((c.ready_pass - 1 for c in candidates), default=0)
        reserve = int(getattr(s, "late_d_bound_reserve_tokens", 0))
        slots = max(1, int(s.req_to_token_pool.available_size()))
        plan = build_plan(candidates, partial=partial, chunk_tokens=s.chunked_prefill_size,
                          page_size=s.page_size, max_input_tokens=getattr(s, "max_prefill_tokens", None),
                          request_slots=slots, horizon=horizon,
                          pass_index=self.pass_index, reserve_bound=reserve)
        if all_count > self.max_reqs:
            plan = replace(plan, tail_reached=False)
        self.plan = plan
        live = {r.rid for r in queue}
        self._peeks = {rid: v for rid, v in self._peeks.items() if rid in live}
        elapsed = time.perf_counter() - start
        self.cpu_seconds += elapsed
        if self.log and getattr(s.ps, "attn_tp_rank", 0) == 0:
            for e in plan.entries.values():
                if e.rid not in self._logged_entries:
                    self._logged_entries.add(e.rid)
                    logger.info(
                        "PREFILL_PLAN_ENTRY rid=%s pass=%d start=%s end=%s gate_open=%s "
                        "uncached=%d host=%d device=%d pull=%d skip=%s",
                        e.rid, self.pass_index, e.start_pass, e.end_pass,
                        e.gate_open_start_pass, e.uncached_tokens, e.host_tokens,
                        e.device_tokens, e.pull_tokens, e.skip_reason,
                    )
            self._logged_entries.intersection_update(live)
            if self.pass_index % 100 == 1:
                logger.info("PREFILL_PLAN passes=%d cpu_ms_total=%.3f horizon=%d rows=%d",
                            self.pass_index, self.cpu_seconds * 1000, horizon, len(plan.entries))
        return plan
