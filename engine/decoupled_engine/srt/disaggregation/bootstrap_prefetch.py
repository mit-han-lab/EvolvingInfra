"""Bootstrap-window host prefetch for PD prefill workers (opt-in).

While a request waits in the prefill bootstrap queue for the decode side to
allocate memory (KVPoll.Bootstrapping, seconds under load), this component
loads the request's host-cached prefix (local HiCache hits, and peer-pulled
pages once their pull has landed in the host pool) to the device and pins the
loaded path until the request is admitted. Admission then finds the prefix on
the device and runs no synchronous load-back.

Data movement only: no forward runs before bootstrap and no request state
other than the radix lock and ``prefetched_host_hit_length`` changes.

Two submission modes, chosen from the host pool:
- concurrent: no host dedup, or the shared dedup pool
  (``SGLANG_DSA_HOST_DEDUP_FANOUT=shared``). Every rank loads its own rows
  (from its own pool, or from its mapping of the one shared pool) with no
  cross-rank GPU dependency; the op runs on its own load stream (the shared
  loader thread's, in shared mode) next to forwards, nothing waits for it,
  and it is submitted on any pass.
- serial: host dedup, with either fan-out. The op shares the H2D stream with
  admission loads, is submitted only on idle passes, and the schedule stream
  waits for it, so no forward runs concurrently with it.
  - NCCL fan-out: its broadcasts never interleave with forward collectives.
  - CUDA IPC fan-out: followers' copies wait on the source's per-layer events.
    Run concurrently with forwards, that dependency
    is unordered against the forwards' spinning all-reduces: the source's
    gather can sit behind (or compete for SMs with) a collective that a
    follower, stalled behind its wait, never joins. Three ranks spin and one
    is parked. Serial submission removes the overlap.

Safety model (both modes)
- At most one prefetch op is in flight. It is outside the 3-slot producer ring
  and outside ``ack_load_queue``; its completion is MIN-reduced over the cache
  TP group once per pass and then finished with the regular ack handler.
- Every decision reads TP-consistent state (queues, all-reduced polls, tree and
  allocator sizes, the reduced completion flag); the wall-clock TTL is
  MAX-reduced.
- Pins are bounded by a token budget, min(--...-max-tokens, --...-max-fraction
  x device pool), counted with an upper bound (host part + matched device
  prefix) so the budget is a hard cap. A candidate is skipped when loading it
  would leave less than the waiting head's admission need free. Pins are
  released on admission, when the request leaves the queues (abort,
  bootstrap/decode failure), on TTL expiry, and newest-first when the waiting
  head does not fit (the same total-token gate as ``PrefillAdder``). A
  released pin leaves an ordinary LRU-evictable device copy.
- While an op is in flight its device slots are invisible to matching; the
  admission guard defers (one pass at a time) any request whose matched path
  contains a node pending on that op instead of reloading or recomputing it.

Window accounting (per op, scheduler clock, pass granularity): load = submit
to the MIN-reduced completion; window = submit to the first time one of its
requests needs the rows (its admission, or a deferral by the guard above);
margin = window - load (< 0: the remainder was exposed to admission); hidden
fraction = min(load, window) / load, token-weighted. Logged per op and in
the summary, and to the shared-dedup cost log.

M5 "issue before need". All default off; depth 1
with the three env flags unset is the behavior above, unchanged.
- Depth (``--disaggregation-bootstrap-prefetch-depth`` > 1, concurrent
  submission only): up to ``depth`` ops in flight, one request per op,
  oldest first, at most one new op per pass, and a new op only once every
  in-flight op has started on every rank (MIN-reduced with the completion
  flags in the same all-reduce). So at most one prefetch job is queued but
  not started on any rank's loader, and an admission load queues behind at
  most that one and the running one (both uncapped once it queues). The
  limit is decided here from all-reduced values only; the loader never
  drops or defers a job (every rank must run the same op sequence). The op
  of the bootstrap queue's head runs uncapped: it is likely due within one
  admission gap. The pin budget is unchanged (max-tokens / max-fraction).
- Poll first (SGLANG_BWP_POLL_FIRST): the completion poll (one MIN
  all-reduce; moved, not added) runs before admission, so an op that landed
  is matched as device-resident in the same pass instead of deferring.
- Consume in flight (SGLANG_BWP_CONSUME_INFLIGHT): a request whose matched
  path is pending on in-flight ops adopts the pending nodes as its device
  prefix (their slots are allocated and committed; only bytes are landing)
  instead of being deferred. Admission never calls load_back on them, and the
  batch's forward waits for the ops at launch (``hicache_extra_waits``: CPU
  gate on the loader's "submitted", then a stream wait). The adopted rows
  are ``req.inflight_hit_length``: the early send of the cached prefix
  leaves them out (they may still be landing, and the transfer does not wait
  for device work); the sends after the forward, which waited for the ops,
  carry them. Adopting (or deferring on) an op lifts the cap of its job and
  of every background job ahead of it on the loader. The decision reads op
  anchors and tree state only (TP-consistent). The op's load lock stays
  until its ack; the request's own lock covers its forward.
- Consume per part (SGLANG_BWP_CONSUME_PER_PART, 3.3; with consume in flight
  and layer-wise loads, SGLANG_DSA_HOST_DEDUP_LAYER_PARTS > 1): prefetch ops
  are copied in the admission loads' layer parts, with their part events in
  a detached slot (``ack.parts_slot``, outside the producer ring). A forward
  that adopted such an op registers it with the layer counter and waits per
  layer for the op's part of that layer (CPU "part enqueued" check, then a
  stream wait), not for the whole op at launch.
- Adaptive cap (SGLANG_DSA_HOST_DEDUP_BWP_ADAPTIVE_CAP, C2g followers): the
  BWP peer cap applies only while a forward is in flight on the rank and the
  op's remaining bytes fit the cap before its estimated window ends (queue
  position x EWMA of the bootstrap-queue departure gap, minus a margin);
  otherwise the op runs uncapped. Rank-local words: timing only.

Chunked (SGLANG_BWP_CHUNKED, parts B and B1; default off;
needs concurrent submission and depth >= 2). Without it a candidate whose
host part plus matched device prefix exceeds the free pin budget is skipped
(and retried only after RETRY_AFTER_PASSES), so a prefix >= the budget is
never prefetched and admission loads it (a cl=1 batch).
- B: the OLDEST candidate (every request ahead of it in the bootstrap queue
  is pinned or examined for good) that does not fit the budget, or whose
  host part exceeds the headroom, starts a chunk state: its matched host
  path is host-locked (the not-yet-loaded rest cannot be evicted from the
  host pool, by LRU or by the demote order), then it is loaded in
  prefix order in chunks of SGLANG_BWP_CHUNK_TOKENS, one op per chunk, at
  most one new op per pass. A chunk end inside a host-only node splits it at
  the page-aligned limit (UnifiedTreeCore.host_load_chunk_end). Each chunk is
  a pin of its own (path lock on the chunk end) counted against a separate
  head budget, SGLANG_BWP_CHUNKED_HEAD_FRACTION x the device pool, and each
  chunk must fit the headroom (which reserves the running chunked request's
  whole remaining need and the waiting head's admission need). Chunk ops
  run uncapped (the head is due next). Admission adopts the chunks still in
  flight (consume in flight) and loads only a part never issued. The state
  ends (host lock released) on admission, when the request leaves, or when
  TTL / pressure released its last pin. At most one chunk state at a time.
- B1: a candidate skipped only for the pin budget waits for the budget
  epoch to change (a pin released, a chunk state ended, a new bootstrap
  queue head) instead of RETRY_AFTER_PASSES.
- Pressure release (newest pins first, never in-flight ones, never the
  waiting head's) also runs on busy passes.
Every decision reads the same TP-consistent state as above (queue order,
tree, allocator sizes, all-reduced op flags); chunk ends are a function of
the tree, the chunk size and the page size only.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import time
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from decoupled_engine.srt.managers.scheduler_components.phase_timers import timed_phase
from decoupled_engine.srt.utils import get_device_module

if TYPE_CHECKING:
    from decoupled_engine.srt.managers.schedule_batch import Req, ScheduleBatch
    from decoupled_engine.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)
device_module = get_device_module()

BOOTSTRAP_PREFETCH_MODES = ("off", "local", "remote", "both")

# Candidates matched per pass (each costs one radix match).
MAX_EXAMINE_PER_PASS = 16
# A candidate that could not be loaded (budget, headroom, busy node) is
# re-examined after this many scheduler passes.
RETRY_AFTER_PASSES = 64
# TTL is checked (with one small all-reduce) every this many passes while pins
# exist. Passes are lockstep across TP ranks, so the cadence is consistent.
TTL_CHECK_EVERY_PASSES = 32
SUMMARY_INTERVAL_S = 60.0
# Adaptive cap: EWMA weight of a bootstrap-queue departure-gap sample.
GAP_EWMA_ALPHA = 0.2
# Consume in flight: a consuming batch whose forward never waited (no launch)
# falls back to the poll-based window accounting after this many passes.
STALL_GIVEUP_PASSES = 256


@dataclasses.dataclass
class _Pin:
    req: Any
    node_id: Any
    lock_params: Any
    cost: int
    host_tokens: int
    created: float
    op_id: int
    # SGLANG_BWP_CHUNKED: a chunk of the chunk state's request (head budget).
    chunked: bool = False


@dataclasses.dataclass
class _Chunk:
    """SGLANG_BWP_CHUNKED: the one request loaded in chunks."""

    req: Any
    host_node: Any  # host-locked: the matched host path end at the start
    host_params: Any
    created: float
    pins: list = dataclasses.field(default_factory=list)
    cost: int = 0  # device tokens its pins protect (head budget)
    tokens: int = 0  # host tokens issued (and still pinned)
    done: bool = False  # nothing evicted is left on its path


@dataclasses.dataclass
class _Op:
    op_id: int
    ack: Any
    anchors: frozenset
    tokens: int
    submitted: float
    # M5: the op's requests (adaptive window), whether every rank's loader
    # has picked it up (MIN-reduced; depth > 1), whether a batch adopted it.
    reqs: tuple = ()
    started_all: bool = True
    consumed: bool = False


@dataclasses.dataclass
class _Window:
    submitted: float
    tokens: int
    completed: Optional[float] = None
    first_need: Optional[float] = None
    gpu_load_ms: Optional[float] = None
    # Consumed in flight (4C, timed): the consuming forward's stall on the
    # op, measured on the forward stream, is the op's exposure.
    consumed: bool = False
    stall_ms: Optional[float] = None


class _ConsumeWait:
    """One batch's wait for the in-flight ops it adopted (4C), handed to its
    forward through ``hicache_extra_waits``; ``wait`` runs at launch on the
    forward stream (after the I5 capture check).

    Whole ops: a CPU gate on each op's "submitted", then a stream wait on its
    finish event. Ops copied in layer parts (3.3, ``counter`` given): the op's
    detached part slot is registered with the layer counter instead, and the
    forward waits per layer for the part that holds it.

    With ``timed`` (the logging rank) the stream's stall on the ops is
    measured as R7 asks ("arrive" before the CPU gates, "resume" after the
    stream waits): once around the whole-op waits, per part wait otherwise."""

    __slots__ = (
        "ops",
        "timed",
        "created",
        "arrive",
        "resume",
        "counter",
        "whole",
        "parts",
    )

    def __init__(self, ops: List[_Op], timed: bool, created: int, counter=None):
        self.ops = ops
        self.timed = timed
        self.created = created
        self.arrive = None
        self.resume = None
        self.counter = counter
        self.whole: List[_Op] = []
        self.parts: List[_Op] = []
        for op in ops:
            per_part = (
                counter is not None and getattr(op.ack, "parts_slot", None) is not None
            )
            (self.parts if per_part else self.whole).append(op)

    @property
    def layerwise(self) -> bool:
        """Some op is consumed per part (the draft extend must then stay on the
        target forward's stream, L-e)."""
        return bool(self.parts)

    def wait(self, stream=None) -> None:
        dm = get_device_module()
        if self.whole:
            if self.timed:
                self.arrive = dm.Event(enable_timing=True)
                self.arrive.record()
            for op in self.whole:
                BootstrapPrefetcher._consume_wait(op).wait()
            if self.timed:
                self.resume = dm.Event(enable_timing=True)
                self.resume.record()
        for op in self.parts:
            slot = op.ack.parts_slot
            slot.timed = self.timed
            self.counter.add_extra_consumer(slot)

    def stall_ms(self) -> Optional[float]:
        """The forward stream's stall on the ops once every wait passed (and,
        per part, the forward waited for the last part); None before."""
        total = 0.0
        if self.whole:
            if self.resume is None or not self.resume.query():
                return None
            total += float(self.arrive.elapsed_time(self.resume))
        for op in self.parts:
            stall = op.ack.parts_slot.stall_ms()
            if stall is None:
                return None
            total += stall
        return total


class BootstrapPrefetcher:
    def __init__(self, scheduler: Scheduler):
        from decoupled_engine.srt.environ import envs
        from decoupled_engine.srt.runtime_context import get_disagg

        self.scheduler = scheduler
        args = get_disagg()
        self.mode = args.disaggregation_bootstrap_prefetch
        need_mode = getattr(args, "disaggregation_late_d_lead_mode", "work") == "need"
        # SGLANG_SYMMETRIC_BWP: a symmetric worker has no bootstrap queue; its
        # candidates are waiting requests, and its pins leave room for its
        # decode and import roles (_symmetric_reserve).
        self.symmetric = (
            getattr(getattr(scheduler, "disaggregation_mode", None), "value", None)
            == "symmetric"
        )
        # An unset or empty SGLANG_BWP_SOURCE selects the default for the
        # lead mode: "waiting" in need mode, "bootstrap" otherwise. A
        # symmetric worker also defaults to "waiting".
        source = envs.SGLANG_BWP_SOURCE.get().strip().lower()
        default_source = "waiting" if need_mode or self.symmetric else "bootstrap"
        self.source = source or default_source
        if self.source not in ("waiting", "bootstrap"):
            raise ValueError("SGLANG_BWP_SOURCE must be waiting or bootstrap")
        if need_mode and self.source != "waiting":
            raise ValueError("need mode requires SGLANG_BWP_SOURCE=waiting")
        if self.symmetric and self.source != "waiting":
            raise ValueError(
                "a symmetric worker has no bootstrap queue: SGLANG_BWP_SOURCE "
                "must be waiting (or unset)"
            )
        self.lead_tokens = max(1, int(os.getenv("SGLANG_BWP_LEAD_TOKENS_PER_PASS", "131072")))
        self.lead_margin = max(0, int(os.getenv("SGLANG_BWP_LEAD_MARGIN", "1")))
        self.due_op_tokens = max(1, int(os.getenv("SGLANG_BWP_DUE_OP_TOKENS", "262144")))
        self._issue_tokens = None
        self._free_budget_seen = None
        if self.mode not in BOOTSTRAP_PREFETCH_MODES or self.mode == "off":
            raise ValueError(f"Invalid bootstrap prefetch mode {self.mode!r}")
        self.max_tokens_arg = int(args.disaggregation_bootstrap_prefetch_max_tokens)
        self.max_fraction = float(
            getattr(args, "disaggregation_bootstrap_prefetch_max_fraction", 0.0) or 0.0
        )
        self.pool_tokens = int(
            getattr(scheduler, "max_total_num_tokens", None)
            or scheduler.token_to_kv_pool_allocator.size
        )
        self.max_tokens = self.max_tokens_arg
        if self.max_fraction > 0:
            self.max_tokens = min(
                self.max_tokens, int(self.max_fraction * self.pool_tokens)
            )
        self.ttl = float(args.disaggregation_bootstrap_prefetch_ttl)
        self.min_tokens = max(1, int(args.disaggregation_bootstrap_prefetch_min_tokens))
        if need_mode:
            self.min_tokens = max(1, int(scheduler.page_size))
        # M5 "issue before need"; all default off.
        self.depth = int(getattr(args, "disaggregation_bootstrap_prefetch_depth", 1))
        self.poll_first = bool(envs.SGLANG_BWP_POLL_FIRST.get())
        self.consume_inflight = bool(envs.SGLANG_BWP_CONSUME_INFLIGHT.get())
        self.per_part = bool(envs.SGLANG_BWP_CONSUME_PER_PART.get())
        self.adaptive_cap = bool(envs.SGLANG_DSA_HOST_DEDUP_BWP_ADAPTIVE_CAP.get())
        self.cap_margin_s = (
            max(0.0, float(envs.SGLANG_DSA_HOST_DEDUP_BWP_ADAPTIVE_CAP_MARGIN_MS.get()))
            / 1e3
        )
        self.cache = scheduler.tree_cache
        # SGLANG_BWP_CHUNKED (parts B and B1); default off.
        self.chunked = bool(envs.SGLANG_BWP_CHUNKED.get())
        page = max(1, int(getattr(scheduler, "page_size", 1) or 1))
        self.chunk_tokens = (int(envs.SGLANG_BWP_CHUNK_TOKENS.get()) // page) * page
        self.head_max_tokens = int(
            float(envs.SGLANG_BWP_CHUNKED_HEAD_FRACTION.get()) * self.pool_tokens
        )
        self._validate()
        dedup = getattr(self.cache.cache_controller, "host_dedup", None)
        # Concurrent without host dedup or with the shared pool: a dedup
        # fan-out creates a cross-rank dependency that must not overlap a
        # forward (see above); the shared pool has none.
        self.concurrent = dedup is None or bool(
            getattr(dedup, "shared", False) or getattr(dedup, "ce_peer", False)
        )
        self.ipc_fanout = (
            getattr(dedup, "ipc_fanout", None) if dedup is not None else None
        )
        if self.depth > 1 and not self.concurrent:
            raise ValueError(
                "--disaggregation-bootstrap-prefetch-depth > 1 requires concurrent "
                "submission (no host dedup, or the shared / ce_peer fan-out)"
            )
        if self.chunked and (
            not self.concurrent
            or self.depth < 2
            or self.chunk_tokens <= 0
            or self.head_max_tokens <= 0
        ):
            raise ValueError(
                "SGLANG_BWP_CHUNKED requires concurrent submission (no host dedup, "
                "or the shared / ce_peer fan-out), "
                "--disaggregation-bootstrap-prefetch-depth >= 2, "
                "SGLANG_BWP_CHUNK_TOKENS >= one page and "
                "SGLANG_BWP_CHUNKED_HEAD_FRACTION > 0"
            )
        if self.per_part:
            cc = self._cc()
            layerwise = getattr(cc, "_layerwise_loads", None)
            if not (
                self.consume_inflight
                and self.concurrent
                and getattr(cc, "_shared_loader", lambda: None)() is not None
                and layerwise is not None
                and layerwise()
            ):
                raise ValueError(
                    "SGLANG_BWP_CONSUME_PER_PART requires SGLANG_BWP_CONSUME_INFLIGHT, "
                    "the shared / ce_peer loader and layer-wise loads "
                    "(SGLANG_DSA_HOST_DEDUP_LAYER_PARTS > 1)"
                )

        self.pins: List[_Pin] = []
        self.pinned_cost = 0
        # In-flight ops, oldest first (at most `depth`); `inflight` is the
        # oldest one (the only one at depth 1).
        self.ops: List[_Op] = []
        # Anchor node id -> the in-flight op whose load_back pinned it.
        self._anchor_ops: Dict[Any, _Op] = {}
        self._polled = False  # polled before admission this pass
        # Adaptive cap: bootstrap-queue departures (rank-local wall clock).
        self._boot_prev: Optional[set] = None
        self._t_departure: Optional[float] = None
        self.gap_ewma: Optional[float] = None
        # rid -> pass number after which the request may be examined again;
        # None means examined for good (loaded, no host hit, or not selected).
        self._next_try: Dict[str, Optional[int]] = {}
        self._pass = 0
        self._op_counter = 0
        self.stats: Counter = Counter()
        # BWP v2 counters (logging rank only; no decision reads them). Kept
        # apart from ``stats``, which the golden decision traces read.
        self.v2_stats: Counter = Counter()
        self._prev_batch_rids: frozenset = frozenset()
        # op_id -> window accounting until the op's first need (or no pins left).
        self._windows: Dict[int, _Window] = {}
        self.window_stats: Dict[str, float] = Counter()
        self._last_summary = time.monotonic()
        self._log = scheduler.ps.attn_tp_rank == 0
        # Consume in flight: consuming batches whose stall is not resolved
        # yet (window accounting on the logging rank only; timing only).
        self._consume_waits: List[_ConsumeWait] = []
        self._stall_timing = self._log and torch.cuda.is_available()
        # SGLANG_BWP_CHUNKED: the chunk state, and B1's budget epoch (rid ->
        # the epoch it was skipped in; re-examined once the epoch moved).
        self._chunk: Optional[_Chunk] = None
        self._budget_epoch = 0
        self._budget_wait: Dict[str, int] = {}
        self._oldest_rid: Optional[str] = None
        self._init_v2(scheduler)
        if self._log:
            logger.info(
                "Bootstrap prefetch enabled: mode=%s submission=%s fanout=%s "
                "max_tokens=%d (arg %d, fraction %.3g of pool %d) ttl=%.1f "
                "min_tokens=%d",
                self.mode,
                "concurrent" if self.concurrent else "serial-idle",
                (
                    "none"
                    if dedup is None
                    else (
                        "shared"
                        if getattr(dedup, "shared", False)
                        else (
                            "ce_peer"
                            if getattr(dedup, "ce_peer", False)
                            else ("ipc" if self.ipc_fanout else "nccl")
                        )
                    )
                ),
                self.max_tokens,
                self.max_tokens_arg,
                self.max_fraction,
                self.pool_tokens,
                self.ttl,
                self.min_tokens,
            )
            if self.chunked:
                logger.info(
                    "Bootstrap prefetch chunked: chunk_tokens=%d head_max_tokens=%d "
                    "(fraction %.3g of pool %d) budget_retry=epoch "
                    "pressure_release=every pass",
                    self.chunk_tokens,
                    self.head_max_tokens,
                    float(envs.SGLANG_BWP_CHUNKED_HEAD_FRACTION.get()),
                    self.pool_tokens,
                )
            if self._m5_on():
                logger.info(
                    "Bootstrap prefetch M5: depth=%d poll_first=%d "
                    "consume_inflight=%d per_part=%d adaptive_cap=%d margin_ms=%.0f",
                    self.depth,
                    int(self.poll_first),
                    int(self.consume_inflight),
                    int(self.per_part),
                    int(self.adaptive_cap),
                    self.cap_margin_s * 1e3,
                )

    def _init_v2(self, scheduler) -> None:
        """BWP v2 parts. Each needs
        SGLANG_BWP_V2=1 and its own flag; all default off (today's rules)."""
        from decoupled_engine.srt.environ import envs

        v2 = bool(envs.SGLANG_BWP_V2.get())
        # C3: a symmetric worker's prefetcher owns its plan (no
        # scheduler.late_d_lookahead alias; the symmetric loop updates it
        # through update_plan, the routing snapshot reads plan_view).
        self.v2_private_plan = (
            v2 and self.symmetric and bool(envs.SGLANG_BWP_SYM_PRIVATE_PLAN.get())
        )
        # C4: admission-first memory (waiting source).
        self.v2_admission_safe = (
            v2 and self.waiting_source and bool(envs.SGLANG_BWP_ADMISSION_SAFE.get())
        )
        # C5: loader backlog guard (waiting source, concurrent submission).
        self.v2_fifo_guard = (
            v2
            and self.waiting_source
            and self.concurrent
            and bool(envs.SGLANG_BWP_FIFO_GUARD.get())
        )
        nominal = float(os.getenv("SGLANG_JIT_PULL_NOMINAL_PASS_S", "0.38"))
        self.fifo_budget = int(
            max(0, int(envs.SGLANG_BWP_FIFO_GUARD_LOAD_TOKENS_PER_S.get())) * nominal
        )
        self._fifo_batch_tokens = 0
        self._fifo_prev_rids: frozenset = frozenset()
        self._fifo_blocked = False
        # C6: gate-free due pass (the due time ignores the late-D gate),
        # waiting source.
        self.v2_gate_free = (
            v2 and self.waiting_source and bool(envs.SGLANG_BWP_GATE_FREE.get())
        )
        if self.v2_gate_free and not (
            self.v2_admission_safe and (self.v2_fifo_guard or not self.concurrent)
        ):
            raise ValueError(
                "SGLANG_BWP_GATE_FREE requires SGLANG_BWP_ADMISSION_SAFE, and "
                "SGLANG_BWP_FIFO_GUARD with concurrent submission (the guards "
                "come before the part that adds candidates)"
            )
        self._gf = None  # (plan, GateFreeView) of the last plan read
        # C7: growth re-check (waiting source); rid -> host tokens seen when
        # the row was examined for good.
        self.v2_growth = (
            v2 and self.waiting_source and bool(envs.SGLANG_BWP_GROWTH_RECHECK.get())
        )
        self._host_seen: Dict[str, int] = {}
        # C8: fresh-row step plan (waiting source); set during one step only.
        self.v2_step_plan = (
            v2 and self.waiting_source and bool(envs.SGLANG_BWP_STEP_PLAN.get())
        )
        if self.v2_step_plan and not (
            self.v2_admission_safe and (self.v2_fifo_guard or not self.concurrent)
        ):
            raise ValueError(
                "SGLANG_BWP_STEP_PLAN requires SGLANG_BWP_ADMISSION_SAFE, and "
                "SGLANG_BWP_FIFO_GUARD with concurrent submission"
            )
        self.step_plan_max_fresh = max(1, int(envs.SGLANG_BWP_STEP_PLAN_MAX_FRESH.get()))
        self._step_plan = None
        self._own_plan = None
        if self.v2_private_plan:
            from decoupled_engine.srt.disaggregation.lookahead_plan import SchedulerLookahead

            self._own_plan = SchedulerLookahead(scheduler)
        if v2:
            # Every TP rank (GPU smoke check: the line on every rank).
            logger.info(
                "BWP v2: tp_rank=%s source=%s symmetric=%d private_plan=%d "
                "admission_safe=%d fifo_guard=%d fifo_budget=%d gate_free=%d "
                "growth_recheck=%d step_plan=%d",
                getattr(getattr(scheduler, "ps", None), "attn_tp_rank", "?"),
                self.source,
                int(self.symmetric),
                int(self.v2_private_plan),
                int(self.v2_admission_safe),
                int(self.v2_fifo_guard),
                self.fifo_budget,
                int(self.v2_gate_free),
                int(self.v2_growth),
                int(self.v2_step_plan),
            )

    def _m5_on(self) -> bool:
        return (
            self.depth > 1
            or self.poll_first
            or self.consume_inflight
            or self.per_part
            or self.adaptive_cap
        )

    @property
    def inflight(self) -> Optional[_Op]:
        """The oldest in-flight op (the only one at depth 1), or None."""
        return self.ops[0] if self.ops else None

    # ------------------------------------------------------------------ setup
    def _validate(self) -> None:
        from decoupled_engine.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        s = self.scheduler
        problems = []
        if not s.enable_hierarchical_cache:
            problems.append("--enable-hierarchical-cache")
        if not isinstance(self.cache, UnifiedRadixCache):
            problems.append("the unified radix cache")
        elif self.cache.cache_controller is None:
            problems.append("a HiCache controller")
        elif getattr(self.cache, "buffer_pipeline", None) is not None:
            problems.append("cache host-memory mode (not buffer_only)")
        if s.ps.pp_size != 1:
            problems.append("pp_size == 1")
        if s.ps.attn_cp_size != 1:
            problems.append("attn_cp_size == 1")
        if getattr(s, "enable_dp_attention", False):
            problems.append("no DP attention")
        if s.is_hybrid_swa or s.is_hybrid_ssm:
            problems.append("a non-SWA, non-mamba model")
        if s.enable_lora:
            problems.append("no LoRA")
        if not s.chunked_prefill_size:
            problems.append("chunked prefill")
        if os.environ.get("SGLANG_HICACHE_PREFETCH_NEXT_BATCH", "").lower() in (
            "1",
            "true",
        ):
            problems.append("SGLANG_HICACHE_PREFETCH_NEXT_BATCH off")
        if self.max_tokens <= 0:
            problems.append(
                "--disaggregation-bootstrap-prefetch-max-tokens > 0 (and a "
                "max-fraction that leaves a positive budget)"
            )
        if self.depth < 1:
            problems.append("--disaggregation-bootstrap-prefetch-depth >= 1")
        if problems:
            raise ValueError(
                "--disaggregation-bootstrap-prefetch requires " + ", ".join(problems)
            )

    # ------------------------------------------------------------- utilities
    def _cc(self):
        return self.cache.cache_controller

    def _poll_inflight(self) -> None:
        """Finish in-flight ops once every rank completed them (TP-consistent).

        Depth 1: one MIN all-reduce of the op's completion flag. Depth > 1:
        one MIN all-reduce of [done flags..., started flags...] (started: the
        rank's loader picked the job up; bounds queued prefetch jobs)."""
        ops = self.ops
        if not ops:
            return
        if self.depth == 1:
            op = ops[0]
            flag = torch.tensor([int(op.ack.finish_event.query())], dtype=torch.int64)
            self.cache._all_reduce(flag, torch.distributed.ReduceOp.MIN)
            if int(flag.item()):
                self._finish(op)
            return
        n = len(ops)
        flags = torch.tensor(
            [int(op.ack.finish_event.query()) for op in ops]
            + [int(self._started(op.ack)) for op in ops],
            dtype=torch.int64,
        )
        self.cache._all_reduce(flags, torch.distributed.ReduceOp.MIN)
        vals = flags.tolist()
        for i, op in enumerate(list(ops)):
            op.started_all = bool(vals[n + i] or vals[i])
            if vals[i]:
                self._finish(op)

    def poll_before_admission(self) -> None:
        """SGLANG_BWP_POLL_FIRST: called by the event loops before batch
        formation (the same point on every rank); this pass's ``step`` then
        skips its poll. No-op with the flag off.

        BWP v2 C4: then the admission-first pressure release for the
        formation that follows (no collective; replicated inputs)."""
        if self.poll_first:
            self._poll_inflight()
            self._polled = True
        if getattr(self, "v2_admission_safe", False) and self.pins:
            self._release_admission_first(within=0)

    @staticmethod
    def _started(ack) -> bool:
        """This rank's loader has picked the op up (True without a loader)."""
        job = getattr(ack.finish_event, "job", None)
        return True if job is None else bool(getattr(job, "started", True))

    def _finish(self, op: _Op) -> None:
        self.cache._settle_load_ack(op.ack)
        self.ops.remove(op)
        for anchor in op.anchors:
            if self._anchor_ops.get(anchor) is op:
                del self._anchor_ops[anchor]
        w = self._windows.get(op.op_id)
        if w is not None:
            w.completed = time.monotonic()
            w.gpu_load_ms = self._gpu_load_ms(op.ack)
            self._close_window(op.op_id)
        if self._log:
            logger.info(
                "Bootstrap prefetch acked: op_id=%d tokens=%d wall_ms=%.1f",
                op.op_id,
                op.tokens,
                (time.monotonic() - op.submitted) * 1000.0,
            )

    def _lift_cap(self, op: _Op) -> None:
        """A request waits for the in-flight op: let a C2g follower finish it,
        and any background job ahead of it on the loader, uncapped (the cap
        only spares forwards while nobody waits)."""
        job = getattr(op.ack.finish_event, "job", None)
        loader = getattr(getattr(self._cc(), "host_dedup", None), "loader", None)
        if (
            job is not None
            and loader is not None
            and hasattr(loader, "lift_background_cap")
        ):
            loader.lift_background_cap(job)

    @staticmethod
    def _gpu_load_ms(ack) -> Optional[float]:
        """This rank's device time of the op (ack events), if timed."""
        if not getattr(ack, "timing_enabled", False):
            return None
        try:
            return float(ack.start_event.elapsed_time(ack.finish_event))
        except Exception:  # noqa: BLE001 - accounting only
            return None

    def _need_window(self, op_id: int) -> None:
        w = self._windows.get(op_id)
        if w is not None and w.first_need is None:
            w.first_need = time.monotonic()

    def _close_window(self, op_id: int) -> None:
        """Account the op once it completed and was first needed; drop it if
        it completed and no pin of it is left to be admitted."""
        w = self._windows.get(op_id)
        if w is None or w.completed is None:
            return
        if w.first_need is None:
            if any(p.op_id == op_id for p in self.pins):
                return
            del self._windows[op_id]
            self.window_stats["unused_ops"] += 1
            self.window_stats["unused_tokens"] += w.tokens
            return
        if w.consumed and w.stall_ms is None:
            return  # the consuming forward's stall is not measured yet
        del self._windows[op_id]
        load = max(w.completed - w.submitted, 1e-6) * 1e3
        if w.stall_ms is not None:
            # Consumed in flight: exposed = the forward stream's stall on it.
            window = load - min(max(w.stall_ms, 0.0), load)
            self.window_stats["stall_timed_ops"] += 1
            self.window_stats["stall_ms"] += w.stall_ms
        else:
            window = (w.first_need - w.submitted) * 1e3
        margin = window - load
        hidden = min(max(window, 0.0), load) / load
        st = self.window_stats
        st["ops"] += 1
        st["hidden_ops"] += int(margin >= 0)
        st["tokens"] += w.tokens
        st["hidden_tokens"] += w.tokens * hidden
        st["load_ms"] += load
        st["window_ms"] += window
        st["exposed_ms"] += max(-margin, 0.0)
        st["exposed_ms_max"] = max(st["exposed_ms_max"], -margin)
        st["margin_ms_min"] = (
            margin if st["ops"] == 1 else min(st["margin_ms_min"], margin)
        )
        if w.gpu_load_ms is not None:
            st["gpu_timed_ops"] += 1
            st["gpu_load_ms"] += w.gpu_load_ms
        from decoupled_engine.srt.mem_cache.ce_2d_load import COST

        # Cost-log fields are non-negative: the margin is split in two.
        COST.add("bwp_load_ms", load)
        COST.add("bwp_window_ms", max(window, 0.0))
        COST.add("bwp_margin_ms", max(margin, 0.0))
        COST.add("bwp_exposed_ms", max(-margin, 0.0))
        COST.add("bwp_hidden_pct", 100.0 * hidden)
        COST.add("bwp_tokens", float(w.tokens))
        COST.add("bwp_hidden_tokens", w.tokens * hidden)
        if self._log:
            logger.info(
                "Bootstrap prefetch window: op_id=%d tokens=%d load_ms=%.1f "
                "gpu_load_ms=%s window_ms=%.1f margin_ms=%.1f hidden_pct=%.1f%s",
                op_id,
                w.tokens,
                load,
                "-" if w.gpu_load_ms is None else f"{w.gpu_load_ms:.1f}",
                window,
                margin,
                100.0 * hidden,
                "" if w.stall_ms is None else f" consumed=1 stall_ms={w.stall_ms:.1f}",
            )

    def window_summary(self) -> str:
        st = self.window_stats
        ops = st["ops"]
        return (
            f"window_ops={int(ops)} hidden_ops={int(st['hidden_ops'])} "
            f"unused_ops={int(st['unused_ops'])} "
            f"hidden_tok_pct={100.0 * st['hidden_tokens'] / max(st['tokens'], 1):.1f} "
            f"load_ms_mean={st['load_ms'] / max(ops, 1):.1f} "
            f"gpu_load_ms_mean={st['gpu_load_ms'] / max(st['gpu_timed_ops'], 1):.1f} "
            f"window_ms_mean={st['window_ms'] / max(ops, 1):.1f} "
            f"margin_ms_min={st['margin_ms_min']:.1f} "
            f"exposed_ms_sum={st['exposed_ms']:.1f} exposed_ms_max={st['exposed_ms_max']:.1f}"
        )

    def _admission_need(self, req: Req) -> int:
        """Free tokens admission's total-token gate needs for ``req``
        (``PrefillAdder.add_one_req``: uncached input including the host part,
        plus clipped max_new_tokens, plus one page)."""
        from decoupled_engine.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS

        uncached = len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
        max_new = 0
        sp = getattr(req, "sampling_params", None)
        if sp is not None and getattr(sp, "max_new_tokens", None) is not None:
            max_new = min(
                max(sp.max_new_tokens - len(getattr(req, "output_ids", ())), 0),
                CLIP_MAX_NEW_TOKENS,
            )
        return max(uncached, 1) + max_new + self.scheduler.page_size

    @property
    def waiting_source(self) -> bool:
        return getattr(self, "source", "bootstrap") == "waiting"

    def _plan_owner(self):
        """The SchedulerLookahead whose plan this prefetcher reads: its own
        (SGLANG_BWP_SYM_PRIVATE_PLAN) or scheduler.late_d_lookahead."""
        own = getattr(self, "_own_plan", None)
        if own is not None:
            return own
        return getattr(self.scheduler, "late_d_lookahead", None)

    def update_plan(self) -> None:
        """The symmetric loop's post-result plan update (every TP rank, the
        same loop point): the plan this prefetcher reads."""
        owner = self._plan_owner()
        if owner is not None:
            owner.update()

    def plan_view(self):
        """The private plan (read-only use; SGLANG_BWP_SYM_PRIVATE_PLAN), or
        None when the plan is scheduler.late_d_lookahead's."""
        own = getattr(self, "_own_plan", None)
        return own.plan if own is not None else None

    def _plan(self):
        step = getattr(self, "_step_plan", None)
        if step is not None:
            return step  # C8: this step's private plan
        return getattr(self._plan_owner(), "plan", None)

    def _plan_offset(self) -> int:
        """Passes between a plan's start 1 and the next formation: one for
        the post-result plan (a formation ran since), none for C8's step
        plan (made after the formation)."""
        return 0 if getattr(self, "_step_plan", None) is not None else 1

    def _candidate_queue(self):
        if not self.waiting_source:
            return self.scheduler.disagg_prefill_bootstrap_queue.queue
        return self.scheduler.waiting_queue

    def _plan_order(self):
        queue = self._candidate_queue()
        if not self.waiting_source:
            return list(queue)
        plan = self._plan()
        if plan is None:
            return []
        by_rid = {r.rid: r for r in queue}
        return [by_rid[rid] for rid in plan.order if rid in by_rid]

    def _plan_entry(self, req):
        plan = self._plan()
        return plan.entries.get(req.rid) if plan is not None else None

    def _plan_start(self, req):
        entry = self._plan_entry(req)
        # The plan was made after the previous result. One formation ran
        # before this pass's BWP step, so its next-start offset is now p - 1.
        return None if entry is None or entry.start_pass is None else max(0, entry.start_pass - self._plan_offset())

    # ----------------------------------------------------- BWP v2 seams (C1)
    # Only these read the plan's forecast for prefetch timing and the
    # admission order for memory. Today's rules: the real plan for both.
    def _due(self, req) -> Optional[int]:
        """Passes until ``req``'s forecast start, for selection, cap lifts,
        uncapping at submission and the adaptive-cap deadline; None: no
        start forecast (never selected). SGLANG_BWP_GATE_FREE: the start with
        every late-D blocker cleared (no late-D gate input is read)."""
        if getattr(self, "v2_gate_free", False):
            view = self._gate_free_view()
            start = None if view is None else view.starts.get(req.rid)
            # Same offset as _plan_start: formation k ran before this step.
            return None if start is None else max(0, start - self._plan_offset())
        return self._plan_start(req)

    def _gate_free_view(self):
        """C6: the plan's gate-free forecast (computed once per plan)."""
        plan = self._plan()
        if plan is None:
            return None
        cached = self._gf
        if cached is not None and cached[0] is plan:
            return cached[1]
        from decoupled_engine.srt.disaggregation.lookahead_plan import gate_free_view

        view = gate_free_view(plan)
        self._gf = (plan, view)
        return view

    def _selection_rows(self, plan) -> list:
        """(rid, entry, position) in the order the waiting-source selection
        walks (demand ahead accumulates in it); position goes to
        ``_submit_op`` (logging). SGLANG_BWP_GATE_FREE: gate-free order."""
        if getattr(self, "v2_gate_free", False):
            view = self._gate_free_view()
            if view is not None:
                return [
                    (rid, plan.entries[rid], i)
                    for i, rid in enumerate(view.order)
                    if rid in plan.entries
                ]
        return [(rid, plan.entries[rid], plan.entries[rid].order) for rid in plan.order]

    def _position(self, req) -> int:
        """``req``'s position in the selection order (-1: not in the plan)."""
        if getattr(self, "v2_gate_free", False):
            view = self._gate_free_view()
            return -1 if view is None else view.position.get(req.rid, -1)
        entry = self._plan_entry(req)
        return entry.order if entry is not None else -1

    def _admission_rows(self, within: int = 1) -> list:
        """The waiting requests the next formation admits per the real plan:
        admittable (``_admittable``, in plan order) with a forecast start
        within ``within`` passes. The running chunked request is not in it.
        Memory only (reserve, pressure release); never prefetch timing."""
        rows = []
        for r in self._plan_order():
            if self._admittable(r):
                start = self._plan_start(r)
                if start is not None and start <= within:
                    rows.append(r)
        return rows

    def _planned_need(self, req, candidate=None):
        entry = self._plan_entry(req)
        from decoupled_engine.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS
        if entry is None:
            full = getattr(req, "full_untruncated_fill_ids", None)
            if full is None:
                full = req.origin_input_ids
            tokens = max(1, len(full) - len(req.prefix_indices))
        else:
            tokens = max(1, entry.uncached_tokens + (0 if req is candidate else entry.host_tokens))
        sp = getattr(req, "sampling_params", None)
        output = max(0, int(getattr(sp, "max_new_tokens", 0) or 0) - len(getattr(req, "output_ids", ())))
        return tokens + min(output, CLIP_MAX_NEW_TOKENS) + self.scheduler.page_size

    def _waiting_head(self):
        return next((r for r in self._plan_order() if self._admittable(r)), None)

    def _bound_reserve(self, candidate=None) -> int:
        """The late-D bound-work reserve (the plan's), recomputed without a
        bound candidate's own host restore."""
        s = self.scheduler
        bound_reserve = int(getattr(self._plan(), "reserve_bound", 0))
        if candidate is not None and not candidate.pending_bootstrap:
            # Recompute the same bound-work set, leaving this candidate's
            # own host restore out. Subtracting it from a scalar maximum can
            # accidentally erase the reserve needed by a different request.
            bound_reserve = max([2 * s.chunked_prefill_size] + [
                self._planned_need(r, candidate) for r in s.waiting_queue
                if not r.pending_bootstrap
            ])
        return bound_reserve

    def _today_reserve(self, candidate=None) -> int:
        """Today's admission reserve without the symmetric part: the largest
        of two chunks, the running chunked request's remaining need, the
        waiting head's planned need and the bound reserve."""
        s = self.scheduler
        head = self._waiting_head()
        partial = s.chunked_req
        return max(
            2 * s.chunked_prefill_size,
            self._admission_need(partial) if partial is not None else 0,
            self._planned_need(head, candidate) if head is not None else 0,
            self._bound_reserve(candidate),
        )

    def _reserve(self, candidate=None):
        if getattr(self, "v2_admission_safe", False):
            reserve = self._admission_first_reserve(candidate)
        else:
            reserve = self._today_reserve(candidate)
        if self.symmetric:
            reserve += self._symmetric_reserve()
        return reserve

    def _symmetric_reserve(self) -> int:
        """SGLANG_SYMMETRIC_BWP: device tokens the worker's other roles need
        from the same pool, kept free of pins (and freed by pressure release):

        - running decodes: each one's remaining output, clipped and scaled by
          the scheduler's new_token_ratio, as ``PrefillAdder`` reserves it
          (``_get_running_request_total_token_offset``);
        - imports: the prompts of bound imports (admitted / held, allocated
          soon) and the late-D claim's need (``ImportQueue.load_view``).
          Deferred imports (no ADMIT yet) are not counted.

        Inputs are the running batch, new_token_ratio and the import queue,
        which are the same on every TP rank."""
        from decoupled_engine.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS

        s = self.scheduler
        ratio = float(getattr(s, "new_token_ratio", 1.0))
        decode = 0.0
        running = getattr(s, "running_batch", None)
        for r in getattr(running, "reqs", None) or ():
            left = r.sampling_params.max_new_tokens - len(r.output_ids)
            decode += min(left, CLIP_MAX_NEW_TOKENS) * ratio
        imports = 0
        queue = getattr(s, "import_queue", None)
        if queue is not None:
            view = queue.load_view()
            imports = int(view.get("import_bound_tokens", 0)) + int(
                view.get("claim_tokens", 0)
            )
        return max(0, int(decode)) + imports

    # ------------------------------------- BWP v2 C4: admission-first memory
    def _faces_park_gate(self, req) -> bool:
        """``req``'s admission passes the late-D park gate (need mode on the
        scheduler, unbound, not launched unbound: late_d_need_skip)."""
        return (
            bool(getattr(self.scheduler, "_late_d_need_enabled", False))
            and bool(getattr(req, "pending_bootstrap", False))
            and not getattr(req, "launched_unbound", False)
        )

    def _row_term(self, req, candidate=None, admitted=False) -> int:
        """Free tokens ``req``'s admission takes from the room the next
        formation's rows share, its own lock aside (``_lock_take``).

        The candidate: its pin loads its host part and locks its device
        prefix (the pin's room, ``_room``), which is what its own admission
        would take, so the pin leaves its total-token check and the rows
        after it alike. Not admitted next: only its park gate is tighter
        (unbound: need without its host part, plus its evictable prefix; a
        bound candidate keeps today's planned need). Admitted next
        (``admitted``): also what the adder deducts for it afterwards (see
        below, without its host load).

        Any other row: the largest of (a) what PrefillAdder deducts for it
        before the next row's checks (its host load, the page-rounded
        extend, clipped max_new and one page, ``_update_prefill_budget``),
        (b) what its own total-token check needs (planned need + 1, strict)
        and (c), unbound, its park-gate terms. With each row's lock take
        and the bound reserve once, the sum bounds what every row of the set
        needs, in any formation order."""
        own = req is candidate
        entry = self._plan_entry(req)
        gate = 0
        if self._faces_park_gate(req):
            from decoupled_engine.srt.disaggregation.late_d_need import (
                evictable_prefix_tokens,
                park_gate_room,
                remaining_need,
            )

            if entry is None:
                need = remaining_need(req)
            else:
                need = max(1, entry.uncached_tokens + (0 if own else entry.host_tokens))
            gate = park_gate_room(need, evictable_prefix_tokens(req))
            if own and not admitted:
                return gate
        elif own and not admitted:
            return self._planned_need(req, candidate)
        take = self._adder_take(req, candidate)
        return max(take, self._planned_need(req, candidate) + 1, gate)

    def _adder_take(self, req, candidate=None) -> int:
        """What PrefillAdder deducts from the room when it admits ``req``
        (``_update_prefill_budget`` plus the load of its host part): host
        part (none for the candidate, whose pin loads it), page-rounded
        extend, clipped max_new and one page. From the plan entry."""
        from decoupled_engine.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS

        entry = self._plan_entry(req)
        page = max(1, int(self.scheduler.page_size))
        if entry is None:
            full = getattr(req, "full_untruncated_fill_ids", None)
            if full is None:
                full = req.origin_input_ids
            rest = max(1, len(full) - len(req.prefix_indices))
            load, extend = 0, rest  # not split: all of it as extend
        else:
            load = 0 if req is candidate else max(0, entry.host_tokens)
            extend = max(1, entry.uncached_tokens)
        sp = getattr(req, "sampling_params", None)
        max_new = min(max(0, int(getattr(sp, "max_new_tokens", 0) or 0)), CLIP_MAX_NEW_TOKENS)
        return load + -(-extend // page) * page + max_new + page

    def _first_due(self):
        """The waiting request with the earliest due pass (selection order
        breaks ties), or None."""
        plan = self._plan()
        if plan is None:
            return None
        by_rid = {r.rid: r for r in self.scheduler.waiting_queue}
        best, best_due = None, None
        for rid, _, _ in self._selection_rows(plan):
            r = by_rid.get(rid)
            if r is None or r.finished():
                continue
            due = self._due(r)
            if due is not None and (best_due is None or due < best_due):
                best, best_due = r, due
        return best

    def _admission_extras(self, rows, candidate=None) -> list:
        """The rows besides ``rows`` (the next formation's) whose room a pin
        must leave: the first row by due pass and the candidate."""
        extras = []
        for extra in (self._first_due(), candidate):
            if (
                extra is not None
                and not any(r is extra for r in rows)
                and not any(r is extra for r in extras)
            ):
                extras.append(extra)
        return extras

    def _rows_total(self, rows, extras=(), candidate=None) -> int:
        """The room ``rows`` (admitted next, in any order) and ``extras``
        need together with the running chunked request: each row's term,
        each row's lock take (shared nodes once; the candidate's own path is
        in its pin's room), the chunked request's whole need, and the bound
        reserve once if any row faces the park gate. Without today's floor
        and without the symmetric part."""
        s = self.scheduler
        seen = set(n.id for n in (self._device_path(candidate) or ()))
        total = 0
        gate = False
        for admitted, group in ((True, rows), (False, extras)):
            for r in group:
                total += self._row_term(r, candidate, admitted)
                if r is not candidate:
                    total += self._lock_take(r, seen)
                gate = gate or self._faces_park_gate(r)
        if s.chunked_req is not None:
            total += self._admission_need(s.chunked_req)
        if gate:
            total += self._bound_reserve(candidate)
        return total

    def _rows_reserve(self, rows, candidate=None) -> int:
        """``_rows_total`` of the next formation's rows, at least today's
        reserve. Without the symmetric part."""
        return max(self._today_reserve(candidate), self._rows_total(rows, (), candidate))

    def _admission_first_reserve(self, candidate=None) -> int:
        rows = self._admission_rows(1)
        extras = self._admission_extras(rows, candidate)
        return max(
            self._today_reserve(candidate), self._rows_total(rows, extras, candidate)
        )

    def _device_path(self, req):
        """The nodes of ``req``'s last matched device path (its
        ``last_node`` up to the root), deepest first; None when it was never
        matched or its last node left the tree. Read only."""
        core = getattr(self.cache, "tree_core", None)
        nid = getattr(req, "last_node", None)
        if core is None or nid is None or not hasattr(core, "node_by_id"):
            return None
        try:
            node = core.node_by_id(nid)
        except KeyError:
            return None
        root = core.root_node
        path = []
        while node is not None and node is not root:
            path.append(node)
            node = node.parent
        return path

    def _lock_take(self, req, seen: set) -> int:
        """Device tokens ``req``'s admission lock takes out of the evictable
        room (``PrefillAdder._req_inc_lock_ref``): the unlocked nodes of its
        last matched device path that ``seen`` does not hold yet (nodes of
        rows counted before, and of the candidate, whose pin locks them),
        plus any device prefix the plan's peek saw beyond that path. A row
        never matched (or whose last node left the tree) counts its plan
        device prefix. Inputs: match results, lock counts and plan peeks,
        the same on every TP rank."""
        from decoupled_engine.srt.mem_cache.unified_cache.components.tree_component import (
            ComponentType,
        )

        entry = self._plan_entry(req)
        planned = max(0, int(entry.device_tokens)) if entry is not None else 0
        path = self._device_path(req)
        if path is None:
            return planned
        total = unlocked = 0
        for node in path:
            n = len(node.key)
            total += n
            if node.id in seen:
                continue
            seen.add(node.id)
            cd = node.component_data[ComponentType.FULL]
            if cd.lock_ref == 0 and cd.value is not None:
                unlocked += n
        return unlocked + max(0, planned - total)

    def _unprotected_prefix(self, req) -> int:
        """Device tokens of ``req``'s matched device prefix that no lock holds
        (a pin locks them, so they leave the evictable room too)."""
        path = self._device_path(req)
        if path is None:
            return len(getattr(req, "prefix_indices", ()))
        from decoupled_engine.srt.mem_cache.unified_cache.components.tree_component import (
            ComponentType,
        )

        total = 0
        for node in path:
            cd = node.component_data[ComponentType.FULL]
            if cd.lock_ref == 0 and cd.value is not None:
                total += len(node.key)
        return total

    def _room(self, req, tokens: int, first: bool) -> int:
        """Free tokens a pin of ``tokens`` loaded tokens takes: with C4 also
        the unlocked device prefix that a first pin of ``req`` locks."""
        if first and getattr(self, "v2_admission_safe", False):
            return tokens + self._unprotected_prefix(req)
        return tokens

    def _headroom_free(self, candidate) -> int:
        """C6: the headroom of a reserve without the real plan's admission
        rows (gate-invariant): the first row by gate-free due pass and the
        candidate, the running chunked request, two chunks, the bound
        reserve, the symmetric reserve."""
        s = self.scheduler
        partial = s.chunked_req
        total = self._rows_total((), self._admission_extras((), candidate), candidate)
        base = max(
            2 * s.chunked_prefill_size,
            self._admission_need(partial) if partial is not None else 0,
            self._bound_reserve(candidate),
        )
        reserve = max(base, total)
        if self.symmetric:
            reserve += self._symmetric_reserve()
        return (
            s.token_to_kv_pool_allocator.available_size()
            + self.cache.evictable_size()
            - reserve
        )

    def _memory_refusal(self, req, room: int) -> None:
        """C6: a pin refused only for the room of the rows the real plan
        admits next (which rows the late-D gate lets in) is an
        admission-first skip, the one place a gate may matter. Counter and
        test event only."""
        if getattr(self, "v2_gate_free", False) and room <= self._headroom_free(req):
            self.v2_stats["admission_first_skips"] += 1
            self._v2_event("admission_first_skip", req.rid, int(room))

    def _release_admission_first(self, within: int = 1) -> None:
        """C4 pressure release (at the step, ``within`` 1; before formation,
        ``within`` 0). Victims: pins of waiting rows that the next formation
        does not admit, not in flight, latest due first (no start: first),
        newest first among equals, the first-due row's pins last. Phase 1
        releases toward today's target (decodes, imports, today's head) as
        today, the first-due row's pins only if that reaches the target;
        phase 2 releases the rest only if that makes the next
        formation's room fit (reason admission_first), counting what a
        release really frees (``_release_plan``) and skipping pins whose
        release frees nothing. Phase 3 keeps the park-gate room
        of gated rows that the release would otherwise ignore: the
        first-due row (any victim, its own pins last) and a row with a
        landed pin of its own (only its own pins), again only if that makes
        the room fit (reason admission_first)."""
        s = self.scheduler
        allocator = s.token_to_kv_pool_allocator

        def free():
            return allocator.available_size() + self.cache.evictable_size()

        rows = self._admission_rows(within)
        head = rows[0] if rows else (self._waiting_head() or s.chunked_req)
        sym = self._symmetric_reserve() if self.symmetric else 0
        today = self._today_reserve(head) + sym  # today's release target
        # The admitted rows' whole needs (host parts included: a row's own
        # landed pin is already out of its plan entry's host part).
        need = max(today, self._rows_reserve(rows) + sym)
        inflight = {op.op_id for op in self.ops}
        first = self._first_due()
        gates = self._gate_rooms(rows, first, sym, inflight)
        if free() >= need and all(free() >= target for _, target, _ in gates):
            return
        protected = {id(r) for r in rows}
        if s.chunked_req is not None:
            protected.add(id(s.chunked_req))
        waiting = {id(r) for r in s.waiting_queue}
        victims = [
            (i, pin)
            for i, pin in enumerate(self.pins)
            if id(pin.req) in waiting
            and id(pin.req) not in protected
            and pin.op_id not in inflight
        ]

        def key(item):
            i, pin = item
            due = self._due(pin.req)
            return (pin.req is first, -(1 << 30) if due is None else -due, -i)

        victims = [pin for _, pin in sorted(victims, key=key)]
        for pin in victims:  # phase 1: today's target, partial as today
            if free() >= today:
                break
            if pin.req is first:
                # D1: the first-due row's pins go only if that
                # reaches today's target; a partial release would drop the
                # prefetch and admit nothing.
                live = {id(p) for p in self.pins}
                gain, _ = self._release_plan(
                    [p for p in victims if p.req is first and id(p) in live]
                )
                if free() + gain < today:
                    break
            self._release(pin, "pressure")
        # Phase 2: the next formation's rows; phase 3: the gate rooms.
        targets = [(None, need, False)] + gates
        for row, target, own_only in targets:
            if free() >= target:
                continue
            live = {id(p) for p in self.pins}
            gain, useful = self._release_plan(
                [
                    p
                    for p in victims
                    if id(p) in live and (not own_only or p.req is row)
                ]
            )
            if free() + gain < target:
                # Releasing them would not make that room fit.
                self.v2_stats["admission_first_blocked"] += 1
                continue
            for pin in useful:
                if free() >= target:
                    break
                self._release(pin, "admission_first")

    def _gate_rooms(self, rows, first, sym: int, inflight) -> list:
        """Gated rows whose late-D park-gate room the release
        keeps, besides the next formation's ``rows``: the first-due row, and
        each row with a landed pin of its own (its own pin keeps its device
        prefix locked, so with the unified tree's zero evictable-prefix term
        the pin closes its gate earlier than BWP off would). Each as (row,
        target, own_only): the room its screen in the next formation reads,
        the symmetric reserve, the running chunked request's whole need,
        the deduction and lock take of the rows admitted before it (queue
        order), its park-gate terms and the bound reserve. Only rows that
        face the park gate and are gated in the real plan."""
        from decoupled_engine.srt.disaggregation.late_d_need import (
            evictable_prefix_tokens,
            park_gate_room,
        )

        s = self.scheduler
        extras = [first] if first is not None else []
        for pin in self.pins:
            if pin.op_id not in inflight and not any(q is pin.req for q in extras):
                extras.append(pin.req)
        if not extras:
            return []
        order = {id(q): i for i, q in enumerate(s.waiting_queue)}
        protected = {id(r) for r in rows}
        base = sym + self._bound_reserve()
        if s.chunked_req is not None:
            base += self._admission_need(s.chunked_req)
        out = []
        for row in extras:
            if id(row) in protected or id(row) not in order or row.finished():
                continue
            entry = self._plan_entry(row)
            if entry is None or not entry.gated or not self._faces_park_gate(row):
                continue
            seen = set()
            ahead = sum(
                self._adder_take(r) + self._lock_take(r, seen)
                for r in rows
                if order.get(id(r), -1) < order[id(row)]
            )
            need = max(1, entry.uncached_tokens + entry.host_tokens)
            gate = park_gate_room(need, evictable_prefix_tokens(row))
            out.append((row, base + ahead + gate, row is not first))
        return out

    def _release_plan(self, pins) -> tuple:
        """(gain, useful) of releasing every pin of ``pins``: the device
        tokens of the nodes on their lock paths that only these pins lock
        (another lock on a node, such as a protected row's pin, keeps it out
        of the evictable room), and those of ``pins`` (same order) that hold
        such a node; releasing any other frees nothing. Lock
        counts are the same on every TP rank. A pin whose node left the tree
        counts its cost (defensive)."""
        core = getattr(self.cache, "tree_core", None)
        if core is None or not hasattr(core, "node_by_id"):
            return sum(p.cost for p in pins), list(pins)
        from decoupled_engine.srt.mem_cache.unified_cache.components.tree_component import (
            ComponentType,
        )

        full = ComponentType.FULL
        root = core.root_node
        held = Counter()
        paths = []
        for pin in pins:
            try:
                node = core.node_by_id(pin.node_id)
            except KeyError:
                paths.append(None)
                continue
            params = pin.lock_params
            skip = params.skip_lock_node_ids.get(full, ()) if params is not None else ()
            path = []
            while node is not None and node is not root:
                if node.id not in skip:  # the lock skipped it (evicted then)
                    path.append(node)
                    held[node.id] += 1
                node = node.parent
            paths.append(path)
        gain = 0
        freed = set()
        useful = []
        for pin, path in zip(pins, paths):
            if path is None:
                gain += pin.cost
                useful.append(pin)
                continue
            frees = False
            for node in path:
                cd = node.component_data[full]
                if cd.value is None or cd.lock_ref > held[node.id]:
                    continue
                frees = True
                if node.id not in freed:
                    freed.add(node.id)
                    gain += len(cd.value)
            if frees:
                useful.append(pin)
        return gain, useful

    # ------------------------------------------ BWP v2 C5: loader backlog guard
    def _fifo_note_batch(self, batch) -> None:
        """Host tokens this pass's formation queued for admission loads: the
        batch's new rows (not a chunked request's next chunk). Replicated
        inputs (batch, match results)."""
        if batch is None:
            self._fifo_batch_tokens = 0
            self._fifo_prev_rids = frozenset()
            return
        mode = getattr(batch, "forward_mode", None)
        if mode is not None and not mode.is_extend():
            self._fifo_batch_tokens = 0
            return
        self._fifo_batch_tokens = sum(
            int(getattr(r, "host_hit_length", 0) or 0)
            for r in batch.reqs
            if r.rid not in self._fifo_prev_rids
        )
        self._fifo_prev_rids = frozenset(r.rid for r in batch.reqs)

    def _fifo_ok(self, tokens: int, rid=None) -> bool:
        """C5: the loader backlog with a new op of ``tokens`` fits one
        nominal forward (False: count it, and the pass submits nothing)."""
        if not getattr(self, "v2_fifo_guard", False):
            return True
        backlog = self._fifo_batch_tokens + sum(op.tokens for op in self.ops)
        if backlog + tokens <= self.fifo_budget:
            return True
        self._fifo_blocked = True
        self.v2_stats["fifo_guard_skips"] += 1
        self._v2_event("fifo_guard_skip", rid, int(backlog), int(tokens))
        return False

    # ------------------------------------------- BWP v2 C8: step plan
    def _build_step_plan(self):
        """C8: the plan of this step when the formation left fresh rows (no
        entry in the post-result plan) among the first max_reqs waiting rows:
        the post-result candidates still waiting (their cached peeks), the
        fresh rows (read-only peeks, not stored), in queue order, and the
        running chunked request after the formation; the plan's options.
        None: no fresh row, or more than SGLANG_BWP_STEP_PLAN_MAX_FRESH."""
        owner = self._plan_owner()
        plan = getattr(owner, "plan", None)
        if (
            plan is None
            or not hasattr(owner, "candidate")
            or not getattr(plan, "options", None)
            or getattr(plan, "candidates", None) is None
        ):
            return None
        s = self.scheduler
        cut = int(getattr(owner, "max_reqs", 0) or 0) or len(s.waiting_queue)
        queue = [r for r in s.waiting_queue if not r.finished()][:cut]
        fresh = [r for r in queue if r.rid not in plan.entries]
        if not fresh:
            return None
        if len(fresh) > self.step_plan_max_fresh:
            self.v2_stats["step_plan_skips"] += 1
            return None
        from decoupled_engine.srt.disaggregation.lookahead_plan import (
            build_plan,
            partial_candidate,
        )

        known = {c.rid: c for c in plan.candidates}
        cands = tuple(
            known[r.rid] if r.rid in known else owner.candidate(r) for r in queue
        )
        partial = None
        if s.chunked_req is not None:
            partial = partial_candidate(s.chunked_req)
            if partial.uncached_tokens <= 0:
                partial = None
        self.v2_stats["step_plans"] += 1
        self.v2_stats["step_plan_fresh_rows"] += len(fresh)
        return build_plan(cands, partial=partial, **plan.options)

    # ------------------------------------------- BWP v2 C7: growth re-check
    def _seen(self, rid, host: int) -> None:
        """C7: the host part seen when ``rid`` was examined for good."""
        if getattr(self, "v2_growth", False):
            self._host_seen[rid] = int(host)

    def _grown(self, rid, entry) -> bool:
        """C7: ``rid``'s plan host part grew by max(min tokens, one page)
        since it was examined for good (plan peeks: replicated)."""
        if not getattr(self, "v2_growth", False) or rid not in self._host_seen:
            return False
        page = max(1, int(getattr(self.scheduler, "page_size", 1) or 1))
        return entry.host_tokens >= self._host_seen[rid] + max(self.min_tokens, page)

    def _v2_event(self, kind, *args) -> None:
        """A v2 decision without a call of its own (tests record it)."""

    def _head_need(self) -> tuple:
        """(request, tokens) admission needs next: the running chunked request's
        whole remaining need, else the first admittable waiting request;
        (None, 0) if none.

        A chunked request was admitted against its whole input, and its later
        chunks allocate without a headroom check (``PrefillAdder.add_chunked_req``
        falls back to a full chunk when the budget is exhausted). So a pin must
        not take any of its remaining tokens, not just the next chunk's (a
        98,816-token concurrent pin under a ~380k-token chunked request ran a
        400k-token pool out mid-request: "Prefill out of memory").
        """
        s = self.scheduler
        if self.waiting_source:
            head = self._waiting_head() or s.chunked_req
            return head, self._reserve(head)
        if s.chunked_req is not None:
            return s.chunked_req, max(
                self._admission_need(s.chunked_req),
                s.chunked_prefill_size + s.page_size,
            )
        head = next((r for r in s.waiting_queue if self._admittable(r)), None)
        if head is None:
            return None, 0
        return head, self._admission_need(head)

    def _headroom(self, candidate=None) -> int:
        s = self.scheduler
        reserve = self._reserve(candidate) if self.waiting_source else max(2 * s.chunked_prefill_size, self._head_need()[1])
        return (
            s.token_to_kv_pool_allocator.available_size()
            + self.cache.evictable_size()
            - reserve
        )

    def _selected(self, req: Req) -> bool:
        if self.mode == "both":
            return True
        hinted = isinstance(getattr(req, "kv_pull_hint", None), dict)
        return hinted if self.mode == "remote" else not hinted

    def _pin_of(self, req: Req) -> Optional[_Pin]:
        for pin in self.pins:
            if pin.req is req:
                return pin
        return None

    def _release(self, pin: _Pin, reason: str) -> None:
        if reason == "admitted":
            self._need_window(pin.op_id)
        self.pins.remove(pin)
        st = None
        if pin.chunked:
            st = self._chunk
            if st is not None and any(p is pin for p in st.pins):
                st.pins = [p for p in st.pins if p is not pin]
                st.cost -= pin.cost
                st.tokens -= pin.host_tokens
                if reason != "admitted":
                    st.done = False  # its rows may be evicted again: reload
            else:
                st = None
        else:
            self.pinned_cost -= pin.cost
            if self.chunked and pin.cost > 0:
                self._budget_epoch += 1  # B1: the pin budget freed
        self._close_window(pin.op_id)
        self.cache.dec_lock_ref(pin.node_id, pin.lock_params)
        if reason != "admitted":
            pin.req.prefetched_host_hit_length = st.tokens if st is not None else 0
        self.stats[f"released_{reason}"] += 1
        self.stats[f"released_{reason}_tokens"] += pin.host_tokens
        if self._log:
            logger.info(
                "Bootstrap prefetch released: rid=%s reason=%s tokens=%d cost=%d "
                "age_s=%.3f op_id=%d",
                pin.req.rid,
                reason,
                pin.host_tokens,
                pin.cost,
                time.monotonic() - pin.created,
                pin.op_id,
            )
        if st is not None and not st.pins and reason in (
            "ttl",
            "pressure",
            "admission_first",
        ):
            # TTL / pressure / admission-first (BWP v2 C4) took its
            # last chunk: drop the state (host lock); the request is retried
            # as an ordinary candidate later.
            self._next_try[st.req.rid] = self._pass + RETRY_AFTER_PASSES
            self._end_chunk(reason)

    def _end_chunk(self, reason: str) -> None:
        """SGLANG_BWP_CHUNKED: end the chunk state and release its host lock
        (its pins are released on their own)."""
        st = self._chunk
        if st is None:
            return
        self._chunk = None
        self._budget_epoch += 1  # B1: the head budget is free again
        try:
            self.cache.dec_host_lock_ref(st.host_node, st.host_params)
        except KeyError:  # the node is gone (cache flushed): nothing held
            self.stats["chunk_host_unlock_missing"] += 1
        self.stats[f"chunk_end_{reason}"] += 1
        if self._log:
            logger.info(
                "Bootstrap prefetch chunked end: rid=%s reason=%s tokens=%d "
                "cost=%d pins=%d age_s=%.3f",
                st.req.rid,
                reason,
                st.tokens,
                st.cost,
                len(st.pins),
                time.monotonic() - st.created,
            )

    # -------------------------------------------------------- scheduler hooks
    def has_pins(self) -> bool:
        return bool(self.pins) or bool(self.ops)

    def retry_after_landing(self, rid: str) -> None:
        """A deferred peer pull of ``rid`` landed (disaggregation/jit_pull.py):
        examine the request again, even if it was examined for good before."""
        self._next_try.pop(rid, None)

    def defer_admission(self, req: Req) -> bool:
        """Called by admission after ``init_next_round_input``.

        True while an in-flight prefetch op owns a node on this request's
        matched path; the request stays queued and is retried next pass. With
        SGLANG_BWP_CONSUME_INFLIGHT the request adopts the op's pending nodes
        instead and is admitted (``_adopt``); it is deferred only when it
        cannot adopt them. Reads only state updated at a fixed point of the
        pass (no collective here).
        """
        if self.consume_inflight:
            req.bwp_consume_ops = None
            req.inflight_hit_length = 0
        if not self.ops:
            return False
        node_id = getattr(req, "best_match_node", None)
        if node_id is None:
            return False
        if self.consume_inflight and self._adopt(req):
            return False
        core = self.cache.tree_core
        node = core.node_by_id(node_id)
        root = core.root_node
        while node is not None and node is not root:
            op = self._anchor_ops.get(node.load_back_pending_id)
            if op is not None:
                self.stats["deferred"] += 1
                self._need_window(op.op_id)  # the load is now exposed
                self._lift_cap(op)
                if self._log:
                    logger.info(
                        "Bootstrap prefetch deferred: rid=%s op_id=%d",
                        req.rid,
                        op.op_id,
                    )
                return True
            node = node.parent
        return False

    def _adopt(self, req: Req) -> bool:
        """SGLANG_BWP_CONSUME_INFLIGHT (4C): let ``req`` consume the in-flight
        ops on its matched path instead of waiting for their acks. False when
        it cannot (the caller then defers it, as without the flag).

        The device match stops in front of load-back-pending nodes, so the
        request's device prefix ends at ``req.last_node`` and the pending
        nodes under it count as host hit. Their device slots are allocated
        and committed (``commit_load_back``); only the bytes are landing.
        Adopting extends the device prefix through the run of nodes pending
        on in-flight prefetch ops right under ``last_node``: admission never
        calls load_back on them (so the double-anchor assert in
        commit_load_back cannot fire), its own load_back covers only evicted
        nodes further down, and the batch's forward waits for the ops (step).
        In-flight nodes already above ``last_node`` (inside the device
        prefix) get the wait only. ``req.inflight_hit_length`` is what the
        early send must leave out: the adopted rows, or the whole prefix
        if an in-flight node sits inside the device prefix (defensive; the
        device match stops in front of pending nodes). Inputs: tree state and
        op anchors, both TP-consistent.
        """
        core = self.cache.tree_core
        root = core.root_node
        last = core.node_by_id(req.last_node) if req.last_node is not None else root
        below = []  # (node, in-flight op or None) under `last`, deepest first
        node = core.node_by_id(req.best_match_node)
        while node is not None and node is not last and node is not root:
            below.append((node, self._anchor_ops.get(node.load_back_pending_id)))
            node = node.parent
        if node is not last:
            return False  # `last` is not on the matched path: keep deferring
        ops: Dict[int, _Op] = {}
        above = False  # an in-flight node at or above `last` (device prefix)
        while node is not None and node is not root:
            op = self._anchor_ops.get(node.load_back_pending_id)
            if op is not None:
                ops[op.op_id] = op
                above = True
            node = node.parent
        below.reverse()
        k = 0
        while k < len(below) and below[k][1] is not None:
            ops[below[k][1].op_id] = below[k][1]
            k += 1
        if any(op is not None for _, op in below[k:]):
            return False  # in flight behind a node it cannot adopt
        if not ops:
            return True  # nothing in flight on its path
        adopted = 0
        if k:
            new_last = below[k - 1][0]
            extra = core.collect_full_device_indices(new_last.id, last.id)
            adopted = len(extra)
            # With M4's Q4 page bookkeeping both parts carry CPU page ids
            # (cpu_page_book.py) that admission reads: keep them. Q4 off: a
            # plain torch.cat (the page size is read only for tagged parts).
            from decoupled_engine.srt.mem_cache.cpu_page_book import cat_tagged

            req.prefix_indices = cat_tagged(
                [req.prefix_indices, extra],
                lambda: core.page_size,
                "bwp adopt prefix",
            )
            req.last_node = new_last.id
            req.host_hit_length = max(0, req.host_hit_length - adopted)
            req.cache_protected_len = len(req.prefix_indices)
        # Rows the early send must not read yet (they may still be landing).
        req.inflight_hit_length = len(req.prefix_indices) if above else adopted
        req.bwp_consume_ops = list(ops.values())
        for op in req.bwp_consume_ops:
            self._lift_cap(op)  # someone waits for its rows now
        self.stats["adopted_reqs"] += 1
        self.stats["adopted_tokens"] += adopted
        if self._log:
            logger.info(
                "Bootstrap prefetch adopted: rid=%s op_ids=%s tokens=%d",
                req.rid,
                [op.op_id for op in req.bwp_consume_ops],
                adopted,
            )
        return True

    def _attach_consume_waits(self, batch: Optional[ScheduleBatch]) -> None:
        """Hand the forward the in-flight ops its requests adopted (4C), once
        per op; the forward waits for them at launch (tp_worker)."""
        if batch is None:
            return
        live = {op.op_id: op for op in self.ops}
        attached: List[_Op] = []
        for req in batch.reqs:
            for op in getattr(req, "bwp_consume_ops", None) or ():
                if live.pop(op.op_id, None) is None:
                    continue  # acked on every rank already, or already attached
                attached.append(op)
                self._need_window(op.op_id)
                if not op.consumed:
                    op.consumed = True
                    self.stats["consumed_ops"] += 1
                    self.stats["consumed_tokens"] += op.tokens
        if not attached:
            return
        cw = _ConsumeWait(
            attached,
            timed=self._stall_timing,
            created=self._pass,
            counter=self._cc().layer_done_counter if self.per_part else None,
        )
        if cw.parts:
            self.stats["consumed_per_part_ops"] += len(cw.parts)
        waits = getattr(batch, "hicache_extra_waits", None)
        if waits is None:
            waits = batch.hicache_extra_waits = []
        waits.append(cw)
        if cw.timed:
            self._consume_waits.append(cw)
            for op in attached:
                w = self._windows.get(op.op_id)
                if w is not None:
                    w.consumed = True

    def _resolve_consume_waits(self) -> None:
        """Window accounting of consumed ops (logging rank, timing only): the
        consuming forward's measured stall replaces the exposure the poll
        would give (completion is seen a pass late; the forward may not have
        waited at all). A batch whose forward never waited falls back."""
        for cw in list(self._consume_waits):
            try:
                stall = cw.stall_ms()
            except Exception:  # noqa: BLE001 - accounting only
                stall, cw.resume, cw.created = None, None, -STALL_GIVEUP_PASSES
            if stall is None:
                if (
                    cw.resume is not None
                    or self._pass - cw.created <= STALL_GIVEUP_PASSES
                ):
                    continue
            self._consume_waits.remove(cw)
            for op in cw.ops:
                w = self._windows.get(op.op_id)
                if w is None:
                    continue
                if stall is None:
                    w.consumed = False
                else:
                    w.stall_ms = stall if w.stall_ms is None else max(w.stall_ms, stall)
                self._close_window(op.op_id)

    @staticmethod
    def _consume_wait(op: _Op):
        """What a consuming forward waits on at launch for a whole ``op``: its
        finish event (CPU gate on "submitted", then one stream wait). Ops
        copied in layer parts (3.3) are waited per layer instead, through
        their detached part slot (``_ConsumeWait``)."""
        return op.ack.finish_event

    @timed_phase("bwp_step")
    def step(self, batch: Optional[ScheduleBatch]) -> None:
        """Called once per scheduler pass, after batch formation."""
        try:
            self._step(batch)
        finally:
            self._step_plan = None  # C8: never kept past its step

    def _step(self, batch: Optional[ScheduleBatch]) -> None:
        self._pass += 1
        if self._polled:
            self._polled = False  # polled before admission this pass
        else:
            self._poll_inflight()
        self._sweep(batch)
        if getattr(self, "v2_fifo_guard", False):
            self._fifo_note_batch(batch)
        if self._log:
            self._count_admissions(batch)
        if getattr(self, "v2_step_plan", False):
            self._step_plan = self._build_step_plan()
        if self.waiting_source:
            for op in self.ops:
                if any((p := self._due(r)) is not None and p <= 1 for r in op.reqs):
                    self._lift_cap(op)
        if self.consume_inflight:
            self._attach_consume_waits(batch)
            if self._consume_waits:
                self._resolve_consume_waits()
        if self.ttl > 0 and self.pins and self._pass % TTL_CHECK_EVERY_PASSES == 0:
            self._release_expired()
        if self.pins and (batch is None or self.chunked):
            # SGLANG_BWP_CHUNKED: on busy passes too (a head pin is large).
            self._release_for_pressure()
        if self._can_submit(batch):
            self._select_and_submit()
        if self._log:
            self._count_pass()
        if self.adaptive_cap:
            self._update_adaptive(batch)
        if self._log and time.monotonic() - self._last_summary >= SUMMARY_INTERVAL_S:
            self._last_summary = time.monotonic()
            logger.info(
                "Bootstrap prefetch summary: mode=%s pins=%d pinned=%d/%d %s %s%s%s",
                self.mode,
                len(self.pins),
                self.pinned_cost,
                self.max_tokens,
                " ".join(f"{k}={v}" for k, v in sorted(self.stats.items()))
                + (
                    f" chunk_pinned={self._chunk.cost if self._chunk else 0}"
                    f"/{self.head_max_tokens}"
                    if self.chunked
                    else ""
                ),
                self.window_summary(),
                (
                    f" inflight={len(self.ops)} gap_ewma_ms="
                    + ("-" if self.gap_ewma is None else f"{self.gap_ewma * 1e3:.0f}")
                    + f" stall_timed_ops={int(self.window_stats['stall_timed_ops'])}"
                    + f" stall_ms_sum={self.window_stats['stall_ms']:.1f}"
                    if self._m5_on()
                    else ""
                ),
                self._v2_summary(),
            )

    # ------------------------------------------------- BWP v2 counters (C2)
    # Logging rank only, logging only: no decision reads them.
    def _v2_summary(self) -> str:
        st = self.v2_stats
        if not st:
            return ""
        return " v2: " + " ".join(f"{k}={v}" for k, v in sorted(st.items()))

    @staticmethod
    def _gate_blocked(entry) -> bool:
        """The plan entry is held back by a late-D admission rule: the park
        gate (``gated``) or an unbound row's hold / intent skip."""
        return bool(entry.gated) or entry.skip_reason in ("hold", "intent")

    def _count_gate_skip(self, req, entry, demand: int) -> None:
        """A row held by a late-D rule that the selection skipped (no
        forecast start, or not due): ``due_gate_skip`` when it is due
        without the gate (a violation of the gate-free rule), else, today's
        rules only, ``gate_hold_skip``. Today's rules: the plan's gate-open
        start. SGLANG_BWP_GATE_FREE: the gate-free view's start, read here
        apart from ``_due``, so the count shows any later change that makes the
        selection's timing read a gate input again."""
        lead = math.ceil(demand / self.lead_tokens)
        if self.v2_gate_free:
            view = self._gate_free_view()
            free = None if view is None else view.starts.get(req.rid)
        else:
            free = entry.gate_open_start_pass
        if free is not None and max(0, free - self._plan_offset()) - self.lead_margin <= lead:
            self.v2_stats["due_gate_skip"] += 1
        elif not self.v2_gate_free:
            self.v2_stats["gate_hold_skip"] += 1

    def _count_admissions(self, batch) -> None:
        """Host tokens that admission loads synchronously (the batch's new
        rows), split into rows with no entry in the plan this step reads
        (fresh: admitted at their first formation) and rows that waited."""
        if batch is None:
            self._prev_batch_rids = frozenset()
            return
        mode = getattr(batch, "forward_mode", None)
        if mode is not None and not mode.is_extend():
            return
        plan = self._plan() if self.waiting_source else None
        for r in batch.reqs:
            if r.rid in self._prev_batch_rids:
                continue  # the next chunk of a chunked request
            waited = plan is None or r.rid in plan.entries
            key = "waited" if waited else "fresh"
            self.v2_stats[f"sync_reqs_{key}"] += 1
            self.v2_stats[f"sync_tokens_{key}"] += int(
                getattr(r, "host_hit_length", 0) or 0
            )
        self._prev_batch_rids = frozenset(r.rid for r in batch.reqs)

    def _admission_need_now(self) -> int:
        """Free tokens the next formation needs (pin_blocked_head)."""
        if getattr(self, "v2_admission_safe", False):
            rows = self._admission_rows(1)
            sym = self._symmetric_reserve() if self.symmetric else 0
            return self._rows_reserve(rows) + sym
        return self._head_need()[1]

    def _count_pass(self) -> None:
        """Per pass: waiting rows the plan has not seen yet (fresh_rows), and
        passes whose admission need does not fit while pins hold device
        memory (pin_blocked_head)."""
        s = self.scheduler
        if self.waiting_source:
            plan = getattr(self._plan_owner(), "plan", None)  # post-result
            if plan is not None:
                cut = int(getattr(self._plan_owner(), "max_reqs", 0) or 0)
                queue = [r for r in s.waiting_queue if not r.finished()]
                self.v2_stats["fresh_rows"] += sum(
                    1 for r in (queue[:cut] if cut else queue) if r.rid not in plan.entries
                )
        if self.pins:
            free = s.token_to_kv_pool_allocator.available_size() + self.cache.evictable_size()
            if free < self._admission_need_now():
                self.v2_stats["pin_blocked_head"] += 1

    # ------------------------------------------------------- adaptive cap
    def _loader(self):
        return getattr(getattr(self._cc(), "host_dedup", None), "loader", None)

    def _update_adaptive(self, batch: Optional[ScheduleBatch]) -> None:
        """Adaptive-cap inputs, once per pass (rank-local; timing only): the
        rank's forward-busy word and each in-flight op's deadline word."""
        now = time.monotonic()
        boot = self._candidate_queue()
        if not self.waiting_source:
            self._track_departures(boot, now)
        loader = self._loader()
        if loader is None or not hasattr(loader, "set_forward_busy"):
            return
        # A forward is in flight (or launches right after this step): this
        # pass's batch, or an earlier launch whose result is not processed.
        loader.set_forward_busy(
            batch is not None or bool(getattr(self.scheduler, "result_queue", None))
        )
        position = {id(r): i for i, r in enumerate(boot)}
        for op in self.ops:
            job = getattr(op.ack.finish_event, "job", None)
            if job is not None and hasattr(job, "deadline_ns"):
                job.deadline_ns.value = self._deadline_ns(op, position, now)

    def _deadline_ns(self, op: _Op, position: Dict[int, int], now: float) -> int:
        """When the op must be done to stay capped: its first request's queue
        position x the EWMA bootstrap-queue departure gap, minus the margin
        (CLOCK_MONOTONIC ns). 0 = no slack (uncapped): the queue head, an op
        a batch already waits for, a request that left the queue, or no gap
        observed yet ("when unsure, uncapped")."""
        if self.waiting_source:
            if op.consumed or not op.reqs:
                return 0
            starts = [self._due(r) for r in op.reqs]
            if any(p is None or p <= 1 for p in starts):
                return 0
            binder = getattr(getattr(self.scheduler, "disagg_prefill_bootstrap_queue", None), "late_d_binder", None)
            lead = getattr(binder, "work", None)
            duration = getattr(lead, "iter_s", None) or float(os.getenv("SGLANG_JIT_PULL_NOMINAL_PASS_S", "0.38"))
            window = min(starts) * duration - self.cap_margin_s
            return int((now + window) * 1e9) if window > 0 else 0
        if op.consumed or self.gap_ewma is None or not op.reqs:
            return 0
        ps = [position.get(id(r)) for r in op.reqs]
        if any(p is None for p in ps):
            return 0
        window = min(ps) * self.gap_ewma - self.cap_margin_s
        return int((now + window) * 1e9) if window > 0 else 0

    def _track_departures(self, boot, now: float) -> None:
        cur = {id(r) for r in boot}
        prev, self._boot_prev = self._boot_prev, cur
        if prev is None:
            return
        left = len(prev - cur)
        if not left:
            return
        if self._t_departure is not None:
            gap = (now - self._t_departure) / left
            self.gap_ewma = (
                gap
                if self.gap_ewma is None
                else (1.0 - GAP_EWMA_ALPHA) * self.gap_ewma + GAP_EWMA_ALPHA * gap
            )
        self._t_departure = now

    # ------------------------------------------------------------ internals
    def _sweep(self, batch: Optional[ScheduleBatch]) -> None:
        s = self.scheduler
        boot = self._candidate_queue()
        if self.pins:
            admitted = {id(r) for r in batch.reqs} if batch is not None else set()
            queued = {id(r) for r in boot}
            queued.update(id(r) for r in s.waiting_queue)
            for pin in list(self.pins):
                if id(pin.req) in admitted:
                    self._release(pin, "admitted")
                elif id(pin.req) not in queued:
                    self._release(pin, "gone")
        if self._next_try:
            live = {r.rid for r in boot}
            for rid in [rid for rid in self._next_try if rid not in live]:
                del self._next_try[rid]
        if self._host_seen:
            live = {r.rid for r in boot}
            for rid in [rid for rid in self._host_seen if rid not in live]:
                del self._host_seen[rid]
        if self.chunked:
            self._sweep_chunked(batch, boot)

    def _sweep_chunked(self, batch: Optional[ScheduleBatch], boot) -> None:
        """SGLANG_BWP_CHUNKED: end the chunk state when its request was
        admitted or left the queues. Advance the budget epoch on a new
        bootstrap queue head (B1, bootstrap source), or when the free budget
        grows (waiting source)."""
        s = self.scheduler
        st = self._chunk
        if st is not None:
            if batch is not None and any(r is st.req for r in batch.reqs):
                self._end_chunk("admitted")
            elif not any(r is st.req for r in boot) and not any(
                r is st.req for r in s.waiting_queue
            ):
                self._end_chunk("gone")
        if self.waiting_source:
            free = self.max_tokens - self.pinned_cost
            previous = getattr(self, "_free_budget_seen", None)
            if previous is not None and free > previous:
                self._budget_epoch += 1
            self._free_budget_seen = free
        else:
            head = boot[0].rid if boot else None
            if head != self._oldest_rid:
                self._oldest_rid = head
                self._budget_epoch += 1
        if self._budget_wait:
            live = {r.rid for r in boot}
            for rid in [rid for rid in self._budget_wait if rid not in live]:
                del self._budget_wait[rid]

    def _release_expired(self) -> None:
        now = time.monotonic()
        flags = torch.tensor(
            [int(now - pin.created >= self.ttl) for pin in self.pins],
            dtype=torch.int64,
            device="cpu",
        )
        # Wall clocks differ across ranks: any rank's expiry expires the pin.
        self.cache._all_reduce(flags, torch.distributed.ReduceOp.MAX)
        for pin, expired in list(zip(self.pins, flags.tolist())):
            if expired:
                self._release(pin, "ttl")

    def _admittable(self, req: Req) -> bool:
        if req.finished() or (req.pending_bootstrap and not self.waiting_source):
            return False
        if self.waiting_source:
            entry = self._plan_entry(req)
            if entry is None or entry.gated or entry.skip_reason is not None or entry.start_pass is None:
                return False
        return req.rid not in self.cache.ongoing_prefetch

    def _release_for_pressure(self) -> None:
        if getattr(self, "v2_admission_safe", False):
            self._release_admission_first(within=1)
            return
        s = self.scheduler
        head, need = self._head_need()
        if head is None and not self.symmetric:
            return
        # Symmetric: decodes and bound imports need room even when no waiting
        # request is admittable (need is then _reserve(None)).
        allocator = s.token_to_kv_pool_allocator
        waiting = {id(r) for r in s.waiting_queue}
        # Pins of requests still bootstrapping go first (newest first); pins of
        # bootstrapped requests are about to be consumed by admission.
        order = sorted(
            range(len(self.pins)),
            key=lambda i: (id(self.pins[i].req) in waiting, -i),
        )
        # The head's own pin is what makes it cheap to admit: keep it.
        if self.waiting_source:
            plan = self._plan()
            positions = {rid: i for i, rid in enumerate(plan.order)} if plan is not None else {}
            order = sorted(range(len(self.pins)), key=lambda i: (positions.get(self.pins[i].req.rid, 1 << 30), i), reverse=True)
        victims = [self.pins[i] for i in order if self.pins[i].req is not head and (not self.waiting_source or self.pins[i].req is not s.chunked_req)]
        inflight = {op.op_id for op in self.ops}
        for pin in victims:
            if allocator.available_size() + self.cache.evictable_size() >= need:
                break
            if pin.op_id in inflight:
                # Its slots stay locked by the load until the ack; releasing
                # now frees nothing. Reconsidered on a later pass.
                continue
            self._release(pin, "pressure")

    def _can_submit(self, batch) -> bool:
        s = self.scheduler
        if self._cc().load_queue:
            return False
        if self.depth == 1:
            if self.ops:
                return False
        elif len(self.ops) >= self.depth or not all(op.started_all for op in self.ops):
            # At most `depth` in flight, and at most one prefetch job queued
            # but not started on any rank's loader (all-reduced flags).
            return False
        if not self._candidate_queue() and not (
            self.chunked and self._chunk is not None and not self._chunk.done
        ):
            return False
        if self.concurrent:
            return True
        # Serial mode: only when the GPU would otherwise idle. This reads the
        # real plan's admittable rows, so with SGLANG_BWP_GATE_FREE a closed
        # late-D gate can only add an idle pass (see environ.py).
        if batch is not None or s.chunked_req is not None:
            return False
        return not any(self._admittable(r) for r in s.waiting_queue)

    def _select_and_submit(self) -> None:
        if self.waiting_source:
            self._select_waiting_and_submit()
            return
        s = self.scheduler
        cache = self.cache
        # M4 load quarantine on: a background load may take dirty free pages
        # (and wait for their forward) instead of evicting cached KV to stay on
        # clean ones. Off: load_back as before.
        from decoupled_engine.srt.mem_cache.allocator.cpu_paged import load_quarantine_active

        lb_kw = (
            {"prefer_clean": False}
            if load_quarantine_active(getattr(s, "token_to_kv_pool_allocator", None))
            else {}
        )
        if self.chunked and self._chunk_step(lb_kw):
            return  # one new op per pass: this pass's is the next chunk
        examined = 0
        new_pins: List[_Pin] = []
        op_id = self._op_counter + 1
        # Depth > 1: one request per op (its own anchors, cap and window).
        one = self.depth > 1
        position = -1
        queue = list(s.disagg_prefill_bootstrap_queue.queue)
        for pos, req in enumerate(queue):
            if examined >= MAX_EXAMINE_PER_PASS:
                break
            if self.max_tokens - self.pinned_cost < self.min_tokens and not (
                self.chunked and self._chunk is None
            ):
                break
            rid = req.rid
            if rid in self._next_try:
                retry = self._next_try[rid]
                if retry is None or retry > self._pass:
                    continue
            if self.chunked and rid in self._budget_wait:
                if self._budget_wait[rid] == self._budget_epoch:
                    continue  # B1: the budget has not changed since its skip
                del self._budget_wait[rid]
                self.stats["budget_retry"] += 1
            if req.finished() or self._pin_of(req) is not None:
                continue
            if not self._selected(req):
                self._next_try[rid] = None
                self.stats["skip_class"] += 1
                continue
            if rid in cache.ongoing_prefetch:
                # The pull (or storage lookup) has not landed yet.
                continue
            examined += 1
            req.init_next_round_input(cache, cow_mamba=False)
            host = req.host_hit_length
            if host < self.min_tokens:
                self._next_try[rid] = None
                self.stats["skip_no_host_hit"] += 1
                continue
            node_id = req.best_match_node
            headroom = self._headroom()
            budget = self.max_tokens - self.pinned_cost
            # Upper bound of what the pin newly protects: the loaded host part
            # plus the matched device prefix (part of it may already be locked).
            bound = host + len(req.prefix_indices)
            if node_id in cache.ongoing_load_back or host > headroom or bound > budget:
                if self.chunked and node_id not in cache.ongoing_load_back:
                    if self._chunk is None and self._chunk_eligible(queue, pos):
                        if self._chunk_start(req, node_id, headroom, lb_kw, op_id):
                            self._next_try[rid] = None  # the chunk state owns it
                            return
                        # The oldest: retried next pass (headroom, head budget).
                        self._next_try[rid] = self._pass + 1
                        continue
                    if host <= headroom:
                        # B1: only the budget is short; wait for it to free.
                        self._budget_wait[rid] = self._budget_epoch
                        self.stats["skip_budget"] += 1
                        self.stats["budget_wait"] += 1
                        continue
                self._next_try[rid] = self._pass + RETRY_AFTER_PASSES
                self.stats[
                    (
                        "skip_busy"
                        if node_id in cache.ongoing_load_back
                        else ("skip_headroom" if host > headroom else "skip_budget")
                    )
                ] += 1
                continue
            before = cache.full_protected_size()
            if not cache.load_back(node_id, mem_quota=headroom, req=req, **lb_kw):
                self._next_try[rid] = self._pass + RETRY_AFTER_PASSES
                self.stats["skip_load_back_failed"] += 1
                continue
            lock = cache.inc_lock_ref(node_id).to_dec_params()
            cost = max(0, cache.full_protected_size() - before)
            pin = _Pin(
                req=req,
                node_id=node_id,
                lock_params=lock,
                cost=cost,
                host_tokens=host,
                created=time.monotonic(),
                op_id=op_id,
            )
            self.pins.append(pin)
            self.pinned_cost += cost
            new_pins.append(pin)
            req.prefetched_host_hit_length = host
            self._next_try[rid] = None
            if one:
                position = pos
                break
        if not new_pins:
            return
        self._submit_op(new_pins, op_id, position, one)

    def _submit_op(
        self, new_pins: List[_Pin], op_id: int, position: int, one: bool
    ) -> None:
        """Submit the loads queued by ``new_pins``' load_backs as op ``op_id``."""
        cc = self._cc()
        self._op_counter = op_id
        if self.concurrent:
            from decoupled_engine.srt.mem_cache.dedup_ipc_fanout import SLOT_BOOTSTRAP_PREFETCH

            kwargs = {}
            if self.per_part:
                kwargs["layerwise"] = True  # 3.3: layer parts, detached slot
            ack = cc.start_loading_detached(
                stream=cc.l2_transfer_engine.bootstrap_prefetch_stream(),
                ipc_slot=SLOT_BOOTSTRAP_PREFETCH,
                wait_on_caller=False,
                background=True,
                **kwargs,
            )
        else:
            kwargs = {}
            if self.ipc_fanout is not None:
                from decoupled_engine.srt.mem_cache.dedup_ipc_fanout import (
                    SLOT_BOOTSTRAP_PREFETCH,
                )

                kwargs["ipc_slot"] = SLOT_BOOTSTRAP_PREFETCH
            ack = cc.start_loading_detached(wait_on_caller=True, **kwargs)
            # Order the next forward after the whole op (dedup fan-out).
            device_module.current_stream().wait_event(ack.finish_event)
        tokens = sum(p.host_tokens for p in new_pins)
        op = _Op(
            op_id=op_id,
            ack=ack,
            anchors=frozenset(ack.node_ids),
            tokens=tokens,
            submitted=time.monotonic(),
            reqs=tuple(p.req for p in new_pins),
            # Started on every rank: unknown until the next reduced poll.
            started_all=getattr(ack.finish_event, "job", None) is None,
        )
        self.ops.append(op)
        for anchor in op.anchors:
            self._anchor_ops[anchor] = op
        self._windows[op_id] = _Window(submitted=op.submitted, tokens=tokens)
        self.stats["ops"] += 1
        self.stats["pins"] += len(new_pins)
        self.stats["tokens"] += tokens
        uncap = position == 0
        if self.waiting_source:
            starts = [self._due(r) for r in op.reqs]
            uncap = any(p is not None and p <= 1 for p in starts)
            if self._log and any(
                e is not None and self._gate_blocked(e)
                for e in (self._plan_entry(r) for r in op.reqs)
            ):
                self.v2_stats["gated_ops"] += 1
        if uncap:
            # Waiting-source ops become uncapped only when their plan is due.
            self._lift_cap(op)
            self.stats["head_uncapped_ops"] += 1
        if self._log and one:
            logger.info(
                "Bootstrap prefetch submitted: op_id=%d reqs=%d tokens=%d bytes=%d "
                "pinned=%d/%d concurrent=%d rids=%s pos=%d inflight=%d",
                op_id,
                len(new_pins),
                tokens,
                getattr(ack, "num_bytes", 0),
                self.pinned_cost,
                self.max_tokens,
                int(self.concurrent),
                [p.req.rid for p in new_pins],
                position,
                len(self.ops),
            )
        elif self._log:
            logger.info(
                "Bootstrap prefetch submitted: op_id=%d reqs=%d tokens=%d bytes=%d "
                "pinned=%d/%d concurrent=%d rids=%s",
                op_id,
                len(new_pins),
                tokens,
                getattr(ack, "num_bytes", 0),
                self.pinned_cost,
                self.max_tokens,
                int(self.concurrent),
                [p.req.rid for p in new_pins],
            )

    def _chunk_limit(self):
        return getattr(self, "_issue_tokens", None) or self.chunk_tokens

    def _select_waiting_and_submit(self):
        """Earliest-deadline host restore, one op per pass, in forecast order.

        The existing submit/poll/consume paths retain the MIN-reduced loader
        start rule and lock ownership. Only candidate selection changes here.
        """
        from decoupled_engine.srt.mem_cache.allocator.cpu_paged import load_quarantine_active

        s, cache, plan = self.scheduler, self.cache, self._plan()
        if plan is None:
            self.stats["skip_no_plan"] += 1
            return
        lb_kw = {"prefer_clean": False} if load_quarantine_active(s.token_to_kv_pool_allocator) else {}
        demand = 0
        examined = 0
        self._fifo_blocked = False
        for rid, entry, position in self._selection_rows(plan):
            demand += max(0, entry.host_tokens) + max(0, entry.pull_tokens)
            req = next((r for r in s.waiting_queue if r.rid == rid), None)
            if req is None or req.finished():
                continue
            state = self._chunk if self.chunked else None
            own_state = state is not None and state.req is req
            if self._pin_of(req) is not None and not own_state:
                continue
            if rid in cache.ongoing_prefetch or entry.pull_tokens > 0:
                self.stats["skip_pull"] += 1
                continue
            if not self._selected(req):
                self._next_try[rid] = None
                continue
            if not own_state and rid in self._next_try:
                retry = self._next_try[rid]
                if retry is None or retry > self._pass:
                    if not (retry is None and self._grown(rid, entry)):
                        continue
                    # C7: its host part grew since it was examined for good.
                    del self._next_try[rid]
                    self.v2_stats["growth_rechecks"] += 1
            if rid in self._budget_wait:
                if self._budget_wait[rid] == self._budget_epoch:
                    continue
                del self._budget_wait[rid]
                self.stats["budget_retry"] += 1
            if entry.host_tokens < self.min_tokens and not own_state:
                self._next_try[rid] = None
                self._seen(rid, entry.host_tokens)
                self.stats["skip_no_host_hit"] += 1
                continue
            start = self._due(req)
            if start is None:
                if self._log and self._gate_blocked(entry):
                    self._count_gate_skip(req, entry, demand)
                continue
            slack = start - self.lead_margin
            lead = math.ceil(demand / self.lead_tokens)
            if slack > lead:
                if self._log and self.v2_gate_free and self._gate_blocked(entry):
                    self._count_gate_skip(req, entry, demand)
                self.stats["skip_not_due"] += 1
                continue
            if examined >= MAX_EXAMINE_PER_PASS:
                break
            examined += 1
            overdue = slack < lead
            page = max(1, s.page_size)
            self._issue_tokens = max(page, ((self.due_op_tokens if overdue else self.chunk_tokens) // page) * page)
            try:
                if own_state:
                    if self._chunk_step(lb_kw):
                        self.stats["arbiter_picks"] += 1
                        self.stats["fast_issues"] += int(overdue)
                        return
                    if self._fifo_blocked:
                        return  # C5: the loader backlog is full this pass
                    continue
                # The forecast did a read-only walk. Match only when pinning a
                # selected due request, never on each speculative queue scan.
                req.init_next_round_input(cache, cow_mamba=False)
                host = req.host_hit_length
                if host < self.min_tokens:
                    self._next_try[rid] = None
                    self._seen(rid, max(host, entry.host_tokens))
                    self.stats["skip_no_host_hit"] += 1
                    continue
                node = req.best_match_node
                if node in cache.ongoing_load_back:
                    self._next_try[rid] = self._pass + RETRY_AFTER_PASSES
                    self.stats["skip_busy"] += 1
                    continue
                headroom = self._headroom(req)
                bound = host + len(req.prefix_indices)
                budget = self.max_tokens - self.pinned_cost
                op_id = self._op_counter + 1
                room = self._room(req, host, True)
                if room > headroom or bound > budget or host > self._issue_tokens:
                    if room > headroom:
                        self._memory_refusal(req, room)
                    if self.chunked and self._chunk is None:
                        if self._chunk_start(req, node, headroom, lb_kw, op_id):
                            self._next_try[rid] = None
                            self.stats["arbiter_picks"] += 1
                            self.stats["fast_issues"] += int(overdue)
                            return
                        if self._fifo_blocked:
                            return  # C5: retried next pass, no retry mark
                    if room <= headroom and bound > budget:
                        self._budget_wait[rid] = self._budget_epoch
                        self.stats["budget_wait"] += 1
                    else:
                        self._next_try[rid] = self._pass + 1
                        self.stats["skip_headroom"] += 1
                    continue
                if not self._fifo_ok(host, rid):
                    return  # C5: retried next pass, no retry mark
                before = cache.full_protected_size()
                if not cache.load_back(node, mem_quota=headroom, req=req, **lb_kw):
                    self._next_try[rid] = self._pass + RETRY_AFTER_PASSES
                    self.stats["skip_load_back_failed"] += 1
                    continue
                lock = cache.inc_lock_ref(node).to_dec_params()
                pin = _Pin(req, node, lock, max(0, cache.full_protected_size() - before), host, time.monotonic(), op_id)
                self.pins.append(pin)
                self.pinned_cost += pin.cost
                req.prefetched_host_hit_length = host
                self._next_try[rid] = None
                self._seen(rid, max(host, entry.host_tokens))
                self._submit_op([pin], op_id, position, True)
                self.stats["arbiter_picks"] += 1
                self.stats["fast_issues"] += int(overdue)
                return
            finally:
                self._issue_tokens = None

    # ------------------------------------------------- chunked (B + B1)
    def _chunk_eligible(self, queue, pos: int) -> bool:
        """The oldest candidate: every request ahead of it is pinned or was
        examined for good (TP-consistent: queue order and _next_try)."""
        for r in queue[:pos]:
            if r.finished() or self._pin_of(r) is not None:
                continue
            if r.rid in self._next_try and self._next_try[r.rid] is None:
                continue
            return False
        return True

    def _chunk_plan(self, req: Req, node_id) -> tuple:
        """(chunk tokens, run tokens, protected-bound) of the next chunk."""
        _, chunk, run = self.cache.bwp_chunk_end(node_id, self._chunk_limit())
        st = self._chunk
        first = st is None or not st.pins
        return chunk, run, chunk + (len(req.prefix_indices) if first else 0)

    def _chunk_start(self, req: Req, node_id, headroom: int, lb_kw, op_id) -> bool:
        """Start the chunk state on the oldest candidate and issue its first
        chunk; False (nothing held) if the chunk does not fit now."""
        chunk, run, bound = self._chunk_plan(req, node_id)
        if run <= 0:
            return False
        if self.waiting_source and chunk > self._chunk_limit():
            # A write-through node cannot be split until its copy settles.
            # Do not turn its whole node into an oversized queued loader job.
            self.stats["chunk_wait_unsplittable"] += 1
            return False
        if self._room(req, chunk, True) > headroom:
            self._memory_refusal(req, self._room(req, chunk, True))
            self.stats["chunk_wait_headroom"] += 1
            return False
        if bound > self.head_max_tokens:
            self.stats["chunk_wait_budget"] += 1
            return False
        if not self._fifo_ok(chunk, req.rid):
            return False
        host_params = self.cache.inc_host_lock_ref(node_id).to_dec_params()
        self._chunk = _Chunk(
            req=req,
            host_node=node_id,
            host_params=host_params,
            created=time.monotonic(),
        )
        self.stats["chunk_reqs"] += 1
        self.stats["chunk_req_tokens"] += run
        if not self._chunk_issue(node_id, headroom, lb_kw, op_id):
            self._end_chunk("start_failed")
            return False
        return True

    def _chunk_step(self, lb_kw) -> bool:
        """Issue the chunk state's next chunk, if one fits this pass."""
        st = self._chunk
        if st is None or st.done:
            return False
        req = st.req
        if req.finished() or req.rid in self.cache.ongoing_prefetch:
            return False
        req.init_next_round_input(self.cache, cow_mamba=False)
        node_id = req.best_match_node
        chunk, run, bound = self._chunk_plan(req, node_id)
        if run <= 0:
            st.done = True  # the rest is in flight or on the device
            return False
        if node_id in self.cache.ongoing_load_back:
            self.stats["chunk_wait_busy"] += 1
            return False
        headroom = self._headroom(req)
        if self.waiting_source and chunk > self._chunk_limit():
            # A write-through node cannot be split until its copy settles.
            # Do not turn its whole node into an oversized queued loader job.
            self.stats["chunk_wait_unsplittable"] += 1
            return False
        if self._room(req, chunk, not st.pins) > headroom:
            self._memory_refusal(req, self._room(req, chunk, not st.pins))
            self.stats["chunk_wait_headroom"] += 1
            return False
        if st.cost + bound > self.head_max_tokens:
            self.stats["chunk_wait_budget"] += 1
            return False
        if not self._fifo_ok(chunk, req.rid):
            return False
        return self._chunk_issue(node_id, headroom, lb_kw, self._op_counter + 1)

    def _chunk_issue(self, node_id, headroom: int, lb_kw, op_id: int) -> bool:
        """Split at the chunk end if needed, load the chunk, pin it and submit
        it as op ``op_id``. Bootstrap source: the op is uncapped (the head is
        due next). Waiting source: the op uses the plan order as its position,
        and _submit_op uncaps it only when the forecast start of the request
        is at most one pass away."""
        st = self._chunk
        cache = self.cache
        end, chunk, run = cache.bwp_chunk_end(node_id, self._chunk_limit(), split=True)
        before = cache.full_protected_size()
        if not cache.load_back(end, mem_quota=headroom, req=st.req, **lb_kw):
            self.stats["chunk_load_back_failed"] += 1
            return False
        lock = cache.inc_lock_ref(end).to_dec_params()
        cost = max(0, cache.full_protected_size() - before)
        pin = _Pin(
            req=st.req,
            node_id=end,
            lock_params=lock,
            cost=cost,
            host_tokens=chunk,
            created=time.monotonic(),
            op_id=op_id,
            chunked=True,
        )
        self.pins.append(pin)
        st.pins.append(pin)
        st.cost += cost
        st.tokens += chunk
        st.req.prefetched_host_hit_length = st.tokens
        if chunk >= run:
            st.done = True
        self.stats["chunk_ops"] += 1
        self.stats["chunk_tokens"] += chunk
        if end != node_id:
            self.stats["chunk_partial_ops"] += 1  # more of the path follows
        position = self._position(st.req) if self.waiting_source else 0
        self._submit_op([pin], op_id, position, True)
        return True
