"""SGLANG_SYMMETRIC_ADMIT_AFTER_LAUNCH: the import walk of a symmetric worker
(``ImportQueue.admit_pending``: the D-leg prealloc, the late-D walk, BEGIN)
runs right after the launch in the symmetric overlap loop
(``Scheduler.event_loop_overlap``) instead of inside ``symmetric_pre_schedule``,
before it.

Why. The walk's first blocking device read waits for everything queued on
``schedule_stream``, including the WAR barrier of the previous launch, i.e.
for the shared reads of the forward in flight. The read is the prealloc's
``torch.tensor(..., device=...)`` length tensors in
``alloc_for_decode_prealloc`` (a non_blocking=False H2D copy synchronizes the
stream); ``_local_state_indices`` (a row read back with ``.cpu()``), eviction
frees (``torch.unique``) and the L2 restore's index moves can follow. Before
the launch that wait delays the launch by most of a forward. GLM-5.2
symmetric D on GB300, phase timers: an allocation step lasts 54 ms
launch to launch against 26 ms; ``sym.import.prealloc.alloc`` takes 19.8 ms of
it; the GPU idles only 1.8 ms but the next launch slips by one forward (the
p90 gap in an A/B comparison). After the launch the same
wait overlaps the forward in flight: the loop processes that forward's
result next and waits for it anyway.

What. ``for_symmetric_loop`` arms the queue (``defer_admission``):
``symmetric_pre_schedule`` skips its ``admit_pending`` and
``ImportQueue.add`` no longer admits at arrival. Every pass that is not
paused then walks exactly once, at one of three points that every TP rank
chooses alike:

- ``after_launch``: right after ``run_batch`` returns, before the deferred
  export spills, the chunk backups, the 6h flush, the pre-WAR mark and the
  WAR barrier. Only when no path can process the previous result before that
  point (``admission_point``).
- before the launch, right after ``get_next_batch_to_run`` (after batch
  formation, before the disable-overlap processing of the previous result):
  an idle pass (nothing launches), a disable-overlap launch, a launch the
  symmetric launch thread takes, a launch whose shared-loader gate may be
  pending (the gate-pending reorder may process the previous result first).
- ``before_formation``, the escape: inside ``symmetric_pre_schedule``, where
  the walk runs with the flag off, in the pass after a walk that left the
  late-D claim memory-blocked (``escape_due``). After each walk the component
  sets ``defer_admission`` to ``not escape_due``. Prefill admissions then
  cannot take the memory that frees first; without the escape a P-leg start
  is charged for the claim only after the escalation (2 x claim_iters
  memory-blocked passes by default). So prefill admissions go ahead of a
  HELD import for at most about claim_iters + 1 passes (the oldest HELD
  entry becomes the claim after claim_iters passes).

Why it is safe.

- TP lockstep. The choice reads only lockstep state: the batch, the loop
  state, env flags, the launch thread's batch rule (``count=False``) and the
  late-D walk state (the claim; every rank walks alike). For the gate-pending
  reorder, whose decision is rank-local (``gate_pending_for_batch`` reads
  loader progress), it uses the lockstep superset ``gate_may_be_pending``:
  the debug knob lists any rank, or the batch consumes a staged load
  (``hicache_consumer_index`` >= 0). So a rank never processes the previous
  result before a walk that another rank runs after it. Decode formation
  clears the consumer that an adopted extend batch object still carries
  (SGLANG_HICACHE_DECODE_CLEAR_CONSUMER, required), so decode passes walk
  after the launch. The walk's own collectives (RankConsensus.agree of the L2
  restore and of the load-back slot vote, the late-D walk digest) run at the
  same point of every rank's loop: after the pre-launch collectives
  (symmetric votes, HiCache lockstep all-reduces, structural digest, the 6h
  mark_pre_launch) and before any result processing, which issues none. The
  walk is never inside ``collective_guard.forbid`` (the launch thread's
  result work).
- Pages and the GPU order (WAR). An import's pages are RDMA targets (the
  pull READ, which no CUDA stream orders after a forward). The reader's IO
  thread starts its READs behind ``row_ready``, an event the walk records on
  ``schedule_stream``. Before the launch of pass k+1 the walk recorded it
  behind barrier k; after that launch it still does (barrier k+1 is queued
  after the walk), and the set of free pages is the same: between batch
  formation and this point nothing frees device pages, and result k (whose
  frees may be pages of batch k+1, the forward just launched) is not yet
  processed. Pages freed by result k are reused only by the next pass's
  walk, behind barrier k+1. This is why every path that processes the
  previous result before the launch walks before the launch.
- Blocking reads. Every blocking device read in the walk now waits for
  barrier k only, i.e. for forward k, whose result the loop processes next.
  None waits for forward k+1, so the wait overlaps instead of stalling.
- Other post-launch work. Relative to the walk, the deferred export spills,
  the chunk backups and the 6h flush keep their order: the previous pass's
  ran before the walk, this pass's run after it, as before. The walk now
  follows this pass's batch formation, the structural digest and the launch.
  The loop's other load submitters around the launch (the BWP step, the
  next-batch host prefetch) are refused with the flag.
- The escape walk is the flag-off walk, at the flag-off point.

What changes besides the timing (all ranks alike).

- Imports are allocated after this pass's prefill and decode allocations
  (before: ahead of them). An import that no longer fits is HELD; the late-D
  claim reserve and the escape above bound its extra wait.
- The decode headroom the walk charges counts the running batch after
  formation plus the waiting queue; requests of the launched batch that are
  in neither (new prefill admissions, not the chunked request, which F-A
  covers) are added (``_launched_outside_queues``), as they were counted in
  the waiting queue before.
- The late-D binder (owner rank 0) and the prefill claim reserve see the
  previous pass's walk; an ADMIT that arrives in pass k+1 is still allocated
  in pass k+1, after its launch. A staged L2 restore is fenced behind the
  forward just launched (one forward later than before).

Refused at startup (``startup_refusal``, from ``dispatch_event_loop``):

- any loop other than ``event_loop_overlap``: the flag would be inert;
- late D off: only late D's claim reserve keeps memory for an import that the
  pass's prefill admissions would take first;
- M4 (CPU page book, load quarantine): untested, and its release epochs and
  clean-page accounting would move with the walk;
- the bootstrap-window prefetch (SGLANG_SYMMETRIC_BWP): its load submit,
  before the launch, would precede the walk's L2-restore slot vote;
- SGLANG_HICACHE_PREFETCH_NEXT_BATCH: the walk's L2 restore would run between
  that prefetch's prepare (before the launch) and its submit (after it);
- SGLANG_HICACHE_DECODE_CLEAR_CONSUMER off (see TP lockstep above).

Off, nothing here is imported or called.

Logs. Startup, every TP rank:
    SYM_ADMIT_AFTER_LAUNCH rank=<r> on
About every 60 s, every TP rank, at a pass count that is a multiple of
LOG_EVERY_PASSES (lines with the same ``passes=`` must be equal on every rank):
    SYM_ADMIT_AFTER_LAUNCH rank=<r> passes=N after_launch=N
        before_formation=N before_idle=N before_disable_overlap=N
        before_launch_thread=N before_gate_reorder=N admitted_after=N
        admitted_before=N held_passes_now=N held_passes_max=N
``held_passes``: walk passes since a pending import first did not fit (the
longest now, and the longest seen).
Phase timers: ``sym_admit_after_launch`` / ``sym_admit_before_launch`` around
the existing ``sym_admit_pending`` (and the ``sym.import.*`` phases inside).
An escape walk keeps the flag-off phases (``sym_admit_pending`` inside
``sym_pre_schedule``).
The after-launch walk of pass k+1 lies between launches k+1 and k+2, but its
phases land in the iter_trace row of result k (processed after it).
SGLANG_SCHED_SYNC_DEBUG: the walk is sync-debug site ``sym_admit``.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any, Callable, Optional

from decoupled_engine.srt.disaggregation.late_d import HELD
from decoupled_engine.srt.managers.scheduler_components.phase_timers import (
    phase,
    sync_debug_site,
)

logger = logging.getLogger(__name__)

STATS: Counter = Counter()
FLAG = "SGLANG_SYMMETRIC_ADMIT_AFTER_LAUNCH"
LOG_TAG = "SYM_ADMIT_AFTER_LAUNCH"
LOG_PERIOD_S = 60.0
LOG_EVERY_PASSES = 64

AFTER_LAUNCH = "after_launch"
BEFORE_FORMATION = "before_formation"
BEFORE_IDLE = "before_idle"
BEFORE_DISABLE_OVERLAP = "before_disable_overlap"
BEFORE_LAUNCH_THREAD = "before_launch_thread"
BEFORE_GATE_REORDER = "before_gate_reorder"
# Every pass that reaches batch formation counts once in "passes" and once in
# exactly one of these.
COUNTERS = (
    AFTER_LAUNCH,
    BEFORE_FORMATION,
    BEFORE_IDLE,
    BEFORE_DISABLE_OVERLAP,
    BEFORE_LAUNCH_THREAD,
    BEFORE_GATE_REORDER,
)


def admit_after_launch_enabled() -> bool:
    from decoupled_engine.srt.environ import envs

    return bool(envs.SGLANG_SYMMETRIC_ADMIT_AFTER_LAUNCH.get())


def startup_refusal(scheduler: Any, pp_size: int) -> Optional[str]:
    """Why a symmetric worker must not start with the flag on (None: it may
    start). ``dispatch_event_loop`` raises ValueError with this text."""
    from decoupled_engine.srt.environ import envs
    from decoupled_engine.srt.mem_cache.cpu_page_book import m4_flags_set

    if (
        scheduler.enable_pdmux
        or not scheduler.enable_overlap
        or scheduler.enable_overlap_mlx
        or pp_size > 1
    ):
        # Only event_loop_overlap moves the walk: elsewhere it would be inert.
        return f"{FLAG} needs the symmetric overlap loop (event_loop_overlap)"
    if getattr(getattr(scheduler, "import_queue", None), "late_d", None) is None:
        return (
            f"{FLAG} needs late D (--disaggregation-late-d alloc or choose): the "
            "walk follows the pass's prefill admissions, and only late D's claim "
            "reserve keeps memory for an import that they would take first"
        )
    if m4_flags_set():
        return (
            f"{FLAG} is not supported with M4 (SGLANG_PREFILL_CPU_PAGE_BOOKKEEPING"
            " / SGLANG_DSA_HOST_DEDUP_LOAD_QUARANTINE): untested, and its release "
            "epochs and clean-page accounting would move with the walk"
        )
    if getattr(scheduler, "bootstrap_prefetch_mode", "off") != "off":
        return (
            f"{FLAG} is not supported with the bootstrap-window prefetch "
            "(SGLANG_SYMMETRIC_BWP, --disaggregation-bootstrap-prefetch): its load "
            "submit, before the launch, would precede the walk's L2-restore slot "
            "vote"
        )
    if envs.SGLANG_HICACHE_PREFETCH_NEXT_BATCH.get() and getattr(
        scheduler, "enable_hierarchical_cache", False
    ):
        return (
            f"{FLAG} is not supported with SGLANG_HICACHE_PREFETCH_NEXT_BATCH: the "
            "walk's L2 restore would run between that prefetch's prepare (before "
            "the launch) and its submit (after it)"
        )
    if not envs.SGLANG_HICACHE_DECODE_CLEAR_CONSUMER.get():
        return (
            f"{FLAG} needs SGLANG_HICACHE_DECODE_CLEAR_CONSUMER=1: without it, a "
            "decode pass on an extend batch object adopted as the running batch "
            "keeps that batch's load consumer and walks before its launch, with "
            "the old wait"
        )
    return None


def escape_due(queue: Any) -> bool:
    """Whether the next pass walks before batch formation, as with the flag
    off: the late-D claim is memory-blocked after the last walk (its miss
    was memory, ``LateDDecode.end_pass``). Lockstep: every rank walks
    alike."""
    late = getattr(queue, "late_d", None)
    return (
        late is not None
        and getattr(late, "claim", None) is not None
        and getattr(late, "claim_memory_blocked", False) is True
    )


def held_passes(queue: Any) -> int:
    """Walk passes since the longest-held pending import first did not fit
    (0 without late D or without a HELD import). Lockstep."""
    late = getattr(queue, "late_d", None)
    if late is None:
        return 0
    now = late.pass_index
    return max(
        (
            now - entry.late_d_held_since
            for entry in queue.pending
            if entry.late_d_state == HELD
        ),
        default=0,
    )


def gate_may_be_pending(batch: Any, forced_any: bool) -> bool:
    """Lockstep superset of ``ce_loader.gate_pending_for_batch``: False means
    that no TP rank finds a pending shared-loader gate for ``batch``. That
    function is True only for a TP rank listed in
    SGLANG_DEBUG_FORCE_GATE_REORDER_RANKS (``forced_any``: the list is not
    empty, the same env on every rank) or for a batch that consumes a staged
    load (``hicache_consumer_index`` >= 0, the same on every rank); whether
    the job is still pending is rank-local and not read here."""
    if forced_any:
        return True
    idx = getattr(batch, "hicache_consumer_index", -1)
    return idx is not None and idx >= 0


def admission_point(
    scheduler: Any,
    batch: Any,
    disable_overlap: bool,
    launch_thread: Any = None,
    gate_reorder: Any = None,
    gate_forced_any: bool = False,
) -> str:
    """Where this pass walks: AFTER_LAUNCH, or the reason it walks before the
    launch. Reads only lockstep state, so every TP rank decides alike."""
    if not batch:
        # The loop's own test (``if batch:``): nothing launches.
        return BEFORE_IDLE
    if disable_overlap:
        # The loop processes the previous result before this launch.
        return BEFORE_DISABLE_OVERLAP
    if scheduler.last_batch is None or not scheduler.result_queue:
        return AFTER_LAUNCH  # no previous result to process early
    if launch_thread is not None and launch_thread.batch_applies(
        scheduler, batch, count=False
    ):
        # SymmetricLaunchThread.use_for is True for this launch: the
        # previous result is processed during it.
        return BEFORE_LAUNCH_THREAD
    if getattr(gate_reorder, "active", False) is True and gate_may_be_pending(
        batch, gate_forced_any
    ):
        # GatePendingReorder.before_launch may process it first on some rank.
        return BEFORE_GATE_REORDER
    return AFTER_LAUNCH


def _launched_outside_queues(scheduler: Any, batch: Any) -> int:
    """Requests of the launched ``batch`` that are in neither the running
    batch nor the waiting queue (new prefill admissions), minus the chunked
    request (the F-A local reserve covers it). The walk before the launch
    counted them in the waiting queue."""
    running = getattr(scheduler, "running_batch", None)
    if batch is None or batch is running:
        return 0
    reqs = getattr(batch, "reqs", None) or ()
    if not reqs:
        return 0
    seen = {id(r) for r in (getattr(running, "reqs", None) or ())}
    chunked = getattr(scheduler, "chunked_req", None)
    return sum(1 for r in reqs if id(r) not in seen and r is not chunked)


class SymmetricAdmitAfterLaunch:
    """The walk points of one symmetric overlap loop: ``before_launch`` right
    after batch formation, ``after_launch`` right after ``run_batch``."""

    @classmethod
    def for_symmetric_loop(
        cls,
        scheduler: Any,
        *,
        launch_thread: Any = None,
        gate_reorder: Any = None,
    ) -> Optional[SymmetricAdmitAfterLaunch]:
        """Resolve once at loop start (after the launch thread and the
        gate-pending reorder). None, and the queue left as it is, when the
        flag is off or the worker has no import queue."""
        if not admit_after_launch_enabled():
            return None
        queue = getattr(scheduler, "import_queue", None)
        if queue is None:
            return None
        from decoupled_engine.srt.mem_cache.ce_loader import debug_force_gate_reorder_ranks

        tp_rank = int(getattr(getattr(scheduler, "ps", None), "tp_rank", 0) or 0)
        inst = cls(
            scheduler,
            queue,
            tp_rank=tp_rank,
            launch_thread=launch_thread,
            gate_reorder=gate_reorder,
            gate_forced_any=bool(debug_force_gate_reorder_ranks()),
        )
        logger.info("%s rank=%d on", LOG_TAG, tp_rank)
        return inst

    def __init__(
        self,
        scheduler: Any,
        queue: Any,
        *,
        tp_rank: int = 0,
        launch_thread: Any = None,
        gate_reorder: Any = None,
        gate_forced_any: bool = False,
    ):
        self.queue = queue
        self.tp_rank = tp_rank
        self.launch_thread = launch_thread
        self.gate_reorder = gate_reorder
        self.gate_forced_any = gate_forced_any
        self._point: Optional[str] = None
        self._extra_active = 0
        self._last_log = time.monotonic()
        # The walk's decode-headroom count also sees the launched batch's new
        # admissions while it runs after formation (_launched_outside_queues).
        base: Callable[[], int] = queue.num_active_reqs
        self._base_num_active_reqs = base
        queue.num_active_reqs = lambda: base() + self._extra_active
        queue.defer_admission = True

    # -- loop hooks ----------------------------------------------------
    def before_launch(self, scheduler: Any, batch: Any, disable_overlap: bool) -> None:
        """Right after ``get_next_batch_to_run`` (and the disable-overlap
        decision), before anything may process the previous result. Walks now
        unless the pass walks after its launch or already walked before
        formation (the escape)."""
        STATS["passes"] += 1
        if self.queue.defer_admission is not True:
            # symmetric_pre_schedule walked this pass, before formation.
            STATS[BEFORE_FORMATION] += 1
            self._point = None
            self._after_walk()
            self._maybe_log()
            return
        point = admission_point(
            scheduler,
            batch,
            disable_overlap,
            self.launch_thread,
            self.gate_reorder,
            self.gate_forced_any,
        )
        STATS[point] += 1
        if point == AFTER_LAUNCH:
            self._point = point
        else:
            self._point = None
            STATS["admitted_before"] += self._walk(
                scheduler, batch, "sym_admit_before_launch"
            )
            self._after_walk()
        self._maybe_log()

    def after_launch(self, scheduler: Any, batch: Any, last_processed: bool) -> None:
        """Right after ``run_batch(batch)`` returned, before the deferred
        spills, the chunk backups, the 6h flush and the WAR barrier."""
        point, self._point = self._point, None
        if point != AFTER_LAUNCH:
            return
        if last_processed:
            # admission_point promised that no path processes the previous
            # result before this launch. A walk now would allocate pages that
            # the forward just launched may still touch, and another rank may
            # have walked before that processing: stop loudly.
            raise RuntimeError(
                f"{LOG_TAG}: the previous result was processed before the launch "
                "although the walk was deferred past it (admission_point out of "
                "date with the loop's early-result paths)"
            )
        STATS["admitted_after"] += self._walk(
            scheduler, batch, "sym_admit_after_launch"
        )
        self._after_walk()

    # -- walk ----------------------------------------------------------
    @sync_debug_site("sym_admit")
    def _walk(self, scheduler: Any, batch: Any, name: str) -> int:
        self._extra_active = _launched_outside_queues(scheduler, batch)
        try:
            with phase(name):
                with phase("sym_admit_pending"):
                    return int(self.queue.admit_pending() or 0)
        finally:
            self._extra_active = 0

    def _after_walk(self) -> None:
        """After each pass's walk, wherever it ran: the wait stats, and where
        the next pass walks (``defer_admission`` False: before formation).
        Lockstep state only."""
        held = held_passes(self.queue)
        STATS["held_passes_now"] = held
        STATS["held_passes_max"] = max(STATS["held_passes_max"], held)
        self.queue.defer_admission = not escape_due(self.queue)

    # -- logs ----------------------------------------------------------
    def stats_line(self) -> str:
        return (
            f"{LOG_TAG} rank={self.tp_rank} passes={STATS['passes']} "
            + " ".join(f"{k}={STATS[k]}" for k in COUNTERS)
            + f" admitted_after={STATS['admitted_after']}"
            + f" admitted_before={STATS['admitted_before']}"
            + f" held_passes_now={STATS['held_passes_now']}"
            + f" held_passes_max={STATS['held_passes_max']}"
        )

    def _maybe_log(self) -> None:
        if STATS["passes"] % LOG_EVERY_PASSES:
            return
        now = time.monotonic()
        if now - self._last_log < LOG_PERIOD_S:
            return
        self._last_log = now
        logger.info(self.stats_line())
