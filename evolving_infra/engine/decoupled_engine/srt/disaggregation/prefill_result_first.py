"""SGLANG_PREFILL_RESULT_FIRST_WHEN_READY: the PD prefill overlap loop
(``event_loop_overlap_disagg_prefill``) processes the previous result before
the launch also when that result's forward has already finished.

Without the flag the loop processes result k before launch k+1 only when that
launch's shared-loader gate is pending (the gate-pending reorder, phase
"reorder_prev_result"). Otherwise result k is processed after launch k+1 and
its WAR barrier, so it can wait behind forward k+1, and so do the
prefill_finished stamps and KV sends of its finishing requests. On GB300
3P6D at concurrency 90, 17 % of the finishing iterations
were such cases: forward k had ended about 18 ms before launch k+1 started,
and result k was stamped at the GPU end of forward k+1.

With the flag on, the loop also takes the early path when the copy_done event
of the queued previous result has completed (``Event.query()``: no wait).
Forward k is then over, so its result processing waits for no GPU work. The
cost: launch k+1 starts after that processing. The early path itself is the
gate-pending reorder's, unchanged (6h write-merge lockstep, PHASES
"reorder_prev_result", the after-launch processing skipped). The gate is
checked first; "ready" is the trigger only when the gate is not pending.

The decision is per rank, as the gate-pending reorder's (which reads
rank-local loader state): the event query reads rank-local GPU progress, so
one TP rank may process the result before the launch and another after it.
That is safe for the same reasons (gate_pending_reorder.py). Neither result
processing nor ``run_batch`` issues a TP collective; the loop's collectives
(the schedule step with the HiCache lockstep and bootstrap polls, the consensus
check, the M2_LOCKSTEP TP reduce at the mark before the decision, the in-flight
poll) run at fixed loop points on every rank. 6h keeps each rank's merged write
ops identical whichever way it decided. The positive control
SGLANG_DEBUG_FORCE_GATE_REORDER_RANKS takes this same path at every launch on
chosen ranks. Off with DP attention (MLP sync), where every rank keeps one
order.

Counters, per rank. Every TP rank logs one line every LOG_PERIOD_S:

    RESULT_FIRST loop=pd rank=<r> when_ready=<on|off> decisions=..
        gate_pending=.. ready=.. not_ready=.. no_event=..
        gate_pending_ms=.. ready_ms=..

- ``decisions``: launches with a previous result queued.
- ``gate_pending`` / ``ready``: early processings per trigger.
- ``not_ready``: flag on, gate not pending, copy_done not yet complete
  (normal order).
- ``no_event``: flag on, the previous result has no copy_done (normal order).
- ``*_ms``: result-processing time of the early processings, per trigger.

The sched trace record that holds a ready-triggered early processing also
holds the zero-length phase "reorder_trigger_ready" next to
"reorder_prev_result" (SGLANG_SCHED_PHASE_TIMERS).
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any, Optional

from decoupled_engine.srt.managers.scheduler_components.phase_timers import PHASES

logger = logging.getLogger(__name__)

STATS: Counter = Counter()
FIELDS = ("decisions", "gate_pending", "ready", "not_ready", "no_event")
LOG_PERIOD_S = 60.0
LOG_TAG = "RESULT_FIRST"
GATE_PENDING = "gate_pending"
READY = "ready"
READY_MARKER_PHASE = "reorder_trigger_ready"


def result_first_when_ready_enabled() -> bool:
    from decoupled_engine.srt.environ import envs

    return bool(envs.SGLANG_PREFILL_RESULT_FIRST_WHEN_READY.get())


class PrefillResultFirst:
    """The early-result triggers of one PD prefill overlap loop."""

    def __init__(self, when_ready: bool = False, tp_rank: int = 0, loop: str = "pd"):
        self.when_ready = bool(when_ready)
        self.tp_rank = tp_rank
        self.loop = loop
        self._last_log = time.monotonic()

    @classmethod
    def for_scheduler(cls, scheduler: Any, loop: str = "pd") -> PrefillResultFirst:
        """Resolve once at loop start; every TP rank logs its setting."""
        tp_rank = int(getattr(getattr(scheduler, "ps", None), "tp_rank", 0) or 0)
        when_ready = result_first_when_ready_enabled()
        note = ""
        if when_ready and getattr(scheduler, "require_mlp_sync", False):
            when_ready = False  # DP attention: every rank keeps one order
            note = " (requested; off with DP attention)"
        logger.info(
            "%s loop=%s rank=%d when_ready=%s%s",
            LOG_TAG,
            loop,
            tp_rank,
            "on" if when_ready else "off",
            note,
        )
        return cls(when_ready, tp_rank, loop)

    def decide(self, gate_pending: bool, result_queue) -> Optional[str]:
        """At a launch with a previous result queued (``result_queue[0]``):
        the trigger of an early processing, or None (after the launch)."""
        st, loop = STATS, self.loop
        st[(loop, "decisions")] += 1
        self._maybe_log()
        if gate_pending:
            return GATE_PENDING
        if not self.when_ready or not result_queue:
            return None
        copy_done = getattr(result_queue[0][1], "copy_done", None)
        if copy_done is None:
            st[(loop, "no_event")] += 1
            return None
        if copy_done.query():
            return READY
        st[(loop, "not_ready")] += 1
        return None

    def processed(self, trigger: str, t0: float) -> None:
        """After the early processing that ``trigger`` fired (started at the
        perf_counter time ``t0``)."""
        st, loop = STATS, self.loop
        st[(loop, trigger)] += 1
        st[(loop, f"{trigger}_us")] += int((time.perf_counter() - t0) * 1e6)
        if trigger == READY:
            PHASES.add(READY_MARKER_PHASE, 0.0)

    def line(self) -> str:
        st, loop = STATS, self.loop
        body = " ".join(f"{f}={st[(loop, f)]}" for f in FIELDS)
        return (
            f"{LOG_TAG} loop={loop} rank={self.tp_rank} "
            f"when_ready={'on' if self.when_ready else 'off'} {body} "
            f"gate_pending_ms={st[(loop, 'gate_pending_us')] / 1e3:.1f} "
            f"ready_ms={st[(loop, 'ready_us')] / 1e3:.1f}"
        )

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log < LOG_PERIOD_S:
            return
        self._last_log = now
        logger.info(self.line())
