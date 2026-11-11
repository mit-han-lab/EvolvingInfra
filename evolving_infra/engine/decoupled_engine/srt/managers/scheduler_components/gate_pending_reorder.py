"""The shared-loader gate-pending reorder of the PD prefill overlap loop
(prefill.py, phase "reorder_prev_result"), in the aggregated overlap loop
(``Scheduler.event_loop_overlap``: symmetric P/D SYMMETRIC and colocated).

With the shared host-dedup loader (SGLANG_DSA_HOST_DEDUP_FANOUT=shared /
ce_peer), a per-rank loader thread submits each admission load, and the job
waits for the forward in flight (the HiCache load fence). A launch that
consumes such a load blocks in ``LayerDoneCounter.set_consumer`` (the
consumer gate) until this rank's loader has submitted the job. Meanwhile the
scheduler thread idles for the rest of the previous forward, and it processes
that forward's result only after the launch returns. When a batch's gate is
still pending at launch, the reorder processes the previous result first: the
FINAL export publish, token streaming of the decode leg, and write-through
queuing. The gate wait then overlaps that work, and the launch itself is
unchanged. The order is that of the disable-overlap path, where the loop
processes the previous result before the launch. With M4, releases made there
carry this launch's epoch (M4 sets its floor before the launch block).

The decision is per rank: ``gate_pending_for_batch`` reads rank-local loader
state, so one rank may process the previous result before the launch and
another after it. That is safe only if neither the result processing nor
``run_batch`` issues a TP-group collective in this loop (audited):
- the symmetric consensus all-reduces (``RankConsensus.agree``, gloo) run in
  ``symmetric_pre_schedule``, called from ``get_next_batch_to_run``;
- the HiCache lockstep all-reduces (``check_hicache_events``: writing_check,
  loading_check) run in ``_get_new_batch_prefill_raw``;
- the TP structural digest runs in ``pre_run_batch_consensus_check``.
All of these sit at fixed points before this one. Result processing and
``run_batch`` reach no CPU-group or NCCL collective outside the forward. The
reorder is off with DP attention (MLP sync) and with pipeline parallelism.

No collective is not enough: lockstep state must not depend on where a rank
processes the result either. Under 6h, the write-throughs it queues are such
state: each 6h flush merges the queued writes into one op with one write ack,
and ``writing_check`` settles acks by a MIN-reduced count. The loop therefore
calls ``DeferredWriteSubmit.mark_pre_launch()`` right before this reorder and
passes its verdict to the flush after the launch, which then keeps the early
result's writes for the next flush, as on a rank that did not reorder
(deferred_write_submit.py).

The previous result always goes through the loop's ``pop_and_process``, which
calls ``process_batch_result``, so the M4 load-quarantine clock sees it.

Gate, as 6h's: SGLANG_OVERLAP_GATE_PENDING_REORDER, or the M2 umbrella
SGLANG_PREFILL_UNSERIALIZE_LOOP=1; an explicit 0 keeps it off under the
umbrella (the A/B switch). Default off: symmetric P/D's loop. (The PD prefill
loop's reorder is unconditional.) It fires only while a shared-loader job is
pending, so without the shared loader (N'sym, NCCL fan-out) the loop is
unchanged even when on. TP0 logs "GATE_REORDER loop=overlap reorders=N
launches=N prev_result_ms=X" every 60 s.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any, Callable

from decoupled_engine.srt.managers.scheduler_components.phase_timers import PHASES
from decoupled_engine.srt.mem_cache.ce_loader import (
    debug_forced_gate_reorder,
    gate_pending_for_batch,
    log_debug_force_gate_reorder,
)

logger = logging.getLogger(__name__)

STATS: Counter = Counter()
LOG_PERIOD_S = 60.0
LOG_TAG = "GATE_REORDER"


class GatePendingReorder:
    """``before_launch`` processes the previous result ahead of a launch
    whose shared-loader gate is pending. An inert instance (no scheduler)
    does nothing."""

    def __init__(self, scheduler: Any = None, loop: str = "", tp_rank: int = 0):
        self.scheduler = scheduler
        self.loop = loop
        self.tp_rank = tp_rank
        self._last_log = time.monotonic()
        # SGLANG_DEBUG_FORCE_GATE_REORDER_RANKS lists this rank: it logs its
        # own GATE_REORDER line too.
        self.forced = scheduler is not None and debug_forced_gate_reorder(scheduler)

    @classmethod
    def for_loop(cls, scheduler: Any, loop: str) -> GatePendingReorder:
        """Resolve once at loop start; an inert instance when off."""
        inst = cls._resolve(scheduler, loop)
        # SGLANG_DEBUG_FORCE_GATE_REORDER_RANKS (positive control): loud line.
        log_debug_force_gate_reorder(scheduler, loop, active=inst.active)
        return inst

    @classmethod
    def _resolve(cls, scheduler: Any, loop: str) -> GatePendingReorder:
        from decoupled_engine.srt.disaggregation.prefill_loop_unserialize import (
            overlap_gate_pending_reorder_enabled,
        )

        if not overlap_gate_pending_reorder_enabled():
            return cls()
        if getattr(scheduler, "require_mlp_sync", False):
            return cls()  # DP attention: every rank keeps one order
        if getattr(getattr(scheduler, "ps", None), "pp_size", 1) != 1:
            return cls()
        cc = getattr(getattr(scheduler, "tree_cache", None), "cache_controller", None)
        if cc is None:
            return cls()  # no HiCache: no loads, no gate
        tp_rank = getattr(getattr(scheduler, "ps", None), "tp_rank", 0)
        if tp_rank == 0:
            logger.info(
                "%s loop=%s on (fires on a pending shared-loader gate)", LOG_TAG, loop
            )
        return cls(scheduler, loop, tp_rank)

    @property
    def active(self) -> bool:
        return self.scheduler is not None

    def before_launch(
        self, batch: Any, disable_overlap: bool, pop_and_process: Callable[[], None]
    ) -> bool:
        """Right before ``run_batch(batch)``. True if the previous result was
        processed here; the loop then skips its after-launch processing."""
        s = self.scheduler
        if s is None:
            return False
        STATS[(self.loop, "launches")] += 1
        self._maybe_log()
        if disable_overlap or s.last_batch is None or not s.result_queue:
            return False  # disable-overlap already processed it; or nothing
        if not gate_pending_for_batch(s, batch):
            return False
        t0 = PHASES.start()
        pop_and_process()
        PHASES.stop("reorder_prev_result", t0)
        STATS[(self.loop, "reorders")] += 1
        STATS[(self.loop, "prev_result_us")] += int((time.perf_counter() - t0) * 1e6)
        return True

    def stats_line(self) -> str:
        st = STATS
        line = (
            f"{LOG_TAG} loop={self.loop} reorders={st[(self.loop, 'reorders')]} "
            f"launches={st[(self.loop, 'launches')]} "
            f"prev_result_ms={st[(self.loop, 'prev_result_us')] / 1e3:.1f}"
        )
        if self.forced:
            line += f" rank={self.tp_rank} forced=yes"
        return line

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log < LOG_PERIOD_S:
            return
        self._last_log = now
        if self.tp_rank == 0 or self.forced:
            logger.info(self.stats_line())
