"""6h (SGLANG_PREFILL_DEFER_WRITE_SUBMIT) outside the PD prefill loop: the
aggregated overlap loop (``Scheduler.event_loop_overlap``, colocated and
symmetric P/D SYMMETRIC) and the PD decode overlap loop
(``event_loop_overlap_disagg_decode``).

With ``--hicache-io-backend direct`` and the ``page_first_direct`` layout
(forced by SGLANG_DSA_HOST_DEDUP_FANOUT=ce_peer), every HiCache write-through
reads its device indices with a blocking ``.cpu()`` on the scheduler thread
(``HiCacheController.move_indices``, COST "move_indices_d2h"). That read waits
for everything queued on ``schedule_stream``. These loops process the previous
result right after ``_apply_war_barrier``, and admission (chunk-boundary stash,
PREBUILT insert of imports, export release) runs before the next launch, so
each such read waited for the forward in flight, and the next launch slipped
behind it. The ``kernel`` IO backend moves the host indices to the GPU instead
and never blocks.

As in the PD prefill loop (prefill.py, 6h), a write queued while held keeps its
host pages, tree commit and write-through lock (taken in ``write()``), and is
submitted at the loop's next flush point: right after the next launch, before
that launch's barrier. ``start_writing`` records the copy's start event on
``schedule_stream`` there, after the same barrier an inline submission would
have followed, so the D2H stream order is unchanged. The index read now waits
only for the forward whose result the loop processes next, which that
processing waits for anyway. An iteration that launches nothing submits
before its result processing and right after it, never inside it.

Gate and eligibility are 6h's: on with SGLANG_PREFILL_DEFER_WRITE_SUBMIT or
the SGLANG_PREFILL_UNSERIALIZE_LOOP umbrella; off under the write_back policy
(eviction waits synchronously for its own acks) and DSv4 window residency (it
drains spill acks from ack_write_queue). Off, every method is a no-op.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any, Optional

from decoupled_engine.srt.managers.scheduler_components.phase_timers import phase

logger = logging.getLogger(__name__)

# Per loop, cumulative: flush points that submitted held writes, the writes
# they submitted, and the time start_writing took there (sum / max, in us;
# the index D2H wait now sits in it). TP0 logs one line per LOG_PERIOD_S:
#   DWS_STATS loop=<overlap|decode> flushes=.. writes=.. flush_us=.. flush_us_max=..
#     early_result_held=.. (writes kept for the next flush after a reorder)
STATS: Counter = Counter()
LOG_PERIOD_S = 60.0
LOG_TAG = "DWS_STATS"


def write_controller_for_deferral(tree_cache: Any) -> Optional[Any]:
    """The tree cache's HiCache controller when 6h may hold its writes, else
    None (same conditions as the PD prefill loop's 6h)."""
    cc = getattr(tree_cache, "cache_controller", None)
    if cc is None:
        return None
    # write_back waits synchronously for its own acks during eviction.
    if getattr(cc, "write_policy", None) == "write_back":
        return None
    # DSv4 window residency drains spill acks from ack_write_queue.
    if getattr(tree_cache, "_swa_window", False):
        return None
    return cc


class DeferredWriteSubmit:
    """Holds a loop's write-through submissions between its flush points.

    ``hold()`` right after each ``_apply_war_barrier`` (and before an idle
    iteration's result processing); ``mark_pre_launch()`` right before each
    launch's gate-pending reorder; ``flush()`` right after each launch,
    before its barrier (and before / after an idle iteration's result
    processing, and before a paused iteration's ``continue``)."""

    def __init__(
        self, cc: Any = None, loop: str = "", tp_rank: int = 0, lockstep: Any = None
    ):
        from decoupled_engine.srt.managers.scheduler_components.m2_lockstep_stats import (
            LockstepStats,
        )

        self.cc = cc
        self.loop = loop
        self.tp_rank = tp_rank
        self._last_log = time.monotonic()
        # M2_LOCKSTEP hazard counters; .fix = SGLANG_M2_WRITE_MERGE_LOCKSTEP.
        self.lockstep = lockstep if lockstep is not None else LockstepStats()
        # Set by mark_pre_launch(), consumed by the next flush(): the writes
        # queued before the launch block.
        self._pre_launch_writes: Optional[int] = None

    @classmethod
    def for_loop(cls, scheduler: Any, loop: str) -> DeferredWriteSubmit:
        """Resolve once at loop start; an inert instance when off."""
        from decoupled_engine.srt.disaggregation.prefill_loop_unserialize import (
            PrefillUnserializeConfig,
        )

        if not PrefillUnserializeConfig.from_env().defer_write_submit:
            return cls()
        cc = write_controller_for_deferral(getattr(scheduler, "tree_cache", None))
        if cc is None:
            return cls()
        tp_rank = getattr(getattr(scheduler, "ps", None), "tp_rank", 0)
        if tp_rank == 0:
            logger.info(
                "%s loop=%s on (6h deferred write submit; io_backend=%s layout=%s)",
                LOG_TAG,
                loop,
                getattr(cc, "io_backend", None),
                getattr(getattr(cc, "mem_pool_host", None), "layout", None),
            )
        from decoupled_engine.srt.managers.scheduler_components.m2_lockstep_stats import (
            LockstepStats,
        )

        return cls(cc, loop, tp_rank, LockstepStats.for_scheduler(scheduler, loop))

    @property
    def active(self) -> bool:
        return self.cc is not None

    def hold(self) -> None:
        """Hold write-through submissions until the next flush()."""
        if self.cc is not None:
            self.cc.defer_start_writing = True

    def mark_pre_launch(self) -> None:
        """Right before the launch block's gate-pending reorder
        (GatePendingReorder.before_launch): hold write-throughs and remember
        how many are queued. The reorder is decided per rank. A rank that
        processes the previous result before the launch passes
        ``after_early_result`` to the next flush(), which then submits only
        the writes queued before that processing. The result's writes wait
        for the flush after, as on a rank that processes the result after the
        launch. Each flush merges its writes into one op with one write ack,
        and writing_check settles acks by a MIN-reduced count, so every rank
        must merge the same writes.
        SGLANG_M2_WRITE_MERGE_LOCKSTEP=0: only remember (the old merge)."""
        cc = self.cc
        if cc is None:
            return
        if self.lockstep.fix:
            cc.defer_start_writing = True
        self._pre_launch_writes = len(cc.write_queue)
        self.lockstep.mark()

    def flush(self, after_early_result: bool = False) -> None:
        """Stop holding and submit the held writes (merged into one op). With
        ``after_early_result`` (the launch just made followed an early result
        processing), only the writes queued before mark_pre_launch()."""
        cc = self.cc
        if cc is None:
            return
        mark, self._pre_launch_writes = self._pre_launch_writes, None
        cc.defer_start_writing = False
        queued = len(cc.write_queue)
        limit = None
        if after_early_result and mark is not None:
            prefix = min(mark, queued)
            if self.lockstep.fix:
                limit = prefix
            self.lockstep.early_flush(
                queued, prefix, kept=0 if limit is None else queued - limit
            )
        n = queued if limit is None else limit
        if n:
            STATS[(self.loop, "flushes")] += 1
            STATS[(self.loop, "writes")] += n
            t0 = time.perf_counter()
            # SGLANG_SCHED_PHASE_TIMERS: the held writes' submit (index move
            # + copy launch) as phase write.flush_submit.
            with phase("write.flush_submit"):
                if limit is None:
                    cc.start_writing()
                else:
                    cc.start_writing(limit=n)
            us = int((time.perf_counter() - t0) * 1e6)
            STATS[(self.loop, "flush_us")] += us
            key = (self.loop, "flush_us_max")
            STATS[key] = max(STATS[key], us)
        if limit is not None and cc.write_queue:
            # The early result's writes: held for the next flush.
            STATS[(self.loop, "early_result_held")] += len(cc.write_queue)
        self._maybe_log()

    def stats_line(self) -> str:
        fields = (
            "flushes",
            "writes",
            "flush_us",
            "flush_us_max",
            "early_result_held",
        )
        return f"{LOG_TAG} loop={self.loop} " + " ".join(
            f"{f}={STATS[(self.loop, f)]}" for f in fields
        )

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log < LOG_PERIOD_S:
            return
        self._last_log = now
        if self.tp_rank == 0:
            logger.info(self.stats_line())
