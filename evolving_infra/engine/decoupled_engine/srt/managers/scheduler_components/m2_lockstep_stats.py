"""Hazard counters for the per-rank gate-pending reorder under 6h.

The shared-loader gate-pending reorder is decided per rank. It runs in the PD
prefill loop and, with T4, in the aggregated loop. Under 6h each flush merges
the queued write-throughs into one op with one write ack, and writing_check
settles acks by a MIN-reduced count. A result processed before the launch on
some ranks only therefore used to put its writes into different merged ops on
different ranks. With SGLANG_M2_WRITE_MERGE_LOCKSTEP (default on), the flush
after such a launch keeps those writes for the next flush, as on the other
ranks.

The counters below measure how often that hazard is exercised, with the fix
on or off. They count each launch whose previous result was processed early
on this rank (with 6h off, early_ops / early_writes count the write ops that
processing submitted inline, one per write: no hazard by construction):

- ``reorders``: early result processings.
- ``reorders_held``: of those, launches with writes already held before the
  processing (the queue was non-empty at the mark).
- ``early_ops``: of those, launches whose early processing queued writes.
  These writes' flush depends on the rank's reorder decision: the hazard
  opportunity.
- ``early_writes``: the writes those processings queued.
- ``split_ops``: early_ops with held writes too. Unfixed, the merged op
  differs at once.
- ``kept_back``: writes the fix kept for the next flush (fix on only).

TP0 logs one line per loop every LOG_PERIOD_S:

    M2_LOCKSTEP loop=<pd|overlap> rank=<r> fix=<on|off> 6h=<on|off>
        reorders=.. reorders_held=.. early_ops=.. early_writes=.. split_ops=..
        kept_back=.. tp_min=[..] tp_max=[..] tp_at=<launch>

``tp_min`` / ``tp_max`` are the element-wise MIN / MAX over the TP ranks of
(reorders, early_ops, early_writes). They are reduced on the TP CPU group
every TP_REDUCE_EVERY launches, at the same loop point on every rank (the
mark right before the reorder). min != max: the ranks took different reorder
decisions. A rank that SGLANG_DEBUG_FORCE_GATE_REORDER_RANKS forces (the
positive control) logs its own line too, with forced=yes after 6h=.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

STATS: Counter = Counter()
FIELDS = (
    "reorders",
    "reorders_held",
    "early_ops",
    "early_writes",
    "split_ops",
    "kept_back",
)
TP_FIELDS = ("reorders", "early_ops", "early_writes")
LOG_PERIOD_S = 60.0
LOG_TAG = "M2_LOCKSTEP"
TP_REDUCE_EVERY = 512


def write_merge_lockstep_enabled() -> bool:
    from decoupled_engine.srt.environ import envs

    return bool(envs.SGLANG_M2_WRITE_MERGE_LOCKSTEP.get())


class LockstepStats:
    """One loop's counters. Inert (no-op) without a loop name."""

    def __init__(
        self,
        loop: str = "",
        tp_rank: int = 0,
        group: Any = None,
        fix: bool = True,
        forced: bool = False,
        six_h: bool = True,
    ):
        self.loop = loop
        self.tp_rank = tp_rank
        self.fix = fix
        # 6h holds this loop's writes (off: the PD loop's counters with the
        # debug knob only).
        self.six_h = six_h
        # SGLANG_DEBUG_FORCE_GATE_REORDER_RANKS lists this rank: it logs its
        # own line too.
        self.forced = forced
        self.group = group if _group_size(group) > 1 else None
        self.marks = 0
        self.tp: Optional[tuple[list[int], list[int], int]] = None
        self._last_log = time.monotonic()

    @classmethod
    def for_scheduler(
        cls, scheduler: Any, loop: str, six_h: bool = True
    ) -> LockstepStats:
        from decoupled_engine.srt.mem_cache.ce_loader import debug_forced_gate_reorder

        tp_rank = getattr(getattr(scheduler, "ps", None), "tp_rank", 0)
        group = getattr(scheduler, "tp_cpu_group", None)
        return cls(
            loop,
            tp_rank,
            group,
            fix=write_merge_lockstep_enabled(),
            forced=debug_forced_gate_reorder(scheduler),
            six_h=six_h,
        )

    def mark(self) -> None:
        """At the mark right before each launch's reorder decision (the same
        loop point on every rank)."""
        if not self.loop:
            return
        self.marks += 1
        if self.group is not None and self.marks % TP_REDUCE_EVERY == 0:
            self._tp_reduce()
        self._maybe_log()

    def early_flush(self, queued: int, prefix: int, kept: int) -> None:
        """The flush right after a launch whose previous result this rank
        processed early. ``queued``: writes queued at the flush; ``prefix``:
        of those, queued before the processing; ``kept``: writes the fix keeps
        for the next flush."""
        if not self.loop:
            return
        st, loop = STATS, self.loop
        st[(loop, "reorders")] += 1
        if prefix > 0:
            st[(loop, "reorders_held")] += 1
        early = queued - prefix
        if early > 0:
            st[(loop, "early_ops")] += 1
            st[(loop, "early_writes")] += early
            if prefix > 0:
                st[(loop, "split_ops")] += 1
        st[(loop, "kept_back")] += kept

    def line(self) -> str:
        st = STATS
        body = " ".join(f"{f}={st[(self.loop, f)]}" for f in FIELDS)
        out = (
            f"{LOG_TAG} loop={self.loop} rank={self.tp_rank} "
            f"fix={'on' if self.fix else 'off'} 6h={'on' if self.six_h else 'off'}"
            f"{' forced=yes' if self.forced else ''} {body}"
        )
        if self.tp is not None:
            lo, hi, at = self.tp
            out += f" tp_min={lo} tp_max={hi} tp_at={at}"
        return out

    def _tp_reduce(self) -> None:
        mine = [int(STATS[(self.loop, f)]) for f in TP_FIELDS]
        both = torch.tensor(mine + [-v for v in mine], dtype=torch.int64)
        torch.distributed.all_reduce(
            both, op=torch.distributed.ReduceOp.MIN, group=self.group
        )
        n = len(mine)
        vals = both.tolist()
        self.tp = (vals[:n], [-v for v in vals[n:]], self.marks)

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log < LOG_PERIOD_S:
            return
        self._last_log = now
        if self.tp_rank == 0 or self.forced:
            logger.info(self.line())


def _group_size(group: Any) -> int:
    if group is None:
        return 1
    try:
        return torch.distributed.get_world_size(group=group)
    except Exception:  # noqa: BLE001 - no process group: no TP reduce
        return 1
