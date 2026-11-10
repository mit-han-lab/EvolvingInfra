"""Request-side host SWA ledger for DSv4 window SWA residency.

Under ``--dsv4-uniform-swa-residency window`` a running request keeps on the
device only the SWA its next forwards can read. Older request-private SWA is
spilled: its device slots are detached from the FULL->SWA mapping, copied to
host SWA slots, and freed once the MIN-reduced write ack arrives
(copy-then-free). The acked host slices stay on the request until it finishes,
when the tree adopts them for the nodes that need them and frees the rest.

UnifiedRadixCache owns the registry below (ack id -> ops); the ledger itself
lives on the Req, because ``ReqKvInfo.mark_released`` zeroes the KV state.

Ownership: a detached device slot belongs to exactly one
op until its ack, or goes back to the request (re-attach) or with it into the
tree; a host slot belongs to one op, one ledger segment or one tree node, and
host slots written by an in-flight copy are freed only in the ack handler.
"""

from __future__ import annotations

import bisect
import enum
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch

# Spill ops share the write ack queue with tree write-throughs, whose ack ids
# are node ids; ids at or above this base belong to spill ops.
SPILL_ACK_BASE = 1 << 60

INFLIGHT = "inflight"
REATTACHED = "reattached"
ORPHANED = "orphaned"


def is_spill_ack_id(ack_id: int) -> bool:
    return ack_id >= SPILL_ACK_BASE


class LedgerDisposition(enum.Enum):
    """What releasing a request without inserting it does with its ledger.
    KEEP holds the acked host slices for a resume;
    RELEASE frees them."""

    KEEP = "keep"
    RELEASE = "release"


@dataclass
class SwaSegment:
    """Acked host SWA for request positions [start, end), page aligned."""

    start: int
    end: int
    host_idx: torch.Tensor
    # Peer page keys of the slice's pages when it was published SWA-only to
    # the peer KV servicer (SGLANG_DSV4_PEER_PUBLISH_SPILL_SWA), else None.
    page_keys: Optional[list] = None


@dataclass(eq=False)
class SwaSpillOp:
    """One request's share of a spill D2H copy.

    ``swa_dev_idx`` holds the detached device slots: owned by the op while
    ``inflight`` or ``orphaned``, by the request or tree once ``reattached``.
    ``host_idx`` is owned by the op until its ack in every state.
    """

    ack_id: int
    start: int
    end: int
    full_idx: torch.Tensor
    swa_dev_idx: torch.Tensor
    host_idx: torch.Tensor
    ledger: Optional[SwaHostLedger]
    state: str = INFLIGHT
    # Peer page keys of pages [start, end), set when the slice is to be
    # published to the peer KV servicer at its ack.
    page_keys: Optional[list] = None

    @property
    def num_tokens(self) -> int:
        return self.end - self.start


@dataclass(eq=False)
class SwaHostLedger:
    """Host SWA a request spilled. Positions below ``base`` (the tree-owned
    prefix when the ledger was opened) are not tracked here."""

    base: int
    segments: list[SwaSegment] = field(default_factory=list)
    spills: deque[SwaSpillOp] = field(default_factory=deque)

    def add_segment(self, seg: SwaSegment) -> None:
        """Insert an acked slice, keeping ``segments`` sorted and disjoint.
        Spill acks append past every earlier slice; a host->host backfill
        (swa_backfill.py) may commit a range below them."""
        i = bisect.bisect_left([s.start for s in self.segments], seg.start)
        assert i == 0 or self.segments[i - 1].end <= seg.start, (
            "overlapping SWA ledger segments",
            (self.segments[i - 1].start, self.segments[i - 1].end),
            (seg.start, seg.end),
        )
        assert i == len(self.segments) or seg.end <= self.segments[i].start, (
            "overlapping SWA ledger segments",
            (seg.start, seg.end),
            (self.segments[i].start, self.segments[i].end),
        )
        self.segments.insert(i, seg)

    def covered_end(self) -> int:
        """End of the contiguous acked coverage starting at ``base``."""
        end = self.base
        for seg in self.segments:
            if seg.start > end:
                break
            end = max(end, seg.end)
        return end

    def inflight_ops(self) -> list[SwaSpillOp]:
        return [op for op in self.spills if op.state == INFLIGHT]

    def segment_tokens(self) -> int:
        return sum(seg.end - seg.start for seg in self.segments)


class SwaLedgerRegistry:
    """Spill ops by ack id, in submit order, plus counters.

    Every mutation happens at a lockstep point of the scheduler (spill
    submit, MIN-reduced ack, request release), so the registry is identical on
    all TP ranks.
    """

    STAT_KEYS = (
        "submitted_tokens",
        "acked_tokens",
        "reattached_tokens",
        "orphaned_tokens",
        "adopted_tokens",
        "dup_freed_tokens",
        "released_tokens",
        # Every truncated finish insert; resume_truncated_tokens is the part
        # clamped at an evicted prefix of a resumed PD decode request.
        "truncated_tokens",
        "resume_truncated_tokens",
        # PD decode (window_host): received-prefix tokens [dpl, T) with no
        # backfill submitted (no servicer / capability, TP or dedup
        # mismatch, backfill off); their finish insert stops at dpl.
        "backfill_skipped_tokens",
        "stalls",
        "drains",
        "spill_bytes",
    )

    def __init__(self) -> None:
        self._next_ack_id = SPILL_ACK_BASE
        self._ops: dict[int, list[SwaSpillOp]] = {}
        # Detached device slots owned by inflight or orphaned ops.
        self.detached_tokens = 0
        # Host slots owned by ops awaiting their ack.
        self.op_host_tokens = 0
        # Requests holding a ledger.
        self.ledger_live = 0
        self.stats = dict.fromkeys(self.STAT_KEYS, 0)

    def next_ack_id(self) -> int:
        ack_id = self._next_ack_id
        self._next_ack_id += 1
        return ack_id

    def add(self, ack_id: int, ops: list[SwaSpillOp]) -> None:
        self._ops[ack_id] = ops
        for op in ops:
            self.detached_tokens += op.num_tokens
            self.op_host_tokens += op.num_tokens
            self.stats["submitted_tokens"] += op.num_tokens

    def pop(self, ack_id: int) -> Optional[list[SwaSpillOp]]:
        """Remove an acked op group. The caller settles every op in it."""
        ops = self._ops.pop(ack_id, None)
        for op in ops or ():
            self.op_host_tokens -= op.num_tokens
            if op.state != REATTACHED:
                self.detached_tokens -= op.num_tokens
        return ops

    def mark_reattached(self, op: SwaSpillOp) -> None:
        assert op.state == INFLIGHT, op.state
        op.state = REATTACHED
        self.detached_tokens -= op.num_tokens
        self.stats["reattached_tokens"] += op.num_tokens

    def mark_orphaned(self, op: SwaSpillOp) -> None:
        assert op.state == INFLIGHT, op.state
        op.state = ORPHANED
        self.stats["orphaned_tokens"] += op.num_tokens

    def live_ops(self) -> Iterator[SwaSpillOp]:
        """Ops awaiting their ack, in submit order."""
        for ops in self._ops.values():
            yield from ops

    def live_count(self) -> int:
        return sum(len(ops) for ops in self._ops.values())

    def digest_inputs(self) -> list[int]:
        return [
            len(self._ops),
            self.detached_tokens,
            self.op_host_tokens,
            self.ledger_live,
            self._next_ack_id - SPILL_ACK_BASE,
        ]
