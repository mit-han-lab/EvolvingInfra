"""Predicted-return order inside retention class 1 (lever L1).

SGLANG_KV_SESSION_RETENTION_ORDER selects how the session-retention host
eviction (UnifiedTreeCore._drive_retention_host_eviction) orders class 1, the
host copies that a live registration covers:

- ``predicted_return_fixed`` (default): the node whose agent is predicted to
  return last goes first. The key gets -m(tau) in front of the CPU tick: tau is the node's
  idle age (seconds since its last host access) and m(tau) = E[G - tau | G >
  tau] is the expected remaining idle time, read from a FIXED table in 1-s
  bins (SGLANG_KV_SESSION_RETENTION_MTAU_JSON, else DEFAULT_FIXED_TABLE).
- ``lru``: by CPU access tick.
- ``predicted_return``: the same key with m learned online from the return
  gaps G seen on this worker (ReturnGapTable); the LRU key until a table
  rebuild has seen MIN_GAPS gaps.

Class 0, the soft pin (active level) and the kind level keep their places in
front of -m, so only the order among idle (or among active) class-1 nodes of
one kind changes. The CPU tick and the node id still break ties.

Clock. tau and G are seconds of one clock that every TP rank reads
identically. The request-broadcast source rank stamps its time.monotonic() as
the first item of the request broadcast (KvSessionClockStamp), and every rank
applies the stamp at the same point of the same list
(Scheduler.handle_kv_session_clock). No rank-local time enters a key, the gap
table or a counter. LockstepClock keeps (access-tick boundary, stamp) samples;
a host access time is the stamp in force when its host tick was taken
(bisection over the samples). The access-tick counter is identical on every TP
rank at a broadcast point (the base host eviction already depends on that), so
the samples and the ages are identical too.

Return gaps (ReturnGapTable). G = (arrival of a newer version of a live,
idle registration) - (when the previous request of that registration
finished on this worker), both on the lockstep clock. 1-s bins up to
GAP_BINS s; a longer gap is clipped into the last bin. The table m(.) is
rebuilt every REFRESH_S lockstep seconds from integer state, so it is
identical on every rank.

All orders keep the clock, the gap table and the KV_SESSION_STATS counters
(class-1 evicted tokens by idle age, gap count, learned m), so a control arm
reports the same fields.
"""

from __future__ import annotations

import bisect
import dataclasses
import hashlib
import json
import math
import os
import time
from typing import Callable, List, Optional, Sequence, Tuple

ORDER_LRU = "lru"
ORDER_PREDICTED_RETURN = "predicted_return"
ORDER_PREDICTED_RETURN_FIXED = "predicted_return_fixed"
RETENTION_ORDERS = (ORDER_LRU, ORDER_PREDICTED_RETURN_FIXED, ORDER_PREDICTED_RETURN)
ORDER_DEFAULT = ORDER_PREDICTED_RETURN_FIXED

# --------------------------------------------------------------- fixed table
# Built-in table of predicted_return_fixed: m(tau) in seconds, measured on the
# P clock (idle age from the P-leg end) in a 3P6D replay of the TRADE agentic
# trace at concurrency 90 (mean remaining idle time by idle age). Linear
# between the anchors; the trace caps waits at 120 s, so bins 116..119 hold
# m(115), and every tau from 120 s up to DEFAULT_FIXED_DEAD_AFTER_S uses bin 119.
# A node idle for DEFAULT_FIXED_DEAD_AFTER_S or longer is taken as finished and
# goes first.
DEFAULT_FIXED_ANCHORS: Tuple[Tuple[int, float], ...] = (
    (0, 21.0),
    (10, 38.0),
    (20, 60.0),
    (30, 70.0),
    (60, 67.0),
    (90, 49.0),
    (110, 33.0),
    (115, 29.0),
)
DEFAULT_FIXED_BINS = 120
DEFAULT_FIXED_LABEL = "builtin-default"
DEFAULT_FIXED_DEAD_AFTER_S = 150.0
MAX_FIXED_BINS = 3600


def interpolate_anchors(
    anchors: Sequence[Tuple[int, float]], bins: int
) -> List[float]:
    """m per 1-s bin: linear between anchors (sorted by tau), the last anchor's
    value after it, rounded to 0.1 s."""
    taus = [a for a, _ in anchors]
    out = []
    for b in range(bins):
        if b >= taus[-1]:
            out.append(round(float(anchors[-1][1]), 1))
            continue
        i = bisect.bisect_right(taus, b) - 1
        (t0, m0), (t1, m1) = anchors[i], anchors[i + 1]
        out.append(round(m0 + (m1 - m0) * (b - t0) / (t1 - t0), 1))
    return out


@dataclasses.dataclass(frozen=True)
class FixedReturnTable:
    """m(tau) for 1-s bins; tau past the last bin uses the last bin. With
    ``dead_after_s``, tau >= dead_after_s gives m = inf (evicted first)."""

    m_s: Tuple[float, ...]
    label: str
    dead_after_s: Optional[float] = None

    @property
    def digest(self) -> str:
        text = json.dumps(
            {"m_s": list(self.m_s), "dead_after_s": self.dead_after_s},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(text.encode()).hexdigest()[:12]

    def m_of_age(self, age_s: float) -> float:
        if self.dead_after_s is not None and age_s >= self.dead_after_s:
            return math.inf
        last = len(self.m_s) - 1
        return self.m_s[last if age_s >= last else int(age_s)]

    def m_text(self, tau: int) -> str:
        v = self.m_of_age(float(tau))
        return "inf" if math.isinf(v) else f"{v:.1f}"


DEFAULT_FIXED_TABLE = FixedReturnTable(
    m_s=tuple(interpolate_anchors(DEFAULT_FIXED_ANCHORS, DEFAULT_FIXED_BINS)),
    label=DEFAULT_FIXED_LABEL,
    dead_after_s=DEFAULT_FIXED_DEAD_AFTER_S,
)


def _number(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def load_fixed_return_table(value: Optional[str]) -> FixedReturnTable:
    """SGLANG_KV_SESSION_RETENTION_MTAU_JSON -> table. Unset or empty: the
    built-in table. Else a file path or inline JSON: a list of m values (s)
    for tau = 0, 1, 2, ... s, or {"m_s": [...], "dead_after_s": s or null,
    "label": "..."}. Every m must be a finite number >= 0 and dead_after_s a
    number > 0; a bad value raises ValueError, so a typo never runs as LRU."""
    if value is None or not value.strip():
        return DEFAULT_FIXED_TABLE
    text, label = value, "inline"
    if os.path.exists(value):
        with open(value) as f:
            text = f.read()
        label = os.path.basename(value)
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"SGLANG_KV_SESSION_RETENTION_MTAU_JSON: {exc}") from None
    dead_after_s = None
    if isinstance(data, dict):
        if set(data) - {"m_s", "dead_after_s", "label"} or "m_s" not in data:
            raise ValueError(
                "SGLANG_KV_SESSION_RETENTION_MTAU_JSON: an object needs m_s and "
                f"may have dead_after_s and label (got {sorted(data)})"
            )
        dead_after_s = data.get("dead_after_s")
        label = str(data.get("label") or label)
        data = data["m_s"]
    if not isinstance(data, list) or not 1 <= len(data) <= MAX_FIXED_BINS:
        raise ValueError(
            "SGLANG_KV_SESSION_RETENTION_MTAU_JSON: m_s must be a list of "
            f"1..{MAX_FIXED_BINS} numbers"
        )
    for v in data:
        if not _number(v) or v < 0:
            raise ValueError(
                f"SGLANG_KV_SESSION_RETENTION_MTAU_JSON: bad m value {v!r}"
            )
    if dead_after_s is not None:
        if not _number(dead_after_s) or dead_after_s <= 0:
            raise ValueError(
                "SGLANG_KV_SESSION_RETENTION_MTAU_JSON: bad dead_after_s "
                f"{dead_after_s!r}"
            )
        dead_after_s = float(dead_after_s)
    return FixedReturnTable(
        m_s=tuple(float(v) for v in data), label=label, dead_after_s=dead_after_s
    )


def parse_retention_order(value: Optional[str]) -> str:
    """The order named by SGLANG_KV_SESSION_RETENTION_ORDER (unset: the default)."""
    order = (value or ORDER_DEFAULT).strip().lower()
    if order not in RETENTION_ORDERS:
        raise ValueError(
            "SGLANG_KV_SESSION_RETENTION_ORDER must be one of "
            f"{', '.join(RETENTION_ORDERS)}, not {value!r}"
        )
    return order


# --------------------------------------------------------------- the clock
# Source rank: a stamp rides a broadcast that carries data at most every
# STAMP_BUSY_INTERVAL_S; an otherwise empty broadcast carries one only after
# STAMP_IDLE_INTERVAL_S.
STAMP_BUSY_INTERVAL_S = 0.1
STAMP_IDLE_INTERVAL_S = 1.0
# Clock samples kept: at most one per stamp (<= 10 per second), so >= 1600 s.
# Older host ticks have an infinite age (the last table bin).
CLOCK_SAMPLES = 16384


@dataclasses.dataclass
class KvSessionClockStamp:
    """Broadcast from the request-source TP rank: its time.monotonic() (s).

    Every rank applies it at the same position of the same request list, so
    the retention clock reads the same value on every rank."""

    t: float


class KvSessionClockSource:
    """Source rank, before the request broadcast: the stamp to put first in the
    list, or None. A list with data gets a stamp if the last one is at least
    STAMP_BUSY_INTERVAL_S old, an empty list only after STAMP_IDLE_INTERVAL_S,
    so an idle worker adds at most one small broadcast per second.
    ``clock`` (tests) replaces time.monotonic."""

    def __init__(self, clock: Optional[Callable[[], float]] = None):
        self._clock = clock
        self._last: Optional[float] = None
        self.stamps = 0

    def __call__(self, recv_reqs) -> Optional[KvSessionClockStamp]:
        now = self._clock() if self._clock is not None else time.monotonic()
        interval = STAMP_BUSY_INTERVAL_S if recv_reqs else STAMP_IDLE_INTERVAL_S
        if self._last is not None and now - self._last < interval:
            return None
        self._last = now
        self.stamps += 1
        return KvSessionClockStamp(t=float(now))


class LockstepClock:
    """Seconds that every TP rank reads identically (see the module doc).

    advance(t, tick) applies a broadcast stamp: ``t`` is the source rank's
    time, ``tick`` the next access tick (peek_time_counter()) at that point. A
    host tick at or after a sample's boundary (tick - 0.5) and before the next
    sample's boundary was taken while that stamp was in force. The 0.5 margin
    covers the -1e-5 steps that a prefix match gives the ancestors of the
    matched node. Before the first stamp every age is 0 (the LRU order)."""

    def __init__(self, size: int = CLOCK_SAMPLES):
        self.now_s = 0.0
        self.start_s: Optional[float] = None
        self.stamps = 0
        self._ticks: List[float] = []
        self._times: List[float] = []
        self._size = int(size)

    @property
    def started(self) -> bool:
        return self.start_s is not None

    @property
    def elapsed_s(self) -> float:
        return 0.0 if self.start_s is None else self.now_s - self.start_s

    def advance(self, t: float, tick: float) -> None:
        t = float(t)
        if self.start_s is None:
            # Accesses before the first stamp count from the first stamp.
            self.start_s = t
            self._ticks.append(-math.inf)
            self._times.append(t)
        elif t < self.now_s:
            t = self.now_s  # one monotonic source; never run backwards
        boundary = float(tick) - 0.5
        if boundary <= self._ticks[-1]:
            # No access tick since the last sample: move that sample instead.
            self._times[-1] = t
        else:
            self._ticks.append(boundary)
            self._times.append(t)
            if len(self._ticks) > self._size + self._size // 4:
                # Amortized trim; the same on every rank.
                del self._ticks[: -self._size]
                del self._times[: -self._size]
        self.now_s = t
        self.stamps += 1

    def time_of(self, tick: float) -> Optional[float]:
        """The stamp in force when ``tick`` was taken (None: older than the
        kept samples)."""
        i = bisect.bisect_right(self._ticks, tick) - 1
        return self._times[i] if i >= 0 else None

    def age_s(self, tick: float) -> float:
        """Seconds since ``tick`` on the lockstep clock: 0 before the first
        stamp, inf for a tick older than the kept samples."""
        if self.start_s is None:
            return 0.0
        t = self.time_of(tick)
        return math.inf if t is None else max(0.0, self.now_s - t)

    def samples(self) -> List[Tuple[float, float]]:
        """(boundary tick, time) pairs (tests and lockstep digests)."""
        return list(zip(self._ticks, self._times))


# --------------------------------------------------------------- online table
GAP_BINS = 600  # 1-s bins; a gap >= 600 s is clipped into bin 599
MIN_GAPS = 200  # gaps a rebuild must see before the predicted key is used
REFRESH_S = 30.0  # lockstep seconds between table rebuilds
MIN_TAIL_FRACTION = 0.01  # m is inf where fewer gaps reach the bin
MIN_TAIL = 2


class ReturnGapTable:
    """Return gaps and the table m(b) = E[G - b | G >= b] for 1-s bins b.

    Integer state only (counts and millisecond sums), so a rebuild from the
    same gaps gives the same table on every rank. m is inf (evicted first)
    where fewer than max(min_tail, min_tail_fraction * gaps) gaps reach the
    bin: an agent idle longer than about 99 % of the observed returns counts
    as gone."""

    def __init__(
        self,
        bins: int = GAP_BINS,
        min_gaps: int = MIN_GAPS,
        refresh_s: float = REFRESH_S,
        min_tail_fraction: float = MIN_TAIL_FRACTION,
        min_tail: int = MIN_TAIL,
    ):
        self.bins = int(bins)
        self.min_gaps = int(min_gaps)
        self.refresh_s = float(refresh_s)
        self.min_tail_fraction = float(min_tail_fraction)
        self.min_tail = int(min_tail)
        self.counts = [0] * self.bins
        self.sums_ms = [0] * self.bins
        self.n = 0
        # m per bin from the last rebuild (None before the first rebuild or
        # without gaps), and the number of gaps it was built from.
        self.table: Optional[List[float]] = None
        self.table_gaps = 0
        self.rebuilds = 0
        self._next_rebuild_s: Optional[float] = None

    def add(self, gap_s: float) -> None:
        g_ms = min(max(0, int(round(gap_s * 1000.0))), self.bins * 1000 - 1)
        b = g_ms // 1000
        self.counts[b] += 1
        self.sums_ms[b] += g_ms
        self.n += 1

    def maybe_rebuild(self, now_s: float) -> bool:
        """Rebuild once per refresh_s of the lockstep clock (the first one
        refresh_s after the first call). True if it rebuilt."""
        if self._next_rebuild_s is None:
            self._next_rebuild_s = now_s + self.refresh_s
            return False
        if now_s < self._next_rebuild_s:
            return False
        self._next_rebuild_s = now_s + self.refresh_s
        self.rebuild()
        return True

    def rebuild(self) -> None:
        self.rebuilds += 1
        self.table_gaps = self.n
        if self.n == 0:
            self.table = None
            return
        need = max(self.min_tail, math.ceil(self.min_tail_fraction * self.n))
        table = [math.inf] * self.bins
        tail_n = tail_ms = 0
        for b in range(self.bins - 1, -1, -1):
            tail_n += self.counts[b]
            tail_ms += self.sums_ms[b]
            if tail_n >= need:
                table[b] = (tail_ms - 1000 * b * tail_n) / (1000.0 * tail_n)
        self.table = table

    @property
    def ready(self) -> bool:
        """The predicted key may be used: a rebuild saw >= min_gaps gaps."""
        return self.table is not None and self.table_gaps >= self.min_gaps

    def m_of_age(self, age_s: float) -> float:
        """m for an idle age (s); needs a table."""
        last = self.bins - 1
        return self.table[last if age_s >= last else int(age_s)]

    def m_text(self, tau: int) -> str:
        if self.table is None:
            return "-"
        v = self.table[min(int(tau), self.bins - 1)]
        return "inf" if math.isinf(v) else f"{v:.1f}"


# Class-1 evicted tokens by idle age (KV_SESSION_STATS c1_evicted_idle_*).
IDLE_AGE_EDGES_S = (10.0, 60.0, 100.0)
IDLE_AGE_NAMES = ("0_10", "10_60", "60_100", "100p")
STATS_TAUS = (0, 30, 60, 90, 110)


def idle_age_bin(age_s: float) -> int:
    """Index into IDLE_AGE_NAMES for an idle age (s)."""
    return bisect.bisect_right(IDLE_AGE_EDGES_S, age_s)
