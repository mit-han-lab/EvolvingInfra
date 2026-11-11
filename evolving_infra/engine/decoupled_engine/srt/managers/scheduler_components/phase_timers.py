"""Scheduler phase timers and sync-debug sites (overlap load-back M0).

Measurement only: nothing
here changes scheduling, and with both flags off every decorator returns the
original function and every inline call is a single attribute check.

- ``SGLANG_SCHED_PHASE_TIMERS``: ``time.perf_counter`` totals per named phase
  (admission sub-steps, load-back, result processing, the synchronizing
  sites S1-S5). The sched tracer takes the totals accumulated since its
  previous record into ``IterTraceRecord.phase_ms``; TP0 also logs a 60 s
  summary ("Scheduler phase timers over ...").
- ``SGLANG_SCHED_SYNC_DEBUG``: inside a *site* (the admission function,
  ``run_batch``, prefill result processing), torch's CUDA sync-debug mode is
  "warn"; every synchronizing CUDA call raises a warning that is captured
  with its Python stack. Per site, the first ``SGLANG_SCHED_SYNC_DEBUG_STACKS``
  distinct stacks are logged once; counts go into the 60 s summary. The mode
  is process-global, so calls made by other threads while a site is open are
  attributed to ``other@<thread>``. For correctness runs only, not timing.

Launch gap (with ``SGLANG_SCHED_PHASE_TIMERS``): ``launch_marker`` on
``Scheduler.run_batch`` records the wall time from the return of one forward
launch to the start of the next as phase ``launch_gap`` (not after an idle
pass: ``launch_gap_after_idle``). Every phase that ends inside that window is
also summed per window; a gap above ``SGLANG_SCHED_LAUNCH_GAP_LOG_MS`` (default
30) is logged on every rank with those per-phase times ("Scheduler launch gap
..."), at most ``SGLANG_SCHED_LAUNCH_GAP_LOG_PER_MIN`` lines per 60 s. Phases
of other threads (``xfer.*``: reader transfer threads, ``io.*``: reader IO
thread) are in the same table, so a window shows the plane work that ran
during it. Nested phases are listed separately (names with dots nest).
"""

from __future__ import annotations

import contextlib
import functools
import logging
import sys
import threading
import time
import traceback
import warnings
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SYNC_MSG = "synchronizing CUDA operation"
SUMMARY_INTERVAL_S = 60.0


def _env_value(name: str, default):
    try:
        from decoupled_engine.srt.environ import envs

        v = getattr(envs, name).get()
        return default if v is None else v
    except Exception:  # noqa: BLE001 - measurement must never break the engine
        return default


def _env_flag(name: str) -> bool:
    return bool(_env_value(name, False))


def apply_gil_switch_interval(value: Optional[float]) -> Tuple[float, bool]:
    """SGLANG_SCHED_GIL_SWITCH_INTERVAL_S at scheduler process start: None
    leaves the interpreter's interval; a value sets it (process-wide, every
    thread). Returns (effective interval in seconds, whether it was set). A
    value <= 0 raises ValueError (sys.setswitchinterval) and stops the start."""
    if value is not None:
        sys.setswitchinterval(float(value))
    return sys.getswitchinterval(), value is not None


class PhaseTimers:
    """Per-phase perf_counter totals; cheap no-ops when disabled."""

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self.log_summary = False  # set on TP0 by the scheduler
        self._lock = threading.Lock()
        self._since_take: Dict[str, List[float]] = {}  # name -> [n, sum_ms, max_ms]
        self._since_log: Dict[str, List[float]] = {}
        # count(): name -> [n, sum, max] of plain values (not times), since
        # the previous summary.
        self._counts: Dict[str, List[float]] = {}
        self._t_log = time.monotonic()
        # Launch gap (launch_begin / launch_end): phases since the last
        # launch end, name -> [n, sum_ms]; None outside a window.
        self.rank_label = ""
        self.gap_log_ms = float(_env_value("SGLANG_SCHED_LAUNCH_GAP_LOG_MS", 30.0))
        self.gap_log_per_min = int(_env_value("SGLANG_SCHED_LAUNCH_GAP_LOG_PER_MIN", 20))
        self._since_launch: Optional[Dict[str, List[float]]] = None
        self._last_launch_end: Optional[float] = None
        self._gap_window_t = time.monotonic()
        self._gap_lines = 0
        self._gap_suppressed = 0

    @staticmethod
    def start() -> float:
        return time.perf_counter()

    def stop(self, name: str, t0: float) -> None:
        if self.enabled:
            self.add(name, (time.perf_counter() - t0) * 1e3)

    def add(self, name: str, ms: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            for table in (self._since_take, self._since_log):
                s = table.setdefault(name, [0, 0.0, 0.0])
                s[0] += 1
                s[1] += ms
                s[2] = max(s[2], ms)
            window = self._since_launch
            if window is not None:
                w = window.setdefault(name, [0, 0.0])
                w[0] += 1
                w[1] += ms
        self.maybe_log()

    # -- launch gap -------------------------------------------------------
    def launch_begin(self, *, idle: bool = False, label: str = "") -> None:
        """Called at the start of a forward launch (Scheduler.run_batch)."""
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            last, self._last_launch_end = self._last_launch_end, None
            window, self._since_launch = self._since_launch, None
        if last is None:
            return
        gap_ms = (now - last) * 1e3
        self.add("launch_gap_after_idle" if idle else "launch_gap", gap_ms)
        if idle or gap_ms <= self.gap_log_ms:
            return
        self.count("launch_gap_over_threshold", gap_ms)
        mono = time.monotonic()
        with self._lock:
            if mono - self._gap_window_t >= SUMMARY_INTERVAL_S:
                if self._gap_suppressed:
                    logger.info(
                        "Scheduler launch gap: %d more gaps above %.0f ms not logged "
                        "in the previous window%s",
                        self._gap_suppressed,
                        self.gap_log_ms,
                        f" ({self.rank_label})" if self.rank_label else "",
                    )
                self._gap_window_t = mono
                self._gap_lines = 0
                self._gap_suppressed = 0
            if self._gap_lines >= self.gap_log_per_min:
                self._gap_suppressed += 1
                return
            self._gap_lines += 1
        parts = [
            f"{k}={v[1]:.1f}ms/{int(v[0])}"
            for k, v in sorted((window or {}).items(), key=lambda kv: -kv[1][1])
        ]
        logger.info(
            "Scheduler launch gap %.1f ms%s%s: %s",
            gap_ms,
            f" {self.rank_label}" if self.rank_label else "",
            f" before {label}" if label else "",
            ", ".join(parts) or "no timed phase",
        )

    def launch_end(self) -> None:
        """Called when a forward launch returns; opens the next gap window."""
        if not self.enabled:
            return
        with self._lock:
            self._last_launch_end = time.perf_counter()
            self._since_launch = {}

    def count(self, name: str, value: float) -> None:
        """Record one value (e.g. the leaves of a heap build) under ``name``;
        the 60 s summary shows n / mean / max per name."""
        if not self.enabled:
            return
        with self._lock:
            s = self._counts.setdefault(name, [0, 0, 0])
            s[0] += 1
            s[1] += value
            s[2] = max(s[2], value)
        self.maybe_log()

    def counts(self) -> Dict[str, List[float]]:
        """The count() table since the previous summary (a copy)."""
        with self._lock:
            return {k: list(v) for k, v in self._counts.items()}

    def take(self) -> Optional[Dict[str, float]]:
        """Summed ms per phase since the previous take (for one trace record)."""
        if not self.enabled:
            return None
        with self._lock:
            out = {k: round(v[1], 3) for k, v in self._since_take.items()}
            self._since_take = {}
        return out or None

    def maybe_log(self, now: Optional[float] = None) -> None:
        if not (self.enabled and self.log_summary):
            return
        now = time.monotonic() if now is None else now
        if now - self._t_log < SUMMARY_INTERVAL_S:
            return
        with self._lock:
            table, self._since_log = self._since_log, {}
            counts, self._counts = self._counts, {}
            dt, self._t_log = now - self._t_log, now
        if not table and not counts:
            return
        parts = [
            f"{k} n={int(v[0])} mean_ms={v[1] / max(v[0], 1):.3f} max_ms={v[2]:.3f} sum_ms={v[1]:.1f}"
            for k, v in sorted(table.items())
        ]
        parts += [
            f"{k} n={int(v[0])} mean={v[1] / max(v[0], 1):.1f} max={v[2]:g}"
            for k, v in sorted(counts.items())
        ]
        logger.info("Scheduler phase timers over %.0f s: %s", dt, "; ".join(parts))


PHASES = PhaseTimers(_env_flag("SGLANG_SCHED_PHASE_TIMERS"))


class _Span:
    __slots__ = ("timers", "name", "t0")

    def __init__(self, timers: PhaseTimers, name: str) -> None:
        self.timers = timers
        self.name = name
        self.t0 = 0.0

    def __enter__(self) -> "_Span":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        self.timers.add(self.name, (time.perf_counter() - self.t0) * 1e3)
        return False


_NULL_SPAN = contextlib.nullcontext()


def phase(name: str, *, timers: Optional[PhaseTimers] = None):
    """``with phase("name"):`` times the block; a shared null context when
    timers are off (no clock read, no lock)."""
    t = PHASES if timers is None else timers
    return _Span(t, name) if t.enabled else _NULL_SPAN


def launch_marker(*, timers: Optional[PhaseTimers] = None) -> Callable:
    """Decorator for Scheduler.run_batch: launch_begin before, launch_end after
    (launch gap). Identity when timers are off at import."""
    t = PHASES if timers is None else timers

    def deco(fn):
        if not t.enabled:
            return fn

        @functools.wraps(fn)
        def wrapper(self, batch, *args, **kwargs):
            mode = getattr(getattr(batch, "forward_mode", None), "name", "?")
            bs = len(getattr(batch, "reqs", ()) or ())
            t.launch_begin(
                idle=bool(getattr(self, "_sched_idled", False)),
                label=f"forward {getattr(self, 'forward_ct', 0) + 1} {mode} bs={bs}",
            )
            try:
                return fn(self, batch, *args, **kwargs)
            finally:
                t.launch_end()

        return wrapper

    return deco


def timed_phase(name: str, *, timers: Optional[PhaseTimers] = None) -> Callable:
    """Decorator: time every call under ``name``. Identity when timers are off
    at import (so the hot path is unchanged with the flag off)."""
    t = PHASES if timers is None else timers

    def deco(fn):
        if not t.enabled:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                t.add(name, (time.perf_counter() - t0) * 1e3)

        return wrapper

    return deco


class SyncDebug:
    """Attribute synchronizing CUDA calls to scheduler sites (sync-debug warn)."""

    def __init__(
        self, enabled: bool, max_stacks: int = 8, set_mode: Optional[Callable] = None
    ):
        self.enabled = bool(enabled)
        self.max_stacks = max(0, int(max_stacks))
        self._local = threading.local()
        self._lock = threading.Lock()
        self._depth = 0  # open sites (process-wide; the mode is global)
        self.counts: Dict[str, int] = {}
        self.stacks: Dict[str, List[tuple]] = {}
        self._installed = False
        self._set_mode = set_mode  # tests inject a fake; None = torch.cuda
        self._period_counts: Dict[str, int] = {}
        self._t_log = time.monotonic()

    def _mode(self, mode: str) -> None:
        if self._set_mode is not None:
            self._set_mode(mode)
            return
        import torch

        torch.cuda.set_sync_debug_mode(mode)

    def _install(self) -> None:
        if self._installed:
            return
        self._installed = True
        warnings.filterwarnings("always", message=f".*{_SYNC_MSG}.*")
        prev = warnings.showwarning

        def showwarning(message, category, filename, lineno, file=None, line=None):
            if _SYNC_MSG in str(message):
                self.record(traceback.extract_stack()[:-1])
                return
            prev(message, category, filename, lineno, file, line)

        warnings.showwarning = showwarning

    # Bound on distinct keys (debug only): short-lived threads with unique
    # names would otherwise each add an "other@<thread>" entry for good.
    MAX_SITES = 256

    def current_site(self) -> str:
        site = getattr(self._local, "site", None)
        if site:
            return site
        key = f"other@{threading.current_thread().name}"
        if key not in self.counts and len(self.counts) >= self.MAX_SITES:
            return "other@(more threads)"
        return key

    def record(self, frames) -> None:
        site = self.current_site()
        sig = tuple(
            (f.filename.rsplit("/decoupled_engine/", 1)[-1], f.lineno, f.name)
            for f in frames
            if "/decoupled_engine/" in f.filename and "phase_timers.py" not in f.filename
        )[-8:]
        new = False
        with self._lock:
            self.counts[site] = self.counts.get(site, 0) + 1
            self._period_counts[site] = self._period_counts.get(site, 0) + 1
            seen = self.stacks.setdefault(site, [])
            if sig not in seen and len(seen) < self.max_stacks:
                seen.append(sig)
                new = True
        if new:
            logger.warning(
                "Sync debug [%s]: synchronizing CUDA call #%d at %s",
                site,
                len(self.stacks[site]),
                " <- ".join(f"{fn}:{ln}({nm})" for fn, ln, nm in reversed(sig))
                or "<no sglang frame>",
            )

    def enter(self, name: str):
        prev = getattr(self._local, "site", None)
        self._local.site = name
        with self._lock:
            self._depth += 1
            first = self._depth == 1
        if first:
            self._install()
            self._mode("warn")
        return prev

    def exit(self, prev) -> None:
        with self._lock:
            self._depth -= 1
            last = self._depth == 0
        if last:
            self._mode("default")
        self._local.site = prev
        self.maybe_log()

    def maybe_log(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        if now - self._t_log < SUMMARY_INTERVAL_S:
            return
        with self._lock:
            counts, self._period_counts = self._period_counts, {}
            dt, self._t_log = now - self._t_log, now
        if counts:
            logger.info(
                "Sync debug counts over %.0f s: %s",
                dt,
                "; ".join(f"{k}={v}" for k, v in sorted(counts.items())),
            )


SYNC_DEBUG = SyncDebug(
    _env_flag("SGLANG_SCHED_SYNC_DEBUG"),
    max_stacks=int(_env_value("SGLANG_SCHED_SYNC_DEBUG_STACKS", 8)),
)


def sync_debug_site(name: str, *, debug: Optional[SyncDebug] = None) -> Callable:
    """Decorator: run the function as sync-debug site ``name``. Identity when
    sync debug is off at import."""
    d = SYNC_DEBUG if debug is None else debug

    def deco(fn):
        if not d.enabled:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            prev = d.enter(name)
            try:
                return fn(*args, **kwargs)
            finally:
                d.exit(prev)

        return wrapper

    return deco
