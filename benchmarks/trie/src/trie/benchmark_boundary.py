"""Fixed-window request admission and bounded shutdown for replay experiments."""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections import Counter
from pathlib import Path


class BenchmarkWindowClosed(asyncio.CancelledError):
    """Expected stop of one workload at the benchmark admission boundary."""


class BenchmarkCancellationTimeout(RuntimeError):
    """Some workload tasks ignored cancellation beyond the cleanup budget."""


def clock_point() -> dict[str, int]:
    before = time.perf_counter_ns()
    wall = time.time_ns()
    mono = time.monotonic_ns()
    after = time.perf_counter_ns()
    return {"ts_ns": wall, "mono_ns": mono, "perf_ns": before,
            "clock_sample_span_ns": after - before}


class BenchmarkBoundary:
    def __init__(self, duration: float, drain_timeout: float, *, start: dict | None = None):
        self.start = clock_point() if start is None else start
        self.deadline = {k: self.start[k] + round(duration * 1e9)
                         for k in ("ts_ns", "mono_ns", "perf_ns")}
        self.zero_drain_guard = drain_timeout == 0
        self.request_stop = False
        self.counters: Counter[str] = Counter()
        path = os.getenv("TRIE_BENCHMARK_TIMING_FILE")
        self.path = Path(path) if path else None
        self.cleanup_timeout = float(os.getenv("TRIE_BENCHMARK_CANCEL_TIMEOUT_S", "10"))
        if not math.isfinite(self.cleanup_timeout) or self.cleanup_timeout <= 0:
            raise ValueError("TRIE_BENCHMARK_CANCEL_TIMEOUT_S must be finite and positive")
        self.data: dict[str, object] = {
            "schema_version": "trie.benchmark-boundary.v1", "status": "running",
            "start": self.start, "deadline": self.deadline,
            "deadline_clock": "monotonic_ns", "wall_deadline_is_projection": True,
            "duration_s": duration, "drain_timeout_s": drain_timeout,
            "zero_drain_request_guard": self.zero_drain_guard,
            "cancel_cleanup_timeout_s": self.cleanup_timeout,
            "admission_stop": None, "request_admission_stop": None,
            "cancel_start": None, "final": None,
            "clock_info": {
                name: {"implementation": time.get_clock_info(name).implementation,
                       "monotonic": time.get_clock_info(name).monotonic,
                       "resolution_s": time.get_clock_info(name).resolution}
                for name in ("monotonic", "perf_counter")
            },
        }

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        last_admit = self.data.get("last_lifecycle_admission")
        payload = {
            **self.data, "counters": dict(self.counters),
            "valid": self.data["status"] == "completed",
            "benchmark_start_wall_ns": self.start["ts_ns"],
            "benchmark_start_monotonic_ns": self.start["mono_ns"],
            "benchmark_deadline_wall_ns": self.deadline["ts_ns"],
            "benchmark_deadline_monotonic_ns": self.deadline["mono_ns"],
            "last_admission_wall_ns": last_admit["ts_ns"] if last_admit else None,
            "last_admission_monotonic_ns": last_admit["mono_ns"] if last_admit else None,
        }
        temporary.write_text(json.dumps(payload,
                                        indent=2, sort_keys=True) + "\n")
        temporary.replace(self.path)

    def check(self, stage: str) -> dict[str, int]:
        point = clock_point()
        if self.request_stop or (self.zero_drain_guard and
                                 point["mono_ns"] >= self.deadline["mono_ns"]):
            self.counters["boundary_guard_stops"] += 1
            self.counters[f"boundary_guard_stops_{stage}"] += 1
            raise BenchmarkWindowClosed("benchmark request admission has stopped")
        if stage in ("lifecycle_admission", "http_send"):
            self.counters[f"{stage}_count"] += 1
            self.data[f"last_{stage}"] = point
        return point

    def stop_trace_admission(self, reason: str) -> None:
        self.data["admission_stop"] = {**clock_point(), "reason": reason}
        self.save()

    def stop_requests(self) -> None:
        self.request_stop = True
        self.data["request_admission_stop"] = clock_point()
        self.data["cancel_start"] = clock_point()
        self.save()

    def __enter__(self):
        self.save()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.request_stop = True
        self.data["final"] = clock_point()
        self.data["status"] = "completed" if exc is None else "failed"
        if exc is not None:
            self.data["error"] = {"class": type(exc).__name__, "message": str(exc)}
        self.save()


async def cancel_workloads(tasks: set[asyncio.Task], boundary: BenchmarkBoundary) -> None:
    """Never wait indefinitely for a workload that suppresses cancellation."""
    boundary.stop_requests()
    for task in tasks:
        if not task.done() and task.cancel():
            boundary.counters["cancel_requested_tasks"] += 1
    if not tasks:
        boundary.counters["pending_tasks_after_cancel"] = 0
        return
    done, pending = await asyncio.wait(tasks, timeout=boundary.cleanup_timeout)
    boundary.counters["pending_tasks_after_cancel"] = len(pending)
    if pending:
        boundary.counters["cancel_cleanup_timeouts"] += 1
        raise BenchmarkCancellationTimeout(
            f"{len(pending)} workload task(s) remain after "
            f"{boundary.cleanup_timeout:g}s cancellation cleanup; benchmark invalid"
        )
    for task in done:
        if not task.cancelled():
            task.result()  # A task exception is not a successful benchmark.
