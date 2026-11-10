"""Optional readiness evidence for the initial stationary-prefix prefills.

This observes the existing replay: it does not block requests, move the benchmark
clock, change admission, or treat a failed warmup as a successful one.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from trie.types import Generate


def pending_traces(traces, cut_s):
    if cut_s is None:
        return set()
    pending = set()
    for trace in traces:
        times = [event.source_t_s for event in trace.events if isinstance(event, Generate)]
        if any(value is None for value in times):
            raise ValueError("Initial-prefix readiness requires recorded generate clocks")
        if any(value < cut_s for value in times) and any(value >= cut_s for value in times):
            pending.add(trace.trace_id)
    return pending


class InitialPrefillReadiness:
    def __init__(self, directory, shard, expected_slots, boundary):
        if expected_slots <= 0:
            raise ValueError("Initial-prefix readiness needs positive concurrency")
        self.path = Path(directory) / f"shard-{shard}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError("Initial-prefix readiness must not reuse an old shard")
        self.expected = expected_slots
        self.pending = {}
        self.failed = {}
        self.ready_ns = None
        self.start_ns = boundary.start["ts_ns"]
        self.deadline_ns = boundary.deadline["ts_ns"]
        self.start_mono_ns = boundary.start["mono_ns"]
        self.deadline_mono_ns = boundary.deadline["mono_ns"]
        self.write()

    def slot(self, index):
        if not 0 <= index < self.expected:
            raise ValueError("Initial slot outside declared concurrency")
        return InitialSlot(self, index)

    def write(self):
        completed = sum(not pending and index not in self.failed for index, pending in self.pending.items())
        ready = len(self.pending) == self.expected and completed == self.expected and not self.failed
        if ready and self.ready_ns is None:
            self.ready_ns = time.time_ns()
        row = {"pid": os.getpid(), "ready": ready, "expected_slots": self.expected,
               "registered_slots": len(self.pending), "ready_slots": completed,
               "pending_prefills": sum(map(len, self.pending.values())),
               "failed_slots": self.failed, "initial_prefill_ready_ns": self.ready_ns,
               "benchmark_start_ns": self.start_ns, "benchmark_deadline_ns": self.deadline_ns,
               "benchmark_start_monotonic_ns": self.start_mono_ns,
               "benchmark_deadline_monotonic_ns": self.deadline_mono_ns}
        temporary = self.path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(row) + "\n")
        temporary.replace(self.path)


class InitialSlot:
    def __init__(self, tracker, index):
        self.tracker, self.index = tracker, index

    def register(self, traces, cut_s):
        if self.index in self.tracker.pending:
            raise ValueError("Initial slot registered twice")
        self.tracker.pending[self.index] = pending_traces(traces, cut_s)
        self.tracker.write()

    def completed(self, trace_id):
        self.tracker.pending[self.index].discard(trace_id)
        self.tracker.write()

    def failed(self, error):
        # A later normal-replay failure belongs in the existing request metrics;
        # it does not retroactively change successful initial-prefix readiness.
        if self.index not in self.tracker.pending or self.tracker.pending[self.index]:
            self.tracker.failed[self.index] = type(error).__name__
            self.tracker.write()
