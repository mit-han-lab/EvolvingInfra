"""Time-based workload composition, switching only between model requests."""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

class PhaseEnded(Exception):
    """A slot stops this conversation without failing or cancelling a request."""


@dataclass
class Schedule:
    workloads: dict[str, str]
    phases: list[dict]
    concurrency: int

    @classmethod
    def load(cls, path: str, concurrency: int, duration: float) -> "Schedule":
        obj = json.loads(Path(path).read_text())
        sources, phases = obj["workloads"], obj["phases"]
        if not sources or not phases:
            raise ValueError("A workload schedule needs sources and phases")
        if any(not re.fullmatch(r"[a-z][a-z0-9_-]*", name) for name in sources):
            raise ValueError("Workload labels must be simple lowercase names")
        sources = {name: str((Path(path).parent / value).resolve()) for name, value in sources.items()}
        for phase in phases:
            seconds, slots = phase["duration_s"], phase["slots"]
            if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("Phase duration must be finite and positive")
            if set(slots) != set(sources) or any(type(n) is not int or n < 0 for n in slots.values()):
                raise ValueError("Every phase must specify nonnegative integer slots for each source")
            if sum(slots.values()) > concurrency:
                raise ValueError("Phase slots exceed the declared concurrency capacity")
        if sum(phases[0]["slots"].values()) != concurrency:
            raise ValueError("The first phase must fill concurrency capacity for initial-prefix readiness")
        if not math.isclose(sum(p["duration_s"] for p in phases), duration, abs_tol=1e-6):
            raise ValueError("Phase durations must sum to the benchmark duration")
        return cls(sources, phases, concurrency)

    def phase_index(self, elapsed: float) -> int:
        end = 0.0
        for index, phase in enumerate(self.phases):
            end += phase["duration_s"]
            if elapsed < end:
                return index
        return len(self.phases)

    def label(self, phase: int, slot: int) -> str | None:
        if not 0 <= slot < self.concurrency:
            raise ValueError("Slot outside schedule")
        offset = 0
        for label in self.workloads:
            offset += self.phases[phase]["slots"][label]
            if slot < offset:
                return label
        return None  # An unused slot waits until a later phase activates it.

    def change_at(self, phase: int, slot: int) -> float:
        label = self.label(phase, slot)
        end = sum(p["duration_s"] for p in self.phases[:phase + 1])
        for next_phase in range(phase + 1, len(self.phases)):
            if self.label(next_phase, slot) != label:
                break
            end += self.phases[next_phase]["duration_s"]
        return end


@dataclass
class Slot:
    schedule: Schedule
    index: int
    label: str
    start: float
    until: float

    def check(self) -> None:
        if time.perf_counter() >= self.until:
            raise PhaseEnded()

    def attrs(self) -> dict:
        return {"workload_label": self.label, "workload_slot": self.index,
                "workload_phase": self.schedule.phase_index(time.perf_counter() - self.start)}


current_slot: contextvars.ContextVar[Slot | None] = contextvars.ContextVar("replay_phase_slot", default=None)


def check_phase() -> None:
    slot = current_slot.get()
    if slot is not None:
        slot.check()


async def phase_wait(awaitable):
    """Interrupt only client waits/dependencies. Never wrap an HTTP request."""
    slot = current_slot.get()
    if slot is None:
        return await awaitable
    task = asyncio.ensure_future(awaitable)
    try:
        remaining = slot.until - time.perf_counter()
        if remaining <= 0:
            raise PhaseEnded()
        done, _ = await asyncio.wait({task}, timeout=remaining)
        if not done:
            raise PhaseEnded()
        return task.result()
    finally:
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def phase_events(events):
    if current_slot.get() is None:
        async for event in events:
            yield event
        return
    try:
        while True:
            check_phase()
            try:
                event = await phase_wait(anext(events))
            except StopAsyncIteration:
                return
            check_phase()
            yield event
    finally:
        await events.aclose()


async def shared_clock(shard_index: int, shard_count: int) -> dict | None:
    """Same-host client shards share a clock after all source lists are ready."""
    from trie.benchmark_boundary import clock_point
    directory = os.getenv("TRIE_PHASE_CLOCK_DIR")
    if not directory:
        if shard_count != 1:
            raise ValueError("Phased replay shards require TRIE_PHASE_CLOCK_DIR")
        return None
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"ready-{shard_index}").write_text(str(os.getpid()))
    deadline = time.monotonic() + 120
    while len(list(root.glob("ready-*"))) != shard_count:
        if time.monotonic() >= deadline:
            raise TimeoutError("Phased replay shards did not reach the start barrier")
        await asyncio.sleep(.05)
    start_file = root / "start.json"
    if shard_index == 0:
        point = clock_point()
        point.update({key: point[key] + 2_000_000_000 for key in ("ts_ns", "mono_ns", "perf_ns")})
        tmp = root / "start.tmp"
        tmp.write_text(json.dumps(point))
        tmp.replace(start_file)
    while not start_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Missing shared replay start clock")
        await asyncio.sleep(.01)
    point = json.loads(start_file.read_text())
    await asyncio.sleep(max(0, (point["perf_ns"] - time.perf_counter_ns()) / 1e9))
    return point
