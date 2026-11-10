"""Start/completion dependencies between agent events, local to one replay task."""

import asyncio
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trie.types import ReplayEvent, ReplayTask, ReplayTrace

EventKey = tuple[str, int]


def validate_dependencies(task: "ReplayTask") -> None:
    """Check references and cycles, including each agent's implicit sequence."""
    traces = {trace.trace_id: trace for trace in task.traces}
    if len(traces) != len(task.traces):
        raise ValueError("multi-agent task trace IDs must be unique")
    nodes = {
        (trace.trace_id, i) for trace in task.traces for i in range(len(trace.events))
    }
    # Separate start/end vertices: a child can start during a parent event,
    # but completion dependencies must still reject real deadlocks.
    phases = {(key, phase) for key in nodes for phase in ("start", "complete")}
    successors = defaultdict(set)
    indegree = dict.fromkeys(phases, 0)

    def connect(before, after) -> None:
        if before not in phases or after not in phases:
            raise ValueError(f"unknown dependency event: {before!r} -> {after!r}")
        if after not in successors[before]:
            successors[before].add(after)
            indegree[after] += 1

    for trace in task.traces:
        if trace.start_offset_s != 0:
            raise ValueError(
                "agents inside a task must have start_offset_s=0; use wait "
                "events after dependencies for tool/dispatch delays"
            )
        for i in range(len(trace.events)):
            key = (trace.trace_id, i)
            connect((key, "start"), (key, "complete"))
            if i:
                connect(((trace.trace_id, i - 1), "complete"), (key, "start"))
    seen = set()
    for dependency in task.dependencies:
        edge = ((dependency.before.key, dependency.when), (dependency.after.key, "start"))
        if edge in seen:
            raise ValueError(f"duplicate dependency: {edge!r}")
        seen.add(edge)
        connect(*edge)
    ready = deque(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        node = ready.popleft()
        visited += 1
        for successor in successors[node]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if visited != len(phases):
        raise ValueError("multi-agent dependencies contain a cycle")


class DependencyState:
    """Allocate signals only for cross-event edges, once per task instance."""

    def __init__(self, task: "ReplayTask") -> None:
        self.incoming: dict[EventKey, list[asyncio.Event]] = defaultdict(list)
        self.completed: dict[EventKey, asyncio.Event] = {}
        self.started: dict[EventKey, asyncio.Event] = {}
        for dependency in task.dependencies:
            key = dependency.before.key
            signals = self.started if dependency.when == "start" else self.completed
            if key not in signals:
                signals[key] = asyncio.Event()
            self.incoming[dependency.after.key].append(signals[key])


async def replay_events(
    trace: "ReplayTrace", state: DependencyState | None
) -> AsyncIterator["ReplayEvent"]:
    for index, event in enumerate(trace.events):
        key = (trace.trace_id, index)
        if state is not None:
            for completed in state.incoming.get(key, ()):
                await completed.wait()
        # Client event start (before request preparation), not engine admission.
        if state is not None and key in state.started:
            state.started[key].set()
        yield event
        # The caller resumes iteration only after successfully processing the
        # event, including response verification. Exceptions never signal it.
        if state is not None and key in state.completed:
            state.completed[key].set()
