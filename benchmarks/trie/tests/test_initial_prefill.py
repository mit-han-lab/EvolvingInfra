import json
from types import SimpleNamespace

import pytest

from trie.initial_prefill import InitialPrefillReadiness, pending_traces
from trie.types import Generate, ReplayTrace


def trace(name, *times):
    return ReplayTrace(name, [Generate(max_tokens=1, source_t_s=value) for value in times])


def tracker(tmp_path, count):
    boundary = SimpleNamespace(start={"ts_ns": 1, "mono_ns": 1},
                               deadline={"ts_ns": 10**20, "mono_ns": 10**20})
    return InitialPrefillReadiness(tmp_path, 0, count, boundary)


def test_only_traces_spanning_cut_need_artificial_prefix_prefill():
    traces = [trace("finished", 0, 1), trace("future", 10, 11), trace("active", 0, 10)]
    assert pending_traces(traces, 5) == {"active"}
    assert pending_traces(traces, None) == set()


def test_every_initial_slot_and_agent_prefix_must_be_ready(tmp_path):
    state = tracker(tmp_path, 2)
    first, second = state.slot(0), state.slot(1)
    first.register([trace("a", 0, 10), trace("b", 1, 10)], 5)
    first.completed("a")
    second.register([trace("future", 10)], 5)
    assert json.loads(state.path.read_text())["ready"] is False
    first.completed("b")
    row = json.loads(state.path.read_text())
    assert row["ready"] and row["ready_slots"] == 2
    first.completed("b")  # A duplicate notification must not advance another slot.
    assert json.loads(state.path.read_text())["ready_slots"] == 2


def test_stale_shard_marker_is_rejected(tmp_path):
    tracker(tmp_path, 1)
    with pytest.raises(FileExistsError):
        tracker(tmp_path, 1)
