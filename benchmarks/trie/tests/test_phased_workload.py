import asyncio
import json
import random
import time

import pytest

from trie.client import Client
from trie.client_lifecycle import ClientLifecycleWriter
from trie.phased_workload import Schedule, shared_clock
from trie.types import ReplayTask, ReplayTrace, Wait
from test_multiagent_replay import ControlledClient, edge, generate


def manifest(tmp_path, durations=(.2, .3, .3)):
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps({
        "workloads": {"codex": "codex.jsonl", "putnam": "putnam.jsonl"},
        "phases": [
            {"duration_s": d, "slots": {"codex": c, "putnam": p}}
            for d, (c, p) in zip(durations, [(2, 0), (1, 1), (0, 2)])
        ],
    }))
    return path


def conversation(offset):
    return ReplayTask("same-task-id", traces=[
        ReplayTrace("main", [generate(offset + 1), Wait(seconds=3600, actor="tool"), generate(offset + 2)]),
        ReplayTrace("child", [generate(offset + 3)]),
    ], dependencies=[edge("main", 0, "child", 0), edge("child", 0, "main", 1)])


@pytest.mark.parametrize("salt", ["session", "global"])
def test_switch_waits_for_inflight_child_but_not_entire_task(tmp_path, monkeypatch, salt):
    """A parent waiting on a child and an hour-long tool wait must not stall switching."""
    schedule = manifest(tmp_path)
    sources = {"codex.jsonl": [conversation(0), conversation(100)],
               "putnam.jsonl": [ReplayTrace("same-task-id", [generate(1001), Wait(seconds=3600, actor="tool")])]}
    monkeypatch.setattr("trie.client.load_workloads", lambda path, **kw: sources[path.rsplit("/", 1)[-1]])

    async def scenario():
        writer = ClientLifecycleWriter(tmp_path / "events.jsonl", "test", "mixed", "off", 2)
        client = ControlledClient(blocked=[103], writer=writer)
        execution = asyncio.create_task(client.run("codex.jsonl", concurrency=2, duration=.8,
            strict_replay=True, cache_salt_mode=salt, workload_schedule=str(schedule)))
        try:
            await asyncio.wait_for(client.started[103].wait(), 1)
            await asyncio.sleep(.28)
            assert not client.started[1001].is_set(), "New source used a slot with a request still in flight"
            client.release[103].set()
            result = await asyncio.wait_for(execution, 2)
        finally:
            await client._client.close()
        assert result.failed_requests == 0
        assert result.completed_requests == 0, "Paused conversations must not count as completed tasks"
        assert client.finished[103].is_set()
        assert not client.started[2].is_set() and not client.started[102].is_set()
        events = [json.loads(line) for line in writer.output_path.read_text().splitlines()]
        assert not [e for e in events if e["event"] in {"REQUEST_FAILED", "TASK_CANCELLED"}]
        assigns = [e for e in events if e["event"] == "SLOT_ASSIGN"]
        assert [(e["attrs"]["workload_slot"], e["attrs"]["workload_label"]) for e in assigns] == [
            (0, "codex"), (1, "codex"), (1, "putnam"), (0, "putnam")]
        slow_admit = next(e for e in events if e["event"] == "CLIENT_ADMIT"
                          and e["attrs"]["workload_slot"] == 1 and e["attrs"]["workload_label"] == "codex"
                          and e["attrs"]["agent_id"] == "child")
        slow_complete = next(e for e in events if e["event"] == "CLIENT_COMPLETE"
                             and e["request"]["rid"] == slow_admit["request"]["rid"])
        assert assigns[2]["mono_ns"] >= slow_complete["mono_ns"]
        admits = [e for e in events if e["event"] == "CLIENT_ADMIT"]
        assert len({e["request"]["rid"] for e in admits}) == len(admits)
        assert {e["attrs"]["workload_label"] for e in admits} == {"codex", "putnam"}
        assert len([e for e in events if e["event"] == "WORKLOAD_PHASE_START"]) == 3

    asyncio.run(scenario())


def test_schedule_rejects_initial_capacity_and_duration_mismatch(tmp_path):
    path = manifest(tmp_path)
    with pytest.raises(ValueError, match="concurrency"):
        Schedule.load(str(path), 3, .8)
    with pytest.raises(ValueError, match="duration"):
        Schedule.load(str(path), 2, 100)


def test_variable_targets_and_idle_slot_boundaries(tmp_path):
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    data["phases"][1]["slots"] = {"codex": 1, "putnam": 0}
    path.write_text(json.dumps(data))
    schedule = Schedule.load(str(path), 2, .8)
    assert schedule.label(1, 1) is None
    assert schedule.change_at(1, 1) == pytest.approx(.5)
    assert schedule.label(2, 1) == "putnam"
    with pytest.raises(ValueError, match="outside"):
        schedule.label(1, 2)
    data["phases"][1]["slots"] = {"codex": 2, "putnam": 1}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="capacity"):
        Schedule.load(str(path), 2, .8)


def test_shrinking_load_drains_request_parks_and_reactivates(tmp_path, monkeypatch):
    path = manifest(tmp_path, durations=(.3, .4, .5))
    data = json.loads(path.read_text())
    data["phases"][1]["slots"] = {"codex": 1, "putnam": 0}
    path.write_text(json.dumps(data))
    sources = {"codex.jsonl": [conversation(0), conversation(100)],
               "putnam.jsonl": [ReplayTrace("putnam", [generate(1001), Wait(seconds=3600, actor="tool")])]}
    monkeypatch.setattr("trie.client.load_workloads", lambda path, **kw: sources[path.rsplit("/", 1)[-1]])

    async def scenario():
        writer = ClientLifecycleWriter(tmp_path / "events.jsonl", "test", "variable", "off", 2)
        client = ControlledClient(blocked=[103], writer=writer)
        execution = asyncio.create_task(client.run("codex.jsonl", concurrency=2, duration=1.2,
            strict_replay=True, cache_salt_mode="session", workload_schedule=str(path)))
        try:
            await asyncio.wait_for(client.started[103].wait(), 1)
            await asyncio.sleep(.4)
            assert not client.finished[103].is_set()
            client.release[103].set()
            await asyncio.sleep(.1)
            assert not client.started[1001].is_set(), "Idle slot admitted the next phase early"
            result = await asyncio.wait_for(execution, 2)
        finally:
            await client._client.close()
        assert result.failed_requests == 0
        events = [json.loads(line) for line in writer.output_path.read_text().splitlines()]
        assert not [e for e in events if e["event"] in {"REQUEST_FAILED", "TASK_CANCELLED"}]
        idle = [e for e in events if e["event"] == "SLOT_IDLE"]
        assert len(idle) == 1 and idle[0]["attrs"]["workload_slot"] == 1
        child = next(e for e in events if e["event"] == "CLIENT_ADMIT"
                     and e["attrs"]["workload_slot"] == 1 and e["attrs"].get("agent_id") == "child")
        complete = next(e for e in events if e["event"] == "CLIENT_COMPLETE"
                        and e["request"]["rid"] == child["request"]["rid"])
        assert idle[0]["mono_ns"] >= complete["mono_ns"]
        activated = [e for e in events if e["event"] == "SLOT_ASSIGN"
                     and e["attrs"]["workload_label"] == "putnam"]
        assert {e["attrs"]["workload_slot"] for e in activated} == {0, 1}
        assert all(e["mono_ns"] > idle[0]["mono_ns"] for e in activated)

    asyncio.run(scenario())


def test_shards_share_one_start_clock(tmp_path, monkeypatch):
    monkeypatch.setenv("TRIE_PHASE_CLOCK_DIR", str(tmp_path / "clock"))

    async def scenario():
        a, b = await asyncio.gather(shared_clock(0, 2), shared_clock(1, 2))
        assert a == b
        assert time.perf_counter_ns() >= a["perf_ns"]

    asyncio.run(scenario())


def test_slots_replay_the_same_templates_and_cuts_for_any_process_count(tmp_path, monkeypatch):
    """A global slot gets the same templates and warm-start cuts with one client process or several."""
    path = manifest(tmp_path, durations=(.2, .2, .2))
    sources = {"codex.jsonl": [ReplayTrace(f"c{i}", [generate(i + 1)]) for i in range(3)],
               "putnam.jsonl": [ReplayTrace(f"p{i}", [generate(i + 101)]) for i in range(5)]}
    monkeypatch.setattr("trie.client.load_workloads", lambda path, **kw: sources[path.rsplit("/", 1)[-1]])

    async def shard(tag, index, count):
        writer = ClientLifecycleWriter(tmp_path / f"{tag}-{index}.jsonl", "test", tag, "off", 2 // count)
        client = ControlledClient(writer=writer)
        client._rng = random.Random(0)
        seen = {}

        async def record(item, *, replay_cycle_id, warm_start_draw, **kwargs):
            seen[divmod(replay_cycle_id, 1_000_000_000)] = (item.trace_id, warm_start_draw)
            await asyncio.sleep(.03)

        client._run_workload = record
        try:
            await client.run("codex.jsonl", concurrency=2 // count, duration=.6, strict_replay=True,
                             cache_salt_mode="global", warm_start="stationary", shard_index=index,
                             shard_count=count, workload_schedule=str(path))
        finally:
            await client._client.close()
        return seen

    async def replay(tag, count):
        monkeypatch.setenv("TRIE_PHASE_CLOCK_DIR", str(tmp_path / f"clock-{tag}"))
        merged = {}
        for seen in await asyncio.gather(*(shard(tag, i, count) for i in range(count))):
            merged.update(seen)
        return merged

    one = asyncio.run(replay("one", 1))
    two = asyncio.run(replay("two", 2))
    for slot in (0, 1):
        assert sum(1 for s, _ in one if s == slot) >= 6 and sum(1 for s, _ in two if s == slot) >= 6
    common = one.keys() & two.keys()
    assert len(common) >= 12
    assert {key: one[key] for key in common} == {key: two[key] for key in common}
    assert {trace for trace, _ in one.values()} == {f"c{i}" for i in range(3)} | {f"p{i}" for i in range(5)}


def test_only_slots_of_the_first_phase_start_at_a_warm_cut(tmp_path, monkeypatch):
    """Sessions that join in a later phase replay from their first turn."""
    path = manifest(tmp_path, durations=(.2, .2, .2))
    sources = {"codex.jsonl": [ReplayTrace(f"c{i}", [generate(i + 1)]) for i in range(3)],
               "putnam.jsonl": [ReplayTrace(f"p{i}", [generate(i + 101)]) for i in range(5)]}
    monkeypatch.setattr("trie.client.load_workloads", lambda path, **kw: sources[path.rsplit("/", 1)[-1]])

    async def scenario():
        writer = ClientLifecycleWriter(tmp_path / "events.jsonl", "test", "cuts", "off", 2)
        client = ControlledClient(writer=writer)
        draws = []

        async def record(item, *, warm_start_draw, **kwargs):
            draws.append((item.trace_id[0], warm_start_draw))
            await asyncio.sleep(.03)

        client._run_workload = record
        try:
            await client.run("codex.jsonl", concurrency=2, duration=.6, strict_replay=True,
                             cache_salt_mode="global", warm_start="stationary", workload_schedule=str(path))
        finally:
            await client._client.close()
        # Phase 0 is two codex slots; putnam slots appear only in later phases.
        assert [d for d in draws if d[1] is not None and d[0] == "c"] and len(
            [d for d in draws if d[1] is not None]) == 2
        assert all(draw is None for source, draw in draws if source == "p")

    asyncio.run(scenario())
