"""CC replay: semantic timing, deletion and unchanged payload checks."""

import asyncio
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from strict_convert.adapters.cc_weka import Chain, Session, build_row
from strict_convert.adapters.cc_timing import apply_policy, check_policy, simulate
from trie.dependencies import DependencyState, replay_events
from trie.types import Dependency, EventRef, Generate, ReplayTask, ReplayTrace, Wait


def req(rid, t, api=1, think=0, out=2, hashes=None):
    return dict(
        id=rid,
        t=t,
        api_time=api,
        think_time=think,
        out=out,
        hash_ids=hashes or [1],
        **{"in": 64 * len(hashes or [1]), "model": "claude"}
    )


def convert(chains, zero="drop-advance"):
    session = Session(
        "session",
        {c.chain_id: c for c in chains},
        sum(len(c.requests) for c in chains),
        0,
    )
    base = build_row(session, list(range(20, 100)), "test", {}, {})
    row = apply_policy(session, base, zero_output=zero)
    assert not check_policy(session, row)["errors"]
    ReplayTask.from_jsonl_row(
        row
    )  # real parser checks control-only agents and cycle graph
    return row


def test_zero_middle_promotes_same_chain_without_changing_payload():
    chain = Chain(
        "main",
        "main",
        None,
        [
            req("a", 0),
            req("zero", 3, think=2, out=0),
            req("next", 10, api=2, think=6, out=7, hashes=[1, 2]),
            req("later", 15, think=3),
        ],
    )
    row = convert([chain])
    generates = [e for e in row["traces"][0]["events"] if e["type"] == "generate"]
    assert [e["source_request_id"] for e in generates] == ["a", "next", "later"]
    assert generates[1]["max_tokens"] == 7
    assert generates[1]["recorded_prompt_block_runs"] == [[1, 2]]
    assert generates[1]["source_t_s"] == 10  # original evidence never rewritten
    slots = row["metadata"]["cc_replay_policy"]["request_slots"]
    assert [s["effective_recorded_start_s"] for s in slots] == [0, 3, 8]


def test_deleted_tail_fork_and_empty_child_keep_join_but_no_model_call():
    parent = Chain(
        "main", "main", None, [req("a", 0), req("zero", 3, out=0)], children=["child"]
    )
    child = Chain(
        "child",
        "derived",
        "main",
        [req("b", 5)],
        anchor=1,
        join_mode="end",
        subtree_end=6,
    )
    row = convert([parent, child])
    assert sum(e["type"] == "generate" for t in row["traces"] for e in t["events"]) == 2
    assert any(
        e["before"] == {"trace_id": "main", "event_index": 3}
        for e in row["dependencies"]
    )
    assert simulate(row)["child", 1, "start"] == 2  # tail wait + zero API removed
    parent = Chain("main", "main", None, [req("a", 0), req("b", 6)], children=["child"])
    child = Chain(
        "child",
        "derived",
        "main",
        [req("zero", 3, out=0)],
        anchor=0,
        join_before=1,
        subtree_end=4,
    )
    row = convert([parent, child])
    assert row["traces"][1]["metadata"]["control_only"] is True
    assert all(e["type"] == "wait" for e in row["traces"][1]["events"])


def test_zero_fork_then_successor_join_cannot_be_collapsed_into_cycle():
    parent = Chain(
        "main",
        "main",
        None,
        [req("zero", 0, out=0), req("next", 6)],
        children=["child"],
    )
    child = Chain(
        "child",
        "derived",
        "main",
        [req("b", 2, api=3)],
        anchor=0,
        join_before=1,
        subtree_end=5,
    )
    row = convert([parent, child])
    times = simulate(row)
    assert times["main", 3, "start"] == times["child", 1, "complete"]


def test_start_anchored_fork_waits_for_live_parent_start_and_correct_offset():
    parent = Chain(
        "main", "main", None, [req("a", 0), req("b", 100)], children=["child"]
    )
    child = Chain(
        "child",
        "derived",
        "main",
        [req("c", 100.5, api=0.1)],
        anchor=0,
        subtree_end=100.6,
    )
    row = convert([parent, child])
    assert any(
        e.get("when") == "start" and e["before"]["event_index"] == 3
        for e in row["dependencies"]
    )
    assert row["traces"][1]["events"][0]["seconds"] == 0.5
    assert simulate(row)["child", 1, "start"] == 100.5
    assert (
        ReplayTask.from_jsonl_row(row).to_jsonl_row()["dependencies"]
        == row["dependencies"]
    )


def test_join_subtracts_child_compute_and_ignores_stale_first_think():
    parent = Chain(
        "main", "main", None, [req("a", 0), req("b", 10, think=6)], children=["child"]
    )
    child = Chain(
        "child",
        "subagent",
        "main",
        [req("c", 2, api=2, think=94)],
        anchor=0,
        join_before=1,
        subtree_end=4,
    )
    row = convert([parent, child])
    assert row["traces"][1]["events"][0]["seconds"] == 1
    assert row["traces"][0]["events"][2]["seconds"] == 6
    assert simulate(row)["main", 3, "start"] == 10
    broken = copy.deepcopy(row)
    broken["traces"][1]["events"][0]["seconds"] = 94
    assert simulate(broken)["main", 3, "start"] == 103


def test_start_and_completion_are_distinct_at_runtime():
    async def scenario():
        parent = ReplayTrace("p", [Generate(1, recorded_prompt_token_ids=[1])])
        child = ReplayTrace("c", [Generate(1, recorded_prompt_token_ids=[2])])
        task = ReplayTask(
            "t",
            [parent, child],
            [Dependency(EventRef("p", 0), EventRef("c", 0), "start")],
        )
        state = DependencyState(task)
        p = replay_events(parent, state)
        c = replay_events(child, state)
        pending = asyncio.create_task(anext(c))
        await asyncio.sleep(0)
        assert not pending.done()
        await anext(p)
        assert await asyncio.wait_for(pending, 1) is child.events[0]
        await p.aclose()
        await c.aclose()

    asyncio.run(scenario())


def test_true_start_dependency_deadlock_is_rejected():
    p = ReplayTrace("p", [Generate(1, recorded_prompt_token_ids=[1])])
    c = ReplayTrace("c", [Generate(1, recorded_prompt_token_ids=[2])])
    with pytest.raises(ValueError, match="cycle"):
        ReplayTask(
            "t",
            [p, c],
            [
                Dependency(EventRef("p", 0), EventRef("c", 0), "start"),
                Dependency(EventRef("c", 0), EventRef("p", 0)),
            ],
        )


def test_consecutive_zero_requests_keep_only_first_slot_wait():
    chain = Chain(
        "main",
        "main",
        None,
        [req("a", 0), req("z1", 3, out=0), req("z2", 8, out=0), req("b", 13)],
    )
    row = convert([chain])
    assert simulate(row)["main", 7, "start"] == 3
    assert len(row["metadata"]["cc_replay_policy"]["removed_requests"]) == 2


def test_converted_task_runs_real_client_without_sending_deleted_requests():
    from test_cc_replay import RecordingClient, _replay

    chain = Chain(
        "main",
        "main",
        None,
        [req("zero", 0, out=0), req("next", 3, out=5), req("later", 8)],
    )
    row = convert([chain])
    client = RecordingClient()
    result = _replay(client, ReplayTask.from_jsonl_row(row))
    assert result.completed_model_requests == 2
    assert len(client.calls) == 2
    assert [len(output) for _, output, _ in client.calls] == [5, 2]
    with pytest.raises(ValueError, match="warm_start=off"):
        _replay(
            RecordingClient(), ReplayTask.from_jsonl_row(row), warm_start="stationary"
        )


def test_validator_rejects_missing_payload_or_timing_tamper():
    chain = Chain(
        "main", "main", None, [req("a", 0), req("zero", 2, out=0), req("b", 6)]
    )
    session = Session("s", {"main": chain}, 3, 0)
    row = apply_policy(
        session,
        build_row(session, [10, 20, 30], "s", {}, {}),
        zero_output="drop-advance",
    )
    for change in ("payload", "wait", "receipt"):
        broken = copy.deepcopy(row)
        if change == "payload":
            broken["traces"][0]["events"][5]["max_tokens"] = 77
        elif change == "wait":
            broken["traces"][0]["events"][4]["seconds"] = 100
        else:
            broken["metadata"]["cc_replay_policy"]["removed_requests"] = []
        assert check_policy(session, broken)["errors"]


def test_regular_pool_excludes_config_only_special_ids(tmp_path):
    import json
    from strict_convert.adapters.cc_weka import allowed_token_ids

    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {"vocab": {"pad": 0, "a": 1, "b": 2, "eos": 3, "extra": 4}},
                "added_tokens": [],
            }
        )
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"vocab_size": 4, "pad_token_id": 0, "eos_token_id": [3]})
    )
    assert allowed_token_ids(str(tmp_path)).tolist() == [1, 2]
