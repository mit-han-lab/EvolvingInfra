import asyncio
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from strict_convert.adapters.cc_fixed_start import cut_row, eligible, timing_row
from strict_convert.adapters.cc_timing import simulate
from strict_convert.adapters.cc_weka import Chain, Session, build_row
from trie.blocks import BlockTokens
from trie.client import Client
from trie.dependencies import DependencyState, replay_events
from trie.types import Generate, ReplayTask, Wait


def req(rid, *, api=1, think=0, t=0, out=2):
    return dict(
        id=rid,
        t=t,
        api_time=api,
        think_time=think,
        out=out,
        hash_ids=[1],
        **{"in": 64, "model": "claude"}
    )


def source():
    parent = Chain(
        "main",
        "main",
        None,
        [req("p", api=30, think=3590), req("p2", t=4000)],
        children=["child"],
    )
    child = Chain("child", "subagent", "main", [req("c", t=3700)], anchor=0)
    session = Session("test", {"main": parent, "child": child}, 3, 0)
    return build_row(session, list(range(10, 100)), "test", {}, {})


def test_crossing_generate_keeps_child_gate_and_full_future_payload():
    before = source()
    frozen = copy.deepcopy(before)
    after = cut_row(before, 3600)
    assert before == frozen
    ReplayTask.from_jsonl_row(after)
    t = simulate(after)
    original = simulate(before)
    assert all(t[k] == pytest.approx(max(0, v - 3600)) for k, v in original.items())
    assert t["child", 0, "start"] == 20
    assert after["dependencies"] == before["dependencies"]
    assert after["traces"][0]["events"][-1] == before["traces"][0]["events"][-1]
    assert after["metadata"]["cc_fixed_start"]["crossing_request_controls"] == 1


def test_boundary_start_kept_end_released_and_wait_clipped():
    row = source()
    for cut in (3590, 3595, 3620):
        after = cut_row(row, cut)
        expected = simulate(row)
        assert all(
            v == pytest.approx(max(0, expected[k] - cut))
            for k, v in simulate(after).items()
        )
    assert cut_row(row, 3590)["traces"][0]["events"][1]["type"] == "generate"
    assert cut_row(row, 3620)["traces"][0]["events"][1]["seconds"] == 0
    assert cut_row(row, 4000) is None
    for cut in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            cut_row(row, cut)


def test_control_only_agent_still_waits_before_join():
    row = source()
    row["dependencies"] = [
        dict(
            before=dict(trace_id="main", event_index=1),
            after=dict(trace_id="child", event_index=0),
        )
    ]
    # Main no longer has any future Generate, but child depends on it finishing.
    row["traces"][0]["events"] = row["traces"][0]["events"][:2]
    after = cut_row(row, 3600)
    task = ReplayTask.from_jsonl_row(after)
    assert task.traces[0].metadata["control_only"] is True
    assert simulate(after)["child", 0, "start"] == 20

    async def run():
        state = DependencyState(task)
        seen = []
        gate = asyncio.Event()

        async def worker(trace):
            async for e in replay_events(trace, state):
                if trace.trace_id == "main" and isinstance(e, Wait) and e.seconds:
                    await gate.wait()
                if isinstance(e, Generate):
                    seen.append(trace.trace_id)

        tasks = [asyncio.create_task(worker(t)) for t in task.traces]
        await asyncio.sleep(0)
        assert not seen
        gate.set()
        await asyncio.gather(*tasks)
        assert seen == ["child"]

    asyncio.run(run())


def test_cap_source_validation_and_eligibility():
    session = Session(
        "x",
        {"main": Chain("main", "main", None, [req("a", api=3601), req("b", t=5000)])},
        2,
        0,
    )
    assert eligible(session, 3600)
    assert not eligible(session, 5000)
    session.chains["main"].requests[0]["think_time"] = 121
    with pytest.raises(ValueError, match="cap"):
        timing_row(session)
    session.chains["main"].requests[0]["think_time"] = 0
    session.chains["main"].requests[0]["out"] = 0
    with pytest.raises(ValueError, match="positive"):
        timing_row(session)


def test_fixed_suffix_same_every_cycle_new_token_content_and_no_second_cut():
    row = cut_row(source(), 3600)
    task = ReplayTask.from_jsonl_row(row)
    spec = task.traces[0].block_spec
    assert BlockTokens(spec, 0).tokens([[1, 1]]) != BlockTokens(spec, 1).tokens(
        [[1, 1]]
    )

    async def run():
        # Guard occurs before Backend/model access, even on later admissions.
        with pytest.raises(ValueError, match="already cut"):
            await Client._run_replay_task(
                object(),
                task,
                benchmark_start=0,
                result=None,
                refresh=lambda: None,
                delay_scale=1,
                replay_cycle_id=217,
                warm_start_draw=0.5,
            )

    asyncio.run(run())
