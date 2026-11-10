import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from strict_convert.adapters.cc_stationary import normalize_clock
from strict_convert.adapters.cc_timing import simulate
from strict_convert.adapters.cc_weka import Chain, Session, build_row
from trie.client import _stationary_cut
from trie.types import ReplayTask


def request(rid, t, think, out=2):
    return dict(id=rid, t=t, think_time=think, api_time=1, out=out,
                hash_ids=[1], model="claude", **{"in": 64})


def test_random_cut_uses_cleaned_clock_and_preserves_payloads_and_dependencies():
    session = Session("root", {"main": Chain("main", "main", None,
        [request("a", 1000, 0), request("b", 5000, 120)], children=["child"]),
        "child": Chain("child", "subagent", "main", [request("c", 5001, 0)], anchor=1)}, 3, 0)
    original = build_row(session, list(range(10, 100)), "test", {}, {})
    frozen = copy.deepcopy(original)
    row = normalize_clock(session, original)
    assert original == frozen and row["dependencies"] == original["dependencies"]
    times = simulate(row)
    for before, after in zip(original["traces"], row["traces"]):
        for i, (old, new) in enumerate(zip(before["events"], after["events"])):
            if new["type"] == "generate":
                assert new["source_t_s"] == times[after["trace_id"], i, "start"]
                assert {k: v for k, v in new.items() if k != "source_t_s"} == {
                    k: v for k, v in old.items() if k != "source_t_s"}
    task = ReplayTask.from_jsonl_row(row)
    assert _stationary_cut(task, 0.5) == pytest.approx(61.5)
    assert row["metadata"]["cc_stationary_timeline"]["original_times"][1]["source_t_s"] == 5000


@pytest.mark.parametrize("think,out", [(121, 2), (0, 0)])
def test_unclean_input_rejected_before_stationary_conversion(think, out):
    session = Session("root", {"main": Chain("main", "main", None,
                      [request("a", 0, think, out)])}, 1, 0)
    row = build_row(session, list(range(10, 100)), "test", {}, {})
    with pytest.raises(ValueError):
        normalize_clock(session, row)
