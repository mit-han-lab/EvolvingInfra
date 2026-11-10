"""Codex sessions: ``sequential_agents`` cc-weka rows (one chain per agent) and the Codex converter."""

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from strict_convert.adapters.cc_weka import build_row, build_session, check_session, row_blocks
from trie.types import ReplayTask


def req(t, hashes, api=1.0, think=0.0):
    return {"t": t, "model": "gpt-5", "in": 64 * len(hashes), "out": 3, "hash_ids": hashes,
            "api_time": api, "think_time": think, "type": "n"}


def row(requests, flag="sequential_agents"):
    out = {"id": "codex-s", "models": ["gpt-5"], "block_size": 64, "hash_id_scope": "local", "requests": requests}
    if flag:
        out[flag] = True
    return out


def task(source):
    session = build_session(source)
    converted = build_row(session, list(range(20, 100)), "test", {}, {})
    assert not check_session(session, converted, row_blocks(converted))["errors"]
    ReplayTask.from_jsonl_row(converted)
    return session, converted


# Turn 2 drops the tail of turn 1 (dropped reasoning / compaction): LCP detection forks it.
SHRINK = [req(0.0, [0, 1, 2, 3]), req(2.0, [0, 1, 2, 3, 4, 5], think=1.0), req(4.0, [0, 1, 6], think=1.0),
          req(6.0, [0, 1, 6, 7], think=1.0)]


@pytest.mark.parametrize("flag", ["sequential_agents", "single_agent"])
def test_sequential_session_is_one_main_chain_without_edges(flag):
    session, converted = task(row(SHRINK, flag=flag))
    assert list(session.chains) == ["main"]
    assert [r["id"] for r in session.chains["main"].requests] == ["r0", "r1", "r2", "r3"]
    assert [t["trace_id"] for t in converted["traces"]] == ["main"]
    assert converted["dependencies"] == []
    assert converted["cache_salt_scope"] == "task"
    assert converted["traces"][0]["metadata"]["routing_key"] == "codex-s"


def test_without_flag_the_shrink_still_goes_through_chain_detection():
    assert build_session(row(SHRINK, flag=None)).chains  # unchanged CC path; may fork


def test_sequential_rejects_overlap():
    with pytest.raises(ValueError, match="starts before"):
        build_session(row([req(0.0, [0], api=5.0), req(2.0, [0, 1])]))


def sub(agent_id, requests, parent=None):
    entry = {"type": "subagent", "agent_id": agent_id, "requests": requests}
    if parent:
        entry["parent_agent_id"] = parent
    return entry


def test_subagents_blocking_background_nested_one_salt_and_key():
    # main: r0 [0,1] spawns a (blocking: main's next turn starts after a ends) and b (background,
    # still running when main resumes, joined before main's last turn); b spawns c (nested).
    source = row([
        req(0.0, [0, 1]),
        sub("a", [req(2.0, [10], api=1.0), req(4.0, [10, 11], api=1.0, think=1.0)]),
        sub("b", [req(2.5, [20], api=8.0)]),
        sub("c", [req(11.0, [30], api=1.0)], parent="b"),
        req(6.0, [0, 1, 2], think=1.0),
        req(20.0, [0, 1, 2, 3], think=8.0),
    ])
    session, converted = task(source)
    chains = session.chains
    assert set(chains) == {"main", "sa:a", "sa:b", "sa:c"}
    assert chains["sa:a"].parent == chains["sa:b"].parent == "main" and chains["sa:c"].parent == "sa:b"
    assert chains["sa:a"].join_mode == "blocking" and chains["sa:a"].join_before == 1
    assert chains["sa:b"].join_mode == "background" and chains["sa:b"].join_before == 2
    assert chains["sa:c"].join_mode == "end"  # b has no later request; c joins b's end, which joins main
    assert converted["cache_salt_scope"] == "task"  # as for CC: one salt and one routing key per session
    assert {t["metadata"]["routing_key"] for t in converted["traces"]} == {"codex-s"}
    assert converted["dependencies"]


def test_unknown_parent_falls_back_to_main_and_cycle_free():
    session, _ = task(row([req(0.0, [0]), sub("a", [req(2.0, [10])], parent="missing")]))
    assert session.chains["sa:a"].parent == "main"


def _event(kind, seconds):
    m, s = divmod(seconds, 60)
    return {"event_type": kind, "timestamp": f"2026-01-01T00:{int(m):02d}:{s:06.3f}Z"}


def _round(sid, i, start, end, n_in, prefix, tools=()):
    return {"session_id": sid, "round_index": i, "user": "u1", "model": "gpt-5", "input_tokens_total": n_in,
            "prefix_tokens": prefix, "output_tokens": 5, "tools": list(tools),
            "timing_events": [_event("user_message", start), _event("usage_report", end)]}


def _convert(tmp_path, rounds):
    src, out = tmp_path / "in.jsonl.gz", tmp_path / "out.jsonl.gz"
    with gzip.open(src, "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rounds)
    report = subprocess.run([sys.executable, str(SCRIPTS / "codex_to_ccweka.py"), str(src), str(out)], check=True,
                            capture_output=True, text=True).stdout
    return [json.loads(line) for line in gzip.open(out, "rt")], json.loads(report)


def test_converter_serializes_overlap_and_clips_idle(tmp_path):
    # round 1 starts 0.5 s before round 0 ends; round 2 follows 200 s of idle (clipped to 120 s).
    rows, report = _convert(tmp_path, [
        _round("codex:x", 0, 1.0, 3.0, 640, 0), _round("codex:x", 1, 2.5, 4.0, 704, 640),
        _round("codex:x", 2, 204.0, 205.0, 128, 0)])
    (session_row,) = rows
    assert session_row["sequential_agents"] is True and "agent_scope" not in session_row
    reqs = session_row["requests"]
    assert [r["hash_ids"][:10] for r in reqs][:2] == [list(range(10)), list(range(10))]
    assert reqs[1]["t"] == pytest.approx(2.0) and reqs[1]["api_time"] == pytest.approx(1.0)  # started at round 0's end
    assert reqs[2]["think_time"] == pytest.approx(120.0) and reqs[2]["t"] == pytest.approx(123.0)
    assert reqs[2]["hash_ids"] == [11, 12]  # prefix 0: all new blocks
    assert report["overlap_serialized"] == 1 and report["idle_clipped"] == 1
    assert list(build_session(session_row).chains) == ["main"]


def test_converter_links_spawned_subagent(tmp_path):
    spawn = {"tool_name": "spawn_agent", "emitted_at": "2026-01-01T00:00:03.100Z",
             "result_at": "2026-01-01T00:00:03.200Z"}
    rows, report = _convert(tmp_path, [
        _round("codex:p", 0, 1.0, 3.0, 640, 0, tools=[spawn]),
        # the parent waits ~400 s for the child; the family is never idle, so nothing is clipped
        _round("codex:p", 1, 410.0, 411.0, 704, 640),
        _round("codex:c", 0, 5.0, 205.0, 256, 0),
        _round("codex:c", 1, 300.0, 400.0, 320, 256),
        _round("codex:other", 0, 500.0, 501.0, 64, 0),  # same user, outside the link window
    ])
    assert report["spawns"] == 1 and report["spawns_linked_unique"] == 1 and report["subagents"] == 1
    by_id = {r["id"]: r for r in rows}
    assert set(by_id) == {"codex-p", "codex-other"}
    parent_row = by_id["codex-p"]
    entries = parent_row["requests"]
    child = next(e for e in entries if e["type"] == "subagent")
    assert child["agent_id"] == "codex-c" and "parent_agent_id" not in child
    assert child["requests"][0]["hash_ids"] == [11, 12, 13, 14]  # no blocks shared with the parent
    top = [e for e in entries if e["type"] == "n"]
    assert top[1]["t"] == pytest.approx(409.0)  # 400 s child gap kept: the family was never idle
    assert top[1]["think_time"] == pytest.approx(10.0)  # measured from the child's end (t=399)
    session, converted = task(parent_row)
    assert session.chains["sa:codex-c"].join_mode == "blocking"
    assert converted["cache_salt_scope"] == "task"

def test_converter_caps_think_per_request(tmp_path):
    # The parent idles 300 s while a background sub-agent keeps the session busy: the session is never idle,
    # so nothing is clipped from the timeline, but the parent's think time is capped at 120 s.
    spawn = {"tool_name": "spawn_agent", "emitted_at": "2026-01-01T00:00:03.100Z",
             "result_at": "2026-01-01T00:00:03.200Z"}
    rows, report = _convert(tmp_path, [
        _round("codex:p", 0, 1.0, 3.0, 640, 0, tools=[spawn]),
        _round("codex:p", 1, 303.0, 304.0, 704, 640),
        _round("codex:c", 0, 5.0, 400.0, 256, 0),
    ])
    (row,) = rows
    top = [e for e in row["requests"] if e["type"] == "n"]
    assert top[1]["t"] == pytest.approx(302.0)  # timeline kept: the session was never idle
    assert top[1]["think_time"] == pytest.approx(120.0)  # capped from 300 s
    assert report["think_capped"] == 1 and report["idle_clipped"] == 0
    assert all(q["think_time"] <= 120.0 for e in row["requests"]
               for q in (e["requests"] if e["type"] == "subagent" else [e]))
    task(row)
