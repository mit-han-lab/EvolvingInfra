"""Replay the EdgeBench capture as recorded, as schema-v3 tasks for warm-start replay.

Input is the GLM-5.2 schema-v2 capture, whose generates carry the prompt and
output token IDs the source server saw. No chat template is applied: every
prompt and output keeps its recorded tokens, mapped one-to-one from the source
vocabulary onto the target's regular vocabulary, so lengths are the recorded
lengths and two prompts share exactly their recorded common prefix. Only waits
and generates are kept. Each Claude Code session becomes one task: its main
lanes, sub-agents and nested sub-agents, with completion edges taken from the
recorded content and times.

- Spawn: the parent generate whose Agent call returned ``agentId: <id>``
  (or, when the parent recorded no launch, the last parent generate received
  before the child). Claude Code starts a child while the parent is still
  streaming, so a child that started before its spawning generate finished
  depends on the event before that generate and waits the recorded offset
  from the generate's start; otherwise it depends on the generate and waits
  the recorded gap after it.
- Join: the first parent turn whose context carries the child's
  ``<task-notification>``. The edge targets that turn's wait, which becomes
  the recorded gap after both the parent's previous generate and every child
  joined there. A child with no notification (the capture ended first) is
  joined only by the task's end.
- Lanes of one session without a parent start at their recorded offset.

Generates carry ``source_t_s`` / ``source_api_time_s`` relative to the task's
first request. All agents share one cache salt and routing key.

The report attests the output and the target tokenizer, so the harness input
check applies; it checks that the token map is one-to-one, that every
generate keeps its recorded counts, and that every edge agrees with the
recorded times.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strict_convert.adapters.cc_weka import allowed_token_ids  # noqa: E402
from trie.types import ReplayTask  # noqa: E402

TOLERANCE_S = 0.05
DEFAULT_SEED = "edgebench-recorded-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _messages(event: dict[str, Any]) -> list[dict[str, Any]]:
    if event["type"] == "replace_context":
        return event["messages"]
    if event["type"] == "append_message":
        return [event["message"]]
    return []


def index_agent(row: dict[str, Any], path: str, offset: int, used: set[int]) -> dict[str, Any]:
    """Everything the task plan needs from one agent row; adds its token IDs to ``used``."""
    meta = row["metadata"]
    times = {req["request_id"]: (req["received_at"], req["finished_at"]) for req in meta["requests"]}
    kinds, generates, launches, notices = [], [], {}, {}
    prompt_tokens = output_tokens = 0
    seen = 0
    for index, event in enumerate(row["events"]):
        kinds.append(event["type"])
        if event["type"] == "generate":
            received, finished = times[event["source_request_id"]]
            generates.append((index, received, finished))
            prompt, output = event["recorded_prompt_token_ids"], event["recorded_output_token_ids"]
            prompt_tokens += len(prompt)
            output_tokens += len(output)
            used.update(prompt)
            used.update(output)
            continue
        messages = _messages(event)
        new = messages[seen:] if event["type"] == "replace_context" and len(messages) >= seen else messages
        if event["type"] == "replace_context":
            seen = len(messages)
        for message in new:
            text = json.dumps(message.get("content"))
            for agent_id in re.findall(r"agentId: (\w+)", text) if message.get("role") == "tool" else ():
                launches.setdefault(agent_id, index)
            for agent_id in re.findall(r"<task-notification>\\n<task-id>(\w+)</task-id>", text):
                notices.setdefault(agent_id, index)
    return {
        "trace_id": row["trace_id"],
        "path": path,
        "offset": offset,
        "session_key": meta["session_key"],
        "parent_session_key": meta.get("parent_session_key"),
        "agent_id": meta.get("native_agent_id"),
        "role": meta["agent_role"],
        "kinds": kinds,
        "generates": generates,
        "launches": launches,
        "notices": notices,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
    }


def token_map(used: set[int], pool: np.ndarray, seed: str) -> np.ndarray:
    """One-to-one map of the used source IDs onto a seeded draw of the target pool."""
    if len(used) > len(pool):
        raise ValueError(f"{len(used)} distinct source tokens exceed the {len(pool)}-token target pool")
    rng = np.random.default_rng(int.from_bytes(hashlib.blake2b(seed.encode(), digest_size=8).digest(), "little"))
    source = np.array(sorted(used), dtype=np.int64)
    table = np.full(int(source.max()) + 1, -1, dtype=np.int64)
    table[source] = rng.permutation(pool)[: len(source)]
    return table


def _generate_before(agent: dict[str, Any], event_index: int) -> tuple[int, float, float] | None:
    earlier = [g for g in agent["generates"] if g[0] < event_index]
    return earlier[-1] if earlier else None


def plan_task(agents: list[dict[str, Any]], counters: dict[str, int]) -> dict[str, Any]:
    """Parents, leading waits, join-turn waits and edges, in source event indices."""
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for agent in agents:
        by_session[agent["session_key"]].append(agent)
    t0 = min(g[1] for agent in agents for g in agent["generates"])
    lead: dict[str, float] = {}
    spawn: dict[str, tuple[str, int]] = {}
    joins: dict[tuple[str, int], list[tuple[str, float]]] = defaultdict(list)
    for agent in agents:
        start = agent["generates"][0][1]
        parents = by_session.get(agent["parent_session_key"] or "", [])
        if not parents:
            lead[agent["trace_id"]] = start - t0
            counters["root_lanes"] += 1
            continue
        launched = [p for p in parents if agent["agent_id"] in p["launches"]]
        if launched:
            parent = launched[0]
            spawning = _generate_before(parent, parent["launches"][agent["agent_id"]])
            counters["spawn_by_launch"] += 1
        else:
            parent = max(parents, key=lambda p: len(p["generates"]))
            earlier = [g for g in parent["generates"] if g[1] <= start]
            spawning = earlier[-1] if earlier else None
            counters["spawn_by_time"] += 1
        pid = parent["trace_id"]
        if spawning is None:
            lead[agent["trace_id"]] = start - t0
            counters["spawn_at_task_start"] += 1
        elif start >= spawning[2] - TOLERANCE_S:
            spawn[agent["trace_id"]] = (pid, spawning[0])
            lead[agent["trace_id"]] = max(start - spawning[2], 0.0)
            counters["spawn_after_generate"] += 1
        else:
            if spawning[0] > 0:
                spawn[agent["trace_id"]] = (pid, spawning[0] - 1)
                lead[agent["trace_id"]] = max(start - spawning[1], 0.0)
            else:
                lead[agent["trace_id"]] = start - t0
            counters["spawn_while_streaming"] += 1
        notice = parent["notices"].get(agent["agent_id"])
        if notice is None:
            counters["join_task_end"] += 1
            continue
        previous = _generate_before(parent, notice)
        turn_first = previous[0] + 1
        if parent["kinds"][turn_first] != "wait":
            raise ValueError(f"{pid}: join turn at event {turn_first} does not start with a wait")
        joins[(pid, turn_first)].append((agent["trace_id"], agent["generates"][-1][2]))
        counters["join_notification"] += 1
    return {"t0": t0, "lead": lead, "spawn": spawn, "joins": joins}


def check_times(plan: dict[str, Any], agents: dict[str, dict[str, Any]]) -> list[str]:
    """Every edge: the source event ended before the target's first request was received."""
    errors = []
    for child, (parent, index) in plan["spawn"].items():
        parent_agent = agents[parent]
        ended = next((g[2] for g in parent_agent["generates"] if g[0] == index), None)
        if ended is None:  # an event before a generate ends when that generate starts
            ended = min(g[1] for g in parent_agent["generates"] if g[0] > index)
        if ended > agents[child]["generates"][0][1] + TOLERANCE_S:
            errors.append(f"{child}: spawn edge from {parent}@{index} ends after the child starts")
    for (parent, index), joined in plan["joins"].items():
        following = min(g[1] for g in agents[parent]["generates"] if g[0] > index)
        for child, end in joined:
            if end > following + TOLERANCE_S:
                errors.append(f"{child}: joins {parent}@{index} after that turn was received")
    return errors


def build_agent(
    row: dict[str, Any], agent: dict[str, Any], plan: dict[str, Any], table: np.ndarray, routing_key: str,
) -> tuple[dict[str, Any], dict[int, int]]:
    """The agent's waits and generates, and a map from source to output event index."""
    trace_id = agent["trace_id"]
    events: list[dict[str, Any]] = []
    index_map: dict[int, int] = {}
    if plan["lead"].get(trace_id, 0.0) > 0:
        events.append({"type": "wait", "seconds": plan["lead"][trace_id], "actor": "other"})
    generates = {g[0]: g for g in agent["generates"]}
    for index, event in enumerate(row["events"]):
        if event["type"] == "wait":
            seconds = event.get("seconds", 0.0)
            joined = plan["joins"].get((trace_id, index))
            if joined:
                previous = _generate_before(agent, index)
                following = min(g for g in agent["generates"] if g[0] > index)
                seconds = max(following[1] - max([previous[2]] + [end for _, end in joined]), 0.0)
            index_map[index] = len(events)
            events.append({"type": "wait", "seconds": seconds, "actor": event.get("actor", "other")})
        elif event["type"] == "generate":
            _, received, finished = generates[index]
            output = table[np.asarray(event["recorded_output_token_ids"], dtype=np.int64)].tolist()
            index_map[index] = len(events)
            events.append({
                "type": "generate",
                "max_tokens": len(output),
                "recorded_prompt_token_ids": table[np.asarray(event["recorded_prompt_token_ids"], dtype=np.int64)].tolist(),
                "recorded_output_token_ids": output,
                "source_request_id": event["source_request_id"],
                "source_model": event.get("source_model"),
                "source_t_s": received - plan["t0"],
                "source_api_time_s": finished - received,
            })
    metadata = {key: value for key, value in row["metadata"].items() if key not in ("requests", "strict_conversion")}
    metadata["routing_key"] = routing_key
    return {"schema_version": 2, "trace_id": trace_id, "events": events, "metadata": metadata}, index_map


def _edge_ref(agent: dict[str, Any], index_map: dict[int, int], index: int) -> int | None:
    """Output event whose completion means source event ``index`` completed."""
    if index in index_map:
        return index_map[index]
    following = index_map[min(g[0] for g in agent["generates"] if g[0] > index)]
    return following - 1 if following > 0 else None


def compose(sources: list[Path], tokenizer: Path, output: Path, seed: str) -> dict[str, Any]:
    index: list[dict[str, Any]] = []
    used: set[int] = set()
    for source in sources:
        offset = 0
        with open(source, "rb") as stream:
            for line in stream:
                if line.strip():
                    index.append(index_agent(json.loads(line), str(source), offset, used))
                    print(f"index: {len(index)} agents", file=sys.stderr, flush=True)
                offset += len(line)
    pool = allowed_token_ids(str(tokenizer))
    table = token_map(used, pool, seed)
    by_session = defaultdict(list)
    for agent in index:
        by_session[agent["session_key"]].append(agent)

    def root(agent: dict[str, Any]) -> str:
        while agent["parent_session_key"] in by_session:
            agent = by_session[agent["parent_session_key"]][0]
        return agent["session_key"]

    tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for agent in index:
        tasks[root(agent)].append(agent)
    counters: dict[str, int] = defaultdict(int)
    errors: list[str] = []
    totals = {"agents": 0, "generates": 0, "prompt_tokens": 0, "output_tokens": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as sink:
        temporary = Path(sink.name)
        try:
            for position, (session_key, agents) in enumerate(tasks.items(), 1):
                agents.sort(key=lambda a: (a["role"] != "main", a["generates"][0][1], a["trace_id"]))
                plan = plan_task(agents, counters)
                by_id = {a["trace_id"]: a for a in agents}
                errors.extend(check_times(plan, by_id))
                task_id = next(a for a in agents if a["session_key"] == session_key)["trace_id"].rsplit("/lane-", 1)[0]
                traces, maps = [], {}
                for agent in agents:
                    with open(agent["path"], "rb") as stream:
                        stream.seek(agent["offset"])
                        row = json.loads(stream.readline())
                    trace, maps[agent["trace_id"]] = build_agent(row, agent, plan, table, task_id)
                    traces.append(trace)
                    del row
                lengths = {t["trace_id"]: len(t["events"]) for t in traces}
                dependencies = []
                for child, (parent, source_index) in plan["spawn"].items():
                    before = _edge_ref(by_id[parent], maps[parent], source_index)
                    if before is not None:
                        dependencies.append({"before": {"trace_id": parent, "event_index": before},
                                             "after": {"trace_id": child, "event_index": 0}})
                for (parent, source_index), joined in plan["joins"].items():
                    for child, _ in joined:
                        dependencies.append({"before": {"trace_id": child, "event_index": lengths[child] - 1},
                                             "after": {"trace_id": parent, "event_index": maps[parent][source_index]}})
                row = {
                    "schema_version": 3,
                    "trace_id": task_id,
                    "cache_salt_scope": "task",
                    "start_offset_s": 0.0,
                    "traces": traces,
                    "dependencies": dependencies,
                    "metadata": {"strict_conversion": {"adapter": "compose_edgebench", "agents": len(agents)}},
                }
                task = ReplayTask.from_jsonl_row(row)  # schema, references and acyclicity
                for trace in task.traces:
                    for event in trace.events:
                        if getattr(event, "recorded_prompt_token_ids", None) is not None:
                            totals["generates"] += 1
                            totals["prompt_tokens"] += len(event.recorded_prompt_token_ids)
                            totals["output_tokens"] += len(event.recorded_output_token_ids)
                del task
                sink.write(json.dumps(row, separators=(",", ":")) + "\n")
                totals["agents"] += len(agents)
                del traces, row
                print(f"compose: {position}/{len(tasks)} tasks", file=sys.stderr, flush=True)
            sink.flush()
            os.chmod(temporary, 0o664)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    recorded = {"agents": len(index), "generates": sum(len(a["generates"]) for a in index),
                "prompt_tokens": sum(a["prompt_tokens"] for a in index),
                "output_tokens": sum(a["output_tokens"] for a in index)}
    if totals != recorded:
        errors.append(f"totals {totals} differ from the source {recorded}")
    report = {
        "workload": str(output),
        "sources": [str(s) for s in sources],
        "tasks": len(tasks),
        **totals,
        "distinct_source_tokens": len(used),
        "target_pool": len(pool),
        "counters": dict(sorted(counters.items())),
        "errors": errors,
        "verdict": "pass" if not errors else "fail",
        "attests": {
            "workload_sha256": _sha256_file(output),
            "tokenizer_sha256": _sha256_file(tokenizer / "tokenizer.json"),
            "source_sha256": {str(s): _sha256_file(s) for s in sources},
            "seed": seed,
        },
    }
    Path(f"{output}.validate.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, action="append", required=True,
                        help="GLM-5.2 schema-v2 capture wave (repeatable)")
    parser.add_argument("--tokenizer", type=Path, required=True, help="target tokenizer dir")
    parser.add_argument("--output", type=Path, required=True, help="schema-v3 workload; report at <output>.validate.json")
    parser.add_argument("--seed", default=DEFAULT_SEED, help="seed of the token map")
    args = parser.parse_args()
    report = compose(args.source, args.tokenizer, args.output, args.seed)
    print(json.dumps({k: v for k, v in report.items() if k != "errors"} | {"errors": report["errors"][:20]}, indent=2))
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
