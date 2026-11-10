"""Opt-in causal timing and zero-output contraction for the CC adapter.

Chain partition and inferred parent/join choices stay in cc_weka. This module
turns those choices into executable waits, checks the recorded-time replay,
and removes model calls without deleting their dependency milestones.
"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from typing import Any

from strict_convert.adapters.cc_weka import AgentLayout, completion_refs, request_end

POLICY = "causal-gaps-v1"


def simulate(row: dict[str, Any]) -> dict[tuple[str, int, str], float]:
    """Execute the event DAG in virtual time using recorded API durations."""
    events = {
        (trace["trace_id"], i): event
        for trace in row["traces"]
        for i, event in enumerate(trace["events"])
    }
    nodes = {(*key, phase) for key in events for phase in ("start", "complete")}
    successors = defaultdict(dict)
    degree = dict.fromkeys(nodes, 0)
    times = dict.fromkeys(nodes, 0.0)

    def edge(a, b, lag=0.0):
        if a not in nodes or b not in nodes:
            raise ValueError(f"Unknown timing dependency: {a} -> {b}")
        if b not in successors[a]:
            degree[b] += 1
        successors[a][b] = lag

    for (trace, index), event in events.items():
        duration = (
            event["seconds"] if event["type"] == "wait" else event["source_api_time_s"]
        )
        edge((trace, index, "start"), (trace, index, "complete"), duration)
        if index:
            edge((trace, index - 1, "complete"), (trace, index, "start"))
    for dep in row["dependencies"]:
        a, b = dep["before"], dep["after"]
        edge(
            (a["trace_id"], a["event_index"], dep.get("when", "complete")),
            (b["trace_id"], b["event_index"], "start"),
        )
    ready = deque(node for node, n in degree.items() if not n)
    visited = 0
    while ready:
        node = ready.popleft()
        visited += 1
        for following, lag in successors[node].items():
            times[following] = max(times[following], times[node] + lag)
            degree[following] -= 1
            if not degree[following]:
                ready.append(following)
    if visited != len(nodes):
        raise ValueError("CC reconstruction contains a dependency cycle")
    return times


def apply_policy(session, row, *, zero_output="raise-to-one"):
    """Preserve payloads; rebuild waits and fork milestones from source times.

    Start is the client event start, not GPU admission. A zero-output request
    becomes a zero-duration control milestone. The next same-chain turn loses
    its own wait, so it takes the deleted request's slot subject to its existing
    joins. This avoids contracting a fork and its join into a dependency cycle.
    """
    if zero_output not in ("raise-to-one", "drop-advance"):
        raise ValueError("Unknown CC zero-output policy")
    row = deepcopy(row)
    chains = session.chains
    agents = {agent["trace_id"]: agent for agent in row["traces"]}
    layouts = {}
    origin = min(req["t"] for chain in chains.values() for req in chain.requests)
    for cid, chain in chains.items():
        payloads = {
            e["source_request_id"]: e
            for e in agents[cid]["events"]
            if e["type"] == "generate"
        }
        events = []
        for req in chain.requests:
            events.extend(
                [
                    {"type": "wait", "seconds": 0.0, "actor": "other"},
                    payloads[req["id"]],
                ]
            )
        agents[cid]["events"] = events
        layouts[cid] = AgentLayout(
            events, list(range(0, len(events), 2)), list(range(1, len(events), 2))
        )

    edges = set()
    for cid, chain in chains.items():
        if chain.parent is None:
            continue
        parent = chains[chain.parent]
        started = [
            i for i, req in enumerate(parent.requests) if req["t"] <= chain.start
        ]
        if started:
            index = started[-1]
            when = (
                "complete"
                if request_end(parent.requests[index]) <= chain.start + 1e-6
                else "start"
            )
            edges.add((parent.chain_id, 2 * index + 1, cid, 0, when))
        if chain.join_before is not None:
            for from_id, from_index in completion_refs(session, layouts, chain):
                edges.add(
                    (
                        from_id,
                        from_index,
                        chain.parent,
                        2 * chain.join_before,
                        "complete",
                    )
                )
    row["dependencies"] = [
        {
            "before": {"trace_id": a, "event_index": i},
            "after": {"trace_id": b, "event_index": j},
            **({"when": when} if when != "complete" else {}),
        }
        for a, i, b, j, when in sorted(edges)
    ]
    incoming = defaultdict(list)
    for a, i, b, j, when in edges:
        req = chains[a].requests[i // 2]
        incoming[b, j].append(req["t"] if when == "start" else request_end(req))
    for cid, chain in chains.items():
        for i, req in enumerate(chain.requests):
            prior = request_end(chain.requests[i - 1]) if i else origin
            gate = max([prior, *incoming[cid, 2 * i]])
            layouts[cid].events[2 * i]["seconds"] = max(0.0, req["t"] - gate)

    # Independent longest-path simulation catches waits counted twice and
    # forks dispatched from an old completed request instead of the live one.
    times = simulate(row)
    errors = [
        abs(times[cid, 2 * i + 1, "start"] - (req["t"] - origin))
        for cid, chain in chains.items()
        for i, req in enumerate(chain.requests)
    ]
    max_error = max(errors, default=0.0)
    if max_error > 0.005:
        raise ValueError(
            f"{session.session_id}: reconstructed timeline error {max_error:.6f}s"
        )

    removed = []
    for cid, chain in chains.items():
        events = layouts[cid].events
        pending = False
        for i, req in enumerate(chain.requests):
            if zero_output == "drop-advance" and req["out"] == 0:
                successor = next(
                    (r for r in chain.requests[i + 1 :] if r["out"] > 0), None
                )
                if pending or successor is None:
                    events[2 * i]["seconds"] = 0.0
                removed.append(
                    {
                        "chain_id": cid,
                        "removed_source_request_id": req["id"],
                        "source_t_s": req["t"],
                        "source_api_time_s": req["api_time"],
                        "successor_source_request_id": (
                            successor["id"] if successor else None
                        ),
                    }
                )
                # Keep references valid, but issue no HTTP request at this node.
                events[2 * i + 1] = {"type": "wait", "seconds": 0.0, "actor": "other"}
                pending = True
            else:
                if pending:
                    events[2 * i]["seconds"] = 0.0
                pending = False
        if not any(e["type"] == "generate" for e in events):
            agents[cid]["metadata"]["control_only"] = True
    effective = simulate(row)
    slots = [
        {
            "chain_id": cid,
            "source_request_id": req["id"],
            "source_t_s": req["t"],
            "source_think_time_s": req["think_time"],
            "effective_recorded_start_s": effective[cid, 2 * i + 1, "start"],
            "event_index": 2 * i + 1,
        }
        for cid, chain in chains.items()
        for i, req in enumerate(chain.requests)
        if not (zero_output == "drop-advance" and req["out"] == 0)
    ]
    row["metadata"]["cc_replay_policy"] = {
        "timing": POLICY,
        "zero_output": zero_output,
        "source_time_origin_s": origin,
        "pre_deletion_timeline_max_error_s": max_error,
        "source_requests": session.source_requests,
        "retained_requests": len(slots),
        "removed_requests": removed,
        "request_slots": slots,
        "dependencies_are_inferred": True,
    }
    return row


def check_policy(session, row):
    """Check exact source coverage, unchanged surviving payloads and timing DAG."""
    from strict_convert.adapters.cc_weka import build_row, row_blocks

    errors = []
    policy = row["metadata"]["cc_replay_policy"]
    if policy.get("timing") != POLICY:
        return {"session": session.session_id, "errors": ["Unknown CC timing policy"]}
    summary = row["metadata"]["strict_conversion"]
    base = build_row(
        session,
        row["metadata"]["block_tokens"]["pool"],
        summary["seed"],
        summary["target_tokenizer"],
        {},
    )
    expected = apply_policy(session, base, zero_output=policy["zero_output"])
    for field in ("traces", "dependencies"):
        if row[field] != expected[field]:
            errors.append(f"{field} differ from the declared CC conversion")
    if policy != expected["metadata"]["cc_replay_policy"]:
        errors.append("CC deletion/timing receipt differs from source reconstruction")
    raw = {
        req["id"]: req for chain in session.chains.values() for req in chain.requests
    }
    wanted = {
        rid
        for rid, req in raw.items()
        if not (policy["zero_output"] == "drop-advance" and req["out"] == 0)
    }
    generates = [
        e for trace in row["traces"] for e in trace["events"] if e["type"] == "generate"
    ]
    actual = [e["source_request_id"] for e in generates]
    if set(actual) != wanted or len(actual) != len(wanted):
        errors.append("Surviving source requests must occur exactly once")
    blocks = row_blocks(row)
    for event in generates:
        req = raw.get(event["source_request_id"])
        if req is None:
            continue
        if (
            blocks[req["id"]] != req["hash_ids"]
            or len(blocks[req["id"]]) * 64 != req["in"]
        ):
            errors.append(f"{req['id']}: source input changed")
        if event["max_tokens"] != max(1, req["out"]):
            errors.append(f"{req['id']}: source output length changed")
        if event["source_t_s"] != req["t"] or event["source_model"] != req["model"]:
            errors.append(f"{req['id']}: source identity/timestamp changed")
    simulate(row)  # reject broken references and cycles even if the receipt is forged
    return {
        "session": session.session_id,
        "timing_policy": POLICY,
        "requests": len(actual),
        "source_requests": len(raw),
        "removed_zero_output": len(raw) - len(actual),
        "pre_deletion_timeline_max_error_s": policy[
            "pre_deletion_timeline_max_error_s"
        ],
        "errors": errors,
    }
