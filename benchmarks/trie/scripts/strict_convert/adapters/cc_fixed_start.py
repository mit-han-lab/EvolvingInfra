"""A fixed suffix of the capped-think-time CC event DAG.

Recorded API durations locate the cut, not predict target service time. After
cutting, every retained Generate executes on the target with its full prompt.
Only an already-started source Generate becomes a residual control Wait.
"""

from copy import deepcopy
import math

from strict_convert.adapters.cc_timing import simulate
from strict_convert.adapters.cc_weka import agent_layouts, dependencies

POLICY = "recorded-think-fixed-cut-v1"


def timing_row(session):
    """Build the timing-only DAG independently of token conversion."""

    def event(chain, i):
        req = chain.requests[i]
        if req["out"] <= 0:
            raise ValueError(
                "Fixed-cut input must already have positive output lengths"
            )
        for key in ("think_time", "api_time"):
            if not math.isfinite(req[key]) or req[key] < 0:
                raise ValueError(f"Invalid {key}")
        if req["think_time"] > 120:
            raise ValueError("Expected the cleaned think-time cap of 120 seconds")
        return {"type": "generate", "source_api_time_s": req["api_time"]}

    layouts = agent_layouts(session, event)
    edges = dependencies(session, layouts)
    return {
        "traces": [
            {"trace_id": cid, "events": layout.events}
            for cid, layout in layouts.items()
        ],
        "dependencies": [
            {
                "before": {"trace_id": a[0], "event_index": a[1]},
                "after": {"trace_id": b[0], "event_index": b[1]},
            }
            for a, b in edges
        ],
    }


def eligible(session, cut_s):
    row = timing_row(session)
    times = simulate(row)
    return any(
        e["type"] == "generate" and times[t["trace_id"], i, "start"] >= cut_s
        for t in row["traces"]
        for i, e in enumerate(t["events"])
    )


def cut_row(row, cut_s):
    if not math.isfinite(cut_s) or cut_s < 0:
        raise ValueError("cut_s must be finite and nonnegative")
    times = simulate(row)
    result = deepcopy(row)
    retained = replaced = crossing = 0
    for trace in result["traces"]:
        cid = trace["trace_id"]
        for i, event in enumerate(trace["events"]):
            start, end = times[cid, i, "start"], times[cid, i, "complete"]
            if event["type"] == "generate" and start >= cut_s:
                retained += 1
                continue
            if event["type"] == "generate":
                replaced += 1
                crossing += end > cut_s
            trace["events"][i] = {
                "type": "wait",
                "seconds": max(0.0, end - max(start, cut_s)),
                "actor": "other",
            }
        if not any(e["type"] == "generate" for e in trace["events"]):
            trace.setdefault("metadata", {})["control_only"] = True
    if not retained:
        return None
    # Every event phase must be exactly the original phase shifted and clipped.
    after = simulate(result)
    error = max(abs(after[k] - max(0.0, t - cut_s)) for k, t in times.items())
    if error > 1e-6:
        raise ValueError(f"Fixed cut changed event dependencies: {error}s")
    result.setdefault("metadata", {})["cc_fixed_start"] = {
        "policy": POLICY,
        "cut_s": cut_s,
        "retained_requests": retained,
        "replaced_requests": replaced,
        "crossing_request_controls": crossing,
        "source_requests": retained + replaced,
        "phase_error_s": error,
        "clock": "capped think_time plus recorded api_time on inferred completion DAG",
        "cache_state": "cold; first retained request sends its full recorded prompt",
        "repeated_instances": "same suffix, new replay_cycle_id token content",
        "source_t_is_observational": True,
    }
    return result
