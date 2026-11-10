"""Use the cleaned CC event clock with the existing stationary replay client."""

from copy import deepcopy

from strict_convert.adapters.cc_fixed_start import timing_row
from strict_convert.adapters.cc_timing import simulate

POLICY = "cleaned-dag-stationary-v1"


def normalize_clock(session, row):
    """Retain every request, payload and dependency; replace observation times.

    Raw source timestamps still contain waits removed by clean-cap120. The
    stationary client must draw and compute residual waits on the same clock
    as the executable event DAG. Original timestamps remain in metadata.
    """
    times = simulate(timing_row(session))
    if simulate(row) != times:
        raise ValueError("Payload conversion changed the cleaned timing DAG")
    result = deepcopy(row)
    original = []
    for trace in result["traces"]:
        for index, event in enumerate(trace["events"]):
            if event["type"] != "generate":
                continue
            original.append(dict(agent=trace["trace_id"], request=event["source_request_id"],
                                 source_t_s=event["source_t_s"]))
            event["source_t_s"] = times[trace["trace_id"], index, "start"]
    if len(original) != session.source_requests:
        raise ValueError("Incomplete source request coverage")
    result.setdefault("metadata", {})["cc_stationary_timeline"] = dict(
        policy=POLICY, clock="capped think_time plus recorded API duration on inferred DAG",
        requests=len(original), original_times=original,
        initialization="Unchanged Trie warm_start=stationary; first-slot random cut and one-token preheat")
    return result
