"""Totals over the requests a Trie lifecycle file admitted in a time window.

A continuous run keeps warmup and steady in one Trie process; the steady
window is cut afterwards by admission time. Warmup requests of a stationary
warm start are excluded, and a request cancelled when Trie stopped is a
truncation, not a failure. In-flight is the time-weighted mean of admitted,
unfinished requests inside the window.

    python scripts/lifecycle_window.py <lifecycle.jsonl> [--skip-seconds 3600] [--window-seconds 3600]

The window starts --skip-seconds after the first lifecycle event (default one
hour of warmup), or at --start-ns, and lasts --window-seconds (default one
hour; 0 runs to the last event).
"""

from __future__ import annotations

import argparse
import json


def lifecycle_window(path: str, start_ns: int, end_ns: int) -> dict:
    admits, ends, completes, failures = {}, {}, {}, {}
    traces = 0
    with open(path) as handle:
        for line in handle:
            event = json.loads(line)
            rid, kind, ts = event["request"]["experiment_request_id"], event["event"], event["ts_ns"]
            if kind == "CLIENT_ADMIT":
                admits[rid] = (ts, bool(event["attrs"].get("warmup")))
            elif kind == "CLIENT_COMPLETE":
                completes[rid], ends[rid] = event["attrs"], ts
            elif kind == "REQUEST_FAILED":
                failures[rid], ends[rid] = event["attrs"].get("error_class"), ts
            elif kind == "TRACE_COMPLETE" and start_ns <= ts < end_ns:
                traces += 1
    busy_ns = sum(max(0, min(ends.get(rid, end_ns), end_ns) - max(ts, start_ns)) for rid, (ts, _) in admits.items())
    window = [rid for rid, (ts, warmup) in admits.items() if start_ns <= ts < end_ns and not warmup]
    done = [completes[rid] for rid in window if rid in completes]
    errors: dict[str, int] = {}
    for rid in window:
        if failures.get(rid) not in (None, "CancelledError"):
            errors[failures[rid]] = errors.get(failures[rid], 0) + 1
    span = (end_ns - start_ns) / 1e9
    prompt = sum(attrs["prompt_tokens"] for attrs in done)
    cached = sum(attrs["cached_prompt_tokens"] for attrs in done)
    completion = sum(attrs["completion_tokens"] for attrs in done)
    return {
        "window_s": span,
        "completed_model_requests": len(done),
        "completed_traces": traces,
        "failed_requests": sum(errors.values()),
        "failures_by_class": errors,
        "cancelled_requests": sum(1 for rid in window if failures.get(rid) == "CancelledError"),
        "completed_prompt_tokens": prompt,
        "cached_prompt_tokens": cached,
        "completed_completion_tokens": completion,
        "cache_hit": cached / prompt if prompt else None,
        "requests_per_s": len(done) / span if span else None,
        "completion_tok_s": completion / span if span else None,
        "new_prompt_tok_s": (prompt - cached) / span if span else None,
        "inflight_mean": busy_ns / (end_ns - start_ns) if end_ns > start_ns else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lifecycle")
    parser.add_argument("--start-ns", type=int)
    parser.add_argument("--skip-seconds", type=float, default=3600.0, help="warmup length after the first event")
    parser.add_argument("--window-seconds", type=float, default=3600.0, help="steady length; 0 runs to the last event")
    args = parser.parse_args()
    with open(args.lifecycle) as handle:
        stamps = [json.loads(line)["ts_ns"] for line in handle]
    start = args.start_ns if args.start_ns is not None else min(stamps) + int(args.skip_seconds * 1e9)
    end = start + int(args.window_seconds * 1e9) if args.window_seconds else max(stamps)
    print(json.dumps({"start_ns": start, "end_ns": end, **lifecycle_window(args.lifecycle, start, end)}, indent=2))


if __name__ == "__main__":
    main()
