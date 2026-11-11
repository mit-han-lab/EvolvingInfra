"""Steady-window score of a run over all Trie processes' lifecycle files (standard library only).

Same definition as the reference runs: the window opens `skip_s` after the first lifecycle event
of any process and lasts `window_s`; a request counts when it was admitted inside the window and is
not a stationary warm-start request; it may complete any time before Trie stops. Requests cancelled
when Trie stops are truncations, not failures. Prefix hit is token-weighted
(cached / prompt tokens of the completed window requests); TTFT is the client's first parsed token
minus its request start, over every completed window request.

  python3 score.py <run dir> --skip-s 600 --window-s 600
"""
from __future__ import annotations

import argparse
import glob
import json
import os

CANCELLED = frozenset(("CancelledError", "_BenchmarkAdmissionStopped"))


def _pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def window_score(directory: str, skip_s: float, window_s: float) -> dict:
    files = sorted(glob.glob(os.path.join(directory, "trie-lifecycle-*.jsonl")))
    firsts = []
    for path in files:
        with open(path) as f:
            line = f.readline()
        if line:
            firsts.append(json.loads(line)["ts_ns"])
    if not firsts:
        raise ValueError(f"no lifecycle events in {directory}")
    start = min(firsts) + int(skip_s * 1e9)
    end = start + int(window_s * 1e9)
    admits, ends, completes, failures, traces, migrated = {}, {}, {}, {}, 0, set()
    for i, path in enumerate(files):
        for line in open(path):
            e = json.loads(line)
            rid, kind, ts = (i, e["request"]["experiment_request_id"]), e["event"], e["ts_ns"]
            if kind == "CLIENT_ADMIT":
                admits[rid] = (ts, bool(e["attrs"].get("warmup")))
            elif kind == "CLIENT_COMPLETE":
                completes[rid], ends[rid] = e["attrs"], ts
            elif kind == "REQUEST_FAILED":
                failures[rid], ends[rid] = e["attrs"].get("error_class"), ts
            elif kind == "TRACE_COMPLETE" and start <= ts < end:
                traces += 1
            elif kind == "CLIENT_MIGRATE":
                migrated.add(rid)
    busy = sum(max(0, min(ends.get(r, end), end) - max(ts, start)) for r, (ts, _) in admits.items())
    window = [r for r, (ts, warm) in admits.items() if start <= ts < end and not warm]
    done = [completes[r] for r in window if r in completes]
    errors: dict = {}
    for r in window:
        if failures.get(r) is not None and failures[r] not in CANCELLED:
            errors[failures[r]] = errors.get(failures[r], 0) + 1
    span = (end - start) / 1e9
    prompt = sum(a["prompt_tokens"] for a in done)
    cached = sum(a["cached_prompt_tokens"] for a in done)
    completion = sum(a["completion_tokens"] for a in done)
    ttft, e2e = [], []
    for r in window:
        if r not in completes:
            continue
        timing = completes[r].get("client_timing") or {}
        s, f = timing.get("request_start_ts_ns"), timing.get("first_parsed_token_ts_ns")
        ttft.append((f - s) / 1e9 if s and f else None)
        e2e.append((ends[r] - s) / 1e9 if s else None)
    return {
        "files": len(files), "window_start_ns": start, "window_end_ns": end, "window_s": span,
        "replay_context_modes": sorted({a.get("replay_context_mode", "recorded") for a in done}),
        "completed_model_requests": len(done), "completed_traces": traces,
        "failed_requests": sum(errors.values()), "failures_by_class": errors,
        "cancelled_requests": sum(1 for r in window if failures.get(r) in CANCELLED),
        "migrated_requests": sum(1 for r in window if r in migrated),
        "warmup_requests": sum(1 for _, warm in admits.values() if warm),
        "requests_per_s": len(done) / span, "completion_tok_s": completion / span,
        "new_prompt_tok_s": (prompt - cached) / span, "cache_hit": cached / prompt if prompt else None,
        "mean_prompt_tokens": prompt / len(done) if done else None, "inflight_mean": busy / (end - start),
        "ttft_p50": _pct(ttft, 50), "ttft_p90": _pct(ttft, 90), "ttft_p95": _pct(ttft, 95),
        "ttft_p99": _pct(ttft, 99), "ttft_mean": _mean(ttft), "e2e_mean": _mean(e2e), "e2e_p95": _pct(e2e, 95),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory")
    parser.add_argument("--skip-s", type=float, required=True)
    parser.add_argument("--window-s", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(window_score(args.directory, args.skip_s, args.window_s), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
