#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""The runtime checks that decide whether a replay run is valid.

Before load every worker drains and stays quiet (wait_for_quiet); after it the steady window
must have completions and zero failures (check_result, or check_window over a continuous run's
lifecycle) and no worker's request counter may go backwards (check_metrics over the snapshots
`scrape` writes). The input check is check_inputs.py; conformance is Trie's
scripts/sglang_strict_replay_conformance.py (capture_conformance.py keeps its native replies).

Command line (each subcommand exits non-zero when its check fails):
  checks.py quiet   --worker URL [--worker URL ...]          wait until all workers are drained and idle
  checks.py scrape  --worker URL ... --out DIR [--interval S] [--count N]
                                                             write <UTC>-w<i>.prom snapshots for `metrics`
  checks.py metrics --dir DIR --workers N [--min-samples K] [--resets-allowed]
  checks.py result  trie-results.json                         zero failures, positive completions
  checks.py window  trie-lifecycle.jsonl --start-ns A --end-ns B
                                                             steady-window totals, prefix hit rate, zero failures
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import http.client
import json
import math
from pathlib import Path
import re
import sys
import time
import urllib.request


def request(command, endpoint, model=None, timeout=30):
    """One `canary`, `flush` or `metrics` request; return the body or raise on HTTP/protocol failure."""
    payload, method, path = None, "GET", "/metrics"
    if command == "canary":
        path, method, timeout = "/v1/chat/completions", "POST", 90
        payload = json.dumps({"model": model, "messages": [{"role": "user", "content": "ping"}],
                              "max_tokens": 4, "temperature": 0}).encode()
    elif command == "flush":
        path, method = "/flush_cache", "POST"
    elif command != "metrics":
        raise ValueError(f"unknown request {command!r}")
    req = urllib.request.Request(endpoint.rstrip("/") + path, data=payload, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
    if command == "canary":
        try:
            content = json.loads(body)["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ValueError("invalid canary response") from exc
        if not isinstance(content, str):
            raise ValueError("canary response has no text")
    return body


def parse_worker_metrics(body, source="metrics"):
    """Read request counters and queues from a live scrape or a saved snapshot."""
    metrics = {}
    for line in body.splitlines():
        match = re.match(r"^(sglang:num_[a-z_]+)(?:\{.*\})?\s+(\S+)", line)
        if not match:
            continue
        key, raw = match.groups()
        if key not in ("sglang:num_requests_total", "sglang:num_running_reqs") \
                and not key.endswith("queue_reqs"):
            continue
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{source}: invalid {key}={raw}")
        metrics[key] = metrics.get(key, 0) + value
    # The request counter is labeled, so a server that has not yet finished a counted request
    # (health probes are not counted; P4D4 skips the startup warmup) declares it with no samples.
    if "sglang:num_requests_total" not in metrics and re.search(
            r"^# TYPE sglang:num_requests(?:_total)? counter$", body, re.M):
        metrics["sglang:num_requests_total"] = 0.0
    required = {"sglang:num_requests_total", "sglang:num_running_reqs", "sglang:num_queue_reqs"}
    if missing := required - metrics.keys():
        raise ValueError(f"{source}: missing metrics {sorted(missing)}")
    return metrics


def worker_metrics(endpoint):
    return parse_worker_metrics(request("metrics", endpoint, timeout=10).decode(), endpoint)


def wait_for_quiet(endpoints, quiet_seconds=60, timeout=900, poll_seconds=5):
    """Wait for all queues to drain and request counters to stay unchanged together.

    Includes PD bootstrap/transfer queues when exported. A failed metrics scrape
    restarts the quiet window; a counter reset fails the check immediately.
    """
    if not endpoints or min(quiet_seconds, timeout, poll_seconds) <= 0 or timeout < quiet_seconds:
        raise ValueError("need workers and positive intervals, with timeout >= quiet window")
    started = time.monotonic()
    previous, quiet_since, last_error = None, None, None
    with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        while time.monotonic() - started < timeout:
            try:
                current = list(pool.map(worker_metrics, endpoints))
            except (OSError, ValueError, http.client.HTTPException) as exc:
                quiet_since, last_error = None, str(exc) or type(exc).__name__
            else:
                now = time.monotonic()
                counters = [item["sglang:num_requests_total"] for item in current]
                if previous is not None and any(new < old for new, old in zip(counters, previous)):
                    raise ValueError("worker request counter reset while waiting for quiet")
                busy = any(value != 0 for item in current for key, value in item.items()
                           if key != "sglang:num_requests_total")
                if busy:
                    quiet_since = None
                elif quiet_since is None or counters != previous:
                    quiet_since = now
                if quiet_since is not None and now - quiet_since >= quiet_seconds:
                    return {"wait_seconds": round(now - started, 1), "quiet_seconds": quiet_seconds,
                            "workers": dict(zip(endpoints, current))}
                previous, last_error = counters, None
            time.sleep(min(poll_seconds, max(0, timeout - (time.monotonic() - started))))
    raise TimeoutError(f"workers did not stay quiet for {quiet_seconds}s within {timeout}s"
                       + (f": {last_error}" if last_error else ""))


def check_result(path):
    result = json.loads(Path(path).read_text())
    if not isinstance(result, dict):
        raise ValueError(f"{path}: expected a Trie result object")
    # Trie cancels traces still open when the window ends, so a short window over long traces can
    # finish model requests without finishing a trace; completed_requests counts traces.
    for field in ("failed_requests", "completed_model_requests"):
        value = result.get(field)
        if type(value) is not int or (value != 0 if field == "failed_requests" else value <= 0):
            expected = "zero" if field == "failed_requests" else "positive"
            raise ValueError(f"{path}: {field} must be {expected}, got {value!r}")
    return result


def check_metrics(directory, workers, minimum, resets_allowed=False):
    """Require enough snapshots from every worker, with no request counter reset.

    resets_allowed: for harnesses whose role switch restarts workers; a reset is returned instead of raised.
    """
    if min(workers, minimum) < 1:
        raise ValueError("workers and minimum samples must be positive")
    counts = {f"w{i}": 0 for i in range(1, workers + 1)}
    previous, resets = {}, []
    for path in sorted(Path(directory).glob("*.prom")):
        match = re.fullmatch(r"\d{8}T\d{6}Z-(w\d+)\.prom", path.name)
        if not match or match[1] not in counts:
            raise ValueError(f"unexpected metrics file: {path}")
        worker = match[1]
        body = path.read_text()
        if resets_allowed and "sglang:num_requests_total" not in body:
            # A worker just restarted in a new role has not declared its request counter yet.
            body += "\n# TYPE sglang:num_requests_total counter\n"
        current = parse_worker_metrics(body, str(path))["sglang:num_requests_total"]
        if worker in previous and current < previous[worker]:
            if not resets_allowed:
                raise ValueError(f"request counter reset at {path}: {previous[worker]} -> {current}")
            resets.append(path.name)
        previous[worker] = current
        counts[worker] += 1
    for worker, count in counts.items():
        if count < minimum:
            raise ValueError(f"insufficient metrics samples for {worker}: {count} < {minimum}")
    return resets


def lifecycle_window(path, start_ns, end_ns):
    """Totals over the requests a Trie lifecycle file admitted in [start_ns, end_ns), warmups excluded.

    Traces still open when Trie stops are cancelled, so a cancelled request is a truncation, not a
    failure. In-flight is the time-weighted mean of admitted, unfinished requests inside the window.
    """
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
    busy_ns = 0
    for rid, (ts, _) in admits.items():
        busy_ns += max(0, min(ends.get(rid, end_ns), end_ns) - max(ts, start_ns))
    window = [rid for rid, (ts, warmup) in admits.items() if start_ns <= ts < end_ns and not warmup]
    done = [completes[rid] for rid in window if rid in completes]
    return {
        "completed_model_requests": len(done),
        "completed_requests": traces,
        "failed_requests": sum(1 for rid in window if failures.get(rid) not in (None, "CancelledError")),
        "cancelled_requests": sum(1 for rid in window if failures.get(rid) == "CancelledError"),
        "completed_prompt_tokens": sum(attrs["prompt_tokens"] for attrs in done),
        "cached_prompt_tokens": sum(attrs["cached_prompt_tokens"] for attrs in done),
        "completed_completion_tokens": sum(attrs["completion_tokens"] for attrs in done),
        "block_aligned_prefix_tokens": [attrs["block_aligned_prefix_tokens"] for attrs in done],
        "warmup_requests": sum(1 for ts, warmup in admits.values() if warmup),
        "inflight_mean": busy_ns / (end_ns - start_ns) if end_ns > start_ns else 0.0,
        "window_s": (end_ns - start_ns) / 1e9,
    }


def check_window(summary):
    """Steady validity for a continuous run: completions in the window and zero failed requests."""
    if summary["completed_model_requests"] <= 0 or summary["failed_requests"] != 0:
        raise ValueError(f"steady window needs completions and zero failures: "
                         f"completed={summary['completed_model_requests']} failed={summary['failed_requests']}")
    return summary


def scrape(endpoints, directory, interval=15.0, count=None):
    """Snapshot every worker's /metrics into DIR/<UTC>-w<i>.prom (i from 1, in --worker order)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    taken = 0
    while count is None or taken < count:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for index, endpoint in enumerate(endpoints, 1):
            body = request("metrics", endpoint, timeout=10)
            (directory / f"{stamp}-w{index}.prom").write_bytes(body)
        taken += 1
        if count is None or taken < count:
            time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    quiet = sub.add_parser("quiet", help="wait for drained, idle workers")
    quiet.add_argument("--worker", action="append", required=True, help="worker base URL (repeat)")
    quiet.add_argument("--quiet-seconds", type=float, default=60)
    quiet.add_argument("--timeout", type=float, default=900)
    snap = sub.add_parser("scrape", help="write periodic /metrics snapshots")
    snap.add_argument("--worker", action="append", required=True, help="worker base URL (repeat)")
    snap.add_argument("--out", required=True)
    snap.add_argument("--interval", type=float, default=15.0)
    snap.add_argument("--count", type=int, help="snapshots per worker (default: until interrupted)")
    metrics = sub.add_parser("metrics", help="monotonic request counters over snapshots")
    metrics.add_argument("--dir", required=True)
    metrics.add_argument("--workers", type=int, required=True)
    metrics.add_argument("--min-samples", type=int, default=2)
    metrics.add_argument("--resets-allowed", action="store_true",
                         help="report instead of fail on resets (workers restarted by a role switch)")
    result = sub.add_parser("result", help="Trie result: zero failures, positive completions")
    result.add_argument("path")
    window = sub.add_parser("window", help="steady-window totals from a Trie lifecycle file")
    window.add_argument("path")
    window.add_argument("--start-ns", type=int, required=True)
    window.add_argument("--end-ns", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "quiet":
            out = wait_for_quiet(args.worker, quiet_seconds=args.quiet_seconds, timeout=args.timeout)
        elif args.command == "scrape":
            scrape(args.worker, args.out, args.interval, args.count)
            return 0
        elif args.command == "metrics":
            out = {"resets": check_metrics(args.dir, args.workers, args.min_samples, args.resets_allowed)}
        elif args.command == "result":
            res = check_result(args.path)
            out = {k: res.get(k) for k in ("completed_model_requests", "completed_requests", "failed_requests")}
        else:
            out = check_window(lifecycle_window(args.path, args.start_ns, args.end_ns))
            prompt = out["completed_prompt_tokens"]
            out["prefix_hit_rate"] = out["cached_prompt_tokens"] / prompt if prompt else None
            out.pop("block_aligned_prefix_tokens")
    except (OSError, ValueError, TimeoutError, http.client.HTTPException) as exc:
        print(f"{args.command} check failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
