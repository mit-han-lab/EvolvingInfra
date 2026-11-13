#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Decode and prefill batch statistics of engines in one window of a replay, from engine logs.

    python engine_batches.py decode  RUN_DIR --skip-s 1200 --window-s 600 ENGINE_LOG [ENGINE_LOG ...]
    python engine_batches.py prefill RUN_DIR --skip-s 1200 --window-s 600 ENGINE_LOG [ENGINE_LOG ...]

RUN_DIR holds trie-lifecycle-*.jsonl (it gives the replay's start). Give the logs of the workers
that held the role in that window. Engine logs carry local time without a zone, so the script
prints the window in log time: check it against the log's timestamps, and pass
--utc-offset-hours (the zone the logs were written in, e.g. -7) when it is off. A wrong offset
still finds lines, from another part of the run.

decode: per bucket of running requests, the accepted draft length, tokens per second per
request, and the time of one decode step (accepted length / per-request rate). Equal step
times with a lower accepted length mean the model's outputs are harder to draft (compare the
replay settings), not a slower engine. prefill: new tokens per second and queue lengths.
"""
import argparse
import collections
import datetime as dt
import glob
import json
import os
import re
import statistics as st
import time

STAMP = r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?) TP0\] "
DECODE = re.compile(STAMP + r"Decode batch, #running-req: (\d+), #token: (\d+),.*?accept len: ([\d.]+)"
                            r".*?gen throughput \(token/s\): ([\d.]+)")
PREFILL = re.compile(STAMP + r"Prefill batch.*?#new-token: (\d+), #cached-token: (\d+).*?#queue-req: (\d+)")


def replay_start(directory):
    firsts = []
    for path in glob.glob(os.path.join(directory, "trie-lifecycle-*.jsonl")):
        with open(path) as f:
            line = f.readline()
        if line:
            firsts.append(json.loads(line)["ts_ns"] / 1e9)
    if not firsts:
        raise SystemExit(f"no trie-lifecycle-*.jsonl in {directory}")
    return min(firsts)


def seconds(stamp, offset_hours):
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in stamp else "%Y-%m-%d %H:%M:%S"
    zone = dt.timezone(dt.timedelta(hours=offset_hours))
    return dt.datetime.strptime(stamp, fmt).replace(tzinfo=zone).timestamp()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=("decode", "prefill"))
    parser.add_argument("directory")
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--skip-s", type=float, required=True)
    parser.add_argument("--window-s", type=float, required=True)
    parser.add_argument("--utc-offset-hours", type=float, default=time.localtime().tm_gmtoff / 3600)
    args = parser.parse_args()
    start = replay_start(args.directory) + args.skip_s
    pattern = DECODE if args.kind == "decode" else PREFILL
    hits = []
    for path in args.logs:
        for line in open(path, errors="replace"):
            m = pattern.match(line)
            if m and start <= seconds(m.group(1), args.utc_offset_hours) < start + args.window_s:
                hits.append(m.groups()[1:])
    zone = dt.timezone(dt.timedelta(hours=args.utc_offset_hours))
    stamp = lambda t: dt.datetime.fromtimestamp(t, zone).strftime("%Y-%m-%d %H:%M:%S")  # noqa: E731
    print(f"window in log time: {stamp(start)} to {stamp(start + args.window_s)} "
          f"(UTC{args.utc_offset_hours:+g}), {len(hits)} batches from {len(args.logs)} logs")
    if not hits:
        raise SystemExit("no matching log lines in the window (check --utc-offset-hours)")
    if args.kind == "prefill":
        new = sum(int(h[0]) for h in hits)
        queue = sorted(int(h[2]) for h in hits)
        print(json.dumps({"workers": len(args.logs), "batches": len(hits),
                          "new_tokens_per_s": round(new / args.window_s),
                          "cached_per_new_token": round(sum(int(h[1]) for h in hits) / max(1, new), 1),
                          "queue_mean": round(st.mean(queue), 1), "queue_p90": queue[int(.9 * len(queue))]}, indent=2))
        return
    buckets = collections.defaultdict(list)
    for running, tokens, accept, rate in hits:
        if float(rate) > 0 and int(running) > 0:
            buckets[min(int(running) // 8 * 8, 64)].append((float(accept), float(rate) / int(running)))
    print(f"{'running':>9} {'batches':>8} {'accept len':>11} {'tok/s/req':>10} {'ms/step':>8}")
    for low in sorted(buckets):
        values = buckets[low]
        accept, rate = st.mean(v[0] for v in values), st.median(v[1] for v in values)
        print(f"{low:>4}-{low + 7:<4} {len(values):>8} {accept:>11.2f} {rate:>10.1f} {1000 * accept / rate:>8.1f}")


if __name__ == "__main__":
    main()
