#!/usr/bin/env python3
"""Per-request work and latency in one window of a replay, from the Trie lifecycle files.

    python request_breakdown.py RUN_DIR --skip-s 1800 --window-s 3600

RUN_DIR holds trie-lifecycle-*.jsonl. The window opens --skip-s after the first event, as in
benchmarks/replay/score.py. Prints the work of a request (prompt, uncached and output tokens)
and where its time goes (time to first token, decode time per output token). Two runs with the
same work and different times differ in the system; different work means different inputs.
"""
import argparse
import glob
import json
import os
import statistics as st


def rows(directory, skip, window):
    files = sorted(glob.glob(os.path.join(directory, "trie-lifecycle-*.jsonl")))
    if not files:
        raise SystemExit(f"no trie-lifecycle-*.jsonl in {directory}")
    first, admits, completes, ends = None, {}, {}, {}
    for i, path in enumerate(files):
        for line in open(path):
            e = json.loads(line)
            first = e["ts_ns"] if first is None else min(first, e["ts_ns"])
            rid = (i, e["request"]["experiment_request_id"])
            if e["event"] == "CLIENT_ADMIT":
                admits[rid] = (e["ts_ns"], bool(e["attrs"].get("warmup")))
            elif e["event"] == "CLIENT_COMPLETE":
                completes[rid], ends[rid] = e["attrs"], e["ts_ns"]
    start = first + skip * 1e9
    out = []
    for rid, (ts, warm) in admits.items():
        if warm or not start <= ts < start + window * 1e9 or rid not in completes:
            continue
        a = completes[rid]
        timing = a.get("client_timing") or {}
        t0, t1 = timing.get("request_start_ts_ns"), timing.get("first_parsed_token_ts_ns")
        if not (t0 and t1):
            continue
        decode = (ends[rid] - t1) / 1e9
        out.append(dict(prompt=a["prompt_tokens"], uncached=a["prompt_tokens"] - a["cached_prompt_tokens"],
                        output=a["completion_tokens"], ttft=(t1 - t0) / 1e9, decode=decode,
                        per_token=decode / max(1, a["completion_tokens"] - 1),
                        context=a.get("replay_context_mode", "recorded")))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory")
    parser.add_argument("--skip-s", type=float, required=True)
    parser.add_argument("--window-s", type=float, required=True)
    args = parser.parse_args()
    data = rows(args.directory, args.skip_s, args.window_s)
    if not data:
        raise SystemExit("no completed requests in the window")
    mean = lambda key: st.mean(r[key] for r in data)        # noqa: E731
    median = lambda key: st.median(r[key] for r in data)    # noqa: E731
    print(json.dumps({
        "requests": len(data), "requests_per_s": round(len(data) / args.window_s, 3),
        "replay_context_modes": sorted({r["context"] for r in data}),
        "prompt_tokens_mean": round(mean("prompt")), "uncached_tokens_mean": round(mean("uncached")),
        "output_tokens_mean": round(mean("output")), "output_tokens_median": round(median("output")),
        "ttft_s_mean": round(mean("ttft"), 2), "ttft_s_median": round(median("ttft"), 2),
        "decode_s_mean": round(mean("decode"), 2),
        "decode_ms_per_token_median": round(1000 * median("per_token"), 1),
        "decode_ms_per_token_mean": round(1000 * mean("per_token"), 1)}, indent=2))


if __name__ == "__main__":
    main()
