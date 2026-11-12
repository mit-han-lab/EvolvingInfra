#!/usr/bin/env python3
"""The planner's timeline of a run: cold start, fit, and every role change.

    python switch_times.py PLANNER_DIR [BENCH_DIR]

PLANNER_DIR is `planner/` in the work directory of the host that runs the router. With
BENCH_DIR (the replay's trie-lifecycle-*.jsonl) times are seconds after the replay's first
event, the clock of benchmarks/replay/score.py --skip-s; without it, seconds after the
planner's first journal entry. Exit code 1 when the planner failed.
"""
import glob
import json
import os
import sys
from pathlib import Path


def main(planner, bench=None):
    rows = [json.loads(line) for line in (Path(planner) / "role-decisions.jsonl").read_text().splitlines()]
    start, clock = rows[0]["at"], "planner clock"
    if bench:
        firsts = [json.loads(open(p).readline())["ts_ns"] / 1e9
                  for p in glob.glob(os.path.join(bench, "trie-lifecycle-*.jsonl"))]
        start, clock = min(firsts), "replay clock"
    print(f"seconds on the {clock}")
    failed = (Path(planner) / "PLANNER_FAILED").exists()
    for row in rows:
        at, event = round(row["at"] - start), row.get("event")
        if event == "learning_started":
            print(f"{at:>6}  cold start over ({row['cold_start'].get('reason')}); learning starts")
        elif event == "warmup_fit_completed":
            print(f"{at:>6}  fit done")
        elif event == "role_update_confirmed":
            roles = (row.get("assignment") or {}).get("roles") or {}
            p = sum(r == "prefill" for r in roles.values())
            print(f"{at:>6}  switched to {p}P{len(roles) - p}D")
        elif event == "role_update_hold":
            print(f"{at:>6}  decision: hold")
        elif event == "controller_failed":
            failed = True
            print(f"{at:>6}  PLANNER FAILED: {row.get('error')}")
    return 1 if failed else 0


if __name__ == "__main__":
    if sys.argv[1:2] in (["-h"], ["--help"]):
        print(__doc__)
        sys.exit(0)
    if len(sys.argv) not in (2, 3) or sys.argv[1].startswith("-"):
        sys.exit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))
