"""Opt-in P capacity calibration including measured cache preparation demand.

This predicts aggregate service time, not individual GPU steps. The three paper
features are retained; host/device prefix terms are empirical corrections, not
identified H2D/network bandwidth. Callers must preserve the original failed fits.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np

from .model import fit_steps


def join_cache_tiers(rows, request_paths):
    # Import here to keep trace's optional capacity path free of an import cycle.
    from .trace import json_lines

    joined = defaultdict(lambda: dict(participants=0, host=0, device=0))
    for path in request_paths:
        source = str(Path(path).parent)
        for req in json_lines(path):
            if req.get("retraction_count", 0):
                raise ValueError("Retracted P requests require separate cache accounting")
            steps = req["prefill_iters"]
            for step in steps:
                joined[source, step]["participants"] += 1
            if steps:
                host, device = req["cached_tokens_host"], req["cached_tokens_device"]
                if min(host, device) < 0 or host + device != req["cached_tokens"]:
                    raise ValueError("Host/device tiers do not cover request cache hits")
                first = joined[source, steps[0]]
                first["host"] += host
                first["device"] += device
    features = []
    for row in rows:
        tier = joined[str(Path(row["source"]).parent), row["forward_iter"]]
        if tier["participants"] != row["launch_bs"]:
            raise ValueError("Incomplete request-to-prefill-step join")
        features.append([1., row["new_prefill_tokens"], row["prefill_attention_pairs"],
                         tier["host"], tier["device"]])
    return features


def fit_capacity_totals(episodes, *, start_unix, end_unix, eligible_steps):
    """60s bins of COMPLETE episodes, split at 70% of the learning wall time.

    Each bin sums service of episodes finishing within it; it is not utilization
    in that minute. Idle is excluded. Groups crossing the train boundary are
    excluded explicitly, never split or partly reused. All exclusions are counted.
    """
    split_time = start_unix + .7 * (end_unix - start_unix)
    groups = {}
    for e in episodes:
        index = int((e["ts"] - start_unix) // 60)
        key = e["source"], index
        g = groups.setdefault(key, dict(
            source=e["source"], ts=start_unix+(index+1)*60,
            start_ts=e["ts"]-e["busy_interval_s"],
            features=np.zeros(len(e["features"])), seconds=0., steps=0, episodes=0))
        g["features"] += e["features"]
        g["seconds"] += e["busy_interval_s"]
        g["steps"] += e["step_count"]
        g["episodes"] += 1
        g["start_ts"] = min(g["start_ts"], e["ts"]-e["busy_interval_s"])
    ordered = sorted(groups.values(), key=lambda g: (g["ts"], g["source"]))
    samples = [g for g in ordered if g["ts"] <= split_time or g["start_ts"] >= split_time]
    split = sum(g["ts"] <= split_time for g in samples)
    if split < 60 or len(samples)-split < 20:
        raise ValueError("Need 60 training and 20 validation capacity groups")
    x = [g["features"].tolist() for g in samples]
    fit = fit_steps(x, [g["seconds"] for g in samples], train_samples=split)
    fit.update(validation_unit="sum of complete busy periods ending in each60s bin",
               split_policy="first70%wall time; no episode crossing train boundary",
               feature_schema=["steps", "uncached_tokens", "attention_pairs", "host_hit", "device_hit"],
               step_coverage=sum(g["steps"] for g in samples)/eligible_steps,
               excluded_boundary_groups=len(ordered)-len(samples),
               worker_total_errors={}, step_join_coverage=1.)
    co = np.asarray(fit["holdout_coefficients"])
    for source in sorted({g["source"] for g in samples[split:]}):
        subset = [g for g in samples[split:] if g["source"] == source]
        fit["worker_total_errors"][source] = abs(
            sum(np.dot(g["features"], co) for g in subset)/sum(g["seconds"] for g in subset)-1)
    fit["passed"] &= (fit["step_coverage"] >= .5 and
                      all(e <= .15 for e in fit["worker_total_errors"].values()))
    fit["groups"] = [dict(g, features=g["features"].tolist()) for g in samples]
    return fit


def fit_cache_capacity(rows, request_paths, start_unix, end_unix):
    from .trace import _busy_episodes

    usable = [r for r in rows if r["mode"] == "extend" and r.get("launch_bs")
              and r.get("dur_ms", 0) > 0 and r.get("prefill_attention_pairs") is not None]
    features = join_cache_tiers(usable, request_paths)
    return fit_capacity_totals(_busy_episodes(usable, features), start_unix=start_unix,
                               end_unix=end_unix, eligible_steps=len(usable))
