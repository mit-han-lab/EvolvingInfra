"""Aggregate P service model using completion counters for real H2D restores.

Counter replicas must be audited against the frozen engine: TP ranks can export
identical Prometheus labels. This module never infers replica count from values.
"""
import bisect
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from .restore_audit import include_scrape_collection_time

from .model import fit_steps, prefill_feature_schema


def counter_bracket(samples, timestamp):
    index = bisect.bisect_right([s[0] for s in samples], timestamp)
    if not 0 < index < len(samples):
        raise ValueError("Missing restore-counter time brackets")
    (a, ca), (b, cb) = samples[index-1:index+1]
    if not 0 < b-a < 10 or cb < ca:
        raise ValueError("Restore counter reset or scrape gap >=10s")
    return ca+(cb-ca)*(timestamp-a)/(b-a), ca, cb, b-a


def read_restore_counters(path, replicas):
    if not isinstance(replicas, int) or replicas < 1:
        raise ValueError("Require audited restore-counter replica count")
    meters = defaultdict(list)
    with gzip.open(path, "rt") as stream:
        for line in stream:
            for node in json.loads(line)["nodes"]:
                if node["role"] != "prefill" or node.get("errors"):
                    continue
                metrics = node.get("metrics", "")
                # A labeled Counter child first appears when inc() is called.
                # Verify this collector through its already-active backup series.
                if "# HELP sglang:hicache_backup_bytes_total " not in metrics:
                    continue
                values = [float(s.split()[-1])/replicas for s in metrics.splitlines()
                          if s.startswith("sglang:load_back_tokens_total{")
                          and 'pool="kv"' in s]
                if len(values) > 1:
                    raise ValueError("Ambiguous restore counter series")
                meters[f"node{node['node']}"].append((node["at"], sum(values)))
    for values in meters.values():
        if any(b[0] <= a[0] or b[1] < a[1] for a, b in zip(values, values[1:])):
            raise ValueError("Restore counter reset or unordered scrapes")
    return meters


def build_restore_windows(rows, request_paths, telemetry, replicas, start, end):
    """Fixed 60s windows: clipped busy wall time and launch-assigned step work.

    Incomplete busy periods at capture edges are excluded; only windows inside
    the completely observed busy/idle horizon are eligible. Boundary mismatch
    between a step's launch and finish remains part of prediction residuals.
    """
    from .trace import json_lines

    meters = read_restore_counters(telemetry, replicas)
    identities = defaultdict(lambda: dict(started=0, participants=0))
    for path in request_paths:
        node = Path(path).parent.parent.name
        for req in json_lines(path):
            for step in req["prefill_iters"]:
                identities[node, step]["participants"] += 1
            if req["prefill_iters"]:
                identities[node, req["prefill_iters"][0]]["started"] += 1
    usable = []
    for row in rows:
        if row["mode"] != "extend" or not row.get("launch_bs"):
            continue
        node = Path(row["source"]).parent.parent.name
        identity = identities[node, row["forward_iter"]]
        if identity["participants"] != row["launch_bs"]:
            raise ValueError("Incomplete request-to-step identity join")
        usable.append(dict(row, node=node, started=identity["started"],
                           launch_wall=row["ts"]-row["dur_ms"]/1000))
    has_vision = any(row.get('vision_matmul_flops', 0) for row in usable)
    periods, pending = defaultdict(list), {}
    for row in usable:
        node = row["node"]
        group = pending.get(node)
        if group and row["forward_iter"] != group[-1]["forward_iter"]+1:
            group = None
        if row["after_idle"]:
            if group:
                a, b = group[0]["launch_wall"], group[-1]["ts"]
                if a < b <= row["launch_wall"]:
                    periods[node].append((a, b, group))
            group = []
        if group is not None:
            group.append(row)
        pending[node] = group
    samples = []
    for node, episodes in periods.items():
        for index in range(int((end-start)//60)):
            a, b = start+index*60, start+(index+1)*60
            if not episodes[0][0] <= a < b <= episodes[-1][1]:
                continue
            steps = [r for _, _, group in episodes for r in group
                     if a <= r["launch_wall"] < b]
            seconds = sum(max(0, min(y, b)-max(x, a)) for x, y, _ in episodes)
            if not steps or seconds <= 0:
                continue
            ca, alo, ahi, agap = counter_bracket(meters[node], a)
            cb, blo, bhi, bgap = counter_bracket(meters[node], b)
            samples.append(dict(node=node, start=a, end=b, seconds=seconds,
                steps=len(steps), requests=sum(r["started"] for r in steps),
                features=[len(steps), sum(r["new_prefill_tokens"] for r in steps),
                          sum(r["prefill_attention_pairs"] for r in steps), cb-ca],
                restore_lo=max(0, blo-ahi), restore_hi=bhi-alo,
                max_scrape_gap_s=max(agap, bgap)))
            if has_vision:
                samples[-1]['features'].append(sum(r.get('vision_matmul_flops', 0) for r in steps))
    return sorted(samples, key=lambda s: (s["end"], s["node"])), len(usable)


def prediction_errors(samples, coefficients):
    if not samples:
        raise ValueError("No capacity validation windows")
    x = np.asarray([s["features"] for s in samples])
    y = np.asarray([s["seconds"] for s in samples])
    prediction = x @ np.asarray(coefficients)
    bounds = []
    for name in ("restore_lo", "restore_hi"):
        alternative = x.copy()
        alternative[:, 3] = [s[name] for s in samples]
        bounds.append(alternative @ coefficients)
    worst = np.maximum(np.abs(bounds[0]-y), np.abs(bounds[1]-y))/y
    total = max(abs(p.sum()/y.sum()-1) for p in bounds)
    worker = {}
    for node in sorted({s["node"] for s in samples}):
        indices = [i for i, s in enumerate(samples) if s["node"] == node]
        worker[node] = max(abs(p[indices].sum()/y[indices].sum()-1) for p in bounds)
    return dict(mean_relative=float(worst.mean()),
                p95_relative=float(np.quantile(worst, .95)),
                total_relative=float(total), worker_total_errors=worker,
                predicted_total_s=float(prediction.sum()), observed_total_s=float(y.sum()))


def fit_measured_restore(rows, request_paths, telemetry, replicas, start, end,
                         *, collection_clock_pad_s=None):
    samples, eligible = build_restore_windows(rows, request_paths, telemetry, replicas, start, end)
    if collection_clock_pad_s is not None:
        include_scrape_collection_time(samples, telemetry, replicas, collection_clock_pad_s)
    split = sum(s["end"] <= start+.7*(end-start) for s in samples)
    if split < 60 or len(samples)-split < 20:
        raise ValueError("Need 60 training and 20 validation capacity windows")
    x = np.asarray([s["features"] for s in samples])
    fit = fit_steps(x, [s["seconds"] for s in samples], train_samples=split)
    bounded = prediction_errors(samples[split:], fit["holdout_coefficients"])
    fit.update(validation_unit="60s busy wall windows with completed restore counters",
               feature_schema=prefill_feature_schema(len(fit['coefficients'])),
               central_errors=fit["errors"], errors={k: bounded[k] for k in
                   ("mean_relative", "p95_relative", "total_relative")},
               worker_total_errors=bounded["worker_total_errors"],
               restore_counter_replicas=replicas, windows=samples,
               step_coverage=sum(s["steps"] for s in samples)/eligible,
               request_samples=sum(s["requests"] for s in samples))
    count = fit["request_samples"]
    if count <= 0:
        raise ValueError("No measured P request arrivals")
    fit["aggregate_request_features"] = (x.sum(axis=0)/count).tolist()
    fit["passed"] &= (fit["step_coverage"] >= .5
        and bounded["mean_relative"] <= .15 and bounded["p95_relative"] <= .30
        and bounded["total_relative"] <= .10
        and all(e <= .15 for e in bounded["worker_total_errors"].values()))
    if collection_clock_pad_s is not None:
        fit.update(full_collection_interval_included=True,
                   counter_clock_padding_s=collection_clock_pad_s)
    return fit
