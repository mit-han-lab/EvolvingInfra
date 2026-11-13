# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Scheduler step traces: loading the iteration records, D step features and groups of consecutive steps."""
from functools import cache

import numpy as np

from evolving_infra.planner.perf_model.step_fit import expert_fraction
from evolving_infra.planner.utils import json_lines


def load_steps(paths, start_unix, end_unix, records=None):
    """Steps of each trace file with ts in [start_unix, end_unix], tagged with their source and the
    launch interval to the next step of the same busy run. `records` maps a path to its rows when they
    are already in memory (as the file reads); those rows are tagged in place."""
    rows = []
    for path in paths:
        previous = None
        source = str(path)
        for row in json_lines(path) if records is None else records[source]:
            if not start_unix <= row["ts"] <= end_unix:
                previous = None
                continue
            row["source"], row["busy_interval_s"] = source, None
            if (
                previous is not None
                and not row["after_idle"]
                and row["forward_iter"] == previous["forward_iter"] + 1
                and row["mode"] == previous["mode"]
            ):
                interval = row["launch_mono"] - previous["launch_mono"]
                if interval > 0:
                    # Interval i -> i+1 belongs to step i, not step i+1.
                    previous["busy_interval_s"] = interval
            rows.append(row)
            previous = row
    return sorted(rows, key=lambda r: (r["ts"], r["source"], r["forward_iter"]))


def launch_wall(row):
    """Wall time a step was launched: its result time minus its duration."""
    return row["ts"] - row["dur_ms"] / 1000


def decode_features(row, experts, top_k):
    """[1, batch, expert fraction] of one D step: the fraction of routed experts its batch*draft tokens activate."""
    b = row["launch_bs"]
    q = row["spec_num_draft_tokens"] or 1
    return [1.0, b, _expert_fraction(b * q, experts, top_k)]


@cache
def _expert_fraction(tokens, experts, top_k):
    """The feature of a decode step, computed once per value: a fit has one row per step."""
    return float(expert_fraction(tokens, experts, top_k))


def consecutive_step_windows(sampled, features, size=8):
    """Sum nonoverlapping, consecutive steps to average overlap launch jitter.

    Linearity preserves the original step coefficients: sum(X @ beta) equals
    sum(X) @ beta. Never bridge an idle gap, missing step, worker, or mode.
    """
    pending = {}
    windows = []
    for row, feature in zip(sampled, features):
        group = pending.setdefault(row["source"], [])
        if group and (
            row["forward_iter"] != group[-1][0]["forward_iter"] + 1
            or row["mode"] != group[-1][0]["mode"]
        ):
            group.clear()
        group.append((row, feature))
        if len(group) == size:
            windows.append(
                dict(
                    ts=row["ts"],
                    interval_start_ts=launch_wall(group[0][0]) if "dur_ms" in group[0][0] else None,
                    interval_end_ts=launch_wall(row) + row["busy_interval_s"] if "dur_ms" in row else None,
                    source=row["source"],
                    busy_interval_s=sum(r["busy_interval_s"] for r, _ in group),
                    features=np.sum([x for _, x in group], axis=0).tolist(),
                )
            )
            group.clear()
    return sorted(windows, key=lambda r: (r["ts"], r["source"]))
