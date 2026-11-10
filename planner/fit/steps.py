"""Scheduler step traces and the nonnegative least-squares step-time fit shared by the P and D models.

All times are seconds. This is a capacity model, not a queue-time simulator or a tail-latency proof.
"""
from functools import cache
import math

import numpy as np
from scipy.optimize import nnls

from ..common import json_lines


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


def decode_features(row, experts, top_k):
    """[1, batch, expert fraction] of one D step: the fraction of routed experts its batch*draft tokens activate."""
    b = row["launch_bs"]
    q = row["spec_num_draft_tokens"] or 1
    return [1.0, b, _expert_fraction(b * q, experts, top_k)]


@cache
def _expert_fraction(tokens, experts, top_k):
    """The feature of a decode step, computed once per value: a fit has one row per step."""
    return float(expert_fraction(tokens, experts, top_k))


def launch_wall(row):
    """Wall time a step was launched: its result time minus its duration."""
    return row["ts"] - row["dur_ms"] / 1000


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


def expert_fraction(tokens, experts: int, top_k: int):
    if not 0 < top_k <= experts:
        raise ValueError("Require 0 < top_k <= experts")
    tokens = np.asarray(tokens, dtype=float)
    if not np.isfinite(tokens).all() or (tokens < 0).any():
        raise ValueError("Token counts must be finite and nonnegative")
    if top_k == experts:
        return (tokens > 0).astype(float)
    return -np.expm1(tokens * math.log1p(-top_k / experts))


def prefill_feature_schema(count):
    """The existing restore-aware model, with actual vision work when present."""
    fields = ['steps', 'new_tokens', 'attention_pairs', 'completed_restore_tokens']
    if count == 5:
        fields.append('vision_matmul_flops')
    elif count != 4:
        raise ValueError('Unknown prefill feature schema')
    return fields


TRAIN_FRACTION = .7  # chronological holdout: the first 70% trains


# Holdout acceptance of a fit (its "passed"), and of each worker's total time where reported.
ERROR_LIMITS = {"mean_relative": .15, "p95_relative": .30, "total_relative": .10}


def split_time(start, end):
    """The train/holdout boundary of the learning interval [start, end]."""
    return start + TRAIN_FRACTION * (end - start)


def chronological(rows, split, start_key, end_key):
    """Rows wholly before or after `split` (a row crossing it is left out) and how many of them train."""
    kept = [r for r in rows if r[end_key] <= split or r[start_key] >= split]
    return kept, sum(r[end_key] <= split for r in kept)


def within_error_limits(errors):
    return all(errors[name] <= limit for name, limit in ERROR_LIMITS.items())


def fit_steps(features, seconds, *, minimum_samples=60, train_samples=None):
    """Chronological 70/30 holdout; scale columns before nonnegative fitting.

    Condition/rank are reported because accurate predictions do not imply that
    individual a,b,c coefficients are separately identifiable. A rank-deficient
    design is rejected instead of inventing a per-token coefficient.
    """
    x, y = np.asarray(features, dtype=float), np.asarray(seconds, dtype=float)
    if x.ndim != 2 or len(x) != len(y) or len(x) < minimum_samples:
        raise ValueError("Insufficient step samples")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Non-finite features or nonpositive durations")
    split = int(len(y) * TRAIN_FRACTION) if train_samples is None else train_samples
    if not isinstance(split, int) or not x.shape[1] <= split < len(y):
        raise ValueError("Invalid chronological training split")
    scale = np.maximum(np.linalg.norm(x[:split], axis=0), 1e-30)
    scaled = x[:split] / scale
    rank = int(np.linalg.matrix_rank(scaled))
    if rank != x.shape[1]:
        raise ValueError(
            f"Step coefficients not identifiable: rank {rank}/{x.shape[1]}"
        )
    coefficients = nnls(scaled, y[:split])[0] / scale
    prediction = x[split:] @ coefficients
    truth = y[split:]
    relative = np.abs(prediction - truth) / truth
    errors = dict(
        mean_relative=float(relative.mean()),
        p95_relative=float(np.quantile(relative, 0.95)),
        total_relative=float(abs(prediction.sum() / truth.sum() - 1)),
    )
    # Refit on all samples only AFTER evaluating the untouched holdout.
    full_scale = np.maximum(np.linalg.norm(x, axis=0), 1e-30)
    final = nnls(x / full_scale, y)[0] / full_scale
    return dict(
        coefficients=final.tolist(),
        holdout_coefficients=coefficients.tolist(),
        samples=len(y),
        train_samples=split,
        errors=errors,
        condition_number=float(np.linalg.cond(scaled)),
        rank=rank,
        feature_min=x.min(axis=0).tolist(),
        feature_max=x.max(axis=0).tolist(),
        passed=within_error_limits(errors),
    )
