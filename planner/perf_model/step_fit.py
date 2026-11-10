"""The nonnegative least-squares step-time fit shared by the P and D models, and its train/holdout split.

All times are seconds. This is a capacity model, not a queue-time simulator or a tail-latency proof.
"""
import math

import numpy as np
from scipy.optimize import nnls


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
