"""Per-worker P and D step-time models, fitted on the learning interval with a chronological 70/30 holdout."""
import math

import numpy as np

from .steps import chronological, consecutive_step_windows, decode_features, fit_steps, prefill_feature_schema, split_time


def fit_prefill_scale(windows, *, split_time, reference_coefficients):
    """Short new-host calibration with a predeclared hardware-model shape.

    Only one parameter is learned: observed/predicted aggregate busy seconds.
    This preserves total capacity on the training interval; it does not identify
    independent hardware coefficients. The holdout is untouched until reporting,
    then all windows determine the multiplier used for the fixed decision.
    Two holdout minutes are descriptive evidence, not a certified P95 bound.
    """
    reference = np.asarray(reference_coefficients, dtype=float)
    if (reference.shape not in ((4,), (5,)) or not np.isfinite(reference).all()
            or (reference < 0).any() or not reference.any()):
        raise ValueError('Need a finite nonnegative prefill reference')
    rows = sorted(windows, key=lambda r: r['end'])
    if len({r['node'] for r in rows}) != 1:
        raise ValueError('Scale calibration requires exactly one current worker')
    if any(not math.isclose(r['end']-r['start'], 60.) for r in rows):
        raise ValueError('Scale calibration requires 60s windows')
    if any(a['end'] > b['start'] for a, b in zip(rows, rows[1:])):
        raise ValueError('Overlapping calibration windows')
    rows, train = chronological(rows, split_time, 'start', 'end')
    if train < 4 or len(rows)-train < 2:
        raise ValueError('Need four training and two validation windows for one scale')
    x = np.asarray([r['features'] for r in rows], dtype=float)
    y = np.asarray([r['seconds'] for r in rows], dtype=float)
    if (x.shape != (len(rows), len(reference)) or not np.isfinite(x).all()
            or (x < 0).any() or not np.isfinite(y).all() or (y <= 0).any()):
        raise ValueError('Invalid calibration work or busy time')
    predicted = x @ reference
    if (predicted <= 0).any():
        raise ValueError('Reference predicts no work for a busy window')
    train_scale = float(y[:train].sum()/predicted[:train].sum())
    final_scale = float(y.sum()/predicted.sum())
    relative = np.abs(predicted[train:]*train_scale-y[train:])/y[train:]
    errors = prediction_errors(rows[train:], reference*train_scale)
    return dict(method='prior_shape_current_node_scale_v1',
        feature_schema=prefill_feature_schema(len(reference)),
        reference_coefficients=reference.tolist(), scale=final_scale,
        holdout_scale=train_scale, coefficients=(reference*final_scale).tolist(),
        holdout_coefficients=(reference*train_scale).tolist(),
        samples=len(rows), train_samples=train, errors=errors,
        central_errors=dict(mean_relative=float(relative.mean()),
            p95_relative=float(np.quantile(relative, .95)),
            total_relative=float(abs(predicted[train:].sum()*train_scale/y[train:].sum()-1))),
        window_scales=(y/predicted).tolist(),
        validation_unit='60s current-worker windows; chronological 70/30 wall-time split',
        validation_scope='One time multiplier only; short holdout, no inherited coefficient validation',
        legacy_thresholds_passed=None)


def fit_prefill(windows, *, split_time):
    """Retain the original wall-time split and restore-counter uncertainty.

    Twenty training minutes on one node cannot certify tail accuracy. Errors are
    observations, not an acceptance threshold or a prediction interval. Refitting
    all data only produces the coefficients used for the subsequent fixed plan.
    """
    rows, train = chronological(sorted(windows, key=lambda r: r['end']), split_time, 'start', 'end')
    if train < 16 or len(rows) - train < 6:
        raise ValueError('Need at least 16 training and 6 validation P windows per node')
    fit = fit_steps([r['features'] for r in rows], [r['seconds'] for r in rows],
                    minimum_samples=22, train_samples=train)
    fit['central_errors'] = fit.pop('errors')
    fit['errors'] = prediction_errors(rows[train:], fit['holdout_coefficients'])
    fit['central_thresholds_passed'] = fit.pop('passed')
    fit['feature_schema'] = prefill_feature_schema(len(fit['coefficients']))
    fit['validation_unit'] = '60s worker windows; same 70% wall-time split as original learning'
    fit['validation_scope'] = 'Exploratory per-node model; no independent model-selection validation'
    return fit


def fit_decode(rows, *, start, end, experts, top_k):
    """Eight consecutive launch intervals; never join across idle/missing steps.

    The three paper terms are retained. Context-dependent residuals and EAGLE
    acceptance remain limitations; the latter is measured independently per node.
    """
    usable = [r for r in rows if r['mode'] in ('decode', 'target_verify')
              and r.get('launch_bs', 0) > 0 and r.get('busy_interval_s', 0)
              and r['busy_interval_s'] > 0]
    if len({r['source'] for r in usable}) != 1:
        raise ValueError('Decode fit must contain exactly one physical worker stream')
    draft = {r['spec_num_draft_tokens'] or 1 for r in usable}
    if len(draft) != 1:
        raise ValueError('Speculative draft budget changed during calibration')
    windows = consecutive_step_windows(usable, [decode_features(r, experts, top_k) for r in usable])
    if any(w['interval_start_ts'] is None or w['interval_end_ts'] is None for w in windows):
        raise ValueError('Per-node wall-time split requires measured launch interval endpoints')
    # Eight-step groups get an explicit beginning; discard groups that cross the split.
    windows = sorted((w for w in windows if w['interval_start_ts'] >= start and w['interval_end_ts'] <= end),
                     key=lambda w: w['interval_start_ts'])
    windows, train = chronological(windows, split_time(start, end), 'interval_start_ts', 'interval_end_ts')
    if train < 40 or len(windows) - train < 20:
        raise ValueError('Insufficient chronological D windows')
    fit = fit_steps([w['features'] for w in windows],
                    [w['busy_interval_s'] for w in windows], train_samples=train)
    fit['legacy_thresholds_passed'] = fit.pop('passed')
    fit['feature_schema'] = ['steps', 'requests', 'expert_fraction']
    fit['step_coverage'] = len(windows) * 8 / len(usable)
    fit['batch_support'] = sorted({r['launch_bs'] for r in usable})
    fit['draft_tokens'] = draft.pop()
    fit['accepted_tokens_per_step'] = sum(r['decode_tokens'] for r in usable) / sum(r['launch_bs'] for r in usable)
    fit['mean_context'] = sum(r['launch_seq_lens_sum'] for r in usable) / sum(r['launch_bs'] for r in usable)
    fit['validation_unit'] = '8 consecutive busy steps; first 70% wall-time train'
    return fit


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
