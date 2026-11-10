"""Per-physical-worker calibration and an external, one-time P/D decision.

This module never changes live routing. The deployment controller stops learning,
freezes the selected worker lists/common D cap, then starts all comparison arms
with that same allocation. No coefficients or fits run on the request hot path.
"""
from itertools import combinations
import math

import numpy as np

from .model import expert_fraction, fit_steps, prefill_feature_schema
from .restore import prediction_errors
from .trace import _capacity_windows, _features


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
    rows = [r for r in rows if r['end'] <= split_time or r['start'] >= split_time]
    train = sum(r['end'] <= split_time for r in rows)
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
    rows = sorted(windows, key=lambda r: r['end'])
    rows = [r for r in rows if r['end'] <= split_time or r['start'] >= split_time]
    train = sum(r['end'] <= split_time for r in rows)
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
    windows = _capacity_windows(usable, [_features(r, 'decode', experts, top_k) for r in usable])
    if any(w['interval_start_ts'] is None or w['interval_end_ts'] is None for w in windows):
        raise ValueError('Per-node wall-time split requires measured launch interval endpoints')
    split = start + .7 * (end - start)
    # Eight-step groups get an explicit beginning; discard groups that cross split.
    windows = sorted((w for w in windows if w['interval_start_ts'] >= start
                      and w['interval_end_ts'] <= end
                      and (w['interval_end_ts'] <= split or w['interval_start_ts'] >= split)),
                     key=lambda w: w['interval_start_ts'])
    train = sum(w['interval_end_ts'] <= split for w in windows)
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


def decode_operating_points(worker, *, output_tokens, mean_context, experts, top_k,
                            min_user_tps=100., memory_fraction=.95):
    fit = worker['decode']
    residual = fit['errors']['p95_relative']
    if not math.isfinite(residual) or not 0 <= residual < 1:
        raise ValueError('D relative-error correction must be finite and below one')
    if (output_tokens <= 0 or mean_context <= 0 or min_user_tps <= 0
            or not 0 < memory_fraction <= 1):
        raise ValueError('Invalid D workload/capacity bounds')
    cap = min(worker['max_running_requests'],
              math.floor(worker['kv_capacity_tokens'] * memory_fraction / mean_context))
    result = {}
    for batch in fit['batch_support']:
        if batch <= 0 or batch > cap:
            continue
        feature = [1., batch, float(expert_fraction(batch * fit['draft_tokens'], experts, top_k))]
        step = float(np.dot(fit['coefficients'], feature))
        if not math.isfinite(step) or step <= 0:
            raise ValueError('Invalid predicted D step time')
        conservative = step / (1 - residual)
        tps = fit['accepted_tokens_per_step'] / conservative
        if tps >= min_user_tps:
            result[batch] = dict(batch=batch, step_s=step, conservative_step_s=conservative,
                                 user_tps=tps, capacity_rps=batch*tps/output_tokens)
    return result


def select_fixed_plan(workers, *, request_features, output_tokens, mean_context,
                      experts, top_k, min_user_tps=100.):
    """Enumerate integer subsets with measured role coverage; no fabricated fits.

    RR gives equal request counts, so D capacity is len(D)*min(single-D capacity).
    A freely balanced sum is retained only as an upper bound. P sum assumes the
    scheduler can use P capacity; cache placement and queueing can violate this.
    The selected common cap satisfies every D's measured batch support/SLO/model
    memory bound and must ALSO be passed to every D engine's actual launch argv.
    """
    ids = sorted(workers)
    if len(ids) < 2 or len(request_features) not in (4, 5):
        raise ValueError('Need multiple workers and the explicit aggregate P schema')
    if not np.isfinite(request_features).all() or min(request_features) < 0:
        raise ValueError('Invalid aggregate request features')
    options, demands = {}, {}
    for key, worker in workers.items():
        if worker.get('prefill'):
            if worker['prefill']['feature_schema'] != prefill_feature_schema(len(request_features)):
                raise ValueError('P feature schema mismatch')
            demands[key] = float(np.dot(worker['prefill']['coefficients'], request_features))
            if not math.isfinite(demands[key]) or demands[key] <= 0:
                raise ValueError('Invalid predicted P demand')
        if worker.get('decode'):
            options[key] = decode_operating_points(worker, output_tokens=output_tokens,
                mean_context=mean_context, experts=experts, top_k=top_k, min_user_tps=min_user_tps)
    candidates, excluded = [], []
    for count in range(1, len(ids)):
        for p_ids in combinations(ids, count):
            d_ids = [w for w in ids if w not in p_ids]
            missing = [f'{w}:P' for w in p_ids if w not in demands]
            missing += [f'{w}:D' for w in d_ids if w not in options]
            if missing:
                excluded.append(dict(prefill_workers=list(p_ids), missing_roles=missing))
                continue
            common = set.intersection(*(set(options[w]) for w in d_ids))
            if not common:
                excluded.append(dict(prefill_workers=list(p_ids), reason='No common supported D cap at requested TPS'))
                continue
            # Maximize capacity within observed support, then prefer the larger cap.
            batch = max(common, key=lambda b: (min(options[w][b]['capacity_rps'] for w in d_ids), b))
            d_capacity = len(d_ids) * min(options[w][batch]['capacity_rps'] for w in d_ids)
            p_capacity = sum(1 / demands[w] for w in p_ids)
            candidates.append(dict(prefill_workers=list(p_ids), decode_workers=d_ids,
                prefill_nodes=count, decode_nodes=len(d_ids), decode_max_running_requests=batch,
                prefill_capacity_rps=p_capacity, decode_rr_capacity_rps=d_capacity,
                decode_balanced_upper_rps=sum(options[w][batch]['capacity_rps'] for w in d_ids),
                predicted_capacity_rps=min(p_capacity, d_capacity),
                per_decode_worker={w:options[w][batch] for w in d_ids}))
    if not candidates:
        raise ValueError('No fully measured allocation satisfies common D constraints')
    best = sorted(candidates, key=lambda c: (-c['predicted_capacity_rps'], c['prefill_nodes'], c['prefill_workers']))[0]
    return dict(schema_version=1, mode='fixed_after_learning', selected=best,
                candidates=candidates, excluded=excluded, min_user_tps=min_user_tps,
                limitations=[
                    'Model-selected capacity, not measured global optimum or tail-latency guarantee.',
                    'P packing/cache restoration and decode context/acceptance can change after role assignment.',
                    'TPS uses measured mean acceptance and an empirical P95 residual correction, not a per-request guarantee.',
                    'Mean KV occupancy is a sizing estimate; long-context memory tails need runtime verification.',
                    'Multiagent task concurrency is not the number of in-flight model requests.',
                ])
