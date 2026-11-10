import copy

import pytest

from evolving_infra.planner.fit.learning import fit_bundle
from evolving_infra.planner.fit.models import fit_prefill_scale, prediction_errors


def windows():
    return [dict(node='node0', start=60*i, end=60*(i+1),
                 seconds=2*(i+1), features=[i+1, 0, 0, 0],
                 restore_lo=0, restore_hi=0) for i in range(10)]


def fit(rows):
    return fit_prefill_scale(rows, split_time=420, reference_coefficients=[1, 0, 0, 0])


def test_scale_recovers_total_and_does_not_claim_four_parameter_fit():
    rows = windows()
    prior = copy.deepcopy(rows)
    result = fit(rows)
    assert result['scale'] == result['holdout_scale'] == 2
    assert result['errors']['p95_relative'] == 0
    assert result['legacy_thresholds_passed'] is None
    assert rows == prior


def test_validation_change_cannot_change_training_scale():
    rows = windows()
    for row in rows[7:]:
        row['seconds'] *= 3
    result = fit(rows)
    assert result['holdout_scale'] == 2
    assert result['scale'] > 2
    assert result['central_errors']['p95_relative'] == pytest.approx(2/3)


@pytest.mark.parametrize('change', ['other_node', 'overlap', 'short', 'nan', 'negative'])
def test_invalid_observations_rejected(change):
    rows = windows()
    if change == 'other_node': rows[-1]['node'] = 'node1'
    if change == 'overlap': rows[2] = copy.deepcopy(rows[1])
    if change == 'short': rows = rows[:8]
    if change == 'nan': rows[0]['seconds'] = float('nan')
    if change == 'negative': rows[0]['features'][1] = -1
    with pytest.raises(ValueError): fit(rows)


def test_restore_uncertainty_is_not_lost_in_short_fit():
    rows = windows()
    for row in rows:
        row['features'][3] = 1
        row['restore_hi'] = 5
    result = fit_prefill_scale(rows, split_time=420, reference_coefficients=[1, 0, 0, 1])
    assert result['errors']['p95_relative'] > result['central_errors']['p95_relative']


def test_vision_calibration_preserves_fifth_term_through_bundle(monkeypatch):
    rows = windows()
    for row in rows:
        row['features'][1] = 64 * row['features'][0]
        row['features'].append(1e12)
        row['started_cached_tokens'] = 0
        row['seconds'] += .02
    monkeypatch.setattr('evolving_infra.planner.fit.learning.fit_decode',
                        lambda *args, **kwargs: {'coefficients': [1, 0, 0]})
    arguments = dict(windows=rows, decode_rows={'node0': [{}]},
                     workers={'node0': dict(kv_capacity_tokens=1000, max_running_requests=8)},
                     start=0, end=600, experts=8, top_k=1)
    # A short run needs a declared shape, and the old text shape cannot silently
    # absorb the time spent on images into its overall scale.
    with pytest.raises(ValueError, match='Missing current-run calibrated role donors'):
        fit_bundle(**arguments, predeclared_prefill_shape=[1, 0, 0, 0])
    bundle = fit_bundle(**arguments, predeclared_prefill_shape=[2, 0, 0, 0, 2e-14])
    model = bundle['workers']['node0']['prefill']['models'][0]
    assert model['feature_schema'][-1] == 'vision_matmul_flops'
    assert model['coefficients'] == pytest.approx([2, 0, 0, 0, 2e-14])


def test_text_only_windows_fit_the_vision_shape_with_zero_vision_work(monkeypatch):
    rows = windows()
    for row in rows:
        row['features'][1] = 64 * row['features'][0]
        row['started_cached_tokens'] = 0
    monkeypatch.setattr('evolving_infra.planner.fit.learning.fit_decode',
                        lambda *args, **kwargs: {'coefficients': [1, 0, 0]})
    bundle = fit_bundle(windows=rows, decode_rows={'node0': [{}]},
                        workers={'node0': dict(kv_capacity_tokens=1000, max_running_requests=8)},
                        start=0, end=600, experts=8, top_k=1,
                        predeclared_prefill_shape=[2, 0, 0, 0, 2e-14])
    model = bundle['workers']['node0']['prefill']['models'][0]
    assert model['feature_schema'][-1] == 'vision_matmul_flops'
    assert model['coefficients'] == pytest.approx([2, 0, 0, 0, 2e-14])
    assert all(len(row['features']) == 4 for row in rows)


def test_validation_propagates_counter_edge_uncertainty():
    samples = [dict(node="node0", features=[1, 100, 1000, 100],
                    seconds=1., restore_lo=0, restore_hi=200)] * 30
    # Center prediction is exact, but the unresolved transfer count adds ±0.5s.
    errors = prediction_errors(samples, [.5, 0, 0, .005])
    assert errors["predicted_total_s"] == errors["observed_total_s"]
    assert errors["p95_relative"] == pytest.approx(.5)
    assert errors["worker_total_errors"]["node0"] == pytest.approx(.5)
