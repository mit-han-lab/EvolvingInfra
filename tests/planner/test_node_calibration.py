import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'python'))
from evolving_infra.planner.calibration.nodes import decode_operating_points, select_fixed_plan


def worker(p_rps=2.75, d_rps=1.):
    return dict(prefill=dict(coefficients=[1/p_rps, 0, 0, 0], feature_schema=[
        'steps', 'new_tokens', 'attention_pairs', 'completed_restore_tokens']),
        decode=dict(coefficients=[1/d_rps, 0, 0], errors={'p95_relative': 0},
                    batch_support=[1], draft_tokens=1, accepted_tokens_per_step=100),
        max_running_requests=1, kv_capacity_tokens=1000)


def choose(workers):
    return select_fixed_plan(workers, request_features=[1, 0, 0, 0],
                             output_tokens=100, mean_context=10, experts=8,
                             top_k=1, min_user_tps=.01)


def test_integer_optimum_not_continuous_rounding():
    ws = {str(i): worker() for i in range(9)}
    before = copy.deepcopy(ws)
    plan = choose(ws)
    assert plan['selected']['prefill_nodes'] == 3
    assert plan['selected']['predicted_capacity_rps'] == 6
    assert ws == before


def test_rr_slowest_worker_is_not_balanced_sum():
    ws = {'p':worker(100), 'fast':worker(d_rps=9), 'slow':worker(d_rps=1)}
    ws['p'].pop('decode')
    for key in ('fast', 'slow'):
        ws[key].pop('prefill')
    plan = choose(ws)['selected']
    assert plan['decode_rr_capacity_rps'] == 2
    assert plan['decode_balanced_upper_rps'] == 10


def test_missing_role_never_inherits_another_nodes_fit():
    ws = {'p':worker(), 'd':worker()}
    ws['p'].pop('decode')
    ws['d'].pop('prefill')
    plan = choose(ws)
    assert plan['selected']['prefill_workers'] == ['p']
    assert plan['excluded'][0]['missing_roles']
    ws['d'].pop('decode')
    with pytest.raises(ValueError, match='No fully measured'):
        choose(ws)


def test_common_cap_intersection_not_per_worker_max():
    ws = {'p':worker(100), 'a':worker(), 'b':worker()}
    ws['p'].pop('decode')
    for key, batches in [('a', [1,2,3]), ('b', [1,2])]:
        ws[key].pop('prefill')
        ws[key]['decode']['batch_support'] = batches
        ws[key]['max_running_requests'] = 100
    assert choose(ws)['selected']['decode_max_running_requests'] == 2


def test_tps_and_memory_bound_and_residual_denominator():
    w = worker()
    w['decode'].update(coefficients=[.01,0,0], errors={'p95_relative':.2},
                       batch_support=[1,2,3], accepted_tokens_per_step=2)
    w.update(max_running_requests=10, kv_capacity_tokens=21)
    kw = dict(output_tokens=100, mean_context=10, experts=8, top_k=1)
    points = decode_operating_points(w, **kw)
    assert set(points) == {1}
    assert points[1]['user_tps'] == pytest.approx(160)
    assert not decode_operating_points(w, **kw, min_user_tps=161)
    w['decode']['errors']['p95_relative'] = 1
    with pytest.raises(ValueError):
        decode_operating_points(w, **kw)


def test_window_split_uses_launch_intervals_not_emission_time():
    from evolving_infra.planner.calibration.trace import _capacity_windows
    rows = [dict(source='a', ts=102+i, dur_ms=2000, busy_interval_s=1,
                 forward_iter=i, mode='decode') for i in range(8)]
    w = _capacity_windows(rows, [[1] for _ in rows])[0]
    assert w['interval_start_ts'] == 100
    assert w['interval_end_ts'] == 108
    assert w['ts'] == 109
