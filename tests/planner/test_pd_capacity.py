import json

import numpy as np
import pytest

from evolving_infra.planner.calibration.capacity import fit_capacity_totals, join_cache_tiers
from evolving_infra.planner.calibration.model import prefill_request_seconds


def test_cache_preparation_is_charged_once_despite_multiple_chunks(tmp_path):
    request_path = tmp_path / "requests.jsonl"
    request = dict(prefill_iters=[1, 2], cached_tokens=150, cached_tokens_host=50,
                   cached_tokens_device=100, retraction_count=0)
    request_path.write_text(json.dumps(request)+"\n")
    rows = [dict(source=str(tmp_path / "steps.jsonl"), forward_iter=i, launch_bs=1,
                 new_prefill_tokens=10, prefill_attention_pairs=2000) for i in (1, 2)]
    features = join_cache_tiers(rows, [request_path])
    assert [f[3:] for f in features] == [[50, 100], [0, 0]]
    rows[1]["launch_bs"] = 2
    with pytest.raises(ValueError, match="Incomplete"):
        join_cache_tiers(rows, [request_path])
    rows[1]["launch_bs"] = 1
    request["cached_tokens_host"] = 200
    request_path.write_text(json.dumps(request)+"\n")
    with pytest.raises(ValueError, match="do not cover"):
        join_cache_tiers(rows, [request_path])


def test_capacity_sample_conserves_service_and_excludes_split_boundary():
    rng = np.random.default_rng(234)
    co = np.array([.03, 1e-5, 1e-10, 2e-6, 2e-7])
    episodes = []
    for minute in range(30):
        for node in range(3):
            steps = int(rng.integers(1, 10))
            features = [steps, int(rng.integers(100, 50000)), int(rng.integers(1e5, 1e9)),
                        int(rng.integers(0, 1e6)), int(rng.integers(0, 2e6))]
            episodes.append(dict(source=str(node), ts=minute*60+45,
                                 busy_interval_s=float(np.dot(features, co)),
                                 features=features, step_count=steps))
    # A complete period spanning the 1260s split must never enter validation.
    episodes.append(dict(source="0", ts=1270., busy_interval_s=20.,
                         features=[2, 100, 100000, 0, 0], step_count=2))
    fit = fit_capacity_totals(episodes, start_unix=0, end_unix=1800,
                             eligible_steps=sum(e["step_count"] for e in episodes))
    assert fit["passed"]
    assert fit["excluded_boundary_groups"] == 1
    assert fit["train_samples"] == 63
    assert fit["coefficients"] == pytest.approx(co)
    assert all(g["start_ts"] >= 1260 for g in fit["groups"][63:])
    retained = [e for e in episodes if not (e["source"] == "0" and 1260 <= e["ts"] < 1320)]
    assert sum(g["seconds"] for g in fit["groups"]) == pytest.approx(sum(e["busy_interval_s"] for e in retained))


def test_request_capacity_requires_complete_tier_accounting():
    co = [0, 0, 0, .001, .002]
    assert prefill_request_seconds(10, 150, 8192, co, host_hit=50, device_hit=100) == pytest.approx(.25)
    with pytest.raises(ValueError, match="requires measured"):
        prefill_request_seconds(10, 150, 8192, co)
    with pytest.raises(ValueError, match="complete cached"):
        prefill_request_seconds(10, 150, 8192, co, host_hit=20, device_hit=100)
