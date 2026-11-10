import json

import numpy as np
import pytest

from evolving_infra.planner.calibration.model import (
    expert_fraction,
    fit_steps,
    freeze,
    load_frozen,
    prefill_request_seconds,
    select_ratio,
)
from evolving_infra.planner.calibration.trace import load_steps
from evolving_infra.planner.calibration.deployment import apply_frozen_ratio


def test_causal_attention_work_is_invariant_under_chunking():
    coefficients = [0, 0.00002, 1e-10]
    hit, total = 17000, 23456
    whole = prefill_request_seconds(total, hit, 8192, coefficients)
    pieces, remaining, prefix = 0.0, total, hit
    while remaining:
        n = min(8192, remaining)
        pieces += prefill_request_seconds(n, prefix, 8192, coefficients)
        remaining -= n
        prefix += n
    assert pieces == pytest.approx(whole)


def test_packed_requests_share_fixed_step_cost():
    coefficients = [0.04, 0.00002, 1e-10]
    pieces = [
        prefill_request_seconds(n, h, 8192, coefficients)
        for n, h in [(2048, 15000), (6144, 30000)]
    ]
    pairs = sum(n * h + n * (n + 1) / 2 for n, h in [(2048, 15000), (6144, 30000)])
    assert sum(pieces) == pytest.approx(0.04 + 0.00002 * 8192 + 1e-10 * pairs)


def test_fit_recovers_coefficients_and_rejects_changed_regime():
    rng = np.random.default_rng(8)
    n = rng.integers(1000, 8193, 400)
    h = rng.integers(0, 130000, 400)
    x = np.column_stack([np.ones(400), n, n * h + n * (n + 1) / 2])
    expected = [0.03, 0.00002, 1e-10]
    y = x @ expected
    good = fit_steps(x, y)
    assert good["passed"]
    assert good["coefficients"] == pytest.approx(expected, rel=1e-6)
    y[280:] *= 2
    assert not fit_steps(x, y)["passed"]
    with pytest.raises(ValueError, match="not identifiable"):
        fit_steps(np.ones((100, 3)), np.ones(100))


def test_interlaunch_interval_belongs_to_previous_step_and_excludes_idle(tmp_path):
    rows = [
        dict(ts=100 + i, forward_iter=i, launch_mono=t, mode="extend", after_idle=idle)
        for i, t, idle in [
            (1, 10, False),
            (2, 10.5, False),
            (3, 40, True),
            (5, 40.2, False),
        ]
    ]
    path = tmp_path / "steps.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"partial":')
    loaded = load_steps([path], 100, 110)
    assert [r["busy_interval_s"] for r in loaded] == [0.5, None, None, None]


def selection_args():
    p = dict(passed=True, coefficients=[0.03, 0.00002, 1e-10])
    d = dict(
        passed=True, coefficients=[0.005, 0.00005, 0.005], errors={"p95_relative": 0.03}
    )
    return dict(
        nodes=9,
        concurrency=128,
        chunk=8192,
        prefill_fit=p,
        decode_fit=d,
        prompt_samples=[{"uncached": 8192, "hit": 70000, "output": 1000}] * 100,
        mean_decode_context=80000,
        kv_capacity_tokens=2000000,
        max_running_requests=128,
        accepted_tokens_per_step=3.0,
        observed_decode_batches=list(range(1, 65)),
        experts=256,
        top_k=8,
        draft_tokens=6,
        mean_wait_s=8.4,
        packing_efficiency=0.98,
    )


def test_ratio_capacity_memory_and_tps_constraints():
    args = selection_args()
    report = select_ratio(**args)
    chosen = report["selected"]
    assert chosen["prefill_nodes"] + chosen["decode_nodes"] == 9
    assert chosen["conservative_user_tps"] >= 100
    assert chosen["batch"] <= 23  # 95% of 2M / 80k
    assert chosen["batch"] == max(r["batch"] for r in report["decode_batch_candidates"])
    assert len(report["candidates"]) == 8
    assert chosen["predicted_capacity_rps"] == max(
        r["predicted_capacity_rps"] for r in report["candidates"]
    )
    assert all(
        r["batch"] in args["observed_decode_batches"] for r in report["candidates"]
    )
    args["min_user_tps"] = 1e8
    with pytest.raises(ValueError, match="No observed"):
        select_ratio(**args)


def test_long_tool_wait_does_not_turn_average_decode_population_into_a_cap():
    args = selection_args()
    initial = select_ratio(**args)
    args["mean_wait_s"] = 1000
    waiting = select_ratio(**args)
    for key in ["batch", "prefill_nodes", "decode_nodes", "predicted_capacity_rps"]:
        assert waiting["selected"][key] == initial["selected"][key]
    assert (
        waiting["selected"]["population_ceiling_estimate_rps"]
        < initial["selected"]["population_ceiling_estimate_rps"]
    )


def test_decode_margin_covers_underprediction_at_measured_relative_error_bound():
    args = selection_args()
    # Predicted 16ms versus observed 20ms has a truth-normalized error of 20%.
    # Multiplying by 1.2 gives only 19.2ms and incorrectly admits 102 TPS.
    args["decode_fit"].update(coefficients=[0.016, 0, 0], errors={"p95_relative": 0.2})
    args["accepted_tokens_per_step"] = 2
    report = select_ratio(**args)
    assert report["selected"]["conservative_step_s"] == pytest.approx(0.020)
    assert report["selected"]["conservative_user_tps"] == pytest.approx(100)
    args["min_user_tps"] = 102
    with pytest.raises(ValueError, match="No observed"):
        select_ratio(**args)


@pytest.mark.parametrize("residual", [-0.1, 1, float("nan"), float("inf")])
def test_invalid_decode_residual_has_no_finite_duration_bound(residual):
    args = selection_args()
    args["decode_fit"]["errors"]["p95_relative"] = residual
    with pytest.raises(ValueError, match="relative-error bound"):
        select_ratio(**args)


def test_aggregate_capacity_ratio_must_be_stable_across_error_envelope():
    args = selection_args()
    args["prefill_fit"] = dict(passed=True, coefficients=[.2, 0, 0, 0],
        aggregate_request_features=[1, 8192, 1e8, 1000],
        feature_schema=["steps", "new_tokens", "attention_pairs", "completed_restore_tokens"],
        errors={"p95_relative": .01})
    chosen = select_ratio(**args)
    assert chosen["prefill_service_demand_s"] == pytest.approx(.2)
    assert chosen["prefill_error_sensitivity"]["prefill_nodes_at_bounds"] == [
        chosen["selected"]["prefill_nodes"]] * 2
    args["prefill_fit"]["errors"]["p95_relative"] = .99
    with pytest.raises(ValueError, match="choice changes"):
        select_ratio(**args)


def test_frozen_decision_cannot_be_overwritten_or_reused_for_other_settings(tmp_path):
    config = {"model": "glm", "chunk": 8192}
    report = {"elapsed_s": 900, "decision": select_ratio(**selection_args())}
    path = tmp_path / "frozen.json"
    freeze(path, report, config)
    assert load_frozen(path, config)["state"] == "frozen"
    with pytest.raises(FileExistsError):
        freeze(path, report, config)
    with pytest.raises(ValueError, match="another experiment"):
        load_frozen(path, {"chunk": 16384})
    altered = json.loads(path.read_text())
    altered["report"]["decision"]["selected"]["prefill_nodes"] = 8
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="checksum"):
        load_frozen(path, config)
    with pytest.raises(ValueError, match="900-second"):
        freeze(tmp_path / "early.json", dict(report, elapsed_s=899), config)


def test_deployment_uses_same_frozen_roles_and_decode_cap_for_both_arms(tmp_path):
    config = dict(
        chunk=8192,
        kv_dtype="fp8_e4m3",
        hicache_size_gb=500,
        host_copies=1,
        model_config_sha256="config",
        model_index_sha256="weights",
    )
    report = dict(elapsed_s=900, decision=select_ratio(**selection_args()))
    path = tmp_path / "frozen.json"
    freeze(path, report, config)
    template = dict(
        nodes=[dict(index=i) for i in range(9)],
        chunked_prefill_size=8192,
        kv_cache_dtype="fp8_e4m3",
        hicache_size_gb=500,
        host_copies=1,
        model_config_sha256="config",
        model_index_sha256="weights",
    )
    baseline = apply_frozen_ratio(dict(template, arm="baseline"), path, config)
    p2p = apply_frozen_ratio(dict(template, arm="p2p"), path, config)
    for key in [
        "prefill_nodes",
        "decode_nodes",
        "decode_max_running_requests",
        "nodes",
    ]:
        assert baseline[key] == p2p[key]
    assert "role" not in template["nodes"][0]
    assert (
        baseline["decode_max_running_requests"]
        == report["decision"]["selected"]["batch"]
    )
    with pytest.raises(ValueError, match="settings differ"):
        apply_frozen_ratio(dict(template, kv_cache_dtype="bf16"), path, config)
    extended = apply_frozen_ratio(template, path, config, duration_s=5400)
    assert (extended["duration_s"], extended["warmup_excluded_s"], extended["steady_s"]) == (5400, 900, 4500)
    assert extended["frozen_ratio_sha256"] == baseline["frozen_ratio_sha256"]
    with pytest.raises(ValueError, match="warmup_excluded_s"):
        apply_frozen_ratio(template, path, config, duration_s=900)


def test_partial_steps_preserve_total_fixed_overhead():
    # Three partially filled steps process 9K tokens in total, shared by four
    # requests. Using the configured 8K limit would undercount their overhead.
    step_tokens = [2048, 4096, 3072]
    mean_step = sum(step_tokens) / len(step_tokens)
    attributed = sum(
        prefill_request_seconds(n, 0, mean_step, [0.04, 0, 0])
        for n in [1024, 2048, 2048, 4096]
    )
    assert attributed == pytest.approx(0.04 * len(step_tokens))
    args = selection_args()
    args["packing_efficiency"] = 0.5
    partial = select_ratio(**args)
    args["packing_efficiency"] = 1
    full = select_ratio(**args)
    assert partial["prefill_service_demand_s"] > full["prefill_service_demand_s"]
    assert partial["mean_tokens_per_step"] == 4096


def test_invalid_decision_cannot_be_frozen(tmp_path):
    with pytest.raises(ValueError, match="No validated"):
        freeze(tmp_path / "invalid.json", {"elapsed_s": 900, "decision": None}, {})


def test_capacity_windows_remove_alternating_overlap_launch_jitter():
    from evolving_infra.planner.calibration.trace import fit_role

    rows = []
    rng = np.random.default_rng(16)
    for window in range(100):
        b = int(rng.integers(1, 32))
        t = 0.005 + 0.0003 * b + 0.004 * float(expert_fraction(6 * b, 256, 8))
        for j in range(8):
            i = window * 8 + j
            rows.append(
                dict(
                    ts=i,
                    source="worker0",
                    forward_iter=i,
                    mode="decode",
                    launch_bs=b,
                    launch_seq_lens_sum=80000 * b,
                    spec_num_draft_tokens=6,
                    forward_gpu_ms=t * 1000,
                    busy_interval_s=t * (0.1 if j % 2 == 0 else 1.9),
                )
            )
    result = fit_role(rows, "decode", 256, 8)["time_models"]
    assert result["gpu"]["passed"]
    assert not result["busy_wall"]["passed"]
    assert result["busy_capacity"]["passed"]
    assert result["busy_capacity"]["coefficients"] == pytest.approx(
        [0.005, 0.0003, 0.004]
    )
    assert result["busy_capacity"]["step_coverage"] == 1
    assert result["busy_capacity"]["validation_unit"] == "8-step total"


def test_short_prefill_busy_periods_include_final_step_and_exclude_idle():
    from evolving_infra.planner.calibration.trace import fit_role

    rng = np.random.default_rng(22)
    rows = []
    clock = 0.0
    coefficients = [0.03, 0.00002, 1e-10]
    for episode in range(101):
        elapsed = 0.0
        for i in range(1 + episode % 4):
            n, h = int(rng.integers(10, 8193)), int(rng.integers(0, 100000))
            pairs = n * h + n * (n + 1) / 2
            seconds = float(np.dot([1, n, pairs], coefficients))
            launch = clock + elapsed - (0.01 if i else 0)
            elapsed += seconds
            rows.append(
                dict(
                    ts=clock + elapsed,
                    source="P0",
                    forward_iter=len(rows),
                    mode="extend",
                    after_idle=i == 0,
                    launch_bs=1,
                    launch_mono=launch,
                    new_prefill_tokens=n,
                    prefill_attention_pairs=pairs,
                    spec_num_draft_tokens=None,
                    forward_gpu_ms=seconds * 1000,
                    dur_ms=(clock + elapsed - launch) * 1000,
                    busy_interval_s=None,
                )
            )
        clock += elapsed + 5  # Idle time must never become service demand.
    models = fit_role(rows, "prefill", 256, 8)["time_models"]
    assert not models["busy_capacity"]["passed"]
    measured = models["busy_episodes"]
    assert measured["passed"]
    assert measured["samples"] == 100  # The final unclosed period is excluded.
    assert measured["step_coverage"] > 0.99
    assert measured["coefficients"] == pytest.approx(coefficients)


def test_freeze_respects_explicit_learning_duration(tmp_path):
    report = {"elapsed_s": 900, "decision": {"selected": {"prefill_nodes": 2, "decode_nodes": 7}}}
    config = {"learning_duration_s": 1800}
    with pytest.raises(ValueError, match="1800-second"):
        freeze(tmp_path / "too-early.json", report, config)
    report["elapsed_s"] = 1800
    freeze(tmp_path / "complete.json", report, config)
    assert load_frozen(tmp_path / "complete.json", config)["report"]["elapsed_s"] == 1800
