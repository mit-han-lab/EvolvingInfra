"""CPU checks of calibration evidence and scheduler-owned ratio decisions.

Fixtures describe an identifiable synthetic machine. Expected service demands
are computed from that machine, independently of the production solver.
"""

from copy import deepcopy
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest

from evolving_infra.planner.pd_ratio import calibrate, recommend


P_COEFFICIENTS = [0.03, 0.00002, 1e-10]
D_COEFFICIENTS = [0.005, 0.00005, 0.005]
OBSERVED_BATCHES = [1, 2, 4, 8, 16, 32, 64]


def engine_configuration():
    """Build a complete synthetic engine input; return a fresh JSON mapping."""
    return dict(
        model_id="synthetic-moe",
        gpu_type="synthetic-gpu",
        tp_size=1,
        chunk_size=8192,
        weight_dtype="bf16",
        kv_cache_dtype="fp8_e4m3",
        experts=256,
        top_k=8,
        draft_tokens=6,
        runtime={"revision": "fixture-v1"},
    )


def step_samples(*, restored=False, context=False):
    """Generate ordered exact timings; optional features produce full-rank fits."""
    rng = random.Random(713)
    prefill, decode = [], []
    for i in range(240):
        tokens = rng.randint(1000, 8192)
        prefix = rng.randint(0, 130000)
        pairs = tokens * prefix + tokens * (tokens + 1) / 2
        p = dict(
            uncached_tokens=tokens,
            attention_pairs=pairs,
            duration_s=0.03 + 0.00002 * tokens + 1e-10 * pairs,
        )
        if restored:
            p["restored_tokens"] = rng.randint(0, 64000)
            p["duration_s"] += 2e-6 * p["restored_tokens"]
        prefill.append(p)

        batch = OBSERVED_BATCHES[i % len(OBSERVED_BATCHES)]
        fraction = 1 - (1 - 8 / 256) ** (6 * batch)
        d = dict(
            batch_size=batch,
            duration_s=0.005 + 0.00005 * batch + 0.005 * fraction,
        )
        if context:
            # Vary mean context independently from batch so the extra column
            # identifies attention cost instead of duplicating the batch term.
            d["context_tokens"] = batch * rng.randint(40000, 120000)
            d["duration_s"] += 1e-10 * d["context_tokens"]
        decode.append(d)
    return dict(prefill=prefill, decode=decode)


def workload_summary():
    """Describe completed requests and measured packing; return solver inputs."""
    return dict(
        requests=[dict(uncached=8192, hit=70000, output=1000)] * 100,
        mean_decode_context=80000,
        packing_efficiency=0.98,
        accepted_tokens_per_step=3.0,
        mean_wait_s=8.4,
    )


def service_constraints():
    """Describe node limits and active request concurrency; return fresh limits."""
    return dict(
        concurrency=128,
        kv_capacity_tokens=2000000,
        max_running_requests=128,
        min_user_tps=100,
        safety_fraction=0.95,
    )


class PDRatioTests(unittest.TestCase):
    """Exercise public input/output contracts without a backend or live cluster."""

    @classmethod
    def setUpClass(cls):
        """Fit one exact synthetic engine; retain a reusable calibration input."""
        cls.config = engine_configuration()
        cls.model = calibrate(step_samples(), cls.config)
        if not cls.model["passed"]:
            raise AssertionError(cls.model)

    def decision(self, *, model=None, workload=None, constraints=None, config=None):
        """Recommend for nine nodes from optional overrides; return the report."""
        return recommend(
            self.model if model is None else model,
            workload_summary() if workload is None else workload,
            total_nodes=9,
            constraints=service_constraints() if constraints is None else constraints,
            engine_config=self.config if config is None else config,
        )

    def assert_rejected(self, function, *args, **kwargs):
        """Call an invalid input; accept an explicit failed report or ValueError."""
        try:
            result = function(*args, **kwargs)
        except ValueError:
            return
        self.assertFalse(result["passed"], result)
        self.assertTrue(result["reason"])
        if "selected" in result:
            self.assertIsNone(result["selected"])

    def test_calibration_recovers_known_machine_and_round_trips_json(self):
        """Fit exact varied timings; recover coefficients and persistable evidence."""
        for role, expected in [("prefill", P_COEFFICIENTS), ("decode", D_COEFFICIENTS)]:
            fit = self.model[f"{role}_fit"]
            self.assertTrue(fit["passed"])
            for actual, target in zip(fit["coefficients"], expected, strict=True):
                self.assertTrue(math.isclose(actual, target, rel_tol=1e-6, abs_tol=1e-14))
            self.assertLess(fit["errors"]["p95_relative"], 1e-8)
        self.assertEqual(self.model["observed_decode_batches"], OBSERVED_BATCHES)
        self.assertEqual(self.model["configuration"], self.config)
        self.assertEqual(len(self.model["configuration_sha256"]), 64)
        self.assertEqual(len(self.model["model_sha256"]), 64)
        restored = json.loads(json.dumps(self.model, allow_nan=False))
        self.assertEqual(self.decision(model=restored), self.decision())

    def test_calibration_preserves_observations_and_configuration(self):
        """Calibrate caller-owned data; preserve samples and detach saved config."""
        samples, config = step_samples(), engine_configuration()
        snapshot = deepcopy((samples, config))
        model = calibrate(samples, config)
        self.assertTrue(model["passed"], model)
        self.assertEqual((samples, config), snapshot)
        config["runtime"]["revision"] = "later-engine"
        self.assertEqual(model["configuration"], snapshot[1])

    def test_integer_choice_matches_independent_machine_capacity(self):
        """Recommend for a known machine; select the best physical bottleneck."""
        report = self.decision()
        self.assertTrue(report["passed"], report)
        # 95% of 2M KV tokens admits 23 mean contexts. Only 16 was observed,
        # so 23 must not become an extrapolated operating batch.
        batch = 16
        step = 0.005 + 0.00005 * batch + 0.005 * (1 - (1 - 8 / 256) ** (6 * batch))
        d_capacity = batch * 3 / step / 1000
        p_demand = 0.03 / 0.98 + 0.00002 * 8192 + 1e-10 * (
            8192 * 70000 + 8192 * 8193 / 2
        )
        independent = [min(p / p_demand, (9 - p) * d_capacity) for p in range(1, 9)]
        self.assertEqual(report["selected"]["prefill_nodes"], 1 + independent.index(max(independent)))
        self.assertAlmostEqual(report["selected"]["predicted_capacity_rps"], max(independent), places=7)
        self.assertAlmostEqual(report["prefill_service_demand_s"], p_demand, places=10)
        self.assertEqual(report["selected"]["batch"], batch)
        self.assertEqual(len(report["candidates"]), 8)
        for candidate in report["candidates"]:
            self.assertEqual(candidate["prefill_nodes"] + candidate["decode_nodes"], 9)
            self.assertAlmostEqual(candidate["predicted_capacity_rps"], min(
                candidate["prefill_capacity_rps"], candidate["decode_capacity_rps"]
            ))

    def test_changed_workload_reuses_model_and_preserves_caller_inputs(self):
        """Reuse calibration for longer completed outputs; shift capacity toward D."""
        workload, constraints = workload_summary(), service_constraints()
        snapshot = deepcopy((self.model, self.config, workload, constraints))
        initial = self.decision(workload=workload, constraints=constraints)
        self.assertEqual((self.model, self.config, workload, constraints), snapshot)
        workload["requests"] = [dict(uncached=8192, hit=70000, output=10000)] * 100
        later = self.decision(workload=workload)
        self.assertTrue(later["passed"], later)
        self.assertLess(later["selected"]["prefill_nodes"], initial["selected"]["prefill_nodes"])
        self.assertEqual(self.model, snapshot[0])
        self.assertEqual(later["mean_output_tokens"], 10000)

    def test_changed_cache_hits_and_packing_update_prefill_demand(self):
        """Change measured packing and cache demand; reuse fit with new service time."""
        initial = self.decision()
        workload = workload_summary()
        workload["packing_efficiency"] = 0.5
        partial = self.decision(workload=workload)
        self.assertGreater(partial["prefill_service_demand_s"], initial["prefill_service_demand_s"])
        workload["requests"] = [dict(uncached=1024, hit=77168, output=1000)] * 100
        cached = self.decision(workload=workload)
        self.assertLess(cached["prefill_service_demand_s"], partial["prefill_service_demand_s"])

    def test_memory_and_active_request_limits_bound_measured_batch(self):
        """Tighten memory, running requests, and concurrency; respect each bound."""
        for limits, expected_batch in [
            ({"kv_capacity_tokens": 720000}, 8),
            ({"max_running_requests": 4}, 4),
            ({"concurrency": 2}, 2),
        ]:
            with self.subTest(limits=limits):
                constraints = dict(service_constraints(), **limits)
                report = self.decision(constraints=constraints)
                self.assertTrue(report["passed"], report)
                self.assertEqual(report["selected"]["batch"], expected_batch)
                self.assertTrue(all(c["batch"] in OBSERVED_BATCHES for c in report["candidates"]))

    def test_tps_constraint_restricts_batch_or_returns_no_selection(self):
        """Raise output speed requirements; reduce batch or explicitly reject all."""
        constraints = dict(service_constraints(), min_user_tps=300)
        limited = self.decision(constraints=constraints)
        self.assertTrue(limited["passed"], limited)
        self.assertEqual(limited["selected"]["batch"], 8)
        self.assertGreaterEqual(limited["selected"]["conservative_user_tps"], 300)
        impossible = self.decision(constraints=dict(constraints, min_user_tps=100000000))
        self.assertFalse(impossible["passed"])
        self.assertIsNone(impossible["selected"])
        self.assertTrue(impossible["reason"])

    def test_idle_wait_does_not_shrink_saturated_operating_batch(self):
        """Increase external wait at fixed request concurrency; preserve capacity."""
        initial = self.decision()
        workload = dict(workload_summary(), mean_wait_s=1000)
        waiting = self.decision(workload=workload)
        for key in ["batch", "prefill_nodes", "decode_nodes", "predicted_capacity_rps"]:
            self.assertEqual(waiting["selected"][key], initial["selected"][key])
        self.assertLess(waiting["selected"]["population_ceiling_estimate_rps"],
                        initial["selected"]["population_ceiling_estimate_rps"])

    def test_no_batch_extrapolation_when_memory_and_population_are_large(self):
        """Provide ample memory and requests; retain maximum observed batch 64."""
        constraints = dict(service_constraints(), concurrency=1024,
                           max_running_requests=1024, kv_capacity_tokens=1000000000)
        report = self.decision(constraints=constraints)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["selected"]["batch"], 64)
        self.assertEqual({r["batch"] for r in report["decode_batch_candidates"]}, set(OBSERVED_BATCHES))

    def test_insufficient_or_unidentifiable_samples_do_not_produce_a_model(self):
        """Remove sample count or feature diversity; return a failed calibration."""
        for role in ["prefill", "decode"]:
            for insufficient in [True, False]:
                with self.subTest(role=role, insufficient=insufficient):
                    samples = step_samples()
                    samples[role] = samples[role][:59] if insufficient else [samples[role][0]] * 240
                    model = calibrate(samples, self.config)
                    self.assertFalse(model["passed"], model)
                    self.assertTrue(model["reason"])
                    self.assert_rejected(self.decision, model=model)

    def test_holdout_regime_change_cannot_be_recommended(self):
        """Double validation timings after training; reject each changed role."""
        for role in ["prefill", "decode"]:
            with self.subTest(role=role):
                samples = step_samples()
                for sample in samples[role][168:]:
                    sample["duration_s"] *= 2
                model = calibrate(samples, self.config)
                self.assertFalse(model["passed"], model)
                self.assertFalse(model[f"{role}_fit"]["passed"])
                self.assertGreater(model[f"{role}_fit"]["errors"]["p95_relative"], 0.3)
                self.assert_rejected(self.decision, model=model)

    def test_configuration_changes_including_extra_runtime_keys_require_recalibration(self):
        """Alter engine settings or extra JSON metadata; reject mismatched evidence."""
        for change in [dict(chunk_size=4096), dict(gpu_type="other-gpu"),
                       dict(runtime={"revision": "fixture-v2"})]:
            with self.subTest(change=change):
                self.assert_rejected(self.decision, config=dict(self.config, **change))

    def test_modified_calibration_evidence_is_rejected(self):
        """Alter coefficients, observed support, or saved config; reject corruption."""
        for field in ["coefficients", "observed_batches", "configuration", "checksum"]:
            with self.subTest(field=field):
                model = deepcopy(self.model)
                if field == "coefficients":
                    model["decode_fit"]["coefficients"][0] /= 2
                elif field == "observed_batches":
                    model["observed_decode_batches"].append(128)
                elif field == "configuration":
                    model["configuration"]["runtime"]["revision"] = "corrupted"
                else:
                    model["model_sha256"] = "0" * 64
                self.assert_rejected(self.decision, model=model)

    def test_decode_context_feature_uses_total_batch_context(self):
        """Fit variable total contexts; recover attention cost and use mean context."""
        model = calibrate(step_samples(context=True), self.config)
        self.assertTrue(model["passed"], model)
        for actual, expected in zip(model["decode_fit"]["coefficients"],
                                    D_COEFFICIENTS + [1e-10], strict=True):
            self.assertTrue(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-14))
        report = self.decision(model=model)
        self.assertTrue(report["passed"], report)
        for op in report["decode_batch_candidates"]:
            batch = op["batch"]
            seconds = (0.005 + 0.00005 * batch + 0.005 * (1 - (1 - 8 / 256) ** (6 * batch))
                       + 1e-10 * batch * 80000)
            self.assertAlmostEqual(op["step_s"], seconds, places=10)

    def test_long_context_cannot_reuse_short_context_calibration_at_small_batch(self):
        """Supply unseen per-request context within total support; refuse extrapolation."""
        model = calibrate(step_samples(context=True), self.config)
        self.assertTrue(model["passed"], model)
        workload = dict(workload_summary(), mean_decode_context=1000000)
        # A batch of one million-token requests lies within measured total
        # contexts for larger batches, but no measured request was this long.
        self.assertLess(model["decode_fit"]["feature_min"][3], 1000000)
        self.assertGreater(model["decode_fit"]["feature_max"][3], 1000000)
        constraints = dict(service_constraints(), kv_capacity_tokens=100000000)
        self.assert_rejected(self.decision, model=model, workload=workload,
                             constraints=constraints)

    def test_impossible_speculative_acceptance_cannot_inflate_decode_capacity(self):
        """Exceed available draft and bonus tokens per step; reject false capacity."""
        workload = dict(workload_summary(), accepted_tokens_per_step=1000000)
        self.assert_rejected(self.decision, workload=workload)

    def test_dense_model_identifies_intercept_without_duplicate_expert_column(self):
        """Fit a dense machine with constant expert cost; retain usable predictions."""
        config = dict(engine_configuration(), experts=1, top_k=1)
        samples = step_samples()
        for row in samples["decode"]:
            row["duration_s"] = 0.010 + 0.00005 * row["batch_size"]
        model = calibrate(samples, config)
        self.assertTrue(model["passed"], model)
        self.assertEqual(len(model["decode_fit"]["coefficients"]), 3)
        intercept, slope, expert = model["decode_fit"]["coefficients"]
        self.assertAlmostEqual(intercept + expert, 0.010)
        self.assertAlmostEqual(slope, 0.00005)
        report = self.decision(model=model, config=config)
        self.assertTrue(report["passed"], report)
        for op in report["decode_batch_candidates"]:
            self.assertAlmostEqual(op["step_s"], 0.010 + 0.00005 * op["batch"])

    def test_restoration_cost_requires_measured_request_features(self):
        """Fit completed restore work; require aggregate features and charge once."""
        model = calibrate(step_samples(restored=True), self.config)
        self.assertTrue(model["passed"], model)
        for actual, expected in zip(model["prefill_fit"]["coefficients"],
                                    P_COEFFICIENTS + [2e-6], strict=True):
            self.assertTrue(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-14))
        self.assert_rejected(self.decision, model=model)
        workload = workload_summary()
        workload["prefill_features_per_request"] = [1, 8192, 8192 * 70000 + 8192 * 8193 / 2, 1000]
        report = self.decision(model=model, workload=workload)
        self.assertTrue(report["passed"], report)
        expected = sum(a * b for a, b in zip(workload["prefill_features_per_request"],
                                           P_COEFFICIENTS + [2e-6], strict=True))
        self.assertAlmostEqual(report["prefill_service_demand_s"], expected, places=10)

    def test_partial_optional_features_cannot_silently_mean_zero(self):
        """Omit one measured optional feature; reject ambiguous missing evidence."""
        for role, field, options in [
            ("prefill", "restored_tokens", dict(restored=True)),
            ("decode", "context_tokens", dict(context=True)),
        ]:
            with self.subTest(role=role):
                samples = step_samples(**options)
                del samples[role][0][field]
                self.assert_rejected(calibrate, samples, self.config)

    def test_invalid_workload_does_not_return_a_selection(self):
        """Pass empty, nonfinite, or nonpositive measurements; reject each input."""
        for change in [dict(requests=[]), dict(mean_decode_context=float("nan")),
                       dict(packing_efficiency=0), dict(accepted_tokens_per_step=0),
                       dict(requests=[dict(uncached=8192, hit=70000, output=0)]),
                       dict(requests=[dict(uncached=-1, hit=70000, output=1000)])]:
            with self.subTest(change=change):
                self.assert_rejected(self.decision, workload=dict(workload_summary(), **change))


class PDRatioImportTests(unittest.TestCase):
    """Check API imports independently from a process that already fitted a model."""

    def test_plain_import_needs_no_optional_numerical_dependencies(self):
        """Import module and scheduler separately; leave numerical packages unloaded."""
        cases = [
            ([], ["from evolving_infra.planner.pd_ratio import calibrate, recommend"]),
            ([], ["from evolving_infra.router.client.scheduler import Scheduler",
                  "scheduler = Scheduler('http://127.0.0.1:1')"]),
        ]
        for interpreter_options, imports in cases:
            with self.subTest(interpreter_options=interpreter_options):
                subprocess.run(
                    [sys.executable, *interpreter_options, "-c", "\n".join([
                        "import sys", *imports,
                        "assert not {'numpy', 'scipy', 'evolving_infra.planner.calibration'} & sys.modules.keys()",
                    ])],
                    cwd=Path(__file__).resolve().parents[2],
                    check=True,
                    capture_output=True,
                    text=True,
                )


if __name__ == "__main__":
    unittest.main()
