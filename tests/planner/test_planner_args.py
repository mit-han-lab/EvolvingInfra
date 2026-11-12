# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Planner arguments, prefill shape priors and the command line."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import pytest

from evolving_infra.planner import planner_args
from evolving_infra.planner.perf_model import priors
from evolving_infra.planner.perf_model.priors import PREFILL_SHAPE_PRIORS
from evolving_infra.planner.planner_args import PlannerArgs

DEPLOYMENT = dict(router_url="http://router", role_control_token_file="/secure/token", run_dir="/run", scratch_dir="/scratch",
                  hosts=[], workers=[], prefill_shape_prior="glm52-gb300-tp4")


def test_prior_is_the_median_of_measured_four_term_models():
    def worker(origin, coefficients, method="per_node_four_term_nnls"):
        return dict(prefill=dict(origin=origin, models=[dict(method=method, coefficients=coefficients)]))
    report = dict(workers=dict(node0=worker("measured", [1, 10, 100, 1000]), node1=worker("measured", [3, 30, 300, 3000]),
                               node2=worker("measured", [2, 20, 200, 2000]),
                               node3=worker("same_run_role_prior", [9, 9, 9, 9]),
                               node4=worker("measured", [7, 7, 7, 7], "prior_shape_current_node_scale_v1")))
    assert priors.build_prior(report) == [2, 20, 200, 2000]
    for prior in PREFILL_SHAPE_PRIORS.values():
        assert len(prior["coefficients"]) in (4, 5) and all(c >= 0 for c in prior["coefficients"])
        assert priors.prior_sha256(prior) == priors.prior_sha256(json.loads(json.dumps(prior)))


def test_prior_of_vision_models_is_five_term_and_schemas_do_not_mix():
    def worker(coefficients, method):
        return dict(prefill=dict(origin="measured", models=[dict(method=method, coefficients=coefficients)]))
    five = "per_node_five_term_nnls"
    report = dict(workers=dict(node0=worker([1, 0, 3, 0, 5], five), node1=worker([0, 2, 4, 6, 8], five),
                               node2=worker([2, 0, 5, 1, 1], five)))
    assert priors.build_prior(report) == [1, 0, 4, 1, 5]
    report["workers"]["node3"] = worker([1, 1, 1, 1], "per_node_four_term_nnls")
    with pytest.raises(ValueError, match="mix"):
        priors.build_prior(report)


def test_prior_resolution_requires_a_prior_and_rejects_unknown_changed_or_malformed_ones():
    with pytest.raises(ValueError, match="required"):
        priors.resolve_prior(None)
    for name in ("dsv4-flash-h200-tp4", "glm52-gb300-tp4", "kimi-k26-gb300-tp4"):
        assert priors.resolve_prior(name) is PREFILL_SHAPE_PRIORS[name]
    inline = dict(coefficients=[0.1, 4e-5, 1.5e-10, 4e-7])
    assert priors.resolve_prior(inline) == inline
    assert priors.resolve_prior(inline, priors.prior_sha256(inline))
    with pytest.raises(ValueError, match="changed"):
        priors.resolve_prior(inline, "0" * 64)
    with pytest.raises(ValueError, match="Unknown"):
        priors.resolve_prior("nope")
    with pytest.raises(ValueError, match="coefficients"):
        priors.resolve_prior(dict(coefficients=[1, 2]))


def parse(argv):
    """PlannerArgs from a `control` command line, as the CLI builds them."""
    parser = argparse.ArgumentParser()
    PlannerArgs.add_cli_args(parser)
    return PlannerArgs.from_cli_args(parser.parse_args(argv))


def test_defaults_match_the_paper_settings_and_the_one_slice_parameter():
    args = PlannerArgs()
    assert (args.learning_secs, args.decision_period_secs, args.slice_secs, args.switch_standard_errors,
            args.post_switch_exclusion_secs) == (1800, 600, 60, 2, 120)
    assert (args.transition_hold_max_secs, args.observation_gap_max_secs, args.cold_start_stable_change,
            args.cold_start_max_wait_secs) == (5, 15, .003, 1800)
    assert not args.disable_prefill_observation and args.trace_port == 18500


def test_flags_override_the_config_file_which_overrides_the_defaults(tmp_path):
    path = tmp_path / "planner.json"
    path.write_text(json.dumps(dict(DEPLOYMENT, decision_period_secs=300, slice_secs=30)))
    args = parse(["--config", str(path), "--slice-secs", "20", "--disable-prefill-observation",
                  "--workers", '[{"node": "node0"}]', "--prefill-shape-prior", '{"coefficients": [1, 2, 3, 4]}'])
    assert (args.decision_period_secs, args.slice_secs, args.learning_secs) == (300, 20, 1800)
    assert args.disable_prefill_observation and args.workers == [{"node": "node0"}]
    assert args.prefill_shape_prior == {"coefficients": [1, 2, 3, 4]}
    assert parse(["--prefill-shape-prior", "glm52-gb300-tp4"]).prefill_shape_prior == "glm52-gb300-tp4"


def test_config_file_rejects_renamed_removed_and_unknown_keys(tmp_path):
    path = tmp_path / "planner.json"
    for given, match in ((dict(DEPLOYMENT, perod_s=300), "unknown keys.*perod_s"),
                         (dict(DEPLOYMENT, learning_s=1800), "learning_s is now learning_secs"),
                         (dict(DEPLOYMENT, token_file="/t"), "token_file is now role_control_token_file"),
                         (dict(DEPLOYMENT, restore_counter_replicas=4), "TP size"),
                         (dict(DEPLOYMENT, mode="benchmark"), "one mode"),
                         (dict(DEPLOYMENT, idle_after_cutoff=True), "no cutoff")):
        path.write_text(json.dumps(given))
        with pytest.raises(ValueError, match=match):
            planner_args.load_config_file(path)


def test_checks_before_the_run_and_the_files_the_fit_reads():
    args = PlannerArgs(**DEPLOYMENT).check()
    assert args.learning_start_path == Path("/run/learning-start.json")
    assert args.args_path == Path("/run/planner_args.json")
    with pytest.raises(ValueError, match="Missing.*--prefill-shape-prior"):
        PlannerArgs(**dict(DEPLOYMENT, prefill_shape_prior=None)).check()
    with pytest.raises(ValueError, match="follow the learning-end fit"):
        PlannerArgs(**DEPLOYMENT, role_script=[[1800, {"node0": "decode"}]]).check()
    with pytest.raises(ValueError, match="two slices"):
        PlannerArgs(**DEPLOYMENT, decision_period_secs=200).check()
    with pytest.raises(ValueError, match="--slice-secs must be a finite positive"):
        PlannerArgs(**DEPLOYMENT, slice_secs=0).check()


def test_cli_help_lists_every_command():
    out = subprocess.run([sys.executable, "-m", "evolving_infra.planner", "--help"], check=True,
                         capture_output=True, text=True).stdout
    for command in ("control", "fit", "serve-traces", "prior"):
        assert command in out


def test_cli_parser_and_trace_server_load_no_numerical_library():
    """The command-line parser and the per-host trace server import no NumPy/SciPy."""
    subprocess.run([sys.executable, "-c", "\n".join([
        "import sys",
        "from evolving_infra.planner.__main__ import build_parser",
        "from evolving_infra.planner.observability.trace_server import serve_traces",
        "build_parser()",
        "assert not {'numpy', 'scipy'} & sys.modules.keys()",
    ])], check=True, capture_output=True)
