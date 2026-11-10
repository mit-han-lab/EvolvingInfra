"""Controller configuration, prefill shape priors and the command line."""
import json
import subprocess
import sys

import pytest

from evolving_infra.planner import config, priors
from evolving_infra.planner.priors import PREFILL_SHAPE_PRIORS

DEPLOYMENT = dict(router_url="http://router", token_file="/secure/token", run_dir="/run", scratch="/scratch",
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
    cfg = dict(config.DEFAULTS)
    with pytest.raises(ValueError, match="required"):
        priors.resolve_prior(cfg)
    for name in ("dsv4-flash-h200-tp4", "glm52-gb300-tp4", "kimi-k26-gb300-tp4"):
        assert priors.resolve_prior(dict(cfg, prefill_shape_prior=name)) is PREFILL_SHAPE_PRIORS[name]
    inline = dict(coefficients=[0.1, 4e-5, 1.5e-10, 4e-7])
    assert priors.resolve_prior(dict(cfg, prefill_shape_prior=inline)) == inline
    assert priors.resolve_prior(dict(cfg, prefill_shape_prior=inline, prefill_shape_prior_sha256=priors.prior_sha256(inline)))
    with pytest.raises(ValueError, match="changed"):
        priors.resolve_prior(dict(cfg, prefill_shape_prior=inline, prefill_shape_prior_sha256="0" * 64))
    with pytest.raises(ValueError, match="Unknown"):
        priors.resolve_prior(dict(cfg, prefill_shape_prior="nope"))
    with pytest.raises(ValueError, match="coefficients"):
        priors.resolve_prior(dict(cfg, prefill_shape_prior=dict(coefficients=[1, 2])))


def test_load_config_fills_defaults_and_records_its_path(tmp_path):
    path = tmp_path / "controller.json"
    path.write_text(json.dumps(dict(DEPLOYMENT, period_s=300)))
    cfg = config.load_config(path)
    assert cfg["period_s"] == 300 and cfg["learning_s"] == config.DEFAULTS["learning_s"]
    assert cfg["mode"] == "service" and cfg["config_path"] == str(path.resolve())


def test_load_config_rejects_missing_unknown_and_removed_keys(tmp_path):
    path = tmp_path / "controller.json"
    for given, match in ((dict(DEPLOYMENT, period_s=300, perod_s=300), "Unknown.*perod_s"),
                         ({k: v for k, v in DEPLOYMENT.items() if k != "prefill_shape_prior"},
                          "Missing.*prefill_shape_prior"),
                         (dict(DEPLOYMENT, slice_s=60.), "slice_s"),
                         (dict(DEPLOYMENT, restore_counter_replicas=4), "TP size")):
        path.write_text(json.dumps(given))
        with pytest.raises(ValueError, match=match):
            config.load_config(path)


def test_control_checks_mode_and_learning_start_file():
    cfg = dict(config.DEFAULTS, **DEPLOYMENT)
    assert config.check_control_config(cfg)[:2] == (600, 1800)
    assert config.learning_start_path(cfg).name == "learning-start.json"
    assert config.learning_start_path(dict(cfg, mode="benchmark")).name == "replay-timing.json"
    with pytest.raises(ValueError, match="mode"):
        config.check_control_config(dict(cfg, mode="replay"))
    with pytest.raises(ValueError, match="two workload slices"):
        config.check_control_config(dict(cfg, period_s=200, exclusion_s=120.))


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
        "from evolving_infra.planner.collect.trace_server import serve_traces",
        "build_parser()",
        "assert not {'numpy', 'scipy'} & sys.modules.keys()",
    ])], check=True, capture_output=True)
