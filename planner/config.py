"""Controller configuration: defaults, loading, and the checks made before a run starts.

Two modes:

- ``service`` (default): learning starts when the controller starts and the controller runs until it is
  stopped. An error such as an unreachable router or engine skips at most one decision and is journaled; only an
  error that leaves the controller unable to continue (listed in ``control.loop``) stops it.
- ``benchmark``: learning starts when the caller writes ``<run_dir>/replay-timing.json`` (``start_wall``,
  ``duration_s``), no role is changed after ``start_wall + duration_s``, and any error stops the controller with
  ``<run_dir>/PLANNER_FAILED`` so a measured run never continues on partial data.
"""
import json
from pathlib import Path

from .collect.trace_server import TRACE_PORT
from .common import positive
from .decide.workload import SLICE_S
from .priors import resolve_prior

MODES = ("service", "benchmark")

DEFAULTS = dict(
    mode="service", learning_s=1800, period_s=600, exclusion_s=120., update_onboarding=True,
    clock_pad_s=0.5, telemetry_period_s=3., clock_probe_period_s=300., poll_s=3., trace_port=TRACE_PORT,
    role_script=None, idle_after_cutoff=True,
)
# Keys without a default: the deployment (router, token, directories, hosts, workers) and the prefill shape prior.
REQUIRED = ("router_url", "token_file", "run_dir", "scratch", "hosts", "workers", "prefill_shape_prior")
OPTIONAL = ("prefill_shape_prior_sha256", "config_path")
# Keys of earlier versions whose values are now fixed or read from the engines.
REMOVED = dict(
    slice_s=f"workload slices are fixed at {SLICE_S:g} s",
    required_standard_errors="the decision rule always requires two standard errors",
    improvement_rule="the decision rule is always the per-minute standard-error rule",
    restore_counter_replicas="the TP size is read from every engine's /server_info",
)


def load_config(path) -> dict:
    """Read a controller config, fill ``DEFAULTS`` and record its own path (the fit subprocess rereads it)."""
    given = json.loads(Path(path).read_text())
    removed = sorted(k for k in given if k in REMOVED)
    if removed:
        raise ValueError("Remove " + "; ".join(f"{k} ({REMOVED[k]})" for k in removed))
    unknown = sorted(set(given) - set(DEFAULTS) - set(REQUIRED) - set(OPTIONAL))
    if unknown:
        raise ValueError(f"Unknown controller config keys: {unknown}")
    missing = [k for k in REQUIRED if k not in given]
    if missing:
        raise ValueError(f"Missing controller config keys: {missing}")
    cfg = dict(DEFAULTS, **given)
    cfg.setdefault("config_path", str(Path(path).resolve()))
    return cfg


def check_control_config(cfg: dict):
    """Reject a configuration before the run starts; returns (decision period, learning, sorted role script)."""
    if cfg["mode"] not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    period, learning = positive(cfg["period_s"], "decision period"), positive(cfg["learning_s"], "learning")
    resolve_prior(cfg)  # fail before the run, not at learning end
    if learning < period:
        raise ValueError("Learning must cover a full decision interval")
    if period - cfg["exclusion_s"] < 2 * SLICE_S:
        raise ValueError("Post-switch exclusion leaves fewer than two workload slices")
    script = sorted(cfg.get("role_script") or [], key=lambda x: x[0])
    if script and script[0][0] <= learning:
        raise ValueError("Scripted role changes must follow the learning-end fit")
    return period, learning, script


def learning_start_path(cfg: dict) -> Path:
    """Where the learning start is recorded: written by the caller (benchmark) or by the controller (service).
    The learning-end fit reads ``start_wall`` from it."""
    name = "replay-timing.json" if cfg["mode"] == "benchmark" else "learning-start.json"
    return Path(cfg["run_dir"]) / name
