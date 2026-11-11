"""Planner arguments: every launch parameter with its default, in the style of SGLang's ``ServerArgs``.

Each field of ``PlannerArgs`` is one command-line flag, derived from its name (``learning_secs`` ->
``--learning-secs``). ``--config FILE`` reads a JSON object of field names; a flag given on the command line overrides
the file, and the file overrides the default. ``python -m evolving_infra.planner control --help`` lists every flag.
The controller logs the resolved arguments at startup and saves them to ``<run_dir>/planner_args.json``; the
learning-end fit subprocess reads that file, so both use the same values.

Internals that are not part of the method (HTTP timeouts, clock probe exchanges) are environment variables
(``envs``). Thresholds of the fit (train fraction, minimum windows, error limits) and the D memory bound are fixed in
the modules that use them.
"""
import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Annotated as A, Any, get_type_hints

from evolving_infra.planner.observability.trace_server import TRACE_PORT
from evolving_infra.planner.perf_model.priors import PREFILL_SHAPE_PRIORS, resolve_prior

VALUE, JSON, FLAG, PRIOR = "value", "json", "flag", "prior"


@dataclasses.dataclass(frozen=True)
class Arg:
    """CLI metadata of a field: its help text and how a command-line string becomes a value."""
    help: str
    kind: str = VALUE



@dataclasses.dataclass
class PlannerArgs:
    # ------------------------------------------------------------------------------------------- deployment
    router_url: A[str, Arg("URL of the router (sgl-router), which must run with --symmetric-pd-external-role-control.")] = None
    role_control_token_file: A[str, Arg(
        "File with the router's role-control token (the router's EVOLVING_INFRA_ROLE_CONTROL_TOKEN_FILE).")] = None
    run_dir: A[str, Arg("Directory for the journals, the fit report and the resolved arguments.")] = None
    scratch_dir: A[str, Arg("Directory for trace copies and snapshots (large; node-local is fine).")] = None
    workers: A[list, Arg(
        "Workers as a JSON list of {node: 'node<idx>', idx, name (its trace subdirectory on the host), host (host "
        "index), ip, url (engine), trace_url (the host's trace server), initial_role ('prefill' or 'decode')}; the "
        "router starts with these roles.", JSON)] = None
    hosts: A[list, Arg("Worker hosts as a JSON list of {index, ip}.", JSON)] = None
    prefill_shape_prior: A[Any, Arg(
        "Prefill step-model shape for a worker whose own fit is not identifiable: a built-in name ("
        + ", ".join(sorted(PREFILL_SHAPE_PRIORS)) + ") or a JSON object with 'coefficients'.", PRIOR)] = None
    prefill_shape_prior_sha256: A[str, Arg("Optional sha256 of the prior, to pin the exact shape.")] = None

    # ------------------------------------------------------------------------------------------- learning and decisions
    learning_secs: A[float, Arg("Learning interval, from the end of the cold start to the fit.")] = 1800.
    decision_period_secs: A[float, Arg("Time between two decisions.")] = 600.
    slice_secs: A[float, Arg(
        "Slice length. Decisions compare the splits slice by slice, the cold start compares the cached share of "
        "consecutive slices, and the P step model is fitted, and the P workers re-observed, on windows of this "
        "length.")] = 60.
    switch_standard_errors: A[float, Arg(
        "A split is chosen only when its mean per-slice gain exceeds this many standard errors of that mean.")] = 2.
    post_switch_exclusion_secs: A[float, Arg(
        "Decision samples skipped after a role change has drained, so the next decision measures the new split.")] = 120.
    transition_hold_max_secs: A[float, Arg(
        "Longest time a decision waits while an earlier role change is still visible on the workers (old-role "
        "requests, an unknown decode population); a drain is always waited for.")] = 5.
    observation_gap_max_secs: A[float, Arg(
        "Largest gap between two router observations within a decision interval; a larger gap skips that "
        "decision.")] = 15.
    disable_prefill_observation: A[bool, Arg(
        "Do not re-observe the P workers' restore ratio and packing before each decision.", FLAG)] = False

    # ------------------------------------------------------------------------------------------- cold start
    cold_start_stable_change: A[float, Arg(
        "The cold start is over when the cached share of a slice is at most this above the previous slice's "
        "(0.003 is 0.3 percentage points).")] = .003
    cold_start_max_wait_secs: A[float, Arg(
        "Learning starts at the latest this long after the first completed request.")] = 1800.

    # ------------------------------------------------------------------------------------------- observation
    router_poll_interval_secs: A[float, Arg("Interval between two reads of the router's roles and completed work.")] = 3.
    restore_scrape_interval_secs: A[float, Arg(
        "Interval between two scrapes of the P workers' restored-token counters.")] = 3.
    clock_probe_interval_secs: A[float, Arg("Interval between two clock probes of every worker host.")] = 300.
    clock_tolerance_secs: A[float, Arg(
        "Largest allowed change of a host's clock offset from the one measured at start.")] = .5
    trace_port: A[int, Arg("Port of the trace server (serve-traces) on every worker host.")] = TRACE_PORT

    # ------------------------------------------------------------------------------------------- testing
    role_script: A[list, Arg(
        "Testing only: role changes instead of decisions, as a JSON list [[seconds after the learning start, "
        "{node: role}], ...], all after the fit.", JSON)] = None

    REQUIRED = ("router_url", "role_control_token_file", "run_dir", "scratch_dir", "workers", "hosts",
                "prefill_shape_prior")

    # ------------------------------------------------------------------------------------------- CLI

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--config", type=Path, default=argparse.SUPPRESS,
                            help="JSON file of planner arguments (field names); flags given here override it.")
        for field, (kind, text) in _field_specs().items():
            flag = "--" + field.replace("_", "-")
            default = PlannerArgs.__dataclass_fields__[field].default
            shown = f"{text} Default: {default}." if default is not None else f"{text} Required." \
                if field in PlannerArgs.REQUIRED else text
            if kind == FLAG:
                parser.add_argument(flag, action="store_true", default=argparse.SUPPRESS, help=text)
            else:
                parser.add_argument(flag, type=_PARSERS[kind](field), default=argparse.SUPPRESS,
                                    metavar=field.upper(), help=shown.replace("%", "%%"))

    @classmethod
    def from_cli_args(cls, namespace: argparse.Namespace) -> "PlannerArgs":
        """Defaults, then the --config file, then the flags given on the command line."""
        values = {}
        if getattr(namespace, "config", None) is not None:
            values.update(load_config_file(namespace.config))
        values.update({k: v for k, v in vars(namespace).items() if k in cls.__dataclass_fields__})
        return cls(**values)

    # ------------------------------------------------------------------------------------------- checks

    def check(self):
        """Reject arguments the controller cannot run with, before anything starts."""
        missing = [f"--{k.replace('_', '-')}" for k in self.REQUIRED if getattr(self, k) is None]
        if missing:
            raise ValueError(f"Missing planner arguments: {missing}")
        for name in ("learning_secs", "decision_period_secs", "slice_secs", "switch_standard_errors",
                     "observation_gap_max_secs", "cold_start_max_wait_secs", "router_poll_interval_secs",
                     "restore_scrape_interval_secs", "clock_probe_interval_secs"):
            _number(self, name, positive=True)
        for name in ("post_switch_exclusion_secs", "transition_hold_max_secs", "cold_start_stable_change",
                     "clock_tolerance_secs"):
            _number(self, name)
        if self.learning_secs < self.decision_period_secs:
            raise ValueError("Learning must cover a full decision interval")
        if self.decision_period_secs - self.post_switch_exclusion_secs < 2 * self.slice_secs:
            raise ValueError("Post-switch exclusion leaves fewer than two slices per decision period")
        script = self.sorted_role_script()
        if script and script[0][0] <= self.learning_secs:
            raise ValueError("Scripted role changes must follow the learning-end fit")
        resolve_prior(self.prefill_shape_prior, self.prefill_shape_prior_sha256)
        return self

    def sorted_role_script(self):
        return sorted(self.role_script or [], key=lambda x: x[0])

    # ------------------------------------------------------------------------------------------- files

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n")

    @property
    def args_path(self) -> Path:
        """The resolved arguments, written by the controller and read by the fit subprocess."""
        return Path(self.run_dir) / "planner_args.json"

    @property
    def learning_start_path(self) -> Path:
        """The learning start, written by the controller; the fit reads ``start_wall`` from it."""
        return Path(self.run_dir) / "learning-start.json"


# Keys of earlier config files, with what replaced them.
RENAMED = dict(
    learning_s="learning_secs", period_s="decision_period_secs", exclusion_s="post_switch_exclusion_secs",
    slice_s="slice_secs", required_standard_errors="switch_standard_errors", clock_pad_s="clock_tolerance_secs",
    telemetry_period_s="restore_scrape_interval_secs", clock_probe_period_s="clock_probe_interval_secs",
    poll_s="router_poll_interval_secs", token_file="role_control_token_file", scratch="scratch_dir",
    update_onboarding="disable_prefill_observation (inverted)",
)
REMOVED = dict(
    improvement_rule="the decision rule is always the per-slice standard-error rule",
    restore_counter_replicas="the TP size is read from every engine's /server_info",
    mode="there is one mode: learning starts once the cold start is over",
    idle_after_cutoff="there is no cutoff: the controller runs until it is stopped",
    config_path="pass the file with --config",
)


def load_config_file(path) -> dict:
    """A --config JSON object of PlannerArgs field names."""
    given = json.loads(Path(path).read_text())
    if not isinstance(given, dict):
        raise ValueError(f"{path}: expected a JSON object of planner arguments")
    problems = [f"{k} is now {RENAMED[k]}" for k in given if k in RENAMED]
    problems += [f"remove {k} ({REMOVED[k]})" for k in given if k in REMOVED]
    unknown = sorted(set(given) - set(PlannerArgs.__dataclass_fields__) - set(RENAMED) - set(REMOVED))
    if unknown:
        problems.append(f"unknown keys {unknown}")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return given


def _field_specs():
    hints = get_type_hints(PlannerArgs, include_extras=True)
    specs = {}
    for name in PlannerArgs.__dataclass_fields__:
        meta = next(m for m in hints[name].__metadata__ if isinstance(m, Arg))
        specs[name] = (meta.kind, meta.help)
    return specs


def _value_parser(name):
    kind = get_type_hints(PlannerArgs)[name]
    return kind if kind in (int, float, str) else str


def _json_parser(name):
    def parse(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            raise argparse.ArgumentTypeError(f"--{name.replace('_', '-')} needs JSON: {error}")
    return parse


def _prior_parser(name):
    return lambda text: json.loads(text) if text.lstrip().startswith("{") else text


_PARSERS = {VALUE: _value_parser, JSON: _json_parser, PRIOR: _prior_parser}


def _number(args, name, positive=False):
    value = getattr(args, name)
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f"--{name.replace('_', '-')} must be a finite {'positive' if positive else 'nonnegative'} "
                         f"number, got {value!r}")
