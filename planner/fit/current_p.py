"""Observe the restore ratio and packing of the current P workers before each decision; every fit stays frozen.

The input is the closed 60-second P busy windows of the decision interval. The seconds per restored token is a
coefficient of the learning-end fit, like the compute terms, and is never re-estimated online. What changes with
cache state and placement is the share of cached tokens that must be restored from host memory. Packing (new
tokens per forward step) is likewise a workload property, not a hardware one: short uncached suffixes still take a
forward each, so the learning average undercounts steps once prefixes hit. The seconds per step stays frozen.
Configs and journals call this observation "onboarding" (``update_onboarding``, ``onboarding_updated``).
"""
import hashlib
import json
import math
from pathlib import Path

from ..collect.clock import Clock
from ..collect.restore_counters import counter_replicas
from ..collect.trace_server import Mirror
from ..common import save
from .learning import capture_interval, snapshot_measurements


def _number(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError('Invalid ' + name)
    return value


def estimate_onboarding(windows):
    """Restored per cached token over one node's latest closed windows.

    Aggregate before dividing; zero restores is a valid observation. Counter
    bounds come from the existing HTTP collection/clock uncertainty audit. They
    are measurement bounds, not statistical confidence intervals. Packing is
    new tokens over forward steps of the same windows; it is left out when the
    windows computed no new tokens, so the warmup packing stays in use.
    """
    if len(windows) < 2:
        raise ValueError('Need at least two closed windows')
    rows = sorted(windows, key=lambda w: w['start'])
    if len({w['node'] for w in rows}) != 1:
        raise ValueError('Onboarding windows must belong to one physical node')
    restored = lower = upper = cached = steps = tokens = 0.
    previous_end = None
    for row in rows:
        start, end = (_number(row[k], k) for k in ('start', 'end'))
        if (not math.isclose(end-start, 60., rel_tol=0., abs_tol=1e-6)
                or (previous_end is not None and start < previous_end)):
            raise ValueError('Expected nonoverlapping closed 60s P windows')
        previous_end = end
        value = _number(row['features'][3], 'restored tokens')
        lo, hi = (_number(row[k], k) for k in ('restore_lo', 'restore_hi'))
        if not lo <= value <= hi:
            raise ValueError('Restore observation is outside its counter bounds')
        restored += value
        lower += lo
        upper += hi
        cached += _number(row['started_cached_tokens'], 'cached tokens')
        steps += _number(row['features'][0], 'forward steps')
        tokens += _number(row['features'][1], 'new tokens')
    if not all(math.isfinite(v) for v in (restored, lower, upper, cached, steps, tokens)):
        raise ValueError('Onboarding aggregate overflow')
    if cached <= 0:
        raise ValueError('No cached tokens; restore ratio is not identifiable')
    if steps <= 0:
        raise ValueError('No forward steps in closed P windows')
    result = dict(node=rows[0]['node'], start=rows[0]['start'], end=rows[-1]['end'],
                  windows=len(rows), restored_tokens=restored, cached_tokens=cached,
                  restored_per_cached_token=restored/cached,
                  restored_per_cached_token_bounds=[lower/cached, upper/cached],
                  forward_steps=steps, new_tokens=tokens,
                  method='closed_window_restore_ratio_and_packing')
    if tokens > 0:
        result['tokens_per_step'] = tokens/steps
    return result


def update_workers(workers, windows):
    """Produce per-node sidecar observations; never modify any fitted model."""
    grouped = {}
    for row in windows:
        node = row['node']
        if node not in workers:
            raise ValueError('Unknown physical worker in onboarding data')
        grouped.setdefault(node, []).append(row)
    updates, failures = {}, {}
    for node, rows in grouped.items():
        try:
            updates[node] = estimate_onboarding(rows)
        except ValueError as error:
            failures[node] = str(error)
    return dict(workers=updates, failures=failures)


def observe_onboarding(cfg: dict, assignment: dict, *, start, end, mirror: Mirror):
    """Restore ratio of the current P workers over the decision window."""
    directory, scratch = Path(cfg["run_dir"]), Path(cfg["scratch"])
    url_node = {w["url"]: w["node"] for w in cfg["workers"]}
    roles = {url_node[u]: r for u, r in assignment["roles"].items() if r == "prefill"}
    Clock(cfg, directory).observe()  # complete the clock bracket after `end`
    name = f"r{assignment['revision']}-{int(end * 1e9)}"
    snapshot, records, audit = capture_interval(cfg, directory, scratch / "onboarding" / name / "snapshot", start=start,
                                       end=end, roles=roles, mirror=mirror)
    raw = (directory / "calibration" / "report.json").read_bytes()
    replicas = counter_replicas(json.loads((directory / "worker-capacities.json").read_text()))
    windows, _, coverage = snapshot_measurements(snapshot, records, replicas=replicas)
    report = update_workers(json.loads(raw)["workers"], windows)
    for node in roles:
        if node not in report["workers"]:
            report["failures"].setdefault(node, "No complete current P windows")
    report.update(assignment=assignment, start=start, end=end, coverage=coverage, clock_audit=audit,
                  compute_model_sha256=hashlib.sha256(raw).hexdigest())
    save(directory / "onboarding" / name / "report.json", report)
    return report
