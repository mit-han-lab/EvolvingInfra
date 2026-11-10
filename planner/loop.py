"""Live P/D role controller: fit each worker once after learning, then move P/D roles through the Router.

The decision rule and the fit are ``evolving_infra.planner.calibration``: a learning interval (default
1800 s) ends in one fit per worker; then every decision period (600 s) every other P count is a
candidate, scored per 60-s workload slice, and the controller switches only when the mean gain exceeds
two standard errors. Prediction samples within 120 s after a confirmed switch are skipped, and P restore
ratios are re-observed before each decision (onboarding). A role change is one Router role assignment
(``PUT /symmetric_pd/roles`` with epoch/revision compare-and-set, never retried; the Router must run with
``--symmetric-pd-external-role-control``). No engine is restarted.

Deployment contract (all paths and URLs come from the JSON config; see ``DEFAULTS`` and
``planner/README.md``):

- every worker host runs ``python -m evolving_infra.planner serve-traces`` over the engines'
  ``--sched-trace-dir`` parent; the controller mirrors the iteration/request traces over HTTP and
  measures each host clock with ``GET /time`` (no shared filesystem or SSH needed);
- worker identity is ``nodeN``; each worker has ``url`` (engine), ``trace_url`` (its host's trace
  server), ``name`` (its trace subdirectory), ``host`` (host index) and ``initial_role``;
- restore counters (``sglang:load_back_tokens_total{pool="kv"}``) are scraped from the current P
  workers; every TP rank increments the counter, so ``restore_counter_replicas`` is the TP size;
- the caller writes ``<run_dir>/replay-timing.json`` (``start_wall``, ``duration_s``) when the measured
  workload starts; no role update is made after ``start_wall + duration_s``;
- a failure writes ``<run_dir>/PLANNER_FAILED``; after the cutoff the process stays alive and idle
  (``idle_after_cutoff``) so a supervisor can tell completion from a crash;
- ``role_script`` (testing only) replaces decisions with timed role changes after the fit.
"""
import hashlib
import http.client
import http.server
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.parse
import urllib.request

from evolving_infra.planner.priors import PREFILL_SHAPE_PRIORS

TRACE_PORT = 18500  # per host: scheduler trace files + /time

DEFAULTS = dict(
    learning_s=1800, period_s=600, slice_s=60., required_standard_errors=2., exclusion_s=120.,
    improvement_rule="minute_standard_error", update_onboarding=True,
    restore_counter_replicas=4, clock_pad_s=0.5, telemetry_period_s=3., clock_probe_period_s=300.,
    poll_s=3., trace_port=TRACE_PORT, role_script=None, idle_after_cutoff=True,
    prefill_shape_prior="dsv4-flash-h200-tp4",
)


def load_config(path) -> dict:
    """Read a controller config, fill ``DEFAULTS`` and record its own path (the fit subprocess rereads it)."""
    cfg = dict(DEFAULTS, **json.loads(Path(path).read_text()))
    cfg.setdefault("config_path", str(Path(path).resolve()))
    return cfg


def resolve_prior(cfg: dict) -> dict:
    """The predeclared P step-model shape: a built-in name or an inline ``{"coefficients": [...]}``.

    Used only for a worker whose own four-term fit is not identifiable (``fit_prefill_scale`` learns one
    multiplier on it). An optional ``prefill_shape_prior_sha256`` in the config pins the exact prior.
    """
    prior = cfg["prefill_shape_prior"]
    if isinstance(prior, str):
        if prior not in PREFILL_SHAPE_PRIORS:
            raise ValueError(f"Unknown prefill shape prior {prior!r}; known: {sorted(PREFILL_SHAPE_PRIORS)}")
        prior = PREFILL_SHAPE_PRIORS[prior]
    coefficients = prior.get("coefficients") if isinstance(prior, dict) else None
    if (not isinstance(coefficients, list) or len(coefficients) not in (4, 5)
            or not all(isinstance(c, (int, float)) and math.isfinite(c) and c >= 0 for c in coefficients)):
        raise ValueError("prefill_shape_prior needs 4 (or 5) finite nonnegative coefficients")
    pinned = cfg.get("prefill_shape_prior_sha256")
    if pinned is not None and pinned != prior_sha256(prior):
        raise ValueError("Predeclared P model shape changed")
    return prior


def prior_sha256(prior: dict) -> str:
    return hashlib.sha256(json.dumps(prior, sort_keys=True).encode()).hexdigest()


def build_prior(report: dict) -> list[float]:
    """Coordinate-wise median of the measured four-term P models of a calibration fit report."""
    rows = [m["coefficients"] for w in report["workers"].values() if w["prefill"]["origin"] == "measured"
            for m in w["prefill"]["models"] if m.get("method") == "per_node_four_term_nnls"]
    if not rows or any(len(r) != 4 for r in rows):
        raise ValueError("No measured four-term prefill models in the report")
    return [statistics.median(r[i] for r in rows) for i in range(4)]


def save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def http_json(url: str, timeout: float = 10, headers: dict | None = None):
    req = urllib.request.Request(url, headers=headers or {})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        return json.loads(response.read())


# ------------------------------------------------------------------------------------------- host side

def serve_traces(root: str, port: int = TRACE_PORT, host: str = "0.0.0.0"):
    """Every host: GET /time -> {"time_ns"}; GET /list?dir=<worker> -> file names; GET /read?path=&offset=
    -> the file's bytes from offset (incremental mirror). Read-only, inside `root` only."""
    base = Path(root).resolve()
    base.mkdir(parents=True, exist_ok=True)

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def reply(self, body: bytes, kind="application/octet-stream", code=200):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urllib.parse.urlsplit(self.path)
            query = dict(urllib.parse.parse_qsl(url.query))
            if url.path == "/time":
                return self.reply(json.dumps({"time_ns": time.time_ns()}).encode(), "application/json")
            target = (base / query.get("dir", query.get("path", ""))).resolve()
            if base not in (target, *target.parents):
                return self.reply(b"outside root", code=403)
            if url.path == "/list":
                names = sorted(p.name for p in target.iterdir()) if target.is_dir() else []
                return self.reply(json.dumps(names).encode(), "application/json")
            if url.path == "/read" and target.is_file():
                with target.open("rb") as f:
                    f.seek(int(query.get("offset", 0)))
                    return self.reply(f.read(64 << 20))
            return self.reply(b"not found", code=404)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ------------------------------------------------------------------------------------------- controller side

class Mirror:
    """Append-only local copies of each worker's iter/req trace stream (one stream per worker: a second
    pid file means the engine restarted, which the fit must not mix)."""

    def __init__(self, root: Path, workers: list[dict]):
        self.root, self.workers = Path(root), {w["node"]: w for w in workers}

    def sync(self, nodes) -> dict:
        out = {}
        for node in nodes:
            w = self.workers[node]
            names = http_json(f"{w['trace_url']}/list?dir={w['name']}", timeout=30)
            item = {}
            for kind, field in (("iter", "steps"), ("req", "requests")):
                found = [n for n in names if re.fullmatch(rf"{kind}_trace\.dp0_pp0\.pid\d+\.jsonl", n)]
                if len(found) != 1:
                    raise ValueError(f"{node}: expected one {kind} trace stream, found {found}")
                local = self.root / node / found[0]
                local.parent.mkdir(parents=True, exist_ok=True)
                while True:
                    offset = local.stat().st_size if local.exists() else 0
                    query = urllib.parse.urlencode({"path": f"{w['name']}/{found[0]}", "offset": offset})
                    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                            f"{w['trace_url']}/read?{query}", timeout=120) as r:
                        data = r.read()
                    with local.open("ab") as f:
                        f.write(data)
                    if len(data) < 64 << 20:
                        break
                item[field] = str(local)
            out[node] = item
        return out


def probe(host: dict, port: int, exchanges: int = 8) -> dict:
    """One host's clock against this controller (best of `exchanges` HTTP keep-alive round trips)."""
    conn = http.client.HTTPConnection(host["ip"], port, timeout=15)
    samples = []
    try:
        for _ in range(exchanges):
            before = time.time_ns()
            conn.request("GET", "/time")
            remote = json.loads(conn.getresponse().read())["time_ns"]
            after = time.time_ns()
            samples.append(dict(roundtrip_s=(after - before) / 1e9, controller_before_ns=before,
                                controller_after_ns=after,
                                worker_minus_controller_s=(remote - (before + after) / 2) / 1e9,
                                lower_s=(remote - after) / 1e9, upper_s=(remote - before) / 1e9))
    finally:
        conn.close()
    best = min(samples, key=lambda s: s["roundtrip_s"])
    return dict(node=host["index"], host=host["ip"], **best, samples=samples)


class Clock:
    """Clock audit over HTTP: a fixed offset per host measured at start; every later exchange must stay
    within clock_pad_s of it, and exchanges must bracket each captured interval."""

    def __init__(self, cfg: dict, directory: Path):
        self.hosts, self.port, self.pad = cfg["hosts"], cfg["trace_port"], cfg["clock_pad_s"]
        self.dir = directory / "diagnostics"
        self.reference = directory / "clock-reference.json"

    def observe(self) -> dict:
        with ThreadPoolExecutor(len(self.hosts)) as pool:
            nodes = list(pool.map(lambda h: probe(h, self.port), self.hosts))
        row = dict(at=time.time(), nodes=nodes)
        if not self.reference.exists():
            save(self.reference, dict(at=row["at"], offsets_s={str(n["node"]): n["worker_minus_controller_s"]
                                                              for n in nodes}))
        self.validate(row)
        save(self.dir / f"clock-offsets-{time.time_ns()}.json", row)
        return row

    def offsets(self) -> dict:
        return {int(k): v for k, v in json.loads(self.reference.read_text())["offsets_s"].items()}

    def validate(self, obs: dict):
        reference = self.offsets()
        if {n["node"] for n in obs["nodes"]} != set(reference):
            raise ValueError("Clock host coverage differs")
        for n in obs["nodes"]:
            lo, hi, shift = n["lower_s"], n["upper_s"], reference[n["node"]]
            if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi or max(abs(lo - shift), abs(hi - shift)) > self.pad:
                raise ValueError(f"Host {n['node']} clock residual exceeds {self.pad} s: {lo - shift:.4f}..{hi - shift:.4f}")

    def verify(self, start: float, end: float, max_gap_s: float = 360) -> dict:
        rows = [json.loads(p.read_text()) for p in sorted(self.dir.glob("clock-offsets-*.json"))]
        for row in rows:
            self.validate(row)
        gap_seen = 0.
        for host in self.hosts:
            ex = sorted((n for r in rows for n in r["nodes"] if n["node"] == host["index"]),
                        key=lambda n: n["controller_before_ns"])
            before = [n for n in ex if n["controller_after_ns"] / 1e9 <= start]
            after = [n for n in ex if n["controller_before_ns"] / 1e9 >= end]
            if not before or not after:
                raise ValueError("Per-host clock exchanges must bracket the measurement")
            part = ex[ex.index(before[-1]):ex.index(after[0]) + 1]
            gap = max((b["controller_after_ns"] - a["controller_before_ns"]) / 1e9 for a, b in zip(part, part[1:]))
            gap_seen = max(gap_seen, gap)
            if gap > max_gap_s:
                raise ValueError(f"Host {host['index']} clock observation gap {gap:.0f} s > {max_gap_s} s")
        return dict(passed=True, pad_s=self.pad, offsets_s=self.offsets(), observations=len(rows),
                    max_observation_gap_s=gap_seen)


class Telemetry(threading.Thread):
    """Restore counters of the current P workers every period (restore-observations.jsonl, one record per
    worker scrape with its own collection interval)."""

    KEEP = ("# HELP sglang:hicache_backup_bytes_total ", "sglang:load_back_tokens_total{")

    def __init__(self, workers: list[dict], path: Path, period_s: float):
        super().__init__(daemon=True)
        self.workers, self.path, self.period = workers, path, period_s
        self.roles: dict = {}  # url -> role, refreshed by the control loop
        self.halt, self.error = threading.Event(), None

    def scrape(self, w):
        began = time.time()
        node = dict(node=w["idx"], role="prefill", at=began, errors={})
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(w["url"] + "/metrics",
                                                                                   timeout=4) as r:
                text = r.read().decode(errors="replace")
            node["metrics"] = "\n".join(x for x in text.splitlines() if x.startswith(self.KEEP)) + "\n"
        except Exception as error:  # noqa: BLE001 - recorded; the fit excludes windows without brackets
            node.update(metrics="", errors=dict(metrics=repr(error)))
        return dict(at=time.time(), nodes=[node])

    def run(self):
        try:
            with self.path.open("a", buffering=1) as out, ThreadPoolExecutor(len(self.workers)) as pool:
                while not self.halt.is_set():
                    began = time.monotonic()
                    current = [w for w in self.workers if self.roles.get(w["url"]) == "prefill"]
                    for row in pool.map(self.scrape, current):
                        out.write(json.dumps(row) + "\n")
                    self.halt.wait(max(0., self.period - (time.monotonic() - began)))
        except Exception as error:  # noqa: BLE001 - raised by the control loop
            self.error = error


# transition guard --------------------------------------------------------------------------------------------

MAX_WAIT_S = 120.0


def adjusted_loads(loads):
    """A /symmetric_pd/backend/loads reply with unknown P reservations left out of active_prefill_reqs."""
    workers = {}
    for url, view in loads["workers"].items():
        view = dict(view)
        active, unknown = int(view.get("active_prefill_reqs") or 0), int(view.get("unknown_prefill_reqs") or 0)
        view["active_prefill_reqs"] = max(0, active - unknown)
        view["unknown_prefill_reqs_excluded"] = min(active, unknown)
        workers[url] = view
    return {**loads, "workers": workers}


class HoldGuard:
    """A hold on one condition blocks decisions for at most max_wait_s (then logged); drains never."""

    def __init__(self, max_wait_s=MAX_WAIT_S, clock=time.monotonic):
        self.max_wait_s, self.clock, self.since = float(max_wait_s), clock, {}

    def hold(self, kind, blocked, detail, journal, overridable=True):
        now = self.clock()
        if not blocked:
            self.since.pop(kind, None)
            return False
        waited = now - self.since.setdefault(kind, now)
        if overridable and waited >= self.max_wait_s:
            journal(dict(event="transition_guard_max_wait", kind=kind, waited_s=round(waited, 3),
                         max_wait_s=self.max_wait_s, detail=detail))
            return False
        return True


def residual_drains(residual):
    return any("draining_age_s" in row for row in residual.values())


# ------------------------------------------------------------------------------------------- capture / fit

def model_experts(urls):
    """(routed experts, experts per token) of the served model (calibration.controller.served_experts)."""
    from evolving_infra.planner.calibration.controller import served_experts
    return served_experts(urls)


def capture_interval(cfg: dict, directory: Path, destination: Path, *, start, end, roles: dict, mirror: Mirror):
    """Mirror the traces of `roles` workers, audit clocks around [start, end], freeze the snapshot."""
    from evolving_infra.planner.calibration.snapshot import capture
    clock = Clock(cfg, directory)
    audit = clock.verify(start, end)
    offsets = clock.offsets()
    by_node = {w["node"]: w for w in cfg["workers"]}
    paths = mirror.sync(sorted(roles))
    inputs = [dict(node=node, host=by_node[node]["ip"], generation=0, role=role, **paths[node])
              for node, role in sorted(roles.items())]
    snapshot = capture(destination, inputs, directory / "restore-observations.jsonl", start=start, end=end,
                       clock_pad_s=cfg["clock_pad_s"],
                       worker_clock_offsets_s={node: offsets[by_node[node]["host"]] for node in roles})
    return snapshot, audit


def fit(cfg: dict):
    """Learning-end fit: one frozen per-worker model of minutes 0..learning."""
    from evolving_infra.planner.calibration.dynamic_fit import fit_snapshot
    directory, scratch = Path(cfg["run_dir"]), Path(cfg["scratch"])
    timing = json.loads((directory / "replay-timing.json").read_text())
    start = timing["start_wall"]
    end = start + cfg["learning_s"]
    prior = resolve_prior(cfg)
    roles = {w["node"]: w["initial_role"] for w in cfg["workers"]}
    snapshot, audit = capture_interval(cfg, directory, scratch / "calibration" / "snapshot", start=start, end=end,
                                       roles=roles, mirror=Mirror(scratch / "mirror", cfg["workers"]))
    capacities = json.loads((directory / "worker-capacities.json").read_text())
    experts, top_k = model_experts(w["url"] for w in cfg["workers"])
    result = fit_snapshot(snapshot, capacities=capacities, replicas=cfg["restore_counter_replicas"],
                          experts=experts, top_k=top_k,
                          predeclared_prefill_shape=prior["coefficients"])
    result.update(clock_audit=audit, experts=experts, top_k=top_k, prefill_shape_prior=prior,
                  prefill_shape_prior_sha256=prior_sha256(prior))
    save(directory / "calibration" / "snapshot-manifest.json", snapshot)
    save(directory / "calibration" / "report.json", result)
    return result


def observe_onboarding(cfg: dict, assignment: dict, *, start, end, mirror: Mirror):
    """Restore ratio of the current P workers over the decision window."""
    from evolving_infra.planner.calibration.dynamic_fit import snapshot_measurements
    from evolving_infra.planner.calibration.onboarding import update_workers
    directory, scratch = Path(cfg["run_dir"]), Path(cfg["scratch"])
    url_node = {w["url"]: w["node"] for w in cfg["workers"]}
    roles = {url_node[u]: r for u, r in assignment["roles"].items() if r == "prefill"}
    Clock(cfg, directory).observe()  # complete the clock bracket after `end`
    name = f"r{assignment['revision']}-{int(end * 1e9)}"
    snapshot, audit = capture_interval(cfg, directory, scratch / "onboarding" / name / "snapshot", start=start,
                                       end=end, roles=roles, mirror=mirror)
    raw = (directory / "calibration" / "report.json").read_bytes()
    windows, _, coverage = snapshot_measurements(snapshot, replicas=cfg["restore_counter_replicas"])
    report = update_workers(json.loads(raw)["workers"], windows)
    for node in roles:
        if node not in report["workers"]:
            report["failures"].setdefault(node, "No complete current P windows")
    report.update(assignment=assignment, start=start, end=end, coverage=coverage, clock_audit=audit,
                  compute_model_sha256=hashlib.sha256(raw).hexdigest())
    save(directory / "onboarding" / name / "report.json", report)
    return report


# ------------------------------------------------------------------------------------------- control loop

def control(cfg: dict):
    from evolving_infra.planner.calibration.controller import Controller, RoleClient, SLICE_S, node_assignment, transition_residual
    from evolving_infra.planner.calibration.dynamic import REQUIRED_STANDARD_ERRORS, positive
    directory, scratch = Path(cfg["run_dir"]), Path(cfg["scratch"])
    period, learning = positive(cfg["period_s"], "decision period"), positive(cfg["learning_s"], "learning")
    resolve_prior(cfg)  # fail before the run, not at learning end
    if learning < period:
        raise ValueError("Learning must cover a full decision interval")
    if (cfg["improvement_rule"] != "minute_standard_error" or cfg["slice_s"] != SLICE_S
            or cfg["required_standard_errors"] != REQUIRED_STANDARD_ERRORS):
        raise ValueError("This controller requires the per-minute standard-error rule")
    if period - cfg["exclusion_s"] < 2 * SLICE_S:
        raise ValueError("Post-switch exclusion leaves fewer than two workload slices")
    script = sorted(cfg.get("role_script") or [], key=lambda x: x[0])
    scripted = bool(script)  # scripted: timed role changes replace decisions for the whole run
    if script and script[0][0] <= learning:
        raise ValueError("Scripted role changes must follow the learning-end fit")
    client = RoleClient(cfg["router_url"], Path(cfg["token_file"]).read_text().strip())
    node_urls = {w["node"]: w["url"] for w in cfg["workers"]}
    initial_roles = {w["url"]: w["initial_role"] for w in cfg["workers"]}
    capacities = {}
    for w in cfg["workers"]:
        info = http_json(w["url"] + "/server_info", timeout=30)
        capacities[w["node"]] = dict(kv_capacity_tokens=int(info["max_total_num_tokens"]),
                                     max_running_requests=int(info["max_running_requests"]))
    save(directory / "worker-capacities.json", capacities)
    clock = Clock(cfg, directory)
    telemetry = Telemetry(cfg["workers"], directory / "restore-observations.jsonl", cfg["telemetry_period_s"])
    telemetry.roles = dict(initial_roles)
    telemetry.start()
    mirror = Mirror(scratch / "mirror", cfg["workers"])
    log = (directory / "role-decisions.jsonl").open("x", buffering=1)
    observations = (directory / "role-observations.jsonl").open("x", buffering=1)

    def journal(value):
        log.write(json.dumps(dict(at=time.time(), monotonic=time.monotonic(), **value)) + "\n")

    client = JournaledRoleClient(client, journal)

    clock_pool, onboarding_pool = ThreadPoolExecutor(1), ThreadPoolExecutor(1)
    clock_task = clock_pool.submit(clock.observe)
    clock_task.result()
    clock_task, next_clock = None, time.monotonic() + cfg["clock_probe_period_s"]
    samples, controller, fit_proc, timing, prediction_initial = [], None, None, None, None
    start_mono = learning_end = deadline = next_decision = first_assignment = None
    first_decision, cutoff_probe = True, False
    onboarding_task = onboarding_samples = None
    guard = HoldGuard()
    try:
        while True:
            now = time.monotonic()
            if telemetry.error is not None:
                raise RuntimeError(f"telemetry sampler failed: {telemetry.error!r}")
            if clock_task is not None and clock_task.done():
                clock_task.result()
                clock_task = None
            if now >= next_clock and clock_task is None:
                clock_task, next_clock = clock_pool.submit(clock.observe), now + cfg["clock_probe_period_s"]
            path = directory / "replay-timing.json"
            if timing is None and path.exists():
                timing = json.loads(path.read_text())
                start_mono = time.monotonic() - (time.time() - timing["start_wall"])
                learning_end, deadline = start_mono + learning, start_mono + timing["duration_s"]
                journal(dict(event="replay_timing", **timing, learning_end_monotonic=learning_end,
                             deadline_monotonic=deadline))
            if timing and now >= deadline:
                journal(dict(event="measurement_cutoff_no_further_role_updates"))
                break
            s = client.request()
            node_assignment(s, node_urls)
            telemetry.roles = dict(s["assignment"]["roles"])
            if first_assignment is None:
                if s["assignment"]["roles"] != initial_roles:
                    raise RuntimeError("Initial roles do not match the declared configuration")
                first_assignment = s["assignment"]
                save(directory / "role-controller-ready.json", dict(at=time.time(), initial_assignment=first_assignment,
                                                                    first_clock_probe_complete=True))
            if controller is None and s["assignment"] != first_assignment:
                raise RuntimeError("Initial roles/incarnation changed before learning")
            row = dict(at=time.monotonic(), wall=time.time(), snapshot=s)
            observations.write(json.dumps(row) + "\n")
            if timing and row["at"] >= learning_end - period and not scripted:
                # Raw observations stay complete; only prediction samples skip the post-switch exclusion.
                disposition = controller.prediction_observation(s, row["at"], journal) if controller else "included"
                if disposition == "excluded":
                    samples = []
                elif disposition == "reset":
                    samples = [row]
                elif not (first_decision and samples and samples[-1]["at"] >= next_decision):
                    samples.append(row)
                if prediction_initial is None:
                    prediction_initial = s
                    next_decision = max(learning_end, row["at"] + period)
                    journal(dict(event="prediction_sampling_started", learning_end_monotonic=learning_end,
                                 actual_start_monotonic=row["at"], first_decision_not_before_monotonic=next_decision))
            if timing and now >= learning_end and controller is None:
                if not cutoff_probe:
                    if clock_task is not None:
                        clock_task.result()
                    clock_task, cutoff_probe = clock_pool.submit(clock.observe), True
                elif fit_proc is None and clock_task is None:
                    (directory / "calibration").mkdir(exist_ok=True)
                    fit_proc = subprocess.Popen(
                        [sys.executable, "-m", "evolving_infra.planner", "fit", cfg["config_path"]],
                        stdout=(directory / "calibration" / "fit.log").open("w"), stderr=subprocess.STDOUT,
                        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
                    journal(dict(event="warmup_fit_started", pid=fit_proc.pid))
                elif fit_proc is not None and fit_proc.poll() is not None:
                    if fit_proc.returncode:
                        raise RuntimeError("Warmup fit failed; see calibration/fit.log")
                    raw = (directory / "calibration" / "report.json").read_bytes()
                    fitted = json.loads(raw)
                    controller = Controller(fitted, node_urls, prediction_initial or s, experts=fitted["experts"],
                                            top_k=fitted["top_k"], fit_digest=hashlib.sha256(raw).hexdigest(),
                                            exclusion_s=cfg["exclusion_s"])
                    journal(dict(event="warmup_fit_completed", learning_s=learning, fit_failures=fitted["fit_failures"],
                                 origins={n: {r: v[r]["origin"] for r in ("prefill", "decode")}
                                          for n, v in fitted["workers"].items()},
                                 model_ready_monotonic=time.monotonic()))
            if controller is not None and script and now >= start_mono + script[0][0]:
                at_s, change = script.pop(0)
                apply_script(s, {node_urls[n]: r for n, r in change.items()}, client, journal, at_s)
            elif controller is not None and not scripted and now >= next_decision and len(samples) >= 2:
                decision_samples = samples
                if cfg["update_onboarding"]:
                    if onboarding_task is None:
                        onboarding_samples = list(samples)
                        onboarding_task = onboarding_pool.submit(
                            observe_onboarding, cfg, s["assignment"], start=samples[0]["wall"],
                            end=samples[-1]["wall"], mirror=mirror)
                    if not onboarding_task.done():
                        time.sleep(cfg["poll_s"])
                        continue
                    try:
                        controller.update_onboarding(s, onboarding_task.result(), journal)
                    except (OSError, ValueError, KeyError) as error:
                        journal(dict(event="onboarding_update_unavailable", reason=str(error)))
                    onboarding_task, decision_samples, onboarding_samples = None, onboarding_samples, None
                try:
                    unknown_d = {u: c[2] for u, c in s["decode_population"]["workers"].items() if c[2]}
                    if guard.hold("unknown_decode_population", bool(unknown_d), unknown_d, journal):
                        raise ValueError("Unknown current decode population")
                    residual = transition_residual(s, adjusted_loads(
                        http_json(cfg["router_url"] + "/symmetric_pd/backend/loads", timeout=5)))
                    if guard.hold("old_role_requests", bool(residual), residual, journal,
                                  overridable=not residual_drains(residual)):
                        raise ValueError("Old role still active: " + repr(residual))
                    decision = controller.propose(decision_samples)
                except ValueError as error:
                    controller.skip(s, str(error), journal)
                else:
                    if time.monotonic() + client.timeout + 1 >= deadline:
                        journal(dict(event="near_cutoff_hold_no_role_apply"))
                        next_decision = deadline + 1
                        continue
                    assignment = controller.apply(s, decision, client, journal)
                    save(directory / "role-state.json", dict(assignment=assignment))
                first_decision = False
                samples = [row]
                next_decision = time.monotonic() + period
            time.sleep(cfg["poll_s"])
        if clock_task is not None:
            clock_task.result(timeout=25)
        telemetry.halt.set()
        save(directory / "role-controller-complete.json", dict(
            at=time.time(), cutoff_reached=True, fit_completed=controller is not None,
            scripted_changes_left=len(script)))
    except BaseException as error:
        (directory / "PLANNER_FAILED").write_text(f"planner controller: {error!r}")
        journal(dict(event="controller_failed", error=repr(error)))
        raise
    finally:
        if fit_proc is not None and fit_proc.poll() is None:
            fit_proc.terminate()
        clock_pool.shutdown(wait=False, cancel_futures=True)
        onboarding_pool.shutdown(wait=False, cancel_futures=True)
        log.close()
        observations.close()
    while cfg["idle_after_cutoff"]:  # alive and idle (no role updates) until the caller stops it
        time.sleep(60)


class JournaledRoleClient:
    """calibration RoleClient whose HTTP errors are journaled (and printed to controller.log) with the
    router's status and response body before they propagate; Controller.apply records only the type."""

    def __init__(self, client, journal):
        self.client, self.journal, self.timeout = client, journal, client.timeout

    def request(self, body=None):
        try:
            return self.client.request(body)
        except urllib.error.HTTPError as error:
            try:
                text = error.read().decode(errors="replace")[:2000]
            except Exception as read_error:  # noqa: BLE001 - the status alone still helps
                text = f"<unreadable: {read_error!r}>"
            event = dict(event="role_request_http_error", method="GET" if body is None else "PUT",
                         status=error.code, reason=str(error.reason), response_body=text)
            self.journal(event)
            print(f"planner controller: {json.dumps(event)}", file=sys.stderr, flush=True)
            raise


def apply_script(snapshot: dict, change: dict, client, journal, at_s):
    """Scripted runs only: one CAS role update with the same journal events as Controller.apply."""
    a = snapshot["assignment"]
    if snapshot.get("draining"):
        raise RuntimeError("Scripted role change while a drain is in progress")
    desired = {**a["roles"], **change}
    body = dict(epoch=a["epoch"], revision=a["revision"], expected=a["roles"], desired=desired)
    journal(dict(event="role_update_intent", scripted=True, at_s=at_s, body=body))
    expected = dict(epoch=a["epoch"], revision=a["revision"] + 1, roles=desired)
    try:
        response = client.request(body)
    except Exception as error:  # noqa: BLE001 - read back, never retry the mutation
        journal(dict(event="role_update_response_unknown", error=type(error).__name__))
        response = client.request()
    if response["assignment"] != expected:
        raise RuntimeError("Scripted role update unconfirmed or changed concurrently")
    journal(dict(event="role_update_confirmed", scripted=True, assignment=response["assignment"],
                 draining=response.get("draining")))
    return response["assignment"]


def last_applied(journal_path: Path, initial: dict | None) -> dict | None:
    """The assignment of the last confirmed role update in a decisions journal (else `initial`)."""
    last = initial
    if journal_path.exists():
        for line in journal_path.open():
            row = json.loads(line)
            if row.get("event") == "role_update_confirmed":
                last = row["assignment"]
    return last
