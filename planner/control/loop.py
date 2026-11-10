"""Live P/D role controller: fit each worker once after learning, then move P/D roles through the Router.

A learning interval (default 1800 s) ends in one fit per worker (``fit.learning``); then every decision period
(600 s) every other P count is a candidate, scored per 60-s workload slice (``decide``), and the controller switches
only when the mean gain exceeds two standard errors. Prediction samples within 120 s after a confirmed switch are
skipped, and the P restore ratios and packing are re-observed before each decision (``fit.current_p``). A role
change is one Router role assignment (``api.RoleClient``). No engine is restarted.

Deployment contract (all paths and URLs come from the JSON config; see ``config`` and ``docs/dynamic-pd.md``):

- every worker host runs ``python -m evolving_infra.planner serve-traces`` over the engines'
  ``--sched-trace-dir`` parent; the controller mirrors the iteration/request traces over HTTP and
  measures each host clock with ``GET /time`` (no shared filesystem or SSH needed);
- worker identity is ``nodeN``; each worker has ``url`` (engine), ``trace_url`` (its host's trace
  server), ``name`` (its trace subdirectory), ``host`` (host index) and ``initial_role``, and the router
  starts with these initial roles;
- restore counters (``sglang:load_back_tokens_total{pool="kv"}``) are scraped from the current P
  workers; every TP rank increments the counter, so values are divided by the engines' TP size;
- ``mode`` (``config``): in ``service`` mode learning starts when the controller starts and there is no cutoff;
  in ``benchmark`` mode the caller writes ``<run_dir>/replay-timing.json`` (``start_wall``, ``duration_s``)
  when the measured workload starts, no role update is made after ``start_wall + duration_s``, and after the
  cutoff the process stays alive and idle (``idle_after_cutoff``) so a supervisor can tell completion from a
  crash;
- a failure that stops the controller writes ``<run_dir>/PLANNER_FAILED``. In benchmark mode every error does.
  In service mode only these do: the router's worker set differs from the config, the roles before the fit
  differ from the declared initial roles, or the learning-end fit fails. Every other error (router or engine
  unreachable, a failed clock probe, the restore-counter scraper stopping, a failed or unconfirmed role update)
  is journaled and the controller keeps running; a role assignment changed outside the controller is adopted;
- ``role_script`` (testing only) replaces decisions with timed role changes after the fit.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from ..api import RoleClient, backend_loads, cas_update, worker_capacities
from ..collect.clock import Clock
from ..collect.restore_counters import Telemetry
from ..collect.trace_server import Mirror
from ..common import save
from ..config import check_control_config, learning_start_path
from ..fit.current_p import observe_onboarding
from .controller import Controller, node_assignment
from .transition import HoldGuard, adjusted_loads, residual_drains, transition_residual

# What one poll of the loop asks for next.
SLEEP, AGAIN, STOP = "sleep", "again", "stop"


def control(cfg: dict):
    """Run the live controller (see the module docstring)."""
    ControlLoop(cfg).run()


class ControlLoop:
    """One controller run. Every `poll_s` it reads the router's roles and completed workload; during the last
    decision period of learning and afterwards those observations are the decision samples. When learning ends it
    runs the fit once (a subprocess); then every `period_s` it re-observes the P workers and decides."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.directory, self.scratch = Path(cfg["run_dir"]), Path(cfg["scratch"])
        self.period, self.learning, self.script = check_control_config(cfg)
        self.service = cfg["mode"] == "service"
        self.scripted = bool(self.script)  # scripted: timed role changes replace decisions for the whole run
        token = Path(cfg["token_file"]).read_text().strip()
        self.node_urls = {w["node"]: w["url"] for w in cfg["workers"]}
        self.initial_roles = {w["url"]: w["initial_role"] for w in cfg["workers"]}
        save(self.directory / "worker-capacities.json", worker_capacities(cfg["workers"]))
        self.clock = Clock(cfg, self.directory)
        self.start_telemetry(self.initial_roles)
        self.mirror = Mirror(self.scratch / "mirror", cfg["workers"])
        self.log = (self.directory / "role-decisions.jsonl").open("x", buffering=1)
        self.observations = (self.directory / "role-observations.jsonl").open("x", buffering=1)
        self.client = RoleClient(cfg["router_url"], token, self.journal)
        self.clock_pool, self.onboarding_pool = ThreadPoolExecutor(1), ThreadPoolExecutor(1)
        self.clock_pool.submit(self.clock.observe).result()
        self.clock_task, self.next_clock = None, time.monotonic() + cfg["clock_probe_period_s"]
        # Measurement timing, on the monotonic clock.
        self.timing = self.start_mono = self.learning_end = self.deadline = None
        # Learning end: one clock probe after the end, then the fit subprocess, then the controller.
        self.cutoff_probe, self.cutoff_task, self.fit_proc, self.controller = False, None, None, None
        # Decision samples and schedule.
        self.samples, self.prediction_initial, self.first_assignment = [], None, None
        self.next_decision, self.first_decision = None, True
        self.onboarding_task = self.onboarding_samples = None
        self.guard = HoldGuard()
        self.router_down = False

    def journal(self, value):
        self.log.write(json.dumps(dict(at=time.time(), monotonic=time.monotonic(), **value)) + "\n")

    def start_telemetry(self, roles):
        self.telemetry = Telemetry(self.cfg["workers"], self.directory / "restore-observations.jsonl",
                                   self.cfg["telemetry_period_s"])
        self.telemetry.roles = dict(roles)
        self.telemetry.start()

    def run(self):
        cfg = self.cfg
        try:
            while True:
                step = self.poll()
                if step == STOP:
                    break
                if step == SLEEP:
                    time.sleep(cfg["poll_s"])
            if self.clock_task is not None:
                self.clock_task.result(timeout=25)
            self.telemetry.halt.set()
            save(self.directory / "role-controller-complete.json", dict(
                at=time.time(), cutoff_reached=True, fit_completed=self.controller is not None,
                scripted_changes_left=len(self.script)))
        except BaseException as error:
            (self.directory / "PLANNER_FAILED").write_text(f"planner controller: {error!r}")
            self.journal(dict(event="controller_failed", error=repr(error)))
            raise
        finally:
            if self.fit_proc is not None and self.fit_proc.poll() is None:
                self.fit_proc.terminate()
            self.clock_pool.shutdown(wait=False, cancel_futures=True)
            self.onboarding_pool.shutdown(wait=False, cancel_futures=True)
            self.log.close()
            self.observations.close()
        while cfg["idle_after_cutoff"]:  # alive and idle (no role updates) until the caller stops it
            time.sleep(60)

    def poll(self):
        """One observation of the router, and whatever is due after it."""
        now = time.monotonic()
        self.check_background(now)
        self.read_timing()
        if self.timing and now >= self.deadline:
            self.journal(dict(event="measurement_cutoff_no_further_role_updates"))
            return STOP
        s = self.read_router()
        if s is None:
            return SLEEP
        node_assignment(s, self.node_urls)
        self.telemetry.roles = dict(s["assignment"]["roles"])
        self.check_initial_assignment(s)
        row = dict(at=time.monotonic(), wall=time.time(), snapshot=s)
        self.observations.write(json.dumps(row) + "\n")
        if self.timing and row["at"] >= self.learning_end - self.period and not self.scripted:
            self.add_sample(s, row)
        if self.timing and now >= self.learning_end and self.controller is None:
            self.finish_learning(s)
        if self.controller is not None and self.script and now >= self.start_mono + self.script[0][0]:
            at_s, change = self.script.pop(0)
            apply_script(s, {self.node_urls[n]: r for n, r in change.items()}, self.client, self.journal, at_s)
        elif (self.controller is not None and not self.scripted and now >= self.next_decision
              and len(self.samples) >= 2):
            return self.decide(s, row)
        return SLEEP

    def read_router(self):
        """The router's roles and completed workload; in service mode None while the router is unreachable."""
        try:
            s = self.client.request()
        except (OSError, ValueError) as error:  # unreachable, an HTTP error, or a reply that is not JSON
            if not self.service:
                raise
            if not self.router_down:
                self.journal(dict(event="router_unavailable", error=repr(error)))
            self.router_down = True
            return None
        if self.router_down:
            self.journal(dict(event="router_available"))
            self.router_down = False
        return s

    def check_background(self, now):
        """Handle a stopped restore-counter scraper; collect the periodic clock probe and start the next one."""
        if self.telemetry.error is not None:
            if not self.service:
                raise RuntimeError(f"telemetry sampler failed: {self.telemetry.error!r}")
            self.journal(dict(event="telemetry_restarted", error=repr(self.telemetry.error)))
            self.start_telemetry(self.telemetry.roles)
        if self.clock_task is not None and self.clock_task.done():
            self.collect_clock_probe()
        if now >= self.next_clock and self.clock_task is None:
            self.clock_task, self.next_clock = (self.clock_pool.submit(self.clock.observe),
                                                now + self.cfg["clock_probe_period_s"])

    def collect_clock_probe(self):
        """Wait for the pending clock probe. In service mode a failed probe is journaled, and a failed probe after
        learning is repeated before the fit, whose snapshot needs a probe after the learning interval."""
        task, self.clock_task = self.clock_task, None
        try:
            task.result()
        except Exception as error:  # noqa: BLE001 - any probe failure; benchmark mode re-raises it
            if not self.service:
                raise
            self.journal(dict(event="clock_probe_failed", error=repr(error), after_learning=task is self.cutoff_task))
            if task is self.cutoff_task:
                self.cutoff_probe = False

    def read_timing(self):
        """Place the learning start on the monotonic clock once: the controller start (service), or when the
        caller writes replay-timing.json (benchmark)."""
        if self.timing is not None:
            return
        if self.service:
            self.timing = dict(start_wall=time.time())
            save(learning_start_path(self.cfg), self.timing)  # the learning-end fit reads the start here
            self.start_mono = time.monotonic()
            self.learning_end, self.deadline = self.start_mono + self.learning, math.inf
            self.journal(dict(event="learning_started", mode="service", **self.timing,
                              learning_end_monotonic=self.learning_end))
            return
        path = learning_start_path(self.cfg)
        if path.exists():
            self.timing = json.loads(path.read_text())
            self.start_mono = time.monotonic() - (time.time() - self.timing["start_wall"])
            self.learning_end = self.start_mono + self.learning
            self.deadline = self.start_mono + self.timing["duration_s"]
            self.journal(dict(event="replay_timing", **self.timing, learning_end_monotonic=self.learning_end,
                              deadline_monotonic=self.deadline))

    def check_initial_assignment(self, s):
        """The router starts with the declared roles, and nothing changes them before the fit (each worker is
        measured in its initial role). In service mode a router restart that keeps those roles is accepted."""
        if self.first_assignment is None:
            if s["assignment"]["roles"] != self.initial_roles:
                raise RuntimeError("Initial roles do not match the declared configuration")
            self.first_assignment = s["assignment"]
            save(self.directory / "role-controller-ready.json", dict(
                at=time.time(), initial_assignment=self.first_assignment, first_clock_probe_complete=True))
        if self.controller is None and s["assignment"] != self.first_assignment:
            if not (self.service and s["assignment"]["roles"] == self.initial_roles):
                raise RuntimeError("Initial roles/incarnation changed before learning")
            self.journal(dict(event="router_restarted_before_fit", before=self.first_assignment,
                              after=s["assignment"]))
            self.first_assignment = s["assignment"]

    def add_sample(self, s, row):
        """Raw observations stay complete; only prediction samples skip the post-switch exclusion."""
        if self.service and self.controller is not None and s["assignment"] != self.controller.assignment:
            self.controller.adopt(s, row["at"], self.journal)
            self.next_decision = max(self.next_decision, row["at"] + self.period)
        disposition = (self.controller.prediction_observation(s, row["at"], self.journal) if self.controller
                       else "included")
        if disposition == "excluded":
            self.samples = []
        elif disposition == "reset":
            self.samples = [row]
        elif not (self.first_decision and self.samples and self.samples[-1]["at"] >= self.next_decision):
            self.samples.append(row)
        if self.prediction_initial is None:
            self.prediction_initial = s
            self.next_decision = max(self.learning_end, row["at"] + self.period)
            self.journal(dict(event="prediction_sampling_started", learning_end_monotonic=self.learning_end,
                              actual_start_monotonic=row["at"], first_decision_not_before_monotonic=self.next_decision))

    def finish_learning(self, s):
        """After learning: one clock probe that brackets its end, then the fit subprocess, then the controller."""
        cfg, directory = self.cfg, self.directory
        if not self.cutoff_probe:
            if self.clock_task is not None:
                self.collect_clock_probe()
            self.clock_task = self.cutoff_task = self.clock_pool.submit(self.clock.observe)
            self.cutoff_probe = True
        elif self.fit_proc is None and self.clock_task is None:
            (directory / "calibration").mkdir(exist_ok=True)
            self.fit_proc = subprocess.Popen(
                [sys.executable, "-m", "evolving_infra.planner", "fit", cfg["config_path"]],
                stdout=(directory / "calibration" / "fit.log").open("w"), stderr=subprocess.STDOUT,
                env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
            self.journal(dict(event="warmup_fit_started", pid=self.fit_proc.pid))
        elif self.fit_proc is not None and self.fit_proc.poll() is not None:
            if self.fit_proc.returncode:
                raise RuntimeError("Warmup fit failed; see calibration/fit.log")
            raw = (directory / "calibration" / "report.json").read_bytes()
            fitted = json.loads(raw)
            self.controller = Controller(fitted, self.node_urls, self.prediction_initial or s,
                                         experts=fitted["experts"], top_k=fitted["top_k"],
                                         fit_digest=hashlib.sha256(raw).hexdigest(), exclusion_s=cfg["exclusion_s"])
            self.journal(dict(event="warmup_fit_completed", learning_s=self.learning,
                              fit_failures=fitted["fit_failures"],
                              origins={n: {r: v[r]["origin"] for r in ("prefill", "decode")}
                                       for n, v in fitted["workers"].items()},
                              model_ready_monotonic=time.monotonic()))

    def decide(self, s, row):
        """A periodic decision: re-observe the P workers, check the transition holds, then propose and apply."""
        cfg = self.cfg
        # Service mode: an unreachable endpoint skips this decision instead of stopping the controller.
        skippable = (ValueError, OSError) if self.service else ValueError
        decision_samples = self.samples
        if cfg["update_onboarding"]:
            if self.onboarding_task is None:
                self.onboarding_samples = list(self.samples)
                self.onboarding_task = self.onboarding_pool.submit(
                    observe_onboarding, cfg, s["assignment"], start=self.samples[0]["wall"],
                    end=self.samples[-1]["wall"], mirror=self.mirror)
            if not self.onboarding_task.done():
                return SLEEP
            try:
                self.controller.update_onboarding(s, self.onboarding_task.result(), self.journal)
            except (Exception if self.service else (OSError, ValueError, KeyError)) as error:
                self.journal(dict(event="onboarding_update_unavailable", reason=str(error)))
            self.onboarding_task, decision_samples, self.onboarding_samples = None, self.onboarding_samples, None
        try:
            unknown_d = {u: c[2] for u, c in s["decode_population"]["workers"].items() if c[2]}
            if self.guard.hold("unknown_decode_population", bool(unknown_d), unknown_d, self.journal):
                raise ValueError("Unknown current decode population")
            residual = transition_residual(s, adjusted_loads(backend_loads(cfg["router_url"])))
            if self.guard.hold("old_role_requests", bool(residual), residual, self.journal,
                               overridable=not residual_drains(residual)):
                raise ValueError("Old role still active: " + repr(residual))
            decision = self.controller.propose(decision_samples)
        except skippable as error:
            self.controller.skip(s, str(error), self.journal)
        else:
            if time.monotonic() + self.client.timeout + 1 >= self.deadline:
                self.journal(dict(event="near_cutoff_hold_no_role_apply"))
                self.next_decision = self.deadline + 1
                return AGAIN
            self.apply(s, decision)
        self.first_decision = False
        self.samples = [row]
        self.next_decision = time.monotonic() + self.period
        return SLEEP

    def apply(self, s, decision):
        """Apply a decision. In service mode a failed or unconfirmed update is journaled and the next poll reads
        whatever the router holds: an update that did arrive is then adopted as an external change."""
        try:
            assignment = self.controller.apply(s, decision, self.client, self.journal)
        except (OSError, ValueError, RuntimeError) as error:
            if not self.service:
                raise
            self.controller.skip(s, f"role update failed: {error!r}", self.journal)
            return
        save(self.directory / "role-state.json", dict(assignment=assignment))


def apply_script(snapshot: dict, change: dict, client, journal, at_s):
    """Scripted runs only: one CAS role update with the same journal events as Controller.apply."""
    a = snapshot["assignment"]
    if snapshot.get("draining"):
        raise RuntimeError("Scripted role change while a drain is in progress")
    response = cas_update(client, a, {**a["roles"], **change}, journal,
                          lambda body: dict(scripted=True, at_s=at_s, body=body),
                          "Scripted role update unconfirmed or changed concurrently")
    journal(dict(event="role_update_confirmed", scripted=True, assignment=response["assignment"],
                 draining=response.get("draining")))
    return response["assignment"]
