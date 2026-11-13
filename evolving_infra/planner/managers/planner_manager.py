# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""The planner's main loop: wait out the cold start, fit each worker once after learning, then move P/D roles.

From the first completed request the planner waits until the share of input tokens served from cache settles
(``managers.cold_start``). A learning interval (``--learning-secs``, 1800 s) then ends in one fit per worker
(``perf_model.calibration``); every decision period (``--decision-period-secs``, 600 s) after it, every other P count
is a candidate, scored slice by slice (``--slice-secs``, 60 s; ``policy.split_policy``), and the planner switches only
when the mean gain exceeds ``--switch-standard-errors`` (2) standard errors. Decision samples within
``--post-switch-exclusion-secs`` (120 s) after a confirmed switch are skipped, and the P workers' restore ratio and
packing are re-observed before each decision (``perf_model.prefill_observation``). A role change is one Router role
assignment (``router_client``). No engine is restarted. The planner runs until it is stopped.

Deployment contract (every value is a planner argument, ``planner_args``; see ``docs/dynamic-pd.md``):

- every worker host runs ``python -m evolving_infra.planner serve-traces`` over the engines'
  ``--sched-trace-dir`` parent; the planner copies the iteration/request traces over HTTP and measures each host
  clock with ``GET /time`` (no shared filesystem or SSH needed);
- worker identity is ``nodeN``; each worker has ``url`` (engine), ``trace_url`` (its host's trace
  server), ``name`` (its trace subdirectory), ``host`` (host index) and ``initial_role``, and the router
  starts with these initial roles;
- restore counters (``sglang:load_back_tokens_total{pool="kv"}``) are scraped from the current P
  workers; every TP rank increments the counter, so values are divided by the engines' TP size;
- the resolved arguments go to ``<run_dir>/planner_args.json`` and the learning start to
  ``<run_dir>/learning-start.json``; the learning-end fit subprocess reads both;
- the planner stops, with ``<run_dir>/PLANNER_FAILED``, only when it cannot continue: the router's worker set
  differs from the arguments, the roles before the fit differ from the declared initial roles, or the learning-end
  fit fails. Every other error (router or engine unreachable, a failed clock probe, the restore-counter scraper
  stopping, a failed or unconfirmed role update) is journaled and skips at most one decision; a role assignment
  changed outside the planner after the fit is adopted;
- ``--role-script`` (testing only) replaces decisions with role changes at the given seconds after the learning
  start, all after the fit.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evolving_infra.planner.engine_client import worker_capacities
from evolving_infra.planner.managers.cold_start import ColdStart
from evolving_infra.planner.managers.split_controller import SplitController, node_assignment
from evolving_infra.planner.managers.transition_guard import (TransitionGuard, adjusted_loads, residual_drains,
                                                              transition_residual)
from evolving_infra.planner.observability.clock_sync import Clock
from evolving_infra.planner.observability.restore_metrics import Telemetry
from evolving_infra.planner.observability.trace_mirror import Mirror
from evolving_infra.planner.perf_model.prefill_observation import observe_prefill
from evolving_infra.planner.router_client import RoleClient, backend_loads, cas_update
from evolving_infra.planner.utils import save


def run_planner(args):
    """Run the planner until it is stopped (see the module docstring)."""
    PlannerManager(args).run()


class PlannerManager:
    """One planner run. Every router poll interval it reads the router's roles and completed workload. Once the cold
    start is over, learning starts; during its last decision period and afterwards the observations are the decision
    samples. When learning ends it runs the fit once (a subprocess); then every decision period it re-observes the P
    workers and decides."""

    def __init__(self, args):
        self.args = args.check()
        print(f"planner_args={args}", file=sys.stderr, flush=True)
        self.directory, self.scratch = Path(args.run_dir), Path(args.scratch_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        args.save(args.args_path)  # the fit subprocess reads the same arguments
        self.period, self.learning = args.decision_period_secs, args.learning_secs
        self.script = args.sorted_role_script()
        self.scripted = bool(self.script)  # scripted: timed role changes replace decisions for the whole run
        token = Path(args.role_control_token_file).read_text().strip()
        self.node_urls = {w["node"]: w["url"] for w in args.workers}
        self.initial_roles = {w["url"]: w["initial_role"] for w in args.workers}
        save(self.directory / "worker-capacities.json", worker_capacities(args.workers))
        self.clock = Clock(args, self.directory)
        self.start_telemetry(self.initial_roles)
        self.mirror = Mirror(self.scratch / "mirror", args.workers)
        self.log = (self.directory / "role-decisions.jsonl").open("x", buffering=1)
        self.observations = (self.directory / "role-observations.jsonl").open("x", buffering=1)
        self.client = RoleClient(args.router_url, token, self.journal)
        self.clock_pool, self.onboarding_pool = ThreadPoolExecutor(1), ThreadPoolExecutor(1)
        self.clock_pool.submit(self.clock.observe).result()
        self.clock_task, self.next_clock = None, time.monotonic() + args.clock_probe_interval_secs
        # Cold start, then the learning interval on the monotonic clock.
        self.cold_start = ColdStart(args.slice_secs, args.cold_start_stable_change, args.cold_start_max_wait_secs)
        self.start_mono = self.learning_end = None
        # Learning end: one clock probe after the end, then the fit subprocess, then the controller.
        self.learning_end_probe, self.learning_end_task, self.fit_proc, self.controller = False, None, None, None
        # Decision samples and schedule.
        self.samples, self.prediction_initial, self.first_assignment = [], None, None
        self.next_decision, self.first_decision = None, True
        self.onboarding_task = self.onboarding_samples = None
        self.guard = TransitionGuard(args.transition_hold_max_secs)
        self.router_down = False

    def journal(self, value):
        self.log.write(json.dumps(dict(at=time.time(), monotonic=time.monotonic(), **value)) + "\n")

    def start_telemetry(self, roles):
        self.telemetry = Telemetry(self.args.workers, self.directory / "restore-observations.jsonl",
                                   self.args.restore_scrape_interval_secs)
        self.telemetry.roles = dict(roles)
        self.telemetry.start()

    def run(self):
        try:
            while True:
                self.poll()
                time.sleep(self.args.router_poll_interval_secs)
        except BaseException as error:
            (self.directory / "PLANNER_FAILED").write_text(f"planner controller: {error!r}")
            self.journal(dict(event="controller_failed", error=repr(error)))
            raise
        finally:
            if self.fit_proc is not None and self.fit_proc.poll() is None:
                self.fit_proc.terminate()
            self.telemetry.halt.set()
            self.clock_pool.shutdown(wait=False, cancel_futures=True)
            self.onboarding_pool.shutdown(wait=False, cancel_futures=True)
            self.log.close()
            self.observations.close()

    def poll(self):
        """One observation of the router, and whatever is due after it."""
        now = time.monotonic()
        self.check_background(now)
        s = self.read_router()
        if s is None:
            return
        node_assignment(s, self.node_urls)
        self.telemetry.roles = dict(s["assignment"]["roles"])
        self.check_initial_assignment(s)
        row = dict(at=time.monotonic(), wall=time.time(), snapshot=s)
        self.observations.write(json.dumps(row) + "\n")
        if self.start_mono is None:
            self.watch_cold_start(row)
        if self.start_mono is not None and row["at"] >= self.learning_end - self.period and not self.scripted:
            self.add_sample(s, row)
        if self.start_mono is not None and now >= self.learning_end and self.controller is None:
            self.finish_learning(s)
        if self.controller is not None and self.script and now >= self.start_mono + self.script[0][0]:
            at_s, change = self.script.pop(0)
            apply_script(s, {self.node_urls[n]: r for n, r in change.items()}, self.client, self.journal, at_s)
        elif (self.controller is not None and not self.scripted and now >= self.next_decision
              and len(self.samples) >= 2):
            self.decide(s, row)

    def read_router(self):
        """The router's roles and completed workload; None while the router is unreachable."""
        try:
            s = self.client.request()
        except (OSError, ValueError) as error:  # unreachable, an HTTP error, or a reply that is not JSON
            if not self.router_down:
                self.journal(dict(event="router_unavailable", error=repr(error)))
            self.router_down = True
            return None
        if self.router_down:
            self.journal(dict(event="router_available"))
            self.router_down = False
        return s

    def check_background(self, now):
        """Restart a stopped restore-counter scraper; collect the periodic clock probe and start the next one."""
        if self.telemetry.error is not None:
            self.journal(dict(event="telemetry_restarted", error=repr(self.telemetry.error)))
            self.start_telemetry(self.telemetry.roles)
        if self.clock_task is not None and self.clock_task.done():
            self.collect_clock_probe()
        if now >= self.next_clock and self.clock_task is None:
            self.clock_task, self.next_clock = (self.clock_pool.submit(self.clock.observe),
                                                now + self.args.clock_probe_interval_secs)

    def collect_clock_probe(self):
        """Wait for the pending clock probe. A failed probe is journaled; a failed probe after learning is repeated
        before the fit, whose snapshot needs a probe after the learning interval."""
        task, self.clock_task = self.clock_task, None
        try:
            task.result()
        except Exception as error:  # noqa: BLE001 - any probe failure
            self.journal(dict(event="clock_probe_failed", error=repr(error), after_learning=task is self.learning_end_task))
            if task is self.learning_end_task:
                self.learning_end_probe = False

    def watch_cold_start(self, row):
        """Journal each closed one-minute slice of the cold start; start learning when it is over."""
        slice_ = self.cold_start.observe(row["at"], row["snapshot"]["completed_workload"])
        if slice_ is not None:
            self.journal(dict(event="cold_start_slice", **slice_))
        done = self.cold_start.done
        if done is None:
            return
        self.start_mono, self.learning_end = row["at"], row["at"] + self.learning
        save(self.args.learning_start_path, dict(start_wall=row["wall"], cold_start=done))
        self.journal(dict(event="learning_started", start_wall=row["wall"], cold_start=done,
                          learning_end_monotonic=self.learning_end))

    def check_initial_assignment(self, s):
        """The router starts with the declared roles, and nothing changes them before the fit (each worker is
        measured in its initial role). A router restart that keeps those roles is accepted."""
        if self.first_assignment is None:
            if s["assignment"]["roles"] != self.initial_roles:
                raise RuntimeError("Initial roles do not match the declared configuration")
            self.first_assignment = s["assignment"]
            save(self.directory / "role-controller-ready.json", dict(
                at=time.time(), initial_assignment=self.first_assignment, first_clock_probe_complete=True))
        if self.controller is None and s["assignment"] != self.first_assignment:
            if s["assignment"]["roles"] != self.initial_roles:
                raise RuntimeError("Initial roles/incarnation changed before learning")
            self.journal(dict(event="router_restarted_before_fit", before=self.first_assignment,
                              after=s["assignment"]))
            self.first_assignment = s["assignment"]

    def add_sample(self, s, row):
        """Raw observations stay complete; only prediction samples skip the post-switch exclusion."""
        if self.controller is not None and s["assignment"] != self.controller.assignment:
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
        args, directory = self.args, self.directory
        if not self.learning_end_probe:
            if self.clock_task is not None:
                self.collect_clock_probe()
            self.clock_task = self.learning_end_task = self.clock_pool.submit(self.clock.observe)
            self.learning_end_probe = True
        elif self.fit_proc is None and self.clock_task is None:
            (directory / "calibration").mkdir(exist_ok=True)
            self.fit_proc = subprocess.Popen(
                [sys.executable, "-m", "evolving_infra.planner", "fit", "--config", str(args.args_path)],
                stdout=(directory / "calibration" / "fit.log").open("w"), stderr=subprocess.STDOUT,
                env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
            self.journal(dict(event="warmup_fit_started", pid=self.fit_proc.pid))
        elif self.fit_proc is not None and self.fit_proc.poll() is not None:
            if self.fit_proc.returncode:
                raise RuntimeError("Warmup fit failed; see calibration/fit.log")
            raw = (directory / "calibration" / "report.json").read_bytes()
            fitted = json.loads(raw)
            self.controller = SplitController(fitted, self.node_urls, self.prediction_initial or s,
                                         experts=fitted["experts"], top_k=fitted["top_k"],
                                         fit_digest=hashlib.sha256(raw).hexdigest(),
                                         exclusion_s=args.post_switch_exclusion_secs, slice_secs=args.slice_secs,
                                         switch_standard_errors=args.switch_standard_errors,
                                         observation_gap_max_secs=args.observation_gap_max_secs)
            self.journal(dict(event="warmup_fit_completed", learning_s=self.learning,
                              fit_failures=fitted["fit_failures"],
                              origins={n: {r: v[r]["origin"] for r in ("prefill", "decode")}
                                       for n, v in fitted["workers"].items()},
                              model_ready_monotonic=time.monotonic()))

    def decide(self, s, row):
        """A periodic decision: re-observe the P workers, check the transition holds, then propose and apply.
        An unreachable endpoint or a failed update skips this decision instead of stopping the controller."""
        args = self.args
        decision_samples = self.samples
        if not args.disable_prefill_observation:
            if self.onboarding_task is None:
                self.onboarding_samples = list(self.samples)
                self.onboarding_task = self.onboarding_pool.submit(
                    observe_prefill, args, s["assignment"], start=self.samples[0]["wall"],
                    end=self.samples[-1]["wall"], mirror=self.mirror)
            if not self.onboarding_task.done():
                return
            try:
                self.controller.update_onboarding(s, self.onboarding_task.result(), self.journal)
            except Exception as error:  # noqa: BLE001 - the decision keeps the last valid observations
                self.journal(dict(event="onboarding_update_unavailable", reason=str(error)))
            self.onboarding_task, decision_samples, self.onboarding_samples = None, self.onboarding_samples, None
        try:
            unknown_d = {u: c[2] for u, c in s["decode_population"]["workers"].items() if c[2]}
            if self.guard.hold("unknown_decode_population", bool(unknown_d), unknown_d, self.journal):
                raise ValueError("Unknown current decode population")
            residual = transition_residual(s, adjusted_loads(backend_loads(args.router_url)))
            if self.guard.hold("old_role_requests", bool(residual), residual, self.journal,
                               overridable=not residual_drains(residual)):
                raise ValueError("Old role still active: " + repr(residual))
            decision = self.controller.propose(decision_samples)
        except (ValueError, OSError) as error:
            self.controller.skip(s, str(error), self.journal)
        else:
            self.apply(s, decision)
        self.first_decision = False
        self.samples = [row]
        self.next_decision = time.monotonic() + self.period

    def apply(self, s, decision):
        """Apply a decision. A failed or unconfirmed update is journaled and the next poll reads whatever the
        router holds: an update that did arrive is then adopted as an external change."""
        try:
            assignment = self.controller.apply(s, decision, self.client, self.journal)
        except (OSError, ValueError, RuntimeError) as error:
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
