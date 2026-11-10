"""The control loop against a fake router and fake engines on a virtual clock: no network, no GPU.

The router serves roles with epoch/revision compare-and-set, drains a switched worker for a few seconds, and reports
completed work that depends only on the virtual time and the roles. P is the bottleneck, so the first decision
adds a P worker.
"""
import json
import random
import threading
import types

import pytest

from evolving_infra.planner import config
from evolving_infra.planner.control import controller as controller_module
from evolving_infra.planner.control import loop as loop_module

WORKERS = [dict(node=f"node{i}", idx=i, name=f"w{i}", host=0, ip="10.0.0.1", url=f"http://10.0.0.{i}:30000",
                trace_url="http://10.0.0.1:18500", initial_role="prefill" if i < 2 else "decode") for i in range(4)]
URLS = [w["url"] for w in WORKERS]
P_MODEL = dict(coefficients=[0.09, 4.3e-5, 1.5e-10, 4e-7], tokens_per_step=4000., restored_per_cached_token=.3)
D_MODEL = dict(coefficients=[0.009, 7e-4, 0.011], draft_tokens=6, batch_support=list(range(1, 257)))
REPORT = dict(experts=256, top_k=6, fit_failures={},
              workers={w["node"]: dict(prefill=dict(origin="measured", models=[P_MODEL]),
                                       decode=dict(origin="measured", models=[D_MODEL]),
                                       kv_capacity_tokens=6461440, max_running_requests=256) for w in WORKERS})


class Clock:
    WALL0 = 1_700_000_000.

    def __init__(self):
        self.m = 1000.

    def time(self):
        return self.m + self.WALL0


class Router:
    def __init__(self, clock):
        self.clock = clock
        self.assignment = dict(epoch="router-1", revision=0, roles={w["url"]: w["initial_role"] for w in WORKERS})
        self.switched_at, self.changed, self.puts = None, [], []
        self.second = int(clock.time())
        self.work = dict(requests=0, input_tokens=0, uncached_tokens=0, causal_attention_pairs=0, output_tokens=0,
                         verified_output_tokens=0, verify_steps=0)
        self.down = False  # unreachable while True

    def advance(self):
        while self.second < int(self.clock.time()):
            rng = random.Random(self.second)
            for _ in range(2):
                u, c, o = rng.randint(4000, 12000), rng.randint(10000, 50000), rng.randint(150, 450)
                for key, value in (("requests", 1), ("input_tokens", u + c), ("uncached_tokens", u),
                                   ("causal_attention_pairs", u * c + u * (u + 1) // 2), ("output_tokens", o),
                                   ("verified_output_tokens", o), ("verify_steps", max(1, o * 2 // 7))):
                    self.work[key] += value
            self.second += 1

    def snapshot(self):
        self.advance()
        roles = self.assignment["roles"]
        out = dict(assignment=json.loads(json.dumps(self.assignment)), completed_workload=dict(self.work),
                   decode_population=dict(fields=["p_completed", "p_pending", "unknown"],
                                          workers={u: [8, 3, 0] for u, r in roles.items() if r == "decode"}))
        if self.switched_at is not None and self.clock.m - self.switched_at < 6:
            out["draining"] = {u: self.clock.m - self.switched_at for u in self.changed}
        return out

    def set_roles(self, roles, epoch=None):
        a = self.assignment
        self.changed = sorted(u for u in roles if roles[u] != a["roles"][u])
        self.assignment = dict(epoch=epoch or a["epoch"], revision=0 if epoch else a["revision"] + 1, roles=dict(roles))
        self.switched_at = self.clock.m

    def request(self, body=None):
        if self.down:
            raise ConnectionRefusedError("router down")
        if body is None:
            return self.snapshot()
        self.puts.append(body)
        a = self.assignment
        assert (body["epoch"], body["revision"], body["expected"]) == (a["epoch"], a["revision"], a["roles"])
        self.set_roles(body["desired"])
        return self.snapshot()


class SyncFuture:
    def __init__(self, fn, *a, **k):
        try:
            self.value, self.error = fn(*a, **k), None
        except Exception as error:  # noqa: BLE001
            self.value, self.error = None, error

    def done(self):
        return True

    def result(self, timeout=None):
        if self.error is not None:
            raise self.error
        return self.value


class SyncPool:
    def __init__(self, *a, **k):
        pass

    def submit(self, fn, *a, **k):
        return SyncFuture(fn, *a, **k)

    def shutdown(self, *a, **k):
        pass


class Env:
    """A run directory, the fakes, and a way to drive the loop to a virtual time."""

    def __init__(self, tmp_path, monkeypatch, mode, fit_returncode=0, **overrides):
        self.clock, self.tmp = Clock(), tmp_path
        self.router = Router(self.clock)
        self.run_dir = tmp_path / "run"
        self.run_dir.mkdir()
        (tmp_path / "token").write_text("secret\n")
        self.clock_failures = []  # monotonic times at which a clock probe fails
        self.loads_down = False
        clock, router, env = self.clock, self.router, self
        fake_time = types.SimpleNamespace(monotonic=lambda: clock.m, time=clock.time,
                                          sleep=lambda s: setattr(clock, "m", clock.m + s))

        class FakeClock:
            def __init__(self, *a):
                pass

            def observe(self):
                if any(abs(clock.m - t) < 1.5 for t in env.clock_failures):
                    raise ValueError("Host 0 clock residual exceeds 0.5 s")
                return dict(at=clock.time(), nodes=[])

        class FakeTelemetry:
            def __init__(self, *a):
                self.roles, self.error, self.halt = {}, None, threading.Event()

            def start(self):
                env.telemetry = self

        class FakePopen:
            pid = 7

            def __init__(self, argv, stdout=None, stderr=None, env=None):
                stdout.close()
                self.returncode = fit_returncode
                if not fit_returncode:
                    (self_run / "calibration" / "report.json").write_text(json.dumps(REPORT))

            def poll(self):
                return self.returncode

            def terminate(self):
                pass

        self_run = self.run_dir

        def observe_onboarding(cfg, assignment, *, start, end, mirror):
            raw = (self_run / "calibration" / "report.json").read_bytes()
            import hashlib
            nodes = sorted(w["node"] for w in WORKERS if assignment["roles"][w["url"]] == "prefill")
            return dict(workers={n: dict(node=n, start=start, end=end, restored_per_cached_token=.3,
                                         cached_tokens=1e6, tokens_per_step=3600.) for n in nodes},
                        failures={}, assignment=assignment, compute_model_sha256=hashlib.sha256(raw).hexdigest())

        def backend_loads(url):
            if env.loads_down:
                raise ConnectionRefusedError("loads endpoint down")
            return dict(workers={u: dict(active_prefill_reqs=0, unknown_prefill_reqs=0) for u in URLS})

        for module in (loop_module, controller_module):
            monkeypatch.setattr(module, "time", fake_time)
        monkeypatch.setattr(loop_module, "RoleClient", lambda url, token, timeout=5: types.SimpleNamespace(
            request=router.request, timeout=timeout))
        monkeypatch.setattr(loop_module, "worker_capacities", lambda workers: {
            w["node"]: dict(kv_capacity_tokens=6461440, max_running_requests=256, tp_size=4) for w in workers})
        monkeypatch.setattr(loop_module, "backend_loads", backend_loads)
        monkeypatch.setattr(loop_module, "Clock", FakeClock)
        monkeypatch.setattr(loop_module, "Telemetry", FakeTelemetry)
        monkeypatch.setattr(loop_module, "Mirror", lambda *a: None)
        monkeypatch.setattr(loop_module, "ThreadPoolExecutor", SyncPool)
        monkeypatch.setattr(loop_module, "observe_onboarding", observe_onboarding)
        monkeypatch.setattr(loop_module, "subprocess", types.SimpleNamespace(Popen=FakePopen, STDOUT=-2))
        self.cfg = dict(config.DEFAULTS, mode=mode, router_url="http://router", token_file=str(tmp_path / "token"),
                        run_dir=str(self.run_dir), scratch=str(tmp_path / "scratch"), hosts=[], workers=WORKERS,
                        prefill_shape_prior="dsv4-flash-h200-tp4", config_path=str(tmp_path / "controller.json"),
                        idle_after_cutoff=False, **overrides)

    def start(self):
        self.loop = loop_module.ControlLoop(self.cfg)
        return self.loop

    def drive(self, until_m, at=None):
        """Poll until virtual monotonic time `until_m`; `at` maps a time to an action run once at that time."""
        actions = dict(at or {})
        while self.clock.m < until_m:
            for t in [t for t in actions if t <= self.clock.m]:
                actions.pop(t)()
            if self.loop.poll() == loop_module.SLEEP:
                self.clock.m += self.cfg["poll_s"]

    def events(self):
        return [json.loads(line)["event"] for line in (self.run_dir / "role-decisions.jsonl").read_text().splitlines()]


def test_benchmark_run_learns_switches_and_stops_at_the_cutoff(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, "benchmark")
    (env.run_dir / "replay-timing.json").write_text(json.dumps(dict(start_wall=env.clock.time() + 10, duration_s=5400)))
    env.start().run()
    events = env.events()
    assert events[:4] == ["replay_timing", "prediction_sampling_started", "warmup_fit_started", "warmup_fit_completed"]
    assert events.count("role_update_confirmed") >= 1 and events[-1] == "measurement_cutoff_no_further_role_updates"
    assert sum(r == "prefill" for r in env.router.assignment["roles"].values()) == 3
    assert json.loads((env.run_dir / "role-controller-complete.json").read_text())["cutoff_reached"]
    assert not (env.run_dir / "PLANNER_FAILED").exists() and not (env.run_dir / "learning-start.json").exists()


def test_benchmark_stops_when_the_router_is_unreachable(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, "benchmark")
    (env.run_dir / "replay-timing.json").write_text(json.dumps(dict(start_wall=env.clock.time(), duration_s=5400)))
    env.start()
    env.router.down = True
    with pytest.raises(ConnectionRefusedError):
        env.loop.run()
    assert "router down" in (env.run_dir / "PLANNER_FAILED").read_text()


def test_service_learns_from_the_start_and_keeps_deciding_without_a_cutoff(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, "service")
    env.start()
    env.drive(1000 + 4 * 3600)
    events = env.events()
    assert events[0] == "learning_started" and "replay_timing" not in events
    start = json.loads((env.run_dir / "learning-start.json").read_text())["start_wall"]
    assert start == pytest.approx(Clock.WALL0 + 1000)
    decisions = events.count("role_update_confirmed") + events.count("role_update_hold")
    assert decisions >= (4 * 3600 - 1800) // 600 - 1  # one decision per period after learning, to the end
    assert "measurement_cutoff_no_further_role_updates" not in events


def test_service_waits_out_a_router_outage(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, "service")
    env.start()
    env.drive(1000 + 2600, at={1000 + 2400: lambda: setattr(env.router, "down", True),
                               1000 + 2460: lambda: setattr(env.router, "down", False)})
    events = env.events()
    assert events.count("router_unavailable") == 1 and events.count("router_available") == 1
    assert events.index("router_unavailable") < events.index("router_available")
    env.drive(1000 + 4000)
    assert env.events().count("role_update_hold") + env.events().count("role_update_confirmed") >= 3


def test_service_adopts_roles_changed_by_another_client(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, "service")
    env.start()
    other = {u: ("prefill" if i < 2 else "decode") for i, u in enumerate(URLS)}
    env.drive(1000 + 3000)
    env.router.set_roles(other)  # e.g. an operator restores 2P2D
    env.drive(1000 + 4500)
    events = env.events()
    assert "external_role_change_adopted" in events
    assert env.loop.controller.assignment == env.router.assignment
    after = events[events.index("external_role_change_adopted"):]
    assert "prediction_sampling_resumed" in after and ("role_update_hold" in after or "role_update_confirmed" in after)


def test_service_accepts_a_router_restart_with_the_declared_roles_before_the_fit(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, "service")
    env.start()
    roles = dict(env.router.assignment["roles"])
    env.drive(1000 + 600, at={1000 + 300: lambda: env.router.set_roles(roles, epoch="router-2")})
    assert env.events().count("router_restarted_before_fit") == 1
    env.router.set_roles({u: "prefill" if i < 3 else "decode" for i, u in enumerate(URLS)}, epoch="router-3")
    with pytest.raises(RuntimeError, match="changed before learning"):
        env.drive(1000 + 700)


def test_service_stops_when_the_fit_fails(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, "service", fit_returncode=1)
    env.start()
    with pytest.raises(RuntimeError, match="Warmup fit failed"):
        env.drive(1000 + 2000)


def test_service_skips_a_decision_when_the_router_load_endpoint_fails(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, "service")
    env.start()
    env.loads_down = True
    env.drive(1000 + 2500)
    rows = [json.loads(line) for line in (env.run_dir / "role-decisions.jsonl").read_text().splitlines()]
    skipped = [r for r in rows if r["event"] == "role_update_hold_unusable_window"]
    assert skipped and "loads endpoint down" in skipped[0]["reason"]
    env.loads_down = False
    env.drive(1000 + 3200)
    assert "role_update_confirmed" in env.events()


def test_service_restarts_the_counter_scraper_and_repeats_a_failed_probe_after_learning(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, "service", clock_probe_period_s=10_000.)  # no periodic probe in this test
    env.start()
    env.clock_failures = [1000 + 1800]  # the probe that brackets the end of learning (first poll at or after it)
    env.drive(1000 + 1500, at={1000 + 1200: lambda: setattr(env.telemetry, "error", RuntimeError("scrape thread"))})
    assert env.events().count("telemetry_restarted") == 1
    env.drive(1000 + 1900)
    rows = [json.loads(line) for line in (env.run_dir / "role-decisions.jsonl").read_text().splitlines()]
    failed = [r for r in rows if r["event"] == "clock_probe_failed"]
    assert len(failed) == 1 and failed[0]["after_learning"]
    assert [r["event"] for r in rows].count("warmup_fit_completed") == 1
    assert rows.index(failed[0]) < [r["event"] for r in rows].index("warmup_fit_started")
