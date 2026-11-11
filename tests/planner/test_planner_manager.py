"""The control loop against a fake router and fake engines on a virtual clock: no network, no GPU.

The router serves roles with epoch/revision compare-and-set, drains a switched worker for a few seconds, and reports
completed work that depends only on the virtual time and the roles. Its traffic has a cold start: nothing is served
from cache for the first two minutes of traffic, the cached share rises until minute five, then stays at 40/46.
P is the bottleneck, so the first decision adds a P worker.
"""
import hashlib
import json
import random
import threading
import types

import pytest

from evolving_infra.planner.managers import planner_manager as loop_module
from evolving_infra.planner.managers import split_controller as controller_module
from evolving_infra.planner.planner_args import PlannerArgs

WORKERS = [dict(node=f"node{i}", idx=i, name=f"w{i}", host=0, ip="10.0.0.1", url=f"http://10.0.0.{i}:30000",
                trace_url="http://10.0.0.1:18500", initial_role="prefill" if i < 2 else "decode") for i in range(4)]
URLS = [w["url"] for w in WORKERS]
P_MODEL = dict(coefficients=[0.09, 4.3e-5, 1.5e-10, 4e-7], tokens_per_step=4000., restored_per_cached_token=.3)
D_MODEL = dict(coefficients=[0.009, 7e-4, 0.011], draft_tokens=6, batch_support=list(range(1, 257)))
REPORT = dict(experts=256, top_k=6, fit_failures={},
              workers={w["node"]: dict(prefill=dict(origin="measured", models=[P_MODEL]),
                                       decode=dict(origin="measured", models=[D_MODEL]),
                                       kv_capacity_tokens=6461440, max_running_requests=256) for w in WORKERS})
UNCACHED, CACHED = 6000, 40000  # tokens of one request once the cache is warm


class Clock:
    WALL0 = 1_700_000_000.

    def __init__(self):
        self.m = 1000.

    def time(self):
        return self.m + self.WALL0


class Router:
    def __init__(self, clock, traffic_from=1000.):
        self.clock, self.traffic_from = clock, traffic_from
        self.assignment = dict(epoch="router-1", revision=0, roles={w["url"]: w["initial_role"] for w in WORKERS})
        self.switched_at, self.changed, self.puts = None, [], []
        self.down = False  # unreachable while True
        self.restart_counters()

    def restart_counters(self):
        self.second = int(self.clock.time())
        self.work = dict(requests=0, input_tokens=0, uncached_tokens=0, causal_attention_pairs=0, output_tokens=0,
                         verified_output_tokens=0, verify_steps=0)

    def advance(self):
        while self.second < int(self.clock.time()):
            elapsed = self.second - (self.traffic_from + Clock.WALL0)
            if elapsed >= 0:
                warm = min(1., max(0., (elapsed - 120) / 180))
                rng = random.Random(self.second)
                for _ in range(2):
                    u, c, o = UNCACHED, int(CACHED * warm), rng.randint(150, 450)
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

    def restart(self, roles, epoch):
        """A new router process: new epoch, revision 0, counters from zero."""
        self.set_roles(roles, epoch=epoch)
        self.restart_counters()

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
    """A run directory, the fakes, and a way to drive the loop on the virtual clock."""

    def __init__(self, tmp_path, monkeypatch, fit_returncode=0, traffic_from=1000., **overrides):
        self.clock, self.tmp = Clock(), tmp_path
        self.router = Router(self.clock, traffic_from)
        self.run_dir = tmp_path / "run"
        self.run_dir.mkdir()
        (tmp_path / "token").write_text("secret\n")
        self.clock_failures = []  # monotonic times at which a clock probe fails
        self.loads_down = False
        clock, router, env, run_dir = self.clock, self.router, self, self.run_dir
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
                    (run_dir / "calibration" / "report.json").write_text(json.dumps(REPORT))

            def poll(self):
                return self.returncode

            def terminate(self):
                pass

        def observe_onboarding(cfg, assignment, *, start, end, mirror):
            raw = (run_dir / "calibration" / "report.json").read_bytes()
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
        monkeypatch.setattr(loop_module, "RoleClient", lambda url, token, journal, timeout=5: types.SimpleNamespace(
            request=router.request, timeout=timeout))
        monkeypatch.setattr(loop_module, "worker_capacities", lambda workers: {
            w["node"]: dict(kv_capacity_tokens=6461440, max_running_requests=256, tp_size=4) for w in workers})
        monkeypatch.setattr(loop_module, "backend_loads", backend_loads)
        monkeypatch.setattr(loop_module, "Clock", FakeClock)
        monkeypatch.setattr(loop_module, "Telemetry", FakeTelemetry)
        monkeypatch.setattr(loop_module, "Mirror", lambda *a: None)
        monkeypatch.setattr(loop_module, "ThreadPoolExecutor", SyncPool)
        monkeypatch.setattr(loop_module, "observe_prefill", observe_onboarding)
        monkeypatch.setattr(loop_module, "subprocess", types.SimpleNamespace(Popen=FakePopen, STDOUT=-2))
        self.args = PlannerArgs(router_url="http://router", role_control_token_file=str(tmp_path / "token"),
                                run_dir=str(run_dir), scratch_dir=str(tmp_path / "scratch"), hosts=[], workers=WORKERS,
                                prefill_shape_prior="dsv4-flash-h200-tp4", **overrides)
        self.loop = loop_module.PlannerManager(self.args)

    def drive(self, until=None, at=None, stop=None):
        """Poll every poll_s until monotonic time `until` or until `stop()` holds; `at` maps a time to an action
        run once at that time."""
        actions = dict(at or {})
        while (until is None or self.clock.m < until) and not (stop and stop()):
            for t in [t for t in actions if t <= self.clock.m]:
                actions.pop(t)()
            self.loop.poll()
            self.clock.m += self.args.router_poll_interval_secs

    def rows(self):
        return [json.loads(line) for line in (self.run_dir / "role-decisions.jsonl").read_text().splitlines()]

    def events(self):
        return [row["event"] for row in self.rows()]


def test_learns_after_the_cold_start_and_keeps_deciding(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    env.drive(1000 + 4 * 3600)
    rows = env.rows()
    started = next(r for r in rows if r["event"] == "learning_started")
    slices = [r for r in rows if r["event"] == "cold_start_slice"]
    # A minute at 0 % does not count; the share settles once the cache is warm (from minute five on).
    assert slices[0]["cached_share"] is None and slices[1]["change"] is None
    assert started["cold_start"]["reason"] == "cache_share_stable"
    assert 5 * 60 <= started["cold_start"]["waited_s"] <= 8 * 60
    assert abs(slices[-1]["change"]) <= 0.01 and slices[-1]["cached_share"] == pytest.approx(CACHED / (UNCACHED + CACHED))
    assert rows.index(started) == len(slices)  # nothing but cold-start slices before learning
    recorded = json.loads((env.run_dir / "learning-start.json").read_text())
    assert recorded["start_wall"] == started["start_wall"] and recorded["cold_start"] == started["cold_start"]
    events = env.events()
    fit = events.index("warmup_fit_completed")
    assert rows[fit]["monotonic"] >= env.loop.start_mono + 1800
    decisions = events.count("role_update_confirmed") + events.count("role_update_hold")
    assert decisions >= (4 * 3600 - started["cold_start"]["waited_s"] - 1800) // 600 - 1
    assert sum(r == "prefill" for r in env.router.assignment["roles"].values()) == 3
    assert not (env.run_dir / "PLANNER_FAILED").exists()


def test_waits_for_traffic_before_the_cold_start(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, traffic_from=1000. + 3600)
    env.drive(1000 + 3600)
    assert env.events() == [] and env.loop.start_mono is None
    env.drive(1000 + 3600 + 900)
    started = next(r for r in env.rows() if r["event"] == "learning_started")
    assert started["monotonic"] >= 1000 + 3600 + 5 * 60


def test_rising_cache_share_starts_learning_after_the_maximum_wait(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    global CACHED
    original = env.router.advance

    def rising():  # the cached share rises by more than 0.3 points every slice: the cache never fills
        global CACHED
        CACHED = 1000 + 1500 * int((env.clock.m - 1003) // 60)
        original()
    env.router.advance = rising
    try:
        env.drive(1000 + 1900)
    finally:
        CACHED = 40000
    started = next(r for r in env.rows() if r["event"] == "learning_started")
    assert started["cold_start"]["reason"] == "max_wait" and started["cold_start"]["waited_s"] == pytest.approx(1800, abs=3)


def test_waits_out_a_router_outage(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    env.drive(stop=lambda: env.loop.controller is not None)
    t = env.clock.m
    env.drive(t + 200, at={t + 60: lambda: setattr(env.router, "down", True),
                           t + 120: lambda: setattr(env.router, "down", False)})
    events = env.events()
    assert events.count("router_unavailable") == 1 and events.count("router_available") == 1
    assert events.index("router_unavailable") < events.index("router_available")
    env.drive(t + 1800)
    assert env.events().count("role_update_hold") + env.events().count("role_update_confirmed") >= 2


def test_adopts_roles_changed_by_another_client(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    env.drive(stop=lambda: "role_update_confirmed" in env.events())
    env.drive(env.clock.m + 300)
    env.router.set_roles({u: ("prefill" if i < 2 else "decode") for i, u in enumerate(URLS)})  # back to 2P2D
    env.drive(env.clock.m + 1500)
    events = env.events()
    assert "external_role_change_adopted" in events
    assert env.loop.controller.assignment == env.router.assignment
    after = events[events.index("external_role_change_adopted"):]
    assert "prediction_sampling_resumed" in after and ("role_update_hold" in after or "role_update_confirmed" in after)


def test_router_restart_restarts_the_cold_start_and_is_accepted_with_the_declared_roles(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    roles = dict(env.router.assignment["roles"])
    env.drive(1000 + 600, at={1000 + 240: lambda: env.router.restart(roles, epoch="router-2")})
    rows = env.rows()
    assert [r["event"] for r in rows].count("router_restarted_before_fit") == 1
    # The detection began again with the new counters (the engines' caches stayed warm, so it settles soon).
    assert any(r["event"] == "cold_start_slice" and r["start"] == 1000 + 243 and r["change"] is None for r in rows)
    assert next(r for r in rows if r["event"] == "learning_started")["monotonic"] > 1000 + 243
    env.router.restart({u: "prefill" if i < 3 else "decode" for i, u in enumerate(URLS)}, epoch="router-3")
    with pytest.raises(RuntimeError, match="changed before learning"):
        env.drive(env.clock.m + 60)


def test_stops_when_the_fit_fails(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, fit_returncode=1)
    with pytest.raises(RuntimeError, match="Warmup fit failed"):
        env.drive(1000 + 3000)


def test_skips_a_decision_when_the_router_load_endpoint_fails(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    env.loads_down = True
    env.drive(stop=lambda: "role_update_hold_unusable_window" in env.events())
    skipped = [r for r in env.rows() if r["event"] == "role_update_hold_unusable_window"]
    assert "loads endpoint down" in skipped[0]["reason"]
    env.loads_down = False
    env.drive(env.clock.m + 700)
    assert "role_update_confirmed" in env.events()


def test_restarts_the_counter_scraper_and_repeats_a_failed_probe_after_learning(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch, clock_probe_interval_secs=100_000.)  # no periodic probe in this test
    env.drive(stop=lambda: env.loop.start_mono is not None)
    env.clock_failures = [env.loop.learning_end]  # the probe that brackets the end of learning
    env.drive(env.clock.m + 300, at={env.clock.m + 120: lambda: setattr(env.telemetry, "error", RuntimeError("scrape"))})
    assert env.events().count("telemetry_restarted") == 1
    env.drive(env.loop.learning_end + 60)
    rows = env.rows()
    failed = [r for r in rows if r["event"] == "clock_probe_failed"]
    assert len(failed) == 1 and failed[0]["after_learning"]
    assert [r["event"] for r in rows].count("warmup_fit_completed") == 1
    assert rows.index(failed[0]) < [r["event"] for r in rows].index("warmup_fit_started")
