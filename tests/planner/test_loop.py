"""Live controller loop (evolving_infra.planner.loop) offline: no engines, no GPU."""
import gzip
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import threading
import http.server

import pytest

pytest.importorskip("scipy")

from evolving_infra.planner import loop as md
from evolving_infra.planner.priors import PREFILL_SHAPE_PRIORS


def free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ------------------------------------------------------------------------------------------- small pieces

def test_prior_is_the_median_of_measured_four_term_models():
    def worker(origin, coefficients, method="per_node_four_term_nnls"):
        return dict(prefill=dict(origin=origin, models=[dict(method=method, coefficients=coefficients)]))
    report = dict(workers=dict(node0=worker("measured", [1, 10, 100, 1000]), node1=worker("measured", [3, 30, 300, 3000]),
                               node2=worker("measured", [2, 20, 200, 2000]),
                               node3=worker("same_run_role_prior", [9, 9, 9, 9]),
                               node4=worker("measured", [7, 7, 7, 7], "prior_shape_current_node_scale_v1")))
    assert md.build_prior(report) == [2, 20, 200, 2000]
    prior = PREFILL_SHAPE_PRIORS[md.DEFAULTS["prefill_shape_prior"]]
    assert len(prior["coefficients"]) == 4 and all(c > 0 for c in prior["coefficients"])
    assert md.prior_sha256(prior) == md.prior_sha256(json.loads(json.dumps(prior)))


def test_prior_resolution_rejects_unknown_changed_or_malformed_priors():
    cfg = dict(md.DEFAULTS)
    assert md.resolve_prior(cfg) is PREFILL_SHAPE_PRIORS["dsv4-flash-h200-tp4"]
    inline = dict(coefficients=[0.1, 4e-5, 1.5e-10, 4e-7])
    assert md.resolve_prior(dict(cfg, prefill_shape_prior=inline)) == inline
    assert md.resolve_prior(dict(cfg, prefill_shape_prior=inline, prefill_shape_prior_sha256=md.prior_sha256(inline)))
    with pytest.raises(ValueError, match="changed"):
        md.resolve_prior(dict(cfg, prefill_shape_prior_sha256="0" * 64))
    with pytest.raises(ValueError, match="Unknown"):
        md.resolve_prior(dict(cfg, prefill_shape_prior="nope"))
    with pytest.raises(ValueError, match="coefficients"):
        md.resolve_prior(dict(cfg, prefill_shape_prior=dict(coefficients=[1, 2])))


def test_load_config_fills_defaults_and_records_its_path(tmp_path):
    path = tmp_path / "controller.json"
    path.write_text(json.dumps(dict(period_s=300)))
    cfg = md.load_config(path)
    assert cfg["period_s"] == 300 and cfg["learning_s"] == md.DEFAULTS["learning_s"]
    assert cfg["config_path"] == str(path.resolve())


def test_cli_help_lists_every_command():
    out = subprocess.run([sys.executable, "-m", "evolving_infra.planner", "--help"], check=True,
                         capture_output=True, text=True).stdout
    for command in ("control", "fit", "serve-traces", "prior", "calibrate", "node-plan"):
        assert command in out


def test_transition_guard_bounds_holds_but_never_overrides_drains():
    now = [0.]
    guard, rows = md.HoldGuard(max_wait_s=120, clock=lambda: now[0]), []
    assert guard.hold("x", True, {}, rows.append)
    now[0] = 121
    assert not guard.hold("x", True, {}, rows.append) and rows[-1]["event"] == "transition_guard_max_wait"
    assert guard.hold("d", True, {}, rows.append, overridable=False)
    now[0] = 1000
    assert guard.hold("d", True, {}, rows.append, overridable=False)
    loads = md.adjusted_loads(dict(workers={"u": dict(active_prefill_reqs=3, unknown_prefill_reqs=2)}))
    assert loads["workers"]["u"]["active_prefill_reqs"] == 1
    assert md.residual_drains({"u": dict(draining_age_s=1)})


def test_mirror_appends_only_new_bytes(tmp_path):
    root = tmp_path / "host"
    (root / "w00").mkdir(parents=True)
    trace = root / "w00" / "iter_trace.dp0_pp0.pid7.jsonl"
    req = root / "w00" / "req_trace.dp0_pp0.pid7.jsonl"
    trace.write_bytes(b'{"a":1}\n{"a":2')
    req.write_bytes(b"")
    port = free_port()
    server = md.serve_traces(str(root), port)
    try:
        worker = dict(node="node0", name="w00", trace_url=f"http://127.0.0.1:{port}")
        mirror = md.Mirror(tmp_path / "mirror", [worker])
        paths = mirror.sync(["node0"])["node0"]
        with trace.open("ab") as f:
            f.write(b'}\n{"a":3}\n')
        mirror.sync(["node0"])
        assert Path(paths["steps"]).read_bytes() == trace.read_bytes()
        (root / "w00" / "iter_trace.dp0_pp0.pid8.jsonl").write_bytes(b"")  # engine restarted
        with pytest.raises(ValueError, match="expected one iter"):
            mirror.sync(["node0"])
        assert md.http_json(f"http://127.0.0.1:{port}/time")["time_ns"] > 0
    finally:
        server.shutdown()


def test_telemetry_rows_parse_as_restore_counters(tmp_path):
    from evolving_infra.planner.calibration.restore import read_restore_counters
    body = ("# HELP sglang:hicache_backup_bytes_total Bytes.\n# TYPE x counter\n"
            'sglang:load_back_tokens_total{model_name="m",pool="kv"} 400.0\n'
            'sglang:load_back_tokens_total{model_name="m",pool="swa"} 8.0\nsglang:other 1\n').encode()

    class Metrics(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Metrics)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        t = md.Telemetry([dict(idx=3, url=f"http://127.0.0.1:{server.server_port}")], tmp_path / "r.jsonl", 3)
        rows = [t.scrape(t.workers[0]) for _ in range(2)]
    finally:
        server.shutdown()
    with gzip.open(tmp_path / "r.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    meters = read_restore_counters(tmp_path / "r.jsonl.gz", 4)
    assert [v for _, v in meters["node3"]] == [100.0, 100.0]  # kv pool only, / 4 TP ranks


# ------------------------------------------------------------------------------------------- fit end to end

TRUE_P = [0.09, 4.3e-5, 1.5e-10, 4.0e-7]  # s/step, s/new token, s/pair, s/restored token
TRUE_D = [0.009, 7.0e-4, 0.011]  # s, s/request, s * expert fraction
EXPERTS, TOP_K, DRAFT = 256, 6, 6


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def synth_prefill(rng, t0, t1, replicas=4):
    """Busy episodes of one-request chunked forwards with cached prefixes and host restores."""
    steps, reqs, restores = [], [], []
    t, it = t0, 0
    while t < t1:
        for k in range(rng.randint(3, 10)):
            n, h = rng.randint(512, 8192), rng.choice([0, rng.randint(1024, 65536)])
            restored = int(h * rng.uniform(0.0, 0.6))
            pairs = n * h + n * (n + 1) // 2
            dur = (TRUE_P[0] + TRUE_P[1] * n + TRUE_P[2] * pairs + TRUE_P[3] * restored) * rng.uniform(0.98, 1.02)
            it += 1
            steps.append(dict(forward_iter=it, ts=t + dur, launch_mono=t, dur_ms=dur * 1e3, mode="extend", bs=1,
                              after_idle=k == 0, new_prefill_tokens=n, decode_tokens=0, spec_num_draft_tokens=None,
                              launch_bs=1, launch_seq_lens_sum=n + h, prefill_attention_pairs=pairs,
                              launch_context_source="batch_sum"))
            reqs.append(dict(rid=f"r{it}", ts=t + dur, input_len=n + h, output_len=1, cached_tokens=h,
                             prefill_iters=[it], requested_output_len=1, decode_ct=0, spec_verify_ct=0))
            restores.append((t, restored * replicas))
            t += dur
        t += rng.uniform(0.3, 2.0)
    return steps, reqs, restores


def synth_decode(rng, t0, t1):
    steps, t, it = [], t0, 0
    while t < t1:
        b, first = rng.choice([4, 8, 16, 32, 64, 96, 128, 192]), True
        stop = t + rng.uniform(20, 60)
        while t < stop:
            frac = -math.expm1(b * DRAFT * math.log1p(-TOP_K / EXPERTS))
            dur = (TRUE_D[0] + TRUE_D[1] * b + TRUE_D[2] * frac) * rng.uniform(0.98, 1.02)
            it += 1
            steps.append(dict(forward_iter=it, ts=t + dur, launch_mono=t, dur_ms=dur * 1e3, mode="decode", bs=b,
                              after_idle=first, new_prefill_tokens=0, decode_tokens=int(b * 3.5),
                              spec_num_draft_tokens=DRAFT, launch_bs=b, launch_seq_lens_sum=b * 30000,
                              prefill_attention_pairs=None, launch_context_source="batch_sum"))
            t, first = t + dur, False
        t += rng.uniform(1, 5)
    return steps


def test_fit_recovers_dsv4_models_through_the_http_trace_path(tmp_path, monkeypatch):
    rng = random.Random(7)
    t0, learning = 1_700_000_000.0, 1800
    host_root, run, scratch = tmp_path / "host", tmp_path / "run", tmp_path / "scratch"
    port = free_port()
    workers = [dict(node=f"node{i}", idx=i, name=f"w0{i}", host=0, ip="127.0.0.1", url=f"http://127.0.0.1:{30000 + i}",
                    trace_url=f"http://127.0.0.1:{port}", initial_role="prefill" if i < 2 else "decode")
               for i in range(4)]
    counters = {}
    for w in workers:
        folder = host_root / w["name"]
        if w["initial_role"] == "prefill":
            steps, reqs, restores = synth_prefill(rng, t0 - 20, t0 + learning + 30)
            counters[w["idx"]] = restores
        else:
            steps, reqs = synth_decode(rng, t0 - 20, t0 + learning + 30), []
        write_jsonl(folder / "iter_trace.dp0_pp0.pid100.jsonl", steps)
        write_jsonl(folder / "req_trace.dp0_pp0.pid100.jsonl", reqs)
    # Controller-side restore telemetry every 3 s (Telemetry's record format), P workers only.
    rows, t = [], t0 - 15
    while t < t0 + learning + 25:
        for idx, restores in counters.items():
            value = sum(v for at, v in restores if at <= t)
            rows.append(dict(at=t + 0.01, nodes=[dict(node=idx, role="prefill", at=t, errors={}, metrics=(
                "# HELP sglang:hicache_backup_bytes_total Bytes.\n"
                f'sglang:load_back_tokens_total{{model_name="m",pool="kv"}} {value}\n'))]))
        t += 3
    write_jsonl(run / "restore-observations.jsonl", rows)
    # Clock: fixed reference offset 0, exchanges every 300 s bracketing the learning interval.
    md.save(run / "clock-reference.json", dict(at=t0 - 60, offsets_s={"0": 0.0}))
    for k in range(8):
        at_ns = int((t0 - 60 + 300 * k) * 1e9)
        md.save(run / "diagnostics" / f"clock-offsets-{at_ns}.json", dict(at=at_ns / 1e9, nodes=[dict(
            node=0, host="127.0.0.1", roundtrip_s=0.001, controller_before_ns=at_ns, controller_after_ns=at_ns + 10**6,
            worker_minus_controller_s=0.0, lower_s=-0.0005, upper_s=0.0005)]))
    md.save(run / "replay-timing.json", dict(start_wall=t0, duration_s=5400))
    md.save(run / "worker-capacities.json", {w["node"]: dict(kv_capacity_tokens=6461440, max_running_requests=256)
                                             for w in workers})
    cfg = dict(md.DEFAULTS, run_dir=str(run), scratch=str(scratch),
               hosts=[dict(index=0, ip="127.0.0.1")], workers=workers, trace_port=port)
    monkeypatch.setattr(md, "model_experts", lambda urls: (EXPERTS, TOP_K))
    server = md.serve_traces(str(host_root), port)
    try:
        report = md.fit(cfg)
    finally:
        server.shutdown()
    assert report["fit_failures"] == {} and (run / "calibration" / "report.json").exists()
    assert (report["experts"], report["top_k"]) == (EXPERTS, TOP_K)
    for node in ("node0", "node1"):
        p = report["workers"][node]["prefill"]
        assert p["origin"] == "measured" and p["models"][0]["method"] == "per_node_four_term_nnls"
        coef = p["models"][0]["coefficients"]
        # 60-s window sums identify the step and token terms; pair and restore trade off against them,
        # so check those through the held-out window predictions.
        assert all(abs(c / t - 1) < 0.15 for c, t in zip(coef[:2], TRUE_P[:2])), coef
        errors = p["models"][0]["errors"]
        assert errors["total_relative"] < 0.05 and errors["p95_relative"] < 0.15, errors
        assert report["workers"][node]["decode"]["origin"] == "same_run_role_prior"
    for node in ("node2", "node3"):
        d = report["workers"][node]["decode"]
        assert d["origin"] == "measured" and d["models"][0]["draft_tokens"] == DRAFT
        assert all(abs(c / t - 1) < 0.1 for c, t in zip(d["models"][0]["coefficients"], TRUE_D)), d["models"][0]
    # The controller accepts this fit for the 16-worker-style node map and evaluates a decision.
    from evolving_infra.planner.calibration.controller import Controller
    urls = {w["node"]: w["url"] for w in workers}
    roles = {w["url"]: w["initial_role"] for w in workers}
    def snap(n, pop):
        work = dict(requests=n, input_tokens=n * 40000, uncached_tokens=n * 4000,
                    causal_attention_pairs=n * (4000 * 36000 + 4000 * 4001 / 2), output_tokens=n * 600,
                    verified_output_tokens=n * 600, verify_steps=n * 170)
        return dict(assignment=dict(epoch="e", revision=0, roles=roles), completed_workload=work,
                    decode_population=dict(fields=["p_completed", "p_pending", "unknown"],
                                           workers={w["url"]: [pop, 0, 0] for w in workers if w["initial_role"] == "decode"}))
    samples = [dict(at=float(3 * k), snapshot=snap(k * 2, 40)) for k in range(0, 201)]
    controller = Controller(report, urls, samples[0]["snapshot"], experts=EXPERTS, top_k=TOP_K)
    decision = controller.propose(samples)
    assert decision["rule"] == "minute_standard_error" and decision["slices"] == 10
    assert decision["action"] in ("hold", "switch")
