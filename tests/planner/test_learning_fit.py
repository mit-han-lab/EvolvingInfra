"""The learning-end fit end to end, through the HTTP trace path: no engines, no GPU."""
import json
import math
from pathlib import Path
import random

import pytest

pytest.importorskip("scipy")

from evolving_infra.planner import config
from evolving_infra.planner.collect.trace_server import serve_traces
from evolving_infra.planner.common import save
from evolving_infra.planner.control.controller import Controller
from evolving_infra.planner.fit import learning as fit_learning

from planner_support import restore_record, router_snapshot


def free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]



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
    # Controller-side restore telemetry every 3 s, P workers only.
    rows, t = [], t0 - 15
    while t < t0 + learning + 25:
        for idx, restores in counters.items():
            rows.append(restore_record(t, sum(v for at, v in restores if at <= t), node=idx))
        t += 3
    write_jsonl(run / "restore-observations.jsonl", rows)
    # Clock: fixed reference offset 0, exchanges every 300 s bracketing the learning interval.
    save(run / "clock-reference.json", dict(at=t0 - 60, offsets_s={"0": 0.0}))
    for k in range(8):
        at_ns = int((t0 - 60 + 300 * k) * 1e9)
        save(run / "diagnostics" / f"clock-offsets-{at_ns}.json", dict(at=at_ns / 1e9, nodes=[dict(
            node=0, host="127.0.0.1", roundtrip_s=0.001, controller_before_ns=at_ns, controller_after_ns=at_ns + 10**6,
            worker_minus_controller_s=0.0, lower_s=-0.0005, upper_s=0.0005)]))
    save(run / "replay-timing.json", dict(start_wall=t0, duration_s=5400))
    save(run / "worker-capacities.json", {w["node"]: dict(kv_capacity_tokens=6461440, max_running_requests=256, tp_size=4)
                                             for w in workers})
    cfg = dict(config.DEFAULTS, mode="benchmark", prefill_shape_prior="dsv4-flash-h200-tp4",
               run_dir=str(run), scratch=str(scratch),
               hosts=[dict(index=0, ip="127.0.0.1")], workers=workers, trace_port=port)
    monkeypatch.setattr(fit_learning, "served_experts", lambda urls: (EXPERTS, TOP_K))
    server = serve_traces(str(host_root), port)
    try:
        report = fit_learning.learning_fit(cfg)
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
    urls = {w["node"]: w["url"] for w in workers}
    roles = {w["url"]: w["initial_role"] for w in workers}
    def snap(n, pop):
        work = dict(requests=n, input_tokens=n * 40000, uncached_tokens=n * 4000,
                    causal_attention_pairs=n * (4000 * 36000 + 4000 * 4001 / 2), output_tokens=n * 600,
                    verified_output_tokens=n * 600, verify_steps=n * 170)
        return router_snapshot(roles, {w["url"]: [pop, 0, 0] for w in workers if w["initial_role"] == "decode"},
                               completed=work)
    samples = [dict(at=float(3 * k), snapshot=snap(k * 2, 40)) for k in range(0, 201)]
    controller = Controller(report, urls, samples[0]["snapshot"], experts=EXPERTS, top_k=TOP_K)
    decision = controller.propose(samples)
    assert decision["rule"] == "minute_standard_error" and decision["slices"] == 10
    assert decision["action"] in ("hold", "switch")
