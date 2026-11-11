"""Commands and configs of the reference benchmark stack (pure functions, no Modal, no I/O).

One 8-GPU node holds two TP-4 DecoupledEngine workers in symmetric mode. Worker `idx` lives on node
idx // 2, slot idx % 2; slot s shifts every per-worker port by 1000 * s. The first `n_p` workers start
as prefill (P), the rest as decode (D). Around them: one frontend (T) per Trie process, the Dynamo
KV indexer, the Rust router and, in dynamic runs, the planner's live controller plus a trace server
on every node.

The settings are those of the validated H200 runs (benchmarks/reproduce) with KV session
retention off; `bench_config` returns them as one JSON-able dict that `node.py` reads and the run
directory keeps.
"""
from __future__ import annotations

import json
import math

MODEL = ("/weights/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/"
         "7872f01b1d1fe23eabc4c98b48bffcef5a386062")
SERVED = "deepseek-v4-flash"
# Claude Code (CC) semi-synthetic workload of the reference runs, with its offline validation report
# beside it (<workload>.validate.json).
WORKLOAD = "/inputs/workloads/cc-semi-062126-kh20261001.v3.jsonl"

REPO = "/opt/evolving-infra"
REPLAY_DIR = f"{REPO}/benchmarks/replay"
TRIE_DIR = f"{REPO}/benchmarks/trie"
SERVICE_PY = "/opt/indexer-venv/bin/python"  # frontend, indexer, planner
TRIE_PY = "/opt/trie-venv/bin/python"
ROUTER_BIN = "/usr/local/bin/sgl-router"

ROUTER_PORT = 30900
ROUTER_URL = f"http://127.0.0.1:{ROUTER_PORT}"
CONFIG_ID = "dsv4"
INDEXER_ID_BASE = 20001
TRACE_PORT = 18500  # planner serve-traces, one per node
TRACE_SUBDIR = "sched-trace"
FRONTEND_PORT_BASE = 28100  # clear of the peer-KV servicer ports 18100-18103 / 19100-19103

# The engine settings of the CC baseline configuration (concurrency 384), then the method's overrides
# (memory fraction 0.71: 0.74 runs out of memory in DSV4 sparse prefill on this engine; host tier =
# baseline ratio 4 / 0.9 because the peer-KV servicer withholds 10 % of it).
ENGINE = dict(
    mem_fraction_static=0.71, max_running_requests=256, chunked_prefill_size=8192, swa_full_tokens_ratio=0.1,
    hicache_ratio=math.ceil(4 / 0.9 * 1000) / 1000, dist_timeout_s=1800, oracle_accept_rate=0.6,
    peer_kv_reserve_ratio_kv=0.10, peer_kv_max_leased_gb=64,
    late_d=dict(mode="choose", lead_mode="need", allocation_trigger="first_chunk", control_port_base=21776),
)

# Engine environment. Kept names: SGLANG_* (see the root README). Session-retention words
# (SGLANG_KV_SESSION_RETENTION_*, host-evict fraction) are left out: retention is off and the engine
# ignores them; the session ADMIT tag belongs to late D and stays.
ENGINE_ENV = {
    # A D declares a P lost after 6 failed bootstrap heartbeats 5 s apart instead of 2 (a worker exit
    # can freeze a Modal node for ~16 s).
    "SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL": "5", "SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE": "6",
    # Symmetric workers cache and hand over like standard P and D.
    "SGLANG_SYMMETRIC_PUBLISH_AFTER_RESULT": "1", "SGLANG_SYMMETRIC_BUDGET_STANDARD_PD": "1",
    "SGLANG_SYMMETRIC_PD_IMPORT_LOCAL_RESERVE": "1", "SGLANG_SYMMETRIC_PD_IMPORT_BUDGET_FRACTION": "1.0",
    "SGLANG_SYMMETRIC_RELEASE_INSERT_CHUNKED": "1", "SGLANG_SYMMETRIC_SKIP_REUSED_PREFIX_PUBLICATION": "1",
    # Late D: admit after launch needs late D's claim reserve; consumer clear and ADMIT tag belong to it.
    "SGLANG_SYMMETRIC_ADMIT_AFTER_LAUNCH": "1", "SGLANG_HICACHE_DECODE_CLEAR_CONSUMER": "1",
    "SGLANG_KV_SESSION_ADMIT_TAG": "1",
    "SGLANG_LATE_D_CATCHUP_TOKENS_PER_PASS": "78000", "SGLANG_LATE_D_PARK_CAP": "0.35",
    "SGLANG_LATE_D_PARK_EXIT": "abort", "SGLANG_LATE_D_PARK_TIMEOUT_S": "300",
    "SGLANG_LATE_D_WARM_HOLD_MIN_TOKENS": "0", "SGLANG_LATE_D_BIND_MARGIN": "1",
    "SGLANG_DECODE_SELECTION_METRICS": "1",
    # Host tier and KV events.
    "SGLANG_PEER_KV_PUSH_FAST": "1", "SGLANG_PREBUILT_NONBLOCKING_H2D": "1",
    "SGLANG_HICACHE_PIPELINED_SYNC": "1", "SGLANG_HICACHE_SKIP_IDLE_SYNC": "1",
    "SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB": "16",
    "SGLANG_KV_EVENTS_LAZY_LOAD_BACK": "1", "SGLANG_KV_EVENTS_LAZY_FAST_BUILD": "1",
    "SGLANG_KV_EVENTS_BATCH_PUBLISH": "1", "SGLANG_KV_EVENTS_BATCH_MAX_BYTES": "524288",
    "SGLANG_KV_EVENTS_BATCH_MAX_EVENTS": "0", "SGLANG_KV_EVENTS_PUBLISH_STATS": "1",
    "SGLANG_SCHED_PHASE_TIMERS": "0",
    # The P routing snapshot the router's cost function reads, sampled by TP0 every 0.1 s.
    "SGLANG_SYMMETRIC_ROUTING_SNAPSHOT": "1", "SGLANG_PREFILL_ROUTING_SNAPSHOT_INTERVAL_S": "0.1",
    "SGLANG_PREFILL_ROUTING_SNAPSHOT_MAX_RECORDS": "256", "SGLANG_PREFILL_ROUTING_SNAPSHOT_MAX_BYTES": "98304",
    "SGLANG_PREFILL_ROUTING_SNAPSHOT_MAX_AGE_S": "1", "SGLANG_PREFILL_TIMELINE_SAMPLES": "1",
    "SGLANG_LOAD_SNAPSHOT_SLOT_BYTES": "131072",
}

ROUTER_ARGS = [
    "--policy", "symmetric_pd", "--symmetric-pd-force-split",
    "--symmetric-pd-p-select", "cost-function", "--symmetric-pd-d-select", "round-robin",
    "--symmetric-pd-import-cache-policy", "retain",
    "--symmetric-pd-load-poll-ms", "100", "--symmetric-pd-pull-hint", "--kv-indexer-timeout-ms", "200",
    "--request-timeout-secs", "3600", "--stale-request-timeout-secs", "3600", "--cb-threshold", "4294967295",
    "--co-schedule-trade", "--co-schedule-trade-copy-lambda", "4.6", "--co-schedule-trade-count-inflight",
    "--co-schedule-uncached-balance", "--co-schedule-load-poll-ms", "100",
    "--kv-pull-hint-min-tokens", "256", "--kv-pull-hint-min-incremental-tokens", "512", "--kv-pull-scope", "any",
    "--symmetric-pd-unknown-expire-secs", "60", "--symmetric-pd-pull-from-any-holder",
    # Late D: the D is chosen at the P leg's first chunk by fit_req over every D.
    "--pd-native-decode-state", "--pd-late-decode", "choose", "--pd-late-decode-rule", "fit_req",
    "--pd-late-decode-candidates", "all", "--pd-late-decode-local", "off", "--pd-late-decode-fit-native",
    "--pd-late-decode-fit-state-max-age-ms", "1000",
]

# Trie arguments of the baseline configuration (besides the per-process ones in trie_args).
TRIE_ARGS = dict(seed=0, lazy_workload=True, warm_start="stationary", cache_salt_mode="global",
                 migration_limit=3, replay_context_mode="actual_output", replay_output_mode="model")


def workers(n_nodes: int) -> list[dict]:
    return [dict(node=r, slot=s, idx=2 * r + s, name=f"w{r}{s}", base_gpu=4 * s, port=30000 + 1000 * s,
                 bootstrap=8998 + 1000 * s, kv_events=5557 + 1000 * s, peer_kv=18100 + 1000 * s,
                 late_d=ENGINE["late_d"]["control_port_base"] + 1000 * s)
            for r in range(n_nodes) for s in range(2)]


def role(w: dict, cfg: dict) -> str:
    return "prefill" if w["idx"] < cfg["n_p"] else "decode"


def worker_url(w: dict, ips: list) -> str:
    return f"http://{ips[w['node']]}:{w['port']}"


def trace_dir(cfg: dict, w: dict) -> str:
    return f"/tmp/run/{cfg['run_id']}/{TRACE_SUBDIR}/{w['name']}"


def engine_command(w: dict, cfg: dict, nics: dict) -> list[str]:
    e, nic = ENGINE, nics[str(w["base_gpu"])]
    chunk = e["chunked_prefill_size"]
    events = {"publisher": "zmq", "endpoint": f"tcp://*:{w['kv_events']}",
              "replay_endpoint": f"tcp://*:{w['kv_events'] + 1}"}
    peer_backend = {"backend_name": "peer_kv",
                    "module_path": "decoupled_engine.srt.mem_cache.storage.peer_kv.peer_kv_backend",
                    "class_name": "PeerKVBackend", "interface_v1": 1, "ib_device": nic}
    ld = e["late_d"]
    cmd = [
        "python3", "-m", "decoupled_engine.launch_server",
        "--model-path", MODEL, "--served-model-name", SERVED, "--trust-remote-code",
        "--tp-size", "4", "--base-gpu-id", str(w["base_gpu"]), "--host", "0.0.0.0", "--port", str(w["port"]),
        "--mem-fraction-static", str(e["mem_fraction_static"]),
        "--max-running-requests", str(e["max_running_requests"]), "--watchdog-timeout", "900",
        "--moe-runner-backend", "flashinfer_mxfp4", "--speculative-algorithm", "DSPARK",
        "--synthetic-oracle-dspark-accept-rate", str(e["oracle_accept_rate"]),
        "--chunked-prefill-size", str(chunk), "--max-prefill-tokens", str(2 * chunk),
        "--swa-full-tokens-ratio", str(e["swa_full_tokens_ratio"]),
        "--enable-hierarchical-cache", "--hicache-ratio", str(e["hicache_ratio"]),
        "--dist-timeout", str(e["dist_timeout_s"]),  # TP0 allocates a large host tier while the others wait
        "--hicache-write-policy", "write_through", "--enable-dsv4-hicache-host-dedup",
        "--disaggregation-mode", "symmetric", "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-ib-device", json.dumps(nics), "--disaggregation-bootstrap-port", str(w["bootstrap"]),
        "--enable-metrics", "--enable-cache-report", "--incremental-streaming-output",
        "--kv-events-config", json.dumps(events),
        "--enable-peer-kv-servicer", "--peer-kv-servicer-port", str(w["peer_kv"]), "--mooncake-ib-device", nic,
        "--hicache-storage-backend", "dynamic", "--hicache-storage-backend-extra-config", json.dumps(peer_backend),
        "--peer-kv-servicer-reserve-ratio-kv", str(e["peer_kv_reserve_ratio_kv"]),
        "--peer-kv-servicer-max-leased-gb", str(e["peer_kv_max_leased_gb"]),
        "--enable-prefill-work-snapshot",
        "--disaggregation-late-d", ld["mode"], "--disaggregation-late-d-control-port", str(w["late_d"]),
        "--disaggregation-late-d-lead-mode", ld["lead_mode"],
        "--disaggregation-late-d-allocation-trigger", ld["allocation_trigger"],
    ]
    if cfg.get("controller"):
        cmd += ["--sched-trace-dir", trace_dir(cfg, w)]  # the planner's step-time traces
    return cmd


def engine_env(ip: str) -> dict:
    # Mooncake and the ZMQ KV index use SGLANG_HOST_IP; the default egress address is not reachable
    # between Modal containers.
    return {"SGLANG_HOST_IP": ip, **ENGINE_ENV}


def frontend_node(cfg: dict, n_nodes: int, i: int) -> int:
    """Frontend i (one per Trie process) runs on node i mod n_nodes."""
    return i % n_nodes


def frontend_urls(cfg: dict, ips: list) -> list[str]:
    return [f"http://{ips[frontend_node(cfg, len(ips), i)]}:{FRONTEND_PORT_BASE + i}"
            for i in range(cfg["trie_procs"])]


def frontend_config(cfg: dict, ips: list, i: int) -> dict:
    ws = workers(len(ips))
    return dict(node_id=f"t{i}", config_id=CONFIG_ID, model_path=MODEL,
                tokenize_endpoint=worker_url(ws[0], ips) + "/tokenize",
                targets={worker_url(w, ips) + "/generate": CONFIG_ID for w in ws},
                page_size=256, context_length=1_048_576, hash_mode="dynamo", is_eagle=False, cache_salt=None,
                callback_targets=[f"http://{ips[0]}:{ROUTER_PORT}/symmetric_pd/events"],
                callback_timeout_seconds=5, callback_outbox_max_plans=4096,
                max_body_bytes=64 * 2**20, max_response_bytes=64 * 2**20, max_staging_bytes=4 * 2**30,
                ttl_seconds=600, request_timeout_seconds=7200)


def frontend_command(path: str, i: int) -> list[str]:
    return [SERVICE_PY, "-m", "evolving_infra.router.frontend", "--config", path,
            "--host", "0.0.0.0", "--port", str(FRONTEND_PORT_BASE + i)]


def indexer_config(ips: list) -> dict:
    """An event bridge per worker + Dynamo's indexer + the hash-query adapter (DSV4 pages are plain
    token pages, not EAGLE): the router queries with the frontend's prepared block hashes."""
    ws = [dict(instance_id=INDEXER_ID_BASE + w["idx"],
               events_endpoint=f"tcp://{ips[w['node']]}:{w['kv_events']}",
               replay_endpoint=f"tcp://{ips[w['node']]}:{w['kv_events'] + 1}",
               bridge_pub_port=26100 + 2 * w["idx"], bridge_replay_port=26100 + 2 * w["idx"] + 1)
          for w in workers(len(ips))]
    return dict(workers=ws, model_name=SERVED, block_size=256, eagle=False,
                indexer_port=8090, query_port=8091, threads=4)


def indexer_command(config: str, output: str) -> list[str]:
    return [SERVICE_PY, "-m", "evolving_infra.router.indexer", "--config", config, "--output", output]


def prepared_config(ips: list) -> dict:
    return dict(callback_url=f"http://{ips[0]}:{ROUTER_PORT}/symmetric_pd/events", max_inflight=4096,
                contracts={CONFIG_ID: dict(page_size=256, is_eagle=False, model_path=MODEL, cache_namespace=None)})


def router_command(cfg: dict, ips: list, rundir: str, query_url: str) -> list[str]:
    ws = workers(len(ips))
    cmd = [ROUTER_BIN, "--host", "0.0.0.0", "--port", str(ROUTER_PORT), "--model-id", SERVED,
           "--tokenizer-path", f"{MODEL}/tokenizer.json", *ROUTER_ARGS,
           "--kv-indexer-url", query_url, "--co-schedule-cost-trace", f"{rundir}/router-cost.jsonl"]
    if c := cfg.get("controller"):
        cmd += ["--symmetric-pd-external-role-control", "--symmetric-pd-role-drain-timeout-secs", str(c["role_drain_timeout_s"])]
    for w in ws:
        weights = "1,0" if role(w, cfg) == "prefill" else "0,1"
        cmd += ["--symmetric-pd-worker", f"{worker_url(w, ips)},{w['bootstrap']},{weights}"]
    worker_map = [dict(id=INDEXER_ID_BASE + w["idx"], url=worker_url(w, ips), servicer=f"{ips[w['node']]}:{w['peer_kv']}")
                  for w in ws]
    return cmd + ["--kv-worker-map", json.dumps(worker_map)]


def token_file(cfg: dict) -> str:
    """Role-control credential: node-local, mode 600, outside the synced run directory."""
    return f"/tmp/evolving-infra-private/{cfg['run_id']}/role-control-token"


def controller_config(cfg: dict, ips: list, run_dir: str) -> dict:
    """planner `control --config` arguments (PlannerArgs field names); the token is passed as a path only."""
    c = cfg["controller"]
    workers_ = [dict(node=f"node{w['idx']}", idx=w["idx"], name=w["name"], host=w["node"], ip=ips[w["node"]],
                     url=worker_url(w, ips), trace_url=f"http://{ips[w['node']]}:{TRACE_PORT}",
                     initial_role=role(w, cfg))
                for w in workers(len(ips))]
    return dict({k: v for k, v in c.items() if k != "role_drain_timeout_s"},
                router_url=ROUTER_URL, role_control_token_file=token_file(cfg), run_dir=run_dir,
                scratch_dir=f"/tmp/planner-scratch/{cfg['run_id']}",
                hosts=[dict(index=i, ip=ip) for i, ip in enumerate(ips)], workers=workers_)


def controller_config_path(run_dir: str) -> str:
    return f"{run_dir}/controller.json"


def controller_command(path: str) -> list[str]:
    return [SERVICE_PY, "-m", "evolving_infra.planner", "control", "--config", path]


def trace_server_command(cfg: dict) -> list[str]:
    return [SERVICE_PY, "-m", "evolving_infra.planner", "serve-traces",
            "--root", f"/tmp/run/{cfg['run_id']}/{TRACE_SUBDIR}", "--port", str(TRACE_PORT)]


def replay_env(cfg: dict, ips: list) -> dict:
    """EVOLVING_INFRA_REPLAY for benchmarks/replay/run_trie.py (process i uses frontend i). No session END:
    retention is off and the engines would refuse it."""
    return dict(router_url=ROUTER_URL, tokenizer_urls=frontend_urls(cfg, ips), config_id=CONFIG_ID,
                model_name=SERVED, model_path=MODEL, incremental_streaming_output=True, cache_salt=None,
                timeout_s=3600, retention_enabled=False)


def trie_command(cfg: dict, shard: str, concurrency: int, outdir: str, i: int) -> list[str]:
    args = dict(preset="none", workload_path=shard, endpoint=f"{ROUTER_URL}/v1", model=SERVED, api_key="EMPTY",
                concurrency=concurrency, delay_scale=1.0, duration=cfg["duration"], strict_replay=True,
                prefix_block_size=256, stream=True, timeout=3600, max_retries=0,
                num_gpus=4 * cfg["n_workers"],
                output_json=f"{outdir}/trie-results-{i}.json", lifecycle_output_jsonl=f"{outdir}/trie-lifecycle-{i}.jsonl",
                lifecycle_run_id=cfg["run_id"], lifecycle_cell_id=cfg["name"], lifecycle_arm="on",
                lifecycle_concurrency=concurrency, strict_replay_openai_fast_path=True, **TRIE_ARGS)
    return [TRIE_PY, f"{REPLAY_DIR}/run_trie.py"] + [f"{k}={v}" for k, v in args.items()]


def conformance_command(outdir: str, worker_url_: str, name: str) -> list[str]:
    """Trie's strict-replay conformance probe against one worker. The router has no native /generate
    route (the probe's boundary, mixed-batch and replay steps need it), so every worker is probed
    directly; a request without router marks runs prefill and decode on that worker."""
    return [TRIE_PY, f"{REPLAY_DIR}/capture_conformance.py",
            "--script", f"{TRIE_DIR}/scripts/sglang_strict_replay_conformance.py",
            "--event-log", f"{outdir}/conformance-{name}-events.jsonl", "--",
            "--endpoint", f"{worker_url_}/v1", "--model", SERVED, "--verify-width", "6",
            "--output", f"{outdir}/conformance-{name}.json"]


# ------------------------------------------------------------------------------------------- configs

CONTROLLER = dict(  # planner arguments (PlannerArgs) of the dynamic run, plus the router's drain timeout
    learning_secs=1800, decision_period_secs=600, post_switch_exclusion_secs=120., role_drain_timeout_s=5,
    role_script=None,
    prefill_shape_prior="dsv4-flash-h200-tp4",
)


def bench_controller(duration: int) -> dict:
    """2 nodes, 2P2D. Learning (1/3 of the replay) starts when the planner sees the cold start end; w00
    P -> D 150 s after the fit (1P3D) and back D -> P 300 s later (2P2D), both before the replay ends when
    the cold start takes at most duration/3 - 450 s. The router refuses any swap at 1P1D (a draining
    D -> P leaves no serving P), hence two nodes."""
    learning = duration // 3
    return dict(CONTROLLER, learning_secs=learning, decision_period_secs=min(600, learning),
                post_switch_exclusion_secs=min(120., min(600, learning) - 120),
                role_script=[[learning + 150, {"node0": "decode"}], [learning + 450, {"node0": "prefill"}]])


def bench_config(run_id: str, dynamic: bool, duration: int = 0, concurrency: int = 16, trie_procs: int = 2,
                 workload: str = WORKLOAD) -> dict:
    """Static: 1 node, 1P1D, 1200 s. Dynamic: 2 nodes, 2P2D -> 1P3D -> 2P2D, 1800 s. Half of the
    replay is warm-up; the second half is the scored window."""
    duration = duration or (1800 if dynamic else 1200)
    return dict(run_id=run_id, name="dynamic-cc" if dynamic else "static-cc",
                n_nodes=2 if dynamic else 1, n_workers=4 if dynamic else 2, n_p=2 if dynamic else 1,
                concurrency=concurrency, trie_procs=trie_procs, duration=duration, warmup_s=duration // 2,
                workload=workload, metrics_period_s=60, trie_grace_s=900, cold_start_deadline_s=3600,
                controller=bench_controller(duration) if dynamic else None)
