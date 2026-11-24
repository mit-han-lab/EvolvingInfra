# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Start this host's part of the stack and keep it running.

Every host runs its engines (and, with the planner, a trace server). The first host also runs the
KV directory, the frontend, the router and the planner, once every engine answers /health.
"""
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from evolving_infra.launch import hardware
from evolving_infra.launch.layout import (FRONTEND_PORT, GATEWAY_PORT, INDEXER_PORT, INDEXER_QUERY_PORT, ROUTER_PORT,
                                          TRACE_PORT, workers)


def engine_python() -> str:
    """The interpreter with the engine: this one, or python3 (the engine's in the runtime image)."""
    return sys.executable if importlib.util.find_spec("decoupled_engine") else "python3"


def _argument_groups(args: list[str]) -> list[list[str]]:
    """Keep each option with its following values, including negative numbers."""
    groups = []
    for arg in args:
        if not groups or arg.startswith("--") or (arg.startswith("-") and arg[1:2].isalpha()):
            groups.append([arg])
        else:
            groups[-1].append(arg)
    return groups


def merge_args(base: list[str], overrides: list[str]) -> list[str]:
    """Replace options together with their values, keeping untouched groups."""
    replaced = {group[0].split("=", 1)[0] for group in _argument_groups(overrides)}
    return [arg for group in _argument_groups(base)
            if group[0].split("=", 1)[0] not in replaced for arg in group] + overrides


class Stack:
    def __init__(self, cfg: dict, hosts: list, rank: int, work_dir: Path):
        self.cfg, self.rank = cfg, rank
        self.ips = [hardware.address(h) for h in hosts]
        self.dir = work_dir
        (self.dir / "logs").mkdir(parents=True, exist_ok=True)
        gpus = hardware.gpu_count()
        per_worker = cfg.get("gpus_per_worker") or (gpus if len(hosts) > 1 else gpus // 2)
        if per_worker < 1:
            raise SystemExit(f"one host runs a prefill and a decode worker: it needs 2 GPUs or more, this one has {gpus}")
        if per_worker > gpus:
            raise SystemExit(f"gpus_per_worker is {per_worker}, but this host has {gpus} GPUs")
        total = (gpus // per_worker) * len(hosts)
        self.workers = workers(self.ips, gpus, per_worker, cfg.get("prefill_workers") or total // 2,
                               dp_attention=bool(cfg.get("dp_attention")))
        # Engine flags and variables per starting role (static P/D only, see config()).
        self.roles = cfg.get("roles") or {}
        self.mine = [w for w in self.workers if w.host == rank]
        self.head = self.ips[0]
        self.planner = cfg.get("planner")  # None: off; a dict of planner arguments: on
        self.eagle = "--speculative-algorithm=EAGLE" in " ".join(cfg.get("engine_args", [])).replace("algorithm ", "algorithm=")
        # KV session retention needs a FULL-only or FULL+SWA cache tree; the engine preset decides the
        # same way per model. `retention: false` in the config turns it off in the router for the others.
        self.retention = cfg.get("retention", True)
        # The agent gateway for coding-agent harnesses runs only with a `gateway` section; its `host`
        # is the bind address (default 127.0.0.1: this host only).
        self.gateway = cfg.get("gateway")
        if self.gateway is not None and not isinstance(self.gateway, dict):
            raise SystemExit("gateway: give a mapping, e.g. {host: 0.0.0.0}")
        self.procs = []

    # ------------------------------------------------------------------------------------------ processes
    def start(self, name, argv, env=None):
        log = open(self.dir / "logs" / f"{name}.log", "w")
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                env={**os.environ, **(env or {})}, cwd=self.dir)
        self.procs.append((name, proc))
        print(f"started {name} (pid {proc.pid}), log {log.name}", flush=True)

    def stop(self, *_):
        for _, proc in self.procs:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        deadline = time.time() + 30
        for _, proc in self.procs:
            try:
                proc.wait(max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)

    def check(self):
        dead = [(name, proc.returncode) for name, proc in self.procs if proc.poll() is not None]
        if dead:
            raise RuntimeError("; ".join(f"{name} exited with {code}; see {self.dir}/logs/{name}.log"
                                         for name, code in dead))

    def wait(self, what, url, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.check()
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        return
            except OSError:
                pass
            time.sleep(5)
        raise TimeoutError(f"{what} not ready after {timeout} s ({url})")

    # ------------------------------------------------------------------------------------------ engines
    def engine_argv(self, w, nics):
        cfg = self.cfg
        argv = [engine_python(), "-m", "decoupled_engine.launch_server",
                "--model-path", cfg["model"], "--served-model-name", cfg["served_model_name"],
                "--tp-size", str(w.gpus), "--base-gpu-id", str(w.base_gpu),
                "--host", "0.0.0.0", "--port", str(w.port), "--page-size", str(cfg["page_size"]),
                "--disaggregation-mode", "symmetric", "--disaggregation-transfer-backend", "mooncake",
                "--disaggregation-bootstrap-port", str(w.bootstrap_port),
                "--peer-kv-servicer-port", str(w.servicer_port),
                "--disaggregation-late-d-control-port", str(w.late_decode_port),
                "--kv-events-config", json.dumps({"publisher": "zmq", "endpoint": f"tcp://*:{w.events_port}",
                                                  "replay_endpoint": f"tcp://*:{w.replay_port}"}),
                "--enable-metrics", "--enable-cache-report", "--incremental-streaming-output"]
        if w.dp > 1:
            argv += ["--dp-size", str(w.dp), "--enable-dp-attention"]
        if cfg.get("context_length"):
            argv += ["--context-length", str(cfg["context_length"])]
        if nics:
            argv += ["--disaggregation-ib-device", json.dumps({str(g): n for g, n in nics.items()})]
        if self.planner is not None:
            argv += ["--sched-trace-dir", str(self.dir / "traces" / w.name)]
        extra = self.config_engine_args(w)
        if not self.peer_kv(w) and "--disable-peer-kv" not in extra:
            argv.append("--disable-peer-kv")
        return argv + extra

    def config_engine_args(self, w) -> list:
        """The config's engine flags of every engine, then those of the worker's starting role."""
        role = self.roles.get(w.role) or {}
        return merge_args(list(self.cfg.get("engine_args", [])), list(role.get("engine_args", [])))

    def peer_kv(self, w) -> bool:
        """Whether worker ``w`` runs the peer KV servicer and the peer_kv reader (the engine preset's default).
        A decode worker with DP attention in a static split runs neither: its engine gets --disable-peer-kv and
        the router's map names no servicer of its ranks. It never prefills a P-leg, so it takes no pull, and no
        pull reads the turns it decoded (docs/placement.md); every rank would still do the servicer's work, and
        every step of its ranks waits for the slowest one. Every other worker runs both: one that may prefill (a
        prefill worker, any worker with the planner), a decode worker with decode host return (it serves the KV
        it decoded) and a decode worker without DP attention (a pull from any holder reads it). The config's
        engine flags decide when they give --disable-peer-kv or --enable-peer-kv-servicer."""
        flags = {arg.split("=", 1)[0] for arg in self.config_engine_args(w)}
        if "--disable-peer-kv" in flags:
            return False
        if "--enable-peer-kv-servicer" in flags:
            return True
        host_return = "--symmetric-pd-decode-host-return" in {
            arg.split("=", 1)[0] for arg in self.cfg.get("router_args", [])}
        return w.role == "prefill" or w.dp == 1 or self.planner is not None or host_return

    def engine_env(self, w) -> dict:
        """The variables of every engine, then those of the worker's starting role."""
        role = self.roles.get(w.role) or {}
        return {k: str(v) for k, v in {**(self.cfg.get("engine_env") or {}), **(role.get("engine_env") or {})}.items()}

    def kv_instances(self) -> list:
        """One KV-directory instance per worker and DP rank: (instance id, worker, rank, bridge index).
        Without DP attention it is the worker's id 20001 + idx and bridge index idx."""
        return [(20001 + w.idx + 1000 * r, w, r, r * len(self.workers) + w.idx)
                for r in range(max(w.dp for w in self.workers)) for w in self.workers if r < w.dp]

    def kv_worker_map(self) -> list:
        """The router's --kv-worker-map: each KV-directory instance with its worker, its DP rank and that
        rank's peer KV servicer, so the router places, prices and pulls per (worker, DP rank). The ranks of a
        worker without a servicer (``peer_kv``) are listed without one: no pull reads from them."""
        return [{"id": iid, "url": w.url, "dp_rank": r,
                 **({"servicer": f"{w.ip}:{w.rank_servicer_port(r)}"} if self.peer_kv(w) else {})}
                for iid, w, r, _ in self.kv_instances()]

    def check_ports(self):
        busy = []
        for w in self.mine:
            for port in w.ports():
                with socket.socket() as sock:
                    try:
                        sock.bind(("0.0.0.0", port))
                    except OSError:
                        busy.append(f"{port} ({w.name})")
        if busy:
            raise RuntimeError("ports already in use on this host: " + ", ".join(busy))

    def start_engines(self):
        self.check_ports()
        nics = hardware.nic_map()
        env = {"SGLANG_HOST_IP": self.ips[self.rank],
               "WITH_NVIDIA_PEERMEM": "1" if hardware.peer_memory() else "0"}
        print(f"host {self.rank}: {self.ips[self.rank]}, RDMA {nics or 'none'}, "
              f"peer memory {env['WITH_NVIDIA_PEERMEM']}", flush=True)
        for w in self.mine:
            self.start(f"engine-{w.name}", self.engine_argv(w, nics), {**env, **self.engine_env(w)})
        if self.planner is not None:
            (self.dir / "traces").mkdir(exist_ok=True)
            self.start("trace-server", [sys.executable, "-m", "evolving_infra.planner", "serve-traces",
                                        "--root", str(self.dir / "traces"), "--port", str(TRACE_PORT)])

    # ------------------------------------------------------------------------------------------ head services
    def write(self, name, value):
        path = self.dir / name
        path.write_text(json.dumps(value, indent=1))
        return str(path)

    def start_services(self):
        cfg, head, ws = self.cfg, self.head, self.workers
        page, model, cid = cfg["page_size"], cfg["model"], "default"
        indexer = self.write("indexer.json", {
            "model_name": cfg["served_model_name"], "block_size": page, "eagle": self.eagle,
            "indexer_port": INDEXER_PORT, "query_port": INDEXER_QUERY_PORT, "threads": 4,
            "workers": [{"instance_id": iid, "events_endpoint": f"tcp://{w.ip}:{w.events_port + r}",
                         "replay_endpoint": f"tcp://{w.ip}:{w.replay_port + r}",
                         "bridge_pub_port": 26100 + 2 * k, "bridge_replay_port": 26101 + 2 * k}
                        for iid, w, r, k in self.kv_instances()]})
        shutil.rmtree(self.dir / "indexer", ignore_errors=True)
        self.start("indexer", [sys.executable, "-m", "evolving_infra.router.indexer", "--config", indexer,
                               "--output", str(self.dir / "indexer")])
        callback = f"http://{head}:{ROUTER_PORT}/symmetric_pd/events"
        frontend = self.write("frontend.json", {
            "node_id": "t0", "config_id": cid, "model_path": model,
            "tokenize_endpoint": f"{ws[0].url}/tokenize",
            "targets": {f"{w.url}/generate": cid for w in ws},
            "page_size": page, "context_length": cfg.get("context_length") or 1048576, "hash_mode": "dynamo",
            "is_eagle": self.eagle, "cache_salt": None, "callback_targets": [callback],
            "public_url": f"http://{head}:{FRONTEND_PORT}",  # workers fetch uploaded images from here
            **(cfg.get("frontend") or {})})  # e.g. image_config and body limits for image input
        self.start("frontend", [sys.executable, "-m", "evolving_infra.router.frontend", "--config", frontend,
                                "--host", "0.0.0.0", "--port", str(FRONTEND_PORT)])
        prepared = self.write("prepared.json", {
            "callback_url": callback, "max_inflight": 4096,
            "contracts": {cid: {"page_size": page, "is_eagle": self.eagle, "model_path": model,
                                "cache_namespace": None,
                                **({"image_config": cfg["frontend"]["image_config"]}
                                   if (cfg.get("frontend") or {}).get("image_config") else {})}}})
        ready = self.dir / "indexer" / "ready.json"
        for _ in range(120):
            self.check()
            if ready.exists():
                break
            time.sleep(2)
        query = json.loads(ready.read_text())["query_url"]
        argv = ["sgl-router", "--host", "0.0.0.0", "--port", str(ROUTER_PORT),
                "--model-id", cfg["served_model_name"],
                "--tokenizer-path", cfg.get("tokenizer") or f"{model}/tokenizer.json",
                "--policy", "symmetric_pd", "--request-timeout-secs", "3600", "--stale-request-timeout-secs", "3600",
                "--kv-indexer-url", query, "--kv-worker-map", json.dumps(self.kv_worker_map())]
        for w in ws:
            p = int(w.role == "prefill")
            argv += ["--symmetric-pd-worker", f"{w.url},{w.bootstrap_port},{p},{1 - p}"]
        if not self.retention:
            argv.append("--symmetric-pd-no-kv-session-retention")
        env = {"EVOLVING_INFRA_PREPARED_CONFIG": prepared}
        if self.planner is not None:
            token = self.dir / "role-token"
            token.write_text(secrets.token_urlsafe(32))
            token.chmod(0o600)
            argv.append("--symmetric-pd-external-role-control")
            env["EVOLVING_INFRA_ROLE_CONTROL_TOKEN_FILE"] = str(token)
        self.start("router", argv + list(cfg.get("router_args", [])), env)
        self.wait("router", f"http://127.0.0.1:{ROUTER_PORT}/readyz", 600)
        if self.gateway is not None:
            self.start("gateway", self.gateway_argv())
            self.wait("gateway", f"{self.gateway_url(local=True)}/gateway/stats", 60)
        if self.planner is not None:
            self.start_planner(token)

    def gateway_argv(self) -> list:
        return ([sys.executable, "-m", "evolving_infra.router.gateway", "--router-url",
                 f"http://127.0.0.1:{ROUTER_PORT}", "--host", self.gateway.get("host", "127.0.0.1"),
                 "--port", str(GATEWAY_PORT)] + ([] if self.retention else ["--no-retention"]))

    def gateway_url(self, local=False) -> str:
        """The gateway's URL; bound to every interface it is reached at the first host's address
        (``local``: at 127.0.0.1)."""
        host = self.gateway.get("host", "127.0.0.1")
        if host == "0.0.0.0":
            host = "127.0.0.1" if local else self.head
        return f"http://{host}:{GATEWAY_PORT}"

    def start_planner(self, token):
        args = dict(self.planner)
        args.update(
            router_url=f"http://127.0.0.1:{ROUTER_PORT}", role_control_token_file=str(token),
            run_dir=str(self.dir / "planner"), scratch_dir=str(self.dir / "planner-scratch"),
            hosts=[{"index": h, "ip": ip} for h, ip in enumerate(self.ips)],
            workers=[{"node": f"node{w.idx}", "idx": w.idx, "name": w.name, "host": w.host, "ip": w.ip,
                      "url": w.url, "trace_url": f"http://{w.ip}:{TRACE_PORT}", "initial_role": w.role}
                     for w in self.workers])
        self.start("planner", [sys.executable, "-m", "evolving_infra.planner", "control",
                               "--config", self.write("planner.json", args)])

    # ------------------------------------------------------------------------------------------ run
    def run(self):
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        try:
            self.start_engines()
            if self.rank == 0:
                for w in self.workers:
                    self.wait(f"engine {w.name}", f"{w.url}/health", 3600)
                self.start_services()
                print(f"ready: http://{self.head}:{ROUTER_PORT}/v1 ({len(self.workers)} workers, "
                      f"{sum(w.role == 'prefill' for w in self.workers)}P"
                      f"{sum(w.role == 'decode' for w in self.workers)}D)", flush=True)
                self.write("stack.json", {  # what a client needs; benchmarks/replay/run_trie.py stack=<this dir>
                    "router_url": f"http://{self.head}:{ROUTER_PORT}",
                    # for clients without agent identities; only with a `gateway` section
                    **({"gateway_url": self.gateway_url()} if self.gateway is not None else {}),
                    "tokenizer_url": f"http://{self.head}:{FRONTEND_PORT}",
                    "model": self.cfg["served_model_name"], "model_path": self.cfg["model"],
                    "page_size": self.cfg["page_size"], "config_id": "default", "retention": self.retention,
                    "gpus": sum(w.gpus for w in self.workers), "workers": len(self.workers),
                    "prefill_workers": sum(w.role == "prefill" for w in self.workers)})
                (self.dir / "READY").write_text(f"http://{self.head}:{ROUTER_PORT}\n")
            while True:
                self.check()
                time.sleep(5)
        finally:
            self.stop()
