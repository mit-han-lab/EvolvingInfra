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
from evolving_infra.launch.layout import (FRONTEND_PORT, INDEXER_PORT, INDEXER_QUERY_PORT, ROUTER_PORT,
                                          TRACE_PORT, workers)


def engine_python() -> str:
    """The interpreter with the engine: this one, or python3 (the engine's in the runtime image)."""
    return sys.executable if importlib.util.find_spec("decoupled_engine") else "python3"


class Stack:
    def __init__(self, cfg: dict, hosts: list, rank: int, work_dir: Path):
        self.cfg, self.rank = cfg, rank
        self.ips = [hardware.address(h) for h in hosts]
        self.dir = work_dir
        (self.dir / "logs").mkdir(parents=True, exist_ok=True)
        gpus = hardware.gpu_count()
        per_worker = cfg.get("gpus_per_worker") or (gpus if len(hosts) > 1 else gpus // 2)
        total = (gpus // per_worker) * len(hosts)
        self.workers = workers(self.ips, gpus, per_worker, cfg.get("prefill_workers") or total // 2)
        self.mine = [w for w in self.workers if w.host == rank]
        self.head = self.ips[0]
        self.planner = cfg.get("planner")  # None: off; a dict of planner arguments: on
        self.eagle = "--speculative-algorithm" in " ".join(cfg.get("engine_args", []))
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
        for name, proc in self.procs:
            if proc.poll() is not None:
                raise RuntimeError(f"{name} exited with {proc.returncode}; see {self.dir}/logs/{name}.log")

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
                "--kv-events-config", json.dumps({"publisher": "zmq", "endpoint": f"tcp://*:{w.events_port}",
                                                  "replay_endpoint": f"tcp://*:{w.events_port + 1}"}),
                "--enable-metrics", "--enable-cache-report", "--incremental-streaming-output"]
        if cfg.get("context_length"):
            argv += ["--context-length", str(cfg["context_length"])]
        if nics:
            argv += ["--disaggregation-ib-device", json.dumps({str(g): n for g, n in nics.items()})]
        if self.planner is not None:
            argv += ["--sched-trace-dir", str(self.dir / "traces" / w.name)]
        return argv + list(cfg.get("engine_args", []))

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
            self.start(f"engine-{w.name}", self.engine_argv(w, nics), env)
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
            "workers": [{"instance_id": 20001 + w.idx, "events_endpoint": f"tcp://{w.ip}:{w.events_port}",
                         "replay_endpoint": f"tcp://{w.ip}:{w.events_port + 1}",
                         "bridge_pub_port": 26100 + 2 * w.idx, "bridge_replay_port": 26101 + 2 * w.idx}
                        for w in ws]})
        shutil.rmtree(self.dir / "indexer", ignore_errors=True)
        self.start("indexer", [sys.executable, "-m", "evolving_infra.router.indexer", "--config", indexer,
                               "--output", str(self.dir / "indexer")])
        callback = f"http://{head}:{ROUTER_PORT}/symmetric_pd/events"
        frontend = self.write("frontend.json", {
            "node_id": "t0", "config_id": cid, "model_path": model,
            "tokenize_endpoint": f"{ws[0].url}/tokenize",
            "targets": {f"{w.url}/generate": cid for w in ws},
            "page_size": page, "context_length": cfg.get("context_length") or 1048576, "hash_mode": "dynamo",
            "is_eagle": self.eagle, "cache_salt": None, "callback_targets": [callback]})
        self.start("frontend", [sys.executable, "-m", "evolving_infra.router.frontend", "--config", frontend,
                                "--host", "0.0.0.0", "--port", str(FRONTEND_PORT)])
        prepared = self.write("prepared.json", {
            "callback_url": callback, "max_inflight": 4096,
            "contracts": {cid: {"page_size": page, "is_eagle": self.eagle, "model_path": model,
                                "cache_namespace": None}}})
        ready = self.dir / "indexer" / "ready.json"
        for _ in range(120):
            self.check()
            if ready.exists():
                break
            time.sleep(2)
        query = json.loads(ready.read_text())["query_url"]
        argv = ["sgl-router", "--host", "0.0.0.0", "--port", str(ROUTER_PORT),
                "--model-id", cfg["served_model_name"], "--tokenizer-path", f"{model}/tokenizer.json",
                "--policy", "symmetric_pd", "--request-timeout-secs", "3600", "--stale-request-timeout-secs", "3600",
                "--kv-indexer-url", query,
                "--kv-worker-map", json.dumps([{"id": 20001 + w.idx, "url": w.url,
                                                "servicer": f"{w.ip}:{w.servicer_port}"} for w in ws])]
        for w in ws:
            p = int(w.role == "prefill")
            argv += ["--symmetric-pd-worker", f"{w.url},{w.bootstrap_port},{p},{1 - p}"]
        env = {"EVOLVING_INFRA_PREPARED_CONFIG": prepared}
        if self.planner is not None:
            token = self.dir / "role-token"
            token.write_text(secrets.token_urlsafe(32))
            token.chmod(0o600)
            argv.append("--symmetric-pd-external-role-control")
            env["EVOLVING_INFRA_ROLE_CONTROL_TOKEN_FILE"] = str(token)
        self.start("router", argv + list(cfg.get("router_args", [])), env)
        self.wait("router", f"http://127.0.0.1:{ROUTER_PORT}/readyz", 600)
        if self.planner is not None:
            self.start_planner(token)

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
                (self.dir / "READY").write_text(f"http://{self.head}:{ROUTER_PORT}\n")
            while True:
                self.check()
                time.sleep(5)
        finally:
            self.stop()
