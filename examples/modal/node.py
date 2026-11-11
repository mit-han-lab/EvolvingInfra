"""What runs in each container of a benchmark cluster (Modal-agnostic apart from the Dict and Volumes
handed in by app.py).

Every node: its frontends (T), in dynamic runs the planner's trace server, then its two engine
workers; a watcher reports a worker that exits. Rank 0 also drives the run:

  input check -> engines up -> indexer, frontends, router -> quiet window + conformance ->
  quiet window -> [planner controller] -> replay (Trie through the prepared path) -> score + checks

and writes `summary.json` with the verdict. The local run directory /tmp/run/<run_id> is mirrored to
the results Volume under <run_id>/node<rank>/. Engines are not stopped inside the container: ending
a process that holds pinned host memory stalls the container for over a minute; Modal reclaims the
container when the function returns.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import shlex
import subprocess
import sys
import threading
import time
import urllib.request

import stack

sys.path.insert(0, stack.REPLAY_DIR)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd: str, timeout: int = 120) -> str:
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    return (p.stdout + p.stderr).strip()


def http_ok(url: str, timeout: float = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001 - not up yet
        return False


def http_json(url: str, timeout: float = 30, headers: dict | None = None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Ctl:
    """modal.Dict with retries: the Dict backend answers 'overloaded' under load, and one uncaught
    error in a node's loop would leave the other nodes waiting."""

    def __init__(self, d, run_id: str):
        self.d, self.run_id = d, run_id

    def get(self, key: str, give_up_s: float = 900):
        t0 = time.time()
        while True:
            try:
                return self.d.get(f"{self.run_id}:{key}")
            except Exception as e:  # noqa: BLE001 - transient control-plane errors
                if time.time() - t0 > give_up_s:
                    raise
                log(f"ctl get {key} failed: {e!r}; retrying")
                time.sleep(5 + 5 * random.random())

    def put(self, key: str, value, give_up_s: float = 900):
        t0 = time.time()
        while True:
            try:
                self.d[f"{self.run_id}:{key}"] = value
                return
            except Exception as e:  # noqa: BLE001
                if time.time() - t0 > give_up_s:
                    raise
                log(f"ctl put {key} failed: {e!r}; retrying")
                time.sleep(5 + 5 * random.random())


def nic_map() -> dict[str, str]:
    """GPU index -> the RDMA NIC with the longest common PCI path (index order as fallback)."""
    nics = sorted(os.listdir("/sys/class/infiniband"))
    gpus = [line.split(",") for line in sh("nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader").splitlines()]
    nic_paths = {n: os.path.realpath(f"/sys/class/infiniband/{n}/device").split("/") for n in nics}
    mapping, used = {}, set()
    for idx, bus in gpus:
        dev = f"/sys/bus/pci/devices/{bus.strip().lower()[-12:]}"
        if not os.path.exists(dev):
            mapping[idx.strip()] = nics[int(idx) % len(nics)]
            continue
        gp = os.path.realpath(dev).split("/")

        def score(n):
            common = next((i for i, (a, b) in enumerate(zip(gp, nic_paths[n])) if a != b), min(len(gp), len(nic_paths[n])))
            return common, n not in used
        best = max(nics, key=score)
        used.add(best)
        mapping[idx.strip()] = best
    return mapping


class Syncer(threading.Thread):
    """Mirror the node-local run directory into the results Volume every `period` seconds (the Volume
    mount refuses hard links, so copy instead of linking)."""

    def __init__(self, src: str, dst: str, volume, period: int = 60):
        super().__init__(daemon=True)
        self.src, self.dst, self.volume, self.period = src, dst, volume, period
        self.lock = threading.Lock()

    def sync(self):
        with self.lock:
            os.makedirs(self.dst, exist_ok=True)
            sh(f"cp -ru {self.src}/. {self.dst}/", timeout=600)
            self.volume.commit()

    def run(self):
        while True:
            time.sleep(self.period)
            try:
                self.sync()
            except Exception as e:  # noqa: BLE001
                log(f"sync failed: {e!r}")


def start(cmd: list[str], directory: str, name: str, env: dict | None = None) -> subprocess.Popen:
    Path(directory, f"{name}.cmd").write_text(shlex.join(cmd))
    out = open(f"{directory}/{name}.log", "w")
    return subprocess.Popen(cmd, env=os.environ | (env or {}), stdout=out, stderr=subprocess.STDOUT,
                            start_new_session=True)


def stop(p: subprocess.Popen | None, timeout: float = 60):
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(p.pid, 15)
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)
    except ProcessLookupError:
        pass


# ------------------------------------------------------------------------------------------- kernel cache

def kcache_tar(cfg: dict) -> Path:
    """JIT kernel caches per engine source tree: /cache (the SGLANG_*/TILELANG/TRITON cache directories)
    and /root/.cache (kernels that ignore those variables)."""
    return Path("/kcache") / f"cache-{cfg['source']['engine_tree'][:12]}-v2.tar"


def restore_kcache(cfg: dict, volume) -> dict | None:
    os.makedirs("/cache", exist_ok=True)
    volume.reload()
    tar = kcache_tar(cfg)
    if not tar.exists():
        log(f"no kernel cache {tar.name}: cold start")
        return None
    t0 = time.time()
    sh(f"tar -xf {tar} -C /", timeout=1800)
    log(f"restored kernel cache {tar.name} in {time.time() - t0:.0f}s")
    return dict(file=tar.name, bytes=tar.stat().st_size, seconds=round(time.time() - t0, 1))


def save_kcache(cfg: dict, volume):
    """Rewrite the tar at the end of a run (then it holds the replay's JIT kernels too); never fails the run."""
    try:
        tar = kcache_tar(cfg)
        tmp = tar.with_suffix(".tar.tmp")
        dirs = " ".join(d for d in ("cache", "root/.cache") if os.path.isdir("/" + d))
        sh(f"tar -cf {tmp} --exclude=root/.cache/pip --exclude=root/.cache/uv -C / {dirs} && mv {tmp} {tar}", timeout=1800)
        volume.commit()
        log(f"saved kernel cache {tar.name} ({tar.stat().st_size / 2**30:.1f} GiB)")
    except Exception as e:  # noqa: BLE001
        log(f"kernel cache not saved: {e!r}")


# ------------------------------------------------------------------------------------------- rank 0

class Driver:
    def __init__(self, cfg: dict, ips: list, rundir: str, ctl: Ctl, commit):
        self.cfg, self.ips, self.rundir, self.ctl, self.commit = cfg, ips, rundir, ctl, commit
        self.ws = stack.workers(len(ips))
        self.urls = [stack.worker_url(w, ips) for w in self.ws]
        self.services: dict[str, subprocess.Popen] = {}
        self.checks: dict = {}
        self.evidence: dict = {}
        self.replay_dir = Path(rundir, "replay")
        self.replay_dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, data):
        Path(self.rundir, name).write_text(json.dumps(data, indent=2))

    def failure(self):
        return self.ctl.get("failed")

    # --------------------------------------------------------------------------------- inputs
    def check_inputs(self) -> list[str]:
        """Workload and tokenizer.json against the offline validation report, then the round-robin
        shards the Trie processes read (kept outside the synced run directory)."""
        import check_inputs
        workload = Path(self.cfg["workload"])
        attest = check_inputs.check_files(workload, Path(stack.MODEL))
        n = self.cfg["trie_procs"]
        out = Path("/tmp/trie-shards", workload.stem)
        out.mkdir(parents=True, exist_ok=True)
        paths = [out / f"shard-{i}.jsonl" for i in range(n)]
        handles = [open(p, "w") for p in paths]
        with open(workload) as f:
            for i, line in enumerate(f):
                handles[i % n].write(line)
        for h in handles:
            h.close()
        # The bytes the replay reads: unchanged source after sharding, shard digests recorded.
        again = check_inputs.check_files(workload, Path(stack.MODEL))
        self.checks["input_conformance"] = dict(passed=attest == again, attests=attest,
                                                report=f"{workload}.validate.json",
                                                shards={p.name: sha256_file(p) for p in paths})
        log(f"input check passed: {attest}")
        return [str(p) for p in paths]

    # --------------------------------------------------------------------------------- services
    def wait_engines(self):
        t0, pending = time.time(), {w["name"]: u for w, u in zip(self.ws, self.urls)}
        while pending:
            for name, url in list(pending.items()):
                if http_ok(url + "/health"):
                    pending.pop(name)
                    log(f"worker {name} ready after {time.time() - t0:.0f}s")
            if failed := self.failure():
                raise RuntimeError(f"worker failure: {failed}")
            if time.time() - t0 > self.cfg["cold_start_deadline_s"]:
                raise TimeoutError(f"workers not ready: {sorted(pending)}")
            time.sleep(10)
        self.evidence["engines_ready_s"] = round(time.time() - t0)

    def wait_http(self, name: str, url: str, deadline_s: float):
        t0 = time.time()
        while not http_ok(url):
            p = self.services.get(name)
            if (p is not None and p.poll() is not None) or time.time() - t0 > deadline_s:
                raise RuntimeError(f"{name} not ready at {url}; see {name}.log")
            time.sleep(2)

    def backend_workers(self) -> list:
        return sorted(http_json(f"{stack.ROUTER_URL}/symmetric_pd/backend/loads", timeout=30)["workers"])

    def start_frontdoor(self):
        """Indexer (+ event bridges), every node's frontends, then the router."""
        c, run = self.cfg, Path(self.rundir)
        (run / "indexer.json").write_text(json.dumps(stack.indexer_config(self.ips), indent=1))
        self.services["indexer"] = start(stack.indexer_command(str(run / "indexer.json"), str(run / "indexer")),
                                         self.rundir, "indexer")
        t0 = time.time()
        while not (run / "indexer" / "ready.json").exists():
            if self.services["indexer"].poll() is not None or time.time() - t0 > 300:
                raise RuntimeError("indexer supervisor not ready; see indexer.log")
            time.sleep(2)
        query_url = json.loads((run / "indexer" / "ready.json").read_text())["query_url"]
        for i, url in enumerate(stack.frontend_urls(c, self.ips)):
            self.wait_http(f"t{i}", url + "/health", 300)
        (run / "prepared.json").write_text(json.dumps(stack.prepared_config(self.ips), indent=1))
        env = {"RUST_LOG": "info", "NO_PROXY": "*", "EVOLVING_INFRA_PREPARED_CONFIG": str(run / "prepared.json")}
        if c.get("controller"):
            token = Path(stack.token_file(c))
            token.parent.mkdir(parents=True, exist_ok=True)
            token.touch(mode=0o600)
            token.write_text(secrets.token_hex(32))
            env["EVOLVING_INFRA_ROLE_CONTROL_TOKEN_FILE"] = str(token)
        self.router_argv = stack.router_command(c, self.ips, self.rundir, query_url)
        self.services["router"] = start(self.router_argv, self.rundir, "router", env)
        self.wait_http("router", f"{stack.ROUTER_URL}/readyz", 900)
        deadline = time.time() + 1800
        while (workers := self.backend_workers()) != sorted(self.urls):
            if time.time() > deadline:
                raise TimeoutError(f"router lists {workers}, expected {sorted(self.urls)}")
            time.sleep(2)
        self.registered_at_start = workers
        log(f"router ready: {c['n_p']}P{len(self.ws) - c['n_p']}D")

    # --------------------------------------------------------------------------------- checks before load
    def quiet(self, label: str):
        import checks
        result = checks.wait_for_quiet(self.urls, quiet_seconds=60, timeout=1800)
        log(f"quiet window ({label}) after {result['wait_seconds']} s")
        self.checks.setdefault("quiet_window", {})[label] = dict(passed=True, wait_seconds=result["wait_seconds"],
                                                                  quiet_seconds=result["quiet_seconds"])

    def conformance(self):
        """Every worker, one after another, after a quiet window."""
        self.quiet("before_conformance")
        workers = {}
        for w, url in zip(self.ws, self.urls):
            t0 = time.time()
            p = start(stack.conformance_command(self.rundir, url, w["name"]), self.rundir, f"conformance-{w['name']}",
                      {"NO_PROXY": "*", "no_proxy": "*"})
            try:
                rc = p.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                stop(p)
                rc = "timeout"
            passed = rc == 0 and Path(self.rundir, f"conformance-{w['name']}.json").is_file()
            workers[w["name"]] = dict(passed=passed, rc=rc, seconds=round(time.time() - t0, 1),
                                      report=f"conformance-{w['name']}.json", log=f"conformance-{w['name']}.log")
            log(f"conformance {w['name']}: {'passed' if passed else 'FAILED'} rc={rc}")
        self.checks["conformance"] = dict(passed=all(v["passed"] for v in workers.values()), workers=workers)
        if not self.checks["conformance"]["passed"]:
            raise RuntimeError("conformance failed; see conformance-<worker>.log")

    def start_controller(self):
        run_dir = self.replay_dir
        conf, path = stack.controller_config(self.cfg, self.ips, str(run_dir)), stack.controller_config_path(str(run_dir))
        Path(path).write_text(json.dumps(conf, indent=1))
        self.services["controller"] = start(stack.controller_command(path), str(run_dir), "controller",
                                            {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                                             "NO_PROXY": "*", "no_proxy": "*"})
        t0 = time.time()
        while not (run_dir / "role-controller-ready.json").exists():
            if self.services["controller"].poll() is not None or time.time() - t0 > 180:
                raise RuntimeError("planner controller not ready; see replay/controller.log")
            time.sleep(1)
        log("planner controller ready (learning starts once the replay's cold start is over)")

    # --------------------------------------------------------------------------------- replay
    def scrape_loop(self, stop_event: threading.Event):
        """Worker /metrics as <UTC>-w<i>.prom for checks.check_metrics; router views beside them."""
        metrics, router = Path(self.rundir, "metrics"), Path(self.rundir, "router-snapshots")
        metrics.mkdir(exist_ok=True)
        router.mkdir(exist_ok=True)
        while True:
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            bodies = {}
            try:
                for i, url in enumerate(self.urls, 1):
                    with urllib.request.urlopen(url + "/metrics", timeout=10) as r:
                        bodies[i] = r.read()
                for i, body in bodies.items():  # a complete set or nothing
                    (metrics / f"{stamp}-w{i}.prom").write_bytes(body)
            except Exception as e:  # noqa: BLE001
                (router / f"{stamp}-metrics.err").write_text(repr(e))
            for name, path in (("loads", "/symmetric_pd/backend/loads"), ("executions", "/symmetric_pd/executions")):
                try:
                    with urllib.request.urlopen(stack.ROUTER_URL + path, timeout=10) as r:
                        (router / f"{stamp}-{name}.json").write_bytes(r.read())
                except Exception as e:  # noqa: BLE001
                    (router / f"{stamp}-{name}.err").write_text(repr(e))
            if stop_event.wait(self.cfg["metrics_period_s"]):
                return

    def replay(self, shards: list[str]) -> dict:
        c, d = self.cfg, self.replay_dir
        n = c["trie_procs"]
        per = [c["concurrency"] // n + (1 if i < c["concurrency"] % n else 0) for i in range(n)]
        env = {"EVOLVING_INFRA_REPLAY": json.dumps(stack.replay_env(c, self.ips)), "NO_PROXY": "*", "no_proxy": "*"}
        (d / "replay.json").write_text(env["EVOLVING_INFRA_REPLAY"])
        stop_event = threading.Event()
        scraper = threading.Thread(target=self.scrape_loop, args=(stop_event,), daemon=True)
        scraper.start()
        t0 = time.time()
        procs = [start(stack.trie_command(c, shards[i], per[i], str(d), i), str(d), f"trie-{i}", env) for i in range(n)]
        log(f"replay: {n} Trie processes, concurrency {per}, {c['duration']} s")
        deadline = t0 + c["duration"] + c["trie_grace_s"]
        while any(p.poll() is None for p in procs):
            if time.time() > deadline:
                log("Trie still running past duration + grace; stopping it")
                for p in procs:
                    stop(p)
                break
            if failed := self.failure():
                log(f"worker failure during the replay: {failed}; stopping Trie")
                for p in procs:
                    stop(p)
                break
            time.sleep(10)
        end = time.time()
        stop_event.set()
        scraper.join(timeout=60)
        point = dict(start=t0, end=end, elapsed_s=end - t0, rc=[p.returncode for p in procs], per_process_concurrency=per)
        (d / "points.json").write_text(json.dumps([point], indent=2))
        log(f"replay done rc={point['rc']} in {end - t0:.0f}s")
        self.commit()
        return point

    # --------------------------------------------------------------------------------- verdict
    def method_evidence(self) -> dict:
        router = self.services.get("router")
        try:
            at_end = self.backend_workers()
        except Exception as e:  # noqa: BLE001
            at_end = repr(e)
        out = dict(router_alive_at_end=router is not None and router.poll() is None,
                   all_workers_registered=self.registered_at_start == sorted(self.urls) and at_end == sorted(self.urls))
        weights = {}
        for flag, value in zip(self.router_argv, self.router_argv[1:]):
            if flag == "--symmetric-pd-worker":
                url, _, p, d = value.split(",")
                weights[url] = "prefill" if (p, d) == ("1", "0") else "decode" if (p, d) == ("0", "1") else "mixed"
        configured = {u: stack.role(w, self.cfg) for w, u in zip(self.ws, self.urls)}
        external = "--symmetric-pd-external-role-control" in self.router_argv
        if not self.cfg.get("controller"):
            out["roles_as_configured"] = weights == configured and not external
            return dict(out, required=["router_alive_at_end", "all_workers_registered", "roles_as_configured"])
        d = self.replay_dir
        journal = d / "role-decisions.jsonl"
        events = [json.loads(line)["event"] for line in journal.open()] if journal.exists() else []
        ready = d / "role-controller-ready.json"
        initial = json.loads(ready.read_text())["initial_assignment"] if ready.exists() else None
        applied = initial
        for line in (journal.open() if journal.exists() else []):
            row = json.loads(line)
            if row.get("event") == "role_update_confirmed":
                applied = row["assignment"]
        try:
            token = Path(stack.token_file(self.cfg)).read_text().strip()
            live = http_json(f"{stack.ROUTER_URL}/symmetric_pd/roles", timeout=10,
                             headers={"Authorization": "Bearer " + token})["assignment"]
        except Exception as e:  # noqa: BLE001
            live = repr(e)
        (d / "roles-final.json").write_text(json.dumps(dict(router=live, last_applied=applied), indent=1))
        controller = self.services.get("controller")
        script = self.cfg["controller"].get("role_script") or []
        out.update(
            initial_roles_as_configured=weights == configured and external,
            controller_alive_at_end=controller is not None and controller.poll() is None,
            fit_completed=(d / "calibration" / "report.json").exists() and "warmup_fit_completed" in events,
            decisions_journal_present=journal.exists() and "learning_started" in events,
            roles_consistent_with_last_applied=applied is not None and live == applied,
            scripted_changes_confirmed=not script or events.count("role_update_confirmed") == len(script),
            role_updates_confirmed=events.count("role_update_confirmed"),
            decisions={e: events.count(e) for e in sorted(set(events))},
            planner_failed=(d / "PLANNER_FAILED").read_text() if (d / "PLANNER_FAILED").exists() else None,
            final_n_p=sum(r == "prefill" for r in live["roles"].values()) if isinstance(live, dict) else None)
        return dict(out, required=["router_alive_at_end", "all_workers_registered", "initial_roles_as_configured",
                                   "controller_alive_at_end", "fit_completed",
                                   "decisions_journal_present", "roles_consistent_with_last_applied",
                                   "scripted_changes_confirmed"])

    def verdict(self, point: dict) -> dict:
        import checks
        import score
        c, d = self.cfg, self.replay_dir
        window = score.window_score(str(d), c["warmup_s"], c["duration"] - c["warmup_s"])
        results = {}
        for i in range(c["trie_procs"]):
            path = d / f"trie-results-{i}.json"
            try:
                r = checks.check_result(path)
                results[path.name] = dict(passed=True, failed_requests=r["failed_requests"],
                                          completed_model_requests=r["completed_model_requests"])
            except (OSError, ValueError) as e:
                results[path.name] = dict(passed=False, error=str(e))
        self.checks["steady_zero_failures"] = dict(
            passed=window["completed_model_requests"] > 0 and window["failed_requests"] == 0
            and all(r["passed"] for r in results.values()) and point["rc"] == [0] * c["trie_procs"]
            and not self.failure() and point["elapsed_s"] >= c["duration"],
            window_failed_requests=window["failed_requests"], window_completed=window["completed_model_requests"],
            trie_results=results, client_exit_codes=point["rc"], worker_failure=self.failure())
        try:
            resets = checks.check_metrics(Path(self.rundir, "metrics"), len(self.ws), 2)
            self.checks["monotonic_counters"] = dict(passed=not resets,
                                                     snapshots=len(list(Path(self.rundir, "metrics").glob("*-w1.prom"))))
        except (OSError, ValueError) as e:
            self.checks["monotonic_counters"] = dict(passed=False, error=str(e))
        evidence = self.method_evidence()
        method_ok = all(evidence.get(k) for k in evidence["required"])
        summary = dict(run_id=c["run_id"], name=c["name"], topology=f"{c['n_nodes']} node(s), {c['n_p']}P{len(self.ws) - c['n_p']}D",
                       window=window, checks=self.checks, method_evidence=dict(evidence, passed=method_ok),
                       elapsed_s=point["elapsed_s"], source=c.get("source"))
        summary["valid"] = bool(all(v.get("passed") for v in self.checks.values() if isinstance(v, dict) and "passed" in v)
                                and all(v["passed"] for v in self.checks["quiet_window"].values()) and method_ok)
        return summary

    # --------------------------------------------------------------------------------- run
    def run(self) -> dict:
        shards = self.check_inputs()
        self.wait_engines()
        self.start_frontdoor()
        self.conformance()
        self.quiet("before_load")
        if self.cfg.get("controller"):
            self.start_controller()
        point = self.replay(shards)
        summary = self.verdict(point)
        self.save("summary.json", summary)
        log("summary: " + json.dumps({k: summary["window"][k] for k in
                                      ("completed_model_requests", "failed_requests", "cache_hit", "ttft_p50")})
            + f" valid={summary['valid']}")
        return summary

    def stop_services(self):
        for name in ("controller", "router", "indexer"):
            stop(self.services.get(name))


# ------------------------------------------------------------------------------------------- every node

def node_main(cfg: dict, cluster, ctl_dict, results, kcache) -> dict | None:
    """`cluster`: modal.experimental.get_cluster_info(); results / kcache: mounted Volumes."""
    rank, ips = cluster.rank, list(cluster.container_ipv4_ips)
    if not ips:  # a one-container cluster gets no cluster addresses
        ips = [sh("hostname -I").split()[0]]
    assert len(ips) == cfg["n_nodes"], (ips, cfg["n_nodes"])
    ctl = Ctl(ctl_dict, cfg["run_id"])
    starts = (ctl.get(f"starts:{rank}") or 0) + 1
    ctl.put(f"starts:{rank}", starts)
    if starts > 1:  # Modal restarts the whole cluster when a container dies at startup
        raise RuntimeError(f"rank {rank}: cluster restarted (start {starts}); not rerunning {cfg['run_id']}")
    local = f"/tmp/run/{cfg['run_id']}"
    os.makedirs(local, exist_ok=True)
    syncer = Syncer(local, f"/results/{cfg['run_id']}/node{rank}", results)
    syncer.start()
    nics = nic_map()
    gpus = sh("nvidia-smi --query-gpu=name --format=csv,noheader").splitlines()
    node = dict(rank=rank, ip=ips[rank], nics=nics, gpus=gpus, task=os.environ.get("MODAL_TASK_ID"),
                kernel_cache=restore_kcache(cfg, kcache))
    Path(local, "node.json").write_text(json.dumps(node, indent=1))
    ctl.put(f"node:{rank}", node)
    log(f"rank {rank}/{len(ips)} ip={ips[rank]} gpus={len(gpus)}x{gpus[0] if gpus else '?'}")
    if rank == 0:
        Path(local, "config.json").write_text(json.dumps(cfg, indent=2))

    services = []
    for i in range(cfg["trie_procs"]):  # this node's frontends
        if stack.frontend_node(cfg, len(ips), i) == rank:
            path = Path(local, f"t{i}.json")
            path.write_text(json.dumps(stack.frontend_config(cfg, ips, i), indent=1))
            services.append(start(stack.frontend_command(str(path), i), local, f"t{i}"))
    if cfg.get("controller"):
        os.makedirs(f"{local}/{stack.TRACE_SUBDIR}", exist_ok=True)
        services.append(start(stack.trace_server_command(cfg), local, "trace-server"))
    procs = {}
    for w in (w for w in stack.workers(len(ips)) if w["node"] == rank):
        env = stack.engine_env(ips[rank])
        Path(local, f"{w['name']}.env.json").write_text(json.dumps(env, indent=1))
        procs[w["name"]] = start(stack.engine_command(w, cfg, nics), local, w["name"], env)
        log(f"launched {w['name']} ({stack.role(w, cfg)}) on GPUs {w['base_gpu']}-{w['base_gpu'] + 3}")

    def watch():
        while ctl.get("phase") != "teardown":
            for name, p in procs.items():
                if p.poll() is not None:
                    ctl.put("failed", f"{name} exited rc={p.returncode}")
                    log(f"!! {name} exited rc={p.returncode}")
                    return
            time.sleep(5)
    threading.Thread(target=watch, daemon=True).start()

    summary = None
    if rank == 0:
        driver = Driver(cfg, ips, local, ctl, syncer.sync)
        try:
            summary = driver.run()
        except Exception as e:  # noqa: BLE001
            log(f"run failed: {e!r}")
            Path(local, "FAILED").write_text(repr(e))
            summary = dict(run_id=cfg["run_id"], valid=False, error=repr(e), checks=driver.checks)
            driver.save("summary.json", summary)
        finally:
            driver.stop_services()
            ctl.put("result", summary)
            ctl.put("phase", "teardown")
    while ctl.get("phase") != "teardown":
        time.sleep(10)
    for p in services:
        stop(p, timeout=10)
    syncer.sync()
    if rank == 0:
        save_kcache(cfg, kcache)
    log("released")
    return summary
