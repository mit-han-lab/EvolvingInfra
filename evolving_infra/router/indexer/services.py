# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Run the existing Dynamo indexer and wire adapters from a deployment JSON.

This foreground supervisor owns only its subprocesses. Run it in a dedicated
process group. It exits if any child fails; it never reuses an unknown service.
Use the same Python environment as the pinned Dynamo installation.

Routers query the adapter at `query_port`, never Dynamo's own `/query`: that
endpoint takes token IDs only and rejects the prepared `block_hashes` the
distributed tokenizer sends (HTTP 422). The adapter hashes token IDs with the
engines' page form (`"eagle"`: EAGLE windows, else plain pages) and forwards
prepared hashes unchanged, both to `/query_by_hash`. ready.json records the
`query_url` a router should use.
"""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def request(url, body=None):
    """Send one JSON request to a local directory service.

    Args:
        url: HTTP health or registration endpoint.
        body: Optional JSON value; None sends GET, otherwise POST.

    Returns:
        Decoded JSON, or None for an empty body. Uses a 3-second timeout and bypasses
        environment proxies.
    """
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=3) as response:
        raw = response.read()
        # Dynamo's health endpoint is HTTP 200 with an empty response body.
        return json.loads(raw) if raw else None


def run(cfg, output):
    """Start and supervise the event bridges, Dynamo indexer and query adapter.

    Args:
        cfg: Deployment mapping: workers, local ports, model, block size and EAGLE
            flag.
        output: Path for child logs and a new ready.json; existing readiness is
            rejected.

    Returns:
        No normal return while healthy. On exit, terminates/waits for owned children
        and closes logs; propagates validation or child failures.
    """
    workers = cfg["workers"]
    if not workers or len({w["instance_id"] for w in workers}) != len(workers):
        raise ValueError("workers must have unique instance IDs")
    eagle = cfg.get("eagle")
    if type(eagle) is not bool or cfg["block_size"] <= 0:
        raise ValueError("the deployment needs a boolean eagle flag and a positive block size")
    ports = [cfg["indexer_port"], cfg["query_port"]]
    ports += [w[k] for w in workers for k in ("bridge_pub_port", "bridge_replay_port")]
    if len(set(ports)) != len(ports) or any(type(p) is not int or not 0 < p < 65536 for p in ports):
        raise ValueError("local service ports must be unique and in 1..65535")
    output.mkdir(parents=True, exist_ok=True)
    ready = output / "ready.json"
    if ready.exists():
        raise FileExistsError("use a new output directory; readiness must not be inherited")
    children, logs = [], []

    def spawn(name, args):
        """Launch one owned Python subprocess with a dedicated log.

        Args:
            name: Child log basename under output.
            args: Arguments after the current Python interpreter and -u.

        Returns:
            Started Popen object, registered for cleanup together with its opened log.
        """
        log = (output / f"{name}.log").open("ab", buffering=0)
        logs.append(log)
        proc = subprocess.Popen([sys.executable, "-u", *args], stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
        children.append(proc)
        return proc

    def healthy():
        """Check liveness of every child recorded by the supervisor.

        Args:
            None. Inspects the child process handles captured by run().

        Returns:
            None if all children are alive; raises RuntimeError when any child has exited.
        """
        for proc in children:
            if proc.poll() is not None:
                raise RuntimeError(f"indexer child {proc.pid} exited: {proc.returncode}")

    def retry(url, body=None):
        """Retry startup HTTP operations while ensuring children remain alive.

        Args:
            url: Service endpoint to probe or register with.
            body: Optional JSON POST body; None performs GET.

        Returns:
            First successful response; propagates child failure or request failure after
            the 90-second retry deadline.
        """
        deadline = time.monotonic() + 90
        while True:
            healthy()
            try:
                return request(url, body)
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.2)

    try:
        registrations = []
        for worker in workers:
            pub = f"tcp://127.0.0.1:{worker['bridge_pub_port']}"
            replay = f"tcp://127.0.0.1:{worker['bridge_replay_port']}"
            name = f"bridge-{worker['instance_id']}"
            spawn(name, ["-m", "evolving_infra.router.indexer.kv_event_bridge",
                         "--upstream-pub", worker["events_endpoint"],
                         "--upstream-replay", worker["replay_endpoint"],
                         "--downstream-pub", pub, "--downstream-replay", replay,
                         "--topic", worker.get("topic", ""),
                         "--stats-file", str(output / f"{name}-stats.json")])
            registrations.append(dict(instance_id=worker["instance_id"], endpoint=pub,
                replay_endpoint=replay, model_name=cfg["model_name"], block_size=cfg["block_size"]))
        spawn("indexer", ["-m", "dynamo.indexer", "--port", str(cfg["indexer_port"]),
                          "--block-size", str(cfg["block_size"]), "--threads", str(cfg.get("threads", 4))])
        upstream = f"http://127.0.0.1:{cfg['indexer_port']}"
        replies = [dict(request=body, response=retry(upstream + "/register", body)) for body in registrations]
        spawn("query", ["-m", "evolving_infra.router.indexer.eagle_indexer_query", "--port", str(cfg["query_port"]),
                        "--upstream", upstream, "--block-size", str(cfg["block_size"]),
                        "--eagle" if eagle else "--no-eagle"])
        query_url = f"http://127.0.0.1:{cfg['query_port']}"
        retry(query_url + "/health")
        temporary = output / "ready.json.tmp"
        temporary.write_text(json.dumps(dict(at=time.time(), registrations=replies, query_url=query_url,
            config=cfg, pids=[p.pid for p in children]), indent=2) + "\n")
        temporary.replace(ready)
        while True:
            healthy()
            time.sleep(0.5)
    finally:
        for proc in reversed(children):
            if proc.poll() is None:
                proc.terminate()
        for proc in reversed(children):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for log in logs:
            log.close()


def main():
    """Parse deployment paths and install termination handling.

    Args:
        None. Uses the enclosing service state or process arguments.

    Returns:
        None on orderly unwinding. Reads CLI/configuration and runs the foreground
        supervisor.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def terminate(*_):
        """Convert SIGTERM into normal Python cleanup unwinding.

        Args:
            _: Signal number and frame supplied by Python; ignored.

        Returns:
            Never returns; raises KeyboardInterrupt so the supervisor finally block cleans
            up children.
        """
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
