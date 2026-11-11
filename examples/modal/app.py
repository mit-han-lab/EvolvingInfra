"""Reference benchmarks of evolving-infra on Modal H200 nodes (see README.md in this directory).

  modal run examples/modal/app.py::check_image                 # CPU: image contents and name checks
  modal run --detach examples/modal/app.py::static             # 1 node, 1P1D, 1200 s replay
  modal run --detach examples/modal/app.py::dynamic            # 2 nodes, 2P2D -> 1P3D -> 2P2D, 1800 s
  python3 examples/modal/app.py --dry-run [--dynamic]          # offline (needs the modal package): commands, configs, name checks

The image is docker/Dockerfile of this checkout, built by Modal. Results go to the Volume RESULTS
under <run_id>/node<rank>/; rank 0's summary.json holds the verdict. App, Volume and Dict names can
be changed with EVOLVING_INFRA_APP / _RESULTS / _KCACHE / _CTL so that parallel deployments never
share state.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import modal
import modal.experimental

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stack  # noqa: E402

# The checkout (locally); inside a container this file is /root/app.py and the repository is in the image.
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path(stack.REPO)
APP = os.environ.get("EVOLVING_INFRA_APP", "evolving-infra-bench")
RESULTS = os.environ.get("EVOLVING_INFRA_RESULTS", "evolving-infra-bench-runs")
KCACHE = os.environ.get("EVOLVING_INFRA_KCACHE", "evolving-infra-bench-kcache")
CTL = os.environ.get("EVOLVING_INFRA_CTL", "evolving-infra-bench-ctl")
# Read-only inputs: model weights, and the volume holding the workload with its validation report
# (layout in README.md). Point these at your own Volumes.
WEIGHTS = os.environ.get("EVOLVING_INFRA_WEIGHTS", "evolving-infra-weights")
INPUTS = os.environ.get("EVOLVING_INFRA_INPUTS", "evolving-infra-inputs")
GPU = "H200:8"

image = (
    modal.Image.from_dockerfile(REPO / "docker" / "Dockerfile", context_dir=REPO)
    .entrypoint([])
    .env({
        # Modal's container runtime: Mooncake busy-polls its completion queues and registers GPU memory
        # through DMA-BUF (peermem registration fails there); see docker/README.md.
        "MC_RDMA_POLLING_ONLY": "1", "WITH_NVIDIA_PEERMEM": "0",
        # JIT caches on local disk (restored from / saved to the kcache Volume by rank 0).
        "SGLANG_CACHE_DIR": "/cache", "TILELANG_CACHE_DIR": "/cache/tilelang",
        "SGLANG_DG_CACHE_DIR": "/cache/deep_gemm", "TRITON_CACHE_DIR": "/cache/triton",
    })
    .add_local_python_source("stack", "node", "score")
)

app = modal.App(APP)
results = modal.Volume.from_name(RESULTS, create_if_missing=True, version=2)
kcache = modal.Volume.from_name(KCACHE, create_if_missing=True)
ctl = modal.Dict.from_name(CTL, create_if_missing=True)
VOLUMES = {
    "/weights": modal.Volume.from_name(WEIGHTS).with_mount_options(read_only=True),
    "/inputs": modal.Volume.from_name(INPUTS).with_mount_options(read_only=True),
    "/results": results,
    "/kcache": kcache,
}
NODE = dict(image=image, gpu=GPU, cloud="gcp", cpu=64, memory=1_900_000, retries=0,
            single_use_containers=True, volumes=VOLUMES)


def cluster_run(cfg: dict) -> dict | None:
    import node
    return node.node_main(cfg, modal.experimental.get_cluster_info(), ctl, results, kcache)


@app.function(**NODE, timeout=3 * 3600)
@modal.experimental.clustered(size=1, rdma=True)
def run_static(cfg: dict):
    return cluster_run(cfg)


@app.function(**NODE, timeout=3 * 3600)
@modal.experimental.clustered(size=2, rdma=True)
def run_dynamic(cfg: dict):
    return cluster_run(cfg)


# ------------------------------------------------------------------------------------------- source

def source() -> dict:
    """Where this run's code came from (recorded, never used to refuse a run)."""
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    return dict(commit=git("rev-parse", "HEAD"), branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(git("status", "--porcelain")), engine_tree=git("rev-parse", "HEAD:engine"),
                app=APP, results_volume=RESULTS, image="docker/Dockerfile (built on Modal)")


def new_config(dynamic: bool, duration: int = 0) -> dict:
    run_id = ("dynamic-" if dynamic else "static-") + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    cfg = stack.bench_config(run_id, dynamic, duration)
    cfg["source"] = source()
    return cfg


def launch(cfg: dict, function):
    print(json.dumps(dict(run_id=cfg["run_id"], app=APP, results=f"{RESULTS}:/{cfg['run_id']}/"), indent=1), flush=True)
    result = function.remote(cfg)
    print(json.dumps(result if result is not None else ctl.get(f"{cfg['run_id']}:result"), indent=2, default=str)[:20000])


@app.local_entrypoint()
def static(duration: int = 0):
    """duration 0: 1200 s (600 s warm-up, 600 s scored)."""
    launch(new_config(False, duration), run_static)


@app.local_entrypoint()
def dynamic(duration: int = 0):
    """duration 0: 1800 s (fit at 600 s, 2P2D -> 1P3D at 900 s, back to 2P2D at 1350 s; scored 900-1800 s)."""
    launch(new_config(True, duration), run_dynamic)


# ------------------------------------------------------------------------------------------- checks

def name_checks(cfg: dict, engine_root: Path, router_cli: Path) -> dict:
    """Every engine flag is a ServerArgs field or alias, every SGLANG_* name an environ.py field, and
    every router flag a clap field of the router CLI."""
    import re
    args = (engine_root / "srt" / "server_args.py").read_text()
    fields = set(re.findall(r"^    ([a-z_0-9]+): A\[", args, re.M)) | set(re.findall(r"^    ([a-z_0-9]+): [A-Za-z\[]", args, re.M))
    aliases = set(re.findall(r'"(--[a-z0-9-]+)"', args))
    environ = set(re.findall(r"^\s+(SGLANG_[A-Z0-9_]+)\s*=", (engine_root / "srt" / "environ.py").read_text(), re.M))
    cli = router_cli.read_text()
    clap = set(re.findall(r"pub ([a-z_0-9]+):", cli)) | {x.replace("-", "_") for x in re.findall(r'long = "([a-z0-9-]+)"', cli)}
    ips, nics = [f"10.0.0.{i + 1}" for i in range(cfg["n_nodes"])], {str(i): f"mlx5_{i}" for i in range(8)}
    w = stack.workers(len(ips))[0]
    engine_flags = {a for a in stack.engine_command(w, cfg, nics) if a.startswith("--")}
    router_flags = {a for a in stack.router_command(cfg, ips, "/tmp/x", "http://127.0.0.1:8091") if a.startswith("--")}
    return dict(
        engine_flags_missing=sorted(f for f in engine_flags if f[2:].replace("-", "_") not in fields and f not in aliases),
        engine_env_missing=sorted(n for n in stack.ENGINE_ENV if n not in environ),
        router_flags_missing=sorted(f for f in router_flags if f[2:].replace("-", "_") not in clap),
        checked=dict(engine_flags=len(engine_flags), engine_env=len(stack.ENGINE_ENV), router_flags=len(router_flags)))


@app.function(image=image, cpu=4, memory=16384, timeout=1800, volumes={"/weights": VOLUMES["/weights"],
                                                                       "/inputs": VOLUMES["/inputs"]})
def check_image() -> dict:
    sh = lambda c: subprocess.run(["bash", "-c", c], capture_output=True, text=True, timeout=900)
    out = {}
    for name, cmd in {
        "engine": "python3 -c 'import decoupled_engine, decoupled_engine.srt.server_args as s; print(decoupled_engine.__file__)'",
        "engine_cli": "python3 -m decoupled_engine.launch_server --help | head -3",
        "router": f"{stack.ROUTER_BIN} --version",
        "services": f"{stack.SERVICE_PY} -c 'import dynamo.llm, evolving_infra.router.frontend.server, "
                    "evolving_infra.router.indexer.services, evolving_infra.planner.managers.planner_manager; print(\"ok\")'",
        "planner_cli": f"{stack.SERVICE_PY} -m evolving_infra.planner --help | head -3",
        "trie": f"{stack.TRIE_PY} -c 'import trie.cli, evolving_infra.router.client; print(\"ok\")'",
        "replay": f"cd {stack.REPLAY_DIR} && {stack.TRIE_PY} -c 'import prepared_replay, session_replay, replay_agent_identity; print(\"ok\")'",
        "mooncake": "python3 -c 'import mooncake, os; print(os.path.dirname(mooncake.__file__))' && grep -c 'disabled for the patched engine' $(python3 -c 'import mooncake, os; print(os.path.dirname(mooncake.__file__))')/buffer_pool.py",
        "inputs": f"ls -la {stack.WORKLOAD} {stack.WORKLOAD}.validate.json {stack.MODEL}/tokenizer.json",
        "versions": "pip list 2>/dev/null | grep -i -E '^(decoupled-engine|sglang-kernel|flashinfer-python|mooncake-transfer-engine|torch|transformers) '",
    }.items():
        p = sh(cmd)
        out[name] = dict(rc=p.returncode, out=(p.stdout + p.stderr).strip()[-1500:])
    cfg = stack.bench_config("check", True)
    out["names"] = name_checks(cfg, Path("/opt/evolving-infra/engine/decoupled_engine"),
                               Path("/opt/evolving-infra/router/core/src/config/cli.rs"))
    return out


@app.local_entrypoint()
def check_image_main():
    print(json.dumps(check_image.remote(), indent=1))


def dry_run(dynamic: bool):
    cfg = stack.bench_config("dry-run", dynamic)
    cfg["source"] = source()
    ips, nics = [f"10.0.0.{i + 1}" for i in range(cfg["n_nodes"])], {str(i): f"mlx5_{i}" for i in range(8)}
    ws = stack.workers(len(ips))
    print("== config\n" + json.dumps(cfg, indent=1))
    for w in ws[:2]:
        print(f"== {w['name']} ({stack.role(w, cfg)})\n" + subprocess.list2cmdline(stack.engine_command(w, cfg, nics)))
    print("== router\n" + subprocess.list2cmdline(stack.router_command(cfg, ips, "/tmp/run/dry-run", "http://127.0.0.1:8091")))
    print("== frontend 0\n" + json.dumps(stack.frontend_config(cfg, ips, 0)))
    print("== indexer\n" + json.dumps(stack.indexer_config(ips)))
    print("== trie 0\n" + subprocess.list2cmdline(stack.trie_command(cfg, "/tmp/trie-shards/x/shard-0.jsonl", 8, "/tmp/run/dry-run/replay", 0)))
    print("== EVOLVING_INFRA_REPLAY\n" + json.dumps(stack.replay_env(cfg, ips)))
    print("== conformance w00\n" + subprocess.list2cmdline(stack.conformance_command("/tmp/run/dry-run", stack.worker_url(ws[0], ips), ws[0]["name"])))
    if dynamic:
        print("== controller.json\n" + json.dumps(stack.controller_config(cfg, ips, "/tmp/run/dry-run/replay"), indent=1))
        print("== trace server\n" + subprocess.list2cmdline(stack.trace_server_command(cfg)))
    print("== name checks\n" + json.dumps(name_checks(cfg, REPO / "engine" / "decoupled_engine",
                                                     REPO / "router" / "core" / "src" / "config" / "cli.rs"), indent=1))


if __name__ == "__main__":
    dry_run("--dynamic" in sys.argv)
