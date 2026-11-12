"""The launcher's layout, NIC detection and generated commands (no GPUs needed)."""
import json
from pathlib import Path

import pytest

from evolving_infra.launch import hardware, serve
from evolving_infra.launch.__main__ import config, parse
from evolving_infra.launch.layout import workers

TOPO = """\
\x1b[4m\tGPU0\tGPU1\tGPU2\tGPU3\tNIC0\tNIC1\tNIC2\tNIC3\tCPU Affinity\tNUMA Affinity
GPU0\t X \tNV18\tNV18\tNV18\tNODE\tNODE\tSYS\tSYS\t0-71\t0
GPU1\tNV18\t X \tNV18\tNV18\tNODE\tNODE\tSYS\tSYS\t0-71\t0
GPU2\tNV18\tNV18\t X \tNV18\tSYS\tSYS\tNODE\tNODE\t72-143\t1
GPU3\tNV18\tNV18\tNV18\t X \tSYS\tSYS\tNODE\tNODE\t72-143\t1
NIC0\tNODE\tNODE\tSYS\tSYS\t X \tPIX\tSYS\tSYS
NIC1\tNODE\tNODE\tSYS\tSYS\tPIX\t X \tSYS\tSYS

Legend:

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
  NIC2: mlx5_2
  NIC3: mlx5_3
"""
NICS = ["mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3"]


def test_each_gpu_gets_its_closest_nics():
    assert hardware.parse_topology(TOPO, NICS) == {
        0: ["mlx5_0", "mlx5_1"], 1: ["mlx5_0", "mlx5_1"], 2: ["mlx5_2", "mlx5_3"], 3: ["mlx5_2", "mlx5_3"]}
    assert hardware.parse_topology(TOPO, ["mlx5_2"])[0] == ["mlx5_2"], "only active NICs"


def test_active_nics_prefer_infiniband(tmp_path):
    for dev, link, state in [("mlx5_0", "InfiniBand", "4: ACTIVE"), ("mlx5_1", "InfiniBand", "1: DOWN"),
                             ("mlx5_bond_0", "Ethernet", "4: ACTIVE")]:
        port = tmp_path / dev / "ports" / "1"
        port.mkdir(parents=True)
        (port / "state").write_text(state)
        (port / "link_layer").write_text(link)
    assert hardware.active_nics(tmp_path) == ["mlx5_0"]


def test_layout():
    one = workers(["10.0.0.1"], 8, 4, 1)
    assert [(w.role, w.base_gpu, w.port, w.servicer_port, w.bootstrap_port) for w in one] == [
        ("prefill", 0, 30000, 18100, 8998), ("decode", 4, 31000, 19100, 9998)]
    assert len(set(one[0].ports()) | set(one[1].ports())) == len(one[0].ports()) + len(one[1].ports())
    nine = workers([f"10.0.0.{i}" for i in range(9)], 4, 4, 6)
    assert [w.role for w in nine].count("prefill") == 6 and {w.port for w in nine} == {30000}
    with pytest.raises(ValueError):
        workers(["10.0.0.1"], 4, 4, 1)  # one worker


@pytest.fixture
def stack(monkeypatch, tmp_path):
    monkeypatch.setattr(hardware, "gpu_count", lambda: 4)
    monkeypatch.setattr(hardware, "address", lambda host: {"a": "10.0.0.1", "b": "10.0.0.2"}[host])

    def make(argv, hosts=("a",), rank=0):
        return serve.Stack(config(parse(["serve", *argv])), list(hosts), rank, tmp_path)
    return make


def test_one_host_splits_its_gpus_into_one_p_and_one_d(stack):
    s = stack(["--model-path", "/models/GLM-5.2-NVFP4"])
    assert [(w.gpus, w.role) for w in s.workers] == [(2, "prefill"), (2, "decode")]
    argv = s.engine_argv(s.workers[1], {0: "mlx5_0", 2: "mlx5_2"})
    assert argv[argv.index("--base-gpu-id") + 1] == "2" and argv[argv.index("--tp-size") + 1] == "2"
    assert json.loads(argv[argv.index("--disaggregation-ib-device") + 1]) == {"0": "mlx5_0", "2": "mlx5_2"}
    assert "--sched-trace-dir" not in argv and argv[-1] == "--trust-remote-code"


def test_config_and_planner(stack):
    assert stack(["--config", "examples/gb300/glm52.yaml"]).planner is None, "off without --planner"
    s = stack(["--config", "examples/gb300/glm52.yaml", "--planner"], hosts=("a", "b"), rank=1)
    assert [(w.ip, w.gpus, w.role) for w in s.workers] == [("10.0.0.1", 4, "prefill"), ("10.0.0.2", 4, "decode")]
    assert s.eagle and s.planner == {"prefill_shape_prior": "glm52-gb300-tp4", "cold_start_max_wait_secs": 1}
    assert [w.name for w in s.mine] == ["w1"]
    argv = s.engine_argv(s.mine[0], {})
    assert "--speculative-algorithm=EAGLE" in argv and "--sched-trace-dir" in argv
    assert argv[argv.index("--context-length") + 1] == "1048576"


def test_own_index(monkeypatch):
    monkeypatch.setattr(hardware.socket, "gethostname", lambda: "rack02-node07")
    monkeypatch.setattr(hardware.socket, "getfqdn", lambda: "rack02-node07.cluster")
    assert hardware.own_index(["rack01-node03", "rack02-node07"]) == 1
    assert hardware.own_index(["rack01-node03.cluster", "rack02-node07.cluster"]) == 1


def test_one_node_config_replaces_flags_of_its_base(stack):
    s = stack(["--config", "examples/gb300/glm52-1node.yaml"])
    args = s.cfg["engine_args"]
    assert "--mem-fraction-static=0.92" in args and "--mem-fraction-static=0.80" not in args
    assert "--speculative-algorithm=EAGLE" in args and s.cfg["model"] == "/models/GLM-5.2-NVFP4"
    assert [(w.gpus, w.role) for w in s.workers] == [(2, "prefill"), (2, "decode")]


def test_h200_config_has_two_tp4_workers_per_node_without_eagle_or_retention(stack, monkeypatch):
    monkeypatch.setattr(hardware, "gpu_count", lambda: 8)
    s = stack(["--config", "examples/h200/dsv4-flash.yaml"], hosts=("a", "b"))
    assert [(w.ip, w.gpus, w.base_gpu) for w in s.workers] == [
        ("10.0.0.1", 4, 0), ("10.0.0.1", 4, 4), ("10.0.0.2", 4, 0), ("10.0.0.2", 4, 4)]
    assert len({(w.ip, p) for w in s.workers for p in w.ports()}) == sum(len(w.ports()) for w in s.workers)
    assert not s.eagle, "DSPARK is not EAGLE: the KV directory hashes plain token pages"
    assert not s.retention and s.cfg["page_size"] == 256
    assert stack(["--config", "examples/gb300/glm52.yaml"]).retention


def test_kimi_config_passes_its_tokenizer_environment_and_image_layout(stack, tmp_path):
    s = stack(["--config", "examples/gb300/kimi-k26.yaml"], hosts=("a", "b"))
    assert s.eagle and s.retention and s.cfg["tokenizer"].endswith("tokenizer.json")
    assert s.cfg["engine_env"]["SGLANG_PREFILL_STATE_PUSH_PORT"] == 26648
    assert s.cfg["frontend"]["image_config"]["layout_type"] == "kimi_k25"
    args = s.cfg["router_args"]
    assert "--trade-wait=e3" in args
    files = [a.split("=", 1)[1] for a in args if "e3/" in a]
    assert len(files) == 2 and all(f.startswith("/") and Path(f).is_file() for f in files), "full paths"


def test_a_config_keeps_the_router_flags_and_env_of_its_base(tmp_path):
    from evolving_infra.launch.__main__ import load
    (tmp_path / "base.yaml").write_text(
        "model: /m\nrouter_args: [--a=1, --b=2]\nengine_env: {X: 1, Y: 2}\nengine_args: [--e=1]\n")
    (tmp_path / "top.yaml").write_text(
        "base: base.yaml\nrouter_args: [--b=3, --c]\nengine_env: {Y: 5}\n")
    cfg = load(tmp_path / "top.yaml")
    assert cfg["router_args"] == ["--a=1", "--b=3", "--c"]
    assert cfg["engine_env"] == {"X": 1, "Y": 5}
    assert cfg["engine_args"] == ["--e=1"]
