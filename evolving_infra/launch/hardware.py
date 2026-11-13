# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""What the launcher reads from the node: GPUs, the RDMA NIC of each GPU, GPU peer memory, addresses."""
import os
from pathlib import Path
import re
import socket
import subprocess

_RELATION = {"X": 0, "PIX": 1, "PXB": 2, "PHB": 3, "NODE": 4, "SYS": 5, "NV": 6}


def gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len([g for g in visible.split(",") if g.strip()])
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        raise SystemExit("nvidia-smi is not on PATH: evolving-infra serve runs on a GPU host, "
                         "inside the image when one is used") from None
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"nvidia-smi failed: {(e.stderr or e.stdout).strip()}") from None
    return len(out.split())


def active_nics(root="/sys/class/infiniband") -> list:
    """RDMA devices with an active port: the InfiniBand ones if there are any, else the RoCE ones."""
    links = {}
    for dev in sorted(Path(root).glob("*")):
        for port in dev.glob("ports/*"):
            if "ACTIVE" in (port / "state").read_text():
                links[dev.name] = (port / "link_layer").read_text().strip()
                break
    infiniband = [d for d, link in links.items() if link == "InfiniBand"]
    return infiniband or list(links)


def parse_topology(text: str, nics: list) -> dict:
    """GPU index -> the active NICs closest to it, from ``nvidia-smi topo -m``."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    names = dict(re.findall(r"^\s*(NIC\d+):\s*(\S+)\s*$", text, re.M))
    lines = [l for l in text.splitlines() if l.strip()]
    header = next((l.split() for l in lines if l.split() and l.split()[0].startswith("GPU0")), None)
    if header is None:
        return {}
    mapping = {}
    for line in lines:
        cells = line.split()
        if not cells or not re.fullmatch(r"GPU\d+", cells[0]):
            continue
        row = dict(zip(header, cells[1:]))
        ranked = {}
        for column, relation in row.items():
            nic = names.get(column)
            if nic in nics:
                ranked.setdefault(_RELATION.get(relation.rstrip("0123456789"), 9), []).append(nic)
        if ranked:
            mapping[int(cells[0][3:])] = sorted(ranked[min(ranked)])
    return mapping


def nic_map(nics=None) -> dict:
    """GPU index -> comma-separated closest NICs; {} without RDMA devices."""
    nics = active_nics() if nics is None else nics
    if not nics:
        return {}
    try:
        topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, check=True).stdout
        mapping = parse_topology(topo, nics)
    except (OSError, subprocess.CalledProcessError):
        mapping = {}
    if len(mapping) < gpu_count():
        return {g: ",".join(nics) for g in range(gpu_count())}
    return {g: ",".join(n) for g, n in mapping.items()}


def peer_memory() -> bool:
    """Whether GPU memory is registered with the NICs through nvidia_peermem (else dma-buf)."""
    return Path("/sys/module/nvidia_peermem").exists()


def address(host: str) -> str:
    """IPv4 address of ``host``."""
    return socket.gethostbyname(host)


def own_index(hosts: list) -> int:
    """Position of this host in ``hosts``, by name or address."""
    names = {socket.gethostname(), socket.getfqdn(), socket.gethostname().split(".")[0]}
    for i, host in enumerate(hosts):
        if host in names or host.split(".")[0] in names:
            return i
    mine = set(socket.gethostbyname_ex(socket.gethostname())[2])
    for i, host in enumerate(hosts):
        if address(host) in mine:
            return i
    raise ValueError(f"this host ({socket.gethostname()}) is not in --hosts {','.join(hosts)}")
