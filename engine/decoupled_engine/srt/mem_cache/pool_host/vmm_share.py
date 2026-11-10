"""CUDA VMM host-NUMA backing for the shared host-dedup pool (backend "vmm").

``SGLANG_DSA_HOST_DEDUP_SHARED_BACKEND=vmm`` (experimental). The
source rank creates the pool as pinned CUDA VMM allocations on host NUMA
nodes (``cuMemCreate`` with ``CU_MEM_LOCATION_TYPE_HOST_NUMA``, one per NUMA
stripe run), maps them back to back in one reserved VA range and exports
POSIX FDs. Followers receive the FDs over a Unix socket (SCM_RIGHTS; driver
FDs cannot be reopened through /proc), import and map them the same way.

Unlike the memfd pool, the pages are driver-pinned from creation (not
shmem, not migratable, no ``cudaHostRegister``), which is meant to avoid
the host memory migration storm. Copies must not span two allocations,
so the run boundaries are published as the buffer's registration ranges
(``ce_2d_load`` / staged CE loads already respect those).

Unverified on GPU: host access flags and RDMA registration (ibv_reg_mr or
dmabuf) are finalized after ce-microbench 144344.
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

# cuda.h enums
CU_MEM_ALLOCATION_TYPE_PINNED = 0x1
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1
CU_MEM_LOCATION_TYPE_DEVICE = 0x1
CU_MEM_LOCATION_TYPE_HOST = 0x2
CU_MEM_LOCATION_TYPE_HOST_NUMA = 0x3
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 0x1


class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]


_DRV = None


def driver():
    """libcuda via ctypes (the process's current context is torch's)."""
    global _DRV
    if _DRV is None:
        lib = ctypes.CDLL("libcuda.so.1")
        u64, sz, vp = ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p
        sig = {
            "cuMemCreate": [
                ctypes.POINTER(u64),
                sz,
                ctypes.POINTER(CUmemAllocationProp),
                u64,
            ],
            "cuMemRelease": [u64],
            "cuMemGetAllocationGranularity": [
                ctypes.POINTER(sz),
                ctypes.POINTER(CUmemAllocationProp),
                ctypes.c_int,
            ],
            "cuMemAddressReserve": [ctypes.POINTER(u64), sz, sz, u64, u64],
            "cuMemAddressFree": [u64, sz],
            "cuMemMap": [u64, sz, sz, u64, u64],
            "cuMemUnmap": [u64, sz],
            "cuMemSetAccess": [u64, sz, ctypes.POINTER(CUmemAccessDesc), sz],
            "cuMemExportToShareableHandle": [vp, u64, ctypes.c_int, u64],
            "cuMemImportFromShareableHandle": [ctypes.POINTER(u64), vp, ctypes.c_int],
        }
        for name, argtypes in sig.items():
            f = getattr(lib, name)
            f.restype = ctypes.c_int
            f.argtypes = argtypes
        _DRV = lib
    return _DRV


def _check(rc: int, what: str) -> None:
    if rc:
        raise RuntimeError(f"{what} failed (CUresult {rc})")


def _prop(node: int, rdma: bool) -> CUmemAllocationProp:
    p = CUmemAllocationProp()
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED
    p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    p.location = CUmemLocation(CU_MEM_LOCATION_TYPE_HOST_NUMA, int(node))
    p.allocFlags.gpuDirectRDMACapable = 1 if rdma else 0
    return p


def granularity(node: int = 0) -> int:
    g = ctypes.c_size_t(0)
    p = _prop(node, rdma=False)
    _check(
        driver().cuMemGetAllocationGranularity(
            ctypes.byref(g), ctypes.byref(p), CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        ),
        "cuMemGetAllocationGranularity",
    )
    return int(g.value)


@dataclass
class VmmMapping:
    ptr: int
    reserved: int
    handles: List[int]
    sizes: List[int]  # allocation sizes (granularity-rounded)
    nodes: List[int]
    fds: List[int] = field(default_factory=list)

    def ranges(self, nbytes: int) -> List[Tuple[int, int]]:
        """(ptr, length) per allocation, clipped to the pool's ``nbytes``."""
        out, off = [], 0
        for size in self.sizes:
            length = min(size, max(nbytes - off, 0))
            if length:
                out.append((self.ptr + off, length))
            off += size
        return out


def alloc_sizes(runs: Sequence[Tuple[int, int, int]], gran: int) -> List[int]:
    """Allocation size per run: runs must start on ``gran``; the last one is
    rounded up (so the mapping is contiguous and every run keeps its node)."""
    sizes = []
    for i, (offset, length, _node) in enumerate(runs):
        if offset % gran:
            raise ValueError(f"run {i} at {offset} is not {gran}-aligned")
        if i + 1 < len(runs) and length % gran:
            raise ValueError(f"run {i} length {length} is not {gran}-aligned")
        sizes.append(math.ceil(length / gran) * gran)
    return sizes


def _map_and_grant(handles: Sequence[int], sizes: Sequence[int], device: int) -> int:
    drv = driver()
    total = sum(sizes)
    va = ctypes.c_uint64(0)
    _check(
        drv.cuMemAddressReserve(ctypes.byref(va), total, max(sizes), 0, 0),
        "cuMemAddressReserve",
    )
    off = 0
    for h, size in zip(handles, sizes):
        _check(drv.cuMemMap(va.value + off, size, 0, h, 0), "cuMemMap")
        off += size
    desc = (CUmemAccessDesc * 2)()
    desc[0].location = CUmemLocation(CU_MEM_LOCATION_TYPE_DEVICE, int(device))
    desc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    # CPU access to HOST_NUMA allocations: a HOST_NUMA access descriptor
    # (id 0 covers every stripe), as verified by ce-microbench 144344.
    desc[1].location = CUmemLocation(CU_MEM_LOCATION_TYPE_HOST_NUMA, 0)
    desc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _check(drv.cuMemSetAccess(va.value, total, desc, 2), "cuMemSetAccess(device+host)")
    return va.value


def create(
    runs: Sequence[Tuple[int, int, int]], device: int, *, rdma: bool = False
) -> VmmMapping:
    """Source: one host-NUMA allocation per (offset, length, node) run, exported."""
    drv = driver()
    gran = max(granularity(node) for _, _, node in runs)
    sizes = alloc_sizes(runs, gran)
    handles, fds = [], []
    t0 = time.monotonic()
    for (_, _, node), size in zip(runs, sizes):
        h = ctypes.c_uint64(0)
        p = _prop(node, rdma)
        _check(
            drv.cuMemCreate(ctypes.byref(h), size, ctypes.byref(p), 0),
            f"cuMemCreate(node {node})",
        )
        handles.append(h.value)
        fd = ctypes.c_int(-1)
        _check(
            drv.cuMemExportToShareableHandle(
                ctypes.byref(fd), h.value, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0
            ),
            "cuMemExportToShareableHandle",
        )
        fds.append(fd.value)
    ptr = _map_and_grant(handles, sizes, device)
    logger.info(
        "Shared host-dedup VMM pool: %d host-NUMA allocations, %.1f GiB, nodes %s, "
        "granularity %d MiB, %.1f s",
        len(handles),
        sum(sizes) / 2**30,
        sorted({n for _, _, n in runs}),
        gran >> 20,
        time.monotonic() - t0,
    )
    return VmmMapping(ptr, sum(sizes), handles, sizes, [n for _, _, n in runs], fds)


def attach(fds: Sequence[int], sizes: Sequence[int], device: int) -> VmmMapping:
    """Follower: import the source's allocations and map them the same way."""
    drv = driver()
    handles = []
    for fd in fds:
        h = ctypes.c_uint64(0)
        _check(
            drv.cuMemImportFromShareableHandle(
                ctypes.byref(h),
                ctypes.c_void_p(int(fd)),
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            ),
            "cuMemImportFromShareableHandle",
        )
        handles.append(h.value)
    ptr = _map_and_grant(handles, sizes, device)
    for fd in fds:
        os.close(fd)
    return VmmMapping(ptr, sum(sizes), handles, list(sizes), [], [])


def tensor_at(ptr: int, nbytes: int, dtype: torch.dtype, dims: tuple) -> torch.Tensor:
    """A CPU tensor over ``nbytes`` at ``ptr`` (host-accessible VMM mapping)."""
    count = math.prod(dims)
    buf = (ctypes.c_uint8 * nbytes).from_address(ptr)
    return torch.frombuffer(buf, dtype=dtype, count=count).reshape(dims)


# ---- FD passing (SCM_RIGHTS over an abstract Unix socket) ---------------


def fd_server(n_clients: int):
    """Source side: a listening socket; returns (name, serve(fds) callable)."""
    name = f"\0sglang-dedup-vmm-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(name)
    srv.listen(max(n_clients, 1))

    def serve(fds: Sequence[int], timeout_s: float = 300.0) -> None:
        srv.settimeout(timeout_s)
        try:
            for _ in range(n_clients):
                conn, _ = srv.accept()
                with conn:
                    socket.send_fds(conn, [len(fds).to_bytes(4, "little")], list(fds))
        finally:
            srv.close()

    return name, serve


def fd_client(name: str, n_fds: int, timeout_s: float = 300.0) -> List[int]:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            cli = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            cli.connect(name)
            break
        except (ConnectionRefusedError, FileNotFoundError):
            cli.close()
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)
    with cli:
        msg, fds, _flags, _addr = socket.recv_fds(cli, 16, max(n_fds, 1))
    if int.from_bytes(msg[:4], "little") != n_fds or len(fds) != n_fds:
        raise RuntimeError(f"VMM FD handoff: expected {n_fds} FDs, got {len(fds)}")
    return list(fds)
