"""Host-dedup pool shared by the TP ranks (``SGLANG_DSA_HOST_DEDUP_FANOUT=shared``).

The dedup source rank (TP0) allocates the one host copy in a memfd and keeps
the descriptor open; every follower maps the same pages through
``/proc/<pid>/fd/<n>``. Each rank then pins its own mapping
(``cudaHostRegister``) and loads its rows straight from host with copy
engines (``--hicache-io-backend direct``, ``page_first_direct`` layout), so a
load has no NCCL, no SM kernel and no cross-rank GPU dependency. Only TP0
writes the pool (backups and peer pulls); followers stay allocator-only for
every write path.

With ``SGLANG_DEDUP_POOL_NUMA_SPLIT=1`` the source applies the NUMA-split stripe
plan to the memfd mapping (``mbind`` sets the shared policy of the shmem
object, so the pages land as planned when TP0 pre-faults them).

Allocation order is identical on every rank (pools are built in the same
order), so each ``allocate`` call is one collective on the dedup gloo group.
"""

from __future__ import annotations

import logging
import math
import mmap
import os

import torch

from decoupled_engine.srt.mem_cache.pool_host.common import HostTensorAllocator

logger = logging.getLogger(__name__)

_PROT_RW = mmap.PROT_READ | mmap.PROT_WRITE
_MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)


def _memfd_map(nbytes: int):
    fd = os.memfd_create("sglang-dedup-host-pool", os.MFD_CLOEXEC)
    os.ftruncate(fd, nbytes)
    mm = mmap.mmap(fd, nbytes, flags=mmap.MAP_SHARED, prot=_PROT_RW)
    return fd, mm


def _numa_split_memfd_allocator_cls():
    from decoupled_engine.srt.mem_cache.pool_host.numa_split import NumaSplitHostTensorAllocator

    class _MemfdNumaSplit(NumaSplitHostTensorAllocator):
        """The NUMA-split plan on a memfd mapping instead of MAP_ANONYMOUS."""

        fd = None

        def _map(self, nbytes: int):
            self.fd, mm = _memfd_map(nbytes)
            return mm

    return _MemfdNumaSplit


# memfd: shmem pages (MAP_SHARED memfd), pinned per rank by cudaHostRegister.
# vmm: CUDA VMM host-NUMA allocations exported as FDs (vmm_share.py); driver
# pinned, no cudaHostRegister. Experimental; see vmm_share.py.
SHARE_BACKENDS = ("memfd", "vmm")


def share_backend() -> str:
    """``SGLANG_DSA_HOST_DEDUP_SHARED_BACKEND``: how the one pool is shared.

    The load path only sees host tensors (data_ptr + registration ranges), so
    a backend swaps here without touching it.
    """
    name = os.environ.get("SGLANG_DSA_HOST_DEDUP_SHARED_BACKEND", "memfd").lower()
    if name not in SHARE_BACKENDS:
        raise ValueError(
            f"SGLANG_DSA_HOST_DEDUP_SHARED_BACKEND={name!r} is not available "
            f"(have: {', '.join(SHARE_BACKENDS)})"
        )
    return name


class SharedDedupHostTensorAllocator(HostTensorAllocator):
    """Shared dedup pool allocator (backend: memfd or vmm)."""

    def __init__(self, host_dedup, numa_split: bool = False):
        super().__init__()
        self.backend = share_backend()
        if numa_split:
            from decoupled_engine.srt.mem_cache.pool_host.numa_split import pool_map_mode

            if pool_map_mode()[0] != "shared":
                raise ValueError(
                    "SGLANG_DSA_HOST_DEDUP_FANOUT=shared maps one pool into every "
                    "rank; it cannot use SGLANG_DEDUP_POOL_MAP=private"
                )
        self.dedup = host_dedup
        self.numa_split = numa_split
        self._src = (
            _numa_split_memfd_allocator_cls()()
            if numa_split and self.backend == "memfd"
            else None
        )
        self.fds: list = []
        self.mms: list = []
        self.vmm: list = []
        # data_ptr -> [(ptr, length)] per VMM allocation: the pool publishes
        # them as its registration ranges instead of cudaHostRegister-ing.
        self.vmm_ranges: dict = {}

    def device_mapped_ranges(self, buffer: torch.Tensor):
        """Ranges for a VMM-backed buffer (skip cudaHostRegister), else None."""
        return self.vmm_ranges.get(buffer.data_ptr())

    def _allocate_vmm(self, dims, dtype, count: int, nbytes: int) -> torch.Tensor:
        from decoupled_engine.srt.mem_cache.pool_host import vmm_share as vs

        device = torch.cuda.current_device()
        world = torch.distributed.get_world_size(self.dedup.group)
        m = serve = None
        if self.dedup.is_src:
            if self.numa_split:
                from decoupled_engine.srt.mem_cache.pool_host.numa_split import (
                    NumaSplitHostTensorAllocator,
                )

                _, nbytes_plan, runs, _ = NumaSplitHostTensorAllocator().plan_runs(
                    dims, dtype
                )
                assert nbytes_plan == nbytes, (nbytes_plan, nbytes)
            else:
                from decoupled_engine.srt.mem_cache.pool_host.numa_split import (
                    memory_nodes,
                    owner_node,
                )

                runs = [(0, nbytes, owner_node(memory_nodes()))]
            m = vs.create(runs, device)
            name, serve = vs.fd_server(world - 1)
            info = (os.getpid(), name, list(m.sizes), nbytes)
        else:
            info = None
        box = [info]
        torch.distributed.broadcast_object_list(
            box, src=self.dedup.src_global_rank, group=self.dedup.group
        )
        _pid, name, sizes, src_bytes = box[0]
        if src_bytes != nbytes:
            raise RuntimeError(
                f"Shared host-dedup VMM pool size mismatch: source {src_bytes} B, "
                f"local {nbytes} B (dims={tuple(dims)}, dtype={dtype})"
            )
        if self.dedup.is_src:
            serve(m.fds)
            for fd in m.fds:
                os.close(fd)
            m.fds = []
        else:
            fds = vs.fd_client(name, len(sizes))
            m = vs.attach(fds, sizes, device)
            logger.info(
                "Shared host-dedup pool: mapped %.1f GiB of the source's VMM pool "
                "(%d allocations) dims=%s",
                nbytes / 2**30,
                len(sizes),
                tuple(dims),
            )
        self.vmm.append(m)
        tensor = vs.tensor_at(m.ptr, nbytes, dtype, dims)
        self.vmm_ranges[tensor.data_ptr()] = m.ranges(nbytes)
        return tensor

    @property
    def fd(self):
        return self.fds[0] if self.fds else None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert device == "cpu", f"got device={device!r}"
        self.dtype, self.dims = dtype, dims
        count = math.prod(dims)
        nbytes = math.ceil(count * dtype.itemsize / mmap.PAGESIZE) * mmap.PAGESIZE
        if self.backend == "vmm":
            return self._allocate_vmm(dims, dtype, count, nbytes)
        if self.dedup.is_src:
            if self._src is not None:
                tensor = self._src.allocate(dims, dtype, device)
                fd = self._src.fd
            else:
                fd, mm = _memfd_map(nbytes)
                tensor = torch.frombuffer(mm, dtype=dtype, count=count).reshape(dims)
                mm.madvise(_MADV_POPULATE_WRITE)
                self.mms.append(mm)
            self.fds.append(fd)
            info = (os.getpid(), fd, nbytes)
        else:
            info = None
        box = [info]
        torch.distributed.broadcast_object_list(
            box, src=self.dedup.src_global_rank, group=self.dedup.group
        )
        pid, src_fd, src_bytes = box[0]
        if src_bytes != nbytes:
            raise RuntimeError(
                f"Shared host-dedup pool size mismatch: source {src_bytes} B, "
                f"local {nbytes} B (dims={tuple(dims)}, dtype={dtype})"
            )
        if self.dedup.is_src:
            return tensor
        fd = os.open(f"/proc/{pid}/fd/{src_fd}", os.O_RDWR | os.O_CLOEXEC)
        try:
            mm = mmap.mmap(fd, nbytes, flags=mmap.MAP_SHARED, prot=_PROT_RW)
        finally:
            os.close(fd)
        self.mms.append(mm)
        logger.info(
            "Shared host-dedup pool: mapped %.1f GiB of the source's memfd "
            "(pid %d fd %d) dims=%s",
            nbytes / 2**30,
            pid,
            src_fd,
            tuple(dims),
        )
        return torch.frombuffer(mm, dtype=dtype, count=count).reshape(dims)

    def log_placement(self, buffer: torch.Tensor) -> None:
        if self._src is not None and self.dedup.is_src:
            self._src.log_placement(buffer)
