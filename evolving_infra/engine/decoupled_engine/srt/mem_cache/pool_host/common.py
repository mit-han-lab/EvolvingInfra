# Adapted from SGLang python/sglang/srt/mem_cache/pool_host/common.py (commit 9a489f8d2f), modified for evolving-infra.
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict

import torch

from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.mem_cache.storage.mmap import alloc_mmap

logger = logging.getLogger(__name__)

_CUDA_HOST_REGISTERED_RANGES_ATTR = "_sglang_cuda_host_registered_ranges"
_VMM_MAPPED_ATTR = "_sglang_vmm_mapped"
# Parallel blocks per SM for page-first host-to-device restoration.
HICACHE_LOAD_BLOCK_QUOTA = 16


def host_buffer_registration_ranges(buffer: torch.Tensor) -> list[tuple[int, int]]:
    """Reuse page-aligned CUDA registration boundaries for peer RDMA.

    Registering a large pinned tensor as one RDMA region can exhaust HCA
    resources even when its individual CUDA registration spans succeed.
    Preserve those spans so neither a page transfer nor a CUDA copy crosses
    a registration boundary. Other allocators keep whole-buffer registration.
    """
    ranges = getattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, None)
    if ranges is not None:
        return ranges
    return [(buffer.data_ptr(), buffer.numel() * buffer.element_size())]


class HostTensorAllocator:
    def __init__(self):
        """Initialize the HostTensorAllocator."""
        self.dtype = None
        self.dims = None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert (
            device == "cpu"
        ), f"HostTensorAllocator only supports CPU allocations; got device={device!r}"
        self.dtype = dtype
        self.dims = dims
        return alloc_mmap(dims, dtype)


class ShmHostTensorAllocator(HostTensorAllocator):
    def __init__(self):
        super().__init__()
        self.fds = []
        self.mms = []

    @property
    def fd(self):
        return self.fds[0] if self.fds else None

    @property
    def mm(self):
        return self.mms[0] if self.mms else None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert (
            device == "cpu"
        ), f"ShmHostTensorAllocator only supports CPU allocations; got device={device!r}"
        self.dtype = dtype
        self.dims = dims
        from decoupled_engine.srt.mem_cache.storage.mmap import alloc_shm

        tensor, fd, mm = alloc_shm(dims, dtype)
        self.fds.append(fd)
        self.mms.append(mm)
        return tensor

    def __del__(self):
        for fd in getattr(self, "fds", []):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.fds = []


def get_allocator_from_storage(allocator_type, host_dedup=None):
    if host_dedup is not None and getattr(host_dedup, "shared", False):
        # One pool mapped by every TP rank (SGLANG_DSA_HOST_DEDUP_FANOUT=shared).
        if allocator_type in ("mooncake", "mori", "shm"):
            raise ValueError(
                "The shared host-dedup pool supports only the default host "
                f"allocator, got {allocator_type!r}"
            )
        from decoupled_engine.srt.mem_cache.pool_host.shared_dedup import (
            SharedDedupHostTensorAllocator,
        )

        # Split only on an explicit SGLANG_DEDUP_POOL_NUMA_SPLIT=1 (strict, as
        # before); the GB300 auto default resolves the source pool below only.
        return SharedDedupHostTensorAllocator(
            host_dedup, numa_split=bool(envs.SGLANG_DEDUP_POOL_NUMA_SPLIT.get())
        )
    # Only the dedup source pool; followers allocate no host memory.
    if host_dedup is not None and not host_dedup.is_dummy:
        from decoupled_engine.srt.mem_cache.pool_host.numa_split_policy import (
            UNSUPPORTED_ALLOCATOR_TYPES,
            resolve_dedup_pool_numa_split,
        )

        decision = resolve_dedup_pool_numa_split(
            allocator_type, dedup_kind=getattr(host_dedup, "dedup_kind", None)
        )
        if decision.enabled:
            # Auto never enables it for these; an explicit =1 is rejected.
            if allocator_type in UNSUPPORTED_ALLOCATOR_TYPES:
                raise ValueError(
                    "SGLANG_DEDUP_POOL_NUMA_SPLIT supports only the default host "
                    f"allocator, got {allocator_type!r}"
                )
            from decoupled_engine.srt.mem_cache.pool_host.numa_split import (
                NumaSplitHostTensorAllocator,
            )

            return NumaSplitHostTensorAllocator(
                strict=decision.strict,
                default_local_fraction=decision.default_local_fraction,
                default_map_mode=decision.map_mode or "shared",
            )
    if allocator_type == "mooncake":
        try:
            from decoupled_engine.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeHostTensorAllocator,
            )

            return MooncakeHostTensorAllocator()
        except ImportError:
            logger.warning(
                "Mooncake's tensor allocator requires mooncake >= 0.3.8.post1. "
                "Please upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'. "
                "Fallback to use default allocator."
            )
            return HostTensorAllocator()
    elif allocator_type == "mori":
        try:
            from decoupled_engine.srt.mem_cache.storage.umbp.umbp_host_allocator import (
                UMBPHostTensorAllocator,
            )

            return UMBPHostTensorAllocator()
        except (ImportError, RuntimeError) as exc:
            logger.warning(
                "UMBPHostTensorAllocator unavailable (%s). "
                "Falling back to torch.empty-based allocator.",
                exc,
            )
            return HostTensorAllocator()
    elif allocator_type == "shm":
        return ShmHostTensorAllocator()
    else:
        return HostTensorAllocator()


def get_allocator_type() -> str:
    """The host-allocator kind the published HiCache configuration asks for."""
    from decoupled_engine.srt.runtime_context import get_memory

    backend = get_memory().hicache_storage_backend
    if backend == "shm":
        return "shm"
    if backend == "dynamic":
        extra_config_str = get_memory().hicache_storage_backend_extra_config
        if extra_config_str:
            try:
                config = json.loads(extra_config_str)
                if config.get("allocator") == "shm":
                    return "shm"
            except Exception:
                pass
    return backend or "default"


def _cuda_host_register(
    buffer: torch.Tensor, registration_granularity_bytes: int | None = None
) -> None:
    # Avoid oversized cudaHostRegister calls on large host pools.
    cudart = torch.cuda.cudart()
    base = buffer.data_ptr()
    total = buffer.numel() * buffer.element_size()
    chunk_limit_bytes = (
        max(envs.SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB.get(), 1) * 1024**3
    )
    # Preserve the legacy single-call behavior unless the caller provides a
    # copy granularity. Splitting an unknown page-first layout at an arbitrary
    # byte offset can make one cudaMemcpyBatchAsync span two registrations.
    chunk_bytes = total
    if registration_granularity_bytes is not None:
        if registration_granularity_bytes <= 0:
            raise ValueError(
                "registration_granularity_bytes must be positive, got "
                f"{registration_granularity_bytes}"
            )
        if registration_granularity_bytes > chunk_limit_bytes:
            raise ValueError(
                "Host registration granularity exceeds the configured chunk limit: "
                f"granularity={registration_granularity_bytes}, "
                f"chunk_limit={chunk_limit_bytes}"
            )
        chunk_bytes = (
            chunk_limit_bytes // registration_granularity_bytes
        ) * registration_granularity_bytes
    registered_ranges: list[tuple[int, int]] = []
    try:
        offset = 0
        while offset < total:
            size = min(chunk_bytes, total - offset)
            ptr = base + offset
            rc = int(cudart.cudaHostRegister(ptr, size, 0))
            if rc != 0:
                raise RuntimeError(
                    f"cudaHostRegister failed (rc={rc}, "
                    f"{cudart.cudaGetErrorString(rc)}) at offset={offset} size={size} "
                    f"(total={total}, chunk_limit={chunk_bytes}); host buffer is not "
                    f"pinned and device transfers may silently return stale data."
                )
            registered_ranges.append((ptr, size))
            offset += size

        # Keep the exact registration bases alive with the tensor. CUDA requires
        # cudaHostUnregister to receive each base pointer, not just the tensor's
        # original base once after several independent registrations.
        setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, registered_ranges)
    except Exception:
        remaining_ranges = _cuda_host_unregister_ranges(
            cudart, registered_ranges, operation="registration rollback"
        )
        if remaining_ranges:
            setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, remaining_ranges)
        raise


def _cuda_host_unregister_ranges(
    cudart, registered_ranges: list[tuple[int, int]], *, operation: str
) -> list[tuple[int, int]]:
    failed_ranges = []
    for ptr, size in reversed(registered_ranges):
        rc = int(cudart.cudaHostUnregister(ptr))
        if rc != 0:
            failed_ranges.append((ptr, size))
            logger.warning(
                "cudaHostUnregister failed during %s (rc=%d, %s) "
                "for ptr=%#x size=%d",
                operation,
                rc,
                cudart.cudaGetErrorString(rc),
                ptr,
                size,
            )
    failed_ranges.reverse()
    return failed_ranges


def _cuda_host_unregister(buffer: torch.Tensor) -> None:
    if getattr(buffer, _VMM_MAPPED_ATTR, False):
        return  # never cudaHostRegister-ed
    cudart = torch.cuda.cudart()
    registered_ranges = getattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, None)
    if registered_ranges is None:
        # Compatibility for buffers registered before range metadata was added.
        registered_ranges = [
            (buffer.data_ptr(), buffer.numel() * buffer.element_size())
        ]
    if not registered_ranges:
        return

    remaining_ranges = _cuda_host_unregister_ranges(
        cudart, registered_ranges, operation="host-pool destroy"
    )
    setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, remaining_ranges)


def alloc_with_host_register(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
    registration_granularity_bytes: int | None = None,
) -> torch.Tensor:
    """
    Allocate tensor and register host memory with cudaHostRegister.
    CudaHostRegister only applies when pin_memory=True.
    """
    buffer = allocator.allocate(dims, dtype=dtype, device=device)
    mapped = getattr(allocator, "device_mapped_ranges", None)
    ranges = mapped(buffer) if mapped is not None else None
    if ranges is not None:
        # Driver-pinned (CUDA VMM) pool: no cudaHostRegister; its allocation
        # boundaries act as registration ranges (no copy spans two).
        setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, list(ranges))
        setattr(buffer, _VMM_MAPPED_ATTR, True)
    elif pin_memory:
        _cuda_host_register(buffer, registration_granularity_bytes)
    log_placement = getattr(allocator, "log_placement", None)
    if log_placement is not None:
        log_placement(buffer)
    return buffer


def alloc_with_pin_memory(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
    registration_granularity_bytes: int | None = None,
) -> torch.Tensor:
    """
    Allocate tensor using PyTorch's built-in pin_memory flag.
    """
    buffer = torch.empty(dims, dtype=dtype, device=device, pin_memory=pin_memory)
    return buffer


ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_pin_memory,
        "musa": alloc_with_pin_memory,
    },
)
