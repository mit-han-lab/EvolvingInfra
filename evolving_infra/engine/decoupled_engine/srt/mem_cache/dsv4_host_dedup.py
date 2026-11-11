"""One host KV payload per DeepSeek-V4 attention-TP group.

Uses the MLA host-dedup design: a source host pool, allocator-only followers,
and layerwise NCCL gather/broadcast/scatter. Physical page IDs remain local.
V4 broadcasts complete packed page rows, including quantization scales,
SWA/draft KV and compressor state, rather than assuming an MLA token layout.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class DeepSeekV4HostDedup:
    # Shared V4/DSA broadcast buffer; larger restores remain chunked.
    STAGING_BYTES = 32 * 1024**2
    # Host-dedup kind for the pool NUMA-split auto policy
    # (pool_host/numa_split_policy.py); DSAHostDedup overrides it.
    dedup_kind = "dsv4"
    # True when every rank maps the source's pool and loads it itself
    # (DSA ``SGLANG_DSA_HOST_DEDUP_FANOUT=shared``): no fan-out at all.
    shared = False

    def __init__(self, group, src_global_rank: int, device, prewarm: bool = True):
        self.group = group
        self.src_global_rank = src_global_rank
        self.is_src = torch.distributed.get_rank() == src_global_rank
        self.device = device
        self.staging = torch.empty(self.STAGING_BYTES, dtype=torch.uint8, device=device)
        if not prewarm:
            # The fan-out does not use NCCL (e.g. the CUDA IPC fan-out).
            return
        # Initialize NCCL before large host allocations and model warmup. This
        # follows the MLA dedup prewarm fix, avoiding late communicator OOM.
        self.staging.zero_()
        torch.distributed.broadcast(
            self.staging, src=self.src_global_rank, group=self.group
        )
        torch.cuda.current_stream().synchronize()

    @property
    def is_dummy(self) -> bool:
        return not self.is_src

    def broadcast_layer(self, host_pool, device_indices, layer_id: int) -> None:
        if device_indices is None or device_indices.numel() == 0:
            return
        # Do not sort: corresponding logical pages may have different physical
        # IDs on each rank, especially after peer leases delay source reuse.
        rows = host_pool._to_page_indices(device_indices).to(
            device=self.device, dtype=torch.int64
        )
        buffers = getattr(host_pool, "device_page_views", None)
        if buffers is None:
            buffers = host_pool.device_buffers
        layer = buffers[layer_id]
        # Whole device rows: with a tail-only window host layout the source
        # restored only each row's tail, and the rest is its never-read head.
        self.broadcast_rows(layer, rows)

    def broadcast_rows(self, layer, rows) -> None:
        """Fan out logical rows while retaining each rank's physical indices."""
        if not layer.is_contiguous():
            raise ValueError("Host dedup requires contiguous device rows")
        layer = layer.view(torch.uint8).reshape(layer.shape[0], -1)
        row_bytes = layer.shape[1]
        rows_per_chunk = self.staging.numel() // row_bytes
        if rows_per_chunk == 0:
            raise ValueError("Host-dedup staging is smaller than one row")
        for start in range(0, rows.numel(), rows_per_chunk):
            indices = rows[start : start + rows_per_chunk]
            chunk = self.staging[: indices.numel() * row_bytes].view(-1, row_bytes)
            if self.is_src:
                torch.index_select(layer, 0, indices, out=chunk)
            self._broadcast_staging(chunk)
            if not self.is_src:
                layer.index_copy_(0, indices, chunk)

    def _broadcast_staging(self, chunk) -> None:
        torch.distributed.broadcast(chunk, src=self.src_global_rank, group=self.group)

    def destroy(self) -> None:
        if self.group is not None:
            torch.distributed.destroy_process_group(self.group)
            self.group = None


def maybe_build_dsv4_host_dedup(params, kvcache, storage_backend, extra_config):
    from decoupled_engine.srt.runtime_context import get_memory

    memory = get_memory()
    if not getattr(memory, "enable_dsv4_hicache_host_dedup", False):
        return None
    return build_host_dedup(
        params, kvcache, storage_backend, extra_config, DeepSeekV4HostDedup
    )


def build_host_dedup(
    params,
    kvcache,
    storage_backend,
    extra_config,
    dedup_cls,
    backend: str = "nccl",
    io_layout: tuple = ("kernel", "page_first"),
    **dedup_kwargs,
):
    from decoupled_engine.srt.runtime_context import get_memory, get_parallel

    memory = get_memory()
    parallel = get_parallel()
    if (
        params.attn_cp_size != 1
        or params.pp_size != 1
        or parallel.dcp_size != 1
        or parallel.dp_size != 1
        or parallel.nnodes != 1
    ):
        raise ValueError("Host dedup currently requires single-node TP, CP=DP=PP=1")
    if memory.hicache_host_memory_mode != "cache":
        raise ValueError("Host dedup requires --hicache-host-memory-mode cache")
    if memory.enable_hisparse:
        raise ValueError("Host dedup is incompatible with --enable-hisparse")
    # The direct layer-first path sorts by rank-local host page IDs, which need
    # not preserve the same logical broadcast order across TP ranks.
    if (memory.hicache_io_backend, memory.hicache_mem_layout) != tuple(io_layout):
        raise ValueError(
            f"This host dedup mode requires --hicache-io-backend {io_layout[0]} and "
            f"--hicache-mem-layout {io_layout[1]}"
        )
    if getattr(kvcache, "_unified_kv", False):
        raise ValueError("Host dedup currently requires separate paged KV pools")
    if kvcache.device != "cuda" and not str(kvcache.device).startswith("cuda:"):
        raise ValueError("Host dedup requires CUDA")
    if storage_backend is not None and not (
        storage_backend == "dynamic"
        and (extra_config or {}).get("backend_name") == "peer_kv"
    ):
        raise ValueError("Host dedup supports local HiCache and peer_kv storage")
    group = params.attn_tp_cache_group or params.tp_cache_group
    if group is None:
        raise ValueError("Host dedup requires an initialized TP cache group")
    ranks = torch.distributed.get_process_group_ranks(group)
    if len(ranks) == 1:
        return None
    from decoupled_engine.srt.distributed.parallel_state import (
        create_custom_parallel_group,
        in_the_same_node_as,
    )

    if not all(in_the_same_node_as(group, source_rank=0)):
        raise ValueError("Host dedup requires all TP ranks on the same host")
    broadcast_group = create_custom_parallel_group(ranks, backend=backend)
    result = dedup_cls(broadcast_group, ranks[0], kvcache.device, **dedup_kwargs)
    logger.info(
        "Host dedup: source=%d, ranks=%s, local_role=%s",
        ranks[0],
        ranks,
        "source" if result.is_src else "allocator-only",
    )
    return result
