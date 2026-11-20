"""Deduplicate MLA/DSA host cache across attention-TP ranks.

Upstream primitives (MLAHostDedupBroadcaster, MLAHostDedupContext) back the
external-linker MLA dedup (unified_cache/linker_mla_dedup.py).

EI (MLAHostDedup, maybe_build_mla_host_dedup, at the end of this file): one
CPU payload per TP group for replicated MLA target and draft KV, built on the
DSA/DSV4 host-dedup machinery (HiCache host tier).

Fan-out of the source's loaded rows to the other TP ranks, selected with its
own word ``SGLANG_MLA_HOST_DEDUP_FANOUT`` (never the DSA word, so a leaked
``SGLANG_DSA_HOST_DEDUP_FANOUT`` cannot change an MLA baseline):

- ``nccl`` (default): layerwise NCCL broadcast (kernel IO, page_first).
- ``ce_peer`` (C2g, opt-in): the source loads its host pool with copy engines;
  followers pull the same rows from its device buffers with copy-engine peer
  copies, CPU-gated (direct IO, page_first_direct). Same machinery as DSA
  (dsa_host_dedup.py, ce_peer.py, ce_loader.py); MLA has no indexer, and a
  separate EAGLE3 draft pool (HiCache SIDECAR) is mapped as its own layers.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import torch

from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.layers.dp_attention import is_dp_attention_enabled
from decoupled_engine.srt.mem_cache.dsa_host_dedup import DSAHostDedup
from decoupled_engine.srt.mem_cache.dsv4_host_dedup import (
    DeepSeekV4HostDedup,
    build_host_dedup,
)
from decoupled_engine.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
)
from decoupled_engine.srt.runtime_context import get_parallel
from decoupled_engine.srt.utils import is_cuda

logger = logging.getLogger(__name__)


# These backends tolerate buffer-less host pools on non-source ranks.
_DEDUP_COMPATIBLE_STORAGE = frozenset({None, "", "file"})


def storage_supports_host_dedup(storage_backend: Optional[str]) -> bool:
    """Whether MLA/DSA host-memory dedup can engage with this storage backend."""
    return storage_backend in _DEDUP_COMPATIBLE_STORAGE


def mla_dedup_rank_and_size() -> tuple[int, int]:
    """Attn-TP rank/size when DP attention is enabled, model-TP otherwise."""
    parallel = get_parallel()
    if is_dp_attention_enabled():
        return parallel.attn_tp_rank, parallel.attn_tp_size
    return parallel.tp_rank, parallel.tp_size


def mla_host_dedup_eligible(kv_cache, storage_backend: Optional[str]) -> bool:
    """Rank-independent gate. CUDA only; FP4 excluded (its per-rank scale
    buffer is not covered by the broadcast)."""
    return (
        isinstance(kv_cache, MLATokenToKVPool)
        and not isinstance(kv_cache, MLATokenToKVPoolFP4)
        and is_cuda()
        and storage_supports_host_dedup(storage_backend)
    )


class MLAHostDedupBroadcaster:
    """Layerwise MLA/DSA broadcast over a dedicated NCCL group."""

    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        group: torch.distributed.ProcessGroup,
        src_global_rank: int,
    ):
        self.device_pool = device_pool
        self.group = group
        self.src_global_rank = src_global_rank
        self.is_src = mla_dedup_rank_and_size()[0] == 0
        self.layer_num = device_pool.layer_num
        self.device = device_pool.device
        self.chunk_tokens = envs.SGLANG_MLA_DEDUP_CHUNK_TOKENS.get()
        if self.chunk_tokens <= 0:
            raise ValueError(
                "SGLANG_MLA_DEDUP_CHUNK_TOKENS must be positive, "
                f"got {self.chunk_tokens}."
            )
        self.kv_staging = torch.empty(
            self.layer_num * self.chunk_tokens * device_pool.kv_cache_dim,
            dtype=device_pool.kv_buffer[0].dtype,
            device=self.device,
        )
        self.idx_bufs = None
        self.idx_elem = None
        self.idx_staging = None
        if isinstance(device_pool, DSATokenToKVPool):
            self.idx_bufs = device_pool.index_k_with_scale_buffer
            self.idx_elem = math.prod(self.idx_bufs[0].shape[1:]) or 1
            self.idx_staging = torch.empty(
                self.layer_num * self.chunk_tokens * self.idx_elem,
                dtype=self.idx_bufs[0].dtype,
                device=self.device,
            )
        logger.info(
            "MLA host-dedup broadcast chunk configured: base_tokens=%d, "
            "effective_layer_tokens=%d",
            self.chunk_tokens,
            self.layer_num * self.chunk_tokens,
        )

    @classmethod
    def build(
        cls,
        device_pool,
        tp_group: torch.distributed.ProcessGroup,
        attn_tp_group: Optional[torch.distributed.ProcessGroup],
    ) -> MLAHostDedupBroadcaster:
        """Build and initialize the NCCL group before host-pool allocation."""
        from decoupled_engine.srt.distributed.parallel_state import create_custom_parallel_group

        base_group = tp_group
        if is_dp_attention_enabled() and attn_tp_group is not None:
            base_group = attn_tp_group
        group_ranks = torch.distributed.get_process_group_ranks(base_group)
        group = create_custom_parallel_group(
            group_ranks=list(group_ranks), backend="nccl"
        )
        broadcaster = cls(device_pool, group, src_global_rank=group_ranks[0])
        broadcaster._warmup_group()
        return broadcaster

    def _warmup_group(self) -> None:
        """Initialize the NCCL communicator before serving."""
        warmup = self.kv_staging[:1]
        if self.is_src:
            warmup.zero_()
        torch.distributed.broadcast(warmup, src=self.src_global_rank, group=self.group)
        torch.cuda.synchronize(self.device)
        logger.info("MLA host-dedup NCCL broadcast group warmup completed")

    def prepare_broadcast(
        self, device_indices: torch.Tensor, load_stream
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare reusable KV/indexer indices for one layerwise load."""
        indices = device_indices
        if not indices.is_cuda:
            indices = indices.to(self.device)
        if indices.is_cuda:
            indices.record_stream(load_stream)

        page_idx = None
        if self.idx_bufs is not None:
            page_size = self.device_pool.page_size
            if page_size > 1:
                if indices.numel() % page_size != 0:
                    raise ValueError(
                        "DSA dedup broadcast expects page-aligned device indices: "
                        f"got {indices.numel()} indices for page_size={page_size}."
                    )
                # Preserve logical page order across rank-local allocations.
                page_idx = indices[::page_size] // page_size
            else:
                page_idx = indices
            if page_idx.is_cuda:
                page_idx.record_stream(load_stream)
        return indices, page_idx

    def broadcast_loaded_layer(
        self,
        layer_id: int,
        prepared: tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> None:
        """Broadcast one loaded KV layer and its optional DSA indexer layer."""
        indices, page_idx = prepared
        self._bcast_layer(
            self.device_pool.kv_buffer,
            self.kv_staging,
            indices,
            self.device_pool.kv_cache_dim,
            layer_id,
        )
        if self.idx_bufs is not None:
            assert page_idx is not None
            self._bcast_layer(
                self.idx_bufs,
                self.idx_staging,
                page_idx,
                self.idx_elem,
                layer_id,
            )

    def _bcast_layer(
        self,
        buf_list,
        staging,
        target,
        elem,
        layer_id: int,
    ) -> None:
        """Broadcast one layer in chunks using the shared staging buffer."""
        n = target.shape[0]
        rows_per_chunk = staging.numel() // elem
        assert rows_per_chunk > 0
        layer_buf = buf_list[layer_id]
        row_shape = layer_buf.shape[1:]

        for start in range(0, n, rows_per_chunk):
            cur = min(rows_per_chunk, n - start)
            idx = target[start : start + cur]
            chunk = staging[: cur * elem]
            chunk_rows = chunk.view(cur, *row_shape)
            if self.is_src:
                torch.index_select(layer_buf, 0, idx, out=chunk_rows)
            torch.distributed.broadcast(
                chunk, src=self.src_global_rank, group=self.group
            )
            if not self.is_src:
                layer_buf.index_copy_(0, idx, chunk_rows)

    def destroy(self) -> None:
        if self.group is None:
            return
        try:
            torch.distributed.destroy_process_group(self.group)
        except Exception:
            pass
        self.group = None


@dataclass
class MLAHostDedupContext:
    """All state owned by the optional MLA host-dedup path."""

    broadcaster: MLAHostDedupBroadcaster
    prefetch_hits_sync_groups: Optional[List[torch.distributed.ProcessGroup]]
    prefetch_completion_sync_groups: Optional[List[torch.distributed.ProcessGroup]]
    producer_stream: Optional[object] = None
    last_write_finish_event: Optional[object] = None

    @property
    def is_src(self) -> bool:
        return self.broadcaster.is_src

    @property
    def is_dummy_rank(self) -> bool:
        return not self.is_src

    def destroy(self) -> None:
        self.broadcaster.destroy()
        groups = (self.prefetch_hits_sync_groups or []) + (
            self.prefetch_completion_sync_groups or []
        )
        for group in groups:
            try:
                torch.distributed.destroy_process_group(group)
            except Exception:
                pass
        self.prefetch_hits_sync_groups = None
        self.prefetch_completion_sync_groups = None


def maybe_create_mla_host_dedup_context(
    kv_cache,
    tp_group: torch.distributed.ProcessGroup,
    attn_cp_group: Optional[torch.distributed.ProcessGroup],
    attn_tp_group: Optional[torch.distributed.ProcessGroup],
    storage_backend: Optional[str],
    enabled: bool = False,
) -> Optional[MLAHostDedupContext]:
    """Create dedup state before host allocation, or preserve the original path."""
    if not enabled:
        return None
    if not mla_host_dedup_eligible(kv_cache, storage_backend):
        return None
    if mla_dedup_rank_and_size()[1] <= 1:
        return None

    broadcaster = MLAHostDedupBroadcaster.build(kv_cache, tp_group, attn_tp_group)
    prefetch_hits_sync_groups = None
    prefetch_completion_sync_groups = None
    if storage_backend is not None:
        prefetch_hits_sync_groups = _prebuild_prefetch_sync_groups(
            tp_group, attn_cp_group, attn_tp_group
        )
        prefetch_completion_sync_groups = _prebuild_prefetch_sync_groups(
            tp_group, attn_cp_group, attn_tp_group
        )
    return MLAHostDedupContext(
        broadcaster,
        prefetch_hits_sync_groups,
        prefetch_completion_sync_groups,
    )


def _prebuild_prefetch_sync_groups(
    tp_group: torch.distributed.ProcessGroup,
    attn_cp_group: Optional[torch.distributed.ProcessGroup],
    attn_tp_group: Optional[torch.distributed.ProcessGroup],
) -> List[torch.distributed.ProcessGroup]:
    """Prebuild one set of HiCache storage synchronization groups."""
    from decoupled_engine.srt.distributed.parallel_state import create_custom_parallel_group

    groups: List[torch.distributed.ProcessGroup] = []
    seen_rank_sets = set()
    if attn_cp_group is not None or attn_tp_group is not None:
        base_groups = [attn_cp_group, attn_tp_group]
    else:
        base_groups = [tp_group]
    for group in base_groups:
        if group is None or torch.distributed.get_world_size(group=group) == 1:
            continue
        ranks = tuple(torch.distributed.get_process_group_ranks(group))
        if ranks in seen_rank_sets:
            continue
        seen_rank_sets.add(ranks)
        groups.append(
            create_custom_parallel_group(group_ranks=list(ranks), backend="gloo")
        )
    return groups


# ---- EI: MLA host dedup on the DSA/DSV4 host-dedup machinery (nccl / ce_peer) ----

MLA_FANOUTS = ("nccl", "ce_peer")


class MLAHostDedup(DSAHostDedup):
    """Reuse the MLA row and packed-draft mapping with a dedicated NCCL group."""

    def __init__(
        self,
        group,
        src_global_rank,
        device,
        fanout: str = "nccl",
        kv_pool=None,
        draft_pools=(),
        sidecar_pools=(),
        page_size=None,
    ):
        # Read by ce_loader.init_layerwise_loadback instead of the DSA word.
        self.fanout = fanout
        if fanout == "nccl":
            # Use the existing collective path; DSA's optional symmetric-memory
            # transport is a separate setting and does not apply to this MLA path.
            DeepSeekV4HostDedup.__init__(self, group, src_global_rank, device)
            self._symm_handle = None
            return
        if fanout != "ce_peer":
            raise ValueError(f"MLA host dedup fan-out must be one of {MLA_FANOUTS}")
        # C2g: DSAHostDedup builds CePeer over layer_buffers(), which reads the
        # sidecar pools from this attribute (MLA-only, set before the init).
        self._sidecar_pools = tuple(sidecar_pools)
        DSAHostDedup.__init__(
            self,
            group,
            src_global_rank,
            device,
            fanout=fanout,
            kv_pool=kv_pool,
            draft_pools=draft_pools,
            page_size=page_size,
        )


def maybe_build_mla_host_dedup(params, kvcache, storage_backend, extra_config):
    from decoupled_engine.srt.runtime_context import get_memory, get_parallel

    if not get_memory().enable_mla_hicache_host_dedup:
        return None
    from decoupled_engine.srt.mem_cache.memory_pool import DSATokenToKVPool, MLATokenToKVPool

    if (
        not isinstance(kvcache, MLATokenToKVPool)
        or isinstance(kvcache, DSATokenToKVPool)
        or getattr(kvcache, "layer_shard_enabled", False)
        or get_parallel().attn_dcp_size != 1
    ):
        raise ValueError("MLA host dedup requires a replicated MLA device pool")
    from decoupled_engine.srt.environ import envs

    fanout = str(envs.SGLANG_MLA_HOST_DEDUP_FANOUT.get()).lower()
    if fanout == "nccl":
        return build_host_dedup(
            params, kvcache, storage_backend, extra_config, MLAHostDedup
        )
    if fanout != "ce_peer":
        raise ValueError(
            "--enable-mla-hicache-host-dedup supports SGLANG_MLA_HOST_DEDUP_FANOUT "
            f"nccl or ce_peer, got {fanout!r}"
        )
    if envs.SGLANG_DSA_HOST_DEDUP_SYMMETRIC_BROADCAST.get():
        raise ValueError(
            "SGLANG_MLA_HOST_DEDUP_FANOUT=ce_peer has no broadcast to replace"
        )
    if envs.SGLANG_DSA_HOST_DEDUP_CE_LOAD.get():
        raise ValueError("SGLANG_DSA_HOST_DEDUP_CE_LOAD requires the IPC fan-out")
    from decoupled_engine.srt.arg_groups.symmetric_defaults import apply_gb300_dedup_tuning

    apply_gb300_dedup_tuning("SGLANG_MLA_HOST_DEDUP_FANOUT")
    return build_host_dedup(
        params,
        kvcache,
        storage_backend,
        extra_config,
        MLAHostDedup,
        backend="gloo",
        io_layout=("direct", "page_first_direct"),
        fanout="ce_peer",
        kv_pool=kvcache,
        draft_pools=tuple(getattr(params, "mtp_draft_device_pools", ()) or ()),
        sidecar_pools=_sidecar_draft_pools(params),
        page_size=params.page_size,
    )


def _sidecar_draft_pools(params) -> tuple:
    """Separate flat-MLA HiCache draft pools (EAGLE3) with at least one layer."""
    from decoupled_engine.srt.mem_cache.memory_pool import DSATokenToKVPool, MLATokenToKVPool

    pools = tuple(getattr(params, "hicache_sidecar_draft_device_pools", ()) or ())
    for pool in pools:
        if not isinstance(pool, MLATokenToKVPool) or isinstance(
            pool, DSATokenToKVPool
        ):
            raise ValueError(
                "MLA host dedup C2g supports only flat MLA sidecar draft pools, "
                f"got {type(pool).__name__}"
            )
    return tuple(pool for pool in pools if pool.layer_num > 0)
