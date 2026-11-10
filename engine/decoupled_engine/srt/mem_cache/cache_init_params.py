# Adapted from SGLang python/sglang/srt/mem_cache/cache_init_params.py (commit 9a489f8d2f), modified for evolving-infra.
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from decoupled_engine.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from decoupled_engine.srt.mem_cache.memory_pool import ReqToTokenPool
    from decoupled_engine.srt.mem_cache.unified_cache.components import ComponentType
    from decoupled_engine.srt.mem_cache.unified_cache.components.tree_component import (
        TreeComponent,
    )


@dataclasses.dataclass
class CacheInitParams:
    disable: bool
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    page_size: int

    is_eagle: bool = False
    tp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    attn_cp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    attn_tp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    pp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    eviction_policy: str = "lru"
    disable_finished_insert: bool = False

    enable_metrics: bool = False
    enable_kv_cache_events: bool = False
    enable_session_radix_cache: bool = False
    enable_kv_session_retention: bool = False

    enable_mamba_extra_buffer: bool = False
    enable_mamba_extra_buffer_lazy: bool = False

    pp_rank: int = 0
    pp_size: int = 1

    attn_cp_rank: int = 0
    attn_cp_size: int = 1

    chunked_prefill_size: Optional[int] = None

    sliding_window_size: Optional[int] = None

    # Time-to-live for cache entries in seconds. If None, TTL is disabled.
    cache_ttl_seconds: Optional[float] = None

    tree_components: Optional[tuple[ComponentType, ...]] = None
    component_registry_override: Optional[dict[ComponentType, type[TreeComponent]]] = (
        None
    )

    mtp_draft_device_pools: tuple[object, ...] = ()
    # Separate (SIDECAR) HiCache draft device pools, e.g. the EAGLE3 draft of
    # Kimi-K2.6. Host dedup's copy-engine fan-out (C2g) maps them at init,
    # before the sidecar host pool is registered with the tree.
    hicache_sidecar_draft_device_pools: tuple[object, ...] = ()
