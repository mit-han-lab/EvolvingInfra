# Adapted from SGLang python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py (commit 297e54fd10), modified for evolving-infra.
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional

from decoupled_engine.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    SidecarPoolSpec,
)
from decoupled_engine.srt.mem_cache.hybrid_cache.host_pool_config import (
    HostPoolGroupConfig,
    check_packed_kv_rows,
    is_mla_pool,
    prepare_host_pool_config,
    with_packed_draft_layer_mapping,
)
from decoupled_engine.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from decoupled_engine.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
    DSV4TailLayout,
    LogicalHostPool,
)
from decoupled_engine.srt.mem_cache.page_interleave_pool import PageInterleaveKVPoolMixin
from decoupled_engine.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from decoupled_engine.srt.mem_cache.pool_host.common import get_allocator_type
from decoupled_engine.srt.mem_cache.pool_host.host_pool_decl import HostPoolDecl
from decoupled_engine.srt.mem_cache.pool_host.mamba import MambaPoolHost
from decoupled_engine.srt.mem_cache.pool_host.mha import (
    MHATokenToKOnlyPoolHost,
    get_mha_host_pool_cls,
)
from decoupled_engine.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from decoupled_engine.srt.mem_cache.pool_host.page_interleave import (
    PageInterleaveHostPool,
)
from decoupled_engine.srt.mem_cache.pool_host.qsa import QSAIndexerPoolHost
from decoupled_engine.srt.mem_cache.pool_host.unified import UnifiedPageEnvelopeHostPool
from decoupled_engine.srt.mem_cache.unified_cache.component_type import ComponentType
from decoupled_engine.srt.runtime_context import get_memory, get_parallel, get_serving
from decoupled_engine.srt.utils import is_npu

if TYPE_CHECKING:
    import torch

    from decoupled_engine.srt.mem_cache.cache_init_params import CacheInitParams
    from decoupled_engine.srt.mem_cache.device_pool_info import DevicePoolInfo
    from decoupled_engine.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from decoupled_engine.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from decoupled_engine.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _resolved(server_args, field):
    """The resolved value of ``field`` (declarations included), not the raw input."""
    from decoupled_engine.srt.arg_groups.overrides import resolution_result

    return resolution_result(server_args, field)


def _get_allocator_type() -> str:
    return get_allocator_type()


def _evict_swa_for_device_alloc(cache: UnifiedRadixCache, required_size: int) -> None:
    from decoupled_engine.srt.mem_cache.base_prefix_cache import EvictParams

    available_size = cache.token_to_kv_pool_allocator.swa_available_size()
    shortfall = max(0, required_size - available_size)
    if shortfall > 0:
        cache.evict_for_alloc(EvictParams(swa_num_tokens=shortfall))


def _evict_mamba_for_device_alloc(cache: UnifiedRadixCache, required_size: int) -> None:
    from decoupled_engine.srt.mem_cache.base_prefix_cache import EvictParams

    available_size = (
        cache.req_to_token_pool.mamba_allocator.schedulable_available_size()
    )
    shortfall = max(0, required_size - available_size)
    if shortfall > 0:
        cache.evict_for_alloc(EvictParams(mamba_num=shortfall))


def _make_layer_mapper(
    layer_mapping: dict[int, int],
    transfer_layer_id_max: int,
) -> Callable[[int], Optional[int]]:
    # The exclusive transfer-ID bound includes holes for uncached layers;
    # each pool skips IDs absent from its mapping.
    def mapper(layer_id: int) -> Optional[int]:
        if not 0 <= layer_id < transfer_layer_id_max:
            return None
        return layer_mapping.get(layer_id)

    return mapper


def _stage_local_layer_mapping(
    layer_mapping: dict[int, int], start_layer: int
) -> dict[int, int]:
    return {
        global_layer - start_layer: pool_layer
        for global_layer, pool_layer in layer_mapping.items()
    }


class _DeepSeekV4LayerMappings(NamedTuple):
    transfer_layer_id_max: int
    full: dict[int, int]
    swa: dict[int, int]
    c4: dict[int, int]
    c128: dict[int, int]
    c4_state: dict[int, int]
    c4_state_global_layers: list[int]


def _resolve_deepseek_v4_layer_mappings(
    kvcache: Any,
) -> _DeepSeekV4LayerMappings:
    transfer_layer_id_max = kvcache.end_layer - kvcache.start_layer
    full = {layer: layer for layer in range(transfer_layer_id_max)}
    swa = full.copy() if kvcache.swa_kv_pool is not None else {}

    c4, c128, c4_state_global_layers = {}, {}, []
    for local_layer, item in enumerate(
        kvcache.layer_mapping[kvcache.start_layer : kvcache.end_layer]
    ):
        if item.compress_ratio == 4:
            c4[local_layer] = item.compress_layer_id
            c4_state_global_layers.append(kvcache.start_layer + local_layer)
        elif item.compress_ratio == 128:
            c128[local_layer] = item.compress_layer_id

    return _DeepSeekV4LayerMappings(
        transfer_layer_id_max=transfer_layer_id_max,
        full=full,
        swa=swa,
        c4=c4,
        c128=c128,
        c4_state={layer: index for index, layer in enumerate(c4)},
        c4_state_global_layers=c4_state_global_layers,
    )


def _mha_host_inputs(pool, drafts=()):
    info = pool.get_device_pool_infos()[0].buffer_info
    return (
        get_mha_host_pool_cls(info),
        info,
        dict(
            device_capacity_tokens=(
                pool.size
                if pool.host_capacity_tokens is None
                else pool.host_capacity_tokens
            ),
            start_layer=pool.start_layer,
            end_layer=pool.end_layer,
            draft_buffer_infos=tuple(
                draft.get_device_pool_infos()[0].buffer_info for draft in drafts
            ),
            write_back_staging=pool.hicache_write_back_staging,
        ),
    )


def _attach_host_dedup_device_pools(host_pool, kv_pool, draft_pools=()) -> None:
    """EI host dedup (dsa_host_dedup.py, ce_2d_load.py, staged_ce_load.py)
    reads the device pools through the host pool (device_pool,
    mtp_draft_device_pools, _host_layer_index); upstream's MLA host pool
    keeps only their buffer infos."""
    host_pool.device_pool = kv_pool
    host_pool.mtp_draft_device_pools = tuple(draft_pools)


def build_kv_host_pool(
    *,
    kv_pool: Any,
    page_size: int,
    use_mla: Optional[bool] = None,
    override_kv_cache_dim: Optional[int] = None,
    host_size: Optional[float] = None,
    mtp_draft_device_pools: tuple[Any, ...] = (),
    pool_label: str = "kv",
    host_dedup=None,
):
    if use_mla is None:
        use_mla = is_mla_pool(kv_pool)
    if use_mla:
        kv_host_pool_cls = MLATokenToKVPoolHost
        host_input = kv_pool.get_device_pool_infos()[0].buffer_info
        mha_kwargs = {}
    else:
        kv_host_pool_cls, host_input, mha_kwargs = _mha_host_inputs(
            kv_pool, mtp_draft_device_pools
        )
    if use_mla and override_kv_cache_dim is None:
        # An fp8 DSA store is wider than kv_lora_rank + qk_rope_head_dim. The
        # device pool already resolved the row width.
        override_kv_cache_dim = kv_pool.kv_cache_dim
    if use_mla or is_npu():
        check_packed_kv_rows(
            kv_pool=kv_pool, drafts=mtp_draft_device_pools, use_mla=use_mla
        )
    kwargs = {}
    if host_dedup is not None:
        kwargs["host_dedup"] = host_dedup
    if override_kv_cache_dim is not None:
        kwargs["override_kv_cache_dim"] = override_kv_cache_dim
    if use_mla:
        kwargs.update(
            device_capacity_tokens=kv_pool.size
            if kv_pool.host_capacity_tokens is None
            else kv_pool.host_capacity_tokens,
            start_layer=kv_pool.start_layer,
            end_layer=kv_pool.end_layer,
            draft_buffer_infos=tuple(
                pool.get_device_pool_infos()[0].buffer_info
                for pool in mtp_draft_device_pools
            ),
            layer_shard_enabled=kv_pool.layer_shard_enabled,
            target_layer_num=(
                (kv_pool.layer_num + kv_pool.layer_shard_size - 1)
                // kv_pool.layer_shard_size
                if kv_pool.layer_shard_enabled
                else kv_pool.layer_num
            ),
        )
    else:
        kwargs.update(mha_kwargs)
    parallel = get_parallel()
    if parallel.dcp_enabled:
        assert use_mla, (
            "HiCache + DCP is only wired for the MLA host pool; the MHA host "
            "pool has no DCP index translation."
        )
        kwargs["dcp_size"] = parallel.attn_dcp_size
        kwargs["dcp_rank"] = parallel.attn_dcp_rank
    host_pool = kv_host_pool_cls(
        host_input,
        get_memory().hicache_ratio,
        get_memory().hicache_size if host_size is None else host_size,
        page_size,
        get_memory().hicache_mem_layout,
        allocator_type=_get_allocator_type(),
        pool_label=pool_label,
        **kwargs,
    )
    if host_dedup is not None:
        _attach_host_dedup_device_pools(host_pool, kv_pool, mtp_draft_device_pools)
    if isinstance(kv_pool, PageInterleaveKVPoolMixin):
        return PageInterleaveHostPool(host_pool, kv_pool.placement)
    return host_pool


def _split_hicache_size(
    hicache_size: int, kv_pools: tuple[Any, ...]
) -> tuple[float, ...]:
    device_pool_sizes = []
    for kv_pool in kv_pools:
        size_bytes = getattr(kv_pool, "host_capacity_bytes", None)
        if size_bytes is None:
            size_bytes = kv_pool.get_kv_size_bytes()
        device_pool_sizes.append(
            sum(size_bytes) if isinstance(size_bytes, tuple) else size_bytes
        )
    total_device_pool_size = sum(device_pool_sizes)
    return tuple(
        hicache_size * size_bytes / total_device_pool_size
        for size_bytes in device_pool_sizes
    )


def build_pool_entry(
    *,
    name: PoolName,
    host_pool: Any,
    device_pool: Any,
    layer_mapping: dict[int, int],
    transfer_layer_id_max: int,
    is_anchor: bool = False,
    host_evict_fn: Optional[Callable[[int], Any]] = None,
    device_evict_fn: Optional[Callable[[int], Any]] = None,
    device_alloc_fn: Optional[Callable[[int], Any]] = None,
    device_free_fn: Optional[Callable[[Any], Any]] = None,
    packed_draft_device_pools: tuple[Any, ...] = (),
    indices_from_pool: Optional[PoolName] = None,
) -> PoolEntry:
    return PoolEntry(
        name=name,
        host_pool=host_pool,
        device_pool=device_pool,
        layer_mapper=_make_layer_mapper(layer_mapping, transfer_layer_id_max),
        is_primary_index_anchor=is_anchor,
        host_evict_fn=host_evict_fn,
        device_evict_fn=device_evict_fn,
        device_alloc_fn=device_alloc_fn,
        device_free_fn=device_free_fn,
        packed_draft_device_pools=packed_draft_device_pools,
        indices_from_pool=indices_from_pool,
    )


def _peer_kv_reserve_pages(num_pages: int, which: str) -> int:
    """kv-p2p pin reserve for an allocator-owning host pool (0 when the
    PeerKVServicer is disabled). Derived from rank-uniform inputs only."""
    from decoupled_engine.srt.mem_cache.pool_host.slot_allocator import reserve_pages_for
    from decoupled_engine.srt.runtime_context import get_server_args

    try:
        # get_global_server_args() is retired upstream (it always raises).
        server_args = get_server_args()
    except Exception:
        return 0
    if server_args is None or not _resolved(server_args, "enable_peer_kv_servicer"):
        return 0
    from decoupled_engine.srt.arg_groups.overrides import resolution_result

    # The resolved value: a model override (window residency raises the SWA
    # default) lives in the declarations, not in the field.
    ratio = resolution_result(
        server_args, f"peer_kv_servicer_reserve_ratio_{which}", 0.0
    )
    return reserve_pages_for(num_pages, float(ratio))


def build_kv_only_group(
    *,
    page_size: int,
    kv_pool: Any,
    full_layer_mapping: dict[int, int],
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
    host_size: Optional[float] = None,
    mtp_draft_device_pools: tuple[Any, ...] = (),
    host_dedup=None,
) -> HostPoolGroup:
    """Anchor-only host pool group for a flat MHA/MLA device pool."""
    transfer_layer_id_max = len(full_layer_mapping)
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=page_size,
        use_mla=use_mla,
        override_kv_cache_dim=override_kv_cache_dim,
        host_size=host_size,
        mtp_draft_device_pools=mtp_draft_device_pools,
        host_dedup=host_dedup,
    )
    if get_memory().enable_peer_kv_servicer and kv_host_pool.slot_allocator is None:
        kv_host_pool.enable_peer_serving(
            reserve_pages=_peer_kv_reserve_pages(kv_host_pool.page_num, "kv")
        )
    if mtp_draft_device_pools:
        full_layer_mapping = with_packed_draft_layer_mapping(
            full_layer_mapping,
            transfer_layer_start=transfer_layer_id_max,
            target_device_layer_num=kv_pool.layer_num,
            draft_layer_num=len(mtp_draft_device_pools),
        )
    host_pool_group = HostPoolGroup(
        [
            build_pool_entry(
                name=PoolName.KV,
                host_pool=kv_host_pool,
                device_pool=kv_pool,
                layer_mapping=full_layer_mapping,
                transfer_layer_id_max=transfer_layer_id_max
                + len(mtp_draft_device_pools),
                is_anchor=True,
                packed_draft_device_pools=mtp_draft_device_pools,
            )
        ]
    )
    host_pool_group.host_dedup = host_dedup
    return host_pool_group


def _swa_allocation_callbacks(allocator) -> dict:
    """Keep allocation and rollback in the same ID space for every SWA stack."""
    from decoupled_engine.srt.mem_cache.allocator.unified_sub_pool import MultiEndedAllocator

    if isinstance(allocator, MultiEndedAllocator):
        # Unified SWA loads into physical reservations, bound before the H2D.
        return dict(
            device_alloc_fn=allocator.alloc_physical,
            device_free_fn=allocator.cancel_physical_reservation,
        )
    if allocator is None:
        return {}
    return dict(device_alloc_fn=allocator.alloc, device_free_fn=allocator.free)


def _uses_unified_page_envelope_host(
    full_kv_pool: Any,
    swa_kv_pool: Any,
    use_mla: bool,
    mtp_device_pools: tuple[Any, ...] = (),
) -> bool:
    from decoupled_engine.srt.mem_cache.unified_memory_pool import UnifiedMHATokenToKVPool

    return (
        not use_mla
        and not mtp_device_pools
        and all(
            isinstance(pool, UnifiedMHATokenToKVPool)
            for pool in (full_kv_pool, swa_kv_pool)
        )
        and {full_kv_pool.grow_direction, swa_kv_pool.grow_direction} == {"up", "down"}
        and HybridCacheController.supports_page_envelope_host(
            get_memory().hicache_storage_backend
        )
    )


def build_hybrid_swa_group(
    *,
    page_size: int,
    full_kv_pool: Any,
    swa_kv_pool: Any,
    full_layer_mapping: dict[int, int],
    swa_layer_mapping: dict[int, int],
    use_mla: bool,
    kv_host_size: Optional[float] = None,
    swa_host_size: Optional[float] = None,
    host_full_evict_fn: Optional[Callable[[int], Any]] = None,
    host_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    device_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    swa_attn_allocator: Any = None,
    mtp_full_device_pools: tuple[Any, ...] = (),
    mtp_swa_device_pools: tuple[Any, ...] = (),
    mtp_swa_full_device_pools: tuple[Any, ...] = (),
    transfer_layer_id_max: Optional[int] = None,
) -> HostPoolGroup:
    """Anchor (full) + SWA host pool group for a hybrid-SWA device pool."""
    if transfer_layer_id_max is None:
        transfer_layer_id_max = (
            max(full_layer_mapping.keys() | swa_layer_mapping.keys()) + 1
        )
    if mtp_swa_full_device_pools and kv_host_size:
        # The sidecar has the anchor's token capacity. Reserve its bytes from
        # the full-KV budget rather than exceeding an explicit --hicache-size.
        anchor_pools = (full_kv_pool, *mtp_full_device_pools)
        shares = _split_hicache_size(
            kv_host_size, (*anchor_pools, *mtp_swa_full_device_pools)
        )
        kv_host_size = sum(shares[: len(anchor_pools)])
    if _uses_unified_page_envelope_host(
        full_kv_pool,
        swa_kv_pool,
        use_mla,
        mtp_full_device_pools + mtp_swa_device_pools + mtp_swa_full_device_pools,
    ):
        memory = get_memory()
        kv_host_pool, swa_host_pool = (
            UnifiedPageEnvelopeHostPool.build_hybrid_swa_pool_pair(
                device_pools=(full_kv_pool, swa_kv_pool),
                host_to_device_ratio=memory.hicache_ratio,
                host_size=memory.hicache_size,
                page_size=page_size,
                layout=memory.hicache_mem_layout,
                allocator_type=_get_allocator_type(),
                mtp_draft_device_pools=mtp_swa_device_pools,
            )
        )
        has_shared_arena = True
    else:
        has_shared_arena = False
        kv_host_pool = build_kv_host_pool(
            kv_pool=full_kv_pool,
            page_size=page_size,
            use_mla=use_mla,
            host_size=kv_host_size,
            mtp_draft_device_pools=mtp_full_device_pools,
            pool_label="full",
        )
        swa_host_pool = build_kv_host_pool(
            kv_pool=swa_kv_pool,
            page_size=page_size,
            use_mla=use_mla,
            host_size=swa_host_size,
            mtp_draft_device_pools=mtp_swa_device_pools,
            pool_label="swa",
        )
    if mtp_full_device_pools:
        full_layer_mapping = with_packed_draft_layer_mapping(
            full_layer_mapping,
            transfer_layer_start=transfer_layer_id_max,
            target_device_layer_num=full_kv_pool.layer_num,
            draft_layer_num=len(mtp_full_device_pools),
        )
    if mtp_swa_device_pools:
        swa_layer_mapping = with_packed_draft_layer_mapping(
            swa_layer_mapping,
            transfer_layer_start=transfer_layer_id_max,
            target_device_layer_num=swa_kv_pool.layer_num,
            draft_layer_num=len(mtp_swa_device_pools),
        )
    group = HostPoolGroup(
        [
            build_pool_entry(
                name=PoolName.KV,
                host_pool=kv_host_pool,
                device_pool=full_kv_pool,
                layer_mapping=full_layer_mapping,
                transfer_layer_id_max=transfer_layer_id_max
                + len(mtp_full_device_pools),
                is_anchor=True,
                host_evict_fn=host_full_evict_fn if has_shared_arena else None,
                packed_draft_device_pools=mtp_full_device_pools,
            ),
            build_pool_entry(
                name=PoolName.SWA,
                host_pool=swa_host_pool,
                device_pool=swa_kv_pool,
                layer_mapping=swa_layer_mapping,
                transfer_layer_id_max=transfer_layer_id_max + len(mtp_swa_device_pools),
                host_evict_fn=host_swa_evict_fn,
                device_evict_fn=device_swa_evict_fn,
                **_swa_allocation_callbacks(swa_attn_allocator),
                packed_draft_device_pools=mtp_swa_device_pools,
            ),
        ]
    )
    if mtp_swa_full_device_pools:
        # SWA drafts with full-token addressing may have different head geometry
        # from full attention. Use a separate host pool with KV indices instead of
        # packing their bytes into either target pool.
        first, *rest = mtp_swa_full_device_pools
        host_cls, host_input, host_kwargs = _mha_host_inputs(first, tuple(rest))
        draft_host_pool = host_cls(
            host_input,
            host_to_device_ratio=kv_host_pool.size / first.size,
            host_size=0,
            page_size=page_size,
            layout=get_memory().hicache_mem_layout,
            allocator_type=_get_allocator_type(),
            **host_kwargs,
            pool_label="draft_swa",
        )
        group.add_entry(
            build_pool_entry(
                name=PoolName.DRAFT_SWA,
                host_pool=draft_host_pool,
                device_pool=first,
                layer_mapping=with_packed_draft_layer_mapping(
                    {0: 0},
                    transfer_layer_start=transfer_layer_id_max,
                    target_device_layer_num=first.layer_num,
                    draft_layer_num=len(rest),
                ),
                transfer_layer_id_max=transfer_layer_id_max + len(rest),
                packed_draft_device_pools=tuple(rest),
            )
        )
    return group


def _mtp_swa_device_pools(
    params: CacheInitParams, *, full_indices: bool
) -> tuple[Any, ...]:
    return tuple(
        pool.swa_kv_pool
        for pool in params.mtp_draft_device_pools
        if pool.swa_kv_pool.layer_num and pool.swa_uses_full_indices == full_indices
    )


def _mtp_swa_sidecar_specs(group: HostPoolGroup) -> list[SidecarPoolSpec]:
    if PoolName.DRAFT_SWA not in group.entry_map:
        return []
    return [SidecarPoolSpec(PoolName.DRAFT_SWA, indices_from_pool=PoolName.KV)]


def build_kv_only_stack(
    *,
    params: CacheInitParams,
    kv_pool: Any,
    full_layer_mapping: dict[int, int],
    load_cache_event,
    storage_backend: Optional[str],
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    from decoupled_engine.srt.mem_cache.mla_host_dedup import maybe_build_mla_host_dedup

    host_dedup = maybe_build_mla_host_dedup(
        params, kv_pool, storage_backend, storage_backend_extra_config
    )
    transfer_layer_id_max = len(full_layer_mapping)
    host_pool_group = build_kv_only_group(
        page_size=params.page_size,
        kv_pool=kv_pool,
        full_layer_mapping=full_layer_mapping,
        use_mla=use_mla,
        override_kv_cache_dim=override_kv_cache_dim,
        mtp_draft_device_pools=params.mtp_draft_device_pools,
        host_dedup=host_dedup,
    )
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        params.page_size,
        params.tp_cache_group,
        load_cache_event=load_cache_event,
        attn_cp_group=params.attn_cp_cache_group,
        attn_tp_group=params.attn_tp_cache_group,
        pp_group=params.pp_cache_group,
        write_policy=get_memory().hicache_write_policy,
        io_backend=get_memory().hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_id_max=transfer_layer_id_max,
        enable_storage_metrics=enable_storage_metrics,
        host_memory_mode=get_memory().hicache_host_memory_mode,
    )
    return host_pool_group, cache_controller


def build_hybrid_swa_stack(
    *,
    params: CacheInitParams,
    full_kv_pool: Any,
    swa_kv_pool: Any,
    full_layer_mapping: dict[int, int],
    swa_layer_mapping: dict[int, int],
    load_cache_event,
    storage_backend: Optional[str],
    use_mla: bool,
    host_full_evict_fn: Optional[Callable[[int], Any]] = None,
    host_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    device_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_id_max = (
        max(full_layer_mapping.keys() | swa_layer_mapping.keys()) + 1
    )
    # Each draft depth may use full or sliding-window attention.
    mtp_full_device_pools = tuple(
        pool.full_kv_pool
        for pool in params.mtp_draft_device_pools
        if pool.full_kv_pool.layer_num
    )
    mtp_swa_device_pools = _mtp_swa_device_pools(params, full_indices=False)

    kv_host_size = swa_host_size = None
    memory = get_memory()
    if memory.hicache_size > 0 and not _uses_unified_page_envelope_host(
        full_kv_pool, swa_kv_pool, use_mla, params.mtp_draft_device_pools
    ):
        kv_host_size, swa_host_size = _split_hicache_size(
            memory.hicache_size, (full_kv_pool, swa_kv_pool)
        )

    host_pool_group = build_hybrid_swa_group(
        page_size=params.page_size,
        full_kv_pool=full_kv_pool,
        swa_kv_pool=swa_kv_pool,
        full_layer_mapping=full_layer_mapping,
        swa_layer_mapping=swa_layer_mapping,
        use_mla=use_mla,
        kv_host_size=kv_host_size,
        swa_host_size=swa_host_size,
        host_full_evict_fn=host_full_evict_fn,
        host_swa_evict_fn=host_swa_evict_fn,
        device_swa_evict_fn=device_swa_evict_fn,
        # For SWA hybrid, device allocation goes through the inner allocator.
        swa_attn_allocator=params.token_to_kv_pool_allocator.swa_attn_allocator,
        mtp_full_device_pools=mtp_full_device_pools,
        mtp_swa_device_pools=mtp_swa_device_pools,
        mtp_swa_full_device_pools=_mtp_swa_device_pools(params, full_indices=True),
    )
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        params.page_size,
        params.tp_cache_group,
        load_cache_event=load_cache_event,
        attn_cp_group=params.attn_cp_cache_group,
        attn_tp_group=params.attn_tp_cache_group,
        pp_group=params.pp_cache_group,
        write_policy=get_memory().hicache_write_policy,
        io_backend=get_memory().hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_id_max=transfer_layer_id_max,
        enable_storage_metrics=enable_storage_metrics,
        host_memory_mode=get_memory().hicache_host_memory_mode,
    )
    return host_pool_group, cache_controller


def _deepseek_v4_num_host_pages(
    *,
    params: CacheInitParams,
    kvcache: Any,
    page_size: int,
    swa_page_size: int,
    bytes_per_page: Optional[int] = None,
) -> tuple[int, int, int]:
    """(KV, SWA, C128) host pages. Window residency: from
    ``_dsv4_window_host_tokens`` (``bytes_per_page``: every host pool's bytes
    per ``page_size`` tokens in the active layout, needed with
    --dsv4-hicache-host-gb)."""
    allocator = params.token_to_kv_pool_allocator
    device_full_size = getattr(allocator, "size_full", kvcache.size)
    device_full_pages = (device_full_size + page_size - 1) // page_size

    device_swa_pages = (kvcache.swa_size + swa_page_size - 1) // swa_page_size

    if get_memory().hicache_size > 0:
        raise ValueError(
            "DeepSeek V4 HiCache currently does not support --hicache-size; "
            "use --hicache-ratio instead (window residency: "
            "--dsv4-hicache-host-gb or --dsv4-hicache-host-tokens)."
        )
    if _dsv4_window_residency():
        # The host owns SWA for every cached page, so the KV anchor and SWA
        # pools hold the same absolute token count. The device SWA pool is a
        # window-sized fraction of FULL and says nothing about host needs.
        host_tokens, source = _dsv4_window_host_tokens(
            page_size=page_size, bytes_per_page=bytes_per_page
        )
        logger.warning(
            "DSV4 window SWA residency sizes the host tier from %s (%d tokens); "
            "--hicache-ratio %s is ignored.",
            source,
            host_tokens,
            get_memory().hicache_ratio,
        )
        full_host_pages = (host_tokens + page_size - 1) // page_size
        # TODO: size an independent NPU C128 pool (c128_attn_allocator) here;
        # C128 takes the page count of the KV-derived (GPU) sidecar.
        return (
            full_host_pages,
            (host_tokens + swa_page_size - 1) // swa_page_size,
            full_host_pages,
        )
    ratio = get_memory().hicache_ratio
    full_host_pages = int(device_full_pages * ratio)
    swa_host_pages = int(device_swa_pages * ratio)

    # NPU sizes the independent C128 host pool from its device page count.
    # For example, host pages = device pages * hicache_ratio.
    # GPU keeps the FULL page count because C128 is a KV-derived sidecar.
    c128_host_pages = full_host_pages
    c128_attn_allocator = getattr(allocator, "c128_attn_allocator", None)
    if c128_attn_allocator is not None:
        c128_host_pages = int(c128_attn_allocator.num_pages * ratio)
    return full_host_pages, swa_host_pages, c128_host_pages


def _dsv4_window_residency() -> bool:
    memory = get_memory()
    return bool(getattr(memory, "dsv4_uniform_cache_lifecycle", False)) and (
        getattr(memory, "dsv4_uniform_swa_residency", "full") == "window"
    )


def _dsv4_window_host_tokens(
    *, page_size: int, bytes_per_page: Optional[int]
) -> tuple[int, str]:
    """Window residency host capacity in tokens, and the flag it came from.

    --dsv4-hicache-host-tokens is taken literally. --dsv4-hicache-host-gb is a
    byte budget per byte-holding rank over every host pool: the whole pages
    of ``bytes_per_page`` (the active layout's bytes per ``page_size`` tokens)
    that fit in it. So the same budget holds about 1.8x the tokens with the
    tail-only SWA and c4 state layouts as with whole pages."""
    memory = get_memory()
    host_tokens = getattr(memory, "dsv4_hicache_host_tokens", None)
    host_gb = getattr(memory, "dsv4_hicache_host_gb", None)
    if host_gb is not None:
        if host_tokens is not None:
            raise ValueError(
                "--dsv4-uniform-swa-residency window takes one host size: "
                "--dsv4-hicache-host-tokens or --dsv4-hicache-host-gb, not both."
            )
        if not host_gb > 0:
            raise ValueError("--dsv4-hicache-host-gb must be positive.")
        if not bytes_per_page:
            raise ValueError(
                "--dsv4-hicache-host-gb needs the host bytes per page "
                "(window residency cache build)."
            )
        num_pages = int(round(host_gb * 1e9)) // bytes_per_page
        return num_pages * page_size, f"--dsv4-hicache-host-gb {host_gb:g}"
    if host_tokens is None or host_tokens <= 0:
        raise ValueError(
            "--dsv4-uniform-swa-residency window requires "
            "--dsv4-hicache-host-tokens or --dsv4-hicache-host-gb."
        )
    return host_tokens, "--dsv4-hicache-host-tokens"


def check_window_host_feasibility(
    *,
    host_tokens: int,
    device_full_tokens: int,
    bytes_per_page: int,
    node_host_budget_bytes: int,
    byte_ranks: int,
    page_size: int = 256,
    host_flag: Optional[str] = None,
) -> None:
    """Fail fast when a window-residency host tier cannot work.

    The host must hold at least 1.5x the device FULL tokens, and all host
    pools together must fit the node's host memory, summed over the ranks
    that hold bytes (one with host dedup, every rank on the node without).
    ``bytes_per_page`` covers every host pool per ``page_size`` tokens.
    ``host_flag``: the --dsv4-hicache-host-gb setting the tokens came from.
    """
    size = (
        f"{host_flag} ({host_tokens} tokens)"
        if host_flag
        else f"--dsv4-hicache-host-tokens {host_tokens}"
    )
    floor_tokens = (3 * device_full_tokens + 1) // 2
    if host_tokens < floor_tokens:
        raise ValueError(
            f"{size} is below 1.5x the device "
            f"FULL pool ({device_full_tokens} tokens): use at least "
            f"{floor_tokens}, or cap the device pool with --max-total-tokens."
        )
    num_pages = -(-host_tokens // page_size)
    requested_bytes = byte_ranks * num_pages * bytes_per_page
    if requested_bytes > node_host_budget_bytes:
        fit_tokens = node_host_budget_bytes // (byte_ranks * bytes_per_page) * page_size
        raise ValueError(
            f"{size} needs "
            f"{requested_bytes / 1e9:.1f} GB of host memory ({byte_ranks} "
            f"rank(s) x {bytes_per_page} B per {page_size}-token page) but only "
            f"{node_host_budget_bytes / 1e9:.1f} GB is available on this node: "
            f"use at most {fit_tokens} host tokens"
            + (
                f" ({fit_tokens // page_size * bytes_per_page / 1e9:.1f} GB per rank)"
                if host_flag
                else ""
            )
            + (" or --enable-dsv4-hicache-host-dedup" if byte_ranks > 1 else "")
            + f". The 1.5x floor of this device pool is {floor_tokens} tokens; "
            "lower --max-total-tokens if that does not fit."
        )


def _dsv4_host_bytes_per_page(
    kvcache: Any,
    *,
    page_size: int,
    mtp_swa_device_buffers: list,
    c4_state_global_layers: list,
    has_c4: bool,
    has_c128: bool,
    is_unified_kv: bool,
    swa_tail_layout: Optional[DSV4TailLayout] = None,
    state_window_rows: int = 0,
) -> int:
    """Host bytes per ``page_size`` tokens over every DSV4 host pool, computed
    the way each pool sizes itself in build_deepseek_v4_hicache_stack."""
    swa_pages_per_page = page_size // kvcache.swa_page_size
    total = 0
    if not is_unified_kv:
        swa_pool = kvcache.swa_kv_pool
        swa_layers = len(swa_pool.kv_buffer) + len(mtp_swa_device_buffers)
        swa_page_bytes = (
            swa_tail_layout.page_bytes
            if swa_tail_layout is not None
            else swa_pool.bytes_per_page_padded
        )
        total += swa_pages_per_page * swa_layers * swa_page_bytes
    if has_c4:
        c4_buffers, c4_item_bytes = _dsv4_compressed_region_buffers(kvcache, 4)
        total += len(c4_buffers) * c4_item_bytes
        # Upstream's indexer regions: one fused FP8 buffer, or FP4 payload +
        # scale (and the NPU split), each mirrored by its own host pool.
        for region in _dsv4_indexer_regions(kvcache, page_size):
            total += len(region.device_buffers) * region.item_bytes
        # TODO: count the unified-fp8 rope siblings (_dsv4_rope_sibling) and
        # the DSv4.1 low-ratio (C1/C2) pools.
        if not is_unified_kv:
            for state_pools in (
                kvcache.compress_state_pools,
                kvcache.indexer_compress_state_pools,
            ):
                for layer_id in c4_state_global_layers:
                    state = state_pools[layer_id]
                    slot_bytes = state.kv_score_buffer.kv_score[0].nbytes
                    rows = state.ring_size
                    if 0 < state_window_rows < rows:
                        # SGLANG_DSV4_WINDOW_HOST_STATE_TAIL_ONLY
                        rows = state_window_rows
                    total += swa_pages_per_page * rows * slot_bytes
    if has_c128:
        c128_buffers, c128_item_bytes = _dsv4_compressed_region_buffers(kvcache, 128)
        total += len(c128_buffers) * c128_item_bytes
    return total


def _check_dsv4_window_host(
    *,
    params: CacheInitParams,
    kvcache: Any,
    host_dedup: Any,
    num_host_pages: int,
    swa_num_host_pages: int,
    page_size: int,
    bytes_per_page: int,
) -> None:
    from decoupled_engine.srt.mem_cache.pool_host.base import (
        host_memory_budget_bytes,
        ranks_per_host,
    )

    local_ranks = ranks_per_host()
    byte_ranks = 1 if host_dedup is not None else local_ranks
    host_tokens, source = _dsv4_window_host_tokens(
        page_size=page_size, bytes_per_page=bytes_per_page
    )
    device_full_tokens = getattr(
        params.token_to_kv_pool_allocator, "size_full", kvcache.size
    )
    check_window_host_feasibility(
        host_tokens=host_tokens,
        device_full_tokens=device_full_tokens,
        bytes_per_page=bytes_per_page,
        node_host_budget_bytes=host_memory_budget_bytes() * local_ranks,
        byte_ranks=byte_ranks,
        page_size=page_size,
        host_flag=None if source == "--dsv4-hicache-host-tokens" else source,
    )
    logger.info(
        "DSV4 window SWA residency: device FULL %d / SWA %d tokens; host %d "
        "tokens (%.2fx device FULL) in %d KV and %d SWA pages; %.2f GB per "
        "byte-holding rank, %d such rank(s) on this node.",
        device_full_tokens,
        kvcache.swa_size,
        host_tokens,
        host_tokens / max(device_full_tokens, 1),
        num_host_pages,
        swa_num_host_pages,
        num_host_pages * bytes_per_page / 1e9,
        byte_ranks,
    )


def _dsv4_compressed_region_buffers(kvcache: Any, ratio: int) -> tuple[list, int]:
    """
    Resolve ``(device_buffers, item_bytes)`` for a DeepSeek V4 C4/C128 main-KV
    HiCache pool, hiding the device KV layout from the stack builder.
    """
    if getattr(kvcache, "_unified_kv", False):
        return kvcache.unified_region_buffers(ratio)
    pool = kvcache.c4_kv_pool if ratio == 4 else kvcache.c128_kv_pool
    return pool.kv_buffer, pool.bytes_per_page_padded


def dsv4_swa_page_tail_geometry(
    *,
    device_page_bytes: int,
    swa_page_size: int,
    token_data_bytes: int,
    token_scale_bytes: int,
    window: int,
) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    """Byte geometry of a DSV4 SWA device page for the tail-only host layout.

    The page is ``[swa_page_size x token_data_bytes]`` (nope+rope of each
    token) then ``[swa_page_size x token_scale_bytes]`` (scales + pad of each
    token) then padding to ``device_page_bytes`` (index_buf_accessor
    _set_k_and_s_triton). With an item of ``token_data_bytes``, token t's data
    is item t and the scale block starts at item ``swa_page_size``.

    Returns ``(start_item, item_bytes, head_byte_ranges)``: the host keeps
    items ``[start_item, device_page_bytes / item_bytes)``, i.e. the data of
    the last ``window`` tokens plus the whole scale block; the head tokens'
    data and scales are ``head_byte_ranges`` (what the poison debug hook
    overwrites).
    """
    if not 0 < window < swa_page_size:
        raise ValueError(
            f"tail-only SWA host pages need 0 < window ({window}) < SWA page "
            f"size ({swa_page_size})"
        )
    scale_start = swa_page_size * token_data_bytes
    if scale_start + swa_page_size * token_scale_bytes > device_page_bytes:
        raise ValueError(
            f"SWA page of {device_page_bytes} B cannot hold {swa_page_size} tokens "
            f"of {token_data_bytes} + {token_scale_bytes} B"
        )
    if device_page_bytes % token_data_bytes != 0:
        raise ValueError(
            f"SWA page of {device_page_bytes} B is not a whole number of "
            f"{token_data_bytes}-B token items"
        )
    head_tokens = swa_page_size - window
    head_ranges = (
        (0, head_tokens * token_data_bytes),
        (scale_start, scale_start + head_tokens * token_scale_bytes),
    )
    return head_tokens, token_data_bytes, head_ranges


# Ring rows of a c4 compress-state page that an extend or decode from a
# page-aligned prefix b reads: create_paged_compress_data_kernel loads the
# blocks of load_pos = b - 4 and load_pos - 4, i.e. positions [b - 8, b),
# rows [ring - 8, ring) of the boundary page's ring (ring | page size). The
# compressor window of get_compress_state_write_pad for ratio 4.
DSV4_C4_STATE_WINDOW_ROWS = 4 * 2


@dataclass(frozen=True)
class DSV4HostTailConfig:
    """SGLANG_DSV4_WINDOW_HOST_TAIL_ONLY / _STATE_TAIL_ONLY / _POISON_HEAD."""

    swa_tail: Optional[DSV4TailLayout] = None
    swa_poison: tuple = ()
    state_tail: bool = False
    state_poison: bool = False
    # Where each layout decision came from ("env", "default", "fallback",
    # "off", ...), for the startup config line only.
    sources: dict = field(default_factory=dict, compare=False)

    @property
    def state_window_rows(self) -> int:
        return DSV4_C4_STATE_WINDOW_ROWS if self.state_tail else 0


def _dsv4_swa_attention_window(params: CacheInitParams) -> int:
    """The widest SWA window any DSV4 attention reads: the backend's
    SWA_WINDOW and the model's sliding window (the DSpark draft reads
    config.sliding_window or config.window_size, the same field)."""
    from decoupled_engine.srt.layers.attention.deepseek_v4_backend import SWA_WINDOW

    windows = [int(SWA_WINDOW)]
    if params.sliding_window_size is not None:
        windows.append(int(params.sliding_window_size))
    return max(windows)


# Roles whose host SWA is read only by window load-backs at page boundaries
# (see _dsv4_tail_problems); the tail layout is their window-mode default.
_DSV4_TAIL_ROLES = ("prefill", "null", "symmetric")


def _dsv4_tail_problems(
    *,
    mode: str,
    memory: Any,
    storage_backend: Optional[str],
    kvcache: Any,
    page_size: int,
) -> list:
    """Why this worker cannot keep tail-only SWA host pages (empty: it can).

    Host SWA is read only by window load-backs at page boundaries on the
    prefill, symmetric and non-disaggregated roles. A symmetric owner's
    readers READ its device pages (export windows stay on the device until
    release); its export spills and the finish insert only write or adopt
    host pages. A decode worker's retraction snapshot and decode-side
    backfill read host SWA outside that rule, and a servicer or a storage
    backend serves whole pages. Host dedup is no problem: only TP0 holds
    host pages, and its load-backs broadcast whole device pages (the
    restored tail and TP0's never-read head) to the other ranks."""
    problems = []
    if mode not in _DSV4_TAIL_ROLES:
        problems.append(
            f"--disaggregation-mode {mode} (prefill, symmetric or null only)"
        )
    if getattr(memory, "enable_peer_kv_servicer", False):
        problems.append("--enable-peer-kv-servicer")
    if getattr(memory, "enable_peer_kv_gpu_export", False):
        problems.append("--enable-peer-kv-gpu-export")
    if storage_backend is not None:
        problems.append(f"--hicache-storage-backend {storage_backend}")
    if memory.hicache_io_backend != "kernel":
        problems.append(f"--hicache-io-backend {memory.hicache_io_backend}")
    if memory.hicache_mem_layout != "page_first":
        problems.append(f"--hicache-mem-layout {memory.hicache_mem_layout}")
    if kvcache.swa_page_size != page_size:
        # Host hits end on FULL page boundaries; the tail must end there.
        problems.append(
            f"SWA page size {kvcache.swa_page_size} != page size {page_size}"
        )
    return problems


def _dsv4_window_host_tail(
    *,
    params: CacheInitParams,
    kvcache: Any,
    storage_backend: Optional[str],
    page_size: int,
    is_unified_kv: bool,
) -> DSV4HostTailConfig:
    """SGLANG_DSV4_WINDOW_HOST_TAIL_ONLY / _STATE_TAIL_ONLY / _POISON_HEAD:
    the SWA host pool's tail layout (or None), the SWA head byte ranges to
    poison after loads, and whether the c4 state host pools keep (and poison)
    only part of each ring.

    Under window residency both layouts default on (the validated W-new10
    configuration) for the prefill, symmetric and non-disaggregated roles,
    with or without host dedup; a worker the tail cannot serve (a storage
    backend, the peer-KV servicer, ...) falls back to whole pages with one
    warning. An explicit value of either env wins: 0 turns the layout off,
    1 requires it."""
    from decoupled_engine.srt.environ import envs
    from decoupled_engine.srt.runtime_context import get_disagg

    tail_env = envs.SGLANG_DSV4_WINDOW_HOST_TAIL_ONLY
    state_env = envs.SGLANG_DSV4_WINDOW_HOST_STATE_TAIL_ONLY
    poison = envs.SGLANG_DSV4_WINDOW_HOST_TAIL_POISON_HEAD.get()
    window_mode = _dsv4_window_residency()
    mode = getattr(get_disagg(), "disaggregation_mode", "null")
    memory = get_memory()

    if tail_env.is_set():
        tail_only, tail_source = tail_env.get(), "env"
    elif window_mode and not is_unified_kv and mode in _DSV4_TAIL_ROLES:
        tail_only, tail_source = True, "default"
    else:
        tail_only = False
        tail_source = f"default-off-{mode}" if window_mode else "off"
    if tail_only:
        problems = _dsv4_tail_problems(
            mode=mode,
            memory=memory,
            storage_backend=storage_backend,
            kvcache=kvcache,
            page_size=page_size,
        )
        if problems and tail_source == "env":
            raise ValueError(
                "SGLANG_DSV4_WINDOW_HOST_TAIL_ONLY is not supported with: "
                + "; ".join(problems)
            )
        if problems:
            logger.warning(
                "Window residency defaults to tail-only SWA host pages "
                "(SGLANG_DSV4_WINDOW_HOST_TAIL_ONLY), but this worker keeps "
                "whole pages because of: %s. Set the env to 0 to silence this "
                "or remove the conflicting option to use the tail layout.",
                "; ".join(problems),
            )
            tail_only, tail_source = False, "fallback"
    if state_env.is_set():
        state_tail, state_source = state_env.get(), "env"
    else:
        state_tail, state_source = tail_only, ("default" if tail_only else "off")
    if state_tail and not tail_only:
        raise ValueError(
            "SGLANG_DSV4_WINDOW_HOST_STATE_TAIL_ONLY requires "
            "SGLANG_DSV4_WINDOW_HOST_TAIL_ONLY."
        )
    sources = {"tail": tail_source, "state_tail": state_source}
    if not tail_only and not poison:
        return DSV4HostTailConfig(sources=sources)
    flags = " / ".join(
        name
        for name, on in (
            ("SGLANG_DSV4_WINDOW_HOST_TAIL_ONLY", tail_only),
            ("SGLANG_DSV4_WINDOW_HOST_TAIL_POISON_HEAD", poison),
        )
        if on
    )
    if not window_mode:
        raise ValueError(f"{flags} requires --dsv4-uniform-swa-residency window.")
    if is_unified_kv:
        raise ValueError(f"{flags} requires a separate SWA host pool (no unified KV).")
    swa_pool = kvcache.swa_kv_pool
    window = _dsv4_swa_attention_window(params)
    start_item, item_bytes, head_ranges = dsv4_swa_page_tail_geometry(
        device_page_bytes=swa_pool.bytes_per_page_padded,
        swa_page_size=kvcache.swa_page_size,
        token_data_bytes=(
            swa_pool.qk_nope_head_dim
            + swa_pool.qk_rope_head_dim * swa_pool.rope_storage_dtype.itemsize
        ),
        token_scale_bytes=(
            swa_pool.qk_nope_head_dim // swa_pool.quantize_block_size
            + swa_pool.scale_pad
        ),
        window=window,
    )
    tail = None
    if tail_only:
        tail = DSV4TailLayout(
            device_page_bytes=swa_pool.bytes_per_page_padded,
            item_bytes=item_bytes,
            start_item=start_item,
        )
        logger.info(
            "SGLANG_DSV4_WINDOW_HOST_TAIL_ONLY (%s): the SWA host pool keeps "
            "items [%d, %d) of %d B of each %d-B SWA page (%d B, the last %d "
            "tokens' data and the scale block), window %d.",
            tail_source,
            start_item,
            tail.device_items_per_page,
            item_bytes,
            swa_pool.bytes_per_page_padded,
            tail.page_bytes,
            window,
            window,
        )
    if state_tail:
        logger.info(
            "SGLANG_DSV4_WINDOW_HOST_STATE_TAIL_ONLY (%s): the c4 state host "
            "pools keep the last %d rows of each page's compress-state ring.",
            state_source,
            DSV4_C4_STATE_WINDOW_ROWS,
        )
    if poison:
        logger.warning(
            "SGLANG_DSV4_WINDOW_HOST_TAIL_POISON_HEAD: debug only; every "
            "host->device SWA page load overwrites bytes %s of each loaded "
            "page with 0xFF, and every c4 state load the ring rows before the "
            "last %d (tail-only %s, state tail-only %s).",
            list(head_ranges),
            DSV4_C4_STATE_WINDOW_ROWS,
            "on" if tail_only else "off",
            "on" if state_tail else "off",
        )
    return DSV4HostTailConfig(
        swa_tail=tail,
        swa_poison=head_ranges if poison else (),
        state_tail=state_tail,
        state_poison=poison,
        sources=sources,
    )


def _log_dsv4_window_config(
    *,
    params: CacheInitParams,
    host_tail: DSV4HostTailConfig,
    num_host_pages: int,
    bytes_per_page: int,
    whole_bytes_per_page: int,
) -> None:
    """Log the effective window configuration once (TP rank 0) as one
    grep-able ``DSV4_WINDOW_CONFIG`` line. ``whole_bytes_per_page``: the
    bytes per page without the tail-only layouts."""
    from decoupled_engine.srt.mem_cache.dsv4_window_defaults import format_window_config_line
    from decoupled_engine.srt.mem_cache.unified_radix_cache import swa_spill_sm_copy
    from decoupled_engine.srt.runtime_context import get_disagg

    group = getattr(params, "tp_cache_group", None)
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank(group=group) if group is not None else dist.get_rank()
            if rank != 0:
                return
    except Exception:  # logging must never break the cache build
        pass
    memory = get_memory()
    page_size = params.page_size
    host_tokens, source = _dsv4_window_host_tokens(
        page_size=page_size, bytes_per_page=bytes_per_page
    )
    mode = getattr(get_disagg(), "disaggregation_mode", "null")
    logger.info(
        format_window_config_line(
            mode=getattr(mode, "value", mode),
            host_tail=host_tail,
            host_sizing="gb" if source != "--dsv4-hicache-host-tokens" else "tokens",
            host_tokens=host_tokens,
            num_host_pages=num_host_pages,
            bytes_per_page=bytes_per_page,
            swa_backfill=getattr(memory, "dsv4_swa_backfill", "auto"),
            decode_spill_sm_copy=swa_spill_sm_copy(decode=True),
            export_spill_sm_copy=swa_spill_sm_copy(decode=False),
        )
    )
    if (
        source == "--dsv4-hicache-host-tokens"
        and host_tail.sources.get("tail") == "default"
        and bytes_per_page < whole_bytes_per_page
    ):
        # The default tail layouts shrink each host page: a token count sized
        # for whole pages now pins less memory instead of reaching further.
        # The suggested budget rounds the whole-page bytes up to 1 MB, so it
        # also keeps every page of a whole-page worker sized the same way.
        whole_bytes = num_host_pages * whole_bytes_per_page
        suggest_mb = -(-whole_bytes // 10**6)
        logger.warning(
            "DSV4 window host sized by --dsv4-hicache-host-tokens %d: with the "
            "default tail-only host layouts (%d B per %d-token page instead of "
            "%d B) that is %.2f GB per rank, not the %.2f GB of whole pages. "
            "To size the host by bytes, pass --dsv4-hicache-host-gb instead "
            "(--dsv4-hicache-host-gb %.3f holds %d tokens here).",
            host_tokens,
            bytes_per_page,
            page_size,
            whole_bytes_per_page,
            num_host_pages * bytes_per_page / 1e9,
            whole_bytes / 1e9,
            suggest_mb / 1000,
            suggest_mb * 10**6 // bytes_per_page * page_size,
        )


def _require_single_row_dsv4_swa_pages(
    *, logical_page_size: int, physical_page_size: int, consumer: str
) -> None:
    """Reject consumers that cannot map one logical SWA page to many rows."""
    if logical_page_size != physical_page_size:
        raise ValueError(
            f"{consumer} does not support the DeepSeek-V4 direct SWA KV layout "
            f"({logical_page_size}-token logical pages stored as "
            f"{physical_page_size}-token physical rows). Disable "
            "SGLANG_OPT_SM120_DIRECT_SWA_KV for this configuration."
        )


def _dsv4_page_aligned_only(pool: Any) -> bool:
    """Whether a pool may only move whole pages: the token-granular copy
    (``transfer_cache_dsv4_mla``) hardcodes the V4 data/scale row split."""
    layout = getattr(pool, "kv_layout", None)
    return layout is not None and layout.value != "v4"


@dataclass(frozen=True)
class _IndexerRegion:
    """One page-contiguous indexer buffer group to mirror on the host."""

    name: PoolName
    device_buffers: list
    item_bytes: int
    slot_page_size: int
    # FP4 page rows group their slots instead of laying tokens out flat, so the
    # fused-row token-granular copy does not apply and transfers must be whole
    # pages. The fused FP8 row has no such restriction.
    page_aligned_only: bool


def _dsv4_indexer_regions(kvcache: Any, page_size: int) -> list[_IndexerRegion]:
    """
    Resolve the indexer HiCache regions, hiding the FP8/FP4 split from the
    stack builder. FP8 keeps key and scale fused in one buffer, while FP4
    stores payload and scale separately, so it maps to two host pools.
    """
    import torch

    pool = kvcache.c4_indexer_kv_pool
    if getattr(pool, "has_npu_storage", False):
        # NPU stores C4 int8 K and fp16 scale in separate PA_ND buffers.  Keep
        # the FULL logical page geometry for HiCache indices. The transfer path
        # derives the native C4 page geometry from these device buffers.
        k_buffers = pool.index_k_buffer
        scale_buffers = pool.index_scale_buffer
        return [
            _IndexerRegion(
                name=PoolName.DEEPSEEK_V4_C4_INDEXER,
                device_buffers=k_buffers,
                item_bytes=int(k_buffers[0][0].numel() * k_buffers[0].element_size()),
                slot_page_size=page_size,
                page_aligned_only=False,
            ),
            _IndexerRegion(
                name=PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE,
                device_buffers=scale_buffers,
                item_bytes=int(
                    scale_buffers[0][0].numel() * scale_buffers[0].element_size()
                ),
                slot_page_size=page_size,
                page_aligned_only=False,
            ),
        ]

    fused = pool.index_k_with_scale_buffer
    if fused is not None:
        return [
            _IndexerRegion(
                name=PoolName.DEEPSEEK_V4_C4_INDEXER,
                device_buffers=fused,
                item_bytes=fused[0].shape[1] * fused[0].element_size(),
                slot_page_size=page_size,
                page_aligned_only=False,
            )
        ]

    payload_ref = pool.index_k_payload_buffer[0]
    scale_ref = pool.index_k_scale_buffer[0]
    # A page row covers ``page_slots`` C4 slots, i.e. one tree page of tokens
    # after 4:1 compression.
    page_slots = payload_ref.shape[3]
    if scale_ref.shape[3] != page_slots:
        raise ValueError(
            "FP4 indexer payload and scale must agree on slots per page: "
            f"payload={page_slots}, scale={scale_ref.shape[3]}"
        )
    if page_size % page_slots != 0:
        raise ValueError(
            f"Tree page size {page_size} must be a multiple of the FP4 indexer "
            f"slots per page {page_slots}"
        )
    payload = [b.view(torch.uint8).flatten(1) for b in pool.index_k_payload_buffer]
    scale = [b.view(torch.uint8).flatten(1) for b in pool.index_k_scale_buffer]
    return [
        _IndexerRegion(
            name=PoolName.DEEPSEEK_V4_C4_INDEXER,
            device_buffers=payload,
            item_bytes=payload[0].shape[1],
            slot_page_size=page_size,
            page_aligned_only=True,
        ),
        _IndexerRegion(
            name=PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE,
            device_buffers=scale,
            item_bytes=scale[0].shape[1],
            slot_page_size=page_size,
            page_aligned_only=True,
        ),
    ]


def _dsv4_low_ratio_regions(
    kvcache: Any, page_size: int
) -> list[tuple[PoolName, Any, list, dict[int, int]]]:
    """(name, device_pool, buffers, layer_mapping) per shared source KV and
    indexer buffer group, in FULL-page units."""
    import torch

    regions = []
    for ratio, names in (
        (
            1,
            (
                PoolName.DEEPSEEK_V4_C1,
                PoolName.DEEPSEEK_V4_C1_INDEXER,
                PoolName.DEEPSEEK_V4_C1_INDEXER_SCALE,
            ),
        ),
        (
            2,
            (
                PoolName.DEEPSEEK_V4_C2,
                PoolName.DEEPSEEK_V4_C2_INDEXER,
                PoolName.DEEPSEEK_V4_C2_INDEXER_SCALE,
            ),
        ),
    ):
        sources = getattr(kvcache, "sources_by_ratio", {}).get(ratio, [])
        if not sources:
            continue
        kv_pool = kvcache.kv_pools[ratio]
        index_pool = kvcache.index_pools[ratio]
        assert page_size % ratio == 0
        slots_per_page = page_size // ratio
        assert slots_per_page % index_pool.page_size == 0
        index_pages_per_full_page = slots_per_page // index_pool.page_size
        layer_mapping = {
            source - kvcache.start_layer: index for index, source in enumerate(sources)
        }
        regions.append((names[0], kv_pool, kv_pool.kv_buffer, layer_mapping))
        if index_pool.index_k_with_scale_buffer is not None:
            index_regions = [(names[1], index_pool.index_k_with_scale_buffer)]
        else:
            index_regions = [
                (names[1], index_pool.index_k_payload_buffer),
                (names[2], index_pool.index_k_scale_buffer),
            ]
        for name, buffers in index_regions:
            # Drop only the padding rows past the FULL page address space.
            rows = []
            for buffer in buffers:
                full_pages = buffer.shape[0] // index_pages_per_full_page
                rows.append(
                    buffer[: full_pages * index_pages_per_full_page]
                    .view(torch.uint8)
                    .reshape(full_pages, -1)
                )
            regions.append((name, index_pool, rows, layer_mapping))
    return regions


def _dsv4_low_ratio_entries(
    kvcache: Any, page_size: int, num_host_pages: int, transfer_layer_id_max: int
):
    """Mirror shared KV; ratio-2 request state is rebuilt at page boundaries."""
    entries = []
    for name, device_pool, buffers, layer_mapping in _dsv4_low_ratio_regions(
        kvcache, page_size
    ):
        entries.append(
            build_pool_entry(
                name=name,
                host_pool=DeepSeekV4PagedHostPool(
                    pool_name=str(name),
                    device_buffers=buffers,
                    item_bytes=buffers[0].shape[1] * buffers[0].element_size(),
                    num_host_pages=num_host_pages,
                    slot_page_size=page_size,
                    layout=get_memory().hicache_mem_layout,
                    allocator_type=_get_allocator_type(),
                    page_aligned_only=True,
                ),
                device_pool=device_pool,
                layer_mapping=layer_mapping,
                transfer_layer_id_max=transfer_layer_id_max,
            )
        )
    return entries


def _dsv4_rope_sibling(
    kvcache: Any, ratio: int
) -> Optional[tuple[PoolName, list, int]]:
    """
    ``(name, device_buffers, item_bytes)`` for the bf16 rope half of an fp8
    two-pool unified_kv row; None for every other layout, which keeps the whole
    row in one buffer.
    """
    if not getattr(kvcache, "_unified_kv", False):
        return None
    region = kvcache.unified_rope_region_buffers(ratio)
    if region is None:
        return None
    buffers, item_bytes = region
    name = (
        PoolName.DEEPSEEK_V4_C4_ROPE if ratio == 4 else PoolName.DEEPSEEK_V4_C128_ROPE
    )
    return name, buffers, item_bytes


def _build_dsv4_rope_entry(
    kvcache: Any,
    ratio: int,
    *,
    device_pool: Any,
    layer_mapping: dict[int, int],
    num_host_pages: int,
    slot_page_size: int,
    transfer_layer_id_max: int,
) -> Optional[PoolEntry]:
    sibling = _dsv4_rope_sibling(kvcache, ratio)
    if sibling is None:
        return None
    name, device_buffers, item_bytes = sibling
    return build_pool_entry(
        name=name,
        host_pool=DeepSeekV4PagedHostPool(
            pool_name=str(name),
            device_buffers=device_buffers,
            item_bytes=item_bytes,
            num_host_pages=num_host_pages,
            slot_page_size=slot_page_size,
            layout=get_memory().hicache_mem_layout,
            allocator_type=_get_allocator_type(),
            page_aligned_only=True,
        ),
        device_pool=device_pool,
        layer_mapping=layer_mapping,
        transfer_layer_id_max=transfer_layer_id_max,
    )


def build_deepseek_v4_hicache_stack(
    *,
    params: CacheInitParams,
    kvcache: Any,
    load_cache_event,
    storage_backend: Optional[str],
    host_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    device_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    host_c128_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
    layer_mappings: Optional[_DeepSeekV4LayerMappings] = None,
) -> tuple[HostPoolGroup, HybridCacheController]:
    from decoupled_engine.srt.mem_cache.dsv4_host_dedup import maybe_build_dsv4_host_dedup

    host_dedup = maybe_build_dsv4_host_dedup(
        params, kvcache, storage_backend, storage_backend_extra_config
    )
    page_size = params.page_size
    layer_mappings = layer_mappings or _resolve_deepseek_v4_layer_mappings(kvcache)
    transfer_layer_id_max = layer_mappings.transfer_layer_id_max
    full_layer_mapping = layer_mappings.full

    is_unified_kv = getattr(kvcache, "_unified_kv", False)
    has_paged_swa = not is_unified_kv and kvcache.swa_kv_pool is not None
    mtp_swa_device_buffers = []
    if not has_paged_swa:
        # Unified KV and encoder replay rebuild SWA state; keep it out of host cache.
        swa_layer_mapping = {}
    else:
        _require_single_row_dsv4_swa_pages(
            logical_page_size=kvcache.swa_page_size,
            physical_page_size=kvcache.swa_kv_pool.page_size,
            consumer="DeepSeek-V4 HiCache",
        )
        if len(kvcache.swa_kv_pool.kv_buffer) != transfer_layer_id_max:
            raise ValueError(
                "DeepSeek V4 SWA KV pool must be PP-stage-local: "
                f"got {len(kvcache.swa_kv_pool.kv_buffer)} buffers for "
                f"{transfer_layer_id_max} local layers"
            )
        swa_layer_mapping = layer_mappings.swa
        # Keep every uncompressed draft SWA layer after the target SWA layers.
        # NextN has one layer per pool, while DSpark keeps all stages in one pool.
        mtp_swa_device_buffers = [
            buffer
            for pool in params.mtp_draft_device_pools
            for buffer in pool.swa_kv_pool.kv_buffer
        ]
        swa_layer_mapping = with_packed_draft_layer_mapping(
            swa_layer_mapping,
            transfer_layer_start=transfer_layer_id_max,
            target_device_layer_num=transfer_layer_id_max,
            draft_layer_num=len(mtp_swa_device_buffers),
        )

    c4_layer_mapping = layer_mappings.c4
    c128_layer_mapping = layer_mappings.c128
    c4_state_mapping = layer_mappings.c4_state
    c4_state_global_layers = layer_mappings.c4_state_global_layers
    host_tail = _dsv4_window_host_tail(
        params=params,
        kvcache=kvcache,
        storage_backend=storage_backend,
        page_size=page_size,
        is_unified_kv=is_unified_kv,
    )
    bytes_per_page = whole_bytes_per_page = None
    if _dsv4_window_residency():
        # Before the page count: --dsv4-hicache-host-gb converts at the
        # active layout's bytes per page.
        page_bytes_args = dict(
            page_size=page_size,
            mtp_swa_device_buffers=mtp_swa_device_buffers,
            c4_state_global_layers=c4_state_global_layers,
            has_c4=bool(c4_layer_mapping),
            has_c128=bool(c128_layer_mapping),
            is_unified_kv=is_unified_kv,
        )
        bytes_per_page = _dsv4_host_bytes_per_page(
            kvcache,
            swa_tail_layout=host_tail.swa_tail,
            state_window_rows=host_tail.state_window_rows,
            **page_bytes_args,
        )
        whole_bytes_per_page = _dsv4_host_bytes_per_page(kvcache, **page_bytes_args)
    (
        num_host_pages,
        swa_num_host_pages,
        c128_num_host_pages,
    ) = _deepseek_v4_num_host_pages(
        params=params,
        kvcache=kvcache,
        page_size=page_size,
        swa_page_size=kvcache.swa_page_size,
        bytes_per_page=bytes_per_page,
    )
    if _dsv4_window_residency():
        # Check the aggregate before any pool pins host memory.
        _check_dsv4_window_host(
            params=params,
            kvcache=kvcache,
            host_dedup=host_dedup,
            num_host_pages=num_host_pages,
            swa_num_host_pages=swa_num_host_pages,
            page_size=page_size,
            bytes_per_page=bytes_per_page,
        )
        _log_dsv4_window_config(
            params=params,
            host_tail=host_tail,
            num_host_pages=num_host_pages,
            bytes_per_page=bytes_per_page,
            whole_bytes_per_page=whole_bytes_per_page,
        )

    logical_host_pool = LogicalHostPool(
        num_host_pages * page_size,
        page_size,
        layout=get_memory().hicache_mem_layout,
        reserve_pages=_peer_kv_reserve_pages(num_host_pages, "kv"),
    )
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=logical_host_pool,
            device_pool=kvcache,
            layer_mapping=full_layer_mapping,
            transfer_layer_id_max=transfer_layer_id_max,
            is_anchor=True,
        ),
    ]

    if has_paged_swa:
        swa_host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(PoolName.SWA),
            device_buffers=[
                *kvcache.swa_kv_pool.kv_buffer,
                *mtp_swa_device_buffers,
            ],
            item_bytes=kvcache.swa_kv_pool.bytes_per_page_padded,
            num_host_pages=swa_num_host_pages,
            slot_page_size=kvcache.swa_page_size,
            layout=get_memory().hicache_mem_layout,
            allocator_type=_get_allocator_type(),
            page_aligned_only=_dsv4_page_aligned_only(kvcache.swa_kv_pool),
            host_dedup=host_dedup,
            reserve_pages=_peer_kv_reserve_pages(swa_num_host_pages, "swa"),
            tail_layout=host_tail.swa_tail,
            poison_byte_ranges=host_tail.swa_poison,
        )
        swa_attn_allocator = params.token_to_kv_pool_allocator.swa_attn_allocator
        entries.append(
            build_pool_entry(
                name=PoolName.SWA,
                host_pool=swa_host_pool,
                device_pool=kvcache.swa_kv_pool,
                layer_mapping=swa_layer_mapping,
                transfer_layer_id_max=transfer_layer_id_max
                + len(mtp_swa_device_buffers),
                host_evict_fn=host_swa_evict_fn,
                device_evict_fn=device_swa_evict_fn,
                device_alloc_fn=swa_attn_allocator.alloc,
                device_free_fn=swa_attn_allocator.free,
                packed_draft_device_pools=tuple(mtp_swa_device_buffers),
            )
        )

    if c4_layer_mapping:
        c4_device_buffers, c4_item_bytes = _dsv4_compressed_region_buffers(kvcache, 4)
        # C4 is KV-derived, so its HiCache indices stay in the FULL logical
        # coordinate space. NPU transfer code derives the native C4 page size
        # (for example 32 rather than 128) from the device-buffer shape.
        c4_host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C4),
            device_buffers=c4_device_buffers,
            item_bytes=c4_item_bytes,
            num_host_pages=num_host_pages,
            slot_page_size=page_size,
            layout=get_memory().hicache_mem_layout,
            allocator_type=_get_allocator_type(),
            page_aligned_only=is_unified_kv
            or _dsv4_page_aligned_only(kvcache.c4_kv_pool),
            host_dedup=host_dedup,
        )
        entries.append(
            build_pool_entry(
                name=PoolName.DEEPSEEK_V4_C4,
                host_pool=c4_host_pool,
                device_pool=kvcache.c4_kv_pool,
                layer_mapping=c4_layer_mapping,
                transfer_layer_id_max=transfer_layer_id_max,
            )
        )
        for region in _dsv4_indexer_regions(kvcache, page_size):
            entries.append(
                build_pool_entry(
                    name=region.name,
                    host_pool=DeepSeekV4PagedHostPool(
                        pool_name=str(region.name),
                        device_buffers=region.device_buffers,
                        item_bytes=region.item_bytes,
                        num_host_pages=num_host_pages,
                        slot_page_size=region.slot_page_size,
                        layout=get_memory().hicache_mem_layout,
                        allocator_type=_get_allocator_type(),
                        page_aligned_only=region.page_aligned_only,
                        host_dedup=host_dedup,
                    ),
                    device_pool=kvcache.c4_indexer_kv_pool,
                    layer_mapping=c4_layer_mapping,
                    transfer_layer_id_max=transfer_layer_id_max,
                )
            )

        # Build c4 rope buffer when using unified fp8 kv
        c4_rope_entry = _build_dsv4_rope_entry(
            kvcache,
            4,
            device_pool=kvcache.c4_kv_pool,
            layer_mapping=c4_layer_mapping,
            num_host_pages=num_host_pages,
            slot_page_size=page_size,
            transfer_layer_id_max=transfer_layer_id_max,
        )
        if c4_rope_entry is not None:
            entries.append(c4_rope_entry)

        if not is_unified_kv:
            c4_state_host_pool = DeepSeekV4StateHostPool(
                pool_name=str(PoolName.DEEPSEEK_V4_C4_STATE),
                state_pools=[
                    kvcache.compress_state_pools[layer_id]
                    for layer_id in c4_state_global_layers
                ],
                num_host_pages=swa_num_host_pages,
                swa_page_size=kvcache.swa_page_size,
                layout=get_memory().hicache_mem_layout,
                allocator_type=_get_allocator_type(),
                host_dedup=host_dedup,
                window_rows=DSV4_C4_STATE_WINDOW_ROWS,
                tail_only=host_tail.state_tail,
                poison_head=host_tail.state_poison,
            )
            c4_indexer_state_host_pool = DeepSeekV4StateHostPool(
                pool_name=str(PoolName.DEEPSEEK_V4_C4_INDEXER_STATE),
                state_pools=[
                    kvcache.indexer_compress_state_pools[layer_id]
                    for layer_id in c4_state_global_layers
                ],
                num_host_pages=swa_num_host_pages,
                swa_page_size=kvcache.swa_page_size,
                layout=get_memory().hicache_mem_layout,
                allocator_type=_get_allocator_type(),
                host_dedup=host_dedup,
                window_rows=DSV4_C4_STATE_WINDOW_ROWS,
                tail_only=host_tail.state_tail,
                poison_head=host_tail.state_poison,
            )
            entries.extend(
                [
                    build_pool_entry(
                        name=PoolName.DEEPSEEK_V4_C4_STATE,
                        host_pool=c4_state_host_pool,
                        device_pool=None,
                        layer_mapping=c4_state_mapping,
                        transfer_layer_id_max=transfer_layer_id_max,
                    ),
                    build_pool_entry(
                        name=PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
                        host_pool=c4_indexer_state_host_pool,
                        device_pool=None,
                        layer_mapping=c4_state_mapping,
                        transfer_layer_id_max=transfer_layer_id_max,
                    ),
                ]
            )

    if c128_layer_mapping:
        c128_device_buffers, c128_item_bytes = _dsv4_compressed_region_buffers(
            kvcache, 128
        )
        # NPU C128 host views use kernel_page_size, for example 16 instead of 128.
        # GPU pools lack this attribute and fall back to the global page_size.
        _c128_pool = getattr(kvcache, "c128_kv_pool", None)
        c128_slot_page_size = getattr(_c128_pool, "kernel_page_size", page_size)
        # NPU derives the independent C128 host budget from its device pool.
        c128_attn_allocator = getattr(
            params.token_to_kv_pool_allocator, "c128_attn_allocator", None
        )
        c128_host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C128),
            device_buffers=c128_device_buffers,
            item_bytes=c128_item_bytes,
            num_host_pages=c128_num_host_pages,
            slot_page_size=c128_slot_page_size,
            layout=get_memory().hicache_mem_layout,
            allocator_type=_get_allocator_type(),
            page_aligned_only=is_unified_kv
            or _dsv4_page_aligned_only(kvcache.c128_kv_pool),
            host_dedup=host_dedup,
        )
        # C128 state pool is intentionally not registered with hicache.
        # page_size=256 % 128 == 0, so state pool is not consumed on load.
        entries.extend(
            [
                build_pool_entry(
                    name=PoolName.DEEPSEEK_V4_C128,
                    host_pool=c128_host_pool,
                    device_pool=kvcache.c128_kv_pool,
                    layer_mapping=c128_layer_mapping,
                    transfer_layer_id_max=transfer_layer_id_max,
                    # NPU C128 uses bare allocator callbacks for independent indices.
                    # FULL leaf eviction also frees each attached C128 host value.
                    # GPU keeps its KV-derived sidecar callbacks unset.
                    host_evict_fn=host_c128_evict_fn,
                    device_alloc_fn=(
                        c128_attn_allocator.alloc
                        if c128_attn_allocator is not None
                        else None
                    ),
                    device_free_fn=(
                        c128_attn_allocator.free
                        if c128_attn_allocator is not None
                        else None
                    ),
                ),
            ]
        )
        # Build c128 rope buffer when using unified fp8 kv
        c128_rope_entry = _build_dsv4_rope_entry(
            kvcache,
            128,
            device_pool=kvcache.c128_kv_pool,
            layer_mapping=c128_layer_mapping,
            num_host_pages=c128_num_host_pages,
            slot_page_size=c128_slot_page_size,
            transfer_layer_id_max=transfer_layer_id_max,
        )
        if c128_rope_entry is not None:
            entries.append(c128_rope_entry)

    # TODO: host-dedup the DSv4.1 low-ratio (C1/C2) pools (every rank keeps its
    # own copy) and count them in the window-residency bytes per page
    # (_dsv4_host_bytes_per_page).
    entries.extend(
        _dsv4_low_ratio_entries(
            kvcache, page_size, num_host_pages, transfer_layer_id_max
        )
    )

    host_pool_group = HostPoolGroup(entries)
    host_pool_group.host_dedup = host_dedup
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        params.tp_cache_group,
        load_cache_event=load_cache_event,
        attn_cp_group=params.attn_cp_cache_group,
        attn_tp_group=params.attn_tp_cache_group,
        pp_group=params.pp_cache_group,
        write_policy=get_memory().hicache_write_policy,
        io_backend=get_memory().hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_id_max=transfer_layer_id_max,
        enable_storage_metrics=enable_storage_metrics,
        host_memory_mode=get_memory().hicache_host_memory_mode,
    )
    return host_pool_group, cache_controller


def build_mamba_host_pool(
    mamba_pool,
    pool_info: DevicePoolInfo,
    *,
    host_size: float,
    layout: str,
    mtp_draft_device_pools=(),
):
    draft_buffer_infos = []
    for draft_pool in mtp_draft_device_pools:
        (draft_info,) = draft_pool.get_device_pool_infos()
        draft_info.buffer_info.validate(layer_ids=draft_info.layer_ids)
        draft_buffer_infos.append(draft_info.buffer_info)
    capacity = mamba_pool.host_capacity_tokens
    if capacity is None:
        capacity = mamba_pool.size
    return MambaPoolHost(
        pool_info.buffer_info,
        get_memory().hicache_ratio,
        host_size,
        device_capacity=capacity,
        mtp_draft_buffer_infos=draft_buffer_infos,
        allocator_type=_get_allocator_type(),
        layout=layout,
    )


def build_hybrid_mamba_stack(
    *,
    params: CacheInitParams,
    decls: tuple[HostPoolDecl, ...],
    mamba_pool: Any,
    full_layer_mapping: dict[int, int],
    start_layer: int,
    load_cache_event,
    storage_backend: Optional[str],
    use_mla: bool,
    qsa_pool: Any = None,
    host_mamba_evict_fn: Optional[Callable[[int], Any]] = None,
    device_mamba_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController, HostPoolGroupConfig]:
    """KV plus every pool the hybrid pool declares (e.g. a sparse indexer),
    then the Mamba state pool, which keeps its own host path."""
    (mamba_info,) = mamba_pool.get_device_pool_infos()
    mamba_info.buffer_info.validate(layer_ids=mamba_info.layer_ids)
    # Controller IDs are stage-local; Info IDs follow the physical layer axis.
    mamba_layer_mapping = {
        layer - start_layer: index for index, layer in enumerate(mamba_info.layer_ids)
    }
    transfer_layer_id_max = (
        max(full_layer_mapping.keys() | mamba_layer_mapping.keys()) + 1
    )
    mamba_allocator = params.req_to_token_pool.mamba_allocator
    qsa_draft_pools = params.mtp_draft_device_pools if qsa_pool is not None else ()
    config = prepare_host_pool_config(
        decls=decls,
        full_layer_mapping=full_layer_mapping,
        transfer_layer_id_max=transfer_layer_id_max,
        transfer_page_size=params.page_size,
        packed_draft_device_pools=params.mtp_draft_device_pools,
    )
    root = config.pools[0]
    kv_pool = root.decl.device_pool
    kv_host_size, mamba_host_size = None, 0
    if get_memory().hicache_size > 0:
        kv_host_size, mamba_host_size = _split_hicache_size(
            get_memory().hicache_size, (kv_pool, mamba_pool)
        )
    if qsa_pool is not None and kv_host_size is not None:
        kv_bytes = sum(kv_pool.get_kv_size_bytes())
        index_bytes = sum(
            b.nbytes
            for pool in (qsa_pool, *qsa_draft_pools)
            for b in pool.qsa_compressed_k_buffer_pool
        )
        kv_host_size *= kv_bytes / (kv_bytes + index_bytes)
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=params.page_size,
        use_mla=use_mla,
        host_size=kv_host_size,
        mtp_draft_device_pools=root.packed_draft_device_pools,
    )
    # MambaPoolHost only supports page_first_direct; the global layout may be
    # page_first_kv_split (e.g. MLA + KDA hybrid on NPU). The Mamba/KDA state
    # pool has no separate K/V buffers, so kv_split does not apply; override
    # to page_first_direct.
    mamba_layout = (
        "page_first_direct"
        if get_memory().hicache_mem_layout == "page_first_kv_split"
        else get_memory().hicache_mem_layout
    )
    mamba_host_pool = build_mamba_host_pool(
        mamba_pool,
        mamba_info,
        host_size=mamba_host_size,
        layout=mamba_layout,
        mtp_draft_device_pools=params.mtp_draft_mamba_pools,
    )
    mamba_layer_mapping = with_packed_draft_layer_mapping(
        mamba_layer_mapping,
        transfer_layer_start=transfer_layer_id_max,
        target_device_layer_num=len(mamba_info.layer_ids),
        draft_layer_num=len(params.mtp_draft_mamba_pools),
    )
    entries = _build_pool_entries(config=config, root_host_pool=kv_host_pool) + [
        build_pool_entry(
            name=PoolName.MAMBA,
            host_pool=mamba_host_pool,
            device_pool=mamba_pool,
            layer_mapping=mamba_layer_mapping,
            transfer_layer_id_max=transfer_layer_id_max
            + len(params.mtp_draft_mamba_pools),
            host_evict_fn=host_mamba_evict_fn,
            device_evict_fn=device_mamba_evict_fn,
            device_alloc_fn=mamba_allocator.alloc,
            device_free_fn=mamba_allocator.free,
            packed_draft_device_pools=params.mtp_draft_mamba_pools,
        ),
    ]
    if qsa_pool is not None:
        entries.append(
            build_pool_entry(
                name=PoolName.INDEXER,
                host_pool=QSAIndexerPoolHost(
                    qsa_pool,
                    kv_host_pool,
                    allocator_type=_get_allocator_type(),
                    mtp_draft_device_pools=qsa_draft_pools,
                ),
                device_pool=qsa_pool,
                # The KV config carries the packed draft tail layers.
                layer_mapping=root.layer_mapping,
                transfer_layer_id_max=root.transfer_layer_id_max,
                packed_draft_device_pools=qsa_draft_pools,
            )
        )
    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        params.page_size,
        params.tp_cache_group,
        load_cache_event=load_cache_event,
        attn_cp_group=params.attn_cp_cache_group,
        attn_tp_group=params.attn_tp_cache_group,
        pp_group=params.pp_cache_group,
        write_policy=get_memory().hicache_write_policy,
        io_backend=get_memory().hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_id_max=transfer_layer_id_max,
        enable_storage_metrics=enable_storage_metrics,
        host_memory_mode=get_memory().hicache_host_memory_mode,
    )
    return host_pool_group, cache_controller, config


def build_hybrid_mamba_swa_stack(
    *,
    params: CacheInitParams,
    full_kv_pool: Any,
    swa_kv_pool: Any,
    mamba_pool: Any,
    full_layer_mapping: dict[int, int],
    swa_layer_mapping: dict[int, int],
    start_layer: int,
    page_size: int,
    tp_group,
    load_cache_event,
    attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
    storage_backend: Optional[str],
    host_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    device_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    host_mamba_evict_fn: Optional[Callable[[int], Any]] = None,
    device_mamba_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    (mamba_info,) = mamba_pool.get_device_pool_infos()
    mamba_info.buffer_info.validate(layer_ids=mamba_info.layer_ids)
    # Controller IDs are stage-local; Info IDs follow the physical layer axis.
    mamba_layer_mapping = {
        layer - start_layer: index for index, layer in enumerate(mamba_info.layer_ids)
    }
    transfer_layer_id_max = (
        max(
            full_layer_mapping.keys()
            | swa_layer_mapping.keys()
            | mamba_layer_mapping.keys()
        )
        + 1
    )
    swa_attn_allocator = params.token_to_kv_pool_allocator.swa_attn_allocator
    mamba_allocator = params.req_to_token_pool.mamba_allocator
    kv_host_size, swa_host_size, mamba_host_size = None, None, 0
    if get_memory().hicache_size > 0:
        kv_host_size, swa_host_size, mamba_host_size = _split_hicache_size(
            get_memory().hicache_size, (full_kv_pool, swa_kv_pool, mamba_pool)
        )
    host_pool_group = build_hybrid_swa_group(
        page_size=page_size,
        full_kv_pool=full_kv_pool,
        swa_kv_pool=swa_kv_pool,
        full_layer_mapping=full_layer_mapping,
        swa_layer_mapping=swa_layer_mapping,
        use_mla=False,
        kv_host_size=kv_host_size,
        swa_host_size=swa_host_size,
        host_swa_evict_fn=host_swa_evict_fn,
        device_swa_evict_fn=device_swa_evict_fn,
        swa_attn_allocator=swa_attn_allocator,
        mtp_full_device_pools=tuple(
            pool.full_kv_pool
            for pool in params.mtp_draft_device_pools
            if pool.full_kv_pool.layer_num
        ),
        mtp_swa_device_pools=_mtp_swa_device_pools(params, full_indices=False),
        mtp_swa_full_device_pools=_mtp_swa_device_pools(params, full_indices=True),
        transfer_layer_id_max=transfer_layer_id_max,
    )
    mamba_host_pool = build_mamba_host_pool(
        mamba_pool,
        mamba_info,
        host_size=mamba_host_size,
        layout=get_memory().hicache_mem_layout,
        mtp_draft_device_pools=params.mtp_draft_mamba_pools,
    )
    mamba_layer_mapping = with_packed_draft_layer_mapping(
        mamba_layer_mapping,
        transfer_layer_start=transfer_layer_id_max,
        target_device_layer_num=len(mamba_info.layer_ids),
        draft_layer_num=len(params.mtp_draft_mamba_pools),
    )
    host_pool_group.add_entry(
        build_pool_entry(
            name=PoolName.MAMBA,
            host_pool=mamba_host_pool,
            device_pool=mamba_pool,
            layer_mapping=mamba_layer_mapping,
            transfer_layer_id_max=transfer_layer_id_max
            + len(params.mtp_draft_mamba_pools),
            host_evict_fn=host_mamba_evict_fn,
            device_evict_fn=device_mamba_evict_fn,
            device_alloc_fn=mamba_allocator.alloc,
            device_free_fn=mamba_allocator.free,
            packed_draft_device_pools=params.mtp_draft_mamba_pools,
        ),
    )
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        attn_cp_group=attn_cp_group,
        attn_tp_group=attn_tp_group,
        pp_group=pp_group,
        write_policy=get_memory().hicache_write_policy,
        io_backend=get_memory().hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_id_max=transfer_layer_id_max,
        enable_storage_metrics=enable_storage_metrics,
        host_memory_mode=get_memory().hicache_host_memory_mode,
    )
    return host_pool_group, cache_controller


def build_host_pool_group(
    *,
    config: HostPoolGroupConfig,
    host_dedup=None,
) -> HostPoolGroup:
    """Allocate host pools and transfer entries from prepared configs.

    ``host_dedup``: the host-dedup role of this rank (DSA host dedup); the
    root pool takes it and layout-following pools inherit it from the root.
    """
    root = config.pools[0]
    root_host_pool = build_kv_host_pool(
        kv_pool=root.decl.device_pool,
        page_size=config.transfer_page_size,
        mtp_draft_device_pools=root.packed_draft_device_pools,
        host_dedup=host_dedup,
    )
    host_pool_group = HostPoolGroup(
        _build_pool_entries(config=config, root_host_pool=root_host_pool)
    )
    host_pool_group.host_dedup = host_dedup
    return host_pool_group


def _build_pool_entries(
    *, config: HostPoolGroupConfig, root_host_pool: Any
) -> list[PoolEntry]:
    host_pools = {config.pools[0].decl.pool_name: root_host_pool}
    entries = []
    for pool_config in config.pools:
        decl = pool_config.decl
        if not decl.is_layout_root:
            host_pools[decl.pool_name] = decl.host_pool_builder.build(
                decl=decl,
                anchor_host=host_pools[decl.layout_source],
                allocator_type=_get_allocator_type(),
                packed_draft_device_pools=pool_config.packed_draft_device_pools,
            )
        entries.append(
            build_pool_entry(
                name=decl.pool_name,
                host_pool=host_pools[decl.pool_name],
                device_pool=decl.device_pool,
                layer_mapping=pool_config.layer_mapping,
                transfer_layer_id_max=pool_config.transfer_layer_id_max,
                is_anchor=decl.is_primary,
                packed_draft_device_pools=pool_config.packed_draft_device_pools,
            )
        )
    return entries


def _build_mha_mla_host_pool(
    *,
    pool: Any,
    host_to_device_ratio: float,
    page_size: int,
    layout: str,
    allocator_type: str,
    pool_label: str,
    host_token_capacity: int | None = None,
    host_dedup=None,
):
    from decoupled_engine.srt.mem_cache.memory_pool import MHATokenToKVPool

    # The global layout is page_first_kv_split only when the target model
    # uses MLA; that layout is MLA-specific, so MHA draft pools must use
    # the non-MLA layout (NPU default: page_first_direct).
    if isinstance(pool, MHATokenToKVPool) and layout == "page_first_kv_split":
        layout = "page_first_direct"

    kwargs = dict(
        host_to_device_ratio=host_to_device_ratio,
        host_size=0,
        page_size=page_size,
        layout=layout,
        allocator_type=allocator_type,
        pool_label=pool_label,
        host_token_capacity=host_token_capacity,
    )
    if isinstance(pool, MHATokenToKVPool):
        if host_dedup is not None:
            raise ValueError("Host dedup requires replicated MLA draft KV")
        host_cls, host_input, host_kwargs = _mha_host_inputs(pool)
        return host_cls(host_input, **host_kwargs, **kwargs)
    host_pool = MLATokenToKVPoolHost(
        pool.get_device_pool_infos()[0].buffer_info,
        device_capacity_tokens=pool.size
        if pool.host_capacity_tokens is None
        else pool.host_capacity_tokens,
        start_layer=pool.start_layer,
        end_layer=pool.end_layer,
        target_layer_num=pool.layer_num,
        override_kv_cache_dim=pool.kv_cache_dim,
        host_dedup=host_dedup,
        **kwargs,
    )
    if host_dedup is not None:
        _attach_host_dedup_device_pools(host_pool, pool)
    return host_pool


def build_full_draft_pools(
    *,
    draft_kv_pool: Any,
    tree_cache: Any,
) -> tuple[list[SidecarPoolSpec], list[PoolEntry]]:
    """Build draft KV/DSA sidecars whose indices follow target full KV."""
    from decoupled_engine.srt.mem_cache.memory_pool import DSATokenToKVPool, HybridLinearKVPool
    from decoupled_engine.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    pool = draft_kv_pool
    if isinstance(pool, HybridLinearKVPool):
        # Hybrid draft runners keep their sole attention layer in this sub-pool.
        pool = pool.full_kv_pool
    if pool.layer_num == 0:
        return [], []

    controller = tree_cache.cache_controller
    host_pool_group = controller.mem_pool_host
    if pool.layer_num > controller.transfer_layer_id_max:
        raise ValueError("Draft HiCache layers exceed the target transfer stages")

    # Note(kpham-sgl): DCP x DSpark draft KV is replicated and spans the virtual
    # loc space, so match the target host's logical_size instead of physical size.
    draft_host_pool = _build_mha_mla_host_pool(
        pool=pool,
        host_to_device_ratio=host_pool_group.logical_size / pool.size,
        page_size=controller.page_size,
        layout=get_memory().hicache_mem_layout,
        allocator_type=_get_allocator_type(),
        pool_label="draft",
        host_token_capacity=host_pool_group.logical_size,
        host_dedup=getattr(host_pool_group, "host_dedup", None),
    )
    draft_layer_mapping = {i: i for i in range(pool.layer_num)}
    # Host-dedup C2g: the draft's source device buffers are keyed apart from
    # the target's (its host layer ids also start at 0).
    from decoupled_engine.srt.mem_cache.dedup_ipc_fanout import SIDECAR_DRAFT_KIND

    draft_host_pool.ce_peer_kind = SIDECAR_DRAFT_KIND

    specs = [
        SidecarPoolSpec(
            pool_name=PoolName.DRAFT,
            indices_from_pool=PoolName.KV,
        )
    ]
    # The entry rides the KV anchor's indices like its spec: the peer-KV
    # servicer derives the served pools from PoolEntry.indices_from_pool, and
    # an unanchored draft entry was "not served", which clipped every P2P
    # grant to 0 pages (the reader requires ALL_PAGES of the draft). The
    # draft host pool's own free list is never used (the tree allocates on KV).
    entries = [
        build_pool_entry(
            name=PoolName.DRAFT,
            host_pool=draft_host_pool,
            device_pool=pool,
            layer_mapping=draft_layer_mapping,
            transfer_layer_id_max=draft_host_pool.layer_num,
            indices_from_pool=PoolName.KV,
        )
    ]

    if isinstance(pool, DSATokenToKVPool) and pool.index_k_with_scale_buffer:
        from decoupled_engine.srt.mem_cache.pool_host.dsa import (
            DSAIndexerPoolHost,
            make_dsa_indexer_pool_decl,
        )

        # Separate draft indexer: its own host pool laid out on the draft KV
        # host pool, but transfer indices still follow the target KV anchor.
        indexer_decl = make_dsa_indexer_pool_decl(pool, name=PoolName.DRAFT_INDEXER)
        indexer_host_pool = DSAIndexerPoolHost(
            decl=indexer_decl,
            anchor_host=draft_host_pool,
            allocator_type=_get_allocator_type(),
        )
        specs.append(indexer_decl.sidecar_spec())
        entries.append(
            build_pool_entry(
                name=PoolName.DRAFT_INDEXER,
                host_pool=indexer_host_pool,
                device_pool=pool,
                layer_mapping=draft_layer_mapping,
                transfer_layer_id_max=indexer_host_pool.layer_num,
                indices_from_pool=PoolName.KV,
            )
        )

    if isinstance(draft_kv_pool, QSATokenToKVPool):
        specs.append(SidecarPoolSpec(PoolName.DRAFT_INDEXER, PoolName.KV))
        entries.append(
            build_pool_entry(
                name=PoolName.DRAFT_INDEXER,
                host_pool=QSAIndexerPoolHost(
                    draft_kv_pool,
                    draft_host_pool,
                    allocator_type=_get_allocator_type(),
                ),
                device_pool=draft_kv_pool,
                layer_mapping=draft_layer_mapping,
                transfer_layer_id_max=draft_host_pool.layer_num,
                indices_from_pool=PoolName.KV,
            )
        )

    return specs, entries


def build_swa_draft_pools(
    *,
    draft_kv_pool: Any,
    tree_cache: Any,
) -> tuple[list[SidecarPoolSpec], list[PoolEntry]]:
    """Build a draft SWA sidecar whose indices follow target SWA."""
    if getattr(tree_cache, "dsv4_uniform_swa_residency", "full") == "window":
        # The sidecar below is TRAILING_PAGES, so the host would not hold the
        # draft SWA prefix that window residency promises.
        raise ValueError(
            "--dsv4-uniform-swa-residency window requires the packed draft SWA "
            "host layout; a separate draft SWA sidecar is not supported."
        )
    draft_swa_pool = draft_kv_pool.swa_kv_pool
    if draft_swa_pool is None:
        raise NotImplementedError(
            "HiCache draft SWA sidecar requires a non-unified draft SWA pool."
        )
    if draft_swa_pool.layer_num == 0:
        return [], []
    controller = tree_cache.cache_controller
    host_pool_group = controller.mem_pool_host
    target_swa_host_pool = host_pool_group.entry_map[PoolName.SWA].host_pool

    if isinstance(target_swa_host_pool, DeepSeekV4PagedHostPool):
        _require_single_row_dsv4_swa_pages(
            logical_page_size=target_swa_host_pool.slot_page_size,
            physical_page_size=draft_swa_pool.page_size,
            consumer="DeepSeek-V4 MTP SWA HiCache sidecar",
        )
        host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(PoolName.DRAFT_SWA),
            device_buffers=draft_swa_pool.kv_buffer,
            item_bytes=draft_swa_pool.bytes_per_page_padded,
            num_host_pages=target_swa_host_pool.num_host_pages,
            slot_page_size=draft_swa_pool.page_size,
            layout=target_swa_host_pool.layout,
            allocator_type=_get_allocator_type(),
        )
    else:
        host_pool = _build_mha_mla_host_pool(
            pool=draft_swa_pool,
            host_to_device_ratio=target_swa_host_pool.size / draft_swa_pool.size,
            page_size=target_swa_host_pool.page_size,
            layout=target_swa_host_pool.layout,
            allocator_type=_get_allocator_type(),
            pool_label="draft_swa",
        )

    layer_mapping = {i: i for i in range(draft_swa_pool.layer_num)}
    spec = SidecarPoolSpec(
        pool_name=PoolName.DRAFT_SWA,
        indices_from_pool=PoolName.SWA,
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )
    entry = build_pool_entry(
        name=PoolName.DRAFT_SWA,
        host_pool=host_pool,
        device_pool=draft_swa_pool,
        layer_mapping=layer_mapping,
        transfer_layer_id_max=host_pool.layer_num,
    )
    return [spec], [entry]


def validate_fused_hicache_draft(
    *,
    target_config: Optional[KVCacheConfigurator] = None,
    draft_device_pools: Optional[tuple[Any, ...]] = None,
    tree_cache: Any = None,
) -> None:
    """Validate fusion at sizing, draft binding, and host registration.

    Sizing supplies only the target config. Draft binding also supplies the
    device pools; registration supplies the pools and the completed cache so
    its page envelopes and shared buffer can be checked.
    """
    if target_config is not None and not (
        target_config.spec_algorithm.is_dflash()
        and target_config.is_hybrid_swa
        and target_config.mambaish_config is None
        and get_memory().hicache_storage_backend is None
    ):
        raise NotImplementedError(
            "--enable-hierarchical-cache fused draft KV requires DFLASH, a "
            "hybrid-SWA target without Mamba, and L2-only storage."
        )

    if draft_device_pools is not None:
        from decoupled_engine.srt.mem_cache.unified_draft_pool import UnifiedDraftKVPool

        if not (
            len(draft_device_pools) == 1
            and isinstance(draft_device_pools[0], UnifiedDraftKVPool)
            and draft_device_pools[0]._host_sub_pool_name == "full"
        ):
            raise NotImplementedError(
                "HiCache fused draft KV requires a full-attention DFLASH draft."
            )

    if tree_cache is None:
        return
    assert draft_device_pools is not None

    entries = tree_cache.cache_controller.mem_pool_host.entry_map
    if any(
        name not in entries
        or not isinstance(entries[name].host_pool, UnifiedPageEnvelopeHostPool)
        for name in (PoolName.KV, PoolName.SWA)
    ):
        raise NotImplementedError(
            "Fused DFLASH HiCache requires unified hybrid-SWA page-envelope host "
            "pools; the selected storage backend or host layout does not preserve "
            "the fused draft bytes."
        )
    target = entries[PoolName.KV].device_pool
    if draft_device_pools[0]._unified_buffer is not target._unified_buffer:
        raise ValueError("Fused DFLASH HiCache draft must share the target FULL pool.")


def build_hicache_draft_sidecars(
    *,
    draft_device_pools: tuple[Any, ...],
    tree_cache: Any,
) -> tuple[list[SidecarPoolSpec], list[PoolEntry]]:
    """Compose the full and SWA draft-sidecar paths."""
    from decoupled_engine.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool

    assert len(draft_device_pools) == 1
    draft_kv_pool = draft_device_pools[0]
    builder = (
        build_swa_draft_pools
        if isinstance(draft_kv_pool, BaseSWAKVPool)
        else build_full_draft_pools
    )
    return builder(
        draft_kv_pool=draft_kv_pool,
        tree_cache=tree_cache,
    )


_COMPONENT_HOST_ATTR: dict[ComponentType, tuple[str, str]] = {
    ComponentType.FULL: ("full_kv_pool_host", "_full_kv_pool_host"),
    ComponentType.SWA: ("swa_kv_pool_host", "_swa_kv_pool_host"),
    ComponentType.MAMBA: ("mamba_pool_host", "_mamba_pool_host"),
    ComponentType.C128: ("c128_kv_pool_host", "_c128_kv_pool_host"),
}


@dataclass
class StackBuildResult:
    host_pool_group: HostPoolGroup
    cache_controller: HybridCacheController
    component_host_pools: dict[ComponentType, Any]
    sidecars: list[SidecarPoolSpec] = field(default_factory=list)
    # Mamba state lives in req_to_token_pool, not in kvcache, so its
    # layer_transfer_counter has to be wired separately.
    register_req_to_token_counter: bool = False
    pools_desc: str = ""
    pool_declarations: Optional[tuple[HostPoolDecl, ...]] = None


class StackStrategy:
    def matches(self, kvcache: Any, components: set[ComponentType]) -> bool:
        raise NotImplementedError

    def build_direct_linker_pool_group(
        self,
        *,
        kvcache: Any,
        params: CacheInitParams,
        page_size: int,
    ):
        raise ValueError(
            "The selected hybrid pool strategy does not support the direct "
            f"external linker: {type(self).__name__}."
        )

    def build(
        self,
        *,
        cache: UnifiedRadixCache,
        kvcache: Any,
        params: CacheInitParams,
        server_args: ServerArgs,
        load_cache_event,
        storage_backend: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        enable_storage_metrics: bool = False,
    ) -> StackBuildResult:
        raise NotImplementedError


def _delegate_c128_host_evict(cache, n: int) -> int:
    """Delegate C128 host pressure to FULL leaf eviction.

    For example, ``n`` C128 slots request ``n * 128`` FULL tokens."""
    return cache.evict_host(n * 128, ComponentType.FULL)


class _DeepSeekV4Strategy(StackStrategy):
    def matches(self, kvcache, components):
        from decoupled_engine.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

        if not isinstance(kvcache, DeepSeekV4TokenToKVPool):
            return False
        return components in (
            {ComponentType.FULL, ComponentType.SWA},
            {ComponentType.FULL, ComponentType.SWA, ComponentType.C128},
        ) or (components == {ComponentType.FULL} and kvcache.swa_kv_pool is None)

    def build_direct_linker_pool_group(self, *, kvcache, params, page_size):
        from decoupled_engine.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
            _build_deepseek_v4_device_pool_group,
        )

        return _build_deepseek_v4_device_pool_group(
            kvcache, page_size, params.mtp_draft_device_pools
        )

    def build(
        self,
        *,
        cache,
        kvcache,
        params,
        server_args,
        load_cache_event,
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        layer_mappings = _resolve_deepseek_v4_layer_mappings(kvcache)
        host_pool_group, cache_controller = build_deepseek_v4_hicache_stack(
            params=params,
            kvcache=kvcache,
            load_cache_event=load_cache_event,
            storage_backend=storage_backend,
            host_swa_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA),
            device_swa_evict_fn=lambda n: _evict_swa_for_device_alloc(cache, n),
            # NPU delegates C128 host pressure to FULL host-leaf eviction.
            # _delegate_c128_host_evict converts C128 slots to FULL tokens.
            host_c128_evict_fn=(
                (lambda n: _delegate_c128_host_evict(cache, n))
                if ComponentType.C128 in cache.components
                else None
            ),
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
            layer_mappings=layer_mappings,
        )
        # NPU drives C128 as an independent tree component, so adding a KV-derived
        # sidecar would duplicate transfers. Add that sidecar only on GPU.
        # The *_ROPE entries only resolve under unified fp8 kv; entry_map filters
        # them out everywhere else.
        _sidecar_srcs = [
            (PoolName.DEEPSEEK_V4_C1, PoolName.KV),
            (PoolName.DEEPSEEK_V4_C1_INDEXER, PoolName.KV),
            (PoolName.DEEPSEEK_V4_C1_INDEXER_SCALE, PoolName.KV),
            (PoolName.DEEPSEEK_V4_C2, PoolName.KV),
            (PoolName.DEEPSEEK_V4_C2_INDEXER, PoolName.KV),
            (PoolName.DEEPSEEK_V4_C2_INDEXER_SCALE, PoolName.KV),
            (PoolName.DEEPSEEK_V4_C4, PoolName.KV),
            (PoolName.DEEPSEEK_V4_C4_ROPE, PoolName.KV),
            (PoolName.DEEPSEEK_V4_C4_INDEXER, PoolName.KV),
            (PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE, PoolName.KV),
            (PoolName.DEEPSEEK_V4_C4_STATE, PoolName.SWA),
            (PoolName.DEEPSEEK_V4_C4_INDEXER_STATE, PoolName.SWA),
            (PoolName.DEEPSEEK_V4_C128_STATE, PoolName.SWA),
        ]
        if ComponentType.C128 not in cache.components:
            _sidecar_srcs.append((PoolName.DEEPSEEK_V4_C128, PoolName.KV))
            _sidecar_srcs.append((PoolName.DEEPSEEK_V4_C128_ROPE, PoolName.KV))
        sidecars = [
            SidecarPoolSpec(
                pool_name=name,
                indices_from_pool=src,
                hit_policy=(
                    PoolHitPolicy.TRAILING_PAGES
                    if src == PoolName.SWA
                    and not getattr(cache, "dsv4_uniform_cache_lifecycle", False)
                    else PoolHitPolicy.ALL_PAGES
                ),
            )
            for name, src in _sidecar_srcs
            if name in host_pool_group.entry_map
        ]
        component_host_pools = {
            ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
        }
        if PoolName.SWA in host_pool_group.entry_map:
            component_host_pools[ComponentType.SWA] = host_pool_group.get_pool(
                PoolName.SWA
            )
        if (
            ComponentType.C128 in cache.components
            and PoolName.DEEPSEEK_V4_C128 in host_pool_group.entry_map
        ):
            component_host_pools[ComponentType.C128] = host_pool_group.get_pool(
                PoolName.DEEPSEEK_V4_C128
            )

        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools=component_host_pools,
            sidecars=sidecars,
            pools_desc="KV + SWA + DeepSeekV4 sidecars",
        )


class _MambaStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from decoupled_engine.srt.mem_cache.memory_pool import HybridLinearKVPool

        return isinstance(kvcache, HybridLinearKVPool) and components == {
            ComponentType.FULL,
            ComponentType.MAMBA,
        }

    def build_direct_linker_pool_group(self, *, kvcache, params, page_size):
        from decoupled_engine.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
            _build_mamba_device_pool_group,
        )

        return _build_mamba_device_pool_group(kvcache, params, page_size)

    def build(
        self,
        *,
        cache,
        kvcache,
        params,
        server_args,
        load_cache_event,
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        from decoupled_engine.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

        qsa_pool = kvcache if isinstance(kvcache, QSATokenToKVPool) else None
        full_layer_mapping = _stage_local_layer_mapping(
            kvcache.full_attention_layer_id_mapping, kvcache.start_layer
        )
        host_pool_group, cache_controller, config = build_hybrid_mamba_stack(
            params=params,
            decls=kvcache.host_pool_decls(),
            mamba_pool=params.req_to_token_pool.mamba_pool,
            full_layer_mapping=full_layer_mapping,
            start_layer=kvcache.start_layer,
            load_cache_event=load_cache_event,
            storage_backend=storage_backend,
            use_mla=kvcache.use_mla,
            qsa_pool=qsa_pool,
            host_mamba_evict_fn=lambda n: cache.evict_host(n, ComponentType.MAMBA),
            device_mamba_evict_fn=lambda n: _evict_mamba_for_device_alloc(cache, n),
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
                ComponentType.MAMBA: host_pool_group.get_pool(PoolName.MAMBA),
            },
            sidecars=[
                c.decl.sidecar_spec() for c in config.pools if not c.decl.is_primary
            ]
            + (
                [SidecarPoolSpec(PoolName.INDEXER, PoolName.KV)]
                if qsa_pool is not None
                else []
            ),
            pool_declarations=tuple(c.decl for c in config.pools),
            register_req_to_token_counter=True,
            pools_desc=" + ".join(
                [c.decl.pool_name.value.upper() for c in config.pools]
                + ["MAMBA"]
                + (["INDEXER"] if qsa_pool is not None else [])
            ),
        )


def _swa_layer_mappings(kvcache) -> tuple[dict[int, int], dict[int, int]]:
    full = {
        gid - kvcache.start_layer: lid
        for gid, (lid, is_swa) in kvcache.layers_mapping.items()
        if not is_swa
    }
    swa = {
        gid - kvcache.start_layer: lid
        for gid, (lid, is_swa) in kvcache.layers_mapping.items()
        if is_swa
    }
    return full, swa


class _SwaStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from decoupled_engine.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
        from decoupled_engine.srt.mem_cache.swa_memory_pool import SWAKVPool

        return (
            isinstance(kvcache, SWAKVPool)
            and not isinstance(kvcache, DeepSeekV4TokenToKVPool)
            and components == {ComponentType.FULL, ComponentType.SWA}
        )

    def build(
        self,
        *,
        cache,
        kvcache,
        params,
        server_args,
        load_cache_event,
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        full_layer_mapping, swa_layer_mapping = _swa_layer_mappings(kvcache)
        host_pool_group, cache_controller = build_hybrid_swa_stack(
            params=params,
            full_kv_pool=kvcache.full_kv_pool,
            swa_kv_pool=kvcache.swa_kv_pool,
            full_layer_mapping=full_layer_mapping,
            swa_layer_mapping=swa_layer_mapping,
            load_cache_event=load_cache_event,
            storage_backend=storage_backend,
            use_mla=False,
            host_full_evict_fn=lambda n: cache.evict_host(n, ComponentType.FULL),
            host_swa_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA),
            device_swa_evict_fn=lambda n: _evict_swa_for_device_alloc(cache, n),
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
                ComponentType.SWA: host_pool_group.get_pool(PoolName.SWA),
            },
            pools_desc="Full + SWA",
            sidecars=_mtp_swa_sidecar_specs(host_pool_group),
        )


class _MambaSwaStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from decoupled_engine.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
        from decoupled_engine.srt.mem_cache.swa_memory_pool import SWAKVPool

        return (
            isinstance(kvcache, SWAKVPool)
            and not isinstance(kvcache, DeepSeekV4TokenToKVPool)
            and components
            == {ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA}
        )

    def build_direct_linker_pool_group(self, *, kvcache, params, page_size):
        from decoupled_engine.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
            _build_mamba_swa_device_pool_group,
        )

        return _build_mamba_swa_device_pool_group(kvcache, params, page_size)

    def build(
        self,
        *,
        cache,
        kvcache,
        params,
        server_args,
        load_cache_event,
        attn_cp_group=None,
        attn_tp_group=None,
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        full_layer_mapping, swa_layer_mapping = _swa_layer_mappings(kvcache)
        host_pool_group, cache_controller = build_hybrid_mamba_swa_stack(
            params=params,
            full_kv_pool=kvcache.full_kv_pool,
            swa_kv_pool=kvcache.swa_kv_pool,
            mamba_pool=params.req_to_token_pool.mamba_pool,
            full_layer_mapping=full_layer_mapping,
            swa_layer_mapping=swa_layer_mapping,
            start_layer=kvcache.start_layer,
            page_size=cache.page_size,
            tp_group=params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=params.pp_cache_group,
            storage_backend=storage_backend,
            host_swa_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA),
            device_swa_evict_fn=lambda n: _evict_swa_for_device_alloc(cache, n),
            host_mamba_evict_fn=lambda n: cache.evict_host(n, ComponentType.MAMBA),
            device_mamba_evict_fn=lambda n: _evict_mamba_for_device_alloc(cache, n),
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
                ComponentType.SWA: host_pool_group.get_pool(PoolName.SWA),
                ComponentType.MAMBA: host_pool_group.get_pool(PoolName.MAMBA),
            },
            register_req_to_token_counter=True,
            pools_desc="KV + SWA + MAMBA",
            sidecars=_mtp_swa_sidecar_specs(host_pool_group),
        )


class _MiniMaxSparseStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from decoupled_engine.srt.mem_cache.memory_pool import MiniMaxSparseKVPool

        return isinstance(kvcache, MiniMaxSparseKVPool) and components == {
            ComponentType.FULL
        }

    def build(
        self,
        *,
        cache,
        kvcache,
        params,
        server_args,
        load_cache_event,
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        host_pool_group, cache_controller = build_minimax_sparse_hicache_stack(
            params=params,
            sparse_pool=kvcache,
            load_cache_event=load_cache_event,
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        sidecars = []
        pools_desc = "KV"
        if kvcache.index_k_pool is not None:
            sidecars.append(
                SidecarPoolSpec(
                    pool_name=PoolName.INDEXER,
                    indices_from_pool=PoolName.KV,
                )
            )
            pools_desc = "KV + INDEXER(k-only)"
        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
            },
            sidecars=sidecars,
            pools_desc=pools_desc,
        )


class _PlainKvStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from decoupled_engine.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
        from decoupled_engine.srt.mem_cache.memory_pool import (
            DSATokenToKVPool,
            HybridLinearKVPool,
            MiniMaxSparseKVPool,
        )
        from decoupled_engine.srt.mem_cache.swa_memory_pool import SWAKVPool

        if isinstance(
            kvcache,
            (
                SWAKVPool,
                HybridLinearKVPool,
                DSATokenToKVPool,
                MiniMaxSparseKVPool,
                DeepSeekV4TokenToKVPool,
            ),
        ):
            return False
        return components == {ComponentType.FULL}

    def build(
        self,
        *,
        cache,
        kvcache,
        params,
        server_args,
        load_cache_event,
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        from decoupled_engine.srt.mem_cache.memory_pool import MLATokenToKVPool

        full_kv_pool = kvcache
        use_mla = isinstance(kvcache, MLATokenToKVPool)
        full_layer_mapping = {i: i for i in range(full_kv_pool.layer_num)}
        host_pool_group, cache_controller = build_kv_only_stack(
            params=params,
            kv_pool=full_kv_pool,
            full_layer_mapping=full_layer_mapping,
            load_cache_event=load_cache_event,
            storage_backend=storage_backend,
            use_mla=use_mla,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
            },
            pools_desc="KV",
        )


class _DsaStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from decoupled_engine.srt.mem_cache.memory_pool import DSATokenToKVPool

        return isinstance(kvcache, DSATokenToKVPool) and components == {
            ComponentType.FULL
        }

    def build_direct_linker_pool_group(self, *, kvcache, params, page_size):
        from decoupled_engine.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
            _build_dsa_device_pool_group,
        )

        return _build_dsa_device_pool_group(
            kvcache, page_size, params.mtp_draft_device_pools
        )

    def build(
        self,
        *,
        cache,
        kvcache,
        params,
        server_args,
        load_cache_event,
        storage_backend=None,
        storage_backend_extra_config=None,
        prefetch_threshold=256,
        model_name=None,
        enable_storage_metrics=False,
    ):
        enable_peer_serving = _resolved(server_args, "enable_peer_kv_servicer")
        if enable_peer_serving and get_memory().hicache_mem_layout not in (
            "page_first",
            "page_first_direct",
        ):
            raise ValueError(
                "DSA peer KV requires a page_first or page_first_direct host layout"
            )
        from decoupled_engine.srt.mem_cache.dsa_host_dedup import maybe_build_dsa_host_dedup

        host_dedup = maybe_build_dsa_host_dedup(
            params, kvcache, storage_backend, storage_backend_extra_config
        )
        config = prepare_host_pool_config(
            decls=kvcache.host_pool_decls(),
            full_layer_mapping={i: i for i in range(kvcache.layer_num)},
            transfer_layer_id_max=kvcache.layer_num,
            transfer_page_size=params.page_size,
            packed_draft_device_pools=params.mtp_draft_device_pools,
        )
        host_pool_group = build_host_pool_group(config=config, host_dedup=host_dedup)
        if enable_peer_serving:
            kv_host_pool = host_pool_group.anchor_entry.host_pool
            kv_host_pool.enable_peer_serving(
                reserve_pages=_peer_kv_reserve_pages(kv_host_pool.page_num, "kv")
            )
        cache_controller = HybridCacheController(
            params.token_to_kv_pool_allocator,
            host_pool_group,
            config.transfer_page_size,
            params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=params.attn_cp_cache_group,
            attn_tp_group=params.attn_tp_cache_group,
            pp_group=params.pp_cache_group,
            write_policy=get_memory().hicache_write_policy,
            io_backend=get_memory().hicache_io_backend,
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            # The controller adds packed draft transfers after target layers.
            transfer_layer_id_max=kvcache.layer_num,
            enable_storage_metrics=enable_storage_metrics,
            host_memory_mode=get_memory().hicache_host_memory_mode,
        )
        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
            },
            sidecars=[
                c.decl.sidecar_spec() for c in config.pools if not c.decl.is_primary
            ],
            pools_desc=" + ".join(c.decl.pool_name.value.upper() for c in config.pools),
            pool_declarations=tuple(c.decl for c in config.pools),
        )


# Resolved first-to-last; _PlainKvStrategy is the catch-all fallback.
_STRATEGIES: list[StackStrategy] = [
    _DeepSeekV4Strategy(),
    _MambaStrategy(),
    _SwaStrategy(),
    _MambaSwaStrategy(),
    _DsaStrategy(),
    _MiniMaxSparseStrategy(),
    _PlainKvStrategy(),
]


def register_stack_strategy(strategy: StackStrategy) -> None:
    """Prepend a strategy so downstream forks can plug in (kvcache, components)
    combinations not in the built-in list."""
    _STRATEGIES.insert(0, strategy)


def _select_strategy(kvcache: Any, components: set[ComponentType]) -> StackStrategy:
    for strategy in _STRATEGIES:
        if strategy.matches(kvcache, components):
            return strategy
    raise AssertionError(
        f"No matching HiCache strategy for kvcache={type(kvcache).__name__}, "
        f"components={sorted(c.name for c in components)}"
    )


# Strategies that assemble from host_pool_decls(), so every declared pool has an
# entry by construction. A miss here is a bug, not an unsupported combination.
_DECLARATION_VERIFIED_STRATEGIES: tuple[type, ...] = (_DsaStrategy, _MambaStrategy)


def _check_declared_pools_present(
    *, kvcache: Any, result: StackBuildResult, strategy: StackStrategy
) -> None:
    entries = result.host_pool_group.entry_map
    decls = result.pool_declarations
    if decls is None:
        decls = kvcache.host_pool_decls()
    missing = [d.pool_name.value for d in decls if d.pool_name not in entries]
    if not missing:
        return
    msg = (
        f"{type(kvcache).__name__} declares host pools {missing} that "
        f"{type(strategy).__name__} did not assemble"
    )
    if isinstance(strategy, _DECLARATION_VERIFIED_STRATEGIES):
        raise ValueError(msg)
    # Pre-declaration strategies: restoring KV without these pools (e.g. the
    # indexer's key buffers) corrupts sparse attention after a host hit.
    # Loud until the path is migrated.
    logger.error("%s; host restore of these pools is unsupported on this path", msg)


def _apply_stack_result(
    cache: UnifiedRadixCache,
    kvcache: Any,
    params: CacheInitParams,
    result: StackBuildResult,
) -> None:
    cache.host_pool_group = result.host_pool_group
    cache.cache_controller = result.cache_controller

    for ct, host_pool in result.component_host_pools.items():
        cache_attr, component_attr = _COMPONENT_HOST_ATTR[ct]
        setattr(cache, cache_attr, host_pool)
        setattr(cache.components[ct], component_attr, host_pool)

    for sidecar in result.sidecars:
        cache.register_sidecar_pool(sidecar)
        # EI: mark the sidecar's PoolEntry too (HostPoolGroup.allocator_entries
        # and anchor_of: peer-KV servicer, settle digest). Upstream reads
        # PoolEntry.indices_from_pool nowhere.
        entry = result.host_pool_group.entry_map.get(sidecar.pool_name)
        if entry is not None and entry.indices_from_pool is None:
            entry.indices_from_pool = sidecar.indices_from_pool

    kvcache.register_layer_transfer_counter(result.cache_controller.layer_done_counter)
    if isinstance(kvcache, PageInterleaveKVPoolMixin):
        # Packed draft KV and indexer loads finish before target layer zero is
        # ready. Their independent gather streams must consume that same event.
        for draft_pool in params.mtp_draft_device_pools:
            draft_pool.register_layer_transfer_counter(
                result.cache_controller.layer_done_counter
            )
    if result.register_req_to_token_counter:
        params.req_to_token_pool.register_layer_transfer_counter(
            result.cache_controller.layer_done_counter
        )

    logger.info(
        "Attached hybrid pool stack to UnifiedRadixCache: pools=%s, transfer_layer_id_max=%s",
        result.pools_desc,
        result.cache_controller.transfer_layer_id_max,
    )


def attach_hybrid_pool_to_unified_cache(
    cache: UnifiedRadixCache,
    params: CacheInitParams,
    server_args: ServerArgs,
    *,
    load_cache_event,
    storage_backend: Optional[str] = None,
    storage_extra_config: Optional[dict] = None,
    storage_prefetch_threshold: int = 256,
) -> None:
    """Attach HostPoolGroup + HybridCacheController to UnifiedRadixCache."""
    try:
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        components = set(cache.components.keys())
        strategy = _select_strategy(kvcache, components)
        result = strategy.build(
            cache=cache,
            kvcache=kvcache,
            params=params,
            server_args=server_args,
            load_cache_event=load_cache_event,
            storage_backend=storage_backend,
            storage_backend_extra_config=storage_extra_config,
            prefetch_threshold=storage_prefetch_threshold,
            model_name=get_serving().served_model_name,
            enable_storage_metrics=cache._enable_metrics_flag,
        )
        _check_declared_pools_present(kvcache=kvcache, result=result, strategy=strategy)
        _apply_stack_result(cache, kvcache, params, result)
    except Exception:
        logger.exception("attach_hybrid_pool_to_unified_cache failed")
        raise


def build_minimax_sparse_hicache_stack(
    *,
    params: CacheInitParams,
    sparse_pool: Any,
    load_cache_event,
    storage_backend: Optional[str],
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    """KV (main_pool) + INDEXER (index_k_pool) host stack for MiniMax M3 sparse."""
    # Mappings are stage-local keyed (controller iterates 0..transfer_layer_id_max).
    # PP>1 stays gated below pending end-to-end validation of the sparse host path.
    if params.pp_size > 1:
        raise NotImplementedError(
            "MiniMax-M3 sparse HiCache does not support pipeline parallelism "
            "(pp_size>1) yet."
        )
    # The Unified-tree strategy path does not reject value-bearing index layers.
    if sparse_pool.index_kv_pool is not None:
        raise ValueError(
            "MiniMax sparse HiCache currently supports index-k-only sparse layers; "
            "index_kv_pool (value-bearing) layers are not cached/restored yet."
        )
    main_pool = sparse_pool.main_pool
    from decoupled_engine.srt.mem_cache.memory_pool import MHATokenToKVPool
    from decoupled_engine.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4KVPool

    nvfp4 = isinstance(sparse_pool, MiniMaxM3Nvfp4KVPool)
    host_size = get_memory().hicache_size
    if nvfp4:
        from decoupled_engine.srt.mem_cache.pool_host.base import host_memory_budget_bytes
        from decoupled_engine.srt.mem_cache.pool_host.minimax_m3_nvfp4 import (
            MiniMaxM3Nvfp4IndexKPoolHost,
            MiniMaxM3Nvfp4MainKVPoolHost,
        )

        if get_memory().hicache_io_backend not in ("kernel", "direct"):
            raise ValueError("M3.1 NVFP4 HiCache requires kernel or direct IO")
        if get_memory().hicache_mem_layout not in (
            "layer_first",
            "page_first",
            "page_first_direct",
        ):
            raise ValueError("Unsupported M3.1 NVFP4 HiCache memory layout")
        index_pool = sparse_pool.index_k_pool
        main_bytes = sum(main_pool.get_kv_size_bytes()) / main_pool.total_slots
        index_bytes = sum(index_pool.get_kv_size_bytes()) / index_pool.total_slots
        draft_bytes = 0
        for draft_pool in params.sidecar_draft_device_pools:
            if (
                type(draft_pool) is not MHATokenToKVPool
                or draft_pool.size != main_pool.size
                or draft_pool.page_size != params.page_size
                or draft_pool.start_layer != 0
                or draft_pool.layer_num > main_pool.layer_num
            ):
                raise ValueError(
                    "M3.1 NVFP4 HiCache requires a dense MHA draft with matching "
                    "token capacity/page size and at most as many layers as the target"
                )
            draft_bytes += sum(draft_pool.get_kv_size_bytes()) / (
                draft_pool.size + draft_pool.page_size
            )
        total_bytes = main_bytes + index_bytes + draft_bytes
        # --hicache-size covers main KV, index payload/scales and draft sidecars.
        host_size *= main_bytes / total_bytes
        slots = (
            int(host_size * 1e9 // main_bytes)
            if host_size > 0
            else int(main_pool.size * get_memory().hicache_ratio)
        )
        slots = (slots // params.page_size + 1) * params.page_size
        if slots * total_bytes > host_memory_budget_bytes():
            raise ValueError(
                "Not enough host memory for M3.1 NVFP4 KV, index and draft caches"
            )
    start_layer = main_pool.start_layer
    transfer_layer_id_max = main_pool.layer_num
    # Stage-local keys (0..transfer_layer_id_max) match the controller's per-layer
    # load loop; values index the host pool's local layer buffer.
    full_layer_mapping = {
        layer_id: layer_id for layer_id in range(transfer_layer_id_max)
    }

    if nvfp4:
        kv_host_pool = MiniMaxM3Nvfp4MainKVPoolHost(
            main_pool.get_device_pool_infos()[0].buffer_info,
            get_memory().hicache_ratio,
            host_size,
            params.page_size,
            get_memory().hicache_mem_layout,
            allocator_type=_get_allocator_type(),
            device_capacity_tokens=main_pool.size,
            start_layer=main_pool.start_layer,
            end_layer=main_pool.end_layer,
        )
    else:
        kv_host_pool = build_kv_host_pool(
            kv_pool=main_pool,
            page_size=params.page_size,
            use_mla=False,
            host_size=host_size,
        )
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=main_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_id_max=transfer_layer_id_max,
            is_anchor=True,
        ),
    ]

    index_k_pool = sparse_pool.index_k_pool
    if index_k_pool is not None:
        if nvfp4:
            index_host_pool = MiniMaxM3Nvfp4IndexKPoolHost(
                index_k_pool.get_device_pool_infos()[0].buffer_info,
                kv_host_pool,
                get_memory().hicache_mem_layout,
                allocator_type=_get_allocator_type(),
                start_layer=index_k_pool.start_layer,
                end_layer=index_k_pool.end_layer,
            )
        else:
            index_host_pool = MHATokenToKOnlyPoolHost(
                index_k_pool,
                kv_host_pool,
                get_memory().hicache_mem_layout,
                allocator_type=_get_allocator_type(),
            )
        entries.append(
            build_pool_entry(
                name=PoolName.INDEXER,
                host_pool=index_host_pool,
                device_pool=index_k_pool,
                layer_mapping={
                    gid - start_layer: sub_id
                    for gid, sub_id in sparse_pool.index_k_layer_id_mapping.items()
                },
                transfer_layer_id_max=transfer_layer_id_max,
            )
        )

    host_pool_group = HostPoolGroup(entries)
    cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        params.page_size,
        params.tp_cache_group,
        load_cache_event=load_cache_event,
        attn_cp_group=params.attn_cp_cache_group,
        attn_tp_group=params.attn_tp_cache_group,
        write_policy=get_memory().hicache_write_policy,
        io_backend=get_memory().hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        pp_group=params.pp_cache_group,
        transfer_layer_id_max=transfer_layer_id_max,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller
