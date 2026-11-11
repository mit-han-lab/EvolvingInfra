# Adapted from SGLang python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py (commit 9a489f8d2f), modified for evolving-infra.
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from decoupled_engine.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    SidecarPoolSpec,
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
from decoupled_engine.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from decoupled_engine.srt.mem_cache.pool_host.common import get_allocator_type
from decoupled_engine.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from decoupled_engine.srt.mem_cache.pool_host.mamba import MambaPoolHost
from decoupled_engine.srt.mem_cache.pool_host.mha import (
    MHATokenToKOnlyPoolHost,
    get_mha_host_pool_cls,
)
from decoupled_engine.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from decoupled_engine.srt.mem_cache.unified_cache.component_type import ComponentType
from decoupled_engine.srt.runtime_context import (
    get_memory,
    get_parallel,
    get_serving,
)

if TYPE_CHECKING:
    import torch

    from decoupled_engine.srt.mem_cache.cache_init_params import CacheInitParams
    from decoupled_engine.srt.mem_cache.hiradix_cache import HiRadixCache
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
    transfer_layer_num: int,
) -> Callable[[int], Optional[int]]:
    def mapper(layer_id: int) -> Optional[int]:
        if not 0 <= layer_id < transfer_layer_num:
            return None
        return layer_mapping.get(layer_id)

    return mapper


def _with_mtp_layer_mapping(
    layer_mapping: dict[int, int],
    *,
    transfer_layer_start: int,
    target_device_layer_num: int,
    draft_layer_num: int,
) -> dict[int, int]:
    return layer_mapping | {
        transfer_layer_start + depth: target_device_layer_num + depth
        for depth in range(draft_layer_num)
    }


def build_kv_host_pool(
    *,
    kv_pool: Any,
    page_size: int,
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
    host_size: Optional[float] = None,
    mtp_draft_device_pools: tuple[Any, ...] = (),
    pool_label: str = "kv",
    host_dedup=None,
):
    kv_host_pool_cls = (
        MLATokenToKVPoolHost if use_mla else get_mha_host_pool_cls(kv_pool)
    )
    kwargs = {}
    if host_dedup is not None:
        kwargs["host_dedup"] = host_dedup
    if override_kv_cache_dim is not None:
        kwargs["override_kv_cache_dim"] = override_kv_cache_dim
    if mtp_draft_device_pools:
        kwargs["mtp_draft_device_pools"] = mtp_draft_device_pools
    parallel = get_parallel()
    if parallel.dcp_enabled:
        assert use_mla, (
            "HiCache + DCP is only wired for the MLA host pool; the MHA host "
            "pool has no DCP index translation."
        )
        kwargs["dcp_size"] = parallel.attn_dcp_size
        kwargs["dcp_rank"] = parallel.attn_dcp_rank
    return kv_host_pool_cls(
        kv_pool,
        get_memory().hicache_ratio,
        get_memory().hicache_size if host_size is None else host_size,
        page_size,
        get_memory().hicache_mem_layout,
        allocator_type=_get_allocator_type(),
        pool_label=pool_label,
        **kwargs,
    )


def _split_hicache_size(
    hicache_size: int, kv_pools: tuple[Any, ...]
) -> tuple[float, ...]:
    device_pool_sizes = []
    for kv_pool in kv_pools:
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
    transfer_layer_num: int,
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
        layer_mapper=_make_layer_mapper(layer_mapping, transfer_layer_num),
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
    from decoupled_engine.srt.server_args import get_global_server_args

    try:
        server_args = get_global_server_args()
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
    transfer_layer_num = len(full_layer_mapping)
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
        full_layer_mapping = _with_mtp_layer_mapping(
            full_layer_mapping,
            transfer_layer_start=transfer_layer_num,
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
                transfer_layer_num=transfer_layer_num + len(mtp_draft_device_pools),
                is_anchor=True,
                packed_draft_device_pools=mtp_draft_device_pools,
            )
        ]
    )
    host_pool_group.host_dedup = host_dedup
    return host_pool_group


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
    host_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    device_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    swa_attn_allocator: Any = None,
    mtp_swa_device_pools: tuple[Any, ...] = (),
) -> HostPoolGroup:
    """Anchor (full) + SWA host pool group for a hybrid-SWA device pool."""
    transfer_layer_num = len(full_layer_mapping | swa_layer_mapping)
    kv_host_pool = build_kv_host_pool(
        kv_pool=full_kv_pool,
        page_size=page_size,
        use_mla=use_mla,
        host_size=kv_host_size,
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
    if mtp_swa_device_pools:
        swa_layer_mapping = _with_mtp_layer_mapping(
            swa_layer_mapping,
            transfer_layer_start=transfer_layer_num,
            target_device_layer_num=swa_kv_pool.layer_num,
            draft_layer_num=len(mtp_swa_device_pools),
        )
    return HostPoolGroup(
        [
            build_pool_entry(
                name=PoolName.KV,
                host_pool=kv_host_pool,
                device_pool=full_kv_pool,
                layer_mapping=full_layer_mapping,
                transfer_layer_num=transfer_layer_num,
                is_anchor=True,
            ),
            build_pool_entry(
                name=PoolName.SWA,
                host_pool=swa_host_pool,
                device_pool=swa_kv_pool,
                layer_mapping=swa_layer_mapping,
                transfer_layer_num=transfer_layer_num + len(mtp_swa_device_pools),
                host_evict_fn=host_swa_evict_fn,
                device_evict_fn=device_swa_evict_fn,
                device_alloc_fn=(
                    swa_attn_allocator.alloc if swa_attn_allocator is not None else None
                ),
                device_free_fn=(
                    swa_attn_allocator.free if swa_attn_allocator is not None else None
                ),
                packed_draft_device_pools=mtp_swa_device_pools,
            ),
        ]
    )


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
    transfer_layer_num = len(full_layer_mapping)
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
        transfer_layer_num=transfer_layer_num,
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
    host_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    device_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_num = len(full_layer_mapping | swa_layer_mapping)
    # MTP draft pools follow the target SWA layout; select their SWA storage.
    mtp_swa_device_pools = tuple(
        pool.swa_kv_pool for pool in params.mtp_draft_device_pools
    )

    kv_host_size = swa_host_size = None
    if get_memory().hicache_size > 0:
        kv_host_size, swa_host_size = _split_hicache_size(
            get_memory().hicache_size, (full_kv_pool, swa_kv_pool)
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
        host_swa_evict_fn=host_swa_evict_fn,
        device_swa_evict_fn=device_swa_evict_fn,
        # For SWA hybrid, device allocation goes through the inner allocator.
        swa_attn_allocator=params.token_to_kv_pool_allocator.swa_attn_allocator,
        mtp_swa_device_pools=mtp_swa_device_pools,
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
        transfer_layer_num=transfer_layer_num,
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
) -> tuple[int, int]:
    """(KV, SWA) host pages. Window residency: from
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
        return (
            (host_tokens + page_size - 1) // page_size,
            (host_tokens + swa_page_size - 1) // swa_page_size,
        )
    ratio = get_memory().hicache_ratio
    full_host_pages = int(device_full_pages * ratio)
    swa_host_pages = int(device_swa_pages * ratio)
    return full_host_pages, swa_host_pages


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
        indexer = kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer
        total += len(indexer) * indexer[0].shape[1] * indexer[0].element_size()
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


def build_deepseek_v4_hicache_stack(
    *,
    params: CacheInitParams,
    kvcache: Any,
    load_cache_event,
    storage_backend: Optional[str],
    host_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    device_swa_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    from decoupled_engine.srt.mem_cache.dsv4_host_dedup import maybe_build_dsv4_host_dedup

    host_dedup = maybe_build_dsv4_host_dedup(
        params, kvcache, storage_backend, storage_backend_extra_config
    )
    page_size = params.page_size
    transfer_layer_num = kvcache.end_layer - kvcache.start_layer
    full_layer_mapping = {layer_id: layer_id for layer_id in range(transfer_layer_num)}

    is_unified_kv = getattr(kvcache, "_unified_kv", False)
    mtp_swa_device_buffers = []
    if is_unified_kv:
        # unified_kv keeps the SWA ring inside the unified pool and never offloads it,
        # so there is no separate SWA host pool to map.
        swa_layer_mapping = {}
    else:
        if len(kvcache.swa_kv_pool.kv_buffer) != transfer_layer_num:
            raise ValueError(
                "DeepSeek V4 SWA KV pool must be PP-stage-local: "
                f"got {len(kvcache.swa_kv_pool.kv_buffer)} buffers for "
                f"{transfer_layer_num} local layers"
            )
        swa_layer_mapping = {
            layer_id: layer_id for layer_id in range(transfer_layer_num)
        }
        # Keep every uncompressed draft SWA layer after the target SWA layers.
        # NextN has one layer per pool, while DSpark keeps all stages in one pool.
        mtp_swa_device_buffers = [
            buffer
            for pool in params.mtp_draft_device_pools
            for buffer in pool.swa_kv_pool.kv_buffer
        ]
        swa_layer_mapping = _with_mtp_layer_mapping(
            swa_layer_mapping,
            transfer_layer_start=transfer_layer_num,
            target_device_layer_num=transfer_layer_num,
            draft_layer_num=len(mtp_swa_device_buffers),
        )

    c4_layer_mapping = {}
    c128_layer_mapping = {}
    c4_state_local_layers = []
    c4_state_global_layers = []
    for local_layer_id, layer_item in enumerate(
        kvcache.layer_mapping[kvcache.start_layer : kvcache.end_layer]
    ):
        global_layer_id = kvcache.start_layer + local_layer_id
        if layer_item.compress_ratio == 4:
            c4_layer_mapping[local_layer_id] = layer_item.compress_layer_id
            c4_state_local_layers.append(local_layer_id)
            c4_state_global_layers.append(global_layer_id)
        elif layer_item.compress_ratio == 128:
            c128_layer_mapping[local_layer_id] = layer_item.compress_layer_id

    c4_state_mapping = {
        layer_id: local_id for local_id, layer_id in enumerate(c4_state_local_layers)
    }
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
    num_host_pages, swa_num_host_pages = _deepseek_v4_num_host_pages(
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
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
    ]

    if not is_unified_kv:
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
                transfer_layer_num=transfer_layer_num + len(mtp_swa_device_buffers),
                host_evict_fn=host_swa_evict_fn,
                device_evict_fn=device_swa_evict_fn,
                device_alloc_fn=swa_attn_allocator.alloc,
                device_free_fn=swa_attn_allocator.free,
                packed_draft_device_pools=tuple(mtp_swa_device_buffers),
            )
        )

    if c4_layer_mapping:
        c4_device_buffers, c4_item_bytes = _dsv4_compressed_region_buffers(kvcache, 4)
        c4_host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C4),
            device_buffers=c4_device_buffers,
            item_bytes=c4_item_bytes,
            num_host_pages=num_host_pages,
            slot_page_size=page_size,
            layout=get_memory().hicache_mem_layout,
            allocator_type=_get_allocator_type(),
            host_dedup=host_dedup,
        )
        c4_indexer_host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C4_INDEXER),
            device_buffers=kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer,
            item_bytes=(
                kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer[0].shape[1]
                * kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer[0].element_size()
            ),
            num_host_pages=num_host_pages,
            slot_page_size=page_size,
            layout=get_memory().hicache_mem_layout,
            allocator_type=_get_allocator_type(),
            host_dedup=host_dedup,
        )
        entries.extend(
            [
                build_pool_entry(
                    name=PoolName.DEEPSEEK_V4_C4,
                    host_pool=c4_host_pool,
                    device_pool=kvcache.c4_kv_pool,
                    layer_mapping=c4_layer_mapping,
                    transfer_layer_num=transfer_layer_num,
                    indices_from_pool=PoolName.KV,
                ),
                build_pool_entry(
                    name=PoolName.DEEPSEEK_V4_C4_INDEXER,
                    host_pool=c4_indexer_host_pool,
                    device_pool=kvcache.c4_indexer_kv_pool,
                    layer_mapping=c4_layer_mapping,
                    transfer_layer_num=transfer_layer_num,
                    indices_from_pool=PoolName.KV,
                ),
            ]
        )

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
                        transfer_layer_num=transfer_layer_num,
                        indices_from_pool=PoolName.SWA,
                    ),
                    build_pool_entry(
                        name=PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
                        host_pool=c4_indexer_state_host_pool,
                        device_pool=None,
                        layer_mapping=c4_state_mapping,
                        transfer_layer_num=transfer_layer_num,
                        indices_from_pool=PoolName.SWA,
                    ),
                ]
            )

    if c128_layer_mapping:
        c128_device_buffers, c128_item_bytes = _dsv4_compressed_region_buffers(
            kvcache, 128
        )
        c128_host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(PoolName.DEEPSEEK_V4_C128),
            device_buffers=c128_device_buffers,
            item_bytes=c128_item_bytes,
            num_host_pages=num_host_pages,
            slot_page_size=page_size,
            layout=get_memory().hicache_mem_layout,
            allocator_type=_get_allocator_type(),
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
                    transfer_layer_num=transfer_layer_num,
                    indices_from_pool=PoolName.KV,
                ),
            ]
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
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
        host_memory_mode=get_memory().hicache_host_memory_mode,
    )
    return host_pool_group, cache_controller


def build_hybrid_mamba_stack(
    *,
    params: CacheInitParams,
    kv_pool: Any,
    mamba_pool: Any,
    full_layer_mapping: dict[int, int],
    mamba_layer_mapping: dict[int, int],
    load_cache_event,
    storage_backend: Optional[str],
    use_mla: bool,
    host_mamba_evict_fn: Optional[Callable[[int], Any]] = None,
    device_mamba_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping)
    mamba_allocator = params.req_to_token_pool.mamba_allocator
    mtp_draft_device_pools = tuple(
        pool.full_kv_pool for pool in params.mtp_draft_device_pools
    )
    kv_host_size, mamba_host_size = None, 0
    if get_memory().hicache_size > 0:
        kv_host_size, mamba_host_size = _split_hicache_size(
            get_memory().hicache_size, (kv_pool, mamba_pool)
        )
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=params.page_size,
        use_mla=use_mla,
        host_size=kv_host_size,
        mtp_draft_device_pools=mtp_draft_device_pools,
    )
    if mtp_draft_device_pools:
        full_layer_mapping = _with_mtp_layer_mapping(
            full_layer_mapping,
            transfer_layer_start=transfer_layer_num,
            target_device_layer_num=kv_pool.layer_num,
            draft_layer_num=len(mtp_draft_device_pools),
        )
    mamba_host_pool = MambaPoolHost(
        mamba_pool,
        get_memory().hicache_ratio,
        mamba_host_size,
        allocator_type=_get_allocator_type(),
        layout=get_memory().hicache_mem_layout,
    )
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num + len(mtp_draft_device_pools),
            is_anchor=True,
            packed_draft_device_pools=mtp_draft_device_pools,
        ),
        build_pool_entry(
            name=PoolName.MAMBA,
            host_pool=mamba_host_pool,
            device_pool=mamba_pool,
            layer_mapping=mamba_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            host_evict_fn=host_mamba_evict_fn,
            device_evict_fn=device_mamba_evict_fn,
            device_alloc_fn=mamba_allocator.alloc,
            device_free_fn=mamba_allocator.free,
        ),
    ]
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
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
        host_memory_mode=get_memory().hicache_host_memory_mode,
    )
    return host_pool_group, cache_controller


def build_hybrid_mamba_swa_stack(
    *,
    params: CacheInitParams,
    full_kv_pool: Any,
    swa_kv_pool: Any,
    mamba_pool: Any,
    full_layer_mapping: dict[int, int],
    swa_layer_mapping: dict[int, int],
    mamba_layer_mapping: dict[int, int],
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
    transfer_layer_num = len(
        full_layer_mapping | swa_layer_mapping | mamba_layer_mapping
    )
    swa_attn_allocator = params.token_to_kv_pool_allocator.swa_attn_allocator
    mamba_allocator = params.req_to_token_pool.mamba_allocator
    kv_host_size, swa_host_size, mamba_host_size = None, None, 0
    if get_memory().hicache_size > 0:
        kv_host_size, swa_host_size, mamba_host_size = _split_hicache_size(
            get_memory().hicache_size, (full_kv_pool, swa_kv_pool, mamba_pool)
        )
    kv_host_pool = build_kv_host_pool(
        kv_pool=full_kv_pool,
        page_size=page_size,
        use_mla=False,
        host_size=kv_host_size,
        pool_label="full",
    )
    swa_host_pool = build_kv_host_pool(
        kv_pool=swa_kv_pool,
        page_size=page_size,
        use_mla=False,
        host_size=swa_host_size,
        pool_label="swa",
    )
    mamba_host_pool = MambaPoolHost(
        mamba_pool,
        get_memory().hicache_ratio,
        mamba_host_size,
        allocator_type=get_memory().hicache_storage_backend,
        layout=get_memory().hicache_mem_layout,
    )
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=full_kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
        build_pool_entry(
            name=PoolName.SWA,
            host_pool=swa_host_pool,
            device_pool=swa_kv_pool,
            layer_mapping=swa_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            host_evict_fn=host_swa_evict_fn,
            device_evict_fn=device_swa_evict_fn,
            device_alloc_fn=swa_attn_allocator.alloc,
            device_free_fn=swa_attn_allocator.free,
        ),
        build_pool_entry(
            name=PoolName.MAMBA,
            host_pool=mamba_host_pool,
            device_pool=mamba_pool,
            layer_mapping=mamba_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            host_evict_fn=host_mamba_evict_fn,
            device_evict_fn=device_mamba_evict_fn,
            device_alloc_fn=mamba_allocator.alloc,
            device_free_fn=mamba_allocator.free,
        ),
    ]
    host_pool_group = HostPoolGroup(entries)
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
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
        host_memory_mode=get_memory().hicache_host_memory_mode,
    )
    return host_pool_group, cache_controller


def build_anchor_sidecar_stack(
    *,
    params: CacheInitParams,
    kv_pool: Any,
    sidecar_pool_name: PoolName,
    full_layer_mapping: dict[int, int],
    load_cache_event,
    storage_backend: Optional[str],
    use_mla: bool,
    override_kv_cache_dim: Optional[int] = None,
    sidecar_host_pool_factory: Callable[[Any], Any],
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
    enable_peer_serving: bool = False,
) -> tuple[HostPoolGroup, HybridCacheController]:
    if enable_peer_serving and get_memory().hicache_mem_layout not in (
        "page_first",
        "page_first_direct",
    ):
        raise ValueError(
            "DSA peer KV requires a page_first or page_first_direct host layout"
        )
    from decoupled_engine.srt.mem_cache.dsa_host_dedup import maybe_build_dsa_host_dedup

    host_dedup = maybe_build_dsa_host_dedup(
        params, kv_pool, storage_backend, storage_backend_extra_config
    )
    transfer_layer_num = len(full_layer_mapping)
    mtp_draft_device_pools = tuple(
        pool for pool in params.mtp_draft_device_pools if pool.index_k_with_scale_buffer
    )
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=params.page_size,
        use_mla=use_mla,
        override_kv_cache_dim=override_kv_cache_dim,
        mtp_draft_device_pools=mtp_draft_device_pools,
        host_dedup=host_dedup,
    )
    sidecar_host_pool = sidecar_host_pool_factory(kv_host_pool)
    if enable_peer_serving:
        kv_host_pool.enable_peer_serving(
            reserve_pages=_peer_kv_reserve_pages(kv_host_pool.page_num, "kv")
        )
    # Expose packed MTP tail layers to the controller's flat transfer builder.
    if mtp_draft_device_pools:
        full_layer_mapping = _with_mtp_layer_mapping(
            full_layer_mapping,
            transfer_layer_start=transfer_layer_num,
            target_device_layer_num=kv_pool.layer_num,
            draft_layer_num=len(mtp_draft_device_pools),
        )
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num + len(mtp_draft_device_pools),
            is_anchor=True,
            packed_draft_device_pools=mtp_draft_device_pools,
        ),
        build_pool_entry(
            name=sidecar_pool_name,
            host_pool=sidecar_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num + len(mtp_draft_device_pools),
            packed_draft_device_pools=mtp_draft_device_pools,
            indices_from_pool=PoolName.KV,
        ),
    ]
    host_pool_group = HostPoolGroup(entries)
    host_pool_group.host_dedup = host_dedup
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
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
        host_memory_mode=get_memory().hicache_host_memory_mode,
    )
    return host_pool_group, cache_controller


def _build_mha_mla_host_pool(
    *,
    pool: Any,
    host_to_device_ratio: float,
    page_size: int,
    layout: str,
    allocator_type: str,
    pool_label: str,
    host_dedup=None,
):
    from decoupled_engine.srt.mem_cache.memory_pool import MHATokenToKVPool

    kwargs = dict(
        host_to_device_ratio=host_to_device_ratio,
        host_size=0,
        page_size=page_size,
        layout=layout,
        allocator_type=allocator_type,
        pool_label=pool_label,
    )
    if isinstance(pool, MHATokenToKVPool):
        if host_dedup is not None:
            raise ValueError("Host dedup requires replicated MLA draft KV")
        return get_mha_host_pool_cls(pool)(pool, **kwargs)
    return MLATokenToKVPoolHost(
        pool,
        override_kv_cache_dim=pool.kv_cache_dim,
        host_dedup=host_dedup,
        **kwargs,
    )


def build_full_draft_pools(
    *,
    draft_kv_pool: Any,
    tree_cache: Any,
) -> tuple[list[SidecarPoolSpec], list[PoolEntry]]:
    """Build draft KV/DSA sidecars whose indices follow target full KV."""
    from decoupled_engine.srt.mem_cache.memory_pool import (
        DSATokenToKVPool,
        HybridLinearKVPool,
    )

    pool = draft_kv_pool
    if isinstance(pool, HybridLinearKVPool):
        # Hybrid draft runners keep their sole attention layer in this sub-pool.
        pool = pool.full_kv_pool
    if pool.layer_num == 0:
        return [], []

    controller = tree_cache.cache_controller
    host_pool_group = controller.mem_pool_host

    # Note(kpham-sgl): DCP x DSpark draft KV is replicated and spans the virtual
    # loc space, so match the target host's logical_size instead of physical size.
    draft_host_pool = _build_mha_mla_host_pool(
        pool=pool,
        host_to_device_ratio=host_pool_group.logical_size / pool.size,
        page_size=controller.page_size,
        layout=get_memory().hicache_mem_layout,
        allocator_type=_get_allocator_type(),
        pool_label="draft",
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
            transfer_layer_num=draft_host_pool.layer_num,
            indices_from_pool=PoolName.KV,
        )
    ]

    if isinstance(pool, DSATokenToKVPool) and pool.index_k_with_scale_buffer:
        indexer_host_pool = DSAIndexerPoolHost(
            pool,
            draft_host_pool,
            get_memory().hicache_mem_layout,
            allocator_type=_get_allocator_type(),
        )
        specs.append(
            SidecarPoolSpec(
                pool_name=PoolName.DRAFT_INDEXER,
                indices_from_pool=PoolName.KV,
            )
        )
        entries.append(
            build_pool_entry(
                name=PoolName.DRAFT_INDEXER,
                host_pool=indexer_host_pool,
                device_pool=pool,
                layer_mapping=draft_layer_mapping,
                transfer_layer_num=indexer_host_pool.layer_num,
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
        transfer_layer_num=host_pool.layer_num,
    )
    return [spec], [entry]


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
    transfer_layer_num: int = 0
    pools_desc: str = ""


class StackStrategy:
    def matches(self, kvcache: Any, components: set[ComponentType]) -> bool:
        raise NotImplementedError

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


class _DeepSeekV4Strategy(StackStrategy):
    def matches(self, kvcache, components):
        from decoupled_engine.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )

        return isinstance(kvcache, DeepSeekV4TokenToKVPool) and components == {
            ComponentType.FULL,
            ComponentType.SWA,
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
        host_pool_group, cache_controller = build_deepseek_v4_hicache_stack(
            params=params,
            kvcache=kvcache,
            load_cache_event=load_cache_event,
            storage_backend=storage_backend,
            host_swa_evict_fn=lambda n: cache.evict_host(n, ComponentType.SWA),
            device_swa_evict_fn=lambda n: _evict_swa_for_device_alloc(cache, n),
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
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
            for name, src in (
                (PoolName.DEEPSEEK_V4_C4, PoolName.KV),
                (PoolName.DEEPSEEK_V4_C4_INDEXER, PoolName.KV),
                (PoolName.DEEPSEEK_V4_C128, PoolName.KV),
                (PoolName.DEEPSEEK_V4_C4_STATE, PoolName.SWA),
                (PoolName.DEEPSEEK_V4_C4_INDEXER_STATE, PoolName.SWA),
                (PoolName.DEEPSEEK_V4_C128_STATE, PoolName.SWA),
            )
            if name in host_pool_group.entry_map
        ]
        component_host_pools = {
            ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
        }
        if PoolName.SWA in host_pool_group.entry_map:
            component_host_pools[ComponentType.SWA] = host_pool_group.get_pool(
                PoolName.SWA
            )

        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools=component_host_pools,
            sidecars=sidecars,
            transfer_layer_num=kvcache.end_layer - kvcache.start_layer,
            pools_desc="KV + SWA + DeepSeekV4 sidecars",
        )


class _MambaStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from decoupled_engine.srt.mem_cache.memory_pool import HybridLinearKVPool

        return isinstance(kvcache, HybridLinearKVPool) and components == {
            ComponentType.FULL,
            ComponentType.MAMBA,
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
        full_layer_mapping = dict(kvcache.full_attention_layer_id_mapping)
        mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)
        host_pool_group, cache_controller = build_hybrid_mamba_stack(
            params=params,
            kv_pool=kvcache.full_kv_pool,
            mamba_pool=params.req_to_token_pool.mamba_pool,
            full_layer_mapping=full_layer_mapping,
            mamba_layer_mapping=mamba_layer_mapping,
            load_cache_event=load_cache_event,
            storage_backend=storage_backend,
            use_mla=kvcache.use_mla,
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
            register_req_to_token_counter=True,
            transfer_layer_num=len(full_layer_mapping | mamba_layer_mapping),
            pools_desc="KV + MAMBA",
        )


def _swa_layer_mappings(kvcache) -> tuple[dict[int, int], dict[int, int]]:
    full = {
        gid: lid for gid, (lid, is_swa) in kvcache.layers_mapping.items() if not is_swa
    }
    swa = {gid: lid for gid, (lid, is_swa) in kvcache.layers_mapping.items() if is_swa}
    return full, swa


class _SwaStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from decoupled_engine.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
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
            transfer_layer_num=len(full_layer_mapping | swa_layer_mapping),
            pools_desc="Full + SWA",
        )


class _MambaSwaStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from decoupled_engine.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
        from decoupled_engine.srt.mem_cache.swa_memory_pool import SWAKVPool

        return (
            isinstance(kvcache, SWAKVPool)
            and not isinstance(kvcache, DeepSeekV4TokenToKVPool)
            and components
            == {ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA}
        )

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
        mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)
        host_pool_group, cache_controller = build_hybrid_mamba_swa_stack(
            params=params,
            full_kv_pool=kvcache.full_kv_pool,
            swa_kv_pool=kvcache.swa_kv_pool,
            mamba_pool=params.req_to_token_pool.mamba_pool,
            full_layer_mapping=full_layer_mapping,
            swa_layer_mapping=swa_layer_mapping,
            mamba_layer_mapping=mamba_layer_mapping,
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
            transfer_layer_num=len(
                full_layer_mapping | swa_layer_mapping | mamba_layer_mapping
            ),
            pools_desc="KV + SWA + MAMBA",
        )


class _DsaStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from decoupled_engine.srt.mem_cache.memory_pool import DSATokenToKVPool

        return isinstance(kvcache, DSATokenToKVPool) and components == {
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
        from decoupled_engine.srt.mem_cache.memory_pool import MLATokenToKVPool

        full_kv_pool = kvcache
        use_mla = isinstance(kvcache, MLATokenToKVPool)
        full_layer_mapping = {i: i for i in range(full_kv_pool.layer_num)}
        host_pool_group, cache_controller = build_anchor_sidecar_stack(
            params=params,
            kv_pool=full_kv_pool,
            sidecar_pool_name=PoolName.INDEXER,
            full_layer_mapping=full_layer_mapping,
            load_cache_event=load_cache_event,
            storage_backend=storage_backend,
            use_mla=use_mla,
            override_kv_cache_dim=full_kv_pool.kv_cache_dim,
            sidecar_host_pool_factory=lambda kv_host_pool: DSAIndexerPoolHost(
                full_kv_pool,
                kv_host_pool,
                get_memory().hicache_mem_layout,
                allocator_type=_get_allocator_type(),
            ),
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
            enable_peer_serving=_resolved(server_args, "enable_peer_kv_servicer"),
        )
        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
            },
            sidecars=[
                SidecarPoolSpec(
                    pool_name=PoolName.INDEXER,
                    indices_from_pool=PoolName.KV,
                ),
            ],
            transfer_layer_num=len(full_layer_mapping),
            pools_desc="KV + INDEXER",
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
            transfer_layer_num=kvcache.main_pool.layer_num,
            pools_desc=pools_desc,
        )


class _PlainKvStrategy(StackStrategy):
    def matches(self, kvcache, components):
        from decoupled_engine.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
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
            transfer_layer_num=len(full_layer_mapping),
            pools_desc="KV",
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

    kvcache.register_layer_transfer_counter(result.cache_controller.layer_done_counter)
    if result.register_req_to_token_counter:
        params.req_to_token_pool.register_layer_transfer_counter(
            result.cache_controller.layer_done_counter
        )

    logger.info(
        "Attached hybrid pool stack to UnifiedRadixCache: pools=%s, transfer_layer_num=%s",
        result.pools_desc,
        result.transfer_layer_num,
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
    # Mappings are stage-local keyed (controller iterates 0..transfer_layer_num).
    # PP>1 stays gated below pending end-to-end validation of the sparse host path.
    if params.pp_size > 1:
        raise NotImplementedError(
            "MiniMax-M3 sparse HiCache does not support pipeline parallelism "
            "(pp_size>1) yet."
        )
    # mirror HiRadix's guard, which the Unified-tree strategy path otherwise skips.
    if sparse_pool.index_kv_pool is not None:
        raise ValueError(
            "MiniMax sparse HiCache currently supports index-k-only sparse layers; "
            "index_kv_pool (value-bearing) layers are not cached/restored yet."
        )
    main_pool = sparse_pool.main_pool
    start_layer = main_pool.start_layer
    transfer_layer_num = main_pool.layer_num
    # Stage-local keys (0..transfer_layer_num) match the controller's per-layer
    # load loop; values index the host pool's local layer buffer.
    full_layer_mapping = {layer_id: layer_id for layer_id in range(transfer_layer_num)}

    kv_host_pool = build_kv_host_pool(
        kv_pool=main_pool,
        page_size=params.page_size,
        use_mla=False,
    )
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=main_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
    ]

    index_k_pool = sparse_pool.index_k_pool
    if index_k_pool is not None:
        index_host_pool = MHATokenToKOnlyPoolHost(
            index_k_pool,
            kv_host_pool,
            get_memory().hicache_mem_layout,
            allocator_type=get_memory().hicache_storage_backend,
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
                transfer_layer_num=transfer_layer_num,
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
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_pool_group, cache_controller


def attach_hybrid_minimax_sparse_pool_to_hiradix_cache(
    radix_cache: HiRadixCache,
    params: CacheInitParams,
    *,
    extra_config: dict,
    prefetch_threshold: int,
    enable_storage_metrics: bool,
    load_cache_event,
) -> None:
    """Attach HostPoolGroup (KV + index K) + HybridCacheController for HiRadixCache."""
    from decoupled_engine.srt.mem_cache.memory_pool import MiniMaxSparseKVPool

    try:
        sparse_pool = radix_cache.kv_cache
        if not isinstance(sparse_pool, MiniMaxSparseKVPool):
            raise TypeError(
                f"Expected MiniMaxSparseKVPool, got {type(sparse_pool).__name__}"
            )
        if sparse_pool.index_kv_pool is not None:
            raise ValueError(
                "MiniMax M3 HiCache L2 currently supports index-k-only sparse layers "
                "(sparse_disable_index_value=1 for all sparse layers). "
                "This model has index_kv_pool layers; INDEXER_KV sidecar is not "
                "implemented yet."
            )

        main_pool = sparse_pool.main_pool
        if sparse_pool.index_k_pool is None:
            host_pool_group, cache_controller = build_kv_only_stack(
                params=params,
                kv_pool=main_pool,
                full_layer_mapping={
                    layer_id: layer_id for layer_id in range(main_pool.layer_num)
                },
                load_cache_event=load_cache_event,
                storage_backend=get_memory().hicache_storage_backend,
                use_mla=False,
                prefetch_threshold=prefetch_threshold,
                model_name=get_serving().served_model_name,
                storage_backend_extra_config=extra_config,
                enable_storage_metrics=enable_storage_metrics,
            )
            pools_desc = "KV"
        else:
            host_pool_group, cache_controller = build_minimax_sparse_hicache_stack(
                params=params,
                sparse_pool=sparse_pool,
                load_cache_event=load_cache_event,
                storage_backend=get_memory().hicache_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=get_serving().served_model_name,
                storage_backend_extra_config=extra_config,
                enable_storage_metrics=enable_storage_metrics,
            )
            pools_desc = "KV + INDEXER(k-only)"

        sparse_pool.register_layer_transfer_counter(cache_controller.layer_done_counter)
        radix_cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
        radix_cache.token_to_kv_pool_host = host_pool_group
        radix_cache.cache_controller = cache_controller
        logger.info(
            "Attached hybrid MiniMax sparse pool stack to HiRadixCache: pools=%s, "
            "transfer_layer_num=%s, sparse_index_k_layers=%s",
            pools_desc,
            main_pool.layer_num,
            len(sparse_pool.index_k_layer_id_mapping),
        )
    except Exception:
        logger.exception("attach_hybrid_minimax_sparse_pool_to_hiradix_cache failed")
        raise


def attach_hybrid_dsa_pool_to_hiradix_cache(
    radix_cache: HiRadixCache,
    params: CacheInitParams,
    *,
    extra_config: dict,
    prefetch_threshold: int,
    enable_storage_metrics: bool,
    load_cache_event,
) -> None:
    """Attach HostPoolGroup (KV + indexer) + HybridCacheController for HiRadixCache.

    This entrypoint is currently intended only for HiRadixCache's DSA path.
    """
    try:
        kv = radix_cache.kv_cache
        layer_mapping = {layer_id: layer_id for layer_id in range(kv.layer_num)}
        host_pool_group, cache_controller = build_anchor_sidecar_stack(
            params=params,
            kv_pool=kv,
            sidecar_pool_name=PoolName.INDEXER,
            full_layer_mapping=layer_mapping,
            load_cache_event=load_cache_event,
            storage_backend=get_memory().hicache_storage_backend,
            use_mla=True,
            override_kv_cache_dim=kv.kv_cache_dim,
            prefetch_threshold=prefetch_threshold,
            sidecar_host_pool_factory=lambda kv_host_pool: DSAIndexerPoolHost(
                kv,
                kv_host_pool,
                get_memory().hicache_mem_layout,
                allocator_type=_get_allocator_type(),
            ),
            model_name=get_serving().served_model_name,
            storage_backend_extra_config=extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        radix_cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)
        radix_cache.token_to_kv_pool_host = host_pool_group
        radix_cache.cache_controller = cache_controller
        logger.info(
            "Attached hybrid DSA pool stack to HiRadixCache: pools=KV + INDEXER, "
            "transfer_layer_num=%s",
            len(layer_mapping),
        )
    except Exception:
        logger.exception("attach_hybrid_dsa_pool_to_hiradix_cache failed")
        raise
