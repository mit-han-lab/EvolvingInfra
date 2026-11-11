# Adapted from SGLang python/sglang/srt/mem_cache/kv_cache_builder.py (commit 9a489f8d2f), modified for evolving-infra.
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True, kw_only=True)
class KVCacheBuildResult:
    is_hybrid_swa: bool
    is_hybrid_ssm: bool
    sliding_window_size: Optional[int]
    full_tokens_per_layer: Optional[int]
    swa_tokens_per_layer: Optional[int]
    req_to_token_pool: object
    token_to_kv_pool_allocator: object
    disable_radix_cache: bool
    tree_cache: object


from typing import TYPE_CHECKING

from decoupled_engine.srt.arg_groups.overrides import resolving_view
from decoupled_engine.srt.configs.hybrid_arch import (
    hybrid_gdn_config,
    hybrid_lightning_config,
    kimi_linear_config,
    linear_attn_model_spec,
    mamba2_config,
)
from decoupled_engine.srt.configs.model_config import ModelImpl, is_deepseek_dsa
from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.hardware_backend.mlx.runtime import use_mlx
from decoupled_engine.srt.managers.mm_schedule import init_mm_embedding_cache
from decoupled_engine.srt.mem_cache.cache_init_params import CacheInitParams
from decoupled_engine.srt.mem_cache.memory_pool import MHATokenToKVPool
from decoupled_engine.srt.mem_cache.registry import TreeCacheBuildContext, create_tree_cache
from decoupled_engine.srt.mem_cache.swa_memory_pool import SWAKVPool
from decoupled_engine.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from decoupled_engine.srt.model_loader.utils import get_resolved_model_impl
from decoupled_engine.srt.runtime_context import (
    get_context,
    get_disagg,
    get_memory,
    get_observability,
    get_parallel,
    get_schedule,
)
from decoupled_engine.srt.utils import is_hip

if TYPE_CHECKING:

    from torch.distributed import ProcessGroup

    from decoupled_engine.srt.configs.model_config import ModelConfig
    from decoupled_engine.srt.distributed.parallel_state import GroupCoordinator
    from decoupled_engine.srt.distributed.parallel_state_wrapper import ParallelState
    from decoupled_engine.srt.managers.tp_worker import BaseTpWorker
    from decoupled_engine.srt.server_args import ServerArgs
    from decoupled_engine.srt.speculative.base_spec_worker import HiCacheDraftPlan
    from decoupled_engine.srt.speculative.spec_info import SpeculativeAlgorithm


def get_draft_kv_pool(
    *,
    draft_worker: BaseTpWorker,
    spec_algorithm: SpeculativeAlgorithm,
    server_args: ServerArgs,
):
    """Return the draft token-to-KV pool for the current draft worker,
    or None when no draft KV pool is available."""
    if draft_worker is None or spec_algorithm.is_ngram():
        return None

    # V2 draft workers exist only on their hosting PP stage; other ranks own no
    # nested draft worker or draft KV pool.
    if draft_worker.draft_worker is None:
        return None

    if resolving_view(server_args).enable_multi_layer_eagle:
        draft_runner = draft_worker.draft_worker.draft_runner_list[0]
    else:
        draft_runner = draft_worker.draft_worker.draft_runner
    return draft_runner.token_to_kv_pool


def maybe_register_hicache_draft(
    *,
    tree_cache,
    draft_plan: HiCacheDraftPlan,
) -> None:
    from decoupled_engine.srt.speculative.base_spec_worker import HiCacheDraftMode

    if draft_plan.mode != HiCacheDraftMode.SIDECAR:
        return

    from decoupled_engine.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    if not isinstance(tree_cache, UnifiedRadixCache):
        raise NotImplementedError("HiCache draft pools require UnifiedRadixCache.")

    from decoupled_engine.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        build_hicache_draft_sidecars,
    )

    specs, entries = build_hicache_draft_sidecars(
        draft_device_pools=draft_plan.device_pools,
        tree_cache=tree_cache,
    )
    for spec, entry in zip(specs, entries, strict=True):
        tree_cache.register_sidecar_pool(spec, entry)


# Host slots a backup-only retraction pool gets, as a fraction of the device
# pool. Sized well under 1.0 because a retraction burst touches a fraction of
# the device tokens; overflow aborts the request rather than pre-reserving.
BACKUP_ONLY_HICACHE_RATIO = 0.2


def uses_ssm_state(model_config) -> bool:
    """Whether the model keeps recurrent/conv state alongside its attention KV."""
    spec = linear_attn_model_spec(model_config)
    return (
        hybrid_gdn_config(model_config) is not None
        or mamba2_config(model_config) is not None
        or (spec.uses_mamba_radix_cache if spec is not None else False)
        or kimi_linear_config(model_config) is not None
        or hybrid_lightning_config(model_config) is not None
    )


def _validate_decode_radix_cache_compatibility(
    *,
    mode: str,
    device: Optional[str],
    model_config: ModelConfig,
    spec_algorithm: SpeculativeAlgorithm,
    is_hybrid_swa: bool,
    is_hybrid_ssm: bool,
    enable_hierarchical_cache: bool,
) -> None:
    """Fail closed before constructing an unsupported decode Radix cache."""
    if mode == "window_host" and not enable_hierarchical_cache:
        # DSv4 window SWA residency keeps the SWA the device drops on the
        # host tier.
        raise ValueError(
            "--disaggregation-decode-radix-cache-mode window_host requires "
            "--enable-hierarchical-cache."
        )
    # window_host has content_only's scope (DSv4 hybrid SWA, DSPARK, CUDA
    # physical KV) and its fresh-SWA-window admission; it additionally keeps
    # host SWA for every inserted page (checked after construction).
    content_only = mode in ("content_only", "window_host")
    is_deepseek_v4 = getattr(model_config, "is_deepseek_v4_arch", False)

    if not content_only and spec_algorithm.is_eagle():
        # A HiCache host tier is checked after construction
        # (_validate_decode_eagle_hicache_draft): it must pack the MTP draft.
        if (
            model_config.hf_config.architectures
            not in (
                ["GlmMoeDsaForCausalLM"],
                ["KimiK25ForConditionalGeneration"],
                ["Glm5vForConditionalGeneration"],
            )
            or str(device).split(":", 1)[0] != "cuda"
            or is_hybrid_swa
            or is_hybrid_ssm
        ):
            raise ValueError(
                "PD decode EAGLE radix cache currently supports only CUDA "
                "GLM DSA or Kimi MLA with target and draft KV at shared indices."
            )

    if content_only:
        if not (is_deepseek_v4 and is_hybrid_swa):
            raise ValueError(
                f"--disaggregation-decode-radix-cache-mode {mode} "
                "currently supports only DeepSeek-V4's hybrid SWA layout."
            )
        if not spec_algorithm.is_dspark():
            raise ValueError(
                f"--disaggregation-decode-radix-cache-mode {mode} "
                "currently requires --speculative-algorithm DSPARK."
            )
        if str(device).split(":", 1)[0] != "cuda":
            raise ValueError(
                f"--disaggregation-decode-radix-cache-mode {mode} "
                "currently supports only the CUDA non-unified physical KV path."
            )

        from decoupled_engine.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        if is_unified_kv_triton():
            raise ValueError(
                f"--disaggregation-decode-radix-cache-mode {mode} does "
                "not support the physical unified-KV layout."
            )

    # Decode-side radix cache supports SWA only through the unified tree, whose
    # component pools preserve the full-attention prefix while transferring the
    # SWA window fresh. The legacy SWA cache and hybrid SSM pools remain
    # incompatible with the prefix-match-and-lock allocation path.
    if is_hybrid_swa:
        # The current registry already selects UnifiedRadixCache for the
        # DSV4 content-only path, and the selected implementation/capabilities
        # are checked below after construction.  Keep the legacy opt-in guard
        # for full-state hybrid-SWA users, but do not require the deprecated
        # environment switch for content_only.
        if not content_only and not (
            envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.get() or use_mlx()
        ):
            raise ValueError(
                "--disaggregation-decode-enable-radix-cache with sliding "
                "window attention (SWA) models requires the unified radix "
                "tree (set SGLANG_ENABLE_UNIFIED_RADIX_TREE=1)."
            )
        if enable_hierarchical_cache and not content_only:
            raise ValueError(
                "--disaggregation-decode-enable-radix-cache with sliding "
                "window attention (SWA) models currently supports only "
                "device-resident cache and is incompatible with "
                "--enable-hierarchical-cache."
            )
        if is_deepseek_v4 and not content_only:
            raise ValueError(
                "--disaggregation-decode-enable-radix-cache does not support "
                "DeepSeek-V4 (DSA) compressed KV (c4/c128/indexer) yet."
            )
        if getattr(model_config, "is_hybrid_swa_compress", False):
            raise ValueError(
                "--disaggregation-decode-enable-radix-cache does not support "
                "SWA-compress models (e.g. Gemma4 / MiMo-V2) yet."
            )
    if is_hybrid_ssm:
        raise ValueError(
            "--disaggregation-decode-enable-radix-cache is incompatible "
            "with Mamba/SSM models"
        )


def _validate_decode_radix_tree_cache(
    *, mode: str, tree_cache: object, token_to_kv_pool: object
) -> None:
    """Validate the cache selected by the registry, not only its input flags."""
    if mode not in ("content_only", "window_host"):
        return
    flag = f"--disaggregation-decode-radix-cache-mode {mode}"
    if not isinstance(tree_cache, UnifiedRadixCache):
        raise ValueError(
            f"{flag} requires "
            "UnifiedRadixCache after backend selection; the selected cache is "
            f"{type(tree_cache).__name__}. Disable custom/CPP/external radix "
            "backends."
        )
    from decoupled_engine.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4TokenToKVPool,
    )
    from decoupled_engine.srt.mem_cache.unified_cache.components import ComponentType

    if not isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool):
        raise ValueError(
            f"{flag} requires "
            "the DeepSeekV4TokenToKVPool physical layout after allocation; "
            f"the selected pool is {type(token_to_kv_pool).__name__}."
        )
    tree_page_size = getattr(tree_cache, "page_size", None)
    pool_page_size = getattr(token_to_kv_pool, "page_size", None)
    if (
        not isinstance(tree_page_size, int)
        or not isinstance(pool_page_size, int)
        or tree_page_size <= 0
        or pool_page_size <= 0
        or tree_page_size != pool_page_size
        or tree_page_size % 128 != 0
    ):
        raise ValueError(
            f"{flag} requires "
            "matching tree/pool page sizes divisible by DeepSeek-V4's c4 and "
            "c128 compression ratios; got "
            f"tree_page_size={tree_page_size!r}, pool_page_size={pool_page_size!r}."
        )
    expected_components = frozenset((ComponentType.FULL, ComponentType.SWA))
    if (
        len(tree_cache.tree_components) != len(expected_components)
        or frozenset(tree_cache.tree_components) != expected_components
    ):
        names = ", ".join(component.name for component in tree_cache.tree_components)
        raise ValueError(
            f"{flag} requires "
            "exactly the FULL and SWA logical tree components; the selected "
            f"cache has [{names}]."
        )
    if mode == "window_host":
        # FULL-only admission match and lock; every insert is all-component,
        # so only the match capability is needed.
        if getattr(tree_cache, "dsv4_uniform_swa_residency", "full") != "window":
            raise ValueError(
                f"{flag} requires a cache built with "
                "--dsv4-uniform-cache-lifecycle --dsv4-uniform-swa-residency "
                "window."
            )
        if not getattr(tree_cache.tree_core, "supports_partial_component_match", False):
            raise ValueError(
                f"{flag} requires a unified tree core that supports "
                "partial-component (FULL-only) match; the selected core is "
                f"{type(tree_cache.tree_core).__name__}."
            )
        return
    if not getattr(tree_cache.tree_core, "supports_partial_component_cache", False):
        raise ValueError(
            f"{flag} requires "
            "a unified tree core that supports partial-component match and "
            f"insert; the selected core is {type(tree_cache.tree_core).__name__}."
        )


def _validate_decode_eagle_hicache_draft(
    *, tree_cache: object, draft_plan: Optional[HiCacheDraftPlan]
) -> None:
    """PD decode EAGLE radix + HiCache: a host restore must bring back the MTP
    draft rows with the target's, in the KV and in the DSA indexer pool.

    The packed MTP host layout stores the draft layers after the target layers
    of both host pools at the same slots, so the D->H backup and the H->D
    restore (``HybridCacheController._l2_load_transfers``) move all four in one
    op. Fail closed unless every draft device pool is packed into both."""
    from decoupled_engine.srt.mem_cache.hicache_storage import PoolName
    from decoupled_engine.srt.mem_cache.memory_pool import DSATokenToKVPool
    from decoupled_engine.srt.speculative.base_spec_worker import HiCacheDraftMode

    mode = getattr(draft_plan, "mode", HiCacheDraftMode.NONE)
    draft_pools = tuple(getattr(draft_plan, "device_pools", ()) or ())
    prefix = "PD decode EAGLE radix cache with --enable-hierarchical-cache"
    if mode == HiCacheDraftMode.SIDECAR and not any(
        isinstance(pool, DSATokenToKVPool) for pool in draft_pools
    ):
        from decoupled_engine.srt.disaggregation.native_eagle3_cache_layout import (
            validate_native_eagle3_cache,
        )

        group = tree_cache.cache_controller.mem_pool_host
        anchors = [
            entry for entry in group.entry_map.values() if entry.is_primary_index_anchor
        ]
        if len(anchors) != 1 or len(draft_pools) != 1:
            raise ValueError(f"{prefix} requires one flat MLA target and draft")
        validate_native_eagle3_cache(
            tree_cache, anchors[0].device_pool, draft_pools[0], host_enabled=True
        )
        return
    if mode != HiCacheDraftMode.PACKED or not draft_pools:
        raise ValueError(
            f"{prefix} requires the packed MTP HiCache draft layout (a NextN/MTP "
            f"draft); got draft plan {getattr(mode, 'value', mode)!r}. A "
            "target-only restore would leave the draft prefix stale."
        )
    host_group = getattr(
        getattr(tree_cache, "cache_controller", None), "mem_pool_host", None
    )
    entry_map = getattr(host_group, "entry_map", None) or {}
    for name in (PoolName.KV, PoolName.INDEXER):
        entry = entry_map.get(name)
        packed = tuple(getattr(entry, "packed_draft_device_pools", ()) or ())
        host_pool = getattr(entry, "host_pool", None)
        target_layers = getattr(host_pool, "target_layer_num", None)
        if (
            entry is None
            or len(packed) != len(draft_pools)
            or any(a is not b for a, b in zip(packed, draft_pools))
            or target_layers is None
            or getattr(host_pool, "layer_num", None) != target_layers + len(draft_pools)
        ):
            raise ValueError(
                f"{prefix} requires every MTP draft pool packed into the "
                f"'{name.value}' host pool (target + draft layers at one slot); "
                f"packed {len(packed)} of {len(draft_pools)} draft pools. A "
                "restore would bring back target rows without the draft's."
            )


def resolve_decode_retraction_backup(*, tp_worker: BaseTpWorker) -> str:
    """Resolve the retraction backend onto the config bags and return it.

    The backend needs the built KV pool, so it cannot resolve in
    ``ServerArgs.__post_init__``; it lands on the bags via ``override`` and
    every reader goes through ``get_disagg()`` / ``get_memory()``.
    """
    disagg = get_disagg()
    memory = get_memory()
    fields = {}

    backend = disagg.disaggregation_decode_retraction_backup
    if backend is None:
        kv_cache = tp_worker.get_memory_pool()[1].get_kvcache()
        full_tokens_per_layer = (
            tp_worker.get_tokens_per_layer_info()[0]
            if tp_worker.is_hybrid_swa
            else None
        )
        # Host-pool retraction transfers full and sliding-window components
        # only, so a model with recurrent state stays on cpu_tensor.
        supports_host_pool = not uses_ssm_state(
            tp_worker.model_runner.model_config
        ) and (
            isinstance(kv_cache, MHATokenToKVPool)
            or (isinstance(kv_cache, SWAKVPool) and full_tokens_per_layer > 0)
        )
        schedule = get_schedule()
        priority_preemption = (
            schedule.enable_priority_scheduling
            and not schedule.disable_priority_preemption
        )
        backend = (
            "host_pool"
            if disagg.disaggregation_mode == "decode"
            # Large ROCm retraction restores can fault the GPU process. Keep
            # host_pool opt-in on HIP until the retraction path is safe at scale.
            and not is_hip()
            and not get_parallel().dcp_enabled
            and not disagg.disaggregation_decode_enable_radix_cache
            # KV offload already owns a host pool; a second one double-books host memory.
            and not disagg.disaggregation_decode_enable_offload_kvcache
            and not priority_preemption
            and supports_host_pool
            else "cpu_tensor"
        )
        fields["disaggregation_decode_retraction_backup"] = backend

    if memory.hicache_ratio is None:
        # Only a decode server reaches resolution with the ratio unset. A
        # backup-only pool can be small: retractions that overflow it abort their
        # request instead of crashing the scheduler. Sharing the pool with
        # HiCache keeps the standard default.
        if backend == "host_pool" and not memory.enable_hierarchical_cache:
            fields["hicache_ratio"] = BACKUP_ONLY_HICACHE_RATIO
        else:
            fields["hicache_ratio"] = 2.0

    if backend == "cpu_tensor" and disagg.disaggregation_mode == "decode":
        # Pools that cannot honor the cpu_tensor contract (DSV4 unified_kv /
        # hisparse / NPU layouts, online-C128 MTP banks) must reject at boot,
        # not at the first retraction mid-serving.
        kv_cache = tp_worker.get_memory_pool()[1].get_kvcache()
        guard = getattr(kv_cache, "_retraction_backup_guard", None)
        if guard is not None:
            guard()

    source = "kv_cache_builder.decode_retraction"
    get_context().override(source, **fields)
    return backend


def _enable_skip_d_role_kv_events(
    tree_cache, tp_rank: int, publishing_rank: bool
) -> None:
    """SGLANG_SYMMETRIC_SKIP_D_ROLE_KV_EVENTS on a symmetric worker: install
    the D-role gate on every TP rank (the same tree state everywhere), or say
    why it stays off. The decision reads only global configuration."""
    if not get_observability().kv_events_config:
        reason = "no --kv-events-config"
    elif not isinstance(tree_cache, UnifiedRadixCache) or tree_cache.disable:
        reason = f"tree cache {type(tree_cache).__name__} is not a unified radix tree"
    elif not hasattr(tree_cache.tree_core, "enable_skip_d_role_kv_events"):
        reason = f"tree core {type(tree_cache.tree_core).__name__} has no D-role gate"
    else:
        tree_cache.enable_skip_d_role_kv_events()
        logger.info(
            "SYMMETRIC_SKIP_D_ROLE_KV_EVENTS on: tp_rank=%d publishing=%d; nodes "
            "added for D-leg requests publish no KV events, P-role nodes publish "
            "as before",
            tp_rank,
            int(bool(publishing_rank)),
        )
        return
    logger.warning("SGLANG_SYMMETRIC_SKIP_D_ROLE_KV_EVENTS has no effect: %s", reason)


def _sidecar_draft_device_pools(draft_plan) -> tuple:
    """Device pools of a SIDECAR HiCache draft plan (EAGLE3), else ()."""
    from decoupled_engine.srt.speculative.base_spec_worker import HiCacheDraftMode

    if getattr(draft_plan, "mode", None) != HiCacheDraftMode.SIDECAR:
        return ()
    return tuple(getattr(draft_plan, "device_pools", ()) or ())


def build_kv_cache(
    *,
    server_args: ServerArgs,
    model_config: ModelConfig,
    tp_worker: BaseTpWorker,
    page_size: int,
    spec_algorithm: SpeculativeAlgorithm,
    attn_tp_cpu_group: ProcessGroup,
    tp_cpu_group: ProcessGroup,
    attn_cp_cpu_group: ProcessGroup,
    enable_metrics: bool,
    enable_kv_cache_events: bool,
    ps: ParallelState,
    tp_group: GroupCoordinator,
    pp_group: GroupCoordinator,
    enable_hierarchical_cache: bool,
    hicache_draft_plan: Optional[HiCacheDraftPlan] = None,
) -> KVCacheBuildResult:
    sliding_window_size: Optional[int] = None
    full_tokens_per_layer: Optional[int] = None
    swa_tokens_per_layer: Optional[int] = None
    uses_transformers_backend = (
        get_resolved_model_impl(model_config) == ModelImpl.TRANSFORMERS
    )

    # Hybrid memory pool
    is_hybrid_swa = tp_worker.is_hybrid_swa
    is_hybrid_ssm = uses_ssm_state(tp_worker.model_runner.model_config)
    is_dsa = is_deepseek_dsa(model_config.hf_config)

    sliding_window_size = None
    if is_hybrid_swa:
        sliding_window_size = tp_worker.sliding_window_size
        full_tokens_per_layer, swa_tokens_per_layer = (
            tp_worker.get_tokens_per_layer_info()
        )

    req_to_token_pool, token_to_kv_pool_allocator = tp_worker.get_memory_pool()
    mtp_draft_device_pools = tp_worker.model_runner.mtp_draft_device_pools

    retraction_backup = resolve_decode_retraction_backup(tp_worker=tp_worker)

    disable_radix_cache = get_memory().disable_radix_cache or (
        model_config.is_multimodal and uses_transformers_backend
    )
    if disable_radix_cache and not get_memory().disable_radix_cache:
        logger.warning(
            "Radix cache is disabled for multimodal models with the "
            "Transformers backend to avoid multimodal prefix-cache mismatches."
        )

    if (
        get_disagg().disaggregation_decode_enable_radix_cache
        and get_disagg().disaggregation_mode == "decode"
    ):
        _validate_decode_radix_cache_compatibility(
            mode=get_disagg().disaggregation_decode_radix_cache_mode,
            device=resolving_view(server_args).device,
            model_config=model_config,
            spec_algorithm=spec_algorithm,
            is_hybrid_swa=is_hybrid_swa,
            is_hybrid_ssm=is_hybrid_ssm,
            enable_hierarchical_cache=enable_hierarchical_cache,
        )
        if is_hybrid_swa:
            logger.warning(
                "EXPERIMENTAL: decode radix cache on a hybrid-SWA model; "
                "prefix reuse follows UnifiedRadixCache SWA semantics"
            )

    effective_chunked_prefill_size = get_schedule().chunked_prefill_size
    if model_config.is_multimodal and uses_transformers_backend:
        effective_chunked_prefill_size = None

    params = CacheInitParams(
        disable=disable_radix_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        # When dcp enabled, kv_pool_allocator.page_size is page_size * dcp_size.
        # TreeCache.page_size should keep the same as allocator.page_size to
        # avoid kv page eviction conflicts.
        page_size=(
            page_size
            if not get_parallel().dcp_enabled
            else token_to_kv_pool_allocator.page_size
        ),
        is_eagle=spec_algorithm.is_eagle(),
        tp_cache_group=(
            attn_tp_cpu_group if get_parallel().enable_dp_attention else tp_cpu_group
        ),
        attn_cp_cache_group=attn_cp_cpu_group,
        attn_tp_cache_group=attn_tp_cpu_group,
        pp_cache_group=pp_group.cpu_group,
        eviction_policy=get_memory().radix_eviction_policy,
        enable_metrics=enable_metrics,
        enable_kv_cache_events=enable_kv_cache_events,
        enable_session_radix_cache=get_memory().enable_session_radix_cache,
        enable_kv_session_retention=get_memory().enable_kv_session_retention,
        enable_mamba_extra_buffer=server_args.enable_mamba_extra_buffer(),
        enable_mamba_extra_buffer_lazy=server_args.enable_mamba_extra_buffer_lazy(),
        pp_rank=ps.pp_rank,
        pp_size=ps.pp_size,
        chunked_prefill_size=effective_chunked_prefill_size,
        sliding_window_size=sliding_window_size,
        mtp_draft_device_pools=mtp_draft_device_pools,
        hicache_sidecar_draft_device_pools=_sidecar_draft_device_pools(
            hicache_draft_plan
        ),
    )

    tree_cache = create_tree_cache(
        TreeCacheBuildContext(
            server_args=server_args,
            params=params,
            is_hybrid_swa=is_hybrid_swa,
            full_tokens_per_layer=full_tokens_per_layer,
            is_hybrid_ssm=is_hybrid_ssm,
            is_dsa=is_dsa,
            enable_hierarchical_cache=enable_hierarchical_cache,
            disable_radix_cache=disable_radix_cache,
            effective_chunked_prefill_size=effective_chunked_prefill_size,
            tp_worker=tp_worker,
            model_config=model_config,
            tp_size=ps.tp_size,
            tp_rank=ps.tp_rank,
            tp_group=tp_group,
        )
    )

    # Native EAGLE3 validates that its separate draft sidecar participates in
    # every restore. Attach it before validating the completed cache layout.
    if (
        enable_hierarchical_cache or retraction_backup == "host_pool"
    ) and hicache_draft_plan is not None:
        maybe_register_hicache_draft(
            tree_cache=tree_cache,
            draft_plan=hicache_draft_plan,
        )

    if (
        get_disagg().disaggregation_mode == "symmetric"
        and envs.SGLANG_SYMMETRIC_IMPORT_L2_RESTORE.get()
    ):
        # This diagnostic reuses the existing full-component L2. SWA, Mamba,
        # buffer mode and the M4/Q4 hierarchy are separate implementations.
        from decoupled_engine.srt.arg_groups.symmetric_defaults import refuse
        from decoupled_engine.srt.mem_cache.allocator.cpu_paged import cpu_paged_allocator

        unsupported = None
        if (
            not enable_hierarchical_cache
            or not isinstance(tree_cache, UnifiedRadixCache)
            or is_hybrid_swa
            or is_hybrid_ssm
            or tree_cache.buffer_pipeline is not None
        ):
            unsupported = (
                "SGLANG_SYMMETRIC_IMPORT_L2_RESTORE requires ordinary "
                "UnifiedRadixCache with HiCache, without SWA/Mamba/buffer mode"
            )
        elif cpu_paged_allocator(token_to_kv_pool_allocator) is not None:
            unsupported = "symmetric L2 restore diagnostic does not support M4/Q4"
        if unsupported is not None:
            refuse("SGLANG_SYMMETRIC_IMPORT_L2_RESTORE", unsupported)
        else:
            if spec_algorithm.is_eagle():
                _validate_decode_eagle_hicache_draft(
                    tree_cache=tree_cache, draft_plan=hicache_draft_plan
                )
            tree_cache.symmetric_import_l2_restore_validated = True

    if (
        get_disagg().disaggregation_mode == "symmetric"
        and is_hybrid_swa
        and envs.SGLANG_SYMMETRIC_LAUNCH_THREAD.get()
    ):
        from decoupled_engine.srt.arg_groups.symmetric_defaults import auto_applied, refuse

        # Run only where it was measured: models without SWA.
        if auto_applied("SGLANG_SYMMETRIC_LAUNCH_THREAD"):
            refuse(
                "SGLANG_SYMMETRIC_LAUNCH_THREAD",
                "SGLANG_SYMMETRIC_LAUNCH_THREAD is not validated on SWA models",
            )

    if (
        get_disagg().disaggregation_mode == "symmetric"
        and envs.SGLANG_SYMMETRIC_SKIP_D_ROLE_KV_EVENTS.get()
    ):
        _enable_skip_d_role_kv_events(tree_cache, ps.tp_rank, enable_kv_cache_events)

    if (
        get_disagg().disaggregation_decode_enable_radix_cache
        and get_disagg().disaggregation_mode == "decode"
    ):
        _validate_decode_radix_tree_cache(
            mode=get_disagg().disaggregation_decode_radix_cache_mode,
            tree_cache=tree_cache,
            token_to_kv_pool=token_to_kv_pool_allocator.get_kvcache(),
        )
        if enable_hierarchical_cache:
            if spec_algorithm.is_eagle() and (
                get_disagg().disaggregation_decode_radix_cache_mode
                not in ("content_only", "window_host")
            ):
                _validate_decode_eagle_hicache_draft(
                    tree_cache=tree_cache, draft_plan=hicache_draft_plan
                )
            from decoupled_engine.srt.disaggregation.decode_hicache_restore_log import (
                warn_if_stalling_restore_fanout,
            )

            warn_if_stalling_restore_fanout(tree_cache)

    if retraction_backup == "host_pool":
        if not isinstance(tree_cache, UnifiedRadixCache):
            raise ValueError(
                "--disaggregation-decode-retraction-backup=host_pool requires "
                "UnifiedRadixCache with HiCache attached."
            )
        tree_cache.validate_retraction_host_capacity()

    embedding_cache_size = envs.SGLANG_VLM_CACHE_SIZE_MB.get()
    init_mm_embedding_cache(embedding_cache_size * 1024 * 1024)

    return KVCacheBuildResult(
        is_hybrid_swa=is_hybrid_swa,
        is_hybrid_ssm=is_hybrid_ssm,
        sliding_window_size=sliding_window_size,
        full_tokens_per_layer=full_tokens_per_layer,
        swa_tokens_per_layer=swa_tokens_per_layer,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        disable_radix_cache=disable_radix_cache,
        tree_cache=tree_cache,
    )
