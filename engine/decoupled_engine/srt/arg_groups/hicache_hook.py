# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang python/sglang/srt/arg_groups/hicache_hook.py (commit 9a489f8d2f), modified for evolving-infra.
"""Server-argument resolution for the hierarchical KV cache."""

from __future__ import annotations

import logging
from typing import Any

from decoupled_engine.srt.arg_groups.overrides import (
    declare_resolution,
    resolving_view,
    use_mla_backend,
)
from decoupled_engine.srt.environ import envs

logger = logging.getLogger(__name__)


def handle_hicache(server_args: Any):
    """Normalize hicache-related knobs into a valid runtime configuration.

    Resolution order:
    1) Layout <-> I/O compatibility for direct conflicts.
    2) Storage <-> layout compatibility (may rewrite layout).
    """
    cfg = resolving_view(server_args)
    # kv-p2p: the PeerKVServicer leases HiCache pinned host pages to peers,
    # so it is meaningless without the host tier. Checked before the
    # early-return below (which would skip it exactly in the failing case).
    if cfg.enable_peer_kv_servicer and not cfg.enable_hierarchical_cache:
        raise ValueError(
            "--enable-peer-kv-servicer requires --enable-hierarchical-cache "
            "(it serves the HiCache pinned host tier)."
        )
    if cfg.peer_kv_demote_source and not cfg.enable_peer_kv_servicer:
        raise ValueError(
            "--peer-kv-demote-source requires --enable-peer-kv-servicer "
            "(it demotes the copies that this worker's servicer leased)."
        )
    if cfg.peer_kv_demote_source:
        # The master switch of the freshness-class policy (lever 1 and every
        # SGLANG_PEER_KV_DEMOTE_* flag). In null and symmetric mode the commit-time marks
        # would run and the recompute NOTICE never: refuse, not half on.
        mode = cfg.disaggregation_mode
        if mode not in ("prefill", "decode"):
            raise ValueError(
                "--peer-kv-demote-source supports --disaggregation-mode prefill "
                f"and decode only; {mode!r} is not supported yet (null and "
                "symmetric)."
            )
        if mode == "decode":
            logger.info("PeerKV demote: launch mode decode")
    validate_kv_session_retention_combination(server_args)
    validate_peer_kv_dedup_fanout(server_args)
    # Skip all normalization when neither hicache nor decode-offload path is active.
    if not (
        cfg.enable_hierarchical_cache
        or cfg.disaggregation_decode_enable_offload_kvcache
        or (
            cfg.disaggregation_mode == "decode"
            and cfg.disaggregation_decode_retraction_backup in (None, "host_pool")
        )
    ):
        return

    validate_hicache_host_memory_mode(server_args)

    # Step 1: Initial layout-io compatibility normalization.
    resolve_layout_io_compatibility(server_args)

    # Step 2: Storage-layout normalization without changing io backend.
    resolve_storage_layout_compatibility(server_args)

    # Step 3: DCP compatibility for the L2 (device<->host) path.
    resolve_hicache_dcp_compatibility(server_args)


def validate_kv_session_retention_combination(server_args: Any):
    """Refuse --enable-kv-session-retention with the freshness-class levers.

    Retention orders host eviction by its own class and the CPU recency
    (host_last_access_time). --peer-kv-demote-source would lose its demoted
    class there, and the deferred-pull TOUCH refreshes only GPU recency, so
    neither would act on host eviction. Refuse at startup, not half on.
    """
    cfg = resolving_view(server_args)
    if not getattr(cfg, "enable_kv_session_retention", False):
        return
    if cfg.peer_kv_demote_source:
        raise ValueError(
            "--enable-kv-session-retention cannot be combined with "
            "--peer-kv-demote-source (retention host eviction ignores the "
            "demoted class)."
        )
    if envs.SGLANG_PEER_KV_TOUCH_DEFERRED_PULLS.get():
        raise ValueError(
            "--enable-kv-session-retention cannot be combined with "
            "SGLANG_PEER_KV_TOUCH_DEFERRED_PULLS=1 (the TOUCH refreshes only "
            "GPU recency; retention host eviction uses CPU recency)."
        )


def validate_peer_kv_dedup_fanout(server_args: Any):
    """Reject the peer-KV servicer with the shared DSA host-dedup fan-out.

    With SGLANG_DSA_HOST_DEDUP_FANOUT=shared the TP followers load the
    source's host pool by their OWN host rows, which match the source's only
    while every rank's allocator hands out the same rows. Only the source runs
    the PeerKVServicer, and a lease keeps a freed page out of the source's
    free lists alone, so the source's later rows differ from the followers'
    and the followers would read wrong pages (TP0 landed writes many pages
    after the other ranks). The other fan-outs copy the source's device rows,
    so that skew is harmless there.
    """
    cfg = resolving_view(server_args)
    if (
        cfg.enable_peer_kv_servicer
        and cfg.enable_dsa_hicache_host_dedup
        and envs.SGLANG_DSA_HOST_DEDUP_FANOUT.get() == "shared"
    ):
        raise ValueError(
            "--enable-peer-kv-servicer is incompatible with "
            "--enable-dsa-hicache-host-dedup under "
            "SGLANG_DSA_HOST_DEDUP_FANOUT=shared: a peer lease withholds freed "
            "host pages on the source rank only, and the shared fan-out's "
            "followers read the source's host pool by their own host rows. Use "
            "SGLANG_DSA_HOST_DEDUP_FANOUT=ce_peer or nccl, or drop "
            "--enable-peer-kv-servicer."
        )


def handle_hicache_ratio_default(server_args: Any):
    """Default the host/device ratio per host memory mode.

    Runs before the dummy-model boundary: direct HostKVCache consumers
    (unit fixtures, dummy-model launches) must never see a None ratio.
    buffer_only stages in flight rather than retaining, so it needs only
    enough to cover the write backlog plus parked prefetches.

    A decode server keeps the ratio unset here: kv_cache_builder resolves
    it against the retraction-backup backend (1.0 for host_pool, else 2.0).
    """
    cfg = resolving_view(server_args)
    if cfg.hicache_ratio is None and cfg.disaggregation_mode != "decode":
        declare_resolution(
            server_args,
            "_handle_hicache_ratio_default",
            hicache_ratio=(
                1.2 if cfg.hicache_host_memory_mode == "buffer_only" else 2.0
            ),
        )


def resolve_hicache_dcp_compatibility(server_args: Any):

    cfg = resolving_view(server_args)
    if cfg.dcp_size <= 1 or not cfg.enable_hierarchical_cache:
        return
    if cfg.hicache_storage_backend is not None:
        raise NotImplementedError(
            "--hicache-storage-backend (L3) with --dcp-size > 1 is not "
            "supported yet: under DCP each rank holds a distinct "
            "interleaved MLA KV shard, so the rank-0-only replicated-MLA "
            "backup and the storage keys must become dcp_rank-aware "
            "first. Run HiCache+DCP with L1/L2 only."
        )
    if cfg.speculative_algorithm not in (None, "DSPARK"):
        raise NotImplementedError(
            "HiCache with --dcp-size > 1 only supports DSPARK speculative "
            "decoding; other draft-model host pools have no DCP index "
            "translation."
        )
    if cfg.enable_lmcache:
        raise NotImplementedError(
            "--enable-lmcache with --dcp-size > 1 is not supported: "
            "LMCache has no DCP-aware index translation."
        )
    if cfg.enable_hisparse:
        raise NotImplementedError(
            "--enable-hisparse with --dcp-size > 1 is not supported: the "
            "HiSparse host pool is constructed without DCP translation."
        )
    if not use_mla_backend(server_args):
        raise NotImplementedError(
            "HiCache with --dcp-size > 1 is only supported for MLA models: "
            "the index translation lives in MLATokenToKVPoolHost, and the "
            "MHA host pool has none."
        )
    logger.info(
        "HiCache + DCP enabled (L1/L2 only): host pool uses widened "
        "logical slot accounting with per-rank physical translation at "
        "the transfer boundary (dcp_size=%d).",
        cfg.dcp_size,
    )


def resolve_layout_io_compatibility(server_args: Any):
    cfg = resolving_view(server_args)
    if (
        cfg.hicache_mem_layout == "page_first_direct"
        and cfg.hicache_io_backend == "kernel"
    ):
        declare_resolution(
            server_args,
            "_resolve_layout_io_compatibility",
            hicache_io_backend="direct",
        )
        logger.warning(
            "Kernel io backend does not support page first direct layout, switching to direct io backend"
        )

    if cfg.hicache_mem_layout == "page_first" and cfg.hicache_io_backend == "direct":
        declare_resolution(
            server_args,
            "_resolve_layout_io_compatibility",
            hicache_mem_layout="page_first_direct",
        )
        logger.warning(
            "Page first layout is not supported with direct IO backend, switching to page first direct layout"
        )


def resolve_storage_layout_compatibility(server_args: Any):
    cfg = resolving_view(server_args)
    if (
        cfg.hicache_storage_backend != "mooncake"
        or cfg.hicache_mem_layout != "layer_first"
    ):
        return

    if cfg.hicache_io_backend == "direct":
        new_layout = "page_first_direct"
    elif cfg.hicache_io_backend == "kernel":
        new_layout = "page_first"
    else:
        # Keep current behavior for unknown backends (e.g., kernel_ascend).
        new_layout = cfg.hicache_mem_layout

    declare_resolution(
        server_args,
        "_resolve_storage_layout_compatibility",
        hicache_mem_layout=new_layout,
    )
    logger.warning(
        f"Mooncake storage backend does not support layer_first layout, "
        f"switching to {new_layout} layout for {cfg.hicache_io_backend} io backend"
    )


def validate_hicache_host_memory_mode(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.hicache_host_memory_mode not in ("cache", "buffer_only"):
        raise ValueError(
            "hicache_host_memory_mode must be 'cache' or 'buffer_only', "
            f"got {cfg.hicache_host_memory_mode!r}"
        )

    # Both modes are defaulted upstream (a decode server resolves the
    # ratio later, in kv_cache_builder), so this fires only if that
    # defaulting regresses -- never build an unsized host pool.
    if (
        cfg.hicache_size <= 0
        and cfg.hicache_ratio is None
        and cfg.disaggregation_mode != "decode"
    ):
        raise ValueError(
            f"--hicache-host-memory-mode {cfg.hicache_host_memory_mode} "
            "requires a host pool size: pass --hicache-size or "
            "--hicache-ratio."
        )

    if cfg.hicache_host_memory_mode == "cache":
        return

    if cfg.hicache_storage_backend is None:
        raise ValueError(
            "--hicache-host-memory-mode buffer_only requires a storage backend "
            "(--hicache-storage-backend): host memory is only a staging buffer "
            "and all cached data lives in storage."
        )
    if cfg.hicache_write_policy == "write_back":
        raise ValueError(
            "--hicache-host-memory-mode buffer_only does not support "
            "--hicache-write-policy write_back; use write_through or "
            "write_through_selective."
        )
    if cfg.disaggregation_mode == "decode":
        raise ValueError(
            "--hicache-host-memory-mode buffer_only is not supported on "
            "decode instances: the decode-side prefetch and offload paths "
            "bypass the buffer-mode pipeline, fetching without its prefix "
            "context and never consuming its staged holds. Prefill "
            "instances share the standard scheduler path and are supported."
        )
