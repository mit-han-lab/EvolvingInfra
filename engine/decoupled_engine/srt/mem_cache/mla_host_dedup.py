"""One CPU payload per TP group for replicated MLA target and draft KV.

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

from decoupled_engine.srt.mem_cache.dsa_host_dedup import DSAHostDedup
from decoupled_engine.srt.mem_cache.dsv4_host_dedup import (
    DeepSeekV4HostDedup,
    build_host_dedup,
)

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
