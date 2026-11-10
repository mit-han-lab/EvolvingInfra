"""Factory for the owner / reader planes used by the symmetric scheduler.

`init_disaggregation` constructs BOTH an ExportTable (owner
side) and an ImportQueue (reader side); the data plane behind them lives in
`disaggregation/pull/` (owner.py, importer.py) and is imported
lazily here. When those modules are absent (or SGLANG_SYMMETRIC_PD_FAKE_PLANES=1),
the in-process loopback planes from `fake_planes.py` are used so the
scheduler state machine can run and be unit-tested on CPU.

Symmetric mode never constructs a CommonKVManager (common/conn.py raises on
non prefill/decode) and never starts the bootstrap HTTP server
(managers/disagg_service.py only does so for PREFILL): the owner's address
comes from the router in the kv_import mark.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional, Tuple

import msgspec

from decoupled_engine.srt.disaggregation.pull.api import OwnerPlane, ReaderPlane

if TYPE_CHECKING:
    from decoupled_engine.srt.disaggregation.base.conn import KVArgs
    from decoupled_engine.srt.disaggregation.utils import MetadataBuffers
    from decoupled_engine.srt.managers.scheduler import Scheduler
    from decoupled_engine.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
    from decoupled_engine.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class PlaneBuildContext(msgspec.Struct, kw_only=True, frozen=True):
    """Everything the planes need at construction time (pull/owner.py
    OwnerPlane and pull/importer.py ReaderPlane constructor contracts)."""

    server_args: ServerArgs
    kv_args: KVArgs
    tp_rank: int
    tp_size: int
    pp_rank: int
    dp_rank: int
    gpu_id: int
    page_size: int
    metadata_buffers: MetadataBuffers
    token_to_kv_pool: KVCache
    token_to_kv_pool_allocator: Any
    req_to_token_pool: ReqToTokenPool
    # Router-visible identity of this Worker ("Worker"); used in
    # log lines next to transfer_id.
    worker_id: str = ""
    attn_tp_size: int = 1


def build_kv_args(scheduler: Scheduler, draft_token_to_kv_pool) -> KVArgs:
    """Describe this rank's KV / aux / state buffers for the read engine.

    Adapted from `PrefillBootstrapQueue._init_kv_manager` (prefill.py) minus
    the CommonKVManager construction; the base `KVArgs` is used because
    symmetric mode is transport-agnostic at this layer.
    """
    from decoupled_engine.srt.disaggregation.base.conn import KVArgs
    from decoupled_engine.srt.disaggregation.prefill import _transfer_start_layer
    from decoupled_engine.srt.disaggregation.utils import (
        build_kv_layer_ids,
        is_mla_backend,
        setup_state_kv_args,
    )
    from decoupled_engine.srt.runtime_context import get_disagg

    token_to_kv_pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    kv_args = KVArgs()
    kv_args.engine_rank = scheduler.ps.tp_rank
    kv_args.pp_rank = scheduler.ps.pp_rank
    kv_args.system_dp_rank = scheduler.ps.dp_rank
    kv_args.kv_cache_dtype_str = scheduler.tp_worker.model_runner.kv_cache_dtype_str
    kv_args.prefill_start_layer = _transfer_start_layer(
        pool=token_to_kv_pool,
        hf_text_config=scheduler.model_config.hf_text_config,
    )
    kv_args.mla_compression_ratios = None
    kv_data_ptrs, kv_data_lens, kv_item_lens = (
        token_to_kv_pool.get_contiguous_buf_infos()
    )
    # Layer-sharded pools carry end_layer; others do not (same read as prefill.py).
    kv_args.prefill_end_layer = getattr(token_to_kv_pool, "end_layer", None)

    num_draft_entries = 0
    if draft_token_to_kv_pool is not None:
        draft_ptrs, draft_lens, draft_item_lens = (
            draft_token_to_kv_pool.get_contiguous_buf_infos()
        )
        kv_data_ptrs += draft_ptrs
        kv_data_lens += draft_lens
        kv_item_lens += draft_item_lens
        num_draft_entries = len(draft_ptrs)

    kv_args.kv_data_ptrs = kv_data_ptrs
    kv_args.kv_data_lens = kv_data_lens
    kv_args.kv_item_lens = kv_item_lens
    kv_args.kv_layer_ids = build_kv_layer_ids(
        token_to_kv_pool=token_to_kv_pool,
        draft_token_to_kv_pool=draft_token_to_kv_pool,
        num_draft_entries=num_draft_entries,
        num_hidden_layers=scheduler.model_config.num_hidden_layers,
    )
    if not is_mla_backend(token_to_kv_pool):
        kv_args.kv_head_num = token_to_kv_pool.head_num
        kv_args.total_kv_head_num = scheduler.model_config.get_total_num_kv_heads()
    kv_args.page_size = token_to_kv_pool.page_size
    kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
        scheduler.disagg_metadata_buffers.get_buf_infos()
    )
    kv_args.ib_device = get_disagg().disaggregation_ib_device
    kv_args.gpu_id = scheduler.ps.gpu_id
    setup_state_kv_args(
        kv_args,
        token_to_kv_pool,
        draft_token_to_kv_pool,
        scheduler.model_config.num_hidden_layers,
        req_to_token_pool=scheduler.req_to_token_pool,
    )
    from decoupled_engine.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    if isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool):
        # DSv4's KVCache is organized by compression-ratio bucket, not by layer.
        kv_args.mla_compression_ratios = list(token_to_kv_pool.compression_ratios)
    return kv_args


def _force_fake_planes() -> bool:
    return os.getenv("SGLANG_SYMMETRIC_PD_FAKE_PLANES", "0") not in ("", "0", "false")


def _late_d_mode() -> str:
    from decoupled_engine.srt.runtime_context import get_disagg

    return get_disagg().disaggregation_late_d


def plane_bind_addrs(server_args: ServerArgs, tp_rank: int) -> Tuple[str, str]:
    """(owner_addr, reader_addr) ZMQ endpoints of this rank's channels.

    Arbitrary scheme: symmetric mode starts no bootstrap server, so the
    bootstrap port range is free; owner = base + 2*tp_rank, reader = +1.
    TODO(symmetric P/D): agree the advertised endpoints with the router
    instead of deriving them here.
    """
    host = zmq_host(advertised_host(server_args))
    base = server_args.disaggregation_bootstrap_port + 2 * tp_rank
    return f"tcp://{host}:{base}", f"tcp://{host}:{base + 1}"


def zmq_host(host: str) -> str:
    """Host part of a ZMQ tcp endpoint: an IPv6 literal goes in brackets
    (tcp://[fdaa::1]:9000); IPv4 addresses and names are unchanged."""
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def advertised_host(server_args: ServerArgs) -> str:
    """Routable IP of this worker: `--host` unless it is a wildcard bind, in
    which case the auto-detected local IP (same rule mooncake/conn.py uses for
    the KV manager's hostname)."""
    from decoupled_engine.srt.utils.network import get_local_ip_auto

    host = server_args.host
    if host in ("0.0.0.0", "", "::", "[::]"):
        host = get_local_ip_auto()
    return host


def rank_addr(base_addr: str, tp_rank: int) -> str:
    """Per-rank endpoint from the rank-0 endpoint the router/driver advertises
    (owner = bootstrap_port, reader = bootstrap_port + 1): rank i lives at
    port + 2*i on both channels (homogeneous TP, rank i talks to rank i)."""
    if tp_rank == 0 or not base_addr.startswith("tcp://"):
        return base_addr
    host, _, port = base_addr[len("tcp://") :].rpartition(":")
    if not port.isdigit():
        return base_addr
    return f"tcp://{host}:{int(port) + 2 * tp_rank}"


def _build_real_planes(ctx: PlaneBuildContext) -> Tuple[OwnerPlane, ReaderPlane]:
    from decoupled_engine.srt.disaggregation.pull.importer import ReaderPlane as ReaderPlaneImpl
    from decoupled_engine.srt.disaggregation.pull.owner import OwnerPlane as OwnerPlaneImpl
    from decoupled_engine.srt.disaggregation.pull.protocol import Registration
    from decoupled_engine.srt.disaggregation.pull.read_engine import (
        MooncakeReadEngine,
        registered_regions_of_kv_args,
    )
    from decoupled_engine.srt.disaggregation.symmetric.export_table import (
        enable_page_generation_tracking,
        make_generation_of,
        registration_pools_for,
    )
    from decoupled_engine.srt.distributed.device_communicators.mooncake_transfer_engine import (
        MooncakeTransferEngine,
    )
    from decoupled_engine.srt.mem_cache.peer_kv_servicer import GpuLeaseTable

    # No CommonKVManager in symmetric mode, so register the KV/aux/state
    # regions with the transfer engine here (same regions the PD conn would).
    # Mooncake advertises hostname:rpc_port as the session id the reader dials,
    # so it must be a routable IP, never the 0.0.0.0 wildcard of --host.
    te = MooncakeTransferEngine(
        hostname=advertised_host(ctx.server_args),
        gpu_id=ctx.gpu_id,
        ib_device=ctx.kv_args.ib_device,
    )
    regions = registered_regions_of_kv_args(ctx.kv_args)
    if regions:
        ptrs, lens = zip(*regions)
        rc = te.batch_register(list(ptrs), list(lens))
        if rc != 0:
            raise RuntimeError(f"symmetric-pd: KV region registration failed rc={rc}")
    # DSv4 gets per-component pools; other models the
    # single "full" layout. The SWA pool is not a kv_data_ptrs entry, so it is
    # added as a ready-made region.
    pool_layouts, extra_pools = registration_pools_for(
        kvcache=ctx.token_to_kv_pool, kv_args=ctx.kv_args
    )
    late_d = _late_d_mode()
    registration = Registration.from_kv_args(
        kv_args=ctx.kv_args,
        session_id=te.get_session_id(),
        attn_tp_size=ctx.attn_tp_size,
        pool_layouts=pool_layouts,
        worker=ctx.worker_id,
        # Readers compare it at REGISTER (mixed late-D flags).
        late_d=late_d,
    )
    if extra_pools:
        registration = msgspec.structs.replace(
            registration, pools={**registration.pools, **extra_pools}
        )
    owner_addr, reader_addr = plane_bind_addrs(ctx.server_args, ctx.tp_rank)
    # Exported pages are request-held (the request stays EXPORTED and never
    # frees), so the Lease needs no radix locker here; generations
    # come from the GPU paged allocator, bumped on free from here on.
    enable_page_generation_tracking(ctx.token_to_kv_pool_allocator)
    lease = GpuLeaseTable(
        ttl_ms=None,
        generation_of=make_generation_of(
            ctx.token_to_kv_pool_allocator, page_size=ctx.page_size
        ),
    )
    # Late D forces the reader-first await mode on both planes: the
    # owner collects BEGIN, the reader sends it at allocation. Off: the
    # SGLANG_SYMMETRIC_PD_EXPORT_AWAIT_READER env decides, as before.
    late_on = late_d != "off"
    owner = OwnerPlaneImpl(
        registration=registration,
        lease=lease,
        bind_addr=owner_addr,
        worker_name=ctx.worker_id,
        await_reader=True if late_on else None,
    )
    # Reader-local aux / state slots travel on ImportMark (import_queue.py
    # _prealloc), so the plane needs no scheduler callback.
    reader = ReaderPlaneImpl(
        registration=registration,
        read_engine=MooncakeReadEngine.from_kv_args(te=te, kv_args=ctx.kv_args),
        bind_addr=reader_addr,
        worker_name=ctx.worker_id,
        send_begin=True if late_on else None,
    )
    logger.info(
        "symmetric-pd planes bound: owner=%s reader=%s session=%s",
        owner_addr,
        reader_addr,
        registration.session_id,
    )
    return owner, reader


def build_planes(ctx: PlaneBuildContext) -> Tuple[OwnerPlane, ReaderPlane]:
    """Return (OwnerPlane, ReaderPlane) for this rank.

    Lazily imports the real pull data plane; falls back to `fake_planes`
    (loopback, no bytes moved) when it is not importable.
    """
    if not _force_fake_planes():
        try:
            return _build_real_planes(ctx)
        except ImportError as e:
            logger.warning(
                "symmetric-pd symmetric: pull data plane unavailable (%s); using "
                "in-process fake planes (no KV bytes are transferred).",
                e,
            )
    else:
        logger.warning(
            "symmetric-pd symmetric: SGLANG_SYMMETRIC_PD_FAKE_PLANES set; using fake planes."
        )

    from decoupled_engine.srt.disaggregation.symmetric.fake_planes import build_fake_planes

    metadata_buffers = ctx.metadata_buffers
    return build_fake_planes(
        aux_reader=lambda final, local_idx: copy_metadata_slot(
            metadata_buffers, src=final.aux_index, dst=local_idx
        ),
        # Late D: ADMITs reach this rank's reader by its advertised endpoint.
        reader_addr=plane_bind_addrs(ctx.server_args, ctx.tp_rank)[1],
    )


def copy_metadata_slot(
    metadata_buffers: MetadataBuffers, *, src: int, dst: int
) -> None:
    """Loopback stand-in for the READ of the owner's aux slot: copy every
    MetadataBuffers row `src` into row `dst` (same process, same buffers)."""
    if src == dst:
        return
    for name in (
        "output_ids",
        "cached_tokens",
        "output_token_logprobs_val",
        "output_token_logprobs_idx",
        "output_top_logprobs_val",
        "output_top_logprobs_idx",
        "output_token_sampling_mask_len",
        "output_token_sampling_mask_idx",
        "output_token_sampling_logprobs",
        "output_topk_p",
        "output_topk_index",
        "output_hidden_states",
        "output_dsa_topk_indices",
        "bootstrap_room",
    ):
        buf = getattr(metadata_buffers, name)  # optional buffers are None
        if buf is not None:
            buf[dst].copy_(buf[src])


def make_worker_id(server_args: ServerArgs, dp_rank: Optional[int]) -> str:
    suffix = f"/dp{dp_rank}" if dp_rank is not None else ""
    return f"{server_args.host}:{server_args.port}{suffix}"
