# Adapted from SGLang python/sglang/srt/disaggregation/prefill.py (commit 9a489f8d2f), modified for evolving-infra.
"""
Life cycle of a request in the prefill server

1. Bootstrap Queue
    a. Initialize a sender for each request
    b. Use the queue to store requests whose bootstrap (handshake and preallocation) has not finished
    c. Poll senders to check bootstrap state
    d. Once bootstrap is complete, move request to Waiting Queue

2. Waiting Queue
    a. Use PrefillAdder to pop requests
    b. Run forward
    c. Add the request to Inflight Queue

3. Inflight Queue
    a. Poll (non-blocking) the sender of the request
    b. Once the transfer has finished, return the request
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import dataclasses
from array import array
from collections import deque
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

from decoupled_engine.srt.disaggregation.base import KVPoll
from decoupled_engine.srt.disaggregation.base.conn import StateType
from decoupled_engine.srt.disaggregation.common.conn import CommonKVManager
from decoupled_engine.srt.disaggregation.common.staging_buffer import (
    compute_grid_segments,
    staging_grid_tokens,
)
from decoupled_engine.srt.disaggregation.late_d import (
    LATE_D_CHOOSE,
    LEAD_NEED,
    LateDConfig,
    LateDPrefillBinder,
    make_late_d_metrics,
)
from decoupled_engine.srt.disaggregation.late_d_link import LateDLink
from decoupled_engine.srt.disaggregation.late_d_need import (
    NeedAdmissionMixin, START_OK, HOLD, catchup_end, force_bound_for_smoke,
)
from decoupled_engine.srt.disaggregation.prefill_launch_thread import PrefillLaunchThread
from decoupled_engine.srt.disaggregation.prefill_loop_unserialize import STATS as M2_STATS
from decoupled_engine.srt.disaggregation.prefill_loop_unserialize import (
    KvPageIdReader,
    PendingEarlySend,
    PrefillUnserializeConfig,
    StatsLogger,
    flush_pending_early_sends,
    page_ids_from_snapshot,
    sender_waits_on_early_send_event,
    snapshot_early_send_page_ids,
)
from decoupled_engine.srt.disaggregation.prefill_result_first import PrefillResultFirst
from decoupled_engine.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    build_kv_layer_ids,
    build_staging_slot_metadata,
    get_dsv4_c128_state_indices,
    get_kv_class,
    is_aborted,
    is_dsv4_c128_online_enabled,
    is_mla_backend,
    poll_and_all_reduce_attn_cp_tp_group,
    poll_and_all_reduce_attn_cp_tp_lanes,
    poll_and_all_reduce_pp,
    prepare_abort,
    setup_state_kv_args,
)
from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    NextBatchPlan,
    Req,
    ScheduleBatch,
)
from decoupled_engine.srt.managers.scheduler_components.kv_events_publisher import (
    kv_events_end_pass_hook,
)
from decoupled_engine.srt.managers.scheduler_components.m2_lockstep_stats import LockstepStats
from decoupled_engine.srt.managers.scheduler_components.phase_timers import (
    PHASES,
    sync_debug_site,
    timed_phase,
)
from decoupled_engine.srt.mem_cache.ce_loader import (
    debug_force_gate_reorder_ranks,
    gate_pending_for_batch,
    log_debug_force_gate_reorder,
)
from decoupled_engine.srt.mem_cache.common import (
    kv_to_page_indices,
    kv_to_page_num,
    maybe_cache_unfinished_req,
    release_kv_cache,
    start_deferring_chunk_backups,
)
from decoupled_engine.srt.mem_cache.cpu_page_book import (
    PrefillCpuPageBook,
    cpu_page_bookkeeping_enabled,
)
from decoupled_engine.srt.mem_cache.cpu_page_book import (
    supported_reason as cpu_page_book_supported_reason,
)
from decoupled_engine.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from decoupled_engine.srt.mem_cache.load_quarantine import (
    LoadQuarantine,
    QuarantineClock,
    load_quarantine_enabled,
)
from decoupled_engine.srt.mem_cache.memory_pool import HybridLinearKVPool
from decoupled_engine.srt.mem_cache.unified_cache.session_ref_tracker import (
    forget_kv_session_request,
    rebind_kv_session_request,
)
from decoupled_engine.srt.observability.req_time_stats import set_schedule_time_batch
from decoupled_engine.srt.runtime_context import (
    get_disagg,
    get_parallel,
    get_schedule,
)
from decoupled_engine.srt.utils import is_npu
from decoupled_engine.srt.utils.nvtx_utils import scheduler_nvtx_method

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from decoupled_engine.srt.managers.scheduler import GenerationBatchResult, Scheduler
    from decoupled_engine.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)

# M2_LOCKSTEP off: a scheduler whose loop never set up the counters.
_NO_LOCKSTEP = LockstepStats()

_is_npu = is_npu()


def should_force_retry(req: Req) -> bool:
    """Test hook to force a request into optimistic prefill retry."""
    retry_prob = envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.get()
    # Force only before/during the first attempt (count is 1 while it runs).
    if retry_prob <= 0 or req.prefill_attempt_count > 1 or req.is_retracted:
        return False

    digest = hashlib.sha256(str(req.rid).encode()).digest()
    return int.from_bytes(digest[:8], "big") < retry_prob * 2**64


def _transfer_start_layer(*, pool, hf_text_config) -> int:
    # Hybrid pools count all layers in start_layer, but peer KV lists contain only
    # full-attention layers, so translate to a full-attention-relative offset.
    if not isinstance(pool, HybridLinearKVPool):
        return pool.start_layer
    return sum(
        1 for lid in hf_text_config.full_attention_layer_ids if lid < pool.start_layer
    )


def maybe_release_metadata_buffer(
    req: Req, allocator: ReqToMetadataIdxAllocator
) -> None:
    """
    Release the metadata buffer index allocated for a request in prefill disaggregation mode.

    This function safely releases the metadata buffer index if it was allocated.

    Args:
        req: The request object that may have a metadata_buffer_index allocated
        allocator: The ReqToMetadataIdxAllocator instance to free the index
    """
    if req.metadata_buffer_index >= 0:
        allocator.free(req.metadata_buffer_index)
        req.metadata_buffer_index = -1


def initialize_waiting_restore_plan(scheduler):
    """Waiting CPU restore uses the same planner with early and late D."""
    if (getattr(scheduler, "_late_d_need_enabled", False)
            or envs.SGLANG_BWP_SOURCE.get().strip().lower() == "waiting"):
        from decoupled_engine.srt.disaggregation.lookahead_plan import SchedulerLookahead
        scheduler.late_d_lookahead = SchedulerLookahead(scheduler)
        scheduler._late_d_chunk_tokens = scheduler.chunked_prefill_size or 8192


class PrefillBootstrapQueue:
    """
    Store the requests in bootstrapping
    """

    # Late D (--disaggregation-late-d): rooms bind on ADMIT (every rank); the
    # ADMIT binder runs on the leader only.
    late_d_enabled = False
    late_d_binder: Optional[LateDPrefillBinder] = None

    def __init__(
        self,
        token_to_kv_pool: KVCache,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        tp_rank: int,
        tp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        max_total_num_tokens: int,
        scheduler: Scheduler,
        pp_rank: int,
        pp_size: int,
        transfer_backend: TransferBackend,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.queue: List[Req] = []
        self.gloo_group = gloo_group
        self.scheduler = scheduler
        self.scheduler._late_d_need_enabled = (
            getattr(get_disagg(), "disaggregation_late_d_lead_mode", "count") == LEAD_NEED
            and get_disagg().disaggregation_late_d != "off"
        )
        self.max_total_num_tokens = (
            self.scheduler.tp_worker.model_runner.effective_max_total_num_tokens
        )
        self.transfer_backend = transfer_backend
        if envs.SGLANG_DISAGG_STAGING_BUFFER.get():
            if self.is_mla_backend:
                raise RuntimeError(
                    "SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models "
                    "(e.g. GQA, MHA). MLA models should not set this flag."
                )
            page_size = self.scheduler.token_to_kv_pool_allocator.page_size
            # Same source as send_kv_chunk's staging grid below, so validation
            # and the grid cannot disagree after a post-publish override.
            chunked_prefill_size = get_schedule().chunked_prefill_size
            cps = chunked_prefill_size or 8192
            # Staging slices each send into a fixed page-aligned grid, so an
            # unbounded (-1) or non-page-aligned chunk size has no valid grid.
            if cps <= 0 or cps % page_size != 0:
                raise RuntimeError(
                    f"SGLANG_DISAGG_STAGING_BUFFER requires a positive "
                    f"chunked_prefill_size that is a multiple of page_size "
                    f"({page_size}); got {chunked_prefill_size}."
                )
            if self.pp_size > 1 and self.transfer_backend != TransferBackend.MOONCAKE:
                raise RuntimeError(
                    "SGLANG_DISAGG_STAGING_BUFFER with pp_size > 1 is only "
                    "supported by Mooncake."
                )
            if get_parallel().enable_prefill_context_parallel:
                # CP rewrites index_slice per rank, breaking the chunk grid.
                raise RuntimeError(
                    "SGLANG_DISAGG_STAGING_BUFFER does not support "
                    "prefill context parallelism."
                )
        self.kv_manager = self._init_kv_manager()
        initialize_waiting_restore_plan(scheduler)
        late_d = self.kv_manager.late_d
        self.late_d_enabled = late_d is not None
        if late_d is not None and late_d.is_leader:
            config = LateDConfig.from_disagg(
                get_disagg(), get_schedule().chunked_prefill_size
            )
            self.late_d_binder = LateDPrefillBinder(
                self.kv_manager,
                config,
                get_schedule().chunked_prefill_size,
                metrics=make_late_d_metrics(scheduler, late_d),
            )
            late_d.metrics = self.late_d_binder.metrics
            if late_d.mode == LATE_D_CHOOSE:
                # Stage 2: the router link; it also sends ADMIT and RELEASE.
                link = LateDLink(
                    self.late_d_binder,
                    get_disagg().disaggregation_late_d_control_port,
                    mode="pd",
                    kv_ip=self.kv_manager.local_ip,
                    choice_timeout_s=config.choice_timeout_s,
                    zmq_ctx=self.kv_manager._zmq_ctx,
                    metrics=self.late_d_binder.metrics,
                )
                late_d.use_sender(link)
                self.late_d_binder.link = link
                link.start()

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        kv_args.engine_rank = self.tp_rank
        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.ps.dp_rank
        kv_args.kv_cache_dtype_str = (
            self.scheduler.tp_worker.model_runner.kv_cache_dtype_str
        )
        layer_shard_enabled = getattr(
            self.token_to_kv_pool, "layer_shard_enabled", False
        )
        layer_shard_rank = getattr(self.token_to_kv_pool, "layer_shard_rank", None)
        layer_shard_size = getattr(self.token_to_kv_pool, "layer_shard_size", 1)
        transfer_draft_cache = (
            not layer_shard_enabled or layer_shard_rank == layer_shard_size - 1
        )
        kv_args.prefill_start_layer = (
            getattr(
                self.token_to_kv_pool,
                "layer_shard_start",
                self.token_to_kv_pool.start_layer,
            )
            if layer_shard_enabled
            else _transfer_start_layer(
                pool=self.token_to_kv_pool,
                hf_text_config=self.scheduler.model_config.hf_text_config,
            )
        )
        kv_args.mla_compression_ratios = None
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.token_to_kv_pool.get_contiguous_buf_infos()
        )
        kv_args.prefill_end_layer = (
            kv_args.prefill_start_layer + len(kv_data_ptrs)
            if layer_shard_enabled
            else getattr(self.token_to_kv_pool, "end_layer", None)
        )

        draft_kv_pool = self.draft_token_to_kv_pool if transfer_draft_cache else None
        num_draft_entries = 0
        if draft_kv_pool is not None:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                draft_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens
            num_draft_entries = len(draft_kv_data_ptrs)

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        kv_args.kv_layer_ids = build_kv_layer_ids(
            token_to_kv_pool=self.token_to_kv_pool,
            draft_token_to_kv_pool=draft_kv_pool,
            num_draft_entries=num_draft_entries,
            num_hidden_layers=self.scheduler.model_config.num_hidden_layers,
        )
        if not self.is_mla_backend:
            kv_args.kv_head_num = self.token_to_kv_pool.head_num
            kv_args.total_kv_head_num = (
                self.scheduler.model_config.get_total_num_kv_heads()
            )
        kv_args.page_size = self.token_to_kv_pool.page_size

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )
        kv_args.ib_device = get_disagg().disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.ps.gpu_id

        req_to_token_pool = getattr(self.scheduler, "req_to_token_pool", None)
        setup_state_kv_args(
            kv_args,
            self.token_to_kv_pool,
            self.draft_token_to_kv_pool if transfer_draft_cache else None,
            self.scheduler.model_config.num_hidden_layers,
            req_to_token_pool=req_to_token_pool,
        )

        if isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool):
            # V4's KVCache is organized by compression-ratio
            # buckets rather than by layer.
            kv_args.mla_compression_ratios = list(
                self.token_to_kv_pool.compression_ratios
            )

        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.PREFILL,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        # Pass KV pool tensor refs to the manager for GPU gather (staging mode)
        if (
            envs.SGLANG_DISAGG_STAGING_BUFFER.get()
            and hasattr(kv_manager, "set_kv_buffer_tensors")
            and not self.is_mla_backend
        ):
            kv_pool = self.token_to_kv_pool
            if hasattr(kv_pool, "full_kv_pool"):
                kv_pool = kv_pool.full_kv_pool
            staging_slots = build_staging_slot_metadata(
                kv_layer_ids=kv_args.kv_layer_ids,
                num_draft_entries=num_draft_entries,
                kv_pool=kv_pool,
                draft_kv_pool=draft_kv_pool,
            )
            if staging_slots is not None:
                k_buffers, v_buffers, slot_layer_ids = staging_slots
                kv_manager.set_kv_buffer_tensors(
                    k_buffers,
                    v_buffers,
                    kv_pool.page_size,
                    slot_layer_ids=slot_layer_ids,
                )
        return kv_manager

    def create_sender(self, req: Req, num_kv_heads: int) -> bool:
        """Create a KV sender for the request without enqueuing it.
        Returns False if the request was concluded instead: aborted at admission
        or larger than this prefill's KV pool."""
        if self._conclude_before_bootstrap(req):
            return False

        req.disagg_kv_sender = self._new_sender(req)
        if self.late_d_enabled and self._sender_backend(req) != TransferBackend.FAKE:
            # Late D, every rank: the room binds only after the leader's ADMIT
            # and uses the late timers. FAKE rooms (warmup) bind as today.
            req.disagg_kv_sender.late_d = True
            req.late_d_need = getattr(self.scheduler, "_late_d_need_enabled", False)
        self._process_req(req)
        req.pending_bootstrap = True
        return True

    def _sender_backend(self, req: Req) -> TransferBackend:
        return (
            TransferBackend.FAKE
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST
            else self.transfer_backend
        )

    def _new_sender(self, req: Req):
        kv_sender_class = get_kv_class(self._sender_backend(req), KVClassType.SENDER)
        return kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            dest_tp_ranks=[self.tp_rank],
            pp_rank=self.pp_rank,
            req_has_disagg_prefill_dp_rank=req.disagg_prefill_dp_rank is not None,
        )

    def _conclude_before_bootstrap(self, req: Req) -> bool:
        """Conclude a request this prefill will not run, before it bootstraps,
        and fail its room on decode. Returns True if it was concluded.

        Two cases: the scheduler aborted it at admission (``to_finish``, e.g. a
        forced output that no longer fits once max_new_tokens is clipped to the
        pool), or its prompt is larger than this prefill's KV pool. The decode
        twin got the full prompt and reserves KV for all of it, so it has to be
        told. Running the 1-token abort stub through bootstrap instead sends a
        1-token transfer into that full-prompt reservation; concluding with no
        sender leaves decode holding the reservation until its waiting timeout.
        """
        if is_aborted(req):
            abort = req.finished_reason
            if not isinstance(abort, FINISH_ABORT):
                abort, req.to_finish = req.to_finish, None
                req.finished_reason = abort
            message = abort.message
        elif len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
        else:
            return False

        req.time_stats.trace_ctx.abort(abort_info={"reason": message})
        self._fail_room_on_decode(req)
        self.scheduler.output_streamer.stream_output([req], req.return_logprob)
        if self.scheduler.enable_hicache_storage:
            # The storage prefetch started when the request was queued.
            self.scheduler.tree_cache.release_aborted_request(req.rid)
        # The request leaves without cache_finished_req: end its retention
        # registration's live count (soft pin).
        forget_kv_session_request(self.scheduler, req)
        return True

    def _fail_room_on_decode(self, req: Req) -> None:
        """Register the room with a sender and abort it at once. Nothing was
        enqueued, so the connector's room failure notice reaches decode (every
        registered decode endpoint while the room is still bootstrapping) and
        the room is tombstoned: a destination decode sends later is answered
        with Failed. Decode then drops the request from its preallocation
        queue, or frees the KV it already allocated, and returns an error."""
        if (
            self._sender_backend(req) == TransferBackend.FAKE
            or req.bootstrap_room is None
        ):
            return
        try:
            sender = self._new_sender(req)
            sender.abort()
        except Exception as e:
            logger.warning(
                f"Could not fail room {req.bootstrap_room} of {req.rid=} on "
                f"decode: {e}"
            )
            return
        try:
            # Concludes the room locally: clears its state and failure record.
            sender.failure_exception()
        except Exception:
            pass  # the abort reason it re-raises is expected

    def ensure_metadata_buffer(self, req: Req) -> bool:
        if req.metadata_buffer_index >= 0:
            return True

        if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
            return False
        req.metadata_buffer_index = self.req_to_metadata_buffer_idx_allocator.alloc()
        assert req.metadata_buffer_index is not None
        return True

    def finalize_bootstrap(self, req: Req) -> bool:
        """Initialize the sender after bootstrap completes.
        Returns False if no metadata buffer is available (non-terminal)."""
        assert req.pending_bootstrap, "finalize_bootstrap is not idempotent"
        if not self.ensure_metadata_buffer(req):
            return False

        req.time_stats.set_bootstrap_done_time()
        decode_prefix_len = req.disagg_kv_sender.pop_decode_prefix_len()
        num_kv_indices = len(req.origin_input_ids)
        req.start_send_idx = decode_prefix_len
        # Base of the staging chunk grid (suffix-relative send coordinates).
        req.disagg_decode_prefix_len = decode_prefix_len
        # The sched req_trace record (d_prefix_len).
        req.sched_trace_d_prefix_len = decode_prefix_len
        num_kv_indices_to_send = num_kv_indices - decode_prefix_len
        num_pages = kv_to_page_num(
            num_kv_indices_to_send,
            self.scheduler.token_to_kv_pool_allocator.page_size,
        )
        if self.tp_rank == 0:
            logger.debug(
                "PD prefill transfer plan: rid=%s bootstrap_room=%s "
                "origin_tokens=%d decode_prefix_len=%d start_send_idx=%d "
                "main_transfer_pages=%d page_size=%d",
                req.rid,
                req.bootstrap_room,
                num_kv_indices,
                decode_prefix_len,
                req.start_send_idx,
                num_pages,
                self.scheduler.token_to_kv_pool_allocator.page_size,
            )
        req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)
        req.pending_bootstrap = False
        if getattr(req, "late_d_need", False):
            req.late_d_bind_perf = time.perf_counter()
        if getattr(req, "late_d_need", False) and req.launched_unbound:
            req.late_d_catchup = True
            self.scheduler.late_d_need_send_prefix(req)
        return True

    def add(self, req: Req, num_kv_heads: int) -> None:
        if not self.create_sender(req, num_kv_heads):
            return
        self.queue.append(req)

    def extend(self, reqs: List[Req], num_kv_heads: int) -> None:
        for req in reqs:
            self.add(req, num_kv_heads)

    def _process_req(self, req: Req) -> None:
        """
        Set max_new_tokens = 1, so PrefillAdder memory estimation is accurate
        """
        if self.scheduler.sched_tracer is not None:
            req.sched_trace_requested_output_len = req.sampling_params.max_new_tokens
        req.sampling_params.max_new_tokens = 1

    @timed_phase("pop_bootstrapped")
    def pop_bootstrapped(
        self,
        return_failed_reqs: bool = False,
        pp_good_rids: Optional[List[str]] = None,
        pp_bad_rids: Optional[List[str]] = None,
    ) -> List[Req] | tuple[List[Req], List[Req]]:
        """
        pop the reqs which has finished bootstrapping

        return_failed_reqs: For PP, on rank 0, also return the failed reqs to notify the next rank
        pp_good_rids: RIDs that PP consensus determined as WaitingForInput.
        pp_bad_rids: RIDs that PP consensus determined as Failed.
        """

        bootstrapped_reqs = []
        failed_reqs = []
        indices_to_remove = set()

        if len(self.queue) == 0:
            if return_failed_reqs is False:
                return []
            else:
                return [], []

        if self.pp_size > 1:
            polls = poll_and_all_reduce_pp(
                (req.rid for req in self.queue),
                KVPoll.WaitingForInput,
                pp_good_rids,
                pp_bad_rids,
            )
            uncovered = [i for i, poll in enumerate(polls) if poll is None]
            if uncovered:
                local_polls = poll_and_all_reduce_attn_cp_tp_group(
                    [self.queue[i].disagg_kv_sender for i in uncovered],
                    self.scheduler.attn_cp_cpu_group,
                    self.scheduler.attn_tp_cpu_group,
                )
                for i, local_poll in zip(uncovered, local_polls):
                    if local_poll == KVPoll.Failed:
                        polls[i] = KVPoll.Failed
        else:
            polls = poll_and_all_reduce_attn_cp_tp_group(
                [req.disagg_kv_sender for req in self.queue],
                self.scheduler.attn_cp_cpu_group,
                self.scheduler.attn_tp_cpu_group,
            )

        for i, (req, poll) in enumerate(zip(self.queue, polls)):
            if poll is None:
                continue

            if poll == KVPoll.Failed:
                self.scheduler.handle_bootstrap_failure(req)
                indices_to_remove.add(i)
                failed_reqs.append(req)
            elif poll == KVPoll.Bootstrapping:
                if (getattr(req, "late_d_need", False) and not req.is_retracted
                        and not force_bound_for_smoke(req)):
                    # Test-only bound controls stay here until D has a
                    # destination, independent of how fast their P forward is.
                    # Retracted rooms must re-enter through WaitingForInput;
                    # only prepare_for_extend clears is_retracted.
                    if not self.ensure_metadata_buffer(req):
                        logger.info("LATE_D_NEED meta_full rid=%s", req.rid)
                        continue
                    bootstrapped_reqs.append(req)
                    indices_to_remove.add(i)
                    req.time_stats.set_wait_queue_entry_time()
                    req.late_d_entry_pass = self.scheduler.forward_ct
                    logger.info("LATE_D_NEED entry rid=%s t_perf=%.6f", req.rid,
                                time.perf_counter())
                    continue
                if (
                    req.prefill_attempt_count < get_disagg().optimistic_prefill_attempts
                    and not req.is_retracted  # engine paused
                ):
                    if not self.ensure_metadata_buffer(req):
                        continue  # no more metadata buffer
                    req.prefill_attempt_count += 1
                    bootstrapped_reqs.append(req)
                    indices_to_remove.add(i)
                    req.time_stats.set_wait_queue_entry_time()
            elif poll == KVPoll.WaitingForInput:
                if should_force_retry(req):  # skip checking for testing
                    if not self.ensure_metadata_buffer(req):
                        continue  # no more metadata buffer
                    req.prefill_attempt_count += 1
                elif not self.finalize_bootstrap(req):
                    continue
                bootstrapped_reqs.append(req)
                indices_to_remove.add(i)
                req.time_stats.set_wait_queue_entry_time()
            else:
                raise RuntimeError(
                    f"Unexpected poll state {poll} for req {req.rid} in pop_bootstrapped"
                )

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        if return_failed_reqs is False:
            return bootstrapped_reqs
        else:
            return bootstrapped_reqs, failed_reqs

    def release_memory_occupation(self):
        self.queue.clear()
        if hasattr(self.kv_manager, "deregister_buffer_to_engine"):
            self.kv_manager.deregister_buffer_to_engine()

    def resume_memory_occupation(self):
        if hasattr(self.kv_manager, "register_buffer_to_engine"):
            self.kv_manager.register_buffer_to_engine()


def init_bootstrap_prefetch(scheduler: Scheduler):
    """Create the opt-in bootstrap-window prefetcher for a disagg prefill loop.

    Returns None when off (the default) without touching anything else on the
    scheduler, so stand-in schedulers in tests need no stub. The mode is read
    and validated in ``Scheduler.init_disaggregation``.
    """
    if getattr(scheduler, "bootstrap_prefetch_mode", "off") == "off":
        return None
    from decoupled_engine.srt.disaggregation.bootstrap_prefetch import BootstrapPrefetcher

    scheduler.bootstrap_prefetch = BootstrapPrefetcher(scheduler)
    if getattr(scheduler, "_late_d_need_enabled", False):
        if not scheduler.bootstrap_prefetch.concurrent or scheduler.bootstrap_prefetch.depth < 2:
            raise ValueError("late D need mode requires concurrent bootstrap prefetch, depth >= 2")
    return scheduler.bootstrap_prefetch


class SchedulerDisaggregationPrefillMixin(NeedAdmissionMixin):
    """
    Mixin for Scheduler to handle disaggregation prefill
    """

    def maybe_prefetch_staging_for_batch(self: Scheduler, batch: ScheduleBatch) -> None:
        """Pre-send STAGING_REQ so decode allocates staging during GPU forward."""
        kv_mgr = self.disagg_prefill_bootstrap_queue.kv_manager
        prefetch = getattr(kv_mgr, "_prefetch_staging_reqs", None)
        if prefetch is None:
            return
        for req in batch.reqs:
            room = getattr(req, "bootstrap_room", None)
            if room is not None and room in kv_mgr.transfer_infos:
                prefetch(room)

    def resolve_waiting_queue_bootstrap(self: Scheduler) -> None:
        """Resolve bootstrap status for waiting prefill requests before admission.

        Covers the window between leaving the bootstrap queue and being admitted
        into a running batch: aborts requests whose decode peer died, and
        finalizes optimistic requests whose bootstrap completed so they skip
        the post-forward bootstrap check.
        """
        candidates = [req for req in self.waiting_queue if not is_aborted(req)]
        waiting_ids = {id(r) for r in candidates}
        need = getattr(self, "_late_d_need_enabled", False)
        if need:
            # Last/single chunks are in the in-flight batch. Failures wait
            # until the result is safe; only WaitingForInput acts in this lane.
            last = getattr(self, "last_batch", None)
            if last is not None:
                candidates.extend(r for r in last.reqs if id(r) not in waiting_ids
                                  and r is not self.chunked_req and not is_aborted(r)
                                  and getattr(r, "late_d_need", False)
                                  and r.pending_bootstrap)
        if not candidates:
            return
        if need:
            binder = self.disagg_prefill_bootstrap_queue.late_d_binder
            total = sum(max(0, len(r.origin_input_ids) - len(r.prefix_indices)
                            - r.host_hit_length) for r in self.waiting_queue)
            codes = [binder.need_start_code(r, total) if binder is not None else 255
                     for r in candidates]
            polls, codes = poll_and_all_reduce_attn_cp_tp_lanes(
                [r.disagg_kv_sender for r in candidates], codes,
                self.attn_cp_cpu_group, self.attn_tp_cpu_group)
        else:
            polls = poll_and_all_reduce_attn_cp_tp_group(
                [req.disagg_kv_sender for req in candidates],
                self.attn_cp_cpu_group, self.attn_tp_cpu_group)
            codes = [0] * len(candidates)
        failed = set()
        for req, poll, code in zip(candidates, polls, codes):
            if need:
                req.late_d_start_ok = bool(code & START_OK)
                req.late_d_hold = bool(code & HOLD)
            if poll == KVPoll.Failed and id(req) in waiting_ids:
                self.handle_bootstrap_failure(req)
                failed.add(req)
            elif (poll == KVPoll.WaitingForInput and req.pending_bootstrap
                  and not should_force_retry(req)):
                self.disagg_prefill_bootstrap_queue.finalize_bootstrap(req)
        if failed:
            self.waiting_queue = [req for req in self.waiting_queue if req not in failed]

    def late_d_first_chunk_start(self, batch):
        """All ranks mark the same admitted batch; only the leader signals D."""
        first_chunk = (getattr(self, "_late_d_need_enabled", False)
                       and get_disagg().disaggregation_late_d_allocation_trigger == "first_chunk")
        if not first_chunk and not envs.SGLANG_DECODE_SELECTION_METRICS.get():
            return
        started = []
        for req in batch.reqs:
            if req.late_d_first_chunk_started_perf == 0.0:
                req.late_d_first_chunk_started_perf = time.perf_counter()
                req.late_d_first_chunk_started_unix_ns = time.time_ns()
                started.append(req)
                logger.info("LATE_D first_chunk_start rid=%s forward_iter=%d unix_ns=%d pending=%d",
                            req.rid, batch.forward_iter, req.late_d_first_chunk_started_unix_ns, int(req.pending_bootstrap))
        binder = self.disagg_prefill_bootstrap_queue.late_d_binder
        if first_chunk and binder is not None and started:
            binder.first_chunk_started([r for r in started if r.pending_bootstrap])

    def late_d_bind_step(self: Scheduler) -> None:
        """The sole leader binds; admission and sends follow reduced state."""
        binder = self.disagg_prefill_bootstrap_queue.late_d_binder
        if binder is not None:
            tracked = self.disagg_prefill_bootstrap_queue.queue
            if getattr(self, "_late_d_need_enabled", False):
                tracked = self.late_d_need_tracked()
                binder.need_plan = getattr(self, "late_d_lookahead", None)
                binder.need_gate_closed = getattr(self, "_late_d_need_gate_closed", False)
                binder.need_parked = {r.bootstrap_room for r in self.disagg_prefill_inflight_queue
                                      if r.pending_bootstrap}
            binder.step(tracked, self.waiting_queue, self.chunked_req,
                        batch_ran=self.last_batch is not None)

    def has_bootstrapped_waiting_req(self: Scheduler) -> bool:
        return any(
            not req.pending_bootstrap and not is_aborted(req)
            for req in self.waiting_queue
        )

    @scheduler_nvtx_method("scheduler.get_next_batch_to_run")
    def get_next_disagg_prefill_batch_to_run(
        self: Scheduler,
        running_batch: ScheduleBatch,
        last_batch: Optional[ScheduleBatch],
    ) -> NextBatchPlan:
        self.process_pending_chunked_abort()

        # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
        # Otherwise, it hangs under high concurrency
        running_batch.batch_is_full = False

        self.resolve_waiting_queue_bootstrap()

        self.process_prefill_chunk(last_batch=last_batch, running_batch=running_batch)

        prefill_plan = self.get_new_batch_prefill(running_batch)
        batch = prefill_plan.batch_to_run
        running_batch = prefill_plan.running_batch
        batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(batch)

        if batch:
            set_schedule_time_batch(batch)

        return NextBatchPlan(batch_to_run=batch, running_batch=running_batch)

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self: Scheduler) -> None:
        """A normal scheduler loop for prefill worker in disaggregation mode."""
        if cpu_page_bookkeeping_enabled() or load_quarantine_enabled():
            # M4 runs only in the overlap loop; a requested flag must not be inert.
            raise ValueError(
                "SGLANG_PREFILL_CPU_PAGE_BOOKKEEPING / "
                "SGLANG_DSA_HOST_DEDUP_LOAD_QUARANTINE need the overlap prefill loop"
            )
        bootstrap_prefetch = init_bootstrap_prefetch(self)
        queued_prefix_guard = getattr(self, "queued_prefix_guard", None)
        # SGLANG_PREFILL_STATE_PUSH_PORT / _ENDPOINT: None when off.
        state_push = getattr(self, "prefill_state_pusher", None)
        # SGLANG_KV_EVENTS_BATCH_PUBLISH: one KV-event flush per pass, after
        # the result. None when off (no call).
        kv_events_end_pass = kv_events_end_pass_hook(self)
        while True:
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue
            self.late_d_bind_step()
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            if bootstrap_prefetch is not None:
                # SGLANG_BWP_POLL_FIRST: settle landed prefetch ops first.
                bootstrap_prefetch.poll_before_admission()

            # Get the next batch to run
            plan = self.get_next_disagg_prefill_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch
            if bootstrap_prefetch is not None:
                bootstrap_prefetch.step(batch)
            if queued_prefix_guard is not None:
                queued_prefix_guard.step(batch)

            # Launch the current batch
            self.pre_run_batch_consensus_check(batch)
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                result = self.run_batch(batch)
                if state_push is not None:
                    # Sent now: the CPU waits for this result next.
                    state_push.on_launch(batch)
                    state_push.flush(self, batch)
                self.process_batch_result(batch, result)
            else:
                self._sched_idled = True
                self.on_idle()
            if kv_events_end_pass is not None:
                # Right after the result, where report_prefill_stats took the
                # events before: they still go out ahead of the routing sample
                # and the result's state push.
                kv_events_end_pass()

            self.sample_prefill_routing(batch)
            self.process_disagg_prefill_inflight_queue()
            if state_push is not None:
                # The processed result, or a heartbeat when idle. In this loop
                # the GPU is idle here, so the message adds to the time
                # between passes.
                state_push.flush(self, batch)

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler) -> None:
        self.result_queue = deque()
        host_prefetch = None
        if envs.SGLANG_HICACHE_PREFETCH_NEXT_BATCH.get():
            from decoupled_engine.srt.managers.scheduler_components.next_batch_host_prefetch import (
                NextBatchHostPrefetch,
            )

            host_prefetch = NextBatchHostPrefetch(self)
        self.next_batch_host_prefetch = host_prefetch
        # Window chunk-boundary backups queued by process_prefill_chunk's stash
        # and run after the launch (SGLANG_DSV4_CHUNK_WINDOW_WORK_AFTER_LAUNCH).
        chunk_backups_after_launch = (
            envs.SGLANG_DSV4_CHUNK_WINDOW_WORK_AFTER_LAUNCH.get()
            and start_deferring_chunk_backups(self.tree_cache)
        )
        bootstrap_prefetch = init_bootstrap_prefetch(self)
        queued_prefix_guard = getattr(self, "queued_prefix_guard", None)
        self.init_prefill_unserialize_loop()
        self.init_cpu_page_book()
        # SGLANG_DEBUG_FORCE_GATE_REORDER_RANKS (positive control): loud line.
        log_debug_force_gate_reorder(self, "pd")
        # The early-result triggers: gate pending, and with
        # SGLANG_PREFILL_RESULT_FIRST_WHEN_READY a finished previous forward.
        result_first = PrefillResultFirst.for_scheduler(self, "pd")
        self._prefill_result_first = result_first
        # SGLANG_PREFILL_LAUNCH_THREAD: the forward block is launched on a
        # worker thread while this thread processes the previous result
        # (prefill_launch_thread.py). None when off.
        launch_thread = PrefillLaunchThread.for_scheduler(self)
        self._prefill_launch_thread = launch_thread
        # SGLANG_PREFILL_STATE_PUSH_PORT / _ENDPOINT: None when off.
        state_push = getattr(self, "prefill_state_pusher", None)
        # SGLANG_PREFILL_STATE_PUSH_LAUNCH_MODE=always: the launch message goes
        # out at every launch, also for the first pass after an idle GPU.
        push_every_launch = getattr(state_push, "launch_always", False) is True
        # SGLANG_KV_EVENTS_BATCH_PUBLISH: the KV events of the previous result
        # (processed before, during or after this pass's launch) go out once,
        # after that result and the launch, outside the launch-thread window,
        # ahead of the routing sample and the state push. None when off (no
        # call).
        kv_events_end_pass = kv_events_end_pass_hook(self)

        while True:
            self._m2_begin_iteration()
            self._q4_begin_iteration()
            if host_prefetch is not None:
                host_prefetch.finish()
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                if host_prefetch is not None:
                    host_prefetch.release()
                self._m2_flush_deferred_writes()
                continue
            self.late_d_bind_step()
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            if bootstrap_prefetch is not None:
                # SGLANG_BWP_POLL_FIRST: settle landed prefetch ops first.
                bootstrap_prefetch.poll_before_admission()

            # Get the next batch to run
            plan = self.get_next_disagg_prefill_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch
            if host_prefetch is not None:
                host_prefetch.release(batch)
                host_prefetch.prepare(batch)
            if bootstrap_prefetch is not None:
                bootstrap_prefetch.step(batch)
            if queued_prefix_guard is not None:
                queued_prefix_guard.step(batch)
            # M4 C1: from here on (the gate-pending reorder may process the
            # previous result before or after the launch, per rank) releases
            # and unlocks carry the epoch of this iteration's launch.
            self._lq_set_floor(batch)

            # Launch the current batch
            self.pre_run_batch_consensus_check(batch)
            last_processed = False
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                # 6h: if the branch below processes the previous result first
                # (per rank), the flush after the launch keeps its writes for
                # the next flush, as on ranks that process it after the launch.
                self._m2_mark_pre_launch_writes()
                # Gate pending: the launch will wait for the shared loader to
                # submit this batch's load (which itself waits for the
                # previous forward). Ready: the previous forward has finished
                # (SGLANG_PREFILL_RESULT_FIRST_WHEN_READY). Either way, finish
                # the previous batch (KV send to D) first. Per rank
                # (prefill_result_first.py); collective order unchanged.
                trigger = None
                concurrent_prev = None
                push_only = None
                if self.last_batch:
                    if launch_thread is not None and launch_thread.applies(batch):
                        # Processed while the worker launches this batch: for
                        # the loop, the early path on every rank.
                        concurrent_prev = self._process_prev_result_concurrently
                        if state_push is not None:
                            # Opt-in state push: first send this launch, if
                            # the previous forward still runs or in "always"
                            # launch mode (launch_now).
                            concurrent_prev = state_push.launch_now_then(
                                self, batch, concurrent_prev
                            )
                    else:
                        trigger = result_first.decide(
                            gate_pending_for_batch(self, batch), self.result_queue
                        )
                elif (
                    push_every_launch
                    and launch_thread is not None
                    and launch_thread.applies(batch)
                ):
                    # State push "always" launch mode, first pass after an idle
                    # GPU: no previous result to process, but the worker thread
                    # still launches this batch, and this thread only sends
                    # the launch message meanwhile (never in front of the
                    # launch). Push rank only; last_processed stays False.
                    push_only = state_push.launch_now_then(self, batch)
                if trigger is not None:
                    _t_rp = PHASES.start()
                    tmp_batch, tmp_result = self.result_queue.popleft()
                    self.process_batch_result(tmp_batch, tmp_result)
                    PHASES.stop("reorder_prev_result", _t_rp)  # RP (R9)
                    last_processed = True
                    result_first.processed(trigger, _t_rp)
                if concurrent_prev is not None:
                    # 6b: the previous result's page-id reads (send_kv_chunk)
                    # run during this launch. An event recorded before it, after
                    # this batch's admission writes, lets them read on the side
                    # stream instead of behind the forward being launched
                    # (send_kv took 135 ms mean without it).
                    self._m2_mark_pre_barrier()
                    batch_result = self.run_batch(
                        batch, concurrent_work=concurrent_prev
                    )
                    last_processed = True
                elif push_only is not None:
                    batch_result = self.run_batch(batch, concurrent_work=push_only)
                else:
                    batch_result = self.run_batch(batch)
                if getattr(self, "_late_d_need_enabled", False):
                    for req in batch.reqs:
                        if req.late_d_need and req.late_d_forward_done_event is None:
                            req.late_d_forward_done_event = batch_result.copy_done
                if host_prefetch is not None:
                    host_prefetch.submit()
                if chunk_backups_after_launch:
                    # Overlaps the forward just launched; before its barrier.
                    self.tree_cache.run_deferred_chunk_backups()
                # 6h: writes deferred from the previous result run here, ahead
                # of this barrier: their index D2H waits only for the previous
                # forward, which processing its result waits for anyway.
                self._m2_flush_deferred_writes(after_early_result=last_processed)
                # 6b: rows as of this launch, readable without the barrier.
                self._m2_mark_pre_barrier()
                self._apply_war_barrier()
                self._m2_defer_writes_behind_barrier()
                self.result_queue.append((batch.copy(), batch_result))
                # 6d: early sends prepared in run_batch, at a fixed loop point.
                self._m2_flush_early_sends()
                if state_push is not None:
                    # launch_now (run_batch or the concurrent work) noted this
                    # launch, and sent it if the previous forward still ran
                    # ("always" launch mode: at every launch; launch_after in
                    # run_batch). A path without it notes the launch here.
                    # Launches not sent yet go out at the end of this iteration.
                    state_push.on_launch(batch)
            else:
                batch_result = None
                self._sched_idled = True
                # No new barrier this iteration: writes deferred from the
                # previous result can be submitted.
                self._m2_flush_deferred_writes()
                if chunk_backups_after_launch:
                    self.tree_cache.run_deferred_chunk_backups()
                # 6h: the result processed below queues its writes too; they
                # are submitted after it (flush below), so their index D2H
                # never runs inside result processing.
                self._m2_defer_idle_result_writes()

            # Process the last batch
            if self.last_batch and not last_processed:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self._lq_on_idle()
                self.on_idle()
            if batch is None:
                # 6h: submit the idle iteration's result writes. That result's
                # forward has finished (processing waited for it) and no new
                # one was launched, so the index D2H waits for nothing.
                self._m2_flush_deferred_writes()
            if kv_events_end_pass is not None:
                # The previous result is processed on every path here (and
                # the launch has returned): its events go out now, ahead of
                # the routing sample and the state push, as when
                # report_prefill_stats took them.
                kv_events_end_pass()

            if getattr(self, "_late_d_need_enabled", False):
                self.late_d_need_flush_catchup()
            if getattr(self, "late_d_lookahead", None) is not None:
                self.late_d_lookahead.update()
            self.sample_prefill_routing(batch)
            self.process_disagg_prefill_inflight_queue()
            if state_push is not None:
                # Results and launches not sent yet, or a heartbeat when idle;
                # it reuses the routing sample just taken (one build per pass).
                state_push.flush(self, batch)

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result, batch)

            # Update last_batch
            self.last_batch = batch

    def _process_prev_result_concurrently(self: Scheduler) -> None:
        """SGLANG_PREFILL_LAUNCH_THREAD: process the queued previous result
        (run_batch calls this while the worker launches the next forward)."""
        tmp_batch, tmp_result = self.result_queue.popleft()
        self.process_batch_result(tmp_batch, tmp_result)

    # ---- M2: unserialize the overlap loop (prefill_loop_unserialize.py) ----

    def init_prefill_unserialize_loop(self: Scheduler) -> None:
        """Resolve the M2 flags once for the overlap prefill loop. With every
        flag off this sets inert defaults and the loop runs as before."""
        cfg = PrefillUnserializeConfig.from_env()
        self._m2_cfg = cfg
        self._prefill_copy_draft_seed = bool(
            cfg.async_draft_d2h and self.spec_algorithm.is_eagle()
        )
        if (getattr(self, "_late_d_need_enabled", False) and self.spec_algorithm.is_eagle()
                and not self._prefill_copy_draft_seed):
            raise ValueError("late D need mode requires host-copied EAGLE seeds; "
                             "enable SGLANG_PREFILL_UNSERIALIZE_LOOP")
        self._kv_page_reader = None
        self._early_send_async = False
        self._pending_early_sends: List[PendingEarlySend] = []
        self._m2_write_controller = None
        # 6h: the writes queued before the launch block
        # (_m2_mark_pre_launch_writes), for the flush after the launch.
        self._m2_pre_launch_writes = None
        self._m2_lockstep = LockstepStats()  # inert until 6h is on
        self._m2_pre_launch_acks = None
        self._m2_stats_logger = None
        if not cfg.any_enabled:
            self._m2_init_lockstep()
            return
        from decoupled_engine.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
        from decoupled_engine.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

        is_cuda_device = torch.cuda.is_available() and self.device != "cpu"
        allocator = self.token_to_kv_pool_allocator
        # 6b covers the plain paged pool (GLM-5.2 DSA: KV, indexer and draft
        # share its page index space). Unified memory (virtual ids), SWA,
        # mamba and staging grids keep the legacy read.
        plain_paged_pool = (
            isinstance(allocator, PagedTokenToKVPoolAllocator)
            and type(allocator).translate_kv_indices_for_transfer
            is BaseTokenToKVPoolAllocator.translate_kv_indices_for_transfer
            and not getattr(self, "enable_unified_memory", False)
        )
        if (
            cfg.kv_page_snapshot
            and is_cuda_device
            and not self.enable_staging
            and plain_paged_pool
        ):
            self._kv_page_reader = KvPageIdReader(
                self.device_module,
                self.req_to_token_pool,
                allocator.page_size,
                check=cfg.kv_page_snapshot_check,
            )
        # The snapshot gathers raw page ids, so it needs the same plain pool.
        self._early_send_async = bool(
            cfg.early_send_async
            and is_cuda_device
            and not self.enable_staging
            and plain_paged_pool
        )
        cc = getattr(self.tree_cache, "cache_controller", None)
        if (
            cfg.defer_write_submit
            and cc is not None
            # write_back waits synchronously for its own acks during eviction.
            and getattr(cc, "write_policy", None) != "write_back"
            # DSv4 window residency drains spill acks from ack_write_queue.
            and not getattr(self.tree_cache, "_swa_window", False)
        ):
            self._m2_write_controller = cc
        # M2_LOCKSTEP hazard counters (TP0 line every 60 s).
        self._m2_init_lockstep()
        self._m2_stats_logger = StatsLogger(self.ps.tp_rank)
        if self.ps.tp_rank == 0:
            logger.info(
                "M2 prefill loop: %s copy_draft_seed=%s kv_page_reader=%s "
                "early_send_async=%s defer_write_submit=%s",
                cfg,
                self._prefill_copy_draft_seed,
                self._kv_page_reader is not None,
                self._early_send_async,
                self._m2_write_controller is not None,
            )

    def _m2_begin_iteration(self: Scheduler) -> None:
        reader = getattr(self, "_kv_page_reader", None)
        if reader is not None:
            reader.begin_iteration()
        stats_logger = getattr(self, "_m2_stats_logger", None)
        if stats_logger is not None:
            stats_logger.maybe_log()

    def _m2_mark_pre_barrier(self: Scheduler) -> None:
        reader = getattr(self, "_kv_page_reader", None)
        if reader is not None:
            reader.mark_pre_barrier(self.schedule_stream)

    def _m2_defer_writes_behind_barrier(self: Scheduler) -> None:
        cc = getattr(self, "_m2_write_controller", None)
        if cc is not None:
            cc.defer_start_writing = True

    def _m2_defer_idle_result_writes(self: Scheduler) -> None:
        """6h, idle iteration: hold the writes of the result processed in this
        iteration; the loop flushes them right after processing, outside the
        result_prefill site (the in-site submit was a torch-flagged index
        D2H)."""
        cc = getattr(self, "_m2_write_controller", None)
        if cc is not None:
            cc.defer_start_writing = True

    def _m2_init_lockstep(self: Scheduler) -> None:
        """The M2_LOCKSTEP counters (m2_lockstep_stats.py): on with 6h, and
        with SGLANG_DEBUG_FORCE_GATE_REORDER_RANKS whatever 6h is (the
        positive control runs it with 6h off)."""
        six_h = getattr(self, "_m2_write_controller", None) is not None
        if not six_h and not debug_force_gate_reorder_ranks():
            return
        tree_cache = getattr(self, "tree_cache", None)
        if getattr(tree_cache, "cache_controller", None) is None:
            return
        self._m2_lockstep = LockstepStats.for_scheduler(self, "pd", six_h=six_h)

    def _m2_mark_pre_launch_writes(self: Scheduler) -> None:
        """6h, right before the gate-pending branch: hold write-throughs and
        remember how many are queued. The branch processes the previous
        result before the launch on some ranks only (gate_pending_for_batch
        reads the rank-local loader state). On those ranks the flush after
        the launch (after_early_result) submits only the writes queued before
        that processing, and the result's writes wait for the next flush, as
        on a rank that processes the result after the launch. Each flush
        merges its writes into one op with one write ack, and writing_check
        settles acks by a MIN-reduced count, so every rank must merge the
        same writes. Otherwise the ranks settle different nodes, then evict
        different ones, and C2g loads get different device pages per rank.
        SGLANG_M2_WRITE_MERGE_LOCKSTEP=0: only remember (the
        old merge; the M2_LOCKSTEP counters still count)."""
        lockstep = getattr(self, "_m2_lockstep", None) or _NO_LOCKSTEP
        cc = getattr(self, "_m2_write_controller", None)
        if cc is not None:
            if lockstep.fix:
                cc.defer_start_writing = True
            self._m2_pre_launch_writes = len(cc.write_queue)
        elif lockstep.loop:
            # 6h off (counters on with the debug knob): an early processing
            # submits its writes inline, one op each.
            acks = self.tree_cache.cache_controller.ack_write_queue
            self._m2_pre_launch_acks = len(acks)
        lockstep.mark()

    def _m2_flush_deferred_writes(
        self: Scheduler, after_early_result: bool = False
    ) -> None:
        """6h: submit the held writes as one op. ``after_early_result``: the
        launch just made followed an early result processing; submit only the
        writes queued before it (_m2_mark_pre_launch_writes)."""
        cc = getattr(self, "_m2_write_controller", None)
        if cc is None:
            if after_early_result:
                self._m2_count_inline_early_result()
            return
        mark = getattr(self, "_m2_pre_launch_writes", None)
        self._m2_pre_launch_writes = None
        cc.defer_start_writing = False
        queued = len(cc.write_queue)
        limit = None
        if after_early_result and mark is not None:
            lockstep = getattr(self, "_m2_lockstep", None) or _NO_LOCKSTEP
            prefix = min(mark, queued)
            if lockstep.fix:
                limit = prefix
            lockstep.early_flush(
                queued, prefix, kept=0 if limit is None else queued - limit
            )
        n = queued if limit is None else limit
        if n:
            M2_STATS["write_flushes"] += 1
            M2_STATS["writes_flushed"] += n
            if limit is None:
                cc.start_writing()
            else:
                cc.start_writing(limit=n)
        if limit is not None and cc.write_queue:
            M2_STATS["writes_held_after_early_result"] += len(cc.write_queue)

    def _m2_count_inline_early_result(self: Scheduler) -> None:
        """6h off: count an early result processing and the write ops it
        submitted inline (M2_LOCKSTEP, with the debug knob)."""
        lockstep = getattr(self, "_m2_lockstep", None) or _NO_LOCKSTEP
        mark = getattr(self, "_m2_pre_launch_acks", None)
        self._m2_pre_launch_acks = None
        if not lockstep.loop or mark is None:
            return
        acks = self.tree_cache.cache_controller.ack_write_queue
        lockstep.early_flush(queued=max(0, len(acks) - mark), prefix=0, kept=0)

    def _m2_flush_early_sends(self: Scheduler) -> None:
        pending = getattr(self, "_pending_early_sends", None)
        if not pending:
            return
        self._pending_early_sends = []

        def send(p: PendingEarlySend) -> None:
            p.req.disagg_kv_sender._early_send_wait_event = p.wait_event
            self.send_kv_chunk(
                p.req, last_chunk=False, end_idx=p.end_idx, early_send_snapshot=p
            )

        flush_pending_early_sends(pending, send, is_aborted)

    # ---- M4: CPU page bookkeeping (Q4) + load quarantine ----

    def init_cpu_page_book(self: Scheduler) -> None:
        """Q4 (mem_cache/cpu_page_book.py) and the load quarantine
        (mem_cache/load_quarantine.py) on the overlap prefill loop. Both
        default off; requested but out of scope is a startup error (there is
        no fallback path)."""
        self._cpu_page_book = None
        self._load_quarantine = None
        book_on = cpu_page_bookkeeping_enabled()
        quarantine_on = load_quarantine_enabled()
        if quarantine_on and not book_on:
            raise ValueError(
                "SGLANG_DSA_HOST_DEDUP_LOAD_QUARANTINE requires "
                "SGLANG_PREFILL_CPU_PAGE_BOOKKEEPING=1"
            )
        if not book_on:
            return
        reason = cpu_page_book_supported_reason(self)
        if reason is not None:
            raise ValueError(
                f"SGLANG_PREFILL_CPU_PAGE_BOOKKEEPING is not supported here: {reason}"
            )
        clock = QuarantineClock() if quarantine_on else None
        book = PrefillCpuPageBook(
            self.token_to_kv_pool_allocator,
            self.req_to_token_pool,
            self.tree_cache,
            clock,
            shadow_check_every=envs.SGLANG_LOAD_QUARANTINE_SHADOW_CHECK_EVERY.get(),
            tp_rank=self.ps.tp_rank,
        )
        if quarantine_on:
            book.quarantine = LoadQuarantine(
                clock, book.allocator, self.tree_cache, self.device_module
            )
            self._load_quarantine = book.quarantine
        self._cpu_page_book = book
        # KV sends read the CPU rows (in place of M2's 6b reader).
        self._kv_page_reader = book.reader
        if self.ps.tp_rank == 0:
            logger.info(
                "M4 Q4 CPU page bookkeeping: on (load quarantine %s, shadow check "
                "every %d iterations)",
                "on" if quarantine_on else "off",
                book.shadow_check_every,
            )

    def _q4_begin_iteration(self: Scheduler) -> None:
        book = getattr(self, "_cpu_page_book", None)
        if book is not None:
            book.begin_iteration()

    def _lq_set_floor(self: Scheduler, batch) -> None:
        lq = getattr(self, "_load_quarantine", None)
        if lq is not None:
            lq.clock.set_floor(self.forward_ct, bool(batch))

    def _lq_on_idle(self: Scheduler) -> None:
        lq = getattr(self, "_load_quarantine", None)
        if lq is None:
            return
        if lq.clock.completed < self.forward_ct:
            # A launch whose result was not processed through copy_done:
            # nothing is queued now, so waiting here is free.
            self.forward_stream.synchronize()
        lq.clock.on_all_processed(self.forward_ct)

    def _lq_on_result_processed(self: Scheduler, batch, result) -> None:
        """M4 outside the PD prefill loop (symmetric): the generic result
        processors waited for ``copy_done``, recorded after the batch's whole
        forward (target, indexer, draft); this one more wait returns at once.
        A result without ``copy_done`` (no forward) leaves the clock alone."""
        lq = getattr(self, "_load_quarantine", None)
        if lq is None:
            return
        copy_done = getattr(result, "copy_done", None)
        if copy_done is None:
            return
        copy_done.synchronize()
        lq.clock.on_result_processed(getattr(batch, "forward_iter", None))

    @timed_phase("result_prefill")
    @sync_debug_site("result_prefill")
    def process_batch_result_disagg_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        """
        Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        Adapted from process_batch_result_prefill
        """
        (
            logits_output,
            next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
            copy_done,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
            result.copy_done,
        )

        # D4a sub-phase timers (SGLANG_SCHED_PHASE_TIMERS): "result_*" phases
        # below split this function; "result_stamp_offset_ms" counts, per
        # finished request, the time from entry to its prefill_finished stamp.
        _t_entry = PHASES.start()
        _t_phase = _t_entry
        if copy_done is not None:
            copy_done.synchronize()
            # M4: this batch's forward is done (copy_done follows it).
            load_quarantine = getattr(self, "_load_quarantine", None)
            if load_quarantine is not None:
                load_quarantine.clock.on_result_processed(
                    getattr(batch, "forward_iter", None)
                )
        PHASES.stop("result_copy_done_sync", _t_phase)
        state_push = getattr(self, "prefill_state_pusher", None)
        if state_push is not None and copy_done is not None:
            # Opt-in state push: stamp this result just after the copy sync
            # (one non-blocking event query; the message is built later).
            # Without copy_done, the token read below is the first wait.
            state_push.on_result(batch, getattr(self, "sched_tracer", None))
        auxiliary_output_starts = (
            self.batch_result_processor.snapshot_auxiliary_output_starts(batch, result)
        )
        auxiliary_output = result.auxiliary_host_output
        if result.routed_experts_output is not None:
            result.routed_experts_output.finalize()
            result.routed_experts_output = None
        if result.indexer_topk_output is not None:
            result.indexer_topk_output.finalize()
            result.indexer_topk_output = None

        logprob_pt = 0
        assert batch.spec_info is result.next_draft_input
        draft_input = result.next_draft_input
        draft_hidden_states_cpu = None
        draft_dsa_topk_indices_cpu = None
        draft_topk_p_cpu = None
        draft_topk_index_cpu = None
        _t_phase = PHASES.start()
        if self.spec_algorithm.is_eagle() and draft_input is not None:
            if getattr(result, "draft_hidden_states_cpu", None) is not None:
                # 6a: copied with the async result D2H at launch; copy_done
                # (synchronized above) covers it. No read behind the barrier.
                draft_hidden_states_cpu = result.draft_hidden_states_cpu
                draft_dsa_topk_indices_cpu = result.draft_dsa_topk_indices_cpu
                # set_buf copies top-k p / index into the metadata buffers; a
                # GPU source there is one more blocking D2H when they live on
                # the host (Mooncake default), so hand it host copies.
                topk_buf = getattr(self.disagg_metadata_buffers, "output_topk_p", None)
                if (getattr(self, "_late_d_need_enabled", False) or
                        (topk_buf is not None and topk_buf.device.type == "cpu")):
                    draft_topk_p_cpu = result.draft_topk_p_cpu
                    draft_topk_index_cpu = result.draft_topk_index_cpu
                M2_STATS["draft_seed_async"] += 1
            else:
                draft_hidden_states_cpu = draft_input.hidden_states.to(
                    "cpu", non_blocking=False
                )
                if batch.spec_info.dsa_topk_indices is not None:
                    draft_dsa_topk_indices_cpu = batch.spec_info.dsa_topk_indices.to(
                        "cpu", non_blocking=False
                    )
        PHASES.stop("S1_S2_draft_d2h", _t_phase)
        # Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        _t_phase = PHASES.start()
        next_token_ids = result.next_token_ids.tolist()
        if state_push is not None and copy_done is None:
            # No copy_done (non-overlap path without an auxiliary output):
            # the blocking token read above waited for this pass.
            state_push.on_result(batch, getattr(self, "sched_tracer", None))
        self.batch_result_processor.move_logprobs_to_cpu(
            batch=batch,
            logits_output=logits_output,
        )
        PHASES.stop("result_tokens_logprobs_d2h", _t_phase)
        _n_stamped = 0

        def advance_logprob_pt(i: int, req: Req) -> None:
            nonlocal logprob_pt
            if not req.return_logprob or extend_input_len_per_req is None:
                return
            extend_logprob_start_len = extend_logprob_start_len_per_req[i]
            extend_input_len = extend_input_len_per_req[i]
            if extend_logprob_start_len < extend_input_len:
                logprob_pt += extend_input_len - extend_logprob_start_len

        for i, (req, next_token_id) in enumerate(
            zip(batch.reqs, next_token_ids, strict=True)
        ):
            if req.inflight_middle_chunks <= 0:
                req.time_stats.set_prefill_finished_time()
                if PHASES.enabled:
                    PHASES.count(
                        "result_stamp_offset_ms",
                        (time.perf_counter() - _t_entry) * 1e3,
                    )
                    PHASES.count("result_stamp_index_in_batch", _n_stamped)
                    _n_stamped += 1

                # Test hook: exercise the release/requeue retry path.
                if req.pending_bootstrap and should_force_retry(req):
                    self.optimistic_release_and_requeue(req)
                    advance_logprob_pt(i, req)
                    continue

                req.output_ids.append(next_token_id)
                _t_phase = PHASES.start()
                maybe_cache_unfinished_req(req, self.tree_cache)
                PHASES.stop("result_req_tree_insert", _t_phase)
                # The NOTICE stamp (the reader's TP0 keeps the
                # records; the yield paths above and below never reach it).
                notice = getattr(self.tree_cache, "peer_notice_prefill_end", None)
                if notice is not None:
                    _t_phase = PHASES.start()
                    notice(req)
                    PHASES.stop("result_req_peer_notice", _t_phase)
                self.disagg_prefill_inflight_queue.append(req)
                if getattr(req, "late_d_need", False) and req.pending_bootstrap:
                    req.late_d_park_started = time.monotonic()
                    logger.info("LATE_D_NEED park rid=%s tokens=%d", req.rid,
                                len(req.origin_input_ids))
                _t_phase = PHASES.start()
                if self.spec_algorithm.is_eagle() and draft_input is not None:
                    if (
                        draft_topk_p_cpu is not None
                        and draft_topk_index_cpu is not None
                    ):
                        req.output_topk_p = draft_topk_p_cpu[i].clone()
                        req.output_topk_index = draft_topk_index_cpu[i].clone()
                    else:
                        req.output_topk_p = draft_input.topk_p[i]
                        req.output_topk_index = draft_input.topk_index[i]
                    req.hidden_states_tensor = draft_hidden_states_cpu[i].clone()
                    if draft_dsa_topk_indices_cpu is not None:
                        req.output_dsa_topk_indices = draft_dsa_topk_indices_cpu[
                            i
                        ].clone()
                    else:
                        req.output_dsa_topk_indices = None
                else:
                    req.hidden_states_tensor = None
                    req.output_dsa_topk_indices = None
                PHASES.stop("result_req_draft_copy", _t_phase)
                _t_phase = PHASES.start()
                if req.return_logprob:
                    assert extend_logprob_start_len_per_req is not None
                    assert extend_input_len_per_req is not None
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    num_input_logprobs = extend_input_len - extend_logprob_start_len
                    self.batch_result_processor.logprob_result_processor.add_logprob_return_values(
                        i,
                        req,
                        logprob_pt,
                        next_token_ids,
                        num_input_logprobs,
                        logits_output,
                    )
                    logprob_pt += num_input_logprobs
                if req.return_sampling_mask:
                    self.batch_result_processor.add_sampling_mask_return_values(
                        i, req, logits_output
                    )
                PHASES.stop("result_req_logprob", _t_phase)
                if not req.pending_bootstrap:
                    _t_phase = PHASES.start()
                    self.send_kv_chunk(req, last_chunk=True)
                    PHASES.stop("result_req_send_kv", _t_phase)
                req.time_stats.set_prefill_transfer_queue_entry_time()

                if req.grammar is not None:
                    try:
                        req.grammar.accept_token(next_token_id)
                    except ValueError as e:
                        error_message = f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                        prepare_abort(
                            req,
                            error_message,
                            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        )
                    req.grammar.finished = req.finished()
            else:
                # being chunked reqs' prefill is not finished
                req.inflight_middle_chunks -= 1

                # Still chunking iff its next chunk was launched: either it is
                # still self.chunked_req, or its final chunk (extend_range
                # reaching the end of the input) is in flight. A yielded req
                # is neither, so do its deferred release here.
                still_chunking = self.chunked_req is req or (
                    req.extend_range is not None
                    and req.extend_range.end >= len(req.origin_input_ids)
                )
                if (req.pending_bootstrap and not still_chunking
                        and not getattr(req, "late_d_need", False)):
                    self.optimistic_release_and_requeue(req)
                    advance_logprob_pt(i, req)
                    req.time_stats.set_last_chunked_prefill_finish_time()
                    continue

                # Optimistic bootstrap can fail while this overlapped chunk is
                # already running. Drop aborted chunks instead of sending KV.
                if is_aborted(req):
                    self.clear_pending_chunk_send(req)
                    advance_logprob_pt(i, req)
                    req.time_stats.set_last_chunked_prefill_finish_time()
                    continue

                if req.return_logprob:
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    if extend_logprob_start_len < extend_input_len:
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.batch_result_processor.logprob_result_processor.add_input_logprob_return_values(
                            i,
                            req,
                            logits_output,
                            logprob_pt,
                            num_input_logprobs,
                            last_prefill_chunk=False,
                        )
                        logprob_pt += num_input_logprobs

                # In non-overlap-mode, KV is sent in process_prefill_chunk
                # Only send when req's sender is initialized
                if getattr(req, "late_d_need", False):
                    req.late_d_computed_end = max(req.late_d_computed_end, req.tmp_end_idx)
                if (self.enable_overlap and not req.pending_bootstrap
                        and not getattr(req, "late_d_catchup", False)):
                    assert (
                        req.metadata_buffer_index >= 0
                    ), f"Req {req.rid} does not have metadata buffer allocated"
                    _t_phase = PHASES.start()
                    self.send_kv_chunk(req, last_chunk=False, end_idx=req.tmp_end_idx)
                    PHASES.stop("result_chunk_send_kv", _t_phase)
                req.time_stats.set_last_chunked_prefill_finish_time()

        _t_phase = PHASES.start()
        if auxiliary_output is not None:
            self.batch_result_processor.consume_auxiliary_output(
                batch,
                auxiliary_output,
                auxiliary_output_starts,
            )

        can_run_cuda_graph = result.can_run_cuda_graph
        self.metrics_reporter.report_prefill_stats(
            batch=batch,
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )
        PHASES.stop("result_aux_and_metrics", _t_phase)
        PHASES.stop("result_total", _t_entry)

    def process_disagg_prefill_inflight_queue(
        self: Scheduler, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        """
        Poll the requests in the middle of transfer. If done, return the request.
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """
        if len(self.disagg_prefill_inflight_queue) == 0:
            return []

        done_reqs = []
        if getattr(self, "_late_d_need_enabled", False):
            self.late_d_need_expire_parks()

        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )

        undone_reqs: List[Req] = []
        # Check .poll() for the reqs in disagg_prefill_inflight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            if rids_to_check is not None:
                if req.rid not in rids_to_check:
                    undone_reqs.append(req)
                    continue

                # In PP mode, the previous rank may have reached a terminal
                # state (Success/Failed) while this rank's local poll is still
                # in a transient state due to clock skew or propagation delay.
                # Treat non-terminal states as undone instead of crashing.
                if poll not in (
                    KVPoll.Success,
                    KVPoll.Failed,
                ):
                    logger.warning_once(
                        f"PP rank {self.ps.pp_rank}: unexpected poll state {poll} for rid {req.rid} "
                        f"from consensus; treating as undone",
                    )
                    undone_reqs.append(req)
                    continue

            if req.pending_bootstrap:
                # Parked: prefill finished before bootstrap completed.
                if self.handle_pending_bootstrap(req, poll):
                    if getattr(req, "late_d_need", False):
                        req.late_d_park_ended = time.monotonic()
                        logger.info("LATE_D_NEED park_end rid=%s age_s=%.3f", req.rid,
                                    req.late_d_park_ended - req.late_d_park_started)
                    self.send_kv_chunk(req, last_chunk=True)
                    undone_reqs.append(req)
                elif poll != KVPoll.Failed:
                    undone_reqs.append(req)
                continue

            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                # todo: set Transferring correctly in backend
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                if not isinstance(req.finished_reason, FINISH_ABORT):
                    req.finished_reason = FINISH_LENGTH(length=0)
                release_kv_cache(req, self.tree_cache)  # unlock the tree
                # FIXME: clean up req's data in transfer engine
                req.disagg_kv_sender.clear()
                done_reqs.append(req)
                req.time_stats.set_prefill_kv_transfer_finish_time()
            elif poll == KVPoll.Failed:
                self.handle_inflight_transfer_failure(req)
                done_reqs.append(req)
            else:
                raise RuntimeError(
                    f"Unexpected poll state {poll} for req {req.rid} in inflight queue"
                )

        for req in done_reqs:
            req.time_stats.set_completion_time()

        for req in done_reqs:
            if isinstance(req.finished_reason, FINISH_ABORT):
                continue
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
                continue
            kv_mgr = getattr(req.disagg_kv_sender, "kv_mgr", None)
            if kv_mgr and getattr(kv_mgr, "is_dummy_cp_rank", False):
                continue
            transfer_metric = req.disagg_kv_sender.get_transfer_metric()
            if self.ps.tp_rank == 0:
                logger.debug(
                    "PD prefill transfer evidence: rid=%s bootstrap_room=%s "
                    "decode_prefix_len=%d main_pages=%s main_bytes=%s "
                    "state_bytes=%s total_bytes=%s",
                    req.rid,
                    req.bootstrap_room,
                    req.disagg_decode_prefix_len,
                    transfer_metric.transfer_main_pages,
                    transfer_metric.transfer_main_bytes,
                    transfer_metric.transfer_state_bytes,
                    transfer_metric.transfer_total_bytes,
                )
            metrics = req.time_stats.compute_and_observe_kv_transfer_metrics(
                transfer_metric
            )
            if metrics:
                # Update last-value for REST API
                if "latency_ms" in metrics:
                    self.metrics_reporter.kv_transfer_latency_ms = metrics["latency_ms"]
                if "speed_gb_s" in metrics:
                    self.metrics_reporter.kv_transfer_speed_gb_s = metrics["speed_gb_s"]

        # Stream requests which have finished transfer
        self.output_streamer.stream_output(
            done_reqs,
            any(req.return_logprob for req in done_reqs),
            None,
        )
        for req in done_reqs:
            req: Req

            maybe_release_metadata_buffer(
                req, self.req_to_metadata_buffer_idx_allocator
            )

        self.disagg_prefill_inflight_queue = undone_reqs

        return done_reqs

    def handle_inflight_transfer_failure(
        self: Scheduler, req: Req
    ) -> Optional[Exception]:
        """Conclude an inflight request whose KV transfer failed."""
        error_message = (
            f"Prefill transfer failed for request rank={self.ps.tp_rank} "
            f"{req.rid=} {req.bootstrap_room=}"
        )
        exc: Optional[Exception] = None
        try:
            req.disagg_kv_sender.failure_exception()
        except Exception as e:
            exc = e
            error_message += f" with exception {e}"
        # Mute error message for propagated exceptions to avoid duplicate logging
        if getattr(exc, "is_from_another_rank", False):
            logger.debug(error_message)
        else:
            logger.warning(error_message)
        req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
        release_kv_cache(req, self.tree_cache)  # unlock the tree
        if not isinstance(req.finished_reason, FINISH_ABORT):
            prepare_abort(
                req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
            )
        if self.metrics_reporter.enable_metrics:
            self.metrics_collector.increment_transfer_failed_reqs()
        return exc

    def clear_pending_chunk_send(self: Scheduler, req: Req) -> None:
        """Drop `req` from the sent-but-unconcluded chunk set.

        Every path that retires a request without a `last_chunk=True` send must
        call this: a stale entry holds the unified-memory compaction gate closed
        for the process lifetime.
        """
        self.disagg_prefill_pending_chunk_rids.discard(req.rid)

    def handle_bootstrap_failure(self: Scheduler, req: Req) -> None:
        self.clear_pending_chunk_send(req)
        error_message = (
            f"Prefill bootstrap failed for request rank={self.ps.tp_rank} "
            f"{req.rid=} {req.bootstrap_room=}"
        )
        is_propagated = False
        try:
            req.disagg_kv_sender.failure_exception()
        except Exception as e:
            error_message += f" with exception {e}"
            is_propagated = getattr(e, "is_from_another_rank", False)
        # Mute error message for propagated exceptions to avoid duplicate logging
        if is_propagated:
            logger.debug(error_message)
        else:
            logger.warning(error_message)
        req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
        if req.is_holding_kv or req.mamba_pool_idx is not None:
            if (
                (getattr(self.tree_cache, "dsv4_uniform_swa_residency", "full")
                 == "window" or getattr(req, "late_d_need", False))
                and req is self.chunked_req
            ):
                # DSv4 window SWA residency: the chunk stashed this iteration
                # may still be running, so a finished insert must not write it
                # through now. Release without insert; the tree queues the
                # unbacked chunk for a re-backup after its result is processed.
                release_kv_cache(req, self.tree_cache, is_insert=False)
            else:
                release_kv_cache(req, self.tree_cache)
        maybe_release_metadata_buffer(req, self.req_to_metadata_buffer_idx_allocator)
        req.pending_bootstrap = False
        prepare_abort(req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
        self.output_streamer.stream_output([req], req.return_logprob)
        if self.metrics_reporter.enable_metrics:
            self.metrics_collector.increment_bootstrap_failed_reqs()
        if self.enable_hicache_storage:
            self.tree_cache.release_aborted_request(req.rid)
        # A request without KV skipped cache_finished_req above.
        forget_kv_session_request(self, req)

    def handle_pending_bootstrap(self: Scheduler, req: Req, poll: KVPoll) -> bool:
        """Return True when bootstrap is finalized and KV transfer can proceed."""
        if poll == KVPoll.Failed:
            self.handle_bootstrap_failure(req)
            return False
        elif poll == KVPoll.Bootstrapping:
            return False
        elif poll == KVPoll.WaitingForInput:
            if should_force_retry(req):  # test hook
                return False
            # Metadata buffer was allocated in pop_bootstrapped before
            # the request entered the waiting queue, so finalize should not fail.
            assert self.disagg_prefill_bootstrap_queue.finalize_bootstrap(req)
            return True
        else:
            raise RuntimeError(
                f"Unexpected poll state {poll} for req {req.rid} in handle_pending_bootstrap"
            )

    def check_bootstrap(self: Scheduler, req: Req) -> bool:
        """Check bootstrap status for an optimistic prefilled request.
        Returns True if bootstrap is finished."""
        if not req.pending_bootstrap:
            return True
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        return self.handle_pending_bootstrap(req, polls[0])

    def process_prefill_chunk(
        self: Scheduler,
        last_batch: Optional[ScheduleBatch],
        running_batch: ScheduleBatch,
    ) -> None:
        chunked_req_to_exclude = set()
        if (req := self.chunked_req) is not None:
            chunked_req_to_exclude.add(req)
            maybe_cache_unfinished_req(req, self.tree_cache, chunked=True)
            if getattr(req, "late_d_need", False):
                # The next result owns this boundary even before D exists.
                req.tmp_end_idx = min(req.extend_range.end, len(req.origin_input_ids))

            if not self.check_bootstrap(req):
                if is_aborted(req):
                    # bootstrap failed
                    self.chunked_req = None
                elif (self.has_bootstrapped_waiting_req()
                      and not getattr(req, "late_d_need", False)):
                    # optimistic request yields to waiting requests
                    self.chunked_req = None
                    if not self.enable_overlap:
                        self.optimistic_release_and_requeue(req)
                # else: still bootstrapping, keep computing without sending
            elif self.enable_overlap:
                # Delay KV transfer to process_batch_result_disagg_prefill when overlap is enabled to ensure results are resolved
                req.tmp_end_idx = min(
                    req.extend_range.end,
                    len(req.origin_input_ids),
                )
            else:
                self.send_kv_chunk(req)

            if self.chunked_req is not None:
                running_batch.batch_is_full = False

        if last_batch and last_batch.forward_mode.is_extend():
            if last_batch.chunked_req:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(last_batch.chunked_req)

            last_bs = last_batch.batch_size()
            last_batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))
            if last_batch.batch_size() < last_bs:
                running_batch.batch_is_full = False

    @timed_phase("S5_early_send")
    def maybe_send_cached_prefix_chunk(self: Scheduler, req: Req) -> None:
        if (not envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.get()
                and not getattr(req, "late_d_need", False)):
            return

        # Rows adopted from bootstrap-prefetch ops still copying (M5 4C) are
        # not at rest yet, and the transfer does not wait for device work:
        # leave them out. The sends after the forward (which waited for the
        # ops) carry them. 0 unless SGLANG_BWP_CONSUME_INFLIGHT adopted rows.
        inflight = getattr(req, "inflight_hit_length", 0)

        # Staging sends into positional grid slots, so the early-send boundary
        # must stay stable across the request's batches: snapshot the at-rest
        # prefix on the first batch. Non-staging reads the live prefix.
        if self.enable_staging and req.early_send_prefix_end is None:
            req.early_send_prefix_end = max(
                0, len(req.prefix_indices) - req.host_hit_length - inflight
            )

        if req.pending_bootstrap:
            if (getattr(req, "late_d_need", False)
                    and not req.late_d_prefix_snapshotted):
                req.late_d_prefix_snapshotted = True
                cached_end = max(0, len(req.prefix_indices) - req.host_hit_length - inflight)
                req.early_send_prefix_end = cached_end
                if cached_end > 0:
                    ev = torch.cuda.Event()
                    ev.record(self.forward_stream)
                    self._defer_early_send(req, cached_end, ev)
                    # Keep the launch snapshot until a destination exists;
                    # the ordinary flush intentionally drops unbound entries.
                    req.late_d_early_snapshot = self._pending_early_sends.pop()
            return
        if getattr(req, "late_d_catchup", False):
            return

        # Device-resident prefix only; page-aligned so start_send_idx stays exact.
        cached_end = (
            req.early_send_prefix_end
            if self.enable_staging
            else len(req.prefix_indices) - req.host_hit_length - inflight
        )
        if cached_end <= req.start_send_idx:
            return
        if cached_end % self.token_to_kv_pool_allocator.page_size != 0:
            # DCP radix hits can end on a logical cache-page boundary that is
            # not a complete physical DCP page. The regular final send covers
            # the full range; only skip this optional early-send optimization.
            return
        # Early-send issues the KV read before this step's forward is enqueued,
        # but under overlap scheduling the PRIOR step's prefill forward may still
        # be writing these prefix pages on forward_stream. Record a completion
        # event now so the transfer worker can wait on those writes before the
        # RDMA read, instead of racing them.
        if self.enable_overlap:
            ev = torch.cuda.Event()
            ev.record(self.forward_stream)
            if getattr(self, "_early_send_async", False) or (
                getattr(self, "_cpu_page_book", None) is not None
                and not sender_waits_on_early_send_event(req.disagg_kv_sender)
            ):
                # 6d: no blocking read inside run_batch. The page ids are
                # copied asynchronously here; the send is issued after the
                # launch (_m2_flush_early_sends), once that copy and the
                # previous forward are done. Q4 reads the ids from the CPU
                # rows, so nothing orders the send after the forward but this
                # (or a connector that waits on the event, 6g).
                self._defer_early_send(req, cached_end, ev)
                return
            req.disagg_kv_sender._early_send_wait_event = ev
        self.send_kv_chunk(req, last_chunk=False, end_idx=cached_end)

    def late_d_need_expire_parks(self):
        """Leader-only failure; the ordinary inflight MIN poll frees all ranks."""
        if getattr(self, "_late_d_need_enabled", False):
            binder = self.disagg_prefill_bootstrap_queue.late_d_binder
            if binder is not None:
                now = time.monotonic()
                timeout = envs.SGLANG_LATE_D_PARK_TIMEOUT_S.get()
                mgr = self.disagg_prefill_bootstrap_queue.kv_manager
                for req in self.disagg_prefill_inflight_queue:
                    if (req.pending_bootstrap and req.late_d_park_started > 0
                            and now - req.late_d_park_started >= timeout):
                        req.late_d_park_timeout = True
                        req.late_d_park_ended = now
                        mgr.record_failure(req.bootstrap_room,
                                           "late D: parked, park timer")
                        mgr.update_status(req.bootstrap_room, KVPoll.Failed)
                        logger.warning("LATE_D_NEED park_timeout rid=%s age_s=%.3f",
                                       req.rid, now - req.late_d_park_started)


    def late_d_need_send_prefix(self, req):
        """At bind, send the launch-time device prefix before any final send."""
        p = req.late_d_early_snapshot
        req.late_d_early_snapshot = None
        req.late_d_bind_perf = time.perf_counter()
        logger.info("LATE_D_NEED bind rid=%s t_perf=%.6f device_prefix=%d "
                    "computed_end=%d decode_prefix=%d parked=%d", req.rid,
                    time.perf_counter(), req.early_send_prefix_end or 0,
                    req.late_d_computed_end, req.start_send_idx,
                    int(req.late_d_park_started > 0))
        if p is None or p.end_idx <= req.start_send_idx:
            return
        start = req.start_send_idx
        page = self.token_to_kv_pool_allocator.page_size
        p = dataclasses.replace(p, start_idx=start,
                                page_ids_cpu=p.page_ids_cpu[(start - p.start_idx) // page:])
        # Local CUDA completion is diagnostic only, never a transfer/admission
        # decision. The saved prefix event still orders the actual send.
        done_event = req.late_d_forward_done_event
        after_forward = bool(done_event.query()) if done_event is not None else bool(req.late_d_park_started)
        req.late_d_device_prefix_sent_tokens += p.end_idx - start
        if after_forward:
            req.late_d_device_prefix_after_forward_tokens += p.end_idx - start
        flush_pending_early_sends([p], lambda item: self.send_kv_chunk(
            req, last_chunk=False, end_idx=item.end_idx, early_send_snapshot=item), is_aborted)
        logger.info("LATE_D_NEED device_prefix_send rid=%s tokens=%d after_forward=%d",
                    req.rid, p.end_idx - start, int(after_forward))

    def late_d_need_flush_catchup(self):
        if not getattr(self, "_late_d_need_enabled", False):
            return
        page = self.token_to_kv_pool_allocator.page_size
        cap = envs.SGLANG_LATE_D_CATCHUP_TOKENS_PER_PASS.get()
        budget = (cap // page) * page if cap > 0 else self.max_total_num_tokens
        seen = set()
        reqs = []
        for batch in (getattr(self, "last_batch", None), getattr(self, "running_batch", None),
                      getattr(self, "cur_batch_for_debug", None)):
            if batch is not None:
                reqs.extend(batch.reqs)
        if self.chunked_req is not None:
            reqs.append(self.chunked_req)
        plan = getattr(getattr(self, "late_d_lookahead", None), "plan", None)
        if plan is not None:
            reqs.sort(key=lambda r: (getattr(plan.entries.get(r.rid), "end_pass", None)
                                    or float("inf")))
        used = 0
        for req in reqs:
            if id(req) in seen:
                continue
            seen.add(id(req))
            if (not getattr(req, "late_d_catchup", False) or req.pending_bootstrap
                    or is_aborted(req) or req in self.disagg_prefill_inflight_queue):
                continue
            end = catchup_end(req.start_send_idx, req.late_d_computed_end, budget, page)
            if end > req.start_send_idx:
                n = end - req.start_send_idx
                self.send_kv_chunk(req, last_chunk=False, end_idx=end)
                req.late_d_catchup_sent_tokens += n
                budget -= n
                used += n
            if req.start_send_idx >= (req.late_d_computed_end // page) * page:
                req.late_d_catchup = False
        if used:
            logger.info("LATE_D_NEED catchup tokens=%d cap=%d", used, cap)

    def _defer_early_send(self: Scheduler, req: Req, end_idx: int, wait_event) -> None:
        from decoupled_engine.srt.managers.utils import _async_d2h

        start_idx = req.start_send_idx
        book = getattr(self, "_cpu_page_book", None)
        if book is not None:
            # Q4: the ids are on the CPU already.
            page_ids_cpu = torch.from_numpy(
                book.rows.page_ids(req.req_pool_idx, start_idx, end_idx)
            )
            ready_event = None
        else:
            page_ids_cpu, ready_event = snapshot_early_send_page_ids(
                self.req_to_token_pool.req_to_token,
                req.req_pool_idx,
                start_idx,
                end_idx,
                self.token_to_kv_pool_allocator.page_size,
                _async_d2h,
                self.device_module.Event,
            )
        self._pending_early_sends.append(
            PendingEarlySend(
                req=req,
                start_idx=start_idx,
                end_idx=end_idx,
                page_ids_cpu=page_ids_cpu,
                ready_event=ready_event,
                wait_event=wait_event,
            )
        )
        M2_STATS["early_send_deferred"] += 1

    def send_kv_chunk(
        self: Scheduler,
        req: Req,
        last_chunk: bool = False,
        end_idx: Optional[int] = None,
        early_send_snapshot: Optional[PendingEarlySend] = None,
    ) -> None:
        """
        Send a prefilled chunk to the decode server
        """
        page_size = self.token_to_kv_pool_allocator.page_size
        start_idx = req.start_send_idx
        transfer_input_len = len(req.origin_input_ids)
        end_idx = (
            end_idx
            if end_idx is not None
            else min(req.extend_range.end, transfer_input_len)
        )

        if not last_chunk:
            # if not the last chunk and the last page is partial, delay the last partial page to the next send
            end_idx = end_idx - end_idx % page_size
            if self.enable_staging:
                # Staging identifies chunks positionally against a uniform
                # prefetched grid, so non-last sends must end on a grid
                # boundary; the remainder rides with the next send.
                grid_tokens = staging_grid_tokens(
                    get_schedule().chunked_prefill_size, page_size
                )
                base = req.disagg_decode_prefix_len
                end_idx = base + ((end_idx - base) // grid_tokens) * grid_tokens

        if end_idx < start_idx:
            logger.debug(
                "send_kv_chunk skip: rid=%s start_send_idx=%s end_idx=%s",
                req.rid,
                start_idx,
                end_idx,
            )
            return

        state_indices: Optional[List] = None
        if last_chunk:
            self.disagg_metadata_buffers.set_buf(req)

            # Most state payloads read token-pool rows and should match the KV
            # range actually materialized on prefill. C128 state is request
            # scoped, so its transfer index must use the logical input length
            # that decode used to register the destination row.
            seq_len = min(req.extend_range.end, transfer_input_len)
            c128_seq_len = transfer_input_len

            def _mamba_payload():
                return [
                    self.req_to_token_pool.translate_mamba_indices(
                        self.req_to_token_pool.req_index_to_mamba_index_mapping[
                            req.req_pool_idx
                        ]
                    )
                    .cpu()
                    .numpy()
                ]

            def _swa_payload():
                window_size = self.sliding_window_size
                window_start = max(req.disagg_decode_prefix_len, seq_len - window_size)
                window_start = (window_start // page_size) * page_size
                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, window_start:seq_len
                ]
                window_kv_indices_swa = (
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        window_kv_indices_full
                    )
                )
                return kv_to_page_indices(window_kv_indices_swa, page_size)

            def _full_kv_pages_payload(prefix_len=0):
                reader = getattr(self, "_kv_page_reader", None)
                if reader is not None:
                    # 6b: not behind the barrier (see KvPageIdReader).
                    return reader.page_ids(req.req_pool_idx, prefix_len, seq_len)
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, prefix_len:seq_len
                ]
                return kv_to_page_indices(kv_indices_full, page_size)

            def _swa_ring_payload():
                # Unified_kv SWA ring rows (req_pool_idx*ring_stride + pos%ring_stride)
                # for the last `window` positions, in ascending position order so
                # decode (its own req_pool_idx) matches positionally.
                _pool = self.token_to_kv_pool_allocator.get_kvcache()
                ring_stride = _pool.unified_swa_ring_size
                window_size = _pool.unified_swa_window
                window_start = max(0, seq_len - window_size)
                positions = np.arange(window_start, seq_len, dtype=np.int64)
                state_slot = int(req.req_pool_idx)
                ring_rows = state_slot * ring_stride + (positions % ring_stride)
                return ring_rows.astype(np.int32)

            def _c128_state_payload():
                online = is_dsv4_c128_online_enabled()
                ring_size = (
                    1
                    if online
                    else self.token_to_kv_pool_allocator.get_kvcache().get_ring_size(
                        128
                    )
                )
                return get_dsv4_c128_state_indices(
                    int(req.req_pool_idx),
                    c128_seq_len,
                    online=online,
                    ring_size=ring_size,
                )

            state_types = (
                self.disagg_prefill_bootstrap_queue.kv_manager.kv_args.state_types
            )
            payloads = {
                StateType.MAMBA: _mamba_payload,
                StateType.SWA: _swa_payload,
                # DSA includes both target and EAGLE draft indexer buffers.
                # A radix hit already owns these pages; never overwrite a
                # shared prefix while another decode request may be reading it.
                StateType.DSA: lambda: _full_kv_pages_payload(
                    req.disagg_decode_prefix_len
                    if self.spec_algorithm.is_eagle()
                    else 0
                ),
                StateType.MINIMAX_INDEX_K: _full_kv_pages_payload,
                StateType.SWA_RING: _swa_ring_payload,
                StateType.C128_STATE: _c128_state_payload,
                StateType.BLOCK_SCALE: _full_kv_pages_payload,
                StateType.BLOCK_SCALE_SWA: _swa_payload,
            }
            if _is_npu and isinstance(
                self.token_to_kv_pool_allocator.get_kvcache(),
                DeepSeekV4TokenToKVPool,
            ):
                from decoupled_engine.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
                    dsv4_state_payloads,
                )

                payloads.update(
                    dsv4_state_payloads(
                        self.req_to_token_pool,
                        req.req_pool_idx,
                        seq_len,
                        page_size,
                        prefix_len=req.disagg_decode_prefix_len,
                    )
                )
            state_indices = [
                payloads[st]() if st in payloads else None for st in state_types
            ]

        if self.enable_staging:
            # One sender.send per grid slot; the sender's cumulative page
            # counter marks only the final sub-send of the final chunk as
            # is_last, routing aux/state correctly.
            segments = compute_grid_segments(
                start_idx,
                end_idx,
                req.disagg_decode_prefix_len,
                staging_grid_tokens(get_schedule().chunked_prefill_size, page_size),
            )
        else:
            segments = [(start_idx, end_idx)]

        kv_page_reader = getattr(self, "_kv_page_reader", None)
        for seg_start, seg_end in segments:
            is_final_segment = seg_end == end_idx
            # 6d: the early send's page ids were copied at launch.
            page_indices = page_ids_from_snapshot(
                early_send_snapshot, seg_start, seg_end
            )
            if page_indices is None and kv_page_reader is not None:
                # 6b: only set for plain paged pools without staging, where the
                # translation below is the identity and there is one segment.
                page_indices = kv_page_reader.page_ids(
                    req.req_pool_idx, seg_start, seg_end
                )
            if page_indices is None:
                kv_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, seg_start:seg_end
                ]
                # Unified memory: req_to_token holds VIRTUAL ids; the transfer needs
                # physical ones. Per segment, since each is its own gather.
                kv_indices = (
                    self.token_to_kv_pool_allocator.translate_kv_indices_for_transfer(
                        kv_indices
                    )
                )
                page_indices = kv_to_page_indices(kv_indices, page_size)
            segment_is_last = last_chunk and is_final_segment
            if not req.disagg_kv_sender.should_send_kv_chunk(
                len(page_indices), segment_is_last
            ):
                continue
            req.disagg_kv_sender.send(
                page_indices,
                state_indices if segment_is_last else None,
                num_kv_tokens=seg_end - seg_start,
            )
        req.start_send_idx = end_idx
        # A last chunk needs no entry: every `last_chunk=True` call site has
        # already put the request on `disagg_prefill_inflight_queue`.
        if last_chunk:
            self.disagg_prefill_pending_chunk_rids.discard(req.rid)
        else:
            self.disagg_prefill_pending_chunk_rids.add(req.rid)

    def optimistic_release_and_requeue(self: Scheduler, req: Req) -> None:
        """Release KV cache and requeue an optimistic prefill request."""
        assert not getattr(req.disagg_kv_sender, "late_d", False), (
            "late D never releases/requeues computed KV")
        max_attempts = get_disagg().optimistic_prefill_attempts
        maybe_cache_unfinished_req(req, self.tree_cache)
        release_kv_cache(req, self.tree_cache)
        # The release dropped the request from its retention registration;
        # it waits again, so it counts as live again (soft pin).
        rebind_kv_session_request(self, req)
        req.reset_for_retract()
        req.output_ids = array("q")
        req.start_send_idx = 0
        self.clear_pending_chunk_send(req)  # re-sends from scratch
        req.tmp_end_idx = -1
        req.disagg_decode_prefix_len = 0
        req.early_send_prefix_end = None
        req.hidden_states_tensor = None
        req.output_dsa_topk_indices = None
        req.pending_bootstrap = True
        req.time_stats.reset_prefill_retry_time()
        if req.prefill_attempt_count >= max_attempts:
            logger.info(
                f"Req {req.rid} exhausted optimistic prefill attempts "
                "falling back to bootstrap queue"
            )
            # Reset it so the next real bootstrap done can be recorded.
            req.time_stats.bootstrap_done_time = 0.0
            self.disagg_prefill_bootstrap_queue.queue.append(req)
        else:
            req.prefill_attempt_count += 1
            logger.info(
                f"Req {req.rid} optimistic prefill yielded "
                f"({req.prefill_attempt_count}/{max_attempts} attempts used)"
            )
            if self.metrics_reporter.enable_metrics:
                self.metrics_collector.increment_prefill_retries(1)
            req.time_stats.set_wait_queue_entry_time()
            self.waiting_queue.insert(0, req)
