# Adapted from SGLang python/sglang/srt/disaggregation/decode.py (commit 297e54fd10), modified for evolving-infra.
"""
Life cycle of a request in the decode server

1. PreallocQueue:
    a. Initialize a receiver for each request
    b. The request handshakes first, and pre-allocate kv once there is available kv.
    c. Move the request to TransferQueue.

2. TransferQueue:
    a. Poll the receiver to check the transfer state
    b. If the transfer has finished, move the request to waiting queue

3. WaitingQueue:
    a. Use the requests in the queue to construct a PrebuiltExtendBatch
    b. Skip the prefill forward but only populate metadata

4. RunningBatch:
    a. Merge the resolved PrebuiltExtendBatch into running batch to run decoding
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
import torch.distributed as dist

from decoupled_engine.srt.configs.mamba_utils import Mamba2CacheParams
from decoupled_engine.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from decoupled_engine.srt.disaggregation.base import KVPoll
from decoupled_engine.srt.disaggregation.base.conn import KVTransferDestination, StateType
from decoupled_engine.srt.disaggregation.checksum import (
    KvChecksumComputer,
    is_health_check_req,
    page_indices_for_request,
    state_indices_for_request,
)
from decoupled_engine.srt.disaggregation.common.conn import CommonKVManager, CommonKVReceiver
from decoupled_engine.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreGatedKVReceiver,
    HiCacheRestoreResult,
)
from decoupled_engine.srt.disaggregation.late_d import (
    LateDConfig,
    LateDControl,
    LateDDecode,
    make_late_d_metrics,
)
from decoupled_engine.srt.disaggregation.minimax import minimax_index_transfer_pages
from decoupled_engine.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    _is_fake_transfer,
    all_reduce_flags,
    build_kv_layer_ids,
    build_staging_slot_metadata,
    get_dsa_tail_state_indices,
    get_kv_class,
    get_kv_transfer_buf_infos,
    get_pd_kv_layout,
    get_qsa_pending_state_indices,
    is_mla_backend,
    is_unadmitted_reject,
    kv_layout_signature,
    poll_and_all_reduce,
    poll_and_all_reduce_pp,
    poll_and_all_reduce_with_staging,
    prepare_abort,
    setup_state_kv_args,
)
from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.managers.schedule_batch import (
    FINISH_ABORT,
    NextBatchPlan,
    ScheduleBatch,
)
from decoupled_engine.srt.managers.schedule_policy import (
    swa_window_decode_allowance,
    swa_window_running_offset,
)
from decoupled_engine.srt.managers.scheduler_components.kv_events_publisher import (
    kv_events_end_pass_hook,
)
from decoupled_engine.srt.managers.scheduler_components.phase_timers import (
    PHASES,
    phase,
    sync_debug_site,
)
from decoupled_engine.srt.managers.scheduler_components.pool_stats_observer import (
    kv_private_swa_tokens,
    kv_private_tokens,
)
from decoupled_engine.srt.managers.utils import GenerationBatchResult
from decoupled_engine.srt.mem_cache.allocation import ensure_mamba_capacity
from decoupled_engine.srt.mem_cache.allocation_sizing import (
    get_alloc_reserve_per_decode,
    get_mamba_tracking_slots,
)
from decoupled_engine.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from decoupled_engine.srt.mem_cache.allocator.swa import is_swa_req_ring
from decoupled_engine.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    TreeLock,
)
from decoupled_engine.srt.mem_cache.common import (
    RetractionBackup,
    discard_kv_cache_backup,
    dsv4_needs_rebootstrap,
    kv_to_page_indices,
    match_kv_cache,
    page_align_floor,
    release_kv_cache,
    release_retract_backup,
    release_retracted_swa_ledger,
    restore_kv_cache,
    retract_backup_unsafe_rids,
    suffix_page_indices,
)
from decoupled_engine.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from decoupled_engine.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from decoupled_engine.srt.mem_cache.kv_cache_builder import decode_retraction_max_tokens
from decoupled_engine.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    HybridReqToTokenPool,
    KVCache,
    ReqToTokenPool,
)
from decoupled_engine.srt.mem_cache.swa_memory_pool import SWAKVPool
from decoupled_engine.srt.mem_cache.unified_cache.component_type import ComponentType
from decoupled_engine.srt.observability.req_time_stats import (
    set_schedule_time_batch,
    set_time_batch,
)
from decoupled_engine.srt.observability.scheduler_stage_metrics import (
    SCHEDULER_STAGE_GET_NEXT_BATCH,
    SCHEDULER_STAGE_PROCESS_QUEUE,
    scheduler_stage_method,
)
from decoupled_engine.srt.runtime_context import (
    get_device,
    get_disagg,
    get_memory,
    get_parallel,
)
from decoupled_engine.srt.utils import ceil_align, get_num_new_pages, is_npu
from decoupled_engine.srt.utils.common import async_h2d
from decoupled_engine.srt.utils.network import NetworkAddress
from decoupled_engine.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from decoupled_engine.utils import is_in_ci

logger = logging.getLogger(__name__)

_is_npu = is_npu()

if TYPE_CHECKING:
    from decoupled_engine.srt.managers.schedule_batch import Req
    from decoupled_engine.srt.managers.scheduler import Scheduler

CLIP_MAX_NEW_TOKEN = envs.SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION.get()


def _decode_growth_reserve(req: Req, ratio: float, floor_tokens: int) -> int:
    """Decode growth that PD decode admission keeps free for ``req`` under
    SGLANG_DECODE_ADMIT_GROWTH_RATIO: max(floor_tokens, ceil(ratio x
    min(max_new_tokens - output tokens, CLIP_MAX_NEW_TOKEN))). floor_tokens is
    num_reserved_decode_tokens, the reserve without the env."""
    remaining = min(
        req.sampling_params.max_new_tokens - len(req.output_ids), CLIP_MAX_NEW_TOKEN
    )
    return max(floor_tokens, math.ceil(ratio * max(remaining, 0)))


def _bootstrap_addr(req: Req) -> str:
    # FIXME: make a property of a req
    return NetworkAddress(req.bootstrap_host, req.bootstrap_port).to_host_port_str()


def _bind_root_prefix(req: Req, tree_cache: BasePrefixCache) -> None:
    """Start a decode-radix request that owns its whole KV row at the root."""
    req.prefix_len = 0
    req.last_node = tree_cache.root_node_handle(req.extra_key)
    req.last_host_node = req.last_node
    req.best_match_node = req.last_node
    req.lock = None
    req.kv.cache_protected_len = 0
    req.kv.cache_inserted_len = 0
    req.num_matched_prefix_tokens = 0
    req.host_hit_length = 0


class DecodeReqToTokenPool:
    """
    The difference of DecodeReqToTokenPool and ReqToTokenPool is that
    DecodeReqToTokenPool subscribes memory for pre-allocated requests.

    In ReqToTokenPool, if `--max-running-requests` is 8,
    #pre-allocated + #transfer + #running <= 8, but there are in fact more memory can carry pre-allocated requests.

    In DecodeReqToTokenPool, if `--max-running-requests` is 8,
    #running <= 8, #pre-allocated + #transfer <= pre_alloc_size, so we can use the free memory to pre-allocate requests to unblock prefill.
    """

    # Mirrors ReqToTokenPool.register_on_alloc_rows.
    _on_alloc_rows: Optional[Callable[[List[int]], None]] = None

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        pre_alloc_size: int,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        # +1 padding row at index 0; see ReqToTokenPool for rationale.
        self._alloc_size = size + pre_alloc_size + 1
        self.max_context_len = max_context_len
        self.device = device
        self.pre_alloc_size = pre_alloc_size
        with memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (self._alloc_size, max_context_len),
                dtype=torch.int32,
                device=device,
            )

        self.free_slots = list(range(1, self._alloc_size))
        # Slot-reuse generation counter; mirrors ReqToTokenPool. Required even
        # here: HybridMambaDecodeReqToTokenPool borrows this __init__ while
        # inheriting ReqToTokenPool.alloc, which bumps it.
        self.req_generation = torch.zeros(self._alloc_size, dtype=torch.int64)
        self._aux_cache: Any = None

    def write(self, indices, values, values_unchanged: bool = False):
        # values_unchanged: as ReqToTokenPool.write (passed only while row-write
        # tracking is on, e.g. by unified_radix_cache).
        if self._track_row_writes and not values_unchanged:
            self._note_row_write(indices)
        if self.cpu_rows is not None and not self._cpu_rows_by_caller:
            self._mirror_row_write(indices, values)
        self.req_to_token[indices] = values

    # M4 Q4 (mem_cache/cpu_page_book.py) on a symmetric (symmetric P/D) worker, whose
    # request pool is this one: the CPU page rows and the hooks ReqToTokenPool
    # has (tagged (row, slice) writes are mirrored; callers that write the CPU
    # rows themselves wrap the device write in cpu_rows_written_by_caller).
    # None = off.
    cpu_rows = None
    _cpu_rows_by_caller: bool = False
    _mirror_row_write = ReqToTokenPool._mirror_row_write
    cpu_rows_written_by_caller = ReqToTokenPool.cpu_rows_written_by_caller

    # Row-write sequence, as ReqToTokenPool has it. A symmetric worker's request
    # pool is this one, and its owner publications read rows after a pre-WAR
    # mark (SGLANG_SYMMETRIC_PUBLISH_ASYNC_READBACK,
    # disaggregation/symmetric/publish_readback.py). Off by default.
    _track_row_writes: bool = False
    enable_row_write_tracking = ReqToTokenPool.enable_row_write_tracking
    _note_row_write = ReqToTokenPool._note_row_write
    row_write_seq = ReqToTokenPool.row_write_seq
    row_written_after = ReqToTokenPool.row_written_after

    def available_size(self):
        return len(self.free_slots)

    def reset_aux_cache_allocator(self) -> None:
        pass

    def schedulable_token_capacity(self, physical_capacity: int) -> int:
        return physical_capacity

    def alloc_aux_to_lengths(
        self,
        *,
        req_pool_indices_cpu: torch.Tensor,
        target_seq_lens_cpu: torch.Tensor,
    ) -> None:
        pass

    def alloc(self, reqs: List[Req]) -> Optional[List[int]]:
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        reusing = [i for i, r in enumerate(reqs) if r.kv.holds_kv]
        assert all(reqs[i].kv.kv_allocated_len > 0 for i in reusing), (
            "a reused row must carry allocated KV"
        )

        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        if self._on_alloc_rows is not None and select_index:
            self._on_alloc_rows(select_index)
        offset = 0
        for r in reqs:
            if not r.kv.holds_kv:
                r.kv.req_pool_idx = select_index[offset]
                self.req_generation[r.kv.req_pool_idx] += 1
                offset += 1
        return [r.kv.req_pool_idx for r in reqs]

    def free(self, req: Req):
        assert req.kv.holds_kv, "request must have req_pool_idx"
        if self.cpu_rows is not None:
            self.cpu_rows.release([req.kv.req_pool_idx])
        self.free_slots.append(req.kv.req_pool_idx)
        req.kv.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(1, self._alloc_size))
        # req_generation must stay monotonic; see ReqToTokenPool.clear().
        if self.cpu_rows is not None:
            self.cpu_rows.clear()

    def register_on_alloc_rows(self, hook: Callable[[List[int]], None]) -> None:
        assert self._on_alloc_rows is None
        self._on_alloc_rows = hook


class HybridMambaDecodeReqToTokenPool(HybridReqToTokenPool):
    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: Mamba2CacheParams,
        mamba_layer_ids: List[int],
        speculative_num_draft_tokens: int,
        enable_mamba_extra_buffer: bool,
        pre_alloc_size: int,
        enable_overlap_schedule: bool,
        mamba_size: int = None,
        start_layer: int = None,
        speculative_eagle_topk: Optional[int] = None,
        linear_replayssm_cache_len: int = 16,
        mamba_envelope_layout: bool = False,
        enable_linear_replayssm_spec: bool = False,
        short_conv_layer_ids: Optional[List[int]] = None,
        short_conv_state_shape: Optional[Tuple[int, int]] = None,
        ngram_context_len: int = 0,
        ngram_eos_token_id: int = 0,
    ):
        DecodeReqToTokenPool.__init__(
            self,
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            pre_alloc_size=pre_alloc_size,
        )

        self.mamba_ping_pong_track_buffer_size = get_mamba_tracking_slots(
            extra_buffer=True, overlap=enable_overlap_schedule
        )
        self.mamba_initial_tracking_slots = get_mamba_tracking_slots(
            extra_buffer=enable_mamba_extra_buffer,
            overlap=enable_overlap_schedule,
            lazy=False,
        )
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_memory_saver = enable_memory_saver
        # Each request needs 1 main mamba slot + ping-pong slots when extra_buffer is enabled.
        # Cap the pool at max concurrent requests * slots_per_req to avoid allocating failed.
        slots_per_req = 1 + self.mamba_initial_tracking_slots
        max_slots_needed = (size + pre_alloc_size) * slots_per_req
        if mamba_size is not None:
            effective_mamba_size = max(mamba_size, max_slots_needed)
            if mamba_size < max_slots_needed:
                logger.warning(
                    "mamba_size (%d) is less than decode side's max_slots_needed (%d = %d reqs * %d slots/req), "
                    "raising effective_mamba_size to %d",
                    mamba_size,
                    max_slots_needed,
                    size + pre_alloc_size,
                    slots_per_req,
                    effective_mamba_size,
                )
        else:
            effective_mamba_size = max_slots_needed
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self._init_mamba_pool(
            mamba_size=effective_mamba_size,
            mamba_spec_state_size=size + pre_alloc_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_mamba_extra_buffer=self.enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            speculative_eagle_topk=speculative_eagle_topk,
            linear_replayssm_cache_len=linear_replayssm_cache_len,
            mamba_envelope_layout=mamba_envelope_layout,
            enable_linear_replayssm_spec=enable_linear_replayssm_spec,
            short_conv_layer_ids=short_conv_layer_ids,
            short_conv_state_shape=short_conv_state_shape,
            ngram_context_len=ngram_context_len,
            ngram_eos_token_id=ngram_eos_token_id,
        )

    def clear(self):
        self.free_slots = list(range(1, self._alloc_size))
        self.mamba_allocator.clear()


@dataclass
class DecodeRequest:
    req: Req
    kv_receiver: CommonKVReceiver
    waiting_for_input: bool = False
    metadata_buffer_index: int = -1
    is_rebootstrap: bool = False
    host_staged: bool = False

    # HiCache Status
    prefix_match: Optional[DecodePrefixMatch] = None
    hicache_restored_kv_indices: Optional[torch.Tensor] = None
    # The lock held on the restored node until the commit hands it to the req.
    hicache_restore_lock: Optional[TreeLock] = None
    hicache_load_consumer_index: int = -1
    hicache_restore_status: HiCacheRestoreResult = HiCacheRestoreResult.PENDING

    # Late D (--disaggregation-late-d, late_d.py): None keeps today's FIFO
    # path; else DEFERRED -> ADMITTED (-> HELD) -> ALLOCATED. Orders and
    # pass counts are identical on every rank; the times are leader-only.
    late_d_state: Optional[str] = None
    late_d_admit_order: int = -1
    late_d_held_since: int = -1
    late_d_skips: int = 0
    late_d_deferred_at: float = 0.0
    late_d_held_at: float = 0.0
    # Leader, PD: when the INTENT was last sent (0: not yet); it is sent
    # again while the entry stays DEFERRED (LateDDecode.INTENT_REFRESH_S).
    late_d_intent_at: float = 0.0
    # Stage 2 (--disaggregation-late-d choose): this leg's candidate index,
    # from the request's late_d field; None for a stage-1 leg.
    late_d_cand_idx: Optional[int] = None
    # The FINISH_ABORT err_type of a late-D end through the Failed poll
    # (late_d_ttl after the DEFERRED TTL); None for any other failure.
    late_d_err_type: Optional[str] = None

    @property
    def seqlen(self) -> int:
        return self.req.seqlen

    @property
    def priority(self) -> Optional[int]:
        return self.req.priority


class DecodePreallocQueue(DecodeHiCachePreallocMixin):
    """
    Store the requests that are preallocating.
    """

    # Late D allocation (--disaggregation-late-d); None when off.
    late_d: Optional[LateDDecode] = None

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        transfer_queue: DecodeTransferQueue,
        tree_cache: BasePrefixCache,
        gpu_id: int,
        bootstrap_port: int,
        max_total_num_tokens: int,
        num_reserved_decode_tokens: int,
        transfer_backend: TransferBackend,
    ):
        parallel = get_parallel()
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.token_to_kv_pool = token_to_kv_pool_allocator.get_kvcache()
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(self.token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.scheduler = scheduler
        # The decode radix cache takes EAGLE and EAGLE3 on any model. With a
        # HiCache host tier, this check fails closed unless EAGLE has DSA
        # target and draft pools at shared indices or EAGLE3 the native
        # flat-MLA layout.
        # TODO: restore EAGLE and EAGLE3 drafts from the host tier on other
        # layouts, and drop this check.
        if (
            get_disagg().disaggregation_decode_enable_radix_cache
            and get_memory().enable_hierarchical_cache
            and scheduler.spec_algorithm.is_eagle()
        ):
            if scheduler.spec_algorithm.is_eagle3():
                from decoupled_engine.srt.disaggregation.native_eagle3_cache_layout import (
                    validate_native_eagle3_cache,
                )

                validate_native_eagle3_cache(
                    tree_cache,
                    self.token_to_kv_pool,
                    draft_token_to_kv_pool,
                    host_enabled=True,
                )
            elif not (
                isinstance(self.token_to_kv_pool, DSATokenToKVPool)
                and isinstance(draft_token_to_kv_pool, DSATokenToKVPool)
                and self.token_to_kv_pool.page_size == draft_token_to_kv_pool.page_size
            ):
                raise ValueError(
                    "PD decode EAGLE radix cache requires target and draft DSA "
                    "pools with the same page size and shared token indices."
                )
        self.transfer_queue = transfer_queue
        self.tree_cache = tree_cache
        self.host_pool = None
        self.host_reserved_tokens = 0
        self.gloo_group = parallel.attn_tp_group.cpu_group
        # Destinations visible to prefill but not yet on the transfer queue.
        self._num_published_destinations = 0
        self.tp_rank = parallel.tp_rank
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.max_total_num_tokens = max_total_num_tokens
        self.pp_rank = parallel.pp_rank
        self.pp_size = parallel.pp_size
        self.num_reserved_decode_tokens = num_reserved_decode_tokens
        self.transfer_backend = transfer_backend
        # Queue for requests pending pre-allocation
        self.queue: List[DecodeRequest] = []
        self.retracted_queue: List[Req] = []
        self.pending_reqs: List[DecodeRequest] = []
        # In-flight authoritative room -> DP-rank lookups, consumed below.
        self._prefill_dp_rank_queries: Dict[
            str, Tuple[Tuple[int, ...], Future[Dict[str, int]]]
        ] = {}
        self._ensure_retry_count: Dict[str, int] = {}
        self._max_ensure_retries: int = 15  # scheduling cycles
        self._ensure_last_attempt_time: Dict[str, float] = {}
        self._ensure_retry_interval: float = 1.0  # seconds
        # Retracted requests staged for rebootstrap while generation is paused.
        # Enqueued into ``self.queue`` only on ``continue_generation`` so the
        # prefix KV is recomputed under the post-retract (updated) weights.
        # NOTE: requests held here are not reachable by ``/abort_request``; to
        # support aborting them we would need an additional fix in the
        # scheduler. In practice this shouldn't arise in the RL scenario.
        self.held_rebootstrap_reqs: List[Req] = []
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        if self.enable_staging and self.is_mla_backend:
            raise RuntimeError(
                "SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models "
                "(e.g. GQA, MHA). MLA models should not set this flag."
            )
        self.kv_manager = self._init_kv_manager()
        if self.enable_staging:
            self.transfer_queue._init_staging_handler(self.kv_manager)
        if self.kv_manager.late_d is not None:
            self.late_d = LateDDecode(
                self.kv_manager.late_d,
                LateDConfig.from_disagg(get_disagg(), None),
                metrics=make_late_d_metrics(scheduler, self.kv_manager.late_d),
            )

        if (
            self.scheduler.tp_worker.is_hybrid_swa
            and not self._uses_swa_tail_prealloc()
        ):
            # Fallback for SWA allocators that still allocate the SWA pool at
            # full prompt length.
            self.max_total_num_tokens = min(
                self.max_total_num_tokens,
                self.scheduler.tp_worker.model_runner.swa_max_total_num_tokens,
            )

        enable_kv_checksum = get_disagg().disaggregation_enable_kv_checksum
        self.kv_checksum_signature = 0
        # Test-only mismatch injector. Its own generator, seeded per rank: the
        # global `random` is seeded identically on every TP rank, so drawing
        # from it would inject the same requests everywhere and never exercise
        # the mismatch all-reduce, while skipping draws on some ranks would
        # desync that shared stream.
        self.kv_checksum_mismatch_injector = random.Random(
            hash((self.scheduler.random_seed, get_parallel().attn_tp_rank))
        )
        if enable_kv_checksum:
            kv_args = self.kv_manager.kv_args
            self.kv_checksum_signature = kv_layout_signature(kv_args)
            self.scheduler.kv_checksum_computer = KvChecksumComputer(
                device=torch.device(f"cuda:{get_device().gpu_id}"),
                kv_data_ptrs=kv_args.kv_data_ptrs,
                kv_item_lens=kv_args.kv_item_lens,
                state_data_ptrs=kv_args.state_data_ptrs,
                state_item_lens=kv_args.state_item_lens,
            )
        else:
            self.scheduler.kv_checksum_computer = None

    def _uses_swa_tail_prealloc(self) -> bool:
        return (
            isinstance(self.token_to_kv_pool, (SWAKVPool, DeepSeekV4TokenToKVPool))
            and self.token_to_kv_pool_allocator.page_size > 1
            and hasattr(self.token_to_kv_pool_allocator, "alloc_extend_swa_tail")
        )

    def _uses_dspark_content_only_radix(self) -> bool:
        """Whether this request path uses the bounded DSV4/DSpark PoC.

        The server-argument and cache-builder guards remain the authoritative
        fail-closed policy. This runtime check intentionally stays narrower:
        it prevents the component-split lifecycle below from changing ordinary
        decode radix caching or another speculative algorithm.
        """
        disagg = get_disagg()
        enabled = disagg.disaggregation_decode_enable_radix_cache and (
            disagg.disaggregation_decode_radix_cache_mode == "content_only"
        )
        if not enabled:
            return False
        if not self.scheduler.spec_algorithm.is_dspark():
            raise RuntimeError(
                "content_only decode RadixCache reached runtime without DSPARK"
            )
        if not isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool):
            raise RuntimeError(
                "content_only decode RadixCache reached runtime without "
                "DeepSeekV4TokenToKVPool"
            )
        return True

    def _uses_window_host_radix(self) -> bool:
        """Whether this decode worker runs DSv4 window SWA residency
        (``--disaggregation-decode-radix-cache-mode window_host``).

        Like content_only, the worker matches and locks only FULL, takes the
        fresh SWA tail page_floor(max(0, L - W)) from prefill and never inserts
        before finish. Unlike content_only, the finish insert covers every
        component: the SWA below the tail is backfilled host->host from the
        prefill's servicer, and decode spills later SWA to its own host.

        The server-argument and cache-builder guards are the fail-closed
        policy; this runtime check re-asserts the exact scope.
        """
        disagg = get_disagg()
        if not (
            disagg.disaggregation_decode_enable_radix_cache
            and disagg.disaggregation_decode_radix_cache_mode == "window_host"
        ):
            return False
        if not self.scheduler.spec_algorithm.is_dspark():
            raise RuntimeError(
                "window_host decode RadixCache reached runtime without DSPARK"
            )
        if not isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool):
            raise RuntimeError(
                "window_host decode RadixCache reached runtime without "
                "DeepSeekV4TokenToKVPool"
            )
        if getattr(self.tree_cache, "dsv4_uniform_swa_residency", "full") != "window":
            raise RuntimeError(
                "window_host decode RadixCache reached runtime without "
                "--dsv4-uniform-swa-residency window"
            )
        return True

    def _mark_window_host_prebuilt(self, req: Req) -> None:
        """window_host: the transferred prompt holds SWA only from its tail on
        until the backfill commits, so no unfinished insert (PREBUILT or
        later) may publish it. The finish insert is all-component
        (``decode_radix_cache_frozen_prompt_len`` stays None) and adopts the
        request's host ledger."""
        if self._uses_window_host_radix():
            req.decode_radix_cache_skip_unfinished_insert = True

    def _freeze_dspark_prompt_cache_boundary(self, req: Req) -> None:
        """Freeze the committed prompt before DSpark can create draft state."""
        if not self._uses_dspark_content_only_radix():
            return

        if req.decode_radix_cache_frozen_prompt_len is None:
            req.decode_radix_cache_frozen_prompt_len = req.kv.kv_committed_len
            if self.tp_rank == 0 and self.pp_rank == 0:
                logger.debug(
                    "PD decode radix freeze: rid=%s bootstrap_room=%s "
                    "frozen_prompt_len=%d page_size=%d speculative_algorithm=DSPARK",
                    req.rid,
                    req.bootstrap_room,
                    req.decode_radix_cache_frozen_prompt_len,
                    self.token_to_kv_pool_allocator.page_size,
                )
        else:
            assert req.decode_radix_cache_frozen_prompt_len <= req.kv.kv_committed_len, (
                "Frozen decode-radix prompt boundary moved beyond committed KV: "
                f"rid={req.rid} frozen={req.decode_radix_cache_frozen_prompt_len} "
                f"committed={req.kv.kv_committed_len}"
            )

        # The finish path performs the only eligible insertion.  In particular,
        # process_prebuilt and verify-result processing must not remap the active
        # request or publish accepted/rejected speculative tails.
        req.decode_radix_cache_skip_unfinished_insert = True

    def _release_matched_prefix_lock(self, req: Req) -> None:
        # The TreeLock receipt replays the acquire (skipped components, an
        # early SWA release), so one unlock undoes _match_prefix_and_lock.
        self.tree_cache.unlock(req.lock)
        req.lock = None

    def _uses_swa_reservation(self) -> bool:
        return (
            self._uses_swa_tail_prealloc()
            or self.token_to_kv_pool_allocator.prealloc_fits_assumes_reclaim()
        )

    def _required_admission_tokens(
        self, req: Req, allocated_tokens: int, prefix_len: int, retractable_tokens: int
    ) -> int:
        return max(
            allocated_tokens + self.num_reserved_decode_tokens,
            self._rebootstrap_prefill_len(req)
            - prefix_len
            + min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
            - retractable_tokens,
        )

    def _prealloc_reservation_fits(
        self,
        full_tokens: int,
        swa_tokens: int,
        *,
        full_allocatable_tokens: int,
        swa_allocatable_tokens: Optional[int],
    ) -> bool:
        return self.token_to_kv_pool_allocator.prealloc_fits(
            self.tree_cache,
            full_tokens,
            swa_tokens,
            full_budget_tokens=full_allocatable_tokens,
            swa_budget_tokens=swa_allocatable_tokens,
        )

    def _reclaim_swa_tail_capacity(
        self, swa_tail_len: int, req_id: str, *, full_len: int = 0
    ) -> Optional[str]:
        allocator = self.token_to_kv_pool_allocator
        page_size = allocator.page_size
        shortfall = allocator.reclaim_for_prealloc(
            self.tree_cache,
            ceil_align(full_len, page_size),
            ceil_align(swa_tail_len, page_size),
        )
        return None if shortfall is None else f"{shortfall}, req={req_id}"

    # SWA caches expose full-attention accounting through full_* accessors.
    def _radix_full_evictable(self) -> int:
        if self.scheduler.tp_worker.is_hybrid_swa:
            return self.tree_cache.full_evictable_size()
        return self.tree_cache.evictable_size()

    def _radix_full_protected(self) -> int:
        if self.scheduler.tp_worker.is_hybrid_swa:
            return self.tree_cache.full_protected_size()
        return self.tree_cache.protected_size()

    def _radix_full_available(self) -> int:
        if self.scheduler.tp_worker.is_hybrid_swa:
            return self.token_to_kv_pool_allocator.full_available_size()
        return self.token_to_kv_pool_allocator.available_size()

    def _swa_tail_len(self, seq_len: int) -> int:
        if not self._uses_swa_tail_prealloc() or seq_len <= 0:
            return max(seq_len, 0)

        window_size = self.scheduler.sliding_window_size
        if window_size is None or window_size <= 0:
            return seq_len

        page_size = self.token_to_kv_pool_allocator.page_size
        if (
            get_disagg().disaggregation_decode_enable_radix_cache
            and not self._uses_dspark_content_only_radix()
            and not self._uses_window_host_radix()
        ):
            # Keep enough SWA before the page-aligned radix-cache insert
            # boundary for the cached key to contain a complete window.
            # `seq_len - 1` is the last committed position.
            window_start = max(0, seq_len - 1 - max(window_size, page_size))
        else:
            # content_only never publishes SWA. Reuse the proven ChunkCache
            # live-window allocation instead of retaining the extra page that
            # an all-component Radix insertion needs. window_host uses the
            # same tail: it must equal the range prefill sends,
            # [page_floor(max(dpl, L - W)), L) (prefill.py _swa_payload), and
            # the SWA below it comes from the host->host backfill.
            window_start = max(0, seq_len - window_size)
        window_start = (window_start // page_size) * page_size
        return seq_len - window_start

    def _resume_swa_tail_len(self, req: Req) -> int:
        """window_host resume: the retract re-attached
        in-flight spills and recorded S = kv.swa_evicted_seqlen, the
        page-aligned end of the SWA the host ledger holds. [S, fill_len) was
        device-resident at the retract and is in the CPU snapshot, so the
        resumed request allocates SWA from S: nothing below S lacks a host
        copy, and load_cpu_copy's subset check holds."""
        fill_len = self._pre_alloc_fill_len(req)
        start = int(req.swa_resume_start)
        page_size = self.token_to_kv_pool_allocator.page_size
        assert 0 <= start < fill_len and start % page_size == 0, (
            f"window_host resume start {start} outside [0, {fill_len}) or "
            f"not page aligned; rid={req.rid}"
        )
        return fill_len - start

    def _prealloc_swa_tail_len(self, req: Req, fill_len: int) -> int:
        """SWA tail a (re)allocation of ``req`` gets: the resume tail for a
        retracted window_host request, otherwise the transfer tail."""
        if (
            getattr(req, "swa_resume_start", None) is not None
            and self._uses_window_host_radix()
        ):
            return self._resume_swa_tail_len(req)
        return self._swa_tail_len(fill_len)

    def _swa_retractable_len(self, req: Req) -> int:
        if not self._uses_swa_tail_prealloc():
            return len(req.origin_input_ids) + len(req.output_ids)
        return self._swa_tail_len(len(req.origin_input_ids)) + len(req.output_ids)

    def _prealloc_kv_lens(self, req: Req) -> Tuple[int, int]:
        allocated_kv_len = self._pre_alloc_fill_len(req)
        if self._uses_swa_tail_prealloc():
            return allocated_kv_len, self._prealloc_swa_tail_len(req, allocated_kv_len)
        return allocated_kv_len, allocated_kv_len

    def _prealloc_required_tokens(self, req: Req) -> Tuple[int, int]:
        full_len, swa_len = self._prealloc_kv_lens(req)
        page_size = self.token_to_kv_pool_allocator.page_size
        if page_size > 1:
            # Match the allocator, which charges whole pages for both pools.
            full_len = ceil_align(full_len, page_size)
            swa_len = ceil_align(swa_len, page_size)
        swa_reserved = self.num_reserved_decode_tokens
        if get_memory().disable_radix_cache:
            swa_reserved = 0
        return (
            full_len + self.num_reserved_decode_tokens,
            swa_len + swa_reserved,
        )

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()

        attn_tp_size = get_parallel().attn_tp_size
        kv_args.engine_rank = self.tp_rank % (attn_tp_size)

        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = get_parallel().dp_rank
        kv_args.kv_cache_dtype_str = (
            self.scheduler.tp_worker.model_runner.kv_cache_dtype_str
        )
        kv_args.kv_layout = get_pd_kv_layout(self.token_to_kv_pool)
        transfer_kv_pool = (
            self.scheduler.hisparse_coordinator.mem_pool_host
            if self.scheduler.enable_hisparse
            else self.token_to_kv_pool
        )
        kv_data_ptrs, kv_data_lens, kv_item_lens = get_kv_transfer_buf_infos(
            transfer_kv_pool
        )
        kv_data_mem_kinds = (
            ["DRAM"] * len(kv_data_ptrs)
            if self.scheduler.enable_hisparse
            else ["VRAM"] * len(kv_data_ptrs)
        )
        if self.scheduler.enable_hisparse and isinstance(
            self.token_to_kv_pool, DeepSeekV4TokenToKVPool
        ):
            device_kv_data_ptrs, device_kv_data_lens, device_kv_item_lens = (
                self.token_to_kv_pool.get_contiguous_buf_infos()
            )
            c4_layer_num = self.scheduler.hisparse_coordinator.mem_pool_host.layer_num
            kv_data_ptrs += device_kv_data_ptrs[c4_layer_num:]
            kv_data_lens += device_kv_data_lens[c4_layer_num:]
            kv_item_lens += device_kv_item_lens[c4_layer_num:]
            kv_data_mem_kinds += ["VRAM"] * len(device_kv_data_ptrs[c4_layer_num:])
        num_draft_entries = 0
        if self.draft_token_to_kv_pool is not None:
            # Draft KV shares target virtual ids. Unified target KV is transferred
            # with physical ids, so it needs a separate draft index vector.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens
            kv_data_mem_kinds += ["VRAM"] * len(draft_kv_data_ptrs)
            num_draft_entries = len(draft_kv_data_ptrs)

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        kv_args.num_draft_entries = num_draft_entries
        kv_args.kv_layer_ids = build_kv_layer_ids(
            token_to_kv_pool=self.token_to_kv_pool,
            draft_token_to_kv_pool=self.draft_token_to_kv_pool,
            num_draft_entries=num_draft_entries,
            num_hidden_layers=self.scheduler.model_config.num_hidden_layers,
        )
        if self.transfer_backend == TransferBackend.NIXL:
            kv_args.kv_data_mem_kinds = kv_data_mem_kinds
        kv_args.page_size = self.token_to_kv_pool.page_size

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )

        setup_state_kv_args(
            kv_args,
            self.token_to_kv_pool,
            self.draft_token_to_kv_pool,
            total_kv_layers=self.scheduler.model_config.num_hidden_layers,
            req_to_token_pool=getattr(self, "req_to_token_pool", None),
        )
        if get_disagg().disaggregation_decode_host_receive_threshold < 1:
            pool = self.token_to_kv_pool
            group = self.tree_cache.host_pool_group
            if (kv_args.state_types and kv_args.state_types != [StateType.SWA]) or any(
                spec.indices_from_pool not in (PoolName.KV, PoolName.SWA)
                for spec in self.tree_cache.sidecar_pool_specs
            ):
                raise ValueError(
                    "Host receive supports full KV and separate paged SWA pools"
                )
            self.host_pool = group.get_pool(PoolName.KV)
            self.host_reserved_tokens = decode_retraction_max_tokens(
                self.req_to_token_pool, pool
            )
            if self.host_pool.logical_size < self.host_reserved_tokens + pool.page_size:
                raise ValueError(
                    "Host pool must hold a retraction and a receive page; "
                    "increase --hicache-size or --hicache-ratio"
                )

            def pool_names(source):
                return {source} | {
                    spec.pool_name
                    for spec in self.tree_cache.sidecar_pool_specs
                    if spec.indices_from_pool == source
                }

            device_buffers, host_buffers = group.get_contiguous_buf_infos(
                pool_names(PoolName.KV)
            )
            if device_buffers != (
                kv_args.kv_data_ptrs,
                kv_args.kv_data_lens,
                kv_args.kv_item_lens,
            ):
                raise ValueError(
                    "Host receive buffers must match the KV transfer layout"
                )
            (
                kv_args.host_kv_data_ptrs,
                kv_args.host_kv_data_lens,
                kv_args.host_kv_item_lens,
            ) = host_buffers
            self.host_swa_pool = None
            self.host_swa_reserved_tokens = 0
            if kv_args.state_types:
                self.host_swa_pool = group.get_pool(PoolName.SWA)
                self.host_swa_reserved_tokens = min(
                    decode_retraction_max_tokens(
                        self.req_to_token_pool, pool.swa_kv_pool
                    ),
                    ceil_align(
                        self.scheduler.sliding_window_size + pool.page_size - 1,
                        pool.page_size,
                    ),
                )
                if self.host_swa_pool.logical_size < (
                    self.host_swa_reserved_tokens + pool.page_size
                ):
                    raise ValueError(
                        "SWA host pool must hold a retraction and a receive page; "
                        "increase --hicache-size or --hicache-ratio"
                    )
                device_state, host_state = group.get_contiguous_buf_infos(
                    pool_names(PoolName.SWA)
                )
                if device_state != (
                    kv_args.state_data_ptrs[0],
                    kv_args.state_data_lens[0],
                    kv_args.state_item_lens[0],
                ):
                    raise ValueError(
                        "Host receive buffers must match the SWA transfer layout"
                    )
                (
                    kv_args.host_state_data_ptrs,
                    kv_args.host_state_data_lens,
                    kv_args.host_state_item_lens,
                ) = ([values] for values in host_state)

        kv_args.ib_device = get_disagg().disaggregation_ib_device
        kv_args.gpu_id = get_device().gpu_id
        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.DECODE,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        if (
            get_disagg().disaggregation_decode_host_receive_threshold < 1
            and not kv_manager.supports_host_destination
        ):
            raise ValueError(
                "Transfer backend does not support decode host KV destinations"
            )
        # Staging buffer setup (only when heterogeneous TP staging is enabled)
        if self.enable_staging and not self.is_mla_backend:
            kv_pool_for_heads = self.token_to_kv_pool
            if hasattr(kv_pool_for_heads, "full_kv_pool"):
                kv_pool_for_heads = kv_pool_for_heads.full_kv_pool
            per_rank_kv_heads = getattr(kv_pool_for_heads, "head_num", 0)
            if per_rank_kv_heads > 0:
                kv_args.kv_head_num = per_rank_kv_heads
                kv_args.total_kv_head_num = (
                    self.scheduler.model_config.get_total_num_kv_heads()
                )
            if hasattr(kv_manager, "set_kv_buffer_tensors"):
                kv_pool = kv_pool_for_heads
                staging_slots = build_staging_slot_metadata(
                    kv_layer_ids=kv_args.kv_layer_ids,
                    num_draft_entries=num_draft_entries,
                    kv_pool=kv_pool,
                    draft_kv_pool=self.draft_token_to_kv_pool,
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

    def add(
        self, req: Req, is_retracted: bool = False, is_rebootstrap: bool = False
    ) -> None:
        """Add a request to the pending queue.

        ``is_rebootstrap`` marks a PD true-retraction request whose prefix KV
        must be recomputed by the original prefill worker under the current
        weights (rather than resumed from stale CPU KV). It otherwise follows the
        same bootstrap-handshake path as a fresh request. Early allocation
        dispatches ``/generate`` after ``send_metadata``; deferred allocation
        dispatches once the peer is resolved, before waiting for completion.
        """
        # See `PrefillBootstrapQueue.add`. A retracted or rebootstrapping
        # request owns a host KV backup that `retracted_queue` releases, and by
        # this point carries none of the markers `is_unadmitted_reject` reads,
        # so take the caller's word for it rather than sniffing.
        if not is_retracted and not is_rebootstrap and is_unadmitted_reject(req):
            self.scheduler.retire_unadmitted_request(req)
            return
        if is_retracted and dsv4_needs_rebootstrap(self.token_to_kv_pool_allocator):
            if req.output_ids:
                req.pd_rebootstrap_forced_output_id = req.output_ids.pop()
            req.pd_rebootstrap_in_progress = True
            req.time_stats.set_retract_time()
            is_retracted = False
            is_rebootstrap = True

        if self._check_if_req_exceed_kv_capacity(req):
            if is_retracted:
                # Aborted instead of queued for resume: a window_host retract
                # kept the request's host SWA ledger (KEEP), release it now.
                release_retracted_swa_ledger(req, self.tree_cache)
                # Its cpu_tensor backup goes with the request.
                release_retract_backup(req, "abort")
            return

        if is_retracted:
            req.retraction_mb_id = None
            self.retracted_queue.append(req)
        else:
            decode_req = self._create_receiver_and_enqueue(
                req, is_rebootstrap=is_rebootstrap
            )

            # NOTE: fake transfer does not need to resolve prefill dp rank in the pending queue
            if _is_fake_transfer(req):
                decode_req.kv_receiver.init(0)
                return

            # Fast path: cache-only lookup, no network calls
            prefill_dp_rank = self._resolve_prefill_dp_rank(req)
            logger.debug(f"prefill_dp_rank: {prefill_dp_rank}")
            if prefill_dp_rank is not None:
                self._init_receiver(decode_req, prefill_dp_rank)
                if self.late_d is not None and decode_req.late_d_state is not None:
                    self.late_d.send_intent(decode_req)
                return

            self.pending_reqs.append(decode_req)

    def _dspark_content_match_limit(self, fill_len: int) -> int:
        """Largest FULL prefix that leaves the live SWA window fresh."""
        fresh_state_tokens = self._swa_tail_len(fill_len)
        return page_align_floor(
            max(0, fill_len - fresh_state_tokens),
            self.token_to_kv_pool_allocator.page_size,
        )

    def _fresh_swa_window_match(self, fill_len: int) -> Tuple[bool, Optional[int]]:
        """(FULL-only, key_limit) of a decode-radix admission match.

        content_only and window_host take the live SWA window fresh from
        prefill, so they match and lock FULL only, capped at the tail start
        page_floor(fill_len - tail). Under window_host the matched prefix
        keeps its SWA on some tier (I1), but decode never reads it. Other modes
        match every component without a cap."""
        fresh = self._uses_dspark_content_only_radix() or self._uses_window_host_radix()
        return fresh, (self._dspark_content_match_limit(fill_len) if fresh else None)

    def _match_prefix_and_lock(
        self,
        req: Req,
        *,
        content_only: bool = False,
        key_limit: Optional[int] = None,
    ) -> DecodePrefixMatch:
        max_prefix_len = None
        if self._uses_swa_tail_prealloc():
            fill_len = self._pre_alloc_fill_len(req)
            max_prefix_len = fill_len - self._swa_tail_len(fill_len)
        if key_limit is not None:
            # content_only / window_host (_fresh_swa_window_match): FULL only,
            # capped at the start of the fresh SWA tail.
            max_prefix_len = (
                key_limit if max_prefix_len is None else min(max_prefix_len, key_limit)
            )
        active_components = (ComponentType.FULL,) if content_only else None
        # Only content_only / window_host restrict the matched components.
        match_kwargs = (
            {} if active_components is None else {"active_components": active_components}
        )
        # Match and lock only reusable FULL KV. The entire SWA tail must be
        # freshly allocated, including when the prefix comes from L2/L3.
        result = match_kv_cache(
            req,
            self.tree_cache,
            req.origin_input_ids,
            cow_mamba=self.tree_cache.supports_mamba(),
            max_prefix_len=max_prefix_len,
            **match_kwargs,
        )
        if content_only:
            # DSV4/DSpark always transfers a fresh live SWA/request-state
            # window. Locking only Full mirrors the component set used for
            # matching; the lock's receipt records SWA as skipped, so the
            # release never decrements a lock we never acquired.
            req.lock = self.tree_cache.lock(
                result.last_device_node, skip_lock_components=(ComponentType.SWA,)
            )
        else:
            req.lock = self.tree_cache.lock(result.last_device_node)
        return self._build_decode_prefix_match(
            req,
            result,
            max_prefix_len=max_prefix_len,
            active_components=active_components,
        )

    def _resolve_prefill_dp_rank(self, req: Req) -> Optional[int]:
        prefill_info = self.kv_manager.prefill_info_table.get(_bootstrap_addr(req))
        # If None, it will go to the slow path and resolve prefill_info by _ensure_prefill_info then cache it
        if prefill_info is None:
            return None

        if req.disagg_prefill_dp_rank is not None:
            return req.disagg_prefill_dp_rank

        if prefill_info.dp_size == 1:
            return 0

        if (
            prefill_info.follow_bootstrap_room
            and not envs.SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK.get()
        ):
            return req.bootstrap_room % prefill_info.dp_size

        return None

    def _create_receiver_and_enqueue(
        self, req: Req, is_rebootstrap: bool = False
    ) -> DecodeRequest:
        backend = (
            TransferBackend.FAKE if _is_fake_transfer(req) else self.transfer_backend
        )
        kv_receiver_class = get_kv_class(backend, KVClassType.RECEIVER)

        kv_receiver = kv_receiver_class(
            mgr=self.kv_manager,
            bootstrap_addr=_bootstrap_addr(req),
            bootstrap_room=req.bootstrap_room,
        )

        decode_req = DecodeRequest(
            req=req, kv_receiver=kv_receiver, is_rebootstrap=is_rebootstrap
        )
        # Late D: hold the request DEFERRED until the prefill admits it. FAKE
        # rooms (warmup) and rebootstraps (decode-initiated) bind as today.
        if (
            self.late_d is not None
            and not is_rebootstrap
            and backend != TransferBackend.FAKE
        ):
            self.late_d.defer(decode_req)
            late_field = getattr(req, "late_d", None)
            if late_field is not None:
                # Stage 2: one of K candidates (validated at the scheduler).
                decode_req.late_d_cand_idx = late_field["cand_idx"]
        self.queue.append(decode_req)
        return decode_req

    def hold_rebootstrap(self, req: Req) -> None:
        """Stage a retracted request for rebootstrap without enqueuing it yet.

        Retraction is always paired with a weight update
        (``pause_generation(mode="retract")`` -> ``update_weights`` ->
        ``continue_generation``). Enqueuing the rebootstrap into ``self.queue``
        here would leave the preallocation queue non-empty, which makes the
        scheduler non-idle so ``update_weights``' post-update cache flush
        asserts and crashes the decode worker. Instead we hold the request and
        enqueue it from ``enqueue_held_rebootstrap`` on resume, so its prefix KV
        is recomputed by the prefill worker under the updated weights.
        """
        self.held_rebootstrap_reqs.append(req)

    def enqueue_held_rebootstrap(self) -> None:
        """Enqueue all staged rebootstrap requests when generation resumes."""
        held = self.held_rebootstrap_reqs
        self.held_rebootstrap_reqs = []
        for req in held:
            self.add(req, is_rebootstrap=True)

    @staticmethod
    def _rebootstrap_prefill_len(req: Req) -> int:
        if getattr(req, "pd_rebootstrap_in_progress", False):
            return len(req.origin_input_ids) + len(req.output_ids)
        return len(req.origin_input_ids)

    @staticmethod
    def _pre_alloc_fill_len(req: Req) -> int:
        if getattr(req, "pd_rebootstrap_in_progress", False):
            # pause_generation(retract) already popped the boundary token out of
            # output_ids (it is replayed via the decode-side override at commit
            # time), so output_ids here is prompt + emitted-tokens-minus-boundary,
            # i.e. the original seqlen - 1. The prefill recomputes KV for *all* of
            # these tokens, leaving no just-sampled "pending" token in the list, so
            # we allocate exactly len(origin)+len(output_ids) with no -1 (unlike
            # normal decode, where the last token's KV has not been written yet).
            # This is the same token count as offloading-based retraction, where
            # offload_kv_cache saves seqlen-1 tokens; the boundary token's KV is
            # (re)computed on the decode side once generation resumes.
            return len(req.origin_input_ids) + len(req.output_ids)
        return len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        message = None
        allocator = self.token_to_kv_pool_allocator
        full_required, swa_required = self._prealloc_required_tokens(req)
        if not self._uses_swa_tail_prealloc():
            swa_required = full_required
        ceiling_fits = allocator.prealloc_ceiling_fits(full_required, swa_required)
        if ceiling_fits is False:
            message = (
                f"Request {req.rid} exceeds the unified FULL/SWA KV byte "
                f"budget: full={full_required}, swa={swa_required}"
            )
        elif ceiling_fits is None:
            # HiSparse admits up to the host-backed logical capacity.
            capacity = (
                self.scheduler.tp_worker.model_runner.max_token_pool_size
                if self.scheduler.enable_hisparse
                else self.max_total_num_tokens
            )
            input_len = self._rebootstrap_prefill_len(req)
            if input_len > capacity:
                message = f"Request {req.rid} exceeds the maximum number of tokens: {input_len} > {capacity}"
            elif self._uses_swa_tail_prealloc():
                _, swa_required = self._prealloc_required_tokens(req)
                swa_capacity = self.token_to_kv_pool_allocator.size_swa
                if swa_required > swa_capacity:
                    message = (
                        f"Request {req.rid} requires too many SWA KV tokens for "
                        f"decode preallocation: {swa_required} > {swa_capacity}"
                    )
        if message is not None:
            logger.error(message)
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.output_streamer.stream_output([req], req.return_logprob)
            return True
        return False

    def extend(self, reqs: List[Req], is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        for req in reqs:
            self.add(req, is_retracted=is_retracted)

    def release_memory_occupation(self):
        self._cancel_prefill_dp_rank_queries()
        self.queue.clear()
        for req in self.retracted_queue:
            discard_kv_cache_backup(
                req,
                self.tree_cache,
                get_disagg().disaggregation_decode_retraction_backup,
            )
        self.retracted_queue.clear()
        if hasattr(self.kv_manager, "deregister_buffer_to_engine"):
            self.kv_manager.deregister_buffer_to_engine()

    def resume_memory_occupation(self):
        if hasattr(self.kv_manager, "register_buffer_to_engine"):
            self.kv_manager.register_buffer_to_engine()

    def resume_retracted_reqs(
        self, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        # TODO refactor the scheduling part, reuse with the unified engine logic as much as possible

        # allocate memory
        resumed_reqs = []
        indices_to_remove = set()
        swa_allocatable_tokens = None
        if self._uses_swa_reservation():
            full_allocatable_tokens, swa_allocatable_tokens = (
                self._swa_aware_allocatable_token_budgets(count_retracted=False)
            )
        else:
            full_allocatable_tokens = self._allocatable_token_budgets(
                count_retracted=False
            )

        for i, req in enumerate(self.retracted_queue):
            if rids_to_check is not None and req.rid not in rids_to_check:
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break

            if not ensure_mamba_capacity(
                self.req_to_token_pool, [req], self.tree_cache
            ):
                break

            full_required, swa_required = self._prealloc_required_tokens(req)
            if not self._prealloc_reservation_fits(
                full_required,
                swa_required,
                full_allocatable_tokens=full_allocatable_tokens,
                swa_allocatable_tokens=swa_allocatable_tokens,
            ):
                break
            if (
                self._uses_swa_tail_prealloc()
                and getattr(req, "swa_resume_start", None) is not None
                and self._uses_window_host_radix()
                and self._reclaim_swa_tail_capacity(
                    self._prealloc_kv_lens(req)[1], req.rid
                )
                is not None
            ):
                # window_host resumes SWA from S, often more than the
                # transfer tail: demote tree SWA for it first (the budget
                # counts evictable SWA, the allocator does not evict).
                break

            if self.token_to_kv_pool_allocator.prealloc_fits_assumes_reclaim():
                full_len, swa_len = self._prealloc_kv_lens(req)
                if (
                    self._reclaim_swa_tail_capacity(swa_len, req.rid, full_len=full_len)
                    is not None
                ):
                    break

            resumed_reqs.append(req)
            indices_to_remove.add(i)
            req.is_retracted = False
            self._pre_alloc(req)
            if get_disagg().disaggregation_decode_enable_radix_cache:
                # Retraction restore repopulates request-private KV; it does
                # not acquire a radix-tree lock. `None` is the explicit no-lock
                # sentinel: the prebuilt path binds such a request at the root
                # (_bind_root_prefix) instead of re-matching it. Reusing a
                # stale node would make a later finish/second retraction
                # release a lock this request never took.
                req.last_node = None
            full_allocatable_tokens = self._allocatable_token_budgets(
                count_retracted=False,
                extra_reserved_reqs=len(resumed_reqs),
                extra_reqs=resumed_reqs,
            )
            if swa_allocatable_tokens is not None:
                swa_allocatable_tokens = self._swa_tail_allocatable_token_budget(
                    count_retracted=False,
                    extra_reserved_reqs=len(resumed_reqs),
                    extra_reqs=resumed_reqs,
                )

            if self.scheduler.enable_overlap:
                # A request that finished last step can still have its forward in
                # flight, writing slots this request was just given.
                self.scheduler.schedule_stream.wait_stream(
                    self.scheduler.forward_stream
                )
            restore_kv_cache(
                req,
                self.tree_cache,
                self.req_to_token_pool,
                self.token_to_kv_pool_allocator,
                get_disagg().disaggregation_decode_retraction_backup,
            )
            # window_host: the resume tail start is consumed; the ledger it
            # matches stays on the request until finish.
            req.swa_resume_start = None

        self.retracted_queue = [
            entry
            for i, entry in enumerate(self.retracted_queue)
            if i not in indices_to_remove
        ]

        return resumed_reqs

    def _init_receiver(self, decode_req: DecodeRequest, prefill_dp_rank: int) -> None:
        decode_req.kv_receiver.init(prefill_dp_rank)
        if (
            get_disagg().disaggregation_decode_allocation_policy == "prefill_complete"
            and decode_req.kv_receiver.conclude_state != KVPoll.Failed
        ):
            decode_req.req.disagg_prefill_dp_rank = prefill_dp_rank
            # Deferred allocation cannot precede the recompute that makes its
            # source ready. Keep the existing leader election and error path.
            if decode_req.is_rebootstrap:
                self._submit_prefill_recompute(decode_req)

    def _submit_prefill_recompute(self, decode_req: DecodeRequest) -> None:
        # A fake transfer has no prefill; its receiver completes once metadata is sent.
        if _is_fake_transfer(decode_req.req):
            return
        self.kv_manager.submit_prefill_recompute(
            decode_req.kv_receiver, decode_req.req.build_rebootstrap_payload()
        )

    def _update_handshake_waiters(
        self,
        rids_to_check: Optional[List[str]] = None,
        pp_good_rids: Optional[List[str]] = None,
        pp_bad_rids: Optional[List[str]] = None,
    ) -> None:
        if not self.queue:
            return

        # Receiver polling observes asynchronous failures while KV allocation
        # is blocked.
        if self.pp_size > 1:
            polls = poll_and_all_reduce_pp(
                (decode_req.req.rid for decode_req in self.queue),
                KVPoll.WaitingForInput,
                pp_good_rids,
                pp_bad_rids,
            )
        else:
            polls = poll_and_all_reduce(
                [decode_req.kv_receiver for decode_req in self.queue], self.gloo_group
            )

        for decode_req, poll in zip(self.queue, polls):
            if poll is None:
                continue
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if poll == KVPoll.Bootstrapping:
                pass
            elif poll == KVPoll.WaitingForInput:
                decode_req.waiting_for_input = True
                decode_req.req.time_stats.set_bootstrap_done_time()
            elif poll == KVPoll.Failed:
                error_message = f"Decode handshake failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                is_propagated = False
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                    is_propagated = getattr(e, "is_from_another_rank", False)
                # Mute error message for propagated exceptions to avoid duplicate logging
                if is_propagated:
                    logger.debug(error_message)
                else:
                    logger.error(error_message)
                late_d_err = getattr(decode_req, "late_d_err_type", None)
                status = HTTPStatus.INTERNAL_SERVER_ERROR
                if late_d_err is not None:
                    status = HTTPStatus.SERVICE_UNAVAILABLE  # late D
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=status,
                    err_type=late_d_err,
                )
                if self.scheduler.metrics_reporter.enable_metrics:
                    self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

    def _ensure_prefill_info(
        self, addr_to_reqs: Dict[str, List[DecodeRequest]]
    ) -> Tuple[Dict[str, List[DecodeRequest]], List[DecodeRequest]]:
        """Non-blocking ensure parallel info for each addr.
        Returns (ready_addrs, remaining_reqs)."""
        ready: Dict[str, List[DecodeRequest]] = {}
        remaining: List[DecodeRequest] = []

        now = time.monotonic()
        for bootstrap_addr, reqs in addr_to_reqs.items():
            last_attempt = self._ensure_last_attempt_time.get(bootstrap_addr)
            if last_attempt is not None and (
                now - last_attempt < self._ensure_retry_interval
            ):
                remaining.extend(reqs)
                continue

            self._ensure_last_attempt_time[bootstrap_addr] = now

            if self.kv_manager.try_ensure_parallel_info(bootstrap_addr):
                if bootstrap_addr in self._ensure_retry_count:
                    del self._ensure_retry_count[bootstrap_addr]
                if bootstrap_addr in self._ensure_last_attempt_time:
                    del self._ensure_last_attempt_time[bootstrap_addr]
                ready[bootstrap_addr] = reqs
                continue

            count = self._ensure_retry_count.get(bootstrap_addr, 0) + 1
            self._ensure_retry_count[bootstrap_addr] = count

            if count >= self._max_ensure_retries:
                error_msg = f"Could not fetch prefill parallel info from {bootstrap_addr} after {count} attempts"
                logger.error(error_msg)
                for decode_req in reqs:
                    # kv_receiver may be None from a prior self.queue cleanup
                    if decode_req.kv_receiver is not None:
                        decode_req.kv_receiver.abort()
                del self._ensure_retry_count[bootstrap_addr]
                del self._ensure_last_attempt_time[bootstrap_addr]
            else:
                remaining.extend(reqs)

        return ready, remaining

    def prefetch_prefill_dp_rank_queries(self) -> None:
        """Start DP-rank lookups before their normal consume point."""
        if not self.pending_reqs:
            return

        queries = self._prefill_dp_rank_queries

        addr_to_reqs: Dict[str, List[DecodeRequest]] = {}
        for decode_req in self.pending_reqs:
            addr = _bootstrap_addr(decode_req.req)
            addr_to_reqs.setdefault(addr, []).append(decode_req)

        for bootstrap_addr in set(queries) - set(addr_to_reqs):
            _, stale_future = queries.pop(bootstrap_addr)
            stale_future.cancel()

        for bootstrap_addr, decode_reqs in addr_to_reqs.items():
            if bootstrap_addr in queries:
                continue
            if self.kv_manager.prefill_info_table.get(bootstrap_addr) is None:
                continue

            rooms = tuple(
                decode_req.req.bootstrap_room
                for decode_req in decode_reqs
                if self._resolve_prefill_dp_rank(decode_req.req) is None
            )
            if not rooms:
                continue

            future = self.kv_manager._ensure_prefill_recompute_executor().submit(
                CommonKVReceiver.query_prefill_dp_ranks,
                bootstrap_addr,
                list(rooms),
            )
            queries[bootstrap_addr] = (rooms, future)

    def _cancel_prefill_dp_rank_queries(self) -> None:
        for _, future in self._prefill_dp_rank_queries.values():
            future.cancel()
        self._prefill_dp_rank_queries.clear()

    def _resolve_pending_reqs(self) -> None:
        """Batch-resolve prefill_dp_ranks for pending requests and initialize receivers."""
        if not self.pending_reqs:
            self._cancel_prefill_dp_rank_queries()
            return

        # Group pending requests by bootstrap_addr
        addr_to_reqs: Dict[str, List[DecodeRequest]] = {}
        for decode_req in self.pending_reqs:
            addr = _bootstrap_addr(decode_req.req)
            addr_to_reqs.setdefault(addr, []).append(decode_req)

        # Pass 1: ensure parallel info for each addr
        ready_addrs, remaining = self._ensure_prefill_info(addr_to_reqs)

        resolved: List[Tuple[DecodeRequest, int]] = []
        for bootstrap_addr, decode_reqs in ready_addrs.items():
            need_query: List[DecodeRequest] = []
            for decode_req in decode_reqs:
                prefill_dp_rank = self._resolve_prefill_dp_rank(decode_req.req)
                if prefill_dp_rank is not None:
                    resolved.append((decode_req, prefill_dp_rank))
                else:
                    need_query.append(decode_req)

            # Pass 2: resolve dp rank for addrs whose info is available
            if need_query:
                rooms = [decode_req.req.bootstrap_room for decode_req in need_query]
                prefetched = self._prefill_dp_rank_queries.pop(bootstrap_addr, None)
                prefetched_rooms = prefetched[0] if prefetched is not None else ()
                if (
                    prefetched is not None
                    and tuple(rooms[: len(prefetched_rooms)]) == prefetched_rooms
                ):
                    room_to_rank = prefetched[1].result()
                    remaining_rooms = rooms[len(prefetched_rooms) :]
                    if remaining_rooms:
                        room_to_rank.update(
                            CommonKVReceiver.query_prefill_dp_ranks(
                                bootstrap_addr, remaining_rooms
                            )
                        )
                else:
                    if prefetched is not None:
                        prefetched[1].cancel()
                    room_to_rank = CommonKVReceiver.query_prefill_dp_ranks(
                        bootstrap_addr, rooms
                    )
                for decode_req in need_query:
                    prefill_dp_rank = room_to_rank.get(
                        str(decode_req.req.bootstrap_room)
                    )
                    if prefill_dp_rank is not None:
                        resolved.append((decode_req, int(prefill_dp_rank)))
                    else:
                        remaining.append(decode_req)
            else:
                prefetched = self._prefill_dp_rank_queries.pop(bootstrap_addr, None)
                if prefetched is not None:
                    prefetched[1].cancel()

        self.pending_reqs = remaining

        for decode_req, prefill_dp_rank in resolved:
            self._init_receiver(decode_req, prefill_dp_rank)
            if self.late_d is not None and decode_req.late_d_state is not None:
                self.late_d.send_intent(decode_req)

    def pop_preallocated(
        self,
        rids_to_check: Optional[List[str]] = None,
        pp_good_rids: Optional[List[str]] = None,
        pp_bad_rids: Optional[List[str]] = None,
    ) -> Tuple[List[DecodeRequest], List[DecodeRequest]]:
        """Pop the preallocated requests from the pending queue (FIFO)."""
        is_pp_mode = self.pp_size > 1
        if is_pp_mode and (pp_good_rids is None or pp_bad_rids is None):
            raise ValueError("PP consensus is required when pp_size > 1")
        if is_pp_mode and rids_to_check is not None:
            raise ValueError("rids_to_check cannot be used in PP mode")

        self._resolve_pending_reqs()
        self._update_handshake_waiters(rids_to_check, pp_good_rids, pp_bad_rids)
        if is_pp_mode:
            rids_to_check = set(pp_good_rids) | set(pp_bad_rids)

        failed_reqs = []
        preallocated_reqs = []
        num_device_preallocated = 0
        indices_to_remove = set()

        # We need to make sure that the sum of inflight tokens and allocatable tokens is greater than maximum input+output length of each inflight request
        # Otherwise it is possible for one request running decode out of memory, while all other requests are in the transfer queue that cannot be retracted.
        retractable_tokens = sum(
            len(r.origin_input_ids) + len(r.output_ids)
            for r in self.scheduler.running_batch.reqs
        )

        uses_swa_tail_prealloc = self._uses_swa_tail_prealloc()
        # SGLANG_DECODE_ADMIT_GROWTH_RATIO: 0.0 = off (and on the SWA
        # reservation path).
        growth_ratio = self._growth_ratio_applies()

        def _admitted_reqs(*extra: Req) -> Sequence[Req]:
            # The growth reserve names the requests this pass admitted with
            # device KV (a host-staged request holds none yet).
            if growth_ratio <= 0:
                return ()
            return [
                *(dr.req for dr in preallocated_reqs if not dr.host_staged),
                *extra,
            ]

        # SGLANG_DECODE_ADMIT_CAP_SAFE_RETRACT: a running request that a
        # retraction now would abort (its cpu_tensor backup over
        # SGLANG_RETRACT_BACKUP_MAX_GB) is not retractable.
        cap_unsafe_rids = (
            frozenset()
            if self._uses_swa_reservation()
            else self._cap_unsafe_running_rids()
        )
        if cap_unsafe_rids:
            retractable_tokens -= sum(
                len(r.origin_input_ids) + len(r.output_ids)
                for r in self.scheduler.running_batch.reqs
                if r.rid in cap_unsafe_rids
            )

        swa_allocatable_tokens = None
        if self._uses_swa_reservation():
            retractable_swa_tokens = sum(
                self._swa_retractable_len(r) for r in self.scheduler.running_batch.reqs
            )
            full_allocatable_tokens, swa_allocatable_tokens = (
                self._swa_aware_allocatable_token_budgets(
                    retractable_tokens=retractable_tokens,
                    retractable_swa_tokens=retractable_swa_tokens,
                    count_retracted=True,
                )
            )
        else:
            retractable_swa_tokens = 0
            full_allocatable_tokens = self._allocatable_token_budgets(
                retractable_tokens=retractable_tokens,
                count_retracted=True,
                cap_unsafe_rids=cap_unsafe_rids,
            )
        reserved_restore_tokens = self._hicache_pending_restore_tokens()
        full_allocatable_tokens -= reserved_restore_tokens
        # Sort by priority before any index-based bookkeeping so that both the
        # abort-scan loop and the preallocation loop operate on the same order.
        if self.scheduler.enable_priority_scheduling:
            priority_sign = (
                1 if self.scheduler.schedule_low_priority_values_first else -1
            )
            self.queue.sort(key=lambda r: r.req.priority * priority_sign)

        # First, remove all failed requests from the queue
        for i, decode_req in enumerate(self.queue):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue
            if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                if not getattr(decode_req.req, "finished_output", False):
                    self.scheduler.output_streamer.stream_output(
                        [decode_req.req],
                        decode_req.req.return_logprob,
                    )
                decode_req.kv_receiver.clear()
                decode_req.kv_receiver = None
                failed_reqs.append(decode_req)
                indices_to_remove.add(i)

        # DecodeRequest is shared between self.queue and self.pending_reqs;
        # drop failed reqs from both
        if failed_reqs:
            failed_ids = {id(r) for r in failed_reqs}
            self.pending_reqs = [
                r for r in self.pending_reqs if id(r) not in failed_ids
            ]

        # HiSparse physical constraint: max requests by device buffer capacity.
        # Each admitted req needs a padded buffer plus its speculative scratch.
        # waiting_queue reqs already have device buffers (allocated in admit_request_direct),
        # only transfer_queue reqs are pending device buffer allocation.
        hisparse_req_budget = float("inf")
        if self.scheduler.enable_hisparse:
            coordinator = self.scheduler.hisparse_coordinator
            per_req_device_tokens = (
                coordinator.padded_buffer_size + coordinator.spec_swap.scratch_capacity
            )
            hisparse_avail = (
                self.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size()
            )
            hisparse_req_budget = max(
                0,
                hisparse_avail // per_req_device_tokens
                - len(self.transfer_queue.queue),
            )

        if self.scheduler.enable_lora:
            running_batches = (
                self.scheduler.running_mbs
                if is_pp_mode
                else (self.scheduler.running_batch,)
            )
            # Include finished requests; GPU work may still use their adapters.
            running_loras = {
                req.lora_id for batch in running_batches for req in batch.reqs
            }
            running_loras.update(r.req.lora_id for r in self.transfer_queue.queue)
            running_loras.update(req.lora_id for req in self.scheduler.waiting_queue)

        # Then, preallocate the remaining requests if possible. Late D visits
        # the ADMITted entries after the others and never the DEFERRED ones.
        late_d = self.late_d
        # (full, swa) tokens the active late-D claim keeps free in this pass.
        claim_need: Optional[Tuple[int, int]] = None
        for i, decode_req in (
            enumerate(self.queue)
            if late_d is None
            else late_d.walk_order(self.queue, indices_to_remove)
        ):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if i in indices_to_remove:
                continue

            if not decode_req.waiting_for_input:
                continue

            if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
                break

            if self.scheduler.enable_lora and not self.scheduler.can_schedule_lora_req(
                decode_req.req, running_loras
            ):
                continue

            if (
                get_disagg().disaggregation_decode_host_receive_threshold < 1
                and not decode_req.is_rebootstrap
                and not _is_fake_transfer(decode_req.req)
                and decode_req.kv_receiver.supports_host_destination
                and self.scheduler.pool_stats_observer.get_pool_stats().full_token_usage
                >= get_disagg().disaggregation_decode_host_receive_threshold
            ):
                if not self._pre_alloc_host(decode_req):
                    break
                preallocated_reqs.append(decode_req)
                indices_to_remove.add(i)
                if self.scheduler.enable_lora:
                    running_loras.add(decode_req.req.lora_id)
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break

            # Reserve enough Mamba capacity for radix COW and tracking buffers
            # before prefix matching can bind state or lock cached entries.
            if not ensure_mamba_capacity(
                self.req_to_token_pool, [decode_req.req], self.tree_cache
            ):
                break

            if hisparse_req_budget <= 0:
                break

            # Memory estimation: don't add if the projected memory cannot be met
            # TODO: add new_token ratio
            origin_input_len = self._rebootstrap_prefill_len(decode_req.req)
            prefix_match: Optional[DecodePrefixMatch] = None
            content_only = False
            content_match_limit: Optional[int] = None
            use_decode_radix_cache = (
                get_disagg().disaggregation_decode_enable_radix_cache
                and not decode_req.is_rebootstrap
            )
            if use_decode_radix_cache:
                fill_len = self._pre_alloc_fill_len(decode_req.req)
                content_only, content_match_limit = self._fresh_swa_window_match(
                    fill_len
                )
                # Match prefix against decode's radix cache.
                prefix_match = self._match_prefix_and_lock(
                    decode_req.req,
                    content_only=content_only,
                    key_limit=content_match_limit,
                )
                prefix_indices = prefix_match.prefix_indices
                # prefix_len: tokens already on device (L1 hit).
                # total_prefix_len: full prefix promised to prefill
                # (L1 + L2 host hit + L3 storage hit), sent as PD
                # protocol's `decode_prefix_len`. The [prefix_len, total)
                # gap is filled by HiCache loadback later.
                prefix_len = prefix_match.l1_prefix_len
                total_prefix_len = prefix_match.decode_prefix_len

                # Decode transfers the SWA tail fresh, so retain only the
                # full-attention prefix lock needed for reuse.
                if (
                    uses_swa_tail_prealloc
                    and not content_only
                    and prefix_match.l1_prefix_len > 0
                    and hasattr(self.tree_cache, "release_swa")
                ):
                    self.tree_cache.release_swa(decode_req.req.lock)

                required_alloc_tokens = self._required_alloc_tokens(
                    fill_len=fill_len, prefix_len=prefix_len
                )
                # Matching may lock previously-evictable radix pages, so refresh
                # the admission budget against the post-lock pool state before we
                # decide whether this request still fits.
                full_allocatable_tokens = self._allocatable_token_budgets(
                    retractable_tokens=retractable_tokens,
                    count_retracted=True,
                    extra_reserved_reqs=num_device_preallocated,
                    hicache_reserved_tokens=reserved_restore_tokens,
                    extra_reqs=_admitted_reqs(),
                    cap_unsafe_rids=cap_unsafe_rids,
                )
                if uses_swa_tail_prealloc:
                    swa_allocatable_tokens = self._swa_tail_allocatable_token_budget(
                        retractable_tokens=retractable_tokens,
                        retractable_swa_tokens=retractable_swa_tokens,
                        count_retracted=True,
                        extra_reserved_reqs=num_device_preallocated,
                        extra_reqs=[
                            dr.req for dr in preallocated_reqs if not dr.host_staged
                        ],
                    )
            else:
                prefix_indices = None
                prefix_len = 0
                total_prefix_len = 0
                required_alloc_tokens = self._required_alloc_tokens(
                    fill_len=self._pre_alloc_fill_len(decode_req.req), prefix_len=0
                )

            full_required_for_admission = self._required_admission_tokens(
                decode_req.req, required_alloc_tokens, prefix_len, retractable_tokens
            )
            if growth_ratio > 0:
                # The candidate keeps its own decode growth, as the active
                # requests do (_active_reserved_tokens).
                required_tokens_for_request = required_alloc_tokens + (
                    _decode_growth_reserve(
                        decode_req.req, growth_ratio, self.num_reserved_decode_tokens
                    )
                )
                full_required_for_admission = max(
                    full_required_for_admission, required_tokens_for_request
                )
            else:
                required_tokens_for_request = (
                    required_alloc_tokens + self.num_reserved_decode_tokens
                )
            swa_required_for_admission = 0
            swa_len = required_alloc_tokens
            if uses_swa_tail_prealloc:
                _, swa_required = self._prealloc_required_tokens(decode_req.req)
                _, swa_len = self._prealloc_kv_lens(decode_req.req)
                max_new_tokens = min(
                    decode_req.req.sampling_params.max_new_tokens,
                    CLIP_MAX_NEW_TOKEN,
                )
                swa_required_for_admission = max(
                    swa_required,
                    swa_len + max_new_tokens - retractable_swa_tokens,
                )
            elif swa_allocatable_tokens is not None:
                swa_required_for_admission = full_required_for_admission

            late = late_d is not None and decode_req.late_d_state is not None
            if late:
                # Late D: the projected-growth checks below, net of the
                # claim's reservation. A miss HOLDs and skips the entry
                # instead of ending the walk.
                late_need = (full_required_for_admission, swa_required_for_admission)
                late_budget = full_allocatable_tokens
                if not late_d.admit_fits(
                    decode_req,
                    late_need,
                    claim_need,
                    full_allocatable_tokens,
                    swa_allocatable_tokens if swa_allocatable_tokens is not None else 0,
                ):
                    if prefix_match is not None and prefix_match.l1_prefix_len > 0:
                        self._release_matched_prefix_lock(decode_req.req)
                    claim_need = late_d.on_no_fit(decode_req, late_need, claim_need)
                    continue

            if not self._prealloc_reservation_fits(
                full_required_for_admission,
                swa_required_for_admission,
                full_allocatable_tokens=full_allocatable_tokens,
                swa_allocatable_tokens=swa_allocatable_tokens,
            ):
                if prefix_match is not None and prefix_match.l1_prefix_len > 0:
                    self.tree_cache.unlock(decode_req.req.lock)
                    decode_req.req.lock = None
                break

            if swa_allocatable_tokens is not None:
                reclaim_error = self._reclaim_swa_tail_capacity(
                    swa_len,
                    decode_req.req.rid,
                    full_len=required_alloc_tokens,
                )
                if reclaim_error is not None:
                    if prefix_match is not None and prefix_match.l1_prefix_len > 0:
                        self.tree_cache.unlock(decode_req.req.lock)
                        decode_req.req.lock = None
                    logger.error(reclaim_error)
                    prepare_abort(
                        decode_req.req,
                        reclaim_error,
                        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                    self.scheduler.output_streamer.stream_output(
                        [decode_req.req], decode_req.req.return_logprob
                    )
                    decode_req.kv_receiver.clear()
                    decode_req.kv_receiver = None
                    failed_reqs.append(decode_req)
                    indices_to_remove.add(i)
                    continue
            if self.scheduler.enable_decode_hicache:
                # Prefill is promised only the L3 span the prefetch took.
                self._start_hicache_prefetch(decode_req.req, prefix_match)
                if prefix_match is not None:
                    total_prefix_len = prefix_match.decode_prefix_len
            dst_kv_indices = self._pre_alloc(
                decode_req.req,
                prefix_indices,
                prefix_len,
                total_prefix_len,
            )
            if envs.SGLANG_DECODE_PREFIX_OBSERVATION.get():
                # Snapshot the successful allocation's existing match. A later
                # rematch or P aux must not replace these D-local observations.
                pm = prefix_match
                decode_req.req.sched_trace_decode_prefix_observation = {
                    "mode": "standard_pd",
                    "matched_device_tokens": pm.l1_prefix_len if pm else None,
                    "matched_host_tokens": pm.l2_host_hit_length if pm else None,
                    # This is the selected storage contribution, not proof
                    # that a disabled storage tier has no matching contents.
                    "selected_storage_tokens": (
                        pm.l3_storage_hit_length if pm else None
                    ),
                    "reused_device_tokens": prefix_len,
                    "transfer_skip_prefix_tokens": total_prefix_len,
                    "local_restore_requested_tokens": max(
                        0, total_prefix_len - prefix_len
                    ),
                    "host_load_back_queued_tokens": 0,
                    "local_restore_committed_tokens": 0,
                }
            if envs.SGLANG_DECODE_SELECTION_METRICS.get():
                decode_req.req.decode_allocation_unix_ns = time.time_ns()
                if self.tp_rank == 0:
                    logger.info(
                        "DECODE_SELECTION allocation rid=%s room=%s unix_ns=%d prompt=%d reserve=%d required=%s",
                        decode_req.req.rid,
                        decode_req.req.bootstrap_room,
                        decode_req.req.decode_allocation_unix_ns,
                        len(decode_req.req.origin_input_ids),
                        self.num_reserved_decode_tokens,
                        required_tokens_for_request,
                    )
            decode_req.prefix_match = prefix_match
            hisparse_req_budget -= 1
            # Recompute from actual pool state for the next queue entry.
            # This accounts for page rounding and newly locked evictable cache.
            if prefix_match is not None:
                reserved_restore_tokens += prefix_match.restore_token_count
            full_allocatable_tokens = self._allocatable_token_budgets(
                retractable_tokens=retractable_tokens,
                count_retracted=True,
                extra_reserved_reqs=num_device_preallocated + 1,
                hicache_reserved_tokens=reserved_restore_tokens,
                extra_reqs=_admitted_reqs(decode_req.req),
                cap_unsafe_rids=cap_unsafe_rids,
            )
            if swa_allocatable_tokens is not None:
                swa_allocatable_tokens = self._swa_tail_allocatable_token_budget(
                    retractable_tokens=retractable_tokens,
                    retractable_swa_tokens=retractable_swa_tokens,
                    count_retracted=True,
                    extra_reserved_reqs=num_device_preallocated + 1,
                    extra_reqs=[
                        *(dr.req for dr in preallocated_reqs if not dr.host_staged),
                        decode_req.req,
                    ],
                )
            decode_req.req.kv.cache_protected_len = total_prefix_len
            if late:
                late_d.on_alloc(
                    decode_req, late_need[0], late_budget, full_allocatable_tokens
                )

            page_size = self.token_to_kv_pool_allocator.page_size
            kv_transfer_page_size = page_size
            raw_kv_indices = self.req_to_token_pool.req_to_token[
                decode_req.req.kv.req_pool_idx
            ][total_prefix_len:origin_input_len]
            if self.scheduler.enable_hisparse:
                # Direct-to-host sends host/C4 rows; keep allocator.page_size
                # logical and use the compressed page size only for these indices.
                kv_transfer_page_size = getattr(
                    self.token_to_kv_pool_allocator,
                    "hisparse_page_size",
                    page_size,
                )
                kv_indices = dst_kv_indices[: origin_input_len - prefix_len]
            else:
                # Only send delta indices (beyond prefix) to prefill.
                kv_indices = (
                    self.token_to_kv_pool_allocator.translate_kv_indices_for_transfer(
                        raw_kv_indices
                    )
                )

            seq_len = origin_input_len

            def _mamba_payload():
                return [
                    self.req_to_token_pool.translate_mamba_indices(
                        self.req_to_token_pool.req_index_to_mamba_index_mapping[
                            decode_req.req.kv.req_pool_idx
                        ]
                    )
                    .cpu()
                    .numpy()
                ]

            def _swa_payload():
                window_size = self.scheduler.sliding_window_size
                window_start = max(total_prefix_len, seq_len - window_size)
                window_start = page_align_floor(window_start, page_size)
                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    decode_req.req.kv.req_pool_idx, window_start:seq_len
                ]
                window_kv_indices_swa = (
                    self.token_to_kv_pool_allocator.translate_swa_indices_for_transfer(
                        window_kv_indices_full
                    )
                )
                return kv_to_page_indices(window_kv_indices_swa, page_size)

            def _full_kv_pages_payload():
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    decode_req.req.kv.req_pool_idx, :seq_len
                ]
                # Indexer lives on device pool; always use device page_size
                device_page_size = self.token_to_kv_pool.page_size
                return kv_to_page_indices(kv_indices_full, device_page_size)

            def _minimax_payload():
                return minimax_index_transfer_pages(
                    self.req_to_token_pool.req_to_token[
                        decode_req.req.kv.req_pool_idx, :seq_len
                    ],
                    page_size=self.token_to_kv_pool.page_size,
                    prefix_len=total_prefix_len,
                )

            def _dsa_pages_payload():
                # Mirror of prefill _dsa_pages_payload: the cached prefix keeps
                # its own indexer rows, and rows still being restored by
                # HiCache are not written until the restore commits.
                return suffix_page_indices(
                    self.req_to_token_pool.req_to_token,
                    decode_req.req.kv.req_pool_idx,
                    total_prefix_len,
                    seq_len,
                    self.token_to_kv_pool.page_size,
                )

            def _dsa_tail_payload():
                return get_dsa_tail_state_indices(
                    self.token_to_kv_pool,
                    decode_req.req.kv.req_pool_idx,
                    seq_len,
                )

            def _qsa_pending_payload():
                # Match the prefill request-pool row positionally; the two
                # req_pool_idx values need not be equal.
                return get_qsa_pending_state_indices(decode_req.req)

            def _swa_ring_payload():
                # Mirror of prefill _swa_ring_payload using this side's req_pool_idx.
                # Same window positions and order -> positional match with prefill.
                ring_stride = self.token_to_kv_pool.unified_swa_ring_size
                window_size = self.token_to_kv_pool.unified_swa_window
                window_start = max(0, seq_len - window_size)
                positions = np.arange(window_start, seq_len, dtype=np.int64)
                state_slot = int(decode_req.req.kv.req_pool_idx)
                ring_rows = state_slot * ring_stride + (positions % ring_stride)
                return ring_rows.astype(np.int32)

            def _request_state_payload():
                return self.token_to_kv_pool.request_state_transfer_indices(
                    int(decode_req.req.kv.req_pool_idx), seq_len
                )

            def _request_row_payload():
                return np.array([decode_req.req.kv.req_pool_idx], dtype=np.int32)

            state_types = self.kv_manager.kv_args.state_types
            if StateType.DSV4_REQUEST_STATE in state_types:
                clear_request_state = getattr(
                    self.token_to_kv_pool, "clear_request_scoped_state", None
                )
                if clear_request_state is not None:
                    clear_request_state(int(decode_req.req.kv.req_pool_idx))
            payloads = {
                StateType.MAMBA: _mamba_payload,
                StateType.QSA_PENDING: _qsa_pending_payload,
                StateType.QSA_COMPRESSED: _full_kv_pages_payload,
                StateType.SWA: _swa_payload,
                StateType.DSA: _dsa_pages_payload,
                StateType.DSA_TAIL: _dsa_tail_payload,
                StateType.MINIMAX_INDEX_K: _minimax_payload,
                StateType.MINIMAX_DENSE_KV: _full_kv_pages_payload,
                StateType.SWA_RING: _swa_ring_payload,
                StateType.DSV4_REQUEST_STATE: _request_state_payload,
                StateType.REQUEST_ROW: _request_row_payload,
                StateType.BLOCK_SCALE: _full_kv_pages_payload,
                StateType.BLOCK_SCALE_SWA: _swa_payload,
            }
            if _is_npu and isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool):
                from decoupled_engine.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
                    dsv4_state_payloads,
                )

                payloads.update(
                    dsv4_state_payloads(
                        self.req_to_token_pool,
                        decode_req.req.kv.req_pool_idx,
                        seq_len,
                        self.token_to_kv_pool_allocator.page_size,
                        prefix_len=total_prefix_len,
                    )
                )
            state_indices: Optional[List] = [
                payloads[st]() if st in payloads else None for st in state_types
            ]

            if (
                content_only
                and self.tp_rank == 0
                and self.pp_rank == 0
                and logger.isEnabledFor(logging.DEBUG)
            ):
                state_counts = ",".join(
                    f"{state_type.value}:{0 if indices is None else len(indices)}"
                    for state_type, indices in zip(
                        state_types, state_indices, strict=True
                    )
                )
                logger.debug(
                    "PD decode radix plan: rid=%s bootstrap_room=%s "
                    "content_only=true origin_tokens=%d frozen_prompt_len=%s "
                    "fresh_state_window_start=%d decode_prefix_len=%d "
                    "main_transfer_pages=%d state_indices={%s}",
                    decode_req.req.rid,
                    decode_req.req.bootstrap_room,
                    origin_input_len,
                    decode_req.req.decode_radix_cache_frozen_prompt_len,
                    content_match_limit,
                    total_prefix_len,
                    len(kv_to_page_indices(kv_indices, kv_transfer_page_size)),
                    state_counts,
                )
            device_page_indices = None
            if (
                self.scheduler.enable_hisparse
                and (
                    isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool)
                    or self.draft_token_to_kv_pool is not None
                )
                and not _is_fake_transfer(decode_req.req)
            ):
                # alloc_logical_only() already allocated the shared logical pages
                # used by the indexer and draft KV. These device buffers do not
                # use target sparse physical slots; carry logical page IDs beside
                # the independently allocated target host page IDs.
                full_kv_indices = self.req_to_token_pool.req_to_token[
                    decode_req.req.kv.req_pool_idx,
                    prefix_len:origin_input_len,
                ]
                device_page_indices = kv_to_page_indices(
                    full_kv_indices,
                    page_size,
                ).astype(np.int32)
                if self.transfer_backend != TransferBackend.MOONCAKE:
                    raise NotImplementedError(
                        "HiSparse device-only PD transfer currently requires the "
                        "Mooncake backend"
                    )
            metadata_kwargs = {"decode_prefix_len": total_prefix_len}
            if device_page_indices is not None:
                metadata_kwargs["device_kv_indices"] = device_page_indices
            self._send_kv_metadata(
                decode_req,
                kv_indices,
                kv_transfer_page_size,
                state_indices,
                **metadata_kwargs,
            )
            num_device_preallocated += 1
            preallocated_reqs.append(decode_req)
            indices_to_remove.add(i)
            if self.scheduler.enable_lora:
                running_loras.add(decode_req.req.lora_id)

        if failed_reqs:
            failed_ids = {id(r) for r in failed_reqs}
            self.pending_reqs = [
                r for r in self.pending_reqs if id(r) not in failed_ids
            ]
            for decode_req in failed_reqs:
                if decode_req.req.kv.holds_mamba and not decode_req.req.kv.holds_kv:
                    release_kv_cache(decode_req.req, self.tree_cache, checkpoint=False)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return preallocated_reqs, failed_reqs

    def _send_kv_metadata(
        self, decode_req, kv_indices, page_size, state_indices=None, **metadata_kwargs
    ) -> None:
        decode_req.metadata_buffer_index = (
            self.req_to_metadata_buffer_idx_allocator.alloc()
        )
        assert decode_req.metadata_buffer_index is not None
        page_indices = self._transfer_page_indices(decode_req, kv_indices, page_size)
        if (
            metadata_kwargs.get("destination") != KVTransferDestination.HOST
            and self.transfer_queue.enable_staging
            and hasattr(decode_req.kv_receiver, "require_staging")
            and decode_req.kv_receiver.require_staging
        ):
            # Register before send_metadata, which triggers the STAGING_REQ
            # prefetch (dropped for an unregistered room); tiny race, correct order.
            self.transfer_queue.staging_handler.register_decode_req(
                decode_req.req.bootstrap_room, decode_req
            )
        decode_req.kv_receiver.send_metadata(
            page_indices,
            decode_req.metadata_buffer_index,
            state_indices,
            **metadata_kwargs,
        )
        if (
            decode_req.is_rebootstrap
            and get_disagg().disaggregation_decode_allocation_policy == "early"
        ):
            self._submit_prefill_recompute(decode_req)
        self._num_published_destinations += 1
        decode_req.req.time_stats.set_decode_transfer_queue_entry_time()

    def _pre_alloc_host(self, decode_req: DecodeRequest) -> bool:
        seq_len = len(decode_req.req.origin_input_ids)
        page_size = self.host_pool.page_size
        num_tokens = ceil_align(seq_len, page_size)
        required_tokens = num_tokens + self.host_reserved_tokens
        if (
            self.host_pool.available_size() < required_tokens
            and not self.tree_cache.disable
        ):
            self.tree_cache.evict_host(required_tokens)
        if self.host_pool.available_size() < required_tokens:
            return False
        swa_num_tokens = 0
        if self.host_swa_pool is not None:
            window_start = page_align_floor(
                max(0, seq_len - self.scheduler.sliding_window_size), page_size
            )
            swa_num_tokens = num_tokens - window_start
            required_swa_tokens = swa_num_tokens + self.host_swa_reserved_tokens
            if self.host_swa_pool.available_size() < required_swa_tokens:
                entry = self.tree_cache.host_pool_group.get_entry(PoolName.SWA)
                if entry.host_evict_fn is not None:
                    entry.host_evict_fn(required_swa_tokens)
            if self.host_swa_pool.available_size() < required_swa_tokens:
                return False
        host_indices = self.host_pool.alloc(num_tokens)
        if host_indices is None:
            return False
        indices_by_pool = {PoolName.KV: host_indices}
        transfers = []
        state_indices = None
        if self.host_swa_pool is not None:
            swa_host_indices = self.host_swa_pool.alloc(swa_num_tokens)
            if swa_host_indices is None:
                self.host_pool.free(host_indices)
                return False
            indices_by_pool[PoolName.SWA] = swa_host_indices
            transfers.append(
                PoolTransfer(name=PoolName.SWA, host_indices=swa_host_indices)
            )
            state_indices = [kv_to_page_indices(swa_host_indices, page_size)]
        assert decode_req.req.kv.retraction_backup is None
        decode_req.req.kv.retraction_backup = RetractionBackup(
            host_indices=host_indices,
            pool_transfers=transfers
            + [
                PoolTransfer(
                    name=spec.pool_name,
                    host_indices=indices_by_pool[spec.indices_from_pool],
                    indices_from_pool=spec.indices_from_pool,
                    hit_policy=spec.hit_policy,
                )
                for spec in self.tree_cache.sidecar_pool_specs
                if num_tokens
            ]
            or None,
        )
        if get_disagg().disaggregation_decode_enable_radix_cache:
            _bind_root_prefix(decode_req.req, self.tree_cache)
        self._send_kv_metadata(
            decode_req,
            host_indices,
            self.token_to_kv_pool_allocator.page_size,
            state_indices,
            decode_prefix_len=0,
            destination=KVTransferDestination.HOST,
        )
        decode_req.host_staged = True
        if self.scheduler.metrics_reporter.current_scheduler_metrics_enabled:
            self.scheduler.metrics_collector.increment_decode_host_receive_reqs()
        return True

    def allocate_host_staged(self, decode_req: DecodeRequest) -> bool:
        if self.req_to_token_pool.available_size() <= 0:
            return False
        req = decode_req.req
        retractable_tokens = sum(
            len(r.origin_input_ids) + len(r.output_ids)
            for r in self.scheduler.running_batch.reqs
        )
        required_tokens = self._required_admission_tokens(
            req,
            self._required_alloc_tokens(
                fill_len=len(req.origin_input_ids), prefix_len=0
            ),
            0,
            retractable_tokens,
        )
        swa_required_tokens = 0
        swa_allocatable_tokens = None
        if self._uses_swa_reservation():
            _, swa_required_tokens = self._prealloc_required_tokens(req)
            full_allocatable_tokens, swa_allocatable_tokens = (
                self._swa_aware_allocatable_token_budgets(
                    retractable_tokens=retractable_tokens,
                    retractable_swa_tokens=sum(
                        self._swa_retractable_len(r)
                        for r in self.scheduler.running_batch.reqs
                    ),
                    count_retracted=True,
                )
            )
        else:
            full_allocatable_tokens = self._allocatable_token_budgets(
                retractable_tokens=retractable_tokens, count_retracted=True
            )
        if not self._prealloc_reservation_fits(
            required_tokens,
            swa_required_tokens,
            full_allocatable_tokens=full_allocatable_tokens,
            swa_allocatable_tokens=swa_allocatable_tokens,
        ):
            return False
        self._pre_alloc(req)
        decode_req.host_staged = False
        return True

    @property
    def has_published_destinations(self) -> bool:
        """Whether any destination address is visible to prefill but not yet
        protected by the transfer queue."""
        return self._num_published_destinations > 0

    def note_destinations_queued(self, count: int) -> None:
        """Hand `count` published destinations over to the transfer queue."""
        self._num_published_destinations = max(
            0, self._num_published_destinations - count
        )

    @property
    def num_tokens_pre_allocated(self):
        return sum(
            decode_req.req.extend_end
            for decode_req in self.transfer_queue.queue
            if not decode_req.host_staged
        )

    def pre_allocated_token_counts(self) -> Tuple[int, int]:
        """(full, SWA) KV slots held by transfer-queue requests: KV reserved
        for an incoming transfer, not yet in the running batch. Pages are
        charged the way the pool stats charge a request's private KV."""
        page_size = self.token_to_kv_pool_allocator.page_size
        full_tokens = swa_tokens = 0
        for decode_req in self.transfer_queue.queue:
            if decode_req.host_staged:
                continue
            full_tokens += kv_private_tokens(decode_req.req.kv, page_size)
            swa_tokens += kv_private_swa_tokens(decode_req.req.kv, page_size)
        return full_tokens, swa_tokens

    def _need_space_for_single_req(
        self,
        retractable_tokens: Optional[int] = None,
        cap_unsafe_rids: frozenset[str] = frozenset(),
    ) -> int:
        if (
            cap_unsafe_rids
            and retractable_tokens is not None
            and len(self.scheduler.running_batch.reqs) > 0
        ):
            # SGLANG_DECODE_ADMIT_CAP_SAFE_RETRACT: retractable_tokens holds
            # only the cap-safe requests. A request that is not retractable is
            # not in it, and its own KV is allocated already.
            return max(
                min(x.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
                + len(x.origin_input_ids)
                - retractable_tokens
                - (
                    len(x.origin_input_ids) + len(x.output_ids)
                    if x.rid in cap_unsafe_rids
                    else 0
                )
                for x in self.scheduler.running_batch.reqs
            )
        need_space_for_single_req = (
            max(
                [
                    min(x.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
                    + len(x.origin_input_ids)
                    - retractable_tokens
                    for x in self.scheduler.running_batch.reqs
                ]
            )
            if retractable_tokens is not None
            and len(self.scheduler.running_batch.reqs) > 0
            else 0
        )
        return need_space_for_single_req

    def _active_req_count(self, extra_reserved_reqs: int = 0) -> int:
        return (
            len(self.scheduler.running_batch.reqs)
            + sum(not r.host_staged for r in self.transfer_queue.queue)
            + len(self.scheduler.waiting_queue)
            + extra_reserved_reqs
        )

    def _active_reserved_tokens(
        self,
        n_active: Optional[int] = None,
        extra_reserved_reqs: int = 0,
        extra_reqs: Sequence[Req] = (),
    ) -> int:
        if n_active is None:
            growth_ratio = self._growth_ratio_applies()
            if growth_ratio > 0:
                return self._growth_reserved_tokens(
                    growth_ratio, extra_reserved_reqs, extra_reqs
                )
            n_active = self._active_req_count(extra_reserved_reqs)
        return self.num_reserved_decode_tokens * n_active

    def _admit_growth_ratio(self) -> float:
        """SGLANG_DECODE_ADMIT_GROWTH_RATIO, read once per queue; 0.0 = off."""
        ratio = self.__dict__.get("_admit_growth_ratio_value")
        if ratio is None:
            ratio = max(0.0, float(envs.SGLANG_DECODE_ADMIT_GROWTH_RATIO.get() or 0))
            self._admit_growth_ratio_value = ratio
            if ratio > 0 and getattr(self, "tp_rank", 0) == 0:
                logger.info(
                    "PD decode admission: decode growth reserve on "
                    "(SGLANG_DECODE_ADMIT_GROWTH_RATIO=%g): each active request "
                    "and the candidate keep max(%d, ceil(%g x min(max_new_tokens "
                    "- output tokens, %d))) tokens; SWA-tail prealloc unchanged",
                    ratio,
                    self.num_reserved_decode_tokens,
                    ratio,
                    CLIP_MAX_NEW_TOKEN,
                )
        return ratio

    def _admit_cap_safe_retract(self) -> bool:
        """SGLANG_DECODE_ADMIT_CAP_SAFE_RETRACT, read once per queue."""
        on = self.__dict__.get("_admit_cap_safe_retract_value")
        if on is None:
            on = bool(envs.SGLANG_DECODE_ADMIT_CAP_SAFE_RETRACT.get())
            self._admit_cap_safe_retract_value = on
            if on and getattr(self, "tp_rank", 0) == 0:
                logger.info(
                    "PD decode admission: only running requests whose "
                    "retraction backup fits under SGLANG_RETRACT_BACKUP_MAX_GB "
                    "count as retractable (SGLANG_DECODE_ADMIT_CAP_SAFE_RETRACT"
                    "=1; cpu_tensor backup only; cap_gb=%s; SWA-tail prealloc "
                    "unchanged)",
                    envs.SGLANG_RETRACT_BACKUP_MAX_GB.get(),
                )
        return on

    def _cap_unsafe_running_rids(self) -> frozenset[str]:
        """SGLANG_DECODE_ADMIT_CAP_SAFE_RETRACT: the running requests that a
        retraction now would abort (cpu_tensor backup over
        SGLANG_RETRACT_BACKUP_MAX_GB). Empty when the env is off, without the
        cap, or with another backup backend."""
        if not self._admit_cap_safe_retract():
            return frozenset()
        if get_disagg().disaggregation_decode_retraction_backup != "cpu_tensor":
            return frozenset()
        return retract_backup_unsafe_rids(
            self.scheduler.running_batch.reqs, self.token_to_kv_pool_allocator
        )

    def _growth_ratio_applies(self) -> float:
        """The growth ratio of this admission: 0.0 when the env is off, and on
        the SWA reservation path (SWA-tail prealloc, or an allocator whose
        prealloc_fits assumes reclaim), which keeps num_reserved_decode_tokens."""
        ratio = self._admit_growth_ratio()
        if ratio > 0 and self._uses_swa_reservation():
            return 0.0
        return ratio

    def _growth_reserved_tokens(
        self, ratio: float, extra_reserved_reqs: int, extra_reqs: Sequence[Req]
    ) -> int:
        """SGLANG_DECODE_ADMIT_GROWTH_RATIO: the decode growth kept for every
        active request (running, transfer queue, waiting queue) and for the
        requests this pass admitted (``extra_reqs``). A request counted in
        ``extra_reserved_reqs`` but not named keeps num_reserved_decode_tokens."""
        floor_tokens = self.num_reserved_decode_tokens
        total = sum(
            _decode_growth_reserve(r, ratio, floor_tokens)
            for r in self.scheduler.running_batch.reqs
        )
        total += sum(
            _decode_growth_reserve(dr.req, ratio, floor_tokens)
            for dr in self.transfer_queue.queue
            if not dr.host_staged
        )
        total += sum(
            _decode_growth_reserve(r, ratio, floor_tokens)
            for r in self.scheduler.waiting_queue
        )
        total += sum(_decode_growth_reserve(r, ratio, floor_tokens) for r in extra_reqs)
        return total + floor_tokens * max(0, extra_reserved_reqs - len(extra_reqs))

    def _swa_aware_allocatable_token_budgets(
        self,
        retractable_tokens: Optional[int] = None,
        retractable_swa_tokens: Optional[int] = None,
        count_retracted: bool = True,
    ) -> Tuple[int, int]:
        n_active = self._active_req_count()
        reserved_tokens = self._active_reserved_tokens(n_active)

        full_allocatable_tokens = self._allocatable_token_budgets(
            retractable_tokens=retractable_tokens,
            count_retracted=count_retracted,
            reserved_tokens=reserved_tokens,
        )

        return full_allocatable_tokens, self._swa_tail_allocatable_token_budget(
            retractable_tokens=retractable_tokens,
            retractable_swa_tokens=retractable_swa_tokens,
            count_retracted=count_retracted,
            n_active=n_active,
            reserved_tokens=reserved_tokens,
        )

    def _allocatable_token_budgets(
        self,
        retractable_tokens: Optional[int] = None,
        count_retracted: bool = True,
        extra_reserved_reqs: int = 0,
        reserved_tokens: Optional[int] = None,
        hicache_reserved_tokens: int = 0,
        extra_reqs: Sequence[Req] = (),
        cap_unsafe_rids: frozenset[str] = frozenset(),
    ) -> int:
        need_space_for_single_req = self._need_space_for_single_req(
            retractable_tokens, cap_unsafe_rids
        )
        if reserved_tokens is None:
            reserved_tokens = self._active_reserved_tokens(
                extra_reserved_reqs=extra_reserved_reqs, extra_reqs=extra_reqs
            )

        if self.scheduler.enable_hisparse:
            logical_allocator = self.token_to_kv_pool_allocator.logical_attn_allocator
            if self._uses_swa_tail_prealloc() and hasattr(
                logical_allocator, "full_available_size"
            ):
                available_size = logical_allocator.full_available_size()
            else:
                # HiSparse pre-alloc only allocates logical indices, so the
                # logical pool is the binding constraint for admission control.
                available_size = logical_allocator.available_size()
        elif self._uses_swa_reservation():
            available_size = self.token_to_kv_pool_allocator.full_available_size()
            if get_disagg().disaggregation_decode_enable_radix_cache:
                available_size += self._radix_full_evictable()
        else:
            available_size = self.token_to_kv_pool_allocator.available_size()
            # Include evictable decode-radix cache entries in the budget -- they
            # can be freed on demand before allocation.
            if get_disagg().disaggregation_decode_enable_radix_cache:
                available_size += self._radix_full_evictable()
        allocatable_tokens = available_size - max(
            reserved_tokens, need_space_for_single_req
        )

        # Note: if the last prebuilt extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
        #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
        if (
            self.scheduler.last_batch
            and self.scheduler.last_batch.forward_mode.is_prebuilt()
        ):
            growth_ratio = self._growth_ratio_applies()
            if growth_ratio > 0:
                allocatable_tokens -= sum(
                    _decode_growth_reserve(
                        r, growth_ratio, self.num_reserved_decode_tokens
                    )
                    for r in self.scheduler.last_batch.reqs
                )
            else:
                allocatable_tokens -= self.num_reserved_decode_tokens * len(
                    self.scheduler.last_batch.reqs
                )

        if count_retracted:
            for req in self.retracted_queue:
                full_required, _ = self._prealloc_required_tokens(req)
                allocatable_tokens -= full_required

        allocatable_tokens -= hicache_reserved_tokens
        return allocatable_tokens

    def _swa_tail_allocatable_token_budget(
        self,
        retractable_tokens: Optional[int] = None,
        retractable_swa_tokens: Optional[int] = None,
        count_retracted: bool = True,
        n_active: Optional[int] = None,
        reserved_tokens: Optional[int] = None,
        extra_reserved_reqs: int = 0,
        extra_reqs: Sequence[Req] = (),
    ) -> int:
        """SWA tokens admission may still hand out. ``extra_reserved_reqs``
        counts requests this pass already preallocated that are in no queue
        yet; ``extra_reqs`` are those requests (the window_host reserve is
        per request)."""
        need_swa_space_for_single_req = self._need_space_for_single_req(
            retractable_tokens
        )
        if (
            retractable_swa_tokens is not None
            and len(self.scheduler.running_batch.reqs) > 0
        ):
            need_swa_space_for_single_req = max(
                self._swa_tail_len(len(x.origin_input_ids))
                + min(x.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
                - retractable_swa_tokens
                for x in self.scheduler.running_batch.reqs
            )

        if n_active is None:
            n_active = self._active_req_count(extra_reserved_reqs)
        if reserved_tokens is None:
            reserved_tokens = self._active_reserved_tokens(n_active)

        # SWA growth is bounded by the sliding window: once a req's SWA
        # footprint reaches `sliding_window_size`, further decode tokens
        # evict old ones and net growth is zero. The linear reservation
        # `num_reserved_decode_tokens * n_active` (correct for the full
        # pool) over-reserves SWA in steady state. Cap by the actual
        # remaining headroom up to per-req window cap.
        window_size = self.scheduler.sliding_window_size or 0
        allocator = self.token_to_kv_pool_allocator
        _, (swa_total, swa_available) = allocator.swa_capacity_and_available(
            full_capacity=allocator.size_full, swa_capacity=allocator.size_swa
        )
        # Per-request SWA ring: cached prefixes still report swa_evictable, but
        # evicting them frees no ring space.
        if is_swa_req_ring(self.token_to_kv_pool_allocator):
            swa_evictable = 0
        elif getattr(self.tree_cache, "dsv4_uniform_swa_residency", "full") == "window":
            # Unbacked tree SWA cannot be evicted until its host copy exists.
            swa_evictable = self.tree_cache.swa_admission_evictable_size()
        else:
            swa_evictable = self.tree_cache.swa_evictable_size()
        swa_used = swa_total - swa_available - swa_evictable
        window_host = (
            getattr(self.tree_cache, "dsv4_uniform_swa_residency", "full") == "window"
            and self._uses_window_host_radix()
        )
        counted_reqs: Dict[int, Req] = {}
        if window_host:
            # window_host: a decoding request
            # keeps its SWA from the tail start up to its spill frontier and
            # frees spilled pages only at the ack, so it peaks at the
            # decode allowance, not at W. Reserve each request
            # the rest of its allowance over the private SWA it holds now,
            # capped by what its remaining decode can still allocate, as
            # local admission does (PrefillAdder.rem_swa_token_offset).
            for req in self._window_host_active_reqs(extra_reqs):
                counted_reqs.setdefault(id(req), req)
            swa_reserved_tokens = self._window_host_swa_growth_reserve(
                counted_reqs.values()
            )
        else:
            swa_growth_potential = max(0, n_active * window_size - swa_used)
            swa_reserved_tokens = min(reserved_tokens, swa_growth_potential)
        swa_allocatable_tokens = (
            swa_available
            + swa_evictable
            - max(swa_reserved_tokens, need_swa_space_for_single_req)
        )

        # Note: if the last prebuilt extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
        #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
        if (
            self.scheduler.last_batch
            and self.scheduler.last_batch.forward_mode.is_prebuilt()
        ):
            prebuilt_reserved_tokens = self.num_reserved_decode_tokens * len(
                self.scheduler.last_batch.reqs
            )
            if window_host:
                swa_allocatable_tokens -= self._window_host_swa_growth_reserve(
                    req
                    for req in self.scheduler.last_batch.reqs
                    if id(req) not in counted_reqs
                )
            else:
                prebuilt_n = len(self.scheduler.last_batch.reqs)
                prebuilt_swa_growth = max(0, prebuilt_n * window_size - swa_used)
                swa_allocatable_tokens -= min(
                    prebuilt_reserved_tokens, prebuilt_swa_growth
                )

        if count_retracted:
            for req in self.retracted_queue:
                _, swa_required = self._prealloc_required_tokens(req)
                swa_allocatable_tokens -= swa_required

        return swa_allocatable_tokens

    def _window_host_active_reqs(self, extra_reqs: Sequence[Req] = ()) -> List[Req]:
        """The requests holding decode SWA that admission must leave growth
        room for: running, in transfer, transferred but not yet prebuilt, and
        those preallocated by the current pass (``extra_reqs``)."""
        return [
            *self.scheduler.running_batch.reqs,
            *(
                decode_req.req
                for decode_req in self.transfer_queue.queue
                if not decode_req.host_staged
            ),
            *self.scheduler.waiting_queue,
            *extra_reqs,
        ]

    def _window_host_swa_growth_reserve(self, reqs: Iterable[Req]) -> int:
        """Sum over ``reqs`` of max(0, min(A - private SWA, reachable -
        ceil_page(kv_allocated_len))), A = the decode allowance
        (schedule_policy.swa_window_running_offset, shared with local
        admission). On a window_host decode worker the private SWA starts at
        T = kv.swa_evicted_seqlen (>= cache_protected_len = dpl)."""
        page_size = self.token_to_kv_pool_allocator.page_size
        reserve = get_alloc_reserve_per_decode()
        allowance = swa_window_decode_allowance(
            page_size=page_size,
            sliding_window_size=self.scheduler.sliding_window_size or 0,
            reserve=reserve,
        )
        return sum(
            swa_window_running_offset(
                req, page_size=page_size, allowance=allowance, reserve=reserve
            )
            for req in reqs
        )

    def _required_alloc_tokens(self, *, fill_len: int, prefix_len: int) -> int:
        page_size = self.token_to_kv_pool_allocator.page_size
        if page_size == 1:
            return fill_len - prefix_len

        num_new_pages = get_num_new_pages(
            seq_lens=torch.tensor([fill_len], dtype=torch.int64),
            prefix_lens=torch.tensor([prefix_len], dtype=torch.int64),
            page_size=page_size,
        )
        return num_new_pages * page_size

    def _alloc_for_decode_prealloc(
        self, allocator: BaseTokenToKVPoolAllocator, **kwargs
    ) -> torch.Tensor:
        return alloc_for_decode_prealloc(allocator, **kwargs)

    def _transfer_page_indices(
        self, decode_req: DecodeRequest, kv_indices: torch.Tensor, page_size: int
    ) -> np.ndarray:
        # int32 for ZMQ serialization -- from_zmq reads np.int32.
        return kv_to_page_indices(kv_indices, page_size).astype(np.int32)

    def _pre_alloc(
        self,
        req: Req,
        prefix_indices: Optional[torch.Tensor] = None,
        prefix_len: Optional[int] = None,
        total_prefix_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Pre-allocate the memory for req_to_token and token_kv_pool.

        ``prefix_len`` is the L1 device-resident prefix length (already
        backed by ``prefix_indices``). ``total_prefix_len`` is the full
        prefix committed to prefill as ``decode_prefix_len`` (L1 + L2 + L3);
        the ``[prefix_len, total_prefix_len)`` gap is filled later by HiCache
        loadback.
        """
        if prefix_len is None:
            prefix_len = 0
        if total_prefix_len is None:
            total_prefix_len = prefix_len

        req_pool_indices = self.req_to_token_pool.alloc([req])

        assert req_pool_indices is not None, (
            "req_pool_indices is full! There is a bug in memory estimation."
        )

        fill_len = self._pre_alloc_fill_len(req)
        req.kv.kv_committed_len = fill_len
        self._freeze_dspark_prompt_cache_boundary(req)
        self._mark_window_host_prebuilt(req)

        if prefix_len > 0:
            self.req_to_token_pool.write(
                (req.kv.req_pool_idx, slice(0, prefix_len)), prefix_indices
            )

        # TODO(retraction): when retraction is implemented with radix cache
        # awareness, a retracted request should re-match the tree here
        # instead of re-allocating from scratch. See resume_retracted_reqs.
        delta_len = fill_len - total_prefix_len
        required_alloc_tokens = self._required_alloc_tokens(
            fill_len=fill_len, prefix_len=prefix_len
        )

        # Evict cached entries if the pool doesn't have enough free pages.
        if (
            get_disagg().disaggregation_decode_enable_radix_cache
            and self._radix_full_available() < required_alloc_tokens
        ):
            num_to_evict = required_alloc_tokens - self._radix_full_available()
            result = self.tree_cache.evict_for_alloc(
                EvictParams(num_tokens=num_to_evict)
            )
            if self._radix_full_available() < required_alloc_tokens:
                logger.warning(
                    f"Eviction insufficient: needed {required_alloc_tokens} tokens, "
                    f"available {self._radix_full_available()} "
                    f"after evicting {result.num_tokens_evicted}/{num_to_evict} tokens. "
                    f"evictable_size={self._radix_full_evictable()}, "
                    f"protected_size={self._radix_full_protected()}, "
                    f"fill_len={fill_len}, prefix_len={prefix_len}, "
                    f"total_prefix_len={total_prefix_len}, delta_len={delta_len}, "
                    f"page_size={self.token_to_kv_pool_allocator.page_size}, "
                    f"req={req.rid}"
                )

        allocator = self.token_to_kv_pool_allocator
        uses_swa_tail = self._uses_swa_tail_prealloc()
        swa_tail_len = self._prealloc_swa_tail_len(req, fill_len)
        swa_pages_charged = (
            uses_swa_tail
            and not is_swa_req_ring(allocator)
            and not self.scheduler.enable_hisparse
        )
        required_swa_tokens = (
            ceil_align(swa_tail_len, allocator.page_size) if swa_pages_charged else 0
        )
        if get_disagg().disaggregation_decode_enable_radix_cache and swa_pages_charged:
            # Admission includes evictable pages; allocation needs free pages.
            # Separate-pool resumes skip the caller's unified-only reclaim.
            reclaim_error = self._reclaim_swa_tail_capacity(
                swa_tail_len, req.rid, full_len=required_alloc_tokens
            )
            if reclaim_error is not None:
                logger.warning("%s", reclaim_error)

        if self.scheduler.enable_hisparse:
            # HiSparse is incompatible with decode-side L1 radix cache. Keep
            # this path on the upstream full-allocation semantics.
            assert prefix_len == 0

            # Direct-to-host path: only allocate logical indices (no hisparse
            # device indices) and allocate host indices for RDMA destination.
            coordinator = self.scheduler.hisparse_coordinator
            kv_loc = alloc_for_decode_prealloc_hisparse(
                allocator,
                req=req,
                fill_len=fill_len,
                uses_swa_tail=uses_swa_tail,
                swa_tail_len=swa_tail_len,
            )
            # Allocate host indices for the RDMA transfer target.
            host_indices = coordinator.mem_pool_host.alloc_paged_token_slots(
                coordinator.req_to_host_pool,
                coordinator.req_to_host_pool_allocated_len,
                req.kv.req_pool_idx,
                0,
                coordinator.host_token_len(fill_len),
            )
        else:
            kv_loc = self._alloc_for_decode_prealloc(
                allocator,
                req=req,
                fill_len=fill_len,
                delta_len=delta_len,
                prefix_len=prefix_len,
                total_prefix_len=total_prefix_len,
                prefix_indices=prefix_indices,
                uses_swa_tail=uses_swa_tail,
                swa_tail_len=swa_tail_len,
                req_to_token_pool=self.req_to_token_pool,
            )
        assert kv_loc is not None, (
            f"KV cache is full! Bug in memory estimation. "
            f"available={self._radix_full_available()}, "
            f"evictable={self._radix_full_evictable()}, "
            f"protected={self._radix_full_protected()}, "
            f"required_alloc={required_alloc_tokens}, delta={delta_len}, "
            f"fill={fill_len}, prefix={prefix_len}, total_prefix={total_prefix_len}, "
            f"swa_available={allocator.swa_available_size() if swa_pages_charged else -1}, "
            f"swa_evictable={self.tree_cache.swa_evictable_size() if swa_pages_charged else -1}, "
            f"required_swa={required_swa_tokens}, swa_tail_len={swa_tail_len}, "
            f"page_size={allocator.page_size}, "
            f"req={req.rid}"
        )

        self.req_to_token_pool.write(
            (
                req.kv.req_pool_idx,
                slice(total_prefix_len, total_prefix_len + len(kv_loc)),
            ),
            kv_loc,
        )

        # Truncate fill_len to kv_committed_len so checkpoint only
        # inserts committed KV into the radix tree. The last output token
        # hasn't had KV committed yet (output_ids is 1 ahead).
        req.full_untruncated_fill_ids = req.origin_input_ids + req.output_ids
        # Decode-disagg allocates before batch assembly, so it binds the prefix
        # here instead of in init_next_round_input.
        req.prefix_len = prefix_len
        req.extend_end = req.kv.kv_committed_len
        self.tree_cache.maybe_hand_to_session(req)

        # Return the transfer destination indices:
        if self.scheduler.enable_hisparse:
            return host_indices
        return kv_loc


def alloc_for_decode_prealloc_hisparse(
    allocator: BaseTokenToKVPoolAllocator,
    *,
    req: Req,
    fill_len: int,
    uses_swa_tail: bool,
    swa_tail_len: int,
) -> torch.Tensor:
    req.kv.kv_allocated_len = fill_len
    device = allocator.device
    prefix_lens = async_h2d([0], dtype=torch.int64, device=device)
    prefix_lens_cpu = torch.tensor([0], dtype=torch.int64)
    seq_lens = async_h2d([fill_len], dtype=torch.int64, device=device)
    seq_lens_cpu = torch.tensor([fill_len], dtype=torch.int64)
    last_loc = async_h2d([-1], dtype=torch.int64, device=device)
    if uses_swa_tail:
        kv_loc = allocator.alloc_extend_swa_tail(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=fill_len,
            swa_tail_len=swa_tail_len,
        )
        swa_evicted_seqlen = fill_len - swa_tail_len
        assert swa_evicted_seqlen >= 0 and swa_evicted_seqlen % allocator.page_size == 0
        req.kv.set_evicted_seqlen(ComponentType.SWA, swa_evicted_seqlen)
    else:
        kv_loc = allocator.alloc_logical_only(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=fill_len,
        )
    return kv_loc


def alloc_for_decode_prealloc(
    allocator: BaseTokenToKVPoolAllocator,
    *,
    req: Req,
    fill_len: int,
    delta_len: int,
    prefix_len: int,
    total_prefix_len: int,
    prefix_indices: Optional[torch.Tensor],
    uses_swa_tail: bool,
    swa_tail_len: int,
    req_to_token_pool: Optional[ReqToTokenPool] = None,
) -> torch.Tensor:
    req.kv.kv_allocated_len = fill_len
    from decoupled_engine.srt.mem_cache.allocator.cpu_paged import cpu_paged_allocator

    q4_allocator = cpu_paged_allocator(allocator)
    if q4_allocator is not None:
        return _alloc_for_decode_prealloc_cpu(
            q4_allocator,
            fill_len=fill_len,
            delta_len=delta_len,
            prefix_len=prefix_len,
            total_prefix_len=total_prefix_len,
            prefix_indices=prefix_indices,
            uses_swa_tail=uses_swa_tail,
        )
    if allocator.page_size == 1:
        kv_loc = allocator.alloc(delta_len)
    else:
        device = allocator.device
        last_loc = (
            prefix_indices[-1:].to(dtype=torch.int64, device=device)
            if prefix_len > 0
            else async_h2d([-1], dtype=torch.int64, device=device)
        )
        extra_kwargs = {}
        dsv4_unwrap_prealloc = None
        if hasattr(allocator, "c128_attn_allocator"):
            assert req_to_token_pool is not None
            from decoupled_engine.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
                dsv4_prealloc_kwargs,
                dsv4_unwrap_prealloc,
            )

            extra_kwargs = dsv4_prealloc_kwargs(
                allocator,
                req,
                fill_len,
                req_to_token_pool,
                device=device,
            )
        if uses_swa_tail:
            # Full-attention layers reuse prefix KV; SWA layers allocate only
            # the live window tail. L2/L3 hits are loaded later, so the fresh
            # allocation starts after the full promised prefix, not only L1.
            kv_loc = allocator.alloc_extend_swa_tail(
                prefix_lens=async_h2d(
                    [total_prefix_len], dtype=torch.int64, device=device
                ),
                prefix_lens_cpu=torch.tensor([total_prefix_len], dtype=torch.int64),
                seq_lens=async_h2d([fill_len], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                last_loc=last_loc,
                extend_num_tokens=delta_len,
                swa_tail_len=swa_tail_len,
                **extra_kwargs,
            )
            swa_evicted_seqlen = fill_len - swa_tail_len
            assert (
                swa_evicted_seqlen >= 0
                and swa_evicted_seqlen % allocator.page_size == 0
            )
            req.kv.set_evicted_seqlen(ComponentType.SWA, swa_evicted_seqlen)
        else:
            kv_loc = allocator.alloc_extend(
                prefix_lens=async_h2d(
                    [total_prefix_len], dtype=torch.int64, device=device
                ),
                prefix_lens_cpu=torch.tensor([total_prefix_len], dtype=torch.int64),
                seq_lens=async_h2d([fill_len], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                last_loc=last_loc,
                extend_num_tokens=delta_len,
                **extra_kwargs,
            )
        if dsv4_unwrap_prealloc is not None:
            kv_loc = dsv4_unwrap_prealloc(
                kv_loc, req_to_token_pool, req, total_prefix_len, fill_len
            )
    return kv_loc


def _alloc_for_decode_prealloc_cpu(
    allocator: BaseTokenToKVPoolAllocator,
    *,
    fill_len: int,
    delta_len: int,
    prefix_len: int,
    total_prefix_len: int,
    prefix_indices: Optional[torch.Tensor],
    uses_swa_tail: bool,
) -> Optional[torch.Tensor]:
    """M4 Q4 prealloc (mem_cache/cpu_page_book.py): the pages of
    ``[total_prefix_len, fill_len)`` from the CPU free lists, no device
    last_loc and no alloc_extend kernel. The KV of these pages arrives by RDMA
    (the symmetric P/D reader's READ, a PD decode receive), which no CUDA stream
    orders after the forward in flight, so they are clean pages only (released
    at an epoch whose result is processed): ``alloc_import``. None when the
    clean pages do not suffice; the caller retries after the next promotions.
    Returns the device slots, tagged with their CPU page ids; the caller writes
    the request's CPU row (ImportQueue._prealloc)."""
    from decoupled_engine.srt.mem_cache.allocator.cpu_paged import load_quarantine_active
    from decoupled_engine.srt.mem_cache.cpu_page_book import pages_of

    if uses_swa_tail:
        raise AssertionError(
            "Q4 page bookkeeping: no SWA-tail prealloc (plain paged allocator only)"
        )
    if not load_quarantine_active(allocator):
        raise AssertionError(
            "Q4 page bookkeeping: an RDMA prealloc (symmetric-pd import, PD decode) needs "
            "the load quarantine (SGLANG_DSA_HOST_DEDUP_LOAD_QUARANTINE): without it "
            "no free page is known to be outside the forward in flight"
        )
    prefix_pages = None
    if prefix_len > 0 and total_prefix_len == prefix_len:
        prefix_pages = pages_of(prefix_indices, "decode prealloc prefix")
    res = allocator.alloc_import(total_prefix_len, fill_len, prefix_pages)
    if res is None:
        return None
    kv_loc, _new_pages, _dirty_epoch = res
    if len(kv_loc) != delta_len:
        raise AssertionError(
            f"Q4 decode prealloc: {len(kv_loc)} slots for {delta_len} tokens"
        )
    return kv_loc


def init_prebuilt_req_input(req: Req, tree_cache: BasePrefixCache) -> None:
    """Rebuild a request's fill ids and extend end before it joins a PREBUILT batch.

    With the decode radix cache the request's tree state (``prefix_len``,
    ``last_node``, ``kv.cache_protected_len``) and its tree lock come only from
    admission (``pop_preallocated`` -> ``_match_prefix_and_lock``). Resumed
    (retracted) and rebootstrapped requests hold request-private KV for the
    whole sequence and no tree lock (``last_node is None``), so they must not
    re-match here: ``match_prefix`` does not lock, and the next checkpoint
    would release a lock the request never took, while the private pages
    under the matched prefix would never be freed. Such a request starts at
    the root instead (``_bind_root_prefix``); its checkpoint dedups the
    private KV against the tree and takes the lock.
    """
    if get_disagg().disaggregation_decode_enable_radix_cache:
        if req.last_node is None:
            _bind_root_prefix(req, tree_cache)
        tree_cache = None
    req.init_next_round_input(tree_cache)
    # Truncate fill_len to kv_committed_len so checkpoint only sees committed
    # KV (full array includes one uncommitted token because
    # init_next_round_input rebuilt it as full).
    if req.kv.kv_committed_len is not None:
        req.extend_end = req.kv.kv_committed_len


def window_host_backfill_source(
    prefill_info: Any, *, attn_tp_size: int, host_dedup: bool
) -> Tuple[Optional[str], str]:
    """(source, reason): the prefill servicer a window_host decode worker
    backfills SWA from, or None with the reason
    backfill is off for this prefill. Peer fetch is rank-symmetric (reader rank
    r reads owner rank r, or both use rank 0 under host dedup), so both sides
    must share the attention TP size and the dedup mode. Only the prefill's
    rank-0 host is published and readers address owner rank r at
    <that host>:port + r, so a prefill spanning several nodes is refused
    without host dedup (``multi_node_owner``) instead of timing out against
    ports nothing listens on. The owner's PING capability is checked again at
    lookup time."""
    from decoupled_engine.srt.mem_cache.peer_kv_servicer import SWA_RANGE_CAPABILITY

    addr = getattr(prefill_info, "peer_kv_servicer_addr", None)
    if prefill_info is None or not addr:
        return None, "no_servicer"
    if SWA_RANGE_CAPABILITY not in (
        getattr(prefill_info, "peer_kv_capabilities", None) or ()
    ):
        return None, "no_capability"
    if int(prefill_info.attn_tp_size) != int(attn_tp_size):
        return None, "tp_mismatch"
    if bool(getattr(prefill_info, "peer_kv_host_dedup", False)) != bool(host_dedup):
        return None, "dedup_mismatch"
    if int(getattr(prefill_info, "peer_kv_nnodes", 1) or 1) > 1 and not host_dedup:
        return None, "multi_node_owner"
    return str(addr), "ok"


def _generate_fake_prefill_handoff_output_id(req: Req) -> int:
    """Generate a deterministic synthetic handoff token for fake-PD requests.

    Fake-PD skips prefill and KV transfer, so its allocated prompt KV has no
    request-specific state and its metadata output token remains zero. Reusing
    token 0 for every request can unrealistically correlate decode workloads.
    Hash the request ID so every TP rank gets the same request-diverse token.
    """
    if req.vocab_size is None or req.vocab_size <= 0:
        raise ValueError("Fake-PD requests require a positive vocabulary size")
    digest = hashlib.blake2b(
        req.rid.encode(), digest_size=8, person=b"fake-pd"
    ).digest()
    return int.from_bytes(digest, byteorder="little") % req.vocab_size


class DecodeTransferQueue(DecodeHiCacheTransferMixin):
    """
    Store the requests that is polling kv
    """

    enable_host_receive: bool = False

    def __init__(
        self,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        tree_cache: BasePrefixCache,
    ):
        self.queue: List[DecodeRequest] = []
        self.gloo_group = get_parallel().attn_tp_group.cpu_group
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = get_parallel().tp_rank
        self.metadata_buffers = metadata_buffers
        self.scheduler = scheduler
        self.enable_host_receive = (
            get_disagg().disaggregation_decode_host_receive_threshold < 1
        )
        self.tree_cache = tree_cache
        self.spec_algorithm = scheduler.spec_algorithm
        # Prefill layouts whose digests cannot be compared, logged once each.
        self._skipped_checksum_signatures: set = set()
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        self.staging_handler = None
        self.enable_deferred_kv_release = (
            envs.SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE.get()
        )
        self.deferred_kv_release_timeout = (
            envs.SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT.get()
        )
        # Aborted-mid-transfer requests whose KV pages/slot are held until drained
        # or timed out. Entries:
        # (decode_req, start_time, deadline, metadata_idx, required_acks).
        self._deferred_releases: List[Tuple[DecodeRequest, float, float, int, int]] = []

    def add(self, decode_req: DecodeRequest) -> None:
        self.queue.append(decode_req)

    def extend(self, decode_reqs: List[DecodeRequest]) -> None:
        self.queue.extend(decode_reqs)
        # This queue now covers them.
        prealloc_queue = self.scheduler.disagg_decode_prealloc_queue
        if prealloc_queue is not None:
            prealloc_queue.note_destinations_queued(len(decode_reqs))

    def _kv_checksum_comparable(self, signature: int) -> bool:
        """Digests compare only when the prefill digested this decode's layout.
        One with another TP width or a PP stage is admitted unverified and
        logged once, not aborted on every request. A prefill that wrote no
        digest (signature 0) is skipped silently; its registration already
        logged the buffer-count mismatch."""
        mine = self.scheduler.disagg_decode_prealloc_queue.kv_checksum_signature
        if signature == mine:
            return True
        if signature and signature not in self._skipped_checksum_signatures:
            self._skipped_checksum_signatures.add(signature)
            logger.warning(
                "KV checksum skipped: prefill layout signature %#x differs from "
                "this decode's %#x (TP width, PP stage, or page size); its "
                "requests are admitted unverified.",
                signature,
                mine,
            )
        return False

    def _commit_transfer_to_req(self, decode_req: DecodeRequest):
        # Keep the checksum before the metadata slot is freed; it is verified
        # when the request enters a batch.
        idx = decode_req.metadata_buffer_index
        if self.scheduler.kv_checksum_computer is not None:
            digest, signature = self.metadata_buffers.get_kv_checksum(idx)
            decode_req.req.expected_kv_checksum = (
                digest if self._kv_checksum_comparable(signature) else 0
            )
        (
            output_id,
            cached_tokens,
            output_token_logprobs_val,
            output_token_logprobs_idx,
            output_top_logprobs_val,
            output_top_logprobs_idx,
            output_token_sampling_mask_len,
            output_token_sampling_mask_idx,
            output_token_sampling_logprobs,
            output_topk_p,
            output_topk_index,
            output_hidden_states,
            output_dsa_topk_indices,
            output_bootstrap_room,
        ) = self.metadata_buffers.get_buf(idx)

        # Validate bootstrap_room to detect context corruption
        actual_room = output_bootstrap_room[0].item()
        expected_room = (
            decode_req.req.bootstrap_room
            if decode_req.req.bootstrap_room is not None
            else 0
        )

        if _is_fake_transfer(decode_req.req):
            pass
        elif actual_room == 0:
            # Should never happen: _poll_with_metadata_gate already confirmed
            # readiness on all TP ranks. Abort deterministically to avoid
            # cross-rank queue divergence.
            logger.error(
                f"Metadata unexpectedly not ready after readiness gate: "
                f"request {decode_req.req.rid}, bootstrap_room={expected_room}, "
                f"metadata_buffer_index={idx}"
            )
            prepare_abort(
                decode_req.req,
                "Metadata unexpectedly not ready after readiness gate "
                "(bootstrap_room=0)",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            decode_req.kv_receiver.clear()
            decode_req.kv_receiver = None
            return
        elif actual_room != expected_room:
            # Real corruption detected (mismatch)
            # Abort the request and remove from the queue
            error_msg = (
                f"Context corruption detected: Request {decode_req.req.rid} "
                f"(bootstrap_room={expected_room}) received metadata from "
                f"bootstrap_room={actual_room}. "
                f"Metadata buffer index: {idx}. "
                f"This indicates metadata buffer index collision."
            )
            logger.error(error_msg)
            prepare_abort(
                decode_req.req,
                "Metadata corruption detected - bootstrap_room mismatch",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            decode_req.kv_receiver.clear()
            decode_req.kv_receiver = None
            return

        self._commit_hicache_local_restore_to_req(decode_req)

        # Case 3: Success - commit the transfer
        # PD true-retraction rebootstrap: the prefill recomputed the prefix KV
        # under the current weights and sampled a fresh handoff token, but when
        # there is a remembered boundary token we are *replaying* an
        # already-emitted token. Override the handoff with it, and skip
        # re-committing a logprob or sampling mask for it -- it keeps its original
        # behavior logprob and sampling mask from before the retract (we never
        # re-score generated tokens under the new policy). A rebootstrap with no
        # boundary token (retracted before emitting any output) falls through to
        # the normal path so its first token and logprob are committed as usual.
        replayed_boundary = (
            decode_req.is_rebootstrap
            and decode_req.req.pd_rebootstrap_forced_output_id is not None
        )
        if replayed_boundary:
            committed_output_id = decode_req.req.pd_rebootstrap_forced_output_id
            decode_req.req.pd_rebootstrap_forced_output_id = None
        elif _is_fake_transfer(decode_req.req):
            committed_output_id = _generate_fake_prefill_handoff_output_id(
                decode_req.req
            )
        else:
            committed_output_id = output_id[0].item()
        decode_req.req.output_ids.append(committed_output_id)
        if not replayed_boundary:
            # The handoff token is generated on the prefill worker, so it does
            # not pass through the decode worker's normal batch-result path.
            # Account for it here using the same request-selected reasoning
            # terminator matcher as subsequent decode tokens.  A rebootstrap
            # boundary has already been accounted for before retraction and
            # must not be consumed twice.
            self.scheduler.batch_result_processor._maybe_update_reasoning_tokens(
                decode_req.req, committed_output_id
            )
        decode_req.req.cached_tokens = cached_tokens[0].item()
        # The prefill node already reported its prefix-cache hit in
        # cached_tokens[0]. Seed already_computed with it so that
        # prepare_for_prebuilt's `cached_tokens += pre_len - already_computed`
        # only adds decode-side reuse *beyond* what prefill counted, instead of
        # double-counting the shared prompt prefix (which would make
        # cached_tokens exceed prompt_tokens when decode radix cache is on).
        decode_req.req.already_computed = decode_req.req.cached_tokens
        decode_req.req.cached_tokens_device = cached_tokens[1].item()
        decode_req.req.cached_tokens_host = cached_tokens[2].item()
        decode_req.req.cached_tokens_storage = cached_tokens[3].item()
        # Multimodal prompt token counts packed into cached_tokens slots 4-6
        # by the prefill node (see MetadataBuffers.set_buf).
        decode_req.req.mm_image_tokens = cached_tokens[4].item()
        decode_req.req.mm_audio_tokens = cached_tokens[5].item()
        decode_req.req.mm_video_tokens = cached_tokens[6].item()
        decode_req.req.mm_token_counts_ready = True
        if not self.spec_algorithm.is_none():
            decode_req.req.output_topk_p = output_topk_p
            decode_req.req.output_topk_index = output_topk_index
            decode_req.req.hidden_states_tensor = output_hidden_states
            if (
                output_dsa_topk_indices is not None
                and torch.all(output_dsa_topk_indices < 0).item()
            ):
                output_dsa_topk_indices = None
            decode_req.req.output_dsa_topk_indices = output_dsa_topk_indices

        if decode_req.req.return_logprob and not replayed_boundary:
            decode_req.req.logprob.output_token_logprobs_val.append(
                output_token_logprobs_val[0].item()
            )
            decode_req.req.logprob.output_token_logprobs_idx.append(
                output_token_logprobs_idx[0].item()
            )
            decode_req.req.logprob.output_top_logprobs_val.append(
                output_top_logprobs_val[
                    : decode_req.req.logprob.top_logprobs_num
                ].tolist()
            )
            decode_req.req.logprob.output_top_logprobs_idx.append(
                output_top_logprobs_idx[
                    : decode_req.req.logprob.top_logprobs_num
                ].tolist()
            )
            if decode_req.req.logprob.token_ids_logprob is not None:
                # Prefill omits these scores; reserve a row for the streaming offset.
                decode_req.req.logprob.output_token_ids_logprobs_val.append([])
                decode_req.req.logprob.output_token_ids_logprobs_idx.append([])
        if decode_req.req.return_sampling_mask and not replayed_boundary:
            assert output_token_sampling_mask_idx is not None, (
                "sampling mask buffer disabled on decode side"
            )
            sampling_mask_len = int(output_token_sampling_mask_len[0].item())
            num_logprobs = (
                sampling_mask_len
                if decode_req.req.sampling_logprobs_mode == "support"
                else 1
            )
            decode_req.req.sampling_mask_rows.append(
                output_token_sampling_mask_idx[:sampling_mask_len].cpu().numpy(),
                output_token_sampling_logprobs[:num_logprobs].cpu().numpy(),
            )

        # window_host: the transfer committed on every TP rank (the poll is
        # all-reduced and the queue is walked in the same order), so this is
        # a lockstep point for the backfill manager's job list.
        self._submit_window_host_backfill(
            decode_req.req, getattr(decode_req.kv_receiver, "prefill_info", None)
        )
        decode_req.kv_receiver.clear()
        decode_req.kv_receiver = None
        decode_req.req.time_stats.set_wait_queue_entry_time()
        return

    def _submit_window_host_backfill(self, req: Req, prefill_info: Any) -> None:
        """window_host: the request holds FULL for its
        whole prompt but SWA only from T = kv.swa_evicted_seqlen on, and the
        matched prefix [0, dpl = cache_protected_len) is tree-owned. Queue the
        host->host backfill of [dpl, T) from the prefill's servicer into the
        request's ledger (opened at base dpl). It retries until prefill
        publishes the prompt (its write-through ack) or times out; whatever is
        missing at finish truncates the insert there."""
        tc = self.tree_cache
        if getattr(tc, "dsv4_uniform_swa_residency", "full") != "window":
            return
        if getattr(req, "skip_radix_cache_insert", False):
            return  # never inserted: nothing to make host-complete
        start = int(req.kv.cache_protected_len)
        end = int(req.kv.get_evicted_seqlen(ComponentType.SWA))
        if end <= start:
            return
        source, reason = window_host_backfill_source(
            prefill_info,
            attn_tp_size=get_parallel().attn_tp_size,
            host_dedup=bool(
                getattr(get_memory(), "enable_dsv4_hicache_host_dedup", False)
            ),
        )
        # prefill_info is this rank's own bootstrap fetch (try_ensure_parallel_info;
        # each rank's heartbeat drops and refetches it), so the verdict can
        # differ across TP ranks. Whether a job exists must not: whenever
        # backfill is on (rank-uniform), submit on every rank and carry a
        # missing source into the job. Its lookup then reports capable=0, the
        # poll's MIN ends it no_capability everywhere, and the finish insert
        # truncates at dpl.
        submitted = getattr(tc, "swa_backfill", None) is not None and bool(
            tc.submit_swa_backfill(
                req,
                start,
                end,
                source,
                tc.swa_backfill_page_keys(req, start, end),
                local_reason=None if source is not None else reason,
            )
        )
        if submitted and source is not None:
            return
        if source is not None:
            reason = "backfill_off"
        registry = getattr(tc, "swa_ledger_registry", None)
        if registry is not None:
            registry.stats["backfill_skipped_tokens"] += end - start
        # Warn once per reason; the per-request detail stays at debug level.
        warned = self.__dict__.setdefault("_window_host_backfill_warned", set())
        log = logger.debug if reason in warned else logger.warning
        warned.add(reason)
        log(
            "window_host rid=%s: no SWA backfill of [%d, %d) (%s); its finish "
            "insert stops at %d, so this worker will not own that prompt on "
            "its host",
            req.rid,
            start,
            end,
            reason,
            start,
        )

    def _pollers(self) -> list:
        if self.scheduler.enable_decode_hicache:
            return [HiCacheRestoreGatedKVReceiver(dr) for dr in self.queue]
        return [dr.kv_receiver for dr in self.queue]

    def _poll_with_metadata_gate(self) -> List[int]:
        return poll_and_all_reduce(
            self._pollers(),
            self.gloo_group,
            decode_reqs=self.queue,
            metadata_buffers=self.metadata_buffers,
        )

    def _poll_with_staging(self) -> list:
        return poll_and_all_reduce_with_staging(
            self.queue,
            self.staging_handler,
            self.gloo_group,
            metadata_buffers=self.metadata_buffers,
            pollers=self._pollers(),
        )

    def _init_staging_handler(self, kv_manager):
        """Create staging handler from kv_manager. Must be called exactly once."""
        from decoupled_engine.srt.disaggregation.common.staging_handler import (
            DecodeStagingHandler,
        )

        self.staging_handler = DecodeStagingHandler.create(
            kv_manager, self.scheduler, self.tp_rank
        )
        kv_manager._staging_handler = self.staging_handler

    def _release_request(self, decode_req: DecodeRequest) -> None:
        # HiSparse host pages are the transfer destination; free them with the KV.
        if self.scheduler.enable_hisparse:
            self.scheduler.hisparse_coordinator.request_finished(decode_req.req)
        if self.enable_host_receive:
            discard_kv_cache_backup(decode_req.req, self.tree_cache, "host_pool")
            if decode_req.host_staged:
                return
        release_kv_cache(decode_req.req, self.tree_cache, checkpoint=False)

    def _is_host_destination(self, decode_req: DecodeRequest) -> bool:
        return self.enable_host_receive and decode_req.host_staged

    def _record_transfer_failure(self, failure_reason: str) -> None:
        if self.scheduler.metrics_reporter.enable_metrics:
            self.scheduler.metrics_collector.increment_transfer_failed_reqs()

    def _abort_failed_transfer(self, decode_req: DecodeRequest) -> None:
        if self._is_host_destination(decode_req):
            decode_req.kv_receiver.abort()

    def pop_transferred(self, rids_to_check: Optional[List[str]] = None) -> List[Req]:
        if not self.queue:
            if self.scheduler.enable_decode_hicache:
                self._poll_hicache_restore_op_log()
            return []

        if self.scheduler.enable_decode_hicache:
            self._process_hicache_local_restores(
                [
                    decode_req
                    for decode_req in self.queue
                    if rids_to_check is None or decode_req.req.rid in rids_to_check
                ]
            )

        if self.enable_staging:
            polls = self._poll_with_staging()
        else:
            polls = self._poll_with_metadata_gate()

        transferred_reqs = []
        indices_to_remove = set()
        # Queue-removed but held for deferred release; excluded from the metadata
        # teardown below.
        deferred_indices = set()
        failed: List[Tuple[int, DecodeRequest, bool]] = []
        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if poll == KVPoll.Failed:
                self._abort_failed_transfer(decode_req)
                error_message = (
                    f"Decode transfer failed for request rank={self.tp_rank} "
                    f"{decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                )
                is_propagated = False
                failure_reason = "KV transfer failed"
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                    is_propagated = getattr(e, "is_from_another_rank", False)
                    failure_reason = getattr(e, "failure_reason", str(e))
                if decode_req.hicache_restore_status == HiCacheRestoreResult.FAILED:
                    is_propagated = False
                    failure_reason = "HiCache restore failed"
                    error_message += " (HiCache local restore failed on this rank)"
                self._clean_hicache_prefetch_resources(decode_req)
                # Mute error message for propagated exceptions to avoid duplicate logging
                if is_propagated:
                    logger.debug(error_message)
                else:
                    logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                self.scheduler.output_streamer.stream_output(
                    [decode_req.req],
                    decode_req.req.return_logprob,
                )
                requires_host_drain = self._is_host_destination(decode_req)
                deferrable = (
                    self.enable_deferred_kv_release
                    and decode_req.kv_receiver.kv_mgr.enable_deferred_decode_kv_release
                )
                if deferrable and not decode_req.kv_receiver.abort_notified:
                    # A failure decode did not initiate (prefill fault, transport
                    # error, hicache restore failure) can still have sibling-rank
                    # writes in flight toward these pages. force_arm: a partial
                    # send_metadata leaves init_time None though earlier ranks
                    # already hold destinations; this path always reaches
                    # _do_release, which clears the tracker.
                    decode_req.kv_receiver.ensure_abort_notified(force_arm=True)
                failed.append(
                    (
                        i,
                        decode_req,
                        requires_host_drain
                        or (deferrable and decode_req.kv_receiver.abort_notified),
                    )
                )
                indices_to_remove.add(i)
                self._record_transfer_failure(failure_reason)
                continue
            elif poll == KVPoll.Success:
                if self._is_host_destination(decode_req):
                    if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                        self.scheduler.output_streamer.stream_output(
                            [decode_req.req], decode_req.req.return_logprob
                        )
                        self._release_request(decode_req)
                        decode_req.kv_receiver.clear()
                        decode_req.kv_receiver = None
                        indices_to_remove.add(i)
                        continue
                    if not self.scheduler.disagg_decode_prealloc_queue.allocate_host_staged(
                        decode_req
                    ):
                        continue
                self._commit_transfer_to_req(decode_req)
                indices_to_remove.add(i)
                # Check if request was aborted due to corruption
                if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                    self.scheduler.output_streamer.stream_output(
                        [decode_req.req],
                        decode_req.req.return_logprob,
                    )
                    self._clean_hicache_prefetch_resources(decode_req)
                    self._release_request(decode_req)
                    self._record_transfer_failure(
                        decode_req.req.finished_reason.message
                    )
                else:
                    transferred_reqs.append(decode_req.req)
            elif poll in [
                KVPoll.Bootstrapping,
                KVPoll.WaitingForInput,
                KVPoll.Transferring,
            ]:
                pass
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

        # Notification state is rank-local. If any rank must hold a failed
        # transfer, all must retain its capacity so admission stays identical.
        held = all_reduce_flags(
            flags=[hold for _, _, hold in failed],
            group=self.gloo_group,
            op=torch.distributed.ReduceOp.MAX,
        )
        for (i, decode_req, _), hold in zip(failed, held):
            if hold:
                self._defer_release(decode_req)
                deferred_indices.add(i)
            else:
                self._release_request(decode_req)
                decode_req.kv_receiver.clear()
                decode_req.kv_receiver = None

        for i in indices_to_remove:
            if i in deferred_indices:
                # Held for deferred release; metadata buffer freed at resolve time.
                continue
            if self.enable_staging and self.staging_handler.is_staging_room(
                self.queue[i].req.bootstrap_room
            ):
                self.staging_handler.unregister_decode_req(
                    self.queue[i].req.bootstrap_room
                )
            idx = self.queue[i].metadata_buffer_index
            assert idx != -1
            # Reset so the next owner sees actual_room == 0 ("not yet written")
            # instead of the stale value, avoiding a false-positive mismatch.
            self.metadata_buffers.clear_slot(idx)
            self.req_to_metadata_buffer_idx_allocator.free(idx)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return transferred_reqs

    def _defer_release(self, decode_req: DecodeRequest) -> None:
        start_time = time.monotonic()
        deadline = start_time + self.deferred_kv_release_timeout
        # Host-capable receivers own their drain tracking. Device destinations
        # snapshot the number of prefill acknowledgements required by the manager.
        # A rank that only holds for a peer has no local acknowledgements to await.
        required_acks = (
            0
            if self._is_host_destination(decode_req)
            or not decode_req.kv_receiver.abort_notified
            else len(decode_req.kv_receiver.bootstrap_infos)
        )
        self._deferred_releases.append(
            (
                decode_req,
                start_time,
                deadline,
                decode_req.metadata_buffer_index,
                required_acks,
            )
        )

    def _do_release(self, decode_req: DecodeRequest, idx: int) -> None:
        room = decode_req.req.bootstrap_room
        if self.enable_staging and self.staging_handler.is_staging_room(room):
            self.staging_handler.unregister_decode_req(room)
        # release pre-allocated kv cache, but don't insert into the tree since it's failed
        self._release_request(decode_req)
        self.metadata_buffers.clear_slot(idx)
        self.req_to_metadata_buffer_idx_allocator.free(idx)
        if not self._is_host_destination(decode_req):
            decode_req.kv_receiver.kv_mgr.clear_deferred_abort_state(room)
        decode_req.kv_receiver.clear()
        decode_req.kv_receiver = None

    def has_pending_deferred_releases(self) -> bool:
        return bool(self._deferred_releases)

    def num_pending_deferred_releases(self) -> int:
        return len(self._deferred_releases)

    def _observe_deferred_kv_release(
        self, duration_seconds: float, outcome: str
    ) -> None:
        if self.scheduler.metrics_reporter.enable_metrics:
            self.scheduler.metrics_collector.observe_decode_deferred_kv_release(
                duration_seconds=duration_seconds,
                outcome=outcome,
            )

    def resolve_deferred_releases(self) -> None:
        """Release drained transfers; device destinations may also time out."""
        if not self._deferred_releases:
            return
        now = time.monotonic()
        drained = []
        ready = []
        for decode_req, _, deadline, _, required_acks in self._deferred_releases:
            if self._is_host_destination(decode_req):
                # A timeout cannot prove that a remote write has stopped.
                is_drained = decode_req.kv_receiver.is_abort_release_safe()
                is_ready = is_drained
            else:
                is_drained = decode_req.kv_receiver.kv_mgr.is_abort_release_safe(
                    decode_req.req.bootstrap_room, required_acks
                )
                is_ready = is_drained or now >= deadline
            drained.append(is_drained)
            ready.append(is_ready)
        # Acks and deadlines are rank-local. Releasing only on some ranks would
        # diverge admission and the sizes of subsequent polling collectives.
        # Entries are appended from agreed holds, so every rank reduces the same
        # list in the same order.
        ready = all_reduce_flags(
            flags=ready, group=self.gloo_group, op=torch.distributed.ReduceOp.MIN
        )
        still_held = []
        to_release = []
        for entry, is_ready, is_drained in zip(self._deferred_releases, ready, drained):
            decode_req, start_time, _, idx, _ = entry
            if is_ready:
                to_release.append(
                    (
                        decode_req,
                        start_time,
                        idx,
                        decode_req.req.bootstrap_room,
                        is_drained,
                    )
                )
            else:
                still_held.append(entry)
        # Commit the survivors before releasing so a _do_release exception can't
        # leave a released entry in the list (double-free / None receiver on retry).
        self._deferred_releases = still_held
        for decode_req, start_time, idx, room, drained in to_release:
            if not drained:
                logger.warning(
                    f"Deferred KV release for room {room} timed out after "
                    f"{self.deferred_kv_release_timeout}s without a full drain "
                    f"ack from prefill; releasing anyway."
                )
            outcome = "drained" if drained else "timeout"
            try:
                self._do_release(decode_req, idx)
            except Exception:
                # Isolate a failed release so the rest still run; entry already dropped.
                logger.exception(f"Deferred KV release failed for room {room}")
                outcome = "error"
            self._observe_deferred_kv_release(
                duration_seconds=max(0.0, now - start_time),
                outcome=outcome,
            )

    def release_memory_occupation(self):
        """Clean up in-flight transfers before releasing GPU memory."""
        self.queue.clear()
        # Pool is being torn down; drop held entries without per-request release.
        self._deferred_releases.clear()

    def resume_memory_occupation(self):
        """Queues are already cleared on release; new transfers can be accepted."""
        pass


class SchedulerDisaggregationDecodeMixin:
    @torch.no_grad()
    def event_loop_normal_disagg_decode(self: Scheduler):
        """A normal scheduler loop for decode worker in disaggregation mode."""
        # SGLANG_KV_EVENTS_BATCH_PUBLISH: one KV-event flush at the end of the
        # pass. None when off (no call).
        kv_events_end_pass = kv_events_end_pass_hook(self)

        while True:
            if self.gracefully_exit:
                break

            # Pending rooms from the prior cycle can overlap request intake and
            # the tail of the in-flight decode graph.
            if not self._engine_paused:
                self.disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries()
            # Receive requests
            self.ingest_requests()
            if self._engine_paused:
                self._record_scheduler_state_for_paused_engine()
                continue
            self.process_decode_queue()

            # Get the next batch to run
            plan = self.get_next_disagg_decode_batch_to_run(
                running_batch=self.running_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states
                self._sched_idled = True
                self.on_idle()

            if kv_events_end_pass is not None:
                kv_events_end_pass()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_decode(self: Scheduler):
        self.result_queue = deque()
        self.last_batch: Optional[ScheduleBatch] = None
        # 6h (SGLANG_PREFILL_DEFER_WRITE_SUBMIT): write-throughs queued behind
        # a WAR barrier (the prebuilt insert, finished requests) are submitted
        # after the next launch (direct IO reads their indices with a blocking
        # .cpu()). Inert when off.
        from decoupled_engine.srt.managers.scheduler_components.deferred_write_submit import (
            DeferredWriteSubmit,
        )

        write_defer = DeferredWriteSubmit.for_loop(self, "decode")

        def pop_and_process():
            with phase("loop.result"):
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)

        # SGLANG_KV_EVENTS_BATCH_PUBLISH: one KV-event flush at the end of the
        # pass. None when off (no call).
        kv_events_end_pass = kv_events_end_pass_hook(self)
        # SGLANG_SCHED_PHASE_TIMERS: the loop steps as loop.* / std.* phases
        # (launch gap attribution, as the symmetric overlap loop); off: one
        # local bool test per step.
        timers_on = PHASES.enabled
        _t = 0.0

        while True:
            if self.gracefully_exit:
                break

            # Pending rooms from the prior cycle can overlap request intake and
            # the tail of the in-flight decode graph.
            if timers_on:
                _t = PHASES.start()
            if not self._engine_paused:
                self.disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries()
            # Receive requests
            self.ingest_requests()
            if timers_on:
                PHASES.stop("loop.recv", _t)
            if self._engine_paused:
                write_defer.flush()
                self._record_scheduler_state_for_paused_engine()
                continue
            if timers_on:
                _t = PHASES.start()
            self.process_decode_queue()
            if timers_on:
                PHASES.stop("std.process_decode_queue", _t)

            # Get the next batch to run
            if timers_on:
                _t = PHASES.start()
            plan = self.get_next_disagg_decode_batch_to_run(
                running_batch=self.running_batch
            )
            if timers_on:
                PHASES.stop("loop.get_next_batch", _t)
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch
            # overlap + spec + grammar is unsupported (would desync DP ranks).
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(
                batch, last_batch=self.last_batch
            )

            if disable_overlap_for_batch and self.last_batch:
                pop_and_process()

            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                # 6h: held writes, before this launch's barrier.
                write_defer.flush()
                self._apply_war_barrier()
                write_defer.hold()
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None
                self._sched_idled = True
                # 6h: no new barrier; hold the result's writes until after it.
                write_defer.flush()
                write_defer.hold()

            # Process the last batch
            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                self.on_idle()
            if batch is None:
                write_defer.flush()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            with phase("loop.sample"):
                self.launch_batch_sample_if_needed(batch_result, batch)

            if kv_events_end_pass is not None:
                with phase("loop.kv_events"):
                    kv_events_end_pass()

            # Update last_batch
            self.last_batch = batch

    def _run_batch_prebuilt(
        self: Scheduler, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        if batch.inner_idle_batch is not None:
            idle_batch = batch.inner_idle_batch
            # Reset the inner idle batch to avoid reusing it.
            batch.inner_idle_batch = None
            return self.run_batch(idle_batch)

        return GenerationBatchResult()

    @scheduler_stage_method(SCHEDULER_STAGE_GET_NEXT_BATCH)
    def get_next_disagg_decode_batch_to_run(
        self: Scheduler, running_batch: ScheduleBatch
    ) -> NextBatchPlan:
        """Process prebuilt batch and schedule the next decode batch."""
        # Process pending prebuilt batch: output processing + filter + merge
        # (admission path phases: std_new_prebuilt_batch > prebuilt_build /
        # prebuilt_wait_stream / prebuilt_draft_input; prebuilt_result;
        # prebuilt_merge; the same prebuilt_* names as the symmetric D loop).
        with phase("std_new_prebuilt_batch"):
            new_prebuilt_batch = self.get_new_prebuilt_batch(running_batch)
        if new_prebuilt_batch:
            assert self.chunked_req is None
            self.batch_result_processor.process_batch_result_prebuilt(
                new_prebuilt_batch
            )
            with phase("prebuilt_merge"):
                new_prebuilt_batch.filter_batch()
                if not new_prebuilt_batch.is_empty():
                    if running_batch.is_empty():
                        running_batch = new_prebuilt_batch
                        if self.enable_hisparse:
                            running_batch.hisparse_coordinator = (
                                self.hisparse_coordinator
                            )
                    else:
                        running_batch.merge_batch(new_prebuilt_batch)

        # Schedule decode batch
        if running_batch.is_empty():
            ret = None
        else:
            running_batch = self.update_running_batch(running_batch)
            ret = running_batch if not running_batch.is_empty() else None

        ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(ret)
        if ret:
            set_schedule_time_batch(ret)
        return NextBatchPlan(batch_to_run=ret, running_batch=running_batch)

    # SGLANG_SCHED_SYNC_DEBUG site (identity when off): the standard D
    # admission's synchronizing CUDA calls (SGLANG_PREBUILT_NONBLOCKING_H2D).
    @sync_debug_site("std_new_prebuilt_batch")
    def get_new_prebuilt_batch(
        self, running_batch: ScheduleBatch
    ) -> Optional[ScheduleBatch]:
        batch = self._get_new_prebuilt_batch(running_batch)
        if batch is None:
            return None
        # Verify after host restore and before process_prebuilt caches any KV.
        self._verify_kv_checksums(batch)
        if batch.is_empty():
            return None
        self.ngram_embedding_manager.prepare_for_forward(
            batch, chunked_req=self.chunked_req
        )
        with phase("prebuilt_draft_input"):
            batch.process_prebuilt(self.future_map)
        return batch

    def _verify_kv_checksums(self, batch: ScheduleBatch) -> None:
        computer: Optional[KvChecksumComputer] = self.kv_checksum_computer
        if computer is None:
            return

        prealloc_queue = self.disagg_decode_prealloc_queue
        kv_manager = prealloc_queue.kv_manager
        mismatch_prob = envs.SGLANG_TEST_DISAGG_KV_CHECKSUM_MISMATCH_PROB.get()
        mismatched = [0] * len(batch.reqs)
        messages: Dict[int, str] = {}
        for i, req in enumerate(batch.reqs):
            expected = req.expected_kv_checksum
            if (
                expected == 0
                or is_health_check_req(req)
                or not kv_manager.should_verify_kv_checksum(_bootstrap_addr(req))
            ):
                continue
            # Verified once. A retraction restores the row after decode has
            # written its own tokens into the last page, so the handoff digest
            # no longer describes it.
            req.expected_kv_checksum = 0
            seq_len = len(req.origin_input_ids)
            # The tree cache owns [0, cache_protected_len): the prefix this
            # decode promised at bootstrap, so prefill sent and digested only
            # what follows it.
            start_idx = req.kv.cache_protected_len
            page_indices_gpu = page_indices_for_request(
                self, req, seq_len, start_idx=start_idx
            )
            state_indices = state_indices_for_request(
                self, req, seq_len, start_idx=start_idx
            )
            actual = computer.compute(page_indices_gpu, state_indices)
            if (
                mismatch_prob > 0
                and prealloc_queue.kv_checksum_mismatch_injector.random()
                < mismatch_prob
            ):
                actual ^= 1
            if actual != expected:
                mismatched[i] = 1
                messages[i] = (
                    f"KV checksum mismatch req={req.rid} "
                    f"bootstrap_room={req.bootstrap_room} "
                    f"expected={expected:#x} got={actual:#x}"
                )

        # Corruption is per rank. Agree on the drop set first, or the TP ranks
        # run different batches and hang on the next collective.
        mismatched = all_reduce_flags(
            mismatched, self.attn_tp_cpu_group, dist.ReduceOp.MAX
        )
        if not any(mismatched):
            return
        for i, req in enumerate(batch.reqs):
            if mismatched[i]:
                msg = messages.get(
                    i,
                    f"KV checksum mismatch on a peer rank req={req.rid} "
                    f"bootstrap_room={req.bootstrap_room}",
                )
                logger.error(msg)
                self._handle_kv_checksum_mismatch(req, msg)
        batch.filter_batch(
            keep_indices=[i for i, bad in enumerate(mismatched) if not bad]
        )

    def _handle_kv_checksum_mismatch(self, req: Req, msg: str) -> None:
        # A mismatch means the KV this worker received is not what prefill sent,
        # so the cause is hardware or transport rather than the request. Serving
        # keeps going and drops just this request; CI fails instead, because a
        # single aborted request is easy to miss in a passing run -- unless the
        # mismatch was injected to exercise this very path.
        if is_in_ci() and envs.SGLANG_TEST_DISAGG_KV_CHECKSUM_MISMATCH_PROB.get() == 0:
            raise RuntimeError(msg)
        prepare_abort(
            req,
            "KV checksum mismatch",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
        self.output_streamer.stream_output([req], req.return_logprob)
        release_kv_cache(req, self.tree_cache, checkpoint=False)
        if self.metrics_reporter.enable_metrics:
            self.metrics_collector.increment_transfer_failed_reqs()

    def _get_new_prebuilt_batch(
        self: Scheduler, running_batch: ScheduleBatch
    ) -> Optional[ScheduleBatch]:
        """Create a schedulebatch for fake completed prefill"""
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if len(self.waiting_queue) == 0:
            return None

        if self.enable_priority_scheduling:
            self.policy.calc_priority(self.waiting_queue, running_batch)

        curr_batch_size = running_batch.batch_size()

        batch_size = min(self.req_to_token_pool.size, self.max_running_requests)

        num_not_used_batch = batch_size - curr_batch_size

        # pop req from waiting queue
        can_run_list: List[Req] = []
        waiting_queue: List[Req] = []

        for i in range(len(self.waiting_queue)):
            req = self.waiting_queue[i]
            # we can only add at least `num_not_used_batch` new batch to the running queue
            if i < num_not_used_batch:
                can_run_list.append(req)
                # Decode radix: never re-match here (see init_prebuilt_req_input).
                init_prebuilt_req_input(req, self.tree_cache)
            else:
                waiting_queue.append(req)

        self.waiting_queue = waiting_queue
        if len(can_run_list) == 0:
            return None

        with phase("prebuilt_build"):
            set_time_batch(can_run_list, "set_forward_entry_time")

            # construct a schedule batch with those requests and mark as decode
            new_batch = ScheduleBatch.init_new(
                can_run_list,
                self.req_to_token_pool,
                self.token_to_kv_pool_allocator,
                self.tree_cache,
                self.model_config,
                self.enable_overlap,
                self.spec_algorithm,
            )

            # construct fake completed prefill
            new_batch.prepare_for_prebuilt()
        if self.enable_overlap:
            # A finished request can still have one redundant forward in flight.
            # Drain it before a prebuilt request seeds a potentially reused row.
            with phase("prebuilt_wait_stream"):
                self.schedule_stream.wait_stream(self.forward_stream)
        if get_disagg().disaggregation_decode_host_receive_threshold < 1:
            for req in new_batch.reqs:
                if req.kv.retraction_backup is not None:
                    restore_kv_cache(
                        req,
                        self.tree_cache,
                        self.req_to_token_pool,
                        self.token_to_kv_pool_allocator,
                        "host_pool",
                    )
        return new_batch

    def pull_late_d_controls(
        self: Scheduler, recv_reqs: Optional[List] = None
    ) -> List[LateDControl]:
        """Late D, attn-TP rank 0 before the request broadcast: ADMITs from
        the decode thread and DEFERRED expiries, as broadcast controls.
        ``recv_reqs``: the raw batch about to be broadcast (PD ignores it)."""
        prealloc_queue = self.disagg_decode_prealloc_queue
        return prealloc_queue.late_d.pull_controls(
            prealloc_queue.queue, () if recv_reqs is None else recv_reqs
        )

    def handle_late_d_control(self: Scheduler, control: LateDControl) -> None:
        """Late D, every rank in the same iteration and order."""
        prealloc_queue = self.disagg_decode_prealloc_queue
        if prealloc_queue is not None and prealloc_queue.late_d is not None:
            prealloc_queue.late_d.apply_control(control, prealloc_queue.queue)

    @scheduler_stage_method(SCHEDULER_STAGE_PROCESS_QUEUE)
    def process_decode_queue(self: Scheduler):
        if self.enable_decode_hicache:
            with phase("std.hicache_events"):
                self.tree_cache.check_hicache_events()

        if get_disagg().disaggregation_decode_enable_offload_kvcache:
            self.decode_offload_manager.check_offload_progress()

        # Resolve held releases every iteration (before the retraction/polling
        # gates below) so their timeouts fire under memory pressure.
        self.disagg_decode_transfer_queue.resolve_deferred_releases()

        # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
        resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs()
        self.waiting_queue.extend(resumed_reqs)
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            return

        if not hasattr(self, "polling_count"):
            self.polling_count = 0
            self.polling_interval = get_disagg().disaggregation_decode_polling_interval

        self.polling_count = (self.polling_count + 1) % self.polling_interval

        if self.polling_count % self.polling_interval == 0:
            if get_disagg().disaggregation_decode_host_receive_threshold == 1:
                # std.poll_all_reduce (utils._all_reduce_polls) is the cross-rank
                # MIN vote inside both pops; it waits for the slowest TP rank.
                with phase("std.prealloc_pop"):
                    req_conns, _ = self.disagg_decode_prealloc_queue.pop_preallocated()
                self.disagg_decode_transfer_queue.extend(req_conns)
            with phase("std.transfer_pop"):
                transferred_reqs = (
                    self.disagg_decode_transfer_queue.pop_transferred()
                )  # the requests which kv has arrived
            if self.enable_hisparse:
                for req in transferred_reqs:
                    # Direct-to-host: KV data already in host pool, skip staging
                    self.hisparse_coordinator.admit_request_direct(req)
            self.waiting_queue.extend(transferred_reqs)
            if get_disagg().disaggregation_decode_host_receive_threshold < 1:
                # Give completed host transfers device space before new arrivals.
                req_conns, _ = self.disagg_decode_prealloc_queue.pop_preallocated()
                self.disagg_decode_transfer_queue.extend(req_conns)
