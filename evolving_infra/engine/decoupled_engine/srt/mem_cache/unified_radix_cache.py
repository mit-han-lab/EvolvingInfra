# Adapted from SGLang python/sglang/srt/mem_cache/unified_radix_cache.py (commit 9a489f8d2f), modified for evolving-infra.
"""Coordinate unified GPU/host cache operations, async IO ACKs and peer returns.

The tree core owns residency and eviction; this layer applies transfer results
at scheduler agreement points. Deferred host returns execute controller-issued
directives through the existing write-through and storage-prefetch paths.
"""

from __future__ import annotations

import atexit
import functools
import json
import logging
import os
import socket
import threading
import time
import uuid
from array import array
from dataclasses import dataclass, replace
from queue import Queue
from typing import TYPE_CHECKING, Iterator, NamedTuple, Optional, Sequence, TypeVar

import torch

from decoupled_engine.srt.disaggregation.kv_events import StorageMedium
from decoupled_engine.srt.distributed.communication_tags import P2PTag
from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.managers.cache_controller import CacheOperation, log_load_op_settled
from decoupled_engine.srt.managers.scheduler_components import collective_guard
from decoupled_engine.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from decoupled_engine.srt.mem_cache.buffer_mode.pipeline import (
    BufferModePipeline,
    validate_buffer_only_stack,
)
from decoupled_engine.srt.mem_cache.buffer_mode.storage_existence_cache import (
    StorageExistenceCache,
)
from decoupled_engine.srt.mem_cache.common import RetractionBackup
from decoupled_engine.srt.mem_cache.cpu_page_book import (
    cat_tagged,
    cpu_rows_of,
    slice_tagged,
    tag_row_slice,
)
from decoupled_engine.srt.mem_cache.hicache_pipelined_sync import ALGOS as PIPELINED_SYNC_ALGOS
from decoupled_engine.srt.mem_cache.hicache_pipelined_sync import (
    PendingStepSync,
    PipelinedSyncLayout,
    create_pipelined_sync_group,
    start_min_reduce,
)
from decoupled_engine.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    SidecarPoolSpec,
    skip_hintless_storage_lookup,
    skip_noop_storage_backup,
)
from decoupled_engine.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from decoupled_engine.srt.mem_cache.memory_pool import MHATokenToKVPool
from decoupled_engine.srt.mem_cache.peer_kv_demote import (
    SCOPE_QUEUED_MAX,
    PeerKVDemoteReq,
    RecordWriter,
    ScopeContext,
    notice_insert_cause,
    scope_mode,
)
from decoupled_engine.srt.mem_cache.peer_kv_touch import PeerKVTouchReq
from decoupled_engine.srt.mem_cache.radix_cache import RadixKey
from decoupled_engine.srt.mem_cache.swa_memory_pool import SWAKVPool
from decoupled_engine.srt.mem_cache.unified_cache.cache_action import (
    BackupKV,
    CacheAction,
    ComponentAction,
    FreeComponentDeviceSlot,
    FreeDeviceKV,
    FreeDeviceKVFullOnly,
    ReplaceWriteThroughOnNodeSplit,
)

# UnifiedTreeNode / UnifiedLRUList live on the tree core; re-exported here
# because other modules and tests import them from this module.
from decoupled_engine.srt.mem_cache.unified_cache.components import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
    FullComponent,
    MambaComponent,
    PrepareLoadBackResult,
    SWAComponent,
    TreeComponent,
    peek_time_counter,
)
from decoupled_engine.srt.mem_cache.unified_cache.retention_order import (
    ORDER_DEFAULT,
    ORDER_PREDICTED_RETURN_FIXED,
    load_fixed_return_table,
    parse_retention_order,
)
from decoupled_engine.srt.mem_cache.unified_cache.session_ref_tracker import (
    KvSessionRetention,
    UnifiedSessionRefTracker,
    validate_kv_session_operations,
)
from decoupled_engine.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from decoupled_engine.srt.mem_cache.unified_cache.swa_ledger import (
    INFLIGHT,
    REATTACHED,
    LedgerDisposition,
    SwaHostLedger,
    SwaLedgerRegistry,
    SwaSegment,
    SwaSpillOp,
    is_spill_ack_id,
)
from decoupled_engine.srt.mem_cache.unified_cache.tree_core_registry import create_tree_core
from decoupled_engine.srt.mem_cache.unified_cache.unified_tree_core import (  # noqa: F401
    NodeId,
    UnifiedLRUList,
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from decoupled_engine.srt.observability.metrics_collector import (
    DSV4WindowMetricsCollector,
    StorageMetrics,
    StorageMetricsCollector,
)
from decoupled_engine.srt.runtime_context import (
    get_context,
    get_disagg,
    get_memory,
    get_observability,
)
from decoupled_engine.srt.session.streaming_session import StreamingSession
from decoupled_engine.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from decoupled_engine.srt.managers.cache_controller import HiCacheAck
    from decoupled_engine.srt.managers.schedule_batch import Req
    from decoupled_engine.srt.mem_cache.cache_init_params import CacheInitParams
    from decoupled_engine.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        PrefetchOperation,
    )
    from decoupled_engine.srt.mem_cache.pool_host import PoolEntry
    from decoupled_engine.srt.server_args import ServerArgs

from decoupled_engine.srt.managers.scheduler_components.phase_timers import timed_phase
from decoupled_engine.srt.utils.rank_consensus_checker import rank_consensus

T = TypeVar("T")


@dataclass
class _PendingHostReturn:
    """A controller-authorized return; holds CPU pages, never a live Req/GPU lock."""

    node_id: NodeId
    lock_params: DecLockRefParams
    key: RadixKey
    prompt_tokens: int
    command: object
    deadline: float
    owner_cache_epoch: str
    notified: bool = False
    kv_session: Optional[dict] = None


# Bound metadata and retained CPU anchors independently of inference concurrency.
_MAX_HOST_RETURNS = 64
_MAX_HOST_RETURN_TOKENS = 4 * 1024 * 1024

# Metric label per component, matching the host pool names used by
# hicache_backup_tokens_total and the host occupancy gauges.
_COMPONENT_POOL_LABEL = {
    ComponentType.FULL: PoolName.KV.value,
    ComponentType.SWA: PoolName.SWA.value,
    ComponentType.MAMBA: PoolName.MAMBA.value,
}


COMPONENT_REGISTRY: dict[ComponentType, type[TreeComponent]] = {
    ComponentType.FULL: FullComponent,
    ComponentType.MAMBA: MambaComponent,
    ComponentType.SWA: SWAComponent,
}


logger = logging.getLogger(__name__)

# Cadence of the controller-side storage counters for backends without native
# stats (_log_storage_metrics).
STORAGE_COUNTERS_LOG_INTERVAL_S = 1.0


class _OngoingWriteThrough(NamedTuple):
    """Tracks an in-flight D→H write-through operation."""

    node_id: NodeId
    lock_params: Optional[DecLockRefParams]
    publish_node_ids: list[NodeId]


class _OngoingLoadBack(NamedTuple):
    """Tracks an in-flight H→D load-back operation."""

    node_id: NodeId
    lock_params: DecLockRefParams
    host_lock_params: DecLockRefParams


class _OngoingPrefetch(NamedTuple):
    """Tracks an in-flight storage→host prefetch operation."""

    anchor_node_id: NodeId
    prefetch_key: RadixKey
    host_indices: torch.Tensor
    operation: PrefetchOperation
    anchor_lock_params: DecLockRefParams
    comp_xfers: dict[ComponentType, list[PoolTransfer]]


class SwaLagLock(NamedTuple):
    """Window SWA residency: the request lock taken one chunk stash earlier. It
    keeps the pages the in-flight chunk still reads locked until the next
    relock, finish or abort."""

    node_id: NodeId
    params: DecLockRefParams


# Engines whose decode-batch spills take the staged copy by default.
_STAGED_DECODE_SPILL_MODES = ("decode", "symmetric")


def swa_spill_sm_copy(decode: bool) -> bool:
    """Whether a window SWA spill copies with SM kernels (True) or takes the
    staged copy-engine path (SGLANG_DSV4_SWA_SPILL_SM_COPY; set, it decides
    for every spill). Unset, by role: a copy-engine D2H stalls every later
    device->host readback, which a prefill's chunk pipeline waits on (a
    symmetric owner's export spills, measured: 9.8 vs 4.0 ms chunk gap), while the
    SM copy takes SMs from the forward, which costs a decode step more (measured:
    2.2 vs 0.8 ms per decode spill). ``decode``: the spill comes from a
    decode batch (ScheduleBatch._maybe_spill_swa_window). Those take the
    staged copy on a DECODE engine and on a SYMMETRIC one (its D-leg); a
    colocated (NULL) engine keeps the SM copy, as its decode batches share
    the loop with its prefill chunks."""
    env = envs.SGLANG_DSV4_SWA_SPILL_SM_COPY
    if env.is_set():
        return env.get()
    if not decode:
        return True
    mode = get_disagg().disaggregation_mode
    return getattr(mode, "value", mode) not in _STAGED_DECODE_SPILL_MODES


def _d_role_kv_events(method):
    """SGLANG_SYMMETRIC_SKIP_D_ROLE_KV_EVENTS: a cache call for a D-leg request
    (``req.kv_events_d_role``, set by the symmetric scheduler at arrival) runs
    as D-role work of the KV-event gate (mem_cache/events.py DRoleEventGate):
    the nodes it adds publish nothing. Other requests, and every request
    without the gate, call through unchanged."""

    @functools.wraps(method)
    def wrapper(self, req, *args, **kwargs):
        if getattr(req, "kv_events_d_role", False) is not True:
            return method(self, req, *args, **kwargs)
        gate = self.tree_core.kv_events.d_role_gate
        if gate is None:
            return method(self, req, *args, **kwargs)
        with gate.d_role():
            return method(self, req, *args, **kwargs)

    return wrapper


class UnifiedRadixCache(BasePrefixCache):
    # Class defaults so partially constructed caches (object.__new__ in unit
    # tests) keep today's behaviour.
    dsv4_uniform_swa_residency = "full"
    # Batch results processed so far (window residency, see
    # on_batch_result_processed).
    _swa_batches_processed = 0
    # Window residency: spill ops by ack id (swa_ledger.py); None otherwise.
    swa_ledger_registry: Optional[SwaLedgerRegistry] = None
    # Window residency: host->host SWA backfills (swa_backfill.py); None when
    # backfill is off or no peer storage backend is configured.
    swa_backfill = None
    # Window residency: SWA window load-backs submitted (tokens, ops).
    _swa_load_back_tokens = 0
    _swa_load_back_ops = 0
    # Debug SGLANG_DSV4_SWA_COPY_VERIFY (window residency only): transfers
    # awaiting their bitwise check, and the per-(phase, pool) counters.
    _swa_copy_verify = False
    _swa_copy_verify_spills: Optional[dict] = None
    _swa_copy_verify_loads: Optional[dict] = None
    _swa_copy_verify_stats: Optional[dict] = None
    # Window residency: (req, node id) chunk-boundary backups queued by the
    # stashes for run_deferred_chunk_backups; a list only while an overlap
    # loop drains them (SGLANG_DSV4_CHUNK_WINDOW_WORK_AFTER_LAUNCH).
    _deferred_chunk_backups: Optional[list] = None
    kv_sessions = None
    enable_kv_session_retention = False

    def __init__(
        self,
        params: CacheInitParams,
    ):
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.disable = params.disable
        # Native prefill must retain the same image-safe restart window as
        # symmetric export. Decode imports only a tail and cannot promise the
        # older prompt SWA exists locally.
        self._native_prefill_image_swa = bool(
            get_context().is_config_namespace_published("disagg")
            and get_disagg().disaggregation_mode == "prefill"
        )

        if params.enable_metrics:
            self.init_metrics_collector()
        self._enable_metrics_flag = params.enable_metrics
        self.enable_storage_metrics = False
        self.storage_metrics_collector: Optional[StorageMetricsCollector] = None
        self.extra_metric_labels = None

        assert params.tree_components is not None
        self.dsv4_uniform_cache_lifecycle = bool(
            get_context().is_config_namespace_published("memory")
            and getattr(get_memory(), "dsv4_uniform_cache_lifecycle", False)
        )
        # Device SWA residency under the uniform lifecycle ("full" or
        # "window"); SWAComponent reads it at construction below.
        self.dsv4_uniform_swa_residency = (
            getattr(get_memory(), "dsv4_uniform_swa_residency", "full")
            if self.dsv4_uniform_cache_lifecycle
            else "full"
        )
        self.dsv4_window_metrics: Optional[DSV4WindowMetricsCollector] = None
        if self.dsv4_uniform_swa_residency == "window":
            logger.warning(
                "EXPERIMENTAL: --dsv4-uniform-swa-residency window. The device "
                "keeps SWA only for matched windows and each running request's "
                "window; decoding requests spill older SWA to the host."
            )
            self.swa_ledger_registry = SwaLedgerRegistry()
            # Ledger counters already published (_update_swa_window_metrics).
            self._swa_metrics_reported: dict[str, int] = {}
            # Read once: when off, the copy check costs nothing per step.
            self._swa_copy_verify = envs.SGLANG_DSV4_SWA_COPY_VERIFY.get()
            if self._swa_copy_verify:
                logger.warning(
                    "SGLANG_DSV4_SWA_COPY_VERIFY: SWA spill and window "
                    "load-back copies are checked bit for bit (host syncs)."
                )
                self._swa_copy_verify_spills = {}
                self._swa_copy_verify_loads = {}
                self._swa_copy_verify_stats = {}
            if params.enable_metrics:
                self.dsv4_window_metrics = DSV4WindowMetricsCollector(
                    labels=dict(self.metrics_collector.labels)
                )
        self.tree_components = tuple(params.tree_components)
        self.enable_session_radix_cache = params.enable_session_radix_cache
        self.enable_kv_session_retention = getattr(
            params, "enable_kv_session_retention", False
        )
        if self.enable_kv_session_retention and (
            self.tree_components != (ComponentType.FULL,)
            or self.enable_session_radix_cache
            or params.disable
            or params.eviction_policy.lower() != "lru"
            or envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.get() != "python"
        ):
            raise ValueError(
                "KV session retention requires Python unified FULL-only LRU cache, "
                "without native session radix caching or disabled cache"
            )
        component_registry = COMPONENT_REGISTRY
        if params.component_registry_override:
            component_registry = {
                **COMPONENT_REGISTRY,
                **params.component_registry_override,
            }
        self.components: dict[ComponentType, TreeComponent] = {
            ct: component_registry[ct](self, params) for ct in self.tree_components
        }
        self._components_tuple: tuple[TreeComponent, ...] = tuple(
            self.components.values()
        )
        # Whether SWA is enabled.
        self.is_swa_enabled = ComponentType.SWA in params.tree_components
        # Whether Mamba is enabled.
        self.is_mamba_enabled = ComponentType.MAMBA in params.tree_components
        # Whether the mamba extra (ping-pong) buffer is enabled.
        self.enable_mamba_extra_buffer = (
            params.enable_mamba_extra_buffer if self.is_mamba_enabled else False
        )
        # SWA window size (None when SWA is not enabled).
        self._sliding_window_size = (
            params.sliding_window_size if self.is_swa_enabled else None
        )
        # The TreeCore owns the tree member-var state (structure, LRUs, sizes,
        # evictable leaves) and drives the components' tree-level hooks.
        self.tree_core = create_tree_core(
            name=envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.get(),
            params=params,
            components=self.components,
        )
        # Components execute boundary actions through the tree core.
        for component in self.components.values():
            component.tree_core = self.tree_core

        # Session ref tracking (--enable-session-radix-cache).
        self.session_refs = UnifiedSessionRefTracker(
            components=self._components_tuple,
            tree_core=self.tree_core,
            enable_session_radix_cache=self.enable_session_radix_cache,
        )
        self.kv_sessions = self._make_kv_session_retention()
        self.tree_core.kv_session_retention = self.kv_sessions
        # The scheduler broadcasts one worker UUID after TP initialization.
        self.kv_session_boot_id = None
        self._kv_session_flush_generation = 0

        self.sidecar_pool_specs: list[SidecarPoolSpec] = []

        # Streaming session: embedded StreamingSession with self as inner.
        # Always on -- zero overhead when no streaming session is open (the
        # try_* entries short-circuit on non-streaming reqs / real TreeNodes).
        # Dispatch methods below pre-check conditions so the session's
        # internal fall-through to self.inner.xxx never fires -- no recursion.
        self.session = StreamingSession(inner=self)

        self.tp_group = params.tp_cache_group
        self.attn_cp_group = params.attn_cp_cache_group
        self.attn_tp_group = params.attn_tp_cache_group
        self.pp_group = params.pp_cache_group
        self.tp_world_size = (
            1
            if self.tp_group is None
            else torch.distributed.get_world_size(group=self.tp_group)
        )
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.work_list: list[torch.distributed.Work] = []

        # HiCache D↔H defaults (overridden by init_hicache)
        self.cache_controller: Optional[HybridCacheController] = None
        self.host_pool_group = None  # set by attach_hybrid_pool_to_unified_cache
        # Owns the storage backend lifecycle; built by init_hicache.
        self._storage_attachment: Optional[StorageAttachment] = None
        # kv-p2p owner-side servicer; set by attach_peer_kv_servicer.
        self.peer_kv_servicer = None
        self.gpu_export_manager = None
        self._gpu_export_has_work = False
        # SGLANG_HICACHE_SKIP_IDLE_SYNC (check_hicache_events, wants_step_sync):
        # the TP source rank's decision for the current step (None = sync),
        # whether no rank had an op in flight at the last sync, and the steps
        # skipped since that sync.
        self._idle_sync_enabled = envs.SGLANG_HICACHE_SKIP_IDLE_SYNC.get()
        self._idle_sync_interval = max(1, envs.SGLANG_HICACHE_IDLE_SYNC_INTERVAL.get())
        self._step_sync_flag: Optional[bool] = None
        self._last_step_sync_flag: Optional[bool] = None
        self._idle_sync_all_idle = False
        self._idle_sync_skipped = 0
        self.idle_sync_skipped_steps = 0
        # SGLANG_HICACHE_PIPELINED_SYNC (check_hicache_events): enabled by
        # init_hicache when the dedicated group can be created; the started
        # step-sync snapshot (PendingStepSync); the generation that drops it
        # after an out-of-band drain; the paused flag of the synchronous drain.
        self._pipelined_sync_active = False
        self._pipelined_sync_pending = None
        self._pipelined_sync_generation = 0
        self._pipelined_sync_paused = False
        self.prefetch_stop_policy = "best_effort"
        self.prefetch_threshold = 256
        self.prefetch_timeout_base = 1.0
        self.prefetch_timeout_per_page = 0.25
        self.hicache_storage_pass_prefix_keys = False
        # Buffer-only host memory mode (host RAM as transient GPU↔storage
        # staging, not an L2 tier); resolved in init_hicache, which also
        # constructs the pipeline collaborator (None = cache mode).
        self.host_memory_mode = "cache"
        self.buffer_pipeline: Optional[BufferModePipeline] = None
        # Write-side dedupe: beliefs about what storage already holds, so
        # re-inserts of hot prefixes skip the redundant backup.
        self.storage_existence_cache = StorageExistenceCache()
        # Cumulative prefetch-outcome counters, exported through the
        # log_storage_metrics flow.
        self._prefetch_outcome_stats: dict[str, float] = {
            "attempts": 0,
            "issued": 0,
            "declined_too_short": 0,
            "declined_rate_limited": 0,
            "declined_anchor_lost": 0,
            "declined_device_covered": 0,
            "revoked_insufficient": 0,
            "revoked_full_miss": 0,
            "l3_demand_requests": 0,
            "l3_miss_tokens": 0,
            "l1l2_miss_tokens": 0,
            "l3_demand_total_tokens": 0,
            "l3_sum_rate_all": 0.0,
            "l3_sum_rate_main_weighted": 0.0,
        }

        self.reset()
        logger.info(
            f"Init Unified Radix Cache. Components: {self.tree_components}. "
            f"Tree Core: {type(self.tree_core).__name__}"
        )

    def _resolve_active_components(
        self,
        active_components: Optional[Sequence[ComponentType]],
        for_match: bool = False,
    ) -> tuple[TreeComponent, ...]:
        """Resolve a per-operation component mask in registration order.

        ``for_match`` marks read-side operations (match, load-back), which
        window SWA residency allows on a FULL-only mask."""
        if active_components is None:
            return self._components_tuple
        if not (
            getattr(self.tree_core, "supports_partial_component_cache", False)
            or (
                for_match
                and getattr(self.tree_core, "supports_partial_component_match", False)
            )
        ):
            raise ValueError(
                "partial-component cache operations require a tree core that "
                "supports active_components"
            )

        requested = tuple(active_components)
        requested_set = set(requested)
        if len(requested_set) != len(requested):
            raise ValueError(
                f"active_components contains duplicates: {active_components!r}"
            )
        unknown = requested_set.difference(self.tree_components)
        if unknown:
            names = ", ".join(sorted(component.name for component in unknown))
            raise ValueError(f"active_components not registered on this cache: {names}")
        if BASE_COMPONENT_TYPE not in requested_set:
            raise ValueError("active_components must include ComponentType.FULL")
        return tuple(
            component
            for component in self._components_tuple
            if component.component_type in requested_set
        )

    def _all_reduce_attn_groups(self, tensor: torch.Tensor, op):
        # The HiCache lockstep all-reduces run in admission and control
        # paths, never inside the symmetric launch thread's result work.
        collective_guard.check("HiCache lockstep all-reduce")
        reduced = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.all_reduce(tensor, op=op, group=group)
                reduced = True
        if not reduced and self.tp_world_size > 1:
            torch.distributed.all_reduce(tensor, op=op, group=self.tp_group)

    def _barrier_attn_groups(self):
        waited = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.barrier(group=group)
                waited = True
        if not waited and self.tp_world_size > 1:
            torch.distributed.barrier(group=self.tp_group)

    def _drain_async_work(self):
        """
        Block until all outstanding async sends are consumed, then clear.

        Called at the start of each event round, so work_list holds the sends
        accumulated since the last round. This bounds it and applies
        backpressure when a downstream PP rank lags. Scheduler thread only.
        """
        for work in self.work_list:
            work.wait()
        self.work_list.clear()

    def _all_reduce(self, data: torch.Tensor, tp_reduce_op: torch.distributed.ReduceOp):
        """
        Synchronize data across all TP and PP ranks.

        In particular, "tp_reduce_op" is performed on all TP ranks of the first PP rank,
        and then the result is propagated to all following PP ranks.

        Must be called in the scheduler thread.
        """
        if self.pp_rank == 0:
            self._all_reduce_attn_groups(data, tp_reduce_op)
        self._pp_sync(data)

    def _all_reduce_all_cache_ranks(
        self, data: torch.Tensor, op: torch.distributed.ReduceOp
    ) -> None:
        """Reduce local storage state across every TP/CP cache rank."""
        if self.pp_size > 1:
            raise RuntimeError(
                "Synchronous storage clear is unsupported with pipeline parallelism."
            )
        self._all_reduce_attn_groups(data, op)

    def _pp_sync(self, data: torch.Tensor) -> None:
        """
        Synchronize data across the PP pipeline, where PPn (n>0) will receive PP0's data.
        """
        if self.pp_size <= 1 or self.pp_group is None:
            return
        if self.pp_rank > 0:
            torch.distributed.recv(
                data,
                group_src=self.pp_rank - 1,
                group=self.pp_group,
                tag=P2PTag.HIRADIX_PP_SYNC,
            )
        if self.pp_rank + 1 < self.pp_size:
            copy_of_data = data.clone()
            send_work = torch.distributed.isend(
                copy_of_data,
                group_dst=self.pp_rank + 1,
                group=self.pp_group,
                tag=P2PTag.HIRADIX_PP_SYNC,
            )
            self.work_list.append(send_work)

    # --peer-kv-demote-source (mem_cache/peer_kv_demote.py) -----------------

    _PEER_DEMOTE_LOG_S = 60.0

    def enable_peer_kv_demote(self, resolver: bool) -> None:
        """Every TP rank applies the broadcast flags; the resolver (the
        request-broadcast source rank, with the servicer) also turns its
        servicer's complete RELEASEs into node ids. The hardening flags
        (peer_kv_demote.py) are read here, the same on every rank."""
        if self.enable_kv_session_retention:
            # The retention host eviction key has no peer_demoted level, so
            # demoted copies lose their first place on the host tier there.
            logger.warning(
                "PeerKV demote with KV session retention: host eviction "
                "follows the retention order and ignores the demoted-first "
                "order; an untagged worker no longer evicts like a "
                "retention-off control"
            )
        lock_aware = bool(envs.SGLANG_PEER_KV_DEMOTE_LOCK_AWARE.get())
        counters = bool(envs.SGLANG_PEER_KV_DEMOTE_COUNTERS.get())
        kw = {}
        if lock_aware or counters:
            kw.update(lock_aware=lock_aware, counters=counters)
        if envs.SGLANG_PEER_KV_DEMOTE_GPU_ORDER.get():
            kw["gpu_order"] = True  # SGLANG_PEER_KV_DEMOTE_GPU_ORDER
        scope = scope_mode(envs.SGLANG_PEER_KV_DEMOTE_SCOPE.get())
        if scope != "off":
            kw["scope"] = scope  # SGLANG_PEER_KV_DEMOTE_SCOPE
            if resolver:
                logger.info(
                    "PeerKV demote: SCOPE=%s (%s; the queued set needs the "
                    "scheduler's queue hook, else E2/E1 see locks only)",
                    scope,
                    (
                        "the v3 walk is counted, today's walk applies"
                        if scope == "shadow"
                        else "the v3 walk applies"
                    ),
                )
        self.tree_core.enable_peer_demote(resolver, **kw)
        self._peer_demote_last_log = (time.monotonic(), None)

    @timed_phase("peer_demote_drain")
    def collect_peer_kv_demote(self) -> Optional[PeerKVDemoteReq]:
        """Resolver rank, scheduler thread, before the request broadcast."""
        core = self.tree_core
        servicer = self.peer_kv_servicer
        if not core.peer_demote_resolver or servicer is None:
            return None
        clock = getattr(core, "peer_demote_clock", None)
        if clock is not None:
            # COUNTERS: one tick -> time sample per pass.
            clock.sample(peek_time_counter(), time.monotonic())
        drain = getattr(servicer, "drain_demote_outbox", None)
        releases = drain() if drain is not None else []
        req = None
        if releases:
            if getattr(core, "peer_demote_scope", "off") != "off":
                # SCOPE: the queued set, built on this rank only when
                # there is a mark to resolve (its cost: scope.build_us).
                node_ids = core.select_peer_demote_nodes(
                    releases, scope=self._peer_scope_context()
                )
            else:
                node_ids = core.select_peer_demote_nodes(releases)
            if node_ids:
                req = PeerKVDemoteReq(node_ids=node_ids, releases=len(releases))
        hardening = (
            core.peer_demote_lock_aware
            or core.peer_demote_counters
            or getattr(core, "peer_demote_scope", "off") != "off"
        )
        if core.peer_demote_counters:
            xst = core.peer_demote_xstats
            for name in (
                "outbox_drops",
                "touch_outbox_drops",
                "commit_pending",
                "commit_ok",
                "commit_expired",
                "commit_unknown",
            ):
                setattr(xst, name, int(getattr(servicer, "demote_" + name, 0)))
            lag = getattr(servicer, "demote_commit_lag", None)
            if lag is not None:
                xst.commit_lag = list(lag)
                xst.commit_lag_max_ms = int(
                    getattr(servicer, "demote_commit_lag_max_ms", 0)
                )
            for name in xst.NOTICE_FIELDS:  # NOTICE
                if name != "grace_access":
                    value = getattr(servicer, "notice_" + name, 0)
                    setattr(xst, "notice_" + name, int(value))
            xst.ignored_role = int(getattr(servicer, "demote_ignored_role", 0))
        now = time.monotonic()
        last_t, last_key = self._peer_demote_last_log
        if now - last_t >= self._PEER_DEMOTE_LOG_S:
            key = core.peer_demote_stats.key()
            if hardening:
                key = (key, core.peer_demote_xstats.key())
            if key != last_key:
                nodes, pages = core.peer_demote_outstanding()
                line = core.peer_demote_stats.line(nodes, pages)
                if hardening:
                    extra = core.peer_demote_xstats.line(
                        core.peer_demote_lock_aware, core.peer_demote_counters
                    )
                    line = f"{line} {extra}"
                records = getattr(self, "_peer_demote_records", None)
                if records is not None:  # RECORDS (C6b): written/dropped
                    line = f"{line} rec={records.written}/{records.dropped}"
                logger.info("PeerKV demote: %s", line)
            self._peer_demote_last_log = (now, key)
        return req

    def _peer_scope_context(self) -> ScopeContext:
        """SCOPE, resolver rank, scheduler thread: the matched-end
        node ids of the queued requests (``peer_scope_queued_nodes``, set by
        the scheduler; at most SCOPE_QUEUED_MAX requests) and those nodes with
        their ancestors. Timed into scope.build_us (sum) and
        scope.build_us_max."""
        core = self.tree_core
        t0 = time.perf_counter()
        ends: set = set()
        paths: set = set()
        truncated = False
        source = getattr(self, "peer_scope_queued_nodes", None)
        if source is not None:
            arena = core._node_arena
            root = core.root_node
            for count, node_id in enumerate(source()):
                if count >= SCOPE_QUEUED_MAX:
                    truncated = True
                    break
                node = arena.get(node_id) if node_id is not None else None
                if node is None or node.deleted or node is root:
                    continue
                ends.add(node.id)
                while node is not None and node is not root and node.id not in paths:
                    paths.add(node.id)
                    node = node.parent
        ctx = ScopeContext(
            apply=core.peer_demote_scope == "on",
            ends=frozenset(ends),
            paths=frozenset(paths),
            truncated=truncated,
        )
        us = int((time.perf_counter() - t0) * 1e6)
        xst = core.peer_demote_xstats
        xst.bump("scope.drains")
        xst.bump("scope.build_us", us)
        xst.trig["scope.build_us_max"] = max(xst.trig.get("scope.build_us_max", 0), us)
        xst.bump("scope.queued_ends", len(ends))
        if truncated:
            xst.bump("scope.queued_truncated")
        return ctx

    def apply_peer_kv_demote(self, req: PeerKVDemoteReq) -> int:
        """Every rank, first in process_input_requests (same point on all)."""
        return self.tree_core.apply_peer_demote(req.node_ids)

    def _open_peer_demote_trace(self, servicer) -> None:
        """SGLANG_PEER_KV_DEMOTE_COUNTERS, resolver rank: one JSON line per
        flagged eviction in the servicer trace dir (none without one). The
        owner (P) is this host and the servicer port."""
        trace_dir = getattr(servicer, "trace_dir", None)
        if not trace_dir:
            return
        owner = f"{socket.gethostname()}:{getattr(servicer, '_bind_port', '')}"
        path = os.path.join(
            trace_dir,
            f"peer_demote_evict.{owner.replace(':', '_')}.pid{os.getpid()}.jsonl",
        )
        os.makedirs(trace_dir, exist_ok=True)
        out = open(path, "a", buffering=1)

        def sink(record: dict) -> None:
            record["owner"] = owner
            out.write(json.dumps(record, sort_keys=True) + "\n")

        self.tree_core.peer_demote_trace_sink = sink
        logger.info("PeerKV demote: flagged-eviction trace %s", path)
        if envs.SGLANG_PEER_KV_DEMOTE_RECORDS.get():
            # RECORDS, R2: one record per host eviction, written by a
            # thread (put never blocks; a full queue drops, counted).
            records = RecordWriter(
                os.path.join(
                    trace_dir,
                    f"peer_host_evict.{owner.replace(':', '_')}"
                    f".pid{os.getpid()}.jsonl",
                )
            )
            self._peer_demote_records = records

            def record_sink(record: dict) -> None:
                record["owner"] = owner
                records.put(record)

            self.tree_core.peer_demote_record_sink = record_sink
            logger.info("PeerKV demote: host-eviction records %s", records.path)

    def _peer_demote_after_commit(
        self,
        req_id: str,
        completed_tokens: int,
        attempt_id: Optional[str] = None,
        landed_node_id: Optional[NodeId] = None,
    ) -> None:
        """SGLANG_PEER_KV_DEMOTE_AFTER_COMMIT, reader side. The commit is
        insert_host at the prefetch completion (_handle_prefetch_result, host
        memory mode "cache"): the pulled pages of ``req_id`` are in this tree
        now; let the owners of the complete leases of this attempt demote
        their copies (DEMOTE_OK). Every other end of a pull sends nothing more
        (_peer_demote_discard)."""
        if not envs.SGLANG_PEER_KV_DEMOTE_AFTER_COMMIT.get():
            return
        backend = getattr(self.cache_controller, "storage_backend", None)
        commit = getattr(backend, "on_prefetch_commit", None)
        if commit is not None:
            commit(req_id, completed_tokens // self.page_size, attempt_id=attempt_id)
            self._maybe_log_demote_commit(backend)
        note = getattr(self.tree_core, "note_peer_landing", None)
        if landed_node_id is not None and note is not None:
            # COUNTERS: the landing window.
            note(landed_node_id)

    def _peer_demote_discard(self, req_id: str, cause: str) -> None:
        """SGLANG_PEER_KV_DEMOTE_AFTER_COMMIT, reader side: the pull of
        ``req_id`` ended without a commit (abort after its reads, hybrid
        discard, dropped insert, buffer mode). No DEMOTE_OK, so no demote;
        the reader counts it by cause."""
        if not envs.SGLANG_PEER_KV_DEMOTE_AFTER_COMMIT.get():
            return
        backend = getattr(self.cache_controller, "storage_backend", None)
        discard = getattr(backend, "on_prefetch_discard", None)
        if discard is not None:
            discard(req_id, cause)
            self._maybe_log_demote_commit(backend)

    def _peer_trim_short_alloc(self, operation, alloc_len: int) -> None:
        backend = getattr(self.cache_controller, "storage_backend", None)
        reconcile = getattr(backend, "on_hit_reconciled", None)
        if reconcile is None:
            return
        options = {}
        if getattr(backend, "supports_complete_endpoints", False):
            options["attempt_id"] = operation.prefetch_attempt_id
        try:
            reconcile(operation.request_id, alloc_len // self.page_size, **options)
        except Exception:
            logger.warning(
                "storage backend on_hit_reconciled (short alloc) failed",
                exc_info=True,
            )

    def _peer_notice_backend(self):
        """The reader backend when it keeps NOTICE records
        (SGLANG_PEER_KV_DEMOTE_NOTICE or COUNTERS, with AFTER_COMMIT, on TP0
        only); else None."""
        backend = getattr(
            getattr(self, "cache_controller", None), "storage_backend", None
        )
        return backend if getattr(backend, "notice_tracking", False) else None

    def peer_notice_prefill_end(self, req) -> None:
        """NOTICE, reader side, right after maybe_cache_unfinished_req at
        the final prefill end (every rank calls it; only TP0 keeps records).
        The backend queues one NOTICE when the hinted owner confirmed more
        pages than this tree committed from it and the request inserted its
        KV with write-through (the insert gate); else it counts why not."""
        backend = self._peer_notice_backend()
        if backend is not None:
            backend.on_prefill_end(req.rid, notice_insert_cause(req))
            self._maybe_log_demote_commit(backend)

    _PEER_DEMOTE_COMMIT_LOG_S = 60.0

    def _maybe_log_demote_commit(self, backend) -> None:
        """At most once per 60 s, on change: the reader's commit-gated demote
        counters (rank-local, no collective)."""
        line_fn = getattr(backend, "commit_stats_line", None)
        if line_fn is None:
            return
        now = time.monotonic()
        last_t, last_line = getattr(self, "_peer_demote_commit_log", (None, None))
        if last_t is not None and now - last_t < self._PEER_DEMOTE_COMMIT_LOG_S:
            return
        line = line_fn()
        self._peer_demote_commit_log = (now, line)
        if line != last_line:
            logger.info("PeerKV demote commit (reader): %s", line)

    # SGLANG_PEER_KV_TOUCH_DEFERRED_PULLS (mem_cache/peer_kv_touch.py) --------

    _PEER_TOUCH_LOG_S = 60.0

    def enable_peer_kv_touch(self, resolver: bool) -> None:
        """Every TP rank applies the broadcast refreshes; the resolver (the
        request-broadcast source rank, with the servicer) also turns its
        servicer's queued TOUCHes into node ids."""
        self.tree_core.enable_peer_touch(resolver)
        self._peer_touch_last_log = (time.monotonic(), None)

    @timed_phase("peer_touch_drain")
    def collect_peer_kv_touch(self) -> Optional[PeerKVTouchReq]:
        """Resolver rank, scheduler thread, before the request broadcast."""
        core = self.tree_core
        servicer = self.peer_kv_servicer
        if not core.peer_touch_resolver or servicer is None:
            return None
        drain = getattr(servicer, "drain_touch_outbox", None)
        # Only this thread advances the counter: ``tick`` is the next tick at
        # the drain. The drain reserves one tick per TOUCH from it, and a
        # LOOKUP after the drain takes its lookup tick after them.
        tick = peek_time_counter()
        touches = drain(tick=tick) if drain is not None else []
        req = None
        if touches:
            # Also without a resolved node: every rank then moves its counter
            # past the reserved ticks.
            req = PeerKVTouchReq(
                node_ids=core.select_peer_touch_nodes(touches),
                touches=len(touches),
                tick=float(tick),
            )
        st = core.peer_touch_stats
        st.touch_after_lookup = int(getattr(servicer, "touch_after_lookup", 0))
        st.touch_dropped_by_lookup = int(
            getattr(servicer, "touch_dropped_by_lookup", 0)
        )
        now = time.monotonic()
        last_t, last_key = self._peer_touch_last_log
        if now - last_t >= self._PEER_TOUCH_LOG_S:
            key = core.peer_touch_stats.key()
            if key != last_key:
                logger.info("PeerKV touch: %s", core.peer_touch_stats.line())
            self._peer_touch_last_log = (now, key)
        return req

    def apply_peer_kv_touch(self, req: PeerKVTouchReq) -> int:
        """Every rank, in process_input_requests (same point on all)."""
        return self.tree_core.apply_peer_touch(req.node_ids, req.tick, req.touches)

    def attach_peer_kv_servicer(self, servicer) -> None:
        """Attach the kv-p2p owner-side servicer (see mem_cache/peer_kv_servicer.py)."""
        self.peer_kv_servicer = servicer
        self.tree_core.peer_kv_servicer = servicer
        self._peer_cache_has_work = False
        self._host_returns = {}
        self._host_return_tokens = 0
        self._host_return_imports = {}
        self._host_returns_dirty = False
        # SGLANG_PEER_KV_DEMOTE_COUNTERS only; getattr: with the flag off no new
        # attribute of the tree is read (partial stand-ins in tests).
        core = self.tree_core
        if getattr(core, "peer_demote_counters", False) and getattr(
            core, "peer_demote_resolver", False
        ):
            self._open_peer_demote_trace(servicer)
        self._peer_kv_tp_participant = True
        # One boot identity for all cache ranks. FINAL carries this identity
        # plus a reset generation to D, so a late return cannot refill a flushed
        # or restarted P. This agreement happens only at attachment/startup.
        nonce = uuid.uuid4().hex if self._peer_control_leader() else None
        for group in self._peer_control_groups():
            values = [nonce]
            source = torch.distributed.get_global_rank(group, 0)
            torch.distributed.broadcast_object_list(values, src=source, group=group)
            nonce = values[0]
        self._host_return_boot = nonce
        self._host_return_generation = 0
        self._tp_digest_mode_cached = None
        # Fail fast if the host-pool geometry the PageSlotAllocator relies on
        # (size, reserve) is not identical on every TP rank.
        if self.host_pool_group is not None and self.pp_size == 1:
            vals = []
            for entry in self.host_pool_group.allocator_entries():
                slots = getattr(entry.host_pool, "slot_allocator", None)
                if slots is None:
                    continue
                vals.extend([slots.num_slots, slots.reserve_pages])
            if vals and self._has_tp_peers():
                t = torch.tensor(vals + [-v for v in vals], dtype=torch.int64)
                self._all_reduce_attn_groups(t, torch.distributed.ReduceOp.MIN)
                lo = t[: len(vals)].tolist()
                hi = [-v for v in t[len(vals) :].tolist()]
                if lo != hi:
                    raise RuntimeError(
                        "peer-KV host pool geometry differs across TP ranks: "
                        f"min={lo} max={hi} (size/reserve per allocator entry)"
                    )

    @property
    def host_return_epoch(self) -> Optional[str]:
        """Current TP-shared cache lifetime; absent on engines without peer serving."""
        boot = getattr(self, "_host_return_boot", None)
        if boot is None:
            return None
        return f"{boot}:{self._host_return_generation}"

    def _peer_control_groups(self):
        """Return CPU agreement groups for this cache, in the normal sync order."""
        # getattr: partial cache stand-ins in tests attach a servicer without
        # the parallel groups; a real cache always has all three.
        groups = []
        for group in (
            getattr(self, "attn_cp_group", None),
            getattr(self, "attn_tp_group", None),
        ):
            if group is not None and torch.distributed.get_world_size(group) > 1:
                if group not in groups:
                    groups.append(group)
        if not groups and getattr(self, "tp_world_size", 1) > 1:
            groups.append(self.tp_group)
        return groups

    def _peer_control_leader(self) -> bool:
        """True only for the CPU-control source; no GPU synchronization is issued."""
        return all(
            torch.distributed.get_rank(g) == 0 for g in self._peer_control_groups()
        )

    def _host_return_expirations(self) -> list[dict]:
        """Return due operation IDs on the leader; decisions are replicated later."""
        if not (
            getattr(self, "_host_returns", None)
            or getattr(self, "_host_return_imports", None)
        ):
            return []
        if not self._peer_control_leader():
            return []
        now = time.monotonic()
        expired = [
            {"type": "host_return_done", "operation_id": oid, "status": "timeout"}
            for oid, pending in getattr(self, "_host_returns", {}).items()
            if pending.deadline <= now
        ]
        expired.extend(
            {"type": "host_return_timeout", "operation_id": oid}
            for oid, pending in getattr(self, "_host_return_imports", {}).items()
            if pending[1] <= now and not pending[2]
        )
        return expired

    def _peer_control_pending(self) -> bool:
        """Local queue/deadline observation, used only as input to the TP reduce."""
        if not getattr(self, "_peer_kv_tp_participant", False):
            return False
        servicer = getattr(self, "peer_kv_servicer", None)
        return (servicer is not None and servicer.has_cache_events()) or bool(
            self._host_return_expirations()
        )

    def _drain_peer_cache_events(self) -> None:
        """Apply a bounded, identical event batch on every rank after agreement.

        Inputs are controller-issued CPU transfers and their completions.
        Network threads never mutate the cache. Object exchange occurs only
        when the existing HiCache count reduction says there is actual work.
        """
        servicer = self.peer_kv_servicer
        events = servicer.drain_cache_events() if servicer is not None else []
        events.extend(self._host_return_expirations())
        for group in self._peer_control_groups():
            gathered = [None] * torch.distributed.get_world_size(group)
            torch.distributed.all_gather_object(gathered, events, group=group)
            events = [event for batch in gathered for event in batch]
        for event in events:
            kind = event.get("type")
            if kind == "host_return_request":
                self._begin_host_return_import(event)
            elif kind == "host_return_done":
                if event["operation_id"] in getattr(self, "_host_returns", {}):
                    self._host_return_stat("d_done_" + str(event.get("status")))
                self._finish_host_return(event["operation_id"])
            elif kind == "host_return_timeout":
                oid = event["operation_id"]
                pending = self._host_return_imports.get(oid)
                if pending is not None and not pending[2]:
                    self._host_return_imports[oid] = (*pending[:2], True)
                    info = self.ongoing_prefetch.get(oid)
                    if info is not None:
                        # Never free an RDMA destination until its IO ACK. The
                        # existing prefetch path owns safe cancellation/drain.
                        self.cache_controller.terminate_prefetch(info[3])
        self._peer_cache_has_work = False

    def _host_return_stat(self, name: str, n: int = 1) -> None:
        """Count one host-return outcome (observability only; never read by a
        decision). D side: d_*; P side: p_*. Logged by _log_host_return_stats."""
        stats = getattr(self, "host_return_stats", None)
        if stats is None:
            stats = self.host_return_stats = {}
        stats[name] = stats.get(name, 0) + n

    def _log_host_return_stats(self) -> None:
        """One cumulative HOST_RETURN line per 60 s on the control leader, only
        after a counter changed. Logging only; no rank agreement."""
        stats = getattr(self, "host_return_stats", None)
        if not stats or not self._peer_control_leader():
            return
        now = time.monotonic()
        last = getattr(self, "_host_return_stats_log", None)
        if last is not None and (now - last[0] < 60 or last[1] == stats):
            return
        self._host_return_stats_log = (now, dict(stats))
        logger.info(
            "HOST_RETURN stats (cumulative): %s; pending holds %d (%d tokens), "
            "pending imports %d",
            json.dumps(stats, sort_keys=True),
            len(getattr(self, "_host_returns", {})),
            getattr(self, "_host_return_tokens", 0),
            len(getattr(self, "_host_return_imports", {})),
        )

    def _queue_host_return(self, req, result: InsertResult, key: RadixKey) -> None:
        """Retain existing D host pages for an optional router return command.

        Input is the exact successful finished-insert key, already clipped to
        committed EAGLE state. Output is a bounded CPU-only hold; replies and
        the GPU request lifecycle never wait for this return.
        """
        mark = getattr(req, "import_mark", None)
        command = getattr(mark, "host_return", None)
        if command is None or key is None:
            return
        from decoupled_engine.srt.managers.schedule_batch import FINISH_ABORT

        if req.finished_reason is None or isinstance(req.finished_reason, FINISH_ABORT):
            self._host_return_stat("d_skip_unfinished_or_aborted")
            return
        owner_epoch = getattr(req, "host_return_owner_epoch", None)
        if not isinstance(owner_epoch, str) or not owner_epoch:
            self._host_return_stat("d_skip_no_owner_epoch")
            return  # An older P cannot fence notifications across its resets.
        pending = getattr(self, "_host_returns", None)
        prompt_tokens = len(req.origin_input_ids)
        prompt_floor = (
            max(0, prompt_tokens - int(key.is_bigram))
            // self.page_size
            * self.page_size
        )
        if (
            pending is None
            or len(pending) >= _MAX_HOST_RETURNS
            or len(key.token_ids) > 1024 * 1024
            or getattr(self, "_host_return_tokens", 0) + len(key.token_ids)
            > _MAX_HOST_RETURN_TOKENS
            or result.last_device_node is None
            or len(key) <= prompt_floor
            or len(key) - prompt_floor > command.max_tokens
        ):
            if pending is not None and len(key) <= prompt_floor:
                self._host_return_stat("d_skip_no_new_page")
            elif pending is not None and len(key) - prompt_floor > command.max_tokens:
                self._host_return_stat("d_skip_over_max_tokens")
            else:
                self._host_return_stat("d_skip_limit")
            return
        oid = f"host-return:{mark.transfer_id:x}:{mark.attempt_id:x}"
        if oid in pending:
            return
        node = self.tree_core.node_by_id(result.last_device_node)
        if not node.backuped:
            # Host allocation can fail under pressure. Do not introduce a
            # second copy or retain GPU pages to turn a cache hint into work.
            self._host_return_stat("d_skip_not_backed_up")
            return
        lock = self.inc_host_lock_ref(node.id).to_dec_params()
        pending[oid] = _PendingHostReturn(
            node.id,
            lock,
            key,
            prompt_tokens,
            command,
            time.monotonic() + command.timeout_ms / 1000,
            owner_epoch,
            # Snapshot the original request, not a later donor session version.
            kv_session=(
                dict(req.kv_session)
                if getattr(req, "kv_session", None) is not None
                else None
            ),
        )
        self._host_return_tokens = getattr(self, "_host_return_tokens", 0) + len(
            key.token_ids
        )
        self._host_returns_dirty = True
        self._host_return_stat("d_queued")
        self._host_return_stat("d_queued_suffix_tokens", len(key) - prompt_floor)

    def _advance_host_returns(self) -> None:
        """Publish CPU-ready returns and retire imports using existing IO results."""
        if getattr(self, "_host_returns_dirty", False):
            self._host_returns_dirty = False
            for oid, pending in self._host_returns.items():
                if pending.notified:
                    continue
                node = self.tree_core.node_by_id(pending.node_id)
                ready = True
                while node is not self.tree_core.root_node:
                    if not node.backuped or node.write_through_pending_id is not None:
                        ready = False
                        break
                    node = node.parent
                if not ready:
                    continue
                pending.notified = True
                self._host_return_stat("d_notified")
                if self._peer_control_leader():
                    command = pending.command
                    remaining_ms = int((pending.deadline - time.monotonic()) * 1000)
                    if remaining_ms <= 0:
                        # The next agreed expiration releases every rank's hold.
                        # Do not renew the return budget after a slow D2H ACK.
                        continue
                    self.peer_kv_servicer.notify_peer(
                        command.destination_servicer_addr,
                        {
                            "op": "host_return_request",
                            "operation_id": oid,
                            "token_ids": pending.key.token_ids,
                            "owner_cache_epoch": pending.owner_cache_epoch,
                            "prompt_tokens": pending.prompt_tokens,
                            "cache_salt": pending.key.cache_salt,
                            "extra_key": pending.key.extra_key,
                            "page_size": self.page_size,
                            "is_bigram": self.tree_core.is_eagle,
                            "source_servicer_addr": command.source_servicer_addr,
                            "timeout_ms": min(command.timeout_ms, remaining_ms),
                            "max_tokens": command.max_tokens,
                            "kv_session": pending.kv_session,
                        },
                    )
        for oid, pending in list(getattr(self, "_host_return_imports", {}).items()):
            if oid in self.ongoing_prefetch:
                continue
            loaded = self.pop_prefetch_loaded_tokens(oid)
            self._host_return_imports.pop(oid)
            if loaded:
                self._host_return_stat("p_loaded_tokens", int(loaded))
            self._notify_host_return_done(
                oid, pending[0], "complete" if loaded else "miss"
            )
        self._log_host_return_stats()

    def _notify_host_return_done(self, oid: str, source: str, status: str) -> None:
        """Send a CPU-only completion; source releases its bounded host hold."""
        self._host_return_stat("p_" + status)
        if self._peer_control_leader():
            self.peer_kv_servicer.notify_peer(
                source,
                {
                    "op": "host_return_done",
                    "operation_id": oid,
                    "status": status,
                },
            )

    def _finish_host_return(self, oid: str) -> None:
        """Release a source CPU hold exactly once after completion or timeout."""
        pending = getattr(self, "_host_returns", {}).pop(oid, None)
        if pending is not None:
            self._host_return_tokens -= len(pending.key.token_ids)
            self.dec_host_lock_ref(pending.node_id, pending.lock_params)

    def _begin_host_return_import(self, event: dict) -> None:
        """Execute a deferred CPU pull using the existing storage-prefetch path.

        This changes no request placement and never issues H2D. A subsequent
        real request reuses the normal asynchronous load/prefill machinery.
        Only a missing generated suffix may be pulled; a lost prompt anchor
        declines the hint rather than refetching an entire long prompt.
        """
        from decoupled_engine.srt.disaggregation.pull.api import _host_return_endpoint
        from decoupled_engine.srt.mem_cache.storage.peer_kv.peer_kv_backend import PeerKVBackend

        oid = event["operation_id"]
        self._host_return_stat("p_request")
        if (
            not self.host_return_epoch
            or event.get("owner_cache_epoch") != self.host_return_epoch
        ):
            self._host_return_stat("p_drop_epoch")
            return
        source = event.get("source_servicer_addr", "")
        try:
            _host_return_endpoint(source)
        except ValueError:
            return
        if oid in self._host_return_imports:
            return
        tokens = event.get("token_ids")
        prompt = event.get("prompt_tokens")
        timeout = event.get("timeout_ms")
        limit = event.get("max_tokens")
        valid = (
            self.enable_storage
            and self.host_memory_mode == "cache"
            and self.cache_controller is not None
            and isinstance(self.cache_controller.storage_backend, PeerKVBackend)
            and len(self._host_return_imports) < _MAX_HOST_RETURNS
            and oid not in self.ongoing_prefetch
            and isinstance(source, str)
            and bool(source)
            and isinstance(tokens, list)
            and 0 < len(tokens) <= 1024 * 1024
            and all(type(t) is int and 0 <= t < (1 << 31) for t in tokens)
            and type(prompt) is int
            and 0 < prompt <= len(tokens)
            and type(timeout) is int
            and 0 < timeout <= 60_000
            and type(limit) is int
            and 0 < limit <= 1_048_576
            and event.get("page_size") == self.page_size
            and event.get("is_bigram") is self.tree_core.is_eagle
            and all(
                event.get(n) is None or isinstance(event[n], str)
                for n in ("cache_salt", "extra_key")
            )
        )
        if not valid:
            self._host_return_stat("p_drop_invalid")
            return
        key = RadixKey(
            array("q", tokens),
            extra_key=event.get("extra_key"),
            cache_salt=event.get("cache_salt"),
            is_bigram=self.tree_core.is_eagle,
        ).page_aligned(self.page_size)
        match = self.match_prefix(MatchPrefixParams(key=key))
        anchor = self.tree_core.node_by_id(match.last_host_node)
        matched, node = 0, anchor
        while node is not self.tree_core.root_node:
            if not node.backuped or node.write_through_pending_id is not None:
                self._notify_host_return_done(oid, source, "anchor_pending")
                return
            matched += len(node.key)
            node = node.parent
        prompt_floor = (
            max(0, prompt - int(key.is_bigram)) // self.page_size * self.page_size
        )
        if matched < prompt_floor or len(key) - matched > limit:
            self._notify_host_return_done(oid, source, "anchor_missing")
            return
        registration = None
        if self.kv_sessions is not None and event.get("kv_session") is not None:
            try:
                registration = self.kv_sessions.prepare_host_return(
                    event["kv_session"], key, prompt
                )
            except ValueError:
                self._notify_host_return_done(oid, source, "invalid_session")
                return
            if registration is None:
                self._notify_host_return_done(oid, source, "session_not_current")
                return
            # Locating the exact session range can split the matched radix edge.
            # Its existing endpoint still names the same complete prefix.
        if matched >= len(key):
            self._notify_host_return_done(oid, source, "already_cached")
            return
        self._host_return_imports[oid] = (
            source,
            time.monotonic() + timeout / 1000,
            False,
        )
        self._host_return_stat("p_import_started")
        self._host_return_stat("p_import_requested_tokens", len(key) - matched)
        self.prefetch_from_storage(
            oid,
            anchor.id,
            key[matched:].token_ids,
            last_hash=self.tree_core.get_last_hash_value(anchor.id),
            prefix_keys=self.tree_core.get_prefix_hash_values(anchor.id),
            matched_prefix_tokens=key.token_ids[:matched],
            extra_key=key.extra_key,
            cache_salt=key.cache_salt,
            kv_pull_hint={
                "source_servicer_addr": source,
                "expected_hit_tokens": len(key),
            },
            kv_session_registration=registration,
        )

    def _clear_host_returns(self) -> None:
        """Invalidate old directives, cancel imports and release bounded CPU holds."""
        self._host_return_generation = getattr(self, "_host_return_generation", 0) + 1
        for oid in list(getattr(self, "_host_return_imports", {})):
            self.release_aborted_request(oid)
        getattr(self, "_host_return_imports", {}).clear()
        for oid in list(getattr(self, "_host_returns", {})):
            self._finish_host_return(oid)
        self._host_return_tokens = 0
        self._host_returns_dirty = False

    # ------------------------------------------------------------------
    # TP structural digest (detects TP rank divergence). Every structural input to
    # a scheduling decision must be identical on all TP ranks; this folds the
    # O(1) rank-uniform quantities into one int and MIN-reduces [d, -d] so a
    # divergence trips an assert (or a CRITICAL log) instead of a NCCL wedge.
    # ------------------------------------------------------------------

    _TP_DIGEST_MASK = (1 << 42) - 1

    def _has_tp_peers(self) -> bool:
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                return True
        return self.tp_world_size > 1

    def tp_digest_mode(self) -> str:
        mode = getattr(self, "_tp_digest_mode_cached", None)
        if mode is None:
            mode = envs.SGLANG_HICACHE_TP_DIGEST_MODE.get().strip().lower()
            if not mode:
                # Off by default: the check adds one CPU all-reduce across the
                # TP group to every scheduler step, which is a measurable share
                # of a decode step. Set "warn" or "assert" to debug TP rank divergence.
                mode = "off"
            if mode not in ("off", "warn", "assert"):
                raise ValueError(
                    f"SGLANG_HICACHE_TP_DIGEST_MODE={mode!r}; expected off|warn|assert"
                )
            self._tp_digest_mode_cached = mode
        return mode

    def _structural_digest_inputs(self) -> tuple[list[int], list[int]]:
        """(assert-grade inputs, warn-grade inputs). Warn-grade currently holds
        the SWA host pool's available_size, which a pre-existing rank-local tail
        release path can legitimately skew."""
        summary = getattr(self.tree_core, "structural_summary", None)
        vals: list[int] = list(summary()) if summary is not None else []
        if self.swa_ledger_registry is not None:
            # Window residency: spill submits and acks must match across ranks.
            vals.extend(self.swa_ledger_registry.digest_inputs())
        for ct in self.tree_core.component_types:
            try:
                vals.append(int(self._component_available_size(ct)))
            except Exception:
                pass
        warn_vals: list[int] = []
        if self.host_pool_group is not None:
            for entry in self.host_pool_group.allocator_entries():
                v = int(self.host_pool_group.available_size(pool=entry.name))
                (warn_vals if entry.name == PoolName.SWA else vals).append(v)
        return vals, warn_vals

    def _fold_digest(self, seed: int, values: list[int]) -> int:
        d = seed
        for v in values:
            d = (
                d * 1000003 + (int(v) & self._TP_DIGEST_MASK) + 1
            ) & self._TP_DIGEST_MASK
        return d

    def structural_digest_pair(self) -> tuple[int, int]:
        vals, warn_vals = self._structural_digest_inputs()
        d = self._fold_digest(0, vals)
        return d, self._fold_digest(d, warn_vals)

    def tp_structural_check(self, batch=None) -> None:
        """Called right before run_batch on every scheduler iteration (all four
        event loops) and piggybacked on the HiCache lockstep all-reduces."""
        mode = self.tp_digest_mode()
        if mode == "off" or self.pp_size != 1 or not self._has_tp_peers():
            return
        d, w = self.structural_digest_pair()
        nreqs = int(batch.batch_size()) if batch is not None else 0
        ntok = (
            int(getattr(batch, "extend_num_tokens", 0) or 0) if batch is not None else 0
        )
        vec = torch.tensor(
            [d, -d, w, -w, nreqs, -nreqs, ntok, -ntok], dtype=torch.int64, device="cpu"
        )
        self._all_reduce_attn_groups(vec, torch.distributed.ReduceOp.MIN)
        v = vec.tolist()
        bad_assert = v[0] != -v[1] or v[4] != -v[5] or v[6] != -v[7]
        bad_warn = v[2] != -v[3]
        if bad_assert or bad_warn:
            self._report_tp_divergence(
                where="pre_run_batch",
                assert_grade=bad_assert,
                local=(d, w, nreqs, ntok),
            )

    def _check_digest_pair(
        self, reduced_min: int, reduced_neg_max: int, where: str
    ) -> None:
        if reduced_min != -reduced_neg_max:
            self._report_tp_divergence(where=where, assert_grade=True, local=None)

    # ------------------------------------------------------------------
    # Step-sync digest: the write_back reclaim digest, folded with two TP
    # lockstep checks (opt in, default off: =1 turns each on; piggybacked on
    # the existing [x, -x] reduce; no extra collective). With both off the
    # step sync checks the reclaim digest alone, as before the checks.
    # writing_check / loading_check settle acks by a MIN-reduced COUNT, which
    # settles the same nodes on every rank only while every rank holds the
    # same acks in the same order.
    # - SGLANG_HICACHE_ACK_LOCKSTEP_CHECK: the shape of the write and load ack
    #   queues. Ranks whose queued writes merged into different ops differ at
    #   the first sync after the merge, before any of those acks settles.
    # - SGLANG_HICACHE_SETTLE_DIGEST_CHECK: a running digest of every write
    #   ack settled so far (position, node count, tokens, and the op's host
    #   rows, or its device rows where a peer-KV servicer makes host rows
    #   rank-local: _settle_digest_folds_host_rows). A settle that released
    #   different writes on some ranks differs at the next sync, whatever the
    #   shared-loader verify setting.
    # TP0 logs "HICACHE_LOCKSTEP checks=.. settled_write_acks=.." every 60 s.
    # ------------------------------------------------------------------

    def _ack_lockstep_check_on(self) -> bool:
        on = getattr(self, "_ack_lockstep_check_cached", None)
        if on is None:
            on = bool(envs.SGLANG_HICACHE_ACK_LOCKSTEP_CHECK.get())
            self._ack_lockstep_check_cached = on
        return on

    def _settle_digest_check_on(self) -> bool:
        on = getattr(self, "_settle_digest_check_cached", None)
        if on is None:
            on = bool(envs.SGLANG_HICACHE_SETTLE_DIGEST_CHECK.get())
            self._settle_digest_check_cached = on
        return on

    def _evict_order_check_mode(self) -> str:
        """SGLANG_HICACHE_EVICT_ORDER_CHECK: off | log | stop."""
        mode = getattr(self, "_evict_order_check_cached", None)
        if mode is None:
            mode = envs.SGLANG_HICACHE_EVICT_ORDER_CHECK.get().strip().lower()
            mode = mode or "off"
            if mode not in ("off", "log", "stop"):
                raise ValueError(
                    f"SGLANG_HICACHE_EVICT_ORDER_CHECK={mode!r}; expected "
                    "off|log|stop"
                )
            self._evict_order_check_cached = mode
        return mode

    def _evict_order_fold_on(self) -> bool:
        # log mode stops folding after the first divergence it reported.
        return self._evict_order_check_mode() != "off" and not getattr(
            self, "_evict_order_diverged", False
        )

    def _evict_order_fields(self) -> tuple[int, int, int, int]:
        """The eviction-order inputs of the step-sync fold, all rank-uniform:
        the host release and device free digests (victim order), the
        flagged-id digest and the next node id. (C7 adds its class epoch.)"""
        tc = self.tree_core
        return (
            int(getattr(tc, "host_release_digest", 0)),
            int(getattr(tc, "device_evict_digest", 0)),
            int(getattr(tc, "_peer_demoted_id_digest", 0)),
            int(UnifiedTreeNode.counter),
        )

    def _settle_digest_folds_host_rows(self) -> bool:
        """Whether the settle digest names a write by its host rows (True) or
        by its device rows (False). Host rows are TP-invariant only while
        every rank's host allocator hands out the same rows. A peer-KV
        servicer breaks that by design (pool_host/slot_allocator.py):
        the serving rank keeps freed pages that a remote reader still
        leases out of its free lists until the lease ends, so its later
        allocations return other rows, while available_size() and every
        decision stay lockstep (invariant I1). Under host dedup only the
        source serves (TP0's write can land over a hundred pages after
        the other ranks', for the same op, node count and tokens).

        Keep the host rows where ranks read them across each other: with
        SGLANG_DSA_HOST_DEDUP_FANOUT=shared, followers load the source's pool
        by their own host indices, so a skew there is a wrong read and must
        trip. Elsewhere each rank reads only rows it allocated itself (the
        nccl / ipc / ce_peer followers hold allocator-only pools and get the
        source's device rows; without dedup every rank owns its pool), and
        the device rows name the write: device allocation is lockstep, and
        the C2g peer copies rely on it. Rank-uniform inputs only: the pools'
        servicer reserve (attach_peer_kv_servicer all-reduces it equal) and
        the fan-out."""
        on = getattr(self, "_settle_host_rows_cached", None)
        if on is None:
            on = True
            group = getattr(self, "host_pool_group", None)
            if group is not None:
                withholds = any(
                    int(
                        getattr(
                            getattr(entry.host_pool, "slot_allocator", None),
                            "reserve_pages",
                            0,
                        )
                        or 0
                    )
                    > 0
                    for entry in group.allocator_entries()
                )
                dedup = getattr(group, "host_dedup", None)
                on = not withholds or bool(getattr(dedup, "shared", False))
            self._settle_host_rows_cached = on
            if not on and self._is_tp_rank0():
                logger.info(
                    "HICACHE_LOCKSTEP settle digest: device rows name the "
                    "settled writes (peer-KV servicer lease withholding makes "
                    "host row ids rank-local; no rank reads another's host rows)"
                )
        return on

    def _note_settled_write(self, ack) -> None:
        """Fold a settled write ack into the running settle digest. The node
        identity is the op's host rows (HiCacheAck.host_sig), or its device
        rows (device_sig) where host rows are rank-local
        (_settle_digest_folds_host_rows): TP-invariant, unlike node ids
        across ranks that a test simulates in one process."""
        if not self._settle_digest_check_on():
            return
        mask = self._TP_DIGEST_MASK
        ids = getattr(ack, "node_ids", None)
        d = getattr(self, "_settled_write_digest", 0)
        d = (d * 1000003 + (0 if ids is None else len(ids)) + 1) & mask
        tokens = int(getattr(ack, "num_tokens", 0) or 0)
        d = (d * 1000003 + (tokens & mask) + 1) & mask
        if self._settle_digest_folds_host_rows():
            rows = int(getattr(ack, "host_sig", 0) or 0)
        elif hasattr(ack, "resolved_device_sig"):
            # Device-resident rows (direct_layout_kernel_backup): their span
            # was copied before the op's start event; finish_event is done.
            rows = int(ack.resolved_device_sig() or 0)
        else:
            rows = int(getattr(ack, "device_sig", 0) or 0)
        self._settled_write_digest = (d * 1000003 + (rows & mask) + 1) & mask
        self._settled_write_acks = getattr(self, "_settled_write_acks", 0) + 1

    def _note_lockstep_check(self) -> None:
        self._lockstep_checks = getattr(self, "_lockstep_checks", 0) + 1
        now = time.monotonic()
        last = getattr(self, "_lockstep_last_log", None)
        if last is not None and now - last < 60.0:
            return
        self._lockstep_last_log = now
        if last is None:
            return
        if self._is_tp_rank0():
            logger.info(
                "HICACHE_LOCKSTEP checks=%d settled_write_acks=%d "
                "ack_queues=%s settled=%s mismatches=0",
                self._lockstep_checks,
                getattr(self, "_settled_write_acks", 0),
                "on" if self._ack_lockstep_check_on() else "off",
                "on" if self._settle_digest_check_on() else "off",
            )
        if self._evict_order_fold_on():
            # Every rank, for an offline compare of the ranks' lines.
            logger.info(
                "HICACHE_EVICT_ORDER rank=%d checks=%d host_release=%d "
                "device_evict=%d flagged=%d next_node_id=%d",
                self._lockstep_tp_rank(),
                self._lockstep_checks,
                *self._evict_order_fields(),
            )

    def _lockstep_tp_rank(self) -> int:
        rank = getattr(self, "_lockstep_tp_rank_cached", None)
        if rank is None:
            rank = -1
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                try:
                    group = getattr(self, "tp_group", None)
                    rank = torch.distributed.get_rank(group=group)
                except Exception:  # noqa: BLE001 - log anyway
                    rank = -1
            self._lockstep_tp_rank_cached = rank
        return rank

    def _is_tp_rank0(self) -> bool:
        rank0 = getattr(self, "_lockstep_rank0", None)
        if rank0 is None:
            rank0 = True
            group = getattr(self, "tp_group", None)
            if (
                group is not None
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                try:
                    rank0 = torch.distributed.get_rank(group=group) == 0
                except Exception:  # noqa: BLE001 - log anyway
                    rank0 = True
            self._lockstep_rank0 = rank0
        return rank0

    @staticmethod
    def _ack_queue_shape(queue) -> list[tuple[int, int]]:
        """(node ids, tokens) per ack. Node ids themselves are left out: they
        match across real TP ranks, but not across ranks that a test
        simulates in one process (one global node counter)."""
        shape = []
        for ack in queue:
            ids = getattr(ack, "node_ids", None)
            tokens = getattr(ack, "num_tokens", 0)
            shape.append((0 if ids is None else len(ids), int(tokens or 0)))
        return shape

    def _step_sync_digest(self, with_order: bool = True) -> int:
        """The value each HiCache step sync checks as [x, -x]: the write_back
        reclaim digest (victim order), folded with the shape of the write and
        load ack queues (ack-queue check), with the running digest of the
        settled write acks (settle check) and with the eviction-order fields
        (SGLANG_HICACHE_EVICT_ORDER_CHECK; ``with_order`` False leaves them
        out). One Python pass over both queues; it rides the existing reduce."""
        digest = self.tree_core.write_back_duplicate_reclaim_digest
        cc = self.cache_controller
        ack_on = cc is not None and self._ack_lockstep_check_on()
        settle_on = self._settle_digest_check_on()
        order_on = with_order and self._evict_order_fold_on()
        if not (ack_on or settle_on or order_on):
            return digest
        # _fold_digest, inlined: this runs at every step sync.
        mask = self._TP_DIGEST_MASK
        d = digest
        if ack_on:
            for queue in (
                getattr(cc, "ack_write_queue", ()),
                getattr(cc, "ack_load_queue", ()),
            ):
                d = (d * 1000003 + len(queue) + 1) & mask
                for ack in queue:
                    ids = getattr(ack, "node_ids", None)
                    d = (d * 1000003 + (0 if ids is None else len(ids)) + 1) & mask
                    tokens = int(getattr(ack, "num_tokens", 0) or 0)
                    d = (d * 1000003 + (tokens & mask) + 1) & mask
        if settle_on:
            settled = getattr(self, "_settled_write_digest", 0)
            d = (d * 1000003 + settled + 1) & mask
        if order_on:
            for v in self._evict_order_fields():
                d = (d * 1000003 + (v & mask) + 1) & mask
        if with_order:
            self._note_lockstep_check()
        return d

    def _check_step_sync_digest(
        self, reduced_min: int, reduced_neg_max: int, where: str
    ) -> None:
        """Raise on every rank if the step-sync digests differed. Each rank
        logs its reclaim digest, ack-queue shapes, settle digest and
        eviction-order fields, gathered over the TP group when
        torch.distributed is up (CRITICAL "HICACHE_LOCKSTEP ... diverged
        across TP ranks at <where>"). SGLANG_HICACHE_EVICT_ORDER_CHECK=log: if
        the gathered details show that only the eviction order diverged, log
        it once, stop folding it and go on."""
        if reduced_min == -reduced_neg_max:
            return
        cc = self.cache_controller
        ack_check = cc is not None and self._ack_lockstep_check_on()
        settle_check = self._settle_digest_check_on()
        order_check = self._evict_order_fold_on()
        dist_ready = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        detail = {
            "rank": torch.distributed.get_rank() if dist_ready else -1,
            "where": where,
            "reclaim_digest": int(self.tree_core.write_back_duplicate_reclaim_digest),
        }
        if ack_check:
            # As of this check; the pipelined sync compared a snapshot taken
            # one sync step earlier.
            for name in ("ack_write_queue", "ack_load_queue"):
                shape = self._ack_queue_shape(getattr(cc, name, ()))
                detail[name] = {"len": len(shape), "head": shape[:16]}
        if settle_check:
            detail["settled_write_acks"] = getattr(self, "_settled_write_acks", 0)
            detail["settled_write_digest"] = getattr(self, "_settled_write_digest", 0)
            detail["settle_rows"] = (
                "host" if self._settle_digest_folds_host_rows() else "device"
            )
        if order_check:
            # As of this check (likewise for the pipelined sync).
            detail["digest_without_order"] = self._step_sync_digest(with_order=False)
            detail["evict_order"] = list(self._evict_order_fields())
        gathered: list = [detail]
        if dist_ready:
            try:
                group = self.attn_tp_group
                if group is None or torch.distributed.get_world_size(group=group) <= 1:
                    group = self.tp_group
                gathered = [None] * torch.distributed.get_world_size(group=group)
                torch.distributed.all_gather_object(gathered, detail, group=group)
            except Exception:
                gathered = [detail]
        parts = ["write_back duplicate-reclaim victims"]
        if ack_check:
            parts.append("HiCache ack queues")
        if settle_check:
            parts.append("settled write acks")
        if order_check:
            parts.append("eviction order")
            base = detail["digest_without_order"]
            if (
                self._evict_order_check_mode() == "log"
                and len(gathered) > 1
                and all(
                    isinstance(g, dict) and g.get("digest_without_order") == base
                    for g in gathered
                )
            ):
                # Every rank saw the same gathered list: all stop folding.
                self._evict_order_diverged = True
                logger.critical(
                    "HICACHE_LOCKSTEP eviction order diverged across TP ranks "
                    "at %s (SGLANG_HICACHE_EVICT_ORDER_CHECK=log: no longer "
                    "folded; =stop raises): %s",
                    where,
                    gathered,
                )
                return
        what = (
            parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " or " + parts[-1]
        )
        logger.critical(
            "HICACHE_LOCKSTEP %s diverged across TP ranks at %s: %s",
            what,
            where,
            gathered,
        )
        raise AssertionError(
            f"{what} diverged across TP ranks at {where} (min={reduced_min} "
            f"max={-reduced_neg_max}). Differently merged write ops settle "
            "different nodes per rank. "
            "SGLANG_HICACHE_ACK_LOCKSTEP_CHECK=0 / "
            "SGLANG_HICACHE_SETTLE_DIGEST_CHECK=0 / "
            "SGLANG_HICACHE_EVICT_ORDER_CHECK=off turn the checks off."
        )

    def _report_tp_divergence(self, *, where: str, assert_grade: bool, local) -> None:
        mode = self.tp_digest_mode()
        vals, warn_vals = self._structural_digest_inputs()
        # Without a default process group (CPU tests that simulate TP ranks by
        # replacing _all_reduce) report the local digest instead of raising.
        dist_ready = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        detail = {
            "rank": torch.distributed.get_rank() if dist_ready else -1,
            "where": where,
            "local": local,
            "structural": vals,
            "warn_grade": warn_vals,
        }
        gathered: list = [detail]
        if dist_ready:
            try:
                group = self.attn_tp_group
                if group is None or torch.distributed.get_world_size(group=group) <= 1:
                    group = self.tp_group
                gathered = [None] * torch.distributed.get_world_size(group=group)
                torch.distributed.all_gather_object(gathered, detail, group=group)
            except Exception:
                gathered = [detail]
        logger.critical(
            "HiCache TP structural digest DIVERGED at %s (assert_grade=%s, mode=%s): %s",
            where,
            assert_grade,
            mode,
            gathered,
        )
        if mode == "assert" and assert_grade:
            raise RuntimeError(
                f"HiCache TP structural digest diverged across TP ranks at {where}"
            )

    def reset(self) -> None:
        self._clear_host_returns()
        # SGLANG_HICACHE_SKIP_IDLE_SYNC: sync on the next step.
        self._idle_sync_all_idle = False
        self._step_sync_flag = None
        self._last_step_sync_flag = None
        # SGLANG_HICACHE_PIPELINED_SYNC: the started snapshot counts acks and
        # jobs the reset drops.
        self._discard_pipelined_sync()
        self._reset_full()
        if self.peer_kv_servicer is not None:
            self.peer_kv_servicer.push_clear()

    def _reset_full(self) -> None:
        """Full reset: destroy entire tree and all state."""
        # Window residency: spill ops own device and host slots until their
        # ack, and the scheduler flushes only when idle.
        registry = self.swa_ledger_registry
        if registry is not None:
            assert registry.live_count() == 0, "cache reset with in-flight SWA spills"
            assert (
                self.swa_backfill is None or self.swa_backfill.pending_count() == 0
            ), "cache reset with pending SWA backfills"
            if self.swa_backfill is not None:
                # No job is left: stop the IO thread (restarted on next use).
                self.swa_backfill.close()
            if registry.ledger_live:
                # No request runs across a reset: these were never released.
                logger.warning(
                    "cache reset drops %d unreleased SWA ledgers",
                    registry.ledger_live,
                )
                registry.ledger_live = 0
        if self._swa_copy_verify:
            self._swa_copy_verify_spills.clear()
            self._swa_copy_verify_loads.clear()
        self.tree_core.reset()
        self.session_refs.reset()
        if self.kv_sessions is not None:
            self.kv_sessions.reset_storage()
            self._kv_session_flush_generation += 1
            if self.kv_session_boot_id is not None:
                self.kv_session_boot_id = (
                    self.kv_session_boot_id.split(":", 1)[0]
                    + f":{self._kv_session_flush_generation}"
                )

        # Reset Controller.
        self.session.slots.clear()
        self.ongoing_write_through: dict[int, _OngoingWriteThrough] = {}
        self.ongoing_load_back: dict[int, _OngoingLoadBack] = {}
        self.enable_storage = False
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        self.ongoing_prefetch: dict[str, _OngoingPrefetch] = {}
        # Rids whose storage prefetch resolved without a usable result;
        # popped by the scheduler to pace availability-check retries.
        self._storage_prefetch_missed_rids: set[str] = set()
        self.ongoing_backup: dict[int, tuple[NodeId, DecLockRefParams]] = {}
        if self.buffer_pipeline is not None:
            self.buffer_pipeline.reset()

        if self.cache_controller is not None:
            self.cache_controller.reset()
            self.cache_controller.mem_pool_host.clear()
            self.enable_storage = self.cache_controller.enable_storage

        self.tree_core.kv_events.record_all_cleared()

    def init_hicache(self, server_args: ServerArgs, params: CacheInitParams) -> None:
        """Initialize HiCache infrastructure."""
        self.host_memory_mode = get_memory().hicache_host_memory_mode
        if self.enable_kv_session_retention and (
            self.host_memory_mode == "buffer_only"
            or get_memory().hicache_write_policy != "write_through"
        ):
            raise ValueError(
                "KV session retention requires cache-mode write_through HiCache"
            )
        if self.host_memory_mode == "buffer_only":
            # FULL and FULL+SWA only: Mamba has no state-handoff channel on
            # the admission-time load-back read path and is not layer-gated.
            # Lifting the fence also needs the admission charge: a staged
            # state slot is request-pinned at consumption and must ride
            # req.mamba_host_hit_length the way the SWA window does.
            supported = {ComponentType.FULL, ComponentType.SWA}
            if not set(self.tree_components) <= supported:
                raise ValueError(
                    "--hicache-host-memory-mode buffer_only supports only "
                    "FULL/SWA unified trees; got components "
                    f"{sorted(ct.name for ct in self.tree_components)}."
                )
        from decoupled_engine.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            attach_hybrid_pool_to_unified_cache,
        )

        self.load_cache_event = threading.Event()
        self.sidecar_pool_specs.clear()
        self.extra_metric_labels = get_observability().extra_metric_labels

        # Parse storage config once, share with assembler and tree
        storage_backend = get_memory().hicache_storage_backend
        storage_extra_config = None
        storage_prefetch_threshold = 256
        prefetch_timeout_base = 1.0
        prefetch_timeout_per_ki_token = 0.25
        hicache_storage_pass_prefix_keys = False
        if storage_backend is not None:
            (
                storage_extra_config,
                storage_prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = HybridCacheController.parse_storage_backend_extra_config(
                get_memory().hicache_storage_backend_extra_config
            )

        attach_hybrid_pool_to_unified_cache(
            self,
            params,
            server_args,
            load_cache_event=self.load_cache_event,
            storage_backend=storage_backend,
            storage_extra_config=storage_extra_config,
            storage_prefetch_threshold=storage_prefetch_threshold,
        )
        # Tag HiCache enablement on the TreeCore.
        if self.cache_controller is not None:
            self.tree_core.set_hicache_enabled()
            if self.supports_swa():
                swa = self.components[ComponentType.SWA]
                self.tree_core.has_swa_host_pool = swa._swa_kv_pool_host is not None

        if self.host_memory_mode == "buffer_only":
            swa = self.components.get(ComponentType.SWA)
            validate_buffer_only_stack(
                sidecar_pool_specs=self.sidecar_pool_specs, swa_component=swa
            )
            self.buffer_pipeline = BufferModePipeline(
                cache=self,
                max_context_len=server_args.context_length or 0,
                swa_window_pages=(
                    swa.full_window_pages
                    if swa is not None and self.tree_core.has_swa_host_pool
                    else 0
                ),
                # Leak backstop only: live queued tokens are intrinsically
                # bounded by the FULL device pool (one intent per node, stale
                # intents swept per tick), so a cap that binds on live
                # content would drop-newest and punch storage holes.
                write_backlog_cap=2 * self.token_to_kv_pool_allocator.size_full,
            )
            self.cache_controller.host_write_staged_tokens_fn = lambda: (
                self.buffer_pipeline.write_staged_tokens_
            )

        # State initialization
        self.write_through_threshold = (
            1 if get_memory().hicache_write_policy == "write_through" else 2
        )
        self.is_write_back = (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        )
        # Pre-seed the dropped-tokens series at 0 per pool
        if self.metrics_collector is not None and self.cache_controller is not None:
            for ct in self.tree_components:
                self.metrics_collector.increment_dropped_tokens(
                    num_tokens=0,
                    reason="host_pressure",
                    pool=_COMPONENT_POOL_LABEL[ct],
                )
            # Full-KV loss counters (failed backups, un-backed deletes,
            # reclaimed host duplicates) also exist from the start.
            full_pool = _COMPONENT_POOL_LABEL[BASE_COMPONENT_TYPE]
            self.metrics_collector.increment_backup_failed(0, 0, pool=full_pool)
            self.metrics_collector.increment_unbacked_evicted_tokens(0, pool=full_pool)
            self.metrics_collector.increment_host_duplicate_reclaimed_tokens(
                0, pool=full_pool
            )
        self.load_back_threshold = 10
        self.prefetch_stop_policy = get_memory().hicache_storage_prefetch_policy

        # Runtime attach/detach of the L3 backend (startup, admin API, atexit).
        self._storage_attachment = StorageAttachment(self)
        atexit.register(self.shutdown)

        if storage_backend is not None:
            self._storage_attachment.apply_runtime_config(
                storage_backend=storage_backend,
                prefetch_threshold=storage_prefetch_threshold,
                prefetch_timeout_base=prefetch_timeout_base,
                prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
                enable_storage=self.cache_controller.enable_storage,
                enable_storage_metrics=self._enable_metrics_flag,
                extra_metric_labels=self.extra_metric_labels,
            )
        if (
            self._swa_window
            and storage_backend is not None
            and getattr(get_memory(), "dsv4_swa_backfill", "auto") != "off"
        ):
            self.init_swa_backfill()
        if envs.SGLANG_HICACHE_PIPELINED_SYNC.get():
            self._init_pipelined_sync()

    def register_sidecar_pool(
        self, spec: SidecarPoolSpec, entry: Optional[PoolEntry] = None
    ) -> None:
        if entry is not None:
            if self.cache_controller is None:
                raise RuntimeError("HiCache controller is not attached.")
            self.cache_controller.register_host_pool_entry(entry)
        self.sidecar_pool_specs.append(spec)

    def release_host_resources(self) -> None:
        if self.host_pool_group is not None:
            self.host_pool_group.destroy()

    @rank_consensus(
        same_params=["params"],
        same_results=["result.full_kv_hit_length", "result.swa_host_hit_length"],
    )
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        if self.kv_sessions is not None and params.req is not None:
            self.note_kv_session_arrival(params.req)
        active_components = self._resolve_active_components(
            params.active_components, for_match=True
        )
        if params.active_components is not None:
            if self.enable_session_radix_cache:
                raise ValueError(
                    "partial-component match_prefix is incompatible with "
                    "--enable-session-radix-cache"
                )
            req = params.req
            session = getattr(req, "session", None) if req is not None else None
            if session is not None and getattr(session, "streaming", False):
                raise ValueError(
                    "partial-component match_prefix is incompatible with "
                    "streaming sessions"
                )
        else:
            result = self.session.try_match_prefix(params)
            if result is not None:
                return result
        if self.disable:
            return self.tree_core.empty_match_result
        result = self.tree_core.match_prefix(params)
        if (
            self.kv_sessions is not None
            and params.req is not None
            and not self.kv_sessions._live(
                getattr(params.req, "_kv_session_registration", None)
            )
        ):
            node = self.tree_core.node_by_id(result.last_host_node)
            if getattr(params.req, "kv_session", None) is None:
                self.kv_sessions.mark_untracked(node)
            else:
                self.kv_sessions.mark_retired(node)
        # Apply the walk's actions (e.g. a pending write-through relocation on
        # a split) before the finalizers, which can evict or raise.
        self._apply_cache_actions(result.cache_actions)
        for component in active_components:
            result = component.finalize_match_result_in_cache(params, result)
        # Finalizers must not emit actions; the walk's were applied above.
        assert not result.cache_actions
        return result

    def is_chunk_cache(self) -> bool:
        return self.disable

    @timed_phase("tree.insert")
    def insert(self, params: InsertParams) -> InsertResult:
        self._resolve_active_components(params.active_components)
        if params.active_components is not None and self.enable_session_radix_cache:
            raise ValueError(
                "partial-component insert is incompatible with "
                "--enable-session-radix-cache"
            )
        if self.disable:
            return InsertResult(prefix_len=0)
        # Fail fast on re-entrancy without touching the in-flight walk.
        assert not self.tree_core.has_ongoing_insert(), "re-entrant insert"
        # Pump the resumable insert, applying each step's actions at its barrier.
        try:
            step = self.tree_core.begin_insert(params)
            while True:
                self._apply_cache_actions(step.actions)
                if step.result is not None:
                    # Walk actions flow through the steps; the result is action-free.
                    assert not step.result.cache_actions
                    result = step.result
                    break
                step = self.tree_core.resume_insert()
        finally:
            # Drain still-pending actions so frees reach the allocator on abort.
            self._apply_cache_actions(self.tree_core.end_insert())
        gate = getattr(self.tree_core.kv_events, "d_role_gate", None)
        if gate is not None and gate.depth == 0 and result.last_device_node is not None:
            # SGLANG_SYMMETRIC_SKIP_D_ROLE_KV_EVENTS: an insert outside D-role
            # work publishes its whole path (DRoleEventGate.promote).
            gate.promote(self.tree_core.node_by_id(result.last_device_node))
        if self.kv_sessions is not None:
            self.kv_sessions.materialized(result.last_device_node)
        return result

    def evict(self, params: EvictParams) -> EvictResult:
        return self._evict(params)

    @timed_phase("evict_for_alloc")
    def evict_for_alloc(self, params: EvictParams) -> EvictResult:
        """Evict until the requested component allocations become feasible.

        ``params`` contains allocator shortfalls, not absolute eviction quotas.
        A component eviction can cascade to its peers; with a shared memory pool,
        those collateral frees can satisfy the original allocation before the
        triggering component's requested count is reached.
        """
        if self.disable:
            return EvictResult()

        request_by_type = self._evict_request_by_type(params)
        available_size_targets = {
            ct: self._component_available_size(ct) + request_cnt
            for ct, request_cnt in request_by_type.items()
            if request_cnt > 0
        }
        return self._evict(params, available_size_targets)

    @staticmethod
    def _evict_request_by_type(params: EvictParams) -> dict[ComponentType, int]:
        return {
            ComponentType.FULL: params.num_tokens,
            ComponentType.SWA: params.swa_num_tokens,
            ComponentType.MAMBA: params.mamba_num,
            ComponentType.C128: 0,
        }

    def _component_available_size(self, component_type: ComponentType) -> int:
        """Return capacity usable by the component's next allocation.

        Shared allocators expose schedulable capacity, which includes peer holes
        that an urgent allocator flush can reclaim without further eviction.
        """
        if component_type == ComponentType.FULL:
            if self.supports_swa():
                return self.token_to_kv_pool_allocator.full_available_size()
            return self.token_to_kv_pool_allocator.available_size()
        if component_type == ComponentType.SWA:
            return self.token_to_kv_pool_allocator.swa_available_size()
        if component_type == ComponentType.MAMBA:
            return self.req_to_token_pool.mamba_allocator.schedulable_available_size()
        raise ValueError(f"Unsupported cache component: {component_type}")

    def _evict(
        self,
        params: EvictParams,
        available_size_targets: Optional[dict[ComponentType, int]] = None,
    ) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        tracker = {ct: 0 for ct in self.tree_components}

        request_by_type = self._evict_request_by_type(params)
        self._evict_components(
            request_by_type,
            tracker,
            available_size_targets=available_size_targets,
        )

        if (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        ):
            self.writing_check(write_back=True)

        # Report full-layer tokens only
        self.update_eviction_metrics(tracker[BASE_COMPONENT_TYPE], start_time)
        return EvictResult(
            num_tokens_evicted=tracker[BASE_COMPONENT_TYPE],
            swa_num_tokens_evicted=tracker.get(ComponentType.SWA, 0),
            mamba_num_evicted=tracker.get(ComponentType.MAMBA, 0),
        )

    def _free_values(
        self,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Free a tree-side step's returned device and host values right away."""
        # Both drains must run even if one raises.
        try:
            self._drain_device_frees(device_frees)
        finally:
            self._drain_host_frees(host_frees)

    def _accumulate_tracker(
        self,
        tracker: dict[ComponentType, int],
        delta: dict[ComponentType, int],
    ) -> None:
        """Fold a step result's evicted delta into the running totals."""
        for ct, n in delta.items():
            tracker[ct] += n

    def _evict_device_next_node(
        self, component_type: ComponentType, tracker: dict[ComponentType, int]
    ) -> tuple[Optional[NodeId], bool]:
        """Advance the eviction walk one node, consuming its step result."""
        result = self.tree_core.evict_device_next_node(component_type, tracker)
        self._free_values(result.device_frees, result.host_frees)
        self._accumulate_tracker(tracker, result.tracker)
        return result.node_id, result.made_progress

    def _evict_device_leaf(
        self, node_id: NodeId, tracker: dict[ComponentType, int]
    ) -> Optional[BackupKV]:
        """Evict one device leaf, consuming its step result; returns the
        deferred write-back BackupKV when one must run before the demote."""
        result = self.tree_core.evict_device_leaf(node_id, self.is_write_back)
        self._free_values(result.device_frees, result.host_frees)
        self._accumulate_tracker(tracker, result.tracker)
        if result.unbacked_deleted_tokens and self.metrics_collector is not None:
            self.metrics_collector.increment_unbacked_evicted_tokens(
                result.unbacked_deleted_tokens,
                pool=_COMPONENT_POOL_LABEL[BASE_COMPONENT_TYPE],
            )
        return result.backup_kv

    def _demote(self, node_id: NodeId, tracker: dict[ComponentType, int]) -> None:
        """Demote a backed-up node, consuming its step result."""
        result = self.tree_core.demote(node_id)
        self._free_values(result.device_frees, result.host_frees)
        self._accumulate_tracker(tracker, result.tracker)

    def _drop_subtree_no_host(
        self, node_id: NodeId, tracker: dict[ComponentType, int]
    ) -> bool:
        """Run the write-back drop fallback, consuming its step result."""
        result = self.tree_core.drop_subtree_no_host(node_id)
        self._free_values(result.device_frees, result.host_frees)
        self._accumulate_tracker(tracker, result.tracker)
        return result.is_dropped

    def _evict_components(
        self,
        request_by_type: dict[ComponentType, int],
        tracker: dict[ComponentType, int],
        available_size_targets: Optional[dict[ComponentType, int]] = None,
    ) -> None:
        # Buffer mode: eviction always wins over queued backup intents — a
        # destroyed victim's intent is stale-swept and the content rewrites
        # after its recompute.

        def target_reached(component_type: ComponentType) -> bool:
            if available_size_targets is None:
                return False
            target = available_size_targets.get(component_type)
            # Do not compact on every eviction step. Shared allocators include
            # drainable peer holes here and flush the peer once in alloc().
            return (
                target is not None
                and self._component_available_size(component_type) >= target
            )

        for ct in self.tree_components:
            request_cnt = request_by_type[ct]
            # A preceding component may have cascade-evicted this component or,
            # on a shared pool, released enough bytes to satisfy its allocation.
            if tracker[ct] >= request_cnt or target_reached(ct):
                continue
            self.tree_core.evict_device_start(ct, request_cnt)
            try:
                while not target_reached(ct):
                    node_id, made_progress = self._evict_device_next_node(ct, tracker)
                    if node_id is None:
                        if made_progress:
                            # Internal tombstone frees are now allocator-visible;
                            # recheck the allocation target before walking again.
                            continue
                        break
                    backup_kv = self._evict_device_leaf(node_id, tracker)
                    if backup_kv is not None:
                        # Deferred demote: run the D->H backup, demote only on success.
                        written = self._execute_and_commit_kv_backup(
                            backup_kv, write_back=True
                        )
                        freed_before_drop = dict(tracker)
                        if written > 0:
                            self.writing_check(write_back=True)
                            self._demote(node_id, tracker)
                        elif self._drop_subtree_no_host(node_id, tracker):
                            self._record_dropped_tokens(tracker, freed_before_drop)
                            logger.warning(
                                "write_back: KV subtree dropped without backup "
                                "due to host memory pressure, root node %d",
                                node_id,
                            )
                        else:
                            logger.warning(
                                "write_back: backup failed under host memory "
                                "pressure but subtree drop declined (node "
                                "locked); root node %d stays device-resident "
                                "until host space frees",
                                node_id,
                            )
            finally:
                self.tree_core.evict_device_end(ct)

    def _record_dropped_tokens(
        self,
        tracker: dict[ComponentType, int],
        freed_before_drop: dict[ComponentType, int],
    ) -> None:
        """Record per-pool tokens dropped without backup under host pressure."""
        if self.metrics_collector is None:
            return
        for ct, freed in tracker.items():
            dropped = freed - freed_before_drop[ct]
            if dropped > 0:
                self.metrics_collector.increment_dropped_tokens(
                    num_tokens=dropped,
                    reason="host_pressure",
                    pool=_COMPONENT_POOL_LABEL[ct],
                )

    def inc_lock_ref(
        self, node_id: NodeId, skip_lock_components: Sequence[ComponentType] = ()
    ) -> IncLockRefResult:
        result = self.session.try_inc_lock_ref(node_id)
        if result is not None:
            return result
        if self.disable:
            return IncLockRefResult()
        return self.tree_core.inc_lock_ref(node_id, skip_lock_components)

    def dec_lock_ref(
        self,
        node_id: NodeId,
        params: Optional[DecLockRefParams] = None,
        skip_swa: bool = False,
    ) -> DecLockRefResult:
        result = self.session.try_dec_lock_ref(node_id, params)
        if result is not None:
            return result
        if self.disable:
            return DecLockRefResult()
        return self.tree_core.dec_lock_ref(node_id, params, skip_swa)

    def _dec_req_lock(self, req: Req, *, skip_swa: bool = False) -> None:
        """Release the tree lock a request holds on its last_node, honoring the
        components it skipped locking so it never drops a lock it never took."""
        if req.last_node is None:
            # A retracted request restored into private KV holds no tree lock.
            return
        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(
                swa_uuid_for_lock=req.swa_uuid_for_lock,
                skip_lock_node_ids=req.skip_lock_node_ids,
                skip_lock_components=getattr(req, "skip_lock_components", ()),
            ),
            skip_swa=skip_swa,
        )

    def dec_swa_lock_only(
        self,
        node_id: NodeId,
        swa_uuid_for_lock: Optional[int] = None,
        skip_lock_node_ids: Optional[dict] = None,
    ) -> None:
        if self.disable:
            return
        result = self.tree_core.dec_swa_lock_only(
            node_id, swa_uuid_for_lock, skip_lock_node_ids
        )
        self._free_values(result.device_frees, result.host_frees)

    def inc_host_lock_ref(self, node_id: NodeId) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult()
        return self.tree_core.inc_host_lock_ref(node_id)

    def dec_host_lock_ref(
        self, node_id: NodeId, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult()
        return self.tree_core.dec_host_lock_ref(node_id, params)

    def suppress_write_through(self):
        """See BasePrefixCache.suppress_write_through: raise the write-through
        threshold for the duration so the nodes inserted meanwhile are not
        backed up to host (a later hit on them still may be)."""
        import contextlib

        cache = self

        @contextlib.contextmanager
        def _cm():
            saved = cache.write_through_threshold
            cache.write_through_threshold = 1 << 30
            try:
                yield
            finally:
                cache.write_through_threshold = saved

        return _cm()

    @timed_phase("tree.cache_finished_req")
    @_d_role_kv_events
    def cache_finished_req(
        self,
        req: Req,
        is_insert: bool = True,
        *,
        kv_len_to_handle: int,
        max_cache_len: Optional[int] = None,
        active_components: Optional[tuple[ComponentType, ...]] = None,
        **kwargs,
    ) -> None:
        uses_partial_api = max_cache_len is not None or active_components is not None
        if uses_partial_api:
            if self.enable_session_radix_cache:
                raise ValueError(
                    "partial-component cache_finished_req is incompatible with "
                    "--enable-session-radix-cache"
                )
            session = getattr(req, "session", None)
            if session is not None and getattr(session, "streaming", False):
                raise ValueError(
                    "partial-component cache_finished_req is incompatible with "
                    "streaming sessions"
                )
            if kv_len_to_handle < 0:
                raise ValueError(
                    f"kv_len_to_handle must be non-negative, got {kv_len_to_handle}"
                )
            if max_cache_len is not None and not 0 <= max_cache_len <= kv_len_to_handle:
                raise ValueError(
                    "max_cache_len must be within the committed boundary: "
                    f"{max_cache_len=} {kv_len_to_handle=}"
                )
        else:
            if self.session.try_cache_finished_req(req, is_insert=is_insert, **kwargs):
                return

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_len_to_handle
            ]
            self.token_to_kv_pool_allocator.free_segment(kv_indices, start_pos=0)
            for comp in self._components_tuple:
                comp.cleanup_after_caching_req(req, is_finished=True)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_len_to_handle
        ]

        result = None
        insert_params = None
        retention_key = None
        active_component_drivers: tuple[TreeComponent, ...] = ()

        if is_insert:
            if not uses_partial_api:
                active_component_drivers = self._components_tuple
                insert_params = InsertParams(
                    prev_prefix_len=req.cache_protected_len,
                    priority=getattr(req, "priority", 0) or 0,
                )
                effective_cache_len = len(token_ids)
                for comp in self._components_tuple:
                    cl = comp.prepare_for_caching_req(
                        req=req,
                        insert_params=insert_params,
                        token_ids_len=len(token_ids),
                        is_finished=True,
                    )
                    if cl is not None:
                        effective_cache_len = min(effective_cache_len, cl)

                kv_indices_full = kv_indices
                tail_free_start = None
                if effective_cache_len < len(token_ids):
                    tail_free_start = max(effective_cache_len, req.cache_protected_len)
                    token_ids = token_ids[:effective_cache_len]
                    kv_indices = kv_indices[:effective_cache_len]

                # Retention needs the actual committed context boundary to
                # distinguish a completed short retry from partial prefill.
                # Insertion still owns only whole native pages.
                retention_key = RadixKey(
                    token_ids,
                    req.extra_key,
                    is_bigram=self.tree_core.is_eagle,
                    cache_salt=req.cache_salt,
                )
                radix_key = retention_key.page_aligned(self.page_size)
                page_aligned_len = len(radix_key)
                values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)
                # Q4: CPU page ids from the request's CPU row (no-op when off).
                rows = cpu_rows_of(self.req_to_token_pool)
                tag_row_slice(values, rows, req.req_pool_idx, 0)

                insert_params.key = radix_key
                insert_params.value = values
                chunk_ends = getattr(req, "release_insert_chunk_ends", None)
                if chunk_ends:
                    # SGLANG_SYMMETRIC_RELEASE_INSERT_CHUNKED: end a node where
                    # each chunk insert of a standard P ends its key
                    # (cache_unfinished_req): page_floor(chunk end), less the
                    # boundary token with bigram keys.
                    shift = 1 if radix_key.is_bigram else 0
                    page = self.page_size
                    insert_params.chain_cuts = [
                        (end - shift) // page * page for end in chunk_ends
                    ]
                    req.release_insert_chunk_ends = None  # used once
                result = self.insert(insert_params)

                segments = [(kv_indices[page_aligned_len:], page_aligned_len)]
                if tail_free_start is not None:
                    segments.append(
                        (kv_indices_full[tail_free_start:], tail_free_start)
                    )
                for seg, seg_start in segments:
                    tag_row_slice(seg, rows, req.req_pool_idx, seg_start)
                self.token_to_kv_pool_allocator.free_segments(segments)
            else:
                active_component_drivers = self._resolve_active_components(
                    active_components
                )
                normalized_active_components = tuple(
                    component.component_type for component in active_component_drivers
                )
                insert_params = InsertParams(
                    prev_prefix_len=req.cache_protected_len,
                    priority=getattr(req, "priority", 0) or 0,
                    active_components=normalized_active_components,
                )

                # The frozen insertion boundary and committed ownership
                # boundary are deliberately independent.
                cache_len_to_insert = (
                    kv_len_to_handle if max_cache_len is None else max_cache_len
                )
                cache_len_to_insert = min(cache_len_to_insert, len(token_ids))
                effective_cache_len = cache_len_to_insert
                for comp in active_component_drivers:
                    cl = comp.prepare_for_caching_req(
                        req=req,
                        insert_params=insert_params,
                        token_ids_len=cache_len_to_insert,
                        is_finished=True,
                    )
                    if cl is not None:
                        if cl < 0:
                            raise ValueError(
                                f"{comp.component_type.name} returned negative "
                                f"cache len {cl}"
                            )
                        effective_cache_len = min(effective_cache_len, cl)

                retention_key = RadixKey(
                    token_ids[:effective_cache_len],
                    req.extra_key,
                    is_bigram=self.tree_core.is_eagle,
                    cache_salt=req.cache_salt,
                )
                radix_key = retention_key.page_aligned(self.page_size)
                page_aligned_len = len(radix_key)
                if page_aligned_len < req.cache_protected_len:
                    raise ValueError(
                        "frozen cache boundary cannot precede the request's "
                        f"protected prefix: {page_aligned_len=} "
                        f"{req.cache_protected_len=}"
                    )
                active_component_types = {
                    component.component_type for component in active_component_drivers
                }
                drops_incoming_swa = (
                    self.is_swa_enabled
                    and ComponentType.SWA not in active_component_types
                )
                if drops_incoming_swa and req.cache_protected_len % self.page_size != 0:
                    raise ValueError(
                        "cache_protected_len must be page-aligned for a FULL-only "
                        f"insert: {req.cache_protected_len=} {self.page_size=}"
                    )

                values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)
                rows = cpu_rows_of(self.req_to_token_pool)
                tag_row_slice(values, rows, req.req_pool_idx, 0)
                insert_params.key = radix_key
                insert_params.value = values
                result = self.insert(insert_params)

                # The tree retains only FULL slots. Drop request-owned SWA
                # mappings for the inserted, non-protected pages; overlap frees
                # clear their mappings first, making this idempotent there.
                if drops_incoming_swa:
                    self.token_to_kv_pool_allocator.free_swa(
                        values[req.cache_protected_len :]
                    )

                # max_cache_len is page-floored by RadixKey. Release exactly
                # [page_floor(max_cache_len), kv_len_to_handle); common.py
                # separately releases page-aligned over-allocation beyond the
                # committed boundary.
                self.token_to_kv_pool_allocator.free_segment(
                    tag_row_slice(
                        kv_indices[page_aligned_len:],
                        rows,
                        req.req_pool_idx,
                        page_aligned_len,
                    ),
                    start_pos=page_aligned_len,
                )
        else:
            self.token_to_kv_pool_allocator.free_segment(
                tag_row_slice(
                    kv_indices[req.cache_protected_len :],
                    cpu_rows_of(self.req_to_token_pool),
                    req.req_pool_idx,
                    req.cache_protected_len,
                ),
                start_pos=req.cache_protected_len,
            )

        self._dec_req_lock(req, skip_swa=req.swa_prefix_lock_released)
        if self._swa_window:
            self._release_swa_lag_lock(req)
            if not is_insert:
                # Unbacked chunk nodes the request no longer locks (abort
                # mid-prefill) are backed up once their forward is processed.
                self._enqueue_swa_rebackup(req.last_node, after_next_result=True)

        if is_insert and result is not None and result.last_device_node is not None:
            req.last_node = result.last_device_node

        if self.kv_sessions is not None:
            if not getattr(req, "_kv_session_arrival_seen", False):
                self.note_kv_session_arrival(req)
            self.kv_sessions.refresh_request(
                req,
                finished=True,
                key=retention_key,
                node_id=result.last_device_node if result is not None else None,
            )
            if (
                result is not None
                and result.last_device_node is not None
                and not self.kv_sessions._live(
                    getattr(req, "_kv_session_registration", None)
                )
            ):
                node = self.tree_core.node_by_id(result.last_device_node)
                if getattr(req, "kv_session", None) is None:
                    self.kv_sessions.mark_untracked(node)
                else:
                    self.kv_sessions.mark_retired(node)

        # cleanup
        for comp in self._components_tuple:
            component_was_active = not uses_partial_api or (
                comp in active_component_drivers
            )
            comp.cleanup_after_caching_req(
                req,
                is_finished=True,
                insert_result=result if component_was_active else None,
                insert_params=insert_params if component_was_active else None,
            )

        if self.enable_session_radix_cache and result is not None:
            from decoupled_engine.srt.managers.schedule_batch import FINISH_ABORT

            if req.finished_reason is not None and not isinstance(
                req.finished_reason, FINISH_ABORT
            ):
                self.session_refs.register_session_ref(req)

        # Reuse the existing write-through, including EAGLE's committed,
        # page-aligned key above. No extra GPU copy or per-token callback.
        if is_insert and result is not None and insert_params is not None:
            self._queue_host_return(req, result, insert_params.key)

    @timed_phase("tree.cache_unfinished_req")
    @_d_role_kv_events
    def cache_unfinished_req(self, req: Req, chunked: bool = False, **kwargs) -> None:
        if self.session.try_cache_unfinished_req(req, chunked=chunked, **kwargs):
            return

        token_ids = req.get_fill_ids()

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(token_ids)
            ]
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
            return

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # components prepare insert data + return effective cache_len
        insert_params = InsertParams(
            prev_prefix_len=req.cache_protected_len,
            chunked=chunked,
            priority=getattr(req, "priority", 0) or 0,
            # This request keeps using its KV after the insert; never dedupe
            # it against load-back-pending slots the match below will reject
            # (restores the new_prefix_len <= len(new_indices) assert).
            stop_at_load_back_pending=True,
        )
        effective_cache_len = len(token_ids)
        for comp in self._components_tuple:
            cl = comp.prepare_for_caching_req(
                req=req,
                insert_params=insert_params,
                token_ids_len=len(token_ids),
                is_finished=False,
            )
            if cl is not None:
                effective_cache_len = min(effective_cache_len, cl)

        radix_key = RadixKey(
            token_ids[:effective_cache_len],
            req.extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=req.cache_salt,
        )

        if envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.get():
            # The frontier lands a page below page_floor(pre_len + 1), which has to
            # be where the insert stops, or the leaf it creates keeps less than a
            # sliding window of live SWA and the match after the insert rejects it.
            # The insert stops at page_floor(len(radix_key)), and a bigram key is
            # one shorter than the tokens it spans, so measure the key.
            for comp in self._components_tuple:
                comp.free_out_of_window_slots(req, len(radix_key) - 1, insert_params)

        rows = cpu_rows_of(self.req_to_token_pool)  # Q4 (None when off)
        if effective_cache_len <= 0:
            req.prefix_indices = tag_row_slice(
                kv_indices_orig.to(dtype=torch.int64, copy=True),
                rows,
                req.req_pool_idx,
                0,
            )
            for comp in self._components_tuple:
                comp.cleanup_after_caching_req(
                    req, is_finished=False, insert_params=insert_params
                )
            return

        kv_indices = kv_indices_orig[:effective_cache_len]

        radix_key = radix_key.page_aligned(self.page_size)
        page_aligned_len = len(radix_key)
        values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)
        tag_row_slice(values, rows, req.req_pool_idx, 0)

        if self._swa_window and chunked and req.last_node is not None:
            # Deferred-by-one-stash write-through: chunked
            # inserts never back up, so back up the chain ending at the
            # previous stash now. Its forward was processed an iteration ago,
            # while this stash's chunk may still be running.
            if self._deferred_chunk_backups is not None:
                # The overlap loop runs it right after this iteration's launch
                # (run_deferred_chunk_backups). The insert below only adds a
                # leaf under this node and splits keep its id on the suffix,
                # so its chain still ends at the previous stash then.
                self._deferred_chunk_backups.append(req.last_node)
            else:
                self._backup_unbacked_chain(req.last_node)

        insert_params.key = radix_key
        insert_params.value = values
        result = self.insert(insert_params)

        if (
            result.stopped_at_load_back_pending
            and result.prefix_len < req.cache_protected_len
        ):
            # The walk stopped inside this request's own protected prefix: the
            # nodes it adopted through init_load_back are still H2D
            # load-back-pending (the ack is drained by check_hicache_events,
            # which runs AFTER process_prefill_chunk in the same scheduler
            # iteration, so a chunk boundary right after admission always
            # sees the pin). The device match below would reject those nodes
            # and return fewer indices than cache_protected_len. Nothing was
            # inserted (stop is in front of the first pending node, before
            # touch/split, and component commits were skipped), so defer:
            # keep the request's own KV and locks untouched and re-offer the
            # whole fill at the next chunk boundary / cache_finished_req.
            # The decision depends only on tree state mutated at lockstep
            # points, so every TP rank defers together.
            logger.debug(
                "cache_unfinished_req deferred for %s: own prefix load-back "
                "pending (walked=%d, protected=%d)",
                req.rid,
                result.prefix_len,
                req.cache_protected_len,
            )
            req.prefix_indices = tag_row_slice(
                kv_indices_orig.to(dtype=torch.int64, copy=True),
                rows,
                req.req_pool_idx,
                0,
            )
            for comp in self._components_tuple:
                comp.cleanup_after_caching_req(
                    req,
                    is_finished=False,
                    insert_result=result,
                    insert_params=insert_params,
                )
            return

        # Match prefix. SWA insertion retains one extra window before the
        # page-aligned boundary, so the normal match remains safe to repoint.
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key, req=req))
        if self.kv_sessions is not None:
            self.kv_sessions.refresh_request(req, node_id=result.last_device_node)
        new_indices = match_result.device_indices
        new_last_node = match_result.last_device_node
        new_prefix_len = result.prefix_len
        assert (
            req.cache_protected_len <= len(new_indices) + self.page_size - 1
        ), f"{req.cache_protected_len=}, {len(new_indices)=}, {page_aligned_len=}"
        assert new_prefix_len <= len(
            new_indices
        ), f"{new_prefix_len=}, {len(new_indices)=}"
        row_write_kwargs = {}
        if (
            getattr(self.req_to_token_pool, "_track_row_writes", False)
            and new_prefix_len <= req.cache_protected_len
        ):
            # The walk met no existing node past the protected prefix, so no
            # page of this range was deduplicated (freed and repointed): the
            # rows already hold these values. Tells the PD prefill page-id
            # reader (prefill_loop_unserialize.py, 6b) that its read is valid.
            row_write_kwargs["values_unchanged"] = True
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            slice_tagged(new_indices, req.cache_protected_len, None, self.page_size),
            **row_write_kwargs,
        )

        lag_lock = None
        if self._swa_window and chunked:
            # Lag lock (I6): the chunk launched before this stash still reads
            # the window below it, so keep the replaced lock until the next
            # relock and release the one before it instead.
            if req.last_node is not None:
                lag_lock = SwaLagLock(
                    req.last_node,
                    DecLockRefParams(
                        swa_uuid_for_lock=req.swa_uuid_for_lock,
                        skip_lock_node_ids=req.skip_lock_node_ids,
                        skip_lock_components=getattr(req, "skip_lock_components", ()),
                    ),
                )
        else:
            self._dec_req_lock(req)
        # Opt-in: leave the matched-prefix mamba evictable during decode (it is
        # already COW'd to the request's own slot, never read from this node again).
        # Safe only because any future COW source is the COWing request's own
        # admission-locked last_node (recorded only if still present, locked before
        # the next alloc) -- not this evictable node. A scheduler that matched a
        # whole batch before locking would break that. Off = original full lock.
        skip_lock_components = (
            (ComponentType.MAMBA,)
            if envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.get()
            else ()
        )
        lock_result = self.inc_lock_ref(
            new_last_node, skip_lock_components=skip_lock_components
        )
        if self._swa_window:
            self._release_swa_lag_lock(req)
            req.swa_lag_lock = lag_lock

        # Update req fields
        if len(new_indices) < len(kv_indices_orig):
            req.prefix_indices = cat_tagged(
                [
                    new_indices,
                    tag_row_slice(
                        kv_indices_orig[len(new_indices) :],
                        rows,
                        req.req_pool_idx,
                        len(new_indices),
                    ),
                ],
                self.page_size,
                "cache_unfinished_req prefix",
            )
        else:
            req.prefix_indices = new_indices
        req.cache_protected_len = len(new_indices)
        req.last_node = new_last_node
        req.swa_uuid_for_lock = lock_result.swa_uuid_for_lock
        # carry the skip set so this node's dec releases only what we locked
        req.skip_lock_node_ids = lock_result.skip_lock_node_ids
        req.skip_lock_components = lock_result.skip_lock_components
        # The rematch acquired a new SWA prefix lock.
        req.swa_prefix_lock_released = False

        # cleanup
        for comp in self._components_tuple:
            comp.cleanup_after_caching_req(
                req,
                is_finished=False,
                insert_result=result,
                insert_params=insert_params,
            )

    # ---- Internal Helpers ----

    @timed_phase("apply_cache_actions")
    def _apply_cache_actions(
        self, actions: list[CacheAction | ComponentAction]
    ) -> None:
        # Apply and consume one at a time: a spent list cannot be double-applied.
        actions.reverse()
        try:
            while actions:
                self._apply_cache_action(actions.pop())
        finally:
            actions.reverse()

    def _apply_cache_action(self, action: CacheAction | ComponentAction) -> None:
        # Component actions route to their component class; the rest are
        # cache-owned and handled here by type.
        if isinstance(action, ComponentAction):
            self.components[action.component_type].apply_component_action(action)
        elif isinstance(action, ReplaceWriteThroughOnNodeSplit):
            self._replace_pending_write_through_node(
                action.ack_id,
                action.old_node_id,
                [action.new_node_id, action.new_child_node_id],
            )
        elif isinstance(action, FreeDeviceKV):
            # tree values are page-aligned copies of a kv row: page-exact segments
            for indices in action.indices:
                self.token_to_kv_pool_allocator.free_segment(indices, start_pos=0)
        elif isinstance(action, FreeDeviceKVFullOnly):
            for indices in action.indices:
                self.token_to_kv_pool_allocator.free_full(indices)
        elif isinstance(action, BackupKV):
            self._execute_and_commit_kv_backup(action)
        else:
            raise AssertionError(f"unhandled CacheAction: {type(action).__name__}")

    def _drain_device_frees(
        self, device_frees: dict[ComponentType, list[torch.Tensor]]
    ) -> None:
        # Free per component device slots, consuming each entry as it frees.
        for ct in list(device_frees):
            self._apply_cache_action(
                FreeComponentDeviceSlot(device_frees.pop(ct), component_type=ct)
            )

    def _drain_host_frees(
        self, host_frees: dict[ComponentType, list[torch.Tensor]]
    ) -> None:
        # Free per component host-pool slots, consuming each entry as it frees.
        for ct in list(host_frees):
            self.components[ct].free_host_values(host_frees.pop(ct))

    @timed_phase("evict_host")
    def evict_host(
        self,
        num_tokens: int,
        component_type: ComponentType = BASE_COMPONENT_TYPE,
        exclude_rid: Optional[str] = None,
    ) -> int:
        """Evict host resources for a specific component to free host pool space.

        With the queued-prefix guard (disaggregation/queued_prefix_guard.py), a
        shortfall releases the guard's newest host locks and evicts again
        (never blocks a backup or fails a pull); ``exclude_rid`` is the request
        whose pull allocation asks, so its own lock is kept."""
        if self.host_memory_mode == "buffer_only":
            # The tree never holds host values in buffer mode, and staging
            # is operation-owned (freed at each ack): nothing is evictable.
            return 0
        result = self.tree_core.drive_host_eviction(component_type, num_tokens)
        self._free_values(result.device_frees, result.host_frees)
        if result.duplicate_reclaimed_tokens and self.metrics_collector is not None:
            self.metrics_collector.increment_host_duplicate_reclaimed_tokens(
                result.duplicate_reclaimed_tokens,
                pool=_COMPONENT_POOL_LABEL[BASE_COMPONENT_TYPE],
            )
        freed = result.tracker.get(component_type, 0)
        guard = getattr(self, "queued_prefix_guard", None)
        if (
            guard is not None
            and freed < num_tokens
            and component_type == BASE_COMPONENT_TYPE
        ):
            freed = guard.relieve_host_pressure(
                num_tokens, freed, exclude_rid, component_type
            )
        return freed

    # ---- Decode retraction ----

    def supports_retraction_backup(self) -> bool:
        if self.cache_controller is None or self.host_pool_group is None:
            return False
        if self.supports_mamba():
            return False

        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(kv_cache, SWAKVPool):
            return (
                self.supports_swa()
                and {
                    PoolName.KV,
                    PoolName.SWA,
                }
                <= self.host_pool_group.entry_map.keys()
            )
        return isinstance(kv_cache, MHATokenToKVPool) and (
            PoolName.KV in self.host_pool_group.entry_map
        )

    def validate_retraction_host_capacity(self) -> None:
        if not self.supports_retraction_backup():
            raise ValueError(
                "--disaggregation-decode-retraction-backup=host_pool requires "
                "an MHA or hybrid-SWA HiCache host stack."
            )

        for spec in self.sidecar_pool_specs:
            source_size = self.host_pool_group.entry_map[
                spec.indices_from_pool
            ].host_pool.logical_size
            sidecar_size = self.host_pool_group.entry_map[
                spec.pool_name
            ].host_pool.logical_size
            if sidecar_size < source_size:
                raise ValueError(
                    "Retraction sidecar host pool is smaller than its index source: "
                    f"pool={spec.pool_name}, host_slots={sidecar_size}, "
                    f"source={spec.indices_from_pool}, source_slots={source_size}."
                )

    @staticmethod
    def _pad_retraction_indices(indices: torch.Tensor, page_size: int) -> torch.Tensor:
        aligned_len = ceil_align(len(indices), page_size)
        if aligned_len == len(indices):
            return indices
        tail = indices[-1] + torch.arange(
            1,
            aligned_len - len(indices) + 1,
            dtype=torch.int64,
            device=indices.device,
        )
        return torch.cat([indices, tail])

    def _retraction_device_transfers(
        self, req: Req
    ) -> tuple[torch.Tensor, list[PoolTransfer]]:
        num_tokens = req.seqlen - 1
        full_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :num_tokens
        ].to(torch.int64)
        full_indices = self._pad_retraction_indices(full_indices, self.page_size)

        component_transfers: dict[ComponentType, list[PoolTransfer]] = {}
        if self.supports_swa():
            kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
            assert self.sliding_window_size is not None
            window_start = (
                0
                if self.dsv4_uniform_cache_lifecycle
                else max(0, num_tokens - self.sliding_window_size)
            )
            window_start = window_start // self.page_size * self.page_size
            window_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, window_start:num_tokens
            ].to(torch.int64)
            swa_indices = kv_cache.translate_loc_from_full_to_swa(window_indices)
            assert bool(
                (swa_indices > 0).all()
            ), f"unmapped SWA window positions for request {req.rid}"
            component_transfers[ComponentType.SWA] = [
                PoolTransfer(
                    name=PoolName.SWA,
                    device_indices=self._pad_retraction_indices(
                        swa_indices, self.page_size
                    ),
                )
            ]

        kv_transfer = PoolTransfer(name=PoolName.KV, device_indices=full_indices)
        extra_transfers = [
            transfer
            for transfers in component_transfers.values()
            for transfer in transfers
        ]
        extra_transfers.extend(
            self._build_sidecar_transfers(
                CacheTransferPhase.BACKUP_HOST,
                kv_transfer,
                component_transfers,
            )
        )
        return full_indices, extra_transfers

    def _reclaim_retraction_host(self, num_tokens: int) -> int:
        if self.disable:
            return 0
        return self.evict_host(num_tokens)

    def retraction_backup(self, req: Req) -> Optional[RetractionBackup]:
        """Back up device KV to the host pool; None when it cannot fit after reclaim."""
        assert req.seqlen > 1

        device_indices, extra_transfers = self._retraction_device_transfers(req)
        host_indices = self.host_pool_group.alloc(len(device_indices))
        if host_indices is None:
            self._reclaim_retraction_host(len(device_indices))
            host_indices = self.host_pool_group.alloc(len(device_indices))
        if host_indices is None:
            return None

        resolved = self.host_pool_group.resolve_host_transfers(
            extra_transfers or None,
            primary_device_indices=device_indices,
            primary_host_indices=host_indices,
        )
        if resolved is None and extra_transfers:
            self.host_pool_group.free(host_indices)
            return None

        backup = RetractionBackup(
            host_indices=host_indices,
            pool_transfers=[replace(x, device_indices=None) for x in resolved or []]
            or None,
        )
        operation = CacheOperation(
            host_indices,
            device_indices,
            node_id=-1,
            pool_transfers=resolved,
        )
        try:
            write_host, write_device, write_pools = (
                self.cache_controller._move_write_operation(operation)
            )
            completion = self.cache_controller.l2_transfer_engine.submit_device_to_host(
                self.cache_controller._l2_transfers(
                    write_host, write_device, write_pools
                )
            )
            completion.finish_event.synchronize()
        except Exception:
            self.retraction_discard(backup)
            raise
        return backup

    def retraction_restore(self, req: Req, backup: RetractionBackup) -> None:
        device_indices, current_transfers = self._retraction_device_transfers(req)
        assert len(backup.host_indices) == len(device_indices), (
            f"Host backup has {len(backup.host_indices)} slots, but restore has "
            f"{len(device_indices)}"
        )

        current_by_name = {transfer.name: transfer for transfer in current_transfers}
        saved_by_name = {
            transfer.name: transfer for transfer in backup.pool_transfers or []
        }
        assert current_by_name.keys() == saved_by_name.keys(), (
            f"Host backup pools {set(saved_by_name)} do not match restore pools "
            f"{set(current_by_name)}"
        )
        restored_transfers = [
            replace(
                saved,
                device_indices=current_by_name[name].device_indices,
            )
            for name, saved in saved_by_name.items()
        ]
        resolved = self.cache_controller._resolve_device_transfers(
            restored_transfers or None,
            kv_device_indices=device_indices,
            kv_host_indices=backup.host_indices,
        )
        assert resolved is not None or not restored_transfers

        operation = CacheOperation(
            backup.host_indices,
            device_indices,
            node_id=-1,
            pool_transfers=resolved,
        )
        load_host, load_device, load_pools = self.cache_controller._move_op_indices(
            operation
        )
        completion = self.cache_controller.l2_transfer_engine.submit_host_to_device(
            self.cache_controller._l2_load_transfers(
                load_host, load_device, load_pools
            ),
            layer_num=self.cache_controller.layer_num,
        )
        completion.finish_event.synchronize()
        self.retraction_discard(backup)

    def retraction_discard(self, backup: RetractionBackup) -> None:
        self.host_pool_group.free(backup.host_indices)
        self.host_pool_group.release_transfers(backup.pool_transfers)

    # ---- HiCache: Backup / LoadBack ----

    @timed_phase("tree.kv_backup_submit")
    def _execute_and_commit_kv_backup(
        self, action: BackupKV, write_back: bool = False
    ) -> int:
        """Run a backup action top-down, stopping at the first failed backup."""
        if self.buffer_pipeline is not None:
            # Buffer mode bypasses the host-backup contiguity below: nothing
            # is ever host-backuped here. Contiguity comes from end-to-end
            # FIFO ordering instead (BackupKV chains are parent-before-child
            # and every pipeline stage drains in order).
            for node_id in action.node_ids:
                self.buffer_pipeline.enqueue_backup_intent(
                    self.tree_core.node_by_id(node_id)
                )
            return 0
        if action.pre_evict and not write_back:
            self._pre_evict_for_backup(action.node_ids)
        written = 0
        for node_id in action.node_ids:
            device_value, comp_xfers = self.tree_core.build_backup_spec(node_id)
            # Overlapping chain actions may revisit nodes with Full KV already
            # backed up. Skip only when no transfer remains.
            if device_value.numel() == 0 and not comp_xfers:
                continue
            sidecar_xfers = self._build_backup_sidecar(device_value, comp_xfers)
            # _execute_kv_backup may evict host memory; the ancestors must keep
            # their host copies meanwhile (write-through ancestry).
            guard = self.tree_core.begin_backup_guard(node_id)
            try:
                host_indices = self._execute_kv_backup(
                    node_id, device_value, comp_xfers, sidecar_xfers
                )
            finally:
                self.tree_core.end_backup_guard(guard)
            if host_indices is None:
                if self.metrics_collector is not None:
                    self.metrics_collector.increment_backup_failed(
                        1,
                        len(device_value),
                        pool=_COMPONENT_POOL_LABEL[BASE_COMPONENT_TYPE],
                    )
                if self._swa_window and not write_back:
                    # The node stays (FD,SD): its SWA cannot be demoted until
                    # a retry backs it up. The chain may end at a chunk whose
                    # forward is not processed yet, so wait for the next result.
                    self._enqueue_swa_rebackup(
                        action.node_ids[-1], after_next_result=True
                    )
                return 0
            self.tree_core.commit_backup(node_id, host_indices, comp_xfers)
            lock_params = None
            if not write_back:
                lock_params = self.inc_lock_ref(node_id).to_dec_params()
            self._track_write_through_node(node_id, lock_params)
            written = len(host_indices)
        return written

    def _pre_evict_for_backup(self, node_ids: list[NodeId]) -> None:
        """BackupKV.pre_evict (a chain of the chunked release insert): make host
        room for all of ``node_ids`` with one host eviction, so a long chain
        does not rebuild the eviction candidates and heap once per node.

        The victims and their order equal those of the per-node calls in
        _execute_kv_backup: no heap key changes between those calls, a chain
        node already written is write-pending (never a candidate), and the last
        per-node call stops at the same running total of freed tokens. The
        guard of the last node (a parent chain, root first) pins every node
        that a per-node guard pins. If the host cannot free enough, the
        per-node calls still back up the nodes that fit, as before."""
        need = sum(
            len(self.tree_core.build_backup_spec(node_id)[0]) for node_id in node_ids
        )
        short = need - self.cache_controller.mem_pool_host.available_size()
        if short <= 0:
            return
        guard = self.tree_core.begin_backup_guard(node_ids[-1])
        try:
            self.evict_host(short)
        finally:
            self.tree_core.end_backup_guard(guard)

    def _build_backup_sidecar(self, device_value, comp_xfers):
        """Gather sidecar transfer spec."""
        kv_xfer = PoolTransfer(name=PoolName.KV, device_indices=device_value)
        return self._build_sidecar_transfers(
            CacheTransferPhase.BACKUP_HOST, kv_xfer, comp_xfers
        )

    def _execute_kv_backup(self, node_id, device_value, comp_xfers, sidecar_xfers):
        """Execute Backup action."""
        kv_tokens = len(device_value)
        host_avail = self.cache_controller.mem_pool_host.available_size()
        if host_avail < kv_tokens:
            needed = kv_tokens - host_avail
            if self.evict_host(needed) < needed:
                return None
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        return self.cache_controller.write(
            device_value, node_id=node_id, extra_pools=aux_xfers or None
        )

    def _track_write_through_node(
        self,
        node_id: NodeId,
        lock_params: Optional[DecLockRefParams],
    ) -> None:
        self.tree_core.mark_write_through_pending(node_id)
        self.ongoing_write_through[node_id] = _OngoingWriteThrough(
            node_id, lock_params, [node_id]
        )

    def _replace_pending_write_through_node(
        self, ack_id: int, old_node_id: NodeId, new_node_ids: list[NodeId]
    ) -> None:
        pending = self.ongoing_write_through.get(ack_id)
        if pending is None:
            return

        lock_node_id, lock_params, publish_node_ids = pending
        updated_node_ids = []
        replaced = False
        for node_id in publish_node_ids:
            if node_id == old_node_id:
                updated_node_ids.extend(new_node_ids)
                replaced = True
            else:
                updated_node_ids.append(node_id)

        if not replaced:
            return

        self.ongoing_write_through[ack_id] = _OngoingWriteThrough(
            lock_node_id,
            lock_params,
            updated_node_ids,
        )

    def _record_prefetch_store(self, insert_result: InsertResult) -> None:
        """Advertise the new host suffix after all prefetch commits finish."""
        remaining = insert_result.total_len - insert_result.prefix_len
        kv_events = self.tree_core.kv_events
        if (
            # With a D-role gate every rank walks, so its flags stay equal.
            not (kv_events.enabled or getattr(kv_events, "d_role_gate", None))
            or insert_result.inserted_host_node is None
            or remaining <= 0
        ):
            return

        # Component commits can split the new suffix. The original leaf id
        # still identifies its end; walk only the new fragments, not overlaps.
        node = self.tree_core.node_by_id(insert_result.inserted_host_node)
        fragments = []
        while remaining > 0:
            assert node.parent is not None and 0 < len(node.key) <= remaining
            assert node.backuped
            fragments.append(node)
            remaining -= len(node.key)
            node = node.parent

        # Publish parent before child, using the recorder's namespace-aware
        # hashes. An in-flight write-through publishes through its own ack.
        # Window residency: like the write-through ack, advertise the CPU tier
        # only for nodes whose host copy includes their full SWA (I3; a
        # prefetch commits ALL_PAGES SWA, so this is defensive).
        window = self._swa_window
        for node in reversed(fragments):
            if node.write_through_pending_id is None and (
                not window or self.tree_core._swa_host_complete(node)
            ):
                self.tree_core.kv_events.record_store(node, medium=StorageMedium.CPU)

    def _peer_push_prefetch_commit(self, insert_result: InsertResult) -> None:
        """Publish the committed tree locations, not the incoming buffers.

        SWA commit can split nodes and restore a window across an existing
        prefix; duplicate incoming slots have already been released. The
        write-through publisher also covers SWA-anchored state pools.
        """
        if insert_result.inserted_host_node is None:
            return
        node = self.tree_core.node_by_id(insert_result.inserted_host_node)
        remaining = insert_result.total_len
        while node.parent is not None and remaining > 0:
            # Existing prefix nodes may still be receiving a D2H backup.
            # Their completion path will publish them when the bytes are ready.
            if node.write_through_pending_id is None:
                self.tree_core._peer_push_servable_node(node, from_prefetch=True)
            remaining -= len(node.key)
            node = node.parent

    def _finish_write_through_ack(self, ack_id: int) -> None:
        if self.buffer_pipeline is not None:
            self.buffer_pipeline.finish_backup_ack(ack_id)
            return

        lock_node_id, lock_params, publish_node_ids = self.ongoing_write_through.pop(
            ack_id
        )
        self.tree_core.finish_write_through(publish_node_ids, ack_id)
        self._host_returns_dirty = True
        if lock_params is not None:
            self.dec_lock_ref(lock_node_id, lock_params)
        if self.enable_storage:
            # Back up each fragment: after a split, lock_node only holds the
            # suffix; the prefix fragment must be persisted as well.
            for node_id in publish_node_ids:
                self.write_backup_storage(node_id)

    def bwp_chunk_end(
        self, node_id: NodeId, max_tokens: int, *, split: bool = False
    ) -> tuple[NodeId, int, int]:
        """Chunked bootstrap prefetch (SGLANG_BWP_CHUNKED): the end of the
        first <= ``max_tokens`` chunk of what ``load_back(node_id)`` would
        load (``UnifiedTreeCore.host_load_chunk_end``)."""
        return self.tree_core.host_load_chunk_end(node_id, max_tokens, split=split)

    @timed_phase("load_back")
    def load_back(
        self,
        node_id: NodeId,
        mem_quota: Optional[int] = None,
        req=None,
        active_components: Optional[Sequence[ComponentType]] = None,
        prefer_clean: bool = True,
    ) -> bool:
        """Load evicted KV data from host back to device (H→D).

        ``prefer_clean`` only matters with M4's load quarantine (see
        ``_load_target_available_size``): False for a background load (the
        bootstrap-window prefetch) that may take dirty free pages and wait for
        their forward instead of evicting cached KV."""
        if self.cache_controller is None:
            return False

        active_component_drivers = self._resolve_active_components(
            active_components, for_match=True
        )
        active_component_types = tuple(
            comp.component_type for comp in active_component_drivers
        )
        skipped_components = tuple(
            ct for ct in self.tree_components if ct not in active_component_types
        )
        if self._swa_window and ComponentType.SWA in active_component_types:
            # Trim the host-only SWA node to the charged window before any lock
            # or transfer references it; the trimmed node
            # keeps its id, so node_id still ends the path.
            split_action = self.tree_core.split_for_swa_window_load(node_id)
            if split_action is not None:
                self._apply_cache_action(split_action)
        host_anchor_params = self.inc_host_lock_ref(node_id).to_dec_params()

        # Lock the path before building transfers (the aux build can evict).
        result = self.inc_lock_ref(node_id, skip_lock_components=skipped_components)
        ancestor_lock_params = result.to_dec_params()

        # Let each component pre-allocate per-request state for the load-back;
        # the finally below lets components recover it unless the load succeeds.
        preps: dict[ComponentType, PrepareLoadBackResult] = {
            comp.component_type: comp.prepare_load_back(node_id, req=req)
            for comp in active_component_drivers
        }
        success = False
        try:
            success = self._load_back_transfers(
                node_id=node_id,
                mem_quota=mem_quota,
                req=req,
                result=result,
                ancestor_lock_params=ancestor_lock_params,
                host_anchor_params=host_anchor_params,
                active_components=active_component_types,
                prefer_clean=prefer_clean,
            )
            return success
        finally:
            for comp in active_component_drivers:
                comp.finalize_load_back(req, preps[comp.component_type], success)

    def _load_target_available_size(self, prefer_clean: bool = True) -> int:
        """Pages a load can take before it evicts. With the M4 load quarantine
        a foreground load counts only clean pages: dirty ones stay for forward
        allocations, and a load that took them would wait for their forward
        (the eviction of cold, clean victims is what lets it overlap). A
        background load (``prefer_clean=False``: the bootstrap-window prefetch,
        which nothing waits for until admission) counts every free page, as
        without the quarantine: the allocator still hands out clean pages
        first, and a dirty one costs that op a wait for its forward instead of
        evicting cached KV."""
        from decoupled_engine.srt.mem_cache.allocator.cpu_paged import cpu_paged_allocator

        alloc = cpu_paged_allocator(self.token_to_kv_pool_allocator)
        if prefer_clean and alloc is not None and alloc._clock is not None:
            return alloc.clean_available_size()
        return self._component_available_size(ComponentType.FULL)

    def _count_background_load(self, kv_tokens: int, avail: int) -> None:
        """M4 page-book stats for a background load under the quarantine:
        ``load_bg_beyond_clean`` counts the loads that needed more than the
        clean pages (with clean-only sizing they would have evicted cached KV),
        ``load_bg_evict_tokens`` the tokens a background load still evicts."""
        from decoupled_engine.srt.mem_cache.allocator.cpu_paged import cpu_paged_allocator
        from decoupled_engine.srt.mem_cache.cpu_page_book import STATS

        alloc = cpu_paged_allocator(self.token_to_kv_pool_allocator)
        if alloc is None or alloc._clock is None:
            return
        if alloc.clean_available_size() < kv_tokens:
            STATS["load_bg_beyond_clean"] += 1
        if avail < kv_tokens:
            STATS["load_bg_evict_tokens"] += kv_tokens - avail

    @timed_phase("load_back_transfers")
    def _load_back_transfers(
        self,
        *,
        node_id: NodeId,
        mem_quota: Optional[int],
        req,
        result: IncLockRefResult,
        ancestor_lock_params: DecLockRefParams,
        host_anchor_params: DecLockRefParams,
        active_components: tuple[ComponentType, ...],
        prefer_clean: bool = True,
    ) -> bool:
        # Build the KV + per-component aux transfers.
        kv_xfer, comp_xfers = self.tree_core.build_load_back_spec(
            node_id, req=req, active_components=active_components
        )
        kv_tokens = len(kv_xfer.host_indices)
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.LOAD_BACK, kv_xfer, comp_xfers
        )

        # Skip if there is nothing to load, or if the Full-KV transfer is too
        # small / exceeds memory quota. Aux transfers should still run even
        # when the Full-KV load is skipped by thresholding. max(1, ...): an
        # entirely empty spec (e.g. foreign-pin rejection) must never report
        # success, even at load_back_threshold <= 0.
        if (kv_tokens < max(1, self.load_back_threshold) and not comp_xfers) or (
            mem_quota is not None and kv_tokens > mem_quota + result.delta
        ):
            self.dec_lock_ref(node_id, ancestor_lock_params)
            self.dec_host_lock_ref(node_id, host_anchor_params)
            return False

        avail = self._load_target_available_size(prefer_clean)
        if not prefer_clean:
            self._count_background_load(kv_tokens, avail)
        if avail < kv_tokens:
            needed = kv_tokens - avail
            self.evict_for_alloc(EvictParams(num_tokens=needed))
            if self._component_available_size(ComponentType.FULL) < kv_tokens:
                self.dec_lock_ref(node_id, ancestor_lock_params)
                self.dec_host_lock_ref(node_id, host_anchor_params)
                return False

        # Load H→D
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        device_indices = self.cache_controller.load(
            host_indices=kv_xfer.host_indices,
            node_id=node_id,
            extra_pools=aux_xfers or None,
        )

        self.dec_lock_ref(node_id, ancestor_lock_params)
        if device_indices is None:
            self.dec_host_lock_ref(node_id, host_anchor_params)
            return False
        if self._swa_window:
            swa_tokens = sum(
                len(x.host_indices)
                for x in comp_xfers.get(ComponentType.SWA, ())
                if x.host_indices is not None
            )
            if swa_tokens:
                self._swa_load_back_tokens += swa_tokens
                self._swa_load_back_ops += 1
            if self._swa_copy_verify and aux_xfers:
                self._swa_copy_verify_loads[node_id] = list(aux_xfers)

        # Commit the loaded KV back onto the node + apply its emitted actions.
        self._apply_cache_actions(
            self.tree_core.commit_load_back(
                node_id, device_indices, kv_xfer, comp_xfers
            )
        )

        self.ongoing_load_back[node_id] = _OngoingLoadBack(
            node_id,
            self.inc_lock_ref(
                node_id, skip_lock_components=result.skip_lock_components
            ).to_dec_params(),
            host_anchor_params,
        )

        return True

    def _build_sidecar_transfers(
        self,
        phase: CacheTransferPhase,
        kv_xfer: PoolTransfer,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> list[PoolTransfer]:
        transfers: list[PoolTransfer] = []
        for spec in self.sidecar_pool_specs:
            if spec.indices_from_pool == PoolName.KV:
                indices_source = kv_xfer
            else:
                source_component = {
                    PoolName.SWA: ComponentType.SWA,
                    PoolName.MAMBA: ComponentType.MAMBA,
                }.get(spec.indices_from_pool)
                if source_component is None:
                    raise AssertionError(
                        f"Unsupported sidecar indices source pool "
                        f"{spec.indices_from_pool}."
                    )
                matching_sources = comp_xfers.get(source_component, ())
                if not matching_sources:
                    continue
                indices_source = matching_sources[0]
                if indices_source.name != spec.indices_from_pool:
                    raise AssertionError(
                        f"Sidecar indices source pool {spec.indices_from_pool} "
                        f"resolved to {indices_source.name} during {phase}."
                    )

            indices = (
                indices_source.device_indices
                if phase == CacheTransferPhase.BACKUP_HOST
                else indices_source.host_indices
            )
            defer_kv_sidecar = (
                phase == CacheTransferPhase.PREFETCH
                and spec.indices_from_pool == PoolName.KV
            )
            if (indices is None or len(indices) == 0) and not defer_kv_sidecar:
                continue
            transfers.append(
                PoolTransfer(
                    name=spec.pool_name,
                    keys=indices_source.keys,
                    hit_policy=spec.hit_policy,
                    indices_from_pool=spec.indices_from_pool,
                )
            )
        return transfers

    def write_backup_storage(self, node_id: NodeId) -> None:
        if not self.enable_storage or self.cache_controller is None:
            return
        if skip_noop_storage_backup(self.cache_controller.storage_backend):
            # The backend acknowledges writes without storing (peer_kv): no
            # backup op, backup-thread pass, ack drain or host lock held
            # until the ack. Rank-uniform (class attribute and process env).
            return
        spec = self.tree_core.build_storage_backup_spec(
            node_id, self.hicache_storage_pass_prefix_keys
        )
        if spec is None:
            return

        kv_xfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=spec.host_value,
            keys=spec.hash_value,
        )
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.BACKUP_STORAGE, kv_xfer, spec.comp_xfers
        )
        aux_xfers = [x for xfers in spec.comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)

        operation_id = self.cache_controller.write_storage(
            spec.host_value,
            spec.token_ids,
            spec.hash_value,
            spec.prefix_keys,
            extra_pools=aux_xfers or None,
        )
        self.ongoing_backup[operation_id] = (
            node_id,
            self.inc_host_lock_ref(node_id).to_dec_params(),
        )

    def is_backuped(self, node_id: NodeId) -> bool:
        return self.tree_core.is_backuped(node_id)

    def is_root(self, node_id: NodeId) -> bool:
        return self.tree_core.is_root(node_id)

    def get_last_hash_value(self, node_id: NodeId) -> Optional[str]:
        return self.tree_core.get_last_hash_value(node_id)

    def get_prefix_hash_values(self, node_id: NodeId) -> list[str]:
        return self.tree_core.get_prefix_hash_values(node_id)

    def query_storage_hit_length(
        self,
        last_host_node_id: NodeId,
        new_input_tokens: list[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[list[str]] = None,
    ) -> int:
        """Synchronously probe L3 storage for the reusable prefix length."""
        if (
            not self.enable_storage
            or self.cache_controller is None
            # The probe never carries a kv_pull_hint: a hint-only backend
            # (peer_kv) would return 0 after the all-reduce below anyway.
            or skip_hintless_storage_lookup(self.cache_controller.storage_backend)
            or self.cache_controller.prefetch_rate_limited()
        ):
            return 0

        extra_key, cache_salt = self.tree_core.prefetch_anchor_info(last_host_node_id)
        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=cache_salt,
        ).page_aligned(self.page_size)
        if len(prefetch_key) < self.prefetch_threshold:
            return 0

        from decoupled_engine.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            PrefetchOperation,
        )

        operation = PrefetchOperation(
            "__storage_hit_query__",
            prefetch_key,
            last_hash,
            prefix_keys,
        )
        _, storage_hit_count = self.cache_controller._storage_hit_query(operation)
        storage_hit_count_tensor = torch.tensor(storage_hit_count, dtype=torch.int)
        self._all_reduce_attn_groups(
            storage_hit_count_tensor, torch.distributed.ReduceOp.MIN
        )
        storage_hit_count = storage_hit_count_tensor.item()
        storage_hit_count -= storage_hit_count % self.page_size
        return storage_hit_count

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node_id: NodeId,
        new_input_tokens: list[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[list[str]] = None,
        matched_prefix_tokens: Optional[list[int]] = None,
        extra_key: Optional[str] = None,
        cache_salt: Optional[str] = None,
        kv_pull_hint: Optional[dict] = None,
        kv_session_registration=None,
    ) -> None:
        if not self.enable_storage or self.cache_controller is None:
            return

        buffer_mode = self.host_memory_mode == "buffer_only"
        # Key the span by the request's namespace, not the anchor's (a root
        # anchor has none): a span published under the wrong namespace gets
        # re-owned by the request's own insert (double free).
        anchor_extra_key, anchor_cache_salt = self.tree_core.prefetch_anchor_info(
            last_host_node_id
        )
        assert (anchor_extra_key is None or anchor_extra_key == extra_key) and (
            anchor_cache_salt is None or anchor_cache_salt == cache_salt
        ), (
            f"prefetch anchor namespace {(anchor_extra_key, anchor_cache_salt)} "
            f"!= request namespace {(extra_key, cache_salt)}"
        )
        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=cache_salt,
        ).page_aligned(self.page_size)
        prefetch_length = len(prefetch_key)
        stats = self._prefetch_outcome_stats
        if prefetch_length > 0:
            stats["attempts"] += 1
        if prefetch_length < self.prefetch_threshold:
            if prefetch_length > 0:
                stats["declined_too_short"] += 1
            # A too-short/fully-matched suffix can become a full recompute if
            # the device match evicts while queued; arm the paced retry.
            self._storage_prefetch_missed_rids.add(req_id)
            return
        if not buffer_mode and self.cache_controller.prefetch_rate_limited():
            stats["declined_rate_limited"] += 1
            self._storage_prefetch_missed_rids.add(req_id)
            return
        if req_id in self.ongoing_prefetch or (
            buffer_mode and self.buffer_pipeline.has_staged(req_id)
        ):
            # A fetch (or an unconsumed hold) already exists for this rid;
            # overwriting would leak its staging slots.
            return

        # Buffer mode holds no tree state during the fetch: buffers are
        # operation-owned, so the anchor needs no pin.
        anchor_lock_params = (
            None
            if buffer_mode
            else self.inc_host_lock_ref(last_host_node_id).to_dec_params()
        )
        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        alloc_failed = False
        for ct in self.tree_components:
            if ct == BASE_COMPONENT_TYPE:
                continue
            # Pre-allocate the component's prefetch host buffer so the build stays pure.
            prep = self.components[ct].prepare_prefetch(
                last_host_node_id, prefetch_tokens=len(prefetch_key)
            )
            if prep.alloc_failed:
                alloc_failed = True
                break
            if prep.host_indices is None:
                continue
            transfers = self.tree_core.build_hicache_transfers(
                ct,
                last_host_node_id,
                CacheTransferPhase.PREFETCH,
                token_ids=prefetch_key.token_ids,
                prefetch_tokens=len(prefetch_key),
                last_hash=last_hash,
                host_indices=prep.host_indices,
            )
            if transfers:
                comp_xfers[ct] = transfers
        kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=None)
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.PREFETCH, kv_xfer, comp_xfers
        )
        if alloc_failed:
            # The whole storage fetch is forfeited over one aux staging
            # alloc (e.g. a single SWA window) — count it, or write-burst
            # starvation of the aux pool reads as generic hit-rate loss.
            if (
                self.enable_storage_metrics
                and self.storage_metrics_collector is not None
            ):
                self.storage_metrics_collector.log_prefetch_aux_alloc_failed_tokens(
                    len(prefetch_key)
                )
            self.cache_controller.append_host_mem_release(
                extra_pools=[x for xfers in comp_xfers.values() for x in xfers],
            )
            if anchor_lock_params is not None:
                self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
            # Forfeited over transient staging pressure; retryable.
            self._storage_prefetch_missed_rids.add(req_id)
            return

        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        operation = self.cache_controller.prefetch(
            req_id,
            prefetch_key,
            last_hash,
            prefix_keys,
            extra_pools=aux_xfers or None,
            kv_pull_hint=kv_pull_hint,
        )
        if self.kv_sessions is not None:
            # Snapshot the version token now; req_id alone can be reused later.
            # Proactive returns have an operation ID but no live P request.
            operation._kv_session_registration = (
                kv_session_registration
                if kv_session_registration is not None
                else self.kv_sessions.requests.get(req_id)
            )
            if kv_session_registration is not None:
                # Same-version request retries may replace their output branch
                # without replacing the registration object. Fence that too.
                operation._kv_session_return_key = kv_session_registration.key
        stats["issued"] += 1
        # Snapshots for the L3 miss accounting at the query outcome (the
        # hit/revoke drains): requested span and total prompt length.
        operation.stats_requested_tokens = prefetch_length
        operation.stats_total_tokens = prefetch_length + len(
            matched_prefix_tokens or []
        )
        self.ongoing_prefetch[req_id] = _OngoingPrefetch(
            last_host_node_id,
            prefetch_key,
            None,
            operation,
            anchor_lock_params,
            comp_xfers,
        )
        if buffer_mode:
            self.buffer_pipeline.set_prefix_ctx(
                req_id,
                matched_prefix_tokens,
                extra_key=extra_key,
                cache_salt=cache_salt,
            )
            # Pin the just-matched anchor now: deferred to IO commit it is
            # often already deleted under churn. The IO-commit call remains
            # as the second chance that decides the fetch's fate.
            self.buffer_pipeline.try_lock_anchor(req_id)
        else:
            # Cache mode reserves the requested span up front; buffer mode
            # grants occupancy later at hit-alloc time, sized to the hit.
            self.cache_controller.prefetch_tokens_occupied += len(prefetch_key)

    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation) -> bool:
        if envs.SGLANG_JIT_PULL_LOOKUP_AWARE_HOLD.get():
            return self._prefetch_timeout_lookup_aware(operation)
        return (
            time.monotonic() - operation.start_time
            > self.prefetch_timeout_base
            + len(operation.hash_value) * self.prefetch_timeout_per_page
        )

    def _prefetch_timeout_lookup_aware(self, operation: PrefetchOperation) -> bool:
        """F3: the LOOKUP phase does not count. Until the prefetch thread has
        the TP-synced hit (``hit_known_at``), ``hash_value`` is empty and the
        linear bound would be the bare base (a large span's LOOKUP alone can
        take longer); bound that phase by LOOKUP_PHASE_MAX_S instead. Then the
        base + pages x per-page clock starts at ``hit_known_at``. The caller's
        MAX all-reduce keeps the decision identical on every rank."""
        now = time.monotonic()
        hit_known_at = getattr(operation, "hit_known_at", None)
        if hit_known_at is None:
            return (
                now - operation.start_time
                > envs.SGLANG_JIT_PULL_LOOKUP_PHASE_MAX_S.get()
            )
        return (
            now - hit_known_at
            > self.prefetch_timeout_base
            + len(operation.hash_value) * self.prefetch_timeout_per_page
        )

    @rank_consensus(same_results=True)
    def _can_terminate_prefetch(self, operation: PrefetchOperation) -> bool:
        if self.prefetch_stop_policy == "best_effort":
            return True
        if self.prefetch_stop_policy == "wait_complete":
            return False
        elif self.prefetch_stop_policy == "timeout":
            # Wall-clock time may differ among ranks, all-reduce is needed to ensure
            # all ranks reach the same final result. Otherwise PP/TP ranks will diverge.
            #
            # For TP, if any rank reaches the timeout, the final result is timeout.
            #
            # For PP, PP0 makes the decision and other ranks follow PP0's decision.
            should_terminate = False
            if self.pp_rank == 0:
                should_terminate = self._prefetch_timeout_check_linear_func(operation)
            should_terminate_tensor = torch.tensor(
                int(should_terminate), dtype=torch.int, device="cpu"
            )
            self._all_reduce(should_terminate_tensor, torch.distributed.ReduceOp.MAX)
            return should_terminate_tensor.item() == 1
        else:
            return True

    @rank_consensus(same_params=True, same_results=True)
    def check_prefetch_progress(self, req_id: str) -> bool:
        if req_id not in self.ongoing_prefetch:
            return True

        _, _, _, operation, _, _ = self.ongoing_prefetch[req_id]

        # Determine whether or not we should terminate this prefetch request.
        was_terminated = operation.is_terminated()
        should_terminate = was_terminated or self._can_terminate_prefetch(operation)

        if not should_terminate:
            return False

        base = getattr(self, "prefetch_timeout_base", None)
        start = getattr(operation, "start_time", None)
        if (
            isinstance(base, (int, float))
            and isinstance(start, (int, float))
            and time.monotonic() - start < base
        ):
            # Numbers only: mocks of the cache or the op skip the note.
            self._note_early_stop(operation, was_terminated)
        self.cache_controller.terminate_prefetch(operation)
        if operation.host_indices is None:
            self._storage_prefetch_missed_rids.add(req_id)
            self.revoke_pending_prefetch(req_id)
        else:
            self._handle_prefetch_result(operation)
        return True

    def _note_early_stop(self, operation, was_terminated: bool) -> None:
        """Pull-miss causes 2.3 (ii): a prefetch stopped younger than the
        timeout base, which the timeout cannot do. Count it (prefetch outcome
        stats ``early_stop``) and log what decided it. Rank-local, no
        collective; the local verdict is evaluated again for the log."""
        count = self._count_prefetch_outcome("early_stop")
        local = None
        if getattr(self, "pp_rank", 0) == 0:
            try:
                local = bool(self._prefetch_timeout_check_linear_func(operation))
            except Exception:
                pass
        dist_ready = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        logger.info(
            "HiCache prefetch early stop req=%s rank=%d age_ms=%.0f "
            "was_terminated=%s local_timeout=%s reduced=%s policy=%s hit_pages=%d "
            "count=%s prefetch_attempt_id=%s",
            getattr(operation, "request_id", None),
            torch.distributed.get_rank() if dist_ready else -1,
            (time.monotonic() - operation.start_time) * 1e3,
            was_terminated,
            local,
            None if was_terminated else True,
            getattr(self, "prefetch_stop_policy", None),
            len(getattr(operation, "hash_value", None) or ()),
            count,
            json.dumps(getattr(operation, "prefetch_attempt_id", None)),
        )

    def _count_prefetch_outcome(self, key: str) -> Optional[int]:
        stats = getattr(self, "_prefetch_outcome_stats", None)
        if stats is None:  # a partial cache object (tests)
            return None
        stats[key] = stats.get(key, 0) + 1
        return stats[key]

    def _note_hit_drop(self, reason: str, operation, **detail) -> None:
        """Pull-miss causes 2.3 (iii): a drained storage hit that ends
        without a read. Count it by reason (prefetch outcome stats
        ``hit_drop_<reason>``) and log it. Rank-local, no collective. Reads
        the operation defensively (partial stand-ins in tests)."""
        count = self._count_prefetch_outcome("hit_drop_" + reason)
        logger.info(
            "HiCache prefetch hit dropped reason=%s req=%s hit_tokens=%s count=%s "
            "prefetch_attempt_id=%s%s",
            reason,
            getattr(operation, "request_id", None),
            getattr(operation, "storage_hit_count", None),
            count,
            json.dumps(getattr(operation, "prefetch_attempt_id", None)),
            "".join(f" {k}={v}" for k, v in detail.items()),
        )

    def _prefetch_trace_args(self, operation: PrefetchOperation) -> tuple:
        # These scheduler-owned statistics are assigned after enqueueing. Read
        # them only at the scheduler's outcome, never in the first LOOKUP.
        requested = getattr(operation, "stats_requested_tokens", None)
        total = getattr(operation, "stats_total_tokens", None)
        local_prefix = (
            total - requested
            if (
                type(total) is int
                and type(requested) is int
                and requested == len(operation.token_ids)
                and total >= requested
            )
            else None
        )
        values = (
            getattr(operation, "prefetch_attempt_id", None),
            getattr(operation, "endpoint_round", None),
            getattr(operation, "initial_query_pages", None),
            local_prefix,
            operation.storage_hit_count // self.page_size,
        )
        # Preserve string IDs (including all-digit UUIDs) and unknown nulls
        # when the existing text log is parsed as structured key=value fields.
        return tuple(json.dumps(value) for value in values)

    def _handle_prefetch_result(self, operation: PrefetchOperation) -> None:
        # This function **owns**:
        # - host_indices[0 : completed_tokens]
        # - sidecar pool hits if operation.pool_transfers_done is true
        #
        # That is, when this function returns the host memory referenced must be inserted
        # into the radix tree or released to pool.

        req_id = operation.request_id
        completed_tokens = operation.completed_tokens
        hash_value = operation.hash_value

        (
            last_host_node_id,
            prefetch_key,
            host_indices,
            _,
            anchor_lock_params,
            comp_xfers,
        ) = self.ongoing_prefetch[req_id]

        # All PP/TP ranks will get the same `min_completed_tokens`, because `completed_tokens`
        # and `pool_hits` in their operations are same.  No need to sync cross-rank here.
        if not self._check_hybrid_prefetch_result(
            req_id,
            operation,
            completed_tokens,
            hash_value,
            host_indices,
            last_host_node_id,
            anchor_lock_params,
            prefetch_key,
        ):
            # Hybrid all-or-nothing check failed; result already discarded.
            self._peer_demote_discard(req_id, "hybrid_discard")
            return

        if self.buffer_pipeline is not None:
            # No graft: release the rank-local tail beyond the synced usable
            # length, then park the bounce for admission-time consumption.
            # No insert into this tree, so no commit-gated demote.
            self._peer_demote_discard(req_id, "buffer_mode")
            return self.buffer_pipeline.stage_completed_prefetch(
                req_id, completed_tokens, hash_value
            )

        fetched_key = prefetch_key[:completed_tokens]
        insert_result = self.tree_core.insert_host(
            last_host_node_id,
            fetched_key,
            host_indices[:completed_tokens],
            hash_value[: completed_tokens // self.page_size],
        )

        # Apply the host-insert walk's actions before the transfer commit.
        self._apply_cache_actions(insert_result.cache_actions)

        if insert_result.host_insert_dropped:
            self.cache_controller.append_host_mem_release(
                host_indices=host_indices[:completed_tokens],
                extra_pools=[x for xfers in comp_xfers.values() for x in xfers],
            )
            loaded_from_storage = 0
            self._peer_demote_discard(req_id, "insert_dropped")
        else:
            commit_actions: list[CacheAction | ComponentAction] = []
            self.tree_core.commit_hicache_transfers(
                last_host_node_id,
                CacheTransferPhase.PREFETCH,
                comp_xfers,
                cache_actions=commit_actions,
                insert_result=insert_result,
                pool_storage_result=operation.pool_storage_result,
            )
            self._apply_cache_actions(commit_actions)
            # The commit emits via commit_actions only; the walk's were applied above.
            assert not insert_result.cache_actions

            self.cache_controller.mem_pool_host.free(
                host_indices[: insert_result.prefix_len]
            )
            self._record_prefetch_store(insert_result)
            # No CPU recency refresh here: each TP rank draws recency from its
            # own process-local counter, and commit-time draws diverged the
            # ranks' host eviction choice (the TP digest check tripped). The
            # request's own match refreshes recency in the same order on all ranks.
            if self.kv_sessions is not None:
                self.kv_sessions.materialized(insert_result.inserted_host_node)
                self.kv_sessions.prefetch_committed(
                    getattr(operation, "_kv_session_registration", None),
                    insert_result.inserted_host_node,
                    expected_key=getattr(operation, "_kv_session_return_key", None),
                )
            if self.peer_kv_servicer is not None:
                # kv-p2p: newly host-inserted (storage-prefetched) pages are
                # now peer-servable. completed_tokens is already
                # rank-reconciled by the commit machinery in this lineage
                # (the old min_completed_tokens reduction is gone).
                self._peer_push_prefetch_commit(insert_result)
            loaded_from_storage = completed_tokens - insert_result.prefix_len
            guard = getattr(self, "queued_prefix_guard", None)
            if guard is not None and insert_result.inserted_host_node is not None:
                # S-PULL: lock the landed pages before the anchor lock goes.
                guard.on_landed(req_id, insert_result.inserted_host_node)
            self._peer_demote_after_commit(
                req_id,
                completed_tokens,
                getattr(operation, "prefetch_attempt_id", None),
                insert_result.inserted_host_node,
            )

        self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)

        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage
        logger.info(
            "HiCache prefetch %s req=%s completed=%d matched=%d loaded=%d occupied=%d dur_ms=%.0f "
            "prefetch_attempt_id=%s endpoint_round=%s initial_query_pages=%s "
            "initial_local_prefix_tokens=%s selected_endpoint_pages=%s",
            "dropped" if insert_result.host_insert_dropped else "success",
            req_id,
            completed_tokens,
            insert_result.prefix_len,
            loaded_from_storage,
            self.cache_controller.prefetch_tokens_occupied,
            (time.monotonic() - operation.start_time) * 1e3,
            *self._prefetch_trace_args(operation),
        )
        if self.enable_storage_metrics and self.storage_metrics_collector is not None:
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)
        return

    def _check_hybrid_prefetch_result(
        self,
        req_id: str,
        operation: PrefetchOperation,
        completed_tokens: int,
        hash_value: list[str],
        host_indices: torch.Tensor,
        last_host_node_id: NodeId,
        anchor_lock_params: DecLockRefParams,
        prefetch_key: RadixKey,
    ) -> bool:
        """Decide the length of usable prefix.

        Two strategies depending on the hybrid layout:

        * DSA-style (Full attention + KV-derived ALL_PAGES sidecar such as the
          DSA / MiniMax indexer): *clamp* to the minimum fetched prefix shared by
          the Full KV pool and every sidecar. A partial prefix is still usable
          because the sidecar is page-aligned with KV and required for every page.
        * Everything else (SWA / Mamba components, mixed DeepSeekV4 stacks):
          *all-or-nothing*. Their pools only cover a window / tail and cannot be
          truncated page by page, so any shortfall discards the whole prefetch.

        Returns true if prefetch success, or false when an all-or-nothing prefetch
        was discarded (the caller should then treat the prefetch as finished).
        """
        # Sync completed tokens and per-pool hit pages across ATTN groups, taking
        # the minimum so every rank agrees on the same usable prefix length.
        #
        # Skip KV-derived pools, which do not report hits in operation.pool_storage_result.
        # Their hit lengths are stored in completed_tokens.
        pool_transfers = [
            transfer
            for transfer in operation.pool_transfers or []
            if transfer.indices_from_pool != PoolName.KV
        ]
        hit_pages = (
            operation.pool_storage_result.extra_pool_hit_pages if pool_transfers else {}
        )
        pool_hit_pages = [hit_pages.get(t.name, 0) for t in pool_transfers]
        completed_tokens = operation.completed_tokens
        # Hybrid cache state is all-or-nothing: every extra pool (SWA / Mamba / ...)
        # must cover the same fetched prefix. If any pool falls short the whole
        # prefetch result is unusable, so discard it and release everything.
        expected_tokens = len(hash_value) * self.page_size
        all_succeeded = completed_tokens == expected_tokens and all(
            transfer.keys is not None and count == len(transfer.keys)
            for transfer, count in zip(pool_transfers, pool_hit_pages)
        )
        if pool_transfers and not all_succeeded:
            # Drop the KV beliefs from the first page any pool failed to serve;
            # the next insert then re-writes that span through one FULL check,
            # restoring the missing aux pages.
            keep_pages = completed_tokens // self.page_size
            for transfer, count in zip(pool_transfers, pool_hit_pages):
                if transfer.keys is None:
                    keep_pages = 0
                elif count < len(transfer.keys):
                    # Aux transfers key the chain's trailing pages.
                    keep_pages = min(
                        keep_pages, max(0, len(hash_value) - len(transfer.keys))
                    )
            self.storage_existence_cache.invalidate_beyond(
                PoolName.KV, hash_value, keep_pages=keep_pages
            )
            # The controller's prefetch IO thread already releases the untransferred
            # tail (host_indices[completed_tokens:])
            self.cache_controller.append_host_mem_release(
                host_indices=host_indices[:completed_tokens],
                extra_pools=pool_transfers if operation.pool_transfers_done else None,
            )
            if anchor_lock_params is not None:
                self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
            if self.buffer_pipeline is not None:
                self.buffer_pipeline.pop_prefix_ctx(req_id)
                self.buffer_pipeline.release_anchor_lock(req_id)
            del self.ongoing_prefetch[req_id]
            self.cache_controller.prefetch_tokens_occupied -= (
                self._prefetch_occupied_span(prefetch_key, host_indices)
            )
            self.prefetch_loaded_tokens_by_reqid[req_id] = 0
            logger.warning(
                "HiCache hybrid prefetch discarded req=%s completed=%d requested=%d "
                "kv_beliefs_kept_pages=%d prefetch_attempt_id=%s endpoint_round=%s "
                "initial_query_pages=%s initial_local_prefix_tokens=%s selected_endpoint_pages=%s",
                req_id,
                completed_tokens,
                expected_tokens,
                keep_pages,
                *self._prefetch_trace_args(operation),
            )
            return False
        return True

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        # The request is being scheduled; a still-unserved miss marker is moot.
        self._storage_prefetch_missed_rids.discard(req_id)
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def pop_storage_prefetch_miss(self, req_id: str) -> bool:
        """True once per resolved storage-prefetch miss for a live request;
        the scheduler uses it to arm the paced availability-check retry."""
        if req_id in self._storage_prefetch_missed_rids:
            self._storage_prefetch_missed_rids.discard(req_id)
            return True
        return False

    def plan_staged_splice(
        self, req_id: str, device_prefix_len: int
    ) -> tuple[int, int]:
        """(kv, swa) host-hit tokens a staged buffer-mode prefetch will splice
        given the request's live device prefix; frees unusable holds."""
        if self.buffer_pipeline is None:
            return 0, 0
        return self.buffer_pipeline.plan_staged_splice(req_id, device_prefix_len)

    def staged_prefetch_swa_tokens(self, req_id: str) -> int:
        """SWA device tokens consuming a staged buffer-mode prefetch will
        allocate; surfaced as the request's swa_host_hit_length."""
        if self.buffer_pipeline is None:
            return 0
        return self.buffer_pipeline.staged_prefetch_swa_tokens(req_id)

    @rank_consensus(same_params=True)
    def release_aborted_request(self, rid: str) -> None:
        notice = self._peer_notice_backend()
        if notice is not None:
            notice.notice_forget(rid)  # no NOTICE after an abort
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)
        self._storage_prefetch_missed_rids.discard(rid)
        if (
            self.buffer_pipeline is not None
            and self.buffer_pipeline.release_staged_hold(rid)
        ):
            return
        if rid not in self.ongoing_prefetch:
            return

        (
            last_host_node_id,
            prefetch_key,
            host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        ) = self.ongoing_prefetch[rid]
        if operation.host_indices is None:
            self.cache_controller.terminate_prefetch(operation)
            self.revoke_pending_prefetch(rid)
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        # Aborted after the host alloc: leases already read in full get no
        # DEMOTE_OK (commit-gated demote); later ones expire.
        self._peer_demote_discard(rid, "abort")
        if anchor_lock_params is not None:
            self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
        del self.ongoing_prefetch[rid]
        if self.buffer_pipeline is not None:
            self.buffer_pipeline.pop_prefix_ctx(rid)
            self.buffer_pipeline.release_anchor_lock(rid)
        pool_transfers = [x for xfers in comp_xfers.values() for x in xfers]
        self.cache_controller.append_host_mem_release(
            host_indices=host_indices[:completed_tokens],
            extra_pools=pool_transfers if operation.pool_transfers_done else None,
        )
        # Buffer mode granted occupancy at hit-alloc, sized to the bounce;
        # cache mode reserved the requested span at enqueue.
        self.cache_controller.prefetch_tokens_occupied -= self._prefetch_occupied_span(
            prefetch_key, host_indices
        )

    def _invalidate_absent_from_hit_query(self, operation) -> None:
        """Drop KV beliefs beyond the folded usable cut (rank-synced): the
        next insert then re-writes the node (all pools), healing stale
        positives and aux holes at the cut through one FULL check."""
        if self.host_memory_mode != "buffer_only":
            return
        chain = operation.all_hash_values
        if chain is None:
            return
        self.storage_existence_cache.invalidate_beyond(
            PoolName.KV, chain, keep_pages=operation.storage_hit_count // self.page_size
        )

    def _account_prefetch_outcome(self, operation, revoked: bool) -> None:
        """Feed the cumulative prefetch-outcome counters at the (rank-synced)
        query outcome: T = prompt tokens, L = requested, m = L3-miss."""
        requested = operation.stats_requested_tokens
        if requested <= 0:
            return
        stats = self._prefetch_outcome_stats
        hit = max(0, min(operation.storage_hit_count, requested))
        if revoked:
            if hit > 0:
                stats["revoked_insufficient"] += 1
            else:
                stats["revoked_full_miss"] += 1
        miss = requested - hit
        total = max(operation.stats_total_tokens, requested, 1)
        stats["l3_demand_requests"] += 1
        stats["l1l2_miss_tokens"] += requested
        stats["l3_miss_tokens"] += miss
        stats["l3_demand_total_tokens"] += total
        stats["l3_sum_rate_all"] += miss / total
        stats["l3_sum_rate_main_weighted"] += (miss / requested) * total

    def prefetch_outcome_stats_snapshot(self) -> dict:
        """Cumulative counters + instantaneous occupancy, in the schema
        log_prefetch_stats consumers expect."""
        cc = self.cache_controller
        cap = max(cc.prefetch_capacity_limit, 1)
        return {
            **self._prefetch_outcome_stats,
            "occupancy_ratio": cc.prefetch_tokens_occupied / cap,
        }

    def _prefetch_occupied_span(self, prefetch_key, host_indices) -> int:
        """Occupancy units held by a prefetch: cache mode reserves the
        requested span at enqueue; buffer mode grants at hit-alloc, sized
        to the allocation (0 while still querying / parked)."""
        if self.host_memory_mode == "buffer_only":
            return len(host_indices) if host_indices is not None else 0
        return len(prefetch_key)

    def revoke_pending_prefetch(self, req_id: str) -> None:
        info = self.ongoing_prefetch.pop(req_id, None)
        if info is None:
            return
        (
            last_host_node_id,
            prefetch_key,
            _host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        ) = info
        self._invalidate_absent_from_hit_query(operation)
        if self.buffer_pipeline is not None:
            self.buffer_pipeline.pop_prefix_ctx(req_id)
            self.buffer_pipeline.release_anchor_lock(req_id)
        cc = self.cache_controller
        # This revoke runs before IO is queued. Return leases for declined
        # hits as well as explicitly cancelled requests.
        cc.terminate_prefetch(operation)
        cc.append_host_mem_release(
            extra_pools=[x for xfers in comp_xfers.values() for x in xfers]
        )
        if anchor_lock_params is not None:
            self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
        # Every revoke path runs before the bounce alloc, so buffer mode
        # holds no occupancy here; post-alloc aborts go through
        # release_aborted_request instead.
        assert _host_indices is None or self.host_memory_mode != "buffer_only"
        cc.prefetch_tokens_occupied = max(
            0,
            cc.prefetch_tokens_occupied
            - self._prefetch_occupied_span(prefetch_key, _host_indices),
        )

    def _allocate_exact_storage_hit(self, operation) -> None:
        """Admit the same pinned endpoint on every cache rank, or revoke it.

        Every synchronized hit-queue item participates, including late results
        of an old request attempt. Those results must never mutate its successor.
        Hybrid peer endpoint transfers currently require PP size one.
        """
        cc = self.cache_controller
        req_id = operation.request_id
        info = self.ongoing_prefetch.get(req_id)
        current = info is not None and info.operation is operation
        buffer_mode = self.host_memory_mode == "buffer_only"
        eligible = (
            current
            and not operation.is_terminated()
            and operation.storage_hit_count >= self.prefetch_threshold
        )
        try:
            if eligible and buffer_mode:
                eligible = (
                    not cc.prefetch_rate_limited()
                    and self.buffer_pipeline.try_lock_anchor(req_id) != "anchor_lost"
                    and not self.buffer_pipeline.staged_span_covered(
                        req_id, operation.storage_hit_count
                    )
                )
        except Exception:
            logger.warning("peer endpoint admission check failed", exc_info=True)
            eligible = False
        ready = torch.tensor(int(eligible), dtype=torch.int)
        self._all_reduce(ready, torch.distributed.ReduceOp.MIN)

        def revoke():
            if current:
                self._storage_prefetch_missed_rids.add(req_id)
                self.revoke_pending_prefetch(req_id)
            else:
                cc.terminate_prefetch(operation)

        if ready.item() == 0:
            if current:
                self._invalidate_absent_from_hit_query(operation)
            revoke()
            return
        self._invalidate_absent_from_hit_query(operation)
        self._account_prefetch_outcome(operation, revoked=False)
        alloc_len = operation.storage_hit_count

        def allocate():
            try:
                return cc.mem_pool_host.alloc(alloc_len), False
            except Exception:
                logger.warning("peer endpoint host allocation failed", exc_info=True)
                return None, True

        def release(indices):
            if indices is None:
                return True
            try:
                cc.mem_pool_host.free(indices)
                return True
            except Exception:
                logger.error("peer endpoint allocation rollback failed", exc_info=True)
                return False

        host_indices, failed = allocate()
        state = torch.tensor(
            [int(host_indices is None), int(operation.is_terminated()), int(failed)],
            dtype=torch.int,
        )
        self._all_reduce(state, torch.distributed.ReduceOp.MAX)
        if state[1].item() or state[2].item():
            release(host_indices)
            revoke()
            return
        if state[0].item():
            # Return the initial local allocation before all ranks run normal
            # host eviction/retry. Never replace E by an arbitrary shorter E'.
            released = release(host_indices)
            host_indices = None
            try:
                if released:
                    self.evict_host(alloc_len, exclude_rid=req_id)
                    host_indices, failed = allocate()
            except Exception:
                logger.warning("peer endpoint host eviction failed", exc_info=True)
            ready = torch.tensor(
                int(host_indices is not None and not operation.is_terminated()),
                dtype=torch.int,
            )
            self._all_reduce(ready, torch.distributed.ReduceOp.MIN)
            if ready.item() == 0:
                release(host_indices)
                revoke()
                return
        operation.host_indices = host_indices
        self.ongoing_prefetch[req_id] = info._replace(host_indices=host_indices)
        if buffer_mode:
            cc.prefetch_tokens_occupied += alloc_len
        cc.prefetch_buffer.put(operation)

    def _drain_storage_control_queues_impl(
        self,
        n_storage_hit: Optional[int],
        n_ack_prefetch: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        extra_release_counts: Optional[dict[PoolName, int]],
        log_metrics: bool,
    ) -> None:
        cc = self.cache_controller

        def _drain_queue(q: Queue[T], n: Optional[int]) -> Iterator[T]:
            """If n is None, consume all items from the queue.
            Otherwise, consume n items from the queue.  Blocking if there are no enough n items.

            In TP, each rank consumes the a minimal number of items of all ranks.
            In PP, each rank consumes the exact number of items of PP0.  Refer to _pp_sync for more details.

            This prevents TP/PP divergence.
            """
            if n is None:
                while not q.empty():
                    item = q.get()
                    yield item
            else:
                for _ in range(n):
                    # Block when there are not enough elements.
                    # All TP/PP ranks must consume the same number of elements.
                    item = q.get()
                    yield item

        buffer_mode = self.host_memory_mode == "buffer_only"

        def _try_alloc_storage_hit(operation) -> bool:
            """Allocate the hit-sized bounce and launch the transfer.
            Returns False when staging pressure defers the allocation
            (buffer mode parks and retries; cache mode revokes)."""
            req_id = operation.request_id
            info = self.ongoing_prefetch.get(req_id)
            if info is None:
                return True  # aborted/cleaned; nothing to retry
            if info.operation is not operation:
                # A parked hit of a superseded attempt (stopped, then the
                # request was pulled again): end that attempt alone; the
                # current operation of this rid is not touched.
                self._note_hit_drop("superseded", operation, parked=True)
                cc.terminate_prefetch(operation)
                return True
            if operation.is_terminated():
                self._note_hit_drop("terminated", operation, parked=buffer_mode)
                self.revoke_pending_prefetch(req_id)
                return True

            if buffer_mode and cc.prefetch_rate_limited():
                # Pool is load-saturated: hold the KNOWN hit until staged
                # prefetches ahead of us are consumed. The op stays in
                # ongoing_prefetch, so wait_complete keeps gating admission.
                return False
            if buffer_mode:
                # IO commit: pin before the bounce alloc so a cancel is a
                # plain revoke and a parked op keeps its pin; a fetch whose
                # splice base is gone is not worth its storage read.
                if self.buffer_pipeline.try_lock_anchor(req_id) == "anchor_lost":
                    self._prefetch_outcome_stats["declined_anchor_lost"] += 1
                    self._note_hit_drop("anchor_lost", operation)
                    # Span still L3-resident: arm the paced retry to re-fetch
                    # from the shorter post-loss match.
                    self._storage_prefetch_missed_rids.add(req_id)
                    self.revoke_pending_prefetch(req_id)
                    return True
                if self.buffer_pipeline.staged_span_covered(
                    req_id, operation.storage_hit_count
                ):
                    # Live tree already covers the span: nothing left to
                    # splice, so skip the storage read.
                    self._prefetch_outcome_stats["declined_device_covered"] += 1
                    self._note_hit_drop("device_covered", operation)
                    self.revoke_pending_prefetch(req_id)
                    return True
            alloc_len = operation.storage_hit_count
            host_indices = cc.mem_pool_host.alloc(alloc_len)
            if host_indices is None:
                self.evict_host(alloc_len, exclude_rid=req_id)
                host_indices = cc.mem_pool_host.alloc(alloc_len)
            if host_indices is None and not buffer_mode:
                # Memory-pressure fallback: a shorter page-aligned prefix.
                # (Cache mode only — buffer mode parks for the full hit.)
                available_size = cc.mem_pool_host.available_size()
                alloc_len = min(
                    operation.storage_hit_count,
                    available_size - (available_size % self.page_size),
                )
                if alloc_len >= self.prefetch_threshold:
                    host_indices = cc.mem_pool_host.alloc(alloc_len)
            if host_indices is None:
                if buffer_mode:
                    return False
                self._note_hit_drop(
                    "host_alloc_failed",
                    operation,
                    shorter_tokens=alloc_len,
                    host_available=cc.mem_pool_host.available_size(),
                )
                self.revoke_pending_prefetch(req_id)
                return True

            if (
                alloc_len < operation.storage_hit_count
                and envs.SGLANG_PEER_KV_DEMOTE_NOTICE.get()
            ):
                # NOTICE: the shorter alloc reads fewer pages;
                # trim the lease to them so it can complete (else the owner
                # keeps the pins until the TTL and no commit follows).
                self._peer_trim_short_alloc(operation, alloc_len)
            operation.storage_hit_count = alloc_len
            operation.hash_value = operation.hash_value[: alloc_len // self.page_size]
            operation.host_indices = host_indices
            self.ongoing_prefetch[req_id] = info._replace(host_indices=host_indices)
            if buffer_mode:
                cc.prefetch_tokens_occupied += alloc_len
            cc.prefetch_buffer.put(operation)
            return True

        def _drain_and_alloc_storage_hit():
            # Parked hits first (FIFO fairness with retries; buffer only).
            if buffer_mode:
                parked = self.buffer_pipeline.pending_hit_allocs
                while parked:
                    if not _try_alloc_storage_hit(parked[0]):
                        break
                    parked.popleft()
            lookup_aware = envs.SGLANG_JIT_PULL_LOOKUP_AWARE_HOLD.get()
            for operation in _drain_queue(cc.prefetch_hit_queue, n_storage_hit):
                if lookup_aware:
                    # F3: the op's LOOKUP + TP sync reached the drain (the same
                    # pass on every rank: the drain count is MIN-reduced).
                    operation.hit_drained = True
                if getattr(operation, "requires_exact_endpoint", False):
                    self._allocate_exact_storage_hit(operation)
                    continue
                req_id = operation.request_id
                info = self.ongoing_prefetch.get(req_id)
                if info is None or info.operation is not operation:
                    # Request already aborted/cleaned up, or the late hit of a
                    # superseded attempt (stopped before its hit was drained,
                    # then the request was pulled again): end that attempt
                    # alone and never touch the current operation of this rid
                    # (as _allocate_exact_storage_hit); still flush the
                    # query's absent-hash feedback.
                    self._note_hit_drop(
                        "aborted" if info is None else "superseded", operation
                    )
                    cc.terminate_prefetch(operation)
                    self._invalidate_absent_from_hit_query(operation)
                    continue
                if operation.is_terminated():
                    # Controller-side miss termination (retryable) or an abort
                    # race (abort cleanup discards the marker).
                    self._note_hit_drop("terminated", operation)
                    self._storage_prefetch_missed_rids.add(req_id)
                    self.revoke_pending_prefetch(req_id)
                    continue
                if operation.storage_hit_count < self.prefetch_threshold:
                    # Below-threshold hit: classify + feed the L3 miss
                    # accounting, then revoke (not enough benefit).
                    self._account_prefetch_outcome(operation, revoked=True)
                    self._storage_prefetch_missed_rids.add(req_id)
                    self.revoke_pending_prefetch(req_id)
                    continue
                self._invalidate_absent_from_hit_query(operation)
                self._account_prefetch_outcome(operation, revoked=False)
                if not _try_alloc_storage_hit(operation):
                    # Counted once at first parking, not per retry tick.
                    self._prefetch_outcome_stats["declined_rate_limited"] += 1
                    self.buffer_pipeline.pending_hit_allocs.append(operation)

        def _drain_ack_prefetch():
            for ack in _drain_queue(cc.ack_prefetch_queue, n_ack_prefetch):
                operation = ack.operation
                info = self.ongoing_prefetch.get(operation.request_id)
                current = info is not None and info.operation is operation
                if ack.completed_tokens is not None:
                    if current:
                        assert operation.completed_tokens <= ack.completed_tokens
                        operation.completed_tokens = ack.completed_tokens
                if ack.pool_hits is not None:
                    if current:
                        operation.pool_storage_result.update_extra_pool_hit_pages(
                            ack.pool_hits
                        )
                        operation.pool_transfers_done = True
                if ack.completed_req:
                    if current:
                        # check_prefetch_progress() is not called for this rid yet.
                        # Let us insert the prefetch result into the radix tree.
                        self._handle_prefetch_result(operation)
                    cc.append_host_mem_release(
                        operation.host_indices[operation.completed_tokens :],
                        (
                            operation.pool_transfers
                            if not operation.pool_transfers_done
                            else None
                        ),
                    )

        def _drain_backup():
            drained = 0
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):
                drained += 1
                if buffer_mode:
                    # Storage write acked: free the staging.
                    self.buffer_pipeline.finish_storage_write_ack(operation.id)
                else:
                    entry = self.ongoing_backup.pop(operation.id, None)
                    if entry is not None:
                        node_id, lock_params = entry
                        self.dec_host_lock_ref(node_id, lock_params)
                if (
                    log_metrics
                    and self.enable_storage_metrics
                    and self.storage_metrics_collector is not None
                ):
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )
            return drained

        def _drain_release():
            host_indices_list = []
            released_tokens = 0
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
                released_tokens += len(host_indices)
            if host_indices_list:
                cc.mem_pool_host.free(torch.cat(host_indices_list, dim=0))
            return len(host_indices_list), released_tokens

        def _drain_extra_release():
            drained: dict[PoolName, tuple[int, int]] = {}
            if not extra_release_counts:
                return drained
            for pool_name, limit in extra_release_counts.items():
                release_queue = cc.extra_host_mem_release_queues.get(pool_name)
                if release_queue is None:
                    continue
                host_indices_list = []
                released_tokens = 0
                for host_indices in _drain_queue(release_queue, limit):
                    host_indices_list.append(host_indices)
                    released_tokens += len(host_indices)
                if host_indices_list:
                    cc.mem_pool_host.free(
                        torch.cat(host_indices_list, dim=0), pool=pool_name
                    )
                drained[pool_name] = (len(host_indices_list), released_tokens)
            return drained

        _drain_and_alloc_storage_hit()
        _drain_ack_prefetch()
        _drain_backup()
        _drain_release()
        _drain_extra_release()

    def drain_storage_control_queues(self) -> None:
        self._invalidate_pipelined_sync()
        cc = self.cache_controller
        extra_release_queues = getattr(cc, "extra_host_mem_release_queues", {})
        extra_pool_names = list(extra_release_queues)
        local_qsize_list = [
            cc.prefetch_hit_queue.qsize(),
            cc.ack_prefetch_queue.qsize(),
            cc.ack_backup_queue.qsize(),
            cc.host_mem_release_queue.qsize(),
            *[
                extra_release_queues[pool_name].qsize()
                for pool_name in extra_pool_names
            ],
        ]
        qsizes = torch.tensor(
            local_qsize_list,
            dtype=torch.int,
        )
        self._all_reduce(qsizes, torch.distributed.ReduceOp.MIN)
        qsize_list = list(map(int, qsizes.tolist()))
        n_storage_hit, n_ack_prefetch, n_backup, n_release = qsize_list[:4]
        extra_release_counts = {
            pool_name: count
            for pool_name, count in zip(extra_pool_names, qsize_list[4:])
        }
        self._drain_storage_control_queues_impl(
            n_storage_hit=n_storage_hit,
            n_ack_prefetch=n_ack_prefetch,
            n_backup=n_backup,
            n_release=n_release,
            extra_release_counts=extra_release_counts,
            log_metrics=True,
        )

    def drain_storage_control_queues_local(self) -> None:
        """Drain the storage control queues without cross-rank synchronization.

        For the detach / shutdown path, where best-effort cleanup matters more than
        keeping the drained counts identical across ranks. The prefetch-hit queue is
        deliberately skipped: servicing it would allocate host pages for a prefetch
        that can no longer complete.
        """
        self._invalidate_pipelined_sync()
        cc = self.cache_controller
        # The storage queues are created by the controller when the storage threads
        # start, so they are still None when a backend was never attached.
        if cc is None or cc.prefetch_hit_queue is None:
            return
        self._drain_storage_control_queues_impl(
            n_storage_hit=0,
            n_ack_prefetch=0,
            n_backup=None,
            n_release=None,
            extra_release_counts={
                name: None for name in cc.extra_host_mem_release_queues
            },
            log_metrics=False,
        )

    # ---- HiCache: Storage backend lifecycle (delegated) ----

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Attach (enable) the HiCache storage backend at runtime."""
        if self._storage_attachment is None:
            return (
                False,
                "HiCache is not initialized; launch with "
                "--enable-hierarchical-cache to attach a storage backend.",
            )
        # The storage queues are replaced: drop a started pipelined snapshot.
        self._invalidate_pipelined_sync()
        return self._storage_attachment.attach(
            storage_backend=storage_backend,
            storage_backend_extra_config_json=storage_backend_extra_config_json,
            served_model_name=served_model_name,
            hicache_storage_prefetch_policy=hicache_storage_prefetch_policy,
            hicache_write_policy=hicache_write_policy,
        )

    def detach_storage_backend(self) -> tuple[bool, str]:
        """Detach (disable) the HiCache storage backend at runtime."""
        if self._storage_attachment is None:
            return False, "HiCache storage backend is not initialized."
        self._invalidate_pipelined_sync()
        result = self._storage_attachment.detach()
        if result[0]:
            self._clear_host_returns()
        return result

    def shutdown(self) -> None:
        """Best-effort auto-detach of the storage backend on process shutdown."""
        # Other ranks may be gone: drop the in-flight step sync unwaited.
        self._pipelined_sync_pending = None
        if self.swa_backfill is not None:
            # Stop backfill IO before the backend it calls is detached.
            self.swa_backfill.close()
        if self._storage_attachment is not None:
            self._storage_attachment.shutdown()

    def _storage_operation_counts(self) -> dict[str, int]:
        controller = self.cache_controller
        extra_release_queues = getattr(controller, "extra_host_mem_release_queues", {})
        counts = {
            "write_through": len(self.ongoing_write_through),
            "load_back": len(self.ongoing_load_back),
            "prefetch": len(self.ongoing_prefetch),
            "host_return": len(getattr(self, "_host_returns", {})),
            "host_return_import": len(getattr(self, "_host_return_imports", {})),
            "peer_events": int(self._peer_control_pending()),
            "backup": len(self.ongoing_backup),
            "write_queue": len(controller.write_queue),
            "ack_write": len(controller.ack_write_queue),
            "load_queue": len(controller.load_queue),
            "ack_load": len(controller.ack_load_queue),
            "prefetch_queue": controller.prefetch_queue.qsize(),
            "prefetch_hit": controller.prefetch_hit_queue.qsize(),
            "backup_queue": controller.backup_queue.qsize(),
            "ack_backup": controller.ack_backup_queue.qsize(),
            "release_queue": controller.host_mem_release_queue.qsize(),
            "extra_release_queues": sum(
                q.qsize() for q in extra_release_queues.values()
            ),
        }
        counts.update(controller.storage_background_operation_counts())
        return counts

    def _drain_storage_operations(self, timeout_s: float = 300.0) -> bool:
        """Drain all pre-clear storage operations on every cache rank."""
        # SGLANG_HICACHE_PIPELINED_SYNC: drain on the synchronous path (each
        # round settles what it counts); the started snapshot is dropped.
        self._discard_pipelined_sync()
        self._pipelined_sync_paused = True
        try:
            return self._drain_storage_operations_loop(timeout_s)
        finally:
            self._pipelined_sync_paused = False

    def _drain_storage_operations_loop(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while True:
            # check_hicache_events contains collectives. Propagate a failure
            # instead of catching it and entering a mismatched next collective.
            self.check_hicache_events()
            counts = self._storage_operation_counts()
            local_pending = any(counts.values())
            state = torch.tensor(
                [
                    int(local_pending),
                    int(local_pending and time.monotonic() >= deadline),
                ],
                dtype=torch.int,
                device="cpu",
            )
            self._all_reduce_all_cache_ranks(state, torch.distributed.ReduceOp.MAX)
            if state[1].item() != 0:
                logger.error(
                    "Failed or timed out draining HiCache storage operations "
                    "before clear: %s",
                    counts,
                )
                return False
            if state[0].item() == 0:
                return True
            time.sleep(0.01)

    def clear_storage_backend(self) -> bool:
        local_success = False
        if self._storage_attachment is not None:
            if self._drain_storage_operations():
                try:
                    local_success = self._storage_attachment.clear()
                    if local_success:
                        # L3 is empty now: every storage-presence belief is stale,
                        # and a retained positive would skip that page's backup forever.
                        self.storage_existence_cache.clear()
                except Exception as e:
                    logger.error(
                        "Failed to clear hierarchical cache storage backend: %s", e
                    )
        else:
            logger.warning("Hierarchical cache storage backend is not enabled.")

        success = torch.tensor(int(local_success), dtype=torch.int, device="cpu")
        self._all_reduce_all_cache_ranks(success, torch.distributed.ReduceOp.MIN)
        if success.item() != 0:
            logger.info("Hierarchical cache storage backend cleared successfully!")
            return True
        return False

    # ---- HiCache: Async Event Management ----

    def _count_ready_acks(self, ack_queue) -> int:
        ready_count = 0
        for ack in ack_queue:
            if not ack.finish_event.query():
                break
            ready_count += 1
        return ready_count

    def _sync_hicache_ready_counts(
        self,
    ) -> tuple[int, int, tuple[int, ...], tuple[PoolName, ...]]:
        cc = self.cache_controller
        if cc is None:
            write_acks = 0
            load_acks = 0
            storage_queue_sizes = ()
            extra_pool_names = ()
        else:
            write_acks = self._count_ready_acks(cc.ack_write_queue)
            load_acks = self._count_ready_acks(cc.ack_load_queue)
            extra_release_queues = getattr(cc, "extra_host_mem_release_queues", {})
            extra_pool_names = (
                tuple(extra_release_queues) if self.enable_storage else ()
            )
            storage_queue_sizes = (
                (
                    cc.prefetch_hit_queue.qsize(),
                    cc.ack_prefetch_queue.qsize(),
                    cc.ack_backup_queue.qsize(),
                    cc.host_mem_release_queue.qsize(),
                    *(extra_release_queues[name].qsize() for name in extra_pool_names),
                )
                if self.enable_storage
                else ()
            )

        # Piggybacked TP checks: [x, -x] MIN-reduces to [min, -max], equal iff
        # x matched on every rank. x = the write_back reclaim digest (victim
        # order) folded with the ack-queue shapes (_step_sync_digest), and the
        # structural digest (free with the reduce).
        digest = self._step_sync_digest()
        sd = self.structural_digest_pair()[0] if self.tp_digest_mode() != "off" else 0
        # Window SWA backfill: the job-list digest, checked every step in any
        # digest mode. poll()'s all-reduce is sized by the job count, so a
        # rank-local submit/cancel would otherwise hang it; this raises first.
        backfill = self.swa_backfill
        bd = backfill.lockstep_digest() if backfill is not None else None
        # Idle-sync skip: -(ops in flight here) MIN-reduces to -(max over ranks).
        track_idle = getattr(self, "_idle_sync_enabled", False)
        ready_counts = torch.tensor(
            [
                write_acks,
                load_acks,
                *storage_queue_sizes,
                *(
                    [
                        -int(
                            bool(self.gpu_export_manager.active)
                            or bool(
                                self.gpu_export_manager.broker
                                and self.gpu_export_manager.broker.has_work()
                            )
                        )
                    ]
                    if getattr(self, "gpu_export_manager", None) is not None
                    else []
                ),
                *(
                    [-int(self._peer_control_pending())]
                    if getattr(self, "_peer_kv_tp_participant", False)
                    else []
                ),
                digest,
                -digest,
                sd,
                -sd,
                *([bd, -bd] if bd is not None else []),
                *([-self._local_hicache_inflight()] if track_idle else []),
            ],
            dtype=torch.int64,
            device="cpu",
        )
        self._all_reduce(ready_counts, torch.distributed.ReduceOp.MIN)

        count_values = list(map(int, ready_counts.tolist()))
        if track_idle:
            # Every rank's in-flight count was 0 before this sync: the TP
            # source may skip the next steps (wants_step_sync).
            self._idle_sync_all_idle = count_values.pop() == 0
            self._idle_sync_skipped = 0
        if bd is not None:
            backfill.check_lockstep_digest(
                count_values[-2], count_values[-1], "check_hicache_events"
            )
            del count_values[-2:]
        self._check_step_sync_digest(
            count_values[-4], count_values[-3], "check_hicache_events"
        )
        self._check_digest_pair(
            count_values[-2], count_values[-1], "check_hicache_events"
        )
        export_fields = 0
        if getattr(self, "_peer_kv_tp_participant", False):
            self._peer_cache_has_work = bool(-count_values.pop(-5))
        if getattr(self, "gpu_export_manager", None) is not None:
            self._gpu_export_has_work = bool(-count_values[-5])
            export_fields = 1
        return (
            count_values[0],
            count_values[1],
            tuple(count_values[2 : -4 - export_fields]),
            extra_pool_names,
        )

    def writing_check(
        self, write_back: bool = False, finish_count: Optional[int] = None
    ) -> None:
        """Poll write-through completions."""
        cc = self.cache_controller
        if cc is None:
            return

        if write_back:
            # Blocking: wait for all pending write-backs
            while self.ongoing_write_through:
                for ack in cc.ack_write_queue:
                    ack.finish_event.synchronize()
                    for ack_id in ack.node_ids:
                        if is_spill_ack_id(ack_id):
                            self._finish_swa_spill_ack(ack_id, ack)
                        elif ack_id in self.ongoing_write_through:
                            self._finish_write_through_ack(ack_id)
                    self._log_write_ack_metrics(ack)
                    self._note_settled_write(ack)
                cc.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        if finish_count is None:
            # Every rank must enter the all_reduce below; ongoing_write_through can
            # diverge across ranks (e.g. write_backup returning 0 on a subset).
            finish_count = 0
            if self.pp_rank == 0:
                finish_count = self._count_ready_acks(cc.ack_write_queue)
            finish_count_tensor = torch.tensor(
                finish_count, dtype=torch.int, device="cpu"
            )
            self._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
            finish_count = finish_count_tensor.item()

        # Process completed acks. Window residency spill acks share the queue
        # with write-through acks and are settled in the same FIFO order.
        while finish_count > 0:
            self._settle_write_ack(cc.ack_write_queue.pop(0))
            finish_count -= 1

    def _settle_write_ack(self, ack: HiCacheAck) -> None:
        """Settle one write ack taken off the head of ack_write_queue."""
        ack.finish_event.synchronize()
        for ack_id in ack.node_ids:
            if is_spill_ack_id(ack_id):
                self._finish_swa_spill_ack(ack_id, ack)
            else:
                self._finish_write_through_ack(ack_id)
        self._log_write_ack_metrics(ack)
        self._note_settled_write(ack)

    def _log_write_ack_metrics(self, ack: HiCacheAck) -> None:
        """Record D->H backup volume and duration for a completed write ack."""
        if self.metrics_collector is None:
            return
        for pool, num_tokens in (ack.num_tokens_by_pool or {}).items():
            if num_tokens > 0:
                self.metrics_collector.increment_backup_num_tokens(
                    num_tokens=num_tokens, pool=pool
                )
        if ack.num_bytes > 0:
            self.metrics_collector.increment_backup_num_bytes(ack.num_bytes)
        if ack.timing_enabled:
            duration_ms = ack.start_event.elapsed_time(ack.finish_event)
            self.metrics_collector.observe_backup_duration(duration_ms / 1000.0)

    @timed_phase("loading_check")
    def loading_check(self, finish_count: Optional[int] = None) -> None:
        """Poll load-back completions."""
        cc = self.cache_controller
        if cc is None:
            return
        if finish_count is None:
            # Every rank must enter the all_reduce below; ongoing_load_back can
            # diverge across ranks.
            finish_count = 0
            if self.pp_rank == 0:
                finish_count = self._count_ready_acks(cc.ack_load_queue)
            # Piggybacked TP check: [digest, -digest] MIN-reduces to [min, -max],
            # equal iff reclaim victim order (and the ack queues) matched on
            # every rank.
            digest = self._step_sync_digest()
            sd = (
                self.structural_digest_pair()[0]
                if self.tp_digest_mode() != "off"
                else 0
            )
            sync_tensor = torch.tensor(
                [finish_count, digest, -digest, sd, -sd],
                dtype=torch.int64,
                device="cpu",
            )
            self._all_reduce(sync_tensor, torch.distributed.ReduceOp.MIN)
            finish_count = int(sync_tensor[0].item())
            self._check_step_sync_digest(
                int(sync_tensor[1].item()), int(sync_tensor[2].item()), "loading_check"
            )
            self._check_digest_pair(
                int(sync_tensor[3].item()), int(sync_tensor[4].item()), "loading_check"
            )

        while finish_count > 0:
            self._settle_load_ack(cc.ack_load_queue.pop(0))
            finish_count -= 1

    def _settle_load_ack(self, ack: HiCacheAck) -> None:
        """Settle one load ack taken off the head of ack_load_queue."""
        ack.finish_event.synchronize()
        log_load_op_settled(ack)
        for ack_id in ack.node_ids:
            if (
                self.buffer_pipeline is not None
                and self.buffer_pipeline.try_finish_load_back(ack_id)
            ):
                continue
            if self._swa_copy_verify:
                self._check_swa_copies(
                    self._swa_copy_verify_loads.pop(ack_id, None), "load"
                )
            node, lock_params, host_lock_params = self.ongoing_load_back.pop(ack_id)
            self.dec_lock_ref(node, lock_params)
            self.dec_host_lock_ref(node, host_lock_params)
            # Unpin the loaded nodes; host copies stay as reclaimable duplicates.
            self.tree_core.finish_load_back(node)

        if self.metrics_collector is not None:
            for pool, num_tokens in (ack.num_tokens_by_pool or {}).items():
                if num_tokens > 0:
                    self.metrics_collector.increment_load_back_num_tokens(
                        num_tokens=num_tokens, pool=pool
                    )
            if ack.num_bytes > 0:
                self.metrics_collector.increment_load_back_num_bytes(ack.num_bytes)
            if ack.timing_enabled:
                duration_ms = ack.start_event.elapsed_time(ack.finish_event)
                self.metrics_collector.observe_load_back_duration(duration_ms / 1000.0)

    # ---- HiCache: Scheduler Entry Points ----

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> tuple[torch.Tensor, NodeId]:
        """Prepare KV cache loading from host to device.
        Returns (device_indices, last_node). Buffer mode dispatches to the
        staged-prefetch consumption (BufferModePipeline.init_load_back)."""
        if self.buffer_pipeline is not None:
            return self.buffer_pipeline.init_load_back(params)
        best_match_node_id = params.best_match_node
        mem_quota = params.mem_quota
        req = params.req
        assert req is not None
        last_best_match_device_node_id = req.last_node

        if (
            self.tree_core.is_full_device_evicted(best_match_node_id)
            or params.host_hit_length > 0
            or (
                req is not None
                and (req.swa_host_hit_length > 0 or req.mamba_host_hit_length > 0)
            )
        ):
            if self.load_back(
                best_match_node_id,
                mem_quota,
                req=req,
                active_components=params.active_components,
            ):
                new_indices = self.tree_core.collect_full_device_indices(
                    best_match_node_id, last_best_match_device_node_id
                )
                if new_indices.numel() == 0:
                    return (
                        self.tree_core.empty_match_result.device_indices,
                        last_best_match_device_node_id,
                    )

                logger.debug(
                    "init_load_back success: loaded %d tokens for node %d",
                    len(new_indices),
                    best_match_node_id,
                )
                return new_indices, best_match_node_id

        return (
            self.tree_core.empty_match_result.device_indices,
            last_best_match_device_node_id,
        )

    def load_back_in_flight(
        self, best_match_node: NodeId, last_device_node: NodeId
    ) -> bool:
        """Whether a node between the device anchor and ``best_match_node`` is
        pinned by a load-back awaiting its ack. The device match stops above
        such nodes and an empty load spec skips them, so admitting now would
        recompute a prefix that is about to be on device."""
        return self.tree_core.load_back_pending_on_path(
            best_match_node, last_device_node
        )

    # ---- SGLANG_HICACHE_SKIP_IDLE_SYNC ----

    def idle_sync_skip_supported(self) -> bool:
        """Whether the scheduler may wire wants_step_sync / set_step_sync_flag
        into its request broadcast (plain TP, cache-mode host tier)."""
        return (
            self._idle_sync_enabled
            and self.cache_controller is not None
            and self.pp_size == 1
            and self.host_memory_mode != "buffer_only"
        )

    def _local_hicache_inflight(self) -> int:
        """HiCache work this rank has issued and not settled: D2H / H2D ops
        (queued or awaiting their ack, window spills included), storage
        prefetch / backup state and its control queues, SWA backfill jobs and
        queued SWA re-backups. 0 means a sync has nothing to settle here."""
        cc = self.cache_controller
        if cc is None:
            return 0
        n = (
            len(cc.ack_write_queue)
            + len(cc.ack_load_queue)
            + len(getattr(cc, "write_queue", ()))
            + len(getattr(cc, "load_queue", ()))
            + len(self.ongoing_write_through)
            + len(self.ongoing_load_back)
            + len(getattr(self, "_host_returns", {}))
            + len(getattr(self, "_host_return_imports", {}))
            + int(self._peer_control_pending())
        )
        if self.enable_storage:
            n += len(self.ongoing_prefetch) + len(self.ongoing_backup)
            queues = [
                cc.prefetch_hit_queue,
                cc.ack_prefetch_queue,
                cc.ack_backup_queue,
                cc.host_mem_release_queue,
                *getattr(cc, "extra_host_mem_release_queues", {}).values(),
            ]
            n += sum(q.qsize() for q in queues if q is not None)
        if self.swa_backfill is not None:
            n += self.swa_backfill.pending_count()
        n += len(getattr(self.tree_core, "swa_rebackup_queue", ()) or ())
        return n

    def wants_step_sync(self) -> bool:
        """Called on the TP source rank once per step, before the request
        broadcast that carries the answer to every rank (set_step_sync_flag).
        False only when no rank had HiCache work in flight at the last sync,
        this rank has issued none since, and the last sync is fewer than
        SGLANG_HICACHE_IDLE_SYNC_INTERVAL steps ago. Ops issued on other ranks
        only (rank-local retries) wait for that periodic sync."""
        if not self._idle_sync_all_idle:
            return True
        if self._idle_sync_skipped >= self._idle_sync_interval:
            return True
        if (
            self.gpu_export_manager is not None
            or self.buffer_pipeline is not None
            or self.tp_digest_mode() != "off"
        ):
            # Per-step TP checks / export brokering stay on every step.
            return True
        return self._local_hicache_inflight() > 0

    def set_step_sync_flag(self, sync: Optional[bool]) -> None:
        """The source rank's wants_step_sync() for this step, on every rank.
        None: no broadcast this step (--scheduler-recv-interval skipped it on
        every rank); the last broadcast decision holds until the periodic sync
        is due. Both inputs are the same on every rank, so is the result. Ops
        issued meanwhile wait for the next broadcast step or periodic sync."""
        if sync is None:
            sync = (
                self._last_step_sync_flag is not False
                or self._idle_sync_skipped >= self._idle_sync_interval
            )
        else:
            self._last_step_sync_flag = bool(sync)
        self._step_sync_flag = bool(sync)

    def check_hicache_events(self) -> None:
        """Called per scheduler step to poll async HiCache events."""
        # Reap the previous round's PP-sync sends before issuing new ones.
        self._drain_async_work()

        # SGLANG_HICACHE_SKIP_IDLE_SYNC: the TP source rank's decision for this
        # step, received with the request broadcast (None: no decision, sync).
        step_sync = getattr(self, "_step_sync_flag", None)
        self._step_sync_flag = None
        skip_sync = step_sync is False and self.pp_size == 1
        # SGLANG_HICACHE_PIPELINED_SYNC: apply the vector the previous sync
        # step started, and start this step's at the end (plain TP only).
        pipelined = (
            getattr(self, "_pipelined_sync_active", False)
            and not getattr(self, "_pipelined_sync_paused", False)
            and self.pp_size == 1
        )
        if skip_sync:
            # No op was in flight on any rank at the last sync and none on the
            # source rank since: no ack, storage or backfill state to settle.
            # An op issued on some ranks only waits for the next sync (at most
            # SGLANG_HICACHE_IDLE_SYNC_INTERVAL steps away). A started
            # pipelined snapshot waits for the next sync step.
            self._idle_sync_skipped += 1
            self.idle_sync_skipped_steps += 1
        elif pipelined:
            if getattr(self, "_idle_sync_enabled", False):
                self._idle_sync_skipped = 0
            pending, self._pipelined_sync_pending = self._pipelined_sync_pending, None
            if pending is not None:
                self._apply_pipelined_sync(pending)
        elif self.pp_size != 1:
            finish_counts = torch.zeros(2, dtype=torch.int, device="cpu")
            if self.pp_rank == 0 and self.cache_controller is not None:
                finish_counts[0] = self._count_ready_acks(
                    self.cache_controller.ack_write_queue
                )
                finish_counts[1] = self._count_ready_acks(
                    self.cache_controller.ack_load_queue
                )
            self._all_reduce(finish_counts, torch.distributed.ReduceOp.MIN)
            write_finish_count, load_finish_count = map(int, finish_counts.tolist())
            self.writing_check(finish_count=write_finish_count)
            self.loading_check(finish_count=load_finish_count)
            if self.enable_storage:
                self.drain_storage_control_queues()
        else:
            (
                write_finish_count,
                load_finish_count,
                storage_queue_sizes,
                extra_pool_names,
            ) = self._sync_hicache_ready_counts()
            self.writing_check(finish_count=write_finish_count)
            self.loading_check(finish_count=load_finish_count)

            if (
                self.enable_storage
                and storage_queue_sizes
                # Idle skip: with every reduced count at 0 the cache-mode drain
                # is a no-op. The counts are the all-reduced MINs, so every
                # rank skips the same steps. Buffer mode also retries parked
                # hits here, so it always drains.
                and (any(storage_queue_sizes) or self.host_memory_mode == "buffer_only")
            ):
                n_storage_hit, n_ack_prefetch, n_backup, n_release = (
                    storage_queue_sizes[:4]
                )
                extra_release_counts = {
                    pool_name: count
                    for pool_name, count in zip(
                        extra_pool_names,
                        storage_queue_sizes[4:],
                    )
                }
                self._drain_storage_control_queues_impl(
                    n_storage_hit=n_storage_hit,
                    n_ack_prefetch=n_ack_prefetch,
                    n_backup=n_backup,
                    n_release=n_release,
                    extra_release_counts=extra_release_counts,
                    log_metrics=True,
                )
        if self._swa_window:
            if self.swa_backfill is not None and not skip_sync and not pipelined:
                # Lockstep point: commits land here, one MIN all-reduce.
                # (Pipelined: the rows rode the step-sync vector above.)
                self.swa_backfill.poll()
            # Scheduler level only, never inside an insert walk.
            self.maintain_swa_host_headroom()
            self.retry_swa_rebackup()
        if self.buffer_pipeline is not None:
            self.buffer_pipeline.flush_pending_writes()
        if (
            getattr(self, "gpu_export_manager", None) is not None
            and self._gpu_export_has_work
        ):
            self.gpu_export_manager.tick()
        if getattr(self, "_peer_kv_tp_participant", False):
            if getattr(self, "_peer_cache_has_work", False) and not skip_sync:
                self._drain_peer_cache_events()
            self._advance_host_returns()
        if pipelined and not skip_sync:
            self._start_pipelined_sync()
        if self.enable_storage_metrics and self.storage_metrics_collector is not None:
            self._log_storage_metrics()

    # ---- SGLANG_HICACHE_PIPELINED_SYNC (mem_cache/hicache_pipelined_sync.py) ----

    def _init_pipelined_sync(self) -> bool:
        """Create the dedicated Gloo group and turn the pipelined step sync on.
        Every rank runs this at the end of init_hicache (the group creation is
        a collective over the whole world); every condition below is the same
        on every rank. Otherwise logs why the per-step sync stays."""
        why = None
        if self.pp_size != 1:
            why = "pipeline parallelism"
        elif self.cache_controller is None:
            why = "no HiCache controller"
        elif self.host_memory_mode == "buffer_only":
            why = "buffer-only host memory mode"
        if why is None:
            group, world = create_pipelined_sync_group(
                (self.attn_cp_group, self.attn_tp_group),
                self.tp_group,
                self.tp_world_size,
            )
            if group is None:
                why = world
        if why is not None:
            logger.warning(
                "SGLANG_HICACHE_PIPELINED_SYNC is set but stays off (%s): the "
                "HiCache step sync stays per step",
                why,
            )
            return False

        def reduce_start(vec, group=group, world=world):
            return start_min_reduce(vec, group, world, self._pipelined_sync_algo)

        self._configure_pipelined_sync(reduce_start)
        self._pipelined_sync_group = group
        # The once-a-minute summary comes from the group's first rank only.
        self._pipelined_sync_log_summary = torch.distributed.get_rank(group) == 0
        logger.info(
            "HiCache pipelined step sync on: dedicated Gloo group of %d ranks, "
            "%s, %d backfill rows",
            world,
            self._pipelined_sync_algo,
            self._pipelined_sync_slots,
        )
        return True

    def _configure_pipelined_sync(
        self,
        reduce_start,
        *,
        slots: Optional[int] = None,
        algo: Optional[str] = None,
    ) -> None:
        """Turn the pipelined step sync on with ``reduce_start(vec)`` starting
        the MIN reduction of ``vec`` in place (returns an object with
        ``wait()``, or None when ``vec`` is already reduced). Tests pass a
        simulated reduction."""
        algo = (algo or envs.SGLANG_HICACHE_PIPELINED_SYNC_ALGO.get()).strip().lower()
        if algo not in PIPELINED_SYNC_ALGOS:
            raise ValueError(
                f"SGLANG_HICACHE_PIPELINED_SYNC_ALGO={algo!r}; expected one of "
                f"{'|'.join(PIPELINED_SYNC_ALGOS)}"
            )
        if slots is None:
            slots = envs.SGLANG_HICACHE_PIPELINED_SYNC_BACKFILL_SLOTS.get()
        self._pipelined_sync_algo = algo
        # At least one row, or no backfill job could ever advance.
        self._pipelined_sync_slots = max(1, int(slots))
        self._pipelined_reduce_start = reduce_start
        self._pipelined_sync_pending = None
        self._pipelined_sync_generation = getattr(self, "_pipelined_sync_generation", 0)
        self._pipelined_sync_paused = False
        self.pipelined_sync_stats = {
            "applied": 0,
            "discarded": 0,
            "wait_s": 0.0,
            "max_wait_s": 0.0,
            "backfill_overflow": 0,
        }
        self._pipelined_sync_log = [time.monotonic(), 0, 0.0, 0.0]
        self._pipelined_sync_log_summary = True
        self._pipelined_sync_active = True

    def _invalidate_pipelined_sync(self) -> None:
        """A storage queue was drained or replaced out of band: the started
        snapshot's counts no longer hold, so its apply drops it."""
        self._pipelined_sync_generation = (
            getattr(self, "_pipelined_sync_generation", 0) + 1
        )

    def _discard_pipelined_sync(self) -> None:
        """Wait for the started snapshot's reduction and drop it (reset, the
        synchronous storage drain). Every rank started it, so the wait ends."""
        pending = getattr(self, "_pipelined_sync_pending", None)
        self._pipelined_sync_pending = None
        self._invalidate_pipelined_sync()
        if pending is not None:
            pending.wait()
            # Unknown in-flight state: sync on the next step.
            self._idle_sync_all_idle = False

    @staticmethod
    def _ready_ack_prefix(ack_queue) -> list:
        ready = []
        for ack in ack_queue:
            if not ack.finish_event.query():
                break
            ready.append(ack)
        return ready

    @staticmethod
    def _settle_acked_prefix(ack_queue, acks, settle) -> None:
        """Settle the agreed head of a snapshot's ready acks. Acks leave the
        queue only from its head, so one that is no longer there was settled
        out of band (a synchronous writing_check / loading_check, a write-back
        drain) or dropped by a reset."""
        for ack in acks:
            if ack_queue and ack_queue[0] is ack:
                settle(ack_queue.pop(0))

    def _start_pipelined_sync(self) -> None:
        """Snapshot this rank's step-sync values and start their MIN reduction
        (the same fields as _sync_hicache_ready_counts plus the backfill rows);
        the next sync step applies it."""
        cc = self.cache_controller
        write_acks, load_acks = [], []
        if cc is not None:
            write_acks = self._ready_ack_prefix(cc.ack_write_queue)
            load_acks = self._ready_ack_prefix(cc.ack_load_queue)
        storage_queues, extra_pool_names = (), ()
        if cc is not None and self.enable_storage:
            extra = getattr(cc, "extra_host_mem_release_queues", {})
            extra_pool_names = tuple(extra)
            storage_queues = (
                cc.prefetch_hit_queue,
                cc.ack_prefetch_queue,
                cc.ack_backup_queue,
                cc.host_mem_release_queue,
                *(extra[name] for name in extra_pool_names),
            )
        exporter = getattr(self, "gpu_export_manager", None)
        digest = self._step_sync_digest()
        sd = self.structural_digest_pair()[0] if self.tp_digest_mode() != "off" else 0
        track_idle = getattr(self, "_idle_sync_enabled", False)
        backfill = self.swa_backfill
        backfill_fields, backfill_snapshot = [], None
        if backfill is not None:
            backfill_fields, backfill_snapshot = backfill.lockstep_rows(
                self._pipelined_sync_slots
            )
            if backfill.pending_count() > self._pipelined_sync_slots:
                self.pipelined_sync_stats["backfill_overflow"] += 1
        layout = PipelinedSyncLayout(
            num_storage=len(storage_queues),
            has_export=exporter is not None,
            has_peer=getattr(self, "_peer_kv_tp_participant", False),
            track_idle=track_idle,
            backfill_fields=len(backfill_fields),
        )
        fields = [
            len(write_acks),
            len(load_acks),
            *(q.qsize() for q in storage_queues),
            *(
                [
                    -int(
                        bool(exporter.active)
                        or bool(exporter.broker and exporter.broker.has_work())
                    )
                ]
                if exporter is not None
                else []
            ),
            *([-int(self._peer_control_pending())] if layout.has_peer else []),
            digest,
            -digest,
            sd,
            -sd,
            *([-self._local_hicache_inflight()] if track_idle else []),
            *backfill_fields,
        ]
        vec = torch.tensor(fields, dtype=torch.int64, device="cpu")
        assert vec.numel() == layout.size, (vec.numel(), layout)
        self._pipelined_sync_pending = PendingStepSync(
            work=self._pipelined_reduce_start(vec),
            vec=vec,
            layout=layout,
            generation=self._pipelined_sync_generation,
            write_acks=write_acks,
            load_acks=load_acks,
            storage_queues=storage_queues,
            extra_pool_names=extra_pool_names,
            backfill=backfill_snapshot,
        )

    def _apply_pipelined_sync(self, pending: PendingStepSync) -> None:
        """Apply a step-sync vector started one sync step earlier, in the
        synchronous path's order: the TP checks, write acks, load acks, the
        storage drains, then the backfill rows."""
        t0 = time.perf_counter()
        values = pending.wait()
        self._note_pipelined_sync_wait(time.perf_counter() - t0)
        if pending.generation != self._pipelined_sync_generation:
            # A storage queue was drained or replaced since the snapshot.
            self._idle_sync_all_idle = False
            self.pipelined_sync_stats["discarded"] += 1
            return
        layout = pending.layout
        backfill = self.swa_backfill if pending.backfill is not None else None
        if layout.track_idle:
            # Every rank's in-flight count was 0 at the snapshot.
            self._idle_sync_all_idle = values[layout.idle] == 0
        if backfill is not None:
            b = layout.backfill
            backfill.check_lockstep_digest(
                values[b], values[b + 1], "check_hicache_events"
            )
        r = layout.reclaim_digest
        self._check_step_sync_digest(
            values[r], values[r + 1], "check_hicache_events (pipelined)"
        )
        s = layout.structural_digest
        self._check_digest_pair(values[s], values[s + 1], "check_hicache_events")
        if layout.has_export:
            self._gpu_export_has_work = bool(-values[layout.export])
        if layout.has_peer:
            self._peer_cache_has_work = bool(-values[layout.peer])
        cc = self.cache_controller
        if cc is not None:
            self._settle_acked_prefix(
                cc.ack_write_queue,
                pending.write_acks[: values[layout.WRITE]],
                self._settle_write_ack,
            )
            self._settle_acked_prefix(
                cc.ack_load_queue,
                pending.load_acks[: values[layout.LOAD]],
                self._settle_load_ack,
            )
        if layout.num_storage and cc is not None and self.enable_storage:
            self._apply_pipelined_storage(
                pending,
                values[layout.STORAGE : layout.STORAGE + layout.num_storage],
            )
        if backfill is not None:
            backfill.apply_lockstep_rows(pending.backfill, values[layout.backfill :])
        self.pipelined_sync_stats["applied"] += 1

    def _apply_pipelined_storage(self, pending: PendingStepSync, counts) -> None:
        cc = self.cache_controller
        extra = getattr(cc, "extra_host_mem_release_queues", {})
        current = (
            cc.prefetch_hit_queue,
            cc.ack_prefetch_queue,
            cc.ack_backup_queue,
            cc.host_mem_release_queue,
            *(extra.get(name) for name in pending.extra_pool_names),
        )
        counts = list(counts)
        for i, (then, now) in enumerate(zip(pending.storage_queues, current)):
            if now is not then:
                # Replaced since the snapshot (the storage threads restarted).
                counts[i] = 0
            elif counts[i] > now.qsize():
                # Only an unaccounted out-of-band drain gets here; a blocking
                # get would hang, so drain what is there.
                logger.error(
                    "HiCache pipelined sync: storage queue %d holds %d items, "
                    "fewer than the %d agreed one step ago",
                    i,
                    now.qsize(),
                    counts[i],
                )
                counts[i] = now.qsize()
        if not (any(counts) or self.host_memory_mode == "buffer_only"):
            return
        n_storage_hit, n_ack_prefetch, n_backup, n_release = counts[:4]
        self._drain_storage_control_queues_impl(
            n_storage_hit=n_storage_hit,
            n_ack_prefetch=n_ack_prefetch,
            n_backup=n_backup,
            n_release=n_release,
            extra_release_counts=dict(zip(pending.extra_pool_names, counts[4:])),
            log_metrics=True,
        )

    def _note_pipelined_sync_wait(self, waited_s: float) -> None:
        """Account the time check_hicache_events blocked on the reduction (0
        when it overlapped the step); a summary is logged once a minute."""
        stats = self.pipelined_sync_stats
        stats["wait_s"] += waited_s
        stats["max_wait_s"] = max(stats["max_wait_s"], waited_s)
        log = self._pipelined_sync_log
        log[1] += 1
        log[2] += waited_s
        log[3] = max(log[3], waited_s)
        now = time.monotonic()
        if now - log[0] >= 60.0:
            self._pipelined_sync_log = [now, 0, 0.0, 0.0]
            if not self._pipelined_sync_log_summary:
                return
            logger.info(
                "HiCache pipelined sync: %d sync steps in %.0f s, blocked on the "
                "reduction %.3f ms mean / %.3f ms max; %d discarded, %d backfill "
                "overflows",
                log[1],
                now - log[0],
                1e3 * log[2] / max(1, log[1]),
                1e3 * log[3],
                stats["discarded"],
                stats["backfill_overflow"],
            )

    def _log_storage_metrics(self) -> None:
        """Per-step storage metrics (rank-local, no collective)."""
        storage_metrics = self.cache_controller.storage_backend.get_stats()
        if storage_metrics is None:
            # Backends without native stats (e.g. file, peer_kv) still carry
            # the controller-side prefetch outcome counters: cumulative counts
            # plus an occupancy gauge, so a 1 Hz cadence loses nothing and
            # spares building them on every scheduler step.
            now = time.monotonic()
            last = getattr(self, "_storage_counters_logged_at", None)
            if last is not None and now - last < STORAGE_COUNTERS_LOG_INTERVAL_S:
                return
            self._storage_counters_logged_at = now
            storage_metrics = StorageMetrics()
        storage_metrics.prefetch_stats = self.prefetch_outcome_stats_snapshot()
        self.storage_metrics_collector.log_storage_metrics(storage_metrics)

    @timed_phase("ready_to_load_host_cache")
    def ready_to_load_host_cache(self) -> int:
        """Notify the cache controller to start the KV cache loading."""
        if self.cache_controller is not None:
            return self.cache_controller.start_loading()
        return 0

    def is_load_back_event_done(self, consumer_index: int) -> bool:
        """Return whether this rank's local load-back event is complete.

        This probe must stay collective-free: TP ranks can observe their local
        events on different scheduler ticks.
        """
        if consumer_index < 0 or self.cache_controller is None:
            return True

        finish_event = self.cache_controller.layer_done_counter.events[
            consumer_index
        ].finish_event
        return finish_event.query()

    # ---- Query / Inspection APIs ----
    # These APIs exist for compatibility with other RadixTree implementations.
    # TODO: simplify and consolidate in a future refactor.

    @property
    def sliding_window_size(self):
        return self._sliding_window_size

    def swa_reprefill_tail_tokens(self) -> int:
        """
        Only unified_kv + HiCache needs this: SWA lives in a per-request ring
        (state_slot/pos), not content-stable and never offloaded to host, so a
        reused prefix's trailing sliding window would read another request's
        stale ring slots. Re-prefilling that window rewrites this request's ring
        (what plain radix reuse does via its SWA match gate). 0 for every other
        layout.
        """
        swa = self.components.get(ComponentType.SWA)
        unified_compress_only_hicache = (
            self.cache_controller is not None
            and swa is not None
            and not self.tree_core.has_swa_host_pool
        )
        return swa.sliding_window_size if unified_compress_only_hicache else 0

    def swa_retain_floor(self, req) -> int | None:
        export_cache_floor = getattr(req, "swa_export_cache_retain_floor", None)
        if export_cache_floor is not None:
            return export_cache_floor
        if (
            getattr(self, "_native_prefill_image_swa", False)
            and self._sliding_window_size is not None
        ):
            from decoupled_engine.srt.layers.attention.dsv4.visible_window import (
                image_restart_swa_retain_floor,
            )

            image_floor = image_restart_swa_retain_floor(
                getattr(req, "multimodal_inputs", None),
                len(req.origin_input_ids),
                self._sliding_window_size,
                self.page_size,
            )
            if image_floor is not None:
                frozen_floor = req.decode_radix_cache_frozen_swa_floor
                return (
                    image_floor
                    if frozen_floor is None
                    else min(image_floor, frozen_floor)
                )
        if req.decode_radix_cache_frozen_swa_floor is not None:
            # The frozen prompt publishes this tail as its checkpoint at finish.
            return req.decode_radix_cache_frozen_swa_floor
        if not self.is_mamba_enabled or self._sliding_window_size is None:
            return None
        checkpoint = req.mamba_last_track_seqlen
        if checkpoint is None:
            return None
        return checkpoint - self._sliding_window_size

    def supports_swa(self) -> bool:
        return self.is_swa_enabled

    def supports_mamba(self) -> bool:
        return self.is_mamba_enabled

    # ---- Session radix cache API (delegates to composed UnifiedSessionRefTracker) ----

    def open_radix_session(self, session_id: str) -> Optional[int]:
        return self.session_refs.open_radix_session(session_id)

    def _make_kv_session_retention(self):
        """The --enable-kv-session-retention registry, else None.

        Reads SGLANG_KV_SESSION_ACTIVE_SOFT_PIN once; 0 is the ablation arm
        (the host eviction key without the active level). Reads
        SGLANG_KV_SESSION_RETENTION_ORDER once (lru | predicted_return_fixed |
        predicted_return, the order inside class 1; retention_order.py) and,
        for predicted_return_fixed, SGLANG_KV_SESSION_RETENTION_MTAU_JSON; a
        bad value refuses to start. Without retention the order flag has no
        effect (warning).
        """
        order = envs.SGLANG_KV_SESSION_RETENTION_ORDER.get()
        if not self.enable_kv_session_retention:
            if (order or ORDER_DEFAULT).strip().lower() != ORDER_DEFAULT:
                logger.warning(
                    "SGLANG_KV_SESSION_RETENTION_ORDER=%s has no effect without "
                    "--enable-kv-session-retention",
                    order,
                )
            return None
        order = parse_retention_order(order)
        fixed_table = None
        if order == ORDER_PREDICTED_RETURN_FIXED:
            fixed_table = load_fixed_return_table(
                envs.SGLANG_KV_SESSION_RETENTION_MTAU_JSON.get()
            )
        return KvSessionRetention(
            self.tree_core,
            self._apply_cache_actions,
            active_soft_pin=envs.SGLANG_KV_SESSION_ACTIVE_SOFT_PIN.get(),
            order=order,
            fixed_table=fixed_table,
        )

    @timed_phase("kv_session.arrival")
    def note_kv_session_arrival(self, req) -> None:
        """Admit full request input once, before capped matching/prefetch begins."""
        if self.kv_sessions is None or getattr(req, "_kv_session_arrival_seen", False):
            return
        key = RadixKey(
            req.origin_input_ids[:],
            req.extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=req.cache_salt,
        )
        req._kv_session_arrival_seen = True
        self.kv_sessions.admit(req, key)

    def kv_session_control(self, operations) -> list[dict]:
        """Apply ordered scheduler control operations; never called by IO threads.

        Mutations are conditional on the worker's current boot/flush identity.
        A status read does not refresh LRU and readiness is never a lasting pin.
        """
        validate_kv_session_operations(operations)
        if self.kv_sessions is None:
            return [
                dict(
                    session_id=op.get("session_id"),
                    version=op.get("version"),
                    status="unknown",
                )
                for op in operations
            ]
        results = []
        for op in operations:
            if (
                op.get("expected_boot_id") is not None
                and op["expected_boot_id"] != self.kv_session_boot_id
            ) or (op.get("op") != "status" and self.kv_session_boot_id is None):
                results.append(
                    dict(
                        session_id=op.get("session_id"),
                        version=op.get("version"),
                        status="unknown",
                    )
                )
            else:
                try:
                    results.append(self.kv_sessions.control(op))
                except ValueError as exc:
                    results.append(
                        dict(
                            session_id=op.get("session_id"),
                            version=op.get("version"),
                            status="unknown",
                            error=str(exc),
                        )
                    )
        return results

    def ensure_session_generation(self, session_id: str) -> int:
        return self.session_refs.ensure_session_generation(session_id)

    def release_radix_session(self, session_id: str) -> int:
        return self.session_refs.release_radix_session(session_id)

    # ---- Streaming session API (delegates to composed StreamingSession) ----

    def supports_streaming_session(self) -> bool:
        return True

    def release_session(self, session_id: str) -> None:
        self.session.release_session(session_id)

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_tokens(active_pool_idxs)

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_full_tokens(active_pool_idxs)

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_swa_tokens(active_pool_idxs)

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_req_count(active_pool_idxs)

    def session_held_mamba_slots(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_mamba_slots(active_pool_idxs)

    def evictable_size(self) -> int:
        return self.tree_core.evictable_size()

    def protected_size(self) -> int:
        return self.tree_core.protected_size()

    def full_evictable_size(self) -> int:
        return self.tree_core.full_evictable_size()

    def full_protected_size(self) -> int:
        return self.tree_core.full_protected_size()

    def swa_evictable_size(self) -> int:
        return self.tree_core.swa_evictable_size()

    def swa_admission_evictable_size(self) -> int:
        """SWA tokens admission may count on reclaiming by eviction.

        ``swa_evictable_size()`` stays raw because PoolStats and the leak
        invariant read it. Window residency subtracts the unlocked device SWA
        that has no host copy yet, which eviction must not drop.
        """
        return self.swa_evictable_size() - self._swa_blocked_size()

    def _swa_blocked_size(self) -> int:
        if not self._swa_window:
            return 0
        return self.tree_core.swa_blocked_size_

    def dsv4_window_info(self) -> dict:
        """``/server_info`` state of window residency. The keys are fixed;
        values stay zero until the state that backs each one exists."""
        registry = self.swa_ledger_registry
        return {
            "swa_residency": self.dsv4_uniform_swa_residency,
            "pools": {
                "swa": {
                    "detached": self.swa_detached_size(),
                    "blocked": self._swa_blocked_size(),
                }
            },
            "ledger_live": registry.ledger_live if registry is not None else 0,
            "spill_inflight_ops": registry.live_count() if registry is not None else 0,
            "tree_host_tokens": 0,
            "host_hole_nodes": 0,
            # SWA window load-backs submitted (a hit whose trailing window is
            # host-only, e.g. inside pages a request spilled).
            "load_back": {
                "swa_tokens": self._swa_load_back_tokens,
                "ops": self._swa_load_back_ops,
            },
            # SGLANG_DSV4_SWA_COPY_VERIFY counters, "<phase>_<pool>_<rows|bad|
            # skipped>"; empty when the check is off.
            "copy_verify": dict(self._swa_copy_verify_stats or {}),
            # Cumulative spill-ledger counters (tokens, except stalls, drains
            # and spill_bytes).
            "spill": dict(
                registry.stats
                if registry is not None
                else dict.fromkeys(SwaLedgerRegistry.STAT_KEYS, 0)
            ),
            # Host->host SWA backfill (None when off).
            "backfill": (
                None
                if self.swa_backfill is None
                else {
                    "pending": self.swa_backfill.pending_count(),
                    "inflight_bytes": self.swa_backfill.inflight_bytes(),
                    "host_tokens": self.swa_backfill.host_tokens(),
                    **self.swa_backfill.stats,
                }
            ),
        }

    def swa_detached_size(self) -> int:
        """Device SWA tokens detached by spills awaiting their ack: allocated,
        yet owned by no request row and no tree node (the leak invariant's
        ``detached`` term)."""
        registry = self.swa_ledger_registry
        return registry.detached_tokens if registry is not None else 0

    # ---- DSv4 window SWA residency: tree-side upkeep ----

    @property
    def _swa_window(self) -> bool:
        return self.dsv4_uniform_swa_residency == "window"

    def on_batch_result_processed(self) -> None:
        """The scheduler processed one more batch result: every node inserted
        before this call now holds final bytes."""
        if self._swa_window:
            self._swa_batches_processed += 1

    def on_no_batch_in_flight(self) -> None:
        """The scheduler is idle with every launched batch processed: queued
        nodes hold final bytes, so a re-backup waiting for the next result is
        ready now instead of waiting for a batch that may never run."""
        if not self._swa_window:
            return
        queue = self.tree_core.swa_rebackup_queue
        for node_id, ready in queue.items():
            if ready > self._swa_batches_processed:
                queue[node_id] = self._swa_batches_processed

    def _enqueue_swa_rebackup(
        self, node_id: Optional[NodeId], after_next_result: bool = False
    ) -> None:
        """Queue an unbacked node for retry_swa_rebackup. ``after_next_result``
        holds it until one more batch result is processed, because its chunk
        may still be in flight (an insert whose forward is not processed)."""
        if node_id is None or self.tree_core.swa_rebackup_action(node_id) is None:
            return
        ready = self._swa_batches_processed + (1 if after_next_result else 0)
        queue = self.tree_core.swa_rebackup_queue
        queue[node_id] = max(queue.get(node_id, ready), ready)

    def _backup_unbacked_chain(self, node_id: NodeId) -> None:
        action = self.tree_core.swa_rebackup_action(node_id)
        if action is not None:
            self._execute_and_commit_kv_backup(action)

    def defer_chunk_backups_after_launch(self) -> bool:
        """SGLANG_DSV4_CHUNK_WINDOW_WORK_AFTER_LAUNCH: from now on a chunked
        stash (cache_unfinished_req) queues its deferred-by-one-stash backup
        instead of running it. Only an overlap loop that calls
        run_deferred_chunk_backups after every launch, and in every iteration
        without one, may turn this on. True when queueing applies (window
        residency); otherwise the backup stays inline."""
        if not self._swa_window:
            return False
        if self._deferred_chunk_backups is None:
            self._deferred_chunk_backups = []
        return True

    def run_deferred_chunk_backups(self) -> None:
        """Run the chunk-boundary backups this iteration's stashes queued, in
        queue order (stashes are lockstep, so every rank runs the same ones).

        The overlap loop calls this after run_batch and before the WAR
        barrier, so the host work (host alloc / eviction, write submission)
        overlaps the forward just launched instead of leaving the GPU idle
        before it. Same backups as inline: each chain ends at the stash before
        the one that queued it, whose forward was processed (final bytes);
        the launched forward only reads it. The write stream orders after the
        schedule stream as at the stash: fenced behind the previous forward,
        not yet behind the new one. A backup submits copies and commits host
        values but frees no device memory (that happens at its ack), so no
        allocation of this iteration waits for it. A chain evicted or backed
        up since its stash (e.g. its request was released) yields no action;
        a released request's own path covers what is left (finish insert, or
        the rebackup queue of a release without insert)."""
        pending = self._deferred_chunk_backups
        if not pending:
            return
        self._deferred_chunk_backups = []
        for node_id in pending:
            self._backup_unbacked_chain(node_id)

    def retry_swa_rebackup(self) -> None:
        """Back up queued unbacked nodes whose bytes are final, while the host
        keeps its headroom. Every rank reaches the same decisions: the queue,
        the processed-result count and the host counts are rank-uniform."""
        queue = self.tree_core.swa_rebackup_queue
        if not self._swa_window or not queue or self.cache_controller is None:
            return
        if self._swa_host_shortfall() > 0:
            return
        for node_id, ready in list(queue.items()):
            if ready > self._swa_batches_processed:
                continue
            self._backup_unbacked_chain(node_id)
            if self.tree_core.swa_rebackup_action(node_id) is not None:
                break  # host still full; keep the rest queued in order
            queue.pop(node_id, None)

    def _release_swa_lag_lock(self, req: Req) -> None:
        lag_lock = getattr(req, "swa_lag_lock", None)
        if lag_lock is not None:
            req.swa_lag_lock = None
            self.dec_lock_ref(lag_lock.node_id, lag_lock.params)

    def _swa_host_shortfall(self) -> int:
        """Host tokens missing to the SWA_HOST_LOW_WATERMARK headroom over the
        pools a window-residency backup allocates from (KV anchor and SWA)."""
        group = self.host_pool_group
        if group is None:
            return 0
        watermark = envs.SGLANG_DSV4_SWA_HOST_LOW_WATERMARK.get()
        shortfall = 0
        for name in (None, PoolName.SWA):
            if name is not None and name not in group.entry_map:
                continue
            pool = group.get_pool(name)
            target = int(pool.size * watermark)
            shortfall = max(shortfall, target - pool.available_size())
        return ceil_align(shortfall, self.page_size) if shortfall > 0 else 0

    def maintain_swa_host_headroom(self) -> int:
        """Keep SWA_HOST_LOW_WATERMARK of the host free:
        evict host-only leaves first, then demote and host-delete the coldest
        backed-up device leaves (W8b). Returns the FULL host tokens freed."""
        if not self._swa_window or self.cache_controller is None:
            return 0
        shortfall = self._swa_host_shortfall()
        if shortfall <= 0:
            return 0
        freed = self.evict_host(shortfall)
        shortfall = self._swa_host_shortfall()
        if shortfall > 0:
            result = self.tree_core.drop_backed_device_leaves(shortfall)
            self._free_values(result.device_frees, result.host_frees)
            freed += result.tracker.get(BASE_COMPONENT_TYPE, 0)
        metrics = self.dsv4_window_metrics
        if freed and metrics is not None:
            metrics.host_maintainer_dropped_tokens.labels(**metrics.labels).inc(freed)
        return freed

    # ---- DSv4 window SWA residency: request spill ledger ----

    def submit_swa_spill(
        self, ranges: Sequence[tuple[Req, int, int]], *, decode: bool = False
    ) -> Optional[int]:
        """Spill request-private SWA: copy positions
        [lo, hi) of every (req, lo, hi) to host SWA slots in one D2H op and
        detach their device slots, which the op frees at its ack.

        ``lo`` must be the request's max(swa_evicted_seqlen,
        cache_protected_len), so each ledger covers one contiguous range.
        ``decode``: a decode batch's spill, which picks the copy path by role
        (swa_spill_sm_copy). Returns the op's ack id, or None when the host
        cannot take the spill: the device keeps the SWA and the caller
        retries at a later step.
        """
        registry = self.swa_ledger_registry
        if registry is None or self.cache_controller is None or not ranges:
            return None
        page_size = self.page_size
        total = 0
        for req, lo, hi in ranges:
            assert (
                lo == max(req.kv.swa_evicted_seqlen, req.cache_protected_len)
                and lo % page_size == 0
                and hi % page_size == 0
                and lo < hi <= req.kv.kv_allocated_len
            ), (req.rid, lo, hi, req.kv.swa_evicted_seqlen, req.cache_protected_len)
            total += hi - lo
        # Detach only once the host slots exist. Host pools and reserves are
        # identical on every rank, so the outcome is rank-uniform.
        host = self.host_pool_group.alloc(
            total,
            pool=PoolName.SWA,
            reclaim=lambda n: self.evict_host(n, ComponentType.SWA),
        )
        if host is None:
            registry.stats["stalls"] += 1
            self._update_swa_window_metrics()
            return None

        req_to_token = self.req_to_token_pool.req_to_token
        full = torch.cat(
            [req_to_token[req.req_pool_idx, lo:hi] for req, lo, hi in ranges]
        ).to(torch.int64)
        allocator = self.token_to_kv_pool_allocator
        # One call for the whole batch: the detach checks cost one host sync.
        swa = allocator.detach_swa(full)
        swa_xfer = PoolTransfer(
            name=PoolName.SWA, device_indices=swa, host_indices=host
        )
        # SWA-anchored sidecars ride along; KV-anchored ones see no KV indices.
        kv_xfer = PoolTransfer(name=PoolName.KV, device_indices=full[:0])
        extra_pools = [
            swa_xfer,
            *self._build_sidecar_transfers(
                CacheTransferPhase.BACKUP_HOST,
                kv_xfer,
                {ComponentType.SWA: [swa_xfer]},
            ),
        ]
        ack_id = registry.next_ack_id()
        # SM copy: a prefill-side spill must not hold the D2H copy engine,
        # which every later device->host readback (.item(), .cpu(), batch
        # results) waits behind; a decode engine's spill takes the staged
        # copy, which slows its step less (SGLANG_DSV4_SWA_SPILL_SM_COPY).
        pool_transfers = self.cache_controller.write_aux(
            extra_pools,
            ack_id=ack_id,
            d2h_sm_copy=swa_spill_sm_copy(decode),
        )
        if pool_transfers is not None and self._swa_copy_verify:
            self._swa_copy_verify_spills[ack_id] = list(pool_transfers)
        if pool_transfers is None:
            allocator.reattach_swa(full, swa)
            self.components[ComponentType.SWA].free_host_values([host])
            registry.stats["stalls"] += 1
            self._update_swa_window_metrics()
            return None

        ops = []
        offset = 0
        for req, lo, hi in ranges:
            n = hi - lo
            ledger = getattr(req, "swa_ledger", None)
            if ledger is None:
                ledger = req.swa_ledger = SwaHostLedger(base=req.cache_protected_len)
                registry.ledger_live += 1
            op = SwaSpillOp(
                ack_id=ack_id,
                start=lo,
                end=hi,
                full_idx=full[offset : offset + n],
                swa_dev_idx=swa[offset : offset + n],
                host_idx=host[offset : offset + n],
                ledger=ledger,
                page_keys=self._peer_spill_page_keys(req, lo, hi),
            )
            ledger.spills.append(op)
            ops.append(op)
            req.kv.swa_evicted_seqlen = hi
            offset += n
        registry.add(ack_id, ops)
        self._update_swa_window_metrics()
        return ack_id

    def _check_swa_copies(
        self, transfers: Optional[list[PoolTransfer]], phase: str, skip: bool = False
    ) -> None:
        """Debug (SGLANG_DSV4_SWA_COPY_VERIFY): compare the device and host
        rows of the SWA-anchored transfers (SWA, c4_state, c4_indexer_state)
        of one finished copy, bit for bit. Runs after the copy's finish event,
        while its device rows still hold the copied bytes: a spill before its
        detached slots are freed, a load-back before its node is unlocked."""
        from decoupled_engine.srt.mem_cache.memory_pool_host import (
            DeepSeekV4PagedHostPool,
            DeepSeekV4StateHostPool,
        )

        stats = self._swa_copy_verify_stats
        for xfer in transfers or ():
            if not (
                xfer.name == PoolName.SWA or xfer.indices_from_pool == PoolName.SWA
            ):
                continue
            name = str(xfer.name)
            if skip:
                key = f"{phase}_{name}_skipped"
                stats[key] = stats.get(key, 0) + 1
                continue
            dev_idx, host_idx = xfer.device_indices, xfer.host_indices
            if dev_idx is None or host_idx is None or dev_idx.numel() == 0:
                continue
            entry = self.host_pool_group.entry_map.get(xfer.name)
            host_pool = getattr(entry, "host_pool", None)
            tail = None
            if isinstance(host_pool, DeepSeekV4PagedHostPool):
                rows_per_page = host_pool.slot_page_size
                row_bytes = getattr(
                    host_pool, "device_page_bytes", host_pool.item_bytes
                )
                # SGLANG_DSV4_WINDOW_HOST_TAIL_ONLY: the host holds the tail
                # bytes of each page only; compare those.
                tail = getattr(host_pool, "tail_layout", None)
                device_rows = [
                    buf.view(torch.uint8).reshape(-1, row_bytes)
                    for buf in host_pool.device_buffers
                ]
            elif isinstance(host_pool, DeepSeekV4StateHostPool):
                rows_per_page = host_pool.swa_page_size
                tail = getattr(host_pool, "tail_layout", None)
                device_rows = list(host_pool.device_page_views)
            else:
                continue
            # SGLANG_DSV4_WINDOW_HOST_TAIL_POISON_HEAD overwrites these device
            # bytes after a load; the load check skips them.
            poisoned = (
                getattr(host_pool, "poison_byte_ranges", ()) if phase == "load" else ()
            )
            host_buf = getattr(host_pool, "kv_buffer", None)
            if host_buf is None or dev_idx.numel() % rows_per_page:
                # Dedup follower (no local host bytes) or a partial page.
                key = f"{phase}_{name}_skipped"
                stats[key] = stats.get(key, 0) + 1
                continue
            dev_pages = dev_idx.reshape(-1, rows_per_page)[:, 0] // rows_per_page
            host_pages = (
                host_idx.reshape(-1, rows_per_page)[:, 0] // rows_per_page
            ).cpu()
            bad = 0
            bad_layers = []
            for layer, rows in enumerate(device_rows):
                dev = rows[dev_pages.to(rows.device)].cpu()
                offset = 0
                if tail is not None:
                    offset = tail.device_start_byte
                    dev = dev[:, offset:]
                    host = host_buf.view(
                        -1, tail.tail_items, host_buf.shape[1], tail.item_bytes
                    )[host_pages, :, layer].reshape(dev.shape)
                elif host_pool.layout == "layer_first":
                    host = host_buf[layer][host_pages]
                else:
                    host = host_buf[host_pages, layer].reshape(dev.shape)
                for lo, hi in poisoned:
                    lo, hi = max(lo - offset, 0), min(hi - offset, dev.shape[1])
                    if lo < hi:
                        dev[:, lo:hi] = host[:, lo:hi]  # dev is a CPU copy
                layer_bad = int((dev != host).any(dim=-1).sum())
                if layer_bad:
                    bad += layer_bad
                    bad_layers.append(layer)
            for suffix, value in (("rows", int(dev_pages.numel())), ("bad", bad)):
                key = f"{phase}_{name}_{suffix}"
                stats[key] = stats.get(key, 0) + value
            if bad:
                logger.error(
                    "SWA copy check failed at %s: pool %s, %d of %d page rows "
                    "differ (layers %s)",
                    phase,
                    name,
                    bad,
                    int(dev_pages.numel()) * len(device_rows),
                    bad_layers[:8],
                )

    def _finish_swa_spill_ack(self, ack_id: int, ack: HiCacheAck) -> None:
        """Settle a spill op at its MIN-reduced ack: only
        now may its detached device slots and its host slots be freed."""
        registry = self.swa_ledger_registry
        ops = registry.pop(ack_id) if registry is not None else None
        if ops is None:
            return
        if self._swa_copy_verify:
            transfers = self._swa_copy_verify_spills.pop(ack_id, None)
            # Re-attached slots belong to a request or the tree again and may
            # have been rewritten; in-flight and orphaned ones are untouched.
            self._check_swa_copies(
                transfers, "spill", skip=any(op.state == REATTACHED for op in ops)
            )
        device_frees, host_frees = [], []
        for op in ops:
            if op.state == INFLIGHT:
                # The copy is final: the host slice joins the request's ledger.
                device_frees.append(op.swa_dev_idx)
                op.ledger.spills.remove(op)
                op.ledger.add_segment(
                    SwaSegment(op.start, op.end, op.host_idx, page_keys=op.page_keys)
                )
                registry.stats["acked_tokens"] += op.num_tokens
                if op.page_keys and self.peer_kv_servicer is not None:
                    # SGLANG_DSV4_PEER_PUBLISH_SPILL_SWA: servable from now on.
                    self.peer_kv_servicer.push_swa_store(
                        op.page_keys, self._host_pages(op.host_idx)
                    )
            elif op.state == REATTACHED:
                # The device slots went back to the request or into the tree.
                host_frees.append(op.host_idx)
            else:
                # Orphaned: the request is gone.
                device_frees.append(op.swa_dev_idx)
                host_frees.append(op.host_idx)
            op.ledger = None
        if device_frees:
            self.token_to_kv_pool_allocator.free_detached_swa(torch.cat(device_frees))
        if host_frees:
            self.components[ComponentType.SWA].free_host_values([torch.cat(host_frees)])
        registry.stats["spill_bytes"] += int(getattr(ack, "num_bytes", 0) or 0)
        self._update_swa_window_metrics()

    def reattach_swa_spills(self, req: Req) -> None:
        """Map a request's in-flight spilled pages back to its row, so an
        insert never adopts a host slice whose copy is still running; the op's
        late ack frees only its host slots."""
        registry = self.swa_ledger_registry
        ledger = getattr(req, "swa_ledger", None)
        if registry is None or ledger is None:
            return
        ops = ledger.inflight_ops()
        if not ops:
            return
        if envs.SGLANG_DSV4_SWA_WINDOW_CHECK.get():
            row = self.req_to_token_pool.req_to_token[req.req_pool_idx]
            for op in ops:
                assert torch.equal(
                    row[op.start : op.end].to(torch.int64), op.full_idx
                ), (req.rid, op.start, op.end)
        self.token_to_kv_pool_allocator.reattach_swa(
            torch.cat([op.full_idx for op in ops]),
            torch.cat([op.swa_dev_idx for op in ops]),
        )
        for op in ops:
            registry.mark_reattached(op)
            ledger.spills.remove(op)
            op.ledger = None
        # Ops ack in submit order, so the acked slices end where the first
        # re-attached op starts; the device holds everything from there on.
        req.kv.swa_evicted_seqlen = ops[0].start
        self._update_swa_window_metrics()

    def prepare_swa_window_insert(
        self, req: Req, insert_params: InsertParams, token_ids_len: int
    ) -> Optional[int]:
        """Finished-request insert under window residency.

        Re-attaches in-flight spills, then hands the acked host slices over
        [cache_protected_len, swa_evicted_seqlen) to the insert. Where they do
        not cover that range, returns the gap start as the insert length:
        the positions past it would have FULL but no SWA.
        """
        # A backfill still running leaves a coverage gap: truncate there
        # rather than wait (a backfill can be slower than a short decode).
        self.cancel_swa_backfill(req)
        self.reattach_swa_spills(req)
        sev = req.kv.swa_evicted_seqlen
        cps = req.cache_protected_len
        insert_params.swa_evicted_seqlen = sev
        ledger = getattr(req, "swa_ledger", None)
        if ledger is not None:
            insert_params.swa_host_segments = [
                (seg.start, seg.end, seg.host_idx) for seg in ledger.segments
            ]
        if sev <= cps:
            return None
        resume_prefix = False
        if ledger is None:
            covered = cps
        elif ledger.base <= cps:
            covered = ledger.covered_end()
        elif req.last_node is None:
            # A resumed PD decode request (retract with a
            # CPU snapshot) holds no tree lock and protects no prefix, yet its
            # ledger starts at base = the prefix the tree owned at admission.
            # The tree still covers [0, base) (every node has SWA on some
            # tier, I1) unless that prefix was evicted meanwhile; then clamp
            # where the tree's copy ends, since past it the insert would
            # create FULL without SWA.
            tree_len = self.tree_core.match_full_prefix_len(
                RadixKey(
                    (req.origin_input_ids + req.output_ids)[:token_ids_len],
                    req.extra_key,
                    is_bigram=self.tree_core.is_eagle,
                    cache_salt=req.cache_salt,
                )
            )
            if tree_len >= ledger.base:
                covered = ledger.covered_end()
            else:
                covered = tree_len
                resume_prefix = True
        else:
            covered = cps
        if covered >= sev:
            return None
        truncated = max(0, token_ids_len // self.page_size * self.page_size - covered)
        req.swa_insert_truncated_tokens = (
            getattr(req, "swa_insert_truncated_tokens", 0) + truncated
        )
        if self.swa_ledger_registry is not None:
            stats = self.swa_ledger_registry.stats
            stats["truncated_tokens"] += truncated
            if resume_prefix:
                stats["resume_truncated_tokens"] += truncated
            self._update_swa_window_metrics()
        logger.warning(
            "window SWA insert of %s truncated at %d: %s covers only "
            "[%d, %d) of the spilled range [%d, %d)",
            req.rid,
            covered,
            "the tree (resumed request)" if resume_prefix else "host SWA",
            cps,
            covered,
            cps,
            sev,
        )
        return covered

    def settle_swa_ledger(
        self, req: Req, insert_result: Optional[InsertResult]
    ) -> None:
        """Close a finished request's ledger after cache_finished_req.

        After an insert the tree owns the adopted slices; every other acked
        slice duplicates a tree copy (below the new leaf) or lies past a
        clamped insert, so it is freed. Without an insert the request's
        disposition decides (release_swa_ledger).
        """
        registry = self.swa_ledger_registry
        ledger = getattr(req, "swa_ledger", None)
        if registry is None or ledger is None:
            return
        if insert_result is None:
            self.release_swa_ledger(
                req,
                getattr(req, "swa_ledger_disposition", None)
                or LedgerDisposition.RELEASE,
            )
            return
        lo, hi = insert_result.swa_host_adopted or (0, 0)
        frees = []
        for seg in ledger.segments:
            for a, b in ((seg.start, min(seg.end, lo)), (max(seg.start, hi), seg.end)):
                if a < b:
                    frees.append(seg.host_idx[a - seg.start : b - seg.start])
                    registry.stats["dup_freed_tokens"] += b - a
                    self._peer_retract_spill_swa(seg, a, b)
            registry.stats["adopted_tokens"] += max(
                0, min(seg.end, hi) - max(seg.start, lo)
            )
        # Defensive: prepare_swa_window_insert re-attached every in-flight op.
        self._orphan_swa_spills(ledger)
        if frees:
            self.components[ComponentType.SWA].free_host_values([torch.cat(frees)])
        req.swa_ledger = None
        registry.ledger_live -= 1
        self._update_swa_window_metrics()

    def release_swa_ledger(
        self,
        req: Req,
        disposition: LedgerDisposition = LedgerDisposition.RELEASE,
    ) -> None:
        """Release a request's ledger without inserting it: in-flight spills
        become orphans that their ack frees; RELEASE frees the acked host
        slices now, KEEP leaves them on the request for a resume. The
        disposition applies once and then reverts to RELEASE."""
        registry = self.swa_ledger_registry
        ledger = getattr(req, "swa_ledger", None)
        if registry is None or ledger is None:
            return
        if disposition != LedgerDisposition.KEEP:
            # A pending backfill's slots stay with its read until it finishes.
            # KEEP (PD decode retract) lets it run on:
            # its commit lands in the ledger the request keeps for the resume.
            self.cancel_swa_backfill(req)
        self._orphan_swa_spills(ledger)
        req.swa_ledger_disposition = LedgerDisposition.RELEASE
        if disposition == LedgerDisposition.KEEP:
            self._update_swa_window_metrics()
            return
        if ledger.segments:
            registry.stats["released_tokens"] += ledger.segment_tokens()
            for seg in ledger.segments:
                self._peer_retract_spill_swa(seg, seg.start, seg.end)
            self.components[ComponentType.SWA].free_host_values(
                [torch.cat([seg.host_idx for seg in ledger.segments])]
            )
        req.swa_ledger = None
        registry.ledger_live -= 1
        self._update_swa_window_metrics()

    # ---- DSv4 window SWA residency: host->host backfill ----

    def init_swa_backfill(self, backend=None, **kwargs):
        """Create the request backfill manager (window residency with a peer
        storage backend). The backend is resolved from the cache controller at
        each lookup unless given."""
        from decoupled_engine.srt.mem_cache.swa_backfill import SwaBackfillManager

        memory = get_memory()
        kwargs.setdefault(
            "timeout_s", float(getattr(memory, "dsv4_swa_backfill_timeout_s", 60.0))
        )
        kwargs.setdefault(
            "max_inflight_bytes",
            int(
                float(getattr(memory, "dsv4_swa_backfill_max_inflight_gb", 16.0))
                * (1 << 30)
            ),
        )
        kwargs.setdefault("metrics", getattr(self, "dsv4_window_metrics", None))
        self.swa_backfill = SwaBackfillManager(self, backend, **kwargs)
        return self.swa_backfill

    def open_swa_ledger(self, req: Req, base: int) -> SwaHostLedger:
        """The request's ledger, opened at ``base`` (the first position it
        tracks) when it has none."""
        ledger = getattr(req, "swa_ledger", None)
        if ledger is None:
            ledger = req.swa_ledger = SwaHostLedger(base=base)
            self.swa_ledger_registry.ledger_live += 1
        return ledger

    def submit_swa_backfill(
        self,
        req: Req,
        start: int,
        end: int,
        source: Optional[str],
        page_hashes: Sequence[str],
        local_reason: Optional[str] = None,
    ) -> bool:
        """Queue a host->host backfill of request positions [start, end) from
        the owner servicer ``source``. Opens the
        request's ledger at ``start`` if needed. Returns False when backfill is
        off, so the finish insert truncates at ``start``. Fewer
        ``page_hashes`` than pages (``swa_backfill_page_keys`` cannot key a
        bigram page whose closing token does not exist yet, and no owner can
        have published it) shorten the range to the keyed pages. ``source``
        None (a rank-local "no usable owner" verdict, ``local_reason``) still
        queues a job so the job list stays rank-uniform; it ends
        no_capability on every rank (SwaBackfillManager.submit)."""
        if self.swa_backfill is None or end <= start:
            return False
        keyed_end = start + len(page_hashes) * self.page_size
        if keyed_end < end:
            end = keyed_end
            if end <= start:
                return False
        self.open_swa_ledger(req, start)
        return self.swa_backfill.submit(
            req,
            start=start,
            end=end,
            source=source,
            page_hashes=page_hashes,
            local_reason=local_reason,
        )

    def cancel_swa_backfill(self, req: Req) -> None:
        if self.swa_backfill is not None:
            self.swa_backfill.cancel(req)

    def _peer_spill_page_keys(self, req: Req, lo: int, hi: int) -> Optional[list]:
        """Peer page keys of a spilled range [lo, hi) to publish at its ack
        (SGLANG_DSV4_PEER_PUBLISH_SPILL_SWA), equal to the reader's
        swa_backfill_page_keys(req, lo, hi); None when off or when a page
        cannot be keyed yet. The request caches its chain hashes, so each page
        is hashed once over all of its spills."""
        if (
            self.peer_kv_servicer is None
            or not envs.SGLANG_DSV4_PEER_PUBLISH_SPILL_SWA.get()
        ):
            return None
        from decoupled_engine.srt.mem_cache.peer_kv_namespace import peer_page_keys
        from decoupled_engine.srt.mem_cache.utils import get_hash_str

        page = self.page_size
        bigram = bool(self.tree_core.is_eagle)
        prompt, output = req.origin_input_ids, req.output_ids
        # A bigram page key needs the token after the page (swa_backfill_page_keys).
        need = hi + int(bigram)
        if lo % page or hi % page or hi <= lo or need > len(prompt) + len(output):
            return None
        chain = getattr(req, "_peer_swa_chain", None)
        if chain is None:
            chain = req._peer_swa_chain = []
        have = len(chain) * page
        if have < hi:
            # Hash only tokens [have, need), chained on the last cached page.
            n = len(prompt)
            if need <= n:
                tokens = prompt[have:need]
            elif have >= n:
                tokens = output[have - n : need - n]
            else:
                tokens = list(prompt[have:]) + list(output[: need - n])
            key = RadixKey(
                tokens, req.extra_key, is_bigram=bigram, cache_salt=req.cache_salt
            )
            chain.extend(
                get_hash_str(key, chain[-1] if chain else None, page_size=page)
            )
        return peer_page_keys(
            chain[lo // page : hi // page], req.cache_salt, req.extra_key
        )

    def _host_pages(self, host_idx: torch.Tensor) -> list:
        page = self.page_size
        return [host_idx[i : i + page] for i in range(0, len(host_idx), page)]

    def _peer_retract_spill_swa(self, seg: SwaSegment, a: int, b: int) -> None:
        """Retract the SWA-only publication of the pages of ledger slice
        ``seg`` that touch positions [a, b), whose host slots are about to be
        freed (no-op when the slice was not published)."""
        if not seg.page_keys or self.peer_kv_servicer is None:
            return
        page = self.page_size
        i, j = (a - seg.start) // page, -(-(b - seg.start) // page)
        self.peer_kv_servicer.push_swa_removed(
            seg.page_keys[i:j],
            self._host_pages(seg.host_idx[i * page : j * page]),
        )

    def swa_backfill_page_keys(self, req: Req, start: int, end: int) -> list[str]:
        """Peer page keys of request pages [start, end): the chain hashes of
        the request's tokens under the RadixKey its inserts use (extra_key,
        cache_salt, bigram mode), namespaced the way owners publish them.

        A bigram (EAGLE) tree keys page k by the bigrams (t[i], t[i+1]) for i
        in [kP, (k+1)P), i.e. raw tokens [kP, (k+1)P + 1): the owner's page
        aligned insert key holds those P full pairs, so the reader's key
        needs the token after ``end``. Pages whose tokens do not all exist yet
        cannot be keyed (no owner can have published them either) and are
        left out, so the result may be shorter than (end - start) / P."""
        from decoupled_engine.srt.mem_cache.swa_backfill import swa_backfill_page_keys

        page = self.page_size
        assert start % page == 0 and end % page == 0, (start, end)
        bigram = bool(self.tree_core.is_eagle)
        need = end + 1 if bigram else end
        tokens = req.origin_input_ids
        if need > len(tokens):
            tokens = list(tokens) + list(req.output_ids)
        end = min(end, (len(tokens) - int(bigram)) // page * page)
        if end <= start:
            return []
        key = RadixKey(
            tokens[: end + int(bigram)],
            req.extra_key,
            is_bigram=bigram,
            cache_salt=req.cache_salt,
        )
        return swa_backfill_page_keys(key, end // page, page)[start // page :]

    def _orphan_swa_spills(self, ledger: SwaHostLedger) -> None:
        registry = self.swa_ledger_registry
        for op in ledger.spills:
            registry.mark_orphaned(op)
            op.ledger = None
        ledger.spills.clear()

    def drain_swa_spills(self) -> int:
        """Complete every queued spill ack now,
        before a retract decision: each rank waits for its queue through the
        last spill-bearing ack, write-through acks ahead of it
        included, then the usual MIN-reduced writing_check settles them.
        Returns the device SWA tokens freed."""
        registry = self.swa_ledger_registry
        cc = self.cache_controller
        if registry is None or cc is None or registry.detached_tokens == 0:
            return 0
        last = -1
        for i, ack in enumerate(cc.ack_write_queue):
            if any(is_spill_ack_id(ack_id) for ack_id in ack.node_ids):
                last = i
        if last < 0:
            return 0
        for ack in cc.ack_write_queue[: last + 1]:
            ack.finish_event.synchronize()
        allocator = self.token_to_kv_pool_allocator
        before = allocator.swa_available_size()
        self.writing_check()
        registry.stats["drains"] += 1
        self._update_swa_window_metrics()
        return allocator.swa_available_size() - before

    def check_swa_window_residency(self, reqs) -> None:
        """Debug (SGLANG_DSV4_SWA_WINDOW_CHECK, one host sync per request):
        every running request's private SWA [max(swa_evicted_seqlen,
        cache_protected_len), kv_allocated_len) is mapped on the device (I5)."""
        mapping = self.token_to_kv_pool_allocator.full_to_swa_index_mapping
        req_to_token = self.req_to_token_pool.req_to_token
        for req in reqs:
            if not req.is_holding_kv:
                continue
            lo = max(req.kv.swa_evicted_seqlen, req.cache_protected_len)
            hi = req.kv.kv_allocated_len
            if hi > lo:
                rows = req_to_token[req.req_pool_idx, lo:hi].to(torch.int64)
                assert bool(
                    (mapping[rows] > 0).all()
                ), f"window SWA of {req.rid} unmapped in [{lo}, {hi})"

    _SWA_SPILL_EVENTS = (
        ("submitted", "submitted_tokens"),
        ("acked", "acked_tokens"),
        ("reattached", "reattached_tokens"),
        ("orphaned", "orphaned_tokens"),
        ("adopted", "adopted_tokens"),
        ("dup_freed", "dup_freed_tokens"),
    )

    def _update_swa_window_metrics(self) -> None:
        """Publish the ledger counters (deltas since the last call) and the
        spill gauges; only in window mode with metrics on."""
        registry = self.swa_ledger_registry
        metrics = getattr(self, "dsv4_window_metrics", None)
        if registry is None or metrics is None:
            return
        reported = self._swa_metrics_reported
        stats = registry.stats

        def delta(key: str) -> int:
            return stats[key] - reported.get(key, 0)

        labels = metrics.labels
        for event, key in self._SWA_SPILL_EVENTS:
            if delta(key):
                metrics.spill_tokens.labels(**labels, event=event).inc(delta(key))
        if delta("stalls"):
            metrics.spill_stall.labels(**labels, reason="host_full").inc(
                delta("stalls")
            )
        if delta("drains"):
            metrics.drain.labels(**labels).inc(delta("drains"))
        if delta("spill_bytes"):
            metrics.spill_bytes.labels(**labels).inc(delta("spill_bytes"))
        # truncated_tokens counts every truncation; resume_truncated_tokens is
        # the part clamped at an evicted prefix of a resumed request.
        resumed = delta("resume_truncated_tokens")
        if delta("truncated_tokens") - resumed:
            metrics.insert_truncated_tokens.labels(**labels, reason="coverage_gap").inc(
                delta("truncated_tokens") - resumed
            )
        if resumed:
            metrics.insert_truncated_tokens.labels(
                **labels, reason="resume_prefix"
            ).inc(resumed)
        reported.update(stats)
        metrics.spill_inflight_pages.labels(**labels).set(
            registry.detached_tokens // self.page_size
        )
        metrics.device_tokens.labels(**labels, state="req_detached").set(
            registry.detached_tokens
        )
        group = self.host_pool_group
        if group is not None and PoolName.SWA in group.entry_map:
            pool = group.get_pool(PoolName.SWA)
            metrics.host_used_tokens.labels(**labels).set(
                pool.size - pool.available_size()
            )

    def mamba_evictable_size(self) -> int:
        return self.tree_core.mamba_evictable_size()

    def swa_protected_size(self) -> int:
        return self.tree_core.swa_protected_size()

    def mamba_protected_size(self) -> int:
        return self.tree_core.mamba_protected_size()

    def total_size(self) -> tuple[int, int]:
        return self.tree_core.total_size()

    def all_values_flatten(self) -> torch.Tensor:
        return self.tree_core.all_values_flatten()

    def all_mamba_values_flatten(self) -> torch.Tensor:
        return self.tree_core.all_mamba_values_flatten()

    def available_and_evictable_str(self) -> str:
        # TODO(zhangmj): need more detailed log info for session reference.
        if self.supports_swa():
            full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        else:
            full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable = self.tree_core.component_evictable_size(BASE_COMPONENT_TYPE)
        lines = [
            f"Available full tokens: {full_available_size + full_evictable} "
            f"(full_available_size={full_available_size} + full_evictable_size_={full_evictable})"
        ]
        for ct in self.tree_components:
            if ct == BASE_COMPONENT_TYPE:
                continue
            if ct.is_swa:
                available_size = self.token_to_kv_pool_allocator.swa_available_size()
            elif ct.is_mamba:
                available_size = self.req_to_token_pool.mamba_allocator.available_size()
            else:
                continue

            lines.append(
                f"Available {ct}: {available_size + self.tree_core.component_evictable_size(ct)} "
                f"(available_size={available_size} + component_evictable_size_={self.tree_core.component_evictable_size(ct)})"
            )
        return "\n".join(lines) + "\n"

    def sanity_check(self):
        """Verify tree invariants.

        TODO(hzh): This method has relatively high latency; simplify the
        check logic once the tree implementation stabilizes.
        """
        # Skip when streaming sessions hold tree locks: the check asserts
        # all nodes are unlocked during idle, which streaming sessions break
        # by design (they hold a first-turn lock across turns).
        if self.session.any_holding_kv():
            return

        # Pass ongoing ops as lightweight (id, node_id) pairs so the tree core
        # can resolve + validate them without reaching into Controller state.
        if self.buffer_pipeline is not None:
            ongoing_write_through = [
                (nid, entry.intent.node_id)
                for nid, entry in self.buffer_pipeline.ongoing_write_through.items()
            ]
        else:
            ongoing_write_through = [
                (nid, wt.node_id) for nid, wt in self.ongoing_write_through.items()
            ]
        ongoing_load_back = [
            (nid, lb.node_id) for nid, lb in self.ongoing_load_back.items()
        ]
        self.tree_core.sanity_check(ongoing_write_through, ongoing_load_back)

    def pretty_print(self) -> None:
        self.tree_core.pretty_print()

    # ---- TreeCore state delegation ----
    # The facade re-exposes tree-owned config (page_size, enable_storage, ...) so its
    # own coordination methods and external callers read them off the cache.

    # ``page_size`` keeps a setter: StreamingSession forwards assignment onto its
    # inner cache (the PrefixCacheTrait surface).
    @property
    def page_size(self):
        return self.tree_core.page_size

    @page_size.setter
    def page_size(self, value) -> None:
        self.tree_core.page_size = value

    @property
    def enable_storage(self):
        return self.tree_core.enable_storage

    @enable_storage.setter
    def enable_storage(self, value) -> None:
        self.tree_core.enable_storage = value

    @property
    def write_through_threshold(self):
        return self.tree_core.write_through_threshold

    @write_through_threshold.setter
    def write_through_threshold(self, value) -> None:
        self.tree_core.write_through_threshold = value

    @property
    def is_write_back(self):
        return self.tree_core.is_write_back

    @is_write_back.setter
    def is_write_back(self, value) -> None:
        self.tree_core.is_write_back = value

    @property
    def device(self):
        return self.tree_core.device

    @property
    def root_node(self):
        return self.tree_core.root_node

    def take_events(self):
        # Drain the KV event queue from the TreeCore.
        return self.tree_core.take_events()

    def kv_event_recorder(self):
        # The TreeCore's recorder (SGLANG_KV_EVENTS_BATCH_PUBLISH).
        return self.tree_core.kv_events

    def enable_skip_d_role_kv_events(self):
        """SGLANG_SYMMETRIC_SKIP_D_ROLE_KV_EVENTS (every TP rank): from now on
        the nodes that cache calls of D-leg requests add publish no KV events
        (mem_cache/events.py DRoleEventGate). Returns the gate."""
        return self.tree_core.enable_skip_d_role_kv_events()

    @property
    def skips_d_role_kv_events(self) -> bool:
        """True once enable_skip_d_role_kv_events ran: the symmetric scheduler
        then marks D-leg requests (req.kv_events_d_role)."""
        return getattr(self.tree_core.kv_events, "d_role_gate", None) is not None

    def resolve_node_handle(self, node_handle):
        """Look up the node object from its NodeId.

        TODO(Jialin): Remove after the Unified Radix Cache split.
        """
        if isinstance(node_handle, int):
            return self.tree_core.node_by_id(node_handle)
        # Internal callers (and the session sentinel / None) pass a non-int through.
        return node_handle

    def root_node_handle(self, extra_key: Optional[str] = None) -> NodeId:
        """The root's NodeId -- URC match results carry NodeIds."""
        return self.tree_core.root_node_handle(extra_key)
