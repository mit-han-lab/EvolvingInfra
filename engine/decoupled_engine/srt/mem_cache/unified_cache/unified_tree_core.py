# Adapted from SGLang python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py (commit 9a489f8d2f), modified for evolving-infra.
"""The radix tree mechanism.

``UnifiedTreeCore`` owns the tree's member-var state -- the tree structure (root
node), the per-component LRU lists, the per-component size counters, the
evictable device/host leaf sets, and the empty match result -- plus ``reset()``.
It also defines the tree's building blocks (``UnifiedTreeNode``,
``UnifiedLRUList``).

The cache builds the component drivers and passes them in; the tree holds them
(``self.components_by_type``) to drive their tree-level hooks. The components hold the
cache for cache-level logic, but the TreeCore itself never touches it.
"""

# TODO(Jialin): Split this file into smaller cohesive modules (e.g. tree
# building blocks, insert/match walks, eviction drivers).

from __future__ import annotations

import heapq
import logging
import sys
import time
from array import array
from collections import OrderedDict, defaultdict
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Sequence

import msgspec
import torch

from decoupled_engine.srt.disaggregation.kv_events import StorageMedium
from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from decoupled_engine.srt.mem_cache.cpu_page_book import cat_tagged, copy_tag, slice_tagged
from decoupled_engine.srt.mem_cache.events import DRoleEventGate, KVCacheEventRecorder
from decoupled_engine.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from decoupled_engine.srt.mem_cache.peer_kv_demote import (
    AGE_FINE_EDGES_S,
    PeerDemoteExtraStats,
    PeerDemoteStats,
    ScopeContext,
    TickClock,
    age_bucket,
    age_fine_bucket,
    release_trigger,
    return_bin,
    select_demote_path,
    select_scope_path,
)
from decoupled_engine.srt.mem_cache.peer_kv_namespace import peer_page_keys
from decoupled_engine.srt.mem_cache.peer_kv_touch import (
    PeerTouchStats,
    child_with_first_key,
    first_page_key,
    resolve_touch_node,
)
from decoupled_engine.srt.mem_cache.radix_cache import RadixKey
from decoupled_engine.srt.mem_cache.unified_cache.cache_action import (
    BackupKV,
    CacheAction,
    ComponentAction,
    FreeDeviceKV,
    FreeDeviceKVFullOnly,
    ReplaceWriteThroughOnNodeSplit,
)
from decoupled_engine.srt.mem_cache.unified_cache.components import (
    _NUM_COMPONENT_TYPES,
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentData,
    ComponentType,
    EvictLayer,
    LRURefreshPhase,
    TreeComponent,
    get_and_increase_time_counter,
    peek_time_counter,
)
from decoupled_engine.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    DecSwaLockOnlyResult,
    DemoteResult,
    DriveHostEvictionResult,
    DropSubtreeNoHostResult,
    EvictDeviceLeafResult,
    EvictDeviceNextNodeResult,
    InsertStepResult,
    NodeId,
    RadixCacheWalkResult,
    UnifiedTreeCoreInterface,
)
from decoupled_engine.srt.mem_cache.utils import (
    compute_node_hash_values,
    get_eviction_strategy,
    split_node_hash_value,
)

if TYPE_CHECKING:
    from decoupled_engine.srt.managers.schedule_batch import Req
    from decoupled_engine.srt.mem_cache.cache_init_params import CacheInitParams

logger = logging.getLogger(__name__)

# 42 bits: digest * 1000003 (< 2^20) stays under 2^62, so the update never
# overflows int64 with plain (non-wrapping) arithmetic in the Rust port, and
# the TP consistency check can still all_reduce [digest, -digest] in int64.
_RECLAIM_DIGEST_MASK = (1 << 42) - 1


class StorageBackupSpec(NamedTuple):
    """A node's device->storage backup spec, gathered tree-side."""

    host_value: torch.Tensor
    token_ids: array
    hash_value: list[str]
    prefix_keys: Optional[list[str]]
    comp_xfers: dict[ComponentType, list[PoolTransfer]]


class UnifiedTreeNode:
    counter = 0
    # Lazy retention state: flag-off caches allocate no per-node session set.
    host_session_count = 0
    # Of host_session_count: registrations with a queued or running request.
    host_session_active = 0
    host_session_known = False
    host_session_unknown = False
    host_session_anchors = None
    # SGLANG_SYMMETRIC_SKIP_D_ROLE_KV_EVENTS (mem_cache/events.py
    # DRoleEventGate): True on a node that D-role work created.
    kv_events_d_role = False

    def __init__(self, tree_components: tuple[ComponentType, ...], priority: int = 0):
        # Plain dict (not defaultdict): a missing-key read must raise, never
        # silently mint an unregistered node outside the TreeCore arena.
        self.children: dict[Any, UnifiedTreeNode] = {}
        self.parent: UnifiedTreeNode | None = None
        self.key: Optional[RadixKey] = None
        self.component_types = tree_components
        # list indexed by ComponentType (int enum 0..N-1)
        self.component_data: list[ComponentData] = [
            ComponentData() for _ in range(_NUM_COMPONENT_TYPES)
        ]
        self.last_access_time = get_and_increase_time_counter()
        # CPU retention is independent of native GPU session refs and recency.
        self.host_last_access_time = self.last_access_time
        self.creation_time = get_and_increase_time_counter()
        self.hash_value = None
        # Namespace-aware hashes used only for external KV events.
        self.event_hash_value: Optional[list[str]] = None
        self.hit_count = 0
        self.priority = priority
        self.lru_prev: list[UnifiedTreeNode | None] = [None] * (
            _NUM_COMPONENT_TYPES * 2
        )
        self.lru_next: list[UnifiedTreeNode | None] = [None] * (
            _NUM_COMPONENT_TYPES * 2
        )
        self.id = UnifiedTreeNode.counter
        UnifiedTreeNode.counter += 1
        # Set by _unregister_node; a deleted node keeps its parent pointer,
        # so `evicted` alone cannot tell it from a live one.
        self.deleted: bool = False
        self.write_through_pending_id: Optional[int] = None
        # Anchor NodeId of an in-flight H->D load-back reading this node's
        # host slots; such host copies must not be reclaimed until the ack.
        self.load_back_pending_id: Optional[int] = None
        # --peer-kv-demote-source: a peer pulled this node's whole subtree
        # path and no local access came after; host eviction takes it first.
        # Set only from the broadcast PeerKVDemoteReq (same on every rank).
        self.peer_demoted: bool = False

    def component(self, component_type: ComponentType) -> ComponentData:
        return self.component_data[component_type]

    @property
    def backuped(self) -> bool:
        """Tree-level: Full KV present on host."""
        return self.component_data[ComponentType.FULL].host_value is not None

    @property
    def evicted(self) -> bool:
        """Tree-level: Full KV not on device (non-root with value=None)."""
        return (
            self.parent is not None
            and self.component_data[ComponentType.FULL].value is None
        )

    def __lt__(self, other: UnifiedTreeNode):
        # (tick, id): a tie falls to the node id, the same on every TP rank.
        if self.last_access_time != other.last_access_time:
            return self.last_access_time < other.last_access_time
        return self.id < other.id

    def get_last_hash_value(self) -> Optional[str]:
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    def get_prefix_hash_values(self, node: UnifiedTreeNode) -> list[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value


class UnifiedLRUList:
    def __init__(
        self,
        component_type: ComponentType,
        tree_components: tuple[ComponentType, ...],
        use_host_ptr: bool = False,
        is_referenced: Optional[Callable[[UnifiedTreeNode], bool]] = None,
    ):
        self.component_type = component_type
        # Pointer slot: host LRU uses offset slots so device/host pointers
        # never collide on the same node.
        self._pt: int = component_type + (_NUM_COMPONENT_TYPES if use_host_ptr else 0)
        self.head = UnifiedTreeNode(tree_components)
        self.tail = UnifiedTreeNode(tree_components)
        self.head.lru_next[self._pt] = self.tail
        self.tail.lru_prev[self._pt] = self.head
        self.cache: dict[int, UnifiedTreeNode] = {}
        # Session partition: [head .. mid) holds session-referenced nodes and
        # (mid .. tail] unreferenced ones, so evictions directly walks (tail -> head).
        self._is_referenced = is_referenced
        self.mid: Optional[UnifiedTreeNode] = None
        self.cursor: Optional[UnifiedTreeNode] = None
        if is_referenced is not None:
            self.mid = UnifiedTreeNode(tree_components)
            self.cursor = UnifiedTreeNode(tree_components)
            for ct in tree_components:
                for node in (self.mid, self.cursor):
                    node.component_data[ct].lock_ref = 1
                    node.component_data[ct].host_lock_ref = 1
            self._add_node_after(self.head, self.mid)

    def _add_node_after(self, prev_node: UnifiedTreeNode, new_node: UnifiedTreeNode):
        pt = self._pt
        new_node.lru_prev[pt] = prev_node
        new_node.lru_next[pt] = prev_node.lru_next[pt]
        prev_node.lru_next[pt].lru_prev[pt] = new_node
        prev_node.lru_next[pt] = new_node

    def _add_node(self, node: UnifiedTreeNode):
        if self._is_referenced is None or self._is_referenced(node):
            self._add_node_after(self.head, node)
        else:
            self._add_node_after(self.mid, node)

    def _remove_node(self, node: UnifiedTreeNode):
        pt = self._pt
        node.lru_prev[pt].lru_next[pt] = node.lru_next[pt]
        node.lru_next[pt].lru_prev[pt] = node.lru_prev[pt]
        # Clear self pointers to break reference cycles among evicted nodes.
        node.lru_prev[pt] = None
        node.lru_next[pt] = None

    def insert_mru(self, node: UnifiedTreeNode):
        assert node.id not in self.cache
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: UnifiedTreeNode):
        assert node.id in self.cache
        del self.cache[node.id]
        self._remove_node(node)

    def reset_node_mru(self, node: UnifiedTreeNode):
        assert node.id in self.cache
        self._remove_node(node)
        self._add_node(node)

    def cursor_begin(self):
        if self.cursor.lru_prev[self._pt] is not None:
            self._remove_node(self.cursor)
        self._add_node_after(self.tail.lru_prev[self._pt], self.cursor)

    def cursor_next(self, *, host_lock: bool = False):
        pt = self._pt
        ct = self.component_type
        x = self.cursor.lru_prev[pt]
        while x is not self.head:
            cd = x.component_data[ct]
            if (cd.host_lock_ref if host_lock else cd.lock_ref) == 0:
                break
            x = x.lru_prev[pt]
        if x is self.head:
            return None
        self._remove_node(self.cursor)
        self._add_node_after(x.lru_prev[pt], self.cursor)
        return x

    def cursor_end(self):
        """Unlink the walk-cursor sentinel after a walk."""
        self._remove_node(self.cursor)

    def reset_node_and_parents_mru(
        self,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
        should_include,
    ):
        is_referenced = self._is_referenced
        prev_ref = self.head
        prev_unref = self.head if self.mid is None else self.mid
        while node != root_node:
            if should_include(node):
                assert node.id in self.cache
                part = is_referenced is None or is_referenced(node)
                self._remove_node(node)
                if part:
                    self._add_node_after(prev_ref, node)
                    prev_ref = node
                else:
                    self._add_node_after(prev_unref, node)
                    prev_unref = node
            node = node.parent

    def reset_node_and_window_ancestors_mru(
        self,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
        window_size: int,
        should_include,
    ):
        is_referenced = self._is_referenced
        prev_ref = self.head
        prev_unref = self.head if self.mid is None else self.mid
        accumulated = 0
        while node != root_node and accumulated < window_size:
            if should_include(node):
                assert node.id in self.cache
                part = is_referenced is None or is_referenced(node)
                self._remove_node(node)
                if part:
                    self._add_node_after(prev_ref, node)
                    prev_ref = node
                else:
                    self._add_node_after(prev_unref, node)
                    prev_unref = node
            accumulated += len(node.key)
            node = node.parent

    def in_list(self, node: Optional[UnifiedTreeNode]):
        return node is not None and node.id in self.cache

    def get_prev_no_lock(self, node: UnifiedTreeNode, check_id: bool = True):
        if check_id:
            assert node.id in self.cache
        pt = self._pt
        ct = self.component_type
        x = node.lru_prev[pt]
        while x.component_data[ct].lock_ref > 0:
            x = x.lru_prev[pt]
        if x == self.head:
            return None
        return x

    def get_prev_leaf_no_lock(self, node: UnifiedTreeNode, check_id: bool = True):
        if check_id:
            assert node.id in self.cache
        pt = self._pt
        ct = self.component_type
        x = node.lru_prev[pt]
        while x.component_data[ct].lock_ref > 0 or len(x.children) > 0:
            x = x.lru_prev[pt]
        if x == self.head:
            return None
        return x

    def get_prev_no_host_lock(self, node: UnifiedTreeNode, check_id: bool = True):
        """Host-LRU walker: skip nodes whose component host_lock_ref > 0."""
        if check_id:
            assert node.id in self.cache
        pt = self._pt
        ct = self.component_type
        x = node.lru_prev[pt]
        while x.component_data[ct].host_lock_ref > 0:
            x = x.lru_prev[pt]
        if x == self.head:
            return None
        return x

    def get_lru_no_lock(self):
        return self.get_prev_no_lock(self.tail, check_id=False)

    def get_leaf_lru_no_lock(self):
        return self.get_prev_leaf_no_lock(self.tail, check_id=False)

    def get_lru_no_host_lock(self):
        return self.get_prev_no_host_lock(self.tail, check_id=False)


# WALK (one node per step) -> COMMIT (leaf + commit hooks) -> TAIL (refresh + backup).
class _InsertPhase(Enum):
    WALK = auto()
    COMMIT = auto()
    TAIL = auto()


class _InsertWalkState(msgspec.Struct):
    """In-flight resumable-insert state persisted across step barriers."""

    phase: _InsertPhase
    node: UnifiedTreeNode
    key: RadixKey
    value: torch.Tensor
    params: InsertParams
    active_components: tuple[TreeComponent, ...]
    priority: int
    total_prefix_length: int = 0
    is_new_leaf: bool = False
    # Walk stopped in front of a load-back-pending node; commit without a tail.
    stop_no_tail: bool = False
    # The commit built a chain (InsertParams.chain_cuts): one host eviction
    # for the tail's whole BackupKV (BackupKV.pre_evict).
    chained: bool = False
    target_node: Optional[UnifiedTreeNode] = None
    result: Optional[InsertResult] = None
    # Emitted actions awaiting the next barrier flush (or the final step).
    pending_actions: list[CacheAction | ComponentAction] = []


class UnifiedTreeCore(UnifiedTreeCoreInterface):
    """The radix tree mechanism: owns the tree structure, per-node values, the
    per-component LRUs, the size/leaf bookkeeping, and the component drivers,
    plus ``reset()``.

    """

    supports_partial_component_cache = True
    # M4 load quarantine clock (mem_cache/load_quarantine.py); None = off.
    _lq_clock = None
    # Session KV retention (--enable-kv-session-retention); None = off.
    kv_session_retention = None

    def __init__(
        self,
        params: CacheInitParams,
        components: dict[ComponentType, TreeComponent],
    ):
        self.page_size = params.page_size
        self.is_eagle = params.is_eagle and ComponentType.MAMBA not in components
        self.enable_hicache = False
        self.enable_storage = False
        self.write_through_threshold = 256
        self.is_write_back = False
        self.has_swa_host_pool = False
        # kv-p2p owner-side servicer (PeerKVServicer); attached by
        # UnifiedRadixCache.attach_peer_kv_servicer, None when disabled.
        self.peer_kv_servicer = None
        # SGLANG_PEER_KV_PUSH_FAST: push a node's pages as one index range
        # (servicer push_*_range) instead of one tensor slice per page. Same
        # shadow index, same call points.
        self.peer_push_fast = bool(envs.SGLANG_PEER_KV_PUSH_FAST.get())
        self.enable_session_radix_cache = params.enable_session_radix_cache
        self.kv_session_retention = None
        self.eviction_policy = params.eviction_policy.lower()
        self.eviction_strategy = get_eviction_strategy(params.eviction_policy.lower())

        # ``device`` is derived from the construction-time allocator; the
        # allocator/pool themselves are owned by the cache, not the tree.
        if params.token_to_kv_pool_allocator:
            self.device = params.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        # The cache builds and owns the component drivers; the tree references
        # them to drive their tree-level hooks, attaching itself as tree_core.
        assert components
        self.component_types = tuple(components.keys())
        self.components_by_type: dict[ComponentType, TreeComponent] = components
        swa = components.get(ComponentType.SWA)
        self.supports_partial_component_cache = not getattr(
            swa, "uniform_lifecycle", False
        )
        # Partial inserts stay off under the uniform lifecycle (a FULL-only
        # node would have no SWA on any tier). A FULL-only match is safe under
        # window residency, whose trees hold SWA for every node on some tier.
        self.supports_partial_component_insert = self.supports_partial_component_cache
        self.supports_partial_component_match = (
            self.supports_partial_component_cache
            or getattr(swa, "device_window", False)
        )
        for component in components.values():
            component.tree_core = self
        self.components: tuple[TreeComponent, ...] = tuple(
            self.components_by_type.values()
        )

        self.kv_events = KVCacheEventRecorder(
            enabled=params.enable_kv_cache_events, page_size=self.page_size
        )

        # --peer-kv-demote-source (mem_cache/peer_kv_demote.py). Off: no flag
        # is ever set and the host eviction keys are unchanged. The resolver
        # (the request-broadcast source rank) also keeps the last-page-key
        # index of published nodes to resolve complete RELEASEs.
        self.peer_demote_enabled = False
        # SGLANG_PEER_KV_DEMOTE_GPU_ORDER, every rank.
        self.peer_demote_gpu_order = False
        # SGLANG_PEER_KV_DEMOTE_SCOPE, every rank: "off",
        # "shadow" or "on" (the tick rules below act only with "on").
        self.peer_demote_scope = "off"
        self.peer_demote_resolver = False
        self.peer_demote_stats = PeerDemoteStats()
        # Hardening flags (peer_kv_demote.py docstring); off: as before.
        self.peer_demote_lock_aware = False
        self.peer_demote_counters = False
        self.peer_demote_xstats = PeerDemoteExtraStats()
        # COUNTERS, resolver rank: the tick -> time ring.
        self.peer_demote_clock: Optional[TickClock] = None
        # SGLANG_PEER_KV_DEMOTE_COUNTERS, resolver rank: called with one dict
        # per flagged eviction (set by the cache glue; None: no records).
        self.peer_demote_trace_sink: Optional[Callable[[dict], None]] = None
        # SGLANG_PEER_KV_DEMOTE_RECORDS, resolver rank: one
        # dict per host eviction (R2); set by the cache glue, None: off.
        self.peer_demote_record_sink: Optional[Callable[[dict], None]] = None
        # SGLANG_PEER_KV_TOUCH_DEFERRED_PULLS (mem_cache/peer_kv_touch.py).
        # Off: nothing reads or changes the tree. The resolver keeps the same
        # last-page-key index to resolve TOUCHes.
        self.peer_touch_enabled = False
        self.peer_touch_resolver = False
        self.peer_touch_stats = PeerTouchStats()

        self.reset()

    def _resolve_active_components(
        self, active_components: Optional[Sequence[ComponentType]]
    ) -> tuple[TreeComponent, ...]:
        """Resolve a per-operation component mask in the tree's stable order.

        FULL owns the radix key/value backbone, so an operation cannot select
        only an auxiliary component. Structural operations (notably node
        splits) still run across ``self.components`` even when this returns a
        strict subset.
        """
        if active_components is None:
            return self.components

        requested = tuple(active_components)
        requested_set = set(requested)
        if len(requested_set) != len(requested):
            raise ValueError(
                f"active_components contains duplicates: {active_components!r}"
            )
        unknown = requested_set.difference(self.component_types)
        if unknown:
            names = ", ".join(sorted(component.name for component in unknown))
            raise ValueError(f"active_components not registered on this tree: {names}")
        if BASE_COMPONENT_TYPE not in requested_set:
            raise ValueError("active_components must include ComponentType.FULL")
        return tuple(
            component
            for component in self.components
            if component.component_type in requested_set
        )

    # ==== Tree API ====

    def _session_lru_predicate(self, ct: ComponentType):
        if not self.enable_session_radix_cache or ct is ComponentType.FULL:
            return None
        return lambda node: node.component_data[ct].session_ref > 0

    def reset(self) -> None:
        """Rebuild the root, LRUs, sizes, evictable-leaf sets, and the empty
        match result."""
        # Maintains the NodeId -> active tree node mapping.
        self._node_arena: dict[NodeId, UnifiedTreeNode] = {}
        # --peer-kv-demote-source: flagged nodes (every rank), the
        # last-page-key -> published node index and the last keys of recently
        # flagged nodes for the re-pull count (resolver rank only).
        self._peer_demoted: dict[NodeId, UnifiedTreeNode] = {}
        self._peer_demote_index: dict[str, UnifiedTreeNode] = {}
        self._peer_demote_recent: OrderedDict[str, int] = OrderedDict()
        # A digest of the flagged node ids (every rank; read by the TP
        # structural digest under SGLANG_PEER_KV_DEMOTE_COUNTERS and by the
        # step-sync fold under SGLANG_HICACHE_EVICT_ORDER_CHECK), and with
        # COUNTERS per flagged node its flag time and the pulling request id
        # (resolver rank only; rank-local, read only for counters and trace
        # records).
        self._peer_demoted_id_digest = 0
        # meta: (flag time, rid, trigger, tick age at the mark, mark tokens).
        self._peer_demote_meta: dict[NodeId, tuple] = {}
        self._peer_demote_selected_rid: dict[NodeId, tuple[str, str, int]] = {}
        # Reader, resolver rank, COUNTERS: committed landings -> their tick.
        self._peer_landed: OrderedDict[NodeId, float] = OrderedDict()
        # RECORDS: node ids whose host copy came from a peer pull (R2 from_pull).
        self._peer_from_pull: OrderedDict[NodeId, bool] = OrderedDict()

        # The single in-flight resumable insert, if suspended at a barrier.
        self._ongoing_insert_walk_state: Optional[_InsertWalkState] = None

        self.root_node = self._new_node()
        self.root_node.priority = -sys.maxsize
        self.root_node.key = RadixKey(array("q"), None)
        self.root_node.component_data[BASE_COMPONENT_TYPE].value = []
        self.root_node.hash_value = []
        for ct in self.component_types:
            self.root_node.component_data[ct].lock_ref = 1

        self.component_evictable_size_ = {ct: 0 for ct in self.component_types}
        self.component_protected_size_ = {ct: 0 for ct in self.component_types}
        # Window SWA residency: unlocked device SWA without a host copy, which
        # eviction must keep (maintained by SWAComponent; 0 otherwise).
        self.swa_blocked_size_ = 0
        # Window SWA residency: unbacked nodes to back up later, in insertion
        # order, mapped to the processed-batch count after which their bytes
        # are final (UnifiedRadixCache.retry_swa_rebackup).
        self.swa_rebackup_queue: dict[NodeId, int] = {}

        self.lru_lists = {
            ct: UnifiedLRUList(
                ct, self.component_types, is_referenced=self._session_lru_predicate(ct)
            )
            for ct in self.component_types
        }

        self.evictable_device_leaves: set[UnifiedTreeNode] = set()
        self.evictable_host_leaves: set[UnifiedTreeNode] = set()
        # Deleted nodes handed back to the leaf sets (stale references).
        self.stale_leaf_set_updates = 0
        # NodeId -> number of in-flight D->H backups below it; see begin_backup_guard.
        self._backup_guarded_ancestors: dict[NodeId, int] = {}
        # The attached peer_kv_servicer must survive reset: flush relies on
        # UnifiedRadixCache.reset() -> push_clear() to empty the shadow index,
        # and the tree must keep feeding it afterwards.
        self.host_lru_lists = {
            ct: UnifiedLRUList(
                ct,
                self.component_types,
                use_host_ptr=True,
                is_referenced=self._session_lru_predicate(ct),
            )
            for ct in self.component_types
        }
        # Full KV on both tiers -> redundant host copy, reclaimed first by
        # write_back; insertion-ordered dict keeps victims TP-deterministic.
        self.full_host_duplicates: dict[NodeId, UnifiedTreeNode] = {}
        # Rolling digest of reclaim victim ids, cross-checked across TP ranks.
        self.write_back_duplicate_reclaim_digest: int = 0
        # Rolling digest of every host-layer release (any path), likewise
        # cross-checked across TP ranks (detects TP rank divergence).
        self.host_release_digest: int = 0
        # Rolling digest of every device-layer free (node id, component,
        # demote class), folded into the step sync by
        # SGLANG_HICACHE_EVICT_ORDER_CHECK.
        self.device_evict_digest: int = 0

        self._empty_match_result = MatchResult(
            device_indices=torch.empty(
                (0,),
                dtype=torch.int64,
                device=self.device,
            ),
            last_device_node=self.root_node.id,
            last_host_node=self.root_node.id,
            best_match_node=self.root_node.id,
            cache_actions=[],
        )

    def node_by_id(self, node_id: NodeId) -> UnifiedTreeNode:
        """Resolve a NodeId back to its tree node.

        TODO(Jialin): Make TreeCore-internal after the Unified Radix Cache split.
        """
        return self._node_arena[node_id]

    def is_backuped(self, node_id: NodeId) -> bool:
        """Whether the node's KV is already backed up to host."""
        return self._node_arena[node_id].backuped

    def is_root(self, node_id: NodeId) -> bool:
        """Whether the node is the tree root."""
        return self.node_by_id(node_id) is self.root_node

    def get_last_hash_value(self, node_id: NodeId) -> Optional[str]:
        """The node's last page hash, or None when it was never hashed."""
        return self.node_by_id(node_id).get_last_hash_value()

    def get_prefix_hash_values(self, node_id: NodeId) -> list[str]:
        """The hash chain of the node's ancestors, in root-to-parent order."""
        node = self.node_by_id(node_id)
        return node.get_prefix_hash_values(node.parent)

    def get_hash_values(self, node_id: NodeId) -> list[str]:
        """The hash values owned by this node, excluding its ancestors."""
        return self.node_by_id(node_id).hash_value or []

    def backfill_missing_hash_values(self) -> int:
        """Hash every node that was built while storage was disabled.

        Page hashes chain from the parent's last hash, so a node whose parent has
        none restarts the chain mid-sequence: its keys then encode only a suffix
        of the prefix they claim to represent, which would alias unrelated
        requests that happen to start with those tokens. Walks parent-before-child
        so each node hashes against an already-filled parent. Returns the number
        of nodes filled.
        """
        filled = 0
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            # The root anchors every chain and is seeded with an empty hash list.
            if node is not self.root_node and node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)
                filled += 1
            stack.extend(node.children.values())
        return filled

    def root_node_handle(self, extra_key: Optional[str] = None) -> NodeId:
        """The NodeId anchoring matches; the single root serves every namespace."""
        return self.root_node.id

    def _new_node(self, priority: int = 0) -> UnifiedTreeNode:
        """Create and register a tree node in the arena."""
        node = UnifiedTreeNode(self.component_types, priority=priority)
        if self._lq_clock is not None:
            # A new node's pages may be in use by the pass's forward until
            # proven otherwise (a split copies the child's epoch instead).
            node.component_data[BASE_COMPONENT_TYPE].unlock_epoch = self._lq_clock.tag()
        self._register_node(node)
        return node

    def _register_node(self, node: UnifiedTreeNode) -> None:
        """Register a tree node in the arena."""
        self._node_arena[node.id] = node

    def _unregister_node(self, node: UnifiedTreeNode) -> None:
        """Drop a tree node from the arena; deletion is final for every holder
        of the object (Full.value is tombstoned here because the delete paths
        skip _cascade_evict, otherwise the node still reads as a D-leaf)."""
        self._node_arena.pop(node.id, None)
        if node.peer_demoted:
            # Host releases clear the flag first (_peer_push_removed); this
            # only covers a delete path that skipped it.
            self._peer_demote_clear(node, evicted=True)
        node.deleted = True
        node.component_data[BASE_COMPONENT_TYPE].value = None
        self.evictable_device_leaves.discard(node)
        self.evictable_host_leaves.discard(node)

    def inc_lock_ref(
        self, node_id: NodeId, skip_lock_components: Sequence[ComponentType] = ()
    ) -> IncLockRefResult:
        node = self.node_by_id(node_id)
        skipped_set = set(skip_lock_components)
        unknown = skipped_set.difference(self.component_types)
        if unknown:
            names = ", ".join(sorted(component.name for component in unknown))
            raise ValueError(
                f"skip_lock_components not registered on this tree: {names}"
            )
        skipped = tuple(
            component_type
            for component_type in self.component_types
            if component_type in skipped_set
        )
        result = IncLockRefResult(skip_lock_components=skipped)
        for component in self.components:
            if component.component_type in skipped_set:
                # Keep the anchor in the legacy per-node skip map as well.
                # Specialized early-release helpers (notably
                # dec_swa_lock_only) receive this map and may release a
                # lower-priority co-located component before the final
                # path-wide dec_lock_ref replays skip_lock_components.
                if node is not self.root_node:
                    result.skip_lock_node_ids.setdefault(
                        component.component_type, set()
                    ).add(node.id)
                continue
            result = component.acquire_component_lock(node=node, result=result)
        self._update_evictable_leaf_sets(node)
        return result

    def dec_lock_ref(
        self,
        node_id: NodeId,
        params: Optional[DecLockRefParams] = None,
        skip_swa: bool = False,
    ) -> DecLockRefResult:
        node = self.node_by_id(node_id)
        skipped_components = set(params.skip_lock_components if params else ())
        if skip_swa:
            skipped_components.add(ComponentType.SWA)
        for component in self.components:
            if component.component_type in skipped_components:
                continue
            component.release_component_lock(node=node, params=params)
        self._update_evictable_leaf_sets(node)
        # TODO: delta is not aggregated from components; no caller uses it yet.
        return DecLockRefResult()

    def dec_swa_lock_only(
        self,
        node_id: NodeId,
        swa_uuid_for_lock: Optional[int],
        skip_lock_node_ids: Optional[dict] = None,
    ) -> DecSwaLockOnlyResult:
        """Early-release the SWA portion of a request's tree lock, plus any
        strictly-lower-priority locks (e.g. Mamba) co-located on the node."""
        result = DecSwaLockOnlyResult()
        node = self.node_by_id(node_id)
        swa_component = self.components_by_type.get(ComponentType.SWA)
        if swa_component is None:
            return result
        if getattr(swa_component, "uniform_lifecycle", False) and not getattr(
            swa_component, "device_window", False
        ):
            raise RuntimeError("uniform cache lifecycle forbids independent SWA unlock")
        swa_component.release_window_lock(
            node, swa_uuid_for_lock, result.device_frees, result.host_frees
        )

        # Drop strictly-lower-priority locks (e.g. Mamba) co-located on the node,
        # honoring skip ids so we don't drop a lock a partial inc never took
        # (matters for FULL+SWA+MAMBA models, e.g. Inkling).
        swa_priority = swa_component.eviction_priority(is_leaf=False)
        dec_params = DecLockRefParams(
            swa_uuid_for_lock=swa_uuid_for_lock,
            skip_lock_node_ids=skip_lock_node_ids or {},
        )
        for comp in self.components:
            if comp.eviction_priority(is_leaf=False) < swa_priority:
                comp.release_component_lock(node, dec_params)
        return result

    def inc_host_lock_ref(self, node_id: NodeId) -> IncLockRefResult:
        node = self.node_by_id(node_id)
        result = IncLockRefResult()
        for component in self.components:
            result = component.acquire_component_lock(
                node=node, result=result, lock_host=True
            )
        self._update_evictable_leaf_sets(node)
        return result

    def dec_host_lock_ref(
        self, node_id: NodeId, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        node = self.node_by_id(node_id)
        for component in self.components:
            component.release_component_lock(node=node, params=params, lock_host=True)
        self._update_evictable_leaf_sets(node)
        return DecLockRefResult()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        if params.active_components is not None and self.enable_session_radix_cache:
            raise ValueError(
                "partial-component match_prefix is incompatible with "
                "--enable-session-radix-cache"
            )
        active_components = self._resolve_active_components(params.active_components)
        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if len(key) == 0:
            return self._empty_match_result
        key = key.page_aligned(self.page_size)
        if len(key) == 0:
            return self._empty_match_result

        (
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
            full_kv_hit_length,
            action,
        ) = self._match_prefix_helper(key, active_components)
        return self._match_post_processor(
            params,
            active_components,
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
            full_kv_hit_length,
            action,
        )

    def _match_prefix_helper(
        self, key: RadixKey, active_components: tuple[TreeComponent, ...]
    ) -> tuple[
        list[torch.Tensor],
        UnifiedTreeNode,
        UnifiedTreeNode,
        int,
        int,
        Optional[CacheAction | ComponentAction],
    ]:
        # Non-HiCache mode has only device-resident matches, so the scheduler
        # device anchor follows the best match. In HiCache mode, host-backed
        # nodes can also match, so we separately track the best device-resident
        # match for scheduler prefix indices and locking.
        node = self.root_node
        child_key = key.child_key(self.page_size)
        value: list[torch.Tensor] = []
        best_match_node = node
        best_match_device_node = node
        best_match_device_value_len = 0
        full_kv_hit_length = 0
        action: Optional[CacheAction | ComponentAction] = None
        separate_device_match = self.enable_hicache
        if separate_device_match:
            validators = tuple(
                comp.create_match_validator() for comp in active_components
            )
            device_validators = tuple(
                comp.create_match_validator(match_device_only=True)
                for comp in active_components
            )
        else:
            validators = tuple(
                comp.create_match_validator(match_device_only=True)
                for comp in active_components
            )

        def _all_valid(validators, node):
            return all([v(node) for v in validators])

        def _update_best_if_valid(node):
            nonlocal best_match_node
            nonlocal best_match_device_value_len, best_match_device_node
            matched = _all_valid(validators, node)
            if matched:
                best_match_node = node

            if not separate_device_match:
                if matched:
                    best_match_device_value_len = len(value)
                    best_match_device_node = node
                return
            if _all_valid(device_validators, node):
                best_match_device_value_len = len(value)
                best_match_device_node = node

        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]

            # HiCache: dead node (evicted + not backuped) — stop traversal
            if child.evicted and not child.backuped:
                break

            prefix_len = child.key.match(key, page_size=self.page_size)
            full_kv_hit_length += prefix_len
            if prefix_len < len(child.key):
                node, action = self._split_node(child.key, child, prefix_len)
                if not node.evicted:
                    value.append(node.component_data[BASE_COMPONENT_TYPE].value)
                _update_best_if_valid(node)
                break

            if not child.evicted:
                value.append(child.component_data[BASE_COMPONENT_TYPE].value)
            node = child
            _update_best_if_valid(node)
            key = key[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)

        return (
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
            full_kv_hit_length,
            action,
        )

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        active_components: tuple[TreeComponent, ...],
        value: list[torch.Tensor],
        best_match_node: UnifiedTreeNode,
        best_match_device_node: UnifiedTreeNode,
        best_match_device_value_len: int,
        full_kv_hit_length: int,
        action: Optional[CacheAction | ComponentAction],
    ) -> MatchResult:
        node_update = best_match_node
        for comp in active_components:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue  # Full uses last_access_time, not LRU
            comp.refresh_lru(LRURefreshPhase.MATCH_END, node_update, self.root_node)

        cur_time = get_and_increase_time_counter()
        while node_update:
            node_update.last_access_time = cur_time
            node_update.host_last_access_time = cur_time
            if node_update.peer_demoted:
                self._peer_demote_clear(node_update, evicted=False)
            cur_time -= 0.00001
            node_update = node_update.parent

        # last_host_node will be used as the starting node for the subsequent
        # `prefetch_from_storage` flow. We directly use best_match_node here,
        # because best_match_node represents the node where all components
        # have reached consensus on both device & host availability.
        last_host_node = (
            best_match_node if self.enable_hicache else best_match_device_node
        )

        if best_match_device_value_len > 0:
            device_indices = cat_tagged(
                value[:best_match_device_value_len],
                lambda: self.page_size,
                "match_prefix",
            )
        else:
            device_indices = self._empty_match_result.device_indices
        result = MatchResult(
            device_indices=device_indices,
            last_device_node=best_match_device_node,
            last_host_node=last_host_node,
            best_match_node=best_match_node,
            host_hit_length=0,
            full_kv_hit_length=full_kv_hit_length,
        )

        for component in active_components:
            result = component.finalize_match_result_in_tree_core(
                result=result,
                params=params,
                value_chunks=value,
                best_value_len=best_match_device_value_len,
            )
        # Expose only NodeIds outside TreeCore.
        return result._replace(
            last_device_node=result.last_device_node.id,
            last_host_node=result.last_host_node.id,
            best_match_node=result.best_match_node.id,
            cache_actions=[action] if action is not None else [],
        )

    @property
    def empty_match_result(self) -> MatchResult:
        """A shared empty MatchResult (empty device indices + boundary NodeIds)."""
        return self._empty_match_result

    def match_full_prefix_len(self, key: RadixKey) -> int:
        """Length of the longest page-aligned prefix of ``key`` the tree holds
        with FULL on some tier, walked like ``match_prefix`` but read-only: no
        node split, no LRU or access-time update. Under window SWA residency
        every such node also has SWA on some tier (I1), so a finish insert of
        a resumed request is complete up to here."""
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if len(key) == 0:
            return 0
        key = key.page_aligned(self.page_size)
        node = self.root_node
        matched = 0
        while len(key) > 0:
            child = node.children.get(key.child_key(self.page_size))
            # A dead node (evicted and not backed up) ends the prefix, as in
            # _match_prefix_helper.
            if child is None or (child.evicted and not child.backuped):
                break
            prefix_len = child.key.match(key, page_size=self.page_size)
            matched += prefix_len
            if prefix_len < len(child.key):
                break
            node = child
            key = key[prefix_len:]
        return matched

    def is_full_device_evicted(self, node_id: NodeId) -> bool:
        """Whether the node's FULL device value has been evicted."""
        return self.node_by_id(node_id).evicted

    def collect_full_device_indices(
        self, from_node_id: NodeId, until_node_id: NodeId
    ) -> torch.Tensor:
        """Concatenate FULL device values from ``from_node`` up to (exclusive)
        ``until_node``, in root order; empty tensor if the path is empty."""
        until_node = self.node_by_id(until_node_id)
        prefix_chunks: list[torch.Tensor] = []
        node = self.node_by_id(from_node_id)
        while node is not until_node:
            value = node.component_data[BASE_COMPONENT_TYPE].value
            assert value is not None
            prefix_chunks.append(value)
            node = node.parent
        if not prefix_chunks:
            return self._empty_match_result.device_indices
        prefix_chunks.reverse()
        # page_size is read only with Q4 on (stand-in trees in tests
        # model no page size).
        return cat_tagged(
            prefix_chunks, lambda: self.page_size, "collect_full_device_indices"
        )

    def _touch_node(
        self,
        node: UnifiedTreeNode,
        components: Optional[Sequence[TreeComponent]] = None,
    ):
        node.last_access_time = get_and_increase_time_counter()
        node.host_last_access_time = node.last_access_time
        if node.peer_demoted:
            self._peer_demote_clear(node, evicted=False)
        if node != self.root_node:
            selected_components = self.components if components is None else components
            for comp in selected_components:
                if comp.component_type == BASE_COMPONENT_TYPE:
                    continue
                comp.refresh_lru(LRURefreshPhase.WALKDOWN, node, self.root_node)

    def touch_host_nodes(self, node_ids) -> None:
        """Mark committed/used CPU nodes MRU; input IDs, no GPU recency change."""
        for node_id in node_ids:
            node = self._node_arena.get(node_id)
            if node is None or node.deleted or node is self.root_node:
                continue
            node.host_last_access_time = get_and_increase_time_counter()
            for ct, lru in self.host_lru_lists.items():
                if ct != BASE_COMPONENT_TYPE and lru.in_list(node):
                    lru.reset_node_mru(node)

    def locate_retention_prefix(self, key, *, start_node=None, start_offset=0):
        """Return exact structural prefix and split actions, without touching LRU.

        The full native key is used: a scheduler hit cap or missing device copy
        cannot be interpreted as context compaction. Splits reuse the same
        component/IO relocation machinery as normal radix matching.
        """
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        end = len(key) // self.page_size * self.page_size
        offset = start_offset
        node, actions = start_node or self.root_node, []
        while offset < end:
            # Bound each temporary slice to one edge (lookup: one page), not
            # the whole remaining suffix. Fragmented long contexts otherwise
            # copy O(tokens * radix_depth) input bytes during admission.
            page_key = key[offset : offset + self.page_size]
            child = node.children.get(page_key.child_key(self.page_size))
            if child is None:
                break
            edge_key = key[offset : min(end, offset + len(child.key))]
            size = child.key.match(edge_key, page_size=self.page_size)
            if size == 0:
                break
            if size < len(child.key):
                node, action = self._split_node(child.key, child, size, touch=False)
                if action is not None:
                    actions.append(action)
                break
            node = child
            offset += size
        return node, actions

    def host_retention_priority(self, node):
        """Return (class, CPU tick, id): class 0 iff no session counts the node.

        A never-tagged node counts as 0, so it is class 0 like a released one.
        host_session_known/unknown no longer set the class; they remain for
        KV_SESSION_STATS. Positive counts are never ranked by magnitude.
        """
        rank = int(node.host_session_count > 0)
        return rank, node.host_last_access_time, node.id

    def _retention_eviction_key(self, node):
        """Heap key of the retention host eviction.

        (class, active, kind, CPU tick, id), or (class, kind, CPU tick, id)
        with SGLANG_KV_SESSION_ACTIVE_SOFT_PIN=0.

        active is 1 when a registration with a queued or running request on
        this worker covers the node (host_session_active > 0). Within a class
        every idle candidate goes first: a soft pin for the prefix of a waiting
        request, never a hard pin and no recency change. A class-0 node has
        count 0, so its active is always 0.

        kind is 1 for a GPU-backed duplicate (a candidate outside
        evictable_host_leaves) in both classes. Each class
        then frees every host-only leaf before any duplicate, as the base
        write-through eviction does: a released duplicate makes a later GPU
        eviction delete that prefix outright. With nothing tagged (every D
        node, never-tagged P paths) all nodes are class 0 with active 0, so
        with --peer-kv-demote-source off the victim sequence equals the base
        path's (retention off) and a retention D evicts in the same order as a
        control D. The key has no peer_demoted level: with demote on, the base
        path frees demoted copies first and this path does not (whether it
        should is still open).

        SGLANG_KV_SESSION_RETENTION_ORDER=predicted_return_fixed (L1,
        retention_order.py; fixed m table), or predicted_return once its
        online return-gap table is ready: a class-1 key puts -m(idle age)
        before the CPU tick, so within (class, active, kind) the node whose
        agent is predicted to return last goes first; the CPU tick, then the
        id, break ties. The idle age is read on the TP-lockstep clock, never
        a rank-local one. Class 0 and the order lru keep the key above
        unchanged.
        """
        priority = self.host_retention_priority(node)
        duplicate = node not in self.evictable_host_leaves
        retention = self.kv_session_retention
        if priority[0] == 1 and retention.predicted_active:
            m = retention.expected_remaining_idle_s(node.host_last_access_time)
            priority = (1, -m) + tuple(priority[1:])
        if retention.active_soft_pin:
            active = int(node.host_session_active > 0)
            return (priority[0], active, int(duplicate)) + tuple(priority[1:])
        return (priority[0], int(duplicate)) + tuple(priority[1:])

    def retention_host_scan(self) -> tuple[int, int, int]:
        """Return (known-zero host tokens, class-0 host tokens, active nodes).

        FULL host copies only. Known-zero = tagged and now count 0 (an earlier
        class-0 rule, kept for comparison); class 0 = count 0 (the current rule);
        active nodes = host nodes with host_session_active > 0. Scans every
        node: for the 60 s KV_SESSION_STATS line only, never on a request path.
        """
        known_zero = class0 = active = 0
        for node in self._node_arena.values():
            host_value = node.component_data[BASE_COMPONENT_TYPE].host_value
            if host_value is None or node is self.root_node:
                continue
            if node.host_session_count == 0:
                class0 += len(host_value)
                if node.host_session_known and not node.host_session_unknown:
                    known_zero += len(host_value)
            elif node.host_session_active > 0:
                active += 1
        return known_zero, class0, active

    def known_zero_host_tokens(self) -> int:
        """FULL host tokens of tagged nodes whose count is now 0 (stats only)."""
        return self.retention_host_scan()[0]

    def _inc_hit_count_and_check(
        self, node: UnifiedTreeNode, chunked: bool = False
    ) -> bool:
        """Increment hit count; check whether a write backup should be fired."""
        if node.evicted or chunked:
            return False
        if self.is_write_back:
            return False
        node.hit_count += 1
        return (
            self.enable_hicache
            and not node.backuped
            and node.hit_count >= self.write_through_threshold
        )

    def begin_insert(self, params: InsertParams) -> InsertStepResult:
        """Start the insert, running to its first barrier or completion."""
        # Insert walks are single-flight; a live walk means re-entrancy.
        assert self._ongoing_insert_walk_state is None, "concurrent insert walks"
        if params.active_components is not None and self.enable_session_radix_cache:
            raise ValueError(
                "partial-component insert is incompatible with "
                "--enable-session-radix-cache"
            )
        active_components = self._resolve_active_components(params.active_components)
        key = params.key
        value = params.value
        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            # Q4: a prefix of the tagged insert value (cpu_page_book.py).
            value = slice_tagged(params.value, 0, len(key), self.page_size)
        else:
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

        priority = params.priority
        if priority is None:
            priority = 0
        self._touch_node(self.root_node, active_components)
        self.root_node.priority = max(self.root_node.priority, priority)
        if len(key) == 0:
            return InsertStepResult(
                actions=[],
                result=InsertResult(
                    prefix_len=0,
                    mamba_exist=True,
                    last_device_node=self.root_node.id,
                ),
            )

        self._ongoing_insert_walk_state = _InsertWalkState(
            phase=_InsertPhase.WALK,
            node=self.root_node,
            key=key,
            value=value,
            params=params,
            active_components=active_components,
            priority=priority,
        )
        return self._advance_insert()

    def resume_insert(self) -> InsertStepResult:
        """Continue the suspended insert after its step actions were executed."""
        assert self._ongoing_insert_walk_state is not None, "no in-flight insert"
        return self._advance_insert()

    def has_ongoing_insert(self) -> bool:
        """Whether an insert walk is suspended at a barrier."""
        return self._ongoing_insert_walk_state is not None

    def end_insert(self) -> list[CacheAction | ComponentAction]:
        """Finish the insert (idempotent); returns still-pending actions to drain."""
        state = self._ongoing_insert_walk_state
        self._ongoing_insert_walk_state = None
        return state.pending_actions if state is not None else []

    def _advance_insert(self) -> InsertStepResult:
        """Run the in-flight insert to its next barrier or to completion."""
        state = self._ongoing_insert_walk_state
        while True:
            flushed_len = len(state.pending_actions)
            if state.phase is _InsertPhase.WALK:
                self._insert_walk_step(state)
            elif state.phase is _InsertPhase.COMMIT:
                self._insert_commit_step(state)
            elif state.phase is _InsertPhase.TAIL:
                self._insert_tail_step(state)
                self._ongoing_insert_walk_state = None
                return InsertStepResult(
                    actions=state.pending_actions, result=state.result
                )
            else:
                raise AssertionError(f"unsupported insert phase: {state.phase}")
            new_actions = state.pending_actions[flushed_len:]
            # Suspend only when a step emitted a non-deferrable action.
            if new_actions and not all(map(self._is_deferrable_action, new_actions)):
                flushed = state.pending_actions
                state.pending_actions = []
                return InsertStepResult(actions=flushed)

    @staticmethod
    def _is_deferrable_action(action: CacheAction | ComponentAction) -> bool:
        """Fire-and-forget actions safe to batch until the next barrier."""
        return isinstance(
            action,
            (FreeDeviceKV, FreeDeviceKVFullOnly, ReplaceWriteThroughOnNodeSplit),
        )

    def _insert_walk_step(self, state: _InsertWalkState) -> None:
        """Process one walked node, appending its barrier actions to the state."""
        key = state.key
        child_key = key.child_key(self.page_size) if len(key) else None
        if child_key not in state.node.children:
            state.phase = _InsertPhase.COMMIT
            return
        step_actions = state.pending_actions
        node = state.node.children[child_key]
        if (
            state.params.stop_at_load_back_pending
            and node.load_back_pending_id is not None
        ):
            # The child's FULL device slots are in-flight H2D load-back
            # targets (commit_load_back), unreadable until the ack, and the
            # device match validator rejects them. Stop in front of the child
            # (before touch/split): the un-walked tail stays request-owned and
            # is re-offered at the next chunk boundary.
            logger.debug(
                "insert walk stopped at load-back-pending node %s "
                "(prefix_len=%d, pending anchor=%s)",
                node.id,
                state.total_prefix_length,
                node.load_back_pending_id,
            )
            state.stop_no_tail = True
            state.phase = _InsertPhase.COMMIT
            return
        scope_on = self.peer_demote_scope == "on"
        if not scope_on:
            self._touch_node(node, state.active_components)
        prefix_len = node.key.match(key, page_size=self.page_size)
        if prefix_len < len(node.key):
            node, action = self._split_node(node.key, node, prefix_len)
            if action is not None:
                step_actions.append(action)
        if scope_on:
            # SCOPE: touch only the matched part, after the split.
            self._touch_node(node, state.active_components)
        node.priority = max(node.priority, state.priority)

        if node.evicted:
            self._unevict_node_on_insert(
                node, slice_tagged(state.value, 0, prefix_len, self.page_size)
            )
            # FULL was restored from the request's fresh KV. Aux
            # components (e.g. SWA) may still hold tombstones and need
            # to rebuild their value from the same slice.
            for component in state.active_components:
                if component.component_type == BASE_COMPONENT_TYPE:
                    continue
                component.recover_after_unevict(
                    node=node,
                    prefix_len=prefix_len,
                    total_prefix_len=state.total_prefix_length,
                    params=state.params,
                    cache_actions=step_actions,
                )
        else:
            value_slice = slice_tagged(state.value, 0, prefix_len, self.page_size)
            consumed_from = prefix_len
            # Let each component claim ownership of overlapping KV slots
            for component in state.active_components:
                comp_consumed_from = component.update_component_on_insert_overlap(
                    node=node,
                    prefix_len=prefix_len,
                    total_prefix_len=state.total_prefix_length,
                    value_slice=value_slice,
                    params=state.params,
                    cache_actions=step_actions,
                )
                consumed_from = min(consumed_from, comp_consumed_from)

            dup_start = max(0, state.params.prev_prefix_len - state.total_prefix_length)
            if dup_start < consumed_from:
                step_actions.append(
                    FreeDeviceKV(
                        [
                            slice_tagged(
                                value_slice, dup_start, consumed_from, self.page_size
                            )
                        ]
                    )
                )

        if self._inc_hit_count_and_check(node, state.params.chunked):
            step_actions.append(self._build_backup_kv_action(node))
        state.node = node
        state.total_prefix_length += prefix_len
        state.key = key[prefix_len:]
        state.value = slice_tagged(state.value, prefix_len, None, self.page_size)

    def _insert_commit_step(self, state: _InsertWalkState) -> None:
        """Create the tail leaf and run the component commit hooks."""
        # Create new leaf for remaining suffix. A leaf survives on its Full
        # value alone; auxiliary components (SWA, Mamba) may legitimately hold
        # only a tombstone for this span (e.g. the whole leaf is outside the SWA
        # window). Materialize it anyway so the Full KV stays cacheable.
        if len(state.key) and not state.stop_no_tail:
            parent, key, value = state.node, state.key, state.value
            if state.params.chain_cuts:
                parent, key, value = self._add_chain_head(state, parent, key, value)
                state.chained = parent is not state.node
            state.target_node = self._add_new_node(
                parent, key, value, priority=state.priority
            )
            state.is_new_leaf = True
        else:
            state.target_node = state.node

        # Finalize: let each component attach its data to the target node.
        # e.g. Mamba attaches mamba_value to the leaf node
        # All hooks run before their emitted actions execute; an action failure
        # fail-stops the process, so partial-commit state is never observed.
        state.result = InsertResult(
            prefix_len=state.total_prefix_length,
            last_device_node=state.target_node.id,
        )
        if state.stop_no_tail:
            # Stopped short: the tail was not inserted, so skip the component
            # commits -- attaching chunk-end aux state (e.g. a Mamba
            # checkpoint) to the shorter boundary node would mis-position it.
            # mamba_exist=True routes cleanup_after_caching_req to free the
            # caller's unconsumed incoming mamba value.
            state.result.mamba_exist = True
            state.result.stopped_at_load_back_pending = True
            state.phase = _InsertPhase.TAIL
            return
        for component in state.active_components:
            component.commit_insert_component_data(
                node=state.target_node,
                is_new_leaf=state.is_new_leaf,
                params=state.params,
                result=state.result,
                cache_actions=state.pending_actions,
            )
        state.phase = _InsertPhase.TAIL

    def _add_chain_head(self, state: _InsertWalkState, parent, key, value):
        """InsertParams.chain_cuts (symmetric release insert): add one node per
        cut inside the new tail, the shape a standard P's chunk inserts leave.
        Returns the last node and the key/value left for the tail leaf.

        Each node gets the one hit the standard P's last-chunk walk gives it.
        The tail's BackupKV lists every unbacked ancestor first, so it covers
        the chain one node at a time, after one host eviction for all of them
        (BackupKV.pre_evict). Each node records its own KV store event;
        the block hashes and parent links equal those of one node. Cuts come
        from scheduler state that is the same on every TP rank."""
        if self.component_types != (BASE_COMPONENT_TYPE,):
            return parent, key, value  # aux components commit on the leaf only
        off = 0  # key offset of the next node; slices copy each token once
        for cut in state.params.chain_cuts:
            end = cut - state.total_prefix_length
            if end <= off or end % self.page_size:
                continue
            if end >= len(key):
                break
            parent = self._add_new_node(
                parent,
                key[off:end],
                slice_tagged(value, off, end, self.page_size),
                priority=state.priority,
            )
            # A fresh node gets the same answer as the tail leaf below, whose
            # BackupKV lists this node as an unbacked ancestor: drop it here.
            assert not parent.evicted and not parent.backuped
            self._inc_hit_count_and_check(parent, state.params.chunked)
            off = end
        if off:
            key, value = key[off:], slice_tagged(value, off, None, self.page_size)
        return parent, key, value

    def _needs_incremental_component_backup(
        self,
        node: UnifiedTreeNode,
        components: Optional[Sequence[TreeComponent]] = None,
    ) -> bool:
        selected_components = self.components if components is None else components
        if self.is_write_back:
            selected_components = tuple(
                component
                for component in selected_components
                if component.component_type == ComponentType.SWA
            )
        return any(
            component.needs_incremental_backup(node)
            for component in selected_components
            if component.component_type != BASE_COMPONENT_TYPE
        )

    def _should_backup_after_insert(self, state: _InsertWalkState) -> bool:
        """Check whether the insert target needs a Host backup."""
        if state.is_new_leaf:
            return self._inc_hit_count_and_check(
                state.target_node, state.params.chunked
            )

        node = state.target_node
        return (
            self.enable_hicache
            and node.backuped
            and node.write_through_pending_id is None
            and self._needs_incremental_component_backup(node, state.active_components)
        )

    def _insert_tail_step(self, state: _InsertWalkState) -> None:
        """Refresh the LRUs and append terminal backup actions."""
        if state.target_node is not self.root_node:
            for component in state.active_components:
                if component.component_type == BASE_COMPONENT_TYPE:
                    continue
                component.refresh_lru(
                    LRURefreshPhase.INSERT_END, state.target_node, self.root_node
                )

        if self._should_backup_after_insert(state):
            state.pending_actions.append(
                self._build_backup_kv_action(state.target_node, pre_evict=state.chained)
            )

    def _split_node(
        self, key: RadixKey, child: UnifiedTreeNode, split_len: int, *, touch=True
    ) -> tuple[UnifiedTreeNode, Optional[CacheAction | ComponentAction]]:
        new_node = self._new_node(priority=child.priority)
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count
        new_node.creation_time = child.creation_time
        if not touch:
            new_node.last_access_time = child.last_access_time
        new_node.host_last_access_time = child.host_last_access_time
        if self.kv_session_retention is not None:
            new_node.host_session_count = child.host_session_count
            new_node.host_session_active = child.host_session_active
            new_node.host_session_known = child.host_session_known
            new_node.host_session_unknown = child.host_session_unknown
        # Split fragments stay on the anchor's root path for the ack's walk.
        new_node.load_back_pending_id = child.load_back_pending_id

        self._for_each_component_lru(child, UnifiedLRUList.remove_node)

        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )
        new_node.event_hash_value, child.event_hash_value = split_node_hash_value(
            child.event_hash_value, split_len, self.page_size
        )
        if child.kv_events_d_role:
            new_node.kv_events_d_role = True  # both fragments stay unpublished

        for component in self.components:
            component.redistribute_on_node_split(new_parent=new_node, child=child)
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        # A split of a backuped node tells the cache to fix its publish list.
        action: Optional[CacheAction | ComponentAction] = None
        if child.write_through_pending_id is not None:
            ack_id = child.write_through_pending_id
            new_node.write_through_pending_id = ack_id
            action = ReplaceWriteThroughOnNodeSplit(
                ack_id=ack_id,
                old_node_id=child.id,
                new_node_id=new_node.id,
                new_child_node_id=child.id,
            )

        self._for_each_component_lru(
            new_node, UnifiedLRUList.insert_mru, skip_existing=True
        )
        self._for_each_component_lru(
            child, UnifiedLRUList.insert_mru, skip_existing=True
        )
        if touch and self.peer_demote_scope != "on":
            child.last_access_time = get_and_increase_time_counter()
        # Else the tail keeps its tick: a touch=False split (retention
        # admission, host insert) is not an access; with SCOPE=on
        # the access that split the node touches the head after this.
        if child.peer_demoted:
            # Both fragments keep the flag; the access that split the node
            # clears it along its own path (match post-processing or touch).
            new_node.peer_demoted = True
            self._peer_demoted[new_node.id] = new_node
            self._peer_demote_digest(new_node.id, 1)
            if self.peer_demote_counters:
                meta = self._peer_demote_meta.get(child.id)
                if meta is not None:
                    self._peer_demote_meta[new_node.id] = meta
        if (
            self.peer_demote_resolver or self.peer_touch_resolver
        ) and new_node.backuped:
            self._peer_demote_index_add(new_node)

        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(child)
        # Only the new fragment needs qualifying; the child keeps its id.
        self._update_duplicate_tracking(new_node)
        return new_node, action

    def _add_new_node(
        self,
        parent: UnifiedTreeNode,
        key: RadixKey,
        value: torch.Tensor,
        priority: int = 0,
    ) -> UnifiedTreeNode:
        new_node = self._new_node(priority=priority)
        new_node.parent = parent
        new_node.key = key
        new_node.component_data[BASE_COMPONENT_TYPE].value = copy_tag(
            value.clone(), value
        )
        parent.children[key.child_key(self.page_size)] = new_node
        self.component_evictable_size_[BASE_COMPONENT_TYPE] += len(value)
        if self.enable_storage:
            new_node.hash_value = compute_node_hash_values(new_node, self.page_size)

        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(parent)
        gate = getattr(self.kv_events, "d_role_gate", None)
        if gate is not None:
            gate.mark_new(new_node)
        self.kv_events.record_store(new_node)
        return new_node

    def _unevict_node_on_insert(
        self, node: UnifiedTreeNode, fresh_value: torch.Tensor
    ) -> None:
        """Restore an evicted node's Full device value from fresh KV indices
        during insert."""
        ct = BASE_COMPONENT_TYPE
        cd = node.component_data[ct]
        assert cd.value is None
        n = len(fresh_value)
        cd.value = copy_tag(fresh_value.clone(), fresh_value)
        if self._lq_clock is not None:
            cd.unlock_epoch = max(cd.unlock_epoch, self._lq_clock.tag())
        self.component_evictable_size_[ct] += n
        self._update_evictable_leaf_sets(node)
        # A backuped node restored from fresh KV is a duplicate right away.
        self._update_duplicate_tracking(node)
        if node.parent is not None:
            self._update_evictable_leaf_sets(node.parent)
        self.kv_events.record_store(node, medium=StorageMedium.GPU)

    def _update_evictable_leaf_sets(self, node: UnifiedTreeNode) -> None:
        """Update both device and host leaf sets for a node."""
        if node.deleted:
            self._on_stale_leaf_set_update(node)
            return
        if self._is_device_leaf(node):
            self.evictable_device_leaves.add(node)
        else:
            self.evictable_device_leaves.discard(node)

        if self._is_host_leaf(node):
            self.evictable_host_leaves.add(node)
        else:
            self.evictable_host_leaves.discard(node)

    def _on_stale_leaf_set_update(self, node: UnifiedTreeNode) -> None:
        self.evictable_device_leaves.discard(node)
        self.evictable_host_leaves.discard(node)
        self.stale_leaf_set_updates += 1
        if self.stale_leaf_set_updates <= 20 or self.stale_leaf_set_updates % 1000 == 0:
            cd = node.component_data[BASE_COMPONENT_TYPE]
            parent = node.parent
            logger.error(
                "unified tree: leaf-set update on deleted node %d (count=%d) "
                "key_len=%s value=%s host_value=%s lock_ref=%s host_lock_ref=%s "
                "children=%d wt_pending=%s lb_pending=%s parent=%s parent_deleted=%s "
                "parent_in_arena=%s",
                node.id,
                self.stale_leaf_set_updates,
                len(node.key) if node.key is not None else None,
                None if cd.value is None else len(cd.value),
                None if cd.host_value is None else len(cd.host_value),
                [c.lock_ref for c in node.component_data],
                [c.host_lock_ref for c in node.component_data],
                len(node.children),
                node.write_through_pending_id,
                node.load_back_pending_id,
                None if parent is None else parent.id,
                None if parent is None else parent.deleted,
                None if parent is None else (parent.id in self._node_arena),
                stack_info=True,
            )

    def _update_duplicate_tracking(self, node: UnifiedTreeNode) -> None:
        """Register where duplicates are born (acks, split, unevict);
        deregistration is lazy, so entries may be stale and re-checked live."""
        if self._is_settled_full_host_duplicate(node):
            self.full_host_duplicates.setdefault(node.id, node)
        else:
            self.full_host_duplicates.pop(node.id, None)

    def _is_settled_full_host_duplicate(self, node: UnifiedTreeNode) -> bool:
        """Full KV present on both tiers with no in-flight DMA on the node's
        host slots; mid-transfer nodes join the tracking at their ack."""
        cd = node.component_data[BASE_COMPONENT_TYPE]
        return (
            node is not self.root_node
            and cd.value is not None
            and cd.host_value is not None
            and node.write_through_pending_id is None
            and node.load_back_pending_id is None
        )

    def _for_each_component_lru(
        self,
        node: UnifiedTreeNode,
        lru_op,
        target: EvictLayer = EvictLayer.DEVICE,
        skip_existing: bool = False,
    ):
        """Apply lru_op to each aux component's LRU that has data on this node.
        If skip_existing=True, skip components already in the target LRU list."""
        lru_dict = self.host_lru_lists if target is EvictLayer.HOST else self.lru_lists
        for ct in self.component_types:
            if ct == BASE_COMPONENT_TYPE:
                continue  # Full uses leaf sets, not LRU
            cd = node.component_data[ct]
            if (cd.host_value if target is EvictLayer.HOST else cd.value) is not None:
                lru = lru_dict[ct]
                if skip_existing and lru.in_list(node):
                    continue
                lru_op(lru, node)

    def _swa_device_window(self) -> bool:
        """Window SWA residency: the device keeps only the SWA window."""
        return getattr(
            self.components_by_type.get(ComponentType.SWA), "device_window", False
        )

    def _uniform_eviction_component(
        self, component_type: ComponentType, host: bool = False
    ) -> ComponentType:
        swa = self.components_by_type.get(ComponentType.SWA)
        if component_type == ComponentType.SWA and getattr(
            swa, "uniform_lifecycle", False
        ):
            # Host SWA is always evicted jointly with FULL. Window residency
            # walks its own SWA LRU on the device (W4 demotion).
            if host or not getattr(swa, "device_window", False):
                return BASE_COMPONENT_TYPE
        return component_type

    def evict_device_start(
        self, component_type: ComponentType, request_cnt: int
    ) -> None:
        """Begin a component's device-eviction walk for up to request_cnt tokens."""
        component_type = self._uniform_eviction_component(component_type)
        self.components_by_type[component_type].evict_device_start(request_cnt)

    def evict_device_next_node(
        self, component_type: ComponentType, tracker: dict[ComponentType, int]
    ) -> EvictDeviceNextNodeResult:
        """Advance one component eviction step and report whether it progressed."""
        requested_component = component_type
        component_type = self._uniform_eviction_component(component_type)
        result = EvictDeviceNextNodeResult()
        # The walk reads running totals for its doneness check; the result
        # carries only this step's delta.
        updated_tracker = defaultdict(int, tracker)
        walk_tracker = updated_tracker
        if requested_component != component_type:
            # The FULL walker chooses paired victims, but its stop condition must
            # measure the original SWA quota, not previous FULL-only accounting.
            walk_tracker = dict(updated_tracker)
            walk_tracker[component_type] = tracker.get(requested_component, 0)
        result.node_id = self.components_by_type[component_type].evict_device_next_node(
            walk_tracker, result.device_frees, result.host_frees
        )
        for ct, n in updated_tracker.items():
            delta = n - tracker.get(ct, 0)
            if delta:
                result.tracker[ct] = delta
        result.made_progress = result.node_id is not None or bool(result.tracker)
        return result

    def evict_device_end(self, component_type: ComponentType) -> None:
        """Finish a component's device-eviction walk."""
        component_type = self._uniform_eviction_component(component_type)
        self.components_by_type[component_type].evict_device_end()

    def evict_device_leaf(
        self, node_id: NodeId, is_write_back: bool
    ) -> EvictDeviceLeafResult:
        """Evict one device leaf (demote if backuped, delete if write-through);
        for an unbacked write-back node, the result carries the BackupKV for
        the cache to execute and then demote."""
        result = EvictDeviceLeafResult()
        node = self.node_by_id(node_id)
        assert self._is_device_leaf(node), f"node {node.id} is not a D-leaf"
        if not node.backuped:
            if is_write_back:
                result.backup_kv = self._build_backup_kv_action(node, write_back=True)
                return result
            # For metrics: these device tokens have no host copy.
            unbacked_tokens = len(node.component_data[BASE_COMPONENT_TYPE].value)
            if node.children:
                # Host-only children of an unbacked node: deleting only the
                # parent would orphan them (child.parent -> dead node).
                if self._drop_unbacked_subtree(
                    node,
                    result.tracker,
                    device_frees=result.device_frees,
                    host_frees=result.host_frees,
                ):
                    result.unbacked_deleted_tokens = unbacked_tokens
                else:
                    logger.warning(
                        "unified tree: unbacked D-leaf %d has %d host-only "
                        "children that are pinned or mid-DMA; left resident",
                        node.id,
                        len(node.children),
                    )
                return result
            # Write-through: node has no backup, delete entirely.
            self._delete_unbacked_device_leaf(
                node,
                result.tracker,
                device_frees=result.device_frees,
                host_frees=result.host_frees,
            )
            result.unbacked_deleted_tokens = unbacked_tokens
            return result
        self._demote(
            node,
            result.tracker,
            device_frees=result.device_frees,
            host_frees=result.host_frees,
        )
        return result

    def drop_subtree_no_host(self, node_id: NodeId) -> DropSubtreeNoHostResult:
        """Write-back fallback when a D-leaf's D->H backup fails under host
        memory pressure: drop the subtree rooted at the unbacked leaf so
        device eviction keeps making progress instead of leaving its KV
        unevictable until host space frees up."""
        result = DropSubtreeNoHostResult(is_dropped=False)
        node = self.node_by_id(node_id)
        assert self._is_device_leaf(node), f"node {node.id} is not a D-leaf"
        # A failed backup never issues the D->H copy, so the subtree root has
        # no host state and no in-flight DMA reading its device slots.
        assert not node.backuped and node.write_through_pending_id is None
        result.is_dropped = self._drop_unbacked_subtree(
            node,
            result.tracker,
            device_frees=result.device_frees,
            host_frees=result.host_frees,
        )
        return result

    def _drop_unbacked_subtree(
        self,
        node: UnifiedTreeNode,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> bool:
        """Delete an unbacked D-leaf together with its (host-only) descendants.
        Declines (returns False, nothing changed) when the node or any
        descendant is pinned or has an in-flight transfer."""
        if any(cd.host_lock_ref > 0 for cd in node.component_data):
            return False
        descendants: list[UnifiedTreeNode] = []
        stack = list(node.children.values())
        while stack:
            cur = stack.pop()
            if any(
                cd.lock_ref > 0 or cd.host_lock_ref > 0 for cd in cur.component_data
            ):
                return False
            if (
                cur.write_through_pending_id is not None
                or cur.load_back_pending_id is not None
            ):
                return False
            descendants.append(cur)
            stack.extend(cur.children.values())
        for desc in reversed(descendants):
            # Host-only by construction: a device descendant would contradict
            # this node being a D-leaf, and D-leaves evict before ancestors.
            assert desc.evicted and desc.backuped, f"node {desc.id} not host-only"
            self._release_all_component_layers(
                desc,
                StorageMedium.CPU,
                tracker,
                device_frees,
                host_frees,
            )
            self._remove_leaf_from_parent(desc)
        self._delete_unbacked_device_leaf(
            node,
            tracker,
            device_frees=device_frees,
            host_frees=host_frees,
        )
        return True

    def _release_all_component_layers(
        self,
        node: UnifiedTreeNode,
        medium: StorageMedium,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Free every component layer on the node and detach it from the LRU
        lists and evictable leaf sets."""
        if medium == StorageMedium.CPU:
            self._peer_push_removed(node)
        self.kv_events.record_remove(node, medium=medium)
        for comp in self.components:
            self._evict_component_and_detach_lru(
                node,
                comp,
                target=EvictLayer.ALL,
                tracker=tracker,
                device_frees=device_frees,
                host_frees=host_frees,
            )
        self.evictable_device_leaves.discard(node)
        self.evictable_host_leaves.discard(node)

    def _delete_unbacked_device_leaf(
        self,
        node: UnifiedTreeNode,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Delete a device leaf that has no host backup, freeing all layers."""
        self._release_all_component_layers(
            node, StorageMedium.GPU, tracker, device_frees, host_frees
        )
        parent = node.parent
        self._remove_leaf_from_parent(node)
        self._update_evictable_leaf_sets(parent)
        self._iteratively_delete_tombstone_leaf(
            node, tracker, device_frees=device_frees, host_frees=host_frees
        )

    def drive_host_eviction(
        self, component_type: ComponentType, num_tokens: int
    ) -> DriveHostEvictionResult:
        """Evict host resources, preferring host-only LRU leaves in write-through.

        FULL-only and uniform caches can then reclaim complete GPU-backed host
        copies. Write-back reclaims duplicates before host-only leaves.
        """
        component_type = self._uniform_eviction_component(component_type, host=True)
        result = DriveHostEvictionResult()
        if self.kv_session_retention is not None:
            self._drive_retention_host_eviction(num_tokens, result)
            return result

        def reclaim_duplicates(reclaim) -> None:
            # For metrics: Full host tokens that this duplicate reclaim frees.
            before = result.tracker.get(BASE_COMPONENT_TYPE, 0)
            reclaim(num_tokens, result.tracker, result.device_frees, result.host_frees)
            result.duplicate_reclaimed_tokens += (
                result.tracker.get(BASE_COMPONENT_TYPE, 0) - before
            )

        comp = self.components_by_type.get(component_type)
        if comp is not None:
            # Window residency never reclaims host duplicates (W8): dropping
            # the host copy of an (FDH,SDH) node would leave device SWA that
            # can no longer be demoted.
            joint_reclaim = (
                getattr(
                    self.components_by_type.get(ComponentType.SWA),
                    "uniform_lifecycle",
                    False,
                )
                and not self._swa_device_window()
            )
            # Write-through GPU eviction deletes unbacked prefixes. Prefer
            # host-only LRU victims before removing GPU-resident backups.
            if (
                joint_reclaim
                and self.is_write_back
                and component_type == BASE_COMPONENT_TYPE
            ):
                reclaim_duplicates(self._reclaim_host_duplicates)
            elif self.is_write_back and component_type == BASE_COMPONENT_TYPE:
                reclaim_duplicates(self._reclaim_full_host_duplicates)
            comp.drive_host_eviction(
                num_tokens,
                result.tracker,
                result.device_frees,
                result.host_frees,
            )
            if (
                component_type == BASE_COMPONENT_TYPE
                and (
                    joint_reclaim
                    or (
                        not self.is_write_back
                        and self.component_types == (BASE_COMPONENT_TYPE,)
                    )
                )
                and result.tracker[BASE_COMPONENT_TYPE] < num_tokens
            ):
                # All-duplicate host pools still need a way to make progress.
                # Host eviction can also expose a parent's redundant bundle.
                reclaim_duplicates(self._reclaim_host_duplicates)
        return result

    def _drive_retention_host_eviction(self, num_tokens, result):
        """Evict eligible FULL host copies by retention class and CPU LRU.

        Class 0 (count 0, tagged or never tagged) goes before class 1, so a
        zero duplicate is not left behind while a referenced host-only prefix
        is freed. Within each class, nodes without a live request go first
        (soft pin; always the case in class 0), then the base write-through
        order: host-only leaves first, duplicates after them (in both
        classes; _retention_eviction_key). With --peer-kv-demote-source off,
        an all-untagged tree thus evicts exactly as the base path does; with it
        on, this path ignores the demoted-first order. Eligibility and DMA
        safety remain the existing engine checks. Class 1 follows
        SGLANG_KV_SESSION_RETENTION_ORDER (_retention_eviction_key); in both
        orders its freed tokens are counted by idle age (KV_SESSION_STATS).
        """
        retention = self.kv_session_retention
        stats = retention.stats
        candidates = set(self.evictable_host_leaves)
        candidates.update(
            n
            for n in self.full_host_duplicates.values()
            if self._can_reclaim_host_duplicate(n)
        )
        # The key ends with the node id: equal keys never compare nodes.
        heap = [(self._retention_eviction_key(n), n) for n in candidates]
        heapq.heapify(heap)
        queued = {n.id for n in candidates}
        while heap and result.tracker[BASE_COMPONENT_TYPE] < num_tokens:
            key, node = heapq.heappop(heap)
            if node.deleted:
                continue
            parent = node.parent
            # Read before the release: deleting a leaf moves its anchors up.
            active = node.host_session_active > 0
            host_tick = node.host_last_access_time
            before = result.tracker[BASE_COMPONENT_TYPE]
            if node in self.evictable_host_leaves:
                duplicate = False
                self._evict_host_leaf(
                    node, result.tracker, result.device_frees, result.host_frees
                )
            elif self._can_reclaim_host_duplicate(node):
                duplicate = True
                self._release_host_duplicate(
                    node, result.tracker, result.device_frees, result.host_frees
                )
            else:
                continue
            freed = result.tracker[BASE_COMPONENT_TYPE] - before
            if duplicate:
                # Same metric as the base duplicate reclaim
                # (sglang:hicache_host_duplicate_reclaimed_tokens_total).
                result.duplicate_reclaimed_tokens += freed
            stats.note_host_eviction(key[0], duplicate, freed, active)
            if key[0] == 1:
                stats.note_class1_idle(retention.idle_age_s(host_tick), freed)
            if (
                parent is not None
                and parent.id not in queued
                and (
                    parent in self.evictable_host_leaves
                    or self._can_reclaim_host_duplicate(parent)
                )
            ):
                heapq.heappush(heap, (self._retention_eviction_key(parent), parent))
                queued.add(parent.id)

    def evict_excess_path_states(
        self,
        tail_node_id: NodeId,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        self.components_by_type[ComponentType.MAMBA]._evict_excess_path_states(
            self.node_by_id(tail_node_id), device_frees, host_frees
        )

    def _can_reclaim_host_duplicate(self, node: UnifiedTreeNode) -> bool:
        """A redundant complete host bundle at the host-coverage frontier.

        Device locks are harmless: all device values remain untouched. Host
        locks and pending DMA protect host addresses. Keeping backed children
        forbids reclaiming their parent, preserving write-through ancestry.
        """
        if node is self.root_node or (
            node.write_through_pending_id is not None
            or node.load_back_pending_id is not None
        ):
            return False
        if self._backup_guarded_ancestors.get(node.id):
            return False  # a descendant is backing up; this copy is not redundant yet
        full = node.component_data[BASE_COMPONENT_TYPE]
        if full.value is None or full.host_value is None:
            return False
        for ct in self.component_types:
            cd = node.component_data[ct]
            if cd.host_lock_ref > 0:
                return False
            if cd.host_value is not None and cd.value is None:
                return False  # Never discard an auxiliary pool's sole copy.
        if getattr(
            self.components_by_type.get(ComponentType.SWA), "uniform_lifecycle", False
        ):
            swa = node.component_data[ComponentType.SWA]
            if (
                swa.value is None
                or swa.host_value is None
                or len(swa.value) != len(full.value)
                or len(swa.host_value) != len(full.host_value)
            ):
                return False
        return not any(child.backuped for child in node.children.values())

    def _reclaim_host_duplicates(
        self,
        num_tokens: int,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Reclaim complete HOST bundles while retaining every GPU component.

        Eligibility depends only on replicated tree state, never rank-local
        leases or allocator counters. Logical access times give LRU ordering;
        node IDs break ties deterministically across TP ranks.
        A released child can expose its parent; no full-tree rescan is needed.
        """
        # --peer-kv-demote-source: demoted copies first (strict priority).
        candidates = [
            (not node.peer_demoted, node.host_last_access_time, node.id, node)
            for node in self.full_host_duplicates.values()
            if self._can_reclaim_host_duplicate(node)
        ]
        heapq.heapify(candidates)
        queued = {c[2] for c in candidates}
        while candidates and tracker[BASE_COMPONENT_TYPE] < num_tokens:
            node = heapq.heappop(candidates)[3]
            if not self._can_reclaim_host_duplicate(node):
                continue
            self._release_host_duplicate(node, tracker, device_frees, host_frees)
            parent = node.parent
            if (
                parent is not None
                and parent.id not in queued
                and self._can_reclaim_host_duplicate(parent)
            ):
                heapq.heappush(
                    candidates,
                    (
                        not parent.peer_demoted,
                        parent.host_last_access_time,
                        parent.id,
                        parent,
                    ),
                )
                queued.add(parent.id)

    def _release_host_duplicate(self, node, tracker, device_frees, host_frees):
        """Release one eligible host bundle; retain every GPU component.

        GPU_ORDER exception: an unlocked demoted D-leaf goes
        whole. Shared by the duplicate reclaim and the retention eviction.
        """
        assert self._can_reclaim_host_duplicate(node)
        whole = (
            self.peer_demote_gpu_order
            and node.peer_demoted
            and not node.children
            and self._is_device_leaf(node)
        )
        # Remove the publication while its exact old KV indices still exist.
        # Derived byte pools share FULL/SWA anchors and own no allocations.
        self._peer_push_removed(node)
        self.kv_events.record_remove(node, medium=StorageMedium.CPU)
        for comp in self.components:
            self._evict_component_and_detach_lru(
                node,
                comp,
                target=EvictLayer.HOST,
                tracker=tracker,
                device_frees=device_frees,
                host_frees=host_frees,
            )
        self.full_host_duplicates.pop(node.id, None)
        if whole:
            # GPU_ORDER: an unlocked class-2 D-leaf goes
            # whole (its newer copy is elsewhere). The GPU frees do not
            # count toward the host tokens asked for.
            self._delete_unbacked_device_leaf(
                node,
                defaultdict(int),
                device_frees=device_frees,
                host_frees=host_frees,
            )
        else:
            self._update_evictable_leaf_sets(node)

    def _reclaim_full_host_duplicates(
        self,
        num_tokens: int,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Reclaim Full host duplicates until num_tokens are freed; pass 1
        spares evictable D-leaves (imminent free demotes), pass 2 takes them."""
        swept_ids: list[NodeId] = []
        for spare_imminent_demotes in (True, False):
            if tracker[BASE_COMPONENT_TYPE] >= num_tokens:
                break
            for node in self.full_host_duplicates.values():
                if tracker[BASE_COMPONENT_TYPE] >= num_tokens:
                    break
                cd = node.component_data[BASE_COMPONENT_TYPE]
                if cd.value is None or cd.host_value is None:
                    swept_ids.append(node.id)  # stale entry
                    continue
                if spare_imminent_demotes and node in self.evictable_device_leaves:
                    continue
                if not self._can_reclaim_full_host_duplicate(node):
                    continue
                self._release_full_host_duplicate(
                    node, tracker, device_frees, host_frees
                )
                swept_ids.append(node.id)  # released -> no longer a duplicate
        # Sweep after the walk: the dict must not be mutated mid-iteration.
        for nid in swept_ids:
            self.full_host_duplicates.pop(nid, None)

    def _can_reclaim_full_host_duplicate(self, node: UnifiedTreeNode) -> bool:
        """Full on both tiers, no in-flight DMA, no Full host lock; checked
        live because tracking may be stale."""
        cd = node.component_data[BASE_COMPONENT_TYPE]
        if node is self.root_node or cd.value is None or cd.host_value is None:
            return False
        if (
            node.write_through_pending_id is not None
            or node.load_back_pending_id is not None
        ):
            return False
        return cd.host_lock_ref == 0

    def _release_full_host_duplicate(
        self,
        node: UnifiedTreeNode,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Free only the Full host layer; aux host slices stay under their own
        pools' LRU (a host-only aux slice may be a sole copy)."""
        assert self._can_reclaim_full_host_duplicate(node)
        self._peer_push_removed(node)
        self.kv_events.record_remove(node, medium=StorageMedium.CPU)
        self._evict_component_and_detach_lru(
            node,
            self.components_by_type[BASE_COMPONENT_TYPE],
            target=EvictLayer.HOST,
            tracker=tracker,
            device_frees=device_frees,
            host_frees=host_frees,
        )
        self.write_back_duplicate_reclaim_digest = (
            self.write_back_duplicate_reclaim_digest * 1000003 + node.id + 1
        ) & _RECLAIM_DIGEST_MASK

    def _evict_host_leaf(
        self,
        node: UnifiedTreeNode,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Atomically evict all components on a host leaf.

        All freed tokens are accumulated into *tracker*."""
        assert self._is_host_leaf(node), f"node {node.id} is not an H-leaf"

        # kv-p2p: eviction is lease-blind (a rank-local lease
        # table must never steer a structural decision). Pages a peer is still
        # reading are withheld from the free list by the host pool's
        # PageSlotAllocator when they are freed below.
        self._peer_push_removed(node)
        self.kv_events.record_remove(node, medium=StorageMedium.CPU)
        for comp in self.components:
            _, hf = self._evict_component_and_detach_lru(
                node,
                comp,
                target=EvictLayer.ALL,
                tracker=None,
                device_frees=device_frees,
                host_frees=host_frees,
            )
            tracker[comp.component_type] += hf
        self.evictable_host_leaves.discard(node)
        self._remove_leaf_from_parent(node)
        self._iteratively_delete_tombstone_leaf(node, tracker, device_frees, host_frees)

    def drop_backed_device_leaves(self, num_tokens: int) -> DriveHostEvictionResult:
        """Window SWA residency host headroom (W8b): demote the least recently
        used unlocked, backed-up device leaves and delete them from the host
        until ``num_tokens`` FULL host tokens are free. Scheduler level only,
        never inside an insert walk; the tracker counts host tokens."""
        result = DriveHostEvictionResult()
        device_tracker: dict[ComponentType, int] = defaultdict(int)
        # Replicated access times with node ids as tie-breakers keep the victim
        # order identical on every TP rank.
        heap = [(n.last_access_time, n.id, n) for n in self.evictable_device_leaves]
        heapq.heapify(heap)
        while heap and result.tracker[BASE_COMPONENT_TYPE] < num_tokens:
            _, _, node = heapq.heappop(heap)
            if not self._can_drop_backed_device_leaf(node):
                continue
            parent = node.parent
            self._demote(node, device_tracker, result.device_frees, result.host_frees)
            if node in self.evictable_host_leaves:
                self._evict_host_leaf(
                    node, result.tracker, result.device_frees, result.host_frees
                )
            if parent in self.evictable_device_leaves:
                heapq.heappush(heap, (parent.last_access_time, parent.id, parent))
        return result

    def _can_drop_backed_device_leaf(self, node: UnifiedTreeNode) -> bool:
        swa = node.component_data[ComponentType.SWA]
        return (
            not node.deleted
            and node in self.evictable_device_leaves
            and not node.children
            and node.backuped
            and swa.host_value is not None
            and node.write_through_pending_id is None
            and node.load_back_pending_id is None
            and not self._backup_guarded_ancestors.get(node.id)
            and all(cd.host_lock_ref == 0 for cd in node.component_data)
        )

    def demote(self, node_id: NodeId) -> DemoteResult:
        """Release a node's device KV once its host copy exists; the node stays in the
        tree, now host-only."""
        result = DemoteResult()
        self._demote(
            self.node_by_id(node_id),
            result.tracker,
            result.device_frees,
            result.host_frees,
        )
        return result

    def _demote(
        self,
        node: UnifiedTreeNode,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        assert not node.evicted and node.backuped
        trigger = self.components_by_type[BASE_COMPONENT_TYPE]
        self._evict_component_and_detach_lru(
            node,
            trigger,
            target=EvictLayer.DEVICE,
            tracker=tracker,
            device_frees=device_frees,
            host_frees=host_frees,
        )
        self._cascade_evict(
            node, trigger, tracker, device_frees=device_frees, host_frees=host_frees
        )
        self.kv_events.record_remove(node, medium=StorageMedium.GPU)

        # after device eviction, insert aux components into host LRU.
        self._for_each_component_lru(
            node, UnifiedLRUList.insert_mru, target=EvictLayer.HOST, skip_existing=True
        )
        self._update_evictable_leaf_sets(node.parent)

    def _cascade_evict(
        self,
        node: UnifiedTreeNode,
        trigger: TreeComponent,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        target: EvictLayer = EvictLayer.DEVICE,
    ):
        """Cascade eviction from trigger to lower-or-equal priority components."""

        is_leaf = self._is_cascade_evict_leaf(node, target)
        base_evicted = False

        for comp in self.components:
            if self._should_cascade_evict_component(
                node, trigger, comp, target, is_leaf
            ):
                self._evict_component_and_detach_lru(
                    node,
                    comp,
                    target=target,
                    tracker=tracker,
                    device_frees=device_frees,
                    host_frees=host_frees,
                )
                if comp.component_type == BASE_COMPONENT_TYPE:
                    base_evicted = True

        # Now that all components (including SWA which depends on Full.value)
        # have been freed, we can safely tombstone Full.value.
        # This is deferred from evict_component because free_swa needs it.
        if (
            target is EvictLayer.DEVICE
            and trigger.component_type == BASE_COMPONENT_TYPE
        ):
            node.component_data[trigger.component_type].value = None

        if EvictLayer.DEVICE in target and base_evicted:
            node.component_data[BASE_COMPONENT_TYPE].value = None

        self._update_evictable_leaf_sets(node)

    def _is_cascade_evict_leaf(self, node: UnifiedTreeNode, target: EvictLayer) -> bool:
        if target == EvictLayer.DEVICE:
            return node in self.evictable_device_leaves
        if target == EvictLayer.HOST:
            return node in self.evictable_host_leaves
        return False

    @staticmethod
    def _should_cascade_evict_component(
        node: UnifiedTreeNode,
        trigger: TreeComponent,
        comp: TreeComponent,
        target: EvictLayer,
        is_leaf: bool,
    ) -> bool:
        """Return whether a component is an unlocked cascade-eviction target."""
        trigger_priority = trigger.eviction_priority(is_leaf)
        if comp.eviction_priority(is_leaf) > trigger_priority:
            return False
        if comp is trigger or not comp.node_has_component_data(node, target):
            return False

        cd = node.component_data[comp.component_type]
        # A comp whose TRUE internal priority outranks the trigger is only a
        # candidate because leaf-collapse flattened priorities; a lock on it is
        # a legitimate pin and must be spared. A lock on a strictly-lower-
        # priority tier is a real strand and must trip the assertions below.
        if comp.eviction_priority(is_leaf=False) >= trigger.eviction_priority(
            is_leaf=False
        ):
            if EvictLayer.DEVICE in target and cd.lock_ref != 0:
                return False
            if EvictLayer.HOST in target and cd.host_lock_ref != 0:
                return False
            if cd.session_ref > 0 and trigger.session_ref(node) == 0:
                return False
        if EvictLayer.DEVICE in target:
            assert cd.lock_ref == 0
        if EvictLayer.HOST in target:
            assert cd.host_lock_ref == 0
        return True

    def _remove_leaf_from_parent(self, node: UnifiedTreeNode):
        if self.kv_session_retention is not None:
            self.kv_session_retention.on_delete(node)
        for component in self.components:
            component.discard_deleted_session_leaf(node)

        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node
        # Deleted nodes must not linger in duplicate tracking as ghosts.
        self.full_host_duplicates.pop(node.id, None)
        self._unregister_node(node)

    def _evict_component_and_detach_lru(
        self,
        node: UnifiedTreeNode,
        comp: TreeComponent,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        target: EvictLayer = EvictLayer.DEVICE,
        tracker: Optional[dict[ComponentType, int]] = None,
    ) -> tuple[int, int]:
        device_freed, host_freed = comp.evict_component(
            node, target=target, device_frees=device_frees, host_frees=host_frees
        )
        if tracker is not None:
            if EvictLayer.DEVICE in target:
                tracker[comp.component_type] += device_freed
            elif EvictLayer.HOST in target:
                tracker[comp.component_type] += host_freed
        if host_freed > 0:
            # Rolling digest of every host-layer release (node id, component),
            # cross-checked across TP ranks by UnifiedRadixCache.structural_digest.
            self.host_release_digest = (
                self.host_release_digest * 1000003
                + node.id * 8
                + int(comp.component_type.value)
                + 1
            ) & _RECLAIM_DIGEST_MASK
        if device_freed > 0:
            if (
                node.peer_demoted
                and self.peer_demote_counters
                and self.peer_demote_resolver
                and comp.component_type == BASE_COMPONENT_TYPE
            ):
                xst = self.peer_demote_xstats
                xst.class2_gpu_evicted_nodes += 1
                xst.class2_gpu_evicted_tokens += device_freed
            self.device_evict_digest = (
                self.device_evict_digest * 1000003
                + node.id * 16
                + int(comp.component_type.value) * 2
                + int(node.peer_demoted)
                + 1
            ) & _RECLAIM_DIGEST_MASK

        # Detach from the appropriate LRU list(s)
        ct = comp.component_type
        for layer, lru_lists in (
            (EvictLayer.DEVICE, self.lru_lists),
            (EvictLayer.HOST, self.host_lru_lists),
        ):
            if layer in target:
                lru = lru_lists[ct]
                if lru.in_list(node):
                    lru.remove_node(node)
        return device_freed, host_freed

    def _iteratively_delete_tombstone_leaf(
        self,
        deleted_node: UnifiedTreeNode,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ):
        """Walk up from *deleted_node* and cascade-delete childless ancestors.

        Only the Full (base) component decides whether a node survives:
          - Full device present  → keep as D-leaf
          - Full host present    → keep as H-leaf
          - neither              → evict all remaining data, delete, continue up
        """
        self._iteratively_delete_tombstone_ancestors(
            deleted_node.parent, tracker, device_frees, host_frees
        )

    def _iteratively_delete_tombstone_ancestors(
        self,
        cur: UnifiedTreeNode,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Delete childless tombstone ancestors until a live or locked node is reached."""
        ct = BASE_COMPONENT_TYPE
        while cur != self.root_node and len(cur.children) == 0:
            if cur.deleted:
                # Orphaned subtree: nothing above is in the tree any more.
                self._on_stale_leaf_set_update(cur)
                break
            if any(
                cd.lock_ref > 0 or cd.host_lock_ref > 0 for cd in cur.component_data
            ):
                break

            has_device = cur.component_data[ct].value is not None
            has_host = cur.component_data[ct].host_value is not None

            if has_device:
                self._update_evictable_leaf_sets(cur)
                break

            # Full device absent — clean up orphaned aux device data.
            for comp in self.components_by_type.values():
                if comp.node_has_component_data(cur):
                    self._evict_component_and_detach_lru(
                        cur,
                        comp,
                        target=EvictLayer.DEVICE,
                        tracker=tracker,
                        device_frees=device_frees,
                        host_frees=host_frees,
                    )

            if has_host:
                self._update_evictable_leaf_sets(cur)
                break

            # Full absent on both layers — evict remaining host data, delete.
            for comp in self.components_by_type.values():
                if comp.node_has_component_data(cur, target=EvictLayer.HOST):
                    self._evict_component_and_detach_lru(
                        cur,
                        comp,
                        target=EvictLayer.HOST,
                        tracker=tracker,
                        device_frees=device_frees,
                        host_frees=host_frees,
                    )

            self.evictable_host_leaves.discard(cur)
            self._remove_leaf_from_parent(cur)
            parent = cur.parent
            self._update_evictable_leaf_sets(parent)
            cur = parent

    def _is_device_leaf(self, node: UnifiedTreeNode) -> bool:
        """D-leaf: Full device value present, no child with Full KV on device,
        unlocked, not root.

        Only the Full (base) component is required; auxiliary components
        (Mamba, SWA) are not mandatory for D-leaf membership."""
        ct = BASE_COMPONENT_TYPE
        if node is self.root_node or node.evicted:
            return False
        if any(cd.lock_ref > 0 for cd in node.component_data):
            return False
        if any(
            child.component_data[ct].value is not None
            for child in node.children.values()
        ):
            return False
        return True

    def _is_host_leaf(self, node: UnifiedTreeNode) -> bool:
        """H-leaf: evicted, Full host value present, no children, unlocked, not root.

        Only the Full (base) component host_value is required; auxiliary
        components are not mandatory for H-leaf membership. In-flight DMA
        marks need no check: marked nodes are never ``evicted``."""
        if node is self.root_node or not node.evicted:
            return False
        if not node.backuped:
            return False
        if any(cd.host_lock_ref > 0 for cd in node.component_data):
            return False
        if len(node.children) > 0:
            return False
        return True

    # ==== HiCache ====

    def set_hicache_enabled(self) -> None:
        self.enable_hicache = True

    def insert_host(
        self,
        node_id: NodeId,
        key: RadixKey,
        host_value: torch.Tensor,
        hash_value: list[str],
    ) -> InsertResult:
        """Insert a host-side (backuped) tree path descending from the given node."""
        node = self.node_by_id(node_id)
        total_len = len(key)
        # A host insert is not a GPU access: host MRU comes from the commit
        # paths (touch_host_nodes). --peer-kv-demote-source keeps its old
        # touches, so the demote experiments keep their order.
        demote = self.peer_demote_enabled
        scope_on = demote and self.peer_demote_scope == "on"
        if demote:
            self._touch_node(node)
        if scope_on:
            # SCOPE: inserting below the anchor uses its whole path;
            # clear the class of its ancestors too (no tick change).
            cur = node.parent
            while cur is not None and cur is not self.root_node:
                if cur.peer_demoted:
                    self._peer_demote_clear(cur, evicted=False)
                cur = cur.parent
        if total_len == 0:
            return InsertResult(prefix_len=0, mamba_exist=True)

        child_key = key.child_key(self.page_size)
        matched_length = 0
        cache_actions: list[CacheAction | ComponentAction] = []
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            if demote and not scope_on:
                self._touch_node(node)
            prefix_len = node.key.match(key, page_size=self.page_size)

            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                node, action = self._split_node(
                    node.key, node, prefix_len, touch=demote
                )
                if action is not None:
                    cache_actions.append(action)
            if scope_on:
                self._touch_node(node)  # the matched part, after the split

            if len(key):
                child_key = key.child_key(self.page_size)

        result = InsertResult(
            prefix_len=matched_length,
            total_len=total_len,
            cache_actions=cache_actions,
        )
        if len(key) == 0:
            if (
                node is not self.root_node
                and node.component_data[BASE_COMPONENT_TYPE].host_value is not None
            ):
                result.inserted_host_node = node.id
            return result

        # Drop the refill only under write-through (a non-write-back policy).
        if node is not self.root_node and not node.backuped and not self.is_write_back:
            logger.info(
                "HiCache prefetch dropped %d-token refill under un-backed-up node %d",
                len(host_value),
                node.id,
            )
            result.host_insert_dropped = True
            return result

        new_node = self._new_node(priority=node.priority)
        new_node.parent = node
        new_node.key = key
        new_node.hash_value = hash_value
        new_node.component_data[BASE_COMPONENT_TYPE].host_value = host_value.clone()
        node.children[child_key] = new_node
        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(node)
        result.inserted_host_node = new_node.id
        return result

    def build_backup_spec(self, node_id: NodeId):
        """Read a node's device->host backup spec (device value + component transfers) now."""
        return self._build_backup_spec(self.node_by_id(node_id))

    def begin_backup_guard(self, node_id: NodeId) -> list[NodeId]:
        """Pin the ancestors' host copies while node_id's D->H backup runs;
        write-through ancestry (child backed => parent backed) must survive the
        backup's own host eviction. Returns the token for end_backup_guard."""
        ids: list[NodeId] = []
        cur = self.node_by_id(node_id).parent
        while cur is not None and cur is not self.root_node:
            ids.append(cur.id)
            self._backup_guarded_ancestors[cur.id] = (
                self._backup_guarded_ancestors.get(cur.id, 0) + 1
            )
            cur = cur.parent
        return ids

    def end_backup_guard(self, ids: list[NodeId]) -> None:
        for nid in ids:
            n = self._backup_guarded_ancestors.get(nid, 0) - 1
            if n <= 0:
                self._backup_guarded_ancestors.pop(nid, None)
            else:
                self._backup_guarded_ancestors[nid] = n

    def _build_backup_spec(self, node: UnifiedTreeNode):
        """Gather missing Full and component transfers for Host backup."""
        device_value = node.component_data[BASE_COMPONENT_TYPE].value
        assert device_value is not None
        if node.backuped:
            device_value = device_value[:0]

        comp_xfers: dict[ComponentType, list] = {}
        for comp in self.components:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            cd = node.component_data[comp.component_type]
            if cd.host_value is not None and not comp.needs_incremental_backup(node):
                continue
            t = comp.build_hicache_transfers(node, CacheTransferPhase.BACKUP_HOST)
            if t:
                comp_xfers[comp.component_type] = t
        return device_value, comp_xfers

    def build_storage_backup_spec(
        self, node_id: NodeId, pass_prefix_keys: bool
    ) -> Optional[StorageBackupSpec]:
        """Gather a node's device->storage backup spec; None if the node is not backuped."""
        node = self.node_by_id(node_id)
        if not node.backuped:
            return None
        prefix_keys = None
        if pass_prefix_keys:
            prefix_keys = node.get_prefix_hash_values(node.parent)
        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        for comp in self.components:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            transfers = comp.build_hicache_transfers(
                node, CacheTransferPhase.BACKUP_STORAGE
            )
            if transfers:
                comp_xfers[comp.component_type] = transfers
        return StorageBackupSpec(
            host_value=node.component_data[BASE_COMPONENT_TYPE].host_value,
            token_ids=node.key.token_ids,
            hash_value=node.hash_value,
            prefix_keys=prefix_keys,
            comp_xfers=comp_xfers,
        )

    def build_hicache_transfers(
        self,
        component_type: ComponentType,
        node_id: NodeId,
        phase: CacheTransferPhase,
        *,
        host_indices: Optional[torch.Tensor] = None,
        token_ids: Optional[Sequence[int]] = None,
        prefetch_tokens: int = 0,
        last_hash: Optional[str] = None,
    ) -> Optional[list[PoolTransfer]]:
        """Route a build_hicache_transfers call to the component for the given type."""
        return self.components_by_type[component_type].build_hicache_transfers(
            self.node_by_id(node_id),
            phase,
            host_indices=host_indices,
            token_ids=token_ids,
            prefetch_tokens=prefetch_tokens,
            last_hash=last_hash,
        )

    def load_back_pending_on_path(
        self, from_node_id: NodeId, until_node_id: NodeId
    ) -> bool:
        """Whether a node from ``from_node`` up to (exclusive) ``until_node`` is
        pinned by an unacked load-back."""
        until_node = self.node_by_id(until_node_id)
        node = self.node_by_id(from_node_id)
        while node is not until_node and node is not self.root_node:
            if node.load_back_pending_id is not None:
                return True
            node = node.parent
        return False

    def build_load_back_spec(
        self,
        node_id: NodeId,
        req: Optional[Req] = None,
        active_components: Optional[Sequence[ComponentType]] = None,
    ) -> tuple[PoolTransfer, dict[ComponentType, list[PoolTransfer]]]:
        """Build the H->D load-back KV transfer plus per-component aux transfers."""
        # Component hooks take primitives, not Req: extract its fields here.
        mamba_pool_idx = req.mamba_pool_idx if req is not None else None
        node = self.node_by_id(node_id)
        kv_xfer = self.components_by_type[BASE_COMPONENT_TYPE].build_hicache_transfers(
            node, CacheTransferPhase.LOAD_BACK
        )[0]
        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        for comp in self._resolve_active_components(active_components):
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            t = comp.build_hicache_transfers(
                node, CacheTransferPhase.LOAD_BACK, mamba_pool_idx=mamba_pool_idx
            )
            if t:
                comp_xfers[comp.component_type] = t
        # Reject transfers that would claim a node pinned by another load-back
        # anchor; the empty spec fails the load and PrefillAdder defers the
        # request until that load's ack (load_back_in_flight).
        if any(
            self.node_by_id(nid).load_back_pending_id not in (None, node_id)
            for xfers in ([kv_xfer], *comp_xfers.values())
            for xfer in xfers
            for nid in xfer.nodes_to_load or ()
        ):
            empty_kv = PoolTransfer(
                name=PoolName.KV,
                host_indices=torch.empty((0,), dtype=torch.int64, device="cpu"),
                nodes_to_load=[],
            )
            return empty_kv, {}
        return kv_xfer, comp_xfers

    def split_for_swa_window_load(
        self, node_id: NodeId
    ) -> Optional[CacheAction | ComponentAction]:
        """Window SWA residency: trim the host-only SWA node that completes the
        window above ``node_id`` before a load-back locks and builds transfers,
        so only ceil_page(W) SWA tokens load. No-op otherwise."""
        if not self._swa_device_window():
            return None
        swa = self.components_by_type[ComponentType.SWA]
        return swa.split_for_window_load(self.node_by_id(node_id))

    def host_load_chunk_end(
        self, node_id: NodeId, max_tokens: int, *, split: bool = False
    ) -> tuple[NodeId, int, int]:
        """Chunked bootstrap prefetch (SGLANG_BWP_CHUNKED): where the first
        chunk of ``load_back(node_id)`` ends.

        ``load_back(node_id)`` loads the evicted run that ends at ``node_id``
        (the host-only nodes up to the first device-on or load-pending
        ancestor). Returns ``(end_id, chunk_tokens, run_tokens)``: ``end_id``
        is the deepest node of that run whose top part (in prefix order) holds
        at most ``max_tokens`` (page-aligned), ``chunk_tokens`` that part's
        length and ``run_tokens`` the whole run's length (0: nothing to load;
        ``end_id`` is then ``node_id``). When the limit falls inside a node and
        ``split`` is set, that node is split at the page-aligned limit and the
        top fragment ends the chunk; without ``split`` the planned length is
        returned and the tree is unchanged. A node that must not be split (a
        write-through in flight, or a limit below one page into it) ends the
        chunk at the node above it, or, if it is the first node of the run, is
        loaded whole. Without ``split`` a planned cut inside a node returns
        that node's id (the top fragment gets a new id when it splits).
        Deterministic: same tree and arguments, same result on
        every TP rank (no wall clock, no set iteration)."""
        page = self.page_size
        node = self.node_by_id(node_id)
        run: list[UnifiedTreeNode] = []
        cur = node
        while cur.evicted:
            run.append(cur)
            cur = cur.parent
        run.reverse()  # prefix order: the node right under the device prefix first
        run_tokens = sum(len(n.key) for n in run)
        if run_tokens == 0:
            return node_id, 0, 0
        limit = max(page, (int(max_tokens) // page) * page)
        if run_tokens <= limit:
            return node_id, run_tokens, run_tokens
        cum = 0
        end: Optional[UnifiedTreeNode] = None
        for n in run:
            n_len = len(n.key)
            if cum + n_len <= limit:
                cum += n_len
                end = n
                continue
            cut = ((limit - cum) // page) * page
            if 0 < cut < n_len and n.write_through_pending_id is None:
                if split:
                    head, action = self._split_node(n.key, n, cut)
                    assert action is None, "split of a host-only node emitted an action"
                    end = head
                    cum += cut
                    return end.id, cum, run_tokens
                return n.id, cum + cut, run_tokens  # planned; the node splits later
            break
        if end is None:  # the first node cannot be split: load it whole
            end = run[0]
            cum = len(end.key)
        return end.id, cum, run_tokens

    def swa_rebackup_action(self, node_id: NodeId) -> Optional[BackupKV]:
        """The BackupKV a queued node still needs, or None once it is gone or
        backed up (FULL, and SWA where a device value exists)."""
        node = self._node_arena.get(node_id)
        if node is None or node is self.root_node or node.evicted:
            return None
        if node.backuped and not self._needs_incremental_component_backup(node):
            return None
        return self._build_backup_kv_action(node)

    def _compute_swa_blocked_size(self) -> int:
        if not self._swa_device_window():
            return 0
        swa = self.components_by_type[ComponentType.SWA]
        return sum(
            swa._blocked_len(node)
            for node in self._collect_all_nodes()
            if node is not self.root_node
        )

    def recompute_swa_blocked_size(self) -> int:
        """Resynchronize swa_blocked_size_ from the tree (debug, fixtures)."""
        self.swa_blocked_size_ = self._compute_swa_blocked_size()
        return self.swa_blocked_size_

    def prefetch_anchor_info(
        self, node_id: NodeId
    ) -> tuple[Optional[str], Optional[str]]:
        """The anchor node's key extra_key and cache_salt."""
        node = self.node_by_id(node_id)
        if node.key is None:
            return None, None
        return node.key.extra_key, node.key.cache_salt

    def _build_backup_kv_action(
        self, node: UnifiedTreeNode, write_back: bool = False, pre_evict: bool = False
    ) -> BackupKV:
        """Build the backup action for a node and its unbacked ancestors."""
        chain = [node]
        if not write_back:
            ancestor = node.parent
            while (
                ancestor is not None
                and ancestor is not self.root_node
                and not ancestor.backuped
            ):
                chain.append(ancestor)
                ancestor = ancestor.parent
            # write_through: Ancestors first to preserve backup invariant
            chain.reverse()
        return BackupKV([target.id for target in chain], pre_evict=pre_evict)

    def commit_hicache_transfers(
        self,
        node_id: NodeId,
        phase: CacheTransferPhase,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
        *,
        cache_actions: list[CacheAction | ComponentAction],
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:
        """Commit each component's HiCache transfers onto the node."""
        node = self.node_by_id(node_id)
        for ct, xfers in comp_xfers.items():
            self.components_by_type[ct].commit_hicache_transfer(
                node,
                phase,
                xfers,
                cache_actions=cache_actions,
                insert_result=insert_result,
                pool_storage_result=pool_storage_result,
            )

    def commit_backup(
        self,
        node_id: NodeId,
        host_indices: torch.Tensor,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> None:
        """Commit a successful backup to the node."""
        node = self.node_by_id(node_id)
        cache_actions: list[CacheAction | ComponentAction] = []
        if host_indices.numel() > 0:
            kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=host_indices)
            self.components_by_type[BASE_COMPONENT_TYPE].commit_hicache_transfer(
                node,
                CacheTransferPhase.BACKUP_HOST,
                transfers=[kv_xfer],
                cache_actions=cache_actions,
            )
        for ct, xfers in comp_xfers.items():
            self.components_by_type[ct].commit_hicache_transfer(
                node,
                CacheTransferPhase.BACKUP_HOST,
                transfers=xfers,
                cache_actions=cache_actions,
            )
        assert not cache_actions  # BACKUP_HOST emits no actions

    def commit_load_back(
        self,
        node_id: NodeId,
        device_indices: torch.Tensor,
        kv_xfer: PoolTransfer,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> list[CacheAction | ComponentAction]:
        """Commit a successful H->D load-back onto the node; the SWA full->swa mapping
        rebuild is deferred to the orchestration layer."""
        node = self.node_by_id(node_id)
        cache_actions: list[CacheAction | ComponentAction] = []
        # Keep in-flight device slots invisible to prefix matching until the
        # H2D ack. Under write-back this also pins the source host slots.
        for nid in kv_xfer.nodes_to_load or ():
            pinned = self.node_by_id(nid)
            assert pinned.load_back_pending_id in (None, node_id), (
                f"node {nid} pinned by load-back "
                f"{pinned.load_back_pending_id}, new anchor {node_id}"
            )
            pinned.load_back_pending_id = node_id
        kv_xfer.device_indices = device_indices
        self.components_by_type[BASE_COMPONENT_TYPE].commit_hicache_transfer(
            node,
            CacheTransferPhase.LOAD_BACK,
            [kv_xfer],
            cache_actions=cache_actions,
        )
        for nid in kv_xfer.nodes_to_load or ():
            # Lazy with SGLANG_KV_EVENTS_LAZY_LOAD_BACK (6f): O(1) per node here;
            # the per-page events are built in queue order at take().
            self.kv_events.record_store(
                self.node_by_id(nid), medium=StorageMedium.GPU, lazy=True
            )
        for ct, xfers in comp_xfers.items():
            self.components_by_type[ct].commit_hicache_transfer(
                node,
                CacheTransferPhase.LOAD_BACK,
                xfers,
                cache_actions=cache_actions,
            )
        self._update_evictable_leaf_sets(node)
        return cache_actions

    def finish_load_back(self, anchor_node_id: NodeId) -> None:
        """Finalize H->D load-back state along the anchor's root path.

        Clear source-node pins and refresh duplicate tracking after the device
        copies become visible. Split fragments stay on the path, so the walk
        covers them.
        """
        node = self.node_by_id(anchor_node_id)
        while node is not None and node is not self.root_node:
            if node.load_back_pending_id == anchor_node_id:
                node.load_back_pending_id = None
            elif self.is_write_back:
                node = node.parent
                continue
            self._update_duplicate_tracking(node)
            self.touch_host_nodes((node.id,))
            node.last_access_time = get_and_increase_time_counter()
            node = node.parent

    def mark_write_through_pending(self, node_id: NodeId) -> None:
        """Mark a node as having an in-flight write-through backup."""
        node = self.node_by_id(node_id)
        node.write_through_pending_id = node_id

    def finish_write_through(self, node_ids: list[NodeId], ack_id: int) -> None:
        """Clear the write-through-pending mark (when it matches ack_id) and record the
        host store event for each acked node."""
        window = self._swa_device_window()
        for node_id in node_ids:
            node = self.node_by_id(node_id)
            if node.write_through_pending_id == ack_id:
                node.write_through_pending_id = None
                # The backed-up copy becomes a tracked duplicate only now.
                self._update_duplicate_tracking(node)
            # Window residency: advertise the CPU tier only once the host
            # holds this node's SWA as well (I3), like the servicer push.
            if not window or self._swa_host_complete(node):
                self.kv_events.record_store(node, medium=StorageMedium.CPU)
            self._peer_push_servable_node(node)
        self.touch_host_nodes(node_ids)

    def _swa_host_complete(self, node: UnifiedTreeNode) -> bool:
        full_hv = node.component_data[BASE_COMPONENT_TYPE].host_value
        swa_hv = node.component_data[ComponentType.SWA].host_value
        return (
            full_hv is not None and swa_hv is not None and len(swa_hv) == len(full_hv)
        )

    # ------------------------------------------------------------------
    # SGLANG_SYMMETRIC_SKIP_D_ROLE_KV_EVENTS (mem_cache/events.py
    # DRoleEventGate).

    def enable_skip_d_role_kv_events(self) -> DRoleEventGate:
        """Install the D-role gate on this tree's recorder (every TP rank)."""
        gate = self.kv_events.d_role_gate
        if gate is None:
            gate = DRoleEventGate(self.kv_events, self.kv_event_held_tiers)
            self.kv_events.d_role_gate = gate
        return gate

    def kv_event_held_tiers(self, node: UnifiedTreeNode) -> list:
        """The tiers whose BlockStored a published node would have sent by
        now (a promotion publishes these): GPU while it holds device KV (sent
        at its insert, refill or load-back commit), CPU once its host copy is
        acked (finish_write_through's rule)."""
        tiers = []
        if node.component_data[BASE_COMPONENT_TYPE].value is not None:
            tiers.append(StorageMedium.GPU)
        if (
            node.backuped
            and node.write_through_pending_id is None
            and (not self._swa_device_window() or self._swa_host_complete(node))
        ):
            tiers.append(StorageMedium.CPU)
        return tiers

    # ------------------------------------------------------------------
    # kv-p2p (PeerKVServicer) shadow-index feeds.
    # The servicer is None unless --enable-peer-kv-servicer; every helper
    # below is a no-op in that case. The tree never READS servicer state:
    # structural decisions must be identical on every TP rank.
    # ------------------------------------------------------------------

    def _peer_push_range_fn(self, servicer: Any, name: str) -> Optional[Callable]:
        """SGLANG_PEER_KV_PUSH_FAST: the servicer's whole-node range push, or
        None (flag off, or a servicer without the range API)."""
        if not self.peer_push_fast:
            return None
        return getattr(servicer, name, None)

    def _peer_kv_page_views(self, full_hv: torch.Tensor, num_pages: int) -> list:
        """One index view per page of the node's FULL host copy."""
        ps = self.page_size
        if self.peer_push_fast:
            # One split (views) instead of one slice op per page.
            return list(full_hv[: num_pages * ps].split(ps))
        return [full_hv[i * ps : (i + 1) * ps] for i in range(num_pages)]

    def _peer_swa_tail_range(
        self, swa_hv: torch.Tensor, num_pages: int
    ) -> tuple[int, torch.Tensor]:
        """The trailing-window SWA slice as one range: (first page with a
        slice, the slots of pages [first, num_pages)). Page i gets
        swa_hv[i * ps - offset : ...] when that start is >= 0 (offset =
        num_pages * ps - len(swa_hv)), exactly the per-page rule
        (PoolHitPolicy.TRAILING_PAGES)."""
        ps = self.page_size
        offset = num_pages * ps - len(swa_hv)
        first = min(num_pages, max(0, -(-offset // ps)))
        start = first * ps - offset
        return first, swa_hv[start : start + (num_pages - first) * ps]

    def _peer_push_removed(self, node: UnifiedTreeNode) -> None:
        # Every host-layer release of a node passes here first.
        if self.peer_demote_clock is not None:
            self._peer_demote_count_host_eviction(node)
        if node.peer_demoted and not (
            self.peer_demote_gpu_order
            and node.component_data[BASE_COMPONENT_TYPE].value is not None
        ):
            # With GPU_ORDER a host-only release keeps the
            # class while the GPU copy stays: the device heap takes it first;
            # the flag ends with the node (_unregister_node) or an access.
            self._peer_demote_clear(node, evicted=True)
        if self.peer_demote_resolver or self.peer_touch_resolver:
            self._peer_demote_index_drop(node)
        servicer = self.peer_kv_servicer
        if servicer is None or not node.hash_value:
            return
        full_hv = node.component_data[BASE_COMPONENT_TYPE].host_value
        num_pages = len(node.hash_value)
        if full_hv is None or len(full_hv) < num_pages * self.page_size:
            return
        push_range = self._peer_push_range_fn(servicer, "push_removed_range")
        if push_range is not None:
            # Snapshots slots and generations before returning, like below.
            push_range(
                peer_page_keys(
                    node.hash_value, node.key.cache_salt, node.key.extra_key
                ),
                full_hv[: num_pages * self.page_size],
                self.page_size,
            )
            return
        kv_pages = self._peer_kv_page_views(full_hv, num_pages)
        # push_removed snapshots slots and generations before the caller frees
        # this backing. No scheduler decision reads the servicer's shadow.
        servicer.push_removed(
            peer_page_keys(node.hash_value, node.key.cache_salt, node.key.extra_key),
            kv_pages,
        )

    def _peer_push_swa_removed(self, node: UnifiedTreeNode) -> None:
        servicer = self.peer_kv_servicer
        if (
            servicer is None
            or not node.hash_value
            or ComponentType.SWA not in self.component_types
        ):
            return
        swa_hv = node.component_data[ComponentType.SWA].host_value
        if swa_hv is None or len(swa_hv) == 0:
            return
        num_pages = len(node.hash_value)
        offset = num_pages * self.page_size - len(swa_hv)
        push_range = self._peer_push_range_fn(servicer, "push_swa_removed_range")
        if push_range is not None:
            first, swa_range = self._peer_swa_tail_range(swa_hv, num_pages)
            push_range(
                peer_page_keys(
                    node.hash_value, node.key.cache_salt, node.key.extra_key
                ),
                swa_range,
                first,
                self.page_size,
            )
            return
        swa_pages: list = [None] * num_pages
        for i in range(num_pages):
            start = i * self.page_size - offset
            if start >= 0:
                swa_pages[i] = swa_hv[start : start + self.page_size]
        servicer.push_swa_removed(
            peer_page_keys(node.hash_value, node.key.cache_salt, node.key.extra_key),
            swa_pages,
        )

    def _peer_push_servable_node(
        self, node: UnifiedTreeNode, from_prefetch: bool = False
    ) -> None:
        """Push a host-backed node's pages into the servicer's shadow index.

        KV-anchored indices come from the FULL host copy (whole node); the
        SWA-anchored indices cover only the trailing window — pages fully
        inside that window get a slice, earlier pages get None (matching
        PoolHitPolicy.TRAILING_PAGES). ``from_prefetch``: the pages came from
        a peer pull (counts re-pulls of prefixes this worker demoted)."""
        if from_prefetch and self.peer_demote_record_sink is not None:
            pulled = self._peer_from_pull
            pulled[node.id] = True
            pulled.move_to_end(node.id)
            while len(pulled) > self._PEER_LANDED_MAX:
                pulled.popitem(last=False)
        servicer = self.peer_kv_servicer
        if servicer is None:
            return
        full_hv = node.component_data[BASE_COMPONENT_TYPE].host_value
        if full_hv is None:
            return
        swa = self.components_by_type.get(ComponentType.SWA)
        if getattr(swa, "uniform_lifecycle", False):
            swa_hv = node.component_data[ComponentType.SWA].host_value
            if swa_hv is None or len(swa_hv) != len(full_hv):
                return
        if node.hash_value is None:
            node.hash_value = compute_node_hash_values(node, self.page_size)
        num_pages = len(node.hash_value)
        if num_pages == 0 or len(full_hv) < num_pages * self.page_size:
            return
        keys = peer_page_keys(node.hash_value, node.key.cache_salt, node.key.extra_key)
        push_range = self._peer_push_range_fn(servicer, "push_store_range")
        if push_range is not None:
            first, swa_range = num_pages, None
            if ComponentType.SWA in self.component_types:
                swa_hv = node.component_data[ComponentType.SWA].host_value
                if swa_hv is not None and len(swa_hv) > 0:
                    first, swa_range = self._peer_swa_tail_range(swa_hv, num_pages)
            push_range(
                keys,
                full_hv[: num_pages * self.page_size],
                swa_range,
                first,
                self.page_size,
            )
        else:
            kv_pages = self._peer_kv_page_views(full_hv, num_pages)
            swa_pages: list = [None] * num_pages
            if ComponentType.SWA in self.component_types:
                swa_hv = node.component_data[ComponentType.SWA].host_value
                if swa_hv is not None and len(swa_hv) > 0:
                    offset = num_pages * self.page_size - len(swa_hv)
                    for i in range(num_pages):
                        start = i * self.page_size - offset
                        if start >= 0:
                            swa_pages[i] = swa_hv[start : start + self.page_size]
            servicer.push_store(keys, kv_pages, swa_pages)
        if self.peer_demote_resolver or self.peer_touch_resolver:
            self._peer_demote_index[keys[-1]] = node
            if self._peer_demote_recent:
                self._peer_demote_count_return(keys, from_prefetch, node.id)

    # ------------------------------------------------------------------
    # --peer-kv-demote-source (mem_cache/peer_kv_demote.py). The resolver
    # rank turns complete RELEASEs into node ids before the request
    # broadcast; every rank applies the broadcast ids at the same point, so
    # the flags never depend on rank-local state.
    # ------------------------------------------------------------------

    _PEER_DEMOTE_RECENT_MAX = 65536

    def enable_peer_demote(
        self,
        resolver: bool,
        lock_aware: bool = False,
        counters: bool = False,
        gpu_order: bool = False,
        scope: str = "off",
    ) -> None:
        self.peer_demote_enabled = True
        self.peer_demote_scope = scope
        self.peer_demote_resolver = bool(resolver)
        self.peer_demote_lock_aware = bool(lock_aware)
        self.peer_demote_counters = bool(counters)
        self.peer_demote_gpu_order = bool(gpu_order)
        if resolver and counters and self.peer_demote_clock is None:
            self.peer_demote_clock = TickClock()

    _PEER_DEMOTE_DIGEST_MASK = (1 << 42) - 1

    def _peer_demote_digest(self, node_id: NodeId, sign: int) -> None:
        """Add or remove a flagged id in the order-free digest of the flagged
        set (identical on every TP rank). Read by the structural summary
        (SGLANG_PEER_KV_DEMOTE_COUNTERS) and the step-sync fold
        (SGLANG_HICACHE_EVICT_ORDER_CHECK)."""
        mixed = (int(node_id) * 0x9E3779B97F4A7C15) & self._PEER_DEMOTE_DIGEST_MASK
        self._peer_demoted_id_digest = (
            self._peer_demoted_id_digest + sign * mixed
        ) & self._PEER_DEMOTE_DIGEST_MASK

    def _peer_demote_no_node_cause(self, keys: list[str]) -> str:
        """Resolver: why the last leased page is not the end of a live node.
        "inside": the deepest node end among the keys has a live child that
        holds the next page (the lease ended inside it); "gone": that child is
        not there; "unresolved": no node end among the keys."""
        for j in range(len(keys) - 2, -1, -1):
            node = self._peer_demote_index.get(keys[j])
            if node is None or node.deleted:
                continue
            child = child_with_first_key(node, keys[j + 1], first_page_key)
            return "inside" if child is not None else "gone"
        return "unresolved"

    def _peer_demote_last_key(self, node: UnifiedTreeNode) -> Optional[str]:
        if not node.hash_value or node.key is None:
            return None
        return peer_page_keys(
            node.hash_value[-1:], node.key.cache_salt, node.key.extra_key
        )[0]

    def _peer_demote_index_add(self, node: UnifiedTreeNode) -> None:
        key = self._peer_demote_last_key(node)
        if key is not None:
            self._peer_demote_index[key] = node

    def _peer_demote_index_drop(self, node: UnifiedTreeNode) -> None:
        key = self._peer_demote_last_key(node)
        if key is not None and self._peer_demote_index.get(key) is node:
            del self._peer_demote_index[key]

    def _peer_demote_host_pages(self, node: UnifiedTreeNode) -> int:
        hv = node.component_data[BASE_COMPONENT_TYPE].host_value
        return 0 if hv is None else len(hv) // self.page_size

    def select_peer_demote_nodes(
        self, releases, scope: Optional[ScopeContext] = None
    ) -> list[NodeId]:
        """Resolver rank only. ``releases``: [(leased page keys, lookup tick)]
        of complete RELEASEs (with SGLANG_PEER_KV_DEMOTE_COUNTERS a third item,
        the pulling request id). Returns the ids to flag (branch-point rule).
        A NOTICE (trigger "notice") starts at the deepest live
        node end among its keys (they end at the owner's confirmed end); a
        RELEASE or a commit starts at its last key only. ``scope``
        (SGLANG_PEER_KV_DEMOTE_SCOPE shadow / on): also run the scope
        walk and count it against this one ("scope.*"); with "on" flag its
        nodes instead."""
        st = self.peer_demote_stats
        xst = self.peer_demote_xstats
        counters = self.peer_demote_counters
        out: list[NodeId] = []
        seen: set[NodeId] = set()
        for release in releases:
            keys, lookup_tick = release[0], release[1]
            st.releases += 1
            trig = release_trigger(release)
            if counters:
                xst.bump(f"{trig}.releases")
            if trig == "notice":
                node = self._peer_demote_notice_start(keys)
            else:
                node = self._peer_demote_index.get(keys[-1]) if keys else None
                if node is not None and node.deleted:
                    del self._peer_demote_index[keys[-1]]
                    node = None
            nodes: list = []
            if node is None:
                # The lease ended inside a node (content outside the leased
                # path: a branch point) or the node is gone.
                st.no_node += 1
                if counters:
                    cause = self._peer_demote_no_node_cause(keys) if keys else "gone"
                    setattr(
                        xst, "no_node_" + cause, getattr(xst, "no_node_" + cause) + 1
                    )
                    xst.bump(f"{trig}.no_node")
            else:
                nodes, outcome = select_demote_path(
                    node, lookup_tick, self.root_node, self.peer_demote_lock_aware
                )
                if outcome == "locked":
                    xst.locked += 1
                else:
                    setattr(st, outcome, getattr(st, outcome) + 1)
                if outcome == "access" and getattr(release, "held", False):
                    # A held NOTICE (grace): the owner used the chain again.
                    xst.notice_grace_access += 1
                if counters:
                    xst.bump(f"{trig}.{outcome}")
            if scope is not None:
                scoped = self._peer_scope_select(keys, lookup_tick, nodes, scope)
                if scope.apply:
                    nodes = scoped
            if not nodes:
                continue
            rid = release[2] if counters and len(release) > 2 else ""
            if counters:
                mark_tokens = self.page_size * sum(
                    self._peer_demote_host_pages(n) for n in nodes
                )
            for n in nodes:
                if n.id not in seen:
                    seen.add(n.id)
                    out.append(n.id)
                    if counters:
                        self._peer_demote_selected_rid[n.id] = (rid, trig, mark_tokens)
        return out

    def _peer_scope_select(
        self,
        keys: list[str],
        lookup_tick: float,
        today: list,
        scope: ScopeContext,
    ):
        """SCOPE (resolver rank): the scope walk of one mark, and
        its counts against today's walk (``today``), all under "scope.*" in
        the extra stats (tokens: host pages x page size). Covered start: the
        node d that holds the last covered key (resolve_touch_node; a list
        that starts at page 0 resolves through the root); if the key is a
        page of d before its last page, the walk starts at d's parent and d
        stays unmarked (an uncovered tail); else (d ends at the key, or among
        the keys when the node of the deeper keys is gone) it starts at d.
        Read-only."""
        xst = self.peer_demote_xstats
        page = self.page_size
        index = self._peer_demote_index
        nodes: list = []
        d = None
        if keys:
            d, _ = resolve_touch_node(keys, True, index, self.root_node)
        if d is None:
            xst.bump("scope.unresolved")
        else:
            # Pages of d past the covered end. 0: d ends at the covered end,
            # or among the keys (its path child is gone: d is wholly covered).
            inside = 0
            if self._peer_demote_last_key(d) != keys[-1]:
                dkeys = peer_page_keys(d.hash_value, d.key.cache_salt, d.key.extra_key)
                if keys[-1] in dkeys:
                    inside = len(dkeys) - 1 - dkeys.index(keys[-1])
            if inside:
                # The covered end is inside d: start at its parent; d stays
                # unmarked (an uncovered tail).
                start, child = d.parent, d
                xst.bump("scope.uncovered_tail")
                xst.bump("scope.uncovered_tail_tokens", inside * page)
            else:
                start, child = d, None
                tails = [
                    x
                    for x in d.children.values()
                    if not x.deleted and not x.peer_demoted
                ]
                if tails:
                    # Class-1 children past the covered end (a longer or a
                    # diverged copy): never marked here.
                    xst.bump("scope.uncovered_tail")
                    xst.bump(
                        "scope.uncovered_tail_tokens",
                        page * sum(self._peer_demote_host_pages(x) for x in tails),
                    )
            nodes, outcome = select_scope_path(
                start, child, lookup_tick, self.root_node, scope
            )
            xst.bump(f"scope.{outcome}")
            marked = set(nodes)
            for n in nodes:
                if any(
                    x not in marked and not x.peer_demoted and not x.deleted
                    for x in n.children.values()
                ):
                    xst.bump("scope.dormant")  # acts once n becomes a leaf
                else:
                    xst.bump("scope.leaf")
        now = set(today)
        more = [n for n in nodes if n not in now]
        fewer = [n for n in today if n not in set(nodes)]
        xst.bump("scope.marks")
        for name, group in (
            ("v3", nodes),
            ("today", today),
            ("more", more),
            ("fewer", fewer),
        ):
            xst.bump(
                f"scope.{name}_tokens",
                page * sum(self._peer_demote_host_pages(n) for n in group),
            )
        return nodes

    def _peer_demote_notice_start(self, keys: list[str]):
        """The deepest live node whose last page key is among ``keys`` (a
        NOTICE's keys end at the owner's confirmed end): the owner may have
        lost the tail since the LOOKUP. None if no live node end is there."""
        index = self._peer_demote_index
        for key in reversed(keys):
            node = index.get(key)
            if node is None:
                continue
            if node.deleted:
                del index[key]
                continue
            return node
        return None

    def apply_peer_demote(self, node_ids: Sequence[NodeId]) -> int:
        """Every rank, from the broadcast PeerKVDemoteReq: flag the nodes."""
        if not self.peer_demote_enabled:
            return 0
        st = self.peer_demote_stats
        flagged = 0
        for node_id in node_ids:
            node = self._node_arena.get(node_id)
            if (
                node is None
                or node is self.root_node
                or node.deleted
                or node.peer_demoted
                or not node.backuped
            ):
                continue
            node.peer_demoted = True
            self._peer_demoted[node.id] = node
            st.flagged_nodes += 1
            st.flagged_pages += self._peer_demote_host_pages(node)
            flagged += 1
            self._peer_demote_digest(node.id, 1)
            if self.peer_demote_counters and self.peer_demote_resolver:
                self._peer_demote_count_mark(node)
        self._peer_demote_selected_rid.clear()
        return flagged

    def _peer_demote_clear(self, node: UnifiedTreeNode, evicted: bool) -> None:
        node.peer_demoted = False
        self._peer_demoted.pop(node.id, None)
        st = self.peer_demote_stats
        pages = self._peer_demote_host_pages(node)
        self._peer_demote_digest(node.id, -1)
        meta = None
        if self.peer_demote_counters:
            meta = self._peer_demote_meta.pop(node.id, None)
            self._peer_demote_count_age(node, evicted, pages, meta)
        if not evicted:
            st.cleared_access_nodes += 1
            st.cleared_access_pages += pages
            if meta is not None:
                self.peer_demote_xstats.bump(f"{meta[2]}.cleared_nodes")
                self.peer_demote_xstats.bump(f"{meta[2]}.cleared_pages", pages)
            return
        st.evicted_nodes += 1
        st.evicted_pages += pages
        value = pages
        if meta is not None:
            # The GPU copy stays on a duplicate reclaim; else it left.
            kept = node.component_data[BASE_COMPONENT_TYPE].value is not None
            where = "gpu_kept" if kept else "left"
            self.peer_demote_xstats.bump(f"{meta[2]}.evicted_{where}_nodes")
            self.peer_demote_xstats.bump(f"{meta[2]}.evicted_{where}_pages", pages)
            value = (pages, meta[2], meta[0], time.monotonic(), node.id, meta[4])
        if self.peer_demote_resolver:
            # Remember the dropped copy: a later publish of its last page is
            # a re-pull (from a peer) or a recompute (local prefill).
            key = self._peer_demote_last_key(node)
            if key is not None:
                recent = self._peer_demote_recent
                recent[key] = value
                recent.move_to_end(key)
                while len(recent) > self._PEER_DEMOTE_RECENT_MAX:
                    recent.popitem(last=False)

    def _peer_demote_count_mark(self, node: UnifiedTreeNode) -> None:
        """COUNTERS, resolver rank: the mark's meta (flag
        time, rid, trigger, tick age at the mark, mark tokens) and the mark
        counters."""
        rid, trig, mark_tokens = self._peer_demote_selected_rid.get(
            node.id, ("", "release", 0)
        )
        now = time.monotonic()
        clock = self.peer_demote_clock
        tick_age = clock.age_s(node.last_access_time, now) if clock else None
        self._peer_demote_meta[node.id] = (now, rid, trig, tick_age, mark_tokens)
        xst = self.peer_demote_xstats
        pages = self._peer_demote_host_pages(node)
        xst.bump(f"{trig}.marked_nodes")
        xst.bump(f"{trig}.marked_pages", pages)
        if node.children:
            xst.marks_dormant += 1
        else:
            xst.marks_leaf += 1
        if node.component_data[BASE_COMPONENT_TYPE].value is not None:
            xst.marks_gpu_pages += pages

    def _peer_demote_count_host_eviction(self, node: UnifiedTreeNode) -> None:
        """COUNTERS, resolver rank: the age (tick clock) of a
        host copy at its release, by class and, for class 2, by trigger; and
        a committed landing released with no access since the landing."""
        xst = self.peer_demote_xstats
        now = time.monotonic()
        age = self.peer_demote_clock.age_s(node.last_access_time, now)
        bucket = age_fine_bucket(age)
        meta = None
        if node.peer_demoted:
            meta = self._peer_demote_meta.get(node.id)
            trig = meta[2] if meta is not None else "release"
            hist = xst.age_host_c2.get(trig)
            if hist is None:
                hist = xst.age_host_c2[trig] = [0] * (len(AGE_FINE_EDGES_S) + 1)
            hist[bucket] += 1
        else:
            xst.age_host_c1[bucket] += 1
        if self._peer_landed:
            landed_tick = self._peer_landed.pop(node.id, None)
            if landed_tick is not None and node.last_access_time == landed_tick:
                xst.landing_evicted_unused += 1
        if self.peer_demote_record_sink is not None:
            self._peer_demote_record_host_eviction(node, now, age, meta)

    def _peer_demote_record_host_eviction(
        self, node: UnifiedTreeNode, now: float, age: Optional[float], meta
    ) -> None:
        """RECORDS, resolver rank (R2): one record per host
        eviction, class 1 too. path: dup_reclaim when the node keeps its GPU
        copy, else heap. The pull destination is unknown to the owner."""
        keys = (
            peer_page_keys(
                [node.hash_value[0], node.hash_value[-1]],
                node.key.cache_salt,
                node.key.extra_key,
            )
            if node.hash_value and node.key is not None
            else [None, None]
        )
        gpu = node.component_data[BASE_COMPONENT_TYPE].value is not None
        record = {
            "event": "host_evict",
            "ts": time.time(),
            "node": node.id,
            "first_key": keys[0],
            "last_key": keys[-1],
            "pages": len(node.hash_value) if node.hash_value else 0,
            "class": 2 if node.peer_demoted else 1,
            "path": "dup_reclaim" if gpu else "heap",
            "tick_age_s": None if age is None else round(age, 3),
            "from_pull": self._peer_from_pull.pop(node.id, False),
        }
        if node.peer_demoted and meta is not None:
            record["cause"] = meta[2]
            record["rid"] = meta[1]
            record["mark_age_s"] = round(max(0.0, now - meta[0]), 3)
            record["tick_age_at_mark_s"] = meta[3]
        try:
            self.peer_demote_record_sink(record)
        except Exception:  # a record must never break eviction
            logger.warning("PeerKV host-eviction record failed", exc_info=True)
            self.peer_demote_record_sink = None

    _PEER_LANDED_MAX = 4096

    def note_peer_landing(self, node_id: NodeId) -> None:
        """Reader, COUNTERS, resolver rank: a
        landing whose DEMOTE_OK was sent. Its release with no access since
        then counts as landing_evicted_unused."""
        if not (self.peer_demote_counters and self.peer_demote_resolver):
            return
        node = self._node_arena.get(node_id)
        if node is None:
            return
        landed = self._peer_landed
        landed[node_id] = node.last_access_time
        landed.move_to_end(node_id)
        while len(landed) > self._PEER_LANDED_MAX:
            landed.popitem(last=False)
        self.peer_demote_xstats.landing_commits += 1

    def _peer_demote_count_age(
        self, node: UnifiedTreeNode, evicted: bool, pages: int, meta
    ) -> None:
        """SGLANG_PEER_KV_DEMOTE_COUNTERS, resolver rank: the flag age by how
        the flag ended, and one trace record per flagged eviction."""
        if meta is None:
            return
        flag_time, rid = meta[0], meta[1]
        age = max(0.0, time.monotonic() - flag_time)
        xst = self.peer_demote_xstats
        (xst.age_evict if evicted else xst.age_clear)[age_bucket(age)] += 1
        sink = self.peer_demote_trace_sink
        if not evicted or sink is None:
            return
        keys = (
            peer_page_keys(
                [node.hash_value[0], node.hash_value[-1]],
                node.key.cache_salt,
                node.key.extra_key,
            )
            if node.hash_value and node.key is not None
            else [None, None]
        )
        try:
            sink(
                {
                    "event": "demoted_evict",
                    "ts": time.time(),
                    "rid": rid,
                    "node": node.id,
                    "first_key": keys[0],
                    "last_key": keys[-1],
                    "pages": pages,
                    "age_s": round(age, 3),
                }
            )
            xst.trace_records += 1
        except Exception:  # a trace write must never break eviction
            logger.warning("PeerKV demote trace write failed", exc_info=True)
            self.peer_demote_trace_sink = None

    def _peer_demote_count_return(
        self, keys: list[str], from_prefetch: bool, node_id: Optional[NodeId] = None
    ) -> None:
        """Resolver: a node was published again after this worker evicted a
        demoted copy that ended at one of its pages. With COUNTERS (an entry
        with its mark) also by trigger, time from the mark and mark size: a
        re-pull, the same node backed up again (rebackup) or a recompute."""
        recent = self._peer_demote_recent
        nodes = pages = 0
        for key in keys:
            n = recent.pop(key, None)
            if n is None:
                continue
            if isinstance(n, tuple):
                n, trig, mark_t, _evict_t, old_id, mark_tokens = n
                kind = (
                    "repull"
                    if from_prefetch
                    else ("rebackup" if old_id == node_id else "recompute")
                )
                since = time.monotonic() - mark_t
                self.peer_demote_xstats.bump(
                    f"{trig}.{return_bin(kind, since, mark_tokens)}_pages", n
                )
            nodes += 1
            pages += n
        if not nodes:
            return
        st = self.peer_demote_stats
        if from_prefetch:
            st.repull_events += 1
            st.repull_nodes += nodes
            st.repull_pages += pages
        else:
            st.recompute_nodes += nodes
            st.recompute_pages += pages

    # ------------------------------------------------------------------
    # SGLANG_PEER_KV_TOUCH_DEFERRED_PULLS (mem_cache/peer_kv_touch.py). The
    # resolver rank turns queued TOUCHes into node ids before the request
    # broadcast; every rank applies the broadcast ids at the same point, so
    # the access times never depend on rank-local state.
    # ------------------------------------------------------------------

    def enable_peer_touch(self, resolver: bool) -> None:
        self.peer_touch_enabled = True
        self.peer_touch_resolver = bool(resolver)

    def select_peer_touch_nodes(self, touches) -> list[NodeId]:
        """Resolver rank only. ``touches``: [(confirmed page keys, from the
        first page)]. Returns the ids of the deepest nodes to refresh."""
        st = self.peer_touch_stats
        out: list[NodeId] = []
        seen: set[NodeId] = set()
        for keys, from_root in touches:
            st.touches += 1
            node, via_root = resolve_touch_node(
                keys, from_root, self._peer_demote_index, self.root_node
            )
            if node is None or node is self.root_node:
                st.unresolved += 1
                continue
            st.resolved += 1
            st.from_root += int(via_root)
            if node.id not in seen:
                seen.add(node.id)
                out.append(node.id)
        st.nodes += len(out)
        return out

    def apply_peer_touch(
        self,
        node_ids: Sequence[NodeId],
        tick: Optional[float] = None,
        reserved: int = 0,
    ) -> int:
        """Every rank, from the broadcast PeerKVTouchReq: refresh each node and
        its ancestors as a prefix match does (access time; a demote flag is
        cleared as by a local access). ``tick`` and ``reserved``: the access
        ticks [tick, tick + reserved) that the drain reserved (the same on every
        rank). The nodes take them in order, then the counter moves past them,
        so a LOOKUP after the drain (its lookup tick is at or after their end)
        sees the refresh as earlier. If the counter already passed ``tick`` (an
        access between the drain and this point; the scheduler loop has none),
        or with no tick, each node takes a fresh tick."""
        if not self.peer_touch_enabled:
            return 0
        reserved_ok = tick is not None and peek_time_counter() <= tick
        st = self.peer_touch_stats
        applied = 0
        for node_id in node_ids:
            node = self._node_arena.get(node_id)
            if node is None or node is self.root_node or node.deleted:
                continue
            cur_time = (
                tick + applied if reserved_ok else get_and_increase_time_counter()
            )
            while node:
                node.last_access_time = cur_time
                if node.peer_demoted:
                    # Intentional: the TOUCH of a
                    # pending pull clears a flag that an EARLIER completed pull
                    # set on this path. A flagged node is evicted first,
                    # whatever its access time, so a refresh that kept the flag
                    # would not protect the pending pull (the miss that F1
                    # fixes). The TOUCH comes before the LOOKUP of its own pull
                    # (peer_kv_touch: order with the pull), so the complete
                    # RELEASE of that pull flags the path again. The policy
                    # stays: demote after every completed transfer, no source
                    # refresh after a transfer. A pull that ends without a
                    # transfer leaves the path unflagged: no transfer, so no
                    # refresh after one.
                    self._peer_demote_clear(node, evicted=False)
                    st.cleared_demote += 1
                cur_time -= 0.00001
                node = node.parent
            applied += 1
        if reserved_ok:
            # Past the reserved ticks (at most one per drained TOUCH, >= the
            # nodes applied), the same on every rank.
            while peek_time_counter() < tick + max(int(reserved), applied):
                get_and_increase_time_counter()
        st.applied += applied
        return applied

    def peer_demote_outstanding(self) -> tuple[int, int]:
        """(flagged nodes, their host pages) now. Read-only: the flag set is
        part of the structural summary, so only tree events change it."""
        pages = 0
        for node in self._peer_demoted.values():
            pages += self._peer_demote_host_pages(node)
        return len(self._peer_demoted), pages

    def set_component_device_value(
        self, node_id: NodeId, component_type: ComponentType, value: torch.Tensor
    ) -> None:
        """Store an auxiliary component's device value onto a node."""
        # Full uses leaf sets, not LRU; its stores go through the insert paths.
        assert component_type != BASE_COMPONENT_TYPE
        node = self.node_by_id(node_id)
        swa = (
            self.components_by_type[component_type]
            if component_type == ComponentType.SWA and self._swa_device_window()
            else None
        )
        blocked_before = swa._blocked_len(node) if swa is not None else 0
        node.component_data[component_type].value = value
        host_lru = self.host_lru_lists[component_type]
        if host_lru.in_list(node):
            host_lru.remove_node(node)
        self.lru_lists[component_type].insert_mru(node)
        self.component_evictable_size_[component_type] += len(value)
        if swa is not None:
            swa._add_blocked(node, blocked_before)

    def get_component_device_value(
        self, node_id: NodeId, component_type: ComponentType
    ) -> Optional[torch.Tensor]:
        """The component's device value on the node, or None if evicted."""
        return self.node_by_id(node_id).component_data[component_type].value

    def component_has_host_value_only(
        self, node_id: NodeId, component_type: ComponentType
    ) -> bool:
        """Whether the component's data is device-evicted but host-backed."""
        cd = self.node_by_id(node_id).component_data[component_type]
        return cd.value is None and cd.host_value is not None

    # ==== Others ====

    def sanity_check(
        self,
        ongoing_write_through: list[tuple[int, NodeId]],
        ongoing_load_back: list[tuple[int, NodeId]],
    ) -> None:
        """Verify tree-structure, leaf-set, LRU, size, and ongoing-op invariants; raise
        AssertionError on any violation. ongoing_* args are (id, node_id) pairs.
        """
        errors: list[str] = []
        E = errors.append
        all_nodes = self._collect_all_nodes()
        all_node_set = set(all_nodes)
        FCT = BASE_COMPONENT_TYPE

        # ── PART 1: Tree Structure ──
        # Root state
        if self.root_node.component_data[FCT].value is None:
            E("[Root] root missing Full device value")
        if self.root_node.component_data[FCT].lock_ref <= 0:
            E(
                f"[Root] root Full lock_ref={self.root_node.component_data[FCT].lock_ref}"
            )
        if self.root_node.parent is not None:
            E("[Root] root has a parent pointer")
        # Parent ↔ child bidirectional consistency
        for node in all_nodes:
            for child in node.children.values():
                if child.parent is not node:
                    pid = child.parent.id if child.parent else None
                    E(f"[Tree] child {child.id} parent={pid}, expected {node.id}")
                if child.key is None:
                    E(f"[Tree] node {child.id} has no key")

        # ── PART 2: Per-node state machine and leaf qualification ──
        expected_dev_leaves: set[UnifiedTreeNode] = set()
        expected_hst_leaves: set[UnifiedTreeNode] = set()
        expected_duplicates: set[UnifiedTreeNode] = set()
        window = self._swa_device_window()

        for node in all_nodes:
            if node is self.root_node:
                continue
            nid = node.id
            full_dev = node.component_data[FCT].value is not None
            full_hst = node.component_data[FCT].host_value is not None

            # Full is the tree backbone, so aux data requires Full data.
            for ct in self.component_types:
                if ct == FCT:
                    continue
                cd = node.component_data[ct]
                if cd.value is not None and not full_dev:
                    E(f"node {nid} {ct} device present but Full.value=None")
                if cd.host_value is not None and not full_hst:
                    # write_back reclaim takes only the Full host layer; an
                    # aux host slice may outlive it while Full device is live.
                    # Window residency may attach SWA host before the FULL
                    # backup (relaxed ancestry, I4).
                    if not (
                        (self.is_write_back or (window and ct == ComponentType.SWA))
                        and full_dev
                    ):
                        E(f"node {nid} {ct} host present but Full.host_value=None")

            # Window residency I1: SWA of full node length on some tier.
            if window and (full_dev or full_hst):
                swa_cd = node.component_data[ComponentType.SWA]
                n = len(node.key)
                if not any(
                    v is not None and len(v) == n
                    for v in (swa_cd.value, swa_cd.host_value)
                ):
                    E(f"node {nid} has FULL but no full-length SWA on any tier")

            # Every node must keep Full data on at least one layer.
            if not full_dev and not full_hst:
                E(f"node {nid} dead: no Full device and no Full host")

            # Parent prefixes must keep data whenever the child does.
            if node.parent is not None and node.parent is not self.root_node:
                p_dev = node.parent.component_data[FCT].value is not None
                p_hst = node.parent.component_data[FCT].host_value is not None
                if full_dev and not p_dev:
                    E(f"node {nid} device present but parent {node.parent.id} evicted")
                if full_hst and not p_hst and not self.is_write_back:
                    E(f"node {nid} backed up but parent {node.parent.id} not backed up")

            # Lock hierarchy and counters must stay sane.
            fl = node.component_data[FCT].lock_ref
            for ct in self.component_types:
                cd = node.component_data[ct]
                if cd.lock_ref < 0:
                    E(f"node {nid} {ct} lock_ref={cd.lock_ref}")
                if cd.host_lock_ref < 0:
                    E(f"node {nid} {ct} host_lock_ref={cd.host_lock_ref}")
                if ct != FCT and fl < cd.lock_ref:
                    E(f"node {nid} full_lock={fl} < {ct}_lock={cd.lock_ref}")
                if cd.value is None and cd.lock_ref > 0:
                    E(f"node {nid} {ct} evicted but lock_ref={cd.lock_ref}")

            # Collect expected leaf qualification (single pass)
            if self._is_device_leaf(node):
                expected_dev_leaves.add(node)
            if self._is_host_leaf(node):
                expected_hst_leaves.add(node)
            if self._is_settled_full_host_duplicate(node):
                expected_duplicates.add(node)

        # ── PART 3: Tracking structures ──

        # Device leaf set must match the expected leaves.
        if self.evictable_device_leaves != expected_dev_leaves:
            extra = self.evictable_device_leaves - expected_dev_leaves
            missing = expected_dev_leaves - self.evictable_device_leaves
            if extra:
                E(f"D-leaf extra: {[n.id for n in list(extra)[:5]]}")
            if missing:
                E(f"D-leaf missing: {[n.id for n in list(missing)[:5]]}")

        # Host leaf set must match the expected leaves.
        if self.evictable_host_leaves != expected_hst_leaves:
            extra = self.evictable_host_leaves - expected_hst_leaves
            missing = expected_hst_leaves - self.evictable_host_leaves
            if extra:
                E(f"H-leaf extra: {[n.id for n in list(extra)[:5]]}")
            if missing:
                E(f"H-leaf missing: {[n.id for n in list(missing)[:5]]}")

        # Lazy deregistration: stale extras are legal; settled duplicates must
        # be tracked and entries must not outlive their node.
        expected_ids = {n.id for n in expected_duplicates}
        dup_ids = set(self.full_host_duplicates.keys())
        if expected_ids - dup_ids:
            E(f"Duplicate missing: {list(expected_ids - dup_ids)[:5]}")
        ghost_ids = dup_ids - {n.id for n in all_nodes}
        if ghost_ids:
            E(f"Duplicate ghosts: {list(ghost_ids)[:5]}")

        # D-leaf ∩ H-leaf = ∅
        overlap = self.evictable_device_leaves & self.evictable_host_leaves
        if overlap:
            E(
                f"[Leaf] {len(overlap)} in both sets: {[n.id for n in list(overlap)[:5]]}"
            )

        if self.enable_session_radix_cache:
            for component in self.components:
                component.validate_session_state(all_node_set, E)

        # Stale nodes: leaf sets must only contain tree-reachable nodes
        stale = self.evictable_device_leaves - all_node_set
        if stale:
            E(
                f"{len(stale)} stale nodes in device_leaves: {[n.id for n in list(stale)[:5]]}"
            )
        stale = self.evictable_host_leaves - all_node_set
        if stale:
            E(
                f"{len(stale)} stale nodes in host_leaves: {[n.id for n in list(stale)[:5]]}"
            )

        # Per-component LRU tracking
        for ct in self.component_types:
            lru = self.lru_lists[ct]
            if ct == FCT:
                # Full uses leaf sets, not LRU
                if len(lru.cache) > 0:
                    E(f"Full device LRU not empty: {len(lru.cache)}")
                if len(self.host_lru_lists[ct].cache) > 0:
                    E(f"Full host LRU not empty: {len(self.host_lru_lists[ct].cache)}")
            else:
                # Aux device values must match the device LRU.
                tree_ids = {
                    n.id
                    for n in all_nodes
                    if n is not self.root_node
                    and n.component_data[ct].value is not None
                }
                lru_ids = set(lru.cache.keys())
                if tree_ids != lru_ids:
                    E(
                        f"{ct} device LRU: "
                        f"+tree={tree_ids - lru_ids}, +lru={lru_ids - tree_ids}"
                    )
                # Aux host-only states must match the host LRU.
                host_lru = self.host_lru_lists[ct]
                s3_ids = {
                    n.id
                    for n in all_nodes
                    if n is not self.root_node
                    and n.component_data[ct].value is None
                    and n.component_data[ct].host_value is not None
                }
                host_lru_ids = set(host_lru.cache.keys())
                if s3_ids != host_lru_ids:
                    E(
                        f"{ct} host LRU: "
                        f"+S3={s3_ids - host_lru_ids}, +lru={host_lru_ids - s3_ids}"
                    )
                # The same aux node must not appear in both device and host LRU.
                inv5_overlap = lru_ids & host_lru_ids
                if inv5_overlap:
                    E(f"{ct} in both device and host LRU: {inv5_overlap}")
                # Linked-list integrity
                self._check_lru_linked_list(lru, ct, "device", errors)
                self._check_lru_linked_list(host_lru, ct, "host", errors)

        # ── PART 4: Size Accounting ──
        for ct in self.component_types:
            evictable = 0
            protected = 0
            for n in all_nodes:
                if n is self.root_node:
                    continue
                cd = n.component_data[ct]
                if cd.value is not None:
                    toks = len(cd.value)
                    if cd.lock_ref > 0:
                        protected += toks
                    else:
                        evictable += toks
            if self.component_evictable_size_[ct] != evictable:
                E(
                    f"[Size] {ct} evictable={self.component_evictable_size_[ct]} "
                    f"!= recomputed={evictable}"
                )
            if self.component_protected_size_[ct] != protected:
                E(
                    f"[Size] {ct} protected={self.component_protected_size_[ct]} "
                    f"!= recomputed={protected}"
                )
        if window:
            blocked = self._compute_swa_blocked_size()
            if self.swa_blocked_size_ != blocked:
                E(
                    f"[Size] SWA blocked={self.swa_blocked_size_} "
                    f"!= recomputed={blocked}"
                )

        # ── PART 5: Ongoing Operations ──
        for nid, node_id in ongoing_write_through:
            n = self._node_arena.get(node_id)
            if n is None or n not in all_node_set:
                E(f"[Ongoing] write_through node {nid} not in tree")
            elif n.component_data[FCT].lock_ref <= 0:
                E(
                    f"[Ongoing] write_through node {nid} lock_ref={n.component_data[FCT].lock_ref}"
                )
        for nid, node_id in ongoing_load_back:
            n = self._node_arena.get(node_id)
            if n is None or n not in all_node_set:
                E(f"[Ongoing] load_back node {nid} not in tree")
            elif n.component_data[FCT].lock_ref <= 0:
                E(
                    f"[Ongoing] load_back node {nid} lock_ref={n.component_data[FCT].lock_ref}"
                )
        # Every in-flight H->D mark must belong to a live load-back; a leaked
        # mark would pin the node's host copy against reclaim forever.
        ongoing_load_ids = {node_id for _, node_id in ongoing_load_back}
        for node in all_nodes:
            if (
                node.load_back_pending_id is not None
                and node.load_back_pending_id not in ongoing_load_ids
            ):
                E(
                    f"[Ongoing] node {node.id} load_back_pending_id="
                    f"{node.load_back_pending_id} has no live load-back"
                )

        if errors:
            msg = (
                f"Sanity check FAILED ({len(errors)} violations "
                f"across {len(all_nodes)} nodes):\n"
                + "\n".join(f"  {e}" for e in errors)
            )
            logger.error(msg)
            self.pretty_print()
            raise AssertionError(msg)

    def _collect_all_nodes(self) -> list[UnifiedTreeNode]:
        nodes = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children.values())
        return nodes

    def _check_lru_linked_list(
        self,
        lru: UnifiedLRUList,
        ct: ComponentType,
        label: str,
        errors: list[str],
    ) -> None:
        """Walk a LRU doubly-linked list, collect integrity errors."""
        pt = lru._pt  # use LRU's own pointer slot
        visited: set[int] = set()
        x = lru.head.lru_next[pt]
        prev = lru.head
        while x is not None and x != lru.tail:
            if x.lru_prev[pt] != prev:
                errors.append(f"[{label}][{ct}] broken prev at node {x.id}")
            if x is lru.mid or x is lru.cursor:
                prev = x
                x = x.lru_next[pt]
                continue
            if x.id not in lru.cache:
                errors.append(f"[{label}][{ct}] node {x.id} in list not cache")
            if x.id in visited:
                errors.append(f"[{label}][{ct}] cycle at node {x.id}")
                break
            visited.add(x.id)
            prev = x
            x = x.lru_next[pt]
        if x is None:
            errors.append(
                f"[{label}][{ct}] broken chain: lru_next is None "
                f"after node {prev.id if hasattr(prev, 'id') else 'head'}"
            )
        if len(visited) != len(lru.cache):
            errors.append(
                f"[{label}][{ct}] list={len(visited)} != cache={len(lru.cache)}"
            )

    def pretty_print(self) -> None:
        stack = [(self.root_node, 0)]
        while stack:
            node, indent = stack.pop()
            component_str = " ".join(
                f"{ct}={'yes' if node.component_data[ct].value is not None else 'no'}"
                for ct in self.component_types
            )
            print(
                " " * indent,
                f"[{node.id}]",
                len(node.key),
                f"full_lock={node.component_data[BASE_COMPONENT_TYPE].lock_ref}",
                component_str,
            )
            for child in node.children.values():
                stack.append((child, indent + 2))

    def evictable_size(self) -> int:
        return self.component_evictable_size_.get(BASE_COMPONENT_TYPE, 0)

    def protected_size(self) -> int:
        return self.component_protected_size_.get(BASE_COMPONENT_TYPE, 0)

    def component_evictable_size(self, component_type: ComponentType) -> int:
        """Evictable token count for one component (0 if the component is absent)."""
        return self.component_evictable_size_.get(component_type, 0)

    def full_evictable_size(self) -> int:
        return self.evictable_size()

    def full_protected_size(self) -> int:
        return self.protected_size()

    def swa_evictable_size(self) -> int:
        return self.component_evictable_size_.get(ComponentType.SWA, 0)

    def mamba_evictable_size(self) -> int:
        return self.component_evictable_size_.get(ComponentType.MAMBA, 0)

    def swa_protected_size(self) -> int:
        return self.component_protected_size_.get(ComponentType.SWA, 0)

    def mamba_protected_size(self) -> int:
        return self.component_protected_size_.get(ComponentType.MAMBA, 0)

    def structural_summary(self) -> list[int]:
        """O(1) rank-uniform structural quantities, folded into the TP
        digest by UnifiedRadixCache.tp_structural_check. Must
        not include anything rank-local (servicer state, slot identities,
        wall clock)."""
        vals = [
            len(self._node_arena),
            len(self.evictable_host_leaves),
            len(self.evictable_device_leaves),
            len(self.full_host_duplicates),
            self.host_release_digest,
            self.write_back_duplicate_reclaim_digest,
            len(self._peer_demoted),
        ]
        if self.peer_demote_counters:
            # Which nodes are flagged, not only how many (every rank).
            vals.append(self._peer_demoted_id_digest)
        for ct in self.component_types:
            vals.append(self.component_evictable_size_[ct])
            vals.append(self.component_protected_size_[ct])
            vals.append(len(self.lru_lists[ct].cache))
            vals.append(len(self.host_lru_lists[ct].cache))
        return vals

    def total_size(self) -> tuple[int, int]:
        total_size = 0
        total_aux_size = 0
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            full_value = node.component_data[BASE_COMPONENT_TYPE].value
            if full_value is not None:
                total_size += len(full_value)
            for ct in self.component_types:
                if ct == BASE_COMPONENT_TYPE:
                    continue
                value = node.component_data[ct].value
                if value is not None:
                    total_aux_size += len(value)
            for child in node.children.values():
                stack.append(child)
        return total_size, total_aux_size

    def walk_for_kv_canary(
        self, unlocked_only: bool, swa_resident_only: bool
    ) -> RadixCacheWalkResult:
        """Flatten every FULL device slot into (slot, position, prev-slot) rows for the KV-canary sweep."""
        slots: list[int] = []
        positions: list[int] = []
        prev_slots: list[int] = []
        swa_filter = swa_resident_only and ComponentType.SWA in self.components_by_type

        def _dfs(node: UnifiedTreeNode, depth: int, parent_last_slot: int) -> None:
            value = node.component_data[BASE_COMPONENT_TYPE].value
            node_slots = value.tolist() if isinstance(value, torch.Tensor) else []

            emit = node is not self.root_node
            if unlocked_only:
                # Unified SWA owns an independent component lock. A node can still
                # hold Full KV for a running request while its SWA slots are unused.
                lock_ct = ComponentType.SWA if swa_filter else BASE_COMPONENT_TYPE
                emit = emit and node.component_data[lock_ct].lock_ref == 0
            if swa_filter:
                emit = emit and node.component_data[ComponentType.SWA].value is not None

            # Skipped nodes still advance the chain/depth so descendants stay consistent.
            chain_last_slot = parent_last_slot
            for j, slot in enumerate(node_slots):
                if emit:
                    slots.append(slot)
                    positions.append(depth + j)
                    prev_slots.append(parent_last_slot if j == 0 else node_slots[j - 1])
                chain_last_slot = slot

            # Device-evicted nodes hold no slots but still span their key length.
            if node is self.root_node or node.key is None:
                child_depth = depth + len(node_slots)
            else:
                child_depth = depth + len(node.key)
            for child in node.children.values():
                _dfs(child, child_depth, chain_last_slot)

        _dfs(self.root_node, 0, -1)
        return RadixCacheWalkResult(
            slot_indices=torch.tensor(slots, dtype=torch.int64),
            positions=torch.tensor(positions, dtype=torch.int64),
            prev_slot_indices=torch.tensor(prev_slots, dtype=torch.int64),
        )

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs(node: UnifiedTreeNode):
            for child in node.children.values():
                v = child.component_data[BASE_COMPONENT_TYPE].value
                if v is not None:
                    values.append(v)
                _dfs(child)

        _dfs(self.root_node)
        if values:
            return torch.cat(values)
        return torch.tensor([], dtype=torch.int64, device=self.device)

    def _all_component_values_flatten(
        self, component_type: ComponentType
    ) -> torch.Tensor:
        if component_type not in self.components_by_type:
            return torch.tensor([], dtype=torch.int64, device=self.device)

        values = []

        def _dfs(node: UnifiedTreeNode):
            value = node.component_data[component_type].value
            if value is not None:
                values.append(value)
            for child in node.children.values():
                _dfs(child)

        _dfs(self.root_node)
        if values:
            return torch.cat(values)
        return torch.tensor([], dtype=torch.int64, device=self.device)

    def all_mamba_values_flatten(self) -> torch.Tensor:
        return self._all_component_values_flatten(ComponentType.MAMBA)
