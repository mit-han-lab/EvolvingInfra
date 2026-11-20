# Adapted from SGLang python/sglang/srt/mem_cache/unified_cache/components/full.py (commit 297e54fd10), modified for evolving-infra.
from __future__ import annotations

import contextlib
import heapq
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import torch

from decoupled_engine.srt.managers.scheduler_components.phase_timers import PHASES
from decoupled_engine.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from decoupled_engine.srt.mem_cache.cpu_page_book import copy_tag, has_pages, slice_tagged
from decoupled_engine.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from decoupled_engine.srt.mem_cache.unified_cache.cache_action import (
    FreeComponentDeviceSlot,
    UpdatePendingBackupOnNodeSplit,
)
from decoupled_engine.srt.mem_cache.unified_cache.components.base import (
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
    TreeComponent,
    next_component_uuid,
)

if TYPE_CHECKING:
    from decoupled_engine.srt.managers.schedule_batch import Req
    from decoupled_engine.srt.mem_cache.radix_cache import RadixKey
    from decoupled_engine.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
    )
    from decoupled_engine.srt.mem_cache.unified_radix_cache import (
        NodeId,
        UnifiedTreeNode,
    )


class FullComponent(TreeComponent):
    component_type = ComponentType.FULL
    # M4 load quarantine clock (mem_cache/load_quarantine.py); None = off.
    _lq_clock = None

    def __init__(self, cache, params):
        super().__init__(cache, params)
        # HiCache state: set to host KV pool when HiCache enabled
        self._full_kv_pool_host = None
        # Lazy bind eviction strategy since tree core is initialized after component init.
        self.session_ref_eviction_strategy = (
            self._session_ref_eviction_strategy
            if cache.enable_session_radix_cache
            else None
        )

    def _ensure_eviction_strategy(self) -> None:
        if self.session_ref_eviction_strategy is None:
            self.session_ref_eviction_strategy = (
                self.tree_core.eviction_strategy.get_priority
            )

    def _dec_session_coverage(self, session_id: str, leaf: UnifiedTreeNode) -> None:
        node = leaf
        while node is not None and node is not self.tree_core.root_node:
            cd = node.component_data[self.component_type]
            assert cd.session_ref > 0
            cd.session_ref -= 1
            node = node.parent

    def _advance_session_coverage(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        old_ancestor: Optional[UnifiedTreeNode],
    ) -> None:
        stop = old_ancestor if old_ancestor is not None else self.tree_core.root_node
        node = leaf
        while (
            node is not None
            and node is not stop
            and node is not self.tree_core.root_node
        ):
            node.component_data[self.component_type].session_ref += 1
            node = node.parent

    def _recede_session_coverage(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        fallback: Optional[UnifiedTreeNode],
    ) -> None:
        stop = fallback if fallback is not None else self.tree_core.root_node
        node = leaf
        while (
            node is not None
            and node is not stop
            and node is not self.tree_core.root_node
        ):
            cd = node.component_data[self.component_type]
            assert cd.session_ref > 0
            cd.session_ref -= 1
            node = node.parent

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        if match_device_only:
            return lambda node: (
                node.component_data[self.component_type].value is not None
                and node.load_back_pending_id is None
            )

        # HiCache: evicted + backuped nodes are valid match boundaries.
        return lambda node: (
            node.component_data[self.component_type].value is not None or node.backuped
        )

    def finalize_match_result_in_tree_core(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        # Compute Full KV host hit length: walk from last_host_node up to
        # last_device_node, summing host_value lengths of evicted nodes.
        ct = self.component_type
        kv_host_hit = 0
        node = result.best_match_node
        root_node = self.tree_core.root_node
        while node is not result.last_device_node and node is not root_node:
            full_host = node.component_data[ct].host_value
            if full_host is not None:
                kv_host_hit += len(full_host)
            node = node.parent
        if kv_host_hit > 0:
            return result._replace(
                host_hit_length=max(result.host_hit_length, kv_host_hit)
            )
        return result

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        ct = self.component_type
        new_parent.component_data[ct].lock_ref = child.component_data[ct].lock_ref
        new_parent.component_data[ct].session_ref = child.component_data[ct].session_ref
        # Both fragments hold pages with the child's history.
        new_parent.component_data[ct].unlock_epoch = child.component_data[
            ct
        ].unlock_epoch
        child_cd = child.component_data[ct]
        new_parent.component_data[ct].host_lock_ref = child_cd.host_lock_ref
        # A Full host lock starts as a one-node segment. Splitting that node
        # extends the segment through the inserted prefix, so its boundary
        # moves to the prefix's older edge just like the SWA segment boundary.
        host_uuid = child_cd.metadata.pop("host_uuid", None)
        if host_uuid is not None:
            new_parent.component_data[ct].metadata["host_uuid"] = host_uuid
        assert new_parent.component_data[ct].session_ids is None
        split_len = len(new_parent.key)
        if child_cd.value is not None and has_pages(child_cd.value):
            # Q4: both fragments keep their CPU page ids (cpu_page_book.py).
            ps = self.tree_core.page_size
            head = slice_tagged(child_cd.value, 0, split_len, ps)
            tail = slice_tagged(child_cd.value, split_len, None, ps)
            new_parent.component_data[ct].value = copy_tag(head.clone(), head)
            child_cd.value = copy_tag(tail.clone(), tail)
        elif child_cd.value is not None:
            new_parent.component_data[ct].value = child_cd.value[:split_len].clone()
            child_cd.value = child_cd.value[split_len:].clone()
        if child_cd.host_value is not None:
            new_parent.component_data[ct].host_value = child_cd.host_value[
                :split_len
            ].clone()
            child_cd.host_value = child_cd.host_value[split_len:].clone()

    def evict_component(
        self,
        node: UnifiedTreeNode,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        cache_actions: list[CacheAction | ComponentAction],
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        cd = node.component_data[self.component_type]
        freed = 0
        host_freed = 0

        # Device layer
        if EvictLayer.DEVICE in target and cd.value is not None:
            if self._lq_clock is not None:
                # Released with the node's unlock epoch, not the eviction
                # time (apply_component_action reads it back).
                cd.value._lq_release_epoch = cd.unlock_epoch
            device_frees[self.component_type].append(cd.value)
            # Progress toward an eviction request and the ledger move are
            # both in allocator units (see reclaimable_value_len).
            freed = self.reclaimable_value_len(node)
            self.tree_core.component_evictable_size_[self.component_type] -= freed
            # NOTE: cd.value = None is deferred to _cascade_evict (Full as trigger)
            # because SWA's free_swa still needs to read Full.value.
            # cd.value = None

        # Host layer
        if EvictLayer.HOST in target and cd.host_value is not None:
            host_freed = len(cd.host_value)
            host_frees[self.component_type].append(cd.host_value)
            cd.host_value = None
        return freed, host_freed

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0 if is_leaf else 2

    def _session_ref_eviction_strategy(self, node: UnifiedTreeNode):
        ref = self.session_ref(node)
        return ref > 0, ref, self.tree_core.eviction_strategy.get_priority(node)

    def _evict_device_start(self, request_cnt: int) -> None:
        self._ensure_eviction_strategy()
        self._evict_device_request_cnt = request_cnt
        self._evict_device_last_node = None
        core = self.tree_core
        key = self.session_ref_eviction_strategy
        if getattr(core, "peer_demote_gpu_order", False) and core._peer_demoted:
            # GPU_ORDER: class 2 first, then the configured
            # order. Without a class-2 node the plain key (same victims).
            base = key

            def key(n):
                return (not n.peer_demoted, base(n))

        self._evict_device_key = key
        # (priority, node id, node): equal priorities pop in id order, the same
        # on every TP rank (the leaf set iterates in address order).
        self._evict_device_heap = [
            (key(n), n.id, n) for n in core.evictable_device_leaves
        ]
        heapq.heapify(self._evict_device_heap)
        if PHASES.enabled:  # SGLANG_SCHED_PHASE_TIMERS
            PHASES.count("evict_device_heap_leaves", len(self._evict_device_heap))

    def _evict_device_next_node(
        self,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        cache_actions: list[CacheAction | ComponentAction],
    ) -> Optional[NodeId]:
        ct = self.component_type
        lv = self._evict_device_last_node
        if (
            lv is not None
            and lv.parent is not None
            and lv.parent in self.tree_core.evictable_device_leaves
        ):
            key = getattr(self, "_evict_device_key", None)
            if key is None:
                key = self.session_ref_eviction_strategy
            heapq.heappush(
                self._evict_device_heap,
                (key(lv.parent), lv.parent.id, lv.parent),
            )
        self._evict_device_last_node = None
        while tracker[ct] < self._evict_device_request_cnt and self._evict_device_heap:
            _, _, x = heapq.heappop(self._evict_device_heap)
            if x not in self.tree_core.evictable_device_leaves:
                continue
            if x.deleted:
                self.tree_core._on_stale_leaf_set_update(x)
                continue
            self._evict_device_last_node = x
            return x.id
        return None

    def _evict_device_end(self) -> None:
        self._evict_device_heap = []
        self._evict_device_last_node = None
        self._evict_device_key = None

    def drive_host_eviction(
        self,
        num_tokens: int,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        """Evict host leaves to free KV host pool space."""
        self._ensure_eviction_strategy()
        # CPU recency (host_last_access_time) orders host eviction; GPU
        # priorities are unchanged.
        priority = self._host_eviction_strategy
        if self.tree_core.peer_demote_enabled:
            # --peer-kv-demote-source: demoted leaves strictly first, then the
            # configured order (the flags are identical on every TP rank).
            base = priority

            def priority(n):
                return (not n.peer_demoted, base(n))

        heap = [(priority(n), n.id, n) for n in self.tree_core.evictable_host_leaves]
        heapq.heapify(heap)
        if PHASES.enabled:  # SGLANG_SCHED_PHASE_TIMERS
            PHASES.count("evict_host_heap_leaves", len(heap))
        ct = self.component_type
        while tracker[ct] < num_tokens and heap:
            _, _, x = heapq.heappop(heap)
            if x not in self.tree_core.evictable_host_leaves:
                continue
            self.tree_core._evict_host_leaf(
                x, tracker, device_frees, host_frees, cache_actions=cache_actions
            )
            if (
                x.parent is not None
                and x.parent in self.tree_core.evictable_host_leaves
            ):
                heapq.heappush(heap, (priority(x.parent), x.parent.id, x.parent))

    def _host_eviction_strategy(self, node):
        """Use CPU recency for host LRU without changing native GPU priorities."""
        if self.tree_core.eviction_policy == "lru":
            if self.tree_core.enable_session_radix_cache:
                ref = self.session_ref(node)
                return ref > 0, ref, node.host_last_access_time
            return node.host_last_access_time
        return self.session_ref_eviction_strategy(node)

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        ct = self.component_type

        # Only the last host node needs to be protected.
        if lock_host:
            cd = node.component_data[ct]
            # write_back mode: the anchor may be device-only (no host_value); pin it anyway.
            if cd.host_value is None and not self.tree_core.is_write_back:
                return result
            if cd.metadata.get("host_uuid") is None:
                cd.metadata["host_uuid"] = next_component_uuid(ct)
            result.set_lock_uuid(ct, cd.metadata["host_uuid"], lock_host=True)
            cd.host_lock_ref += 1
            self.tree_core._update_evictable_leaf_sets(node)
            return result

        root = self.tree_core.root_node
        cur = node

        # The bottom device-evicted segment is locked too (no ledger move —
        # nothing is on device); a load-back that materializes a value under
        # lock credits protected directly.
        while cur is not root and cur.component_data[ct].value is None:
            cur.component_data[ct].lock_ref += 1
            cur = cur.parent

        # Lock the device-on segment up to root
        delta = 0
        while cur is not root:
            cd = cur.component_data[ct]
            assert cd.value is not None, (
                f"FULL invariant broken: evicted ancestor {cur.id} above device-on segment"
            )
            if cd.lock_ref == 0:
                key_len = self.reclaimable_value_len(cur)
                self.tree_core.component_evictable_size_[ct] -= key_len
                self.tree_core.component_protected_size_[ct] += key_len
                delta += key_len
            cd.lock_ref += 1
            self.tree_core.evictable_device_leaves.discard(cur)
            cur = cur.parent
        result.delta = delta
        return result

    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: DecLockRefParams,
        lock_host: bool = False,
    ) -> None:
        ct = self.component_type
        if lock_host:
            if ct not in params.component_host_lock_uuids:
                # This receipt did not acquire the Full host component.
                return
            boundary_uuid = params.get_lock_uuid(ct, lock_host=True)
            assert boundary_uuid is not None, "Full host lock receipt has no boundary"
            while True:
                cd = node.component_data[ct]
                if cd.host_value is None and not self.tree_core.is_write_back:
                    return
                assert cd.host_lock_ref > 0, (
                    f"Full host segment release hit host_lock_ref=0 on node {node.id}"
                )
                cd.host_lock_ref -= 1
                self.tree_core._update_evictable_leaf_sets(node)
                if cd.metadata.get("host_uuid") == boundary_uuid:
                    break
                assert node.parent is not None, (
                    f"Full host lock boundary {boundary_uuid} is not an ancestor "
                    f"of receipt anchor {params.node_id}"
                )
                node = node.parent
            return

        root = self.tree_core.root_node
        cur = node
        while cur != root:
            cd = cur.component_data[ct]
            assert cd.lock_ref > 0, (
                f"FULL segment release hit lock_ref=0 on node {cur.id}"
            )
            if cd.lock_ref == 1 and cd.value is not None:
                key_len = self.reclaimable_value_len(cur)
                self.tree_core.component_evictable_size_[ct] += key_len
                self.tree_core.component_protected_size_[ct] -= key_len
                if self._lq_clock is not None:
                    # The forwards that used the node through this lock are
                    # at most the pass's epoch (lockstep tag).
                    cd.unlock_epoch = max(cd.unlock_epoch, self._lq_clock.tag())
            cd.lock_ref -= 1
            if cd.lock_ref == 0:
                self.tree_core._update_evictable_leaf_sets(cur)
            cur = cur.parent

    # ---- HiCache Hooks ----

    def build_hicache_transfers(
        self,
        node: Optional[UnifiedTreeNode],
        phase: CacheTransferPhase,
        *,
        mamba_pool_idx: Optional[torch.Tensor] = None,
        host_indices: Optional[torch.Tensor] = None,
        token_ids: Optional[Sequence[int]] = None,
        prefetch_tokens: int = 0,
        staging_tokens: int = 0,
        last_hash: Optional[str] = None,
        # Buffer-mode PREFETCH describes prefix_key without a resident node.
        prefix_key: Optional[RadixKey] = None,
    ) -> Optional[list[PoolTransfer]]:
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            # Full KV backup is handled by the main flow
            # (cache_controller.write on host_value directly).
            # No extra PoolTransfer needed.
            return None

        if phase == CacheTransferPhase.LOAD_BACK:
            assert node is not None
            # `node` is best_match_node. FULL device evict only from leaves,
            # so once we hit a device-on node, everything above is also device-on
            backed_up: list[torch.Tensor] = []
            nodes: list = []
            cur = node
            while cur.evicted:
                cd = cur.component_data[ct]
                assert cd.host_value is not None
                backed_up.append(cd.host_value)
                nodes.append(cur)
                cur = cur.parent
            backed_up.reverse()
            nodes.reverse()
            return [
                PoolTransfer(
                    name=PoolName.KV,
                    host_indices=(
                        torch.cat(backed_up)
                        if backed_up
                        else torch.empty((0,), dtype=torch.int64, device="cpu")
                    ),
                    device_indices=None,
                    nodes_to_load=[n.id for n in nodes],
                )
            ]

        return None

    def commit_hicache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer] = (),
        *,
        cache_actions: list[CacheAction | ComponentAction],
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            if transfers and transfers[0].host_indices is not None:
                node.component_data[ct].host_value = transfers[0].host_indices.clone()

        elif phase == CacheTransferPhase.LOAD_BACK:
            if not transfers or transfers[0].device_indices is None:
                self.tree_core._update_evictable_leaf_sets(node)
                return

            xfer = transfers[0]
            device_indices = xfer.device_indices
            offset = 0
            for nid in xfer.nodes_to_load or []:
                n = self.tree_core.node_by_id(nid)
                cd = n.component_data[ct]
                n_len = len(cd.host_value)
                if has_pages(device_indices):
                    # Q4: the node keeps its CPU page ids (cpu_page_book.py).
                    piece = slice_tagged(
                        device_indices, offset, offset + n_len, self.tree_core.page_size
                    )
                    cd.value = copy_tag(piece.clone(), piece)
                else:
                    cd.value = device_indices[offset : offset + n_len].clone()
                if self._lq_clock is not None:
                    cd.unlock_epoch = max(cd.unlock_epoch, self._lq_clock.tag())
                offset += n_len
                # Full uses leaf sets, not LRU. A value materialized under
                # lock is protected; the last release moves it to evictable.
                ledger = self.reclaimable_value_len(n)
                if cd.lock_ref > 0:
                    self.tree_core.component_protected_size_[ct] += ledger
                else:
                    self.tree_core.component_evictable_size_[ct] += ledger
                self.tree_core._update_evictable_leaf_sets(n)

            self.tree_core._update_evictable_leaf_sets(node)

    def _full_allocator(self):
        """The allocator that owns the full-attention pool alone."""
        allocator = self.cache.token_to_kv_pool_allocator
        return allocator.full_attn_allocator if self.cache.is_swa_enabled else allocator

    def build_external_linker_transfer(
        self,
        phase: LinkerTransferPhase,
        node: Optional[UnifiedTreeNode],
        keys: Optional[Sequence[str]],
    ) -> Optional[PoolTransfer]:
        if phase == LinkerTransferPhase.OFFLOAD:
            if node is None or not node.hash_value:
                return None
            value = node.component_data[self.component_type].value
            if value is None:
                return None
            return PoolTransfer(
                name=PoolName.KV,
                device_indices=value.to(torch.int64),
                keys=list(node.hash_value),
            )

        if not keys:
            return None

        if phase == LinkerTransferPhase.LOOKUP:
            return PoolTransfer(name=PoolName.KV, keys=list(keys))

        if phase == LinkerTransferPhase.LOAD:
            allocator = self._full_allocator()
            num_tokens = len(keys) * self.cache.page_size
            shortfall = max(0, num_tokens - allocator.available_size())
            if shortfall:
                self.cache.evict(EvictParams(num_tokens=shortfall))
            slots = allocator.alloc(num_tokens)
            if slots is None:
                return None

            return PoolTransfer(
                name=PoolName.KV,
                device_indices=slots.to(torch.int64),
                keys=list(keys),
            )

    def update_external_linker_load(
        self,
        phase: ExternalLinkerLoadPhase,
        req: Req,
        full_transfer: PoolTransfer,
        transfer: PoolTransfer,
        prefix_len: int,
    ) -> Optional[PoolTransfer]:
        if phase == ExternalLinkerLoadPhase.ABORT:
            self._full_allocator().free(transfer.device_indices)
            return None
        return transfer

    def free_host_values(self, host_values: list[torch.Tensor]) -> None:
        if self._full_kv_pool_host is None:
            return
        for host_value in host_values:
            self.cache.host_pool_group.free(host_value, pool=PoolName.KV)

    def apply_component_action(self, action: ComponentAction) -> None:
        if isinstance(action, UpdatePendingBackupOnNodeSplit):
            # Custom FULL components track buffer-write dependencies.
            pass
        elif isinstance(action, FreeComponentDeviceSlot):
            alloc = self.cache.token_to_kv_pool_allocator
            clock = self._lq_clock
            for indices in action.indices:
                # tree values are page-aligned copies of a kv row: page-exact segments
                if self.cache.is_swa_enabled:
                    alloc.full_attn_allocator.free_segment(indices, start_pos=0)
                    continue
                # M4: an evicted node's pages carry its unlock epoch.
                release_at = (
                    clock.releasing_at(getattr(indices, "_lq_release_epoch", None))
                    if clock is not None
                    else contextlib.nullcontext()
                )
                with release_at:
                    # EI 6c-ev (fixed-shape free, SGLANG_PREFILL_FIXED_SHAPE_FREE)
                    # is upstream's unconditional free_segment now (#37876).
                    alloc.free_segment(indices, start_pos=0)
        else:
            raise AssertionError(
                f"FullComponent: unhandled ComponentAction {type(action).__name__}"
            )
