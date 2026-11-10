"""Scheduler-owned GPU export using the ordinary HiCache host tier.

All TP ranks execute the same transaction. Host space is reserved on every rank
before any copy is submitted. The regular backup ACK and peer lease machinery
provide DMA completion and RDMA lifetime protection; no private staging pointers
or independently expiring GPU references are exported.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch

from decoupled_engine.srt.managers.cache_controller import CacheOperation
from decoupled_engine.srt.mem_cache.hicache_storage import PoolName
from decoupled_engine.srt.mem_cache.peer_gpu_export import GPUExportBroker, export_signature
from decoupled_engine.srt.mem_cache.peer_kv_namespace import peer_page_keys
from decoupled_engine.srt.mem_cache.unified_cache.component_type import ComponentType
from decoupled_engine.srt.mem_cache.utils import compute_node_hash_values

logger = logging.getLogger(__name__)


@dataclass
class _Export:
    export_id: str
    hashes: list[str]
    state: str = "preparing"
    node: object = None
    gpu_lock: object = None
    host_locks: list = field(default_factory=list)
    pages: int = 0
    draining: bool = False
    started: float = field(default_factory=time.monotonic)


class GPUExportManager:
    def __init__(self, cache, *, group, max_tokens, timeout_s):
        self.cache = cache
        self.group = group
        self.ranks = (
            torch.distributed.get_process_group_ranks(group)
            if group is not None
            else [0]
        )
        self.rank = torch.distributed.get_rank(group) if group is not None else 0
        self.max_tokens = max_tokens
        self.broker = (
            GPUExportBroker(
                max_pages=max_tokens // cache.page_size, timeout_s=timeout_s
            )
            if self.rank == 0
            else None
        )
        self.active = None
        self._next_retry = 0.0

    def _vote(self, values):
        vote = torch.tensor(values, dtype=torch.int64, device="cpu")
        if len(self.ranks) > 1:
            torch.distributed.all_reduce(
                vote, op=torch.distributed.ReduceOp.MIN, group=self.group
            )
        return vote.tolist()

    def tick(self):
        # One short metadata broadcast per scheduler iteration when enabled;
        # it never depends on rank-local network arrival order.
        plan = None
        if self.rank == 0:
            plan = self.broker.plan({self.active.export_id} if self.active else set())
            now = time.monotonic()
            plan["retry"] = now >= self._next_retry
            if plan["retry"]:
                self._next_retry = now + 0.05
        box = [plan]
        if len(self.ranks) > 1:
            torch.distributed.broadcast_object_list(
                box, src=self.ranks[0], group=self.group
            )
        plan = box[0]
        if plan["start"] is not None:
            assert self.active is None
            self.active = _Export(**plan["start"])
        tx = self.active
        if tx is None:
            return
        tx.draining |= tx.export_id in plan["retire"]
        if tx.state == "copying":
            # Existing ACK processing precedes this tick. Splits retain their
            # pending IDs and locks, so inspect the current path, not stale IDs.
            ready = all(
                n.write_through_pending_id is None for n in self._chain(tx.node)
            )
            if not self._vote([int(ready)])[0]:
                return
            self._release_gpu(tx)
            tx.state = "ready"
            if self.broker:
                self.broker.update(tx.export_id, "ready", pages=tx.pages)
            logger.info(
                "peer_gpu_export ready id=%s pages=%d prepare_ms=%.3f",
                tx.export_id,
                tx.pages,
                (time.monotonic() - tx.started) * 1000,
            )
        if tx.draining:
            self._finish(tx)
        elif tx.state == "preparing" and plan["retry"]:
            self._prepare(tx)

    def _chain(self, node):
        nodes = []
        while node.parent is not None:
            nodes.append(node)
            node = node.parent
        return list(reversed(nodes))

    def _gpu_index(self):
        # Discovery runs only for a host miss. Index only complete GPU paths:
        # FULL-only matches, inflight loadbacks and a namespace from another
        # request cannot be prepared. No GPU tensor is read by the REP thread.
        core = self.cache.tree_core
        pages = {}
        stack = list(core.root_node.children.values())
        while stack:
            node = stack.pop()
            if node.load_back_pending_id is not None:
                continue
            if any(
                node.component_data[ct].value is None
                or len(node.component_data[ct].value) != len(node.key)
                for ct in (ComponentType.FULL, ComponentType.SWA)
            ):
                continue
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, core.page_size)
            keys = peer_page_keys(
                node.hash_value, node.key.cache_salt, node.key.extra_key
            )
            for offset, key in enumerate(keys, 1):
                pages[key] = (node, offset * core.page_size)
            stack.extend(node.children.values())
        return pages

    def _prepare(self, tx):
        core = self.cache.tree_core
        try:
            index = self._gpu_index()
            local_pages = next(
                (i for i, h in enumerate(tx.hashes) if h not in index), len(tx.hashes)
            )
        except Exception:
            logger.exception("peer_gpu_export discovery failed")
            index, local_pages = {}, -1
        pages = self._vote([local_pages])[0]
        if pages < 0:
            self._fail(tx, "discovery_error")
            return
        if not pages:
            self._fail(tx, "gpu_prefix_unavailable")
            return
        node, offset = index[tx.hashes[pages - 1]]
        path_keys = []
        for ancestor in self._chain(node):
            keys = peer_page_keys(
                ancestor.hash_value, ancestor.key.cache_salt, ancestor.key.extra_key
            )
            path_keys.extend(
                keys if ancestor is not node else keys[: offset // core.page_size]
            )
        valid = path_keys[-pages:] == tx.hashes[:pages]
        if not self._vote([int(valid)])[0]:
            self._fail(tx, "noncontiguous_prefix")
            return
        sig = export_signature(tx.hashes[:pages], node.id, offset)
        lo, neg_hi = self._vote([sig, -sig])
        if lo != -neg_hi:
            self._fail(tx, "tp_endpoint_disagreement")
            return
        if offset < len(node.key):
            node, action = core._split_node(node.key, node, offset)
            if action is not None:
                self.cache._apply_cache_actions([action])
        chain = self._chain(node)
        protected = sum(len(n.key) for n in chain)
        if protected > self.max_tokens:
            self._fail(tx, "source_protection_budget")
            return
        tx.node, tx.pages = node, pages
        tx.gpu_lock = core.inc_lock_ref(node.id).to_dec_params()
        # Protect existing host ancestors while reserving the missing suffix.
        for n in chain:
            if n.backuped:
                tx.host_locks.append((n, core.inc_host_lock_ref(n.id).to_dec_params()))
        # Do not enqueue more backup work on nodes with an outstanding write.
        if not self._vote(
            [int(all(n.write_through_pending_id is None for n in chain))]
        )[0]:
            self._release_locks(tx)
            return
        cc = self.cache.cache_controller
        specs = []
        valid_spec = True
        try:
            for n in chain:
                device, components = core.build_backup_spec(n.id)
                if device.numel() == 0 and not components:
                    continue
                pools = [p for ps in components.values() for p in ps]
                pools.extend(self.cache._build_backup_sidecar(device, components))
                specs.append((n, device, components, pools))
        except Exception:
            logger.exception("peer_gpu_export all-pool manifest failed")
            valid_spec = False
        manifest = [
            f"{n.id}:{len(device)}:"
            + ",".join(
                f"{p.name}:{len(p.device_indices) if p.device_indices is not None else -1}"
                for p in pools
            )
            for n, device, _, pools in specs
        ]
        sig = export_signature(manifest, node.id, protected)
        valid_all, lo, neg_hi = self._vote([int(valid_spec), sig, -sig])
        if not valid_all or lo != -neg_hi:
            self._fail(tx, "tp_backing_disagreement")
            return

        # Uniform FULL/SWA bundles are the only independent allocators in this
        # mode. Reclaim complete duplicates before reserving; DMA has not begun.
        needed = sum(len(device) for _, device, _, _ in specs)
        swa_needed = sum(
            len(p.device_indices)
            for _, _, _, pools in specs
            for p in pools
            if p.name == PoolName.SWA and p.indices_from_pool is None
        )
        deficit = max(
            0,
            needed - cc.mem_pool_host.available_size(),
            swa_needed - cc.mem_pool_host.get_pool(PoolName.SWA).available_size(),
        )
        deficit = -self._vote([-deficit])[0]
        if deficit:
            self.cache.evict_host(deficit)
        reserved = []
        success = True
        uncommitted_host = None
        try:
            for n, device, components, pools in specs:
                host = cc.mem_pool_host.alloc(len(device))
                if host is None:
                    success = False
                    break
                uncommitted_host = host
                resolved = cc.mem_pool_host.resolve_host_transfers(
                    pools,
                    primary_device_indices=device,
                    primary_host_indices=host,
                    allow_reclaim=False,
                )
                if pools and resolved is None:
                    cc.mem_pool_host.free(host)
                    uncommitted_host = None
                    success = False
                    break
                op = CacheOperation(host, device, n.id, pool_transfers=resolved or None)
                reserved.append((n, components, op))
                uncommitted_host = None
        except Exception:
            if uncommitted_host is not None:
                cc.mem_pool_host.free(uncommitted_host)
            logger.exception("peer_gpu_export reservation failed")
            success = False
        if not self._vote([int(success)])[0]:
            for _, _, op in reserved:
                cc.mem_pool_host.release_transfers(op.pool_transfers)
                cc.mem_pool_host.free(op.host_indices)
            self._release_locks(tx)
            if self.broker:
                self.broker.update(
                    tx.export_id, "preparing", reason="host_capacity_wait"
                )
            return
        # Commit and submit only after every rank owns the whole reservation.
        for n, components, op in reserved:
            core.commit_backup(n.id, op.host_indices, components)
            self.cache._track_write_through_node(n.id, None)
            tx.host_locks.append((n, core.inc_host_lock_ref(n.id).to_dec_params()))
            cc.write_queue.append(op)
        if reserved:
            cc.start_writing()
        else:
            for n in chain:
                core._peer_push_servable_node(n)
        tx.state = "copying"

    def _release_gpu(self, tx):
        if tx.gpu_lock is not None:
            self.cache.tree_core.dec_lock_ref(tx.node.id, tx.gpu_lock)
            tx.gpu_lock = None

    def _release_locks(self, tx):
        for node, params in reversed(tx.host_locks):
            self.cache.tree_core.dec_host_lock_ref(node.id, params)
        tx.host_locks.clear()
        self._release_gpu(tx)

    def _finish(self, tx):
        self._release_locks(tx)
        if self.broker:
            self.broker.forget(tx.export_id)
        self.active = None

    def _fail(self, tx, reason):
        self._release_locks(tx)
        if self.broker:
            self.broker.update(tx.export_id, "unavailable", reason=reason)
        logger.info("peer_gpu_export refused id=%s reason=%s", tx.export_id, reason)
        self.active = None
