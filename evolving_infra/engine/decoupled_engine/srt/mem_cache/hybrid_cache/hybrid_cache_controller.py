# Adapted from SGLang python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py (commit 9a489f8d2f), modified for evolving-infra.
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import replace
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import torch

from decoupled_engine.srt.disaggregation.prefill_loop_unserialize import (
    fixed_shape_free_enabled,
)
from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.managers.cache_controller import (
    CacheOperation,
)
from decoupled_engine.srt.managers.cache_controller import (
    HiCacheController as BaseHiCacheController,
)
from decoupled_engine.srt.managers.cache_controller import (
    LayerDoneCounter,
    PrefetchAck,
)
from decoupled_engine.srt.managers.cache_controller import (
    StorageOperation as BaseStorageOperation,
)
from decoupled_engine.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
    count_pool_hits,
)
from decoupled_engine.srt.mem_cache.l2_transfer import L2Transfer
from decoupled_engine.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from decoupled_engine.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

if TYPE_CHECKING:
    from decoupled_engine.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

logger = logging.getLogger(__name__)


class StorageOperation(BaseStorageOperation):
    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        super().__init__(host_indices, token_ids, last_hash, hash_value, prefix_keys)
        self.pool_transfers = pool_transfers
        self.pool_storage_result = PoolTransferResult.empty()


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
        kv_pull_hint: Optional[dict] = None,
    ):
        self.request_id = request_id
        self.prefetch_attempt_id = uuid.uuid4().hex
        # Trace-only operation context; the input suffix is unchanged by retries.
        self.endpoint_round = 0
        self.initial_query_pages = None
        self.requires_exact_endpoint = False
        self._lock = threading.Lock()
        self._terminated_flag = False
        self.storage_hit_count = 0
        self.start_time = time.monotonic()
        # Cross-engine KV pull hint (kv-p2p); opaque, surfaced to the storage
        # backend via HiCacheStorageExtraInfo.extra_info on exists/get calls.
        self.kv_pull_hint = kv_pull_hint
        # Capture the request's trusted tree namespace before token slicing.
        # Never infer it from the router's (optional) peer hint.
        self.peer_namespace = {
            "cache_salt": getattr(token_ids, "cache_salt", None),
            "extra_key": getattr(token_ids, "extra_key", None),
        }
        super().__init__(
            None,
            token_ids,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=pool_transfers,
        )
        self.pool_transfers_done = not bool(pool_transfers)

    def storage_extra_info(self) -> Optional[dict]:
        if self.kv_pull_hint is None:
            return None
        # req_id rides along so the peer_kv fetch trace can attribute every
        # lookup/get op to the request (observability only).
        return {
            "kv_pull_hint": self.kv_pull_hint,
            "req_id": self.request_id,
            "peer_namespace": self.peer_namespace,
            "prefetch_attempt_id": self.prefetch_attempt_id,
            "endpoint_round": self.endpoint_round,
            "initial_query_pages": self.initial_query_pages,
            # Process-local only; peer RPC messages copy named wire fields.
            "peer_is_cancelled": self.is_terminated,
        }

    def mark_terminate(self):
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        with self._lock:
            return self._terminated_flag


class HybridCacheController(BaseHiCacheController):
    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: Any,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
        attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
        pp_group: Optional[torch.distributed.ProcessGroup] = None,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        transfer_layer_num: Optional[int] = None,
        enable_storage_metrics: bool = False,
        host_memory_mode: str = "cache",
    ):
        startup_storage_backend = storage_backend
        self.extra_host_mem_release_queues: dict[PoolName, Queue[torch.Tensor]] = {}
        super().__init__(
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            mem_pool_host=mem_pool_host,
            page_size=page_size,
            tp_group=tp_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=pp_group,
            write_policy=write_policy,
            io_backend=io_backend,
            storage_backend=None,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
            host_memory_mode=host_memory_mode,
        )
        # Override layer_num: hybrid models transfer all layers (For example, Linear Model (KV + Mamba)),
        # not just the full attention layers reported by full_kv_pool.
        if transfer_layer_num is not None and transfer_layer_num != self.layer_num:
            self.layer_num = transfer_layer_num
            self.layer_done_counter = LayerDoneCounter(self.layer_num)

        self.storage_host_pool = mem_pool_host.anchor_entry.host_pool
        if startup_storage_backend is not None:
            self.attach_storage_backend(
                storage_backend=startup_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=model_name,
                storage_backend_extra_config=storage_backend_extra_config,
                host_pools=getattr(mem_pool_host, "entries", None),
            )

    def _start_storage_threads(self):
        super()._start_storage_threads()
        self._init_extra_host_mem_release_queues()

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        host_pools: Optional[list[PoolEntry]] = None,
    ):
        super().attach_storage_backend(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
        )

        for entry in host_pools or []:
            if not getattr(entry.host_pool, "_is_dummy", False):
                self.storage_backend.register_mem_host_pool_v2(
                    entry.host_pool, entry.name
                )

    def register_host_pool_entry(self, entry: PoolEntry) -> None:
        if not isinstance(self.mem_pool_host, HostPoolGroup):
            raise TypeError("Dynamic HiCache sidecars require HostPoolGroup.")
        self.mem_pool_host.add_entry(entry)
        if not entry.is_primary_index_anchor:
            self.extra_host_mem_release_queues.setdefault(entry.name, Queue())
        if self.enable_storage and self.storage_backend is not None:
            if not getattr(entry.host_pool, "_is_dummy", False):
                self.storage_backend.register_mem_host_pool_v2(
                    entry.host_pool, entry.name
                )

    @staticmethod
    def parse_storage_backend_extra_config(
        storage_backend_extra_config: Optional[str],
    ) -> tuple[dict, int, float, float, bool]:
        extra_config = {}
        if storage_backend_extra_config:
            if storage_backend_extra_config.startswith("@"):
                path = storage_backend_extra_config[1:]
                ext = os.path.splitext(path)[1].lower()
                with open(path, "rb" if ext == ".toml" else "r") as f:
                    if ext == ".json":
                        extra_config = json.load(f)
                    elif ext == ".toml":
                        import tomllib

                        extra_config = tomllib.load(f)
                    elif ext in (".yaml", ".yml"):
                        import yaml

                        extra_config = yaml.safe_load(f)
                    else:
                        raise ValueError(
                            f"Unsupported config file {path} (config format: {ext})"
                        )
            else:
                extra_config = json.loads(storage_backend_extra_config)

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                "prefetch_timeout_per_ki_token must be number, got "
                f"{type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )

    def clear_storage_backend(self) -> bool:
        if not self.enable_storage:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False
        if not hasattr(self.storage_backend, "clear"):
            logger.warning(
                "Storage backend %s does not support clear operation.",
                type(self.storage_backend).__name__,
            )
            return False
        result = self.storage_backend.clear()
        if result is None:
            return True
        if isinstance(result, bool):
            return result
        logger.error(
            "Storage backend %s returned an invalid clear result: %r",
            type(self.storage_backend).__name__,
            result,
        )
        return False

    def _init_extra_host_mem_release_queues(self) -> None:
        self.extra_host_mem_release_queues = {}
        entries = getattr(self.mem_pool_host, "entries", None) or []
        anchor_entry = getattr(self.mem_pool_host, "anchor_entry", None)
        for entry in entries:
            if entry is anchor_entry or entry.is_primary_index_anchor:
                continue
            self.extra_host_mem_release_queues[entry.name] = Queue()

    def _append_host_mem_release_pages(
        self, release_queue: Queue, host_indices: torch.Tensor, page_size: int
    ) -> None:
        if host_indices.numel() == 0:
            return
        for page in host_indices.split(page_size):
            release_queue.put(page)

    def append_host_mem_release(
        self,
        host_indices: Optional[torch.Tensor] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ):
        if host_indices is not None:
            self._append_host_mem_release_pages(
                self.host_mem_release_queue,
                host_indices,
                self.mem_pool_host.page_size,
            )
        for transfer in extra_pools or []:
            if transfer.host_indices is None or transfer.host_indices.numel() == 0:
                continue
            entry = self.mem_pool_host.entry_map.get(transfer.name)
            if (
                entry is None
                or entry.is_primary_index_anchor
                or transfer.indices_from_pool is not None
            ):
                continue
            release_queue = self.extra_host_mem_release_queues.get(transfer.name)
            if release_queue is None:
                continue
            self._append_host_mem_release_pages(
                release_queue, transfer.host_indices, entry.host_pool.page_size
            )

    def reset(self):
        super().reset()
        if self.enable_storage:
            self.host_mem_release_queue.queue.clear()
            for release_queue in self.extra_host_mem_release_queues.values():
                release_queue.queue.clear()
            self.prefetch_tokens_occupied = 0

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        # Q4: CPU token slots from the tag (no index D2H); unchanged when off.
        device_indices = self._q4_transfer_indices(device_indices)
        pool_transfers = self.mem_pool_host.resolve_host_transfers(
            extra_pools,
            primary_device_indices=device_indices,
            primary_host_indices=host_indices,
        )
        if pool_transfers is None and extra_pools:
            self.mem_pool_host.free(host_indices)
            return None

        self.write_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=pool_transfers or None,
                d2h_sm_copy=envs.SGLANG_HICACHE_WRITE_BACK_SM_COPY.get(),
            )
        )
        self.start_writing()
        return host_indices

    def write_aux(
        self,
        extra_pools: list[PoolTransfer],
        ack_id: int,
        d2h_sm_copy: bool = False,
    ) -> Optional[list[PoolTransfer]]:
        """Back up component pools with no KV-anchor transfer (the DSv4 window
        SWA spill). The copy is acked in ack_write_queue under ``ack_id`` like a
        write-through; returns the resolved transfers, or None when a side
        pool's host allocation fails. ``d2h_sm_copy``: copy with SM kernels
        where every pool supports it (CacheOperation.d2h_sm_copy)."""
        empty_host = torch.empty((0,), dtype=torch.int64)
        empty_device = torch.empty((0,), dtype=torch.int64, device=self.device)
        pool_transfers = self.mem_pool_host.resolve_host_transfers(
            extra_pools,
            primary_device_indices=empty_device,
            primary_host_indices=empty_host,
        )
        if pool_transfers is None:
            return None
        self.write_queue.append(
            CacheOperation(
                empty_host,
                empty_device,
                ack_id,
                pool_transfers=pool_transfers,
                d2h_sm_copy=d2h_sm_copy,
            )
        )
        self.start_writing()
        return pool_transfers

    def _move_op_indices(
        self, op: CacheOperation
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list[PoolTransfer]]]:
        return self.move_hybrid_indices(op)

    def _d2h_sm_copy_supported(self, op: CacheOperation) -> bool:
        """Every host pool this write touches has an SM-kernel backup."""
        if self.io_backend != "kernel":
            return False
        host_group = self.mem_pool_host
        pools = []
        if op.host_indices.numel() > 0:
            pools.append(host_group.anchor_entry.host_pool)
        for transfer in op.pool_transfers or []:
            entry = host_group.entry_map.get(transfer.name)
            if entry is None:
                return False
            pools.append(entry.host_pool)
        return all(getattr(pool, "supports_sm_copy_backup", False) for pool in pools)

    def _write_host_pools(self, op: CacheOperation) -> list:
        """The host pools a write op touches; [] when one is unknown."""
        host_group = self.mem_pool_host
        pools = []
        if op.host_indices.numel() > 0:
            pools.append(host_group.anchor_entry.host_pool)
        for transfer in op.pool_transfers or []:
            entry = host_group.entry_map.get(transfer.name)
            if entry is None:
                return []
            pools.append(entry.host_pool)
        return pools

    def _move_write_operation(
        self, op: CacheOperation
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list[PoolTransfer]]]:
        host_group = self.mem_pool_host
        # Flag off (anchor pool flag False): not evaluated, the old path as is.
        anchor_entry = getattr(host_group, "anchor_entry", None)
        anchor_pool = getattr(anchor_entry, "host_pool", None)
        if getattr(
            anchor_pool, "pfd_kernel_backup", False
        ) and self._direct_layout_kernel_backup(op, self._write_host_pools(op)):
            op.d2h_sm_copy = False
            return op.host_indices, op.device_indices, op.pool_transfers
        if getattr(op, "d2h_sm_copy", False):
            if self._d2h_sm_copy_supported(op):
                # SM kernels read the host indices on the device.
                return self.move_hybrid_indices(op)
            op.d2h_sm_copy = False
        if self.io_backend != "kernel" or host_group.layout != "page_first":
            return self.move_hybrid_indices(op)
        if not getattr(host_group, "supports_per_pool_backup_indices", False):
            if not getattr(host_group, "can_use_write_back_jit", False):
                return self.move_hybrid_indices(op)
            return op.host_indices, op.device_indices, op.pool_transfers

        def move_for_pool(host_pool, host_indices, device_indices):
            if getattr(host_pool, "can_use_write_back_jit", False):
                if host_indices.is_cuda:
                    host_indices = host_indices.cpu()
                return host_indices, device_indices
            return self.move_indices(host_indices, device_indices)

        host_indices, device_indices = move_for_pool(
            host_group.anchor_entry.host_pool,
            op.host_indices,
            op.device_indices,
        )
        pool_transfers = []
        for transfer in op.pool_transfers or []:
            entry = host_group.entry_map[transfer.name]
            transfer_host_indices, transfer_device_indices = move_for_pool(
                entry.host_pool,
                transfer.host_indices,
                transfer.device_indices,
            )
            pool_transfers.append(
                replace(
                    transfer,
                    host_indices=transfer_host_indices,
                    device_indices=transfer_device_indices,
                )
            )
        return host_indices, device_indices, pool_transfers or None

    def _l2_transfers(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ) -> list[L2Transfer]:
        anchor = self.mem_pool_host.anchor_entry
        transfers = []
        if host_indices.numel() > 0:
            transfers.append(
                L2Transfer(
                    host_pool=anchor.host_pool,
                    device_pool=anchor.device_pool,
                    host_indices=host_indices,
                    device_indices=device_indices,
                    layer_mapper=anchor.layer_mapper,
                )
            )
        for pool_transfer in pool_transfers or []:
            if (
                pool_transfer.host_indices is None
                or pool_transfer.device_indices is None
            ):
                raise ValueError(f"Unresolved L2 transfer for {pool_transfer.name}.")
            entry = self.mem_pool_host.entry_map[pool_transfer.name]
            transfers.append(
                L2Transfer(
                    host_pool=entry.host_pool,
                    device_pool=entry.device_pool,
                    host_indices=pool_transfer.host_indices,
                    device_indices=pool_transfer.device_indices,
                    layer_mapper=entry.layer_mapper,
                )
            )
        return transfers

    def _l2_load_transfers(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ) -> list[L2Transfer]:
        transfers = self._l2_transfers(host_indices, device_indices, pool_transfers)
        transfers_by_entry = {
            (id(t.host_pool), id(t.device_pool)): t for t in transfers
        }
        for entry in self.mem_pool_host.entry_map.values():
            target_transfer = transfers_by_entry.get(
                (id(entry.host_pool), id(entry.device_pool))
            )
            if target_transfer is None or target_transfer.layer_mapper is None:
                continue
            for depth, draft_device_pool in enumerate(entry.packed_draft_device_pools):
                draft_host_layer = target_transfer.layer_mapper(self.layer_num + depth)
                if draft_host_layer is None:
                    continue

                def draft_layer_mapper(
                    layer_id: int,
                    *,
                    expected_layer_id: int = depth,
                    host_layer_id: int = draft_host_layer,
                ) -> Optional[int]:
                    if layer_id == expected_layer_id:
                        return host_layer_id
                    return None

                transfers.append(
                    L2Transfer(
                        host_pool=target_transfer.host_pool,
                        device_pool=draft_device_pool,
                        host_indices=target_transfer.host_indices,
                        device_indices=target_transfer.device_indices,
                        layer_mapper=draft_layer_mapper,
                        is_draft=True,
                    )
                )
        return transfers

    def _num_tokens_by_pool(self, op: CacheOperation) -> dict[str, int]:
        """Per-pool token counts for a merged transfer op (anchor + extra
        pools), shared by D->H write and H->D load acks; sidecar transfers
        reusing another pool's indices are excluded."""
        counts = {self.mem_pool_host.anchor_entry.name.value: len(op.device_indices)}
        for transfer in op.pool_transfers or []:
            if transfer.indices_from_pool is not None or transfer.host_indices is None:
                continue
            name = transfer.name.value
            counts[name] = counts.get(name, 0) + len(transfer.host_indices)
        return counts

    def _transfer_num_bytes(self, op: CacheOperation) -> int:
        """Total bytes moved by a merged transfer op across all pools.

        Sidecar transfers riding another pool's indices are included here but
        excluded from the per-pool token counts.
        """
        kv_tokens = len(op.device_indices)
        num_bytes = kv_tokens * self.mem_pool_host.anchor_entry.host_pool.size_per_token
        # Slot counts of the pools sidecars can ride on.
        source_len = {self.mem_pool_host.anchor_entry.name: kv_tokens}
        for t in op.pool_transfers or []:
            if t.indices_from_pool is None and t.host_indices is not None:
                source_len[t.name] = len(t.host_indices)
        for t in op.pool_transfers or []:
            entry = self.mem_pool_host.entry_map.get(t.name)
            if entry is None:
                continue
            if t.indices_from_pool is not None:
                num_slots = source_len.get(t.indices_from_pool, 0)
            else:
                num_slots = len(t.host_indices) if t.host_indices is not None else 0
            num_bytes += num_slots * entry.host_pool.size_per_token
        return num_bytes

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        need_load_kv = host_indices.numel() > 0

        full_allocator = getattr(
            self.mem_pool_device_allocator,
            "full_attn_allocator",
            self.mem_pool_device_allocator,
        )
        allocation = None
        if not need_load_kv:
            device_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            from decoupled_engine.srt.mem_cache.allocator.cpu_paged import cpu_paged_allocator

            if cpu_paged_allocator(full_allocator) is not None:
                # Q4: CPU page ids; clean pages first under the M4 quarantine.
                allocation = full_allocator.alloc_load(len(host_indices))
                device_indices = None if allocation is None else allocation.indices
            else:
                device_indices = full_allocator.alloc(len(host_indices))
            if device_indices is None:
                return None
        # The transfers address the destination by CPU token slots under Q4.
        op_device_indices = (
            allocation.indices_cpu if allocation is not None else device_indices
        )

        pool_transfers = self._resolve_device_transfers(
            extra_pools,
            kv_device_indices=op_device_indices,
            kv_host_indices=host_indices,
        )
        if pool_transfers is None and extra_pools:
            if need_load_kv:
                if fixed_shape_free_enabled():
                    # alloc() returned whole pages in page-major order: no
                    # torch.unique (a device sync on the admission path).
                    full_allocator.free_segment(device_indices, start_pos=0)
                else:
                    full_allocator.free(device_indices)
            return None

        op = CacheOperation(
            host_indices,
            op_device_indices,
            node_id,
            priority,
            pool_transfers=pool_transfers or None,
        )
        if allocation is not None:
            op.set_load_allocation(allocation)
        self.load_queue.append(op)
        return device_indices

    def prefetch(
        self,
        request_id: str,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
        kv_pull_hint: Optional[dict] = None,
    ) -> PrefetchOperation:
        operation = PrefetchOperation(
            request_id,
            new_input_tokens,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
            kv_pull_hint=kv_pull_hint,
        )
        self.prefetch_queue.put(operation)
        return operation

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> int:
        operation = StorageOperation(
            host_indices,
            token_ids,
            hash_value=hash_value,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.backup_queue.put(operation)
        return operation.id

    def _storage_hit_query(self, operation, max_pages=None) -> tuple[list[str], int]:
        hash_value = self.get_hash_str(
            operation.token_ids, operation.last_hash, page_size=self.page_size
        )
        operation.all_hash_values = hash_value
        if getattr(operation, "initial_query_pages", None) is None:
            operation.initial_query_pages = len(hash_value)
        if max_pages is not None:
            hash_value = hash_value[:max_pages]

        if getattr(self, "host_dedup", None) is not None and self.host_dedup.is_dummy:
            # The source holds the only lease and host payload. Followers provide
            # an upper bound; endpoint consensus selects the source's exact hit.
            return hash_value, len(hash_value) * self.page_size

        extra_info = HiCacheStorageExtraInfo(
            prefix_keys=(
                operation.prefix_keys.copy() if operation.prefix_keys else None
            ),
            extra_info=operation.storage_extra_info(),
        )
        if operation.pool_transfers:
            hit_result = self.storage_backend.batch_exists_v2(
                hash_value, operation.pool_transfers, extra_info
            )
        else:
            kv_hit_count = self.storage_backend.batch_exists(hash_value, extra_info)
            hit_result = PoolTransferResult(
                kv_hit_pages=kv_hit_count, extra_pool_hit_pages={}
            )

        kv_hit_pages = hit_result.kv_hit_pages
        if self._prefetch_requires_exact_endpoint(operation):
            operation.pool_storage_result = PoolTransferResult.empty()
        operation.pool_storage_result.update_kv_hit_pages(kv_hit_pages)

        return (
            hash_value[:kv_hit_pages],
            kv_hit_pages * self.page_size,
        )

    def _prefetch_requires_exact_endpoint(self, operation) -> bool:
        return bool(
            getattr(self.storage_backend, "supports_complete_endpoints", False)
            and any(
                t.hit_policy == PoolHitPolicy.TRAILING_PAGES
                or (t.name == PoolName.SWA and t.hit_policy == PoolHitPolicy.ALL_PAGES)
                for t in operation.pool_transfers or []
            )
        )

    def _synchronize_storage_hit(self, operation, hash_value, storage_hit_count):
        if not self._prefetch_requires_exact_endpoint(operation):
            return super()._synchronize_storage_hit(
                operation, hash_value, storage_hit_count
            )
        operation.requires_exact_endpoint = True
        try:
            hashes = self.get_hash_str(
                operation.token_ids, operation.last_hash, page_size=self.page_size
            )
            contract = sorted(
                (
                    str(t.name),
                    str(t.hit_policy),
                    len(t.keys) if t.keys else 1,
                    str(t.indices_from_pool),
                )
                for t in operation.pool_transfers or []
            )
            encoded = json.dumps(
                [
                    operation.request_id,
                    operation.last_hash,
                    hashes,
                    contract,
                    getattr(operation.token_ids, "extra_key", None),
                    getattr(operation.token_ids, "cache_salt", None),
                ],
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
            digest = hashlib.sha256(encoded).digest()
            signature = [
                int.from_bytes(digest[i : i + 4], "big") for i in range(0, 16, 4)
            ]
        except Exception:
            logger.warning("peer endpoint contract failed", exc_info=True)
            signature = [0] * 4
            storage_hit_count = 0
        # Valid hybrid endpoints need not form a prefix. MIN is only a new
        # search upper bound; use an endpoint only when every rank pinned it.
        # Descending search is finite. Bound round trips under churn/disjoint
        # checkpoint sets and fall back to compute on exhaustion.
        for round_index in range(8):
            if operation.is_terminated():
                storage_hit_count = 0
            pages = storage_hit_count // self.page_size
            fields = [pages, -pages]
            if round_index == 0:
                fields += [v for part in signature for v in (part, -part)]
            endpoints = torch.tensor(fields, dtype=torch.int64)
            self._all_reduce(
                endpoints,
                torch.distributed.ReduceOp.MIN,
                self.prefetch_hits_sync_groups,
            )
            lower, upper = endpoints[0].item(), -endpoints[1].item()
            if lower == 0 or (
                round_index == 0
                and any(
                    endpoints[i].item() != -endpoints[i + 1].item()
                    for i in range(2, len(fields), 2)
                )
            ):
                operation.pool_storage_result = PoolTransferResult.empty()
                return [], 0
            if getattr(self, "host_dedup", None) is not None:
                # All followers queried the full logical span without IO, so MIN
                # is the source's exact endpoint. Signature/cancellation checks
                # above still apply on every rank; no follower lease is needed.
                operation.pool_storage_result = PoolTransferResult(lower, {})
                return hash_value[:lower], lower * self.page_size
            if lower == upper:
                operation.pool_storage_result = PoolTransferResult(lower, {})
                return hash_value[:lower], lower * self.page_size
            if round_index == 7:
                logger.warning(
                    "peer endpoint negotiation exhausted for req=%s",
                    operation.request_id,
                )
                operation.pool_storage_result = PoolTransferResult.empty()
                return [], 0
            try:
                released = self.storage_backend.release_for_endpoint_retry(
                    operation.request_id, operation.prefetch_attempt_id
                )
                if released and not operation.is_terminated():
                    operation.endpoint_round = round_index + 1
                    hash_value, storage_hit_count = self._storage_hit_query(
                        operation, max_pages=lower
                    )
                else:
                    hash_value, storage_hit_count = [], 0
            except Exception:
                # Rank-local failures cannot skip the next collective.
                logger.warning("peer endpoint retry failed", exc_info=True)
                hash_value, storage_hit_count = [], 0
        raise AssertionError("bounded endpoint negotiation did not return")

    def move_hybrid_indices(
        self, operation: CacheOperation
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list[PoolTransfer]]]:
        host_indices, device_indices = self.move_indices(
            operation.host_indices, operation.device_indices
        )
        resolved_pool_transfers = None
        if operation.pool_transfers:
            resolved_pool_transfers = []
            for transfer in operation.pool_transfers:
                transfer_host_indices, transfer_device_indices = self.move_indices(
                    transfer.host_indices, transfer.device_indices
                )
                # Keep the original PoolTransfer unchanged because tree-owned
                # transfers may still reference radix-tree host state. The
                # controller only needs a normalized execution-time copy.
                resolved_pool_transfers.append(
                    PoolTransfer(
                        name=transfer.name,
                        host_indices=transfer_host_indices,
                        device_indices=transfer_device_indices,
                        keys=transfer.keys,
                        hit_policy=transfer.hit_policy,
                        indices_from_pool=transfer.indices_from_pool,
                    )
                )
        return host_indices, device_indices, resolved_pool_transfers

    def _page_transfer(self, operation: PrefetchOperation) -> bool:
        # KV pools and KV-derived pools first — determines actual completed page count
        kv_completed_pages = super()._page_transfer(operation)

        # Read non-KV derived sidecar pool, e.g. SWA, Mamba.
        self._page_transfer_sidecar(operation, kv_completed_pages)

    def _page_transfer_sidecar(
        self, operation: PrefetchOperation, kv_completed_pages: int
    ) -> None:
        if operation.pool_transfers is None:
            return

        # Extra pools only after KV fully completes. If KV terminated early
        # (IO failure, timeout, TP mismatch), skip extra IO entirely to avoid
        # data misalignment.
        pool_hits: dict[str, int] = {}
        if not operation.is_terminated() and kv_completed_pages == len(
            operation.hash_value
        ):
            # KV-derived sidecar pools are handled in CacheController._page_transfer_kv_batch.
            # Only handle non-KV-derived sidecar pools here.
            transfers_nonkv = [
                transfer
                for transfer in operation.pool_transfers
                if transfer.indices_from_pool != PoolName.KV
            ]
            self._sync_trailing_keys(
                transfers_nonkv, operation.hash_value, kv_completed_pages
            )
            self._resolve_sidecar_nonkv_derived_pool_transfers(operation)
            # Same extra_info shape as the KV batches (request id +
            # kv_pull_hint) so a lease-based backend matches this trailing
            # read to the operation's own lease.
            extra_info = HiCacheStorageExtraInfo(
                prefix_keys=(
                    operation.prefix_keys.copy() if operation.prefix_keys else None
                ),
                extra_info=operation.storage_extra_info(),
            )
            if (
                getattr(self, "host_dedup", None) is not None
                and self.host_dedup.is_dummy
            ):
                pool_hits = {str(t.name): len(t.keys or []) for t in transfers_nonkv}
            else:
                results = self.storage_backend.batch_get_v2(transfers_nonkv, extra_info)
                pool_hits = count_pool_hits(results)
        # Emit PrefetchAck to prefetch_sync_queue, even the operation has been canceled by the
        # scheduler thread.  The prefetch sync thread expects the same number of PrefetchAck objects
        # to perform all_reduce.
        self.prefetch_sync_queue.put(
            PrefetchAck(
                rid=operation.request_id,
                operation=operation,
                pool_hits=pool_hits,
            )
        )
        return

    def _page_backup(self, operation):
        # MLA KV is replicated across TP ranks and should still be written only
        # by TP0. Rank-sharded sidecars still need every TP rank.
        backup_transfers = [
            transfer
            for transfer in operation.pool_transfers or []
            if self.should_backup(transfer)
        ]

        if backup_transfers:
            self._resolve_sidecar_kv_derived_pool_transfers(operation)
            self._resolve_sidecar_nonkv_derived_pool_transfers(operation)
            results = self.storage_backend.batch_set_v2(backup_transfers)
            pool_hits = count_pool_hits(results)
            operation.pool_storage_result.update_extra_pool_hit_pages(pool_hits)

        if not self.backup_skip:
            super()._page_backup(operation)
        else:
            sidecar_ok = bool(backup_transfers)
            if sidecar_ok:
                for transfer in backup_transfers:
                    result = results.get(transfer.name)
                    if result is None:
                        result = results.get(transfer.name.value)
                    expected = len(transfer.keys or [])
                    if expected == 0 and transfer.host_indices is not None:
                        expected = int(transfer.host_indices.numel())
                    if (
                        not isinstance(result, (list, tuple))
                        or len(result) != expected
                        or not all(bool(ok) for ok in result)
                    ):
                        sidecar_ok = False
                        break
            operation.completed_tokens = (
                len(operation.hash_value) * self.page_size if sidecar_ok else 0
            )

    def should_backup(self, transfer: PoolTransfer) -> bool:
        if not self.backup_skip:
            return True

        # Kimi-K3 Mamba/KDA state is TP-sharded even when the primary MLA KV
        # pool is replicated.
        if transfer.name == PoolName.MAMBA:
            return True

        # Mooncake gives MHA draft and draft-SWA objects rank-specific keys.
        # MLA/DeepSeek-V4 draft pools remain TP0-only.
        if self.storage_backend_type == "mooncake" and transfer.name in (
            PoolName.DRAFT,
            PoolName.DRAFT_SWA,
        ):
            entry = self.mem_pool_host.entry_map.get(transfer.name)
            return entry is not None and isinstance(
                entry.host_pool, MHATokenToKVPoolHost
            )

        return False

    def backup_thread_func(self):
        """Back up rank-sharded sidecars on every TP rank.

        The base implementation skips the entire operation on non-zero MLA TP
        ranks. That optimization is valid for replicated MLA KV, but not for
        hybrid rank-sharded pools such as Kimi-K3 Mamba state.
        """
        backup_queue = self.backup_queue
        while not self.storage_stop_event.is_set():
            try:
                operation = backup_queue.get(block=True, timeout=1)
            except Empty:
                continue
            try:
                if operation is None:
                    continue
                self._page_backup(operation)
                self.ack_backup_queue.put(operation)
            finally:
                backup_queue.task_done()

    def _resolve_sidecar_kv_derived_pool_transfers(self, operation):
        for transfer in operation.pool_transfers:
            if transfer.indices_from_pool == PoolName.KV:
                transfer.host_indices = operation.host_indices
                if transfer.keys is None:
                    transfer.keys = operation.hash_value

    def _resolve_sidecar_nonkv_derived_pool_transfers(self, operation):
        for transfer in operation.pool_transfers:
            if transfer.indices_from_pool is None:
                continue
            if transfer.indices_from_pool != PoolName.KV:
                source = next(
                    (
                        t
                        for t in operation.pool_transfers
                        if t.indices_from_pool is None
                        and t.name == transfer.indices_from_pool
                    ),
                    None,
                )
                if source is None:
                    raise AssertionError(
                        "Storage sidecar derived pool source missing: "
                        f"{transfer.name} from {transfer.indices_from_pool}."
                    )
                transfer.host_indices = source.host_indices
                if transfer.keys is None:
                    transfer.keys = source.keys
            else:
                pass

    def _sync_trailing_keys(
        self,
        pool_transfers: list[PoolTransfer],
        all_hashes: list[str],
        kv_hit_pages: int,
    ) -> None:
        """Re-align trailing-page sidecar keys after KV hit truncation.

        When the storage hit is shorter than the original target prefix, each
        pool transfer's keys must be updated to the last N hashes of the actual
        hit range instead of the last N hashes of the original target range.
        For mamba (N=1) this is just the last hit page hash; for SWA (N>1) it
        is a sliding window of the last N hit pages.
        """
        for transfer in pool_transfers:
            full_swa = transfer.hit_policy == PoolHitPolicy.ALL_PAGES and (
                transfer.name == PoolName.SWA
                or transfer.indices_from_pool == PoolName.SWA
            )
            if transfer.hit_policy != PoolHitPolicy.TRAILING_PAGES and not full_swa:
                continue
            trailing_n = len(transfer.keys) if transfer.keys else 1
            transfer.keys = all_hashes[
                (0 if full_swa else max(0, kv_hit_pages - trailing_n)) : kv_hit_pages
            ]
            if full_swa and transfer.indices_from_pool is not None:
                # Derived sidecars share the SWA allocation: only its anchor
                # may return unused slots. Resolve sidecar views afterwards.
                continue
            if transfer.host_indices is None:
                continue
            entry = self.mem_pool_host.entry_map.get(transfer.name)
            pool_page_size = (
                self.page_size
                if transfer.name == PoolName.SWA
                else (
                    entry.host_pool.page_size if entry is not None else self.page_size
                )
            )
            needed = len(transfer.keys) * pool_page_size
            if transfer.host_indices.numel() > needed:
                # The hit undershot the pre-allocated window buffer. Backends
                # fetch keys zipped against the buffer head, so shrink the
                # transfer to match and release the tail now — otherwise the
                # length mismatch makes batch_get_v2 fetch nothing and the
                # whole window is silently lost downstream.
                self.append_host_mem_release(
                    extra_pools=[
                        PoolTransfer(
                            name=transfer.name,
                            host_indices=transfer.host_indices[needed:],
                        )
                    ]
                )
                transfer.host_indices = transfer.host_indices[:needed]

    def _resolve_device_transfers(
        self,
        extra_pools: Optional[list[PoolTransfer]],
        kv_device_indices: Optional[torch.Tensor] = None,
        kv_host_indices: Optional[torch.Tensor] = None,
    ) -> Optional[list[PoolTransfer]]:
        """Allocate unresolved side-pool device indices atomically."""
        if not extra_pools:
            return None
        newly_allocated: list[tuple[PoolTransfer, Callable, torch.Tensor]] = []
        derived_transfers: list[PoolTransfer] = []

        def rollback_allocated() -> None:
            for prev_pool, prev_free_fn, prev_indices in newly_allocated:
                prev_free_fn(prev_indices)
                prev_pool.device_indices = None

        for pool in extra_pools:
            if pool.indices_from_pool is not None:
                derived_transfers.append(pool)
                continue
            entry = self.mem_pool_host.entry_map.get(pool.name)
            if entry is None:
                continue
            if pool.device_indices is not None or pool.host_indices is None:
                continue
            # device_alloc_fn / device_free_fn override entry.device_pool's
            # methods for pools whose device_pool is a raw KV pool (layout)
            # rather than an allocator (e.g. SWA).
            alloc_fn = entry.device_alloc_fn or entry.device_pool.alloc
            free_fn = entry.device_free_fn or entry.device_pool.free
            evict_fn = entry.device_evict_fn
            size = len(pool.host_indices)
            indices = alloc_fn(size)
            if indices is None and evict_fn:
                evict_fn(size)
                indices = alloc_fn(size)
            if indices is None:
                # Atomic rollback: free everything we successfully allocated.
                rollback_allocated()
                return None
            pool.device_indices = indices
            newly_allocated.append((pool, free_fn, indices))

        # Assign indices to deferred pools from their source.
        for pool in derived_transfers:
            if pool.indices_from_pool == PoolName.KV:
                pool.host_indices = kv_host_indices
                pool.device_indices = kv_device_indices
                continue

            source = next(
                (
                    transfer
                    for transfer in extra_pools
                    if transfer.indices_from_pool is None
                    and transfer.name == pool.indices_from_pool
                ),
                None,
            )
            if source is None:
                rollback_allocated()
                return None
            pool.host_indices = source.host_indices
            pool.device_indices = source.device_indices
        return extra_pools

    def _reduce_prefetch_ack(self, ack: PrefetchAck) -> None:
        # Handle KV-derived pool.
        super()._reduce_prefetch_ack(ack)

        # Handle other sidecar pools, e.g. SWA, Mamba.
        if ack.pool_hits is not None:
            # "for ... in PoolName" ensures the same order across all ranks.
            # On prefetch failure, pool_hits may be empty dict.
            packed = torch.tensor(
                [ack.pool_hits.get(pool.value, 0) for pool in PoolName],
                dtype=torch.int,
            )
            self._all_reduce(
                packed,
                torch.distributed.ReduceOp.MIN,
                self.prefetch_completion_sync_groups,
            )
            for i, pool in enumerate(PoolName):
                ack.pool_hits[pool.value] = packed[i].item()
