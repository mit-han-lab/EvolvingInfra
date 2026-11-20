# Adapted from SGLang python/sglang/srt/mem_cache/memory_pool_host.py (commit 297e54fd10), modified for evolving-infra.
from __future__ import annotations

import logging
import threading
from typing import Optional

import torch

from decoupled_engine.kernels.ops.kvcache.hicache import (
    can_use_write_back_jit_kernel,
)
from decoupled_engine.kernels.ops.kvcache.hicache import (
    transfer_hicache_all_layer_mla_staged_lf_pf as jit_transfer_hicache_all_layer_mla_staged_lf_pf,
)
from decoupled_engine.kernels.ops.kvcache.hisparse import transfer_cache_dsv4_mla
from decoupled_engine.srt.mem_cache.pool_host._kvcacheio import (
    MLA_KVCACHEIO_OPS,
    require_kvcacheio_ops,
    unavailable_stub,
)
from decoupled_engine.srt.platforms import current_platform
from decoupled_engine.srt.utils import is_cuda, is_hip, is_mps, is_npu, is_xpu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_xpu = is_xpu()
_is_mps = is_mps()
_dsv4_mla_unavailable_reason: Optional[str] = None
if _is_xpu:
    try:
        from sgl_kernel import transfer_cache_dsv4_mla
    except ImportError as e:
        # Only DSV4 needs this op; a wheel without it must not break scheduler import.
        _dsv4_mla_unavailable_reason = (
            f"sgl_kernel.transfer_cache_dsv4_mla failed to import ({e}); XPU "
            "takes it from the out-of-tree sgl-kernel-xpu wheel"
        )
        transfer_cache_dsv4_mla = unavailable_stub(
            "transfer_cache_dsv4_mla", _dsv4_mla_unavailable_reason
        )
else:
    from decoupled_engine.kernels.ops.kvcache.hisparse import transfer_cache_dsv4_mla
if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import (
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_all_layer_mla,
        transfer_kv_all_layer_mla_lf_pf,
        transfer_kv_direct,
        transfer_kv_per_layer_direct_pf_lf,
        transfer_kv_per_layer_mla,
        transfer_kv_per_layer_mla_pf_lf,
    )
elif _is_xpu:
    from decoupled_engine.srt.mem_cache.pool_host._kvcacheio import (
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_all_layer_mla,
        transfer_kv_all_layer_mla_lf_pf,
        transfer_kv_direct,
        transfer_kv_per_layer_direct_pf_lf,
        transfer_kv_per_layer_mla,
        transfer_kv_per_layer_mla_pf_lf,
    )

if _is_npu:
    from sgl_kernel_npu.kvcacheio import TransferDirection, transfer_kv_dim_exchange

logger = logging.getLogger(__name__)


from decoupled_engine.srt.mem_cache.pool_host import HostKVCache
from decoupled_engine.srt.mem_cache.pool_host.base import (
    _WRITE_BACK_STAGING_PAGE_CHUNK,
    host_memory_budget_bytes,
    ranks_per_host,
    synchronized,
)
from decoupled_engine.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    HICACHE_LOAD_BLOCK_QUOTA,
    get_allocator_from_storage,
    make_kernel_ptr_table,
)
from decoupled_engine.srt.mem_cache.pool_host.hisparse import HiSparseHostPoolMixin
from decoupled_engine.srt.mem_cache.pool_host.slot_allocator import PageSlotAllocator

# ---- V4 Compressed KV Host Pools ----


def _init_dsv4_dedup_follower(pool, host_dedup) -> bool:
    pool.host_dedup = host_dedup
    pool._is_dummy = host_dedup is not None and host_dedup.is_dummy
    if not pool._is_dummy:
        return False
    pool.kv_buffer = None
    pool.data_refs = []
    pool.data_ptrs = None
    pool.device_ptrs = None
    pool.staging_buffer = None
    pool.can_use_jit = False
    pool.can_use_write_back_jit = False
    logger.info(
        "V4 host dedup: allocator-only pool '%s', layers=%d, pages=%d, host_bytes=0",
        pool.pool_name,
        pool.layer_num,
        pool.num_host_pages,
    )
    return True


def _dsv4_host_memory_budget(host_dedup) -> int:
    budget = host_memory_budget_bytes()
    # The source is the only process allocating these payloads on this host.
    # Retain the existing whole-machine reserve while removing the TP split.
    return budget if host_dedup is None else budget * ranks_per_host()


class LogicalHostPool:
    """Pure-logical anchor pool for V4 HiCache.

    The pool manages page-aligned token slots but holds no KV tensor. V4
    compressed side pools use these logical FULL indices as stable page anchors.
    """

    shared_allocation_domain = None
    stores_page_envelope = False
    storage_format_tag = None
    # It copies nothing, so any write may take the SM-copy path.
    supports_sm_copy_backup = True

    def __init__(
        self,
        size: int,
        page_size: int,
        layout: str = "layer_first",
        reserve_pages: int = 0,
    ):
        if size % page_size != 0:
            raise ValueError(
                "LogicalHostPool size must be page-aligned, "
                f"got size={size}, page_size={page_size}"
            )
        self.size = size
        # Stands in for a host pool (and group anchor); DCP never widens it.
        self.logical_size = size
        self.page_size = page_size
        self.device = "cpu"
        self.layout = layout
        self.dtype = torch.uint8
        self.layer_num = 0
        self.start_layer = 0
        self.end_layer = 0
        self.kv_buffer = None
        self.size_per_token = 0
        self.allocator = None
        self.can_use_write_back_jit = True
        self.lock = threading.RLock()
        # Free-list + kv-p2p slot pins live in the shared helper (see
        # pool_host/slot_allocator.py); reserve_pages is 0 unless the peer-KV
        # servicer is enabled, which keeps stock behaviour byte-for-byte.
        self._slots = PageSlotAllocator(
            size, page_size, reserve_pages=reserve_pages, lock=self.lock, name="kv"
        )

    # Free-list state is owned by the helper; these mirror the stock attributes.
    @property
    def free_slots(self) -> torch.Tensor:
        return self._slots.free_slots

    @free_slots.setter
    def free_slots(self, value: torch.Tensor) -> None:
        self._slots.free_slots = value

    @property
    def release_slots(self) -> list:
        return self._slots.release_slots

    @property
    def num_release_slots(self) -> int:
        return self._slots.num_release_slots

    @property
    def slot_allocator(self) -> PageSlotAllocator:
        return self._slots

    @synchronized
    def clear(self):
        self._slots.clear()

    def destroy(self) -> None:
        """Logical anchors own no backing buffers or registrations to release."""
        return None

    @synchronized
    def available_size(self):
        return self._slots.available_size()

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size % self.page_size != 0:
            raise ValueError(
                "LogicalHostPool allocation must be page-aligned, "
                f"got need_size={need_size}, page_size={self.page_size}"
            )
        return self._slots.alloc(need_size)

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        if len(indices) % self.page_size != 0:
            raise ValueError(
                "LogicalHostPool free must be page-aligned, "
                f"got len(indices)={len(indices)}, page_size={self.page_size}"
            )
        return self._slots.free(indices)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend, sm_copy=False
    ):
        pass

    def prepare_transfer_indices(self, host_indices, device_indices, io_backend):
        return host_indices, device_indices

    def backup_from_device_all_layer_physical(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        pass

    def load_to_device_per_layer_physical(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        pass

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        pass

    def get_data_page(self, index, flat=True):
        return torch.empty(0, dtype=torch.uint8)

    def get_dummy_flat_data_page(self):
        return torch.empty(0, dtype=torch.uint8)

    def set_from_flat_data_page(self, index, data_page):
        pass

    def get_page_buffer_meta(self, indices):
        return None

    def get_page_buffer_element_size(self, split_factor: int = 1):
        return None

    def get_ksize_per_token(self):
        return 0


class DSV4TailLayout:
    """Tail-only host layout of a DSV4 SWA page (SGLANG_DSV4_WINDOW_HOST_TAIL_ONLY).

    A device SWA page is ``device_items_per_page`` items of ``item_bytes``
    (one item is one token's nope+rope bytes; the page's scale block and pad
    follow its tokens). The host keeps items ``[start_item, device_items_per_page)``
    of every page: the last tokens' data and the whole scale block, one
    contiguous byte range. Host item ``h * tail_items + j`` holds device item
    ``r * device_items_per_page + start_item + j`` of the page row pair (h, r).
    """

    def __init__(self, *, device_page_bytes: int, item_bytes: int, start_item: int):
        if item_bytes <= 0 or item_bytes % 16 != 0:
            raise ValueError(
                f"tail item size {item_bytes} must be a positive multiple of 16"
            )
        if device_page_bytes % item_bytes != 0:
            raise ValueError(
                f"device page of {device_page_bytes} B is not a whole number of "
                f"{item_bytes}-B items"
            )
        self.item_bytes = item_bytes
        self.device_items_per_page = device_page_bytes // item_bytes
        if not 0 < start_item < self.device_items_per_page:
            raise ValueError(
                f"tail start item {start_item} outside (0, {self.device_items_per_page})"
            )
        self.start_item = start_item
        self.tail_items = self.device_items_per_page - start_item
        self.page_bytes = self.tail_items * item_bytes
        self.device_start_byte = start_item * item_bytes
        self._offsets: dict = {}

    def _arange(self, device) -> torch.Tensor:
        key = str(device)
        offsets = self._offsets.get(key)
        if offsets is None:
            offsets = torch.arange(self.tail_items, dtype=torch.int64, device=device)
            self._offsets[key] = offsets
        return offsets

    def host_items(self, host_rows: torch.Tensor) -> torch.Tensor:
        rows = host_rows.to(torch.int64).unsqueeze(1)
        return (rows * self.tail_items + self._arange(host_rows.device)).reshape(-1)

    def device_items(self, device_rows: torch.Tensor) -> torch.Tensor:
        rows = device_rows.to(torch.int64).unsqueeze(1)
        base = rows * self.device_items_per_page + self.start_item
        return (base + self._arange(device_rows.device)).reshape(-1)


class _DSV4TailHostMixin:
    """Tail-only host rows (DSV4TailLayout) and the head-poison debug hook for
    the DSV4 page-row host pools. Hosts set tail_layout, poison_byte_ranges,
    pool_name, layout, layer_num, kv_buffer, device_ptrs, staging_buffer and
    can_use_write_back_jit; the defaults keep ``__new__`` test pools working."""

    tail_layout: Optional[DSV4TailLayout] = None
    poison_byte_ranges: tuple = ()
    _poison_calls = 0
    _poison_pages = 0

    def _refuse_tail(self, what: str) -> None:
        if self.tail_layout is not None:
            raise NotImplementedError(
                f"{self.pool_name}: {what} is not supported with "
                "SGLANG_DSV4_WINDOW_HOST_TAIL_ONLY (tail-only host pages)"
            )

    def _check_tail_backend(self, io_backend: str) -> None:
        if io_backend != "kernel" or self.layout != "page_first":
            raise ValueError(
                f"{self.pool_name}: the tail-only host layout needs the kernel io "
                f"backend and page_first, got {io_backend}/{self.layout}"
            )

    def _alloc_tail_buffers(self, num_host_pages: int, alloc_func) -> None:
        """Item-major host buffer: host page h is rows [h * tail_items,
        (h + 1) * tail_items), each row one item for every layer, so the
        existing page_first kernels copy it with item_size = tail item bytes."""
        tail = self.tail_layout
        self.kv_buffer = alloc_func(
            (num_host_pages * tail.tail_items, self.layer_num, tail.item_bytes),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
            registration_granularity_bytes=self.layer_num * tail.page_bytes,
        )

    def _init_tail_staging(self, staging_page_capacity: int) -> None:
        # Staged page = the tail's items of one page (page_size=tail_items).
        tail = self.tail_layout
        self.can_use_write_back_jit = _is_cuda and can_use_write_back_jit_kernel(
            element_size=tail.item_bytes * self.dtype.itemsize,
        )
        self.staging_buffer = torch.empty(
            (staging_page_capacity * tail.tail_items, self.layer_num, tail.item_bytes),
            dtype=self.dtype,
            device=self.gpu_device,
        )

    def _backup_tail(self, host_rows, device_rows, io_backend, sm_copy) -> None:
        """Copy the tail items of every page row pair device -> host."""
        self._check_tail_backend(io_backend)
        tail = self.tail_layout
        device_items = tail.device_items(device_rows)
        host_items = tail.host_items(host_rows)
        if self.can_use_write_back_jit and not sm_copy:
            # One staged "page" is one page's tail: tail_items consecutive
            # device items from its first index, one contiguous host block.
            jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                ptr_src=self.device_ptrs,
                src_indices=device_items,
                dst_indices=host_items,
                staging=self.staging_buffer,
                dst=self.kv_buffer,
                page_size=tail.tail_items,
                element_size=tail.item_bytes,
            )
            return
        if not host_items.is_cuda:
            host_items = host_items.to(device_items.device, non_blocking=True)
        transfer_kv_all_layer_mla_lf_pf(
            src_layers=self.device_ptrs,
            dst=self.kv_buffer,
            src_indices=device_items,
            dst_indices=host_items,
            item_size=tail.item_bytes,
            dst_layout_dim=self.layer_num * tail.item_bytes,
            num_layers=self.layer_num,
        )

    def _load_tail(self, host_rows, device_rows, layer_id, dst, io_backend) -> None:
        """Copy the tail items of every page row pair host -> device ``dst``
        (one layer's device page rows)."""
        self._check_tail_backend(io_backend)
        tail = self.tail_layout
        host_items = tail.host_items(host_rows)
        device_items = tail.device_items(device_rows)
        if not host_items.is_cuda:
            host_items = host_items.to(device_items.device, non_blocking=True)
        transfer_kv_per_layer_mla_pf_lf(
            src=self.kv_buffer,
            dst=dst,
            src_indices=host_items,
            dst_indices=device_items,
            layer_id=layer_id,
            item_size=tail.item_bytes,
            src_layout_dim=self.layer_num * tail.item_bytes,
            block_quota=HICACHE_LOAD_BLOCK_QUOTA,
        )

    def _poison_loaded_heads(
        self, device_rows: torch.Tensor, layer_id: int, buf: torch.Tensor, tag: str
    ) -> None:
        """SGLANG_DSV4_WINDOW_HOST_TAIL_POISON_HEAD: overwrite the head bytes
        of the just-loaded device page rows of ``buf`` with 0xFF, on the load
        stream."""
        rows = device_rows.to(device=buf.device, dtype=torch.int64)
        for start, end in self.poison_byte_ranges:
            buf[:, start:end].index_fill_(0, rows, 0xFF)
        if layer_id == 0:
            self._poison_calls += 1
            self._poison_pages += int(rows.numel())
            if self._poison_calls == 1 or self._poison_calls % 64 == 0:
                if tag == "DSV4_POISON_HEAD":
                    logger.info(
                        "DSV4_POISON_HEAD pages=%d cumulative=%d",
                        int(rows.numel()),
                        self._poison_pages,
                    )
                else:
                    logger.info(
                        "%s pool=%s pages=%d cumulative=%d",
                        tag,
                        self.pool_name,
                        int(rows.numel()),
                        self._poison_pages,
                    )


class DeepSeekV4PagedHostPool(_DSV4TailHostMixin, HiSparseHostPoolMixin, HostKVCache):
    """Host mirror for a DeepSeek V4 paged KV/indexer sub-pool."""

    # backup_from_device_all_layer(sm_copy=True): the kernel io backend's
    # page_first copy runs on SM kernels (layer_first always does).
    supports_sm_copy_backup = True

    def __init__(
        self,
        pool_name: str,
        device_buffers: list[torch.Tensor],
        item_bytes: int,
        num_host_pages: int,
        slot_page_size: int,
        layout: str = "layer_first",
        device: str = "cpu",
        pin_memory: bool = True,
        allocator_type: str = "default",
        page_aligned_only: bool = False,
        reserve_pages: int = 0,
        host_dedup=None,
        tail_layout: Optional[DSV4TailLayout] = None,
        poison_byte_ranges: tuple = (),
    ):
        if _is_xpu:
            require_kvcacheio_ops(
                pool=type(self).__name__, layout=layout, ops=self._kvcacheio_ops
            )
            # Token-granular transfers go through transfer_cache_dsv4_mla whatever
            # the io backend and layout, so it is checked here, not in the table.
            if not page_aligned_only and _dsv4_mla_unavailable_reason is not None:
                raise ValueError(
                    f"HiCache cannot build the {type(self).__name__} host pool here: "
                    f"{_dsv4_mla_unavailable_reason}. Launch without HiCache "
                    "(--enable-hierarchical-cache)."
                )
        self.pool_name = pool_name
        self.layer_num = len(device_buffers)
        # Device bytes per page row; item_bytes below is the host bytes per
        # page row, which differ only with a tail layout.
        self.device_page_bytes = item_bytes
        self.tail_layout = tail_layout
        if tail_layout is not None:
            if layout != "page_first":
                raise ValueError(
                    "SGLANG_DSV4_WINDOW_HOST_TAIL_ONLY needs --hicache-mem-layout "
                    f"page_first, got {layout}"
                )
            # Host dedup: only the source holds (tail) host bytes, and a
            # load-back broadcasts whole device rows from it, so every rank
            # gets the restored tail and the source's never-read head.
            if tail_layout.device_items_per_page * tail_layout.item_bytes != item_bytes:
                raise ValueError(
                    f"tail layout covers {tail_layout.device_items_per_page} x "
                    f"{tail_layout.item_bytes} B but the device page is {item_bytes} B"
                )
            for buf in device_buffers:
                if buf.dim() != 2 or buf.shape[1] * buf.element_size() != item_bytes:
                    raise ValueError(
                        f"{pool_name}: tail layout needs device rows of {item_bytes} B, "
                        f"got shape {tuple(buf.shape)}"
                    )
            item_bytes = tail_layout.page_bytes
        # (start, end) byte ranges of a device page row to overwrite with 0xFF
        # after each load (SGLANG_DSV4_WINDOW_HOST_TAIL_POISON_HEAD).
        self.poison_byte_ranges = tuple(poison_byte_ranges)
        self.item_bytes = item_bytes
        # The token-granular copy below addresses the fused DSv4 C4 row. Neither
        # the FP4 indexer, whose page rows group their slots, nor unified_kv,
        # whose page row is a flat run of tokens, has that layout, so those may
        # only move whole pages.
        self.page_aligned_only = page_aligned_only
        self.num_host_pages = num_host_pages
        self.slot_page_size = slot_page_size
        # kv-p2p: pages withheld from the free lists on every rank (see
        # PageSlotAllocator); 0 unless the peer-KV servicer is enabled.
        self.reserve_pages = reserve_pages
        self.dtype = torch.uint8
        self.device = device
        self.pin_memory = pin_memory
        self.allocator = get_allocator_from_storage(allocator_type, host_dedup)
        self.page_size = slot_page_size
        self.size = num_host_pages * slot_page_size
        self.layout = layout
        self.size_per_token = item_bytes
        self.start_layer = 0
        self.end_layer = self.layer_num
        self.lock = threading.RLock()

        self.device_buffers = device_buffers
        self.gpu_device = device_buffers[0].device if device_buffers else device

        if _init_dsv4_dedup_follower(self, host_dedup):
            self.clear()
            return

        requested_bytes = self.layer_num * num_host_pages * self.item_bytes
        available_bytes = _dsv4_host_memory_budget(host_dedup)
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for V4 paged pool {pool_name}. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )

        # ALLOC_MEMORY_FUNCS is keyed by device *type* string ("npu"/"musa"/...),
        # not torch.device objects; a torch.device key silently falls back to
        # cudaHostRegister, which fails on NPU. Resolve the alloc func by type str.
        _alloc_key = (
            self.gpu_device.type
            if isinstance(self.gpu_device, torch.device)
            else str(self.gpu_device)
        )
        alloc_func = ALLOC_MEMORY_FUNCS[_alloc_key]
        self.data_refs = []
        if self.layout == "layer_first":
            self.kv_buffer = [
                alloc_func(
                    (num_host_pages, self.item_bytes),
                    dtype=self.dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
                for _ in range(self.layer_num)
            ]
            self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        elif self.layout == "page_first" and self.tail_layout is not None:
            self._alloc_tail_buffers(num_host_pages, alloc_func)
        elif self.layout == "page_first":
            self.kv_buffer = alloc_func(
                (num_host_pages, self.layer_num, self.item_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
                registration_granularity_bytes=self.layer_num * self.item_bytes,
            )
        elif self.layout == "page_first_direct":
            self.kv_buffer = alloc_func(
                (num_host_pages, self.layer_num, 1, self.item_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
                registration_granularity_bytes=self.layer_num * self.item_bytes,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

        logger.info(
            "Allocating %.2f GB host memory for V4 paged pool '%s' "
            "(layers=%d, pages=%d, item_bytes=%d, layout=%s).",
            requested_bytes / 1e9,
            self.pool_name,
            self.layer_num,
            num_host_pages,
            self.item_bytes,
            self.layout,
        )

        self.device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_buffers],
            dtype=torch.uint64,
            device=self.gpu_device,
        )
        self.data_ptrs = (
            make_kernel_ptr_table(
                self.data_refs,
                self.gpu_device,
                host_memory_registered=self.pin_memory,
            )
            if self.data_refs
            else None
        )
        self.can_use_jit = False
        self.can_use_write_back_jit = False
        self._init_write_back_staging_buffers()
        self.clear()

    def _init_write_back_staging_buffers(self):
        self.staging_buffer = None
        if (
            self.layout != "page_first"
            or not current_platform.capabilities.hicache_device_kernels
        ):
            return

        staging_page_capacity = min(self.num_host_pages, _WRITE_BACK_STAGING_PAGE_CHUNK)
        if self.tail_layout is not None:
            self._init_tail_staging(staging_page_capacity)
            return
        self.can_use_write_back_jit = _is_cuda and can_use_write_back_jit_kernel(
            element_size=self.item_bytes * self.dtype.itemsize,
        )
        self.staging_buffer = torch.empty(
            (staging_page_capacity, self.layer_num, self.item_bytes),
            dtype=self.dtype,
            device=self.gpu_device,
        )

    def _host_page_view(self, l: int) -> torch.Tensor:
        """View a host layer as ``[pages, 1, P, 1, dim]``."""
        if self.layout == "layer_first":
            layer_buffer = self.kv_buffer[l]
        elif self.layout == "page_first":
            layer_buffer = self.kv_buffer[:, l, :]
        elif self.layout == "page_first_direct":
            layer_buffer = self.kv_buffer[:, l, 0, :]
        else:
            raise ValueError(
                f"{self.pool_name} _host_page_view: kernel_ascend requires "
                "layer_first/page_first/page_first_direct layout, "
                f"got {self.layout!r}"
            )
        device_buffer = self.device_buffers[l]
        return layer_buffer.view(device_buffer.dtype).view(
            self.num_host_pages,
            1,
            device_buffer.shape[1],
            1,
            device_buffer.shape[-1],
        )

    def get_contiguous_buf_infos(self):
        """Return per-layer page-row buffers for PD direct-to-host transfer."""
        self._refuse_tail("PD direct-to-host transfer")
        data_ptrs = [tensor.data_ptr() for tensor in self.data_refs]
        data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        item_lens = [self.item_bytes * self.dtype.itemsize] * self.layer_num
        return data_ptrs, data_lens, item_lens

    def _to_page_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return indices.reshape(-1, self.slot_page_size)[:, 0] // self.slot_page_size

    def _to_native_page_indices(
        self, indices: torch.Tensor, native_page_size: int
    ) -> torch.Tensor:
        """Expand logical page indices into a device buffer's native slots."""
        rows = self._to_page_indices(indices)
        offsets = torch.arange(native_page_size, dtype=rows.dtype, device=rows.device)
        return (rows[:, None] * native_page_size + offsets).reshape(-1)

    def _unaligned_transfer_error(
        self, host_indices: torch.Tensor, device_indices: torch.Tensor
    ) -> ValueError:
        return ValueError(
            f"{self.pool_name} expects page-aligned indices: got "
            f"{host_indices.numel()} host and {device_indices.numel()} device "
            f"indices for page size {self.slot_page_size}."
        )

    def _has_transfer_indices(
        self, host_indices: torch.Tensor | None, device_indices: torch.Tensor | None
    ) -> bool:
        if host_indices is None or device_indices is None:
            return False
        if host_indices.numel() != device_indices.numel():
            raise ValueError(
                f"{self.pool_name} transfer index size mismatch: "
                f"host={host_indices.numel()}, device={device_indices.numel()}"
            )
        return host_indices.numel() > 0

    def get_size_per_token(self):
        return self.item_bytes

    def get_ksize_per_token(self):
        return self.item_bytes

    def init_kv_buffer(self):
        return self.kv_buffer

    def get_hybrid_pool_buffer(self):
        return self.kv_buffer if isinstance(self.kv_buffer, list) else [self.kv_buffer]

    # Free-list state is owned by the PageSlotAllocator helper (kv-p2p slot
    # pins, rank-uniform reserve). Built lazily in clear() so lightweight
    # test constructions (``__new__`` + size/slot_page_size/lock) keep working.
    def _ensure_slot_allocator(self) -> PageSlotAllocator:
        slots = getattr(self, "_slots", None)
        if slots is None:
            slots = PageSlotAllocator(
                self.size,
                self.slot_page_size,
                reserve_pages=getattr(self, "reserve_pages", 0),
                lock=self.lock,
                name=getattr(self, "pool_name", "dsv4"),
            )
            self._slots = slots
        return slots

    @property
    def free_slots(self) -> torch.Tensor:
        return self._ensure_slot_allocator().free_slots

    @free_slots.setter
    def free_slots(self, value: torch.Tensor) -> None:
        self._ensure_slot_allocator().free_slots = value

    @property
    def release_slots(self) -> list:
        return self._ensure_slot_allocator().release_slots

    @property
    def num_release_slots(self) -> int:
        return self._ensure_slot_allocator().num_release_slots

    @property
    def slot_allocator(self) -> PageSlotAllocator:
        return self._ensure_slot_allocator()

    @synchronized
    def clear(self):
        self._ensure_slot_allocator().clear()

    @synchronized
    def available_size(self):
        return self._ensure_slot_allocator().available_size()

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        need_size = (
            (need_size + self.slot_page_size - 1) // self.slot_page_size
        ) * self.slot_page_size
        return self._ensure_slot_allocator().alloc(need_size)

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        # Whole pages only (the helper raises on an unaligned run): the rows
        # of this pool are addressed per page, and kv-p2p pins are per page.
        return self._ensure_slot_allocator().free(indices)

    if _is_xpu:
        # {io_backend: {layout: kvcacheio ops}} the transfer methods below call,
        # checked at construction; keep in step with their branches.
        _kvcacheio_ops = MLA_KVCACHEIO_OPS

    def backup_tokens_from_device_all_layer(self, host_indices, device_indices):
        """Copy individual tokens; the page path below would copy whole pages."""
        if not self._has_transfer_indices(host_indices, device_indices):
            return
        # Token-granular DSV4 C4 copy needs this helper because a token is
        # not one contiguous byte range in the paged row:
        # [value0..value63][scale0..scale63].
        if self.page_aligned_only:
            raise self._unaligned_transfer_error(host_indices, device_indices)
        transfer_cache_dsv4_mla(
            src_ptrs=self.device_ptrs,
            dst_ptrs=self.data_ptrs,
            src_indices=device_indices.to(dtype=torch.int64),
            dst_indices=host_indices.to(dtype=torch.int64),
        )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend, sm_copy=False
    ):
        """``sm_copy``: the kernel io backend's page_first copy uses the SM
        kernel (host indices on the device) instead of the staged copy-engine
        path; see CacheOperation.d2h_sm_copy."""
        if not self._has_transfer_indices(host_indices, device_indices):
            return
        if (
            host_indices.numel() % self.slot_page_size != 0
            or device_indices.numel() % self.slot_page_size != 0
        ):
            self._refuse_tail("a token-granular (partial page) copy")
            self.backup_tokens_from_device_all_layer(host_indices, device_indices)
            return
        # Whole C4 pages can use the normal HiCache page-row copy below.
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        if self.tail_layout is not None:
            self._backup_tail(host_rows, device_rows, io_backend, sm_copy)
            return
        if io_backend == "kernel" and self.layout == "layer_first":
            transfer_kv_all_layer_mla(
                src_layers=self.device_ptrs,
                dst_layers=self.data_ptrs,
                src_indices=device_rows,
                dst_indices=host_rows,
                item_size=self.item_bytes,
                num_layers=self.layer_num,
            )
        elif io_backend == "kernel" and self.layout == "page_first":
            if self.can_use_write_back_jit and not sm_copy:
                jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                    ptr_src=self.device_ptrs,
                    src_indices=device_rows,
                    dst_indices=host_rows,
                    staging=self.staging_buffer,
                    dst=self.kv_buffer,
                    page_size=1,
                    element_size=self.item_bytes,
                )
            else:
                if not host_rows.is_cuda:
                    host_rows = host_rows.to(device_rows.device, non_blocking=True)
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=self.device_ptrs,
                    dst=self.kv_buffer,
                    src_indices=device_rows,
                    dst_indices=host_rows,
                    item_size=self.item_bytes,
                    dst_layout_dim=self.layer_num * self.item_bytes,
                    num_layers=self.layer_num,
                )
        elif io_backend == "direct" and self.layout == "layer_first":
            transfer_kv_direct(
                src_layers=self.device_buffers,
                dst_layers=self.data_refs,
                src_indices=device_rows,
                dst_indices=host_rows,
                page_size=1,
            )
        elif io_backend == "direct" and self.layout == "page_first_direct":
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=self.device_buffers,
                dst_ptrs=[self.kv_buffer],
                src_indices=device_rows,
                dst_indices=host_rows,
                page_size=1,
            )
        elif io_backend == "kernel_ascend":
            # HiCache keeps KV-derived pools in the FULL logical coordinate
            # space. Ascend consumes the native slots of the shaped device
            # buffer, so translate the already resolved page rows only at this
            # backend boundary.
            native_page_size = int(self.device_buffers[0].shape[1])
            native_device_indices = self._to_native_page_indices(
                device_indices, native_page_size
            )
            native_host_indices = self._to_native_page_indices(
                host_indices, native_page_size
            )
            for l in range(self.layer_num):
                dev_view = self.device_buffers[l].unsqueeze(0)
                # dev_view: [1, dev_pages, native_page_size, 1, kv_dim]
                host_view = self._host_page_view(l)
                # host_view: [num_host_pages, 1, native_page_size, 1, kv_dim]
                transfer_kv_dim_exchange(
                    device_k=dev_view,
                    host_k=host_view,
                    device_v=torch.empty(0, device=dev_view.device),
                    host_v=torch.empty(0, device="cpu"),
                    device_indices=native_device_indices,
                    host_indices=native_host_indices,
                    page_size=native_page_size,
                    direction=TransferDirection.D2H,
                )
        else:
            raise ValueError(
                f"Unsupported V4 paged host layout/backend: {self.layout}/{io_backend}"
            )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if not self._has_transfer_indices(host_indices, device_indices):
            return
        if (
            host_indices.numel() % self.slot_page_size != 0
            or device_indices.numel() % self.slot_page_size != 0
        ):
            self._refuse_tail("a token-granular (partial page) load")
            # Same DSV4 C4 layout issue as backup: this is token-granular
            # preload, so it cannot use the normal HiCache page-row copy.
            if self.page_aligned_only:
                raise self._unaligned_transfer_error(host_indices, device_indices)
            transfer_cache_dsv4_mla(
                src_ptrs=self.data_ptrs[layer_id : layer_id + 1],
                dst_ptrs=self.device_ptrs[layer_id : layer_id + 1],
                src_indices=host_indices.to(dtype=torch.int64),
                dst_indices=device_indices.to(dtype=torch.int64),
            )
            return
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        self._load_page_rows(
            host_rows, device_rows, layer_id, io_backend, host_indices, device_indices
        )
        if self.poison_byte_ranges:
            self._poison_loaded_heads(
                device_rows, layer_id, self.device_buffers[layer_id], "DSV4_POISON_HEAD"
            )

    def _load_page_rows(
        self, host_rows, device_rows, layer_id, io_backend, host_indices, device_indices
    ) -> None:
        # host_indices/device_indices: the slot indices behind the rows, for
        # the kernel_ascend branch (native slots of the shaped device buffer).
        if self.tail_layout is not None:
            self._load_tail(
                host_rows,
                device_rows,
                layer_id,
                self.device_buffers[layer_id],
                io_backend,
            )
            return
        if io_backend == "kernel" and self.layout == "layer_first":
            transfer_kv_per_layer_mla(
                src=self.data_refs[layer_id],
                dst=self.device_buffers[layer_id],
                src_indices=host_rows,
                dst_indices=device_rows,
                item_size=self.item_bytes,
            )
        elif io_backend == "kernel" and self.layout == "page_first":
            transfer_kv_per_layer_mla_pf_lf(
                src=self.kv_buffer,
                dst=self.device_buffers[layer_id],
                src_indices=host_rows,
                dst_indices=device_rows,
                layer_id=layer_id,
                item_size=self.item_bytes,
                src_layout_dim=self.layer_num * self.item_bytes,
                block_quota=HICACHE_LOAD_BLOCK_QUOTA,
            )
        elif io_backend == "direct" and self.layout == "layer_first":
            transfer_kv_direct(
                src_layers=[self.data_refs[layer_id]],
                dst_layers=[self.device_buffers[layer_id]],
                src_indices=host_rows,
                dst_indices=device_rows,
                page_size=1,
            )
        elif io_backend == "direct" and self.layout == "page_first_direct":
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[self.kv_buffer],
                dst_ptrs=[self.device_buffers[layer_id]],
                src_indices=host_rows,
                dst_indices=device_rows,
                layer_id=layer_id,
                page_size=1,
            )
        elif io_backend == "kernel_ascend":
            # NPU whole-page H2D via Ascend dim-exchange op, for layer_id only.
            native_page_size = int(self.device_buffers[layer_id].shape[1])
            native_device_indices = self._to_native_page_indices(
                device_indices, native_page_size
            )
            native_host_indices = self._to_native_page_indices(
                host_indices, native_page_size
            )
            dev_view = self.device_buffers[layer_id].unsqueeze(0)
            # dev_view: [1, dev_pages, native_page_size, 1, kv_dim]
            host_view = self._host_page_view(layer_id)
            # host_view: [num_host_pages, 1, native_page_size, 1, kv_dim]
            transfer_kv_dim_exchange(
                device_k=dev_view,
                host_k=host_view,
                device_v=torch.empty(0, device=dev_view.device),
                host_v=torch.empty(0, device="cpu"),
                device_indices=native_device_indices,
                host_indices=native_host_indices,
                page_size=native_page_size,
                direction=TransferDirection.H2D,
            )
        else:
            raise ValueError(
                f"Unsupported V4 paged host layout/backend: {self.layout}/{io_backend}"
            )

    def get_data_page(self, index, flat=True):
        self._refuse_tail("get_data_page (storage)")
        index = int(index) // self.slot_page_size
        if self.layout == "layer_first":
            data_page = torch.stack(
                [self.kv_buffer[i][index] for i in range(self.layer_num)]
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            data_page = self.kv_buffer[index]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self):
        self._refuse_tail("get_dummy_flat_data_page (storage)")
        return torch.zeros(
            (self.layer_num, self.item_bytes),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index, data_page):
        self._refuse_tail("set_from_flat_data_page (storage)")
        index = int(index) // self.slot_page_size
        if self.layout == "layer_first":
            data = data_page.view(self.dtype).reshape(self.layer_num, self.item_bytes)
            for i in range(self.layer_num):
                self.kv_buffer[i][index].copy_(data[i])
        elif self.layout == "page_first":
            self.kv_buffer[index].copy_(
                data_page.view(self.dtype).reshape(self.layer_num, self.item_bytes)
            )
        elif self.layout == "page_first_direct":
            self.kv_buffer[index].copy_(
                data_page.view(self.dtype).reshape(self.layer_num, 1, self.item_bytes)
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        self._refuse_tail("get_page_buffer_meta (storage)")
        ptr_list = []
        rows = self._to_page_indices(indices).tolist()
        if self.layout == "layer_first":
            for row in rows:
                page_index = int(row)
                for layer_id in range(self.layer_num):
                    ptr = (
                        self.kv_buffer[layer_id].data_ptr()
                        + page_index * self.item_bytes * self.dtype.itemsize
                    )
                    ptr_list.append(ptr)
            element_size = self.item_bytes * self.dtype.itemsize
            return ptr_list, [element_size] * len(ptr_list)
        if self.layout in ["page_first", "page_first_direct"]:
            page_bytes = self.layer_num * self.item_bytes * self.dtype.itemsize
            for row in rows:
                ptr_list.append(self.kv_buffer[int(row)].data_ptr())
            return ptr_list, [page_bytes] * len(ptr_list)
        raise ValueError(f"Unsupported layout: {self.layout}")

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        if self.layout not in ["page_first", "page_first_direct"]:
            return False
        if self.tail_layout is not None:
            return False
        page_bytes = self.layer_num * self.item_bytes * self.dtype.itemsize
        return (
            self.kv_buffer.data_ptr() % page_size_bytes == 0
            and page_bytes % page_size_bytes == 0
        )


class DeepSeekV4StateHostPool(_DSV4TailHostMixin, HostKVCache):
    """Host pool for V4 CompressStatePool page rows."""

    # See DeepSeekV4PagedHostPool.supports_sm_copy_backup.
    supports_sm_copy_backup = True

    def __init__(
        self,
        pool_name: str,
        state_pools: list,
        num_host_pages: int,
        swa_page_size: int,
        layout: str = "layer_first",
        device: str = "cpu",
        pin_memory: bool = True,
        allocator_type: str = "default",
        host_dedup=None,
        window_rows: int = 0,
        tail_only: bool = False,
        poison_head: bool = False,
    ):
        """``window_rows``: the ring rows a page-aligned boundary's readers
        need (the compressor window, 2 * ratio for c4); with ``tail_only``
        (SGLANG_DSV4_WINDOW_HOST_STATE_TAIL_ONLY) the host keeps only the
        last ``window_rows`` rows of each page's ring, and with
        ``poison_head`` (SGLANG_DSV4_WINDOW_HOST_TAIL_POISON_HEAD) the rows
        before them are overwritten with 0xFF after every load."""
        if any(pool is None for pool in state_pools):
            raise ValueError(f"{pool_name} state_pools must not contain None")
        if _is_xpu:
            require_kvcacheio_ops(
                pool=type(self).__name__, layout=layout, ops=self._kvcacheio_ops
            )

        self.pool_name = pool_name
        self.state_pools = state_pools
        self.layer_num = len(state_pools)
        self.num_host_pages = num_host_pages
        self.swa_page_size = swa_page_size
        self.dtype = torch.uint8
        self.device = device
        self.pin_memory = pin_memory
        self.allocator = get_allocator_from_storage(allocator_type, host_dedup)
        self.page_size = swa_page_size
        self.size = num_host_pages * swa_page_size
        self.layout = layout
        self.start_layer = 0
        self.end_layer = self.layer_num
        self.lock = threading.RLock()

        self.ring_size = 0
        self.state_page_bytes = 0
        self.slot_bytes = 0
        self.device_page_views = []
        self.gpu_device = device
        self._init_device_page_views()
        self._init_state_tail(window_rows, tail_only, poison_head)
        # Host bytes per page row; differs from the device ring bytes
        # (state_page_bytes) only with a tail layout.
        self.host_page_bytes = (
            self.tail_layout.page_bytes
            if self.tail_layout is not None
            else self.state_page_bytes
        )
        self.size_per_token = self.host_page_bytes

        if _init_dsv4_dedup_follower(self, host_dedup):
            return

        requested_bytes = self.layer_num * num_host_pages * self.host_page_bytes
        available_bytes = _dsv4_host_memory_budget(host_dedup)
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for V4 state pool {pool_name}. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )

        # ALLOC_MEMORY_FUNCS is keyed by device *type* string ("npu"/"musa"/...),
        # not torch.device objects; resolve the key the same way PagedHostPool does.
        _state_alloc_key = (
            self.gpu_device.type
            if isinstance(self.gpu_device, torch.device)
            else str(self.gpu_device)
        )
        alloc_func = ALLOC_MEMORY_FUNCS[_state_alloc_key]
        self.data_refs = []
        if self.layout == "layer_first":
            self.kv_buffer = [
                alloc_func(
                    (num_host_pages, self.state_page_bytes),
                    dtype=self.dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
                for _ in range(self.layer_num)
            ]
            self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        elif self.layout == "page_first" and self.tail_layout is not None:
            self._alloc_tail_buffers(num_host_pages, alloc_func)
        elif self.layout == "page_first":
            self.kv_buffer = alloc_func(
                (num_host_pages, self.layer_num, self.state_page_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
                registration_granularity_bytes=(self.layer_num * self.state_page_bytes),
            )
        elif self.layout == "page_first_direct":
            self.kv_buffer = alloc_func(
                (num_host_pages, self.layer_num, 1, self.state_page_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
                registration_granularity_bytes=(self.layer_num * self.state_page_bytes),
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        logger.info(
            "Allocating %.2f GB host memory for V4 state pool '%s' "
            "(layers=%d, pages=%d, state_page_bytes=%d, host_page_bytes=%d, "
            "layout=%s).",
            requested_bytes / 1e9,
            self.pool_name,
            self.layer_num,
            num_host_pages,
            self.state_page_bytes,
            self.host_page_bytes,
            self.layout,
        )
        self.device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_page_views],
            dtype=torch.uint64,
            device=self.gpu_device,
        )
        self.data_ptrs = (
            make_kernel_ptr_table(
                self.data_refs,
                self.gpu_device,
                host_memory_registered=self.pin_memory,
            )
            if self.data_refs
            else None
        )
        self.can_use_jit = False
        self.can_use_write_back_jit = False
        self._init_write_back_staging_buffers()

    def _init_device_page_views(self) -> None:
        expected_ring_size = None
        expected_state_page_bytes = None
        for pool in self.state_pools:
            state_tensor = pool.kv_score_buffer.kv_score
            if not state_tensor.is_contiguous():
                raise ValueError(f"{self.pool_name} state tensor must be contiguous")
            ring_size = pool.ring_size
            slot_bytes = state_tensor[0].nbytes
            state_page_bytes = ring_size * slot_bytes
            if expected_ring_size is None:
                expected_ring_size = ring_size
                expected_state_page_bytes = state_page_bytes
                self.gpu_device = state_tensor.device
            elif (
                expected_ring_size != ring_size
                or expected_state_page_bytes != state_page_bytes
            ):
                raise ValueError(
                    f"{self.pool_name} state pools must share ring size and slot bytes"
                )

            state_bytes = state_tensor.view(torch.uint8).reshape(
                state_tensor.shape[0], -1
            )
            usable_slots = (state_tensor.shape[0] // ring_size) * ring_size
            self.device_page_views.append(
                state_bytes[:usable_slots].reshape(-1, state_page_bytes)
            )
            self.slot_bytes = slot_bytes

        self.ring_size = expected_ring_size or 0
        self.state_page_bytes = expected_state_page_bytes or 0

    def _init_state_tail(self, window_rows, tail_only, poison_head) -> None:
        """Tail layout (rows [ring - window_rows, ring) of each page's ring)
        and the head rows to poison after loads. With host dedup, as for the
        SWA pool: the source's host copy is tail-only and a load-back
        broadcasts whole rings."""
        self.tail_layout = None
        self.poison_byte_ranges = ()
        if not (tail_only or poison_head):
            return
        head_rows = self.ring_size - int(window_rows)
        if window_rows <= 0 or head_rows <= 0:
            # The ring is one compressor window (no MTP): nothing to trim.
            logger.info(
                "%s: ring of %d rows has no rows before the %d-row window; "
                "state tail / poison is a no-op.",
                self.pool_name,
                self.ring_size,
                window_rows,
            )
            return
        if tail_only:
            if self.layout != "page_first":
                raise ValueError(
                    "SGLANG_DSV4_WINDOW_HOST_STATE_TAIL_ONLY needs "
                    f"--hicache-mem-layout page_first, got {self.layout}"
                )
            self.tail_layout = DSV4TailLayout(
                device_page_bytes=self.state_page_bytes,
                item_bytes=self.slot_bytes,
                start_item=head_rows,
            )
        if poison_head:
            self.poison_byte_ranges = ((0, head_rows * self.slot_bytes),)

    def _init_write_back_staging_buffers(self):
        self.staging_buffer = None
        if (
            self.layout != "page_first"
            or not current_platform.capabilities.hicache_device_kernels
        ):
            return

        staging_page_capacity = min(self.num_host_pages, _WRITE_BACK_STAGING_PAGE_CHUNK)
        if self.tail_layout is not None:
            self._init_tail_staging(staging_page_capacity)
            return
        self.can_use_write_back_jit = _is_cuda and can_use_write_back_jit_kernel(
            element_size=self.state_page_bytes * self.dtype.itemsize,
        )
        self.staging_buffer = torch.empty(
            (staging_page_capacity, self.layer_num, self.state_page_bytes),
            dtype=self.dtype,
            device=self.gpu_device,
        )

    def _to_page_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() % self.swa_page_size != 0:
            raise ValueError(
                f"{self.pool_name} transfer indices must be SWA-page-aligned, "
                f"got numel={indices.numel()}, swa_page_size={self.swa_page_size}"
            )
        return indices.reshape(-1, self.swa_page_size)[:, 0] // self.swa_page_size

    def _ring_op_indices(self, rows: torch.Tensor) -> torch.Tensor:
        """Expand each SWA page row into ``ring_size`` operator indices.

        For example, row ``r`` maps to ``r * ring_size + arange(ring_size)``."""
        ar = torch.arange(self.ring_size, device=rows.device, dtype=rows.dtype)
        return (rows.reshape(-1, 1) * self.ring_size + ar).reshape(-1)

    def _state_host_page_view(self, l: int) -> torch.Tensor:
        """View host state layer ``l`` as ``[pages, 1, R, 1, last_dim]``.

        ``last_dim = state_page_bytes // R // state_dtype.itemsize``."""
        state_dtype = self.state_pools[l].kv_score_buffer.kv_score.dtype
        last_dim = self.state_page_bytes // self.ring_size // state_dtype.itemsize
        if self.layout == "layer_first":
            layer_buffer = self.kv_buffer[l]
        elif self.layout == "page_first":
            layer_buffer = self.kv_buffer[:, l, :]
        elif self.layout == "page_first_direct":
            layer_buffer = self.kv_buffer[:, l, 0, :]
        else:
            raise ValueError(
                f"{self.pool_name} _state_host_page_view: kernel_ascend requires "
                "layer_first/page_first/page_first_direct layout, "
                f"got {self.layout!r}"
            )
        return layer_buffer.view(state_dtype).view(
            self.num_host_pages, 1, self.ring_size, 1, last_dim
        )

    def get_size_per_token(self):
        return self.host_page_bytes

    def get_ksize_per_token(self):
        return self.host_page_bytes

    def init_kv_buffer(self):
        return self.kv_buffer

    def get_hybrid_pool_buffer(self):
        return self.kv_buffer if isinstance(self.kv_buffer, list) else [self.kv_buffer]

    def clear(self):
        pass

    def available_size(self):
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no allocator"
        )

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no allocator"
        )

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no free list"
        )

    if _is_xpu:
        # {io_backend: {layout: kvcacheio ops}} the transfer methods below call,
        # checked at construction; keep in step with their branches.
        _kvcacheio_ops = MLA_KVCACHEIO_OPS

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend, sm_copy=False
    ):
        """``sm_copy``: see DeepSeekV4PagedHostPool.backup_from_device_all_layer."""
        if host_indices is None or device_indices is None:
            return
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        if self.tail_layout is not None:
            self._backup_tail(host_rows, device_rows, io_backend, sm_copy)
            return
        if io_backend == "kernel" and self.layout == "layer_first":
            assert self.data_ptrs is not None
            transfer_kv_all_layer_mla(
                src_layers=self.device_ptrs,
                dst_layers=self.data_ptrs,
                src_indices=device_rows,
                dst_indices=host_rows,
                item_size=self.state_page_bytes,
                num_layers=self.layer_num,
            )
        elif io_backend == "kernel" and self.layout == "page_first":
            if self.can_use_write_back_jit and not sm_copy:
                jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                    ptr_src=self.device_ptrs,
                    src_indices=device_rows,
                    dst_indices=host_rows,
                    staging=self.staging_buffer,
                    dst=self.kv_buffer,
                    page_size=1,
                    element_size=self.state_page_bytes,
                )
            else:
                if not host_rows.is_cuda:
                    host_rows = host_rows.to(device_rows.device, non_blocking=True)
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=self.device_ptrs,
                    dst=self.kv_buffer,
                    src_indices=device_rows,
                    dst_indices=host_rows,
                    item_size=self.state_page_bytes,
                    dst_layout_dim=self.layer_num * self.state_page_bytes,
                    num_layers=self.layer_num,
                )
        elif io_backend == "direct" and self.layout == "layer_first":
            transfer_kv_direct(
                src_layers=self.device_page_views,
                dst_layers=self.data_refs,
                src_indices=device_rows,
                dst_indices=host_rows,
                page_size=1,
            )
        elif io_backend == "direct" and self.layout == "page_first_direct":
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=self.device_page_views,
                dst_ptrs=[self.kv_buffer],
                src_indices=device_rows,
                dst_indices=host_rows,
                page_size=1,
            )
        elif io_backend == "kernel_ascend":
            # Ascend copies ring_size state slots for each page-aligned SWA row.
            # _ring_op_indices expands device and host rows into operator indices.
            dev_op_indices = self._ring_op_indices(device_rows)
            host_op_indices = self._ring_op_indices(host_rows)
            for l in range(self.layer_num):
                state_dtype = self.state_pools[l].kv_score_buffer.kv_score.dtype
                last_dim = (
                    self.state_page_bytes // self.ring_size // state_dtype.itemsize
                )
                num_dev_pages = self.device_page_views[l].shape[0]
                dev_view = (
                    self.device_page_views[l]
                    .view(state_dtype)
                    .view(num_dev_pages, self.ring_size, 1, last_dim)
                    .unsqueeze(0)
                )
                # dev_view: [1, num_dev_pages, R, 1, last_dim]
                host_view = self._state_host_page_view(l)
                # host_view: [num_host_pages, 1, R, 1, last_dim]
                transfer_kv_dim_exchange(
                    device_k=dev_view,
                    host_k=host_view,
                    device_v=torch.empty(0, device=dev_view.device),
                    host_v=torch.empty(0, device="cpu"),
                    device_indices=dev_op_indices,
                    host_indices=host_op_indices,
                    page_size=self.ring_size,
                    direction=TransferDirection.D2H,
                )
        else:
            raise ValueError(
                f"Unsupported V4 state host layout/backend: {self.layout}/{io_backend}"
            )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if host_indices is None or device_indices is None:
            return
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        self._load_state_rows(host_rows, device_rows, layer_id, io_backend)
        if self.poison_byte_ranges:
            self._poison_loaded_heads(
                device_rows,
                layer_id,
                self.device_page_views[layer_id],
                "DSV4_POISON_STATE_HEAD",
            )

    def _load_state_rows(self, host_rows, device_rows, layer_id, io_backend) -> None:
        if self.tail_layout is not None:
            self._load_tail(
                host_rows,
                device_rows,
                layer_id,
                self.device_page_views[layer_id],
                io_backend,
            )
            return
        if io_backend == "kernel" and self.layout == "layer_first":
            transfer_kv_per_layer_mla(
                src=self.data_refs[layer_id],
                dst=self.device_page_views[layer_id],
                src_indices=host_rows,
                dst_indices=device_rows,
                item_size=self.state_page_bytes,
            )
        elif io_backend == "kernel" and self.layout == "page_first":
            transfer_kv_per_layer_mla_pf_lf(
                src=self.kv_buffer,
                dst=self.device_page_views[layer_id],
                src_indices=host_rows,
                dst_indices=device_rows,
                layer_id=layer_id,
                item_size=self.state_page_bytes,
                src_layout_dim=self.layer_num * self.state_page_bytes,
                block_quota=HICACHE_LOAD_BLOCK_QUOTA,
            )
        elif io_backend == "direct" and self.layout == "layer_first":
            transfer_kv_direct(
                src_layers=[self.data_refs[layer_id]],
                dst_layers=[self.device_page_views[layer_id]],
                src_indices=host_rows,
                dst_indices=device_rows,
                page_size=1,
            )
        elif io_backend == "direct" and self.layout == "page_first_direct":
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[self.kv_buffer],
                dst_ptrs=[self.device_page_views[layer_id]],
                src_indices=host_rows,
                dst_indices=device_rows,
                layer_id=layer_id,
                page_size=1,
            )
        elif io_backend == "kernel_ascend":
            # NPU whole-page H2D via Ascend dim-exchange op, for layer_id only.
            # See backup_from_device_all_layer: indices must be ring-row indices
            # (ring_size entries per SWA page), not raw SWA locs.
            R = self.ring_size
            dev_op_indices = self._ring_op_indices(device_rows)
            host_op_indices = self._ring_op_indices(host_rows)
            state_dtype = self.state_pools[layer_id].kv_score_buffer.kv_score.dtype
            last_dim = self.state_page_bytes // R // state_dtype.itemsize
            num_dev_pages = self.device_page_views[layer_id].shape[0]
            dev_view = (
                self.device_page_views[layer_id]
                .view(state_dtype)
                .view(num_dev_pages, R, 1, last_dim)
                .unsqueeze(0)
            )
            # dev_view: [1, num_dev_pages, R, 1, last_dim]
            host_view = self._state_host_page_view(layer_id)
            # host_view: [num_host_pages, 1, R, 1, last_dim]
            transfer_kv_dim_exchange(
                device_k=dev_view,
                host_k=host_view,
                device_v=torch.empty(0, device=dev_view.device),
                host_v=torch.empty(0, device="cpu"),
                device_indices=dev_op_indices,
                host_indices=host_op_indices,
                page_size=R,
                direction=TransferDirection.H2D,
            )
        else:
            raise ValueError(
                f"Unsupported V4 state host layout/backend: {self.layout}/{io_backend}"
            )

    def get_data_page(self, index, flat=True):
        self._refuse_tail("get_data_page (storage)")
        index = int(index) // self.swa_page_size
        if self.layout == "layer_first":
            data_page = torch.stack(
                [self.kv_buffer[i][index] for i in range(self.layer_num)]
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            data_page = self.kv_buffer[index]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self):
        self._refuse_tail("get_dummy_flat_data_page (storage)")
        return torch.zeros(
            (self.layer_num, self.state_page_bytes),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index, data_page):
        self._refuse_tail("set_from_flat_data_page (storage)")
        index = int(index) // self.swa_page_size
        if self.layout == "layer_first":
            data = data_page.view(self.dtype).reshape(
                self.layer_num, self.state_page_bytes
            )
            for i in range(self.layer_num):
                self.kv_buffer[i][index].copy_(data[i])
        elif self.layout == "page_first":
            self.kv_buffer[index].copy_(
                data_page.view(self.dtype).reshape(
                    self.layer_num, self.state_page_bytes
                )
            )
        elif self.layout == "page_first_direct":
            self.kv_buffer[index].copy_(
                data_page.view(self.dtype).reshape(
                    self.layer_num, 1, self.state_page_bytes
                )
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        self._refuse_tail("get_page_buffer_meta (storage)")
        ptr_list = []
        rows = self._to_page_indices(indices).tolist()
        if self.layout == "layer_first":
            for row in rows:
                page_index = int(row)
                for layer_id in range(self.layer_num):
                    ptr = (
                        self.kv_buffer[layer_id].data_ptr()
                        + page_index * self.state_page_bytes * self.dtype.itemsize
                    )
                    ptr_list.append(ptr)
            element_size = self.state_page_bytes * self.dtype.itemsize
            return ptr_list, [element_size] * len(ptr_list)
        if self.layout in ["page_first", "page_first_direct"]:
            page_bytes = self.layer_num * self.state_page_bytes * self.dtype.itemsize
            for row in rows:
                ptr_list.append(self.kv_buffer[int(row)].data_ptr())
            return ptr_list, [page_bytes] * len(ptr_list)
        raise ValueError(f"Unsupported layout: {self.layout}")

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        if self.layout not in ["page_first", "page_first_direct"]:
            return False
        if self.tail_layout is not None:
            return False
        page_bytes = self.layer_num * self.state_page_bytes * self.dtype.itemsize
        return (
            self.kv_buffer.data_ptr() % page_size_bytes == 0
            and page_bytes % page_size_bytes == 0
        )
