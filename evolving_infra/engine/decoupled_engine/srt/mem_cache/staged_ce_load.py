"""Copy-engine host->device load for page_first DSA host pools (opt-in).

The default load-back runs one SM gather kernel per layer that reads 576-byte
token rows strided 45 KB apart in the page_first host layout. This loader
instead copies whole host pages (all layers of 64 tokens are contiguous) into
a device staging buffer with ``cudaMemcpyAsync`` (copy engine; one copy per
run of consecutive host pages), then scatters each layer from staging into
the per-layer device buffers with ``index_copy_`` (local device memory).

Used on the host-dedup source rank of the CUDA IPC fan-out
(``SGLANG_DSA_HOST_DEDUP_CE_LOAD=1``). Transfers it cannot handle (other
layouts, unaligned or device-resident host indices) fall back to the gather
kernel. All work is enqueued on the caller's (load) stream.

Large host pools are registered with ``cudaHostRegister`` in several ranges
(``SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB``, page-granular for page_first). One
``cudaMemcpyAsync`` must not span two registrations: it fails with
``cudaErrorInvalidValue`` (seen on GB300 with a 525 GiB KV pool in three
ranges). Coalesced page runs are therefore split at registration boundaries.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


def page_runs(pages: Sequence[int]) -> List[Tuple[int, int, int]]:
    """(dst_offset, first_page, count) for runs of consecutive page ids."""
    runs: List[Tuple[int, int, int]] = []
    for i, p in enumerate(pages):
        if runs and runs[-1][1] + runs[-1][2] == p:
            d, s, n = runs[-1]
            runs[-1] = (d, s, n + 1)
        else:
            runs.append((i, p, 1))
    return runs


def split_runs(
    runs: Sequence[Tuple[int, int, int]], boundaries: Sequence[int]
) -> List[Tuple[int, int, int]]:
    """Split (dst, src, n) page runs so none crosses a boundary page index
    (the first page of another host registration)."""
    out: List[Tuple[int, int, int]] = []
    for dst, src, n in runs:
        end = src + n
        for b in boundaries:
            if src < b < end:
                out.append((dst, src, b - src))
                dst, src = dst + (b - src), b
        out.append((dst, src, end - src))
    return out


def registration_boundaries(
    host_buf: torch.Tensor, page_bytes: int
) -> Optional[List[int]]:
    """Page indices where a new cudaHostRegister range of ``host_buf`` starts,
    or None if a boundary does not fall on a page (the caller falls back)."""
    from decoupled_engine.srt.mem_cache.pool_host.common import _CUDA_HOST_REGISTERED_RANGES_ATTR

    ranges = getattr(host_buf, _CUDA_HOST_REGISTERED_RANGES_ATTR, None) or ()
    base = host_buf.data_ptr()
    total = host_buf.numel() * host_buf.element_size()
    out: List[int] = []
    for ptr, _size in ranges:
        off = ptr - base
        if off <= 0 or off >= total:
            continue
        if off % page_bytes:
            return None
        out.append(off // page_bytes)
    return sorted(out)


def token_pages(
    host_indices_cpu: torch.Tensor, page_size: int
) -> Optional[torch.Tensor]:
    """Page ids of page-aligned, in-page-contiguous token indices, else None."""
    n = host_indices_cpu.numel()
    if n == 0 or n % page_size:
        return None
    rows = host_indices_cpu.reshape(-1, page_size).to(torch.int64)
    first = rows[:, :1]
    if first.remainder(page_size).any():
        return None
    if not torch.equal(rows - first, torch.arange(page_size).expand_as(rows)):
        return None
    return (first[:, 0] // page_size).contiguous()


class StagedCELoader:
    def __init__(self, page_size: int, staging_bytes: int, device):
        self.page_size = page_size
        self.staging_bytes = staging_bytes
        self.device = torch.device(device)
        self._staging: Dict[object, torch.Tensor] = {}
        self.stats = {"ops": 0, "copies": 0, "bytes": 0, "fallback": 0}
        self._pinned: Dict[int, bool] = {}
        self._boundaries: Dict[Tuple[int, int], Optional[List[int]]] = {}

    def _host_pinned(self, host_buf) -> bool:
        """A pageable source would make every copy synchronous: fall back."""
        key = host_buf.data_ptr()
        if key not in self._pinned:
            self._pinned[key] = bool(host_buf.is_pinned())
            if not self._pinned[key]:
                logger.warning("CE load disabled for an unpinned host pool")
        return self._pinned[key]

    def _staging_for(self, stream) -> torch.Tensor:
        key = getattr(stream, "cuda_stream", None) or id(stream)
        buf = self._staging.get(key)
        if buf is None:
            buf = torch.empty(self.staging_bytes, dtype=torch.uint8, device=self.device)
            self._staging[key] = buf
        return buf

    @staticmethod
    def _kind(host_pool) -> Optional[str]:
        from decoupled_engine.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
        from decoupled_engine.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        if getattr(host_pool, "layout", None) != "page_first":
            return None
        if isinstance(host_pool, DSAIndexerPoolHost):
            return "idx"
        if isinstance(host_pool, MLATokenToKVPoolHost):
            return "kv"
        return None

    def load(
        self,
        plan: Iterable[Tuple[int, list]],
        host_indices_cpu: Dict[int, torch.Tensor],
        stream,
    ) -> set:
        """Load every (transfer, layer) of ``plan`` it can; return handled ids."""
        groups: Dict[int, list] = {}
        for _, layer_transfers in plan:
            for transfer, local_layer_id in layer_transfers:
                groups.setdefault(id(transfer.host_pool), []).append(
                    (transfer, local_layer_id)
                )
        handled: set = set()
        staging = self._staging_for(stream)
        for items in groups.values():
            host_pool = items[0][0].host_pool
            kind = self._kind(host_pool)
            cpu = host_indices_cpu.get(id(items[0][0].host_indices))
            same = all(t.host_indices is items[0][0].host_indices for t, _ in items)
            pages = token_pages(cpu, self.page_size) if cpu is not None else None
            if (
                kind is None
                or pages is None
                or not same
                or getattr(host_pool, "_is_dummy", False)
            ):
                self.stats["fallback"] += 1
                continue
            host_buf = (
                host_pool.kv_buffer
                if kind == "kv"
                else host_pool.index_k_with_scale_buffer
            )
            if not self._host_pinned(host_buf):
                self.stats["fallback"] += 1
                continue
            if self._page_boundaries(host_buf, kind) is None:
                self.stats["fallback"] += 1
                continue
            if kind == "kv":
                self._load_kv(host_pool, items, pages, staging)
            else:
                self._load_idx(host_pool, items, pages, staging)
            handled.update(id(t) for t, _ in items)
        if handled:
            self.stats["ops"] += 1
        return handled

    @staticmethod
    def _layer_target(host_pool, transfer, local_layer_id):
        """(host layer, device layer) as load_to_device_per_layer would use."""
        if transfer.is_draft:
            return local_layer_id, 0
        if not host_pool._is_device_layer_owned(transfer.device_pool, local_layer_id):
            return None
        return host_pool._host_layer_index(local_layer_id), local_layer_id

    def _page_bytes(self, host_buf, kind) -> int:
        rows_per_page = self.page_size if kind == "kv" else 1
        return rows_per_page * host_buf[0].numel() * host_buf.element_size()

    def _page_boundaries(self, host_buf, kind) -> Optional[List[int]]:
        key = (host_buf.data_ptr(), host_buf.numel())
        if key not in self._boundaries:
            b = registration_boundaries(host_buf, self._page_bytes(host_buf, kind))
            self._boundaries[key] = b
            if b is None:
                logger.warning(
                    "CE load disabled for a host pool whose registration ranges "
                    "do not fall on page boundaries"
                )
        return self._boundaries[key]

    def _copy_pages(
        self, host_buf, rows_per_page, pages_chunk, staging_view, boundaries=()
    ):
        for dst, src, n in split_runs(page_runs(pages_chunk), boundaries):
            staging_view[dst * rows_per_page : (dst + n) * rows_per_page].copy_(
                host_buf[src * rows_per_page : (src + n) * rows_per_page],
                non_blocking=True,
            )
            self.stats["copies"] += 1

    def _load_kv(self, host_pool, items, pages, staging) -> None:
        host_buf = host_pool.kv_buffer  # [tokens, layers, 1, dim]
        per_token = host_buf[0].numel() * host_buf.element_size()
        per_page = per_token * self.page_size
        chunk_pages = max(1, staging.numel() // per_page)
        page_list = pages.tolist()
        for start in range(0, len(page_list), chunk_pages):
            chunk = page_list[start : start + chunk_pages]
            k = len(chunk)
            view = (
                staging[: k * per_page]
                .view(host_buf.dtype)
                .view(k * self.page_size, *host_buf.shape[1:])
            )
            self._copy_pages(
                host_buf,
                self.page_size,
                chunk,
                view,
                self._page_boundaries(host_buf, "kv"),
            )
            tok0, tok1 = start * self.page_size, (start + k) * self.page_size
            for transfer, local_layer_id in items:
                target = self._layer_target(host_pool, transfer, local_layer_id)
                if target is None:
                    continue
                host_layer, device_layer = target
                dst = transfer.device_pool.kv_buffer[device_layer]
                rows = transfer.device_indices[tok0:tok1]
                dst.index_copy_(0, rows.to(dst.device), view[:, host_layer])
            self.stats["bytes"] += k * per_page

    def _load_idx(self, host_pool, items, pages, staging) -> None:
        host_buf = host_pool.index_k_with_scale_buffer  # [pages, layers, 1, stride]
        per_page = host_buf[0].numel() * host_buf.element_size()
        chunk_pages = max(1, staging.numel() // per_page)
        page_list = pages.tolist()
        for start in range(0, len(page_list), chunk_pages):
            chunk = page_list[start : start + chunk_pages]
            k = len(chunk)
            view = (
                staging[: k * per_page]
                .view(host_buf.dtype)
                .view(k, *host_buf.shape[1:])
            )
            self._copy_pages(
                host_buf, 1, chunk, view, self._page_boundaries(host_buf, "idx")
            )
            tok0, tok1 = start * self.page_size, (start + k) * self.page_size
            for transfer, local_layer_id in items:
                target = self._layer_target(host_pool, transfer, local_layer_id)
                if target is None:
                    continue
                host_layer, device_layer = target
                dst = transfer.device_pool.index_k_with_scale_buffer[device_layer]
                dev_tok = transfer.device_indices[tok0:tok1]
                dev_pages = dev_tok.reshape(-1, self.page_size)[:, 0] // self.page_size
                src = view[:, host_layer].reshape(k, -1)
                dst.view(dst.shape[0], -1).index_copy_(0, dev_pages.to(dst.device), src)
            self.stats["bytes"] += k * per_page
