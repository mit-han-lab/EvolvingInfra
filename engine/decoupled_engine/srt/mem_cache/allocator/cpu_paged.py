"""Q4 (CPU page bookkeeping): the PD prefill paged allocator with
its free lists on the CPU (mem_cache/cpu_page_book.py has the overview).

``free_pages`` and ``release_pages`` keep their roles from the base class
(frees collect in ``release_pages`` until a merge sorts them into
``free_pages``) but are CPU int64 tensors. Allocations compute page ids on the
CPU, copy the token slots to the device with a pinned non-blocking H2D on the
current (scheduler) stream and tag the result with its page ids. Frees take
the page ids from the tag of what they free.

With the load quarantine (``clock`` set), a release whose epoch
is not completed goes to a dirty segment ``(epoch, pages)``; segments move to
``release_pages`` at the iteration start once their epoch is completed.
Forward allocations take dirty pages first (they are stream-ordered after
every earlier forward); loads take clean pages first and, if those do not
suffice, the oldest dirty pages with ``dirty_epoch`` (the newest epoch among
them, which the load must wait for).

``convert_to_cpu_paged`` switches an idle ``PagedTokenToKVPoolAllocator`` in
place (its references are held by the tree, the scheduler and the cache
controller).
"""

from __future__ import annotations

import dataclasses
from typing import Any, List, Optional, Sequence

import numpy as np
import torch

from decoupled_engine.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from decoupled_engine.srt.mem_cache.cpu_page_book import (
    STATS,
    expand_pages,
    extend_page_plan,
    extend_token_slots,
    pages_of,
    tag,
)
from decoupled_engine.srt.utils.common import is_pin_memory_available

_EMPTY = torch.empty((0,), dtype=torch.int64)


@dataclasses.dataclass
class DirtySegment:
    epoch: int
    pages: torch.Tensor  # CPU page ids


@dataclasses.dataclass
class LoadAllocation:
    indices: torch.Tensor  # device token slots, page-major, tagged
    indices_cpu: torch.Tensor  # the same slots on the CPU (transfers)
    page_ids_cpu: torch.Tensor
    # None: no quarantine (the load keeps the fence); 0: clean pages only;
    # > 0: the load must wait for forward_done[dirty_epoch].
    dirty_epoch: Optional[int]


class CpuPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
    """Built only by ``convert_to_cpu_paged``."""

    _clock = None  # load quarantine clock (QuarantineClock) or None
    _gen = None  # CPU page generations (numpy int64 [num_pages + 1])

    def _q4_setup(self, clock) -> None:
        self._clock = clock
        self._pin = is_pin_memory_available(self.device)
        # Page generations (symmetric P/D lease checks, paged.py generation_of) move
        # to the CPU with the free lists; the pool is idle, so this one read
        # at startup is the last device access they need.
        dev_gen = getattr(self, "_page_gen", None)
        if dev_gen is not None:
            self._gen = dev_gen.cpu().numpy().astype(np.int64, copy=True)
        self._page_gen = None
        self.clear()

    # ---- helpers ----

    def _to_device(self, cpu_tokens: torch.Tensor) -> torch.Tensor:
        """Pinned non-blocking copy on the current stream (the forward waits
        for the scheduler stream)."""
        src = cpu_tokens.pin_memory() if self._pin else cpu_tokens
        return src.to(self.device, non_blocking=self._pin)

    def _take_clean(self, n: int) -> torch.Tensor:
        pages = self.free_pages[:n]
        self.free_pages = self.free_pages[n:]
        return pages

    @property
    def dirty_count(self) -> int:
        return self._dirty_count

    def _release(self, pages: torch.Tensor) -> None:
        if len(pages) == 0:
            return
        if self.track_page_generations:
            # Every free (clean or dirty) ends the page's generation, as
            # paged.py _release_page_ids does on the device.
            self._gen[pages.numpy()] += 1
        clock = self._clock
        if clock is not None:
            epoch = clock.tag()
            if not clock.is_clean(epoch):
                self._dirty.append(DirtySegment(epoch, pages))
                self._dirty_count += len(pages)
                STATS["release_dirty"] += 1
                STATS["release_dirty_pages"] += len(pages)
                return
            STATS["release_clean"] += 1
        self.release_pages = torch.cat((pages, self.release_pages))

    def _consume_dirty_head(self, n: int) -> torch.Tensor:
        """The first ``n`` dirty pages in release order (forward allocations)."""
        taken = []
        while n > 0:
            seg = self._dirty[0]
            k = len(seg.pages)
            if k <= n:
                taken.append(seg.pages)
                self._dirty.pop(0)
                self._dirty_count -= k
                n -= k
            else:
                taken.append(seg.pages[:n])
                self._dirty[0] = DirtySegment(seg.epoch, seg.pages[n:])
                self._dirty_count -= n
                n = 0
        return taken[0] if len(taken) == 1 else torch.cat(taken)

    def _take_oldest_dirty(self, n: int) -> tuple[torch.Tensor, int]:
        """``n`` dirty pages from the oldest epochs (the shortest wait for a
        load) and the newest epoch among them."""
        order = sorted(range(len(self._dirty)), key=lambda i: self._dirty[i].epoch)
        taken, epoch, left, rest = [], 0, n, {}
        for i in order:
            if left == 0:
                break
            seg = self._dirty[i]
            k = min(left, len(seg.pages))
            taken.append(seg.pages[:k])
            epoch = max(epoch, seg.epoch)
            left -= k
            rest[i] = seg.pages[k:]
        assert left == 0
        self._dirty = [
            DirtySegment(seg.epoch, rest.get(i, seg.pages))
            for i, seg in enumerate(self._dirty)
            if len(rest.get(i, seg.pages))
        ]
        self._dirty_count -= n
        return (taken[0] if len(taken) == 1 else torch.cat(taken)), epoch

    def dirty_pages(self) -> torch.Tensor:
        if not self._dirty:
            return _EMPTY
        return torch.cat([seg.pages for seg in self._dirty])

    def free_page_ids_all(self) -> torch.Tensor:
        """Every free page id (clean and dirty), for checks."""
        return torch.cat((self.free_pages, self.release_pages, self.dirty_pages()))

    # ---- iteration (load quarantine) ----

    def begin_iteration(self) -> None:
        """Lockstep point at the loop top: segments whose epoch is completed
        become clean (prepended to release_pages; free_pages keeps its order)."""
        clock = self._clock
        if clock is None or not self._dirty:
            return
        keep, moved = [], []
        for seg in self._dirty:
            (moved if clock.is_clean(seg.epoch) else keep).append(seg)
        if not moved:
            return
        self._dirty = keep
        n = sum(len(seg.pages) for seg in moved)
        self._dirty_count -= n
        STATS["promoted_pages"] += n
        self.release_pages = torch.cat(
            [seg.pages for seg in moved] + [self.release_pages]
        )

    # ---- allocation ----

    def alloc(self, need_size: int):
        """Whole clean pages (never dirty ones: the caller's writes need not be
        ordered after the forward that released them)."""
        if self.debug_mode:
            assert need_size % self.page_size == 0
        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None
        pages = self._take_clean(num_pages)
        return tag(self._to_device(expand_pages(pages, self.page_size)), pages)

    def alloc_load(self, need_size: int) -> Optional[LoadAllocation]:
        """Destination pages of a HiCache load: clean first; with the
        quarantine and too few clean pages, the oldest dirty ones."""
        ps = self.page_size
        num_pages = need_size // ps
        if num_pages > len(self.free_pages) and len(self.release_pages) > 0:
            self.merge_and_sort_free()
        num_clean = min(num_pages, len(self.free_pages))
        num_dirty = num_pages - num_clean
        if num_dirty > self._dirty_count:
            return None
        pages = self._take_clean(num_clean)
        dirty_epoch = None if self._clock is None else 0
        if num_dirty:
            dirty, dirty_epoch = self._take_oldest_dirty(num_dirty)
            pages = torch.cat((pages, dirty)) if num_clean else dirty
            STATS["load_dirty_ops"] += 1
            STATS["load_dirty_pages"] += num_dirty
        STATS["load_ops"] += 1
        tokens = expand_pages(pages, ps)
        return LoadAllocation(
            indices=tag(self._to_device(tokens), pages),
            indices_cpu=tokens,
            page_ids_cpu=pages,
            dirty_epoch=dirty_epoch,
        )

    def alloc_extend_cpu(
        self,
        prefix_lens: Sequence[int],
        seq_lens: Sequence[int],
        last_pages: Sequence[int],
        extend_num_tokens: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """``alloc_extend`` with CPU inputs: ``last_pages[i]`` is the page of
        position ``prefix_lens[i] - 1`` when that page is partial (else
        ignored). Returns the device token slots (tagged with every page they
        touch) and the new pages in batch order; None if the pool is short."""
        ps = self.page_size
        bs = len(prefix_lens)
        num_new = extend_page_plan(prefix_lens, seq_lens, ps)
        total = sum(num_new)
        # Same merge trigger as the base kernel path (bs + 1 spare pages).
        if (
            self.need_sort
            and extend_num_tokens // ps + bs + 1
            > len(self.free_pages) + self._dirty_count
        ):
            self.merge_and_sort_free()
        if total > len(self.free_pages) + self._dirty_count:
            return None
        from_dirty = min(total, self._dirty_count)
        parts = []
        if from_dirty:
            parts.append(self._consume_dirty_head(from_dirty))
            STATS["extend_dirty_pages"] += from_dirty
        if total - from_dirty:
            parts.append(self._take_clean(total - from_dirty))
        new_pages = (
            _EMPTY if not parts else parts[0] if len(parts) == 1 else torch.cat(parts)
        )
        tokens = extend_token_slots(prefix_lens, seq_lens, last_pages, new_pages, ps)
        if len(tokens) != extend_num_tokens:
            raise AssertionError(
                f"Q4 alloc_extend_cpu: {len(tokens)} slots for {extend_num_tokens} tokens"
            )
        touched = []
        off = 0
        for pre, seq, last, n in zip(prefix_lens, seq_lens, last_pages, num_new):
            if seq <= pre:
                continue
            if pre % ps:
                touched.append(torch.tensor([int(last)], dtype=torch.int64))
            touched.append(new_pages[off : off + n])
            off += n
        out = self._to_device(tokens)
        return tag(out, torch.cat(touched) if touched else _EMPTY), new_pages

    def alloc_extend(self, *args, **kwargs):
        raise AssertionError("Q4 page bookkeeping: use alloc_extend_cpu")

    def alloc_decode(self, *args, **kwargs):
        raise AssertionError("Q4 page bookkeeping: no decode on the prefill worker")

    # ---- frees: page ids come from the tags ----

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        pages = pages_of(free_index, "allocator.free")
        if self.free_group is not None:
            self._group_pages.append(pages)
        else:
            self._release(pages)
        if self.debug_mode:
            self._debug_check_no_duplicate_pages()

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int):
        self.free(free_index)

    def free_full(self, free_index: torch.Tensor):
        self.free(free_index)

    def free_segments(self, segments):
        """Consecutive segments of one row: a page shared by a segment's end
        and the next one's start is freed once."""
        prev_last = None
        for free_index, _start_pos in segments:
            if free_index.numel() == 0:
                continue
            pages = pages_of(free_index, "allocator.free_segments")
            last = int(pages[-1])
            if prev_last is not None and int(pages[0]) == prev_last:
                pages = pages[1:]
            prev_last = last
            if self.free_group is not None:
                self._group_pages.append(pages)
            else:
                self._release(pages)
        if self.debug_mode:
            self._debug_check_no_duplicate_pages()

    def free_group_begin(self):
        self.free_group = []
        self._group_pages = []

    def free_group_end(self):
        self.free_group = None
        group, self._group_pages = self._group_pages, []
        if group:
            self._release(torch.cat(group))
        if self.debug_mode:
            self._debug_check_no_duplicate_pages()

    # ---- state ----

    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.sort(
                torch.cat((self.free_pages, self.release_pages))
            )[0]
            self.release_pages = _EMPTY
            STATS["merges"] += 1

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(1, self.num_pages + 1, dtype=torch.int64)
        self.release_pages = _EMPTY
        self.free_group = None
        self.free_page_reps_group = []
        self._group_pages: List[torch.Tensor] = []
        self._dirty: List[DirtySegment] = []
        self._dirty_count = 0
        # paged.py _reset_page_generations: page ids are 1-based (slot 0 is
        # the padded page); a clear() of a live pool frees every page.
        n = self.num_pages + 1
        if self._gen is None or len(self._gen) != n:
            self._gen = np.zeros(n, dtype=np.int64)
        else:
            self._gen += 1

    def available_size(self):
        # Dirty pages count: forward allocations can use them.
        return (
            len(self.free_pages) + len(self.release_pages) + self._dirty_count
        ) * self.page_size

    def clean_available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def _debug_check_no_duplicate_pages(self):
        pages = self.free_page_ids_all()
        assert len(torch.unique(pages)) == len(pages)

    def generation_of(self, page_indices: Sequence[int]) -> List[int]:
        """paged.py generation_of from the CPU copy: equal generations mean
        the page was not freed in between. No device read."""
        if len(page_indices) == 0:
            return []
        return self._gen[np.asarray(page_indices, dtype=np.int64)].tolist()

    # ---- decode leg: plain decode and spec decode runs ----

    def alloc_decode_cpu(
        self,
        rows: Any,
        req_pool_indices: Sequence[int],
        seq_lens_cpu: Sequence[int],
        token_per_req: int = 1,
    ) -> Optional[torch.Tensor]:
        """``alloc_decode`` from the CPU rows (``CpuPageRows``): positions
        ``[s, s + token_per_req)`` of every request, ``s = seq_lens_cpu[i]``
        (its row must hold ``[0, s)``). A position that starts a page gets a
        new one: dirty pages first, then the clean head, in batch order. One
        token makes the decode kernel's choice (and merge trigger), more make
        ``alloc_extend_cpu``'s. Every other position stays on the request's
        page in ``rows``. Writes the CPU rows, then copies the token slots
        (request-major, ``bs * token_per_req``) to the device with one pinned
        non-blocking H2D on the current stream; the result is tagged with the
        pages each request touches. None if the pool is short (rows and free
        lists unchanged)."""
        from decoupled_engine.srt.mem_cache.cpu_page_book import int64_array

        ps = self.page_size
        t = int(token_per_req)
        r = int64_array(req_pool_indices)
        s = int64_array(seq_lens_cpu)
        bs = len(s)
        # Merge triggers: the decode kernel's for one token (bs > free
        # pages), alloc_extend_cpu's (bs + 1 spare pages) for more.
        merge_at = bs if t == 1 else bs * t // ps + bs + 1
        if t != 1:
            return self._alloc_runs(rows, r, s, s + t, merge_at, "decode")
        self._check_rows_hold(rows, r, s, "decode")
        stay = s % ps != 0  # the position continues its page
        new_pages = self._take_forward_pages(bs - int(stay.sum()), merge_at, "decode")
        if new_pages is None:
            return None
        pages = rows.rows.numpy()[r, s // ps]
        pages[~stay] = new_pages
        rows.write_positions(r, s, pages)
        return self._tagged_slots(pages * ps + s % ps, pages)

    def alloc_runs_cpu(
        self,
        rows: Any,
        req_pool_indices: Sequence[int],
        starts: Sequence[int],
        ends: Sequence[int],
    ) -> Optional[torch.Tensor]:
        """``alloc_extend_cpu`` for runs ``[starts[i], ends[i])`` that continue
        rows the CPU rows hold up to ``starts[i]`` (spec decode: ``[cur,
        nxt)``), vectorised (no per-request Python or torch work): the same
        page choice (its merge trigger; dirty pages first, then the clean
        head, batch order), slot layout (request-major) and tag, with a run's
        partial first page taken from ``rows``. Writes the CPU rows (what
        ``assign_req_to_token_pool`` writes to the device rows), then one
        pinned non-blocking H2D on the current stream. None if the pool is
        short (rows and free lists unchanged)."""
        from decoupled_engine.srt.mem_cache.cpu_page_book import int64_array

        r = int64_array(req_pool_indices)
        s = int64_array(starts)
        e = int64_array(ends)
        # alloc_extend_cpu's merge trigger (the extend kernel's: bs + 1 spare
        # pages).
        merge_at = int((e - s).sum()) // self.page_size + len(s) + 1
        return self._alloc_runs(rows, r, s, e, merge_at, "spec_decode")

    def _alloc_runs(self, rows: Any, r, s, e, merge_at: int, stat: str):
        """Runs ``[s[i], e[i])`` of rows ``r[i]`` (numpy int64)."""
        import numpy as np

        from decoupled_engine.srt.mem_cache.cpu_page_book import run_pages

        ps = self.page_size
        self._check_rows_hold(rows, r, s, stat)
        if (e < s).any():
            raise AssertionError(f"Q4 {stat}: a run ends before it starts")
        total = int(((e + ps - 1) // ps - (s + ps - 1) // ps).sum())
        new_pages = self._take_forward_pages(total, merge_at, stat)
        if new_pages is None:
            return None
        table = rows.rows.numpy()
        idx, last, pages = run_pages(table, r, s, e, new_pages, ps)
        rows.write_positions(r[idx], last, pages)
        # Token slots, request-major, read back from the rows just written.
        n = e - s
        req = np.repeat(np.arange(len(s)), n)
        pos = s[req] + np.arange(len(req)) - (np.cumsum(n) - n)[req]
        return self._tagged_slots(table[r[req], pos // ps] * ps + pos % ps, pages)

    def _check_rows_hold(self, rows: Any, r, s, stat: str) -> None:
        """Every row holds the positions before its allocation (else a device
        row write was not mirrored and its pages would be unknown)."""
        import numpy as np

        held = rows.lens[r]
        short = np.flatnonzero(held < s)
        if short.size:
            i = int(short[0])
            raise AssertionError(
                f"Q4 {stat}: row {int(r[i])} holds {int(held[i])} positions, "
                f"the request has {int(s[i])} (a row write was not mirrored)"
            )

    def _take_forward_pages(self, total: int, merge_at: int, stat: str):
        """``total`` new pages of a forward allocation (numpy), or None if the
        pool is short: merge when the caller's trigger ``merge_at`` exceeds
        the free pages, then dirty pages first and the clean head
        (``alloc_extend_cpu``'s order). The array may view the free lists'
        storage: callers copy it into their own arrays and never write it."""
        if self.need_sort and merge_at > len(self.free_pages) + self._dirty_count:
            self.merge_and_sort_free()
        if total > len(self.free_pages) + self._dirty_count:
            return None
        from_dirty = min(total, self._dirty_count)
        parts = []
        if from_dirty:
            parts.append(self._consume_dirty_head(from_dirty))
            STATS[f"{stat}_dirty_pages"] += from_dirty
        if total - from_dirty:
            parts.append(self._take_clean(total - from_dirty))
        STATS[f"{stat}_ops"] += 1
        STATS[f"{stat}_new_pages"] += total
        if not parts:
            return _EMPTY.numpy()
        return (parts[0] if len(parts) == 1 else torch.cat(parts)).numpy()

    def _tagged_slots(self, slots, pages) -> torch.Tensor:
        """One pinned non-blocking H2D of the token slots, tagged."""
        return tag(self._to_device(torch.from_numpy(slots)), torch.from_numpy(pages))

    # ---- M4-SYM: destinations of RDMA writes (symmetric P/D import, PD decode) ----

    def alloc_import(
        self,
        prefix_len: int,
        fill_len: int,
        prefix_pages: Optional[torch.Tensor],
        *,
        allow_dirty: bool = False,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, Optional[int]]]:
        """Pages for positions ``[prefix_len, fill_len)`` of a request whose
        KV arrives by RDMA (the symmetric P/D reader's READ, a PD decode receive);
        ``prefix_len`` is page-aligned and ``prefix_pages`` (if given) are the
        reused prefix's pages.

        The transfer engine writes these pages outside every CUDA stream:
        nothing orders that write after a forward still reading or writing a
        page released at an epoch that is not completed. So clean pages first
        and, by default, clean pages only: None when they do not suffice, and
        the caller retries after the next promotions. ``allow_dirty``: the
        shortfall from the oldest dirty segments; ``dirty_epoch`` (their
        newest epoch) is then the forward the caller must wait for
        (``forward_done[dirty_epoch]``) before anything writes the pages.

        Returns ``(out, new_pages, dirty_epoch)``: ``out`` the device token
        slots (one pinned H2D on the current stream, tagged with
        ``new_pages``); ``dirty_epoch`` None without the quarantine, 0 for
        clean pages only. None if the pool is short (nothing changes)."""
        ps = self.page_size
        if prefix_len % ps:
            raise AssertionError(
                f"Q4 alloc_import: prefix {prefix_len} is not page-aligned (page size {ps})"
            )
        if prefix_pages is not None and len(prefix_pages) != prefix_len // ps:
            raise AssertionError(
                f"Q4 alloc_import: {len(prefix_pages)} prefix pages for {prefix_len} tokens"
            )
        num_tokens = max(fill_len - prefix_len, 0)
        num_pages = (prefix_len + num_tokens + ps - 1) // ps - prefix_len // ps
        if num_pages > len(self.free_pages) and len(self.release_pages) > 0:
            self.merge_and_sort_free()
        num_clean = min(num_pages, len(self.free_pages))
        num_dirty = num_pages - num_clean
        if num_dirty > self._dirty_count:
            return None
        if num_dirty and not allow_dirty:
            # Enough free pages, but some are dirty: wait for their promotion.
            STATS["import_wait_clean"] += 1
            return None
        pages = self._take_clean(num_clean)
        dirty_epoch = None if self._clock is None else 0
        if num_dirty:
            dirty, dirty_epoch = self._take_oldest_dirty(num_dirty)
            pages = torch.cat((pages, dirty)) if num_clean else dirty
            STATS["import_dirty_ops"] += 1
            STATS["import_dirty_pages"] += num_dirty
        STATS["import_ops"] += 1
        STATS["import_pages"] += num_pages
        tokens = expand_pages(pages, ps)[:num_tokens]
        return tag(self._to_device(tokens), pages), pages, dirty_epoch


def convert_to_cpu_paged(
    allocator: Any, clock: Any = None
) -> CpuPagedTokenToKVPoolAllocator:
    """Switch an idle plain paged allocator to CPU free lists, in place."""
    if type(allocator) is not PagedTokenToKVPoolAllocator:
        raise AssertionError(
            f"Q4 needs a plain PagedTokenToKVPoolAllocator, got {type(allocator).__name__}"
        )
    num_free = len(allocator.free_pages) + len(allocator.release_pages)
    if num_free != allocator.num_pages or allocator.free_group is not None:
        raise RuntimeError(
            f"Q4 page bookkeeping must start on an idle pool: {num_free} of "
            f"{allocator.num_pages} pages free"
        )
    allocator.__class__ = CpuPagedTokenToKVPoolAllocator
    allocator._q4_setup(clock)
    return allocator


def cpu_paged_allocator(allocator: Any) -> Optional[CpuPagedTokenToKVPoolAllocator]:
    """The allocator if it keeps CPU page bookkeeping (strict type check: mocks
    never count)."""
    return allocator if isinstance(allocator, CpuPagedTokenToKVPoolAllocator) else None


def load_quarantine_active(allocator: Any) -> bool:
    """True if ``allocator`` keeps CPU page bookkeeping with the load quarantine
    on (it has clean and dirty free pages)."""
    alloc = cpu_paged_allocator(allocator)
    return alloc is not None and alloc._clock is not None
