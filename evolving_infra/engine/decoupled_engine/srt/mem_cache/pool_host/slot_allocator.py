"""PageSlotAllocator — free-list bookkeeping for allocator-owning host pools,
with slot-level pins for the kv-p2p PeerKVServicer.

Why this exists. Every TP rank runs its own scheduler with its
own copy of the radix tree and host pools. Stock SGLang keeps the ranks in
lock-step by feeding them identical inputs; anything asynchronous is
MIN/MAX-reduced before it may touch the tree. The peer-KV servicer thread is
asynchronous and rank-local (remote LOOKUP/RELEASE, wall-clock TTL), so it
must never influence a structural decision. Instead of gating *eviction* on
leases (rank-divergent), the tree evicts lease-blind and this allocator keeps
the freed pages a remote reader may still be RDMA-reading out of the free
lists until the lease ends (rank-local, invisible to decisions).

Invariants:

  I1 consensus   available_size() == num_slots - allocated_slots - R0
                 on every rank. ``allocated`` only changes via alloc/free on
                 the scheduler thread; R0 (the reserve) is fixed at init from
                 rank-uniform inputs. Every servicer action (pin, quarantine
                 on free, unpin, cooling) only re-partitions *non-allocated*
                 pages among {free_slots, release_slots, withheld} and cancels
                 out of available_size().
  I2 safety      A page with pin_count > 0, or with an armed cool_until in the
                 future, never enters free_slots/release_slots. try_pin
                 accepts a page only if it is allocated and its generation
                 matches the generation recorded when the page was advertised,
                 checked under the same lock free() takes.
  I3 sufficiency The number of *protected* pages (pinned, armed, or withheld)
                 never exceeds reserve_pages — enforced at pin time — so
                 ``n <= available_size()`` implies the free lists really hold
                 n slots. A violation is a bug and asserts (rank-local crash,
                 never a silent None on one rank).
  I4 blindness   No scheduler-thread decision reads servicer state. The only
                 scheduler -> servicer interface is a queue put plus one
                 page_generation() read under the pool lock at store time.

Granule = the pool's allocation unit in slots (page_size for the DeepSeek-V4
anchors). All indices passed in are slot indices; pages are groups of
``granule`` consecutive slots starting at a multiple of ``granule``.

Thread model: every method must be called with ``lock`` held (the pool's
RLock — the pool's ``@synchronized`` wrappers do that for the scheduler-thread
API; the servicer acquires the same lock explicitly for pin/unpin/drain).
"""

from __future__ import annotations

import heapq
import logging
import math
import threading
import time
from typing import Callable, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


class PageSlotAllocator:
    def __init__(
        self,
        num_slots: int,
        granule: int,
        reserve_pages: int = 0,
        lock: Optional[threading.RLock] = None,
        *,
        name: str = "",
    ) -> None:
        if granule <= 0 or num_slots % granule != 0:
            raise ValueError(
                f"PageSlotAllocator({name}): num_slots={num_slots} must be a "
                f"positive multiple of granule={granule}"
            )
        self.name = name
        self.num_slots = int(num_slots)
        self.granule = int(granule)
        self.num_pages = self.num_slots // self.granule
        if reserve_pages < 0 or reserve_pages > self.num_pages:
            raise ValueError(
                f"PageSlotAllocator({name}): reserve_pages={reserve_pages} out of "
                f"range [0, {self.num_pages}]"
            )
        self.reserve_pages = int(reserve_pages)
        self.reserve_slots = self.reserve_pages * self.granule
        self.lock = lock if lock is not None else threading.RLock()

        # Free-list state (as in the stock pools).
        self.free_slots: torch.Tensor
        self.release_slots: List[torch.Tensor]
        self.num_release_slots: int

        # Per-page state. ``gen`` is monotonic across clear() so a shadow entry
        # recorded before a flush can never validate afterwards.
        self.used = torch.zeros(self.num_pages, dtype=torch.bool)
        self.gen = torch.zeros(self.num_pages, dtype=torch.int64)
        self._pin_state_ready = False
        self.pin_count: torch.Tensor
        self.protected: torch.Tensor  # pinned | armed | withheld
        self.withheld: torch.Tensor  # freed while protected; not in free lists
        self.cool_until: torch.Tensor  # monotonic seconds; 0 = not armed

        # Counters (pages).
        self.num_used = 0
        self.num_protected = 0
        self.num_withheld = 0
        # Min-heaps of (release_after_mono, page_id).
        self._cooling: List[Tuple[float, int]] = []
        self._armed: List[Tuple[float, int]] = []

        # Diagnostics.
        self.stats_pin_refused_stale = 0
        self.stats_pin_refused_reserve = 0
        self.stats_partial_pins = 0
        self.stats_quarantined_total = 0
        self.stats_cooled_total = 0
        # Optional hook fired with the slot tensor of pages entering the free
        # lists (debug poison detector); never fired for protected pages.
        self.poison_hook: Optional[Callable[[torch.Tensor], None]] = None

        self.clear()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _ensure_pin_state(self) -> None:
        if self._pin_state_ready:
            return
        self.pin_count = torch.zeros(self.num_pages, dtype=torch.int32)
        self.protected = torch.zeros(self.num_pages, dtype=torch.bool)
        self.withheld = torch.zeros(self.num_pages, dtype=torch.bool)
        self.cool_until = torch.zeros(self.num_pages, dtype=torch.float64)
        self._pin_state_ready = True

    def _pages_of(self, indices: torch.Tensor) -> torch.Tensor:
        """Slot indices (whole pages, each page's slots consecutive) -> page ids."""
        indices = indices.to(dtype=torch.int64, device="cpu").flatten()
        if indices.numel() % self.granule != 0:
            raise ValueError(
                f"PageSlotAllocator({self.name}): {indices.numel()} slots is not a "
                f"multiple of granule={self.granule}"
            )
        if self.granule == 1:
            return indices
        heads = indices.view(-1, self.granule)[:, 0]
        if bool((heads % self.granule).any()):
            raise ValueError(
                f"PageSlotAllocator({self.name}): slot run not page-aligned "
                f"(granule={self.granule})"
            )
        return heads // self.granule

    def _slots_of(self, pages: torch.Tensor) -> torch.Tensor:
        if self.granule == 1:
            return pages.to(torch.int64)
        base = pages.to(torch.int64) * self.granule
        return (base[:, None] + torch.arange(self.granule, dtype=torch.int64)).reshape(-1)

    def _merge_release_slots(self) -> None:
        if self.num_release_slots == 0:
            return
        if len(self.free_slots) == 0 and len(self.release_slots) == 1:
            self.free_slots = self.release_slots[0]
        else:
            self.free_slots = torch.cat([self.free_slots, *self.release_slots])
        self.release_slots = []
        self.num_release_slots = 0

    def _push_release(self, pages: torch.Tensor) -> None:
        if pages.numel() == 0:
            return
        slots = self._slots_of(pages)
        if self.poison_hook is not None:
            self.poison_hook(slots)
        self.release_slots.append(slots)
        self.num_release_slots += slots.numel()

    def check_invariants(self) -> None:
        free = len(self.free_slots) + self.num_release_slots
        assert (
            free + (self.num_withheld + self.num_used) * self.granule == self.num_slots
        ), (
            f"PageSlotAllocator({self.name}) accounting broken: free={free} "
            f"withheld={self.num_withheld} used={self.num_used} granule={self.granule} "
            f"num_slots={self.num_slots}"
        )
        assert self.num_withheld <= self.num_protected <= self.reserve_pages, (
            f"PageSlotAllocator({self.name}) reserve overrun: withheld={self.num_withheld} "
            f"protected={self.num_protected} reserve={self.reserve_pages}"
        )

    # ------------------------------------------------------------------
    # scheduler-thread API (pool alloc/free/clear/available_size)
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Flush. Allocated pages become free; protected pages stay out of the
        free lists (pinned -> quarantine, armed -> cooling); generations keep
        counting so pre-flush shadow entries never validate again."""
        if self._pin_state_ready and self.num_protected > 0:
            prot_pages = torch.nonzero(self.protected, as_tuple=False).flatten()
            # Protected pages that were still allocated are freed by the flush:
            # pinned -> quarantine; armed -> cooling.
            newly = prot_pages[~self.withheld[prot_pages]]
            for p in newly.tolist():
                if int(self.pin_count[p]) == 0:
                    heapq.heappush(self._cooling, (float(self.cool_until[p]), p))
                    self.stats_cooled_total += 1
                else:
                    self.stats_quarantined_total += 1
            self.withheld[newly] = True
            self.num_withheld += int(newly.numel())
            keep = ~self.protected
            # Armed entries of now-withheld pages are superseded by cooling ones.
            self._armed = [(t, p) for (t, p) in self._armed if not bool(self.withheld[p])]
            heapq.heapify(self._armed)
        else:
            keep = torch.ones(self.num_pages, dtype=torch.bool)
        self.used.zero_()
        self.num_used = 0
        free_pages = torch.nonzero(keep, as_tuple=False).flatten()
        self.free_slots = self._slots_of(free_pages)
        self.release_slots = []
        self.num_release_slots = 0
        self.check_invariants()

    def available_size(self) -> int:
        return (
            len(self.free_slots)
            + self.num_release_slots
            + self.num_withheld * self.granule
            - self.reserve_slots
        )

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """``need_size`` slots (already rounded/validated by the pool)."""
        if need_size <= 0:
            return torch.empty(0, dtype=torch.int64)
        if need_size > self.available_size():
            return None
        if need_size > len(self.free_slots):
            self._merge_release_slots()
        # I3: the reserve guarantees the free lists back available_size().
        assert len(self.free_slots) >= need_size, (
            f"PageSlotAllocator({self.name}): available_size()={self.available_size()} "
            f">= {need_size} but free lists hold {len(self.free_slots)} slots "
            f"(withheld={self.num_withheld} pages, reserve={self.reserve_pages})"
        )
        select = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        pages = self._pages_of(select)
        assert not bool(self.used[pages].any()), (
            f"PageSlotAllocator({self.name}): double-alloc of pages "
            f"{pages[self.used[pages]].tolist()}"
        )
        self.used[pages] = True
        self.gen[pages] += 1
        self.num_used += int(pages.numel())
        return select

    def free(self, indices: torch.Tensor) -> int:
        indices = indices.to(dtype=torch.int64, device="cpu").flatten()
        if indices.numel() == 0:
            return 0
        pages = self._pages_of(indices)
        assert bool(self.used[pages].all()), (
            f"PageSlotAllocator({self.name}): double-free of pages "
            f"{pages[~self.used[pages]].tolist()}"
        )
        self.used[pages] = False
        self.num_used -= int(pages.numel())
        if self._pin_state_ready and self.num_protected > 0:
            prot = self.protected[pages]
            if bool(prot.any()):
                held = pages[prot]
                for p in held.tolist():
                    if int(self.pin_count[p]) == 0:
                        # Armed (lease expired while the page was still allocated).
                        heapq.heappush(self._cooling, (float(self.cool_until[p]), p))
                        self.stats_cooled_total += 1
                    else:
                        self.stats_quarantined_total += 1
                self.withheld[held] = True
                self.num_withheld += int(held.numel())
                pages = pages[~prot]
        self._push_release(pages)
        return int(indices.numel())

    # ------------------------------------------------------------------
    # servicer-thread API (under the same lock)
    # ------------------------------------------------------------------

    def page_generation(self, indices: torch.Tensor) -> List[int]:
        """Generation per page of ``indices`` (scheduler thread, store time)."""
        return self.gen[self._pages_of(indices)].tolist()

    def try_pin(self, indices: torch.Tensor, gens: Sequence[int]) -> int:
        """Pin the longest prefix of pages that is still allocated with the
        recorded generation and fits the reserve. Returns the number of pages
        pinned (0 = nothing pinned). Pages are pinned in order, so the reader's
        contiguous-prefix lease model is preserved."""
        self._ensure_pin_state()
        pages = self._pages_of(indices)
        n = int(pages.numel())
        if n != len(gens):
            raise ValueError(
                f"PageSlotAllocator({self.name}): {n} pages but {len(gens)} generations"
            )
        used = self.used[pages].tolist()
        gen_now = self.gen[pages].tolist()
        prot = self.protected[pages].tolist()
        budget = self.reserve_pages - self.num_protected
        accepted = 0
        new_protected = 0
        seen: set = set()
        for i in range(n):
            p = int(pages[i])
            if not used[i] or gen_now[i] != int(gens[i]):
                self.stats_pin_refused_stale += 1
                break
            if not prot[i] and p not in seen:
                if new_protected + 1 > budget:
                    self.stats_pin_refused_reserve += 1
                    break
                new_protected += 1
                seen.add(p)
            accepted = i + 1
        if accepted == 0:
            return 0
        if accepted < n:
            self.stats_partial_pins += 1
        acc = pages[:accepted]
        self.pin_count[acc] += 1
        self.protected[acc] = True
        self.num_protected += new_protected
        return accepted

    def unpin(
        self, indices: torch.Tensor, cool_s: float = 0.0, now: Optional[float] = None
    ) -> None:
        """Drop one pin per page. Pages reaching zero pins are released to the
        free lists (if withheld) or unprotected (if still allocated), unless
        ``cool_s > 0`` arms a cooling window first (expiry path: a reader may
        still be finishing a read that started before the TTL ran out)."""
        self._ensure_pin_state()
        pages = self._pages_of(indices)
        if now is None:
            now = time.monotonic()
        release_pages: List[int] = []
        for p in pages.tolist():
            cnt = int(self.pin_count[p]) - 1
            assert cnt >= 0, f"PageSlotAllocator({self.name}): unpin below zero on page {p}"
            self.pin_count[p] = cnt
            if cnt > 0:
                continue
            until = max(float(self.cool_until[p]), now + cool_s if cool_s > 0 else 0.0)
            if until > now:
                self.cool_until[p] = until
                if bool(self.withheld[p]):
                    heapq.heappush(self._cooling, (until, p))
                    self.stats_cooled_total += 1
                else:
                    heapq.heappush(self._armed, (until, p))
                continue
            # No cooling: unprotect now.
            self.cool_until[p] = 0.0
            self.protected[p] = False
            self.num_protected -= 1
            if bool(self.withheld[p]):
                self.withheld[p] = False
                self.num_withheld -= 1
                release_pages.append(p)
        if release_pages:
            self._push_release(torch.tensor(release_pages, dtype=torch.int64))

    def drain_cooling(self, now: Optional[float] = None) -> int:
        """Release withheld pages whose cooling window passed and unprotect
        armed pages whose window passed. Returns pages released to the free lists."""
        if not self._pin_state_ready:
            return 0
        if now is None:
            now = time.monotonic()
        release_pages: List[int] = []
        while self._cooling and self._cooling[0][0] <= now:
            until, p = heapq.heappop(self._cooling)
            if float(self.cool_until[p]) > until:
                continue  # a later cooling entry for this page supersedes it
            assert bool(self.withheld[p]) and int(self.pin_count[p]) == 0
            self.cool_until[p] = 0.0
            self.protected[p] = False
            self.withheld[p] = False
            self.num_protected -= 1
            self.num_withheld -= 1
            release_pages.append(p)
        while self._armed and self._armed[0][0] <= now:
            until, p = heapq.heappop(self._armed)
            if float(self.cool_until[p]) > until:
                continue  # a later arm supersedes this entry
            if not bool(self.used[p]) or int(self.pin_count[p]) > 0:
                continue  # freed (now in cooling) or pinned again
            if not bool(self.protected[p]):
                continue  # already unprotected
            self.cool_until[p] = 0.0
            self.protected[p] = False
            self.num_protected -= 1
        if release_pages:
            self._push_release(torch.tensor(release_pages, dtype=torch.int64))
        return len(release_pages)

    def stats(self) -> dict:
        return {
            "reserve_pages": self.reserve_pages,
            "protected_pages": self.num_protected,
            "withheld_pages": self.num_withheld,
            "pinned_pages": (
                int((self.pin_count > 0).sum()) if self._pin_state_ready else 0
            ),
            "cooling_pages": len(self._cooling),
            "armed_pages": len(self._armed),
            "used_pages": self.num_used,
            "pin_refused_stale": self.stats_pin_refused_stale,
            "pin_refused_reserve": self.stats_pin_refused_reserve,
            "partial_pins": self.stats_partial_pins,
            "quarantined_total": self.stats_quarantined_total,
            "cooled_total": self.stats_cooled_total,
        }


def reserve_pages_for(num_pages: int, ratio: float) -> int:
    """Rank-uniform reserve size: ceil(num_pages * ratio), clamped to [0, num_pages]."""
    if ratio <= 0:
        return 0
    return max(0, min(num_pages, int(math.ceil(num_pages * ratio))))
