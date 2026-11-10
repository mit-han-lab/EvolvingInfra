"""Page quarantine for HiCache loads (called "M4" in comments and messages).

A HiCache load writes device KV pages on a loader stream. Today it waits for
the forward in flight at prepare time (the load fence and the producer's
``start_event``), because a page it gets from the allocator may still be read
or written by that forward. Quarantine makes that wait per page instead of per
load: every release is tagged with the newest forward that may still touch
the page, and loads get pages whose forward is known to be done ("clean").
Such a load needs no fence and starts while the forward runs ("Q3").

It builds on Q4, the CPU page bookkeeping (``cpu_page_book.py``,
``allocator/cpu_paged.py``): the free lists, the dirty segments and every page
id a load needs are on the CPU. Pieces:

- ``QuarantineClock``: pass-level lockstep epochs. ``launched`` is the
  scheduler's ``forward_ct`` of the newest launch. ``floor`` is set once per
  iteration of the overlap loop, before the C2g gate-pending reorder, to the
  epoch of the forward this iteration launches. Releases and lock -> 0
  transitions are tagged ``max(launched, floor)``. The reorder processes the
  previous result before the launch on some TP ranks and after it on others
  (it reads loader-thread state); with the floor, every rank tags the same
  pages with the same epoch. ``completed`` is the newest epoch whose result
  was processed after ``copy_done.synchronize()``; decisions use its value at
  the start of the iteration (``completed_snapshot``), so they are the same on
  every rank. No decision reads ``event.query()``.
- ``ForwardDoneRing``: an event recorded on ``forward_stream`` after
  every launch's forward (target, indexer and EAGLE draft extend). A slot keeps
  the epoch it holds; asking for an epoch newer than the slot is a bug.
- Clean and dirty containers in ``CpuPagedTokenToKVPoolAllocator``.
- Tree nodes keep ``unlock_epoch`` (``ComponentData``); evicted pages are
  released with it, not with the eviction time.
- Shared-loader loads ("Q3") carry ``dirty_epoch``; the loader skips the fence
  and ``start_event`` and waits on ``forward_done[dirty_epoch]`` only when the
  load took dirty pages (``cache_controller.py``, ``ce_loader.py``).

Flags: ``SGLANG_DSA_HOST_DEDUP_LOAD_QUARANTINE`` (default off; requires
``SGLANG_PREFILL_CPU_PAGE_BOOKKEEPING``).
"""

from __future__ import annotations

import contextlib
from typing import Any, List, Optional

from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.mem_cache.cpu_page_book import STATS


def load_quarantine_enabled() -> bool:
    return bool(envs.SGLANG_DSA_HOST_DEDUP_LOAD_QUARANTINE.get())


class QuarantineClock:
    """Lockstep epochs for page releases."""

    def __init__(self) -> None:
        self.launched = 0
        self.floor = 0
        self.completed = 0
        self.completed_snapshot = 0
        self._override: Optional[int] = None

    # -- driven by the overlap loop and run_batch --

    def begin_iteration(self) -> None:
        self.floor = 0
        self.completed_snapshot = self.completed

    def on_launch(self, forward_ct: int) -> None:
        self.launched = forward_ct

    def set_floor(self, forward_ct: int, launches: bool) -> None:
        """Set before the gate-pending reorder; same value on every rank."""
        self.floor = forward_ct + (1 if launches else 0)

    def on_result_processed(self, epoch: Optional[int]) -> None:
        if epoch is not None and epoch > self.completed:
            self.completed = epoch

    def on_all_processed(self, forward_ct: int) -> None:
        """Idle with nothing in flight: every launched forward is done."""
        self.completed = max(self.completed, forward_ct)

    # -- read by the allocator and the tree --

    def tag(self) -> int:
        if self._override is not None:
            return self._override
        return max(self.launched, self.floor)

    def is_clean(self, epoch: int) -> bool:
        return epoch <= self.completed_snapshot

    @contextlib.contextmanager
    def releasing_at(self, epoch: Optional[int]):
        """Tag the releases in this block with ``epoch`` (an evicted node's
        ``unlock_epoch``) instead of the current pass epoch."""
        if epoch is None:
            yield
            return
        prev, self._override = self._override, int(epoch)
        try:
            yield
        finally:
            self._override = prev


class ForwardDoneRing:
    """``forward_done[epoch]`` events, with the epoch each slot holds."""

    def __init__(self, size: int = 8) -> None:
        self.size = size
        self.slots: List[Optional[tuple[int, Any]]] = [None] * size

    def record(self, epoch: int, event: Any) -> None:
        self.slots[epoch % self.size] = (epoch, event)

    def event_for(self, epoch: int, clock: QuarantineClock) -> Optional[Any]:
        """The event a load must wait on before writing pages released at
        ``epoch``; None when that forward is known to be done."""
        if clock.is_clean(epoch):
            return None
        slot = self.slots[epoch % self.size]
        # An unrecorded or older slot would make the wait a silent no-op.
        assert (
            slot is not None and slot[0] >= epoch
        ), f"forward_done ring slot {slot and slot[0]} does not cover epoch {epoch}"
        # A newer slot is conservative: forwards complete in stream order.
        return slot[1]


class LoadQuarantine:
    """Owned by the PD prefill scheduler: the clock and the forward_done ring;
    the allocator (already CPU-paged with this clock) holds the containers."""

    def __init__(
        self,
        clock: QuarantineClock,
        allocator: Any,
        tree_cache: Any,
        device_module: Any,
    ) -> None:
        self.clock = clock
        self.ring = ForwardDoneRing()
        self.allocator = allocator
        self.device_module = device_module
        attach_tree(tree_cache, clock)
        cc = getattr(tree_cache, "cache_controller", None)
        if cc is not None:
            cc.load_quarantine = self

    def begin_iteration(self) -> None:
        self.clock.begin_iteration()
        self.allocator.begin_iteration()

    def record_forward_done(self, epoch: int, forward_stream: Any) -> None:
        ev = self.device_module.Event()
        ev.record(forward_stream)
        self.ring.record(epoch, ev)

    def dirty_event(self, dirty_epoch: Optional[int]) -> Optional[Any]:
        """What a quarantined load waits on (None: nothing)."""
        if not dirty_epoch:
            return None
        ev = self.ring.event_for(dirty_epoch, self.clock)
        STATS["load_dirty_wait" if ev is not None else "load_dirty_done"] += 1
        return ev

    def stats_fields(self) -> str:
        return (
            f"dirty_pages={self.allocator.dirty_count} "
            f"completed={self.clock.completed} launched={self.clock.launched}"
        )


def attach_tree(tree_cache: Any, clock: QuarantineClock) -> None:
    """The tree stamps unlock epochs and new-node defaults with ``clock``."""
    from decoupled_engine.srt.mem_cache.unified_cache.component_type import ComponentType

    tree_core = getattr(tree_cache, "tree_core", None)
    if tree_core is not None:
        tree_core._lq_clock = clock
    components = getattr(tree_cache, "components", None)
    if components is not None:
        full = (
            components.get(ComponentType.FULL) if hasattr(components, "get") else None
        )
        if full is not None:
            full._lq_clock = clock
