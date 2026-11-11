"""Unserialize the PD prefill overlap loop.

In ``event_loop_overlap_disagg_prefill`` the scheduler launches ``B_t`` and then
calls ``_apply_war_barrier()``: ``schedule_stream.wait_stream(forward_stream)``.
The scheduler thread runs with ``schedule_stream`` as its current stream, so
every blocking device read it issues afterwards (``.cpu()``, ``.to("cpu")``,
``torch.unique``, a non-blocking=False copy) waits for ``forward(B_t)``.
``process_batch_result(B_{t-1})`` runs right after the barrier, so the loop
used to be serial: ``B_{t+1}`` was admitted only after ``forward(B_t)`` ended,
and the KV send of ``B_{t-1}`` waited one extra forward.

Every GPU ordering stays as it is (barrier, load fence, C2g gate). Only the
CPU waits behind the barrier move:

- 6a ``SGLANG_PREFILL_ASYNC_DRAFT_D2H``: the EAGLE draft seed (hidden states,
  top-k p / index, DSA top-k) is copied with the async result D2H at launch.
- 6b ``SGLANG_PREFILL_KV_PAGE_SNAPSHOT``: the send's KV page ids are read on a
  side stream that waits for an event recorded on ``schedule_stream`` right
  before the barrier (so after every scheduler write that admission made) and
  not for the forward. Rows rewritten after that event fall back to the
  legacy read.
- 6c / 6c-ev ``SGLANG_PREFILL_FIXED_SHAPE_FREE``: eviction frees tree values
  with ``free_segment`` (no ``torch.unique``).
- 6d ``SGLANG_PREFILL_EARLY_SEND_ASYNC``: the early send copies its page ids
  asynchronously at launch and is issued after the launch, at a fixed point
  of the loop (same point on every TP rank).
- 6g ``SGLANG_MOONCAKE_EARLY_SEND_WAIT``: the Mooncake transfer worker waits
  for the early send's ``_early_send_wait_event`` on its own thread before
  posting the RDMA read (only Mori did). 6d already orders the send after that
  forward when 6d is on; 6g also covers sends that keep the legacy path.
- 6h ``SGLANG_PREFILL_DEFER_WRITE_SUBMIT``: write-through submissions made
  behind the barrier (``move_indices`` does a blocking index D2H for the
  ``direct`` IO backend) run after the next launch instead. In an iteration
  that launches nothing they run right after its result is processed, never
  inside that processing.

``SGLANG_PREFILL_UNSERIALIZE_LOOP`` turns on every sub-flag that is not set
explicitly. Loop changes default to off. Eligible FULL-cache eviction uses
fixed-shape frees independently; its explicit sub-flag still allows rollback.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections import Counter
from typing import Any, Callable, List, Optional

import numpy as np
import torch

from decoupled_engine.srt.environ import envs

logger = logging.getLogger(__name__)


def _sub_flag(flag) -> bool:
    """An explicitly set sub-flag wins; otherwise follow the umbrella."""
    if flag.is_set():
        return bool(flag.get())
    return bool(envs.SGLANG_PREFILL_UNSERIALIZE_LOOP.get())


def fixed_shape_free_enabled(*, default: bool = False) -> bool:
    """Allow a safe local default without enabling other M2 loop changes."""
    flag = envs.SGLANG_PREFILL_FIXED_SHAPE_FREE
    if flag.is_set():
        return bool(flag.get())
    return default or bool(envs.SGLANG_PREFILL_UNSERIALIZE_LOOP.get())


def mooncake_early_send_wait_enabled() -> bool:
    return _sub_flag(envs.SGLANG_MOONCAKE_EARLY_SEND_WAIT)


def overlap_gate_pending_reorder_enabled() -> bool:
    """The aggregated overlap loop's gate-pending reorder (default off)."""
    return _sub_flag(envs.SGLANG_OVERLAP_GATE_PENDING_REORDER)


def sender_waits_on_early_send_event(sender: Any) -> bool:
    """Whether the KV sender's connector waits on ``_early_send_wait_event``
    before it reads the pages (Mooncake with 6g on)."""
    return getattr(getattr(sender, "kv_mgr", None), "_early_send_wait", False) is True


@dataclasses.dataclass(frozen=True)
class PrefillUnserializeConfig:
    async_draft_d2h: bool = False
    kv_page_snapshot: bool = False
    kv_page_snapshot_check: bool = False
    fixed_shape_free: bool = False
    early_send_async: bool = False
    defer_write_submit: bool = False

    @classmethod
    def from_env(cls) -> PrefillUnserializeConfig:
        snapshot = _sub_flag(envs.SGLANG_PREFILL_KV_PAGE_SNAPSHOT)
        return cls(
            async_draft_d2h=_sub_flag(envs.SGLANG_PREFILL_ASYNC_DRAFT_D2H),
            kv_page_snapshot=snapshot,
            # Debug only: never implied by the umbrella.
            kv_page_snapshot_check=snapshot
            and bool(envs.SGLANG_PREFILL_KV_PAGE_SNAPSHOT_CHECK.get()),
            fixed_shape_free=fixed_shape_free_enabled(),
            early_send_async=_sub_flag(envs.SGLANG_PREFILL_EARLY_SEND_ASYNC),
            defer_write_submit=_sub_flag(envs.SGLANG_PREFILL_DEFER_WRITE_SUBMIT),
        )

    @property
    def any_enabled(self) -> bool:
        return any(dataclasses.astuple(self))


# Loop counters (one line per minute, TP0).
STATS: Counter = Counter()


class StatsLogger:
    def __init__(self, tp_rank: int, period_s: float = 60.0):
        self.tp_rank = tp_rank
        self.period_s = period_s
        self._last = time.monotonic()

    def maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last < self.period_s:
            return
        self._last = now
        if self.tp_rank == 0 and STATS:
            logger.info(
                "M2 prefill-loop stats: %s",
                " ".join(f"{k}={v}" for k, v in sorted(STATS.items())),
            )


def _legacy_page_ids(view: torch.Tensor, page_size: int) -> np.ndarray:
    from decoupled_engine.srt.mem_cache.common import kv_to_page_indices

    return kv_to_page_indices(view, page_size)


class KvPageIdReader:
    """6b: page ids of ``req_to_token`` rows without waiting for the forward.

    ``mark_pre_barrier`` records an event on ``schedule_stream`` just before
    ``_apply_war_barrier``. Up to that point ``schedule_stream`` holds the
    previous barrier (the previous forward, whose result is processed in this
    iteration and which ``copy_done`` already waits for) plus the admission
    writes of the batch just launched. A read on the side stream after that
    event therefore sees every row the scheduler wrote before the launch, and
    blocks the CPU only until those writes ran, not until the new forward
    ends.

    Rows written after the event (``cache_unfinished_req`` repointing a
    request's rows at deduplicated tree pages during result processing) are
    detected through the pool's row-write sequence and use the legacy read on
    ``schedule_stream``, which orders after them. The pre-rewrite pages were
    freed by that insert, so sending them would be a correctness bug, not just
    a stale read.
    """

    def __init__(
        self,
        device_module: Any,
        req_to_token_pool: Any,
        page_size: int,
        check: bool = False,
    ):
        self.device_module = device_module
        self.pool = req_to_token_pool
        self.page_size = page_size
        self.check = check
        self.stream = device_module.Stream()
        self.event = None
        self.event_seq = 0
        self.pool.enable_row_write_tracking()

    def begin_iteration(self) -> None:
        # Admission of this iteration may rewrite rows before the next event.
        self.event = None

    def mark_pre_barrier(self, schedule_stream) -> None:
        ev = self.device_module.Event()
        ev.record(schedule_stream)
        self.event = ev
        self.event_seq = self.pool.row_write_seq()

    def page_ids(self, req_pool_idx: int, start: int, end: int) -> np.ndarray:
        view = self.pool.req_to_token[req_pool_idx, start:end]
        ev = self.event
        if ev is None:
            # No forward was launched behind a barrier in this iteration: the
            # legacy read on schedule_stream does not wait for a running forward.
            STATS["kv_ids_legacy_no_event"] += 1
            return _legacy_page_ids(view, self.page_size)
        if self.pool.row_written_after(req_pool_idx, self.event_seq):
            STATS["kv_ids_legacy_rewritten"] += 1
            return _legacy_page_ids(view, self.page_size)
        self.stream.wait_event(ev)
        with self.device_module.stream(self.stream):
            ids = _legacy_page_ids(view, self.page_size)
        STATS["kv_ids_side_stream"] += 1
        if self.check:
            ref = _legacy_page_ids(view, self.page_size)
            if not np.array_equal(ids, ref):
                STATS["kv_ids_check_mismatch"] += 1
                logger.error(
                    "M2 KV page-id snapshot mismatch: row=%d range=[%d,%d) "
                    "snapshot=%s legacy=%s; using legacy",
                    req_pool_idx,
                    start,
                    end,
                    ids[:8].tolist(),
                    ref[:8].tolist(),
                )
                return ref
            STATS["kv_ids_check_ok"] += 1
        return ids


@dataclasses.dataclass
class PendingEarlySend:
    """6d: an early send whose page ids are being copied asynchronously."""

    req: Any
    start_idx: int
    end_idx: int
    page_ids_cpu: torch.Tensor
    ready_event: Any
    wait_event: Any


def snapshot_early_send_page_ids(
    req_to_token: torch.Tensor,
    req_pool_idx: int,
    start: int,
    end: int,
    page_size: int,
    async_d2h: Callable[[torch.Tensor], torch.Tensor],
    event_factory: Callable[[], Any],
) -> tuple[torch.Tensor, Any]:
    """Enqueue the page-id gather + D2H on the current stream; return the
    pinned host tensor and an event recorded after the copy."""
    ids = req_to_token[req_pool_idx, start:end][::page_size] // page_size
    ids_cpu = async_d2h(ids)
    ready = event_factory()
    ready.record()
    return ids_cpu, ready


def flush_pending_early_sends(
    pending: List[PendingEarlySend],
    send: Callable[[PendingEarlySend], None],
    is_aborted: Callable[[Any], bool],
) -> None:
    """Issue deferred early sends in order, each only after the forward that
    may still write its prefix pages has finished. ``wait_event`` was recorded
    on ``forward_stream`` when the send was deferred (before this launch's
    forward), so it covers that forward. Under the coarse WAR barrier
    ``ready_event`` already implies it; waiting on ``wait_event`` too keeps
    the order when the barrier waits only for a read-done event. This orders
    the RDMA read for connectors that ignore ``_early_send_wait_event``
    (Mooncake without 6g). These are synchronize() calls, never query()."""
    for p in pending:
        if p.ready_event is not None:  # None: the ids came from the CPU (Q4)
            p.ready_event.synchronize()
        if p.wait_event is not None:
            p.wait_event.synchronize()
        req = p.req
        if is_aborted(req) or req.pending_bootstrap:
            STATS["early_send_dropped"] += 1
            continue
        if req.start_send_idx != p.start_idx:
            # Nothing between the launch and this flush sends KV; keep the
            # legacy behaviour (blocking read) if that ever changes.
            STATS["early_send_start_moved"] += 1
            p = dataclasses.replace(p, page_ids_cpu=None)
        send(p)
        STATS["early_send_flushed"] += 1


def page_ids_from_snapshot(
    p: Optional[PendingEarlySend], start_idx: int, end_idx: int
) -> Optional[np.ndarray]:
    if p is None or p.page_ids_cpu is None:
        return None
    if start_idx != p.start_idx or end_idx != p.end_idx:
        return None
    return p.page_ids_cpu.numpy()


# ---- 6a-sym: the EAGLE seed of symmetric P/D symmetric exports ----
#
# A symmetric P-leg publishes FINAL from the aggregated loop's result
# processing (batch_result_processor.process_batch_result_prefill), behind the
# WAR barrier. Its EAGLE seed used to be read there with blocking D2H copies
# (hidden states, DSA top-k), and MetadataBuffers.set_buf copied top-k p /
# index from the device into the host aux slots. With 6a on, run_batch copies
# the seed of every launch that may finish an export with the async result
# D2H, as the PD prefill loop does.


def symmetric_async_draft_seed(scheduler: Any) -> bool:
    """6a for symmetric P/D SYMMETRIC: SGLANG_PREFILL_ASYNC_DRAFT_D2H (or the
    umbrella) on an overlap EAGLE scheduler."""
    spec = getattr(scheduler, "spec_algorithm", None)
    return bool(
        PrefillUnserializeConfig.from_env().async_draft_d2h
        and getattr(scheduler, "enable_overlap", False)
        and spec is not None
        and spec.is_eagle()
    )


def may_finish_export(batch: Any) -> bool:
    """A launch that may carry an export's last chunk: an extend batch with an
    export request. Its result publishes FINAL with the EAGLE seed. A superset
    is harmless (an unused staged copy is dropped with the result)."""
    return batch.forward_mode.is_extend() and any(
        getattr(req, "export_mark", None) is not None for req in batch.reqs
    )


def metadata_on_host(metadata_buffers: Any) -> bool:
    """The aux slots that carry the seed are host memory (the default;
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK keeps them on the GPU).
    Host seed rows then copy into them without a device read."""
    buf = getattr(metadata_buffers, "output_topk_p", None)
    return buf is not None and buf.device.type == "cpu"
