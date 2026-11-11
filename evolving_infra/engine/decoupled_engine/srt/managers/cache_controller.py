# Adapted from SGLang python/sglang/srt/managers/cache_controller.py (commit 9a489f8d2f), modified for evolving-infra.
from __future__ import annotations

"""
Copyright 2023-2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import logging
import threading
import time
import uuid
from dataclasses import dataclass
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Callable, List, NamedTuple, Optional

import torch

from decoupled_engine.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
    count_pool_hits,
)

if TYPE_CHECKING:
    from decoupled_engine.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from decoupled_engine.srt.mem_cache.pool_host import HostKVCache

from decoupled_engine.srt.layers.dp_attention import (
    get_attention_dp_rank,
    is_dp_attention_enabled,
)
from decoupled_engine.srt.managers.scheduler_components.phase_timers import timed_phase
from decoupled_engine.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from decoupled_engine.srt.mem_cache.memory_pool import MLATokenToKVPool
from decoupled_engine.srt.runtime_context import get_parallel
from decoupled_engine.srt.utils import get_device_module

logger = logging.getLogger(__name__)

device_module = get_device_module()


_BCG_CAPTURING = None


def _lookup_aware_hold() -> bool:
    """F3 flag (SGLANG_JIT_PULL_LOOKUP_AWARE_HOLD); read per call so tests and
    runs that set it after import see it."""
    from decoupled_engine.srt.environ import envs

    return bool(envs.SGLANG_JIT_PULL_LOOKUP_AWARE_HOLD.get())


def _stream_capturing(stream) -> bool:
    """I5 / M-a: capture state of ``stream``. On CUDA, BCG's
    ``cudaStreamGetCaptureInfo`` helper (``torch.cuda.is_current_stream_capturing``
    is unreliable on NVIDIA per BCG); elsewhere, and with fake device modules
    in CPU tests, the device module's ``is_current_stream_capturing``."""
    global _BCG_CAPTURING
    if _BCG_CAPTURING is None:
        _BCG_CAPTURING = False
        try:
            if torch.cuda.is_available():
                from decoupled_engine.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
                    _is_stream_capturing,
                )

                _BCG_CAPTURING = _is_stream_capturing
        except Exception:  # noqa: BLE001 - fall back to the device module
            _BCG_CAPTURING = False
    if _BCG_CAPTURING and getattr(stream, "cuda_stream", None) is not None:
        return bool(_BCG_CAPTURING(stream))
    capturing = getattr(device_module, "is_current_stream_capturing", None)
    return bool(capturing is not None and capturing())


class _FwdLoadWaitTimer:
    """M0: GPU time the forward stream spends in
    load waits, from an "arrive" event recorded before any CPU wait and a
    "resume" event after the stream wait (R7), resolved lazily (never
    synchronized; skipped while a CUDA graph captures, I5). Also the
    remaining time of the in-flight forward at load issue (R5). On only with
    SGLANG_DSA_HOST_DEDUP_LOAD_TIMING. Logging only."""

    def __init__(self):
        self.enabled: Optional[bool] = None
        self.pending: list = []
        # This forward's stall event pairs, taken by the sched tracer at the
        # forward's end (per-iteration stall in the trace; logging only).
        # Collected only once a tracer drains them (TP0 with
        # --sched-trace-dir); other ranks keep none. Capped as a backstop.
        self.forward_pairs: list = []
        self.collect_forward_pairs = False
        self.timing_stream = None
        self._mark_failed = False
        self._resolve_failed = False

    def on(self) -> bool:
        if self.enabled is None:
            from decoupled_engine.srt.environ import envs

            self.enabled = bool(envs.SGLANG_DSA_HOST_DEDUP_LOAD_TIMING.get())
        return self.enabled

    def begin(self):
        if not self.on():
            return None
        if _stream_capturing(device_module.current_stream()):
            return None
        ev = device_module.Event(enable_timing=True)
        ev.record()
        return ev

    def end(self, ev, name: str = "fwd_load_wait_gpu_ms") -> None:
        if ev is None:
            return
        done = device_module.Event(enable_timing=True)
        done.record()
        self.pending.append((ev, done, name))
        if self.collect_forward_pairs and name != "caller_load_wait_gpu_ms":
            if len(self.forward_pairs) >= _MAX_FORWARD_PAIRS:
                self.forward_pairs.clear()  # undrained: never grow without bound
            self.forward_pairs.append((ev, done))  # forward-side waits only

    def take_forward_pairs(self) -> list:
        pairs, self.forward_pairs = self.forward_pairs, []
        return pairs

    def drain_forward_pairs_with(self, tracer) -> None:
        """Install ``tracer.stall_probe`` and start collecting (TP0 only)."""
        tracer.stall_probe = self.take_forward_pairs
        self.collect_forward_pairs = True

    def issue_mark(self, fence_event) -> None:
        """R5: at load issue, an event on an otherwise idle stream; with the
        (timing) load fence that follows the in-flight forward it gives the
        forward's remaining time at issue (0 if it had already ended). The
        stream comes from the high-priority pool, which no engine stream uses
        (they are all low-priority pool streams); a shared hardware queue can
        still delay it, so the value is an upper bound [logging only]."""
        if not self.on() or self._mark_failed:
            return
        try:
            if self.timing_stream is None:
                self.timing_stream = device_module.Stream(priority=-1)
            ev = device_module.Event(enable_timing=True)
            ev.record(self.timing_stream)
            self.pending.append((ev, fence_event, "fwd_remaining_at_issue_ms"))
        except Exception as e:  # noqa: BLE001 - measurement must never break loads
            self._mark_failed = True
            logger.warning("M0 remaining-forward-at-issue timing disabled: %s", e)

    def resolve(self) -> None:
        if not self.pending:
            return
        from decoupled_engine.srt.mem_cache.ce_2d_load import COST

        keep = []
        for a, b, name in self.pending:
            # Both ends: R5's pair spans two streams, so b done does not
            # imply a done, and elapsed_time raises on an unfinished event.
            if not (b.query() and a.query()):
                keep.append((a, b, name))
                continue
            try:
                COST.add(name, max(0.0, a.elapsed_time(b)))
            except Exception as e:  # noqa: BLE001 - logging only, never break loads
                if not self._resolve_failed:
                    self._resolve_failed = True
                    logger.warning("M0 load timing: dropped a %s sample: %s", name, e)
        if len(keep) > _MAX_FORWARD_PAIRS:  # backstop: never grow without bound
            del keep[: len(keep) - _MAX_FORWARD_PAIRS]
        self.pending = keep


_MAX_FORWARD_PAIRS = 4096
_FWD_WAIT = _FwdLoadWaitTimer()


def log_load_op_settled(ack) -> None:
    """M0 logging only (SGLANG_DSA_HOST_DEDUP_LOAD_TIMING; P and D alike):
    size of a settled load-back op and its submit -> settle latency as the
    scheduler observed it (includes the poll delay)."""
    if not _FWD_WAIT.on():
        return
    from decoupled_engine.srt.mem_cache.ce_2d_load import COST

    COST.add("load_op_tokens", float(ack.num_tokens))
    COST.add("load_op_mb", float(ack.num_bytes) / 1e6)
    t_submit = getattr(ack, "t_submit", 0.0)
    if t_submit:
        COST.add("load_op_submit_to_settle_ms", (time.perf_counter() - t_submit) * 1e3)


def loads_in_flight(tree_cache) -> int:
    """M0 logging only (I4: never decides anything): load-back ops of this
    rank submitted but not finished (event queries, no sync). The sched
    trace records it per launched batch, e.g. to split D decode step times
    by whether a restore was copying alongside."""
    cc = getattr(tree_cache, "cache_controller", None)
    if cc is None:
        return 0
    n = 0
    for ack in getattr(cc, "ack_load_queue", ()):
        if not ack.finish_event.query():
            n += 1
    return n


def _log_forward_in_flight(fence_event) -> None:
    """M0 logging only (invariant I4: a rank-local event query never decides
    anything): 1 if the forward the load fence follows was still running when
    the load was issued, else 0. The COST mean is the in-flight fraction."""
    if fence_event is None or not _FWD_WAIT.on():
        return
    from decoupled_engine.srt.mem_cache.ce_2d_load import COST

    COST.add("fwd_in_flight_at_issue", 0.0 if fence_event.query() else 1.0)
    _FWD_WAIT.issue_mark(fence_event)
    _FWD_WAIT.resolve()  # bounded on D too (no consumer waits there)


def _stream_key(stream):
    handle = getattr(stream, "cuda_stream", None)
    return handle if handle is not None else id(stream)


def layerwise_consumer_stream(counter, batch):
    """L-e: the stream key of the forward that consumes a layer-wise load
    (None if ``batch`` consumes none). Call before the target forward."""
    if any(
        getattr(w, "layerwise", False)
        for w in getattr(batch, "hicache_extra_waits", None) or ()
    ):
        # M5 3.3: a bootstrap-prefetch op adopted in flight, consumed per part.
        return _stream_key(device_module.current_stream())
    idx = getattr(batch, "hicache_consumer_index", -1)
    events = getattr(counter, "events", None)
    if not events or idx is None or idx < 0:
        return None
    if getattr(events[idx], "part_of_layer", None) is None:
        return None
    return _stream_key(device_module.current_stream())


def check_draft_on_consumer_stream(key) -> None:
    """L-e / I6: the MTP draft rows of a layer-wise op sit in its last part and
    are covered only by stream order behind the target forward's last-part
    wait, so the draft extend must run on the target forward's stream."""
    if key is not None and _stream_key(device_module.current_stream()) != key:
        raise RuntimeError(
            "Layer-wise load: the draft extend of a consumer batch runs on a "
            "different stream than its target forward"
        )


class LayerLoadingEvent:
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_events = [device_module.Event() for _ in range(num_layers)]
        self.start_event = device_module.Event()  # start event on controller stream
        # Shared host-dedup loader job (ce_loader) that records this slot's
        # events; None when the events are recorded at submission.
        self.gate_job = None
        # Layer-group consumption (overlap M1, SGLANG_DSA_HOST_DEDUP_LAYER_PARTS
        # > 1), set per op by the producer: layer -> part, and one event per
        # part that the loader records after the part's KV + indexer rows (the
        # MTP draft rows go in the last part). None = whole op (today).
        self.part_of_layer: Optional[List[int]] = None
        self.part_events: list = []  # created on demand, reused by the slot
        self._n_parts = 0
        # Highest part this slot's consumer already waited for, per stream
        # (wait_until runs 2-3 times per layer); reset per consumer.
        self.waited_part: dict = {}

    def set_parts(self, parts):
        """Producer side, per op: layer ranges [(l0, l1), ...] covering every
        layer, or None (whole op). Returns the part events the loader records
        (None for a whole op)."""
        self.waited_part = {}
        if not parts:
            self.part_of_layer = None
            self._n_parts = 0
            return None
        if parts[0][0] != 0 or parts[-1][1] != self._num_layers:
            raise ValueError(
                f"layer parts {parts} do not cover {self._num_layers} layers"
            )
        while len(self.part_events) < len(parts):
            self.part_events.append(device_module.Event())
        part_of_layer = [0] * self._num_layers
        for p, (l0, l1) in enumerate(parts):
            for layer in range(l0, l1):
                part_of_layer[layer] = p
        self.part_of_layer = part_of_layer
        self._n_parts = len(parts)
        return self.part_events[: self._n_parts]

    def complete(self, layer_index: int):
        assert 0 <= layer_index < self._num_layers
        self.load_events[layer_index].record()

    def wait(self, layer_index: int):
        if self.part_of_layer is not None:
            self._wait_part(self.part_of_layer[layer_index])
            return
        device_module.current_stream().wait_event(self.load_events[layer_index])

    def _wait_part(self, p: int) -> None:
        """Layer-wise consumer: CPU-wait until the loader has recorded part
        ``p``'s event (I3), then a stream wait on it. Parts are recorded in
        order on one loader stream, so part p covers every part before it."""
        stream = device_module.current_stream()
        key = _stream_key(stream)
        if self.waited_part.get(key, -1) >= p:
            return
        if _stream_capturing(stream):
            # I5: a captured wait would replay against a stale record.
            raise RuntimeError("Layer-wise load wait during CUDA graph capture")
        if self.gate_job is None:
            # Its per-layer events are never recorded: fail, never skip (I7).
            raise RuntimeError("Layer-wise load slot has no loader job")
        t = _FWD_WAIT.begin()  # "arrive", before the CPU wait (R7)
        try:
            self.gate_job.wait_part_enqueued(p)
        except BaseException:
            # I7 / L-a: the forward is partly enqueued with collectives; the
            # exception must reach the scheduler's crash path (SIGQUIT to the
            # parent), never be swallowed.
            logger.error("Layer-wise load: part %d wait failed mid-launch", p)
            raise
        stream.wait_event(self.part_events[p])
        _FWD_WAIT.end(t, "fwd_part_wait_gpu_ms")
        _FWD_WAIT.resolve()
        self.waited_part[key] = p

    @property
    def finish_event(self):
        ev = self.load_events[-1]
        if self.part_of_layer is not None:
            ev = self.part_events[self._n_parts - 1]  # recorded after the whole op
        if self.gate_job is not None:
            from decoupled_engine.srt.mem_cache.ce_loader import GatedEvent

            return GatedEvent(ev, self.gate_job)
        return ev

    def wait_gate(self) -> None:
        """Consumer side of a loader job: CPU-wait for its submission, then a
        stream wait on its last event (no-op without a loader job). A
        layer-wise job is gated per part in wait(): here only surface an
        error the loader already hit."""
        if self.gate_job is not None:
            if self.part_of_layer is not None:
                if self.gate_job.error is not None:
                    raise RuntimeError(
                        "Shared host-dedup loader failed"
                    ) from self.gate_job.error
                return
            t = _FWD_WAIT.begin()  # "arrive", before the CPU wait (R7)
            self.gate_job.ready(True)
            device_module.current_stream().wait_event(self.load_events[-1])
            _FWD_WAIT.end(t)
            _FWD_WAIT.resolve()


class DetachedLoadParts(LayerLoadingEvent):
    """The layer parts of ONE bootstrap-prefetch op (M5): its part events live
    here, outside the producer ring (whose slots belong to admission loads),
    one object per op and never reused, so a stale consumer can only wait on
    events of an op that already completed.

    A batch that adopted the op in flight registers it with the layer counter
    as an extra consumer (``LayerDoneCounter.add_extra_consumer``); each layer
    then waits only for the op's part that holds it (the M1 consumer wait:
    CPU "part enqueued" check, I3; stream wait; deduplicated per stream and
    part; never during capture, I5). With ``timed`` (the prefetcher's logging
    rank) every part wait is bracketed by timing events, as R7 asks ("arrive"
    before the CPU wait, "resume" after the stream wait): the op's exposure."""

    detached = True  # l2_transfer: copy this background op in layer parts

    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_events = []  # never recorded: the op is consumed per part
        self.start_event = None
        self.gate_job = None
        self.part_of_layer = None
        self.part_events = []
        self._n_parts = 0
        self.waited_part = {}
        self.timed = False
        self.stall_pairs: list = []

    def wait(self, layer_index: int):
        if self.part_of_layer is None:
            raise RuntimeError("Detached load parts without parts")
        self._wait_part(self.part_of_layer[layer_index])

    def _wait_part(self, p: int) -> None:
        stream = device_module.current_stream()
        if self.waited_part.get(_stream_key(stream), -1) >= p:
            return
        arrive = None
        if self.timed and not _stream_capturing(stream):
            arrive = device_module.Event(enable_timing=True)
            arrive.record()
        super()._wait_part(p)
        if arrive is not None:
            resume = device_module.Event(enable_timing=True)
            resume.record()
            self.stall_pairs.append((arrive, resume))

    def all_parts_waited(self) -> bool:
        """Some stream waited for the last part (the forward passed its last
        layer's wait)."""
        return (
            self._n_parts > 0
            and max(self.waited_part.values(), default=-1) >= self._n_parts - 1
        )

    def stall_ms(self) -> Optional[float]:
        """Summed stream stall of the part waits once the last part was waited
        and every timing pair completed; None before."""
        if not self.all_parts_waited():
            return None
        total = 0.0
        for arrive, resume in self.stall_pairs:
            if not resume.query():
                return None
            total += float(arrive.elapsed_time(resume))
        return total


class LayerDoneCounter:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # extra producer and consumer counters for overlap mode
        self.num_counters = 3
        self.events = [LayerLoadingEvent(num_layers) for _ in range(self.num_counters)]
        self.producer_index = -1
        self.consumer_index = -1
        # M5 3.3: per-part bootstrap-prefetch ops the running forward adopted
        # (DetachedLoadParts), waited per layer next to the consumer slot.
        self.extra_consumers: list = []

    def update_producer(self):
        self.producer_index = (self.producer_index + 1) % self.num_counters
        slot = self.events[self.producer_index]
        if slot.gate_job is not None:
            slot.gate_job.ready(True)  # its events must exist before reuse
            slot.gate_job = None
        if slot.part_of_layer is not None:
            # M-c: a layer-wise consumer launched after part 0, so nothing has
            # synchronized with the op's end: wait for its last part before
            # the slot's part events are reused (no decision reads this, I4).
            slot.finish_event.synchronize()
        assert (
            slot.finish_event.query()
        ), "Producer finish event should be ready before being reused."
        return self.producer_index

    def set_consumer(self, index: int):
        self.consumer_index = index
        if index >= 0:
            # Shared host-dedup loader: the consuming forward waits for the
            # whole op here, at launch, not at load submission (layer-wise
            # ops: per part, in wait_until).
            self.events[index].waited_part = {}
            self.events[index].wait_gate()

    def add_extra_consumer(self, parts: DetachedLoadParts) -> None:
        """M5 3.3: the forward about to launch adopted ``parts``' op in flight;
        wait_until also waits for its part of each layer until end_consumer."""
        parts.waited_part = {}
        self.extra_consumers.append(parts)

    def end_consumer(self) -> None:
        """L-b: after the consuming target forward returned, forwards that set
        no consumer must not wait on this slot again (it may be refilled).
        Layer-wise slots only: whole-op behavior is unchanged. Extra (adopted
        prefetch) consumers end here too."""
        idx = self.consumer_index
        if idx >= 0 and self.events[idx].part_of_layer is not None:
            self.consumer_index = -1
        if self.extra_consumers:
            self.extra_consumers = []

    def wait_until(self, threshold: int):
        for parts in self.extra_consumers:
            parts.wait(threshold)
        if self.consumer_index < 0:
            return
        self.events[self.consumer_index].wait(threshold)

    def reset(self):
        self.producer_index = -1
        self.consumer_index = -1
        self.extra_consumers = []


class CacheOperation:

    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
        pool_transfers: Optional[List[PoolTransfer]] = None,
        d2h_sm_copy: bool = False,
    ):
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.node_ids = [node_id]
        self.data = None
        self.pool_transfers = pool_transfers
        # Write (device->host) only: copy with SM kernels instead of the
        # staged copy-engine path where every pool of the op supports it. A
        # copy-engine D2H holds the GPU's D2H engine, so every later
        # device->host readback on any stream (.item(), .cpu(), result
        # copies) queues behind it; an SM copy does not. The controller
        # clears the flag when a pool cannot do it.
        self.d2h_sm_copy = d2h_sm_copy
        # Write only, set by _move_write_operation: the op takes the staged
        # kernel copy into page_first_direct host pools with its device
        # indices left on the GPU (SGLANG_HICACHE_DIRECT_LAYOUT_KERNEL_BACKUP).
        self.d2h_pfd_kernel = False
        # Load only, M4 load quarantine (Q4 CPU page bookkeeping): None = not
        # quarantined (the load keeps the fence); 0 = clean destination pages;
        # > 0 = the newest release epoch among them, whose forward the load
        # waits for (forward_done ring).
        self.dirty_epoch: Optional[int] = None

        self.id = CacheOperation.counter
        CacheOperation.counter += 1
        # default priority is the order of creation
        self.priority = priority if priority is not None else self.id

    def set_load_allocation(self, allocation) -> None:
        self.dirty_epoch = allocation.dirty_epoch

    @staticmethod
    def _merge_pool_transfers(
        ops: List[CacheOperation],
    ) -> Optional[List[PoolTransfer]]:
        grouped: dict[tuple[PoolName, Optional[PoolName]], List[PoolTransfer]] = {}
        for op in ops:
            for transfer in op.pool_transfers or []:
                grouped.setdefault(
                    (transfer.name, transfer.indices_from_pool), []
                ).append(transfer)
        if not grouped:
            return None

        def cat_or_none(tensors):
            parts = [tensor for tensor in tensors if tensor is not None]
            return torch.cat(parts) if parts else None

        return [
            PoolTransfer(
                name=transfers[0].name,
                host_indices=cat_or_none(t.host_indices for t in transfers),
                device_indices=cat_or_none(t.device_indices for t in transfers),
                keys=[key for t in transfers if t.keys for key in t.keys] or None,
                hit_policy=transfers[0].hit_policy,
                indices_from_pool=transfers[0].indices_from_pool,
            )
            for transfers in grouped.values()
        ]

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        assert ops
        if len(ops) == 1:
            return ops[0]
        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged_op = CacheOperation(
            host_indices,
            device_indices,
            -1,
            priority,
            pool_transfers=CacheOperation._merge_pool_transfers(ops),
            d2h_sm_copy=all(op.d2h_sm_copy for op in ops),
        )
        merged_op.node_ids = node_ids
        if all(op.dirty_epoch is not None for op in ops):
            merged_op.dirty_epoch = max(op.dirty_epoch for op in ops)
        return merged_op

    def __lt__(self, other: CacheOperation):
        return self.priority < other.priority


class HiCacheAck(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    node_ids: List[int]
    num_tokens: int = 0
    timing_enabled: bool = False
    # Tokens transferred per host pool (PoolName value -> count).
    num_tokens_by_pool: Optional[dict[str, int]] = None
    # Total bytes moved by the op across all pools, including draft piggyback
    # and sidecar transfers that the per-pool token counts exclude.
    num_bytes: int = 0
    # perf_counter() at submission (load ops; 0 = unknown). M0 logging only.
    t_submit: float = 0.0
    # M5 3.3: the layer parts of a detached (bootstrap-prefetch) load op, for
    # consumers that adopt it in flight; None = whole op.
    parts_slot: Any = None
    # Write ops: the op's first and last host row, folded (host_rows_sig); 0 =
    # unknown. It identifies the settled writes in the unified radix cache's
    # settle digest while every rank's host allocator hands out the same rows.
    # A peer-KV servicer breaks that by design: the serving rank keeps freed
    # pages that a reader still leases out of its free lists
    # (pool_host/slot_allocator.py), so its later writes get other rows. The
    # digest then names the write by device_sig instead
    # (UnifiedRadixCache._settle_digest_folds_host_rows).
    host_sig: int = 0
    # Write ops: the op's lowest and highest device row, folded (rows_span_sig;
    # 0 = unknown, e.g. device-resident indices). TP-invariant: device
    # allocation is lockstep, and the C2g followers' peer copies rely on it.
    device_sig: int = 0
    # Write ops with device-resident indices (direct_layout_kernel_backup)
    # under SGLANG_HICACHE_SETTLE_DIGEST_CHECK: pinned CPU [min, max] of the
    # device rows, copied on the submitting stream before the op's start
    # event, so it is valid once finish_event completed (settle time).
    # resolved_device_sig() folds it as rows_span_sig would. None = unused.
    device_span: Any = None

    def resolved_device_sig(self) -> int:
        """device_sig, or the device_span folded the same way (call only
        after finish_event completed)."""
        if self.device_sig or self.device_span is None:
            return self.device_sig
        lo, hi = (int(x) for x in self.device_span.tolist())
        return span_sig(lo, hi)


_HOST_SIG_MASK = (1 << 42) - 1


def host_rows_sig(host_indices) -> int:
    """First and last host row of an op, folded; read on the CPU only (0 for
    a missing, empty or device-resident index tensor: never a sync)."""
    if host_indices is None or getattr(host_indices, "is_cuda", False):
        return 0
    if host_indices.numel() == 0:
        return 0
    first, last = int(host_indices[0]), int(host_indices[-1])
    return ((first & _HOST_SIG_MASK) * 1000003 + last + 1) & _HOST_SIG_MASK


def rows_span_sig(indices) -> int:
    """Lowest and highest row of an op, folded as host_rows_sig. It does not
    depend on the rows' order: the direct io backend's layer_first path
    reorders device rows by host row, and host rows may differ across ranks.
    CPU only (0 for a missing, empty or device-resident tensor: never a sync)."""
    if indices is None or getattr(indices, "is_cuda", False):
        return 0
    if indices.numel() == 0:
        return 0
    return span_sig(int(indices.min()), int(indices.max()))


def span_sig(lo: int, hi: int) -> int:
    """rows_span_sig of rows whose lowest / highest row are lo / hi."""
    return ((lo & _HOST_SIG_MASK) * 1000003 + hi + 1) & _HOST_SIG_MASK


def device_span_async(device_indices: torch.Tensor) -> Optional[torch.Tensor]:
    """[min, max] of device-resident rows into pinned CPU memory, queued on
    the current stream (no sync); None for an empty tensor."""
    if device_indices is None or device_indices.numel() == 0:
        return None
    span = torch.stack((device_indices.min(), device_indices.max()))
    out = torch.empty(2, dtype=span.dtype, pin_memory=True)
    out.copy_(span, non_blocking=True)
    return out


def _settle_digest_check_on() -> bool:
    from decoupled_engine.srt.environ import envs

    return bool(envs.SGLANG_HICACHE_SETTLE_DIGEST_CHECK.get())


@dataclass
class PrefetchAck:
    """ACK for prefetch operation.

    A sequence of PrefetchAck is sent to the scheduler thread via ack_prefetch_queue,
    indicating progress or completion of the prefetch operation.

    For example, a prefetch operation may results into the following sequence of PrefetchAck:

    1. PrefetchAck(completed_tokens = 128)
    2. PrefetchAck(completed_tokens = 256)
    3. PrefetchAck(pool_hits={INDEXER: 256})
    4. PrefetchAck(completed_req = True)

    The last PrefetchAck always specifies completed_req = True.
    """

    rid: str
    operation: PrefetchOperation
    # Number of hits in KV pool.
    completed_tokens: Optional[int] = None
    # Number of hits in extra pools.
    pool_hits: Optional[dict[str, int]] = None
    completed_req: Optional[bool] = None


class StorageOperation:
    counter = 0

    def __init__(
        self,
        host_indices: Optional[torch.Tensor],
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.host_indices = host_indices
        self.token_ids = token_ids
        self.last_hash = last_hash
        self.completed_tokens = 0
        self.hash_value = hash_value if hash_value is not None else []
        self.prefix_keys = prefix_keys
        # Full queried page-hash chain, set by _storage_hit_query before
        # hash_value is truncated to the hit boundary; the tail is the
        # absence signal that invalidates buffer-mode existence beliefs.
        self.all_hash_values: Optional[List[str]] = None
        # Prefetch-outcome accounting, set at enqueue by the tree cache.
        self.stats_requested_tokens = 0
        self.stats_total_tokens = 0

        self.id = StorageOperation.counter
        StorageOperation.counter += 1

    def __lt__(self, other: StorageOperation):
        return self.id < other.id


# Buffer-mode staging budgets. Prefetch staging is latency-critical
# (wait_complete gates TTFT), so loads may fill the pool up to this fraction
# before new prefetches are declined.
HICACHE_LOAD_POOL_USAGE_FRACTION = 0.9
# Write-staging floor: writes are deferrable, so the flush gate grows the
# write window dynamically into whatever load staging is not using.
HICACHE_WRITE_STAGING_POOL_FRACTION = 0.2


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        kv_pull_hint: Optional[dict] = None,
    ):
        self.request_id = request_id
        self.prefetch_attempt_id = uuid.uuid4().hex
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

        super().__init__(None, token_ids, last_hash, prefix_keys=prefix_keys)

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
        }

    def mark_terminate(self):
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        with self._lock:
            return self._terminated_flag


class HiCacheController:

    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
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
        enable_storage_metrics: bool = False,
        host_memory_mode: str = "cache",
    ):
        self.tp_group = tp_group
        self.host_memory_mode = host_memory_mode
        self.attn_cp_group = attn_cp_group
        self.attn_tp_group = attn_tp_group
        self.pp_group = pp_group
        self.prefetch_hits_sync_groups: List[torch.distributed.ProcessGroup] = []
        self.prefetch_completion_sync_groups: List[torch.distributed.ProcessGroup] = []
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        from decoupled_engine.srt.mem_cache.memory_pool import HybridLinearKVPool

        if isinstance(mem_pool_device, HybridLinearKVPool):
            mem_pool_device = mem_pool_device.full_kv_pool
        self.mem_pool_device = mem_pool_device
        self.mem_pool_host = mem_pool_host
        self.host_dedup = getattr(mem_pool_host, "host_dedup", None)
        self.storage_host_pool = mem_pool_host
        self.write_policy = write_policy
        self.page_size = page_size
        self.io_backend = io_backend
        self.enable_storage = False
        self.storage_backend = None
        self.storage_backend_type = None
        self.enable_storage_metrics = enable_storage_metrics
        # Buffer mode: wired by the tree cache after attach; the load rate
        # limiter subtracts write staging from actual pool usage.
        self.host_write_staged_tokens_fn: Optional[Callable[[], int]] = None

        # Default storage page IO functions (may be overridden by attach).
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set

        # Dedicated stop event for storage background threads (prefetch/backup).
        self.storage_stop_event = threading.Event()

        # Storage control queues, (re)created whenever the storage threads start.
        self.prefetch_buffer: Optional[Queue[PrefetchOperation]] = None
        self.prefetch_sync_queue: Optional[Queue[PrefetchAck]] = None
        self.prefetch_hit_queue: Optional[Queue[StorageOperation]] = None
        self.ack_prefetch_queue = Queue[PrefetchAck]()
        self.ack_backup_queue: Optional[Queue[StorageOperation]] = None
        self.host_mem_release_queue: Optional[Queue[torch.Tensor]] = None

        self.device = self.mem_pool_device.device
        self.layer_num = self.mem_pool_device.layer_num
        self.layer_done_counter = LayerDoneCounter(self.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        # self.write_queue = PriorityQueue[CacheOperation]()
        self.load_queue: List[CacheOperation] = []
        self.write_queue: List[CacheOperation] = []
        self.ack_load_queue: List[HiCacheAck] = []
        # Set by the scheduler to the forward stream; gates load-back H2D
        # behind in-flight forwards (see start_loading).
        self.load_fence_stream = None
        self.ack_write_queue: List[HiCacheAck] = []

        self.l2_transfer_engine = L2TransferEngine(io_backend)

        # If a storage backend is provided at startup, treat it as an implicit attach,
        # so init/runtime share the same lifecycle semantics and code paths.
        if storage_backend is not None:
            try:
                self.attach_storage_backend(
                    storage_backend=storage_backend,
                    prefetch_threshold=prefetch_threshold,
                    model_name=model_name,
                    storage_backend_extra_config=storage_backend_extra_config,
                )
            except ValueError as e:
                # Preserve the historical error shape on init for unknown backends.
                raise ValueError(f"Failed to create storage backend: {e}") from e

    def get_attn_cp_rank_and_size(self) -> tuple[int, int]:
        """Derive CP rank/size from the attn_cp process group."""
        if self.attn_cp_group is not None:
            return (
                torch.distributed.get_rank(group=self.attn_cp_group),
                torch.distributed.get_world_size(group=self.attn_cp_group),
            )
        return 0, 1

    def _create_sync_groups(self) -> List[torch.distributed.ProcessGroup]:
        from decoupled_engine.srt.distributed.parallel_state import create_custom_parallel_group

        groups: List[torch.distributed.ProcessGroup] = []
        seen_rank_sets = set()

        if self.attn_cp_group is not None or self.attn_tp_group is not None:
            base_groups = [self.attn_cp_group, self.attn_tp_group]
        else:
            base_groups = [self.tp_group]
        if self.pp_group is not None:
            base_groups.append(self.pp_group)

        for group in base_groups:
            if group is None or torch.distributed.get_world_size(group=group) == 1:
                continue
            group_ranks = tuple(torch.distributed.get_process_group_ranks(group))
            if group_ranks in seen_rank_sets:
                continue
            seen_rank_sets.add(group_ranks)
            groups.append(
                create_custom_parallel_group(
                    group_ranks=list(group_ranks), backend="gloo"
                )
            )
        return groups

    def _destroy_sync_groups(
        self, groups: List[torch.distributed.ProcessGroup]
    ) -> None:
        for group in groups:
            try:
                torch.distributed.destroy_process_group(group)
            except Exception:
                pass

    def _all_reduce(
        self,
        tensor: torch.Tensor,
        op,
        groups: List[torch.distributed.ProcessGroup],
    ) -> None:
        for group in groups:
            torch.distributed.all_reduce(tensor, op=op, group=group)

    def _start_storage_threads(self):
        """Start storage prefetch/backup threads and their queues.

        This is used by runtime attach, and also by reset when storage is enabled.
        """
        assert self.enable_storage
        assert not self.storage_stop_event.is_set()

        self.prefetch_thread = threading.Thread(
            target=self.prefetch_thread_func, daemon=True
        )
        self.prefetch_io_aux_thread = threading.Thread(
            target=self.prefetch_io_aux_func, daemon=True
        )
        self.prefetch_sync_thread = threading.Thread(
            target=self.prefetch_sync_thread_func, daemon=True
        )
        self.backup_thread = threading.Thread(
            target=self.backup_thread_func, daemon=True
        )
        self.prefetch_queue = Queue()
        self.backup_queue = Queue()
        self.prefetch_buffer = Queue()
        self.prefetch_sync_queue = Queue()
        self.prefetch_hit_queue = Queue()
        self.ack_prefetch_queue = Queue()
        self.ack_backup_queue = Queue()
        self.host_mem_release_queue = Queue()

        self.prefetch_thread.start()
        self.prefetch_io_aux_thread.start()
        self.prefetch_sync_thread.start()
        self.backup_thread.start()

    def _stop_storage_threads(self):
        """Stop storage prefetch/backup threads and drain internal queues.

        Caller should ensure no in-flight requests.
        """
        # Always request stop. This is safe even when storage is already disabled,
        # and makes detach truly idempotent (previous partial detach may have left
        # threads alive).
        # NOTE: do NOT clear storage_stop_event unless threads have fully stopped; otherwise
        # a still-alive thread may resume and touch released state.
        self.storage_stop_event.set()

        # Best-effort wakeups so threads exit promptly even if blocked on queues.
        try:
            if hasattr(self, "prefetch_queue"):
                self.prefetch_queue.put_nowait(None)
            if hasattr(self, "backup_queue"):
                self.backup_queue.put_nowait(None)
            if hasattr(self, "prefetch_buffer"):
                self.prefetch_buffer.put_nowait(None)
            if hasattr(self, "prefetch_sync_queue"):
                self.prefetch_sync_queue.put_nowait(None)
        except Exception:
            pass

        # Best-effort joins (threads are daemon, but join keeps state clean).
        threads = []
        if hasattr(self, "prefetch_thread"):
            threads.append(self.prefetch_thread)
        if hasattr(self, "backup_thread"):
            threads.append(self.backup_thread)
        if hasattr(self, "prefetch_io_aux_thread"):
            threads.append(self.prefetch_io_aux_thread)
        if hasattr(self, "prefetch_sync_thread"):
            threads.append(self.prefetch_sync_thread)

        for t in threads:
            try:
                t.join(timeout=10)
            except Exception:
                pass

        alive = [t for t in threads if getattr(t, "is_alive", lambda: False)()]
        if alive:
            logger.error(
                "Failed to stop HiCache storage threads cleanly: %s",
                [getattr(t, "name", repr(t)) for t in alive],
            )
            raise RuntimeError("Failed to stop HiCache storage threads cleanly.")

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        """Attach (enable) storage backend at runtime.

        Requirement: no in-flight requests. This call is expected to run on the scheduler
        thread (control path), not concurrently with prefetch/backup.
        """
        if self.enable_storage:
            raise RuntimeError("Storage backend already attached.")
        if getattr(self, "host_dedup", None) is not None and not (
            storage_backend == "dynamic"
            and (storage_backend_extra_config or {}).get("backend_name") == "peer_kv"
        ):
            raise ValueError("V4 host dedup supports only the peer_kv storage backend")

        # Defensive: a previous partial detach may have flipped `enable_storage` but
        # left background threads alive. Attaching on top of them is unsafe.
        try:
            self._stop_storage_threads()
        except Exception as e:
            raise RuntimeError(
                "Cannot attach storage backend: previous detach did not stop storage threads cleanly."
            ) from e

        # Rollback-safe init: if creation fails, keep controller state consistent
        # for future attach attempts.
        self.storage_backend_type = storage_backend
        from decoupled_engine.srt.mem_cache.utils import get_hash_str

        self.get_hash_str = get_hash_str
        self.storage_config = self._generate_storage_config(
            model_name, storage_backend_extra_config
        )
        # for MLA models, only one rank needs to backup the KV cache
        self.backup_skip = (
            self.storage_config.is_mla_model
            # todo: load balancing
            and self.storage_config.tp_rank != 0
        )

        # Use storage backend factory for dynamic backend creation
        from decoupled_engine.srt.mem_cache.storage import StorageBackendFactory

        try:
            self.storage_backend = StorageBackendFactory.create_backend(
                storage_backend, self.storage_config, self.storage_host_pool
            )
            if getattr(self, "host_dedup", None) is None or self.host_dedup.is_src:
                self.storage_backend.register_mem_pool_host(self.storage_host_pool)

            self.enable_storage = True
            # todo: threshold policy for prefetching
            self.prefetch_threshold = max(prefetch_threshold, self.page_size)
            if self.host_memory_mode == "buffer_only":
                # The whole pool is transient staging; loads may fill it up
                # to this fraction, and the tree's write flush gate yields
                # to live fetch demand (the write fraction is a floor).
                self.prefetch_capacity_limit = int(
                    HICACHE_LOAD_POOL_USAGE_FRACTION * self.mem_pool_host.size
                )
            else:
                # Budget speculative prefetch at half the host pool, leaving the rest for the write-back staging path.
                self.prefetch_capacity_limit = int(0.5 * self.mem_pool_host.size)
            # tracking the number of tokens locked in prefetching, updated by the main scheduler thread
            self.prefetch_tokens_occupied = 0

            # Use dedicated gloo groups so storage prefetch sync is isolated
            # from other collectives and consistent across CPxTP participants.
            self.prefetch_hits_sync_groups = self._create_sync_groups()
            self.prefetch_completion_sync_groups = self._create_sync_groups()

            # Select the get and set functions
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set

            if (
                self.storage_backend_type
                in ["hf3fs", "mooncake", "eic", "nixl", "simm", "mori"]
            ) or (
                self.storage_backend_type == "dynamic"
                and bool(self.storage_config.extra_config.get("interface_v1", 0))
            ):
                self.page_get_func = self._page_get_zero_copy
                self.page_set_func = self._page_set_zero_copy

            # Ensure stop_event is clear before starting threads.
            self.storage_stop_event.clear()
            self._start_storage_threads()
        except Exception:
            # Best-effort cleanup for partial init.
            try:
                self._stop_storage_threads()
            except Exception:
                pass
            self._destroy_sync_groups(self.prefetch_hits_sync_groups)
            self._destroy_sync_groups(self.prefetch_completion_sync_groups)
            self.prefetch_hits_sync_groups = []
            self.prefetch_completion_sync_groups = []
            try:
                if (
                    hasattr(self, "storage_backend")
                    and self.storage_backend is not None
                ):
                    if hasattr(self.storage_backend, "close"):
                        self.storage_backend.close()
            except Exception:
                pass
            self.storage_backend = None
            self.storage_backend_type = None
            self.enable_storage = False
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set
            raise

    def detach_storage_backend(self):
        """Detach (disable) storage backend at runtime.

        Requirement: no in-flight requests. This will stop storage threads and release
        the backend instance (best-effort close).
        """
        # Idempotent cleanup: even if `enable_storage` is already False,
        # we may still have leftover resources (threads/backend/process group) from a
        # previous partial detach. We attempt cleanup whenever possible.
        try:
            self._stop_storage_threads()
        except Exception as e:
            # Do not proceed tearing down backend/process group if threads are not
            # fully stopped; otherwise still-alive threads may touch released state.
            # Caller can retry detach.
            logger.exception("Stop storage threads failed: %s", e)
            # IMPORTANT: Do not silently succeed. Upper layers rely on exceptions here
            # to avoid flipping `enable_storage` flags while threads are still alive.
            raise RuntimeError("Stop storage threads failed; detach aborted.") from e

        # Best-effort destroy process groups created for storage ops.
        self._destroy_sync_groups(
            self.prefetch_hits_sync_groups + self.prefetch_completion_sync_groups
        )
        self.prefetch_hits_sync_groups = []
        self.prefetch_completion_sync_groups = []

        # Best-effort close (some backends rely on GC/destructor).
        try:
            if (
                hasattr(self, "storage_backend")
                and self.storage_backend is not None
                and hasattr(self.storage_backend, "close")
            ):
                self.storage_backend.close()
        except Exception:
            logger.exception("Failed to close storage backend cleanly.")

        self.storage_backend = None
        self.storage_backend_type = None
        self.enable_storage = False
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set
        # Now it's safe to clear the stop event for future re-attach.
        self.storage_stop_event.clear()

    def _generate_storage_config(
        self,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        if storage_backend_extra_config is None:
            storage_backend_extra_config = {}

        if is_dp_attention_enabled():
            self.tp_rank = get_parallel().attn_tp_rank
            self.tp_size = get_parallel().attn_tp_size
            self.dp_rank = get_attention_dp_rank()
        else:
            self.tp_rank = get_parallel().tp_rank
            self.tp_size = get_parallel().tp_size
            self.dp_rank = 0

        self.pp_rank = get_parallel().pp_rank
        self.pp_size = get_parallel().pp_size

        # Currently, NPUMLATokenToKVPool is the subclass of MLATokenToKVPool.
        # DeepSeekV4TokenToKVPool has compressed MLA-style rank-replicated cache
        # data. storage only needs rank 0 to write it back.
        from decoupled_engine.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

        is_mla_model = isinstance(self.mem_pool_device, MLATokenToKVPool)
        is_compressed_mla_model = isinstance(
            self.mem_pool_device, DeepSeekV4TokenToKVPool
        )
        is_rank_replicated = is_mla_model or is_compressed_mla_model
        # Least Common Multiple among heterogeneous tp size
        tp_lcm_size = storage_backend_extra_config.pop("tp_lcm_size", None)
        should_split_heads = False

        if tp_lcm_size:
            assert (
                tp_lcm_size % self.tp_size == 0
            ), "tp_lcm_size must be divisible by tp_size."
            should_split_heads = (
                not is_rank_replicated
                and self.mem_pool_host.layout == "page_head"
                and tp_lcm_size > self.tp_size
            )

        attn_cp_rank, attn_cp_size = self.get_attn_cp_rank_and_size()

        return HiCacheStorageConfig(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=attn_cp_size,
            # TODO(hzh): Rename is_mla_model to is_rank_replicated.
            is_mla_model=is_rank_replicated,
            enable_storage_metrics=self.enable_storage_metrics,
            is_page_first_layout=self.mem_pool_host.layout == "page_first",
            model_name=model_name,
            tp_lcm_size=tp_lcm_size,
            should_split_heads=should_split_heads,
            extra_config=storage_backend_extra_config,
        )

    def reset(self):
        self.write_queue.clear()
        self.load_queue.clear()
        self.ack_write_queue.clear()
        self.ack_load_queue.clear()
        if self.enable_storage:
            # Include the auxiliary prefetch-I/O thread. Replacing queues while
            # that thread is still using an old queue can let a late L3 read
            # survive a cache flush.
            self._stop_storage_threads()
            self.prefetch_tokens_occupied = 0

        self.storage_stop_event.clear()

        if self.enable_storage:
            self._start_storage_threads()

    @staticmethod
    def _queue_unfinished_tasks(queue: Queue) -> int:
        with queue.all_tasks_done:
            return queue.unfinished_tasks

    def storage_background_operation_counts(self) -> dict[str, int]:
        """Return queued plus actively executing storage operations."""
        return {
            "prefetch_query": self._queue_unfinished_tasks(self.prefetch_queue),
            "prefetch_io": self._queue_unfinished_tasks(self.prefetch_buffer),
            "prefetch_sync": self._queue_unfinished_tasks(self.prefetch_sync_queue),
            "backup_io": self._queue_unfinished_tasks(self.backup_queue),
        }

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Back up KV caches from device memory to host memory.
        """
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.write_queue.append(
            CacheOperation(
                host_indices,
                self._q4_transfer_indices(device_indices),
                node_id,
                priority,
            )
        )
        self.start_writing()
        return host_indices

    def _q4_transfer_indices(self, device_indices: torch.Tensor) -> torch.Tensor:
        """Q4 (cpu_page_book.py): the CPU token slots of tagged device indices,
        so no transfer moves indices off the device; unchanged when off."""
        from decoupled_engine.srt.mem_cache.allocator.cpu_paged import cpu_paged_allocator
        from decoupled_engine.srt.mem_cache.cpu_page_book import tokens_of

        allocator = getattr(self, "mem_pool_device_allocator", None)
        allocator = getattr(allocator, "full_attn_allocator", allocator)
        if cpu_paged_allocator(allocator) is None:
            return device_indices
        return tokens_of(device_indices, allocator.page_size, "HiCache transfer")

    # Set by the PD prefill overlap loop (SGLANG_PREFILL_DEFER_WRITE_SUBMIT)
    # while schedule_stream waits for an in-flight forward: queued writes keep
    # their host pages, tree commit and write-through lock, and are submitted
    # at the loop's next flush point, where the index D2H in move_indices does
    # not wait for that forward. Default False: submit right away.
    defer_start_writing: bool = False

    def start_writing(self, limit: Optional[int] = None) -> None:
        """Submit the queued writes as one merged op (one write ack).

        ``limit``: submit only the first ``limit`` queued ops and leave the
        rest queued for the next call. The 6h flush after a gate-pending
        reorder uses it (scheduler_components/deferred_write_submit.py), so
        every TP rank merges the same writes into each op: writing_check
        settles write acks by count, which assumes identical ops on every
        rank."""
        if len(self.write_queue) == 0:
            return
        if self.defer_start_writing:
            return
        ops = self.write_queue if limit is None else self.write_queue[:limit]
        if not ops:
            return

        op = CacheOperation.merge_ops(ops)
        host_sig = host_rows_sig(op.host_indices)
        # May clear op.d2h_sm_copy (a pool without an SM copy path).
        host_indices, device_indices, pool_transfers = self._move_write_operation(op)
        # After the move: CPU rows under the direct io backend (its index
        # D2H already ran); 0 for rows that stay on the device.
        device_sig = rows_span_sig(device_indices)
        device_span = None
        if (
            getattr(op, "d2h_pfd_kernel", False)
            and device_sig == 0
            and _settle_digest_check_on()
        ):
            # Before submit records the op's start event on this stream.
            device_span = device_span_async(device_indices)
        if limit is None:
            self.write_queue.clear()
        else:
            del self.write_queue[:limit]

        completion = self.l2_transfer_engine.submit_device_to_host(
            self._l2_transfers(host_indices, device_indices, pool_transfers),
            sm_copy=getattr(op, "d2h_sm_copy", False),
        )

        self.ack_write_queue.append(
            HiCacheAck(
                start_event=completion.start_event,
                finish_event=completion.finish_event,
                node_ids=op.node_ids,
                num_tokens=len(op.device_indices),
                timing_enabled=completion.timing_enabled,
                num_tokens_by_pool=self._num_tokens_by_pool(op),
                num_bytes=self._transfer_num_bytes(op),
                host_sig=host_sig,
                device_sig=device_sig,
                device_span=device_span,
            )
        )

    def _transfer_num_bytes(self, op: CacheOperation) -> int:
        return len(op.device_indices) * self.mem_pool_host.size_per_token

    def _num_tokens_by_pool(self, op: CacheOperation) -> dict[str, int]:
        return {PoolName.KV.value: len(op.device_indices)}

    @timed_phase("cache_controller_load")
    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Load KV caches from host memory to device memory.
        """
        from decoupled_engine.srt.mem_cache.allocator.cpu_paged import cpu_paged_allocator

        allocator = self.mem_pool_device_allocator
        if cpu_paged_allocator(allocator) is not None:
            # Q4: CPU page ids; clean pages first under the M4 quarantine.
            allocation = allocator.alloc_load(len(host_indices))
            if allocation is None:
                return None
            op = CacheOperation(host_indices, allocation.indices_cpu, node_id, priority)
            op.set_load_allocation(allocation)
            self.load_queue.append(op)
            return allocation.indices
        device_indices = allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        op = CacheOperation(host_indices, device_indices, node_id, priority)
        self.load_queue.append(op)
        return device_indices

    def move_indices(
        self, host_indices: torch.Tensor, device_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # move indices to GPU if using kernels, to host if using direct indexing
        if self.io_backend == "kernel":
            if not host_indices.is_cuda:
                host_indices = host_indices.to(self.device, non_blocking=True)
            return host_indices, device_indices
        elif self.io_backend == "direct":
            if self.mem_pool_host.layout == "layer_first":
                device_indices = device_indices.cpu()
                host_indices, idx = host_indices.sort()
                return host_indices, device_indices.index_select(0, idx)
            elif self.mem_pool_host.layout == "page_first_direct":
                if not device_indices.is_cuda or getattr(
                    self, "_loader_moves_device_indices", False
                ):
                    # The shared loader thread does this D2H off the scheduler.
                    return host_indices, device_indices
                # Blocking D2H on the scheduler thread (waits for the current
                # stream); its cost is logged with the shared-dedup costs.
                t0 = time.perf_counter()
                device_indices = device_indices.cpu()
                from decoupled_engine.srt.mem_cache.ce_2d_load import COST

                COST.add("move_indices_d2h", (time.perf_counter() - t0) * 1e3)
                return host_indices, device_indices
            else:
                raise ValueError(
                    f"Unsupported layout {self.mem_pool_host.layout!r} for io backend 'direct'"
                )
        elif self.io_backend == "kernel_ascend":
            return host_indices, device_indices.cpu()
        else:
            raise ValueError(f"Unsupported io backend")

    def _move_write_operation(
        self, op: CacheOperation
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[List[PoolTransfer]]]:
        """Keep CPU host indices only for page-first staged write-back."""
        # Single-pool controllers keep their pool's copy path.
        op.d2h_sm_copy = False
        # Flag off (pool flag False): not evaluated, the old path as is.
        if getattr(
            self.mem_pool_host, "pfd_kernel_backup", False
        ) and self._direct_layout_kernel_backup(op, [self.mem_pool_host]):
            return op.host_indices, op.device_indices, op.pool_transfers
        if (
            self.io_backend == "kernel"
            and self.mem_pool_host.layout == "page_first"
            and getattr(self.mem_pool_host, "can_use_write_back_jit", False)
        ):
            return op.host_indices, op.device_indices, op.pool_transfers
        return self._move_op_indices(op)

    def _direct_layout_kernel_backup(self, op: CacheOperation, pools) -> bool:
        """SGLANG_HICACHE_DIRECT_LAYOUT_KERNEL_BACKUP: with the direct io
        backend, every host pool this write touches is page_first_direct with
        the staged kernel copy (pool.pfd_kernel_backup), the device indices
        are on the GPU and the host indices on the CPU. Then the op keeps its
        device indices on the GPU: no blocking .cpu() on the scheduler
        thread, and the pools copy with the kernel io backend's staged path
        (pool_host/mla.py, pool_host/dsa.py). Marks the op. Rank-invariant:
        pool flags are set on every rank (host-dedup followers too) and the
        index placement is lockstep. Never a sync (is_cuda is metadata)."""
        if self.io_backend != "direct" or not pools:
            return False
        if not all(getattr(pool, "pfd_kernel_backup", False) for pool in pools):
            return False
        pairs = [(op.host_indices, op.device_indices, True)]
        pairs += [
            (t.host_indices, t.device_indices, False) for t in op.pool_transfers or []
        ]
        for host_indices, device_indices, is_anchor in pairs:
            if host_indices is None or device_indices is None:
                return False
            if is_anchor and host_indices.numel() == 0:
                continue  # no anchor transfer (_l2_transfers skips it)
            if host_indices.is_cuda or not device_indices.is_cuda:
                return False
        op.d2h_pfd_kernel = True
        return True

    def _move_op_indices(
        self, op: CacheOperation
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[List[PoolTransfer]]]:
        return (*self.move_indices(op.host_indices, op.device_indices), None)

    def _l2_transfers(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        pool_transfers: Optional[List[PoolTransfer]] = None,
    ) -> list[L2Transfer]:
        transfers = [
            L2Transfer(
                host_pool=self.mem_pool_host,
                device_pool=self.mem_pool_device,
                host_indices=host_indices,
                device_indices=device_indices,
            )
        ]
        return transfers

    def _l2_load_transfers(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        pool_transfers: Optional[List[PoolTransfer]] = None,
    ) -> list[L2Transfer]:
        return self._l2_transfers(host_indices, device_indices, pool_transfers)

    def _shared_loader(self):
        return getattr(getattr(self, "host_dedup", None), "loader", None)

    def _move_load_op_indices(self, op):
        """_move_op_indices for loads; with the shared loader, device indices
        stay on the GPU (the loader thread moves them)."""
        if self._shared_loader() is None:
            return self._move_op_indices(op)
        self._loader_moves_device_indices = True
        try:
            return self._move_op_indices(op)
        finally:
            self._loader_moves_device_indices = False

    def _new_loader_fence(self):
        if self.load_fence_stream is None:
            return None
        # M0 (LOAD_TIMING): a timing event, so the remaining forward time at
        # issue can be resolved against it (R5).
        ev = (
            device_module.Event(enable_timing=True)
            if _FWD_WAIT.on()
            else device_module.Event()
        )
        ev.record(self.load_fence_stream)
        return ev

    # M4 load quarantine (mem_cache/load_quarantine.py); set by the PD
    # prefill scheduler. None = off.
    load_quarantine = None

    def _q3_load(self, op: CacheOperation) -> bool:
        """M4 Q3: a quarantined load through the shared loader needs no fence
        and no start_event wait: its pages are clean, or it waits on
        forward_done[dirty_epoch]. L-g: the whole op goes through the loader
        (submit_host_to_device asserts the route)."""
        return (
            self.load_quarantine is not None
            and op.dirty_epoch is not None
            and self._shared_loader() is not None
            and getattr(getattr(self, "host_dedup", None), "ipc_fanout", None) is None
        )

    def _q3_kwargs(self, op: CacheOperation) -> dict:
        return {
            "q3": True,
            "q3_dirty_event": self.load_quarantine.dirty_event(op.dirty_epoch),
        }

    def _record_loader_fence(self, producer_id: int) -> None:
        """The load fence as an event (the loader copies on its own stream)."""
        if self._shared_loader() is None:
            return
        if not hasattr(self, "_loader_fences"):
            self._loader_fences = {}
        self._loader_fences[producer_id] = self._new_loader_fence()

    def start_loading(self, *, wait_on_caller: bool = True) -> int:
        prepared = self.prepare_loading()
        return self.submit_loading(prepared, wait_on_caller=wait_on_caller)

    def prepare_loading(self):
        """Resolve addresses and capture the old-forward fence before lookahead.

        No transfer is submitted here. A caller may enqueue an independent
        forward before submit_loading without changing this safety boundary.
        """
        if len(self.load_queue) == 0:
            return None

        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices, pool_transfers = self._move_load_op_indices(op)
        self.load_queue.clear()
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()
        if not hasattr(self, "_loader_q3"):
            self._loader_q3 = {}
        self._loader_q3.pop(producer_id, None)
        if self._q3_load(op):
            # M4 Q3: start_event is for timing only; no fence.
            getattr(self, "_loader_fences", {}).pop(producer_id, None)
            self._loader_q3[producer_id] = self._q3_kwargs(op)
            return producer_id, op, host_indices, device_indices, pool_transfers
        self._record_loader_fence(producer_id)
        _log_forward_in_flight(getattr(self, "_loader_fences", {}).get(producer_id))

        if self.load_fence_stream is not None:
            # in overlap scheduling, reclaimed pages might still be written by the forward thread
            # therefore a fence is needed for loading thread to prevent memory corruption
            # todo: it's possible to use a finer-grained fence
            self.l2_transfer_engine.host_to_device_stream.wait_stream(
                self.load_fence_stream
            )

        return producer_id, op, host_indices, device_indices, pool_transfers

    def submit_loading(
        self, prepared, *, wait_on_caller: bool = True, defer_broadcast: bool = False
    ) -> int:
        if prepared is None:
            return -1
        producer_id, op, host_indices, device_indices, pool_transfers = prepared
        producer_event = self.layer_done_counter.events[producer_id]

        transfers = self._l2_load_transfers(
            host_indices, device_indices, pool_transfers
        )
        kwargs = {}
        if self._shared_loader() is not None:
            kwargs["fence_event"] = getattr(self, "_loader_fences", {}).pop(
                producer_id, None
            )
            kwargs.update(getattr(self, "_loader_q3", {}).pop(producer_id, {}))
        fanout = getattr(getattr(self, "host_dedup", None), "ipc_fanout", None)
        if fanout is not None:
            kwargs["ipc_slot"] = producer_id
            if getattr(fanout, "ce_loader", None) is not None:
                kwargs["host_indices_cpu"] = self._cpu_host_index_map(
                    op, host_indices, pool_transfers
                )
        set_parts = getattr(producer_event, "set_parts", None)
        if set_parts is not None:
            # Whole op unless the shared loader splits it into layer parts
            # (overlap M1); it then sets this slot's parts itself.
            set_parts(None)
            if self._layerwise_loads():
                kwargs["layer_events"] = producer_event
        completion = self.l2_transfer_engine.submit_host_to_device(
            transfers,
            start_event=producer_event.start_event,
            on_layer_done=producer_event.complete,
            layer_num=self.layer_num,
            wait_on_caller=wait_on_caller,
            defer_broadcast=defer_broadcast,
            **kwargs,
        )
        # A loader job records this slot's events later; consumers gate on it.
        producer_event.gate_job = getattr(completion, "job", None)

        self.ack_load_queue.append(
            HiCacheAck(
                start_event=completion.start_event,
                finish_event=completion.finish_event,
                node_ids=op.node_ids,
                num_tokens=len(op.device_indices),
                timing_enabled=completion.timing_enabled,
                num_tokens_by_pool=self._num_tokens_by_pool(op),
                num_bytes=self._transfer_num_bytes(op),
                t_submit=time.perf_counter(),
            )
        )
        return producer_id

    def _layerwise_loads(self) -> bool:
        """Overlap M1: hand ring ops to the shared loader as layer-wise ops
        (SGLANG_DSA_HOST_DEDUP_LAYER_PARTS > 1; P workers, the only consumers).
        Needs the forward's per-layer ``wait_until`` to reach this
        controller's counter, i.e. the device pool's registered one; if a
        subclass replaced the counter after registration, loads stay whole-op
        (gated at launch). Static per process: the same on every rank (I4)."""
        ok = getattr(self, "_layerwise_ok", None)
        if ok is None:
            from decoupled_engine.srt.environ import envs

            ok = False
            if envs.SGLANG_DSA_HOST_DEDUP_LAYER_PARTS.get() > 1:
                pool = getattr(self, "mem_pool_device", None)
                ok = (
                    getattr(pool, "layer_transfer_counter", None)
                    is self.layer_done_counter
                )
                if not ok:
                    logger.warning(
                        "Layer-wise load-back disabled: the device pool's per-layer "
                        "waits use another layer counter; loads stay whole-op"
                    )
            self._layerwise_ok = ok
        return ok

    @staticmethod
    def _cpu_host_index_map(op, host_indices, pool_transfers) -> dict:
        """id(moved host indices) -> the op's original CPU host indices."""
        out = {}
        if op.host_indices is not None and not op.host_indices.is_cuda:
            out[id(host_indices)] = op.host_indices
        for moved, orig in zip(pool_transfers or (), op.pool_transfers or ()):
            if (
                orig.host_indices is not None
                and moved.host_indices is not None
                and not orig.host_indices.is_cuda
            ):
                out[id(moved.host_indices)] = orig.host_indices
        return out

    def start_loading_detached(
        self,
        *,
        stream=None,
        ipc_slot=None,
        wait_on_caller: bool = False,
        background: bool = False,
        layerwise: bool = False,
    ):
        """Submit the queued loads as one op outside the producer ring.

        Used by the bootstrap-window prefetch: no forward consumes the op's
        per-layer events, and its acknowledgement is owned by the caller (it
        is NOT appended to ack_load_queue). Returns the HiCacheAck or None.
        ``background``: nothing waits for the op until admission (concurrent
        prefetch); the shared loader may rate-cap its C2g peer reads.
        ``layerwise`` (M5 3.3): with layer-wise loads on (M1), the shared
        loader copies the op in layer parts whose events go to a detached slot
        (``ack.parts_slot``), so a forward that adopts the op in flight waits
        per part. Static per process and the same on every rank (I4).
        """
        if len(self.load_queue) == 0:
            return None
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices, pool_transfers = self._move_load_op_indices(op)
        self.load_queue.clear()
        start_event = device_module.Event()
        start_event.record()
        h2d = (
            stream
            if stream is not None
            else self.l2_transfer_engine.host_to_device_stream
        )
        q3 = self._q3_load(op)
        if self.load_fence_stream is not None and not q3:
            # Reclaimed pages may still be written by the in-flight forward.
            h2d.wait_stream(self.load_fence_stream)
        kwargs = {}
        if q3:
            kwargs.update(self._q3_kwargs(op))
        elif self._shared_loader() is not None:
            kwargs["fence_event"] = self._new_loader_fence()
            _log_forward_in_flight(kwargs["fence_event"])
        fanout = getattr(getattr(self, "host_dedup", None), "ipc_fanout", None)
        if fanout is not None and getattr(fanout, "ce_loader", None) is not None:
            kwargs["host_indices_cpu"] = self._cpu_host_index_map(
                op, host_indices, pool_transfers
            )
        if background:
            kwargs["background"] = True
        parts_slot = None
        if layerwise and self._shared_loader() is not None and self._layerwise_loads():
            parts_slot = DetachedLoadParts(self.layer_num)
            kwargs["layer_events"] = parts_slot
        completion = self.l2_transfer_engine.submit_host_to_device(
            self._l2_load_transfers(host_indices, device_indices, pool_transfers),
            start_event=start_event,
            on_layer_done=None,
            layer_num=self.layer_num,
            wait_on_caller=wait_on_caller,
            stream=stream,
            ipc_slot=ipc_slot,
            **kwargs,
        )
        if parts_slot is not None:
            parts_slot.gate_job = getattr(completion, "job", None)
            if parts_slot.part_of_layer is None or parts_slot.gate_job is None:
                parts_slot = None  # copied whole-op: consumed whole-op
        ack = HiCacheAck(
            start_event=completion.start_event,
            finish_event=completion.finish_event,
            node_ids=op.node_ids,
            num_tokens=len(op.device_indices),
            timing_enabled=completion.timing_enabled,
            num_tokens_by_pool=self._num_tokens_by_pool(op),
            num_bytes=self._transfer_num_bytes(op),
        )
        if parts_slot is not None:
            ack = ack._replace(parts_slot=parts_slot)
        return ack

    def evict_device(self, device_indices: torch.Tensor) -> int:
        self.mem_pool_device_allocator.free(device_indices)
        return len(device_indices)

    def evict_host(self, host_indices: torch.Tensor, backup_only: bool = True) -> int:
        if not backup_only:
            raise ValueError("Other eviction policies are not supported yet.")

        self.mem_pool_host.free(host_indices)
        return len(host_indices)

    def prefetch(
        self,
        request_id: str,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        kv_pull_hint: Optional[dict] = None,
    ) -> PrefetchOperation:
        """
        Prefetch KV caches from storage backend to host memory.
        """
        operation = PrefetchOperation(
            request_id, new_input_tokens, last_hash, prefix_keys, kv_pull_hint
        )
        self.prefetch_queue.put(operation)
        return operation

    def terminate_prefetch(self, operation):
        """
        Request to terminate a prefetch operation.

        Must be called in the scheduler thread.

        Asynchronous prefetch tasks may be running in background threads.  When all prefetch
        tasks are terminated, a PrefetchAck with completed_req=True will be sent to ack_prefetch_queue.
        """
        operation.mark_terminate()
        # Before allocation no reader can use the lease. Once the buffer has
        # been allocated, only the IO thread may release it, after its reads.
        if operation.host_indices is None:
            self._notify_prefetch_terminated(operation)
        return operation.completed_tokens, operation.hash_value

    def _notify_prefetch_terminated(self, operation):
        """Notify only before IO ownership or after synchronous reads return."""
        on_terminated = getattr(self.storage_backend, "on_prefetch_terminated", None)
        if on_terminated is not None:
            try:
                if getattr(self.storage_backend, "supports_complete_endpoints", False):
                    on_terminated(
                        operation.request_id, attempt_id=operation.prefetch_attempt_id
                    )
                else:
                    on_terminated(operation.request_id)
            except Exception:
                logger.warning(
                    "storage backend on_prefetch_terminated failed", exc_info=True
                )

    def append_host_mem_release(self, host_indices: torch.Tensor):
        if host_indices.numel() == 0:
            return
        pages = host_indices.split(self.mem_pool_host.page_size)
        for page in pages:
            self.host_mem_release_queue.put(page)

    def _page_get_zero_copy(
        self, operation, hash_values, host_indices, extra_info=None
    ) -> int:
        results = self.storage_backend.batch_get_v1(
            hash_values, host_indices, extra_info
        )
        inc = 0
        for i in range(len(hash_values)):
            if not results[i]:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            inc += 1
        return inc

    # todo: deprecate
    def _generic_page_get(
        self, operation, hash_values, host_indices, extra_info=None
    ) -> int:
        dummy_page_dst = [
            self.storage_host_pool.get_dummy_flat_data_page() for _ in hash_values
        ]
        page_data = self.storage_backend.batch_get(hash_values, dummy_page_dst)
        if page_data is None:
            return 0
        count = 0
        for i in range(len(hash_values)):
            if page_data[i] is None:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            if operation.is_terminated():
                break
            self.storage_host_pool.set_from_flat_data_page(
                host_indices[i * self.page_size],
                page_data[i],
            )
            count += 1
        return count

    def _page_transfer(self, operation: PrefetchOperation) -> int:
        # Transfer batch by batch
        prefix_keys = operation.prefix_keys
        kv_derived_transfers = [
            transfer
            for transfer in getattr(operation, "pool_transfers", None) or []
            if transfer.indices_from_pool == PoolName.KV
        ]
        all_success = True
        completed_pages = 0
        for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
            # When an error is occurred, we should keep looping and produce the same number of
            # PrefetchAck as other ranks do, because prefetch_sync_thread (i.e. consumer of
            # prefetch_sync_queue) perform reduce on the results.  This is so tricky.
            if all_success and operation.is_terminated():
                all_success = False
                # Let lease-based backends give the owner its pins back now
                # instead of waiting for TTL expiry (peer_kv).
                self._notify_prefetch_terminated(operation)
            if all_success:
                batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
                batch_host_indices = operation.host_indices[
                    i * self.page_size : (i + len(batch_hashes)) * self.page_size
                ]

                # Get one batch token, and update the completed_tokens if succeed
                extra_info = HiCacheStorageExtraInfo(
                    prefix_keys=prefix_keys, extra_info=operation.storage_extra_info()
                )

                hit_pages = self._page_transfer_kv_batch(
                    operation,
                    batch_hashes,
                    batch_host_indices,
                    extra_info,
                    kv_derived_transfers,
                )
                # Check termination
                if hit_pages != len(batch_hashes):
                    all_success = False
                if prefix_keys and len(prefix_keys) > 0:
                    prefix_keys += batch_hashes
                completed_pages += hit_pages
            ack = PrefetchAck(
                rid=operation.request_id,
                completed_tokens=completed_pages * self.page_size,
                operation=operation,
            )
            self.prefetch_sync_queue.put(ack)
        return completed_pages

    def _page_transfer_kv_batch(
        self,
        operation: PrefetchOperation,
        batch_hashes: List[str],
        batch_host_indices: torch.Tensor,
        extra_info: HiCacheStorageExtraInfo,
        kv_derived_transfers: List[PoolTransfer],
    ) -> int:
        """Read a single batch from KV and KV-derived pools (e.g. indexer pool).

        Return the number of hit pages.  If the hits from KV and KV-derived pools differ,
        clamp to the minimal number of hits.

        Here, "batch" means a single unit of L3 read, not a "batch" in model forward.
        """
        # Followers emit the same per-batch acknowledgements but perform no IO.
        # MIN reduction includes the source's actual result before publication.
        if getattr(self, "host_dedup", None) is not None and self.host_dedup.is_dummy:
            return len(batch_hashes)
        # Read from KV pool.
        kv_hits = self.page_get_func(
            operation, batch_hashes, batch_host_indices, extra_info
        )

        # Read from KV-derived sidecar pools, if any.
        sidecar_hits: dict[str, int] = {}
        if len(kv_derived_transfers) > 0:
            current_kv_derived_transfers = [
                PoolTransfer(
                    name=transfer.name,
                    host_indices=batch_host_indices,
                    keys=batch_hashes,
                )
                for transfer in kv_derived_transfers
            ]
            # extra_info carries the request id (and kv_pull_hint) so a
            # lease-based backend can attribute this read to the same
            # operation as the KV read above.
            sidecar_results = self.storage_backend.batch_get_v2(
                current_kv_derived_transfers, extra_info
            )
            sidecar_hits = count_pool_hits(sidecar_results)

        # Clamp to minimal number of hits.
        return min([kv_hits, *sidecar_hits.values()])

    def prefetch_io_aux_func(self):
        """
        Auxiliary function conducting IO operations for prefetching.
        """
        prefetch_buffer = self.prefetch_buffer
        while not self.storage_stop_event.is_set():
            try:
                operation = prefetch_buffer.get(block=True, timeout=1)
            except Empty:
                continue
            try:
                if operation is None:
                    continue
                try:
                    self._page_transfer(operation)
                finally:
                    # Includes hybrid trailing-pool reads. Scheduler-side
                    # cancellation must not unpin their source buffers early.
                    if operation.is_terminated():
                        self._notify_prefetch_terminated(operation)

                self.prefetch_sync_queue.put(
                    PrefetchAck(
                        rid=operation.request_id,
                        completed_req=True,
                        operation=operation,
                    )
                )
            finally:
                prefetch_buffer.task_done()

    def prefetch_rate_limited(self) -> bool:
        """
        Rate limit the prefetching operations to avoid overwhelming the storage backend.
        """
        if self.host_memory_mode == "buffer_only":
            # Gate on real pool usage: buffer mode allocates hit-sized, so
            # prefetch_tokens_occupied's requested spans overstate it. Pool
            # state mutates only at scheduler-thread lockstep points, so this
            # stays TP-deterministic. Write staging is the write budget's
            # usage; charging it here would park hits behind its storage drain.
            used = self.mem_pool_host.size - self.mem_pool_host.available_size()
            if self.host_write_staged_tokens_fn is not None:
                used -= self.host_write_staged_tokens_fn()
            return max(0, used) >= self.prefetch_capacity_limit
        # cancel prefetch if too much memory is occupied
        if self.prefetch_tokens_occupied >= self.prefetch_capacity_limit:
            return True
        # todo: more sophisticated rate limiting based on storage backend performance
        return False

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        last_hash = operation.last_hash
        tokens_to_fetch = operation.token_ids
        prefix_keys = operation.prefix_keys.copy() if operation.prefix_keys else None

        storage_query_count = 0
        hash_value = []
        page_hashes = self.get_hash_str(
            tokens_to_fetch, last_hash, page_size=self.page_size
        )
        operation.all_hash_values = page_hashes

        for start in range(0, len(page_hashes), STORAGE_BATCH_SIZE):
            batch_hashes = page_hashes[start : start + STORAGE_BATCH_SIZE]
            extra_info = HiCacheStorageExtraInfo(
                prefix_keys=prefix_keys, extra_info=operation.storage_extra_info()
            )
            hit_page_num = self.storage_backend.batch_exists(batch_hashes, extra_info)
            hash_value.extend(batch_hashes[:hit_page_num])
            storage_query_count += hit_page_num * self.page_size
            if hit_page_num < len(batch_hashes):
                break
            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

        return hash_value, storage_query_count

    def _prefetch_requires_exact_endpoint(self, operation) -> bool:
        return False

    def _synchronize_storage_hit(self, operation, hash_value, storage_hit_count):
        if operation.is_terminated():
            storage_hit_count = 0
        hit = torch.tensor(storage_hit_count, dtype=torch.int)
        self._all_reduce(
            hit, torch.distributed.ReduceOp.MIN, self.prefetch_hits_sync_groups
        )
        return hash_value, hit.item()

    def prefetch_thread_func(self):
        """
        Manage prefetching operations from storage backend to host memory.
        """
        prefetch_queue = self.prefetch_queue
        while (not self.storage_stop_event.is_set()) or not prefetch_queue.empty():
            try:
                operation = prefetch_queue.get(block=True, timeout=1)
            except Empty:
                continue
            try:
                if operation is None:
                    continue
                operation.requires_exact_endpoint = (
                    self._prefetch_requires_exact_endpoint(operation)
                )
                if operation.is_terminated():
                    hash_value, storage_hit_count = [], 0
                else:
                    try:
                        hash_value, storage_hit_count = self._storage_hit_query(
                            operation
                        )
                    except Exception:
                        if not operation.requires_exact_endpoint:
                            raise
                        # Every rank must still enter endpoint agreement when
                        # one owner's initial LOOKUP fails or is malformed.
                        logger.warning("peer endpoint query failed", exc_info=True)
                        hash_value, storage_hit_count = [], 0
                hash_value, storage_hit_count = self._synchronize_storage_hit(
                    operation, hash_value, storage_hit_count
                )

                # Lease-based backends (peer_kv): tell the backend what will
                # actually be fetched after the TP MIN so a rank that was
                # granted more than its peers releases/trims its lease now
                # instead of holding the owner's pins until TTL.
                reconcile = getattr(self.storage_backend, "on_hit_reconciled", None)
                if reconcile is not None and not (
                    getattr(self, "host_dedup", None) is not None
                    and self.host_dedup.is_dummy
                ):
                    try:
                        options = {}
                        if getattr(
                            self.storage_backend, "supports_complete_endpoints", False
                        ):
                            options["attempt_id"] = operation.prefetch_attempt_id
                        reconcile(
                            operation.request_id,
                            storage_hit_count // self.page_size,
                            **options,
                        )
                    except Exception:
                        logger.warning(
                            "storage backend on_hit_reconciled failed", exc_info=True
                        )

                # Record the TP-synced hit count; the scheduler thread decides
                # at drain time whether to revoke (below threshold) or allocate.
                operation.hash_value = hash_value[
                    : (storage_hit_count // self.page_size)
                ]
                operation.storage_hit_count = storage_hit_count
                if _lookup_aware_hold():
                    # F3: the LOOKUP phase ends here (after the TP sync); the
                    # timeout's per-page clock starts now.
                    operation.hit_known_at = time.monotonic()
                self.prefetch_hit_queue.put(operation)
            finally:
                prefetch_queue.task_done()

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> int:
        """
        Write KV caches from host memory to storage backend.
        """
        operation = StorageOperation(
            host_indices, token_ids, hash_value=hash_value, prefix_keys=prefix_keys
        )
        self.backup_queue.put(operation)
        return operation.id

    # todo: deprecate
    def _generic_page_set(self, hash_values, host_indices, extra_info=None) -> bool:
        data = [
            self.storage_host_pool.get_data_page(host_indices[i * self.page_size])
            for i in range(len(hash_values))
        ]
        return self.storage_backend.batch_set(hash_values, data)

    def _page_set_zero_copy(self, hash_values, host_indices, extra_info=None) -> bool:
        return all(
            self.storage_backend.batch_set_v1(hash_values, host_indices, extra_info)
        )

    # Backup batch by batch
    def _page_backup(self, operation):
        # Backup batch by batch
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
            batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]
            # Set one batch token, and record if success.
            # todo: allow partial success
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            success = self.page_set_func(batch_hashes, batch_host_indices, extra_info)
            if not success:
                logger.warning(
                    f"Write page to storage: {len(batch_hashes)} pages failed."
                )
                break

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes
            operation.completed_tokens += self.page_size * len(batch_hashes)

    def backup_thread_func(self):
        """
        Manage backup operations from host memory to storage backend.
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

                if not self.backup_skip:
                    self._page_backup(operation)
                self.ack_backup_queue.put(operation)
            finally:
                backup_queue.task_done()

    def prefetch_sync_thread_func(self):
        """Synchronize prefetch results across all PP and TP ranks."""
        prefetch_sync_queue = self.prefetch_sync_queue
        while not self.storage_stop_event.is_set() or not prefetch_sync_queue.empty():
            try:
                ack = prefetch_sync_queue.get(block=True, timeout=1)
            except Empty:
                continue
            try:
                if ack is None:
                    continue
                self._reduce_prefetch_ack(ack)
                self.ack_prefetch_queue.put(ack)
            finally:
                prefetch_sync_queue.task_done()

    def _reduce_prefetch_ack(self, ack: PrefetchAck) -> None:
        """Synchronize all ranks to agree on a PrefetchAck."""
        if ack.completed_tokens is not None:
            # Determine the minimal successful prefix of tokens.
            completed_tokens_tensor = torch.tensor(
                ack.completed_tokens, dtype=torch.int
            )
            self._all_reduce(
                completed_tokens_tensor,
                torch.distributed.ReduceOp.MIN,
                self.prefetch_completion_sync_groups,
            )
            ack.completed_tokens = completed_tokens_tensor.item()
