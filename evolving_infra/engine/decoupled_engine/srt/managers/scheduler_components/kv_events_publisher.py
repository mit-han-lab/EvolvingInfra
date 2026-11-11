# Adapted from SGLang python/sglang/srt/managers/scheduler_components/kv_events_publisher.py (commit 9a489f8d2f), modified for evolving-infra.
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
)

import msgspec
import zmq

from decoupled_engine.srt.disaggregation.kv_events import (
    EventPublisherFactory,
    KVEventBatch,
    is_kv_publisher_rank,
    kv_events_batch_caps,
    select_kv_publisher_dp_rank,
)
from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.managers.io_struct import hook_custom_types, sock_send

logger = logging.getLogger(__name__)

# Seconds between the SGLANG_KV_EVENTS_PUBLISH_STATS lines.
PUBLISH_STATS_INTERVAL_S = 60.0

if TYPE_CHECKING:
    from decoupled_engine.srt.distributed.parallel_state_wrapper import ParallelState
    from decoupled_engine.srt.mem_cache.base_prefix_cache import BasePrefixCache


class SchedulerStats: ...  # type: ignore[no-redef]


class KvMetrics(msgspec.Struct, tag=True, kw_only=True, array_like=True):
    request_active_slots: int = 0
    request_total_slots: int = 0
    kv_active_blocks: int = 0
    kv_total_blocks: int = 0
    num_requests_waiting: int = 0
    gpu_cache_usage_perc: float = 0.0
    gpu_prefix_cache_hit_rate: float = 0.0
    data_parallel_rank: int = 0


hook_custom_types(KvMetrics)


class _PublishStats:
    """Scheduler-side counters of KV-event publication on the publishing
    rank (SGLANG_KV_EVENTS_PUBLISH_STATS); one INFO line per ~60 s."""

    def __init__(self) -> None:
        self.t_log = time.monotonic()
        # perf_counter of the first deferred call not yet flushed.
        self.defer_t0: Optional[float] = None
        self.lazy_seen = (0, 0)  # recorder (lazy_entries, lazy_fast_built)
        # SGLANG_SYMMETRIC_SKIP_D_ROLE_KV_EVENTS: the gate's counters at the
        # previous line (DRoleEventGate.counters).
        self.d_role_seen = (0, 0, 0, 0)
        self.total_calls = 0
        self.total_flushes = 0
        self.total_events = 0
        self._reset()

    def _reset(self) -> None:
        self.calls = 0
        self.deferred = 0
        self.flushes = 0
        self.events = 0
        self.take_s = 0.0
        self.take_max_s = 0.0
        self.publish_s = 0.0
        self.publish_max_s = 0.0
        self.defer_max_s = 0.0

    def on_call(self, deferred: bool) -> None:
        self.calls += 1
        self.total_calls += 1
        if deferred:
            self.deferred += 1
            if self.defer_t0 is None:
                self.defer_t0 = time.perf_counter()

    def on_flush(self, n_events: int, t0: float, t_taken: float, t_end: float) -> None:
        self.flushes += 1
        self.total_flushes += 1
        self.events += n_events
        self.total_events += n_events
        take_s, publish_s = t_taken - t0, t_end - t0
        self.take_s += take_s
        self.take_max_s = max(self.take_max_s, take_s)
        self.publish_s += publish_s
        self.publish_max_s = max(self.publish_max_s, publish_s)
        if self.defer_t0 is not None:
            self.defer_max_s = max(self.defer_max_s, t0 - self.defer_t0)
            self.defer_t0 = None

    def maybe_log(self, mode: str, recorder: Any, publisher: Any) -> None:
        now = time.monotonic()
        dt = now - self.t_log
        if dt < PUBLISH_STATS_INTERVAL_S:
            return
        self.t_log = now
        lazy = (
            int(getattr(recorder, "lazy_entries", 0)),
            int(getattr(recorder, "lazy_fast_built", 0)),
        )
        lazy_entries, lazy_fast = (a - b for a, b in zip(lazy, self.lazy_seen))
        self.lazy_seen = lazy
        take_wire = getattr(publisher, "take_wire_stats", None)
        wire = take_wire() if take_wire is not None else None
        wire_msg = "wire n/a"
        if wire is not None:
            wire_msg = (
                "wire batches=%d messages=%d bytes=%d max_msg_bytes=%d "
                "split_batches=%d split_events=%d over_cap=%d encode_send_ms=%.1f "
                "(totals messages=%d bytes=%d max_msg_bytes=%d split_events=%d "
                "over_cap=%d)"
                % (
                    wire["batches"],
                    wire["messages"],
                    wire["bytes"],
                    wire["max_bytes"],
                    wire["split_batches"],
                    wire["split_events"],
                    wire["over_cap"],
                    wire["encode_send_ms"],
                    wire["total_messages"],
                    wire["total_bytes"],
                    wire["total_max_bytes"],
                    wire["total_split_events"],
                    wire["total_over_cap"],
                )
            )
        msg = (
            "KV events publish over %.0f s: mode=%s calls=%d deferred=%d "
            "flushes=%d events=%d lazy_entries=%d lazy_fast_built=%d "
            "take_ms=%.1f take_max_ms=%.2f publish_ms=%.1f publish_max_ms=%.2f "
            "defer_max_ms=%.2f | %s | totals calls=%d flushes=%d events=%d"
        )
        args = [
            dt,
            mode,
            self.calls,
            self.deferred,
            self.flushes,
            self.events,
            lazy_entries,
            lazy_fast,
            self.take_s * 1e3,
            self.take_max_s * 1e3,
            self.publish_s * 1e3,
            self.publish_max_s * 1e3,
            self.defer_max_s * 1e3,
            wire_msg,
            self.total_calls,
            self.total_flushes,
            self.total_events,
        ]
        gate = getattr(recorder, "d_role_gate", None)
        if gate is not None:
            # SGLANG_SYMMETRIC_SKIP_D_ROLE_KV_EVENTS only; the line is
            # unchanged without it.
            now_counts = gate.counters()
            delta = [a - b for a, b in zip(now_counts, self.d_role_seen)]
            self.d_role_seen = now_counts
            msg += (
                " | d_role skipped_stores=%d skipped_removes=%d promoted_nodes=%d "
                "promotion_stores=%d (totals %d %d %d %d)"
            )
            args += delta + list(now_counts)
        logger.info(msg, *args)
        self._reset()


def kv_events_end_pass_hook(scheduler: Any) -> Optional[Callable[[], None]]:
    """The once-per-pass hook of batched KV-event publication for a scheduler
    loop (SGLANG_KV_EVENTS_BATCH_PUBLISH): the loop calls it once in every
    pass, after the pass's result processing (the overlap loops: also after
    the launch; loops with a routing sample or a prefill state push: ahead of
    them, so KV events keep going out before the push of the same result).
    None when the flag is off (no attribute is read) or when
    ``begin_pass_batching`` declines; the loop then makes no call."""
    if not envs.SGLANG_KV_EVENTS_BATCH_PUBLISH.get():
        return None
    publisher = getattr(scheduler, "kv_events_publisher", None)
    if publisher is None:
        return None
    return publisher.begin_pass_batching()


@dataclass(kw_only=True, slots=True)
class SchedulerKvEventsPublisher:
    kv_events_config: Optional[str]
    ps: ParallelState
    attn_tp_rank: int
    attn_cp_rank: int
    attn_dp_rank: int
    dp_rank: Optional[int]
    tree_cache: BasePrefixCache
    send_metrics_from_scheduler: Optional[zmq.Socket]
    max_running_requests: int
    max_total_num_tokens: int
    get_stats: Callable
    enable_kv_cache_events: bool = False
    kv_event_publisher: Any = None
    # SGLANG_KV_EVENTS_BATCH_PUBLISH, set by begin_pass_batching(): a publish
    # call closes the recorder's queued segment; end_pass() sends the closed
    # segments once per scheduler pass.
    batching: bool = field(default=False, init=False)
    _recorder: Any = field(default=None, init=False)
    _flush_due: bool = field(default=False, init=False)
    # SGLANG_KV_EVENTS_PUBLISH_STATS counters (publishing rank only).
    _stats: Optional[_PublishStats] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.init_kv_events(self.kv_events_config)

    def init_kv_events(self, kv_events_config: Optional[str]):
        self.enable_kv_cache_events = is_kv_publisher_rank(kv_events_config, self.ps)

        if self.enable_kv_cache_events:
            self.kv_event_publisher = EventPublisherFactory.create(
                kv_events_config,
                select_kv_publisher_dp_rank(
                    self.ps.attn_dp_size, self.ps.attn_dp_rank, self.ps.dp_rank
                ),
            )
            if envs.SGLANG_KV_EVENTS_PUBLISH_STATS.get():
                self._stats = _PublishStats()

    def begin_pass_batching(self) -> Optional[Callable[[], None]]:
        """Switch to batched publication (SGLANG_KV_EVENTS_BATCH_PUBLISH) for
        a scheduler loop that calls the returned hook (``end_pass``) once in
        every pass. None, and publication stays immediate, when the flag is
        off, on a rank that does not publish, or for a cache without an
        enabled event recorder."""
        if not (
            self.enable_kv_cache_events and envs.SGLANG_KV_EVENTS_BATCH_PUBLISH.get()
        ):
            return None
        get_recorder = getattr(self.tree_cache, "kv_event_recorder", None)
        recorder = get_recorder() if get_recorder is not None else None
        if recorder is None or not getattr(recorder, "enabled", False):
            logger.warning(
                "SGLANG_KV_EVENTS_BATCH_PUBLISH: the tree cache %s has no "
                "enabled KV event recorder; publishing at once",
                type(self.tree_cache).__name__,
            )
            return None
        if not self.batching:
            self._recorder = recorder
            self.batching = True
            max_bytes, max_events = kv_events_batch_caps()
            logger.info(
                "KV events: batched publication, one flush per scheduler pass "
                "(max message bytes %d, max events per message %d, lazy fast "
                "build %d, stats %d)",
                max_bytes,
                max_events,
                int(getattr(recorder, "lazy_fast", False)),
                int(self._stats is not None),
            )
        return self.end_pass

    def emit_kv_metrics(self):
        if not self.enable_kv_cache_events:
            return

        kv_metrics = KvMetrics()
        kv_metrics.request_active_slots = self.get_stats().num_running_reqs.total
        kv_metrics.request_total_slots = self.max_running_requests
        kv_metrics.kv_active_blocks = int(
            self.get_stats().token_usage * self.max_total_num_tokens
        )
        kv_metrics.kv_total_blocks = self.max_total_num_tokens
        kv_metrics.num_requests_waiting = self.get_stats().num_queue_reqs.total
        kv_metrics.gpu_cache_usage_perc = self.get_stats().token_usage
        kv_metrics.gpu_prefix_cache_hit_rate = self.get_stats().cache_hit_rate
        kv_metrics.data_parallel_rank = (
            self.ps.dp_rank if self.ps.dp_rank is not None else 0
        )

        if not self.send_metrics_from_scheduler.closed:
            sock_send(self.send_metrics_from_scheduler, kv_metrics)

    def publish_kv_events(self, now: bool = False):
        """Publish the queued KV events.

        With batched publication (``begin_pass_batching``) a call only closes
        the queued segment, and the loop's ``end_pass`` sends it later in
        this pass; ``now`` (the idle and pause calls, which may sleep or stop
        the loop next) sends the closed segments at once."""
        if not self.enable_kv_cache_events:
            return
        stats = self._stats
        if self.batching:
            self._recorder.mark_boundary()
            if stats is not None:
                stats.on_call(deferred=not now)
            if now:
                self._flush_closed()
            else:
                self._flush_due = True
            return
        if stats is not None:
            stats.on_call(deferred=False)
            self._publish_counted()
            return

        events = self.tree_cache.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

    def end_pass(self) -> None:
        """Once per scheduler pass (SGLANG_KV_EVENTS_BATCH_PUBLISH): send the
        segments this pass's publish calls closed. A pass without such a call
        sends nothing, so the cadence stays that of the call sites."""
        if self._flush_due:
            self._flush_closed()

    def _flush_closed(self) -> None:
        self._flush_due = False
        stats = self._stats
        t0 = time.perf_counter() if stats is not None else 0.0
        events = self._recorder.take_closed()
        t_taken = time.perf_counter() if stats is not None else 0.0
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)
        if stats is not None:
            stats.on_flush(len(events), t0, t_taken, time.perf_counter())
            stats.maybe_log("batch", self._recorder, self.kv_event_publisher)

    def _publish_counted(self) -> None:
        """The immediate publication, timed (SGLANG_KV_EVENTS_PUBLISH_STATS)."""
        stats = self._stats
        t0 = time.perf_counter()
        events = self.tree_cache.take_events()
        t_taken = time.perf_counter()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)
        stats.on_flush(len(events), t0, t_taken, time.perf_counter())
        get_recorder = getattr(self.tree_cache, "kv_event_recorder", None)
        stats.maybe_log(
            "immediate",
            get_recorder() if get_recorder is not None else None,
            self.kv_event_publisher,
        )
