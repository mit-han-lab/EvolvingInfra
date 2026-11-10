"""Bounded, causal TP0 prefill state for a future routing predictor.

Capture runs after result processing. A bounded optional launch hook freezes
already-computed Python cache-provenance integers before overlap can mutate
request state. Capture reads those copied values and an existing plan. It does not peek/match the radix
cache, inspect a CUDA event/tensor, call an admission method, or predict a
future result. Null means unknown; an omitted RID never means completion.

Symmetric workers (SGLANG_SYMMETRIC_ROUTING_SNAPSHOT, ``symmetric=True``):
the same snapshot, from symmetric sources (``_sources``). Bootstrap rows are
the P-legs held for their reader (in no queue; ``state.held`` true), and
inflight rows are the computed exports whose KV the reader has not released
(phase "inflight" or "parked"). The plan is the scheduler's lookahead plan
(SGLANG_SYMMETRIC_BWP), else a private one that only this tracker updates,
at most once per interval, with its log lines off: it peeks the radix tree
read-only, and it is never installed as ``scheduler.late_d_lookahead``. A
decode pass (a symmetric worker also decodes) keeps null features except
batch_size: decode batches carry stale prefill lengths. Its result does not
change ``prev``, which stays the last prefill pass. TP0 logs a
"SYM_ROUTING_SNAPSHOT ..." counter line every 60 s.
"""

from __future__ import annotations

import logging
import math
import os
import time
from array import array
from collections import deque
from functools import lru_cache
from itertools import islice
from pathlib import Path
from uuid import uuid4

import msgspec

logger = logging.getLogger(__name__)

SYM_LOG_TAG = "SYM_ROUTING_SNAPSHOT"
SYM_LOG_PERIOD_S = 60.0


class RoutingPolicy(msgspec.Struct, array_like=True):
    chunk_tokens: int | None = None
    page_size: int | None = None
    max_input_tokens: int | None = None
    share_enabled: bool | None = None
    share_unit: int | None = None
    max_reqs: int | None = None
    warm_max_uncached: int | None = None
    age_passes: int | None = None
    schedule_policy: str | None = None


class RoutingPlan(msgspec.Struct, array_like=True):
    sampled_pass: int | None = None
    start_pass: int | None = None
    end_pass: int | None = None
    gate_open_start_pass: int | None = None
    gated: bool | None = None
    skip_reason: str | None = None
    uncached_tokens: int | None = None
    host_demand: int | None = None
    pull_tokens: int | None = None


class RoutingState(msgspec.Struct, array_like=True):
    intent_start_ok: bool | None = None
    intent_received: bool | None = None
    bound: bool | None = None
    launched_unbound: bool | None = None
    gate_blocked: bool | None = None
    held: bool | None = None
    parked: bool | None = None
    pull_pending: bool | None = None
    pull_inflight: bool | None = None
    restore_inflight: bool | None = None


class RoutingRequest(msgspec.Struct, array_like=True):
    rid: str | None
    phase: str
    input_tokens: int | None = None
    device_prefix_tokens: int | None = None
    host_prefix_tokens: int | None = None
    storage_prefix_tokens: int | None = None
    storage_hit_tokens: int | None = None
    committed_prefix_tokens: int | None = None
    inflight_new_tokens: int | None = None
    remaining_compute_tokens: int | None = None
    remaining_definition: str = "unlaunched_compute"
    recv_age_s: float | None = None
    queue_age_s: float | None = None
    prefix_source: str = "unknown"
    plan: RoutingPlan | None = None
    state: RoutingState | None = None
    ready_after_s: float | None = None
    pull_remaining_s: float | None = None
    restore_remaining_s: float | None = None
    bind_remaining_s: float | None = None
    load_exposed: bool | None = None
    unknown: list[str] = msgspec.field(default_factory=list)
    bind_phase: str | None = None
    bind_phase_age_s: float | None = None
    bind_phase_age_source: str | None = None
    pull_phase: str | None = None
    pull_phase_age_s: float | None = None
    pull_phase_age_source: str | None = None
    restore_phase_age_s: float | None = None
    restore_remaining_tokens: int | None = None
    pull_remaining_tokens: int | None = None


class RoutingFeatures(msgspec.Struct, array_like=True):
    new_tokens: int | None = None
    attention_pairs: int | None = None
    topk_work: int | None = None
    batch_size: int | None = None
    first_requests: int | None = None
    host_restore_tokens: int | None = None
    storage_restore_tokens: int | None = None
    load_exposed: bool | None = None


class RoutingBatch(msgspec.Struct, array_like=True):
    forward_iter: int | None = None
    launch_age_s: float | None = None
    status: str = "unknown"
    request_ids: list[str | None] = msgspec.field(default_factory=list)
    omitted_requests: int = 0
    features: RoutingFeatures = msgspec.field(default_factory=RoutingFeatures)
    finishing_requests: int | None = None
    feature_capture_us: float | None = None


class RoutingPrevious(msgspec.Struct, array_like=True):
    finished_requests: int | None = None
    batch_size: int | None = None
    gpu_s: float | None = None


class PrefillRoutingSnapshot(msgspec.Struct):
    schema_version: int = 1
    epoch: str = ""
    sequence: int = 0
    sampled_unix_s: float | None = None
    sampled_monotonic_s: float | None = None
    clock_domain: str = "engine_monotonic"
    clock_domain_id: str | None = None
    scheduler_pass: int | None = None
    available: bool = False
    reason: str | None = "not_sampled"
    age_s: float | None = None
    stale: bool = True
    max_age_s: float = 1.0
    capture_us: float | None = None
    wire_bytes: int = 0
    launch_capture_failures: int = 0
    truncated: bool = False
    total_waiting: int = 0
    omitted_waiting: int = 0
    total_inflight_requests: int = 0
    omitted_inflight_requests: int = 0
    inflight_requests: list[RoutingRequest] = msgspec.field(default_factory=list)
    inflight_requests_known: bool = False
    inflight_batches: list[RoutingBatch] = msgspec.field(default_factory=list)
    inflight_batches_known: bool = False
    total_inflight_batches: int = 0
    omitted_inflight_batches: int = 0
    total_bootstrap: int = 0
    omitted_bootstrap: int = 0
    limits: dict[str, int] = msgspec.field(default_factory=dict)
    waiting_order: str = "stored_scheduler_queue"
    policy: RoutingPolicy = msgspec.field(default_factory=RoutingPolicy)
    waiting: list[RoutingRequest] = msgspec.field(default_factory=list)
    bootstrap: list[RoutingRequest] = msgspec.field(default_factory=list)
    partial: RoutingRequest | None = None
    partial_known: bool = False
    current_batch_known: bool = False
    current_batch: RoutingBatch | None = None
    prev: RoutingPrevious = msgspec.field(default_factory=RoutingPrevious)
    engine_revision: str | None = None
    model_id: str | None = None
    admission_resources: dict = msgspec.field(default_factory=dict)
    timeline: dict | None = None
    restore_feature_definition: str = "first_chunk_disjoint_cache_provenance"
    load_exposed_definition: str = "consumer_index_nonnegative_not_measured_stall"


def public_dict(value):
    """JSON objects even though compact nested SHM structs are array-like."""
    if isinstance(value, msgspec.Struct):
        return {
            key: public_dict(getattr(value, key)) for key in value.__struct_fields__
        }
    if isinstance(value, list):
        return [public_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: public_dict(item) for key, item in value.items()}
    return value


@lru_cache(maxsize=1)
def clock_domain():
    # Shared between scheduler and API processes on one kernel. A ZMQ reader
    # on another kernel cannot compare monotonic clocks and must decline.
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def public_snapshot(value, now_monotonic=None, domain=None):
    out = public_dict(value)
    now = time.monotonic() if now_monotonic is None else now_monotonic
    domain = clock_domain() if domain is None else domain
    compatible = value.clock_domain_id is not None and value.clock_domain_id == domain
    stamp = value.sampled_monotonic_s
    age = now - stamp if stamp is not None and compatible else None
    out["age_s"] = age if age is not None and age >= 0 else None
    out["stale"] = age is None or age < 0 or age > value.max_age_s
    if out["stale"] and out["reason"] is None:
        out["reason"] = "clock_domain_unknown" if not compatible else "stale"
    return out


def _short_string(value):
    return value if isinstance(value, str) and 0 < len(value) <= 256 else None


def _int(value):
    return value if type(value) is int and value >= 0 else None


def _bool(value):
    return value if type(value) is bool else None


def _length(value):
    # Do not even ask a tensor for its length. These fields are host containers.
    return len(value) if isinstance(value, (list, tuple, array)) else None


def _age(stamp, now):
    return now - stamp if type(stamp) in (float, int) and 0 < stamp <= now else None


def _rid(req):
    rid = getattr(req, "rid", None)
    return rid if isinstance(rid, str) and len(rid) <= 256 else None


def _is_decode(batch):
    """A decode pass. Its prefix_lens / extend_lens are stale prefill lengths
    (merge_batch, filter_batch and prepare_for_decode keep them)."""
    is_decode = getattr(getattr(batch, "forward_mode", None), "is_decode", None)
    return bool(is_decode()) if callable(is_decode) else False


def _is_idle(batch):
    """An idle pass (DP attention); it has no prefill work."""
    is_idle = getattr(getattr(batch, "forward_mode", None), "is_idle", None)
    return bool(is_idle()) if callable(is_idle) else False


def _host_rows(batch):
    reqs = getattr(batch, "reqs", None)
    prefixes = getattr(batch, "prefix_lens", None)
    extends = getattr(batch, "extend_lens", None)
    if not all(isinstance(x, (list, tuple)) for x in (reqs, prefixes, extends)):
        return None
    if len(reqs) != len(prefixes) or len(reqs) != len(extends):
        return None
    if not all(_int(x) is not None for x in (*prefixes, *extends)):
        return None
    return reqs, prefixes, extends


def admission_resources(scheduler, observed_adder=None, now=None):
    free = getattr(getattr(scheduler, "req_to_token_pool", None), "free_slots", None)
    result = dict(
        request_slots_free=len(free) if isinstance(free, (list, tuple)) else None,
        token_pool_capacity=_int(getattr(scheduler, "max_total_num_tokens", None)),
        unbound_prompt_tokens_last_scan=_int(
            getattr(scheduler, "_late_d_need_unbound", None)
        ),
        bound_reserve_tokens_last_scan=_int(
            getattr(scheduler, "_late_d_need_reserve", None)
        ),
        park_gate_closed_last_scan=_bool(
            getattr(scheduler, "_late_d_need_gate_closed", None)
        ),
        free_and_evictable_tokens=None,
        note="CPU scalar observations; last admission scan, not a new eligibility check",
    )
    if observed_adder is not None:
        result["adder"] = dict(observed_adder)
        stamp = result["adder"].pop("observed_mono_s", None)
        result["adder"]["age_s"] = _age(stamp, time.monotonic() if now is None else now)
    return result


class PrefillRoutingTracker:
    """Small copied snapshot; holds no Req, cache-node, batch, or tensor refs."""

    def __init__(
        self,
        *,
        max_records=32,
        interval_s=0.1,
        max_age_s=1.0,
        max_bytes=4096,
        timeline_samples=False,
        symmetric=False,
    ):
        # The approved 144-session run can put more than 128 waiting requests
        # on one P. Keep the old default, but permit a complete bounded view.
        self.max_records = max(1, min(256, max_records))
        self.interval_s = max(0.01, interval_s)
        self.max_age_s = max(self.interval_s, max_age_s)
        self.max_bytes = max(1024, min(96 * 1024, max_bytes))
        from .prefill_timeline import PrefillTimeline

        self.timeline = PrefillTimeline() if timeline_samples else None
        self.epoch = uuid4().hex
        self.sequence = 0
        self.last_capture = -math.inf
        self.completed_iter = -1
        self.launch_capture_failures = 0
        self.adder_observation = None
        self.prev = RoutingPrevious()
        self.value = PrefillRoutingSnapshot(epoch=self.epoch, max_age_s=self.max_age_s)
        # Symmetric worker sources (_sources); PD prefill when False.
        self.symmetric = symmetric
        self.plan_source = "none"
        self._private_plan = None
        self._private_plan_at = -math.inf
        self.decode_passes = 0
        self.sym_stats = dict(
            captures=0,
            unavailable=0,
            bootstrap_rows_max=0,
            inflight_rows_max=0,
            capture_us=0.0,
            wire_bytes_max=0,
        )
        self._sym_log_at = time.monotonic()

    def on_admission(self, scheduler, adder, stage):
        """Copy the real PrefillAdder's budgets; never run speculative admission.

        The budget accessors used here are the same metadata-only accessors
        used by add_one_req. They read allocator lengths and cache counters;
        they do not allocate, evict, match, poll a transfer, or read GPU values.
        An observation identifies a scan, not a future readiness guarantee.
        """
        try:

            def number(value):
                return (
                    value
                    if type(value) in (int, float) and math.isfinite(value)
                    else None
                )

            running = getattr(adder.running_batch, "reqs", ())
            result = dict(
                stage=stage,
                observed_mono_s=time.monotonic(),
                scheduler_forward_ct=_int(getattr(scheduler, "forward_ct", None)),
                remaining_total_tokens=number(adder.rem_total_tokens),
                remaining_current_tokens=number(adder.cur_rem_tokens),
                total_token_offset=number(adder.rem_total_token_offset),
                current_token_offset=number(adder.cur_rem_token_offset),
                remaining_input_tokens=number(adder.rem_input_tokens),
                remaining_chunk_tokens=number(adder.rem_chunk_tokens),
                page_size=_int(adder.page_size),
                max_running_requests=_int(adder.max_running_requests),
                prefill_max_requests=_int(adder.prefill_max_requests),
                max_prefill_bs=_int(adder.max_prefill_bs),
                running_requests=_length(running),
                admitted_requests=_length(adder.can_run_list),
                waiting_requests=_length(scheduler.waiting_queue),
                request_slots_free=_length(
                    getattr(scheduler.req_to_token_pool, "free_slots", None)
                ),
                unbound_prompt_tokens=_int(
                    getattr(scheduler, "_late_d_need_unbound", None)
                ),
                bound_reserve_tokens=_int(
                    getattr(scheduler, "_late_d_need_reserve", None)
                ),
                park_gate_closed=_bool(
                    getattr(scheduler, "_late_d_need_gate_closed", None)
                ),
                load_back_stats=dict(adder.load_back_stats),
            )
            self.adder_observation = result
        except Exception as error:
            self.adder_observation = dict(
                stage=stage,
                observed_mono_s=time.monotonic(),
                unavailable_reason=type(error).__name__,
            )

    def on_launch(self, batch):
        # Optional telemetry cannot stop an engine. Unknown fields force the
        # consumer to decline; failures remain visible in the sampled counter.
        try:
            batch._prefill_routing_load_features = None
            self._copy_launch(batch)
            if self.timeline is not None:
                self.timeline.on_launch(batch)
        except Exception:
            self.launch_capture_failures += 1
            batch._prefill_routing_load_features = None

    def _copy_launch(self, batch):
        """Freeze only CPU integers, at most max_records, with no trace dependency.

        This mirrors the archived S2 batch adapter: RH and RS sum disjoint
        first-chunk cache provenance. It does NOT measure this chunk's DMA.
        Retracted or inconsistent metadata remains unknown.
        """
        started = time.perf_counter()
        if _is_decode(batch):
            # No prefill work; the stale lengths must not read as prefill.
            self.decode_passes += 1
            return
        reqs = getattr(batch, "reqs", None)
        if not isinstance(reqs, (list, tuple)) or len(reqs) > self.max_records:
            return
        host = _host_rows(batch)
        consumer = getattr(batch, "hicache_consumer_index", None)
        consumes = consumer >= 0 if type(consumer) is int else None
        first = rh = rs = None
        if host is not None:
            reqs, prefixes, _ = host
            indices = [_int(getattr(req, "extend_batch_idx", None)) for req in reqs]
            if all(n is not None and n > 0 for n in indices) and all(
                getattr(req, "retracted_stain", None) is False for req in reqs
            ):
                first = sum(n == 1 for n in indices)
                rh = rs = 0
                for req, prefix, idx in zip(reqs, prefixes, indices):
                    if idx != 1:
                        continue
                    tiers = [
                        _int(getattr(req, name, None))
                        for name in (
                            "cached_tokens_device",
                            "cached_tokens_host",
                            "cached_tokens_storage",
                        )
                    ]
                    if (
                        getattr(req, "_cache_breakdown_computed", False) is not True
                        or getattr(req, "retracted_stain", None) is not False
                        or any(n is None for n in tiers)
                        or sum(tiers) != prefix
                    ):
                        rh = rs = None
                        break
                    rh += tiers[1]
                    rs += tiers[2]
        if consumes is False:
            rh = rs = 0  # these model terms are gated off, not measured zero DMA
        batch._prefill_routing_load_features = (
            first,
            rh,
            rs,
            consumes,
            (time.perf_counter() - started) * 1e6,
        )

    def on_result(self, batch):
        """Only after the result has been processed; never query GPU timing."""
        if self.timeline is not None:
            try:
                self.timeline.on_result(batch)
            except Exception:
                self.launch_capture_failures += 1
        iteration = _int(getattr(batch, "forward_iter", None))
        if iteration is not None:
            self.completed_iter = max(self.completed_iter, iteration)
        if _is_decode(batch) or (self.symmetric and _is_idle(batch)):
            # A symmetric worker also decodes: ``prev`` stays the last prefill
            # pass (PD prefill never has a decode pass; its idle passes keep
            # the old rule).
            return
        reqs = getattr(batch, "reqs", None)
        count = _length(reqs)
        # A completed batch's per-request completion can be inferred only from
        # explicit CPU finish reasons, never a future token count or event.
        finished = None
        if count is not None and count <= self.max_records:
            reasons = [getattr(req, "finished_reason", None) for req in reqs]
            if all(hasattr(req, "finished_reason") for req in reqs):
                finished = sum(reason is not None for reason in reasons)
        self.prev = RoutingPrevious(finished_requests=finished, batch_size=count)

    def _batch(self, batch, now, scheduler=None):
        if batch is None:
            return None, {}
        iteration = _int(getattr(batch, "forward_iter", None))
        pending = iteration is not None and iteration > self.completed_iter
        status = (
            "unknown" if iteration is None else ("pending" if pending else "completed")
        )
        reqs = getattr(batch, "reqs", None)
        total = _length(reqs)
        selected = list(islice(reqs, self.max_records)) if total is not None else []
        result = RoutingBatch(
            forward_iter=iteration,
            launch_age_s=_age(getattr(batch, "launch_ts", None), now),
            status=status,
            request_ids=[_rid(req) for req in selected],
            omitted_requests=max(0, (total or 0) - len(selected)),
            features=RoutingFeatures(batch_size=total),
        )
        host = (
            _host_rows(batch)
            if total is not None and total <= self.max_records and not _is_decode(batch)
            else None
        )
        launched = {}
        if host is not None:
            reqs, prefixes, extends = host
            result.features.new_tokens = sum(extends)
            result.features.attention_pairs = sum(
                n * h + n * (n + 1) // 2 for h, n in zip(prefixes, extends)
            )
            result.features.topk_work = sum(
                (m := min(n, max(0, 2048 - h))) * h + m * (m + 1) // 2 + (n - m) * 2048
                for h, n in zip(prefixes, extends)
            )
            histories = [getattr(req, "prefill_iters", None) for req in reqs]
            if all(isinstance(h, list) and h and type(h[0]) is int for h in histories):
                result.features.first_requests = sum(
                    h[0] == iteration for h in histories
                )
            frozen = getattr(batch, "_prefill_routing_load_features", None)
            if isinstance(frozen, tuple) and len(frozen) == 5:
                (
                    result.features.first_requests,
                    result.features.host_restore_tokens,
                    result.features.storage_restore_tokens,
                    result.features.load_exposed,
                    result.feature_capture_us,
                ) = frozen
            else:
                # Old/test batches can still expose the tracer's consumer flag;
                # do not read mutable first-chunk provenance after the launch.
                launch_loads = getattr(
                    getattr(scheduler, "sched_tracer", None), "_launch_loads", {}
                )
                load = (
                    launch_loads.get(iteration)
                    if isinstance(launch_loads, dict)
                    else None
                )
                if isinstance(load, tuple) and len(load) == 2 and type(load[1]) is bool:
                    result.features.load_exposed = load[1]
                    if load[1] is False:
                        result.features.host_restore_tokens = (
                            result.features.storage_restore_tokens
                        ) = 0
            counts = [_length(getattr(req, "origin_input_ids", None)) for req in reqs]
            if all(n is not None for n in counts):
                result.finishing_requests = sum(
                    h + n >= total for total, h, n in zip(counts, prefixes, extends)
                )
            for req, prefix, extend in zip(reqs, prefixes, extends):
                if iteration is not None:
                    launched[id(req)] = (
                        prefix if pending else prefix + extend,
                        extend if pending else 0,
                    )
        return result, launched

    def _request(
        self, req, phase, scheduler, plan, launched, restore_ids, restore_known, now
    ):
        rid = _rid(req)
        input_tokens = _length(getattr(req, "origin_input_ids", None))
        entries = getattr(plan, "entries", {})
        entry = (
            entries.get(rid) if isinstance(entries, dict) and rid is not None else None
        )
        device = host = committed = inflight = remaining = None
        source = "unknown"
        if entry is not None:
            device = _int(getattr(entry, "device_tokens", None))
            host = _int(getattr(entry, "host_tokens", None))
            source = "existing_lookahead"
        if phase == "inflight" and input_tokens is not None:
            # This queue is entered only after the final prefill result was
            # processed. late_d_computed_end tracks middle chunks, so it may
            # be zero for a one-pass prefill or omit the last chunk. Binding
            # and KV transfer may still be pending, but compute is complete.
            committed, inflight, device = input_tokens, 0, input_tokens
            remaining, source = 0, "completed_prefill_transfer_queue"
        elif id(req) in launched and launched[id(req)] is not None:
            committed, inflight = launched[id(req)]
            device, source = committed, "batch_host_lengths"
            if input_tokens is not None:
                remaining = max(0, input_tokens - committed - inflight)
        elif phase in ("partial", "parked", "inflight") and id(req) not in launched:
            end = _int(getattr(req, "late_d_computed_end", None))
            if end is not None and end > 0:
                committed, inflight, device = end, 0, end
                source = "processed_need_result"
                if input_tokens is not None:
                    remaining = max(0, input_tokens - committed)
        if (
            remaining is None
            and phase not in ("partial", "parked", "inflight")
            and None not in (input_tokens, device, host)
        ):
            remaining = max(0, input_tokens - device - host)
        storage = (
            _int(getattr(req, "storage_hit_length", None))
            if entry is not None or committed is not None
            else None
        )
        pending = _bool(getattr(req, "pending_bootstrap", None))
        park_start = getattr(req, "late_d_park_started", None)
        park_end = getattr(req, "late_d_park_ended", None)
        parked = (
            (park_start > 0 and (not park_end or park_end < park_start))
            if type(park_start) in (int, float)
            else None
        )
        ep1_parked = getattr(req, "ep1_parked", False)
        if type(ep1_parked) in (int, bool) and ep1_parked > 0:
            parked = True
        pulls = getattr(
            getattr(scheduler, "tree_cache", None), "ongoing_prefetch", None
        )
        state = RoutingState(
            intent_start_ok=_bool(getattr(req, "late_d_start_ok", None)),
            bound=not pending if pending is not None else None,
            launched_unbound=_bool(getattr(req, "launched_unbound", None)),
            gate_blocked=_bool(getattr(req, "late_d_gate_blocked", None)),
            held=_bool(getattr(req, "late_d_hold", None)),
            parked=parked,
            pull_pending=_bool(getattr(req, "jit_pull_pending", None)),
            pull_inflight=(
                rid in pulls if isinstance(pulls, dict) and rid is not None else None
            ),
            restore_inflight=id(req) in restore_ids if restore_known else None,
        )
        if self.symmetric and phase == "bootstrap":
            # A held P-leg waits for its reader; it is in no queue.
            state.held = True
        times = getattr(req, "time_stats", None)
        row = RoutingRequest(
            rid=rid,
            phase="parked" if parked else phase,
            input_tokens=input_tokens,
            device_prefix_tokens=device,
            host_prefix_tokens=host,
            storage_prefix_tokens=storage,
            committed_prefix_tokens=committed,
            inflight_new_tokens=inflight,
            remaining_compute_tokens=remaining,
            recv_age_s=_age(getattr(times, "scheduler_recv_time", None), now),
            queue_age_s=_age(getattr(times, "wait_queue_entry_time", None), now),
            prefix_source=source,
            state=state,
            storage_hit_tokens=storage,
        )
        # All ages below identify observed starts. They are not ETAs. Reading
        # scalar metadata never advances a pull, polls an ack, or inspects KV.
        admit_stamp = getattr(times, "late_d_admit_time", None)
        admit_age = _age(admit_stamp, now)
        if state.bound is True:
            row.bind_phase = "bound"
            row.bind_phase_age_s = _age(getattr(req, "late_d_bind_perf", None), now)
            row.bind_phase_age_source = (
                "bound" if row.bind_phase_age_s is not None else None
            )
        elif state.bound is False:
            if admit_age is not None:
                row.bind_phase, row.bind_phase_age_s = "admit_pending", admit_age
                row.bind_phase_age_source = "admit"
            elif type(admit_stamp) in (int, float) and admit_stamp == 0:
                row.bind_phase, row.bind_phase_age_s = "before_admit", row.recv_age_s
                row.bind_phase_age_source = "p_receive"
        issued = getattr(req, "jit_pull_issued", None)
        if state.pull_inflight is True:
            row.pull_phase = "inflight"
            row.pull_phase_age_s = _age(getattr(issued, "t_issue", None), now)
            if row.pull_phase_age_s is not None:
                row.pull_phase_age_source = "jit_issue"
            else:
                info = pulls.get(rid)
                op = getattr(info, "operation", None)
                row.pull_phase_age_s = _age(getattr(op, "start_time", None), now)
                if row.pull_phase_age_s is not None:
                    row.pull_phase_age_source = "operation_created"
        elif state.pull_pending is True:
            row.pull_phase = "pending"  # no exact deferred-start stamp exists
        if state.restore_inflight is True:
            row.restore_phase_age_s = _age(restore_ids.get(id(req)), now)
        # Remaining bytes/tokens are intentionally unknown: shared restore ops
        # and CPU-local pull progress do not prove per-request/all-rank residue.
        row.storage_prefix_tokens = (
            0 if state.pull_pending is False and state.pull_inflight is False else None
        )
        if entry is not None:
            row.plan = RoutingPlan(
                sampled_pass=_int(getattr(plan, "pass_index", None)),
                **{
                    name: getattr(entry, name, None)
                    for name in (
                        "start_pass",
                        "end_pass",
                        "gate_open_start_pass",
                        "gated",
                        "skip_reason",
                        "uncached_tokens",
                        "host_demand",
                        "pull_tokens",
                    )
                },
            )
        clear = all(
            flag is False
            for flag in (
                state.gate_blocked,
                state.held,
                state.parked,
                state.pull_pending,
                state.pull_inflight,
                state.restore_inflight,
            )
        )
        if clear and (state.bound is True or state.intent_start_ok is True):
            row.ready_after_s = 0.0
        if rid is None:
            row.unknown.append("rid_missing_or_too_long")
        if remaining is None:
            row.unknown.append("compute_not_observed")
        if committed is None:
            row.unknown.append("committed_prefix_not_observed")
        return row

    def capture(self, scheduler, batch, *, now=None, wall=None):
        now = time.monotonic() if now is None else now
        if now - self.last_capture < self.interval_s:
            return False
        started = time.perf_counter()
        self.last_capture = now
        self.sequence += 1
        self.value = self.build(
            scheduler,
            batch,
            now=now,
            wall=wall,
            sequence=self.sequence,
            started=started,
        )
        if self.symmetric:
            self._symmetric_stats(self.value, now)
        return True

    def build(
        self,
        scheduler,
        batch,
        *,
        now=None,
        wall=None,
        sequence=None,
        started=None,
        result_queue=None,
    ):
        """Return one bounded snapshot of the current state.

        capture() calls this after its sample throttle. The opt-in state push
        (prefill_state_push.py) calls it for its own messages; at a pass
        launch it passes ``result_queue`` = the overlap result queue plus the
        pass being launched (the loop appends that pass after the launch).
        This method does not change the sample throttle, the sequence or the
        published value, so a push does not change /v1/loads.
        """
        started = time.perf_counter() if started is None else started
        now = time.monotonic() if now is None else now
        sequence = self.sequence if sequence is None else sequence
        waiting = scheduler.waiting_queue
        boot, inflight, plan = self._sources(scheduler, now)
        partial = getattr(scheduler, "chunked_req", None)
        share = getattr(getattr(scheduler, "chunk_share", None), "cfg", None)
        args = scheduler.server_args
        current, launched = self._batch(batch, now, scheduler)
        pending_batches = []
        queue = (
            getattr(scheduler, "result_queue", None)
            if result_queue is None
            else result_queue
        )
        queue_known = isinstance(queue, (deque, list, tuple))
        total_batches = len(queue) if queue_known else 0
        pending_launched = {}
        if queue_known:
            seen = set()
            for pair in islice(queue, self.max_records):
                observed_batch = (
                    pair[0] if isinstance(pair, (tuple, list)) and pair else None
                )
                item, ranges = self._batch(observed_batch, now, scheduler)
                if item is None or item.forward_iter is None:
                    queue_known = False
                    continue
                if item.forward_iter in seen:
                    continue
                seen.add(item.forward_iter)
                if item.status != "pending":
                    queue_known = False
                    continue
                pending_batches.append(item)
                for key, (prefix, extend) in ranges.items():
                    old = pending_launched.get(key)
                    if key not in pending_launched:
                        pending_launched[key] = (prefix, extend)
                    elif old is not None and old[0] + old[1] == prefix:
                        pending_launched[key] = (old[0], old[1] + extend)
                    else:
                        pending_launched[key] = None
            launched.update(pending_launched)
        elif not getattr(scheduler, "enable_overlap", False):
            queue_known = True
            if current is not None and current.status == "pending":
                pending_batches = [current]
                total_batches = 1
        ops = getattr(getattr(scheduler, "bootstrap_prefetch", None), "ops", None)
        restore_ids = {}
        restore_known = isinstance(ops, list) and len(ops) <= self.max_records
        if restore_known:
            for op in ops:
                reqs = getattr(op, "reqs", ())
                if not isinstance(reqs, tuple) or len(reqs) > self.max_records:
                    restore_known = False
                    break
                stamp = getattr(op, "submitted", None)
                for req in reqs:
                    key = id(req)
                    if key not in restore_ids:
                        restore_ids[key] = stamp
                    else:
                        old = restore_ids[key]
                        restore_ids[key] = (
                            min(old, stamp)
                            if _age(old, now) is not None
                            and _age(stamp, now) is not None
                            else None
                        )
        value = PrefillRoutingSnapshot(
            epoch=self.epoch,
            sequence=sequence,
            sampled_unix_s=time.time() if wall is None else wall,
            sampled_monotonic_s=now,
            clock_domain_id=clock_domain(),
            scheduler_pass=_int(getattr(plan, "pass_index", None)),
            available=True,
            reason=None,
            age_s=0.0,
            stale=False,
            max_age_s=self.max_age_s,
            launch_capture_failures=self.launch_capture_failures,
            timeline=self.timeline.snapshot(now) if self.timeline is not None else None,
            admission_resources=admission_resources(
                scheduler, self.adder_observation, now
            ),
            limits={"max_records": self.max_records, "max_bytes": self.max_bytes},
            total_waiting=len(waiting),
            total_bootstrap=len(boot) if isinstance(boot, list) else 0,
            partial_known=True,
            current_batch_known=batch is None or current.forward_iter is not None,
            current_batch=current,
            prev=self.prev,
            inflight_batches=pending_batches,
            inflight_batches_known=queue_known,
            total_inflight_batches=total_batches,
            omitted_inflight_batches=max(0, total_batches - len(pending_batches)),
            inflight_requests_known=isinstance(inflight, list),
            total_inflight_requests=len(inflight) if isinstance(inflight, list) else 0,
            engine_revision=_short_string(os.getenv("WXU_ENGINE_REVISION")),
            model_id=_short_string(getattr(args, "served_model_name", None)),
            policy=RoutingPolicy(
                chunk_tokens=_int(getattr(scheduler, "chunked_prefill_size", None)),
                page_size=_int(getattr(scheduler, "page_size", None)),
                max_input_tokens=_int(getattr(scheduler, "max_prefill_tokens", None)),
                share_enabled=share is not None,
                share_unit=_int(getattr(share, "unit", None)),
                max_reqs=_int(
                    getattr(share, "max_reqs", None)
                    if share is not None
                    else getattr(scheduler, "max_running_requests", None)
                ),
                warm_max_uncached=_int(getattr(share, "warm_max_uncached", None)),
                age_passes=_int(getattr(share, "age_passes", None)),
                schedule_policy=_short_string(getattr(args, "schedule_policy", None)),
            ),
        )

        def row(req, phase):
            return self._request(
                req, phase, scheduler, plan, launched, restore_ids, restore_known, now
            )

        left = self.max_records
        if partial is not None:
            value.partial = row(partial, "partial")
            left -= 1
        value.waiting = [row(req, "waiting") for req in islice(waiting, left)]
        left -= len(value.waiting)
        if isinstance(boot, list):
            value.bootstrap = [
                row(getattr(req, "req", req), "bootstrap") for req in islice(boot, left)
            ]
        else:
            value.available, value.reason = False, "bootstrap_queue_unavailable"
        left -= len(value.bootstrap)
        if isinstance(inflight, list):
            value.inflight_requests = [
                row(getattr(req, "req", req), "inflight")
                for req in islice(inflight, left)
            ]
        # Bounded message size, including adversarial request IDs. Nested structs
        # use compact arrays on SHM; public_snapshot restores named JSON fields.
        while True:
            value.omitted_waiting = value.total_waiting - len(value.waiting)
            value.omitted_bootstrap = value.total_bootstrap - len(value.bootstrap)
            value.omitted_inflight_requests = value.total_inflight_requests - len(
                value.inflight_requests
            )
            value.omitted_inflight_batches = value.total_inflight_batches - len(
                value.inflight_batches
            )
            value.truncated = bool(
                value.omitted_waiting
                or value.omitted_bootstrap
                or value.omitted_inflight_requests
                or value.omitted_inflight_batches
                or (current and current.omitted_requests)
                or not value.partial_known
                or not value.current_batch_known
            )
            encoded = len(msgspec.msgpack.encode(value))
            if encoded <= self.max_bytes - 64:
                break
            if value.inflight_requests:
                value.inflight_requests.pop()
            elif value.bootstrap:
                value.bootstrap.pop()
            elif value.waiting:
                value.waiting.pop()
            elif value.partial is not None:
                value.partial, value.partial_known = None, False
            elif value.inflight_batches:
                value.inflight_batches.pop()
            elif value.current_batch is not None:
                value.current_batch, value.current_batch_known = None, False
            else:
                value.available, value.reason = False, "payload_budget_exhausted"
                break
        value.capture_us = (time.perf_counter() - started) * 1e6
        value.wire_bytes = len(msgspec.msgpack.encode(value))
        return value

    def _sources(self, scheduler, now):
        """(bootstrap rows, inflight rows, plan) of one capture; a row source
        that is not a list makes the snapshot unavailable (bootstrap) or its
        rows unknown (inflight).

        PD prefill: the bootstrap queue, the inflight (KV transfer) queue and
        the late-D lookahead plan. Symmetric worker: the P-legs held for their
        reader, the computed exports whose KV the reader has not released,
        and the scheduler's plan or else the private one."""
        if not self.symmetric:
            boot = getattr(
                getattr(scheduler, "disagg_prefill_bootstrap_queue", None),
                "queue",
                None,
            )
            inflight = getattr(scheduler, "disagg_prefill_inflight_queue", None)
            plan = getattr(getattr(scheduler, "late_d_lookahead", None), "plan", None)
            return boot, inflight, plan
        table = getattr(scheduler, "export_table", None)
        boot = table.held_reqs() if table is not None else None
        inflight = table.computed_reqs() if table is not None else None
        return boot, inflight, self._symmetric_plan(scheduler, now)

    def _symmetric_plan(self, scheduler, now):
        view = getattr(getattr(scheduler, "bootstrap_prefetch", None), "plan_view", None)
        plan = view() if callable(view) else None
        if plan is not None:
            # BWP v2 C3: the prefetcher's private plan (read-only here).
            self.plan_source = "prefetcher"
            return plan
        owner = getattr(scheduler, "late_d_lookahead", None)
        if owner is not None:
            # SGLANG_SYMMETRIC_BWP: updated by the loop after every result.
            self.plan_source = "scheduler"
            return getattr(owner, "plan", None)
        try:
            if self._private_plan is None:
                from decoupled_engine.srt.disaggregation.lookahead_plan import (
                    SchedulerLookahead,
                )

                # Never scheduler.late_d_lookahead: JIT pull and the BWP
                # prefetcher read that attribute.
                self._private_plan = SchedulerLookahead(scheduler, log=False)
            if now - self._private_plan_at >= self.interval_s:
                self._private_plan_at = now
                self._private_plan.update()
        except Exception:
            # Unknown prefixes: the rows stay, their prefix fields are null.
            self.plan_source = "none"
            return None
        self.plan_source = "private"
        return self._private_plan.plan

    def _symmetric_stats(self, value, now):
        st = self.sym_stats
        st["captures"] += 1
        st["unavailable"] += int(not value.available)
        st["bootstrap_rows_max"] = max(st["bootstrap_rows_max"], len(value.bootstrap))
        st["inflight_rows_max"] = max(
            st["inflight_rows_max"], len(value.inflight_requests)
        )
        st["capture_us"] += value.capture_us or 0.0
        st["wire_bytes_max"] = max(st["wire_bytes_max"], value.wire_bytes)
        if now - self._sym_log_at < SYM_LOG_PERIOD_S:
            return
        self._sym_log_at = now
        logger.info(self.symmetric_stats_line())

    def symmetric_stats_line(self):
        st = self.sym_stats
        return (
            f"{SYM_LOG_TAG} captures={st['captures']} "
            f"unavailable={st['unavailable']} plan={self.plan_source} "
            f"bootstrap_rows_max={st['bootstrap_rows_max']} "
            f"inflight_rows_max={st['inflight_rows_max']} "
            f"decode_passes={self.decode_passes} "
            f"capture_us_mean={st['capture_us'] / max(1, st['captures']):.1f} "
            f"wire_bytes_max={st['wire_bytes_max']}"
        )

    def unavailable(self, reason):
        if self.symmetric:
            self.sym_stats["unavailable"] += 1
        self.value = PrefillRoutingSnapshot(
            epoch=self.epoch,
            sequence=self.sequence,
            sampled_monotonic_s=time.monotonic(),
            sampled_unix_s=time.time(),
            clock_domain_id=clock_domain(),
            reason=reason,
            max_age_s=self.max_age_s,
        )

    def snapshot(self):
        return self.value
