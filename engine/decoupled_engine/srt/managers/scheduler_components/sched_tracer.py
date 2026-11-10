"""Scheduler JSONL trace streams (per-iteration + per-finished-request).

Enabled by ``--sched-trace-dir``. The tracer is constructed only on
``attn_tp_rank == 0`` (mirroring the request-time-stats gate), so one
scheduler group writes one file pair:

- ``iter_trace.dp{dp}_pp{pp}.pid{pid}.jsonl`` — one record per forward
  iteration, emitted at the end of ``Scheduler.process_batch_result``.
  With the overlap schedule the record for iteration N is emitted one
  loop pass later; ``dur_ms`` therefore measures launch -> result
  processed, which includes the overlap window by design.
- ``req_trace.dp{dp}_pp{pp}.pid{pid}.jsonl`` — one record per finished
  request, emitted from ``SchedulerOutputStreamer`` next to the existing
  time-stats log. Timestamps are ``time.perf_counter()`` values copied
  from ``SchedulerReqTimeStats``; add ``mono_to_unix`` to convert.

Files are opened unbuffered (one write per record), so records survive a
crash without a shutdown hook.

Field names are a stable contract consumed by offline analysis scripts;
treat renames as breaking changes.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import msgspec

from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.managers.scheduler_components.phase_timers import PHASES
from decoupled_engine.srt.managers.utils import GenerationBatchResult
from decoupled_engine.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from decoupled_engine.srt.managers.schedule_batch import Req, ScheduleBatch


class IterTraceRecord(msgspec.Struct, gc=False):
    forward_iter: int
    ts: float  # wall clock (time.time()) at emission
    launch_mono: float  # batch.launch_ts (time.monotonic() in run_batch)
    dur_ms: float  # launch -> result processed (monotonic delta)
    mode: str  # forward_mode name, lowercase
    bs: int  # requests in this batch
    num_waiting_reqs: int
    seq_lens_sum: int
    after_idle: bool
    retract_ct: int  # cumulative retracted requests seen by this scheduler
    new_prefill_tokens: int = 0  # EXTEND/MIXED batches only
    decode_tokens: int = 0  # DECODE/TARGET_VERIFY: bs + num_correct_drafts
    num_correct_drafts: int = 0  # speculative: accepted drafts, no bonus
    spec_num_draft_tokens: Optional[int] = None  # speculative draft stride
    # Launch snapshots: request/batch fields can change before overlap results.
    launch_bs: Optional[int] = None
    launch_seq_lens_sum: Optional[int] = None
    prefill_attention_pairs: Optional[int] = None
    launch_context_source: Optional[str] = None
    # Whole forward stream interval, including MTP and stream waits; not a
    # sum of kernel durations. Never synchronize just to obtain this metric.
    forward_gpu_ms: Optional[float] = None
    # SGLANG_SCHED_PHASE_TIMERS (overlap load-back M0): perf_counter ms per
    # scheduler phase summed since the previous record (None when off).
    phase_ms: Optional[Dict[str, float]] = None
    # SGLANG_DSA_HOST_DEDUP_LOAD_TIMING (M0): load-back ops of this rank not
    # yet finished when the batch launched (event queries; None when off),
    # and whether the batch consumes a load-back op (R8 successor type).
    load_in_flight: Optional[int] = None
    consumes_load: Optional[bool] = None
    # SGLANG_SCHED_TRACE_GPU_TIME (M0, R8): GPU gap from the previous traced
    # forward's end event to this forward's start event.
    gpu_gap_before_ms: Optional[float] = None
    # SGLANG_DSA_HOST_DEDUP_LOAD_TIMING (M0): GPU stall of this forward in
    # load waits (whole-op gate or layer parts; arrive before the CPU wait).
    load_stall_gpu_ms: Optional[float] = None
    vision_matmul_flops: int = 0


class ReqTraceRecord(msgspec.Struct, gc=False):
    rid: str
    ts: float  # wall clock at emission
    mono_to_unix: float  # time.time() - time.perf_counter() at emission
    input_len: int
    output_len: int
    finish_reason: Optional[dict]
    # time.perf_counter() stamps from SchedulerReqTimeStats
    scheduler_recv_time: float
    wait_queue_entry_time: float
    forward_entry_time: float
    prefill_finished_time: float
    completion_time: float
    last_decode_finish_time: float
    decode_ct: int
    # cache reuse breakdown (tokens)
    cached_tokens: int
    cached_tokens_device: int
    cached_tokens_host: int
    cached_tokens_storage: int
    # chunked prefill: forward_iter of every EXTEND/MIXED pass for this req
    prefill_iters: List[int]
    num_prefill_chunks: int
    retraction_count: int
    # speculative decoding, cumulative per request
    spec_verify_ct: int
    spec_num_correct_drafts: int
    spec_num_block_accept_tokens: int
    spec_num_cap_tokens: int
    spec_correct_drafts_histogram: List[int]
    # Planned length is also available on P, avoiding completed-D length bias
    # in fixed-window capacity calibration with ignore_eos trace replay.
    requested_output_len: Optional[int] = None
    # Host hit tokens recomputed because the load-back restored less.
    host_hit_recomputed_tokens: int = 0
    # P side: the decode worker's prefix length at bootstrap (tokens).
    d_prefix_len: int = 0
    # D-local allocation evidence, independent of cache counts received from P.
    # None means unobserved (including when the observation flag is off).
    decode_prefix_observation: Optional[dict] = None
    vision_matmul_flops: int = 0
    image_attention_pairs: int = 0


class NeedReqTraceRecord(ReqTraceRecord, gc=False):
    """Compute-before-bind state, emitted only by the enabled path."""

    late_d_need: dict = msgspec.field(default_factory=dict)


@dataclass
class _VisionWork:
    reqs: list[object]
    before: int
    flops: int = 0


class WindowReqTraceRecord(ReqTraceRecord, gc=False):
    """Per-request record under DSv4 window SWA residency; other engines keep
    emitting the plain record unchanged. The request fields behind these read
    0 until the window ledger and backfill paths set them."""

    swa_host_hit_tokens: int = 0
    swa_backfill_pages: int = 0
    swa_insert_truncated_tokens: int = 0


_json_encoder = msgspec.json.Encoder()


def _is_real_prefill(mode: ForwardMode) -> bool:
    # is_extend() also matches TARGET_VERIFY / SPLIT_PREFILL / DLLM_EXTEND;
    # chunk accounting only wants passes that prefill prompt tokens.
    return mode == ForwardMode.EXTEND or mode == ForwardMode.MIXED


class SchedTracer:
    """Writes the two JSONL trace streams; one instance per traced scheduler."""

    def __init__(
        self, *, trace_dir: str, dp_rank: int, pp_rank: int, dsv4_window: bool = False
    ) -> None:
        self._dsv4_window = dsv4_window
        os.makedirs(trace_dir, exist_ok=True)
        suffix = f"dp{dp_rank}_pp{pp_rank}.pid{os.getpid()}"
        self._iter_file = open(
            os.path.join(trace_dir, f"iter_trace.{suffix}.jsonl"), "ab", buffering=0
        )
        self._req_file = open(
            os.path.join(trace_dir, f"req_trace.{suffix}.jsonl"), "ab", buffering=0
        )
        self._retract_ct = 0
        self._launch_features = {}
        self._vision_work = {}
        self._gpu_intervals = {}
        self._gpu_timing = envs.SGLANG_SCHED_TRACE_GPU_TIME.get()
        # Set by the scheduler with SGLANG_DSA_HOST_DEDUP_LOAD_TIMING:
        # () -> load-back ops in flight at launch (logging only).
        self.loads_probe = None
        self._launch_loads = {}
        self._prev_gpu_end = None
        # Set with loads_probe: () -> this forward's stall event pairs.
        self.stall_probe = None
        self._fwd_stalls = {}

    def on_retract(self, *, num_reqs: int) -> None:
        self._retract_ct += num_reqs

    @property
    def gpu_timing_enabled(self) -> bool:
        return self._gpu_timing

    def gpu_events(self, forward_iter):
        """The (start, end) CUDA events of a pass whose record is not emitted
        yet (SGLANG_SCHED_TRACE_GPU_TIME), or None. Read-only: the opt-in
        prefill state push queries them without a synchronization."""
        return self._gpu_intervals.get(forward_iter)

    def on_run_batch(self, batch: ScheduleBatch, stream=None) -> None:
        # Called from Scheduler.run_batch right after forward_iter is set, so
        # a request finishing on its last prefill pass still records that pass
        # before its finish record is emitted.
        if _is_real_prefill(batch.forward_mode):
            for req in batch.reqs:
                req.prefill_iters.append(batch.forward_iter)
            self._vision_work[batch.forward_iter] = _VisionWork(
                list(batch.reqs),
                sum(getattr(req, "vision_matmul_flops", 0) for req in batch.reqs),
            )
        pairs = None
        if batch.forward_mode == ForwardMode.EXTEND:
            # Exact causal token/context pairs. A chunk of n tokens after h
            # prefix tokens attends n*h + n*(n+1)/2 positions. Sum over the
            # requests actually admitted to THIS chunk, not the whole queue.
            pairs = sum(
                n * h + n * (n + 1) // 2
                for h, n in zip(batch.prefix_lens, batch.extend_lens, strict=True)
            )
            pairs += getattr(batch, "prefill_image_attention_pairs", 0)
        context_sum = batch.seq_lens_sum
        context_source = "batch_sum"
        if context_sum is None:
            cpu_lengths = getattr(batch, "seq_lens_cpu", None)
            if cpu_lengths is not None:
                context_sum = int(cpu_lengths.sum())
                context_source = "cpu_lengths"
            else:
                # GPU-only speculative decode deliberately omits CPU lengths.
                # Use committed host state; under overlap it can lag one step.
                # Do not add a device synchronization just for telemetry.
                context_sum = sum(req.kv_committed_len for req in batch.reqs)
                context_source = "committed_kv"
        self._launch_features[batch.forward_iter] = (
            len(batch.reqs),
            context_sum,
            pairs,
            context_source,
        )
        if self.loads_probe is not None:
            consumer = getattr(batch, "hicache_consumer_index", -1)
            self._launch_loads[batch.forward_iter] = (
                self.loads_probe(),
                consumer is not None and consumer >= 0,
            )
        if self._gpu_timing and not (
            batch.forward_mode.is_prebuilt() or batch.forward_mode.is_idle()
        ):
            import torch

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            self._gpu_intervals[batch.forward_iter] = (start, end)

    def on_forward_end(self, batch: ScheduleBatch, stream=None) -> None:
        vision = self._vision_work.get(batch.forward_iter)
        if vision is not None:
            # Capture before another overlapping forward changes the same Req.
            vision.flops = (
                sum(getattr(req, "vision_matmul_flops", 0) for req in vision.reqs)
                - vision.before
            )
            vision.reqs.clear()
        interval = self._gpu_intervals.get(batch.forward_iter)
        if interval is not None:
            interval[1].record(stream)
        if self.stall_probe is not None:
            stall_pairs = self.stall_probe()
            if stall_pairs:
                self._fwd_stalls[batch.forward_iter] = stall_pairs

    def on_batch_result(
        self,
        *,
        batch: ScheduleBatch,
        result,
        num_waiting_reqs: int,
    ) -> None:
        if batch.forward_iter is None:
            return
        mode = batch.forward_mode
        # EmbeddingBatchResult carries no speculative fields.
        if isinstance(result, GenerationBatchResult):
            num_correct_drafts = result.num_correct_drafts
            spec_num_draft_tokens = result.speculative_num_draft_tokens
        else:
            num_correct_drafts = 0
            spec_num_draft_tokens = None
        bs = len(batch.reqs)
        new_prefill_tokens = 0
        decode_tokens = 0
        if _is_real_prefill(mode):
            new_prefill_tokens = batch.extend_num_tokens or 0
        elif mode.is_decode() or mode.is_target_verify():
            decode_tokens = bs + num_correct_drafts
        launch_bs, launch_seq_lens_sum, pairs, context_source = (
            self._launch_features.pop(batch.forward_iter, (None, None, None, None))
        )
        interval = self._gpu_intervals.pop(batch.forward_iter, None)
        vision = self._vision_work.pop(batch.forward_iter, None)
        gpu_ms = gap_ms = None
        if interval is not None and interval[1].query():
            gpu_ms = interval[0].elapsed_time(interval[1])
            if self._prev_gpu_end is not None:
                gap_ms = self._prev_gpu_end.elapsed_time(interval[0])
            self._prev_gpu_end = interval[1]
        loads, consumes = self._launch_loads.pop(batch.forward_iter, (None, None))
        stall_ms = None
        # Not "pairs": that name holds this pass's prefill_attention_pairs.
        stall_pairs = self._fwd_stalls.pop(batch.forward_iter, None)
        if stall_pairs is not None and all(b.query() for _, b in stall_pairs):
            stall_ms = sum(max(0.0, a.elapsed_time(b)) for a, b in stall_pairs)
        elif self.stall_probe is not None and consumes:
            stall_ms = (
                0.0 if stall_pairs is None else None
            )  # no load wait; or unresolved
        record = IterTraceRecord(
            forward_iter=batch.forward_iter,
            ts=time.time(),
            launch_mono=batch.launch_ts,
            dur_ms=(time.monotonic() - batch.launch_ts) * 1e3,
            mode=mode.name.lower(),
            bs=bs,
            num_waiting_reqs=num_waiting_reqs,
            seq_lens_sum=batch.seq_lens_sum or 0,
            after_idle=batch.after_idle_gap,
            retract_ct=self._retract_ct,
            new_prefill_tokens=new_prefill_tokens,
            decode_tokens=decode_tokens,
            num_correct_drafts=num_correct_drafts,
            spec_num_draft_tokens=spec_num_draft_tokens,
            launch_bs=launch_bs,
            launch_seq_lens_sum=launch_seq_lens_sum,
            prefill_attention_pairs=pairs,
            launch_context_source=context_source,
            forward_gpu_ms=gpu_ms,
            phase_ms=PHASES.take(),
            load_in_flight=loads,
            consumes_load=consumes,
            gpu_gap_before_ms=gap_ms,
            load_stall_gpu_ms=stall_ms,
            vision_matmul_flops=vision.flops if vision is not None else 0,
        )
        self._iter_file.write(_json_encoder.encode(record) + b"\n")

    def on_request_finish(self, *, req: Req) -> None:
        if req.has_sched_trace_record:
            return
        req.has_sched_trace_record = True
        t = req.time_stats
        record_cls, window_fields = ReqTraceRecord, {}
        if getattr(req, "late_d_need", False):
            record_cls = NeedReqTraceRecord
            fields = (
                "launched_unbound",
                "late_d_entry_pass",
                "late_d_first_launch_perf",
                "late_d_bind_perf",
                "late_d_park_started",
                "late_d_park_ended",
                "late_d_park_timeout",
                "late_d_device_prefix_sent_tokens",
                "late_d_device_prefix_after_forward_tokens",
                "late_d_gate_skips",
                "late_d_intent_skips",
                "late_d_hold_skips",
                "late_d_catchup_sent_tokens",
                "late_d_arrival_host_tokens",
            )
            window_fields["late_d_need"] = {
                name: getattr(req, name, None) for name in fields
            }
        if self._dsv4_window:
            record_cls = WindowReqTraceRecord
            window_fields = dict(
                swa_host_hit_tokens=getattr(req, "swa_host_hit_tokens", 0),
                swa_backfill_pages=getattr(req, "swa_backfill_pages", 0),
                swa_insert_truncated_tokens=getattr(
                    req, "swa_insert_truncated_tokens", 0
                ),
            )
        record = record_cls(
            rid=req.rid,
            ts=time.time(),
            mono_to_unix=time.time() - time.perf_counter(),
            input_len=len(req.origin_input_ids),
            output_len=len(req.output_ids),
            finish_reason=(
                req.finished_reason.to_json()
                if req.finished_reason is not None
                else None
            ),
            scheduler_recv_time=t.scheduler_recv_time,
            wait_queue_entry_time=t.wait_queue_entry_time,
            forward_entry_time=t.forward_entry_time,
            prefill_finished_time=t.prefill_finished_time,
            completion_time=t.completion_time,
            last_decode_finish_time=t.last_decode_finish_time,
            decode_ct=t.decode_ct,
            cached_tokens=req.cached_tokens,
            vision_matmul_flops=getattr(req, "vision_matmul_flops", 0),
            image_attention_pairs=getattr(req, "image_attention_pairs", 0),
            cached_tokens_device=req.cached_tokens_device,
            cached_tokens_host=req.cached_tokens_host,
            cached_tokens_storage=req.cached_tokens_storage,
            prefill_iters=req.prefill_iters,
            num_prefill_chunks=len(req.prefill_iters),
            retraction_count=req.retraction_count,
            spec_verify_ct=req.spec_verify_ct,
            spec_num_correct_drafts=req.spec_num_correct_drafts,
            spec_num_block_accept_tokens=req.spec_num_block_accept_tokens,
            spec_num_cap_tokens=req.spec_num_cap_tokens,
            spec_correct_drafts_histogram=req.spec_correct_drafts_histogram,
            requested_output_len=getattr(
                req,
                "sched_trace_requested_output_len",
                req.sampling_params.max_new_tokens,
            ),
            host_hit_recomputed_tokens=getattr(req, "host_hit_recomputed_tokens", 0),
            d_prefix_len=getattr(req, "sched_trace_d_prefix_len", 0),
            decode_prefix_observation=getattr(
                req, "sched_trace_decode_prefix_observation", None
            ),
            **window_fields,
        )
        self._req_file.write(_json_encoder.encode(record) + b"\n")
