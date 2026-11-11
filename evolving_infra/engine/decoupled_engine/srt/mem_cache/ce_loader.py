"""Shared host-dedup loads submitted by a per-rank loader thread.

Copy-engine loads block the submitting thread once the stream queue fills,
and the HiCache load fence (``h2d.wait_stream(forward_stream)``) keeps them
queued until the in-flight forward ends. On the scheduler thread that held
the event loop for (rest of forward + copy). Here the scheduler only builds a
job; the loader thread does the index D2H, the cross-rank row check, the 2D
copies (paced so no call blocks on a full queue), and records the layer and
ack events.

Hazards and how they are closed:

- A stream wait on a not-yet-recorded event is a no-op, and ``query()`` on a
  never-recorded event returns True. Every event the loader records is
  handed out as a ``GatedEvent``: ``query`` is False and ``wait`` /
  ``synchronize`` / ``elapsed_time`` block (CPU, GIL released) until the job
  is submitted. The consuming forward's whole-op wait runs at forward launch
  (``LayerDoneCounter.set_consumer``), not at load submission.
- A loader failure marks the job and every later gate raises (fail loudly).
- No GPU-side handoff (stream wait-value): with a Python loader, a GIL-holding
  blocking CUDA call on a dependent stream could hang the engine.

Background jobs (the bootstrap-window prefetch): nothing waits for them until
admission, but they run next to forwards, and a C2g follower's NVLink peer
reads slow the forward's all-reduce (measured on GB300: +2.6% p50 uncapped,
+1.5% at 50 GB/s). A follower caps a background job's peer reads at
SGLANG_DSA_HOST_DEDUP_BWP_PEER_RATE_GBPS through a live rate word, and lifts
the cap as soon as an admission load is submitted behind it (a forward will
wait for that one; the queue is FIFO), or when admission defers a request on
(or adopts) a prefetch op (``lift_background_cap``: that request now waits for
the op's rows, and for every background job ahead of it in the FIFO, so those
are lifted too). Admission loads keep SGLANG_DSA_HOST_DEDUP_PEER_RATE_GBPS
(default uncapped). Every rank runs the same job sequence (C2g op seq):
``submit`` never drops or defers a job, and the loader checks that it runs them
in submission order.

Adaptive cap (SGLANG_DSA_HOST_DEDUP_BWP_ADAPTIVE_CAP):
the capped job also gets this rank's forward-busy word (written by the
bootstrap prefetcher once per scheduler pass) and its own deadline word
(when its window is estimated to end); the native pacer applies the cap only
while a forward is in flight and the rest of the op fits the cap before the
deadline. Both words are rank-local and change timing only.
"""

from __future__ import annotations

import ctypes
import logging
import queue
import threading
import time
from typing import Any, List, Optional

import torch

logger = logging.getLogger(__name__)


class LoadJob:
    __slots__ = (
        "transfers",
        "layer_num",
        "start_event",
        "fence_event",
        "ack_start",
        "ack_finish",
        "on_layer_done",
        "submitted",
        "error",
        "t_enqueue",
        "per_layer_fallback",
        "peer_plans",
        "background",
        "lifted",
        "parts",
        "part_events",
        "enqueued_parts",
        "gate_timeout_s",
        "seq0",
        "part_plan",
        "started",
        "deadline_ns",
        "index",
        "q3",
        "dirty_event",
    )

    def __init__(
        self,
        transfers,
        layer_num,
        start_event,
        fence_event,
        ack_start,
        ack_finish,
        on_layer_done,
        per_layer_fallback,
        peer_plans=None,
        background=False,
        parts=None,
        part_events=None,
        gate_timeout_s=120.0,
        part_plan="page",
        q3=False,
        dirty_event=None,
    ):
        self.transfers = transfers
        self.layer_num = layer_num
        self.start_event = start_event
        self.fence_event = fence_event
        self.ack_start = ack_start
        self.ack_finish = ack_finish
        self.on_layer_done = on_layer_done
        self.per_layer_fallback = per_layer_fallback
        # C2g follower: transfers -> 2D peer-copy plans from the source's GPU.
        self.peer_plans = peer_plans
        # Bootstrap-window prefetch: no forward consumes it (capped peer reads).
        self.background = bool(background)
        self.lifted = False  # someone waits for its rows: run uncapped
        # Set by the loader thread when it picks the job up (the bootstrap
        # prefetcher MIN-reduces it to bound queued background jobs).
        self.started = False
        # Adaptive cap: CLOCK_MONOTONIC ns by which the op should be done if
        # it is to stay capped (0 = no known slack: uncapped). Rank-local.
        self.deadline_ns = ctypes.c_uint64(0)
        self.index: Optional[int] = None  # submission order (SharedLoader.submit)
        # Q3 (load quarantine): CPU destination ids and clean pages; skip
        # start_event and the fence, wait on dirty_event (if any) instead.
        self.q3 = bool(q3)
        self.dirty_event = dirty_event
        self.submitted = threading.Event()
        self.error: Optional[BaseException] = None
        self.t_enqueue = time.perf_counter()
        # Layer-group consumption: layer ranges [(l0, l1), ...] and one
        # device event per part, recorded by the loader thread after the
        # part's KV + indexer copies (the MTP draft rows go in the last part).
        # None = whole op. enqueued_parts is written by the loader thread only.
        self.parts = parts
        self.part_events = part_events
        self.enqueued_parts = 0
        self.gate_timeout_s = float(gate_timeout_s)
        self.seq0 = None  # first C2g seq of the op's seq block (loader thread)
        # Copy shape of a layer-wise op (PART_PLAN): "page" = one copy per page
        # and layer run (ce_2d_load.op_arrays), "hybrid" = hybrid copies on the
        # source, run-coalesced copies + whole-part gate on followers.
        self.part_plan = "hybrid" if parts and part_plan == "hybrid" else "page"

    def ready(self, block: bool) -> bool:
        if not self.submitted.is_set():
            if not block:
                return False
            t0 = time.perf_counter()
            self.submitted.wait()
            from decoupled_engine.srt.mem_cache.ce_2d_load import COST

            COST.add("gate_wait", (time.perf_counter() - t0) * 1e3)
        if self.error is not None:
            raise RuntimeError("Shared host-dedup loader failed") from self.error
        return True

    def wait_part_enqueued(self, p: int) -> None:
        """CPU-wait (GIL released, 50 us sleeps) until part ``p``'s event has
        been recorded by the loader thread (invariant I3: a stream wait on a
        not-yet-recorded event would be a silent no-op). Raises on a loader
        error or after ``gate_timeout_s``."""
        if self.enqueued_parts > p and self.error is None:
            return
        t0 = time.perf_counter()
        deadline = t0 + self.gate_timeout_s
        while self.enqueued_parts <= p:
            if self.error is not None or self.submitted.is_set():
                break
            if time.perf_counter() > deadline:
                raise RuntimeError(
                    f"Shared host-dedup loader: part {p} not enqueued after "
                    f"{self.gate_timeout_s:.0f} s"
                )
            time.sleep(5e-5)
        if self.error is not None:
            raise RuntimeError("Shared host-dedup loader failed") from self.error
        if self.enqueued_parts <= p:
            raise RuntimeError(
                f"Shared host-dedup loader: job ended with {self.enqueued_parts} "
                f"part(s) enqueued, part {p} missing"
            )
        from decoupled_engine.srt.mem_cache.ce_2d_load import COST

        COST.add("part_wait", (time.perf_counter() - t0) * 1e3)


def layer_part_bounds(layer_num: int, n_parts: int, spec: str = ""):
    """Layer ranges [(l0, l1), ...] of a layer-wise load op, or None (whole op).

    ``spec`` ("0,20,39,58,78") overrides the uniform split and must have
    ``n_parts + 1`` strictly increasing bounds from 0 to ``layer_num``."""
    n_parts = int(n_parts)
    if n_parts <= 1 or layer_num <= 1:
        return None
    if spec:
        b = [int(x) for x in str(spec).split(",") if x.strip()]
        if len(b) != n_parts + 1:
            raise ValueError(f"layer part bounds {spec!r}: need {n_parts + 1} values")
    else:
        n = min(n_parts, layer_num)
        b = [round(k * layer_num / n) for k in range(n + 1)]
    if b[0] != 0 or b[-1] != layer_num or any(x >= y for x, y in zip(b, b[1:])):
        raise ValueError(f"layer part bounds {b} invalid for {layer_num} layers")
    return list(zip(b[:-1], b[1:]))


# Part digest check: a part digest no real op has (real fields are >= -1), sent
# by a follower that cannot plan the op so every rank fails the check together.
_NO_PLAN_DIGEST = (-3, -3, -3, -3, -3)

# SGLANG_DSA_HOST_DEDUP_PART_PLAN values (copy shapes of layer-wise ops).
PART_PLANS = ("page", "hybrid")


def part_digest(part_lists, arrs_list, shape_free=False):
    """Part digest check: per part (2D copies, 1D copies, first and last device
    page, summed run heights). TP0 and followers must agree for the per-chunk
    gate.

    ``shape_free`` (PART_PLAN=hybrid: the source and the followers copy
    different shapes and followers gate whole parts): (bytes copied, pages,
    first and last device page, summed run heights) instead, so the ranks
    still agree on the part's rows and on every byte of them."""
    from decoupled_engine.srt.mem_cache.ce_2d_load import arrays_bytes

    out = []
    for plans, arrs in zip(part_lists, arrs_list):
        first = last = -1
        heights = pages = 0
        for pl in plans:
            if pl.dev_pages.numel():
                if first < 0:
                    first = int(pl.dev_pages[0])
                last = int(pl.dev_pages[-1])
            pages += int(pl.dev_pages.numel())
            heights += sum(int(r.height) for r in pl.runs)
        if shape_free:
            out.append((arrays_bytes(arrs), pages, first, last, heights))
            continue
        n1d = int(arrs["d1_dst"].size) if "d1_dst" in arrs else 0
        out.append((int(arrs["dst"].size), n1d, first, last, heights))
    return out


def validate_layerwise_config(fanout: Optional[str] = None) -> None:
    """Startup checks for layer-wise load-back (LAYER_PARTS > 1):
    the per-layer waits must run at replay, so the prefill graph is
    breakable or disabled (FULL captures whole bodies; tc_piecewise is
    unaudited); L-c, no NCCL fan-out loader (it hung on launch order while
    the scheduler blocks mid-forward). ``fanout``: the dedup's own fan-out
    when it has its own word (MLA: SGLANG_MLA_HOST_DEDUP_FANOUT); default
    SGLANG_DSA_HOST_DEDUP_FANOUT."""
    from decoupled_engine.srt.environ import envs

    plan = str(envs.SGLANG_DSA_HOST_DEDUP_PART_PLAN.get()).lower()
    if plan not in PART_PLANS:
        raise ValueError(
            f"SGLANG_DSA_HOST_DEDUP_PART_PLAN={plan!r}: expected one of {PART_PLANS}"
        )
    if envs.SGLANG_DSA_HOST_DEDUP_LAYER_PARTS.get() <= 1:
        return
    word = "SGLANG_DSA_HOST_DEDUP_FANOUT"
    if fanout is None:
        fanout = envs.SGLANG_DSA_HOST_DEDUP_FANOUT.get()
    else:
        word = "the host dedup fan-out (SGLANG_MLA_HOST_DEDUP_FANOUT for MLA)"
    if str(fanout).lower() == "nccl":
        raise ValueError(
            f"SGLANG_DSA_HOST_DEDUP_LAYER_PARTS > 1 needs {word} "
            "ce_peer or shared, not nccl"
        )
    from decoupled_engine.srt.model_executor.cuda_graph_config import (
        Backend,
        Phase,
        check_cuda_graph_backend,
    )

    for bad in (Backend.FULL, Backend.TC_PIECEWISE):
        if check_cuda_graph_backend(Phase.PREFILL, bad):
            raise ValueError(
                "SGLANG_DSA_HOST_DEDUP_LAYER_PARTS > 1 needs --cuda-graph-backend-prefill "
                f"breakable or disabled, not {bad}: its per-layer load waits would "
                "not run at replay"
            )


def overlap_self_test(
    dm, forward_stream, streams, *, nbytes=16 << 20, sleep_cycles=100_000_000
):
    """I9 (startup only, never on the load path): two dependent sleep kernels
    on ``forward_stream``, then one CE copy on each of ``streams``. Each copy
    must end before the first kernel does; a stream that shares an in-order
    hardware queue with the forward stream queues its copy behind them.
    16 MiB per copy (a few ms at H2D speed against a ~50 ms sleep) keeps the
    transient device allocation small after the KV pools are sized."""
    import torch

    src = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    dsts = [torch.empty(nbytes, dtype=torch.uint8, device="cuda") for _ in streams]
    t0, k1 = dm.Event(enable_timing=True), dm.Event(enable_timing=True)
    ends = [dm.Event(enable_timing=True) for _ in streams]
    with dm.stream(forward_stream):
        t0.record(forward_stream)
        torch.cuda._sleep(sleep_cycles)
        k1.record(forward_stream)
        torch.cuda._sleep(sleep_cycles)
    for st, dst, end in zip(streams, dsts, ends):
        with dm.stream(st):
            dst.copy_(src, non_blocking=True)
            end.record(st)
    torch.cuda.synchronize()
    k1_ms = t0.elapsed_time(k1)
    copy_ms = [t0.elapsed_time(e) for e in ends]
    return (
        all(c < k1_ms for c in copy_ms),
        f"kernel1 ends {k1_ms:.2f} ms, copies end {[round(c, 2) for c in copy_ms]} ms",
    )


def init_layerwise_loadback(
    cache_controller, forward_stream, draft_worker=None
) -> None:
    """Scheduler startup, once ``forward_stream`` exists (every rank): the
    layer-wise config checks and the loader's I9 overlap check / redraw."""
    dedup = getattr(cache_controller, "host_dedup", None)
    if dedup is None:
        return
    # MLA host dedup selects its fan-out with its own word (mla_host_dedup.py).
    fanout = getattr(dedup, "fanout", None)
    if fanout is None:
        validate_layerwise_config()
    else:
        validate_layerwise_config(fanout=fanout)
    from decoupled_engine.srt.environ import envs

    if (
        envs.SGLANG_DSA_HOST_DEDUP_LAYER_PARTS.get() > 1
        and draft_worker is not None
        and type(draft_worker).__name__ != "EAGLEWorkerV2"
    ):
        # L-e: only EAGLEWorkerV2 places its draft extend on the consumer
        # stream (and checks it); other speculative workers are unaudited.
        raise ValueError(
            "SGLANG_DSA_HOST_DEDUP_LAYER_PARTS > 1 supports no speculative worker "
            f"or EAGLEWorkerV2, not {type(draft_worker).__name__}"
        )

    # The bounds every load will use (uniform or LAYER_PART_BOUNDS): fail at
    # startup, not at the first ring load after warmup.
    layer_part_bounds(
        int(getattr(cache_controller, "layer_num", 0) or 0),
        envs.SGLANG_DSA_HOST_DEDUP_LAYER_PARTS.get(),
        envs.SGLANG_DSA_HOST_DEDUP_LAYER_PART_BOUNDS.get(),
    )
    loader = getattr(dedup, "loader", None)
    if loader is not None and forward_stream is not None:
        loader.check_overlap(forward_stream, group=getattr(dedup, "verify_group", None))


class GatedEvent:
    """A device event recorded by the loader thread; valid once its job is submitted."""

    __slots__ = ("event", "job")

    def __init__(self, event, job: LoadJob):
        self.event = event
        self.job = job

    def query(self) -> bool:
        return self.job.ready(False) and self.event.query()

    def synchronize(self) -> None:
        self.job.ready(True)
        self.event.synchronize()

    def wait(self, stream=None) -> None:
        self.job.ready(True)
        self.event.wait(stream)

    def elapsed_time(self, end) -> float:
        self.job.ready(True)
        if isinstance(end, GatedEvent):
            end.job.ready(True)
            end = end.event
        return self.event.elapsed_time(end)

    def record(self, stream=None):
        raise RuntimeError("GatedEvent is recorded by the shared loader thread only")


def gate_pending_for_batch(scheduler, batch) -> bool:
    """True if ``batch`` consumes a shared load whose job is not yet submitted.

    SGLANG_DEBUG_FORCE_GATE_REORDER_RANKS (debug positive control, default
    off): always True on the listed TP ranks."""
    if debug_forced_gate_reorder(scheduler):
        return True
    idx = getattr(batch, "hicache_consumer_index", -1)
    if idx is None or idx < 0:
        return False
    cc = getattr(getattr(scheduler, "tree_cache", None), "cache_controller", None)
    if cc is None:
        return False
    job = getattr(cc.layer_done_counter.events[idx], "gate_job", None)
    if job is None:
        return False
    if getattr(job, "parts", None):
        # Layer-wise: the forward can launch once part 0 is enqueued.
        return job.enqueued_parts < 1 and not job.submitted.is_set()
    return not job.submitted.is_set()


# ---------------------------------------------------------------------------
# SGLANG_DEBUG_FORCE_GATE_REORDER_RANKS: a positive control for the per-rank
# gate-pending reorder. The natural gate-pending reorder fires per rank,
# timing-bound and rare. On the listed TP ranks this knob takes its
# early-result path at every launch that has a previous result to process (the
# call sites check that). That covers every launch with a shared load or with
# 6h writes held, and with 6h off (nothing is ever held) every launch too. Only
# where each rank processes the previous result moves, as with the natural
# reorder. Result processing and the launch issue no TP collective
# (gate_pending_reorder.py, T4 audit), so the collective order is unchanged.
# Debug only: forced ranks give up the overlap of result processing with the
# forward.
# ---------------------------------------------------------------------------

FORCE_GATE_REORDER_TAG = "DEBUG_FORCE_GATE_REORDER"


def debug_force_gate_reorder_ranks() -> frozenset:
    """The TP ranks SGLANG_DEBUG_FORCE_GATE_REORDER_RANKS lists (empty: off)."""
    from decoupled_engine.srt.environ import envs

    raw = (envs.SGLANG_DEBUG_FORCE_GATE_REORDER_RANKS.get() or "").strip()
    if not raw:
        return frozenset()
    try:
        return frozenset(int(x) for x in raw.replace(" ", "").split(",") if x)
    except ValueError:
        raise ValueError(
            f"SGLANG_DEBUG_FORCE_GATE_REORDER_RANKS={raw!r}: expected a "
            "comma-separated list of TP ranks, e.g. 2,3"
        ) from None


def _scheduler_tp_rank(scheduler) -> int:
    rank = getattr(getattr(scheduler, "ps", None), "tp_rank", None)
    if rank is None:
        rank = getattr(scheduler, "tp_rank", 0)
    return int(rank or 0)


def debug_forced_gate_reorder(scheduler) -> bool:
    """True if this scheduler's TP rank is forced (resolved once)."""
    forced = getattr(scheduler, "_debug_force_gate_reorder", None)
    if forced is None:
        ranks = debug_force_gate_reorder_ranks()
        forced = bool(ranks) and _scheduler_tp_rank(scheduler) in ranks
        try:
            scheduler._debug_force_gate_reorder = forced
        except AttributeError:
            pass
    return forced


def log_debug_force_gate_reorder(scheduler, loop: str, active: bool = True) -> None:
    """At loop start, on every TP rank: a loud line when the knob is set."""
    ranks = debug_force_gate_reorder_ranks()
    if not ranks:
        return
    rank = _scheduler_tp_rank(scheduler)
    if not active:
        what = "no effect: this loop's gate-pending reorder is off"
    elif debug_forced_gate_reorder(scheduler):
        what = "FORCED: the previous result is processed before every launch"
    else:
        what = "not forced: natural gate-pending reorder only"
    logger.warning(
        "%s DEBUG POSITIVE CONTROL SGLANG_DEBUG_FORCE_GATE_REORDER_RANKS=%s "
        "loop=%s tp_rank=%d: %s",
        FORCE_GATE_REORDER_TAG,
        ",".join(str(r) for r in sorted(ranks)),
        loop,
        rank,
        what,
    )


def wait_consumed_loads(waits) -> None:
    """The consuming forward's wait, at launch on the current (forward)
    stream, for bootstrap-prefetch ops its batch adopted while they were
    still copying, next to the admission
    load's ``set_consumer`` gate.

    Each entry has ``wait()``: the bootstrap prefetcher's per-batch
    ``_ConsumeWait``, which waits on each adopted op's finish event (a
    ``GatedEvent`` with the shared / C2g loader: CPU gate on "submitted",
    then a stream wait on an event the loader has already recorded, I3; a
    loader error raises, I7; or a device event recorded at submission
    without host dedup). Never during CUDA graph capture (I5).
    """
    if not waits:
        return
    if _capturing():
        raise RuntimeError("bootstrap-prefetch load wait during CUDA graph capture")
    t0 = time.perf_counter()
    for ev in waits:
        ev.wait()
    from decoupled_engine.srt.mem_cache.ce_2d_load import COST

    COST.add("bwp_consume_gate", (time.perf_counter() - t0) * 1e3)


def _capturing() -> bool:
    """I5 guard: a (breakable) CUDA graph capture is active on this thread.
    BCG's capture context first, then the CUDA runtime's capture status of
    the current stream; BCG avoids ``torch.cuda.is_current_stream_capturing``
    on NVIDIA as unreliable. Without
    cuda-python, the torch query."""
    if not torch.cuda.is_available():
        return False
    try:
        from decoupled_engine.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            breakable_cuda_graph as bcg,
        )
    except Exception:  # noqa: BLE001 - fall back to the torch query
        return bool(torch.cuda.is_current_stream_capturing())
    if bcg._current_capture_var.get() is not None:
        return True
    if bcg.rt is None:
        return bool(torch.cuda.is_current_stream_capturing())
    return bool(bcg._is_stream_capturing(bcg.get_current_stream()))


def _blocking_event(dm):
    """An event whose synchronize sleeps instead of spinning (if supported)."""
    try:
        return dm.Event(blocking=True)
    except TypeError:
        return dm.Event()


class SharedLoader:
    """One daemon thread per rank; jobs run FIFO (the same order on every rank)."""

    def __init__(
        self,
        device,
        dedup,
        *,
        device_module,
        chunk_pages: int = 256,
        window: int = 2,
        start_thread: bool = True,
    ):
        self.device = device
        self.dedup = dedup
        self.dm = device_module
        from decoupled_engine.srt.environ import envs
        from decoupled_engine.srt.mem_cache.ce_2d_load import PacedSubmitter

        self.chunk_pages = max(1, int(chunk_pages))
        self.window = max(1, int(window))
        # Draft (height-1) copies: cudaMemcpyBatchAsync batches of this many
        # descriptors, or height-1 2D copies when 0 (shorter driver calls).
        draft_batch = int(envs.SGLANG_DSA_HOST_DEDUP_DRAFT_BATCH.get())
        self.batch_1d = draft_batch > 0
        # One native call per op; chunk = copies (one per page and layer run).
        self.paced = PacedSubmitter(
            chunk_ops=self.chunk_pages, window=self.window, chunk_1d=max(draft_batch, 1)
        )
        self.gate_timeout_s = float(envs.SGLANG_DSA_HOST_DEDUP_PEER_TIMEOUT_S.get())
        # Load timing: GPU time this stream sat waiting for
        # the op's start event and the load fence, from event pairs resolved
        # lazily on a later op (never synchronized). Logging only.
        self.load_timing = bool(envs.SGLANG_DSA_HOST_DEDUP_LOAD_TIMING.get())
        # Debug only: > 0 = H2D rate cap in GB/s on non-follower
        # ranks plus destination poison before every foreground part.
        self.race_amplifier = float(envs.SGLANG_DSA_HOST_DEDUP_RACE_AMPLIFIER.get())
        self._timing_pending: List[Any] = []
        self.peer_rate_gbps = float(envs.SGLANG_DSA_HOST_DEDUP_PEER_RATE_GBPS.get())
        self.bwp_rate_gbps = float(envs.SGLANG_DSA_HOST_DEDUP_BWP_PEER_RATE_GBPS.get())
        # Live cap of the running background job (bytes/s; 0 = lifted). Each
        # counter has one writer: submitted (scheduler), done (loader thread).
        self._bg_rate = ctypes.c_uint64(0)
        self._fg_submitted = 0
        self._fg_done = 0
        # The background job whose copies are being issued (for lifts).
        self._bg_running: Optional[LoadJob] = None
        # Background jobs submitted and not finished, FIFO (scheduler thread
        # appends, loader thread removes): a lift covers the jobs ahead.
        self._bg_lock = threading.Lock()
        self._bg_pending: List[LoadJob] = []
        # Submission order: every submitted job runs, in order (lockstep op
        # sequence across ranks).
        self._n_submitted = 0
        self._n_run = 0
        # Adaptive cap: 1 while a forward is in flight on this rank (written
        # by the scheduler thread once per pass; read by the native pacer).
        self.adaptive_cap = bool(envs.SGLANG_DSA_HOST_DEDUP_BWP_ADAPTIVE_CAP.get())
        self.busy_word = ctypes.c_uint64(0)
        # C2g op sequence, the same on every rank (lockstep FIFO); starts at 1
        # because an all-zero progress page must not read as "op 0 complete".
        self._seq = 1
        self._peer_attr = None
        self._hybrid_logged = False
        self.stream = (
            device_module.Stream(device=device)
            if device is not None
            else device_module.Stream()
        )
        self.side_stream = (
            device_module.Stream(device=device)
            if device is not None
            else device_module.Stream()
        )
        self.q: queue.SimpleQueue[LoadJob] = queue.SimpleQueue()
        self.error: Optional[BaseException] = None
        self._chunk_events: List[Any] = []
        # Q3: (event before the dirty wait, ack_start) pairs; their GPU gap
        # is the wait, logged once both are done (loader thread, not a decision).
        self._q3_waits: List[tuple] = []
        self.thread = None
        if start_thread:
            self.thread = threading.Thread(
                target=self._loop, name="sglang-shared-loader", daemon=True
            )
            self.thread.start()

    def _new_stream(self):
        dm = self.dm
        return dm.Stream(device=self.device) if self.device is not None else dm.Stream()

    def check_overlap(self, forward_stream, group=None, *, self_test=None) -> bool:
        """I9: the loader streams must neither alias ``forward_stream`` nor
        share an in-order hardware queue with it, or loads never overlap the
        forward. With SGLANG_DSA_HOST_DEDUP_LOAD_TIMING: check and log.
        With SGLANG_DSA_HOST_DEDUP_STREAM_REDRAW: redraw both streams while
        the check fails (<= 64 tries, as the scheduler's schedule_stream) and
        raise if it still fails. Startup only (no load is queued yet).
        Returns the TP-wide result (MIN over ``group``); both flags off: True,
        nothing runs."""
        from decoupled_engine.srt.environ import envs

        timing = bool(envs.SGLANG_DSA_HOST_DEDUP_LOAD_TIMING.get())
        redraw = bool(envs.SGLANG_DSA_HOST_DEDUP_STREAM_REDRAW.get())
        if not (timing or redraw):
            return True
        test = self_test if self_test is not None else overlap_self_test
        fwd = getattr(forward_stream, "cuda_stream", None)

        def check():
            """(ok, why, retryable): a redraw can only fix an alias or a
            failed overlap, not a self-test error (e.g. allocation)."""
            streams = (self.stream, self.side_stream)
            if any(getattr(st, "cuda_stream", None) == fwd for st in streams):
                return False, "a loader stream aliases forward_stream", True
            try:
                ok, why = test(self.dm, forward_stream, streams)
            except Exception as e:  # noqa: BLE001 - logged and reported, never retried
                return False, f"self-test error: {e!r}", False
            return ok, why, True

        ok, why, retryable = check()
        tries = 0
        while not ok and retryable and redraw and tries < 64:
            self.stream, self.side_stream = self._new_stream(), self._new_stream()
            tries += 1
            ok, why, retryable = check()
        all_ok = ok
        if group is not None:
            import torch

            t = torch.tensor([int(ok)], dtype=torch.int64)
            torch.distributed.all_reduce(
                t, op=torch.distributed.ReduceOp.MIN, group=group
            )
            all_ok = bool(int(t.item()))
        log = logger.info if ok else logger.warning
        log(
            "Shared host-dedup loader I9 overlap check: rank %s, all ranks %s, "
            "%d redraws (%s)",
            "PASS" if ok else "FAIL",
            "PASS" if all_ok else "FAIL",
            tries,
            why,
        )
        if redraw and not all_ok:
            raise RuntimeError(
                "Shared host-dedup loader: copy streams cannot overlap forward_stream "
                f"on some rank after redraws (this rank: {why})"
            )
        return all_ok

    def submit(self, job: LoadJob) -> None:
        if self.error is not None:
            raise RuntimeError(
                "Shared host-dedup loader failed earlier"
            ) from self.error
        if not job.background:
            # A forward will wait for this job and it queues behind any
            # running background job: lift that job's cap now.
            self._fg_submitted += 1
            self._bg_rate.value = 0
        else:
            with self._bg_lock:
                self._bg_pending.append(job)
        # Never dropped or deferred here, whatever the loader thread's
        # progress: every rank must run the same job sequence. Callers
        # bound background jobs from all-reduced state (bootstrap prefetch).
        job.index = self._n_submitted
        self._n_submitted += 1
        self.q.put(job)

    def lift_background_cap(self, job: LoadJob) -> None:
        """Run background ``job`` uncapped: a request waits for its rows
        (admission deferred it on, or adopted, the prefetch op). Before the
        job starts, it starts uncapped; while it runs, the live cap drops to
        0; after it ended, nothing changes (the next job is unaffected).
        Background jobs ahead of it in the FIFO are lifted too: the request
        waits for them as well."""
        job.lifted = True
        with self._bg_lock:
            if job in self._bg_pending:
                for ahead in self._bg_pending:
                    ahead.lifted = True
                    if ahead is job:
                        break
        running = self._bg_running
        if running is not None and running.lifted:
            self._bg_rate.value = 0

    def set_forward_busy(self, busy: bool) -> None:
        """Adaptive cap: whether a forward is in flight on this rank."""
        self.busy_word.value = 1 if busy else 0

    def _background_rate(self, job: LoadJob) -> ctypes.c_uint64:
        """Arm the live cap for a background job (lifted if a foreground job
        is already queued). With the GIL, either this check sees submit's
        increment or submit's lift lands after this store."""
        self._bg_rate.value = int(self.bwp_rate_gbps * 1e9)
        if self._fg_submitted > self._fg_done or job.lifted:
            self._bg_rate.value = 0
        return self._bg_rate

    def _timing_events(self, n: int):
        return [self.dm.Event(enable_timing=True) for _ in range(n)]

    def resolve_timing(self) -> int:
        """Add the GPU wait times of every op whose timing events completed
        (query only, no sync); returns how many ops are still pending."""
        from decoupled_engine.srt.mem_cache.ce_2d_load import COST

        keep = []
        for item in self._timing_pending:
            evs, bg = item[0], item[1]
            if not evs[-1].query():
                keep.append(item)
                continue
            e0, e1, e2, e3 = evs
            COST.add(f"{bg}loader_start_wait_gpu_ms", e0.elapsed_time(e1))
            COST.add(f"{bg}loader_fence_wait_gpu_ms", e1.elapsed_time(e2))
            # Copies (on a follower also its CPU gate waits for the source).
            COST.add(f"{bg}loader_copy_span_gpu_ms", e2.elapsed_time(e3))
            if len(item) > 2:
                # Layer-wise: time from the fence to part p ready, per rank.
                for p, ev in enumerate(item[2]):
                    COST.add(f"{bg}part{p}_ready_gpu_ms", e2.elapsed_time(ev))
        self._timing_pending = keep
        return len(keep)

    def _loop(self) -> None:
        if self.device is not None and hasattr(self.dm, "set_device"):
            self.dm.set_device(self.device)
        while True:
            job = self._next_job()
            self.run_job(job)

    def _next_job(self, idle_s: float = 1.0) -> LoadJob:
        """The next queued job. While Q3 dirty waits are pending, an idle
        loader wakes every ``idle_s`` and logs those already done, so the last
        waits before an idle period are not held until the next job. Nothing
        pending (Q3 off, or only clean ops): the blocking get as before."""
        while True:
            if not self._q3_waits:
                return self.q.get()
            try:
                return self.q.get(timeout=idle_s)
            except queue.Empty:
                self._drain_q3_waits()

    def run_job(self, job: LoadJob) -> None:
        job.started = True
        try:
            if job.index is not None:
                # FIFO without gaps: a dropped or reordered job would desync
                # the ranks' op sequences; fail loudly instead.
                if job.index != self._n_run:
                    raise RuntimeError(
                        f"Shared host-dedup loader: job {job.index} ran, "
                        f"expected job {self._n_run} (dropped or reordered)"
                    )
                self._n_run += 1
            self._run(job)
        except BaseException as e:  # noqa: BLE001 - surfaced through the gate
            logger.exception("Shared host-dedup loader job failed")
            job.error = e
            self.error = e
            peer = getattr(self.dedup, "peer", None)
            if peer is not None and getattr(peer, "pub_addr", 0):
                peer.set_error()  # ranks pulling from this one fail too
        finally:
            if not job.background and self._fg_done < self._fg_submitted:
                self._fg_done += 1
            if self._bg_running is job:
                self._bg_running = None
            if job.background:
                with self._bg_lock:
                    if job in self._bg_pending:
                        self._bg_pending.remove(job)
            job.submitted.set()

    # ---- the job ------------------------------------------------------

    def _drain_q3_waits(self) -> None:
        from decoupled_engine.srt.mem_cache.ce_2d_load import COST

        while self._q3_waits:
            pre, ack, bg = self._q3_waits[0]
            try:
                if not ack.query():
                    return
                ms = pre.elapsed_time(ack)
            except Exception:  # noqa: BLE001 - timing only
                ms = None
            self._q3_waits.pop(0)
            if ms is not None:
                COST.add(f"{bg}q3_dirty_wait_gpu", ms)

    def _host_side_indices(self, job: LoadJob):
        """Device indices to CPU on the side stream (after the producer's start)."""
        dm = self.dm
        moved = {}
        out = []
        need = [t for t in job.transfers if getattr(t.device_indices, "is_cuda", False)]
        if need:
            with dm.stream(self.side_stream):
                job.start_event.wait(self.side_stream)
                for t in need:
                    key = id(t.device_indices)
                    if key not in moved:
                        moved[key] = t.device_indices.to("cpu", non_blocking=True)
                ev = _blocking_event(dm)
                ev.record(self.side_stream)
            t0 = time.perf_counter()
            ev.synchronize()
            from decoupled_engine.srt.mem_cache.ce_2d_load import COST

            COST.add("loader_index_d2h", (time.perf_counter() - t0) * 1e3)
        for t in job.transfers:
            if getattr(t.device_indices, "is_cuda", False):
                t = t._replace(device_indices=moved[id(t.device_indices)])
            out.append(t)
        return out

    def _run(self, job: LoadJob) -> None:
        from decoupled_engine.srt.mem_cache import ce_2d_load
        from decoupled_engine.srt.mem_cache.ce_2d_load import COST

        dm = self.dm
        t_start = time.perf_counter()
        cpu0 = time.thread_time()
        bg = "bwp_" if job.background else ""
        COST.add(f"{bg}loader_queue_delay", (t_start - job.t_enqueue) * 1e3)
        self._drain_q3_waits()
        if job.q3 and any(
            getattr(t.device_indices, "is_cuda", False) for t in job.transfers
        ):
            # L-g / Q4: a Q3 op has every destination id on the CPU.
            raise RuntimeError("M4 Q3 load with device-side destination indices")
        transfers = self._host_side_indices(job)
        dedup = self.dedup
        ce_peer = bool(getattr(dedup, "ce_peer", False))
        follower = ce_peer and not dedup.is_src
        if dedup is not None:
            dedup.check_host_indices(
                transfers,
                group=getattr(dedup, "verify_group", None),
                field="device_indices" if ce_peer else "host_indices",
            )
        # Layer-wise op: one C2g seq per part on every rank, so the
        # per-chunk follower gate stays valid within each part; a whole op
        # (parts None) takes one seq, as before.
        parts = job.parts
        # PART_PLAN=hybrid (layer-wise only): hybrid copies on the source,
        # run-coalesced copies on followers, which then gate whole parts.
        hybrid = bool(parts) and getattr(job, "part_plan", "page") == "hybrid"
        kw = {}
        seqs = None
        if ce_peer:
            # The whole block is reserved at op start on every rank,
            # whatever path the op then takes (2D plans or per-layer fallback).
            n_seq = len(parts) if parts else 1
            seqs = list(range(self._seq, self._seq + n_seq))
            kw = dict(seq=seqs[0], timeout_s=self.gate_timeout_s)
            self._seq += n_seq
            job.seq0 = seqs[0]
        capped = False
        if follower:
            plans = job.peer_plans(transfers)
            if plans is None:
                if parts and dedup is not None and hasattr(dedup, "check_part_digest"):
                    # The other ranks are about to enter the part-digest
                    # all-reduce: make them fail it (a digest nobody else has)
                    # instead of waiting there for this rank.
                    try:
                        dedup.check_part_digest(
                            [_NO_PLAN_DIGEST] * len(parts),
                            len(parts),
                            group=getattr(dedup, "verify_group", None),
                        )
                    except Exception:  # noqa: BLE001 - this rank raises below
                        pass
                raise RuntimeError("C2g follower: no peer-copy plan for this op")
            if self._peer_attr is None:
                self._peer_attr = self.paced.peer_attr(dedup.peer.resolve_src_device())
            kw.update(
                gate=dedup.peer.gate_addr,
                pub=dedup.peer.pub_addr,  # chain / tree: this rank feeds others
                kind=ce_2d_load.CUDA_MEMCPY_DEFAULT,
                attr=self._peer_attr,
            )
            if hybrid:
                # The source's chunks hold other rows (another shape): wait
                # for its whole part, even when the chunk counts happen to match.
                kw["gate_whole"] = True
            if job.background and self.bwp_rate_gbps > 0:
                self._bg_running = job
                kw["rate_word"] = self._background_rate(job)
                capped = bool(self._bg_rate.value)
                if self.adaptive_cap:
                    kw["busy_word"] = self.busy_word
                    kw["deadline_word"] = job.deadline_ns
            else:
                kw["rate_gbps"] = self.peer_rate_gbps
                capped = self.peer_rate_gbps > 0
        else:
            plans = job.per_layer_fallback(transfers, plan_only=True)
            if ce_peer:
                kw.update(pub=dedup.peer.pub_addr)
            if self.race_amplifier > 0 and not job.background:
                # Debug: slow this rank's H2D to widen any race.
                kw["rate_gbps"] = self.race_amplifier
        # Layer-wise: plans is one list per part; whole op: one list. Every
        # part's copy arrays are built before the first submit, so TP0's copy
        # engine does not idle on Python at part boundaries.
        part_lists = arrs_list = None
        if plans is not None:
            part_lists = plans if parts else [plans]
            if hybrid:
                build = (
                    ce_2d_load.coalesced_arrays
                    if follower
                    else ce_2d_load.hybrid_arrays
                )
                arrs_list = [build(pl) for pl in part_lists]
                if not getattr(self, "_hybrid_logged", False):
                    self._hybrid_logged = True
                    logger.info(
                        "Shared host-dedup part plan: hybrid (%s copies%s)",
                        "run-coalesced" if follower else "hybrid page/layer-major",
                        ", whole-part gate" if follower else "",
                    )
            elif parts:  # whole ops build theirs after ack_start
                arrs_list = [
                    ce_2d_load.op_arrays(pl, batch_1d=self.batch_1d)
                    for pl in part_lists
                ]
        if parts and dedup is not None and hasattr(dedup, "check_part_digest"):
            # Part digest check (with SHARED_VERIFY): same parts, same copy
            # composition on every rank (hybrid: same rows and bytes); a rank
            # on the per-layer fallback matches anything.
            dedup.check_part_digest(
                (
                    None
                    if plans is None
                    else part_digest(part_lists, arrs_list, shape_free=hybrid)
                ),
                len(parts),
                group=getattr(dedup, "verify_group", None),
            )
        s = self.stream
        tev = ptev = None
        if self.load_timing:
            self.resolve_timing()
            tev = self._timing_events(4)
            if parts:
                ptev = self._timing_events(len(parts))  # time to part ready
        with dm.stream(s):
            if tev is not None:
                tev[0].record(s)
            pre_wait = None
            if job.q3:
                # Clean pages: nothing to wait for unless the load took dirty
                # ones (forward_done of their release epoch). Load timing: no
                # start wait; the dirty wait is booked as the fence wait.
                if tev is not None:
                    tev[1].record(s)
                if job.dirty_event is not None:
                    pre_wait = dm.Event(enable_timing=True)
                    pre_wait.record(s)
                    job.dirty_event.wait(s)
                    COST.add(f"{bg}q3_dirty_wait_ops", 1.0)
                else:
                    COST.add(f"{bg}q3_clean_ops", 1.0)
            else:
                job.start_event.wait(s)
                if tev is not None:
                    tev[1].record(s)
                if job.fence_event is not None:
                    job.fence_event.wait(s)
            if tev is not None:
                tev[2].record(s)
            job.ack_start.record(s)
            if pre_wait is not None:
                self._q3_waits.append((pre_wait, job.ack_start, bg))
            if plans is not None:
                calls = chunks = 0
                waited_ms = gate_ms = rate_ms = 0.0
                handle = getattr(s, "cuda_stream", 0)
                if arrs_list is None:
                    arrs_list = [
                        ce_2d_load.op_arrays(part_lists[0], batch_1d=self.batch_1d)
                    ]
                tails = None
                if "deadline_word" in kw and len(arrs_list) > 1:
                    # Adaptive cap on a layer-wise job: each part's
                    # deadline check also counts the parts still to come.
                    sizes = [ce_2d_load.arrays_bytes(a) for a in arrs_list]
                    tails = [sum(sizes[q + 1 :]) for q in range(len(sizes))]
                for p, arrs in enumerate(arrs_list):
                    if seqs is not None:
                        kw["seq"] = seqs[p]
                    if tails is not None:
                        kw["tail_bytes"] = tails[p]
                    if self.race_amplifier > 0 and not job.background:
                        ce_2d_load.poison(arrs, handle)
                    c, w, k = self.paced.submit(arrs, handle, **kw)
                    calls += c
                    waited_ms += w
                    chunks += k
                    gate_ms += self.paced.last_gate_ms
                    rate_ms += self.paced.last_rate_ms
                    if parts:
                        # Recorded after BOTH families (and the draft in the
                        # last part); then published to the consumer (I3).
                        job.part_events[p].record(s)
                        if ptev is not None:
                            ptev[p].record(s)
                        job.enqueued_parts = p + 1
                COST.add("calls_2d", float(calls))
                COST.add("chunks", float(chunks))
                COST.add("chunk_wait", waited_ms)
                if follower:
                    COST.add("peer_gate_wait", gate_ms)
                    if capped:
                        COST.add(f"{bg}peer_rate_sleep", rate_ms)
                        if job.background and not self._bg_rate.value:
                            COST.add("bwp_cap_lifted", 1.0)  # an admission load queued
                        if "busy_word" in kw:
                            COST.add(
                                "bwp_cap_flips",
                                float(getattr(self.paced, "last_flips", 0)),
                            )
                run_len, long_frac = ce_2d_load.joint_runs(part_lists[0])
                COST.add("joint_run_pages", run_len)
                COST.add("long_run_pct", 100.0 * long_frac)
            else:
                if ce_peer:
                    words = ce_2d_load.progress_page(kw["pub"])
                    words[1] = (seqs[0] << ce_2d_load.SEQ_SHIFT) | 1
                job.per_layer_fallback(transfers, plan_only=False)
                if ce_peer:
                    done = _blocking_event(dm)
                    done.record(s)
                    done.synchronize()
                    # The whole seq block completes at once, and only after
                    # the copies: pub[0] first, then pub[1]. A later seq of the
                    # block is never published early (the follower's
                    # "pub[1] seq > mine => complete" shortcut, ce_submit.c).
                    last = (seqs[-1] << ce_2d_load.SEQ_SHIFT) | 1
                    words[0] = last
                    if len(seqs) > 1:
                        words[1] = last
                if parts:
                    for p, ev in enumerate(job.part_events):
                        ev.record(s)
                        if ptev is not None:
                            ptev[p].record(s)
                    job.enqueued_parts = len(parts)
            if tev is not None:
                tev[3].record(s)
                self._timing_pending.append(
                    (tev, bg) if ptev is None else (tev, bg, ptev)
                )
            if job.on_layer_done is not None and not parts:
                for layer_id in range(job.layer_num):
                    job.on_layer_done(layer_id)
            job.ack_finish.record(s)
            for t in transfers:
                for x in (t.host_indices, t.device_indices):
                    if getattr(x, "is_cuda", False):
                        x.record_stream(s)
            for t in job.transfers:
                if getattr(t.device_indices, "is_cuda", False):
                    t.device_indices.record_stream(s)
        COST.add(f"{bg}loader_submit", (time.perf_counter() - t_start) * 1e3)
        COST.add("loader_cpu", (time.thread_time() - cpu0) * 1e3)
