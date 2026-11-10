"""Opt-in push of the prefill routing state at each pass launch and result.

Off unless SGLANG_PREFILL_STATE_PUSH_PORT (or _ENDPOINT) is set. When it is
set, the push runs on the prefill attention TP0/CP0/PP0 rank, next to the
routing tracker (SGLANG_PREFILL_ROUTING_SNAPSHOT=1), and binds one ZMQ PUB
socket. A router subscribes to it instead of polling /v1/loads.

When a message goes out:

- At a launch in the overlap loop (launch_now). With the launch thread this
  is on the scheduler thread while the worker thread launches the pass, at
  the start of the previous result's processing; without it, just before the
  launch call in run_batch (that call can block for most of the previous
  forward). The snapshot shows the new pass in flight.
  SGLANG_PREFILL_STATE_PUSH_LAUNCH_MODE selects the launches that go out
  there:
  - "busy" (default): only when the previous pass still runs on the GPU. The
    scheduler thread waits for that pass next, so the message is built while
    it would wait anyway. When the previous pass has ended, the GPU waits for
    this launch or a result is ready: the launch then goes out at the end of
    the iteration.
  - "always": every launch. With the launch thread it goes out while the
    worker launches, also after an ended pass; for the first pass after an
    idle GPU the loop also launches on the worker, and the scheduler thread
    only sends the message meanwhile. Before an inline launch (no launch
    thread) on an idle GPU the launch is held and goes out right after the
    launch call returns (launch_after), never in front of that launch.
- At the end of each loop iteration (flush): the results processed in it and
  launches not sent yet, else a heartbeat when nothing went out for
  HEARTBEAT_S. When the /v1/loads snapshot was sampled in the same iteration,
  after these events, the message reuses it instead of building the state
  again.
- The non-overlap loop sends after run_batch returns and at the end of each
  iteration; the second message adds to the GPU idle time between passes.

- GPU times come from non-blocking queries of the CUDA events that the sched
  tracer already records (SGLANG_SCHED_TRACE_GPU_TIME=1). There is no GPU
  synchronization. An event that has not completed gives null.
- The send never blocks. The PUB socket drops a message for a subscriber
  whose queue is at the HWM, for that subscriber only; the subscriber sees
  the gap in seq. Other subscribers are not affected.
- The work is bounded. The snapshot keeps the tracker's record and byte
  limits, and the message has its own byte cap.
- The /v1/loads snapshot, its sample throttle and its sequence do not change.

Symmetric workers (SGLANG_SYMMETRIC_ROUTING_SNAPSHOT=1 with PORT/ENDPOINT):
the pusher runs with ``extend_only``. A symmetric worker also decodes; its
launch and result hooks act on extend passes (EXTEND, MIXED) only, so decode
passes produce no events (they still show in ``inflight`` while queued).
The symmetric overlap loop sends the launch message at the start of the
launch thread's work (SGLANG_SYMMETRIC_LAUNCH_THREAD), notes launches after
its launch block and flushes after its routing sample; the result stamp is
taken right after the copy sync in process_batch_result_prefill.
"""

from __future__ import annotations

import atexit
import logging
import math
import time
from collections import deque
from itertools import islice

import msgspec
import zmq
from msgspec.structs import asdict

from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.managers.scheduler_components.prefill_routing_snapshot import (
    clock_domain,
)
from decoupled_engine.srt.utils.network import is_zmq_endpoint_ipv6

logger = logging.getLogger(__name__)

TOPIC = b"prefill_state"
SCHEMA = "sglang.prefill_state_push"
SCHEMA_VERSION = 1
DEFAULT_HWM = 16
DEFAULT_MAX_BYTES = 1 << 20
MIN_MAX_BYTES = 64 << 10
MAX_MAX_BYTES = 16 << 20
# Launch and result notes waiting for the next message. A flush follows every
# loop pass, so more than a few means the loop is not flushing; the oldest are
# dropped.
MAX_PENDING_EVENTS = 64
MAX_INFLIGHT_PASSES = 8
STATS_LOG_INTERVAL_S = 60.0
# SGLANG_PREFILL_STATE_PUSH_LAUNCH_MODE values.
LAUNCH_MODES = ("busy", "always")

_encoder = msgspec.msgpack.Encoder()


class PassEvent(msgspec.Struct, gc=False):
    """One prefill pass at one observation point (a named msgpack map)."""

    # "launch": the pass was launched. "result": its result was processed.
    # "inflight": launched, result not processed, at this message.
    kind: str
    forward_iter: int | None = None
    # Engine monotonic time when run_batch stamped the pass (launch_ts).
    launch_mono_s: float | None = None
    # Engine monotonic time of the observation. Launch: when the push noted
    # it, during the launch (launch_now) or after the loop's launch block.
    # Result: just after the first wait for the pass's output (its copy
    # sync). Inflight: null.
    observed_mono_s: float | None = None
    # Non-blocking CUDA event queries; null when unknown.
    gpu_started: bool | None = None
    gpu_ended: bool | None = None
    # Forward-stream interval of the pass (the iter_trace forward_gpu_ms).
    gpu_ms: float | None = None
    # Previous pass's end event to this pass's start event.
    gpu_gap_before_ms: float | None = None
    # The events mapped to engine monotonic time (GpuClock); null without a
    # clock sample of the last MAX_SAMPLE_AGE_S.
    gpu_start_mono_s: float | None = None
    gpu_end_mono_s: float | None = None


class PrefillStatePush(msgspec.Struct, gc=False):
    """The message payload (a named msgpack map)."""

    schema: str
    schema_version: int
    # Changes when the engine process restarts; seq restarts at 1.
    epoch: str
    # +1 per message the engine tried to send. A gap is a drop.
    seq: int
    # Socket rank (port offset): attention-DP rank, or DP replica index.
    rank: int
    # "launch" (a launch event is included), "result" or "heartbeat".
    trigger: str
    sent_mono_s: float
    sent_unix_s: float
    # Add to an engine monotonic stamp to get the engine wall clock.
    mono_to_unix_s: float
    clock_domain_id: str | None
    # Launches and results since the previous message, in observation order.
    events: list[PassEvent]
    # Launched passes whose results are not processed, oldest first.
    inflight: list[PassEvent]
    # "cuda_events" when the sched tracer records GPU times, else
    # "unavailable" (all GPU fields null).
    gpu_timing: str
    gpu_clock_samples: int
    # Send calls that the socket refused at once (zmq.Again). The PUB socket
    # drops per subscriber without telling the engine; see seq gaps.
    dropped_total: int
    errors_total: int
    oversize_total: int
    build_us: float
    convert_us: float
    prev_encode_us: float | None
    prev_send_us: float | None
    prev_bytes: int | None
    # The whole previous push call (launch_now or flush) that sent a message.
    prev_flush_us: float | None
    # "push_build" (built for this message) or "v1_loads_capture" (the
    # /v1/loads sample of the same loop iteration); null without a snapshot.
    snapshot_source: str | None
    # The /v1/loads prefill_routing object (named fields); null with a
    # reason in snapshot_omitted.
    snapshot: dict | None
    snapshot_omitted: str | None = None


def _request_dict(row):
    out = asdict(row)
    if row.plan is not None:
        out["plan"] = asdict(row.plan)
    if row.state is not None:
        out["state"] = asdict(row.state)
    return out


def _batch_dict(batch):
    out = asdict(batch)
    out["features"] = asdict(batch.features)
    return out


def named_snapshot(value):
    """public_dict(value) of a PrefillRoutingSnapshot, without generic recursion.

    It has the same keys and nesting as the /v1/loads object; the unit test
    test_named_snapshot_equals_public_dict pins that. Lists and dicts of
    builtins are shared with the snapshot, which is not changed later.
    """
    out = asdict(value)
    out["policy"] = asdict(value.policy)
    out["prev"] = asdict(value.prev)
    out["waiting"] = [_request_dict(r) for r in value.waiting]
    out["bootstrap"] = [_request_dict(r) for r in value.bootstrap]
    out["inflight_requests"] = [_request_dict(r) for r in value.inflight_requests]
    if value.partial is not None:
        out["partial"] = _request_dict(value.partial)
    out["inflight_batches"] = [_batch_dict(b) for b in value.inflight_batches]
    if value.current_batch is not None:
        out["current_batch"] = _batch_dict(value.current_batch)
    return out


def _int(value):
    return value if type(value) is int and value >= 0 else None


def _time(value):
    if type(value) in (int, float) and math.isfinite(value) and value > 0:
        return float(value)
    return None


def extend_pass(batch):
    """A prefill pass: forward mode EXTEND or MIXED (not decode, idle, verify
    or prebuilt). Unknown modes are not."""
    mode = getattr(batch, "forward_mode", None)
    is_extend = getattr(mode, "is_extend", None)
    if not callable(is_extend) or not is_extend():
        return False
    is_verify = getattr(mode, "is_target_verify", None)
    return not (callable(is_verify) and is_verify())


def _launch_key(batch):
    """The pass of a launch note: forward_iter, else the batch object id."""
    iteration = _int(getattr(batch, "forward_iter", None))
    return iteration if iteration is not None else ("id", id(batch))


def newest_pass_running(queue):
    """Whether the newest pass of the overlap result queue has not ended.

    A non-blocking query of its result's copy_done event, which the loop
    records after the forward and the result D2H. False when the queue is
    empty or the event is missing.
    """
    if not isinstance(queue, (deque, list, tuple)) or not queue:
        return False
    pair = queue[-1]
    result = pair[1] if isinstance(pair, (tuple, list)) and len(pair) > 1 else None
    done = getattr(result, "copy_done", None)
    return done is not None and not done.query()


class GpuClock:
    """Map completed CUDA events to engine monotonic time without a sync.

    A sample pairs a host time with an event that had completed before that
    time (the caller queried the event first). The host time minus the event's
    time after the anchor event is then an upper bound of the anchor's host
    time. The estimate is the minimum over the samples of the last
    MAX_SAMPLE_AGE_S (at most WINDOW). It is tight when the scheduler waited
    in the result copy sync, which is the usual case while the GPU is busy.
    expire() drops older samples before each message, so drift between the
    GPU and host clocks stays out of the minimum; after an idle gap longer
    than MAX_SAMPLE_AGE_S the mapping is null until the next sample. After
    REANCHOR_S the anchor moves to the newest event, so cudaEventElapsedTime's
    float32 milliseconds stay precise.
    """

    WINDOW = 64
    MAX_SAMPLE_AGE_S = 30.0
    REANCHOR_S = 20.0

    def __init__(self):
        self.anchor = None
        # (host time of the sample, upper bound of the anchor's host time)
        self.samples = deque(maxlen=self.WINDOW)
        # Minimum bound of the samples (computed when they change), or None.
        self.anchor_host = None

    def expire(self, now):
        """Drop the samples older than MAX_SAMPLE_AGE_S at engine time now."""
        if not self.samples or now - self.samples[0][0] <= self.MAX_SAMPLE_AGE_S:
            return
        while self.samples and now - self.samples[0][0] > self.MAX_SAMPLE_AGE_S:
            self.samples.popleft()
        self._update()

    def observe(self, event, observed_mono):
        """event completed before observed_mono (engine monotonic)."""
        self.expire(observed_mono)
        if self.anchor is None or not self.samples:
            # No sample to carry over: the new event is the anchor.
            self.anchor, rel = event, 0.0
        else:
            rel = self.anchor.elapsed_time(event) / 1e3
            if not math.isfinite(rel) or rel < 0:
                return
        self.samples.append((observed_mono, observed_mono - rel))
        if rel > self.REANCHOR_S:
            # The same instants, expressed relative to the newer event.
            self.samples = deque(
                ((t, bound + rel) for t, bound in self.samples), maxlen=self.WINDOW
            )
            self.anchor = event
        self._update()

    def _update(self):
        self.anchor_host = min((bound for _, bound in self.samples), default=None)

    def to_mono(self, event):
        """Engine monotonic time of a completed event, or None."""
        if self.anchor is None or self.anchor_host is None:
            return None
        return self.anchor_host + self.anchor.elapsed_time(event) / 1e3


def resolve_endpoint(port, endpoint, rank):
    """The bind address: ENDPOINT as given (one socket), else tcp://*:PORT+rank."""
    if endpoint:
        if rank != 0:
            raise ValueError(
                "SGLANG_PREFILL_STATE_PUSH_ENDPOINT names one socket, but this "
                f"engine has routing rank {rank}; use SGLANG_PREFILL_STATE_PUSH_PORT "
                "(each rank binds PORT + rank)"
            )
        return endpoint
    if not 0 < port + rank <= 65535:
        raise ValueError(
            f"SGLANG_PREFILL_STATE_PUSH_PORT={port} + rank {rank} is not a TCP port"
        )
    return f"tcp://*:{port + rank}"


def open_push_socket(endpoint, hwm):
    """Bind the PUB socket. Module-level so that tests can stub the bind.

    A plain (lossy) PUB socket: when the queue of one subscriber is at the
    HWM, that subscriber loses the message and the others still get it. A
    send never blocks or fails for a full queue. (XPUB_NODROP would instead
    refuse the message for every subscriber while any one queue is full, so a
    stalled or half-open subscriber would cut off the router.)
    """
    sock = zmq.Context.instance().socket(zmq.PUB)
    try:
        sock.setsockopt(zmq.SNDHWM, hwm)
        sock.setsockopt(zmq.LINGER, 0)
        if is_zmq_endpoint_ipv6(endpoint):
            sock.setsockopt(zmq.IPV6, 1)
        sock.bind(endpoint)
    except Exception:
        sock.close()
        raise
    return sock


def create_prefill_state_pusher(tracker, *, rank=0, pp_size=1, extend_only=False):
    """The pusher for a routing rank, or None when the push is off.

    Explicit opt-in: a bad configuration or a failed bind stops the startup
    instead of running without the push.
    """
    port = envs.SGLANG_PREFILL_STATE_PUSH_PORT.get() or 0
    endpoint = envs.SGLANG_PREFILL_STATE_PUSH_ENDPOINT.get() or ""
    if not port and not endpoint:
        return None
    if pp_size > 1:
        raise ValueError(
            "SGLANG_PREFILL_STATE_PUSH_PORT/_ENDPOINT is not supported with "
            f"pipeline parallelism (pp_size {pp_size}): the PD prefill PP loop "
            "has no push points, so the push would never send"
        )
    if tracker is None:
        raise ValueError(
            "SGLANG_PREFILL_STATE_PUSH_PORT/_ENDPOINT needs "
            "SGLANG_PREFILL_ROUTING_SNAPSHOT=1: the push sends that snapshot"
        )
    endpoint = resolve_endpoint(port, endpoint, rank)
    launch_mode = parse_launch_mode(envs.SGLANG_PREFILL_STATE_PUSH_LAUNCH_MODE.get())
    hwm = max(1, min(1024, envs.SGLANG_PREFILL_STATE_PUSH_HWM.get()))
    try:
        sock = open_push_socket(endpoint, hwm)
    except Exception as error:
        raise RuntimeError(
            f"prefill state push: cannot bind {endpoint}: {error}"
        ) from error
    pusher = PrefillStatePusher(
        tracker,
        socket=sock,
        endpoint=endpoint,
        rank=rank,
        max_bytes=envs.SGLANG_PREFILL_STATE_PUSH_MAX_BYTES.get(),
        heartbeat_s=envs.SGLANG_PREFILL_STATE_PUSH_HEARTBEAT_S.get(),
        launch_mode=launch_mode,
        extend_only=extend_only,
    )
    atexit.register(pusher.close)
    logger.info(
        "prefill state push: PUB %s (rank %d, hwm %d, max %d B, heartbeat %.2f s, "
        "launch mode %s%s)",
        endpoint,
        rank,
        hwm,
        pusher.max_bytes,
        pusher.heartbeat_s,
        pusher.launch_mode,
        ", extend passes only" if pusher.extend_only else "",
    )
    return pusher


def parse_launch_mode(value):
    """SGLANG_PREFILL_STATE_PUSH_LAUNCH_MODE: "busy" (also when empty) or
    "always". Another value stops the startup."""
    mode = (value or "busy").strip().lower()
    if mode not in LAUNCH_MODES:
        raise ValueError(
            f"SGLANG_PREFILL_STATE_PUSH_LAUNCH_MODE={value!r} is not one of "
            f"{', '.join(LAUNCH_MODES)}"
        )
    return mode


class PrefillStatePusher:
    """Builds and sends one message per launch_now(), launch_after() or
    flush() with news.

    Every method runs on the scheduler thread, so only that thread uses the
    socket. on_result() and launch_now() can run while the prefill launch
    thread launches the next pass; on_result only appends a stamp.
    """

    def __init__(
        self,
        tracker,
        *,
        socket,
        endpoint="",
        rank=0,
        max_bytes=DEFAULT_MAX_BYTES,
        heartbeat_s=1.0,
        launch_mode="busy",
        extend_only=False,
    ):
        self.tracker = tracker
        self.socket = socket
        self.endpoint = endpoint
        self.rank = rank
        self.max_bytes = max(MIN_MAX_BYTES, min(MAX_MAX_BYTES, int(max_bytes)))
        # <= 0 turns the heartbeat off.
        self.heartbeat_s = float(heartbeat_s)
        # "busy": a launch goes out at the launch only while the previous
        # pass runs on the GPU; "always": at every launch (launch_now).
        self.launch_mode = parse_launch_mode(launch_mode)
        self.launch_always = self.launch_mode == "always"
        # Symmetric worker: the launch and result hooks skip passes that are
        # not extend passes (extend_pass); decode passes produce no events.
        self.extend_only = bool(extend_only)
        self.epoch = tracker.epoch
        self.seq = 0
        self.sent_total = 0
        self.dropped_total = 0
        self.errors_total = 0
        self.oversize_total = 0
        self.lost_events_total = 0
        self.last_send_mono = -math.inf
        self.clock = GpuClock()
        self._results = []
        self._launches = []
        self._prev_end = None  # (forward_iter, end event) of the last result
        # The pass that launch_now() noted; on_launch() skips it once.
        self._early_key = None
        # "always" mode: the pass whose launch launch_now() held before an
        # inline launch on an idle GPU; launch_after() sends it.
        self._held_key = None
        # The /v1/loads sample sequence at the previous flush() call.
        self._seen_capture = getattr(tracker, "sequence", None)
        # sampled_monotonic_s of the newest snapshot sent.
        self._last_sampled = -math.inf
        self._prev_encode_us = self._prev_send_us = self._prev_bytes = None
        self._prev_flush_us = None
        self._last_cost = None
        self._last_launches = (0, False)
        self._logged = set()
        self._stats = self._new_stats()
        self._stats_since = time.monotonic()

    # -- hooks ---------------------------------------------------------------

    def _skip(self, batch):
        """extend_only (symmetric worker): a pass that is not an extend pass."""
        return self.extend_only and not extend_pass(batch)

    def on_result(self, batch, tracer=None):
        """Stamp a processed result; called just after its first output wait.

        The end event is queried before the host time is read, so an event
        that has completed did so before the stamp (a valid clock sample).
        """
        try:
            if self._skip(batch):
                return
            iteration = getattr(batch, "forward_iter", None)
            get = getattr(tracer, "gpu_events", None)
            events = (
                get(iteration)
                if get is not None and _int(iteration) is not None
                else None
            )
            done = bool(events[1].query()) if events is not None else False
            self._results.append(
                (
                    iteration,
                    getattr(batch, "launch_ts", None),
                    time.monotonic(),
                    events,
                    done,
                )
            )
            if len(self._results) > MAX_PENDING_EVENTS:
                del self._results[0]
                self.lost_events_total += 1
        except Exception as error:
            self.errors_total += 1
            self._log_once("on_result", error)

    def launch_now(self, scheduler, batch, *, concurrent=False):
        """Note the pass being launched; send it now when the mode allows.

        Called during the launch of ``batch`` in the overlap loop: just before
        the blocking launch call in run_batch (``concurrent`` False), or, with
        the launch thread, on this thread while the worker thread launches the
        pass (``concurrent`` True: before the previous result's processing,
        or alone for the first pass after an idle GPU in "always" mode).
        ``batch`` is not in the result queue yet; the message shows it in
        flight.

        "busy" mode: sent only when the newest queued pass has not ended on
        the GPU (newest_pass_running), because this thread then waits for that
        pass next. Otherwise the GPU waits for this launch or a result is
        ready, and the launch goes out with the end-of-iteration flush().
        "always" mode: sent at once when the worker launches meanwhile or the
        previous pass still runs. Before an inline launch on an idle GPU it is
        held, and launch_after() sends it after the launch call, so the
        message never runs in front of that launch. Never raises; returns
        whether it sent a message.
        """
        started = time.perf_counter()
        try:
            if self._skip(batch):
                return False
            key = _launch_key(batch)
            if key == self._early_key:
                return False
            self._early_key = key
            self._note_launch(batch)
            queue = getattr(scheduler, "result_queue", None)
            if not ((concurrent and self.launch_always) or newest_pass_running(queue)):
                if self.launch_always:
                    self._held_key = key
                return False
            sent = self._publish_launch(scheduler, batch, queue)
        except Exception as error:
            self.errors_total += 1
            self._log_once("launch_now", error)
            return False
        self._finish(started)
        return sent

    def launch_now_then(self, scheduler, batch, work=None):
        """The launch thread's concurrent ``work`` (None: no work), with
        launch_now() first. It runs on the scheduler thread while the worker
        thread launches ``batch``."""

        def run():
            self.launch_now(scheduler, batch, concurrent=True)
            if work is not None:
                work()

        return run

    def launch_after(self, scheduler, batch):
        """In "always" mode, send the launch that launch_now() held before an
        inline launch on an idle GPU, now that the launch call has returned.

        Called in run_batch right after the inline launch call, before the
        sched tracer records the pass's end event. A no-op in "busy" mode and
        when launch_now() sent the launch already. Never raises; returns
        whether it sent a message.
        """
        if self._held_key is None:
            return False
        started = time.perf_counter()
        try:
            if self._skip(batch):
                return False
            key, self._held_key = self._held_key, None
            if key != _launch_key(batch):
                return False
            sent = self._publish_launch(
                scheduler, batch, getattr(scheduler, "result_queue", None)
            )
        except Exception as error:
            self.errors_total += 1
            self._log_once("launch_after", error)
            return False
        self._finish(started)
        return sent

    def on_launch(self, batch):
        """Note a launched pass after the loop's launch block, unless
        launch_now() already noted it."""
        try:
            if self._skip(batch):
                return
            if _launch_key(batch) == self._early_key:
                self._early_key = None
                return
            self._note_launch(batch)
        except Exception as error:
            self.errors_total += 1
            self._log_once("on_launch", error)

    def flush(self, scheduler, batch):
        """The end-of-iteration message: the launches and results not sent
        yet, or a heartbeat when nothing went out for heartbeat_s.

        ``batch`` is the pass the loop launched in this iteration (None when
        idle). Never raises; returns whether a message was sent.
        """
        started = time.perf_counter()
        try:
            # Whether /v1/loads sampled a snapshot since the previous call,
            # i.e. in this loop iteration (one flush per iteration).
            sequence = getattr(self.tracker, "sequence", None)
            captured = sequence != self._seen_capture
            self._seen_capture = sequence
            now = time.monotonic()
            if not (self._launches or self._results) and (
                self.heartbeat_s <= 0 or now - self.last_send_mono < self.heartbeat_s
            ):
                return False
            sent = self._publish(scheduler, batch, now, reuse=captured)
        except Exception as error:
            self.errors_total += 1
            self._log_once("flush", error)
            return False
        self._finish(started)
        return sent

    def close(self):
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None

    # -- message -------------------------------------------------------------

    def _note_launch(self, batch):
        self._launches.append(
            PassEvent(
                kind="launch",
                forward_iter=_int(getattr(batch, "forward_iter", None)),
                launch_mono_s=_time(getattr(batch, "launch_ts", None)),
                observed_mono_s=time.monotonic(),
            )
        )
        if len(self._launches) > MAX_PENDING_EVENTS:
            del self._launches[0]
            self.lost_events_total += 1

    def _captured(self, events):
        """The tracker's /v1/loads value, if a capture() sampled it after
        every event of this message and after the last snapshot sent."""
        value = getattr(self.tracker, "value", None)
        stamp = getattr(self.tracker, "last_capture", None)
        # unavailable() also replaces the value; its stamp differs.
        if value is None or stamp is None or value.sampled_monotonic_s != stamp:
            return None
        if events and stamp < events[-1].observed_mono_s:
            return None
        if stamp < self._last_sampled:
            return None
        return value

    def _publish_launch(self, scheduler, batch, queue):
        """The launch message of ``batch``, which is being launched and is not
        in the result queue yet: the snapshot and the in-flight list show it
        after the queued passes."""
        view = list(queue) if isinstance(queue, (deque, list, tuple)) else []
        view.append((batch, None))
        return self._publish(
            scheduler, batch, time.monotonic(), queue=view, launching=batch
        )

    def _publish(
        self, scheduler, batch, now, *, queue=None, launching=None, reuse=False
    ):
        tracer = getattr(scheduler, "sched_tracer", None)
        gpu_events = getattr(tracer, "gpu_events", None)
        timing = bool(getattr(tracer, "gpu_timing_enabled", False))
        self.clock.expire(now)
        results, self._results = self._results, []
        events = [self._result_event(*item) for item in results] + self._launches
        self._launches = []
        # Every noted launch goes out in this message, a held one too.
        self._held_key = None
        events.sort(key=lambda e: e.observed_mono_s)
        launches = sum(e.kind == "launch" for e in events)
        if launches:
            trigger = "launch"
        else:
            trigger = "result" if events else "heartbeat"
        if queue is None:
            queue = getattr(scheduler, "result_queue", None)
        inflight = self._inflight(queue, gpu_events if timing else None, launching)
        self.seq += 1
        started = time.perf_counter()
        snapshot = omitted = source = None
        domain = max_age = sampled = None
        try:
            value = self._captured(events) if reuse else None
            if value is None:
                value = self.tracker.build(
                    scheduler, batch, now=now, sequence=self.seq, result_queue=queue
                )
                source = "push_build"
            else:
                source = "v1_loads_capture"
            built = time.perf_counter()
            snapshot = named_snapshot(value)
            domain, max_age = value.clock_domain_id, value.max_age_s
            sampled = value.sampled_monotonic_s
        except Exception as error:
            built = time.perf_counter()
            snapshot, source = None, None
            omitted = "build_error:" + type(error).__name__
            self.errors_total += 1
            self._log_once("build", error)
        sent_mono, sent_unix = time.monotonic(), time.time()
        if snapshot is not None:
            snapshot["sequence"] = self.seq
            # As public_snapshot() would report it at send time.
            age = sent_mono - sampled
            snapshot["age_s"] = age
            snapshot["stale"] = age < 0 or age > max_age
            if snapshot["stale"] and snapshot["reason"] is None:
                snapshot["reason"] = "stale"
        converted = time.perf_counter()
        message = PrefillStatePush(
            schema=SCHEMA,
            schema_version=SCHEMA_VERSION,
            epoch=self.epoch,
            seq=self.seq,
            rank=self.rank,
            trigger=trigger,
            sent_mono_s=sent_mono,
            sent_unix_s=sent_unix,
            mono_to_unix_s=sent_unix - sent_mono,
            clock_domain_id=domain if domain is not None else clock_domain(),
            events=events,
            inflight=inflight,
            gpu_timing="cuda_events" if timing else "unavailable",
            gpu_clock_samples=len(self.clock.samples),
            dropped_total=self.dropped_total,
            errors_total=self.errors_total,
            oversize_total=self.oversize_total,
            build_us=(built - started) * 1e6,
            convert_us=(converted - built) * 1e6,
            prev_encode_us=self._prev_encode_us,
            prev_send_us=self._prev_send_us,
            prev_bytes=self._prev_bytes,
            prev_flush_us=self._prev_flush_us,
            snapshot_source=source,
            snapshot=snapshot,
            snapshot_omitted=omitted,
        )
        payload = _encoder.encode(message)
        if len(payload) > self.max_bytes:
            self.oversize_total += 1
            message.oversize_total = self.oversize_total
            message.snapshot, message.snapshot_omitted = None, "over_max_bytes"
            message.snapshot_source = None
            payload = _encoder.encode(message)
        encoded = time.perf_counter()
        sent = self._send(payload)
        done = time.perf_counter()
        self.last_send_mono = sent_mono
        if message.snapshot is not None:
            self._last_sampled = sampled
        self._prev_encode_us = (encoded - converted) * 1e6
        self._prev_send_us = (done - encoded) * 1e6
        self._prev_bytes = len(payload)
        self._last_cost = (
            message.build_us,
            message.convert_us,
            self._prev_encode_us,
            self._prev_send_us,
            len(payload),
            sent,
            source == "v1_loads_capture",
        )
        # Launch events in this message, and whether it went out at a launch.
        self._last_launches = (launches, launching is not None)
        return sent

    def _send(self, payload):
        try:
            self.socket.send_multipart(
                (TOPIC, self.seq.to_bytes(8, "big"), payload), flags=zmq.NOBLOCK
            )
        except zmq.Again:
            self.dropped_total += 1
            return False
        self.sent_total += 1
        return True

    def _result_event(self, iteration, launch_ts, observed, events, done):
        event = PassEvent(
            kind="result",
            forward_iter=_int(iteration),
            launch_mono_s=_time(launch_ts),
            observed_mono_s=observed,
        )
        if events is None:
            return event
        try:
            start, end = events
            if not end.query():
                event.gpu_ended = False
                event.gpu_started = bool(start.query())
                return event
            event.gpu_started = event.gpu_ended = True
            event.gpu_ms = start.elapsed_time(end)
            prev = self._prev_end
            if prev is not None and prev[0] + 1 == iteration:
                event.gpu_gap_before_ms = prev[1].elapsed_time(start)
            self._prev_end = (iteration, end)
            if done:
                # Completed before the stamp: a valid upper-bound sample.
                self.clock.observe(end, observed)
            end_mono = self.clock.to_mono(end)
            if end_mono is not None:
                event.gpu_end_mono_s = end_mono
                event.gpu_start_mono_s = end_mono - event.gpu_ms / 1e3
        except Exception as error:
            self.errors_total += 1
            self._log_once("gpu_events", error)
        return event

    def _inflight(self, queue, gpu_events, launching=None):
        """The queued passes; ``launching`` is the pass whose launch is in
        progress (launch_now), the last entry of the queue view."""
        if not isinstance(queue, (deque, list, tuple)):
            return []
        out = []
        for pair in islice(queue, MAX_INFLIGHT_PASSES):
            batch = pair[0] if isinstance(pair, (tuple, list)) and pair else None
            iteration = getattr(batch, "forward_iter", None)
            event = PassEvent(
                kind="inflight",
                forward_iter=_int(iteration),
                launch_mono_s=_time(getattr(batch, "launch_ts", None)),
            )
            events = (
                gpu_events(iteration)
                if gpu_events is not None and _int(iteration) is not None
                else None
            )
            if events is not None:
                try:
                    start, end = events
                    event.gpu_started = bool(start.query())
                    if launching is not None and batch is launching:
                        # The sched tracer records the end event after the
                        # launch; an unrecorded CUDA event reads as complete.
                        event.gpu_ended = False
                    else:
                        event.gpu_ended = event.gpu_started and bool(end.query())
                    if event.gpu_started:
                        event.gpu_start_mono_s = self.clock.to_mono(start)
                    if event.gpu_ended:
                        event.gpu_ms = start.elapsed_time(end)
                        event.gpu_end_mono_s = self.clock.to_mono(end)
                except Exception as error:
                    self.errors_total += 1
                    self._log_once("gpu_events", error)
            out.append(event)
        return out

    # -- diagnostics ---------------------------------------------------------

    def _finish(self, started):
        """After a message: time the whole call (event work, snapshot,
        conversion, encode, send, freeing the message) for the next one."""
        total = (time.perf_counter() - started) * 1e6
        self._prev_flush_us = total
        if self._last_cost is not None:
            self._account(total)

    @staticmethod
    def _new_stats():
        return dict(
            messages=0,
            sent=0,
            reused=0,
            bytes=0,
            max_bytes=0,
            build_us=0.0,
            max_build_us=0.0,
            convert_us=0.0,
            encode_us=0.0,
            send_us=0.0,
            flush_us=0.0,
            max_flush_us=0.0,
            # Launch events sent by launch_now()/launch_after() (at the
            # launch) and by flush() (at the end of the iteration); the
            # messages sent at the launch and their whole-call time.
            launches_at_launch=0,
            launches_at_end=0,
            launch_msgs=0,
            launch_msg_us=0.0,
        )

    def _account(self, total):
        build, convert, encode, send, nbytes, sent, reused = self._last_cost
        launches, at_launch = self._last_launches
        st = self._stats
        if at_launch:
            st["launches_at_launch"] += launches
            st["launch_msgs"] += 1
            st["launch_msg_us"] += total
        else:
            st["launches_at_end"] += launches
        st["messages"] += 1
        st["sent"] += int(sent)
        st["reused"] += int(reused)
        st["bytes"] += nbytes
        st["max_bytes"] = max(st["max_bytes"], nbytes)
        st["build_us"] += build
        st["max_build_us"] = max(st["max_build_us"], build)
        st["convert_us"] += convert
        st["encode_us"] += encode
        st["send_us"] += send
        st["flush_us"] += total
        st["max_flush_us"] = max(st["max_flush_us"], total)
        now = time.monotonic()
        if now - self._stats_since < STATS_LOG_INTERVAL_S:
            return
        n = max(1, st["messages"])
        # The launch fields come last: parsers of the older line still match.
        logger.info(
            "prefill state push: %d msgs (%d sent, %d dropped total, %d reused "
            "the /v1/loads sample) in %.0f s; mean us flush %.0f build %.0f "
            "convert %.0f encode %.0f send %.0f; max us flush %.0f build %.0f; "
            "mean %.0f B max %d B; errors %d oversize %d lost events %d; "
            "launches sent at the launch %d at the iteration end %d; "
            "launch msgs %d mean us %.0f (launch mode %s)",
            st["messages"],
            st["sent"],
            self.dropped_total,
            st["reused"],
            now - self._stats_since,
            st["flush_us"] / n,
            st["build_us"] / n,
            st["convert_us"] / n,
            st["encode_us"] / n,
            st["send_us"] / n,
            st["max_flush_us"],
            st["max_build_us"],
            st["bytes"] / n,
            st["max_bytes"],
            self.errors_total,
            self.oversize_total,
            self.lost_events_total,
            st["launches_at_launch"],
            st["launches_at_end"],
            st["launch_msgs"],
            st["launch_msg_us"] / max(1, st["launch_msgs"]),
            self.launch_mode,
        )
        self._stats = self._new_stats()
        self._stats_since = now

    def _log_once(self, where, error):
        if where in self._logged:
            return
        self._logged.add(where)
        logger.warning(
            "prefill state push: %s failed (%s: %s); the engine continues and "
            "errors_total counts later failures",
            where,
            type(error).__name__,
            error,
        )
