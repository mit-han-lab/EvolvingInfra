"""SGLANG_PREFILL_LAUNCH_THREAD: launch the PD prefill forward on a worker
thread, and process the previous result on the scheduler thread meanwhile.

Why. ``event_loop_overlap_disagg_prefill`` launches forward k+1 and only then
processes result k. The launch blocks inside CUDA launch calls whenever the
stream's pending-launch queue is full. In GLM-5.2 profiles on GB300 (TP0)
an eager forward's blocked launches returned with about
1,055 launches queued and the CPU about 330 ms (one forward) ahead; with the
8192-token prefill graph, cudaGraphLaunch blocked with about 171 graph
launches queued. So run_batch lasts about one GPU forward although Python is
not slow, and result k (its prefill_finished stamps and its KV sends to D)
waits for that although forward k has ended.

What. With the flag on, when a previous result is queued, the loop passes its
processing to ``run_batch`` as ``concurrent_work``. run_batch does everything
as before on the scheduler thread except the forward block (the part inside
``forward_stream_ctx``: input resolve, the model forward, publish/relay and
the result D2H on copy_stream), which one worker thread runs. While the
worker launches, the scheduler thread runs ``concurrent_work``: it waits for
result k's copy_done (forward k) and processes the result. Then it waits for
the launch to finish and run_batch continues as before. Result k is then
processed right after forward k ends instead of after the launch of k+1.

Loop order. The loop takes this path at every launch with a previous result
queued, on every TP rank, instead of the result-first decision (the
gate-pending reorder and SGLANG_PREFILL_RESULT_FIRST_WHEN_READY). For the
loop it is the early path: the result is processed after
``_m2_mark_pre_launch_writes`` and before the flush after the launch
(``after_early_result``), so 6h holds its write-throughs for the next flush on
every rank alike. Neither result processing nor run_batch issues a TP
collective (prefill_result_first.py, T4 audit); the loop's collectives stay at
their fixed points.

GPU order. The worker enters ``forward_stream_ctx`` (the current stream is
per thread); the scheduler thread keeps ``schedule_stream``. Result k's own
GPU work (the tree insert's writes to its requests' req_to_token rows, frees,
small copies) is enqueued on schedule_stream after forward k+1's
``wait_stream(schedule_stream)`` and before its WAR barrier. So on the GPU it
is ordered neither before nor after forward k+1. That is safe because forward
k+1 reads none of it: requests that finish in batch k are not in batch k+1, a
chunked request's next chunk reads only rows written before its launch, and
pages freed here are reallocated only after the barrier (the next admission).
The early path already runs the 6h writes of a result concurrently with the
next forward in the same way.

Threads. One worker per scheduler, created at loop start. It sets the
scheduler's CUDA device and runs each launch under DynamicGradMode, like the
event loop. The GIL is released while a launch blocks (aten ops, CUDA graph
replay, Triton launchers), so the scheduler thread runs then. An exception in
the launch is re-raised on the scheduler thread after concurrent_work ends.

Not used for a batch with grammar (the grammar barrier advances the previous
batch's grammar inside the forward) or with a speculative algorithm other
than EAGLE. Refused at loop start with DP attention (MLP sync), PP > 1, the
M4 load quarantine, the unified memory pool and pdmux.

Phases (SGLANG_SCHED_PHASE_TIMERS): "launch_thread_prev_result" (the
concurrent result processing), "launch_thread_join_wait" (the wait for the
launch after it); count "launch_thread_result_before_launch_end" (1 when the
processing ended while the launch was still running). Every TP rank logs one
line every LOG_PERIOD_S:

    LAUNCH_THREAD rank=<r> concurrent=.. sync_grammar=.. sync_spec=..
        result_before_launch_end=..
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Tuple

from decoupled_engine.srt.managers.scheduler_components.phase_timers import PHASES

logger = logging.getLogger(__name__)

STATS: Counter = Counter()
LOG_PERIOD_S = 60.0
LOG_TAG = "LAUNCH_THREAD"
THREAD_NAME = "prefill-launch"


def launch_thread_enabled() -> bool:
    from decoupled_engine.srt.environ import envs

    return bool(envs.SGLANG_PREFILL_LAUNCH_THREAD.get())


def _incompatible(scheduler: Any) -> Optional[str]:
    if not getattr(scheduler, "enable_overlap", False):
        return "the overlap scheduler is off"
    if getattr(scheduler, "require_mlp_sync", False):
        return "DP attention (MLP sync) keeps one order on every rank"
    if int(getattr(getattr(scheduler, "ps", None), "pp_size", 1) or 1) != 1:
        return "pipeline parallelism"
    if getattr(scheduler, "_load_quarantine", None) is not None:
        return "the M4 load quarantine records forward_done inside the launch"
    if getattr(scheduler, "enable_unified_memory", False):
        return "the unified memory pool records forward_done inside the launch"
    if getattr(scheduler, "enable_pdmux", False):
        return "pdmux"
    return None


class PrefillLaunchThread:
    """The launch worker of one PD prefill overlap loop."""

    def __init__(
        self,
        tp_rank: int,
        device_index: Optional[int],
        set_device: Optional[Callable[[int], None]] = None,
        grad_mode: Optional[Callable[[], Any]] = None,
    ):
        self.tp_rank = tp_rank
        self._device_index = device_index
        self._set_device = set_device
        self._grad_mode = grad_mode
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=THREAD_NAME,
            initializer=self._init_thread,
        )
        self._t_log = time.monotonic()

    @classmethod
    def for_scheduler(cls, scheduler: Any) -> Optional[PrefillLaunchThread]:
        """Resolve once at loop start: None when the flag is off. Raises when
        the flag is on with a configuration it does not support."""
        tp_rank = int(getattr(getattr(scheduler, "ps", None), "tp_rank", 0) or 0)
        if not launch_thread_enabled():
            logger.info("%s rank=%d off", LOG_TAG, tp_rank)
            return None
        reason = _incompatible(scheduler)
        if reason is not None:
            raise ValueError(
                f"SGLANG_PREFILL_LAUNCH_THREAD is not supported with {reason}"
            )
        from decoupled_engine.srt.utils import DynamicGradMode

        # The scheduler's device module (torch.cuda on GPU).
        module = getattr(scheduler, "device_module", None)
        device_index = None
        set_device = None
        if (
            getattr(scheduler, "device", "cuda") != "cpu"
            and hasattr(module, "current_device")
            and hasattr(module, "set_device")
        ):
            device_index = module.current_device()
            set_device = module.set_device
        logger.info("%s rank=%d on device_index=%s", LOG_TAG, tp_rank, device_index)
        return cls(tp_rank, device_index, set_device, DynamicGradMode)

    def _init_thread(self) -> None:
        # The CUDA current device is per host thread.
        if self._set_device is not None and self._device_index is not None:
            self._set_device(self._device_index)

    def _launch(self, launch: Callable[[], Any]) -> Any:
        if self._grad_mode is None:
            return launch()
        with self._grad_mode():
            return launch()

    def applies(self, batch: Any) -> bool:
        """Whether run_batch may launch ``batch`` on the worker."""
        if getattr(batch, "has_grammar", False):
            STATS["sync_grammar"] += 1
            return False
        spec = getattr(batch, "spec_algorithm", None)
        if spec is not None and not (spec.is_none() or spec.is_eagle()):
            STATS["sync_spec"] += 1
            return False
        return True

    def run(
        self, launch: Callable[[], Any], concurrent_work: Callable[[], None]
    ) -> Any:
        """Run ``launch`` on the worker and ``concurrent_work`` here; return
        the launch's value once both are done."""
        STATS["concurrent"] += 1
        future = self._executor.submit(self._launch, launch)
        t_work = PHASES.start()
        try:
            concurrent_work()
        finally:
            PHASES.stop("launch_thread_prev_result", t_work)
            before_end = not future.done()
            if before_end:
                STATS["result_before_launch_end"] += 1
            PHASES.count("launch_thread_result_before_launch_end", int(before_end))
            t_join = PHASES.start()
            try:
                value = future.result()
            finally:
                PHASES.stop("launch_thread_join_wait", t_join)
        self._maybe_log()
        return value

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._t_log < LOG_PERIOD_S:
            return
        self._t_log = now
        logger.info(
            "%s rank=%d %s",
            LOG_TAG,
            self.tp_rank,
            " ".join(
                f"{k}={STATS[k]}"
                for k in (
                    "concurrent",
                    "sync_grammar",
                    "sync_spec",
                    "result_before_launch_end",
                )
            ),
        )

    def thread_names(self) -> Tuple[str, ...]:
        return tuple(
            t.name for t in threading.enumerate() if t.name.startswith(THREAD_NAME)
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
