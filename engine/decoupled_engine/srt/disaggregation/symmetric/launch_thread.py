"""SGLANG_SYMMETRIC_LAUNCH_THREAD: the PD prefill launch thread
(disaggregation/prefill_launch_thread.py, SGLANG_PREFILL_LAUNCH_THREAD) in the
symmetric overlap loop (``Scheduler.event_loop_overlap``).

Why. The symmetric loop launches forward k+1 and only then processes result k.
The launch blocks inside CUDA launch calls whenever the pending-launch queue
is full, so ``run_batch`` lasts about one GPU forward, and result k (its
FINAL publication, token streaming, frees) waits for it. When a launch must
wait for a pending shared-loader gate, the gate-pending reorder processes
result k first instead, and the GPU then idles while the scheduler thread
does that work (GLM-5.2 3P6D on GB300 at concurrency 90: 112-150 ms of
"reorder_prev_result" before host-load passes; P GPU gaps 192-198 ms vs 67-71
ms on standard P).

What. When the flag is on and a launch qualifies (below), the loop passes the
processing of result k to ``run_batch`` as ``concurrent_work``. run_batch then
runs its forward block on one worker thread (``PrefillLaunchThread.run``):
input resolve, the model forward, future-map publish, the 6a-sym seed
staging, ``copy_done``, the relay and the result D2H on copy_stream. The
scheduler thread processes result k meanwhile, then waits for the launch.
Everything else of run_batch (forward_ct, ``symmetric_forward_start``, the
tracker hooks before the launch; the EAGLE snapshot and the tracer end event
after it) stays on the scheduler thread, in the same order as before.

Which launches. ``use_for`` is True only for a plain extend pass with a
previous result queued: overlap not disabled for the batch, forward mode
EXTEND without decoding requests (no mixed chunk), no grammar, spec none or
EAGLE, no live beam group and no request with multimodal inputs. Decode
passes, mixed passes, the first launch and the first launch after idle keep
today's path (inline, with the gate-pending reorder). The rule reads only
lockstep state (batch content, loop state, env), never rank-local state
(gate pending, loader progress, write-queue length), so every TP rank takes
the same path at every launch. A threaded launch does not consult the
gate-pending reorder: when its gate is pending, the worker waits in the
consumer gate while the scheduler thread processes result k.

Why it is safe on a worker that both prefills and decodes.
- TP collectives. The collectives of a pass stay at fixed points before the
  launch (request broadcast, BWP polls, the symmetric pre-schedule votes,
  HiCache lockstep all-reduces in admission, the structural digest, the 6h
  lockstep MIN-reduce in ``mark_pre_launch``). Result processing issues none
  (gate_pending_reorder.py, T4 audit). The work runs inside
  ``collective_guard.forbid``: a symmetric vote or a HiCache lockstep
  all-reduce reached from it raises RuntimeError instead of hanging a rank.
- Requests. Extend batch k+1 holds new admissions and at most the chunked
  request. A request that finished prefill in batch k went to the running
  batch, not to batch k+1; a decode result k is about running requests;
  retraction runs only in decode iterations. The chunked request's middle
  chunk result does no tree insert and no row rewrite; its publication reads
  rows written at admission k or earlier, while forward k+1 writes only the
  pages of the new chunk.
- GPU order. Result k's GPU work goes on schedule_stream around forward k+1's
  ``wait_stream`` and before barrier k+1: neither reads what the other
  writes. Pages freed by result k are reused only by admission k+2, after
  barrier k+1. With SGLANG_SYMMETRIC_PUBLISH_UNFENCED the loop records the
  pre-WAR event before a threaded launch too, so publication readbacks see
  the rows as of this launch and do not wait for the forward being launched.
- 6h. ``mark_pre_launch`` runs before every path and a threaded launch flushes
  with ``after_early_result=True``, so every rank merges the same writes.
- Exports, late D. ``symmetric_forward_start(k+1)`` runs before the worker,
  ``symmetric_forward_finished(k)`` in the work: the order of today's path.
  Publication, binder and import steps keep their points.
- Shared CPU state. The worker reads batch k+1, request fields fixed at
  arrival, ``req.export_mark`` (only set -> None), streams and model state.
  Configurations that break this split are refused at loop start (below).

Refused (ValueError at loop start): the PD list (overlap off, DP attention,
PP > 1, the M4 load quarantine, the unified memory pool, pdmux), plus the KV
canary token assert, adaptive speculative decoding, dLLM, hierarchical sparse
attention, decode KV offload, embedding models and, until it is checked on
an H100 DSv4 worker, DSv4 window SWA residency.

Logs. Startup, every TP rank:

    SYM_LAUNCH_THREAD rank=<r> on device_index=<i> rule=extend_with_prev_result

About every 60 s, every TP rank, at a decision count that is a multiple of
LOG_EVERY_DECISIONS (so the ranks of a worker usually log at the same count;
lines with the same ``decisions=`` must have equal counters, except
``result_before_launch_end``, which depends on timing):

    SYM_LAUNCH_THREAD rank=<r> decisions=N threaded=N inline_no_prev=N
        inline_disable_overlap=N inline_decode=N inline_other_mode=N
        inline_mixed=N inline_grammar=N inline_spec=N inline_beam=N
        inline_mm=N result_before_launch_end=N

Phase timers are the PD names ("launch_thread_prev_result",
"launch_thread_join_wait", count "launch_thread_result_before_launch_end").
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any, Callable, Optional

from decoupled_engine.srt.disaggregation import prefill_launch_thread as pd_launch_thread
from decoupled_engine.srt.disaggregation.prefill_launch_thread import PrefillLaunchThread
from decoupled_engine.srt.managers.scheduler_components import collective_guard
from decoupled_engine.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger(__name__)

STATS: Counter = Counter()
LOG_TAG = "SYM_LAUNCH_THREAD"
LOG_PERIOD_S = 60.0
LOG_EVERY_DECISIONS = 64
RULE = "extend_with_prev_result"
GUARD_REASON = "symmetric launch thread: previous result"
# Every launch with a batch counts once in "decisions" and once in exactly
# one of these.
COUNTERS = (
    "threaded",
    "inline_no_prev",
    "inline_disable_overlap",
    "inline_decode",
    "inline_other_mode",
    "inline_mixed",
    "inline_grammar",
    "inline_spec",
    "inline_beam",
    "inline_mm",
)


def symmetric_launch_thread_enabled() -> bool:
    from decoupled_engine.srt.environ import envs

    return bool(envs.SGLANG_SYMMETRIC_LAUNCH_THREAD.get())


def symmetric_incompatible(scheduler: Any) -> Optional[str]:
    """Why the flag cannot run with this scheduler, or None."""
    reason = pd_launch_thread._incompatible(scheduler)
    if reason is not None:
        return reason
    from decoupled_engine.srt.environ import envs

    if envs.SGLANG_KV_CANARY_ENABLE_VERIFY_TOKEN_ASSERT.get():
        return (
            "the KV canary token assert (the forward reads output_ids, which "
            "result processing appends to)"
        )
    server_args = getattr(scheduler, "server_args", None)
    if getattr(server_args, "speculative_adaptive", False) is True:
        return (
            "adaptive speculative decoding (decode results update the "
            "controller that the forward reads)"
        )
    if getattr(scheduler, "dllm_config", None) is not None:
        return "a diffusion LLM"
    if getattr(scheduler, "enable_hisparse", False) is True:
        return "hierarchical sparse attention"
    if getattr(scheduler, "decode_offload_manager", None) is not None:
        return "decode KV offload"
    if getattr(scheduler, "is_generation", True) is False:
        return (
            "an embedding or reward model (run_batch has no forward block to "
            "launch on the worker)"
        )
    tree_cache = getattr(scheduler, "tree_cache", None)
    if getattr(tree_cache, "dsv4_uniform_swa_residency", "full") == "window":
        return "DSv4 window SWA residency (not validated yet)"
    return None


class SymmetricLaunchThread(PrefillLaunchThread):
    """The launch worker of one symmetric overlap loop."""

    @classmethod
    def for_symmetric_loop(cls, scheduler: Any) -> Optional[SymmetricLaunchThread]:
        """Resolve once at loop start. None (and no log line) when the flag is
        off; raises when it is on with a configuration it does not support."""
        if not symmetric_launch_thread_enabled():
            return None
        reason = symmetric_incompatible(scheduler)
        if reason is not None:
            from decoupled_engine.srt.arg_groups.symmetric_defaults import refuse

            refuse(
                "SGLANG_SYMMETRIC_LAUNCH_THREAD",
                f"SGLANG_SYMMETRIC_LAUNCH_THREAD is not supported with {reason}",
            )
            return None
        from decoupled_engine.srt.utils import DynamicGradMode

        tp_rank = int(getattr(getattr(scheduler, "ps", None), "tp_rank", 0) or 0)
        # The scheduler's device module (torch.cuda on GPU), as for_scheduler.
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
        logger.info(
            "%s rank=%d on device_index=%s rule=%s",
            LOG_TAG,
            tp_rank,
            device_index,
            RULE,
        )
        return cls(tp_rank, device_index, set_device, DynamicGradMode)

    @staticmethod
    def refusal(scheduler: Any, batch: Any) -> Optional[str]:
        """The counter of the first batch rule (3-7) that ``batch`` breaks,
        or None for a plain extend pass."""
        mode = getattr(batch, "forward_mode", None)
        if mode == ForwardMode.MIXED or getattr(batch, "decoding_reqs", None):
            return "inline_mixed"
        if mode != ForwardMode.EXTEND:
            if mode == ForwardMode.DECODE:
                return "inline_decode"
            return "inline_other_mode"
        # PrefillLaunchThread.applies: the grammar barrier advances the
        # previous batch's grammar inside the forward; only spec none / EAGLE.
        if getattr(batch, "has_grammar", False):
            return "inline_grammar"
        spec = getattr(batch, "spec_algorithm", None)
        if spec is not None and not (spec.is_none() or spec.is_eagle()):
            return "inline_spec"
        # Beam groups register only when a request arrives, so the count
        # cannot rise during the launch; with none live the worker's relay
        # returns at once.
        coordinator = getattr(scheduler, "beam_coordinator", None)
        live = getattr(coordinator, "_num_live_groups", 0)
        if isinstance(live, int) and live > 0:
            return "inline_beam"
        # Result processing clears multimodal inputs of finished requests; an
        # aborted chunked request can be finished and still be in this batch.
        reqs = getattr(batch, "reqs", None) or ()
        if any(getattr(req, "multimodal_inputs", None) is not None for req in reqs):
            return "inline_mm"
        return None

    def batch_applies(self, scheduler: Any, batch: Any, count: bool = True) -> bool:
        """Whether ``batch`` itself may launch on the worker: a plain extend
        pass (rules 3-7). Counts its refusal reason when not, unless
        ``count`` is False (the state push's TP0-only path: the counters must
        stay equal on every TP rank)."""
        reason = self.refusal(scheduler, batch)
        if reason is not None and count:
            STATS[reason] += 1
        return reason is None

    def use_for(self, scheduler: Any, batch: Any, disable_overlap: bool) -> bool:
        """Whether the loop launches ``batch`` on the worker and processes the
        queued previous result meanwhile. Reads only lockstep state, so every
        TP rank decides alike."""
        STATS["decisions"] += 1
        if disable_overlap:
            # The loop processed the previous result before this launch.
            STATS["inline_disable_overlap"] += 1
            applies = False
        elif scheduler.last_batch is None or not scheduler.result_queue:
            STATS["inline_no_prev"] += 1
            applies = False
        else:
            applies = self.batch_applies(scheduler, batch)
            if applies:
                STATS["threaded"] += 1
        self._log_if_due()
        return applies

    @staticmethod
    def guarded(work: Callable[[], None]) -> Callable[[], None]:
        """``work`` inside collective_guard.forbid (on the calling thread)."""

        def run() -> None:
            with collective_guard.forbid(GUARD_REASON):
                work()

        return run

    def _maybe_log(self) -> None:
        # PrefillLaunchThread.run calls this after each threaded launch; the
        # symmetric line is written from use_for instead (every launch).
        return

    def stats_line(self) -> str:
        return (
            f"{LOG_TAG} rank={self.tp_rank} decisions={STATS['decisions']} "
            + " ".join(f"{k}={STATS[k]}" for k in COUNTERS)
            + " result_before_launch_end="
            + str(pd_launch_thread.STATS["result_before_launch_end"])
        )

    def _log_if_due(self) -> None:
        if STATS["decisions"] % LOG_EVERY_DECISIONS:
            return
        now = time.monotonic()
        if now - self._t_log < LOG_PERIOD_S:
            return
        self._t_log = now
        logger.info(self.stats_line())
