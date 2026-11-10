"""Just-in-time peer pull for PD prefill workers (opt-in).

Stage 1 of queued-prefix protection: (b2) re-pull at admission and (b1)
deferred pull near the queue head.

A non-owner request's peer pull lands in the host pool seconds after dispatch.
In a long prefill queue the request then waits for minutes, and nothing locks
the landed pages (they are host-only leaves): write-through backups and other
pulls evict them, and the request recomputes the prefix it had pulled.

(b2) ``--hicache-storage-repull-on-admission``: admission checks each hinted
request after its prefix match and after the bootstrap-prefetch deferral. When
the device+host match is short of the hint's ``expected_hit_tokens`` by at
least max(hint ``min_local_deficit_tokens``,
``--hicache-storage-repull-min-tokens``), it issues the pull again through the
arrival path (``Scheduler._prefetch_kvcache``: the same hint gate, anchored at
the end of the current match) and skips the request in this pass. The
existing ``check_prefetch_progress`` gate then holds the request until the
pull ends, and the admission loop admits the requests behind it meanwhile. At
most ``--hicache-storage-repull-max-attempts`` re-pulls per request.

(b1) ``--hicache-storage-jit-pull queue``: an arriving hinted request is not
pulled when more than ``--hicache-storage-jit-pull-now-reqs`` requests
(bootstrap + waiting queue) or more than ``--hicache-storage-jit-pull-now-
tokens`` input tokens (plus the chunked request's remainder) are ahead of it;
it keeps its hint and is marked pending. Each pass, after the HiCache events
and before admission, the first lead-reqs + ``LEAD_SCAN_EXTRA`` requests of the
waiting queue are scanned in queue order, and a pending one is pulled (through
the arrival path) when at most ``--hicache-storage-jit-pull-lead-reqs``
requests, or at most ``--hicache-storage-jit-pull-lead-tokens`` input tokens,
are ahead of it. A request in the bootstrap queue is not runnable and is not
pulled, except with late D (``--disaggregation-late-d``): there a late room
waits in the bootstrap queue until its decode binds it, and the binder sends
the ADMIT only a few iterations before the room reaches the head. So the late
rooms that wait for their bind count as queued behind the waiting queue, in
bootstrap-queue order (the binder's ADMIT order), and the scan continues into
them: a pending one within the lead is pulled before its ADMIT. With late D
off the scan reads the waiting queue only, as before. When admission reaches a
request whose pull is still deferred (the
requests ahead of it are held, or the policy moved it up), the pull is issued
there and the request is held like a re-pull. So pulled pages wait seconds,
not minutes, for their request. (b2), when on, then repairs what is lost
anyway.

Bounded wait: the hold is the prefetch stop policy's own bound (``best_effort``
is refused: it stops a pull at its first progress check). ``timeout`` stops a
pull after prefetch_timeout_base + pages x prefetch_timeout_per_page (the
expiry of any rank stops it on every rank: MAX all-reduce); ``wait_complete``
waits until the operation ends (backend LOOKUP and read timeouts). A stopped
pull keeps the pages that landed. A pull that starts no operation (anchor not
backed up, span below the prefetch threshold, rate limit, staging allocation)
admits the request in the same pass.

LOOKUP-aware hold (F3, ``SGLANG_JIT_PULL_LOOKUP_AWARE_HOLD``, default off):
until the TP-synced hit is known, ``hash_value`` is empty, so the ``timeout``
bound is the bare base (1 s) while a large span's LOOKUP alone can take longer
(serial prefetch thread, hashing, the owner's pinning). With the flag the
LOOKUP phase is bounded by ``SGLANG_JIT_PULL_LOOKUP_PHASE_MAX_S`` and the
per-page clock starts when the hit is known
(``UnifiedRadixCache._prefetch_timeout_lookup_aware``). And when a pull issued
here ended before its LOOKUP reached the hit drain, admission holds the request
(no settle, no re-pull behind that LOOKUP) until the drain marks the operation,
for at most ``SGLANG_JIT_PULL_LOOKUP_HOLD_MAX_PASSES`` passes after the issue.

Owner touch (F1, ``SGLANG_PEER_KV_TOUCH_DEFERRED_PULLS``, default off;
``mem_cache/peer_kv_touch.py``): on attention-TP rank 0, a deferred pull sends
TOUCH to its owner at the deferral and every
``SGLANG_PEER_KV_TOUCH_INTERVAL_S`` while it stays deferred, so the owner's
host LRU keeps the hinted prefix until the pull. It stops at the issue, at the
request's end, or after ``SGLANG_PEER_KV_TOUCH_MAX_AGE_S``. Rank-local and
best effort: no decision reads it, and the wall clock paces the touches only.

TP lockstep: the decisions read only the broadcast hint, the radix match
(lockstep, as for every admission decision), per-request counters and flags
(changed at the same pass on every rank), ``ongoing_prefetch`` membership
(changed only at lockstep points) and, for (b1), queue lengths, queue order,
input-token counts and the chunked request's progress (all replicated; with
late D also each bootstrap room's late flag, set at arrival on every rank, and
its abort state). The
issue path is the arrival path, the hold is the consensus progress check, and
no collective is added. The wall clock feeds the logs only.

BWP: BWP may mark a deferred request in the bootstrap queue "examined for
good" (no host hit yet). When its deferred pull lands, its retry entry is
cleared (``CLEAR_BWP_RETRY_ON_LANDING``) so BWP can examine it again. While
BWP scans only the bootstrap queue, the entry is gone by then anyway (BWP drops
the entries of requests that left the queue); the clear matters once BWP also
covers the waiting queue.

Hooks for the next steps:
- (b3) late-D bind: the deferred pull of a late room now starts before its
  ADMIT (the scan above). An ``on_bound(req)`` hook could still run
  ``_maybe_repull`` when the decode binds the room.
- Pulled-page lock (stage 2): ``req.jit_pull_issued`` is set while a pull
  issued here is in flight. The landing hook in
  ``UnifiedRadixCache._handle_prefetch_result`` uses it to tell a re-pull or a
  deferred pull (S-REPULL) from an arrival pull (S-PULL) and locks the
  inserted host node.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import time
from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Optional

from decoupled_engine.srt.disaggregation.utils import DisaggregationMode, is_aborted
from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.mem_cache.peer_kv_touch import TouchSender, TouchSpan
from decoupled_engine.srt.runtime_context import get_memory

if TYPE_CHECKING:
    from decoupled_engine.srt.managers.schedule_batch import Req
    from decoupled_engine.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

REPULL = "repull"
JIT = "jit"
JIT_PULL_MODES = ("off", "queue")
SUMMARY_INTERVAL_S = 60.0
# Issue-to-settle samples kept per kind and summary window (logs only).
MAX_WAIT_SAMPLES = 4096
# (b1) Requests scanned past the lead each pass.
LEAD_SCAN_EXTRA = 4
# (b1) with BWP: clear BWP's retry entry of a request when
# its deferred pull lands, so BWP may examine (and pin) it again.
CLEAR_BWP_RETRY_ON_LANDING = True


@dataclasses.dataclass
class IssuedPull:
    """A pull issued here for a queued request. It stays in
    ``req.jit_pull_issued`` until admission sees the pull end."""

    kind: str
    attempt: int
    expected: int
    local: int
    t_issue: float  # time.monotonic(); logs only
    pass_issue: int
    # F3 (SGLANG_JIT_PULL_LOOKUP_AWARE_HOLD only): the prefetch operation, to
    # see whether its LOOKUP reached the hit drain; None with the flag off.
    op: object = None
    lookup_hold_passes: int = 0


@dataclasses.dataclass
class _TouchEntry:
    req: Req
    addr: str
    span: TouchSpan
    t_first: float  # time.monotonic()
    next_due: float


class _DeferredPullTouch:
    """F1, destination side (attention-TP rank 0 only): TOUCH the owner of each
    deferred pull at the deferral and every interval while the pull stays
    deferred (``mem_cache/peer_kv_touch.py``). Rank-local and best effort:
    nothing here feeds a scheduling decision; the wall clock paces the
    touches only. The sender thread hashes the span and sends. When the pull
    is issued or the request ends, the span is marked stopped, and the
    sender drops it if it still waits in the queue."""

    SWEEP_S = 1.0

    def __init__(self, jit: JitPull, sender=None, clock=time.monotonic):
        cache = jit.cache
        backend = getattr(
            getattr(cache, "cache_controller", None), "storage_backend", None
        )
        self._target = getattr(backend, "_target_addr", None)
        self.page_size = int(cache.page_size)
        self.is_bigram = bool(
            getattr(getattr(cache, "tree_core", None), "is_eagle", False)
        )
        self.interval_s = float(envs.SGLANG_PEER_KV_TOUCH_INTERVAL_S.get())
        self.max_age_s = float(envs.SGLANG_PEER_KV_TOUCH_MAX_AGE_S.get())
        self.max_keys = int(envs.SGLANG_PEER_KV_TOUCH_MAX_KEYS.get())
        self.sender = sender if sender is not None else TouchSender()
        self._clock = clock
        self._pending: Dict[str, _TouchEntry] = {}
        self._next_sweep = 0.0
        self.counts: Counter = Counter()

    def _addr(self, kv_pull_hint) -> Optional[str]:
        """The owner servicer's address for this rank, as the LOOKUP uses it."""
        if not isinstance(kv_pull_hint, dict):
            return None
        if self._target is not None:
            try:
                return self._target(kv_pull_hint)
            except Exception:
                return None
        addr = kv_pull_hint.get("source_servicer_addr")
        if not isinstance(addr, str):
            return None
        _, sep, port = addr.rpartition(":")
        return addr if sep and port.isdigit() else None

    def register(self, req: Req, expected: int) -> None:
        """At the deferral: the first TOUCH now, the next one after the
        interval."""
        addr = self._addr(req.kv_pull_hint)
        num_pages = int(expected) // self.page_size
        if addr is None or num_pages <= 0:
            self.counts["skipped"] += 1
            return
        span = TouchSpan(
            req.origin_input_ids,
            num_pages,
            page_size=self.page_size,
            is_bigram=self.is_bigram,
            extra_key=getattr(req, "extra_key", None),
            cache_salt=getattr(req, "cache_salt", None),
            max_keys=self.max_keys,
            rid=req.rid,
        )
        now = self._clock()
        self._pending[req.rid] = _TouchEntry(
            req, addr, span, now, now + self.interval_s
        )
        self.counts["registered"] += 1
        self._send(addr, span)

    def stop(self, rid: str, reason: str) -> None:
        entry = self._pending.pop(rid, None)
        if entry is not None:
            entry.span.stopped = True
            self.counts["stop_" + reason] += 1

    def sweep(self) -> None:
        """Each pass (at most every SWEEP_S): drop the entries whose pull is
        no longer deferred, TOUCH again the ones that are due."""
        if not self._pending:
            return
        now = self._clock()
        if now < self._next_sweep:
            return
        self._next_sweep = now + self.SWEEP_S
        for rid, entry in list(self._pending.items()):
            reason = self._ended(entry.req, now - entry.t_first)
            if reason is not None:
                del self._pending[rid]
                entry.span.stopped = True
                self.counts["stop_" + reason] += 1
            elif now >= entry.next_due:
                entry.next_due = now + self.interval_s
                self.counts["retouches"] += 1
                self._send(entry.addr, entry.span)

    def _ended(self, req: Req, age: float) -> Optional[str]:
        if not req.jit_pull_pending:
            return "issued"
        if is_aborted(req) or req.finished():
            return "ended"
        if age > self.max_age_s:
            return "age"
        return None

    def _send(self, addr: str, span: TouchSpan) -> None:
        self.counts["touches"] += 1
        self.sender.submit(addr, span)

    def line(self) -> str:
        counts = " ".join(f"touch_{k}={v}" for k, v in sorted(self.counts.items()))
        sender = " ".join("touch_" + kv for kv in self.sender.line().split())
        return f"{counts} touch_pending={len(self._pending)} {sender}".strip()


def create_jit_pull(scheduler: Scheduler) -> Optional[JitPull]:
    """Build the component for ``Scheduler.init_disaggregation``.

    None when both ``--hicache-storage-repull-on-admission`` and
    ``--hicache-storage-jit-pull`` are off (the scheduler is not read), and on
    a decode worker (P and D may share one launch config, so there the flags
    are logged as inactive, not refused). Every other unsupported setup is
    refused at startup."""
    repull = get_memory().hicache_storage_repull_on_admission
    jit_mode = get_memory().hicache_storage_jit_pull
    if not repull and jit_mode == "off":
        return None
    flags = " and ".join(
        ["--hicache-storage-repull-on-admission"] * bool(repull)
        + [f"--hicache-storage-jit-pull {jit_mode}"] * (jit_mode != "off")
    )
    if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
        logger.info(
            "%s: inactive on a decode worker (it applies to prefill only).", flags
        )
        return None
    problems = _unsupported(scheduler, repull, jit_mode)
    if problems:
        raise ValueError(f"{flags} requires " + ", ".join(problems))
    return JitPull(
        scheduler,
        min_tokens=get_memory().hicache_storage_repull_min_tokens,
        max_attempts=get_memory().hicache_storage_repull_max_attempts,
        repull=repull,
        jit_mode=jit_mode,
        now_reqs=get_memory().hicache_storage_jit_pull_now_reqs,
        now_tokens=get_memory().hicache_storage_jit_pull_now_tokens,
        lead_reqs=get_memory().hicache_storage_jit_pull_lead_reqs,
        lead_tokens=get_memory().hicache_storage_jit_pull_lead_tokens,
    )


def _unsupported(scheduler: Scheduler, repull: bool, jit_mode: str) -> List[str]:
    from decoupled_engine.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    problems = []
    if scheduler.disaggregation_mode != DisaggregationMode.PREFILL:
        problems.append("--disaggregation-mode prefill")
    if not scheduler.enable_hierarchical_cache:
        problems.append("--enable-hierarchical-cache")
    cache = scheduler.tree_cache
    if not isinstance(cache, UnifiedRadixCache):
        problems.append("the unified radix cache")
    if get_memory().hicache_host_memory_mode != "cache":
        problems.append("--hicache-host-memory-mode cache")
    backend = getattr(getattr(cache, "cache_controller", None), "storage_backend", None)
    if (
        not scheduler.enable_hicache_storage
        or getattr(backend, "requires_kv_pull_hint", False) is not True
    ):
        problems.append("a hint-only HiCache storage backend (peer_kv)")
    if get_memory().hicache_storage_prefetch_policy == "best_effort":
        problems.append(
            "--hicache-storage-prefetch-policy timeout or wait_complete "
            "(best_effort stops a pull at its first progress check)"
        )
    if scheduler.is_hybrid_ssm:
        # Admission skips the request after a match that may stage a mamba
        # copy-on-write slot; only a rejected add_one_req reverts that.
        problems.append("a non-mamba model")
    if scheduler.ps.pp_size != 1:
        problems.append("pp_size == 1")
    if scheduler.ps.attn_cp_size != 1:
        problems.append("attn_cp_size == 1")
    if scheduler.enable_dp_attention:
        problems.append("no DP attention")
    if repull and get_memory().hicache_storage_repull_min_tokens < 1:
        problems.append("--hicache-storage-repull-min-tokens >= 1")
    if repull and get_memory().hicache_storage_repull_max_attempts < 1:
        problems.append("--hicache-storage-repull-max-attempts >= 1")
    if jit_mode not in JIT_PULL_MODES:
        problems.append("--hicache-storage-jit-pull off or queue")
    if jit_mode != "off":
        if get_memory().hicache_storage_jit_pull_now_reqs < 0:
            problems.append("--hicache-storage-jit-pull-now-reqs >= 0")
        if get_memory().hicache_storage_jit_pull_now_tokens < 0:
            problems.append("--hicache-storage-jit-pull-now-tokens >= 0")
        if get_memory().hicache_storage_jit_pull_lead_reqs < 0:
            problems.append("--hicache-storage-jit-pull-lead-reqs >= 0")
        if get_memory().hicache_storage_jit_pull_lead_tokens < 0:
            problems.append("--hicache-storage-jit-pull-lead-tokens >= 0")
    return problems


def _percentile(samples: List[float], q: float) -> str:
    if not samples:
        return "-"
    ordered = sorted(samples)
    return "%.0f" % ordered[max(0, math.ceil(q * len(ordered)) - 1)]


@dataclasses.dataclass(frozen=True)
class PullLeadConfig:
    """Replicated constants; wall clocks must not select a pull."""
    host_tokens_per_pass: int = 131072
    host_margin: int = 1
    fixed_seconds: float = 0.78
    seconds_per_100k: float = 0.22
    nominal_pass_seconds: float = 0.38
    issue_passes: int = 1
    margin: int = 1

    @classmethod
    def from_env(cls):
        c = cls(
            int(os.getenv("SGLANG_BWP_LEAD_TOKENS_PER_PASS", "131072")),
            int(os.getenv("SGLANG_BWP_LEAD_MARGIN", "1")),
            float(os.getenv("SGLANG_JIT_PULL_SPAN_FIXED_S", "0.78")),
            float(os.getenv("SGLANG_JIT_PULL_SPAN_S_PER_100K", "0.22")),
            float(os.getenv("SGLANG_JIT_PULL_NOMINAL_PASS_S", "0.38")),
            int(os.getenv("SGLANG_JIT_PULL_ISSUE_PASSES", "1")),
            int(os.getenv("SGLANG_JIT_PULL_LEAD_MARGIN", "1")),
        )
        if c.host_tokens_per_pass <= 0 or c.nominal_pass_seconds <= 0 or min(
            c.host_margin, c.fixed_seconds, c.seconds_per_100k, c.issue_passes, c.margin
        ) < 0:
            raise ValueError("JIT pull plan constants must be nonnegative with positive token/pass rates")
        return c

    def passes(self, host_demand: int, pull_tokens: int, io_ahead_tokens: int, page_size: int) -> int:
        host = 1 + self.host_margin + math.ceil(max(0, host_demand) / self.host_tokens_per_pass)
        # The serial IO thread copies about 128 pages per 10 ms. Do not sum
        # each earlier pull's fixed/hold time: those phases overlap.
        io_seconds = max(0, io_ahead_tokens) * 0.010 / (128 * max(1, page_size))
        span = self.fixed_seconds + self.seconds_per_100k * max(0, pull_tokens) / 100000
        return host + self.issue_passes + math.ceil((span + io_seconds) / self.nominal_pass_seconds) + self.margin


class JitPull:
    """Per-scheduler state of the just-in-time pull (stage 1: (b2) and (b1))."""

    def __init__(
        self,
        scheduler: Scheduler,
        min_tokens: int,
        max_attempts: int,
        *,
        repull: bool = True,
        jit_mode: str = "off",
        now_reqs: int = 8,
        now_tokens: int = 1_100_000,
        lead_reqs: int = 4,
        lead_tokens: int = 131_072,
    ):
        self.scheduler = scheduler
        self.cache = scheduler.tree_cache
        self.min_tokens = int(min_tokens)
        self.max_attempts = int(max_attempts)
        self.repull = bool(repull)
        self.jit_mode = jit_mode
        self.now_reqs = int(now_reqs)
        self.now_tokens = int(now_tokens)
        self.lead_reqs = int(lead_reqs)
        self.lead_tokens = int(lead_tokens)
        self.lead_mode = os.getenv("SGLANG_JIT_PULL_LEAD", "count").strip().lower()
        if self.lead_mode not in ("count", "plan"):
            raise ValueError("SGLANG_JIT_PULL_LEAD must be count or plan")
        self.plan_lead = PullLeadConfig.from_env() if self.lead_mode == "plan" else None
        if self.lead_mode == "plan" and not envs.SGLANG_JIT_PULL_LOOKUP_AWARE_HOLD.get():
            raise ValueError("SGLANG_JIT_PULL_LEAD=plan requires SGLANG_JIT_PULL_LOOKUP_AWARE_HOLD=1")
        # Cumulative counters, per kind (repull, jit): <kind>_issued /
        # _not_issued / _settled, <kind>_deficit_tokens (expected - local at
        # issue), <kind>_gain_tokens (local match gained by the pull) and
        # <kind>_short_tokens (expected - local after it: owner gone or
        # evicted again). (b1) adds jit_deferred(_tokens), jit_landed (ended
        # before the admission visit) and jit_issued_at_admission.
        self.stats: Counter = Counter()
        self._wait_ms: Dict[str, List[float]] = {}
        self._lead_passes: Dict[str, List[int]] = {}
        # (b1) Deferred pulls in flight, in issue order (rid -> request).
        self._jit_inflight: Dict[str, Req] = {}
        self._pass = 0
        self._log = scheduler.ps.attn_tp_rank == 0
        self._last_summary = time.monotonic()
        # F1 (SGLANG_PEER_KV_TOUCH_DEFERRED_PULLS): attention-TP rank 0 only,
        # queue mode only; None otherwise.
        self._touch: Optional[_DeferredPullTouch] = None
        if (
            self._log
            and self.jit_mode == "queue"
            and envs.SGLANG_PEER_KV_TOUCH_DEFERRED_PULLS.get()
        ):
            self._touch = _DeferredPullTouch(self)
            logger.info(
                "JIT pull owner touch on: interval_s=%.1f max_age_s=%.1f "
                "max_keys=%d",
                self._touch.interval_s,
                self._touch.max_age_s,
                self._touch.max_keys,
            )
        if self._log:
            logger.info(
                "JIT pull enabled: repull_on_admission=%d min_tokens=%d "
                "max_attempts=%d jit_pull=%s now_reqs=%d now_tokens=%d "
                "lead_reqs=%d lead_tokens=%d",
                int(self.repull),
                self.min_tokens,
                self.max_attempts,
                self.jit_mode,
                self.now_reqs,
                self.now_tokens,
                self.lead_reqs,
                self.lead_tokens,
            )

    # ---------------------------------------------------------- (b1) arrival
    def defer_at_arrival(self, req: Req) -> bool:
        """(b1) Called by the PD prefill arrival in place of the arrival pull.
        True: the pull of this hinted request waits until it nears the head
        of the waiting queue (``req.jit_pull_pending``)."""
        if self.jit_mode != "queue" or req.kv_pull_hint is None:
            return False
        s = self.scheduler
        if self.lead_mode == "plan":
            return self._defer_plan_arrival(req)
        queued = s.disagg_prefill_bootstrap_queue.queue
        n_ahead = len(queued) + len(s.waiting_queue)
        t_ahead = self._chunk_remainder()
        t_ahead += sum(len(r.origin_input_ids) for r in queued)
        t_ahead += sum(len(r.origin_input_ids) for r in s.waiting_queue)
        if n_ahead <= self.now_reqs and t_ahead <= self.now_tokens:
            return False
        req.jit_pull_pending = True
        expected = self._expected(req, len(req.origin_input_ids)) or 0
        self.stats[JIT + "_deferred"] += 1
        self.stats[JIT + "_deferred_tokens"] += expected
        if self._touch is not None:
            self._touch.register(req, expected)
        if self._log:
            logger.info(
                "JIT pull deferred: rid=%s n_ahead=%d t_ahead=%d expected=%d",
                req.rid,
                n_ahead,
                t_ahead,
                expected,
            )
        return True

    # ------------------------------------------------------- (b1) each pass
    def issue_lead_pulls(self, waiting_queue: List[Req]) -> None:
        """(b1) Called once per pass after the HiCache events and before
        admission (it also keeps the pass count that the logs use). The scan
        covers the first lead-reqs + ``LEAD_SCAN_EXTRA`` requests in
        admission order: the waiting queue and, with late D on, then the
        late rooms that wait for their bind (``_issue_unbound_lead_pulls``)."""
        self._pass += 1
        if self._jit_inflight:
            self._observe_landings()
        if self.jit_mode != "queue":
            return
        if self.lead_mode == "plan":
            self._issue_plan_pulls(waiting_queue)
            if self._touch is not None:
                self._touch.sweep()
            return
        ahead = self._chunk_remainder()
        limit = self.lead_reqs + LEAD_SCAN_EXTRA
        window = waiting_queue[:limit]
        for position, req in enumerate(window):
            if req.jit_pull_pending and (
                position <= self.lead_reqs or ahead <= self.lead_tokens
            ):
                self._issue_jit(req, f"position={position} ahead_tokens={ahead}")
            ahead += len(req.origin_input_ids)
        if len(window) < limit:
            # The whole waiting queue is in the window, so ``ahead`` counts
            # all of it: the unbound late rooms queue behind it.
            self._issue_unbound_lead_pulls(len(window), ahead, limit)
        if self._touch is not None:
            # After the lead scan: a pull issued in this pass sends no re-touch.
            self._touch.sweep()

    def _io_ahead_tokens(self) -> int:
        """Stable outstanding demand, never loader-thread progress or time."""
        total = 0
        for rid, info in self.cache.ongoing_prefetch.items():
            req = self._jit_inflight.get(rid)
            issued = getattr(req, "jit_pull_issued", None)
            if issued is not None:
                total += max(0, issued.expected - issued.local)
            else:
                op = getattr(info, "operation", info)
                # Arrival pulls have no IssuedPull. Their immutable request
                # span is a conservative bound until the replicated landing.
                total += len(getattr(op, "token_ids", ()))
        return total

    def _lead_for_entry(self, entry) -> int:
        return self.plan_lead.passes(entry.host_demand, entry.pull_tokens,
                                     self._io_ahead_tokens(), self.scheduler.page_size)

    def _defer_plan_arrival(self, req) -> bool:
        adapter = getattr(self.scheduler, "late_d_lookahead", None)
        expected = self._expected(req, len(req.origin_input_ids)) or 0
        if expected <= 0:
            return False
        if adapter is not None:
            c = adapter.candidate(req, ignore_intent=True)
            pull = max(0, expected - c.device_tokens - c.host_tokens)
            candidate = dataclasses.replace(c, uncached_tokens=max(1, c.uncached_tokens - pull),
                                            pull_tokens=pull, skip_reason=None, ready_pass=1)
            entry = adapter.plan.tail_candidate(candidate)
            if entry is not None and entry.start_pass is not None and entry.start_pass <= self._lead_for_entry(entry):
                self.stats["plan_arrival_inside_lead"] += 1
                return False
        req.jit_pull_pending = True
        self.stats[JIT + "_deferred"] += 1
        self.stats[JIT + "_deferred_tokens"] += expected
        self.stats["plan_arrival_deferred"] += 1
        if self._touch is not None:
            self._touch.register(req, expected)
        if self._log:
            logger.info("JIT pull deferred: rid=%s plan=1 expected=%d", req.rid, expected)
        return True

    def _issue_plan_pulls(self, waiting_queue):
        """Issue in forecast order. The bootstrap queue is never inspected."""
        plan = getattr(getattr(self.scheduler, "late_d_lookahead", None), "plan", None)
        if plan is None:
            self.stats["plan_missing"] += 1
            return
        queued = {r.rid: r for r in waiting_queue}
        for rid in plan.order:
            req = queued.get(rid)
            if req is None or not req.jit_pull_pending or is_aborted(req) or req.finished():
                continue
            entry, start, lead = self._planned_deadline(req)
            if start is not None and start <= lead:
                self.stats["plan_due"] += 1
                self._issue_jit(req, f"plan=1 start={start} lead={lead} host_demand={entry.host_demand} pull={entry.pull_tokens}")
            else:
                self.stats["plan_not_due"] += 1

    def _planned_deadline(self, req):
        """Read one replicated forecast; an absent row is never implicitly due.

        In particular, tail_candidate cannot be used for an arbitrary omitted
        row: it would omit newer waiting requests ahead of that row. The next
        fixed-point update includes fresh arrivals; bounded-plan overflow has
        the separate idle admission escape below.
        """
        plan = getattr(getattr(self.scheduler, "late_d_lookahead", None), "plan", None)
        entry = plan.entries.get(req.rid) if plan is not None else None
        if entry is None:
            return None, None, None
        start = entry.start_pass
        if start is None and entry.gated:
            start = entry.gate_open_start_pass
        return entry, start, self._lead_for_entry(entry)

    def _idle_admission_allowed(self, req, adder, reserve_tokens):
        """Narrow progress escape, using replicated state and pure budgets.

        Admission already checked LoRA, request slots and completed prefetches.
        No current compute may be displaced. Only the ordinary full-attention
        path is covered here; specialized memory/delayer paths wait for the
        normal plan. At most one idle issue is allowed per scheduler pass.
        """
        s = self.scheduler
        if adder is None or getattr(self, "_idle_issue_pass", -1) == self._pass:
            return False
        if s.chunked_req is not None or adder.can_run_list or getattr(s, "result_queue", ()):
            return False
        if any(getattr(getattr(s, name, None), "reqs", ())
               for name in ("last_batch", "running_batch")):
            return False
        if (getattr(adder, "is_hybrid_swa", False)
                or getattr(adder, "_mamba_slot_cost", 0)
                or getattr(adder, "rem_mamba_slots", None) is not None
                or getattr(adder, "dllm_config", None) is not None
                or getattr(adder, "prefill_delayer_single_pass", None) is not None):
            return False
        page = s.page_size
        chunk = adder.rem_chunk_tokens
        if chunk is None or chunk < page or adder.rem_input_tokens < page:
            return False
        from decoupled_engine.srt.disaggregation.late_d_need import evictable_prefix_tokens
        from decoupled_engine.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS, PrefillAdder

        prefix = len(req.prefix_indices)
        need = len(req.full_untruncated_fill_ids) - prefix
        output = min(max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
                     CLIP_MAX_NEW_TOKENS)
        available = adder.rem_total_tokens - evictable_prefix_tokens(req)
        if need + output + page >= available - reserve_tokens:
            return False
        # This existing eligibility hook updates replicated skip counters and
        # gate flags. It is not a pure read; it makes no allocation or IO call.
        if getattr(s, "_late_d_need_enabled", False) and s.late_d_need_skip(req, adder):
            return False
        # With no running partial, use the ordinary chunked-admission plan.
        # This check is read-only and does not allocate or start a request.
        compute = max(1, need - getattr(req, "host_hit_length", 0))
        reject, _ = PrefillAdder._plan_extend(
            adder, adder.ceil_paged_tokens(compute), prefix + getattr(req, "host_hit_length", 0),
            chunk, s.truncation_align_size)
        return reject is None

    def _hold_planned_pull(self, req, adder, reserve_tokens):
        entry, start, lead = self._planned_deadline(req)
        if start is not None and start <= lead:
            self.stats["plan_due_at_admission"] += 1
            return self._issue_jit(req, f"plan=1 start={start} lead={lead} "
                                   f"host_demand={entry.host_demand} pull={entry.pull_tokens} admission=1")
        # A known future deadline remains authoritative even at an idle visit.
        # Escape is for absent/unforecastable rows only, including >max_reqs.
        if start is None and self._idle_admission_allowed(req, adder, reserve_tokens):
            self._idle_issue_pass = self._pass
            self.stats["plan_idle_fallback"] += 1
            return self._issue_jit(req, "plan=1 idle_fallback=1 reason=no_forecast_idle")
        self.stats["plan_admission_deferred"] += 1
        if self._log and not getattr(req, "jit_plan_defer_logged", False):
            req.jit_plan_defer_logged = True
            logger.info("JIT pull admission deferred: rid=%s plan=1 start=%s lead=%s reason=%s",
                        req.rid, start, lead, "not_due" if start is not None else "no_forecast")
        # True skips admission while keeping the pull pending. False would
        # admit a cold recompute and silently discard the planned transfer.
        return True

    def _issue_unbound_lead_pulls(self, position: int, ahead: int, limit: int) -> None:
        """(b1) with late D (``--disaggregation-late-d``); no-op when it is
        off. A late room stays in the bootstrap queue until its decode binds
        it, and the binder sends the ADMIT only when the room is a few
        iterations from the head. A scan of the waiting queue alone issues
        the room's pull after the bind, too late to land before admission.
        So the late rooms that wait for their bind count as queued behind the
        waiting queue, in bootstrap-queue order (the binder's ADMIT order),
        and a pending one within the lead is pulled before its ADMIT. Rooms
        that bind as today (FAKE) and aborted rooms are skipped, as the
        binder skips them. ``position`` and ``ahead``: the requests and input
        tokens queued ahead of the first unbound room."""
        boot = self.scheduler.disagg_prefill_bootstrap_queue
        if not getattr(boot, "late_d_enabled", False):
            return
        for req in boot.queue:
            if position >= limit:
                return
            if not getattr(req.disagg_kv_sender, "late_d", False) or is_aborted(req):
                continue
            if req.jit_pull_pending and (
                position <= self.lead_reqs or ahead <= self.lead_tokens
            ):
                self._issue_jit(
                    req, f"unbound=1 position={position} ahead_tokens={ahead}"
                )
            ahead += len(req.origin_input_ids)
            position += 1

    # ------------------------------------------------------- admission hook
    def hold_for_repull(self, req: Req, *, adder=None, reserve_tokens=0) -> bool:
        """Called by admission after ``init_next_round_input`` and the
        bootstrap-prefetch deferral, so only when no prefetch of ``req`` is in
        flight. Settles the pull issued here before, if any.

        True: a pull of the missing prefix of ``req`` was just issued: (b1) its
        deferred pull, which admission reached first, or (b2) a re-pull. The
        caller skips ``req`` in this pass and admits the requests behind it;
        ``check_prefetch_progress`` holds ``req`` until the pull ends."""
        issued = req.jit_pull_issued
        if issued is not None:
            if issued.op is not None and self._lookup_in_flight(issued):
                # F3: the pull ended before its LOOKUP reached the hit drain
                # (it was stopped while the owner was still answering). A
                # settle now would re-pull behind that LOOKUP (the prefetch
                # thread is serial) or admit a recompute: hold the request.
                issued.lookup_hold_passes += 1
                self.stats["lookup_hold_passes"] += 1
                return True
            req.jit_pull_issued = None
            self._settle(req, issued)
        if req.jit_pull_pending:
            if self.lead_mode == "plan":
                held = self._hold_planned_pull(req, adder, reserve_tokens)
            else:
                held = self._issue_jit(req, "at_admission=1")
        else:
            held = self.repull and self._maybe_repull(req)
        if self._log and time.monotonic() - self._last_summary >= SUMMARY_INTERVAL_S:
            self._log_summary()
        return held

    def _lookup_in_flight(self, issued: IssuedPull) -> bool:
        """F3: True while the issued pull's LOOKUP + TP sync has not reached
        the hit drain, for at most LOOKUP_HOLD_MAX_PASSES passes after the
        issue. Lockstep without a collective: the drain marks the op at the
        same pass on every rank (MIN-reduced drain count), and the pass count
        is replicated."""
        if getattr(issued.op, "hit_drained", False):
            return False
        limit = envs.SGLANG_JIT_PULL_LOOKUP_HOLD_MAX_PASSES.get()
        if self._pass - issued.pass_issue < limit:
            return True
        self.stats["lookup_hold_expired"] += 1
        return False

    def threshold(self, kv_pull_hint: dict) -> int:
        """Smallest deficit that triggers a re-pull for this hint."""
        minimum = kv_pull_hint.get("min_local_deficit_tokens")
        if type(minimum) is not int or minimum < 1:
            minimum = 1
        return max(minimum, self.min_tokens)

    # ------------------------------------------------------------ internals
    @staticmethod
    def _expected(req: Req, input_len: int) -> Optional[int]:
        """The hint's ``expected_hit_tokens`` (a match never covers the last
        input token), or None when the request has no usable hint."""
        kv_pull_hint = req.kv_pull_hint
        if not isinstance(kv_pull_hint, dict):
            return None
        expected = kv_pull_hint.get("expected_hit_tokens")
        if type(expected) is not int or expected <= 0:
            return None
        return min(expected, req._compute_max_prefix_len(input_len))

    def _chunk_remainder(self) -> int:
        chunked = self.scheduler.chunked_req
        if chunked is None:
            return 0
        return max(0, len(chunked.origin_input_ids) - len(chunked.prefix_indices))

    def _maybe_repull(self, req: Req) -> bool:
        expected = self._expected(req, len(req.full_untruncated_fill_ids))
        if expected is None:
            return False
        local = len(req.prefix_indices) + req.host_hit_length
        if expected - local < self.threshold(req.kv_pull_hint):
            return False
        if req.qp_repull_attempts >= self.max_attempts:
            return False
        if req.rid in self.cache.ongoing_prefetch:
            return False
        req.qp_repull_attempts += 1
        return self._issue(req, REPULL, req.qp_repull_attempts)

    def _issue_jit(self, req: Req, note: str) -> bool:
        req.jit_pull_pending = False
        if self._touch is not None:
            self._touch.stop(req.rid, "issued")
        if not self._issue(req, JIT, 0, note):
            return False
        self._jit_inflight[req.rid] = req
        if note.startswith("at_admission"):
            self.stats[JIT + "_issued_at_admission"] += 1
        return True

    def _issue(self, req: Req, kind: str, attempt: int, note: str = "") -> bool:
        """Pull the missing prefix of ``req`` through the arrival path. True
        when a prefetch operation started."""
        self.scheduler._prefetch_kvcache(req)
        expected = self._expected(req, len(req.full_untruncated_fill_ids)) or 0
        local = len(req.prefix_indices) + req.host_hit_length
        if req.rid not in self.cache.ongoing_prefetch:
            # Nothing started: admit now. A declined prefetch leaves a miss
            # marker; drop it so the paced retry does not re-issue it.
            self.cache.pop_storage_prefetch_miss(req.rid)
            self.stats[kind + "_not_issued"] += 1
            if self._log:
                logger.info(
                    "JIT pull not issued: kind=%s rid=%s attempt=%d expected=%d "
                    "local=%d deficit=%d %s",
                    kind,
                    req.rid,
                    attempt,
                    expected,
                    local,
                    expected - local,
                    note,
                )
            return False
        op = None
        if envs.SGLANG_JIT_PULL_LOOKUP_AWARE_HOLD.get():
            info = self.cache.ongoing_prefetch[req.rid]
            op = getattr(info, "operation", info)
        req.jit_pull_issued = IssuedPull(
            kind, attempt, expected, local, time.monotonic(), self._pass, op
        )
        self.stats[kind + "_issued"] += 1
        self.stats[kind + "_deficit_tokens"] += max(0, expected - local)
        if self._log:
            logger.info(
                "JIT pull issued: kind=%s rid=%s attempt=%d expected=%d local=%d "
                "device=%d host=%d deficit=%d %s",
                kind,
                req.rid,
                attempt,
                expected,
                local,
                len(req.prefix_indices),
                req.host_hit_length,
                expected - local,
                note,
            )
        return True

    def _observe_landings(self) -> None:
        """(b1) Deferred pulls that ended (landed, or stopped) before their
        request's admission visit."""
        ongoing = self.cache.ongoing_prefetch
        bwp = getattr(self.scheduler, "bootstrap_prefetch", None)
        for rid in [rid for rid in self._jit_inflight if rid not in ongoing]:
            del self._jit_inflight[rid]
            self.stats[JIT + "_landed"] += 1
            if CLEAR_BWP_RETRY_ON_LANDING and bwp is not None:
                bwp.retry_after_landing(rid)

    def _settle(self, req: Req, issued: IssuedPull) -> None:
        local = len(req.prefix_indices) + req.host_hit_length
        gain = max(0, local - issued.local)
        short = max(0, issued.expected - local)
        wait_ms = (time.monotonic() - issued.t_issue) * 1e3
        lead_passes = self._pass - issued.pass_issue
        kind = issued.kind
        self._jit_inflight.pop(req.rid, None)
        self.stats[kind + "_settled"] += 1
        self.stats[kind + "_gain_tokens"] += gain
        self.stats[kind + "_short_tokens"] += short
        samples = self._wait_ms.setdefault(kind, [])
        if len(samples) < MAX_WAIT_SAMPLES:
            samples.append(wait_ms)
            self._lead_passes.setdefault(kind, []).append(lead_passes)
        if self._log:
            logger.info(
                "JIT pull settled: kind=%s rid=%s attempt=%d expected=%d "
                "local_before=%d local_after=%d gain=%d short=%d wait_ms=%.0f "
                "passes=%d%s",
                kind,
                req.rid,
                issued.attempt,
                issued.expected,
                issued.local,
                local,
                gain,
                short,
                wait_ms,
                lead_passes,
                (
                    ""
                    if issued.op is None
                    else " lookup_hold_passes=%d hit_drained=%d hit_tokens=%d"
                    % (
                        issued.lookup_hold_passes,
                        int(bool(getattr(issued.op, "hit_drained", False))),
                        int(getattr(issued.op, "storage_hit_count", 0) or 0),
                    )
                ),
            )

    def _log_summary(self) -> None:
        self._last_summary = time.monotonic()
        windows = []
        for kind, samples in sorted(self._wait_ms.items()):
            passes = self._lead_passes.get(kind, [])
            windows.append(
                f"{kind}_wait_ms_p50={_percentile(samples, 0.5)} "
                f"{kind}_wait_ms_p90={_percentile(samples, 0.9)} "
                f"{kind}_passes_p50={_percentile(passes, 0.5)} "
                f"{kind}_passes_p90={_percentile(passes, 0.9)} "
                f"{kind}_wait_n={len(samples)}"
            )
            samples.clear()
            passes.clear()
        logger.info(
            "JIT pull summary: %s %s%s",
            " ".join(f"{k}={v}" for k, v in sorted(self.stats.items())),
            " ".join(windows),
            "" if self._touch is None else " " + self._touch.line(),
        )
