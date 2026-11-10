"""Scheduler glue for `--disaggregation-mode symmetric` (symmetric P/D).

The event loop is the plain NULL loop; this mixin adds the
per-tick hooks the loop calls under SYMMETRIC only:

- `init_symmetric_disaggregation`: build metadata buffers, the planes, the
  ExportTable and the ImportQueue (no bootstrap server, no CommonKVManager).
- `symmetric_add_request`: route Import -> ImportQueue (pages allocated up
  front, waits outside the batch), Export -> waiting queue with
  `export_stop_after_first_token`, else Local.
- `symmetric_pre_schedule`: each tick, free released/expired exports, admit
  and PREBUILT ready imports, recompute failed/timed-out imports.
- `symmetric_on_prefill_chunk`: publish each finished prefill chunk of an
  Export request; after the last chunk + first token publish FINAL and keep
  the pages (EXPORTED) instead of freeing them.

Late D (`--disaggregation-late-d alloc`, late_d_sym.py):
P-legs are held until their reader allocated the import after the owner's
ADMIT; D-legs wait DEFERRED for that ADMIT. The owner's binder runs on rank 0
at the end of `symmetric_pre_schedule`; reader controls ride the request
broadcast (`symmetric_pull_late_d_controls`, `symmetric_handle_late_d_control`).

Everything here is a no-op for null / prefill / decode modes.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import time
from array import array
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from decoupled_engine.srt.disaggregation.late_d import (
    ADMITTED,
    CONTROL_LOCAL,
    DEFERRED,
    HELD,
    LATE_D_CHOOSE,
    LATE_D_OFF,
    LATE_D_OUTCOME_LOCAL,
    LATE_D_READER_GONE_ERR,
    insert_by_arrival,
)
from decoupled_engine.srt.disaggregation.symmetric import consensus as _consensus
from decoupled_engine.srt.disaggregation.symmetric.export_table import (
    COMPUTE_ABORT,
    LATE_BIND_TIMEOUT,
    LATE_READER_GONE,
    READER_BEGUN,
    RELEASED,
    RELEASED_BEFORE_EXPORT,
    RELEASED_WHILE_HELD,
    ExportTable,
    owner_swa_valid_floor,
    window_export_spill_range,
    window_export_swa_floor,
)
from decoupled_engine.srt.disaggregation.symmetric.import_queue import ImportQueue, ceil_align
from decoupled_engine.srt.disaggregation.symmetric.marks import MarkError, parse_kv_marks
from decoupled_engine.srt.disaggregation.utils import (
    DisaggregationMode,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    prepare_abort,
)
from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from decoupled_engine.srt.managers.scheduler_components.phase_timers import (
    phase,
    sync_debug_site,
    timed_phase,
)
from decoupled_engine.srt.mem_cache.common import maybe_cache_unfinished_req, release_kv_cache
from decoupled_engine.srt.mem_cache.cpu_page_book import cpu_rows_of, tag_row_slice
from decoupled_engine.srt.mem_cache.unified_cache.session_ref_tracker import (
    forget_kv_session_request,
)
from decoupled_engine.srt.runtime_context import get_disagg, get_schedule

if TYPE_CHECKING:
    from decoupled_engine.srt.managers.io_struct import AbortReq, TokenizedGenerateReqInput
    from decoupled_engine.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


def _resolved(server_args, field):
    """The resolved value of ``field`` (declarations included), not the raw input."""
    from decoupled_engine.srt.arg_groups.overrides import resolution_result

    return resolution_result(server_args, field)


def _is_swa_window(tree_cache) -> bool:
    """DSv4 uniform lifecycle with window SWA residency."""
    return getattr(tree_cache, "dsv4_uniform_swa_residency", "full") == "window"


def _release_insert_chunked(sched: Scheduler) -> bool:
    """SGLANG_SYMMETRIC_RELEASE_INSERT_CHUNKED: the chunk-boundary stash of an
    exporting prompt records its chunk ends (req.release_insert_chunk_ends),
    and its release insert (cache_finished_req) ends a tree node at each one,
    as a standard P's chunk inserts do. Needs a FULL-only unified tree and
    chunked prefill; otherwise the flag stays inactive."""
    if not envs.SGLANG_SYMMETRIC_RELEASE_INSERT_CHUNKED.get():
        return False
    from decoupled_engine.srt.mem_cache.unified_cache.components.tree_component import (
        ComponentType,
    )
    from decoupled_engine.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    tree, chunk = sched.tree_cache, sched.chunked_prefill_size
    active = (
        isinstance(tree, UnifiedRadixCache)
        and tuple(tree.tree_components) == (ComponentType.FULL,)
        and chunk is not None
    )
    if sched.ps.tp_rank == 0:
        if active:
            bigram = bool(tree.tree_core.is_eagle)
            logger.info(
                "symmetric release insert chunked: chunk=%d page=%d bigram_keys=%s "
                "(a node ends at each recorded chunk end e, key position "
                "page_floor(e%s))",
                chunk,
                tree.page_size,
                bigram,
                " - 1" if bigram else "",
            )
        else:
            logger.warning(
                "symmetric release insert chunked: inactive (needs a FULL-only "
                "unified tree and chunked prefill; tree %s, chunk %s)",
                type(tree).__name__,
                chunk,
            )
    return active


def await_reader_timeout_s() -> Optional[float]:
    """F-C (SGLANG_SYMMETRIC_PD_EXPORT_AWAIT_READER): None when off, else how long a
    held P-leg waits for its reader's BEGIN before it is queued anyway
    (SGLANG_SYMMETRIC_PD_EXPORT_AWAIT_READER_TIMEOUT; 0 = the import waiting timeout,
    after which the reader gives up on a pending import and aborts it)."""
    if not envs.SGLANG_SYMMETRIC_PD_EXPORT_AWAIT_READER.get():
        return None
    timeout = float(envs.SGLANG_SYMMETRIC_PD_EXPORT_AWAIT_READER_TIMEOUT.get())
    if timeout <= 0:
        timeout = float(envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get())
    return timeout


def _late_cfg(sched: Scheduler):
    """The symmetric late-D config, or None when late D is off (also for a
    scheduler stand-in without the attribute)."""
    return getattr(sched, "_sym_late_cfg", None)


# SGLANG_SYMMETRIC_BUDGET_PRE_BIND_ONLY / SGLANG_SYMMETRIC_BUDGET_STANDARD_PD:
# seconds between two SYMMETRIC_PUBLISH_BUDGET lines (TP rank 0).
_PUBLISH_BUDGET_LOG_S = 60.0

# What each ExportTable.publish_budget_rule meters (the startup line).
_PUBLISH_BUDGET_RULE_TEXT = {
    "pre_bind": "the budget meters only KV computed before the bind",
    "standard_pd": (
        "the budget meters only the catch-up of an export launched before "
        "its bind, as standard P/D"
    ),
}


def _publish_budget_rule(table) -> Optional[str]:
    """ExportTable.publish_budget_rule, or None (also for a table
    stand-in)."""
    rule = getattr(table, "publish_budget_rule", None)
    return rule if isinstance(rule, str) else None


def _log_publish_budget(sched: Scheduler) -> None:
    """A publication budget rule (SGLANG_SYMMETRIC_BUDGET_PRE_BIND_ONLY or
    SGLANG_SYMMETRIC_BUDGET_STANDARD_PD): on TP rank 0, at most once per
    _PUBLISH_BUDGET_LOG_S, log the publication counters of the window
    (ExportTable.take_publish_budget_window): the KV published without the
    budget per pass and the largest chunk, i.e. the READ bursts that the
    readers see. Every rank publishes the same chunks, so rank 0 speaks for
    the worker. A window without publications is not logged.

    symmetric_pre_schedule calls this before the first publish_computed call
    of a pass, so a window holds whole passes: a pre-schedule call and the
    after-result calls that continue its pass."""
    rank = getattr(getattr(sched, "ps", None), "tp_rank", None)
    if rank is None:
        rank = getattr(sched, "tp_rank", 0)
    if rank != 0:
        return
    now = time.monotonic()
    last = getattr(sched, "_sym_publish_budget_log_t", None)
    if last is None:
        # First call on a scheduler stand-in: the window starts now.
        sched._sym_publish_budget_log_t = now
        return
    if now - last < _PUBLISH_BUDGET_LOG_S:
        return
    sched._sym_publish_budget_log_t = now
    w = sched.export_table.take_publish_budget_window()
    if not (w["chunks"] or w["catchup_waits"]):
        return
    logger.info(
        "SYMMETRIC_PUBLISH_BUDGET rule=%s window_s=%.1f passes=%d "
        "unbudgeted_passes=%d unbudgeted_tokens=%d max_pass_unbudgeted=%d "
        "chunks=%d max_chunk_tokens=%d catchup_tokens=%d catchup_waits=%d "
        "finals=%d total_unbudgeted_tokens=%d total_catchup_tokens=%d",
        _publish_budget_rule(sched.export_table),
        now - last,
        w["passes"],
        w["unbudgeted_passes"],
        w["unbudgeted_tokens"],
        w["max_pass_unbudgeted"],
        w["chunks"],
        w["max_chunk_tokens"],
        w["catchup_tokens"],
        w["catchup_waits"],
        w["finals"],
        w["total_unbudgeted_tokens"],
        w["total_catchup_tokens"],
    )


def _last_iter_type(sched: Scheduler) -> str:
    """The last iteration: idle (no batch), prefill (an extend batch, mixed
    included) or decode."""
    last = sched.last_batch
    if last is None:
        return "idle"
    mode = last.forward_mode
    if mode.is_extend() and not mode.is_target_verify():
        return "prefill"
    return "decode"


def _queue_for_prefill(sched: Scheduler, req: Req) -> None:
    """A held P-leg leaves the hold (F-C): the waiting queue, as
    symmetric_add_request queues an Export without the flag. A late P-leg
    started its storage prefetch at the hold and goes to its arrival
    position."""
    cfg = _late_cfg(sched)
    entry = sched.export_table.entries.get(req.transfer_id) if cfg is not None else None
    if entry is None or not entry.late:
        sched._prefetch_kvcache(req)
        sched.waiting_queue.append(req)
    elif cfg.arrival_order:
        insert_by_arrival(sched.waiting_queue, req)
    else:
        sched.waiting_queue.append(req)
    req.time_stats.set_wait_queue_entry_time()


def _host_return_validation_error(sched: Scheduler, req: Req) -> Optional[str]:
    """Reject unsupported return directives before allocating an import.

    The host-return path consumes the normal write-through result; it must
    never silently enable a second backup or change the engine's cache policy.
    Peer serving is initialized collectively, including allocator-only ranks.
    """
    if getattr(req.import_mark, "host_return", None) is None:
        return None
    parallel = getattr(sched, "ps", None)
    if (
        getattr(parallel, "pp_size", None) != 1
        or getattr(parallel, "attn_cp_size", None) != 1
        or getattr(parallel, "attn_dcp_size", None) != 1
        or getattr(sched, "enable_dp_attention", False)
        or getattr(parallel, "attn_tp_size", None) != getattr(parallel, "tp_size", None)
    ):
        return "host_return requires plain attention TP, PP=CP=1 and no DP attention"
    cache = sched.tree_cache
    controller = getattr(cache, "cache_controller", None)
    if (
        not getattr(sched, "enable_hierarchical_cache", False)
        or getattr(cache, "host_pool_group", None) is None
        or controller is None
        or getattr(cache, "disable", False)
    ):
        return "host_return requires an enabled HiCache host cache"
    if getattr(controller, "write_policy", None) != "write_through":
        return "host_return requires hicache_write_policy=write_through"
    if getattr(cache, "host_memory_mode", None) != "cache":
        return "host_return requires hicache_host_memory_mode=cache"
    if not getattr(cache, "_peer_kv_tp_participant", False):
        return "host_return requires an initialized peer KV servicer"
    if getattr(req, "skip_radix_cache_insert", False) or getattr(
        req, "hicache_skip_write_through", False
    ):
        return "host_return requires request cache insertion and write-through"
    return None


def _export_spill(sched: Scheduler, req: Req, e_final: int) -> None:
    """Window residency: spill the exporting request's SWA in
    window_export_spill_range(e_prev=e_final) through the regular spill
    ledger (UnifiedRadixCache.submit_swa_spill): device slots are detached
    now and freed at the MIN-reduced ack, the host slices are adopted by
    the finish insert. A full host keeps the device SWA (retried at the
    next stash). Lockstep: every rank stashes / processes the same chunk."""
    if not _is_swa_window(sched.tree_cache) or not req.is_holding_kv:
        return
    page_size = sched.token_to_kv_pool_allocator.page_size
    lo, hi = window_export_spill_range(
        prompt_len=len(req.origin_input_ids),
        e_prev=min(int(e_final), req.kv.kv_allocated_len),
        cps=req.cache_protected_len,
        sev=req.kv.swa_evicted_seqlen,
        sliding_window_size=int(sched.sliding_window_size),
        page_size=page_size,
    )
    table = sched.export_table
    if hi > lo:
        if table is not None:
            # The peak is the span before this spill moves swa_evicted_seqlen.
            table.swa_device_tokens()
        sched.tree_cache.submit_swa_spill([(req, lo, hi)])
    if table is not None:
        table.swa_device_tokens()


def _defer_spill_after_launch(sched: Scheduler) -> bool:
    """SGLANG_DSV4_EXPORT_SPILL_AFTER_LAUNCH, in a loop that runs deferred
    spills (event_loop_overlap sets sched._deferred_export_spills)."""
    return (
        getattr(sched, "_deferred_export_spills", None) is not None
        and envs.SGLANG_DSV4_EXPORT_SPILL_AFTER_LAUNCH.get()
    )


def _publication_stream_ctx(sched: Scheduler):
    """SGLANG_SYMMETRIC_PUBLISH_UNFENCED: the context for a published chunk's
    page-index readbacks (ExportTable._make_chunk). They run on a side stream
    ordered after the event symmetric_mark_pre_war recorded on the schedule
    stream before the last WAR barrier, so they do not wait for the forward
    in flight. They read only launched, export-held state: the request's
    req_to_token rows (written before that chunk's launch), the generations
    of its pages (bumped only when they are freed) and their full->SWA
    mapping; a read beside the forward's reads is WAR-safe. Without an event
    (no launch yet) the current stream is used."""
    ev = sched._pre_war_event
    stream = sched._publish_stream
    if ev is None or stream is None:
        return contextlib.nullcontext()
    stream.wait_event(ev)
    return sched.device_module.stream(stream)


def _release_swa_ledger(tree_cache, req: Req) -> None:
    """Drop a request's window SWA ledger and backfill (disposition
    RELEASE). Both calls are no-ops when there is nothing to release."""
    from decoupled_engine.srt.mem_cache.unified_cache.swa_ledger import LedgerDisposition

    tc = tree_cache
    backfill = getattr(tc, "swa_backfill", None)
    if backfill is not None:
        backfill.cancel(req)
    release = getattr(tc, "release_swa_ledger", None)
    if callable(release):
        release(req, LedgerDisposition.RELEASE)


def _symmetric_late_info(sched: Scheduler) -> Optional[dict]:
    """`symmetric_pd.late_d` of /server_info; None when late D is off."""
    cfg = _late_cfg(sched)
    if cfg is None:
        return None
    queue, table = sched.import_queue, sched.export_table
    if queue is None or queue.late_d is None or table is None:
        return None
    late = queue.late_d
    states = {DEFERRED: 0, ADMITTED: 0, HELD: 0}
    for entry in queue.pending:
        if entry.late_d_state in states:
            states[entry.late_d_state] += 1
    info = {
        "mode": get_disagg().disaggregation_late_d,
        "timers_s": {
            "admission": cfg.admission_timeout_s,
            "bind": cfg.bind_timeout_s,
            "deferred_ttl": cfg.deferred_ttl_s,
            "held_ttl": cfg.held_ttl_s,
            "early_control_ttl": cfg.early_control_ttl_s,
        },
        "aux_reserve": cfg.export_aux_reserve,
        "reader": {
            **states,
            "claim": late.claim.transfer_id if late.claim is not None else None,
            "claim_blocked_passes": late.claim_blocked_passes,
            "pass_index": late.pass_index,
            "early_held": late.stats_early_held,
            "early_dropped": dict(late.stats_early_dropped),
            "misses": dict(queue.stats_late_miss),
            "gates": dict(queue.stats_late_gate),
            "ended": dict(queue.stats_late_ended),
            "claim_skips": queue.stats_claim_skips,
            "prealloc_retries": queue.stats_prealloc_retries,
        },
        "owner": {
            "held": len(table.late_unbound_reqs()),
            "held_aux_wait": table.stats_held_aux_wait,
            "timer_failed": table.stats_late_timer_failed,
            "reader_gone": table.stats_late_reader_gone,
        },
    }
    binder = getattr(sched, "_sym_late_binder", None)
    if binder is not None:
        info["owner"].update(
            r_ahead=binder.r_ahead,
            boost=binder.boost,
            ewma=round(binder.ewma, 3),
            rooms=len(binder.rooms),
            admits=binder.seq,
        )
    return info


class SchedulerSymmetricMixin:
    export_table: Optional[ExportTable] = None
    import_queue: Optional[ImportQueue] = None
    # (req, e_prev) export spills deferred past the launch; a list only in a
    # loop that runs them (event_loop_overlap under SYMMETRIC).
    _deferred_export_spills: Optional[list] = None
    # SGLANG_SYMMETRIC_PUBLISH_UNFENCED: the readback side stream and the
    # schedule-stream event recorded before the last WAR barrier.
    _publish_stream: Optional[object] = None
    _pre_war_event: Optional[object] = None
    # Late D (None = off): the symmetric late-D config (every rank), the
    # owner binder (rank 0) and the arrival counter (every rank alike).
    _sym_late_cfg: Optional[object] = None
    _sym_late_binder: Optional[object] = None
    _sym_arrival_seq: int = 0
    # Late D stage 2, rank 0: the owner's own controls (``local``), appended
    # to the request broadcast after the reader's (LC-1).
    _sym_owner_inbox: Optional[object] = None

    # -- construction ---------------------------------------------------
    def init_symmetric_disaggregation(
        self: Scheduler,
        *,
        draft_token_to_kv_pool,
        disagg_hidden_size: int,
        disagg_hidden_states_dtype,
        output_dsa_topk_indices_dim,
    ) -> None:
        from decoupled_engine.srt.disaggregation.symmetric.planes import (
            PlaneBuildContext,
            build_kv_args,
            build_planes,
            make_worker_id,
        )

        extra_slots = get_disagg().disaggregation_decode_extra_slots or 0
        # Exports and imports share the aux slots; *2 headroom like PD.
        buffer_size = (self.max_running_requests + extra_slots) * 2
        self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
            buffer_size
        )
        self.disagg_metadata_buffers = MetadataBuffers(
            buffer_size,
            hidden_size=disagg_hidden_size,
            hidden_states_dtype=disagg_hidden_states_dtype,
            custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            output_dsa_topk_indices_dim=output_dsa_topk_indices_dim,
        )
        page_size = self.token_to_kv_pool_allocator.page_size
        kv_args = build_kv_args(self, draft_token_to_kv_pool)
        ctx = PlaneBuildContext(
            server_args=self.server_args,
            kv_args=kv_args,
            tp_rank=self.ps.tp_rank,
            tp_size=self.ps.tp_size,
            pp_rank=self.ps.pp_rank,
            dp_rank=self.ps.dp_rank or 0,
            gpu_id=self.ps.gpu_id,
            page_size=page_size,
            metadata_buffers=self.disagg_metadata_buffers,
            token_to_kv_pool=self.token_to_kv_pool_allocator.get_kvcache(),
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
            worker_id=make_worker_id(self.server_args, self.ps.dp_rank),
            attn_tp_size=self.ps.attn_tp_size,
        )
        owner_plane, reader_plane = build_planes(ctx)
        # One gloo group for READY/DONE agreement across this scheduler's TP
        # ranks (same group choice as DecodeKVCacheOffloadManager).
        from decoupled_engine.srt.disaggregation.symmetric.consensus import RankConsensus

        if self.enable_dp_attention:
            consensus_group, consensus_size = (
                self.attn_tp_cpu_group,
                self.ps.attn_tp_size,
            )
        else:
            consensus_group, consensus_size = self.tp_cpu_group, self.ps.tp_size
        rank_consensus = RankConsensus(consensus_group, consensus_size)
        self.import_queue = ImportQueue(
            reader_plane,
            num_active_reqs=lambda: (
                len(self.running_batch.reqs) + len(self.waiting_queue)
            ),
            # init_disaggregation runs before init_output_streamer in
            # Scheduler.__init__, so resolve the streamer at call time.
            stream_output=lambda reqs, return_logprob: (
                self.output_streamer.stream_output(reqs, return_logprob)
            ),
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            metadata_buffers=self.disagg_metadata_buffers,
            req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
            num_reserved_decode_tokens=get_disagg().num_reserved_decode_tokens,
            max_total_num_tokens=self.max_total_num_tokens,
            page_size=page_size,
            spec_is_none=self.spec_algorithm.is_none(),
            waiting_timeout_s=float(envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get()),
            rank_consensus=rank_consensus,
            # D-role SWA-tail import (import_queue.py): needs the window size;
            # auto-off on uniform-lifecycle engines unless they run DSv4 window
            # SWA residency.
            sliding_window_size=getattr(self, "sliding_window_size", None),
            # F-A (import_queue.py, default off): keep the in-flight chunked
            # prefill's remaining KV out of the import admission budget.
            local_reserve=(
                self.symmetric_local_inflight_tokens
                if envs.SGLANG_SYMMETRIC_PD_IMPORT_LOCAL_RESERVE.get()
                else None
            ),
        )
        if (
            self.import_queue.swa_tail_import
            and not self.import_queue.swa_window
            and not self.import_queue._content_only_reuse()
        ):
            logger.warning(
                "symmetric-pd import: SWA-tail import without content-only prefix reuse "
                "(tree %s): imported prompts are neither reused nor cached",
                type(self.tree_cache).__name__,
            )
        # Owner-side SWA validity floor (export_table.owner_swa_valid_floor):
        # an engine with the decode pool shape (hybrid SWA allocator, no uniform
        # lifecycle) frees out-of-window SWA while it prefills, so as an OWNER
        # it must announce those SWA pages as -1 instead of translating them
        # through the cleared full->swa mapping. Uniform-lifecycle engines (the
        # current P / free roles) keep every SWA page: floor stays None.
        swa_valid_floor = None
        swa_tail_window = None
        window = int(getattr(self, "sliding_window_size", None) or 0)
        swa_window_residency = _is_swa_window(self.tree_cache)
        if swa_window_residency:
            # DSv4 window residency: the export floor
            # F_exp = page_floor(L - 1 - max(W, P)) equals a window reader's
            # tail start. SWA below it is announced -1 from the first chunk,
            # so the owner may spill it to the host while it exports.
            assert window > 0, "window SWA residency needs a sliding window size"
            swa_valid_floor = functools.partial(
                window_export_swa_floor, sliding_window_size=window, page_size=page_size
            )
        elif (
            window > 0
            and getattr(self.token_to_kv_pool_allocator, "swa_attn_allocator", None)
            is not None
            and not getattr(self.tree_cache, "dsv4_uniform_cache_lifecycle", False)
        ):
            swa_valid_floor = functools.partial(
                owner_swa_valid_floor, sliding_window_size=window, page_size=page_size
            )
            # Symmetric engines are homogeneous: readers import the SWA tail
            # exactly when this engine does as a reader.
            if self.import_queue.swa_tail_import:
                swa_tail_window = window
        # Readers backfill the SWA below their received tail from this owner's
        # servicer (FinalMeta.owner_peer_kv_addr). Only uniform-lifecycle
        # owners hold every SWA page on the host; others advertise nothing.
        owner_peer_kv_addr = None
        if _resolved(self.server_args, "enable_peer_kv_servicer") and getattr(
            self.tree_cache, "dsv4_uniform_cache_lifecycle", False
        ):
            from decoupled_engine.srt.disaggregation.symmetric.planes import advertised_host

            owner_peer_kv_addr = (
                f"{advertised_host(self.server_args)}:"
                f"{self.server_args.peer_kv_servicer_port}"
            )
        late_d_mode = getattr(get_disagg(), "disaggregation_late_d", LATE_D_OFF)
        self.export_table = ExportTable(
            owner_plane,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            metadata_buffers=self.disagg_metadata_buffers,
            req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
            page_size=page_size,
            # Lease expiry class of timeout.
            export_timeout_s=float(envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get()),
            rank_consensus=rank_consensus,
            swa_valid_floor=swa_valid_floor,
            swa_tail_window=swa_tail_window,
            swa_sentinel_guard=swa_window_residency,
            owner_peer_kv_addr=owner_peer_kv_addr,
            window_metrics=getattr(self.tree_cache, "dsv4_window_metrics", None),
            # F-C (default off): hold P-legs until the reader's BEGIN.
            await_reader_timeout_s=await_reader_timeout_s(),
            # Late D forces the await mode.
            late_d=late_d_mode != LATE_D_OFF,
            compute_before_allocation=(
                getattr(get_disagg(), "disaggregation_late_d_lead_mode", "count")
                == "need"
                and late_d_mode != LATE_D_OFF
            ),
            park_timeout_s=envs.SGLANG_LATE_D_PARK_TIMEOUT_S.get(),
            catchup_tokens_per_pass=envs.SGLANG_LATE_D_CATCHUP_TOKENS_PER_PASS.get(),
            budget_pre_bind_only=envs.SGLANG_SYMMETRIC_BUDGET_PRE_BIND_ONLY.get(),
            budget_standard_pd=envs.SGLANG_SYMMETRIC_BUDGET_STANDARD_PD.get(),
            # Park-gate aging (default off; late_d_need.py, serving_hook.py).
            park_age_s=envs.SGLANG_LATE_D_PARK_AGE_S.get(),
            park_aged_block_s=envs.SGLANG_LATE_D_PARK_AGED_BLOCK_S.get(),
            owner_cache_epoch_getter=lambda: getattr(
                self.tree_cache, "host_return_epoch", None
            ),
        )
        budget_rule = self.export_table.publish_budget_rule
        if budget_rule is not None:
            # The first SYMMETRIC_PUBLISH_BUDGET window starts now.
            self._sym_publish_budget_log_t = time.monotonic()
            if self.ps.tp_rank == 0:
                logger.info(
                    "SYMMETRIC_PUBLISH_BUDGET rule=%s "
                    "catchup_tokens_per_pass=%d compute_before_allocation=%s: %s",
                    budget_rule,
                    self.export_table.catchup_tokens_per_pass,
                    self.export_table.compute_before_allocation,
                    _PUBLISH_BUDGET_RULE_TEXT[budget_rule],
                )
        if self.export_table.late_d:
            self._init_symmetric_late_d()
        self._symmetric_release_insert_chunked = _release_insert_chunked(self)
        # 6a-sym (prefill_loop_unserialize.py): run_batch copies the EAGLE seed
        # of launches that may finish an export with the async result D2H, so
        # FINAL's result processing reads no device tensor behind the barrier.
        from decoupled_engine.srt.disaggregation.prefill_loop_unserialize import (
            metadata_on_host,
            symmetric_async_draft_seed,
        )

        self._symmetric_copy_draft_seed = symmetric_async_draft_seed(self)
        self._symmetric_seed_on_host = metadata_on_host(self.disagg_metadata_buffers)
        if self._symmetric_copy_draft_seed and self.ps.tp_rank == 0:
            logger.info(
                "6a-sym: EAGLE export seed with the async result D2H "
                "(aux slots on host: %s)",
                self._symmetric_seed_on_host,
            )
        logger.info(
            "symmetric-pd symmetric scheduler ready: owner_plane=%s reader_plane=%s "
            "aux_slots=%d extra_req_slots=%d rank_consensus=%s(size=%d) "
            "swa_tail_import=%s swa_tail_export=%s import_local_reserve=%s "
            "import_budget_fraction=%s export_await_reader_s=%s",
            type(owner_plane).__name__,
            type(reader_plane).__name__,
            buffer_size,
            extra_slots,
            "gloo" if rank_consensus.enabled else "local",
            consensus_size,
            self.import_queue.swa_tail_import,
            swa_tail_window is not None,
            self.import_queue.local_reserve is not None,
            self.import_queue.budget_fraction,
            self.export_table.await_reader_timeout_s,
        )

    def _init_symmetric_late_d(self: Scheduler) -> None:
        """Late D: the walk core and reader port on every rank, the
        owner binder, timer and ADMIT sender on rank 0 (DP attention and
        PP > 1 are refused at startup, so rank 0 is TP rank 0)."""
        from decoupled_engine.srt.disaggregation.late_d import (
            LateDConfig,
            LateDDecode,
            LateDInbox,
            LateDPrefillBinder,
            LateDSender,
            make_late_d_metrics,
        )
        from decoupled_engine.srt.disaggregation.late_d_link import LateDLink
        from decoupled_engine.srt.disaggregation.symmetric.late_d_sym import (
            SymOwnerBinderPort,
            SymOwnerTimer,
            SymReaderPort,
        )

        chunked = get_schedule().chunked_prefill_size
        cfg = LateDConfig.for_symmetric(get_disagg(), chunked)
        self._sym_late_cfg = cfg
        self._late_d_need_enabled = self.export_table.compute_before_allocation
        self._late_d_chunk_tokens = chunked or 8192
        is_leader = self.ps.tp_rank == 0
        port = SymReaderPort(self.import_queue, is_leader=is_leader)
        late = LateDDecode(port, cfg, metrics=make_late_d_metrics(self, port))
        self.import_queue.enable_late_d(late, port, aux_reserve=cfg.export_aux_reserve)
        # The planes and the table must wait for the reader.
        assert self.export_table.await_reader, "late D needs the await mode"
        if is_leader:
            # Fake planes route ADMIT in process; real ones over the sender's
            # own PUSH sockets (never the OwnerPlane IO thread, D5).
            fake_sender = getattr(self.export_table.owner_plane, "late_sender", None)
            sender = (
                fake_sender()
                if callable(fake_sender)
                else LateDSender(name="LateDSymAdmitSender", label="ADMIT")
            )
            choose = get_disagg().disaggregation_late_d == LATE_D_CHOOSE
            self._sym_owner_inbox = LateDInbox()
            owner_port = SymOwnerBinderPort(
                self.export_table,
                sender,
                arrival_order=cfg.arrival_order,
                choose=choose,
                owner_inbox=self._sym_owner_inbox,
            )
            owner_port.bound_view = self._symmetric_bound_view
            # SGLANG_KV_SESSION_ADMIT_TAG: the ADMIT carries the P leg's tag
            # only while its registration is live in this registry.
            owner_port.kv_sessions = lambda: getattr(
                self.tree_cache, "kv_sessions", None
            )
            binder = LateDPrefillBinder(
                owner_port,
                cfg,
                chunked,
                metrics=make_late_d_metrics(self, owner_port),
            )
            self.export_table.late_timer = SymOwnerTimer(
                cfg.admission_timeout_s, cfg.bind_timeout_s
            )
            self.export_table.late_metrics = binder.metrics
            self._sym_late_binder = binder
            if choose and not callable(fake_sender):
                # Stage 2 (C8): the router link on owner rank 0; it also
                # sends ADMIT and RELEASE (one thread, no second hop).
                link = LateDLink(
                    binder,
                    get_disagg().disaggregation_late_d_control_port,
                    mode="symmetric",
                    kv_ip="",
                    choice_timeout_s=cfg.choice_timeout_s,
                    metrics=binder.metrics,
                    owner_addr=getattr(
                        self.export_table.owner_plane, "advertised_addr", None
                    ),
                )
                owner_port.sender = link
                binder.link = link
                link.start()
            logger.info(
                "symmetric-pd late D (%s) on: await mode forced (F-C env %s); timers "
                "admission=%.0fs bind=%.0fs deferred_ttl=%.0fs held_ttl=%.0fs "
                "early_control_ttl=%.0fs; aux_reserve=%d claim_holds_local=%s "
                "escalate_passes=%d arrival_order=%s walk_digest_every=%d",
                get_disagg().disaggregation_late_d,
                "set" if envs.SGLANG_SYMMETRIC_PD_EXPORT_AWAIT_READER.get() else "unset",
                cfg.admission_timeout_s,
                cfg.bind_timeout_s,
                cfg.deferred_ttl_s,
                cfg.held_ttl_s,
                cfg.early_control_ttl_s,
                cfg.export_aux_reserve,
                cfg.claim_holds_local,
                cfg.claim_escalate_passes,
                cfg.arrival_order,
                cfg.walk_digest_every,
            )

    def symmetric_pull_late_d_controls(
        self: Scheduler, recv_reqs: Optional[List] = None
    ) -> List:
        """Late D, rank 0 before the request broadcast: the
        reader's admit / fail controls (held back until their D-leg is known;
        a D-leg in ``recv_reqs`` counts) and the DEFERRED / HELD TTLs."""
        queue = self.import_queue
        controls = queue.late_d.pull_controls(
            queue.pending, () if recv_reqs is None else recv_reqs
        )
        inbox = self._sym_owner_inbox
        if inbox is not None:
            # LC-1: the owner's ``local`` controls, for keys this rank holds;
            # never held back.
            controls.extend(inbox.drain())
        return controls

    def symmetric_handle_late_d_control(self: Scheduler, control) -> None:
        """Late D, every rank in the same iteration and order."""
        if control.kind == CONTROL_LOCAL:
            self._symmetric_make_local(control.room)
            return
        queue = self.import_queue
        if queue is None or queue.late_d is None:
            return
        queue.apply_late_control(control, _last_iter_type(self))

    def _symmetric_bound_view(self: Scheduler) -> int:
        """Owner rank 0: num_total_tokens (used +
        waiting-queue tokens) + held P-legs + bound imports, the same sum
        that /v1/loads reports as late_d.bound_total_tokens."""
        used, _ = self.pool_stats_observer.get_pool_stats().get_kv_token_stats()
        waiting = sum(req.seqlen for req in self.waiting_queue)
        held = self.export_table.load_view(self.waiting_queue)["export_held_tokens"]
        bound = self.import_queue.load_view()["import_bound_tokens"]
        return int(used) + waiting + held + bound

    def _symmetric_make_local(self: Scheduler, key: int) -> None:
        """Late D stage 2, LOCAL (LC-4, sketch i), every rank: the held
        P-leg becomes a Local request (full decode, no export) at its arrival
        position. A key that is no longer held and unpublished (aborted,
        failed, or bound) is ignored."""
        table = self.export_table
        req = table.convert_to_local(key)
        if req is None:
            logger.debug("LATE_D local ignored: transfer %s is not held", key)
            return
        release = getattr(table.owner_plane, "release_local", None)
        if callable(release):
            release(key)
        req.export_mark = None
        req.export_stop_after_first_token = False
        req.bootstrap_room = None
        req.transfer_id = None
        req.late_d_outcome = LATE_D_OUTCOME_LOCAL
        # The storage prefetch started at the hold: no second one.
        cfg = _late_cfg(self)
        if cfg is not None and cfg.arrival_order:
            insert_by_arrival(self.waiting_queue, req)
        else:
            self.waiting_queue.append(req)
        req.time_stats.set_wait_queue_entry_time()
        logger.info("LATE_D local rid=%s transfer=%s", req.rid, key)

    def _symmetric_late_fail_export(
        self: Scheduler, req: Req, err_type: str, message: str
    ) -> None:
        """A held late P-leg ends with HTTP 503 and no prefill (rule X): ABORT
        to its reader, release a storage prefetch started at the hold."""
        self.export_table.abort_export(req, err_type)
        if self.enable_hicache_storage:
            self.tree_cache.release_aborted_request(req.rid)
        prepare_abort(
            req, message, status_code=HTTPStatus.SERVICE_UNAVAILABLE, err_type=err_type
        )
        self.output_streamer.stream_output([req], req.return_logprob)
        # No prefill and no KV, so no cache_finished_req: end the request's
        # retention registration count (soft pin).
        forget_kv_session_request(self, req)

    # -- request ingress -------------------------------------------------
    def symmetric_parse_marks(
        self: Scheduler, recv_req: TokenizedGenerateReqInput
    ) -> Tuple[Optional[object], Optional[object], Optional[str]]:
        """(export_mark, import_mark, error). Marks are only honoured under
        SYMMETRIC; other modes reject them so nothing is silently ignored."""
        kv_export = recv_req.kv_export
        kv_import = recv_req.kv_import
        if kv_export is None and kv_import is None:
            return None, None, None
        if self.disaggregation_mode != DisaggregationMode.SYMMETRIC:
            return (
                None,
                None,
                "kv_export/kv_import marks require --disaggregation-mode symmetric",
            )
        try:
            export_mark, import_mark = parse_kv_marks(kv_export, kv_import)
        except MarkError as e:
            return None, None, f"Invalid request: {e}"
        # The router advertises rank-0 endpoints; every TP rank talks to its
        # own peer rank (homogeneous TP), so rewrite the
        # address to this rank's channel (planes.plane_bind_addrs convention).
        from decoupled_engine.srt.disaggregation.symmetric.planes import rank_addr

        tp_rank = self.ps.tp_rank
        if export_mark is not None and export_mark.candidates:
            # Late D stage 2: every candidate reader, before any bind.
            for cand in export_mark.candidates:
                cand.reader_addr = rank_addr(cand.reader_addr, tp_rank)
        elif export_mark is not None:
            export_mark.reader_addr = rank_addr(export_mark.reader_addr, tp_rank)
        if import_mark is not None:
            import_mark.owner_addr = rank_addr(import_mark.owner_addr, tp_rank)
        return export_mark, import_mark, None

    def symmetric_reject(self: Scheduler, req: Req, error_msg: str) -> None:
        logger.error("%s rid=%s", error_msg, req.rid)
        prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
        self.output_streamer.stream_output([req], req.return_logprob)
        # Rejected before any KV: end its retention registration count.
        forget_kv_session_request(self, req)

    @timed_phase("sym.add_request")
    def symmetric_add_request(
        self: Scheduler, req: Req, is_retracted: bool = False
    ) -> None:
        """SYMMETRIC arm of `_add_request_to_queue`."""
        late_cfg = _late_cfg(self)
        if late_cfg is not None:
            # Every request's arrival position, in broadcast order (the
            # same on every rank).
            req.sym_arrival_seq = self._sym_arrival_seq
            self._sym_arrival_seq += 1
        if is_retracted and req.import_mark is not None:
            # A retracted D-leg already consumed its import; it recomputes
            # locally like any retracted request (fallback ladder).
            req.import_mark = None
            if getattr(req, "swa_ledger", None) is not None:
                # DSv4 window residency: the recompute
                # starts from scratch, so the imported prompt's SWA ledger and
                # its backfill go. The retraction released them already
                # (release_kv_cache without insert, disposition RELEASE); this
                # only covers a path that did not.
                _release_swa_ledger(self.tree_cache, req)
        if req.import_mark is not None and not req.finished():
            if getattr(self.tree_cache, "skips_d_role_kv_events", False) is True:
                # SGLANG_SYMMETRIC_SKIP_D_ROLE_KV_EVENTS: this D-leg's cache
                # calls (prompt insert at merge, decode, finish, a retraction
                # or fallback recompute) add nodes that publish no KV events.
                req.kv_events_d_role = True
            error = _host_return_validation_error(self, req)
            if error is not None:
                self.symmetric_reject(req, error)
                return
            self.import_queue.add(req)
            req.time_stats.set_decode_prealloc_queue_entry_time()
            return
        if self._abort_on_queued_limit(req):
            return
        if not self._symmetric_materialize_images(req):
            return
        if req.export_mark is not None and not req.finished():
            late = late_cfg is not None
            began = (
                self.export_table.begin_export(
                    req, late=True, arrival_seq=req.sym_arrival_seq
                )
                if late
                else self.export_table.begin_export(req)
            )
            if not began:
                self.symmetric_reject(
                    req, "No free export aux slot; retry later (symmetric-pd ExportTable)"
                )
                return
            if self.export_table.compute_before_allocation:
                # Fixed reader identity was broadcast with this request. It
                # permits P compute, but does not allocate any reader pages.
                req.late_d_need = True
                req.pending_bootstrap = True
                req.late_d_start_ok = True
                req.late_d_entry_pass = self.forward_ct
            elif self.export_table.await_reader:
                # F-C, reader-first: queued by symmetric_pre_schedule once the
                # reader admitted the import (ExportTable.poll_begun). Late D:
                # the binder sends the ADMIT; the storage prefetch runs while
                # the P-leg is held.
                if late:
                    self._prefetch_kvcache(req)
                    req.time_stats.late_d_hold_time = time.perf_counter()
                self.export_table.hold(req)
                return
        self._prefetch_kvcache(req)
        self.waiting_queue.append(req)
        req.time_stats.set_wait_queue_entry_time()

    def init_symmetric_bootstrap_prefetch(self: Scheduler):
        """SGLANG_SYMMETRIC_BWP: the bootstrap prefetcher of a symmetric
        worker, or None when off (the mode was left "off" by
        init_disaggregation without the flag).

        A symmetric worker has no bootstrap queue, so the prefetcher takes its
        candidates from the waiting queue through the lookahead plan
        (``SGLANG_BWP_SOURCE=waiting``; the forecast is updated by the overlap
        loop after each result). Its pins also leave room for running decodes
        and bound imports (``BootstrapPrefetcher._symmetric_reserve``).

        BWP v2 C3 (SGLANG_BWP_V2 + SGLANG_BWP_SYM_PRIVATE_PLAN): the
        prefetcher owns its plan and no ``late_d_lookahead`` is installed, so
        the late-D bound reserve keeps its prefetch-off input."""
        if getattr(self, "bootstrap_prefetch_mode", "off") == "off":
            return None
        from decoupled_engine.srt.disaggregation.bootstrap_prefetch import BootstrapPrefetcher
        from decoupled_engine.srt.disaggregation.lookahead_plan import SchedulerLookahead
        from decoupled_engine.srt.environ import envs

        if not (envs.SGLANG_BWP_V2.get() and envs.SGLANG_BWP_SYM_PRIVATE_PLAN.get()):
            self.late_d_lookahead = SchedulerLookahead(self)
        self.bootstrap_prefetch = BootstrapPrefetcher(self)
        return self.bootstrap_prefetch

    # -- per tick ---------------------------------------------------------
    def symmetric_publish_after_result(self: Scheduler) -> None:
        """SGLANG_SYMMETRIC_PUBLISH_AFTER_RESULT: called by event_loop_overlap
        right after each processed result. A compute-first owner publishes
        the KV that this result made visible (and FINAL when its prefill is
        done) now, within the rest of this pass's catch-up budget, instead of
        in the next symmetric_pre_schedule (with
        SGLANG_SYMMETRIC_BUDGET_PRE_BIND_ONLY only the KV computed before the
        bind uses it, with SGLANG_SYMMETRIC_BUDGET_STANDARD_PD only the
        catch-up of an export launched before its bind).
        Under the gate-pending reorder the result was processed before the
        launch, and that launch would otherwise delay the publication by a
        whole forward.

        Every TP rank calls this at the same point of its loop (after the
        processing, wherever the reorder put it), and reader_ready only
        changes in symmetric_pre_schedule's consensus round, so all ranks
        publish the same chunks. No collective runs here."""
        table = self.export_table
        if (
            table is None
            or getattr(table, "compute_before_allocation", False) is not True
        ):
            return
        before = table.chunks_published_total()
        finals = table.publish_computed(new_pass=False)
        table.stats_after_result_calls += 1
        table.stats_after_result_publications += table.chunks_published_total() - before
        table.stats_after_result_finals += finals
        now = time.monotonic()
        last_log = getattr(self, "_sym_after_result_log_t", 0.0)
        if getattr(self, "tp_rank", 0) == 0 and now - last_log >= 60.0:
            self._sym_after_result_log_t = now
            logger.info(
                "SYMMETRIC_PUBLISH_AFTER_RESULT calls=%d chunks=%d finals=%d",
                table.stats_after_result_calls,
                table.stats_after_result_publications,
                table.stats_after_result_finals,
            )

    # Admission path: sym_pre_schedule >
    # sym_pop_ready (> sym_pop_ready_vote1 / _commit / _vote2) > sym_merge_prebuilt
    # (> prebuilt_*) > sym_admit_pending. Nested phases are recorded separately.
    @timed_phase("sym_pre_schedule")
    # SGLANG_SCHED_SYNC_DEBUG site (identity when off): the admission tick's
    # synchronizing CUDA calls (SGLANG_PREBUILT_NONBLOCKING_H2D smoke).
    @sync_debug_site("sym_pre_schedule")
    def symmetric_pre_schedule(
        self: Scheduler, running_batch: ScheduleBatch
    ) -> ScheduleBatch:
        """Called from get_next_batch_to_run after the last-batch merge."""
        with phase("sym.export.poll_released"):
            for req, reason in self.export_table.poll_released():
                self._symmetric_free_exported(req, reason)
        if getattr(self.export_table, "compute_before_allocation", False) is True:
            for entry in list(self.export_table.entries.values()):
                if entry.compute_before_allocation and entry.abort_reason is not None:
                    self._symmetric_stop_computed(entry.req, entry.abort_reason)
            if _publish_budget_rule(self.export_table) is not None:
                # Before the first call of the new pass: a log window then
                # ends between two passes. The after-result calls of a pass
                # count in the window of its pre-schedule call.
                _log_publish_budget(self)
            with phase("sym.export.publish_computed"):
                self.export_table.publish_computed()
        # F-C: held P-legs whose reader admitted the import on every rank (or
        # whose wait timed out: queued as without the flag).
        with phase("sym.export.poll_begun"):
            held_out = (
                self.export_table.poll_begun()
                if getattr(self.export_table, "await_reader", False)
                else ()
            )
        for req, why in held_out:
            if why == LATE_BIND_TIMEOUT:
                # 503, ABORT to the reader, never queued.
                err_type = self.export_table.late_fail_err_type(req)
                self._symmetric_late_fail_export(
                    req, err_type, f"Late D: P-leg not bound in time ({err_type})"
                )
                continue
            if why != READER_BEGUN:
                logger.warning(
                    "symmetric-pd export rid=%s transfer_id=%s: %s after %.0f s; "
                    "queued without its reader",
                    req.rid,
                    req.transfer_id,
                    why,
                    self.export_table.await_reader_timeout_s or 0.0,
                )
            _queue_for_prefill(self, req)

        chunked_req = self.chunked_req
        if (
            chunked_req is not None
            and chunked_req.export_mark is not None
            and chunked_req.extend_range is not None
        ):
            # Overlap: the chunk result is processed after the next chunk has
            # advanced extend_range, so remember the end of the chunk that is
            # in flight now (same trick as prefill.py:process_prefill_chunk).
            chunked_req.tmp_end_idx = min(
                chunked_req.extend_range.end, len(chunked_req.origin_input_ids)
            )

        # READY imports still need an execution slot, just like native PD's
        # prebuilt queue. Preallocated transfer slots are not running slots.
        capacity = min(self.req_to_token_pool.size, self.max_running_requests)
        occupied = running_batch.batch_size()
        if chunked_req is not None and chunked_req not in running_batch.reqs:
            # An admitted chunk must continue even when prefill admission is
            # full; retain its slot until its final chunk joins the batch.
            occupied += 1
        ready, failed = self.import_queue.pop_ready(
            max_ready=max(0, capacity - occupied)
        )
        with phase("sym.import.fallback"):
            for req, reason in failed:
                self._symmetric_import_fallback(req, reason)
        if ready:
            running_batch = self._symmetric_merge_prebuilt(ready, running_batch)
        # This call site only (admit_pending also runs at request arrival:
        # phase sym.import.admit covers every call).
        if getattr(self.import_queue, "defer_admission", False) is not True:
            with phase("sym_admit_pending"):
                self.import_queue.admit_pending()
        # else SGLANG_SYMMETRIC_ADMIT_AFTER_LAUNCH: the overlap loop walks
        # once per pass, after batch formation; it clears defer_admission for
        # a pass that must walk here (memory-blocked late-D claim, the escape
        # in admit_after_launch.py).
        binder = getattr(self, "_sym_late_binder", None)
        if binder is not None:
            # Late D: after the BEGIN pickup and the walk.
            with phase("sym.late_binder"):
                binder.step(
                    self.export_table.late_unbound_reqs(),
                    self.waiting_queue,
                    self.chunked_req,
                    # The launch-rate EWMA counts prefill iterations only.
                    batch_ran=_last_iter_type(self) == "prefill",
                )
        return running_batch

    def symmetric_local_inflight_tokens(self: Scheduler) -> int:
        """F-A (SGLANG_SYMMETRIC_PD_IMPORT_LOCAL_RESERVE): KV tokens the in-flight
        chunked prefill (an Export's P-leg or a Local request) still has to
        allocate: the rest of its page-aligned input past what its chunks
        already allocated, plus the page alloc_extend may add. A Local request
        also gets one decode headroom: it will decode, and while chunked it is
        in neither running_batch nor waiting_queue, so the import queue's
        per-request headroom does not count it. 0 when nothing is in flight.

        PrefillAdder lets a started chunk continue while any page is free
        (add_chunked_req), but import admission runs first in each tick
        (symmetric_pre_schedule) and could take those pages; with the peer
        worker waiting for this P-leg that wedges both workers (the last
        chunk never runs)."""
        req = self.chunked_req
        if req is None or req.finished():
            return 0
        page = self.token_to_kv_pool_allocator.page_size
        total = max(
            len(req.origin_input_ids),
            len(getattr(req, "full_untruncated_fill_ids", None) or ()),
        )
        remaining = ceil_align(total, page) - int(req.kv.kv_allocated_len)
        if remaining <= 0:
            return 0
        reserve = remaining + page
        if req.export_mark is None and self.import_queue is not None:
            reserve += self.import_queue.num_reserved_decode_tokens
        return reserve

    def symmetric_reqs_to_exclude_from_merge(
        self: Scheduler, last_batch: ScheduleBatch
    ) -> List[Req]:
        """Export requests finish at the end of prefill and must not enter
        the running (decode) batch; mirrors prefill.py's skip-merge."""
        return [r for r in last_batch.reqs if r.export_mark is not None]

    def symmetric_on_prefill_chunk(self: Scheduler, req: Req, last: bool) -> bool:
        """Export hook from process_batch_result_prefill. Returns True when the
        table took ownership of the request's pages (caller must not free)."""
        if req.export_mark is None or req.transfer_id not in self.export_table:
            return False
        entry = self.export_table.entries[req.transfer_id]
        if entry.compute_before_allocation:
            if (
                isinstance(req.finished_reason, FINISH_ABORT)
                or req.beam_group is not None
            ):
                return self.export_table.abort_export(req, COMPUTE_ABORT)
            end = (
                len(req.origin_input_ids)
                if last
                else (req.tmp_end_idx if self.enable_overlap else req.extend_range.end)
            )
            if end is not None and end >= 0:
                if last:
                    entry.total_tokens = min(end, req.extend_range.end)
                    req.time_stats.set_completion_time()
                self.export_table.record_compute_result(req, end, last=last)
            return True
        if not last:
            end_idx = req.tmp_end_idx if self.enable_overlap else req.extend_range.end
            if end_idx is None or end_idx < 0:
                return True
            self.export_table.publish_progress(req, end_idx)
            return True
        if isinstance(req.finished_reason, FINISH_ABORT) or req.beam_group is not None:
            return self.export_table.abort_export(req, "request aborted before export")
        # The prompt's forward is done: of its SWA only the reader's tail stays.
        self.export_table.trim_swa(req, len(req.origin_input_ids))
        # DSv4 window residency: the whole prompt's KV is final now (this is
        # its last chunk's result), and no batch reads this request's SWA
        # again, so spill everything below the export floor. Only the pages
        # announced to the reader, [F_exp, L), stay on the device under the
        # Lease (prefill window + <= 2 tail pages).
        # By default the spill runs after FINAL (below): the spilled range
        # ends at or below F_exp, so FINAL announces it -1 either way.
        spill_after_final = envs.SGLANG_DSV4_EXPORT_SPILL_AFTER_FINAL.get()
        if not spill_after_final:
            _export_spill(self, req, len(req.origin_input_ids))
        # Hand the KV to the Lease WITHOUT inserting it into the radix tree
        # first. The request keeps holding its pages (is_holding_kv) until the
        # reader releases the export, and _symmetric_free_exported then runs
        # release_kv_cache (insert + free) like a Local finish. An early
        # cache_unfinished_req would dedupe the prompt against an existing
        # prefix: the request is re-pointed to the tree's slots and its own,
        # already-announced pages are freed -> their generation moves, the
        # reader's FINAL check fails (`owner_evicted`, 0.08 % of imports on
        # H100) and a READ still in flight would target freed pages.
        # Window residency: that finish insert also adopts the spilled SWA.
        final = self.export_table.finish_export(req)
        req.time_stats.set_completion_time()
        if spill_after_final:
            # The spill's D2H copy (~40 MB per 1K tokens per TP rank on
            # DSv4-Flash) occupies the GPU's D2H copy engine, and every later
            # device->host readback on any stream waits behind it, including
            # finish_export's page-index readbacks: spilling first delayed
            # FINAL by the whole copy (H100: +2.6 to +15.6 ms per 1K spilled
            # tokens, depending on the host). The reader never reads these
            # pages, and the release-time insert that adopts them runs at a
            # later iteration (poll_released).
            _export_spill(self, req, len(req.origin_input_ids))
        logger.debug(
            "symmetric-pd export rid=%s transfer_id=%d tokens=%d aux=%d: EXPORTED",
            req.rid,
            final.transfer_id,
            final.total_tokens,
            final.aux_index,
        )
        return True

    def symmetric_stash_chunked_request(self: Scheduler, req: Req) -> bool:
        """Chunk-boundary stash (Scheduler.stash_chunked_request) for a request
        with an active export. Returns True when handled.

        The regular stash inserts the finished chunks into the radix tree
        (cache_unfinished_req). With hicache/unified cache that insert dedupes
        the chunk against a prefix another request inserted meanwhile (the same
        session's previous turn releasing its export, a sibling prompt): the
        request is re-pointed to the tree's slots and its own pages -- already
        announced to the reader by publish_progress -- are freed. Their
        generation moves and the reader's FINAL check fails (`owner_evicted`,
        recompute fallback; ~0.06 % of imports at concurrency 256 once the
        finish-time insert was removed). An exporting request therefore keeps
        its own pages until the lease is released: only extend
        `prefix_indices` so the next chunk continues from the right offset
        (chunk_cache.cache_unfinished_req semantics). The full prompt is
        inserted once at release (_symmetric_free_exported -> release_kv_cache).
        Without that insert nothing else frees the out-of-window SWA, so the
        SWA-tail trim below takes the place of the tree's.

        DSv4 window residency (export ledger): the stash
        also spills the request's SWA below min(page_floor(E_prev - 1 -
        max(W, P)), F_exp) to the host, E_prev being the previous stash end
        (that chunk's result is processed, so its KV is final; the chunk
        launched since reads only above it). A request that stopped exporting
        before its FINAL (RELEASED_BEFORE_EXPORT) but spilled already keeps
        this no-insert stash: its ledger is adopted by its finish insert, like
        an export's at release."""
        if req.extend_range is None or self.export_table is None:
            return False
        exporting = req.export_mark is not None and req.transfer_id in self.export_table
        former_export = (
            req.export_mark is None
            and getattr(req, "swa_ledger", None) is not None
            and _is_swa_window(self.tree_cache)
        )
        if not (exporting or former_export):
            return False
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.extend_range.end
        ]
        # M4 Q4: the prefix carries its CPU page ids (no-op when Q4 is off).
        req.prefix_indices = tag_row_slice(
            kv_indices.to(dtype=torch.int64, copy=True),
            cpu_rows_of(self.req_to_token_pool),
            req.req_pool_idx,
            0,
        )
        if exporting and getattr(self, "_symmetric_release_insert_chunked", False):
            # A standard P inserts the fill up to here now; the release insert
            # ends a node here instead (SGLANG_SYMMETRIC_RELEASE_INSERT_CHUNKED).
            ends = getattr(req, "release_insert_chunk_ends", None)
            if ends is None:
                ends = req.release_insert_chunk_ends = []
            if not ends or ends[-1] != req.extend_range.end:
                ends.append(req.extend_range.end)
        # Under overlap the chunk in extend_range may still be running and reads
        # SWA from extend_range.start - window on.
        self.export_table.trim_swa(req, req.extend_range.start)
        if _is_swa_window(self.tree_cache):
            e_prev = getattr(req, "swa_export_stash_end", 0) or 0
            if _defer_spill_after_launch(self):
                # Runs in symmetric_run_deferred_export_spills, right after
                # this iteration's launch (its range stays below what the
                # chunk in flight and the one launched next read).
                self._deferred_export_spills.append((req, e_prev))
            else:
                _export_spill(self, req, e_prev)
            req.swa_export_stash_end = req.extend_range.end
        return True

    @timed_phase("sym.export.deferred_spills")
    def symmetric_run_deferred_export_spills(self: Scheduler) -> None:
        """Run the chunk-boundary export spills queued at this iteration's
        stash. The overlap loop calls this after run_batch and before the WAR
        barrier: the schedule stream is not yet fenced behind the forward just
        launched, so the spill's detach check never waits on it. In queue
        order, on every rank alike (the stash is lockstep). A request that no
        longer holds its KV is skipped (_export_spill)."""
        pending = self._deferred_export_spills
        if not pending:
            return
        self._deferred_export_spills = []
        for req, e_prev in pending:
            _export_spill(self, req, e_prev)

    def symmetric_mark_pre_war(self: Scheduler) -> None:
        """SGLANG_SYMMETRIC_PUBLISH_UNFENCED: record the schedule stream's
        position right before _apply_war_barrier fences it behind the forward
        just launched; publications order their readbacks after this event
        instead (_publication_stream_ctx). Installs the side stream and the
        export table's readback context on first use."""
        if self.export_table is None:
            return
        if self._publish_stream is None:
            self._publish_stream = self.device_module.Stream()
            self.export_table.readback_stream_ctx = functools.partial(
                _publication_stream_ctx, self
            )
        ev = self.device_module.Event()
        ev.record(self.schedule_stream)
        self._pre_war_event = ev

    def init_symmetric_publish_readback(self: Scheduler):
        """SGLANG_SYMMETRIC_PUBLISH_ASYNC_READBACK, once at the start of the
        overlap loop, on the scheduler thread (every TP rank, no collective):
        draw the readback side stream with the loader's I9 overlap self-test
        (no forward is in flight yet), install an ExportPageReader on the
        export table and in the owner plane's Lease (its generation checks at
        CHUNK_READY and FINAL), and return it for the loop's marks. None
        without an export table."""
        table = self.export_table
        if table is None:
            return None
        from decoupled_engine.srt.disaggregation.symmetric.publish_readback import (
            ExportPageReader,
            draw_readback_stream,
            schedule_overlaps_forward,
        )

        self_test = None
        if self.device_module is torch.cuda:
            from decoupled_engine.srt.mem_cache.ce_loader import overlap_self_test

            self_test = overlap_self_test
        # The side stream must share no hardware queue with the forward stream
        # nor with the schedule stream (it holds the WAR-barrier wait).
        stream, ok, redraws, why = draw_readback_stream(
            self.device_module,
            self.forward_stream,
            self.schedule_stream,
            self_test=self_test,
        )
        # Diagnostic: the engine's own pair (E completes after the forward when
        # they share a queue).
        sched_ok, sched_why = schedule_overlaps_forward(
            self.device_module,
            self.forward_stream,
            self.schedule_stream,
            self_test=self_test,
        )
        tp_rank = getattr(getattr(self, "ps", None), "tp_rank", 0)
        reader = ExportPageReader(
            device_module=self.device_module,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            stream=stream,
            check=envs.SGLANG_SYMMETRIC_PUBLISH_ASYNC_READBACK_CHECK.get(),
            tp_rank=tp_rank,
        )
        table.readback_page_reader = reader
        lease = getattr(table.owner_plane, "lease", None)
        lease_wrapped = bool(
            lease is not None
            and callable(getattr(lease, "wrap_generation_of", None))
            and lease.wrap_generation_of(reader.wrap_generation_of)
        )
        rows_tracked = reader.rows_tracked
        healthy = ok and sched_ok is not False and rows_tracked
        (logger.info if healthy else logger.warning)(
            "SYMMETRIC_PUBLISH_ASYNC_READBACK on: rank %s, side-stream overlap "
            "check %s after %d redraws (%s), schedule/forward queues %s (%s), "
            "row tracking %s (%s), lease generations %s, check mode %s",
            tp_rank,
            "PASS" if ok else "FAIL (reads may wait for the forward)",
            redraws,
            why,
            {
                True: "separate",
                False: "SHARED (reads wait for the forward)",
                None: "unchecked",
            }[sched_ok],
            sched_why,
            "on" if rows_tracked else "UNSUPPORTED (every read takes the old path)",
            type(self.req_to_token_pool).__name__,
            "async" if lease_wrapped else "unchanged (no generation_of)",
            reader.check,
        )
        return reader

    def symmetric_stage_export_seed(self: Scheduler, batch: ScheduleBatch) -> bool:
        """6a-sym, from run_batch: copy this launch's EAGLE draft seed with the
        async result D2H (the launch may finish an export)."""
        from decoupled_engine.srt.disaggregation.prefill_loop_unserialize import (
            may_finish_export,
        )

        return bool(getattr(self, "_symmetric_copy_draft_seed", False)) and (
            may_finish_export(batch)
        )

    def symmetric_abort_queued(self: Scheduler, recv_req: AbortReq) -> None:
        from decoupled_engine.srt.managers.scheduler import _make_abort_req

        # Completed P responses can still own an EXPORTED request. They are
        # absent from the running/waiting batches but must receive revocation.
        for req in self.export_table.exported_reqs():
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                self.export_table.abort_export(req, "aborted by client")
                if getattr(req, "late_d_need", False):
                    self._symmetric_stop_computed(req, "aborted by client")
        # F-C: a held P-leg is in no queue; abort it like a queued request.
        for req in self.export_table.held_reqs():
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                self.export_table.abort_export(req, "aborted by client")
                if _late_cfg(self) is not None:
                    # Release a prefetch started at the hold; the
                    # binder must not count the room as launched.
                    if self.enable_hicache_storage:
                        self.tree_cache.release_aborted_request(req.rid)
                    prepare_abort(req, "aborted by client")
                self.ipc_channels.send_to_tokenizer.send_output(
                    _make_abort_req(req), req
                )
                # A held P-leg has no KV: no cache_finished_req follows.
                forget_kv_session_request(self, req)
                logger.debug("Abort held export request. rid=%s", req.rid)
        for req in self.import_queue.abort_matching(recv_req.rid, recv_req.abort_all):
            self.ipc_channels.send_to_tokenizer.send_output(_make_abort_req(req), req)
            # A pending import holds no KV (no cache_finished_req); for an
            # active one the release already did this.
            forget_kv_session_request(self, req)
            logger.debug("Abort queued import request. rid=%s", req.rid)

    def symmetric_is_idle(self: Scheduler, for_health_check: bool = False) -> bool:
        idle = len(self.import_queue) == 0
        if not for_health_check:
            idle &= len(self.export_table) == 0
            # DSv4 window residency: unacked spills own
            # detached device SWA and host slots, and pending backfills own
            # host slots a read may still write. Flush and the idle leak check
            # must wait for them. Other trees have neither attribute.
            tc = self.tree_cache
            # These maps are replicated scheduler state. A return can retain
            # CPU pages after its request finishes; flush/detach must wait for
            # its completion or timeout, without inspecting rank-local IO queues.
            for name in ("_host_returns", "_host_return_imports"):
                pending = getattr(tc, name, None)
                idle &= not (isinstance(pending, dict) and pending)
            registry = getattr(tc, "swa_ledger_registry", None)
            if registry is not None:
                idle &= registry.live_count() == 0
            backfill = getattr(tc, "swa_backfill", None)
            if backfill is not None:
                idle &= backfill.pending_count() == 0
        return idle

    def symmetric_has_leased_pages(self: Scheduler) -> bool:
        return self.export_table is not None and self.export_table.has_exported()

    def symmetric_info(self: Scheduler) -> dict:
        """`symmetric_pd` section of /server_info (per TP rank). Gives a driver or
        router the endpoints to put in kv_export.reader_addr /
        kv_import.owner_addr (rank-0 values; ranks derive theirs by
        planes.rank_addr) plus queue / Lease / transfer telemetry."""
        from decoupled_engine.srt.disaggregation.symmetric.planes import plane_bind_addrs

        table, queue = self.export_table, self.import_queue
        owner_base, reader_base = plane_bind_addrs(self.server_args, 0)
        owner_addr, reader_addr = plane_bind_addrs(self.server_args, self.ps.tp_rank)
        info = {
            "mode": "symmetric",
            "tp_rank": self.ps.tp_rank,
            "tp_size": self.ps.tp_size,
            "page_size": self.token_to_kv_pool_allocator.page_size,
            # What a router/driver should advertise (rank-0 endpoints).
            "owner_addr": owner_base,
            "reader_addr": reader_base,
            # This rank's own channels.
            "rank_owner_addr": owner_addr,
            "rank_reader_addr": reader_addr,
            # ExportTable / ImportQueue define __len__, so an empty one is
            # falsy: test for None explicitly.
            # DSv4 SWA residency of this worker ("full" | "window"). A router
            # pairs and picks the import cache policy from it.
            "swa_residency": "window" if _is_swa_window(self.tree_cache) else "full",
            "export_table": {
                "entries": len(table) if table is not None else 0,
                "exported": len(table.exported_reqs()) if table is not None else 0,
                "swa_tail_export": (
                    table.swa_tail_window is not None if table is not None else False
                ),
                "swa_trimmed_tokens": (
                    table.stats_swa_trimmed_tokens if table is not None else 0
                ),
                # SWA pages hidden below the owner floor, and pages at or above
                # it announced -1 because unmapped (window residency; must be 0).
                "swa_floor_pages": (
                    table.stats_swa_floor_pages if table is not None else 0
                ),
                "swa_sentinel_pages": (
                    table.stats_swa_sentinel_pages if table is not None else 0
                ),
                # Device SWA held privately by exports: last sample and peak.
                "swa_tokens": table.stats_swa_tokens if table is not None else 0,
                "swa_peak_tokens": (
                    table.stats_swa_peak_tokens if table is not None else 0
                ),
                "owner_peer_kv_addr": (
                    table.owner_peer_kv_addr if table is not None else None
                ),
                # F-C reader-first exports (default off): P-legs held now, and
                # how the held ones left (BEGIN / timeout / reader gave up).
                "await_reader_s": (
                    table.await_reader_timeout_s if table is not None else None
                ),
                "held": len(table.held_reqs()) if table is not None else 0,
                "held_total": table.stats_held if table is not None else 0,
                "held_begun": table.stats_held_begun if table is not None else 0,
                "held_timeout": (table.stats_held_timeout if table is not None else 0),
                "held_released": (
                    table.stats_held_released if table is not None else 0
                ),
                "held_max_s": table.stats_held_max_s if table is not None else 0.0,
            },
            "import_queue": {
                "pending": len(queue.pending) if queue is not None else 0,
                "active": len(queue.active) if queue is not None else 0,
                "prefix_reuse": (
                    bool(queue.prefix_reuse) if queue is not None else False
                ),
                "prefix_reuse_requests": (
                    queue.stats_prefix_reuse_requests if queue is not None else 0
                ),
                "prefix_reuse_tokens": (
                    queue.stats_prefix_reuse_tokens if queue is not None else 0
                ),
                "imported_tokens": (
                    queue.stats_imported_tokens if queue is not None else 0
                ),
                "prealloc_retries": (
                    queue.stats_prealloc_retries if queue is not None else 0
                ),
                "swa_tail_import": (
                    bool(queue.swa_tail_import) if queue is not None else False
                ),
                "swa_tail_requests": (
                    queue.stats_swa_tail_requests if queue is not None else 0
                ),
                "swa_tail_skipped_tokens": (
                    queue.stats_swa_tail_skipped_tokens if queue is not None else 0
                ),
                "content_only_reuse": (
                    queue._content_only_reuse() if queue is not None else False
                ),
                "host_load_back_tokens": (
                    queue.stats_host_load_back_tokens if queue is not None else 0
                ),
                "host_load_back_skipped": (
                    queue.stats_host_load_back_skipped if queue is not None else 0
                ),
                # Window residency: gpu_only policies coerced to retain, and
                # SWA backfills submitted at import commit / skipped because
                # backfill is off or the owner gave no servicer address.
                "cache_policy_overridden": (
                    queue.stats_cache_policy_overridden if queue is not None else 0
                ),
                "swa_backfill_submitted": (
                    queue.stats_swa_backfill_submitted if queue is not None else 0
                ),
                "swa_backfill_skipped": (
                    queue.stats_swa_backfill_skipped if queue is not None else 0
                ),
                # Admission liveness (F-A / F-B, default off): configuration,
                # tokens the active imports hold, and admissions held back.
                "local_reserve": (
                    queue.local_reserve is not None if queue is not None else False
                ),
                "budget_fraction": (
                    queue.budget_fraction if queue is not None else 0.0
                ),
                "active_import_tokens": (
                    queue.active_import_tokens() if queue is not None else 0
                ),
                "budget_held": queue.stats_budget_held if queue is not None else 0,
                "reserve_held": (queue.stats_reserve_held if queue is not None else 0),
            },
            # Unified radix tree self-heal counter (deleted node handed back to
            # the leaf sets; see UnifiedTreeCore._unregister_node). Non-zero
            # means a stale node reference exists somewhere -- check the
            # engine log for the "leaf-set update on deleted node" stack.
            "tree_cache": {
                "stale_leaf_set_updates": getattr(
                    getattr(self.tree_cache, "tree_core", None),
                    "stale_leaf_set_updates",
                    None,
                ),
            },
        }
        info["late_d"] = _symmetric_late_info(self)
        for name, plane in (
            ("owner", table.owner_plane if table is not None else None),
            ("reader", queue.reader_plane if queue is not None else None),
        ):
            snap = getattr(plane, "snapshot", None)
            info[name] = {"plane": type(plane).__name__}
            if callable(snap):
                try:
                    info[name].update(snap())
                except Exception as e:  # introspection must never break the tick
                    info[name]["error"] = repr(e)
        return info

    # -- helpers ----------------------------------------------------------
    @timed_phase("sym.forward_start")
    def symmetric_forward_start(self, batch: ScheduleBatch) -> None:
        if (
            not self.export_table.compute_before_allocation
            or not batch.forward_mode.is_extend()
        ):
            return
        started = []
        for req in batch.reqs:
            entry = (
                self.export_table.entries.get(req.transfer_id)
                if req.export_mark is not None
                else None
            )
            if entry is None or not entry.compute_before_allocation:
                continue
            entry.pending_forwards += 1
            if req.late_d_first_chunk_started_perf == 0:
                req.late_d_first_chunk_started_perf = time.perf_counter()
                req.late_d_first_chunk_started_unix_ns = time.time_ns()
                started.append(req)
                logger.info(
                    "LATE_D first_chunk_start rid=%s forward_iter=%d unix_ns=%d pending=%d symmetric=1",
                    req.rid,
                    batch.forward_iter,
                    req.late_d_first_chunk_started_unix_ns,
                    int(req.pending_bootstrap),
                )
        binder = getattr(self, "_sym_late_binder", None)
        if binder is not None and started:
            binder.first_chunk_started(started)

    @timed_phase("sym.forward_finished")
    def symmetric_forward_finished(self, batch: ScheduleBatch) -> None:
        if (
            not self.export_table.compute_before_allocation
            or not batch.forward_mode.is_extend()
        ):
            return
        for req in batch.reqs:
            entry = (
                self.export_table.entries.get(req.transfer_id)
                if req.export_mark is not None
                else None
            )
            if entry is not None and entry.compute_before_allocation:
                assert (
                    entry.pending_forwards > 0
                ), "export result without a matching launch"
                entry.pending_forwards -= 1
                if isinstance(req.finished_reason, FINISH_ABORT):
                    self.export_table.abort_export(req, COMPUTE_ABORT)

    def _symmetric_stop_computed(self, req: Req, reason: str) -> None:
        # Do not free here: a final forward, published remote READ, or both
        # can still use the pages. Table consensus owns the only free path.
        if not isinstance(req.finished_reason, FINISH_ABORT):
            # A finished compute result need not yet have passed the output
            # streamer. Its explicit emitted marker owns P-leg completion.
            response_finished = req.finished_output
            prepare_abort(req, reason, status_code=HTTPStatus.SERVICE_UNAVAILABLE)
            if self.enable_hicache_storage:
                self.tree_cache.release_aborted_request(req.rid)
            if not response_finished:
                self.output_streamer.stream_output([req], req.return_logprob)
        self.waiting_queue = [r for r in self.waiting_queue if r is not req]
        if self.chunked_req is req:
            self.chunked_req = None
        # The request is aborted. A queued one never reaches
        # cache_finished_req (COMPUTE_ABORT releases KV only when held): end
        # its retention registration count here. Repeats change nothing.
        forget_kv_session_request(self, req)

    def _symmetric_free_exported(self: Scheduler, req: Req, reason: str) -> None:
        if reason == COMPUTE_ABORT:
            self._symmetric_stop_computed(req, reason)
            if req.is_holding_kv:
                release_kv_cache(req, self.tree_cache, is_insert=False)
            return
        if reason == LATE_READER_GONE:
            # Rule X: the reader of a held late P-leg gave up; 503, no
            # prefill. The table already sent ABORT and freed any aux slot.
            logger.info(
                "symmetric-pd export rid=%s transfer_id=%s: late-D reader gone while "
                "the P-leg was held; 503",
                req.rid,
                req.transfer_id,
            )
            if self.enable_hicache_storage:
                self.tree_cache.release_aborted_request(req.rid)
            prepare_abort(
                req,
                "Late D: the reader gave up before the P-leg was bound",
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                err_type=LATE_D_READER_GONE_ERR,
            )
            self.output_streamer.stream_output([req], req.return_logprob)
            # No prefill and no KV: end its retention registration count.
            forget_kv_session_request(self, req)
            return
        if reason == RELEASED_WHILE_HELD:
            # F-C: the reader gave up while the P-leg was held (it never ran
            # and sits in no queue). Queue it as Local, as RELEASED_BEFORE_EXPORT
            # leaves a queued P-leg: it still answers with its first token.
            logger.info(
                "symmetric-pd export rid=%s transfer_id=%s: reader gone while the "
                "P-leg awaited it; queued as Local",
                req.rid,
                req.transfer_id,
            )
            req.export_mark = None
            _queue_for_prefill(self, req)
            return
        if reason == RELEASED_BEFORE_EXPORT:
            # P-leg still computing; it now finishes as a plain request (its
            # first-token clamp stays) and frees its own KV on completion.
            # Window residency: SWA it already spilled stays in its ledger; its
            # remaining chunk stashes keep the no-insert export stash and its
            # finish insert adopts the ledger (symmetric_stash_chunked_request).
            logger.info(
                "symmetric-pd export rid=%s transfer_id=%s: reader gone before export; "
                "continuing as Local",
                req.rid,
                req.transfer_id,
            )
            req.export_mark = None
            # Its next chunk inserts are the standard ones: the recorded chunk
            # ends no longer describe how its prompt enters the tree.
            req.release_insert_chunk_ends = None
            # Its next chunk insert caches the window kept at its branching point.
            self.export_table.reopen_swa_branch_window(req)
            return
        if reason != RELEASED:
            logger.warning(
                "symmetric-pd export rid=%s transfer_id=%s: %s; freeing leased pages",
                req.rid,
                req.transfer_id,
                reason,
            )
        if req.is_holding_kv:
            # Inserts the (finished) request's prefix into the radix tree and
            # frees the rest, exactly like a Local finish.
            if isinstance(req.finished_reason, FINISH_ABORT):
                release_kv_cache(req, self.tree_cache, is_insert=False)
            else:
                if (
                    req.swa_branching_seqlen is not None
                    and req.swa_branching_seqlen > req.cache_protected_len
                ):
                    # A standard prefill's chunk insert caches the SWA window at
                    # the branching point; the finish insert stops there instead.
                    self.export_table.reopen_swa_branch_window(req)
                    maybe_cache_unfinished_req(req, self.tree_cache)
                release_kv_cache(req, self.tree_cache)

    def _symmetric_import_fallback(self: Scheduler, req: Req, reason: str) -> None:
        """Import failed / timed out: recompute locally."""
        compute_first = (
            getattr(self.export_table, "compute_before_allocation", False) is True
        )
        logger.warning(
            "symmetric-pd import rid=%s transfer_id=%s failed (%s); %s",
            req.rid,
            req.transfer_id,
            reason,
            "ending request" if compute_first else "recomputing locally",
        )
        self.import_queue.release_failed(req)
        if compute_first:
            # This protocol promises either the selected owner's result or a
            # terminal error. A failed/expired allocation must not recompute.
            prepare_abort(
                req,
                f"Compute-before-allocation import failed: {reason}",
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        if req.finished():
            self.output_streamer.stream_output([req], req.return_logprob)
            # release_failed reached cache_finished_req only if the import
            # held KV: end its retention registration count either way.
            forget_kv_session_request(self, req)
            return
        req.output_ids = array("q")
        self._add_request_to_queue(req)

    def _symmetric_materialize_images(self: Scheduler, req: Req) -> bool:
        """Recover original pixels before local prefill, including D retraction."""
        mm_input = getattr(req, "multimodal_inputs", None)
        if mm_input is not None and any(
            item.is_image()
            and "prepared_image_source" in (item.model_specific_data or {})
            for item in mm_input.mm_items
        ):
            recovered = True
            try:
                kimi_items = [
                    item
                    for item in mm_input.mm_items
                    if (item.model_specific_data or {})
                    .get("image_config", {})
                    .get("layout_type")
                    == "kimi_k25"
                ]
                deepseek_items = [
                    item
                    for item in mm_input.mm_items
                    if item.is_image()
                    and (item.model_specific_data or {})
                    .get("image_config", {})
                    .get("layout_type")
                    is None
                ]
                pending = [
                    item
                    for item in mm_input.mm_items
                    if item.is_image() and item.feature is None
                ]
                if any(
                    (item.model_specific_data or {})
                    .get("image_config", {})
                    .get("layout_type")
                    not in (None, "kimi_k25")
                    for item in pending
                ):
                    raise ValueError("Unsupported metadata-only image layout")
                if kimi_items:
                    from decoupled_engine.srt.multimodal.processors.kimi_k25 import (
                        materialize_prepared_images as materialize_kimi_images,
                    )

                    materialize_kimi_images(kimi_items)
                if deepseek_items:
                    from decoupled_engine.srt.multimodal.processors.deepseek_v4_vl import (
                        materialize_prepared_images as materialize_deepseek_images,
                    )

                    materialize_deepseek_images(deepseek_items)
            except Exception as exc:
                recovered = False
                logger.warning(
                    "symmetric-pd image recovery rid=%s failed (%s)",
                    req.rid,
                    type(exc).__name__,
                )
            # All TP ranks must either recompute or abort together, including
            # when just one rank's HTTP retrieval fails.
            agreed = self.import_queue.consensus.agree(
                [_consensus.READY if recovered else _consensus.FAILED]
            )
            if agreed[0] != _consensus.READY:
                prepare_abort(
                    req,
                    "Original images could not be recovered for KV recomputation.",
                    status_code=HTTPStatus.BAD_GATEWAY,
                )
                self.output_streamer.stream_output([req], req.return_logprob)
                # Aborted with no KV, so no cache_finished_req: end the
                # request's retention registration count (soft pin).
                forget_kv_session_request(self, req)
                return False
        return True

    @timed_phase("sym_merge_prebuilt")
    def _symmetric_merge_prebuilt(
        self: Scheduler, reqs: List[Req], running_batch: ScheduleBatch
    ) -> ScheduleBatch:
        """Build the PREBUILT batch for READY imports and merge it into the
        running batch. Adapted from
        SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch /
        get_next_disagg_decode_batch_to_run (decode.py)."""
        from decoupled_engine.srt.observability.req_time_stats import set_time_batch

        # Phases (same names on the standard D loop, decode.py): prebuilt_build,
        # prebuilt_wait_stream, prebuilt_draft_input (process_prebuilt: host
        # data to the GPU on the schedule stream; see its inner prebuilt_*
        # phases), prebuilt_result (decorator), prebuilt_merge.
        with phase("prebuilt_build"):
            for req in reqs:
                # No local prefix reuse yet (see ImportQueue._prealloc TODO):
                # keep the preallocated row as the extend range.
                req.init_next_round_input(None)
                req.set_extend_range(len(req.prefix_indices), req.kv_committed_len)
            set_time_batch(reqs, "set_forward_entry_time")

            new_batch = ScheduleBatch.init_new(
                reqs,
                self.req_to_token_pool,
                self.token_to_kv_pool_allocator,
                self.tree_cache,
                self.model_config,
                self.enable_overlap,
                self.spec_algorithm,
            )
            new_batch.prepare_for_prebuilt()
        if self.enable_overlap:
            with phase("prebuilt_wait_stream"):
                self.schedule_stream.wait_stream(self.forward_stream)
        with phase("prebuilt_draft_input"):
            new_batch.process_prebuilt(self.future_map)
        self.batch_result_processor.process_batch_result_prebuilt(new_batch)
        with phase("prebuilt_merge"):
            new_batch.filter_batch()
            if new_batch.is_empty():
                return running_batch
            if running_batch.is_empty():
                new_batch.batch_is_full = running_batch.batch_is_full
                return new_batch
            running_batch.merge_batch(new_batch)
            return running_batch
