"""Reader-side queue of Import requests in symmetric mode.

Covers the Import lifecycle, chunk publication, and import timeout -> recompute.

Import:  WAITING_REMOTE_KV (pages allocated, READ per published chunk)
         -> all chunks + aux read -> PREBUILT (first token installed) -> DECODE

Requests wait here, outside the batch. Page allocation happens at admission
via the decode prealloc semantics (`kv_committed_len` = prompt length,
`num_reserved_decode_tokens` headroom), reusing
`disaggregation.decode.alloc_for_decode_prealloc`. Admission also allocates
the reader-local aux slot and per-state-type slots and records them on the
request's `ImportMark` (`local_aux_index` / `local_state_indices`) so the
ReaderPlane READs the owner's FINAL items straight into them. On READY the
scheduler installs the first token from that local aux slot and builds a
PREBUILT batch (decode_schedule_batch_mixin).
"""

from __future__ import annotations

import logging
import math
import time
import zlib
from array import array
from contextlib import nullcontext as _nullcontext
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import msgspec

from decoupled_engine.srt.disaggregation.late_d import (
    ADMITTED,
    CONTROL_ADMIT,
    DEFERRED,
    HELD,
    sym_metrics,
)
from decoupled_engine.srt.disaggregation.pull.api import (
    ChunkReady,
    FinalMeta,
    ReaderPlane,
    TransferState,
)
from decoupled_engine.srt.disaggregation.symmetric import consensus as _consensus
from decoupled_engine.srt.disaggregation.symmetric.consensus import RankConsensus
from decoupled_engine.srt.disaggregation.symmetric.export_table import (
    POOL_STATE,
    dst_pages_for_range,
    page_floor,
    swa_tail_start,
)
from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.managers.scheduler_components.phase_timers import phase, timed_phase
from decoupled_engine.srt.mem_cache.cpu_page_book import cat_tagged, copy_tag, slice_tagged

if TYPE_CHECKING:
    from decoupled_engine.srt.disaggregation.utils import (
        MetadataBuffers,
        ReqToMetadataIdxAllocator,
    )
    from decoupled_engine.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

IMPORT_TIMEOUT = "import waiting timeout"


class ImportPreallocRetry(Exception):
    """_prealloc could not get the pages _can_admit promised; retry next tick."""


IMPORT_PEER_FAILED = "import failed on a peer TP rank"

# Late D walk: the admission dimensions of a late
# import, in the order of ImportQueue._late_need_and_budgets, and the misses
# that count as memory-blocked for the claim's prefill reserve (6.4).
LATE_DIMS = ("full", "swa", "budget", "clean")
LATE_MEMORY_MISSES = ("full", "swa", "clean", "prealloc_retry")


def ceil_align(length: int, page_size: int) -> int:
    if page_size <= 1:
        return length
    return (length + page_size - 1) // page_size * page_size


def commit_first_token_from_aux(
    req: Req,
    metadata_buffers: MetadataBuffers,
    idx: int,
    *,
    spec_is_none: bool,
    expected_room: int,
) -> Optional[str]:
    """Install the P-leg's first token and aux into `req` from the local aux
    slot the reader READ FINAL into. Adapted from
    `DecodeTransferQueue._commit_transfer_to_req` (decode.py); returns an error
    string instead of aborting so the caller can take the recompute path."""
    (
        output_id,
        cached_tokens,
        output_token_logprobs_val,
        output_token_logprobs_idx,
        output_top_logprobs_val,
        output_top_logprobs_idx,
        output_token_sampling_mask_len,
        output_token_sampling_mask_idx,
        output_token_sampling_logprobs,
        output_topk_p,
        output_topk_index,
        output_hidden_states,
        output_dsa_topk_indices,
        output_bootstrap_room,
    ) = metadata_buffers.get_buf(idx)

    actual_room = int(output_bootstrap_room[0].item())
    if actual_room == 0:
        return "aux slot not written (room=0) although plane reported READY"
    if actual_room != expected_room:
        return (
            f"aux slot room mismatch: expected transfer_id={expected_room}, "
            f"got {actual_room} (aux index {idx})"
        )

    req.output_ids.append(int(output_id[0].item()))
    req.cached_tokens = int(cached_tokens[0].item())
    # prepare_for_prebuilt adds only reader-side reuse beyond this.
    req.already_computed = req.cached_tokens
    req.cached_tokens_device = int(cached_tokens[1].item())
    req.cached_tokens_host = int(cached_tokens[2].item())
    req.cached_tokens_storage = int(cached_tokens[3].item())
    req.mm_image_tokens = int(cached_tokens[4].item())
    req.mm_audio_tokens = int(cached_tokens[5].item())
    req.mm_video_tokens = int(cached_tokens[6].item())
    if not spec_is_none:
        req.output_topk_p = output_topk_p
        req.output_topk_index = output_topk_index
        req.hidden_states_tensor = output_hidden_states
        if output_dsa_topk_indices is not None and bool(
            (output_dsa_topk_indices < 0).all().item()
        ):
            output_dsa_topk_indices = None
        req.output_dsa_topk_indices = output_dsa_topk_indices
    if req.return_logprob:
        req.logprob.output_token_logprobs_val.append(
            float(output_token_logprobs_val[0].item())
        )
        req.logprob.output_token_logprobs_idx.append(
            int(output_token_logprobs_idx[0].item())
        )
        req.logprob.output_top_logprobs_val.append(
            output_top_logprobs_val[: req.logprob.top_logprobs_num].tolist()
        )
        req.logprob.output_top_logprobs_idx.append(
            output_top_logprobs_idx[: req.logprob.top_logprobs_num].tolist()
        )
    if req.return_sampling_mask:
        if output_token_sampling_mask_idx is None:
            return "sampling mask buffer disabled on the reader side"
        mask_len = int(output_token_sampling_mask_len[0].item())
        if mask_len < 0:
            req.output_token_sampling_mask.append(None)
            req.output_token_sampling_logprobs.append(None)
        else:
            req.output_token_sampling_mask.append(
                output_token_sampling_mask_idx[:mask_len].cpu().tolist()
            )
            req.output_token_sampling_logprobs.append(
                float(output_token_sampling_logprobs[0].item())
            )
    return None


def mark_reused_pages(
    pages: Dict[str, Sequence[int]], token_start: int, token_end: int, skip_before: int
) -> Dict[str, Sequence[int]]:
    """Replace the local page indices that cover tokens [token_start,
    min(skip_before, token_end)) with -1 (per pool, proportionally to the
    pool's pages-per-token; exact when `skip_before` is page-aligned). State
    rows (POOL_STATE) are FINAL items, never chunk pages, and are left alone."""
    cut = min(max(skip_before - token_start, 0), token_end - token_start)
    if cut <= 0:
        return pages
    span = token_end - token_start
    out: Dict[str, Sequence[int]] = {}
    for pool, lst in pages.items():
        n = len(lst)
        if pool == POOL_STATE or n == 0 or span <= 0:
            out[pool] = lst
            continue
        k = (cut * n) // span
        out[pool] = [-1] * k + list(lst[k:])
    return out


def _is_swa_window_tree(tree_cache: Any) -> bool:
    """DSv4 uniform lifecycle with window SWA residency. A uniform tree without
    the residency field predates it and keeps full residency."""
    return bool(getattr(tree_cache, "dsv4_uniform_cache_lifecycle", False)) and (
        getattr(tree_cache, "dsv4_uniform_swa_residency", "full") == "window"
    )


def apply_import_cache_policy(req: Req, policy: str) -> None:
    """Translate the router's `kv_import.cache_policy` into the request flags
    the finish path honours (mem_cache/common.py): `gpu_only` keeps the
    imported prompt in the radix tree without the hicache host write-through
    (upstream PD decode behaviour), `drop` frees it without inserting,
    `retain` (default) follows the engine's write policy."""
    if policy == "gpu_only":
        req.hicache_skip_write_through = True
    elif policy == "drop":
        req.skip_radix_cache_insert = True


class ImportEntry(msgspec.Struct, kw_only=True):
    req: Any  # Req
    transfer_id: int
    added_at: float
    admitted_at: Optional[float] = None
    # Prefix tokens admission started loading back from the host tier.
    host_loaded_tokens: int = 0
    # The owner's FinalMeta as _commit read it (pull.api.FinalMeta). pop_ready
    # hands it to _apply_ready after the second agreement (DSv4 window
    # residency: FinalMeta.owner_peer_kv_addr is the SWA backfill source).
    final_meta: Any = None
    # Late D (--disaggregation-late-d, late_d.LateDDecode; the names of
    # decode.DecodeRequest): None keeps today's path; else DEFERRED ->
    # ADMITTED (-> HELD) -> ALLOCATED. Orders and pass counts are identical
    # on every rank; the times are rank-0 only.
    late_d_state: Optional[str] = None
    late_d_admit_order: int = -1
    late_d_held_since: int = -1
    late_d_skips: int = 0
    late_d_deferred_at: float = 0.0
    late_d_held_at: float = 0.0
    # Stage 2: this D-leg's candidate index; None for a stage-1 leg.
    late_d_cand_idx: Optional[int] = None
    # Rank 0: monotonic receipt of the ADMIT (HELD TTL, reader bind latency)
    # and the reader's last iteration when it applied it.
    late_d_admitted_at: float = 0.0
    late_d_admit_iter: str = ""


class ImportQueue:
    """Reader side of the Import lifecycle for one scheduler rank."""

    # Late D (enable_late_d): None while off, also on a bare instance.
    late_d: Any = None
    late_port: Any = None
    # SGLANG_SYMMETRIC_ADMIT_AFTER_LAUNCH (symmetric/admit_after_launch.py):
    # True once the overlap loop owns the walk; then neither add() nor
    # symmetric_pre_schedule calls admit_pending. The loop sets it False for
    # a pass whose walk runs in symmetric_pre_schedule (the escape). False on
    # a bare instance.
    defer_admission: bool = False

    def __init__(
        self,
        reader_plane: ReaderPlane,
        *,
        # Narrow scheduler accessors instead of the Scheduler itself.
        num_active_reqs: Callable[[], int],
        stream_output: Callable[[List[Req], bool], None],
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        tree_cache: Any,
        metadata_buffers: Optional[MetadataBuffers],
        req_to_metadata_buffer_idx_allocator: Optional[ReqToMetadataIdxAllocator],
        num_reserved_decode_tokens: int,
        max_total_num_tokens: int,
        page_size: int,
        spec_is_none: bool = True,
        waiting_timeout_s: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
        rank_consensus: Optional[RankConsensus] = None,
        prefix_reuse: Optional[bool] = None,
        sliding_window_size: Optional[int] = None,
        swa_tail_import: Optional[bool] = None,
        local_reserve: Optional[Callable[[], int]] = None,
        budget_fraction: Optional[float] = None,
    ) -> None:
        self.reader_plane = reader_plane
        # Admission liveness (cross-worker import wedge), both off by default:
        # an import preallocates its whole prompt and then waits for the peer's
        # P-leg, so two workers whose pools fill up with such imports can run
        # neither P-leg. F-A (local_reserve, SGLANG_SYMMETRIC_PD_IMPORT_LOCAL_RESERVE):
        # the KV the local in-flight chunked prefill still has to allocate
        # (SchedulerSymmetricMixin.symmetric_local_inflight_tokens) is kept
        # out of the admission budget, so a started P-leg always finishes.
        # None = off.
        self.local_reserve = local_reserve
        # F-B (SGLANG_SYMMETRIC_PD_IMPORT_BUDGET_FRACTION): ACTIVE imports may hold at
        # most budget_fraction * max_total_num_tokens (each counted as
        # _required_tokens: whole prompt + decode headroom), so (1 - f) of the
        # pool stays for P-legs to start. An import above the cap is admitted
        # only when no other import is active. 0 = off.
        self.budget_fraction = (
            float(envs.SGLANG_SYMMETRIC_PD_IMPORT_BUDGET_FRACTION.get())
            if budget_fraction is None
            else float(budget_fraction)
        )
        # Admission attempts held back by F-B / F-A (per _can_admit call).
        self.stats_budget_held = 0
        self.stats_reserve_held = 0
        self._held_logged_at: Optional[float] = None
        # Reader-side prefix reuse (upstream PD `decode_prefix_len`): the
        # D-leg prompt is matched against this engine's radix tree, the
        # matched pages are locked and reused, and only the missing suffix is
        # pre-allocated and READ. The full-prompt import kept the D-role pool
        # at 94-99 % and made every import pay for the whole ~75k-token prompt.
        self.prefix_reuse = (
            bool(envs.SGLANG_SYMMETRIC_PD_IMPORT_PREFIX_REUSE.get())
            if prefix_reuse is None
            else prefix_reuse
        )
        self.stats_prefix_reuse_requests = 0
        self.stats_prealloc_retries = 0
        self.stats_prefix_reuse_tokens = 0
        self.stats_imported_tokens = 0
        self.stats_swa_tail_requests = 0
        # SWA tokens of imported prompts that were neither allocated nor READ.
        self.stats_swa_tail_skipped_tokens = 0
        # Reused prefix tokens loaded back from the host tier instead of READ.
        self.stats_host_load_back_tokens = 0
        # Host hits READ from the owner because the load ring slot was busy.
        self.stats_host_load_back_skipped = 0
        # DSv4 window residency: gpu_only policies coerced to retain, and the
        # SWA backfills submitted at commit / skipped (backfill off, or no
        # owner servicer address in FINAL).
        self.stats_cache_policy_overridden = 0
        self.stats_swa_backfill_submitted = 0
        self.stats_swa_backfill_skipped = 0
        # TP ranks must commit/fail the same imports in the same tick
        # (symmetric/consensus.py).
        self.consensus = rank_consensus or _consensus.local_consensus()
        # Late D (enable_late_d; None = off): the walk core
        # (late_d.LateDDecode), its reader port (late_d_sym.SymReaderPort)
        # and the aux slots imports leave to bound P-legs.
        self.late_d: Any = None
        self.late_port: Any = None
        self.export_aux_reserve = 0
        self.late_metrics: Any = None
        self.stats_late_miss: Dict[str, int] = {}
        self.stats_late_gate: Dict[str, int] = {}
        self.stats_late_ended: Dict[str, int] = {}
        self.stats_claim_skips = 0
        self.num_active_reqs = num_active_reqs
        self.stream_output = stream_output
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.tree_cache = tree_cache
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.num_reserved_decode_tokens = num_reserved_decode_tokens
        self.max_total_num_tokens = max_total_num_tokens
        self.page_size = page_size
        self.spec_is_none = spec_is_none
        self.waiting_timeout_s = waiting_timeout_s
        self.clock = clock
        # Not yet allocated (budget / slots), FIFO.
        self.pending: List[ImportEntry] = []
        # Allocated and begin_import()ed; waiting for READY.
        self.active: Dict[int, ImportEntry] = {}
        # D-role SWA-tail import (closes a capacity gap): traditional
        # decode keeps SWA only for the live window (--swa-full-tokens-ratio
        # 0.1 -> 6.2x more full-attention tokens than a uniform-lifecycle
        # symmetric engine). When this engine has that pool shape, allocate
        # and READ SWA pages only for the sliding-window tail of an imported
        # prompt, exactly like upstream `DecodePreallocQueue` with
        # `alloc_extend_swa_tail`; the pages below the tail are reported -1 in
        # the page plan and the request starts with `kv.swa_evicted_seqlen`
        # at the tail start. Auto: on iff the allocator has the SWA-tail
        # capability, a sliding window is known and the tree is either not the
        # uniform DSv4 lifecycle, or runs it with window SWA residency. Uniform
        # lifecycle with full residency keeps every SWA page on the device and
        # never evicts it: off.
        self.sliding_window_size = sliding_window_size
        want = (
            bool(envs.SGLANG_SYMMETRIC_PD_IMPORT_SWA_TAIL.get())
            if swa_tail_import is None
            else bool(swa_tail_import)
        )
        self.swa_tail_import = want and self._supports_swa_tail()
        # Window residency reader: tail-only SWA like the
        # upstream decode shape, but the imported prompt is still inserted.
        # The prefix reuse matches and locks FULL only (capped at the tail
        # start T), the SWA below T is backfilled host->host from the owner's
        # servicer into the request's ledger, and the finish insert adopts it.
        self.swa_window = self.swa_tail_import and _is_swa_window_tree(tree_cache)
        if _is_swa_window_tree(tree_cache) and not self.swa_tail_import:
            logger.warning(
                "symmetric-pd import: window SWA residency without SWA-tail import "
                "(SGLANG_SYMMETRIC_PD_IMPORT_SWA_TAIL=0 or no tail-capable allocator): "
                "window owners announce no SWA below the export floor, so imports "
                "from them fail with layout_mismatch and recompute locally"
            )
        if self.swa_window:
            logger.info(
                "symmetric-pd import: SWA-tail import ON with window residency "
                "(window=%d page=%d): the tail's SWA is READ, the SWA below it "
                "is backfilled from the owner's servicer, prompts are inserted",
                sliding_window_size,
                page_size,
            )
        elif want and not self.swa_tail_import:
            logger.info(
                "symmetric-pd import: SWA-tail import off (allocator=%s uniform_lifecycle=%s "
                "sliding_window_size=%s page_size=%d)",
                type(token_to_kv_pool_allocator).__name__,
                bool(getattr(tree_cache, "dsv4_uniform_cache_lifecycle", False)),
                sliding_window_size,
                page_size,
            )
        elif self.swa_tail_import:
            logger.info(
                "symmetric-pd import: SWA-tail import ON (window=%d page=%d): imported prompts "
                "get SWA pages for the tail only; content-only prefix reuse=%s",
                sliding_window_size,
                page_size,
                self._content_only_reuse(),
            )

    def _supports_swa_tail(self) -> bool:
        alloc = self.token_to_kv_pool_allocator
        if (
            self.page_size <= 1
            or not self.sliding_window_size
            or self.sliding_window_size <= 0
        ):
            return False
        if getattr(
            self.tree_cache, "dsv4_uniform_cache_lifecycle", False
        ) and not _is_swa_window_tree(self.tree_cache):
            return False
        if type(alloc).__name__.startswith("PureSWA"):
            return False
        if not (
            hasattr(alloc, "alloc_extend_swa_tail")
            and hasattr(alloc, "swa_attn_allocator")
            and hasattr(alloc, "full_available_size")
            and hasattr(alloc, "swa_available_size")
        ):
            return False
        get_kv = getattr(alloc, "get_kvcache", None)
        kvcache = get_kv() if callable(get_kv) else None
        # DSv4 unified_kv keeps SWA in a per-request ring, not in pages.
        if kvcache is not None and getattr(kvcache, "_unified_kv", False):
            return False
        return True

    def _content_only_reuse(self) -> bool:
        """SWA-tail import reuses and publishes only the FULL component of an
        imported prompt, like upstream PD decode's `content_only` radix cache:
        the SWA tail is always READ fresh, so the reused prefix never needs SWA.
        Needs a tree core with partial-component match and insert. Never under
        DSv4 window residency, whose readers insert every component (the
        uniform lifecycle keeps partial-component inserts off anyway)."""
        tree_core = getattr(self.tree_cache, "tree_core", None)
        return (
            self.swa_tail_import
            and not getattr(self, "swa_window", False)
            and self._tree_supports_reuse()
            and bool(getattr(tree_core, "supports_partial_component_cache", False))
        )

    def _swa_tail_len(self, seq_len: int) -> int:
        """SWA tokens to allocate/READ for a prompt of `seq_len` tokens: the
        radix-cache form of `DecodePreallocQueue._swa_tail_len`, one page longer
        than the live window so the last page is never all-tombstone. Content-only
        readers need it too: they publish the prompt with this tail as its SWA
        checkpoint (`decode_radix_cache_frozen_swa_floor`). The owner always keeps
        this form."""

        if not self.swa_tail_import or seq_len <= 0:
            return max(seq_len, 0)
        return seq_len - swa_tail_start(
            seq_len,
            sliding_window_size=self.sliding_window_size,
            page_size=self.page_size,
        )

    def _tree_evictable(self, pool: str) -> int:
        """Evictable tokens of the radix tree for `pool` ("full" | "swa").
        SWA trees expose full_/swa_evictable_size(); some return a tuple from
        evictable_size()."""
        tc = self.tree_cache
        fn = getattr(tc, f"{pool}_evictable_size", None)
        if (
            pool == "swa"
            and getattr(tc, "dsv4_uniform_swa_residency", "full") == "window"
        ):
            # DSv4 window residency: unbacked tree SWA cannot be evicted yet.
            fn = tc.swa_admission_evictable_size
        if callable(fn):
            try:
                return int(fn())
            except Exception:
                pass
        v = tc.evictable_size()
        if isinstance(v, (tuple, list)):
            return int(v[0] if pool == "full" else v[-1])
        return int(v)

    # -- late D -----------------------------------------------------------
    def enable_late_d(self, late_d: Any, port: Any, *, aux_reserve: int) -> None:
        """Late D (S-1a): imports wait DEFERRED for the owner's ADMIT and are
        allocated by the late walk; see symmetric/late_d_sym.py."""
        self.late_d = late_d
        self.late_port = port
        self.export_aux_reserve = max(0, int(aux_reserve))
        self.late_metrics = late_d.metrics

    def apply_late_control(self, control: Any, iter_type: str = "") -> None:
        """Every rank, from the request broadcast."""
        entry = None
        if control.kind == CONTROL_ADMIT:
            entry = next(
                (e for e in self.pending if e.transfer_id == control.room), None
            )
        before = entry.late_d_state if entry is not None else None
        self.late_d.apply_control(control, self.pending)
        if entry is not None and before != entry.late_d_state:
            entry.late_d_admit_iter = iter_type
            if entry.late_d_state == ADMITTED and control.kv_session is not None:
                self._register_admit_kv_session(entry.req, control.kv_session)

    def _kv_sessions(self) -> Any:
        """The tree's retention registry; None without
        --enable-kv-session-retention."""
        return getattr(self.tree_cache, "kv_sessions", None)

    @timed_phase("kv_session.forget")
    def _forget_kv_session(self, req: Req) -> None:
        """A D leg that ends here without KV never reaches
        cache_finished_req: end its retention registration count (soft pin).
        Same as session_ref_tracker.forget_kv_session_request; untagged
        requests and repeats change nothing."""
        tracker = self._kv_sessions()
        if tracker is not None:
            tracker.refresh_request(req, finished=True)

    @timed_phase("kv_session.register")
    def _register_admit_kv_session(self, req: Req, tag: Tuple[str, int]) -> None:
        """Every rank, in the ADMIT's iteration:
        register the admitted D leg with the owner's P-leg tag. Loser
        candidates never get an ADMIT, so they never register. A D leg that
        arrived tagged (the router tagged its body) keeps that registration.
        An admit error (capacity, version reused) is counted by the registry
        and the leg runs untagged: the D never aborts for it. A tag the
        registry fences (an older version, an ended session) binds the leg to
        no live registration, as on any worker. stats.admit_tags counts live
        registrations only. All ranks have the same registry state, so all
        take the same branch.

        There is no D ledger yet: on a worker that also
        holds the agent's P registration at this version, the D leg joins it.
        A run that pins P-only and D-only workers never has both."""
        tracker = self._kv_sessions()
        if tracker is None or getattr(req, "kv_session", None) is not None:
            return
        req.kv_session = {"session_id": tag[0], "version": tag[1]}
        # An untagged arrival may have been seen already (admit returned None).
        req._kv_session_arrival_seen = False
        try:
            self.tree_cache.note_kv_session_arrival(req)
        except ValueError as exc:
            req.kv_session = None
            logger.warning(
                "symmetric-pd late import rid=%s: ADMIT kv_session tag not registered "
                "(%s); the D leg runs untagged",
                req.rid,
                exc,
            )
            return
        if tracker._live(getattr(req, "_kv_session_registration", None)):
            tracker.stats.admit_tags += 1

    def late_end(
        self,
        entry: ImportEntry,
        *,
        err_type: str,
        message: str,
        notify_owner: bool,
        status_code: HTTPStatus = HTTPStatus.SERVICE_UNAVAILABLE,
    ) -> None:
        """Every rank: end a never-allocated late entry with HTTP 503 (stage 2
        RELEASE: 409) and no recompute (rule X). ``notify_owner``: ABORT to
        the owner (TTL, mode mismatch); else the owner ended it already
        or released this candidate."""
        from decoupled_engine.srt.disaggregation.utils import prepare_abort

        self.pending = [e for e in self.pending if e is not entry]
        req = entry.req
        mark = req.import_mark
        if mark is not None:
            if notify_owner:
                self.reader_plane.abort_pending(mark, err_type)
            else:
                self.reader_plane.release_pending(mark, err_type)
        if self.late_port is not None:
            self.late_port.note_ended(entry.transfer_id)
        self.stats_late_ended[err_type] = self.stats_late_ended.get(err_type, 0) + 1
        prepare_abort(req, message, status_code=status_code, err_type=err_type)
        self.stream_output([req], req.return_logprob)
        # Never allocated, so no cache_finished_req: a tagged D leg (DEFERRED
        # expiry, RELEASE, owner abort, mode mismatch, HELD expiry) ends its
        # registration count here.
        self._forget_kv_session(req)

    def claim_prefill_reserve(self, req: Req) -> int:
        """The tokens a new prefill must leave
        free for a memory-blocked claim. A request without an export mark
        (Local, recompute) is charged the claim's need; a P-leg nothing
        until the claim has been memory-blocked for claim_escalate_passes
        passes, then at most the F-B cap minus the active imports (R5-b)."""
        late = self.late_d
        if late is None or late.claim is None or not late.claim_memory_blocked:
            return 0
        need = late.claim_reserve()
        if not need:
            return 0
        need_tokens = int(need[0])
        if req.export_mark is None:
            return need_tokens if late.cfg.claim_holds_local else 0
        passes = late.cfg.claim_escalate_passes
        budget = self._budget_tokens()
        if passes <= 0 or late.claim_blocked_passes < passes or budget is None:
            return 0
        return max(0, min(need_tokens, budget - self.active_import_tokens()))

    def note_claim_skip(self) -> None:
        self.stats_claim_skips += 1
        metrics = sym_metrics(self.late_metrics)
        if metrics is not None:
            metrics.claim_skips.inc()

    def load_view(self) -> Dict[str, int]:
        """The import half of /v1/loads ``symmetric``:
        late entries waiting for ADMIT (deferred); the other waiting entries
        (bound: ADMITTED, HELD, or not late); the preallocated ones (active);
        the F-B cap (0: off), the F-A reserve and the claim's need."""
        deferred_reqs = deferred_tokens = bound_reqs = bound_tokens = 0
        for entry in self.pending:
            tokens = len(entry.req.origin_input_ids)
            if entry.late_d_state == DEFERRED:
                deferred_reqs += 1
                deferred_tokens += tokens
            else:
                bound_reqs += 1
                bound_tokens += tokens
        claim = self.late_d.claim_reserve() if self.late_d is not None else None
        return {
            "import_deferred_reqs": deferred_reqs,
            "import_deferred_tokens": deferred_tokens,
            "import_bound_reqs": bound_reqs,
            "import_bound_tokens": bound_tokens,
            "import_active_reqs": len(self.active),
            "import_active_tokens": self.active_import_tokens(),
            "import_budget_tokens": self._budget_tokens() or 0,
            "local_reserve_tokens": self._local_reserve_tokens(),
            "claim_tokens": int(claim[0]) if claim else 0,
        }

    # -- admission ------------------------------------------------------
    @timed_phase("sym.import.add")
    def add(self, req: Req) -> None:
        mark = req.import_mark
        assert mark is not None
        prompt_len = len(req.origin_input_ids)
        if mark.expected_tokens is not None and mark.expected_tokens != prompt_len:
            logger.warning(
                "symmetric-pd import rid=%s transfer_id=%d: kv_import.expected_tokens=%d "
                "!= prompt length %d; using the prompt length",
                req.rid,
                mark.transfer_id,
                mark.expected_tokens,
                prompt_len,
            )
        if prompt_len > self.max_total_num_tokens:
            from decoupled_engine.srt.disaggregation.utils import prepare_abort

            message = (
                f"Request {req.rid} exceeds the maximum number of tokens: "
                f"{prompt_len} > {self.max_total_num_tokens}"
            )
            logger.error(message)
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.stream_output([req], req.return_logprob)
            # Rejected before any KV: end its retention registration count.
            self._forget_kv_session(req)
            return
        req.bootstrap_room = mark.transfer_id
        entry = ImportEntry(
            req=req, transfer_id=mark.transfer_id, added_at=self.clock()
        )
        self.pending.append(entry)
        if self.late_d is not None:
            # Late D: DEFERRED until the owner's ADMIT; the walk
            # allocates it. Register with the owner now: a DEFERRED
            # entry may never reach begin_import, where it happens otherwise.
            self.late_d.defer(entry)
            request_register = getattr(self.reader_plane, "request_register", None)
            if callable(request_register):
                request_register(mark.owner_addr)
            return
        if self.defer_admission:
            # The loop's walk of this pass admits it (admit_after_launch.py).
            return
        self.admit_pending()

    @timed_phase("sym.import.admit")
    def admit_pending(self) -> int:
        """Allocate pages for waiting imports while the prealloc budget allows.
        Returns the number admitted this call."""
        if self.late_d is not None:
            return self._admit_pending_late()
        admitted = 0
        while self.pending:
            entry = self.pending[0]
            if not self._can_admit(entry.req):
                break
            self.pending.pop(0)
            try:
                entry.host_loaded_tokens = self._prealloc(entry.req)
            except ImportPreallocRetry as e:
                # pages vanished between _can_admit and the allocation (tree /
                # allocator raced with the running batch): keep the entry at the
                # head of the queue and try again next tick instead of dying
                self.pending.insert(0, entry)
                self.stats_prealloc_retries += 1
                if (
                    self.stats_prealloc_retries <= 5
                    or self.stats_prealloc_retries % 100 == 0
                ):
                    logger.warning(
                        "symmetric-pd import rid=%s: prealloc deferred (%s); retries=%d",
                        entry.req.rid,
                        e,
                        self.stats_prealloc_retries,
                    )
                break
            entry.admitted_at = self.clock()
            self.active[entry.transfer_id] = entry
            with phase("sym.import.begin"):
                self.reader_plane.begin_import(
                    entry.req.import_mark, entry.req.rid, self._dst_pages_fn(entry.req)
                )
            admitted += 1
        return admitted

    def _admit_pending_late(self) -> int:
        """Late walk: every rank, in
        lockstep. DEFERRED entries are skipped; the claim first, then ADMIT
        order. A miss in any admission dimension (or a prealloc retry) holds
        the entry and continues; the request-slot and aux-reserve gates stop
        the walk (skipping cannot help). Entries that are not late (none
        exist while late D is on) keep the FIFO rule."""
        late = self.late_d
        admitted = 0
        claim_need = None
        taken = set()
        allocated: List[int] = []
        for i, entry in late.walk_order(self.pending):
            req = entry.req
            gate = self._late_slot_gate()
            if gate is not None:
                self.stats_late_gate[gate] = self.stats_late_gate.get(gate, 0) + 1
                break
            need, budgets = self._late_need_and_budgets(req)
            before = budgets[0]
            if entry.late_d_state is None:
                if not self._can_admit(req):
                    break
            elif not late.admit_fits(entry, need, claim_need, *budgets):
                reason = self._miss_reason(need, budgets)
                self.stats_late_miss[reason] = self.stats_late_miss.get(reason, 0) + 1
                claim_need = late.on_no_fit(entry, need, claim_need, reason=reason)
                continue
            try:
                entry.host_loaded_tokens = self._prealloc(req)
            except ImportPreallocRetry as e:
                self.stats_prealloc_retries += 1
                if entry.late_d_state is None:
                    break
                logger.debug("symmetric-pd late import rid=%s: %s; held", req.rid, e)
                claim_need = late.on_no_fit(
                    entry, need, claim_need, reason="prealloc_retry"
                )
                continue
            entry.admitted_at = self.clock()
            self.active[entry.transfer_id] = entry
            with phase("sym.import.begin"):
                self.reader_plane.begin_import(
                    req.import_mark, req.rid, self._dst_pages_fn(req)
                )
            if entry.late_d_state is not None:
                late.on_alloc(entry, need[0], before, self._allocatable_tokens())
                self._note_late_alloc(entry)
            taken.add(i)
            allocated.append(entry.transfer_id)
            admitted += 1
        if taken:
            self.pending = [e for j, e in enumerate(self.pending) if j not in taken]
        late.end_pass(claim_need, LATE_MEMORY_MISSES)
        self._walk_digest(allocated)
        return admitted

    def _late_slot_gate(self) -> Optional[str]:
        if self.req_to_token_pool.available_size() <= 0:
            return "no_req_slot"
        alloc = self.req_to_metadata_buffer_idx_allocator
        if alloc is not None and alloc.available_size() <= self.export_aux_reserve:
            # The last aux slots stay for bound P-legs.
            return "no_aux"
        return None

    def _late_need_and_budgets(
        self, req: Req
    ) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
        """(need, budget) per LATE_DIMS: full tokens (net of F-A), SWA tokens
        (SWA-tail import), the F-B cap left (inf with no other import active:
        an import above the cap is admitted alone), clean pages (M4)."""
        required = self._required_tokens(req)
        clean = self._allocatable_clean_tokens()
        need_swa = self._required_swa_tokens(req) if self.swa_tail_import else 0
        need_clean = (
            ceil_align(len(req.origin_input_ids), self.page_size)
            if clean is not None
            else 0
        )
        budget = self._budget_tokens()
        budget_left = (
            budget - self.active_import_tokens()
            if budget is not None and self.active
            else math.inf
        )
        swa = self._allocatable_swa_tokens() if self.swa_tail_import else math.inf
        need = (required, need_swa, required, need_clean)
        budgets = (
            self._allocatable_tokens(),
            swa,
            budget_left,
            clean if clean is not None else math.inf,
        )
        return need, budgets

    @staticmethod
    def _miss_reason(need: Tuple[int, ...], budgets: Tuple[float, ...]) -> str:
        for name, n, b in zip(LATE_DIMS, need, budgets):
            if n > b:
                return name
        return "claim_reserve"

    def _note_late_alloc(self, entry: ImportEntry) -> None:
        """The key left the late states (time stats on every rank). Rank 0:
        the reader's part of the bind latency (ADMIT receipt -> BEGIN), by
        its iteration type."""
        if self.late_port is not None:
            self.late_port.note_ended(entry.transfer_id)
        stats = entry.req.time_stats
        stats.late_d_alloc_time = time.perf_counter()
        stats.late_d_admit_iter = entry.late_d_admit_iter
        metrics = sym_metrics(self.late_metrics)
        if (
            metrics is not None
            and entry.late_d_admitted_at > 0
            and entry.late_d_admitted_at != math.inf
        ):
            metrics.reader_bind.labels(
                **metrics.labels, iter=entry.late_d_admit_iter or "idle"
            ).observe(max(0.0, time.monotonic() - entry.late_d_admitted_at))

    def _walk_digest(self, allocated: List[int]) -> None:
        """SGLANG_LATE_D_WALK_DIGEST_EVERY (debug): every N walk passes all
        ranks compare a digest of the walk; a mismatch fails loudly (K1)."""
        late = self.late_d
        every = late.cfg.walk_digest_every
        if every <= 0 or late.pass_index % every:
            return
        held = sorted(e.transfer_id for e in self.pending if e.late_d_state == HELD)
        claim = late.claim.transfer_id if late.claim is not None else -1
        blob = repr((late.pass_index, sorted(allocated), held, claim)).encode()
        digest = zlib.crc32(blob) & 0x7FFFFFFF
        low, neg_high = self.consensus.agree([digest, -digest])
        if low != digest or -neg_high != digest:
            raise RuntimeError(
                f"late D walk digest mismatch across TP ranks at pass "
                f"{late.pass_index}: this rank {digest}, min {low}, max {-neg_high}"
            )

    def _required_tokens(self, req: Req) -> int:
        # Admission estimate: pages for the WHOLE prompt plus the decode
        # headroom. Deliberately ignores the reusable local prefix: those pages
        # are (mostly) evictable and already counted in _allocatable_tokens(),
        # and _prealloc locks them, so subtracting them here counted them twice
        # and let _prealloc run out of pages.
        return (
            ceil_align(len(req.origin_input_ids), self.page_size)
            + self.num_reserved_decode_tokens
        )

    def _tree_supports_reuse(self) -> bool:
        tc = self.tree_cache
        if not self.prefix_reuse or tc is None:
            return False
        if getattr(tc, "disable", False):
            return False
        is_chunk = getattr(tc, "is_chunk_cache", None)
        if callable(is_chunk) and is_chunk():
            return False
        return hasattr(tc, "match_prefix") and hasattr(tc, "inc_lock_ref")

    @timed_phase("sym.import.prealloc.match")
    def _match_local_prefix(self, req: Req) -> int:
        """Match the D-leg prompt against the local radix tree (no lock).
        Sets `req.prefix_indices` / `req.last_node` like the prefill path and
        returns the page-aligned device prefix length to reuse; 0 when reuse
        is off or nothing matches. Always leaves at least one page to import
        so the reader has a non-empty extend range and the owner's FINAL
        (aux, state rows) still applies to the tail."""
        import torch

        if not self._tree_supports_reuse():
            return 0
        from decoupled_engine.srt.managers.schedule_policy import match_prefix_for_req

        prompt_len = len(req.origin_input_ids)
        match_kwargs = {}
        reuse_limit = None
        if self._ordinary_l2_restore_enabled():
            from decoupled_engine.srt.disaggregation.decode_hicache_mixin import (
                _tree_uses_bigram_keys,
            )

            # Limit raw tokens before matching, so host restore cannot consume
            # the nonempty FINAL tail. EAGLE needs N+1 raw tokens for N KV rows.
            reuse_limit = page_floor(max(prompt_len - 1, 0), self.page_size)
            match_kwargs["key_limit"] = reuse_limit + int(
                _tree_uses_bigram_keys(self.tree_cache)
            )
        if self.swa_window:
            # Window residency: the D-leg reads SWA only
            # from its received tail [T, L) on, so reuse matches FULL only
            # (the prefix's SWA may be host-only) and stops at T, keeping the
            # tail's SWA pages this request's own.
            from decoupled_engine.srt.mem_cache.unified_cache.component_type import (
                ComponentType,
            )

            reuse_limit = prompt_len - self._swa_tail_len(prompt_len)
            match_kwargs = dict(
                active_components=(ComponentType.FULL,), key_limit=reuse_limit
            )
        try:
            supports_mamba = getattr(self.tree_cache, "supports_mamba", None)
            match_result = match_prefix_for_req(
                self.tree_cache,
                req,
                req.origin_input_ids,
                cow_mamba=bool(supports_mamba()) if callable(supports_mamba) else False,
                include_req=True,
                **match_kwargs,
            )
            if envs.SGLANG_DECODE_PREFIX_OBSERVATION.get():
                # These are existing scalar/shape metadata, not GPU reads.
                # Host availability is observed here but is not promised,
                # locked, or restored by this device-only import path.
                req._decode_prefix_match_observation = (
                    len(match_result.device_indices),
                    match_result.host_hit_length,
                )
        except Exception as e:  # never let reuse break admission
            logger.warning(
                "symmetric-pd import rid=%s: local prefix match failed (%r); importing all",
                req.rid,
                e,
            )
            req.prefix_indices = torch.empty((0,), dtype=torch.int64)
            req.last_node = None
            if self._ordinary_l2_restore_enabled():
                req.host_hit_length = 0
                req.best_match_node = None
            return 0
        n = len(req.prefix_indices)
        n = (n // self.page_size) * self.page_size
        max_reuse = ((prompt_len - 1) // self.page_size) * self.page_size
        if reuse_limit is not None:
            max_reuse = min(max_reuse, reuse_limit)
        n = max(0, min(n, max_reuse))
        if n < len(req.prefix_indices):
            # M4 Q4: keeps the match result's CPU page ids (plain slice when off).
            req.prefix_indices = slice_tagged(req.prefix_indices, 0, n, self.page_size)
        return n

    def _allocatable_tokens(self) -> int:
        """Adapted from `DecodePreallocQueue._allocatable_token_budgets`:
        free pages (+ evictable radix pages) minus the decode headroom every
        active request is promised. In SWA-tail mode this is the FULL pool
        budget (the SWA pool is checked separately by _allocatable_swa_tokens);
        otherwise the hybrid allocator's available_size() is min(full, swa)."""
        n_active = self.num_active_reqs() + len(self.active)
        if self.swa_tail_import:
            available = self.token_to_kv_pool_allocator.full_available_size()
            available += self._tree_evictable("full")
        else:
            available = self.token_to_kv_pool_allocator.available_size()
            available += int(self.tree_cache.evictable_size())
        return (
            available
            - self.num_reserved_decode_tokens * n_active
            - self._local_reserve_tokens()
        )

    def _local_reserve_tokens(self) -> int:
        """F-A: KV the local in-flight chunked prefill still needs (0 when off).
        Charged to the full-token budget only; the SWA budget is unchanged."""
        if self.local_reserve is None:
            return 0
        return max(0, int(self.local_reserve()))

    def _budget_tokens(self) -> Optional[int]:
        """F-B cap on the tokens of active imports; None when off."""
        if self.budget_fraction <= 0:
            return None
        return int(self.budget_fraction * self.max_total_num_tokens)

    def active_import_tokens(self) -> int:
        """Tokens the active (preallocated, not yet READY-committed) imports
        hold, counted like admission counts them (_required_tokens)."""
        return sum(self._required_tokens(e.req) for e in self.active.values())

    def _within_budget(self, req: Req) -> bool:
        """F-B. Off, or no other import active (an import above the cap is
        admitted alone, so it cannot starve), or the active imports plus this
        one stay within the cap."""
        budget = self._budget_tokens()
        if budget is None or not self.active:
            return True
        return self.active_import_tokens() + self._required_tokens(req) <= budget

    def _note_held(self, why: str, req: Req, required: int) -> None:
        """Count an admission held back by F-B / F-A; log at most every 30 s."""
        if why == "budget":
            self.stats_budget_held += 1
        else:
            self.stats_reserve_held += 1
        now = self.clock()
        if self._held_logged_at is not None and now - self._held_logged_at < 30.0:
            return
        self._held_logged_at = now
        logger.info(
            "symmetric-pd import admission held (%s): rid=%s required=%d active=%d "
            "active_tokens=%d budget=%s local_reserve=%d pending=%d "
            "held_budget=%d held_reserve=%d",
            why,
            req.rid,
            required,
            len(self.active),
            self.active_import_tokens(),
            self._budget_tokens(),
            self._local_reserve_tokens(),
            len(self.pending),
            self.stats_budget_held,
            self.stats_reserve_held,
        )

    def _required_swa_tokens(self, req: Req) -> int:
        return (
            ceil_align(self._swa_tail_len(len(req.origin_input_ids)), self.page_size)
            + self.num_reserved_decode_tokens
        )

    def _allocatable_swa_tokens(self) -> int:
        n_active = self.num_active_reqs() + len(self.active)
        available = self.token_to_kv_pool_allocator.swa_available_size()
        available += self._tree_evictable("swa")
        return available - self.num_reserved_decode_tokens * n_active

    @timed_phase("sym.import.can_admit")
    def _can_admit(self, req: Req) -> bool:
        if self.req_to_token_pool.available_size() <= 0:
            return False
        if (
            self.req_to_metadata_buffer_idx_allocator is not None
            and self.req_to_metadata_buffer_idx_allocator.available_size() <= 0
        ):
            return False
        required = self._required_tokens(req)
        if required > self._allocatable_tokens():
            if (
                self.local_reserve is not None
                and required
                <= self._allocatable_tokens() + self._local_reserve_tokens()
            ):
                self._note_held("local_reserve", req, required)
            return False
        if (
            self.swa_tail_import
            and self._required_swa_tokens(req) > self._allocatable_swa_tokens()
        ):
            return False
        clean = self._allocatable_clean_tokens()
        if clean is not None and (
            ceil_align(len(req.origin_input_ids), self.page_size) > clean
        ):
            # M4: the prompt's pages must be clean (alloc_import); wait here
            # for promotions instead of failing in _prealloc.
            return False
        if not self._within_budget(req):
            self._note_held("budget", req, required)
            return False
        return True

    # -- M4 (Q4 CPU page bookkeeping + load quarantine) ------------------
    def _q4_rows(self):
        """The request pool's CPU page rows when the allocator keeps M4's CPU
        page bookkeeping (mem_cache/cpu_page_book.py), else None: flags off,
        every path below is the legacy one. Looked up per call, since the
        loop converts the allocator after this queue is built."""
        from decoupled_engine.srt.mem_cache.allocator.cpu_paged import cpu_paged_allocator
        from decoupled_engine.srt.mem_cache.cpu_page_book import cpu_rows_of

        if cpu_paged_allocator(self.token_to_kv_pool_allocator) is None:
            return None
        rows = cpu_rows_of(self.req_to_token_pool)
        if rows is None:
            raise AssertionError(
                "Q4 page bookkeeping: a CPU-paged allocator without CPU rows on "
                f"the request pool ({type(self.req_to_token_pool).__name__})"
            )
        return rows

    def _allocatable_clean_tokens(self) -> Optional[int]:
        """With the M4 load quarantine: clean free pages plus the evictable
        radix pages, all an import's clean-only prealloc can use
        (_q4_evict_for_import sizes its eviction the same way). None when the
        quarantine is off. CPU counters only (no device read)."""
        from decoupled_engine.srt.mem_cache.allocator.cpu_paged import (
            cpu_paged_allocator,
            load_quarantine_active,
        )

        alloc = cpu_paged_allocator(self.token_to_kv_pool_allocator)
        if alloc is None or not load_quarantine_active(alloc):
            return None
        return alloc.clean_available_size() + int(self.tree_cache.evictable_size())

    def _q4_evict_for_import(self, required: int) -> None:
        """Eviction ahead of an M4 prealloc. The READ lands in the pages
        outside every CUDA stream, so alloc_import hands out clean pages only;
        size the eviction on them, as a foreground load-back does
        (UnifiedRadixCache._load_target_available_size): cold victims are
        clean. A victim whose unlock epoch is still running is released dirty
        and fails this attempt (ImportPreallocRetry); the next tick's retry
        finds it promoted."""
        from decoupled_engine.srt.mem_cache.base_prefix_cache import EvictParams
        from decoupled_engine.srt.mem_cache.cpu_page_book import STATS

        alloc = self.token_to_kv_pool_allocator
        short = required - alloc.clean_available_size()
        if short <= 0:
            return
        beyond = short - max(required - alloc.available_size(), 0)
        if beyond > 0:
            # Tokens evicted only because that many free pages are dirty.
            STATS["import_dirty_evict_tokens"] += beyond
        self.tree_cache.evict_for_alloc(EvictParams(num_tokens=short))

    def _q4_write_rows(
        self, req: Req, rows: Any, prefix_indices: Any, prefix_len: int, kv_loc: Any
    ) -> None:
        """The import's CPU row (the prefix's page ids from its tag, then the
        allocation's), then the device row with the mirror off: the CPU row is
        complete already, and DecodeReqToTokenPool (the pool symmetric workers
        build) has no mirror at all."""
        import torch

        from decoupled_engine.srt.mem_cache.cpu_page_book import pages_of

        row = req.req_pool_idx
        pages = pages_of(kv_loc, "import prealloc")
        if prefix_len > 0:
            pages = torch.cat((pages_of(prefix_indices, "import prefix"), pages))
        end = prefix_len + len(kv_loc)
        rows.write(row, 0, end, pages)
        by_caller = getattr(self.req_to_token_pool, "cpu_rows_written_by_caller", None)
        with by_caller() if by_caller is not None else _nullcontext():
            if prefix_len > 0:
                self.req_to_token_pool.write(
                    (row, slice(0, prefix_len)), prefix_indices
                )
            self.req_to_token_pool.write((row, slice(prefix_len, end)), kv_loc)

    @timed_phase("sym.import.prealloc")
    def _prealloc(self, req: Req) -> int:
        """Pre-allocate req_to_token + KV pages for the whole prompt (decode
        prealloc semantics: `kv_committed_len` = promised length) plus the
        reader-local aux / state slots, recorded on `req.import_mark`.
        Returns the prefix tokens being loaded back from the host tier.

        M4 (Q4 + load quarantine): the pages come from the CPU free lists and
        are clean pages only (alloc_import; the READ is not ordered after the
        forward in flight), and the CPU row is written here (_q4_write_rows);
        no device read."""
        import torch

        from decoupled_engine.srt.disaggregation.decode import alloc_for_decode_prealloc

        observe_prefix = envs.SGLANG_DECODE_PREFIX_OBSERVATION.get()
        if observe_prefix:
            # Allocation retries must not inherit a previous match or success.
            req._decode_prefix_match_observation = None
            req._symmetric_l2_admission_observation = None
            req.sched_trace_decode_prefix_observation = None
        req_pool_indices = self.req_to_token_pool.alloc([req])
        assert (
            req_pool_indices is not None
        ), "req_to_token_pool is full (bug in _can_admit)"

        fill_len = len(req.origin_input_ids)
        req.kv_committed_len = fill_len
        swa_tail_len = self._swa_tail_len(fill_len)
        # Reader-side prefix reuse: re-match (the tree may have changed since
        # _can_admit) and lock the matched node like PrefillAdder does; the
        # finish path (`cache_finished_req` -> dec lock on `req.last_node`)
        # releases exactly this lock, as it does for decode-side reuse in
        # upstream PD (`DecodePreallocQueue._match_prefix_and_lock`).
        # SWA-tail mode on the upstream decode shape (content-only reuse)
        # reuses the FULL component only, capped at the SWA tail start so the
        # tail the allocator must own is always imported. Window residency
        # reuses a FULL-only prefix capped at the tail
        # start too (_match_local_prefix) but inserts the whole prompt at
        # finish with its backfilled host SWA. The two never meet:
        # _content_only_reuse needs partial-component inserts, which the
        # uniform lifecycle keeps off.
        content_only = self._content_only_reuse()
        reuse = content_only or (
            self._tree_supports_reuse()
            and (not self.swa_tail_import or self.swa_window)
        )
        host_loaded = 0
        device_reused = 0
        req._symmetric_l2_restore_reason = "disabled"
        req._symmetric_l2_restore_tokens = 0
        if content_only:
            prefix_len, host_loaded = self._match_content_prefix(
                req, limit=fill_len - swa_tail_len
            )
        elif reuse:
            prefix_len = self._match_local_prefix(req)
            device_reused = prefix_len
            if self._ordinary_l2_restore_enabled():
                host_loaded = self._try_ordinary_l2_restore(req, prefix_len)
                prefix_len += host_loaded
        else:
            prefix_len = 0
        if reuse and req.last_node is not None:
            self._lock_prefix(req, content_only=content_only)
        else:
            req.last_node = None
        delta_len = fill_len - prefix_len
        required = ceil_align(fill_len, self.page_size) - prefix_len
        allocator = self.token_to_kv_pool_allocator
        # M4 Q4: the CPU rows (None when off). The prefix stays page-aligned
        # (_match_local_prefix), as alloc_import requires.
        q4_rows = self._q4_rows()
        from decoupled_engine.srt.mem_cache.base_prefix_cache import EvictParams

        with phase("sym.import.prealloc.evict"):
            if self.swa_tail_import:
                required_swa = ceil_align(swa_tail_len, self.page_size)
                short_full = required - allocator.full_available_size()
                short_swa = required_swa - allocator.swa_available_size()
                if short_full > 0 or short_swa > 0:
                    self.tree_cache.evict_for_alloc(
                        EvictParams(
                            num_tokens=max(short_full, 0),
                            swa_num_tokens=max(short_swa, 0),
                        )
                    )
            elif q4_rows is not None:
                self._q4_evict_for_import(required)
            elif allocator.available_size() < required:
                self.tree_cache.evict_for_alloc(
                    EvictParams(num_tokens=required - allocator.available_size())
                )
        prefix_indices = req.prefix_indices if prefix_len > 0 else None
        with phase("sym.import.prealloc.alloc"):
            kv_loc = alloc_for_decode_prealloc(
                allocator,
                req=req,
                fill_len=fill_len,
                delta_len=delta_len,
                prefix_len=prefix_len,
                total_prefix_len=prefix_len,
                prefix_indices=prefix_indices,
                # SWA-tail mode (D-role pool shape): full/c4/c128 pages for the
                # whole prompt, SWA pages only for the live window; the allocator
                # sets req.kv.swa_evicted_seqlen = fill_len - swa_tail_len.
                uses_swa_tail=self.swa_tail_import,
                swa_tail_len=swa_tail_len if self.swa_tail_import else fill_len,
                req_to_token_pool=self.req_to_token_pool,
            )
        if kv_loc is None:
            # undo this attempt: release the prefix lock and the req slot
            if req.last_node is not None:
                self._release_prefix_lock(req)
            self.req_to_token_pool.free(req)  # takes the Req; resets req.req_pool_idx
            if self.swa_tail_import:
                detail = (
                    f"full pool has {allocator.full_available_size()} free tokens, need {required}; "
                    f"swa pool has {allocator.swa_available_size()}, need {ceil_align(swa_tail_len, self.page_size)} "
                    f"(prompt {fill_len}, evictable full/swa {self._tree_evictable('full')}/{self._tree_evictable('swa')})"
                )
            elif q4_rows is not None:
                detail = (
                    f"KV pool has {allocator.clean_available_size()} clean and "
                    f"{allocator.dirty_count * self.page_size} dirty free tokens, need {required} clean "
                    f"(prompt {fill_len}, reused prefix {prefix_len}, evictable {int(self.tree_cache.evictable_size())})"
                )
            else:
                detail = (
                    f"KV pool has {allocator.available_size()} free tokens, need {required} "
                    f"(prompt {fill_len}, reused prefix {prefix_len}, evictable {int(self.tree_cache.evictable_size())})"
                )
            raise ImportPreallocRetry(detail)
        with phase("sym.import.prealloc.rows"):
            if q4_rows is not None:
                self._q4_write_rows(req, q4_rows, prefix_indices, prefix_len, kv_loc)
            else:
                if prefix_len > 0:
                    self.req_to_token_pool.write(
                        (req.req_pool_idx, slice(0, prefix_len)), prefix_indices
                    )
                self.req_to_token_pool.write(
                    (req.req_pool_idx, slice(prefix_len, prefix_len + len(kv_loc))),
                    kv_loc,
                )
        req.full_untruncated_fill_ids = req.origin_input_ids + req.output_ids
        if prefix_len == 0:
            req.prefix_indices = torch.empty((0,), dtype=torch.int64)
        # The reused prefix belongs to the tree: not this request's to free.
        req.cache_protected_len = prefix_len
        req.import_prefix_len = prefix_len
        # Tokens below this have no local SWA pages (page plan reports -1).
        req.import_swa_skip_before = (
            fill_len - swa_tail_len if self.swa_tail_import else 0
        )
        if self.swa_tail_import:
            assert req.kv.swa_evicted_seqlen == req.import_swa_skip_before, (
                req.kv.swa_evicted_seqlen,
                req.import_swa_skip_before,
            )
            self.stats_swa_tail_requests += 1
            self.stats_swa_tail_skipped_tokens += req.import_swa_skip_before
        if not content_only:
            self.stats_host_load_back_tokens += host_loaded
        if content_only:
            # Upstream content_only freeze: no unfinished insert, and only a
            # successful finish publishes the prompt (mem_cache/common.py
            # release_kv_cache). Unlike upstream FULL-only decode entries, the
            # SWA tail is kept through decode and published with the prompt, so
            # the entry is a checkpoint for peer pulls and after a D->P switch.
            req.decode_radix_cache_frozen_prompt_len = fill_len
            req.decode_radix_cache_frozen_swa_floor = req.import_swa_skip_before
            req.decode_radix_cache_skip_unfinished_insert = True
            self.stats_host_load_back_tokens += host_loaded
        if self.swa_window:
            # The prompt holds no SWA below its tail until the backfill
            # commits, so the PREBUILT unfinished insert (process_prebuilt) is
            # skipped; the finish insert adopts the backfilled ledger and
            # truncates at a remaining gap.
            req.decode_radix_cache_skip_unfinished_insert = True
        req.set_extend_range(prefix_len, fill_len)
        self.stats_imported_tokens += delta_len
        if prefix_len > 0:
            self.stats_prefix_reuse_requests += 1
            self.stats_prefix_reuse_tokens += prefix_len
            logger.debug(
                "symmetric-pd import rid=%s transfer_id=%d: reusing %d/%d prompt tokens "
                "from the local radix tree; importing %d",
                req.rid,
                req.import_mark.transfer_id,
                prefix_len,
                fill_len,
                delta_len,
            )
        if self.req_to_metadata_buffer_idx_allocator is not None:
            idx = self.req_to_metadata_buffer_idx_allocator.alloc()
            assert idx is not None, "metadata slots exhausted (bug in _can_admit)"
            req.metadata_buffer_index = idx
        # The ReaderPlane READs FINAL's owner slots into these (pull/api.py
        # ImportMark); begin_import requires local_aux_index to be set.
        mark = req.import_mark
        # Advertise only the ordinary validated path, after its prefix lock and
        # row writes. Staged host pages retain their existing DMA pins; READY
        # still waits for their load acknowledgement in pop_ready.
        mark.local_reuse_prefix_tokens = (
            prefix_len if envs.SGLANG_SYMMETRIC_SKIP_REUSED_PREFIX_PUBLICATION.get()
            and self._ordinary_l2_restore_enabled() and not content_only
            and not self.swa_tail_import and not self.swa_window
            and q4_rows is None and req.last_node is not None else 0
        )
        mark.local_aux_index = (
            req.metadata_buffer_index if req.metadata_buffer_index >= 0 else None
        )
        mark.local_state_indices = self._local_state_indices(req, fill_len)
        if observe_prefix:
            matched = req._decode_prefix_match_observation
            req.sched_trace_decode_prefix_observation = {
                "mode": "symmetric",
                "matched_device_tokens": matched[0] if matched is not None else None,
                "matched_host_tokens": matched[1] if matched is not None else None,
                # Ordinary symmetric matching does not query storage coverage.
                "selected_storage_tokens": None,
                # The content-only path has different restore semantics; do not
                # infer its original device hit from the final prefix length.
                "reused_device_tokens": device_reused if not content_only else None,
                "transfer_skip_prefix_tokens": prefix_len,
                "local_restore_requested_tokens": host_loaded if not content_only else None,
                "host_load_back_queued_tokens": host_loaded,
                "local_restore_committed_tokens": 0 if not content_only else None,
                "local_restore_fallback_reason": req._symmetric_l2_restore_reason,
                "local_restore_admission": req._symmetric_l2_admission_observation,
            }
        return host_loaded

    def _ordinary_l2_restore_enabled(self) -> bool:
        return (
            envs.SGLANG_SYMMETRIC_IMPORT_L2_RESTORE.get()
            and getattr(self.tree_cache, "symmetric_import_l2_restore_validated", False)
            is True
        )

    @timed_phase("sym.import.prealloc.l2_restore")
    def _try_ordinary_l2_restore(self, req: Req, device_len: int) -> int:
        """Stage a complete existing-L2 restore, or retain the original READ.

        There is no wait queue: all ranks must have the same coverage, a free
        load slot and enough native admission capacity. Existing load_back may
        reclaim evictable device cache while its host/device path is locked.
        Staged pages have the tree's separate DMA pins; the request acquires its
        own prefix lock afterwards.
        Unexpected partial staging is fatal before issuing DMA, not a fallback
        that can race DMA or silently discard its pins.
        """
        import copy

        from decoupled_engine.srt.mem_cache.base_prefix_cache import InitLoadBackParams

        tree = self.tree_cache
        controller = tree.cache_controller
        host_len = int(getattr(req, "host_hit_length", 0))
        limit = page_floor(max(len(req.origin_input_ids) - 1, 0), self.page_size)
        admission_need = self._required_tokens(req)
        admission_budget = self._allocatable_tokens()
        if envs.SGLANG_DECODE_PREFIX_OBSERVATION.get():
            # Record the exact budget used by this decision. The raw counters
            # are adjacent scalar observations, not an atomic pool snapshot.
            req._symmetric_l2_admission_observation = {
                "raw_free_tokens": self.token_to_kv_pool_allocator.available_size(),
                "evictable_tokens": int(tree.evictable_size()),
                "admission_budget_tokens": admission_budget,
                "admission_required_tokens": admission_need,
                "matched_device_tokens": device_len,
                "matched_host_extension_tokens": host_len,
            }
        reason = "ready"
        slot = -1
        if controller is None:
            reason = "no_controller"
        elif host_len <= 0:
            reason = "no_host_extension"
        elif (
            device_len != len(req.prefix_indices)
            or host_len % self.page_size
            or device_len + host_len > limit
            or getattr(req, "swa_host_hit_length", 0)
            or getattr(req, "mamba_host_hit_length", 0)
        ):
            reason = "incomplete_ordinary_coverage"
        elif self._load_back_pending(req.best_match_node):
            reason = "restore_already_pending"
        elif controller.load_queue:
            reason = "other_load_staged"
        else:
            counter = controller.layer_done_counter
            slot = (counter.producer_index + 1) % counter.num_counters
            if not tree.is_load_back_event_done(slot):
                reason = "load_slot_busy"
            elif admission_need > admission_budget:
                # Use the same whole-prompt + decode-headroom admission rule.
                # Do not subtract the device prefix: it can still be counted in
                # evictable capacity until load_back locks its ancestor path.
                # Native load_back reclaims its own shortfall under those locks;
                # the unchanged suffix preallocation reclaims any remainder.
                reason = "insufficient_admission_budget"
            elif host_len < max(1, tree.load_back_threshold):
                reason = "below_existing_load_threshold"
        # Every rank enters this vote, including rank-local no-hit/slot misses.
        low, neg_high, dev_low, dev_neg_high, slot_low, slot_neg_high, ready = (
            self.consensus.agree(
                [
                    host_len, -host_len, device_len, -device_len,
                    slot, -slot, int(reason == "ready"),
                ]
            )
        )
        if (
            not ready
            or low != -neg_high
            or dev_low != -dev_neg_high
            or slot_low != -slot_neg_high
        ):
            req._symmetric_l2_restore_reason = (
                reason if reason != "ready" else "peer_not_ready_or_coverage_differs"
            )
            self.stats_host_load_back_skipped += int(host_len > 0)
            return 0

        match_req = copy.copy(req)
        loaded, node, stage_error = None, None, None
        try:
            result = tree.init_load_back(
                InitLoadBackParams(
                    best_match_node=req.best_match_node,
                    host_hit_length=host_len,
                    mem_quota=host_len,
                    req=match_req,
                )
            )
            if result is not None:
                loaded, node = result
        except Exception as error:
            # Peers must still enter the staging vote before all fail closed.
            # A thrown allocation error is not proof that no pins were installed.
            stage_error = error
        loaded_len = len(loaded) if loaded is not None else 0
        # init_load_back does not issue DMA. It can fail locally despite the
        # preflight; check its actual result and pending state on every rank.
        pending = self._load_back_pending(req.best_match_node)
        staged = bool(controller.load_queue)
        low, neg_high, none_safe, complete = self.consensus.agree(
            [
                loaded_len,
                -loaded_len,
                int(stage_error is None and loaded_len == 0 and not pending and not staged),
                int(stage_error is None and loaded_len == host_len and pending and staged),
            ]
        )
        if none_safe:
            req._symmetric_l2_restore_reason = "load_allocation_unavailable"
            self.stats_host_load_back_skipped += 1
            return 0
        if not complete or low != -neg_high:
            raise RuntimeError(
                "symmetric L2 restore staged incomplete or divergent coverage; "
                f"cannot safely issue DMA or fall back to READ (rid={req.rid}, "
                f"local_loaded={loaded_len}, expected={host_len}, "
                f"pending={pending}, staged={staged})"
            ) from stage_error
        # All ranks issue the same packed target/draft KV + indexer operation.
        # loading_check owns DMA locks until its synchronized completion ack.
        tree.ready_to_load_host_cache()
        req.prefix_indices = cat_tagged(
            [
                req.prefix_indices,
                copy_tag(loaded.to(dtype=req.prefix_indices.dtype), loaded),
            ],
            self.page_size,
            "ordinary symmetric host load-back",
        )
        req.last_node = node
        req._symmetric_l2_restore_tokens = loaded_len
        req._symmetric_l2_restore_reason = "queued"
        return loaded_len

    @timed_phase("sym.import.prealloc.match")
    def _match_content_prefix(self, req: Req, *, limit: int) -> Tuple[int, int]:
        """Upstream content_only decode match (`DecodePreallocQueue.
        _match_prefix_and_lock`): FULL component only, at most `limit` tokens
        (the SWA tail start). A host-tier hit is loaded back to the device
        (`DecodeHiCacheTransferMixin._try_hicache_queue_load_back`). Sets
        `req.prefix_indices` / `req.last_node`; returns (page-aligned reused
        length, tokens being loaded back)."""
        import torch

        from decoupled_engine.srt.managers.schedule_policy import match_prefix_for_req
        from decoupled_engine.srt.mem_cache.unified_cache.components import ComponentType

        limit = page_floor(max(limit, 0), self.page_size)
        try:
            match_prefix_for_req(
                self.tree_cache,
                req,
                req.origin_input_ids,
                include_req=True,
                active_components=(ComponentType.FULL,),
                key_limit=limit,
            )
        except Exception as e:  # never let reuse break admission
            logger.warning(
                "symmetric-pd import rid=%s: content-only prefix match failed (%r); importing all",
                req.rid,
                e,
            )
            req.prefix_indices = torch.empty((0,), dtype=torch.int64)
            req.last_node = None
            return 0, 0
        host_loaded = self._load_back_host_hit(req) if req.host_hit_length > 0 else 0
        n = min(page_floor(len(req.prefix_indices), self.page_size), limit)
        if n < len(req.prefix_indices):
            req.prefix_indices = slice_tagged(req.prefix_indices, 0, n, self.page_size)
        return n, host_loaded

    def _load_back_host_hit(self, req: Req) -> int:
        """Start the H2D load of the host-only part of a content-only match and
        extend `req.prefix_indices` / `req.last_node` over it. The loaded nodes
        stay pinned (`load_back_pending_id`) until the ack; pop_ready holds the
        import until then. Returns the tokens being loaded (0 = READ them)."""
        from decoupled_engine.srt.mem_cache.base_prefix_cache import InitLoadBackParams
        from decoupled_engine.srt.mem_cache.unified_cache.components import ComponentType

        tree = self.tree_cache
        controller = tree.cache_controller
        if controller is None or self._load_back_pending(req.best_match_node):
            return 0
        # Host dedup broadcasts every loaded layer from TP rank 0, so all ranks
        # must issue the same loads; the load ring also asserts that the slot
        # it reuses has finished.
        counter = controller.layer_done_counter
        slot_free = tree.is_load_back_event_done(
            (counter.producer_index + 1) % counter.num_counters
        )
        agreed = self.consensus.agree(
            [_consensus.READY if slot_free else _consensus.PENDING]
        )
        if agreed[0] != _consensus.READY:
            self.stats_host_load_back_skipped += 1
            return 0
        loaded, node = tree.init_load_back(
            InitLoadBackParams(
                best_match_node=req.best_match_node,
                host_hit_length=req.host_hit_length,
                mem_quota=req.host_hit_length,
                req=req,
                active_components=(ComponentType.FULL,),
            )
        )
        if len(loaded) == 0:
            return 0
        tree.ready_to_load_host_cache()
        # M4 Q4: the loaded slots carry their CPU page ids (plain cat when off).
        req.prefix_indices = cat_tagged(
            [
                req.prefix_indices,
                copy_tag(loaded.to(dtype=req.prefix_indices.dtype), loaded),
            ],
            self.page_size,
            "import host load-back",
        )
        req.last_node = node
        return len(loaded)

    def _load_back_pending(self, node_handle) -> bool:
        """Whether a node on the root path of `node_handle` still has an H2D
        load in flight. Load acks are MIN-synced across TP ranks
        (UnifiedRadixCache.loading_check), so every rank sees the same answer."""
        tree = self.tree_cache
        node = tree.resolve_node_handle(node_handle)
        while node is not None and node is not tree.root_node:
            if node.load_back_pending_id is not None:
                return True
            node = node.parent
        return False

    @timed_phase("sym.import.prealloc.lock")
    def _lock_prefix(self, req: Req, *, content_only: bool) -> None:
        """Lock the matched node like PrefillAdder; a content-only match locks
        FULL only, and the finish path then releases it without SWA
        (`swa_prefix_lock_released`), as upstream content_only decode does.
        A window-residency match is FULL-only too (the
        prefix's SWA may be host-only), and is locked and released the same
        way."""
        full_only = content_only or self.swa_window
        if full_only:
            from decoupled_engine.srt.mem_cache.unified_cache.components import ComponentType

            lock = self.tree_cache.inc_lock_ref(
                req.last_node, skip_lock_components=(ComponentType.SWA,)
            )
        else:
            lock = self.tree_cache.inc_lock_ref(req.last_node)
        req.swa_uuid_for_lock = getattr(lock, "swa_uuid_for_lock", None)
        req.skip_lock_node_ids = getattr(lock, "skip_lock_node_ids", {}) or {}
        req.skip_lock_components = getattr(lock, "skip_lock_components", ()) or ()
        req.swa_prefix_lock_released = full_only

    def _release_prefix_lock(self, req: Req) -> None:
        """Undo the inc_lock_ref taken in _prealloc (mirrors upstream
        DecodePreallocQueue._release_matched_prefix_lock)."""
        try:
            from decoupled_engine.srt.mem_cache.base_prefix_cache import DecLockRefParams

            params = DecLockRefParams(
                swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None),
                skip_lock_node_ids=getattr(req, "skip_lock_node_ids", {}) or {},
                skip_lock_components=getattr(req, "skip_lock_components", ()) or (),
            )
            if req.swa_prefix_lock_released:
                self.tree_cache.dec_lock_ref(req.last_node, params, skip_swa=True)
            else:
                self.tree_cache.dec_lock_ref(req.last_node, params)
        except TypeError:
            self.tree_cache.dec_lock_ref(req.last_node)
        except Exception as e:  # never let lock bookkeeping kill the scheduler
            logger.warning(
                "symmetric-pd import rid=%s: releasing prefix lock failed (%r)", req.rid, e
            )
        req.last_node = None
        req.swa_uuid_for_lock = None
        req.skip_lock_components = ()
        req.swa_prefix_lock_released = False

    def _local_state_indices(self, req: Req, total_tokens: int) -> Dict[str, List[int]]:
        """Reader-local per-state-type slots FINAL's `state_indices` are READ
        into, keyed like `ExportTable.finish_export` keys them (POOL_STATE for
        the DSv4 request-scoped C128 rows; empty for models without state)."""
        pools = dst_pages_for_range(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req=req,
            token_start=0,
            token_end=total_tokens,
            page_size=self.page_size,
            include_state=True,
            total_tokens=total_tokens,
            swa_skip_before=int(getattr(req, "import_swa_skip_before", 0) or 0),
        )
        rows = pools.get(POOL_STATE)
        return {POOL_STATE: [int(r) for r in rows]} if rows else {}

    def _dst_pages_fn(
        self, req: Req
    ) -> Callable[[ChunkReady], Dict[str, Sequence[int]]]:
        """Per-chunk {pool: reader page indices}; read-only lookup, called on the
        plane's IO thread. Aux / state slots are not chunk pages: they travel on
        the ImportMark and are READ at FINAL."""
        import torch

        # The plane's IO thread starts on CUDA device 0; on TP rank i the pool
        # tensors live on device i, so pin the current device for any index
        # tensors the translation creates.
        device = self.req_to_token_pool.req_to_token.device
        # The scheduler writes the row on its own CUDA stream. The IO thread
        # must observe those writes before translating it into RDMA pages.
        row_ready = None
        if device.type == "cuda":
            row_ready = torch.cuda.Event()
            row_ready.record(torch.cuda.current_stream(device))
        # Pages below the reused prefix are already resident: mark them -1 so
        # importer._chunk_plan drops those (remote, local) pairs and no READ is
        # issued for them (the chunk still counts as read; generations are
        # still verified against FINAL for every announced page).
        skip_before = int(getattr(req, "import_prefix_len", 0) or 0)
        # SWA-tail import: SWA pages below this token are not allocated here
        # (export_table.pool_pages_for_range reports them -1).
        swa_skip_before = int(getattr(req, "import_swa_skip_before", 0) or 0)

        def dst_pages(chunk: ChunkReady) -> Dict[str, Sequence[int]]:
            with torch.cuda.device(device) if device.type == "cuda" else _nullcontext():
                if row_ready is not None:
                    torch.cuda.current_stream(device).wait_event(row_ready)
                pages = dst_pages_for_range(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    req=req,
                    token_start=chunk.token_start,
                    token_end=chunk.token_end,
                    page_size=self.page_size,
                    swa_skip_before=swa_skip_before,
                    # Mirrors ExportTable._make_chunk, which attaches POOL_STATE
                    # rows to the is_last CHUNK_READY as well as to
                    # FINAL.state_indices.
                    include_state=chunk.is_last,
                    total_tokens=len(req.origin_input_ids) if chunk.is_last else None,
                )
            return mark_reused_pages(
                pages, chunk.token_start, chunk.token_end, skip_before
            )

        return dst_pages

    # -- progress -------------------------------------------------------
    @timed_phase("sym_pop_ready")
    def pop_ready(
        self, *, max_ready: Optional[int] = None
    ) -> Tuple[List[Req], List[Tuple[Req, str]]]:
        """(ready requests with first token installed, [(req, reason)] failed).

        Two consensus rounds when TP > 1 (none when nothing is queued):
        1. per-import state (READY / pending / failed-or-timed-out) is
           MIN-reduced so every rank takes the same transition this tick;
        2. after the local first-token commits, their outcomes are reduced
           again so a rank-local commit error fails the import everywhere.

        max_ready is the TP-consistent execution budget supplied by the
        scheduler. Excess READY imports retain their pages, aux slot and cancel
        ownership; failures/timeouts are processed even with no free slots.
        """
        ready: List[Req] = []
        failed: List[Tuple[Req, str]] = []
        now = self.clock()

        active_ids = sorted(self.active)
        # Late entries (DEFERRED / ADMITTED / HELD) end by their own controls,
        # never by this vote.
        pending_entries = [e for e in self.pending if e.late_d_state is None]
        if not active_ids and not pending_entries:
            return ready, failed

        local: List[int] = []
        reasons: Dict[int, str] = {}
        with phase("sym.import.poll"):
            for transfer_id in active_ids:
                entry = self.active[transfer_id]
                state = self.reader_plane.poll(transfer_id)
                if state == TransferState.READY and not self._host_load_pending(entry):
                    local.append(_consensus.READY)
                elif state in (TransferState.FAILED, TransferState.ABORTED):
                    reasons[transfer_id] = (
                        self.reader_plane.failure_reason(transfer_id) or state.name
                    )
                    local.append(_consensus.FAILED)
                elif self._timed_out(entry, now):
                    reasons[transfer_id] = IMPORT_TIMEOUT
                    local.append(_consensus.FAILED)
                else:
                    local.append(_consensus.PENDING)
            for entry in pending_entries:
                local.append(
                    _consensus.FAILED
                    if self._timed_out(entry, now)
                    else _consensus.PENDING
                )
        # The vote waits for the slowest TP rank (gloo all-reduce). Vote 1 runs
        # in every tick with an active or pending import.
        with phase("sym_pop_ready_vote1"):
            agreed = self.consensus.agree(local)

        committed: List[Tuple[int, Req, Optional[str], Optional[FinalMeta]]] = []
        for transfer_id, state in zip(active_ids, agreed[: len(active_ids)]):
            entry = self.active[transfer_id]
            req = entry.req
            if state == _consensus.READY:
                if max_ready is not None and len(committed) >= max_ready:
                    continue
                self.active.pop(transfer_id)
                error = self._commit(entry)
                committed.append((transfer_id, req, error, entry.final_meta))
            elif state == _consensus.FAILED:
                self.active.pop(transfer_id)
                reason = reasons.get(transfer_id)
                if reason is None:
                    # A peer rank failed / timed out; this rank's transfer may
                    # still be in flight: abort it so the owner's Lease ends.
                    reason = IMPORT_PEER_FAILED
                    self._abort_import(transfer_id, reason)
                elif reason == IMPORT_TIMEOUT:
                    self._abort_import(transfer_id, IMPORT_TIMEOUT)
                self._forget(transfer_id)
                failed.append((req, reason))

        if committed:
            # Vote 2: only in ticks that commit an import.
            with phase("sym_pop_ready_vote2"):
                outcome = self.consensus.agree(
                    [
                        _consensus.READY if error is None else _consensus.FAILED
                        for _, _, error, _ in committed
                    ]
                )
            for (transfer_id, req, error, final), state in zip(committed, outcome):
                if state == _consensus.READY:
                    ready.append(req)
                    # After the second agreement, so every rank applies the
                    # same settings and submits the same backfills in the same
                    # order (the manager's lockstep contract); a rank-local
                    # commit error never gets here, on any rank.
                    self._apply_ready(req, final)
                    self._submit_swa_backfill(req)
                else:
                    reason = error or IMPORT_PEER_FAILED
                    self._abort_import(transfer_id, reason)
                    failed.append((req, reason))
                # The plane's record is consumed either way; drop it so
                # concluded imports do not accumulate (in a soak test the
                # reader import_count grew 40 -> 168 over 8 rounds).
                self._forget(transfer_id)

        timed_out = set()
        for entry, state in zip(pending_entries, agreed[len(active_ids) :]):
            if state == _consensus.FAILED:
                self.reader_plane.abort_pending(entry.req.import_mark, IMPORT_TIMEOUT)
                failed.append((entry.req, IMPORT_TIMEOUT))
                timed_out.add(id(entry))
        if timed_out:
            # Late entries stay, in order.
            self.pending = [e for e in self.pending if id(e) not in timed_out]
        return ready, failed

    def _abort_import(self, transfer_id: int, reason: str) -> None:
        # ReaderPlane.abort_import blocks until the import's READs drained.
        with phase("sym.import.abort_wait"):
            self.reader_plane.abort_import(transfer_id, reason)

    def _host_load_pending(self, entry: ImportEntry) -> bool:
        # A prefix loaded back from the host tier is decodable only once acked.
        return entry.host_loaded_tokens > 0 and self._load_back_pending(
            entry.req.last_node
        )

    def _forget(self, transfer_id: int) -> None:
        forget = getattr(self.reader_plane, "forget", None)
        if callable(forget):
            forget(transfer_id)

    def _timed_out(self, entry: ImportEntry, now: float) -> bool:
        # A late import waited DEFERRED before its allocation: its timeout
        # counts from the admission.
        start = entry.added_at
        if entry.late_d_state is not None and entry.admitted_at is not None:
            start = entry.admitted_at
        return (
            self.waiting_timeout_s is not None and now - start > self.waiting_timeout_s
        )

    @timed_phase("sym_pop_ready_commit")
    def _commit(self, entry: ImportEntry) -> Optional[str]:
        """Install the first token from the READER-LOCAL aux slot on the
        ImportMark; `final_meta` carries the OWNER's indices and is only used
        for consistency checks. Returns the error, None on success. Runs
        before the second agreement, so it sets nothing a failed import would
        have to undo on some ranks only: it records the FinalMeta on the entry
        (`entry.final_meta`), and pop_ready's _apply_ready applies the cache
        policy and the backfill source once every rank agreed READY."""
        req = entry.req
        final = self.reader_plane.final_meta(entry.transfer_id)
        entry.final_meta = final
        if final is None:
            return "READY without FinalMeta"
        if final.total_tokens != len(req.origin_input_ids):
            return (
                f"FINAL total_tokens={final.total_tokens} != prompt length "
                f"{len(req.origin_input_ids)}"
            )
        mark = req.import_mark
        local_aux = mark.local_aux_index if mark is not None else None
        if self.metadata_buffers is None or local_aux is None:
            return "no local aux slot to commit the first token from"
        error = commit_first_token_from_aux(
            req,
            self.metadata_buffers,
            local_aux,
            spec_is_none=self.spec_is_none,
            expected_room=entry.transfer_id,
        )
        self._free_aux(req)
        if error is None:
            req.time_stats.set_wait_queue_entry_time()
        return error

    @timed_phase("sym.import.apply_ready")
    def _apply_ready(self, req: Req, final: Optional[FinalMeta]) -> None:
        """Settings of an import every rank agreed READY (second agreement).

        The cache policy decides the finish insert (drop ->
        skip_radix_cache_insert) and the host write-through (gpu_only ->
        hicache_skip_write_through). Set before the agreement, a rank-local
        commit error would leave them on the other ranks only, and the local
        recompute (release_failed) would then insert / write through on some
        ranks: the radix and host trees would diverge across TP ranks.
        release_failed cannot reset skip_radix_cache_insert either, since
        Req.__init__ derives it from the bootstrap host."""
        observation = getattr(req, "sched_trace_decode_prefix_observation", None)
        if observation is not None and getattr(req, "_symmetric_l2_restore_tokens", 0):
            observation["local_restore_committed_tokens"] = req._symmetric_l2_restore_tokens
        mark = req.import_mark
        requested = mark.cache_policy if mark is not None else "retain"
        policy = self._effective_cache_policy(mark)
        if self.swa_window and policy != requested:
            # gpu_only -> retain; counted once per import, on every rank.
            self.stats_cache_policy_overridden += 1
            metrics = getattr(self.tree_cache, "dsv4_window_metrics", None)
            if metrics is not None:
                metrics.import_cache_policy_overridden.labels(
                    **metrics.labels, **{"from": requested}
                ).inc()
        apply_import_cache_policy(req, policy)
        # Window residency: where the backfill reads the SWA below the tail
        # (None: owner without a servicer, or predating the field).
        req.swa_backfill_source = getattr(final, "owner_peer_kv_addr", None)
        if getattr(mark, "host_return", None) is not None:
            # Capture only after TP READY agreement. A legacy owner supplies
            # no epoch: generation succeeds, and the optional return is skipped.
            req.host_return_owner_epoch = getattr(final, "owner_cache_epoch", None)

    def _effective_cache_policy(self, mark) -> str:
        """The policy an import follows (no side effects; _apply_ready counts
        the window coercion)."""
        policy = mark.cache_policy if mark is not None else "retain"
        if self.swa_window:
            # Window residency: the prompt is inserted
            # with backfilled host SWA. gpu_only would skip its write-through
            # and leave device nodes that are never demotable nor published,
            # so it becomes retain; drop still discards.
            return "retain" if policy == "gpu_only" else policy
        if self.swa_tail_import and not self._content_only_reuse():
            # Only the window tail of the SWA pool exists locally, and without
            # a FULL-only insert the whole prompt would be published with a
            # tombstoned SWA prefix.
            return "drop"
        return policy

    @timed_phase("sym.import.swa_backfill")
    def _submit_swa_backfill(self, req: Req) -> None:
        """Window residency: the D-leg holds FULL for its
        whole prompt but SWA only from T = kv.swa_evicted_seqlen on; the
        reused prefix [0, R = cache_protected_len) is tree-owned. Queue the
        host->host backfill of [R, T) from the owner's servicer into the
        request's ledger (opened at base R). It retries until the owner
        publishes the prompt (after DONE -> release -> FULL ack) or times out;
        whatever is missing at finish truncates the insert there (section
        8.3). Nothing to do for a dropped prompt."""
        if not self.swa_window or getattr(req, "skip_radix_cache_insert", False):
            return
        start = int(req.cache_protected_len)
        end = int(req.kv.swa_evicted_seqlen)
        if end <= start:
            return
        tc = self.tree_cache
        source = getattr(req, "swa_backfill_source", None)
        if not source or getattr(tc, "swa_backfill", None) is None:
            self.stats_swa_backfill_skipped += 1
            logger.debug(
                "symmetric-pd import rid=%s: no SWA backfill of [%d, %d) (source=%s, "
                "backfill %s); its finish insert stops at %d",
                req.rid,
                start,
                end,
                source,
                "off" if getattr(tc, "swa_backfill", None) is None else "on",
                start,
            )
            return
        keys = tc.swa_backfill_page_keys(req, start, end)
        if tc.submit_swa_backfill(req, start, end, source, keys):
            self.stats_swa_backfill_submitted += 1
        else:
            self.stats_swa_backfill_skipped += 1

    # -- failure / abort ------------------------------------------------
    @timed_phase("sym.import.release_failed")
    def release_failed(self, req: Req) -> None:
        """Undo the preallocation of a failed/aborted import so the request can
        be recomputed locally (fallback ladder) or dropped."""
        from decoupled_engine.srt.mem_cache.common import release_kv_cache

        self._free_aux(req)
        if req.is_holding_kv:
            # Nothing valid was written; never insert into the radix tree.
            # (Window residency: this also releases a ledger/backfill.)
            release_kv_cache(req, self.tree_cache, is_insert=False)
        import torch

        req.prefix_indices = torch.empty((0,), dtype=torch.int64)
        req.extend_range = None
        req.kv_committed_len = 0
        req.output_ids = array("q")
        req.already_computed = 0
        req.import_mark = None
        req.bootstrap_room = None
        # The local recompute caches like any request, not like an import.
        req.decode_radix_cache_frozen_prompt_len = None
        req.decode_radix_cache_frozen_swa_floor = None
        req.decode_radix_cache_skip_unfinished_insert = False
        req.swa_prefix_lock_released = False
        # Window reader (_prealloc): also undo the FULL-only lock components.
        # The cache-policy flags and the backfill source need no reset:
        # _apply_ready sets them only on imports every rank agreed READY,
        # which never come back here (the source is cleared defensively).
        if self.swa_window:
            req.skip_lock_components = ()
            req.swa_backfill_source = None

    def abort_matching(self, rid_prefix: Optional[str], abort_all: bool) -> List[Req]:
        """Remove and release imports matching an AbortReq; returns them so the
        scheduler can emit the abort outputs."""

        def match(req: Req) -> bool:
            return abort_all or (
                rid_prefix is not None and req.rid.startswith(rid_prefix)
            )

        aborted: List[Req] = []
        keep: List[ImportEntry] = []
        for entry in self.pending:
            if match(entry.req):
                self.reader_plane.abort_pending(
                    entry.req.import_mark, "aborted by client"
                )
                aborted.append(entry.req)
                if entry.late_d_state is not None and self.late_port is not None:
                    self.late_port.note_ended(entry.transfer_id)
            else:
                keep.append(entry)
        self.pending = keep
        for transfer_id, entry in list(self.active.items()):
            if match(entry.req):
                self.active.pop(transfer_id)
                self._abort_import(transfer_id, "aborted by client")
                self._forget(transfer_id)
                aborted.append(entry.req)
        for req in aborted:
            self.release_failed(req)
        return aborted

    # -- introspection --------------------------------------------------
    def waiting_reqs(self) -> List[Req]:
        return [e.req for e in self.pending] + [e.req for e in self.active.values()]

    def __len__(self) -> int:
        return len(self.pending) + len(self.active)

    def _free_aux(self, req: Req) -> None:
        if (
            self.req_to_metadata_buffer_idx_allocator is not None
            and req.metadata_buffer_index >= 0
        ):
            self.req_to_metadata_buffer_idx_allocator.free(req.metadata_buffer_index)
            req.metadata_buffer_index = -1
        if req.import_mark is not None:
            # The slot is gone; do not leave a stale local index on the mark.
            req.import_mark.local_aux_index = None
