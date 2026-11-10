"""Owner-side bookkeeping of Export requests in symmetric mode.

Covers the Export lifecycle, chunk publication and DSv4 multi-component KV.

Export:  WAITING -> PREFILL chunks (publish each chunk) -> emit first token ->
         EXPORTED (KV under Lease, req leaves batch, no free) -> released -> free

The table never frees announced pages itself; `poll_released()` hands finished
exports back to the scheduler, which calls `release_kv_cache`. The only pages it
frees are SWA pages it announced -1 (`trim_swa`, SWA-tail export).
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import msgspec

from decoupled_engine.srt.disaggregation.base.conn import StateType
from decoupled_engine.srt.disaggregation.late_d_need import AGED_NONE, aged_vote
from decoupled_engine.srt.disaggregation.pull.api import (
    ChunkReady,
    FinalMeta,
    OwnerPlane,
    PoolPages,
)
from decoupled_engine.srt.disaggregation.pull.protocol import PoolLayout, PoolRegion
from decoupled_engine.srt.disaggregation.symmetric import consensus as _consensus
from decoupled_engine.srt.disaggregation.symmetric.consensus import RankConsensus
from decoupled_engine.srt.environ import envs

if TYPE_CHECKING:
    from decoupled_engine.srt.disaggregation.utils import (
        MetadataBuffers,
        ReqToMetadataIdxAllocator,
    )
    from decoupled_engine.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


def state_pool_key(state_type: StateType) -> str:
    """Key of one state component in FinalMeta.state_indices; must equal the key
    protocol.Registration.from_kv_args gives Registration.states (str(state_type))."""
    return str(state_type)


# Pool names carried in ChunkReady.pools / dst_pages and Registration.pools
# The reader allocates per pool; both sides derive the
# same layout for one model (registration_pools_for).
POOL_FULL = "full"  # allocator page index space; on DSv4 it carries no bytes
POOL_SWA = "swa"  # SWA physical pages (DSv4: swa_page_size tokens each)
POOL_C4 = "c4"  # DSv4 CSA pages + c4 indexer pages, addressed by the FULL page index
POOL_C128 = "c128"  # DSv4 HCA pages, addressed by the FULL page index
# DSv4 request-scoped C128 state rows: FINAL only, never a chunk pool.
POOL_STATE = state_pool_key(StateType.C128_STATE)

RELEASED = "released"
EXPORT_TIMEOUT = "export timeout"
# Reader gave up (ABORT/DONE) before this P-leg finished its prefill: the request
# is still queued/running and owns its KV; the scheduler must only detach the
# export mark and let it finish like a Local request (freeing it here
# double-released the radix lock -> lock_ref underflow).
RELEASED_BEFORE_EXPORT = "released before export"
# Reader-first exports (SGLANG_SYMMETRIC_PD_EXPORT_AWAIT_READER, ExportTable.hold):
# the reader gave up while the P-leg was still held, outside every queue. The
# scheduler queues it as a Local request (what RELEASED_BEFORE_EXPORT does to
# a queued P-leg).
RELEASED_WHILE_HELD = "released while awaiting the reader"
# Why poll_begun hands a held P-leg back to the scheduler.
READER_BEGUN = "reader began"
AWAIT_READER_TIMEOUT = "await-reader timeout"
# Late D (--disaggregation-late-d): a held late P-leg failed by the owner's
# admission or bind timer (rank 0 votes FAILED, the MIN spreads it), and a
# held late P-leg whose reader gave up. Both end with HTTP 503 and no prefill
# (rule X); never queued as Local.
LATE_BIND_TIMEOUT = "late-D bind timeout"
LATE_READER_GONE = "late-D reader gone"
COMPUTE_ABORT = "compute-before-allocation aborted"
COMPUTE_PARK_TIMEOUT = "compute-before-allocation park timeout"
COMPUTE_READER_GONE = "compute-before-allocation chosen reader aborted"
# Private compute-state vote, MIN-reduced with the existing state vector.
# A distinct negative value preserves the reader-failure reason on every rank.
_COMPUTE_READER_FAILED = -2
# Stage 2: rank 0's late_fail_phase when the committed winner aborted
# before its BEGIN (the P-leg then ends as LATE_BIND_TIMEOUT does, with the
# reader-gone err_type).
LATE_PHASE_READER_GONE = "winner aborted before BEGIN"


def page_floor(length: int, page_size: int) -> int:
    return length - (length % page_size) if page_size > 1 else length


def _is_dsv4_pool(kvcache: Any) -> bool:
    # Avoid importing deepseek_v4_memory_pool (CUDA-only) at module import.
    return any(cls.__name__.startswith("DeepSeekV4") for cls in type(kvcache).__mro__)


def _is_unified_kv(kvcache: Any) -> bool:
    # DeepSeekV4TokenToKVPool._unified_kv: SWA lives in a per-request ring inside
    # the unified buffers (StateType.SWA_RING), not in page-addressable SWA pages.
    return _is_dsv4_pool(kvcache) and bool(kvcache._unified_kv)


def _sub_allocators(token_to_kv_pool_allocator: Any) -> Tuple[Any, Optional[Any]]:
    """(full, swa) sub-allocators. Hybrid SWA allocators split into
    full_/swa_attn_allocator (same capability check decode.py uses:
    hasattr(allocator, "alloc_extend_swa_tail")); plain allocators are their own
    FULL pool and have no SWA pool."""
    if hasattr(token_to_kv_pool_allocator, "swa_attn_allocator"):
        return (
            token_to_kv_pool_allocator.full_attn_allocator,
            token_to_kv_pool_allocator.swa_attn_allocator,
        )
    return token_to_kv_pool_allocator, None


def _swa_pages_per_page(kvcache: Any, page_size: int) -> int:
    """Physical SWA pages per allocator page. DSv4 sizes its SWA pool by
    swa_page_size (deepseek_v4_memory_pool.py DeepSeekV4TokenToKVPool.__init__:
    `assert page_size % swa_page_size == 0`; kv_cache_configurator.py
    _build_dsv4_kv_pool passes swa_page_size == page_size on CUDA, so this is 1 there).
    """
    swa_page_size = kvcache.swa_page_size if _is_dsv4_pool(kvcache) else page_size
    assert page_size % swa_page_size == 0, (page_size, swa_page_size)
    return page_size // swa_page_size


# -- generations ----------------------------------------------


def page_generations(
    sub_allocator: Any, pages: Sequence[int], page_size: int
) -> List[int]:
    """Per-page generation for eviction detection.

    GPU paged allocators expose `generation_of(page_indices)`
    (mem_cache/allocator/paged.py, bumped on free); host slot allocators expose
    `page_generation(token_indices)` (mem_cache/pool_host/slot_allocator.py).
    Anything else (token allocators, test fakes) reports generation 0, so a
    reader cannot detect an owner eviction from the generation alone and
    relies on the Lease pin.
    """
    from decoupled_engine.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

    pages = [int(p) for p in pages]
    if not pages:
        return []
    if isinstance(sub_allocator, PagedTokenToKVPoolAllocator):
        return [int(g) for g in sub_allocator.generation_of(pages)]
    # Host slot allocators (duck-typed: pool_host/slot_allocator.py PageSlotAllocator).
    fn = getattr(sub_allocator, "page_generation", None)
    if callable(fn):
        import torch

        gens = list(fn(torch.tensor(pages, dtype=torch.int64) * page_size))
        if len(gens) == len(pages):
            return [int(g) for g in gens]
    return [0] * len(pages)


def make_generation_of(
    token_to_kv_pool_allocator: Any, *, page_size: int
) -> Callable[[str, Sequence[int]], List[int]]:
    """GpuLeaseTable(generation_of=...) callback: (pool, page_indices) -> the
    owner's current generation per page, from the allocator that owns the
    pool's index space. c4/c128 share the FULL page index (and allocator);
    SWA physical pages map to their allocator page by _swa_pages_per_page."""
    full_alloc, swa_alloc = _sub_allocators(token_to_kv_pool_allocator)
    kvcache = token_to_kv_pool_allocator.get_kvcache()
    swa_ratio = _swa_pages_per_page(kvcache, page_size) if swa_alloc is not None else 1

    def generation_of(pool: str, page_indices: Sequence[int]) -> List[int]:
        pages = [int(p) for p in page_indices]
        # -1 = "no page on this side" (pool_pages_for_range swa_skip_before):
        # never an allocator index, reported with generation 0.
        live = [p for p in pages if p >= 0]
        if pool in (POOL_FULL, POOL_C4, POOL_C128):
            gens = page_generations(full_alloc, live, page_size)
        elif pool == POOL_SWA and swa_alloc is not None:
            gens = page_generations(
                swa_alloc, [p // swa_ratio for p in live], page_size
            )
        else:
            # State rows are request-scoped (no allocator generation).
            return [0] * len(pages)
        if len(live) == len(pages):
            return gens
        it = iter(gens)
        return [next(it) if p >= 0 else 0 for p in pages]

    return generation_of


def enable_page_generation_tracking(token_to_kv_pool_allocator: Any) -> None:
    """Turn on free-time generation bumps on every GPU paged sub-allocator."""
    from decoupled_engine.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

    for alloc in _sub_allocators(token_to_kv_pool_allocator):
        if isinstance(alloc, PagedTokenToKVPoolAllocator):
            alloc.enable_page_generation_tracking()


# -- Registration layouts -----------------


def registration_pools_for(
    *, kvcache: Any, kv_args: Any
) -> Tuple[Optional[List[PoolLayout]], Dict[str, PoolRegion]]:
    """(pool_layouts for Registration.from_kv_args, extra page pools) of this
    rank's KVArgs. DSA shares one page index space for MLA KV and indexer
    bytes (including draft layers); both ride in the "full" pool. Other
    non-DSv4 models keep the default layout. DSv4:

    kv_data_ptrs (get_contiguous_buf_infos) is ordered [c4 kv x c4_layer_num,
    c4 indexer x c4_layer_num, c128 kv x c128_layer_num] and every entry is a
    2D [num_pages, bytes_per_page] buffer, so kv_item_lens[e] is already the
    page stride (mooncake/conn.py addresses `ptr + page_index * item_len`);
    the token-major default `item_len * page_size` does not apply. c4 and c128
    pages are addressed by the FULL page index; "full" itself owns no bytes.
    SWA pages live in the StateType.SWA state entries (get_state_buf_infos),
    one page of swa_page_size tokens per index, and become the "swa" pool.
    """

    if any(cls.__name__ == "DSATokenToKVPool" for cls in type(kvcache).__mro__):
        if kvcache.layer_shard_enabled:
            raise NotImplementedError("symmetric-pd DSA requires replicated device pools")
        if StateType.DSA not in kv_args.state_types:
            raise ValueError("symmetric-pd DSA requires KV and indexer buffers")
        i = kv_args.state_types.index(StateType.DSA)
        # Both APIs report bytes per PAGE, not per token. Keeping the
        # indexer in the same pool also shares its lease and generation check.
        return [], {
            POOL_FULL: PoolRegion(
                ptrs=list(kv_args.kv_data_ptrs) + list(kv_args.state_data_ptrs[i]),
                page_nbytes=list(kv_args.kv_item_lens)
                + list(kv_args.state_item_lens[i]),
            )
        }
    if any(cls.__name__ == "MLATokenToKVPool" for cls in type(kvcache).__mro__):
        # Native MLA buffer descriptors already use bytes per allocator page.
        # Supplying them as token strides would multiply page_size twice.
        return [
            PoolLayout(
                name=POOL_FULL,
                entry_indices=list(range(len(kv_args.kv_data_ptrs))),
                page_nbytes=list(kv_args.kv_item_lens),
            )
        ], {}
    if not _is_dsv4_pool(kvcache):
        return None, {}
    if (
        kvcache.c4_kv_pool is not None
        and "HiSparse" in type(kvcache.c4_kv_pool).__name__
    ):
        raise NotImplementedError(
            "symmetric-pd symmetric: DSv4 HiSparse c4 pages are remapped through a device "
            "mapping and are not addressable by the FULL page index"
        )
    stage_ratios = list(
        kvcache.compression_ratios[kvcache._stage_start : kvcache._stage_end]
    )
    c4_layer_num = sum(1 for r in stage_ratios if r == 4)
    c128_layer_num = sum(1 for r in stage_ratios if r == 128)
    num_entries = len(kv_args.kv_data_ptrs)
    if num_entries != 2 * c4_layer_num + c128_layer_num:
        raise NotImplementedError(
            f"symmetric-pd symmetric: DSv4 kv_data_ptrs has {num_entries} entries, expected "
            f"{2 * c4_layer_num + c128_layer_num} (c4 kv + c4 indexer + c128 kv); "
            "draft/MTP KV entries are not laid out yet"
        )
    item_lens = [int(n) for n in kv_args.kv_item_lens]
    c4_entries = list(range(0, 2 * c4_layer_num))
    c128_entries = list(range(2 * c4_layer_num, num_entries))
    layouts = [
        PoolLayout(name=POOL_FULL, entry_indices=[], page_nbytes=[]),
        PoolLayout(
            name=POOL_C4,
            entry_indices=c4_entries,
            page_nbytes=[item_lens[e] for e in c4_entries],
        ),
        PoolLayout(
            name=POOL_C128,
            entry_indices=c128_entries,
            page_nbytes=[item_lens[e] for e in c128_entries],
        ),
    ]
    extra: Dict[str, PoolRegion] = {}
    if not _is_unified_kv(kvcache):
        state_types = list(kv_args.state_types)
        if StateType.SWA not in state_types:
            raise NotImplementedError(
                "symmetric-pd symmetric: DSv4 without a StateType.SWA component"
            )
        i = state_types.index(StateType.SWA)
        extra[POOL_SWA] = PoolRegion(
            ptrs=[int(p) for p in kv_args.state_data_ptrs[i]],
            page_nbytes=[int(n) for n in kv_args.state_item_lens[i]],
        )
    return layouts, extra


# -- per-chunk page lists -----------------------------------------------------


def swa_tail_start(seq_len: int, *, sliding_window_size: int, page_size: int) -> int:
    """First token of the SWA tail an SWA-tail reader imports for a prompt of
    `seq_len` tokens. Owner (ExportTable.swa_retain_floor) and reader
    (ImportQueue._swa_tail_len) must agree on it: the owner hides SWA below it,
    the reader needs SWA from it on. One page more than the window, like the
    radix form of DecodePreallocQueue._swa_tail_len, so the last page is never
    all-tombstone."""
    return page_floor(
        max(0, seq_len - 1 - max(sliding_window_size, page_size)), page_size
    )


def owner_swa_valid_floor(req: Req, *, sliding_window_size: int, page_size: int) -> int:
    """Owner-side SWA validity floor for an export on a NON-uniform SWA tree:
    the first token whose SWA page is guaranteed live.

    `[cache_protected_len, kv.swa_evicted_seqlen)` was freed by the request
    (mem_cache/common.py free_swa_out_of_window_slots / decode prealloc), and
    of a matched prefix `[0, cache_protected_len)` only the last window is
    locked live by SWARadixCache.match_prefix (earlier SWA may be tombstoned).
    Translating those ranges through the cleared full->swa mapping would
    advertise page 0 with a plausible generation and a reader would READ
    garbage undetected.
    Uniform-lifecycle owners keep every SWA page and never use this (floor 0).
    """
    protected = req.cache_protected_len
    # free_swa_out_of_window_slots first raises swa_evicted_seqlen to the
    # protected prefix without freeing anything; the prefix window stays live.
    evicted = req.kv.swa_evicted_seqlen
    floor = evicted if evicted > protected else 0
    window = int(sliding_window_size or 0)
    if window > 0 and protected > window:
        prefix_floor = protected - window
        if page_size > 1:
            prefix_floor -= prefix_floor % page_size
        floor = max(floor, prefix_floor)
    return max(floor, 0)


def _window_floor(length: int, *, sliding_window_size: int, page_size: int) -> int:
    """page_floor(length - 1 - max(W, P)): the first page a window of W tokens
    ending at ``length`` can still read, kept one page below the page-aligned
    insert boundary (mem_cache/common.py free_swa_out_of_window_slots)."""
    span = max(int(sliding_window_size or 0), int(page_size))
    return page_floor(max(0, int(length) - 1 - span), page_size)


def window_export_swa_floor(
    req: Req, *, sliding_window_size: int, page_size: int
) -> int:
    """Export SWA floor F_exp of a DSv4 window-residency owner: SWA pages below
    it are announced -1 from the first CHUNK_READY on.

    F_exp = page_floor(L - 1 - max(W, P)) with L the prompt length. It equals
    the tail a window-residency reader imports SWA for (import_queue.py
    ImportQueue._swa_tail_len: T = L - tail), so the reader never needs an
    SWA page below it, and it depends only on L, so it is constant for the
    whole export: a page announced live is never announced -1 later. Below
    F_exp the owner may spill its SWA to the host (window_export_spill_range)
    and free the device slots, which is why those pages must never be
    announced. The owner validity floor (the matched prefix's window page,
    owner_swa_valid_floor) never exceeds F_exp; the max keeps the stricter one
    should it ever do so.
    """
    f_exp = _window_floor(
        len(req.origin_input_ids),
        sliding_window_size=sliding_window_size,
        page_size=page_size,
    )
    return max(
        owner_swa_valid_floor(
            req, sliding_window_size=sliding_window_size, page_size=page_size
        ),
        f_exp,
    )


def window_export_spill_range(
    *,
    prompt_len: int,
    e_prev: int,
    cps: int,
    sev: int,
    sliding_window_size: int,
    page_size: int,
) -> Tuple[int, int]:
    """Request SWA an exporting window-residency owner may spill ("export
    ledger"): [lo, hi) with lo = max(sev, cps) (the
    request's private SWA starts there; the ledger stays contiguous) and
    hi = min(page_floor(e_prev - 1 - max(W, P)), F_exp).

    ``e_prev`` is the end of the prompt whose KV is final (the previous chunk
    stash end, or the prompt length once the last chunk was processed). The
    chunk in flight reads SWA from e_prev - W on (plus the c4 ring a few
    tokens back), which is above hi, and pages at or above F_exp were or will
    be announced to the reader, so they stay on the device until release.
    Returns (lo, lo) when there is nothing to spill."""
    lo = max(int(sev), int(cps))
    hi = min(
        _window_floor(
            e_prev, sliding_window_size=sliding_window_size, page_size=page_size
        ),
        _window_floor(
            prompt_len, sliding_window_size=sliding_window_size, page_size=page_size
        ),
    )
    if hi <= lo:
        return lo, lo
    return lo, hi


def pool_pages_for_range(
    *,
    req_to_token_pool: Any,
    token_to_kv_pool_allocator: Any,
    req: Req,
    token_start: int,
    token_end: int,
    page_size: int,
    swa_skip_before: int = 0,
) -> List[PoolPages]:
    """Page indices per pool for `req`'s tokens [token_start, token_end).

    `swa_skip_before`: tokens below it have no live SWA pages on this side.
    Reader: D-role SWA-tail import (import_queue.py allocates SWA only for the
    sliding-window tail, like upstream `DecodePreallocQueue` with
    `alloc_extend_swa_tail`). Owner: `owner_swa_valid_floor` on a non-uniform
    SWA tree (out-of-window SWA freed by `free_swa_out_of_window_slots`,
    tombstoned prefix). Those SWA entries are reported as -1 with generation 0:
    importer._chunk_plan drops a pair the reader does not need and fails the
    import (layout_mismatch -> local recompute) when the reader needs a page
    the owner cannot supply; the FULL / c4 / c128 lists are unaffected.

    Same translation prefill.py uses for the push path (send_kv_chunk):
    req_to_token row -> `translate_kv_indices_for_transfer` (unified memory
    virtual->physical) -> `kv_to_page_indices`. DSv4: the
    c4 and c128 pools are addressed by the same page indices (and carry the
    same generations) as FULL; SWA is reached through
    `translate_loc_from_full_to_swa` and paged by swa_page_size. State rows are
    not chunk pages (see state_indices_for).

    M4 Q4 (mem_cache/cpu_page_book.py): the page ids come from the CPU rows
    and the generations from the allocator's CPU copy, with no device read
    (owner publications on the scheduler thread; reader lookups on the plane
    IO thread read rows that the import wrote before it started).
    """
    from decoupled_engine.srt.mem_cache.common import kv_to_page_indices
    from decoupled_engine.srt.mem_cache.cpu_page_book import cpu_rows_of

    pools: List[PoolPages] = []
    if token_end <= token_start:
        return pools

    full_alloc, swa_alloc = _sub_allocators(token_to_kv_pool_allocator)
    cpu_rows = cpu_rows_of(req_to_token_pool)
    if cpu_rows is not None:
        # Q4 runs only on the plain paged allocator (supported_reason): no
        # virtual ids, no SWA pool.
        if swa_alloc is not None:
            raise AssertionError("Q4 page bookkeeping: SWA pools are out of scope")
        kv_indices = None
        full_pages = [
            int(p)
            for p in cpu_rows.page_ids(int(req.req_pool_idx), token_start, token_end)
        ]
    else:
        kv_indices = req_to_token_pool.req_to_token[
            req.req_pool_idx, token_start:token_end
        ]
        # Unified memory: req_to_token holds VIRTUAL ids (allocator/base.py).
        kv_indices = token_to_kv_pool_allocator.translate_kv_indices_for_transfer(
            kv_indices
        )
        full_pages = [int(p) for p in kv_to_page_indices(kv_indices, page_size)]
    full_gens = page_generations(full_alloc, full_pages, page_size)
    pools.append(
        PoolPages(pool=POOL_FULL, page_indices=full_pages, generations=full_gens)
    )

    kvcache = token_to_kv_pool_allocator.get_kvcache()
    if swa_alloc is not None and not _is_unified_kv(kvcache):
        # Uniform lifecycle keeps SWA pages resident for the whole
        # sequence, so the per-chunk window translation is well defined.
        swa_ratio = _swa_pages_per_page(kvcache, page_size)
        swa_page_tokens = page_size // swa_ratio
        # D-role SWA-tail import: no SWA pages exist below swa_skip_before
        # (the full->swa mapping there is cleared), so translate only the tail
        # and pad the front with -1, one per SWA page the owner announces.
        cut = min(max(int(swa_skip_before) - token_start, 0), token_end - token_start)
        cut -= cut % swa_page_tokens
        swa_indices = token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            kv_indices[cut:]
        )
        swa_pages = [-1] * (cut // swa_page_tokens) + [
            int(p) for p in kv_to_page_indices(swa_indices, swa_page_tokens)
        ]
        live = [p for p in swa_pages if p >= 0]
        live_gens = page_generations(
            swa_alloc, [p // swa_ratio for p in live], page_size
        )
        pools.append(
            PoolPages(
                pool=POOL_SWA,
                page_indices=swa_pages,
                generations=[0] * (len(swa_pages) - len(live)) + live_gens,
            )
        )

    if _is_dsv4_pool(kvcache):
        for pool in (POOL_C4, POOL_C128):
            pools.append(
                PoolPages(
                    pool=pool,
                    page_indices=list(full_pages),
                    generations=list(full_gens),
                )
            )
    return pools


def dst_pages_for_range(
    *, include_state: bool = False, total_tokens: Optional[int] = None, **kwargs
) -> Dict[str, List[int]]:
    """Reader-side view of `pool_pages_for_range`: {pool: local page indices}.
    `include_state` merges the reader's `state_indices_for` rows under
    POOL_STATE (import_queue._local_state_indices reads them from here)."""
    out = {pp.pool: pp.page_indices for pp in pool_pages_for_range(**kwargs)}
    if include_state:
        out.update(
            state_indices_for(
                token_to_kv_pool_allocator=kwargs["token_to_kv_pool_allocator"],
                req=kwargs["req"],
                total_tokens=(
                    total_tokens if total_tokens is not None else kwargs["token_end"]
                ),
            )
        )
    return out


def state_indices_for(
    *, token_to_kv_pool_allocator: Any, req: Req, total_tokens: int
) -> Dict[str, List[int]]:
    """FinalMeta.state_indices for `req` (owner: the rows to publish; reader:
    the local rows to READ into), keyed by state_pool_key. DSv4 attaches the
    request-scoped C128 state rows (prefill.py _c128_state_payload); other
    models have none."""
    kvcache = token_to_kv_pool_allocator.get_kvcache()
    if not _is_dsv4_pool(kvcache):
        return {}
    from decoupled_engine.srt.disaggregation.utils import (
        get_dsv4_c128_state_indices,
        is_dsv4_c128_online_enabled,
    )

    online = is_dsv4_c128_online_enabled()
    ring_size = 1 if online else kvcache.get_ring_size(128)
    rows = get_dsv4_c128_state_indices(
        int(req.req_pool_idx), int(total_tokens), online=online, ring_size=ring_size
    )
    # No rows when total_tokens % 128 == 0 (prefill.py _c128_state_payload):
    # publish no state entry at all so owner and reader agree (otherwise
    # 1536-token prompts fail with "no matching local slots").
    if len(rows) == 0:
        return {}
    return {POOL_STATE: [int(r) for r in rows]}


class ExportEntry(msgspec.Struct, kw_only=True):
    req: Any  # Req
    transfer_id: int
    published_end: int = 0  # publication cursor, including negotiated reusable prefix
    publication_skipped_prefix_tokens: int = 0
    chunk_seq: int = 0
    exported_at: Optional[float] = None  # monotonic time of publish_final
    total_tokens: int = 0
    abort_reason: Optional[str] = None
    # Reader-first exports: clock time the P-leg was held (not queued) until
    # its reader's BEGIN; None once queued, or when not held.
    held_since: Optional[float] = None
    # Late D (--disaggregation-late-d): the P-leg binds late (held until its
    # reader allocates after the owner's ADMIT); its arrival position among
    # this worker's requests (every rank alike); and, on rank 0 only, the
    # table-clock time of its first ADMIT (bind timer).
    late: bool = False
    arrival_seq: int = -1
    first_admit_at: Optional[float] = None
    # Rank 0: which owner timer failed the late room (admission or bind).
    late_fail_phase: Optional[str] = None
    # Unlike held_since, this room may already own KV and an in-flight final
    # forward. BEGIN enables transfer; it must never enqueue this P leg again.
    compute_before_allocation: bool = False
    created_at: float = 0.0
    reader_ready: bool = False
    bound_at: float = 0.0
    pending_forwards: int = 0
    computed_end: int = 0
    prefill_finished_at: Optional[float] = None
    # page_floor(computed_end) when the reader became ready (_bind_computed):
    # the end of the KV computed before the bind. Under
    # SGLANG_SYMMETRIC_BUDGET_PRE_BIND_ONLY only [published_end, catchup_end)
    # is charged to the per-pass budget (not the standard P/D catch-up; see
    # ExportTable._publish_computed_pre_bind).
    catchup_end: int = 0
    # SGLANG_SYMMETRIC_BUDGET_STANDARD_PD: the export is in catch-up, as
    # prefill.py's late_d_catchup. Set at the bind when the export was
    # launched before it (a forward in flight, or computed KV that is not
    # published); cleared when its published KV reaches
    # page_floor(computed_end) (ExportTable._publish_computed_standard_pd).
    pd_catchup: bool = False
    # ... and the results still to come of the forwards that were in flight
    # at the bind. The catch-up cannot end before they are recorded:
    # prefill.py meters its catch-up only after it processed that result.
    pd_catchup_pending_results: int = 0
    # Pass index at which this room first had something to publish
    # (SGLANG_SYMMETRIC_PUBLISH_SMALL_FIRST_AGE_PASSES aging); -1 before.
    publishable_since_pass: int = -1


class _PublishBudgetWindow:
    """Counters of one SYMMETRIC_PUBLISH_BUDGET log window
    (SGLANG_SYMMETRIC_BUDGET_PRE_BIND_ONLY or
    SGLANG_SYMMETRIC_BUDGET_STANDARD_PD). A pass is one
    publish_computed(new_pass=True) call and the new_pass=False calls after
    it."""

    __slots__ = (
        "passes",
        "unbudgeted_passes",
        "unbudgeted_tokens",
        "max_pass_unbudgeted",
        "chunks",
        "max_chunk_tokens",
        "catchup_tokens",
        "catchup_waits",
        "finals",
    )

    def __init__(self) -> None:
        self.passes = 0
        self.unbudgeted_passes = 0
        self.unbudgeted_tokens = 0
        self.max_pass_unbudgeted = 0
        self.chunks = 0
        self.max_chunk_tokens = 0
        self.catchup_tokens = 0
        self.catchup_waits = 0
        self.finals = 0


class ExportTable:
    """Owner side of the Export lifecycle for one scheduler rank."""

    def __init__(
        self,
        owner_plane: OwnerPlane,
        *,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        metadata_buffers: Optional[MetadataBuffers],
        req_to_metadata_buffer_idx_allocator: Optional[ReqToMetadataIdxAllocator],
        page_size: int,
        export_timeout_s: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
        rank_consensus: Optional[RankConsensus] = None,
        swa_valid_floor: Optional[Callable[[Req], int]] = None,
        swa_tail_window: Optional[int] = None,
        swa_sentinel_guard: bool = False,
        owner_peer_kv_addr: Optional[str] = None,
        window_metrics: Any = None,
        await_reader_timeout_s: Optional[float] = None,
        late_d: bool = False,
        compute_before_allocation: bool = False,
        park_timeout_s: float = 300.0,
        catchup_tokens_per_pass: int = 77824,
        budget_pre_bind_only: bool = False,
        budget_standard_pd: bool = False,
        park_age_s: float = 0.0,
        park_aged_block_s: float = 30.0,
        owner_cache_epoch_getter: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        self.owner_plane = owner_plane
        # Late D: the await mode follows late D, so the BEGIN
        # collection, the hold and the held drain run with the F-C env unset.
        # Rank 0 also gets ``late_timer`` (SymOwnerTimer) and ``late_metrics``.
        self.late_d = bool(late_d)
        self.compute_before_allocation = bool(compute_before_allocation)
        self.park_timeout_s = park_timeout_s
        # Park-gate aging (SGLANG_LATE_D_PARK_AGE_S; 0 = off): poll_released
        # appends one aged vote per computing row (late_d_need.aged_vote).
        self.park_age_s = float(park_age_s)
        self.park_aged_block_s = float(park_aged_block_s)
        self.catchup_tokens_per_pass = catchup_tokens_per_pass
        # Catch-up budget left in the current pass, for a publish_computed
        # call that continues the pass (new_pass=False).
        self._pass_budget_left: Optional[int] = None
        # SGLANG_SYMMETRIC_BUDGET_PRE_BIND_ONLY (default off): the budget
        # meters only the KV computed before the bind (ExportEntry.catchup_end).
        self.budget_pre_bind_only = bool(budget_pre_bind_only)
        # SGLANG_SYMMETRIC_BUDGET_STANDARD_PD (default off): the budget meters
        # what standard P/D meters, the catch-up of an export launched before
        # its bind (ExportEntry.pd_catchup).
        self.budget_standard_pd = bool(budget_standard_pd)
        if self.budget_pre_bind_only and self.budget_standard_pd:
            raise ValueError(
                "SGLANG_SYMMETRIC_BUDGET_PRE_BIND_ONLY and "
                "SGLANG_SYMMETRIC_BUDGET_STANDARD_PD select different "
                "publication budget rules; set at most one of them"
            )
        # The counters of both rules: tokens published without the budget in
        # the current pass, the totals, and the window that the scheduler
        # logs (take_publish_budget_window).
        self._pass_unbudgeted_tokens = 0
        self.stats_unbudgeted_tokens = 0
        self.stats_catchup_tokens = 0
        self._budget_window = _PublishBudgetWindow()
        # publish_computed passes started (rank-identical: one per pre-schedule).
        self._pass_index = 0
        self.small_first_age_passes = int(
            envs.SGLANG_SYMMETRIC_PUBLISH_SMALL_FIRST_AGE_PASSES.get()
        )
        # The two budget rules publish in transfer-id order and do not count
        # passes, so the small-first order would have no effect with them.
        if self.small_first_age_passes > 0 and (
            self.budget_pre_bind_only or self.budget_standard_pd
        ):
            raise ValueError(
                "SGLANG_SYMMETRIC_PUBLISH_SMALL_FIRST_AGE_PASSES has no effect "
                "with SGLANG_SYMMETRIC_BUDGET_PRE_BIND_ONLY or "
                "SGLANG_SYMMETRIC_BUDGET_STANDARD_PD; set only one of them"
            )
        self.stats_after_result_calls = 0
        self.stats_after_result_publications = 0
        self.stats_after_result_finals = 0
        self.late_timer: Any = None
        self.late_metrics: Any = None
        self.stats_held_aux_wait = 0
        self.stats_late_timer_failed = 0
        self.stats_late_reader_gone = 0
        # Stage 2: held P-legs the owner ran itself (LOCAL, LC-2).
        self.stats_late_local = 0
        # Reader-first exports (F-C, SGLANG_SYMMETRIC_PD_EXPORT_AWAIT_READER; None =
        # off): the scheduler holds a new P-leg (hold) instead of queueing it,
        # and poll_begun hands it back once every rank saw its reader's BEGIN
        # or after this many seconds (fallback: queued as without the flag).
        self.await_reader_timeout_s = await_reader_timeout_s
        self._begun_locally: set = set()
        self._begun_out: List[Tuple[Req, str]] = []
        self.stats_held = 0
        self.stats_held_begun = 0
        self.stats_held_timeout = 0
        self.stats_held_released = 0
        self.stats_held_max_s = 0.0
        # Owners whose SWA can be missing below some token (scheduler_mixin.py):
        # req -> first token with a live SWA page; SWA pages below are
        # announced as -1. Non-uniform owners use owner_swa_valid_floor, DSv4
        # window-residency owners window_export_swa_floor. None = every SWA
        # page is live (uniform lifecycle, full residency).
        self.swa_valid_floor = swa_valid_floor
        # SWA-tail export: sliding window size when every reader imports SWA
        # only for the prompt's tail (ImportQueue.swa_tail_import); SWA below
        # swa_retain_floor is announced -1 from the first chunk on and freed by
        # trim_swa. None = readers may need every SWA page; keep them all.
        self.swa_tail_window = swa_tail_window
        self.stats_swa_floor_chunks = 0
        self.stats_swa_floor_pages = 0
        self.stats_swa_trimmed_tokens = 0
        # Sentinel guard (DSv4 window-residency owners): SWA pages at
        # or above the floor that translate to slot 0 (unmapped) are announced
        # -1 instead of page 0. Must stay 0; a reader that needs such a page
        # fails the import with layout_mismatch and recomputes. Off keeps the
        # announcements of other owners exactly as before.
        self.swa_sentinel_guard = bool(swa_sentinel_guard)
        self.stats_swa_sentinel_pages = 0
        # Written into FinalMeta: a window-residency
        # reader backfills the SWA below its received tail from this servicer.
        self.owner_peer_kv_addr = owner_peer_kv_addr
        # Read at completion, rather than freezing a pre-flush generation in
        # the long-lived table. The getter performs no IO or cache mutation.
        self.owner_cache_epoch_getter = owner_cache_epoch_getter
        # DSv4 window metrics (DSV4WindowMetricsCollector) or None.
        self.window_metrics = window_metrics
        # Device SWA held privately by exports (window-residency telemetry):
        # the last sample and the peak (swa_device_tokens()).
        self.stats_swa_tokens = 0
        self.stats_swa_peak_tokens = 0
        # Pages are freed only when EVERY TP rank saw its reader's DONE or
        # drained ABORT: a rank that frees early would let its pages be reused
        # while the peer rank's reader is still READing (symmetric/consensus.py).
        self.consensus = rank_consensus or _consensus.local_consensus()
        self._released_locally: set = set()
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.page_size = page_size
        self.export_timeout_s = export_timeout_s
        self.clock = clock
        self.entries: Dict[int, ExportEntry] = {}
        # Context for a chunk's page-index readbacks (_make_chunk); the
        # scheduler installs a side-stream one under
        # SGLANG_SYMMETRIC_PUBLISH_UNFENCED. None = the current stream.
        self.readback_stream_ctx: Optional[Callable[[], Any]] = None
        # SGLANG_SYMMETRIC_PUBLISH_ASYNC_READBACK: the scheduler installs a
        # publish_readback.ExportPageReader; _make_chunk asks it first and runs
        # the path above when it returns None. None = off.
        self.readback_page_reader: Optional[Any] = None

    # -- lifecycle ------------------------------------------------------
    def begin_export(
        self, req: Req, *, late: bool = False, arrival_seq: int = -1
    ) -> bool:
        """Register the P-leg request. Returns False if no aux slot is free.
        Late D: a late P-leg takes no aux slot here; _release_held takes
        it once its reader's BEGIN is agreed, so it never fails here."""
        mark = req.export_mark
        assert mark is not None
        if not late and self.req_to_metadata_buffer_idx_allocator is not None:
            idx = self.req_to_metadata_buffer_idx_allocator.alloc()
            if idx is None:
                return False
            req.metadata_buffer_index = idx
        # The aux slot's room field carries transfer_id so the reader can
        # validate the FINAL it read (MetadataBuffers.set_buf schema unchanged).
        req.bootstrap_room = mark.transfer_id
        req.export_stop_after_first_token = True
        self.entries[mark.transfer_id] = ExportEntry(
            req=req,
            transfer_id=mark.transfer_id,
            late=late,
            arrival_seq=arrival_seq,
            compute_before_allocation=late and self.compute_before_allocation,
            created_at=self.clock(),
        )
        self.owner_plane.begin_export(mark, req.rid)
        return True

    def publish_progress(self, req: Req, end_idx: int) -> Optional[ChunkReady]:
        """Publish the prefill chunk that just completed (pages [published_end,
        page_floor(end_idx))). Called once per finished middle chunk."""
        entry = self.entries.get(req.transfer_id)
        if entry is None:
            return None
        end = page_floor(min(end_idx, len(req.origin_input_ids)), self.page_size)
        if end <= entry.published_end:
            return None
        chunk = self._make_chunk(entry, entry.published_end, end, is_last=False)
        self.owner_plane.publish_chunk(chunk)
        entry.published_end = end
        entry.chunk_seq += 1
        return chunk

    def finish_export(self, req: Req) -> FinalMeta:
        """Last chunk + first token: publish the remaining pages with is_last,
        write the aux slot and publish FINAL. The request is now EXPORTED."""
        entry = self.entries[req.transfer_id]
        total = len(req.origin_input_ids)
        if req.extend_range is not None:
            total = min(total, req.extend_range.end)
        entry.total_tokens = total
        chunk = self._make_chunk(entry, entry.published_end, total, is_last=True)
        self.owner_plane.publish_chunk(chunk)
        entry.published_end = total
        entry.chunk_seq += 1
        if self.metadata_buffers is not None:
            self.metadata_buffers.set_buf(req)
        final = FinalMeta(
            transfer_id=entry.transfer_id,
            total_tokens=total,
            aux_index=req.metadata_buffer_index,
            state_indices=state_indices_for(
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                req=req,
                total_tokens=total,
            ),
            owner_peer_kv_addr=self.owner_peer_kv_addr,
            owner_cache_epoch=(
                self.owner_cache_epoch_getter()
                if self.owner_cache_epoch_getter is not None
                else None
            ),
        )
        self.owner_plane.publish_final(final)
        entry.exported_at = self.clock()
        return final

    def swa_retain_floor(self, req: Req) -> int:
        """Keep the reader's tail and the window at a reusable image boundary.

        A page-aligned prompt cache entry can end inside the last image. The
        next prefill backs up before that image, so keeping only the decode
        tail leaves its restart window missing. The request also exposes the
        earlier floor to release-time cache insertion, whose branch cleanup
        must not free that window before the final insert.
        """
        if self.swa_tail_window is None:
            return 0
        from decoupled_engine.srt.layers.attention.dsv4.visible_window import (
            image_restart_swa_retain_floor,
        )

        prompt_len = len(req.origin_input_ids)
        retain_floor = swa_tail_start(
            prompt_len,
            sliding_window_size=self.swa_tail_window,
            page_size=self.page_size,
        )
        image_floor = image_restart_swa_retain_floor(
            getattr(req, "multimodal_inputs", None),
            prompt_len,
            self.swa_tail_window,
            self.page_size,
        )
        if image_floor is not None:
            retain_floor = min(retain_floor, image_floor)
            req.swa_export_cache_retain_floor = retain_floor
        return retain_floor

    def swa_branch_window(self, req: Req) -> Optional[Tuple[int, int]]:
        """[page_floor(B - 1 - max(W, P)), B) for a pending SWA branching point B
        (Req.swa_branching_seqlen, set at match): the SWA window a standard
        prefill's chunk insert caches at B. An export inserts nothing until its
        reader is done, so trim_swa keeps this window for the release insert."""
        branch = req.swa_branching_seqlen
        if (
            self.swa_tail_window is None
            or branch is None
            or branch <= req.cache_protected_len
        ):
            return None
        start = _window_floor(
            branch, sliding_window_size=self.swa_tail_window, page_size=self.page_size
        )
        return start, branch

    def reopen_swa_branch_window(self, req: Req) -> None:
        """Make the window trim_swa kept at the branching point live again for the
        next insert: the trim moved swa_evicted_seqlen past it."""
        window = self.swa_branch_window(req)
        if window is not None and req.kv.swa_evicted_seqlen > window[0]:
            req.kv.swa_evicted_seqlen = max(window[0], req.cache_protected_len)

    def trim_swa(self, req: Req, pre_len: int) -> None:
        """SWA-tail export: free the exporting request's SWA below both the
        out-of-window frontier of its first `pre_len` tokens and
        swa_retain_floor. `_make_chunk` announces exactly those pages -1, so no
        lease pin or FINAL generation check covers them; the release-time
        insert tombstones them. Without it the request pins SWA for the whole
        prompt until the reader's DONE (no tree insert while exporting). The
        window at a pending branching point (swa_branch_window) stays: the
        frontier stops at it, then jumps past it, so it lies below
        swa_evicted_seqlen while still allocated."""
        if self.swa_tail_window is None:
            return
        held_from = max(req.kv.swa_evicted_seqlen, req.cache_protected_len)
        retain_floor = self.swa_retain_floor(req)
        window = self.swa_branch_window(req)
        kept = 0
        if window is None or req.kv.swa_evicted_seqlen >= window[1]:
            self._free_swa_below(req, pre_len, retain_floor)
        else:
            start, end = window
            self._free_swa_below(req, pre_len, min(retain_floor, start))
            # free_swa_out_of_window_slots' frontier for `pre_len`.
            frontier = page_floor(
                min(pre_len - max(self.swa_tail_window, self.page_size), retain_floor),
                self.page_size,
            )
            if req.kv.swa_evicted_seqlen >= start and frontier > end:
                kept = end - req.kv.swa_evicted_seqlen
                req.kv.swa_evicted_seqlen = end
                self._free_swa_below(req, pre_len, retain_floor)
        self.stats_swa_trimmed_tokens += max(
            0, req.kv.swa_evicted_seqlen - held_from - kept
        )

    def _free_swa_below(self, req: Req, pre_len: int, retain_floor: int) -> None:
        from decoupled_engine.srt.mem_cache.common import free_swa_out_of_window_slots

        free_swa_out_of_window_slots(
            req,
            pre_len,
            sliding_window_size=self.swa_tail_window,
            page_size=self.page_size,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            retain_floor=retain_floor,
        )

    def abort_export(self, req: Req, reason: str) -> bool:
        """Revoke an export; True means published pages must remain held."""
        entry = self.entries.get(req.transfer_id)
        if entry is None:
            return False
        entry.abort_reason = reason
        self.owner_plane.abort_export(entry.transfer_id, reason)
        if entry.compute_before_allocation and not entry.chunk_seq:
            # No descriptor was published, so the reader cannot have issued
            # a READ. Local final forwards still participate in release votes.
            self._released_locally.add(entry.transfer_id)
        if entry.chunk_seq or entry.compute_before_allocation:
            # The caller must not take the regular request-free path. Reuse
            # the exported lifecycle until all reader ranks acknowledge.
            if entry.exported_at is None:
                entry.exported_at = self.clock()
            return True
        self.entries.pop(req.transfer_id)
        self._released_locally.discard(req.transfer_id)
        self._begun_locally.discard(req.transfer_id)
        self._free_aux(req)
        return False

    # -- reader-first exports (F-C) ---------------------------------------
    @property
    def await_reader(self) -> bool:
        return self.await_reader_timeout_s is not None or self.late_d

    def hold(self, req: Req) -> None:
        """Keep a begun export out of the prefill queue until its reader
        admitted the import (poll_begun). The request holds its aux slot only:
        no KV, no lease."""
        entry = self.entries[req.transfer_id]
        entry.held_since = self.clock()
        self.stats_held += 1

    def held_reqs(self) -> List[Req]:
        return [e.req for e in self.entries.values() if e.held_since is not None]

    def late_unbound_reqs(self) -> List[Req]:
        """Late D: the held late P-legs, those waiting for an aux slot
        after their BEGIN included, in arrival order (the binder's unbound
        rooms)."""
        held = [
            e
            for e in self.entries.values()
            if e.late
            and (
                e.held_since is not None
                or (e.compute_before_allocation and not e.reader_ready)
            )
            and e.abort_reason is None
        ]
        held.sort(key=lambda e: e.arrival_seq)
        return [e.req for e in held]

    def load_view(self, waiting_queue: List[Req]) -> Dict[str, int]:
        """The export half of /v1/loads ``symmetric``: P-legs held for their
        reader (those waiting for an aux slot after BEGIN included), P-legs
        queued for prefill, and published exports, with their prompt tokens."""
        held_reqs = held_tokens = exported_reqs = exported_tokens = 0
        for entry in self.entries.values():
            tokens = len(entry.req.origin_input_ids)
            if entry.held_since is not None:
                held_reqs += 1
                held_tokens += tokens
            elif entry.exported_at is not None:
                exported_reqs += 1
                exported_tokens += tokens
        return {
            "export_held_reqs": held_reqs,
            "export_held_tokens": held_tokens,
            "export_queued_reqs": sum(
                1 for req in waiting_queue if req.export_mark is not None
            ),
            "exported_reqs": exported_reqs,
            "exported_tokens": exported_tokens,
        }

    def poll_begun(self) -> List[Tuple[Req, str]]:
        """Held P-legs to queue now, with why: READER_BEGUN (every rank saw the
        reader's BEGIN) or AWAIT_READER_TIMEOUT. Agreed in poll_released's
        consensus round, so call it right after poll_released."""
        out, self._begun_out = self._begun_out, []
        return out

    def record_compute_result(self, req: Req, end: int, *, last: bool) -> None:
        entry = self.entries[req.transfer_id]
        entry.computed_end = max(
            entry.computed_end, min(end, len(req.origin_input_ids))
        )
        req.late_d_computed_end = entry.computed_end
        if entry.pd_catchup_pending_results > 0:
            # SGLANG_SYMMETRIC_BUDGET_STANDARD_PD: the result of a forward
            # that was in flight at the bind.
            entry.pd_catchup_pending_results -= 1
        if last and entry.prefill_finished_at is None:
            entry.total_tokens = entry.computed_end
            entry.prefill_finished_at = self.clock()
            if not entry.reader_ready:
                req.late_d_park_started = time.perf_counter()
                logger.info(
                    "SYMMETRIC_COMPUTE park_start rid=%s room=%s tokens=%d",
                    req.rid,
                    entry.transfer_id,
                    entry.computed_end,
                )

    def chunks_published_total(self) -> int:
        """Chunks published so far by the live entries (statistics only)."""
        return sum(e.chunk_seq for e in self.entries.values())

    def publish_computed(self, *, new_pass: bool = True) -> int:
        """Only processed results are visible. One bounded catch-up per pass.

        A running later chunk may use these pages, but cannot overwrite the
        completed prefix. Published pages remain leased until reader drain.

        new_pass=False continues the current pass with its remaining budget
        (SGLANG_SYMMETRIC_PUBLISH_AFTER_RESULT: the call after a processed
        result), so a pass never publishes more than one budget. Returns the
        number of FINALs published.

        SGLANG_SYMMETRIC_BUDGET_PRE_BIND_ONLY: _publish_computed_pre_bind.
        SGLANG_SYMMETRIC_BUDGET_STANDARD_PD: _publish_computed_standard_pd.
        """
        if self.budget_pre_bind_only:
            return self._publish_computed_pre_bind(new_pass)
        if self.budget_standard_pd:
            return self._publish_computed_standard_pd(new_pass)
        bounded = self.catchup_tokens_per_pass > 0
        if new_pass or self._pass_budget_left is None:
            budget = (
                max(
                    self.page_size,
                    page_floor(self.catchup_tokens_per_pass, self.page_size),
                )
                if bounded
                else 0
            )
            self._pass_index += 1
        else:
            budget = self._pass_budget_left
        finals = 0
        order = sorted(self.entries.values(), key=lambda e: e.transfer_id)
        if self.small_first_age_passes > 0:
            order = self._small_first_order(order)
        for entry in order:
            if (
                not entry.compute_before_allocation
                or not entry.reader_ready
                or entry.abort_reason is not None
                or entry.exported_at is not None
            ):
                continue
            if bounded and budget <= 0:
                break
            before = entry.published_end
            end = entry.computed_end
            if bounded:
                end = min(end, before + budget)
            if entry.prefill_finished_at is not None and end >= entry.total_tokens:
                self.finish_export(entry.req)
                finals += 1
                logger.info(
                    "SYMMETRIC_COMPUTE final rid=%s room=%s tokens=%d",
                    entry.req.rid,
                    entry.transfer_id,
                    entry.total_tokens,
                )
            else:
                self.publish_progress(entry.req, end)
            if bounded:
                budget -= entry.published_end - before
        self._pass_budget_left = budget
        return finals

    def _publish_computed_pre_bind(self, new_pass: bool) -> int:
        """publish_computed with SGLANG_SYMMETRIC_BUDGET_PRE_BIND_ONLY.

        The per-pass budget meters only [published_end, catchup_end): the KV
        that the export had computed before its reader became ready.
        catchup_end is fixed at the bind. An export with such KV left and no
        budget waits for the next pass. When that KV ends in this call, the
        export continues: it publishes up to computed_end, and its FINAL
        once its prefill is done, without the budget, in the same call.

        This is not the standard P/D rule. prefill.py puts a request that it
        launched before its bind in catch-up (late_d_catchup, set in
        finalize_bootstrap). It does not send the results of that request at
        once (process_batch_result_disagg_prefill), and it meters
        [start_send_idx, late_d_computed_end) until the sent KV reaches the
        computed KV. That end moves with each result
        (late_d_need_flush_catchup). The differences:

        - More permissive here: the result of a forward that was in flight
          at the bind. prefill.py meters it, here it is not metered. The
          usual case is a multi-chunk export that binds during its first
          forward (computed 0 at the bind): here its whole first result (the
          cached prefix and the first chunk) goes out at once.
        - More permissive here: results that arrive while the catch-up
          drains. prefill.py meters them. Here they go out without the
          budget, in the chunk that ends the catch-up.
        - Stricter here: a parked export (prefill done before the bind).
          prefill.py sends all its KV at the bind without the budget
          (process_disagg_prefill_inflight_queue,
          send_kv_chunk(last_chunk=True)). Here that KV is metered.
        - Stricter here: the catch-up left at the last result. prefill.py
          sends it at once with the last chunk. Here it stays metered.
        - The same: an export bound before its first forward is not metered,
          and a one-chunk export that binds during its only forward
          publishes that result and FINAL without the budget.

        Publication stays contiguous (one cursor, published_end), so the KV
        after the catch-up never goes out before the catch-up. Every TP rank
        publishes the same chunks: the order is transfer_id, reader_ready and
        catchup_end change only in the consensus round (_bind_computed), and
        computed_end changes only at results, which every rank processes
        between the same two consensus rounds. No collective runs here.
        """
        bounded = self.catchup_tokens_per_pass > 0
        window = self._budget_window
        budget = self._budget_of_call(new_pass)
        finals = 0
        for entry in sorted(self.entries.values(), key=lambda e: e.transfer_id):
            if (
                not entry.compute_before_allocation
                or not entry.reader_ready
                or entry.abort_reason is not None
                or entry.exported_at is not None
            ):
                continue
            before = entry.published_end
            # Unbounded (budget 0 in the env): nothing is metered.
            catchup = max(before, entry.catchup_end) if bounded else before
            end = entry.computed_end
            if catchup > before:
                if budget <= 0:
                    window.catchup_waits += 1
                    continue
                if before + budget < catchup:
                    # Only part of the catch-up fits; the rest and all later
                    # KV wait for the next pass.
                    end = before + budget
                    window.catchup_waits += 1
            if entry.prefill_finished_at is not None and end >= entry.total_tokens:
                self.finish_export(entry.req)
                finals += 1
                window.finals += 1
                logger.info(
                    "SYMMETRIC_COMPUTE final rid=%s room=%s tokens=%d",
                    entry.req.rid,
                    entry.transfer_id,
                    entry.total_tokens,
                )
            else:
                self.publish_progress(entry.req, end)
            published = entry.published_end - before
            if published > 0:
                metered = max(0, min(entry.published_end, catchup) - before)
                budget -= metered
                self._note_publication(published, metered)
        self._pass_budget_left = budget
        return finals

    def _publish_computed_standard_pd(self, new_pass: bool) -> int:
        """publish_computed with SGLANG_SYMMETRIC_BUDGET_STANDARD_PD: the
        per-pass budget meters what prefill.py meters for a late-D request.

        - Catch-up. An export that was launched before its bind (a forward
          in flight, or computed KV that is not published) is in catch-up
          from the bind (ExportEntry.pd_catchup; prefill.py sets
          late_d_catchup in finalize_bootstrap). In catch-up, the budget
          meters [published_end, page_floor(computed_end)), and that end
          moves with each result (late_d_need_flush_catchup). The catch-up
          ends in the first call that leaves published_end at that end, but
          not before the results of the forwards that were in flight at the
          bind are recorded (pd_catchup_pending_results): prefill.py meters
          its catch-up only after it processed that result.
        - After the catch-up, and for an export that was bound before its
          first forward, the KV of each result goes out at once, as
          prefill.py sends the chunk of a bound request at its result.
        - FINAL, with all KV that is not published, goes out at once in the
          first call after the last result, also during a catch-up, as
          prefill.py sends the last chunk (send_kv_chunk(last_chunk=True)).
          A parked export (prefill done before the bind) publishes FINAL in
          the call of its bind round, as prefill.py sends all its KV at the
          bind (process_disagg_prefill_inflight_queue).

        Exports in catch-up share the pass budget in transfer-id order. An
        export with no budget left waits for the next pass, and the exports
        after it continue. Two differences from prefill.py stay: prefill.py
        serves its catch-ups in look-ahead plan order, and it sends the
        device-resident prefix at the bind without the budget
        (late_d_need_send_prefix). Here the cached prefix becomes
        publishable only at the first result (record_compute_result), so it
        is part of the catch-up.

        Every TP rank publishes the same chunks. The order is transfer_id.
        pd_catchup and pd_catchup_pending_results are set in the consensus
        round (_bind_computed) from pending_forwards, computed_end and
        published_end, which are the same on every rank there: all ranks
        launch and process the same batches between two rounds.
        pd_catchup_pending_results and computed_end change only when a
        result is recorded. This call reads neither pending_forwards nor a
        clock. No collective runs here.
        """
        bounded = self.catchup_tokens_per_pass > 0
        window = self._budget_window
        budget = self._budget_of_call(new_pass)
        finals = 0
        for entry in sorted(self.entries.values(), key=lambda e: e.transfer_id):
            if (
                not entry.compute_before_allocation
                or not entry.reader_ready
                or entry.abort_reason is not None
                or entry.exported_at is not None
            ):
                continue
            before = entry.published_end
            metered = 0
            if (
                entry.prefill_finished_at is not None
                and entry.computed_end >= entry.total_tokens
            ):
                # The last result, or a parked export: all of it at once.
                entry.pd_catchup = False
                self.finish_export(entry.req)
                finals += 1
                window.finals += 1
                logger.info(
                    "SYMMETRIC_COMPUTE final rid=%s room=%s tokens=%d",
                    entry.req.rid,
                    entry.transfer_id,
                    entry.total_tokens,
                )
            elif bounded and entry.pd_catchup:
                target = page_floor(entry.computed_end, self.page_size)
                if before < target:
                    if budget <= 0:
                        window.catchup_waits += 1
                        continue
                    self.publish_progress(entry.req, min(target, before + budget))
                    metered = entry.published_end - before
                    budget -= metered
                    if entry.published_end < target:
                        window.catchup_waits += 1
                if (
                    entry.pd_catchup_pending_results == 0
                    and entry.published_end >= target
                ):
                    entry.pd_catchup = False
            else:
                self.publish_progress(entry.req, entry.computed_end)
            published = entry.published_end - before
            if published > 0:
                self._note_publication(published, metered)
        self._pass_budget_left = budget
        return finals

    def _budget_of_call(self, new_pass: bool) -> int:
        """The budget of a publish_computed call under a budget rule
        (SGLANG_SYMMETRIC_BUDGET_PRE_BIND_ONLY or
        SGLANG_SYMMETRIC_BUDGET_STANDARD_PD): a new budget for a new pass
        (new_pass, or no pass yet), which also starts the pass in the
        counters; else what the current pass left. 0 when unbounded."""
        if not new_pass and self._pass_budget_left is not None:
            return self._pass_budget_left
        self._pass_unbudgeted_tokens = 0
        self._budget_window.passes += 1
        if self.catchup_tokens_per_pass <= 0:
            return 0
        return max(
            self.page_size,
            page_floor(self.catchup_tokens_per_pass, self.page_size),
        )

    def _note_publication(self, published: int, metered: int) -> None:
        """Counters of _publish_computed_pre_bind and
        _publish_computed_standard_pd: one chunk of ``published`` tokens, of
        which ``metered`` were charged to the budget."""
        window = self._budget_window
        unbudgeted = published - metered
        window.chunks += 1
        window.max_chunk_tokens = max(window.max_chunk_tokens, published)
        window.catchup_tokens += metered
        self.stats_catchup_tokens += metered
        if unbudgeted <= 0:
            return
        if self._pass_unbudgeted_tokens == 0:
            window.unbudgeted_passes += 1
        self._pass_unbudgeted_tokens += unbudgeted
        window.unbudgeted_tokens += unbudgeted
        window.max_pass_unbudgeted = max(
            window.max_pass_unbudgeted, self._pass_unbudgeted_tokens
        )
        self.stats_unbudgeted_tokens += unbudgeted

    @property
    def publish_budget_rule(self) -> Optional[str]:
        """The publication budget rule for SYMMETRIC_PUBLISH_BUDGET:
        "pre_bind" (SGLANG_SYMMETRIC_BUDGET_PRE_BIND_ONLY), "standard_pd"
        (SGLANG_SYMMETRIC_BUDGET_STANDARD_PD), or None (one budget for every
        publication, no counters)."""
        if self.budget_pre_bind_only:
            return "pre_bind"
        if self.budget_standard_pd:
            return "standard_pd"
        return None

    def take_publish_budget_window(self) -> Dict[str, int]:
        """publish_budget_rule set: the counters since the previous call (a
        fresh window starts), with the totals.

        passes: publish passes (pre-schedule calls); unbudgeted_passes: those
        that published KV without the budget; unbudgeted_tokens /
        max_pass_unbudgeted: that KV, summed and the largest per pass;
        chunks / max_chunk_tokens: chunks published (FINAL chunks included)
        and the largest one, a single READ burst on the reader;
        catchup_tokens: KV charged to the budget; catchup_waits: export calls
        whose catch-up the budget cut or skipped; finals: FINALs published.
        """
        window, self._budget_window = self._budget_window, _PublishBudgetWindow()
        out = {name: getattr(window, name) for name in _PublishBudgetWindow.__slots__}
        out["total_unbudgeted_tokens"] = self.stats_unbudgeted_tokens
        out["total_catchup_tokens"] = self.stats_catchup_tokens
        return out

    @staticmethod
    def _remaining_publication(entry: ExportEntry) -> int:
        end = (
            entry.total_tokens
            if entry.prefill_finished_at is not None
            else entry.computed_end
        )
        return max(0, end - entry.published_end)

    def _small_first_order(self, fifo: List[ExportEntry]) -> List[ExportEntry]:
        """SGLANG_SYMMETRIC_PUBLISH_SMALL_FIRST_AGE_PASSES: publishable rooms
        aged >= N passes first (FIFO), then the rest by smallest remaining
        publication (ties: transfer id). Only rank-identical state is used:
        bind, computed/published cursors and the pass index."""
        aged: List[ExportEntry] = []
        young: List[ExportEntry] = []
        for entry in fifo:
            if (
                not entry.compute_before_allocation
                or not entry.reader_ready
                or entry.abort_reason is not None
                or entry.exported_at is not None
            ):
                continue
            remaining = self._remaining_publication(entry)
            if remaining <= 0 and entry.prefill_finished_at is None:
                continue
            if entry.publishable_since_pass < 0:
                entry.publishable_since_pass = self._pass_index
            if (
                self._pass_index - entry.publishable_since_pass
                >= self.small_first_age_passes
            ):
                aged.append(entry)
            else:
                young.append(entry)
        young.sort(key=lambda e: (self._remaining_publication(e), e.transfer_id))
        return aged + young

    def _compute_state(self, entry: ExportEntry, now: float) -> int:
        if entry.abort_reason is not None:
            return _consensus.PENDING
        # Candidate losers may abort before BEGIN; only the committed winner
        # counts. Rank 0 knows the choice, and this vote propagates to all ranks.
        gave_up = getattr(self.owner_plane, "winner_gave_up", None)
        if callable(gave_up) and gave_up(entry.transfer_id):
            return _COMPUTE_READER_FAILED
        # Only the leader owns timers. Computing is not a parked request;
        # the retention deadline starts when its final result is processed.
        if self.late_timer is not None:
            if (
                entry.prefill_finished_at is not None
                and now - entry.prefill_finished_at >= self.park_timeout_s
            ):
                return _consensus.FAILED
            if (
                entry.req.late_d_first_chunk_started_perf == 0
                and now - entry.created_at >= self.late_timer.admission_s
            ):
                return _consensus.FAILED
        if entry.transfer_id in self._begun_locally:
            return _consensus.READY
        return _consensus.PENDING

    def _aged_state(self, entry: ExportEntry, now: float) -> int:
        """Park-gate aging vote of one computing row. Only the leader owns
        timers (as in _compute_state): every other rank votes the neutral
        AGED_NONE, so the MIN is rank 0's value whatever the local clocks."""
        if self.late_timer is None or entry.abort_reason is not None:
            return AGED_NONE
        return aged_vote(entry.req, now, self.park_age_s, self.park_aged_block_s)

    def _bind_computed(
        self, ids: List[int], states: List[int], now: float, prefixes=None
    ) -> None:
        for tid, state, prefix in zip(ids, states, prefixes or [0] * len(ids)):
            entry = self.entries.get(tid)
            if entry is None or state == _consensus.PENDING:
                continue
            if state in (_consensus.FAILED, _COMPUTE_READER_FAILED):
                reader_failed = state == _COMPUTE_READER_FAILED
                reason = (
                    COMPUTE_READER_GONE
                    if reader_failed
                    else (
                        COMPUTE_PARK_TIMEOUT
                        if entry.prefill_finished_at is not None
                        else COMPUTE_ABORT
                    )
                )
                if reader_failed:
                    entry.late_fail_phase = LATE_PHASE_READER_GONE
                    self.stats_late_reader_gone += 1
                logger.warning(
                    "SYMMETRIC_COMPUTE revoke rid=%s room=%s reason=%s pending_forwards=%d published_tokens=%d",
                    entry.req.rid,
                    tid,
                    reason,
                    entry.pending_forwards,
                    entry.published_end - entry.publication_skipped_prefix_tokens,
                )
                self.abort_export(entry.req, reason)
                entry.req.late_d_park_timeout = reason == COMPUTE_PARK_TIMEOUT
                continue
            allocator = self.req_to_metadata_buffer_idx_allocator
            if allocator is not None and entry.req.metadata_buffer_index < 0:
                idx = allocator.alloc()
                if idx is None:
                    continue
                entry.req.metadata_buffer_index = idx
            if entry.published_end == 0 and entry.chunk_seq == 0:
                limit = page_floor(
                    max(len(entry.req.origin_input_ids) - 1, 0), self.page_size
                )
                # PrefillAdder sets extend_range for each ordinary prefill
                # batch, including the first chunk before BEGIN arrives.
                # It is a compute range, not a partial-export contract.
                supported = (
                    self.swa_valid_floor is None and self.swa_tail_window is None
                )
                skip = (
                    page_floor(min(max(int(prefix), 0), limit), self.page_size)
                    if supported
                    else 0
                )
                entry.published_end = entry.publication_skipped_prefix_tokens = skip
                if skip:
                    logger.info(
                        "SYMMETRIC_COMPUTE publication_skip rid=%s room=%s tokens=%d",
                        entry.req.rid,
                        tid,
                        skip,
                    )
            # KV computed before the bind (the catch-up). computed_end moves
            # only at results, which every rank processes between the same
            # two consensus rounds, so this is the same on every rank.
            entry.catchup_end = page_floor(entry.computed_end, self.page_size)
            if self.budget_standard_pd and entry.prefill_finished_at is None:
                # prefill.py finalize_bootstrap: late_d_catchup for a request
                # launched before its bind. pending_forwards is the same on
                # every rank here: all ranks launch and process the same
                # batches between two consensus rounds.
                in_flight = entry.pending_forwards
                entry.pd_catchup = (
                    in_flight > 0 or entry.published_end < entry.catchup_end
                )
                entry.pd_catchup_pending_results = in_flight
            entry.reader_ready = True
            entry.bound_at = time.perf_counter()
            entry.req.pending_bootstrap = False
            entry.req.late_d_bind_perf = entry.bound_at
            if entry.req.late_d_park_started:
                entry.req.late_d_park_ended = entry.bound_at
                logger.info(
                    "SYMMETRIC_COMPUTE park_end rid=%s room=%s duration_s=%.6f",
                    entry.req.rid,
                    tid,
                    entry.bound_at - entry.req.late_d_park_started,
                )
            self._begun_locally.discard(tid)
            logger.info(
                "SYMMETRIC_COMPUTE reader_ready rid=%s room=%s computed=%d pending_forwards=%d",
                entry.req.rid,
                tid,
                entry.computed_end,
                entry.pending_forwards,
            )

    def _held_state(self, entry: ExportEntry, now: float) -> int:
        """MIN-reduced: READY once every rank saw BEGIN, FAILED (queue it
        anyway) as soon as one rank's wait exceeded the timeout. Late D: only
        rank 0 (``late_timer``) votes FAILED, by the admission or bind timer;
        the F-C timeout never applies to a late room."""
        if entry.late:
            gave_up = getattr(self.owner_plane, "winner_gave_up", None)
            if callable(gave_up) and gave_up(entry.transfer_id):
                # Rank 0 only (the rank that committed): the winner
                # aborted before its BEGIN. FAILED here fails the room on
                # every rank in this round (503, never a Local prefill).
                if entry.late_fail_phase is None:
                    entry.late_fail_phase = LATE_PHASE_READER_GONE
                return _consensus.FAILED
            timer = self.late_timer
            if timer is not None:
                verdict = timer.verdict(
                    created=entry.held_since, first_admit=entry.first_admit_at, now=now
                )
                if verdict is not None:
                    if entry.late_fail_phase is None:
                        entry.late_fail_phase = verdict[0]
                        logger.error(
                            "LATE_D owner timer: rid=%s transfer_id=%d %s, %.0f s",
                            getattr(entry.req, "rid", "?"),
                            entry.transfer_id,
                            verdict[0],
                            verdict[1],
                        )
                    return _consensus.FAILED
        if entry.transfer_id in self._begun_locally:
            return _consensus.READY
        timeout = self.await_reader_timeout_s
        if not entry.late and timeout is not None and now - entry.held_since > timeout:
            return _consensus.FAILED
        return _consensus.PENDING

    def _release_held(self, held: List[int], agreed: List[int], now: float) -> None:
        # Sorted transfer ids, the same on every rank: the late aux-slot
        # allocations below leave every rank's allocator in the same state.
        for transfer_id, state in zip(held, agreed):
            entry = self.entries.get(transfer_id)
            if state == _consensus.PENDING or entry is None:
                continue  # still waiting, or released by its reader this round
            if (
                state == _consensus.READY
                and entry.late
                and self.req_to_metadata_buffer_idx_allocator is not None
                and entry.req.metadata_buffer_index < 0
            ):
                # The aux slot comes at the bind. None free: stay held,
                # still begun, and retry next round (BEGUN_WAIT_AUX).
                idx = self.req_to_metadata_buffer_idx_allocator.alloc()
                if idx is None:
                    self.stats_held_aux_wait += 1
                    if self.late_metrics is not None:
                        self.late_metrics.held_aux_wait.inc()
                    continue
                entry.req.metadata_buffer_index = idx
            self.stats_held_max_s = max(self.stats_held_max_s, now - entry.held_since)
            entry.held_since = None
            self._begun_locally.discard(transfer_id)
            if state == _consensus.READY:
                self.stats_held_begun += 1
                self._note_late_bound(entry, now)
                self._begun_out.append((entry.req, READER_BEGUN))
            elif entry.late:
                self.stats_late_timer_failed += 1
                if self.late_metrics is not None:
                    self.late_metrics.late_timer_fail.inc()
                self._begun_out.append((entry.req, LATE_BIND_TIMEOUT))
            else:
                self.stats_held_timeout += 1
                self._begun_out.append((entry.req, AWAIT_READER_TIMEOUT))

    def _note_late_bound(self, entry: ExportEntry, now: float) -> None:
        """Rank 0: a late room that bound after 80% of its bind timer."""
        timer = self.late_timer
        if not entry.late or timer is None or entry.first_admit_at is None:
            return
        if now - entry.first_admit_at > 0.8 * timer.bind_s:
            logger.warning(
                "LATE_D bound near the bind timer: rid=%s transfer_id=%d %.0f s "
                "of %.0f s",
                getattr(entry.req, "rid", "?"),
                entry.transfer_id,
                now - entry.first_admit_at,
                timer.bind_s,
            )
            if self.late_metrics is not None:
                self.late_metrics.bind_near_timeout.inc()

    def late_fail_err_type(self, req: Req) -> str:
        """Rank 0 knows which owner timer failed a late P-leg (or that the
        stage-2 winner gave up before BEGIN); elsewhere the bind timer is
        reported."""
        from decoupled_engine.srt.disaggregation.late_d import (
            LATE_D_ADMISSION_TIMEOUT_ERR,
            LATE_D_BIND_TIMEOUT_ERR,
            LATE_D_READER_GONE_ERR,
        )

        entry = self.entries.get(req.transfer_id)
        phase = entry.late_fail_phase if entry is not None else None
        if phase == LATE_PHASE_READER_GONE:
            return LATE_D_READER_GONE_ERR
        if phase is not None and "admission" in phase:
            return LATE_D_ADMISSION_TIMEOUT_ERR
        return LATE_D_BIND_TIMEOUT_ERR

    def convert_to_local(self, transfer_id: int) -> Optional[Req]:
        """Late D stage 2, LOCAL (LC-2), every rank: forget a held,
        unpublished late P-leg so the owner can run it as a Local request.
        Returns the request, or None (the export is gone, began, failed or
        published something): the ``local`` control is then ignored."""
        entry = self.entries.get(transfer_id)
        if (
            entry is None
            or not entry.late
            or entry.held_since is None
            or entry.chunk_seq != 0
            or entry.abort_reason is not None
        ):
            return None
        self.entries.pop(transfer_id)
        self._begun_locally.discard(transfer_id)
        self._released_locally.discard(transfer_id)
        self._free_aux(entry.req)
        self.stats_late_local += 1
        return entry.req

    def poll_released(self) -> List[Tuple[Req, str]]:
        """Exports whose Lease ended on every rank: (req, RELEASED |
        EXPORT_TIMEOUT). The scheduler frees their pages.
        With reader-first exports the same consensus round
        also settles the held P-legs (poll_begun), and a held P-leg whose
        reader gave up comes back as RELEASED_WHILE_HELD."""
        for transfer_id in self.owner_plane.poll_released():
            if transfer_id in self.entries:
                self._released_locally.add(transfer_id)
            # else: duplicate DONE
        if self.await_reader:
            for transfer_id in self.owner_plane.poll_begun():
                entry = self.entries.get(transfer_id)
                if entry is not None and (
                    entry.held_since is not None or entry.compute_before_allocation
                ):
                    self._begun_locally.add(transfer_id)
        if not self.entries:
            return []
        now = self.clock()
        ids = sorted(self.entries)
        # Held P-legs, in the same rank-independent order (rank-local BEGINs
        # and timeouts are agreed below, never acted on directly).
        held = [t for t in ids if self.entries[t].held_since is not None]
        computing = [
            t
            for t in ids
            if self.entries[t].compute_before_allocation
            and not self.entries[t].reader_ready
        ]
        local: List[int] = []
        for transfer_id in ids:
            entry = self.entries[transfer_id]
            if transfer_id in self._released_locally:
                local.append(
                    _consensus.PENDING
                    if entry.req.inflight_middle_chunks > 0
                    or entry.pending_forwards > 0
                    else _consensus.READY
                )
            elif (
                self.export_timeout_s is not None
                and entry.exported_at is not None
                and entry.abort_reason is None
                and now - entry.exported_at > self.export_timeout_s
            ):
                entry.abort_reason = EXPORT_TIMEOUT
                self.owner_plane.abort_export(transfer_id, EXPORT_TIMEOUT)
                # A timeout starts revocation, never grants page reuse.
                local.append(_consensus.PENDING)
            else:
                local.append(_consensus.PENDING)
        # Append prefix ownership to the existing MIN vote. With the flag off
        # the vector shape and ordinary BEGIN behavior are unchanged.
        prefix_reader = getattr(self.owner_plane, "reusable_prefix_tokens", None)
        prefix_vote = envs.SGLANG_SYMMETRIC_SKIP_REUSED_PREFIX_PUBLICATION.get()
        prefixes = (
            [
                (
                    prefix_reader(t)
                    if callable(prefix_reader) and t in self._begun_locally
                    else 0
                )
                for t in computing
            ]
            if prefix_vote
            else []
        )
        # Park-gate aging: one more block on the same vote (flag off: none, so
        # the vector is unchanged). Rank 0 votes, the others vote AGED_NONE.
        aging = self.park_age_s > 0 and self.compute_before_allocation
        aged = (
            [self._aged_state(self.entries[t], now) for t in computing]
            if aging
            else []
        )
        agreed = self.consensus.agree(
            local
            + [self._held_state(self.entries[t], now) for t in held]
            + [self._compute_state(self.entries[t], now) for t in computing]
            + prefixes
            + aged
        )
        held_agreed = agreed[len(ids) : len(ids) + len(held)]
        start = len(ids) + len(held)
        computed_agreed = agreed[start : start + len(computing)]
        tail = start + len(computing)
        prefix_agreed = agreed[tail : tail + len(prefixes)] if prefix_vote else None
        if aging:
            tail += len(prefixes)
            for transfer_id, value in zip(computing, agreed[tail : tail + len(aged)]):
                # Every rank adopts the agreed value (lockstep admission input).
                self.entries[transfer_id].req.late_d_aged = value
        agreed = agreed[: len(ids)]
        out: List[Tuple[Req, str]] = []
        for transfer_id, state in zip(ids, agreed):
            if state != _consensus.READY:
                continue
            entry = self.entries.pop(transfer_id)
            self._released_locally.discard(transfer_id)
            if entry.compute_before_allocation:
                self._begun_locally.discard(transfer_id)
                self._free_aux(entry.req)
                reason = (
                    RELEASED
                    if entry.exported_at is not None and entry.abort_reason is None
                    else COMPUTE_ABORT
                )
                if reason != RELEASED:
                    self.owner_plane.abort_export(
                        transfer_id, entry.abort_reason or COMPUTE_ABORT
                    )
                logger.info(
                    "SYMMETRIC_COMPUTE release_safe rid=%s room=%s reason=%s published_tokens=%d",
                    entry.req.rid,
                    transfer_id,
                    entry.abort_reason or reason,
                    entry.published_end - entry.publication_skipped_prefix_tokens,
                )
                out.append((entry.req, reason))
            elif entry.held_since is not None:
                # Reader gave up before it admitted the import (pending
                # timeout / abort): the P-leg never ran and sits in no queue.
                self._begun_locally.discard(transfer_id)
                self.stats_held_released += 1
                self.owner_plane.abort_export(transfer_id, RELEASED_BEFORE_EXPORT)
                self._free_aux(entry.req)
                if entry.late:
                    # Rule X: 503, never a Local prefill.
                    self.stats_late_reader_gone += 1
                    if self.late_metrics is not None:
                        self.late_metrics.reader_gone.inc()
                    out.append((entry.req, LATE_READER_GONE))
                else:
                    out.append((entry.req, RELEASED_WHILE_HELD))
            elif entry.exported_at is None:
                # Not published yet (reader aborted early): no Lease to end,
                # aux slot goes back, KV stays with the running request.
                self.owner_plane.abort_export(transfer_id, RELEASED_BEFORE_EXPORT)
                self._free_aux(entry.req)
                out.append((entry.req, RELEASED_BEFORE_EXPORT))
            elif entry.abort_reason is not None:
                self._free_aux(entry.req)
                out.append((entry.req, entry.abort_reason))
            else:
                self._free_aux(entry.req)
                out.append((entry.req, RELEASED))
        if held:
            self._release_held(held, held_agreed, now)
        self._bind_computed(computing, computed_agreed, now, prefix_agreed)
        return out

    # -- introspection --------------------------------------------------
    def has_exported(self) -> bool:
        return any(
            e.exported_at is not None
            or (e.compute_before_allocation and e.req.is_holding_kv)
            for e in self.entries.values()
        )

    def exported_reqs(self) -> List[Req]:
        return [
            e.req
            for e in self.entries.values()
            if e.exported_at is not None or e.compute_before_allocation
        ]

    def computed_reqs(self) -> List[Req]:
        """The exports whose prefill compute is done and whose KV the reader
        has not released: FINAL published, or (compute-first) the last result
        recorded. Aborted entries are left out. In table order; read only
        (the inflight rows of the symmetric routing snapshot)."""
        return [
            e.req
            for e in self.entries.values()
            if e.abort_reason is None
            and (e.exported_at is not None or e.prefill_finished_at is not None)
        ]

    def swa_device_tokens(self) -> int:
        """Device SWA the requests of this table hold privately:
        [max(swa_evicted_seqlen, cache_protected_len), ceil_page(allocated))
        plus the slots their unacked spills detached (freed only at the
        MIN-reduced ack). Samples it, tracks the peak and publishes the peak
        gauge. _export_spill samples before and after each spill."""
        page = self.page_size
        total = 0
        for entry in self.entries.values():
            req = entry.req
            kv = getattr(req, "kv", None)
            if kv is None or not getattr(req, "is_holding_kv", False):
                continue
            allocated = kv.kv_allocated_len
            if page > 1:
                allocated = -(-allocated // page) * page
            base = max(kv.swa_evicted_seqlen, req.cache_protected_len)
            total += max(0, allocated - base)
            ledger = getattr(req, "swa_ledger", None)
            inflight = getattr(ledger, "inflight_ops", None)
            if callable(inflight):
                total += sum(op.num_tokens for op in inflight())
        self.stats_swa_tokens = total
        if total > self.stats_swa_peak_tokens:
            self.stats_swa_peak_tokens = total
            metrics = self.window_metrics
            if metrics is not None:
                metrics.export_swa_peak_tokens.labels(**metrics.labels).set(total)
        return total

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, transfer_id: int) -> bool:
        return transfer_id in self.entries

    # -- helpers --------------------------------------------------------
    def _guard_swa_sentinels(self, entry: ExportEntry, pools: List[PoolPages]) -> None:
        """Sentinel guard. Pages below the
        floor are already -1; an SWA page at or above it translating to slot 0
        means its full->swa mapping is cleared (a bug: it was freed or spilled
        while announced). Announce it -1 with generation 0 so a reader that
        needs it fails with layout_mismatch and recomputes, instead of READing
        slot 0's bytes under a plausible generation."""
        for pp in pools:
            if pp.pool != POOL_SWA:
                continue
            hits = [i for i, p in enumerate(pp.page_indices) if p == 0]
            if not hits:
                continue
            for i in hits:
                pp.page_indices[i] = -1
                pp.generations[i] = 0
            self.stats_swa_sentinel_pages += len(hits)
            metrics = self.window_metrics
            if metrics is not None:
                metrics.swa_sentinel_pages.labels(**metrics.labels).inc(len(hits))
            logger.error(
                "symmetric-pd export rid=%s transfer_id=%d: %d SWA page(s) at or above "
                "the export floor are unmapped (slot 0); announced -1 "
                "(symmetric_pd_swa_sentinel_pages_total must stay 0)",
                getattr(entry.req, "rid", "?"),
                entry.transfer_id,
                len(hits),
            )

    def _free_aux(self, req: Req) -> None:
        if (
            self.req_to_metadata_buffer_idx_allocator is not None
            and req.metadata_buffer_index >= 0
        ):
            self.req_to_metadata_buffer_idx_allocator.free(req.metadata_buffer_index)
            req.metadata_buffer_index = -1

    def _make_chunk(
        self, entry: ExportEntry, start: int, end: int, *, is_last: bool
    ) -> ChunkReady:
        floor = 0
        if self.swa_valid_floor is not None:
            floor = int(self.swa_valid_floor(entry.req) or 0)
        floor = max(floor, self.swa_retain_floor(entry.req))
        pools = None
        reader = self.readback_page_reader
        if reader is not None:
            # SGLANG_SYMMETRIC_PUBLISH_ASYNC_READBACK: copy-engine reads after
            # the last pre-WAR mark; None for a shape it does not cover.
            pools = reader.read(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                req=entry.req,
                token_start=start,
                token_end=end,
                page_size=self.page_size,
                swa_skip_before=min(floor, end),
            )
        if pools is None:
            ctx = self.readback_stream_ctx
            with ctx() if ctx is not None else contextlib.nullcontext():
                pools = pool_pages_for_range(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    req=entry.req,
                    token_start=start,
                    token_end=end,
                    page_size=self.page_size,
                    swa_skip_before=min(floor, end),
                )
        if floor > start:
            hidden = sum(
                1
                for pp in pools
                if pp.pool == POOL_SWA
                for p in pp.page_indices
                if p < 0
            )
            if hidden:
                self.stats_swa_floor_chunks += 1
                self.stats_swa_floor_pages += hidden
        if self.swa_sentinel_guard:
            self._guard_swa_sentinels(entry, pools)
        return ChunkReady(
            transfer_id=entry.transfer_id,
            chunk_seq=entry.chunk_seq,
            token_start=start,
            token_end=end,
            pools=pools,
            is_last=is_last,
            tier="gpu",
        )
