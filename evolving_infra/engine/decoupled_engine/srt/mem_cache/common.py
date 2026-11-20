# Adapted from SGLang python/sglang/srt/mem_cache/common.py (commit 297e54fd10), modified for evolving-infra.
from __future__ import annotations

import logging
from array import array
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Sequence, cast

import numpy as np
import torch

from decoupled_engine.kernels.ops.memory.common import (
    _get_last_loc_safe_kernel as _get_last_loc_safe_kernel,
)
from decoupled_engine.kernels.ops.memory.common import get_last_loc_kernel as get_last_loc_kernel
from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.managers.scheduler_components.phase_timers import timed_phase
from decoupled_engine.srt.mem_cache.allocator.page_interleave import page_interleave_shard_size
from decoupled_engine.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    EvictResult,
    MatchPrefixParams,
    MatchResult,
    zero_match_result,
)
from decoupled_engine.srt.mem_cache.hicache_storage import PoolTransfer
from decoupled_engine.srt.mem_cache.memory_pool import ReqToTokenPool
from decoupled_engine.srt.mem_cache.radix_cache import RadixKey
from decoupled_engine.srt.mem_cache.unified_cache.component_type import ComponentType
from decoupled_engine.srt.runtime_context import get_parallel, get_serving, get_spec
from decoupled_engine.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from decoupled_engine.srt.managers.schedule_batch import Req
    from decoupled_engine.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from decoupled_engine.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)


class RetractionBackup(NamedTuple):
    cpu_tensors: Any = None
    host_indices: Optional[torch.Tensor] = None
    pool_transfers: Optional[list[PoolTransfer]] = None
    # Set when the KV pool leaves the recurrent state to the caller.
    mamba_cpu: Any = None


@timed_phase("S3_kv_to_page_indices")
def kv_to_page_indices(kv_indices: torch.Tensor, page_size: int) -> np.ndarray:
    return (kv_indices[::page_size] // page_size).cpu().numpy()


def suffix_page_indices(
    req_to_token: torch.Tensor,
    req_pool_idx: int,
    prefix_len: int,
    seq_len: int,
    page_size: int,
) -> np.ndarray:
    """Pages of a request's tokens [prefix_len, seq_len).

    PD prefill and decode both call this with the same page-aligned prefix
    (decode_prefix_len), so their lists pair up page by page even though their
    rows point at different physical pages.
    """
    return kv_to_page_indices(req_to_token[req_pool_idx, prefix_len:seq_len], page_size)


def kv_to_page_num(num_kv_indices: int, page_size: int):
    return (num_kv_indices + page_size - 1) // page_size


def page_align_floor(length: int, page_size: int) -> int:
    return (length // page_size) * page_size


def free_swa_out_of_window_slots(
    req: Req,
    pre_len: int,
    *,
    sliding_window_size: int,
    page_size: int,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    supports_prefix_sharing: bool = True,
    retain_floor: int | None = None,
    component_type: ComponentType = ComponentType.SWA,
    free_segment: Callable[..., None] | None = None,
    eviction_interval: int = 1,
) -> None:
    if not req.kv.holds_kv:
        return

    # For SWA-capable tree caches, we need to evict the tokens that are not in the tree cache and also not in the sliding window
    assert req.kv.cache_protected_len % page_size == 0, (
        "cache_protected_len must be page aligned"
    )
    # Protected rows limit what can be freed, not where the interval starts.
    evicted_seqlen = req.kv.get_evicted_seqlen(component_type)
    if pre_len - sliding_window_size < evicted_seqlen + eviction_interval:
        return
    dead_lo = (
        req.kv.swa_dead_lo(page_size)
        if component_type == ComponentType.SWA
        else req.kv.cache_protected_len
    )
    evicted_seqlen = max(evicted_seqlen, dead_lo)
    req.kv.set_evicted_seqlen(component_type, evicted_seqlen)

    if not supports_prefix_sharing:
        # Nothing is inserted into a tree, so no tombstone-leaf concern; evict
        # up to the window boundary (the trailing floor keeps it page-aligned).
        evict_threshold = pre_len - sliding_window_size
    else:
        # Prefix-sharing cache: keep max(window, page) rounded up to whole pages, so
        # a key that ends at page_floor(pre_len) still has a full window of live SWA,
        # and the last leaf, below page_floor(seq_len), is never all-tombstone.
        evict_threshold = pre_len - ceil_align(
            max(sliding_window_size, page_size), page_size
        )
    if retain_floor is not None and supports_prefix_sharing:
        # The caller owns where the floor is (see BasePrefixCache.swa_retain_floor);
        # this only promises not to free past it. Without prefix sharing a retained
        # checkpoint could never be matched, so holding it is pure cost.
        evict_threshold = min(evict_threshold, retain_floor)

    new_evicted_seqlen = max(evicted_seqlen, evict_threshold)

    if page_size > 1:
        new_evicted_seqlen = (new_evicted_seqlen // page_size) * page_size

    if new_evicted_seqlen > evicted_seqlen:
        free_slots = req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, evicted_seqlen:new_evicted_seqlen
        ]
        if free_segment is None:
            assert component_type == ComponentType.SWA
            free_segment = token_to_kv_pool_allocator.free_swa_segment
        free_segment(free_slots, start_pos=evicted_seqlen)
        req.kv.set_evicted_seqlen(component_type, new_evicted_seqlen)


def start_deferring_chunk_backups(tree_cache: BasePrefixCache) -> bool:
    """SGLANG_DSV4_CHUNK_WINDOW_WORK_AFTER_LAUNCH, for an overlap loop that
    calls tree_cache.run_deferred_chunk_backups() after every launch and in
    every iteration without one: True when the cache now queues its window
    chunk-boundary backups (UnifiedRadixCache in window residency). Other
    caches keep everything inline."""
    defer = getattr(tree_cache, "defer_chunk_backups_after_launch", None)
    return defer is not None and defer() is True


def _skips_unfinished_insert(req: Req) -> bool:
    # content_only freeze (PD decode radix): unfinished inserts stay off for the
    # request's lifetime; the frozen prompt publishes once at successful finish.
    return getattr(req, "skip_radix_cache_insert", False) or getattr(
        req, "decode_radix_cache_skip_unfinished_insert", False
    )


def _write_through_ctx(req: Req, tree_cache: BasePrefixCache):
    # symmetric P/D D-leg `cache_policy=gpu_only` (import_queue.py): the imported
    # prompt is inserted (reader-side reuse) but not backed up to host.
    if getattr(req, "hicache_skip_write_through", False):
        return tree_cache.suppress_write_through()
    return nullcontext()


def maybe_cache_unfinished_req(req: Req, tree_cache: BasePrefixCache, **kwargs):
    """Checkpoint the request's fill ids up to ``req.extend_end``, as
    ``checkpoint_kv_cache`` does, without asserting that the request is
    unfinished: the symmetric export finish path calls it before
    ``release_kv_cache``. Keyword arguments are ignored."""
    if not _skips_unfinished_insert(req):
        with _write_through_ctx(req, tree_cache):
            tree_cache.checkpoint(req, up_to=req.extend_end)
    if req.extend_end is not None:
        req.prefix_len = req.extend_end


def coalesce_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge adjacent half-open ranges so a split that falls mid-page frees that page once."""
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start == merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def free_kv_row_segments(
    allocator: BaseTokenToKVPoolAllocator,
    segments: list[tuple[torch.Tensor, int]],
    *,
    swa_evicted_seqlen: int,
    swa_dead_lo: int = 0,
) -> None:
    """Free ascending disjoint ``(kv_indices, start_pos)`` segments of one
    request's kv row; ``[swa_dead_lo, swa_evicted_seqlen)`` goes back full-side only."""
    dead_lo, dead_hi = swa_dead_lo, max(swa_evicted_seqlen, swa_dead_lo)
    swa_dead: list[tuple[torch.Tensor, int]] = []
    swa_alive: list[tuple[torch.Tensor, int]] = []
    for kv_indices, start_pos in segments:
        end_pos = start_pos + kv_indices.numel()
        lo = min(max(dead_lo, start_pos), end_pos)
        hi = min(max(dead_hi, start_pos), end_pos)
        # start_pos <= lo <= hi <= end_pos
        # inside [lo, hi) is the swa dead segment
        if hi <= lo:
            swa_alive.append((kv_indices, start_pos))
            continue
        swa_dead.append((kv_indices[lo - start_pos : hi - start_pos], lo))

        if lo > start_pos:
            swa_alive.append((kv_indices[: lo - start_pos], start_pos))
        if end_pos > hi:
            swa_alive.append((kv_indices[hi - start_pos :], hi))

    if swa_dead and swa_alive:
        # The two sides are separate calls, so neither one's page-disjointness
        # check sees a boundary that splits a page between them.
        assert swa_evicted_seqlen % allocator.page_size == 0, (
            f"SWA eviction cursor {swa_evicted_seqlen} splits a page "
            f"(page_size {allocator.page_size})"
        )
    if swa_dead:
        allocator.free_full_segments(swa_dead)
    if swa_alive:
        allocator.free_segments(swa_alive)


def _req_radix_key(
    tree_cache: BasePrefixCache,
    req: Req,
    token_ids: array[int],
    max_prefix_len: Optional[int] = None,
) -> RadixKey:
    # A reused prefix may carry stale or unwritten SWA (the unified_kv ring, decoder
    # bounded replay); cap the match so the window is re-prefilled.
    reprefill_tail = tree_cache.swa_reprefill_tail_tokens(req)
    key_limit = max(0, len(token_ids) - reprefill_tail) if reprefill_tail else None
    if max_prefix_len is not None:
        key_limit = (
            max_prefix_len if key_limit is None else min(key_limit, max_prefix_len)
        )
    return RadixKey(
        token_ids=token_ids,
        extra_key=req.extra_key,
        limit=key_limit,
        cache_salt=req.cache_salt,
    )


def touch_waiting_prefix(req: Req, tree_cache: BasePrefixCache) -> None:
    tree_cache.touch_prefix(
        _req_radix_key(tree_cache, req, req.origin_input_ids + req.output_ids)
    )


def match_kv_cache(
    req: Req,
    tree_cache: BasePrefixCache,
    token_ids: Optional[array] = None,
    *,
    cow_mamba: bool = False,
    max_prefix_len: Optional[int] = None,
    active_components: Optional[Sequence[ComponentType]] = None,
) -> MatchResult:
    """Match token_ids against tree_cache and adopt the hit as req's prefix.

    EI: ``active_components`` restricts the match to those components (the
    content-only match of PD decode radix and symmetric imports; it must
    include FULL). None matches every registered component."""
    if token_ids is None:
        token_ids = req.origin_input_ids + req.output_ids

    match_result = tree_cache.match_prefix(
        MatchPrefixParams(
            key=_req_radix_key(tree_cache, req, token_ids, max_prefix_len),
            cow_mamba=cow_mamba,
            req=req,
            active_components=(
                None if active_components is None else tuple(active_components)
            ),
        )
    )
    if envs.SGLANG_RADIX_FORCE_MISS.get():
        match_result = zero_match_result(
            tree_cache, match_result, extra_key=req.extra_key
        )
    req.prefix_len = match_result.device_prefix_len
    (
        req.last_node,
        req.last_host_node,
        req.best_match_node,
        req.host_hit_length,
        req.swa_host_hit_length,
        req.mamba_host_hit_length,
    ) = (
        match_result.last_device_node,
        match_result.last_host_node,
        match_result.best_match_node,
        match_result.host_hit_length,
        match_result.swa_host_hit_length,
        match_result.mamba_host_hit_length,
    )
    max_len = req._compute_max_prefix_len(len(token_ids))
    req.num_matched_prefix_tokens = min(req.prefix_len + req.host_hit_length, max_len)
    req.swa_branching_seqlen = match_result.swa_branching_seqlen
    # A probe match keeps what it did not report; a new round resets both.
    if match_result.mamba_branching_seqlen is not None:
        req.mamba_branching_seqlen = match_result.mamba_branching_seqlen
    if match_result.cache_protected_len is not None:
        req.kv.cache_protected_len = match_result.cache_protected_len
    return match_result


def checkpoint_kv_cache(req: Req, tree_cache: BasePrefixCache) -> None:
    """Publish what the running request has computed so far, unless it is
    barred from the tree."""
    # The tree reads req.finished() to tell a checkpoint from the final
    # insert; a finished request belongs in release_kv_cache.
    assert not req.finished(), f"checkpointing finished request {req.rid}"
    # EI: a content_only decode-radix request also stays out of the tree, and
    # a symmetric P/D D-leg gpu_only request skips the host write-through.
    if not _skips_unfinished_insert(req):
        with _write_through_ctx(req, tree_cache):
            tree_cache.checkpoint(req, up_to=req.extend_end)
    # The next extend resumes after this one, published or not.
    req.prefix_len = req.extend_end


def evict_from_tree_cache(
    tree_cache: BasePrefixCache | None, num_tokens: int
) -> bool | None:
    if tree_cache is not None and tree_cache.supports_prefix_sharing():
        return tree_cache.token_to_kv_pool_allocator.evict_to_free_tokens(
            tree_cache, num_tokens
        )


def _evict_until_allocatable(
    tree_cache: BasePrefixCache, allocator, num_tokens: int
) -> EvictResult:
    """Keep evicting until `num_tokens` are allocatable; return all evicted counts.

    Under classed page sharding available_size() reports the MIN-CLASS
    capacity floor, so a single evict() sized in tokens can raise that floor by
    less than the number of tokens it freed: the evicted pages spread across
    all owner classes. Looping is deterministic, so it stays mirrored across
    the ranks of a shard group. Stock allocators need no extra pass.
    """
    total = EvictResult()
    if page_interleave_shard_size(allocator) <= 1:
        return total
    while True:
        available_size = allocator.available_size()
        if available_size >= num_tokens:
            return total
        shortfall = num_tokens - available_size
        result = tree_cache.evict(
            EvictParams(num_tokens=max(shortfall, allocator.page_size))
        )
        total.num_tokens_evicted += result.num_tokens_evicted
        total.swa_num_tokens_evicted += result.swa_num_tokens_evicted
        total.mamba_num_evicted += result.mamba_num_evicted
        if result.num_tokens_evicted == 0:
            return total


_GIB = 1 << 30


class RetractBackupLedger:
    """Host bytes of the cpu_tensor retraction backups that this TP rank holds,
    by request id, while SGLANG_RETRACT_BACKUP_MAX_GB is set.

    Only scheduler events change it, and every TP rank runs them in the same
    order: the backup (backup_kv_cache), the resume (restore_kv_cache),
    the discard of a retracted request (discard_kv_cache_backup) and the abort of a
    retracted request when it is queued again (DecodePreallocQueue.add). So
    every TP rank has the same ledger. Host memory state and garbage
    collection do not enter."""

    def __init__(self) -> None:
        self.nbytes_by_rid: dict[str, int] = {}
        self.num_cap_aborts = 0

    def outstanding_nbytes(self) -> int:
        return sum(self.nbytes_by_rid.values())

    def clear(self) -> None:
        self.nbytes_by_rid.clear()
        self.num_cap_aborts = 0


RETRACT_BACKUP_LEDGER = RetractBackupLedger()
_cap_unsupported_warned = False


def _is_cap_log_rank() -> bool:
    """The RETRACT_BACKUP_CAP lines: one line per event, from attn-TP rank 0."""
    try:
        return get_parallel().attn_tp_rank == 0
    except Exception:  # parallel state not initialized (unit tests)
        return True


def _pp_size() -> int:
    try:
        return int(get_parallel().pp_size)
    except (AttributeError, ValueError):  # not published (unit tests)
        return 1


def _projected_backup_nbytes(
    req: Req, token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
) -> Optional[int]:
    """Host bytes of this request's cpu_tensor backup on this TP rank, or None
    when there is no projection that is the same on every rank: the KV pool
    has none (cpu_copy_nbytes), or pipeline parallel stages hold other
    layers."""
    global _cap_unsupported_warned
    kvcache = token_to_kv_pool_allocator.get_kvcache()
    project = getattr(kvcache, "cpu_copy_nbytes", None)
    nbytes = None
    if project is not None and _pp_size() == 1:
        # offload_kv_cache copies req_to_token[req_pool_idx, : seqlen - 1].
        nbytes = project(max(int(req.seqlen) - 1, 0))
    if nbytes is not None:
        return int(nbytes)
    if not _cap_unsupported_warned:
        _cap_unsupported_warned = True
        logger.warning(
            "SGLANG_RETRACT_BACKUP_MAX_GB is set, but the retraction backup of "
            "%s (pp_size %d) has no projection that is the same on every rank: "
            "the backup is not capped",
            type(kvcache).__name__,
            _pp_size(),
        )
    return None


def log_retract_backup_cap_config(
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator, backend: str
) -> None:
    """Decode worker startup, attn-TP rank 0: one line with the cap setting,
    the KV pool and index-K classes, and whether the cap can project their
    backup (projected=1). capped=1: the cap is set and applies."""
    if not _is_cap_log_rank():
        return
    kvcache = token_to_kv_pool_allocator.get_kvcache()
    index_cache = getattr(kvcache, "index_key_cache", None)
    project = getattr(kvcache, "cpu_copy_nbytes", None)
    per_mi_tokens = None
    if backend == "cpu_tensor" and project is not None and _pp_size() == 1:
        per_mi_tokens = project(1 << 20)
    cap_gb = envs.SGLANG_RETRACT_BACKUP_MAX_GB.get()
    capped = per_mi_tokens is not None and cap_gb is not None and cap_gb >= 0
    logger.info(
        "RETRACT_BACKUP_CAP config backend=%s cap_gb=%s pageable=%d kv_pool=%s "
        "index_cache=%s projected=%d capped=%d gib_per_mi_tokens=%s",
        backend,
        "unset" if cap_gb is None else f"{cap_gb:g}",
        int(envs.SGLANG_RETRACT_BACKUP_PAGEABLE.get()),
        type(kvcache).__name__,
        "none" if index_cache is None else type(index_cache).__name__,
        int(per_mi_tokens is not None),
        int(capped),
        "na" if per_mi_tokens is None else f"{per_mi_tokens / _GIB:.3f}",
    )


def admit_retract_backup(
    req: Req, token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
) -> bool:
    """cpu_tensor retraction: whether the request's backup may be taken under
    SGLANG_RETRACT_BACKUP_MAX_GB. Admitted backups are recorded in
    RETRACT_BACKUP_LEDGER.

    Over the cap when the backups that this TP rank holds plus this one would
    take more than the cap. The inputs are the env, the request's token count,
    the KV pool shapes and the ledger. They are the same on every TP rank, so
    every rank takes the same decision without a collective."""
    cap_gb = envs.SGLANG_RETRACT_BACKUP_MAX_GB.get()
    if cap_gb is None or cap_gb < 0:
        return True
    nbytes = _projected_backup_nbytes(req, token_to_kv_pool_allocator)
    if nbytes is None:
        return True
    ledger = RETRACT_BACKUP_LEDGER
    ledger.nbytes_by_rid.pop(req.rid, None)  # a request holds one backup
    held = ledger.outstanding_nbytes()
    admitted = held + nbytes <= cap_gb * _GIB
    if admitted:
        ledger.nbytes_by_rid[req.rid] = nbytes
    else:
        ledger.num_cap_aborts += 1
    if _is_cap_log_rank():
        logger.log(
            logging.INFO if admitted else logging.WARNING,
            "RETRACT_BACKUP_CAP %s rid=%s tokens=%d projected_gb=%.3f "
            "outstanding_gb=%.3f cap_gb=%g",
            "hold" if admitted else "abort",
            req.rid,
            max(int(req.seqlen) - 1, 0),
            nbytes / _GIB,
            ledger.outstanding_nbytes() / _GIB,
            cap_gb,
        )
    return admitted


_MAX_FIT_TOKENS = 1 << 40
# retract_backup_token_limit: the KV pool that the cached limits are for, and
# the limits by (pageable, cap_gb, held bytes).
_FIT_LIMIT_CACHE: dict = {"kvcache": None, "limits": {}}


def retract_backup_token_limit(
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    held_nbytes: int,
    cap_gb: float,
) -> Optional[int]:
    """The most backup tokens (seqlen - 1) that admit_retract_backup holds next
    to ``held_nbytes`` of held backups: held + projection(n) <= the cap. -1
    when nothing fits; None when the backup has no projection that is the same
    on every rank (the cap does not apply then).

    The projection does not get smaller when n grows, so a search finds the
    limit. The limit is cached per KV pool, copy mode, cap and held bytes;
    held bytes change only at retraction events."""
    kvcache = token_to_kv_pool_allocator.get_kvcache()
    project = getattr(kvcache, "cpu_copy_nbytes", None)
    if project is None or _pp_size() != 1:
        return None
    cache = _FIT_LIMIT_CACHE
    if cache["kvcache"] is not kvcache:
        cache["kvcache"], cache["limits"] = kvcache, {}
    pageable = bool(envs.SGLANG_RETRACT_BACKUP_PAGEABLE.get())
    key = (pageable, float(cap_gb), int(held_nbytes))
    limits = cache["limits"]
    if key in limits:
        return limits[key]
    cap = cap_gb * _GIB

    def fits(num_tokens: int) -> bool:
        # The admit_retract_backup test: held + projected <= the cap.
        return held_nbytes + project(num_tokens) <= cap

    if project(0) is None:
        limit = None
    elif not fits(0):
        limit = -1
    else:
        lo, hi = 0, 1
        while hi <= _MAX_FIT_TOKENS and fits(hi):
            lo, hi = hi, hi * 2
        if hi > _MAX_FIT_TOKENS:
            limit = _MAX_FIT_TOKENS
        else:
            while hi - lo > 1:  # fits(lo) and not fits(hi)
                mid = (lo + hi) // 2
                if fits(mid):
                    lo = mid
                else:
                    hi = mid
            limit = lo
    if len(limits) >= 64:
        limits.clear()
    limits[key] = limit
    return limit


def retract_backup_would_fit(
    req: Req, token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
) -> bool:
    """Side-effect-free twin of admit_retract_backup: whether a cpu_tensor
    retraction of ``req`` now would hold its backup under
    SGLANG_RETRACT_BACKUP_MAX_GB. The same cap, projection and ledger; it
    changes no ledger entry or counter and logs nothing. True when no cap
    applies."""
    cap_gb = envs.SGLANG_RETRACT_BACKUP_MAX_GB.get()
    if cap_gb is None or cap_gb < 0:
        return True
    ledger = RETRACT_BACKUP_LEDGER
    held = ledger.outstanding_nbytes() - ledger.nbytes_by_rid.get(req.rid, 0)
    limit = retract_backup_token_limit(token_to_kv_pool_allocator, held, cap_gb)
    return limit is None or max(int(req.seqlen) - 1, 0) <= limit


def retract_backup_unsafe_rids(
    reqs: list[Req], token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
) -> frozenset[str]:
    """PD decode admission (SGLANG_DECODE_ADMIT_CAP_SAFE_RETRACT): the ids of
    the requests in ``reqs`` that a cpu_tensor retraction now would abort
    (retract_backup_would_fit is False). Empty when no cap applies. One limit
    search per ledger state, then one comparison per request."""
    cap_gb = envs.SGLANG_RETRACT_BACKUP_MAX_GB.get()
    if cap_gb is None or cap_gb < 0 or not reqs:
        return frozenset()
    ledger = RETRACT_BACKUP_LEDGER
    limit = retract_backup_token_limit(
        token_to_kv_pool_allocator, ledger.outstanding_nbytes(), cap_gb
    )
    if limit is None:
        return frozenset()
    unsafe = []
    for req in reqs:
        if req.rid in ledger.nbytes_by_rid:
            # It holds a backup already (admit_retract_backup replaces it).
            fits = retract_backup_would_fit(req, token_to_kv_pool_allocator)
        else:
            fits = max(int(req.seqlen) - 1, 0) <= limit
        if not fits:
            unsafe.append(req.rid)
    return frozenset(unsafe)


def release_retract_backup(req: Req, reason: str) -> None:
    """Drops the request from RETRACT_BACKUP_LEDGER: its backup was loaded
    back ("resume"), discarded ("discard"), or goes with the aborted request
    ("abort")."""
    nbytes = RETRACT_BACKUP_LEDGER.nbytes_by_rid.pop(req.rid, None)
    if nbytes is not None and _is_cap_log_rank():
        logger.info(
            "RETRACT_BACKUP_CAP release rid=%s reason=%s projected_gb=%.3f "
            "outstanding_gb=%.3f",
            req.rid,
            reason,
            nbytes / _GIB,
            RETRACT_BACKUP_LEDGER.outstanding_nbytes() / _GIB,
        )


def dsv4_needs_rebootstrap(
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
) -> bool:
    from decoupled_engine.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    pool = token_to_kv_pool_allocator.get_kvcache()
    # No DeepSeek-V4 pool keeps a host copy of a request's KV; the prefill recomputes it.
    return isinstance(pool, DeepSeekV4TokenToKVPool)


def backup_kv_cache(
    req: Req,
    tree_cache: BasePrefixCache,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    backend: str,
) -> bool:
    """Returns False when the request can be neither backed up nor recomputed;
    the caller aborts it. The backup cannot be kept when the host pool cannot
    hold it (host_pool), or the backups held on this TP rank would pass
    SGLANG_RETRACT_BACKUP_MAX_GB (cpu_tensor)."""
    if backend == "none":
        return False
    if dsv4_needs_rebootstrap(token_to_kv_pool_allocator):
        if req.multimodal_inputs is not None:
            # The recompute payload carries token ids only; images would be lost.
            return False
        # Drain the in-flight forward before its slots can receive recomputed KV.
        device = token_to_kv_pool_allocator.get_kvcache().device
        torch.get_device_module(device).synchronize(device)
        return True
    if backend == "cpu_tensor":
        if not admit_retract_backup(req, token_to_kv_pool_allocator):
            return False
        req.offload_kv_cache(req_to_token_pool, token_to_kv_pool_allocator)
        return True
    if backend != "host_pool":
        raise ValueError(f"Unknown retraction backup backend: {backend}")
    if req.seqlen <= 1:
        return True

    unified_cache = cast("UnifiedRadixCache", tree_cache)
    req.kv.retraction_backup = unified_cache.backup_kv_cache(req)
    return req.kv.retraction_backup is not None


def restore_kv_cache(
    req: Req,
    tree_cache: BasePrefixCache,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    backend: str,
) -> None:
    if backend == "cpu_tensor":
        req.load_kv_cache(req_to_token_pool, token_to_kv_pool_allocator)
        release_retract_backup(req, "resume")
        return
    if backend != "host_pool":
        raise ValueError(f"Unknown retraction backup backend: {backend}")
    if req.seqlen <= 1:
        return

    unified_cache = cast("UnifiedRadixCache", tree_cache)
    assert req.kv.retraction_backup is not None
    unified_cache.restore_kv_cache(req, req.kv.retraction_backup)
    req.kv.retraction_backup = None


def prepare_window_retraction(req: Req, tree_cache: BasePrefixCache) -> bool:
    """DSv4 window SWA residency, PD decode retract with a CPU snapshot.
    Call before the snapshot is taken.

    In-flight spills are re-attached first: their detached pages are not
    mapped, so the snapshot would skip them, and an orphaned op frees its host
    copy at the ack. Re-attached, they are device-resident again and land in
    the snapshot. The ledger is opened at the request's protected prefix
    (dpl) if needed, so the finish after the resume knows where tree
    ownership ends, and S = kv.swa_evicted_seqlen, the page-aligned end of
    the acked host coverage, is recorded as the SWA start of the resume
    allocation. Returns whether the caller should keep the ledger across the
    release (KEEP) once the snapshot exists."""
    if getattr(tree_cache, "dsv4_uniform_swa_residency", "full") != "window":
        return False
    if getattr(req, "skip_radix_cache_insert", False):
        # Never inserted: it frees out-of-window SWA the legacy way and has
        # no ledger; the resume uses the default tail.
        return False
    tree_cache.reattach_swa_spills(req)
    tree_cache.open_swa_ledger(req, req.kv.cache_protected_len)
    req.swa_resume_start = req.kv.get_evicted_seqlen(ComponentType.SWA)
    return True


def release_retracted_swa_ledger(req: Req, tree_cache: BasePrefixCache) -> None:
    """A retracted request is dropped instead of resumed (abort, memory
    release, capacity check): free the host SWA ledger its retract kept."""
    if getattr(req, "swa_resume_start", None) is not None:
        req.swa_resume_start = None
    release = getattr(tree_cache, "release_swa_ledger", None)
    if release is not None and getattr(req, "swa_ledger", None) is not None:
        release(req)


def discard_kv_cache_backup(
    req: Req, tree_cache: BasePrefixCache, backend: str
) -> None:
    release_retracted_swa_ledger(req, tree_cache)
    if backend == "cpu_tensor":
        req.kv.retraction_backup = None
        release_retract_backup(req, "discard")
        return
    if backend != "host_pool":
        raise ValueError(f"Unknown retraction backup backend: {backend}")
    if req.kv.retraction_backup is None:
        return

    unified_cache = cast("UnifiedRadixCache", tree_cache)
    unified_cache.discard_kv_cache_backup(req.kv.retraction_backup)
    req.kv.retraction_backup = None


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, *, checkpoint: bool):
    """Give the request's kv row back; with ``checkpoint`` the tree first keeps
    what it can key."""
    assert (not req.kv.holds_kv) == req.kv.is_kv_released
    # A mamba-capable cache may alloc mamba state before alloc KV cache
    if not req.kv.holds_kv:
        assert tree_cache.supports_mamba(), (
            "Only a mamba-capable tree cache allows freeing before alloc"
        )
        # TODO (csy, hanming): clean up this early allocation logic
        if req.kv.holds_mamba:
            tree_cache.req_to_token_pool.mamba_allocator.free(
                req.kv.mamba_pool_idx.unsqueeze(-1)
            )
            req.kv.mamba_pool_idx = None
        return
    if tree_cache.claim_kv_row(req):
        # A streaming session detached the kv record to keep the row.
        assert not req.kv.holds_kv
        return

    owned_kv_len = req.owned_kv_len()
    checkpoint = checkpoint and not req.skip_radix_cache_insert
    checkpoint_up_to = owned_kv_len
    checkpoint_kwargs = {}
    frozen_prompt_len = getattr(req, "decode_radix_cache_frozen_prompt_len", None)
    if frozen_prompt_len is not None:
        # EI PD decode radix content_only / window_host: the finish publishes
        # only the frozen prompt.
        from decoupled_engine.srt.managers.schedule_batch import FINISH_ABORT

        if not 0 <= frozen_prompt_len <= owned_kv_len:
            raise AssertionError(
                "Invalid frozen decode-radix prompt boundary: "
                f"rid={req.rid} frozen={frozen_prompt_len} "
                f"committed={owned_kv_len}"
            )
        # A prompt-only entry is published exactly once, and only after a
        # successful request finish.  Abort/cancel/transfer-failure paths must
        # release all request-owned pages without making them discoverable.
        checkpoint = (
            checkpoint
            and req.finished_reason is not None
            and not isinstance(req.finished_reason, FINISH_ABORT)
        )
        if checkpoint:
            # The tree page-floors this logical boundary; the free below still
            # releases everything the request owns past it.
            checkpoint_up_to = frozen_prompt_len
            checkpoint_kwargs = {
                # A kept SWA tail (symmetric import) is published with the
                # prompt as its checkpoint; otherwise the entry is FULL-only.
                "active_components": (
                    (ComponentType.FULL,)
                    if req.decode_radix_cache_frozen_swa_floor is None
                    else (ComponentType.FULL, ComponentType.SWA)
                ),
                # EI's pre-v0.5.23 name of the boundary (up_to now carries it).
                "max_cache_len": frozen_prompt_len,
            }
        logger.debug(
            "PD decode radix finish: rid=%s frozen_prompt_len=%d "
            "owned_kv_len=%d insert=%s finish_reason=%s",
            req.rid,
            frozen_prompt_len,
            owned_kv_len,
            checkpoint,
            type(req.finished_reason).__name__,
        )
    if checkpoint:
        # A tree that takes over component state (mamba) must see the request
        # finished, or the insert forks the state and the slot leaks.
        assert req.finished() or not tree_cache.supports_mamba(), (
            f"releasing unfinished request {req.rid} into a mamba tree"
        )
        # The fill-id array lags output_ids until the next prepare_for_decode.
        req.refresh_fill_ids()
        # symmetric P/D D-leg `cache_policy=gpu_only`: keep the imported prompt in the
        # radix tree but do not back it up to host (upstream PD decode behaviour).
        with _write_through_ctx(req, tree_cache):
            tree_cache.checkpoint(req, up_to=checkpoint_up_to, **checkpoint_kwargs)
    # The protected prefix is not this req's to free.
    tree_cache.free_kv_row(req.kv, [(req.kv.cache_protected_len, owned_kv_len)])
    tree_cache.unlock(req.lock)
    req.lock = None
    _release_overallocated_kv_indices(
        req, owned_kv_len, req.kv.kv_allocated_len, tree_cache
    )
    tree_cache.on_release(req, checkpointed=checkpoint)

    # The DSV4-NPU ReqToTokenPool subclass's free() additionally releases the
    # c4/c128 state pages; other ReqToTokenPool subclasses are a no-op here.
    tree_cache.req_to_token_pool.free(req)
    req.kv.mark_kv_released()


def _release_overallocated_kv_indices(
    req: Req, start_p: int, end_p: int, tree_cache: BasePrefixCache
) -> None:
    allocator = tree_cache.token_to_kv_pool_allocator
    page_size = allocator.page_size
    spec_algo = get_spec().speculative_algorithm

    # strip_thinking_cache intentionally reports output tokens as overallocated
    # so they fall into the free path below (#22373).
    if spec_algo is None and not get_serving().strip_thinking_cache:
        # A stop landing before the last committed token does the same, via
        # effective_kv_committed_len().
        assert start_p == end_p or (
            req.finished_len is not None
            and len(req.origin_input_ids) + req.finished_len < req.kv.kv_committed_len
        ), (
            f"Unexpected overallocated KV cache, {req.kv.kv_committed_len=}, {req.kv.kv_allocated_len=}"
        )

    # Align to the ALLOCATOR's page, which under DCP is wider than the kernel
    # page: paged free() releases the whole page containing any freed index, so
    # a boundary aligned only to the kernel page could free a widened page whose
    # head rows are still live.
    if page_size > 1:
        start_p = ceil_align(start_p, page_size)

    if start_p < end_p:
        # start_p is aligned to the allocator's page above, so it never shares a
        # page with the tail free_kv_row in this group. (Q4: free_kv_row tags
        # the slice with the CPU row's page ids; no-op when off.)
        tree_cache.free_kv_row(req.kv, [(start_p, end_p)])


def available_and_evictable_str(tree_cache: BasePrefixCache) -> str:
    return tree_cache.available_and_evictable_str()
