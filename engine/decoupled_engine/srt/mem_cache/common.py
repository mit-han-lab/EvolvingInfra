# Adapted from SGLang python/sglang/srt/mem_cache/common.py (commit 9a489f8d2f), modified for evolving-infra.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, cast

import numpy as np
import torch

from decoupled_engine.kernels.ops.memory.common import (
    _get_last_loc_safe_kernel as _get_last_loc_safe_kernel,
)
from decoupled_engine.kernels.ops.memory.common import get_last_loc_kernel as get_last_loc_kernel
from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.managers.scheduler_components.phase_timers import timed_phase
from decoupled_engine.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from decoupled_engine.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from decoupled_engine.srt.mem_cache.cpu_page_book import cpu_rows_of, tag_row_slice
from decoupled_engine.srt.mem_cache.hicache_storage import PoolTransfer
from decoupled_engine.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from decoupled_engine.srt.runtime_context import get_parallel, get_serving, get_spec
from decoupled_engine.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from decoupled_engine.srt.managers.schedule_batch import Req
    from decoupled_engine.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from decoupled_engine.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

# Needs 2 + 1 slots for mamba request with prefix cache. 2 for ping pong cache, 1 for running mamba state.
MAMBA_STATE_PER_REQ_PREFIX_CACHE = 3
# Lazy mode: 1 + 1 slots (1 ping-pong + 1 running), second ping-pong allocated on demand at boundary.
MAMBA_STATE_PER_REQ_PREFIX_CACHE_LAZY = 2
MAMBA_STATE_PER_REQ_NO_CACHE = 1

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
    is_chunk_cache: bool = False,
    retain_floor: int | None = None,
) -> None:
    if not req.is_holding_kv:
        return

    # For swa radix cache, we need to evict the tokens that are not in the tree cache and also not in the sliding window
    assert (
        req.cache_protected_len % page_size == 0
    ), "cache_protected_len must be page aligned"
    evict_floor = max(req.cache_protected_len, getattr(req, "swa_evict_floor", 0))
    if page_size > 1 and evict_floor > req.cache_protected_len:
        evict_floor = -(-evict_floor // page_size) * page_size
    req.kv.swa_evicted_seqlen = max(req.kv.swa_evicted_seqlen, evict_floor)

    if is_chunk_cache:
        # Chunk cache builds no radix tree, so no tombstone-leaf concern; evict
        # up to the window boundary (the trailing floor keeps it page-aligned).
        evict_threshold = pre_len - sliding_window_size
    else:
        # Radix cache: keep max(window, page). The trailing floor page-aligns the
        # frontier, and subtracting at least one page keeps it below the insert
        # boundary (page_floor(seq_len)) so the last leaf is never all-tombstone.
        # No extra page margin is needed.
        evict_threshold = pre_len - max(sliding_window_size, page_size)
    if retain_floor is not None and not is_chunk_cache:
        # The caller owns where the floor is (see BasePrefixCache.swa_retain_floor);
        # this only promises not to free past it. Chunk cache has no tree, so a
        # retained checkpoint could never be matched and holding it is pure cost.
        evict_threshold = min(evict_threshold, retain_floor)

    new_swa_evicted_seqlen = max(
        req.kv.swa_evicted_seqlen,
        evict_threshold,
    )

    if page_size > 1:
        new_swa_evicted_seqlen = (new_swa_evicted_seqlen // page_size) * page_size

    if new_swa_evicted_seqlen > req.kv.swa_evicted_seqlen:
        free_slots = req_to_token_pool.req_to_token[
            req.req_pool_idx, req.kv.swa_evicted_seqlen : new_swa_evicted_seqlen
        ]
        token_to_kv_pool_allocator.free_swa(free_slots)
        req.kv.swa_evicted_seqlen = new_swa_evicted_seqlen


def start_deferring_chunk_backups(tree_cache: BasePrefixCache) -> bool:
    """SGLANG_DSV4_CHUNK_WINDOW_WORK_AFTER_LAUNCH, for an overlap loop that
    calls tree_cache.run_deferred_chunk_backups() after every launch and in
    every iteration without one: True when the cache now queues its window
    chunk-boundary backups (UnifiedRadixCache in window residency). Other
    caches keep everything inline."""
    defer = getattr(tree_cache, "defer_chunk_backups_after_launch", None)
    return defer is not None and defer() is True


def maybe_cache_unfinished_req(req: Req, tree_cache: BasePrefixCache, **kwargs):
    if getattr(req, "decode_radix_cache_skip_unfinished_insert", False):
        # content_only freeze: unfinished inserts stay off for the request's
        # lifetime; the frozen prompt publishes once at successful finish.
        return
    if getattr(req, "skip_radix_cache_insert", False):
        return

    if getattr(req, "hicache_skip_write_through", False):
        # symmetric P/D D-leg `cache_policy=gpu_only` (import_queue.py): the imported
        # prompt is inserted (reader-side reuse) but not backed up to host.
        with tree_cache.suppress_write_through():
            tree_cache.cache_unfinished_req(req, **kwargs)
        return
    tree_cache.cache_unfinished_req(req, **kwargs)


def evict_from_tree_cache(tree_cache: BasePrefixCache | None, num_tokens: int):
    if tree_cache is None:
        return

    if tree_cache.is_chunk_cache():
        return

    allocator = tree_cache.token_to_kv_pool_allocator

    if isinstance(allocator, SWATokenToKVPoolAllocator):
        # Hybrid allocator
        full_available_size = allocator.full_available_size()
        swa_available_size = allocator.swa_available_size()

        if full_available_size < num_tokens or swa_available_size < num_tokens:
            full_num_tokens = max(0, num_tokens - full_available_size)
            swa_num_tokens = max(0, num_tokens - swa_available_size)
            tree_cache.evict_for_alloc(
                EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
            )
    else:
        # Standard allocator: evict only the shortfall (mirrors the SWA arm)
        available_size = allocator.available_size()
        if available_size < num_tokens:
            tree_cache.evict_for_alloc(
                EvictParams(num_tokens=num_tokens - available_size)
            )


_GIB = 1 << 30


class RetractBackupLedger:
    """Host bytes of the cpu_tensor retraction backups that this TP rank holds,
    by request id, while SGLANG_RETRACT_BACKUP_MAX_GB is set.

    Only scheduler events change it, and every TP rank runs them in the same
    order: the backup (retraction_backup), the resume (retraction_restore),
    the discard of a retracted request (retraction_discard) and the abort of a
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


def retraction_backup(
    req: Req,
    tree_cache: BasePrefixCache,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    backend: str,
) -> bool:
    """Returns False when the backup cannot be kept: the host pool cannot hold
    it (host_pool), or the backups held on this TP rank would pass
    SGLANG_RETRACT_BACKUP_MAX_GB (cpu_tensor). The caller aborts the request
    since its KV cannot be preserved."""
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
    req.retraction_backup = unified_cache.retraction_backup(req)
    return req.retraction_backup is not None


def retraction_restore(
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
    assert req.retraction_backup is not None
    unified_cache.retraction_restore(req, req.retraction_backup)
    req.retraction_backup = None


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
    tree_cache.open_swa_ledger(req, req.cache_protected_len)
    req.swa_resume_start = req.kv.swa_evicted_seqlen
    return True


def release_retracted_swa_ledger(req: Req, tree_cache: BasePrefixCache) -> None:
    """A retracted request is dropped instead of resumed (abort, memory
    release, capacity check): free the host SWA ledger its retract kept."""
    if getattr(req, "swa_resume_start", None) is not None:
        req.swa_resume_start = None
    release = getattr(tree_cache, "release_swa_ledger", None)
    if release is not None and getattr(req, "swa_ledger", None) is not None:
        release(req)


def retraction_discard(req: Req, tree_cache: BasePrefixCache, backend: str) -> None:
    release_retracted_swa_ledger(req, tree_cache)
    if backend == "cpu_tensor":
        req.retraction_backup = None
        release_retract_backup(req, "discard")
        return
    if backend != "host_pool":
        raise ValueError(f"Unknown retraction backup backend: {backend}")
    if req.retraction_backup is None:
        return

    unified_cache = cast("UnifiedRadixCache", tree_cache)
    unified_cache.retraction_discard(req.retraction_backup)
    req.retraction_backup = None


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    assert (not req.is_holding_kv) == req.kv.is_released
    # MambaRadixCache may alloc mamba state before alloc KV cache
    if not req.is_holding_kv:
        assert (
            tree_cache.supports_mamba()
        ), "Only MambaRadixCache allow freeing before alloc"
        # TODO (csy, hanming): clean up this early allocation logic
        if req.mamba_pool_idx is not None:
            tree_cache.req_to_token_pool.mamba_allocator.free(
                req.mamba_pool_idx.unsqueeze(-1)
            )
            req.mamba_pool_idx = None
        return

    effective_kv_committed_len = req.effective_kv_committed_len()
    frozen_prompt_len = getattr(req, "decode_radix_cache_frozen_prompt_len", None)
    cache_kwargs = {}
    if frozen_prompt_len is not None:
        from decoupled_engine.srt.managers.schedule_batch import FINISH_ABORT
        from decoupled_engine.srt.mem_cache.unified_cache.components import ComponentType

        if not 0 <= frozen_prompt_len <= effective_kv_committed_len:
            raise AssertionError(
                "Invalid frozen decode-radix prompt boundary: "
                f"rid={req.rid} frozen={frozen_prompt_len} "
                f"committed={effective_kv_committed_len}"
            )
        # A prompt-only entry is published exactly once, and only after a
        # successful request finish.  Abort/cancel/transfer-failure paths must
        # release all request-owned pages without making them discoverable.
        is_insert = (
            is_insert
            and not getattr(req, "skip_radix_cache_insert", False)
            and req.finished_reason is not None
            and not isinstance(req.finished_reason, FINISH_ABORT)
        )
        if is_insert:
            cache_kwargs = {
                # A kept SWA tail (symmetric import) is published with the
                # prompt as its checkpoint; otherwise the entry is FULL-only.
                "active_components": (
                    (ComponentType.FULL,)
                    if req.decode_radix_cache_frozen_swa_floor is None
                    else (ComponentType.FULL, ComponentType.SWA)
                ),
                # The cache implementation page-floors this logical boundary;
                # kv_len_to_handle remains the independent release boundary.
                "max_cache_len": frozen_prompt_len,
            }
        logger.debug(
            "PD decode radix finish: rid=%s frozen_prompt_len=%d "
            "effective_committed_len=%d insert=%s finish_reason=%s",
            req.rid,
            frozen_prompt_len,
            effective_kv_committed_len,
            is_insert,
            type(req.finished_reason).__name__,
        )

    # symmetric P/D D-leg `cache_policy=gpu_only`: keep the imported prompt in the
    # radix tree but do not back it up to host (upstream PD decode behaviour).
    import contextlib

    wt_ctx = (
        tree_cache.suppress_write_through()
        if getattr(req, "hicache_skip_write_through", False)
        else contextlib.nullcontext()
    )
    with wt_ctx:
        tree_cache.cache_finished_req(
            req,
            is_insert=is_insert and not getattr(req, "skip_radix_cache_insert", False),
            kv_len_to_handle=effective_kv_committed_len,
            **cache_kwargs,
        )

    # StreamingSession.cache_finished_req handles speculative tail trim
    # internally, then sets req_pool_idx = None.
    assert (not req.is_holding_kv) == req.kv.is_released
    if not req.is_holding_kv:
        return

    start_p, end_p = effective_kv_committed_len, req.kv.kv_allocated_len
    _release_overallocated_kv_indices(req, start_p, end_p, tree_cache)

    # If the prefix cache doesn't manage mamba states, we must free them here.
    if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
        not tree_cache.supports_mamba()
    ):
        assert (
            req.mamba_pool_idx is not None
        ), "mamba state is freed while the tree cache does not manage mamba states"
        tree_cache.req_to_token_pool.free_mamba_cache(req)
    # The DSV4-NPU ReqToTokenPool subclass's free() additionally releases the
    # c4/c128 state pages; other ReqToTokenPool subclasses are a no-op here.
    tree_cache.req_to_token_pool.free(req)
    req.kv.mark_released()


def _release_overallocated_kv_indices(
    req: Req, start_p: int, end_p: int, tree_cache: BasePrefixCache
) -> None:
    allocator = tree_cache.token_to_kv_pool_allocator
    page_size = allocator.page_size
    spec_algo = get_spec().speculative_algorithm

    # strip_thinking_cache intentionally reports output tokens as overallocated
    # so they fall into the free path below (#22373).
    if spec_algo is None and not get_serving().strip_thinking_cache:
        assert (
            start_p == end_p
        ), f"Unexpected overallocated KV cache, {req.kv_committed_len=}, {req.kv.kv_allocated_len=}"

    if page_size > 1:
        start_p = ceil_align(start_p, page_size)

    if start_p < end_p:
        indices_to_free = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][
            start_p:end_p
        ]
        # Q4: the page ids from the CPU row (no-op when off).
        tag_row_slice(
            indices_to_free,
            cpu_rows_of(tree_cache.req_to_token_pool),
            req.req_pool_idx,
            start_p,
        )
        # start_p is aligned to the allocator's physical page size above, so it
        # never shares a page with cache_finished_req's tail free in this group.
        allocator.free_segment(indices_to_free, start_pos=start_p)


def available_and_evictable_str(tree_cache: BasePrefixCache) -> str:
    return tree_cache.available_and_evictable_str()
