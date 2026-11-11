"""CPU-side page bookkeeping on the PD prefill worker (called "Q4" in comments
and messages; ``SGLANG_PREFILL_CPU_PAGE_BOOKKEEPING``).

Without it the prefill scheduler learns page ids from the GPU. The KV send
reads ``req_to_token`` rows (``kv_to_page_indices(...).cpu()``), the
copy-engine loader moves load destinations to the host
(``device_indices.cpu()``), write-through moves its source indices the same
way, and the allocator's free lists live on the device. Each read waits for
whatever the scheduler stream waits for, which in the overlap loop is the
in-flight forward. Here the CPU is authoritative for every page id of the full
KV pool:

- Allocator (``allocator/cpu_paged.py``): the free lists are CPU tensors.
  Allocations compute page ids on the CPU and copy the token slots to the
  device (pinned, non-blocking, on the scheduler stream).
- Page tags: every device token-index tensor that the allocator or the radix
  tree hands out carries its CPU page ids (``tag`` / ``pages_of``). A tensor
  derived by slicing or concatenation is re-tagged where it is derived
  (``slice_tagged``, ``cat_tagged``, ``copy_tag``). Frees, load and
  write-through transfers and admission read the tag; a missing tag raises
  ``MissingPageIds``. There is no device-read fallback.
- Request rows (``CpuPageRows``): a CPU page row per ``req_to_token`` row,
  written next to every prefill write of the device row. KV sends read them
  (``CpuRowPageReader``).
- Shadow check (debug, ``SGLANG_LOAD_QUARANTINE_SHADOW_CHECK_EVERY``): rows and
  tree values against the device, and no page both in use and free.

Tag convention: ``pages`` lists the pages a tensor touches, in position order.
For tensors that start on a page boundary (allocations, tree values, prefix
indices) ``pages[k]`` is the page of positions ``[k * ps, (k + 1) * ps)``;
``slice_tagged`` / ``cat_tagged`` require such starts. Only free paths tag
slices that start inside a page (``CpuPageRows.tag_slice``).

Scope: ``supported_reason``. Flag ``SGLANG_PREFILL_CPU_PAGE_BOOKKEEPING``
(default off; off leaves every path unchanged: nothing is tagged and the
helpers reduce to plain slicing / ``torch.cat``). The M4 load quarantine
(``load_quarantine.py``) requires it.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any, Callable, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch

from decoupled_engine.srt.environ import envs

logger = logging.getLogger(__name__)

STATS: Counter = Counter()

PAGES_ATTR = "_q4_pages"
_EMPTY = torch.empty((0,), dtype=torch.int64)

# Exact log prefixes (GPU test stop rules grep for them).
SHADOW_FAIL_PREFIX = "Q4 shadow check FAILED"
MISSING_IDS_PREFIX = "Q4 page bookkeeping: device indices without CPU page ids"


def cpu_page_bookkeeping_enabled() -> bool:
    return bool(envs.SGLANG_PREFILL_CPU_PAGE_BOOKKEEPING.get())


def m4_flags_set() -> bool:
    """SGLANG_PREFILL_CPU_PAGE_BOOKKEEPING or SGLANG_DSA_HOST_DEDUP_LOAD_QUARANTINE
    (the latter read directly: load_quarantine imports this module)."""
    return cpu_page_bookkeeping_enabled() or bool(
        envs.SGLANG_DSA_HOST_DEDUP_LOAD_QUARANTINE.get()
    )


# Exact prefix of the inactive-role line (one INFO line per worker).
INACTIVE_PREFIX = "M4 inactive on"


def log_inactive_role(role: str, tp_rank: int = 0) -> bool:
    """M4 runs in the PD prefill overlap loop and the symmetric overlap loop
    (``init_cpu_page_book``). On another role (decode, null) its flags are
    inert, so P and D can share one set of args: say so once (TP0, INFO),
    never raise. True if an M4 flag is set."""
    if not m4_flags_set():
        return False
    if tp_rank == 0:
        logger.info(
            "%s %s: SGLANG_PREFILL_CPU_PAGE_BOOKKEEPING / "
            "SGLANG_DSA_HOST_DEDUP_LOAD_QUARANTINE apply to the PD prefill and "
            "symmetric overlap loops only",
            INACTIVE_PREFIX,
            role,
        )
    return True


class MissingPageIds(AssertionError):
    """A device index tensor reached a Q4 consumer without its CPU page ids."""


# ---------------------------------------------------------------------------
# Page tags
# ---------------------------------------------------------------------------


def tag(t: torch.Tensor, pages: torch.Tensor) -> torch.Tensor:
    setattr(t, PAGES_ATTR, pages)
    return t


def has_pages(t: Any) -> bool:
    return getattr(t, PAGES_ATTR, None) is not None


def pages_of(t: Optional[torch.Tensor], where: str) -> torch.Tensor:
    """CPU page ids of ``t``; empty for an empty tensor; raises if untagged."""
    p = getattr(t, PAGES_ATTR, None)
    if p is not None:
        return p
    if t is None or t.numel() == 0:
        return _EMPTY
    STATS["missing_page_ids"] += 1
    raise MissingPageIds(f"{MISSING_IDS_PREFIX} at {where} (len={t.numel()})")


def _num_pages(start: int, n: int, page_size: int) -> int:
    return (start % page_size + n + page_size - 1) // page_size if n else 0


def slice_tagged(
    t: torch.Tensor, start: int, end: Optional[int], page_size: int
) -> torch.Tensor:
    """``t[start:end]``, tagged when ``t`` is (``t`` starts on a page boundary;
    so must ``start``)."""
    out = t[start:end]
    p = getattr(t, PAGES_ATTR, None)
    if p is None:
        return out
    n = len(out)
    if n == 0:
        return tag(out, _EMPTY)
    if start % page_size:
        raise AssertionError(
            f"Q4 tagged slice must start on a page boundary: {start=} {page_size=}"
        )
    k = start // page_size
    return tag(out, p[k : k + _num_pages(0, n, page_size)])


def copy_tag(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """``dst`` holds the same indices as ``src`` (a copy or a dtype change)."""
    p = getattr(src, PAGES_ATTR, None)
    if p is not None:
        tag(dst, p)
    return dst


def cat_tagged(
    parts: Sequence[torch.Tensor],
    page_size: Union[int, Callable[[], int]],
    where: str = "cat",
) -> torch.Tensor:
    """``torch.cat(parts)``, tagged when any part is (then every non-empty
    part must be, and must start on a page boundary of the result).

    ``page_size`` may be a zero-argument callable, called only when a part is
    tagged (Q4 on): with Q4 off the call reads no attribute the plain
    ``torch.cat`` it replaces did not (stand-in trees in tests have none)."""
    out = torch.cat(list(parts))
    if not any(has_pages(x) for x in parts):
        return out
    if callable(page_size):
        page_size = page_size()
    pages: List[torch.Tensor] = []
    pos = 0
    for x in parts:
        n = len(x)
        if n == 0:
            continue
        if pos % page_size:
            raise AssertionError(
                f"Q4 tagged cat at {where}: a part starts at {pos}, not on a page boundary"
            )
        pages.append(pages_of(x, where))
        pos += n
    return tag(out, torch.cat(pages) if pages else _EMPTY)


def expand_pages(pages: torch.Tensor, page_size: int) -> torch.Tensor:
    """Token slots of whole pages, page-major (CPU int64)."""
    return (
        pages[:, None] * page_size + torch.arange(page_size, dtype=torch.int64)
    ).reshape(-1)


def tokens_of(t: torch.Tensor, page_size: int, where: str) -> torch.Tensor:
    """The CPU token indices equal to ``t``'s device values (``t`` starts on a
    page boundary): no device read."""
    n = len(t)
    if n == 0:
        return torch.empty((0,), dtype=torch.int64)
    return expand_pages(pages_of(t, where), page_size)[:n]


def tag_row_slice(t: torch.Tensor, rows: Optional[CpuPageRows], row: int, start: int):
    """Tag ``t`` = device ``req_to_token[row, start:start + len(t)]`` from the
    CPU row (no-op when Q4 is off, i.e. ``rows`` is None)."""
    if rows is not None:
        rows.tag_slice(t, row, start)
    return t


# ---------------------------------------------------------------------------
# CPU request rows
# ---------------------------------------------------------------------------


class CpuPageRows:
    """Page ids of every ``req_to_token`` row (position ``p`` of row ``r`` is
    token slot ``rows[r, p // ps] * ps + p % ps``).

    Written next to every prefill write of a device row: admission
    (``alloc_for_extend``), the ``cache_unfinished_req`` rewrite
    (``ReqToTokenPool.write``) and slot release. ``lens[r]`` is the number of
    positions written since the slot was taken (shadow check, ownership)."""

    def __init__(self, num_rows: int, max_context_len: int, page_size: int):
        self.page_size = page_size
        self.max_pages = (max_context_len + page_size - 1) // page_size
        self.rows = torch.zeros((num_rows, self.max_pages), dtype=torch.int64)
        self.lens = np.zeros(num_rows, dtype=np.int64)

    def write(self, row: int, start: int, end: int, pages: torch.Tensor) -> None:
        """Mirror a device write of positions ``[start, end)`` whose token
        slots lie on ``pages`` (the pages touched, in order)."""
        if end <= start:
            return
        ps = self.page_size
        p0 = start // ps
        p1 = (end + ps - 1) // ps
        if len(pages) != p1 - p0:
            raise AssertionError(
                f"Q4 row write: {len(pages)} pages for positions [{start}, {end}) "
                f"of row {row} (page size {ps})"
            )
        if start % ps and self.lens[row] > p0 * ps:
            # The device keeps [p0 * ps, start) of this page: same page.
            if int(self.rows[row, p0]) != int(pages[0]):
                raise AssertionError(
                    f"Q4 row write splits page {p0} of row {row}: "
                    f"{int(self.rows[row, p0])} != {int(pages[0])}"
                )
        self.rows[row, p0:p1] = pages
        if end > self.lens[row]:
            self.lens[row] = end

    def pages(self, row: int, start: int, end: int) -> torch.Tensor:
        """Pages touched by positions ``[start, end)`` (a copy)."""
        if end <= start:
            return _EMPTY
        ps = self.page_size
        return self.rows[row, start // ps : (end + ps - 1) // ps].clone()

    def page_ids(self, row: int, start: int, end: int) -> np.ndarray:
        """``kv_to_page_indices(req_to_token[row, start:end], ps)`` without a
        device read: the pages of positions ``start, start + ps, ...`` (int32,
        a copy, like the device read)."""
        ps = self.page_size
        if end <= start:
            return np.empty((0,), dtype=np.int32)
        k0 = start // ps
        n = (end - start + ps - 1) // ps
        return self.rows[row, k0 : k0 + n].numpy().astype(np.int32)

    def tokens(self, row: int, start: int, end: int) -> torch.Tensor:
        """CPU token slots of positions ``[start, end)``."""
        if end <= start:
            return torch.empty((0,), dtype=torch.int64)
        ps = self.page_size
        off = start % ps
        return expand_pages(self.pages(row, start, end), ps)[off : off + end - start]

    def tag_slice(self, t: torch.Tensor, row: int, start: int) -> torch.Tensor:
        return tag(t, self.pages(row, start, start + len(t)))

    def release(self, rows: Iterable[int]) -> None:
        for r in rows:
            self.lens[int(r)] = 0

    def clear(self) -> None:
        self.lens[:] = 0

    def live_rows(self) -> np.ndarray:
        return np.nonzero(self.lens)[0]

    def write_positions(
        self,
        rows: Sequence[int],
        positions: Sequence[int],
        pages: Sequence[int],
    ) -> None:
        """Mirror a decode-step write of the device rows, vectorised (no
        per-entry Python work): position ``positions[i]`` of row ``rows[i]``
        lies on page ``pages[i]``. One entry per request (decode), or one per
        page a run touches, at the run's last position on that page (spec
        decode, ``run_pages``). ``lens[r]`` grows to the largest position + 1
        of its entries. A position inside a page the row already holds must
        keep that page (as in ``write``)."""
        r = int64_array(rows)
        if r.size == 0:
            return
        pos = int64_array(positions)
        pg = int64_array(pages)
        ps = self.page_size
        k = pos // ps
        table = self.rows.numpy()
        # The device keeps [k * ps, pos) of a partial page: the same page.
        keep = (pos % ps != 0) & (self.lens[r] > k * ps)
        if keep.any():
            bad = np.flatnonzero(keep & (table[r, k] != pg))
            if bad.size:
                i = int(bad[0])
                raise AssertionError(
                    f"Q4 row write splits page {int(k[i])} of row {int(r[i])}: "
                    f"{int(table[r[i], k[i]])} != {int(pg[i])}"
                )
        table[r, k] = pg
        np.maximum.at(self.lens, r, pos + 1)


class CpuRowPageReader:
    """KV page ids for sends from the CPU rows (same interface as M2's
    ``KvPageIdReader``): no device read, nothing to wait for."""

    def __init__(self, rows: CpuPageRows):
        self.rows = rows

    def begin_iteration(self) -> None:
        pass

    def mark_pre_barrier(self, schedule_stream) -> None:
        pass

    def page_ids(self, req_pool_idx: int, start: int, end: int) -> np.ndarray:
        STATS["send_ids_cpu"] += 1
        return self.rows.page_ids(int(req_pool_idx), start, end)


def cpu_rows_of(req_to_token_pool: Any) -> Optional[CpuPageRows]:
    """The pool's CPU page rows, or None when Q4 is off (strict type check:
    mocks and other pools never count)."""
    rows = getattr(req_to_token_pool, "cpu_rows", None)
    return rows if isinstance(rows, CpuPageRows) else None


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def extend_page_plan(
    prefix_lens: Sequence[int], seq_lens: Sequence[int], page_size: int
) -> List[int]:
    """New pages per request of an extend batch (``get_num_new_pages`` per
    request): ``ceil(seq / ps) - ceil(prefix / ps)``."""
    ps = page_size
    return [
        (s + ps - 1) // ps - (p + ps - 1) // ps for p, s in zip(prefix_lens, seq_lens)
    ]


def extend_token_slots(
    prefix_lens: Sequence[int],
    seq_lens: Sequence[int],
    last_pages: Sequence[int],
    new_pages: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """CPU token slots of an extend batch, in ``alloc_extend_kernel``'s
    layout: per request, positions ``[prefix, seq)``; the rest of a partial
    last prefix page (``last_pages[i]``), then ``new_pages`` in batch order."""
    ps = page_size
    parts = []
    off = 0
    arange = torch.arange(ps, dtype=torch.int64)
    for pre, seq, last in zip(prefix_lens, seq_lens, last_pages):
        if seq <= pre:
            continue
        first_new = (pre + ps - 1) // ps * ps
        num1 = min(seq, first_new) - pre
        if num1:
            parts.append(int(last) * ps + (pre % ps) + arange[:num1])
        n_new = (seq + ps - 1) // ps - (pre + ps - 1) // ps
        if n_new:
            pages = new_pages[off : off + n_new]
            off += n_new
            parts.append(expand_pages(pages, ps)[: seq - first_new])
    if not parts:
        return torch.empty((0,), dtype=torch.int64)
    return torch.cat(parts)


# ---------------------------------------------------------------------------
# Scope and orchestration
# ---------------------------------------------------------------------------


def _symmetric_reason(scheduler: Any) -> Optional[str]:
    """The symmetric P/D worker's extra scope: the aggregated overlap
    loop's legs that keep CPU page ids."""
    from decoupled_engine.srt.speculative.spec_info import SpeculativeAlgorithm

    if not getattr(scheduler, "enable_overlap", False):
        return "symmetric mode without the overlap loop"
    if not envs.SGLANG_DSA_HOST_DEDUP_LOAD_QUARANTINE.get():
        # Imports land by RDMA, outside every stream: their pages must be
        # clean (released at a completed epoch), which only the quarantine knows.
        return "symmetric mode without the load quarantine (SGLANG_DSA_HOST_DEDUP_LOAD_QUARANTINE)"
    if getattr(scheduler, "is_mixed_chunk", False):
        return "mixed chunked prefill (decode tokens in an extend batch)"
    if getattr(getattr(scheduler, "ps", None), "pp_size", 1) > 1:
        return "pipeline parallelism"
    algo = getattr(scheduler, "spec_algorithm", None)
    if algo is not None and not algo.is_none():
        if algo not in (SpeculativeAlgorithm.EAGLE, SpeculativeAlgorithm.EAGLE3):
            return f"speculative algorithm {algo.name} (EAGLE / EAGLE3 v2 only)"
        server_args = getattr(scheduler, "server_args", None)
        topk = getattr(server_args, "speculative_eagle_topk", 1) or 1
        if topk != 1:
            return f"EAGLE topk {topk} (topk 1 only)"
        if getattr(server_args, "enable_multi_layer_eagle", False):
            return "multi-layer EAGLE"
    return None


def supported_reason(scheduler: Any) -> Optional[str]:
    """None if Q4 can run in this scheduler (the PD prefill overlap loop or a
    symmetric worker's aggregated overlap loop), else why not. Out of scope
    means an error at startup (there is no fallback path)."""
    from decoupled_engine.srt.disaggregation.utils import DisaggregationMode
    from decoupled_engine.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    from decoupled_engine.srt.mem_cache.memory_pool import DSATokenToKVPool, ReqToTokenPool
    from decoupled_engine.srt.mem_cache.unified_cache.component_type import ComponentType
    from decoupled_engine.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

    symmetric = (
        getattr(scheduler, "disaggregation_mode", None) == DisaggregationMode.SYMMETRIC
    )
    allocator = scheduler.token_to_kv_pool_allocator
    if type(allocator) is not PagedTokenToKVPoolAllocator:
        return f"allocator {type(allocator).__name__} is not the plain paged pool"
    if not allocator.need_sort:
        return "allocator without need_sort (not a PD worker)"
    if allocator.page_size <= 1:
        return "page size 1"
    if getattr(allocator, "track_page_generations", False) and not symmetric:
        # Only symmetric mode keeps generations (on the CPU, cpu_paged._gen).
        return "page generation tracking outside symmetric mode"
    if symmetric:
        reason = _symmetric_reason(scheduler)
        if reason is not None:
            return reason
    # GLM DSA prefill only (the validated target): KV, indexer and draft
    # pools share this allocator's page ids.
    kv_pool = allocator.get_kvcache()
    if not isinstance(kv_pool, DSATokenToKVPool):
        return (
            f"KV pool {type(kv_pool).__name__} is not a DSA pool (GLM DSA prefill only)"
        )
    if getattr(scheduler, "enable_unified_memory", False):
        return "unified memory"
    if getattr(scheduler, "enable_staging", False):
        return "prefill staging"
    pool = scheduler.req_to_token_pool
    plain_pools = (ReqToTokenPool,)
    if symmetric:
        from decoupled_engine.srt.disaggregation.decode import DecodeReqToTokenPool

        # Symmetric workers build the decode pool (extra rows for imports that
        # wait outside the batch); it carries the same CPU-row hooks.
        plain_pools = (ReqToTokenPool, DecodeReqToTokenPool)
    if type(pool) not in plain_pools:
        return f"request pool {type(pool).__name__} is not the plain pool"
    if getattr(pool, "_aux_cache", None) is not None:
        return "request pool with an aux cache"
    tree = scheduler.tree_cache
    if not isinstance(getattr(tree, "tree_core", None), UnifiedTreeCore):
        return f"tree cache {type(tree).__name__} has no python unified tree core"
    if tuple(getattr(tree, "tree_components", ())) != (ComponentType.FULL,):
        return f"tree components {getattr(tree, 'tree_components', None)} != (FULL,)"
    if getattr(tree, "disable", False):
        return "radix cache disabled"
    if getattr(tree, "enable_session_radix_cache", False):
        return "session radix cache"
    if getattr(tree, "buffer_pipeline", None) is not None:
        return "HiCache buffer mode"
    # kv-p2p: the peer-KV servicer and the peer_kv
    # reader address host pages only (host_value slices, host-pool leases,
    # RDMA into the host pool); pulled pages reach the device through the
    # tagged load-back. Only the GPU export leases device pages. Decided from
    # the server args, the same on every rank: under host dedup only the
    # source rank has a servicer.
    server_args = getattr(scheduler, "server_args", None)
    if getattr(server_args, "enable_peer_kv_gpu_export", False) or (
        getattr(tree, "gpu_export_manager", None) is not None
    ):
        return "peer-KV GPU export (device page leases)"
    cc = getattr(tree, "cache_controller", None)
    if cc is not None:
        if getattr(cc, "io_backend", None) != "direct":
            return (
                f"HiCache IO backend {getattr(cc, 'io_backend', None)} (needs direct)"
            )
        layout = getattr(getattr(cc, "mem_pool_host", None), "layout", None)
        if layout != "page_first_direct":
            return f"HiCache layout {layout} (needs page_first_direct)"
        if getattr(cc, "write_policy", None) == "write_back":
            return "HiCache write_back policy"
    is_dllm = getattr(scheduler, "dllm_config", None) is not None
    if is_dllm:
        return "dLLM"
    from decoupled_engine.srt.utils import is_npu

    if is_npu():
        return "NPU"
    return None


class PrefillCpuPageBook:
    """Owned by the PD prefill scheduler: converts the allocator, attaches the
    CPU rows, drives the load quarantine (if any) at the iteration start and
    runs the shadow check."""

    def __init__(
        self,
        allocator: Any,
        req_to_token_pool: Any,
        tree_cache: Any,
        clock: Any = None,
        shadow_check_every: int = 0,
        tp_rank: int = 0,
        log_period_s: float = 60.0,
    ) -> None:
        from decoupled_engine.srt.mem_cache.allocator.cpu_paged import convert_to_cpu_paged

        self.allocator = convert_to_cpu_paged(allocator, clock)
        self.page_size = self.allocator.page_size
        self.pool = req_to_token_pool
        self.tree_cache = tree_cache
        num_rows, max_context_len = req_to_token_pool.req_to_token.shape
        self.rows = CpuPageRows(num_rows, max_context_len, self.page_size)
        req_to_token_pool.cpu_rows = self.rows
        self.reader = CpuRowPageReader(self.rows)
        self.quarantine = None  # LoadQuarantine, set by the scheduler
        self.shadow_check_every = shadow_check_every
        self.tp_rank = tp_rank
        self.log_period_s = log_period_s
        self._last_log = time.monotonic()
        self._iterations = 0

    def begin_iteration(self) -> None:
        """Lockstep point at the top of the overlap loop."""
        if self.quarantine is not None:
            self.quarantine.begin_iteration()
        self._iterations += 1
        if self.shadow_check_every and self._iterations % self.shadow_check_every == 0:
            self.shadow_check()
        now = time.monotonic()
        if now - self._last_log >= self.log_period_s:
            self._last_log = now
            if self.tp_rank == 0 and STATS:
                logger.info(
                    "M4 Q4 page-book stats: free_pages=%d live_rows=%d %s%s",
                    self.allocator.available_size() // self.page_size,
                    len(self.rows.live_rows()),
                    (
                        self.quarantine.stats_fields() + " "
                        if self.quarantine is not None
                        else ""
                    ),
                    " ".join(f"{k}={v}" for k, v in sorted(STATS.items())),
                )

    # -- shadow check (debug; blocking device reads) --

    def shadow_check(self) -> int:
        t0 = time.perf_counter()
        problems = check_page_book(
            self.rows,
            self.pool.req_to_token,
            self.tree_cache,
            self.allocator,
        )
        STATS["shadow_checks"] += 1
        STATS["shadow_ms_x1000"] += int((time.perf_counter() - t0) * 1e6)
        if problems:
            STATS["shadow_mismatch"] += len(problems)
            for p in problems[:16]:
                logger.error("%s: %s", SHADOW_FAIL_PREFIX, p)
            # A divergence would send or free the wrong pages without a crash.
            raise RuntimeError(
                f"{SHADOW_FAIL_PREFIX}: {len(problems)} problems (first: {problems[0]})"
            )
        return 0


def _tree_full_values(tree_cache: Any) -> List[torch.Tensor]:
    from decoupled_engine.srt.mem_cache.unified_cache.component_type import ComponentType

    core = tree_cache.tree_core
    out = []
    stack = [core.root_node]
    while stack:
        node = stack.pop()
        stack.extend(node.children.values())
        if node is core.root_node:
            continue
        value = node.component_data[ComponentType.FULL].value
        if value is not None and len(value) > 0:
            out.append(value)
    return out


def check_page_book(
    rows: CpuPageRows, req_to_token: torch.Tensor, tree_cache: Any, allocator: Any
) -> List[str]:
    """Debug: (1) CPU rows == device rows over the written positions; (2)
    every tree value's tag == its device page ids; (3) no page is both in use
    (a live row or a tree value) and in a free list; (4) no duplicate free
    page. Returns problem descriptions."""
    problems: List[str] = []
    ps = rows.page_size

    live = rows.live_rows()
    used = set()
    if len(live):
        max_len = int(rows.lens[live].max())
        rows_idx = torch.as_tensor(live, dtype=torch.int64, device=req_to_token.device)
        dev = req_to_token[rows_idx, :max_len]
        dev = dev.cpu().to(torch.int64)
        for i, r in enumerate(live.tolist()):
            n = int(rows.lens[r])
            cpu = rows.tokens(r, 0, n)
            if not torch.equal(dev[i, :n], cpu):
                bad = int((dev[i, :n] != cpu).nonzero()[0])
                problems.append(
                    f"row {r} position {bad}: device {int(dev[i, bad])} cpu {int(cpu[bad])} (len {n})"
                )
            used.update(rows.pages(r, 0, n).tolist())

    values = _tree_full_values(tree_cache)
    tagged = []
    for v in values:
        p = getattr(v, PAGES_ATTR, None)
        if p is None:
            problems.append(f"tree value of {len(v)} tokens without page ids")
        else:
            tagged.append((v, p))
    if tagged:
        # One gather of the page representatives.
        dev_pages = (
            (torch.cat([v[::ps] for v, _ in tagged]) // ps).cpu().to(torch.int64)
        )
        cpu_pages = torch.cat([p for _, p in tagged])
        if len(cpu_pages) != len(dev_pages) or not torch.equal(cpu_pages, dev_pages):
            off = 0
            for v, p in tagged:
                k = (len(v) + ps - 1) // ps
                if len(p) != k or not torch.equal(p, dev_pages[off : off + k]):
                    problems.append(
                        f"tree value tag {p[:4].tolist()}.. != device "
                        f"{dev_pages[off:off + 4].tolist()}.."
                    )
                off += k
        used.update(cpu_pages.tolist())

    free = allocator.free_page_ids_all()
    free_list = free.tolist()
    if len(set(free_list)) != len(free_list):
        problems.append(f"duplicate page in the free lists ({len(free_list)} entries)")
    both = used.intersection(free_list)
    if both:
        problems.append(f"{len(both)} pages in use and free, e.g. {sorted(both)[:8]}")
    return problems


# ---------------------------------------------------------------------------
# Decode leg (plain decode and EAGLE / MTP spec decode)
# ---------------------------------------------------------------------------


def int64_array(x: Any) -> np.ndarray:
    """``x`` (a CPU tensor, an array or a sequence of ints) as an int64 numpy
    array, without a copy for an int64 CPU tensor. A device tensor raises:
    the decode path never reads the device."""
    if isinstance(x, torch.Tensor):
        return x.numpy().astype(np.int64, copy=False)
    return np.asarray(x, dtype=np.int64)


def run_pages(
    table: np.ndarray,
    rows: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    new_pages: np.ndarray,
    page_size: int,
) -> tuple:
    """The pages of the runs ``[starts[i], ends[i])`` of ``rows[i]`` in
    ``extend_token_slots``' order, vectorised: a run that starts inside a
    page stays on the row's page there (``table[row, start // ps]``, the page
    of position ``start - 1``); every other page it touches is the next of
    ``new_pages`` (batch order). Returns, per touched page (request-major,
    position order): the batch index, the run's last position on that page
    (``CpuPageRows.write_positions`` grows ``lens`` to the run's end) and the
    page id. An empty run touches nothing."""
    ps = page_size
    k0 = starts // ps
    m = np.where(ends > starts, (ends + ps - 1) // ps - k0, 0)
    idx = np.repeat(np.arange(len(starts)), m)
    first = np.cumsum(m) - m
    k = k0[idx] + (np.arange(len(idx)) - first[idx])
    stay = (k == k0[idx]) & (starts[idx] % ps != 0)
    fresh = ~stay
    if int(np.count_nonzero(fresh)) != len(new_pages):
        raise AssertionError(
            f"Q4 runs touch {int(np.count_nonzero(fresh))} new pages, "
            f"got {len(new_pages)}"
        )
    pages = np.empty(len(idx), dtype=np.int64)
    pages[stay] = table[rows[idx[stay]], k[stay]]
    pages[fresh] = new_pages
    last = np.minimum(ends[idx], (k + 1) * ps) - 1
    return idx, last, pages
