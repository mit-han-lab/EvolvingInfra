# Adapted from SGLang python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py (commit 297e54fd10), modified for evolving-infra.
"""Per-query sparse-index combiner for the FlashMLA sparse prefill path.

Adapts vllm's ``combine_topk_swa_indices`` to sglang's flat-workspace layout.
Reference:
https://github.com/vllm-project/vllm/blob/124fac10cb0ea83aee2ffeabac0b413d6b759b26/vllm/models/deepseek_v4/common/ops/cache_utils.py#L476

For each
query token in a prefill chunk, emits one row of combined indices into the
chunk's bf16 KV workspace:

    [ topk indices into compressed cache (rebased)   ]
    [ swa positional indices (rebased)               ]
    [ -1 padding up to a multiple of 128             ]

The workspace is a single flat ``(total_workspace_tokens, 512)`` tensor
formed by concatenating, per request, that request's compressed-region
gather followed by all requests' SWA-region gathers. Two per-request
offset tensors describe the layout:

  * ``compressed_base[r]`` — flat index where request r's compressed
    region begins. Topk indices ``topk_indices[token, j]`` are local to
    request r's compressed region (in ``[0, compressed_gather_len[r])``)
    and get rebased to flat space by adding ``compressed_base[r]``.
  * ``swa_base[r]`` — flat index where request r's SWA region begins.
    Per-query SWA indices are computed positionally as
    ``swa_base[r] + (pos - swa_len + 1 - gather_start) + j``.

This is the natural layout for ``flash_mla_sparse_fwd``'s ``kv: (s_kv, 1,
d_qk)`` argument, where ``s_kv`` is the total flat workspace length.

For SWA-only layers callers pass ``topk=0``, ``compressed_base = 0`` (the
compressed branch becomes a no-op) and any ``compress_ratio >= 1``.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from decoupled_engine.kernels.ops.attention.dsv4.dequant_k_cache import DIM_NOPE, DIM_ROPE
from decoupled_engine.srt.utils.common import async_h2d

# Bf16 workspace per-token width, matching ``dequantize_k_cache_paged_dsv4``'s
# output: 448 fp8 nope (dequanted) + 64 bf16 rope = 512.
WORKSPACE_DIM = DIM_NOPE + DIM_ROPE
DSV4_Q8KV8_PREFILL_ENV = "SGLANG_DSV4_Q8KV8_PREFILL"
DSV4_Q8KV8_PREFILL_LOG_ENV = "SGLANG_DSV4_Q8KV8_PREFILL_LOG"
# The Q8KV8 path appends one sentinel row after the SWA rows; the group
# planner keeps room for it so both paths share one plan.
SPARSE_PREFILL_RESERVED_ROWS = 1

logger = logging.getLogger(__name__)

from decoupled_engine.kernels.ops.attention.dsv4.sparse_prefill_kernels import (
    build_swa_token_ids,
    combine_topk_swa_indices,
    combined_topk_width,
)


def use_dsv4_q8kv8_sparse_prefill(dsv4_prefill_backend: str = "auto") -> bool:
    """Return whether DeepSeek-V4 sparse prefill should use Q8KV8.

    ``dsv4_prefill_backend`` is the production configuration. The environment
    variable remains as a debug override while the runtime path is being
    hardened: truthy values force Q8 on, falsy values force it off.
    """
    env_value = os.getenv(DSV4_Q8KV8_PREFILL_ENV)
    if env_value is not None:
        return env_value.lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    return dsv4_prefill_backend == "flashmla_sparse_q8"


class SparsePrefillKVBuffer:
    """Backend-owned scratch storage for sparse prefill KV dequantization.

    Callers overwrite the rows they read, except head rows ``get_with_head``
    reports as filled, so token buckets and compression ratios can safely share
    one buffer. Sparse prefill executes eagerly and serially on the supported
    paths, which makes it safe to replace the scratch allocation when a larger
    extent is needed.

    ``budget_rows`` > 0 bounds the workspace: the first allocation takes the
    whole budget at once (so the buffer never grows step by step while the
    allocator is fragmented), and ``SparsePrefillChunkCache`` splits a chunk
    into request groups that each fit the budget. Only one request whose own
    rows exceed the budget can grow the buffer past it. ``budget_rows`` <= 0
    grows on demand (no bound).

    Every (re)allocation logs one INFO line with the requested rows, the old
    and new capacity and, through ``describe``, the chunk shape (requests,
    longest sequence, sum of c4 rows). Lines closer than
    ``LOG_MIN_INTERVAL_S`` apart are dropped and counted in the next line.
    """

    LOG_MIN_INTERVAL_S = 10.0

    def __init__(self, device: torch.device, budget_rows: int = 0):
        self.device = device
        self.budget_rows = int(budget_rows)
        self._buffer: Optional[torch.Tensor] = None
        self._head_source: Optional[tuple] = None
        self._warned_over_budget = False
        self.num_allocations = 0
        self._last_log_time: Optional[float] = None
        self._suppressed_logs = 0

    def _log_allocation(
        self,
        num_tokens: int,
        old_capacity: int,
        new_capacity: int,
        dtype: torch.dtype,
        describe: Optional[Callable[[], str]],
    ) -> None:
        now = time.monotonic()
        if (
            self._last_log_time is not None
            and now - self._last_log_time < self.LOG_MIN_INTERVAL_S
        ):
            self._suppressed_logs += 1
            return
        suppressed, self._suppressed_logs = self._suppressed_logs, 0
        self._last_log_time = now
        nbytes = new_capacity * WORKSPACE_DIM * dtype.itemsize
        logger.info(
            "DSV4 sparse prefill workspace allocation #%d: %d rows requested, "
            "capacity %d -> %d rows (%.1f MiB %s), budget %d rows%s%s",
            self.num_allocations,
            num_tokens,
            old_capacity,
            new_capacity,
            nbytes / (1 << 20),
            str(dtype).replace("torch.", ""),
            self.budget_rows,
            f", {describe()}" if describe is not None else "",
            f" ({suppressed} earlier allocation lines dropped)" if suppressed else "",
        )

    def get(
        self,
        num_tokens: int,
        dtype: torch.dtype = torch.bfloat16,
        describe: Optional[Callable[[], str]] = None,
    ) -> torch.Tensor:
        """Rows ``[:num_tokens]`` of the scratch buffer. ``describe`` is called
        only when the buffer is (re)allocated, for the log line."""
        self._head_source = None
        return self._reserve(num_tokens, dtype, describe)

    def get_with_head(
        self,
        num_tokens: int,
        head_source: tuple,
        describe: Optional[Callable[[], str]] = None,
    ) -> Tuple[torch.Tensor, bool]:
        """Returns the rows and whether the head already holds ``head_source``."""
        buffer = self._reserve(num_tokens, torch.bfloat16, describe)
        head_filled = head_source == self._head_source
        self._head_source = head_source
        return buffer, head_filled

    def forget_head(self) -> None:
        """Per forward: the next forward's tensors may reuse freed addresses."""
        self._head_source = None

    def _reserve(
        self,
        num_tokens: int,
        dtype: torch.dtype,
        describe: Optional[Callable[[], str]] = None,
    ) -> torch.Tensor:
        assert num_tokens > 0
        current_capacity = self._buffer.shape[0] if self._buffer is not None else 0
        current_dtype = self._buffer.dtype if self._buffer is not None else None
        if num_tokens > current_capacity or dtype != current_dtype:
            if (
                self.budget_rows > 0
                and num_tokens > self.budget_rows
                and not self._warned_over_budget
            ):
                self._warned_over_budget = True
                logger.warning(
                    "DSV4 sparse prefill: one request group needs %d workspace rows, "
                    "above the budget of %d rows; growing the workspace.",
                    num_tokens,
                    self.budget_rows,
                )
            # Release the old scratch before the new allocation, so a regrow
            # does not hold the old and the new buffer at the same time.
            self._buffer = None
            new_capacity = max(num_tokens, self.budget_rows)
            self._buffer = torch.empty(
                (new_capacity, 1, WORKSPACE_DIM),
                dtype=dtype,
                device=self.device,
            )
            self._head_source = None
            self.num_allocations += 1
            self._log_allocation(
                num_tokens, current_capacity, new_capacity, dtype, describe
            )
        return self._buffer[:num_tokens]


class SparsePrefillOutputBuffer:
    """Backend-owned output storage of the sparse prefill attention kernels.

    The kernels produce ``[s_q, h_q, d_v]`` bf16 per layer call (h_q padded to
    64 on TP ranks): 512 MiB at 8,192 query rows. A fresh tensor of that size
    on every call can fail on a fragmented allocator although enough memory is
    free in total. This buffer reserves ``reserve_rows`` query rows (the
    prefill chunk size) once, at the first sparse prefill call, which is the
    prefill graph capture warm-up, after the KV pool is sized. Later calls
    take views of it.

    ``get`` returns rows ``[:num_rows]`` of the output, and with
    ``with_stats`` also of the fp32 ``max_logits`` / ``lse`` arrays that the
    Q8KV8 kernel writes. A call with more rows or another head shape replaces
    the buffer (the old one is dropped first) and logs one INFO line.

    A returned view is valid until the next sparse prefill call of the same
    backend. The model consumes the attention output in the same layer
    (inverse RoPE in place, then ``wo_a``), on the same stream, before the next
    layer's attention runs.
    """

    def __init__(self, device: torch.device, reserve_rows: int = 0):
        self.device = device
        self.reserve_rows = max(int(reserve_rows), 0)
        self._out: Optional[torch.Tensor] = None
        self._max_logits: Optional[torch.Tensor] = None
        self._lse: Optional[torch.Tensor] = None
        self.num_allocations = 0

    @property
    def capacity(self) -> int:
        return 0 if self._out is None else self._out.shape[0]

    def _allocate(self, num_rows: int, num_heads: int, head_dim: int) -> None:
        old_capacity = self.capacity
        # Drop the old buffer before the new allocation (no double hold).
        self._out = self._max_logits = self._lse = None
        new_capacity = max(int(num_rows), self.reserve_rows, 1)
        self._out = torch.empty(
            (new_capacity, num_heads, head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.num_allocations += 1
        logger.info(
            "DSV4 sparse prefill output buffer allocation #%d: %d rows "
            "requested, capacity %d -> %d rows x %d heads x %d (%.1f MiB bf16), "
            "reserve %d rows",
            self.num_allocations,
            num_rows,
            old_capacity,
            new_capacity,
            num_heads,
            head_dim,
            self._out.numel() * self._out.element_size() / (1 << 20),
            self.reserve_rows,
        )

    def _ensure(self, num_rows: int, num_heads: int, head_dim: int) -> None:
        out = self._out
        if (
            out is None
            or num_rows > out.shape[0]
            or out.shape[1] != num_heads
            or out.shape[2] != head_dim
        ):
            self._allocate(num_rows, num_heads, head_dim)

    def reserve(self, num_heads: int, head_dim: int) -> None:
        """Allocate ``reserve_rows`` rows now if nothing is allocated yet."""
        if self._out is None and self.reserve_rows > 0:
            self._ensure(self.reserve_rows, num_heads, head_dim)

    def get(
        self,
        num_rows: int,
        num_heads: int,
        head_dim: int,
        with_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        self._ensure(num_rows, num_heads, head_dim)
        if not with_stats:
            return self._out[:num_rows], None, None
        if self._max_logits is None:
            shape = (self._out.shape[0], num_heads)
            self._max_logits = torch.empty(
                shape, dtype=torch.float32, device=self.device
            )
            self._lse = torch.empty(shape, dtype=torch.float32, device=self.device)
        return (
            self._out[:num_rows],
            self._max_logits[:num_rows],
            self._lse[:num_rows],
        )


def default_sparse_prefill_workspace_rows(
    context_len: int, max_extend_tokens: int, swa_window: int
) -> int:
    """Workspace rows that one request at the context limit needs.

    One request of ``seq_len <= context_len`` with ``extend <= max_extend_tokens``
    gathers ``seq_len // 4`` c4 rows (c128 needs fewer, SWA-only none) plus
    ``min(seq_len, extend + swa_window - 1)`` SWA rows, plus the Q8 sentinel.
    """
    return (
        int(context_len) // 4
        + max(int(max_extend_tokens), 0)
        + int(swa_window)
        - 1
        + SPARSE_PREFILL_RESERVED_ROWS
    )


@dataclass(frozen=True)
class SparsePrefillGroup:
    """A contiguous request range that shares one workspace fill and one
    attention call. Ranges are half-open: requests ``[req_start, req_end)``,
    query rows ``[q_start, q_end)``, rows of the packed compressed token ids
    ``[c_start, c_end)`` and rows of ``swa_token_ids`` ``[s_start, s_end)``.
    The last group also owns the padded query rows of a graph bucket."""

    req_start: int
    req_end: int
    q_start: int
    q_end: int
    c_start: int
    c_end: int
    s_start: int
    s_end: int

    @property
    def num_compressed(self) -> int:
        return self.c_end - self.c_start

    @property
    def num_swa(self) -> int:
        return self.s_end - self.s_start

    @property
    def num_rows(self) -> int:
        return self.num_compressed + self.num_swa


def _exclusive_cumsum(values: Sequence[int]) -> List[int]:
    out = [0]
    for v in values:
        out.append(out[-1] + int(v))
    return out


def plan_sparse_prefill_groups(
    compressed_counts: Sequence[int],
    swa_counts: Sequence[int],
    extend_lens: Sequence[int],
    num_qo_tokens: int,
    budget_rows: int,
    reserved_rows: int = SPARSE_PREFILL_RESERVED_ROWS,
) -> List[SparsePrefillGroup]:
    """Split requests into contiguous groups whose workspace fits the budget.

    Request ``i`` needs ``compressed_counts[i] + swa_counts[i]`` rows. Requests
    are added greedily in batch order; a group closes when the next request
    would make ``rows + reserved_rows`` exceed ``budget_rows``. One request
    that alone exceeds the budget forms its own group. ``budget_rows <= 0``
    gives one group (the unbounded layout).
    """
    n = len(compressed_counts)
    assert n > 0 and len(swa_counts) == n and len(extend_lens) == n
    c_off = _exclusive_cumsum(compressed_counts)
    s_off = _exclusive_cumsum(swa_counts)
    q_off = _exclusive_cumsum(extend_lens)

    bounds = []
    start = 0
    rows = 0
    for i in range(n):
        need = int(compressed_counts[i]) + int(swa_counts[i])
        if i > start and budget_rows > 0 and rows + need + reserved_rows > budget_rows:
            bounds.append((start, i))
            start, rows = i, 0
        rows += need
    bounds.append((start, n))

    groups = []
    for a, b in bounds:
        groups.append(
            SparsePrefillGroup(
                req_start=a,
                req_end=b,
                q_start=q_off[a],
                q_end=num_qo_tokens if b == n else q_off[b],
                c_start=c_off[a],
                c_end=c_off[b],
                s_start=s_off[a],
                s_end=s_off[b],
            )
        )
    return groups


def sparse_prefill_group_bases(
    groups: Sequence[SparsePrefillGroup],
    compressed_counts: Sequence[int],
    swa_counts: Sequence[int],
) -> Tuple[List[int], List[int]]:
    """Per-request workspace bases, relative to the request's group.

    A group's workspace is ``[compressed rows of its requests][SWA rows of its
    requests]``, both packed in request order. With one group this is
    ``compressed_base[i] = sum(compressed_counts[:i])`` and
    ``swa_base[i] = sum(compressed_counts) + sum(swa_counts[:i])``.
    """
    c_off = _exclusive_cumsum(compressed_counts)
    s_off = _exclusive_cumsum(swa_counts)
    compressed_base: List[int] = []
    swa_base: List[int] = []
    for g in groups:
        for i in range(g.req_start, g.req_end):
            compressed_base.append(c_off[i] - c_off[g.req_start])
            swa_base.append(g.num_compressed + s_off[i] - s_off[g.req_start])
    return compressed_base, swa_base


def _int32_to_device(values: Sequence[int], device: torch.device) -> torch.Tensor:
    # Pinned + non_blocking: no host wait on the work already queued.
    host = torch.tensor(list(values), dtype=torch.int32)
    if device.type == "cuda":
        host = host.pin_memory()
    return host.to(device, non_blocking=True)


def packed_compressed_token_ids(
    per_req_slots: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    total: int,
    page_size: Optional[int] = None,
) -> torch.Tensor:
    """Gather each request's first ``counts[r]`` compressed slots, packed.

    Args:
        per_req_slots: (num_reqs, width) per-request table. With ``page_size``
            it holds page ids (slot = page * page_size + k % page_size);
            without it, slot ids for k directly.
        counts: (num_reqs,) int64, rows of request r (``seq_len // ratio``).
        offsets: (num_reqs + 1,) int64 exclusive cumsum of ``counts``.
        total: ``sum(counts)`` as a Python int (no device sync).

    Returns:
        (total,) int32 slot ids; request r occupies
        ``[offsets[r], offsets[r] + counts[r])``. Negative slots clamp to 0,
        as in the padded layout.
    """
    device = per_req_slots.device
    if total == 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    req = torch.repeat_interleave(counts, output_size=total)
    k = torch.arange(total, dtype=torch.int64, device=device) - offsets[req]
    if page_size is None:
        slots = per_req_slots[req, k]
    else:
        slots = (
            per_req_slots[req, torch.div(k, page_size, rounding_mode="floor")]
            * page_size
            + k % page_size
        )
    return slots.clamp_min(0).to(torch.int32)


@dataclass
class CompressedGather:
    """Positional layout of one compressed cache inside the workspace."""

    flat_token_ids: torch.Tensor  # (total gathered rows,) int32
    compressed_base: torch.Tensor  # (num_reqs,) int32
    swa_base: torch.Tensor  # (num_reqs,) int32
    # Tail stays at the -1 sentinel because the valid prefix length is
    # chunk-invariant per request; subsequent layers only overwrite that prefix.
    combined_indices: Optional[torch.Tensor] = None
    combined_lens: Optional[torch.Tensor] = None


@dataclass
class SparsePrefillChunkCache:
    """Cache prefill-chunk metadata shared across layers.

    Fields depend on request/token mappings and compressed page tables, not
    per-layer k_cache; per-layer top-k combinations are recomputed into reused
    buffers.
    """

    # Geometry computed once per chunk.
    num_reqs: int
    num_qo_tokens: int
    # Actual maximum sequence length in this forward. CUDA-graph metadata may
    # have a much wider page table sized for the capture limit; gather only the
    # live sequence extent instead of materializing that padded capacity.
    max_seq_len: int
    # Model's SWA window — the per-query attention range. Used by
    # combine_topk_swa_indices' WINDOW_SIZE and by build_swa_token_ids's
    # gather_lens. Must match SWA_WINDOW from the backend (e.g. 128), NOT
    # the SWA pool's storage page size (often 256).
    swa_window_size: int
    # SWA cache pool's storage page size — used as the dequant kernel's
    # ``page_size`` so that ``slot // page_size`` recovers the right page.
    swa_page_size: int
    seq_lens: torch.Tensor  # (num_reqs,) int32
    query_start_loc: torch.Tensor  # (num_reqs+1,) int32, query rows per request
    query_pos: torch.Tensor  # (num_qo_tokens,) int32, sequence position per row

    # SWA-side (every layer needs these, all chunk-invariant).
    swa_token_ids: torch.Tensor  # (total_swa,) int32
    swa_first_pos: torch.Tensor  # (num_reqs,) int32
    swa_gather_lens: torch.Tensor  # (num_reqs,) int32
    swa_offsets: torch.Tensor  # (num_reqs+1,) int32

    # c0 pre-computed combine output (entire input set is chunk-invariant).
    c0_combined_indices: torch.Tensor = field(default=None)
    c0_combined_lens: torch.Tensor = field(default=None)
    # Compressed caches keyed by compress ratio: c128 (every block, combined once
    # per chunk) and the top-k ratios (combined per layer).
    compressed: Dict[int, CompressedGather] = field(default_factory=dict)
    # Per-layer scratch only: combine overwrites all entries before attention.
    # c0/c128 are cached across layers and must retain separate storage.
    low_ratio_combined: dict = field(default_factory=dict)
    swa_indices: Optional[torch.Tensor] = None
    swa_lengths: Optional[torch.Tensor] = None
    # Host lengths avoid a device synchronization when sizing ragged gathers.
    seq_lens_cpu: Optional[list[int]] = None

    # Optional per-query visible-window overrides (DSV4-VL image spans). The
    # existing union gather [seq_len - extend - (W-1), seq_len) already covers
    # every visible window, since a visible window [win_start, span_end) of a
    # fully-contained span satisfies win_start >= span_start - (W-1) >=
    # seq_len - extend - (W-1) and span_end <= seq_len.
    swa_win_starts: Optional[torch.Tensor] = None  # (num_qo_tokens,) int32
    swa_win_lens: Optional[torch.Tensor] = None  # (num_qo_tokens,) int32
    swa_win_max_len: Optional[int] = None

    # EI: CPU copies of the per-request lengths (no device sync to plan groups).
    extend_seq_lens_cpu: Optional[List[int]] = None
    swa_gather_lens_cpu: Optional[List[int]] = None
    # Workspace row budget per attention call (<= 0: one group, unbounded).
    workspace_budget_rows: int = 0
    # Request groups per compress ratio (0: SWA-only); each group is one
    # workspace fill + one attention call. A missing ratio runs as one group.
    groups: Dict[int, List[SparsePrefillGroup]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        query_lens: torch.Tensor,
        query_pos: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        full_to_swa: torch.Tensor,
        swa_window_size: int,
        swa_page_size: int,
        num_qo_tokens: int,
        max_seq_len: int,
        total_swa: int,
        request_window_layout=None,
        seq_lens_cpu: Optional[list[int]] = None,
        swa_win_starts: Optional[torch.Tensor] = None,
        swa_win_lens: Optional[torch.Tensor] = None,
        extend_seq_lens_cpu: Optional[Sequence[int]] = None,
        workspace_budget_rows: int = 0,
    ) -> "SparsePrefillChunkCache":
        """``query_lens`` / ``query_pos``: the rows this forward runs (the extend, or
        a CP rank's interleaved share of it); the SWA gather spans the whole extend.

        EI: ``extend_seq_lens_cpu`` (with ``seq_lens_cpu``) plans request groups
        that each fit ``workspace_budget_rows``; pass it only when the query rows
        are the whole extend of each request (no CP share, no RequestWindow
        layout). Without it the chunk runs as one group."""
        device = seq_lens.device
        num_reqs = seq_lens.shape[0]

        query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        query_start_loc[1:] = torch.cumsum(query_lens, dim=0).to(torch.int32)

        if request_window_layout is None:
            swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets = (
                build_swa_token_ids(
                    seq_lens=seq_lens,
                    extend_seq_lens=extend_seq_lens,
                    req_pool_indices=req_pool_indices,
                    req_to_token=req_to_token,
                    full_to_swa=full_to_swa,
                    swa_window=swa_window_size,
                    total_swa=total_swa,
                )
            )
            swa_indices = swa_lengths = None
        else:
            swa_token_ids = torch.arange(
                request_window_layout.size, dtype=torch.int32, device=device
            )
            swa_first_pos = torch.zeros_like(seq_lens)
            swa_gather_lens = torch.zeros_like(seq_lens)
            # RequestWindow indices already address the shared workspace.
            swa_offsets = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
            swa_indices = request_window_layout.indices
            swa_lengths = request_window_layout.lengths

        swa_gather_lens_cpu = None
        if (
            request_window_layout is None
            and seq_lens_cpu is not None
            and extend_seq_lens_cpu is not None
        ):
            extend_seq_lens_cpu = [int(x) for x in extend_seq_lens_cpu]
            assert len(seq_lens_cpu) == num_reqs == len(extend_seq_lens_cpu)
            swa_gather_lens_cpu = [
                min(int(seq_len), extend_len + swa_window_size - 1)
                for seq_len, extend_len in zip(seq_lens_cpu, extend_seq_lens_cpu)
            ]
            assert sum(swa_gather_lens_cpu) == total_swa, (
                sum(swa_gather_lens_cpu),
                total_swa,
            )
        else:
            extend_seq_lens_cpu = None

        swa_win_max_len = None
        if swa_win_lens is not None:
            assert swa_win_starts is not None
            # Computed once per chunk; reused by every layer's combine.
            swa_win_max_len = int(swa_win_lens.max().item())

        cache = cls(
            num_reqs=num_reqs,
            seq_lens_cpu=seq_lens_cpu,
            num_qo_tokens=num_qo_tokens,
            max_seq_len=max_seq_len,
            swa_window_size=swa_window_size,
            swa_page_size=swa_page_size,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            query_pos=query_pos,
            swa_token_ids=swa_token_ids,
            swa_first_pos=swa_first_pos,
            swa_gather_lens=swa_gather_lens,
            swa_offsets=swa_offsets,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            swa_win_starts=swa_win_starts,
            swa_win_lens=swa_win_lens,
            swa_win_max_len=swa_win_max_len,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            swa_gather_lens_cpu=swa_gather_lens_cpu,
            workspace_budget_rows=int(workspace_budget_rows),
        )

        # Pre-compute the c0 combine output: TOPK=0, compressed_base=0,
        # swa_base = swa_offsets[:-1] (one group). All inputs are chunk-invariant.
        zero_topk = torch.zeros((num_qo_tokens, 1), dtype=torch.int32, device=device)
        zero_compressed_base = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        c0_groups = cache._plan_groups(0, [0] * num_reqs)
        if c0_groups is None or len(c0_groups) == 1:
            c0_swa_base = swa_offsets[:-1].to(torch.int32)
        else:
            _, c0_swa_base_cpu = sparse_prefill_group_bases(
                c0_groups, [0] * num_reqs, swa_gather_lens_cpu
            )
            c0_swa_base = _int32_to_device(c0_swa_base_cpu, device)
        cache.c0_combined_indices, cache.c0_combined_lens = combine_topk_swa_indices(
            topk_indices=zero_topk,
            query_start_loc=query_start_loc,
            query_pos=query_pos,
            seq_lens=seq_lens,
            gather_lens=swa_gather_lens,
            compressed_base=zero_compressed_base,
            swa_base=c0_swa_base,
            window_size=swa_window_size,
            compress_ratio=1,
            topk=0,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            swa_win_starts=swa_win_starts,
            swa_win_lens=swa_win_lens,
            swa_win_max_len=swa_win_max_len,
        )
        return cache

    def _workspace_bases(self, c_max: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Flat workspace offsets of each request's compressed and SWA regions:
        ``c_max`` compressed slots per request, then the SWA gather."""
        device = self.seq_lens.device
        compressed_base = (
            torch.arange(self.num_reqs, dtype=torch.int32, device=device) * c_max
        ).to(torch.int32)
        swa_base = (self.num_reqs * c_max + self.swa_offsets[:-1]).to(torch.int32)
        return compressed_base, swa_base

    def layer_inputs(
        self,
        compress_ratio: int,
        core_attn_metadata,
        c_page_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(flat_token_ids, combined_indices, combined_lens)`` for one compressed
        layer. c128 gathers every block and combines once per chunk; a top-k ratio
        gathers from the page table once and combines per layer from the indexer's
        raw top-k."""
        if compress_ratio == 128:
            page_indices = core_attn_metadata.sparse_page_indices(128)
            assert page_indices is not None
            gather = self.ensure_c128(page_indices)
            return gather.flat_token_ids, gather.combined_indices, gather.combined_lens
        raw_indices = core_attn_metadata.sparse_raw_indices(compress_ratio)
        assert raw_indices is not None, (
            f"sparse-prefill c{compress_ratio} path requires the raw top-k indices "
            "(allocated in init_flashmla_related when is_prefill=True)"
        )
        gather = self.ensure_compressed(
            compress_ratio, core_attn_metadata.page_table, c_page_size
        )
        combined_indices, combined_lens = self.combine_compressed(
            compress_ratio, raw_indices[: self.num_qo_tokens]
        )
        return gather.flat_token_ids, combined_indices, combined_lens

    def ensure_c128(self, c128_page_indices: torch.Tensor) -> CompressedGather:
        """Populate c128-side fields from per-query c128 page indices.

        ``c128_page_indices[q, j]`` carries slot ids derived from
        ``page_table[q]`` (request-keyed; same across queries of a request)
        but masked per-token by ``j < seq_lens_casual[q] // 128`` — entries
        beyond that are -1. We need a row whose mask covers every j the
        combine kernel might reference, i.e. up to ``seq_lens[r] // 128``;
        that's the *last* query's mask. Pulling the first query in a fresh
        prefill (``seq_lens_casual = 1``) yields an all-`-1` row that
        clamp_min(0) collapses to slot 0, sending dequant to a polluted
        slot and producing garbage c128 entries.
        """
        gather = self.compressed.get(128)
        if gather is not None:
            return gather
        device = self.seq_lens.device
        c128_max = max(self.max_seq_len // 128, 1)
        assert c128_max <= c128_page_indices.shape[-1], (
            f"live c128 extent {c128_max} exceeds metadata capacity "
            f"{c128_page_indices.shape[-1]}"
        )
        # A request without rows on this rank gathers into a region nothing reads.
        last_q_per_req = (self.query_start_loc[1:] - 1).clamp_min(0).long()
        per_req_c128 = c128_page_indices.narrow(1, 0, c128_max).index_select(
            0, last_q_per_req
        )
        # EI: packed. Request r keeps only its own seq_len // 128 entries, all
        # valid in its last query's row. A query reads j < (pos + 1) // 128
        # <= seq_len // 128, so it sees the same slots as in a padded layout.
        # Clamp -1 -> 0 so dequant doesn't OOB.
        counts_cpu = self._compressed_counts(128)
        flat_c128_ids, compressed_base, swa_base = self._packed_layout(
            per_req_c128,
            counts_cpu,
            self._plan_groups(128, counts_cpu),
            page_size=None,
        )
        # Pre-compute the c128 combine output. topk_indices[q, j] = j is the
        # arange-broadcast pattern. The kernel reads it through stride(0), so
        # a stride-0 broadcast row gives the same values without a
        # (num_qo_tokens, c128_max) int32 copy (236 MB at 8192 x 1M context).
        topk_indices = torch.arange(c128_max, dtype=torch.int32, device=device)[
            None, :
        ].expand(self.num_qo_tokens, -1)
        combined_indices, combined_lens = combine_topk_swa_indices(
            topk_indices=topk_indices,
            query_start_loc=self.query_start_loc,
            query_pos=self.query_pos,
            seq_lens=self.seq_lens,
            gather_lens=self.swa_gather_lens,
            compressed_base=compressed_base,
            swa_base=swa_base,
            window_size=self.swa_window_size,
            compress_ratio=128,
            topk=c128_max,
            swa_indices=self.swa_indices,
            swa_lengths=self.swa_lengths,
            swa_win_starts=self.swa_win_starts,
            swa_win_lens=self.swa_win_lens,
            swa_win_max_len=self.swa_win_max_len,
        )

        gather = CompressedGather(
            flat_token_ids=flat_c128_ids,
            compressed_base=compressed_base,
            swa_base=swa_base,
            combined_indices=combined_indices,
            combined_lens=combined_lens,
        )
        self.compressed[128] = gather
        return gather

    def ensure_compressed(
        self,
        compress_ratio: int,
        page_table: torch.Tensor,
        c_page_size: int,
    ) -> CompressedGather:
        """``page_table`` is (num_qo_tokens, max_blocks); rows within a request
        are duplicates. The combine output is per-layer (depends on the
        layer's remapped topk_indices), so we only cache the gather-side
        scaffolding plus compressed/swa bases.
        """
        gather = self.compressed.get(compress_ratio)
        if gather is not None:
            return gather
        device = self.seq_lens.device
        c_max = max(self.max_seq_len // compress_ratio, 1)
        c_capacity = page_table.shape[-1] * c_page_size
        assert c_max <= c_capacity, (
            f"live c{compress_ratio} extent {c_max} exceeds metadata capacity {c_capacity}"
        )
        first_q_per_req = (
            self.query_start_loc[:-1].clamp_max(self.num_qo_tokens - 1).long()
        )
        # Each request needs only its own live compressed history. Keeping a
        # rectangular [num_reqs, c_max] gather duplicates the longest extent
        # for every short request in a mixed batch.
        seq_lens_cpu = self.seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = self.seq_lens.tolist()
        lengths_cpu = [int(length) // compress_ratio for length in seq_lens_cpu]
        total_compressed = sum(lengths_cpu)
        lengths = async_h2d(lengths_cpu, dtype=torch.int32, device=device)
        ends = lengths.cumsum(0, dtype=torch.int32)
        compressed_base = ends - lengths
        swa_base = total_compressed + self.swa_offsets[:-1]
        # EI: a chunk split into request groups addresses each group's own
        # workspace ([its compressed rows][its SWA rows]).
        groups = self._plan_groups(compress_ratio, lengths_cpu)
        if groups is not None and len(groups) > 1:
            compressed_base_cpu, swa_base_cpu = sparse_prefill_group_bases(
                groups, lengths_cpu, self.swa_gather_lens_cpu
            )
            compressed_base = _int32_to_device(compressed_base_cpu, device)
            swa_base = _int32_to_device(swa_base_cpu, device)

        # output_size is known on the host, so repeat_interleave does not need
        # to synchronize the device to determine its output allocation.
        request_ids = torch.repeat_interleave(
            torch.arange(self.num_reqs, device=device),
            lengths,
            output_size=total_compressed,
        )
        positions = (
            torch.arange(total_compressed, device=device) - compressed_base[request_ids]
        )
        pages = page_table[first_q_per_req[request_ids], positions // c_page_size]
        flat_ids = (pages * c_page_size + positions % c_page_size).to(torch.int32)
        flat_ids.clamp_min_(0)

        gather = CompressedGather(
            flat_token_ids=flat_ids,
            compressed_base=compressed_base,
            swa_base=swa_base,
        )
        self.compressed[compress_ratio] = gather
        return gather

    def combine_compressed(
        self,
        compress_ratio: int,
        sparse_raw_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``sparse_raw_indices`` is the top-k as request-local compressed
        positions, already the workspace coordinate ``combine_topk_swa_indices``
        expects, so no remap is needed.
        """
        gather = self.compressed[compress_ratio]
        topk = sparse_raw_indices.shape[-1]
        width = combined_topk_width(topk, self.eff_swa_window_size)
        if compress_ratio in (1, 2) and width in self.low_ratio_combined:
            gather.combined_indices, gather.combined_lens = self.low_ratio_combined[
                width
            ]
        if gather.combined_indices is None:
            device = self.seq_lens.device
            gather.combined_indices = torch.full(
                (self.num_qo_tokens, width),
                -1,
                dtype=torch.int32,
                device=device,
            )
            gather.combined_lens = torch.zeros(
                self.num_qo_tokens, dtype=torch.int32, device=device
            )
        if compress_ratio in (1, 2):
            self.low_ratio_combined[width] = (
                gather.combined_indices,
                gather.combined_lens,
            )
        return combine_topk_swa_indices(
            topk_indices=sparse_raw_indices,
            query_start_loc=self.query_start_loc,
            query_pos=self.query_pos,
            seq_lens=self.seq_lens,
            gather_lens=self.swa_gather_lens,
            compressed_base=gather.compressed_base,
            swa_base=gather.swa_base,
            window_size=self.swa_window_size,
            compress_ratio=compress_ratio,
            topk=topk,
            out_indices=gather.combined_indices,
            out_lens=gather.combined_lens,
            swa_indices=self.swa_indices,
            swa_lengths=self.swa_lengths,
            swa_win_starts=self.swa_win_starts,
            swa_win_lens=self.swa_win_lens,
            swa_win_max_len=self.swa_win_max_len,
        )

    def _compressed_counts(self, ratio: int) -> List[int]:
        if self.seq_lens_cpu is None:
            # A cache built without the CPU lengths (direct construction, not
            # build()): read the device lengths once.
            self.seq_lens_cpu = [int(x) for x in self.seq_lens.tolist()]
        return [int(seq_len) // ratio for seq_len in self.seq_lens_cpu]

    def _plan_groups(
        self, compress_ratio: int, compressed_counts: Sequence[int]
    ) -> Optional[List[SparsePrefillGroup]]:
        if self.swa_gather_lens_cpu is None or self.extend_seq_lens_cpu is None:
            return None  # no CPU lengths to plan with: one group (groups_for)
        groups = plan_sparse_prefill_groups(
            compressed_counts,
            self.swa_gather_lens_cpu,
            self.extend_seq_lens_cpu,
            self.num_qo_tokens,
            self.workspace_budget_rows,
        )
        self.groups[compress_ratio] = groups
        return groups

    def _packed_layout(
        self,
        per_req_slots: torch.Tensor,
        counts_cpu: Sequence[int],
        groups: Optional[Sequence[SparsePrefillGroup]],
        page_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Packed flat token ids plus group-relative compressed/SWA bases."""
        device = self.seq_lens.device
        total = sum(counts_cpu)
        # ``counts`` comes from the same CPU list as ``total`` and the group
        # plan: repeat_interleave(output_size=total) needs sum(counts) ==
        # total, which a separate device-side seq_lens // ratio would not
        # guarantee. One small pinned non-blocking copy per ratio per chunk.
        counts = _int32_to_device(counts_cpu, device).to(torch.int64)
        offsets = torch.zeros(self.num_reqs + 1, dtype=torch.int64, device=device)
        offsets[1:] = torch.cumsum(counts, dim=0)
        flat_ids = packed_compressed_token_ids(
            per_req_slots, counts, offsets, total, page_size=page_size
        )
        if groups is None or len(groups) == 1:
            # Device-side bases: no host-to-device copy on the common path.
            compressed_base = offsets[:-1].to(torch.int32)
            swa_base = (total + self.swa_offsets[:-1]).to(torch.int32)
        else:
            compressed_base_cpu, swa_base_cpu = sparse_prefill_group_bases(
                groups, counts_cpu, self.swa_gather_lens_cpu
            )
            compressed_base = _int32_to_device(compressed_base_cpu, device)
            swa_base = _int32_to_device(swa_base_cpu, device)
        return flat_ids, compressed_base, swa_base

    def describe_for_log(self, group: Optional[SparsePrefillGroup] = None) -> str:
        """Chunk shape for the workspace allocation log line."""
        text = f"num_reqs={self.num_reqs}, max_seq_len={self.max_seq_len}"
        if self.seq_lens_cpu is not None:
            text += f", sum_seq_div4={sum(int(s) // 4 for s in self.seq_lens_cpu)}"
        text += f", num_qo_tokens={self.num_qo_tokens}"
        if group is not None and (group.req_start, group.req_end) != (
            0,
            self.num_reqs,
        ):
            text += f", group requests [{group.req_start}, {group.req_end})"
        return text

    def groups_for(
        self, compress_ratio: int, num_compressed: int
    ) -> List[SparsePrefillGroup]:
        """Request groups for this ratio; one whole-chunk group when no plan
        was built (callers that fill the scaffolding themselves)."""
        groups = self.groups.get(compress_ratio)
        if groups is None:
            return [
                SparsePrefillGroup(
                    req_start=0,
                    req_end=self.num_reqs,
                    q_start=0,
                    q_end=self.num_qo_tokens,
                    c_start=0,
                    c_end=num_compressed,
                    s_start=0,
                    s_end=self.swa_token_ids.shape[0],
                )
            ]
        return groups

    @property
    def eff_swa_window_size(self) -> int:
        """SWA width the combine output must hold: the causal window, or the
        max visible-window length when image-span overrides are present."""
        if self.swa_win_max_len is None:
            return self.swa_window_size
        return max(self.swa_window_size, self.swa_win_max_len)
