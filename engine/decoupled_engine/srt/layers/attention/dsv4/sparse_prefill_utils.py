# Adapted from SGLang python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py (commit 9a489f8d2f), modified for evolving-infra.
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
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import triton

from decoupled_engine.kernels.ops.attention.dsv4.dequant_k_cache import DIM_NOPE, DIM_ROPE
from decoupled_engine.srt.utils import ceil_align

# FlashMLA sparse prefill asserts ``params.topk % B_TOPK == 0``. B_TOPK is 64
# for the h_q=64 kernel and 128 for h_q=128; pad to 128 to satisfy both.
SPARSE_PREFILL_TOPK_ALIGNMENT = 128
# Bf16 workspace per-token width, matching ``dequantize_k_cache_paged``'s
# output: 448 fp8 nope (dequanted) + 64 bf16 rope = 512.
WORKSPACE_DIM = DIM_NOPE + DIM_ROPE
DSV4_Q8KV8_PREFILL_ENV = "SGLANG_DSV4_Q8KV8_PREFILL"
DSV4_Q8KV8_PREFILL_LOG_ENV = "SGLANG_DSV4_Q8KV8_PREFILL_LOG"
# The Q8KV8 path appends one sentinel row after the SWA rows; the group
# planner keeps room for it so both paths share one plan.
SPARSE_PREFILL_RESERVED_ROWS = 1

logger = logging.getLogger(__name__)

from decoupled_engine.kernels.ops.attention.dsv4.sparse_prefill_kernels import (
    _build_swa_token_ids_kernel,
    _combine_topk_swa_indices_kernel,
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


class SparsePrefillWorkspace:
    """Backend-owned scratch storage for sparse prefill KV dequantization.

    The workspace contents are fully overwritten before every attention call,
    so token buckets and compression ratios can safely share one buffer. Sparse
    prefill executes eagerly and serially on the supported paths, which makes it
    safe to replace the scratch allocation when a larger extent is needed.

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


def combined_topk_width(topk: int, window_size: int) -> int:
    """Width of the padded combined_indices last dim that
    ``combine_topk_swa_indices`` would produce for these args."""
    return ceil_align(topk + window_size, SPARSE_PREFILL_TOPK_ALIGNMENT)


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    compressed_base: torch.Tensor,
    swa_base: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    out_indices: Optional[torch.Tensor] = None,
    out_lens: Optional[torch.Tensor] = None,
    swa_win_starts: Optional[torch.Tensor] = None,
    swa_win_lens: Optional[torch.Tensor] = None,
    swa_win_max_len: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine topk + SWA indices into a single ``flash_mla_sparse_fwd`` row.

    Args:
        topk_indices: (num_tokens, K) int32. Per-query indices into the
            compressed-cache region, **already in request-local space** —
            i.e. in ``[0, compressed_gather_len[r])`` for the request that
            owns each token. Pad entries can be any value; they are ignored
            beyond ``topk_len``.
        query_start_loc: (num_reqs+1,) int32. Cumulative query lengths; may
            be in global (cross-chunk) space — kernel rebases by subtracting
            ``query_start_loc[0]``.
        seq_lens: (num_reqs,) int32. Each request's full sequence length.
        gather_lens: (num_reqs,) int32. Trailing tokens dequanted into the
            SWA region for that request.
        compressed_base: (num_reqs,) int32. Flat workspace offset where
            request r's compressed region begins. Pass all-zeros (or any
            value) for SWA-only layers since topk=0 disables this branch.
        swa_base: (num_reqs,) int32. Flat workspace offset where request
            r's SWA region begins.
        window_size: SWA window size (the causal window; also the width when
            no visible-window overrides are given).
        compress_ratio: must be ``>= 1`` even when topk==0.
        topk: configured topk; pass 0 for SWA-only layers.
        out_indices: optional preallocated ``(num_tokens, combined_topk)``
            int32 buffer. If provided, the kernel writes the per-query prefix
            ``[0, topk_len + swa_len)``; positions beyond are not touched.
            Caller must pre-fill with ``-1`` sentinels (and the chunk-invariant
            valid-prefix length must hold across reuses).
        out_lens: optional preallocated ``(num_tokens,)`` int32 buffer; the
            kernel fully overwrites it, so any dtype-correct buffer works.
        swa_win_starts: optional (num_tokens,) int32 per-query window start
            (absolute seq position) for the visible window of image spans.
            Must be paired with ``swa_win_lens``; None keeps the causal
            window ``[pos - swa_len + 1, pos]``.
        swa_win_lens: optional (num_tokens,) int32 per-query window length.
        swa_win_max_len: optional precomputed ``int(swa_win_lens.max())`` to
            avoid a device sync per call; combined width and the kernel's SWA
            arange are sized to ``max(window_size, swa_win_max_len)``.

    Returns:
        combined_indices: (num_tokens, padded_topk_swa) int32, padded to a
            multiple of 128 with -1 sentinels.
        combined_lens: (num_tokens,) int32, valid prefix length per token.
    """
    assert topk_indices.dtype == torch.int32
    assert query_start_loc.dtype == torch.int32
    assert seq_lens.dtype == torch.int32
    assert gather_lens.dtype == torch.int32
    assert compressed_base.dtype == torch.int32
    assert swa_base.dtype == torch.int32
    assert compress_ratio >= 1, "compress_ratio must be >= 1 (use topk=0 for SWA-only)"
    assert (
        topk_indices.shape[-1] >= topk
    ), f"topk_indices width {topk_indices.shape[-1]} must be >= topk {topk}"

    has_swa_win = swa_win_lens is not None
    eff_window = window_size
    if has_swa_win:
        assert swa_win_starts is not None
        assert swa_win_starts.dtype == torch.int32
        assert swa_win_lens.dtype == torch.int32
        if swa_win_max_len is None:
            swa_win_max_len = int(swa_win_lens.max().item())
        eff_window = max(window_size, swa_win_max_len)

    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_topk = combined_topk_width(topk, eff_window)
    if out_indices is None:
        combined_indices = torch.full(
            (num_tokens, combined_topk),
            -1,
            dtype=torch.int32,
            device=topk_indices.device,
        )
    else:
        assert out_indices.shape == (num_tokens, combined_topk)
        assert out_indices.dtype == torch.int32
        combined_indices = out_indices
    if out_lens is None:
        combined_lens = torch.zeros(
            num_tokens, dtype=torch.int32, device=topk_indices.device
        )
    else:
        assert out_lens.shape == (num_tokens,)
        assert out_lens.dtype == torch.int32
        combined_lens = out_lens

    NUM_WORKERS = 128
    _combine_topk_swa_indices_kernel[(num_reqs, NUM_WORKERS)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc,
        seq_lens,
        gather_lens,
        compressed_base,
        swa_base,
        swa_win_starts if has_swa_win else combined_lens,
        swa_win_lens if has_swa_win else combined_lens,
        top_k=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        SWA_BLOCK=triton.next_power_of_2(eff_window),
        PADDED_TOP_K=triton.next_power_of_2(topk_indices.shape[-1]),
        HAS_SWA_WIN=has_swa_win,
    )
    return combined_indices, combined_lens


def build_swa_token_ids(
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa: torch.Tensor,
    swa_window: int,
    total_swa: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a flat list of physical SWA-cache token IDs covering each
    request's positional union of every query's SWA window.

    Per request, the union spans seq positions
    ``[max(0, seq_len - extend - W + 1), seq_len)``, of length
    ``min(seq_len, extend + W - 1)``. Each position is translated through
    ``req_to_token`` (full kv-cache id) and then ``full_to_swa`` (SWA
    cache id) to land in the SWA-cache token-id space that
    ``dequantize_k_cache_paged`` consumes.

    Args:
        seq_lens: (num_reqs,) int32, per-request total sequence length.
        extend_seq_lens: (num_reqs,) int32, per-request query length.
        req_pool_indices: (num_reqs,) int32, per-request row in
            ``req_to_token``.
        req_to_token: (num_reqs_max, max_seq_len) int32. Full kv-cache id
            per (request, seq position).
        full_to_swa: (full_pool_size + extra,) int64. Maps full kv id to
            SWA-cache id.
        swa_window: int. SWA window size.
        total_swa: Number of token IDs to allocate, computed from CPU lengths.

    Returns:
        swa_token_ids: (total_swa,) int32, flat physical SWA-cache token IDs.
        swa_first_pos: (num_reqs,) int32, first seq position covered per req.
        swa_gather_lens: (num_reqs,) int32, gather length per request.
        swa_offsets: (num_reqs+1,) int32, exclusive cumsum of swa_gather_lens.
    """
    assert seq_lens.dtype == torch.int32
    assert extend_seq_lens.dtype == torch.int32
    assert req_pool_indices.dtype == torch.int32
    assert req_to_token.dtype == torch.int32
    assert full_to_swa.dtype == torch.int64

    num_reqs = seq_lens.shape[0]
    device = seq_lens.device

    swa_gather_lens = torch.minimum(seq_lens, extend_seq_lens + (swa_window - 1)).to(
        torch.int32
    )
    swa_first_pos = (seq_lens - swa_gather_lens).to(torch.int32)
    swa_offsets = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    swa_offsets[1:] = torch.cumsum(swa_gather_lens, dim=0).to(torch.int32)
    swa_token_ids = torch.empty(total_swa, dtype=torch.int32, device=device)
    if total_swa == 0:
        return swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets

    NUM_WORKERS = 128
    _build_swa_token_ids_kernel[(num_reqs, NUM_WORKERS)](
        swa_token_ids,
        swa_first_pos,
        swa_gather_lens,
        swa_offsets,
        req_pool_indices,
        req_to_token,
        req_to_token.stride(0),
        full_to_swa,
    )
    return swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets


@dataclass
class SparsePrefillChunkCache:
    """Chunk-invariant scaffolding for ``_forward_prefill_sparse``.

    The fields here depend only on the prefill chunk (forward_batch,
    req_to_token, full_to_swa_index_mapping, and the c4/c128 page tables)
    and not on the per-layer k_cache. Reused across every layer in the
    chunk to avoid rebuilding tiny tensors 61 times per forward pass.
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
    query_start_loc: torch.Tensor  # (num_reqs+1,) int32

    # SWA-side (every layer needs these, all chunk-invariant).
    swa_token_ids: torch.Tensor  # (total_swa,) int32
    swa_first_pos: torch.Tensor  # (num_reqs,) int32
    swa_gather_lens: torch.Tensor  # (num_reqs,) int32
    swa_offsets: torch.Tensor  # (num_reqs+1,) int32

    # Optional per-query visible-window overrides (DSV4-VL image spans). The
    # existing union gather [seq_len - extend - (W-1), seq_len) already covers
    # every visible window, since a visible window [win_start, span_end) of a
    # fully-contained span satisfies win_start >= span_start - (W-1) >=
    # seq_len - extend - (W-1) and span_end <= seq_len.
    swa_win_starts: Optional[torch.Tensor] = None  # (num_qo_tokens,) int32
    swa_win_lens: Optional[torch.Tensor] = None  # (num_qo_tokens,) int32
    swa_win_max_len: Optional[int] = None

    # CPU copies of the per-request lengths (no device sync to plan groups).
    seq_lens_cpu: Optional[List[int]] = None
    extend_seq_lens_cpu: Optional[List[int]] = None
    swa_gather_lens_cpu: Optional[List[int]] = None
    # Workspace row budget per attention call (<= 0: one group, unbounded).
    workspace_budget_rows: int = 0

    # Request groups per ratio; each group is one workspace fill + one
    # attention call. None means one group over the whole chunk.
    c0_groups: Optional[List[SparsePrefillGroup]] = None
    c4_groups: Optional[List[SparsePrefillGroup]] = None
    c128_groups: Optional[List[SparsePrefillGroup]] = None

    # c0 pre-computed combine output (entire input set is chunk-invariant).
    c0_combined_indices: torch.Tensor = field(default=None)
    c0_combined_lens: torch.Tensor = field(default=None)
    # c128: packed positional layout of the c128 cache + pre-computed combine.
    c128_flat_token_ids: Optional[torch.Tensor] = None  # (sum seq_len // 128,) int32
    c128_combined_indices: Optional[torch.Tensor] = None
    c128_combined_lens: Optional[torch.Tensor] = None

    # c4: packed positional layout of the c4 cache (combine output is per-layer).
    # Request r owns rows [sum(seq_lens[:r] // 4), ... + seq_lens[r] // 4) of
    # its group, not r * (max_seq_len // 4): a padded layout made the
    # workspace num_reqs * max_seq_len // 4 rows, unbounded in num_reqs.
    c4_flat_token_ids: Optional[torch.Tensor] = None  # (sum seq_len // 4,) int32
    c4_page_size: Optional[int] = None
    c4_compressed_base: Optional[torch.Tensor] = None  # (num_reqs,) int32
    c4_swa_base: Optional[torch.Tensor] = None  # (num_reqs,) int32
    # Tail stays at the -1 sentinel because the valid prefix length is
    # chunk-invariant per request — subsequent layers only overwrite that prefix.
    c4_combined_indices: Optional[torch.Tensor] = None
    c4_combined_lens: Optional[torch.Tensor] = None

    @classmethod
    def build(
        cls,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        full_to_swa: torch.Tensor,
        swa_window_size: int,
        swa_page_size: int,
        num_qo_tokens: int,
        max_seq_len: int,
        total_swa: int,
        swa_win_starts: Optional[torch.Tensor] = None,
        swa_win_lens: Optional[torch.Tensor] = None,
        seq_lens_cpu: Optional[Sequence[int]] = None,
        extend_seq_lens_cpu: Optional[Sequence[int]] = None,
        workspace_budget_rows: int = 0,
    ) -> "SparsePrefillChunkCache":
        device = seq_lens.device
        num_reqs = seq_lens.shape[0]
        if seq_lens_cpu is None:
            seq_lens_cpu = seq_lens.tolist()
        if extend_seq_lens_cpu is None:
            extend_seq_lens_cpu = extend_seq_lens.tolist()
        seq_lens_cpu = [int(x) for x in seq_lens_cpu]
        extend_seq_lens_cpu = [int(x) for x in extend_seq_lens_cpu]
        assert len(seq_lens_cpu) == num_reqs == len(extend_seq_lens_cpu)
        swa_gather_lens_cpu = [
            min(seq_len, extend_len + swa_window_size - 1)
            for seq_len, extend_len in zip(seq_lens_cpu, extend_seq_lens_cpu)
        ]
        assert sum(swa_gather_lens_cpu) == total_swa, (
            sum(swa_gather_lens_cpu),
            total_swa,
        )

        query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        query_start_loc[1:] = torch.cumsum(extend_seq_lens, dim=0).to(torch.int32)

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

        swa_win_max_len = None
        if swa_win_lens is not None:
            assert swa_win_starts is not None
            # Computed once per chunk; reused by every layer's combine.
            swa_win_max_len = int(swa_win_lens.max().item())

        cache = cls(
            num_reqs=num_reqs,
            num_qo_tokens=num_qo_tokens,
            max_seq_len=max_seq_len,
            swa_window_size=swa_window_size,
            swa_page_size=swa_page_size,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            swa_token_ids=swa_token_ids,
            swa_first_pos=swa_first_pos,
            swa_gather_lens=swa_gather_lens,
            swa_offsets=swa_offsets,
            swa_win_starts=swa_win_starts,
            swa_win_lens=swa_win_lens,
            swa_win_max_len=swa_win_max_len,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            swa_gather_lens_cpu=swa_gather_lens_cpu,
            workspace_budget_rows=int(workspace_budget_rows),
        )

        # Pre-compute the c0 combine output: TOPK=0, compressed_base=0,
        # swa_base = swa_offsets[:-1] (one group). All inputs are chunk-invariant.
        zero_topk = torch.zeros((num_qo_tokens, 1), dtype=torch.int32, device=device)
        zero_compressed_base = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        cache.c0_groups = cache._plan_groups([0] * num_reqs)
        if len(cache.c0_groups) == 1:
            c0_swa_base = swa_offsets[:-1].to(torch.int32)
        else:
            _, c0_swa_base_cpu = sparse_prefill_group_bases(
                cache.c0_groups, [0] * num_reqs, swa_gather_lens_cpu
            )
            c0_swa_base = _int32_to_device(c0_swa_base_cpu, device)
        cache.c0_combined_indices, cache.c0_combined_lens = combine_topk_swa_indices(
            topk_indices=zero_topk,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            gather_lens=swa_gather_lens,
            compressed_base=zero_compressed_base,
            swa_base=c0_swa_base,
            window_size=swa_window_size,
            compress_ratio=1,
            topk=0,
            swa_win_starts=swa_win_starts,
            swa_win_lens=swa_win_lens,
            swa_win_max_len=swa_win_max_len,
        )
        return cache

    def ensure_c128(self, c128_page_indices: torch.Tensor) -> None:
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
        if self.c128_flat_token_ids is not None:
            return
        device = self.seq_lens.device
        c128_max = max(self.max_seq_len // 128, 1)
        assert c128_max <= c128_page_indices.shape[-1], (
            f"live c128 extent {c128_max} exceeds metadata capacity "
            f"{c128_page_indices.shape[-1]}"
        )
        last_q_per_req = (self.query_start_loc[1:] - 1).long()
        per_req_c128 = c128_page_indices.narrow(1, 0, c128_max).index_select(
            0, last_q_per_req
        )
        # Packed: request r keeps only its own seq_len // 128 entries, all
        # valid in its last query's row. A query reads j < (pos + 1) // 128
        # <= seq_len // 128, so it sees the same slots as in a padded layout.
        # Clamp -1 -> 0 so dequant doesn't OOB.
        counts_cpu = self._compressed_counts(128)
        self.c128_groups = self._plan_groups(counts_cpu)
        flat_c128_ids, compressed_base, swa_base = self._packed_layout(
            per_req_c128, counts_cpu, self.c128_groups, page_size=None
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
            seq_lens=self.seq_lens,
            gather_lens=self.swa_gather_lens,
            compressed_base=compressed_base,
            swa_base=swa_base,
            window_size=self.swa_window_size,
            compress_ratio=128,
            topk=c128_max,
            swa_win_starts=self.swa_win_starts,
            swa_win_lens=self.swa_win_lens,
            swa_win_max_len=self.swa_win_max_len,
        )

        self.c128_flat_token_ids = flat_c128_ids
        self.c128_combined_indices = combined_indices
        self.c128_combined_lens = combined_lens

    def ensure_c4(
        self,
        page_table: torch.Tensor,
        c4_page_size: int,
    ) -> None:
        """Populate c4-side fields from the per-query page table.

        ``page_table`` is (num_qo_tokens, max_blocks); rows within a request
        are duplicates. The combine output is per-layer (depends on the
        layer's remapped topk_indices), so we only cache the gather-side
        scaffolding plus compressed/swa bases.
        """
        if self.c4_flat_token_ids is not None:
            return
        c4_max = max(self.max_seq_len // 4, 1)
        c4_capacity = page_table.shape[-1] * c4_page_size
        assert (
            c4_max <= c4_capacity
        ), f"live c4 extent {c4_max} exceeds metadata capacity {c4_capacity}"
        first_q_per_req = self.query_start_loc[:-1].long()
        num_blocks = (c4_max + c4_page_size - 1) // c4_page_size
        assert num_blocks <= page_table.shape[1]
        per_req_page_table = page_table.narrow(1, 0, num_blocks).index_select(
            0, first_q_per_req
        )

        # Packed: request r keeps only its own seq_len // 4 entries. The C4
        # indexer emits min((pos + 1) // 4, topk) positions below
        # (pos + 1) // 4 <= seq_len // 4, so every read stays in r's rows.
        counts_cpu = self._compressed_counts(4)
        self.c4_groups = self._plan_groups(counts_cpu)
        flat_c4_ids, compressed_base, swa_base = self._packed_layout(
            per_req_page_table,
            counts_cpu,
            self.c4_groups,
            page_size=c4_page_size,
        )

        self.c4_flat_token_ids = flat_c4_ids
        self.c4_page_size = c4_page_size
        self.c4_compressed_base = compressed_base
        self.c4_swa_base = swa_base

    def _compressed_counts(self, ratio: int) -> List[int]:
        if self.seq_lens_cpu is None:
            # A cache built without the CPU lengths (direct construction, not
            # build()): read the device lengths once.
            self.seq_lens_cpu = [int(x) for x in self.seq_lens.tolist()]
        return [seq_len // ratio for seq_len in self.seq_lens_cpu]

    def _plan_groups(
        self, compressed_counts: Sequence[int]
    ) -> Optional[List[SparsePrefillGroup]]:
        if self.swa_gather_lens_cpu is None or self.extend_seq_lens_cpu is None:
            return None  # no CPU lengths to plan with: one group (groups_for)
        return plan_sparse_prefill_groups(
            compressed_counts,
            self.swa_gather_lens_cpu,
            self.extend_seq_lens_cpu,
            self.num_qo_tokens,
            self.workspace_budget_rows,
        )

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
            text += f", sum_seq_div4={sum(s // 4 for s in self.seq_lens_cpu)}"
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
        groups = {0: self.c0_groups, 4: self.c4_groups, 128: self.c128_groups}[
            compress_ratio
        ]
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

    def combine_c4_layer(
        self,
        c4_sparse_raw_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-layer combine for c4. ``c4_sparse_raw_indices`` is the topk
        kernel's positional output (``block_in_seq * c_page_size + in_page``)
        — already in the request-local workspace coordinate that
        ``combine_topk_swa_indices`` expects, so no remap is needed.

        Reuses preallocated ``c4_combined_indices`` / ``c4_combined_lens``
        buffers across layers — the kernel only overwrites the valid prefix.
        """
        topk = c4_sparse_raw_indices.shape[-1]
        if self.c4_combined_indices is None:
            device = self.seq_lens.device
            self.c4_combined_indices = torch.full(
                (
                    self.num_qo_tokens,
                    combined_topk_width(topk, self.eff_swa_window_size),
                ),
                -1,
                dtype=torch.int32,
                device=device,
            )
            self.c4_combined_lens = torch.zeros(
                self.num_qo_tokens, dtype=torch.int32, device=device
            )
        return combine_topk_swa_indices(
            topk_indices=c4_sparse_raw_indices,
            query_start_loc=self.query_start_loc,
            seq_lens=self.seq_lens,
            gather_lens=self.swa_gather_lens,
            compressed_base=self.c4_compressed_base,
            swa_base=self.c4_swa_base,
            window_size=self.swa_window_size,
            compress_ratio=4,
            topk=topk,
            out_indices=self.c4_combined_indices,
            out_lens=self.c4_combined_lens,
            swa_win_starts=self.swa_win_starts,
            swa_win_lens=self.swa_win_lens,
            swa_win_max_len=self.swa_win_max_len,
        )
