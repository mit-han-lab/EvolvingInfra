"""SGLANG_SYMMETRIC_PUBLISH_ASYNC_READBACK: the owner's publication readbacks
without a wait for the forward in flight and without SM kernels.

Problem (measured on Kimi-K2.6 6P3D symmetric workers with phase timers on).
In the overlap loop the WAR barrier fences ``schedule_stream`` behind forward k
(``Scheduler._apply_war_barrier``; the EAGLE prefill has no early shared-read
event, so the barrier waits for the whole forward). Then result k-1 is
processed and published (SGLANG_SYMMETRIC_PUBLISH_AFTER_RESULT ->
``ExportTable.publish_computed`` -> ``_make_chunk`` -> ``pool_pages_for_range``,
then ``OwnerPlane.publish_chunk`` / ``publish_final``). Four host reads of device
tensors on that fenced stream wait until forward k ends:

1. ``kv_to_page_indices`` (``mem_cache/common.py``): ``.cpu()`` of the row slice
   (S3_kv_to_page_indices, 1,110-1,194 ms per pass: the first one pays);
2. ``page_generations`` -> ``PagedTokenToKVPoolAllocator.generation_of``
   (an H2D ``torch.as_tensor`` with a stream sync, then ``.tolist()``);
3. ``GpuLeaseTable.pin`` (CHUNK_READY) -> ``generation_of`` again;
4. ``GpuLeaseTable.current_generations`` (FINAL) -> ``generation_of`` again.

Only after them does the thread receive, admit and prepare k+1, with the GPU
idle (6.7-7.7 % of P busy; computer-use passes with an image launch inline, not
on the launch thread). SGLANG_SYMMETRIC_PUBLISH_UNFENCED moves 1 and 2 to a
side stream, but 3 and 4 then pay the same wait.

What this reader does. Publication keeps its program point, its order and its
values; only the reads change.

- ``mark(schedule_stream)``: the loop records an event E on ``schedule_stream``
  right before each WAR barrier and before a threaded launch (the points of
  the UNFENCED mark), together with the request pool's row-write sequence
  number.
- ``read(...)`` (1 and 2, from ``ExportTable._make_chunk``): a side stream
  waits for E and copies the chunk's ``req_to_token`` row slice (contiguous
  int32) into a pinned host buffer; the host waits for that copy only (one
  event). Page ids are ``ids[::page_size] // page_size`` on the host, the
  arithmetic of ``kv_to_page_indices``. A second copy brings the span of the
  allocator's page-generation counters that holds those pages. Only
  copy-engine DMAs run, so a read does not queue for SMs behind the forward.
  The pinned buffers are sized once at install (a request row; the counters).
- ``wrap_generation_of(inner)`` (3 and 4): the owner plane's Lease calls it on
  the scheduler thread; FULL-pool pages get the same span copy after E. Other
  threads, pools and shapes call ``inner``.
- Row guard (the 6b rule, ``prefill_loop_unserialize.KvPageIdReader``): a
  request row written through the pool's ``write`` after E's snapshot takes
  the legacy read on the current stream, which is ordered after that write.
  The symmetric worker's pool is ``DecodeReqToTokenPool`` (decode role); it and
  ``ReqToTokenPool`` keep the same row-write sequence.
- A shape this reader does not cover returns None, and ``_make_chunk`` runs
  the flag-off path: SWA pools, DSv4 pools, M4 Q4 CPU rows (already
  sync-free), virtual-id pools (``translate_kv_indices_for_transfer``
  overridden), allocators other than the plain paged one, a request pool
  without the row-write sequence, another thread, no mark yet.

Why the values equal the legacy reads: two strict invariants.

R (rows). Every row entry of a published range [start, end) was written before
the newest mark E, in stream order, and is not written with another value
afterwards while the range can still be published.
- A published range ends at or below ``computed_end`` (or the classic path's
  ``tmp_end_idx``), which moves only when a result is processed, so its tokens
  belong to batches launched before E. Admission wrote those entries on
  ``schedule_stream`` before that launch.
- An exporting request is never inserted into the tree while it exports
  (``symmetric_stash_chunked_request``, ``symmetric_on_prefill_chunk``), so its
  slots are never deduplicated and repointed. Writes through the pool's
  ``write`` after E are caught by the row guard.
- Kernel writers bypass the guard: ``write_req_to_token_pool_triton``
  (``alloc_for_extend``) and ``assign_req_to_token_pool``
  (``alloc_for_spec_decode``). In the usual order they run in admission, before
  the launch and so before E. On a disable-overlap turn and on the
  gate-pending reorder path the previous result is processed after the next
  batch's admission and before its mark, so those kernel writes can be
  unordered with the side copy. They still never change a published entry:
  the next chunk of an exporting request rewrites [0, prefix_len) with
  ``prefix_indices``, which the stash copied from the same row (equal values),
  and writes new slots at [prefix_len, seq_len), at or beyond ``computed_end``;
  ``alloc_for_spec_decode`` writes decoding requests at or beyond their current
  length, never a P-leg's published range.

G (generations). No page of a published range, and no page pinned in the Lease
for a live export, is freed (generation bumped) after the newest mark E and
before its CHUNK_READY / FINAL reads. The side copy is unordered with
``schedule_stream`` work enqueued after E, so a violation would be a per-rank
race: one TP rank could read the bump and another not. ``GpuLeaseTable.pin``
could then refuse on one rank only (``lease_refused``), or FINAL's
``current_generations`` could miss an eviction. G holds by construction: a
published page is held by its export from allocation until the reader's
release (request-held, never in the tree), and its last bump came from the
previous owner's free, which preceded the allocation in stream order. The CHECK
smoke must show zero "check mismatch" ERROR lines on every rank.

KV. Unchanged: the result's ``copy_done`` was synchronized before the
publication (``process_batch_result_prefill``), as with the flag off.

TP ranks. No decision moves: every rank publishes the same chunks at the same
program point. The guard reads lockstep state (row writes), so every rank
takes the same path, and both paths give the same values under R and G. No
collective.

Liveness. No thread and no deferred work: nothing to drain or stop. The only
host wait is for the side-stream copies, which depend on E and on the copy
engine. If the side stream shared an in-order hardware queue
(CUDA_DEVICE_MAX_CONNECTIONS=8, round-robin pool streams) with
``forward_stream`` or with ``schedule_stream`` (which holds the WAR-barrier
wait at every publication), its copies would queue behind the forward again.
``draw_readback_stream`` runs the loader's I9 self-test
(``mem_cache/ce_loader.overlap_self_test``) with the sleep on each of the two
and redraws. ``schedule_overlaps_forward`` reports whether the engine's own
schedule and forward streams share a queue (then E itself completes only after
the forward: reads stay correct, without gain). Errors propagate (fail-stop).

SGLANG_SYMMETRIC_PUBLISH_ASYNC_READBACK_CHECK (smoke only): also run the
legacy reads, which wait for the forward, and compare; a mismatch is logged as
an ERROR on its rank and the legacy value is used.

Log (every TP rank, about every 60 s once it reads, counters since start):

    SYMMETRIC_PUBLISH_ASYNC_READBACK rank=R reads=N lease_reads=N
        legacy_rewritten=N legacy_out_of_range=N unsupported={reason: N}
        lease_inner={reason: N} gate_pending=N wait_ms_total=X wait_ms_max=Y
        slow_reads=N check_ok=N check_mismatch=N seed_async=N seed_blocking=N

``reads`` / ``lease_reads`` count the copy-engine reads (chunk pages, Lease
generations); ``unsupported`` / ``lease_inner`` the calls left to the flag-off
code, by reason (``no_mark``: before the first launch); ``gate_pending`` the
reads that found the mark not reached yet; ``wait_ms_*`` the host wait of the
reads and ``slow_reads`` those above SLOW_READ_MS (a wait near the forward time
means the side stream is serialized behind the forward); ``seed_*`` the 6a-sym
EAGLE seed reads of FINAL (batch_result_processor ``_SYM_SEED_STATS``;
``seed_blocking`` > 0 is a device read behind the barrier). The smoke gate
reads ``wait_ms_max`` and ``slow_reads`` on every rank.
"""

from __future__ import annotations

import collections
import logging
import sys
import threading
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Reads whose host wait exceeds this count as slow.
SLOW_READ_MS = 5.0
LOG_EVERY_S = 60.0
_SEED_STATS_MODULE = "decoupled_engine.srt.managers.scheduler_components.batch_result_processor"


def unsupported_reason(req_to_token_pool: Any, allocator: Any) -> Optional[str]:
    """Why ``ExportPageReader`` cannot serve this pool pair (None: it can).
    Checked per call; a few attribute reads."""
    import torch

    from decoupled_engine.srt.disaggregation.symmetric.export_table import _is_dsv4_pool
    from decoupled_engine.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
    from decoupled_engine.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    from decoupled_engine.srt.mem_cache.cpu_page_book import cpu_rows_of

    if cpu_rows_of(req_to_token_pool) is not None:
        return "q4_cpu_rows"
    rows = getattr(req_to_token_pool, "req_to_token", None)
    if not isinstance(rows, torch.Tensor) or rows.dim() != 2:
        return "rows"
    if not getattr(req_to_token_pool, "_track_row_writes", False) or not callable(
        getattr(req_to_token_pool, "row_written_after", None)
    ):
        # No row-write sequence: the row guard cannot work.
        return "rows"
    if hasattr(allocator, "swa_attn_allocator"):
        return "swa_pool"
    get_kvcache = getattr(allocator, "get_kvcache", None)
    if callable(get_kvcache) and _is_dsv4_pool(get_kvcache()):
        return "dsv4_pool"
    if (
        type(allocator).translate_kv_indices_for_transfer
        is not BaseTokenToKVPoolAllocator.translate_kv_indices_for_transfer
    ):
        return "virtual_ids"
    if (
        not isinstance(allocator, PagedTokenToKVPoolAllocator)
        or type(allocator).generation_of
        is not PagedTokenToKVPoolAllocator.generation_of
        or not isinstance(getattr(allocator, "_page_gen", None), torch.Tensor)
    ):
        return "generations"
    return None


def draw_readback_stream(
    device_module: Any,
    forward_stream: Any,
    schedule_stream: Any,
    *,
    self_test: Optional[Callable[..., Tuple[bool, str]]] = None,
    max_redraws: int = 64,
) -> Tuple[Any, bool, int, str]:
    """A side stream for the readbacks. It must not be ``forward_stream`` or
    ``schedule_stream`` and (``self_test``, the loader's I9 check, run with the
    sleep on each) must not share an in-order hardware queue with either: the
    forward stream runs the forward in flight, and the schedule stream holds
    the WAR-barrier wait for it at every publication. Redraws up to
    ``max_redraws`` times, as the loader does with
    SGLANG_DSA_HOST_DEDUP_STREAM_REDRAW. Returns (stream, ok, redraws, why); a
    stream that still fails is returned anyway (its reads stay correct and may
    wait for the forward, as with the flag off)."""
    sleepers = (("forward_stream", forward_stream), ("schedule_stream", schedule_stream))

    def check(stream) -> Tuple[bool, str, bool]:
        handle = getattr(stream, "cuda_stream", None)
        if handle is not None:
            for name, other in sleepers:
                if handle == getattr(other, "cuda_stream", None):
                    return False, f"aliases {name}", True
        if self_test is None:
            return True, "no overlap self-test (not CUDA)", False
        whys = []
        for name, other in sleepers:
            try:
                ok, why = self_test(device_module, other, [stream])
            except Exception as e:  # noqa: BLE001 - reported, never retried
                return False, f"self-test error vs {name}: {e!r}", False
            if not ok:
                return False, f"shares a hardware queue with {name} ({why})", True
            whys.append(f"vs {name}: {why}")
        return True, "; ".join(whys), True

    stream = device_module.Stream()
    ok, why, retryable = check(stream)
    redraws = 0
    while not ok and retryable and redraws < max_redraws:
        stream = device_module.Stream()
        redraws += 1
        ok, why, retryable = check(stream)
    return stream, ok, redraws, why


def schedule_overlaps_forward(
    device_module: Any,
    forward_stream: Any,
    schedule_stream: Any,
    *,
    self_test: Optional[Callable[..., Tuple[bool, str]]] = None,
) -> Tuple[Optional[bool], str]:
    """Diagnostic for the start line: True when a copy on ``schedule_stream``
    ends while ``forward_stream`` sleeps (separate hardware queues). False:
    they share a queue, so the mark E completes only after the forward in
    flight and every read waits for it (correct, no gain). None: not checked."""
    if self_test is None:
        return None, "no overlap self-test (not CUDA)"
    handle = getattr(schedule_stream, "cuda_stream", None)
    if handle is not None and handle == getattr(forward_stream, "cuda_stream", None):
        return False, "schedule_stream aliases forward_stream"
    try:
        ok, why = self_test(device_module, forward_stream, [schedule_stream])
    except Exception as e:  # noqa: BLE001 - diagnostic only
        return None, f"self-test error: {e!r}"
    return bool(ok), why


class ExportPageReader:
    """Owner-side page ids and generations of a published chunk, read with
    copy-engine DMAs on ``stream`` after the newest pre-WAR mark (module
    docstring). One per scheduler; its reads run on the thread that created
    it (the scheduler thread)."""

    def __init__(
        self,
        *,
        device_module: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        stream: Any,
        check: bool = False,
        tp_rank: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.dm = device_module
        self.pool = req_to_token_pool
        self.allocator = token_to_kv_pool_allocator
        self.stream = stream
        self.check = bool(check)
        self.tp_rank = tp_rank
        self.clock = clock
        self.thread_ident = threading.get_ident()
        self.event = None
        self.event_seq = 0
        if not getattr(req_to_token_pool, "_track_row_writes", False):
            # The 6b row-write sequence (ReqToTokenPool, DecodeReqToTokenPool).
            # Kept when another reader enabled it already: a reset would move
            # the sequence numbers that reader snapshotted. A pool without it
            # stays untracked, and every read takes the flag-off path ("rows").
            enable = getattr(req_to_token_pool, "enable_row_write_tracking", None)
            if callable(enable):
                enable()
        self._ids_host = None
        self._gens_host = None
        self._presize_host_buffers()
        self.stats: collections.Counter = collections.Counter()
        self.wait_ms_total = 0.0
        self.wait_ms_max = 0.0
        self._last_log = clock()

    @property
    def rows_tracked(self) -> bool:
        return bool(getattr(self.pool, "_track_row_writes", False))

    # -- loop side --------------------------------------------------------
    def mark(self, schedule_stream: Any) -> None:
        """Right before a WAR barrier (and before a threaded launch): the
        schedule stream's position and the row-write sequence number there."""
        ev = self.dm.Event()
        ev.record(schedule_stream)
        self.event = ev
        seq = getattr(self.pool, "row_write_seq", None)
        self.event_seq = seq() if callable(seq) and self.rows_tracked else 0

    # -- publication side: ExportTable._make_chunk ---------------------------
    def read(
        self,
        *,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        req: Any,
        token_start: int,
        token_end: int,
        page_size: int,
        swa_skip_before: int = 0,
    ) -> Optional[List[Any]]:
        """``pool_pages_for_range`` for ``_make_chunk``. Returns the pools, or
        None when the caller must run the flag-off path."""
        from decoupled_engine.srt.disaggregation.pull.api import PoolPages
        from decoupled_engine.srt.disaggregation.symmetric.export_table import (
            POOL_FULL,
            pool_pages_for_range,
        )

        if token_end <= token_start:
            return []
        why = self._unsupported(req_to_token_pool, token_to_kv_pool_allocator)
        if why is not None:
            self.stats["unsupported_" + why] += 1
            return None
        legacy_kwargs = dict(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            req=req,
            token_start=token_start,
            token_end=token_end,
            page_size=page_size,
            swa_skip_before=swa_skip_before,
        )
        row = int(req.req_pool_idx)
        if req_to_token_pool.row_written_after(row, self.event_seq):
            # Written after the mark: only the current stream orders after it.
            self.stats["legacy_rewritten"] += 1
            return pool_pages_for_range(**legacy_kwargs)

        t0 = time.perf_counter()
        self._note_gate()
        src = req_to_token_pool.req_to_token[row, token_start:token_end]
        ids_host = self._host_buffer("_ids_host", src)
        self._copy_and_wait(ids_host, src)
        pages = np.floor_divide(ids_host.numpy()[::page_size], page_size)
        gens = self._generations(token_to_kv_pool_allocator, pages)
        if gens is None:
            # Not a valid page id: the flag-off read fails or succeeds as it
            # always did.
            self.stats["legacy_out_of_range"] += 1
            return pool_pages_for_range(**legacy_kwargs)
        self._note_wait(t0, "reads")
        pools = [PoolPages(pool=POOL_FULL, page_indices=pages.tolist(), generations=gens)]
        if self.check:
            ref = pool_pages_for_range(**legacy_kwargs)
            pools = self._checked(
                pools,
                ref,
                "rid=%s row=%d range=[%d,%d)"
                % (getattr(req, "rid", None), row, token_start, token_end),
            )
        self._maybe_log()
        return pools

    # -- publication side: the owner plane's Lease ------------------------
    def wrap_generation_of(
        self, inner: Callable[[str, Sequence[int]], List[int]]
    ) -> Callable[[str, Sequence[int]], List[int]]:
        """``GpuLeaseTable.wrap_generation_of`` hook: ``pin`` (CHUNK_READY) and
        ``current_generations`` (FINAL) on the scheduler thread read FULL-pool
        generations with one span copy after the newest mark; anything else
        calls ``inner`` (``export_table.make_generation_of``)."""

        def generation_of(pool: str, page_indices: Sequence[int]) -> List[int]:
            gens = self._lease_generations(pool, page_indices)
            if gens is None:
                return inner(pool, page_indices)
            if self.check:
                gens = self._checked(
                    gens, inner(pool, page_indices), "lease pool=%s" % pool
                )
            return gens

        return generation_of

    def _lease_generations(
        self, pool: str, page_indices: Sequence[int]
    ) -> Optional[List[int]]:
        from decoupled_engine.srt.disaggregation.symmetric.export_table import POOL_FULL

        if pool != POOL_FULL:
            self.stats["lease_inner_pool"] += 1
            return None
        why = self._unsupported(self.pool, self.allocator)
        if why is not None:
            self.stats["lease_inner_" + why] += 1
            return None
        pages = np.asarray([int(p) for p in page_indices], dtype=np.int64)
        if pages.size == 0:
            return []
        t0 = time.perf_counter()
        self._note_gate()
        gens = self._generations(self.allocator, pages)
        if gens is None:
            # -1 ("no page on this side") or an id outside the pool.
            self.stats["lease_inner_out_of_range"] += 1
            return None
        self._note_wait(t0, "lease_reads")
        self._maybe_log()
        return gens

    # -- helpers -----------------------------------------------------------
    def _unsupported(self, req_to_token_pool: Any, allocator: Any) -> Optional[str]:
        if threading.get_ident() != self.thread_ident:
            # The host buffers and the mark belong to the scheduler thread.
            return "other_thread"
        why = unsupported_reason(req_to_token_pool, allocator)
        if why is not None:
            return why
        if req_to_token_pool is not self.pool or allocator is not self.allocator:
            return "other_pool"
        if self.event is None:
            # No launch behind a barrier yet: nothing runs on the GPU that the
            # flag-off read could wait for.
            return "no_mark"
        return None

    def _generations(self, allocator: Any, pages: np.ndarray) -> Optional[List[int]]:
        """Generations of ``pages`` (valid page ids, at least one) from one copy
        of their span of ``allocator._page_gen``; None if one is out of range."""
        gen = allocator._page_gen
        lo, hi = int(pages.min()), int(pages.max())
        if lo < 0 or hi >= int(gen.shape[0]):
            return None
        span = gen[lo : hi + 1]
        gens_host = self._host_buffer("_gens_host", span)
        self._copy_and_wait(gens_host, span)
        return gens_host.numpy()[pages - lo].tolist()

    def _presize_host_buffers(self) -> None:
        """Size the host buffers once: a whole request row and all
        page-generation counters, so no read allocates (pinned when the source
        is on the GPU)."""
        import torch

        for name, src in (
            ("_ids_host", getattr(self.pool, "req_to_token", None)),
            ("_gens_host", getattr(self.allocator, "_page_gen", None)),
        ):
            if not isinstance(src, torch.Tensor) or src.dim() < 1:
                continue
            n = int(src.shape[-1]) if name == "_ids_host" else int(src.shape[0])
            setattr(
                self,
                name,
                torch.empty(n, dtype=src.dtype, pin_memory=bool(src.is_cuda)),
            )

    def _host_buffer(self, name: str, like: Any) -> Any:
        """A host buffer of ``like``'s dtype and length, pinned when ``like``
        is on the GPU. Sized at install; grown by powers of two only if a
        source outgrows it (every copy into it is waited for before the next
        read)."""
        import torch

        n = int(like.shape[0])
        buf = getattr(self, name)
        if buf is None or buf.dtype != like.dtype or buf.shape[0] < n:
            cap = 1 << max(10, (n - 1).bit_length())
            buf = torch.empty(cap, dtype=like.dtype, pin_memory=bool(like.is_cuda))
            setattr(self, name, buf)
            self.stats["buffer_grow"] += 1
        return buf[:n]

    def _copy_and_wait(self, dst: Any, src: Any) -> None:
        """One DMA on the side stream, ordered after the newest mark, then a
        host wait for it alone."""
        self.stream.wait_event(self.event)
        with self.dm.stream(self.stream):
            dst.copy_(src, non_blocking=True)
            done = self.dm.Event()
            done.record(self.stream)
        done.synchronize()

    def _note_gate(self) -> None:
        """Diagnostic only: the mark not reached yet (the copies wait for it)."""
        query = getattr(self.event, "query", None)
        if callable(query) and not query():
            self.stats["gate_pending"] += 1

    def _note_wait(self, t0: float, kind: str) -> None:
        wait_ms = (time.perf_counter() - t0) * 1e3
        self.stats[kind] += 1
        self.wait_ms_total += wait_ms
        self.wait_ms_max = max(self.wait_ms_max, wait_ms)
        if wait_ms > SLOW_READ_MS:
            self.stats["slow_reads"] += 1

    def _checked(self, got: Any, ref: Any, what: str) -> Any:
        """Check mode: the legacy value wins on a mismatch."""
        if got == ref:
            self.stats["check_ok"] += 1
            return got
        self.stats["check_mismatch"] += 1
        logger.error(
            "SYMMETRIC_PUBLISH_ASYNC_READBACK check mismatch rank=%d (%s): "
            "async=%s legacy=%s; using the legacy read",
            self.tp_rank,
            what,
            str(got)[:400],
            str(ref)[:400],
        )
        return ref

    def stats_line(self) -> str:
        def group(prefix: str) -> dict:
            return {
                k[len(prefix) :]: v
                for k, v in sorted(self.stats.items())
                if k.startswith(prefix)
            }

        # 6a-sym seed counters, read only if the module is loaded (no import).
        seed = getattr(sys.modules.get(_SEED_STATS_MODULE), "_SYM_SEED_STATS", {})
        return (
            "SYMMETRIC_PUBLISH_ASYNC_READBACK rank=%d reads=%d lease_reads=%d "
            "legacy_rewritten=%d legacy_out_of_range=%d unsupported=%s "
            "lease_inner=%s gate_pending=%d wait_ms_total=%.1f "
            "wait_ms_max=%.2f slow_reads=%d check_ok=%d check_mismatch=%d "
            "seed_async=%d seed_blocking=%d"
            % (
                self.tp_rank,
                self.stats["reads"],
                self.stats["lease_reads"],
                self.stats["legacy_rewritten"],
                self.stats["legacy_out_of_range"],
                group("unsupported_"),
                group("lease_inner_"),
                self.stats["gate_pending"],
                self.wait_ms_total,
                self.wait_ms_max,
                self.stats["slow_reads"],
                self.stats["check_ok"],
                self.stats["check_mismatch"],
                seed.get("async", 0),
                seed.get("blocking", 0),
            )
        )

    def _maybe_log(self) -> None:
        """Every rank: the smoke gate reads wait_ms_max and slow_reads per
        rank."""
        now = self.clock()
        if now - self._last_log < LOG_EVERY_S:
            return
        self._last_log = now
        logger.info(self.stats_line())
