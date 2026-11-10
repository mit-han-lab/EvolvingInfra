# Adapted from SGLang python/sglang/srt/managers/scheduler_components/load_inquirer.py (commit 9a489f8d2f), modified for evolving-infra.
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from decoupled_engine.srt.disaggregation.utils import DisaggregationMode
from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.managers.load_snapshot import (
    DisaggregationMetrics,
    LateDLoadMetrics,
    LoadSnapshot,
    LoRAMetrics,
    MemoryMetrics,
    QueueMetrics,
    SpeculativeMetrics,
    SymmetricLoadMetrics,
)
from decoupled_engine.srt.runtime_context import get_lora

if TYPE_CHECKING:
    from decoupled_engine.srt.distributed.parallel_state_wrapper import ParallelState
    from decoupled_engine.srt.managers.scheduler_components.pool_stats_observer import (
        SchedulerPoolStatsObserver,
    )
    from decoupled_engine.srt.managers.tp_worker import BaseTpWorker
    from decoupled_engine.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from decoupled_engine.srt.server_args import ServerArgs
    from decoupled_engine.srt.speculative.spec_info import SpeculativeAlgorithm


logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerLoadInquirer:
    disaggregation_mode: DisaggregationMode
    ps: ParallelState
    server_args: ServerArgs
    max_total_num_tokens: int
    max_running_requests: int
    pool_stats_observer: SchedulerPoolStatsObserver
    tp_worker: BaseTpWorker
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    spec_algorithm: SpeculativeAlgorithm
    get_running_batch: Callable
    get_waiting_queue: Callable
    waiting_queue_prefix_matched: Callable
    get_recent_cache_hit_rate: Callable
    get_stats: Callable
    get_chunked_req: Callable
    get_disagg_prefill_bootstrap_queue: Callable
    get_disagg_prefill_inflight_queue: Callable
    get_disagg_decode_prealloc_queue: Callable
    get_disagg_decode_transfer_queue: Callable
    get_spec_total_num_accept_tokens: Callable
    get_spec_total_num_forward_ct: Callable
    get_total_prefill_uncached_tokens: Callable
    get_total_prefill_busy_us: Callable
    get_decode_moment_totals: Callable
    get_prefill_work: Callable | None = None
    get_prefill_routing: Callable | None = None
    get_retraction_totals: Callable | None = None
    # Symmetric mode (symmetric P/D): the ImportQueue and the ExportTable.
    get_import_queue: Callable | None = None
    get_export_table: Callable | None = None

    def _get_num_pending_tokens(self, chunk_deduct: int = 0) -> int:
        """Get the total number of tokens pending prefill.

        This includes tokens from waiting queue requests plus remaining tokens
        from the currently chunked request.

        Args:
            chunk_deduct: extra tokens to subtract from the chunked request's
                remaining count. At batch-scheduling time the current chunk
                has been planned but ``prefix_indices`` does not yet include it,
                so callers pass ``extend_input_len`` here. At load-reporting
                time ``prefix_indices`` is already up-to-date, so the default
                0 is correct.
        """
        num_pending_tokens = sum(req.seqlen for req in self.get_waiting_queue())
        if self.get_chunked_req() is not None:
            req = self.get_chunked_req()
            num_pending_tokens += req.seqlen - len(req.prefix_indices) - chunk_deduct
        return num_pending_tokens

    def get_num_waiting_uncached_tokens(self) -> int:
        """Estimate input tokens waiting for prefill compute."""
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            return 0
        waiting_queue_prefix_matched = self.waiting_queue_prefix_matched()
        cache_miss_rate = 1.0 - self.get_recent_cache_hit_rate()
        num_tokens = 0
        for req in self.get_waiting_queue():
            if waiting_queue_prefix_matched:
                num_tokens += max(0, req.seqlen - req.num_matched_prefix_tokens)
            else:
                num_tokens += int(req.seqlen * cache_miss_rate)
        cr = self.get_chunked_req()
        if cr is not None:
            num_tokens += max(0, cr.seqlen - len(cr.prefix_indices))
        return num_tokens + _held_uncached_tokens(self, cache_miss_rate)

    def get_loads(self) -> LoadSnapshot:
        """Build the per-DP-rank load snapshot for DP balancing and /v1/loads."""
        stats = self.get_stats()
        num_running_reqs = len(self.get_running_batch().reqs)

        waiting_queues = [self.get_waiting_queue()]
        pending_token_queues = [self.get_waiting_queue()]
        awaiting_kv_tokens = 0
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            prefill_bootstrap_queue = self.get_disagg_prefill_bootstrap_queue().queue
            waiting_queues.append(prefill_bootstrap_queue)
            pending_token_queues.append(prefill_bootstrap_queue)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            decode_prealloc_queue = self.get_disagg_decode_prealloc_queue().queue
            decode_transfer_queue = self.get_disagg_decode_transfer_queue().queue
            decode_retracted_queue = (
                self.get_disagg_decode_prealloc_queue().retracted_queue
            )
            waiting_queues.append(decode_prealloc_queue)
            waiting_queues.append(decode_transfer_queue)
            waiting_queues.append(decode_retracted_queue)
            # In disaggregated decode, transfer-queue requests and transferred
            # waiting-queue requests have already pre-allocated decode-side KV
            # slots, so they are already included in num_used_tokens.
            pending_token_queues = [decode_prealloc_queue, decode_retracted_queue]
            # KV not yet arrived from the prefill side.
            awaiting_kv_tokens = sum(
                req.seqlen
                for queue in (decode_prealloc_queue, decode_transfer_queue)
                for req in queue
            )

        num_waiting_reqs = sum(len(queue) for queue in waiting_queues)
        num_used_tokens, kv_token_usage = (
            self.pool_stats_observer.get_pool_stats().get_kv_token_stats()
        )
        num_total_tokens = num_used_tokens + sum(
            req.seqlen for queue in pending_token_queues for req in queue
        )
        num_active_tokens = max(0, num_total_tokens - awaiting_kv_tokens)

        memory = None
        try:
            memory = MemoryMetrics(
                weight_gb=round(self.tp_worker.model_runner.weight_load_mem_usage, 3),
                kv_cache_gb=round(
                    self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 3
                ),
                graph_gb=round(sum(self.tp_worker.graph_memory_usage.values()), 3),
                token_capacity=int(self.max_total_num_tokens),
            )
        except (AttributeError, TypeError) as e:
            logger.debug(f"Memory metrics not available: {e}")

        speculative = None
        if (
            not self.spec_algorithm.is_none()
            and self.get_spec_total_num_forward_ct() > 0
        ):
            speculative = SpeculativeMetrics(
                accept_length=(
                    self.get_spec_total_num_accept_tokens()
                    / self.get_spec_total_num_forward_ct()
                ),
                accept_rate=stats.spec_accept_rate,
            )

        lora = None
        if get_lora().enable_lora:
            lora = LoRAMetrics(
                slots_used=stats.lora_pool_slots_used,
                slots_total=stats.lora_pool_slots_total,
                utilization=stats.lora_pool_utilization,
            )

        mode_str = "null"
        prefill_bootstrap = prefill_inflight = 0
        decode_prealloc = decode_transfer = decode_retracted = 0
        decode_prealloc_ready = 0
        retractions = retract_aborts = retract_cap_aborts = 0
        late_d = None
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            mode_str = "prefill"
            prefill_bootstrap = len(self.get_disagg_prefill_bootstrap_queue().queue)
            prefill_inflight = len(self.get_disagg_prefill_inflight_queue())
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            mode_str = "decode"
            decode_prealloc = len(self.get_disagg_decode_prealloc_queue().queue)
            decode_transfer = len(self.get_disagg_decode_transfer_queue().queue)
            decode_retracted = len(
                self.get_disagg_decode_prealloc_queue().retracted_queue
            )
            decode_prealloc_ready = sum(
                1
                for decode_req in self.get_disagg_decode_prealloc_queue().queue
                if decode_req.waiting_for_input
            )
            if self.get_retraction_totals is not None:
                totals = self.get_retraction_totals()
                retractions, retract_aborts, retract_cap_aborts = totals
            prealloc_queue = self.get_disagg_decode_prealloc_queue()
            if prealloc_queue.late_d is not None:
                deferred_reqs, deferred_tokens = prealloc_queue.late_d.deferred_load(
                    prealloc_queue.queue
                )
                late_d = LateDLoadMetrics(
                    deferred_reqs=deferred_reqs,
                    deferred_tokens=deferred_tokens,
                    bound_total_tokens=max(0, num_total_tokens - deferred_tokens),
                )
        if (self.disaggregation_mode == DisaggregationMode.DECODE
                and envs.SGLANG_DECODE_SELECTION_METRICS.get()):
            from decoupled_engine.srt.disaggregation.decode_selection import selection_load
            if late_d is None:
                late_d = LateDLoadMetrics(bound_total_tokens=num_total_tokens)
            for key, value in selection_load(self.get_disagg_decode_prealloc_queue()).items():
                setattr(late_d, key, value)
        disaggregation = DisaggregationMetrics(
            mode=mode_str,
            prefill_bootstrap_queue_reqs=prefill_bootstrap,
            prefill_inflight_queue_reqs=prefill_inflight,
            decode_prealloc_queue_reqs=decode_prealloc,
            decode_transfer_queue_reqs=decode_transfer,
            decode_retracted_queue_reqs=decode_retracted,
            kv_transfer_speed_gb_s=stats.kv_transfer_speed_gb_s,
            kv_transfer_latency_ms=stats.kv_transfer_latency_ms,
            decode_retractions_total=retractions,
            decode_retract_aborts_total=retract_aborts,
            decode_retract_cap_aborts_total=retract_cap_aborts,
        )

        queues = QueueMetrics(
            waiting=len(self.get_waiting_queue()),
            grammar=stats.num_grammar_queue_reqs,
            paused=stats.num_paused_reqs,
            retracted=stats.num_retracted_reqs,
            prealloc_ready=decode_prealloc_ready,
        )

        totals = self.get_decode_moment_totals()
        decode_moments = list(totals) if totals[0] > 0 else None
        if late_d is None:
            late_d = _sym_late_d(self, num_total_tokens)
        if (self.disaggregation_mode == DisaggregationMode.SYMMETRIC
                and envs.SGLANG_DECODE_SELECTION_METRICS.get()):
            late_d = _sym_selection_late_d(self, late_d, num_total_tokens)

        return LoadSnapshot(
            dp_rank=int(self.ps.dp_rank) if self.ps.dp_rank is not None else 0,
            timestamp=time.time(),
            num_running_reqs=num_running_reqs,
            num_waiting_reqs=num_waiting_reqs,
            num_waiting_uncached_tokens=self.get_num_waiting_uncached_tokens(),
            num_used_tokens=num_used_tokens,
            num_total_tokens=num_total_tokens,
            num_active_tokens=num_active_tokens,
            max_total_num_tokens=self.max_total_num_tokens,
            max_running_requests=self.max_running_requests,
            token_usage=round(kv_token_usage, 4),
            gen_throughput=round(stats.gen_throughput, 2),
            cache_hit_rate=round(stats.cache_hit_rate, 4),
            utilization=round(stats.utilization, 4),
            memory=memory,
            speculative=speculative,
            lora=lora,
            disaggregation=disaggregation,
            queues=queues,
            total_prefill_uncached_tokens=self.get_total_prefill_uncached_tokens(),
            total_prefill_busy_us=self.get_total_prefill_busy_us(),
            decode_moments=decode_moments,
            prefill_work=self.get_prefill_work() if self.get_prefill_work else None,
            prefill_routing=(
                self.get_prefill_routing() if self.get_prefill_routing else None
            ),
            late_d=late_d,
            symmetric=_symmetric_load(self),
        )


# Symmetric mode (symmetric P/D). Module functions: they read the symmetric fields
# only on a symmetric inquirer.
def _export_table(inquirer: SchedulerLoadInquirer):
    if inquirer.disaggregation_mode != DisaggregationMode.SYMMETRIC:
        return None
    get = inquirer.get_export_table
    return get() if get is not None else None


def _held_uncached_tokens(inquirer: SchedulerLoadInquirer, miss_rate: float) -> int:
    """Late D: a P-leg held for its reader is in no
    queue, so the router's backlog would lose it after one poll. Count it at
    prompt x recent cache-miss rate (no prefix match).
    SGLANG_SYMMETRIC_PD_LOADS_COUNT_HELD; unset: on with late D."""
    table = _export_table(inquirer)
    if table is None:
        return 0
    count_held = envs.SGLANG_SYMMETRIC_PD_LOADS_COUNT_HELD.get()
    if not (table.late_d if count_held is None else count_held):
        return 0
    return sum(int(req.seqlen * miss_rate) for req in table.held_reqs())


def _sym_late_d(
    inquirer: SchedulerLoadInquirer, num_total_tokens: int
) -> LateDLoadMetrics | None:
    """Late D on a symmetric worker: DEFERRED
    imports, and the bound view = num_total_tokens + held P-legs + bound
    imports (ADMITTED, HELD, not late). Held P-legs and bound imports are
    certain work here that num_total_tokens leaves out."""
    table = _export_table(inquirer)
    sym = _symmetric_load(inquirer) if table is not None and table.late_d else None
    if sym is None:
        return None
    return LateDLoadMetrics(
        deferred_reqs=sym.import_deferred_reqs,
        deferred_tokens=sym.import_deferred_tokens,
        bound_total_tokens=(
            num_total_tokens + sym.export_held_tokens + sym.import_bound_tokens
        ),
    )


def _sym_selection_late_d(
    inquirer: SchedulerLoadInquirer,
    late_d: LateDLoadMetrics | None,
    num_total_tokens: int,
) -> LateDLoadMetrics | None:
    """SGLANG_DECODE_SELECTION_METRICS on a symmetric worker: the native
    decode-capacity fields of a decode worker (decode_selection.selection_load),
    from this worker's ImportQueue and pool (symmetric_selection_load), so the
    router's late-D fit_req reads the same view from every D-role worker."""
    table = _export_table(inquirer)
    get_queue = inquirer.get_import_queue if table is not None else None
    queue = get_queue() if get_queue is not None else None
    if queue is None:
        return late_d
    from decoupled_engine.srt.disaggregation.decode_selection import symmetric_selection_load

    if late_d is None:
        late_d = LateDLoadMetrics(bound_total_tokens=num_total_tokens)
    fields = symmetric_selection_load(
        queue,
        running_reqs=inquirer.get_running_batch().reqs,
        waiting_queue=inquirer.get_waiting_queue(),
        chunked_req=inquirer.get_chunked_req(),
        export_table=table,
    )
    for key, value in fields.items():
        setattr(late_d, key, value)
    return late_d


def _symmetric_load(inquirer: SchedulerLoadInquirer) -> SymmetricLoadMetrics | None:
    table = _export_table(inquirer)
    get_queue = inquirer.get_import_queue if table is not None else None
    queue = get_queue() if get_queue is not None else None
    if table is None or queue is None:
        return None
    return SymmetricLoadMetrics(
        **queue.load_view(), **table.load_view(inquirer.get_waiting_queue())
    )
