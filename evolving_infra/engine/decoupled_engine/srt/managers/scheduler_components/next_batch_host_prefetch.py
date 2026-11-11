"""One-batch host lookahead without changing request admission or progress.

Prepare after the current batch owns its KV, but BEFORE launching its forward.
The existing load fence then covers older forwards, not this independent one.
Keep the normal WAR barrier; only the lookahead load omits the caller-stream
wait. At the next iteration, finish the load before allowing prefix matching.
"""

import logging
import os
import time

import torch

from decoupled_engine.srt.environ import envs

logger = logging.getLogger(__name__)


class NextBatchHostPrefetch:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.cache = scheduler.tree_cache
        self.pins = []
        self.completion = None
        self.prepared = None
        self.max_tokens = envs.SGLANG_HICACHE_PREFETCH_MAX_TOKENS.get()
        from decoupled_engine.srt.mem_cache.memory_pool import DSATokenToKVPool
        from decoupled_engine.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        if not (
            scheduler.enable_overlap
            and scheduler.enable_hierarchical_cache
            and isinstance(self.cache, UnifiedRadixCache)
            and self.cache.buffer_pipeline is None
            and isinstance(
                scheduler.token_to_kv_pool_allocator.get_kvcache(), DSATokenToKVPool
            )
            and not scheduler.is_hybrid_swa
            and not scheduler.is_hybrid_ssm
            and not scheduler.enable_lora
            and scheduler.ps.pp_size == 1
            and scheduler.ps.attn_cp_size == 1
            and scheduler.chunked_prefill_size
            and self.max_tokens > 0
        ):
            raise ValueError("Next-batch host prefetch requires overlap DSA HiCache/TP")
        # The NCCL dedup fan-out runs concurrently with the forward; the CUDA
        # IPC fan-out (SGLANG_DSA_HOST_DEDUP_FANOUT=ipc) issues no NCCL work.
        dedup = getattr(self.cache.cache_controller, "host_dedup", None)
        uses_ipc_fanout = getattr(dedup, "ipc_fanout", None) is not None
        if not uses_ipc_fanout and (
            os.environ.get("NCCL_LAUNCH_ORDER_IMPLICIT") != "1"
            or torch.cuda.nccl.version() < (2, 26, 0)
        ):
            raise ValueError("Next-batch host prefetch requires NCCL implicit ordering")

    def finish(self):
        if self.completion is None:
            return
        started = time.perf_counter()
        # A local query is insufficient: every TP rank must finish before the
        # normal collective loading_check publishes the GPU prefix as reusable.
        self.completion.finish_event.synchronize()
        self.cache.loading_check()
        logger.info(
            "Next-batch host prefetch ready: requests=%d tokens=%d tail_wait_ms=%.3f",
            len(self.pins),
            self.completion.num_tokens,
            (time.perf_counter() - started) * 1000,
        )
        self.completion = None

    def release(self, batch=None):
        """Normal admission now owns its locks; unchosen/canceled rows unpin too."""
        admitted = set(batch.reqs) if batch is not None else set()
        for req, node, lock in self.pins:
            self.cache.dec_lock_ref(node, lock)
            if req not in admitted:
                req.prefetched_host_hit_length = 0
        if self.pins:
            logger.info(
                "Next-batch host prefetch consumed: next_batch=%d prepared=%d",
                sum(req in admitted for req, _, _ in self.pins),
                len(self.pins),
            )
        self.pins.clear()

    def prepare(self, batch):
        s = self.scheduler
        cc = self.cache.cache_controller
        assert self.completion is None and self.prepared is None and not self.pins
        # Admission may have a queued failed/partial load; never mix it into
        # lookahead or change the event belonging to the current batch.
        if batch is None or cc.load_queue:
            return
        budget = min(s.chunked_prefill_size, s.max_prefill_tokens)
        if s.chunked_req is not None:
            req = s.chunked_req
            budget -= min(
                budget,
                len(req.full_untruncated_fill_ids) - req.extend_range.end,
            )
        # max_prefill_bs is a delayer statistic (normally zero), not an
        # admission limit. Request-pool capacity includes in-flight PD rows.
        slots = s.req_to_token_pool.available_size()
        max_reqs = s.server_args.prefill_max_requests
        if max_reqs is not None:
            slots = min(slots, max_reqs - int(s.chunked_req is not None))
        loaded = 0
        for req in s.waiting_queue:
            if budget <= 0 or slots <= 0:
                break
            if req.finished() or req.pending_bootstrap or req.beam_group is not None:
                continue
            if s.enable_hicache_storage:
                if not self.cache.check_prefetch_progress(req.rid):
                    continue
                storage_hit = self.cache.pop_prefetch_loaded_tokens(req.rid)
                if storage_hit:
                    req.storage_hit_length = storage_hit
            req.init_next_round_input(self.cache)
            host_hit = req.host_hit_length
            uncached = max(
                1,
                len(req.full_untruncated_fill_ids) - len(req.prefix_indices) - host_hit,
            )
            budget -= min(
                budget, ((uncached + s.page_size - 1) // s.page_size) * s.page_size
            )
            slots -= 1
            if host_hit <= 0 or loaded + host_hit > self.max_tokens:
                continue
            # Preserve room for the next computation, even under cache pressure.
            capacity = (
                s.token_to_kv_pool_allocator.available_size()
                + self.cache.evictable_size()
                - 2 * s.chunked_prefill_size
            )
            if host_hit > capacity:
                continue
            node = req.best_match_node
            if node in self.cache.ongoing_load_back:
                continue
            if self.cache.load_back(node, mem_quota=capacity, req=req):
                lock = self.cache.inc_lock_ref(node).to_dec_params()
                self.pins.append((req, node, lock))
                req.prefetched_host_hit_length = host_hit
                loaded += host_hit
        if not self.pins:
            return
        # This call occurs before run_batch: the existing load fence waits for
        # the OLD forward only. Current-batch pages are already allocated and
        # locked. No global WAR fence or model shared-read contract is removed.
        self.prepared = cc.prepare_loading()

    def submit(self):
        if self.prepared is None:
            return
        cc = self.cache.cache_controller
        cc.submit_loading(self.prepared, wait_on_caller=False, defer_broadcast=True)
        self.prepared = None
        self.completion = cc.ack_load_queue[-1]
        logger.info(
            "Next-batch host prefetch submitted: rids=%s tokens=%d bytes=%d",
            [req.rid for req, _, _ in self.pins],
            self.completion.num_tokens,
            self.completion.num_bytes,
        )
