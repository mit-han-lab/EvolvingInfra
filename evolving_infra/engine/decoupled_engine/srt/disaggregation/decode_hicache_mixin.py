# Adapted from SGLang python/sglang/srt/disaggregation/decode_hicache_mixin.py (commit 297e54fd10), modified for evolving-infra.
"""HiCache integration mixins for the decode side of PD disaggregation"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from decoupled_engine.srt.disaggregation.base import KVPoll
from decoupled_engine.srt.disaggregation.decode_hicache_restore_log import DecodeRestoreOpLog
from decoupled_engine.srt.mem_cache.base_prefix_cache import (
    CacheRequestOutcome,
    InitLoadBackParams,
    MatchPrefixParams,
)
from decoupled_engine.srt.mem_cache.common import _req_radix_key
from decoupled_engine.srt.mem_cache.unified_cache.component_type import ComponentType

if TYPE_CHECKING:
    from decoupled_engine.srt.disaggregation.decode import DecodeRequest
    from decoupled_engine.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


def _tree_uses_bigram_keys(tree_cache: Any) -> bool:
    """Whether the radix tree keys KV position i by the bigram (t[i], t[i+1])
    (EAGLE), so a match of N positions needs N + 1 raw tokens."""
    tree_core = getattr(tree_cache, "tree_core", None)
    owner = tree_core if tree_core is not None else tree_cache
    return getattr(owner, "is_eagle", False) is True


@dataclass
class DecodePrefixMatch:
    prefix_indices: torch.Tensor
    l2_host_hit_length: int
    l3_storage_hit_length: int
    last_device_node: Any
    last_host_node: Any = None
    prefetch_registered: bool = False
    active_components: Optional[tuple[ComponentType, ...]] = None

    @property
    def l1_prefix_len(self) -> int:
        return len(self.prefix_indices)

    @property
    def decode_prefix_len(self) -> int:
        return self.l1_prefix_len + self.l2_host_hit_length + self.l3_storage_hit_length

    @property
    def needs_local_restore(self) -> bool:
        return self.decode_prefix_len > self.l1_prefix_len

    @property
    def restore_token_count(self) -> int:
        """Number of tokens that need L2/L3 load_back to device."""
        return self.decode_prefix_len - self.l1_prefix_len


class HiCacheRestoreResult(Enum):
    """Outcome of one tick of the HiCache local-restore state machine."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


def _bigram_boundary_tokens(tree_cache) -> int:
    # An EAGLE tree keys N bigrams with N + 1 token ids; prefix lengths count bigrams.
    return int(tree_cache.tree_core.is_eagle)


class DecodeHiCachePreallocMixin:
    """HiCache hooks for ``DecodePreallocQueue``: issue prefetch + reserve tokens."""

    def _build_decode_prefix_match(
        self,
        req: Req,
        result: Any,
        *,
        max_prefix_len: Optional[int] = None,
        active_components: Optional[tuple[ComponentType, ...]] = None,
    ) -> DecodePrefixMatch:
        """Convert a ``match_kv_cache`` result into ``DecodePrefixMatch``.

        Performs the optional L3 storage hit length query when decode-side
        HiCache is enabled and the last host node is backed up.
        """
        prefix_indices = self.tree_cache.prefix_device_indices(req)
        l1_prefix_len = len(prefix_indices)
        l2_host_hit_length = result.host_hit_length
        if active_components is not None and self._uses_window_host_radix():
            # DSv4 window residency, stage S5: the host
            # restore gates structural decisions on rank-local event queries,
            # which window residency's rank determinism (I7) forbids. Reuse
            # the device (L1) prefix only; L3 is skipped below as for every
            # partial-component match.
            l2_host_hit_length = 0

        l3_storage_hit_length = 0
        last_host_node = None
        # Partial-component decode cache currently uses only the local host
        # tier. L3 prefetch remains all-component and is intentionally skipped.
        if (
            self.scheduler.enable_decode_hicache
            and self.tree_cache.enable_storage
            and active_components is None
        ):
            last_host_node = result.last_host_node
            if self.tree_cache.is_backuped(last_host_node) or self.tree_cache.is_root(
                last_host_node
            ):
                matched_len = l1_prefix_len + l2_host_hit_length
                suffix_tokens = req.origin_input_ids[matched_len:max_prefix_len]
                last_hash = self.tree_cache.get_last_hash_value(last_host_node)
                prefix_keys = (
                    self.tree_cache.get_prefix_hash_values(last_host_node)
                    if self.tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                l3_storage_hit_length = self.tree_cache.query_storage_hit_length(
                    result.last_host_node,
                    suffix_tokens,
                    last_hash,
                    prefix_keys,
                    extra_key=req.extra_key,
                    cache_salt=req.cache_salt,
                )

        return DecodePrefixMatch(
            prefix_indices=prefix_indices,
            l2_host_hit_length=l2_host_hit_length,
            l3_storage_hit_length=l3_storage_hit_length,
            last_device_node=result.last_device_node,
            last_host_node=(
                result.last_host_node if l3_storage_hit_length > 0 else None
            ),
            active_components=active_components,
        )

    def _start_hicache_prefetch(
        self, req: Req, prefix_match: Optional[DecodePrefixMatch]
    ) -> None:
        """Issue L3 storage prefetch after admission succeeds.

        Clears the L3 span unless the prefetch was taken, so prefill is never
        promised tokens nothing will load.
        """
        if (
            prefix_match is None
            or prefix_match.l3_storage_hit_length <= 0
            or prefix_match.last_host_node is None
        ):
            return
        try:
            matched_len = prefix_match.l1_prefix_len + prefix_match.l2_host_hit_length
            last_hash = self.tree_cache.get_last_hash_value(prefix_match.last_host_node)
            prefix_keys = (
                self.tree_cache.get_prefix_hash_values(prefix_match.last_host_node)
                if self.tree_cache.hicache_storage_pass_prefix_keys
                else None
            )
            # Same span as the prefill scheduler: the queried hit end, plus the
            # bigram boundary token, with no second storage query.
            ticketed = self.tree_cache.prefetch_from_storage(
                req.cache_request_handle,
                prefix_match.last_host_node,
                req.origin_input_ids[matched_len:],
                last_hash,
                prefix_keys,
                matched_prefix_tokens=req.origin_input_ids[:matched_len],
                extra_key=req.extra_key,
                cache_salt=req.cache_salt,
                storage_hit_end=matched_len + prefix_match.l3_storage_hit_length,
            )
            prefix_match.prefetch_registered = self.tree_cache.has_ongoing_prefetch(
                req.cache_request_handle
            )
            if not (prefix_match.prefetch_registered or ticketed):
                # Declined (below prefetch_threshold or rate limited): nothing loads.
                prefix_match.l3_storage_hit_length = 0
        except Exception as e:
            logger.warning(
                "HiCache L3 prefetch failed for rid=%s: %s; falling back to L2-only LoadingBack",
                req.rid,
                e,
            )
            prefix_match.l3_storage_hit_length = 0
            prefix_match.prefetch_registered = False

    def _hicache_pending_restore_tokens(self) -> int:
        """Total device tokens reserved for pending HiCache L2/L3 load_back."""
        if not self.scheduler.enable_decode_hicache:
            return 0
        return sum(
            dr.prefix_match.restore_token_count
            for dr in self.transfer_queue.queue
            if dr.prefix_match is not None and dr.hicache_restore_lock is None
        )


class HiCacheRestoreGatedKVReceiver:
    """Wraps a kv_receiver so the HiCache restore outcome enters the KV poll."""

    def __init__(self, decode_req: DecodeRequest):
        self.decode_req = decode_req

    def poll(self) -> KVPoll:
        poll = self.decode_req.kv_receiver.poll()
        if poll == KVPoll.Success:
            restore_status = self.decode_req.hicache_restore_status
            if restore_status == HiCacheRestoreResult.PENDING:
                return KVPoll.Transferring
            if restore_status == HiCacheRestoreResult.FAILED:
                return KVPoll.Failed
        return poll


class DecodeHiCacheTransferMixin:
    """HiCache hooks for ``DecodeTransferQueue``: drive restore state machine."""

    def _clean_hicache_prefetch_resources(self, decode_req: DecodeRequest) -> None:
        if (
            decode_req.prefix_match is not None
            and decode_req.prefix_match.prefetch_registered
        ):
            self.tree_cache.finish(
                decode_req.req.cache_request_handle, CacheRequestOutcome.ABORT
            )
        self.tree_cache.unlock(decode_req.hicache_restore_lock)
        decode_req.hicache_restore_lock = None

    def _try_hicache_queue_load_back(self, dr: DecodeRequest) -> bool:
        """Prepare the promised prefix and hold its lock until commit/abort.

        On success, ``dr.hicache_restore_lock`` and ``hicache_restored_kv_indices``
        are populated; the lock is held until commit/abort. Device-visible indices
        can still have DMA in flight, so readiness is decided by the load-back
        event even when this request queues no copy.
        """
        pm = dr.prefix_match

        # Wait for L3 -> L2 prefetch to drain (skip when no L3 hit).
        if pm.l3_storage_hit_length > 0:
            if not self.tree_cache.check_prefetch_progress(dr.req.cache_request_handle):
                return False
            self.tree_cache.pop_prefetch_loaded_tokens(dr.req.cache_request_handle)

        # Partial-component (content_only / window_host) matches restore only
        # the components admission matched and locked.
        component_kwargs = (
            {}
            if pm.active_components is None
            else {"active_components": pm.active_components}
        )
        # The transfer suffix is already allocated: rematch only promised tokens,
        # without overwriting the request's preallocation state or lock receipt.
        rematch_token_len = pm.decode_prefix_len + _bigram_boundary_tokens(
            self.tree_cache
        )
        rematch = self.tree_cache.match_prefix(
            MatchPrefixParams(
                key=_req_radix_key(
                    self.tree_cache,
                    dr.req,
                    dr.req.origin_input_ids,
                    rematch_token_len,
                ),
                cow_mamba=False,
                req=dr.req,
                **component_kwargs,
            )
        )
        owned_node = dr.req.last_node
        try:
            dr.req.last_node = rematch.last_device_node
            loaded_len, restored_node = self.tree_cache.init_load_back(
                InitLoadBackParams(
                    best_match_node=rematch.best_match_node,
                    host_hit_length=rematch.host_hit_length,
                    req=dr.req,
                    **component_kwargs,
                )
            )
        finally:
            dr.req.last_node = owned_node
        observation = getattr(dr.req, "sched_trace_decode_prefix_observation", None)
        if observation is not None:
            # Count rows returned by load-back, including a partial result on
            # failure. A match is not a queued load; a queued load is not READY.
            observation["host_load_back_queued_tokens"] += loaded_len
        # Failback: total coverage < required prefix means device alloc likely failed.
        if rematch.device_prefix_len + loaded_len < pm.decode_prefix_len:
            logger.warning(
                "HiCache load_back failed for rid=%s: device_indices=%d, "
                "new_indices=%d, expected decode_prefix_len=%d (l1=%d, l2=%d, l3=%d)",
                dr.req.rid,
                rematch.device_prefix_len,
                loaded_len,
                pm.decode_prefix_len,
                pm.l1_prefix_len,
                pm.l2_host_hit_length,
                pm.l3_storage_hit_length,
            )
            dr.hicache_restore_status = HiCacheRestoreResult.FAILED
            return False

        # The load published its span: the restored path holds the whole prefix.
        dr.hicache_restored_kv_indices = self.tree_cache.path_device_indices(
            restored_node
        )[pm.l1_prefix_len : pm.decode_prefix_len]
        # Lock the same components the admission lock took (content_only skips
        # SWA), so the commit can hand this lock to the request.
        admission_lock = dr.req.lock
        skipped = (
            tuple(admission_lock.receipt.skipped_lock_components)
            if admission_lock is not None
            else ()
        )
        if skipped:
            dr.hicache_restore_lock = self.tree_cache.lock(
                restored_node, skip_lock_components=skipped
            )
        else:
            dr.hicache_restore_lock = self.tree_cache.lock(restored_node)

        return True

    def _hicache_restore_op_log(self) -> DecodeRestoreOpLog:
        op_log = self.__dict__.get("_restore_op_log")
        if op_log is None:
            from decoupled_engine.srt.environ import envs

            enabled = (
                getattr(self, "tp_rank", 0) == 0
                and envs.SGLANG_DECODE_HICACHE_RESTORE_LOG.get()
            )
            device_module = None
            if enabled and torch.cuda.is_available():
                from decoupled_engine.srt.utils import get_device_module

                device_module = get_device_module()
            op_log = DecodeRestoreOpLog(
                enabled=enabled,
                page_size=getattr(self.tree_cache, "page_size", 1),
                device_module=device_module,
            )
            self._restore_op_log = op_log
        return op_log

    def _poll_hicache_restore_op_log(self) -> None:
        """Log restore ops that finished, also when no request is left in the
        transfer queue (a failed transfer drops an op's requests, and an op
        that finishes on the tick its requests are committed may have been
        polled just before)."""
        op_log = self.__dict__.get("_restore_op_log")
        if op_log is not None:
            op_log.poll()

    def _process_hicache_local_restores(self, decode_reqs: List[DecodeRequest]) -> None:
        # Load-back slots free up in lockstep on every rank, and a restore's
        # readiness enters the MIN-reduced KV poll (HiCacheRestoreGatedKVReceiver),
        # so this method runs no collectives.
        if not hasattr(self.tree_cache, "is_load_back_event_done"):
            return
        op_log = self._hicache_restore_op_log()

        # L1 followers can see indices published before the restoring DMA ends.
        active = [
            dr
            for dr in decode_reqs
            if dr.hicache_restore_status == HiCacheRestoreResult.PENDING
        ]
        if not active:
            op_log.poll()
            return

        # Poll each local event once; transfer polling synchronizes request readiness.
        done = {
            index: self.tree_cache.is_load_back_event_done(index)
            for index in sorted({dr.hicache_load_consumer_index for dr in active})
            if index >= 0
        }
        for dr in active:
            if done.get(dr.hicache_load_consumer_index, False):
                dr.hicache_restore_status = HiCacheRestoreResult.READY
        # An op is logged done no later than the tick that marks its requests
        # READY (they may leave the queue on this tick).
        op_log.poll()

        # Phase B: queue new load_back ops if the next slot is free on every rank.
        if not self.tree_cache.has_free_load_back_slot():
            return
        queued = [
            dr
            for dr in active
            if dr.hicache_restore_status == HiCacheRestoreResult.PENDING
            and dr.hicache_load_consumer_index < 0
            and (
                dr.prefix_match is None
                or not dr.prefix_match.needs_local_restore
                or self._try_hicache_queue_load_back(dr)
            )
        ]
        if not queued:
            return

        counter = self.tree_cache.cache_controller.layer_done_counter
        pre = op_log.before_issue()
        consumer_index = self.tree_cache.ready_to_load_host_cache()
        op_log.after_issue(
            self.tree_cache.cache_controller, consumer_index, len(queued), pre
        )
        if consumer_index < 0:
            # No new copy: earlier copies on the load stream still protect L1 hits.
            consumer_index = counter.producer_index
        ready = self.tree_cache.is_load_back_event_done(consumer_index)
        for dr in queued:
            dr.hicache_load_consumer_index = consumer_index
            if ready:
                dr.hicache_restore_status = HiCacheRestoreResult.READY

    def _commit_hicache_local_restore_to_req(self, decode_req: DecodeRequest) -> None:
        prefix_match = decode_req.prefix_match
        if prefix_match is None or not prefix_match.needs_local_restore:
            return

        req = decode_req.req
        restore_lock = decode_req.hicache_restore_lock
        assert restore_lock is not None
        self.tree_cache.unlock(req.lock)

        self.tree_cache.req_to_token_pool.write(
            (
                decode_req.req.kv.req_pool_idx,
                slice(prefix_match.l1_prefix_len, prefix_match.decode_prefix_len),
            ),
            decode_req.hicache_restored_kv_indices,
        )
        req.prefix_len = prefix_match.decode_prefix_len
        req.last_node = restore_lock.node
        req.lock = restore_lock
        # Prevent abort cleanup from releasing the transferred lock.
        decode_req.hicache_restore_lock = None
        observation = getattr(req, "sched_trace_decode_prefix_observation", None)
        if observation is not None:
            # Coverage committed after READY; can include rows already restored
            # by another request. This is deliberately not called H2D traffic.
            observation["local_restore_committed_tokens"] = len(
                decode_req.hicache_restored_kv_indices
            )
