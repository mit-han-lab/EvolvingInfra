# Adapted from SGLang python/sglang/srt/disaggregation/decode_hicache_mixin.py (commit 9a489f8d2f), modified for evolving-infra.
"""HiCache integration mixins for the decode side of PD disaggregation"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from decoupled_engine.srt.disaggregation.base import KVPoll
from decoupled_engine.srt.disaggregation.decode_hicache_restore_log import DecodeRestoreOpLog
from decoupled_engine.srt.managers.schedule_policy import match_prefix_for_req
from decoupled_engine.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    InitLoadBackParams,
)
from decoupled_engine.srt.mem_cache.unified_cache.components import ComponentType

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


class DecodeHiCachePreallocMixin:
    """HiCache hooks for ``DecodePreallocQueue``: issue prefetch + reserve tokens."""

    def _build_decode_prefix_match(
        self,
        req: Req,
        result: Any,
        *,
        active_components: Optional[tuple[ComponentType, ...]] = None,
    ) -> DecodePrefixMatch:
        """Convert a ``match_prefix_for_req`` result into ``DecodePrefixMatch``.

        Performs the optional L3 storage hit length query when decode-side
        HiCache is enabled and the last host node is backed up.
        """
        prefix_indices = result.device_indices
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
        if self.scheduler.enable_decode_hicache and active_components is None:
            last_host_node = self.tree_cache.resolve_node_handle(result.last_host_node)
            if last_host_node.backuped or last_host_node is self.tree_cache.root_node:
                matched_len = l1_prefix_len + l2_host_hit_length
                suffix_tokens = req.origin_input_ids[matched_len:]
                last_hash = last_host_node.get_last_hash_value()
                prefix_keys = (
                    last_host_node.get_prefix_hash_values(last_host_node.parent)
                    if self.tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                l3_storage_hit_length = self.tree_cache.query_storage_hit_length(
                    result.last_host_node,
                    suffix_tokens,
                    last_hash,
                    prefix_keys,
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

        On failure, degrades to L2-only restore by clearing l3 fields.
        """
        if (
            prefix_match is None
            or prefix_match.l3_storage_hit_length <= 0
            or prefix_match.last_host_node is None
        ):
            return
        try:
            node = self.tree_cache.resolve_node_handle(prefix_match.last_host_node)
            matched_len = prefix_match.l1_prefix_len + prefix_match.l2_host_hit_length
            suffix = req.origin_input_ids[
                matched_len : matched_len + prefix_match.l3_storage_hit_length
            ]
            last_hash = node.get_last_hash_value()
            prefix_keys = (
                node.get_prefix_hash_values(node.parent)
                if self.tree_cache.hicache_storage_pass_prefix_keys
                else None
            )
            self.tree_cache.prefetch_from_storage(
                req.rid,
                prefix_match.last_host_node,
                suffix,
                last_hash,
                prefix_keys,
                extra_key=req.extra_key,
                cache_salt=req.cache_salt,
            )
            prefix_match.prefetch_registered = (
                req.rid in self.tree_cache.ongoing_prefetch
            )
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
            if dr.prefix_match is not None
            and dr.hicache_restore_status == HiCacheRestoreResult.PENDING
            and dr.hicache_restored_node is None
        )


class HiCacheRestoreGatedKVReceiver:
    """Wraps a kv_receiver so KVPoll.Success is gated on HiCache restore READY."""

    def __init__(self, decode_req: DecodeRequest):
        self.decode_req = decode_req

    def poll(self) -> KVPoll:
        if self.decode_req.hicache_restore_status == HiCacheRestoreResult.FAILED:
            # Feed local host-restore failures into the existing rank-wide
            # transfer poll, so no peer admits a request with stale cache state.
            return KVPoll.Failed
        poll = self.decode_req.kv_receiver.poll()
        if (
            poll == KVPoll.Success
            and self.decode_req.hicache_restore_status == HiCacheRestoreResult.PENDING
        ):
            return KVPoll.Transferring
        return poll


class DecodeHiCacheTransferMixin:
    """HiCache hooks for ``DecodeTransferQueue``: drive restore state machine."""

    def _clean_hicache_prefetch_resources(self, decode_req: DecodeRequest) -> None:
        if (
            decode_req.prefix_match is not None
            and decode_req.prefix_match.prefetch_registered
        ):
            self.tree_cache.release_aborted_request(decode_req.req.rid)
        if decode_req.hicache_restored_node is not None:
            skipped = getattr(decode_req.req, "skip_lock_components", ())
            self.tree_cache.dec_lock_ref(
                decode_req.hicache_restored_node,
                DecLockRefParams(skip_lock_components=skipped) if skipped else None,
            )
            decode_req.hicache_restored_node = None

    def _try_hicache_queue_load_back(self, dr: DecodeRequest) -> bool:
        """Queue one L2->L1 load_back op for ``dr``; True iff a DMA was queued.

        On success, ``dr.hicache_restored_node`` and ``hicache_restored_kv_indices``
        are populated, and an inc_lock_ref is held until commit/abort.
        Trivial cases (all-on-device / no needed coverage) auto-flip to READY.
        Failback paths flip to FAILED.
        """
        pm = dr.prefix_match

        # Wait for L3 -> L2 prefetch to drain (skip when no L3 hit).
        if pm.l3_storage_hit_length > 0:
            if not self.tree_cache.check_prefetch_progress(dr.req.rid):
                return False
            self.tree_cache.pop_prefetch_loaded_tokens(dr.req.rid)

        # Partial matches use a shallow request copy so a failed restore cannot
        # move the request away from the lock acquired during admission. The
        # EAGLE (bigram) path does the same: its admission lock and prefix are
        # what the abort path releases.
        bigram = _tree_uses_bigram_keys(self.tree_cache)
        match_req = (
            copy.copy(dr.req) if pm.active_components is not None or bigram else dr.req
        )
        rematch = match_prefix_for_req(
            self.tree_cache,
            match_req,
            match_req.origin_input_ids,
            cow_mamba=False,
            include_req=True,
            active_components=pm.active_components,
            # The limit caps raw tokens. A bigram key of N KV positions spans
            # N + 1 raw tokens; capping at N would drop the last page and fail
            # every EAGLE restore as "coverage shrank".
            key_limit=pm.decode_prefix_len + int(bigram),
        )
        node = self.tree_cache.resolve_node_handle(rematch.best_match_node)
        while node is not self.tree_cache.root_node:
            if node.load_back_pending_id is not None:
                return False
            node = node.parent

        restore_tokens = pm.decode_prefix_len - len(rematch.device_indices)
        if rematch.host_hit_length < restore_tokens:
            logger.warning(
                "HiCache coverage shrank for rid=%s: device=%d host=%d expected=%d",
                dr.req.rid,
                len(rematch.device_indices),
                rematch.host_hit_length,
                pm.decode_prefix_len,
            )
            dr.hicache_restore_status = HiCacheRestoreResult.FAILED
            return False

        skipped = getattr(dr.req, "skip_lock_components", ())
        if restore_tokens == 0:
            dr.hicache_restored_kv_indices = rematch.device_indices[pm.l1_prefix_len :]
            dr.hicache_restored_node = rematch.last_device_node
            self.tree_cache.inc_lock_ref(
                rematch.last_device_node, skip_lock_components=skipped
            )
            dr.hicache_restore_status = HiCacheRestoreResult.READY
            return False

        new_indices, restored_node = self.tree_cache.init_load_back(
            InitLoadBackParams(
                best_match_node=rematch.best_match_node,
                host_hit_length=rematch.host_hit_length,
                mem_quota=restore_tokens,
                req=match_req,
                active_components=pm.active_components,
            )
        )
        observation = getattr(dr.req, "sched_trace_decode_prefix_observation", None)
        if observation is not None:
            # Count rows returned by load-back, including a partial result on
            # failure. A match is not a queued load; a queued load is not READY.
            observation["host_load_back_queued_tokens"] += len(new_indices)
        if len(new_indices) != restore_tokens:
            logger.warning(
                "HiCache load_back failed for rid=%s: device_indices=%d, "
                "new_indices=%d, expected decode_prefix_len=%d (l1=%d, l2=%d, l3=%d)",
                dr.req.rid,
                len(rematch.device_indices),
                len(new_indices),
                pm.decode_prefix_len,
                pm.l1_prefix_len,
                pm.l2_host_hit_length,
                pm.l3_storage_hit_length,
            )
            dr.hicache_restore_status = HiCacheRestoreResult.FAILED
            return False

        dr.hicache_restored_kv_indices = torch.cat(
            [rematch.device_indices[pm.l1_prefix_len :], new_indices]
        )
        dr.hicache_restored_node = restored_node
        self.tree_cache.inc_lock_ref(
            restored_node,
            skip_lock_components=skipped,
        )
        return True

    def _restore_flags_on_all_ranks(self, flags: List[bool]) -> List[bool]:
        # Restore events are rank-local; agree before advancing requests or
        # issuing another batch of host-dedup broadcasts.
        agreed = torch.tensor(flags, dtype=torch.int32)
        for group in (getattr(self, "attn_cp_cpu_group", None), self.gloo_group):
            if group is not None and torch.distributed.get_world_size(group) > 1:
                torch.distributed.all_reduce(
                    agreed, op=torch.distributed.ReduceOp.MIN, group=group
                )
        return [bool(value) for value in agreed.tolist()]

    def _restore_slot_free_on_all_ranks(self, slot_free: bool) -> bool:
        return self._restore_flags_on_all_ranks([slot_free])[0]

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
        if not hasattr(self.tree_cache, "is_load_back_event_done"):
            return
        op_log = self._hicache_restore_op_log()

        # Filter once: keep only PENDING reqs that still need restore work;
        # trivially-done reqs (no prefix_match / nothing to restore) flip to READY.
        active: List[DecodeRequest] = []
        for dr in decode_reqs:
            if dr.hicache_restore_status != HiCacheRestoreResult.PENDING:
                continue
            pm = dr.prefix_match
            if pm is None or not pm.needs_local_restore:
                dr.hicache_restore_status = HiCacheRestoreResult.READY
                continue
            active.append(dr)

        ready = [
            dr.hicache_restored_node is not None
            and self.tree_cache.is_load_back_event_done(dr.hicache_load_consumer_index)
            for dr in active
        ]
        eagle_slot_free = None
        if _tree_uses_bigram_keys(self.tree_cache):
            # GLM EAGLE restores target and draft rows together. A rank-local
            # READY transition can make peers enqueue different collectives on
            # the next tick; agree on completion and the next slot in one poll.
            counter = self.tree_cache.cache_controller.layer_done_counter
            flags = self._restore_flags_on_all_ranks(
                [
                    *ready,
                    self.tree_cache.is_load_back_event_done(
                        (counter.producer_index + 1) % counter.num_counters
                    ),
                ]
            )
            ready, eagle_slot_free = flags[:-1], flags[-1]

        # Phase A: advance in-flight DMAs only after the required rank agreement.
        for dr, done in zip(active, ready, strict=True):
            if done:
                dr.hicache_restore_status = HiCacheRestoreResult.READY
        # After Phase A: an op is logged done no later than the tick that marks
        # its requests READY (they may leave the queue on this tick).
        op_log.poll()

        # Phase B: queue new load_back ops if the next slot is free.
        # The (producer_index + 1) check ensures we never overwrite a still-in-flight slot:
        # if a previous req holds that slot and isn't done, its event won't be signaled.
        slot_free = eagle_slot_free
        if slot_free is None:
            counter = self.tree_cache.cache_controller.layer_done_counter
            slot_free = self._restore_slot_free_on_all_ranks(
                self.tree_cache.is_load_back_event_done(
                    (counter.producer_index + 1) % counter.num_counters
                )
            )
        if not slot_free:
            return
        queued = [
            dr
            for dr in active
            if dr.hicache_restored_node is None
            and self._try_hicache_queue_load_back(dr)
        ]
        if not queued:
            return

        # Phase C: kick off merged DMA, bind consumer_index for Phase A polling next tick.
        pre = op_log.before_issue()
        consumer_index = self.tree_cache.ready_to_load_host_cache()
        op_log.after_issue(
            self.tree_cache.cache_controller, consumer_index, len(queued), pre
        )
        if consumer_index < 0:
            for dr in queued:
                dr.hicache_restore_status = HiCacheRestoreResult.READY
            return
        for dr in queued:
            dr.hicache_load_consumer_index = consumer_index

    def _commit_hicache_local_restore_to_req(self, decode_req: DecodeRequest) -> None:
        prefix_match = decode_req.prefix_match
        if prefix_match is None or not prefix_match.needs_local_restore:
            return

        skipped = getattr(decode_req.req, "skip_lock_components", ())
        self.tree_cache.dec_lock_ref(
            prefix_match.last_device_node,
            DecLockRefParams(skip_lock_components=skipped) if skipped else None,
        )

        self.tree_cache.req_to_token_pool.write(
            (
                decode_req.req.req_pool_idx,
                slice(prefix_match.l1_prefix_len, prefix_match.decode_prefix_len),
            ),
            decode_req.hicache_restored_kv_indices,
        )
        decode_req.req.prefix_indices = torch.cat(
            [prefix_match.prefix_indices, decode_req.hicache_restored_kv_indices]
        )
        decode_req.req.last_node = decode_req.hicache_restored_node
        decode_req.hicache_restored_node = None
        observation = getattr(
            decode_req.req, "sched_trace_decode_prefix_observation", None
        )
        if observation is not None:
            # Coverage committed after READY; can include rows already restored
            # by another request. This is deliberately not called H2D traffic.
            observation["local_restore_committed_tokens"] = len(
                decode_req.hicache_restored_kv_indices
            )
