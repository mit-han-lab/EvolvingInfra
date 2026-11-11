"""Compute-first late decode: replicated admission and bounded catch-up.

Clock decisions belong to the leader binder. Admission reads only replicated
request/cache state or the leader lane received in the sender MIN reduction.
"""
from __future__ import annotations

import logging
import math
import time
from collections import Counter

from decoupled_engine.srt.environ import envs

logger = logging.getLogger(__name__)
START_OK = 1
HOLD = 2


def force_bound_for_smoke(req):
    prefix = envs.SGLANG_TEST_LATE_D_FORCE_BOUND_RID_PREFIX.get()
    return bool(prefix) and req.rid.startswith(prefix)


def admit_delay_for_smoke(req):
    prefix = envs.SGLANG_TEST_LATE_D_DELAY_ADMIT_RID_PREFIX.get()
    if prefix and not req.rid.startswith(prefix):
        return 0.0
    return max(0.0, envs.SGLANG_TEST_LATE_D_DELAY_ADMIT_S.get())


def park_gate_room(new_need, evictable_prefix):
    """Free tokens one unbound row's park gate takes before the bound reserve
    (its second condition): its new need plus its evictable prefix. Shared
    with the BWP v2 admission-first reserve (bootstrap_prefetch.py)."""
    return new_need + evictable_prefix


def park_gate(*, unbound_tokens, locked_prefix, new_need, pool, remaining_total,
              evictable_prefix, reserve, cap=0.35):
    """Check the final parked footprint and leave room for bound work.

    A first room can exceed the park fraction, but never the bound reserve.
    The latter subtracts a currently evictable prefix before it gets pinned.
    """
    projected = unbound_tokens + locked_prefix + new_need
    cap_ok = unbound_tokens == 0 or projected <= cap * pool
    return cap_ok and remaining_total - park_gate_room(new_need, evictable_prefix) >= reserve


# Park-gate aging (SGLANG_LATE_D_PARK_AGE_S > 0; symmetric compute-first
# workers; default off). ExportTable.poll_released appends one value per
# computing row to its per-pass MIN vote. Owner rank 0 owns the clock, as for
# the park and admission timers: it votes its decision and every other rank
# votes the neutral AGED_NONE, so the MIN is rank 0's value on every rank.
# A row ages from its first refusal, by the park gate or by an aged head's
# hold-back. Logged refusal classes: CAP (the park fraction refused it), ROOM
# (only the room rule did) and BLOCK (only an aged head's hold-back did).
# SGLANG_LATE_D_PARK_AGE_S=1e9 is a log-only mode: nothing ages, admission is
# stock, and the aging diagnostics (gate_skip rem/evictable/class, admit_wait,
# aging_stats) are logged.
AGED_RELEASED = 0  # aged head whose hold-back window (AGED_BLOCK_S) is over
AGED_ON = 1  # refused for >= AGE_S (a head: its window not started or running)
AGED_NONE = 2  # neutral: not aged, launched, bound, or a non-zero rank
AGING_STATS_EVERY_S = 60.0


def park_aging_enabled():
    return envs.SGLANG_LATE_D_PARK_AGE_S.get() > 0


def park_aged_cap():
    """The aged head's park cap: SGLANG_LATE_D_PARK_AGED_CAP, or the stock cap."""
    aged = envs.SGLANG_LATE_D_PARK_AGED_CAP.get()
    return aged if aged > 0 else envs.SGLANG_LATE_D_PARK_CAP.get()


def aged_vote(req, now, age_s, block_s):
    """Owner rank 0's vote for one computing row, on rank 0's clock.

    A waiting unbound row is aged once its first refusal is ``age_s`` old.
    The hold-back window of the worker's aged head runs from its first
    hold-back (``late_d_head_since``: stamped when the head is refused while
    AGED_ON, cleared whenever it stops being head), so time in which the head
    held nobody back does not use the window, and a new head never starts with
    an old window.
    """
    if not getattr(req, "pending_bootstrap", False) or getattr(
        req, "launched_unbound", False
    ):
        return AGED_NONE
    first = getattr(req, "late_d_refused_at", None)
    if first is None or now - first < age_s:
        return AGED_NONE
    head = getattr(req, "late_d_head_since", None)
    if head is not None and now - head >= block_s:
        return AGED_RELEASED
    return AGED_ON


def gate_class(unbound_tokens, locked_prefix, new_need, pool, cap):
    """CAP when the park fraction refused the row, ROOM when only the room
    rule (free + evictable against need + reserve) did."""
    if unbound_tokens > 0 and unbound_tokens + locked_prefix + new_need > cap * pool:
        return "CAP"
    return "ROOM"


def catchup_end(start, computed_end, budget, page_size):
    """Apply the per-pass cap BEFORE page alignment. Zero bytes are legal."""
    end = min(computed_end, start + max(0, budget))
    return max(start, end - end % page_size)


def need_bind_lead(bind_seconds, pass_seconds, chunk_tokens, host_tokens,
                   uncached_tokens, catchup_budget=78000, margin=1):
    b = 1 + math.ceil(min(max(bind_seconds, 0), 2.0) / max(pass_seconds, .001))
    multi = uncached_tokens > chunk_tokens
    backlog = max(0, host_tokens + uncached_tokens - chunk_tokens) if multi else 0
    c = math.ceil(backlog / catchup_budget) if catchup_budget > 0 else 0
    return b, b + margin + c


def freeze_need_work_at_launch(req, prefix_tokens, device_tokens):
    """Keep the first-launch work sizes for the room's catch-up lead.

    Later prefix matches include KV computed by this room. That KV still
    needs transfer, so it must not reduce the catch-up work h + u.
    Before launch the binder may continue using fresh cache estimates.
    """
    if getattr(req, "late_d_initial_uncached_tokens", None) is None:
        req.late_d_initial_uncached_tokens = max(
            0, len(req.origin_input_ids) - prefix_tokens)
        req.late_d_initial_host_tokens = max(0, prefix_tokens - device_tokens)


def evictable_prefix_tokens(req):
    """Read locks without touching radix nodes or changing LRU order."""
    node = getattr(req, "last_node", None)
    total = 0
    while node is not None and getattr(node, "parent", None) is not None:
        value = getattr(node, "value", None)
        if value is not None and getattr(node, "lock_ref", 0) == 0:
            total += len(value)
        node = node.parent
    return total


def remaining_need(req):
    full = len(getattr(req, "origin_input_ids", ()))
    prefix = len(getattr(req, "prefix_indices", ()))
    # Before load adoption host_hit is still outside prefix_indices. It is
    # included in full - device prefix. After adoption it is in prefix_indices.
    return max(0, full - prefix)


class NeedAdmissionMixin:
    def late_d_need_enabled(self):
        return bool(getattr(self, "_late_d_need_enabled", False))

    def late_d_need_begin_admission(self, adder, chunked_req=None):
        if not self.late_d_need_enabled():
            return
        self._late_d_need_counts = getattr(self, "_late_d_need_counts", Counter())
        chunk = int(self._late_d_chunk_tokens)
        tracked = self.late_d_need_tracked()
        started = [r for r in tracked if getattr(r, "launched_unbound", False)]
        # Count each request's full eventual footprint: conservative when
        # requests share a locked prefix, but safe under the no-loss rule.
        self._late_d_need_unbound = sum(len(r.origin_input_ids) for r in started)
        self._late_d_need_seen = {id(r) for r in started}
        lookahead = getattr(self, "late_d_lookahead", None)
        def bound_need(r):
            if lookahead is None:
                return remaining_need(r)
            candidate = lookahead.candidate(r)  # fresh, read-only; no LRU writes
            return candidate.uncached + candidate.host_tokens
        reserve = max([2 * chunk] + [bound_need(r) for r in self.waiting_queue
                                    if not r.pending_bootstrap])
        self._late_d_need_reserve = reserve
        self.late_d_bound_reserve_tokens = reserve
        self._late_d_need_gate_closed = False
        # Park-gate aging: the aged head (lockstep: agreed aged values and
        # arrival order) and whether its refusal holds the others back.
        self._late_d_aged_block = False
        if getattr(self, "_late_d_aging", None) is None:
            self._late_d_aging = park_aging_enabled()
        if self._late_d_aging:
            self._late_d_select_head()
            self._late_d_aging_stats()

    def late_d_need_skip(self, req, adder):
        if not self.late_d_need_enabled() or not req.pending_bootstrap:
            return False
        if getattr(req, "launched_unbound", False):
            return False
        counts = self._late_d_need_counts
        chunk = int(self._late_d_chunk_tokens)
        uncached = max(0, remaining_need(req) - getattr(req, "host_hit_length", 0))
        if uncached > chunk and not req.late_d_start_ok:
            counts["intent_skip"] += 1
            req.late_d_intent_skips = getattr(req, "late_d_intent_skips", 0) + 1
            return True
        if req.late_d_hold:
            counts["warm_hold_skip"] += 1
            req.late_d_hold_skips = getattr(req, "late_d_hold_skips", 0) + 1
            return True
        aging = getattr(self, "_late_d_aging", False)
        head = getattr(self, "_late_d_aged_head", None) if aging else None
        if head is not None and req is not head and getattr(self, "_late_d_aged_block", False):
            # The aged head was refused in this pass (it is scanned first):
            # no other unbound request may take its cap share or room.
            counts["aged_block"] += 1
            req.late_d_gate_blocked = True
            self._late_d_need_gate_closed = True
            self._late_d_note_refused(req, "BLOCK", blocked=True)
            return True
        pool = int(self.max_total_num_tokens)
        prefix = len(req.prefix_indices)
        need = remaining_need(req)
        evictable = evictable_prefix_tokens(req)
        cap = envs.SGLANG_LATE_D_PARK_CAP.get()
        if head is not None and req is head:
            cap = max(cap, park_aged_cap())
        ok = park_gate(
            unbound_tokens=self._late_d_need_unbound, locked_prefix=prefix,
            new_need=need, pool=pool, remaining_total=adder.rem_total_tokens,
            evictable_prefix=evictable,
            reserve=self._late_d_need_reserve,
            cap=cap)
        req.late_d_gate_blocked = not ok
        if not ok:
            counts["gate_skip"] += 1
            req.late_d_gate_skips = getattr(req, "late_d_gate_skips", 0) + 1
            self._late_d_need_gate_closed = True
            if aging:
                cls = gate_class(self._late_d_need_unbound, prefix, need, pool, cap)
                self._late_d_note_refused(req, cls)
                is_head = req is head
                if is_head and getattr(req, "late_d_aged", AGED_NONE) == AGED_ON:
                    if not self._late_d_aged_block:
                        counts["aged_block_passes"] += 1
                    self._late_d_aged_block = True
                    if getattr(req, "late_d_head_since", None) is None:
                        # The window starts at the first hold-back (rank 0's
                        # stamp is the one its vote reads).
                        req.late_d_head_since = self._late_d_need_now()
                        if self._late_d_need_rank0():
                            logger.info("LATE_D_NEED aged_hold rid=%s wait_s=%.3f class=%s",
                                        req.rid, self._late_d_wait_s(req), cls)
                logger.info("LATE_D_NEED gate_skip rid=%s need=%d prefix=%d unbound=%d reserve=%d "
                            "rem=%d evictable=%d class=%s cap=%.3f head=%d",
                            req.rid, need, prefix, self._late_d_need_unbound,
                            self._late_d_need_reserve, adder.rem_total_tokens, evictable, cls,
                            cap, int(is_head))
            else:
                logger.info("LATE_D_NEED gate_skip rid=%s need=%d prefix=%d unbound=%d reserve=%d",
                            req.rid, need, prefix, self._late_d_need_unbound,
                            self._late_d_need_reserve)
        return not ok

    def late_d_need_admitted(self, req):
        if not self.late_d_need_enabled() or not req.pending_bootstrap:
            return
        if id(req) not in self._late_d_need_seen:
            self._late_d_need_seen.add(id(req))
            self._late_d_need_unbound += len(req.origin_input_ids)
        req.late_d_gate_blocked = False
        if getattr(self, "_late_d_aging", False):
            self._late_d_aging_admitted(req)

    # -- park-gate aging (SGLANG_LATE_D_PARK_AGE_S > 0) ---------------------
    def late_d_need_scan_order(self, queue):
        """The admission scan order: the aged head first, then ``queue`` as is.
        Without aging (or without a head) ``queue`` itself is returned."""
        head = getattr(self, "_late_d_aged_head", None)
        if not getattr(self, "_late_d_aging", False) or head is None:
            return queue
        if not any(r is head for r in queue):
            return queue  # never scan (and so never launch) a row twice
        return [head] + [r for r in queue if r is not head]

    def _late_d_need_now(self):
        """The export table's clock (the one poll_released votes with)."""
        clock = getattr(getattr(self, "export_table", None), "clock", None)
        return clock() if callable(clock) else time.monotonic()

    def _late_d_need_rank0(self):
        ps = getattr(self, "ps", None)
        return getattr(ps, "tp_rank", getattr(self, "tp_rank", 0)) == 0

    def _late_d_note_refused(self, req, cls, blocked=False):
        if getattr(req, "late_d_refused_at", None) is None:
            req.late_d_refused_at = self._late_d_need_now()
        req.late_d_refusals = getattr(req, "late_d_refusals", 0) + 1
        if blocked:
            req.late_d_aged_blocks = getattr(req, "late_d_aged_blocks", 0) + 1
        else:
            req.late_d_refused_class = cls

    def _late_d_wait_s(self, req):
        first = getattr(req, "late_d_refused_at", None)
        return 0.0 if first is None else max(0.0, self._late_d_need_now() - first)

    def _late_d_head_ok(self, req, in_queue=False):
        """Lockstep test: an aged (agreed value), waiting, unbound, unlaunched
        request still in the waiting queue."""
        if getattr(req, "late_d_aged", AGED_NONE) not in (AGED_ON, AGED_RELEASED):
            return False
        if not getattr(req, "pending_bootstrap", False) or getattr(
            req, "launched_unbound", False
        ):
            return False
        if getattr(req, "finished_reason", None) is not None:
            return False
        return in_queue or any(r is req for r in self.waiting_queue)

    def _late_d_select_head(self):
        """Keep the current aged head while it is valid (sticky: the head
        changes only when it leaves), else choose the aged request that arrived
        first. Only lockstep state is read; rank 0 stamps its own clock."""
        counts = self._late_d_need_counts
        head = getattr(self, "_late_d_aged_head", None)
        if head is not None and not self._late_d_head_ok(head):
            head.late_d_head_since = None
            head = None
        if head is None:
            best = None
            for i, r in enumerate(self.waiting_queue):
                if self._late_d_head_ok(r, in_queue=True):
                    key = (getattr(r, "sym_arrival_seq", i), i)
                    if best is None or key < best[0]:
                        best = (key, r)
            if best is not None:
                head = best[1]
                head.late_d_head_since = None  # the window starts at its first hold-back
                counts["aged_heads"] += 1
                if self._late_d_need_rank0():
                    logger.info(
                        "LATE_D_NEED aged rid=%s wait_s=%.3f class=%s need=%d prefix=%d unbound=%d",
                        head.rid, self._late_d_wait_s(head),
                        getattr(head, "late_d_refused_class", "BLOCK"),
                        remaining_need(head), len(head.prefix_indices),
                        self._late_d_need_unbound)
        self._late_d_aged_head = head
        if (head is not None and getattr(head, "late_d_aged", AGED_NONE) == AGED_RELEASED
                and not getattr(head, "late_d_release_logged", False)):
            head.late_d_release_logged = True
            counts["aged_releases"] += 1
            if self._late_d_need_rank0():
                logger.info("LATE_D_NEED aged_release rid=%s wait_s=%.3f class=%s",
                            head.rid, self._late_d_wait_s(head),
                            getattr(head, "late_d_refused_class", "BLOCK"))

    def _late_d_aging_admitted(self, req):
        counts = self._late_d_need_counts
        rank0 = self._late_d_need_rank0()
        if req is getattr(self, "_late_d_aged_head", None):
            counts["aged_admits"] += 1
            if rank0:
                logger.info("LATE_D_NEED aged_admit rid=%s wait_s=%.3f class=%s released=%d",
                            req.rid, self._late_d_wait_s(req),
                            getattr(req, "late_d_refused_class", "BLOCK"),
                            int(getattr(req, "late_d_release_logged", False)))
            self._late_d_aged_head = None
            self._late_d_aged_block = False
        req.late_d_head_since = None
        if rank0 and getattr(req, "late_d_refused_at", None) is not None:
            logger.info("LATE_D_NEED admit_wait rid=%s gate_s=%.3f refusals=%d blocked=%d class=%s aged=%d",
                        req.rid, self._late_d_wait_s(req), getattr(req, "late_d_refusals", 0),
                        getattr(req, "late_d_aged_blocks", 0),
                        getattr(req, "late_d_refused_class", "BLOCK"),
                        int(getattr(req, "late_d_aged", AGED_NONE) != AGED_NONE))

    def _late_d_aging_stats(self):
        """Rank 0, about every AGING_STATS_EVERY_S: cumulative aging counters
        and the current head (its rid and wait), to measure the hold-back."""
        if not self._late_d_need_rank0():
            return
        now = self._late_d_need_now()
        last = getattr(self, "_late_d_aging_stats_at", None)
        if last is None:
            self._late_d_aging_stats_at = now
            return
        if now - last < AGING_STATS_EVERY_S:
            return
        self._late_d_aging_stats_at = now
        c = self._late_d_need_counts
        head = getattr(self, "_late_d_aged_head", None)
        logger.info(
            "LATE_D_NEED aging_stats blocked_passes=%d blocked_rows=%d heads=%d releases=%d admits=%d "
            "gate_skips=%d head=%s head_wait_s=%.3f head_state=%d",
            c["aged_block_passes"], c["aged_block"], c["aged_heads"], c["aged_releases"],
            c["aged_admits"], c["gate_skip"], head.rid if head is not None else "-",
            self._late_d_wait_s(head) if head is not None else 0.0,
            getattr(head, "late_d_aged", AGED_NONE) if head is not None else AGED_NONE)

    def late_d_need_tracked(self):
        from decoupled_engine.srt.disaggregation.utils import is_aborted
        table = getattr(self, "export_table", None)
        if table is not None and getattr(table, "compute_before_allocation", False):
            return [e.req for e in table.entries.values()
                    if e.compute_before_allocation and not e.reader_ready
                    and e.abort_reason is None]
        queues = [self.disagg_prefill_bootstrap_queue.queue, self.waiting_queue,
                  self.disagg_prefill_inflight_queue,
                  [self.chunked_req] if self.chunked_req is not None else []]
        for attr in ("last_batch", "running_batch"):
            batch = getattr(self, attr, None)
            if batch is not None:
                queues.append(batch.reqs)
        seen, result = set(), []
        for queue in queues:
            for req in queue:
                if id(req) in seen or is_aborted(req):
                    continue
                seen.add(id(req))
                if getattr(req, "late_d_need", False) and req.pending_bootstrap:
                    result.append(req)
        return result
