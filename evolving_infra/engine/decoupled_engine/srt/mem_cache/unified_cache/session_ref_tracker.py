# Adapted from SGLang python/sglang/srt/mem_cache/unified_cache/session_ref_tracker.py (commit 9a489f8d2f), modified for evolving-infra.
"""Native session references and optional versioned CPU KV retention.

Native streaming-session semantics are unchanged. ``KvSessionRetention`` tracks
one current context per external trajectory, independently of GPU eviction and
physical IO pins. Its methods run only on the ordered scheduler thread.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from decoupled_engine.srt.managers.scheduler_components.phase_timers import timed_phase
from decoupled_engine.srt.mem_cache.unified_cache.retention_order import (
    DEFAULT_FIXED_TABLE,
    IDLE_AGE_NAMES,
    ORDER_LRU,
    ORDER_PREDICTED_RETURN,
    ORDER_PREDICTED_RETURN_FIXED,
    STATS_TAUS,
    LockstepClock,
    ReturnGapTable,
    idle_age_bin,
    parse_retention_order,
)

if TYPE_CHECKING:
    from decoupled_engine.srt.managers.schedule_batch import Req
    from decoupled_engine.srt.mem_cache.unified_cache.components.tree_component import (
        TreeComponent,
    )
    from decoupled_engine.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

logger = logging.getLogger(__name__)

# Bounded guard against a request finishing after close. If a session id falls
# out of this LRU after 8192 later closes, an extremely late finish can tag
# again; explicit register_session clears the tombstone for intentional id reuse.
_CLOSED_SESSION_TOMBSTONE_LIMIT = 8192


def merge_kv_session_replies(replies):
    """Combine TP control replies; return READY only for whole-worker readiness.

    Input is one ordered ``{boot_id, results}`` response per TP rank. Output
    keeps that operation order, takes the minimum committed token count, and
    fails closed on version, generation or state disagreement. This helper is
    used only by background control RPCs, never per token or forward step.
    """
    if not replies:
        raise ValueError("expected at least one TP reply")
    first = replies[0]
    compatible = all(
        reply["boot_id"] == first["boot_id"]
        and len(reply["results"]) == len(first["results"])
        for reply in replies
    )
    results = []
    for index, result in enumerate(first["results"]):
        merged = dict(result)
        peers = [reply["results"][index] for reply in replies] if compatible else []
        if not peers or any(
            peer["session_id"] != result["session_id"]
            or peer["version"] != result["version"]
            for peer in peers
        ):
            merged.update(status="unknown", ready_tokens=0)
        else:
            states = {peer["status"] for peer in peers}
            ready = min(peer.get("ready_tokens", 0) for peer in peers)
            required = max(peer.get("required_tokens", 0) for peer in peers)
            if states <= {"ready", "pending"}:
                # Required ranges must also agree: a shorter rank cannot attest
                # that another rank's longer current context is complete.
                same_range = all(
                    peer.get("required_tokens", 0) == required for peer in peers
                )
                status = (
                    "ready"
                    if states == {"ready"} and same_range and ready >= required
                    else "pending"
                )
            else:
                status = next(iter(states)) if len(states) == 1 else "unknown"
            merged.update(status=status, ready_tokens=ready, required_tokens=required)
        results.append(merged)
    return {"boot_id": first["boot_id"], "results": results}


def parse_kv_session(value):
    """Validate optional wire metadata; return ``(session_id, version)`` or None."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"session_id", "version"}:
        raise ValueError("kv_session must contain only session_id and version")
    sid, version = value["session_id"], value["version"]
    if not isinstance(sid, str) or not sid or len(sid) > 256:
        raise ValueError("kv_session.session_id must contain 1..256 characters")
    if type(version) is not int or not 1 <= version <= 2**63 - 1:
        raise ValueError("kv_session.version must be a positive signed 64-bit integer")
    return sid, version


def validate_kv_session_operations(operations):
    """Validate an entire bounded batch before mutation; return the input list."""
    if not isinstance(operations, list) or not 1 <= len(operations) <= 128:
        raise ValueError("KV session control requires 1..128 operations")
    for operation in operations:
        if not isinstance(operation, dict) or set(operation) - {
            "op",
            "session_id",
            "version",
            "expected_boot_id",
        }:
            raise ValueError("invalid KV session control fields")
        if operation.get("op") not in ("status", "release", "end", "protect"):
            raise ValueError("unknown KV session operation")
        parse_kv_session(
            {
                "session_id": operation.get("session_id"),
                "version": operation.get("version"),
            }
        )
        boot = operation.get("expected_boot_id")
        if operation["op"] != "status" and (
            not isinstance(boot, str) or not boot or len(boot) > 256
        ):
            raise ValueError("KV session mutation requires expected_boot_id")
        if boot is not None and (not isinstance(boot, str) or len(boot) > 256):
            raise ValueError("invalid KV session expected_boot_id")
    return operations


@timed_phase("kv_session.forget")
def forget_kv_session_request(scheduler, req) -> None:
    """Drop a request that leaves the worker without cache_finished_req.

    Scheduler paths that remove a queued request (abort, waiting timeout,
    queue limit, priority refusal, bootstrap failure) call this, so the
    registration stops counting it as live and its soft pin ends. Any tree
    cache type is accepted: without --enable-kv-session-retention, and for an
    untagged request, nothing changes. Repeating it, or a later finish
    callback for the same request, changes nothing more.
    """
    tracker = getattr(getattr(scheduler, "tree_cache", None), "kv_sessions", None)
    if tracker is not None:
        tracker.refresh_request(req, finished=True)


def rebind_kv_session_request(scheduler, req) -> None:
    """Count a request that waits again as live in its registration again.

    Retraction, priority preemption, pause_generation(retract), an optimistic
    prefill requeue and a symmetric import fallback release the request's KV
    through cache_finished_req, which drops the request from its registration
    (its soft pin ends). The request then waits in a queue again, so its
    registration must count it again (a request is live while queued or running).
    Only a request whose registration is still the session's current version
    is bound again; an untagged, stale or still bound request changes nothing.
    Any tree cache type is accepted.
    """
    tracker = getattr(getattr(scheduler, "tree_cache", None), "kv_sessions", None)
    if tracker is not None:
        tracker.rebind_request(req)


def _aborted_before_admission(req) -> bool:
    """True if an admission check already aborted the request (to_finish or
    finished_reason is FINISH_ABORT). set_finish_with_abort replaces the input
    with [0], and the request runs no real prefill."""
    from decoupled_engine.srt.managers.schedule_batch import FINISH_ABORT

    return isinstance(getattr(req, "to_finish", None), FINISH_ABORT) or isinstance(
        getattr(req, "finished_reason", None), FINISH_ABORT
    )


@dataclass
class KvSessionStats:
    """Cumulative CPU counters of one worker's retention registry.

    Plain ints, changed only on the scheduler thread. TP0 logs them as the
    KV_SESSION_STATS line every 60 s and at shutdown; no GPU state is read.
    """

    tagged_arrivals: int = 0
    # Rejected tagged arrivals (the scheduler aborts the request), by reason.
    aborts: dict[str, int] = field(default_factory=dict)
    mark_untracked: int = 0
    mark_retired: int = 0
    # Live registrations whose anchor a storage/P2P import commit moved down.
    prefetch_extended: int = 0
    # Host tokens freed by the retention eviction, by (class, kind): class 0
    # has no session reference; kind "host" is a host-only leaf, "dup" the
    # host copy of a GPU-resident node.
    evicted_tokens: dict[tuple[int, str], int] = field(default_factory=dict)
    # Of those, tokens of nodes that a live request's registration covered
    # (host_session_active > 0). The soft pin makes them the last victims of
    # their class; with SGLANG_KV_SESSION_ACTIVE_SOFT_PIN=0 this counts the live-session
    # tokens that the soft pin would have kept.
    evicted_active_tokens: int = 0
    # Symmetric D legs that a late-D ADMIT tag (SGLANG_KV_SESSION_ADMIT_TAG)
    # bound to a live registration. A failed admit (counted in aborts) and a
    # fenced tag (an older version, an ended session) do not count. A gate
    # on D registration reads the live gauge; KV_SESSION_STATS shows this
    # counter when non-zero.
    admit_tags: int = 0
    # Class-1 host tokens freed, by the victim's idle age on the lockstep
    # clock (retention_order.IDLE_AGE_NAMES: 0-10, 10-60, 60-100, >= 100 s).
    # Kept in both orders (SGLANG_KV_SESSION_RETENTION_ORDER).
    evicted_c1_idle_tokens: list[int] = field(
        default_factory=lambda: [0] * len(IDLE_AGE_NAMES)
    )

    def note_abort(self, reason: str) -> None:
        self.aborts[reason] = self.aborts.get(reason, 0) + 1

    def note_host_eviction(
        self, rank: int, duplicate: bool, tokens: int, active: bool = False
    ) -> None:
        key = (rank, "dup" if duplicate else "host")
        self.evicted_tokens[key] = self.evicted_tokens.get(key, 0) + tokens
        if active:
            self.evicted_active_tokens += tokens

    def note_class1_idle(self, age_s: float, tokens: int) -> None:
        self.evicted_c1_idle_tokens[idle_age_bin(age_s)] += tokens


@dataclass
class _KvRegistration:
    """One version's identity, desired native key, and currently resident anchor."""

    session_id: str
    version: int
    input_key: object
    key: object
    anchor: object = None
    state: str = "pending"
    request_ids: set[str] = field(default_factory=set)
    wait_key: object = None
    # True while the anchor path carries this registration in
    # host_session_active, i.e. while request_ids is non-empty or the
    # registration is protected.
    active: bool = False
    # Handover protection (PROTECT control): this worker was the session's P
    # and now decodes, so its host copy may be the prefix the next P pulls.
    # The registration stays active (soft-pinned) without a local request
    # until its release, END or a newer version.
    protected: bool = False
    # Lockstep-clock time (s) at which a request of this registration last
    # finished on this worker; None before the first finish. The next
    # version's arrival minus this is a return gap (retention_order).
    last_done_s: Optional[float] = None


class KvSessionRetention:
    """Versioned retention hints, not memory locks or native session handles.

    The node counters count distinct local sessions, not requests or transfers.
    host_session_active counts the subset whose registration has at least one
    queued or running request on this worker (``request_ids`` non-empty).
    Released/ended records are retained to fence delayed network messages. At a
    configured bound, new identities fail closed instead of discarding fences.
    A controller should namespace session IDs across its own restarts.
    """

    def __init__(
        self,
        tree_core,
        apply_actions,
        max_sessions=100_000,
        active_soft_pin=True,
        order=ORDER_LRU,
        return_gaps=None,
        fixed_table=None,
    ):
        """Bind the native tree/action sink; allocate no per-token state.

        ``active_soft_pin`` (SGLANG_KV_SESSION_ACTIVE_SOFT_PIN) only selects
        the host eviction key; host_session_active is kept either way.
        ``order`` (SGLANG_KV_SESSION_RETENTION_ORDER, retention_order.py):
        "lru", "predicted_return_fixed" or "predicted_return", the order
        inside class 1. The lockstep clock, the return gaps and their
        counters run in every order.
        ``return_gaps``: a ReturnGapTable (tests: other thresholds).
        ``fixed_table``: the m(tau) table of predicted_return_fixed
        (retention_order.FixedReturnTable; None: the built-in table).
        """
        self.tree = tree_core
        self.apply_actions = apply_actions
        self.max_sessions = max_sessions
        self.active_soft_pin = active_soft_pin
        self.order = parse_retention_order(order)
        # Advanced only by the broadcast stamps (advance_clock): the same
        # seconds on every TP rank.
        self.clock = LockstepClock()
        self.return_gaps = return_gaps if return_gaps is not None else ReturnGapTable()
        self.fixed_table = None
        if self.order == ORDER_PREDICTED_RETURN_FIXED:
            self.fixed_table = (
                fixed_table if fixed_table is not None else DEFAULT_FIXED_TABLE
            )
        # The class-1 key uses -m(idle age): always with
        # predicted_return_fixed; with predicted_return once a table rebuild
        # saw >= return_gaps.min_gaps gaps. Changes only in advance_clock,
        # never inside an eviction.
        self.predicted_active = self.fixed_table is not None
        self.sessions: dict[str, _KvRegistration] = {}
        self.requests: dict[str, _KvRegistration] = {}
        # One entry per incomplete context, keyed by its missing next native
        # page. New materialization wakes only sessions interested in that
        # path, including sessions whose previous copy has been evicted.
        self.pending: dict[tuple, dict[str, _KvRegistration]] = {}
        self.stats = KvSessionStats()

    def stats_line(self) -> str:
        """The KV_SESSION_STATS payload: registry gauges and cumulative counters.

        One pass over the registry and one over the tree (CPU only); the
        scan_ms field reports what the pass cost the scheduler thread.
        """
        start = time.perf_counter()
        states = Counter(entry.state for entry in self.sessions.values())
        # Registrations under handover protection (shown only when non-zero).
        protected = sum(1 for entry in self.sessions.values() if entry.protected)
        known_zero, class0, active_nodes = self.tree.retention_host_scan()
        st = self.stats
        evicted = st.evicted_tokens
        reasons = ",".join(f"{k}:{v}" for k, v in sorted(st.aborts.items()))
        return (
            f"live={states['pending']} released={states['released']} "
            f"ended={states['ended']} tagged_arrivals={st.tagged_arrivals} "
            f"aborts={sum(st.aborts.values())} abort_reasons={reasons or '-'} "
            f"mark_untracked={st.mark_untracked} mark_retired={st.mark_retired} "
            f"known_zero_host_tokens={known_zero} "
            f"evicted_c0_host_only={evicted.get((0, 'host'), 0)} "
            f"evicted_c0_dup={evicted.get((0, 'dup'), 0)} "
            f"evicted_c1_host_only={evicted.get((1, 'host'), 0)} "
            f"evicted_c1_dup={evicted.get((1, 'dup'), 0)} "
            f"prefetch_extended={st.prefetch_extended} "
            f"soft_pin={int(self.active_soft_pin)} "
            f"class0_host_tokens={class0} active_nodes={active_nodes} "
            f"evicted_active_tokens={st.evicted_active_tokens} "
            f"scan_ms={(time.perf_counter() - start) * 1e3:.1f}"
            + (f" admit_tags={st.admit_tags}" if st.admit_tags else "")
            + (f" protected={protected}" if protected else "")
        )

    def order_stats_line(self) -> str:
        """The class-1 order fields of the KV_SESSION_STATS line (all orders).

        order / order_active (the predicted key is in use) / table (the
        m(tau) table of the key: label:digest of the fixed table, "online",
        or "-"), return_gaps (all gaps seen) / table_gaps (gaps of the current
        online table), m0..m110: the LEARNED expected remaining idle time (s)
        at idle age 0/30/60/90/110 s ("-" before the first rebuild, "inf"
        without enough longer gaps; with predicted_return_fixed the key uses
        the fixed table, see the startup line), the class-1 evicted tokens by
        idle age, and the lockstep clock (stamps applied, seconds since the
        first stamp).
        """
        gaps = self.return_gaps
        idle = self.stats.evicted_c1_idle_tokens
        return (
            f"order={self.order} order_active={int(self.predicted_active)} "
            f"table={self.table_label()} "
            f"return_gaps={gaps.n} table_gaps={gaps.table_gaps} "
            + " ".join(f"m{tau}={gaps.m_text(tau)}" for tau in STATS_TAUS)
            + " "
            + " ".join(
                f"c1_evicted_idle_{name}={idle[i]}"
                for i, name in enumerate(IDLE_AGE_NAMES)
            )
            + f" clock_stamps={self.clock.stamps} clock_s={self.clock.elapsed_s:.1f}"
        )

    def table_label(self) -> str:
        """The m(tau) table of the class-1 key: fixed label:digest, online, -."""
        if self.fixed_table is not None:
            return f"{self.fixed_table.label}:{self.fixed_table.digest}"
        return "online" if self.order == ORDER_PREDICTED_RETURN else "-"

    def order_identity(self) -> tuple[str, str]:
        """(order, fixed-table digest or "-"): equal on every TP rank
        (Scheduler.init_kv_session_retention checks it)."""
        digest = self.fixed_table.digest if self.fixed_table is not None else "-"
        return self.order, digest

    def order_startup_line(self) -> str:
        """The fields of the KV_SESSION_RETENTION startup line (TP0)."""
        gaps = self.return_gaps
        line = (
            f"order={self.order} soft_pin={int(self.active_soft_pin)} "
            f"clock=tp0_broadcast table={self.table_label()}"
        )
        if self.fixed_table is not None:
            ft = self.fixed_table
            dead = "-" if ft.dead_after_s is None else f"{ft.dead_after_s:g}"
            line += f" bins={len(ft.m_s)} dead_after_s={dead} " + " ".join(
                f"m{tau}={ft.m_text(tau)}" for tau in STATS_TAUS + (119, 120)
            )
        return line + (
            f" online_min_gaps={gaps.min_gaps} online_refresh_s={gaps.refresh_s:.0f}"
            f" online_bins={gaps.bins}"
            f" online_min_tail_fraction={gaps.min_tail_fraction:.2f}"
        )

    def advance_clock(self, t):
        """Every rank, from the broadcast KvSessionClockStamp: apply the
        source rank's time ``t`` (s) at this point of the request list, then
        rebuild the gap table if REFRESH_S lockstep seconds have passed."""
        from decoupled_engine.srt.mem_cache.unified_cache.components.tree_component import (
            peek_time_counter,
        )

        self.clock.advance(t, peek_time_counter())
        if (
            self.return_gaps.maybe_rebuild(self.clock.now_s)
            and self.order == ORDER_PREDICTED_RETURN
        ):
            self.predicted_active = self.return_gaps.ready

    def idle_age_s(self, tick) -> float:
        """Seconds since the host access that took ``tick`` (lockstep clock)."""
        return self.clock.age_s(tick)

    def expected_remaining_idle_s(self, tick) -> float:
        """m(idle age) of a node whose host access tick is ``tick``, from the
        fixed table or the online table; only while predicted_active."""
        age = self.clock.age_s(tick)
        if self.fixed_table is not None:
            return self.fixed_table.m_of_age(age)
        return self.return_gaps.m_of_age(age)

    def _return_gap(self, current):
        """Seconds from the last finish of ``current`` (the version that a
        newer one replaces) to now, or None: no live previous version, a
        request of it still open, no finish yet, or no clock yet."""
        if (
            current is None
            or not self.clock.started
            or current.last_done_s is None
            or current.request_ids
            or not self._live(current)
        ):
            return None
        return self.clock.now_s - current.last_done_s

    @staticmethod
    def _same_key(left, right):
        """Compare complete native content identities, including namespace/tail."""
        return (
            left.extra_key == right.extra_key
            and left.cache_salt == right.cache_salt
            and left.is_bigram == right.is_bigram
            and left.raw_token_ids() == right.raw_token_ids()
        )

    def _path(self, node):
        """Return the anchor-to-root path, excluding root; never inspect KV bytes."""
        path = []
        while node is not None and node is not self.tree.root_node:
            path.append(node)
            node = node.parent
        return path

    def _replace_anchor(self, entry, new_anchor):
        """Apply the symmetric path difference; shared prefixes keep their count.

        An active registration moves its host_session_active level the same way.
        """
        self._remove_waiter(entry)
        old_path, new_path = self._path(entry.anchor), self._path(new_anchor)
        old_set, new_set = set(old_path), set(new_path)
        for node in old_set - new_set:
            assert node.host_session_count > 0
            node.host_session_count -= 1
            if entry.active:
                assert node.host_session_active > 0
                node.host_session_active -= 1
        for node in new_set - old_set:
            node.host_session_count += 1
            node.host_session_known = True
            if entry.active:
                node.host_session_active += 1
        if entry.anchor is not None:
            anchors = entry.anchor.host_session_anchors
            if anchors is not None:
                anchors.discard(entry.session_id)
                if not anchors:
                    entry.anchor.host_session_anchors = None
        entry.anchor = new_anchor
        if new_anchor is not None and new_anchor is not self.tree.root_node:
            if new_anchor.host_session_anchors is None:
                new_anchor.host_session_anchors = set()
            new_anchor.host_session_anchors.add(entry.session_id)
        self._watch_missing(entry, sum(len(node.key) for node in new_path))

    def _set_active(self, entry, active):
        """Add/remove one host_session_active level along the current anchor path.

        Called when ``request_ids`` becomes non-empty or empty, and before an
        anchor leaves a registration without _replace_anchor. Every caller is
        a scheduler event, so all TP ranks apply the same changes.
        """
        if entry.active == active:
            return
        entry.active = active
        delta = 1 if active else -1
        for node in self._path(entry.anchor):
            assert active or node.host_session_active > 0
            node.host_session_active += delta

    def _remove_waiter(self, entry):
        """Remove only this exact version's missing-prefix subscription."""
        if entry.wait_key is None:
            return
        bucket = self.pending.get(entry.wait_key)
        if bucket is not None and bucket.get(entry.session_id) is entry:
            del bucket[entry.session_id]
            if not bucket:
                del self.pending[entry.wait_key]
        entry.wait_key = None

    def _watch_missing(self, entry, covered_tokens):
        """Index the next missing native page, including an empty/root prefix."""
        if (
            entry.key is None
            or entry.anchor is None
            or entry.state in ("released", "ended")
        ):
            return
        page = self.tree.page_size
        required = len(entry.key) // page * page
        if covered_tokens >= required:
            return
        next_page = entry.key[covered_tokens : covered_tokens + page]
        entry.wait_key = (entry.anchor.id, next_page.child_key(page))
        self.pending.setdefault(entry.wait_key, {})[entry.session_id] = entry

    def materialized(self, node_id):
        """Attach pending interests when *any* request creates their missing KV.

        Input is the real insert/import endpoint after the native operation.
        Snapshot matching buckets before locating contexts: locating may split
        an edge and re-index a registration. No full registry scan is needed.
        """
        if node_id is None or not self.pending:
            return
        node = self.tree.node_by_id(node_id)
        candidates = {}
        for child in self._path(node):
            # Every non-root waiter is indexed on its anchor. Skip native
            # page-key construction on unrelated ancestors; a long prompt
            # fragmented into many edges must not be re-hashed per chunk.
            if (
                child.parent is not self.tree.root_node
                and not child.parent.host_session_anchors
            ):
                continue
            watch_key = (child.parent.id, child.key.child_key(self.tree.page_size))
            for entry in self.pending.get(watch_key, {}).values():
                candidates[id(entry)] = entry
        for entry in candidates.values():
            if self._live(entry):
                self._replace_anchor(entry, self._locate(entry.key, entry))

    def _locate(self, key, previous=None):
        """Reuse a proven old prefix, then find/split only the changed suffix."""
        start, offset = None, 0
        if (
            previous is not None
            and previous.anchor is not None
            and previous.key is not None
            and previous.key.extra_key == key.extra_key
            and previous.key.cache_salt == key.cache_salt
        ):
            # Native array matching compares growing slices in C; this is
            # cheap even for a long shared prefix fragmented into many nodes.
            common = (
                len(key)
                if previous.key is key
                else previous.key.match(key, page_size=self.tree.page_size)
            )
            path = self._path(previous.anchor)
            offset = sum(len(node.key) for node in path)
            start = previous.anchor
            while start is not self.tree.root_node and offset > common:
                offset -= len(start.key)
                start = start.parent
        node, actions = self.tree.locate_retention_prefix(
            key, start_node=start, start_offset=offset
        )
        self.apply_actions(actions)
        return node

    def admit(self, req, input_key):
        """Bind a request to an immutable version; newer input replaces old coverage.

        ``input_key`` contains the *full* input, before cache-hit/SWA/chunk caps.
        Same-version retries must have identical content. Old requests keep an
        object token, so their later insert/prefetch callbacks cannot revive a
        released, ended, or replaced registration.
        """
        try:
            metadata = parse_kv_session(getattr(req, "kv_session", None))
        except ValueError:
            self.stats.note_abort("invalid")
            raise
        if metadata is None:
            return None
        self.stats.tagged_arrivals += 1
        if _aborted_before_admission(req):
            # init_req_max_new_tokens does not return after it aborts (a
            # forced output that does not fit, a non-greedy request on a
            # synthetic-oracle server), so the request arrives here with input
            # [0]. Admitting it would retire the session's real context (count
            # 0, so class 0) and make a retry with the real input fail with
            # version_reused. The request leaves without KV: do not admit it.
            self.stats.note_abort("pre_aborted")
            return None
        sid, version = metadata
        current = self.sessions.get(sid)
        if current is not None:
            if current.state == "ended" or version < current.version:
                self._bind_request(
                    req, _KvRegistration(sid, version, None, None, state="released")
                )
                return None
            if version == current.version:
                if current.state == "released":
                    self._bind_request(req, current)
                    return None
                if not self._same_key(current.input_key, input_key):
                    self.stats.note_abort("version_reused")
                    raise ValueError("kv_session version reused for different input")
                self._bind_request(req, current)
                return current
        elif len(self.sessions) >= self.max_sessions:
            self.stats.note_abort("capacity")
            raise ValueError("KV session retention registry capacity exceeded")

        gap = self._return_gap(current)
        node = self._locate(input_key, current)
        entry = _KvRegistration(sid, version, input_key, input_key)
        # Update atomically in the scheduler thread. No allocator or network
        # operation can observe the transient decrement/increment here.
        if current is not None:
            # The new version takes over the anchor and its counts, not the
            # old version's live requests or protection: their active level
            # ends here.
            current.protected = False
            self._set_active(current, False)
            entry.anchor = current.anchor
            current.anchor = None
            self._retire(current, "released")
        self._replace_anchor(entry, node)
        self.sessions[sid] = entry
        self._bind_request(req, entry)
        if gap is not None:
            # A return of this agent to this worker (retention_order).
            self.return_gaps.add(gap)
        return entry

    def _bind_request(self, req, entry):
        """Capture live or fenced version identity for later asynchronous results."""
        self.requests[req.rid] = entry
        entry.request_ids.add(req.rid)
        req._kv_session_registration = entry
        self._set_active(entry, True)

    def rebind_request(self, req):
        """Bind a request that waits again (retraction, preemption, requeue).

        cache_finished_req dropped it from its registration when its KV was
        released. If that registration is still the session's current
        version, the request counts as live again, and the active level
        returns to the anchor path. A stale, released or ended registration,
        or a request that is still bound, changes nothing.
        """
        entry = getattr(req, "_kv_session_registration", None)
        if self._live(entry) and req.rid not in entry.request_ids:
            self._bind_request(req, entry)

    def _live(self, entry):
        """Return whether a callback's object token is still the current version."""
        return (
            entry is not None
            and self.sessions.get(entry.session_id) is entry
            and entry.state not in ("released", "ended")
        )

    def prepare_host_return(self, metadata, key, prompt_tokens):
        """Bind a proactive D->P return to its existing local registration.

        The P request may already have finished; its session still owns a native
        input identity. Returns cannot create sessions, advance their version,
        replace a different output branch, or revive a release/END. Recording the
        desired suffix only references bytes already materialized locally; the
        normal missing-page index and prefetch commit attach later arrivals.
        """
        parsed = parse_kv_session(metadata)
        if parsed is None:
            return None
        sid, version = parsed
        entry = self.sessions.get(sid)
        if not self._live(entry) or entry.version != version:
            return None
        original, current = entry.input_key, entry.key
        if (
            type(prompt_tokens) is not int
            or len(original.raw_token_ids()) != prompt_tokens
            or len(key) < max(len(original), len(current))
        ):
            return None
        # Compare raw prefixes: even a zero-length bigram prompt has one
        # boundary token, which a logical key[:0] slice would discard.
        raw = key.raw_token_ids()
        for prefix in (original, current):
            prefix_raw = prefix.raw_token_ids()
            if (
                prefix.extra_key != key.extra_key
                or prefix.cache_salt != key.cache_salt
                or prefix.is_bigram != key.is_bigram
                or prefix_raw != raw[: len(prefix_raw)]
            ):
                return None
        node = self._locate(key, entry)
        entry.key = key
        self._replace_anchor(entry, node)
        return entry

    def _advance_anchor(self, entry, node_id):
        """Use an existing insert/import frontier; do not rescan the full prompt."""
        if node_id is None:
            return
        node = self.tree.node_by_id(node_id)
        if node is entry.anchor:
            return
        # A capped chunk/match may end above our existing context anchor. It
        # supplies no evidence of compaction: only admit(new version) may
        # shrink or change the branch.
        old = entry.anchor
        current = node
        while current is not None and current is not old:
            current = current.parent
        if old is None or current is old:
            self._replace_anchor(entry, node)

    def refresh_request(self, req, *, finished=False, key=None, node_id=None):
        """Attach newly inserted exact content; ignore all stale request callbacks.

        A finished request may extend its context by actual generated tokens.
        Intermediate chunks never replace the full desired input with a shorter
        fill buffer. The final callback also drops the request lookup entry.
        """
        entry = getattr(req, "_kv_session_registration", None)
        if self._live(entry):
            if (
                finished
                and key is not None
                and node_id is not None
                and len(key) >= len(entry.input_key)
            ):
                # A same-input retry can produce a different output suffix.
                # Its real final insert identifies that suffix; do not leave
                # an old output anchor paired with the new key in status().
                entry.key = key
                self._replace_anchor(entry, self.tree.node_by_id(node_id))
            else:
                self._advance_anchor(entry, node_id)
        if finished:
            if self.requests.get(req.rid) is entry:
                self.requests.pop(req.rid, None)
            if entry is not None:
                entry.request_ids.discard(req.rid)
                self._set_active(entry, bool(entry.request_ids) or entry.protected)
                if self.clock.started:
                    entry.last_done_s = self.clock.now_s

    def prefetch_committed(self, entry, node_id=None, *, expected_key=None):
        """Use captured version and, for returns, desired-context identity.

        ``materialized`` may still attach current interests to the same bytes;
        an obsolete return itself must not move a session onto its old branch.
        """
        if self._live(entry) and (expected_key is None or entry.key is expected_key):
            anchor = entry.anchor
            self._advance_anchor(entry, node_id)
            if entry.anchor is not anchor:
                self.stats.prefetch_extended += 1
        elif node_id is not None:
            node = self.tree.node_by_id(node_id)
            if entry is not None:
                self.mark_retired(node)
            else:
                self.mark_untracked(node)

    def mark_untracked(self, node):
        """Keep paths used without lifecycle metadata UNKNOWN after tracked END."""
        self.stats.mark_untracked += 1
        for ancestor in self._path(node):
            ancestor.host_session_unknown = True

    def mark_retired(self, node):
        """Classify late tagged results as known, without restoring references."""
        self.stats.mark_retired += 1
        for ancestor in self._path(node):
            ancestor.host_session_known = True

    def on_delete(self, node):
        """Move affected frontier anchors to their surviving parent before unlink.

        The session ids are visited in sorted order. A set of str can iterate
        in a different order in each TP rank process, because str hashing is
        randomized per process unless the hash seed is fixed. Each move
        re-indexes the session's missing-page wait, and materialized() later
        splits the new content once per waiter, in that order. So the set
        order would decide which split fragment gets which node id: a TP
        digest and eviction tie-break difference between ranks."""
        for sid in sorted(node.host_session_anchors or ()):
            entry = self.sessions[sid]
            self._replace_anchor(entry, node.parent)

    def reset_storage(self):
        """Invalidate active callbacks on cache flush while retaining message fences."""
        for entry in self.sessions.values():
            # The tree reset dropped every node and its counters.
            entry.anchor = None
            entry.active = False
            entry.protected = False
            entry.key = entry.input_key = None
            entry.request_ids.clear()
            entry.wait_key = None
            if entry.state != "ended":
                entry.state = "released"
        self.requests.clear()
        self.pending.clear()

    def _retire(self, entry, state):
        """Drop prompt/request memory but keep the small version/lifetime fence."""
        entry.protected = False
        self._set_active(entry, False)
        self._replace_anchor(entry, None)
        entry.state = state
        entry.input_key = entry.key = None
        for rid in entry.request_ids:
            if self.requests.get(rid) is entry:
                self.requests.pop(rid, None)
        entry.request_ids.clear()

    def control(self, operation):
        """Apply one status/release/end/protect operation; return bounded JSON metadata."""
        op = operation.get("op")
        if op not in ("status", "release", "end", "protect"):
            raise ValueError("unknown KV session operation")
        sid, version = parse_kv_session(
            {
                "session_id": operation.get("session_id"),
                "version": operation.get("version"),
            }
        )
        entry = self.sessions.get(sid)
        reply = {
            "session_id": sid,
            "version": version,
            "ready_tokens": 0,
            "required_tokens": 0,
        }
        if op == "end":
            if entry is None:
                if len(self.sessions) >= self.max_sessions:
                    raise ValueError("KV session retention registry capacity exceeded")
                entry = _KvRegistration(sid, version, None, None, state="ended")
                self.sessions[sid] = entry
            else:
                self._retire(entry, "ended")
            reply["status"] = "ended"
            return reply
        if op == "release" and (entry is None or entry.version < version):
            # A RELEASE may overtake a queued request on this worker. Fence the
            # version even when no local arrival has executed yet; otherwise
            # a later callback can recreate the supposedly retired placement.
            if entry is None:
                if len(self.sessions) >= self.max_sessions:
                    raise ValueError("KV session retention registry capacity exceeded")
            elif entry.state == "ended":
                reply["status"] = "ended"
                return reply
            else:
                self._retire(entry, "released")
            entry = _KvRegistration(sid, version, None, None, state="released")
            self.sessions[sid] = entry
        if entry is None:
            reply["status"] = "missing"
            return reply
        if entry.state == "ended":
            reply["status"] = "ended"
            return reply
        if version != entry.version:
            reply["status"] = "stale"
            return reply
        if op == "release":
            self._retire(entry, "released")
        if entry.state == "released":
            reply["status"] = "released"
            return reply
        if op == "protect":
            # Only this exact live version; a newer placement, release or END
            # answers above and never revives an old registration.
            entry.protected = True
            self._set_active(entry, True)
            reply["status"] = "protected"
            return reply
        required = len(entry.key) // self.tree.page_size * self.tree.page_size
        ready = 0
        for node in reversed(self._path(entry.anchor)):
            if not node.backuped or node.write_through_pending_id is not None:
                break
            ready += len(node.key)
        reply.update(
            required_tokens=required,
            ready_tokens=min(ready, required),
            status="ready" if ready >= required else "pending",
        )
        return reply


@dataclass(kw_only=True)
class UnifiedSessionRefTracker:
    """Tags radix KV by session id; ``release_radix_session`` (close) releases a
    session's tagged reference. Each component maintains its own session_ids,
    session_ref and so on."""

    components: tuple[TreeComponent, ...]
    tree_core: UnifiedTreeCore
    enable_session_radix_cache: bool

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._closed_session_ids: OrderedDict[str, None] = OrderedDict()
        self._session_incarnation_counter: int = 0
        self._session_generations: dict[str, int] = {}
        for component in self.components:
            component.reset_session_state()

    def session_id_for_req(self, req: Req) -> Optional[str]:
        session_id = req.session_id
        if session_id is None and req.session is not None:
            session_id = req.session.session_id
        return session_id

    def register_session_ref(self, req: Req) -> None:
        """Register a non-streaming request's reusable leaves with each component."""
        if not self.enable_session_radix_cache:
            return

        session = req.session
        if session is not None and session.streaming:
            return

        session_id = self.session_id_for_req(req)
        if session_id is None or session_id in self._closed_session_ids:
            return

        current_generation = self._session_generations.get(session_id)
        if current_generation is None or req.session_generation != current_generation:
            logger.warning("register_session_ref called for stale request; Skip it.")
            return

        assert req.last_node is not None
        last_node = self.tree_core.node_by_id(req.last_node)
        if last_node is self.tree_core.root_node:
            return

        for component in self.components:
            leaf = component.resolve_session_leaf(req, last_node)
            component.register_session_leaf(session_id, leaf)

    def _remember_closed_session(self, session_id: str) -> None:
        self._closed_session_ids[session_id] = None
        self._closed_session_ids.move_to_end(session_id)
        while len(self._closed_session_ids) > _CLOSED_SESSION_TOMBSTONE_LIMIT:
            self._closed_session_ids.popitem(last=False)

    def open_radix_session(self, session_id: str) -> Optional[int]:
        self._closed_session_ids.pop(session_id, None)
        self._session_incarnation_counter += 1
        self._session_generations[session_id] = self._session_incarnation_counter
        return self._session_incarnation_counter

    def ensure_session_generation(self, session_id: str) -> int:
        generation = self._session_generations.get(session_id)
        if generation is None:
            generation = self.open_radix_session(session_id)
        return generation

    def release_radix_session(self, session_id: str) -> int:
        if not self.enable_session_radix_cache or session_id is None:
            return 0

        if session_id in self._closed_session_ids:
            return 0

        self._remember_closed_session(session_id)
        self._session_generations.pop(session_id, None)

        indexed = 0
        for component in self.components:
            indexed += component.release_session(session_id)

        logger.info(
            "release_session %s: indexed %d component leaves",
            session_id,
            indexed,
        )
        return 0
