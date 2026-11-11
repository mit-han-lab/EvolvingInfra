"""Demote the source copy of a P2P-pulled prefix (``--peer-kv-demote-source``).

After a peer pull, the session's prefix lives on two prefill workers: the reader
holds the pulled copy, and the owner (this worker) still holds its own copy at
plain LRU rank. When sessions move between workers, these copies double the
per-worker host footprint. The host pool then turns over faster, so an idle
session loses its prefix on every worker.

A demoted node is evicted before any other host leaf (strict priority in the
FULL host-leaf heap and in the duplicate reclaim). The flags work like this:

1. The reader sends ``RELEASE {complete: true, pages: n}`` only after it has
   read every leased page of every pool (``_PeerLease.is_complete``).
2. The owner's servicer, on the TP rank that sources the request broadcast,
   puts ``(page keys[:n], lookup tick)`` in an outbox. Other ranks keep no
   outbox. (With host dedup, only that rank serves peers.)
3. On that rank, the scheduler thread drains the outbox before the request
   broadcast. It finds the deepest leased node with the last leased page key,
   then walks toward the root (``select_demote_path``):

   - Flag a node only if (a) it has no child outside the leased path, and
     (b) it had no local access since the lookup.
   - Stop at the first node that fails (the first branch point).
   - A lease that ends inside a node flags nothing: the rest of that node is
     outside the leased path, so the first branch point is inside it.

4. The node ids go first in the broadcast request list as a ``PeerKVDemoteReq``.
   Every rank applies them before any tree operation of that step, so the
   flags, and therefore the host eviction order, are the same on every rank.
   No rank-local state steers the tree, so TP ranks cannot diverge.
5. A later local access to a flagged node (a match through it, an insert or a
   touch) clears its flag. A split copies the flag to both parts; the access
   that caused the split then clears the part it reached. A host eviction of a
   flagged node also clears it and counts as a demoted eviction.
6. Counters (``PeerDemoteStats``, logged by the resolver every 60 s when they
   change) include the flags cleared by a local access, the demoted copies
   evicted, the returns of an evicted demoted copy (pulled back from a peer, or
   prefilled again here) and the flagged pages still outstanding.

Hardening (each part off by default):

- ``SGLANG_PEER_KV_DEMOTE_AFTER_COMMIT`` (reader): the demote waits until the
  reader committed the pulled pages to its tree (DEMOTE_OK after the commit).
  The commit is insert_host at the prefetch completion
  (``UnifiedRadixCache._handle_prefetch_result``, host memory mode "cache"),
  for the leases of that prefetch attempt only. A pull that is aborted after
  its reads, discarded (hybrid check), whose insert is dropped, or that runs
  in buffer_only mode (no insert into the tree) sends no DEMOTE_OK, so the
  owner demotes nothing; the reader counts it by cause
  (``PeerKVBackend.COMMIT_DROP_CAUSES``, log "PeerKV demote commit (reader)"),
  and the owner's pending entry expires (``commit_expired``).
- ``SGLANG_PEER_KV_DEMOTE_LOCK_AWARE`` (owner): the walk also stops at a node
  with a host lock (outcome ``locked``): a queued local request, a prefetch
  anchor or a backup holds it, so it is in local use.
- ``SGLANG_PEER_KV_DEMOTE_COUNTERS`` (owner): the flag age at eviction and at
  a clear (histograms, ``AGE_EDGES_S``), the causes of ``no_node``, outbox
  drops, one trace record per flagged eviction, and the flagged node ids in the
  TP structural digest.
"""

from __future__ import annotations

import bisect
import dataclasses
import json
import logging
import queue
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Access ticks are whole numbers. The lookup peeks the next tick, so an access
# at or after the lookup has a tick >= lookup tick. A match writes its tick to
# the deepest node and the tick minus 1e-5 per level to the ancestors, so half
# a tick separates "at or after the lookup" from "before". (A margin of one
# tick would count the owner's own last access as later when the owner was
# idle until the lookup, and nothing would be flagged.)
LOOKUP_TICK_MARGIN = 0.5

OUTCOMES = ("root", "branch", "access", "children", "no_host", "locked")

# Flag-age histogram bucket edges in seconds (SGLANG_PEER_KV_DEMOTE_COUNTERS):
# bucket i counts ages below AGE_EDGES_S[i]; the last one, ages at or above
# the last edge.
AGE_EDGES_S = (5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0)


def age_bucket(age_s: float) -> int:
    for i, edge in enumerate(AGE_EDGES_S):
        if age_s < edge:
            return i
    return len(AGE_EDGES_S)


def _age_line(name: str, counts: List[int], edges=AGE_EDGES_S) -> str:
    labels = [f"<{e:g}" for e in edges] + [f">={edges[-1]:g}"]
    return f"{name}=" + ",".join(f"{lb}:{c}" for lb, c in zip(labels, counts))


# SGLANG_PEER_KV_DEMOTE_COUNTERS. The trigger of a mark:
# "release", a complete RELEASE (legacy); "commit", DEMOTE_OK (AFTER_COMMIT);
# "notice", a NOTICE.
TRIGGERS = ("release", "commit", "notice")
# SGLANG_PEER_KV_DEMOTE_NOTICE. A mark (DEMOTE_OK, NOTICE)
# older than this at the reader's send, or a NOTICE whose age_ms is above it
# at the owner, marks nothing.
NOTICE_MAX_AGE_S = 30.0
# A NOTICE carries at most this many page keys, ending at the owner's
# confirmed end.
NOTICE_MAX_KEYS = 512
# Why a request that is eligible for a NOTICE sends none (the insert gate): its
# KV was not inserted with write-through.
NOTICE_SKIP_CAUSES = ("skip_insert", "gpu_only", "frozen", "abort")


def notice_insert_cause(req: Any) -> Optional[str]:
    """The insert gate of a NOTICE: None when ``req`` inserted
    its computed KV with write-through at its prefill end; else the cause
    (``NOTICE_SKIP_CAUSES``). It reads request fields only (the same on every
    rank)."""
    if getattr(req, "decode_radix_cache_skip_unfinished_insert", False):
        return "frozen"
    if getattr(req, "skip_radix_cache_insert", False):
        return "skip_insert"  # bootstrap_host 2.2.2.2 (probes only)
    if getattr(req, "hicache_skip_write_through", False):
        return "gpu_only"
    from decoupled_engine.srt.managers.schedule_batch import FINISH_ABORT

    for name in ("to_finish", "finished_reason"):
        if isinstance(getattr(req, name, None), FINISH_ABORT):
            return "abort"
    return None


# Finer age buckets (seconds) for the age at host eviction by class; the last
# bucket also takes ages older than the tick clock's window.
AGE_FINE_EDGES_S = (5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180, 240, 300)
# RELEASE -> DEMOTE_OK lag buckets (seconds), owner side.
LAG_EDGES_S = (0.5, 1, 2, 5, 10, 20, 30, 60)
# A return (re-pull / rebackup / recompute) of an evicted marked copy is
# binned by the time from the mark and by the size of the mark (tokens).
RETURN_EARLY_S = 60.0
BIG_MARK_TOKENS = 400_000


def _bucket(value: float, edges) -> int:
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges)


def age_fine_bucket(age_s: Optional[float]) -> int:
    """None (older than the tick clock's window) goes to the last bucket."""
    return len(AGE_FINE_EDGES_S) if age_s is None else _bucket(age_s, AGE_FINE_EDGES_S)


def lag_bucket(lag_s: float) -> int:
    return _bucket(lag_s, LAG_EDGES_S)


def return_bin(kind: str, since_mark_s: float, mark_tokens: int) -> str:
    """The counter key part of a return: <kind>_<lt60|ge60>[_big]."""
    early = "lt60" if since_mark_s < RETURN_EARLY_S else "ge60"
    big = "_big" if mark_tokens >= BIG_MARK_TOKENS else ""
    return f"{kind}_{early}{big}"


class DemoteOutboxEntry(tuple):
    """A demote outbox entry (keys, lookup tick, rid) that also carries its
    trigger. It still equals the plain tuple."""

    def __new__(cls, items, trigger: str):
        entry = super().__new__(cls, items)
        entry.trigger = trigger
        return entry


def release_trigger(release: Any) -> str:
    return getattr(release, "trigger", "release")


class RecordWriter:
    """SGLANG_PEER_KV_DEMOTE_RECORDS: JSON lines through a
    bounded queue and one daemon thread. put() never blocks the scheduler
    thread: a full queue drops the record (counted in ``dropped``)."""

    def __init__(self, path: str, maxsize: int = 8192):
        self.path = path
        self.dropped = 0
        self.written = 0
        self.errors = 0
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._file = open(path, "a", buffering=1)
        self._thread = threading.Thread(
            target=self._run, name="peer-kv-records", daemon=True
        )
        self._thread.start()

    def put(self, record: dict) -> None:
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    @staticmethod
    def _write(f, record: dict) -> None:
        f.write(json.dumps(record, sort_keys=True) + "\n")

    def _run(self) -> None:
        while True:
            record = self._queue.get()
            if record is None:
                return
            try:
                self._write(self._file, record)
                self.written += 1
            except Exception:  # a record must never stop the writer
                self.errors += 1
                if self.errors == 1:
                    logger.warning("PeerKV record write failed", exc_info=True)

    def close(self, timeout: float = 10.0) -> None:
        """Write what is queued, then stop (tests and shutdown)."""
        self._queue.put(None, timeout=timeout)
        self._thread.join(timeout)
        self._file.close()


class TickClock:
    """TP0 tick -> time ring: (tick, monotonic) samples, one
    per pass when the access counter moved, at most every ``min_interval_s``,
    the last ``size`` kept. ``time_of(tick)`` is the time of the last sample
    at or before ``tick`` (bisection): at most one sample interval (plus a
    pass) before the access. Counters and records only, never a key."""

    def __init__(self, size: int = 4096, min_interval_s: float = 0.1):
        self._ticks: List[float] = []
        self._times: List[float] = []
        self._size = int(size)
        self._min_interval_s = float(min_interval_s)

    def __len__(self) -> int:
        return len(self._ticks)

    def sample(self, tick: float, now: float) -> None:
        if self._ticks and (
            tick <= self._ticks[-1] or now - self._times[-1] < self._min_interval_s
        ):
            return
        self._ticks.append(float(tick))
        self._times.append(float(now))
        if len(self._ticks) > self._size:
            del self._ticks[0], self._times[0]

    def time_of(self, tick: float) -> Optional[float]:
        i = bisect.bisect_right(self._ticks, tick) - 1
        return self._times[i] if i >= 0 else None

    def age_s(self, tick: float, now: float) -> Optional[float]:
        t = self.time_of(tick)
        return None if t is None else max(0.0, now - t)


@dataclasses.dataclass
class PeerKVDemoteReq:
    """Broadcast from the request-source TP rank. Node ids are the same on every
    rank because nodes are created in the same order on every rank."""

    node_ids: List[int]
    releases: int = 0


@dataclasses.dataclass
class PeerDemoteStats:
    """Counters, cumulative since start. Each rank applies the same flags, but
    only the request-source rank logs them. The outstanding values are
    computed from the live flag set when a line is logged."""

    releases: int = 0  # complete RELEASEs drained from the outbox
    # The last leased page is not the last page of a live published node: the
    # lease ended inside a node (its other pages are outside the leased path,
    # so that is the first branch point) or the node is gone.
    no_node: int = 0
    root: int = 0  # the walk reached the root: the whole path is flagged
    branch: int = 0  # the walk stopped at a branch point
    access: int = 0  # the walk stopped at a node with a local access
    children: int = 0  # the deepest leased node has children
    no_host: int = 0  # a node on the path has no host copy
    flagged_nodes: int = 0
    flagged_pages: int = 0
    cleared_access_nodes: int = 0
    cleared_access_pages: int = 0
    evicted_nodes: int = 0  # host copy dropped while flagged
    evicted_pages: int = 0
    # After a demoted copy was evicted, its last page was published again on
    # this worker: pulled back from a peer (re-pull) or prefilled again here.
    repull_events: int = 0
    repull_nodes: int = 0
    repull_pages: int = 0
    recompute_nodes: int = 0
    recompute_pages: int = 0

    def line(self, outstanding_nodes: int, outstanding_pages: int) -> str:
        return (
            f"releases={self.releases} no_node={self.no_node} root={self.root} "
            f"branch={self.branch} access={self.access} children={self.children} "
            f"no_host={self.no_host} flagged_nodes={self.flagged_nodes} "
            f"flagged_pages={self.flagged_pages} "
            f"cleared_access_nodes={self.cleared_access_nodes} "
            f"cleared_access_pages={self.cleared_access_pages} "
            f"evicted_nodes={self.evicted_nodes} evicted_pages={self.evicted_pages} "
            f"repull_events={self.repull_events} repull_nodes={self.repull_nodes} "
            f"repull_pages={self.repull_pages} "
            f"recompute_nodes={self.recompute_nodes} "
            f"recompute_pages={self.recompute_pages} "
            f"outstanding_nodes={outstanding_nodes} "
            f"outstanding_pages={outstanding_pages}"
        )

    def key(self) -> Tuple[int, ...]:
        return dataclasses.astuple(self)


@dataclasses.dataclass
class PeerDemoteExtraStats:
    """Counters of the hardening flags (resolver rank). Kept apart from
    ``PeerDemoteStats`` so that its stats line stays the same with the flags
    off; ``line`` holds only the fields of the flags that are on."""

    # SGLANG_PEER_KV_DEMOTE_LOCK_AWARE: the walk stopped at a host-locked node.
    locked: int = 0
    # SGLANG_PEER_KV_DEMOTE_COUNTERS. The causes of no_node: the last leased
    # page is inside a live node (the lease ended inside it), no live node
    # holds it (evicted, or never published here), or no node end shows among
    # the leased keys.
    no_node_inside: int = 0
    no_node_gone: int = 0
    no_node_unresolved: int = 0
    # Copied from the servicer at each drain: entries that the bounded demote
    # and touch outboxes dropped, and the commit-gated demotes
    # (SGLANG_PEER_KV_DEMOTE_AFTER_COMMIT): pending, confirmed by DEMOTE_OK,
    # expired without one, DEMOTE_OK for no pending lease.
    outbox_drops: int = 0
    touch_outbox_drops: int = 0
    commit_pending: int = 0
    commit_ok: int = 0
    commit_expired: int = 0
    commit_unknown: int = 0
    # Flag age (nodes) when the flag ended by a host eviction / by a clear.
    age_evict: List[int] = dataclasses.field(
        default_factory=lambda: [0] * (len(AGE_EDGES_S) + 1)
    )
    age_clear: List[int] = dataclasses.field(
        default_factory=lambda: [0] * (len(AGE_EDGES_S) + 1)
    )
    trace_records: int = 0  # flagged-eviction trace records written
    # COUNTERS, resolver rank. "<trigger>.<name>" counters:
    # releases and walk outcomes; marked nodes / pages; the fate of marked
    # nodes (cleared_*, evicted_left_* / evicted_gpu_kept_*); returns after an
    # eviction in pages (repull / rebackup / recompute, return_bin()).
    trig: Dict[str, int] = dataclasses.field(default_factory=dict)
    marks_leaf: int = 0  # marked nodes with no child
    marks_dormant: int = 0  # marked inner nodes (act once they are leaves)
    marks_gpu_pages: int = 0  # marked pages whose node was GPU-resident
    class2_gpu_evicted_nodes: int = 0  # device frees of marked nodes
    class2_gpu_evicted_tokens: int = 0
    # Reader side: committed landings (DEMOTE_OK), and those host-evicted
    # with no access since the landing (before their request used them).
    landing_commits: int = 0
    landing_evicted_unused: int = 0
    # Copied from the servicer at each drain: RELEASE -> DEMOTE_OK lag.
    commit_lag: List[int] = dataclasses.field(
        default_factory=lambda: [0] * (len(LAG_EDGES_S) + 1)
    )
    commit_lag_max_ms: int = 0
    # Age at host eviction (tick clock), class 1 and class 2 by trigger.
    age_host_c1: List[int] = dataclasses.field(
        default_factory=lambda: [0] * (len(AGE_FINE_EDGES_S) + 1)
    )
    age_host_c2: Dict[str, List[int]] = dataclasses.field(default_factory=dict)
    # NOTICE, copied from the servicer at each drain: NOTICEs
    # received, queued for the walk, unknown (no switch or NOTICE off at the
    # owner), dropped (no age_ms / age_ms > 30 s / bad keys or tick), held
    # now by the grace (gauge) and dropped while held (flush). Resolver: a
    # held NOTICE whose walk stopped at a local use (E3, "access").
    notice_received: int = 0
    notice_queued: int = 0
    notice_unknown: int = 0
    notice_dropped_no_age: int = 0
    notice_dropped_age: int = 0
    notice_dropped_bad: int = 0
    notice_held: int = 0
    notice_held_dropped: int = 0
    notice_grace_access: int = 0
    # Owner role check, copied from the servicer: mark messages that a
    # decode-launched owner ignored (shown only when not 0).
    ignored_role: int = 0

    NOTICE_FIELDS = (
        "received",
        "queued",
        "unknown",
        "dropped_no_age",
        "dropped_age",
        "dropped_bad",
        "held",
        "held_dropped",
        "grace_access",
    )

    def bump(self, key: str, n: int = 1) -> None:
        self.trig[key] = self.trig.get(key, 0) + n

    def line(self, lock_aware: bool, counters: bool) -> str:
        parts = []
        if lock_aware:
            parts.append(f"locked={self.locked}")
        if counters:
            parts.append(
                f"no_node_inside={self.no_node_inside} "
                f"no_node_gone={self.no_node_gone} "
                f"no_node_unresolved={self.no_node_unresolved} "
                f"outbox_drops={self.outbox_drops} "
                f"touch_outbox_drops={self.touch_outbox_drops} "
                f"commit_pending={self.commit_pending} "
                f"commit_ok={self.commit_ok} "
                f"commit_expired={self.commit_expired} "
                f"commit_unknown={self.commit_unknown} "
                f"trace_records={self.trace_records} "
                + _age_line("age_evict_s", self.age_evict)
                + " "
                + _age_line("age_clear_s", self.age_clear)
            )
            parts.append(
                f"marks_leaf={self.marks_leaf} marks_dormant={self.marks_dormant} "
                f"marks_gpu_pages={self.marks_gpu_pages} "
                f"class2_gpu_evicted_nodes={self.class2_gpu_evicted_nodes} "
                f"class2_gpu_evicted_tokens={self.class2_gpu_evicted_tokens} "
                f"landing_commits={self.landing_commits} "
                f"landing_evicted_unused={self.landing_evicted_unused} "
                f"commit_lag_max_ms={self.commit_lag_max_ms} "
                + _age_line("commit_lag_s", self.commit_lag, LAG_EDGES_S)
                + " "
                + _age_line("age_host_c1_s", self.age_host_c1, AGE_FINE_EDGES_S)
            )
            for trigger in sorted(self.age_host_c2):
                parts.append(
                    _age_line(
                        f"age_host_c2_{trigger}_s",
                        self.age_host_c2[trigger],
                        AGE_FINE_EDGES_S,
                    )
                )
            parts.append(
                "trig=" + ",".join(f"{k}:{v}" for k, v in sorted(self.trig.items()))
            )
        elif self.trig:
            # Without COUNTERS only the SCOPE "scope.*" counters
            # (SGLANG_PEER_KV_DEMOTE_SCOPE) are in trig.
            parts.append(
                "trig=" + ",".join(f"{k}:{v}" for k, v in sorted(self.trig.items()))
            )
        if self.ignored_role:
            parts.append(f"demote_ignored_role={self.ignored_role}")
        notice = [getattr(self, "notice_" + k) for k in self.NOTICE_FIELDS]
        if any(notice):  # NOTICE traffic only: the line is unchanged without
            parts.append(
                " ".join(f"notice_{k}={v}" for k, v in zip(self.NOTICE_FIELDS, notice))
            )
        return " ".join(parts)

    def key(self) -> Tuple[Any, ...]:
        return dataclasses.astuple(self)


def host_locked(node: Any) -> bool:
    """A component of ``node`` holds a host lock (queued-prefix guard, prefetch
    anchor, backup): the copy is in local use."""
    return any(cd.host_lock_ref > 0 for cd in node.component_data)


def device_locked(node: Any) -> bool:
    """A component of ``node`` holds a device lock (a running request)."""
    return any(cd.lock_ref > 0 for cd in node.component_data)


# SGLANG_PEER_KV_DEMOTE_SCOPE.
SCOPE_MODES = ("off", "shadow", "on")
SCOPE_OUTCOMES = ("root", "live_sibling", "offpath", "queued", "locked", "access")
# At most this many queued requests feed the queued set of one drain.
SCOPE_QUEUED_MAX = 4096


def scope_mode(value: Optional[str]) -> str:
    mode = (value or "off").strip().lower()
    if mode not in SCOPE_MODES:
        raise ValueError(
            f"SGLANG_PEER_KV_DEMOTE_SCOPE={value!r}: use one of {SCOPE_MODES}"
        )
    return mode


@dataclasses.dataclass
class ScopeContext:
    """One drain's scope inputs, built on the resolver rank
    before the walk: the mode, the matched-end node ids of the queued
    requests (E2) and those nodes with their ancestors (E1)."""

    apply: bool  # SCOPE=on: the v3 walk's nodes are flagged
    ends: frozenset = frozenset()
    paths: frozenset = frozenset()
    truncated: bool = False


def select_scope_path(
    start: Any, child: Any, lookup_tick: float, root: Any, scope: ScopeContext
) -> Tuple[List[Any], str]:
    """SCOPE: the scope walk from the covered
    start. ``child``: the path child below ``start`` when the covered end is
    inside it (that node stays unmarked), else None. Flag a node only if it
    has a host copy and passes:

    - E1: every class-1 child other than the path child is older than the
      path child, unlocked and not on a queued request's path (class-2
      children never block). Below the start the children are uncovered
      tails: they block only when locked or queued (else a dormant mark).
    - E2: no host or device lock; not a queued request's matched end.
    - E3: no access at or after the lookup tick (LOOKUP_TICK_MARGIN).
    - E4: not used after its path child (tick(n) > tick(c) stops).

    Stop at the first node that fails. Returns (nodes deepest first, outcome:
    one of SCOPE_OUTCOMES or "no_host"). Read-only."""
    out: List[Any] = []
    node, c = start, child
    while node is not None and node is not root:
        if getattr(node, "deleted", False) or not node.backuped:
            return out, "no_host"
        if host_locked(node) or device_locked(node):
            return out, "locked"
        if node.id in scope.ends:
            return out, "queued"
        if node.last_access_time >= lookup_tick - LOOKUP_TICK_MARGIN:
            return out, "access"
        if c is not None and node.last_access_time > c.last_access_time:
            return out, "offpath"
        for x in node.children.values():
            if x is c or getattr(x, "deleted", False) or x.peer_demoted:
                continue
            if host_locked(x) or device_locked(x) or x.id in scope.paths:
                return out, "live_sibling"
            if c is not None and x.last_access_time >= c.last_access_time:
                return out, "live_sibling"
        if not node.peer_demoted:
            out.append(node)
        c, node = node, node.parent
    return out, "root"


def select_demote_path(
    deepest: Any, lookup_tick: float, root: Any, lock_aware: bool = False
) -> Tuple[List[Any], str]:
    """The nodes to flag for one complete pull, deepest first, and the reason
    the walk stopped (one of ``OUTCOMES``).

    ``deepest`` is the node whose last page is the last leased page. The reader
    now holds every page on the path from the root to it: the leased pages it
    pulled, and the earlier pages that were its local hit (the lease starts
    there). Walk from there toward the root. Flag a node only if:

    (a) it has no child outside the leased path. The deepest node may have no
        child; an ancestor may have only the path child.
    (b) with ``lock_aware``, it has no host lock (``host_locked``).
    (c) it had no access at or after the lookup tick (``LOOKUP_TICK_MARGIN``).

    Stop at the first node that fails. The root is never flagged. A node that
    is already flagged passes through and is not returned again.
    """
    out: List[Any] = []
    node, path_child = deepest, None
    while node is not None and node is not root:
        if getattr(node, "deleted", False) or not node.backuped:
            return out, "no_host"
        allowed_children = 0 if path_child is None else 1
        if len(node.children) > allowed_children:
            return out, "children" if path_child is None else "branch"
        if lock_aware and host_locked(node):
            return out, "locked"
        if node.last_access_time >= lookup_tick - LOOKUP_TICK_MARGIN:
            return out, "access"
        if not node.peer_demoted:
            out.append(node)
        path_child, node = node, node.parent
    return out, "root"
