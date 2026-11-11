"""Owner LRU refresh ("touch") for deferred hinted peer pulls (pull-miss fix F1).

Opt-in: ``SGLANG_PEER_KV_TOUCH_DEFERRED_PULLS`` (default off). Set it on every
prefill worker: each one is a destination of pulls and an owner.

Why: a hinted request can wait tens
of seconds between the router's decision and the LOOKUP of its pull (the
bootstrap queue, then the JIT deferral). Meanwhile the owner's host LRU ages
the hinted prefix out, and the pull misses although the hint was right.

How:

1. Destination, attention-TP rank 0 only (``JitPull`` with
   ``--hicache-storage-jit-pull queue``): when the pull of a hinted request is
   deferred, it sends TOUCH to the owner's servicer, and again every
   ``SGLANG_PEER_KV_TOUCH_INTERVAL_S`` while the pull stays deferred. It stops
   when the pull is issued (lead scan or admission), when the request ends,
   or after ``SGLANG_PEER_KV_TOUCH_MAX_AGE_S``. A TOUCH carries the page keys
   of the last ``SGLANG_PEER_KV_TOUCH_MAX_KEYS`` pages of the hinted span: the
   namespaced keys that a LOOKUP of those pages sends. A background thread
   (``TouchSender``) hashes once per request and sends, best effort; the
   scheduler thread only queues.
2. Owner servicer, request-broadcast source rank only: TOUCH confirms the
   published prefix of the keys in the shadow index (no pin, no lease) and
   queues the confirmed keys. The servicer thread never reads or changes the
   tree, so TP ranks cannot diverge.
3. Owner scheduler, the same rank, before the request broadcast: the cache
   resolves each queued key list to the deepest node that holds its last
   confirmed page (``resolve_touch_node``) and puts a ``PeerKVTouchReq`` with
   the node ids in the request list, after a demote request and before the
   tokenizer requests.
4. Every rank applies it at the same point of the step: each node and its
   ancestors get a fresh access time, as a prefix match gives them, and a
   demote flag (``--peer-kv-demote-source``) is cleared, as a local access
   clears it. So the host eviction order stays the same on every rank.

Order with the pull (lever 1, ``--peer-kv-demote-source``): a TOUCH refreshes
the owner only BEFORE the LOOKUP of its own pull. After the transfer the
complete RELEASE demotes the source copy as without F1, and no TOUCH of that
pull refreshes it again:

- Destination: the issue of the pull stops its TOUCHes. A stopped span that
  waits in the sender queue is dropped (``TouchSpan.stopped``), and the sweep
  runs after the lead scan, so a re-touch due in the issue pass is not queued.
- Owner servicer: a TOUCH carries the request id; a LOOKUP carries it in its
  request metadata. A LOOKUP drops the queued TOUCHes of its request
  (``touch_dropped_by_lookup``) and is remembered for the lease TTL + 60 s; a
  TOUCH of a remembered request is not queued (``touch_after_lookup``). Both
  run on the servicer thread before the LOOKUP takes its lookup tick.
- Owner scheduler: a TOUCH drained before the LOOKUP can still be applied
  after the LOOKUP took its tick (the drain and the apply are in the same step,
  around the request broadcast). So the drain reserves one access tick per
  drained TOUCH, from the next tick on, and every rank refreshes the nodes with
  these ticks (unique, so the eviction order has no ties) and then moves the
  counter past them. A LOOKUP takes its lookup tick at or after the end of the
  reserved ticks (``PeerKVServicer._lookup_tick``), so the demote of that pull
  is never stopped by a TOUCH drained before its LOOKUP.

A TOUCH of another request's pending pull on the same path is not blocked: it
refreshes the path before ITS transfer, and the complete RELEASE of that
transfer demotes the path again. When that pull ends without a transfer
(aborted, failed, lease expired), nothing demotes the path again: there was
no transfer, so there is no source refresh after a transfer.

Limits: the refresh sets the access time that orders the FULL host leaves
(and the uniform-lifecycle host eviction). It does not move nodes in a
separate component host LRU (non-uniform SWA or mamba host copies). A touch
keeps the whole deepest node, also its pages after the hinted span.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

import msgspec
import zmq

from decoupled_engine.srt.mem_cache.peer_kv_namespace import peer_page_keys

logger = logging.getLogger(__name__)

_enc = msgspec.msgpack.Encoder()
_dec = msgspec.msgpack.Decoder()


@dataclasses.dataclass
class PeerKVTouchReq:
    """Broadcast from the request-source TP rank. Node ids are the same on every
    rank because nodes are created in the same order on every rank. ``tick``:
    the first of the ``touches`` access ticks that the drain reserved (the same
    values on every rank); None takes a fresh tick per node. A drain that
    resolves no node still sends the request, so that every rank moves its
    counter past the reserved ticks."""

    node_ids: List[int]
    touches: int = 0
    tick: Optional[float] = None


@dataclasses.dataclass
class PeerTouchStats:
    """Owner-side counters, cumulative since start. Every rank applies the same
    requests; only the request-source rank resolves and logs."""

    touches: int = 0  # confirmed TOUCH key lists drained from the outbox
    resolved: int = 0
    # No published node ends among the keys and the list does not start at
    # the first page, or the node is gone.
    unresolved: int = 0
    from_root: int = 0  # resolved through the root's children
    nodes: int = 0  # node ids broadcast (unique per drain)
    applied: int = 0  # nodes refreshed (with their ancestors)
    cleared_demote: int = 0  # demote flags cleared by a touch
    # Servicer counters (copied at each drain): TOUCHes not queued because
    # their request already sent its LOOKUP, and queued TOUCHes dropped by the
    # LOOKUP of their request.
    touch_after_lookup: int = 0
    touch_dropped_by_lookup: int = 0

    def line(self) -> str:
        return " ".join(
            f"{f.name}={getattr(self, f.name)}" for f in dataclasses.fields(self)
        )

    def key(self) -> Tuple[int, ...]:
        return dataclasses.astuple(self)


def touch_page_keys(
    token_ids: Any,
    num_pages: int,
    *,
    page_size: int,
    is_bigram: bool = False,
    extra_key: Optional[str] = None,
    cache_salt: Optional[str] = None,
    max_keys: int = 512,
) -> Tuple[List[str], int]:
    """``(keys, first_page)``: the namespaced page keys of pages
    ``[first_page, num_pages)`` of ``token_ids``, with ``first_page =
    max(0, num_pages - max_keys)``. The page hashes are chained from the first
    page of the sequence (as the owner's tree computes them), so they equal the
    keys that a LOOKUP of those pages sends."""
    from decoupled_engine.srt.mem_cache.radix_cache import RadixKey
    from decoupled_engine.srt.mem_cache.utils import get_hash_str

    extra = 1 if is_bigram else 0
    num_pages = min(int(num_pages), max(0, len(token_ids) - extra) // page_size)
    if num_pages <= 0 or max_keys <= 0:
        return [], 0
    key = RadixKey(
        token_ids,
        extra_key=extra_key,
        is_bigram=is_bigram,
        limit=num_pages * page_size + extra,
        cache_salt=cache_salt,
    )
    hashes = get_hash_str(key, None, page_size=page_size)
    first = max(0, len(hashes) - int(max_keys))
    return peer_page_keys(hashes[first:], cache_salt, extra_key), first


def first_page_key(node: Any) -> Optional[str]:
    """The namespaced key of a published node's first page, or None."""
    if not node.hash_value or node.key is None:
        return None
    keys = peer_page_keys(node.hash_value[:1], node.key.cache_salt, node.key.extra_key)
    return keys[0]


def resolve_touch_node(
    keys: List[str],
    from_root: bool,
    last_key_index: Dict[str, Any],
    root: Any,
    first_key_of: Callable[[Any], Optional[str]] = first_page_key,
) -> Tuple[Optional[Any], bool]:
    """``(node, via_root)``: the deepest live node that holds the page of
    ``keys[-1]``, or None.

    ``keys``: consecutive page keys of one path; the last one is the deepest
    confirmed page. ``last_key_index``: the resolver's map from the last page
    key of each published node to the node. ``from_root``: ``keys[0]`` is the
    first page of the sequence.

    The walk goes from the deepest key toward the first. The first key found in
    the index is the end of a node N. If that is the deepest key, N holds it.
    If not, the deeper pages are in the child of N whose first page has the
    next key (N itself when that child is gone). When no node ends among the
    keys, a list that starts at the first page resolves through the root's
    children. Any other list is not resolved: its deepest node starts before
    the first key."""
    n = len(keys)
    for j in range(n - 1, -1, -1):
        node = last_key_index.get(keys[j])
        if node is None or node.deleted:
            continue
        if j == n - 1:
            return node, False
        child = _child_with_first_key(node, keys[j + 1], first_key_of)
        return (node if child is None else child), False
    if from_root and n:
        return _child_with_first_key(root, keys[0], first_key_of), True
    return None, False


def _child_with_first_key(
    parent: Any, key: str, first_key_of: Callable[[Any], Optional[str]]
) -> Optional[Any]:
    for child in parent.children.values():
        if not child.deleted and first_key_of(child) == key:
            return child
    return None


# The live child of ``parent`` whose first page has ``key``, or None (also used
# by the demote no_node cause, unified_tree_core._peer_demote_no_node_cause).
child_with_first_key = _child_with_first_key


class TouchSpan:
    """The hinted span of one deferred request. The sender thread hashes it once
    (``page_keys``) and drops the token reference; the scheduler thread only
    creates it, and sets ``stopped`` when the pull is issued or the request
    ends (the sender then drops the span)."""

    __slots__ = (
        "_tokens",
        "num_pages",
        "page_size",
        "is_bigram",
        "extra_key",
        "cache_salt",
        "max_keys",
        "keys",
        "first_page",
        "rid",
        "stopped",
    )

    def __init__(
        self,
        tokens: Any,
        num_pages: int,
        *,
        page_size: int,
        is_bigram: bool = False,
        extra_key: Optional[str] = None,
        cache_salt: Optional[str] = None,
        max_keys: int = 512,
        rid: str = "",
    ):
        self._tokens = tokens
        self.num_pages = int(num_pages)
        self.page_size = int(page_size)
        self.is_bigram = bool(is_bigram)
        self.extra_key = extra_key
        self.cache_salt = cache_salt
        self.max_keys = int(max_keys)
        self.keys: Optional[List[str]] = None
        self.first_page = 0
        self.rid = rid if isinstance(rid, str) else ""
        self.stopped = False

    def page_keys(self) -> List[str]:
        if self.keys is None:
            self.keys, self.first_page = touch_page_keys(
                self._tokens,
                self.num_pages,
                page_size=self.page_size,
                is_bigram=self.is_bigram,
                extra_key=self.extra_key,
                cache_salt=self.cache_salt,
                max_keys=self.max_keys,
            )
            self._tokens = None
        return self.keys


class TouchSender:
    """Background thread: sends TOUCH {hashes, first_page, rid} over its own
    REQ socket per owner address (never the LOOKUP sockets). Best effort: a
    timeout drops the socket, a full queue drops the touch, a stopped span is
    dropped (its pull was issued or its request ended), and nothing here
    reaches a scheduling decision. Counters are written by one thread each
    and read for logs only."""

    QUEUE_MAX = 1024
    _WARN_INTERVAL_S = 60.0

    def __init__(self, timeout_ms: int = 1000):
        self._timeout_ms = int(timeout_ms)
        self._queue: Queue = Queue()
        self._sockets: Dict[str, zmq.Socket] = {}
        self._ctx = zmq.Context.instance()
        self._last_warn = 0.0
        # Sender thread only.
        self.sent = 0
        self.ok = 0
        self.failed = 0
        self.pages = 0  # pages that the owners confirmed
        self.stopped = 0  # stopped spans dropped before the send
        self.blocked = 0  # the owner had the LOOKUP of the request already
        self.touch_timeouts = 0  # no reply within the poll (also in failed)
        # Scheduler thread only.
        self.dropped = 0
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="peer-kv-touch"
        )
        self._thread.start()

    def submit(self, addr: str, span: TouchSpan) -> bool:
        if self._queue.qsize() >= self.QUEUE_MAX:
            self.dropped += 1
            return False
        self._queue.put((addr, span))
        return True

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5)

    def line(self) -> str:
        return (
            f"sent={self.sent} ok={self.ok} failed={self.failed} "
            f"pages={self.pages} dropped={self.dropped} stopped={self.stopped} "
            f"blocked={self.blocked} timeouts={self.touch_timeouts}"
        )

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                try:
                    self._send(*item)
                except TimeoutError:
                    # The owner did not answer within the poll (an owner
                    # stall): expected and handled, so no traceback.
                    self.failed += 1
                    self.touch_timeouts += 1
                    self._warn(
                        "peer_kv touch: no reply from %s within %d ms "
                        "(timeouts=%d; socket dropped)",
                        item[0],
                        self._timeout_ms,
                        self.touch_timeouts,
                        exc_info=False,
                    )
                except Exception:
                    self.failed += 1
                    self._warn("peer_kv touch: send to %s failed", item[0])
        finally:
            for sock in self._sockets.values():
                sock.close(linger=0)
            self._sockets.clear()

    def _send(self, addr: str, span: TouchSpan) -> None:
        if span.stopped:
            self.stopped += 1
            return
        keys = span.page_keys()
        if not keys:
            return
        if span.stopped:  # stopped while the keys were hashed
            self.stopped += 1
            return
        sock = self._sockets.get(addr)
        if sock is None:
            sock = self._ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(f"tcp://{addr}")
            self._sockets[addr] = sock
        self.sent += 1
        try:
            sock.send(
                _enc.encode(
                    {
                        "op": "touch",
                        "hashes": keys,
                        "first_page": span.first_page,
                        "rid": span.rid,
                    }
                )
            )
            if sock.poll(self._timeout_ms, zmq.POLLIN) == 0:
                raise TimeoutError("no reply")
            reply = _dec.decode(sock.recv())
        except Exception:
            # The REQ socket is stuck mid-state: drop it; the next touch to
            # this owner opens a new one.
            sock.close(linger=0)
            self._sockets.pop(addr, None)
            raise
        if isinstance(reply, dict) and reply.get("ok"):
            self.ok += 1
            pages = reply.get("pages")
            self.pages += pages if type(pages) is int and pages > 0 else 0
            self.blocked += int(reply.get("after_lookup") is True)
        else:
            self.failed += 1

    def _warn(self, fmt: str, *args: Any, exc_info: bool = True) -> None:
        now = time.monotonic()
        if now - self._last_warn >= self._WARN_INTERVAL_S:
            self._last_warn = now
            logger.warning(fmt, *args, exc_info=exc_info)
