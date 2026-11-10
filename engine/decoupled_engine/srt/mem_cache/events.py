# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Adapted from SGLang python/sglang/srt/mem_cache/events.py (commit 9a489f8d2f), modified for evolving-infra.
"""KV cache placement event recording.

Produces the ``BlockStored`` / ``BlockRemoved`` / ``AllBlocksCleared`` events
consumed by KV-aware routers (e.g. dynamo). A cache holds one recorder and calls
it; the recorder owns the queue and needs nothing back from its owner.

Lazy load-back stores (SGLANG_KV_EVENTS_LAZY_LOAD_BACK):
``record_store(..., lazy=True)`` queues one O(1) entry per node (its
token-id array, logical length, bigram flag, cache_salt, hash values, the
parent block hash computed now, medium) in place of the per-page events, and
``take`` builds them in queue order with the same coalescing, so the stream is
identical to the eager one. This relies on radix keys and hash lists never
being mutated in place: splits and slices make new objects. With lazy mode on,
the recorder logs one INFO line per ~60 s: entries queued, entries checked
against their commit-time events (SGLANG_KV_EVENTS_LAZY_LOAD_BACK_CHECK) and
mismatches since the previous line, with totals.

Publication segments (SGLANG_KV_EVENTS_BATCH_PUBLISH): ``mark_boundary`` closes
the queued segment where a ``take`` used to run; nothing queued later coalesces
into it. ``take_closed`` then returns the closed segments, built and coalesced
each on its own: the concatenation of what one ``take`` per boundary returned,
byte for byte. Events after the last boundary stay queued.

D-role nodes of a symmetric worker (SGLANG_SYMMETRIC_SKIP_D_ROLE_KV_EVENTS):
a ``DRoleEventGate`` set as the recorder's ``d_role_gate`` drops the stores and
removals of the radix nodes that D-role work created; see its docstring. Without
a gate the recorder is unchanged.
"""

import contextlib
import logging
import time
from typing import Any, Callable, Iterable, Optional

from decoupled_engine.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    BlockStoredMetadata,
    BlockStoredWithMetadata,
    StorageMedium,
)
from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.mem_cache.utils import (
    compute_node_event_hash_values,
    compute_node_hash_values,
    hash_str_to_int64,
)

logger = logging.getLogger(__name__)

# 6f: seconds between the lazy-mode counter lines (logged from take() and lazy
# record_store calls once the interval has passed).
LAZY_LOG_INTERVAL_S = 60.0


class _LazyStore:
    """A node's store events, captured at record time and built at take()."""

    __slots__ = (
        "raw",
        "logical_len",
        "is_bigram",
        "cache_salt",
        "hash_values",
        "parent_block_hash",
        "medium",
        "check",
    )

    def __init__(
        self,
        raw,
        logical_len,
        is_bigram,
        cache_salt,
        hash_values,
        parent_block_hash,
        medium,
    ):
        self.raw = raw
        self.logical_len = logical_len
        self.is_bigram = is_bigram
        self.cache_salt = cache_salt
        self.hash_values = hash_values
        self.parent_block_hash = parent_block_hash
        self.medium = medium
        self.check = None  # commit-time events (SGLANG_KV_EVENTS_LAZY_LOAD_BACK_CHECK)


class _SegmentBoundary:
    """Queue marker that ends a publication segment (mark_boundary). It is no
    event type, so ``_coalesce`` appends after it and never merges into it."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<kv-events segment boundary>"


_BOUNDARY = _SegmentBoundary()

# Node attribute of SGLANG_SYMMETRIC_SKIP_D_ROLE_KV_EVENTS: True on a radix node
# that D-role work created (UnifiedTreeNode has the class default False).
D_ROLE_ATTR = "kv_events_d_role"


class DRoleEventGate:
    """SGLANG_SYMMETRIC_SKIP_D_ROLE_KV_EVENTS: no KV events for the radix nodes
    that D-role work of a symmetric worker created.

    A node is D-role when the tree adds it inside ``d_role()`` (the cache calls
    of a D-leg request: its import's prompt insert, its decode and finish
    inserts, a local recompute), and both fragments of a split D-role node are
    D-role. Every other node is P-role. The recorder asks the gate before each
    store and removal, on every TP rank, so the flags are the same on all ranks
    (only TP0 publishes):

    - A D-role node publishes nothing: no BlockStored and no BlockRemoved, on
      any tier (its insert, the host copy's ack, a load-back, its evictions).
    - A P-role node publishes exactly as without the gate.
    - Mixed path. Before the first store of a P-role node whose parent is
      D-role (P-role work inserting below a D-role prefix), ``promote`` makes
      the D-role ancestors P-role, root side first, and each publishes one
      BlockStored per tier it holds now (``held_tiers``: GPU while it holds
      device KV, CPU once its host copy is acked). The cache also promotes the
      last node of every insert made outside ``d_role()``, so a P-role insert
      that only walks a D-role prefix (a full hit) publishes that prefix too.
      A P-role request that matches a D-role prefix without inserting
      publishes nothing.

    Hence every published store links to a published parent (or the root), and
    every published removal follows a published store of its block. A promoted
    node stays P-role. D-role work on a P-role node (an import that reuses a
    P-role prefix, its refill or load-back) publishes as before, so the
    P-role stream is never cut.
    """

    def __init__(
        self,
        recorder: "KVCacheEventRecorder",
        held_tiers: Callable[[Any], Iterable[StorageMedium]],
    ):
        self.recorder = recorder
        self.held_tiers = held_tiers
        self.depth = 0  # > 0 while D-role work runs (d_role())
        # Totals since start, every rank (the publish-stats line reads TP0's).
        self.skipped_stores = 0
        self.skipped_removes = 0
        self.promoted_nodes = 0
        self.promotion_stores = 0

    @contextlib.contextmanager
    def d_role(self):
        """Nodes the tree adds meanwhile are D-role."""
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1

    def mark_new(self, node: Any) -> None:
        """The tree added ``node`` (before its first store)."""
        if self.depth:
            setattr(node, D_ROLE_ATTR, True)

    def allow_store(self, node: Any) -> bool:
        if getattr(node, D_ROLE_ATTR, False):
            self.skipped_stores += 1
            return False
        parent = node.parent
        if parent is not None and getattr(parent, D_ROLE_ATTR, False):
            self.promote(parent)
        return True

    def allow_remove(self, node: Any) -> bool:
        if getattr(node, D_ROLE_ATTR, False):
            self.skipped_removes += 1
            return False
        return True

    def promote(self, node: Any) -> None:
        """Make ``node`` and its D-role ancestors P-role, root side first, each
        publishing the tiers it holds. O(1) on a P-role node."""
        chain = []
        while node is not None and getattr(node, D_ROLE_ATTR, False):
            chain.append(node)
            node = node.parent
        if not chain:
            return
        publish = self.recorder.enabled
        for n in reversed(chain):
            setattr(n, D_ROLE_ATTR, False)
            self.promoted_nodes += 1
            if publish:
                for medium in self.held_tiers(n):
                    self.promotion_stores += 1
                    self.recorder.record_store(n, medium=medium)

    def counters(self) -> tuple:
        return (
            self.skipped_stores,
            self.skipped_removes,
            self.promoted_nodes,
            self.promotion_stores,
        )


class KVCacheEventRecorder:
    """Collects KV placement events for one cache.

    ``enabled=False`` makes every ``record_*`` call a no-op and ``take`` return an
    empty list, so callers never have to guard.
    """

    def __init__(self, *, enabled: bool, page_size: int):
        self.enabled = enabled
        self.page_size = page_size
        self._queue: list = []
        # Bigram (EAGLE) keys make a page's radix identity depend on the next
        # page's first token, so two chains that diverge exactly at a page
        # boundary cache the last shared page twice.  With this flag both
        # copies publish the same block hash (chained over the page's own
        # tokens), and ``_variants`` tracks which copies (by their bigram
        # ``hash_value``) back each published (medium, block hash).
        self.unigram_bigram = enabled and envs.SGLANG_KV_EVENTS_UNIGRAM_BLOCK_HASH.get()
        self._variants: dict = {}
        # 6f: lazy load-back stores (entries pending in the queue).
        self.lazy_load_back = enabled and envs.SGLANG_KV_EVENTS_LAZY_LOAD_BACK.get()
        self.lazy_check = (
            self.lazy_load_back and envs.SGLANG_KV_EVENTS_LAZY_LOAD_BACK_CHECK.get()
        )
        self.lazy_fast = (
            self.lazy_load_back and envs.SGLANG_KV_EVENTS_LAZY_FAST_BUILD.get()
        )
        self.lazy_fast_built = 0
        # Check mode with the fast build: entries whose merged build was
        # compared with the per-page build coalesced, and the mismatches.
        self.lazy_fast_checked = 0
        self.lazy_fast_mismatches = 0
        self._lazy_pending = 0
        self.lazy_entries = 0
        self.lazy_checked = 0
        self.lazy_mismatches = 0
        self._lazy_n = [0, 0, 0]  # entries, checked, mismatches since the last line
        self._lazy_log_t = time.monotonic()
        # Publication segments (mark_boundary / take_closed): queue items up
        # to the last boundary, and the lazy entries among them. Both stay 0
        # unless mark_boundary runs (SGLANG_KV_EVENTS_BATCH_PUBLISH).
        self._closed = 0
        self._lazy_closed = 0
        # SGLANG_SYMMETRIC_SKIP_D_ROLE_KV_EVENTS: a DRoleEventGate, set on
        # every TP rank by the cache (enabled or not); None: no gate.
        self.d_role_gate: Optional[DRoleEventGate] = None

    def enqueue(self, event) -> None:
        """Append an event, coalescing it with a compatible queue tail.

        KV event batches already support multiple block hashes.  Combining them
        here avoids emitting one event per page while preserving ordering and
        the parent-linked store chains consumers use to rebuild the cache tree.
        """
        self._coalesce(self._queue, event)

    @staticmethod
    def _coalesce(queue: list, event) -> None:
        # A lazy entry as the tail is neither event type: append after it;
        # take() replays the queue through here once the entry is built.
        if queue:
            tail = queue[-1]

            if isinstance(tail, BlockRemoved) and isinstance(event, BlockRemoved):
                if tail.medium == event.medium:
                    tail.block_hashes.extend(event.block_hashes)
                    return

            elif isinstance(tail, BlockStored) and isinstance(event, BlockStored):
                tail_metadata = (
                    tail.metadata if isinstance(tail, BlockStoredWithMetadata) else None
                )
                event_metadata = (
                    event.metadata
                    if isinstance(event, BlockStoredWithMetadata)
                    else None
                )
                if (
                    tail.medium == event.medium
                    and tail.lora_id == event.lora_id
                    and tail.block_size == event.block_size
                    and tail_metadata == event_metadata
                    and tail.block_hashes
                    and event.parent_block_hash == tail.block_hashes[-1]
                ):
                    tail.block_hashes.extend(event.block_hashes)
                    tail.token_ids.extend(event.token_ids)
                    return

        queue.append(event)

    def _uses_event_hash(self, node: Any) -> bool:
        return node.key.cache_salt is not None or (
            self.unigram_bigram and node.key.is_bigram
        )

    def _node_event_hash_values(self, node: Any) -> list:
        """Hash values to publish for ``node``, computing them if not yet set."""
        if node.hash_value is None:
            node.hash_value = compute_node_hash_values(node, self.page_size)
        if self._uses_event_hash(node):
            return compute_node_event_hash_values(
                node, self.page_size, unigram_bigram=self.unigram_bigram
            )
        return node.hash_value

    def _parent_block_hash(self, node: Any) -> Optional[int]:
        """The hash the first page of ``node`` links back to.

        ``None`` when the parent is the tree root: a root carries an empty
        ``hash_value`` and no event hash, so it contributes no link. Every other
        node on the path has a parent, which is what distinguishes the two.
        """
        parent = node.parent
        if parent is None or parent.parent is None:
            return None
        if self._uses_event_hash(node):
            parent_hash_values = parent.event_hash_value
            assert parent_hash_values is not None
        else:
            parent_hash_values = parent.hash_value
        if not parent_hash_values:
            return None
        return hash_str_to_int64(parent_hash_values[-1])

    def record_store(self, node: Any, medium=None, lazy: bool = False) -> None:
        # One BlockStored per ``page_size`` chunk.
        # ``medium`` defaults to StorageMedium.GPU but callers may override
        # for lower-tier insertions (e.g. StorageMedium.CPU for host/L2 cache).
        # ``lazy`` (host->device load-back): with SGLANG_KV_EVENTS_LAZY_LOAD_BACK
        # the events are built at take(); otherwise (or for unigram bigram
        # pages, whose variant bookkeeping must happen now) as below.
        gate = self.d_role_gate
        if gate is not None and not gate.allow_store(node):
            return
        if not self.enabled:
            return
        if medium is None:
            medium = StorageMedium.GPU

        event_hash_values = self._node_event_hash_values(node)
        parent_block_hash = self._parent_block_hash(node)
        is_bigram = node.key.is_bigram
        unigram = self.unigram_bigram and is_bigram
        if lazy and self.lazy_load_back and not unigram:
            entry = _LazyStore(
                node.key.token_ids,
                len(node.key),
                is_bigram,
                node.key.cache_salt,
                event_hash_values,
                parent_block_hash,
                medium,
            )
            if self.lazy_check:
                entry.check = list(self._lazy_events(entry))
            self._queue.append(entry)
            self._lazy_pending += 1
            self.lazy_entries += 1
            self._lazy_n[0] += 1
            self._lazy_tick()
            return

        page_index = 0
        logical_len = len(node.key)
        raw = node.key.token_ids
        for start in range(0, logical_len, self.page_size):
            end = min(start + self.page_size, logical_len)
            if end <= start:
                continue
            # Preserve historical event payload: bigram pages expose tuples,
            # unless the block hash is the page's unigram hash.
            if is_bigram and not unigram:
                page_tokens = [(raw[j], raw[j + 1]) for j in range(start, end)]
            else:
                page_tokens = list(raw[start:end])

            block_hash = hash_str_to_int64(event_hash_values[page_index])
            if unigram:
                self._variants.setdefault((medium, block_hash), set()).add(
                    node.hash_value[page_index]
                )

            event_args = {
                "block_hashes": [block_hash],
                "parent_block_hash": parent_block_hash,
                "token_ids": page_tokens,
                "block_size": len(page_tokens),
                "lora_id": None,
                "medium": medium,
            }
            if node.key.cache_salt is None:
                event = BlockStored(**event_args)
            else:
                event = BlockStoredWithMetadata(
                    **event_args,
                    metadata=BlockStoredMetadata(cache_salt=node.key.cache_salt),
                )
            self.enqueue(event)

            parent_block_hash = block_hash
            page_index += 1

    def record_remove(self, node: Any, medium=None) -> None:
        # One BlockRemoved per radix node.
        # ``medium`` defaults to StorageMedium.GPU but callers may override for
        # lower-tier removals (e.g. StorageMedium.CPU when evicting from host).
        gate = self.d_role_gate
        if gate is not None and not gate.allow_remove(node):
            return
        if not self.enabled:
            return
        if medium is None:
            medium = StorageMedium.GPU

        # Hash values must match what was stored.
        event_hash_values = self._node_event_hash_values(node)

        block_hashes = []
        logical_len = len(node.key)
        unigram = self.unigram_bigram and node.key.is_bigram
        page_index = 0
        for start in range(0, logical_len, self.page_size):
            end = min(start + self.page_size, logical_len)
            if end <= start:
                continue

            block_hash = hash_str_to_int64(event_hash_values[page_index])
            if unigram:
                variants = self._variants.get((medium, block_hash))
                if variants is not None:
                    variants.discard(node.hash_value[page_index])
                    if variants:
                        # Another bigram variant of this page is still cached
                        # on this tier; the published block stays valid.
                        page_index += 1
                        continue
                    del self._variants[(medium, block_hash)]
            block_hashes.append(block_hash)
            page_index += 1

        if block_hashes:
            self.enqueue(BlockRemoved(block_hashes=block_hashes, medium=medium))

    def record_all_cleared(self) -> None:
        if not self.enabled:
            return
        self._variants.clear()
        self.enqueue(AllBlocksCleared())

    def _lazy_events(self, entry: _LazyStore):
        """The events ``record_store`` would have queued for ``entry``'s node
        at record time (no unigram bigram pages: those are never lazy)."""
        raw = entry.raw
        logical_len = entry.logical_len
        parent_block_hash = entry.parent_block_hash
        page_index = 0
        for start in range(0, logical_len, self.page_size):
            end = min(start + self.page_size, logical_len)
            if end <= start:
                continue
            if entry.is_bigram:
                page_tokens = [(raw[j], raw[j + 1]) for j in range(start, end)]
            else:
                page_tokens = list(raw[start:end])
            block_hash = hash_str_to_int64(entry.hash_values[page_index])
            event_args = {
                "block_hashes": [block_hash],
                "parent_block_hash": parent_block_hash,
                "token_ids": page_tokens,
                "block_size": len(page_tokens),
                "lora_id": None,
                "medium": entry.medium,
            }
            if entry.cache_salt is None:
                yield BlockStored(**event_args)
            else:
                yield BlockStoredWithMetadata(
                    **event_args,
                    metadata=BlockStoredMetadata(cache_salt=entry.cache_salt),
                )
            parent_block_hash = block_hash
            page_index += 1

    def _lazy_events_merged(self, entry: _LazyStore) -> Optional[list]:
        """``_lazy_events(entry)`` coalesced among themselves, built without
        per-page objects (SGLANG_KV_EVENTS_LAZY_FAST_BUILD).

        The per-page events of one entry share medium, lora and metadata, and
        each links to the previous page's hash, so ``_coalesce`` merges every
        full page into the first one; a short last page has another
        ``block_size`` and stays separate. Coalescing the merged event into the
        queue tail then gives what coalescing the pages one by one gives.
        Returns None (use the per-page path) if the entry is shorter than its
        pages need, where the per-page path would raise."""
        raw, n, ps = entry.raw, entry.logical_len, self.page_size
        if n <= 0:
            return []
        if entry.is_bigram:
            tokens = list(zip(raw[0:n], raw[1 : n + 1]))
        else:
            tokens = list(raw[0:n])
        n_pages = (n + ps - 1) // ps
        if len(tokens) != n or len(entry.hash_values) < n_pages:
            return None
        hashes = [hash_str_to_int64(h) for h in entry.hash_values[:n_pages]]
        full = n // ps
        segments = []
        if full:
            segments.append(
                (hashes[:full], entry.parent_block_hash, tokens[: full * ps], ps)
            )
        if n > full * ps:
            parent = hashes[full - 1] if full else entry.parent_block_hash
            segments.append(
                ([hashes[full]], parent, tokens[full * ps :], n - full * ps)
            )
        events = []
        for block_hashes, parent, token_ids, size in segments:
            event_args = {
                "block_hashes": block_hashes,
                "parent_block_hash": parent,
                "token_ids": token_ids,
                "block_size": size,
                "lora_id": None,
                "medium": entry.medium,
            }
            if entry.cache_salt is None:
                events.append(BlockStored(**event_args))
            else:
                events.append(
                    BlockStoredWithMetadata(
                        **event_args,
                        metadata=BlockStoredMetadata(cache_salt=entry.cache_salt),
                    )
                )
        return events

    def _check_fast_build(self, entry: _LazyStore) -> None:
        """Check mode with SGLANG_KV_EVENTS_LAZY_FAST_BUILD: compare the merged
        build of ``entry`` with its per-page events coalesced, both built now
        from new objects (the commit-time events stay untouched and are still
        the ones published). A mismatch is a fast-build defect, apart from the
        commit-vs-take staleness the check-mode compare counts."""
        merged = self._lazy_events_merged(entry)
        if merged is None:
            return  # the publication path would use the per-page build
        coalesced: list = []
        for event in self._lazy_events(entry):
            self._coalesce(coalesced, event)
        self.lazy_fast_checked += 1
        if merged != coalesced:
            self.lazy_fast_mismatches += 1
            logger.error(
                "KV events lazy fast build mismatch #%d: the merged build differs "
                "from the per-page build (%d vs %d events); publishing the "
                "commit-time events",
                self.lazy_fast_mismatches,
                len(merged),
                len(coalesced),
            )

    def _materialize(self, items: list) -> list:
        """Build the lazy entries of ``items`` in place and coalesce as
        ``enqueue`` does, in queue order: the eager stream, byte for byte."""
        out: list = []
        for item in items:
            if not isinstance(item, _LazyStore):
                # A segment boundary is appended like an event that merges
                # with nothing, so no event merges across it either.
                self._coalesce(out, item)
                continue
            events = None
            if self.lazy_fast and item.check is None:
                events = self._lazy_events_merged(item)
                if events is not None:
                    self.lazy_fast_built += 1
            if events is None:
                events = list(self._lazy_events(item))
            if item.check is not None:
                if self.lazy_fast:
                    self._check_fast_build(item)
                self.lazy_checked += 1
                self._lazy_n[1] += 1
                if events != item.check:
                    self.lazy_mismatches += 1
                    self._lazy_n[2] += 1
                    logger.error(
                        "KV events lazy load-back mismatch #%d: the node's keys or "
                        "hashes changed between commit and take(); publishing the "
                        "commit-time events",
                        self.lazy_mismatches,
                    )
                events = item.check
            for event in events:
                self._coalesce(out, event)
        return out

    def take(self) -> list:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enabled:
            return []
        events = self._queue
        self._queue = []
        if self._lazy_pending:
            self._lazy_pending = 0
            events = self._materialize(events)
        if self._closed:
            # Boundaries queued (SGLANG_KV_EVENTS_BATCH_PUBLISH): drop them.
            self._closed = self._lazy_closed = 0
            events = [e for e in events if e is not _BOUNDARY]
        if self.lazy_load_back:
            self._lazy_tick()
        return events

    def mark_boundary(self) -> None:
        """Close the queued publication segment (SGLANG_KV_EVENTS_BATCH_PUBLISH).

        Called where ``take`` used to run. Nothing queued later coalesces into
        the closed segment, so ``take_closed`` returns what one ``take`` here
        returned. O(1); a no-op on an empty segment, where ``take`` returned
        nothing."""
        if not self.enabled:
            return
        queue = self._queue
        if queue and queue[-1] is not _BOUNDARY:
            queue.append(_BOUNDARY)
            self._closed = len(queue)
            self._lazy_closed = self._lazy_pending

    def take_closed(self) -> list:
        """Take the events of the closed segments, in queue order.

        Each segment is built and coalesced on its own, so the result is the
        concatenation of what one ``take`` per ``mark_boundary`` returned, byte
        for byte. Items queued after the last boundary stay queued."""
        if not self.enabled or not self._closed:
            return []
        queue, n_closed = self._queue, self._closed
        if n_closed == len(queue):
            head, self._queue = queue, []
        else:
            head, self._queue = queue[:n_closed], queue[n_closed:]
        lazy_head = self._lazy_closed
        self._closed = self._lazy_closed = 0
        if lazy_head:
            self._lazy_pending -= lazy_head
            head = self._materialize(head)
        events = [e for e in head if e is not _BOUNDARY]
        if self.lazy_load_back:
            self._lazy_tick()
        return events

    def _lazy_tick(self) -> None:
        """6f counter line, at most one per LAZY_LOG_INTERVAL_S (lazy mode only)."""
        now = time.monotonic()
        dt = now - self._lazy_log_t
        if dt < LAZY_LOG_INTERVAL_S:
            return
        self._lazy_log_t = now
        entries, checked, mismatches = self._lazy_n
        self._lazy_n = [0, 0, 0]
        msg = (
            "KV events lazy load-back over %.0f s: entries=%d checked=%d "
            "mismatches=%d check=%d (totals entries=%d checked=%d mismatches=%d)"
        )
        args = [
            dt,
            entries,
            checked,
            mismatches,
            int(self.lazy_check),
            self.lazy_entries,
            self.lazy_checked,
            self.lazy_mismatches,
        ]
        if self.lazy_fast:
            # SGLANG_KV_EVENTS_LAZY_FAST_BUILD only; the line is unchanged
            # without it.
            msg += " fast_build totals built=%d checked=%d mismatches=%d"
            args += [
                self.lazy_fast_built,
                self.lazy_fast_checked,
                self.lazy_fast_mismatches,
            ]
        logger.info(msg, *args)
