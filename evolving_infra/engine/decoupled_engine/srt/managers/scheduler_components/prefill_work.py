"""CPU-only progress of launched prefill chunks for router work accounting.

The router estimates requests that have not reached their first forward. Once
launched, batch prefix/extend lengths replace that estimate. Launching a chunk
does not retire its work: overlapping chunks stay outstanding until their
results have actually been processed. A missing RID never means completion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import msgspec

from decoupled_engine.srt.managers.load_snapshot import PrefillWorkMetrics


@dataclass
class _RequestWork:
    req: object
    unlaunched: int = 0
    chunks: dict[int, int] = field(default_factory=dict)

    @property
    def remaining(self) -> int:
        unlaunched = 0 if self.req.finished() else self.unlaunched
        return unlaunched + sum(self.chunks.values())


class PrefillWorkTracker:
    # Keep this optional section well within the existing 16 KiB SHM slot.
    # Oversized snapshots rotate coverage; omitted requests retain their router
    # reservations instead of being incorrectly treated as completed.
    PAYLOAD_BUDGET = 8192

    def __init__(self):
        self.epoch = uuid4().hex
        self.sequence = 0
        self.requests: dict[str, _RequestWork] = {}
        self.batches: dict[int, list[str]] = {}
        self._cursor = 0

    def on_launch(self, batch) -> None:
        if batch.forward_mode.name not in ("EXTEND", "MIXED"):
            return
        iteration = batch.forward_iter
        if iteration in self.batches:
            raise ValueError("Prefill batch registered twice")
        rids = []
        for req, prefix, extend in zip(
            batch.reqs, batch.prefix_lens, batch.extend_lens, strict=True
        ):
            # Symmetric workers can mix decode rows into an extend batch.
            # Those rows have already processed the whole input prompt.
            if prefix >= len(req.origin_input_ids):
                continue
            work = self.requests.get(req.rid)
            if work is None:
                work = _RequestWork(req)
                self.requests[req.rid] = work
            elif work.req is not req:
                if work.chunks:
                    raise ValueError("Prefill request ID reused while work is active")
                work = _RequestWork(req)
                self.requests[req.rid] = work
            work.unlaunched = max(0, len(req.origin_input_ids) - prefix - extend)
            work.chunks[iteration] = extend
            rids.append(req.rid)
        self.batches[iteration] = rids

    def on_result(self, batch) -> None:
        for rid in self.batches.pop(batch.forward_iter, []):
            self.requests[rid].chunks.pop(batch.forward_iter)
        # Every TP rank tracks work, but only the publishing rank (TP0) calls
        # snapshot(). Retire here too, or the other ranks keep every finished
        # Req alive (and with it its GPU prefix_indices) for the process
        # lifetime. Pruning is idempotent, so published snapshots are unchanged.
        self._retire_finished()

    def _retire_finished(self) -> None:
        # A completed prefill waiting for P->D transfer remains visible as zero.
        # A canceled request with an in-flight forward retains that forward's
        # work until its result, even if Req.finished() is already true.
        for rid, work in list(self.requests.items()):
            if not work.chunks and work.req.finished():
                del self.requests[rid]

    def snapshot(self) -> PrefillWorkMetrics:
        self._retire_finished()
        self.sequence += 1
        remaining = {}
        entries = list(self.requests.items())
        count = len(entries)
        size = 0
        visited = 0
        if count:
            for offset in range(count):
                index = (self._cursor + offset) % count
                rid, work = entries[index]
                # Account for both encoded key and value, with map headroom.
                item_size = len(msgspec.msgpack.encode({rid: work.remaining}))
                visited += 1
                if size + item_size > self.PAYLOAD_BUDGET:
                    if remaining:
                        visited -= 1
                        break
                    # An unusually long RID must not break load publication.
                    continue
                remaining[rid] = work.remaining
                size += item_size
            self._cursor = (self._cursor + visited) % count
        return PrefillWorkMetrics(
            epoch=self.epoch,
            sequence=self.sequence,
            remaining=remaining,
            truncated=len(remaining) < count,
        )
