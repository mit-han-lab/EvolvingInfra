"""Nonblocking, bounded control plane for on-demand GPU -> host peer export.

Only the scheduler touches cache state. The REP thread owns no CUDA tensors and
never waits for D2H. A ticket holds preparation ownership, not a DMA lease: once
LOOKUP grants a normal peer lease, that lease protects every RDMA source page.
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class ExportIntent:
    export_id: str
    hashes: tuple[str, ...]
    created: float
    deadline: float
    state: str = "queued"
    reason: str | None = None
    pages: int = 0
    waiters: set[str] = field(default_factory=set)
    started: bool = False


class GPUExportBroker:
    """Thread-safe metadata queue; scheduler drains it in one TP-wide order.

    The epoch is part of every ticket. A late poll/release cannot act on a restarted
    engine or a successor preparation for the same hashes. Deadlines never extend
    when another reader joins; resource ownership stays bounded under fanout.
    """

    def __init__(
        self,
        *,
        max_pending=256,
        max_waiters=2048,
        max_pages=1024,
        timeout_s=30.0,
        clock=time.monotonic,
    ):
        self.max_pending = max_pending
        self.max_waiters = max_waiters
        self.max_pages = max_pages
        self.timeout_s = timeout_s
        self.clock = clock
        self.epoch = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._intents = OrderedDict()
        self._keys = {}
        self._tickets = {}

    def request(self, hashes, waiter):
        if (
            not isinstance(hashes, list)
            or not hashes
            or len(hashes) > self.max_pages
            or any(not isinstance(h, str) or not h or len(h) > 256 for h in hashes)
            or not isinstance(waiter, str)
            or not waiter
            or len(waiter) > 128
        ):
            return {"ok": False, "state": "rejected", "reason": "invalid_or_oversized"}
        key = tuple(hashes)
        now = self.clock()
        with self._lock:
            if waiter in self._tickets:
                intent = self._intents[self._tickets[waiter]]
                if intent.hashes != key:
                    return {"ok": False, "reason": "waiter_identity_conflict"}
                if now >= intent.deadline:
                    return {
                        "ok": True,
                        "state": "expired",
                        "reason": "prepare_deadline",
                    }
                return self._reply(intent, waiter)
            if len(self._tickets) >= self.max_waiters:
                return {"ok": True, "state": "busy", "reason": "waiter_budget"}
            intent = self._intents.get(self._keys.get(key))
            if intent is None:
                if len(self._intents) >= self.max_pending:
                    return {"ok": True, "state": "busy", "reason": "queue_budget"}
                export_id = self.epoch + ":" + uuid.uuid4().hex
                intent = ExportIntent(export_id, key, now, now + self.timeout_s)
                self._intents[export_id] = intent
                self._keys[key] = export_id
            if now >= intent.deadline:
                return {"ok": True, "state": "expired", "reason": "prepare_deadline"}
            intent.waiters.add(waiter)
            self._tickets[waiter] = intent.export_id
            return self._reply(intent, waiter)

    def status(self, export_id, waiter):
        with self._lock:
            intent = self._intents.get(export_id)
            if intent is None or waiter not in intent.waiters:
                return {"ok": True, "state": "expired", "reason": "unknown_ticket"}
            if self.clock() >= intent.deadline:
                return {"ok": True, "state": "expired", "reason": "prepare_deadline"}
            return self._reply(intent, waiter)

    def release(self, export_id, waiter):
        with self._lock:
            intent = self._intents.get(export_id)
            if intent is not None and self._tickets.get(waiter) == export_id:
                intent.waiters.discard(waiter)
                self._tickets.pop(waiter, None)
        return {"ok": True}

    def _reply(self, intent, waiter):
        return {
            "ok": True,
            "version": 1,
            "export_id": intent.export_id,
            "waiter": waiter,
            "state": intent.state,
            "reason": intent.reason,
            "pages": intent.pages,
        }

    def plan(self, active):
        """Called on owner rank zero; no rank-local cache decisions here."""
        with self._lock:
            now = self.clock()
            retire = []
            for intent in list(self._intents.values()):
                if not intent.waiters or now >= intent.deadline:
                    retire.append(intent.export_id)
                    # Do not remove active metadata until the scheduler drains DMA.
                    if intent.export_id not in active:
                        self._forget(intent.export_id)
            start = None
            if not active:
                for intent in self._intents.values():
                    if (
                        not intent.started
                        and intent.waiters
                        and now < intent.deadline
                        and intent.state == "queued"
                    ):
                        intent.started = True
                        intent.state = "preparing"
                        start = {
                            "export_id": intent.export_id,
                            "hashes": list(intent.hashes),
                        }
                        break
            return {"start": start, "retire": retire}

    def has_work(self):
        with self._lock:
            return bool(self._intents)

    def update(self, export_id, state, *, pages=0, reason=None):
        with self._lock:
            intent = self._intents.get(export_id)
            if intent is not None:
                intent.state, intent.pages, intent.reason = state, pages, reason

    def forget(self, export_id):
        with self._lock:
            self._forget(export_id)

    def _forget(self, export_id):
        intent = self._intents.pop(export_id, None)
        if intent is None:
            return
        if self._keys.get(intent.hashes) == export_id:
            self._keys.pop(intent.hashes)
        for waiter in intent.waiters:
            if self._tickets.get(waiter) == export_id:
                self._tickets.pop(waiter)


def export_signature(hashes, node_id, endpoint_tokens):
    """TP plan identity, kept within signed int64 for [x, -x] votes."""
    payload = "\0".join([*hashes, str(node_id), str(endpoint_tokens)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:6], "big")
