"""Store serialized prepared inputs in a bounded, expiring process-local cache.

Taking an entry transfers body ownership to an uploader. This cache contains
input JSON, not model KV pages or completed-request history.
"""

from __future__ import annotations

import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable


class ProtocolError(Exception):
    """Carry a protocol failure from storage or request handling to the HTTP layer.

    Input: HTTP status, error code and message. Output: a ProtocolError instance.
    """
    def __init__(self, status: int, code: str, message: str):
        """Attach HTTP status and a stable error code to a protocol exception.

        Args:
            status: HTTP error status code.
            code: Machine-readable protocol error code.
            message: Human-readable explanation safe for the HTTP error envelope.

        Returns:
            None; initializes the exception message and public error fields.
        """
        self.status, self.code = status, code
        super().__init__(message)


@dataclass(slots=True)
class Entry:
    """Hold one prepared request's input until transfer or removal.

    Input: request_id, ref, monotonic deadline, body bytes and stream flag.
    Output: an Entry instance without agent state, KV data or completion history.
    """
    request_id: str
    ref: str
    deadline: float
    data: bytes
    stream: bool


class Store:
    """Keep ready-to-send JSON bodies in an event-loop-owned expiring LRU cache.

    Input: byte capacity, TTL seconds and an optional monotonic clock callable.
    Output: a Store instance providing put, find, take, release and sweep operations.

    Capacity counts actual body bytes, not a context-length reservation.
    Taking an entry transfers ownership to the uploader; no history remains.
    """

    def __init__(self, *, max_bytes: int, ttl: float, clock: Callable = time.monotonic):
        """Initialize an empty expiring cache with byte-based capacity accounting.

        Args:
            max_bytes: Positive maximum sum of retained serialized body bytes.
            ttl: Positive entry lifetime in seconds.
            clock: Callable returning monotonic seconds; injectable for tests.

        Returns:
            None; starts with no entries and zero accounting counters.

        Raises:
            ValueError: Capacity or lifetime is not positive.
        """
        if min(max_bytes, ttl) <= 0:
            raise ValueError("Cache capacity and lifetime must be positive")
        self.max_bytes, self.ttl, self.clock = max_bytes, ttl, clock
        self.entries: OrderedDict[str, Entry] = OrderedDict()
        self.bytes_used = self.peak_bytes = self.evicted = self.expired = 0

    def put(self, request_id: str, data: bytes, stream: bool) -> Entry:
        """Retain a prepared body after expiring or evicting unused entries.

        May remove old entries and updates retained-byte and eviction counters.

        Args:
            request_id: Identity that must match later lookup or transfer.
            data: Nonempty serialized request-body bytes.
            stream: Whether the stored generation request streams output.

        Returns:
            A new Entry containing an opaque reference and expiry deadline.

        Raises:
            ProtocolError: The body is empty or exceeds the entire cache capacity.
        """
        if not data or len(data) > self.max_bytes:
            raise ProtocolError(413, "INPUT_TOO_LARGE", "One result exceeds cache capacity.")
        self.sweep()
        while self.bytes_used + len(data) > self.max_bytes:
            ref = next(iter(self.entries))
            self._remove(ref).data = b""
            self.evicted += 1
        ref = secrets.token_urlsafe(24)
        while ref in self.entries:
            ref = secrets.token_urlsafe(24)
        entry = Entry(request_id, ref, self.clock() + self.ttl, data, stream)
        self.entries[ref] = entry
        self.bytes_used += len(data)
        self.peak_bytes = max(self.peak_bytes, self.bytes_used)
        return entry

    def find(self, ref: str, request_id: str) -> Entry:
        """Find a live matching entry and mark it most recently used.

        Args:
            ref: Opaque reference returned by put.
            request_id: Expected identity of the prepared request.

        Returns:
            The retained Entry; its expiry deadline is unchanged.

        Raises:
            ProtocolError: The entry is absent/expired or its identity differs.
        """
        self.sweep()
        entry = self.entries.get(ref)
        if entry is None:
            raise ProtocolError(404, "REF_NOT_FOUND", "Result is absent, expired or already sent.")
        if entry.request_id != request_id:
            raise ProtocolError(409, "IDENTITY_MISMATCH", "Reference and request ID differ.")
        self.entries.move_to_end(ref)
        return entry

    def take(self, ref: str, request_id: str) -> Entry:
        """Remove a matching entry and transfer body ownership to the caller.

        No history or body remains in the store after a successful take.

        Args:
            ref: Opaque reference returned by put.
            request_id: Expected identity of the prepared request.

        Returns:
            The removed Entry with body bytes intact.

        Raises:
            ProtocolError: The entry is absent/expired or its identity differs.
        """
        entry = self.find(ref, request_id)
        self._remove(ref)
        return entry

    def release(self, ref: str, request_id: str) -> str:
        """Remove an unused prepared input and clear its retained body.

        Args:
            ref: Opaque reference returned by put.
            request_id: Expected identity of the prepared request.

        Returns:
            released if removed, or not_found if already absent.

        Raises:
            ProtocolError: The supplied request identity differs from the stored one.
        """
        try:
            entry = self.take(ref, request_id)
        except ProtocolError as exc:
            if exc.code == "REF_NOT_FOUND":
                return "not_found"
            raise
        entry.data = b""
        return "released"

    def _remove(self, ref: str) -> Entry:
        """Remove an existing entry and decrement retained-byte accounting.

        Args:
            ref: Reference known to exist in the store.

        Returns:
            The removed Entry, leaving its body intact.

        Raises:
            KeyError: The reference does not exist.
        """
        entry = self.entries.pop(ref)
        self.bytes_used -= len(entry.data)
        return entry

    def sweep(self) -> None:
        """Remove expired entries and release their retained body bytes.

        Uses the injected clock without extending any surviving deadline.

        Args:
            None.

        Returns:
            None; updates cache byte accounting and the expiry counter.
        """
        now = self.clock()
        for ref, entry in list(self.entries.items()):
            if entry.deadline <= now:
                self._remove(ref).data = b""
                self.expired += 1
