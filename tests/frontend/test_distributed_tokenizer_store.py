# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
from dataclasses import dataclass

import pytest

from evolving_infra.router.frontend.store import ProtocolError, Store


@dataclass
class Clock:
    """Provide a manually adjustable time source for expiry tests.

    Input: optional initial now value. Output: a callable Clock instance.
    """
    now: float = 0

    def __call__(self):
        """Read simulated time without advancing it.

        Input: this clock's now field. Output: the current simulated time value.
        """
        return self.now


def make_store(**kwargs):
    """Create a small cache with overridable settings for storage tests.

    Input: Store constructor overrides. Output: a configured Store instance.
    """
    return Store(**{"max_bytes": 20, "ttl": 300, **kwargs})


def test_actual_bytes_not_context_or_fixed_record_reservations():
    """Verify that entries charge actual body bytes and repeated IDs receive distinct refs.

    Input: no arguments; uses two fixed bodies. Output: None, with assertion failures on bugs.
    """
    store = make_store()
    a = store.put("same-id", b"123", False)
    b = store.put("same-id", b"4567", True)
    assert a.ref != b.ref and store.bytes_used == 7
    assert len(store.entries) == 2 and store.peak_bytes == 7


def test_ttl_is_fixed_and_access_does_not_extend_it():
    """Verify fixed expiry, body cleanup and absent-result release behavior.

    Input: no arguments; advances a simulated clock. Output: None if assertions pass.
    """
    clock = Clock()
    store = make_store(clock=clock)
    entry = store.put("id", b"1234", False)
    clock.now = 299
    assert store.find(entry.ref, "id") is entry
    clock.now = 300
    store.sweep()
    assert store.bytes_used == 0 and not store.entries and store.expired == 1
    assert entry.data == b""
    assert store.release(entry.ref, "id") == "not_found"


def test_lru_uses_last_access_and_clears_evicted_data():
    """Verify that capacity eviction follows access order and clears the evicted body.

    Input: no arguments; fills an eight-byte cache. Output: None if assertions pass.
    """
    store = make_store(max_bytes=8)
    a = store.put("a", b"aaaa", False)
    b = store.put("b", b"bbbb", False)
    store.find(a.ref, "a")
    c = store.put("c", b"cccc", False)
    assert list(store.entries) == [a.ref, c.ref]
    assert b.data == b"" and store.bytes_used == 8 and store.evicted == 1


def test_take_transfers_ownership_without_retaining_history():
    """Verify that taking removes cache state while leaving the caller's body intact.

    Input: no arguments; uses a four-byte entry. Output: None if assertions pass.
    """
    store = make_store(max_bytes=4)
    entry = store.put("a", b"aaaa", True)
    taken = store.take(entry.ref, "a")
    assert taken is entry and taken.data == b"aaaa"
    assert store.bytes_used == 0 and not store.entries
    store.put("a", b"bbbb", False)
    assert taken.data == b"aaaa"  # Cache operations cannot clear an active upload.
    with pytest.raises(ProtocolError) as error:
        store.take(entry.ref, "a")
    assert error.value.code == "REF_NOT_FOUND"


def test_wrong_identity_cannot_consume_or_refresh_an_entry():
    """Verify that an identity mismatch neither removes an entry nor changes its LRU order.

    Input: no arguments; uses a wrong request ID for an existing ref. Output: None.
    """
    store = make_store()
    a = store.put("a", b"a", False)
    b = store.put("b", b"b", False)
    with pytest.raises(ProtocolError) as error:
        store.take(a.ref, "wrong")
    assert error.value.code == "IDENTITY_MISMATCH"
    assert list(store.entries) == [a.ref, b.ref] and store.bytes_used == 2


def test_single_oversized_result_does_not_evict_valid_results():
    """Verify that an oversized individual body is rejected before existing entries are evicted.

    Input: no arguments; submits a body larger than the test cache. Output: None.
    """
    store = make_store()
    a = store.put("a", b"a", False)
    with pytest.raises(ProtocolError) as error:
        store.put("b", b"b" * 21, False)
    assert error.value.status == 413 and list(store.entries) == [a.ref]


def test_completed_requests_do_not_accumulate_and_release_is_idempotent():
    """Verify that repeated preparation and release leave no completed-request history.

    Input: no arguments; performs 5000 fixed prepare/release cycles. Output: None.
    """
    store = make_store()
    for _ in range(5000):
        entry = store.put("same-id", b"abc", False)
        assert store.release(entry.ref, entry.request_id) == "released"
        assert store.release(entry.ref, entry.request_id) == "not_found"
    assert store.bytes_used == 0 and not store.entries and store.evicted == 0
