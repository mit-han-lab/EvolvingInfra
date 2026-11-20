# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Symmetric P/D on workers with DP attention: where the DP controller puts each leg
(symmetric/dp_placement.py) and how a decode leg finds the prefill leg's DP rank
(pull/protocol.py, pull/importer.py). Needs the engine (decoupled_engine) and zmq."""
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("torch")
pytest.importorskip("zmq")
pytest.importorskip("decoupled_engine")

from decoupled_engine.srt.disaggregation.pull import protocol as wire  # noqa: E402
from decoupled_engine.srt.disaggregation.pull.api import ChunkReady, ImportMark, PoolPages  # noqa: E402
from decoupled_engine.srt.disaggregation.pull.importer import ReaderPlane  # noqa: E402
from decoupled_engine.srt.disaggregation.symmetric.dp_placement import (  # noqa: E402
    SymmetricDpPlacement,
)
from decoupled_engine.srt.disaggregation.symmetric.planes import (  # noqa: E402
    dp_pairing,
    import_dp_rank,
)

PAGE = 4


def snapshot(rank, queued, stamp, waiting=0):
    return SimpleNamespace(dp_rank=rank, num_waiting_uncached_tokens=queued, num_waiting_reqs=waiting,
                           num_running_reqs=0, timestamp=stamp)


def test_dp_pairing_needs_attention_tp_one_and_several_dp_ranks():
    assert dp_pairing(SimpleNamespace(attn_dp_size=4, attn_tp_size=1, attn_cp_size=1))
    assert not dp_pairing(SimpleNamespace(attn_dp_size=1, attn_tp_size=4, attn_cp_size=1))
    assert not dp_pairing(SimpleNamespace(attn_dp_size=2, attn_tp_size=2, attn_cp_size=1))


def test_a_decode_leg_goes_where_its_owner_sends():
    placement = SymmetricDpPlacement(dp_size=4, page_size=PAGE)
    ranks = [placement.place([1] * 64, None, {"transfer_id": t}) for t in range(400)]
    assert ranks == [import_dp_rank(t, 4) for t in range(400)]
    # Consecutive transfer ids (the router's counter) still reach every rank evenly.
    assert all(80 <= ranks.count(r) <= 120 for r in range(4))


def test_the_next_turn_returns_to_the_rank_that_holds_the_conversation():
    placement = SymmetricDpPlacement(dp_size=4, page_size=PAGE)
    turn1 = list(range(100, 140))
    first = placement.place(turn1, {"transfer_id": 1}, None)
    # Another conversation loads the other ranks, the holder becomes the busiest.
    placement.update_loads([snapshot(r, 10 if r == first else 0, 1.0) for r in range(4)])
    turn2 = turn1 + list(range(500, 508))
    assert placement.place(turn2, {"transfer_id": 2}, None) == first


def test_a_prefix_every_rank_holds_prefers_the_least_loaded_rank():
    placement = SymmetricDpPlacement(dp_size=4, page_size=PAGE)
    system = list(range(40))
    # Conversations longer than twice the shared prefix go by load, which leaves the prefix on every rank.
    for t, rank in enumerate(range(4)):
        placement.update_loads([snapshot(r, 0 if r == rank else 1000, t + 1.0) for r in range(4)])
        assert placement.place(system + [9000 + t] * 48, {"transfer_id": t}, None) == rank
    placement.update_loads([snapshot(r, 0 if r == 2 else 50, 9.0) for r in range(4)])
    assert placement.place(system + [7] * 8, {"transfer_id": 9}, None) == 2


def test_a_turn_stays_on_its_rank_when_that_rank_has_the_longest_queue():
    # The queues measured on a saturated 6P6D prefill worker at c=1024: waiting requests and queued uncached
    # tokens per DP rank. The holder's queue is 1.36M tokens longer than the shortest one.
    placement = SymmetricDpPlacement(dp_size=4, page_size=PAGE)
    prompt = list(range(4096))
    holder = placement.place(prompt, {"transfer_id": 1}, None)
    loads = {holder: (42, 6_349_056)}
    others = [r for r in range(4) if r != holder]
    loads.update(zip(others, [(43, 4_989_056), (43, 5_469_888), (25, 5_767_296)]))
    placement.update_loads([snapshot(r, loads[r][1], 2.0, waiting=loads[r][0]) for r in range(4)])
    assert placement.place(prompt + list(range(9000, 9064)), {"transfer_id": 2}, None) == holder


def test_ranks_imbalanced_by_requests_take_the_least_loaded_rank():
    # The router's imbalance fast path: more than 32 requests apart and more than 1.1 times the least loaded.
    placement = SymmetricDpPlacement(dp_size=4, page_size=PAGE)
    prompt = list(range(64))
    holder = placement.place(prompt, {"transfer_id": 1}, None)
    other = (holder + 1) % 4
    placement.update_loads([snapshot(r, 0, 2.0, waiting=10 if r == other else 80) for r in range(4)])
    assert placement.place(prompt + [1, 2, 3, 4], {"transfer_id": 2}, None) == other


def test_stale_snapshots_keep_the_queue_estimate():
    placement = SymmetricDpPlacement(dp_size=2, page_size=PAGE)
    placement.update_loads([snapshot(0, 0, 1.0), snapshot(1, 0, 1.0)])
    a = placement.place(list(range(8)), {"transfer_id": 1}, None)
    placement.update_loads([snapshot(0, 0, 1.0), snapshot(1, 0, 1.0)])  # same timestamps
    b = placement.place(list(range(100, 108)), {"transfer_id": 2}, None)
    assert a != b  # the first request's 8 tokens still count on its rank


def test_registrations_match_on_the_kv_shard():
    def reg(tp_rank, kv_shard, attn_tp_size=1):
        region = wire.ItemRegion(ptrs=[0], item_lens=[64])
        return wire.Registration(session_id="s", tp_rank=tp_rank, attn_tp_size=attn_tp_size,
                                 page_size=PAGE, pools={}, aux=region, kv_shard=kv_shard)

    # DP attention: different DP ranks, the same (whole) KV.
    assert wire.check_compatible(local=reg(1, 0), remote=reg(3, 0)) is None
    # Plain TP: rank i only with rank i; a peer without kv_shard compares tp_rank.
    assert wire.check_compatible(local=reg(1, 1, 4), remote=reg(3, 3, 4)) is not None
    assert wire.check_compatible(local=reg(2, None, 4), remote=reg(2, 2, 4)) is None
    assert wire.check_compatible(local=reg(2, None, 4), remote=reg(1, 1, 4)) is not None
    blob = reg(3, 0).to_wire()
    assert wire.Registration.from_wire(blob).kv_shard == 0
    del blob["kv_shard"]
    assert wire.Registration.from_wire(blob).kv_shard is None


def test_a_publication_names_its_owner_rank():
    chunk = ChunkReady(transfer_id=7, chunk_seq=0, token_start=0, token_end=4,
                       pools=[PoolPages(pool="full", page_indices=[1], generations=[0])])
    msg = wire.decode(wire.encode(wire.make_chunk_ready(chunk=chunk, attempt_id=1,
                                                        owner_addr="tcp://p:9004")))
    assert msg.owner_addr == "tcp://p:9004" and msg.chunk().token_end == 4
    assert wire.rank_channel("tcp://p:9000", 2) == "tcp://p:9004"


@pytest.fixture
def reader():
    region = wire.ItemRegion(ptrs=[0], item_lens=[64])
    registration = wire.Registration(session_id="s", tp_rank=1, attn_tp_size=1, page_size=PAGE,
                                     pools={}, aux=region, kv_shard=0)
    plane = ReaderPlane(registration=registration, read_engine=mock.Mock(),
                        bind_addr="inproc://test-dp-reader", owner_ranks=4, waiting_timeout_s=30)
    yield plane
    plane.close()


def test_the_reader_moves_an_import_to_the_owner_rank_of_its_first_publication(reader):
    sent = []
    reader._send = lambda addr, msg: sent.append((addr, msg.type))
    mark = ImportMark(transfer_id=7, owner_worker="p", owner_addr="tcp://p:9000", local_aux_index=0)
    reader.begin_import(mark, "r7", lambda chunk: {})
    imp = reader._imports[7]
    # Before any publication an ABORT reaches every rank of the prefill worker.
    with reader._mutex:
        assert reader._owner_addrs(imp) == [f"tcp://p:{9000 + 2 * r}" for r in range(4)]
    chunk = ChunkReady(transfer_id=7, chunk_seq=0, token_start=0, token_end=4,
                       pools=[PoolPages(pool="full", page_indices=[1], generations=[0])])
    msg = wire.make_chunk_ready(chunk=chunk, attempt_id=1, owner_addr="tcp://p:9004")
    admitted, owner_reg = reader._admit(msg)
    assert admitted is imp and owner_reg is None  # REGISTER from that rank still pending
    assert imp.mark.owner_addr == "tcp://p:9004" and imp.owner_known
    assert "tcp://p:9004" in reader._owners
    with reader._mutex:
        assert reader._owner_addrs(imp) == ["tcp://p:9004"]
    reader.abort_pending(ImportMark(transfer_id=8, owner_worker="p", owner_addr="tcp://p:9000",
                                    local_aux_index=1), "client gone")
    assert [a for a, t in sent if t == wire.MSG_ABORT] == [f"tcp://p:{9000 + 2 * r}" for r in range(4)]
