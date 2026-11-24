# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Peer KV pulls per (worker, DP rank): the port each rank binds, the port that names
a DP rank in hints and publications, the port a reader dials; under DP attention a
reader takes only a pull of the FULL KV; a failed sidecar read releases its lease.
Needs the engine (decoupled_engine), torch and zmq."""
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zmq")
pytest.importorskip("decoupled_engine")

from decoupled_engine.srt.mem_cache.peer_kv_servicer import (  # noqa: E402
    servicer_bind_port,
    servicer_group_port,
)
from decoupled_engine.srt.mem_cache.hicache_storage import PoolName, PoolTransfer  # noqa: E402
from decoupled_engine.srt.mem_cache.storage.peer_kv.peer_kv_backend import PeerKVBackend  # noqa: E402
from decoupled_engine.srt.mem_cache.unified_cache.components import ComponentType  # noqa: E402
from decoupled_engine.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402

BASE = 18100
FULL, SWA, MAMBA = ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA
HINT = {"source_worker_id": "http://p0:30000", "source_servicer_addr": "p0:18100", "expected_hit_tokens": 4096}


def reader(attn_tp_rank):
    # The reader's rank is its HiCache storage rank: the attention-TP rank under DP attention.
    backend = object.__new__(PeerKVBackend)
    backend._rank = attn_tp_rank
    backend._warned_bad_hint_ts = 0.0
    return backend


def test_each_dp_rank_of_a_worker_binds_its_own_servicer_port():
    # DP attention over 4 GPUs: every rank is attention-TP rank 0, so the port comes
    # from the TP rank: four ranks, four ports.
    ports = [servicer_bind_port(BASE, tp_rank) for tp_rank in range(4)]
    assert ports == [18100, 18101, 18102, 18103]
    # A rank's hint port is its own port, and a reader rank (attention-TP rank 0 under DP
    # attention) dials exactly it.
    for dp_rank in range(4):
        group = servicer_group_port(BASE, tp_rank=dp_rank, attn_tp_rank=0)
        assert group == ports[dp_rank]
        assert reader(0)._target_addr({"source_servicer_addr": f"h:{group}"}) == f"h:{group}"
    # A decode worker over 6 hosts: global TP rank 13 is on its fourth host, port BASE + 13,
    # the port the launcher reserves on that host.
    assert servicer_bind_port(BASE, 13) == 18113


def test_attention_tp_inside_a_dp_rank_reads_shard_to_shard():
    # TP 4 as 2 DP ranks of attention TP 2: DP rank 1 is TP ranks 2 and 3.
    assert [servicer_bind_port(BASE, r) for r in (2, 3)] == [18102, 18103]
    group = servicer_group_port(BASE, tp_rank=3, attn_tp_rank=1)
    assert group == 18102
    # Shard i of the reader reads shard i of the owner rank.
    assert reader(0)._target_addr({"source_servicer_addr": f"h:{group}"}) == "h:18102"
    assert reader(1)._target_addr({"source_servicer_addr": f"h:{group}"}) == "h:18103"
    # Plain TP: the group is the worker's base port.
    assert servicer_group_port(BASE, tp_rank=3, attn_tp_rank=3) == BASE
    assert reader(1)._target_addr({"source_servicer_addr": "h:bad"}) is None


class _Controller:
    """A cache controller that a dropped hint must not reach: any use but its backend fails."""

    def __init__(self):
        self.storage_backend = type("Backend", (), {"requires_kv_pull_hint": True})()

    def __getattr__(self, name):
        raise AssertionError(f"the prefetch reached the cache controller ({name})")


class _GatedTree(UnifiedRadixCache):
    """A radix cache whose other attributes fail the test: a dropped hint never gets past the pull gate."""

    def __getattr__(self, name):
        raise AssertionError(f"the prefetch went on past the pull gate ({name})")


def tree(components, replay=False, attn_dp_size=4, cls=UnifiedRadixCache):
    # A radix cache with only what the pull gate reads (and a peer KV backend).
    t = object.__new__(cls)
    t.tree_core = types.SimpleNamespace(enable_storage=True)
    t.tree_components = tuple(components)
    t.swa_branch_replay = replay
    t.attn_dp_size = attn_dp_size
    t._peer_pull_drop_logged = False
    t.cache_controller = _Controller()
    return t


@pytest.mark.parametrize(
    "components, replay, attn_dp_size, takes",
    [
        ((FULL,), False, 4, True),  # FULL-only tree under DP attention
        ((FULL, SWA), True, 4, True),  # V4.1 with SWA branch replay: FULL-only pull
        ((FULL, SWA), False, 4, False),  # replay off: the pull would read SWA
        ((FULL, MAMBA), False, 4, False),  # a mamba tree under DP attention
        ((FULL, SWA), False, 1, True),  # no DP attention: every tree pulls
    ],
)
def test_a_dp_attention_reader_takes_only_a_pull_of_the_full_kv(components, replay, attn_dp_size, takes):
    assert tree(components, replay, attn_dp_size).peer_pull_reads_full_only() is takes


@pytest.mark.parametrize(
    "components, replay, attn_dp_size, info",
    [
        ((FULL, SWA), True, 4, {"takes": True, "replay_tokens": 256}),  # V4.1: FULL-only pull, one window replayed
        ((FULL, SWA), False, 4, {"takes": False, "replay_tokens": 0}),  # the pull would read SWA: none taken
        ((FULL,), False, 4, {"takes": True, "replay_tokens": 0}),  # FULL-only tree: an exact pull
        ((FULL, SWA), False, 1, {"takes": True, "replay_tokens": 0}),  # no DP attention: exact up to a window
    ],
)
def test_a_rank_reports_the_pulls_it_takes_and_their_replay(components, replay, attn_dp_size, info):
    # The router reads this (/server_info internal state `peer_kv_pull`) to hint and
    # price a pull only for a reader that takes one, net of the tokens it replays.
    t = tree(components, replay, attn_dp_size)
    t._swa_branch_replay_len = 256 if replay else 0
    assert t.peer_pull_info() == info
    # Without a peer KV backend no rank takes a pull.
    t.cache_controller = None
    assert t.peer_pull_info() == {"takes": False, "replay_tokens": 0}


def test_a_hint_on_a_hybrid_tree_without_branch_replay_issues_no_lookup():
    # DP attention, FULL + SWA, no replay: the hint is dropped before any lookup,
    # so no owner leases pages for it and the request computes its prefix.
    t = tree((FULL, SWA), replay=False, cls=_GatedTree)
    request = types.SimpleNamespace(rid="rid-1")
    assert t.prefetch_from_storage(request, 0, list(range(4096)), kv_pull_hint=dict(HINT)) is None
    assert t._peer_pull_drop_logged


class _Lease:
    lease_id, req_id = "lease-1", "rid-1"

    def is_complete(self):
        return False


def test_a_sidecar_read_that_raises_releases_the_lease_at_once():
    # The read of an SWA sidecar raised (no host staging): the owner's lease is
    # released as failed, not left to its TTL.
    backend = object.__new__(PeerKVBackend)
    lease, released = _Lease(), []
    backend._match_lease_pages = lambda keys, req_id, attempt_id: (lease, [0])
    backend._release = lambda held, complete=False: released.append((held, complete))
    backend._trace = lambda **fields: None
    pool = type(
        "Pool", (), {"kv_buffer": torch.zeros(1), "get_page_buffer_meta": lambda self, idx: idx.reshape(-1)}
    )()
    backend.registered_pools = {PoolName.SWA: pool}
    with pytest.raises(AttributeError):
        backend.batch_get_v2([PoolTransfer(name=PoolName.SWA, keys=["h1"], host_indices=None)])
    assert released == [(lease, False)]
