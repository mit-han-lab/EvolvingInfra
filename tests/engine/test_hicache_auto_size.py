# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""HiCache auto-sizing of a DeepSeek-V4 cache. The paper preset turns the host tier on for every
symmetric worker and leaves its size to auto-sizing unless host dedup sets the ratio, as under DP
attention. Auto-sizing takes the device bytes of the page state the DSv4 host stack mirrors: the
buffers P/D transfers and the drafts' SWA layers, one host page per target SWA page."""
from types import SimpleNamespace

import pytest


@pytest.fixture
def sizing(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("decoupled_engine")
    from decoupled_engine.srt.mem_cache import hicache_auto_size
    from decoupled_engine.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    chosen = {}
    monkeypatch.setattr(
        hicache_auto_size,
        "get_context",
        lambda: SimpleNamespace(override=lambda _name, **fields: chosen.update(fields)),
    )

    def run(budget_bytes, requested_ratio=2.0, fraction=0.5):
        monkeypatch.setattr(
            hicache_auto_size,
            "get_memory",
            lambda: SimpleNamespace(hicache_host_memory_fraction=fraction, hicache_ratio=requested_ratio),
        )
        monkeypatch.setattr(hicache_auto_size, "host_memory_budget_bytes", lambda *a, **k: budget_bytes)
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        # FULL-paged compressed/indexer pools 600 + 200 B; SWA pages and c4 state 100 B.
        pool.get_contiguous_buf_infos = lambda: ([0, 0], [600, 200], [60, 20])
        pool.get_state_buf_infos = lambda: ([0], [100], [10])
        pool.swa_kv_pool = SimpleNamespace(kv_buffer=[torch.empty(10, 10, dtype=torch.uint8)])
        # A draft SWA layer of 8 B per page with fewer pages: the host gives it one per target page.
        draft = SimpleNamespace(swa_kv_pool=SimpleNamespace(kv_buffer=[torch.empty(5, 8, dtype=torch.uint8)]))
        params = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: pool),
            req_to_token_pool=None,
            mtp_draft_device_pools=(draft,),
        )
        with hicache_auto_size.auto_size_hicache(params, None, enabled=True):
            pass
        return chosen["hicache_ratio"]

    return run


def test_dsv4_ratio_fits_the_host_budget(sizing):
    # Device bytes 600 + 200 + 100 + 8 * 10 = 980; the budget is 0.5 * 4000 B, less the slack.
    assert sizing(budget_bytes=4000) == pytest.approx(2000 * 0.95 / 980)


def test_dsv4_keeps_the_requested_ratio_when_it_fits(sizing):
    assert sizing(budget_bytes=10**9) == 2.0
