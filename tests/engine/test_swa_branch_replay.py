# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""A prompt that branches inside an earlier prompt reuses its cached FULL KV up to one SWA window before the
branch point (SWA branch replay, UnifiedRadixCache._match_swa_branch_replay).

The radix tree keeps an exported prompt's SWA (sliding-window) KV only on its tail pages
(symmetric/export_table.py trim_swa), and a match may end only after a cached SWA window
(SWAComponent.create_match_validator). A later prompt whose shared prefix ends inside the earlier one (a branch
point: a subagent preamble, a rewritten tail, a fork) used to back off to the deepest page with a cached window,
often to 0, although every FULL page was cached. With SWA branch replay the match ends R = ceil_page(window)
before the branch point; the request recomputes the rest from an empty window.

Needs the engine (decoupled_engine), DeepSeek-V4.1's geometry: page 256, SWA window 128, so R = 256. Real paged
pools and a real radix tree. The host-tier tests (the HiCache controller) and the sparse-prefill test (Triton
kernels) need CUDA; the others run on CPU.
"""
import dataclasses
import unittest
from array import array

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("decoupled_engine")

from decoupled_engine.srt.mem_cache.base_prefix_cache import (  # noqa: E402
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from decoupled_engine.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from decoupled_engine.srt.mem_cache.common import match_kv_cache  # noqa: E402
from decoupled_engine.srt.mem_cache.memory_pool import ReqToTokenPool  # noqa: E402
from decoupled_engine.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from decoupled_engine.srt.mem_cache.unified_cache.components import ComponentType  # noqa: E402
from decoupled_engine.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from decoupled_engine.test.unified_allocator_fixtures import (  # noqa: E402
    build_swa_pool,
    setup_allocator_context,
)

PAGE, WINDOW = 256, 128
REPLAY = 256  # ceil_page(WINDOW): a replay starts on a page boundary
FULL, SWA = ComponentType.FULL, ComponentType.SWA


def _pages(n, start):
    return list(range(start, start + n * PAGE))


# An earlier prompt of 8 pages, cached as an export leaves it: SWA only on its last 2 pages.
EARLIER = _pages(8, 0)
KEPT_TAIL = 6 * PAGE


def _cache(window=WINDOW):
    setup_allocator_context()
    allocator = build_swa_pool(page_size=PAGE, lazy=False)
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=ReqToTokenPool(
            size=4, max_context_len=100 * PAGE, device="cpu", enable_memory_saver=False
        ),
        token_to_kv_pool_allocator=allocator,
        page_size=PAGE,
        sliding_window_size=window,
        tree_components=(FULL, SWA),
        tree_core_backend="python",
    )
    # Set as an attribute, so the same tests run on a base without the option.
    params.swa_branch_replay = True
    return allocator, UnifiedRadixCache(params)


def _cache_prompt(allocator, cache, tokens, swa_from, value=None):
    """Cache a prompt as an export's release insert leaves it: FULL on every page, SWA only from `swa_from` on
    (export_table.trim_swa freed the rest)."""
    if value is None:
        value = allocator.alloc(len(tokens))
    allocator.free_swa(value[:swa_from])
    params = InsertParams(key=RadixKey(array("q", tokens)), value=value)
    params.set_evicted_seqlen(SWA, swa_from)
    cache.insert(params)


def _match(cache, tokens):
    return cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))


def _new_req(req_to_token_pool, tokens):
    from decoupled_engine.srt.managers.schedule_batch import Req
    from decoupled_engine.srt.sampling.sampling_params import SamplingParams

    req = Req(
        rid="replay", origin_input_text="", origin_input_ids=array("q", tokens),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    req.full_untruncated_fill_ids = req.origin_input_ids
    req_to_token_pool.alloc([req])
    return req


def _prefill_replay(cache, req, own):
    """The scheduler's prefill of an admitted replay match: the request locks its anchor (req.last_node, which can
    end past the reused prefix) and computes the rest of its prompt into its own slots `own`, reading no SWA below
    its first row (prepare_for_extend)."""
    start = req.prefix_len
    req.kv.cache_protected_len = start
    req.lock = cache.lock(req.last_node)
    row = torch.cat([cache.prefix_device_indices(req), own])
    cache.req_to_token_pool.write((req.kv.req_pool_idx, slice(0, len(row))), row)
    req.kv.kv_committed_len = len(row)
    req.swa_valid_start, req.swa_branch_replay = start, False


def _replay(allocator, cache, tokens):
    """Admit `tokens` (match_kv_cache) and prefill them; returns the request and the match."""
    req = _new_req(cache.req_to_token_pool, tokens)
    match = match_kv_cache(req, cache, req.full_untruncated_fill_ids)
    _prefill_replay(cache, req, allocator.alloc(len(tokens) - req.prefix_len))
    return req, match


def test_a_branch_inside_an_earlier_prompt_reuses_its_full_kv_up_to_one_window_before_the_branch():
    allocator, cache = _cache()
    _cache_prompt(allocator, cache, EARLIER, swa_from=KEPT_TAIL)
    branch = 5 * PAGE  # shares 5 pages, inside the earlier prompt's pages without SWA
    match = _match(cache, EARLIER[:branch] + _pages(3, 100_000))
    assert match.device_prefix_len == branch - REPLAY, (
        f"the match backs off to {match.device_prefix_len} although FULL is cached up to {branch}"
    )
    assert match.swa_branch_replay
    assert match.swa_branching_seqlen == branch  # the request caches its window there


def test_an_exact_continuation_reuses_the_cached_window_without_replay():
    allocator, cache = _cache()
    _cache_prompt(allocator, cache, EARLIER, swa_from=KEPT_TAIL)
    match = _match(cache, EARLIER + _pages(2, 100_000))
    assert match.device_prefix_len == len(EARLIER)
    assert not match.swa_branch_replay
    assert match.swa_branching_seqlen is None


def test_a_branch_just_before_the_kept_tail_replays_from_the_page_before():
    allocator, cache = _cache()
    _cache_prompt(allocator, cache, EARLIER, swa_from=KEPT_TAIL)
    # The window below the branch point is gone; the kept tail starts right at it.
    match = _match(cache, EARLIER[:KEPT_TAIL] + _pages(3, 100_000))
    assert match.device_prefix_len == KEPT_TAIL - REPLAY
    assert match.swa_branch_replay


@pytest.mark.parametrize("window, replay", [(128, 256), (384, 512)])
def test_the_replay_starts_on_a_page_and_covers_a_whole_window(window, replay):
    allocator, cache = _cache(window)
    _cache_prompt(allocator, cache, EARLIER, swa_from=KEPT_TAIL)
    branch = 5 * PAGE
    match = _match(cache, EARLIER[:branch] + _pages(3, 100_000))
    start = match.device_prefix_len
    # A request cannot write into a page of the tree, and recomputes at least one window before the branch.
    assert start % PAGE == 0 and branch - start >= window
    assert start == branch - replay


def test_a_branch_within_one_replay_of_the_last_window_is_a_plain_hit():
    allocator, cache = _cache()
    _cache_prompt(allocator, cache, EARLIER, swa_from=KEPT_TAIL)
    match = _match(cache, EARLIER[:REPLAY] + _pages(3, 100_000))  # branch - REPLAY = 0: nothing to reuse
    assert match.device_prefix_len == 0
    assert not match.swa_branch_replay


def test_the_window_a_replay_caches_at_the_branch_point_serves_the_next_branch_there():
    allocator, cache = _cache()
    _cache_prompt(allocator, cache, EARLIER, swa_from=KEPT_TAIL)
    req, match = _replay(allocator, cache, EARLIER[: 5 * PAGE] + _pages(3, 100_000))
    branch = match.swa_branching_seqlen
    # The chunk insert at the branch point (SWAComponent.prepare_for_caching_req) caches the window there.
    cache.checkpoint(req, up_to=branch)
    again = _match(cache, EARLIER[: 5 * PAGE] + _pages(3, 200_000))
    assert again.device_prefix_len == branch
    assert not again.swa_branch_replay


def test_a_replays_insert_keeps_the_earlier_prompts_full_kv_below_the_branch():
    """The replay recomputed [F', F) from an empty window, so its FULL KV there is not the earlier prompt's: the
    tree keeps its own FULL pages and takes only the request's SWA window (RecoverSWAWithLockedFull)."""
    allocator, cache = _cache()
    value = allocator.alloc(len(EARLIER))
    original = value.clone()
    _cache_prompt(allocator, cache, EARLIER, swa_from=KEPT_TAIL, value=value)
    req, match = _replay(allocator, cache, EARLIER[: 5 * PAGE] + _pages(3, 100_000))
    start, branch = match.device_prefix_len, match.swa_branching_seqlen
    cache.checkpoint(req, up_to=branch)
    # An exact continuation of the earlier prompt (no replay of its own) reads [F', F) from the tree.
    cont = _match(cache, EARLIER + _pages(1, 300_000))
    assert cont.device_prefix_len == len(EARLIER) and not cont.swa_branch_replay
    path = cache.path_device_indices(cont.last_device_node)
    assert torch.equal(path[start:branch], original[start:branch]), (
        f"the replay's insert replaced the earlier prompt's FULL KV on [{start}, {branch}) with the pages it "
        "recomputed from an empty window"
    )


def _resolve(monkeypatch, **changes):
    """deepseek_v4_hook.resolve_swa_branch_replay on a DeepSeek-V4.1 prefill config (CUDA, paged SWA,
    flashmla_mega, no CP) with `changes`; the option's resolved value, or what an explicit setting left."""
    from types import SimpleNamespace

    from decoupled_engine.srt.arg_groups import deepseek_v4_hook as hook

    ratios = changes.pop("compress_ratios", [0, 0, 2, 2, 1, 1, 0])
    cfg = SimpleNamespace(
        enable_swa_branch_replay=None, enable_encoder_swa_bounded_replay=False, disable_radix_cache=False,
        dsv4_attn_backend="flashmla_mega", enable_prefill_cp=False, attn_cp_size=1,
        disaggregation_mode="symmetric",
    )
    for name, value in changes.items():
        setattr(cfg, name, value)
    resolved = {}
    monkeypatch.setattr(hook, "resolving_view", lambda server_args: cfg)
    monkeypatch.setattr(
        hook, "model_config_of",
        lambda server_args: SimpleNamespace(hf_config=None, hf_text_config=SimpleNamespace(compress_ratios=ratios)),
    )
    monkeypatch.setattr(hook, "get_platform", lambda: SimpleNamespace(is_cuda=changes.get("is_cuda", True)))
    monkeypatch.setattr(hook, "declare_resolution", lambda server_args, source, **fields: resolved.update(fields))
    hook.resolve_swa_branch_replay(object())
    return resolved.get("enable_swa_branch_replay", cfg.enable_swa_branch_replay)


@pytest.mark.parametrize(
    "changes, resolved",
    [
        ({}, None),  # unset: a later default decides
        ({"disaggregation_mode": "null"}, None),
        ({"disaggregation_mode": "decode"}, False),  # a PD decode server never prefills a matched prompt
        ({"dsv4_attn_backend": "trtllm"}, False),  # its sparse MLA reads a fixed SWA length per row
        ({"enable_encoder_swa_bounded_replay": True}, False),  # the tree does not own the window
        ({"enable_prefill_cp": True}, False),
        ({"is_cuda": False}, False),
        ({"compress_ratios": [0, 4, 128]}, False),  # c4/c128 keep a compressor state in SWA pages
        ({"enable_swa_branch_replay": False}, False),  # the operator's setting wins
        ({"enable_swa_branch_replay": True}, True),
    ],
)
def test_deepseek_v41_keeps_swa_branch_replay_off_where_prefill_cannot_floor_the_window(
        monkeypatch, changes, resolved):
    assert _resolve(monkeypatch, **changes) is resolved


def test_an_explicit_swa_branch_replay_on_an_unsupported_config_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="trtllm"):
        _resolve(monkeypatch, enable_swa_branch_replay=True, dsv4_attn_backend="trtllm")


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _host_tier_replay(kit, *, evict_before_match=True, load_branch=True, tokens=None, key_cap=None):
    """The earlier prompt backed up to the host tier, and a request branching inside it (by default at 5 pages),
    admitted with its replay match and loaded back as the scheduler does. `evict_before_match`: the earlier prompt's
    FULL is on the host tier only at the match (an L2-hit replay), else on the device. `load_branch=False`: the
    load-back restores only the reused prefix, as when the branch node's load fails. `key_cap`: the match key's
    length limit (the scheduler's max_prefix_len). Returns (cache, allocator, req, match, loaded, node)."""
    from unittest import mock

    from decoupled_engine.srt.mem_cache.unified_cache.tree_core_registry import _TREE_CORE_REGISTRY
    from decoupled_engine.test.kits.unified_radix_cache.fixtures import build_fixture
    from decoupled_engine.test.kits.unified_radix_cache.unified_tree_core_inspector import (
        UnifiedTreeCoreInspector,
    )

    kit_cache, allocator, req_to_token_pool = build_fixture(kit.cfg)
    params = dataclasses.replace(kit_cache.cache_init_params, tree_core_backend="python")
    params.swa_branch_replay = True
    # The kit's backup and load-back steps walk the tree through its inspection interface.
    with mock.patch.dict(_TREE_CORE_REGISTRY, {"python": UnifiedTreeCoreInspector}):
        cache = UnifiedRadixCache(params)
    cache.cache_init_params = params
    kit._init_hicache(cache)
    value = kit._alloc(allocator, len(EARLIER))
    _cache_prompt(allocator, cache, EARLIER, swa_from=KEPT_TAIL, value=value)
    kit._backup_tree(cache)
    if evict_before_match:
        cache.evict(EvictParams(num_tokens=len(EARLIER)))  # FULL stays on the host tier only

    req = kit._make_req(req_to_token_pool)
    tokens = array("q", EARLIER[: 5 * PAGE] + _pages(3, 100_000) if tokens is None else tokens)
    req.origin_input_ids = req.full_untruncated_fill_ids = tokens
    match = match_kv_cache(req, cache, tokens, max_prefix_len=key_cap)
    if not load_branch:
        req.swa_branch_node = None  # as when the branch node's load-back fails
    loaded, node = 0, req.last_node
    if req.needs_host_load_back():
        loaded, node = cache.init_load_back(
            InitLoadBackParams(best_match_node=req.best_match_node, host_hit_length=req.host_hit_length, req=req)
        )
        if cache.ongoing_load_back:
            assert cache.ready_to_load_host_cache() != -1
            for ack in list(cache.cache_controller.ack_load_queue):
                ack.finish_event.synchronize()
            cache.loading_check()
        kit._release_ongoing_load_back_locks(cache)
        # As the scheduler applies a load-back (PrefillAdder.add_one_req).
        req.prefix_len += loaded
        req.last_node = node
    return cache, allocator, req, match, loaded, node


def _host_tier_kit():
    from decoupled_engine.test.kits.unified_radix_cache.config import CacheConfig
    from decoupled_engine.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures

    class Kit(UnifiedRadixCacheFixtures, unittest.TestCase):
        cfg = CacheConfig(
            page_size=PAGE, components=(FULL, SWA), sliding_window_size=WINDOW,
            kv_size=40 * PAGE, max_context_len=40 * PAGE,
        )

        def runTest(self):
            pass

    return Kit()


@needs_cuda  # the HiCache controller
def test_a_branch_whose_full_kv_is_on_the_host_tier_loads_full_down_to_the_branch():
    """The branch's FULL KV is only on the host tier (L2): the replay is a host hit. Its load-back restores FULL
    only, since the matched path holds no SWA on any tier, and down to the branch point: the request reuses it up
    to the replay start and anchors at the branch point."""
    kit = _host_tier_kit()
    try:
        cache, _, req, match, loaded, node = _host_tier_replay(kit)
        branch = 5 * PAGE
        assert (match.device_prefix_len, match.host_hit_length) == (0, branch - REPLAY), (
            f"device {match.device_prefix_len} + host {match.host_hit_length} although FULL is on the host "
            f"tier up to {branch}"
        )
        assert req.swa_branch_replay
        assert loaded == req.prefix_len == branch - REPLAY
        assert len(cache.path_device_indices(node)) == branch
    finally:
        kit.doCleanups()


@needs_cuda  # the HiCache controller
@pytest.mark.parametrize("evicted", ["before the match", "before the insert"])
def test_a_replay_caches_its_prompt_when_the_earlier_prompt_leaves_the_device(evicted):
    """The replay's insert keeps the earlier prompt's FULL on [F', F) (F2), so that node must be on the device
    then. An L2-hit replay restores it with its load-back (the earlier prompt evicted before the match); a
    device-hit replay holds it with its lock (evicted before the insert). Either way the replay caches its whole
    prompt, and the session's next turn hits it instead of replaying from F' again."""
    kit = _host_tier_kit()
    try:
        cache, allocator, req, _, _, _ = _host_tier_replay(kit, evict_before_match=evicted == "before the match")
        start, branch, computed = req.prefix_len, 5 * PAGE, len(req.full_untruncated_fill_ids)
        own = kit._alloc(allocator, computed - start)
        _prefill_replay(cache, req, own)
        if evicted == "before the insert":
            cache.evict(EvictParams(num_tokens=len(EARLIER)))
        cache.checkpoint(req, up_to=computed)  # after the prefill: the window at the branch point
        cache.checkpoint(req, up_to=computed)  # the release's insert of the rest
        device_len, _, _ = cache.tree_core.match_full_device_prefix(
            RadixKey(array("q", req.full_untruncated_fill_ids))
        )
        assert device_len == computed, (
            f"the replay cached its prompt only up to {device_len} of {computed}: the earlier prompt's "
            f"[{start}, {branch}) was on the host tier only at its insert"
        )
        nxt = _match(cache, list(req.full_untruncated_fill_ids) + _pages(1, 300_000))
        assert nxt.device_prefix_len == computed and not nxt.swa_branch_replay, (
            f"the next turn hits {nxt.device_prefix_len} (replay {nxt.swa_branch_replay}) of the {computed} cached"
        )
        # [F', F) is the earlier prompt's FULL, not the pages the replay recomputed.
        path = cache.path_device_indices(nxt.last_device_node)
        assert not torch.equal(path[start:branch].cpu(), own[: branch - start].cpu())
    finally:
        kit.doCleanups()


@needs_cuda  # the HiCache controller
def test_a_replay_of_a_prompt_ending_in_the_earlier_prompts_host_only_tail_caches_up_to_it():
    """A prompt that ends inside the earlier prompt, a page past its branch point (the match key stops 128 tokens
    short of the prompt, as the scheduler's decoder-replay cap does): the replay branches at 6 pages, and its insert
    key reaches 7 pages, into the earlier prompt's tail, still on the host tier only. The insert stops there, not at
    the reused prefix: it caches the window at the branch point, and the next turn hits it without a replay."""
    kit = _host_tier_kit()
    try:
        tokens = EARLIER[: 7 * PAGE + 64]
        cap = len(tokens) - 128
        cache, allocator, req, match, _, _ = _host_tier_replay(kit, tokens=tokens, key_cap=cap)
        start, branch = req.prefix_len, 6 * PAGE
        assert (start, match.swa_branching_seqlen) == (branch - REPLAY, branch)
        _prefill_replay(cache, req, kit._alloc(allocator, len(tokens) - start))
        cache.checkpoint(req, up_to=len(tokens))  # after the prefill: the window at the branch point
        cache.checkpoint(req, up_to=len(tokens))  # the release's insert
        nxt = _match(cache, tokens[:cap])
        assert nxt.device_prefix_len == branch and not nxt.swa_branch_replay, (
            f"the next turn hits {nxt.device_prefix_len} (replay {nxt.swa_branch_replay}): the replay's insert "
            f"stopped at {start}, short of the branch point {branch}"
        )
        # The earlier prompt's tail stays its own, on the host tier.
        device_len, _, _ = cache.tree_core.match_full_device_prefix(RadixKey(array("q", EARLIER)))
        assert device_len == branch
        assert cache.tree_core.match_full_prefix_len(RadixKey(array("q", EARLIER))) == len(EARLIER)
    finally:
        kit.doCleanups()


@needs_cuda  # the HiCache controller
def test_a_replay_whose_branch_node_was_not_restored_leaves_it_with_the_earlier_prompts_full_kv():
    """The fallback, when the load-back could not restore the branch node (here: not asked to): [F', F) of the
    earlier prompt is still on the host tier only at the replay's insert, which would put the request's recomputed
    pages on that node as its device FULL KV. So the request inserts nothing past its reused prefix, and its own SWA
    window still slides with what it has computed."""
    kit = _host_tier_kit()
    try:
        cache, allocator, req, _, loaded, _ = _host_tier_replay(kit, load_branch=False)
        computed = len(req.full_untruncated_fill_ids)
        _prefill_replay(cache, req, kit._alloc(allocator, computed - loaded))
        cache.checkpoint(req, up_to=computed)  # a chunk-boundary insert of the whole prompt
        key = RadixKey(array("q", EARLIER))
        device_len, _, _ = cache.tree_core.match_full_device_prefix(key)
        assert device_len == loaded, (
            f"the replay's insert restored the earlier prompt's host-only [{loaded}, {device_len}) on the GPU "
            "from the pages it recomputed from an empty window"
        )
        assert cache.tree_core.match_full_prefix_len(key) == len(EARLIER)
        # The window keeps the last ceil_page(window) rows below page_floor(computed).
        assert req.kv.get_evicted_seqlen(SWA) == computed - 2 * PAGE
    finally:
        kit.doCleanups()


@needs_cuda  # the sparse-prefill index kernels are Triton
def test_the_sparse_prefill_of_a_replay_reads_no_swa_below_its_first_row():
    """P's encoder layers run the sparse FlashMLA prefill, whose chunk cache gathers each request's SWA window
    through full_to_swa. Below a replay's first row the tree holds no SWA (full_to_swa maps those pages to the
    padding slot 0), so every row's window stops at Req.swa_valid_start, in SWA-only and compressed layers
    (DeepseekV4AttnBackend._build_sparse_prefill_chunk_cache, shared by the BF16, Q8KV8 and MegaAttention
    prefills). A request without a replay in the same batch keeps its whole window."""
    from types import SimpleNamespace

    from decoupled_engine.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend

    dev = torch.device("cuda")
    start = 4 * PAGE  # request 0 replays from here: its prefix pages have FULL only
    seq_lens, extend = [start + 2 * PAGE, 2 * PAGE], [2 * PAGE, PAGE]
    req_to_token = torch.arange(2 * 4096, dtype=torch.int32).reshape(2, 4096)
    full_to_swa = torch.zeros(2 * 4096 + 1, dtype=torch.int64)
    full_to_swa[-1] = -1  # as SWATokenToKVPoolAllocator builds it
    full_to_swa[start : seq_lens[0]] = torch.arange(1000, 1000 + extend[0])
    full_to_swa[4096 : 4096 + seq_lens[1]] = torch.arange(5000, 5000 + seq_lens[1])
    rows = torch.cat([torch.arange(s - e + 1, s + 1) for s, e in zip(seq_lens, extend)]).to(torch.int32)
    backend = SimpleNamespace(
        sparse_prefill_kv_buffer=SimpleNamespace(forget_head=lambda: None, budget_rows=0),
        forward_metadata=SimpleNamespace(late_layer_tail=None),
        mega_attn=False,
        req_to_token=req_to_token.to(dev),
        token_to_kv_pool=SimpleNamespace(
            full_to_swa_index_mapping=full_to_swa.to(dev), get_swa_key_page_size=lambda: 64
        ),
    )
    batch = SimpleNamespace(
        seq_lens=torch.tensor(seq_lens, device=dev),
        seq_lens_cpu=torch.tensor(seq_lens),
        extend_seq_lens=torch.tensor(extend, device=dev),
        extend_seq_lens_cpu=extend,
        req_pool_indices=torch.tensor([0, 1], device=dev),
        swa_valid_starts=[start, 0],  # ForwardBatch.init_new: each request's Req.swa_valid_start
    )
    num_rows = sum(extend)
    cache = DeepseekV4AttnBackend._build_sparse_prefill_chunk_cache(
        backend, batch, SimpleNamespace(request_window_layout=None, seq_lens_casual=rows.to(dev)),
        num_qo_tokens=num_rows,
    )
    first_pos, offsets = cache.swa_first_pos.tolist(), cache.swa_offsets.tolist()
    slots = cache.swa_token_ids.tolist()
    # A compressed layer (ratio 2, one top-k column) combines the same windows behind its top-k entries.
    cache.ensure_compressed(2, torch.zeros((num_rows, 12), dtype=torch.int32, device=dev), 64)
    c2_indices, c2_lens = cache.combine_compressed(2, torch.full((num_rows, 1), -1, dtype=torch.int32, device=dev))
    c2_swa_base = cache.compressed[2].swa_base.tolist()

    def windows(indices, lens, swa_base, ratio, topk):
        """Each row's SWA window, behind its top-k entries, as (positions, SWA slots)."""
        out, indices, lens = [], indices.tolist(), lens.tolist()
        for row, pos in enumerate((rows - 1).tolist()):
            r = 0 if row < extend[0] else 1
            at = [e - swa_base[r] for e in indices[row][min((pos + 1) // ratio, topk) : lens[row]]]
            out.append(([first_pos[r] + a for a in at], [slots[offsets[r] + a] for a in at]))
        return out

    for name, got in (
        ("SWA-only", windows(cache.c0_combined_indices, cache.c0_combined_lens, offsets[:-1], 1, 0)),
        ("compressed", windows(c2_indices, c2_lens, c2_swa_base, 2, 1)),
    ):
        replay, plain = got[: extend[0]], got[extend[0] :]
        below = [(p, s) for positions, rslots in replay for p, s in zip(positions, rslots) if p < start]
        assert not below, (
            f"{name}: the replay's rows read {len(below)} SWA rows below its swa_valid_start {start}, from SWA "
            f"slot(s) {sorted({s for _, s in below})}"
        )
        assert replay[0] == ([start], [1000])  # the first row's window is that row alone
        assert replay[WINDOW - 1][0] == list(range(start, start + WINDOW))
        first = seq_lens[1] - extend[1]
        assert plain[0][0] == list(range(first - WINDOW + 1, first + 1))
