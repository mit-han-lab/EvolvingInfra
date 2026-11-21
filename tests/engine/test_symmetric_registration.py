# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Native registration and bounded page copies without CUDA import dependencies.

Compile the GPU-independent definitions from their source files. The address
math and copy engine are the production implementations, not test replicas.
"""
import ast
import enum
import logging
import math
from pathlib import Path
import threading
from types import SimpleNamespace
import typing

import pytest

msgspec = pytest.importorskip("msgspec")
SRT = Path(__file__).resolve().parents[2] / "evolving_infra/engine/decoupled_engine/srt"


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in selected} == names
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *selected,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture(scope="module")
def native():
    ns = dict(vars(typing), msgspec=msgspec, math=math, enum=enum, threading=threading,
              PROTOCOL_VERSION=1, logger=logging.getLogger(__name__))
    definitions(SRT / "mem_cache/memory_pool.py", {"KvBufferDesc"}, ns)
    definitions(SRT / "disaggregation/pull/protocol.py", {
        "ProtocolError", "PoolRegion", "ItemRegion", "PoolLayout", "Registration",
        "_pool_region_from_layout", "_contiguous_runs", "build_page_read_plan",
    }, ns)
    ns.update(POOL_FULL="full", POOL_SWA="swa", POOL_C4="c4", POOL_C128="c128", POOL_C1="c1", POOL_C2="c2",
              COMPRESSED_POOLS={4: "c4", 128: "c128", 1: "c1", 2: "c2"},
              StateType=enum.Enum("StateType", "SWA DSA DSV4_REQUEST_STATE"))
    export_table = SRT / "disaggregation/symmetric/export_table.py"
    defined = {node.name for node in ast.parse(export_table.read_text()).body if isinstance(node, ast.FunctionDef)}
    helpers = {"_is_unified_kv", "_dsv4_stage_ratios", "_dsv4_page_pools", "_dsv4_pool_entries"} & defined
    definitions(export_table, {"_is_dsv4_pool", "registration_pools_for"} | helpers, ns)
    definitions(SRT / "disaggregation/pull/read_engine.py", {"FakeMemory", "FakeReadEngine"}, ns)
    return SimpleNamespace(**ns)


@pytest.mark.parametrize("page_size", [1, 4, 64, 128])
@pytest.mark.parametrize("pool_classes", [
    ("MHATokenToKVPool",),
    ("MHATokenToKVPool", "MHATokenToKVPoolFP4"),
    ("MLATokenToKVPool",),
    # Wrappers whose get_contiguous_buf_infos returns their inner pool's values.
    ("HybridLinearKVPool",),
    ("SWAKVPool",),
    ("MiniMaxSparseKVPool",),
], ids=lambda names: names[-1])
@pytest.mark.parametrize(("source_pages", "target_pages"), [
    ([1], [8]), ([1, 2, 4, 5], [5, 6, 1, 2]), ([8, 3], [1, 7]),
])
def test_registration_copies_pages_within_the_registered_span(
        native, page_size, pool_classes, source_pages, target_pages):
    cls = object
    for name in pool_classes:
        cls = type(name, (cls,), {})
    pool = cls()
    row_bytes, size = 2, 8 * page_size
    descriptor = native.KvBufferDesc("k", (size + page_size, 1), row_bytes=row_bytes, tokens_per_row=1)
    span = descriptor.final_span_bytes(size, page_size)
    stride = descriptor.item_len_bytes(page_size)

    def register(engine, fill):
        ptrs = [engine.memory.alloc(span, fill=fill) for _ in range(2)]
        args = SimpleNamespace(
            page_size=page_size, kv_data_ptrs=ptrs, kv_item_lens=[stride, stride],
            kv_layer_ids=[0, 0], aux_data_ptrs=[], aux_item_lens=[],
            state_types=[], state_data_ptrs=[], state_item_lens=[], engine_rank=0,
        )
        layouts, extra = native.registration_pools_for(kvcache=pool, kv_args=args)
        assert not extra
        registration = native.Registration.from_kv_args(
            kv_args=args, session_id=engine.session_id, attn_tp_size=1, pool_layouts=layouts)
        return registration.pools["full"], ptrs

    owner = native.FakeReadEngine(session_id="owner")
    reader = native.FakeReadEngine(session_id="reader")
    try:
        remote, source_ptrs = register(owner, 0)
        local, target_ptrs = register(reader, 255)
        for entry, ptr in enumerate(source_ptrs):
            for page in range(9):
                owner.memory.view(ptr + page * stride, stride)[:] = bytes([16 * entry + page]) * stride
        local_ptrs, remote_ptrs, lengths = native.build_page_read_plan(
            local=local, remote=remote, local_pages=target_pages, remote_pages=source_pages)
        assert sum(lengths) == len(source_pages) * page_size * row_bytes * 2
        assert reader.batch_read("owner", local_ptrs, remote_ptrs, lengths) == 0
        mapping = dict(zip(target_pages, source_pages))
        for entry, ptr in enumerate(target_ptrs):
            for page in range(9):
                value = 16 * entry + mapping[page] if page in mapping else 255
                assert bytes(reader.memory.view(ptr + page * stride, stride)) == bytes([value]) * stride
    finally:
        owner.close()
        reader.close()


def dsv4_cache(kv_layers, index_buffers, ratios):
    """A DeepSeek-V4 cache as registration reads it: per compression ratio (in the
    pool's order) its KV buffers and its indexer's page-row buffers."""
    cache = type("DeepSeekV4TokenToKVPool", (), {})()
    cache.compression_ratios, cache._stage_start, cache._stage_end = ratios, 0, len(ratios)
    cache._unified_kv, cache.c4_kv_pool = False, None
    cache.kv_pools = {r: SimpleNamespace(kv_buffer=[object()] * n) if n else None for r, n in kv_layers.items()}
    cache.index_pools = {r: SimpleNamespace(contiguous_page_row_buffers=lambda n=n: [object()] * n)
                         for r, n in index_buffers.items()}
    return cache


@pytest.mark.parametrize(("kv_layers", "index_buffers", "ratios", "pools"), [
    # DeepSeek-V4: the c4 KV and indexer buffers, then the c128 KV buffers.
    ({4: 2, 128: 1}, {4: 2}, [0, 4, 128, 4], {"c4": [0, 1, 2, 3], "c128": [4]}),
    # DeepSeek-V4.1: one ratio-1 and three ratio-2 layers with their indexers, no c4 or c128 layers.
    ({4: 0, 128: 0, 1: 1, 2: 3}, {1: 1, 2: 3}, [0, 1, 2, 2, 2, 0],
     {"c4": [], "c128": [], "c1": [0, 1], "c2": [2, 3, 4, 5, 6, 7]}),
], ids=["v4", "v4.1"])
def test_dsv4_registration_gives_each_compression_ratio_a_full_indexed_pool(
        native, kv_layers, index_buffers, ratios, pools):
    cache = dsv4_cache(kv_layers, index_buffers, ratios)
    entries = sum(len(e) for e in pools.values())
    args = SimpleNamespace(
        page_size=256, kv_data_ptrs=[1000 * (e + 1) for e in range(entries)],
        kv_item_lens=[64 * (e + 1) for e in range(entries)], kv_layer_ids=list(range(entries)),
        aux_data_ptrs=[], aux_item_lens=[], state_types=[native.StateType.SWA],
        state_data_ptrs=[[7]], state_item_lens=[[8]], engine_rank=0,
    )
    layouts, extra = native.registration_pools_for(kvcache=cache, kv_args=args)
    assert {layout.name: list(layout.entry_indices) for layout in layouts} == {"full": [], **pools}
    assert all(list(layout.page_nbytes) == [64 * (e + 1) for e in layout.entry_indices] for layout in layouts)
    assert list(extra) == ["swa"]


def test_dsv4_swa_pages_carry_a_drafts_swa_layers(native):
    """setup_state_kv_args lists a DSpark or NextN draft's SWA layers as a second
    StateType.SWA component on the target's SWA page indices. A reader that imports
    a prompt's SWA tail gets the draft's window with the target's: without it the
    reader's draft attends to whatever its pages held before."""
    cache = dsv4_cache({4: 0, 128: 0, 1: 1, 2: 3}, {1: 1, 2: 3}, [0, 1, 2, 2, 2, 0])
    pages = 9
    strides = [6, 6, 4]  # two target SWA layers, then the draft's

    def register(engine, fill):
        target = [engine.memory.alloc(pages * strides[0], fill=fill) for _ in range(2)]
        draft = [engine.memory.alloc(pages * strides[2], fill=fill)]
        args = SimpleNamespace(
            page_size=256, kv_data_ptrs=list(range(1, 9)), kv_item_lens=[64] * 8,
            kv_layer_ids=list(range(8)), aux_data_ptrs=[], aux_item_lens=[],
            state_types=[native.StateType.SWA, native.StateType.DSV4_REQUEST_STATE, native.StateType.SWA],
            state_data_ptrs=[target, [7], draft], state_item_lens=[strides[:2], [8], strides[2:]],
            engine_rank=0,
        )
        _, extra = native.registration_pools_for(kvcache=cache, kv_args=args)
        return extra["swa"], target + draft

    owner = native.FakeReadEngine(session_id="owner")
    reader = native.FakeReadEngine(session_id="reader")
    try:
        remote, source_ptrs = register(owner, 0)
        local, target_ptrs = register(reader, 255)
        for entry, (ptr, stride) in enumerate(zip(source_ptrs, strides)):
            for page in range(pages):
                owner.memory.view(ptr + page * stride, stride)[:] = bytes([16 * entry + page]) * stride
        source_pages, target_pages = [2, 3], [5, 6]
        local_ptrs, remote_ptrs, lengths = native.build_page_read_plan(
            local=local, remote=remote, local_pages=target_pages, remote_pages=source_pages)
        assert reader.batch_read("owner", local_ptrs, remote_ptrs, lengths) == 0
        mapping = dict(zip(target_pages, source_pages))
        for entry, (ptr, stride) in enumerate(zip(target_ptrs, strides)):
            for page in range(pages):
                value = 16 * entry + mapping[page] if page in mapping else 255
                got = bytes(reader.memory.view(ptr + page * stride, stride))
                assert got == bytes([value]) * stride, f"SWA layer {entry} page {page} not imported"
    finally:
        owner.close()
        reader.close()
