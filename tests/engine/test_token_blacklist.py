# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""SGLANG_V41_DECODE_TOKEN_BLACKLIST: the listed token ids get -inf on the target's
next-token logits, including a target-verify vocab shard. The helpers load by path
(no torch); the LogitsProcessor tests need the engine and torch."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ENGINE = Path(__file__).resolve().parents[2] / "evolving_infra/engine"
_spec = importlib.util.spec_from_file_location(
    "token_blacklist", ENGINE / "decoupled_engine/srt/sampling/token_blacklist.py"
)
blacklist = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(blacklist)


def test_parse():
    assert blacklist.parse_token_blacklist("") == ()
    assert blacklist.parse_token_blacklist("1") == (1,)
    assert blacklist.parse_token_blacklist(" 1, 7,1 ,") == (1, 7)
    with pytest.raises(ValueError):
        blacklist.parse_token_blacklist("-1")


def test_local_columns_of_a_vocab_shard():
    assert blacklist.local_blacklist_columns((1, 70), 0, 64) == [1]
    assert blacklist.local_blacklist_columns((1, 70), 64, 64) == [6]
    assert blacklist.local_blacklist_columns((1, 70), 128, 64) == []


@pytest.fixture
def engine():
    torch = pytest.importorskip("torch")
    pytest.importorskip("decoupled_engine")
    from decoupled_engine.srt.layers import logits_processor

    return SimpleNamespace(torch=torch, module=logits_processor)


def _processor(engine, token_ids, *, defer=False, vocab_size=8):
    processor = object.__new__(engine.module.LogitsProcessor)
    processor._token_blacklist = tuple(token_ids)
    processor.defer_verify_gather = defer
    processor.vocab_size = vocab_size
    return processor


def test_apply_sets_columns_to_minus_inf(engine):
    torch = engine.torch
    logits = torch.zeros(3, 8)
    blacklist.apply_token_blacklist(logits, (1, 5))
    assert torch.isneginf(logits[:, [1, 5]]).all()
    assert int(torch.isneginf(logits).sum()) == 6
    assert torch.all(torch.softmax(logits, dim=-1)[:, 1] == 0)
    assert blacklist.apply_token_blacklist(None, (1,)) is None


def test_processor_masks_full_vocab(engine):
    torch = engine.torch
    logits = torch.zeros(2, 8)
    _processor(engine, (1,))._mask_token_blacklist(logits)
    assert torch.isneginf(logits[:, 1]).all()
    assert int(torch.isneginf(logits).sum()) == 2


def test_processor_without_blacklist_keeps_logits(engine):
    torch = engine.torch
    logits = torch.zeros(2, 8)
    _processor(engine, ())._mask_token_blacklist(logits)
    assert not torch.isneginf(logits).any()


def test_deferred_vocab_shard_uses_its_rank_offset(engine):
    torch = engine.torch
    parallel = SimpleNamespace(tp_group=SimpleNamespace(rank_in_group=1))
    with patch.object(engine.module, "get_parallel", return_value=parallel):
        shard = torch.zeros(2, 4)  # rank 1 holds vocab columns 4..7
        _processor(engine, (1, 6), defer=True)._mask_token_blacklist(shard)
    assert torch.isneginf(shard[:, 2]).all()
    assert int(torch.isneginf(shard).sum()) == 2
