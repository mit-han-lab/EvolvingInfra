"""Token content for block-run events of hash-block traces (WekaTrace and similar corpora).

Such traces record a prompt only as a sequence of fixed-size block ids; equal ids mean equal
content. A generate event can carry its prompt, and its output, as runs of block ids,
``[[first_id, count], ...]``, which the client expands just before sending. Each block id maps
to ``block_size`` token ids drawn from the trace's token pool by a keyed 64-bit mix of
(seed, block id, position), so equal block ids give equal tokens and the prefix sharing the
server sees is exactly the trace's. The trace metadata holds the parameters:

    "block_tokens": {"block_size": 64, "seed": <int>, "pool": [<token id>, ...]}

A trace without ``pool`` is not bound to a model: the client draws from the regular vocabulary
of the tokenizer it replays with (``pool_from_tokenizer``), which is the pool the converters
embed for that tokenizer.

Every replay of a trace (``replay_cycle_id``) gets its own content, so concurrent or repeated
replays of one trace never share a prefix, whatever the cache salt covers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_MASK = (1 << 64) - 1
_K_BLOCK = np.uint64(0x9E3779B97F4A7C15)
_K_POS = np.uint64(0xD1B54A32D192ED03)
_K_CYCLE = 0xA0761D6478BD642F
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31 = np.uint64(30), np.uint64(27), np.uint64(31)
_MAX_TOKEN_ID = 4294967295


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer on wrapping uint64 arithmetic."""
    x = x ^ (x >> _S30)
    x = x * _M1
    x = x ^ (x >> _S27)
    x = x * _M2
    return x ^ (x >> _S31)


def validate_block_runs(runs: Any, field: str) -> None:
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{field} must be a non-empty list of [first_id, count] pairs")
    for run in runs:
        if (
            not isinstance(run, list)
            or len(run) != 2
            or any(type(v) is not int for v in run)
            or run[0] < 0
            or run[1] < 1
            or run[0] + run[1] > _MASK
        ):
            raise ValueError(f"{field} must be a non-empty list of [first_id, count] pairs")


def run_blocks(runs: Sequence[Sequence[int]]) -> int:
    return sum(count for _, count in runs)


def pool_from_tokenizer(tokenizer_dir: str | Path) -> list[int]:
    """Regular vocabulary ids of the tokenizer in ``tokenizer_dir``, sorted: everything except
    added and special tokens."""
    directory = Path(tokenizer_dir)
    data = json.loads((directory / "tokenizer.json").read_text())
    vocab = data["model"]["vocab"]
    ids = set(vocab.values() if isinstance(vocab, dict) else range(len(vocab)))
    ids -= {token["id"] for token in data.get("added_tokens", [])}
    # Some models declare PAD/BOS/EOS only in config.json, not added_tokens.
    config_path = directory / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
            value = config.get(name)
            ids.difference_update(value if isinstance(value, list) else [value])
        if "vocab_size" in config:
            ids = {token for token in ids if 0 <= token < config["vocab_size"]}
    return sorted(ids)


def validate_block_spec(spec: Any) -> None:
    if not isinstance(spec, dict):
        raise ValueError("metadata.block_tokens must be an object")
    size, seed, pool = spec.get("block_size"), spec.get("seed"), spec.get("pool")
    if type(size) is not int or size <= 0:
        raise ValueError("metadata.block_tokens.block_size must be a positive integer")
    if type(seed) is not int or not 0 <= seed <= _MASK:
        raise ValueError("metadata.block_tokens.seed must be an unsigned 64-bit integer")
    if pool is not None and (
        not isinstance(pool, list)
        or len(pool) < 2
        or any(type(t) is not int or not 0 <= t <= _MAX_TOKEN_ID for t in pool)
    ):
        raise ValueError("metadata.block_tokens.pool must list at least two token ids")


class BlockTokens:
    """Expands block runs of one replay of one trace."""

    def __init__(
        self,
        spec: dict[str, Any],
        replay_cycle_id: int = 0,
        pool: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        """``pool`` serves a spec without its own (a trace not bound to a model)."""
        validate_block_spec(spec)
        self.block_size = spec["block_size"]
        if spec.get("pool") is not None:
            pool = spec["pool"]
        if pool is None or len(pool) < 2:
            raise ValueError("the trace has no block_tokens.pool: give the tokenizer to draw it from")
        self._pool = np.asarray(pool, dtype=np.int64)
        self._n_pool = np.uint64(len(self._pool))
        self._seed = np.uint64((spec["seed"] ^ (replay_cycle_id * _K_CYCLE)) & _MASK)
        self._pos = np.arange(self.block_size, dtype=np.uint64) * _K_POS

    def tokens(self, runs: Sequence[Sequence[int]], n_tokens: int | None = None) -> list[int]:
        """Token ids of the blocks in `runs`, truncated to `n_tokens` when given."""
        ids = np.concatenate(
            [np.arange(first, first + count, dtype=np.uint64) for first, count in runs]
        )
        x = _mix((ids[:, None] * _K_BLOCK + self._pos[None, :]) ^ self._seed)
        out = self._pool[(x % self._n_pool).astype(np.intp)].ravel()
        if n_tokens is not None:
            if n_tokens > out.size:
                raise ValueError(f"block runs hold {out.size} tokens, {n_tokens} requested")
            out = out[:n_tokens]
        return out.tolist()
