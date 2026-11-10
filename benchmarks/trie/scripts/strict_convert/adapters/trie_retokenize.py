"""Adapter: schema-v2 tier-A traces under a different tokenizer.

The current project use is GLM-5.2 source traces replayed by DeepSeek-V4-Flash.
Foreign token ids must never cross that tokenizer boundary even when their
numeric values happen to be in range. Source ids are used only to recover the
authoritative emitted text and any missing ``reasoning_content``; target ids
are generated afresh. Source counts and SHA-256 values remain in
``metadata.strict_conversion.original_counts`` for provenance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterator

from strict_convert._vendor import tokenizer_manager_module, types_module
from strict_convert.adapters import _trie_common
from strict_convert.canonical import (
    CanonicalTrace,
    strip_think_prefix,
)

THINK_CLOSE = "</think>"


def list_units(options: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    return _trie_common.list_jsonl_units(options["inputs"])


def prepare_options(options: dict[str, Any], units: list[dict[str, Any]]) -> None:
    del units
    source = options.get("source_tokenizer")
    if not source:
        raise ValueError("trie_retokenize requires --source-tokenizer")
    if _tokenizer_json_sha(source) == _tokenizer_json_sha(options["target_tokenizer"]):
        raise ValueError(
            "source and target tokenizers are identical; this adapter exists "
            "to re-tokenize across vocabularies and refuses same-tokenizer "
            "runs (pass the source ids through unchanged instead)"
        )


def _tokenizer_json_sha(model_path: str) -> str | None:
    tokenizer_json = Path(model_path) / "tokenizer.json"
    if not tokenizer_json.is_file():
        return None
    return hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()


class _Context:
    def __init__(self, options: dict[str, Any]) -> None:
        manager_cls = tokenizer_manager_module().TokenizerManager
        self.source = manager_cls(options["source_tokenizer"])
        self.token_ids_sha256 = types_module().token_ids_sha256


def make_context(options: dict[str, Any], renderer: Any) -> _Context:
    del renderer
    return _Context(options)


def _make_hook(context: _Context) -> _trie_common.AssistantHook:
    def hook(
        event: dict[str, Any],
        info: dict[str, Any] | None,
        stats: dict[str, Any],
    ) -> tuple[dict[str, Any], bool, int | None, dict[str, Any]]:
        output_ids = event.get("recorded_output_token_ids")
        if output_ids is None:
            raise _trie_common.MissingAssistant()
        assistant = dict(event.get("recorded_assistant") or {"role": "assistant"})
        decoded = context.source.decode(output_ids)
        if not assistant.get("reasoning_content"):
            body = strip_think_prefix(decoded)
            if THINK_CLOSE in body:
                recovered = body.split(THINK_CLOSE)[0]
                if recovered:
                    assistant["reasoning_content"] = recovered
                    counters = stats.setdefault("adapter_counters", {})
                    counters["reasoning_recovered_from_decode"] = (
                        counters.get("reasoning_recovered_from_decode", 0) + 1
                    )
        prompt_ids = event.get("recorded_prompt_token_ids")
        provenance: dict[str, Any] = {
            "source_output_token_count": len(output_ids),
            "source_output_token_ids_sha256": context.token_ids_sha256(output_ids),
        }
        if prompt_ids is not None:
            provenance["source_prompt_token_count"] = len(prompt_ids)
            provenance["source_prompt_token_ids_sha256"] = context.token_ids_sha256(
                prompt_ids
            )
        return (
            assistant,
            _trie_common.request_is_truncated(info),
            None,
            provenance,
        )

    return hook


def convert_unit(
    unit: dict[str, Any],
    context: _Context,
    options: dict[str, Any],
    stats: dict[str, Any],
) -> Iterator[CanonicalTrace]:
    del options
    row = _trie_common.load_row(unit)
    try:
        yield from _trie_common.walk_schema_v2_row(row, stats, _make_hook(context))
    except _trie_common.MissingAssistant:
        stats["dropped_traces"]["generate_without_source_token_ids"] = (
            stats["dropped_traces"].get("generate_without_source_token_ids", 0) + 1
        )
