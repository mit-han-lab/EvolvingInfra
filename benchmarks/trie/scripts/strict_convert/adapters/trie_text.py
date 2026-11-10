"""Adapter: schema-v2 tier-C text traces -> tier-A (putnam, browsecomp-glm5.2).

The source rows already carry the event stream and a structured
``recorded_assistant`` per generate; only the token ids are missing. Each
generate is re-rendered with the target tokenizer. Truncation comes from the
per-request source metadata (``finish_reason == "length"`` for the putnam
mini-swe capture, ``status == "incomplete"`` for the browsecomp pi harness).
Original per-event token counts land in
``metadata.strict_conversion.original_counts`` because generate events have
no free-form metadata in the schema.
"""

from __future__ import annotations

from typing import Any, Iterator

from strict_convert.adapters import _trie_common
from strict_convert.canonical import CanonicalTrace


def list_units(options: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    return _trie_common.list_jsonl_units(options["inputs"])


def prepare_options(options: dict[str, Any], units: list[dict[str, Any]]) -> None:
    del options, units


def make_context(options: dict[str, Any], renderer: Any) -> None:
    del options, renderer
    return None


def _assistant_hook(
    event: dict[str, Any],
    info: dict[str, Any] | None,
    stats: dict[str, Any],
) -> tuple[dict[str, Any], bool, int | None, dict[str, Any]]:
    del stats
    assistant = event.get("recorded_assistant")
    if assistant is None:
        raise _trie_common.MissingAssistant()
    return assistant, _trie_common.request_is_truncated(info), None, {}


def convert_unit(
    unit: dict[str, Any],
    context: None,
    options: dict[str, Any],
    stats: dict[str, Any],
) -> Iterator[CanonicalTrace]:
    del context, options
    row = _trie_common.load_row(unit)
    try:
        yield from _trie_common.walk_schema_v2_row(row, stats, _assistant_hook)
    except _trie_common.MissingAssistant:
        stats["dropped_traces"]["generate_without_recorded_assistant"] = (
            stats["dropped_traces"].get("generate_without_recorded_assistant", 0) + 1
        )
