"""Canonical intermediate representation shared by every corpus adapter.

Adapters translate a source corpus into ``CanonicalTrace`` objects; the core
builder (builder.py) turns those into schema-v2 rows. The dataclass style
matches the vendored ``trie.types`` module (msgspec is unavailable in the
trace-conversion venv, see README).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# A context operation applied before a generate:
#   ("replace", messages, reason)  -> replace_context event
#   ("append", [message, ...])     -> one append_message event per message
ContextOp = tuple


@dataclass
class CanonicalTurn:
    assistant: dict[str, Any]
    context_ops: list[ContextOp] = field(default_factory=list)
    waits_before: list[tuple[float, str]] = field(default_factory=list)
    truncated: bool = False
    truncated_raw_output: str | None = None
    oracle_output_tokens: int | None = None
    original_counts: dict[str, Any] | None = None
    source_request_id: str | None = None
    source_timestamp: str | None = None
    # True when the source's NEXT recorded prompt extends prompt+output at
    # the token level: that capture re-fed the full raw emission (reasoning
    # included) into history, so the history form must equal the emission.
    # Only derivable when source token ids exist; defaults to the
    # reasoning-not-refed convention otherwise.
    history_is_emission: bool = False


@dataclass
class CanonicalTrace:
    trace_id: str
    turns: list[CanonicalTurn]
    initial_messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    start_offset_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    source_model: str | None = None
    # Trace-level form of CanonicalTurn.history_is_emission: enough recorded
    # links prove the capture re-fed raw emissions, so recorded assistant
    # messages inside context ops must also render with their structured
    # fields inlined (a per-turn replace_context otherwise discards the
    # builder's own history form entirely).
    history_refed: bool = False


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def split_inline_think(text: str) -> tuple[str, str]:
    """Split assistant text with inline think tags into (reasoning, content).

    Mirrors the think-tag extraction used by the supported source traces
    (split on the tags plus ``strip('\\n')``), so a canonical assistant with
    structured ``reasoning_content`` re-renders to exactly the same context
    text as the raw inline form the serving harness stored. Text without a
    closing tag is returned unchanged as content (the template renders it
    verbatim); truncated mid-think turns must be handled by the caller
    before this function.
    """
    if THINK_CLOSE not in text:
        return "", text
    reasoning = (
        text.split(THINK_CLOSE)[0].strip("\n").split(THINK_OPEN)[-1].strip("\n")
    )
    content = text.split(THINK_CLOSE)[-1].strip("\n")
    return reasoning, content


def strip_think_prefix(text: str) -> str:
    """Drop a leading think-open tag the prompt already forces.

    A target generation prompt may already end with ``<think>\\n``; recorded
    raw emissions can re-include the opening tag. Replaying them verbatim
    would duplicate it, so strip one leading ``<think>\\n`` or ``<think>``.
    """
    for prefix in (THINK_OPEN + "\n", THINK_OPEN):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def truncated_output_from_fields(reasoning: str | None, content: str | None) -> str:
    """Raw emission text for a length-stopped turn from structured fields.

    Mid-think stops replay the recorded reasoning with no closing tag. When
    visible content exists the model did close the think block, so the
    emission is reconstructed with the template's canonical separators.
    """
    raw = strip_think_prefix(reasoning or "")
    if content:
        raw = raw + "\n" + THINK_CLOSE + "\n\n" + content
    return raw


def serialize_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """Deterministic inline text for tool calls a target template drops.

    Tier-A conversion is token-count-authoritative, not byte-faithful to the
    target model's native tool markup; this compact form keeps the recorded
    call volume in the emission and in re-fed history (sort_keys pins the
    byte form so re-conversion is reproducible).
    """
    import json as _json

    parts: list[str] = []
    for call in tool_calls:
        function = (call or {}).get("function") or {}
        name = function.get("name") or ""
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = (
                _json.dumps(arguments, ensure_ascii=False, sort_keys=True)
                if arguments is not None
                else ""
            )
        parts.append(f"<tool_call>{name}\n{arguments}</tool_call>")
    return "".join(parts)


def assistant_message(
    reasoning_content: str | None,
    content: str | None,
    tool_calls: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message
