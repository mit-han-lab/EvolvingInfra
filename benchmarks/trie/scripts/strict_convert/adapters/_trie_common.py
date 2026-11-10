"""Shared plumbing for corpora that are already trie schema-v2 JSONL.

Both trie adapters walk the recorded event stream (append_message /
replace_context / wait / generate), maintain the message state exactly the
way the replay client does, and hand each generate to the core builder as a
canonical turn. They differ only in how the assistant turn is derived, which
is injected as a hook.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterator

from strict_convert.canonical import CanonicalTrace, CanonicalTurn

_TRACE_ID_RE = re.compile(rb'"trace_id"\s*:\s*"([^"]+)"')


class MissingAssistant(Exception):
    """A tier-C generate without recorded_assistant cannot be re-rendered."""

# hook(generate_event, request_info, stats)
#   -> (assistant, truncated, oracle_tokens, extra_original_counts)
AssistantHook = Callable[
    [dict[str, Any], dict[str, Any] | None, dict[str, Any]],
    tuple[dict[str, Any], bool, int | None, dict[str, Any]],
]


def list_jsonl_units(paths: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """One unit per JSONL line; streams for offsets, never keeps lines.

    trace_id is sniffed from the line head and, failing that, the line tail:
    some corpora (edgebench) serialize events first, leaving trace_id at the
    end of multi-hundred-MB lines. ``bytes`` feeds size-aware scheduling.
    """
    units: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in paths:
        offset = 0
        line_number = 0
        with open(path, "rb") as source:
            for line in source:
                if line.strip():
                    match = _TRACE_ID_RE.search(line[:4096])
                    if match is None:
                        match = _TRACE_ID_RE.search(line[-16384:])
                    trace_id = match.group(1).decode("utf-8") if match else ""
                    units.append(
                        {
                            "name": f"{path}:{line_number}",
                            "path": path,
                            "offset": offset,
                            "trace_id": trace_id,
                            "bytes": len(line),
                        }
                    )
                offset += len(line)
                line_number += 1
        if not units:
            warnings.append(f"no traces found in {path}")
    return units, warnings


def load_row(unit: dict[str, Any]) -> dict[str, Any]:
    with open(unit["path"], "rb") as source:
        source.seek(unit["offset"])
        return json.loads(source.readline())


def _request_infos(row: dict[str, Any]) -> list[dict[str, Any]]:
    requests = (row.get("metadata") or {}).get("requests")
    return requests if isinstance(requests, list) else []


def request_is_truncated(info: dict[str, Any] | None) -> bool:
    if not info:
        return False
    return info.get("finish_reason") == "length" or info.get("status") == "incomplete"


def scalar_fields(info: dict[str, Any] | None) -> dict[str, Any]:
    if not info:
        return {}
    return {
        key: value
        for key, value in info.items()
        if isinstance(value, (str, int, float, bool))
    }


def _generates(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in row.get("events", []) if e.get("type") == "generate"]


def _history_refed_flags(row: dict[str, Any]) -> dict[int, bool]:
    """Per-generate flags: did the source re-feed the full emission?

    When consecutive generates carry recorded token ids and the next prompt
    extends prompt+output exactly, that capture kept the raw emission
    (reasoning included) in history — e.g. a GLM agent loop whose tool
    results are not user turns, so the template's last_user_index keeps
    every think block. The flag drives the builder's history form per turn.
    """
    gens = [e for e in row.get("events", []) if e.get("type") == "generate"]
    flags: dict[int, bool] = {}
    for index in range(len(gens) - 1):
        prompt = gens[index].get("recorded_prompt_token_ids")
        output = gens[index].get("recorded_output_token_ids")
        nxt = gens[index + 1].get("recorded_prompt_token_ids")
        if not prompt or not output or not nxt:
            continue
        realized_len = len(prompt) + len(output)
        if len(nxt) >= realized_len and (
            nxt[: len(prompt)] == prompt
            and nxt[len(prompt) : realized_len] == output
        ):
            flags[index] = True
    return flags


def walk_schema_v2_row(
    row: dict[str, Any],
    stats: dict[str, Any],
    assistant_hook: AssistantHook,
) -> Iterator[CanonicalTrace]:
    stats["traces_in"] += 1
    requests = _request_infos(row)
    turns: list[CanonicalTurn] = []
    pending_ops: list[tuple] = []
    pending_waits: list[tuple[float, str]] = []
    generate_index = 0
    refed_flags = _history_refed_flags(row)
    for event in row.get("events", []):
        event_type = event.get("type")
        if event_type == "wait":
            pending_waits.append(
                (event.get("seconds", 0.0), event.get("actor", "other"))
            )
        elif event_type == "append_message":
            pending_ops.append(("append", [event["message"]]))
        elif event_type == "replace_context":
            pending_ops.append(
                ("replace", event["messages"], event.get("reason", "other"))
            )
        elif event_type == "generate":
            stats["generates_in"] += 1
            info = requests[generate_index] if generate_index < len(requests) else None
            assistant, truncated, oracle, extra_counts = assistant_hook(
                event, info, stats
            )
            original_counts = {"max_tokens": event.get("max_tokens")}
            original_counts.update(scalar_fields(info))
            original_counts.update(extra_counts)
            turns.append(
                CanonicalTurn(
                    assistant=assistant,
                    context_ops=pending_ops,
                    waits_before=pending_waits,
                    truncated=truncated,
                    oracle_output_tokens=oracle,
                    original_counts=original_counts,
                    source_request_id=event.get("source_request_id")
                    or (info or {}).get("request_id"),
                    source_timestamp=event.get("source_timestamp"),
                    history_is_emission=refed_flags.get(generate_index, False),
                )
            )
            pending_ops, pending_waits = [], []
            generate_index += 1
    dropped = stats["dropped_events"]
    trailing = len(pending_ops) + len(pending_waits)
    if trailing:
        dropped["trailing_non_generate"] = (
            dropped.get("trailing_non_generate", 0) + trailing
        )
    if not turns:
        stats["dropped_traces"]["no_source_generates"] = (
            stats["dropped_traces"].get("no_source_generates", 0) + 1
        )
        return
    source_model = next(
        (
            event.get("source_model")
            for event in row.get("events", [])
            if event.get("type") == "generate" and event.get("source_model")
        ),
        (row.get("metadata") or {}).get("model"),
    )
    gens = _generates(row)
    verifiable_links = sum(
        1
        for a, b in zip(gens, gens[1:])
        if a.get("recorded_prompt_token_ids")
        and a.get("recorded_output_token_ids")
        and b.get("recorded_prompt_token_ids")
    )
    refed_links = sum(1 for flag in refed_flags.values() if flag)
    yield CanonicalTrace(
        trace_id=row["trace_id"],
        turns=turns,
        initial_messages=row.get("initial_messages") or [],
        tools=row.get("tools"),
        start_offset_s=row.get("start_offset_s") or 0.0,
        metadata=dict(row.get("metadata") or {}),
        source_model=source_model,
        # Arbitrary; no evidence 0.3 is the right threshold — it only needs
        # to separate "convention" (edgebench ~half of links, the rest are
        # compaction boundaries) from stray coincidences.
        history_refed=bool(
            verifiable_links and refed_links / verifiable_links >= 0.3
        ),
    )
