"""Core token-chain builder: canonical traces -> tier-A schema-v2 rows.

For each generate the builder renders the prompt and the completed context
with the target chat template, derives the output text as the string
continuation, and maintains a running (text, ids) chain so consecutive
prompts extend the previous prompt+output ids exactly — the property the
radix cache sees at replay. Deltas are encoded incrementally; a full
re-encode happens only when the chain breaks (history rewrites, think-tag
stripping, truncation), which is recorded as data, not treated as an error.

Prefix-fallback rule (documented in README): when the completed render is
not a string extension of the prompt (e.g. an assistant turn without
reasoning under a template whose generation prompt force-appends
``<think>\\n``), the prompt is encoded fresh, the output text is the
completed render minus the longest common prefix with the prompt, the output
ids are boundary-checked against the prompt tail, and the chain never
carries into the next turn.
"""

from __future__ import annotations

import copy
from typing import Any

from strict_convert.canonical import (
    CanonicalTrace,
    CanonicalTurn,
    serialize_tool_calls,
    truncated_output_from_fields,
)
from strict_convert.rendering import BoundaryError, TargetRenderer

_WAIT_ACTORS = ("user", "tool", "other")


def new_stats() -> dict[str, Any]:
    return {
        "traces_out": 0,
        "generates_out": 0,
        "chained": 0,
        "fresh": 0,
        "breaks_by_reason": {},
        "prefix_fallback_events": 0,
        "truncated_events": 0,
        "boundary_repairs": 0,
        "dropped_events": {},
        "dropped_traces": {},
        "prompt_tokens": 0,
        "output_tokens": 0,
        "max_context_per_trace": [],
        "oracle_drift": [],
        "role_mappings": {},
    }


def merge_stats(total: dict[str, Any], part: dict[str, Any]) -> None:
    for key, value in part.items():
        if isinstance(value, dict):
            bucket = total.setdefault(key, {})
            for sub, count in value.items():
                bucket[sub] = bucket.get(sub, 0) + count
        elif isinstance(value, list):
            total.setdefault(key, []).extend(value)
        else:
            total[key] = total.get(key, 0) + value


def _bump(bucket: dict[str, int], reason: str) -> None:
    bucket[reason] = bucket.get(reason, 0) + 1


class _TraceDrop(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _ChainState:
    """Running (text, ids) state through the last completed turn."""

    def __init__(self) -> None:
        self.text: str | None = None
        self.ids: list[int] | None = None
        self.next_break_reason = "first_generate"
        # "emission": ids end with the recorded output; "history": ids end
        # with the re-fed history form instead (reasoning never re-fed), so
        # the next prompt extends the previous prompt but not prompt+output.
        self.anchor: str | None = None

    def set(self, text: str, ids: list[int], anchor: str = "emission") -> None:
        self.text = text
        self.ids = ids
        self.next_break_reason = "context_rewrite"
        self.anchor = anchor

    def reset(self, reason: str) -> None:
        self.text = None
        self.ids = None
        self.next_break_reason = reason
        self.anchor = None


def build_trace(
    trace: CanonicalTrace,
    renderer: TargetRenderer,
    adapter: str,
    stats: dict[str, Any],
) -> dict[str, Any] | None:
    """Build one schema-v2 row; returns None (and counts why) on a drop.

    Per-trace statistics accumulate into a fresh partial and merge into
    ``stats`` only on success, so a mid-build drop never leaves phantom
    generate counts behind; dropped traces account their source turns under
    ``dropped_events.generates_in_dropped_traces``.
    """
    partial = new_stats()
    try:
        row = _build(trace, renderer, adapter, partial)
    except _TraceDrop as drop:
        _bump(stats["dropped_traces"], drop.reason)
        dropped = stats["dropped_events"]
        dropped["generates_in_dropped_traces"] = (
            dropped.get("generates_in_dropped_traces", 0) + len(trace.turns)
        )
        return None
    merge_stats(stats, partial)
    return row


def _build(
    trace: CanonicalTrace,
    renderer: TargetRenderer,
    adapter: str,
    stats: dict[str, Any],
) -> dict[str, Any]:
    role_counts: dict[str, int] = {}
    initial_messages, counts = renderer.remap_roles(trace.initial_messages, 0)
    if trace.history_refed:
        initial_messages = _inline_structured_messages(
            initial_messages, renderer, stats
        )
    _merge_counts(role_counts, counts)

    state_messages = list(initial_messages)
    chain = _ChainState()
    events: list[dict[str, Any]] = []
    chain_records: list[dict[str, Any]] = []
    original_counts: list[dict[str, Any]] = []
    max_context = 0
    render_failure: str | None = None
    turns_done = 0

    for turn in trace.turns:
        if renderer.reference is not None and renderer.thinking_mode == "chat" and turn.assistant.get("reasoning_content"):
            raise _TraceDrop("reasoning_output_requires_thinking_mode")
        pre_events, state_messages = _context_events(
            turn, state_messages, renderer, role_counts,
            stats, trace.history_refed,
        )
        try:
            prompt_text = renderer.render(
                state_messages, tools=trace.tools, add_generation_prompt=True
            )
        except Exception as exc:  # jinja raise_exception / normalization errors
            render_failure = f"prompt render failed: {exc}"
            break
        events.extend(pre_events)

        prompt_ids, chained, break_reason = _prompt_ids(
            prompt_text, chain, renderer, stats
        )
        prompt_anchor = chain.anchor if chained else None

        emission, history = _emission_and_history(turn, renderer, stats)
        try:
            output_text, output_ids, rule = _output_ids(
                turn, emission, state_messages, trace.tools,
                prompt_text, prompt_ids, chain, renderer, stats,
            )
        except _RenderFailure as exc:
            render_failure = str(exc)
            break

        turns_done += 1
        if not output_ids:
            state_messages = state_messages + [history]
            events.append({"type": "append_message", "message": history})
            _bump(stats["dropped_events"], "empty_output_demoted")
            chain.reset("prev_demoted")
            continue

        if history is not emission and chain.text is not None:
            # The recorded output carries reasoning/tool-call text the source
            # serving stack never re-feeds; later prompts extend the history
            # render, so the chain must re-anchor on it.
            _rechain_on_history(
                history, state_messages, trace.tools,
                prompt_text, prompt_ids, chain, renderer, stats,
            )
        state_messages = state_messages + [history]

        event_index = len(events)
        events.append(
            _generate_event(turn, trace, turn.assistant if renderer.reference is not None else history, prompt_ids, output_ids)
        )
        record = {
            "event_index": event_index,
            "prompt": "chained" if chained else "fresh",
            "break_reason": break_reason,
            "output_rule": rule,
            "truncated": turn.truncated,
        }
        if prompt_anchor is not None:
            record["anchored_on"] = prompt_anchor
        chain_records.append(record)
        if turn.original_counts is not None:
            original_counts.append(
                {"event_index": event_index, **turn.original_counts}
            )
        if turn.oracle_output_tokens is not None:
            stats["oracle_drift"].append(len(output_ids) - turn.oracle_output_tokens)

        stats["generates_out"] += 1
        stats["chained" if chained else "fresh"] += 1
        if break_reason is not None:
            _bump(stats["breaks_by_reason"], break_reason)
        if rule == "prefix_fallback":
            stats["prefix_fallback_events"] += 1
        if turn.truncated:
            stats["truncated_events"] += 1
        stats["prompt_tokens"] += len(prompt_ids)
        stats["output_tokens"] += len(output_ids)
        max_context = max(max_context, len(prompt_ids) + len(output_ids))

    if not any(event["type"] == "generate" for event in events):
        raise _TraceDrop(
            "render_failure_no_generates" if render_failure else "no_generates"
        )

    metadata = dict(trace.metadata)
    metadata.setdefault("routing_key", trace.trace_id)
    # Adapters may stage extra provenance destined for the in-band
    # strict_conversion namespace (e.g. the synthetic adapter's shape spec).
    extras = metadata.pop("strict_conversion_extras", None)
    conversion: dict[str, Any] = {
        "adapter": adapter,
        "target_tokenizer": renderer.identity_summary(),
        "chain": chain_records,
        "rendering": renderer.rendering,
    }
    if trace.history_refed:
        conversion["history_refed"] = True
    if isinstance(extras, dict):
        for key, value in extras.items():
            conversion.setdefault(key, value)
    if original_counts:
        conversion["original_counts"] = original_counts
    if role_counts:
        conversion["role_mappings"] = role_counts
    if render_failure:
        conversion["truncated_by_render_failure"] = render_failure
        _bump(stats["dropped_events"], "render_failure_truncated_trace")
        remaining = len(trace.turns) - turns_done
        if remaining:
            stats["dropped_events"]["generates_dropped_after_render_failure"] = (
                stats["dropped_events"].get(
                    "generates_dropped_after_render_failure", 0
                )
                + remaining
            )
    metadata["strict_conversion"] = conversion

    _merge_counts(stats["role_mappings"], role_counts)
    stats["traces_out"] += 1
    stats["max_context_per_trace"].append(max_context)
    return {
        "schema_version": 2,
        "trace_id": trace.trace_id,
        "start_offset_s": max(float(trace.start_offset_s), 0.0),
        "initial_messages": initial_messages,
        "tools": trace.tools,
        "events": events,
        "metadata": metadata,
    }


def _merge_counts(total: dict[str, int], part: dict[str, int]) -> None:
    for key, value in part.items():
        total[key] = total.get(key, 0) + value


def _inline_structured_messages(
    messages: list[dict[str, Any]],
    renderer: TargetRenderer,
    stats: dict[str, Any],
) -> list[dict[str, Any]]:
    """Inline dropped structured fields of recorded assistant messages.

    Only used for history_refed traces: their context ops carry recorded
    assistant messages whose reasoning/tool calls the capture demonstrably
    re-fed, so a template that drops the fields must render them from
    content or the prompt loses that volume.
    """
    counters = stats.setdefault("adapter_counters", {})
    out: list[dict[str, Any]] = []
    for message in messages:
        reasoning = message.get("reasoning_content")
        calls = message.get("tool_calls")
        force = getattr(renderer, "force_history_inline", False)
        inline_reasoning = bool((renderer.reasoning_field_dropped or force) and reasoning)
        inline_calls = bool((renderer.tool_calls_dropped or force) and calls)
        if message.get("role") != "assistant" or not (
            inline_reasoning or inline_calls
        ):
            out.append(message)
            continue
        clone = copy.deepcopy(message)
        base = clone.get("content") or ""
        think = f"<think>{reasoning}</think>" if inline_reasoning else ""
        calls_text = serialize_tool_calls(calls) if inline_calls else ""
        clone["content"] = f"{think}{base}{calls_text}"
        if inline_reasoning:
            clone.pop("reasoning_content", None)
        _bump(counters, "context_assistant_structured_inlined")
        out.append(clone)
    return out


def _context_events(
    turn: CanonicalTurn,
    state_messages: list[dict[str, Any]],
    renderer: TargetRenderer,
    role_counts: dict[str, int],
    stats: dict[str, Any],
    inline_structured: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    for seconds, actor in turn.waits_before:
        events.append(
            {
                "type": "wait",
                "seconds": max(float(seconds), 0.0),
                "actor": actor if actor in _WAIT_ACTORS else "other",
            }
        )
    for op in turn.context_ops:
        if op[0] == "replace":
            messages, counts = renderer.remap_roles(op[1], 0)
            if inline_structured:
                messages = _inline_structured_messages(messages, renderer, stats)
            _merge_counts(role_counts, counts)
            event: dict[str, Any] = {"type": "replace_context", "messages": messages}
            if len(op) > 2 and op[2] in ("compaction", "reset", "branch", "other"):
                event["reason"] = op[2]
            events.append(event)
            state_messages = list(messages)
        elif op[0] == "append":
            messages, counts = renderer.remap_roles(op[1], len(state_messages))
            if inline_structured:
                messages = _inline_structured_messages(messages, renderer, stats)
            _merge_counts(role_counts, counts)
            for message in messages:
                events.append({"type": "append_message", "message": message})
            state_messages = state_messages + list(messages)
        else:
            raise ValueError(f"unknown context op kind: {op[0]!r}")
    return events, state_messages


def _prompt_ids(
    prompt_text: str,
    chain: _ChainState,
    renderer: TargetRenderer,
    stats: dict[str, Any],
) -> tuple[list[int], bool, str | None]:
    if chain.text is not None and prompt_text.startswith(chain.text):
        assert chain.ids is not None
        delta = prompt_text[len(chain.text) :]
        try:
            delta_ids = renderer.continuation_ids(chain.ids, delta) if delta else []
            return chain.ids + delta_ids, True, None
        except BoundaryError:
            stats["boundary_repairs"] += 1
            break_reason = "prompt_boundary"
    else:
        break_reason = chain.next_break_reason
    prompt_ids = renderer.encode(prompt_text)
    if renderer.decode(prompt_ids) != prompt_text:
        raise _TraceDrop("prompt_not_losslessly_tokenizable")
    return prompt_ids, False, break_reason


class _RenderFailure(Exception):
    pass


def _emission_and_history(
    turn: CanonicalTurn,
    renderer: TargetRenderer,
    stats: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a recorded turn into its emission and history forms.

    The emission is the full recorded generation (reasoning + content + tool
    calls) and defines the output ids. History is what the source serving
    stack re-feeds on later turns: content and tool calls, never reasoning
    (GLM/OpenAI-style APIs do not re-send it — GLM's own template renders
    reasoning only after the last user turn). For templates that drop the
    structured fields, both forms inline them into content; the history form
    keeps the structured fields so replay-side context reconstruction and
    validation still see them. When the two forms coincide the same object
    is returned twice and the caller keeps the single-render fast path.
    """
    original = copy.deepcopy(turn.assistant)
    original["role"] = "assistant"
    counters = stats.setdefault("adapter_counters", {})
    reasoning = original.get("reasoning_content")
    calls = original.get("tool_calls")
    inline_reasoning = bool(renderer.reasoning_field_dropped and reasoning)
    inline_calls = bool(renderer.tool_calls_dropped and calls)
    # A template that renders the reasoning field would leak reasoning into
    # re-fed history; strip it from the history form in that case.
    feeds_reasoning = bool(reasoning) and not renderer.reasoning_field_dropped
    if not inline_reasoning and not inline_calls and not feeds_reasoning:
        return original, original
    if turn.history_is_emission:
        # The source demonstrably re-fed the full raw emission (recorded ids
        # chain across turns), so history and emission coincide.
        emission = copy.deepcopy(original)
        if inline_reasoning:
            emission.pop("reasoning_content", None)
            base = original.get("content") or ""
            calls_text = serialize_tool_calls(calls) if inline_calls else ""
            emission["content"] = f"<think>{reasoning}</think>{base}{calls_text}"
            _bump(counters, "reasoning_inlined_for_dropping_template")
            if inline_calls:
                _bump(counters, "tool_calls_inlined_for_dropping_template")
        elif inline_calls:
            emission["content"] = (
                f"{original.get('content') or ''}{serialize_tool_calls(calls)}"
            )
            _bump(counters, "tool_calls_inlined_for_dropping_template")
        _bump(counters, "history_refed_emission")
        return emission, emission

    base = original.get("content") or ""
    calls_text = serialize_tool_calls(calls) if inline_calls else ""
    if inline_calls:
        _bump(counters, "tool_calls_inlined_for_dropping_template")

    emission = copy.deepcopy(original)
    if inline_reasoning:
        # Reconstruct the original inline emission ("<think>...</think>" +
        # content) so the dropping template renders it verbatim.
        emission.pop("reasoning_content", None)
        emission["content"] = f"<think>{reasoning}</think>{base}{calls_text}"
        _bump(counters, "reasoning_inlined_for_dropping_template")
    elif inline_calls:
        emission["content"] = f"{base}{calls_text}"

    if inline_calls or feeds_reasoning:
        history = copy.deepcopy(original)
        history["content"] = f"{base}{calls_text}"
        if feeds_reasoning:
            history.pop("reasoning_content", None)
    else:
        history = original
    return emission, history


def _rechain_on_history(
    history: dict[str, Any],
    state_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    prompt_text: str,
    prompt_ids: list[int],
    chain: _ChainState,
    renderer: TargetRenderer,
    stats: dict[str, Any],
) -> None:
    """Re-anchor the chain on the history render of the completed turn."""
    try:
        completed = renderer.render(
            state_messages + [history], tools=tools, add_generation_prompt=False
        )
    except Exception:
        chain.reset("history_render_failed")
        return
    if not completed.startswith(prompt_text):
        chain.reset("history_prefix_mismatch")
        return
    delta = completed[len(prompt_text):]
    try:
        delta_ids = renderer.continuation_ids(prompt_ids, delta) if delta else []
    except BoundaryError:
        stats["boundary_repairs"] += 1
        chain.reset("history_boundary")
        return
    chain.set(completed, prompt_ids + delta_ids, anchor="history")


def _output_ids(
    turn: CanonicalTurn,
    emission: dict[str, Any],
    state_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    prompt_text: str,
    prompt_ids: list[int],
    chain: _ChainState,
    renderer: TargetRenderer,
    stats: dict[str, Any],
) -> tuple[str, list[int], str]:
    """Derive (output_text, output_ids, rule) and advance the chain state."""
    if renderer.reference is not None and turn.truncated:
        # A length stop has no generated EOS. Mid-think stops must also stay
        # open; the source fields, not a previously inlined message, own the text.
        if turn.truncated_raw_output is not None:
            output_text = turn.truncated_raw_output
        elif turn.assistant.get("reasoning_content") and not (
            turn.assistant.get("content") or turn.assistant.get("tool_calls")
        ):
            output_text = turn.assistant["reasoning_content"]
        else:
            partial = copy.deepcopy(emission)
            partial["wo_eos"] = True
            completed = renderer.render(state_messages + [partial], tools=tools, add_generation_prompt=False)
            if not completed.startswith(prompt_text):
                raise _RenderFailure("DSV4 truncated completion does not extend prompt")
            output_text = completed[len(prompt_text):]
        output_ids = _continuation_with_repair(renderer, prompt_ids, prompt_text, output_text, stats)
        chain.reset("prev_truncated")
        return output_text, output_ids, "truncated_raw"
    if turn.truncated and not turn.assistant.get("tool_calls"):
        if turn.truncated_raw_output is not None:
            output_text = turn.truncated_raw_output
        else:
            # Rebuild from the source fields, not the emission form — its
            # inlined "<think>" content would double the markup here.
            output_text = truncated_output_from_fields(
                turn.assistant.get("reasoning_content"),
                turn.assistant.get("content"),
            )
        if not output_text:
            chain.reset("prev_truncated")
            return output_text, [], "truncated_raw"
        output_ids = _continuation_with_repair(
            renderer, prompt_ids, prompt_text, output_text, stats
        )
        chain.reset("prev_truncated")
        return output_text, output_ids, "truncated_raw"

    try:
        completed_text = renderer.render(
            state_messages + [emission], tools=tools, add_generation_prompt=False
        )
    except Exception as exc:
        raise _RenderFailure(f"completed render failed: {exc}") from exc

    if completed_text.startswith(prompt_text):
        output_text = completed_text[len(prompt_text) :]
        if not output_text:
            chain.reset("prev_demoted")
            return output_text, [], "continuation"
        output_ids = _continuation_with_repair(
            renderer, prompt_ids, prompt_text, output_text, stats
        )
        if turn.truncated:
            # truncated turn that still carries tool calls: keep the
            # template-rendered continuation but never chain across it.
            chain.reset("prev_truncated")
            _bump(
                stats.setdefault("adapter_counters", {}),
                "truncated_with_tool_calls_rendered",
            )
        else:
            chain.set(completed_text, prompt_ids + output_ids)
        return output_text, output_ids, "continuation"

    # Prefix property failed (e.g. assistant without think while the
    # generation prompt force-appends "<think>\n"). Clamp the raw character
    # LCP so it never eats into the template's forced generation suffix:
    # past the shared assistant-turn header, prompt characters ("<think>\n")
    # can match assistant content only by coincidence ("<t" of "<table>"),
    # and consuming them would destroy the head of the recorded output.
    common = _common_prefix_length(prompt_text, completed_text)
    structural_limit = len(prompt_text) - renderer.forced_suffix_len
    if common > structural_limit:
        common = structural_limit
    output_text = completed_text[common:]
    if not output_text:
        chain.reset("prev_prefix_fallback")
        return output_text, [], "prefix_fallback"
    output_ids = _continuation_with_repair(
        renderer, prompt_ids, prompt_text, output_text, stats
    )
    chain.reset("prev_prefix_fallback")
    return output_text, output_ids, "prefix_fallback"


def _continuation_with_repair(
    renderer: TargetRenderer,
    prompt_ids: list[int],
    prompt_text: str,
    output_text: str,
    stats: dict[str, Any],
) -> list[int]:
    try:
        return renderer.continuation_ids(prompt_ids, output_text)
    except BoundaryError:
        stats["boundary_repairs"] += 1
        full_ids = renderer.encode(prompt_text + output_text)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise _TraceDrop("output_boundary_failure") from None
        candidate = full_ids[len(prompt_ids) :]
        # Mirror the fresh-prompt path: repaired ids must decode back to the
        # exact recorded text, or the chain state (which stores the
        # unnormalized text) would poison every later chained prompt (e.g.
        # NFD text that the tokenizer NFC-normalizes).
        if renderer.decode(prompt_ids + candidate) != prompt_text + output_text:
            raise _TraceDrop("output_not_losslessly_tokenizable") from None
        return candidate


def _common_prefix_length(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    low, high = 0, limit
    while low < high:
        mid = (low + high + 1) // 2
        if a[:mid] == b[:mid]:
            low = mid
        else:
            high = mid - 1
    return low


def _generate_event(
    turn: CanonicalTurn,
    trace: CanonicalTrace,
    assistant: dict[str, Any],
    prompt_ids: list[int],
    output_ids: list[int],
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "generate",
        "max_tokens": len(output_ids),
        "recorded_assistant": assistant,
        "recorded_prompt_token_ids": prompt_ids,
        "recorded_output_token_ids": output_ids,
    }
    if turn.source_request_id:
        event["source_request_id"] = turn.source_request_id
    if trace.source_model is not None:
        event["source_model"] = trace.source_model
    if turn.source_timestamp is not None:
        event["source_timestamp"] = turn.source_timestamp
    return event
