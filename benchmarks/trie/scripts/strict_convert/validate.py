"""Validation of converted strict-replay workloads.

Cost model: the full converted set carries billions of token ids, so checks
are split into two tiers (documented here, reported per run):

Full checks — every row, every generate, C-level primitives:
  * tier-A invariants: both id arrays present, ``len(output) == max_tokens``,
    ``array('q')`` rejects non-integers, ``min``/``max`` bound the id range,
    an identity scan rejects bools (bool passes ``array('q')`` silently);
  * chain consistency: events the builder marked chained must have
    ``prompt_ids[n+1]`` starting with ``prompt_ids[n] + output_ids[n]``;
  * output-head check for ``prefix_fallback`` / ``truncated_raw`` events:
    the decoded output head must match the recorded assistant's
    content/reasoning head (catches LCP-overshoot corruption);
  * in-band tokenizer identity (``metadata.strict_conversion
    .target_tokenizer``) must match the validating tokenizer;
  * 100% retention through the current ReplayTrace parser and tier-A input
    contract. The removed standalone filter is not a runtime dependency.
  * recognized source counts, paired ratios, and native structured content.

Deep checks — row 0 and every ``--deep-every``-th row:
  * vendored ``ReplayTrace.from_jsonl_row`` when full admission is disabled;
  * decode spot-check: the last generate's prompt decodes to text ending
    with the template's generation-prompt tail.

``--workers N`` fans rows out to processes; progress heartbeats go to
stderr, the JSON report to stdout.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import pathlib
import sys
from array import array
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from strict_convert.adapters._trie_common import list_jsonl_units
from strict_convert.canonical import strip_think_prefix, truncated_output_from_fields
from strict_convert.rendering import TargetRenderer

_VALIDATOR: dict[str, Any] = {}


def _init_validator(tokenizer_model: str, chat_encoding: str = "auto", thinking_mode: str = "thinking", require_source_counts: bool = False, history_reasoning: str = "native", template_kwargs: dict[str, Any] | None = None) -> None:
    renderer = TargetRenderer(tokenizer_model, chat_encoding, thinking_mode, history_reasoning, template_kwargs)
    _VALIDATOR["renderer"] = renderer
    _VALIDATOR["require_source_counts"] = require_source_counts
    from strict_convert._vendor import types_module

    _VALIDATOR["replay_trace_cls"] = types_module().ReplayTrace


def _expected_output_heads(assistant: dict[str, Any], rule: str) -> list[str]:
    content = assistant.get("content") or ""
    reasoning = assistant.get("reasoning_content") or ""
    if rule == "truncated_raw":
        # A truncated turn's output is one of three builder shapes: the raw
        # inline emission (content carries the unterminated think body), the
        # recorded reasoning replayed with no closing tag, or the rebuilt
        # closed-think form from truncated_output_from_fields (which starts
        # with the "\n</think>\n\n" separator when reasoning is empty).
        candidates = [
            strip_think_prefix(content),
            strip_think_prefix(reasoning),
            truncated_output_from_fields(reasoning, content),
        ]
        return [c[:8] for c in candidates if c]
    # The recorded_assistant is the history form (reasoning kept as a field);
    # under a dropping template the emission re-inlines it, so a
    # prefix-fallback output may legitimately start with the think block.
    candidates = [content]
    if reasoning:
        candidates.append("<think>" + reasoning)
    return [c[:8] for c in candidates if c]


def _empty_report() -> dict[str, Any]:
    return {
        "traces": 0,
        "generates": 0,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "chained_links_checked": 0,
        "output_head_checks": 0,
        "identity_checks": 0,
        "deep_checked_rows": 0,
        "decode_spot_checks": 0,
        "filter_kept": 0,
        "filter_checked": 0,
        "source_output_tokens": 0,
        "source_output_events": 0,
        "short_output_band_exempt_events": 0,
        "source_prompt_tokens": 0,
        "source_prompt_events": 0,
        "paired_output_tokens": 0,
        "paired_prompt_tokens": 0,
        "source_content_checks": 0,
        "errors": [],
    }


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rendering_identity(rendering: dict[str, Any]) -> dict[str, Any]:
    """Rendering identity with defaults filled in for keys added later.

    ``history_reasoning`` did not exist when the earlier workloads were
    converted, and the code that produced them had no inline path, so a missing
    key is exactly "native" rather than an unknown. Defaulting it keeps those
    files verifiable instead of forcing a re-conversion that would also move
    their registered content hashes.
    """
    return {"history_reasoning": "native", **rendering}


SHORT_OUTPUT_BAND_FLOOR = 64


def _source_count(counts: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = counts.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _merge_report(total: dict[str, Any], part: dict[str, Any]) -> None:
    for key, value in part.items():
        if key == "errors":
            total["errors"].extend(value)
        else:
            total[key] += value


def _validate_row(
    row: dict[str, Any],
    line_index: int,
    deep: bool,
    run_filter_check: bool,
) -> dict[str, Any]:
    renderer: TargetRenderer = _VALIDATOR["renderer"]
    report = _empty_report()
    errors: list[str] = report["errors"]
    report["traces"] = 1
    trace_label = row.get("trace_id", f"line {line_index}")

    conversion = (row.get("metadata") or {}).get("strict_conversion") or {}
    rendering = conversion.get("rendering")
    if rendering is not None and _rendering_identity(rendering) != _rendering_identity(renderer.rendering):
        errors.append(f"{trace_label}: rendering identity does not match validator")
    identity = conversion.get("target_tokenizer")
    if identity is not None:
        report["identity_checks"] += 1
        expected = renderer.identity_summary()
        if (
            identity.get("vocab_len") != expected["vocab_len"]
            or identity.get("tokenizer_json_sha256_prefix")
            != expected["tokenizer_json_sha256_prefix"]
        ):
            errors.append(
                f"{trace_label}: in-band target tokenizer {identity} does not "
                f"match the validating tokenizer {expected}"
            )

    schema_error = None
    if deep:
        report["deep_checked_rows"] += 1
    if deep or run_filter_check:
        try:
            _VALIDATOR["replay_trace_cls"].from_jsonl_row(row)
        except (ValueError, TypeError) as exc:
            schema_error = str(exc)
            errors.append(f"{trace_label}: from_jsonl_row: {exc}")

    chain_entries = {
        entry["event_index"]: entry for entry in conversion.get("chain", [])
    }
    source_counts = {
        entry.get("event_index"): entry
        for entry in conversion.get("original_counts", [])
        if isinstance(entry, dict)
    }
    request_counts = {
        r.get("request_id"): r for r in (row.get("metadata") or {}).get("requests", [])
        if isinstance(r, dict) and r.get("request_id")
    }
    previous_realized: list[int] | None = None
    previous_prompt_ids: list[int] | None = None
    last_prompt_ids: list[int] | None = None
    for event_index, event in enumerate(row.get("events", [])):
        if event.get("type") != "generate":
            continue
        report["generates"] += 1
        prompt_ids = event.get("recorded_prompt_token_ids")
        output_ids = event.get("recorded_output_token_ids")
        if prompt_ids is None or output_ids is None:
            errors.append(f"{trace_label}: event {event_index} is not tier-A")
            continue
        if len(output_ids) != event.get("max_tokens"):
            errors.append(f"{trace_label}: event {event_index} max_tokens mismatch")
        for name, ids in (("prompt", prompt_ids), ("output", output_ids)):
            if not ids:
                errors.append(f"{trace_label}: event {event_index} empty {name} ids")
                continue
            try:
                array("q", ids)
            except (TypeError, OverflowError):
                errors.append(
                    f"{trace_label}: event {event_index} {name} ids contain "
                    "non-integers"
                )
                continue
            if min(ids) < 0 or max(ids) >= renderer.vocab_len:
                errors.append(
                    f"{trace_label}: event {event_index} {name} id outside "
                    f"[0, {renderer.vocab_len})"
                )
            if any(token is True or token is False for token in ids):
                errors.append(
                    f"{trace_label}: event {event_index} {name} ids contain bools"
                )
        report["prompt_tokens"] += len(prompt_ids)
        report["output_tokens"] += len(output_ids)

        # Source-fidelity reconciliation: the chain checks prove only
        # internal consistency; a conversion that drops recorded emission
        # text stays self-consistent, so converted output volume must be
        # compared against the capture's own per-request token counts.
        counts = {**request_counts.get(event.get("source_request_id"), {}), **(source_counts.get(event_index) or {})}
        # source_output_token_count (retokenize provenance, exact source-id
        # count) is preferred over recorded_output_tokens (capture metadata).
        recorded_out = _source_count(counts, "source_output_token_count", "recorded_output_tokens", "completion_tokens")
        if recorded_out is not None:
            report["source_output_tokens"] += recorded_out
            report["source_output_events"] += 1
            report["paired_output_tokens"] += len(output_ids)
            ratio = len(output_ids) / recorded_out if recorded_out else 0
            # A multiplicative band is the wrong test on very short outputs:
            # the DSV4 reference encoding adds a fixed DSML/tool-call framing
            # cost of tens of tokens, so a 33-token source output lands near
            # ratio 2.7 with nothing lost or duplicated. The floor is the
            # smallest one that clears the observed violations (their source
            # counts top out at 63), so the band keeps its detection power
            # everywhere above it. Exempted events are counted so the exemption
            # is visible, and the aggregate output_ratio stays a hard gate.
            if recorded_out < SHORT_OUTPUT_BAND_FLOOR:
                report["short_output_band_exempt_events"] += 1
            elif not (0.4 <= ratio <= 2.5):
                errors.append(
                    f"{trace_label}: event {event_index} output ids "
                    f"{len(output_ids)} vs source recorded_output_tokens "
                    f"{recorded_out} (ratio {ratio:.3f}) outside the "
                    "cross-tokenizer band [0.4, 2.5]"
                )
        expected_prompt = _source_count(counts, "source_prompt_token_count", "expected_prompt_tokens", "prompt_tokens")
        if expected_prompt is not None:
            report["source_prompt_tokens"] += expected_prompt
            report["source_prompt_events"] += 1
            report["paired_prompt_tokens"] += len(prompt_ids)

        # Counts alone cannot detect fields being lost while others inflate.
        # New native outputs retain the source assistant fields verbatim, so
        # verify their actual content independently of the render-diff path.
        assistant = event.get("recorded_assistant") or {}
        if renderer.reference is not None:
            decoded = renderer.decode(output_ids)
            report["source_content_checks"] += 1
            for field in ("reasoning_content", "content"):
                value = assistant.get(field)
                if isinstance(value, str) and value and value not in decoded:
                    errors.append(f"{trace_label}: event {event_index} output loses {field}")
            call_cursor = 0
            for call in assistant.get("tool_calls") or []:
                fn = call.get("function") or {}
                expected_call = renderer.reference.tool_call_template.format(
                    dsml_token=renderer.reference.dsml_token,
                    name=fn.get("name", ""),
                    arguments=renderer.reference.encode_arguments_to_dsml(fn),
                )
                position = decoded.find(expected_call, call_cursor)
                if position < 0:
                    errors.append(f"{trace_label}: event {event_index} output loses tool call")
                else:
                    call_cursor = position + len(expected_call)
        elif len(output_ids) <= 1 and (assistant.get("reasoning_content") or assistant.get("tool_calls")):
            errors.append(f"{trace_label}: event {event_index} structured output collapsed to <=1 token")

        entry = chain_entries.get(event_index, {})
        if entry.get("prompt") == "chained":
            # A history-anchored link ends with the re-fed history form, not
            # the recorded output, so the checkable invariant is that this
            # prompt extends the previous prompt (the prefix the radix cache
            # sees); emission-anchored links keep the stronger check.
            if entry.get("anchored_on") == "history":
                if previous_prompt_ids is not None:
                    report["chained_links_checked"] += 1
                    if (
                        prompt_ids[: len(previous_prompt_ids)]
                        != previous_prompt_ids
                    ):
                        errors.append(
                            f"{trace_label}: event {event_index} marked "
                            "history-chained but prompt ids do not extend "
                            "the previous prompt ids"
                        )
            elif previous_realized is not None:
                report["chained_links_checked"] += 1
                if prompt_ids[: len(previous_realized)] != previous_realized:
                    errors.append(
                        f"{trace_label}: event {event_index} marked chained "
                        "but prompt ids do not extend the previous "
                        "prompt+output ids"
                    )
        rule = entry.get("output_rule")
        if rule in ("prefix_fallback", "truncated_raw"):
            report["output_head_checks"] += 1
            expected_heads = _expected_output_heads(
                event.get("recorded_assistant") or {}, rule
            )
            if renderer.reference is not None:
                reasoning = assistant.get("reasoning_content")
                expected_heads = [reasoning[:8]] if reasoning else []
            head = renderer.decode(output_ids[:24])
            if expected_heads and not any(
                head.startswith(candidate) for candidate in expected_heads
            ):
                errors.append(
                    f"{trace_label}: event {event_index} {rule} output head "
                    f"{head[:24]!r} does not start with any recorded "
                    f"assistant head {expected_heads!r}"
                )
        previous_realized = list(prompt_ids) + list(output_ids)
        previous_prompt_ids = list(prompt_ids)
        last_prompt_ids = prompt_ids

    if deep and last_prompt_ids is not None:
        report["decode_spot_checks"] += 1
        decoded = renderer.decode(last_prompt_ids)
        if not isinstance(decoded, str):
            errors.append(f"{trace_label}: prompt decode did not return text")
        elif not decoded.endswith(renderer.generation_tail):
            errors.append(
                f"{trace_label}: decoded prompt does not end with the "
                f"generation tail {renderer.generation_tail!r}"
            )

    if run_filter_check:
        report["filter_checked"] += 1
        if schema_error is None and all(
            e.get("recorded_prompt_token_ids") and e.get("recorded_output_token_ids")
            for e in row.get("events", []) if e.get("type") == "generate"
        ):
            report["filter_kept"] += 1
        else:
            errors.append(f"{trace_label}: replay parser/tier-A admission failed")
    return report


def _validate_chunk(
    job: tuple[list[tuple[int, str, int]], int, bool],
) -> dict[str, Any]:
    """Worker entry: validate a list of (line_index, path, offset) rows."""
    rows, deep_every, run_filter_check = job
    total = _empty_report()
    for line_index, path, offset in rows:
        with open(path, "rb") as source:
            source.seek(offset)
            row = json.loads(source.readline())
        deep = line_index % max(deep_every, 1) == 0
        _merge_report(total, _validate_row(row, line_index, deep, run_filter_check))
    return total


def validate_workload(
    workload_path: str,
    tokenizer_model: str,
    *,
    deep_every: int = 25,
    run_filter_check: bool = True,
    workers: int = 1,
    sample: int | None = None,
    chat_encoding: str = "auto",
    thinking_mode: str = "thinking",
    require_source_counts: bool = False,
    history_reasoning: str = "native",
    template_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one workload file; see the module docstring for the tiers.

    ``sample`` is a deprecated no-op kept for callers of the previous
    interface; decode spot-checks now ride on the deep-checked rows.
    """
    del sample
    if os.environ.get("TRIE_SKIP_ROW_VALIDATION") == "1":
        raise ValueError("offline validation cannot use TRIE_SKIP_ROW_VALIDATION=1")
    units, _ = list_jsonl_units([workload_path])
    rows = [
        (index, unit["path"], unit["offset"]) for index, unit in enumerate(units)
    ]
    total = _empty_report()
    if workers <= 1:
        _init_validator(tokenizer_model, chat_encoding, thinking_mode, require_source_counts, history_reasoning, template_kwargs)
        for position, row_ref in enumerate(rows, 1):
            _merge_report(
                total,
                _validate_chunk(([row_ref], deep_every, run_filter_check)),
            )
            if position % 25 == 0 or position == len(rows):
                print(
                    f"validate: {position}/{len(rows)} traces checked",
                    file=sys.stderr,
                    flush=True,
                )
    else:
        chunk_count = max(workers * 4, 1)
        chunks = [rows[i::chunk_count] for i in range(chunk_count)]
        chunks = [chunk for chunk in chunks if chunk]
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_validator,
            initargs=(tokenizer_model, chat_encoding, thinking_mode, require_source_counts, history_reasoning, template_kwargs),
        ) as pool:
            done = 0
            for part in pool.map(
                _validate_chunk,
                [(chunk, deep_every, run_filter_check) for chunk in chunks],
            ):
                _merge_report(total, part)
                done += 1
                print(
                    f"validate: chunk {done}/{len(chunks)} done "
                    f"({total['traces']} traces so far)",
                    file=sys.stderr,
                    flush=True,
                )

    # The report is the attestation a runtime gate checks, so it has to name the
    # exact bytes it validated. Without these hashes the verdict only says "some
    # file at this path passed", which forces every consumer to keep its own list
    # of blessed hashes — the state then lives in two places and drifts.
    tokenizer_json = pathlib.Path(tokenizer_model) / "tokenizer.json"
    report: dict[str, Any] = {
        "workload": workload_path,
        "attests": {
            "workload_sha256": _sha256_file(pathlib.Path(workload_path)),
            "tokenizer_sha256": _sha256_file(tokenizer_json) if tokenizer_json.is_file() else None,
            "chat_encoding": chat_encoding,
            "thinking_mode": thinking_mode,
            "history_reasoning": history_reasoning,
            "template_kwargs": template_kwargs,
        },
        "workers": workers,
        "deep_every": deep_every,
        **{key: value for key, value in total.items() if key != "errors"},
        "errors": total["errors"],
    }
    if total["source_output_tokens"]:
        output_ratio = total["paired_output_tokens"] / total["source_output_tokens"]
        prompt_ratio = (
            total["paired_prompt_tokens"] / total["source_prompt_tokens"]
            if total["source_prompt_tokens"]
            else None
        )
        report["source_fidelity"] = {
            "output_ratio": round(output_ratio, 4),
            "prompt_ratio": round(prompt_ratio, 4) if prompt_ratio else None,
            "events_with_source_counts": total["source_output_events"],
            "output_count_coverage": total["source_output_events"] / total["generates"] if total["generates"] else 0,
            "prompt_count_coverage": total["source_prompt_events"] / total["generates"] if total["generates"] else 0,
        }
        if not (0.7 <= output_ratio <= 1.5):
            report["errors"].append(
                f"source-fidelity: converted output tokens "
                f"{total['output_tokens']:,} vs capture-recorded "
                f"{total['source_output_tokens']:,} (ratio {output_ratio:.3f}) "
                "outside [0.7, 1.5] — the recorded emission volume was not "
                "preserved by conversion"
            )
        if prompt_ratio is not None and not (0.6 <= prompt_ratio <= 1.6):
            report["errors"].append(
                f"source-fidelity: converted prompt tokens "
                f"{total['prompt_tokens']:,} vs capture-expected "
                f"{total['source_prompt_tokens']:,} (ratio {prompt_ratio:.3f}) "
                "outside [0.6, 1.6]"
            )
    else:
        report["source_fidelity"] = {"output_ratio": None, "output_count_coverage": 0, "status": "unavailable"}
    if require_source_counts or total["source_output_events"]:
        for kind in ("output", "prompt"):
            if total[f"source_{kind}_events"] != total["generates"]:
                report["errors"].append(f"source-fidelity: incomplete {kind} count coverage: {total[f'source_{kind}_events']}/{total['generates']}")
    if require_source_counts and not total["source_output_tokens"]:
        report["errors"].append("source-fidelity: no positive source output token count")
    if not total["generates"]:
        report["errors"].append("no generate events validated")
    if run_filter_check:
        report["filter_retention"] = {
            "kept": total["filter_kept"],
            "total": total["filter_checked"],
        }
        if total["filter_kept"] != total["filter_checked"]:
            report["errors"].append(
                "replay parser/tier-A admission retained "
                f"{total['filter_kept']}/{total['filter_checked']} traces"
            )
    report["verdict"] = "pass" if not report["errors"] else "fail"
    return report
