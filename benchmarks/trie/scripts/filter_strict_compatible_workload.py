"""Filter a workload down to traces whose text reconstructs under strict replay.

Strict replay of text-only traces rebuilds each assistant turn as the exact
string continuation of its rendered prompt. Chat templates that rewrite
earlier turns when a new message is appended (for example reasoning-tag
stripping) break the prefix property for some traces; the client fails those
traces closed at preparation. This tool applies the same prefix and
token-boundary checks up front and writes the passing subset, so every arm of
an A/B replays one identical, hash-pinned workload with zero admission
failures.

Usage:
    python filter_strict_compatible_workload.py \
        --workload decode-heavy-100.workload.jsonl \
        --tokenizer-model /shared/model/MiniMax-M2.7 \
        --output decode-heavy-100-strict.workload.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json

from trie.tokenizer_manager import TokenizerManager


def trace_is_strict_compatible(row: dict, tokenizer: TokenizerManager) -> tuple[bool, str]:
    """Mirror the client's text-reconstruction admission checks for one trace."""
    messages = copy.deepcopy(row.get("initial_messages") or [])
    tools = row.get("tools")
    for event in row.get("events", []):
        event_type = event.get("type")
        if event_type in ("wait", "append_message"):
            if event_type == "append_message" and event.get("message"):
                messages.append(copy.deepcopy(event["message"]))
            continue
        if event_type != "generate":
            continue
        recorded = event.get("recorded_assistant")
        if event.get("recorded_output_token_ids") is not None:
            if recorded is not None:
                messages.append(copy.deepcopy(recorded))
            continue
        if recorded is None:
            return False, "generate event without recorded_assistant"
        prompt = tokenizer.render_chat(
            messages, tools=tools, add_generation_prompt=True
        )
        completed = tokenizer.render_chat(
            messages + [copy.deepcopy(recorded)],
            tools=tools,
            add_generation_prompt=False,
        )
        if not completed.startswith(prompt):
            return False, "assistant is not a string continuation of its prompt"
        continuation = completed[len(prompt) :]
        payload_ids = tokenizer.encode(continuation)
        prompt_tail = prompt[-1024:]
        boundary_ids = tokenizer.encode(prompt_tail)
        combined_ids = tokenizer.encode(prompt_tail + continuation)
        if (
            combined_ids[: len(boundary_ids)] != boundary_ids
            or combined_ids[len(boundary_ids) :] != payload_ids
        ):
            return False, "tokenization crosses the prompt boundary"
        messages.append(copy.deepcopy(recorded))
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = TokenizerManager(args.tokenizer_model)
    kept = 0
    dropped: list[tuple[str, str]] = []
    with open(args.workload) as source, open(args.output, "w") as sink:
        for line in source:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ok, reason = trace_is_strict_compatible(row, tokenizer)
            if ok:
                sink.write(line + "\n")
                kept += 1
            else:
                dropped.append((row.get("trace_id", "?"), reason))

    print(f"kept {kept} traces -> {args.output}")
    print(f"dropped {len(dropped)}:")
    for trace_id, reason in dropped:
        print(f"  {trace_id}: {reason}")
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
