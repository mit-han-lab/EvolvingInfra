"""Validation of cc_weka workloads against their source Weka sessions.

Per session: every source request appears once, no chain holds two requests
that overlap in recorded time, each spawn and join sits where the recorded
times put it, the task's agents and completion edges are exactly the chain
split's and no edge contradicts the recorded order, and every prompt is
exactly its recorded hash IDs as block IDs. The task's ``block_tokens`` must
draw 64-token blocks from the target's regular vocabulary with the session's
seed; output blocks must be fresh and hold ``max_tokens``; and, expanded on
the first replay cycle, distinct blocks must give distinct tokens, so prompt
prefixes (consecutive chain requests and the session-wide seen set) match the
hash prefixes token for token. The row must parse as a Trie replay task. The report attests the validated
workload and tokenizer bytes for the point runner's input check.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from strict_convert.adapters import cc_weka
from strict_convert.adapters._trie_common import list_jsonl_units

_STATE: dict[str, Any] = {}


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _init(tokenizer: str, sources: list[str]) -> None:
    import importlib

    from strict_convert._vendor import types_module

    types = types_module()
    units, _ = cc_weka.list_units({"inputs": sources})
    _STATE.update(pool=cc_weka.allowed_token_ids(tokenizer).tolist(),
                  sources={unit["trace_id"]: unit for unit in units},
                  replay_task=types.ReplayTask, blocks=importlib.import_module("trie.blocks"))


def _check_blocks(row: dict[str, Any], errors: list[str]) -> int:
    """Generation spec, fresh output blocks, and distinct tokens for distinct blocks."""
    trace_id = row.get("trace_id", "?")
    spec = (row.get("metadata") or {}).get("block_tokens") or {}
    seed_salt = ((row.get("metadata") or {}).get("strict_conversion") or {}).get("seed", cc_weka.DEFAULT_SEED)
    if spec.get("block_size") != cc_weka.BLOCK_TOKENS or spec.get("pool") != _STATE["pool"]:
        errors.append(f"{trace_id}: block_tokens must use 64-token blocks over the target's regular vocabulary")
        return 0
    if spec.get("seed") != cc_weka.block_seed(seed_salt, trace_id):
        errors.append(f"{trace_id}: block_tokens seed is not derived from the session")
    prompt_ids: set[int] = set()
    output_ids: list[int] = []
    output_tokens = 0
    for trace in row.get("traces", []):
        for event in trace["events"]:
            if event["type"] != "generate":
                continue
            prompt_ids.update(i for start, count in event["recorded_prompt_block_runs"] for i in range(start, start + count))
            runs = event["recorded_output_block_runs"]
            output_ids.extend(i for start, count in runs for i in range(start, start + count))
            if sum(count for _, count in runs) * cc_weka.BLOCK_TOKENS < event["max_tokens"]:
                errors.append(f"{trace_id}: {event['source_request_id']} output blocks hold fewer than max_tokens")
            output_tokens += event["max_tokens"]
    if len(set(output_ids)) != len(output_ids) or (prompt_ids and output_ids and min(output_ids) <= max(prompt_ids)):
        errors.append(f"{trace_id}: output blocks must be fresh, above every prompt block and distinct")
    # Distinct block IDs must give distinct tokens (checked on the first replay cycle).
    ids = sorted(prompt_ids) + output_ids
    tokens = np.asarray(_STATE["blocks"].BlockTokens(spec).tokens(cc_weka._runs(ids)), dtype=np.int64)
    rows = tokens.reshape(len(ids), cc_weka.BLOCK_TOKENS)
    if np.unique(rows, axis=0).shape[0] != rows.shape[0]:
        errors.append(f"{trace_id}: two distinct blocks expand to the same tokens")
    return output_tokens


def _check_row(row: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    trace_id = row.get("trace_id", "?")
    if row.get("schema_version") != 3 or row.get("cache_salt_scope") != "task":
        errors.append(f"{trace_id}: not a v3 task sharing one cache salt")
    output_tokens = _check_blocks(row, errors)
    try:
        _STATE["replay_task"].from_jsonl_row(row)
    except (TypeError, ValueError) as exc:
        errors.append(f"{trace_id}: from_jsonl_row: {exc}")
    unit = _STATE["sources"].get(trace_id)
    if unit is None:
        return {"session": trace_id, "errors": errors + [f"{trace_id}: not in the source corpus"]}
    session = cc_weka.build_session(cc_weka.load_source(unit))
    if row.get("metadata", {}).get("cc_replay_policy"):
        from strict_convert.adapters.cc_timing import check_policy
        report = check_policy(session, row)
    else:
        report = cc_weka.check_session(session, row, cc_weka.row_blocks(row))
    report["errors"] = errors + report["errors"]
    report["output_tokens"] = output_tokens
    report["prompt_tokens"] = sum(count * cc_weka.BLOCK_TOKENS for t in row["traces"]
                                  for e in t["events"] if e["type"] == "generate"
                                  for _, count in e["recorded_prompt_block_runs"])
    return report


def _check_unit(ref: tuple[str, int]) -> dict[str, Any]:
    path, offset = ref
    with open(path, "rb") as source:
        source.seek(offset)
        return _check_row(json.loads(source.readline()))


def validate_cc_workload(workload: str, tokenizer: str, sources: list[str], *, workers: int = 1) -> dict[str, Any]:
    units, _ = list_jsonl_units([workload])
    refs = [(unit["path"], unit["offset"]) for unit in units]
    if workers <= 1:
        _init(tokenizer, sources)
        sessions = [_check_unit(ref) for ref in refs]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init, initargs=(tokenizer, sources)) as pool:
            sessions = list(pool.map(_check_unit, refs))
    errors = [error for session in sessions for error in session["errors"]]
    source_units, _ = cc_weka.list_units({"inputs": sources})
    source_ids = {unit["trace_id"] for unit in source_units}
    actual_ids = [session["session"] for session in sessions]
    if len(source_ids) != len(source_units) or len(set(actual_ids)) != len(actual_ids):
        errors.append("Duplicate session IDs in source or converted corpus")
    if any(s.get("timing_policy") for s in sessions) and set(actual_ids) != source_ids:
        errors.append("Converted session IDs do not cover the full source corpus")
    for session in sessions:
        print(f"validate_cc: {session['session']}: {len(session['errors'])} errors", file=sys.stderr)
    return {
        "workload": workload,
        "sources": sources,
        "traces": len(sessions),
        "generates": sum(s.get("requests", 0) for s in sessions),
        "prompt_tokens": sum(s.get("prompt_tokens", 0) for s in sessions),
        "output_tokens": sum(s.get("output_tokens", 0) for s in sessions),
        "sessions": [{k: v for k, v in s.items() if k != "errors"} for s in sessions],
        "attests": {"workload_sha256": _sha256_file(workload),
                    "tokenizer_sha256": _sha256_file(str(Path(tokenizer) / "tokenizer.json")),
                    "model_config_sha256": _sha256_file(str(Path(tokenizer) / "config.json"))
                    if (Path(tokenizer) / "config.json").is_file() else None,
                    "source_sha256": {path: _sha256_file(path) for path in sources}},
        "errors": errors[:200],
        "error_count": len(errors),
        "verdict": "fail" if errors else "pass",
    }
