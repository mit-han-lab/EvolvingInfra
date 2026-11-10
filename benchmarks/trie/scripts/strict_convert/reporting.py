"""Per-corpus conversion report assembly."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from strict_convert._vendor import TRIE_ROOT


def distribution(values: list[int | float]) -> dict[str, Any] | None:
    if not values:
        return None
    ordered = sorted(values)

    def pick(fraction: float) -> int | float:
        index = min(int(fraction * (len(ordered) - 1) + 0.5), len(ordered) - 1)
        return ordered[index]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": pick(0.50),
        "p95": pick(0.95),
        "max": ordered[-1],
    }


def git_marker() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(TRIE_ROOT), *args],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "git_head": head,
        "git_dirty": bool(status) if status is not None else None,
        # Deployed conversion checkouts may be plain copies without git
        # metadata; the source hash always resolves, so a report can never
        # again claim unknown code identity.
        "code_files_sha256": _code_files_sha256(),
    }


def _code_files_sha256() -> str:
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(package_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_report(
    *,
    adapter: str,
    inputs: list[str],
    output_path: str,
    output_bytes: int,
    tokenizer_identity: dict[str, Any],
    source_tokenizer_identity: dict[str, Any] | None,
    stats: dict[str, Any],
    traces_in: int,
    generates_in: int,
    warnings: list[str],
    renderer_probes: dict[str, Any],
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chained = stats.get("chained", 0)
    fresh = stats.get("fresh", 0)
    drift = stats.get("oracle_drift", [])
    report: dict[str, Any] = {
        "adapter": adapter,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "converter": git_marker(),
        "target_tokenizer": tokenizer_identity,
        "renderer_probes": renderer_probes,
        "inputs": inputs,
        "selection": selection,
        "warnings": warnings,
        "traces_in": traces_in,
        "traces_out": stats.get("traces_out", 0),
        "generates_in": generates_in,
        "generates_out": stats.get("generates_out", 0),
        "dropped_events": stats.get("dropped_events", {}),
        "dropped_traces": stats.get("dropped_traces", {}),
        "chain": {
            "chained": chained,
            "fresh": fresh,
            "chained_pct": round(100.0 * chained / (chained + fresh), 2)
            if (chained + fresh)
            else None,
            "breaks_by_reason": stats.get("breaks_by_reason", {}),
            "prefix_fallback_events": stats.get("prefix_fallback_events", 0),
            "truncated_events": stats.get("truncated_events", 0),
            "boundary_repairs": stats.get("boundary_repairs", 0),
        },
        "tokens": {
            "prompt_total": stats.get("prompt_tokens", 0),
            "output_total": stats.get("output_tokens", 0),
        },
        "max_context_tokens_per_trace": distribution(
            stats.get("max_context_per_trace", [])
        ),
        "role_mappings": stats.get("role_mappings", {}),
        "adapter_counters": stats.get("adapter_counters", {}),
        "output": {"workload": output_path, "bytes": output_bytes},
    }
    if source_tokenizer_identity is not None:
        report["source_tokenizer"] = source_tokenizer_identity
    if drift:
        report["oracle_drift"] = {
            **(distribution(drift) or {}),
            "zero": sum(1 for d in drift if d == 0),
            "mean": round(sum(drift) / len(drift), 4),
        }
    return report


def write_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w") as sink:
        sink.write(json.dumps(report, indent=2, sort_keys=True))
        sink.write("\n")
