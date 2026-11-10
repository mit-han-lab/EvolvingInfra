#!/usr/bin/env python3
"""Check a workload against its offline validation report.

`convert_traces_to_strict.py validate --report <workload>.validate.json` records the
workload and tokenizer.json sha256 it checked. A passing report for these exact bytes
lets the replay client skip its own row-by-row schema pass, which on the multi-GB
inputs outlasts the warmup watchdog.

Run it before starting GPUs and again on the bytes the replay actually reads, before
conformance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def default_report(workload: pathlib.Path | str) -> pathlib.Path:
    return pathlib.Path(f"{workload}.validate.json")


def check_report(report: dict, workload_sha256: str, tokenizer_sha256: str) -> None:
    """Raise ValueError unless ``report`` passed and covers these exact bytes."""
    if not isinstance(report, dict) or report.get("verdict") != "pass":
        raise ValueError("validation verdict is not pass")
    checked = report.get("attests") or {}
    for key, actual in (("workload_sha256", workload_sha256), ("tokenizer_sha256", tokenizer_sha256)):
        if checked.get(key) != actual:
            raise ValueError(f"report {key}={checked.get(key)} but the input has {actual}; re-run validate")


def check_files(workload: pathlib.Path, tokenizer_dir: pathlib.Path, report: pathlib.Path | None = None) -> str:
    report = report or default_report(workload)
    workload_sha256 = sha256_file(workload)
    tokenizer_sha256 = sha256_file(pathlib.Path(tokenizer_dir) / "tokenizer.json")
    check_report(json.loads(pathlib.Path(report).read_text()), workload_sha256, tokenizer_sha256)
    return f"workload_sha256={workload_sha256} tokenizer_sha256={tokenizer_sha256}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workload", type=pathlib.Path, required=True, help="workload jsonl")
    parser.add_argument("--tokenizer-dir", type=pathlib.Path, required=True, help="tokenizer directory")
    parser.add_argument("--report", type=pathlib.Path, help="validation report (default <workload>.validate.json)")
    args = parser.parse_args()
    try:
        print(check_files(args.workload, args.tokenizer_dir, args.report))
    except Exception as exc:  # noqa: BLE001 - a mismatch or a missing file: the replay cannot use it
        print(f"input check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
