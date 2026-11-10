"""Convert and audit clean-cap120 CC into a fixed-cut schema-v3 task pool.

No requests/GPU calls. The complete source corpus is examined, including every
ineligible root. Output is atomically published only after all rows validate.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strict_convert.adapters import cc_weka
from strict_convert.adapters.cc_fixed_start import cut_row, eligible, timing_row
from strict_convert.adapters.cc_timing import simulate
from trie.types import ReplayTask


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", required=True, help="Directory of cleaned Weka JSON roots"
    )
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cut-seconds", type=float, default=3600)
    a = p.parse_args()
    output = Path(a.output)
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise ValueError("Refusing to overwrite converted data or a partial attempt")
    pool = cc_weka.allowed_token_ids(a.tokenizer).tolist()
    identity = {
        "path": a.tokenizer,
        "tokenizer_json_sha256": digest(Path(a.tokenizer) / "tokenizer.json"),
    }
    files = sorted(Path(a.input).glob("*.json"))
    if not files:
        raise ValueError("No source roots")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    roots, ids = [], set()
    with partial.open("w") as sink:
        for path in files:
            session = cc_weka.build_session(json.loads(path.read_text()))
            if session.session_id in ids:
                raise ValueError("Duplicate source task")
            ids.add(session.session_id)
            candidate = eligible(session, a.cut_seconds)
            # Independent timing-only path checks admission before token synthesis.
            if not candidate:
                roots.append(
                    dict(
                        id=session.session_id,
                        eligible=False,
                        source_sha256=digest(path),
                    )
                )
                continue
            base = cc_weka.build_row(session, pool, cc_weka.DEFAULT_SEED, identity, {})
            if simulate(base) != simulate(timing_row(session)):
                raise ValueError(
                    "Payload conversion changed the authoritative timing graph"
                )
            result = cut_row(base, a.cut_seconds)
            if result is None:
                raise ValueError("Eligibility differs from conversion")
            ReplayTask.from_jsonl_row(result)
            # Exact retained payloads and all edges; no tokenization/re-encoding.
            original = {
                (t["trace_id"], i): e
                for t in base["traces"]
                for i, e in enumerate(t["events"])
            }
            for t in result["traces"]:
                for i, event in enumerate(t["events"]):
                    if (
                        event["type"] == "generate"
                        and event != original[t["trace_id"], i]
                    ):
                        raise ValueError("Retained payload changed")
            times = simulate(timing_row(session))
            expected = {
                k
                for k, e in original.items()
                if e["type"] == "generate" and times[*k, "start"] >= a.cut_seconds
            }
            actual = {
                (t["trace_id"], i)
                for t in result["traces"]
                for i, e in enumerate(t["events"])
                if e["type"] == "generate"
            }
            if actual != expected or result["metadata"]["cc_fixed_start"][
                "retained_requests"
            ] != len(expected):
                raise ValueError(
                    "Retained request set differs from the complete source suffix"
                )
            if base["dependencies"] != result["dependencies"]:
                raise ValueError("Dependency changed")
            sink.write(json.dumps(result, separators=(",", ":")) + "\n")
            roots.append(
                dict(
                    id=session.session_id,
                    eligible=True,
                    source_sha256=digest(path),
                    **result["metadata"]["cc_fixed_start"],
                )
            )
            print(
                f"validated {len(roots)}/{len(files)} {session.session_id}", flush=True
            )
    if not any(r["eligible"] for r in roots):
        raise ValueError("No task has a Generate starting at or after the fixed cut")
    partial.rename(output)
    report = dict(
        status="pass",
        cut_s=a.cut_seconds,
        source_roots=len(roots),
        eligible_roots=sum(r["eligible"] for r in roots),
        retained_requests=sum(r.get("retained_requests", 0) for r in roots),
        workload_sha256=digest(output),
        tokenizer=identity,
        model_config_sha256=digest(Path(a.tokenizer) / "config.json"),
        roots=roots,
        limitation="Synthetic source-time suffix; not a snapshot of target engine state. "
        "No pre-cut KV is installed. Source API durations only retain crossing control waits.",
    )
    Path(str(output) + ".validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "roots"}, indent=2))


if __name__ == "__main__":
    main()
