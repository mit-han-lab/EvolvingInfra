"""Convert the full clean-cap120 CC trace for the existing stationary initializer."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strict_convert.adapters import cc_weka
from strict_convert.adapters.cc_stationary import normalize_clock, POLICY
from trie.types import ReplayTask


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    partial = a.output.with_suffix(a.output.suffix + ".partial")
    if a.output.exists() or partial.exists():
        raise ValueError("Refusing to overwrite conversion or partial data")
    pool = cc_weka.allowed_token_ids(str(a.tokenizer)).tolist()
    identity = dict(path=str(a.tokenizer), tokenizer_json_sha256=sha(a.tokenizer / "tokenizer.json"))
    roots, seen, maximum, total = [], set(), None, 0
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.input, "rt") as source, partial.open("w") as sink:
        for line in source:
            raw = json.loads(line)
            session = cc_weka.build_session(raw)
            if session.session_id in seen:
                raise ValueError("Duplicate source root")
            seen.add(session.session_id)
            base = cc_weka.build_row(session, pool, cc_weka.DEFAULT_SEED, identity, {})
            row = normalize_clock(session, base)
            ReplayTask.from_jsonl_row(row)
            original = {(t["trace_id"], i): e for t in base["traces"] for i, e in enumerate(t["events"])}
            assert row["dependencies"] == base["dependencies"]
            count = 0
            for trace in row["traces"]:
                for i, event in enumerate(trace["events"]):
                    before = original[trace["trace_id"], i]
                    if {k: v for k, v in event.items() if k != "source_t_s"} != {
                        k: v for k, v in before.items() if k != "source_t_s"}:
                        raise ValueError("Non-clock payload changed")
                    if event["type"] != "generate":
                        continue
                    count += 1
                    prompt = sum(n for _, n in event["recorded_prompt_block_runs"]) * 64
                    output = event["max_tokens"]
                    if prompt + output > 1048576 or output <= 0:
                        raise ValueError("Request outside approved context/output limits")
                    request = dict(root=session.session_id, request=event["source_request_id"],
                                   input=prompt, output=output, total=prompt + output)
                    if maximum is None or request["total"] > maximum["total"]:
                        maximum = request
            if count != session.source_requests:
                raise ValueError("Request count changed")
            total += count
            roots.append(dict(id=session.session_id, requests=count,
                              source_row_sha256=hashlib.sha256(line.encode()).hexdigest()))
            sink.write(json.dumps(row, separators=(",", ":")) + "\n")
            if len(roots) % 25 == 0:
                print(f"Converted {len(roots)} roots / {total} requests", flush=True)
    if not roots:
        raise ValueError("Empty source")
    partial.rename(a.output)
    report = dict(status="pass", passed=True, policy=POLICY, start_mode="stationary",
                  source_input=str(a.input), source_input_sha256=sha(a.input),
                  source_roots=len(roots), eligible_roots=len(roots), retained_requests=total,
                  workload_sha256=sha(a.output), maximum_request=maximum, tokenizer=identity,
                  model_config_sha256=sha(a.tokenizer / "config.json"), roots=roots,
                  checks=["all cleaned source roots and positive-output requests retained",
                          "payloads/waits/dependencies unchanged", "real ReplayTask parser",
                          "source_t_s follows cleaned DAG clock", "all requests within1M"],
                  limitation="Synthetic prompts and inferred agent dependencies; source API durations locate random cuts only")
    Path(str(a.output) + ".validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "roots"}, indent=2))


if __name__ == "__main__":
    main()
