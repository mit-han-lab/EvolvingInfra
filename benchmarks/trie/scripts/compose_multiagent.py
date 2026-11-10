"""Compose converted v2 agents and an explicit dependency manifest into v3 tasks."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from trie.client import load_workloads
from trie.types import ReplayTask, ReplayTrace


def compose(traces_path: str, manifest_path: str, output_path: str) -> int:
    traces = load_workloads(traces_path)
    if any(not isinstance(trace, ReplayTrace) for trace in traces):
        raise ValueError("input must contain only schema-v2 agent traces")
    by_id = {trace.trace_id: trace for trace in traces}
    if len(by_id) != len(traces):
        raise ValueError("input agent trace IDs must be unique")
    manifest = json.loads(Path(manifest_path).read_text())
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("manifest must be a non-empty array of task objects")
    output = Path(output_path)
    if output.resolve() in (Path(traces_path).resolve(), Path(manifest_path).resolve()):
        raise ValueError("output must not overwrite an input")
    used_task_ids = set()
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            for entry in manifest:
                row = dict(entry)
                trace_ids = row.pop("trace_ids")
                if not isinstance(trace_ids, list) or not trace_ids:
                    raise ValueError("trace_ids must be a non-empty array")
                if row["trace_id"] in used_task_ids:
                    raise ValueError("manifest task IDs must be unique")
                used_task_ids.add(row["trace_id"])
                row["schema_version"] = 3
                row["traces"] = [
                    by_id[trace_id].to_jsonl_row() for trace_id in trace_ids
                ]
                task = ReplayTask.from_jsonl_row(row)
                stream.write(
                    json.dumps(task.to_jsonl_row(), separators=(",", ":")) + "\n"
                )
            stream.flush()
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    return len(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces", required=True, help="Converted schema-v2 agents JSONL"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Task IDs, trace_ids and completion dependencies JSON",
    )
    parser.add_argument("--output", required=True, help="Schema-v3 workload JSONL")
    args = parser.parse_args()
    print(f"Wrote {compose(args.traces, args.manifest, args.output)} multi-agent tasks")
