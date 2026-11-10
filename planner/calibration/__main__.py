"""CLI used by an experiment controller; writes data, never controls engines."""

import argparse
import hashlib
import json
from pathlib import Path


def add_arguments(parser):
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)


def run(args):
    from .model import freeze
    from .trace import calibrate, json_lines

    manifest = json.loads(args.manifest.read_text())
    config = manifest["configuration"]
    requests = []
    for path in manifest["prefill_requests"]:
        before = len(requests)
        for row in json_lines(path):
            if not manifest["start_unix"] <= row["ts"] <= manifest["end_unix"]:
                continue
            if (row.get("finish_reason") or {}).get("type") != "length":
                continue
            output = row.get("requested_output_len")
            if type(output) is not int or output <= 0:
                raise ValueError(
                    "Need positive original requested output length, not P-only output_len"
                )
            n = row["input_len"] - row["cached_tokens"]
            if n < 0:
                raise ValueError("Invalid cache token count")
            requests.append(dict(uncached=n, hit=row["cached_tokens"], output=output,
                                 host_hit=row.get("cached_tokens_host"),
                                 device_hit=row.get("cached_tokens_device")))
        if config.get('require_all_worker_traces') and len(requests) == before:
            raise ValueError(f'No completed P requests in learning window: {path}')
    report = calibrate(
        prefill_paths=manifest["prefill_steps"],
        decode_paths=manifest["decode_steps"],
        requests=requests,
        start_unix=manifest["start_unix"],
        end_unix=manifest["end_unix"],
        configuration=config,
        kv_capacity_tokens=manifest["kv_capacity_tokens"],
        prefill_request_paths=manifest["prefill_requests"],
        telemetry_path=manifest.get("telemetry_path"),
    )
    report["calibration_source_sha256"] = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ["__main__.py", "model.py", "trace.py", "capacity.py", "restore.py", "restore_audit.py"]
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.freeze:
        freeze(args.freeze, report, config)
    print(
        json.dumps({"decision": report["decision"], "error": report["decision_error"]})
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
