"""Command-line interface for the strict-replay trace conversion toolkit.

Usage:
    python convert_traces_to_strict.py <adapter> \
        --input <corpus path> [--input ...] \
        --output /path/to/workload.jsonl \
        --target-tokenizer /path/to/deepseek-v4-flash \
        [--limit N] [--include NAME] [--workers N] \
        [--source-tokenizer PATH]

    python convert_traces_to_strict.py validate \
        --workload /path/to/workload.jsonl \
        --tokenizer /path/to/deepseek-v4-flash \
        [--sample N] [--skip-filter-check] [--report PATH]

Adapters: trie_text, trie_retokenize, synthetic, cc_weka.
A cc_weka workload is validated with ``validate --source <traces.jsonl>``.
Every conversion writes ``<output>`` plus ``<output>.report.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from strict_convert import adapters
from strict_convert.rendering import TargetRenderer
from strict_convert.reporting import build_report, write_report
from strict_convert.runner import run_units, select_units, unit_matches
from strict_convert.validate import validate_workload


def _add_convert_arguments(parser: argparse.ArgumentParser) -> None:
    _add_render_arguments(parser)
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        required=True,
        help="corpus file or directory; repeatable",
    )
    parser.add_argument("--output", required=True, help="output workload JSONL path")
    parser.add_argument(
        "--target-tokenizer",
        required=True,
        help="local tokenizer directory of the replay target",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="convert at most N units (traces or files, adapter-dependent)",
    )
    parser.add_argument(
        "--include",
        action="append",
        dest="includes",
        default=[],
        help="force-include units whose name/trace_id contains this string",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--source-tokenizer",
        default=None,
        help="tokenizer of the recorded token ids (trie_retokenize only)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="replicate each shape-spec row N times with distinct trace ids "
        "and content (synthetic only)",
    )
    parser.add_argument(
        "--seed-salt",
        default=None,
        help="content sampling salt (synthetic: default strict-synth-v1; cc_weka: default cc-weka-v1)",
    )


def _add_render_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--chat-encoding", choices=("auto", "template", "dsv4"), default="auto",
                        help="auto uses the serving DSV4 encoder for deepseek-v4 tokenizer directory names; template is legacy compatibility")
    parser.add_argument("--thinking-mode", choices=("thinking", "chat"), default="thinking",
                        help="DSV4 generation mode (default thinking)")
    parser.add_argument("--history-reasoning", choices=("native", "inline"), default="native",
                        help="history_refed corpora: 'native' renders history the way the target "
                             "encoding does (DSV4 does not re-feed an earlier turn's reasoning); "
                             "'inline' folds the recorded emission into the context content, "
                             "reproducing a capture whose harness did re-feed it. The two are "
                             "different workloads and both are recorded in the rendering identity")
    parser.add_argument("--template-kwargs", type=json.loads, default=None,
                        help="JSON object of extra chat-template variables, as serving's "
                             "chat_template_kwargs (e.g. '{\"clear_thinking\": false}' for a GLM "
                             "capture whose server re-fed every earlier turn's reasoning)")


def _tokenizer_identity(model_path: str) -> dict[str, object]:
    tokenizer_json = Path(model_path) / "tokenizer.json"
    sha = (
        hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()
        if tokenizer_json.is_file()
        else None
    )
    return {"path": model_path, "tokenizer_json_sha256": sha}


def _run_convert(adapter_name: str, args: argparse.Namespace) -> int:
    adapter = adapters.get(adapter_name)
    options: dict[str, object] = {
        "inputs": args.inputs,
        "output": args.output,
        "target_tokenizer": args.target_tokenizer,
        "source_tokenizer": args.source_tokenizer,
        "repeat": args.repeat,
        "seed_salt": args.seed_salt,
        "chat_encoding": args.chat_encoding,
        "thinking_mode": args.thinking_mode,
        "history_reasoning": args.history_reasoning,
        "template_kwargs": args.template_kwargs,
    }
    if adapter_name == "cc_weka":
        options.update(cc_timing=args.cc_timing, cc_zero_output=args.cc_zero_output)
        if args.cc_zero_output == "drop-advance" and args.cc_timing != "causal-gaps":
            raise ValueError("--cc-zero-output=drop-advance requires --cc-timing=causal-gaps")
    units, warnings = adapter.list_units(options)
    for include in args.includes:
        if not any(unit_matches(unit, include) for unit in units):
            warnings.append(
                f"--include {include!r} matched zero units (trace_id sniffing "
                "covers line head and tail; check the spelling)"
            )
    selected = select_units(units, args.includes, args.limit)
    if not selected:
        print("no units selected; nothing to convert")
        for warning in warnings:
            print(f"  {warning}")
        return 1
    adapter.prepare_options(options, selected)

    if Path(args.output).exists() or Path(args.output + ".report.json").exists():
        raise FileExistsError(f"refusing to overwrite existing conversion output: {args.output}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    stats, unit_warnings = run_units(adapter_name, options, selected, args.workers)
    warnings.extend(unit_warnings)

    selection = {
        "limit": args.limit,
        "includes": args.includes,
        "workers": args.workers,
        "units_listed": len(units),
        "units_selected": len(selected),
        "selected_units_sha256": hashlib.sha256(
            json.dumps([unit["name"] for unit in selected]).encode("utf-8")
        ).hexdigest(),
    }
    renderer = TargetRenderer(args.target_tokenizer, args.chat_encoding, args.thinking_mode,
                              args.history_reasoning, args.template_kwargs)
    report = build_report(
        adapter=adapter_name,
        inputs=args.inputs,
        output_path=args.output,
        output_bytes=os.path.getsize(args.output),
        tokenizer_identity=renderer.tokenizer_identity(),
        source_tokenizer_identity=_tokenizer_identity(args.source_tokenizer)
        if args.source_tokenizer
        else None,
        stats=stats,
        traces_in=stats.pop("traces_in", 0),
        generates_in=stats.pop("generates_in", 0),
        warnings=warnings,
        renderer_probes={
            "generation_tail": renderer.generation_tail,
            "forced_suffix_len": renderer.forced_suffix_len,
            "developer_role_dropped_by_template": renderer.developer_dropped,
            "nonfirst_system_dropped_by_template": renderer.nonfirst_system_dropped,
            "reasoning_field_dropped_by_template": renderer.reasoning_field_dropped,
            "tool_calls_dropped_by_template": renderer.tool_calls_dropped,
        },
        selection=selection,
    )
    report_path = args.output + ".report.json"
    write_report(report, report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"workload -> {args.output}")
    print(f"report   -> {report_path}")
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    if args.sources:
        from strict_convert.validate_cc import validate_cc_workload

        report = validate_cc_workload(args.workload, args.tokenizer, args.sources, workers=args.workers)
    else:
        report = validate_workload(
            args.workload,
            args.tokenizer,
            deep_every=args.deep_every,
            run_filter_check=not args.skip_filter_check,
            workers=args.workers,
            chat_encoding=args.chat_encoding,
            thinking_mode=args.thinking_mode,
            history_reasoning=args.history_reasoning,
            template_kwargs=args.template_kwargs,
            require_source_counts=args.require_source_counts,
        )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        with open(args.report, "w") as sink:
            sink.write(rendered + "\n")
    return 0 if report["verdict"] == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="convert_traces_to_strict",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in adapters.NAMES:
        sub = subparsers.add_parser(name, help=f"convert the {name} corpus")
        _add_convert_arguments(sub)
        if name == "cc_weka":
            sub.add_argument("--cc-timing", choices=("recorded-think", "causal-gaps"), default="recorded-think")
            sub.add_argument("--cc-zero-output", choices=("raise-to-one", "drop-advance"), default="raise-to-one")

    validate_parser = subparsers.add_parser(
        "validate", help="validate a converted workload"
    )
    validate_parser.add_argument("--workload", required=True)
    _add_render_arguments(validate_parser)
    validate_parser.add_argument("--require-source-counts", action="store_true",
                                 help="fail unless every request has recognized source prompt and output counts")
    validate_parser.add_argument("--tokenizer", required=True)
    validate_parser.add_argument(
        "--deep-every",
        type=int,
        default=25,
        help="run deep (jsonschema/bool-scan/decode) checks on every Nth row",
    )
    validate_parser.add_argument("--workers", type=int, default=1)
    validate_parser.add_argument("--skip-filter-check", action="store_true")
    validate_parser.add_argument("--report", default=None)
    validate_parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        default=[],
        help="source Weka traces of a cc_weka workload; selects its validator (repeatable)",
    )

    args = parser.parse_args()
    if args.command == "validate":
        return _run_validate(args)
    return _run_convert(args.command, args)


if __name__ == "__main__":
    raise SystemExit(main())
