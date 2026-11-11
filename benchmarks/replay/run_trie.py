#!/usr/bin/env python3
"""Run the unchanged Trie CLI through the router's prepared path, with agent identities.

Usage: python run_trie.py stack=<serve work dir> workload_path=<file> concurrency=<n> duration=<s> output_dir=<dir>

Every other setting comes from the `paper` preset (PAPER below): leaving an argument out runs the
paper's replay, and an argument given explicitly overrides it. The resolved settings and every
difference from the preset are written to <output dir>/replay-settings-<i>.json, and score.py
reports them. Arguments of this script (all others are passed to the `trie` command):
  stack=DIR      the work directory of `evolving-infra serve` on the host that runs the router;
                 its stack.json gives the endpoint, the model, the tokenizer, the block size and
                 the GPU count;
  output_dir=DIR results, lifecycle and settings files of this process go here;
  processes=N process=I   this is process I of N: it gets concurrency / N and writes
                 trie-results-<I>.json; give each process its shard as workload_path;
  preset=none    no defaults: every argument is given as to the `trie` command.

Install order (each wraps what is installed before it; prepared_replay does not chain, so it goes first):
  1. prepared_replay: Trie -> frontend /prepare -> router /symmetric_pd/prepared -> frontend /forward_pd
     (one fixed cache_salt = None, no connect retries; headers limited to x-agent-context /
     x-request-id / x-smg-routing-key);
  2. replay_agent_identity: x-agent-context {agent_id, subagent_id, request_id} per model request,
     request_id monotonic per branch;
  3. session_replay: the identity reaches the router; with retention enabled, END
     (POST <router>/kv_sessions/end) when a branch ends with one of `end_outcomes`;
  4. a per-conversation X-SMG-Routing-Key on every request.

EVOLVING_INFRA_REPLAY (environment, JSON) replaces the settings derived from stack=. It holds
prepared_replay.Config fields plus:
  tokenizer_urls     one frontend URL per Trie process; process i is the i in
                     output_json=.../trie-results-<i>.json (a single URL may be given as tokenizer_url);
  retention_enabled  send session END (default false; the validated DSV4 runs ran with KV session
                     retention off, and a refused END would count as a client failure);
  end_outcomes       branch outcomes that send END when retention is enabled
                     (default ["complete", "cancelled", "skipped_or_incomplete", "failed"]).
Trie must run with cache_salt_mode=global: the prepared path carries one fixed cache salt.
The session lifecycle (END records) is written beside output_json as session-lifecycle[-<i>].jsonl.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_END_OUTCOMES = ("complete", "cancelled", "skipped_or_incomplete", "failed")

# The replay of the paper's runs. Each value decides what is measured:
PAPER = {
    "strict_replay": "true",                    # recorded prompts and output lengths, checked token by token
    "context_source": "recorded",
    "replay_output_mode": "model",              # the model generates; outputs are not forced
    "replay_context_mode": "actual_output",     # later prompts carry the model's own earlier outputs
    "warm_start": "stationary",                 # the first agents start inside their trace
    "seed": "0",
    "workload_order": "natural",
    "delay_scale": "1",                         # tool and user waits at recorded length
    "trajectory_prefix_tokens": "10",           # a per-agent prefix, so agents do not share a cache by accident
    "cache_salt_mode": "global",
    "stream": "true",
    "strict_replay_openai_fast_path": "true",
    "max_retries": "0",                         # a failed request counts as failed
    "migration_limit": "0",
    "timeout": "3600",
    "drain_timeout": "0",
    "api_key": "EMPTY",
}
# Environment of the replay client in the paper's runs (set unless already given).
PAPER_ENV = {
    "TRIE_BENCHMARK_CANCEL_TIMEOUT_S": "180",   # time to cancel in-flight requests when the replay ends
    "TRIE_SKIP_ROW_VALIDATION": "1",            # the published workloads were validated when they were built
}
OWN = ("preset", "stack", "output_dir", "processes", "process")


def _same(a: str, b: str) -> bool:
    try:
        return float(a) == float(b)
    except ValueError:
        return a.lower() == b.lower()


def resolve(argv: list, environ: dict) -> tuple:
    """(trie arguments, EVOLVING_INFRA_REPLAY settings, record of what was resolved)."""
    given = dict(a.split("=", 1) for a in argv)
    preset = given.get("preset", "paper")
    if preset not in ("paper", "none"):
        raise SystemExit("preset is paper or none")
    args = {k: v for k, v in given.items() if k not in OWN}
    stack = None
    if "stack" in given:
        path = Path(given["stack"])
        stack = json.loads((path / "stack.json" if path.is_dir() else path).read_text())
        for key, value in (("endpoint", stack["router_url"] + "/v1"), ("model", stack["model"]),
                           ("tokenizer_model", stack["model_path"]), ("prefix_block_size", stack["page_size"]),
                           ("num_gpus", stack["gpus"])):
            args.setdefault(key, str(value))
    if preset == "paper":
        for key, value in PAPER.items():
            args.setdefault(key, value)
    process, processes = int(given.get("process", 0)), int(given.get("processes", 1))
    if not 0 <= process < processes:
        raise SystemExit("process is 0 .. processes-1")
    if processes > 1:
        total = int(args["concurrency"])
        if total % processes:
            raise SystemExit(f"concurrency {total} is not a multiple of {processes} processes")
        args["concurrency"] = str(total // processes)
    if "output_dir" in given:
        out = Path(given["output_dir"])
        args.setdefault("output_json", str(out / f"trie-results-{process}.json"))
        args.setdefault("lifecycle_output_jsonl", str(out / f"trie-lifecycle-{process}.jsonl"))
        args.setdefault("lifecycle_run_id", environ.get("SLURM_JOB_ID") or out.resolve().parent.name)
        args.setdefault("lifecycle_cell_id", "bench")
        args.setdefault("lifecycle_arm", "on")
        args.setdefault("lifecycle_concurrency", args.get("concurrency", "0"))
    if environ.get("EVOLVING_INFRA_REPLAY"):
        settings = json.loads(environ["EVOLVING_INFRA_REPLAY"])
    elif stack is not None:
        settings = {"router_url": stack["router_url"], "tokenizer_urls": [stack["tokenizer_url"]],
                    "config_id": stack["config_id"], "model_name": args["model"],
                    "model_path": args["tokenizer_model"], "incremental_streaming_output": True,
                    "retention_enabled": bool(stack["retention"])}
    else:
        raise SystemExit("give stack=<serve work dir>, or EVOLVING_INFRA_REPLAY in the environment")
    differs = {k: {"value": args.get(k), "preset": v} for k, v in PAPER.items()
               if k not in args or not _same(args[k], v)} if preset == "paper" else None
    record = {"preset": preset, "differs_from_preset": differs, "process": process, "processes": processes,
              "arguments": args, "replay": settings}
    return [f"{k}={v}" for k, v in args.items()], settings, record


def argument(name: str, default: str | None = None) -> str | None:
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(name + "=")), default)


def install_routing_key(client_type) -> None:
    """Give every conversation (trace, replay cycle) its own sticky routing key."""
    nonce = uuid.uuid4().hex
    original = client_type._run_replay_trace

    @functools.wraps(original)
    async def trace_with_key(self, trace, **kwargs):
        identity = [nonce, kwargs.get("identity_trace_id") or trace.trace_id, kwargs.get("replay_cycle_id", 0)]
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        trace = dataclasses.replace(trace, metadata={**trace.metadata, "routing_key": key})
        return await original(self, trace, **kwargs)

    client_type._run_replay_trace = trace_with_key


def main() -> int:
    arguments, settings, record = resolve(sys.argv[1:], os.environ)
    if record["preset"] == "paper":
        for key, value in PAPER_ENV.items():
            os.environ.setdefault(key, value)
        record["environment"] = {key: os.environ[key] for key in PAPER_ENV}
    sys.argv[1:] = arguments
    output = Path(argument("output_json") or "trie-results.json")
    if argument("cache_salt_mode", "session") != "global":
        raise SystemExit("the prepared path carries one fixed cache salt; pass cache_salt_mode=global")
    urls = settings.pop("tokenizer_urls", None) or [settings.pop("tokenizer_url")]
    match = re.fullmatch(r"trie-results-(\d+)\.json", output.name)
    settings["tokenizer_url"] = urls[(int(match.group(1)) if match else 0) % len(urls)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.name.replace("trie-results", "replay-settings")).write_text(json.dumps(record, indent=1))
    retention = bool(settings.pop("retention_enabled", False))
    outcomes = tuple(settings.pop("end_outcomes", DEFAULT_END_OUTCOMES)) if retention else ()

    import prepared_replay
    import replay_agent_identity
    import session_replay
    import trie.client
    import trie.client_lifecycle

    client, writer = trie.client.Client, trie.client_lifecycle.ClientLifecycleWriter
    config = prepared_replay.Config(**settings)
    prepared_replay.install(client, config)
    if record["processes"] > 1:
        replay_agent_identity.PROCESS = f"process-{record['process']}:"
    replay_agent_identity.install(client, writer)
    name = output.name.replace("trie-results", "session-lifecycle")
    log = output.with_name(name).with_suffix(".jsonl").open("x", buffering=1)

    def record(value):
        log.write(json.dumps(value, separators=(",", ":")) + "\n")

    ends = session_replay.install(client, writer, config, retention_enabled=retention, record=record,
                                  end_outcomes=outcomes)
    record(dict(event="RETENTION_REPLAY_INSTALLED", tokenizer_url=config.tokenizer_url,
                retention_enabled=retention, end_outcomes=list(outcomes), wall_ns=time.time_ns()))
    install_routing_key(client)

    from trie.cli import main as trie_main
    status = trie_main()
    if ends.cleanup_error is not None:
        print(f"retention END cleanup failed: {ends.cleanup_error!r}", file=sys.stderr)
        return status or 1
    return status


if __name__ == "__main__":
    sys.exit(main())
