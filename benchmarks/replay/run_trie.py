#!/usr/bin/env python3
"""Run the unchanged Trie CLI through the router's prepared path, with agent identities.

Usage: python run_trie.py <trie key=value arguments...>   (the same arguments as the `trie` command)

Install order (each wraps what is installed before it; prepared_replay does not chain, so it goes first):
  1. prepared_replay: Trie -> frontend /prepare -> router /symmetric_pd/prepared -> frontend /forward_pd
     (one fixed cache_salt = None, no connect retries; headers limited to x-agent-context /
     x-request-id / x-smg-routing-key);
  2. replay_agent_identity: x-agent-context {agent_id, subagent_id, request_id} per model request,
     request_id monotonic per branch;
  3. session_replay: the identity reaches the router; with retention enabled, END
     (POST <router>/kv_sessions/end) when a branch ends with one of `end_outcomes`;
  4. a per-conversation X-SMG-Routing-Key on every request.

EVOLVING_INFRA_REPLAY (environment, JSON) holds prepared_replay.Config fields plus:
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
    output = Path(argument("output_json") or "trie-results.json")
    if argument("cache_salt_mode", "session") != "global":
        raise SystemExit("the prepared path carries one fixed cache salt; pass cache_salt_mode=global")
    settings = json.loads(os.environ["EVOLVING_INFRA_REPLAY"])
    urls = settings.pop("tokenizer_urls", None) or [settings.pop("tokenizer_url")]
    match = re.fullmatch(r"trie-results-(\d+)\.json", output.name)
    settings["tokenizer_url"] = urls[int(match.group(1)) if match else 0]
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
    replay_agent_identity.install(client, writer)
    name = output.name.replace("trie-results", "session-lifecycle")
    log = output.with_name(name).with_suffix(".jsonl").open("x", buffering=1)

    def record(value):
        log.write(json.dumps(value, separators=(",", ":")) + "\n")

    session_replay.install(client, writer, config, retention_enabled=retention, record=record,
                           end_outcomes=outcomes)
    record(dict(event="RETENTION_REPLAY_INSTALLED", tokenizer_url=config.tokenizer_url,
                retention_enabled=retention, end_outcomes=list(outcomes), wall_ns=time.time_ns()))
    install_routing_key(client)

    from trie.cli import main as trie_main
    return trie_main()


if __name__ == "__main__":
    sys.exit(main())
