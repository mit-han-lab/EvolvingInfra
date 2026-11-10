# Codex trace replay

Strict replay of the cleaned Codex agent sessions with the same machinery as the
[Claude Code trace replay](cc-trace-replay.md). The Codex requests are first
rewritten into the cc-weka session format, sub-agents included. The `cc_weka`
adapter then turns each session into one [multi-agent task](multiagent-replay.md),
with the same spawn and join placement, completion edges, cache salt and
routing key as a CC session. The source carries no text, so tokens are
synthetic.

## Inputs

- **Source:** a cleaned Codex request trace (`requests.jsonl.gz`), one row per model request.
  - Upstream: TraceLab (https://github.com/uw-syfi/TraceLab, CC BY 4.0).
  - Tool calls longer than 2 min were clipped to 2 min in the cleaning.
- **Fields used per request:**
  - `session_id`, `round_index`, `user`, `model`;
  - `input_tokens_total`;
  - `prefix_tokens`: the provider's `cached_input_tokens`, always a multiple of 128;
  - `output_tokens`, which includes reasoning tokens;
  - `timing_events`;
  - `tools`: only `spawn_agent` calls are used.

The converted trace and workloads are not distributed with this repository.

## Conversion

### Step 1: Codex requests to cc-weka sessions, sub-agents included

The script uses the standard library only.

```bash
python3 scripts/codex_to_ccweka.py requests.jsonl.gz codex.ccweka.jsonl.gz --report codex.ccweka.report.json
```

**Sub-agents (on by default).**
- Codex starts a sub-agent with the `spawn_agent` tool and records it as a session of its own.
- The released trace does not export TraceLab's `parent_thread_id`, so each spawn is linked to a session by this rule:
  - the earliest not-yet-linked session of the same user;
  - whose first request starts within [spawn emitted − 1 s, spawn result + 60 s];
  - spawns are processed in time order, and a link that would make a cycle is skipped.
- A linked session becomes a `subagent` entry of its parent's root session. A sub-agent spawned by a sub-agent names
  that sub-agent in `parent_agent_id`.
- On the full trace [M]:
  - 664 `spawn_agent` calls: 319 link to one candidate, 233 to the first of several (mostly parallel spawns of
    siblings), and 112 find none;
  - linked sub-agents start p50 2.9 s after their spawn;
  - a sub-agent's first prompt is about 14K tokens against its parent's 121K, so it starts from a fresh context, not a
    fork.

**Agents.**
- Every agent (the root and each sub-agent) is one sequential chain, and its row is marked `"sequential_agents": true`.
- A request that starts before the agent's previous request ends is started at that end (`overlap_serialized` in the report).

**Prefix.**
- Within an agent, a request shares `floor(prefix_tokens / 64)` blocks with the agent's previous request. Every other
  block is new, and agents share no blocks with each other.
- Codex prompts are not append-only at the token level: reasoning is dropped from the next prompt, and contexts are
  compacted or rewritten.
  - Previous input minus `prefix_tokens` is 271 tokens at p50, and at least 1,024 tokens in 13 % of request pairs.
  - The provider's reported cache hit therefore serves as the shared prefix, a lower bound of the true common prefix.

**Sizes.** `in` is `input_tokens_total` rounded to a multiple of 64; `out` is `output_tokens`.

**Timing.**
- A request starts at its latest `tool_result` / `user_message` event and ends at its `usage_report`.
- Per session (root plus sub-agents), every stretch in which no request of the session is in flight is clipped to
  120 s, and the clipped time is removed from the session's timeline.
- A parent waiting for a running sub-agent is not clipped. The replay makes the parent wait for the sub-agent's actual
  completion, instead of the cleaned trace's `wait_agent` latency, which the 2-min tool clip cut at 120 s.
- `think_time` is measured from the later of the agent's previous end and the end of its last finished sub-agent, as in
  cc-weka, and is capped at 120 s per request, as in the cleaned CC trace. (Without the cap, an agent that idles while
  another agent of its session is busy would keep its full idle time; 207 requests, up to 3,840 s.)

**Hash ids are session-local**, so the system prompt shared across sessions is not shared in the replay.

### Step 2: cc-weka sessions to a Trie workload, per target tokenizer

This step is the same as for CC.

```bash
gunzip -k codex.ccweka.jsonl.gz
uv run python scripts/convert_traces_to_strict.py cc_weka --input codex.ccweka.jsonl \
  --target-tokenizer <tokenizer dir> --output codex.workload.jsonl --workers 8
uv run python scripts/convert_traces_to_strict.py validate --workload codex.workload.jsonl \
  --tokenizer <tokenizer dir> --source codex.ccweka.jsonl --workers 8 --report codex.workload.jsonl.validate.json
```

The only difference from CC is how the agents are found; everything after that is shared code in
`scripts/strict_convert/adapters/cc_weka.py`.

- **No LCP chain detection.** A `sequential_agents` row keeps every agent as one chain: `main` for the root, `sa:<id>`
  for each sub-agent. CC's LCP chain detection would fork a new agent at every prompt that drops part of the previous
  one, and it takes about an hour on the longest session (21,351 requests).
- **Parent.** A sub-agent's parent is its `parent_agent_id`, defaulting to main.
- **Shared with CC:**
  - spawn and join placement (blocking, background, or at the parent's end) and the completion edges;
  - one cache salt per session (`cache_salt_scope: "task"`) and one routing key per session (the session id), so
    sticky routing keeps a session and its sub-agents on one worker;
  - the validator.
- Rows marked with the older `"single_agent": true` are read the same way.

## Replay

Use the canonical replay protocol: 30 min warmup and 60 min scored in one Trie
process, stationary warm start, constant concurrency. `concurrency` counts
sessions; a session and all its sub-agents share one slot.

```bash
uv run trie workload_path=codex.workload.jsonl endpoint=http://<router>/v1 model=<served name> \
  strict_replay=true max_retries=0 stream=true strict_replay_openai_fast_path=true \
  concurrency=<sessions> duration=5400 delay_scale=1 cache_salt_mode=session seed=0 \
  lazy_workload=true warm_start=stationary timeout=3600 prefix_block_size=256 \
  output_json=<run>/results.json lifecycle_output_jsonl=<run>/lifecycle.jsonl \
  lifecycle_run_id=<run> lifecycle_cell_id=<cell> lifecycle_arm=<arm> lifecycle_concurrency=<sessions>
uv run python scripts/lifecycle_window.py <run>/lifecycle.jsonl --skip-seconds 1800 --window-seconds 3600   # scored window [1800, 5400)
```

The stationary warm start re-sends, cold, the last prompt of every agent still running at the cut. For Codex that is
69K tokens per slot on average (p90 138K), against 219K for the CC 062126 trace.

## Workload shape

Per recorded session:

| measure | p50 | p90 | p99 | max |
|---|---|---|---|---|
| turns | 24 | 241 | 1,664 | 21,351 |
| first input (tokens) | 14.7 K | 26.6 K | 199 K | 463 K |
| peak context (tokens) | 84 K | 235 K | 252 K | 463 K |
| output per request (tokens) | 188 | 831 | 3,068 | 34.7 K |
| gap before a request (s, before clipping) | 5.0 | 37 | 928 | — |

- 96.2 % of input tokens are reported cached.
- 3.3 % of request pairs shrink the prompt (compaction).
- 4.7 % of gaps exceed 120 s.
- **Sub-agent families** [M] (the family statistics come from an earlier pass of the same link rule):
  - 28 sessions hold the 552 linked sub-agents, and these families hold about 25 % of all requests;
  - sub-agents per family: p50 3, max 213;
  - peak concurrently live agents per family: p50 3, max 11;
  - peak in-flight requests per family: p50 2, max 7;
  - for 5.2 % of family busy time, two or more requests are in flight.
- Compared with the CC trace, prompts are shorter and gaps are short, so a Codex session keeps more requests in flight.
- An engine `--context-length` of 262,144 covers the peak context of about 99 % of sessions; 0.36 % of requests are
  longer. Report the number of rejected requests.

## Limits

- The same limits as the CC replay apply: outputs are synthetic and never appear in the next prompt, and timing is
  reconstructed from event times.
- **Sub-agent links are inferred** from user and time, not recorded:
  - 112 spawns found no session;
  - with parallel spawns, siblings may be swapped, which only changes which parallel child is which.
- **The prefix is a lower bound.** A hit the provider missed (for example, a cache eviction at the provider) becomes
  new blocks in the replay.
- **One target model.** The trace mixes 17 OpenAI models (mostly gpt-5.x and codex variants), and all are served by
  the one target model.
