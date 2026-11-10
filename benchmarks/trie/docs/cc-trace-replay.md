# Claude Code trace replay

Strict replay of the SemiAnalysis AgentX Claude Code sessions: every request
is a token-ID prompt built from the recorded prefix hashes, sub-agents and
flattened parallel agents run as concurrent agents of one
[multi-agent task](multiagent-replay.md) whose completion edges put each
spawn and join where the recorded times do, and each session slot can start
at a cut through its recorded span. The source carries no text, so tokens are
synthetic.

## Inputs

- Source: Hugging Face dataset `semianalysisai/cc-traces-weka-062126`,
  revision `23f152f6f0f9399a85901b89a6458def0ef16729`, file `traces.jsonl`
  (393 sessions, 98,827 requests).
- Tokenizer: `tokenizer.json` of `deepseek-ai/DeepSeek-V4-Flash-0731` at
  revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` (sha256 `8f9f37ca...`).

Conversion is deterministic. With this code, the source and the tokenizer
above, and the default seed `cc-weka-v1`, the workload sha256 is
`00051ac01183b577158173b3524d5cf42dee33fda1f87ff623f8ff1bedb98f42`
(346 MB); the validate report records it under `attests.workload_sha256`.

```bash
uv sync
uv run python scripts/convert_traces_to_strict.py cc_weka --input traces.jsonl \
  --target-tokenizer <tokenizer dir> --output cc.workload.jsonl --workers 8
uv run python scripts/convert_traces_to_strict.py validate --workload cc.workload.jsonl \
  --tokenizer <tokenizer dir> --source traces.jsonl --workers 8 --report cc.workload.jsonl.validate.json
```

Conversion takes about 30 seconds and validation about 3 minutes with 8
workers. Validation checks that no agent holds two requests overlapping in
recorded time, that every spawn and join sits where the recorded times put it,
that the completion edges are exactly those and none contradicts the recorded
order, that every prompt is exactly its recorded hash IDs, that output blocks
are fresh, and that distinct blocks expand to distinct tokens, so prompt
prefixes match the hash prefixes token for token.

## Format

Each row is one session as a schema-v3 task with one agent per chain (`main`,
`fa:NNN` derived chains, `sa:<id>` sub-agents and their derived chains).

- Prompts are `recorded_prompt_block_runs` over the recorded hash IDs, and each
  output gets fresh block IDs in `recorded_output_block_runs`; the client
  expands blocks with `trie.blocks` just before sending. The task's
  `metadata.block_tokens` (64-token blocks, a seed per session, the target's
  regular vocabulary as the pool) serves every agent. Equal IDs give equal
  tokens within one replay of a task and every replay cycle gets new content;
  expanding every prompt in the file would be 21.6 B tokens.
- A spawn is an edge from the parent generate that last completed before the
  child started to the child's first event. A join is an edge from the last
  event of the child, and of every descendant joined at its end, to the first
  event of the parent turn that waited for it; background children are joined
  by the first parent turn after they ended, or only by the task's end.
- `cache_salt_scope: "task"` gives every agent of a task invocation one cache
  salt, so sub-agents keep their real prefix reuse with each other and the
  parent; per-agent salts would raise the uncached prefill of the corpus by 65%.
- Generates carry `source_t_s` and `source_api_time_s` for the warm start.

Rules for chains, parents, joins and waits are in
`scripts/strict_convert/adapters/cc_weka.py`.

## Replay

```bash
uv run trie workload_path=cc.workload.jsonl endpoint=http://<router>/v1 model=<served name> \
  strict_replay=true max_retries=0 stream=true strict_replay_openai_fast_path=true \
  concurrency=64 duration=7200 delay_scale=1 cache_salt_mode=session seed=0 \
  lazy_workload=true warm_start=stationary timeout=3600 prefix_block_size=256 \
  output_json=cc-results.json lifecycle_output_jsonl=cc-lifecycle.jsonl \
  lifecycle_run_id=<run> lifecycle_cell_id=<run> lifecycle_arm=off lifecycle_concurrency=64
```

- `concurrency` counts sessions; a session and all its agents share one slot.
- `delay_scale=1` keeps the recorded think times, uncapped; sessions mostly
  wait, so about 0.05 requests are in flight per session.
- `lazy_workload=true` parses a row only when its session is admitted.
- `warm_start=stationary` starts each slot's first session at a uniform cut:
  requests before the cut are skipped and complete their dependency edges at
  once, each agent still running re-sends its last prompt before the cut with
  one output token (lifecycle `warmup: true`), its first turn after the cut
  waits at most the recorded time left after the cut, and admissions from the
  second on start at turn 0.
- `timeout=3600`: prompts reach 990k tokens and can queue past the default 600 s.
- Fix the DSPARK acceptance with `--synthetic-oracle-dspark-accept-rate 0.6`
  on the decode server, since the synthetic outputs have no natural
  acceptance.

## Steady window

Run warmup and steady as one Trie process (`duration` = warmup + steady), so
session state and cache carry across; a second process would restart every
slot at a new cut with fresh salts and put the warmup prefills in its steady
window. Then cut the steady window from the lifecycle by admission time:

```bash
uv run python scripts/lifecycle_window.py cc-lifecycle.jsonl   # one hour of warmup, then a one-hour steady window
```

The first cl admissions each prefill a long cold prompt; size the warmup from
the prefill rate. On 1P1D DeepSeek-V4-Flash (8xH200 each, prefill HiCache
auto-sized) 64 slots warmed up at about 10k prompt tokens/s; 256 slots
without HiCache overran the prefill KV pool and ran at 3k.

## Limits

- Outputs are synthetic and never appear in the next prompt, so decode-side
  KV never hits; real sessions reuse their visible output.
- Chains, joins and blocking versus background are inferred from times and
  prefixes; there is no content to check them against.
- Per request the client materializes the prompt, hashes it for the ideal-hit
  registry and parses the echoed prompt IDs: about 0.6 s of CPU for a
  990k-token prompt.

For the opt-in GLM-5.2 preparation of **061326**, with corrected causal waits
and user-requested zero-output deletion/advancement, see
[CC 061326 / GLM-5.2](cc-061326-glm52.md). That profile uses full-start replay
and real model output, rather than the stationary-cut/forced-output example
above. Existing defaults remain unchanged.

For the Codex agent sessions (single-agent, converted to this format first),
see [Codex trace replay](codex-trace-replay.md).
