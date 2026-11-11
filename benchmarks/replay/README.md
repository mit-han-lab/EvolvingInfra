# Replay through the router, and the checks that judge a run

Trie (`benchmarks/trie`) replays a workload against an OpenAI-compatible endpoint. This
directory adds what a run through evolving-infra needs on top of the unchanged Trie:

| File | Role |
|---|---|
| `run_trie.py` | Entry point. Installs the adapters below on Trie's client, then runs the Trie CLI with the same `key=value` arguments. |
| `prepared_replay.py` | Token-authoritative prepared path: Trie → frontend `/prepare` → router `/symmetric_pd/prepared` (picks P and D) → frontend `/forward_pd`. Trie's workload, tool waits, token checks and latency accounting stay in use. |
| `replay_agent_identity.py` | `x-agent-context` = {agent, sub-agent, request id} on every model request; request ids are monotonic per branch, including across Trie request migrations. |
| `session_replay.py` | Passes the identity to the router; when session retention is on, sends session END when a branch ends. |
| `check_inputs.py` | Input check: the workload and `tokenizer.json` match the sha256 its offline validation report attests. |
| `capture_conformance.py` | Runs Trie's `scripts/sglang_strict_replay_conformance.py` unchanged and keeps every native reply, also on failure. |
| `checks.py` | Quiet window, steady-state zero failures, monotonic request counters, and steady-window totals. |
| `score.py` | Score of the scored window over every Trie process's lifecycle file: requests per second, prefix hit, TTFT and end-to-end latency. |

All scripts run with the Trie venv's Python (`/opt/trie-venv/bin/python` in the image), which
also has the router client SDK (`evolving_infra.router.client`) installed.

## Running a replay

Workloads: download them from
[crmsndu/evolving-infra-traces](https://huggingface.co/datasets/crmsndu/evolving-infra-traces).

```bash
export EVOLVING_INFRA_REPLAY='{"router_url": "http://127.0.0.1:30900",
  "tokenizer_urls": ["http://127.0.0.1:28100"], "config_id": "dsv4",
  "model_name": "deepseek-v4-flash", "model_path": "/models/DeepSeek-V4-Flash",
  "incremental_streaming_output": true}'
/opt/trie-venv/bin/python benchmarks/replay/run_trie.py \
  workload_path=<workload.jsonl> endpoint=http://127.0.0.1:30900/v1 model=deepseek-v4-flash \
  api_key=EMPTY strict_replay=true strict_replay_openai_fast_path=true stream=true max_retries=0 \
  cache_salt_mode=global prefix_block_size=256 concurrency=<n> duration=<s> \
  output_json=<run>/trie-results-0.json lifecycle_output_jsonl=<run>/trie-lifecycle-0.jsonl \
  lifecycle_run_id=<run> lifecycle_cell_id=bench lifecycle_arm=off lifecycle_concurrency=<n>
```

- `EVOLVING_INFRA_REPLAY` holds `prepared_replay.Config`. `config_id` and `model_path` must match the
  frontend's config and the router's prepared contract.
- Several Trie processes: give one frontend URL per process in `tokenizer_urls`; process `i`
  writes `trie-results-<i>.json` and uses the `i`-th URL.
- `cache_salt_mode=global` is required: the prepared path carries one fixed cache salt.
- `lifecycle_output_jsonl` and the four `lifecycle_*` arguments are required: the agent-identity
  adapter records each request through Trie's lifecycle writer.
- Session END is off by default (`"retention_enabled": false`). With DSV4, KV session retention
  is off in the engines and router, which refuse END; a refused END counts as a
  client failure. Turn it on only with retention enabled on both sides.
- Session lifecycle records go to `session-lifecycle[-<i>].jsonl` beside the results.

## Judging a run

A run counts only if all four checks pass:

1. **Input conformance.** Before starting GPUs and again on the files the replay reads:
   `python check_inputs.py --workload W --tokenizer-dir T` (report default `W.validate.json`,
   written by `benchmarks/trie/scripts/convert_traces_to_strict.py validate --report`). Then the
   live conformance probe against the router:
   `python capture_conformance.py --script benchmarks/trie/scripts/sglang_strict_replay_conformance.py --event-log <run>/conformance-events.jsonl -- --endpoint http://127.0.0.1:30900/v1 --model deepseek-v4-flash --output <run>/conformance.json`.
2. **Quiet window** before load: `python checks.py quiet --worker http://<w0> --worker http://<w1>`
   waits until every worker's queues are empty and its request counter has not moved for 60 s
   (a counter reset fails the check).
3. **Steady-state zero failures.** `python checks.py result <run>/trie-results-0.json` for a
   fixed-length run, or `python checks.py window <lifecycle.jsonl> --start-ns A --end-ns B` over
   the measured window of a continuous run. Both need completions and zero failed requests;
   requests cancelled at the end of the window are truncations, not failures. `window` also
   prints the token-weighted prefix hit rate (`cached_prompt_tokens / completed_prompt_tokens`).
4. **Monotonic counters.** Run `python checks.py scrape --worker ... --out <run>/metrics` during
   the replay, then `python checks.py metrics --dir <run>/metrics --workers N`. With a planner
   that restarts workers on role switches, pass `--resets-allowed`: resets are reported, not
   fatal.

The checks read the engine's `sglang:num_requests_total`, `sglang:num_running_reqs` and
`sglang:*queue_reqs` metrics.
