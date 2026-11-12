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
/opt/trie-venv/bin/python benchmarks/replay/run_trie.py stack=<serve work dir> \
  workload_path=<workload.jsonl> concurrency=<n> duration=<s> output_dir=<run>/bench
/opt/trie-venv/bin/python benchmarks/replay/score.py <run>/bench --skip-s <warmup> --window-s <s>
```

These five arguments are the interface. `stack=` is the work directory of `evolving-infra
serve` on the host that runs the router: its `stack.json` gives the endpoint, the model, the
tokenizer, the block size and the GPU count. Every other setting comes from the `paper`
preset:

| Setting | Paper | What it decides |
|---|---|---|
| `strict_replay`, `context_source` | `true`, `recorded` | prompts and output lengths are the recorded ones, checked token by token |
| `replay_output_mode` | `model` | the model generates its outputs; they are not forced |
| `replay_context_mode` | `actual_output` | later prompts carry the model's own earlier outputs, which speculative decoding predicts better than recorded ones |
| `warm_start`, `seed` | `stationary`, `0` | the first agents start inside their trace, at points drawn from the seed |
| `workload_order`, `delay_scale` | `natural`, `1` | tasks in file order; tool and user waits at recorded length |
| `trajectory_prefix_tokens` | `10` | a per-agent prefix, so agents share a cache only where the trace says so |
| `cache_salt_mode` | `global` | the prepared path carries one cache salt |
| `max_retries`, `migration_limit` | `0`, `0` | a failed request counts as failed |
| `stream`, `strict_replay_openai_fast_path`, `timeout`, `drain_timeout` | `true`, `true`, `3600`, `0` | transport |

The preset also sets the client's environment as in the paper's runs unless it is already
set: `TRIE_BENCHMARK_CANCEL_TIMEOUT_S=180` and `TRIE_SKIP_ROW_VALIDATION=1` (the published
workloads were validated when they were built).

An argument given explicitly overrides the preset. `run_trie.py` writes the resolved settings
and every difference from the preset to `<output dir>/replay-settings-<i>.json`, and
`score.py` reports them as `replay_differs_from_preset`: an empty object means the paper's
replay. For a workload with several client processes, start each with `processes=N
process=I` and its shard as `workload_path`; each gets `concurrency / N`. `preset=none` turns
the defaults off, and every argument is then given as to the `trie` command, with the
connection settings in `EVOLVING_INFRA_REPLAY`:

- `EVOLVING_INFRA_REPLAY` holds `prepared_replay.Config`. `config_id` and `model_path` must match the
  frontend's config and the router's prepared contract.
- Several Trie processes: give one frontend URL per process in `tokenizer_urls`; process `i`
  writes `trie-results-<i>.json` and uses the `i`-th URL.
- `replay_context_mode=actual_output` puts the model's own earlier outputs, not the recorded ones,
  into later prompts, as in the paper's runs. With recorded outputs in the context, speculative
  decoding accepts fewer draft tokens and decode is slower.
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
   live conformance probe against the router's `/generate` (or each worker directly, as the Modal
   harness does):
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
   the replay, then `python checks.py metrics --dir <run>/metrics --workers N`. A role switch does
   not restart an engine, so a reset means a worker restarted; `--resets-allowed` reports resets
   instead of failing, for harnesses that restart workers.

The checks read the engine's `sglang:num_requests_total`, `sglang:num_running_reqs` and
`sglang:*queue_reqs` metrics.
