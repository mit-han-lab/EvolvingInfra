# Reproducing the paper's experiments

How the stationary-workload experiments of the [paper](../../tech_report/README.md) (Section 6.2,
Figure 6) were run, the results of those runs, and the commands that run them from this
repository.

## Setup

Every run uses nine GB300 nodes with four GPUs each, on one RDMA network:

- **Engines.** One TP4 worker per node, in `--disaggregation-mode symmetric` with the `paper`
  preset ([Configuration](../../docs/configuration.md)): host KV tier 4.445× the GPU cache with
  host dedup, peer KV pulls, late decode choice and KV session retention. Models are quantized
  to NVFP4 with an FP8 KV cache and decode speculatively (EAGLE).
- **Router.** `--policy symmetric_pd` with the `paper` preset: prefill placement by the TRADE
  cost function with peer pulls, round-robin decode, late decode choice (`fit_req`), and role
  changes from the planner.
- **Planner.** The nine workers start as 6P3D. The planner learns its iteration-time models
  during the first 1800 s, then decides at the end of every planning window, on 60 s slices,
  and switches when the mean gain exceeds two standard errors. The tables below list the
  splits it chose.
- **Replay.** Trie replays the recorded agents through the prepared path for 5400 s at a fixed
  number of concurrent agents, replacing each finished agent with a new one. Later prompts
  carry the model's own earlier outputs (`replay_context_mode=actual_output`,
  `replay_output_mode=model`); the first agents start at a random point of their trace
  (`warm_start=stationary`, `seed=0`).
- **Score.** The first 1800 s are warmup. The score covers the hour after it, [1800 s, 5400 s).

| | Putnam | Claude Code | OSWorld |
|---|---|---|---|
| Agent | mini-SWE-agent on PutnamBench | Claude Code, real users | computer-use agent on OSWorld 2.0 |
| Model | GLM-5.2 | GLM-5.2 | Kimi-K2.6 |
| Concurrent agents | 224, 256, 288 | 90, 108, 126 | 128, 144, 160 |
| Replay client processes | 16 | 1 | 16 |
| Planning window | 600 s | 600 s | 300 s |
| Router wait model (`--co-schedule-trade-wait`) | `ledger`, λ = 4.6 | `ledger`, λ = 4.6 | `e3`, λ and rate `auto` |
| Workload ([Hugging Face](https://huggingface.co/datasets/crmsndu/evolving-infra-traces)) | `putnam/decode-heavy-100.stationary.jsonl` | `claude-code/cc062126-stationary-full-v1.jsonl` | `osworld/osworld2-kimi-formal.jsonl` and `osworld/screenshots/` |

With 16 client processes, each process replays one shard of the workload at 1/16 of the
concurrency; `shards.json` and `make_shards.py` in the dataset give the shards.

## Run

[`examples/gb300`](../../examples/gb300/README.md) runs the Claude Code experiment at 126
concurrent agents as one Slurm job: it serves on nine nodes, replays, and writes the score.

```bash
python examples/gb300/submit.py --account <account> --partition <partition> --image <image>
```

For 90 or 108 agents, set `benchmark.concurrency` in
[`glm52.yaml`](../../examples/gb300/glm52.yaml). To score another window of a finished run:

```bash
python benchmarks/replay/score.py runs/<job id>/bench --skip-s 1800 --window-s 3600
```

## Results of the paper's runs

Each setting ran twice. The tables give both runs, scored with `benchmarks/replay/score.py` on
[1800 s, 5400 s): completed model requests per second, the token-weighted prefix hit
(`cached / prompt` tokens), the client's time to first token, and the mean end-to-end latency
of a request.

**Claude Code (GLM-5.2)**

| Agents | Requests/s | Prefix hit | TTFT p50 | TTFT p95 | End-to-end mean | Planner |
|---:|---:|---:|---:|---:|---:|---|
| 90 | 5.43, 5.34 | 97.2 %, 97.1 % | 2.0 s, 2.3 s | 9.1 s, 11.0 s | 7.6 s, 8.0 s | 6P3D → 3P6D |
| 108 | 5.74, 5.78 | 96.9 %, 96.9 % | 3.2 s, 3.6 s | 13.1 s, 13.8 s | 10.0 s, 9.7 s | 6P3D → 3P6D → 4P5D → 3P6D; 6P3D → 3P6D |
| 126 | 5.89, 5.76 | 96.3 %, 96.3 % | 5.0 s, 5.2 s | 19.4 s, 22.9 s | 12.9 s, 13.2 s | 6P3D → 3P6D → 4P5D |

**Putnam (GLM-5.2)**

| Agents | Requests/s | Prefix hit | TTFT p50 | TTFT p95 | End-to-end mean | Prefill workers after 6P3D |
|---:|---:|---:|---:|---:|---:|---|
| 224 | 9.11, 8.77 | 98.7 %, 98.5 % | 0.6 s, 1.3 s | 2.1 s, 6.2 s | 13.8 s, 14.9 s | 1, 2, 1, 2; 1, 2, 1 |
| 256 | 7.75, 9.15 | 98.7 %, 98.9 % | 0.9 s, 0.4 s | 4.6 s, 0.8 s | 15.8 s, 16.9 s | 2, 3, 1; 3, 4 |
| 288 | 8.49, 10.09 | 98.8 %, 98.6 % | 0.6 s, 0.5 s | 1.6 s, 18.9 s | 16.3 s, 17.6 s | 2, 3, 2; 2, 3, 1, 3, 2 |

Putnam is decode-heavy: the planner leaves one to four prefill workers. The first run at 256
agents had 572 failed requests; every other run of the three workloads had none.

**OSWorld (Kimi-K2.6)**

| Agents | Requests/s | Prefix hit | TTFT p50 | TTFT p95 | End-to-end mean | Planner |
|---:|---:|---:|---:|---:|---:|---|
| 128 | 4.25, 4.29 | 94.0 %, 94.0 % | 20.9 s, 20.6 s | 35.1 s, 35.5 s | 24.6 s, 24.3 s | 6P3D → 8P1D |
| 144 | 4.29, 4.30 | 93.9 %, 94.0 % | 24.4 s, 24.1 s | 38.6 s, 39.1 s | 28.0 s, 28.0 s | 6P3D → 8P1D |
| 160 | 4.33, 4.35 | 93.9 %, 93.9 % | 27.3 s, 27.2 s | 43.3 s, 41.4 s | 31.4 s, 31.3 s | 6P3D → 8P1D |

## Comparing a run with these

- Compare on the same window. The planner changes the split during a run, so a difference
  over the whole hour can be a difference in when it switched; `score.py` scores any window.
- Compare with the same replay settings. With recorded outputs in the context
  (`replay_context_mode=recorded`) GLM-5.2's speculative decoding accepts 4.3 draft tokens per
  step instead of 5.2 and decode takes 37 % longer per token.
- Engine logs give the two quantities that separate hardware from workload: the time of a
  decode step at a given number of running requests, and the accepted length
  (`Decode batch, #running-req: ..., accept len: ...`).

## Benchmarks on Modal

[`deploy/modal`](../../deploy/modal/README.md) runs two benchmarks on Modal H200 nodes. A correct
build should land in these ranges (one run takes 20–30 minutes plus cold start):

| Benchmark | Topology | Requests in window | Failures | Prefix hit | TTFT p50 |
|---|---|---|---|---|---|
| static | 1 node, 1P1D | 530–550 | 0 | 97.6–97.9 % | 0.39–0.46 s |
| dynamic | 2 nodes, 2P2D → 1P3D → 2P2D | 1190–1250 | 0 | 96.3–96.4 % | 0.41–0.44 s |

Common setup:

- Model DeepSeek-V4-Flash (FP4 MoE, `--moe-runner-backend flashinfer_mxfp4`), served as
  `deepseek-v4-flash`; DSPARK speculative decoding with a fixed synthetic
  oracle accept rate.
- Two TP 4 workers per 8-GPU node, all in `--disaggregation-mode symmetric` over Mooncake RDMA;
  hierarchical (host) cache with DSV4 host dedup; peer-KV servicer on every worker; KV events to
  the Dynamo indexer (`ai-dynamo==1.4.0`) through the hash-query adapter; static memory fraction
  0.71.
- Router policy `symmetric_pd` with forced P/D split, cost-function P selection, KV pull hints from any
  holder, and late D choice (`fit_req` over every D at the P leg's first chunk).
- Replay: Trie with `actual_output` context and model outputs, through the prepared path
  (`benchmarks/replay/run_trie.py`, `cache_salt_mode=global`).
- **KV session retention off** in engines and router, and no session END from the replay: the
  engine does not yet support retention on DSV4's FULL+SWA cache tree.
- The dynamic benchmark adds the planner's live control loop with external role control on the
  router: a role switch is one `PUT /symmetric_pd/roles` on the router and no engine restarts, so
  worker request counters stay monotonic.

A run is valid only when the four checks in [`benchmarks/replay`](../replay/README.md) pass:
input conformance, quiet window, steady-state zero failures and monotonic request counters.
Prefix hit is token-weighted (`cached_prompt_tokens / completed_prompt_tokens` over the window).

[`deploy/modal`](../../deploy/modal/README.md) runs both benchmarks with these checks.
The [quickstart](../../docs/quickstart.md) shows the same stack on one node.
