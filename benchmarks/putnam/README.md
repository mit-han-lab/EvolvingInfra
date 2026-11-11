# Putnam workload

mini-SWE-agent solving PutnamBench theorems in Lean, recorded with GLM-5.2: 100 tasks, 6,348
requests, 14.5 M output tokens. It is decode-heavy.

File ([Hugging Face](https://huggingface.co/datasets/crmsndu/evolving-infra-traces)):
`putnam/decode-heavy-100.stationary.jsonl`, with `putnam/shards.json`.

## What a file holds

Events carry the recorded chat messages, tool calls and tool outputs. The client renders each
prompt with the model's chat template and asks for the recorded number of output tokens.

## Reproduce the paper

Nine GB300 nodes, GLM-5.2, starting as 6P3D, planning window 600 s, router wait model `ledger`
with λ = 4.6; the engine and router settings are those of
[`examples/gb300/glm52.yaml`](../../examples/gb300/glm52.yaml). The replay uses 16 client
processes: each replays one shard at 1/16 of the concurrency (224, 256 or 288 agents in
total). Write the shards with the dataset's script:

```bash
hf download crmsndu/evolving-infra-traces --repo-type dataset --local-dir traces && cd traces
python make_shards.py putnam/decode-heavy-100.stationary.jsonl putnam/shards.json shards/putnam
```

Each process runs the command of [`replay/README.md`](../replay/README.md) on its shard and
writes `trie-results-<i>.json`; [`replay/score.py`](../replay/score.py) scores the directory.

Results of the paper's runs on [1800 s, 5400 s), two runs per setting:

| Agents | Requests/s | Prefix hit | TTFT p50 | TTFT p95 | End-to-end mean | Prefill workers after 6P3D |
|---:|---:|---:|---:|---:|---:|---|
| 224 | 9.11, 8.77 | 98.7 %, 98.5 % | 0.6 s, 1.3 s | 2.1 s, 6.2 s | 13.8 s, 14.9 s | 1, 2, 1, 2; 1, 2, 1 |
| 256 | 7.75, 9.15 | 98.7 %, 98.9 % | 0.9 s, 0.4 s | 4.6 s, 0.8 s | 15.8 s, 16.9 s | 2, 3, 1; 3, 4 |
| 288 | 8.49, 10.09 | 98.8 %, 98.6 % | 0.6 s, 0.5 s | 1.6 s, 18.9 s | 16.3 s, 17.6 s | 2, 3, 2; 2, 3, 1, 3, 2 |

The planner leaves one to four prefill workers. The first run at 256 agents had 572 failed
requests; the others had none.

## Another model

The messages are text: replay the file as it is. Prompt token counts follow the model's
tokenizer and chat template.
