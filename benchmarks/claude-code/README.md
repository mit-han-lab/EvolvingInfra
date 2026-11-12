# Claude Code workload

Sessions of real Claude Code users, from SemiAnalysis's
[AgentX Claude Code traces](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126)
(Apache-2.0): 393 tasks, 98,786 requests, 21.6 B prompt tokens, prompts up to 987 K tokens.
It is prefill-heavy with long shared prefixes.

| File ([Hugging Face](https://huggingface.co/datasets/crmsndu/evolving-infra-traces)) | Model |
|---|---|
| `claude-code/cc062126-stationary-full-v1.jsonl` | GLM-5.2, the paper's GB300 runs |
| `claude-code/cc062126-dsv4-flash.jsonl` | DeepSeek-V4-Flash, the paper's H200 runs |
| `claude-code/cc062126.portable.jsonl` | any model with a `tokenizer.json` |

## What a file holds

The source records each prompt as a sequence of 64-token block ids, without text; equal ids
mean equal content. An event carries `recorded_prompt_block_runs`, and the client expands each
block id into 64 token ids, so the server sees the recorded prefix sharing between requests
and sub-agents ([`trie/docs/cc-trace-replay.md`](../trie/docs/cc-trace-replay.md)). Tool and
user waits longer than 120 s are clipped to 120 s.

## Reproduce the paper

Nine GB300 nodes, GLM-5.2, starting as 6P3D, one replay client process, planning window 600 s,
token-count wait (`--prefill-wait tokens`) with λ = 4.6. [`examples/gb300`](../../examples/gb300/README.md)
runs it at 126 concurrent agents as one Slurm job:

```bash
python examples/slurm/submit.py --account <account> --partition <partition> --image <image>
```

For 90 or 108 agents set `benchmarks.claude-code.concurrency` in
[`glm52.yaml`](../../examples/gb300/glm52.yaml).

Results of the paper's runs on [1800 s, 5400 s), two runs per setting:

| Agents | Requests/s | Prefix hit | TTFT p50 | TTFT p95 | End-to-end mean | Planner |
|---:|---:|---:|---:|---:|---:|---|
| 90 | 5.43, 5.34 | 97.2 %, 97.1 % | 2.0 s, 2.3 s | 9.1 s, 11.0 s | 7.6 s, 8.0 s | 6P3D → 3P6D |
| 108 | 5.74, 5.78 | 96.9 %, 96.9 % | 3.2 s, 3.6 s | 13.1 s, 13.8 s | 10.0 s, 9.7 s | 6P3D → 3P6D → 4P5D → 3P6D; 6P3D → 3P6D |
| 126 | 5.89, 5.76 | 96.3 %, 96.3 % | 5.0 s, 5.2 s | 19.4 s, 22.9 s | 12.9 s, 13.2 s | 6P3D → 3P6D → 4P5D |

[`examples/gb300`](../../examples/gb300/README.md#results) has the 126-agent result per P/D
split.

## Another model

Replay `cc062126.portable.jsonl` with `tokenizer_model=<model directory>`. A hash-block file
names the token ids its blocks draw from in `metadata.block_tokens.pool`, the regular
vocabulary of one tokenizer. The portable file has no pool, and the client reads the
vocabulary from the model's `tokenizer.json`; with GLM-5.2's tokenizer it sends the token ids
of the GLM-5.2 file. [`trie/scripts/portable_blocks.py`](../trie/scripts/portable_blocks.py)
removes the pool from any hash-block file.
