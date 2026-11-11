# Workloads

The agent workloads of the paper are on Hugging Face:
[crmsndu/evolving-infra-traces](https://huggingface.co/datasets/crmsndu/evolving-infra-traces).
Each is one JSONL file that the replay client reads as it is (`workload_path=`); every file has
its sha256 beside it.

```bash
hf download crmsndu/evolving-infra-traces --repo-type dataset --local-dir traces
```

| Workload | File | Tasks | Requests | Model of the paper's runs |
|---|---|---:|---:|---|
| Claude Code | `claude-code/cc062126-stationary-full-v1.jsonl` | 393 | 98,786 | GLM-5.2 |
| Claude Code | `claude-code/cc062126-dsv4-flash.jsonl` | 393 | 98,786 | DeepSeek-V4-Flash |
| Claude Code | `claude-code/cc062126.portable.jsonl` | 393 | 98,786 | any (see below) |
| Putnam | `putnam/decode-heavy-100.stationary.jsonl` | 100 | 6,348 | GLM-5.2 |
| OSWorld | `osworld/osworld2-kimi-formal.jsonl`, `osworld/screenshots/` | 98 | 14,920 | Kimi-K2.6 |

## What a file holds

One line is one task: an agent session with its sub-agents. A task lists `generate` events
(a request with a fixed output length), `wait` events (tool and user time) and the edges that
say when a sub-agent starts and joins.

- **Claude Code** (schema 3, hash blocks). The source records each prompt as a sequence of
  64-token block ids, without text; equal ids mean equal content. An event carries
  `recorded_prompt_block_runs`, and the client expands each block id into 64 token ids, so the
  prefixes the server sees are the recorded ones ([`trie/docs/cc-trace-replay.md`](trie/docs/cc-trace-replay.md)).
- **Putnam** (schema 2, messages). Events carry the recorded chat messages and tool calls; the
  client renders each prompt with the model's chat template.
- **OSWorld** (schema 2, token ids and images). Events carry the prompt token ids of the
  recording model and the screenshots of the step (`recorded_image_data`, files under
  `metadata.image_root`).

## Replaying

[`examples/gb300/submit.py`](../examples/gb300/submit.py) replays Claude Code as in the paper;
[`replay/README.md`](replay/README.md) has the command and its arguments. The paper replays
Claude Code with one client process and Putnam and OSWorld with 16; `make_shards.py` and
`shards.json` in the dataset write the per-process files, and `--image-root` points the OSWorld
rows at the downloaded screenshots.

## On another model

| Workload | Bound to | On another model |
|---|---|---|
| Claude Code | the portable file: nothing | Replay `cc062126.portable.jsonl` with `tokenizer_model=<model directory>`. |
| Putnam | nothing | Replay as is. |
| OSWorld | Kimi-K2.6 | Bind the rows to the model's image layout: [`trie/scripts/bind_cua_kimi_images.py`](trie/scripts/bind_cua_kimi_images.py), [`trie/scripts/convert_cua_deepseek_v4.py`](trie/scripts/convert_cua_deepseek_v4.py). |

A hash-block file names the token ids its blocks draw from in `metadata.block_tokens.pool`: the
regular vocabulary of one tokenizer. Without `pool` the client reads that vocabulary from
`tokenizer_model` (a model directory with `tokenizer.json`), so one file serves every such
model. With GLM-5.2's tokenizer the portable file sends the token ids of the GLM-5.2 file.
[`trie/scripts/portable_blocks.py`](trie/scripts/portable_blocks.py) removes the pool from any
hash-block file.
