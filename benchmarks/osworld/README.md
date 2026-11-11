# OSWorld workload

A computer-use agent working on OSWorld 2.0 tasks through screenshots and normalized click and
keyboard tools, recorded with Kimi-K2.6: 98 tasks, 14,920 requests, 2.0 M output tokens.
Every step adds a screenshot, so it is prefill-heavy.

Files ([Hugging Face](https://huggingface.co/datasets/crmsndu/evolving-infra-traces)):
`osworld/osworld2-kimi-formal.jsonl`, `osworld/shards.json`, and `osworld/screenshots/`
(12,838 files named by sha256, 5.9 GB).

## What a file holds

Events carry the prompt token ids of Kimi-K2.6 and the screenshots of the step
(`recorded_image_data`, files under `metadata.image_root`).

## Reproduce the paper

Nine GB300 nodes, Kimi-K2.6 (EAGLE3), starting as 6P3D, planning window 300 s, router wait
model `e3` with λ and rate `auto` (`--co-schedule-trade-wait`). The replay uses 16 client
processes: each replays one shard at 1/16 of the concurrency (128, 144 or 160 agents in
total). Write the shards and point them at the screenshots:

```bash
hf download crmsndu/evolving-infra-traces --repo-type dataset --local-dir traces && cd traces
python make_shards.py osworld/osworld2-kimi-formal.jsonl osworld/shards.json shards/osworld \
  --image-root "$PWD/osworld/screenshots"
```

Each process runs the command of [`replay/README.md`](../replay/README.md) on its shard and
writes `trie-results-<i>.json`; [`replay/score.py`](../replay/score.py) scores the directory.

Results of the paper's runs on [1800 s, 5400 s), two runs per setting:

| Agents | Requests/s | Prefix hit | TTFT p50 | TTFT p95 | End-to-end mean | Planner |
|---:|---:|---:|---:|---:|---:|---|
| 128 | 4.25, 4.29 | 94.0 %, 94.0 % | 20.9 s, 20.6 s | 35.1 s, 35.5 s | 24.6 s, 24.3 s | 6P3D → 8P1D |
| 144 | 4.29, 4.30 | 93.9 %, 94.0 % | 24.4 s, 24.1 s | 38.6 s, 39.1 s | 28.0 s, 28.0 s | 6P3D → 8P1D |
| 160 | 4.33, 4.35 | 93.9 %, 93.9 % | 27.3 s, 27.2 s | 43.3 s, 41.4 s | 31.4 s, 31.3 s | 6P3D → 8P1D |

## Another model

The rows are bound to Kimi-K2.6: its token ids and its image token layout. Bind the messages
and screenshots to the other model's image layout first:
[`trie/scripts/bind_cua_kimi_images.py`](../trie/scripts/bind_cua_kimi_images.py),
[`trie/scripts/convert_cua_deepseek_v4.py`](../trie/scripts/convert_cua_deepseek_v4.py).
