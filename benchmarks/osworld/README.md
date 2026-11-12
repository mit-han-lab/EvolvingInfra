# OSWorld workload

A computer-use agent working on OSWorld 2.0 tasks through screenshots and normalized click and
keyboard tools, recorded with Kimi-K2.6: 98 tasks, 14,920 requests, 2.0 M output tokens.
Every step adds a screenshot, so it is prefill-heavy.

Files ([Hugging Face](https://huggingface.co/datasets/crmsndu/evolving-infra-traces)):
`osworld/osworld2-kimi-formal.jsonl`, `osworld/shards.json`, and `osworld/screenshots.tar`
(12,838 screenshots named by sha256, 5.9 GB; it unpacks to `screenshots/`).

## What a file holds

Events carry the prompt token ids of Kimi-K2.6 and the screenshots of the step
(`recorded_image_data`, files under `metadata.image_root`).

## Reproduce the paper

Nine GB300 nodes, Kimi-K2.6 (NVFP4, EAGLE3 with a draft model), starting as 6P3D, planning
window 300 s, the router's prefill completion forecast and global queue (the router defaults)
with the copy price and the prefill rate measured while serving.
[`examples/gb300/kimi-k26.yaml`](../../examples/gb300/kimi-k26.yaml) holds all of it: the engine
flags, the image layout for the frontend, and the router's flags with the forecast fit in
[`examples/gb300/forecast/`](../../examples/gb300/forecast). The replay uses
16 client processes, each on one shard at 1/16 of the concurrency (128, 144 or 160 agents in
total). As one Slurm job, at 128 agents:

```bash
python examples/slurm/submit.py --config examples/gb300/kimi-k26.yaml \
  --account <account> --partition <partition> --image <image>
```

The job downloads the workload and the screenshots, writes the shards with the screenshots'
path, and starts the 16 processes. For 144 or 160 agents set `benchmarks.osworld.concurrency`.

The router reads a `tokenizers` file, and Kimi-K2.6 ships a tiktoken vocabulary. Export it
once into the model directory, where the config's `tokenizer` points:

```bash
python benchmarks/osworld/scripts/export_tokenizer.py /models/Kimi-K2.6-NVFP4
```

Results of the paper's runs on [1800 s, 5400 s), two runs per setting:

| Agents | Requests/s | Prefix hit | TTFT p50 | TTFT p95 | End-to-end mean | Planner |
|---:|---:|---:|---:|---:|---:|---|
| 128 | 4.25, 4.29 | 94.0 %, 94.0 % | 20.9 s, 20.6 s | 35.1 s, 35.5 s | 24.6 s, 24.3 s | 6P3D → 8P1D |
| 144 | 4.29, 4.30 | 93.9 %, 94.0 % | 24.4 s, 24.1 s | 38.6 s, 39.1 s | 28.0 s, 28.0 s | 6P3D → 8P1D |
| 160 | 4.33, 4.35 | 93.9 %, 93.9 % | 27.3 s, 27.2 s | 43.3 s, 41.4 s | 31.4 s, 31.3 s | 6P3D → 8P1D |

The 128-agent experiment run from this repository with `slurm/submit.py` scored 4.22
requests/s on the same window with no failed request: prefix hit 94.0 %, TTFT p50 20.8 s,
p95 34.0 s, end-to-end mean 24.7 s. The planner went from 6P3D to 8P1D at 1863 s and held.

## Another model

The rows are bound to Kimi-K2.6: its token ids and its image token layout. Bind the messages
and screenshots to the other model's image layout first:
[`trie/scripts/bind_cua_kimi_images.py`](../trie/scripts/bind_cua_kimi_images.py),
[`trie/scripts/convert_cua_deepseek_v4.py`](../trie/scripts/convert_cua_deepseek_v4.py).
