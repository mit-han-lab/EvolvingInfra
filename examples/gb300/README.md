# GLM-5.2 on GB300

The paper's Claude Code experiment: GLM-5.2 (NVFP4) on nine 4-GPU GB300 nodes, one TP4 worker
per node, EAGLE speculative decoding, the planner changing the P/D split while serving.

| | |
|---|---|
| Nodes | 9 × GB300 (4 GPUs each), one RDMA network |
| Workers | 9 × TP4, starting as 6P3D |
| Workload | Claude Code traces, 126 concurrent tasks, 5400 s; scored on [1800 s, 5400 s) |
| Config | [`glm52.yaml`](glm52.yaml) |

## Prerequisites

- The runtime image ([`docker/`](../../docker/README.md)) and GLM-5.2-NVFP4 at the same path on
  every node (`/models/GLM-5.2-NVFP4` in the config).
- For [`slurm/submit.py`](../slurm/submit.py): Slurm with [Pyxis](https://github.com/NVIDIA/pyxis).

## Run it with Slurm

```bash
python examples/slurm/submit.py --config examples/gb300/glm52.yaml \
  --account <account> --partition <partition> --image <image>
```

The job starts `evolving-infra serve` on every node, waits until the router is ready, downloads
the workload from [Hugging Face](https://huggingface.co/datasets/crmsndu/evolving-infra-traces),
replays it and writes `runs/<job id>/score.json`. Each node's logs are in `runs/<job id>/<host>/logs/`,
the planner's decisions in `runs/<job id>/<first host>/planner/role-decisions.jsonl`.

## Run it without Slurm

Run the same command on all nine nodes, inside the image:

```bash
evolving-infra serve --config examples/gb300/glm52.yaml --hosts <node0>,<node1>,...,<node8> \
  --planner --prefill-workers 6
```

Each node finds itself in `--hosts`. The first one also runs the KV directory, the frontend, the router and the planner. When it prints
`ready`, replay the workload from it as in [benchmarks/replay](../../benchmarks/replay/README.md)
and score it with `benchmarks/replay/score.py <dir> --skip-s 1800 --window-s 3600`.

## On one node

```bash
evolving-infra serve --config examples/gb300/glm52-1node.yaml
```

The node's 4 GPUs become two TP2 workers, one P and one D, and the router listens on port 30900.
[`glm52-1node.yaml`](glm52-1node.yaml) starts from `glm52.yaml` and fits the model at TP2.

## Results

The paper's run of this experiment, scored with `benchmarks/replay/score.py`. Times are
seconds after the replay's first event.

| Window | Split | Requests/s | Prefix hit | TTFT p50 | TTFT p95 | End-to-end mean | Failed |
|---|---|---:|---:|---:|---:|---:|---:|
| [1800, 5400), the score | changes | 5.89 | 96.3 % | 5.0 s | 19.4 s | 12.9 s | 0 |
| [1200, 1800) | 6P3D | 4.91 | 96.6 % | 7.0 s | 28.0 s | 17.9 s | 0 |
| [1937, 3024) | 3P6D | 5.67 | 96.7 % | 5.1 s | 19.5 s | 12.8 s | 0 |
| [3144, 5400) | 4P5D | 6.00 | 96.1 % | 4.9 s | 19.4 s | 12.9 s | 0 |

The planner switched from 6P3D to 3P6D at 1818 s and to 4P5D at 3025 s, and held after that.
In the 6P3D window a request has 4,298 uncached prompt tokens and 1,126 output tokens on
average, and decode takes 7.0 ms per output token (median). Three runs of this configuration
scored 5.76 to 5.89 requests/s on the whole window and 4.71 to 4.91 on the 6P3D window.

The same experiment run from this repository with `slurm/submit.py` scored 5.83 requests/s
on the whole window with no failed request: 4.78 at 6P3D, 5.77 at 3P6D after the switch at
1878 s, and 5.88 at 4P5D after the switch at 3085 s.

The time of the first switch decides the whole-window score. `planner.cold_start_max_wait_secs: 1`
in the config starts the planner's 1800 s of learning with the replay, as in the paper's
runs. With the planner's default, which first waits for the cache-hit share to settle,
learning started at 1443 s, the first switch came at 3258 s, and the whole window scored 4.35.
