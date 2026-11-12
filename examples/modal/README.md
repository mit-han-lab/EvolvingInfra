# Benchmarks on Modal

`app.py` runs the [Modal benchmarks](../../benchmarks/reproduce/README.md#benchmarks-on-modal) on
Modal H200 nodes, with every component built from this checkout. Its settings are constants in
`stack.py`; it does not read the YAML configs of `examples/`.

| Benchmark | Nodes | Workers | Replay | Scored window |
|---|---|---|---|---|
| `static` | 1 | 1P1D | 1200 s | 600-1200 s |
| `dynamic` | 2 | 2P2D; the planner learns for 600 s once the cold start is over, then switches to 1P3D 150 s after the fit and back 300 s later | 1800 s | 900-1800 s |
| `paper` | 8 (64 GPUs) | the paper's H200 run: 12P4D, the planner learns for 1800 s from the first completed request and decides every 600 s; Claude Code at 384 agents from 32 Trie processes | 5400 s | 1800-5400 s |

```bash
m() { uv run --no-project --with modal==1.5.5 modal "$@"; }
uv run --no-project --with modal==1.5.5 python examples/modal/app.py --dry-run [--dynamic]  # offline: commands, configs, flag/env name checks
m run examples/modal/app.py::check_image_main            # CPU container: image contents, name checks
m run --detach examples/modal/app.py::static
m run --detach examples/modal/app.py::dynamic
m run --detach examples/modal/app.py::paper
```

## What it builds and starts

- **Image**: `docker/Dockerfile`, built by Modal from this checkout (`modal.Image.from_dockerfile`),
  plus the three harness modules. Nothing is built locally.
- **Every node**: two TP-4 `decoupled_engine` workers in symmetric mode over Mooncake RDMA, the
  frontends assigned to the node (one per Trie process, `python -m evolving_infra.router.frontend`),
  and in dynamic runs the planner's trace server (`python -m evolving_infra.planner serve-traces`).
- **Rank 0** additionally runs the indexer supervisor (`python -m evolving_infra.router.indexer`,
  Dynamo 1.4.0), the router (`evolving_infra/router/core`, binary `sgl-router`), the planner controller in dynamic
  runs (`python -m evolving_infra.planner control`), and Trie through
  `benchmarks/replay/run_trie.py`.

`stack.py` holds every command and setting (memory fraction 0.71,
host tier 4.445, DSPARK with a fixed oracle accept rate 0.6, peer-KV servicer, late D, cost-function
P selection with pull hints). KV session retention is off in engines and router and the replay sends
no session END: the engine does not support retention on DSV4's FULL+SWA tree yet.

## How a run is judged

Rank 0 writes `summary.json`; `valid` is true only when all of these pass:

1. **Input conformance**: the workload and `tokenizer.json` match the sha256 its offline
   validation report attests (`benchmarks/replay/check_inputs.py`), before and after sharding; the
   shard digests are recorded.
2. **Conformance**: Trie's strict-replay conformance probe against each worker directly
   (`--mixed-batch-lag strict`), after a quiet window.
3. **Quiet window** before load: every worker drained, request counters unchanged for 60 s.
4. **Steady-state zero failures**: requests admitted in the scored window completed with zero
   failures, every Trie result has zero failed requests, clients exit 0, no worker exited, and the
   replay ran its full duration.
5. **Monotonic counters**: no worker's `sglang:num_requests_total` goes backwards across the
   periodic snapshots (role switches do not restart engines).
6. **Method evidence**: router alive and every worker registered at start and end; static roles as
   configured; for the planner runs also the planner's fit completed, the scripted switches
   confirmed (dynamic), controller alive at the end of the replay and the router's roles equal to the last applied update.

The window score ([`benchmarks/replay/score.py`](../../benchmarks/replay/score.py)) reports requests admitted in the window (warm-start
requests excluded), token-weighted prefix hit, client TTFT percentiles.

## Isolation and data

- App `evolving-infra-bench`, results Volume `evolving-infra-bench-runs`, kernel-cache Volume
  `evolving-infra-bench-kcache` and control Dict `evolving-infra-bench-ctl` belong to this harness only;
  `EVOLVING_INFRA_APP`, `_RESULTS`, `_KCACHE` and `_CTL` rename them for a parallel deployment.
- Read-only inputs, which you provide as two existing Volumes (defaults `evolving-infra-weights` and
  `evolving-infra-inputs`; set `EVOLVING_INFRA_WEIGHTS` / `EVOLVING_INFRA_INPUTS` to use others):
  - weights, mounted at `/weights`: a Hugging Face cache holding `deepseek-ai/DeepSeek-V4-Flash-0731`
    at the snapshot `stack.MODEL` names (`hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/...`);
  - inputs, mounted at `/inputs`: the workload `stack.WORKLOAD` (`workloads/cc062126-dsv4-flash.jsonl`)
    and its offline validation report beside it (`<workload>.validate.json`). The workload is the Claude Code
    trace of [cc-trace-replay](../../benchmarks/trie/docs/cc-trace-replay.md) for the DeepSeek-V4-Flash
    tokenizer (tool waits capped at 2 min), as replayed in the paper's H200 runs. Download it from
    [Hugging Face](https://huggingface.co/datasets/crmsndu/evolving-infra-traces) and put it on the
    Volume, with the report under the name the input check reads:

    ```bash
    hf download crmsndu/evolving-infra-traces --repo-type dataset --local-dir traces \
      --include 'claude-code/cc062126-dsv4-flash.jsonl*'
    modal volume put evolving-infra-inputs traces/claude-code/cc062126-dsv4-flash.jsonl \
      workloads/cc062126-dsv4-flash.jsonl
    modal volume put evolving-infra-inputs traces/claude-code/cc062126-dsv4-flash.jsonl.validation.json \
      workloads/cc062126-dsv4-flash.jsonl.validate.json
    ```

    The harness checks it against the sha256 its validation report attests (`2916be77…`).

  No secrets are used.
- Each run writes `<run_id>/node<rank>/` (engine, router, frontend and indexer logs and commands,
  `metrics/`, `replay/` with Trie results and lifecycles, planner outputs in dynamic runs).
  The role-control token stays on the node outside that directory.
- The first run on a new engine tree compiles kernels (cold start); rank 0 saves the JIT caches to
  the kernel-cache Volume at the end, keyed by the engine's git tree hash.

## Limits

- H200 on GCP only; GPU type and placement are fixed so that results are comparable across runs.
- Modal can pre-empt GPU containers; a pre-empted run is invalid and is rerun with a new run id.
- The harness covers one, two and eight nodes.
