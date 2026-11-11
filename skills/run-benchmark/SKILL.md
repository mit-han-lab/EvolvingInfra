---
name: run-benchmark
description: Run the paper's replay benchmark on Evolving Infra, score it and compare it with the reference results. Use when asked to reproduce the paper's numbers, benchmark a change, or produce the performance table of a pull request.
---

# Run the replay benchmark

**Inputs**: a config ([`examples/gb300/glm52.yaml`](../../examples/gb300/glm52.yaml) for the
paper's Claude Code experiment), the cluster (Slurm account, partition, the runtime image),
and what the result is compared with.
**Returns**: the score table with its windows, the planner's switch times, whether the run is
valid, and the comparison.

A full run holds nine GPU nodes for about two hours (20 minutes to start, 90 minutes of
replay). Confirm with the owner before submitting.

## 1. Submit

```bash
python examples/gb300/submit.py --config <config> --account <a> --partition <p> --image <image> \
  --mounts <model directory>:/models --extra-args "<more sbatch options>" --output-dir <shared dir>/runs --dry-run
```

Read the dry run: the `#SBATCH` line, the mounts (the model must be at the config's `model`
path inside the container), and the replay command (`replay_context_mode=actual_output`).
Then run it without `--dry-run`. `--output-dir` is on storage every node sees. Site settings
can also stay in a copy of the config (`slurm:` section). Without Slurm: start the stack with
the [`serve`](../serve/SKILL.md) skill, then run the replay and scoring commands of the dry
run on the host that runs the router.

## 2. Watch

Under `<output dir>/<job id>/`: `serve.log` (ends with `ready: ... (9 workers, 6P3D)`), one
directory per host with `logs/<process>.log`, `bench.log` (the replay client), and at the end
`score.json`. The host directory that holds `READY` is the one that runs the router and the
planner ("the head" below). The job stops the stack when the replay ends, so `srun: ...
Killed` lines at the end of `serve.log` are the normal stop. If `serve.log` shows
`<name> exited` or the job ends early, use the [`debug-run`](../debug-run/SKILL.md) skill.

## 3. Read the result

```bash
cat <run>/score.json                                           # the scored window [warmup, duration)
python skills/run-benchmark/scripts/switch_times.py <run>/<head>/planner <run>/bench
```

The run is valid when all of these hold:

- `failed_requests` is 0 (`cancelled_requests` are requests cut off when the replay ends);
- `switch_times.py` exits 0 (the planner did not fail);
- `replay_context_modes` is `["actual_output"]`. A run replayed with another mode does not
  compare with the reference.

`cache_hit` is the prefix hit: cached prompt tokens over prompt tokens.

## 4. Compare

The reference is in [`examples/gb300/README.md`](../../examples/gb300/README.md): the
whole-window result, the reference run's switch times, and its result on a window of each
P/D split.

1. Compare the whole scored window.
2. Compare one window per split. `switch_times.py` gives this run's switches on the replay
   clock; start a window 120 s after a switch and end it at the next one:

   ```bash
   python benchmarks/replay/score.py <run>/bench --skip-s <switch + 120> --window-s <length>
   ```

   A whole-window gap with equal per-split results is a gap in when the planner switched.
3. Three runs of the reference configuration differ by about 4 % in requests per second, on
   the whole window and on the 6P3D window. For a larger gap on a like-for-like window, use
   "Slower than the reference" in [`debug-run`](../debug-run/SKILL.md) before concluding.

## 5. Report

One line with the setup (model, nodes, starting split, workload, concurrency), then a table
with a row per window for the run and the reference: window, split, requests per second,
prefix hit, time to first token (p50, p95), mean end-to-end latency, failed requests. Then
both runs' switch times and the reproduction command.
