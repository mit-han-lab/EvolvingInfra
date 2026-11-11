---
name: run-benchmark
description: Run the paper's replay benchmark on Evolving Infra, score it and compare it with the reference results. Use when asked to reproduce the paper's numbers, benchmark a change, or produce the performance table of a pull request. Not for finding why a run failed or is slower than the reference; use `diagnose-run`.
---

# Run the replay benchmark

Run one of the paper's replay benchmarks on Evolving Infra, score it, and compare it with the
reference results.

## Scope

- **Reads**: `benchmarks/`, `examples/`: the workload's README and `AGENTS.md`, the configs and
  `examples/slurm/submit.py`.
- **Writes**: the run directory under `--output-dir`; nothing in the repository.
- **Never changes**: `evolving_infra/`, `benchmarks/`: a benchmark measures the code as it is,
  and a run with a replay setting other than the reference's does not compare with it.
- **Outside the repository**: submits a Slurm job that holds nine GPU nodes for about two
  hours (20 minutes to start, 90 minutes of replay). Confirm with the owner before submitting.

## Inputs

- The workload: [`claude-code`](../../benchmarks/claude-code/README.md),
  [`putnam`](../../benchmarks/putnam/README.md) or
  [`osworld`](../../benchmarks/osworld/README.md).
- A config: [`examples/gb300/glm52.yaml`](../../examples/gb300/glm52.yaml) for the paper's
  Claude Code experiment.
- The cluster: Slurm account, partition, the runtime image.
- What the result is compared with.

## Steps

### 1. Submit

Read the workload's README and `AGENTS.md` first: they give the paper's setup, its results and
the rules of that workload. `submit.py --benchmark <name>` runs a benchmark of the config as
one job: `claude-code` (the default) and `putnam` in `examples/gb300/glm52.yaml`. OSWorld
has no benchmark entry: start the stack with the [`deploy-stack`](../deploy-stack/SKILL.md) skill, write
the shards as its README says, run one replay process per shard on the host that runs the
router, then score and compare as in steps 3 and 4.

```bash
python examples/slurm/submit.py --config <config> --account <a> --partition <p> --image <image> \
  --mounts <model directory>:/models --extra-args "<more sbatch options>" --output-dir <shared dir>/runs --dry-run
```

Read the dry run: the `#SBATCH` line, the mounts (the model must be at the config's `model`
path inside the container), and the replay command (`run_trie.py` with `stack`, `workload_path`, `concurrency`,
`duration` and `output_dir`; every other replay setting is the paper's by default).
Then run it without `--dry-run`. `--output-dir` is on storage every node sees. Site settings
can also stay in a copy of the config (`slurm:` section). Without Slurm: start the stack with
the [`deploy-stack`](../deploy-stack/SKILL.md) skill, then run the replay and scoring commands of the dry
run on the host that runs the router.

**Check**: the dry run shows the `#SBATCH` line, the model mounted at the config's `model`
path, and the `run_trie.py` command; the submission prints a job ID.

### 2. Watch

Under `<output dir>/<job id>/`: `serve.log` (ends with `ready: ... (9 workers, 6P3D)`), one
directory per host with `logs/<process>.log`, `bench.log` (the replay client), and at the end
`score.json`. The host directory that holds `READY` is the one that runs the router and the
planner ("the head" below). The job stops the stack when the replay ends, so `srun: ...
Killed` lines at the end of `serve.log` are the normal stop. If `serve.log` shows
`<name> exited` or the job ends early, use the [`diagnose-run`](../diagnose-run/SKILL.md) skill.

**Check**: `serve.log` shows `ready: ... (9 workers, 6P3D)` and `score.json` exists when the
job ends.

### 3. Read the result

```bash
cat <run>/score.json                                           # the scored window [warmup, duration)
python skills/run-benchmark/scripts/switch_times.py <run>/<head>/planner <run>/bench
```

The run is valid when all of these hold:

- `failed_requests` is 0 (`cancelled_requests` are requests cut off when the replay ends);
- `switch_times.py` exits 0 (the planner did not fail);
- `replay_differs_from_preset` is `{}`. A run with another replay setting does not compare
  with the reference unless that setting is what is being tested.

`cache_hit` is the prefix hit: cached prompt tokens over prompt tokens.

**Check**: all three conditions above hold. Otherwise the run is not valid: report why and do
not compare it.

### 4. Compare

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
   "Slower than the reference" in [`diagnose-run`](../diagnose-run/SKILL.md) before concluding.

**Check**: both runs are compared on the same window and the same P/D split.

## Returns

One line with the setup (model, nodes, starting split, workload, concurrency), then a table
with a row per window for the run and the reference: window, split, requests per second,
prefix hit, time to first token (p50, p95), mean end-to-end latency, failed requests. Then
both runs' switch times and the reproduction command.
