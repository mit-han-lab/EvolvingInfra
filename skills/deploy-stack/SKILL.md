---
name: deploy-stack
description: Install and start Evolving Infra on one host or several and confirm it answers a request. Use when asked to deploy, launch, serve or try Evolving Infra on a machine or cluster, to serve a model or GPU the examples do not cover, or to adapt the launch to an environment (Docker, Slurm with Pyxis, source install). Not for a stack that does not start or answer; use `diagnose-run`.
---

# Deploy Evolving Infra

Start Evolving Infra on one host or several and confirm that it answers a request. Needs GPU
hosts with RDMA NICs and the model at the same path on each.

## Scope

- **Reads**: `evolving_infra/launch/`, `examples/`, `docker/`: the launcher, the configs and
  the image.
- **Writes**: configs under `examples/<hardware>/` for a model or GPU without an example, the
  job script, and the work directory with the logs.
- **Never changes**: `evolving_infra/engine/`, `evolving_infra/router/`,
  `evolving_infra/planner/`: a new model or cluster does not need them. If it seems to, stop
  and say why.
- **Outside the repository**: starts the stack's processes on the GPU hosts, or submits a
  Slurm job that holds them. Confirm the hosts or the job with the owner first.

## Inputs

- The model directory.
- The hosts (one, or a list), and how commands reach them (a shell on each, or a scheduler).
  Ask for what is missing; do not guess host names.
- Never ask for a token or password in the conversation. Registry and Hugging Face logins are
  done by the person; you use the resulting credentials file by path.

## Steps

### 1. Look at the environment

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
ls /sys/class/infiniband 2>/dev/null                       # RDMA devices; none: no P/D over RDMA
uname -m; for c in evolving-infra sgl-router docker podman enroot srun; do command -v $c; done
```

Run these where the stack will run: on a GPU host, inside the image when one is used. On a
cluster's login node, run them through the scheduler (`srun -N1 --container-image=<image> ...`).

Each host needs GPUs with RDMA NICs and the model at the same path. The launcher finds the NIC
of every GPU and whether `nvidia_peermem` is loaded; you do not pass them. It prints what it
found; `peer memory 0` means the module is not loaded and GPU memory is registered with the
NICs through dma-buf (`WITH_NVIDIA_PEERMEM=0`), as on GB300 nodes.

**Check**: every host lists its GPUs and at least one RDMA device, and finds `evolving-infra`
or the image's runtime. Without an RDMA device P/D transfers cannot run: say so and stop.

### 2. Get the software

- **Image** (every component inside): `docker build -f docker/Dockerfile -t evolving-infra .`,
  or pull the published tag in [`README.md`](../../README.md). Run it with
  `--gpus all --network host --ipc host --device /dev/infiniband --ulimit memlock=-1 -v <models>:/models`.
- **Source**: the steps of [Without the image](../../docker/README.md#without-the-image)
  (engine, router, and the launcher's own venv). `sgl-router` must be on `PATH`.

**Check**: `evolving-infra serve --help` prints its usage on every host (under a scheduler,
in a job step).

### 3. Pick the config

| Hosts | Config | Layout |
|---|---|---|
| one | [`examples/gb300/glm52-1node.yaml`](../../examples/gb300/glm52-1node.yaml) | the host's GPUs split into one P and one D |
| several | [`examples/gb300/glm52.yaml`](../../examples/gb300/glm52.yaml) | one worker per host; `--prefill-workers N` start as P |

These two are the paper's setup (GLM-5.2 on GB300); use them unchanged to reproduce it. For
any other model or GPU, write the config yourself with the next section. Config paths are relative to the repository checkout; the image has the same files under
`/opt/evolving-infra` (its working directory). A child config with `base:` replaces single
flags of its base and needs that file beside it, so copy both when copying one.

#### A model or GPU without an example

The repository has no config to copy for it. Derive one, on one host first:

1. **Start from the nearest example.** Copy `glm52-1node.yaml` with its base into a new
   directory `examples/<hardware>/` and set `model` and `served_model_name`. The model
   directory needs `tokenizer.json`: the router reads it (`tokenizer:` in the config names
   another file).
2. **Engine flags.** The engine takes SGLang's flags. Take the model's own flags
   (quantization, attention and MoE backends, `--trust-remote-code`) from its model card or
   its SGLang launch command, and replace the GLM-5.2 ones. Drop every `--dsa-*` flag for a
   model without DSA attention. Keep the flags that are not about the model (CUDA graphs,
   timeouts, loader).
3. **`page_size`.** The KV block size, shared by engine, router and KV directory; the launcher
   passes this one value to all three. Tested values: 64 (GLM-5.2), 256 (DeepSeek-V4-Flash,
   `examples/modal/stack.py`). Keep 64 unless the model's attention backend requires another.
4. **Speculative decoding.** Keep the `--speculative-*` flags only when the model ships a draft
   head, with that head's algorithm; otherwise remove them all. The launcher tells the router
   from the flags.
5. **Memory and TP.** Start it. If the engine stops with "Loaded weights leave no GPU memory
   for the KV cache", the message gives the smallest `--mem-fraction-static`; raise it, or
   give each worker more GPUs with `gpus_per_worker`. Lower `--max-running-requests` and the
   prefill chunk when little KV memory is left.
6. **Verify** with step 5 below on one host, then on several.
7. **Planner last.** `--planner` needs `planner.prefill_shape_prior`: a named prior of
   `evolving_infra/planner/perf_model/priors.py` or the coefficients fitted for this model and GPU
   ([`docs/dynamic-pd.md`](../../docs/dynamic-pd.md)). Serve with a fixed split until then.

Report the config with the reply it produced.

**Check**: the config's `model` path exists on every host (inside the container when one is
used), and the config or its base sets `served_model_name`.

### 4. Start

One host:

```bash
evolving-infra serve --config examples/gb300/glm52-1node.yaml --work-dir run
```

Several hosts: the same command on every host. Each host finds itself in `--hosts`; the first
also runs the KV directory, the frontend and the router.

```bash
evolving-infra serve --config examples/gb300/glm52.yaml --hosts h0,h1,h2 --prefill-workers 1 --work-dir run
```

Add `--planner` only when the split should change while serving: the router then accepts
prepared-path requests only ([`docs/dynamic-pd.md`](../../docs/dynamic-pd.md)).

Under Slurm with Pyxis, run the same command in a job. `$RUN` is on a filesystem every node
shares and is mounted in the container; the serve step stays in the background and the job
ends when the script does. For several nodes, `python examples/slurm/submit.py --dry-run
--image <image>` prints this script with `--hosts` filled in from the allocation.

```bash
#!/bin/bash
#SBATCH -N 1 --exclusive -t 01:00:00
RUN=<shared dir>/$SLURM_JOB_ID; mkdir -p $RUN
C="--container-image=<image> --container-mounts=<models>:/models,$RUN:$RUN --no-container-mount-home"
E="set -a; . /etc/environment; set +a; ulimit -l unlimited;"   # Pyxis keeps the job's environment, not the image's
srun --cpu-bind=none $C bash -c "$E evolving-infra serve --config examples/gb300/glm52-1node.yaml --work-dir $RUN" > $RUN/serve.log 2>&1 &
until [ -e $RUN/READY ]; do kill -0 $! 2>/dev/null || { echo "serving failed, see $RUN"; exit 1; }; sleep 10; done
srun --overlap -N1 -n1 $C bash -c '<the request of step 5>' > $RUN/reply.json
```

**Check**: `evolving-infra serve` keeps running and prints no `<name> exited` line;
`run/logs/` holds one log per process.

### 5. Wait and verify

The first host prints `ready: http://<ip>:30900/v1 (N workers, xPyD)` and writes `run/READY`.
A first start takes about 20 minutes (weights, CUDA graphs). Then, for a stack started without
`--planner`:

```bash
curl -s http://127.0.0.1:30900/v1/completions -H 'Content-Type: application/json' \
  -H 'x-agent-context: {"agent_id": "check", "subagent_id": 0, "request_id": 1}' \
  -d '{"model": "<served_model_name>", "prompt": "The capital of France is", "max_tokens": 16, "temperature": 0}'
```

`<served_model_name>` is the config's `served_model_name` (`glm-5.2` in the examples; a child
config takes it from its base). The request runs on the first host, or in a job step on it.
The reply's `choices[0].text` holds generated text. Report the URL, the workers, `run/` and
the reply. Each process logs to `run/logs/<name>.log`; `SIGTERM` to `evolving-infra serve`
stops everything it started.

With `--planner` the same request works: the router serves and counts plain requests too.

**Check**: `run/READY` exists and the reply's `choices[0].text` holds generated text. If it does not become
ready, or the request fails, use the [`diagnose-run`](../diagnose-run/SKILL.md) skill.

## Returns

- The router URL and the workers with their starting roles (the `ready:` line).
- The work directory with the logs, and the reply to the test request.
- For a model or GPU without an example: the config, which belongs in a pull request with a
  row in [`docs/README.md`](../../docs/README.md#models-and-hardware).

## Pitfalls

- The engine stops with "Loaded weights leave no GPU memory for the KV cache": the message
  gives the smallest `--mem-fraction-static`; raise it in the config, or give each worker more
  GPUs (`gpus_per_worker`). A config written for one worker per host gets half the GPUs per
  worker on one host.
- Copy a child config together with its `base:` file.
