---
name: serve
description: Install and start Evolving Infra on one host or several and confirm it answers a request. Use when asked to deploy, launch, serve or try Evolving Infra on a machine or cluster, or to adapt the launch to an environment (Docker, Slurm with Pyxis, source install, another model).
---

# Serve Evolving Infra

**Inputs**: the model directory, the hosts (one, or a list), and how commands reach them
(a shell on each, or a scheduler). Ask for what is missing; do not guess host names.
**Returns**: the router URL, the workers and their starting roles, the work directory with
the logs, and the reply to one test request.

Never ask for a token or password in the conversation. Registry and Hugging Face logins are
done by the person; you use the resulting credentials file by path.

## 1. Look at the environment

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
ls /sys/class/infiniband 2>/dev/null                       # RDMA devices; none: no P/D over RDMA
uname -m; for c in evolving-infra sgl-router docker podman enroot srun; do command -v $c; done
```

Run these where the stack will run: on a GPU host, inside the image when one is used. On a
cluster's login node, run them through the scheduler (`srun -N1 --container-image=<image> ...`).

Each host needs GPUs with RDMA NICs and the model at the same path. The launcher finds the NIC
of every GPU and whether `nvidia_peermem` is loaded; you do not pass them. It prints what it
found; `peer memory 0` means the module is not loaded and transfers go through host memory.

## 2. Get the software

- **Image** (every component inside): `docker build -f docker/Dockerfile -t evolving-infra .`,
  or pull the published tag in [`README.md`](../../README.md). Run it with
  `--gpus all --network host --ipc host --device /dev/infiniband --ulimit memlock=-1 -v <models>:/models`.
- **Source**: the three install lines of [`README.md`](../../README.md) (engine, the Python
  package, `cargo build` of the router). `sgl-router` must be on `PATH`.
- `evolving-infra serve --help` must work before going on (under a scheduler, the job's first
  step can be this check).

## 3. Pick the config

| Hosts | Config | Layout |
|---|---|---|
| one | [`examples/gb300/glm52-1node.yaml`](../../examples/gb300/glm52-1node.yaml) | the host's GPUs split into one P and one D |
| several | [`examples/gb300/glm52.yaml`](../../examples/gb300/glm52.yaml) | one worker per host; `--prefill-workers N` start as P |

For another model or GPU, copy a config, set `model`, `served_model_name`, `page_size` and the
`engine_args` the model needs ([`docs/README.md`](../../docs/README.md#models-and-hardware)).
Config paths are relative to the repository checkout; the image has the same files under
`/opt/evolving-infra` (its working directory). A child config with `base:` replaces single
flags of its base and needs that file beside it, so copy both when copying one. If the engine stops with
"Loaded weights leave no GPU memory for the KV cache", the message gives the smallest
`--mem-fraction-static`; raise it in the config or give each worker more GPUs
(`--gpus-per-worker`).

## 4. Start

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
ends when the script does. For several nodes, `python examples/gb300/submit.py --dry-run
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

## 5. Wait and verify

The first host prints `ready: http://<ip>:30900/v1 (N workers, xPyD)` and writes `run/READY`.
A first start takes about 20 minutes (weights, CUDA graphs). Then:

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

If it does not become ready, or the request fails, use the
[`debug-run`](../debug-run/SKILL.md) skill.
