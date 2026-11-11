#!/usr/bin/env python3
"""Run the paper's experiment as one Slurm job: serve on every node, replay the workload, score it.

    python examples/gb300/submit.py --config examples/gb300/glm52.yaml --account A --partition P --image IMG

Results go to <output-dir>/<job id>/: score.json, the replay's results and every service's logs.
The job script runs `evolving-infra serve` on every node, so the cluster only needs Slurm with Pyxis.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

SCRIPT = """#!/bin/bash
#SBATCH -N {nodes} --ntasks-per-node=1 --exclusive -t {time} -J evolving-infra {account}{partition}{extra}
#SBATCH -o {out}/%j.log
set -uo pipefail
export RUN={out}/$SLURM_JOB_ID; mkdir -p $RUN
cat > $RUN/config.yaml <<'CONFIG'
{config}CONFIG
HOSTS=$(scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd,); HEAD=${{HOSTS%%,*}}
C="--container-image={image} --container-mounts={mounts},{out}:{out} --no-container-mount-home"
# In the container: the image's environment (Pyxis keeps the job's), then slurm.setup.
cat > $RUN/setup.sh <<'SETUP'
set -a; [ -f /etc/environment ] && . /etc/environment; set +a
{setup}
SETUP
P=". $RUN/setup.sh;"
srun --ntasks={nodes} --cpu-bind=none $C bash -c \\
  "$P ulimit -l unlimited; evolving-infra serve --config $RUN/config.yaml --hosts $HOSTS \\
   --planner --prefill-workers {prefill} --work-dir $RUN/\\$(hostname -s)" > $RUN/serve.log 2>&1 &
SERVE=$!
until [ -e $RUN/$HEAD/READY ]; do kill -0 $SERVE 2>/dev/null || {{ echo "serving failed, see $RUN"; exit 1; }}; sleep 10; done
srun --overlap -N1 -n1 -w $HEAD $C bash -c "$P "{bench} > $RUN/bench.log 2>&1
kill $SERVE; wait $SERVE
cat $RUN/score.json
"""

BENCH = """set -e; cd /opt/evolving-infra; R={run}; mkdir -p $R/bench
/opt/trie-venv/bin/python -c "from huggingface_hub import hf_hub_download as d; print(d('crmsndu/evolving-infra-traces', '{workload}', repo_type='dataset', local_dir='$R/traces'))"
export EVOLVING_INFRA_REPLAY='{replay}'
/opt/trie-venv/bin/python benchmarks/replay/run_trie.py workload_path=$R/traces/{workload} \\
  endpoint=http://127.0.0.1:30900/v1 api_key=EMPTY model={name} tokenizer_model={model} concurrency={concurrency} duration={duration} \\
  warm_start=stationary workload_order=natural seed=0 delay_scale=1 strict_replay=true context_source=recorded \\
  replay_output_mode=model trajectory_prefix_tokens=10 migration_limit=0 max_retries=0 stream=true \\
  strict_replay_openai_fast_path=true cache_salt_mode=global prefix_block_size={page} timeout=3600 drain_timeout=0 \\
  num_gpus={gpus} output_json=$R/bench/trie-results-0.json lifecycle_output_jsonl=$R/bench/trie-lifecycle-0.jsonl \\
  lifecycle_run_id=$SLURM_JOB_ID lifecycle_cell_id=bench lifecycle_arm=on lifecycle_concurrency={concurrency}
python3 benchmarks/replay/score.py $R/bench --skip-s {warmup} --window-s {window} > $R/score.json
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(Path(__file__).with_name("glm52.yaml")))
    for key in ("account", "partition", "image", "mounts", "time", "extra-args"):
        parser.add_argument(f"--{key}")
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--dry-run", action="store_true", help="print the job script instead of submitting it")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    slurm, bench = dict(cfg.get("slurm") or {}), cfg["benchmark"]
    for key in ("account", "partition", "image", "mounts", "time", "nodes", "extra_args"):
        if getattr(args, key) is not None:
            slurm[key] = getattr(args, key)
    if not slurm.get("image"):
        sys.exit("give --image (the runtime image) or slurm.image in the config")
    out = Path(args.output_dir).resolve()
    run = "$RUN"  # exported by the job script
    replay = {"router_url": "http://127.0.0.1:30900", "tokenizer_urls": ["http://127.0.0.1:28100"],
              "config_id": "default", "model_name": cfg["served_model_name"], "model_path": cfg["model"],
              "incremental_streaming_output": True, "retention_enabled": True}
    bench_script = BENCH.format(
        run=run, workload=bench["workload"], replay=json.dumps(replay), name=cfg["served_model_name"],
        model=cfg["model"], concurrency=bench["concurrency"], duration=bench["duration"],
        page=cfg.get("page_size", 64), gpus=4 * slurm["nodes"], warmup=bench["warmup"],
        window=bench["duration"] - bench["warmup"])
    script = SCRIPT.format(
        prefill=bench["prefill_workers"], nodes=slurm["nodes"], time=slurm.get("time", "03:00:00"),
        account=f"-A {slurm['account']} " if slurm.get("account") else "",
        partition=f"-p {slurm['partition']}" if slurm.get("partition") else "",
        out=out, image=slurm["image"], mounts=slurm.get("mounts", "/models:/models"),
        extra=f" {slurm['extra_args']}" if slurm.get("extra_args") else "",
        setup=slurm.get("setup") or "",
        config=Path(args.config).read_text(), bench=shlex.quote(bench_script))
    if args.dry_run:
        print(script)
        return
    out.mkdir(parents=True, exist_ok=True)
    job = subprocess.run(["sbatch", "--parsable"], input=script, text=True, capture_output=True, check=True)
    job_id = job.stdout.strip().split(";")[0]
    print(f"submitted job {job_id}; results in {out}/{job_id}/")


if __name__ == "__main__":
    main()
