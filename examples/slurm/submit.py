#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Run one of a config's benchmarks as one Slurm job: serve on every node, replay the workload, score it.

    python examples/slurm/submit.py --config examples/gb300/glm52.yaml --benchmark claude-code \\
        --account A --partition P --image IMG

The config is any `evolving-infra serve` config with a `benchmarks` section, after resolving
`base:`; `slurm` holds site settings. A benchmark's `router_args` and `engine_args` replace the
served config's flags of the same name (the name before `=`) and add the others; its `planner`
is merged key by key.

Results go to <output-dir>/<job id>/: score.json, the replay's results and every service's logs.
The job script runs `evolving-infra serve` on every node, so the cluster only needs Slurm with Pyxis.
"""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evolving_infra.launch.__main__ import _flag, load  # noqa: E402  (resolves `base:`)

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
  "$P ulimit -l unlimited; cd /opt/evolving-infra; evolving-infra serve --config $RUN/config.yaml --hosts $HOSTS \\
   {planner}--prefill-workers {prefill} --work-dir $RUN/\\$(hostname -s)" > $RUN/serve.log 2>&1 &
SERVE=$!
until [ -e $RUN/$HEAD/READY ]; do kill -0 $SERVE 2>/dev/null || {{ echo "serving failed, see $RUN"; exit 1; }}; sleep 10; done
srun --overlap -N1 -n1 -w $HEAD $C bash -c "$P "{bench} > $RUN/bench.log 2>&1
kill $SERVE; wait $SERVE
cat $RUN/score.json
"""

BENCH = """set -e; cd /opt/evolving-infra; R={run}; T={traces}; PY=/opt/trie-venv/bin/python
# The workload files: downloaded once into $T, which slurm.traces can point at a directory kept between runs.
for f in {files}; do [ -e $T/$f ] || $PY -c "from huggingface_hub import hf_hub_download as d; print(d('crmsndu/evolving-infra-traces', '$f', repo_type='dataset', local_dir='$T'))"; done
{images}# Every replay setting not given here is the paper's (benchmarks/replay/run_trie.py, preset=paper).
REPLAY="$PY benchmarks/replay/run_trie.py stack=$R/$(hostname -s) concurrency={concurrency} duration={duration} output_dir=$R/bench"
{replay}
python3 benchmarks/replay/score.py $R/bench --skip-s {warmup} --window-s {window} > $R/score.json
"""
ONE = "$REPLAY workload_path=$T/{workload}"
# The images the workload's requests refer to: one archive of the dataset, unpacked beside it.
IMAGES = """[ -d $T/{unpacked} ] || {{ $PY -c "from huggingface_hub import hf_hub_download as d; print(d('crmsndu/evolving-infra-traces', '{images}', repo_type='dataset', local_dir='$T'))"
  tar -xf $T/{images} -C $T/{parent}; }}
"""
# Several client processes: process i replays shard i at concurrency / N; they start 2 s apart.
MANY = """$PY $T/make_shards.py $T/{workload} $T/{shards} $R/shards{image_root}
for i in $(seq 0 {last}); do
  $REPLAY processes={processes} process=$i workload_path=$R/shards/shard-$i.jsonl > $R/bench-$i.log 2>&1 &
  sleep 2
done
status=0; for job in $(jobs -p); do wait $job || status=1; done; [ $status = 0 ]"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "gb300/glm52.yaml"))
    for key in ("account", "partition", "image", "mounts", "time"):
        parser.add_argument(f"--{key}")
    parser.add_argument("--extra-args", help='more sbatch options, e.g. --extra-args "--gpus-per-node=4"')
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--benchmark", help="a name under `benchmarks` in the config (default: the first)")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--dry-run", action="store_true", help="print the job script instead of submitting it")
    argv = sys.argv[1:]
    for i, arg in enumerate(argv[:-1]):  # sbatch options start with "-": "--extra-args --gpus-per-node=4"
        if arg == "--extra-args":
            argv[i:i + 2] = [f"--extra-args={argv[i + 1]}"]
            break
    args = parser.parse_args(argv)

    cfg = load(Path(args.config))
    benchmarks = cfg.pop("benchmarks", None)
    if not benchmarks:
        sys.exit(f"{args.config} defines no benchmark under `benchmarks`; "
                 "use a config that does, such as examples/gb300/glm52.yaml")
    name = args.benchmark or next(iter(benchmarks))
    if name not in benchmarks:
        sys.exit(f"benchmark {name!r} is not in the config; it has {', '.join(benchmarks)}")
    slurm, bench = dict(cfg.pop("slurm", None) or {}), benchmarks[name]
    for key in ("router_args", "engine_args"):  # the benchmark's flags win over the model's
        if bench.get(key):
            replaced = {_flag(a) for a in bench[key]}
            cfg[key] = [a for a in cfg.get(key) or [] if _flag(a) not in replaced] + list(bench[key])
    if bench.get("planner"):
        cfg["planner"] = {**(cfg.get("planner") or {}), **bench["planner"]}
    for key in ("account", "partition", "image", "mounts", "time", "nodes", "extra_args"):
        if getattr(args, key) is not None:
            slurm[key] = getattr(args, key)
    if not slurm.get("image"):
        sys.exit("give --image (the runtime image) or slurm.image in the config")
    out = Path(args.output_dir).resolve()
    run = "$RUN"  # exported by the job script
    processes = bench.get("processes", 1)
    files = [bench["workload"]] + ([bench["shards"], "make_shards.py"] if processes > 1 else [])
    images = bench.get("images")  # an archive of the dataset, or a directory already on the nodes
    local = bool(images) and images.startswith("/")
    replay = ONE if processes == 1 else MANY.format(
        workload=bench["workload"], shards=bench.get("shards"), processes=processes, last=processes - 1,
        image_root=f" --image-root {images if local else '$T/' + images.removesuffix('.tar')}" if images else "")
    bench_script = BENCH.format(
        run=run, files=" ".join(files), traces=slurm.get("traces") or "$R/traces",
        images=IMAGES.format(images=images, parent=str(Path(images).parent), unpacked=images.removesuffix(".tar"))
        if images and not local else "", replay=replay.format(workload=bench["workload"]) if processes == 1 else replay,
        concurrency=bench["concurrency"], duration=bench["duration"],
        warmup=bench["warmup"], window=bench["duration"] - bench["warmup"])
    script = SCRIPT.format(
        prefill=bench["prefill_workers"], planner="--planner " if bench.get("serve_planner", True) else "", nodes=slurm["nodes"], time=slurm.get("time", "03:00:00"),
        account=f"-A {slurm['account']} " if slurm.get("account") else "",
        partition=f"-p {slurm['partition']}" if slurm.get("partition") else "",
        out=out, image=slurm["image"],
        mounts=slurm.get("mounts", "/models:/models") + (f",{slurm['traces']}:{slurm['traces']}" if slurm.get("traces") else ""),
        extra=f" {slurm['extra_args']}" if slurm.get("extra_args") else "",
        setup=slurm.get("setup") or "",
        config=yaml.safe_dump(cfg, sort_keys=False, width=200), bench=shlex.quote(bench_script))
    if args.dry_run:
        print(script)
        return
    out.mkdir(parents=True, exist_ok=True)
    job = subprocess.run(["sbatch", "--parsable"], input=script, text=True, capture_output=True, check=True)
    job_id = job.stdout.strip().split(";")[0]
    print(f"submitted job {job_id}; results in {out}/{job_id}/")


if __name__ == "__main__":
    main()
