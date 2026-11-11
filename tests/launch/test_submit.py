"""The Slurm job script of the GB300 example."""
from pathlib import Path
import subprocess
import sys

SUBMIT = Path(__file__).resolve().parents[2] / "examples/slurm/submit.py"


def test_replay_command_gives_only_the_experiment_and_takes_the_rest_from_the_preset():
    script = subprocess.run([sys.executable, str(SUBMIT), "--image", "IMG", "--dry-run"],
                            capture_output=True, text=True, check=True).stdout
    lines = [l for l in script.splitlines() if l.startswith(("REPLAY=", "$REPLAY "))]
    names = sorted(a.split("=")[0] for l in lines for a in l.split("run_trie.py")[-1].strip('"').split() if "=" in a)
    assert names == ["concurrency", "duration", "output_dir", "stack", "workload_path"]


def dry_run(*argv):
    return subprocess.run([sys.executable, str(SUBMIT), "--image", "IMG", "--dry-run", *argv],
                          capture_output=True, text=True, check=True).stdout


def test_putnam_runs_sixteen_processes_on_the_published_shards_with_host_return():
    script = dry_run("--benchmark", "putnam")
    assert "make_shards.py $T/putnam/decode-heavy-100.stationary.jsonl $T/putnam/shards.json $R/shards" in script
    assert "processes=16 process=$i workload_path=$R/shards/shard-$i.jsonl" in script and "seq 0 15" in script
    assert "concurrency=224" in script and "--skip-s 1800 --window-s 3600" in script
    served = script.split("<<'CONFIG'")[1].split("CONFIG")[0]
    assert "- --symmetric-pd-decode-host-return" in served and "benchmarks" not in served and "slurm" not in served
    assert "--symmetric-pd-decode-host-return" not in dry_run().split("<<'CONFIG'")[1].split("CONFIG")[0]


def test_h200_benchmark_serves_a_fixed_split():
    script = dry_run("--config", str(SUBMIT.parents[1] / "h200/dsv4-flash.yaml"))
    assert "--planner" not in script and "--prefill-workers 1" in script and "-N 1 " in script
