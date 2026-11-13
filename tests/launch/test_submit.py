# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
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


def test_osworld_unpacks_the_screenshot_archive_and_points_the_shards_at_it():
    script = dry_run("--config", str(SUBMIT.parents[1] / "gb300/kimi-k26.yaml"))
    assert "tar -xf $T/osworld/screenshots.tar -C $T/osworld" in script and "[ -d $T/osworld/screenshots ] ||" in script
    assert "[ -e $T/$f ] ||" in script, "files already there are not downloaded again"
    assert "--image-root $T/osworld/screenshots\n" in script and "processes=16" in script and "concurrency=128" in script


def test_a_benchmark_flag_replaces_the_served_flag_of_the_same_name(tmp_path):
    config = tmp_path / "bench.yaml"
    config.write_text(
        f"base: {SUBMIT.parents[1] / 'gb300/glm52.yaml'}\n"
        "benchmarks:\n"
        "  claude-code:\n"
        "    prefill_workers: 6\n"
        "    workload: claude-code/cc062126-stationary-full-v1.jsonl\n"
        "    concurrency: 8\n"
        "    duration: 600\n"
        "    warmup: 60\n"
        "    router_args: [--prefill-wait=forecast, --global-queue, --symmetric-pd-decode-host-return]\n")
    served = dry_run("--config", str(config)).split("<<'CONFIG'")[1].split("CONFIG")[0]
    assert "- --prefill-wait=forecast" in served and "--prefill-wait=tokens" not in served
    assert "- --global-queue\n" in served and "--global-queue=false" not in served
    assert "- --symmetric-pd-decode-host-return" in served, "a new flag is added"
