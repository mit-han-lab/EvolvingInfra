"""The Slurm job script of the GB300 example."""
from pathlib import Path
import subprocess
import sys

SUBMIT = Path(__file__).resolve().parents[2] / "examples/slurm/submit.py"


def test_replay_command_gives_only_the_experiment_and_takes_the_rest_from_the_preset():
    script = subprocess.run([sys.executable, str(SUBMIT), "--image", "IMG", "--dry-run"],
                            capture_output=True, text=True, check=True).stdout
    line = next(l for l in script.rsplit("run_trie.py", 1)[1].replace("\\\n", " ").splitlines() if l)
    assert sorted(a.split("=")[0] for a in line.split() if "=" in a) == [
        "concurrency", "duration", "output_dir", "stack", "workload_path"]
