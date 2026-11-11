"""The Slurm job script of the GB300 example."""
from pathlib import Path
import subprocess
import sys

SUBMIT = Path(__file__).resolve().parents[2] / "examples/gb300/submit.py"


def test_replay_puts_the_models_own_outputs_in_later_prompts():
    script = subprocess.run([sys.executable, str(SUBMIT), "--image", "IMG", "--dry-run"],
                            capture_output=True, text=True, check=True).stdout
    assert "replay_context_mode=actual_output" in script
