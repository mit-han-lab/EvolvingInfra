"""AGENTS.md files and skills follow docs/contributing/agent-instructions.md."""
import importlib.util
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "tools/check_agent_files.py"
spec = importlib.util.spec_from_file_location("check_agent_files", PATH)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_agent_files_follow_the_rules():
    assert check.problems() == []
