# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""AGENTS.md files and skills follow docs/contributing/agent-instructions.md."""
import importlib.util
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "tools/check_agent_files.py"
spec = importlib.util.spec_from_file_location("check_agent_files", PATH)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_agent_files_follow_the_rules():
    assert check.problems() == []


TEMPLATE = check.ROOT / "skills/SKILL_TEMPLATE.md"


def test_the_template_follows_the_skill_format():
    assert check.skill_format(TEMPLATE, TEMPLATE.read_text()) == []


def test_skill_format_names_each_departure_from_the_template():
    text = TEMPLATE.read_text()
    cases = {
        "### 1. <Action>": ("### 0. <Action>", "steps are not"),
        "**Check**: <...>.": ("", "has no **Check**: line"),
        "- **Never changes**:": ("- **Leaves alone**:", "Scope lists"),
        "## Returns": ("## Output", "sections are"),
        "Or `repository`.": ("Or `repository`, `benchmarks/missing/`.", "Scope path benchmarks/missing/"),
        "## Pitfalls": ("[guide](../missing.md)\n\n## Pitfalls", "link ../missing.md"),
    }
    for old, (new, problem) in cases.items():
        assert old in text
        found = check.skill_format(TEMPLATE, text.replace(old, new, 1))
        assert any(problem in item for item in found), (old, found)
