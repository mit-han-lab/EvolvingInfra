#!/usr/bin/env python3
"""Check AGENTS.md files and skills against docs/contributing/agent-instructions.md.

    python tools/check_agent_files.py

Exit code 1 lists every problem: a file over its line budget, a skill whose frontmatter name
differs from its directory or lacks a description, a skill without its line in AGENTS.md. Standard library only.
"""
import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUDGET = {"root": 200, "guide": 300, "skill": 500}


def tracked(pattern: str) -> list:
    out = subprocess.run(["git", "ls-files", pattern], cwd=ROOT, capture_output=True, text=True).stdout
    return [ROOT / line for line in out.splitlines()]


def lines(path: Path) -> int:
    return len(path.read_text().splitlines())


def problems() -> list:
    found = []
    root = ROOT / "AGENTS.md"
    if lines(root) > BUDGET["root"]:
        found.append(f"AGENTS.md has {lines(root)} lines (limit {BUDGET['root']})")
    for guide in tracked("*/AGENTS.md") + tracked("**/AGENTS.md"):
        if guide != root and lines(guide) > BUDGET["guide"]:
            found.append(f"{guide.relative_to(ROOT)} has {lines(guide)} lines (limit {BUDGET['guide']})")
    index = root.read_text()
    for skill in sorted((ROOT / "skills").glob("*/SKILL.md")):
        name = skill.parent.name
        text = skill.read_text()
        front = re.match(r"---\n(.*?)\n---\n", text, re.S)
        fields = dict(re.findall(r"^(\w+):\s*(.+)$", front.group(1), re.M)) if front else {}
        if fields.get("name") != name:
            found.append(f"{name}: frontmatter name is {fields.get('name')!r}")
        if len(fields.get("description", "")) < 40:
            found.append(f"{name}: description must say what the skill does and when to use it")
        if lines(skill) > BUDGET["skill"]:
            found.append(f"{name}: SKILL.md has {lines(skill)} lines (limit {BUDGET['skill']})")
        if f"skills/{name}/SKILL.md" not in index:
            found.append(f"{name}: not listed in AGENTS.md")
    return found


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog=__doc__.split("\n\n", 2)[2],
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    found = problems()
    for item in found:
        print(item, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
