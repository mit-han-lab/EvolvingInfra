#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Check AGENTS.md files and skills against docs/contributing/agent-instructions.md.

    python tools/check_agent_files.py

Exit code 1 lists every problem: a file over its line budget; a skill whose frontmatter name
differs from its directory or whose description lacks "Use when"; a skill whose sections differ
from skills/SKILL_TEMPLATE.md (Scope with Reads, Writes, Never changes and Outside the
repository; Inputs; Steps numbered from 1, each with a Check line; Returns; optional Pitfalls);
a path or link in a skill that does not exist; a skill without its line in AGENTS.md
and in skills/README.md; CLAUDE.md, or the .claude/skills or .agents/skills link, missing.
Standard library only.
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


SECTIONS = ["Scope", "Inputs", "Steps", "Returns"]
SCOPE = ["Reads", "Writes", "Never changes", "Outside the repository"]


def section(text: str, title: str) -> str:
    match = re.search(rf"^## {title}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    return match.group(1) if match else ""


def skill_format(skill: Path, text: str) -> list:
    """Return how one SKILL.md differs from skills/SKILL_TEMPLATE.md."""
    found = []
    prose = re.sub(r"^```.*?^```", "", text, flags=re.M | re.S)  # headings inside code are not sections
    titles = re.findall(r"^## (.+)$", prose, re.M)
    if titles not in (SECTIONS, SECTIONS + ["Pitfalls"]):
        found.append(f"sections are {titles}, expected {SECTIONS} and optionally Pitfalls")
    labels = re.findall(r"^- \*\*(.+?)\*\*:", section(prose, "Scope"), re.M)
    if labels != SCOPE:
        found.append(f"Scope lists {labels}, expected {SCOPE}")
    for path in re.findall(r"`([^`\s]+)`", section(prose, "Scope")):
        if "<" not in path and not path.startswith("-") and ("/" in path or "." in path):
            if not (ROOT / path.rstrip("/")).exists():
                found.append(f"Scope path {path} does not exist")
    steps = re.split(r"^### ", section(prose, "Steps"), flags=re.M)[1:]
    numbers = [int(m.group(1)) for s in steps if (m := re.match(r"(\d+)\. ", s))]
    if not steps or numbers != list(range(1, len(steps) + 1)):
        found.append("steps are not ### 1., ### 2., ... in order")
    found += [f"step {s.split(chr(10), 1)[0]} has no **Check**: line" for s in steps if "**Check**:" not in s]
    for link in re.findall(r"\]\(([^)\s]+)\)", prose):
        target = link.split("#", 1)[0]
        if target and not re.match(r"[a-z]+:", target) and not (skill.parent / target).exists():
            found.append(f"link {link} does not exist")
    return found


def problems() -> list:
    found = []
    root = ROOT / "AGENTS.md"
    if lines(root) > BUDGET["root"]:
        found.append(f"AGENTS.md has {lines(root)} lines (limit {BUDGET['root']})")
    for guide in tracked("*/AGENTS.md") + tracked("**/AGENTS.md"):
        if guide != root and lines(guide) > BUDGET["guide"]:
            found.append(f"{guide.relative_to(ROOT)} has {lines(guide)} lines (limit {BUDGET['guide']})")
    claude = ROOT / "CLAUDE.md"
    if not claude.exists() or claude.read_text().strip() != "@AGENTS.md":
        found.append("CLAUDE.md must hold the one line @AGENTS.md, so Claude Code reads AGENTS.md")
    for link, agent in ((".claude/skills", "Claude Code"), (".agents/skills", "Codex")):
        path = ROOT / link
        if not path.is_symlink() or path.resolve() != (ROOT / "skills").resolve():
            found.append(f"{link} must be a link to ../skills, so {agent} finds the skills")
    index = root.read_text()
    catalog = (ROOT / "skills/README.md").read_text()
    for skill in sorted((ROOT / "skills").glob("*/SKILL.md")):
        name = skill.parent.name
        text = skill.read_text()
        front = re.match(r"---\n(.*?)\n---\n", text, re.S)
        fields = dict(re.findall(r"^(\w+):\s*(.+)$", front.group(1), re.M)) if front else {}
        if fields.get("name") != name:
            found.append(f"{name}: frontmatter name is {fields.get('name')!r}")
        if len(fields.get("description", "")) < 40 or "Use when" not in fields.get("description", ""):
            found.append(f"{name}: description must say what the skill does and, with \"Use when\", when to use it")
        if lines(skill) > BUDGET["skill"]:
            found.append(f"{name}: SKILL.md has {lines(skill)} lines (limit {BUDGET['skill']})")
        found += [f"{name}: {problem}" for problem in skill_format(skill, text)]
        if f"skills/{name}/SKILL.md" not in index:
            found.append(f"{name}: not listed in AGENTS.md")
        if f"]({name}/SKILL.md)" not in catalog:
            found.append(f"{name}: not listed in skills/README.md")
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
