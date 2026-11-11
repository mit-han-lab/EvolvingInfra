# Editing Agent Instructions

Read this before changing `AGENTS.md`, a component guide (`<dir>/AGENTS.md`) or a skill.

## Size

`AGENTS.md` loads on every agent request, a component guide when the agent enters that
directory, a skill when it is used. Keep `AGENTS.md` under 200 lines, a component guide under
300 and a `SKILL.md` under 500. Over budget: split or remove; do not compress the prose.
`tools/check_agent_files.py` checks these limits and runs in the test suite.

## When not to add a rule

- **Agents already do it.** Try the task without the rule first.
- **It happened once.** Prefer a test, a startup check or an error message over a sentence here.
- **It repeats a doc.** Link to `docs/` or a README.
- **It names a path that moves.** Say what to search for.
- **It contradicts another rule.** Search every guide first and merge them.

## Where content goes

| Scope | File |
|---|---|
| Every change: contribution policy, commands, the repository map | `AGENTS.md` |
| One component: what its code does not show, which tests to run, the order of a change | `<dir>/AGENTS.md` |
| A task with steps: opening a pull request, serving, running a benchmark, debugging a run | a skill |
| What the system does, for users | `docs/` |

A good entry is something the code and the public docs do not say: a convention that differs
from the usual one, a change that needs several files in a known order, a mistake that was
made more than once. Keep it short, specific and actionable.

## Skills

Skills live in `skills/<name>/` and follow the
[Agent Skills specification](https://agentskills.io/specification).

- Frontmatter: `name` equals the directory name; `description` says what the skill does and
  when to use it.
- A skill states its purpose, its inputs, numbered steps, and what it returns. Scripts go in
  `scripts/`, long reference material in `references/`.
- A skill enters the repository after a fresh agent session has followed it to the end on a
  real task. Say in the pull request what that task was.
- List the skill in `AGENTS.md`.
