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
[Agent Skills specification](https://agentskills.io/specification). Start a skill from
[`skills/SKILL_TEMPLATE.md`](../../skills/SKILL_TEMPLATE.md); `tools/check_agent_files.py`
checks every rule below that a script can check.

- **Frontmatter**: `name` equals the directory name. `description` says what the skill does
  and, after "Use when", the requests it answers; when another skill covers a nearby case,
  "Not for ...; use `<skill>`" says which.
- **Sections**, in this order: a one- or two-sentence summary under the title, `## Scope`,
  `## Inputs`, `## Steps`, `## Returns`, and optionally `## Pitfalls`.
- **Scope** says where in the repository the skill works, in four lines in this order:
  - `**Reads**`: the paths the agent reads, or `repository`.
  - `**Writes**`: the paths it may create or change, or `nothing`. Anything else is out of
    scope: the agent stops and says why.
  - `**Never changes**`: what this task must not touch, and why.
  - `**Outside the repository**`: GPU jobs, cluster submissions, pushes, pull requests and
    messages, with their cost, or `none`. The agent confirms each with the owner first.

  Paths are in backticks, from the repository root, and must exist; a `<placeholder>` part is
  allowed.
- **Inputs**: what the agent needs before starting, and what to ask the person for. Never ask
  for a token or password.
- **Steps**: `### 1.`, `### 2.`, ... in order. Each ends with a `**Check**:` line: the output,
  file or log line that shows the step worked. Scripts go in `scripts/`, long reference
  material in `references/`.
- **Exact commands or guidance.** A step that reproduces the paper gives the exact command and
  config, so results match. A step that depends on the reader's model, GPU or cluster says
  how to find the answer (what to inspect, what each setting means, how to check it) and does
  not ship a guessed command.
- **Returns**: what the agent reports at the end, with the evidence for each item and the
  decisions that are the owner's.
- **Pitfalls**: mistakes seen more than once, each with what to do instead.
- Links are relative to the skill's file and must resolve.
- A skill enters the repository after a fresh agent session has followed it to the end on a
  real task. Say in the pull request what that task was.
- Name it with a verb and its object (`deploy-stack`, `check-pr`).
- List the skill in `AGENTS.md` and in [`skills/README.md`](../../skills/README.md).

Agents find the instructions and the skills without setup. Codex and most other agents read
`AGENTS.md`; Claude Code reads `CLAUDE.md`, which holds the one line `@AGENTS.md`. Codex finds
skills through `.agents/skills` and Claude Code through `.claude/skills`, both links to
`skills/`. Write skills only in `skills/`.
