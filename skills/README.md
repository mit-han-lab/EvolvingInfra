# Skills

A skill tells a coding agent (Claude Code, Codex, ...) how to do one task in this repository,
step by step: what it may read and change, what to ask for, how to check each step, and what to
report. To use one, ask the agent for the task and point it at the skill, for example "deploy
GLM-5.2 on these hosts, following `skills/deploy-stack/SKILL.md`".

| Skill | Use it to | Changes in the repository | Outside the repository |
|---|---|---|---|
| [`deploy-stack`](deploy-stack/SKILL.md) | install and start Evolving Infra on one host or several, including a model or GPU without an example | configs under `examples/` | starts GPU processes or a Slurm job |
| [`connect-agent`](connect-agent/SKILL.md) | connect an agent or agent framework to a running stack: identity on every request, the prepared path under the planner, ending branches | nothing | edits the agent's code |
| [`run-benchmark`](run-benchmark/SKILL.md) | run a paper benchmark (Claude Code, Putnam, OSWorld), score it and compare it with the reference | nothing | a Slurm job on nine GPU nodes for about two hours |
| [`diagnose-run`](diagnose-run/SKILL.md) | find why a run did not start, failed a request, or scored below the reference | the fix, once the cause is named | none |
| [`check-pr`](check-pr/SKILL.md) | check a change against the contribution rules and fill the pull request description, before opening or re-reviewing a pull request | the fixes for what fails | opens the pull request, publishes the transcript |

They follow one another: `deploy-stack` starts the stack, `connect-agent` sends an agent's
requests to it, `run-benchmark` measures it, `diagnose-run` explains a failure or a gap, and
`check-pr` comes before the change goes up for review.

Agents find these skills on their own: `.agents/skills` (Codex) and `.claude/skills` (Claude
Code) link to this directory, and [`AGENTS.md`](../AGENTS.md) lists them for every other agent.

## What a skill directory holds

```
skills/<skill-name>/
  SKILL.md       the skill: Scope, Inputs, Steps, Returns (required)
  scripts/       scripts the steps run (optional)
  references/    long reference material the steps link to (optional)
```

## Adding a skill

1. Name it with a verb and its object, lowercase words joined by `-`: `deploy-stack`, not
   `serving`.
2. Copy [`SKILL_TEMPLATE.md`](SKILL_TEMPLATE.md) to `skills/<skill-name>/SKILL.md` and fill it.
   The rules are in
   [`docs/contributing/agent-instructions.md`](../docs/contributing/agent-instructions.md#skills).
3. Add a row to the table above and to the skills table of [`AGENTS.md`](../AGENTS.md).
4. Run `python tools/check_agent_files.py`; it checks the format, the paths and the links.
5. Have a fresh agent session follow the skill to the end on a real task, and say in the pull
   request what that task was.
