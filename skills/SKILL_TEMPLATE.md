---
name: <skill-name>
description: <What the skill does, in one sentence>. Use when <the requests it answers>. Not for <a nearby case>; use `<other-skill>`.
---

<!--
Template for a skill. Copy it to skills/<skill-name>/SKILL.md, replace every <...>, and delete
this comment. The rules are in docs/contributing/agent-instructions.md (Skills);
`python tools/check_agent_files.py` checks the format.

- name: the directory name, lowercase words joined by "-".
- Paths in backticks are from the repository root and must exist; a <placeholder> part is allowed.
  Markdown links are relative to this file (../../ is the repository root).
- Every section below is required except Pitfalls. Keep SKILL.md under 500 lines; scripts go in
  scripts/, long reference material in references/.
-->

# <Title: the task, as a verb phrase>

<One or two sentences: what the task produces, and what must already be running or exist.>

## Scope

- **Reads**: `<path/>`, `<path/>`: <what the agent learns there>. Or `repository`.
- **Writes**: `<path/>`: <what it creates or changes>. Or `nothing`. Anything else is out of
  scope: stop and say why.
- **Never changes**: `<path/>`: <why this task must not touch it>.
- **Outside the repository**: <GPU jobs, cluster submissions, pushes, pull requests, messages,
  with their cost>. Or `none`. Confirm each with the owner before doing it.

## Inputs

- <Input>: <where it comes from>. Ask the person when it is missing; do not guess it.
- Never ask for a token or password; the person logs in and gives a path to the credentials file.

## Steps

### 1. <Action>

<The exact command, when the step reproduces a documented result. Otherwise how to find the
right value: what to inspect, what each setting means.>

```bash
<command>
```

**Check**: <the output, file or log line that shows this step worked, and what to do if it did not>.

### 2. <Action>

<...>

**Check**: <...>.

## Returns

- <Item>: <the value or table to report, and the evidence for it (command, log line, file)>.
- The decisions that are the owner's.

## Pitfalls

- <A mistake seen more than once>: <what to do instead>.
