# Agent Instructions for Evolving Infra

These instructions apply to every change made with a coding agent. [`README.md`](README.md)
says what Evolving Infra is; [`docs/architecture.md`](docs/architecture.md) says how a request
moves through it.

## 1. Contribution policy

- **A person owns every pull request.** The submitter reads every changed line, can explain
  it, and answers the review. Name that person in the pull request.
- **One purpose per pull request.** Do not open a pull request for a lone typo or a style
  change; bundle such edits with substantive work.
- **Claims come with evidence.** Every claim in the description (a fix, a speedup, a new
  flag) has the command that shows it and the result, at the level it was checked:
  unit tests, one GPU node, or several nodes. Say what was not run.
- **A fix states its root cause.** Say what went wrong, how you found it, and why the change
  removes it.
- **Attach the session transcript**, scrubbed with
  [`tools/scrub_transcript.py`](tools/scrub_transcript.py). Never paste a token, password or
  key into a session; log in yourself and give the agent a path to the credentials file.
- **User-visible changes update the docs in the same pull request**: flags, defaults,
  environment variables, config keys, endpoints.
- Before opening a pull request or asking for another review, run the
  [`check-pr`](skills/check-pr/SKILL.md) skill and address what it finds.

## 2. Repository map

| Path | Contents | Read first |
|---|---|---|
| `evolving_infra/engine/` | Serving engine, derived from SGLang (`decoupled_engine`) | [`evolving_infra/engine/README.md`](evolving_infra/engine/README.md) |
| `evolving_infra/router/core/` | Rust router (`sgl-router`): placement, late decode choice, role API | [`evolving_infra/router/core/README.md`](evolving_infra/router/core/README.md) |
| `evolving_infra/router/frontend/`, `evolving_infra/router/indexer/`, `evolving_infra/router/client/` | Tokenizer service, KV directory, Python client | [`evolving_infra/router/frontend/README.md`](evolving_infra/router/frontend/README.md), [`evolving_infra/router/indexer/README.md`](evolving_infra/router/indexer/README.md) |
| `evolving_infra/planner/` | Online P/D planner | [`evolving_infra/planner/README.md`](evolving_infra/planner/README.md) |
| `evolving_infra/launch/` | `evolving-infra serve`: starts every component on one or more hosts | [`examples/gb300/README.md`](examples/gb300/README.md) |
| `examples/` | A config per hardware and model (`gb300/`, `h200/`) and running one under Slurm (`slurm/`); the Modal H200 benchmark (`modal/`), whose settings are constants in `modal/stack.py` | [`examples/README.md`](examples/README.md) |
| `benchmarks/` | Trie replay client, replay adapters, run checks, scoring | [`benchmarks/README.md`](benchmarks/README.md) |
| `docker/` | The runtime image | [`docker/README.md`](docker/README.md) |
| `docs/` | User documentation; [`docs/paper.md`](docs/paper.md) maps the paper to the code | [`docs/README.md`](docs/README.md) |
| `tests/`, `tools/` | Python tests, one directory per component; repository tools | |
| `skills/` | Agent skills (see [Skills](#4-skills)) | |

## 3. Development workflow

Python 3.11 or newer. Three Python environments exist in the runtime image because their
dependencies conflict: the engine (system Python), `/opt/indexer-venv` (launcher, frontend, KV
directory, planner) and `/opt/trie-venv` (replay client).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[client,frontend,indexer,planner,test]"      # launcher, every Python component, test tools
python -m pytest -q tests                                     # no GPU; tests that need decoupled-engine skip
(cd evolving_infra/router/core && cargo test --locked)   # router: unit and proxy tests
python3 -m venv .venv-trie && .venv-trie/bin/pip install -e 'benchmarks/trie[test]'
.venv-trie/bin/python -m pytest -q benchmarks/trie/tests      # replay client
```

GPU validation, smallest first:

```bash
evolving-infra serve --config examples/gb300/glm52-1node.yaml          # one node, 1P1D
python examples/slurm/submit.py --account A --partition P --image IMG   # nine nodes, scored
```

Tests: extend the nearest existing test file; test behavior through public entry points; one
behavior per test. For a bug, run the new test on the base commit first and show that it
fails for the reported reason.

Docs and comments: plain statements of what the code does. No hedging ("not tested yet",
"may change"), no history of how a file came to be.

File headers: a new source file starts with `SPDX-FileCopyrightText: Copyright (c) 2026 The
evolving-infra Authors` and `SPDX-License-Identifier: MIT` (Apache-2.0 under
`evolving_infra/engine/` and `evolving_infra/router/core/`). Files copied from another project
keep its header ([`NOTICE`](NOTICE)).

Commits: imperative subject that says what changed for the user, body that says why.

## 4. Skills

Skills live in `skills/`. To use one, read its `SKILL.md` and follow it.

| Skill | Scope | Use it to |
|---|---|---|
| [`check-pr`](skills/check-pr/SKILL.md) | repository | check a change and its description before a pull request or a re-review |
| [`deploy-stack`](skills/deploy-stack/SKILL.md) | `evolving_infra/launch/`, `examples/`, `docker/` | install and start the stack on one host or several; serve a model or GPU without an example |
| [`connect-agent`](skills/connect-agent/SKILL.md) | outside the repository (the agent's code) | connect an agent or agent framework to a running stack: identity, prepared path, ending branches |
| [`connect-harness`](skills/connect-harness/SKILL.md) | `examples/agents/`, read-only | run Pi, Claude Code or Codex on a running stack with the launch scripts |
| [`add-harness`](skills/add-harness/SKILL.md) | `examples/agents/` | add a launch script for another agent harness and check it with one prompt and one tool call |
| [`run-benchmark`](skills/run-benchmark/SKILL.md) | `examples/`, `benchmarks/` | run the paper's replay benchmark, score it and compare it with the reference |
| [`diagnose-run`](skills/diagnose-run/SKILL.md) | repository; writes the fix once the cause is named | find why a run did not start, failed a request, or scored below the reference |

Each skill follows [`skills/SKILL_TEMPLATE.md`](skills/SKILL_TEMPLATE.md): its scope (what it
reads, writes, never changes, and does outside the repository), its inputs, numbered steps that
each end with a check, and what it returns. Commands that reproduce the paper are exact; for
anything that depends on your model, GPU or cluster, the skill says how to work it out.

## 5. Component guides

Read the guide of a directory before changing code in it. If the guide conflicts with the
requested change, stop and say so.

- [`evolving_infra/engine/AGENTS.md`](evolving_infra/engine/AGENTS.md): resolved settings, where defaults live, ports, TP ranks.
- [`evolving_infra/router/core/AGENTS.md`](evolving_infra/router/core/AGENTS.md): the preset, the test helper, which routes
  late decode choice and role control accept.
- [`evolving_infra/planner/AGENTS.md`](evolving_infra/planner/AGENTS.md): launch parameters, the role API, what stops the planner.
- [`evolving_infra/launch/AGENTS.md`](evolving_infra/launch/AGENTS.md): interpreters, port layout, hardware detection, hosts.
- [`benchmarks/AGENTS.md`](benchmarks/AGENTS.md): replay settings, scoring windows, comparing runs;
  one guide per workload under it.

Two rules cross components:

- **Engine and router presets match.** `--symmetric-preset` (engine) and
  `--symmetric-pd-preset` (router) are both `paper` by default; set both or neither to `none`.
- **A performance number names its window and its P/D split.** The planner changes the split
  during a run, so a difference over a whole run can be a difference in when it switched.

Rules for editing this file and the skills:
[`docs/contributing/agent-instructions.md`](docs/contributing/agent-instructions.md).
