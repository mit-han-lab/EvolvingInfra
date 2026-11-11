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
  [`pr-checklist`](.agents/skills/pr-checklist/SKILL.md) skill and address what it finds.

## 2. Repository map

| Path | Contents | Read first |
|---|---|---|
| `engine/` | Serving engine, derived from SGLang (`decoupled_engine`) | [`engine/README.md`](engine/README.md) |
| `router/core/` | Rust router (`sgl-router`): placement, late decode choice, role API | [`router/core/README.md`](router/core/README.md) |
| `router/frontend/`, `router/indexer/`, `router/client/` | Tokenizer service, KV directory, Python client | [`router/frontend/README.md`](router/frontend/README.md), [`router/indexer/README.md`](router/indexer/README.md) |
| `planner/` | Online P/D planner | [`planner/README.md`](planner/README.md) |
| `launch/` | `evolving-infra serve`: starts every component on one or more hosts | [`examples/gb300/README.md`](examples/gb300/README.md) |
| `examples/` | Model and experiment configs, the Slurm submit script | [`examples/gb300/README.md`](examples/gb300/README.md) |
| `benchmarks/` | Trie replay client, replay adapters, run checks, scoring | [`benchmarks/README.md`](benchmarks/README.md) |
| `deploy/modal/`, `docker/` | Benchmarks on Modal, the runtime image | their READMEs |
| `docs/` | User documentation; [`docs/paper.md`](docs/paper.md) maps the paper to the code | [`docs/README.md`](docs/README.md) |
| `tests/`, `tools/` | Python tests, one directory per component; repository tools | |
| `.agents/skills/` | Agent skills (see [Skills](#4-skills)) | |

## 3. Development workflow

Python 3.11 or newer. Three Python environments exist in the runtime image because their
dependencies conflict: the engine (system Python), `/opt/indexer-venv` (launcher, frontend, KV
directory, planner) and `/opt/trie-venv` (replay client).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[planner,test]"                              # launcher, planner, test tools
python -m pytest -q tests/launch tests/planner tests/engine tests/tools   # no GPU needed
(cd router/core && cargo test --locked)                       # router: unit and proxy tests
python3 -m venv .venv-trie && .venv-trie/bin/pip install -e 'benchmarks/trie[test]'
.venv-trie/bin/python -m pytest -q benchmarks/trie/tests      # replay client
```

GPU validation, smallest first:

```bash
evolving-infra serve --config examples/gb300/glm52-1node.yaml          # one node, 1P1D
python examples/gb300/submit.py --account A --partition P --image IMG  # nine nodes, scored
```

Tests: extend the nearest existing test file; test behavior through public entry points; one
behavior per test. For a bug, run the new test on the base commit first and show that it
fails for the reported reason.

Docs and comments: plain statements of what the code does. No hedging ("not tested yet",
"may change"), no history of how a file came to be.

Commits: imperative subject that says what changed for the user, body that says why.

## 4. Skills

Skills live in `.agents/skills/`; `.claude/skills/` holds symlinks to them. Edit the files in
`.agents/skills/`.

- [`pr-checklist`](.agents/skills/pr-checklist/SKILL.md): check a change and its description
  before a pull request or a re-review.

## 5. Rules the code does not show

- **Engine: read resolved settings, not raw arguments.** Startup resolution declares values
  (`declare_resolution`); the raw `ServerArgs` field keeps what the operator typed. Read
  `resolution_result(server_args, field)`. A raw read ignores every default the engine turned on.
- **Engine and router presets match.** `--symmetric-preset` (engine) and
  `--symmetric-pd-preset` (router) are both `paper` by default. With one side set to `none`
  the router refuses the worker.
- **Ports on one host do not overlap.** An engine binds two ports per GPU from
  `--disaggregation-bootstrap-port` and one per GPU from `--peer-kv-servicer-port`.
  `launch/layout.py` assigns them; change it there.
- **Benchmarks replay with `replay_context_mode=actual_output`**, as the paper's runs do: later
  prompts carry the model's own earlier outputs. Recorded outputs in the context change how
  many draft tokens speculative decoding accepts, so the results do not compare.
- **Compare runs on the same window and the same P/D split.** The planner changes the split
  during a run; a throughput difference over the whole window can be a difference in when it
  switched. `benchmarks/replay/score.py --skip-s --window-s` scores any window.
- **Hosts find themselves.** `evolving-infra serve` locates the host in `--hosts` by name. A
  scheduler's node index is not the position in a sorted host list.

Rules for editing this file and the skills:
[`docs/contributing/agent-instructions.md`](docs/contributing/agent-instructions.md).
