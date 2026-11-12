---
name: add-harness
description: Add a launch script that runs another agent harness (OpenCode, Aider, Cline, Goose, an in-house CLI, ...) on Evolving Infra, next to the Pi, Claude Code and Codex scripts in examples/agents. Use when asked to connect an agent harness that examples/agents has no script for. Not for the three harnesses that have one; use `connect-harness`. Not for changing an agent's own code to send agent identities; use `connect-agent`.
---

# Add a harness

Write `examples/agents/<harness>.sh`, which starts the harness with Evolving Infra as its model,
and check it with one prompt and one tool call. Needs a running stack served with an
`examples/agents/<model>.yaml` config ([`connect-harness`](../connect-harness/SKILL.md), step 1)
and the harness installed.

## Scope

- **Reads**: `examples/agents/`: the existing launch scripts, `examples/agents/common.sh` and the translator; the
  harness's own documentation for custom model providers.
- **Writes**: `examples/agents/<harness>.sh` (and a provider file beside it if the harness needs
  one), and a row in the table of `examples/agents/README.md`.
- **Never changes**: `evolving_infra/`: the router stays as it is; the harness's settings in the
  home directory: the script passes everything per process.
- **Outside the repository**: runs the harness against the stack; may start the translator
  (`examples/agents/translator.sh`). Confirm with the owner first.

## Inputs

- The harness, its version and its documentation for custom providers or base URLs. Ask for the
  documentation link when it is not clear which API the harness sends.
- The gateway URL of the running stack (`gateway_url` in `stack.json`, port 30800).
- Never ask for a token or password: the stack needs none, and the translator's key is local.

## Steps

### 1. Find which API the harness sends

Read the harness's provider documentation and find:

| The harness sends | Path | Example |
|---|---|---|
| OpenAI chat completions (`/v1/chat/completions`) | to the gateway directly: `$EVOLVING_INFRA_URL/v1` | `examples/agents/pi.sh` |
| Anthropic Messages (`/v1/messages`) | through the translator: `$TRANSLATOR_URL` | `examples/agents/claude-code.sh` |
| OpenAI Responses (`/v1/responses`) | through the translator: `$TRANSLATOR_URL/v1` | `examples/agents/codex.sh` |

The gateway serves chat completions only (`evolving_infra/router/gateway/`); the translator turns the other
two into chat completions. Another API needs a translator change: stop and ask the owner.

**Check**: the documentation names one of the three APIs and how to set its base URL.

### 2. Find how to set the provider for one process

In order of preference: command-line flags (Codex `-c`), a provider file loaded by a flag (Pi
`-e`), environment variables (Claude Code `ANTHROPIC_*`), a config directory chosen by an
environment variable. Do not write into the harness's files in the home directory.

Set: the base URL from step 1, the model name (`$EVOLVING_INFRA_MODEL`), and an API key. Directly
to the gateway any key works (`unused`); through the translator the key is `$TRANSLATOR_KEY`, which
`need_translator` reads. If the harness has separate model slots (a small model for background
work, a model for sub-agents), set every slot to the served model.

**Check**: the harness's documentation shows the setting applies without editing its files.

### 3. Write the script

Copy the closest example and keep its shape:

```bash
#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
need_translator            # only through the translator
exec "${<HARNESS>_BIN:-<harness>}" <provider flags> "$@"
```

`common.sh` sets `EVOLVING_INFRA_URL`, `EVOLVING_INFRA_MODEL` and `TRANSLATOR_URL`; arguments after
the script name go to the harness.

**Check**: `bash -n examples/agents/<harness>.sh` passes.

### 4. Run one prompt and one tool call

```bash
export EVOLVING_INFRA_URL=<gateway url>
bash examples/agents/<harness>.sh <the harness's one-shot flag> "Reply with the word ready"
bash examples/agents/<harness>.sh <the harness's one-shot flag> "List the files in this directory"
```

**Check**: the first prints the model's reply; the second runs the harness's file or shell tool
and answers from its output. A tool call printed as text means the stack was not served with
an `examples/agents/<model>.yaml` config, which has the parsers.

### 5. Document it

Add a row to the table of `examples/agents/README.md`: the harness, the API it speaks, the path
(direct or translator) and the script.

**Check**: `python tools/check_agent_files.py` passes.

## Returns

- The script, the harness version it was run with, and the output of step 4.
- What the harness does that the script does not cover (several model slots, a model list it
  checks, a context window it assumes), as decisions for the owner.

## Pitfalls

- A harness that also sends a model-list or token-count request (`/v1/models`,
  `/v1/messages/count_tokens`) may stop when it fails: run it through the translator, which
  serves both.
- A harness started from inside another agent's session can inherit that session's credentials
  through environment variables; test it from a plain terminal.
