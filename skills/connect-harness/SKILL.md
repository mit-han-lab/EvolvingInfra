---
name: connect-harness
description: Run a coding-agent harness (Pi, Claude Code or Codex) on a running Evolving Infra stack with the launch scripts of examples/agents. Use when asked to use Evolving Infra as the model of Pi, Claude Code, Codex or another off-the-shelf agent harness. Not for changing an agent's own code to send agent identities; use `connect-agent`.
---

# Connect a coding-agent harness

Point Pi, Claude Code or Codex at a running stack, so the harness's model requests are served
by Evolving Infra. Needs a running stack ([`deploy-stack`](../deploy-stack/SKILL.md)) and the
harness installed on the machine where it runs.

## Scope

- **Reads**: `examples/agents/`: the launch scripts, the translator and the configs with parsers;
  `docs/api-reference.md`: the router's routes.
- **Writes**: nothing in this repository. A harness this skill has no script for gets a script in
  `examples/agents/` only when the owner asks for one.
- **Never changes**: `evolving_infra/`: connecting a harness needs no change to the server. The
  harness's own settings (its directory in the home directory): the scripts pass everything per
  process.
- **Outside the repository**: starts a LiteLLM proxy on 127.0.0.1:4000 for Claude Code and Codex,
  and writes its key to ~/.config/evolving-infra/translator-key. Confirm with the owner first.

## Inputs

- The router URL: `router_url` in the `stack.json` of the work directory of `evolving-infra serve`
  on the first host, reachable from the machine that runs the harness.
- Which harness, and whether it is installed (`pi --version`, `claude --version`,
  `codex --version`). Do not install a harness without asking.
- Never ask for a token or password: the scripts use no credential of the person's.

## Steps

### 1. Check the stack serves tool calls

The stack must run with the tool-call and reasoning parsers of its model
(`examples/agents/<model>.yaml`) and without `--planner`. Read the engine command in the stack's
work directory and check `--tool-call-parser` is set. Then send one request with a tool:

```bash
curl -s $EVOLVING_INFRA_URL/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "<model>",
  "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "parameters":
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}'
```

**Check**: the reply has `choices[0].message.tool_calls` with `get_weather`. A tool call inside
`content` means the parser is missing: restart the stack with `examples/agents/<model>.yaml`. An
HTTP 400 about prepared lifecycle callbacks means the stack runs with `--planner`.

### 2. Start the translator (Claude Code and Codex only)

Pi speaks chat completions and needs no translator: go to step 3.

```bash
EVOLVING_INFRA_URL=<router url> bash examples/agents/translator.sh &
```

**Check**: `curl -s 127.0.0.1:4000/health/liveliness` answers `"I'm alive!"`. Start only one
translator: a second one rewrites the key file and the first one then refuses every request.

### 3. Run one prompt

```bash
export EVOLVING_INFRA_URL=<router url>
bash examples/agents/pi.sh -p "Reply with the word ready"
bash examples/agents/claude-code.sh -p "Reply with the word ready"
bash examples/agents/codex.sh exec --skip-git-repo-check "Reply with the word ready"
```

**Check**: the harness prints the model's reply. The engine log of the stack
(`<work dir>/logs/engine-w<i>.log`) shows the request. On Claude Code, `No connected db` or an
authentication error means the translator's key does not match: restart the translator.

### 4. Run one tool call

Ask for a task that needs a tool, for example "List the files in this directory".

**Check**: the harness runs its tool (it shows the command or the file list) and then answers.
Text that looks like a tool call but does not run means the parser is missing (step 1).

## Returns

- The harness, its version, the router URL and the served model.
- The reply of step 3 and the tool run of step 4.
- The limits that apply ([examples/agents](../../examples/agents/README.md#limits)): no agent
  identity, no `--planner`; decisions about them are the owner's.

## Pitfalls

- Run inside another Claude Code session, `claude-code.sh` inherits that session's login
  through `CLAUDE_CODE_*` variables and sends its key to the translator. Run it from a plain
  terminal, or with `env -i HOME=$HOME PATH=$PATH`.
- `pkill -f litellm` also matches the shell that runs it; stop the translator by its PID.
