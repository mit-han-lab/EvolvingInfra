# Coding agents on Evolving Infra

Run Pi, Claude Code or Codex against a running Evolving Infra stack. The harnesses stay where
they are installed; these scripts only point them at the stack, for one process, without
changing their own settings.

| Harness | Speaks | Path to the router | Launch |
|---|---|---|---|
| [Pi](https://pi.dev) | OpenAI chat completions | direct | [`pi.sh`](pi.sh) (provider in [`pi-provider.ts`](pi-provider.ts)) |
| [Claude Code](https://code.claude.com) | Anthropic Messages | [`translator.sh`](translator.sh) | [`claude-code.sh`](claude-code.sh) |
| [Codex](https://github.com/openai/codex) | OpenAI Responses | [`translator.sh`](translator.sh) | [`codex.sh`](codex.sh) |

The router serves OpenAI chat completions only. `translator.sh` runs a LiteLLM proxy on
`127.0.0.1:4000` that turns Anthropic Messages and OpenAI Responses requests into chat
completions; it writes a random key to `~/.config/evolving-infra/translator-key` (mode 600), which
the launch scripts read.

## Run

1. Serve a model with the config of this directory: it adds the tool-call and reasoning parsers,
   so the harness gets OpenAI `tool_calls`, and turns late decode choice off (see Limits). Start the stack
   without `--planner` (see Limits):

   ```bash
   evolving-infra serve --config examples/agents/glm52-1node.yaml     # GB300, GLM-5.2
   ```

   [`dsv4-flash.yaml`](dsv4-flash.yaml) (H200) and [`kimi-k26.yaml`](kimi-k26.yaml) (GB300) do the
   same for the other models.
2. On the machine where you run the harness:

   ```bash
   export EVOLVING_INFRA_URL=http://<first host>:30900   # default http://localhost:30900
   bash examples/agents/pi.sh                             # Pi: no translator needed
   bash examples/agents/translator.sh &                   # Claude Code and Codex: once
   bash examples/agents/claude-code.sh
   bash examples/agents/codex.sh
   ```

   Arguments go to the harness: `pi.sh -p "..."`, `claude-code.sh -p "..."`,
   `codex.sh exec "..."`. The model is the first one the router lists; set
   `EVOLVING_INFRA_MODEL` to choose another.

The [`connect-harness`](../../skills/connect-harness/SKILL.md) skill does the same steps with
checks, for a coding agent. For another harness, [`add-harness`](../../skills/add-harness/SKILL.md)
writes its launch script.

## Limits

- Harness requests carry no agent identity (`x-agent-context`): a harness cannot number its own
  requests per branch. They are served on the plain path with prefix caching, but without KV
  session retention for their branches. To keep the KV of live branches, connect the agent's code
  with [`connect-agent`](../../skills/connect-agent/SKILL.md).
- A stack started with `--planner` (KV retention on) refuses plain requests, so harnesses need a
  stack without `--planner`.
- The configs here turn late decode choice off on engines and router: it serves `/v1/completions`
  only, and harnesses send chat completions (a stack with the `paper` preset answers them with
  HTTP 501 `late_d_route_unsupported`).
- Requests with tools are placed without the router's prefix lookup (the router does not render
  tool schemas), so placement does not follow cached prefixes for them.
- Claude Code and Codex do not know the served model: Claude Code assumes a 200K-token context
  window and Codex uses fallback model metadata.
