# Coding agents on Evolving Infra

Run Pi, Claude Code or Codex against a running Evolving Infra stack. The harnesses stay where
they are installed; these scripts only point them at the stack, for one process, without
changing their own settings.

| Harness | Speaks | Path to the gateway | Launch |
|---|---|---|---|
| [Pi](https://pi.dev) | OpenAI chat completions | direct | [`pi.sh`](pi.sh) (provider in [`pi-provider.ts`](pi-provider.ts)) |
| [Claude Code](https://code.claude.com) | Anthropic Messages | [`translator.sh`](translator.sh) | [`claude-code.sh`](claude-code.sh) |
| [Codex](https://github.com/openai/codex) | OpenAI Responses | [`translator.sh`](translator.sh) | [`codex.sh`](codex.sh) |

Harnesses talk to the agent gateway (port 30800), which `evolving-infra serve` starts next to the
router when the config has a `gateway` section. The configs here bind it to every interface
(`host: 0.0.0.0`); without `host` it listens on 127.0.0.1 only. A harness sends no agent identity; the gateway infers it from the messages (a request
that starts with an earlier request's messages continues that branch; any other starts a new
agent), sends the request to the router with `x-agent-context`, and ends a branch after 600 s
without a request. The router then keeps the KV of each live conversation for its next turn.

The gateway serves OpenAI chat completions. `translator.sh` runs a LiteLLM proxy on
`127.0.0.1:4000` that turns Anthropic Messages and OpenAI Responses requests into chat
completions for the gateway; it writes a random key to `~/.config/evolving-infra/translator-key` (mode 600), which
the launch scripts read.

## Run

1. Serve a model with the config of this directory: it adds the tool-call and reasoning parsers,
   so the harness gets OpenAI `tool_calls`, and places requests with tools on their cached prefix. Start the stack
   without `--planner` (see Limits):

   ```bash
   evolving-infra serve --config examples/agents/glm52-1node.yaml     # GB300, GLM-5.2
   ```

   [`dsv4-flash.yaml`](dsv4-flash.yaml) (H200) and [`kimi-k26.yaml`](kimi-k26.yaml) (GB300) do the
   same for the other models.
2. On the machine where you run the harness:

   ```bash
   export EVOLVING_INFRA_URL=http://<first host>:30800   # the gateway; default http://localhost:30800
   bash examples/agents/pi.sh                             # Pi: no translator needed
   bash examples/agents/translator.sh &                   # Claude Code and Codex: once
   bash examples/agents/claude-code.sh
   bash examples/agents/codex.sh
   ```

   Arguments go to the harness: `pi.sh -p "..."`, `claude-code.sh -p "..."`,
   `codex.sh exec "..."`. The model is the first one the stack lists; set
   `EVOLVING_INFRA_MODEL` to choose another.

The [`connect-harness`](../../skills/connect-harness/SKILL.md) skill does the same steps with
checks, for a coding agent. For another harness, [`add-harness`](../../skills/add-harness/SKILL.md)
writes its launch script.

## Limits

- The gateway's identities come from the messages. A harness that rewrites earlier messages
  between turns (compaction) starts a new agent; the old branch ends after its idle time. Code that
  knows its agents and branches gives them exactly with
  [`connect-agent`](../../skills/connect-agent/SKILL.md).
- A harness's sub-agent (a Claude Code Task, a Codex sub-agent) is a conversation of its own, so
  the gateway gives it its own agent, not a branch of its parent. A branch of an agent is a
  concurrent continuation or a fork from an earlier turn of the same conversation.
- The gateway keeps its identities in memory. A restarted gateway sends no END for the branches
  it knew, and the router keeps their sessions until it restarts too.
- KV session retention needs a FULL-only cache tree: DeepSeek-V4-Flash (FULL+SWA) runs without it,
  and the gateway then sends no session END.
- The gateway sends chat completions on the plain path. A stack started with `--planner` (KV
  retention on) refuses plain requests, and the planner learns only from the prepared path, so
  harnesses need a stack without `--planner`.
- The router's chat encoder renders messages only. The configs here pass `--chat-tools-tokenize`,
  so for a request with tools the router asks an engine's `/v1/tokenize` for the token ids and
  places the request on them; this adds one tokenization per request. Without the flag such a
  request matches no cached prefix.
- Claude Code and Codex do not know the served model: Claude Code assumes a 200K-token context
  window and Codex uses fallback model metadata.
