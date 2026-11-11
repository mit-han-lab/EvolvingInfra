#!/usr/bin/env bash
# Claude Code on Evolving Infra, through translator.sh (Anthropic Messages -> chat completions).
#   bash examples/agents/translator.sh &          # once
#   bash examples/agents/claude-code.sh [claude arguments...]
# Every model slot (main, background, sub-agents) is the served model. Your Claude login is not used
# and not changed: the environment below applies to this process only.
set -euo pipefail
source "$(dirname "$0")/common.sh"
need_translator
unset ANTHROPIC_API_KEY
export ANTHROPIC_BASE_URL="$TRANSLATOR_URL" ANTHROPIC_AUTH_TOKEN="$TRANSLATOR_KEY"
export ANTHROPIC_MODEL="$EVOLVING_INFRA_MODEL" ANTHROPIC_DEFAULT_OPUS_MODEL="$EVOLVING_INFRA_MODEL" \
  ANTHROPIC_DEFAULT_SONNET_MODEL="$EVOLVING_INFRA_MODEL" ANTHROPIC_DEFAULT_HAIKU_MODEL="$EVOLVING_INFRA_MODEL" \
  CLAUDE_CODE_SUBAGENT_MODEL="$EVOLVING_INFRA_MODEL" CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
exec "${CLAUDE_BIN:-claude}" --model "$EVOLVING_INFRA_MODEL" "$@"
