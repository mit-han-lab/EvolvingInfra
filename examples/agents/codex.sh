#!/usr/bin/env bash
# Codex on Evolving Infra, through translator.sh (OpenAI Responses -> chat completions).
#   bash examples/agents/translator.sh &          # once
#   bash examples/agents/codex.sh [codex arguments...]   e.g. bash examples/agents/codex.sh exec "Summarize README.md"
# The provider is given with -c on the command line; ~/.codex/config.toml does not change.
set -euo pipefail
source "$(dirname "$0")/common.sh"
need_translator
export EVOLVING_INFRA_API_KEY="$TRANSLATOR_KEY"
exec "${CODEX_BIN:-codex}" \
  -c model_provider=evolving-infra \
  -c "model_providers.evolving-infra={name=\"Evolving Infra\", base_url=\"$TRANSLATOR_URL/v1\", env_key=\"EVOLVING_INFRA_API_KEY\", wire_api=\"responses\"}" \
  -m "$EVOLVING_INFRA_MODEL" "$@"
