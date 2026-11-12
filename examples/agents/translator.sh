#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
# The translator that Claude Code and Codex need: the gateway serves OpenAI chat completions only,
# Claude Code sends Anthropic Messages (/v1/messages) and Codex sends OpenAI Responses (/v1/responses).
# A LiteLLM proxy on this machine turns both into chat completions for the gateway. Leave it running.
#   bash examples/agents/translator.sh            # listens on 127.0.0.1:${TRANSLATOR_PORT:-4000}
set -euo pipefail
source "$(dirname "$0")/common.sh"
port="${TRANSLATOR_PORT:-4000}"
config="$(mktemp "${TMPDIR:-/tmp}/evolving-infra-translator.XXXXXX.yaml")"
trap 'rm -f "$config"' EXIT
chmod 600 "$config"
mkdir -p "$(dirname "$TRANSLATOR_KEY_FILE")"
( umask 077; python3 -c 'import secrets; print("sk-" + secrets.token_urlsafe(32))' > "$TRANSLATOR_KEY_FILE" )
cat > "$config" <<YAML
model_list:
  - model_name: $EVOLVING_INFRA_MODEL
    litellm_params:
      model: openai/$EVOLVING_INFRA_MODEL
      api_base: $EVOLVING_INFRA_URL/v1
      api_key: unused
      use_chat_completions_api: true   # Responses requests become chat completions too
litellm_settings:
  drop_params: true
general_settings:
  master_key: $(cat "$TRANSLATOR_KEY_FILE")
YAML
echo "translator on 127.0.0.1:$port -> $EVOLVING_INFRA_URL ($EVOLVING_INFRA_MODEL)" >&2
uvx --from "litellm[proxy]==${LITELLM_VERSION:-1.106.4}" litellm --config "$config" --host 127.0.0.1 --port "$port"
