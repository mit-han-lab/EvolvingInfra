# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
# Sourced by the launch scripts in this directory. Sets:
#   EVOLVING_INFRA_URL    the agent gateway (default http://localhost:30800; the first host of `evolving-infra serve`)
#   EVOLVING_INFRA_MODEL  the served model name (default: the first model the stack lists)
#   TRANSLATOR_URL        the local Anthropic/Responses translator of translator.sh (default http://127.0.0.1:4000)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVOLVING_INFRA_URL="${EVOLVING_INFRA_URL:-http://localhost:30800}"
EVOLVING_INFRA_URL="${EVOLVING_INFRA_URL%/}"
TRANSLATOR_URL="${TRANSLATOR_URL:-http://127.0.0.1:${TRANSLATOR_PORT:-4000}}"
if [[ -z "${EVOLVING_INFRA_MODEL:-}" ]]; then
  EVOLVING_INFRA_MODEL=$(curl -sf --max-time 10 "$EVOLVING_INFRA_URL/v1/models" \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["data"][0]["id"])') \
    || { echo "no model list at $EVOLVING_INFRA_URL/v1/models: is the stack up? (or set EVOLVING_INFRA_MODEL)" >&2; exit 1; }
fi
# The translator's master key: random, written by translator.sh, readable by this user only.
TRANSLATOR_KEY_FILE="${TRANSLATOR_KEY_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/evolving-infra/translator-key}"
export EVOLVING_INFRA_URL EVOLVING_INFRA_MODEL TRANSLATOR_URL TRANSLATOR_KEY_FILE

need_translator() {
  curl -sf --max-time 5 "$TRANSLATOR_URL/health/liveliness" >/dev/null && [[ -r "$TRANSLATOR_KEY_FILE" ]] \
    || { echo "no translator at $TRANSLATOR_URL: start examples/agents/translator.sh first" >&2; exit 1; }
  TRANSLATOR_KEY="$(cat "$TRANSLATOR_KEY_FILE")"
}
