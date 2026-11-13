#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
# Pi on Evolving Infra. Pi speaks OpenAI chat completions, so it talks to the agent gateway directly.
#   bash examples/agents/pi.sh [pi arguments...]     e.g. bash examples/agents/pi.sh -p "Summarize README.md"
# The provider comes from pi-provider.ts (loaded with -e); nothing in ~/.pi changes.
set -euo pipefail
source "$(dirname "$0")/common.sh"
exec "${PI_BIN:-pi}" -e "$HERE/pi-provider.ts" --provider evolving-infra --model "$EVOLVING_INFRA_MODEL" "$@"
