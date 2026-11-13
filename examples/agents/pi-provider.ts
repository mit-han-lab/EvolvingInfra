// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: MIT
// Pi provider for the Evolving Infra agent gateway (OpenAI chat completions), loaded by pi.sh with
// `pi -e`. The gateway and the model come from EVOLVING_INFRA_URL and EVOLVING_INFRA_MODEL.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  const url = (process.env.EVOLVING_INFRA_URL ?? "http://localhost:30800").replace(/\/$/, "");
  const model = process.env.EVOLVING_INFRA_MODEL ?? "glm-5.2";
  pi.registerProvider("evolving-infra", {
    name: "Evolving Infra",
    baseUrl: `${url}/v1`,
    apiKey: "unused",
    api: "openai-completions",
    models: [
      {
        type: "chat",
        id: model,
        name: `${model} (Evolving Infra)`,
        reasoning: true,
        input: ["text"],
        contextWindow: Number(process.env.EVOLVING_INFRA_CONTEXT ?? 1048576),
        maxTokens: Number(process.env.EVOLVING_INFRA_MAX_TOKENS ?? 32768),
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      },
    ],
  });
}
