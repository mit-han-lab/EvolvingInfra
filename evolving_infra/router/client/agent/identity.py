# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Stable problem / agent-branch / model-step identity, independent of KV salt."""
from dataclasses import dataclass
import hashlib
import json
from typing import Any

CONTEXT_HEADER = "x-agent-context"


@dataclass(frozen=True)
class RequestIdentity:
    agent_id: str
    subagent_id: int
    request_id: int

    def __post_init__(self):
        """Validate the problem identifier and numeric branch/step ranges.

        Args:
            None.

        Returns:
            None; leaves a valid immutable identity unchanged.

        Raises:
            ValueError: agent_id is empty/overlong or a numeric ID is not uint64.
        """
        if not isinstance(self.agent_id, str) or not self.agent_id or len(self.agent_id.encode("utf-8")) > 512:
            raise ValueError("agent_id must contain 1..512 UTF-8 bytes")
        for name in ("subagent_id", "request_id"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**64:
                raise ValueError(f"{name} must be an unsigned 64-bit integer")

    def as_dict(self) -> dict[str, Any]:
        """Represent the three-part identity using its public field names.

        Args:
            None.

        Returns:
            A new dictionary with agent_id, subagent_id, and request_id.
        """
        return dict(agent_id=self.agent_id, subagent_id=self.subagent_id, request_id=self.request_id)

    @property
    def engine_rid(self) -> str:
        # Binary framing matches evolving_infra/router/core/src/agent_identity.rs. Delimiters inside
        # agent_id cannot collide, and Unicode normalization is never inferred.
        """Derive the engine request ID from unambiguous binary identity framing.

        The framing matches the Rust runtime. It does not normalize Unicode or
        change KV cache salt.

        Args:
            None.

        Returns:
            A agent_-prefixed SHA-256 hexadecimal identifier.
        """
        agent = self.agent_id.encode("utf-8")
        data = (len(agent).to_bytes(8, "big") + agent
                + self.subagent_id.to_bytes(8, "big") + self.request_id.to_bytes(8, "big"))
        return "agent_" + hashlib.sha256(data).hexdigest()

    def headers(self) -> dict[str, str]:
        """Build HTTP identity metadata for the router and inference engine.

        Args:
            None.

        Returns:
            A dictionary containing the JSON context and engine request-ID headers.
        """
        return {CONTEXT_HEADER: json.dumps(self.as_dict(), separators=(",", ":"), ensure_ascii=True),
                "x-request-id": self.engine_rid}
