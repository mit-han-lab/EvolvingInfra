"""A model step and its stable identity, shared by all backend clients."""
from dataclasses import dataclass, field
from typing import Any, Mapping

from .identity import RequestIdentity


@dataclass(frozen=True)
class Request:
    identity: RequestIdentity
    payload: Mapping[str, Any] = field(repr=False)
    parent_subagent_id: int | None = None

    def wire_payload(self, *, stream: bool) -> dict[str, Any]:
        """Copy the payload and attach its stable engine ID and streaming mode.

        Agent identity does not alter cache salt, prompt tokens, or affinity.

        Args:
            stream: Whether the wire request asks for streaming output.

        Returns:
            A new request-body dictionary; the original payload is unchanged.

        Raises:
            ValueError: A supplied rid conflicts with this request's identity.
        """
        body = dict(self.payload)
        expected = self.identity.engine_rid
        if body.get("rid") not in (None, expected):
            raise ValueError("payload rid conflicts with the three-part request identity")
        body["rid"] = expected
        body["stream"] = stream
        # Agent identity does not change cache_salt, prompt tokens or affinity.
        return body
