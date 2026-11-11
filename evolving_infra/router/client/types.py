"""Shared Python request contracts; no node state or duplicate load accounting.

``RequestIdentity.request_id`` is a numeric step within one subagent;
``RequestIdentity.engine_rid`` is the globally scoped string sent to engines.
Transport and scheduler protocols let callers supply clients without depending
on a concrete SGLang implementation. Actual engine adapters live in Rust.
"""
from collections.abc import AsyncIterator
from typing import Any, Mapping, Protocol

from evolving_infra.router.client.agent.identity import RequestIdentity
from evolving_infra.router.client.agent.request import Request

__all__ = ["BackendClient", "Request", "RequestIdentity", "RequestScheduler"]


class RequestScheduler(Protocol):
    """The operation Agent needs from its scheduler client."""

    async def end_trajectory(self, agent_id: str, subagent_id: int) -> dict[str, Any]:
        """Input: branch identity. Output: async end acceptance; does not close transport."""
        ...

    async def submit(self, request: Request, *, chat: bool = False) -> dict[str, Any]:
        """Submit an identified model request through a scheduler implementation.

        Args:
            request: Model payload and stable request identity.
            chat: Select the chat-completions interface when True.

        Returns:
            The decoded JSON model response.
        """
        ...


class BackendClient(Protocol):
    """Transport interface used by the Python Scheduler facade."""

    async def end_trajectory(
        self, endpoint: str, agent_id: str, subagent_id: int
    ) -> dict[str, Any]:
        """Input: runtime and branch identity. Output: async end acceptance."""
        ...

    async def get_backend_loads(self, endpoint: str) -> dict[str, Any]:
        """Read the external runtime's authoritative merged workload.

        Args:
            endpoint: Rust scheduler HTTP(S) base URL.

        Returns:
            The JSON workload snapshot.
        """
        ...

    async def generate(
        self, endpoint: str, request: Request, *, chat: bool = False
    ) -> dict[str, Any]:
        """Send an identified model request through the configured runtime.

        Args:
            endpoint: Rust scheduler HTTP(S) base URL.
            request: Model payload and stable request identity.
            chat: Select the chat-completions interface when True.

        Returns:
            The decoded JSON model response.
        """
        ...

    def stream(
        self, endpoint: str, request: Request, *, chat: bool = False
    ) -> AsyncIterator[bytes]:
        """Send an identified model request through the configured runtime.

        Args:
            endpoint: Rust scheduler HTTP(S) base URL.
            request: Model payload and stable request identity.
            chat: Select the chat-completions interface when True.

        Returns:
            An async iterator of raw response bytes.
        """
        ...

    async def generate_prepared(
        self,
        endpoint: str,
        tokenizer_endpoint: str,
        prepared: Mapping[str, Any],
        *,
        identity: RequestIdentity | None = None,
    ) -> dict[str, Any]:
        """Schedule a compact prepared-input descriptor and forward its stored input.

        Args:
            endpoint: Rust scheduler HTTP(S) base URL.
            tokenizer_endpoint: Service retaining the prepared input.
            prepared: Descriptor returned by that service's /prepare endpoint.

        Returns:
            The decoded JSON decode response.
        """
        ...

    def stream_prepared(
        self,
        endpoint: str,
        tokenizer_endpoint: str,
        prepared: Mapping[str, Any],
        *,
        identity: RequestIdentity | None = None,
    ) -> AsyncIterator[bytes]:
        """Schedule a compact prepared-input descriptor and forward its stored input.

        Args:
            endpoint: Rust scheduler HTTP(S) base URL.
            tokenizer_endpoint: Service retaining the prepared input.
            prepared: Descriptor returned by that service's /prepare endpoint.

        Returns:
            An async iterator of decode-response bytes.
        """
        ...

    async def close(self) -> None:
        """Release transport resources owned by the implementation.

        Args:
            None.

        Returns:
            None; caller-owned resources follow the implementation's ownership contract.
        """
        ...
