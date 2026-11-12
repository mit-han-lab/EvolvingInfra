# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Python frontend to the single Rust scheduler that owns node placement."""
from collections.abc import AsyncIterator
from typing import Any, Mapping

from evolving_infra.router.client.backend.backend import Backend, endpoint_url
from evolving_infra.router.client.types import BackendClient, Request
from evolving_infra.router.client.agent.identity import RequestIdentity

class Scheduler:
    def __init__(self, endpoint: str, *, backend: BackendClient | None = None):
        """Configure a Python facade for one external Rust scheduler endpoint.

        Args:
            endpoint: Rust scheduler HTTP(S) base URL.
            backend: Optional caller-owned transport; None creates an owned Backend.

        Returns:
            None; records the endpoint and backend ownership.

        Raises:
            ValueError: endpoint is not an accepted base URL.
        """
        self.endpoint = endpoint_url(endpoint)
        self.backend = backend if backend is not None else Backend()
        self._owns_backend = backend is None

    async def submit(self, request: Request, *, chat: bool = False) -> dict[str, Any]:
        """Send an ordinary model request through the external scheduler.

        Args:
            request: Identified model request to submit.
            chat: Select chat completions instead of text completions.

        Returns:
            The decoded JSON generation response.
        """
        return await self.backend.generate(self.endpoint, request, chat=chat)

    def stream(self, request: Request, *, chat: bool = False) -> AsyncIterator[bytes]:
        """Open an ordinary streaming model request through the scheduler.

        The caller must consume or close the iterator to release HTTP resources.

        Args:
            request: Identified model request to submit.
            chat: Select chat completions instead of text completions.

        Returns:
            An async iterator of raw response bytes.
        """
        return self.backend.stream(self.endpoint, request, chat=chat)

    async def submit_prepared(
        self,
        tokenizer_endpoint: str,
        prepared: Mapping[str, Any],
        *,
        identity: RequestIdentity | None = None,
    ) -> dict[str, Any]:
        """Schedule an existing tokenizer descriptor and return decode output.

        Token arrays stay at the input-preparation service. This does not tokenize
        input or change its previously stored generation settings.

        Args:
            tokenizer_endpoint: Input-preparation service HTTP(S) base URL.
            prepared: Compact descriptor returned by /prepare; contains no raw tokens.
            identity: Optional original RequestIdentity, enabling trajectory retention.

        Returns:
            The decoded JSON response from the selected D engine.
        """
        if identity is None:
            return await self.backend.generate_prepared(
                self.endpoint, tokenizer_endpoint, prepared
            )
        return await self.backend.generate_prepared(
            self.endpoint, tokenizer_endpoint, prepared, identity=identity
        )

    def stream_prepared(
        self,
        tokenizer_endpoint: str,
        prepared: Mapping[str, Any],
        *,
        identity: RequestIdentity | None = None,
    ) -> AsyncIterator[bytes]:
        """Schedule a prepared streaming request through the external runtime.

        The caller must close the iterator when stopping before stream completion.

        Args:
            tokenizer_endpoint: Input-preparation service HTTP(S) base URL.
            prepared: Compact descriptor of input already prepared for streaming.
            identity: Optional original RequestIdentity, enabling trajectory retention.

        Returns:
            An async iterator relaying decode-response bytes.
        """
        if identity is None:
            return self.backend.stream_prepared(
                self.endpoint, tokenizer_endpoint, prepared
            )
        return self.backend.stream_prepared(
            self.endpoint, tokenizer_endpoint, prepared, identity=identity
        )

    async def end_trajectory(self, agent_id: str, subagent_id: int) -> dict[str, Any]:
        """Input: unique trajectory identity. Output: async retention-release acceptance.

        This does not stop sibling trajectories or close the client transport.
        """
        return await self.backend.end_trajectory(self.endpoint, agent_id, subagent_id)

    async def get_backend_loads(self) -> dict[str, Any]:
        """Observe the Rust Backend's current merged workload without placement.

        This client call does not schedule a probe or advance decode round-robin.

        Args:
            None.

        Returns:
            The runtime's JSON load snapshot.
        """
        return await self.backend.get_backend_loads(self.endpoint)

    async def close(self):
        """Close the transport only when this facade created it.

        Args:
            None.

        Returns:
            None; caller-supplied backends remain caller-owned.
        """
        if self._owns_backend:
            await self.backend.close()

    async def __aenter__(self):
        """Enter the scheduler's asynchronous context.

        Args:
            None.

        Returns:
            This Scheduler instance.
        """
        return self

    async def __aexit__(self, *exc):
        """Release an owned backend when leaving the asynchronous context.

        Args:
            *exc: Context exit exception type, value, and traceback; unused.

        Returns:
            None; any context exception remains unsuppressed.
        """
        await self.close()
