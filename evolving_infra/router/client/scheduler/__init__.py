# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Expose the Python Scheduler facade (agent and backend classes are re-exported lazily).
"""
from evolving_infra.router.client.types import Request, RequestIdentity

__all__ = ["Agent", "Backend", "Request", "RequestIdentity", "Scheduler"]


def __getattr__(name):
    # Identity is useful in an existing HTTP client without loading a second
    # transport stack. Import aiohttp only when the caller uses our transport.
    """Lazily resolve scheduler, agent, or backend compatibility exports.

    Args:
        name: Public attribute name to resolve.

    Returns:
        The requested public class.

    Raises:
        AttributeError: The name has no lazy export.
    """
    if name == "Agent":
        from evolving_infra.router.client.agent import Agent
        return Agent
    if name == "Backend":
        from evolving_infra.router.client.backend import Backend
        return Backend
    if name == "Scheduler":
        from .scheduler import Scheduler
        return Scheduler
    raise AttributeError(name)
