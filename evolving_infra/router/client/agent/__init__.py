# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Export request identities and lazily load the Agent identity allocator.
"""
from .identity import RequestIdentity
from .request import Request

__all__ = ["Agent", "Request", "RequestIdentity"]


def __getattr__(name):
    """Load the Agent class only when its public name is requested.

    Args:
        name: Public attribute name to resolve.

    Returns:
        The Agent class.

    Raises:
        AttributeError: The name is not Agent.
    """
    if name == "Agent":
        from .agent import Agent
        return Agent
    raise AttributeError(name)
