"""Python client for the Router: agent identities and the Scheduler facade.

Importing this package loads no HTTP transport; ``Agent``, ``Backend`` and
``Scheduler`` are resolved on first use.
"""
from evolving_infra.router.client.types import BackendClient, Request, RequestIdentity, RequestScheduler

__all__ = ["Agent", "Backend", "BackendClient", "Request", "RequestIdentity", "RequestScheduler", "Scheduler"]


def __getattr__(name):
    """Resolve ``Agent``, ``Backend`` or ``Scheduler`` on first use.

    Args:
        name: Public attribute name.

    Returns:
        The requested class.

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
        from evolving_infra.router.client.scheduler import Scheduler
        return Scheduler
    raise AttributeError(name)
