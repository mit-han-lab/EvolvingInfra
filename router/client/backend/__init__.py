"""Lazily expose the Python HTTP client for the authoritative Rust Backend.
"""

__all__ = ["Backend"]


def __getattr__(name):
    """Load the HTTP Backend class on demand.

    Args:
        name: Public attribute name to resolve.

    Returns:
        The Backend class.

    Raises:
        AttributeError: The name is not Backend.
    """
    if name == "Backend":
        from .backend import Backend
        return Backend
    raise AttributeError(name)
