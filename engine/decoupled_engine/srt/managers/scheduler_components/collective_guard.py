"""Fail fast on a TP collective issued from code that must not issue one.

The symmetric launch thread (disaggregation/symmetric/launch_thread.py) runs
the previous result's processing on the scheduler thread while the worker
thread launches the next forward. Every TP rank takes that path at the same
launches, but the processing must still issue no TP collective: the
collectives of a pass stay at fixed points before the launch (gate-pending
reorder audit, gate_pending_reorder.py). If a later change adds one there, a
rank could hang in it while its peers wait in a different collective.

``forbid(reason)`` marks the current thread for the duration of a block, and
``check(where)`` (called first in the guarded collectives) raises
``RuntimeError`` inside such a block instead of entering the collective. The
mark is per thread: the model forward's own collectives run on the worker
thread and are not affected. Outside a ``forbid`` block (and so always when
the launch thread is off) ``check`` only reads one thread-local list.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Iterator, List

_state = threading.local()


def _stack() -> List[str]:
    stack = getattr(_state, "stack", None)
    if stack is None:
        stack = _state.stack = []
    return stack


@contextlib.contextmanager
def forbid(reason: str) -> Iterator[None]:
    """No guarded TP collective may run on this thread inside the block."""
    stack = _stack()
    stack.append(reason)
    try:
        yield
    finally:
        stack.pop()


def check(where: str) -> None:
    """Raise if a ``forbid`` block of the current thread is open."""
    stack = getattr(_state, "stack", None)
    if stack:
        raise RuntimeError(f"TP collective {where} inside {stack[-1]}")


def active() -> bool:
    """Whether a ``forbid`` block of the current thread is open (tests)."""
    return bool(getattr(_state, "stack", None))
