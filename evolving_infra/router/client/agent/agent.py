"""A branch of one problem, with stable IDs across its model calls."""
from dataclasses import dataclass, field
import threading
from typing import Any
import uuid

from evolving_infra.router.client.types import RequestScheduler

from .identity import RequestIdentity
from .request import Request


@dataclass
class _Problem:
    agent_id: str
    next_subagent: int = 1
    lock: threading.Lock = field(default_factory=threading.Lock)


class Agent:
    """Root branch is 0; fork() allocates IDs across the entire problem.

    The frontend that creates the problem owns branch/step allocation. Its
    IDs may be sent to remote executors; independent allocators must not create
    the same agent_id. Caller-supplied identities can use Request directly.
    This is identity management, not automatic agent execution or a new DAG.
    """

    def __init__(self, scheduler: RequestScheduler, agent_id: str | None = None):
        """Create the root branch and shared identity allocator for one problem.

        Args:
            scheduler: Client used to submit model requests.
            agent_id: Problem identifier; None creates a random UUID identifier.

        Returns:
            None; initializes branch zero without submitting work.

        Raises:
            ValueError: The problem identifier fails RequestIdentity validation.
        """
        identifier = uuid.uuid4().hex if agent_id is None else agent_id
        RequestIdentity(identifier, 0, 0)  # Validate once before allocating work.
        self.scheduler = scheduler
        self._problem = _Problem(identifier)
        self.subagent_id = 0
        self.parent_subagent_id: int | None = None
        self._next_request = 0
        self._closed = False

    @property
    def agent_id(self) -> str:
        """Read the problem identifier shared by this branch and its siblings.

        Args:
            None.

        Returns:
            The stable problem identifier.
        """
        return self._problem.agent_id

    def fork(self) -> "Agent":
        """Allocate a child branch with a problem-wide unique subagent number.

        Allocation holds the problem lock. The child records this branch as its
        parent and starts request numbering at zero.

        Args:
            None.

        Returns:
            A new Agent sharing the scheduler and problem allocator.

        Raises:
            RuntimeError: This branch is closed.
            ValueError: The next subagent number exceeds the identity range.
        """
        with self._problem.lock:
            if self._closed:
                raise RuntimeError("Agent is closed")
            child = object.__new__(Agent)
            child.scheduler = self.scheduler
            child._problem = self._problem
            child.subagent_id = self._problem.next_subagent
            RequestIdentity(self.agent_id, child.subagent_id, 0)
            self._problem.next_subagent += 1
            child.parent_subagent_id = self.subagent_id
            child._next_request = 0
            child._closed = False
            return child

    def request(self, payload: dict[str, Any]) -> Request:
        """Allocate the next model-step identity and copy its request payload.

        Advances only this branch's request counter under the shared problem lock.

        Args:
            payload: Generation request fields; copied shallowly.

        Returns:
            A Request containing identity, payload, and parent branch metadata.

        Raises:
            RuntimeError: This branch is closed.
            ValueError: The next request number exceeds the identity range.
        """
        with self._problem.lock:
            if self._closed:
                raise RuntimeError("Agent is closed")
            identity = RequestIdentity(self.agent_id, self.subagent_id, self._next_request)
            self._next_request += 1
        return Request(identity, dict(payload), self.parent_subagent_id)

    async def submit(self, payload: dict[str, Any], *, chat: bool = False):
        """Allocate a request identity and submit the model call through the scheduler.

        Consumes a request number even if transport later fails; submission errors
        propagate to the caller.

        Args:
            payload: Generation request fields.
            chat: Use the chat-completions endpoint when True.

        Returns:
            The scheduler's decoded JSON response.
        """
        return await self.scheduler.submit(self.request(payload), chat=chat)

    async def end_trajectory(self):
        """Stop this branch and explicitly release its KV retention registrations.

        Args:
            None. The caller must have stopped issuing work for this branch.

        Returns:
            The runtime's async acceptance. Retry this method on transport
            failure; it is idempotent and siblings are unaffected.
        """
        self.close()
        return await self.scheduler.end_trajectory(self.agent_id, self.subagent_id)

    def close(self):
        """Stop allocating new requests or children from this branch.

        Existing identities remain valid. Closing does not close sibling branches,
        cancel submitted work, or reclaim KV cache.

        Args:
            None.

        Returns:
            None; marks this branch closed.
        """
        with self._problem.lock:
            self._closed = True
