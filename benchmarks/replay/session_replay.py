"""Extra-only replay wiring: preserve prepared identities and signal real branch END.

install(end_outcomes=...) names the branch outcomes that send END (default: only "complete"). With
session retention off in the engines and router, pass retention_enabled=False: no END is sent.

Install after the frozen prepared/identity adapters. This file changes neither
sampling, trace timing, placement nor production source. OFF and ON use the same
request path; only ON sends END to the enabled retention controller.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
import json
import time

from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError

from evolving_infra.router.client.agent.identity import RequestIdentity
from evolving_infra.router.client.backend import Backend
from evolving_infra.router.client.scheduler import Scheduler
from prepared_replay import ReplayAdapter


@dataclass
class Branch:
    """One actual replay branch; completion comes from TRACE_COMPLETE, not EOF."""

    trace_id: str
    identity: RequestIdentity | None = None
    complete: bool = False
    failed: bool = False


_branch: ContextVar[Branch | None] = ContextVar("extra_retention_branch", default=None)
_request: ContextVar[RequestIdentity | None] = ContextVar("extra_retention_request", default=None)


class IdentityScheduler(Scheduler):
    """Add the original identity to the existing compact prepared-plan HTTP call."""

    def stream_prepared(self, tokenizer_endpoint, prepared, *, identity=None):
        """Input: prepared descriptor. Output: original stream with verified identity."""
        identity = identity or _request.get()
        if identity is None or identity.engine_rid != prepared.get("request_id"):
            raise ValueError("Prepared replay lost its trajectory identity")
        return super().stream_prepared(tokenizer_endpoint, prepared, identity=identity)


class SessionReplayAdapter(ReplayAdapter):
    """Reuse all existing stream, timing, validation and cancellation machinery."""

    def __init__(self, config):
        """Input: frozen prepared configuration. Output: isolated adapter instance."""
        super().__init__(config)
        self.scheduler = IdentityScheduler(config.router_url, backend=self.backend)

    async def create(self, client, request, **kwargs):
        """Input: original completion request. Output: unchanged completion stream."""
        headers = {key.lower(): value for key, value in request.get("extra_headers", {}).items()}
        identity = RequestIdentity(**json.loads(headers["x-agent-context"]))
        if identity.engine_rid != kwargs.get("rid"):
            raise ValueError("Replay identity differs from request RID")
        state = _branch.get()
        if state is not None:
            old = state.identity
            if old is not None and (old.agent_id, old.subagent_id) != (identity.agent_id, identity.subagent_id):
                raise ValueError("One replay branch changed trajectory identity")
            state.identity = identity
        token = _request.set(identity)
        try:
            return await ReplayAdapter.create(self, client, request, **kwargs)
        finally:
            _request.reset(token)


class EndQueue:
    """Bounded, single-writer END delivery; never block a model request on END."""

    def __init__(self, endpoint, enabled, record, *, timeout_s=2.0, capacity=4096,
                 max_attempts=3, retry_delay_s=0.1):
        """Input: router, arm flag, JSON-record callback. Output: dormant queue."""
        self.endpoint, self.enabled, self.record = endpoint, enabled, record
        self.timeout_s = timeout_s
        if max_attempts < 1 or retry_delay_s < 0:
            raise ValueError("Invalid END retry budget")
        self.max_attempts, self.retry_delay_s = max_attempts, retry_delay_s
        self.queue = asyncio.Queue(maxsize=capacity)
        self.worker = None
        self.backend = None
        self.failures = 0
        self.cleanup_error = None

    def emit(self, event, **details):
        """Input: event and non-token metadata. Output: one auditable record."""
        self.record(dict(event=event, wall_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(), **details))

    def completed(self, identity):
        """Input: truly completed branch identity. Output: queued END or OFF record."""
        value = dict(agent_id=identity.agent_id, subagent_id=identity.subagent_id)
        if not self.enabled:
            self.emit("END_DISABLED", **value)
            return
        self.queue.put_nowait(value)  # Exhaustion is a real failed trial, never silent loss.
        self.emit("END_QUEUED", **value)
        if self.worker is None:
            self.worker = asyncio.create_task(self._deliver())

    async def _deliver(self):
        """Input: queued branch identities. Output: bounded HTTP END acknowledgements."""
        self.backend = Backend(timeout_s=self.timeout_s)
        scheduler = Scheduler(self.endpoint, backend=self.backend)
        while True:
            value = await self.queue.get()
            try:
                await self._send_end(scheduler, value)
            except asyncio.CancelledError:
                self.emit("END_CANCELLED", **value)
                raise
            except Exception as exc:
                self.failures += 1
                self.emit("END_FAILED", **value, error_class=type(exc).__name__)
            finally:
                self.queue.task_done()

    async def _send_end(self, scheduler, value):
        """Input: one terminal identity. Output: ACK or bounded delivery failure.

        A lost response cannot tell us whether the controller already accepted
        END. Its same-identity END is idempotent, so retry only this operation;
        never retry an inference POST or create another version/session here.
        HTTP acceptance still does not prove engine-side release completion.
        """
        for attempt in range(1, self.max_attempts + 1):
            self.emit("END_ATTEMPT", **value, attempt=attempt)
            try:
                async with asyncio.timeout(self.timeout_s):
                    reply = await scheduler.end_trajectory(**value)
                    if reply.get("accepted") is not True:
                        raise ValueError("END was not accepted")
            except (TimeoutError, ClientConnectionError, ClientPayloadError, ClientResponseError) as exc:
                transient = not isinstance(exc, ClientResponseError) or (
                    exc.status in (408, 429) or 500 <= exc.status < 600)
                if not transient or attempt == self.max_attempts:
                    raise
                delay = self.retry_delay_s * attempt
                self.emit("END_RETRY", **value, attempt=attempt,
                          error_class=type(exc).__name__, retry_delay_s=delay)
                await asyncio.sleep(delay)
            else:
                self.emit("END_ACCEPTED", **value, attempt=attempt,
                          engine_release_confirmed=reply.get("engine_release_confirmed"))
                return

    async def close(self, timeout_s=10.0):
        """Input: cleanup budget. Output: drained queue or explicit failed trial."""
        try:
            async with asyncio.timeout(timeout_s):
                await self.queue.join()
        finally:
            if self.worker is not None:
                self.worker.cancel()
                await asyncio.gather(self.worker, return_exceptions=True)
            if self.backend is not None:
                await self.backend.close()
        if self.failures:
            raise RuntimeError(f"{self.failures} retention END control(s) failed")


def install(client_type, writer_type, config, *, retention_enabled, record, end_outcomes=("complete",)):
    """Input: dedicated frozen replay subclasses/config. Output: END queue receipt.

    Requires the old adapters to be installed first. TRACE_COMPLETE is emitted
    only after the frozen client validates all branch responses. Exceptions and
    cancelled/deadline branches are recorded without inventing trajectory END.
    """
    if getattr(client_type, "_extra_retention_installed", False):
        raise ValueError("Extra retention replay is already installed")
    if not getattr(client_type, "_prepared_replay_installed", False):
        raise ValueError("Install the frozen prepared replay adapter first")
    original_create = client_type._create_completion_with_connect_retry
    original_trace = client_type._run_replay_trace
    original_run = client_type.run
    original_emit = writer_type.emit
    ends = EndQueue(config.router_url, retention_enabled, record)

    async def create(client, request, **kwargs):
        """Input: original request. Output: original stream via identity-aware adapter."""
        adapter = getattr(client, "_prepared_replay_adapter", None)
        if adapter is None:
            client._prepared_replay_adapter = SessionReplayAdapter(config)
        elif not isinstance(adapter, SessionReplayAdapter) and not getattr(adapter, "_extra_identity_wired", False):
            # The prepared execute wrapper creates its adapter before create.
            # Upgrade only its Scheduler and bound create method, preserving
            # live owner/stream state and the existing connection pool.
            adapter.scheduler = IdentityScheduler(config.router_url, backend=adapter.backend)
            adapter.create = SessionReplayAdapter.create.__get__(adapter, SessionReplayAdapter)
            adapter._extra_identity_wired = True
        return await original_create(client, request, **kwargs)

    def emit(writer, event, *args, **kwargs):
        """Input: unchanged lifecycle event. Output: original log plus local state."""
        original_emit(writer, event, *args, **kwargs)
        state = _branch.get()
        if state is not None:
            if event == "TRACE_COMPLETE":
                state.complete = True
            elif event == "REQUEST_FAILED":
                state.failed = True

    async def trace(client, item, *args, **kwargs):
        """Input: one original branch. Output: original result and accurate END event."""
        state = Branch(item.trace_id)
        token = _branch.set(state)
        outcome = "incomplete"
        try:
            result = await original_trace(client, item, *args, **kwargs)
            outcome = "complete" if state.complete and not state.failed else "failed" if state.failed else "skipped_or_incomplete"
            return result
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except BaseException:
            outcome = "failed"
            raise
        finally:
            _branch.reset(token)
            identity = state.identity
            ends.emit("BRANCH_TERMINAL", trace_id=state.trace_id, outcome=outcome,
                      agent_id=identity.agent_id if identity else None,
                      subagent_id=identity.subagent_id if identity else None)
            if outcome in end_outcomes and identity is not None:
                ends.completed(identity)

    async def run(client, *args, **kwargs):
        """Input: unchanged replay arguments. Output: original results and drained ENDs."""
        original_error = None
        try:
            return await original_run(client, *args, **kwargs)
        except BaseException as exc:
            original_error = exc
            raise
        finally:
            try:
                await ends.close()
            except Exception as cleanup:
                # Keep the replay's results: Trie writes output_json only after run returns.
                # run_trie.py turns the recorded error into a failing exit status.
                ends.cleanup_error = cleanup
                ends.emit("END_CLEANUP_FAILED", error_class=type(cleanup).__name__, error=str(cleanup))
                if original_error is not None:
                    original_error.add_note(f"Retention END cleanup failed: {type(cleanup).__name__}")

    client_type._create_completion_with_connect_retry = create
    client_type._run_replay_trace = trace
    client_type.run = run
    writer_type.emit = emit
    client_type._extra_retention_installed = True
    return ends
