# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Opt-in task / child / model-step identities for the frozen DAG replay client.

request_id = event_index * MIGRATION_SLOTS + attempt, so a Trie request migration (a fresh rid_<hex>
after an abort, migration_limit <= 3) gets the next step identity of the same branch and its engine
RID is rewritten to match; request IDs stay monotonic per (agent, subagent).

Adds identity headers only. It does not choose workers, force sticky routing,
change cache namespaces, or retain a process-wide dictionary of request IDs.
Install after the optional prepared transport adapter and before Client.run.
"""
from contextvars import ContextVar
from dataclasses import dataclass, replace
import hashlib
import json
import re

from evolving_infra.router.client.agent.identity import RequestIdentity


MIGRATION_SLOTS = 4  # attempt 0 plus Trie's migration_limit 3
_MIGRATED_RID = re.compile(r"rid_[0-9a-f]{32}")


@dataclass(frozen=True)
class TaskScope:
    agent_id: str
    cycle_id: int
    children: dict[str, int]
    lifecycle_task_trace: str | None = None


# One of several replay processes: its name. Processes replay the same traces with the same cycle
# numbers, and two live agents with one identity are refused by the router.
PROCESS = ""


def _agent_id(kind, trace_id, cycle_id):
    if not isinstance(trace_id, str) or not trace_id or type(cycle_id) is not int or cycle_id < 0:
        raise ValueError("Replay identity needs a trace ID and nonnegative cycle")
    digest = hashlib.sha256((PROCESS + trace_id).encode("utf-8")).hexdigest()
    return f"{kind}:{digest}:cycle:{cycle_id}"


def install(client_type, writer_type, *, sticky_agent=False):
    """Wrap dedicated client/writer subclasses; return a CPU-test uninstall hook.

    The prepared transport validates these identities but currently sends only
    the resulting RID to the runtime. This is not an agent-aware routing policy.
    """
    if getattr(client_type, "_task_identity_installed", False):
        raise ValueError("Task identity adapter is already installed")
    if getattr(writer_type, "_task_identity_installed", False):
        raise ValueError("Use a separate lifecycle writer for each identity adapter")
    if not hasattr(client_type, "_run_replay_task"):
        raise ValueError("Require the schema-v3 DAG replay client")
    original_task = client_type._run_replay_task
    original_trace = client_type._run_replay_trace
    original_create = client_type._create_completion_with_connect_retry
    original_identity = writer_type.request_identity
    scope = ContextVar("replay_task_scope", default=None)
    current_request = ContextVar("replay_request_identity", default=None)

    def check_writer(client):
        if not isinstance(getattr(client, "_lifecycle_writer", None), writer_type):
            raise ValueError("Task identity adapter requires its lifecycle writer")

    async def run_task(client, task, *args, **kwargs):
        check_writer(client)
        cycle = kwargs["replay_cycle_id"]
        children = [trace.trace_id for trace in task.traces
                    if trace.metadata.get("control_only") is not True]
        if len(set(children)) != len(children):
            raise ValueError("Duplicate agent trace in a replay task")
        value = TaskScope(_agent_id("task", task.trace_id, cycle), cycle,
            {json.dumps(["task", task.trace_id, child]): i for i, child in enumerate(children)},
            json.dumps(["task", task.trace_id]))
        token, request_token = scope.set(value), current_request.set(None)
        try:
            return await original_task(client, task, *args, **kwargs)
        finally:
            current_request.reset(request_token)
            scope.reset(token)

    async def run_trace(client, trace, *args, **kwargs):
        check_writer(client)
        # Each TaskGroup child inherits the root scope. A standalone schema-v2
        # trace gets a one-agent scope; the original execution flow is retained.
        if scope.get() is not None:
            return await original_trace(client, trace, *args, **kwargs)
        cycle = kwargs.get("replay_cycle_id", 0)
        value = TaskScope(_agent_id("trace", trace.trace_id, cycle), cycle,
                          {kwargs.get("identity_trace_id") or trace.trace_id: 0})
        token, request_token = scope.set(value), current_request.set(None)
        try:
            return await original_trace(client, trace, *args, **kwargs)
        finally:
            current_request.reset(request_token)
            scope.reset(token)

    def identity(**kwargs):
        original = original_identity(**kwargs)
        value = scope.get()
        if value is None:
            # Other clients using this writer are unaffected. An adapted
            # request without an identity is still rejected by create().
            return original
        if kwargs["trace_id"] == value.lifecycle_task_trace:
            # TASK_START/COMPLETE is accounting, not a generated request.
            return original
        if kwargs["cycle_id"] != value.cycle_id or kwargs["trace_id"] not in value.children:
            raise ValueError("Replay identity does not belong to its task")
        request = RequestIdentity(value.agent_id, value.children[kwargs["trace_id"]],
                                  kwargs["event_index"] * MIGRATION_SLOTS)
        current_request.set(request)
        return replace(original, experiment_request_id=request.engine_rid)

    async def create(client, request_kwargs, *args, **kwargs):
        request = current_request.get()
        rid = kwargs.get("rid")
        if (request is not None and rid != request.engine_rid and isinstance(rid, str)
                and _MIGRATED_RID.fullmatch(rid) and request.request_id % MIGRATION_SLOTS < MIGRATION_SLOTS - 1):
            # Trie migrated the request under a fresh rid: next step identity, same branch.
            request = RequestIdentity(request.agent_id, request.subagent_id, request.request_id + 1)
            current_request.set(request)
            kwargs["rid"] = request.engine_rid
            extra = request_kwargs.get("extra_body")
            if isinstance(extra, dict) and "rid" in extra:
                request_kwargs["extra_body"] = {**extra, "rid": request.engine_rid}
        if request is None or request.engine_rid != kwargs.get("rid"):
            raise ValueError("Model request lost its task/child/step identity")
        headers = request.headers()
        if sticky_agent:
            # Same task replay instance and same child agent; excludes model-step ID.
            key = json.dumps([request.agent_id, request.subagent_id], separators=(",", ":"))
            headers["x-smg-routing-key"] = hashlib.sha256(key.encode()).hexdigest()
        old = request_kwargs.get("extra_headers", {})
        request_kwargs["extra_headers"] = {
            **{key: value for key, value in old.items() if key.lower() not in headers},
            **headers,
        }
        return await original_create(client, request_kwargs, *args, **kwargs)

    client_type._run_replay_task = run_task
    client_type._run_replay_trace = run_trace
    client_type._create_completion_with_connect_retry = create
    writer_type.request_identity = staticmethod(identity)
    writer_type._task_identity_installed = True
    client_type._task_identity_installed = True

    def uninstall():
        client_type._run_replay_task = original_task
        client_type._run_replay_trace = original_trace
        client_type._create_completion_with_connect_retry = original_create
        writer_type.request_identity = staticmethod(original_identity)
        del writer_type._task_identity_installed
        del client_type._task_identity_installed

    return uninstall
