"""Opt-in KV session identity for routers with agent-aware KV retention.

Each agent trace gets one identity. Every request carries
``x-kv-session-context: {"agent_id", "subagent_id": 0, "request_id": n}`` with
``n`` increasing per agent in send order, and the agent's end is reported once
with ``POST {router}/kv_sessions/end``. Request bodies are unchanged, so rid,
cache salt and prompt token IDs stay byte-identical.
"""

from concurrent.futures import ThreadPoolExecutor, wait
import contextvars
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import uuid

HEADER = "x-kv-session-context"
END_PATH = "/kv_sessions/end"
AGENT_ID_MAX_BYTES = 512
END_TIMEOUT_S = 2.0

_current = contextvars.ContextVar("trie_kv_session_agent", default=None)
_lock = threading.Lock()
_pending = set()
_executor = None
STATS = dict(installed=False, agents=0, agents_control_only=0, tagged_requests=0,
             end_sent=0, end_ok=0, end_400=0, end_err=0, end_skipped_zero=0,
             end_skipped_deadline=0, errors=[])


class AgentSession:
    __slots__ = ("agent_id", "counter")

    def __init__(self, agent_id):
        self.agent_id, self.counter = agent_id, 0

    def header(self):
        return json.dumps({"agent_id": self.agent_id, "subagent_id": 0,
                           "request_id": self.counter}, separators=(",", ":"))


def _count(key, value=1):
    with _lock:
        STATS[key] += value


def _note(message):
    with _lock:
        if len(STATS["errors"]) < 20:
            STATS["errors"].append(message[:300])


def _router_root(client):
    root = os.environ.get("TRIE_KV_SESSION_ROUTER")
    if root:
        return root.rstrip("/")
    base = str(getattr(getattr(client, "_client", None), "base_url", "") or "").rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def _past_deadline(client):
    boundary = getattr(client, "_benchmark_boundary", None)
    if boundary is None:
        return False
    if getattr(boundary, "request_stop", False):
        return True
    deadline = (getattr(boundary, "deadline", None) or {}).get("mono_ns")
    return deadline is not None and time.monotonic_ns() >= deadline


def _post_end(url, body):
    request = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=END_TIMEOUT_S) as reply:
            status = reply.status
    except urllib.error.HTTPError as error:
        status = error.code
    except Exception as error:  # noqa: BLE001 - END is advisory; never fail replay
        _count("end_err")
        _note(f"END {type(error).__name__}: {error}")
        return
    if 200 <= status < 300:
        _count("end_ok")
    elif status == 400:  # unknown session, e.g. one the router already retired
        _count("end_400")
    else:
        _count("end_err")
        _note(f"END HTTP {status}")


def _schedule_end(client, agent):
    if agent.counter <= 0:
        _count("end_skipped_zero")
        return
    if _past_deadline(client):
        _count("end_skipped_deadline")
        return
    body = json.dumps({"agent_id": agent.agent_id, "subagent_id": 0}).encode()
    future = _executor.submit(_post_end, _router_root(client) + END_PATH, body)
    _count("end_sent")
    with _lock:
        _pending.add(future)
    future.add_done_callback(lambda f: _pending.discard(f))


def install(client_cls):
    """Wrap one agent trace and every completion send of ``client_cls``."""
    global _executor
    if STATS["installed"]:
        return
    _executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="trie-kv-end")
    run_trace = client_cls._run_replay_trace
    send = client_cls._create_completion_with_connect_retry

    async def _run_replay_trace(self, trace, *args, **kwargs):
        if (getattr(trace, "metadata", None) or {}).get("control_only") is True:
            _count("agents_control_only")
            return await run_trace(self, trace, *args, **kwargs)
        identity = kwargs.get("identity_trace_id") or trace.trace_id
        agent_id = f"{uuid.uuid4().hex}:{identity}".encode()[:AGENT_ID_MAX_BYTES]
        agent = AgentSession(agent_id.decode("utf-8", "ignore"))
        _count("agents")
        token = _current.set(agent)
        try:
            return await run_trace(self, trace, *args, **kwargs)
        finally:
            _current.reset(token)
            try:
                _schedule_end(self, agent)
            except Exception as error:  # noqa: BLE001
                _count("end_err")
                _note(f"END scheduling {type(error).__name__}: {error}")

    async def _create_completion_with_connect_retry(self, request_kwargs, *args, **kwargs):
        agent = _current.get()
        if agent is not None:
            agent.counter += 1
            headers = dict(request_kwargs.get("extra_headers") or {})
            headers[HEADER] = agent.header()
            request_kwargs = dict(request_kwargs, extra_headers=headers)
            _count("tagged_requests")
        return await send(self, request_kwargs, *args, **kwargs)

    client_cls._run_replay_trace = _run_replay_trace
    client_cls._create_completion_with_connect_retry = _create_completion_with_connect_retry
    STATS["installed"] = True


def save_stats(timing_file=None):
    """Wait briefly for queued END posts and write the counters next to the timing file."""
    with _lock:
        pending = set(_pending)
    if pending:
        wait(pending, timeout=10.0)
    path = os.environ.get("TRIE_KV_SESSION_STATS")
    if not path and (timing_file or os.environ.get("TRIE_BENCHMARK_TIMING_FILE")):
        path = str(Path(timing_file or os.environ["TRIE_BENCHMARK_TIMING_FILE"])
                   .with_name(f"kv-session-client-stats-{os.getpid()}.json"))
    if path:
        with _lock:
            Path(path).write_text(json.dumps({**STATS, "end_pending_at_exit": len(_pending)}, indent=1))
