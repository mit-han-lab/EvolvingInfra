import asyncio
import json
from types import SimpleNamespace

import pytest

from trie import kv_session
from trie.actual_output_context import ActualOutputContext
from trie.client import Client


def test_actual_output_replaces_retained_assistant_tokens():
    context = ActualOutputContext()
    first = list(range(40))
    assert context.prepare(first)[0] == first
    context.commit(first, [900, 901, 902], recorded_output=[50, 51, 52])
    # The next recorded prompt retains the recorded assistant turn and appends a tool result.
    second = first + [50, 51, 52] + [60, 61]
    prompt, attrs = context.prepare(second)
    assert prompt == first + [900, 901, 902] + [60, 61]
    assert attrs["actual_output_history_tokens"] == 3


def test_actual_output_requires_model_generation():
    with pytest.raises(ValueError):
        Client(endpoint="http://127.0.0.1:9/v1", model="m", replay_output_mode="forced",
               replay_context_mode="actual_output")


def test_kv_session_tags_requests_per_agent_and_reports_end(monkeypatch):
    posts = []
    monkeypatch.setattr(kv_session, "_post_end", lambda url, body: posts.append((url, json.loads(body))))
    sent = []

    class Fake:
        _client = SimpleNamespace(base_url="http://router:1/v1/")
        _benchmark_boundary = None

        async def _run_replay_trace(self, trace, **kwargs):
            await self._create_completion_with_connect_retry({"extra_headers": {"X-SMG-Routing-Key": "k"}})
            await self._create_completion_with_connect_retry({})

        async def _create_completion_with_connect_retry(self, request_kwargs, **kwargs):
            sent.append(request_kwargs.get("extra_headers") or {})

    monkeypatch.setattr(kv_session, "STATS", {**kv_session.STATS, "installed": False})
    kv_session.install(Fake)
    trace = SimpleNamespace(trace_id="t", metadata={})
    asyncio.run(Fake()._run_replay_trace(trace, identity_trace_id="agent-a"))
    for future in list(kv_session._pending):
        future.result(timeout=5)
    contexts = [json.loads(h[kv_session.HEADER]) for h in sent]
    assert [c["request_id"] for c in contexts] == [1, 2]
    assert contexts[0]["agent_id"] == contexts[1]["agent_id"]
    assert contexts[0]["agent_id"].endswith(":agent-a")
    assert sent[0]["X-SMG-Routing-Key"] == "k"
    assert posts == [("http://router:1/kv_sessions/end",
                      {"agent_id": contexts[0]["agent_id"], "subagent_id": 0})]
