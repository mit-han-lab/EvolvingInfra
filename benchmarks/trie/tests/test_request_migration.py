import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI

from trie.client import Client, RequestAborted
from trie.client_lifecycle import RequestTiming
from trie.results import BenchmarkResult, StreamAccumulator


def _chunk(choices: list, usage: dict | None = None) -> bytes:
    body: dict = {
        "id": "cmpl-1",
        "choices": choices,
        "created": 1,
        "model": "fake",
        "object": "text_completion",
    }
    if usage is not None:
        body["usage"] = usage
    return b"data: " + json.dumps(body).encode() + b"\n\n"


def _token(token_id: int, *, prompt_ids: list[int] | None = None) -> bytes:
    choice: dict = {
        "finish_reason": None,
        "index": 0,
        "logprobs": None,
        "text": "x",
        "token_ids": [token_id],
    }
    if prompt_ids is not None:
        choice["prompt_token_ids"] = prompt_ids
    return _chunk([choice])


def _usage(prompt: int, completion: int) -> bytes:
    return _chunk(
        [],
        usage={
            "completion_tokens": completion,
            "prompt_tokens": prompt,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": prompt},
        },
    )


_ABORT = _chunk([{"finish_reason": "abort", "index": 0, "logprobs": None, "text": ""}])
_DONE = b"data: [DONE]\n\n"
# Token 7 streamed, then the worker was removed.
_ABORTED = _token(7, prompt_ids=[1, 2, 3]) + _ABORT + _DONE
_ABORTED_EMPTY = _ABORT + _DONE
_ERROR_EVENT = (
    b'data: {"error": {"message": "worker shut down", "type": "ServiceUnavailable",'
    b' "code": 503}}\n\n'
)
_COMPLETE = _token(7, prompt_ids=[1, 2, 3]) + _token(8) + _usage(3, 2) + _DONE
# The continuation: prompt plus the streamed token 7, then the remaining token 8.
_CONTINUED = _token(8, prompt_ids=[1, 2, 3, 7]) + _usage(4, 1) + _DONE


class _Lifecycle:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, event: str, request: object, *, attrs=None, **_: object) -> None:
        self.events.append((event, attrs or {}))


def _client(
    bodies: list[bytes | httpx.AsyncByteStream], *, migration_limit: int, fast_path: bool = True
) -> tuple[Client, list]:
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        body = bodies[len(sent) - 1]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            **({"stream": body} if isinstance(body, httpx.AsyncByteStream) else {"content": body}),
        )

    client = Client.__new__(Client)
    client._client = AsyncOpenAI(
        base_url="http://test/v1",
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    client._model = "fake"
    client._strict_replay_openai_fast_path = fast_path
    client._migration_limit = migration_limit
    client._lifecycle_writer = _Lifecycle()
    return client, sent


async def _send(client: Client, timing: RequestTiming, result: BenchmarkResult):
    first_tokens: list[tuple[int, int]] = []
    generation = await client._execute_with_migration(
        [1, 2, 3],
        2,
        rid="rid_" + "0" * 32,
        on_first_token=lambda ts_ns, mono_ns: first_tokens.append((ts_ns, mono_ns)),
        request_timing=timing,
        result=result,
        lifecycle_request=object(),
        forced_output_token_ids=[7, 8],
        stream=True,
        trace_start=0.0,
        stream_acc=StreamAccumulator(),
        return_token_ids=True,
        deterministic_sampling=True,
    )
    return generation, first_tokens


@pytest.mark.parametrize("fast_path", [True, False])
def test_aborted_request_continues_after_the_streamed_tokens(fast_path: bool) -> None:
    client, sent = _client([_ABORTED, _CONTINUED], migration_limit=1, fast_path=fast_path)
    timing = RequestTiming.start()
    result = BenchmarkResult()
    generation, first_tokens = asyncio.run(_send(client, timing, result))

    assert len(sent) == 2
    assert sent[1]["prompt"] == [1, 2, 3, 7]
    assert sent[1]["max_tokens"] == 1
    assert sent[1]["forced_output_token_ids"] == [8]
    assert sent[1]["rid"].startswith("rid_")
    assert not sent[1]["rid"].startswith(sent[0]["rid"])
    # The client sees one request: original prompt, the whole output.
    assert generation.prompt_token_ids == [1, 2, 3]
    assert generation.generated_token_ids == [7, 8]
    assert generation.prompt_tokens == 3
    assert generation.completion_tokens == 2
    assert generation.usage.prompt_tokens == 3
    assert generation.usage.completion_tokens == 2
    assert generation.cached_tokens == 3
    assert result.migrated_requests == 1
    assert len(first_tokens) == 1
    assert [e for e, _ in client._lifecycle_writer.events] == ["CLIENT_MIGRATE"]
    assert client._lifecycle_writer.events[0][1]["continued_tokens"] == 1
    # Latency is measured from the first attempt's start.
    assert timing.request_start.mono_ns <= timing.first_parsed_token.mono_ns


@pytest.mark.parametrize("fast_path", [True, False])
def test_failed_stream_is_resent_from_the_start(fast_path: bool) -> None:
    client, sent = _client([_ERROR_EVENT, _COMPLETE], migration_limit=1, fast_path=fast_path)
    result = BenchmarkResult()
    generation, first_tokens = asyncio.run(_send(client, RequestTiming.start(), result))

    assert len(sent) == 2
    assert sent[1]["prompt"] == [1, 2, 3]
    assert sent[1]["max_tokens"] == 2
    assert generation.generated_token_ids == [7, 8]
    assert result.migrated_requests == 1
    assert len(first_tokens) == 1


@pytest.mark.parametrize("fast_path", [True, False])
def test_transport_disconnect_is_migrated_without_duplicate_first_token(fast_path):
    class Disconnect(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield _token(7, prompt_ids=[1, 2, 3])
            raise httpx.RemoteProtocolError("worker terminated during response")

    client, sent = _client([Disconnect(), _COMPLETE], migration_limit=1, fast_path=fast_path)
    result = BenchmarkResult()
    generation, first_tokens = asyncio.run(_send(client, RequestTiming.start(), result))
    assert len(sent) == 2
    assert sent[1]["prompt"] == [1, 2, 3]
    assert generation.generated_token_ids == [7, 8]
    assert len(first_tokens) == 1
    assert result.migrated_requests == 1


def test_without_migration_an_abort_fails_the_request() -> None:
    client, sent = _client([_ABORTED, _CONTINUED], migration_limit=0)
    with pytest.raises(RequestAborted):
        asyncio.run(_send(client, RequestTiming.start(), BenchmarkResult()))
    assert len(sent) == 1


def test_migration_limit_bounds_the_attempts() -> None:
    client, sent = _client([_ABORTED, _ABORTED_EMPTY, _CONTINUED], migration_limit=1)
    result = BenchmarkResult()
    with pytest.raises(RequestAborted):
        asyncio.run(_send(client, RequestTiming.start(), result))
    assert len(sent) == 2
    assert result.migrated_requests == 1
