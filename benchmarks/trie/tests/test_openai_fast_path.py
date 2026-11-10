import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI

from trie import client as client_module
from trie.client import Client


_SSE_BODY = b"".join(
    [
        b'data: {"id":"cmpl-1","choices":[{"finish_reason":null,'
        b'"index":0,"logprobs":null,"text":"x","prompt_token_ids":'
        b'[1,2,3],"token_ids":[7]}],"created":1,"model":"fake",'
        b'"object":"text_completion"}\n\n',
        b'data: {"id":"cmpl-1","choices":[],"created":1,"model":'
        b'"fake","object":"text_completion","usage":{"completion_tokens":1,'
        b'"prompt_tokens":3,"total_tokens":4}}\n\n',
        b"data: [DONE]\n\n",
    ]
)


def _request_kwargs(prompt: object = None) -> dict:
    if prompt is None:
        prompt = [1, 2, 3]
    return {
        "model": "fake",
        "prompt": prompt,
        "max_tokens": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        "extra_body": {
            "ignore_eos": True,
            "return_token_ids": True,
            "return_cached_tokens_details": True,
            "top_k": 1,
            "forced_output_token_ids": [7],
            "cache_salt": "session-1",
            "rid": "opx_1",
        },
        "temperature": 0.0,
        "top_p": 1.0,
        "extra_headers": {"X-SMG-Routing-Key": "route-1"},
    }


def _test_client(
    *,
    fast_path: bool,
    handler,
) -> tuple[Client, AsyncOpenAI]:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    sdk_client = AsyncOpenAI(
        base_url="http://test/v1",
        api_key="test",
        max_retries=0,
        http_client=http_client,
    )
    client = Client.__new__(Client)
    client._client = sdk_client
    client._strict_replay_openai_fast_path = fast_path
    client._strict_replay_connect_retry_delay_s = 0.0
    return client, sdk_client


async def _send_and_collect(client: Client, request_kwargs: dict) -> list:
    stream = await client._create_completion(request_kwargs)
    return [chunk async for chunk in stream]


def test_fast_path_preserves_wire_body_routing_header_and_sse_parser() -> None:
    requests: dict[bool, httpx.Request] = {}

    async def run() -> tuple[list, list]:
        clients: list[AsyncOpenAI] = []
        chunks: dict[bool, list] = {}
        try:
            for fast_path in (False, True):
                def handler(
                    request: httpx.Request,
                    *,
                    arm: bool = fast_path,
                ) -> httpx.Response:
                    requests[arm] = request
                    return httpx.Response(
                        200,
                        headers={"content-type": "text/event-stream"},
                        content=_SSE_BODY,
                    )

                client, sdk_client = _test_client(
                    fast_path=fast_path,
                    handler=handler,
                )
                clients.append(sdk_client)
                chunks[fast_path] = await _send_and_collect(
                    client,
                    _request_kwargs(),
                )
            return chunks[False], chunks[True]
        finally:
            for sdk_client in clients:
                await sdk_client.close()

    standard_chunks, fast_chunks = asyncio.run(run())

    assert requests[False].content == requests[True].content
    assert json.loads(requests[True].content) == {
        "model": "fake",
        "prompt": [1, 2, 3],
        "max_tokens": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "return_token_ids": True,
        "return_cached_tokens_details": True,
        "top_k": 1,
        "forced_output_token_ids": [7],
        "cache_salt": "session-1",
        "rid": "opx_1",
    }
    assert requests[False].headers["x-smg-routing-key"] == "route-1"
    assert requests[True].headers["x-smg-routing-key"] == "route-1"
    assert fast_chunks == standard_chunks
    assert fast_chunks[0].choices[0].text == "x"
    assert fast_chunks[1].usage is not None
    assert fast_chunks[1].usage.prompt_tokens == 3


@pytest.mark.parametrize("prompt", ["text", [], [1, True]])
def test_fast_path_rejects_non_authoritative_prompt(prompt: object) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"HTTP must not be reached: {request.url}")

    async def run() -> None:
        client, sdk_client = _test_client(fast_path=True, handler=handler)
        try:
            with pytest.raises(ValueError, match=r"non-empty list\[int\]"):
                await client._create_completion(_request_kwargs(prompt))
        finally:
            await sdk_client.close()

    asyncio.run(run())


def test_fast_path_rejects_unreviewed_request_field() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"HTTP must not be reached: {request.url}")

    async def run() -> None:
        client, sdk_client = _test_client(fast_path=True, handler=handler)
        kwargs = _request_kwargs()
        kwargs["timeout"] = 1.0
        try:
            with pytest.raises(ValueError, match="unsupported request fields"):
                await client._create_completion(kwargs)
        finally:
            await sdk_client.close()

    asyncio.run(run())


def test_fast_path_preserves_outer_predispatch_connect_retry() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("not connected", request=request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_SSE_BODY,
        )

    async def run() -> list:
        client, sdk_client = _test_client(fast_path=True, handler=handler)
        try:
            stream = await client._create_completion_with_connect_retry(
                _request_kwargs(),
                connect_retry_limit=1,
                rid="opx_1",
            )
            return [chunk async for chunk in stream]
        finally:
            await sdk_client.close()

    chunks = asyncio.run(run())
    assert calls == 2
    assert chunks[0].choices[0].text == "x"


def test_fast_path_requires_reviewed_openai_version(monkeypatch) -> None:
    monkeypatch.setattr(client_module, "OPENAI_VERSION", "2.31.0")
    with pytest.raises(RuntimeError, match="requires openai==2.30.0"):
        Client(
            endpoint="http://127.0.0.1:1/v1",
            model="fake",
            strict_replay_openai_fast_path=True,
        )


@pytest.mark.parametrize(
    ("strict_replay", "stream", "message"),
    [
        (False, True, "requires strict_replay=true"),
        (True, False, "requires stream=true"),
    ],
)
def test_fast_path_run_fails_closed_outside_streamed_strict_replay(
    strict_replay: bool,
    stream: bool,
    message: str,
) -> None:
    client = Client.__new__(Client)
    client._max_retries = 0
    client._strict_replay_openai_fast_path = True
    with pytest.raises(ValueError, match=message):
        asyncio.run(
            client.run(
                [],
                concurrency=1,
                strict_replay=strict_replay,
                stream=stream,
            )
        )
