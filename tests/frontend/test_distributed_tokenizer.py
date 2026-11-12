# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
import asyncio
import hashlib
import json
import os
import struct
from unittest.mock import patch

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer
import httpx
import pytest

from evolving_infra.router.frontend.forward import ForwardResponse
from evolving_infra.router.frontend.protocol import NodeConfig
from evolving_infra.router.frontend.server import TokenizerService
from evolving_infra.router.frontend.tokenizer import NativeHasher

TOKENIZE = "http://inference.example/v1/tokenize"
TARGET = "http://inference.example/generate"


def config(**overrides):
    """Build a tokenizer configuration with isolated test endpoints and overridable limits.

    Input: NodeConfig field overrides. Output: a validated NodeConfig instance.
    """
    return NodeConfig(
        **{
            "node_id": "cpu-a",
            "config_id": "test",
            "model_path": "/model",
            "tokenize_endpoint": TOKENIZE,
            "targets": {TARGET: "test"},
            "context_length": 4096,
            "page_size": 4,
            **overrides,
        }
    )


def body(request_id="id", stream=False):
    """Build a valid preparation request for the fixed test prompt.

    Input: request ID and stream flag. Output: a protocol-v1 prepare dictionary.
    """
    return {
        "protocol_version": 1,
        "request_id": request_id,
        "config_id": "test",
        "input": {"prompt": "test", "add_special_tokens": False},
        "generation": {
            "sampling_params": {"max_new_tokens": 4, "temperature": 0},
            "stream": stream,
        },
    }


def identity(info):
    """Extract the fields used to address a prepared result in tests.

    Input: prepare response info. Output: protocol/request/ref/config dictionary.
    """
    return {k: info[k] for k in ("protocol_version", "request_id", "ref", "config_id")}


def forward(info, target=TARGET):
    """Build a test forwarding envelope from a prepared result.

    Input: prepare response info and target URL. Output: a forward request dictionary.
    """
    return {
        **identity(info),
        "target": {"transport": "sglang_generate", "endpoint": target, "config_id": "test"},
    }


def reference_hash(ids, page_size=4, seed=b""):
    """Compute independent chained SHA-256 hashes for test token pages.

    Input: token IDs, positive page size and optional prefix seed bytes.
    Output: ordered hexadecimal page-hash strings.
    """
    result = []
    for offset in range(0, len(ids), page_size):
        page = ids[offset : offset + page_size]
        seed = hashlib.sha256(seed + struct.pack(f"<{len(page)}I", *page)).digest()
        result.append(seed.hex())
    return result


async def run_case(check, *, settings=None, tokenize_override=None, generate_override=None):
    """Run an assertion callback with an in-process API and real TCP fake upstream.

    Input: async check(service, api, seen), optional config and upstream override callbacks.
    Output: None after the callback completes and local test resources close.
    """
    seen = []

    async def respond(req):
        """Record an upstream request and select its default or overridden fake response.

        Input: aiohttp request plus captured overrides. Output: an httpx Response.
        """
        data = await req.json()
        seen.append((req.path, data))
        if req.path == "/v1/tokenize":
            if tokenize_override:
                return await tokenize_override(req)
            await asyncio.sleep(0)
            return httpx.Response(
                200, json={"tokens": [10, 11, 12], "count": 3, "max_model_len": 4096}
            )
        if generate_override:
            return await generate_override(req)
        return httpx.Response(
            200, json={"text": "ok", "meta_info": {"finish_reason": {"type": "length"}}}
        )

    async def handler(req):
        """Adapt a fake response to the real TCP test server's HTTP response type.

        Input: aiohttp request. Output: an aiohttp web.Response with matching status/body.
        """
        result = await respond(req)
        return web.Response(
            status=result.status_code, body=result.content, headers=dict(result.headers)
        )

    app = web.Application()
    app.router.add_post("/v1/tokenize", handler)
    app.router.add_post("/generate", handler)
    async with TestServer(app) as upstream:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:

            class LocalUpstream:
                """Redirect configured test endpoints to the ephemeral local HTTP server.

                Input: captured TestServer and aiohttp session. Output: an HTTP-client adapter.
                """
                def post(self, url, **kwargs):
                    """Preserve the request path while replacing its host with the test server.

                    Input: target URL and aiohttp POST options. Output: a request context manager.
                    """
                    return session.post(upstream.make_url(httpx.URL(url).path), **kwargs)

                async def close(self):
                    """Close the captured HTTP client session during service teardown.

                    Input: captured aiohttp session. Output: None.
                    """
                    await session.close()

            service = TokenizerService(settings or config(), reference_hash, LocalUpstream())
            async with service.lifespan(service.app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=service.app), base_url="http://proxy"
                ) as api:
                    await check(service, api, seen)


def test_prepare_only_tokenizes_and_forward_preserves_payload_and_frees_cache():
    """Verify separate preparation/generation, exact forwarding and immediate input cleanup.

    Input: no arguments; uses the standard local HTTP fixture. Output: None.
    """
    async def check(service, api, seen):
        """Assert the prepare/forward contract and request-ID reuse without history.

        Input: service, API client and recorded upstream requests. Output: None.
        """
        original = body()
        info = (await api.post("/prepare", json=original)).json()
        assert len(seen) == 1 and seen[0][0] == "/v1/tokenize"
        assert seen[0][1] == {**original["input"], "model": "/model"}
        assert info["prompt_tokens"] == 3 and "input_ids" not in info
        assert info["content"]["blocks"] == [
            {"hash": reference_hash([10, 11, 12])[0], "valid_tokens": 3}
        ]
        stored = service.store.entries[info["ref"]]
        assert isinstance(stored.data, bytes) and service.store.bytes_used == len(stored.data)
        response = await api.post("/forward", json=forward(info))
        assert response.status_code == 200 and response.json()["text"] == "ok"
        assert seen[-1] == (
            "/generate",
            {
                **original["generation"],
                "rid": "id",
                "input_ids": [10, 11, 12],
                "return_prompt_token_ids": False,
            },
        )
        assert not service.store.entries and service.store.bytes_used == service.upload_bytes == 0
        assert (await api.post("/forward", json=forward(info))).status_code == 404
        assert len(seen) == 2
        assert (await api.post("/prepare", json=original)).status_code == 200

    asyncio.run(run_case(check))


@pytest.mark.parametrize(
    "change,status",
    [
        ({"protocol_version": True}, 400),
        ({"protocol_version": 2}, 400),
        ({"agent_id": "extra"}, 400),
        ({"config_id": "wrong"}, 409),
        ({"input": {"prompt": "a", "messages": []}}, 400),
        ({"input": {"prompt": ["batch"], "add_special_tokens": False}}, 400),
        ({"input": {"prompt": "a", "add_special_tokens": "false"}}, 400),
        ({"input": {"model": "/wrong", "prompt": "a", "add_special_tokens": False}}, 409),
        ({"input": {"messages": [{"role": "user", "content": 5}]}}, 400),
        ({"input": {"messages": [{"role": "user", "content": {"type": "text"}}]}}, 400),
        (
            {
                "input": {
                    "messages": [
                        {"role": "user", "content": [{"type": "image_url", "image_url": {}}]}
                    ]
                }
            },
            422,
        ),
        ({"generation": {"sampling_params": {"max_new_tokens": -1}, "stream": True}}, 422),
        ({"generation": {"sampling_params": {"max_new_tokens": 4, "n": 2}, "stream": True}}, 422),
        (
            {
                "generation": {
                    "sampling_params": {"max_new_tokens": 4, "beam_width": 2},
                    "stream": True,
                }
            },
            422,
        ),
        ({"generation": {"sampling_params": {"max_new_tokens": 4}, "stream": 1}}, 400),
        (
            {
                "generation": {
                    "sampling_params": {"max_new_tokens": 4},
                    "stream": True,
                    "input_ids": [1],
                }
            },
            422,
        ),
        (
            {
                "generation": {
                    "sampling_params": {"max_new_tokens": 4},
                    "stream": True,
                    "return_prompt_token_ids": 1,
                }
            },
            422,
        ),
    ],
)
def test_invalid_requests_are_errors_not_silently_changed(change, status):
    """Verify invalid request options fail with their expected status before upstream calls.

    Input: request-field changes and expected HTTP status. Output: None.
    """
    async def check(service, api, seen):
        """Submit the changed request and assert rejection without stored or upstream work.

        Input: service, API client, upstream records and captured change/status. Output: None.
        """
        req = body()
        req.update(change)
        response = await api.post("/prepare", json=req)
        assert response.status_code == status, response.text
        assert not seen and not service.store.entries

    asyncio.run(run_case(check))


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e9999"])
def test_invalid_json_number_never_reaches_sglang(number):
    """Verify nonfinite or overflowing JSON numbers are rejected before tokenization.

    Input: a malformed JSON numeric literal. Output: None if the request is rejected.
    """
    async def check(service, api, seen):
        """Send a raw invalid-number payload and check that no upstream request occurs.

        Input: service, API client, upstream records and captured numeric literal. Output: None.
        """
        raw = json.dumps(body()).replace('"temperature": 0', '"temperature": ' + number)
        response = await api.post(
            "/prepare", content=raw, headers={"content-type": "application/json"}
        )
        assert response.status_code == 400 and not seen

    asyncio.run(run_case(check))


def test_oversized_input_and_oversized_cache_item_are_explicit():
    """Verify oversized incoming bodies and prepared results produce explicit HTTP errors.

    Input: no arguments; configures small body/cache limits. Output: None.
    """
    async def check(service, api, seen):
        """Assert body-size rejection before tokenization and cache-size rejection afterward.

        Input: service, API client and recorded upstream requests. Output: None.
        """
        response = await api.post(
            "/prepare", content=b"x" * 1025, headers={"content-type": "application/json"}
        )
        assert response.status_code == 413 and not seen
        response = await api.post("/prepare", json=body())
        assert response.status_code == 413 and len(seen) == 1

    asyncio.run(run_case(check, settings=config(max_body_bytes=1024, max_staging_bytes=1)))


def test_bad_forward_target_keeps_input_and_release_does_not_generate():
    """Verify target validation preserves input and release never submits generation.

    Input: no arguments; uses invalid endpoint/config variants. Output: None.
    """
    async def check(service, api, seen):
        """Check rejected forwards and repeated release against the same stored result.

        Input: service, API client and recorded upstream requests. Output: None.
        """
        info = (await api.post("/prepare", json=body())).json()
        assert (
            await api.post("/forward", json=forward(info, "http://wrong/generate"))
        ).status_code == 403
        assert info["ref"] in service.store.entries
        bad = forward(info)
        bad["target"]["config_id"] = "wrong"
        assert (await api.post("/forward", json=bad)).status_code == 409
        assert (await api.post("/release", json=identity(info))).json()["status"] == "released"
        assert (await api.post("/release", json=identity(info))).json()["status"] == "not_found"
        assert len(seen) == 1 and service.store.bytes_used == 0

    asyncio.run(run_case(check))


@pytest.mark.parametrize("concurrency", [32, 128, 256, 512])
def test_burst_has_no_admission_rejections_and_every_request_is_processed(concurrency):
    """Verify a concurrent burst completes once per request without cache/history leftovers.

    Input: the parametrized concurrency level. Output: None if all assertions pass.
    """
    async def check(service, api, seen):
        """Run the captured burst and verify unique generation IDs and empty input storage.

        Input: service, API client, upstream records and captured concurrency. Output: None.
        """
        async def one(i):
            """Prepare and forward one uniquely numbered request in the test burst.

            Input: request index and captured API client. Output: None after status checks.
            """
            prepared = await api.post("/prepare", json=body(str(i)))
            assert prepared.status_code == 200, prepared.text
            response = await api.post("/forward", json=forward(prepared.json()))
            assert response.status_code == 200

        await asyncio.gather(*(one(i) for i in range(concurrency)))
        generated = [v["rid"] for path, v in seen if path == "/generate"]
        assert len(generated) == len(set(generated)) == concurrency
        assert service.store.bytes_used == service.upload_bytes == 0
        assert not service.store.entries and service.store.evicted == 0

    asyncio.run(run_case(check))


@pytest.mark.parametrize(
    "error,status",
    [(aiohttp.ClientConnectionError("offline"), 502), (TimeoutError("timeout"), 504)],
)
def test_forward_error_cleans_input_and_does_not_retry(error, status):
    """Verify a failed upstream send releases input and is not retried.

    Input: injected transport exception and expected HTTP status. Output: None.
    """
    async def check(service, api, seen):
        """Inject a send failure and assert one attempt, the error response and cleanup.

        Input: service, API client, upstream records and captured error/status. Output: None.
        """
        info = (await api.post("/prepare", json=body())).json()
        with patch.object(service.client, "post", side_effect=error) as submit:
            response = await api.post("/forward", json=forward(info))
        assert submit.call_count == 1
        assert response.status_code == status
        assert response.json()["error"]["forward_started"] is True
        assert not service.store.entries and service.upload_bytes == 0
        assert (await api.post("/forward", json=forward(info))).status_code == 404
        assert len(seen) == 1

    asyncio.run(run_case(check))


def test_sglang_tokenization_failure_is_not_dropped_or_retried():
    """Verify an upstream tokenization failure reaches the caller without retry or storage.

    Input: no arguments; injects a 503 tokenization response. Output: None.
    """
    async def broken(req):
        """Return a synthetic unavailable response from the fake tokenizer.

        Input: upstream request, unused. Output: an httpx Response with HTTP 503.
        """
        return httpx.Response(503, json={"error": "unavailable"})

    async def check(service, api, seen):
        """Assert that tokenization fails visibly after exactly one upstream request.

        Input: service, API client and recorded upstream requests. Output: None.
        """
        response = await api.post("/prepare", json=body())
        assert response.status_code == 503 and response.json()["error"]["code"] == "TOKENIZE_ERROR"
        assert len(seen) == 1 and not service.store.entries

    asyncio.run(run_case(check, tokenize_override=broken))


@pytest.mark.parametrize(
    "native",
    [
        {"tokens": [], "count": 0},
        {"tokens": [1], "count": 2},
        {"error": "bad"},
        {"tokens": "bad", "count": 3},
    ],
)
def test_invalid_native_result_is_not_stored(native):
    """Verify malformed tokenization results cannot become prepared cache entries.

    Input: an invalid native response dictionary. Output: None if assertions pass.
    """
    async def invalid(req):
        """Return the captured malformed tokenization result with a successful HTTP status.

        Input: upstream request, unused, and captured native dict. Output: an httpx Response.
        """
        return httpx.Response(200, json=native)

    async def check(service, api, seen):
        """Assert malformed upstream token data produces HTTP 502 and no stored entry.

        Input: service, API client and recorded upstream requests. Output: None.
        """
        response = await api.post("/prepare", json=body())
        assert response.status_code == 502 and not service.store.entries

    asyncio.run(run_case(check, tokenize_override=invalid))


def test_context_limit_is_checked_without_changing_output_budget():
    """Verify input-plus-output overflow is rejected rather than silently shrinking the budget.

    Input: no arguments; submits a context-exceeding output budget. Output: None.
    """
    async def check(service, api, seen):
        """Assert a context-overflow request cannot be cached or submitted for generation.

        Input: service, API client and recorded upstream requests. Output: None.
        """
        req = body()
        req["generation"]["sampling_params"]["max_new_tokens"] = 4096
        assert (await api.post("/prepare", json=req)).status_code == 413
        assert not service.store.entries and all(path == "/v1/tokenize" for path, _ in seen)

    asyncio.run(run_case(check))


def test_native_error_stream_is_preserved_and_nonstream_buffer_is_bounded():
    """Verify streamed upstream errors pass through and nonstream responses obey their cap.

    Input: no arguments; uses fixed error events and an eight-byte response cap. Output: None.
    """
    events = b'data: {"error":{"message":"native error"}}\n\ndata: [DONE]\n\n'

    async def error_stream(req):
        """Supply a fake native error event stream as a successful HTTP response.

        Input: upstream request, unused, and captured event bytes. Output: an httpx Response.
        """
        return httpx.Response(200, content=events, headers={"content-type": "text/event-stream"})

    async def check(service, api, seen):
        """Check stream passthrough and nonstream overflow cleanup using the same events.

        Input: service, API client, upstream records and captured events. Output: None.
        """
        info = (await api.post("/prepare", json=body(stream=True))).json()
        result = await api.post("/forward", json=forward(info))
        assert result.content == events and service.upload_bytes == 0
        info = (await api.post("/prepare", json=body())).json()
        result = await api.post("/forward", json=forward(info))
        assert result.status_code == 502 and not service.store.entries and service.upload_bytes == 0

    asyncio.run(
        run_case(check, generate_override=error_stream, settings=config(max_response_bytes=8))
    )


def test_cancellation_during_tokenization_never_stores_or_generates():
    """Verify cancelling preparation before tokenization completes leaves no input or generation.

    Input: no arguments; pauses tokenization with async events. Output: None.
    """
    async def check(service, api, seen):
        """Cancel a blocked prepare call and assert no cache or upstream work remains.

        Input: service, API client and recorded upstream requests. Output: None.
        """
        entered, finish = asyncio.Event(), asyncio.Event()
        original = service.tokenize

        async def blocked(inp):
            """Pause tokenization at a test-controlled event before calling the original method.

            Input: native input dict and captured events/original method.
            Output: tokenization JSON bytes if unblocked; propagates cancellation otherwise.
            """
            entered.set()
            await finish.wait()
            return await original(inp)

        service.tokenize = blocked
        task = asyncio.create_task(api.post("/prepare", json=body()))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not seen and not service.store.entries

    asyncio.run(run_case(check))


def test_real_tcp_upload_frees_data_before_remote_first_response():
    """Verify token-body memory is released before a real TCP upstream sends response headers.

    Input: no arguments; creates an isolated delayed-response server. Output: None.
    """
    async def scenario():
        """Exercise a large upload while withholding upstream output and inspect memory counters.

        Input: no arguments; owns its service, clients and test server. Output: None.
        """
        received, allow_output = asyncio.Event(), asyncio.Event()
        tasks = set()

        async def worker(reader, writer):
            """Serve fake tokenization or delayed generation on one raw HTTP connection.

            Input: asyncio stream reader/writer and captured synchronization events.
            Output: None; writes an HTTP response and closes the connection.
            """
            tasks.add(asyncio.current_task())
            try:
                head = await reader.readuntil(b"\r\n\r\n")
                headers = dict(
                    line.lower().split(b":", 1) for line in head.split(b"\r\n")[1:] if b":" in line
                )
                data = json.loads(await reader.readexactly(int(headers[b"content-length"])))
                if b"/v1/tokenize" in head.split(b"\r\n")[0]:
                    raw = json.dumps(
                        {"tokens": list(range(20000)), "count": 20000, "max_model_len": 22000}
                    ).encode()
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
                        + str(len(raw)).encode()
                        + b"\r\n\r\n"
                        + raw
                    )
                else:
                    assert data["input_ids"] == list(range(20000))
                    received.set()
                    await allow_output.wait()
                    writer.write(
                        b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {"text":"ok"}\n\ndata: [DONE]\n\n'
                    )
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(worker, "127.0.0.1", 0)
        base = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        settings = config(
            tokenize_endpoint=base + "/v1/tokenize",
            targets={base + "/generate": "test"},
            context_length=22000,
        )
        service = TokenizerService(settings, reference_hash)
        try:
            async with service.lifespan(service.app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=service.app), base_url="http://proxy"
                ) as api:
                    info = (await api.post("/prepare", json=body(stream=True))).json()
                    task = asyncio.create_task(
                        api.post("/forward", json=forward(info, base + "/generate"))
                    )
                    try:
                        await asyncio.wait_for(received.wait(), 3)
                        await asyncio.sleep(0)
                        assert service.store.bytes_used == service.upload_bytes == 0
                        assert not service.store.entries and not task.done()
                        assert (await api.get("/health")).status_code == 200
                    finally:
                        allow_output.set()
                    response = await asyncio.wait_for(task, 3)
                    assert b"[DONE]" in response.content
        finally:
            allow_output.set()
            server.close()
            await server.wait_closed()
            if tasks:
                await asyncio.gather(*tasks)

    asyncio.run(scenario())


def test_stream_disconnect_closes_the_upstream():
    """Verify a downstream disconnect closes upstream output and releases retained input.

    Input: no arguments; uses a synthetic upstream stream and ASGI disconnect. Output: None.
    """
    from starlette.requests import ClientDisconnect

    class Source:
        """Mimic the upstream response methods required by ForwardResponse.

        Input: no constructor arguments. Output: a Source instance with cleanup flags.
        """
        status, closed, waited = 200, False, False

        @property
        def content(self):
            """Expose this fake response as its own content reader.

            Input: this Source instance. Output: the same instance.
            """
            return self

        async def iter_any(self):
            """Yield the single output chunk used to trigger downstream disconnect handling.

            Input: this fake content source. Output: one yielded bytes chunk.
            """
            yield b"data: first\n\n"

        def close(self):
            """Record that the response-close callback ran.

            Input: this Source instance. Output: None; sets closed to True.
            """
            self.closed = True

        async def wait_for_close(self):
            """Record that asynchronous response cleanup was awaited.

            Input: this Source instance. Output: None; sets waited to True.
            """
            self.waited = True

    async def scenario():
        """Run the streaming response against disconnecting ASGI callbacks and check cleanup.

        Input: no arguments; constructs fake response and upload state. Output: None.
        """
        from evolving_infra.router.frontend.forward import Upload
        from types import SimpleNamespace

        source = Source()
        counter = SimpleNamespace(upload_bytes=0, peak_upload_bytes=0)
        upload = Upload(b"unused", counter, "disconnect")
        response = ForwardResponse(source, {}, upload)

        async def send(message):
            """Simulate a client connection failing when the response body is sent.

            Input: an ASGI message. Output: None for headers; raises OSError for body messages.
            """
            if message["type"] == "http.response.body":
                raise OSError("disconnected")

        async def receive():
            """Supply the synthetic downstream-disconnect event.

            Input: no arguments. Output: an ASGI http.disconnect message dictionary.
            """
            return {"type": "http.disconnect"}

        with pytest.raises(ClientDisconnect):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        assert source.closed and source.waited and counter.upload_bytes == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("salt", [None, "example-salt"])
@pytest.mark.parametrize("page_size", [1, 4, 16])
def test_native_hash_matches_engine_wrapper_and_independent_reference(salt, page_size):
    """Verify native hashes match both reference implementations without initializing CUDA.

    Input: optional salt, page size and opt-in native-test environment.
    Output: None if checks pass; skips when native testing is not enabled.
    """
    if not os.environ.get("TOKENIZER_TEST_NATIVE"):
        pytest.skip("Set TOKENIZER_TEST_NATIVE=1 in an environment with decoupled-engine installed")
    import torch
    from decoupled_engine.srt.mem_cache.utils import get_hash_str

    settings = config(cache_salt=salt, page_size=page_size)
    hasher = NativeHasher(settings)
    ids = [10, 511, 154879, 25, 777, 22, 456]
    seed = hashlib.sha256(b"sglang-cache-salt-v1\0" + salt.encode()).digest() if salt else b""
    assert hasher(ids) == reference_hash(ids, page_size, seed)
    assert hasher(ids) == get_hash_str(ids, seed.hex() if seed else None, page_size=page_size)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("return_ids", [None, False, True])
def test_prepared_input_ids_preserve_exact_ids_without_tokenizing(return_ids):
    tokens = [0, 4294967295, 12, 111, 12]
    async def generate(req):
        payload = await req.json()
        result = {"text": "ok"}
        if payload["return_prompt_token_ids"]:
            result["prompt_token_ids"] = payload["input_ids"]
        return httpx.Response(200, json=result)
    async def check(service, api, seen):
        request = body()
        request["input"] = {"model": "/model", "input_ids": tokens}
        if return_ids is not None:
            request["generation"]["return_prompt_token_ids"] = return_ids
        with patch.object(service, "tokenize", side_effect=AssertionError("must not tokenize supplied IDs")):
            result = await api.post("/prepare", json=request)
        assert result.status_code == 200, result.text
        info = result.json()
        assert info["prompt_tokens"] == len(tokens) and "input_ids" not in info
        assert [b["hash"] for b in info["content"]["blocks"]] == reference_hash(tokens)
        assert not seen
        response = await api.post("/forward", json=forward(info))
        assert response.status_code == 200
        assert len(seen) == 1 and seen[0][1]["input_ids"] == tokens
        assert seen[0][1]["return_prompt_token_ids"] is (True if return_ids else False)
        if return_ids:
            assert response.json()["prompt_token_ids"] == tokens
        assert service.upload_bytes == service.store.bytes_used == 0
    asyncio.run(run_case(check, generate_override=generate))


@pytest.mark.parametrize("input_value,status", [
    ({"input_ids": []}, 400),
    ({"input_ids": [True]}, 400),
    ({"input_ids": [-1]}, 400),
    ({"input_ids": [2**32]}, 400),
    ({"input_ids": [1.0]}, 400),
    ({"input_ids": ["1"]}, 400),
    ({"input_ids": "not an array"}, 400),
    ({"input_ids": [1], "prompt": "mixed", "add_special_tokens": False}, 400),
    ({"input_ids": [1], "messages": [{"role": "user", "content": "mixed"}]}, 400),
    ({"input_ids": [1], "add_special_tokens": False}, 422),
    ({"input_ids": [1], "model": "/wrong-model"}, 409),
])
def test_prepared_input_ids_reject_invalid_tokens_or_mixed_inputs(input_value, status):
    async def check(service, api, seen):
        request = body()
        request["input"] = input_value
        response = await api.post("/prepare", json=request)
        assert response.status_code == status, response.text
        assert not seen and not service.store.entries
    asyncio.run(run_case(check))


def test_prepared_input_ids_obey_the_same_input_plus_output_context_budget():
    async def check(service, api, seen):
        request = body()
        request["input"] = {"input_ids": [1] * 4093}
        response = await api.post("/prepare", json=request)
        assert response.status_code == 413
        assert not seen and not service.store.entries
        assert request["generation"]["sampling_params"]["max_new_tokens"] == 4
    asyncio.run(run_case(check))


def test_text_prepare_also_preserves_explicit_return_prompt_ids():
    async def check(service, api, seen):
        request = body()
        request["generation"]["return_prompt_token_ids"] = True
        prepared = await api.post("/prepare", json=request)
        assert prepared.status_code == 200
        assert "input_ids" not in prepared.json()
        result = await api.post("/forward", json=forward(prepared.json()))
        assert result.status_code == 200 and seen[-1][1]["return_prompt_token_ids"] is True
    asyncio.run(run_case(check))
