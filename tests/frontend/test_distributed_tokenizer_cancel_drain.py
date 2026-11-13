# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Real TCP cancellation fixtures; abort ACKs deliberately precede engine termination."""
import asyncio
import json
from unittest.mock import patch

import pytest
import aiohttp
from aiohttp import web
from starlette.requests import ClientDisconnect, Request
from evolving_infra.router.frontend.forward import PDForward

from test_distributed_tokenizer import body
from test_distributed_tokenizer_pd import harness, plan


def terminal(events, leg):
    return any(e["leg"] == leg and (e["state"] == "completed" or
               e["state"] == "failed" and e.get("terminal_response")) for e in events)


async def until(check, seconds=2):
    async with asyncio.timeout(seconds):
        while not check():
            await asyncio.sleep(.005)


def abort_body():
    return {"meta_info": {"finish_reason": {"type": "abort", "message": "aborted by client"}}}


async def response_for(service, api, base):
    info = (await api.post("/prepare", json=body(stream=True))).json()
    data = json.dumps(plan(info, base)).encode()
    scope = {"type": "http", "headers": [(b"content-type", b"application/json")],
             "asgi": {"spec_version": "2.4"}, "method": "POST", "path": "/forward_pd"}
    async def receive():
        return {"type": "http.request", "body": data, "more_body": False}
    return await service.forward_pd(Request(scope, receive)), scope, receive


def test_cancel_before_headers_keeps_original_posts_until_real_abort_terminal():
    async def scenario():
        release, received = asyncio.Event(), asyncio.Event()
        count = 0
        aborts = []
        async def abort(req):
            aborts.append((req.path, await req.json()))
            return web.json_response({"cancellation_requested": True})
        async def backend(event):
            if event["state"] == "cancelled":
                async with aiohttp.ClientSession() as client:
                    for leg in ("p", "d"):
                        async with client.post(base + f"/{leg}/abort_request",
                                               json={"rid": event["request_id"]}) as response:
                            assert response.status == 200
                            await response.read()
            return web.json_response({"accepted": True})
        async def engine(req, value):
            nonlocal count
            count += 1
            if count == 2:
                received.set()
            await release.wait()
            return web.json_response(abort_body())
        async with harness(engine, engine, callback=backend, abort_http=abort) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body())).json()
            call = asyncio.create_task(api.post("/forward_pd", json=plan(info, base)))
            try:
                await asyncio.wait_for(received.wait(), 2)
                call.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(call, 1)
                await until(lambda: {e["leg"] for e in events if e["state"] == "cancelled"}
                            == {"prefill", "decode"})
                await until(lambda: len(aborts) == 4)
                assert {path for path, _ in aborts} == {"/p/abort_request", "/d/abort_request"}
                assert all(value == {"rid": "id"} for _, value in aborts)
                assert not any(terminal(events, leg) for leg in ("prefill", "decode"))
                # Actual HTTP abort ACKs arrived, but the original generates are still held.
                assert len(seen) == 2
                release.set()
                await until(lambda: all(terminal(events, leg) for leg in ("prefill", "decode")))
                await until(lambda: service.callback_outbox.summary()["reserved_plans"] == 0)
                assert len(seen) == 2 and service.upload_bytes == 0
            finally:
                release.set()
    asyncio.run(scenario())


def test_disconnect_in_half_sse_preserves_parser_and_reader_ownership():
    async def scenario():
        release = asyncio.Event()
        whole = b'data: ' + json.dumps(abort_body()).encode() + b'\n\n'
        cut = whole.index(b'abort') + 2
        async def d(req, value):
            response = web.StreamResponse(headers={"content-type": "text/event-stream"})
            await response.prepare(req)
            # Only whole SSE lines are relayed: the caller disconnects on the first
            # event while the abort event is still half parsed.
            await response.write(b'data: {"text":"first"}\n\n' + whole[:cut])
            await release.wait()
            await response.write(whole[cut:])
            return response
        async with harness(decode=d) as (service, api, base, seen, events):
            response, scope, receive = await response_for(service, api, base)
            async def send(message):
                if message["type"] == "http.response.body":
                    raise OSError("caller disconnected inside SSE line")
            try:
                with pytest.raises(ClientDisconnect):
                    await response(scope, receive, send)
                await until(lambda: any(e["leg"] == "decode" and e["state"] == "cancelled" for e in events))
                assert not terminal(events, "decode")
                release.set()
                await until(lambda: terminal(events, "decode"))
                await until(lambda: service.callback_outbox.summary()["reserved_plans"] == 0)
                assert len(seen) == 2 and service.upload_bytes == 0
            finally:
                release.set()
    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [False, True])
def test_eof_or_done_without_terminal_payload_remains_unknown(stream):
    async def d(req, value):
        if stream:
            return web.Response(body=b'data: {"text":"unfinished"}\n\ndata: [DONE]\n\n',
                                content_type="text/event-stream")
        return web.json_response({"text": "unfinished"})
    async def scenario():
        async with harness(decode=d) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body(stream=stream))).json()
            if stream:
                with pytest.raises(RuntimeError, match="response already started"):
                    await api.post("/forward_pd", json=plan(info, base))
            else:
                assert (await api.post("/forward_pd", json=plan(info, base))).status_code == 502
            await until(lambda: service.callback_outbox.summary()["reserved_plans"] == 0)
            assert not terminal(events, "decode")
            assert any(e["leg"] == "decode" and e["state"] == "failed" for e in events)
            assert not service.pd_exchanges and len(seen) == 2
    asyncio.run(scenario())


def test_timeout_retains_unknown_and_holds_bounded_outbox_slot_until_cleanup():
    async def scenario():
        release = asyncio.Event()
        async def d(req, value):
            response = web.StreamResponse(headers={"content-type": "text/event-stream"})
            await response.prepare(req)
            await response.write(b'data: {"text":"first"}\n\n')
            await release.wait()
            return response
        with patch.object(PDForward, "CANCEL_GRACE_SECONDS", .1):
            async with harness(decode=d, outbox_max_plans=1) as (service, api, base, seen, events):
                second = (await api.post("/prepare", json=body("second"))).json()
                response, scope, receive = await response_for(service, api, base)
                exchange = response.exchange
                async def send(message):
                    if message["type"] == "http.response.body":
                        raise OSError("disconnected")
                try:
                    with pytest.raises(ClientDisconnect):
                        await response(scope, receive, send)
                    assert service.callback_outbox.summary()["reserved_plans"] == 1
                    assert (await api.post("/forward_pd", json=plan(second, base))).status_code == 503
                    deadline = exchange.cleanup_deadline
                    await exchange.aclose()
                    assert exchange.cleanup_deadline == deadline
                    await until(lambda: not service.pd_exchanges, seconds=1)
                    assert not terminal(events, "decode")
                    assert exchange.responses["decode"].closed
                    assert service.upload_bytes == 0 and len(seen) == 2
                    assert service.callback_outbox.summary()["reserved_plans"] == 0
                finally:
                    release.set()
    asyncio.run(scenario())


def test_shutdown_keeps_session_alive_until_detached_drain_finishes():
    async def scenario():
        release = asyncio.Event()
        async def d(req, value):
            response = web.StreamResponse(headers={"content-type": "text/event-stream"})
            await response.prepare(req)
            await response.write(b'data: {"text":"first"}\n\n')
            await release.wait()
            await response.write(b'data: ' + json.dumps(abort_body()).encode() + b'\n\n')
            return response
        async with harness(decode=d) as (service, api, base, seen, events):
            response, scope, receive = await response_for(service, api, base)
            async def send(message):
                if message["type"] == "http.response.body":
                    raise OSError("disconnected")
            with pytest.raises(ClientDisconnect):
                await response(scope, receive, send)
            async def during_shutdown():
                await until(lambda: service.callback_outbox.closed)
                assert not service.client.closed
                assert service.pd_exchanges
                release.set()
            observer = asyncio.create_task(during_shutdown())
        await observer
        assert service.client.closed and not service.pd_exchanges
        assert terminal(events, "decode")
        assert service.callback_outbox.summary()["reserved_plans"] == 0
    asyncio.run(scenario())


def test_weak_decode_failure_can_upgrade_after_prefill_abort_without_resubmission():
    async def scenario():
        release = asyncio.Event()
        async def p(req, value):
            return web.json_response(abort_body())
        async def d(req, value):
            await release.wait()
            return web.json_response(abort_body())
        async with harness(p, d) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body())).json()
            try:
                assert (await api.post("/forward_pd", json=plan(info, base))).status_code == 502
                await until(lambda: any(e["leg"] == "decode" and e["state"] == "failed" for e in events))
                assert not terminal(events, "decode")
                release.set()
                await until(lambda: terminal(events, "decode"))
                await until(lambda: not service.pd_exchanges)
                failed = [e for e in events if e["leg"] == "decode" and e["state"] == "failed"]
                assert len(failed) == 2 and failed[0]["event_id"] == failed[1]["event_id"]
                assert not failed[0].get("terminal_response") and failed[1]["terminal_response"]
                assert len(seen) == 2
            finally:
                release.set()
    asyncio.run(scenario())


def test_normal_stream_preserves_bytes_and_finishes_without_response_drain():
    async def scenario():
        output = (b'data: {"text":"token","meta_info":{"finish_reason":null}}\n\n' * 32 +
                  b'data: {"text":"unchanged","meta_info":{"finish_reason":{"type":"length"}}}\n\n'
                  b'data: [DONE]\n\n')
        async def d(req, value):
            return web.Response(body=output, content_type="text/event-stream")
        async with harness(decode=d) as (service, api, base, seen, events):
            response, scope, receive = await response_for(service, api, base)
            result = bytearray()
            async def send(message):
                if message["type"] == "http.response.body":
                    result.extend(message.get("body", b""))
            with patch.object(PDForward, "check_result", wraps=PDForward.check_result) as parse:
                await response(scope, receive, send)
            decode_parses = [call for call in parse.call_args_list if call.args[1] == "decode"]
            assert 1 <= len(decode_parses) <= 2  # No JSON parse per intermediate token.
            assert bytes(result) == output
            assert response.exchange.cleanup_task.done()
            assert not hasattr(response.exchange, "cleanup_deadline")
            assert not service.pd_exchanges
            assert all(terminal(events, leg) for leg in ("prefill", "decode"))
            assert service.upload_bytes == 0 and len(seen) == 2
    asyncio.run(scenario())


@pytest.mark.parametrize("reset", [False, True])
def test_valid_terminal_payload_survives_missing_done_or_socket_reset(reset):
    async def d(req, value):
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(req)
        await response.write(b'data: {"meta_info":{"finish_reason":{"type":"length"}}}\n\n')
        if reset:
            await asyncio.sleep(.02)
            req.transport.close()
        return response
    async def scenario():
        async with harness(decode=d) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body(stream=True))).json()
            with pytest.raises(RuntimeError, match="response already started"):
                await api.post("/forward_pd", json=plan(info, base))
            await until(lambda: terminal(events, "decode"))
            await until(lambda: not service.pd_exchanges)
            assert any(e["leg"] == "decode" and e["state"] == "failed" and e.get("terminal_response")
                       for e in events)
            assert not any(e["leg"] == "decode" and e["state"] == "completed" for e in events)
    asyncio.run(scenario())


def test_nonstream_caller_disconnect_cancels_both_legs():
    import socket
    import uvicorn

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def d(req, value):
            entered.set()
            await release.wait()  # The D engine is still generating.
            return web.json_response({"text": "late", "meta_info": {"finish_reason": {"type": "length"}}})

        async with harness(decode=d) as (service, api, base, seen, events):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            url = "http://127.0.0.1:%d" % sock.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(service.app, lifespan="off", log_level="error"))
            serving = asyncio.create_task(server.serve(sockets=[sock]))
            try:
                await until(lambda: server.started)
                info = (await api.post("/prepare", json=body())).json()
                async with aiohttp.ClientSession() as client:
                    call = asyncio.create_task(client.post(url + "/forward_pd", json=plan(info, base)))
                    await asyncio.wait_for(entered.wait(), 5)
                    call.cancel()  # The caller closes its TCP connection.
                    await asyncio.gather(call, return_exceptions=True)
                await until(lambda: any(e["leg"] == "decode" and e["state"] == "cancelled" for e in events))
                assert any(e["leg"] == "prefill" and e["state"] in ("completed", "cancelled") for e in events)
                release.set()  # The cleanup drains the original D response.
                await until(lambda: not service.pd_exchanges, seconds=5)
                assert len(seen) == 2  # Generation is never resubmitted.
            finally:
                release.set()
                server.should_exit = True
                await asyncio.wait_for(serving, 10)
    asyncio.run(scenario())
