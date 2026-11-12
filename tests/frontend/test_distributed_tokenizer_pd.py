# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""CPU protocol tests with real local TCP upstreams; these are not GPU benchmarks."""
import asyncio
import json
import sys
import types
from contextlib import asynccontextmanager
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer
import httpx
import pytest

from evolving_infra.router.frontend.server import TokenizerService
from evolving_infra.router.frontend.forward import SharedBody, SharedUpload
from evolving_infra.router.frontend.tokenizer import DynamoHasher, hash_count, hash_spec
from test_distributed_tokenizer import body, config, identity, reference_hash


async def delivered(check):
    """Cancellation metadata is delivered by the service-owned cleanup task."""
    async with asyncio.timeout(2):
        while not check():
            await asyncio.sleep(.005)


from evolving_infra.router.frontend import _engine

try:
    _engine.image_layout()
    HAS_IMAGE_LAYOUT = True
except ModuleNotFoundError:
    HAS_IMAGE_LAYOUT = False
NO_IMAGE_LAYOUT = "needs the engine's image layout module (install decoupled-engine)"

@asynccontextmanager
async def harness(prefill=None, decode=None, callback=None, token_count=19, callback_http=None,
                  outbox_max_plans=1024, abort_http=None, router=None,
                  image_config=None, public_url=None):
    seen, events = [], []

    async def tokenize(req):
        return web.json_response({"tokens": list(range(token_count)), "count": token_count})

    async def invoke(req):
        value = await req.json()
        seen.append((req.path, value))
        handler = prefill if req.path == "/p/generate" else decode
        if handler:
            return await handler(req, value)
        return web.json_response({"text": req.path, "meta_info": {"finish_reason": {"type": "length"}}})

    async def report(req):
        assert req.headers["authorization"] == "Bearer test-secret"
        value = await req.json()
        events.append(value)
        if callback_http:
            return await callback_http(req, value)
        if callback:
            return await callback(value)
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/tokenize", tokenize)
    app.router.add_post("/p/generate", invoke)
    app.router.add_post("/d/generate", invoke)
    app.router.add_post("/events", report)
    if abort_http is not None:
        app.router.add_post("/p/abort_request", abort_http)
        app.router.add_post("/d/abort_request", abort_http)
    for path, handler in (router or {}).items():  # Backend routes beside /events
        app.router.add_post(path, handler)
    async with TestServer(app) as upstream:
        base = str(upstream.make_url("/")).rstrip("/")
        settings = config(
            tokenize_endpoint=base + "/tokenize",
            targets={base + "/p/generate": "test", base + "/d/generate": "test"},
            callback_targets=[base + "/events"], context_length=max(token_count + 10, 4096),
            callback_outbox_max_plans=outbox_max_plans,
            image_config=image_config, public_url=public_url,
        )
        service = TokenizerService(settings, reference_hash)
        async with service.lifespan(service.app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service.app),
                                         base_url="http://tokenizer") as api:
                yield service, api, base, seen, events


def plan(info, base):
    def target(leg):
        return {"transport": "sglang_generate", "endpoint": base + f"/{leg}/generate", "config_id": "test"}
    return {
        **identity(info), "reservation_id": "reserved-" + info["request_id"], "transfer_id": 77,
        "prefill": {"target": target("p"), "kv_export": {"transfer_id": 77, "reader_addr": "tcp://d:10"},
                    "pull_hint": {"source": "cpu-owner"}},
        "decode": {"target": target("d"), "kv_import": {"transfer_id": 77, "owner_addr": "tcp://p:10"}},
        "callback": {"endpoint": base + "/events", "token": "test-secret"},
    }


def test_pd_uploads_concurrently_releases_input_before_compute_and_returns_only_d():
    async def scenario():
        received = {"p": asyncio.Event(), "d": asyncio.Event()}
        release = asyncio.Event()

        async def engine(req, value):
            received[req.path[1]].set()
            await release.wait()
            return web.json_response({"text": req.path, "meta_info": {"finish_reason": {"type": "length"}}})

        async with harness(engine, engine, token_count=20000) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body())).json()
            entry = service.store.entries[info["ref"]]
            with patch.object(service.store, "take", wraps=service.store.take) as take:
                call = asyncio.create_task(api.post("/forward_pd", json=plan(info, base)))
                try:
                    await asyncio.wait_for(asyncio.gather(*(event.wait() for event in received.values())), 3)
                    await asyncio.sleep(0)
                    assert not call.done()  # Both uploads finish before either P or D computes.
                    assert take.call_count == 1
                    assert service.store.bytes_used == service.upload_bytes == 0
                    assert entry.data == b""
                finally:
                    release.set()
                response = await asyncio.wait_for(call, 3)
            assert response.status_code == 200 and response.json()["text"] == "/d/generate"
            p, d = (next(v for path, v in seen if path == f"/{leg}/generate") for leg in ("p", "d"))
            assert p["input_ids"] == d["input_ids"] == list(range(20000))
            assert p["sampling_params"] == d["sampling_params"] == body()["generation"]["sampling_params"]
            assert p["stream"] is False
            assert "kv_import" not in p and "kv_export" not in d and "kv_pull_hint" not in d
            assert p["kv_pull_hint"] == {"source": "cpu-owner"}
            assert {(e["leg"], e["state"]) for e in events} == {
                (leg, state) for leg in ("prefill", "decode") for state in ("uploaded", "completed")
            } | {("prefill", "claimed")}
            assert all(e["transfer_id"] == 77 and e["reservation_id"] == "reserved-id" for e in events)
            assert (await api.post("/forward_pd", json=plan(info, base))).status_code == 404
            assert len(seen) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("attempt_id", [1, 2**64 - 1])
@pytest.mark.parametrize("kv_session", [None, {"session_id": "return-session", "version": 7}])
@pytest.mark.parametrize("host_return", [None, {
    "destination_servicer_addr": "p:18100", "source_servicer_addr": "d:19100",
    "timeout_ms": 10000, "max_tokens": 65536,
}])
def test_pd_preserves_complete_attempt_marks_on_both_engine_requests(attempt_id, host_return, kv_session):
    async def scenario():
        async with harness() as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body())).json()
            request = plan(info, base)
            if kv_session is not None:
                request["prefill"]["kv_session"] = kv_session
                request["decode"]["kv_session"] = kv_session
            export = request["prefill"]["kv_export"]
            export.update(attempt_id=attempt_id, reader_worker=base + "/d")
            imported = request["decode"]["kv_import"]
            imported.update(attempt_id=attempt_id, owner_worker=base + "/p",
                            expected_tokens=19, cache_policy="retain")
            if host_return is not None:
                imported["host_return"] = host_return
            response = await api.post("/forward_pd", json=request)
            assert response.status_code == 200
            assert len(seen) == 2
            p, d = (next(v for path, v in seen if path == f"/{leg}/generate")
                    for leg in ("p", "d"))
            assert p["kv_export"] == export
            assert d["kv_import"] == imported
            assert p.get("kv_session") == d.get("kv_session") == kv_session
            assert type(p["kv_export"]["attempt_id"]) is int
            assert type(d["kv_import"]["attempt_id"]) is int
            assert "kv_import" not in p and "kv_export" not in d
            assert service.store.bytes_used == service.upload_bytes == 0
    asyncio.run(scenario())


def test_shared_input_is_not_freed_when_only_one_uploader_finishes():
    async def scenario():
        service = types.SimpleNamespace(upload_bytes=0, peak_upload_bytes=0)
        shared = SharedBody(b'large common input', service)
        p, d = SharedUpload(shared, b'p', lambda: None), SharedUpload(shared, b'd', lambda: None)
        assert p.common is d.common
        await p.aclose()
        assert shared.data and service.upload_bytes == len(shared.data) + 1
        assert b''.join([part async for part in d]) == b'large common inputd'
        await d.aclose()
        assert shared.data == b'' and service.upload_bytes == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["target", "callback", "transfer"])
def test_plan_validation_happens_before_consuming_input(failure):
    async def scenario():
        async with harness() as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body())).json()
            req = plan(info, base)
            if failure == "target":
                req["decode"]["target"]["endpoint"] = "http://forbidden/generate"
            elif failure == "callback":
                req["callback"]["endpoint"] = "http://forbidden/events"
            else:
                req["decode"]["kv_import"]["transfer_id"] = 88
            result = await api.post("/forward_pd", json=req)
            assert result.status_code == (400 if failure == "transfer" else 403)
            assert info["ref"] in service.store.entries and not seen and not events
    asyncio.run(scenario())


@pytest.mark.parametrize("failed_leg", ["p", "d"])
def test_leg_submission_failure_cleans_both_sides_and_does_not_retry(failed_leg):
    async def scenario():
        release = asyncio.Event()
        async def engine(req, value):
            if req.path == f"/{failed_leg}/generate":
                return web.json_response({"error": "unavailable"}, status=503)
            await release.wait()
            return web.json_response({"text": "unused", "meta_info": {"finish_reason": {"type": "length"}}})
        async with harness(engine, engine) as (service, api, base, seen, events):
            try:
                info = (await api.post("/prepare", json=body())).json()
                response = await asyncio.wait_for(api.post("/forward_pd", json=plan(info, base)), 3)
                assert response.status_code == 502
                await delivered(lambda: len(seen) == 2 and
                                {e["leg"] for e in events if e["state"] in ("failed", "cancelled")}
                                == {"prefill", "decode"})
                assert service.upload_bytes == service.store.bytes_used == 0
                assert not service.store.entries
                assert len(seen) == 2
                assert any(e["leg"] == ("prefill" if failed_leg == "p" else "decode") and
                           e["state"] == "failed" for e in events)
                assert {e["leg"] for e in events if e["state"] in ("failed", "cancelled")} == {"prefill", "decode"}
                assert not any(e.get("terminal_response") is True for e in events)
                duplicate = await api.post("/forward_pd", json=plan(info, base))
                assert duplicate.status_code == 409
                assert duplicate.json()["error"]["code"] == "PLAN_ALREADY_FORWARDED"
                assert len(seen) == 2
                release.set()
                await delivered(lambda: not service.pd_exchanges)
                assert (await api.post("/forward_pd", json=plan(info, base))).status_code == 404
                assert len(seen) == 2
            finally:
                release.set()
    asyncio.run(scenario())


def test_cancellation_while_both_engines_are_running_reports_and_cleans():
    async def scenario():
        submitted, release = asyncio.Event(), asyncio.Event()
        count = 0
        async def engine(req, value):
            nonlocal count
            count += 1
            if count == 2:
                submitted.set()
            await release.wait()
            return web.json_response({"text": "unused", "meta_info": {"finish_reason": {"type": "length"}}})
        async with harness(engine, engine) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body())).json()
            task = asyncio.create_task(api.post("/forward_pd", json=plan(info, base)))
            try:
                await asyncio.wait_for(submitted.wait(), 3)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                await delivered(lambda: {e["leg"] for e in events if e["state"] == "cancelled"}
                                == {"prefill", "decode"})
                assert service.upload_bytes == service.store.bytes_used == 0
                assert {e["leg"] for e in events if e["state"] == "cancelled"} == {"prefill", "decode"}
                assert not any(e.get("terminal_response") is True for e in events)
            finally:
                release.set()
    asyncio.run(scenario())


def test_callback_retry_is_idempotent_and_never_resubmits_generate():
    async def scenario():
        counts = {}
        async def callback(value):
            key = value["event_id"]
            counts[key] = counts.get(key, 0) + 1
            return web.json_response({}, status=503 if counts[key] == 1 else 200)
        async with harness(callback=callback) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body())).json()
            response = await api.post("/forward_pd", json=plan(info, base))
            assert response.status_code == 200
            assert len(counts) == 5 and set(counts.values()) == {2}
            assert len(seen) == 2 and service.callback_failures == 0
    asyncio.run(scenario())


def test_pd_burst_has_one_p_and_one_d_submission_per_request():
    async def scenario():
        async with harness() as (service, api, base, seen, events):
            async def one(i):
                info = (await api.post("/prepare", json=body(str(i)))).json()
                result = await api.post("/forward_pd", json=plan(info, base))
                assert result.status_code == 200
            await asyncio.gather(*(one(i) for i in range(64)))
            assert len(seen) == 128
            assert len({(path, req["rid"]) for path, req in seen}) == 128
            assert len(events) == 320 and len({e["event_id"] for e in events}) == 320
            assert service.upload_bytes == service.store.bytes_used == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("eagle", [False, True])
def test_dynamo_hash_delegates_exact_flags_and_emits_signed_query_values(eagle):
    calls = []
    def native(tokens, block_size, **options):
        calls.append((tokens, block_size, options))
        count = max(0, len(tokens) - int(options["is_eagle"])) // block_size
        return [2**63 + 3] * count
    module = types.ModuleType("dynamo.llm")
    module.compute_block_hash_for_seq = native
    settings = config(hash_mode="dynamo", is_eagle=eagle, cache_salt="session-1")
    with patch.dict(sys.modules, {"dynamo.llm": module}):
        hasher = DynamoHasher(settings)
        values = hasher(list(range(8)))
    assert values == [-2**63 + 3] * hash_count(settings, 8)
    assert calls[-1] == (list(range(8)), 4, {"is_eagle": eagle, "cache_namespace": "session-1"})
    assert hash_spec(settings)["partial_blocks"] is False
    assert hash_spec(settings)["encoding"] == "signed_i64"


@pytest.mark.parametrize("failed_leg", ["p", "d"])
def test_http_200_engine_abort_is_failure_not_completion(failed_leg):
    async def engine(req, value):
        reason = "abort" if req.path == f"/{failed_leg}/generate" else "length"
        return web.json_response({"meta_info": {"finish_reason": {"type": reason}}})
    async def scenario():
        async with harness(engine, engine) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body())).json()
            response = await api.post("/forward_pd", json=plan(info, base))
            assert response.status_code == 502
            leg = "prefill" if failed_leg == "p" else "decode"
            await delivered(lambda: any(e["leg"] == leg and e.get("terminal_response") for e in events))
            assert any(e["leg"] == leg and e["state"] == "failed" for e in events)
            assert any(e["leg"] == leg and e["state"] == "failed" and e.get("terminal_response") is True
                       for e in events)
            assert not any(e["leg"] != leg and e.get("terminal_response") is True for e in events)
            assert not any(e["leg"] == leg and e["state"] == "completed" for e in events)
            assert service.upload_bytes == 0
    asyncio.run(scenario())


def test_decode_abort_is_reported_without_waiting_for_prefill_response():
    async def scenario():
        release = asyncio.Event()

        async def prefill(req, value):
            await release.wait()
            return web.json_response({"text": "prefill", "meta_info": {"finish_reason": {"type": "length"}}})

        async def decode(req, value):
            return web.json_response({"meta_info": {"finish_reason": {"type": "abort"}}})

        async with harness(prefill, decode) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body())).json()
            try:
                response = await asyncio.wait_for(api.post("/forward_pd", json=plan(info, base)), 1)
                assert response.status_code == 502
                await delivered(lambda: any(e["leg"] == "decode" and e.get("terminal_response") for e in events))
                assert any(e["leg"] == "decode" and e["state"] == "failed"
                           and e.get("terminal_response") for e in events)
                assert not any(e["leg"] == "prefill" and e["state"] == "completed" for e in events)
                assert service.upload_bytes == 0
            finally:
                release.set()
    asyncio.run(scenario())


def test_stream_can_deliver_d_output_before_prefill_http_completion():
    async def scenario():
        from starlette.requests import Request
        finish_p = asyncio.Event()
        async def p(req, value):
            await finish_p.wait()
            return web.json_response({"text": "prefill", "meta_info": {"finish_reason": {"type": "length"}}})
        async def d(req, value):
            response = web.StreamResponse(headers={"content-type": "text/event-stream"})
            await response.prepare(req)
            await response.write(b'data: {"text":"first"}\n\n')
            await finish_p.wait()
            await response.write(b'data: {"meta_info":{"finish_reason":{"type":"length"}}}\n\n')
            await response.write(b'data: [DONE]\n\n')
            return response
        async with harness(p, d) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body(stream=True))).json()
            data = json.dumps(plan(info, base)).encode()
            async def receive():
                return {"type": "http.request", "body": data, "more_body": False}
            scope = {"type": "http", "headers": [(b"content-type", b"application/json")],
                     "asgi": {"spec_version": "2.4"}, "method": "POST", "path": "/forward_pd"}
            response = await service.forward_pd(Request(scope, receive))
            output = bytearray()
            async def send(message):
                if message["type"] == "http.response.body":
                    output.extend(message.get("body", b""))
                    if b'first' in output and not finish_p.is_set():
                        assert not any(e["leg"] == "prefill" and e["state"] == "completed" for e in events)
                        finish_p.set()
            try:
                await asyncio.wait_for(response(scope, receive, send), 3)
                assert b'[DONE]' in output
                assert {e["leg"] for e in events if e["state"] == "completed"} == {"prefill", "decode"}
                assert service.upload_bytes == 0
            finally:
                finish_p.set()
    asyncio.run(scenario())


def test_completed_events_carry_engine_token_counts():
    async def scenario():
        async def p(req, value):
            return web.json_response({"text": "p", "meta_info": {
                "prompt_tokens": 19, "cached_tokens": 16, "completion_tokens": 1,
                "finish_reason": {"type": "length"}}})

        async def d(req, value):
            return web.json_response({"text": "d", "meta_info": {
                "prompt_tokens": 19, "cached_tokens": 3, "completion_tokens": 64, "spec_verify_ct": 16,
                "finish_reason": {"type": "length"}}})

        async with harness(p, d) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body())).json()
            assert (await api.post("/forward_pd", json=plan(info, base))).status_code == 200
            done = {e["leg"]: e for e in events if e["state"] == "completed"}
            assert done["prefill"]["usage"] == {"prompt_tokens": 19, "cached_tokens": 16}
            assert done["decode"]["usage"] == {"completion_tokens": 64, "spec_verify_ct": 16}
            assert not any("usage" in e for e in events if e["state"] != "completed")
    asyncio.run(scenario())


def test_stream_completion_reports_counts_from_the_last_event():
    async def scenario():
        async def d(req, value):
            response = web.StreamResponse(headers={"content-type": "text/event-stream"})
            await response.prepare(req)
            await response.write(b'data: {"text":"a","meta_info":{"completion_tokens":1}}\n\n')
            await response.write(
                b'data: {"text":"ab","meta_info":{"completion_tokens":2,"spec_verify_ct":1,'
                b'"finish_reason":{"type":"length"}}}\n\n')
            await response.write(b'data: [DONE]\n\n')
            return response

        async with harness(None, d) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body(stream=True))).json()
            response = await api.post("/forward_pd", json=plan(info, base))
            assert response.status_code == 200 and b"[DONE]" in response.content
            done = {e["leg"]: e for e in events if e["state"] == "completed"}
            assert done["decode"]["usage"] == {"completion_tokens": 2, "spec_verify_ct": 1}
            assert "usage" not in done["prefill"]  # the default engine response has no counts
    asyncio.run(scenario())


def test_stream_disconnect_notifies_backend_and_releases_both_legs():
    async def scenario():
        from starlette.requests import Request, ClientDisconnect
        release = asyncio.Event()
        async def d(req, value):
            response = web.StreamResponse(headers={"content-type": "text/event-stream"})
            await response.prepare(req)
            await response.write(b'data: {"text":"first"}\n\n')
            await release.wait()
            return response
        async with harness(decode=d) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body(stream=True))).json()
            data = json.dumps(plan(info, base)).encode()
            async def receive():
                return {"type": "http.request", "body": data, "more_body": False}
            scope = {"type": "http", "headers": [(b"content-type", b"application/json")],
                     "asgi": {"spec_version": "2.4"}, "method": "POST", "path": "/forward_pd"}
            response = await service.forward_pd(Request(scope, receive))
            async def send(message):
                if message["type"] == "http.response.body":
                    raise OSError("disconnected")
            try:
                with pytest.raises(ClientDisconnect):
                    await response(scope, receive, send)
                await delivered(lambda: any(e["leg"] == "decode" and e["state"] == "cancelled" for e in events))
                assert service.upload_bytes == service.store.bytes_used == 0
                assert any(e["leg"] == "decode" and e["state"] == "cancelled" for e in events)
                assert not any(e["leg"] == "decode" and e["state"] == "completed" for e in events)
            finally:
                release.set()
    asyncio.run(scenario())


def test_callback_failure_is_visible_and_generate_is_never_retried():
    async def scenario():
        async def unavailable(event):
            return web.json_response({}, status=200 if event["state"] == "claimed" else 503)
        async with harness(callback=unavailable) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body())).json()
            response = await api.post("/forward_pd", json=plan(info, base))
            assert response.status_code == 200
            assert len(seen) == 2
            health = (await api.get("/health")).json()
            assert health["callback_failures"] >= 1
            assert health["callback_outbox"]["pending_events"] == 4
            assert health["callback_outbox"]["pending_plans"] == 1
            assert health["callback_outbox"]["active_plans"] == 0
    asyncio.run(scenario())


def test_lost_completion_acks_retry_in_order_after_request_closes():
    async def scenario():
        recovered = asyncio.Event()
        accepted, completions = set(), []

        async def backend(req, event):
            if event["state"] == "completed":
                accepted.add(event["event_id"])  # Idempotent Backend state transition.
                completions.append(event["leg"])
                if not recovered.is_set():
                    req.transport.close()  # State was accepted, but its ACK is lost.
            return web.json_response({"accepted": True})

        async with harness(callback_http=backend) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body())).json()
            response = await api.post("/forward_pd", json=plan(info, base))
            assert response.status_code == 200
            assert service.callback_outbox.summary()["pending_events"] == 2
            assert service.callback_outbox.summary()["active_plans"] == 0
            first = completions[0]
            assert completions == [first, first]  # Other leg cannot bypass this ACK.
            recovered.set()
            async with asyncio.timeout(3):
                while service.callback_outbox.summary()["reserved_plans"]:
                    await asyncio.sleep(0.01)
            assert len(accepted) == 2
            assert completions[:3] == [first] * 3
            assert completions[-1] != first
            assert len(seen) == 2  # Retrying metadata never resends generation.
            assert service.upload_bytes == 0
    asyncio.run(scenario())


def test_outbox_capacity_rejects_before_take_claim_or_engine_submission():
    async def scenario():
        recover = asyncio.Event()

        async def backend(event):
            return web.json_response({}, status=200 if event["state"] == "claimed" or recover.is_set() else 503)

        async with harness(callback=backend, outbox_max_plans=1) as (service, api, base, seen, events):
            first = (await api.post("/prepare", json=body())).json()
            second_body = body()
            second_body["request_id"] = "second"
            second = (await api.post("/prepare", json=second_body)).json()
            assert (await api.post("/forward_pd", json=plan(first, base))).status_code == 200
            response = await api.post("/forward_pd", json=plan(second, base))
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "CALLBACK_BACKLOG"
            assert second["ref"] in service.store.entries
            assert len(seen) == 2
            assert not any(e["request_id"] == "second" for e in events)
            assert (await api.post("/prepare", json=body())).status_code == 503
            recover.set()
            async with asyncio.timeout(4):
                while service.callback_outbox.summary()["reserved_plans"]:
                    await asyncio.sleep(0.01)
            assert (await api.post("/forward_pd", json=plan(second, base))).status_code == 200
            assert len(seen) == 4
    asyncio.run(scenario())


def test_permanently_rejected_callback_is_dropped_and_later_events_still_arrive():
    async def scenario():
        rejected = {}  # request_id -> the one event_id the Backend always rejects

        async def backend(event):
            # The Backend answers 400 for an execution it no longer knows (e.g. after
            # a restart); here it rejects the first non-claim event of every plan.
            if event["state"] != "claimed" and \
                    rejected.setdefault(event["request_id"], event["event_id"]) == event["event_id"]:
                return web.json_response({"error": "unknown execution"}, status=400)
            return web.json_response({"accepted": True})

        async with harness(callback=backend, outbox_max_plans=1) as (service, api, base, seen, events):
            for rid in ("a", "b", "c"):  # One slot: each plan must release it.
                info = (await api.post("/prepare", json=body(rid))).json()
                assert (await api.post("/forward_pd", json=plan(info, base))).status_code == 200
                await delivered(lambda: service.callback_outbox.summary()["reserved_plans"] == 0)
            ids = [e["event_id"] for e in events if e["state"] != "claimed"]
            assert len(ids) == len(set(ids)) == 3 * 4  # Each event is sent exactly once.
            assert {e["state"] for e in events} >= {"uploaded", "completed"}
            assert (await api.get("/health")).json()["callback_dropped"] == 3
    asyncio.run(scenario())


def test_transiently_failing_callback_is_given_up_after_the_retry_bound(monkeypatch):
    from evolving_infra.router.frontend import delivery
    monkeypatch.setattr(delivery, "RETRY_SECONDS", 0.2, raising=False)

    async def scenario():
        async def unavailable(event):
            return web.json_response({}, status=200 if event["state"] == "claimed" else 503)

        async with harness(callback=unavailable, outbox_max_plans=1) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body())).json()
            assert (await api.post("/forward_pd", json=plan(info, base))).status_code == 200
            assert service.callback_outbox.summary()["pending_events"] == 4
            async with asyncio.timeout(5):
                while service.callback_outbox.summary()["reserved_plans"]:
                    await asyncio.sleep(0.01)
            assert service.callback_dropped == 4
            assert (await api.post("/prepare", json=body("next"))).status_code == 200
    asyncio.run(scenario())


def test_truncated_stream_is_not_reported_as_success():
    async def d(req, value):
        return web.Response(body=b'data: {"text":"incomplete"}\n\n',
                            content_type="text/event-stream")
    async def scenario():
        from evolving_infra.router.frontend.store import ProtocolError
        async with harness(decode=d) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body(stream=True))).json()
            # HTTP 200 headers already went out. ASGI cannot turn this into a
            # second error response; it closes the stream with the original cause.
            with pytest.raises(RuntimeError, match="response already started") as failure:
                await api.post("/forward_pd", json=plan(info, base))
            assert isinstance(failure.value.__cause__, ProtocolError)
            assert "Truncated" in str(failure.value.__cause__)
            await delivered(lambda: any(e["leg"] == "decode" and e["state"] == "failed" for e in events))
            assert not any(e["leg"] == "decode" and e["state"] == "completed" for e in events)
            assert any(e["leg"] == "decode" and e["state"] == "failed" for e in events)
            assert service.upload_bytes == 0
    asyncio.run(scenario())


def test_claim_must_be_accepted_before_either_generate_request():
    async def scenario():
        claimed = False
        order = []
        async def callback(event):
            nonlocal claimed
            if event["state"] == "claimed":
                assert event["leg"] == "prefill"
                assert event["event_id"] == "reserved-id:prefill:claimed"
                claimed = True
                order.append("claimed")
            return web.json_response({"ok": True})
        async def engine(req, value):
            assert claimed
            order.append(req.path)
            return web.json_response({"text": "ok", "meta_info": {"finish_reason": {"type": "length"}}})
        async with harness(engine, engine, callback) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body())).json()
            response = await api.post("/forward_pd", json=plan(info, base))
            assert response.status_code == 200
            assert order[0] == "claimed" and len(order) == 3
            assert len([e for e in events if e["state"] == "claimed"]) == 1
            assert service.upload_bytes == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("claim_status", [410, 503])
def test_expired_or_unaccepted_claim_never_sends_generate(claim_status):
    async def callback(event):
        if event["state"] == "claimed":
            return web.json_response({"error": "plan expired or unavailable"}, status=claim_status)
        assert event["state"] == "failed" and event["not_submitted"] is True
        return web.json_response({"accepted": True})
    async def scenario():
        async with harness(callback=callback) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body())).json()
            response = await api.post("/forward_pd", json=plan(info, base))
            assert response.status_code == 502
            assert response.json()["error"]["code"] == "PD_CLAIM_FAILED"
            assert not seen and len(events) == 4
            assert not service.store.entries and service.upload_bytes == service.store.bytes_used == 0
            assert events[0] == events[1]  # Bounded retry is metadata-only and idempotent.
    asyncio.run(scenario())


def test_accepted_claim_with_lost_acks_reports_both_legs_not_submitted():
    async def scenario():
        accepted_claims, released = [], []
        async def callback_http(req, event):
            if event["state"] == "claimed":
                accepted_claims.append(event)  # Backend accepted; ACK connection is lost.
                req.transport.close()
                return web.json_response({"accepted": True})
            assert event["state"] == "failed" and event["not_submitted"] is True
            assert "terminal_response" not in event
            released.append(event["leg"])
            return web.json_response({"accepted": True})
        async with harness(callback_http=callback_http) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body())).json()
            response = await api.post("/forward_pd", json=plan(info, base))
            assert response.status_code == 502 and response.json()["error"]["code"] == "PD_CLAIM_FAILED"
            assert len(accepted_claims) == 2 and accepted_claims[0] == accepted_claims[1]
            assert released == ["prefill", "decode"]
            assert not seen and service.upload_bytes == service.store.bytes_used == 0
    asyncio.run(scenario())


def sse(*values):
    return b"".join(b"data: " + (v if isinstance(v, bytes) else json.dumps(v).encode()) + b"\n\n"
                    for v in values)


def events_of(content):
    return [json.loads(line[5:]) for line in content.split(b"\n")
            if line.startswith(b"data:") and line[5:].strip() != b"[DONE]"]


def resumable_harness(migrating=True, first=None):
    """A D engine that streams 3 of 6 forced tokens and is then aborted, a Backend
    that reports the abort as a migration and places the resumed request, and a
    second D run that streams the remaining tokens."""
    forced = [101, 102, 103, 104, 105, 106]
    placed = []

    async def d(req, value):
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(req)
        if value["rid"] == "id":
            await response.write(first or sse(
                {"text": "a", "output_ids": [101, 102], "prompt_token_ids": list(range(19)),
                 "meta_info": {"id": "id", "prompt_tokens": 19, "cached_tokens": 7,
                               "completion_tokens": 2, "finish_reason": None}},
                {"text": "b", "output_ids": [103],
                 "meta_info": {"id": "id", "prompt_tokens": 19, "cached_tokens": 7,
                               "completion_tokens": 3, "finish_reason": None}},
                {"text": "c", "output_ids": [104],
                 "meta_info": {"id": "id", "prompt_tokens": 19, "cached_tokens": 7, "completion_tokens": 4,
                               "finish_reason": {"type": "abort", "message": "migrated"}}},
                b"[DONE]"))
        else:
            assert value["input_ids"] == list(range(19)) + [101, 102, 103]
            assert value["sampling_params"]["max_new_tokens"] == 3
            assert value["sampling_params"]["forced_output_token_ids"] == [104, 105, 106]
            await response.write(sse(
                {"text": "d", "output_ids": [104, 105], "prompt_token_ids": value["input_ids"],
                 "meta_info": {"id": value["rid"], "prompt_tokens": 22, "cached_tokens": 21,
                               "completion_tokens": 2, "finish_reason": None}},
                {"text": "e", "output_ids": [106],
                 "meta_info": {"id": value["rid"], "prompt_tokens": 22, "cached_tokens": 21,
                               "completion_tokens": 3, "finish_reason": {"type": "length"}}},
                b"[DONE]"))
        return response

    async def migration(req):
        assert req.headers["authorization"] == "Bearer test-secret"
        assert (await req.json()) == {"reservation_id": "reserved-id"}
        return web.json_response({"migrating": migrating})

    async def prepared(req):
        desc = (await req.json())["prepared"]
        placed.append(desc)
        base = str(req.url.origin())
        return web.json_response(plan(desc, base) | {"reservation_id": "reserved-" + desc["request_id"]})

    gen = {"sampling_params": {"max_new_tokens": 6, "temperature": 0, "forced_output_token_ids": forced},
           "stream": True, "return_prompt_token_ids": True}
    return d, {"/migration": migration, "/prepared": prepared}, placed, gen


def test_migrated_decode_resumes_on_a_new_pair_as_one_stream():
    async def scenario():
        d, router, placed, gen = resumable_harness()
        async with harness(decode=d, router=router) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body(stream=True) | {"generation": gen})).json()
            response = await api.post("/forward_pd", json=plan(info, base))
            assert response.status_code == 200
            out = events_of(response.content)
            # The abort event is gone; the resumed events continue the original's counts.
            assert [t for e in out for t in e["output_ids"]] == [101, 102, 103, 104, 105, 106]
            assert [e["meta_info"]["completion_tokens"] for e in out] == [2, 3, 5, 6]
            assert all(e["meta_info"]["prompt_tokens"] == 19 and e["meta_info"]["cached_tokens"] == 7
                       and e["meta_info"]["id"] == "id" for e in out)
            assert "prompt_token_ids" in out[0] and not any("prompt_token_ids" in e for e in out[1:])
            assert out[-1]["meta_info"]["finish_reason"] == {"type": "length"}
            assert response.content.count(b"[DONE]") == 1 and response.content.endswith(b"data: [DONE]\n\n")
            assert [x["request_id"] for x in placed] == ["m3-id"]  # an abort of "id" cannot match it
            assert placed[0]["prompt_tokens"] == 22 and placed[0]["max_new_tokens"] == 3
            assert placed[0]["node_id"] == "cpu-a" and len(placed[0]["content"]["blocks"]) == 6
            resumed = [v for path, v in seen if v["rid"] == "m3-id"]
            assert sorted(v["stream"] for v in resumed) == [False, True]
            assert all(v["return_prompt_token_ids"] is False for v in resumed)  # sent with event 1
            by = lambda rid, leg, state: [e for e in events if e["reservation_id"] == "reserved-" + rid
                                          and e["leg"] == leg and e["state"] == state]
            await delivered(lambda: by("m3-id", "decode", "completed"))
            assert by("id", "decode", "failed")[0]["terminal_response"] is True
            assert not by("id", "decode", "completed") and by("id", "prefill", "completed")
            assert by("m3-id", "prefill", "completed")
            assert by("m3-id", "decode", "completed")[0]["usage"] == {"completion_tokens": 3}
            assert service.upload_bytes == service.store.bytes_used == 0 and not service.pd_exchanges
    asyncio.run(scenario())


@pytest.mark.parametrize("case", ["not_migrating", "cumulative_stream"])
def test_decode_abort_is_resumed_only_for_a_migration_of_an_incremental_stream(case):
    async def scenario():
        from evolving_infra.router.frontend.store import ProtocolError
        cumulative = sse(
            {"text": "a", "output_ids": [101, 102],
             "meta_info": {"prompt_tokens": 19, "cached_tokens": 7, "completion_tokens": 2}},
            {"text": "ab", "output_ids": [101, 102, 103],
             "meta_info": {"prompt_tokens": 19, "cached_tokens": 7, "completion_tokens": 3}},
            {"text": "abc", "output_ids": [101, 102, 103],
             "meta_info": {"completion_tokens": 3, "finish_reason": {"type": "abort"}}},
            b"[DONE]")
        d, router, placed, gen = resumable_harness(
            migrating=case != "not_migrating", first=cumulative if case == "cumulative_stream" else None)
        async with harness(decode=d, router=router) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body(stream=True) | {"generation": gen})).json()
            with pytest.raises(RuntimeError, match="response already started") as failure:
                await api.post("/forward_pd", json=plan(info, base))
            assert isinstance(failure.value.__cause__, ProtocolError)
            assert not placed and all(v["rid"] == "id" for _, v in seen)
            await delivered(lambda: any(e["leg"] == "decode" and e["state"] == "failed" for e in events))
            assert service.upload_bytes == service.store.bytes_used == 0
    asyncio.run(scenario())


def test_decode_migrated_before_its_first_token_resumes_with_the_same_input():
    async def scenario():
        async def d(req, value):
            response = web.StreamResponse(headers={"content-type": "text/event-stream"})
            await response.prepare(req)
            if value["rid"] == "id":
                await response.write(sse({"text": "", "output_ids": [], "meta_info": {
                    "completion_tokens": 0, "finish_reason": {"type": "abort"}}}, b"[DONE]"))
            else:
                assert value["input_ids"] == list(range(19)) and value["return_prompt_token_ids"] is True
                assert value["sampling_params"]["max_new_tokens"] == 6
                await response.write(sse(
                    {"text": "a", "output_ids": [101], "prompt_token_ids": value["input_ids"],
                     "meta_info": {"prompt_tokens": 19, "cached_tokens": 16, "completion_tokens": 1}},
                    {"text": "b", "output_ids": [102, 103, 104, 105, 106], "meta_info": {
                        "prompt_tokens": 19, "cached_tokens": 16, "completion_tokens": 6,
                        "finish_reason": {"type": "length"}}},
                    b"[DONE]"))
            return response
        _, router, placed, gen = resumable_harness()
        async with harness(decode=d, router=router) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body(stream=True) | {"generation": gen})).json()
            response = await api.post("/forward_pd", json=plan(info, base))
            assert response.status_code == 200
            out = events_of(response.content)
            assert [x["request_id"] for x in placed] == ["m0-id"]
            assert out[0]["prompt_token_ids"] == list(range(19))
            assert [e["meta_info"]["completion_tokens"] for e in out] == [1, 6]
            assert out[-1]["meta_info"]["cached_tokens"] == 16
    asyncio.run(scenario())


@pytest.mark.skipif(not HAS_IMAGE_LAYOUT, reason=NO_IMAGE_LAYOUT)
def test_image_whose_header_exceeds_the_pixel_limit_is_rejected():
    import base64
    import struct
    import zlib

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    # A 67-byte PNG whose header claims 20000x20000 pixels.
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 20000, 20000, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(b"\0\0\0\0")) + chunk(b"IEND", b""))
    image_cfg = _engine.image_layout().image_config(dict(
        vision_patch_size=14, vision_downsample_ratio=3, vision_max_n_token=384,
        vision_min_pixels=147456, vision_max_wh_ratio=8), 100)

    async def scenario():
        async with harness(image_config=image_cfg, public_url="http://t") as (service, api, base, seen, events):
            request = body()
            request["input"] = {"input_ids": [11, 100, 12],
                                "image_data": ["data:image/png;base64," + base64.b64encode(png).decode()]}
            response = await api.post("/prepare", json=request)
            assert response.status_code == 413, response.text
            assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"
            assert not service.store.entries
    asyncio.run(scenario())


@pytest.mark.skipif(not HAS_IMAGE_LAYOUT, reason=NO_IMAGE_LAYOUT)
@pytest.mark.parametrize("kind", ["dsv4", "kimi_k25"])
def test_uploaded_images_go_only_to_p_and_remain_available_for_decode_recovery(kind):
    import base64
    import io
    from PIL import Image
    image_config = _engine.image_layout().image_config
    prepare_image_layout = _engine.image_layout().prepare_image_layout

    out = io.BytesIO()
    Image.new('RGB', (32, 32), 'red').save(out, format='PNG')
    png = out.getvalue()
    images = ['data:image/png;base64,' + base64.b64encode(png).decode()]
    image_cfg = image_config(dict(vision_patch_size=14, vision_downsample_ratio=3,
                                 vision_max_n_token=384, vision_min_pixels=147456,
                                 vision_max_wh_ratio=8), 100)
    if kind == "kimi_k25":
        image_cfg = image_config({"model_type": "kimi_k25"}, 100, {
            "patch_size": 14, "merge_kernel_size": 2, "in_patch_limit": 16384,
            "patch_limit_on_one_side": 512, "fixed_output_tokens": None,
            "image_mean": [0.5] * 3, "image_std": [0.5] * 3,
        })

    async def scenario():
        entered, finish = asyncio.Event(), asyncio.Event()

        async def decode(req, value):
            assert 'image_data' not in value
            assert 'prepared_image_layout' in value
            entered.set()
            await finish.wait()
            return web.json_response({'text': 'ok', 'meta_info': {'finish_reason': {'type': 'length'}}})

        async with harness(decode=decode, image_config=image_cfg,
                           public_url='http://tokenizer') as (service, api, base, seen, events):
            req = body()
            req['input'] = {'input_ids': [11, 100, 12], 'image_data': images}
            response = await api.post('/prepare', json=req)
            assert response.status_code == 200, response.text
            info = response.json()
            layout, _, cache_ids, _ = prepare_image_layout([11, 100, 12], images, image_cfg)
            assert info['prompt_tokens'] == len(cache_ids)
            assert [block['hash'] for block in info['content']['blocks']] == reference_hash(cache_ids)
            assert info['content']['hash_spec']['image_config'] == image_cfg
            image_req = {**identity(info), 'image_index': 0}
            assert (await api.post('/image', json=image_req)).content == png
            pending = asyncio.create_task(api.post('/forward_pd', json=plan(info, base)))
            await entered.wait()
            recovery = await api.post('/image', json=image_req)
            assert recovery.status_code == 200 and recovery.content == png
            p_body = next(value for path, value in seen if path == '/p/generate')
            d_body = next(value for path, value in seen if path == '/d/generate')
            assert p_body['image_data'] == images
            assert p_body['prepared_image_layout'] == d_body['prepared_image_layout'] == layout
            assert p_body['input_ids'] == d_body['input_ids'] == [11, 100, 12]
            assert d_body['prepared_image_source']['ref'] == info['ref']
            finish.set()
            result = await pending
            assert result.status_code == 200, result.text
            await delivered(lambda: not service.pd_exchanges)
            assert service.upload_bytes == 0
            assert (await api.post('/image', json=image_req)).status_code == 404

def test_kv_session_router_metadata_reaches_both_real_http_legs():
    """Input: prepared plan tag. Output: exact tag on both engine uploads, unchanged tokens/salt."""

    async def scenario():
        async with harness() as (_, api, base, seen, _):
            prepared = (await api.post("/prepare", json=body())).json()
            envelope = plan(prepared, base)
            metadata = {"session_id": "controller:trajectory", "version": 7}
            envelope["prefill"]["kv_session"] = metadata
            envelope["decode"]["kv_session"] = metadata
            response = await api.post("/forward_pd", json=envelope)
            assert response.status_code == 200, response.text
            assert len(seen) == 2
            assert all(payload["kv_session"] == metadata for _, payload in seen)
            assert seen[0][1]["input_ids"] == seen[1][1]["input_ids"]

    asyncio.run(scenario())
