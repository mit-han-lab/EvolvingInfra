"""Late D choose through the distributed tokenizer: one D-leg per candidate.

CPU protocol tests with real local TCP upstreams (fake engines); not GPU tests.
The prefill admits one candidate (its stream starts with generation output) and
releases the others (an abort event with err_type late_d_released on /generate).
"""
import asyncio
from contextlib import asynccontextmanager

from aiohttp import web
from aiohttp.test_utils import TestServer
import httpx
import pytest

from evolving_infra.router.frontend.forward import engine_abort_message, first_data, late_d_outcome
from evolving_infra.router.frontend.protocol import ForwardPD
from evolving_infra.router.frontend.server import TokenizerService
from test_distributed_tokenizer import body, config, identity, reference_hash

RELEASED = (b'data: {"text":"","meta_info":{"completion_tokens":0,"finish_reason":'
            b'{"type":"abort","status_code":409,"err_type":"late_d_released","message":"room 77 released"}}}\n\n'
            b'data: [DONE]\n\n')
WINNER = (b'data: {"text":"a","output_ids":[5],"meta_info":{"completion_tokens":1}}\n\n',
          b'data: {"text":"ab","output_ids":[6],"meta_info":{"completion_tokens":2,"spec_verify_ct":1,'
          b'"finish_reason":{"type":"length"}}}\n\n',
          b'data: [DONE]\n\n')


@asynccontextmanager
async def harness(decoders):
    """Fake P at /p and candidates /d0.. whose handlers are `decoders` (None = released)."""
    seen, events = [], []

    async def tokenize(req):
        return web.json_response({"tokens": list(range(19)), "count": 19})

    async def prefill(req):
        seen.append((req.path, await req.json()))
        return web.json_response({"text": "", "meta_info": {"finish_reason": {"type": "length"}}})

    def decode(idx):
        async def handler(req):
            seen.append((req.path, await req.json()))
            if decoders[idx] is not None:
                return await decoders[idx](req)
            response = web.StreamResponse(headers={"content-type": "text/event-stream"})
            await response.prepare(req)
            await asyncio.sleep(.01)
            await response.write(RELEASED)
            return response
        return handler

    async def report(req):
        events.append(await req.json())
        return web.json_response({"accepted": True})

    app = web.Application()
    app.router.add_post("/tokenize", tokenize)
    app.router.add_post("/p/generate", prefill)
    for i in range(len(decoders)):
        app.router.add_post(f"/d{i}/generate", decode(i))
    app.router.add_post("/events", report)
    async with TestServer(app) as upstream:
        base = str(upstream.make_url("/")).rstrip("/")
        targets = {base + "/p/generate": "test"}
        targets.update({base + f"/d{i}/generate": "test" for i in range(len(decoders))})
        settings = config(tokenize_endpoint=base + "/tokenize", targets=targets,
                          callback_targets=[base + "/events"], context_length=4096)
        service = TokenizerService(settings, reference_hash)
        async with service.lifespan(service.app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service.app),
                                         base_url="http://tokenizer") as api:
                yield service, api, base, seen, events


def late_plan(info, base, k, default_idx=0):
    """A late-D plan as the router's /symmetric_pd/prepared returns it."""
    def target(leg):
        return {"transport": "sglang_generate", "endpoint": base + f"/{leg}/generate", "config_id": "test"}
    late = {"v": 1, "mode": "choose", "router_epoch": 9, "k": k, "default_idx": default_idx, "local_ok": False}
    cands = [{"target": target(f"d{i}"),
              "kv_import": {"transfer_id": 77, "attempt_id": 100 + i, "owner_addr": "tcp://p:10"},
              "late_d": {"v": 1, "mode": "choose", "router_epoch": 9, "k": k, "cand_idx": i}}
             for i in range(k)]
    return {
        **identity(info), "reservation_id": "reserved-" + info["request_id"], "transfer_id": 77,
        "prefill": {"target": target("p"), "late_d": late,
                    "kv_export": {"transfer_id": 77, "candidates": [{"cand_idx": i} for i in range(k)]},
                    "pull_hint": None, "kv_session": {"session_id": "s", "version": 1}},
        "decode": {"target": cands[default_idx]["target"], "kv_import": cands[default_idx]["kv_import"]},
        "decode_candidates": cands,
        "callback": {"endpoint": base + "/events", "token": "test-secret"},
    }


def winner(delay=.05):
    async def handler(req):
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(req)
        await asyncio.sleep(delay)  # admitted after the others were released
        for part in WINNER:
            await response.write(part)
        return response
    return handler


def failing(req):
    async def handler(req):
        return web.json_response({"error": "boom"}, status=500)
    return handler


def released_409(header):
    """A /v1/completions-style release: HTTP 409 (with or without the header)."""
    async def handler(req):
        headers = {"X-Late-D-Reason": "released"} if header else {}
        return web.json_response({"error": {"type": "late_d_released"}}, status=409, headers=headers)
    return handler


def conflict_409(req):
    async def handler(req):
        return web.json_response({"error": "conflict"}, status=409)
    return handler


def test_first_data_and_outcome():
    assert first_data(b'data: {"a"') is None
    assert first_data(b': ping\n\ndata: {"a":1}\n\n') == b'{"a":1}'
    assert late_d_outcome(first_data(RELEASED)) == "released"
    assert late_d_outcome(first_data(WINNER[0])) is None
    assert late_d_outcome(b'{"error":"x"}') == "failed"
    assert late_d_outcome(b'{"meta_info":{"finish_reason":{"type":"abort","err_type":"late_d_ttl"}}}') == "failed"
    assert late_d_outcome(b"[DONE]") == "failed"


def test_protocol_checks_the_candidates():
    info = {"protocol_version": 1, "request_id": "r", "ref": "x", "config_id": "test"}
    good = late_plan(info, "http://h", 3, 1)
    assert ForwardPD.model_validate(good).decode_candidates[2].late_d["cand_idx"] == 2
    for mutate in (
        lambda p: p["prefill"]["late_d"].update(k=2),
        lambda p: p["decode_candidates"][1]["late_d"].update(cand_idx=0),
        lambda p: p["decode_candidates"][0]["kv_import"].update(transfer_id=78),
        lambda p: p["decode"].update(kv_session={"session_id": "s", "version": 1}),
        lambda p: p.update(decode_candidates=[]),
        lambda p: p["prefill"].pop("late_d"),
    ):
        bad = late_plan(info, "http://h", 3, 1)
        mutate(bad)
        with pytest.raises(ValueError):
            ForwardPD.model_validate(bad)
    # A plain plan with a P late_d field and no candidates is refused too.
    plain = late_plan(info, "http://h", 1)
    plain.pop("decode_candidates")
    with pytest.raises(ValueError):
        ForwardPD.model_validate(plain)


def test_late_d_streams_the_admitted_candidate_and_reports_it():
    async def scenario():
        async with harness([None, winner(), failing(None)]) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body(stream=True))).json()
            response = await api.post("/forward_pd", json=late_plan(info, base, 3))
            assert response.status_code == 200
            assert response.content == b"".join(WINNER)
            p = next(v for path, v in seen if path == "/p/generate")
            assert p["late_d"]["k"] == 3 and p["kv_export"]["candidates"]
            assert p["kv_session"] == {"session_id": "s", "version": 1}
            for i in range(3):
                d = next(v for path, v in seen if path == f"/d{i}/generate")
                assert d["late_d"]["cand_idx"] == i and d["kv_import"]["attempt_id"] == 100 + i
                assert "kv_session" not in d and d["stream"] is True
                assert d["input_ids"] == p["input_ids"]
            async with asyncio.timeout(2):
                while not any(e["leg"] == "decode" and e["state"] == "completed" for e in events):
                    await asyncio.sleep(.005)
            cand = [(e["candidate"], e["state"]) for e in events if "candidate" in e]
            assert sorted(cand) == [(1, "bound"), (2, "failed")]
            bound = next(e for e in events if e.get("candidate") == 1)
            assert bound["event_id"] == "reserved-id:decode#1:bound"
            plain = [(e["leg"], e["state"]) for e in events if "candidate" not in e]
            assert ("decode", "uploaded") in plain and ("decode", "completed") in plain
            assert plain.index(("decode", "uploaded")) < plain.index(("decode", "completed"))
            done = next(e for e in events if e["leg"] == "decode" and e["state"] == "completed")
            assert done["usage"] == {"completion_tokens": 2, "spec_verify_ct": 1}
            async with asyncio.timeout(2):
                while service.pd_exchanges:
                    await asyncio.sleep(.005)
            assert service.upload_bytes == 0
    asyncio.run(scenario())


def test_late_d_without_an_admitted_candidate_fails_terminally():
    async def scenario():
        async with harness([None, None]) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body(stream=True))).json()
            response = await api.post("/forward_pd", json=late_plan(info, base, 2))
            assert response.status_code == 502
            async with asyncio.timeout(2):
                while service.pd_exchanges:
                    await asyncio.sleep(.005)
            failed = [e for e in events if e["leg"] == "decode" and e["state"] == "failed"
                      and "candidate" not in e]
            assert failed and failed[0].get("terminal_response") is True
            assert not [e for e in events if "candidate" in e]  # released is no failure
            assert service.upload_bytes == 0
    asyncio.run(scenario())


def test_late_d_http_409_release_is_not_a_failure():
    async def scenario():
        decoders = [released_409(True), winner(), released_409(False), conflict_409(None)]
        async with harness(decoders) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body(stream=True))).json()
            response = await api.post("/forward_pd", json=late_plan(info, base, 4))
            assert response.status_code == 200
            assert response.content == b"".join(WINNER)
            async with asyncio.timeout(2):
                while not any(e["leg"] == "decode" and e["state"] == "completed" for e in events):
                    await asyncio.sleep(.005)
            cand = sorted((e["candidate"], e["state"]) for e in events if "candidate" in e)
            assert cand == [(1, "bound"), (3, "failed")]
    asyncio.run(scenario())


def test_engine_abort_message_of_a_503_answer():
    """A symmetric prefill abort answers HTTP 503; its message goes to the router."""
    assert engine_abort_message(b'{"object":"error","message":"aborted by client","code":503}') == "aborted by client"
    assert engine_abort_message(b'{"error":{"message":"Compute-before-allocation aborted"}}') == \
        "compute-before-allocation aborted"
    assert engine_abort_message(
        b'{"text":"","meta_info":{"finish_reason":{"type":"abort","message":"x","status_code":503}}}'
    ) == "finish_reason abort: x"
    assert engine_abort_message(b'{"error":"out of memory"}') is None
    assert engine_abort_message(b"not json, aborted by client") == "aborted by client"
    assert engine_abort_message(b"") is None
