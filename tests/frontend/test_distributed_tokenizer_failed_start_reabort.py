"""A prefill failure must not leave the D request running (execution-ledger race).

The tokenizer submits the P and D legs concurrently. In an open-loop dry run the fake P engine
answered HTTP 500 while the D body (718k tokens) was still being uploaded and parsed. The Backend
aborted both engines in reply to the tokenizer's failure reports (+14 ms) and once more for the
late D upload (+17 ms), but the D engine admitted the D request only at +47 ms: every abort missed
(`hit 0`, an engine ignores an abort for a request it does not have). The D engine kept the request,
the tokenizer's cleanup drain timed out (`pd_cleanup_unknown terminal_legs=[]`) and no terminal
D report reached the Backend: its D-leg stayed unresolved until the 300-s expiry.
The fix: after a prefill failure in start(), the cleanup aborts the D request on its D engine again
(at once, when the D submit answers, then with backoff, bounded) until the drain reads the engine's
own abort event, which is reported as terminal D evidence.
CPU protocol tests with real local TCP; no engine or GPU.
"""
import asyncio
import logging
import time
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web

from evolving_infra.router.frontend.forward import PDForward

from test_distributed_tokenizer import body
from test_distributed_tokenizer_cancel_drain import terminal, until
from test_distributed_tokenizer_pd import harness, plan, sse

LOGGER = "evolving_infra.router.frontend.forward"
WAITING_ABORTED = {"text": "", "output_ids": [],
                   "meta_info": {"id": "id", "prompt_tokens": 19, "completion_tokens": 0,
                                 "finish_reason": {"type": "abort", "message": "Abort by router"}}}
FINISHED = {"text": "x", "output_ids": [7],
            "meta_info": {"id": "id", "prompt_tokens": 19, "completion_tokens": 1,
                          "finish_reason": {"type": "length", "length": 1}}}


class Fleet:
    """Fake P/D engines with SGLang's abort semantics and a Backend that aborts like the router.

    An engine abort ends only a request the engine has admitted (`hit`); the Backend sends one
    abort to each engine for every failed/cancelled report and for an upload reported after one
    (evolving_infra/router/core/src/backend/execution.rs `event()`), as the Rust router does.
    """

    def __init__(self):
        self.live = {}          # rid -> asyncio.Event: the D engine's admitted requests
        self.aborts = []        # (monotonic time, source, path, rid, hit)
        self.reports = []       # (monotonic time, event)
        self.failed = set()     # legs the Backend saw failed/cancelled
        self.seen = set()
        self.backend_done = asyncio.Event()
        self.admitted_at = None
        self.answered_at = None
        self.ended = []

    async def abort(self, req):
        value = await req.json()
        rid = value["rid"]
        hit = 0
        if req.path == "/d/abort_request" and rid in self.live:
            self.live[rid].set()
            hit = 1
        self.aborts.append((time.monotonic(), req.headers.get("X-Abort-Source", "tokenizer"),
                            req.path, rid, hit))
        return web.json_response({"success": True})

    async def backend(self, req, value):
        self.reports.append((time.monotonic(), value))
        leg, state = value["leg"], value["state"]
        again = state == "uploaded" and leg in self.failed
        if state in ("failed", "cancelled") or again:
            self.failed.add(leg)
            origin = str(req.url.origin())
            async with aiohttp.ClientSession() as client:
                for engine in ("p", "d"):
                    async with client.post(origin + f"/{engine}/abort_request", json={"rid": value["request_id"]},
                                           headers={"X-Abort-Source": "backend"}) as response:
                        await response.read()
        self.seen.add((leg, state))
        if {("decode", "cancelled"), ("decode", "uploaded")} <= self.seen:
            self.backend_done.set()
        return web.json_response({"accepted": True})

    def tokenizer_aborts(self, path="/d/abort_request"):
        return [a for a in self.aborts if a[1] == "tokenizer" and a[2] == path]

    async def admit_late(self, rid):
        """The D engine admits the request only after the Backend's aborts were answered."""
        await asyncio.wait_for(self.backend_done.wait(), 5)
        self.admitted_at = time.monotonic()
        self.live[rid] = asyncio.Event()
        return self.live[rid]

    async def decode(self, req, value, stream, honour_aborts=True, seconds=10):
        """A D request waiting for KV that never comes: it ends only by an engine abort."""
        rid = value["rid"]
        abort = await self.admit_late(rid) if honour_aborts else asyncio.Event()
        if not honour_aborts:
            await asyncio.wait_for(self.backend_done.wait(), 5)
            self.admitted_at = time.monotonic()
        try:
            if stream:
                response = web.StreamResponse(headers={"content-type": "text/event-stream"})
                await response.prepare(req)
                self.answered_at = time.monotonic()
            try:
                await asyncio.wait_for(abort.wait(), seconds)
                result, self.ended = WAITING_ABORTED, self.ended + ["aborted"]
            except TimeoutError:
                result, self.ended = FINISHED, self.ended + ["finished"]
            if stream:
                await response.write(sse(result, b"[DONE]"))
                return response
            self.answered_at = time.monotonic()
            return web.json_response(result)
        finally:
            self.live.pop(rid, None)


async def failing_prefill(req, value):
    return web.json_response({"error": {"message": "injected prefill failure (dry run)"}}, status=500)


def decode_reports(fleet):
    return [(e["state"], e.get("terminal_response")) for _, e in fleet.reports
            if e["leg"] == "decode" and "candidate" not in e]


def messages(caplog, needle):
    return [r.getMessage() for r in caplog.records if needle in r.getMessage()]


@pytest.mark.parametrize("stream", [True, False])
def test_prefill_failure_with_decode_submit_in_flight_ends_the_decode_request(stream, caplog):
    """End to end: every Backend abort misses the D engine; the D-leg still ends terminally."""
    async def scenario():
        fleet = Fleet()

        async def d(req, value):
            return await fleet.decode(req, value, stream)

        async with harness(failing_prefill, d, callback_http=fleet.backend,
                           abort_http=fleet.abort) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body(stream=stream))).json()
            with patch.object(PDForward, "CANCEL_GRACE_SECONDS", 3):
                response = await asyncio.wait_for(api.post("/forward_pd", json=plan(info, base)), 3)
                assert response.status_code == 502
                assert response.json()["error"]["code"] == "PD_SUBMIT_FAILED"
                await until(lambda: service.callback_outbox.summary()["reserved_plans"] == 0, seconds=6)
            # Precondition of the race: the Backend's aborts for the failure reports all
            # reached the engines before the D engine admitted the request, and missed.
            early = [a for a in fleet.aborts if a[1] == "backend" and a[0] < fleet.admitted_at]
            assert len(early) >= 8 and not any(a[4] for a in early)
            # The D engine ended the request by an abort, not by a disconnect or a timeout.
            assert fleet.ended == ["aborted"] and not fleet.live
            assert any(a[4] for a in fleet.tokenizer_aborts())
            assert not fleet.tokenizer_aborts("/p/abort_request")  # the P engine is the Backend's
            # The Backend got terminal D evidence at once, long before any expiry.
            assert terminal(events, "decode"), decode_reports(fleet)
            at = next(t for t, e in fleet.reports if e["leg"] == "decode" and e.get("terminal_response"))
            assert at - fleet.admitted_at < 1.5
            assert not terminal(events, "prefill")  # an HTTP 500 is not P terminal evidence
            assert len(seen) == 2  # no generation was resubmitted
            assert service.upload_bytes == 0 and not service.pd_exchanges
            assert messages(caplog, "pd_cleanup_unknown") == ["pd_cleanup_unknown rid='id' terminal_legs=['decode']"]
            [line] = messages(caplog, "pd_decode_reabort")
            assert "decode_terminal=True" in line and "answered=True" in line

    with caplog.at_level(logging.INFO, logger=LOGGER):
        asyncio.run(scenario())


def test_reabort_is_bounded_when_the_engine_ignores_aborts(caplog):
    """An engine that never ends the request: aborts stop after the window; the old fallback remains."""
    async def scenario():
        fleet = Fleet()

        async def d(req, value):
            return await fleet.decode(req, value, True, honour_aborts=False)

        async with harness(failing_prefill, d, callback_http=fleet.backend,
                           abort_http=fleet.abort) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body(stream=True))).json()
            with patch.object(PDForward, "CANCEL_GRACE_SECONDS", 2), \
                    patch.object(PDForward, "REABORT_WINDOW_SECONDS", 0.6):
                start = time.monotonic()
                response = await asyncio.wait_for(api.post("/forward_pd", json=plan(info, base)), 3)
                assert response.status_code == 502
                await until(lambda: service.callback_outbox.summary()["reserved_plans"] == 0, seconds=6)
            sent = fleet.tokenizer_aborts()
            assert 3 <= len(sent) <= 10
            assert sent[-1][0] - start < 0.6 + 0.5
            assert not terminal(events, "decode")  # no engine evidence: left to the Backend's expiry
            assert messages(caplog, "pd_cleanup_unknown") == ["pd_cleanup_unknown rid='id' terminal_legs=[]"]
            [line] = messages(caplog, "pd_decode_reabort")
            assert "decode_terminal=False" in line

    with caplog.at_level(logging.INFO, logger=LOGGER):
        asyncio.run(scenario())


def test_reabort_stops_when_the_decode_submit_fails(caplog):
    """A D engine that answers HTTP 503 has no request to end: no abort after its answer."""
    async def scenario():
        fleet = Fleet()

        async def d(req, value):
            await asyncio.wait_for(fleet.backend_done.wait(), 5)
            fleet.answered_at = time.monotonic()
            return web.json_response({"error": "unavailable"}, status=503)

        async with harness(failing_prefill, d, callback_http=fleet.backend,
                           abort_http=fleet.abort) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body(stream=True))).json()
            with patch.object(PDForward, "CANCEL_GRACE_SECONDS", 3):
                response = await asyncio.wait_for(api.post("/forward_pd", json=plan(info, base)), 3)
                assert response.status_code == 502
                await until(lambda: service.callback_outbox.summary()["reserved_plans"] == 0, seconds=6)
            assert len([a for a in fleet.tokenizer_aborts() if a[0] > fleet.answered_at]) <= 1
            assert ("failed", None) in decode_reports(fleet)
            assert not terminal(events, "decode")  # unchanged: an HTTP error is not terminal evidence
            [line] = messages(caplog, "pd_decode_reabort")
            assert "answered=False" in line

    with caplog.at_level(logging.INFO, logger=LOGGER):
        asyncio.run(scenario())


# ---- unchanged behaviour: no tokenizer abort outside the failed-start path ----

@pytest.mark.parametrize("stream", [True, False])
def test_healthy_exchange_sends_no_engine_abort(stream, caplog):
    async def scenario():
        fleet = Fleet()

        async def d(req, value):
            if stream:
                response = web.StreamResponse(headers={"content-type": "text/event-stream"})
                await response.prepare(req)
                await response.write(sse(FINISHED, b"[DONE]"))
                return response
            return web.json_response(FINISHED)

        async with harness(decode=d, callback_http=fleet.backend,
                           abort_http=fleet.abort) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body(stream=stream))).json()
            response = await api.post("/forward_pd", json=plan(info, base))
            assert response.status_code == 200
            await until(lambda: service.callback_outbox.summary()["reserved_plans"] == 0)
            assert terminal(events, "prefill") and terminal(events, "decode")
            assert not fleet.aborts
            assert not messages(caplog, "pd_decode_reabort") and not messages(caplog, "pd_cleanup_unknown")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        asyncio.run(scenario())


def test_decode_failure_first_sends_no_tokenizer_abort(caplog):
    """D fails while P still runs: the existing path (Backend aborts only)."""
    async def scenario():
        fleet = Fleet()
        release = asyncio.Event()

        async def p(req, value):
            await release.wait()
            return web.json_response({"text": "", "meta_info": {"finish_reason": {"type": "length"}}})

        async def d(req, value):
            return web.json_response({"error": "unavailable"}, status=503)

        async with harness(p, d, callback_http=fleet.backend,
                           abort_http=fleet.abort) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body(stream=True))).json()
            try:
                response = await asyncio.wait_for(api.post("/forward_pd", json=plan(info, base)), 3)
                assert response.status_code == 502
                await until(lambda: ("decode", "cancelled") in fleet.seen)
                release.set()
                await until(lambda: service.callback_outbox.summary()["reserved_plans"] == 0)
            finally:
                release.set()
            assert fleet.aborts and not fleet.tokenizer_aborts() and not fleet.tokenizer_aborts("/p/abort_request")
            assert terminal(events, "prefill")
            assert not messages(caplog, "pd_decode_reabort")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        asyncio.run(scenario())


def test_client_cancel_before_answers_sends_no_tokenizer_abort(caplog):
    """A client cancel while both legs run: unchanged (only the Backend aborts)."""
    async def scenario():
        fleet = Fleet()
        submitted, release = asyncio.Event(), asyncio.Event()
        count = 0

        async def engine(req, value):
            nonlocal count
            count += 1
            if count == 2:
                submitted.set()
            await release.wait()
            return web.json_response({"meta_info": {"finish_reason": {"type": "abort", "message": "x"}}})

        async with harness(engine, engine, callback_http=fleet.backend,
                           abort_http=fleet.abort) as (service, api, base, seen, events):
            info = (await api.post("/prepare", json=body())).json()
            call = asyncio.create_task(api.post("/forward_pd", json=plan(info, base)))
            try:
                await asyncio.wait_for(submitted.wait(), 3)
                call.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await call
                await until(lambda: ("decode", "cancelled") in fleet.seen)
                await asyncio.sleep(0.3)
                release.set()
                await until(lambda: all(terminal(events, leg) for leg in ("prefill", "decode")))
                await until(lambda: service.callback_outbox.summary()["reserved_plans"] == 0)
            finally:
                release.set()
            assert fleet.aborts and not fleet.tokenizer_aborts() and not fleet.tokenizer_aborts("/p/abort_request")
            assert not messages(caplog, "pd_decode_reabort")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        asyncio.run(scenario())
