"""Decode-abort evidence must survive a client cancellation (execution-ledger leak).

An open-loop dry run left one execution unresolved forever: the D engine's abort
event and `[DONE]` arrived in one read, the reader awaited the Backend's
/symmetric_pd/migration answer for the abort, and the client disconnect cancelled it
there. The D-leg was reported `cancelled`, `[DONE]` stayed unparsed in `pending`,
and the cleanup drain then hit EOF and reported a non-terminal "Truncated decode
event stream": `pd_cleanup_unknown terminal_legs=['prefill']`, and the Backend kept
the execution (D `unknown` in decode_population, one prepared-execution slot).
CPU protocol tests; no engine or GPU.
"""
import asyncio
import contextlib
import json
import logging
import types

import pytest
from aiohttp import web

from evolving_infra.router.frontend.forward import PDForward, response_usage
from evolving_infra.router.frontend.store import ProtocolError

from test_distributed_tokenizer_cancel_drain import response_for, terminal, until
from test_distributed_tokenizer_pd import harness, sse

FIRST = {"text": "a", "output_ids": [101, 102],
         "meta_info": {"id": "id", "prompt_tokens": 19, "cached_tokens": 7,
                       "completion_tokens": 2, "finish_reason": None}}
ABORTED = {"text": "", "output_ids": [103],
           "meta_info": {"id": "id", "prompt_tokens": 19, "cached_tokens": 7, "completion_tokens": 3,
                         "finish_reason": {"type": "abort", "message": "aborted by client"}}}
ABORT_DATA = json.dumps(ABORTED).encode()
FINISHED = {"text": "b", "output_ids": [103],
            "meta_info": {"completion_tokens": 3, "finish_reason": {"type": "length"}}}


def decode_reports(events):
    return [e for e in events if e["leg"] == "decode" and "candidate" not in e]


def test_abort_and_done_in_one_read_cancelled_inside_migrated_is_terminal(caplog):
    """The abort/done race end to end: real TCP engine and Backend, PDForwardResponse, cleanup."""
    async def scenario():
        asked, release = asyncio.Event(), asyncio.Event()
        seen_pending = []
        exchange = None

        async def d(req, value):
            response = web.StreamResponse(headers={"content-type": "text/event-stream"})
            await response.prepare(req)
            # One write: a token event, the abort event and [DONE] reach the
            # tokenizer in one read.
            await response.write(sse(FIRST, ABORTED, b"[DONE]"))
            return response

        async def migration(req):
            seen_pending.append(bytes(exchange.pending))
            asked.set()
            await release.wait()
            return web.json_response({"migrating": False})

        async with harness(decode=d, router={"/migration": migration}) as (service, api, base, seen, events):
            response, scope, receive = await response_for(service, api, base)
            exchange = response.exchange
            relayed = bytearray()

            async def send(message):
                if message["type"] == "http.response.body":
                    relayed.extend(message.get("body", b""))

            serving = asyncio.create_task(response(scope, receive, send))
            try:
                await asyncio.wait_for(asked.wait(), 5)
                # Precondition of the race: [DONE] arrived with the abort event
                # and is still unparsed while the Backend is asked.
                assert b"[DONE]" in seen_pending[0] and not exchange.saw_done
                serving.cancel()  # the client disconnect lands inside migrated()
                with pytest.raises(asyncio.CancelledError):
                    await serving
                await until(lambda: not service.pd_exchanges, seconds=5)
                await until(lambda: service.callback_outbox.summary()["reserved_plans"] == 0, seconds=5)
                assert terminal(events, "prefill")
                assert terminal(events, "decode"), decode_reports(events)
                assert not [r for r in caplog.records if "pd_cleanup_unknown" in r.getMessage()]
                assert len(seen) == 2  # no generation was resubmitted
            finally:
                release.set()

    with caplog.at_level(logging.WARNING, logger="evolving_infra.router.frontend.forward"):
        asyncio.run(scenario())


# ---- unit level: one PDForward with a scripted D response (no sockets) ----

class _Content:
    def __init__(self, parts):
        self.parts = list(parts)

    async def readany(self):
        return self.parts.pop(0) if self.parts else b""


class _Response:
    status = 200

    def __init__(self, parts):
        self.content = _Content(parts)


class _BlockingBackend:
    """A /symmetric_pd/migration POST that does not answer until released."""

    def __init__(self):
        self.asked, self.release = asyncio.Event(), asyncio.Event()

    def post(self, url, **kwargs):
        backend = self

        class Call:
            async def __aenter__(self):
                backend.asked.set()
                await backend.release.wait()

                async def answer():
                    return {"migrating": False}
                return types.SimpleNamespace(status=200, json=answer)

            async def __aexit__(self, *exc):
                return False
        return Call()


def exchange(parts=(), pending=b"", last_data=b"", client=None, resumable=False):
    service = types.SimpleNamespace(
        callback_outbox=types.SimpleNamespace(reserve=lambda plan: None),
        pd_exchanges=set(), client=client,
        config=types.SimpleNamespace(max_response_bytes=1 << 20, callback_timeout_seconds=5))
    plan = types.SimpleNamespace(
        request_id="rid", reservation_id="res", transfer_id=1,
        callback=types.SimpleNamespace(endpoint="http://backend/symmetric_pd/events", token="t"))
    x = PDForward(service, plan)
    x.stream = True
    x.resumable = resumable
    x.input_body = b'{"input_ids":[1,2,3]' if resumable else None
    x.responses["decode"] = _Response(parts)
    x.pending.extend(pending)
    x.last_data = last_data
    return x


def queued(x):
    out = []
    while not x.events.empty():
        out.append(x.events.get_nowait())
    return out


async def drain(x):
    with contextlib.suppress(ProtocolError):
        async for _ in x.chunks(draining=True):
            pass
    return queued(x)


def test_cancellation_inside_migrated_reports_the_read_abort_as_terminal():
    async def scenario():
        backend = _BlockingBackend()
        x = exchange([sse(FIRST, ABORTED, b"[DONE]")], client=backend, resumable=True)
        done = asyncio.get_running_loop().create_future()
        done.set_result(None)
        x.tasks["prefill"] = done

        async def downstream():
            async for _ in x.chunks():
                pass
        reader = asyncio.create_task(downstream())
        await asyncio.wait_for(backend.asked.wait(), 2)
        assert b"[DONE]" in x.pending  # the race's precondition
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        events = queued(x)
        assert [(e["state"], e.get("terminal_response")) for e in events] == [("failed", True)]
        assert "decode" in x.terminal
    asyncio.run(scenario())


@pytest.mark.parametrize("pending, last_data", [
    (b"data: [DONE]\n\n", ABORT_DATA),                              # abort parsed, [DONE] left over
    (b"data: " + ABORT_DATA + b"\n\ndata: [DONE]\n\n", b""),        # both left over
    (b"data: " + ABORT_DATA + b"\n\n", b""),                        # abort left over, no [DONE]
])
def test_eof_parses_left_over_complete_lines_before_truncation(pending, last_data):
    async def scenario():
        x = exchange(pending=pending, last_data=last_data)
        events = await drain(x)
        assert [(e["state"], e.get("terminal_response")) for e in events] == [("failed", True)]
        assert "decode" in x.terminal
    asyncio.run(scenario())


def test_eof_left_over_success_lines_complete_the_leg():
    async def scenario():
        x = exchange(pending=b"data: " + json.dumps(FINISHED).encode() + b"\n\ndata: [DONE]\n\n")
        events = await drain(x)
        assert [e["state"] for e in events] == ["completed"]
    asyncio.run(scenario())


@pytest.mark.parametrize("pending", [
    b'data: {"text":"x","meta_info":{"finish_reason":null}}\n\n',   # no terminal event, no [DONE]
    b'data: {"text":"x","meta_info":{"finish_reason":null}}\n\ndata: [DONE]\n\n',  # [DONE] only
    b'data: {"text":"x"',                                            # an incomplete line
])
def test_eof_without_terminal_evidence_stays_unknown(pending):
    """Unchanged: no engine evidence, no terminal report."""
    async def scenario():
        x = exchange(pending=pending)
        events = await drain(x)
        assert [(e["state"], bool(e.get("terminal_response"))) for e in events] == [("failed", False)]
        assert "decode" not in x.terminal
    asyncio.run(scenario())


def test_healthy_stream_is_unchanged():
    async def scenario():
        x = exchange([sse(FIRST), sse(FINISHED, b"[DONE]")])
        events = await drain(x)
        assert [e["state"] for e in events] == ["completed"]
        assert not getattr(x, "decode_aborted", False)
    asyncio.run(scenario())


# ---- more orderings of the same race ----


def _states(events):
    return [(e["state"], bool(e.get("terminal_response")), e.get("error")) for e in events]


def _prefill_done(x):
    done = asyncio.get_running_loop().create_future()
    done.set_result(None)
    x.tasks["prefill"] = done


class _RaisingBackend:
    """migrated()'s POST raises an exception that migrated() does not catch."""

    def post(self, url, **kwargs):
        raise RuntimeError("Session is closed")


class _MigratingBackend:
    """The Backend confirms the migration at once."""

    def post(self, url, **kwargs):
        class Call:
            async def __aenter__(self):
                async def answer():
                    return {"migrating": True}
                return types.SimpleNamespace(status=200, json=answer)

            async def __aexit__(self, *exc):
                return False
        return Call()


def test_abort_read_then_migrated_raises_is_terminal():
    """Before the fix: `failed` without terminal evidence and [DONE] left in `pending` (the same leak)."""
    async def scenario():
        x = exchange([sse(FIRST, ABORTED, b"[DONE]")], client=_RaisingBackend(), resumable=True)
        _prefill_done(x)
        with pytest.raises(RuntimeError):
            async for _ in x.chunks():
                pass
        assert _states(queued(x)) == [("failed", True, "RuntimeError")]
    asyncio.run(scenario())


def test_stream_closed_at_yield_after_confirmed_migration_is_terminal():
    """The consumer closes the stream at the yield before resume(). Before the fix: `cancelled` (the same leak)."""
    async def scenario():
        x = exchange([sse(FIRST, ABORTED, b"[DONE]")], client=_MigratingBackend(), resumable=True)
        _prefill_done(x)
        gen = x.chunks()
        first = await gen.__anext__()          # the FIRST line, yielded before resume()
        assert b"output_ids" in first
        await gen.aclose()                     # GeneratorExit at that yield
        assert _states(queued(x)) == [("failed", True, "PD_ENGINE_ERROR")]
    asyncio.run(scenario())


def test_abort_and_done_in_separate_reads_cancel_inside_migrated_is_terminal():
    """[DONE] in a later read: terminal at once (before the fix the drain reached it; no leak either way)."""
    async def scenario():
        backend = _BlockingBackend()
        x = exchange([sse(FIRST, ABORTED), sse(b"[DONE]")], client=backend, resumable=True)
        _prefill_done(x)

        async def downstream():
            async for _ in x.chunks():
                pass
        reader = asyncio.create_task(downstream())
        await asyncio.wait_for(backend.asked.wait(), 2)
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        assert _states(queued(x)) == [("failed", True, "PD_ENGINE_ERROR")]
        assert "decode" in x.terminal
    asyncio.run(scenario())


@pytest.mark.parametrize("split", [1, 7, 33, 10_000])
def test_healthy_stream_bytes_and_usage_unchanged_for_any_read_size(split):
    """Unchanged: the same relayed bytes and one `completed` report with the last event's usage."""
    async def scenario():
        body = sse(FIRST, FINISHED, b"[DONE]")
        x = exchange([body[i:i + split] for i in range(0, len(body), split)])
        _prefill_done(x)
        out = bytearray()
        async for part in x.chunks():
            out += part
        assert bytes(out) == body
        events = queued(x)
        assert [e["state"] for e in events] == ["completed"]
        assert events[0].get("usage") == response_usage(json.dumps(FINISHED).encode(), "decode")
    asyncio.run(scenario())


def test_abort_and_done_in_one_read_without_migration_is_terminal_once():
    """Unchanged: not cancelled, the Backend answers not migrating -> one terminal `failed`."""
    async def scenario():
        backend = _BlockingBackend()
        x = exchange([sse(FIRST, ABORTED, b"[DONE]")], client=backend, resumable=True)
        _prefill_done(x)
        backend.release.set()
        with pytest.raises(Exception):
            async for _ in x.chunks():
                pass
        assert _states(queued(x)) == [("failed", True, "ProtocolError")]
    asyncio.run(scenario())
