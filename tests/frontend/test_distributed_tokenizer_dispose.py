# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Normal completion cleanup must survive cancellation of its HTTP caller."""

import asyncio

import pytest
from aiohttp import web

from test_distributed_tokenizer import body
from test_distributed_tokenizer_cancel_drain import terminal, until
from test_distributed_tokenizer_pd import harness, plan


@pytest.mark.parametrize("finish", ["repeat_close", "shutdown"])
def test_cancel_during_normal_dispose_keeps_cleanup_owned(finish):
    async def scenario():
        callback_started, release = asyncio.Event(), asyncio.Event()

        async def callback(event):
            if event["state"] == "completed":
                callback_started.set()
                await release.wait()
            return web.json_response({"accepted": True})

        observer = None
        try:
            async with harness(callback=callback) as (service, api, base, seen, events):
                info = (await api.post("/prepare", json=body())).json()
                call = asyncio.create_task(api.post("/forward_pd", json=plan(info, base)))
                await asyncio.wait_for(callback_started.wait(), 1)
                exchange = next(iter(service.pd_exchanges))
                # Both original engine responses are terminal. Only callback delivery
                # remains, and the normal aclose path is now inside _dispose.
                await until(lambda: exchange.closed and len(exchange.terminal) == 2)
                assert service.callback_outbox.summary()["reserved_plans"] == 1
                call.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(call, 1)

                if finish == "repeat_close":
                    closes = [asyncio.create_task(exchange.aclose()) for _ in range(2)]
                    release.set()
                    await asyncio.wait_for(asyncio.gather(*closes), 1)
                    await until(lambda: not service.pd_exchanges, seconds=1)
                else:
                    async def during_shutdown():
                        await until(lambda: service.callback_outbox.closed)
                        assert not service.client.closed
                        release.set()
                    observer = asyncio.create_task(during_shutdown())

            if observer is not None:
                await observer
            assert not service.pd_exchanges
            assert service.callback_outbox.summary()["reserved_plans"] == 0
            assert exchange.reporter.done() and not exchange.reporter.cancelled()
            assert all(terminal(events, leg) for leg in ("prefill", "decode"))
            assert not any(e["state"] in ("cancelled", "failed") for e in events)
            assert len(seen) == 2 and service.upload_bytes == 0
        finally:
            release.set()
            if observer is not None:
                await asyncio.gather(observer, return_exceptions=True)
    asyncio.run(scenario())


def test_cancel_during_normal_dispose_callback_timeout_retains_outbox_metadata():
    async def scenario():
        callback_started, release = asyncio.Event(), asyncio.Event()

        async def callback(event):
            if event["state"] == "completed":
                callback_started.set()
                await release.wait()
            return web.json_response({"accepted": True})

        try:
            async with harness(callback=callback) as (service, api, base, seen, events):
                service.config = service.config.model_copy(update={"callback_timeout_seconds": .03})
                info = (await api.post("/prepare", json=body())).json()
                call = asyncio.create_task(api.post("/forward_pd", json=plan(info, base)))
                await asyncio.wait_for(callback_started.wait(), 1)
                exchange = next(iter(service.pd_exchanges))
                await until(lambda: exchange.closed and len(exchange.terminal) == 2)
                call.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(call, 1)
                await asyncio.wait_for(exchange.aclose(), 1)
                await until(lambda: not service.pd_exchanges, seconds=1)
                # Local resources can close after bounded callback attempts, but
                # unacknowledged metadata must stay in the existing retry outbox.
                assert exchange.delivery.finished
                assert exchange.reporter.done() and not exchange.reporter.cancelled()
                assert service.callback_outbox.summary()["pending_events"] == 2
                assert service.callback_outbox.summary()["reserved_plans"] == 1
                assert len(seen) == 2 and service.upload_bytes == 0
                release.set()
                await until(lambda: service.callback_outbox.summary()["reserved_plans"] == 0)
                assert all(terminal(events, leg) for leg in ("prefill", "decode"))
        finally:
            release.set()
    asyncio.run(scenario())
