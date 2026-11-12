"""Agent gateway: identities inferred from messages, and END for idle branches."""
import asyncio
import json

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from evolving_infra.router.gateway.server import Gateway
from evolving_infra.router.gateway.sessions import Tracker, message_key

SYS = {"role": "system", "content": "You are a coding agent."}
U1 = {"role": "user", "content": "Fix the bug."}
A1 = {"role": "assistant", "content": None,
      "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{\"cmd\":\"ls\"}"}}]}
T1 = {"role": "tool", "tool_call_id": "c1", "content": "main.py"}


def test_turns_of_one_conversation_share_a_branch_and_count_up():
    t = Tracker()
    a = t.assign([SYS, U1])
    t.finish(a, reached_router=True)
    b = t.assign([SYS, U1, A1, T1])
    assert (b.branch.agent_id, b.branch.subagent_id, b.request_id) == (a.branch.agent_id, 0, 1)


def test_moving_cache_markers_and_part_splitting_do_not_break_a_match():
    t = Tracker()
    first = [SYS, {"role": "user", "content": [{"type": "text", "text": "Fix the bug.", "cache_control": {"type": "ephemeral"}}]}]
    a = t.assign(first)
    t.finish(a, reached_router=True)
    b = t.assign([SYS, U1, A1, T1])
    assert b.branch is a.branch and b.request_id == 1


def test_an_unrelated_conversation_is_a_new_agent():
    t = Tracker()
    a = t.assign([SYS, U1])
    t.finish(a, reached_router=True)
    b = t.assign([SYS, {"role": "user", "content": "Write a title for this chat."}])
    assert b.branch.agent_id != a.branch.agent_id and b.request_id == 0


def test_a_concurrent_continuation_gets_its_own_branch_of_the_same_agent():
    t = Tracker()
    a = t.assign([SYS, U1])
    t.finish(a, reached_router=True)
    main = t.assign([SYS, U1, A1, T1])          # in flight
    side = t.assign([SYS, U1, {"role": "user", "content": "summarize"}])
    assert side.branch.agent_id == main.branch.agent_id
    assert side.branch.subagent_id != main.branch.subagent_id and side.request_id == 0


def test_idle_branches_end_and_are_not_matched_again():
    t = Tracker()
    a = t.assign([SYS, U1], now=0.0)
    t.finish(a, reached_router=True, now=1.0)
    assert t.idle(600, now=100.0) == []
    assert t.idle(600, now=700.0) == [a.branch]
    b = t.assign([SYS, U1, A1, T1], now=701.0)
    assert b.branch.agent_id != a.branch.agent_id


def test_message_key_ignores_ids_but_not_content():
    assert message_key({"role": "user", "content": "x"}) == message_key({"role": "user", "content": [{"type": "text", "text": "x"}]})
    assert message_key({"role": "user", "content": "x"}) != message_key({"role": "user", "content": "y"})


async def _router(seen):
    async def chat(request):
        seen.append(("chat", request.headers.get("x-agent-context"), await request.json()))
        return web.json_response({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    async def end(request):
        seen.append(("end", None, await request.json()))
        return web.json_response({"accepted": True})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/kv_sessions/end", end)
    server = TestServer(app)
    await server.start_server()
    return server


def test_gateway_sends_identity_and_ends_idle_branches():
    async def run():
        seen = []
        router = await _router(seen)
        gateway = Gateway(str(router.make_url("")), idle_end_secs=0.05, sweep_secs=0.02)
        client = TestClient(TestServer(gateway.app()))
        await client.start_server()
        try:
            r = await client.post("/v1/chat/completions", json={"model": "m", "messages": [SYS, U1], "rid": "x"})
            assert r.status == 200 and (await r.json())["choices"][0]["message"]["content"] == "ok"
            r = await client.post("/v1/chat/completions", json={"model": "m", "messages": [SYS, U1, A1, T1]})
            assert r.status == 200
            await asyncio.sleep(0.2)
        finally:
            await client.close()
            await router.close()
        chats = [s for s in seen if s[0] == "chat"]
        ids = [json.loads(h) for _, h, _ in chats]
        assert [i["request_id"] for i in ids] == [0, 1] and ids[0]["agent_id"] == ids[1]["agent_id"]
        assert "rid" not in chats[0][2]
        ends = [s[2] for s in seen if s[0] == "end"]
        assert ends == [{"agent_id": ids[0]["agent_id"], "subagent_id": 0}]

    asyncio.run(run())


def test_without_retention_no_end_is_sent():
    async def run():
        seen = []
        router = await _router(seen)
        gateway = Gateway(str(router.make_url("")), retention=False, idle_end_secs=0.05, sweep_secs=0.02)
        client = TestClient(TestServer(gateway.app()))
        await client.start_server()
        try:
            await client.post("/v1/chat/completions", json={"model": "m", "messages": [SYS, U1]})
            await asyncio.sleep(0.2)
        finally:
            await client.close()
            await router.close()
        assert [s[0] for s in seen] == ["chat"]

    asyncio.run(run())


def test_two_conversations_that_open_alike_stay_apart():
    t = Tracker()
    a = t.assign([SYS, U1])
    t.finish(a, reached_router=True)
    b = t.assign([SYS, U1])
    assert b.branch.agent_id != a.branch.agent_id


def test_a_retry_of_a_failed_request_continues_its_branch():
    t = Tracker()
    a = t.assign([SYS, U1])
    t.finish(a, reached_router=True, ok=False)
    b = t.assign([SYS, U1])
    assert b.branch is a.branch and b.request_id == 1


def test_unknown_content_parts_are_part_of_the_key():
    x = {"role": "user", "content": [{"type": "file", "file": {"file_id": "a"}}]}
    y = {"role": "user", "content": [{"type": "file", "file": {"file_id": "b"}}]}
    assert message_key(x) != message_key(y)


def test_a_branch_whose_request_failed_still_gets_end():
    async def run():
        seen = []

        async def chat(request):
            seen.append(("chat", request.headers.get("x-agent-context"), None))
            return web.json_response({"error": "boom"}, status=500)

        async def end(request):
            seen.append(("end", None, await request.json()))
            return web.json_response({"accepted": True})

        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat)
        app.router.add_post("/kv_sessions/end", end)
        router = TestServer(app)
        await router.start_server()
        gateway = Gateway(str(router.make_url("")), idle_end_secs=0.05, sweep_secs=0.02)
        client = TestClient(TestServer(gateway.app()))
        await client.start_server()
        try:
            r = await client.post("/v1/chat/completions", json={"model": "m", "messages": [SYS, U1]})
            assert r.status == 500
            await asyncio.sleep(0.2)
        finally:
            await client.close()
            await router.close()
        assert [s[0] for s in seen] == ["chat", "end"]

    asyncio.run(run())


def test_a_placement_refused_during_a_role_switch_is_retried():
    async def run():
        calls = []

        async def chat(request):
            calls.append(request.headers.get("x-agent-context"))
            if len(calls) == 1:
                return web.json_response({"error": {"code": "symmetric_pd_no_placement"}}, status=503)
            return web.json_response({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat)
        router = TestServer(app)
        await router.start_server()
        client = TestClient(TestServer(Gateway(str(router.make_url(""))).app()))
        await client.start_server()
        try:
            r = await client.post("/v1/chat/completions", json={"model": "m", "messages": [SYS, U1]})
            assert r.status == 200
        finally:
            await client.close()
            await router.close()
        assert len(calls) == 2 and calls[0] == calls[1]

    asyncio.run(run())
