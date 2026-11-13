# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Identity and real HTTP protocol checks without a GPU or SGLang process."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import unittest

from aiohttp import web

from evolving_infra.router.client.agent import Agent
from evolving_infra.router.client.agent.identity import CONTEXT_HEADER
from evolving_infra.router.client.backend import Backend
from evolving_infra.router.client.scheduler import Scheduler
from evolving_infra.router.client.types import Request, RequestIdentity


class PublicAPICompatibilityTests(unittest.TestCase):
    def test_package_exports_share_the_canonical_classes(self):
        import evolving_infra.router.client as client
        from evolving_infra.router.client import scheduler as facade

        for public, package_alias, facade_alias in [
            (Agent, client.Agent, facade.Agent),
            (Backend, client.Backend, facade.Backend),
            (Request, client.Request, facade.Request),
            (RequestIdentity, client.RequestIdentity, facade.RequestIdentity),
        ]:
            self.assertIs(public, package_alias)
            self.assertIs(public, facade_alias)
        self.assertIs(Scheduler, client.Scheduler)

    def test_agent_and_request_contracts_do_not_load_http_transport(self):
        # A frontend that uses its own transport can allocate identities without
        # loading aiohttp or engine/tokenizer dependencies.
        subprocess.run(
            [sys.executable, "-c", "\n".join([
                "import sys",
                "from evolving_infra.router.client.agent import Agent, Request, RequestIdentity",
                "from evolving_infra.router.client.types import BackendClient, RequestScheduler",
                "from evolving_infra.router.client.scheduler import Request as CompatibleRequest",
                "assert CompatibleRequest is Request",
                "assert 'aiohttp' not in sys.modules, sorted(sys.modules)",
            ])],
            check=True,
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
        )


class IdentityTests(unittest.TestCase):
    def test_forks_share_problem_and_have_independent_steps(self):
        root = Agent(Scheduler('http://localhost:9000'), 'problem')
        with ThreadPoolExecutor(max_workers=8) as pool:
            children = list(pool.map(lambda _: root.fork(), range(100)))
        self.assertEqual({a.subagent_id for a in children}, set(range(1, 101)))
        grandchild = children[5].fork()
        self.assertEqual(grandchild.agent_id, root.agent_id)
        self.assertEqual(grandchild.parent_subagent_id, children[5].subagent_id)
        self.assertEqual(grandchild.subagent_id, 101)
        self.assertEqual(children[5].request({}).identity.request_id, 0)
        self.assertEqual(children[5].request({}).identity.request_id, 1)
        self.assertEqual(grandchild.request({}).identity.request_id, 0)
        with ThreadPoolExecutor(max_workers=8) as pool:
            requests = list(pool.map(lambda _: root.request({}), range(100)))
        self.assertEqual({r.identity.request_id for r in requests}, set(range(100)))
        self.assertEqual(len({r.identity.engine_rid for r in requests}), 100)

    def test_closed_branch_does_not_close_siblings_or_reuse_ids(self):
        root = Agent(Scheduler('http://localhost:9000'), 'problem')
        first, second = root.fork(), root.fork()
        first.close()
        with self.assertRaises(RuntimeError): first.request({})
        with self.assertRaises(RuntimeError): first.fork()
        self.assertEqual(second.request({}).identity.subagent_id, 2)
        self.assertEqual(root.fork().subagent_id, 3)

    def test_invalid_identity_and_conflicting_rid(self):
        for args in [('', 0, 0), ('a', -1, 0), ('a', True, 0), ('a', 0, 2**64), ('a', 1.0, 0)]:
            with self.assertRaises(ValueError): RequestIdentity(*args)
        identity = RequestIdentity('problem', 0, 1)
        with self.assertRaises(ValueError):
            Request(identity, {'rid':'different'}).wire_payload(stream=False)
        self.assertNotEqual(RequestIdentity('a:1', 2, 3).engine_rid,
                            RequestIdentity('a', 1, 23).engine_rid)

    def test_retry_keeps_identity_and_cache_namespace_is_explicit(self):
        identity = RequestIdentity('question/你好', 2, 7)
        # Shared with evolving_infra/router/core/src/agent_identity.rs; moving the Python classes
        # must not change which request the Rust runtime sees.
        self.assertEqual(
            identity.engine_rid,
            'agent_e471509b10f0c301d8c86761c428ff065faf2d6d9ad0e8939059aa277dda0759',
        )
        request = Request(identity, {'model':'glm', 'prompt':[1,2], 'cache_salt':'shared'})
        one = request.wire_payload(stream=False)
        self.assertEqual(one, request.wire_payload(stream=False))
        self.assertEqual(one['cache_salt'], 'shared')
        self.assertEqual(one['prompt'], [1,2])
        self.assertNotIn('cache_salt', Request(identity, {}).wire_payload(stream=False))
        context = json.loads(identity.headers()[CONTEXT_HEADER])
        self.assertEqual(context, identity.as_dict())
        self.assertTrue(identity.headers()[CONTEXT_HEADER].isascii())


class BackendHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.received = []
        async def generate(incoming):
            payload = await incoming.json()
            self.received.append((dict(incoming.headers), payload))
            if payload.get('fail'):
                raise web.HTTPServiceUnavailable(text='worker unavailable')
            if payload['stream']:
                response = web.StreamResponse(headers={'Content-Type':'text/event-stream'})
                await response.prepare(incoming)
                await response.write(b'data: {"text":"a"}\n\n')
                await response.write(b'data: [DONE]\n\n')
                await response.write_eof()
                return response
            return web.json_response({'rid': payload['rid'], 'ok': True})
        async def loads(incoming):
            self.assertEqual(incoming.query['include'], 'core,queues,prefill_work')
            return web.json_response({'loads':[{'prefill_work':{'epoch':'e', 'sequence':1, 'remaining':{}}}]})
        app = web.Application()
        app.router.add_post('/v1/completions', generate)
        app.router.add_post('/v1/chat/completions', generate)
        app.router.add_get('/v1/loads', loads)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.backend = Backend()
        self.scheduler = Scheduler(f'http://127.0.0.1:{port}', backend=self.backend)

    async def asyncTearDown(self):
        await self.backend.close()
        await self.runner.cleanup()

    async def test_fanout_transmits_identity_and_stream_verbatim(self):
        root = Agent(self.scheduler, 'q')
        agents = [root, root.fork(), root.fork()]
        requests = [a.request({'model':'glm','prompt':[1,2,3]}) for a in agents]
        responses = await asyncio.gather(*(self.scheduler.submit(r) for r in requests))
        self.assertEqual({r['rid'] for r in responses}, {r.identity.engine_rid for r in requests})
        contexts = [json.loads(h[CONTEXT_HEADER]) for h, _ in self.received]
        self.assertEqual({c['subagent_id'] for c in contexts}, {0,1,2})
        self.assertTrue(all('cache_salt' not in payload for _, payload in self.received))
        output = b''.join([chunk async for chunk in self.scheduler.stream(root.request({'model':'glm','prompt':[1]}))])
        self.assertEqual(output, b'data: {"text":"a"}\n\ndata: [DONE]\n\n')
        self.assertEqual(len(self.received), 4)

    async def test_errors_propagate_without_implicit_retry(self):
        import aiohttp
        request = Request(RequestIdentity('q',0,0), {'model':'glm','fail':True})
        with self.assertRaises(aiohttp.ClientResponseError):
            await self.scheduler.submit(request)
        self.assertEqual(len(self.received), 1)
        self.assertEqual((await self.backend.get_progress(self.scheduler.endpoint))['loads'][0]['prefill_work']['epoch'], 'e')

    async def test_scheduler_does_not_close_caller_owned_backend(self):
        await self.scheduler.close()
        self.assertTrue((await self.backend.get_progress(self.scheduler.endpoint))['loads'])


class BackendIdleConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_idle_peer_close_during_busy_loop_does_not_retry_post(self):
        # The peer must keep running while this client's event loop is blocked.
        # stdlib HTTP/1.1 provides the same five-second idle FIN as Uvicorn,
        # without adding a server dependency to these SDK tests.
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from itertools import count
        from threading import Thread
        import time
        import aiohttp

        connection_ids, received = count(), []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def setup(self):
                super().setup()
                self.connection.settimeout(5)
                self.connection_id = next(connection_ids)

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                received.append((self.connection_id, request['rid']))
                body = json.dumps({'connection_id': self.connection_id}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = f'http://127.0.0.1:{server.server_port}'
        try:
            for legacy in (True, False):
                with self.subTest(legacy_keepalive=legacy):
                    session = (aiohttp.ClientSession(connector=aiohttp.TCPConnector(
                        limit=0, keepalive_timeout=15)) if legacy else None)
                    backend = Backend(session=session)
                    start = len(received)
                    def request(step):
                        return Request(RequestIdentity(str(legacy), 0, step),
                                       {'model': 'm', 'prompt': [1]})
                    try:
                        first = await backend.generate(endpoint, request(0))
                        second = await backend.generate(endpoint, request(1))
                        if legacy:
                            self.assertEqual(first, second)  # Caller retains its pooling policy.
                        else:
                            self.assertNotEqual(first, second)
                        deadline = time.monotonic() + 5.5
                        while time.monotonic() < deadline:
                            pass  # Delay FIN callbacks and the client's idle cleanup timer.
                        if legacy:
                            with self.assertRaises(aiohttp.ServerDisconnectedError):
                                await backend.generate(endpoint, request(2))
                            self.assertEqual(len(received) - start, 2)
                        else:
                            third = await backend.generate(endpoint, request(2))
                            self.assertNotEqual(first, third)
                            self.assertEqual(len(received) - start, 3)
                    finally:
                        await backend.close()
                        if session is not None:
                            self.assertFalse(session.closed)  # Caller retains ownership.
                            await session.close()
        finally:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            thread.join(timeout=1)

    async def test_delayed_response_cannot_refresh_a_closed_connection_for_next_post(self):
        # The server continues sending/closing while the client loop is paused.
        # Unlike the already-pooled case above, the response arrives DURING the
        # pause: the old 4s pool timeout sees an almost-zero-age closed socket.
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from itertools import count
        from threading import Thread
        import time
        import aiohttp

        connection_ids, received = count(), []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def setup(self):
                super().setup()
                self.connection.settimeout(.2)
                self.connection_id = next(connection_ids)

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                received.append((self.connection_id, request['rid']))
                if request['prompt'] == [0]:
                    time.sleep(.08)  # Response follows the client's pause start.
                body = json.dumps({'connection_id': self.connection_id}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = f'http://127.0.0.1:{server.server_port}'
        try:
            for caller_owned in (True, False):
                with self.subTest(caller_owned=caller_owned):
                    sent = asyncio.Event()
                    trace = aiohttp.TraceConfig()

                    async def headers_sent(session, context, params):
                        sent.set()

                    trace.on_request_headers_sent.append(headers_sent)
                    trace.freeze()
                    session = (aiohttp.ClientSession(connector=aiohttp.TCPConnector(
                        limit=0, keepalive_timeout=4)) if caller_owned else None)
                    backend = Backend(session=session)
                    # Observe the actual owned connector; do not replace its
                    # factory, socket callbacks or acquisition/release methods.
                    backend._client()._trace_configs.append(trace)
                    start = len(received)
                    requests = [Request(RequestIdentity(str(caller_owned), 0, step),
                                        {'model': 'm', 'prompt': [step]}) for step in (0, 1)]

                    async def pause_after_headers():
                        await sent.wait()
                        await asyncio.sleep(.01)
                        deadline = time.monotonic() + .75
                        while time.monotonic() < deadline:
                            pass

                    blocker = asyncio.create_task(pause_after_headers())
                    try:
                        # Keep both calls in this coroutine: joining a child
                        # response task gives the FIN callback an extra turn.
                        first = await backend.generate(endpoint, requests[0])
                        if caller_owned:
                            # Python 3.10 reports the same closed connection as a body write error.
                            with self.assertRaises((aiohttp.ServerDisconnectedError, aiohttp.ClientOSError)):
                                await backend.generate(endpoint, requests[1])
                            self.assertEqual(received[start:], [
                                (first['connection_id'], requests[0].identity.engine_rid)])
                        else:
                            second = await backend.generate(endpoint, requests[1])
                            self.assertNotEqual(first, second)
                            self.assertEqual(received[start:], [
                                (first['connection_id'], requests[0].identity.engine_rid),
                                (second['connection_id'], requests[1].identity.engine_rid)])
                    finally:
                        blocker.cancel()
                        await asyncio.gather(blocker, return_exceptions=True)
                        await backend.close()
                        if session is not None:
                            self.assertFalse(session.closed)
                            await session.close()
        finally:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            thread.join(timeout=1)


if __name__ == '__main__':
    unittest.main()


class RetentionLifecycleTests(unittest.IsolatedAsyncioTestCase):
    """Explicit END is independent of local close and never ends sibling branches."""

    async def test_end_is_explicit_and_retry_keeps_branch_identity(self):
        class Recorder:
            def __init__(self):
                self.ended = []

            async def end_trajectory(self, agent_id, subagent_id):
                self.ended.append((agent_id, subagent_id))
                return {"accepted": True}

        scheduler = Recorder()
        root = Agent(scheduler, "unique-lifetime")
        child = root.fork()
        root.close()
        self.assertEqual(scheduler.ended, [])
        await root.end_trajectory()
        await root.end_trajectory()
        self.assertEqual(scheduler.ended, [("unique-lifetime", 0)] * 2)
        self.assertEqual(child.request({}).identity.subagent_id, 1)
        with self.assertRaises(RuntimeError):
            root.request({})
