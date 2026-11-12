"""Agent gateway: an OpenAI-compatible endpoint that adds agent identities for coding-agent harnesses.

Harnesses (Pi, Claude Code, Codex, ...) send plain chat completions without `x-agent-context`.
The gateway infers each request's agent and branch from its messages (sessions.Tracker), sends
it to the router with that identity, and ends a branch with POST /kv_sessions/end once it has
been idle, so the router keeps the KV of live branches and releases the rest. Everything else
of the request and of the reply passes through unchanged.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time

import aiohttp
from aiohttp import web

from evolving_infra.router.gateway.sessions import Tracker

log = logging.getLogger("evolving_infra.gateway")
PLACEMENT_RETRIES = 2  # 503 symmetric_pd_no_placement: the router's pair changed role during placement
HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "x-agent-context",
       "authorization", "x-api-key"}


class Gateway:
    def __init__(self, router_url: str, *, retention: bool = True, idle_end_secs: float = 600.0,
                 sweep_secs: float = 30.0):
        self.router = router_url.rstrip("/")
        self.retention = retention
        self.idle_end_secs = idle_end_secs
        self.sweep_secs = sweep_secs
        self.tracker = Tracker()
        self.session: aiohttp.ClientSession | None = None
        self.sweeper: asyncio.Task | None = None
        self.ended = 0

    # ---------------------------------------------------------------------------------- lifecycle
    async def start(self, _app=None):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=30))
        self.sweeper = asyncio.create_task(self.sweep())

    async def stop(self, _app=None):
        if self.sweeper:
            self.sweeper.cancel()
        for branch in self.tracker.idle(0.0, now=float("inf")):
            await self.end(branch)
        if self.session:
            await self.session.close()

    async def sweep(self):
        while True:
            await asyncio.sleep(self.sweep_secs)
            for branch in self.tracker.idle(self.idle_end_secs):
                await self.end(branch)

    async def end(self, branch) -> None:
        if not (self.retention and branch.sent):
            return
        try:
            async with self.session.post(f"{self.router}/kv_sessions/end", json={
                    "agent_id": branch.agent_id, "subagent_id": branch.subagent_id}) as r:
                if r.status >= 300 and "unknown KV session" not in (text := await r.text()):
                    log.warning("end %s/%d: HTTP %d %s", branch.agent_id, branch.subagent_id, r.status, text[:200])
                else:
                    self.ended += 1
        except Exception as e:  # noqa: BLE001 - the branch is forgotten either way
            log.warning("end %s/%d: %r", branch.agent_id, branch.subagent_id, e)

    # ---------------------------------------------------------------------------------- routes
    async def chat(self, request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"error": {"message": "body must be JSON", "type": "invalid_request_error"}},
                                     status=400)
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return await self.forward(request, body, None)
        body.pop("rid", None)  # the router derives it from the identity
        assignment = self.tracker.assign(messages)
        outcome = {"sent": False, "ok": False}
        try:
            return await self.forward(request, body, assignment.header, outcome)
        finally:  # also when the client disconnects and this handler is cancelled
            self.tracker.finish(assignment, reached_router=outcome["sent"], ok=outcome["ok"])

    async def forward(self, request: web.Request, body, identity: str | None, outcome: dict | None = None):
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        headers["Content-Type"] = "application/json"
        if identity:
            headers["x-agent-context"] = identity
        outcome = {} if outcome is None else outcome
        out = None
        try:
            outcome["sent"] = True  # from here the router may have registered the branch
            for attempt in range(PLACEMENT_RETRIES + 1):
                upstream = await self.session.post(self.router + request.path, json=body, headers=headers)
                if upstream.status != 503 or attempt == PLACEMENT_RETRIES:
                    break
                text = await upstream.text()
                upstream.release()
                if "symmetric_pd_no_placement" not in text:  # another 503: pass it on as it is
                    return web.Response(status=503, text=text, content_type="application/json")
                await asyncio.sleep(0.5 * (attempt + 1))  # a worker left its role while placed: place again
            async with upstream:
                out = web.StreamResponse(status=upstream.status, headers={
                    k: v for k, v in upstream.headers.items() if k.lower() not in HOP})
                await out.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await out.write(chunk)
                await out.write_eof()
                outcome["ok"] = 200 <= upstream.status < 300
        except aiohttp.ClientError as e:
            if out is not None and out.prepared:  # headers are sent: end the reply where it broke
                log.warning("upstream broke mid-reply: %r", e)
                return out
            out = web.json_response({"error": {"message": f"router unreachable: {e}", "type": "server_error"}},
                                    status=502)
        return out

    async def passthrough(self, request: web.Request) -> web.StreamResponse:
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        data = await request.read()
        async with self.session.request(request.method, self.router + request.path_qs, data=data,
                                        headers=headers) as upstream:
            return web.Response(status=upstream.status, body=await upstream.read(), headers={
                k: v for k, v in upstream.headers.items() if k.lower() not in HOP})

    async def stats(self, _request: web.Request) -> web.Response:
        branches = self.tracker.branches
        return web.json_response({"agents": len({b.agent_id for b in branches}), "branches": len(branches),
                                  "inflight": sum(b.inflight for b in branches), "ended": self.ended,
                                  "retention": self.retention, "idle_end_secs": self.idle_end_secs})

    def app(self) -> web.Application:
        app = web.Application(client_max_size=256 * 2**20)
        app.on_startup.append(self.start)
        app.on_cleanup.append(self.stop)
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_get("/gateway/stats", self.stats)
        for path in ("/v1/completions", "/v1/models", "/v1/tokenize", "/v1/detokenize", "/health", "/readyz"):
            app.router.add_route("*", path, self.passthrough)
        return app


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m evolving_infra.router.gateway",
                                     description="OpenAI-compatible gateway that adds agent identities.")
    parser.add_argument("--router-url", default="http://127.0.0.1:30900")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30800)
    parser.add_argument("--no-retention", action="store_true",
                        help="the router runs without KV session retention: send no session END")
    parser.add_argument("--idle-end-secs", type=float, default=600.0,
                        help="end a branch after this long without a request (default 600)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    gateway = Gateway(args.router_url, retention=not args.no_retention, idle_end_secs=args.idle_end_secs)
    # A client that disconnects cancels its handler, which closes the request to the router.
    web.run_app(gateway.app(), host=args.host, port=args.port, handler_cancellation=True, print=None)


if __name__ == "__main__":
    main()
