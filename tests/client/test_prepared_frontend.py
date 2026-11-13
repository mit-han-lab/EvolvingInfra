# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Real TCP protocol tests for descriptor-only placement and prepared forwarding."""
import asyncio
import copy
import json
import unittest

from aiohttp import web

from evolving_infra.router.client.backend import Backend
from evolving_infra.router.client.scheduler import Scheduler
from evolving_infra.router.client.types import RequestIdentity


from evolving_infra.router.frontend import _engine

try:
    _engine.image_layout()
    HAS_IMAGE_LAYOUT = True
except ModuleNotFoundError:
    HAS_IMAGE_LAYOUT = False
NO_IMAGE_LAYOUT = "needs the engine's image layout module (install decoupled-engine)"

def kimi_image_config():
    return {
        "layout_type": "kimi_k25", "image_token_id": 100,
        "patch_size": 14, "merge_kernel_size": 2,
        "in_patch_limit": 16384, "patch_limit_on_one_side": 512,
        "fixed_output_tokens": None,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
    }


class PreparedFrontendTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls, self.runners = [], []
        self.mode, self.plan_change = "json", None
        self.last_identity_header = None
        self.cancel_status = 200
        self.loads_status = 200
        self.merged_loads = {"workers": {"http://prefill:8000": {
            "observation": None, "status": "missing", "sample_age_ms": None,
            "reserved_uncached_tokens": 8192, "active_prefill_reqs": 2,
            "unknown_prefill_reqs": 1, "progress_observed_reqs": 0,
            "workload_mode": "rid_remaining",
        }}}
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.secret = "private-callback-credential"
        self.prepared = {
            "protocol_version": 1, "node_id": "T0", "request_id": RequestIdentity("question", 3, 7).engine_rid,
            "ref": "input-ref", "config_id": "glm-test", "prompt_tokens": 4,
            "max_new_tokens": 2, "expires_in_ms": 300_000,
            "content": {
                "namespace": "glm-test",
                "hash_spec": {"mode": "dynamo", "page_size": 4, "is_eagle": False,
                              "model": "glm", "encoding": "signed_i64"},
                "blocks": [{"hash": -42, "valid_tokens": 4}],
            },
        }

        async def placement(incoming):
            self.last_identity_header = incoming.headers.get("x-agent-context")
            self.calls.append(("place", await incoming.json()))
            plan = self.make_plan()
            if self.plan_change:
                self.plan_change(plan)
            return web.json_response(plan)

        async def cancel(incoming):
            self.calls.append(("cancel", await incoming.json(), incoming.headers.get("Authorization")))
            self.release.set()
            return web.json_response(
                {"cancellation_requested": True, "engine_termination_confirmed": False},
                status=self.cancel_status,
            )

        async def forward(incoming):
            self.calls.append(("forward", await incoming.json()))
            self.entered.set()
            if self.mode == "failure":
                raise web.HTTPBadGateway(text="decode rejected")
            if self.mode == "diagnostic_failure":
                return web.Response(status=502, body=self.error_body, content_type="application/json")
            if self.mode == "redirect":
                raise web.HTTPTemporaryRedirect(location=self.router_url + "/leak")
            if self.mode == "pending":
                await self.release.wait()
                return web.json_response({"text": "late"})
            if self.mode in ("stream", "slow_stream"):
                response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await response.prepare(incoming)
                await response.write(b'data: {"text":"a"}\n\n')
                if self.mode == "slow_stream":
                    await self.release.wait()
                try:
                    await response.write(b"data: [DONE]\n\n")
                    await response.write_eof()
                except ConnectionResetError:
                    pass  # The test deliberately closes its consumer early.
                return response
            return web.json_response({"text": "answer", "meta_info": {"id": self.prepared["request_id"]}})

        async def leak(incoming):
            self.calls.append(("leak", await incoming.read()))
            return web.json_response({})

        async def backend_loads(incoming):
            self.calls.append(("backend_loads", incoming.method, incoming.query_string))
            return web.json_response(self.merged_loads, status=self.loads_status)

        router = web.Application()
        router.router.add_post("/symmetric_pd/prepared", placement)
        router.router.add_post("/symmetric_pd/cancel", cancel)
        router.router.add_post("/leak", leak)
        router.router.add_get("/symmetric_pd/backend/loads", backend_loads)
        self.router_url = await self.serve(router)
        tokenizer = web.Application()
        tokenizer.router.add_post("/forward_pd", forward)
        self.tokenizer_url = await self.serve(tokenizer)
        self.backend = Backend()
        self.scheduler = Scheduler(self.router_url, backend=self.backend)

    async def serve(self, app):
        runner = web.AppRunner(app)
        await runner.setup()
        self.runners.append(runner)
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        return "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1])

    async def asyncTearDown(self):
        self.release.set()
        await self.backend.close()
        for runner in self.runners:
            await runner.cleanup()

    def make_plan(self):
        identity = {key: self.prepared[key] for key in ("protocol_version", "request_id", "config_id", "ref")}
        target = {"transport": "sglang_generate", "config_id": "glm-test"}
        return {
            **identity, "reservation_id": "reservation", "transfer_id": 17,
            "callback": {"endpoint": self.router_url + "/symmetric_pd/events", "token": self.secret},
            "prefill": {"target": {**target, "endpoint": "http://prefill:8000/generate"},
                        "kv_export": {"transfer_id": 17}},
            "decode": {"target": {**target, "endpoint": "http://decode:8000/generate"},
                       "kv_import": {"transfer_id": 17}},
        }

    def assert_cancelled_once(self):
        cancellations = [call for call in self.calls if call[0] == "cancel"]
        self.assertEqual(cancellations, [("cancel", {"reservation_id": "reservation"}, "Bearer " + self.secret)])

    async def test_forward_error_retains_safe_cause_without_payload_or_retry(self):
        self.mode = "diagnostic_failure"
        self.error_body = json.dumps({"error": {"code": "UPSTREAM_IO_ERROR",
            "message": "ServerDisconnectedError", "forward_started": True},
            "prompt": self.secret}).encode()
        with self.assertRaisesRegex(RuntimeError, "HTTP 502.*UPSTREAM_IO_ERROR.*ServerDisconnectedError") as caught:
            await self.scheduler.submit_prepared(self.tokenizer_url, self.prepared)
        self.assertNotIn(self.secret, str(caught.exception))
        self.assertEqual(sum(call[0] == "forward" for call in self.calls), 1)
        self.assert_cancelled_once()

    async def test_stream_error_does_not_include_untrusted_or_oversized_body(self):
        self.mode = "diagnostic_failure"
        for body in [b'not-json-' + self.secret.encode(),
                     b'[' * 1500 + b']' * 1500,
                     json.dumps({"error": {"code": "UPSTREAM_IO_ERROR", "message": self.secret}}).encode(),
                     b'{"error":' + b' ' * 8192 + self.secret.encode()]:
            with self.subTest(size=len(body)):
                self.error_body = body
                with self.assertRaisesRegex(RuntimeError, "HTTP 502") as caught:
                    async for _ in self.scheduler.stream_prepared(self.tokenizer_url, self.prepared):
                        self.fail("No output expected")
                self.assertNotIn(self.secret, str(caught.exception))
                self.assertLess(len(str(caught.exception)), 256)
        self.assertEqual(sum(call[0] == "forward" for call in self.calls), 4)

    async def test_json_places_metadata_then_forwards_exact_plan_once(self):
        original = copy.deepcopy(self.prepared)
        result = await self.scheduler.submit_prepared(self.tokenizer_url, self.prepared)
        self.assertEqual(result["text"], "answer")
        self.assertEqual(result["meta_info"]["id"], original["request_id"])
        self.assertEqual(self.calls, [("place", {"prepared": original}), ("forward", self.make_plan())])
        self.assertEqual(self.prepared, original)

    async def test_image_contracts_are_preserved_for_json_and_streaming(self):
        dsv4 = {
            "image_token_id": 100, "vision_patch_size": 14,
            "vision_downsample_ratio": 3, "vision_max_n_token": 384,
            "vision_min_pixels": 147456, "vision_max_wh_ratio": None,
        }
        for image in (kimi_image_config(), dsv4):
            for mode in ("json", "stream"):
                with self.subTest(layout=image.get("layout_type", "dsv4"), mode=mode):
                    if image.get("layout_type") == "kimi_k25" and not HAS_IMAGE_LAYOUT:
                        self.skipTest(NO_IMAGE_LAYOUT)
                    self.calls.clear()
                    self.mode = mode
                    self.prepared["content"]["hash_spec"]["image_config"] = image
                    original = copy.deepcopy(self.prepared)
                    if mode == "stream":
                        output = b"".join([part async for part in self.scheduler.stream_prepared(
                            self.tokenizer_url, self.prepared)])
                        self.assertIn(b"[DONE]", output)
                    else:
                        result = await self.scheduler.submit_prepared(self.tokenizer_url, self.prepared)
                        self.assertEqual(result["text"], "answer")
                    self.assertEqual(self.calls, [
                        ("place", {"prepared": original}), ("forward", self.make_plan()),
                    ])
                    self.assertEqual(self.prepared, original)

    @unittest.skipUnless(HAS_IMAGE_LAYOUT, NO_IMAGE_LAYOUT)
    async def test_invalid_kimi_image_contract_never_contacts_router(self):
        invalid = []
        for field, value in (("patch_size", 0), ("merge_kernel_size", True),
                             ("image_std", [0.5, 0, 0.5]),
                             ("image_mean", [float("nan"), 0.5, 0.5]),
                             ("fixed_output_tokens", 256),
                             ("image_mean", (0.5, 0.5, 0.5))):
            invalid.append(kimi_image_config() | {field: value})
        missing = kimi_image_config()
        del missing["image_token_id"]
        invalid.extend([missing, kimi_image_config() | {"unexpected": "field"}])
        for image in invalid:
            with self.subTest(image=image):
                self.prepared["content"]["hash_spec"]["image_config"] = image
                with self.assertRaisesRegex(ValueError, "invalid image preprocessing contract"):
                    await self.scheduler.submit_prepared(self.tokenizer_url, self.prepared)
                with self.assertRaisesRegex(ValueError, "invalid image preprocessing contract"):
                    async for _ in self.scheduler.stream_prepared(self.tokenizer_url, self.prepared):
                        self.fail("Invalid image metadata must not start generation")
        self.assertEqual(self.calls, [])

    async def test_retention_identity_is_optional_and_plan_tags_are_preserved(self):
        identity = RequestIdentity("question", 3, 7)
        tag = {"session_id": "controller:branch", "version": 4}

        def tagged(plan):
            plan["prefill"]["kv_session"] = tag
            plan["decode"]["kv_session"] = tag

        self.plan_change = tagged
        await self.scheduler.submit_prepared(
            self.tokenizer_url, self.prepared, identity=identity
        )
        self.assertEqual(json.loads(self.last_identity_header), identity.as_dict())
        forwarded = [call[1] for call in self.calls if call[0] == "forward"][0]
        self.assertEqual(forwarded["prefill"]["kv_session"], tag)
        self.assertEqual(forwarded["decode"]["kv_session"], tag)
        self.assertEqual(self.prepared["request_id"], identity.engine_rid)

    async def test_retention_rejects_mismatched_request_and_mismatched_legs(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            await self.scheduler.submit_prepared(
                self.tokenizer_url,
                self.prepared,
                identity=RequestIdentity("other", 3, 7),
            )
        self.assertEqual(self.calls, [])
        self.plan_change = lambda plan: plan["prefill"].update(
            kv_session={"session_id": "sid", "version": 1}
        )
        with self.assertRaisesRegex(ValueError, "invalid prepared execution plan"):
            await self.scheduler.submit_prepared(self.tokenizer_url, self.prepared)
        self.assertFalse(any(call[0] == "forward" for call in self.calls))

    async def test_merged_load_query_uses_get_without_placement_or_client_ledger(self):
        first = await self.scheduler.get_backend_loads()
        self.assertEqual(first, self.merged_loads)
        self.merged_loads["workers"]["http://prefill:8000"]["reserved_uncached_tokens"] = 2048
        second = await self.backend.get_backend_loads(self.router_url)
        self.assertEqual(second["workers"]["http://prefill:8000"]["reserved_uncached_tokens"], 2048)
        self.assertEqual(first["workers"]["http://prefill:8000"]["reserved_uncached_tokens"], 8192)
        self.assertEqual(self.calls, [("backend_loads", "GET", ""), ("backend_loads", "GET", "")])

    async def test_merged_load_query_error_does_not_fallback_to_placement(self):
        self.loads_status = 503
        with self.assertRaisesRegex(RuntimeError, "Backend load query returned HTTP 503"):
            await self.scheduler.get_backend_loads()
        self.assertEqual(self.calls, [("backend_loads", "GET", "")])

    async def test_stream_is_forwarded_verbatim_and_not_cancelled_on_completion(self):
        self.mode = "stream"
        output = b"".join([part async for part in self.scheduler.stream_prepared(self.tokenizer_url, self.prepared)])
        self.assertEqual(output, b'data: {"text":"a"}\n\ndata: [DONE]\n\n')
        self.assertEqual([call[0] for call in self.calls], ["place", "forward"])

    async def test_wrong_identity_never_forwards_and_error_hides_secret(self):
        self.plan_change = lambda plan: plan.update(request_id="another-request")
        with self.assertRaisesRegex(ValueError, "invalid prepared execution plan") as error:
            await self.scheduler.submit_prepared(self.tokenizer_url, self.prepared)
        self.assertNotIn(self.secret, str(error.exception))
        self.assertEqual([call[0] for call in self.calls], ["place", "cancel"])
        self.assert_cancelled_once()

    async def test_mismatched_transfer_is_rejected_before_forward(self):
        self.plan_change = lambda plan: plan["decode"]["kv_import"].update(transfer_id=18)
        with self.assertRaises(ValueError):
            await self.scheduler.submit_prepared(self.tokenizer_url, self.prepared)
        self.assertEqual([call[0] for call in self.calls], ["place", "cancel"])

    async def test_raw_input_is_rejected_before_router_call(self):
        for raw in ({**self.prepared, "input_ids": [1, 2, 3]},
                    {**self.prepared, "content": {**self.prepared["content"], "tokens": [1, 2, 3]}}):
            with self.assertRaises(ValueError):
                await self.scheduler.submit_prepared(self.tokenizer_url, raw)
        self.assertEqual(self.calls, [])

    async def test_forward_failure_cancels_without_generation_retry(self):
        self.mode = "failure"
        with self.assertRaisesRegex(RuntimeError, "HTTP 502"):
            await self.scheduler.submit_prepared(self.tokenizer_url, self.prepared)
        self.assertEqual([call[0] for call in self.calls], ["place", "forward", "cancel"])
        self.assert_cancelled_once()

    async def test_redirect_does_not_disclose_plan_callback_secret(self):
        self.mode = "redirect"
        with self.assertRaisesRegex(RuntimeError, "HTTP 307"):
            await self.scheduler.submit_prepared(self.tokenizer_url, self.prepared)
        self.assertEqual([call[0] for call in self.calls], ["place", "forward", "cancel"])

    async def test_explicit_stream_close_notifies_rust_backend(self):
        self.mode = "slow_stream"
        stream = self.scheduler.stream_prepared(self.tokenizer_url, self.prepared)
        self.assertIn(b'"text":"a"', await anext(stream))
        await stream.aclose()
        self.assert_cancelled_once()

    async def test_cancelling_waiting_consumer_notifies_rust_backend(self):
        self.mode = "pending"
        task = asyncio.create_task(self.scheduler.submit_prepared(self.tokenizer_url, self.prepared))
        await asyncio.wait_for(self.entered.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assert_cancelled_once()

    async def test_cancelling_stream_consumer_notifies_rust_backend(self):
        self.mode = "slow_stream"
        first_chunk = asyncio.Event()

        async def consume():
            async for _ in self.scheduler.stream_prepared(self.tokenizer_url, self.prepared):
                first_chunk.set()

        task = asyncio.create_task(consume())
        await asyncio.wait_for(first_chunk.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assert_cancelled_once()

    async def test_cleanup_failure_keeps_original_error_and_redacts_credentials(self):
        self.mode, self.cancel_status = "failure", 503
        with self.assertLogs("evolving_infra.router.client.backend.backend", level="WARNING") as logs:
            with self.assertRaisesRegex(RuntimeError, "forwarding returned HTTP 502"):
                await self.scheduler.submit_prepared(self.tokenizer_url, self.prepared)
        self.assertNotIn(self.secret, " ".join(logs.output))
        self.assertIn("Backend must reconcile", " ".join(logs.output))
        self.assert_cancelled_once()


if __name__ == "__main__":
    unittest.main()
