"""Serve CPU input preparation, staging, forwarding, and release over HTTP.

SGLang performs text tokenization; this service validates tokens, computes
cache-query hashes, and sends prepared inputs to caller-selected engines.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import os
import time
import types
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
import anyio
import orjson
from fastapi import FastAPI, Request
from fastapi.responses import Response

from .store import ProtocolError, Store
from .protocol import (
    NodeConfig, Prepare, Release, Forward, ForwardPD, ImageRequest,
)
from .tokenizer import NativeHasher, DynamoHasher, hash_spec, hash_count
from .forward import Upload, ForwardResponse, PDForward, backend_url
from .delivery import CallbackOutbox

LOG = logging.getLogger(__name__)


def encode_json(value) -> bytes:
    """Serialize a JSON-compatible value for HTTP transport.

    Args:
        value: Python value accepted by orjson.dumps.

    Returns:
        UTF-8 JSON bytes.

    Raises:
        TypeError: The value cannot be serialized by orjson.
    """
    return orjson.dumps(value)

class TokenizerService:
    """Serve preparation, forwarding and release through one CPU result store.

    Input: NodeConfig and optional injectable hasher/HTTP client.
    Output: a service instance exposing its FastAPI application as app.
    """
    def __init__(self, config: NodeConfig, hasher=None, client=None):
        """Configure preparation/forwarding routes, local input storage, and counters.

        Args:
            config: Validated fixed node, model, endpoint, and capacity settings.
            hasher: Optional injected token-hash callable; initialized on startup if absent.
            client: Optional injected async HTTP client; service shutdown closes it.

        Returns:
            None; exposes the FastAPI app without starting an HTTP listener.
        """
        self.config, self.hasher = config, hasher
        self.store = Store(max_bytes=config.max_staging_bytes, ttl=config.ttl_seconds)
        self.client = client
        self.ready = False
        self.upload_bytes = self.peak_upload_bytes = 0
        self.callback_failures = 0
        self.callback_outbox = CallbackOutbox(self)
        # The outbox reserves a bounded slot before an exchange enters this set.
        # It also owns disconnected exchanges until their original responses drain.
        self.pd_exchanges = set()
        self.app = FastAPI(lifespan=self.lifespan, docs_url=None, redoc_url=None)
        self.app.add_api_route("/prepare", self.prepare, methods=["POST"])
        self.app.add_api_route("/forward", self.forward, methods=["POST"])
        self.app.add_api_route("/forward_pd", self.forward_pd, methods=["POST"])
        self.app.add_api_route("/release", self.release, methods=["POST"])
        self.app.add_api_route("/image", self.original_image, methods=["POST"])
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_exception_handler(ProtocolError, self.error)
        self.app.add_exception_handler(aiohttp.ClientError, self.http_error)
        self.app.add_exception_handler(TimeoutError, self.http_error)

    @asynccontextmanager
    async def lifespan(self, app):
        """Initialize service dependencies and release input/HTTP resources on shutdown.

        Initializes hashing off the event loop, runs an expiry janitor, and drains
        owned P/D cleanup before closing callback retries and the HTTP client.

        Args:
            app: FastAPI application supplied by its lifespan interface; otherwise unused.

        Yields:
            None while the service is ready inside the lifespan context.
        """
        if self.client is None:
            # A response delivered during a loop stall can enter the idle pool
            # after the peer already closed it. Use fresh connections; a short
            # keepalive timeout alone cannot prevent that one-shot POST race.
            self.client = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0, limit_per_host=0, force_close=True),
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_seconds),
                cookie_jar=aiohttp.DummyCookieJar(),
                trust_env=False,
            )
        try:
            if self.hasher is None:
                implementation = DynamoHasher if self.config.hash_mode == "dynamo" else NativeHasher
                self.hasher = await asyncio.to_thread(implementation, self.config)
            self.ready = True

            async def sweep():
                """Periodically expire staged inputs until application shutdown cancels this task.

                Reads the enclosing service's configured TTL and process-local store.

                Args:
                    None.

                Returns:
                    No normal return; runs until cancellation.
                """
                while True:
                    await asyncio.sleep(min(1, self.config.ttl_seconds))
                    self.store.sweep()

            janitor = asyncio.create_task(sweep())
            try:
                yield
            finally:
                self.ready = False
                janitor.cancel()
                await asyncio.gather(janitor, return_exceptions=True)
                for entry in list(self.store.entries.values()):
                    self.store.release(entry.ref, entry.request_id)
        finally:
            self.callback_outbox.closed = True
            closings = await asyncio.gather(
                *(exchange.aclose() for exchange in list(self.pd_exchanges)), return_exceptions=True)
            draining = await asyncio.gather(
                *(exchange.cleanup_task for exchange in list(self.pd_exchanges)
                  if exchange.cleanup_task is not None), return_exceptions=True)
            for failure in closings + draining:
                if isinstance(failure, BaseException):
                    LOG.error("pd_cleanup_shutdown_failed type=%s", type(failure).__name__)
            await self.callback_outbox.close()
            await self.client.close()

    async def read(self, request: Request, model):
        """Read a bounded JSON request and validate protocol/configuration identity.

        Args:
            request: Incoming FastAPI request whose body has not yet been consumed.
            model: Pydantic request class expected for this endpoint.

        Returns:
            A validated request-model instance also attached to request.state.

        Raises:
            ProtocolError: Readiness, content type, size, timeout, envelope, or config
                validation fails.
        """
        if not self.ready:
            raise ProtocolError(503, "NOT_READY", "Service has not initialized.")
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
            raise ProtocolError(400, "INVALID_REQUEST", "Content-Type must be application/json.")
        data = bytearray()
        try:
            async with asyncio.timeout(self.config.request_timeout_seconds):
                async for chunk in request.stream():
                    if len(data) + len(chunk) > self.config.max_body_bytes:
                        raise ProtocolError(413, "INPUT_TOO_LARGE", "HTTP input exceeds limit.")
                    data.extend(chunk)

            def parse():
                """Decode the captured body and validate its versioned request envelope.

                Runs off the event loop using the captured body bytes and model class.

                Args:
                    None.

                Returns:
                    An instance of the enclosing expected Pydantic request model.

                Raises:
                    ProtocolError: The protocol envelope or version is invalid.
                    ValueError: JSON decoding or Pydantic field validation fails.
                """
                obj = orjson.loads(data)
                if not isinstance(obj, dict) or type(obj.get("protocol_version")) is not int:
                    raise ProtocolError(400, "INVALID_REQUEST", "Invalid request envelope.")
                if obj["protocol_version"] != 1:
                    raise ProtocolError(400, "UNSUPPORTED_VERSION", "Expected protocol version 1.")
                return model.model_validate(obj)

            result = await asyncio.to_thread(parse)
        except (ValueError, RecursionError) as exc:
            raise ProtocolError(400, "INVALID_REQUEST", "Invalid JSON or request fields.") from exc
        except TimeoutError as exc:
            raise ProtocolError(408, "INVALID_REQUEST", "Request body upload timed out.") from exc
        request.state.identity = result
        if result.config_id != self.config.config_id:
            raise ProtocolError(409, "CONFIG_MISMATCH", "Request config differs from this node.")
        return result

    @staticmethod
    def validate_prepare(req: Prepare):
        """Reject unsupported input and sampling fields before preparation.

        Args:
            req: Validated Prepare envelope with input and generation dictionaries.

        Returns:
            None when the request uses a supported single-output input format.

        Raises:
            ProtocolError: Input format, token IDs, sampling options, or generation fields are
                invalid.
        """
        inp, gen = req.input, req.generation
        keys = {
            "model",
            "prompt",
            "messages",
            "input_ids",
            "image_data",
            "add_special_tokens",
            "tools",
            "tool_choice",
            "reasoning_effort",
            "continue_final_message",
            "chat_template_kwargs",
        }
        if set(inp) - keys or sum(key in inp for key in ("prompt", "messages", "input_ids")) != 1:
            raise ProtocolError(
                400, "INVALID_REQUEST", "Supply exactly one supported input format."
            )
        if "input_ids" in inp:
            if not TokenizerService.valid_ids(inp["input_ids"]):
                raise ProtocolError(400, "INVALID_REQUEST", "input_ids must be a nonempty uint32 array.")
            if set(inp) - {"model", "input_ids", "image_data"}:
                raise ProtocolError(422, "UNSUPPORTED_OPTION", "Text/template options cannot change supplied input_ids.")
        elif "prompt" in inp:
            if (
                not isinstance(inp["prompt"], str)
                or type(inp.get("add_special_tokens")) is not bool
            ):
                raise ProtocolError(
                    400,
                    "INVALID_REQUEST",
                    "Prompt requires a string and explicit special-token option.",
                )
            if set(inp) - {"model", "prompt", "add_special_tokens", "image_data"}:
                raise ProtocolError(
                    422,
                    "UNSUPPORTED_OPTION",
                    "Chat options cannot be applied to a rendered prompt.",
                )
        elif not isinstance(inp["messages"], list) or not inp["messages"]:
            raise ProtocolError(400, "INVALID_REQUEST", "Messages must be a nonempty array.")
        if "image_data" in inp:
            if "messages" in inp or not isinstance(inp["image_data"], list) or not inp["image_data"]:
                raise ProtocolError(400, "INVALID_REQUEST", "image_data requires a prompt or input_ids and uploaded images.")
        for message in inp.get("messages", []):
            if not isinstance(message, dict):
                raise ProtocolError(400, "INVALID_REQUEST", "Each message must be an object.")
            content = message.get("content")
            if isinstance(content, list) and any(
                not isinstance(p, dict) or p.get("type") not in {"text", "image_url"} for p in content
            ):
                raise ProtocolError(422, "UNSUPPORTED_INPUT", "Only text and uploaded images are supported.")
        if set(gen) - {"sampling_params", "stream", "return_prompt_token_ids"}:
            raise ProtocolError(
                422, "UNSUPPORTED_OPTION", "Generation option is outside the supported protocol."
            )
        params = gen.get("sampling_params")
        if not isinstance(params, dict) or type(gen.get("stream")) is not bool:
            raise ProtocolError(
                400, "INVALID_REQUEST", "Sampling parameters and stream are required."
            )
        budget = params.get("max_new_tokens")
        if type(budget) is not int or not 0 <= budget <= 4294967295:
            raise ProtocolError(422, "UNSUPPORTED_OPTION", "Invalid max_new_tokens budget.")
        if (
            type(params.get("n", 1)) is not int
            or params.get("n", 1) != 1
            or params.get("beam_width") not in (None, 1)
        ):
            raise ProtocolError(422, "UNSUPPORTED_OPTION", "Only one output sequence is supported.")
        if type(gen.get("return_prompt_token_ids", False)) is not bool:
            raise ProtocolError(422, "UNSUPPORTED_OPTION", "return_prompt_token_ids must be a boolean.")
        internal = {
            "is_normalized",
            "stop_strs",
            "stop_regex_strs",
            "stop_str_max_len",
            "stop_regex_max_len",
        }
        if internal & params.keys():
            raise ProtocolError(
                422, "UNSUPPORTED_OPTION", "Internal sampling fields cannot be supplied."
            )

    async def tokenize(self, inp: dict) -> bytes:
        """Call the configured SGLang tokenizer and bound its response size.

        Args:
            inp: Native SGLang tokenization request dictionary.

        Returns:
            Raw JSON bytes from a successful tokenizer HTTP response.

        Raises:
            ProtocolError: The response exceeds its bound or the tokenizer returns an error
                status.
        """
        body = await asyncio.to_thread(encode_json, inp)
        # Enough for a JSON list of uint32 IDs plus count/model metadata.
        limit = self.config.context_length * 12 + 65536
        async with self.client.post(
            self.config.tokenize_endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            allow_redirects=False,
        ) as response:
            data = bytearray()
            async for part in response.content.iter_any():
                if len(data) + len(part) > limit:
                    raise ProtocolError(413, "INPUT_TOO_LARGE", "Tokenize response exceeds limit.")
                data.extend(part)
            if response.status != 200:
                raise ProtocolError(
                    response.status,
                    "TOKENIZE_ERROR",
                    f"SGLang tokenize returned HTTP {response.status}.",
                )
            return bytes(data)

    @staticmethod
    def valid_ids(ids):
        """Check that token IDs form a nonempty list of unsigned 32-bit integers.

        Args:
            ids: Untrusted token-list value.

        Returns:
            True for valid IDs, otherwise False; booleans are rejected.
        """
        return isinstance(ids, list) and bool(ids) and all(
            type(token) is int and 0 <= token <= 2**32 - 1 for token in ids
        )

    def pack(self, raw: bytes, req: Prepare):
        """Validate a native tokenizer response and prepare its engine input body.

        Args:
            raw: JSON bytes returned by the native SGLang tokenizer.
            req: Preparation request containing identity and generation options.

        Returns:
            A tuple of serialized input bytes, token count, and cache-query metadata.

        Raises:
            ProtocolError: Tokenizer data, context budget, or hash output is invalid.
        """
        try:
            decoded = orjson.loads(raw)
            ids = decoded["tokens"]
            if not self.valid_ids(ids) or decoded["count"] != len(ids):
                raise ValueError("Invalid token list/count")
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise ProtocolError(502, "TOKENIZE_ERROR", "Invalid SGLang tokenization result.") from exc
        return self.pack_ids(ids, req)

    def pack_ids(self, ids: list[int], req: Prepare):
        """Hash validated input tokens and serialize their generation request.

        Args:
            ids: Already validated model token IDs; no retokenization occurs.
            req: Preparation request providing identity and generation settings.

        Returns:
            A tuple of serialized input bytes, prompt-token count, and hash metadata.

        Raises:
            ProtocolError: The context budget is exceeded or hash/serialization output is
                invalid.
        """
        try:
            image_data = getattr(req, "input", {}).get("image_data")
            image_fields = {}
            cache_ids = ids
            if image_data is not None:
                if self.config.image_config is None:
                    raise ProtocolError(422, "UNSUPPORTED_INPUT", "This configuration does not accept images.")
                from ._engine import image_layout

                prepare_image_layout = image_layout().prepare_image_layout

                try:
                    layout, _expanded, cache_ids, _items = prepare_image_layout(
                        ids, image_data, self.config.image_config
                    )
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    raise ProtocolError(400, "INVALID_IMAGE", "Uploaded images or their layout are invalid.") from exc
                image_fields = {"image_data": image_data, "prepared_image_layout": layout}
            count = len(cache_ids)
            if (
                count + req.generation["sampling_params"]["max_new_tokens"]
                > self.config.context_length
            ):
                raise ProtocolError(413, "INPUT_TOO_LARGE", "Input and output exceed context.")
            hashes = self.hasher(cache_ids)
            if len(hashes) != hash_count(self.config, count):
                raise ValueError("Invalid hash count")
            content = {
                "namespace": self.config.config_id,
                "hash_spec": hash_spec(self.config),
                "blocks": [
                    {
                        "hash": digest,
                        "valid_tokens": min(
                            self.config.page_size, count - i * self.config.page_size
                        ),
                    }
                    for i, digest in enumerate(hashes)
                ],
            }
            payload = {
                **req.generation,
                "rid": req.request_id,
                "input_ids": ids,
                **image_fields,
                "return_prompt_token_ids": req.generation.get("return_prompt_token_ids", False),
            }
            if self.config.cache_salt:
                payload["cache_salt"] = self.config.cache_salt
            if self.config.extra_key:
                payload["extra_key"] = self.config.extra_key
            return encode_json(payload), count, content
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise ProtocolError(
                502, "TOKENIZE_ERROR", "Invalid SGLang tokenization result."
            ) from exc

    async def prepare(self, request: Request):
        """Prepare and stage input without submitting generation to an engine.

        Text calls the native tokenizer; supplied IDs bypass tokenization. Stores
        only the serialized engine input and releases it if response construction fails.

        Args:
            request: HTTP /prepare request with identity, one input format, and generation
                settings.

        Returns:
            A JSON Response with opaque reference, hashes, token counts, and expiry.

        Raises:
            ProtocolError: Preparation, admission, model/config checks, or staging capacity
                fails.
        """
        started = time.perf_counter()
        req = await self.read(request, Prepare)
        self.callback_outbox.check_capacity()
        self.validate_prepare(req)
        if self.config.image_config is None and (
            "image_data" in req.input or any(
                part.get("type") == "image_url"
                for message in req.input.get("messages", [])
                for part in (message.get("content") or []) if isinstance(part, dict)
            )
        ):
            raise ProtocolError(422, "UNSUPPORTED_INPUT", "This configuration does not accept images.")
        if req.input.get("model", self.config.model_path) != self.config.model_path:
            raise ProtocolError(409, "CONFIG_MISMATCH", "Input model differs from fixed config.")
        input_format = next(key for key in ("prompt", "messages", "input_ids") if key in req.input)
        remote_started = time.perf_counter()
        image_data = req.input.get("image_data")
        if input_format == "input_ids":
            ids = req.input.pop("input_ids")
            req.input = {"image_data": image_data} if image_data is not None else {}
            remote_finished = remote_started  # This path deliberately performs no tokenization.
            data, count, content = await asyncio.to_thread(self.pack_ids, ids, req)
            del ids
        else:
            tokenize_input = {**req.input, "model": self.config.model_path}
            tokenize_input.pop("image_data", None)
            if input_format == "messages":
                # The native /tokenize endpoint renders image placeholders without
                # fetching images. Do not upload the image bytes to it as well.
                image_data = []
                messages = []
                for message in req.input["messages"]:
                    copied = dict(message)
                    if isinstance(message.get("content"), list):
                        parts = []
                        for part in message["content"]:
                            if part["type"] == "image_url":
                                image = part.get("image_url")
                                if not isinstance(image, dict) or not isinstance(image.get("url"), str):
                                    raise ProtocolError(400, "INVALID_IMAGE", "image_url requires uploaded image content.")
                                image_data.append(image["url"])
                                part = {**part, "image_url": {**image, "url": "https://prepared.invalid/image"}}
                            parts.append(part)
                        copied["content"] = parts
                    messages.append(copied)
                tokenize_input["messages"] = messages
                image_data = image_data or None
            if image_data is not None and self.config.image_config is None:
                raise ProtocolError(422, "UNSUPPORTED_INPUT", "This configuration does not accept images.")
            raw = await self.tokenize(tokenize_input)
            remote_finished = time.perf_counter()
            req.input = {"image_data": image_data} if image_data is not None else {}
            data, count, content = await asyncio.to_thread(self.pack, raw, req)
            del raw
        req.input.clear()  # Serialized ownership now includes any uploaded images.
        packed = time.perf_counter()
        entry = self.store.put(req.request_id, data, req.generation["stream"])
        del data
        result = {
            "protocol_version": 1,
            "node_id": self.config.node_id,
            "request_id": req.request_id,
            "ref": entry.ref,
            "config_id": req.config_id,
            "prompt_tokens": count,
            "max_new_tokens": req.generation["sampling_params"]["max_new_tokens"],
            "content": content,
            "expires_in_ms": self.config.ttl_seconds * 1000,
        }
        try:
            response = Response(
                await asyncio.to_thread(encode_json, result), media_type="application/json"
            )
        except BaseException:
            self.store.release(entry.ref, entry.request_id)
            raise
        LOG.info(
            "prepared request_id=%r input_format=%s seconds=%.6f tokenize_http_seconds=%.6f hash_pack_seconds=%.6f tokens=%d cached_bytes=%d",
            req.request_id,
            input_format,
            time.perf_counter() - started,
            remote_finished - remote_started,
            packed - remote_finished,
            count,
            self.store.bytes_used,
        )
        return response

    async def forward(self, request: Request):
        """Transfer one stored input to the caller-selected allowed engine once.

        Taking the input removes it from the store. The uploader owns retained
        bytes until completion or cleanup; failed generation is not retried.

        Args:
            request: HTTP /forward request naming the prepared reference and selected target.

        Returns:
            A buffered upstream Response or a streaming ForwardResponse.

        Raises:
            ProtocolError: The request/target/reference is invalid or the response exceeds its
                bound.
        """
        req = await self.read(request, Forward)
        self.validate_target(req.target, req.config_id)
        entry = self.store.take(req.ref, req.request_id)
        stream = entry.stream
        upload = Upload(entry.data, self, req.request_id)
        entry.data = b""
        upstream = None
        handed_off = False
        try:
            request.state.forward_started = True
            upstream = await self.client.post(
                req.target.endpoint,
                data=upload,
                headers={"Content-Type": "application/json", "Content-Length": str(upload.size)},
                allow_redirects=False,
            )
            headers = {
                "X-Tokenizer-Protocol-Version": "1",
                "X-Tokenizer-Response-Origin": "upstream",
            }
            if "content-type" in upstream.headers:
                headers["content-type"] = upstream.headers["content-type"]
            if stream:
                result = ForwardResponse(upstream, headers, upload)
                handed_off = True
                return result
            body = bytearray()
            async for part in upstream.content.iter_any():
                if len(body) + len(part) > self.config.max_response_bytes:
                    raise ProtocolError(
                        502, "FORWARD_IO_ERROR", "Upstream response exceeds buffer limit."
                    )
                body.extend(part)
            return Response(bytes(body), status_code=upstream.status, headers=headers)
        finally:
            if upstream is not None and not handed_off:
                upstream.close()
            if not handed_off:
                await upload.aclose()

    def validate_target(self, target, config_id):
        """Check that an engine endpoint is allowed for this input configuration.

        Args:
            target: Validated target carrying endpoint and config_id.
            config_id: Configuration identity attached to the prepared input.

        Returns:
            None when the endpoint and both configuration IDs match.

        Raises:
            ProtocolError: The endpoint is not allowed or its configuration differs.
        """
        if target.endpoint not in self.config.targets:
            raise ProtocolError(403, "TARGET_NOT_ALLOWED", "Target is not allowed.")
        if (
            target.config_id != config_id
            or self.config.targets[target.endpoint] != config_id
        ):
            raise ProtocolError(409, "CONFIG_MISMATCH", "Target config does not match input.")

    async def forward_pd(self, request: Request):
        """Consume a prepared input and execute an already selected split P/D plan.

        Reserves callback capacity before taking input, then transfers ownership
        to PDForward. Does not select nodes or retry generation.

        Args:
            request: HTTP /forward_pd request with distinct targets and authorized callback.

        Returns:
            The decode Response or PDForwardResponse stream.

        Raises:
            ProtocolError: Plan, targets, callback, stored input, or engine execution fails.
        """
        req = await self.read(request, ForwardPD)
        self.validate_pd(req)
        exchange = PDForward(self, req)
        entry = None
        handed_off = False
        try:
            entry = self.store.take(req.ref, req.request_id)
            await exchange.prepare(entry)
            request.state.forward_started = True
            result = await exchange.response()
            if entry.stream:
                handed_off = True
            return result
        finally:
            if entry is not None:
                entry.data = b""
            if not handed_off:
                await exchange.aclose()

    def validate_pd(self, req):
        """Check a split P/D plan's targets and callback before consuming its input.

        Args:
            req: Validated ForwardPD plan.

        Returns:
            None when both targets are allowed and distinct and the callback is allowed.

        Raises:
            ProtocolError: A target or the callback is not allowed, or P and D coincide.
        """
        for leg in (req.prefill, req.decode, *(req.decode_candidates or ())):
            self.validate_target(leg.target, req.config_id)
        if any(req.prefill.target.endpoint == leg.target.endpoint
               for leg in (req.decode, *(req.decode_candidates or ()))):
            raise ProtocolError(400, "INVALID_REQUEST", "Same-node execution uses /forward.")
        if req.callback.endpoint not in self.config.callback_targets:
            raise ProtocolError(403, "CALLBACK_NOT_ALLOWED", "Callback is not allowed.")

    async def continue_pd(self, plan, payload, generated):
        """Start the rest of a request whose D-leg the Backend migrated.

        Prepares the input plus the already generated tokens as a new request,
        has the Backend that issued `plan` place it, and starts its P/D exchange.
        The new request keeps the sampling settings with the remaining output
        budget and forced tokens.

        Args:
            plan: The migrated request's ForwardPD plan.
            payload: The migrated request's engine input body, without stream.
            generated: Output token IDs it already sent downstream.

        Returns:
            A started PDForward whose decode stream continues the output.

        Raises:
            ProtocolError: Nothing is left to generate, or preparation, placement,
                plan validation or engine submission fails.
        """
        sampling = dict(payload["sampling_params"])
        sampling["max_new_tokens"] -= len(generated)
        if sampling["max_new_tokens"] <= 0:
            raise ProtocolError(502, "PD_RESUME_FAILED", "No output is left to resume.")
        if "forced_output_token_ids" in sampling:
            sampling["forced_output_token_ids"] = sampling["forced_output_token_ids"][len(generated):]
        # Engines abort every request whose ID starts with the aborted one, and the
        # Backend aborts the migrated request's ID on its engines: the resumed ID
        # must not extend it.
        rid = f"m{len(generated)}-{payload['rid']}"
        generation = {key: value for key, value in payload.items()
                      if key not in ("rid", "input_ids", "cache_salt", "extra_key",
                                     "image_data", "prepared_image_layout", "prepared_image_source")}
        # The engine sends prompt token IDs with the first event only; the client
        # already has them unless the migrated request sent nothing.
        generation.update(sampling_params=sampling, stream=True,
                          return_prompt_token_ids=bool(payload.get("return_prompt_token_ids")) and not generated)
        req = types.SimpleNamespace(request_id=rid, generation=generation,
                                    input={"image_data": payload["image_data"]} if "image_data" in payload else {})
        data, count, content = await asyncio.to_thread(self.pack_ids, payload["input_ids"] + generated, req)
        entry = self.store.put(rid, data, True)
        del data
        prepared = {
            "protocol_version": 1, "node_id": self.config.node_id, "request_id": rid,
            "ref": entry.ref, "config_id": plan.config_id, "prompt_tokens": count,
            "max_new_tokens": sampling["max_new_tokens"], "content": content,
            "expires_in_ms": self.config.ttl_seconds * 1000,
        }
        exchange = None
        try:
            placement = backend_url(plan, "prepared")
            async with self.client.post(placement, json={"prepared": prepared},
                                        allow_redirects=False) as response:
                if response.status != 200:
                    raise ProtocolError(502, "PD_RESUME_FAILED",
                                        f"Resumed request placement returned HTTP {response.status}.")
                value = await response.json()
            try:
                follow = ForwardPD.model_validate(value)
            except ValueError as exc:
                raise ProtocolError(502, "PD_RESUME_FAILED", "Invalid resumed execution plan.") from exc
            if (follow.request_id, follow.ref, follow.config_id) != (rid, entry.ref, plan.config_id):
                raise ProtocolError(502, "PD_RESUME_FAILED", "Resumed plan identity differs.")
            self.validate_pd(follow)
            exchange = PDForward(self, follow)
            await exchange.prepare(self.store.take(follow.ref, follow.request_id))
            await exchange.start()
            return exchange
        except BaseException:
            with anyio.CancelScope(shield=True):
                if entry.ref in self.store.entries:
                    self.store.release(entry.ref, rid)
                if exchange is not None:
                    await exchange.aclose()
            raise

    async def original_image(self, request: Request):
        """Return uploaded file bytes only during staging or active P/D execution.

        Decode normally needs no pixels. A failed import may prefill locally;
        its opaque request reference permits retrieval from the existing owner.
        No second image cache or independent image lifetime is introduced.
        """
        req = await self.read(request, ImageRequest)
        self.store.sweep()
        entry = self.store.entries.get(req.ref)
        data = entry.data if entry is not None and entry.request_id == req.request_id else None
        if data is None:
            for exchange in self.pd_exchanges:
                if (exchange.plan.ref, exchange.plan.request_id) == (req.ref, req.request_id):
                    if exchange.input_body is not None:
                        data = exchange.input_body + b"}"
                    break
        if data is None:
            raise ProtocolError(404, "REF_NOT_FOUND", "Original image owner has released this request.")

        def extract():
            images = orjson.loads(data).get("image_data", [])
            if req.image_index >= len(images):
                raise ProtocolError(400, "INVALID_IMAGE", "Image index is outside this request.")
            encoded = images[req.image_index]
            if encoded.startswith("data:"):
                encoded = encoded.split(",", 1)[1]
            return base64.b64decode(encoded, validate=True)

        return Response(await asyncio.to_thread(extract), media_type="application/octet-stream")

    async def release(self, request: Request):
        """Delete an unused staged input without cancelling generation or deleting KV.

        Args:
            request: HTTP /release request naming the input reference and identity.

        Returns:
            A JSON Response echoing identity with released or not_found status.

        Raises:
            ProtocolError: The request envelope or stored identity is invalid.
        """
        req = await self.read(request, Release)
        return Response(
            encode_json(
                {**req.model_dump(), "status": self.store.release(req.ref, req.request_id)}
            ),
            media_type="application/json",
        )

    async def health(self):
        """Report readiness, supported input modes, and local resource counters.

        Reports current staged/upload bytes and callback backlog without engine
        queries or scheduling work.

        Args:
            None.

        Returns:
            A JSON Response with HTTP 200 when ready, otherwise 503.
        """
        return Response(
            encode_json(
                {
                    "protocol_version": 1,
                    "node_id": self.config.node_id,
                    "ready": self.ready,
                    "config_id": self.config.config_id,
                    "input_formats": ["prompt", "messages", "input_ids"],
                    "forward_transports": ["sglang_generate"],
                    "forward_modes": ["single", "pd"],
                    "hash_spec": hash_spec(self.config),
                    "cache": {
                        "entries": len(self.store.entries),
                        "bytes": self.store.bytes_used,
                        "peak_bytes": self.store.peak_bytes,
                        "capacity_bytes": self.store.max_bytes,
                        "evicted": self.store.evicted,
                        "expired": self.store.expired,
                    },
                    "upload_bytes": self.upload_bytes,
                    "peak_upload_bytes": self.peak_upload_bytes,
                    "callback_failures": self.callback_failures,
                    "callback_outbox": self.callback_outbox.summary(),
                }
            ),
            media_type="application/json",
            status_code=200 if self.ready else 503,
        )

    async def http_error(self, request: Request, exc: Exception):
        """Translate upstream client/time-out failures into the service error envelope.

        Args:
            request: Incoming request carrying any known identity and forwarding state.
            exc: Upstream client or timeout exception.

        Returns:
            An HTTP 504 response for timeouts, otherwise HTTP 502.
        """
        status = 504 if isinstance(exc, TimeoutError) else 502
        return await self.error(
            request, ProtocolError(status, "UPSTREAM_IO_ERROR", type(exc).__name__)
        )

    async def error(self, request: Request, exc: ProtocolError):
        """Format a protocol failure with safe request identity and forwarding metadata.

        Args:
            request: Incoming request with optional validated identity in request.state.
            exc: ProtocolError containing status, stable code, and safe message.

        Returns:
            A JSON Response using the exception's HTTP status.
        """
        identity = getattr(request.state, "identity", None)
        LOG.warning("protocol_error rid=%r status=%s code=%s forward_started=%s",
                    getattr(identity, "request_id", None), exc.status, exc.code,
                    getattr(request.state, "forward_started", None))
        return Response(
            encode_json(
                {
                    "protocol_version": 1,
                    "request_id": getattr(identity, "request_id", None),
                    "ref": getattr(identity, "ref", None),
                    "error": {
                        "code": exc.code,
                        "message": str(exc),
                        "forward_started": getattr(request.state, "forward_started", None),
                    },
                }
            ),
            media_type="application/json",
            status_code=exc.status,
        )


def main():
    """Load command-line configuration and run one CPU input-preparation service.

    Reads --config, --host, and --port from argv. Hides CUDA devices in this
    process, validates the JSON configuration, then starts Uvicorn.

    Args:
        None.

    Returns:
        None after server shutdown.

    Raises:
        SystemExit: Argument parsing exits on invalid or help arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    config = NodeConfig.model_validate_json(Path(args.config).read_text())
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        TokenizerService(config).app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=5,
        timeout_graceful_shutdown=config.request_timeout_seconds,
    )


if __name__ == "__main__":
    main()
