"""Opt-in adapter for token-authoritative trie replay through the frontend (T).

Call ``install(trie.client.Client, Config(...))`` before replay. The frozen
trie workload, tool waits, token checks and latency accumulator remain in use.
This module depends on OpenAI only when constructing its replay response types.
It does not implement placement or retry generation.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
import json
import logging
import math
import time

import aiohttp

from evolving_infra.router.client.agent.identity import RequestIdentity
from evolving_infra.router.client.backend import Backend
from evolving_infra.router.client.backend.backend import PreparedHTTPError, endpoint_url
from evolving_infra.router.client.scheduler import Scheduler


LOG = logging.getLogger(__name__)


class ReplayCleanupTimeout(RuntimeError):
    """Replay owners or streams did not stop within the cleanup budget."""


def _cleanup_failed(original: BaseException, cleanup: BaseException):
    # Transport exceptions can contain request credentials; retain the original
    # exception, but expose only the cleanup exception's class in diagnostics.
    original.add_note(f"Prepared replay cleanup also failed: {type(cleanup).__name__}")
    LOG.warning("Prepared replay cleanup also failed (%s)", type(cleanup).__name__)


def _observe_task(task):
    if not task.cancelled():
        task.exception()


@dataclass(frozen=True)
class Config:
    router_url: str
    tokenizer_url: str
    config_id: str
    model_name: str
    model_path: str
    incremental_streaming_output: bool
    cache_salt: str | None = None
    timeout_s: float = 3600
    max_sse_bytes: int = 16 * 1024**2

    def __post_init__(self):
        endpoint_url(self.router_url)
        endpoint_url(self.tokenizer_url)
        if not all(isinstance(v, str) and v for v in
                   (self.config_id, self.model_name, self.model_path)):
            raise ValueError("fixed tokenizer/model identities are required")
        if type(self.incremental_streaming_output) is not bool:
            raise ValueError("engine incremental_streaming_output must be explicit")
        if self.timeout_s <= 0 or self.max_sse_bytes <= 0:
            raise ValueError("positive transport limits are required")


def prepare_body(config: Config, request: dict, rid: str | None) -> dict:
    """Translate exact replay IDs and uploaded images; reject unsupported options."""
    allowed = {"model", "prompt", "max_tokens", "stream", "stream_options",
               "temperature", "top_p", "extra_body", "extra_headers"}
    if set(request) - allowed or request.get("stream") is not True:
        raise ValueError("prepared replay supports the explicit streaming completion contract")
    if request.get("model") != config.model_name:
        raise ValueError("replay model differs from configured model")
    ids = request.get("prompt")
    if (not isinstance(ids, list) or not ids
            or any(type(v) is not int or not 0 <= v < 2**32 for v in ids)):
        raise ValueError("prepared replay requires exact input token IDs")
    budget = request.get("max_tokens")
    if type(budget) is not int or not 0 <= budget < 2**32:
        raise ValueError("invalid output length")
    if request.get("stream_options", {"include_usage": True}) != {"include_usage": True}:
        raise ValueError("unsupported stream options")
    extra = request.get("extra_body", {})
    supported = {"ignore_eos", "top_k", "stream_interval", "forced_output_token_ids",
                 "return_token_ids", "return_cached_tokens_details", "cache_salt", "rid",
                 "image_data"}
    if not isinstance(extra, dict) or set(extra) - supported:
        # In particular strict_replay_deadline_audit has an OpenAI-only
        # process-local validation marker and cannot be enabled via /generate.
        raise ValueError("unsupported replay generation fields (including strict deadline audit)")
    if not isinstance(rid, str) or not rid or extra.get("rid", rid) != rid:
        raise ValueError("a consistent caller-provided request ID is required")
    if extra.get("cache_salt") != config.cache_salt:
        raise ValueError("request cache salt differs from fixed T configuration")
    if extra.get("return_token_ids") is not True:
        raise ValueError("exact replay requires return_token_ids=true")
    headers = request.get("extra_headers", {})
    if not isinstance(headers, dict):
        raise ValueError("extra_headers must be a mapping")
    lowered = {k.lower(): v for k, v in headers.items()}
    if set(lowered) - {"x-agent-context", "x-request-id", "x-smg-routing-key"}:
        raise ValueError("unsupported request headers")
    if lowered.get("x-request-id", rid) != rid:
        raise ValueError("request header identity mismatch")
    if "x-agent-context" in lowered:
        identity = RequestIdentity(**json.loads(lowered["x-agent-context"]))
        if identity.engine_rid != rid:
            raise ValueError("agent identity differs from request ID")
    sampling = {"max_new_tokens": budget}
    for key in ("temperature", "top_p"):
        if key in request:
            sampling[key] = request[key]
    for key in ("ignore_eos", "top_k", "stream_interval", "forced_output_token_ids"):
        if key in extra:
            sampling[key] = extra[key]
    inp = {"model": config.model_path, "input_ids": ids}
    if "image_data" in extra:
        images = extra["image_data"]
        if not isinstance(images, list) or not images or any(not isinstance(value, str) for value in images):
            raise ValueError("image_data requires uploaded image strings")
        inp["image_data"] = images
    return {
        "protocol_version": 1, "request_id": rid, "config_id": config.config_id,
        "input": inp,
        "generation": {"sampling_params": sampling, "stream": True,
                       "return_prompt_token_ids": True},
    }


def _openai_error(exc: BaseException, url: str) -> BaseException | None:
    """The OpenAI SDK error the baseline (OpenAI-compatible) path raises for this transport
    failure, or None for anything else. Trie counts an OpenAI error as one failed request and
    fails only its task; any other exception aborts the whole Trie process and its lanes."""
    import httpx
    import openai

    request = httpx.Request("POST", url)
    if isinstance(exc, PreparedHTTPError):
        response = httpx.Response(exc.status, request=request)
        cls = openai.InternalServerError if exc.status >= 500 else openai.APIStatusError
        return cls(str(exc), response=response, body=None)
    if isinstance(exc, (aiohttp.ClientError, TimeoutError)):
        return openai.APIConnectionError(message=f"{type(exc).__name__}: {exc}", request=request)
    return None


class CompletionStream:
    """Convert native SSE framing and token deltas, without changing measurements."""
    def __init__(self, raw, first: bytes, config: Config, rid: str, *, owner_callback=None):
        self.raw, self.config, self.rid = raw, config, rid
        self.buffer = first
        self.output_ids: list[int] = []
        self.text = ""
        self.prompt_ids = None
        self.closed = False
        self.saw_terminal = False
        self.saw_done = False
        self.owner_callback = owner_callback

    def __aiter__(self):
        return self

    async def aclose(self):
        if not self.closed:
            await self.raw.aclose()
            self.closed = True

    async def __anext__(self):
        if self.closed:
            if not self.saw_done:
                raise asyncio.CancelledError("native stream was closed before [DONE]")
            raise StopAsyncIteration
        if self.owner_callback is not None:
            self.owner_callback()
        try:
            while True:
                # SSE permits CRLF. Normalize only after joining byte chunks;
                # a CR/LF pair may have been split by TCP.
                normalized = self.buffer.replace(b"\r\n", b"\n")
                end = normalized.find(b"\n\n")
                if end < 0:
                    if len(self.buffer) > self.config.max_sse_bytes:
                        raise ValueError("native SSE event exceeds configured bound")
                    try:
                        self.buffer += await anext(self.raw)
                    except StopAsyncIteration:
                        raise ValueError("native stream ended without [DONE]") from None
                    continue
                event, self.buffer = normalized[:end], normalized[end + 2:]
                if len(event) > self.config.max_sse_bytes:
                    raise ValueError("native SSE event exceeds configured bound")
                lines = [line[5:].lstrip(b" ") for line in event.split(b"\n")
                         if line.startswith(b"data:")]
                if not lines:
                    continue
                data = b"\n".join(lines)
                if data == b"[DONE]":
                    if not self.saw_terminal:
                        raise ValueError("native stream lacks terminal engine result")
                    if self.buffer.strip():
                        raise ValueError("unexpected data after native [DONE]")
                    # Reach EOF so Backend records a completed response instead
                    # of interpreting a generator close as client cancellation.
                    async for trailing in self.raw:
                        if trailing.strip():
                            raise ValueError("unexpected data after native [DONE]")
                    self.saw_done = True
                    await self.aclose()
                    raise StopAsyncIteration
                return self.convert(json.loads(data))
        except BaseException as original:
            try:
                await self.aclose()
            except Exception as cleanup:
                _cleanup_failed(original, cleanup)
            mapped = _openai_error(original, self.config.tokenizer_url + "/forward_pd")
            if mapped is not None:
                raise mapped from original
            raise

    def convert(self, value: dict):
        from openai.types import Completion
        from openai.types.completion_choice import CompletionChoice
        from openai.types.completion_usage import CompletionUsage

        if not isinstance(value, dict) or "error" in value:
            raise ValueError("native generation returned an error")
        meta = value.get("meta_info")
        if not isinstance(meta, dict):
            raise ValueError("native generation omitted metadata")
        ids, text = value.get("output_ids"), value.get("text", "")
        if (not isinstance(ids, list) or any(type(v) is not int or v < 0 for v in ids)
                or not isinstance(text, str)):
            raise ValueError("native generation omitted output token IDs/text")
        if self.config.incremental_streaming_output:
            delta_ids, delta_text = ids, text
            self.output_ids.extend(ids)
            self.text += text
        else:
            if ids[:len(self.output_ids)] != self.output_ids or not text.startswith(self.text):
                raise ValueError("non-monotonic cumulative generation output")
            delta_ids, delta_text = ids[len(self.output_ids):], text[len(self.text):]
            self.output_ids, self.text = ids, text
        prompt_ids = value.get("prompt_token_ids")
        first_prompt = self.prompt_ids is None
        if prompt_ids is not None:
            if self.prompt_ids is not None and prompt_ids != self.prompt_ids:
                raise ValueError("native generation changed prompt token IDs")
            self.prompt_ids = prompt_ids
        counts = {key: meta.get(key) for key in ("prompt_tokens", "completion_tokens", "cached_tokens")}
        if any(type(v) is not int or v < 0 for v in counts.values()):
            raise ValueError("native generation omitted token usage")
        if counts["completion_tokens"] != len(self.output_ids):
            raise ValueError("native output IDs differ from completion usage")
        reason = meta.get("finish_reason")
        finish = None
        if reason is not None:
            if not isinstance(reason, dict) or reason.get("type") not in {"length", "stop"}:
                raise ValueError("native generation aborted or has unsupported finish reason")
            finish = reason["type"]
            self.saw_terminal = True
        choice = {"index": 0, "text": delta_text, "token_ids": delta_ids,
                  "finish_reason": finish, "logprobs": None}
        if prompt_ids is not None and first_prompt:
            choice["prompt_token_ids"] = prompt_ids
        # OpenAI's wire parser also uses unchecked construction for streaming
        # CompletionChoice: its annotation excludes the legitimate interim
        # finish_reason=None. Validate native values above instead of inventing
        # a terminal reason to satisfy that nonstreaming annotation.
        return Completion.model_construct(**{
            "id": self.rid, "object": "text_completion", "created": int(time.time()),
            "model": self.config.model_name,
            "choices": [CompletionChoice.model_construct(**choice)],
            "usage": CompletionUsage.model_validate({"prompt_tokens": counts["prompt_tokens"],
                      "completion_tokens": counts["completion_tokens"],
                      "total_tokens": counts["prompt_tokens"] + counts["completion_tokens"],
                      "prompt_tokens_details": {"cached_tokens": counts["cached_tokens"]}}),
            "sglext": {"cached_tokens_details": meta.get("cached_tokens_details")},
        })


class ReplayAdapter:
    def __init__(self, config: Config):
        self.config = config
        self.backend = Backend(timeout_s=config.timeout_s)
        self.scheduler = Scheduler(config.router_url, backend=self.backend)
        self.streams: set[CompletionStream] = set()
        self.request_streams: ContextVar[list | None] = ContextVar("prepared_replay_streams", default=None)
        self.request_owners: set[asyncio.Task] = set()
        self.closing = False

    def register_request_owner(self):
        if self.closing:
            raise asyncio.CancelledError("prepared replay is shutting down")
        task = asyncio.current_task()
        if task not in self.request_owners:
            self.request_owners.add(task)
            task.add_done_callback(self._owner_done)

    def _owner_done(self, task):
        self.request_owners.discard(task)
        # Client.run can leave multiple completed tasks unobserved when the
        # first task.result() raises. Retrieving does not change task.result().
        _observe_task(task)

    async def create(self, client, request: dict, *, connect_retry_limit: int,
                     rid: str | None, request_timing=None):
        self.register_request_owner()
        if connect_retry_limit != 0:
            raise ValueError("prepared generation is never retried")
        body = prepare_body(self.config, request, rid)
        client._check_benchmark_admission("sdk_call")
        if request_timing is not None:
            scheduled = request_timing.mark_sdk_call_start()
            asyncio.get_running_loop().call_soon(request_timing.observe_event_loop_turn, scheduled)
        prepare_url = endpoint_url(self.config.tokenizer_url) + "/prepare"
        try:
            async with self.backend._client().post(prepare_url, json=body,
                                                   allow_redirects=False) as response:
                if response.status != 200:
                    raise PreparedHTTPError("T preparation", response.status)
                prepared = await response.json()
        except (PreparedHTTPError, aiohttp.ClientError, TimeoutError) as exc:
            raise _openai_error(exc, prepare_url) from exc
        raw = self.scheduler.stream_prepared(self.config.tokenizer_url, prepared)
        try:
            # Do not report headers when only the lazy Python generator exists.
            # Retain the first actual byte chunk for the unchanged trie parser.
            first = await anext(raw)
            if request_timing is not None:
                request_timing.mark_headers_or_first_byte()
        except BaseException as original:
            try:
                await raw.aclose()
            except Exception as cleanup:
                _cleanup_failed(original, cleanup)
            # Forwarding may never have consumed T's store (e.g. placement
            # failed). This only releases a T CPU body, never engine KV state.
            try:
                async with asyncio.timeout(2):
                    async with self.backend._client().post(
                        endpoint_url(self.config.tokenizer_url) + "/release",
                        json={key: prepared[key] for key in
                              ("protocol_version", "request_id", "config_id", "ref")},
                        allow_redirects=False,
                    ) as released:
                        await released.read()
            except Exception:
                pass  # T's bounded TTL remains responsible for failed cleanup.
            mapped = _openai_error(original, endpoint_url(self.config.tokenizer_url) + "/forward_pd")
            if mapped is not None:
                raise mapped from original
            raise
        stream = CompletionStream(raw, first, self.config, rid,
                                  owner_callback=self.register_request_owner)
        self.streams.add(stream)
        group = self.request_streams.get()
        if group is not None:
            group.append(stream)
        return stream

    async def close(self, *, timeout_s: float = 10):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("positive finite cleanup timeout is required")
        self.closing = True
        deadline = asyncio.get_running_loop().time() + timeout_s
        original = None
        try:
            # A request may still be preparing or waiting for its first byte,
            # before there is a CompletionStream to close. Stop its owner first.
            owners = self.request_owners - {asyncio.current_task()}
            for task in owners:
                if not task.done():
                    task.cancel()
            if owners:
                _, pending = await asyncio.wait(owners, timeout=timeout_s)
                if pending:
                    raise ReplayCleanupTimeout(
                        f"{len(pending)} prepared replay task(s) remain after "
                        f"{timeout_s:g}s cancellation cleanup"
                    )
            closers = {asyncio.create_task(stream.aclose()) for stream in self.streams}
            if closers:
                done, pending = await asyncio.wait(
                    closers, timeout=max(0, deadline - asyncio.get_running_loop().time())
                )
                errors = [task.exception() for task in done if not task.cancelled()]
                for task in pending:
                    task.add_done_callback(_observe_task)
                    task.cancel()
                if pending:
                    raise ReplayCleanupTimeout(
                        f"{len(pending)} prepared replay stream(s) exceeded cleanup budget"
                    )
                if any(task.cancelled() for task in done):
                    raise RuntimeError("prepared replay stream cleanup was cancelled")
                failures = [error for error in errors if error is not None]
                if failures:
                    for cleanup in failures[1:]:
                        _cleanup_failed(failures[0], cleanup)
                    raise failures[0]
            self.streams.clear()
        except BaseException as exc:
            original = exc
            raise
        finally:
            pool_close = asyncio.create_task(self.backend.close())
            pool_close.add_done_callback(_observe_task)
            try:
                done, _ = await asyncio.wait(
                    {pool_close}, timeout=max(0, deadline - asyncio.get_running_loop().time())
                )
                if not done:
                    pool_close.cancel()
                    raise ReplayCleanupTimeout("prepared replay connection pool exceeded cleanup budget")
                if pool_close.cancelled():
                    raise RuntimeError("prepared replay connection pool cleanup was cancelled")
                pool_close.result()
            except ReplayCleanupTimeout as cleanup:
                raise cleanup from original
            except Exception as cleanup:
                if original is None:
                    raise
                _cleanup_failed(original, cleanup)


def install(client_class, config: Config):
    """Install opt-in hooks; close pools before the existing run loop exits.

    Returns the original methods for a caller that needs to undo installation.
    Calling twice on the same class is rejected instead of stacking wrappers.
    """
    if getattr(client_class, "_prepared_replay_installed", False):
        raise ValueError("prepared replay is already installed")
    originals = {name: getattr(client_class, name) for name in
                 ("_create_completion_with_connect_retry", "_execute_stream_request", "run")}
    # The pinned trie sync_run recognizes this sentinel and avoids an unbounded
    # asyncio.Runner.close() after cancellation has demonstrably timed out.
    timeout_error = originals["run"].__globals__.get(
        "BenchmarkCancellationTimeout", ReplayCleanupTimeout
    )

    def adapter(client):
        instance = getattr(client, "_prepared_replay_adapter", None)
        if instance is None:
            instance = ReplayAdapter(config)
            client._prepared_replay_adapter = instance
        return instance

    async def create(client, request, **kwargs):
        return await adapter(client).create(client, request, **kwargs)

    async def execute(client, *args, **kwargs):
        instance, streams = adapter(client), []
        instance.register_request_owner()
        token = instance.request_streams.set(streams)
        original = None
        try:
            return await originals["_execute_stream_request"](client, *args, **kwargs)
        except BaseException as exc:
            original = exc
            raise
        finally:
            instance.request_streams.reset(token)
            cleanup_failure = None
            for stream in streams:
                try:
                    await stream.aclose()
                except Exception as cleanup:
                    if original is not None:
                        _cleanup_failed(original, cleanup)
                    elif cleanup_failure is None:
                        cleanup_failure = cleanup
                    else:
                        _cleanup_failed(cleanup_failure, cleanup)
                else:
                    instance.streams.discard(stream)
            if original is None and cleanup_failure is not None:
                raise cleanup_failure

    async def run(client, *args, **kwargs):
        original = None
        try:
            return await originals["run"](client, *args, **kwargs)
        except BaseException as exc:
            original = exc
            raise
        finally:
            instance = getattr(client, "_prepared_replay_adapter", None)
            if instance is not None:
                boundary = getattr(client, "_benchmark_boundary", None)
                try:
                    await instance.close(timeout_s=getattr(boundary, "cleanup_timeout", 10))
                except ReplayCleanupTimeout as cleanup:
                    # Keep the initiating error as the cause, while retaining
                    # sync_run's bounded-shutdown behavior for this fatal case.
                    raise timeout_error(str(cleanup)) from (original or cleanup)
                except Exception as cleanup:
                    if original is None:
                        raise
                    _cleanup_failed(original, cleanup)
                finally:
                    client._prepared_replay_adapter = None

    client_class._create_completion_with_connect_retry = create
    client_class._execute_stream_request = execute
    client_class.run = run
    client_class._prepared_replay_installed = True
    return originals
