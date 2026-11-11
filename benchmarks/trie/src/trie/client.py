import asyncio
import base64
import copy
import json
import math
import mimetypes
import os
import random
import re
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import chain, cycle
from pathlib import Path

import numpy as np
from typing import TYPE_CHECKING, Any, Literal

import httpx
import structlog
from openai import (
    APIConnectionError,
    AsyncOpenAI,
    AsyncStream,
    DefaultAsyncHttpxClient,
    OpenAIError,
    RequestOptions,
)
from openai import (
    __version__ as OPENAI_VERSION,
)
from openai.types import Completion, CompletionChoice
from openai.types.completion_usage import CompletionUsage
from rich.live import Live

from trie.actual_output_context import ActualOutputContext
from trie.blocks import BlockTokens, pool_from_tokenizer
from trie.benchmark_boundary import (
    BenchmarkBoundary,
    BenchmarkCancellationTimeout,
    BenchmarkWindowClosed,
    cancel_workloads,
)
from trie.client_lifecycle import (
    ClientLifecycleRequest,
    ClientLifecycleWriter,
    RequestTiming,
)
from trie.dependencies import DependencyState, replay_events
from trie.initial_prefill import InitialPrefillReadiness, InitialSlot
from trie.phased_workload import (
    PhaseEnded, Schedule, Slot, check_phase, current_slot, phase_events, phase_wait, shared_clock,
)
from trie.prefix import (
    GlobalPrefixRegistry,
    block_aligned_prefix_length,
    common_prefix_length,
)
from trie.reporting import build_progress_lines, console, log_summary
from trie.results import (
    BenchmarkResult,
    ServerMetrics,
    StreamAccumulator,
)
from trie.types import (
    AppendMessage,
    Generate,
    ReplaceContext,
    ReplayTask,
    ReplayTrace,
    TraceWorkload,
    Wait,
    Workload,
    agent_traces,
    iter_generates,
    token_ids_sha256,
)

if TYPE_CHECKING:
    from trie.tokenizer_manager import TokenizerManager

logger = structlog.get_logger(__name__)

Prompt = str | list[int]

_STRICT_REPLAY_FAST_PATH_OPENAI_VERSION = "2.30.0"
_STRICT_REPLAY_FAST_PATH_FIELDS = frozenset(
    {
        "extra_body",
        "extra_headers",
        "max_tokens",
        "model",
        "prompt",
        "stream",
        "stream_options",
        "temperature",
        "top_p",
    }
)


class RequestAborted(ValueError):
    """The server ended the stream with finish_reason abort, e.g. /abort_request on a
    worker that is being removed; carries the output streamed before the abort."""

    def __init__(self, message: str, *, generated_token_ids: list[int], text: str) -> None:
        super().__init__(message)
        self.generated_token_ids = generated_token_ids
        self.text = text


def _is_predispatch_connect_failure(exc: APIConnectionError) -> bool:
    """Return true only when the HTTP request could not establish a connection.

    Read/write/protocol failures are intentionally excluded because the server may
    already have accepted the request. Retrying those failures can duplicate a PD
    dispatch and its KV-transfer side effects.
    """

    cause: BaseException | None = exc.__cause__
    while cause is not None:
        if isinstance(cause, (httpx.ConnectError, httpx.ConnectTimeout)):
            return True
        cause = cause.__cause__
    return False


def _root_cause_name(exc: BaseException) -> str:
    """Class name of the innermost cause, e.g. the httpx error under an APIConnectionError."""
    seen = set()
    while id(exc) not in seen and (exc.__cause__ or exc.__context__) is not None:
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return type(exc).__name__


_TRACE_ID_RE = re.compile(rb'"trace_id"\s*:\s*"([^"]+)"')
_SCHEMA_RE = re.compile(rb'"schema_version"\s*:\s*(\d+)')


@dataclass(frozen=True)
class LazyRow:
    """A schema-v2 trace or v3 task parsed only when it is admitted."""

    path: str
    offset: int
    trace_id: str
    schema_version: int

    def load(self) -> ReplayTrace | ReplayTask:
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            row = json.loads(f.readline())
        if row.get("schema_version") == 3:
            return ReplayTask.from_jsonl_row(row)
        if row.get("schema_version") == 2:
            return ReplayTrace.from_jsonl_row(row)
        raise ValueError(f"lazy workload row at byte {self.offset} is not schema v2/v3")


def _lazy_rows(path: str) -> list[LazyRow]:
    rows: list[LazyRow] = []
    offset = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                head = line[:4096]
                trace_id, version = _TRACE_ID_RE.search(head), _SCHEMA_RE.search(head)
                if trace_id is None or version is None:
                    parsed = json.loads(line)
                    ids = (parsed["trace_id"], int(parsed["schema_version"]))
                else:
                    ids = (trace_id.group(1).decode("utf-8"), int(version.group(1)))
                rows.append(LazyRow(path, offset, *ids))
            offset += len(line)
    return rows


def load_workloads(
    workload: list[TraceWorkload] | str,
    *,
    lazy: bool = False,
) -> list[TraceWorkload] | list[LazyRow]:
    if not isinstance(workload, str):
        return workload
    if lazy:
        return _lazy_rows(workload)
    with open(workload, "r", encoding="utf-8") as f:
        rows: list[TraceWorkload] = []
        for line_number, line in enumerate(f, start=1):
            row = json.loads(line)
            try:
                parsed = (
                    ReplayTask.from_jsonl_row(row)
                    if row.get("schema_version") == 3
                    else ReplayTrace.from_jsonl_row(row)
                    if row.get("schema_version") == 2
                    else Workload.from_jsonl_row(row)
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid workload row {line_number}: {exc}") from exc
            rows.append(parsed)
        return rows


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    prompt_token_ids: list[int] | None = None
    generated_token_ids: list[int] | None = None
    usage: CompletionUsage | None = None
    request_started_at_s: float | None = None
    first_token_at_s: float | None = None
    request_finished_at_s: float | None = None
    inter_token_latencies_ms: list[float] | None = None
    cached_tokens_details: dict[str, Any] | None = None


@dataclass
class _ReplayGenerationTokens:
    event: Generate
    prompt: str | None
    completed_context: str | None
    assistant_payload_ids: list[int] | None


def _extension_field(value: object, field: str) -> Any:
    if isinstance(value, dict):
        return value.get(field)
    direct = getattr(value, field, None)
    if direct is not None:
        return direct
    model_extra = getattr(value, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get(field)
    return None


def _extension_token_ids(value: object, field: str) -> list[int] | None:
    raw_ids = _extension_field(value, field)
    if raw_ids is None:
        return None
    if not isinstance(raw_ids, (list, tuple)):
        raise TypeError(f"response field {field} must be a list of token IDs")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in raw_ids
    ):
        raise ValueError(f"response field {field} contains a non-integer token ID")
    return list(raw_ids)


def _extension_mapping(value: object, field: str) -> dict[str, Any] | None:
    raw = _extension_field(value, field)
    if raw is None:
        return None
    if isinstance(raw, dict):
        return dict(raw)
    model_dump = getattr(raw, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    raise TypeError(f"response field {field} must be an object")


def _cached_prompt_tokens(usage: CompletionUsage | None) -> int:
    if usage is None or usage.prompt_tokens_details is None:
        return 0
    return usage.prompt_tokens_details.cached_tokens or 0


def _continued_generation(
    generation: GenerationResult,
    *,
    prompt: list[int],
    done_ids: list[int],
    done_text: str,
) -> GenerationResult:
    """One result for a migrated request whose last attempt carried the earlier output
    in its prompt: that output becomes the head of the generation again."""
    if (
        generation.prompt_token_ids is not None
        and generation.prompt_token_ids != prompt + done_ids
    ):
        raise ValueError("continued request echoed a different prompt")
    completion_tokens = len(done_ids) + generation.completion_tokens
    # Cached tokens of the continuation count against the original prompt only.
    cached_tokens = min(generation.cached_tokens, len(prompt))
    usage = generation.usage
    if usage is not None:
        details = usage.prompt_tokens_details
        usage = usage.model_copy(
            update={
                "prompt_tokens": len(prompt),
                "completion_tokens": completion_tokens,
                "total_tokens": len(prompt) + completion_tokens,
                "prompt_tokens_details": (
                    details.model_copy(update={"cached_tokens": cached_tokens})
                    if details is not None
                    else None
                ),
            }
        )
    return replace(
        generation,
        text=done_text + generation.text,
        prompt_tokens=len(prompt),
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        prompt_token_ids=(
            list(prompt) if generation.prompt_token_ids is not None else None
        ),
        generated_token_ids=(
            done_ids + generation.generated_token_ids
            if generation.generated_token_ids is not None
            else None
        ),
        usage=usage,
    )


def _validate_token_id_counts(
    *,
    prompt_token_ids: list[int] | None,
    generated_token_ids: list[int] | None,
    usage: CompletionUsage | None,
) -> None:
    if usage is None:
        return
    if prompt_token_ids is not None and len(prompt_token_ids) != usage.prompt_tokens:
        raise ValueError(
            "response prompt_token_ids length does not match usage.prompt_tokens"
        )
    if (
        generated_token_ids is not None
        and len(generated_token_ids) != usage.completion_tokens
    ):
        raise ValueError(
            "response token_ids length does not match usage.completion_tokens"
        )


class _InflightGauge:
    """Model requests in flight: current, peak and the time-weighted mean."""

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0
        self._area = 0.0
        self._start = self._since = time.perf_counter()

    def _advance(self) -> None:
        now = time.perf_counter()
        self._area += self.current * (now - self._since)
        self._since = now

    def enter(self) -> None:
        self._advance()
        self.current += 1
        self.peak = max(self.peak, self.current)

    def exit(self) -> None:
        self._advance()
        self.current -= 1

    def summary(self) -> dict[str, float | int]:
        self._advance()
        elapsed = self._since - self._start
        return {
            "mean": self._area / elapsed if elapsed > 0 else 0.0,
            "peak": self.peak,
            "current": self.current,
        }


def _stationary_cut(workload: ReplayTrace | ReplayTask, draw: float) -> float:
    """Cut time placed uniformly in the workload's recorded span."""
    generates = list(iter_generates(workload))
    if any(event.source_t_s is None for event in generates):
        raise ValueError(
            "warm_start=stationary requires source_t_s on every generate: "
            f"trace={workload.trace_id!r}"
        )
    start = min(event.source_t_s for event in generates)
    end = max(event.source_t_s + (event.source_api_time_s or 0.0) for event in generates)
    return start + draw * (end - start)


class Client:
    """Runs synthetic multi-turn workloads against an OpenAI-compatible endpoint."""

    _reasoning_effort: str | None = None
    _strict_replay_connect_retries: int = 0
    _strict_replay_connect_retry_delay_s: float = 1.0
    _strict_replay_openai_fast_path: bool = False
    _stream_interval: int = 1
    _replay_output_mode: str = "forced"
    _replay_context_mode: str = "recorded"
    _tokenizer_block_pool = None  # the vocabulary pool for block traces without their own
    _migration_limit: int = 0
    # Class-level default so duck-typed test doubles that skip __init__ still
    # resolve the attribute; __init__ overwrites it per instance.
    _lifecycle_writer: ClientLifecycleWriter | None = None
    _inflight: _InflightGauge = _InflightGauge()

    def __init__(
        self,
        endpoint: str,
        model: str,
        tokenizer_model: str | None = None,
        api_key: str | None = None,
        seed: int | None = None,
        max_retries: int = 2,
        timeout: float = 600.0,
        reasoning_effort: str | None = None,
        strict_replay_connect_retries: int = 0,
        strict_replay_connect_retry_delay_s: float = 1.0,
        strict_replay_openai_fast_path: bool = False,
        lifecycle_writer: ClientLifecycleWriter | None = None,
        stream_interval: int = 1,
        replay_output_mode: Literal["forced", "model"] = "forced",
        replay_context_mode: Literal["recorded", "actual_output"] = "recorded",
        migration_limit: int = 0,
    ) -> None:
        if replay_output_mode not in ("forced", "model"):
            raise ValueError("replay_output_mode must be forced or model")
        self._replay_output_mode = replay_output_mode
        if replay_context_mode not in ("recorded", "actual_output"):
            raise ValueError("replay_context_mode must be recorded or actual_output")
        if replay_context_mode == "actual_output" and replay_output_mode != "model":
            raise ValueError("actual_output context requires replay_output_mode=model")
        self._replay_context_mode = replay_context_mode
        if migration_limit < 0:
            raise ValueError("migration_limit must be greater than or equal to 0")
        self._migration_limit = migration_limit
        if max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0")
        if strict_replay_connect_retries < 0:
            raise ValueError(
                "strict_replay_connect_retries must be greater than or equal to 0"
            )
        if (
            not math.isfinite(strict_replay_connect_retry_delay_s)
            or strict_replay_connect_retry_delay_s < 0
        ):
            raise ValueError(
                "strict_replay_connect_retry_delay_s must be finite and nonnegative"
            )
        if stream_interval <= 0:
            raise ValueError("stream_interval must be positive")
        if (
            strict_replay_openai_fast_path
            and OPENAI_VERSION != _STRICT_REPLAY_FAST_PATH_OPENAI_VERSION
        ):
            raise RuntimeError(
                "strict_replay_openai_fast_path requires openai=="
                f"{_STRICT_REPLAY_FAST_PATH_OPENAI_VERSION}; found {OPENAI_VERSION}"
            )
        self._model = model
        self._max_retries = max_retries
        self._strict_replay_connect_retries = strict_replay_connect_retries
        self._strict_replay_connect_retry_delay_s = (
            strict_replay_connect_retry_delay_s
        )
        self._strict_replay_openai_fast_path = strict_replay_openai_fast_path
        self._client = AsyncOpenAI(
            base_url=endpoint,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
            http_client=DefaultAsyncHttpxClient(event_hooks={"request": [self._before_http_send]}),
        )
        self._tokenizer_model = tokenizer_model or model
        self._tokenizer_manager: TokenizerManager | None = None
        self._reasoning_effort = reasoning_effort
        self._seed = seed
        self._rng = random.Random(seed)
        self._result: BenchmarkResult | None = None
        self._benchmark_start: float | None = None
        self._strict_replay_output_ids: dict[int, list[int]] = {}
        self._lifecycle_writer = lifecycle_writer
        self._stream_interval = stream_interval

    def _check_benchmark_admission(self, stage: str):
        boundary = getattr(self, "_benchmark_boundary", None)
        return boundary.check(stage) if boundary is not None else None

    async def _before_http_send(self, request: httpx.Request) -> None:
        self._check_benchmark_admission("http_send")

    async def _create_completion(self, request_kwargs: dict[str, Any]) -> Any:
        if not self._strict_replay_openai_fast_path:
            return await self._client.completions.create(**request_kwargs)

        unexpected_fields = (
            set(request_kwargs) - _STRICT_REPLAY_FAST_PATH_FIELDS
        )
        if unexpected_fields:
            raise ValueError(
                "strict replay OpenAI fast path received unsupported request fields: "
                f"{sorted(unexpected_fields)}"
            )
        prompt = request_kwargs.get("prompt")
        if (
            not isinstance(prompt, list)
            or not prompt
            or any(type(token_id) is not int for token_id in prompt)
        ):
            raise ValueError(
                "strict replay OpenAI fast path requires a non-empty list[int] prompt"
            )
        if request_kwargs.get("stream") is not True:
            raise ValueError(
                "strict replay OpenAI fast path requires stream=true"
            )

        # Generated endpoint methods run async_maybe_transform() over the prompt
        # union. For list[int] prompts that becomes several Python passes over
        # every token before the SDK submits the request. The exported post()
        # method enters the same headers/retry/HTTP/SSE path without that
        # generated-model transform. Keep the existing insertion order so the
        # serialized request is byte-for-byte equivalent for this fixed schema.
        body = dict(request_kwargs)
        extra_headers = body.pop("extra_headers", None)
        extra_body = body.pop("extra_body", None)
        options: RequestOptions = {}
        if extra_headers is not None:
            if not isinstance(extra_headers, dict):
                raise TypeError("extra_headers must be a dict")
            options["headers"] = extra_headers
        if extra_body is not None:
            if not isinstance(extra_body, dict):
                raise TypeError("extra_body must be a dict")
            options["extra_json"] = extra_body
        return await self._client.post(
            "/completions",
            body=body,
            options=options,
            cast_to=Completion,
            stream=True,
            stream_cls=AsyncStream[Completion],
        )

    async def _create_completion_with_connect_retry(
        self,
        request_kwargs: dict[str, Any],
        *,
        connect_retry_limit: int,
        rid: str | None,
        request_timing: RequestTiming | None = None,
    ) -> Any:
        connect_retry = 0
        while True:
            try:
                self._check_benchmark_admission("sdk_call")
                if request_timing is not None:
                    scheduled_mono_ns = request_timing.mark_sdk_call_start()
                    asyncio.get_running_loop().call_soon(
                        request_timing.observe_event_loop_turn,
                        scheduled_mono_ns,
                    )
                response = await self._create_completion(request_kwargs)
                if request_timing is not None and bool(request_kwargs.get("stream")):
                    request_timing.mark_headers_or_first_byte()
                return response
            except APIConnectionError as exc:
                if (
                    connect_retry >= connect_retry_limit
                    or not _is_predispatch_connect_failure(exc)
                ):
                    raise
                connect_retry += 1
                logger.warning(
                    "strict replay connection establishment failed; retrying",
                    rid=rid,
                    retry=connect_retry,
                    retry_limit=connect_retry_limit,
                    delay_s=self._strict_replay_connect_retry_delay_s,
                )
                await asyncio.sleep(self._strict_replay_connect_retry_delay_s)

    def _block_pool(self, spec: dict[str, Any]):
        """The pool of a block trace that carries none: the regular vocabulary of
        ``tokenizer_model`` (a model directory), read once."""
        if spec.get("pool") is not None:
            return None
        if self._tokenizer_block_pool is None:
            directory = Path(self._tokenizer_model or "")
            if not (directory / "tokenizer.json").is_file():
                raise ValueError(
                    "this trace is not bound to a model: set tokenizer_model to the directory "
                    "of the model being served (it must hold tokenizer.json)"
                )
            self._tokenizer_block_pool = np.asarray(pool_from_tokenizer(directory), dtype=np.int64)
        return self._tokenizer_block_pool

    def _require_tokenizer_manager(self, purpose: str) -> "TokenizerManager":
        if self._tokenizer_manager is not None:
            return self._tokenizer_manager
        if self._tokenizer_model is None:
            raise ValueError(
                f"tokenizer_model is required for {purpose}; fully token-authoritative "
                "strict replay traces can omit it"
            )

        from trie.tokenizer_manager import TokenizerManager

        self._tokenizer_manager = TokenizerManager(
            self._tokenizer_model,
            seed=self._seed,
        )
        return self._tokenizer_manager

    async def _execute_stream_request(
        self,
        prompt: Prompt,
        max_tokens: int,
        *,
        trace_start: float,
        stream_acc: StreamAccumulator,
        cache_salt: str | None = None,
        routing_key: str | None = None,
        return_token_ids: bool = False,
        deterministic_sampling: bool = False,
        forced_output_token_ids: list[int] | None = None,
        image_data: list[str] | None = None,
        rid: str | None = None,
        strict_replay_deadline_audit: bool = False,
        disconnect_abort_delay_seconds: float | None = None,
        connect_retry_limit: int = 0,
        on_first_token: Callable[[int, int], None] | None = None,
        request_timing: RequestTiming | None = None,
    ) -> GenerationResult:
        if disconnect_abort_delay_seconds is not None and (
            not math.isfinite(disconnect_abort_delay_seconds)
            or disconnect_abort_delay_seconds < 0
        ):
            raise ValueError(
                "disconnect_abort_delay_seconds must be finite and nonnegative"
            )
        if strict_replay_deadline_audit:
            if (
                not isinstance(prompt, list)
                or not prompt
                or any(type(token) is not int for token in prompt)
                or return_token_ids is not True
                or deterministic_sampling is not True
                or not isinstance(forced_output_token_ids, list)
                or not forced_output_token_ids
                or any(type(token) is not int for token in forced_output_token_ids)
                or len(forced_output_token_ids) != max_tokens
                or not isinstance(rid, str)
                or re.fullmatch(r"rid_[0-9a-f]{32}", rid) is None
                or disconnect_abort_delay_seconds != 0.0
            ):
                raise ValueError(
                    "strict replay deadline audit requires a streamed, "
                    "deterministic token-authoritative request with exact output "
                    "IDs, a rid_<32 lowercase hex> identity, and zero abort delay"
                )
        elif disconnect_abort_delay_seconds is not None:
            raise ValueError(
                "disconnect_abort_delay_seconds requires strict replay deadline audit"
            )
        request_start = time.perf_counter()
        if request_timing is not None:
            request_timing.mark_request_start()
        extra_body: dict[str, object] = {"ignore_eos": True}
        if image_data is not None:
            extra_body["image_data"] = image_data
        stream_interval = getattr(self, "_stream_interval", 1)
        if stream_interval != 1:
            extra_body["stream_interval"] = stream_interval
        if return_token_ids:
            extra_body["return_token_ids"] = True
            extra_body["return_cached_tokens_details"] = True
        if deterministic_sampling:
            extra_body["top_k"] = 1
        if forced_output_token_ids is not None:
            extra_body["forced_output_token_ids"] = list(forced_output_token_ids)
        if cache_salt is not None:
            extra_body["cache_salt"] = cache_salt
        if rid is not None:
            extra_body["rid"] = rid
        if strict_replay_deadline_audit:
            extra_body["strict_replay_deadline_audit"] = True
        if disconnect_abort_delay_seconds is not None:
            extra_body["disconnect_abort_delay_seconds"] = (
                disconnect_abort_delay_seconds
            )
        request_kwargs: dict[str, object] = {}
        if deterministic_sampling:
            request_kwargs.update(temperature=0.0, top_p=1.0)
        if routing_key is not None:
            request_kwargs["extra_headers"] = {"X-SMG-Routing-Key": routing_key}
        stream = await self._create_completion_with_connect_retry(
            {
                "model": self._model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
                "extra_body": extra_body,
                **request_kwargs,
            },
            connect_retry_limit=connect_retry_limit,
            rid=rid,
            request_timing=request_timing,
        )
        text_parts: list[str] = []
        last_token_at_s: float | None = None
        inter_token_latencies_ms: list[float] = []
        ttft_s: float | None = None
        first_token_offset_s: float | None = None
        usage: CompletionUsage | None = None
        prompt_token_ids: list[int] | None = None
        generated_token_ids: list[int] = []
        cached_tokens_details: dict[str, Any] | None = None
        saw_generated_token_ids = False
        finish_reason: str | None = None
        async for chunk in stream:
            chunk_sglext = _extension_field(chunk, "sglext")
            if chunk_sglext is not None:
                details = _extension_mapping(
                    chunk_sglext, "cached_tokens_details"
                )
                if details is not None:
                    cached_tokens_details = details
            if chunk.choices:
                choice = chunk.choices[0]
                if isinstance(choice, CompletionChoice) and choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
                text = choice.text or ""
                if text:
                    text_parts.append(text)
                chunk_generated_token_ids = _extension_token_ids(choice, "token_ids")
                if chunk_generated_token_ids is None:
                    chunk_generated_token_ids = _extension_token_ids(chunk, "token_ids")
                if chunk_generated_token_ids is not None:
                    token_count = len(chunk_generated_token_ids)
                elif not text:
                    token_count = 0
                elif return_token_ids:
                    raise ValueError("stream response missing token_ids")
                else:
                    token_count = self._require_tokenizer_manager(
                        "counting text-only stream responses"
                    ).count_tokens(text)
                if token_count > 0:
                    chunk_at_s = time.perf_counter()
                    if ttft_s is None:
                        ttft_s = chunk_at_s - request_start
                        first_token_offset_s = chunk_at_s - trace_start
                        if on_first_token is not None:
                            if request_timing is not None:
                                first_token_clock = (
                                    request_timing.mark_first_parsed_token()
                                )
                            else:
                                first_token_clock = (
                                    time.time_ns(),
                                    time.monotonic_ns(),
                                )
                            on_first_token(*first_token_clock)
                        elif request_timing is not None:
                            request_timing.mark_first_parsed_token()
                    if last_token_at_s is not None:
                        gap_ms = (chunk_at_s - last_token_at_s) * 1000.0 / token_count
                        inter_token_latencies_ms.extend([gap_ms] * token_count)
                    last_token_at_s = chunk_at_s
                chunk_prompt_token_ids = _extension_token_ids(
                    choice, "prompt_token_ids"
                )
                if chunk_prompt_token_ids is None:
                    chunk_prompt_token_ids = _extension_token_ids(
                        chunk, "prompt_token_ids"
                    )
                if chunk_prompt_token_ids is not None:
                    if (
                        prompt_token_ids is not None
                        and chunk_prompt_token_ids != prompt_token_ids
                    ):
                        raise ValueError(
                            "stream returned inconsistent prompt_token_ids"
                        )
                    prompt_token_ids = chunk_prompt_token_ids

                if chunk_generated_token_ids is not None:
                    generated_token_ids.extend(chunk_generated_token_ids)
                    saw_generated_token_ids = True
            if chunk.usage is not None:
                usage = chunk.usage
        latency_s = time.perf_counter() - request_start
        if finish_reason == "abort":
            raise RequestAborted(
                f"server aborted the request: rid={rid!r}",
                generated_token_ids=generated_token_ids,
                text="".join(text_parts),
            )
        if ttft_s is None or first_token_offset_s is None:
            raise ValueError("stream produced no generated tokens")
        exact_generated_token_ids = (
            generated_token_ids if saw_generated_token_ids else None
        )
        _validate_token_id_counts(
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=exact_generated_token_ids,
            usage=usage,
        )
        completion_tokens = (
            usage.completion_tokens
            if usage is not None
            else (
                len(exact_generated_token_ids)
                if exact_generated_token_ids is not None
                else max_tokens
            )
        )
        stream_acc.record_turn_stream_metrics(
            ttft_s,
            latency_s,
            completion_tokens,
            first_token_offset_s=first_token_offset_s,
            inter_token_latencies_ms=inter_token_latencies_ms,
        )
        return GenerationResult(
            text="".join(text_parts),
            prompt_tokens=(
                usage.prompt_tokens
                if usage is not None
                else (
                    len(prompt_token_ids)
                    if prompt_token_ids is not None
                    else (
                        len(prompt)
                        if isinstance(prompt, list)
                        else self._require_tokenizer_manager(
                            "counting text prompts without response usage"
                        ).count_tokens(prompt)
                    )
                )
            ),
            completion_tokens=completion_tokens,
            cached_tokens=_cached_prompt_tokens(usage),
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=exact_generated_token_ids,
            usage=usage,
            request_started_at_s=request_start,
            first_token_at_s=request_start + ttft_s,
            request_finished_at_s=request_start + latency_s,
            inter_token_latencies_ms=inter_token_latencies_ms,
            cached_tokens_details=cached_tokens_details,
        )

    async def _execute_request(
        self,
        prompt: Prompt,
        max_tokens: int,
        *,
        stream: bool,
        trace_start: float,
        stream_acc: StreamAccumulator,
        cache_salt: str | None = None,
        routing_key: str | None = None,
        return_token_ids: bool = False,
        deterministic_sampling: bool = False,
        forced_output_token_ids: list[int] | None = None,
        image_data: list[str] | None = None,
        rid: str | None = None,
        strict_replay_deadline_audit: bool = False,
        disconnect_abort_delay_seconds: float | None = None,
        connect_retry_limit: int = 0,
        on_first_token: Callable[[int, int], None] | None = None,
        request_timing: RequestTiming | None = None,
    ) -> GenerationResult:
        if strict_replay_deadline_audit and stream is not True:
            raise ValueError("strict replay deadline audit requires a streamed request")
        if not strict_replay_deadline_audit and (
            disconnect_abort_delay_seconds is not None
        ):
            if (
                not math.isfinite(disconnect_abort_delay_seconds)
                or disconnect_abort_delay_seconds < 0
            ):
                raise ValueError(
                    "disconnect_abort_delay_seconds must be finite and nonnegative"
                )
            raise ValueError(
                "disconnect_abort_delay_seconds requires strict replay deadline audit"
            )
        if stream:
            return await self._execute_stream_request(
                prompt,
                max_tokens,
                trace_start=trace_start,
                stream_acc=stream_acc,
                cache_salt=cache_salt,
                routing_key=routing_key,
                return_token_ids=return_token_ids,
                deterministic_sampling=deterministic_sampling,
                forced_output_token_ids=forced_output_token_ids,
                image_data=image_data,
                rid=rid,
                strict_replay_deadline_audit=strict_replay_deadline_audit,
                disconnect_abort_delay_seconds=disconnect_abort_delay_seconds,
                connect_retry_limit=connect_retry_limit,
                on_first_token=on_first_token,
                request_timing=request_timing,
            )

        extra_body: dict[str, object] = {"ignore_eos": True}
        if image_data is not None:
            extra_body["image_data"] = image_data
        if return_token_ids:
            extra_body["return_token_ids"] = True
            extra_body["return_cached_tokens_details"] = True
        if deterministic_sampling:
            extra_body["top_k"] = 1
        if forced_output_token_ids is not None:
            extra_body["forced_output_token_ids"] = list(forced_output_token_ids)
        if cache_salt is not None:
            extra_body["cache_salt"] = cache_salt
        if rid is not None:
            extra_body["rid"] = rid
        if strict_replay_deadline_audit:
            extra_body["strict_replay_deadline_audit"] = True
        if disconnect_abort_delay_seconds is not None:
            if (
                not math.isfinite(disconnect_abort_delay_seconds)
                or disconnect_abort_delay_seconds < 0
            ):
                raise ValueError(
                    "disconnect_abort_delay_seconds must be finite and nonnegative"
                )
            extra_body["disconnect_abort_delay_seconds"] = (
                disconnect_abort_delay_seconds
            )
        request_kwargs: dict[str, object] = {}
        if deterministic_sampling:
            request_kwargs.update(temperature=0.0, top_p=1.0)
        if routing_key is not None:
            request_kwargs["extra_headers"] = {"X-SMG-Routing-Key": routing_key}
        request_start = time.perf_counter()
        response = await self._create_completion_with_connect_retry(
            {
                "model": self._model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "extra_body": extra_body,
                **request_kwargs,
            },
            connect_retry_limit=connect_retry_limit,
            rid=rid,
            request_timing=request_timing,
        )
        request_finished = time.perf_counter()
        if response.usage is None:
            raise ValueError("response.usage must not be None")
        choice = response.choices[0]
        prompt_token_ids = _extension_token_ids(choice, "prompt_token_ids")
        if prompt_token_ids is None:
            prompt_token_ids = _extension_token_ids(response, "prompt_token_ids")
        generated_token_ids = _extension_token_ids(choice, "token_ids")
        if generated_token_ids is None:
            generated_token_ids = _extension_token_ids(response, "token_ids")
        _validate_token_id_counts(
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            usage=response.usage,
        )
        return GenerationResult(
            text=choice.text or "",
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cached_tokens=_cached_prompt_tokens(response.usage),
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            usage=response.usage,
            request_started_at_s=request_start,
            request_finished_at_s=request_finished,
            cached_tokens_details=(
                _extension_mapping(
                    _extension_field(response, "sglext"),
                    "cached_tokens_details",
                )
                if _extension_field(response, "sglext") is not None
                else None
            ),
        )

    async def _execute_with_migration(
        self,
        prompt: Prompt,
        max_tokens: int,
        *,
        rid: str | None,
        on_first_token: Callable[[int, int], None] | None,
        request_timing: RequestTiming | None,
        result: BenchmarkResult | None,
        lifecycle_request: ClientLifecycleRequest | None,
        forced_output_token_ids: list[int] | None,
        expected_prompt_token_ids: list[int] | None = None,
        **request_kwargs: Any,
    ) -> GenerationResult:
        """Re-send an interrupted request up to migration_limit times, as Dynamo's
        frontend request migration does: output streamed before an abort is appended
        to the prompt and generation continues, so the client still sees one request."""
        migrations = 0
        done_ids: list[int] = []
        done_text = ""
        while True:
            self._check_benchmark_admission("request_call")
            # One first-token event per request, from the attempt that streamed it first.
            first_token_seen = (
                request_timing is not None
                and request_timing.first_parsed_token is not None
            )
            try:
                generation = await self._execute_request(
                    prompt + done_ids if done_ids else prompt,
                    max_tokens - len(done_ids),
                    rid=rid,
                    on_first_token=None if first_token_seen else on_first_token,
                    request_timing=request_timing,
                    forced_output_token_ids=(
                        forced_output_token_ids[len(done_ids):]
                        if forced_output_token_ids is not None
                        else None
                    ),
                    **request_kwargs,
                )
            except (OpenAIError, RequestAborted, httpx.TransportError) as exc:
                if migrations >= self._migration_limit:
                    raise
                migrations += 1
                # Only a token prompt can carry streamed output; anything else, and an
                # abort after the last token, is re-sent from the start.
                if isinstance(exc, RequestAborted) and isinstance(prompt, list):
                    done_ids = done_ids + exc.generated_token_ids
                    done_text += exc.text
                    if len(done_ids) >= max_tokens:
                        done_ids, done_text = [], ""
                # A fresh rid: the server aborts by rid prefix, and the interrupted
                # attempt may still be alive on a decode worker.
                rid = None if rid is None else f"rid_{uuid.uuid4().hex}"
                if result is not None:
                    result.record_migration()
                if self._lifecycle_writer is not None and lifecycle_request is not None:
                    self._lifecycle_writer.emit(
                        "CLIENT_MIGRATE",
                        lifecycle_request,
                        attrs={
                            "attempt": migrations,
                            "error_class": type(exc).__name__,
                            "rid": rid,
                            "continued_tokens": len(done_ids),
                        },
                    )
                logger.warning(
                    "request interrupted; re-sending",
                    error=str(exc),
                    attempt=migrations,
                    continued_tokens=len(done_ids),
                )
                continue
            if not done_ids:
                return generation
            assert isinstance(prompt, list)
            return _continued_generation(
                generation,
                prompt=expected_prompt_token_ids if expected_prompt_token_ids is not None else prompt,
                done_ids=done_ids, done_text=done_text,
            )

    async def _run_legacy_workload(
        self,
        workload: Workload,
        *,
        benchmark_start: float,
        result: BenchmarkResult,
        stream: bool,
        refresh: Callable[[], None],
    ) -> None:
        tokenizer = self._require_tokenizer_manager("synthetic workloads")
        request_start = time.perf_counter()
        turns_completed = 0
        metrics = ServerMetrics()
        stream_acc = StreamAccumulator()
        try:
            # Generate one fewer token than requested to compensate for BOS token
            # which is added by vLLM, SGLang, and TensorRT-LLM.
            prompt = tokenizer.get_prompt(max(workload.input_prompt_length - 1, 0))
            for turn_index, turn in enumerate(workload.turns):
                response = await self._execute_request(
                    prompt,
                    turn.assistant_response_length,
                    stream=stream,
                    trace_start=request_start,
                    stream_acc=stream_acc,
                )
                if response.usage is not None:
                    metrics.record_usage(response.usage)
                result.record_turn(
                    workload,
                    turn_index,
                    timestamp=time.perf_counter() - benchmark_start,
                )
                turns_completed = turn_index + 1
                refresh()
                prompt += response.text
                await asyncio.sleep(turn.tool_call_latency)
                prompt += tokenizer.get_prompt(turn.tool_call_output_length)

            final_response = await self._execute_request(
                prompt,
                workload.final_assistant_response_length,
                stream=stream,
                trace_start=request_start,
                stream_acc=stream_acc,
            )
            if final_response.usage is not None:
                metrics.record_usage(final_response.usage)
            if not stream or metrics.prompt_tokens > 0:
                result.server_metrics.append(metrics)
            else:
                logger.warning("stream response usage missing; omitting server metrics")
            event_timestamp = time.perf_counter()
            result.record_success(
                workload,
                latency=event_timestamp - request_start,
                timestamp=event_timestamp - benchmark_start,
            )
            if stream:
                stream_acc.commit(result)
        except asyncio.CancelledError:
            return
        except (OpenAIError, ValueError) as e:
            result.record_failure()
            logger.warning("request failed", error=str(e), turn=turns_completed)
        refresh()

    @staticmethod
    def _event_delay(
        event: Wait,
        *,
        delay_scale: float,
        max_user_delay: float | None,
        max_tool_delay: float | None,
    ) -> float:
        delay = event.seconds
        if event.actor == "user" and max_user_delay is not None:
            delay = min(delay, max_user_delay)
        elif event.actor == "tool" and max_tool_delay is not None:
            delay = min(delay, max_tool_delay)
        return delay * delay_scale

    def _generated_assistant_content(
        self,
        event: Generate,
        generation: GenerationResult,
    ) -> str:
        recorded = event.recorded_assistant
        if recorded is None or not recorded.get("tool_calls"):
            return generation.text

        text_parts = [
            recorded.get("reasoning_content") or "",
            recorded.get("content") or "",
        ]
        tokenizer = self._require_tokenizer_manager(
            "generated-context tool-call reconstruction"
        )
        text_budget = len(
            tokenizer.encode("\n".join(part for part in text_parts if part))
        )
        if text_budget == 0:
            return ""
        assert generation.generated_token_ids is not None
        return tokenizer.decode(generation.generated_token_ids[:text_budget])

    def _prepare_strict_replay_output_ids(
        self,
        traces: list[ReplayTrace],
        *,
        merge: bool = False,
    ) -> None:
        prepared: dict[int, list[int]] = {}
        if self._replay_output_mode == "model":
            # Exact input IDs and output lengths suffice for real generation.
            # Do not invent output tokens for traces that only recorded usage.
            for trace in traces:
                for event in trace.events:
                    if isinstance(event, Generate) and (
                        event.recorded_prompt_token_ids is None
                        and event.recorded_prompt_block_runs is None
                        and event.recorded_assistant is None
                    ):
                        raise ValueError("model replay needs exact prompt IDs or recorded context")
            if not merge:
                self._strict_replay_output_ids = {}
            return
        explicit_events = 0
        canonical_events = 0
        ambiguous_events = 0

        for trace in traces:
            messages = copy.deepcopy(trace.initial_messages)
            generations: list[_ReplayGenerationTokens] = []
            request_index = 0
            missing_context_request: int | None = None
            for event in trace.events:
                if isinstance(event, AppendMessage):
                    messages.append(copy.deepcopy(event.message))
                    continue
                if isinstance(event, ReplaceContext):
                    messages = copy.deepcopy(event.messages)
                    missing_context_request = None
                    continue
                if isinstance(event, Wait):
                    continue

                assert isinstance(event, Generate)
                fully_token_authoritative = (
                    event.recorded_prompt_token_ids is not None
                    or event.recorded_prompt_block_runs is not None
                ) and (
                    event.recorded_output_token_ids is not None
                    or event.recorded_output_block_runs is not None
                )
                compact_token_contract = (
                    event.recorded_prompt_token_count is not None
                    and event.recorded_output_token_ids is not None
                )
                if event.recorded_assistant is None and not fully_token_authoritative:
                    raise ValueError(
                        "strict replay requires recorded_assistant unless both prompt "
                        f"and output token IDs are recorded: trace={trace.trace_id!r} "
                        f"request={request_index}"
                    )
                if fully_token_authoritative or compact_token_contract:
                    generations.append(_ReplayGenerationTokens(event, None, None, None))
                    explicit_events += 1
                    if event.recorded_assistant is None:
                        if missing_context_request is None:
                            missing_context_request = request_index
                    else:
                        messages.append(copy.deepcopy(event.recorded_assistant))
                    request_index += 1
                    continue
                if missing_context_request is not None:
                    raise ValueError(
                        "strict replay requires recorded_assistant for request "
                        f"{missing_context_request} because request {request_index} "
                        f"needs text reconstruction: trace={trace.trace_id!r}"
                    )
                assert event.recorded_assistant is not None
                tokenizer = self._require_tokenizer_manager(
                    "strict replay traces without authoritative prompt and output "
                    "token IDs"
                )
                prompt = tokenizer.render_chat(
                    messages,
                    tools=trace.tools,
                    add_generation_prompt=True,
                    reasoning_effort=self._reasoning_effort,
                )
                if event.recorded_output_token_ids is not None:
                    generations.append(
                        _ReplayGenerationTokens(event, prompt, None, None)
                    )
                    explicit_events += 1
                else:
                    completed_context = tokenizer.render_chat(
                        messages + [copy.deepcopy(event.recorded_assistant)],
                        tools=trace.tools,
                        add_generation_prompt=False,
                        reasoning_effort=self._reasoning_effort,
                    )
                    if not completed_context.startswith(prompt):
                        raise ValueError(
                            "recorded assistant is not a tokenizable continuation "
                            "of its prompt: "
                            f"trace={trace.trace_id!r} request={request_index}"
                        )
                    continuation = completed_context[len(prompt) :]
                    assistant_payload_ids = tokenizer.encode(continuation)

                    # Validate the generation boundary without re-tokenizing the
                    # potentially very large context.
                    prompt_tail = prompt[-1024:]
                    boundary_ids = tokenizer.encode(prompt_tail)
                    combined_ids = tokenizer.encode(prompt_tail + continuation)
                    if (
                        combined_ids[: len(boundary_ids)] != boundary_ids
                        or combined_ids[len(boundary_ids) :] != assistant_payload_ids
                    ):
                        raise ValueError(
                            "recorded assistant tokenization crosses the prompt "
                            "boundary: "
                            f"trace={trace.trace_id!r} request={request_index}"
                        )
                    generations.append(
                        _ReplayGenerationTokens(
                            event,
                            prompt,
                            completed_context,
                            assistant_payload_ids,
                        )
                    )
                    canonical_events += 1

                messages.append(copy.deepcopy(event.recorded_assistant))
                request_index += 1

            for index, generation in enumerate(generations):
                event = generation.event
                if event.recorded_output_token_ids is not None:
                    prepared[id(event)] = list(event.recorded_output_token_ids)
                    continue
                if event.recorded_output_block_runs is not None:
                    # expanded at send time: the content depends on the replay cycle
                    prepared[id(event)] = None
                    continue

                assert generation.assistant_payload_ids is not None
                payload_ids = generation.assistant_payload_ids
                if len(payload_ids) >= event.max_tokens:
                    output_ids = payload_ids[: event.max_tokens]
                    if len(payload_ids) != event.max_tokens:
                        ambiguous_events += 1
                else:
                    tokenizer = self._require_tokenizer_manager(
                        "reconstructing output tokens from recorded text"
                    )
                    terminal_id = tokenizer.eos_token_id()
                    recorded = event.recorded_assistant
                    assert recorded is not None
                    if recorded.get("tool_calls") and index + 1 < len(generations):
                        completed_context = generation.completed_context
                        assert completed_context is not None
                        next_prompt = generations[index + 1].prompt
                        if next_prompt is not None and next_prompt.startswith(
                            completed_context
                        ):
                            separator = next_prompt[len(completed_context) :][:1024]
                            separator_ids = tokenizer.encode(separator)
                            if separator_ids:
                                terminal_id = separator_ids[0]

                    missing = event.max_tokens - len(payload_ids)
                    filler_ids = tokenizer.encode("\n")
                    filler_id = filler_ids[0] if filler_ids else terminal_id
                    output_ids = (
                        payload_ids + [filler_id] * max(missing - 1, 0) + [terminal_id]
                    )
                    if missing != 1:
                        ambiguous_events += 1

                if len(output_ids) != event.max_tokens:
                    raise AssertionError("strict replay output length mismatch")
                prepared[id(event)] = output_ids

        # A lazily loaded workload adds its streams to those still running.
        if merge:
            self._strict_replay_output_ids.update(prepared)
        else:
            self._strict_replay_output_ids = prepared
        log_preparation = logger.warning if ambiguous_events else logger.info
        log_preparation(
            "prepared strict replay token streams",
            traces=len(traces),
            events=len(prepared),
            explicit_token_events=explicit_events,
            canonical_text_events=canonical_events,
            ambiguous_text_events=ambiguous_events,
        )

    async def _run_replay_trace(
        self,
        trace: ReplayTrace,
        *,
        benchmark_start: float,
        request_semaphore: asyncio.Semaphore | None = None,
        result: BenchmarkResult,
        stream: bool,
        refresh: Callable[[], None],
        delay_scale: float,
        max_user_delay: float | None,
        max_tool_delay: float | None,
        context_source: Literal["generated", "recorded"],
        prefix_block_size: int,
        prefix_registry: GlobalPrefixRegistry | None = None,
        cache_salt_mode: Literal["global", "session"],
        run_cache_salt: str | None,
        strict_replay: bool = False,
        replay_cycle_id: int = 0,
        dependency_state: DependencyState | None = None,
        identity_trace_id: str | None = None,
        task_identity: ClientLifecycleRequest | None = None,
        cache_salt_override: str | None = None,
        cut_s: float | None = None,
        tag_requests: bool = False,
        trajectory_prefix_ids: tuple[int, ...] = (),
        initial_prefill_slot: InitialSlot | None = None,
    ) -> None:
        metrics = ServerMetrics()
        stream_acc = StreamAccumulator()
        messages = copy.deepcopy(trace.initial_messages)
        previous_actual_ids: list[int] | None = None
        actual_context = (
            ActualOutputContext()
            if self._replay_context_mode == "actual_output"
            else None
        )
        request_index = 0
        active_lifecycle_request: ClientLifecycleRequest | None = None
        active_request_timing: RequestTiming | None = None
        active_lifecycle_complete = False
        last_lifecycle_request: ClientLifecycleRequest | None = None
        prefix_attrs = (
            {
                "trajectory_prefix_tokens": len(trajectory_prefix_ids),
                "trajectory_prefix_sha256": token_ids_sha256(list(trajectory_prefix_ids)),
            }
            if trajectory_prefix_ids
            else {}
        )
        if actual_context is not None:
            prefix_attrs["replay_context_mode"] = "actual_output"
        if cache_salt_mode == "session":
            if run_cache_salt is None:
                raise ValueError("session cache salt requires a run UUID")
            cache_salt = cache_salt_override or (
                f"{run_cache_salt}:{trace.trace_id}:{uuid.uuid4().hex}"
            )
        else:
            cache_salt = None
        routing_key = trace.metadata.get("routing_key")
        if routing_key is None:
            # Compatibility for traces produced before routing became a
            # model- and benchmark-neutral schema field.
            routing_key = trace.metadata.get("session_key")
        if routing_key is not None and (
            not isinstance(routing_key, str) or not routing_key
        ):
            raise ValueError("trace metadata routing_key must be a non-empty string")

        slot = current_slot.get()
        if slot is not None and routing_key is not None:
            routing_key = f"{slot.label}:{routing_key}"

        if strict_replay and self._replay_output_mode == "forced" and any(
            id(event) not in self._strict_replay_output_ids
            for event in trace.events
            if isinstance(event, Generate)
        ):
            self._prepare_strict_replay_output_ids([trace], merge=True)

        # A warm-start cut skips generates recorded before cut_s; the last of
        # them is re-sent as a one-token warmup before the first generate after
        # the cut, which then waits at most the recorded time left after it.
        block_tokens = (
            BlockTokens(trace.block_spec, replay_cycle_id, self._block_pool(trace.block_spec))
            if strict_replay and trace.block_spec is not None
            else None
        )
        cutting = cut_s is not None
        pending_delay = 0.0
        warm_event: tuple[Generate, int] | None = None
        # The context at the warm event, for rows that carry messages rather than prompt IDs.
        warm_messages: list[dict[str, Any]] = []
        generate_position = 0
        active_prompt_tokens: int | None = None
        try:
            if trace.start_offset_s:
                start_delay = (
                    benchmark_start
                    + trace.start_offset_s * delay_scale
                    - time.perf_counter()
                )
                if start_delay > 0:
                    await phase_wait(asyncio.sleep(start_delay))
            trace_start = time.perf_counter()

            async for event in phase_events(replay_events(trace, dependency_state)):
                if isinstance(event, AppendMessage):
                    messages.append(copy.deepcopy(event.message))
                    continue
                if isinstance(event, ReplaceContext):
                    messages = copy.deepcopy(event.messages)
                    continue
                if isinstance(event, Wait):
                    delay = self._event_delay(
                        event,
                        delay_scale=delay_scale,
                        max_user_delay=max_user_delay,
                        max_tool_delay=max_tool_delay,
                    )
                    if cutting:
                        pending_delay = delay
                    elif delay:
                        await phase_wait(asyncio.sleep(delay))
                    continue

                assert isinstance(event, Generate)
                position = generate_position
                generate_position += 1
                if cutting:
                    assert cut_s is not None and event.source_t_s is not None
                    if event.source_t_s < cut_s:
                        warm_event, pending_delay = (event, position), 0.0
                        warm_messages = list(messages)
                        if event.recorded_assistant is not None:
                            messages.append(copy.deepcopy(event.recorded_assistant))
                        continue
                    cutting = False
                    if warm_event is not None:
                        previous_actual_ids = await self._send_warmup(
                            trace,
                            *warm_event,
                            block_tokens=block_tokens,
                            identity_trace_id=identity_trace_id or trace.trace_id,
                            task_identity=task_identity,
                            replay_cycle_id=replay_cycle_id,
                            request_semaphore=request_semaphore,
                            prefix_registry=prefix_registry,
                            stream=stream,
                            trace_start=trace_start,
                            cache_salt=cache_salt,
                            routing_key=routing_key,
                            strict_replay=strict_replay,
                            trajectory_prefix_ids=trajectory_prefix_ids,
                            messages=warm_messages,
                        )
                        if initial_prefill_slot is not None:
                            initial_prefill_slot.completed(trace.trace_id)
                    residual = min(
                        pending_delay, (event.source_t_s - cut_s) * delay_scale
                    )
                    if residual > 0:
                        await phase_wait(asyncio.sleep(residual))
                check_phase()
                self._check_benchmark_admission("generate_prepare")
                request_timing = (
                    RequestTiming.start()
                    if self._lifecycle_writer is not None and stream
                    else None
                )
                active_request_timing = request_timing
                recorded_prompt = (
                    self._recorded_prompt(event, block_tokens)
                    if strict_replay or event.recorded_image_data is not None else None
                )
                if recorded_prompt is not None:
                    prompt: Prompt = list(recorded_prompt)
                else:
                    tokenizer = self._require_tokenizer_manager(
                        "rendering replay prompts without authoritative token IDs"
                    )
                    rendered_prompt = tokenizer.render_chat(
                        messages,
                        tools=trace.tools,
                        add_generation_prompt=True,
                        reasoning_effort=self._reasoning_effort,
                    )
                    prompt = (
                        tokenizer.encode(rendered_prompt)
                        if trajectory_prefix_ids or (
                            strict_replay and event.recorded_prompt_token_count is not None
                        )
                        else rendered_prompt
                    )
                if (
                    strict_replay
                    and event.recorded_prompt_token_count is not None
                    and (
                        not isinstance(prompt, list)
                        or len(prompt) != event.recorded_prompt_token_count
                        or token_ids_sha256(prompt)
                        != event.recorded_prompt_token_ids_sha256
                    )
                ):
                    raise ValueError(
                        "rendered prompt token IDs differ from strict replay contract: "
                        f"trace={trace.trace_id!r} request={request_index}"
                    )
                context_source_ids = None
                if actual_context is not None:
                    if not isinstance(prompt, list):
                        prompt = self._require_tokenizer_manager(
                            "actual-output replay context"
                        ).encode(prompt)
                    context_source_ids = list(prompt)
                    prompt, _ = actual_context.prepare(context_source_ids)
                # Validate the source first; prefix only the wire prompt. The same
                # prefix survives every turn, compaction and child-agent request.
                if trajectory_prefix_ids:
                    prompt[:0] = trajectory_prefix_ids
                image_data, expected_prompt, cache_prompt = await self._prepare_replay_images(
                    trace, event, prompt
                )
                forced_output_token_ids = (
                    self._strict_replay_output_ids[id(event)]
                    if strict_replay and self._replay_output_mode == "forced"
                    else None
                )
                if (
                    strict_replay
                    and self._replay_output_mode == "forced"
                    and event.recorded_output_block_runs is not None
                ):
                    assert block_tokens is not None
                    forced_output_token_ids = block_tokens.tokens(
                        event.recorded_output_block_runs, event.max_tokens
                    )
                request_acquired = False
                if request_semaphore is not None:
                    await phase_wait(request_semaphore.acquire())
                    request_acquired = True
                lifecycle_writer = self._lifecycle_writer
                # The generate's position equals request_index unless a cut skipped
                # generates, and keeps warmup and cut identities distinct.
                lifecycle_request = (
                    lifecycle_writer.request_identity(
                        trace_id=identity_trace_id or trace.trace_id,
                        cycle_id=replay_cycle_id,
                        event_index=position,
                        routing_key=routing_key,
                    )
                    if lifecycle_writer is not None
                    else None
                )
                active_prompt_tokens = len(prompt) if isinstance(prompt, list) else None
                try:
                    check_phase()
                    admit_clock = self._check_benchmark_admission("lifecycle_admission")
                except (asyncio.CancelledError, PhaseEnded):
                    if request_acquired:
                        request_semaphore.release()
                    raise
                active_lifecycle_request = lifecycle_request
                active_lifecycle_complete = False
                if lifecycle_writer is not None and lifecycle_request is not None:
                    request_attrs = self._request_tags(
                        trace, task_identity, tag_requests, warmup=False
                    )
                    if prefix_attrs:
                        request_attrs = {**(request_attrs or {}), **prefix_attrs}
                    lifecycle_writer.emit(
                        "CLIENT_ADMIT",
                        lifecycle_request,
                        **({"ts_ns": admit_clock["ts_ns"], "mono_ns": admit_clock["mono_ns"]}
                           if admit_clock is not None else {}),
                        attrs=request_attrs,
                    )
                # Send-time lookup: the ideal bound must reflect what a single
                # infinite-cache engine could have cached when this request was
                # dispatched, not what completed while it was in flight.
                ideal_cached_tokens = (
                    prefix_registry.match_tokens(prompt)
                    if prefix_registry is not None and isinstance(prompt, list)
                    else None
                )

                def on_first_token(
                    ts_ns: int,
                    mono_ns: int,
                    *,
                    writer: ClientLifecycleWriter | None = lifecycle_writer,
                    request: ClientLifecycleRequest | None = lifecycle_request,
                ) -> None:
                    if writer is not None and request is not None:
                        writer.emit(
                            "CLIENT_FIRST_TOKEN",
                            request,
                            ts_ns=ts_ns,
                            mono_ns=mono_ns,
                        )

                self._inflight.enter()
                try:
                    generation = await self._execute_with_migration(
                        prompt,
                        (
                            len(forced_output_token_ids)
                            if forced_output_token_ids is not None
                            else event.max_tokens
                        ),
                        stream=stream,
                        trace_start=trace_start,
                        stream_acc=stream_acc,
                        cache_salt=cache_salt,
                        routing_key=routing_key,
                        return_token_ids=True,
                        deterministic_sampling=strict_replay,
                        forced_output_token_ids=forced_output_token_ids,
                        connect_retry_limit=(
                            self._strict_replay_connect_retries
                            if strict_replay
                            else 0
                        ),
                        rid=(
                            lifecycle_request.rid
                            if lifecycle_request is not None
                            else None
                        ),
                        on_first_token=on_first_token,
                        request_timing=request_timing,
                        result=result,
                        lifecycle_request=lifecycle_request,
                        **({"image_data": image_data,
                            "expected_prompt_token_ids": expected_prompt}
                           if image_data is not None else {}),
                    )
                finally:
                    self._inflight.exit()
                    if request_acquired:
                        request_semaphore.release()
                prompt_token_ids = generation.prompt_token_ids
                generated_token_ids = generation.generated_token_ids
                if prompt_token_ids is None:
                    raise ValueError(
                        "return_token_ids response missing prompt_token_ids"
                    )
                if generated_token_ids is None:
                    raise ValueError("return_token_ids response missing token_ids")
                if (
                    (recorded_prompt is not None or trajectory_prefix_ids)
                    and prompt_token_ids != expected_prompt
                ):
                    raise ValueError(
                        "server prompt token IDs differ from strict replay trace: "
                        f"trace={trace.trace_id!r} request={request_index}"
                    )
                if (
                    strict_replay
                    and event.recorded_prompt_token_count is not None
                    and prompt_token_ids != expected_prompt
                ):
                    raise ValueError(
                        "server prompt token IDs differ from strict replay contract: "
                        f"trace={trace.trace_id!r} request={request_index}"
                    )
                if (
                    forced_output_token_ids is not None
                    and generated_token_ids != forced_output_token_ids
                ):
                    raise ValueError(
                        "server output token IDs differ from strict replay trace: "
                        f"trace={trace.trace_id!r} request={request_index}"
                    )
                if (strict_replay and self._replay_output_mode == "model"
                        and generation.completion_tokens != event.max_tokens):
                    raise ValueError("server output length differs from strict replay trace")
                # SGLang echoes expanded image placeholders; cache reuse must
                # compare the image-specific IDs, so changed pixels break LCP.
                cache_prompt_ids = cache_prompt if image_data is not None else prompt_token_ids
                client_prefix = (
                    common_prefix_length(previous_actual_ids, cache_prompt_ids)
                    if previous_actual_ids is not None
                    else 0
                )
                reusable_prefix = (
                    min(
                        client_prefix,
                        max(len(previous_actual_ids) - 1, 0),
                        max(len(cache_prompt_ids) - 1, 0),
                    )
                    if previous_actual_ids is not None
                    else 0
                )
                block_prefix = block_aligned_prefix_length(
                    reusable_prefix, prefix_block_size
                )
                cached_tokens = generation.cached_tokens
                if generation.usage is not None:
                    cached_tokens = metrics.record_usage(
                        generation.usage,
                        eligible_prompt_tokens=block_prefix,
                    )
                timestamp = time.perf_counter() - benchmark_start
                result.record_replay_request(
                    timestamp=timestamp,
                    prompt_tokens=generation.prompt_tokens,
                    completion_tokens=generation.completion_tokens,
                    client_prefix_tokens=client_prefix,
                    block_aligned_prefix_tokens=block_prefix,
                    server_cached_tokens=cached_tokens,
                )
                request_index += 1
                refresh()

                previous_actual_ids = cache_prompt_ids + generated_token_ids
                if actual_context is not None:
                    recorded_output = event.recorded_output_token_ids
                    if event.recorded_output_block_runs is not None:
                        assert block_tokens is not None
                        recorded_output = block_tokens.tokens(
                            event.recorded_output_block_runs, event.max_tokens
                        )
                    actual_context.commit(
                        context_source_ids, generated_token_ids,
                        recorded_output=recorded_output,
                        rid=lifecycle_request.rid if lifecycle_request is not None else None,
                    )
                if prefix_registry is not None:
                    prefix_registry.insert(previous_actual_ids)
                if context_source == "recorded":
                    if event.recorded_assistant is None and not (
                        recorded_prompt is not None
                        and (event.recorded_output_token_ids is not None
                             or event.recorded_output_block_runs is not None
                             or self._replay_output_mode == "model")
                    ):
                        raise ValueError(
                            "recorded context requested but generate event has no "
                            f"recorded_assistant: trace={trace.trace_id!r} "
                            f"request={request_index - 1}"
                        )
                    assistant_message = copy.deepcopy(event.recorded_assistant)
                else:
                    assistant_message = {
                        "role": "assistant",
                        "content": self._generated_assistant_content(event, generation),
                    }
                    if (
                        event.recorded_assistant is not None
                        and event.recorded_assistant.get("tool_calls")
                    ):
                        assistant_message["tool_calls"] = copy.deepcopy(
                            event.recorded_assistant["tool_calls"]
                        )
                if assistant_message is not None:
                    assistant_message["role"] = "assistant"
                    messages.append(assistant_message)
                if lifecycle_writer is not None and lifecycle_request is not None:
                    complete_attrs: dict[str, object] = {
                        "prompt_tokens": generation.prompt_tokens,
                        "cached_prompt_tokens": cached_tokens,
                        "completion_tokens": generation.completion_tokens,
                        "ideal_cached_tokens": ideal_cached_tokens,
                        "client_prefix_tokens": client_prefix,
                        "block_aligned_prefix_tokens": block_prefix,
                        **prefix_attrs,
                    }
                    if trajectory_prefix_ids:
                        complete_attrs["source_prompt_tokens"] = (
                            len(prompt) - len(trajectory_prefix_ids)
                        )
                    if request_timing is not None:
                        complete_attrs["client_timing"] = (
                            request_timing.as_lifecycle_attrs()
                        )
                    lifecycle_writer.emit(
                        "CLIENT_COMPLETE",
                        lifecycle_request,
                        attrs=complete_attrs,
                    )
                    active_lifecycle_complete = True
                    last_lifecycle_request = lifecycle_request

            if trace.metadata.get("control_only") is True:
                return
            if request_index == 0 and cut_s is not None:
                return  # the cut skipped every generate of this agent
            if not stream or metrics.prompt_tokens > 0:
                result.server_metrics.append(metrics)
            else:
                logger.warning("stream response usage missing; omitting server metrics")
            if dependency_state is None:
                result.record_trace_success(time.perf_counter() - trace_start)
            if stream:
                stream_acc.commit(result)
            lifecycle_writer = getattr(self, "_lifecycle_writer", None)
            if lifecycle_writer is not None and last_lifecycle_request is not None:
                lifecycle_writer.emit("TRACE_COMPLETE", last_lifecycle_request)
        except PhaseEnded:
            if stream:
                stream_acc.commit(result)
            if metrics.prompt_tokens > 0:
                result.server_metrics.append(metrics)
            if self._lifecycle_writer is not None and last_lifecycle_request is not None:
                self._lifecycle_writer.emit("TRACE_PAUSED", last_lifecycle_request)
        except asyncio.CancelledError as exc:
            if initial_prefill_slot is not None:
                initial_prefill_slot.failed(exc)
            lifecycle_writer = getattr(self, "_lifecycle_writer", None)
            if (
                lifecycle_writer is not None
                and active_lifecycle_request is not None
                and not active_lifecycle_complete
            ):
                failure_attrs: dict[str, object] = {
                    # Preserve the lifecycle schema consumed by cutoff audits.
                    "error_class": (
                        "CancelledError"
                        if isinstance(exc, BenchmarkWindowClosed)
                        else type(exc).__name__
                    )
                }
                if tag_requests and active_prompt_tokens is not None:
                    failure_attrs["prompt_tokens"] = active_prompt_tokens
                if active_request_timing is not None:
                    failure_attrs["client_timing"] = (
                        active_request_timing.as_lifecycle_attrs()
                    )
                lifecycle_writer.emit(
                    "REQUEST_FAILED",
                    active_lifecycle_request,
                    attrs=failure_attrs,
                )
            if dependency_state is not None:
                if not asyncio.current_task().cancelling():
                    if isinstance(exc, BenchmarkWindowClosed):
                        raise
                    raise ValueError(
                        "agent request cancelled before completion"
                    ) from exc
                raise
            return
        except (OpenAIError, ValueError, httpx.TransportError) as exc:
            if initial_prefill_slot is not None:
                initial_prefill_slot.failed(exc)
            # A stream cut mid-response surfaces as a bare httpx error on the fast path; it fails
            # this request like an API error instead of ending the client process.
            lifecycle_writer = getattr(self, "_lifecycle_writer", None)
            if (
                lifecycle_writer is not None
                and active_lifecycle_request is not None
                and not active_lifecycle_complete
            ):
                failure_attrs = {"error_class": type(exc).__name__,
                                 "error_cause": _root_cause_name(exc)}
                if tag_requests and active_prompt_tokens is not None:
                    failure_attrs["prompt_tokens"] = active_prompt_tokens
                if active_request_timing is not None:
                    failure_attrs["client_timing"] = (
                        active_request_timing.as_lifecycle_attrs()
                    )
                lifecycle_writer.emit(
                    "REQUEST_FAILED",
                    active_lifecycle_request,
                    attrs=failure_attrs,
                )
            if dependency_state is not None:
                raise
            result.record_failure()
            logger.warning(
                "replay trace failed",
                error=str(exc),
                trace_id=trace.trace_id,
                request=request_index,
            )
        except Exception as exc:
            if initial_prefill_slot is not None:
                initial_prefill_slot.failed(exc)
            lifecycle_writer = getattr(self, "_lifecycle_writer", None)
            if (
                lifecycle_writer is not None
                and active_lifecycle_request is not None
                and not active_lifecycle_complete
            ):
                failure_attrs = {"error_class": type(exc).__name__}
                if tag_requests and active_prompt_tokens is not None:
                    failure_attrs["prompt_tokens"] = active_prompt_tokens
                if active_request_timing is not None:
                    failure_attrs["client_timing"] = (
                        active_request_timing.as_lifecycle_attrs()
                    )
                lifecycle_writer.emit(
                    "REQUEST_FAILED",
                    active_lifecycle_request,
                    attrs=failure_attrs,
                )
            raise
        refresh()

    @staticmethod
    async def _prepare_replay_images(
        trace: ReplayTrace, event: Generate, prompt: Prompt
    ) -> tuple[list[str] | None, Prompt, Prompt]:
        """Load images on the client and derive expanded and cache IDs for replay.

        Only data URIs leave the client. Image file paths are never sent to workers.
        """
        if event.recorded_image_data is None:
            return None, prompt, prompt
        if not isinstance(prompt, list) or not isinstance(trace.metadata.get("image_config"), dict):
            raise ValueError("image replay requires prompt token IDs and metadata.image_config")

        def prepare():
            from trie.engine_files import image_layout

            prepare_image_layout = image_layout().prepare_image_layout

            root = Path(trace.metadata.get("image_root", "."))
            images = []
            for source in event.recorded_image_data:
                if source.startswith("data:"):
                    images.append(source)
                else:
                    path = root / source
                    mime = mimetypes.guess_type(path)[0] or "image/png"
                    images.append(f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii"))
            _, expanded, cached, _ = prepare_image_layout(prompt, images, trace.metadata["image_config"])
            return images, expanded, cached

        return await asyncio.to_thread(prepare)

    @staticmethod
    def _recorded_prompt(
        event: Generate, block_tokens: BlockTokens | None
    ) -> list[int] | None:
        """The generate's authoritative prompt IDs, recorded or expanded from block runs."""
        if event.recorded_prompt_token_ids is not None:
            return event.recorded_prompt_token_ids
        if event.recorded_prompt_block_runs is not None:
            assert block_tokens is not None
            return block_tokens.tokens(event.recorded_prompt_block_runs)
        return None

    @staticmethod
    def _request_tags(
        trace: ReplayTrace,
        task_identity: ClientLifecycleRequest | None,
        tag_requests: bool,
        *,
        warmup: bool,
    ) -> dict[str, object] | None:
        attrs: dict[str, object] = {}
        if task_identity is not None:
            attrs["task_request_id"] = task_identity.rid
        if tag_requests:
            attrs.update(agent_id=trace.trace_id, warmup=warmup)
        return attrs or None

    async def _send_warmup(
        self,
        trace: ReplayTrace,
        event: Generate,
        position: int,
        *,
        identity_trace_id: str,
        task_identity: ClientLifecycleRequest | None,
        replay_cycle_id: int,
        request_semaphore: asyncio.Semaphore | None,
        prefix_registry: GlobalPrefixRegistry | None,
        stream: bool,
        trace_start: float,
        cache_salt: str | None,
        routing_key: str | None,
        strict_replay: bool,
        block_tokens: BlockTokens | None,
        trajectory_prefix_ids: tuple[int, ...] = (),
        messages: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Prefill a skipped turn's prompt with one output token; not a measured request.

        Rows without prompt IDs render the turn's context with the chat template, as a
        measured request of the same row does.
        """
        recorded_prompt = (
            self._recorded_prompt(event, block_tokens) if strict_replay else None
        )
        if recorded_prompt is None and messages and event.recorded_image_data is None:
            tokenizer = self._require_tokenizer_manager("rendering warm-start prompts")
            recorded_prompt = tokenizer.encode(tokenizer.render_chat(
                messages, tools=trace.tools, add_generation_prompt=True,
                reasoning_effort=self._reasoning_effort))
        if recorded_prompt is None:
            raise ValueError(
                f"warm start needs authoritative prompt IDs: trace={trace.trace_id!r}"
            )
        if (
            trajectory_prefix_ids
            and event.recorded_prompt_token_count is not None
            and (
                len(recorded_prompt) != event.recorded_prompt_token_count
                or token_ids_sha256(recorded_prompt)
                != event.recorded_prompt_token_ids_sha256
            )
        ):
            raise ValueError(
                f"warmup prompt token IDs differ from strict replay contract: trace={trace.trace_id!r}"
            )
        prompt = list(recorded_prompt)
        if trajectory_prefix_ids:
            prompt[:0] = trajectory_prefix_ids
        image_data, expected_prompt, cache_prompt = await self._prepare_replay_images(
            trace, event, prompt
        )
        forced = None
        if self._replay_output_mode == "forced":
            if event.recorded_output_block_runs is not None:
                assert block_tokens is not None
                forced = block_tokens.tokens(event.recorded_output_block_runs, 1)
            else:
                forced = self._strict_replay_output_ids[id(event)][:1]
        lifecycle_writer = self._lifecycle_writer
        lifecycle_request = (
            lifecycle_writer.request_identity(
                trace_id=identity_trace_id,
                cycle_id=replay_cycle_id,
                event_index=position,
                routing_key=routing_key,
            )
            if lifecycle_writer is not None
            else None
        )
        tags = self._request_tags(trace, task_identity, True, warmup=True)
        if trajectory_prefix_ids:
            tags = {
                **(tags or {}),
                "trajectory_prefix_tokens": len(trajectory_prefix_ids),
                "trajectory_prefix_sha256": token_ids_sha256(list(trajectory_prefix_ids)),
                "source_prompt_tokens": len(recorded_prompt),
            }
        acquired = False
        completed = False
        try:
            if request_semaphore is not None:
                await phase_wait(request_semaphore.acquire())
                acquired = True
            check_phase()
            admit_clock = self._check_benchmark_admission("lifecycle_admission")
            if lifecycle_writer is not None and lifecycle_request is not None:
                lifecycle_writer.emit(
                    "CLIENT_ADMIT",
                    lifecycle_request,
                    **({"ts_ns": admit_clock["ts_ns"], "mono_ns": admit_clock["mono_ns"]}
                       if admit_clock is not None else {}),
                    attrs=tags,
                )
            ideal_cached_tokens = (
                prefix_registry.match_tokens(cache_prompt)
                if prefix_registry is not None
                else None
            )
            self._inflight.enter()
            try:
                generation = await self._execute_with_migration(
                    prompt,
                    1,
                    stream=stream,
                    trace_start=trace_start,
                    stream_acc=StreamAccumulator(),
                    cache_salt=cache_salt,
                    routing_key=routing_key,
                    return_token_ids=True,
                    deterministic_sampling=strict_replay,
                    forced_output_token_ids=forced,
                    connect_retry_limit=self._strict_replay_connect_retries,
                    rid=lifecycle_request.rid if lifecycle_request is not None else None,
                    on_first_token=None,
                    request_timing=None,
                    result=None,
                    lifecycle_request=lifecycle_request,
                    **({"image_data": image_data,
                        "expected_prompt_token_ids": expected_prompt}
                       if image_data is not None else {}),
                )
            finally:
                self._inflight.exit()
                if acquired:
                    request_semaphore.release()
                    acquired = False
            if generation.prompt_token_ids != expected_prompt:
                raise ValueError(
                    f"server prompt token IDs differ from warmup: trace={trace.trace_id!r}"
                )
            generated = generation.generated_token_ids or []
            if (forced is not None and generated != forced) or len(generated) != 1:
                raise ValueError(
                    f"server output token IDs differ from warmup: trace={trace.trace_id!r}"
                )
            actual_ids = cache_prompt + generated
            if prefix_registry is not None:
                prefix_registry.insert(actual_ids)
            if lifecycle_writer is not None and lifecycle_request is not None:
                lifecycle_writer.emit(
                    "CLIENT_COMPLETE",
                    lifecycle_request,
                    attrs={
                        "prompt_tokens": generation.prompt_tokens,
                        "cached_prompt_tokens": generation.cached_tokens,
                        "completion_tokens": generation.completion_tokens,
                        "ideal_cached_tokens": ideal_cached_tokens,
                        **(tags or {}),
                    },
                )
            completed = True
            return actual_ids
        except BaseException as exc:
            if acquired:
                request_semaphore.release()
            if (
                not completed
                and not isinstance(exc, PhaseEnded)
                and lifecycle_writer is not None
                and lifecycle_request is not None
            ):
                lifecycle_writer.emit(
                    "REQUEST_FAILED",
                    lifecycle_request,
                    attrs={
                        "error_class": type(exc).__name__,
                        "prompt_tokens": len(prompt),
                        **(tags or {}),
                    },
                )
            raise

    async def _run_replay_task(
        self,
        task: ReplayTask,
        *,
        benchmark_start: float,
        result: BenchmarkResult,
        refresh: Callable[[], None],
        delay_scale: float,
        replay_cycle_id: int,
        warm_start_draw: float | None = None,
        initial_prefill_slot: InitialSlot | None = None,
        **kwargs: Any,
    ) -> None:
        if warm_start_draw is not None and task.metadata.get("cc_replay_policy", {}).get("zero_output") == "drop-advance":
            raise ValueError("CC drop-advance rewrites the timeline; use warm_start=off")
        if warm_start_draw is not None and task.metadata.get("cc_fixed_start"):
            raise ValueError("Fixed-cut CC data requires warm_start=off; it is already cut")
        state = DependencyState(task)
        cut_s = (
            _stationary_cut(task, warm_start_draw)
            if warm_start_draw is not None
            else None
        )
        if initial_prefill_slot is not None:
            initial_prefill_slot.register(task.traces, cut_s)
        shared_salt = (
            f"{kwargs['run_cache_salt']}:{task.trace_id}:{uuid.uuid4().hex}"
            if task.cache_salt_scope == "task" and kwargs["cache_salt_mode"] == "session"
            else None
        )
        lifecycle_writer = self._lifecycle_writer
        task_identity = (
            lifecycle_writer.request_identity(
                trace_id=json.dumps(["task", task.trace_id]),
                cycle_id=replay_cycle_id,
                event_index=0,
                routing_key=None,
            )
            if lifecycle_writer is not None
            else None
        )

        def emit(event: str, **attrs: object) -> None:
            if lifecycle_writer is not None and task_identity is not None:
                lifecycle_writer.emit(event, task_identity, attrs=attrs)

        try:
            delay = (
                benchmark_start
                + task.start_offset_s * delay_scale
                - time.perf_counter()
            )
            if delay > 0:
                await phase_wait(asyncio.sleep(delay))
            started = time.perf_counter()
            emit("TASK_START", agents=sum(
                trace.metadata.get("control_only") is not True for trace in task.traces
            ))
            try:
                # A failed/cancelled group cannot release dependents as successes.
                # TaskGroup also joins cancelled siblings before the slot is reused.
                async with asyncio.TaskGroup() as group:
                    for trace in task.traces:
                        group.create_task(
                            self._run_replay_trace(
                                trace,
                                benchmark_start=benchmark_start,
                                result=result,
                                refresh=refresh,
                                delay_scale=delay_scale,
                                replay_cycle_id=replay_cycle_id,
                                dependency_state=state,
                                identity_trace_id=json.dumps(
                                    ["task", task.trace_id, trace.trace_id]
                                ),
                                task_identity=task_identity,
                                cache_salt_override=shared_salt,
                                cut_s=cut_s,
                                tag_requests=True,
                                **({"initial_prefill_slot": initial_prefill_slot}
                                   if initial_prefill_slot is not None else {}),
                                **kwargs,
                            )
                        )
            except* (OpenAIError, ValueError, httpx.TransportError) as exc:
                result.record_failure()
                emit("TASK_FAILED", error_class=type(exc).__name__)
                logger.warning(
                    "multi-agent replay failed", trace_id=task.trace_id, error=str(exc)
                )
            else:
                check_phase()
                self._check_benchmark_admission("task_complete")
                latency = time.perf_counter() - started
                result.record_trace_success(latency)
                emit("TASK_COMPLETE", latency_s=latency)
        except PhaseEnded:
            emit("TASK_PAUSED")
        except asyncio.CancelledError:
            emit("TASK_CANCELLED")
            raise
        refresh()

    async def _run_workload(
        self,
        workload: TraceWorkload,
        *,
        benchmark_start: float,
        request_semaphore: asyncio.Semaphore | None = None,
        result: BenchmarkResult,
        stream: bool,
        refresh: Callable[[], None],
        delay_scale: float,
        max_user_delay: float | None,
        max_tool_delay: float | None,
        context_source: Literal["generated", "recorded"],
        prefix_block_size: int,
        prefix_registry: GlobalPrefixRegistry | None = None,
        cache_salt_mode: Literal["global", "session"],
        run_cache_salt: str | None,
        strict_replay: bool,
        replay_cycle_id: int = 0,
        warm_start_draw: float | None = None,
        trajectory_prefix_tokens: int = 0,
        initial_prefill_slot: InitialSlot | None = None,
    ) -> None:
        lazy = isinstance(workload, LazyRow)
        if lazy:
            workload = await asyncio.to_thread(workload.load)
        if isinstance(workload, (ReplayTrace, ReplayTask)):
            # One prefix per admitted complete task, shared through **kwargs with
            # every child. A repeated or concurrent instance gets a fresh prefix.
            prefix_ids = (
                self._require_tokenizer_manager("trajectory prefixes").random_prefix_tokens(
                    trajectory_prefix_tokens
                )
                if trajectory_prefix_tokens
                else ()
            )
            if isinstance(workload, ReplayTask):
                run_replay = self._run_replay_task
                extra: dict[str, Any] = {"warm_start_draw": warm_start_draw}
            else:
                run_replay = self._run_replay_trace
                extra = {
                    "cut_s": (
                        _stationary_cut(workload, warm_start_draw)
                        if warm_start_draw is not None
                        else None
                    ),
                    "tag_requests": warm_start_draw is not None,
                }
                if initial_prefill_slot is not None:
                    initial_prefill_slot.register([workload], extra["cut_s"])
            if initial_prefill_slot is not None:
                extra["initial_prefill_slot"] = initial_prefill_slot
            try:
                await run_replay(
                    workload,
                    benchmark_start=benchmark_start,
                    request_semaphore=request_semaphore,
                    result=result,
                    stream=stream,
                    refresh=refresh,
                    delay_scale=delay_scale,
                    max_user_delay=max_user_delay,
                    max_tool_delay=max_tool_delay,
                    context_source=context_source,
                    prefix_block_size=prefix_block_size,
                    prefix_registry=prefix_registry,
                    cache_salt_mode=cache_salt_mode,
                    run_cache_salt=run_cache_salt,
                    strict_replay=strict_replay,
                    replay_cycle_id=replay_cycle_id,
                    trajectory_prefix_ids=prefix_ids,
                    **extra,
                )
            except BenchmarkWindowClosed:
                # Keep ordinary/external cancellation distinct from admission cutoff.
                pass
            finally:
                if lazy:
                    for event in iter_generates(workload):
                        self._strict_replay_output_ids.pop(id(event), None)
            return
        await self._run_legacy_workload(
            workload,
            benchmark_start=benchmark_start,
            result=result,
            stream=stream,
            refresh=refresh,
        )

    def _order_workloads(
        self,
        workloads: list[TraceWorkload],
        *,
        workload_order: Literal["natural", "shuffle", "prefix"],
        prefix_sort_tokens: int,
    ) -> list[TraceWorkload]:
        ordered = list(workloads)
        if workload_order == "shuffle":
            self._rng.shuffle(ordered)
            return ordered
        if workload_order != "prefix":
            return ordered

        def prefix_key(workload: TraceWorkload) -> tuple[int, tuple[int, ...]]:
            if isinstance(workload, ReplayTask):
                workload = workload.traces[0]
            if not isinstance(workload, ReplayTrace):
                return (1, ())
            first_generate = next(
                (event for event in workload.events if isinstance(event, Generate)),
                None,
            )
            if (
                first_generate is not None
                and first_generate.recorded_prompt_token_ids is not None
            ):
                return (
                    0,
                    tuple(
                        first_generate.recorded_prompt_token_ids[:prefix_sort_tokens]
                    ),
                )
            tokenizer = self._require_tokenizer_manager(
                "prefix ordering traces without authoritative prompt token IDs"
            )
            prompt = tokenizer.render_chat(
                workload.initial_messages,
                tools=workload.tools,
                add_generation_prompt=True,
                reasoning_effort=self._reasoning_effort,
            )
            return (
                0,
                tuple(tokenizer.encode(prompt)[:prefix_sort_tokens]),
            )

        ordered.sort(key=prefix_key)
        return ordered

    async def run(
        self,
        workload: list[TraceWorkload] | str,
        concurrency: int | None = None,
        duration: float = 300.0,
        arrival_rate: float | None = None,
        request_concurrency: int | None = None,
        duration_update_interval: float | None = None,
        num_gpus: int | None = None,
        stream: bool = False,
        delay_scale: float = 1.0,
        max_user_delay: float | None = None,
        max_tool_delay: float | None = None,
        context_source: Literal["generated", "recorded"] = "generated",
        prefix_block_size: int = 16,
        cache_salt_mode: Literal["global", "session"] = "session",
        workload_order: Literal["natural", "shuffle", "prefix"] = "natural",
        prefix_sort_tokens: int = 256,
        replay_once: bool = False,
        drain_timeout: float = 0.0,
        strict_replay: bool = False,
        warm_start: Literal["off", "stationary"] = "off",
        lazy_workload: bool = False,
        trajectory_prefix_tokens: int = 0,
        shard_index: int = 0,
        shard_count: int = 1,
        workload_schedule: str | None = None,
    ) -> BenchmarkResult:
        if not 0 <= shard_index < shard_count:
            raise ValueError("shard_index must be in [0, shard_count)")
        if type(trajectory_prefix_tokens) is not int or trajectory_prefix_tokens < 0:
            raise ValueError("trajectory_prefix_tokens must be a nonnegative integer")
        if trajectory_prefix_tokens and not strict_replay:
            raise ValueError("trajectory_prefix_tokens requires strict_replay=true")
        if strict_replay and self._max_retries != 0:
            raise ValueError("strict_replay requires max_retries=0")
        if self._strict_replay_openai_fast_path and not strict_replay:
            raise ValueError(
                "strict_replay_openai_fast_path requires strict_replay=true"
            )
        if self._strict_replay_openai_fast_path and not stream:
            raise ValueError(
                "strict_replay_openai_fast_path requires stream=true"
            )
        if warm_start not in ("off", "stationary"):
            raise ValueError("warm_start must be off or stationary")
        if warm_start == "stationary" and concurrency is None:
            raise ValueError("warm_start=stationary requires concurrency")
        if lazy_workload and (not strict_replay or workload_order == "prefix"):
            raise ValueError(
                "lazy_workload requires strict_replay and a natural or shuffled order"
            )
        schedule = None
        source_lists = {}
        if workload_schedule is not None:
            if (concurrency is None or not strict_replay or replay_once or arrival_rate is not None
                    or drain_timeout != 0 or self._lifecycle_writer is None):
                # Either cache salt mode works: request identities carry the slot label, and a
                # global salt only lets different sessions share an identical prefix.
                raise ValueError("Phased replay requires fixed slots, strict replay, lifecycle and zero final drain")
            schedule = Schedule.load(workload_schedule, concurrency * shard_count, duration)
            for label, path in schedule.workloads.items():
                # Every process holds every template, so a global slot replays the same templates
                # (and, with one seed for all processes, the same cuts) whatever the process count.
                source_lists[label] = self._order_workloads(
                    load_workloads(path, lazy=lazy_workload),
                    workload_order=workload_order, prefix_sort_tokens=prefix_sort_tokens,
                )
                if not source_lists[label]:
                    raise ValueError(f"Empty workload shard: {label}")
            workload_list = [item for items in source_lists.values() for item in items]
        else:
            # Partition before per-client randomization so seeds cannot overlap shards.
            workload_list = load_workloads(workload, lazy=lazy_workload)[shard_index::shard_count]
            workload_list = self._order_workloads(
                workload_list, workload_order=workload_order, prefix_sort_tokens=prefix_sort_tokens,
            )
        if not workload_list:
            raise ValueError("workload must contain at least one trace")
        if concurrency is not None and arrival_rate is not None:
            raise ValueError("concurrency and arrival_rate are mutually exclusive")
        if (
            concurrency is None
            and arrival_rate is None
            and not (replay_once and request_concurrency is not None)
        ):
            raise ValueError(
                "specify concurrency or arrival_rate, or use request_concurrency "
                "with replay_once"
            )
        if concurrency is not None and concurrency <= 0:
            raise ValueError("concurrency must be greater than 0")
        if arrival_rate is not None and arrival_rate <= 0:
            raise ValueError("arrival_rate must be greater than 0")
        if request_concurrency is not None and request_concurrency <= 0:
            raise ValueError("request_concurrency must be greater than 0")
        if duration_update_interval is not None and duration_update_interval <= 0:
            raise ValueError("duration_update_interval must be greater than 0")
        if delay_scale < 0:
            raise ValueError("delay_scale must be non-negative")
        if max_user_delay is not None and max_user_delay < 0:
            raise ValueError("max_user_delay must be non-negative")
        if max_tool_delay is not None and max_tool_delay < 0:
            raise ValueError("max_tool_delay must be non-negative")
        if context_source not in ("generated", "recorded"):
            raise ValueError("context_source must be generated or recorded")
        if prefix_block_size <= 0:
            raise ValueError("prefix_block_size must be greater than 0")
        if cache_salt_mode not in ("global", "session"):
            raise ValueError("cache_salt_mode must be global or session")
        if workload_order not in ("natural", "shuffle", "prefix"):
            raise ValueError("workload_order must be natural, shuffle, or prefix")
        if prefix_sort_tokens <= 0:
            raise ValueError("prefix_sort_tokens must be greater than 0")
        if drain_timeout < 0:
            raise ValueError("drain_timeout must be non-negative")
        replay_traces = [
            trace
            for item in workload_list
            for trace in (item.traces if isinstance(item, ReplayTask) else [item])
            if isinstance(trace, ReplayTrace)
        ]
        if strict_replay:
            if any(
                not isinstance(item, (ReplayTrace, ReplayTask, LazyRow))
                for item in workload_list
            ):
                raise ValueError(
                    "strict_replay only supports schema-v2/v3 replay traces"
                )
            context_source = "recorded"
            self._prepare_strict_replay_output_ids(replay_traces)
        else:
            self._strict_replay_output_ids = {}
            if any(isinstance(item, Workload) for item in workload_list):
                self._require_tokenizer_manager("synthetic workloads")
            if replay_traces:
                self._require_tokenizer_manager(
                    "replay prompts without strict authoritative token IDs"
                )
        run_cache_salt = uuid.uuid4().hex if cache_salt_mode == "session" else None
        if trajectory_prefix_tokens:
            # Load/cache the vocabulary before the measured window, not per request.
            self._require_tokenizer_manager("trajectory prefixes").regular_token_ids
        benchmark_log_fields = {
            "model": self._model,
            "workload_templates": len(workload_list),
            "duration": duration,
            "num_gpus": num_gpus,
            "delay_scale": delay_scale,
            "context_source": context_source,
            "cache_salt_mode": cache_salt_mode,
            "trajectory_prefix_tokens": trajectory_prefix_tokens,
            "workload_order": workload_order,
            "replay_once": replay_once,
            "drain_timeout": drain_timeout,
            "strict_replay": strict_replay,
            "strict_replay_openai_fast_path": (
                self._strict_replay_openai_fast_path
            ),
        }
        if warm_start != "off":
            benchmark_log_fields["warm_start"] = warm_start
        if lazy_workload:
            benchmark_log_fields["lazy_workload"] = lazy_workload
        if concurrency is not None:
            benchmark_log_fields["concurrency"] = concurrency
        if arrival_rate is not None:
            benchmark_log_fields["arrival_rate"] = arrival_rate
        if request_concurrency is not None:
            benchmark_log_fields["request_concurrency"] = request_concurrency
        if duration_update_interval is not None:
            benchmark_log_fields["duration_update_interval"] = duration_update_interval
        phase_clock = await shared_clock(shard_index, shard_count) if schedule else None
        logger.info(
            "starting benchmark",
            **benchmark_log_fields,
        )
        result = BenchmarkResult()
        if replay_once:
            result.expected_requests = len(workload_list)
        boundary = BenchmarkBoundary(duration, drain_timeout, start=phase_clock)
        readiness_directory = os.environ.get("TRIE_INITIAL_PREFILL_READY_DIR")
        initial_prefills = None
        if readiness_directory:
            if concurrency is None or not strict_replay:
                raise ValueError("Initial-prefix readiness requires fixed-concurrency strict replay")
            initial_prefills = InitialPrefillReadiness(readiness_directory, shard_index, concurrency, boundary)
        self._benchmark_boundary = boundary
        benchmark_start = boundary.start["perf_ns"] / 1e9
        request_semaphore = (
            asyncio.Semaphore(request_concurrency)
            if request_concurrency is not None
            else None
        )
        self._result = result
        self._benchmark_start = benchmark_start
        self._inflight = _InflightGauge()

        with boundary, Live(
            build_progress_lines(result, 0.0, duration),
            console=console,
            auto_refresh=False,
            transient=True,
        ) as live:
            next_duration_update = duration_update_interval

            def refresh() -> None:
                live.update(
                    build_progress_lines(
                        result,
                        time.perf_counter() - benchmark_start,
                        duration,
                    ),
                    refresh=True,
                )

            def log_duration_update() -> None:
                nonlocal next_duration_update
                if next_duration_update is None:
                    return
                elapsed = time.perf_counter() - benchmark_start
                if elapsed < next_duration_update:
                    return
                logger.info(
                    "benchmark in progress",
                    remaining_duration=round(max(duration - elapsed, 0.0), 1),
                    active_workloads=len(active),
                    inflight_requests=self._inflight.current,
                )
                next_duration_update = elapsed + duration_update_interval

            def check_completed_workload(task: asyncio.Task[None]) -> None:
                # The request guard can cancel task_complete while asyncio.wait
                # returns that same task at the fixed deadline. This is the
                # normal zero-drain cutoff, not a failed workload/teardown.
                try:
                    task.result()  # Preserve errors and cancellation before cutoff.
                except asyncio.CancelledError as exc:
                    # Only the boundary's own stop counts as cutoff; an
                    # independent cancellation still fails the run.
                    if (boundary.zero_drain_guard
                            and time.monotonic_ns() >= boundary.deadline["mono_ns"]
                            and isinstance(exc, BenchmarkWindowClosed)):
                        boundary.counters["deadline_completed_task_cancellations"] += 1
                        return
                    raise

            active: set[asyncio.Task[None]] = set()
            # A lazily loaded row is repeatable: admit_once needs the parsed metadata.
            repeatable = [
                item
                for item in workload_list
                if not (isinstance(item, ReplayTrace) and item.metadata.get("admit_once"))
            ]
            workload_iter = (
                iter(workload_list)
                if replay_once or not repeatable
                else chain(workload_list, cycle(repeatable))
            )
            next_arrival = benchmark_start
            arrival_interval = 1.0 / arrival_rate if arrival_rate is not None else None
            exhausted = False
            admitted_workloads = 0
            replay_cycles: dict[tuple[int, str], int] = defaultdict(int)
            prefix_registry = GlobalPrefixRegistry(prefix_block_size)

            if schedule is not None:
                seeds = [self._rng.getrandbits(64) for _ in range(concurrency * shard_count)]

                async def run_slot(local_slot: int) -> None:
                    index = local_slot * shard_count + shard_index
                    rng = random.Random(seeds[index])
                    sequence = defaultdict(int)
                    seen = set()
                    assigned = None
                    instance = 0
                    while time.perf_counter() < benchmark_start + duration:
                        phase = schedule.phase_index(time.perf_counter() - benchmark_start)
                        if phase == len(schedule.phases):
                            return
                        label = schedule.label(phase, index)
                        if label is None:
                            if assigned is not None:
                                identity = self._lifecycle_writer.request_identity(
                                    trace_id=f"slot:{index}", cycle_id=instance, event_index=0, routing_key=None,
                                )
                                self._lifecycle_writer.emit("SLOT_IDLE", identity, attrs={
                                    "workload_slot": index, "workload_label": None,
                                    "workload_phase": phase, "previous_workload": assigned,
                                    "inflight_requests": 0,
                                })
                                assigned = None
                            # The previous workload has returned after its in-flight
                            # requests finished. Park only this empty admission slot.
                            await asyncio.sleep(max(0, benchmark_start
                                + schedule.change_at(phase, index) - time.perf_counter()))
                            continue
                        slot = Slot(schedule, index, label, benchmark_start,
                                    benchmark_start + schedule.change_at(phase, index))
                        token = current_slot.set(slot)
                        try:
                            identity = self._lifecycle_writer.request_identity(
                                trace_id=f"slot:{index}", cycle_id=instance, event_index=0, routing_key=None,
                            )
                            if assigned != label:
                                self._lifecycle_writer.emit("SLOT_ASSIGN", identity,
                                    attrs={"previous_workload": assigned, "target_change_elapsed_s": slot.until - benchmark_start})
                                assigned = label
                            items = source_lists[label]
                            # Visit every template even when pool size and slot count share a divisor.
                            stride = concurrency * shard_count
                            while math.gcd(stride, len(items)) != 1:
                                stride += 1
                            item = items[(index + sequence[label] * stride) % len(items)]
                            sequence[label] += 1
                            first = instance == 0
                            # The warm start skips the cold start of the run only: a slot that takes up a
                            # workload in a later phase is a new user and replays it from its first turn.
                            cut = (rng.random() if warm_start == "stationary" and label not in seen
                                   and phase == 0 else None)
                            seen.add(label)
                            await self._run_workload(
                                item, benchmark_start=benchmark_start,
                                request_semaphore=request_semaphore, result=result, stream=stream,
                                refresh=refresh, delay_scale=delay_scale,
                                max_user_delay=max_user_delay, max_tool_delay=max_tool_delay,
                                context_source=context_source, prefix_block_size=prefix_block_size,
                                prefix_registry=prefix_registry, cache_salt_mode=cache_salt_mode,
                                run_cache_salt=f"{run_cache_salt}:{label}", strict_replay=strict_replay,
                                replay_cycle_id=index * 1_000_000_000 + instance, warm_start_draw=cut,
                                trajectory_prefix_tokens=trajectory_prefix_tokens,
                                **({"initial_prefill_slot": initial_prefills.slot(local_slot)}
                                   if first and initial_prefills is not None else {}),
                            )
                            instance += 1
                            if time.perf_counter() >= slot.until:
                                self._lifecycle_writer.emit("SLOT_RELEASE", identity,
                                    attrs={"reason": "phase_boundary", "inflight_requests": 0})
                        finally:
                            current_slot.reset(token)

                async def mark_phases() -> None:
                    elapsed = 0.0
                    for i, phase in enumerate(schedule.phases):
                        await asyncio.sleep(max(0, benchmark_start + elapsed - time.perf_counter()))
                        identity = self._lifecycle_writer.request_identity(
                            trace_id="phase", cycle_id=i, event_index=0, routing_key=None,
                        )
                        self._lifecycle_writer.emit("WORKLOAD_PHASE_START", identity,
                            attrs={"phase_index": i, "target_elapsed_s": elapsed, "target_slots": phase["slots"]})
                        elapsed += phase["duration_s"]

                active = {asyncio.create_task(run_slot(i)) for i in range(concurrency)}
                active.add(asyncio.create_task(mark_phases()))
                done, active = await asyncio.wait(active,
                    timeout=max(0, benchmark_start + duration - time.perf_counter()),
                    return_when=asyncio.FIRST_EXCEPTION)
                for task in done:
                    check_completed_workload(task)

            while schedule is None and time.perf_counter() - benchmark_start < duration:
                log_duration_update()
                if arrival_interval is not None:
                    done = {task for task in active if task.done()}
                    active -= done
                    for task in done:
                        check_completed_workload(task)

                if concurrency is not None and len(active) == concurrency:
                    elapsed = time.perf_counter() - benchmark_start
                    timeout = max(duration - elapsed, 0.0)
                    if next_duration_update is not None:
                        timeout = min(
                            timeout,
                            max(next_duration_update - elapsed, 0.0),
                        )
                    done, active = await asyncio.wait(
                        active,
                        timeout=timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        check_completed_workload(task)
                    if not done:
                        continue

                if arrival_interval is not None:
                    wait_until = min(next_arrival, benchmark_start + duration)
                    if next_duration_update is not None:
                        wait_until = min(
                            wait_until,
                            benchmark_start + next_duration_update,
                        )
                    timeout = wait_until - time.perf_counter()
                    if timeout > 0:
                        await asyncio.sleep(timeout)
                        continue

                # The waits above may return a completed workload at or just after
                # the fixed-duration deadline.  Recheck immediately before
                # admission so a closed-loop refill cannot start outside the
                # measurement window.
                if time.perf_counter() - benchmark_start >= duration:
                    break

                try:
                    next_workload = next(workload_iter)
                except StopIteration:
                    exhausted = True
                    break
                replay_cycle_id = 0
                replayable = isinstance(next_workload, (ReplayTrace, ReplayTask, LazyRow))
                if replayable:
                    cycle_key = (next_workload.schema_version, next_workload.trace_id)
                    replay_cycle_id = replay_cycles[cycle_key]
                    replay_cycles[cycle_key] += 1
                # The first admission of every slot starts at a cut.
                warm_start_draw = (
                    self._rng.random()
                    if warm_start == "stationary"
                    and replayable
                    and concurrency is not None
                    and admitted_workloads < concurrency
                    else None
                )
                active.add(
                    asyncio.create_task(
                        self._run_workload(
                            next_workload,
                            benchmark_start=benchmark_start,
                            request_semaphore=request_semaphore,
                            result=result,
                            stream=stream,
                            refresh=refresh,
                            delay_scale=delay_scale,
                            max_user_delay=max_user_delay,
                            max_tool_delay=max_tool_delay,
                            context_source=context_source,
                            prefix_block_size=prefix_block_size,
                            prefix_registry=prefix_registry,
                            cache_salt_mode=cache_salt_mode,
                            run_cache_salt=run_cache_salt,
                            strict_replay=strict_replay,
                            replay_cycle_id=replay_cycle_id,
                            warm_start_draw=warm_start_draw,
                            trajectory_prefix_tokens=trajectory_prefix_tokens,
                            **({"initial_prefill_slot": initial_prefills.slot(admitted_workloads)}
                               if initial_prefills is not None and admitted_workloads < concurrency else {}),
                        )
                    )
                )
                admitted_workloads += 1
                if replay_once and admitted_workloads == len(workload_list):
                    exhausted = True
                    break
                if arrival_interval is not None:
                    next_arrival += arrival_interval
            if exhausted:
                logger.info("all replay traces admitted", active_traces=len(active))
            boundary.stop_trace_admission("exhausted" if exhausted else "duration")
            if active and drain_timeout > 0:
                done, pending = await asyncio.wait(active, timeout=drain_timeout)
                for task in done:
                    task.result()
                active = pending
            await cancel_workloads(active, boundary)
        result.wall_time = time.perf_counter() - benchmark_start
        result.inflight_requests = self._inflight.summary()
        logger.info("in-flight model requests", **result.inflight_requests)
        log_summary(result, num_gpus=num_gpus)
        return result

    def sync_run(
        self,
        workload: list[TraceWorkload] | str,
        concurrency: int | None = None,
        duration: float = 300.0,
        arrival_rate: float | None = None,
        request_concurrency: int | None = None,
        duration_update_interval: float | None = None,
        num_gpus: int | None = None,
        stream: bool = False,
        delay_scale: float = 1.0,
        max_user_delay: float | None = None,
        max_tool_delay: float | None = None,
        context_source: Literal["generated", "recorded"] = "generated",
        prefix_block_size: int = 16,
        cache_salt_mode: Literal["global", "session"] = "session",
        workload_order: Literal["natural", "shuffle", "prefix"] = "natural",
        prefix_sort_tokens: int = 256,
        replay_once: bool = False,
        drain_timeout: float = 0.0,
        strict_replay: bool = False,
        warm_start: Literal["off", "stationary"] = "off",
        lazy_workload: bool = False,
        trajectory_prefix_tokens: int = 0,
        shard_index: int = 0,
        shard_count: int = 1,
        workload_schedule: str | None = None,
    ) -> BenchmarkResult:
        runner = asyncio.Runner()
        cleanup_failed = False
        try:
            return runner.run(
                self.run(
                    workload,
                    concurrency,
                    duration,
                    arrival_rate=arrival_rate,
                    request_concurrency=request_concurrency,
                    duration_update_interval=duration_update_interval,
                    num_gpus=num_gpus,
                    stream=stream,
                    delay_scale=delay_scale,
                    max_user_delay=max_user_delay,
                    max_tool_delay=max_tool_delay,
                    context_source=context_source,
                    prefix_block_size=prefix_block_size,
                    cache_salt_mode=cache_salt_mode,
                    workload_order=workload_order,
                    prefix_sort_tokens=prefix_sort_tokens,
                    replay_once=replay_once,
                    drain_timeout=drain_timeout,
                    strict_replay=strict_replay,
                    warm_start=warm_start,
                    lazy_workload=lazy_workload,
                    trajectory_prefix_tokens=trajectory_prefix_tokens,
                    shard_index=shard_index,
                    shard_count=shard_count,
                    workload_schedule=workload_schedule,
                )
            )
        except BenchmarkCancellationTimeout:
            cleanup_failed = True
            runner.get_loop().close()
            raise
        except KeyboardInterrupt:
            logger.warning("benchmark interrupted")
            assert self._result is not None and self._benchmark_start is not None
            result = self._result
            result.wall_time = time.perf_counter() - self._benchmark_start
            log_summary(result, num_gpus=num_gpus)
            return result
        finally:
            if not cleanup_failed:
                runner.close()
            self._result = None
            self._benchmark_start = None
