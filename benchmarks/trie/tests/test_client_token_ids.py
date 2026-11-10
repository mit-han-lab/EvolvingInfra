import asyncio
from types import SimpleNamespace

import pytest
from openai.types.completion_choice import CompletionChoice
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails

from trie.client import Client, _extension_token_ids
from trie.client_lifecycle import RequestTiming
from trie.results import StreamAccumulator


class ByteTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text.encode())


class AsyncChunks:
    def __init__(self, chunks: list[object]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class FakeCompletions:
    def __init__(
        self,
        response: object,
        *,
        yield_before_response: bool = False,
    ) -> None:
        self._response = response
        self._yield_before_response = yield_before_response
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._yield_before_response:
            await asyncio.sleep(0)
        return self._response


def _usage() -> CompletionUsage:
    return CompletionUsage(
        completion_tokens=2,
        prompt_tokens=3,
        total_tokens=5,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=1),
    )


def _client(
    response: object,
    *,
    yield_before_response: bool = False,
) -> tuple[Client, FakeCompletions]:
    completions = FakeCompletions(
        response,
        yield_before_response=yield_before_response,
    )
    client = Client.__new__(Client)
    client._model = "fake"
    client._client = SimpleNamespace(completions=completions)
    client._tokenizer_manager = ByteTokenizer()
    return client, completions


def test_client_rejects_nonpositive_stream_interval() -> None:
    with pytest.raises(ValueError, match="stream_interval must be positive"):
        Client(
            endpoint="http://127.0.0.1:1/v1",
            model="fake",
            stream_interval=0,
        )


def test_non_stream_response_reads_vllm_choice_token_ids() -> None:
    choice = CompletionChoice(
        finish_reason="length",
        index=0,
        logprobs=None,
        text="hi",
        prompt_token_ids=[1, 2, 3],
        token_ids=[32, 48],
    )
    response = SimpleNamespace(
        choices=[choice],
        usage=_usage(),
        model_extra={
            "sglext": {
                "cached_tokens_details": {
                    "device": 1,
                    "host": 0,
                    "storage": 2,
                    "storage_backend": "file",
                },
            }
        },
    )
    client, completions = _client(response)

    result = asyncio.run(
        client._execute_request(
            "abc",
            2,
            stream=False,
            trace_start=0.0,
            stream_acc=StreamAccumulator(),
            return_token_ids=True,
        )
    )

    assert result.prompt_token_ids == [1, 2, 3]
    assert result.generated_token_ids == [32, 48]
    assert result.cached_tokens == 1
    assert result.cached_tokens_details == {
        "device": 1,
        "host": 0,
        "storage": 2,
        "storage_backend": "file",
    }
    assert result.request_started_at_s is not None
    assert result.request_finished_at_s is not None
    assert result.request_finished_at_s >= result.request_started_at_s
    assert completions.calls[0]["extra_body"]["return_token_ids"] is True
    assert (
        completions.calls[0]["extra_body"]["return_cached_tokens_details"]
        is True
    )


def test_stream_response_accumulates_choice_token_id_deltas() -> None:
    chunks = AsyncChunks(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text="h",
                        model_extra={
                            "prompt_token_ids": [1, 2, 3],
                            "token_ids": [32],
                        },
                    )
                ],
                usage=None,
                model_extra={},
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text="i",
                        model_extra={
                            "prompt_token_ids": None,
                            "token_ids": [48],
                        },
                    )
                ],
                usage=None,
                model_extra={},
            ),
            SimpleNamespace(
                choices=[],
                usage=None,
                model_extra={
                    "sglext": {
                        "cached_tokens_details": {
                            "device": 1,
                            "host": 0,
                            "storage": 2,
                            "storage_backend": "file",
                        },
                    }
                },
            ),
            SimpleNamespace(choices=[], usage=_usage(), model_extra={}),
        ]
    )
    client, completions = _client(chunks)

    result = asyncio.run(
        client._execute_request(
            "abc",
            2,
            stream=True,
            trace_start=0.0,
            stream_acc=StreamAccumulator(),
            return_token_ids=True,
        )
    )

    assert result.text == "hi"
    assert result.prompt_token_ids == [1, 2, 3]
    assert result.generated_token_ids == [32, 48]
    assert result.cached_tokens == 1
    assert result.cached_tokens_details == {
        "device": 1,
        "host": 0,
        "storage": 2,
        "storage_backend": "file",
    }
    assert result.request_started_at_s is not None
    assert result.first_token_at_s is not None
    assert result.request_finished_at_s is not None
    assert (
        result.request_started_at_s
        <= result.first_token_at_s
        <= result.request_finished_at_s
    )
    assert completions.calls[0]["extra_body"]["return_token_ids"] is True
    assert (
        completions.calls[0]["extra_body"]["return_cached_tokens_details"]
        is True
    )


def test_stream_interval_is_forwarded_to_sglang() -> None:
    chunks = AsyncChunks(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text="hi",
                        model_extra={
                            "prompt_token_ids": [1, 2, 3],
                            "token_ids": [32, 48],
                        },
                    )
                ],
                usage=None,
                model_extra={},
            ),
            SimpleNamespace(choices=[], usage=_usage(), model_extra={}),
        ]
    )
    client, completions = _client(chunks)
    client._stream_interval = 64

    asyncio.run(
        client._execute_request(
            "abc",
            2,
            stream=True,
            trace_start=0.0,
            stream_acc=StreamAccumulator(),
            return_token_ids=True,
        )
    )

    assert completions.calls[0]["extra_body"]["stream_interval"] == 64


def test_stream_response_uses_token_ids_when_text_is_empty() -> None:
    chunks = AsyncChunks(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text="",
                        model_extra={
                            "prompt_token_ids": [1, 2, 3],
                            "token_ids": [32],
                        },
                    )
                ],
                usage=None,
                model_extra={},
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text="",
                        model_extra={"token_ids": [48]},
                    )
                ],
                usage=None,
                model_extra={},
            ),
            SimpleNamespace(choices=[], usage=_usage(), model_extra={}),
        ]
    )
    client, _ = _client(chunks)

    result = asyncio.run(
        client._execute_request(
            "abc",
            2,
            stream=True,
            trace_start=0.0,
            stream_acc=StreamAccumulator(),
            return_token_ids=True,
        )
    )

    assert result.text == ""
    assert result.generated_token_ids == [32, 48]


def test_non_replay_request_does_not_request_token_ids() -> None:
    choice = CompletionChoice(
        finish_reason="length",
        index=0,
        logprobs=None,
        text="hi",
    )
    response = SimpleNamespace(choices=[choice], usage=_usage(), model_extra={})
    client, completions = _client(response)

    asyncio.run(
        client._execute_request(
            "abc",
            2,
            stream=False,
            trace_start=0.0,
            stream_acc=StreamAccumulator(),
        )
    )

    assert completions.calls[0]["extra_body"] == {"ignore_eos": True}


def test_non_replay_stream_does_not_request_token_ids() -> None:
    chunks = AsyncChunks(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(text="hi", model_extra={})],
                usage=None,
                model_extra={},
            ),
            SimpleNamespace(choices=[], usage=_usage(), model_extra={}),
        ]
    )
    client, completions = _client(chunks)

    asyncio.run(
        client._execute_request(
            "abc",
            2,
            stream=True,
            trace_start=0.0,
            stream_acc=StreamAccumulator(),
        )
    )

    assert completions.calls[0]["extra_body"] == {"ignore_eos": True}


def test_deterministic_sampling_uses_greedy_parameters() -> None:
    choice = CompletionChoice(
        finish_reason="length",
        index=0,
        logprobs=None,
        text="hi",
        prompt_token_ids=[1, 2, 3],
        token_ids=[32, 48],
    )
    response = SimpleNamespace(choices=[choice], usage=_usage(), model_extra={})
    client, completions = _client(response)

    asyncio.run(
        client._execute_request(
            "abc",
            2,
            stream=False,
            trace_start=0.0,
            stream_acc=StreamAccumulator(),
            return_token_ids=True,
            deterministic_sampling=True,
        )
    )

    call = completions.calls[0]
    assert call["temperature"] == 0.0
    assert call["top_p"] == 1.0
    assert call["extra_body"]["top_k"] == 1


def test_forced_output_token_ids_are_forwarded() -> None:
    choice = CompletionChoice(
        finish_reason="length",
        index=0,
        logprobs=None,
        text="hi",
        prompt_token_ids=[1, 2, 3],
        token_ids=[32, 48],
    )
    response = SimpleNamespace(choices=[choice], usage=_usage(), model_extra={})
    client, completions = _client(response)

    asyncio.run(
        client._execute_request(
            "abc",
            2,
            stream=False,
            trace_start=0.0,
            stream_acc=StreamAccumulator(),
            return_token_ids=True,
            forced_output_token_ids=[32, 48],
        )
    )

    assert completions.calls[0]["extra_body"]["forced_output_token_ids"] == [32, 48]


def test_deadline_audit_request_identity_and_zero_abort_delay_are_forwarded() -> None:
    response = AsyncChunks(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text="hi",
                        model_extra={
                            "prompt_token_ids": [1, 2, 3],
                            "token_ids": [32, 48],
                        },
                    )
                ],
                usage=None,
                model_extra={},
            ),
            SimpleNamespace(choices=[], usage=_usage(), model_extra={}),
        ]
    )
    client, completions = _client(response)

    asyncio.run(
        client._execute_request(
            [1, 2, 3],
            2,
            stream=True,
            trace_start=0.0,
            stream_acc=StreamAccumulator(),
            return_token_ids=True,
            deterministic_sampling=True,
            forced_output_token_ids=[32, 48],
            rid="rid_0123456789abcdef0123456789abcdef",
            strict_replay_deadline_audit=True,
            disconnect_abort_delay_seconds=0.0,
        )
    )

    extra = completions.calls[0]["extra_body"]
    assert extra["rid"] == "rid_0123456789abcdef0123456789abcdef"
    assert extra["strict_replay_deadline_audit"] is True
    assert extra["disconnect_abort_delay_seconds"] == 0.0
    assert completions.calls[0]["temperature"] == 0.0
    assert completions.calls[0]["top_p"] == 1.0
    assert extra["top_k"] == 1


def test_deadline_audit_abort_delay_rejects_nonfinite_values() -> None:
    client, _ = _client(object())

    try:
        asyncio.run(
            client._execute_request(
                "abc",
                2,
                stream=False,
                trace_start=0.0,
                stream_acc=StreamAccumulator(),
                disconnect_abort_delay_seconds=float("nan"),
            )
        )
    except ValueError as error:
        assert "finite and nonnegative" in str(error)
    else:
        raise AssertionError("non-finite disconnect abort delay was accepted")


def test_rid_is_forwarded_to_sglang() -> None:
    choice = CompletionChoice(
        finish_reason="length",
        index=0,
        logprobs=None,
        text="hi",
        prompt_token_ids=[1, 2, 3],
        token_ids=[32, 48],
    )
    response = SimpleNamespace(choices=[choice], usage=_usage(), model_extra={})
    client, completions = _client(response)

    asyncio.run(
        client._execute_request(
            "abc",
            2,
            stream=False,
            trace_start=0.0,
            stream_acc=StreamAccumulator(),
            return_token_ids=True,
            rid="opx_join_key",
        )
    )

    assert completions.calls[0]["extra_body"]["rid"] == "opx_join_key"


def test_stream_first_token_callback_runs_once() -> None:
    chunks = AsyncChunks(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text="h",
                        model_extra={
                            "prompt_token_ids": [1, 2, 3],
                            "token_ids": [32],
                        },
                    )
                ],
                usage=None,
                model_extra={},
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text="i",
                        model_extra={"token_ids": [48]},
                    )
                ],
                usage=None,
                model_extra={},
            ),
            SimpleNamespace(choices=[], usage=_usage(), model_extra={}),
        ]
    )
    client, _ = _client(chunks)
    first_tokens: list[tuple[int, int]] = []

    asyncio.run(
        client._execute_request(
            "abc",
            2,
            stream=True,
            trace_start=0.0,
            stream_acc=StreamAccumulator(),
            return_token_ids=True,
            rid="opx_join_key",
            on_first_token=lambda ts_ns, mono_ns: first_tokens.append(
                (ts_ns, mono_ns)
            ),
        )
    )

    assert len(first_tokens) == 1
    assert all(value > 0 for value in first_tokens[0])


def test_stream_request_timing_preserves_request_and_ttft_boundaries() -> None:
    chunks = AsyncChunks(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text="h",
                        model_extra={
                            "prompt_token_ids": [1, 2, 3],
                            "token_ids": [32],
                        },
                    )
                ],
                usage=None,
                model_extra={},
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text="i",
                        model_extra={"token_ids": [48]},
                    )
                ],
                usage=None,
                model_extra={},
            ),
            SimpleNamespace(choices=[], usage=_usage(), model_extra={}),
        ]
    )
    client, completions = _client(chunks, yield_before_response=True)

    async def run_request():
        timing = RequestTiming.start()
        first_tokens: list[tuple[int, int]] = []
        result = await client._execute_request(
            "abc",
            2,
            stream=True,
            trace_start=0.0,
            stream_acc=StreamAccumulator(),
            return_token_ids=True,
            on_first_token=lambda ts_ns, mono_ns: first_tokens.append(
                (ts_ns, mono_ns)
            ),
            request_timing=timing,
        )
        return timing, first_tokens, result

    timing, first_tokens, result = asyncio.run(run_request())
    attrs = timing.as_lifecycle_attrs()

    prepare_mono_ns = attrs["generate_prepare_start_mono_ns"]
    request_start_mono_ns = attrs["request_start_mono_ns"]
    sdk_call_mono_ns = attrs["sdk_call_start_mono_ns"]
    headers_mono_ns = attrs["headers_or_first_byte_mono_ns"]
    first_token_mono_ns = attrs["first_parsed_token_mono_ns"]
    assert isinstance(prepare_mono_ns, int)
    assert isinstance(request_start_mono_ns, int)
    assert isinstance(sdk_call_mono_ns, int)
    assert isinstance(headers_mono_ns, int)
    assert isinstance(first_token_mono_ns, int)
    assert prepare_mono_ns <= request_start_mono_ns <= sdk_call_mono_ns
    assert sdk_call_mono_ns <= headers_mono_ns
    assert headers_mono_ns <= first_token_mono_ns
    assert first_tokens == [(attrs["first_parsed_token_ts_ns"], first_token_mono_ns)]
    assert attrs["sdk_call_attempt_count"] == 1
    assert isinstance(attrs["event_loop_lag_ns"], int)
    assert attrs["event_loop_lag_ns"] >= 0
    assert attrs["event_loop_lag_scope"] == "sdk_call_start_to_next_loop_turn"
    assert attrs["headers_or_first_byte_observation"] == "response_headers"
    assert (
        attrs["generate_prepare_to_request_start_ns"]
        == request_start_mono_ns - prepare_mono_ns
    )
    assert (
        attrs["request_start_to_sdk_call_ns"]
        == sdk_call_mono_ns - request_start_mono_ns
    )
    assert (
        attrs["sdk_call_to_headers_or_first_byte_ns"]
        == headers_mono_ns - sdk_call_mono_ns
    )
    assert (
        attrs["headers_or_first_byte_to_first_parsed_token_ns"]
        == first_token_mono_ns - headers_mono_ns
    )
    assert (
        attrs["sdk_call_to_first_parsed_token_ns"]
        == first_token_mono_ns - sdk_call_mono_ns
    )
    assert (
        attrs["request_start_to_first_parsed_token_ns"]
        == first_token_mono_ns - request_start_mono_ns
    )
    assert result.request_started_at_s is not None
    assert result.first_token_at_s is not None
    assert result.request_finished_at_s is not None
    assert (
        result.request_started_at_s
        <= result.first_token_at_s
        <= result.request_finished_at_s
    )
    ttft_s = result.first_token_at_s - result.request_started_at_s
    assert attrs["request_start_to_first_parsed_token_ns"] / 1e9 == pytest.approx(
        ttft_s,
        abs=0.001,
    )
    assert completions.calls == [
        {
            "model": "fake",
            "prompt": "abc",
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": {
                "ignore_eos": True,
                "return_token_ids": True,
                "return_cached_tokens_details": True,
            },
        }
    ]


def test_extension_token_ids_supports_model_extra() -> None:
    value = SimpleNamespace(model_extra={"token_ids": [7, 8]})

    assert _extension_token_ids(value, "token_ids") == [7, 8]
