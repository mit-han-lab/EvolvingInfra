import asyncio
import json
import random
import time
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError
from openai.types.completion_usage import CompletionUsage
from trie.client import Client, GenerationResult
from trie.client_lifecycle import ClientLifecycleWriter
from trie.results import BenchmarkResult
from trie.types import (
    AppendMessage,
    Generate,
    ReplaceContext,
    ReplayTrace,
    Wait,
    token_ids_sha256,
)


class SimpleTokenizer:
    def render_chat(
        self,
        messages,
        *,
        tools=None,
        add_generation_prompt=True,
        reasoning_effort=None,
    ):
        rendered = "".join(
            (
                f"<{message['role']}>{message.get('content') or ''}"
                + (
                    json.dumps(message["tool_calls"], sort_keys=True)
                    if message.get("tool_calls")
                    else ""
                )
            )
            for message in messages
        )
        return rendered + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text: str) -> list[int]:
        return list(text.encode())

    def decode(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode()

    def eos_token_id(self) -> int:
        return 0


class FakeClient(Client):
    def __init__(self, outputs: list[str | tuple[str, list[int]]]) -> None:
        self._model = "fake"
        self._max_retries = 0
        self._rng = random.Random(0)
        self._result = None
        self._benchmark_start = None
        self._tokenizer_manager = SimpleTokenizer()
        self._strict_replay_output_ids = {}
        self._outputs = iter(outputs)
        self.prompts: list[str | list[int]] = []
        self.cache_salts: list[str | None] = []
        self.routing_keys: list[str | None] = []
        self.request_ids: list[str | None] = []
        self.max_tokens: list[int] = []
        self.forced_output_token_ids: list[list[int] | None] = []

    async def _execute_request(self, prompt, max_tokens, **kwargs):
        self.prompts.append(prompt)
        self.cache_salts.append(kwargs.get("cache_salt"))
        self.routing_keys.append(kwargs.get("routing_key"))
        self.request_ids.append(kwargs.get("rid"))
        self.max_tokens.append(max_tokens)
        request_timing = kwargs.get("request_timing")
        if request_timing is not None:
            request_timing.mark_request_start()
            scheduled_mono_ns = request_timing.mark_sdk_call_start()
            asyncio.get_running_loop().call_soon(
                request_timing.observe_event_loop_turn,
                scheduled_mono_ns,
            )
            await asyncio.sleep(0)
            request_timing.mark_headers_or_first_byte()
        on_first_token = kwargs.get("on_first_token")
        if on_first_token is not None:
            first_token_clock = (
                request_timing.mark_first_parsed_token()
                if request_timing is not None
                else (time.time_ns(), time.monotonic_ns())
            )
            on_first_token(*first_token_clock)
        forced_ids = kwargs.get("forced_output_token_ids")
        self.forced_output_token_ids.append(
            list(forced_ids) if forced_ids is not None else None
        )
        output = next(self._outputs)
        if isinstance(output, tuple):
            text, generated_token_ids = output
        else:
            text = output
            generated_token_ids = list(text.encode())
        prompt_token_ids = (
            list(prompt)
            if isinstance(prompt, list)
            else self._tokenizer_manager.encode(prompt)
        )
        return GenerationResult(
            text=text,
            prompt_tokens=len(prompt_token_ids),
            completion_tokens=len(text),
            cached_tokens=0,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
        )


class TokenOnlyClient(Client):
    def __init__(self) -> None:
        self._model = "served-model-alias"
        self._max_retries = 0
        self._tokenizer_model = None
        self._tokenizer_manager = None
        self._rng = random.Random(0)
        self._result = None
        self._benchmark_start = None
        self._strict_replay_output_ids = {}
        self.prompts: list[list[int]] = []

    async def _execute_request(self, prompt, max_tokens, **kwargs):
        assert isinstance(prompt, list)
        forced_ids = kwargs["forced_output_token_ids"]
        assert isinstance(forced_ids, list)
        self.prompts.append(prompt)
        return GenerationResult(
            text="",
            prompt_tokens=len(prompt),
            completion_tokens=len(forced_ids),
            cached_tokens=0,
            prompt_token_ids=list(prompt),
            generated_token_ids=list(forced_ids),
        )


def test_client_defers_tokenizer_initialization() -> None:
    client = Client(
        endpoint="http://127.0.0.1:1/v1",
        model="served-model-alias",
        api_key="test",
        max_retries=0,
    )

    assert client._tokenizer_manager is None
    assert client._tokenizer_model == "served-model-alias"
    asyncio.run(client._client.close())


def _run_context_source(
    context_source: str,
    *,
    prefix_block_size: int = 16,
) -> tuple[FakeClient, BenchmarkResult]:
    client = FakeClient(["live-one", "live-two"])
    result = BenchmarkResult()
    trace = ReplayTrace(
        trace_id="trace-1",
        initial_messages=[{"role": "user", "content": "hello"}],
        events=[
            Generate(
                max_tokens=8,
                recorded_assistant={"role": "assistant", "content": "recorded-one"},
            ),
            AppendMessage(message={"role": "user", "content": "continue"}),
            Generate(
                max_tokens=8,
                recorded_assistant={"role": "assistant", "content": "recorded-two"},
            ),
        ],
    )
    asyncio.run(
        client._run_replay_trace(
            trace,
            benchmark_start=0.0,
            result=result,
            stream=False,
            refresh=lambda: None,
            delay_scale=0.0,
            max_user_delay=None,
            max_tool_delay=None,
            context_source=context_source,
            prefix_block_size=prefix_block_size,
            cache_salt_mode="session",
            run_cache_salt="run-1",
        )
    )
    return client, result


def test_generated_context_uses_live_output() -> None:
    client, _ = _run_context_source("generated")
    assert "live-one" in client.prompts[1]
    assert "recorded-one" not in client.prompts[1]


def test_generated_context_does_not_duplicate_recorded_tool_calls() -> None:
    client = FakeClient(["abcdefghij", "done"])
    result = BenchmarkResult()
    tool_calls = [
        {
            "id": "tool-1",
            "type": "function",
            "function": {"name": "Read", "arguments": {"path": "a.py"}},
        }
    ]
    trace = ReplayTrace(
        trace_id="trace-tools",
        initial_messages=[{"role": "user", "content": "inspect"}],
        events=[
            Generate(
                max_tokens=10,
                recorded_assistant={
                    "role": "assistant",
                    "content": "ok",
                    "tool_calls": tool_calls,
                },
            ),
            AppendMessage(
                message={
                    "role": "tool",
                    "tool_call_id": "tool-1",
                    "content": "result",
                }
            ),
            Generate(max_tokens=4),
        ],
    )

    asyncio.run(
        client._run_replay_trace(
            trace,
            benchmark_start=0.0,
            result=result,
            stream=False,
            refresh=lambda: None,
            delay_scale=0.0,
            max_user_delay=None,
            max_tool_delay=None,
            context_source="generated",
            prefix_block_size=1,
            cache_salt_mode="session",
            run_cache_salt="run-1",
        )
    )

    assert "<assistant>ab" in client.prompts[1]
    assert "abcdefghij" not in client.prompts[1]
    assert '"id": "tool-1"' in client.prompts[1]
    first_prompt_tokens = len(client.prompts[0].encode())
    assert result.client_prefix_tokens == [0, first_prompt_tokens + len("ab")]


def test_generated_tool_only_turn_reuses_only_the_previous_prompt() -> None:
    client = FakeClient(["generated-but-discarded", "done"])
    result = BenchmarkResult()
    tool_calls = [
        {
            "id": "tool-1",
            "type": "function",
            "function": {"name": "Bash", "arguments": {"command": "true"}},
        }
    ]
    trace = ReplayTrace(
        trace_id="trace-tool-only",
        initial_messages=[{"role": "user", "content": "run it"}],
        events=[
            Generate(
                max_tokens=23,
                recorded_assistant={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                },
            ),
            AppendMessage(
                message={
                    "role": "tool",
                    "tool_call_id": "tool-1",
                    "content": "ok",
                }
            ),
            Generate(max_tokens=4),
        ],
    )

    asyncio.run(
        client._run_replay_trace(
            trace,
            benchmark_start=0.0,
            result=result,
            stream=False,
            refresh=lambda: None,
            delay_scale=0.0,
            max_user_delay=None,
            max_tool_delay=None,
            context_source="generated",
            prefix_block_size=1,
            cache_salt_mode="session",
            run_cache_salt="run-1",
        )
    )

    assert "generated-but-discarded" not in client.prompts[1]
    assert '"id": "tool-1"' in client.prompts[1]
    assert result.client_prefix_tokens == [0, len(client.prompts[0].encode())]


def test_recorded_context_uses_recorded_output() -> None:
    client, _ = _run_context_source("recorded")
    assert "recorded-one" in client.prompts[1]
    assert "live-one" not in client.prompts[1]


def test_strict_replay_sends_canonical_fixed_length_token_ids() -> None:
    expected_ids = list(b"abc") + [0]
    client = FakeClient([("abc", expected_ids)])
    result = BenchmarkResult()
    trace = ReplayTrace(
        trace_id="strict-canonical",
        initial_messages=[{"role": "user", "content": "hello"}],
        events=[
            Generate(
                max_tokens=4,
                recorded_assistant={"role": "assistant", "content": "abc"},
            )
        ],
    )

    asyncio.run(
        client._run_replay_trace(
            trace,
            benchmark_start=0.0,
            result=result,
            stream=False,
            refresh=lambda: None,
            delay_scale=0.0,
            max_user_delay=None,
            max_tool_delay=None,
            context_source="recorded",
            prefix_block_size=1,
            cache_salt_mode="session",
            run_cache_salt="run-1",
            strict_replay=True,
        )
    )

    assert client.max_tokens == [4]
    assert client.forced_output_token_ids == [expected_ids]
    assert result.completed_requests == 1
    assert result.failed_requests == 0


def test_strict_replay_prefers_recorded_output_token_ids() -> None:
    expected_ids = [501, 502]
    client = FakeClient([("ignored", expected_ids)])
    result = BenchmarkResult()
    trace = ReplayTrace(
        trace_id="strict-exact",
        initial_messages=[{"role": "user", "content": "hello"}],
        events=[
            Generate(
                max_tokens=2,
                recorded_assistant={"role": "assistant", "content": "different"},
                recorded_output_token_ids=expected_ids,
            )
        ],
    )

    asyncio.run(
        client._run_replay_trace(
            trace,
            benchmark_start=0.0,
            result=result,
            stream=False,
            refresh=lambda: None,
            delay_scale=0.0,
            max_user_delay=None,
            max_tool_delay=None,
            context_source="recorded",
            prefix_block_size=1,
            cache_salt_mode="session",
            run_cache_salt="run-1",
            strict_replay=True,
        )
    )

    assert client.forced_output_token_ids == [expected_ids]
    assert result.completed_requests == 1


def test_strict_replay_uses_and_validates_recorded_prompt_token_ids() -> None:
    prompt_ids = [10, 20, 30]
    output_ids = [501, 502]
    client = FakeClient([("ignored", output_ids)])
    result = BenchmarkResult()
    trace = ReplayTrace(
        trace_id="strict-exact-prompt",
        metadata={"routing_key": "session-route-1"},
        events=[
            Generate(
                max_tokens=2,
                recorded_assistant={"role": "assistant", "content": "different"},
                recorded_prompt_token_ids=prompt_ids,
                recorded_output_token_ids=output_ids,
            )
        ],
    )

    asyncio.run(
        client._run_replay_trace(
            trace,
            benchmark_start=0.0,
            result=result,
            stream=False,
            refresh=lambda: None,
            delay_scale=0.0,
            max_user_delay=None,
            max_tool_delay=None,
            context_source="recorded",
            prefix_block_size=1,
            cache_salt_mode="session",
            run_cache_salt="run-1",
            strict_replay=True,
        )
    )

    assert client.prompts == [prompt_ids]
    assert client.routing_keys == ["session-route-1"]
    assert result.completed_requests == 1
    assert result.failed_requests == 0


def test_strict_replay_materializes_and_validates_compact_prompt_contract() -> None:
    tokenizer = SimpleTokenizer()
    rendered = tokenizer.render_chat(
        [{"role": "user", "content": "hello"}],
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(rendered)
    output_ids = [501, 502]
    client = FakeClient([("ignored", output_ids)])
    result = BenchmarkResult()
    trace = ReplayTrace(
        trace_id="strict-compact-prompt",
        initial_messages=[{"role": "user", "content": "hello"}],
        events=[
            Generate(
                max_tokens=2,
                recorded_assistant={"role": "assistant", "content": "different"},
                recorded_prompt_token_count=len(prompt_ids),
                recorded_prompt_token_ids_sha256=token_ids_sha256(prompt_ids),
                recorded_output_token_ids=output_ids,
            )
        ],
    )

    asyncio.run(
        client._run_replay_trace(
            trace,
            benchmark_start=0.0,
            result=result,
            stream=False,
            refresh=lambda: None,
            delay_scale=0.0,
            max_user_delay=None,
            max_tool_delay=None,
            context_source="recorded",
            prefix_block_size=1,
            cache_salt_mode="session",
            run_cache_salt="run-1",
            strict_replay=True,
        )
    )
    assert client.prompts == [prompt_ids]
    assert result.completed_requests == 1
    assert result.failed_requests == 0

    trace.events[0].recorded_prompt_token_ids_sha256 = "0" * 64
    failed_client = FakeClient([("ignored", output_ids)])
    failed_result = BenchmarkResult()
    asyncio.run(
        failed_client._run_replay_trace(
            trace,
            benchmark_start=0.0,
            result=failed_result,
            stream=False,
            refresh=lambda: None,
            delay_scale=0.0,
            max_user_delay=None,
            max_tool_delay=None,
            context_source="recorded",
            prefix_block_size=1,
            cache_salt_mode="session",
            run_cache_salt="run-1",
            strict_replay=True,
        )
    )
    assert failed_client.prompts == []
    assert failed_result.failed_requests == 1


def test_compact_prompt_contract_skips_eager_prompt_materialization() -> None:
    client = TokenOnlyClient()
    output_ids = [30, 40]
    trace = ReplayTrace(
        trace_id="strict-compact-prepare",
        events=[
            Generate(
                max_tokens=2,
                recorded_assistant={"role": "assistant", "content": "kept"},
                recorded_prompt_token_count=2,
                recorded_prompt_token_ids_sha256=token_ids_sha256([10, 20]),
                recorded_output_token_ids=output_ids,
            )
        ],
    )
    client._prepare_strict_replay_output_ids([trace])
    assert client._strict_replay_output_ids[id(trace.events[0])] == output_ids

    missing_context = ReplayTrace(
        trace_id="strict-compact-missing-context",
        events=[
            Generate(
                max_tokens=2,
                recorded_prompt_token_count=2,
                recorded_prompt_token_ids_sha256=token_ids_sha256([10, 20]),
                recorded_output_token_ids=output_ids,
            )
        ],
    )
    with pytest.raises(ValueError, match="requires recorded_assistant"):
        client._prepare_strict_replay_output_ids([missing_context])


def test_fully_token_authoritative_trace_needs_no_tokenizer_or_assistant() -> None:
    client = TokenOnlyClient()
    trace = ReplayTrace(
        trace_id="strict-token-only",
        events=[
            Generate(
                max_tokens=2,
                recorded_prompt_token_ids=[10, 20],
                recorded_output_token_ids=[30, 40],
            ),
            AppendMessage(message={"role": "tool", "content": "not rendered"}),
            Generate(
                max_tokens=1,
                recorded_prompt_token_ids=[10, 20, 30, 40, 50],
                recorded_output_token_ids=[60],
            ),
        ],
    )

    result = asyncio.run(
        client.run(
            [trace],
            concurrency=1,
            duration=1.0,
            replay_once=True,
            drain_timeout=1.0,
            strict_replay=True,
        )
    )

    assert client._tokenizer_manager is None
    assert client.prompts == [[10, 20], [10, 20, 30, 40, 50]]
    assert result.completed_requests == 1
    assert result.failed_requests == 0


def test_text_fallback_lazily_loads_model_tokenizer(monkeypatch) -> None:
    client = TokenOnlyClient()
    client._tokenizer_model = "model-tokenizer"
    client._seed = 7
    loaded: list[tuple[str, int | None]] = []

    def load_tokenizer(model: str, seed: int | None = None) -> SimpleTokenizer:
        loaded.append((model, seed))
        return SimpleTokenizer()

    monkeypatch.setattr("trie.tokenizer_manager.TokenizerManager", load_tokenizer)
    trace = ReplayTrace(
        trace_id="strict-text-fallback",
        events=[
            Generate(
                max_tokens=1,
                recorded_assistant={"role": "assistant", "content": "x"},
                recorded_output_token_ids=[1],
            )
        ],
    )

    client._prepare_strict_replay_output_ids([trace])

    assert isinstance(client._tokenizer_manager, SimpleTokenizer)
    assert loaded == [("model-tokenizer", 7)]


def test_strict_replay_rejects_prompt_token_id_mismatch() -> None:
    class MismatchedPromptClient(FakeClient):
        async def _execute_request(self, prompt, max_tokens, **kwargs):
            result = await super()._execute_request(prompt, max_tokens, **kwargs)
            result.prompt_token_ids = [999]
            return result

    client = MismatchedPromptClient([("ignored", [501])])
    result = BenchmarkResult()
    trace = ReplayTrace(
        trace_id="strict-bad-prompt",
        events=[
            Generate(
                max_tokens=1,
                recorded_assistant={"role": "assistant", "content": "different"},
                recorded_prompt_token_ids=[10, 20],
                recorded_output_token_ids=[501],
            )
        ],
    )

    asyncio.run(
        client._run_replay_trace(
            trace,
            benchmark_start=0.0,
            result=result,
            stream=False,
            refresh=lambda: None,
            delay_scale=0.0,
            max_user_delay=None,
            max_tool_delay=None,
            context_source="recorded",
            prefix_block_size=1,
            cache_salt_mode="session",
            run_cache_salt="run-1",
            strict_replay=True,
        )
    )

    assert result.completed_requests == 0
    assert result.failed_requests == 1


def test_strict_replay_rejects_missing_context_before_http_request() -> None:
    client = FakeClient([])
    trace = ReplayTrace(
        trace_id="strict-missing-context",
        events=[
            Generate(
                max_tokens=1,
                recorded_output_token_ids=[1],
            ),
            ReplaceContext(messages=[{"role": "user", "content": "reset"}]),
            Generate(
                max_tokens=1,
                recorded_prompt_token_ids=[2],
                recorded_output_token_ids=[3],
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="requires recorded_assistant unless both prompt and output",
    ):
        asyncio.run(
            client.run(
                [trace],
                concurrency=1,
                duration=1.0,
                strict_replay=True,
            )
        )

    assert client.prompts == []


def test_event_delay_applies_actor_caps_before_scaling() -> None:
    assert (
        Client._event_delay(
            Wait(seconds=10.0, actor="user"),
            delay_scale=0.5,
            max_user_delay=4.0,
            max_tool_delay=None,
        )
        == 2.0
    )
    assert (
        Client._event_delay(
            Wait(seconds=10.0, actor="tool"),
            delay_scale=0.25,
            max_user_delay=None,
            max_tool_delay=6.0,
        )
        == 1.5
    )
    assert (
        Client._event_delay(
            Wait(seconds=10.0, actor="other"),
            delay_scale=0.1,
            max_user_delay=1.0,
            max_tool_delay=1.0,
        )
        == 1.0
    )


def test_replay_trace_executes_scaled_start_and_event_waits(monkeypatch) -> None:
    sleeps: list[float] = []
    clock = [100.0]

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    monkeypatch.setattr("trie.client.time.perf_counter", lambda: clock[0])
    client = FakeClient(["done"])
    result = BenchmarkResult()
    trace = ReplayTrace(
        trace_id="trace-waits",
        start_offset_s=4.0,
        events=[
            Wait(seconds=10.0, actor="user"),
            Wait(seconds=8.0, actor="tool"),
            Wait(seconds=3.0, actor="other"),
            Generate(max_tokens=4),
        ],
    )

    asyncio.run(
        client._run_replay_trace(
            trace,
            benchmark_start=100.0,
            result=result,
            stream=False,
            refresh=lambda: None,
            delay_scale=0.5,
            max_user_delay=2.0,
            max_tool_delay=4.0,
            context_source="generated",
            prefix_block_size=16,
            cache_salt_mode="session",
            run_cache_salt="run-1",
        )
    )

    assert sleeps == [2.0, 1.0, 2.0, 1.5]
    assert result.completed_requests == 1
    assert result.latencies == [4.5]


def test_prefix_uses_exact_generated_token_ids() -> None:
    client, result = _run_context_source("generated", prefix_block_size=1)

    previous_sequence_tokens = len(client.prompts[0].encode()) + len("live-one")
    assert result.client_prefix_tokens == [0, previous_sequence_tokens]


def test_prefix_does_not_reencode_generated_text() -> None:
    client = FakeClient([("live-one", [999]), "live-two"])
    result = BenchmarkResult()
    trace = ReplayTrace(
        trace_id="trace-exact-ids",
        initial_messages=[{"role": "user", "content": "hello"}],
        events=[Generate(max_tokens=8), Generate(max_tokens=8)],
    )

    asyncio.run(
        client._run_replay_trace(
            trace,
            benchmark_start=0.0,
            result=result,
            stream=False,
            refresh=lambda: None,
            delay_scale=0.0,
            max_user_delay=None,
            max_tool_delay=None,
            context_source="generated",
            prefix_block_size=1,
            cache_salt_mode="session",
            run_cache_salt="run-1",
        )
    )

    assert result.client_prefix_tokens == [0, len(client.prompts[0].encode())]


def test_eligible_prefix_is_block_aligned() -> None:
    client, result = _run_context_source("generated", prefix_block_size=16)

    previous_sequence_tokens = len(client.prompts[0].encode()) + len("live-one")
    expected = previous_sequence_tokens - (previous_sequence_tokens % 16)
    assert result.block_aligned_prefix_tokens == [0, expected]


def test_eligible_prefix_excludes_uncached_terminal_token() -> None:
    initial_messages = [{"role": "user", "content": "hello"}]
    initial_prompt = SimpleTokenizer().render_chat(initial_messages)
    first_output = "x" * (32 - len(initial_prompt.encode()))
    client = FakeClient([first_output, "done"])
    result = BenchmarkResult()
    trace = ReplayTrace(
        trace_id="trace-block-boundary",
        initial_messages=initial_messages,
        events=[Generate(max_tokens=len(first_output)), Generate(max_tokens=4)],
    )

    asyncio.run(
        client._run_replay_trace(
            trace,
            benchmark_start=0.0,
            result=result,
            stream=False,
            refresh=lambda: None,
            delay_scale=0.0,
            max_user_delay=None,
            max_tool_delay=None,
            context_source="generated",
            prefix_block_size=16,
            cache_salt_mode="session",
            run_cache_salt="run-1",
        )
    )

    assert result.client_prefix_tokens == [0, 32]
    assert result.block_aligned_prefix_tokens == [0, 16]


def test_replace_context_drops_pre_compaction_history() -> None:
    client = FakeClient(["live-before-compact", "live-after-compact"])
    result = BenchmarkResult()
    trace = ReplayTrace(
        trace_id="trace-compact",
        initial_messages=[{"role": "user", "content": "old history"}],
        events=[
            Generate(max_tokens=8),
            ReplaceContext(
                messages=[{"role": "user", "content": "compact summary"}],
                reason="compaction",
            ),
            Generate(max_tokens=8),
        ],
    )

    asyncio.run(
        client._run_replay_trace(
            trace,
            benchmark_start=0.0,
            result=result,
            stream=False,
            refresh=lambda: None,
            delay_scale=0.0,
            max_user_delay=None,
            max_tool_delay=None,
            context_source="generated",
            prefix_block_size=16,
            cache_salt_mode="session",
            run_cache_salt="run-1",
        )
    )

    assert "compact summary" in client.prompts[1]
    assert "old history" not in client.prompts[1]
    assert "live-before-compact" not in client.prompts[1]
    assert result.block_aligned_prefix_tokens == [0, 0]


def test_session_cache_salt_is_stable_within_trace() -> None:
    client, _ = _run_context_source("generated")

    assert client.cache_salts[0] == client.cache_salts[1]
    assert client.cache_salts[0] is not None
    assert client.cache_salts[0].startswith("run-1:trace-1:")


def test_session_cache_salt_changes_between_runs() -> None:
    client = FakeClient(["one", "two"])
    trace = ReplayTrace(trace_id="same-trace", events=[Generate(max_tokens=1)])

    for _ in range(2):
        asyncio.run(
            client.run(
                [trace],
                concurrency=1,
                duration=1.0,
                replay_once=True,
                drain_timeout=1.0,
                cache_salt_mode="session",
            )
        )

    assert client.cache_salts[0] != client.cache_salts[1]
    assert all(salt and ":same-trace:" in salt for salt in client.cache_salts)


def test_session_cache_salt_changes_between_trace_invocations() -> None:
    client = FakeClient(["one", "two"])
    trace = ReplayTrace(trace_id="same-trace", events=[Generate(max_tokens=1)])

    asyncio.run(
        client.run(
            [trace, trace],
            concurrency=1,
            duration=1.0,
            replay_once=True,
            drain_timeout=1.0,
            cache_salt_mode="session",
        )
    )

    assert len(client.cache_salts) == 2
    assert client.cache_salts[0] != client.cache_salts[1]


def test_global_cache_salt_remains_none() -> None:
    client = FakeClient(["one"])
    trace = ReplayTrace(trace_id="trace-global", events=[Generate(max_tokens=1)])

    asyncio.run(
        client.run(
            [trace],
            concurrency=1,
            duration=1.0,
            replay_once=True,
            drain_timeout=1.0,
            cache_salt_mode="global",
        )
    )

    assert client.cache_salts == [None]


class SchedulerClient(Client):
    def __init__(self) -> None:
        self._model = "fake"
        self._max_retries = 0
        self._tokenizer_model = None
        self._tokenizer_manager = SimpleTokenizer()
        self._rng = random.Random(0)
        self._result = None
        self._benchmark_start = None
        self._strict_replay_output_ids = {}

    async def _run_workload(self, workload, *, result, **kwargs) -> None:
        await asyncio.sleep(0.01)
        result.record_trace_success(0.01)


class SlowSchedulerClient(SchedulerClient):
    async def _run_workload(self, workload, *, result, **kwargs) -> None:
        await asyncio.sleep(0.2)
        result.record_trace_success(0.2)


class RequestConcurrencyClient(FakeClient):
    def __init__(self, outputs: list[str]) -> None:
        super().__init__(outputs)
        self.in_flight = 0
        self.max_in_flight = 0

    async def _execute_request(self, prompt, max_tokens, **kwargs):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            return await super()._execute_request(prompt, max_tokens, **kwargs)
        finally:
            self.in_flight -= 1


def test_strict_replay_rejects_transport_retries() -> None:
    client = FakeClient([])
    client._max_retries = 1
    trace = ReplayTrace(
        trace_id="strict-retries",
        events=[
            Generate(
                max_tokens=1,
                recorded_assistant={"role": "assistant", "content": "x"},
                recorded_output_token_ids=[1],
            )
        ],
    )

    with pytest.raises(ValueError, match="strict_replay requires max_retries=0"):
        asyncio.run(
            client.run(
                [trace],
                concurrency=1,
                strict_replay=True,
            )
        )


class FakeCompletions:
    def __init__(self, failures: int, failure_kind: str = "connect") -> None:
        self.failures = failures
        self.failure_kind = failure_kind
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            request = httpx.Request("POST", "http://127.0.0.1:1/v1/completions")
            error_type = (
                httpx.ConnectError
                if self.failure_kind == "connect"
                else httpx.ReadError
            )
            try:
                raise error_type("transport failure", request=request)
            except (httpx.ConnectError, httpx.ReadError) as cause:
                raise APIConnectionError(request=request) from cause

        usage = CompletionUsage(
            completion_tokens=1,
            prompt_tokens=1,
            total_tokens=2,
        )
        choice = SimpleNamespace(text="", prompt_token_ids=[3], token_ids=[7])
        if not kwargs.get("stream"):
            return SimpleNamespace(choices=[choice], usage=usage)

        async def response_stream():
            yield SimpleNamespace(choices=[choice], usage=None)
            yield SimpleNamespace(choices=[], usage=usage)

        return response_stream()


class ConnectFailureClient(Client):
    def __init__(self, failures: int, failure_kind: str = "connect") -> None:
        self._model = "fake"
        self._max_retries = 0
        self._client = SimpleNamespace(
            completions=FakeCompletions(failures, failure_kind=failure_kind)
        )
        self._tokenizer_model = None
        self._tokenizer_manager = None
        self._reasoning_effort = None
        self._seed = 0
        self._rng = random.Random(0)
        self._result = None
        self._benchmark_start = None
        self._strict_replay_output_ids = {}
        self._strict_replay_connect_retries = 5
        self._strict_replay_connect_retry_delay_s = 0.0
        self.execute_calls = 0

    async def _execute_request(self, prompt, max_tokens, **kwargs):
        self.execute_calls += 1
        return await super()._execute_request(prompt, max_tokens, **kwargs)


def _run_connect_retry_trace(client: ConnectFailureClient) -> BenchmarkResult:
    result = BenchmarkResult()
    trace = ReplayTrace(
        trace_id="strict-connect-retry",
        initial_messages=[],
        events=[
            Generate(
                max_tokens=1,
                recorded_prompt_token_ids=[3],
                recorded_prompt_token_count=1,
                recorded_prompt_token_ids_sha256=token_ids_sha256([3]),
                recorded_output_token_ids=[7],
                recorded_assistant={"role": "assistant", "content": ""},
            )
        ],
    )
    asyncio.run(
        client._run_replay_trace(
            trace,
            benchmark_start=0.0,
            result=result,
            stream=True,
            refresh=lambda: None,
            delay_scale=1.0,
            max_user_delay=None,
            max_tool_delay=None,
            context_source="recorded",
            prefix_block_size=16,
            cache_salt_mode="global",
            run_cache_salt=None,
            strict_replay=True,
        )
    )
    return result


def test_strict_replay_retries_five_predispatch_connect_failures() -> None:
    client = ConnectFailureClient(failures=5)
    result = _run_connect_retry_trace(client)

    assert client.execute_calls == 1
    assert client._client.completions.calls == 6
    assert result.failed_requests == 0


def test_strict_replay_stops_after_five_connect_retries() -> None:
    client = ConnectFailureClient(failures=6)
    result = _run_connect_retry_trace(client)

    assert client.execute_calls == 1
    assert client._client.completions.calls == 6
    assert result.failed_requests == 1


def test_strict_replay_does_not_retry_ambiguous_read_failure() -> None:
    client = ConnectFailureClient(failures=1, failure_kind="read")
    result = _run_connect_retry_trace(client)

    assert client.execute_calls == 1
    assert client._client.completions.calls == 1
    assert result.failed_requests == 1


def test_request_concurrency_admits_one_shot_timeline_together() -> None:
    traces = [
        ReplayTrace(trace_id=f"trace-{index}", events=[Generate(max_tokens=1)])
        for index in range(4)
    ]
    client = RequestConcurrencyClient(["a", "b", "c", "d"])

    result = asyncio.run(
        client.run(
            traces,
            request_concurrency=2,
            duration=1.0,
            replay_once=True,
            drain_timeout=1.0,
        )
    )

    assert result.completed_requests == len(traces)
    assert result.expected_requests == len(traces)
    assert client.max_in_flight == 2


def test_replay_once_drains_after_all_traces_are_admitted() -> None:
    client = SchedulerClient()
    traces = [
        ReplayTrace(trace_id=f"trace-{index}", events=[Generate(max_tokens=1)])
        for index in range(2)
    ]

    result = asyncio.run(
        client.run(
            traces,
            concurrency=4,
            duration=1.0,
            replay_once=True,
            drain_timeout=1.0,
        )
    )

    assert result.completed_requests == len(traces)
    assert result.expected_requests == len(traces)


def test_replay_once_refills_concurrency_until_all_traces_run() -> None:
    client = SchedulerClient()
    traces = [
        ReplayTrace(trace_id=f"trace-{index}", events=[Generate(max_tokens=1)])
        for index in range(3)
    ]

    result = asyncio.run(
        client.run(
            traces,
            concurrency=1,
            duration=1.0,
            replay_once=True,
            drain_timeout=1.0,
        )
    )

    assert result.completed_requests == len(traces)
    assert result.expected_requests == len(traces)


def test_replay_once_zero_drain_timeout_cancels_active_traces() -> None:
    client = SchedulerClient()
    trace = ReplayTrace(trace_id="trace", events=[Generate(max_tokens=1)])

    result = asyncio.run(
        client.run(
            [trace],
            concurrency=1,
            duration=1.0,
            replay_once=True,
            drain_timeout=0.0,
        )
    )

    assert result.completed_requests == 0


def test_arrival_rate_replay_once_does_not_wait_for_extra_arrival() -> None:
    client = SchedulerClient()
    trace = ReplayTrace(trace_id="trace", events=[Generate(max_tokens=1)])

    result = asyncio.run(
        client.run(
            [trace],
            arrival_rate=2.0,
            duration=1.0,
            replay_once=True,
            drain_timeout=1.0,
        )
    )

    assert result.completed_requests == 1
    assert result.wall_time < 0.25


def test_duration_cancels_full_concurrency_wait() -> None:
    client = SlowSchedulerClient()
    trace = ReplayTrace(trace_id="trace", events=[Generate(max_tokens=1)])

    result = asyncio.run(
        client.run(
            [trace],
            concurrency=1,
            duration=0.05,
        )
    )

    assert result.completed_requests == 0
    assert result.wall_time < 0.15


def test_duration_does_not_refill_after_wait_crosses_deadline(monkeypatch) -> None:
    client = SchedulerClient()
    trace = ReplayTrace(trace_id="trace", events=[Generate(max_tokens=1)])
    original_wait = asyncio.wait
    original_create_task = asyncio.create_task
    workload_tasks_created = 0

    async def wait_across_deadline(*args, **kwargs):
        done, pending = await original_wait(*args, **kwargs)
        if done:
            await asyncio.sleep(0.03)
        return done, pending

    def track_create_task(coro, *args, **kwargs):
        nonlocal workload_tasks_created
        code = getattr(coro, "cr_code", None)
        if code is not None and code.co_name == "_run_workload":
            workload_tasks_created += 1
        return original_create_task(coro, *args, **kwargs)

    monkeypatch.setattr(asyncio, "wait", wait_across_deadline)
    monkeypatch.setattr(asyncio, "create_task", track_create_task)

    result = asyncio.run(
        client.run(
            [trace],
            concurrency=1,
            duration=0.02,
        )
    )

    assert result.completed_requests == 1
    assert workload_tasks_created == 1


def test_guard_cancellation_racing_with_deadline_wait_finishes_normally(monkeypatch, tmp_path) -> None:
    class DeadlineClient(SchedulerClient):
        calls = 0
        async def _run_workload(self, workload, **kwargs):
            self.calls += 1
            await asyncio.sleep(0.03)
            self._check_benchmark_admission("task_complete")

    timing = tmp_path / "timing.json"
    monkeypatch.setenv("TRIE_BENCHMARK_TIMING_FILE", str(timing))
    original_wait = asyncio.wait
    async def delayed_wait(tasks, **kwargs):
        # Reproduce a scheduler turn that sees the workload's guard cancellation
        # in done, instead of seeing an empty timeout set just before it.
        kwargs["timeout"] = None
        return await original_wait(tasks, **kwargs)
    monkeypatch.setattr(asyncio, "wait", delayed_wait)
    client = DeadlineClient()
    trace = ReplayTrace(trace_id="deadline", events=[Generate(max_tokens=1)])
    result = asyncio.run(client.run([trace], concurrency=1, duration=0.02, drain_timeout=0))
    timing_data = json.loads(timing.read_text())
    assert result.completed_requests == 0
    assert client.calls == 1  # No closed-loop refill after the boundary.
    assert timing_data["status"] == "completed"
    assert timing_data["counters"]["deadline_completed_task_cancellations"] == 1


@pytest.mark.parametrize("early,kind,drain_timeout", [
    (True, "cancel", 0.0), (False, "error", 0.0), (False, "cancel", 1.0)])
def test_unexpected_task_failures_are_not_hidden_by_cutoff(monkeypatch, early, kind, drain_timeout):
    class FailedClient(SchedulerClient):
        async def _run_workload(self, workload, **kwargs):
            await asyncio.sleep(0 if early else 0.03)
            if kind == "error":
                raise RuntimeError("real workload failure")
            raise asyncio.CancelledError("unexpected cancellation")

    original_wait = asyncio.wait
    async def delayed_wait(tasks, **kwargs):
        kwargs["timeout"] = None
        return await original_wait(tasks, **kwargs)
    monkeypatch.setattr(asyncio, "wait", delayed_wait)
    client = FailedClient()
    trace = ReplayTrace(trace_id="failed", events=[Generate(max_tokens=1)])
    expected = RuntimeError if kind == "error" else asyncio.CancelledError
    with pytest.raises(expected):
        asyncio.run(client.run([trace], concurrency=1, duration=0.02, drain_timeout=drain_timeout))


def test_lifecycle_ids_are_unique_join_keys_and_do_not_leak_content(tmp_path) -> None:
    trace_id = "raw-private-trace-id"
    routing_key = "raw-private-routing-key"
    prompt_text = "raw-private-prompt"
    output_text = "raw-private-output"
    lifecycle_path = tmp_path / "private" / "client.jsonl"
    writer = ClientLifecycleWriter(
        output_path=lifecycle_path,
        run_id="run-1",
        cell_id="01-c24-a0",
        arm="off",
        concurrency=1,
    )
    client = FakeClient([output_text, output_text])
    client._lifecycle_writer = writer
    trace = ReplayTrace(
        trace_id=trace_id,
        initial_messages=[{"role": "user", "content": prompt_text}],
        metadata={"routing_key": routing_key},
        events=[Generate(max_tokens=len(output_text))],
    )

    asyncio.run(
        client.run(
            [trace, trace],
            concurrency=1,
            duration=1.0,
            replay_once=True,
            drain_timeout=1.0,
            stream=True,
            cache_salt_mode="global",
        )
    )

    assert len(client.request_ids) == 2
    assert all(
        request_id and request_id.startswith("opx_")
        for request_id in client.request_ids
    )
    assert len(set(client.request_ids)) == 2
    rows = [
        json.loads(line)
        for line in lifecycle_path.read_text(encoding="utf-8").splitlines()
    ]
    request_rows = [
        row
        for row in rows
        if row["event"] in {"CLIENT_ADMIT", "CLIENT_FIRST_TOKEN", "CLIENT_COMPLETE"}
    ]
    assert {row["request"]["rid"] for row in request_rows} == set(client.request_ids)
    assert {row["request"]["cycle_id"] for row in request_rows} == {0, 1}
    for request_id in client.request_ids:
        assert [
            row["event"] for row in request_rows if row["request"]["rid"] == request_id
        ] == ["CLIENT_ADMIT", "CLIENT_FIRST_TOKEN", "CLIENT_COMPLETE"]
        first_token_row = next(
            row
            for row in request_rows
            if row["request"]["rid"] == request_id
            and row["event"] == "CLIENT_FIRST_TOKEN"
        )
        complete_row = next(
            row
            for row in request_rows
            if row["request"]["rid"] == request_id
            and row["event"] == "CLIENT_COMPLETE"
        )
        timing = complete_row["attrs"]["client_timing"]
        assert (
            timing["generate_prepare_start_mono_ns"]
            <= timing["request_start_mono_ns"]
        )
        assert timing["request_start_mono_ns"] <= timing["sdk_call_start_mono_ns"]
        assert (
            timing["sdk_call_start_mono_ns"]
            <= timing["headers_or_first_byte_mono_ns"]
        )
        assert (
            timing["headers_or_first_byte_mono_ns"]
            <= timing["first_parsed_token_mono_ns"]
        )
        assert timing["first_parsed_token_ts_ns"] == first_token_row["ts_ns"]
        assert timing["first_parsed_token_mono_ns"] == first_token_row["mono_ns"]
        assert timing["sdk_call_attempt_count"] == 1
        assert isinstance(timing["event_loop_lag_ns"], int)
        assert timing["event_loop_lag_ns"] >= 0
    serialized = lifecycle_path.read_text(encoding="utf-8")
    for private_value in (trace_id, routing_key, prompt_text, output_text):
        assert private_value not in serialized
    assert lifecycle_path.stat().st_mode & 0o777 == 0o600
    assert lifecycle_path.parent.stat().st_mode & 0o777 == 0o700
