"""Use the real replay/admission/DAG loops; only model execution is stubbed."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from test_multiagent_replay import edge, run
from trie.client import Client, GenerationResult
from trie.client_lifecycle import ClientLifecycleWriter
from trie.tokenizer_manager import TokenizerManager
from trie.types import Generate, ReplayTask, ReplayTrace, ReplaceContext, token_ids_sha256


def tokenizer():
    manager = object.__new__(TokenizerManager)
    # Sparse IDs and special IDs ensure sampling is from the real vocabulary.
    manager._tokenizer = SimpleNamespace(
        all_special_ids=[0, 10000],
        get_vocab=lambda: {**{str(n): n for n in range(100, 400)}, "eos": 0, "bos": 10000},
    )
    return manager


def generate(ids, t=0):
    return Generate(2, recorded_prompt_token_ids=ids,
                    recorded_prompt_token_count=len(ids),
                    recorded_prompt_token_ids_sha256=token_ids_sha256(ids),
                    recorded_output_token_ids=[7, 8], source_t_s=t)


class CaptureClient(Client):
    def __init__(self, writer=None, tamper=False, mode="forced"):
        super().__init__(endpoint="http://127.0.0.1:1/v1", model="fake",
                         api_key="test", max_retries=0, seed=0, lifecycle_writer=writer,
                         replay_output_mode=mode)
        self._tokenizer_manager = tokenizer()
        self.calls = []
        self.tamper = tamper

    async def _execute_request(self, prompt, max_tokens, **kwargs):
        self.calls.append((list(prompt), max_tokens, kwargs))
        await asyncio.sleep(0)
        returned = list(prompt)
        if self.tamper:
            returned[0] += 1
        output = kwargs["forced_output_token_ids"]
        if output is None:
            output = [9] * max_tokens
        return GenerationResult("", len(prompt), max_tokens, 0,
                                prompt_token_ids=returned, generated_token_ids=output)


@pytest.mark.parametrize("mode", ["forced", "model"])
def test_repeated_instances_keep_turn_prefix_and_original_contract(mode, tmp_path):
    async def scenario():
        writer = ClientLifecycleWriter(tmp_path / "events.jsonl", "run", "cell", "off", 2)
        client = CaptureClient(writer, mode=mode)
        first = [1, 2, 3]
        second = first + [7, 8, 4]
        trace = ReplayTrace("edgebench", [generate(first), generate(second)])
        before = copy.deepcopy(trace.to_jsonl_row())
        result = await run(client, [trace, trace], concurrency=2, trajectory_prefix_tokens=10)
        assert result.failed_requests == 0
        assert result.completed_model_requests == 4
        prefixes = {tuple(prompt[:10]) for prompt, _, _ in client.calls}
        assert len(prefixes) == 2
        for prefix in prefixes:
            prompts = [p for p, _, _ in client.calls if tuple(p[:10]) == prefix]
            assert prompts == [list(prefix) + first, list(prefix) + second]
            assert prompts[1][:len(prompts[0])] == prompts[0]
        assert trace.to_jsonl_row() == before
        assert all(n == 2 for _, n, _ in client.calls)
        assert all(k["forced_output_token_ids"] is None for _, _, k in client.calls) == (mode == "model")
        completed = [json.loads(s) for s in writer.output_path.read_text().splitlines()
                     if json.loads(s)["event"] == "CLIENT_COMPLETE"]
        for row in completed:
            attrs = row["attrs"]
            assert attrs["prompt_tokens"] == attrs["source_prompt_tokens"] + 10
            assert attrs["trajectory_prefix_tokens"] == 10
        assert len({r["attrs"]["trajectory_prefix_sha256"] for r in completed}) == 2

    asyncio.run(scenario())


def test_task_children_and_compaction_share_prefix_but_next_instance_does_not():
    async def scenario():
        client = CaptureClient()
        task = ReplayTask("roadmap", [
            ReplayTrace("parent", [generate([1, 2]), ReplaceContext([]), generate([5])]),
            ReplayTrace("child", [generate([1, 2, 3])]),
        ], [edge("parent", 0, "child", 0), edge("child", 0, "parent", 2)])
        result = await run(client, [task, task], trajectory_prefix_tokens=10)
        assert result.completed_requests == 2 and result.failed_requests == 0
        prompts = [p for p, _, _ in client.calls]
        for group in (prompts[:3], prompts[3:]):
            assert len({tuple(p[:10]) for p in group}) == 1
            assert [p[10:] for p in group] == [[1, 2], [1, 2, 3], [5]]
        assert prompts[0][:10] != prompts[3][:10]
    asyncio.run(scenario())


def test_stationary_warmup_uses_the_measured_requests_prefix():
    async def scenario():
        client = CaptureClient()
        trace = ReplayTrace("edgebench", [generate([1], t=0), generate([1, 7, 8, 2], t=10)])
        result = await run(client, [trace], trajectory_prefix_tokens=10, warm_start="stationary", delay_scale=0)
        assert result.failed_requests == 0 and result.completed_model_requests == 1
        assert [n for _, n, _ in client.calls] == [1, 2]
        assert client.calls[0][0][:10] == client.calls[1][0][:10]
        assert client.calls[0][0][10:] == [1]
    asyncio.run(scenario())


def test_source_hash_failure_still_prevents_send():
    async def scenario():
        client = CaptureClient()
        event = generate([1, 2])
        event.recorded_prompt_token_ids[0] = 3  # corrupt after schema validation
        result = await run(client, [ReplayTrace("bad", [event])], trajectory_prefix_tokens=10)
        assert result.failed_requests == 1
        assert client.calls == []
    asyncio.run(scenario())


def test_changed_wire_prefix_is_rejected():
    async def scenario():
        client = CaptureClient(tamper=True)
        result = await run(client, [ReplayTrace("bad", [generate([1])])], trajectory_prefix_tokens=10)
        assert result.failed_requests == 1
        assert result.completed_model_requests == 0
    asyncio.run(scenario())


def test_corrupted_source_warmup_is_not_sent():
    async def scenario():
        client = CaptureClient()
        warmup = generate([1], t=0)
        warmup.recorded_prompt_token_ids[0] = 3
        trace = ReplayTrace("bad", [warmup, generate([1, 7, 8, 2], t=10)])
        result = await run(client, [trace], trajectory_prefix_tokens=10,
                           warm_start="stationary", delay_scale=0)
        assert result.failed_requests == 1
        assert client.calls == []
    asyncio.run(scenario())


def test_disabled_is_exact_and_does_not_load_tokenizer():
    async def scenario():
        client = CaptureClient()
        client._tokenizer_manager = None
        result = await run(client, [ReplayTrace("edgebench", [generate([1, 2])])])
        assert result.failed_requests == 0
        assert client.calls[0][0] == [1, 2]
        assert client._tokenizer_manager is None
    asyncio.run(scenario())


def test_sampler_uses_ordinary_ids_without_advancing_workload_rng():
    manager = tokenizer()
    samples = [manager.random_prefix_tokens(10) for _ in range(100)]
    assert len(set(samples)) == 100
    assert all(len(s) == 10 and all(100 <= x < 400 for x in s) for s in samples)
    # The prefix sampler works without TokenizerManager._rng altogether.
    assert not hasattr(manager, "_rng")


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_bad_prefix_length_rejected(value):
    async def scenario():
        client = CaptureClient()
        with pytest.raises(ValueError, match="nonnegative integer"):
            await run(client, [ReplayTrace("t", [generate([1])])], trajectory_prefix_tokens=value)
    asyncio.run(scenario())


def test_prefix_requires_token_authoritative_replay():
    async def scenario():
        client = CaptureClient()
        with pytest.raises(ValueError, match="requires strict_replay"):
            await run(client, [ReplayTrace("t", [generate([1])])], trajectory_prefix_tokens=10, strict_replay=False)
    asyncio.run(scenario())
