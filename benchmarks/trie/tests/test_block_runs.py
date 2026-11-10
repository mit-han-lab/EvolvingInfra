import asyncio
import json
import random
from pathlib import Path

import jsonschema
import pytest
from trie.blocks import BlockTokens
from trie.client import Client, GenerationResult
from trie.types import Generate, ReplayTrace, Wait

SCHEMA = Path(__file__).parents[1] / "schema" / "replay-trace-v2.schema.json"
SPEC = {"block_size": 4, "seed": 12345, "pool": list(range(100, 164))}


def test_block_tokens_are_deterministic_per_block_and_cycle() -> None:
    a = BlockTokens(SPEC)
    t = a.tokens([[0, 3]])
    assert len(t) == 12 and all(100 <= x < 164 for x in t)
    assert BlockTokens(SPEC).tokens([[0, 3]]) == t
    # a block's content does not depend on the blocks around it
    assert a.tokens([[1, 1]]) == t[4:8]
    assert a.tokens([[2, 1], [0, 1]]) == t[8:12] + t[0:4]
    assert a.tokens([[0, 3]], 10) == t[:10]
    with pytest.raises(ValueError, match="requested"):
        a.tokens([[0, 1]], 5)
    # another replay cycle or seed gives other content
    assert BlockTokens(SPEC, replay_cycle_id=1).tokens([[0, 3]]) != t
    assert BlockTokens(dict(SPEC, seed=1)).tokens([[0, 3]]) != t


def test_block_tokens_cover_the_pool_evenly() -> None:
    spec = dict(SPEC, block_size=64, pool=list(range(1000, 1016)))
    counts = {}
    for x in BlockTokens(spec).tokens([[0, 2000]]):
        counts[x] = counts.get(x, 0) + 1
    assert sorted(counts) == list(range(1000, 1016))
    assert max(counts.values()) / min(counts.values()) < 1.1


def _row(events, **meta) -> dict:
    return {
        "schema_version": 2,
        "trace_id": "blocks-1",
        "start_offset_s": 0,
        "initial_messages": [],
        "tools": None,
        "events": events,
        "metadata": {"block_tokens": SPEC, **meta},
    }


def _gen(prompt_runs, output_runs, max_tokens) -> dict:
    return {
        "type": "generate",
        "max_tokens": max_tokens,
        "recorded_assistant": {"role": "assistant", "content": ""},
        "recorded_prompt_block_runs": prompt_runs,
        "recorded_output_block_runs": output_runs,
    }


def test_block_run_rows_validate_and_round_trip() -> None:
    row = _row([_gen([[0, 3]], [[3, 1]], 3), {"type": "wait", "seconds": 1.5, "actor": "tool"},
                _gen([[0, 5]], [[9, 2]], 6)])
    jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text())).validate(row)
    trace = ReplayTrace.from_jsonl_row(json.loads(json.dumps(row)))
    assert trace.to_jsonl_row() == row


@pytest.mark.parametrize(
    "event, meta, match",
    [
        (_gen([[0, 0]], [[3, 1]], 3), {}, "schema validation failed"),
        (_gen([[-1, 2]], [[3, 1]], 3), {}, "schema validation failed"),
        (_gen([[0, 2, 1]], [[3, 1]], 3), {}, "schema validation failed"),
        (_gen([[0, 2]], [[3, 1]], 5), {}, "fewer tokens than max_tokens"),
        (_gen([[0, 2]], [[3, 1]], 3), {"block_tokens": dict(SPEC, pool=[1])}, "pool"),
        (_gen([[0, 2]], [[3, 1]], 3), {"block_tokens": None}, "block_tokens"),
        (dict(_gen([[0, 2]], [[3, 1]], 3), recorded_prompt_token_ids=[1]), {}, "mutually exclusive"),
    ],
)
def test_invalid_block_run_rows_are_rejected(event, meta, match) -> None:
    with pytest.raises(ValueError, match=match):
        ReplayTrace.from_jsonl_row(_row([event], **meta))


class EchoClient(Client):
    """Echoes the prompt and the forced output, optionally corrupting the echoed prompt."""

    def __init__(self, corrupt_prompt: bool = False) -> None:
        self._model = "echo"
        self._max_retries = 0
        self._tokenizer_model = None
        self._tokenizer_manager = None
        self._rng = random.Random(0)
        self._result = None
        self._benchmark_start = None
        self._strict_replay_output_ids = {}
        self.corrupt_prompt = corrupt_prompt
        self.sent: list[tuple[list[int], list[int]]] = []

    async def _execute_request(self, prompt, max_tokens, **kwargs):
        forced = kwargs["forced_output_token_ids"]
        assert isinstance(prompt, list) and isinstance(forced, list) and len(forced) == max_tokens
        self.sent.append((list(prompt), list(forced)))
        echoed = list(prompt)
        if self.corrupt_prompt:
            echoed[-1] += 1
        return GenerationResult(
            text="", prompt_tokens=len(prompt), completion_tokens=len(forced), cached_tokens=0,
            prompt_token_ids=echoed, generated_token_ids=list(forced),
        )


def _chained_trace(trace_id: str = "chain", **meta) -> ReplayTrace:
    # request 2 keeps request 1's 3 blocks and continues with blocks 3..5; request 1's output is
    # the first 6 tokens of blocks 3..4, so the second prompt extends prompt 1 + output 1
    return ReplayTrace(
        trace_id=trace_id,
        metadata={"block_tokens": SPEC, **meta},
        events=[
            Generate(max_tokens=6, recorded_assistant={"role": "assistant", "content": ""},
                     recorded_prompt_block_runs=[[0, 3]], recorded_output_block_runs=[[3, 2]]),
            Wait(seconds=0.0, actor="tool"),
            Generate(max_tokens=2, recorded_assistant={"role": "assistant", "content": ""},
                     recorded_prompt_block_runs=[[0, 6]], recorded_output_block_runs=[[100, 1]]),
        ],
    )


def test_strict_replay_expands_block_runs_without_a_tokenizer() -> None:
    client = EchoClient()
    result = asyncio.run(client.run([_chained_trace()], concurrency=1, duration=5.0,
                                    replay_once=True, drain_timeout=5.0, strict_replay=True))
    assert client._tokenizer_manager is None
    assert result.failed_requests == 0
    (p1, o1), (p2, o2) = client.sent
    expand = BlockTokens(SPEC)
    assert p1 == expand.tokens([[0, 3]]) and o1 == expand.tokens([[3, 2]], 6)
    assert p2 == expand.tokens([[0, 6]]) and o2 == expand.tokens([[100, 1]], 2)
    assert p2[: len(p1) + len(o1)] == p1 + o1


def test_server_prompt_mismatch_fails_the_trace() -> None:
    client = EchoClient(corrupt_prompt=True)
    result = asyncio.run(client.run([_chained_trace()], concurrency=1, duration=5.0,
                                    replay_once=True, drain_timeout=5.0, strict_replay=True))
    assert result.failed_requests == 1
    assert len(client.sent) == 1


def test_admit_once_traces_are_not_cycled_and_cycles_get_new_content() -> None:
    client = EchoClient()
    once = _chained_trace("once", admit_once=True)
    again = _chained_trace("again")
    asyncio.run(client.run([once, again], concurrency=1, duration=0.3, drain_timeout=5.0,
                           strict_replay=True))
    first_prompts = [p for p, _ in client.sent[::2]]
    assert len(first_prompts) >= 3
    # "once" and "again" share block ids and seed: the first pass (cycle 0 of each) sends the
    # same content twice, every later admission is "again" with a new replay cycle
    cycle0 = BlockTokens(SPEC).tokens([[0, 3]])
    assert first_prompts[0] == first_prompts[1] == cycle0
    later = first_prompts[2:]
    assert all(p != cycle0 for p in later)
    assert len({tuple(p) for p in later}) == len(later)
