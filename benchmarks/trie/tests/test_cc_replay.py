import asyncio
import json

import pytest

from trie.blocks import BlockTokens
from trie.client import Client, GenerationResult
from trie.types import ReplayTask

SPEC = {"block_size": 2, "seed": 7, "pool": list(range(100, 164))}


def _generate(prompt_runs, output_runs, max_tokens, t, api=1.0):
    return {
        "type": "generate",
        "max_tokens": max_tokens,
        "recorded_prompt_block_runs": prompt_runs,
        "recorded_output_block_runs": output_runs,
        "source_t_s": t,
        "source_api_time_s": api,
    }


def _session(scope="task"):
    """main turns at t=0 and t=10; a child spawned after turn 0 that main joins before turn 1."""
    return ReplayTask.from_jsonl_row(
        {
            "schema_version": 3,
            "trace_id": "session",
            "cache_salt_scope": scope,
            "metadata": {"block_tokens": SPEC},
            "traces": [
                {"schema_version": 2, "trace_id": "main",
                 "events": [_generate([[0, 1]], [[10, 1]], 1, t=0.0), _generate([[0, 2]], [[11, 1]], 2, t=10.0)]},
                {"schema_version": 2, "trace_id": "child",
                 "events": [_generate([[2, 1]], [[12, 1]], 1, t=2.0), _generate([[2, 2]], [[13, 1]], 1, t=4.0)]},
            ],
            "dependencies": [
                {"before": {"trace_id": "main", "event_index": 0}, "after": {"trace_id": "child", "event_index": 0}},
                {"before": {"trace_id": "child", "event_index": 1}, "after": {"trace_id": "main", "event_index": 1}},
            ],
        }
    )


class RecordingClient(Client):
    def __init__(self, seed=0):
        super().__init__(endpoint="http://127.0.0.1:1/v1", model="fake", api_key="test", max_retries=0, seed=seed)
        self.calls = []

    async def _execute_request(self, prompt, max_tokens, **kwargs):
        output = list(kwargs["forced_output_token_ids"])
        self.calls.append((list(prompt), output, kwargs["cache_salt"]))
        await asyncio.sleep(0.01)
        return GenerationResult(text="", prompt_tokens=len(prompt), completion_tokens=len(output),
                                cached_tokens=0, prompt_token_ids=list(prompt), generated_token_ids=output)


def _replay(client, task, **kwargs):
    return asyncio.run(client.run([task], concurrency=1, duration=30, stream=True, replay_once=True,
                                  drain_timeout=30, strict_replay=True, delay_scale=0.0, **kwargs))


def test_task_block_tokens_serve_every_agent_with_one_salt() -> None:
    client = RecordingClient()
    result = _replay(client, _session())

    blocks = BlockTokens(SPEC)  # the task's first replay cycle
    assert result.completed_model_requests == 4
    assert [(prompt, output) for prompt, output, _ in client.calls] == [
        (blocks.tokens([[0, 1]]), blocks.tokens([[10, 1]], 1)),
        (blocks.tokens([[2, 1]]), blocks.tokens([[12, 1]], 1)),
        (blocks.tokens([[2, 2]]), blocks.tokens([[13, 1]], 1)),
        (blocks.tokens([[0, 2]]), blocks.tokens([[11, 1]], 2)),
    ]
    assert len({salt for _, _, salt in client.calls}) == 1

    agent_salts = RecordingClient()
    _replay(agent_salts, _session(scope="agent"))
    assert len({salt for _, _, salt in agent_salts.calls}) == 2


def test_stationary_cut_warms_the_last_turn_and_skips_finished_agents() -> None:
    client = RecordingClient()
    # Random(0).random() = 0.844 of the 0..11 s span: only main's last turn is left,
    # the child ended before the cut, and main's first turn is re-sent for one token.
    result = _replay(client, _session(), warm_start="stationary")

    blocks = BlockTokens(SPEC)
    assert [(prompt, output) for prompt, output, _ in client.calls] == [
        (blocks.tokens([[0, 1]]), blocks.tokens([[10, 1]], 1)),
        (blocks.tokens([[0, 2]]), blocks.tokens([[11, 1]], 2)),
    ]
    assert result.completed_model_requests == 1


def test_initial_prefill_readiness_follows_successful_warmup(tmp_path, monkeypatch):
    monkeypatch.setenv("TRIE_INITIAL_PREFILL_READY_DIR", str(tmp_path))
    observed = []

    class ObservedClient(RecordingClient):
        async def _execute_request(self, prompt, max_tokens, **kwargs):
            observed.append(json.loads((tmp_path / "shard-0.json").read_text())["ready"])
            return await super()._execute_request(prompt, max_tokens, **kwargs)

    client = ObservedClient()
    result = _replay(client, _session(), warm_start="stationary")
    # The skipped child's turns create no warmup; main's one-token warmup
    # must finish and pass exact-token checks before its normal request.
    assert observed == [False, True]
    assert result.completed_model_requests == 1
    row = json.loads((tmp_path / "shard-0.json").read_text())
    assert row["ready_slots"] == row["expected_slots"] == 1
    assert row["pending_prefills"] == 0 and not row["failed_slots"]
    assert row["benchmark_start_ns"] <= row["initial_prefill_ready_ns"] < row["benchmark_deadline_ns"]


def test_failed_initial_prefill_never_enables_controller(tmp_path, monkeypatch):
    monkeypatch.setenv("TRIE_INITIAL_PREFILL_READY_DIR", str(tmp_path))

    class FailedWarmup(RecordingClient):
        async def _execute_request(self, prompt, max_tokens, **kwargs):
            raise ValueError("warmup token mismatch")

    result = _replay(FailedWarmup(), _session(), warm_start="stationary")
    row = json.loads((tmp_path / "shard-0.json").read_text())
    assert result.failed_requests == 1
    assert row["ready"] is False
    assert row["failed_slots"] == {"0": "ValueError"}


def test_task_row_round_trips_with_agents_inheriting_its_block_tokens() -> None:
    row = _session().to_jsonl_row()
    again = ReplayTask.from_jsonl_row(json.loads(json.dumps(row)))

    assert again.to_jsonl_row() == row
    assert "block_tokens" not in row["traces"][0]["metadata"]
    assert again.traces[1].block_spec == SPEC


def test_client_shards_partition_the_sessions() -> None:
    sessions = [ReplayTask.from_jsonl_row({**_session().to_jsonl_row(), "trace_id": f"session-{i}"}) for i in range(5)]
    admitted = []
    for shard in range(2):
        client = RecordingClient()
        original = client._run_workload

        async def record(workload, **kwargs):
            admitted.append((shard, workload.trace_id))
            await original(workload, **kwargs)

        client._run_workload = record
        asyncio.run(client.run(sessions, concurrency=1, duration=30, stream=True, replay_once=True, drain_timeout=30,
                               strict_replay=True, delay_scale=0.0, shard_index=shard, shard_count=2))
    assert sorted(trace for _, trace in admitted) == [f"session-{i}" for i in range(5)]
    assert {trace for shard, trace in admitted if shard == 0} == {"session-0", "session-2", "session-4"}


def test_a_stream_cut_mid_response_fails_the_task_not_the_client() -> None:
    import httpx

    class CutClient(RecordingClient):
        async def _execute_request(self, prompt, max_tokens, **kwargs):
            raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")

    result = _replay(CutClient(), _session())
    assert result.failed_requests == 1 and result.completed_requests == 0


@pytest.mark.parametrize("lazy_workload", [False, True])
def test_shuffled_client_shards_are_disjoint_with_different_seeds(tmp_path, lazy_workload) -> None:
    rows = [{**_session().to_jsonl_row(), "trace_id": f"session-{i}"} for i in range(12)]
    workload_path = tmp_path / "sessions.jsonl"
    workload_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    shards = []
    for shard in range(4):
        client = RecordingClient(seed=shard)
        admitted = []
        original = client._run_workload

        async def record(workload, **kwargs):
            admitted.append(workload.trace_id)
            await original(workload, **kwargs)

        client._run_workload = record
        result = asyncio.run(client.run(
            str(workload_path), concurrency=1, duration=30, stream=True,
            replay_once=True, drain_timeout=30, strict_replay=True,
            delay_scale=0.0, workload_order="shuffle", lazy_workload=lazy_workload,
            shard_index=shard, shard_count=4,
        ))
        assert result.failed_requests == 0
        assert len(admitted) == len(set(admitted))
        shards.append(set(admitted))
    assert set.union(*shards) == {row["trace_id"] for row in rows}
    assert sum(map(len, shards)) == len(set.union(*shards))
    for shard, admitted in enumerate(shards):
        assert admitted == {rows[i]["trace_id"] for i in range(shard, len(rows), 4)}
