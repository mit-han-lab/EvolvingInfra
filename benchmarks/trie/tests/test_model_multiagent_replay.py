"""Regression checks for real generation and parallel external-tool waits."""

import asyncio

import pytest
from test_multiagent_replay import ControlledClient, edge, generate, run
from trie.client import GenerationResult
from trie.types import Generate, ReplayTask, ReplayTrace, Wait


def test_model_output_keeps_lengths_without_inventing_or_forcing_tokens():
    class ModelClient(ControlledClient):
        async def _execute_request(self, prompt, max_tokens, **kwargs):
            assert kwargs["forced_output_token_ids"] is None
            assert kwargs["deterministic_sampling"] is True
            assert max_tokens == 3
            self.calls.append(prompt)
            return GenerationResult(
                "",
                len(prompt),
                3,
                0,
                prompt_token_ids=prompt,
                generated_token_ids=[9, 8, 7],
            )

    async def scenario():
        client = ModelClient()
        client._replay_output_mode = "model"
        child = ReplayTrace("child", [Generate(3, recorded_prompt_token_ids=[1, 2])])
        result = await run(client, [ReplayTask("t", [child], [])])
        assert result.completed_requests == 1
        assert result.completed_model_requests == 1
        assert result.failed_requests == 0
        assert client._strict_replay_output_ids == {}

    asyncio.run(scenario())


def test_external_wait_overlaps_child_instead_of_running_after_join():
    async def scenario():
        client = ControlledClient(blocked=[3])
        task = ReplayTask(
            "t",
            [
                ReplayTrace("main", [generate(1), generate(2)]),
                ReplayTrace("child", [generate(3)]),
                ReplayTrace(
                    "tool", [Wait(0.04, actor="tool")], metadata={"control_only": True}
                ),
            ],
            [
                edge("main", 0, "child", 0),
                edge("main", 0, "tool", 0),
                edge("child", 0, "main", 1),
                edge("tool", 0, "main", 1),
            ],
        )
        task = ReplayTask.from_jsonl_row(task.to_jsonl_row())
        execution = asyncio.create_task(run(client, [task]))
        await asyncio.wait_for(client.started[3].wait(), 1)
        await asyncio.sleep(0.07)
        assert not client.started[2].is_set()
        client.release[3].set()
        await asyncio.wait_for(client.started[2].wait(), 0.03)
        result = await execution
        assert result.completed_model_requests == 3
        assert result.completed_requests == 1

    asyncio.run(scenario())


def test_control_lane_cannot_hide_model_calls():
    with pytest.raises(ValueError, match="only wait"):
        ReplayTrace("bad", [generate(1)], metadata={"control_only": True})
    with pytest.raises(ValueError, match="generate event"):
        ReplayTask(
            "bad", [ReplayTrace("tool", [Wait(1)], metadata={"control_only": True})], []
        )
