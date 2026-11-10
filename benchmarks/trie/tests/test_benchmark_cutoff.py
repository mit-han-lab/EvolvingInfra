"""Exercise cutoff through Client.run, including a sibling awaiting a response."""
import asyncio
import json
import time

import pytest

from trie.benchmark_boundary import BenchmarkCancellationTimeout
from trie.client import Client, GenerationResult
from trie.client_lifecycle import ClientLifecycleWriter
from trie.reporting import build_json_summary
from trie.types import Generate, ReplayTask, ReplayTrace


def workloads(schema):
    def trace(label):
        events = [Generate(max_tokens=1, recorded_prompt_token_ids=[label],
                           recorded_output_token_ids=[7])]
        if schema == 2:
            events.append(Generate(max_tokens=1, recorded_prompt_token_ids=[3],
                                   recorded_output_token_ids=[7]))
        return ReplayTrace(str(label), events)
    return [trace(i) if schema == 2 else ReplayTask(str(i), [trace(i)], [])
            for i in (1, 2)]


class CrossingClient(Client):
    def __init__(self, writer):
        super().__init__(endpoint="http://127.0.0.1:1/v1", model="fake",
                         api_key="unused", max_retries=0, lifecycle_writer=writer)
        self.sibling_started = asyncio.Event()
        self.sibling_cancelled = False
        self.sibling_finished = False
        self.calls = []
        self.tasks = set()

    async def _run_workload(self, *args, **kwargs):
        self.tasks.add(asyncio.current_task())
        return await super()._run_workload(*args, **kwargs)

    async def _execute_request(self, prompt, max_tokens, **kwargs):
        self.calls.append(prompt[0])
        if prompt[0] == 2:
            self.sibling_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.sibling_cancelled = True
                raise
            finally:
                self.sibling_finished = True
        await self.sibling_started.wait()
        # A synchronous crossing ensures the workload's guard wins the race
        # against the parent asyncio.wait timeout, as in the real failure.
        deadline = self._benchmark_boundary.deadline["mono_ns"]
        time.sleep(max(0, (deadline - time.monotonic_ns()) / 1e9) + .002)
        return GenerationResult(text="", prompt_tokens=len(prompt), completion_tokens=1,
                                cached_tokens=0, prompt_token_ids=list(prompt),
                                generated_token_ids=[7])


@pytest.mark.parametrize("schema", [2, 3])
def test_deadline_guard_finishes_and_joins_other_workloads(tmp_path, monkeypatch, schema):
    timing = tmp_path / "timing.json"
    monkeypatch.setenv("TRIE_BENCHMARK_TIMING_FILE", str(timing))
    monkeypatch.setenv("TRIE_BENCHMARK_CANCEL_TIMEOUT_S", "1")
    writer = ClientLifecycleWriter(output_path=tmp_path / "events.jsonl", run_id="cutoff",
                                   cell_id="c", arm="test", concurrency=2)

    async def scenario():
        client = CrossingClient(writer)
        try:
            result = await client.run(workloads(schema), concurrency=2, duration=.03,
                                      drain_timeout=0, strict_replay=True,
                                      cache_salt_mode="global")
            assert client.calls == [1, 2]  # no next request or closed-loop refill
            assert client.sibling_cancelled and client.sibling_finished
            assert all(task.done() for task in client.tasks)
            assert result.completed_requests == 0
            # The actual CLI serializer must be reachable with the returned result.
            (tmp_path / "results.json").write_text(json.dumps(build_json_summary(result)))
        finally:
            for task in client.tasks:
                task.cancel()
            await asyncio.gather(*client.tasks, return_exceptions=True)
            await client._client.close()
    asyncio.run(scenario())
    data = json.loads(timing.read_text())
    assert data["valid"] and data["status"] == "completed"
    assert data["admission_stop"] and data["request_admission_stop"] and data["cancel_start"]
    assert data["counters"]["pending_tasks_after_cancel"] == 0
    assert data["counters"]["cancel_requested_tasks"] >= 1
    assert data["last_admission_monotonic_ns"] < data["benchmark_deadline_monotonic_ns"]
    assert data["counters"]["boundary_guard_stops"] >= 1
    assert json.loads((tmp_path / "results.json").read_text())["completed_requests"] == 0


@pytest.mark.parametrize("at_deadline", [False, True])
def test_plain_workload_cancellation_is_not_a_success(tmp_path, monkeypatch, at_deadline):
    timing = tmp_path / "timing.json"
    monkeypatch.setenv("TRIE_BENCHMARK_TIMING_FILE", str(timing))

    class CancelledClient(Client):
        async def _run_replay_task(self, *args, **kwargs):
            if at_deadline:
                deadline = self._benchmark_boundary.deadline["mono_ns"]
                time.sleep(max(0, (deadline - time.monotonic_ns()) / 1e9) + .002)
            raise asyncio.CancelledError("independent failure")

    async def scenario():
        client = CancelledClient(endpoint="http://127.0.0.1:1/v1", model="fake",
                                 api_key="unused", max_retries=0)
        try:
            with pytest.raises(asyncio.CancelledError, match="independent failure"):
                await client.run(workloads(3)[:1], concurrency=1, duration=.03,
                                 drain_timeout=0, strict_replay=True,
                                 cache_salt_mode="global")
        finally:
            await client._client.close()
    asyncio.run(scenario())
    data = json.loads(timing.read_text())
    assert data["status"] == "failed" and not data["valid"]


@pytest.mark.parametrize("external_cancel", [False, True])
def test_other_cancellation_failures_stay_failed(tmp_path, monkeypatch, external_cancel):
    timing = tmp_path / "timing.json"
    monkeypatch.setenv("TRIE_BENCHMARK_TIMING_FILE", str(timing))
    monkeypatch.setenv("TRIE_BENCHMARK_CANCEL_TIMEOUT_S", ".02")

    class StuckClient(Client):
        async def _run_replay_task(self, *args, **kwargs):
            self.child = asyncio.current_task()
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                await self.release.wait()

    async def scenario():
        client = StuckClient(endpoint="http://127.0.0.1:1/v1", model="fake",
                             api_key="unused", max_retries=0)
        client.started, client.release = asyncio.Event(), asyncio.Event()
        parent = asyncio.create_task(client.run(workloads(3)[:1], concurrency=1,
            duration=1 if external_cancel else .02, drain_timeout=0,
            strict_replay=True, cache_salt_mode="global"))
        try:
            await client.started.wait()
            if external_cancel:
                parent.cancel()
            expected = asyncio.CancelledError if external_cancel else BenchmarkCancellationTimeout
            with pytest.raises(expected):
                await parent
        finally:
            client.release.set()
            await asyncio.gather(client.child, return_exceptions=True)
            await client._client.close()
    asyncio.run(scenario())
    data = json.loads(timing.read_text())
    assert data["status"] == "failed" and not data["valid"]
    if not external_cancel:
        assert data["counters"]["cancel_cleanup_timeouts"] == 1


def test_admitted_request_stopped_before_send_keeps_cancellation_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("TRIE_BENCHMARK_TIMING_FILE", str(tmp_path / "timing.json"))
    writer = ClientLifecycleWriter(output_path=tmp_path / "events.jsonl", run_id="cutoff",
                                   cell_id="c", arm="test", concurrency=1)

    class LateSendClient(Client):
        async def _execute_request(self, *args, **kwargs):
            deadline = self._benchmark_boundary.deadline["mono_ns"]
            time.sleep(max(0, (deadline - time.monotonic_ns()) / 1e9) + .002)
            self._check_benchmark_admission("http_send")
            pytest.fail("HTTP must not be sent after the fixed cutoff")

    async def scenario():
        client = LateSendClient(endpoint="http://127.0.0.1:1/v1", model="fake",
                                api_key="unused", max_retries=0, lifecycle_writer=writer)
        try:
            await client.run(workloads(3)[:1], concurrency=1, duration=.02,
                             strict_replay=True, drain_timeout=0, cache_salt_mode="global")
        finally:
            await client._client.close()
    asyncio.run(scenario())
    rows = [json.loads(line) for line in writer.output_path.read_text().splitlines()]
    failed = [r for r in rows if r["event"] == "REQUEST_FAILED"]
    assert len(failed) == 1 and failed[0]["attrs"]["error_class"] == "CancelledError"
    timing = json.loads((tmp_path / "timing.json").read_text())
    assert timing["valid"] and failed[0]["mono_ns"] >= timing["benchmark_deadline_monotonic_ns"]
    assert timing["counters"]["lifecycle_admission_count"] == 1
    assert timing["counters"].get("http_send_count", 0) == 0


def test_independent_request_cancellation_after_deadline_is_not_boundary_success(tmp_path):
    writer = ClientLifecycleWriter(output_path=tmp_path / "events.jsonl", run_id="cutoff",
                                   cell_id="c", arm="test", concurrency=1)

    class CancelledRequestClient(Client):
        async def _execute_request(self, *args, **kwargs):
            deadline = self._benchmark_boundary.deadline["mono_ns"]
            time.sleep(max(0, (deadline - time.monotonic_ns()) / 1e9) + .002)
            raise asyncio.CancelledError("independent upstream cancellation")

    async def scenario():
        client = CancelledRequestClient(endpoint="http://127.0.0.1:1/v1", model="fake",
            api_key="unused", max_retries=0, lifecycle_writer=writer)
        try:
            result = await client.run(workloads(3)[:1], concurrency=1, duration=.02,
                drain_timeout=0, strict_replay=True, cache_salt_mode="global")
            assert result.failed_requests == 1
        finally:
            await client._client.close()
    asyncio.run(scenario())
    rows = [json.loads(line) for line in writer.output_path.read_text().splitlines()]
    assert sum(r["event"] == "TASK_FAILED" for r in rows) == 1
    assert not any(r["event"] == "TASK_COMPLETE" for r in rows)
