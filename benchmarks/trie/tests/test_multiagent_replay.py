import asyncio
import copy
import json
import runpy
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from trie.client import Client, GenerationResult, load_workloads
from trie.client_lifecycle import ClientLifecycleWriter
from trie.reporting import build_json_summary
from trie.types import Dependency, EventRef, Generate, ReplayTask, ReplayTrace, Wait


def generate(label):
    return Generate(
        max_tokens=1, recorded_prompt_token_ids=[label], recorded_output_token_ids=[7]
    )


def edge(before, before_index, after, after_index):
    return Dependency(EventRef(before, before_index), EventRef(after, after_index))


def fork_join(name="task", offset=0):
    return ReplayTask(
        trace_id=name,
        traces=[
            ReplayTrace("main", [generate(offset + 1), generate(offset + 2)]),
            ReplayTrace("child-a", [generate(offset + 3)]),
            ReplayTrace("child-b", [generate(offset + 4)]),
        ],
        dependencies=[
            edge("main", 0, "child-a", 0),
            edge("main", 0, "child-b", 0),
            edge("child-a", 0, "main", 1),
            edge("child-b", 0, "main", 1),
        ],
    )


class ControlledClient(Client):
    """Use the real replay loop; gates control when model responses complete."""

    def __init__(self, *, blocked=(), failing=(), cancelled=(), writer=None):
        super().__init__(
            endpoint="http://127.0.0.1:1/v1",
            model="fake",
            api_key="test",
            max_retries=0,
            lifecycle_writer=writer,
        )
        self.started = defaultdict(asyncio.Event)
        self.finished = defaultdict(asyncio.Event)
        self.release = {label: asyncio.Event() for label in blocked}
        self.failing = set(failing)
        self.cancelled = set(cancelled)
        self.calls = []
        self.active = 0
        self.peak = 0

    async def _execute_request(self, prompt, max_tokens, **kwargs):
        label = prompt[0]
        self.calls.append(("start", label, kwargs.get("rid"), time.perf_counter()))
        self.active += 1
        self.peak = max(self.active, self.peak)
        self.started[label].set()
        try:
            if label in self.release:
                await self.release[label].wait()
            await asyncio.sleep(0)
            if label in self.failing:
                raise ValueError("injected model failure")
            if label in self.cancelled:
                raise asyncio.CancelledError()
            self.calls.append(("finish", label, kwargs.get("rid"), time.perf_counter()))
            self.finished[label].set()
            return GenerationResult(
                text="",
                prompt_tokens=len(prompt),
                completion_tokens=max_tokens,
                cached_tokens=0,
                prompt_token_ids=list(prompt),
                generated_token_ids=list(kwargs["forced_output_token_ids"]),
            )
        finally:
            self.active -= 1


async def run(client, tasks, **kwargs):
    options = dict(
        concurrency=1,
        duration=3,
        drain_timeout=2,
        replay_once=True,
        strict_replay=True,
        cache_salt_mode="global",
    )
    options.update(kwargs)
    try:
        return await asyncio.wait_for(client.run(tasks, **options), timeout=5)
    finally:
        await client._client.close()


def test_parallel_fork_waits_for_slow_child_and_records_one_task(tmp_path):
    async def scenario():
        writer = ClientLifecycleWriter(
            output_path=tmp_path / "events.jsonl",
            run_id="test",
            cell_id="c",
            arm="off",
            concurrency=1,
        )
        client = ControlledClient(blocked=[3], writer=writer)
        execution = asyncio.create_task(run(client, [fork_join()]))
        await asyncio.wait_for(client.finished[4].wait(), timeout=1)
        assert client.started[3].is_set()  # both children execute in parallel
        assert client.finished[1].is_set()  # children did not precede dispatch
        assert not client.started[2].is_set()  # join waits for the slower child
        await asyncio.sleep(0.03)
        client.release[3].set()
        result = await execution
        assert result.completed_requests == 1
        assert result.completed_model_requests == 4
        assert result.failed_requests == 0
        assert result.latencies[0] >= 0.03
        assert client.peak == 2
        events = [
            json.loads(line) for line in writer.output_path.read_text().splitlines()
        ]
        assert sum(e["event"] == "TASK_COMPLETE" for e in events) == 1
        assert sum(e["event"] == "CLIENT_COMPLETE" for e in events) == 4
        starts = [e for e in events if e["event"] == "CLIENT_ADMIT"]
        assert len({e["request"]["experiment_request_id"] for e in starts}) == 4
        assert build_json_summary(result)["completed_requests"] == 1

    asyncio.run(scenario())


def test_waiting_parent_does_not_hold_request_concurrency_slot():
    async def scenario():
        client = ControlledClient()
        result = await run(client, [fork_join()], request_concurrency=1)
        assert result.completed_requests == 1
        assert result.completed_model_requests == 4
        assert client.peak == 1

    asyncio.run(scenario())


def test_task_concurrency_counts_entire_groups():
    async def scenario():
        client = ControlledClient(blocked=[3])
        execution = asyncio.create_task(
            run(client, [fork_join(), fork_join("other", 10)])
        )
        await asyncio.wait_for(client.finished[4].wait(), timeout=1)
        assert not client.started[11].is_set()
        client.release[3].set()
        result = await execution
        assert result.completed_requests == 2
        order = [(kind, label) for kind, label, *_ in client.calls]
        assert order.index(("finish", 2)) < order.index(("start", 11))

    asyncio.run(scenario())


def test_two_tasks_do_not_share_completion_signals_or_request_ids(tmp_path):
    async def scenario():
        writer = ClientLifecycleWriter(
            output_path=tmp_path / "events.jsonl",
            run_id="r",
            cell_id="c",
            arm="off",
            concurrency=2,
        )
        client = ControlledClient(blocked=[13], writer=writer)
        execution = asyncio.create_task(
            run(client, [fork_join(), fork_join("other", 10)], concurrency=2)
        )
        await asyncio.wait_for(client.finished[2].wait(), timeout=1)
        await asyncio.wait_for(client.finished[14].wait(), timeout=1)
        assert not client.started[12].is_set()
        client.release[13].set()
        result = await execution
        assert result.completed_requests == 2
        ids = [rid for kind, _, rid, _ in client.calls if kind == "start"]
        assert len(set(ids)) == 8

    asyncio.run(scenario())


@pytest.mark.parametrize("cancelled", [False, True])
def test_failed_child_cancels_siblings_without_releasing_parent(cancelled, tmp_path):
    async def scenario():
        writer = ClientLifecycleWriter(
            output_path=tmp_path / "events.jsonl",
            run_id="r",
            cell_id="c",
            arm="off",
            concurrency=1,
        )
        client = ControlledClient(
            blocked=[4],
            failing=[] if cancelled else [3],
            cancelled=[3] if cancelled else [],
            writer=writer,
        )
        result = await run(client, [fork_join()])
        assert result.completed_requests == 0
        assert result.failed_requests == 1
        assert not client.started[2].is_set()
        assert client.active == 0
        events = [
            json.loads(line)["event"]
            for line in writer.output_path.read_text().splitlines()
        ]
        assert events.count("TASK_FAILED") == 1
        assert "TASK_COMPLETE" not in events

    asyncio.run(scenario())


def test_cutoff_cancels_group_and_all_children(tmp_path):
    async def scenario():
        writer = ClientLifecycleWriter(
            output_path=tmp_path / "events.jsonl",
            run_id="r",
            cell_id="c",
            arm="off",
            concurrency=1,
        )
        client = ControlledClient(blocked=[3, 4], writer=writer)
        result = await run(client, [fork_join()], drain_timeout=0.08)
        assert result.completed_requests == result.failed_requests == 0
        assert not client.started[2].is_set()
        assert client.active == 0
        events = [
            json.loads(line)["event"]
            for line in writer.output_path.read_text().splitlines()
        ]
        assert events.count("TASK_CANCELLED") == 1
        assert "TASK_COMPLETE" not in events

    asyncio.run(scenario())


def test_repeated_instances_get_fresh_signals_and_ids(tmp_path):
    async def scenario():
        writer = ClientLifecycleWriter(
            output_path=tmp_path / "events.jsonl",
            run_id="r",
            cell_id="c",
            arm="off",
            concurrency=2,
        )
        client = ControlledClient(writer=writer)
        task = fork_join()
        result = await run(client, [task, task], concurrency=2)
        assert result.completed_requests == 2
        ids = [rid for kind, _, rid, _ in client.calls if kind == "start"]
        assert len(set(ids)) == 8

    asyncio.run(scenario())


def test_nested_agent_dependency_and_tool_wait():
    async def scenario():
        task = fork_join()
        task.traces[1].events = [generate(3), Wait(0.03, actor="tool"), generate(5)]
        task.traces.append(ReplayTrace("grandchild", [generate(6)]))
        task.dependencies[-2] = edge("child-a", 2, "main", 1)
        task.dependencies += [
            edge("child-a", 0, "grandchild", 0),
            edge("grandchild", 0, "child-a", 1),
        ]
        task = ReplayTask.from_jsonl_row(task.to_jsonl_row())
        client = ControlledClient()
        result = await run(client, [task])
        times = {(kind, label): t for kind, label, _, t in client.calls}
        assert times["start", 6] >= times["finish", 3]
        assert times["start", 5] - times["finish", 6] >= 0.025
        assert times["start", 2] >= times["finish", 5]
        assert result.completed_model_requests == 6

    asyncio.run(scenario())


def test_delay_scale_zero_preserves_dependencies():
    async def scenario():
        client = ControlledClient()
        result = await run(client, [fork_join()], delay_scale=0)
        order = [(kind, label) for kind, label, *_ in client.calls]
        assert order.index(("finish", 3)) < order.index(("start", 2))
        assert result.completed_requests == 1

    asyncio.run(scenario())


def test_group_start_offset_and_mixed_v2_v3_workload():
    async def scenario():
        task = fork_join()
        task.start_offset_s = 0.03
        client = ControlledClient()
        result = await run(
            client,
            [task, ReplayTrace("single", [generate(99)])],
            concurrency=None,
            request_concurrency=2,
        )
        assert client.calls[0][1] == 99
        assert result.expected_requests == result.completed_requests == 2
        assert result.completed_model_requests == 5
        starts = {label: t for kind, label, _, t in client.calls if kind == "start"}
        assert starts[1] - starts[99] >= 0.02

    asyncio.run(scenario())


def test_schema_roundtrip_and_jsonl_loading(tmp_path):
    row = fork_join().to_jsonl_row()
    assert ReplayTask.from_jsonl_row(row).to_jsonl_row() == row
    path = tmp_path / "task.jsonl"
    path.write_text(json.dumps(row) + "\n")
    assert load_workloads(str(path))[0].to_jsonl_row() == row
    schema = files("trie").joinpath("schema", "replay-task-v3.schema.json")
    public = Path(__file__).parents[1] / "schema" / schema.name
    assert schema.read_text() == public.read_text()
    v2 = json.loads(
        files("trie").joinpath("schema", "replay-trace-v2.schema.json").read_text()
    )
    registry = Registry().with_resource(v2["$id"], Resource.from_contents(v2))
    fixture = json.loads((public.parent / "fixtures/multiagent-v3.json").read_text())
    Draft202012Validator(json.loads(public.read_text()), registry=registry).validate(
        fixture
    )
    assert ReplayTask.from_jsonl_row(fixture).to_jsonl_row() == fixture


@pytest.mark.parametrize(
    "change,match",
    [
        (
            lambda r: r["dependencies"][0]["after"].update(trace_id="missing"),
            "unknown dependency",
        ),
        (
            lambda r: r["dependencies"][0]["after"].update(event_index=5),
            "unknown dependency",
        ),
        (
            lambda r: r["dependencies"][0]["after"].update(event_index=True),
            "schema validation",
        ),
        (
            lambda r: r["dependencies"][0]["after"].update(event_index=0.0),
            "schema validation",
        ),
        (
            lambda r: r["dependencies"][0]["after"].update(event_index=-1),
            "schema validation",
        ),
        (
            lambda r: r["dependencies"].append(copy.deepcopy(r["dependencies"][0])),
            "duplicate dependency",
        ),
        (
            lambda r: r["traces"].append(copy.deepcopy(r["traces"][0])),
            "IDs must be unique",
        ),
        (lambda r: r["traces"][1].update(start_offset_s=1), "start_offset_s=0"),
        (lambda r: r["dependencies"][0]["before"].update(event_index=1), "cycle"),
        (lambda r: r.update(dependecies=[]), "schema validation"),
        (lambda r: r.update(schema_version=3.0), "schema validation"),
    ],
)
def test_invalid_graph_rejected_before_execution(change, match, monkeypatch):
    # Even the legacy fast-load bypass must not disable graph validation.
    monkeypatch.setenv("TRIE_SKIP_ROW_VALIDATION", "1")
    row = fork_join().to_jsonl_row()
    change(row)
    with pytest.raises(ValueError, match=match):
        ReplayTask.from_jsonl_row(row)


def test_composer_preserves_agents_and_does_not_publish_invalid_tasks(tmp_path):
    task = fork_join()
    agents = tmp_path / "agents.jsonl"
    agents.write_text("".join(json.dumps(t.to_jsonl_row()) + "\n" for t in task.traces))
    entry = task.to_jsonl_row()
    entry.pop("traces")
    entry.pop("schema_version")
    entry["trace_ids"] = [t.trace_id for t in task.traces]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([entry]))
    output = tmp_path / "tasks.jsonl"
    script = Path(__file__).parents[1] / "scripts/compose_multiagent.py"
    compose = runpy.run_path(str(script))["compose"]
    assert compose(str(agents), str(manifest), str(output)) == 1
    assert load_workloads(str(output))[0].to_jsonl_row() == task.to_jsonl_row()
    saved = output.read_bytes()
    entry["dependencies"][0]["before"]["event_index"] = 1
    manifest.write_text(json.dumps([entry]))
    with pytest.raises(ValueError, match="cycle"):
        compose(str(agents), str(manifest), str(output))
    assert output.read_bytes() == saved


@pytest.mark.parametrize("stream", [False, True])
def test_fork_join_through_real_http_and_openai_parser(stream, tmp_path):
    """Exercise socket I/O, body serialization and token verification without GPUs."""
    release = threading.Event()
    child_b_done = threading.Event()
    seen = []
    errors = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            label = body["prompt"][0]
            seen.append(label)
            if (
                body.get("forced_output_token_ids") != [7]
                or body.get("max_tokens") != 1
            ):
                errors.append(body)
            if label == 3 and not release.wait(3):
                errors.append("slow-child gate timed out")
            response = {
                "id": f"cmpl-{label}",
                "object": "text_completion",
                "created": 1,
                "model": "fake",
                "choices": [
                    {
                        "index": 0,
                        "text": "x",
                        "logprobs": None,
                        "finish_reason": "length",
                        "prompt_token_ids": body["prompt"],
                        "token_ids": [7],
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            raw = json.dumps(response).encode()
            if stream:
                raw = b"data: " + raw + b"\n\ndata: [DONE]\n\n"
            self.send_response(200)
            self.send_header(
                "Content-Type", "text/event-stream" if stream else "application/json"
            )
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            self.wfile.flush()
            if label == 4:
                child_b_done.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()

    async def scenario():
        client = Client(
            endpoint=f"http://127.0.0.1:{server.server_port}/v1",
            model="fake",
            api_key="test",
            max_retries=0,
        )
        source = tmp_path / "task.jsonl"
        source.write_text(json.dumps(fork_join().to_jsonl_row()) + "\n")
        execution = asyncio.create_task(run(client, str(source), stream=stream))
        assert await asyncio.to_thread(child_b_done.wait, 2)
        assert 1 in seen and 3 in seen and 2 not in seen
        release.set()
        result = await execution
        assert not errors
        assert result.completed_requests == 1
        assert result.completed_model_requests == 4
        assert result.failed_requests == 0
        assert result.completed_prompt_tokens == result.completed_completion_tokens == 4
        assert seen[-1] == 2

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
