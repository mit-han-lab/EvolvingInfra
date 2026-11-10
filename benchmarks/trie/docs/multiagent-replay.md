# Multi-agent replay with completion dependencies

Schema v3 groups several schema-v2 agent traces into one replay task. Each agent
keeps its own context and sequential event list. Explicit dependencies connect
those lists: a downstream event starts only after all its predecessors have
**actually completed in this replay**, including model response verification.
No inference-engine changes are required.

This supports parallel children, joins, nested agents, and dependencies between
any recorded agent events. It replays a recorded execution, not a live agent:
it does not execute tools, discover new branches, or substitute newly generated
child content into another agent's recorded prompt.

## Example: dispatch two children, then resume the parent

The executable format example is [multiagent-v3.json](../schema/fixtures/multiagent-v3.json).
It contains these agent event lists:

| Agent | Event 0 | Event 1 |
|---|---|---|
| main | Generate the dispatch | Generate the continuation |
| child-a | Generate the child response | — |
| child-b | Generate the child response | — |

Its four completion edges are:

```json
[
  {"before":{"trace_id":"main","event_index":0}, "after":{"trace_id":"child-a","event_index":0}},
  {"before":{"trace_id":"main","event_index":0}, "after":{"trace_id":"child-b","event_index":0}},
  {"before":{"trace_id":"child-a","event_index":0}, "after":{"trace_id":"main","event_index":1}},
  {"before":{"trace_id":"child-b","event_index":0}, "after":{"trace_id":"main","event_index":1}}
]
```

`event_index` is zero-based and counts **all** events, including waits and context
updates, not just model calls. To wait for a whole child, reference its last
event. Multiple edges targeting one event mean wait for **all** of them.
Every agent's own event order remains an implicit dependency.

If child-a originally took 10 seconds but this replay takes 30 seconds, the
parent waits for the actual 30 seconds. Child-b can run in parallel. Waiting
for dependencies never occupies a model-request concurrency slot.

## Converting existing traces

First convert each recorded agent session into the existing schema-v2 format,
preserving prompt IDs/context replacements, output lengths and external waits.
Then provide a manifest with one object per complete task:

```json
[
  {
    "trace_id": "task-1",
    "trace_ids": ["main", "child-a", "child-b"],
    "dependencies": [
      {"before":{"trace_id":"main","event_index":0}, "after":{"trace_id":"child-a","event_index":0}},
      {"before":{"trace_id":"main","event_index":0}, "after":{"trace_id":"child-b","event_index":0}},
      {"before":{"trace_id":"child-a","event_index":0}, "after":{"trace_id":"main","event_index":1}},
      {"before":{"trace_id":"child-b","event_index":0}, "after":{"trace_id":"main","event_index":1}}
    ]
  }
]
```

From `benchmarks/trie/`:

```bash
uv run python scripts/compose_multiagent.py \
  --traces sessions.schema-v2.jsonl \
  --manifest tasks.dependencies.json \
  --output tasks.schema-v3.jsonl

uv run trie \
  workload_path=tasks.schema-v3.jsonl \
  endpoint=http://127.0.0.1:30000/v1 model=served-model \
  strict_replay=true max_retries=0 concurrency=64 \
  duration=3600 drain_timeout=0
```

The composer rejects missing references, duplicate IDs/edges and cycles before
publishing output. An invalid manifest does not replace an existing output file.
The public [v3 schema](../schema/replay-task-v3.schema.json) references the bundled
v2 schema; the runtime resolves/validates agent content locally, without network
access. Existing v2 workloads still work and can be mixed with v3 task rows.

For RoadmapBench, use the original task/tool-call links to reconstruct dispatch
and join edges, and its reconstruction supplement for each agent's exact prompt
context (including compaction). **Do not infer dependencies just from timestamps,
and do not flatten child agents into independent top-level traces.** The generic
composer consumes already converted agent traces and explicit edges; it is not
an Oh My Pi archive parser or prompt reconstructor.

## Waits and concurrency

- `concurrency=64` means 64 active **complete tasks** for a v3-only workload,
  regardless of how many child agents each task contains. For v2 rows the unit
  remains one agent trace. `arrival_rate` similarly admits top-level workloads.
- `request_concurrency`, if set, caps model requests across all agents/tasks.
  `concurrency=1 request_concurrency=1` can finish a task with many dependent
  children; blocked parents do not consume that request slot.
- Child `start_offset_s` must be zero. To preserve dispatch/tool overhead, put
  an explicit `wait` event **after** the relevant dependency. A group's top-level
  `start_offset_s` remains a benchmark-relative admission delay, like v2.
- Preserve real external/tool delays, but **remove the captured wait for child
  model execution that a join now represents**. For example, if the original
  parent waited 10 seconds for children, do not append that same 10-second sleep
  after waiting for the children in the replay. Add only independently measured
  dispatch/return overhead; do not double count overlapping intervals.
- `delay_scale=0` removes recorded time delays but does **not** remove dependencies.
- Dependencies and completion signals are isolated per task invocation, also
  when the same task is repeated or concurrently admitted more than once.

## Block runs, shared salts and warm start

- Generates may give `recorded_prompt_block_runs` / `recorded_output_block_runs`
  instead of token IDs (see `trie.blocks`); a task's `metadata.block_tokens`
  serves every agent without its own, and all agents expand with the task's
  replay cycle, so equal block IDs give equal tokens across the task.
- `cache_salt_scope` is `agent` by default: every agent of a task gets its own
  cache salt, as before. `task` gives all agents of one task invocation one
  salt, for agents that really share prefixes.
- `warm_start=stationary` needs `source_t_s` on every generate: generates
  recorded before a uniform cut are skipped and complete their edges at once,
  and each agent still running re-sends its last skipped prompt for one token.
  `lazy_workload=true` parses a row only when its task is admitted.
- Lifecycle `CLIENT_ADMIT` of v3 requests carries `agent_id` and `warmup`, and
  a failed request records its `prompt_tokens`.

[Claude Code trace replay](cc-trace-replay.md) uses all four.

For Roadmap/EdgeBench runs that must isolate repeated task instances, use
[`trajectory_prefix_tokens=10`](trajectory-prefix.md). All agents and turns of
one complete task share its prefix; each new instance gets a fresh one.

## Failure and measurements

If an agent request fails validation or returns an API error, the task fails
once. Unfinished siblings are cancelled and awaited, and dependent events are
not admitted. A benchmark cutoff cancels the entire task, records
`TASK_CANCELLED`, and never counts it as completed. Existing duration/drain
settings remain in effect; the feature does not redefine that policy.

For v3 rows, `completed_requests`, `failed_requests`, `trace_per_s` and `latency_s`
refer to **whole tasks**. Task latency starts after the top-level start offset
and ends when every agent has completed; it includes dependency and external
waits. `completed_model_requests` counts individual successful model calls.
Stream TTFT/TPS and server-cache samples retain their existing per-agent scope;
they must not be interpreted as task TTFT. Mixed v2/v3 latency summaries mix the
two workload units, so use a homogeneous workload when comparing task latency.

Lifecycle logs add `TASK_START`, `TASK_COMPLETE`, `TASK_FAILED` and `TASK_CANCELLED`.
Each child model request's `CLIENT_ADMIT.attrs.task_request_id` links it to its
task. Model-request identities include the enclosing task, agent and repeat
cycle, preventing collisions when tasks reuse local names such as `main`.

Validation is linear in the number of events plus edges. At replay time each
edge is waited/signalled once, without polling or rescanning all queued tasks.
Completion signals are allocated only for referenced events. All of this runs
in the replay client, outside the router and inference engine.

## Verification

```bash
uv run --extra test pytest -q tests
```

The tests exercise delayed children and parallel joins, nested agents and real
tool waits, a single model-request slot, task admission limits, repeated task
isolation, failure/cutoff cleanup, token-only lifecycle accounting, invalid
graphs, and the normal OpenAI HTTP request/response path. These are CPU-side
functional checks, not GPU throughput measurements.

A dependency may additionally specify `"when": "start"`. It releases its
successor when the referenced client event begins processing; it does not
wait for that event to finish and is not a server/GPU admission signal.
Omitting `when` preserves the existing completion behavior. Cycle validation
includes separate start/completion milestones and the implicit sequence in
each agent. The CC causal converter uses this for a fork during a parent call.
