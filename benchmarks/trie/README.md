# Trie strict replay

Trie is the replay client used to benchmark evolving-infra. It drives the
DecoupledEngine (`engine/`, derived from SGLang) and the router through their
OpenAI-compatible and native HTTP contracts. Keeping it in the same repository
keeps the replay schema, HTTP extension and conformance tests in step with the
engine.

The imported Trie source revision and update policy are documented in
[SGLANG_VENDOR.md](SGLANG_VENDOR.md). This copy was taken from the SGLang fork
the published results used; the tree is otherwise unchanged apart from removing
internal data locations.

## Repository boundary

- Trie is a standalone Python distribution (`trie`) under `benchmarks/trie/`,
  with its own `pyproject.toml`, lock file, source package, schema and tests.
  It pins `transformers==4.57.6` and requires Python >= 3.11, so install it in
  its own virtual environment, not beside the engine.
- Trie talks to the engine and router only over HTTP. Neither package imports
  the other. The prepared-path adapter that routes Trie through the
  router frontend lives in [`benchmarks/replay`](../replay).

## Strict replay contract

A schema-v2 `generate` event may contain:

- `recorded_prompt_token_ids`: authoritative IDs after the source server's
  prompt rendering and tokenization;
- `recorded_output_token_ids`: authoritative IDs committed by the source run;
- `recorded_assistant`: structured assistant content used by text-compatible
  tooling;
- generic provenance such as `source_request_id`, `source_model`, and
  `source_timestamp`.

For bit-exact replay, prompt IDs must be non-empty and output IDs must contain
exactly `max_tokens` non-negative integers. Trie sends the prompt as an integer
array to SGLang `/v1/completions`, requests:

```json
{
  "forced_output_token_ids": [123, 456, 2],
  "return_token_ids": true,
  "ignore_eos": true
}
```

and fails the trace if SGLang returns different prompt or output IDs. The target
model, scheduler, KV cache, and speculative verifier still run; replay controls
the tokens committed to request state.

`metadata.routing_key` is the benchmark-neutral sticky-routing field. Trie sends
it as `X-SMG-Routing-Key`. `metadata.session_key` is accepted only for legacy
trace compatibility.

The language-neutral contract is
[`schema/replay-trace-v2.schema.json`](schema/replay-trace-v2.schema.json), with
a complete fixture under [`schema/fixtures`](schema/fixtures).

## Run

Multi-agent tasks with parallel children and completion dependencies use schema
v3; see [multi-agent replay](docs/multiagent-replay.md) for the task format,
conversion helper, concurrency units, and fork/join timing semantics. Schema-v2
single-agent replay remains supported. Claude Code session replay builds v3
tasks with hash-block prompts, one cache salt per task and a stationary warm
start; see [Claude Code trace replay](docs/cc-trace-replay.md).

For Roadmap/EdgeBench task isolation, add
[`trajectory_prefix_tokens=10`](docs/trajectory-prefix.md): one random prefix
per admitted trajectory, retained by its later turns and child agents.

```bash
uv sync --extra test

uv run trie \
  workload_path=/path/to/replay.schema-v2.jsonl \
  endpoint=http://127.0.0.1:8000/v1 \
  model=served-model-name \
  strict_replay=true \
  max_retries=0 \
  replay_once=true \
  request_concurrency=8 \
  duration=3600 \
  drain_timeout=3600
```

`request_concurrency` admits all one-shot traces onto the same
benchmark-relative timeline, honors each `start_offset_s` from that common
origin, and limits concurrent model requests. Use `concurrency` instead only
when intentionally limiting the number of active traces; that policy can
serialize a captured multi-session timeline.

`duration` bounds trace admission; `drain_timeout` bounds how long admitted
traces may finish afterward. Its default is `0`, which cancels active traces
immediately when admission stops, so one-shot replay should set it explicitly.

Trie loads a local tokenizer only for synthetic workloads, legacy text fallback,
or prefix ordering without recorded prompt IDs. `tokenizer_model` defaults to
`model`; set it explicitly when the served model name is only an alias. Fully
authoritative traces, where every generate event has recorded prompt and output
IDs, do not import or initialize a local tokenizer and may omit
`recorded_assistant`.

## Scope boundary

This package contains only generic replay schema, workload execution, metrics,
and SGLang protocol integration. Benchmark collectors, protected capture
proxies, task/session manifests, model launch recipes, and cluster deployment
files belong to their adapter repositories.

See [docs/strict-replay-sglang.md](docs/strict-replay-sglang.md) for runtime
semantics and verification requirements.

## Test

```bash
uv run --frozen --extra test pytest -q tests
```

The engine's matching forced-output tests live with the engine.

For an end-to-end inference/replay check against any small served model:

```bash
uv run python scripts/sglang_strict_replay_conformance.py \
  --endpoint http://127.0.0.1:30000/v1 \
  --model served-model-name
```

The script collects authoritative IDs once, then asks Trie and SGLang to replay
the same request twice and fails on any prompt/output mismatch.
