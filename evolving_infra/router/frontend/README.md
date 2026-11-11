# Distributed token preparation and P/D input forwarding

Importing this package starts no service.

```text
protocol.py    Versioned request/config models; reject unsupported fields.
tokenizer.py   Engine-native or Dynamo hashing, selected explicitly.
store.py       One expiring, byte-limited staging store in the event-loop process.
forward.py     Upload ownership, P/D concurrent sending, output and callbacks.
delivery.py    Bounded in-memory delivery of Backend completion callbacks.
server.py      HTTP endpoints, native tokenization and service lifecycle.
```

Install the root project with the `[frontend]` extra. Start with
`python -m evolving_infra.router.frontend --config /path/node.json`;
[node.example.json](node.example.json) lists the node settings. Tests are in
[tests/frontend](../../../tests/frontend).

Engine code is optional and imported lazily (`_engine.py`): `hash_mode=sglang_native`
(the default) needs `decoupled_engine.srt.mem_cache.cpp_utils.native_hash` from
the `decoupled-engine` distribution, and image inputs need the engine's CPU-only
`decoupled_engine/image_layout.py`, loaded by file path (so no Torch import) from
`DECOUPLED_ENGINE_DIR`, the installed `decoupled_engine` package, or this checkout. `hash_mode=dynamo` needs the `[indexer]`
extra instead. The configuration values `sglang_native` and `sglang_generate`
keep their names for compatibility with the Router and existing configs.

## Input preparation and hash contract

`POST /prepare` retains the protocol-v1 input/generation envelope. For text/chat,
it calls the configured native `/v1/tokenize` endpoint, stores token IDs locally and returns
`request_id`, `ref`, `config_id`, `prompt_tokens`, `max_new_tokens`, expiry and
`content`. Full token IDs are never returned to the central scheduler.

An explicit pretokenized input is also supported:
`"input": {"model": "/same/configured/model", "input_ids": [1, 2, 3]}`.
Choose exactly one of `prompt`, `messages` or `input_ids`. IDs must be a nonempty
uint32 array (booleans, floats and strings are rejected). Model/config checks and
the input-plus-output context budget are identical. This path hashes and forwards
the exact supplied text IDs; it never detokenizes or tokenizes them again. A timing
result for this path is not a tokenizer throughput measurement.

`generation.return_prompt_token_ids` is an optional boolean, default `false`.
When explicitly `true`, it is passed unchanged to the engines so the caller can
validate replay input from the generation response, through the tokenizer's
output connection. The prepared descriptor and Rust planning request still
contain only hashes/counts, never the full token array.

Here `request_id` is the globally unique **engine RID**, derived upstream from
`agent_id/subagent_id/request_id`. It is not the numeric agent step alone.

`content.hash_spec` and `/health.hash_spec` explicitly identify the hash contract:

- `hash_mode=sglang_native` uses the engine's `get_native_hash`:
  chained hex hashes, including a partial trailing page. This is not a Dynamo
  query hash.
- `hash_mode=dynamo` calls the installed
  `dynamo.llm.compute_block_hash_for_seq` with `page_size`, `is_eagle` and
  `cache_namespace=cache_salt`. It emits signed-i64 JSON values accepted by
  `query_by_hash`. Only full query windows are included: without EAGLE,
  `floor(N/page_size)`; with EAGLE, `floor(max(0,N-1)/page_size)`.
- Both include the configured model, block size and namespace in the contract.
  The indexer model/namespace and tokenizer settings must match. The adapter
  does not reimplement a hash function or pretend these hash modes are equivalent.
- Dynamo is an optional external dependency. Its installed binding must support
  `cache_namespace` and `is_eagle`; missing support fails at startup. `extra_key`
  is rejected for Dynamo until an exact compatible mapping exists.

The `content.blocks` array retains the order and `{hash, valid_tokens}` fields.
Consumers must validate `hash_spec`, not guess the scheme from the policy name.

## Uploaded images

DeepSeek-V4 Vision deployments set `image_config` to the checkpoint's fixed
`image_token_id`, `vision_patch_size`, `vision_downsample_ratio`,
`vision_max_n_token`, `vision_min_pixels` and `vision_max_wh_ratio`. The matching
engine's `decoupled_engine/image_layout.py` must be reachable as above; the frontend
uses this file without loading Torch or running ViT. `public_url` must name a
distributed-tokenizer address reachable from every worker. Omitting `image_config`
retains the text-only deployment contract.

The client uploads PNG/JPEG contents as Base64 strings or data URLs in
`input.image_data`, alongside `input_ids` or `prompt`. Chat requests instead use
uploaded data URLs in `messages[].content[].image_url.url`. Local paths and remote
image URLs are rejected. Supplied token IDs contain one image placeholder per
uploaded image, in the same order; callers must not expand the placeholders.

The distributed-tokenizer reads image headers, hashes the original file contents
and determines the checkpoint's image token layout. It hashes cache IDs that
include each image's identity, and returns the expanded `prompt_tokens` count.
The caller relays this descriptor to the router. The descriptor includes hashes,
counts and the fixed image configuration; it contains no images or full token
array. The router requires an exact match with its configured image contract.

For `/forward_pd`, the distributed-tokenizer sends the original token IDs,
uploaded images, image layout and generation parameters to the prefill worker.
The prefill worker decodes and resizes the images, computes ViT and aligner
outputs, and runs LLM prefill. The decode worker receives token IDs, layout,
generation parameters, `kv_import` and `prepared_image_source`, without an initial
image upload. It imports the prefill worker's KV and continues generation.

If KV import fails and the decode worker must prefill locally, it requests the
original file bytes from `POST /image` on the distributed-tokenizer. This request
includes `protocol_version`, `config_id`, `request_id`, `ref` and a zero-based
`image_index`. Images remain with their existing staged or active request owner;
release/completion removes that ownership. There is no separate image cache or
image lifetime. A released reference returns `404 REF_NOT_FOUND`.

## Forward one input to P and D

Single-target `POST /forward` is unchanged. Use it for local full execution.
For split execution the Backend sends an already selected plan to
`POST /forward_pd`:

```json
{
  "protocol_version": 1,
  "request_id": "globally-unique-engine-rid",
  "config_id": "glm-config",
  "ref": "prepare-reference",
  "reservation_id": "backend-owned-attempt-id",
  "transfer_id": 123,
  "prefill": {
    "target": {"transport": "sglang_generate", "endpoint": "http://p/generate", "config_id": "glm-config"},
    "kv_export": {"transfer_id": 123, "reader_addr": "tcp://d:port"},
    "pull_hint": {"...": "optional existing engine pull-hint fields"}
  },
  "decode": {
    "target": {"transport": "sglang_generate", "endpoint": "http://d/generate", "config_id": "glm-config"},
    "kv_import": {"transfer_id": 123, "owner_addr": "tcp://p:port"}
  },
  "callback": {"endpoint": "http://router/internal/tokenizer-events", "token": "per-plan-secret"}
}
```

`kv_export`, `kv_import` and optional `pull_hint` are produced by the engine
Backend adapter, including all engine-required fields. The tokenizer does not
choose nodes, create transfer IDs or invent KV policy. `transfer_id` is a JSON
u64 and must match both marks. The P and D endpoints must differ, be in `targets`
and have the prepared input's `config_id`. The callback URL must be in
`callback_targets`; both target checks happen before the one-shot `Store.take`.

The service consumes the reference exactly once. For text it serializes one common
input body shared by both uploaders; for images it creates a separate decode body
without image bytes. It attaches small per-leg suffixes. P gets
`kv_export`, optional `kv_pull_hint`, and `stream=false`. D gets `kv_import` and
the original stream setting. Both retain the original sampling/output budget:
the symmetric engine's export mode controls P's first-token completion.

Before either generation request, the tokenizer synchronously sends a `claimed`
event (`leg=prefill`, representing the whole plan) and requires Backend acceptance.
The Backend atomically claims both reservations. A missing/rejected/expired claim
prevents **all** generation requests; claim metadata has at most two attempts.
This lets the Backend safely expire plans that were never claimed, including a
plan whose response was lost. A late claim after expiry must be rejected, and a
previously claimed plan requires engine reconciliation rather than blind expiry.
If claim ACKs are lost, the tokenizer still has not created any engine POST task.
It makes bounded metadata-only reports for both legs with `state=failed` and
`not_submitted=true`. This is evidence of no submission, not an engine terminal
response. The Backend may release an accepted claim using that evidence only
while the corresponding leg has never uploaded. If cleanup notifications also
fail, unresolved state must remain visible for reconciliation.

After claim acceptance, P and D submit concurrently. D does not wait for P's HTTP response. Only D's
output is returned to the caller. Each uploader releases its suffix after sending;
upload ownership is released after both uploads finish. A resumable stream or
an image request retains its original input until the exchange closes, for
migration or failed-import recovery. HTTP buffers and temporary serialization
memory are not included in the staging-store capacity.

## Completion, cancellation and no duplicate generation

The Backend reserves/accounting state before forwarding and retains ownership of
engine-side cancellation and reconciliation. The tokenizer reports:

```json
{
  "protocol_version": 1,
  "request_id": "globally-unique-engine-rid",
  "reservation_id": "backend-owned-attempt-id",
  "transfer_id": 123,
  "leg": "prefill",
  "state": "completed",
  "event_id": "backend-owned-attempt-id:prefill:completed",
  "http_status": 200
}
```

`leg` is `prefill` or `decode`; `state` is `claimed`, `uploaded`, `completed`,
`failed` or `cancelled`. `claimed` is synchronous and nonterminal; other events
remain asynchronous. Errors can include an `error` code/class. Callbacks use
`Authorization: Bearer <token>`. Events are ordered per exchange; the Backend must
validate plan identity/token and deduplicate by `event_id`. A failed/cancelled leg
must trigger Backend cleanup of the whole plan, including an already finished P
whose export may still be waiting for D acknowledgment.

A `completed` event also carries the engine's own counts from the response
`meta_info` when present: `"usage": {"prompt_tokens", "cached_tokens",
"vision_matmul_flops", "image_attention_pairs"}` for P and
`"usage": {"completion_tokens", "spec_verify_ct"}` for D (for SSE, from the last
data event). The Router's P/D ratio controller sums them per window; the event is
unchanged when the engine reports none.

`uploaded` means the local HTTP sender consumed the input, **not** that the engine
accepted/completed it. `completed` means a successful response was fully consumed;
HTTP-200 engine errors/aborts are failures. D stream errors close the downstream
stream and report failure; consumers must not classify a truncated stream as a
successful request.

An explicit engine error/abort response adds `terminal_response=true` only for
the responding leg. A transport failure, missing callback, partial upload or
disconnect never establishes this flag. The Backend can retire an explicitly
terminal leg while reconciling other legs whose engine state remains uncertain.

There is no generation retry, except resuming a D-leg that the Backend migrated
(below). A consumed `ref` cannot be forwarded again; a 404
cannot establish that no computation happened. Cancellation/partial submission
closes both local upload/response resources and reports terminal states. Closing
HTTP is not proof the GPU stopped or export pages were released: the Backend must
call engine abort/reconcile as appropriate.

Only idempotent state callbacks are retried. Normal delivery uses two immediate
attempts with a configured per-attempt timeout. A failed callback and subsequent
events for that plan remain ordered in a service-owned in-memory outbox, even
after the HTTP request has ended. Background retries use the same event IDs with
exponential delays from 0.5 to 30 seconds. They never resend generation. The
initial `claimed` authorization is never deferred: without an immediate ACK,
neither engine POST starts. Its `not_submitted` cleanup events can be deferred.

`callback_outbox_max_plans` (default 1024) bounds the number of active plus
undelivered plans. A slot is reserved before consuming the input or claiming a
plan, so concurrent completions cannot overflow the outbox. At capacity, new
`/prepare` and `/forward_pd` calls return 503 `CALLBACK_BACKLOG`; a rejected forward
retains its unconsumed input. Existing uploads, outputs and cleanup continue.
No full input or load-accounting ledger is kept in this outbox.

`/health.callback_outbox` reports reserved/active/pending plan and event counts.
Failed attempts increment `callback_failures`; shutdown logs any unresolved
counts without exposing input or callback credentials. This outbox is **not
crash-durable**. A distributed-tokenizer process crash or an unreachable callback
endpoint can still leave claimed Backend reservations unresolved. Restarting the
distributed-tokenizer does not establish engine termination. The current Rust
Backend deliberately retains that unknown
state and does not yet implement a general engine reconciliation service.

## Resuming a migrated D-leg

When the Router drains a worker that moves from decode to prefill, it aborts the
D-legs still running there and marks them as migrating. A streamed exchange relays
whole SSE lines, so the distributed-tokenizer can hold back an abort event. On a
D abort it asks
`POST <callback base>/migration` (`/symmetric_pd/migration` next to the callback
endpoint) with the plan's callback token. Only for `{"migrating": true}`, and only
for incremental streams whose events add up to the reported `completion_tokens`,
the distributed-tokenizer resumes; any other abort fails the stream as before.

To resume, the distributed-tokenizer reports the D-leg failed with terminal
evidence (the Backend then aborts a P-leg that was still running) and prepares a new request: the original
input followed by the output tokens already sent downstream, the remaining
`max_new_tokens` and forced output tokens, and engine RID `m<sent>-<original>`.
The new RID must not extend the original one: the engine aborts every request whose
RID starts with the aborted RID. The distributed-tokenizer asks
`POST <callback base>/prepared` for a plan and runs it as a normal exchange.
Its events are rewritten into the original
stream: `completion_tokens` continues from the sent count, prompt and cached
counts and the engine ID stay the original's, and prompt token IDs are not
repeated once the original sent them. The tokens of the abort event itself were
never sent, so the client sees each output token once.

## Validation boundary

CPU tests use real local HTTP sockets to check simultaneous uploads, shared input
lifetime, cancellation/partial failure, callback retry, streaming and 64 concurrent
P/D plans. They also check Dynamo API argument/encoding adaptation using a stub;
that test does not certify a specific installed Dynamo build. Native engine hash
comparison remains opt-in via `TOKENIZER_TEST_NATIVE=1`. These tests do not claim
GPU correctness or performance equivalence to native PD.
