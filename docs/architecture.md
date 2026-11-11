# Architecture

evolving-infra serves agent workloads on a pool of identical engine workers. Any worker can
run the prefill (P) leg or the decode (D) leg of a request. A central Rust router picks P and D
per request. A planner changes how many workers act as P or D while the service runs, and no
engine restarts.

```text
 client        frontend (T)          router               indexer        engine P      engine D
   │ (1) /prepare  │                    │                     │               │             │
   │──────────────▶│ tokenize, hash     │                     │               │             │
   │◀── descriptor─│                    │                     │               │             │
   │ (2) /symmetric_pd/prepared ───────▶│── POST /query ─────▶│               │             │
   │◀─────────────────────────── plan ──│◀── hits per worker ─│               │             │
   │(3) forward_pd │                    │                     │               │             │
   │─ plan ───────▶│── (4) /generate + kv_export ────────────────────────────▶│             │
   │               │── (4) /generate + kv_import ──────────────────────────────────────────▶│
   │               │                    │                     │               │─ KV, RDMA ─▶│
   │◀──── D output │◀───────────────────────────────────────────────────────── output of D ─│
   │               │─ (5) events ──────▶│                     │               │             │
   │               │                    │                     │◀────────── KV events (ZMQ) ─│

 planner ── GET /symmetric_pd/roles, PUT /symmetric_pd/roles (P/D weights; engines are not restarted) ──▶ router
```

## Components

| Component | Directory | Language | Role |
|---|---|---|---|
| DecoupledEngine | [`evolving_infra/engine/`](../evolving_infra/engine/README.md) | Python (CUDA kernels) | SGLang-derived engine. `--disaggregation-mode symmetric` makes every worker able to run either leg. Moves KV from P to D with one-sided Mooncake RDMA reads while P is still computing, keeps a host-memory KV tier, and can pull KV from a peer's host pool. |
| Router | [`evolving_infra/router/core/`](../evolving_infra/router/core/README.md) | Rust (`sgl-router`) | Serves one model. Under `--policy symmetric_pd` it picks a P and a D for every request, keeps per-worker work reservations, accepts role changes, and keeps the session registry used for KV retention. |
| Frontend (T) | [`evolving_infra/router/frontend/`](../evolving_infra/router/frontend/README.md) | Python | Tokenizes and hashes the input, stores the token IDs, uploads the input to the chosen P and D, streams D's output to the caller, and reports progress to the router. One per replay or client process. |
| Indexer | [`evolving_infra/router/indexer/`](../evolving_infra/router/indexer/README.md) | Python, runs Dynamo | Supervises the pinned `python -m dynamo.indexer` (`ai-dynamo==1.4.0`), one KV event bridge per worker, and a hash-query adapter. It answers which worker holds which prefix blocks. |
| Client SDK | [`evolving_infra/router/client/`](../evolving_infra/router/client/__init__.py) | Python | `Agent`, `Scheduler` and `Backend` for the prepared path. Gives each request an agent / sub-agent / request identity. |
| Planner | [`evolving_infra/planner/`](../evolving_infra/planner/README.md) | Python | Fits per-worker step-time models from engine traces, then switches the P/D split through the router's role API when the predicted gain is significant. |
| Replay | [`benchmarks/`](../benchmarks/README.md) | Python | Trie strict-replay client, the adapter that runs it through the prepared path, and the run validity checks. |

## Request path (prepared path)

The replay benchmarks use this path. The router never sees full token arrays, and Python keeps
no second load ledger.

1. **Prepare.** The client sends the request to a frontend, `POST /prepare`. The frontend
   tokenizes it (through the engine's `/tokenize` or the given `input_ids`), stores the token
   IDs locally, and returns a compact descriptor: a reference, the prompt length, and page
   hashes in the indexer's hash scheme (`hash_mode`).
2. **Plan.** The client sends the descriptor to the router, `POST /symmetric_pd/prepared`. The
   router queries the indexer with the hashes, scores the P candidates (cache-first, cost
   function, see [placement](placement.md)), picks a D, reserves the
   work, mints a transfer ID, and returns a plan. The plan holds both legs, their KV marks, and a
   per-plan callback token.
3. **Forward.** The client sends the plan to the same frontend, `POST /forward_pd`. The
   frontend claims the plan with the router and sends the input to both engines at once:
   - P gets the input with a `kv_export` mark.
   - D gets the input with a `kv_import` mark that names P.

   The frontend streams D's response back to the client.
4. **Transfer.** P computes the prefill and publishes KV chunk by chunk. D reads the chunks from
   P over RDMA while the prefill is still running, then decodes. With late decode choice, the
   frontend sends the request to several D candidates. P asks the router which one to admit at its
   first prefill chunk, and the others release ([placement](placement.md#picking-d)).
   With pull hints, P first fetches a host-resident prefix from a peer's servicer instead of
   recomputing it.
5. **Report.** The frontend reports claim, upload, completion and failure to the router,
   `POST /symmetric_pd/events`, with the engine counts (cached, completion and verify tokens). The
   router retires the reservations. It also polls each engine's `/v1/loads` for progress.

The SDK for this path is in the [API reference](api-reference.md#client-sdk).
The router also serves `/v1/completions` and `/v1/chat/completions` directly, and
`POST /symmetric_pd/place` returns a placement to an external ingress. The replay benchmarks use
neither.

## KV directory

Engines publish KV events (`--kv-events-config`, ZMQ) when blocks are stored or removed. One
bridge per worker forwards them to the Dynamo indexer. The router queries the indexer for each
request (`POST /query` on the hash-query adapter, with the frontend's page hashes). It turns
the reply into per-worker GPU and host hit lengths through `--kv-worker-map`. Queries fail open:
when the indexer is unavailable, placement runs on loads and role weights only. A directory
hit is not a transfer lease. The engine still checks that the pages are there and owns transfer
safety.

## Dynamic P/D

Each worker has a prefill and a decode weight in the router. The planner fits a step-time model
per worker from engine traces and changes the weights with `PUT /symmetric_pd/roles` when another
split is predicted to be faster. Engines do not restart; a worker moving from decode to prefill
drains or migrates its decode legs first. See [Dynamic P/D](dynamic-pd.md).

## KV retention

The router tells the engines which context each agent branch still references, with a versioned
`kv_session` tag on the request and background control messages, and the engines evict
unreferenced host KV first. It is off in the tested configurations. See
[KV retention](kv-retention.md).

## Repository structure

```text
evolving-infra/
├── evolving_infra/            # the system; also the Python import root (evolving_infra.planner, ...)
│   ├── engine/                # DecoupledEngine (SGLang-derived), its own package decoupled_engine; srt/ holds the runtime
│   ├── router/
│   │   ├── core/              # Rust router, binary sgl-router (sgl-kv-indexer/ is its KV index client crate)
│   │   ├── frontend/          # tokenizer frontend (T): /prepare, /forward_pd
│   │   ├── indexer/           # KV directory: supervises Dynamo's indexer, event bridges, query adapter
│   │   └── client/            # Python client SDK: Agent, Scheduler, Backend
│   ├── planner/               # online P/D controller, step-time fit, trace servers
│   └── launch/                # evolving-infra serve
├── benchmarks/                # trie/ (strict-replay client), replay/ (adapter, run checks), reference results
├── examples/                  # a config per hardware and model; slurm/ and modal/ run one
├── docker/                    # runtime image
└── tests/                     # Python tests; Rust tests are in evolving_infra/router/core/tests/
```

The Python components install as one distribution, `evolving-infra`, with imports under
`evolving_infra.router.{client,frontend,indexer}` and `evolving_infra.planner`. The engine is
the `decoupled-engine` distribution, and Trie is its own `trie` distribution.

## Names kept from SGLang

The engine package is renamed from `sglang` to `decoupled_engine`, but names that other
components and dashboards read stay as in SGLang: the `SGLANG_*` environment variables, the
`sglang:` Prometheus metric prefix, the router binary `sgl-router`, the wire schema
`sglang.prefill_state_push`, and the config values `sglang_native` / `sglang_generate`. The
reference deployment's ports and settings are in [`examples/modal/stack.py`](../examples/modal/stack.py).

## Provenance

The engine and the router core are derived from [SGLang](https://github.com/sgl-project/sglang)
at upstream commit `9a489f8d2f`; [evolving_infra/engine/README.md](../evolving_infra/engine/README.md#relation-to-sglang)
summarizes what the engine changes.
[evolving_infra/router/core/PROVENANCE.md](../evolving_infra/router/core/PROVENANCE.md) records the router's origin, and
[NOTICE](../NOTICE) lists every included third-party work.
