# Evolving Infra Documentation

**Evolving Infra** is an agent-adaptive LLM serving stack. Its engine workers can each run the
prefill or the decode leg of a request, a planner changes the prefill/decode split while the
service runs, and an agent-aware KV manager keeps the context agents still need in a shared
host-memory tier.

## Key Features

- **Symmetric P/D workers**: any DecoupledEngine worker can take either leg. KV moves from P to D
  over RDMA, chunk by chunk, while the prefill runs.
- **Online P/D planner**: fits a step-time model per worker from engine traces and switches the
  split through the router's role API, without restarting engines.
- **Agent-aware KV retention**: branch-level references keep an agent's context in host memory
  between turns. Unreferenced KV is evicted first.
- **Cache- and load-aware routing**: the Rust router places each request using a KV directory
  built on Dynamo's indexer and per-worker work reservations.
- **Prepared path**: the frontend tokenizes once and uploads inputs straight to the chosen
  workers. The router handles only compact descriptors.
- **Strict replay benchmarks**: replays recorded agent traces token by token, with checks that
  decide whether a run is valid.

## Models and hardware

| Model | H200 | B300 | GB300 | Speculative decoding |
|---|:-:|:-:|:-:|---|
| DeepSeek-V4-Flash | ✅ | ✅ | 🚧 | DSPARK |
| GLM-5.2 | 🚧 | ✅ | ✅ | EAGLE |
| Kimi-K2.6 | 🚧 | ✅ | ✅ | EAGLE3 |

✅ tested · 🚧 untested.

## Contents

| Page | Description |
|---|---|
| [GLM-5.2 on GB300](../examples/gb300/README.md) | `evolving-infra serve` on one node, and the paper's nine-node experiment with one command |
| [Quickstart](quickstart.md) | Run the stack on one node and send a request; install from source; benchmarks on Modal |
| [Quickstart on GB300](quickstart-gb300.md) | The paper's configuration on two GB300 nodes with GLM-5.2 |
| [Architecture](architecture.md) | Components, request path, repository layout, names kept for compatibility |
| [Configuration](configuration.md) | Engine, router, frontend and indexer flags the benchmarks use |
| [Placement](placement.md) | How the router picks P and D: KV directory, cost function, peer pulls, late decode choice |
| [Dynamic P/D](dynamic-pd.md) | Changing the P/D split at runtime: role API, planner, configuration |
| [KV Retention](kv-retention.md) | Keeping agents' context in host memory between turns |
| [API Reference](api-reference.md) | Router HTTP endpoints and the client SDK |
| [From the Paper to the Code](paper.md) | Where each mechanism and parameter of the paper lives, and the flags that turn it on |

Component READMEs: [engine](../evolving_infra/engine/README.md), [router core](../evolving_infra/router/core/README.md),
[frontend](../evolving_infra/router/frontend/README.md), [indexer](../evolving_infra/router/indexer/README.md),
[planner](../evolving_infra/planner/README.md), [benchmarks](../benchmarks/README.md),
[Modal benchmarks](../examples/modal/README.md), [runtime image](../docker/README.md).
