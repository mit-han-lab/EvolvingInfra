# From the Paper to the Code

Where each mechanism of *Evolving Infra: Agent-Adaptive LLM Serving at Scale* lives in this
repository, and the flags that control it. With `--disaggregation-mode symmetric` on the engines
and `--policy symmetric_pd` on the router, every mechanism below is on by default
([Configuration](configuration.md)). Engine paths are under `engine/decoupled_engine/srt/`.

## Mechanisms

| Paper | Flags | Code |
|---|---|---|
| §4.1 Symmetric engines: every worker serves both phases and switches phase without a restart | Engine `--disaggregation-mode symmetric`. Router `--policy symmetric_pd`, `--symmetric-pd-external-role-control` (`PUT /symmetric_pd/roles`) | `disaggregation/symmetric/`, `router/core/src/server/app.rs` |
| §4.2 Shared host-memory KV pool: prefill workers read KV from a peer's host memory over RDMA | Engine `--enable-hierarchical-cache --hicache-write-policy write_through`, `--enable-peer-kv-servicer`. Router `--symmetric-pd-pull-hint`, `--symmetric-pd-pull-from-any-holder` | `mem_cache/peer_kv_servicer.py`, `mem_cache/storage/peer_kv/`, `disaggregation/pull/` |
| §4.2 Deduplication across the GPUs of a node | Engine `--enable-dsa-hicache-host-dedup` (GLM-5.2), `--enable-mla-hicache-host-dedup` (Kimi-K2.6), `--enable-dsv4-hicache-host-dedup` (DeepSeek-V4) | `mem_cache/dsa_host_dedup.py`, `mem_cache/mla_host_dedup.py`, `mem_cache/dsv4_host_dedup.py` |
| §4.3 Agent-aware KV directory: KV placement plus the active branches that reference each block | Router `--kv-indexer-url`, `--kv-worker-map`, `--symmetric-pd-kv-session-retention`. Engine `--enable-kv-session-retention`. Clients send `x-agent-context` and `POST /kv_sessions/end` | `router/indexer/`, `router/core/src/agent_identity.rs`, `router/core/src/backend/kv_sessions.rs`, `mem_cache/unified_cache/session_ref_tracker.py` |
| §5.1 Prefill and decode iteration-time models, fitted online | Engine `--sched-trace-dir`; the planner fits during the learning window | `planner/fit/` (`steps.py`, `models.py`, `learning.py`) |
| §5.2 Online phase-split planner | `python -m evolving_infra.planner control` ([Dynamic P/D](dynamic-pd.md)) | `planner/control/loop.py`, `planner/decide/` |
| §5.3 Eviction: unreferenced KV first; among referenced KV, the branch predicted to return last | Engine `--enable-kv-session-retention` | `mem_cache/unified_cache/retention_order.py` |
| §5.3 Routing by predicted time to first token, with the fetch cost and the copy price | Router `--symmetric-pd-p-select cost-function --co-schedule-trade --co-schedule-uncached-balance`. Engine `--enable-prefill-work-snapshot` | `router/core/src/scheduler/cost_function.rs`, `router/core/src/scheduler/co_schedule.rs`, `managers/scheduler_components/prefill_routing_snapshot.py` |
| §5.3 Decode bound when the prefill starts | Router `--pd-late-decode choose --pd-late-decode-rule fit_req`. Engine `--disaggregation-late-d choose` | `disaggregation/late_d.py` |

## Parameters (Appendix E.3)

| Paper | Value | Where |
|---|---|---|
| Learning window T_learn | 1800 s | Planner config `learning_s` (default) |
| Planning window T_win | 600 s | Planner config `period_s` (default) |
| Slices per window m | 60 s slices, m = 10 | Fixed: `planner/decide/workload.py` `SLICE_S` |
| Switch rule | Mean gain above 2 standard errors | Fixed: `planner/decide/rule.py` `REQUIRED_STANDARD_ERRORS` |
| Pause after a switch | 120 s | Planner config `exclusion_s` (default) |
| Chunk size | 8192 tokens | Engine `--chunked-prefill-size 8192` |
| Fetch time τ(x) | 0.45 s + 1 µs per token | Router defaults `DEFAULT_TRADE_PULL_FIXED_MS`, `DEFAULT_TRADE_PULL_MS_PER_100K` |
| Smallest fetch | 512 tokens | Router `--kv-pull-hint-min-incremental-tokens 512` |
| Copy price κ | 4.6 | Router default `DEFAULT_TRADE_COPY_LAMBDA` (`--co-schedule-trade-copy-lambda`) |
| Age of a prefill worker's reported state | 1.5 s | Router default `DEFAULT_TRADE_EFT_MAX_VIEW_AGE_S` |
| Return-time table for eviction | Built-in table; a branch idle for 150 s or longer counts as finished | `retention_order.DEFAULT_FIXED_TABLE` |

[Configuration](configuration.md) lists every flag above with the rest of the benchmark setup.
The engine flags turn on the environment variables these mechanisms use, as in the paper's runs
([Environment](configuration.md#environment)).
