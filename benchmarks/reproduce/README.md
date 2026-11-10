# Benchmark setup and expected results

[`deploy/modal`](../../deploy/modal/README.md) runs two benchmarks on Modal H200 nodes. A correct
build should land in these ranges (one run takes 20–30 minutes plus cold start):

| Benchmark | Topology | Requests in window | Failures | Prefix hit | TTFT p50 |
|---|---|---|---|---|---|
| static | 1 node, 1P1D | 530–550 | 0 | 97.6–97.9 % | 0.39–0.46 s |
| dynamic | 2 nodes, 2P2D → 1P3D → 2P2D | 1190–1250 | 0 | 96.3–96.4 % | 0.41–0.44 s |

Common setup:

- Model DeepSeek-V4-Flash (FP4 MoE, `--moe-runner-backend flashinfer_mxfp4`), served as
  `deepseek-v4-flash`; DSPARK speculative decoding with a fixed synthetic
  oracle accept rate.
- Two TP 4 workers per 8-GPU node, all in `--disaggregation-mode symmetric` over Mooncake RDMA;
  hierarchical (host) cache with DSV4 host dedup; peer-KV servicer on every worker; KV events to
  the Dynamo indexer (`ai-dynamo==1.4.0`) through the hash-query adapter; static memory fraction
  0.71.
- Router policy `symmetric_pd` with forced P/D split, cost-function P selection, KV pull hints from any
  holder, and late D choice (`fit_req` over every D at the P leg's first chunk).
- Replay: Trie with `actual_output` context and model outputs, through the prepared path
  (`benchmarks/replay/run_trie.py`, `cache_salt_mode=global`).
- **KV session retention off** in engines and router, and no session END from the replay: the
  engine does not yet support retention on DSV4's FULL+SWA cache tree.
- The dynamic benchmark adds the planner's live control loop with external role control on the
  router: a role switch is one `PUT /symmetric_pd/roles` on the router and no engine restarts, so
  worker request counters stay monotonic.

A run is valid only when the four checks in [`benchmarks/replay`](../replay/README.md) pass:
input conformance, quiet window, steady-state zero failures and monotonic request counters.
Prefix hit is token-weighted (`cached_prompt_tokens / completed_prompt_tokens` over the window).

[`deploy/modal`](../../deploy/modal/README.md) runs both benchmarks with these checks.
The [quickstart](../../docs/quickstart.md) shows the same stack on one node.
