# Configuration

The flags you need to run Evolving Infra, grouped by component. The [quickstart](quickstart.md)
uses the minimal set; the reference benchmarks add the ones marked **bench**. The complete
benchmark configuration is in [`deploy/modal/stack.py`](../deploy/modal/stack.py). Run
`sgl-router --help` or `python -m decoupled_engine.launch_server --help` for every flag.

## Engine (`decoupled_engine.launch_server`)

The engine takes SGLang's server arguments. These are the ones Evolving Infra adds or depends on.

### P/D and KV transfer

| Flag | Description |
|---|---|
| `--disaggregation-mode symmetric` | Any worker can run the prefill or the decode leg. Required. |
| `--disaggregation-transfer-backend mooncake` | Move KV between workers with Mooncake over RDMA. |
| `--disaggregation-ib-device JSON` | RDMA device of every GPU, e.g. `{"0": "mlx5_0", ...}`. List all GPUs the worker uses. |
| `--disaggregation-bootstrap-port N` | Base port of the worker's KV transfer side channel. Must match the router's `--symmetric-pd-worker`. |
| `--incremental-streaming-output` | Required by the frontend. |

### Cache and KV directory

| Flag | Description |
|---|---|
| `--enable-hierarchical-cache` | Host-memory KV tier behind the GPU cache. |
| `--hicache-ratio R`, `--hicache-write-policy write_through` | **bench** Host tier size (as a multiple of the GPU cache) and write-through, so every GPU page has a host copy. |
| `--enable-dsv4-hicache-host-dedup`, `--enable-dsa-hicache-host-dedup`, `--enable-mla-hicache-host-dedup` | **bench** Store host pages once per node instead of once per TP rank, for DeepSeek-V4, GLM-5.2 (DSA) and Kimi-K2.6 (MLA). Without it every rank allocates its own host pool. For GLM-5.2 and Kimi-K2.6 in symmetric mode the GPUs of a node share the loaded pages through copy engines, with `--hicache-io-backend direct --hicache-mem-layout page_first_direct` (set automatically). |
| `--kv-events-config JSON` | Publish KV store and remove events over ZMQ for the KV directory. |
| `--enable-metrics`, `--enable-cache-report` | Prometheus metrics and per-request cached-token counts. The router and planner read both. |

### Peer KV pulls

A prefill worker can fetch a prefix that another worker holds in host memory instead of
recomputing it. Enable this on every worker and give the router the servicer addresses
(`--kv-worker-map`, below).

| Flag | Description |
|---|---|
| `--enable-peer-kv-servicer`, `--peer-kv-servicer-port N` | **bench** Serve this worker's host pages to peers. |
| `--hicache-storage-backend dynamic --hicache-storage-backend-extra-config JSON` | **bench** Load the peer KV backend (`decoupled_engine.srt.mem_cache.storage.peer_kv.peer_kv_backend`). |
| `--peer-kv-servicer-reserve-ratio-kv`, `--peer-kv-servicer-max-leased-gb` | **bench** Host memory set aside for pulls. |

### Router support

| Flag | Description |
|---|---|
| `--enable-prefill-work-snapshot` | **bench** Report each request's remaining prefill work. Needed by the router's cost-function P selection. |
| `--disaggregation-late-d choose`, `--disaggregation-late-d-control-port N` | **bench** Late decode choice: P asks the router which D to admit at its first prefill chunk. |
| `--sched-trace-dir DIR` | Write scheduler step traces for the [planner](dynamic-pd.md). |
| `--enable-kv-session-retention` | [KV retention](kv-retention.md). Not on DeepSeek-V4. |

### Environment

The engine keeps SGLang's `SGLANG_*` variables. Set these yourself:

| Variable | Description |
|---|---|
| `SGLANG_HOST_IP` | Address Mooncake and the KV events advertise. Set it when the default route is not reachable from the other workers. |
| `SGLANG_DECODE_SELECTION_METRICS=1` | Report decode capacity, needed by late D's `fit_req` rule. Without it every late D choice falls back to the simpler `d1` rule. |
| `WITH_NVIDIA_PEERMEM=0` | Register GPU memory with RDMA through dma-buf. Set it on hosts without the `nvidia_peermem` module, such as GB300 nodes and gVisor sandboxes. |

The flags above turn on the variables their features need. The engine logs them in one
`Symmetric defaults:` line at startup. Set a variable to `0` to keep it off.

| When | Variables turned on |
|---|---|
| `--disaggregation-mode symmetric` with `--disaggregation-late-d alloc` or `choose` | `SGLANG_SYMMETRIC_PD_IMPORT_LOCAL_RESERVE=1`, `SGLANG_SYMMETRIC_PD_IMPORT_BUDGET_FRACTION=1.0` |
| `--disaggregation-mode symmetric` | `SGLANG_PREBUILT_NONBLOCKING_H2D`; with `--enable-peer-kv-servicer` `SGLANG_PEER_KV_PUSH_FAST`; with `--enable-hierarchical-cache` on plain TP `SGLANG_HICACHE_PIPELINED_SYNC` and `SGLANG_HICACHE_SKIP_IDLE_SYNC` |
| Symmetric with `--disaggregation-late-d choose` | `SGLANG_SYMMETRIC_IMPORT_L2_RESTORE`, `SGLANG_SYMMETRIC_SKIP_REUSED_PREFIX_PUBLICATION`, `SGLANG_SYMMETRIC_PUBLISH_AFTER_RESULT`, `SGLANG_SYMMETRIC_BUDGET_STANDARD_PD`, `SGLANG_SYMMETRIC_LAUNCH_THREAD`, `SGLANG_SYMMETRIC_ADMIT_AFTER_LAUNCH`, `SGLANG_HICACHE_DECODE_CLEAR_CONSUMER`; with `--enable-kv-session-retention` also `SGLANG_KV_SESSION_ADMIT_TAG` |
| `--enable-prefill-work-snapshot` | `SGLANG_PREFILL_ROUTING_SNAPSHOT`, in symmetric mode `SGLANG_SYMMETRIC_ROUTING_SNAPSHOT`; snapshot size `SGLANG_PREFILL_ROUTING_SNAPSHOT_MAX_RECORDS=256`, `SGLANG_PREFILL_ROUTING_SNAPSHOT_MAX_BYTES=98304`, `SGLANG_LOAD_SNAPSHOT_SLOT_BYTES=131072` |
| Symmetric with `--enable-dsa-hicache-host-dedup` or `--enable-mla-hicache-host-dedup` | `SGLANG_DSA_HOST_DEDUP_FANOUT=ce_peer` or `SGLANG_MLA_HOST_DEDUP_FANOUT=ce_peer`, and the direct host layout. Set the variable to `nccl` for the NCCL broadcast with `--hicache-io-backend kernel --hicache-mem-layout page_first`. |

A variable turned on this way that the model or the scheduler loop does not support, such as
the L2 restore on DeepSeek-V4, is turned off again with a log line.

## Router (`sgl-router`)

### Required

| Flag | Description |
|---|---|
| `--model-id NAME` | The served model name. The router serves one model. |
| `--tokenizer-path PATH` | Local `tokenizer.json` or a Hugging Face repo id. |
| `--policy symmetric_pd` | Pick a P and a D for every request over symmetric workers. |
| `--symmetric-pd-force-split` | Always use two workers (never run both legs on one). The prepared path requires it. |
| `--symmetric-pd-worker URL,BOOTSTRAP_PORT,P_WEIGHT,D_WEIGHT` | One per worker. The weights set the starting role: `1,0` prefill, `0,1` decode; `0` excludes the worker from that leg. |
| `EVOLVING_INFRA_PREPARED_CONFIG=FILE` | Prepared-path contracts (environment variable). See [Prepared path](#prepared-path-config). |

### KV directory and peer pulls

| Flag | Description |
|---|---|
| `--kv-indexer-url URL` | The KV directory's query URL (`query_url` in the indexer's `ready.json`). Without it, placement uses loads and role weights only. |
| `--kv-worker-map JSON` | Maps indexer instance IDs to worker URLs: `[{"id": 20001, "url": "...", "servicer": "host:port"}]`. `servicer` is the worker's peer KV servicer, needed for pulls. |
| `--symmetric-pd-pull-hint` | **bench** Tell P to pull a prefix from the worker that holds it. |
| `--kv-pull-hint-min-incremental-tokens N` | **bench** Pull only when it saves at least N tokens of prefill (bench: 512). |
| `--symmetric-pd-pull-from-any-holder` | **bench** Allow pulls from any worker, including D-role ones (for example a former P after a role switch). |

### P and D selection

| Flag | Description |
|---|---|
| `--symmetric-pd-p-select cache-first\|cost-function` | `cache-first` (default): the worker with the best cache hit unless it is overloaded. `cost-function` (**bench**): the lowest predicted wait plus compute plus pull cost. See [Placement](placement.md). |
| `--co-schedule-trade --co-schedule-uncached-balance --co-schedule-cost-trace FILE` | **bench** Required by `cost-function`. The trace file must not exist yet. |
| `--co-schedule-trade-copy-lambda X` | **bench** Price per pulled token of the second copy a pull creates (bench: 4.6). |
| `--symmetric-pd-d-select score\|round-robin\|cache-first` | How D is picked when late D is off, and the fallback D when it is on. Bench: `round-robin`. |
| `--symmetric-pd-load-poll-ms N` | How often the router polls each worker's load (bench: 100). |
| `--pd-late-decode choose --pd-late-decode-rule fit_req --pd-late-decode-fit-native --pd-native-decode-state` | **bench** Late decode choice: pick D when P reaches its first prefill chunk, among the D workers whose free KV fits the request. Applies to `/v1/completions`; a native `/generate` request gets its D at arrival, and `/v1/chat/completions` is refused. |

### Dynamic P/D and retention

| Flag | Description |
|---|---|
| `--symmetric-pd-external-role-control` | Accept role changes on `PUT /symmetric_pd/roles`. Needed by the [planner](dynamic-pd.md). |
| `EVOLVING_INFRA_ROLE_CONTROL_TOKEN_FILE=FILE` | Token the role API requires (environment variable; at least 32 characters). |
| `--symmetric-pd-role-drain-timeout-secs N` | How long a worker moving from decode to prefill waits for its decode legs before moving them elsewhere (default 5). |
| `--symmetric-pd-kv-session-retention` | [KV retention](kv-retention.md). Needs the engine flag too. |

### Timeouts

| Flag | Description |
|---|---|
| `--request-timeout-secs`, `--stale-request-timeout-secs` | Agent requests with long contexts can run for many minutes. The benchmarks use 3600. |

<a id="prepared-path-config"></a>

## Prepared path

The frontend and the router share one contract per model configuration: page size, EAGLE or not,
model path and cache namespace must match on both sides and match the engines.

Router (`EVOLVING_INFRA_PREPARED_CONFIG`):

```json
{"callback_url": "http://router:30900/symmetric_pd/events", "max_inflight": 4096,
 "contracts": {"dsv4": {"page_size": 256, "is_eagle": false, "model_path": "/models/dsv4", "cache_namespace": null}}}
```

Frontend (`python -m evolving_infra.router.frontend --config FILE`):

| Key | Description |
|---|---|
| `config_id` | The contract name in the router config. |
| `model_path`, `page_size`, `is_eagle`, `cache_salt` | Must match the contract (`cache_salt` matches `cache_namespace`). |
| `tokenize_endpoint` | An engine's `/tokenize`. |
| `targets` | Every engine `/generate` URL the frontend may send to, mapped to the `config_id`. |
| `hash_mode` | `dynamo`: hash pages the way the KV directory does. |
| `callback_targets` | Must contain the router's `callback_url`. |
| `context_length`, `max_staging_bytes`, `ttl_seconds` | Input size limit, memory for staged inputs, and how long an unused prepared input is kept. |
| `request_timeout_seconds` | Limit on one request including its whole streamed output (default 7200). |

Run one frontend per client process.

## KV directory (`evolving_infra.router.indexer`)

One JSON file: `model_name`, `block_size` (the engine page size), `eagle`, the indexer and query
ports, and one entry per worker with its `instance_id` (the `id` in `--kv-worker-map`) and the
engine's KV event endpoints. The [quickstart](quickstart.md#2-start-the-kv-directory) has a
complete example.

## Planner

See [Dynamic P/D](dynamic-pd.md#configuration).
