# Configuration

Evolving Infra runs the paper's configuration by default. With `--disaggregation-mode symmetric`
on the engines and `--policy symmetric_pd` on the router, every feature below is on unless a flag
says otherwise; set `--symmetric-preset none` (engine) and `--symmetric-pd-preset none` (router)
to take the flags as given, and use the same preset on both sides. Run `sgl-router --help` or
`python -m decoupled_engine.launch_server --help` for every flag.

## Engine (`decoupled_engine.launch_server`)

The engine takes SGLang's server arguments. These are the ones Evolving Infra adds or depends on.

### Required

| Flag | Description |
|---|---|
| `--disaggregation-mode symmetric` | Any worker can run the prefill or the decode leg. |
| `--disaggregation-transfer-backend mooncake` | Move KV between workers with Mooncake over RDMA. |
| `--disaggregation-ib-device JSON` | RDMA device of every GPU, e.g. `{"0": "mlx5_0", ...}`. List all GPUs the worker uses. Peer KV pulls use the same devices. |
| `--disaggregation-bootstrap-port N` | Base port of the worker's KV transfer side channel; it uses two ports per GPU from N, so engines on one node need ranges that do not overlap. Must match the router's `--symmetric-pd-worker`. |
| `--kv-events-config JSON` | Publish KV store and remove events over ZMQ for the KV directory. |
| `--enable-metrics`, `--enable-cache-report` | Prometheus metrics and per-request cached-token counts. The router and planner read both. |
| `--incremental-streaming-output` | Required by the frontend. |

### On by default in symmetric mode

`--symmetric-preset paper` (the default) turns these on when the flags leave them off, and logs
them in one `Symmetric features:` line at startup.

| Feature | Flags it sets | Default |
|---|---|---|
| Host KV tier | `--enable-hierarchical-cache`, `--hicache-write-policy write_through` | Host tier 4× the GPU cache with host dedup (4.445× with the peer KV servicer, which keeps 10 % for pulls); `--hicache-ratio` sets it |
| Host dedup | `--enable-dsa-hicache-host-dedup` (GLM-5.2), `--enable-mla-hicache-host-dedup` (Kimi-K2.6), `--enable-dsv4-hicache-host-dedup` (DeepSeek-V4) | The model's attention kind, on single-node TP. Each node stores a host page once instead of once per GPU |
| Peer KV pulls | `--enable-peer-kv-servicer`, `--hicache-storage-backend dynamic` with the peer KV backend | On when `--disaggregation-ib-device` is set. The servicer listens on `--peer-kv-servicer-port` (default 18100) and the next ports, one per GPU; give each engine on a node its own range |
| Late decode choice | `--disaggregation-late-d choose`, `--disaggregation-late-d-lead-mode need`, `--disaggregation-late-d-allocation-trigger first_chunk` | The router-link port is a free port, reported in `/server_info` |
| Prefill work snapshot | `--enable-prefill-work-snapshot` | Read by the router's cost function |
| KV session retention | `--enable-kv-session-retention` | On models with a FULL-only cache tree, such as GLM-5.2 and Kimi-K2.6. Without it the engine ignores the router's session tags |

### Other flags

| Flag | Description |
|---|---|
| `--sched-trace-dir DIR` | Write scheduler step traces for the [planner](dynamic-pd.md). |
| `--hicache-ratio R` | Host tier size as a multiple of the GPU cache. |

### Environment

The engine keeps SGLang's `SGLANG_*` variables. Set these yourself:

| Variable | Description |
|---|---|
| `SGLANG_HOST_IP` | Address Mooncake and the KV events advertise. Set it when the default route is not reachable from the other workers. |
| `WITH_NVIDIA_PEERMEM=0` | Register GPU memory with RDMA through dma-buf. Set it on hosts without the `nvidia_peermem` module, such as GB300 nodes and gVisor sandboxes. |

The features turn on the variables they use, and the engine logs them in one
`Symmetric defaults:` line. Set a variable to `0` to keep it off.

| When | Variables turned on |
|---|---|
| Late decode (`alloc` or `choose`) | `SGLANG_SYMMETRIC_PD_IMPORT_LOCAL_RESERVE=1`, `SGLANG_SYMMETRIC_PD_IMPORT_BUDGET_FRACTION=1.0` |
| Symmetric mode | `SGLANG_PREBUILT_NONBLOCKING_H2D`, `SGLANG_PREFILL_UNSERIALIZE_LOOP`, `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1500`, `SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1500`, `SGLANG_RETRACT_BACKUP_MAX_GB=20`; with the peer KV servicer `SGLANG_PEER_KV_PUSH_FAST`; with the host tier on plain TP `SGLANG_HICACHE_PIPELINED_SYNC` and `SGLANG_HICACHE_SKIP_IDLE_SYNC`; with `--kv-events-config` `SGLANG_KV_EVENTS_LAZY_LOAD_BACK`, `SGLANG_KV_EVENTS_LAZY_FAST_BUILD`, `SGLANG_KV_EVENTS_BATCH_PUBLISH` |
| Late decode choice | `SGLANG_SYMMETRIC_IMPORT_L2_RESTORE`, `SGLANG_SYMMETRIC_SKIP_REUSED_PREFIX_PUBLICATION`, `SGLANG_SYMMETRIC_PUBLISH_AFTER_RESULT`, `SGLANG_SYMMETRIC_BUDGET_STANDARD_PD`, `SGLANG_SYMMETRIC_LAUNCH_THREAD`, `SGLANG_SYMMETRIC_ADMIT_AFTER_LAUNCH`, `SGLANG_HICACHE_DECODE_CLEAR_CONSUMER`, `SGLANG_DECODE_SELECTION_METRICS`; with retention also `SGLANG_KV_SESSION_ADMIT_TAG` |
| Prefill work snapshot | `SGLANG_PREFILL_ROUTING_SNAPSHOT`, `SGLANG_SYMMETRIC_ROUTING_SNAPSHOT`, `SGLANG_PREFILL_ROUTING_SNAPSHOT_MAX_RECORDS=256`, `SGLANG_PREFILL_ROUTING_SNAPSHOT_MAX_BYTES=98304`, `SGLANG_LOAD_SNAPSHOT_SLOT_BYTES=131072` |
| DSA or MLA host dedup | `SGLANG_DSA_HOST_DEDUP_FANOUT=ce_peer` or `SGLANG_MLA_HOST_DEDUP_FANOUT=ce_peer` with `--hicache-io-backend direct --hicache-mem-layout page_first_direct`; on GB300 also `SGLANG_DSA_HOST_DEDUP_LAYER_PARTS=8`, `SGLANG_DSA_HOST_DEDUP_PART_PLAN=hybrid`, `SGLANG_DSA_HOST_DEDUP_STREAM_REDRAW=1`; with DSA dedup `SGLANG_DSA_HOST_DEDUP_SHARED_VERIFY=0`. Set the fan-out variable to `nccl` for the NCCL broadcast with `--hicache-io-backend kernel --hicache-mem-layout page_first`. |

A variable turned on this way that the model or the scheduler loop does not support, such as
the L2 restore on DeepSeek-V4, is turned off again with a log line.

## Router (`sgl-router`)

### Required

| Flag | Description |
|---|---|
| `--model-id NAME` | The served model name. The router serves one model. |
| `--tokenizer-path PATH` | Local `tokenizer.json` or a Hugging Face repo id. |
| `--policy symmetric_pd` | Pick a P and a D for every request over symmetric workers. |
| `--symmetric-pd-worker URL,BOOTSTRAP_PORT,P_WEIGHT,D_WEIGHT` | One per worker. The weights set the starting role: `1,0` prefill, `0,1` decode; `0` excludes the worker from that leg. |
| `--kv-indexer-url URL` | The KV directory's query URL (`query_url` in the indexer's `ready.json`). |
| `--kv-worker-map JSON` | Maps indexer instance IDs to workers and their peer KV servicers: `[{"id": 20001, "url": "...", "servicer": "host:18100"}]`. |
| `EVOLVING_INFRA_PREPARED_CONFIG=FILE` | Prepared-path contracts (environment variable). See [Prepared path](#prepared-path-config). |

### On by default with `--policy symmetric_pd`

`--symmetric-pd-preset paper` (the default) sets these when the flags leave them unset. A flag
that is given wins.

| Feature | Flags it sets |
|---|---|
| Prefill placement by predicted time to first token, with peer pulls | `--symmetric-pd-p-select cost-function`, `--symmetric-pd-pull-hint`, `--symmetric-pd-pull-from-any-holder`, `--kv-pull-hint-min-incremental-tokens 512`, `--trade-load-poll-ms 100`. On when `--kv-indexer-url` is set and `--kv-worker-map` names the servicers; otherwise `cache-first` |
| Late decode choice | `--pd-late-decode choose --pd-late-decode-rule fit_req --pd-late-decode-candidates all --pd-late-decode-local off --pd-late-decode-fit-native --pd-native-decode-state` |
| KV session retention | `--symmetric-pd-kv-session-retention`; `--symmetric-pd-no-kv-session-retention` keeps it off for a model whose engine does not support it (DeepSeek-V4) |
| Placement basics | `--symmetric-pd-force-split`, `--symmetric-pd-d-select round-robin`, `--symmetric-pd-load-poll-ms 100`, `--symmetric-pd-import-cache-policy retain` (a D may become a P later, so imported KV follows the host write policy), `--kv-indexer-timeout-ms 200` |

### Tuning

| Flag | Description |
|---|---|
| `--trade-copy-lambda X\|auto` | Price per pulled token of the second copy a pull creates (default 4.6). `auto` measures it while serving; `--trade-copy-lambda-min`, `-max` and `-fallback` bound it and give the value before the first measurement. |
| `--trade-rate-tokens-per-s R\|auto` | Prefill tokens one P computes per second, which turns pull time into tokens (default 13500). `auto` measures it. |
| `--trade-pull-fixed-ms N`, `--trade-pull-ms-per-100k N` | Time of one KV pull: a fixed part (default 450 ms) and a part per 100k pulled tokens (default 100 ms). |
| `--trade-wait ledger\|e3` | Where a P's wait comes from. `ledger` (default): the router's count of uncached tokens routed to it. `e3`: a forecast from the engine's state, for a model whose prefill time is not proportional to tokens (images); needs `--trade-e3-bound-table FILE`. |
| `--trade-e3-frozen-fit FILE`, `--trade-e3-push-port PORT`, `--trade-e3-next-pass-lag`, `--trade-e3-max-view-age-ms N`, `--trade-gate-m M` | With `--trade-wait e3`: the offline constants of the forecast, the port of the engines' state push, the lag of a request's next pass, the oldest engine state the forecast uses (default 1500 ms), and how far a gain must exceed the forecast's error bound before a request leaves the worker that holds its cache. |
| `--symmetric-pd-gwq-auto`, `--symmetric-pd-gwq-room tokens\|predicted` | Hold a request that no P is bound to in a router queue until its P has room, by queued tokens or by the forecast drain time. |
| `--trade-load-poll-ms N` | How often the router reads each P's prefill progress (default 100). |
| `--trade-cost-trace FILE` | Write every placement decision to a new JSONL file. |
| `--symmetric-pd-p-select cache-first\|cost-function` | `cache-first`: the worker with the best cache hit unless it is overloaded. `cost-function`: the lowest predicted wait plus compute plus pull cost. See [Placement](placement.md). |
| `--pd-late-decode-rule fit_req\|d1` | How late decode choice picks D. Late decode choice applies to `/v1/completions` and `/v1/chat/completions`; a native `/generate` request gets its D at arrival. A chat request with `n` above 1 is refused, as a batch prompt is. |
| `--chat-tools-tokenize` | A chat request with `tools` is placed on token ids from an engine's `/v1/tokenize` (prompt length, KV directory query, pull hint). The router's chat encoder renders messages only, so without the flag such a request matches no cached prefix. The ids are not forwarded to the engine; a tokenize that fails or takes over 3 s leaves the request placed as without the flag. Off by default; the configs in `examples/agents` turn it on. |

### Dynamic P/D

| Flag | Description |
|---|---|
| `--symmetric-pd-external-role-control` | Accept role changes on `PUT /symmetric_pd/roles`. Needed by the [planner](dynamic-pd.md). |
| `EVOLVING_INFRA_ROLE_CONTROL_TOKEN_FILE=FILE` | Token the role API requires (environment variable; at least 32 characters). |
| `--symmetric-pd-role-drain-timeout-secs N` | How long a worker moving from decode to prefill waits for its decode legs before moving them elsewhere (default 5). |

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

## Launcher config

`evolving-infra serve --config <file>` reads one YAML file ([`examples/`](../examples/README.md)).
The launcher detects GPUs and RDMA NICs, lays out the ports and passes the presets; the file
holds what it cannot detect.

| Key | Meaning | Default |
|---|---|---|
| `model` | model directory, the same path on every host; holds `tokenizer.json` | required |
| `served_model_name` | the model name clients send | the directory's name, lower case |
| `page_size` | KV block size of engine, router and KV directory | 64 |
| `context_length` | maximum prompt length | 1048576 for the frontend |
| `gpus_per_worker` | TP size of one worker | all GPUs of a host; half of them on a single host |
| `prefill_workers` | workers that start as prefill | half of the workers |
| `retention` | KV session retention in the router; `false` for a model whose cache tree is not FULL-only | `true` |
| `tokenizer` | the `tokenizers` file the router reads | `<model>/tokenizer.json` |
| `engine_env` | environment variables of every engine | none |
| `frontend` | settings added to the tokenizer frontend's config: `image_config` for image input, body and staging limits | none |
| `engine_args`, `router_args` | flags appended to every engine and to the router | `--trust-remote-code`; none |
| `planner` | planner arguments ([Dynamic P/D](dynamic-pd.md)); used with `--planner` | none |
| `base` | another config file to start from; `engine_args` replace the base's flags of the same name | none |

`examples/slurm/submit.py` also reads `benchmarks` (a map of name to replay: `workload`,
`concurrency`, `duration`, `warmup`, `prefill_workers`; `processes` and `shards` for several
client processes; `router_args`, `engine_args` and `planner` added to the served config for
that benchmark) and `slurm` (site settings).

When the stack is ready the first host writes `READY` and `stack.json` (router and tokenizer
URLs, model, block size, GPU and worker counts) to its work directory;
`benchmarks/replay/run_trie.py stack=<work dir>` reads it.
