# KV directory (indexer)

`evolving_infra.router.indexer` integrates the existing Dynamo indexer
(`ai-dynamo==1.4.0`, installed by the `[indexer]` extra) with the Router through
three processes: an event bridge per worker (`kv_event_bridge.py`), the native
`python -m dynamo.indexer`, and an EAGLE hash-query adapter
(`eagle_indexer_query.py`). `services.py` supervises all of them.

The directory answers where a prefix has been observed. The Router decides
where to execute and which cache source to request; the engine confirms page
availability and owns transfer safety. A directory hit is not a transfer lease.

## Running

```bash
pip install -e ".[indexer]"
python -m evolving_infra.router.indexer --config deployment.json --output new-log-directory
```

[dynamo-services.example.json](dynamo-services.example.json) shows the
deployment file: each worker gives an instance ID, event/replay endpoints, two
unique local bridge ports and an optional topic. This command does not install
Dynamo or start an engine. Check `ready.json` and supervisor liveness together:
readiness is historical evidence once the supervisor has exited.

`fastapi` is pinned below 0.142 because 0.142 enables OpenTelemetry export at
startup, which makes the frontend exit when it runs in the same environment.

## Semantics

The Router queries the adapter at `query_port`. `"eagle"` names the engines'
page hashing: EAGLE overlapping windows (`true`) or plain pages (`false`). The
adapter hashes `token_ids` with that form and forwards prepared `block_hashes`
unchanged, both to Dynamo's `/query_by_hash`; Dynamo's own `/query` takes
`token_ids` only and answers the prepared path with HTTP 422. `ready.json`
records the `query_url` to use.

`eagle_indexer_query.py` accepts either `token_ids` or prepared `block_hashes`,
never both. The prepared path preserves signed 64-bit Dynamo hash bits and calls
the same `query_by_hash` endpoint without tokenization or rehashing. The client
Backend validates the frontend's model, namespace, page size and EAGLE hash
contract before querying. Storage event hashes and query hashes must not be
assumed interchangeable.

Native Dynamo keeps storage, event ingestion, namespace and hash-query matching.
The bridge converts the salted event representation and replay topic framing; it
rejects unknown schemas and exits on an unrepaired sequence gap. A supervisor
failure makes the deployment unhealthy and the caller must stop traffic; the
Rust query client's transient-failure fallback alone does not establish
directory freshness. There is no silent directory rebuild or automatic engine
restart.

The engines' KV event publisher must be enabled before cache entries are
generated. Registration and health alone do not prove cache correctness: a
deployment preflight should generate a real prefix, observe its hit, flush, and
observe removal before measured traffic.

The indexer runs as a separate process on the same machine as the Router. This
directory holds event/query adapters and a supervisor, not a new implementation
of Dynamo's directory.
