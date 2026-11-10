# sgl-router

The Rust router of Evolving Infra. It serves one model over a pool of engine workers and, under
`--policy symmetric_pd`, picks a prefill worker and a decode worker for every request. It also serves
the P/D role API the planner uses, the prepared path the frontend uses, and the session registry
for KV retention. It started from SGLang's router; see [PROVENANCE.md](PROVENANCE.md).

## Build

From `router/core/`. The toolchain comes from `rust-toolchain.toml`; `protoc` must be on `PATH`
or set in `PROTOC`.

```bash
cargo build --release --locked       # binary: target/release/sgl-router
cargo test --locked
```

The binary keeps the upstream name `sgl-router` for CLI compatibility.

## Run

Two symmetric workers, one starting as P and one as D, with the KV directory:

```bash
EVOLVING_INFRA_PREPARED_CONFIG=prepared.json sgl-router --host 0.0.0.0 --port 30900 \
  --model-id deepseek-v4-flash --tokenizer-path /models/dsv4/tokenizer.json \
  --policy symmetric_pd --symmetric-pd-force-split \
  --symmetric-pd-worker http://10.0.0.1:30000,8998,1,0 \
  --symmetric-pd-worker http://10.0.0.2:30000,8998,0,1 \
  --kv-indexer-url http://10.0.0.3:8091 \
  --kv-worker-map '[{"id": 20001, "url": "http://10.0.0.1:30000"}, {"id": 20002, "url": "http://10.0.0.2:30000"}]'
```

`--tokenizer-path` takes a local `tokenizer.json` or a Hugging Face repo id; without it the router
downloads the tokenizer of `--model-id`. `EVOLVING_INFRA_PREPARED_CONFIG` is needed only for the
prepared path.

The router also keeps upstream's plain policies (`round_robin`, `power_of_two`,
`cache_aware_zmq`, ...) with `--worker-urls` or Kubernetes discovery (`--service-discovery`,
`--selector`), but Evolving Infra uses only `symmetric_pd`.

## Documentation

- [Placement](../../docs/placement.md): how P and D are chosen, peer pulls, late decode choice.
- [Configuration](../../docs/configuration.md#router-sgl-router): the flags the benchmarks use.
- [API reference](../../docs/api-reference.md): HTTP endpoints.
- [BENCHMARKS.md](BENCHMARKS.md): CPU microbenchmarks of the routing primitives.

`sgl-router --help` lists every flag.

## License

Apache-2.0.
