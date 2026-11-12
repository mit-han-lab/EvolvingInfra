# Forecast fixtures: provenance

The goldens are frozen regression data for the prefill completion forecast in `src/forecast/`.
They come from a Python reference model of the forecast (not included in this repository). Tests:
`src/forecast/golden_tests.rs`, `src/prefill_forecast.rs`, `src/prefill_state_push.rs` and
`src/scheduler/cost_function.rs`.

Worker addresses recorded on the original cluster are replaced by the documentation address
`198.51.100.1` (RFC 5737). The replacement has the same length, so no value, offset or float
changed. The provenance strings `fit_cells` (`bound-params.json`), `source`
(`kimi-k26-fit.json`) and `source.cell` (`routing-snapshot-recorded.json`) name the runs
neutrally; nothing reads them. The schema strings name the public terms (`forecast.v2`,
`forecast-error-bound.v2`, `forecast-error-table.v2`); every number is the reference output. The
hashes below are of the committed files.

| Fixture | sha256 | Bytes |
|---|---|---:|
| `bound-params.json` | `b400d808fec64937d90852063973ba2e07fa4d0bf276f17e0374e9c1c91287d0` | 52,642 |
| `bound-tables.json` | `64dbbfdc5812e603d31c2a3d54ef2df765ba696ebecd3175cb2e64a2e358cb8a` | 115,607 |
| `fit-goldens-subset.jsonl` | `976cd0731f41000db9ef64c9c8bb68eb16cf5c11af76e1ec64f542a856f065f9` | 1,174,271 |
| `goldens-subset.jsonl` | `1dbc1e3ce787f5b96e596469457b33b2c1aa560c2bbbc0ce83c3a710b558dc9f` | 2,362,120 |
| `kimi-k26-fit.json` | `b733564e17f588d845f980db632f2a65a0b69c22c87f039250e06f7e284a1fbe` | 1,618 |
| `prefill-state-push-recorded.json` | `8351eec0502d8f1f2976425f36145d8f85a3043427946a6a4d7f7992a5c2421a` | 1,828 |
| `prefill-state-push-recorded.msgpack` | `246118d9a2f7cd296296a4011c8d8a4f0065517d959a10a7de7bbc4a43d5d9cf` | 59,036 |
| `prefill-state-push-oversize.msgpack` | `62a910b56469f480d847ad3e227969d159666cb31c0496ca067512d69c3bf6e2` | 845 |
| `routing-snapshot-recorded.json` | `43569ccb7c9da0e086b4a8c775a16943122ca08e8aa9441ae66769ba22f9f9ed` | 74,582 |
| `synth-goldens.jsonl` | `458da2880c1a391bca2e268b273dcc8f894ded8083e60440072992f2935a9534` | 70,223 |

## Goldens

`goldens-subset.jsonl` (600 forecasts) and `fit-goldens-subset.jsonl` (156 fits) are a
deterministic, stratified subset of the full reference goldens (8,880 forecasts, 445 fits)
computed from 8 replay runs of a 3P6D deployment on GB300; each line is one line of the source
file. The subset covers every forecast mode and every fit function (`launch_rows`,
`arrival_rates`, `rx_line`, `router_lag_next`, `calibrate`, ...). `synth-goldens.jsonl` holds the
67 synthetic reference cases. `bound-tables.json` (37 modes) and `bound-params.json` (the error
table: 9 deploy modes, recording the sha256 of `bound-tables.json`) are the offline error bounds.
`bound-params.json` is the router's built-in error table
(`data/forecast-error-table.json`).

The Rust port matches the fixture goldens exactly (max |diff| 0, also with
`FORECAST_GOLDENS_EXACT=1`). The full sets are not committed; `cargo test --lib
forecast::golden_tests -- --ignored` with `FORECAST_GOLDENS_DIR` set to a directory holding
`goldens.jsonl`, `fit-goldens.jsonl` and `bound-tables.json` runs them.

## Recorded routing snapshot (forecast wait)

`routing-snapshot-recorded.json` is one real router poll of a busy prefill worker (GB300, GLM-5.2,
3P6D, concurrency 90, cache-aware P): one `decision_input` record of the router prefill observer,
reduced to the decision (88,704 input tokens) and one candidate P with its raw `prefill_routing`
reply, the poll envelope and the router rows the snapshot did not show. The reply has an in-flight
pass, a partial request, 33 waiting rows (one without a published prefix) and 64 timeline samples.

## State-push payloads (`--forecast-push-port`)

`prefill-state-push-recorded.msgpack`, `prefill-state-push-oversize.msgpack` and
`prefill-state-push-recorded.json` (the values a decoder must read back) are frame 3 (the payload)
of two state-push messages. They were encoded with the engine's own `PassEvent` / `PrefillStatePush`
message structs and `msgspec.msgpack.Encoder()` (msgspec 0.21.1); no GPU run recorded them. The
snapshot is the recorded reply of `routing-snapshot-recorded.json` with the three fields the pusher
sets at send time (`sequence`, `age_s`, `stale`); the oversize message omits the snapshot
(`snapshot_omitted` `over_max_bytes`). Both payloads decode to maps with string keys only, with no
`bin`, `ext` or float32 values.

## Forecast fit (`--forecast-fit`)

`kimi-k26-fit.json` holds the forecast constants of an offline Kimi-K2.6 step model (TP4
GB300, 6P3D, concurrency 128, fit window 300-1800 s after the benchmark start), with a `source`
note. It is the same file as `examples/gb300/forecast/kimi-k26-fit.json`. The GPU class `large` is the busy launch-interval model in forecast units (ms over
`[1, new tokens, attention pairs / 1e6, host tokens, batch size]`); `small` is not identified (all
busy passes are full chunks). On the 1800-5400 s score window its forecast APE is mean 13.5 %, p50
6.7 %, p95 36.8 % (n 7,871).
