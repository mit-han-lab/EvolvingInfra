# e3 golden fixtures: provenance

Golden vectors of the Python reference implementation of E3 (`router_e3.py`, code version r5, run
under CPython 3.12.3; not part of this repository) for the Rust port in `src/e3/`. Tests:
`src/e3/golden_tests.rs`, `src/trade_e3.rs`, `src/trade_e3_push.rs` and
`src/scheduler/cost_function.rs`.

Worker addresses recorded on the original cluster are replaced by the documentation address
`198.51.100.1` (RFC 5737). The replacement has the same length, so no value, offset or float
changed. The provenance strings `fit_cells` (`bound-params.json`), `source`
(`kimi-frozen-fit.json`) and `source.cell` (`routing-snapshot-recorded.json`) name the runs
neutrally; nothing reads them. The hashes below are of the
committed files.

| Fixture | sha256 | Bytes |
|---|---|---:|
| `bound-params.json` | `45efd36a2a344d7fc1712be3ec6ae4119147f707d5bb511512c079c491e83e60` | 52,599 |
| `bound-tables.json` | `9df02c05d5d4b1334f3e35ce17bfd29302122c7f9792c02052b38175c89cf239` | 115,422 |
| `fit-goldens-subset.jsonl` | `976cd0731f41000db9ef64c9c8bb68eb16cf5c11af76e1ec64f542a856f065f9` | 1,174,271 |
| `goldens-subset.jsonl` | `937e2ca513bd353d4241b6b3224d244fd43eeb2ad9af524832a2d3d5c38a4a3a` | 2,363,320 |
| `kimi-frozen-fit.json` | `1f1399919e93120bd58a2970b73b42e376894c0726f815f1f4cc56b5c5860050` | 1,718 |
| `prefill-state-push-recorded.json` | `e45564f2fef7618afee2573a30f623b1e0ec13aebf4296fbd5cfcb841cd477d6` | 1,825 |
| `prefill-state-push-recorded.msgpack` | `246118d9a2f7cd296296a4011c8d8a4f0065517d959a10a7de7bbc4a43d5d9cf` | 59,036 |
| `prefill-state-push-oversize.msgpack` | `62a910b56469f480d847ad3e227969d159666cb31c0496ca067512d69c3bf6e2` | 845 |
| `routing-snapshot-recorded.json` | `43569ccb7c9da0e086b4a8c775a16943122ca08e8aa9441ae66769ba22f9f9ed` | 74,582 |
| `synth-goldens.jsonl` | `938e41c0079f1707d8fcbbe9fbec5f82284bc5498002e6a928e57af75ff61ff9` | 70,081 |

## Goldens

`goldens-subset.jsonl` (600 forecasts) and `fit-goldens-subset.jsonl` (156 fits) are a
deterministic, stratified subset of the full r5 goldens (8,880 forecasts, 445 fits) computed from
8 replay runs of a 3P6D deployment on GB300; each line is one line of the source file. The subset
covers every forecast mode and every fit function (`launch_rows`, `arrival_rates`, `rx_line`,
`router_lag_next`, `calibrate`, ...). `synth-goldens.jsonl` holds the 67 synthetic reference cases.
`bound-tables.json` (37 modes) and `bound-params.json` (9 deploy modes, recording the sha256 of
`bound-tables.json`) are the offline bound curves.

The reference recomputes every fixture case exactly. The Rust port matches the fixture goldens and
the full sets exactly (max |diff| 0, also with `E3_GOLDENS_EXACT=1`). The full sets are not
committed; `cargo test --lib e3::golden_tests -- --ignored` with `E3_GOLDENS_DIR` set to a
directory holding `goldens.jsonl`, `fit-goldens.jsonl` and `bound-tables.json` runs them.

## Recorded routing snapshot (TRADE wait e3)

`routing-snapshot-recorded.json` is one real router poll of a busy prefill worker (GB300, GLM-5.2,
3P6D, concurrency 90, cache-aware P): one `decision_input` record of the router prefill observer,
reduced to the decision (88,704 input tokens) and one candidate P with its raw `prefill_routing`
reply, the poll envelope and the router rows the snapshot did not show. The reply has an in-flight
pass, a partial request, 33 waiting rows (one without a published prefix) and 64 timeline samples.

## State-push payloads (TRADE wait e3 with `--trade-e3-push-port`)

`prefill-state-push-recorded.msgpack`, `prefill-state-push-oversize.msgpack` and
`prefill-state-push-recorded.json` (the values a decoder must read back) are frame 3 (the payload)
of two state-push messages. They were encoded with the engine's own `PassEvent` / `PrefillStatePush`
message structs and `msgspec.msgpack.Encoder()` (msgspec 0.21.1); no GPU run recorded them. The
snapshot is the recorded reply of `routing-snapshot-recorded.json` with the three fields the pusher
sets at send time (`sequence`, `age_s`, `stale`); the oversize message omits the snapshot
(`snapshot_omitted` `over_max_bytes`). Both payloads decode to maps with string keys only, with no
`bin`, `ext` or float32 values.

## Frozen fit (`--trade-e3-frozen-fit`)

`kimi-frozen-fit.json` is `kimi.e3_constants` of an offline Kimi-K2.6 step model (TP4 GB300, 6P3D,
concurrency 128, fit window 300-1800 s after the benchmark start), with a `source` note. The GPU
class `large` is the busy launch-interval model in E3 units (ms over `[1, new tokens, attention
pairs / 1e6, host tokens, batch size]`); `small` is not identified (all busy passes are full
chunks). On the 1800-5400 s score window its E3 APE is mean 13.5 %, p50 6.7 %, p95 36.8 % (n 7,871).