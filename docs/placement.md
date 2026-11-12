# Placement

For every request the router (`--policy symmetric_pd`) picks a prefill worker (P) and a decode worker
(D) from the symmetric pool. P computes the prefill and publishes KV chunk by chunk; D reads the
chunks over RDMA while the prefill runs, then decodes.

```text
 request ──▶ KV directory query ──▶ pick P ──▶ pull hint? ──▶ pick D (now, or at P's first chunk)
              (who holds which        (cache-first or     (P fetches a prefix     (round-robin, or
               prefix blocks)          cost-function)      from a peer)            late D fit_req)
```

## KV directory

Engines publish an event whenever they store or remove a KV block (`--kv-events-config`). The
indexer feeds these events to Dynamo's KV indexer. For each request the router queries it with
the prompt's page hashes and gets, per worker, how many prefix tokens it holds on GPU and in host
memory.

The directory is advisory. If it is down or slow (`--kv-indexer-timeout-ms`), the router places
on load and role weights alone. The engine still checks that the pages are there before using
them.

## Picking P

Only workers with a prefill weight above 0 are candidates.

**`cache-first`** (the default without a KV directory). Take the worker with the longest
cached prefix, unless the workers' active prefill counts are badly imbalanced, in which case take
the least-loaded one. Below a hit ratio of `--symmetric-pd-cache-threshold` (0.5), take the
least-loaded worker.

**`cost-function`** (the default with a KV directory and servicers). The placement cost
function estimates, in tokens of prefill work, when the request would finish on each candidate
and takes the lowest:

```text
cost(P) = backlog(P) + uncached(P) + R · max(0, fetch_time − backlog(P) / R) + λ · fetched(P)
```

- `backlog(P)`: prefill work already routed to P and not finished. With `--prefill-wait tokens`
  the router counts it from its own reservations and each P's progress reports
  (`--enable-prefill-work-snapshot`). With `--prefill-wait forecast` (the default) it is `R`
  times the P's forecast wait (below).
- `uncached(P)`: what P still has to compute for this request after any fetch.
- `fetched(P)`: tokens P would fetch from a peer, the best holder's prefix minus P's own. Below
  `--kv-pull-hint-min-incremental-tokens` there is no fetch.
- The fetch time (`--fetch-fixed-ms`, `--fetch-ms-per-100k-tokens`) is charged only where P's
  queue does not hide it. `R` (`--prefill-rate-tokens-per-s`) converts seconds to tokens
  (default 13,500 tokens/s).
- `λ` (`--copy-price`) prices the second copy the fetch leaves in P's host memory, which takes
  space other prefixes could use.

Ties go to the worker with the longest own prefix. With `--placement-trace FILE` the router writes
every decision, with each candidate's terms, to a JSONL file (schema `placement-trace.v1`).

**Prefill completion forecast** (`--prefill-wait forecast`). The router runs a step-time model
of each P's scheduler on the state the engine reports (its waiting queue, the running pass and
recent pass timings) and predicts when the request would start and finish there. Prefill time
is then not assumed proportional to tokens, which matters for image inputs. A request leaves the
worker that holds its cache only when the predicted gain exceeds `--move-margin` times the
forecast's error bound, taken from an error table (built in; `--forecast-error-table` replaces
it). The built-in table covers engines with 8192-token chunks and 64-token pages. A P that cannot
have a forecast (another chunk or page size, no routing snapshot) is placed exactly as under
`--prefill-wait tokens`, and the router logs a warning naming the reason; with no forecast on
any P the decision is the token-count decision. A P with a transient gap (a report older than
`--max-state-age-ms`, no fit yet) keeps the token count and does not take a request away from
the worker that holds its cache while other P have forecasts.

**Global queue** (`--global-queue`). A request whose cache hit binds it to one P (its best hit
exceeds the second best by at least one engine chunk) is placed at arrival. Other requests wait
in one router queue, in HRRN order, until their cheapest P is about to have room: its queued
work divided by `R` is within one chunk plus the measured time from a release to the engine's
first report of the request. With `--global-queue-room forecast` a P also has room while its
forecast drain time is within that lead. A request is held at most 300 s.

The `--symmetric-pd-preset paper` default turns on the forecast and the global queue with the
cost function. The paper's runs used `--prefill-wait tokens --global-queue=false`; the GLM-5.2 and
DeepSeek-V4-Flash example configs pin these settings.

### Names in the code

| Term here | Code | Earlier name |
|---|---|---|
| Placement cost function | `scheduler/placement_cost.rs`, `PlacementCostParams` | trade |
| Prefill completion forecast | `forecast/`, `prefill_forecast.rs` | e3 |
| Global queue | `global_queue.rs`, `scheduler/cost_function/queue.rs` | gwq |
| Token-count wait (`--prefill-wait tokens`) | `PrefillWait::Tokens`, `prefill_work.rs` (`PrefillWorkBook`) | ledger |

## Peer pulls

With `--symmetric-pd-pull-hint`, when another worker holds a longer prefix in host memory than the
chosen P, the router adds a `kv_pull_hint` to the P leg. P then fetches those pages from the
other worker's peer KV servicer over RDMA instead of recomputing them. The servicer addresses
come from `--kv-worker-map`. With `--symmetric-pd-pull-from-any-holder`, D-role workers can be pull
sources too.

## Picking D

`--symmetric-pd-d-select` picks D at arrival: `round-robin` (benchmarks), `score`, or `cache-first`.

**Late decode choice** (`--pd-late-decode choose`, benchmarks) delays the choice. The request
goes to P and to every D candidate at once. When P reaches its first prefill chunk it asks the
router which D to admit, and the router picks, by the `fit_req` rule, the D with the fewest
requests among those whose free KV memory fits the request. The other candidates drop their
copy. If no answer arrives in time, P admits the arrival pick. Waiting until the first chunk
means the choice uses D's memory state just before KV starts to move, not when the request
arrived, which can be minutes earlier under load.

Late D needs `--disaggregation-late-d choose` on the engines and `SGLANG_DECODE_SELECTION_METRICS=1`
for the free-memory reports.

## Request paths

- **Prepared path** (the benchmarks). The client sends the request to a frontend, which
  tokenizes and hashes it once (`POST /prepare`). The router gets only the hashes and lengths
  (`POST /symmetric_pd/prepared`) and returns a plan; the frontend uploads the input to both legs
  (`POST /forward_pd`). Use it through the [client SDK](api-reference.md#client-sdk).
- **Direct**. `/generate` and `/v1/completions` on the router; the router sends both legs itself.
- **External ingress**. `POST /symmetric_pd/place` returns a placement without dispatching it, for a
  gateway that sends the legs on its own ([API reference](api-reference.md#post-symmetric_pdplace)).

`sgl-router --help` lists the other P selection rules and options.
