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
memory. A worker with DP attention publishes one instance per DP rank. When `--kv-worker-map`
names each instance's `dp_rank` (the launcher does), the cost function places per DP rank, with
each rank's own hit (below). Without ranks in the map the router places by worker, and the
worker's DP controller sends the request to the rank that holds the prefix.

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
- `fetched(P)`: tokens P would fetch from a peer and then not compute: the best holder's prefix,
  less the tokens P replays at the end of a pulled prefix (below), minus P's own. Below
  `--kv-pull-hint-min-incremental-tokens` there is no fetch.
- The fetch time (`--fetch-fixed-ms`, `--fetch-ms-per-100k-tokens`) is charged only where P's
  queue does not hide it. `R` (`--prefill-rate-tokens-per-s`) converts seconds to tokens
  (default 13,500 tokens/s).
- `λ` (`--copy-price`) prices the second copy the fetch leaves in P's host memory, which takes
  space other prefixes could use.

Ties go to the worker with the longest own prefix; when the cost's choice is a DP rank (below), an
exact tie among DP ranks goes to the rank with the fewest outstanding requests, then to the next
rank in turn, so cold requests spread over idle ranks. With `--placement-trace FILE` the router writes
every decision, with each candidate's terms, to a JSONL file (schema `placement-trace.v1`).

**DP attention.** A worker whose `--kv-worker-map` entries name its DP ranks is one candidate per
rank (`<url>#dp<rank>` in the logs and the trace). Each rank has its own prefix hit, its own
backlog (one progress query of the worker gives every rank's row its own entry), its own pulls and
its own servicer, so a rank that does not hold a prompt's prefix fetches it from the rank that
does, on its own worker or another. The router sends the chosen rank with the P leg as
`routed_dp_rank`; the worker's DP controller places the leg there and records it, and places by
its own rule only the legs that name no rank. A D leg runs on its import rank: the router sends it
no rank (also none that the client gave), and the engine refuses a D leg that names another rank
(HTTP 400). The prefill completion forecast keeps its state per worker, so with DP ranks in the
map the router refuses `--prefill-wait forecast` and `--global-queue-room forecast`.

A DP rank's pull source is the holder whose tokens save the most prefill time net of the time to
read them: `gain / R − fetch_time(gain + queued)`, with the decision's `R` (measured with `auto`).
The router does not see a pull finish, so `queued` is the pull work sent to that source's servicer
whose priced fetch time (from its decision, behind the pulls queued before it) has not elapsed and
whose first prefill chunk no progress report has shown. A rank that serves many pulls thus loses
ties to another holder of the same prefix, and the cost counts the same queued work in the
candidate's fetch time. A worker without ranks in the map pulls from the holder with the longest
prefix.

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

| Term here | Code |
|---|---|
| Placement cost function | `scheduler/placement_cost.rs`, `PlacementCostParams` |
| Prefill completion forecast | `forecast/`, `prefill_forecast.rs` |
| Global queue | `global_queue.rs`, `scheduler/cost_function/queue.rs` |
| Token-count wait (`--prefill-wait tokens`) | `PrefillWait::Tokens`, `prefill_work.rs` (`PrefillWorkBook`) |

## Peer pulls

With `--symmetric-pd-pull-hint`, when another worker holds a longer prefix in host memory than the
chosen P, the router adds a `kv_pull_hint` to the P leg. P then fetches those pages from the
other worker's peer KV servicer over RDMA instead of recomputing them. The servicer addresses
come from `--kv-worker-map`, one per DP rank under DP attention, so any rank reads from any other
rank's host tier. With `--symmetric-pd-pull-from-any-holder`, D-role workers can be pull
sources too, except a decode worker with DP attention in a static split: it runs no servicer
(`--disable-peer-kv`) and the map lists its ranks without one.

A model with sliding-window attention needs the window at the end of a reused prefix. The owner
keeps a prompt's window only at the prompt's tail, so a pulled prefix that ends inside a prompt
has none. With SWA branch replay (`--enable-swa-branch-replay`) P pulls the full KV only and
replays one window before the prefix's end. Without it, a P without DP attention pulls up to the
owner's deepest prefix with a window. Under DP attention a P takes a pull only of the full KV (a
model whose cache is the full KV only, or SWA branch replay); it drops any other pull hint and
computes the prefix.

Each worker reports in `/server_info` (the internal state `peer_kv_pull` of each rank) whether its
ranks take a pull and how many tokens a request recomputes at the end of a pulled prefix (the
replayed window, 256 on DeepSeek-V4.1; 0 for an exact pull). The cost function gives no pull hint
to a worker whose ranks take none, so it prices no pull there, and credits a pull with the prefix
the reader then matches: the source's prefix less that replay. A worker that does not report is
hinted as one that takes every pull without a replay.

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
