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

**`cache-first`** (default). Take the worker with the longest cached prefix, unless the
workers' active prefill counts are badly imbalanced, in which case take the least-loaded one.
Below a hit ratio of `--symmetric-pd-cache-threshold` (0.5), take the least-loaded worker.

**`cost-function`** (benchmarks). Estimate, in tokens of prefill work, when the request would
finish on each candidate and take the lowest:

```text
cost(P) = backlog(P) + uncached(P) + R · max(0, pull_time − backlog(P) / R) + λ · pulled(P)
```

- `backlog(P)`: prefill work already routed to P and not finished, tracked by the router from
  its own reservations and each P's progress reports (`--enable-prefill-work-snapshot`).
- `uncached(P)`: what P still has to compute for this request after any pull.
- `pulled(P)`: tokens P would pull from a peer, the best holder's prefix minus P's own. Below
  `--kv-pull-hint-min-incremental-tokens` there is no pull.
- The pull time is charged only where P's queue does not hide it. `R` converts seconds to tokens
  (default 13,500 tokens/s).
- `λ` (`--trade-copy-lambda`) prices the second copy the pull leaves in P's host
  memory, which takes space other prefixes could use.

Ties go to the worker with the longest own prefix. The router writes every decision, with each
candidate's terms, to the `--trade-cost-trace` file.

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
