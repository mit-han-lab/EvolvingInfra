# DSpark controlled accept ratio under strict replay

## Scope

DecoupledEngine supports a synthetic-oracle dial that controls the
DSpark speculative draft-accept ratio during strict replay while preserving
byte-exact output commitment:

```
--speculative-algorithm DSPARK --synthetic-oracle-dspark-accept-rate R
```

with `R ∈ (0, 1]`. Per forced-output row and verify round, the engine samples
an injected-correct count `k` from a truncated geometric whose chain
probability `p` solves `Σ_{i=1..γ} p^i = R·γ`, overwrites the first
`min(k, eligible)` draft positions with the recorded trace tokens, and
corrupts the next position so the trace-authoritative verify accepts exactly
the sampled count. Injection happens after the confidence computation and the
ragged-verify planner (the adaptive schedule keeps acting on the real draft)
and before the target-verify forward (committed KV is written from the
injected tokens, as a natural full hit would). Implementation:
`evolving_infra/engine/decoupled_engine/srt/speculative/dspark_components/dspark_synthetic_oracle.py`.

Guard rails: DSPARK-only; mutually exclusive with `SGLANG_SIMULATE_ACC_LEN`
and `--speculative-adaptive`; oracle servers reject non-greedy requests at
admission, so every client (this one included) must send `temperature=0`,
`top_p=1`, `top_k=1` — which `strict_replay=true` already does.

## Running an arm with this client

```
TRIE_SKIP_ROW_VALIDATION=1 trie \
  workload_path=<schema-v2 replay jsonl> \
  endpoint=http://<engine>:30000/v1 \
  model=<model id> api_key=EMPTY \
  concurrency=16 duration=300 replay_once=true drain_timeout=7200 \
  stream=true strict_replay=true context_source=recorded \
  cache_salt_mode=session prefix_block_size=64 workload_order=natural \
  seed=<fixed> delay_scale=0 max_retries=0 timeout=3600 \
  output_json=<run dir>/trie-result.json \
  lifecycle_output_jsonl=<run dir>/lifecycle/trie.jsonl \
  lifecycle_run_id=<arm> lifecycle_arm=off lifecycle_cell_id=<arm> \
  lifecycle_concurrency=16
```

- `TRIE_SKIP_ROW_VALIDATION=1` is safe only for sha-attested corpora (per-row
  Draft2020-12 validation costs hours on multi-GB workloads).
- Size `drain_timeout` to the longest trace, not the average: a 1210-turn
  trace needs ~2h to drain at c=16; a 3600 s ceiling truncates it mid-flight
  (reported as a cancellation, not a mismatch — but the arm loses trace
  coverage).
- Byte-exactness verdict: `failed_requests == 0` in `trie-result.json`
  (fail-closed; the client verifies returned token IDs against the trace).
- A fresh engine's first 1-3 completions streams can hit a transport-level
  RemoteProtocolError once; the client re-admits the trace. Expect one retry
  burst at arm start or prewarm with a throwaway request.

The `/v1/completions` surface does not expose `meta_info`, so accept-rate
accounting needs a second pass with a `/generate` client sending
`input_ids` + `forced_output_token_ids` and reading
`meta_info.spec_synthetic_oracle_stats` per request:
`num_eligible_drafts`, `num_expect_drafts`, `num_correct_drafts`,
`num_verify_rounds`, `cap_bind_ct`.

## Reading the numbers — denominators matter

Never quote an accept rate without its denominator:

| metric | definition |
|---|---|
| realized R (full-γ) | `Σ correct / (Σ rounds · γ)` — canonical |
| realized R (eligible) | `Σ correct / Σ eligible`, `eligible = clamp(remaining−1, 0, γ)` |
| uncapped R̂ (pre-cap) | `Σ expect / Σ eligible` — the sampler's rate before the adaptive cap |
| cap-bind rate | `Σ cap_bind / Σ rounds` — rounds where the confidence cap trimmed the sampled count |

Two statistical properties of the design (measured on a 2,773-request ×
2.58M-token validation matrix):

1. **Truncation bias.** `min(k, eligible)` on narrow tail rounds biases the
   eligible-width rate up and the full-γ rate down by O(1e-4..1e-3),
   deliberately uncompensated. The exact expectation is computable by a DP
   over the round structure (each round commits `injected+1` tokens, prefill
   commits 1; the identity `Σ correct + Σ rounds = Σ (len−1)` holds
   integer-exactly and validates the model).
2. **Chain variance.** Chained acceptance (`P(k≥i) = p^i`) is not
   `eligible` independent Bernoulli trials — a binomial CI understates the
   variance ~3×. Calibration checks should test observed `Σ correct` against
   the DP expectation with the DP variance, not a binomial interval.

## Determinism expectations

The oracle's uniforms are keyed on `(blake2b(forced ids), replay cursor,
depth)` — no cross-round state, no batch dependence. Measured consequences:

- **Static verify mode**: per-request `spec_accept_sequence_digest` is
  byte-identical across reruns, regardless of batch composition, concurrency,
  or KV-pool size (2,773/2,773 across four R settings).
- **Compact (adaptive) mode**: the injected count stays deterministic but the
  accepted count is `min(injected, cap)` with the cap coming from the live
  confidence head — digests legitimately differ across runs while aggregate
  counts reproduce to <0.05%. Do not gate compact arms on digest equality.
- **Retraction**: a retraction re-prefill commits one token outside any
  verify round, shifting the round lattice — the digest changes on that
  request while byte-exactness and per-round invariants hold. Separate
  retraction-shifted digests (nonzero `num_retractions`) from real
  mismatches before declaring nondeterminism.

## Interplay with the compact schedule

Pre-cap `uncapped R̂` stays calibrated at all R while the live schedule
trims: on the validation corpus the cap-bind rate rose 9.6% → 61.6% as R
went 0.25 → 1.0, and at R=1.0 (every draft correct) the schedule still
admitted only ~67% of eligible draft width — the dial and the schedule are
observable independently, which is the point of exporting both `expect` and
`correct`.
