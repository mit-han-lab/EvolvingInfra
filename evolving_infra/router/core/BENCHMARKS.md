# Router microbenchmarks

CPU-only Criterion benchmarks of routing-decision latency. They need no GPU and
catch algorithmic regressions in the routing primitives; end-to-end throughput
needs a GPU deployment and is measured with `benchmarks/` at the repository root.

```bash
cd evolving_infra/router/core
cargo bench --bench tree_lookup   -- --sample-size 30 --measurement-time 3
cargo bench --bench policy_select -- --sample-size 30 --measurement-time 3
```

- `tree_lookup`: insert and prefix match on the cache-aware `HashTree` over
  pre-hashed `i64` block hashes. The block hashing step (`compute_block_hashes`)
  is upstream of the tree and not measured; in real traffic it dominates the
  tree walk.
- `policy_select`: `round_robin`, `random` and `power_of_two` selection over 4 to
  256 workers. All three are O(1) in the worker count.

Reference points from a quick run (M1 laptop, release profile, reduced sample
size, so only the order of magnitude is meaningful): `hashtree_insert/128`
about 21.5 us, `hashtree_match_prefix/w64_bpw128_q64` about 47 ns,
`round_robin` about 2.5 ns at every worker count. Nothing gates on these numbers.
