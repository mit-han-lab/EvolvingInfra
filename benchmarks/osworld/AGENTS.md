# benchmarks/osworld

Read [`README.md`](README.md) first. Rules for running and changing this workload:

- **The rows are Kimi-K2.6's.** Token ids and image placeholders match that model only. For
  another model run a binding script; do not edit rows by hand.
- **Screenshots must be on the replay host.** `make_shards.py --image-root` writes their
  absolute path into the shards; a wrong path fails the replay.
- **16 client processes, one shard each**, as for Putnam; the `osworld` benchmark of
  `examples/gb300/kimi-k26.yaml` runs them.
- **The router needs `tokenizer.json`**, which the model does not ship: export it with
  `scripts/export_tokenizer.py` before the first start. Without it the router does not start.
- **The `e3` wait model needs its engine side**: `engine_env` in the config makes every
  prefill worker push its state to the router's port. Change the port in both places.
- **Its settings differ from the other two workloads**: planning window 300 s, router wait
  model `e3`. With the Claude Code settings the planner decides half as often.
- **Throughput is flat across concurrency** (4.25 to 4.35 requests/s from 128 to 160 agents):
  the system is saturated on prefill and the planner goes to 8P1D. Read a change in time to
  first token, not only in requests/s.
