# engine

SGLang's runtime renamed to `decoupled_engine` ([`README.md`](README.md) lists what changed).
Environment variables keep the `SGLANG_` prefix. Files that say "Adapted from SGLang" in
their header are modified upstream files; keep that line.

- **Read resolved settings.** Startup resolution declares values with `declare_resolution`;
  the raw `ServerArgs` field keeps what the operator typed. Any code that decides something
  from a setting resolution can change reads `resolution_result(server_args, field)`. A raw
  read (`server_args.x`, `getattr(server_args, "x", ...)`) ignores every default the engine
  turned on, and nothing fails until that default matters.
- **Defaults of symmetric mode live in one file**:
  `decoupled_engine/srt/arg_groups/symmetric_defaults.py`. `plan_features` turns on features
  (server-argument fields), `plan` turns on switches (environment variables). To add one:
  add it there, add the expectation to `tests/engine/test_symmetric_defaults.py`, and add the
  row to `docs/configuration.md`. A value the operator set always wins. A switch that
  resolution turned on and a later startup check rejects is turned off with `refuse()`, not
  raised.
- **A new environment variable is declared in `decoupled_engine/srt/environ.py`** with a
  comment that says what it changes and what the default does.
- **Ports per engine**: two per GPU from `--disaggregation-bootstrap-port` (owner and reader
  channels, `disaggregation/symmetric/planes.py`), one per GPU from
  `--peer-kv-servicer-port`. Two engines on a host need disjoint ranges.
- **An engine without KV session retention** ignores the router's session tags and answers
  session control calls with `ended`. Do not turn that into an error: the router keeps
  retention on for every worker.
- **TP ranks must stay in step.** Scheduler code runs on every rank; a branch that one rank
  takes and another skips before a collective hangs the engine. Decide from values that are
  equal on all ranks.

Tests: `python -m pytest -q tests/engine` (no GPU; the test loads `symmetric_defaults.py` by
path). Anything on the serving path is then run with
`evolving-infra serve --config examples/gb300/glm52-1node.yaml` and one request; the engine
logs what resolution turned on in a `Symmetric features:` line.
