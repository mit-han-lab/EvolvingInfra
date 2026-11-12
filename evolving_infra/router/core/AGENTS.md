# evolving_infra/router/core

The Rust router (`sgl-router`). [`README.md`](README.md) has the build;
[`docs/placement.md`](../../../docs/placement.md) says how it picks P and D.

- **Build and test**: `cargo test --locked` from this directory (unit tests in `src/`, proxy
  tests with mock workers in `tests/proxy/`). `protoc` must be on `PATH` or in `PROTOC`.
- **The paper's configuration is `apply_symmetric_pd_preset`** in `src/config/cli.rs`. It
  fills only what the operator left unset. A feature of the paper's configuration goes there,
  with an assertion in `symmetric_pd_preset_paper_is_the_default` and a row in
  `docs/configuration.md`.
- **The test helper `into_config` adds `--symmetric-pd-preset none`** unless the arguments
  name a preset, so most tests see flags exactly as given. A test of a default passes
  `--symmetric-pd-preset paper` itself.
- **Engine and router presets match.** With late decode choice on in the router and off in an
  engine, the router refuses that worker at registration.
- **Late decode choice applies to `/v1/completions`.** A native `/generate` request gets its D
  at arrival (`Ingress::late_d_choose` in `src/server/routes/completions.rs`). With late decode
  choice on (`--pd-late-decode choose`, the `paper` preset's default under
  `--policy symmetric_pd`), `/v1/chat/completions` returns HTTP 501 `late_d_route_unsupported`
  (`src/server/routes/chat.rs`). Chat completions are served when late decode choice is off on
  the router (`--pd-late-decode off`) and on the engines (`--disaggregation-late-d=off`).
- **Role control with KV session retention needs the prepared path.** With
  `--symmetric-pd-external-role-control` and retention on, a plain `/v1/completions` request is
  refused: the router needs the frontend's lifecycle callbacks to count work per session.
- **Placement constants** are named in `src/scheduler/co_schedule.rs`
  (`DEFAULT_TRADE_COPY_LAMBDA`); [`docs/paper.md`](../../../docs/paper.md) points at them. Change
  the doc with the constant.
- **A flag that only makes sense with `--policy symmetric_pd`** is listed in the check that
  rejects it under other policies (search `symmetric_pd_import_cache_policy.is_some()`).
