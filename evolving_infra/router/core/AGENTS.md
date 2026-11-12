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
- **Late decode choice applies to `/v1/completions` and `/v1/chat/completions`.** Both go
  through `dispatch` in `src/server/routes/completions.rs`; the chat handler hands its request
  over when late decode choice is on (`Ingress::Chat`). A native `/generate` request gets its D
  at arrival (`Ingress::late_d_choose`). The engines answer a released candidate with HTTP 409
  on both routes (`late_d_error_response` in the engine's `serving_base.py`).
- **Role control with KV session retention needs the prepared path.** With
  `--symmetric-pd-external-role-control` and retention on, a plain `/v1/completions` request is
  refused: the router needs the frontend's lifecycle callbacks to count work per session.
- **Placement constants** are named in `src/scheduler/placement_cost.rs`
  (`DEFAULT_COPY_PRICE`); [`docs/paper.md`](../../../docs/paper.md) points at them. Change
  the doc with the constant. The built-in forecast error table is
  `data/forecast-error-table.json`, the same file as `tests/fixtures/forecast/bound-params.json`.
- **A flag that only makes sense with `--policy symmetric_pd`** is listed in the check that
  rejects it under other policies (search `symmetric_pd_import_cache_policy.is_some()`).
