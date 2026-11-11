# Provenance

`evolving_infra/router/core/` is derived from SGLang's experimental `sgl-router` (Apache-2.0)
at upstream base `9a489f8d2f`, imported through a private SGLang fork and then
developed for this project. The import kept the Rust workspace and its
`sgl-kv-indexer` path dependency. Original SPDX and "SGLang Authors" notices are
kept; `LICENSE` here is the Apache-2.0 text of the imported repository.

The binary and crate names (`sgl-router`, `sgl_router`, `sgl-kv-indexer`) are
kept for CLI compatibility. Building and running needs no SGLang or engine
source checkout. The `sgl-kv-indexer` crate is a build dependency only:
deployments that use Dynamo run the pinned Dynamo indexer (`evolving_infra/router/indexer/`),
not a second cache directory.

The upstream directory did not track a lockfile; `Cargo.lock` was resolved
once for this router and is committed and frozen, so build with `--locked`.
`rust-toolchain.toml` pins the compiler (1.90).
