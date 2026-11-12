// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Evolving Infra router and retained SGLang router interfaces.
//!
//! `scheduler` owns placement policy, `backend` owns request accounting and engine
//! adaptation, and `server` exposes their HTTP APIs. Tokenizer, cache-index lookup,
//! worker discovery and proxy modules support both managed execution and retained
//! compatibility routes. Python-facing APIs live in `router/client/` and
//! `router/frontend/` of the repository; see `docs/api-reference.md`
//! for the prepared-input protocol.

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod agent_identity;
pub mod backend;
pub mod co_schedule;
pub mod config;
pub mod cost_balance;
pub mod discovery;
pub mod dynamo_prefix;
pub mod e3;
pub mod eft_cheap;
pub mod global_queue;
pub mod health;
pub mod image_layout;
pub mod image_source;
pub mod kv_hint;
pub mod late_bind;
pub mod late_fit;
pub mod load_refresh;
pub mod olp;
pub mod policies;
pub mod proxy;
pub mod scheduler;
pub mod server;
pub mod tokenizer;
pub mod trade_e3;
pub mod trade_e3_push;
pub mod workers;

/// Lowercase hex SHA-256 of `data`.
pub(crate) fn sha256_hex(data: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
