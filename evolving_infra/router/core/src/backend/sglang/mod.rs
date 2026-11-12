// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! SGLang-specific wire protocol and engine observations.
//!
//! Groups adapters for request bodies and cancellation, per-request KV exchange
//! marks, and engine-load parsing. It exports the SGLang adapter without deciding
//! placements or managing the physical KV pages inside the engine.
pub mod kv;
pub mod progress;
pub mod request;
pub use kv::{Marks, SglangBackend};
