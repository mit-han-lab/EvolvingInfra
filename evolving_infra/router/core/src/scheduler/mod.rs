// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! External scheduling: policy selects per-request P/D roles on capable backends.
//! Exposes pure placement/cost extension APIs and runtime orchestration. Backend
//! remains the owner of worker observations, reservations, and execution state.
pub mod cost_function;
pub mod placement;
pub mod placement_cost;
pub mod policy;
pub mod role_drain;
mod runtime;
pub use placement::*;
pub use policy::*;
pub use runtime::*;
