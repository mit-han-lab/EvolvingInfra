//! External scheduling: policy selects per-request P/D roles on capable backends.
//! Exposes pure placement/cost extension APIs and runtime orchestration. Backend
//! remains the owner of worker observations, reservations, and execution state.
pub mod cost_function;
pub mod placement;
pub mod policy;
pub mod role_drain;
mod runtime;
pub mod trade;
pub use placement::*;
pub use policy::*;
pub use runtime::*;
