//! Compatibility namespace for existing symmetric_pd callers.
//! Implementation now lives in scheduler and backend.
//! Re-exports the public scheduler API so older imports remain valid; this
//! namespace creates no independent scheduler, backend state, or protocol.
pub mod placement;
pub use crate::scheduler::*;
