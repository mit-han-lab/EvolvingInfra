//! Compatibility path; placement belongs to the scheduler.
//! Re-exports the pure placement types/functions without changing their
//! selection semantics or adding I/O and reservation accounting.
pub use crate::scheduler::placement::*;
