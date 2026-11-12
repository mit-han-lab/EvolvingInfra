//! Backend owns node state and engine adaptation; scheduler owns placement.
//!
//! Constructs the shared registry, synchronized request ledger and SGLang adapter,
//! then starts background engine-load polling. Placement decisions remain in
//! `crate::scheduler`; this module exposes their engine-facing dependencies.
pub mod execution;
pub mod kv_sessions;
pub mod refresh;
pub mod registry;
pub mod sglang;
pub mod state;
pub mod workload;

use self::{registry::BackendRegistry, sglang::SglangBackend, state::BackendState};
use crate::{config::SymmetricPdImportCachePolicy, discovery::ModelId, workers::WorkerRegistry};
use parking_lot::Mutex;
use std::sync::Arc;
#[cfg(test)]
use std::time::Instant;
use tokio::sync::Notify;

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("cannot resolve KV side-channel endpoints for {worker}: {reason}")]
    Endpoints { worker: String, reason: String },
}

/// Shared state for all requests. The SGLang adapter implements the currently
/// supported engine protocol; a future vLLM adapter belongs beside it, not in
/// scheduler policies. There is no duplicate Python request ledger.
pub struct Backend {
    pub registry: BackendRegistry,
    pub state: Mutex<BackendState>,
    pub adapter: SglangBackend,
    pub changed: Arc<Notify>,
    pub kv_sessions: Option<Arc<kv_sessions::KvSessions>>,
}
impl Backend {
    /// Create the shared backend and start its engine-load refresh task.
    ///
    /// # Inputs
    /// `registry` and `model` identify the available engines; `import_cache_policy`
    /// controls D-side KV retention. `poll_ms` sets the refresh period (at least
    /// 100 ms); `per_request_progress` selects per-RID remaining-work accounting.
    ///
    /// # Outputs
    /// Returns an `Arc<Backend>` with an empty ledger and configured adapter.
    /// Starts polling when a Tokio runtime is present; otherwise logs a warning
    /// and returns the backend without a refresh task.
    #[allow(clippy::too_many_arguments)] // one call site per policy; a builder adds nothing
    pub fn new(
        registry: Arc<WorkerRegistry>,
        model: ModelId,
        import_cache_policy: SymmetricPdImportCachePolicy,
        poll_ms: u64,
        per_request_progress: bool,
        kv_session_retention: bool,
        kv_session_key_file: Option<&std::path::Path>,
        unknown_reconcile: Option<state::UnknownReconcile>,
    ) -> Arc<Self> {
        let changed = Arc::new(Notify::new());
        let mut ledger = BackendState::with_progress(Arc::clone(&changed), per_request_progress);
        ledger.set_unknown_reconcile(unknown_reconcile);
        let backend = Arc::new(Self {
            registry: BackendRegistry::new(registry, model),
            state: Mutex::new(ledger),
            adapter: SglangBackend {
                import_cache_policy,
            },
            changed,
            kv_sessions: kv_session_retention
                .then(|| kv_sessions::KvSessions::spawn(kv_session_key_file)),
        });
        refresh::spawn(
            &backend,
            poll_ms,
            per_request_progress,
            unknown_reconcile.is_some(),
        );
        backend
    }
    #[cfg(test)]
    /// Inject an aggregate engine-load observation for a test.
    ///
    /// # Inputs
    /// `self` is the backend under test; `worker` names the engine. `uncached`,
    /// `running` and `waiting` are measured counts; `sent_at` orders the sample.
    ///
    /// # Outputs
    /// Returns `()` after updating the locked ledger and notifying waiters,
    /// unless the ledger rejects the sample as older than an applied poll.
    pub fn apply_poll(
        &self,
        worker: &str,
        uncached: u64,
        running: u64,
        waiting: u64,
        sent_at: Instant,
    ) {
        self.state.lock().apply_poll(
            worker,
            state::LoadObservation {
                uncached_tokens: Some(uncached),
                running: Some(running),
                waiting: Some(waiting),
            },
            sent_at,
        );
    }
}
