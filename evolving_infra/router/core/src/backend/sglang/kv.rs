//! SGLang symmetric KV side-channel adapter. Engines own actual page lifetimes.
//!
//! Resolves worker side-channel endpoints and creates matching export/import
//! metadata for one execution attempt. This is protocol construction only: actual
//! KV movement, ownership and reclamation remain inside SGLang.
use crate::backend::BackendError;
use crate::scheduler::placement::Placement;
use crate::{config::SymmetricPdImportCachePolicy, workers::Worker};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    OnceLock,
};

pub const KV_EXPORT_KEY: &str = "kv_export";
pub const KV_IMPORT_KEY: &str = "kv_import";
pub(super) const BOOTSTRAP_KEYS: [&str; 3] = ["bootstrap_host", "bootstrap_port", "bootstrap_room"];

/// Independent of transfer IDs: rebuilding a plan must not reuse its attempt.
/// Randomize the process's sequence on restart, then allocate atomically across
/// all adapter instances. Zero is reserved by the engine's legacy protocol.
/// Allocate a nonzero process-local KV exchange attempt identifier.
///
/// # Inputs
/// No arguments. Uses a lazily initialized, randomized atomic counter
/// shared by all SGLang adapter instances in this process.
///
/// # Outputs
/// Returns the next nonzero `u64`; advances the counter atomically and
/// skips zero on wraparound. Random initialization reduces restart collisions.
fn next_attempt_id() -> u64 {
    static COUNTER: OnceLock<AtomicU64> = OnceLock::new();
    let counter = COUNTER.get_or_init(|| AtomicU64::new(rand::random()));
    loop {
        let id = counter.fetch_add(1, Ordering::Relaxed);
        if id != 0 {
            return id;
        }
    }
}

#[derive(Debug)]
pub struct SglangBackend {
    pub import_cache_policy: SymmetricPdImportCachePolicy,
}

/// Per-request marks, ready to embed into the two leg bodies.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Marks {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_export: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_import: Option<Value>,
}

impl SglangBackend {
    /// Encode the scheduler-selected deferred CPU return; no placement or I/O.
    /// The existing transfer/attempt IDs identify this command. Peer addresses
    /// are rank-0 endpoints; the engine derives each rank's port by offset.
    pub fn add_host_return(
        &self,
        marks: &mut Marks,
        destination_servicer_addr: &str,
        source_servicer_addr: &str,
        timeout_ms: u64,
        max_tokens: u64,
    ) {
        if let Some(import) = marks.kv_import.as_mut() {
            import["host_return"] = json!({
                "destination_servicer_addr": destination_servicer_addr,
                "source_servicer_addr": source_servicer_addr,
                "timeout_ms": timeout_ms,
                "max_tokens": max_tokens,
            });
        }
    }

    /// Rank-0 `(owner_addr, reader_addr)` of a worker (see module docs).
    /// Resolve the worker's KV ownership and read side-channel addresses.
    ///
    /// # Inputs
    /// `worker` supplies advertised symmetric endpoints or a bootstrap host
    /// and port; `self` carries no endpoint-specific mutable state.
    ///
    /// # Outputs
    /// Returns `(owner_addr, reader_addr)`. Advertised addresses take priority;
    /// fallback uses the bootstrap port and port + 1. Returns `BackendError`
    /// when the fallback port is absent or reader-port arithmetic overflows.
    pub fn endpoints(&self, worker: &Worker) -> Result<(String, String), BackendError> {
        let info = worker.symmetric_pd();
        if let (Some(o), Some(r)) = (info.owner_addr, info.reader_addr) {
            return Ok((o, r));
        }
        let host = worker.bootstrap_host();
        let Some(port) = worker.bootstrap_port() else {
            return Err(BackendError::Endpoints {
                worker: worker.url.clone(),
                reason: "no /server_info symmetric_pd section and no bootstrap_port configured"
                    .to_string(),
            });
        };
        let reader = port.checked_add(1).ok_or_else(|| BackendError::Endpoints {
            worker: worker.url.clone(),
            reason: "bootstrap_port + 1 overflows".to_string(),
        })?;
        Ok((
            format!("tcp://{host}:{port}"),
            format!("tcp://{host}:{reader}"),
        ))
    }

    /// Build the two marks for a placement (empty for Local).
    /// Construct paired KV export/import metadata for a placement attempt.
    ///
    /// # Inputs
    /// `placement` provides the transfer ID; `p_worker` and `d_worker` identify
    /// the exchange peers; `prompt_len` is the expected imported token count.
    /// `self.import_cache_policy` determines D-side retention metadata.
    ///
    /// # Outputs
    /// Returns empty marks for a placement without a transfer ID. Otherwise
    /// returns paired marks sharing a new attempt ID, or an endpoint-resolution
    /// error. Constructs metadata only and does not initiate KV transfer.
    pub fn marks(
        &self,
        placement: &Placement,
        p_worker: &Worker,
        d_worker: &Worker,
        prompt_len: u64,
    ) -> Result<Marks, BackendError> {
        let Some(transfer_id) = placement.transfer_id else {
            return Ok(Marks::default());
        };
        let (owner_addr, _) = self.endpoints(p_worker)?;
        let (_, reader_addr) = self.endpoints(d_worker)?;
        let cache_policy = self.import_cache_policy_for(d_worker);
        let attempt_id = next_attempt_id();
        Ok(Marks {
            kv_export: Some(json!({
                "transfer_id": transfer_id,
                "attempt_id": attempt_id,
                "reader_worker": d_worker.url,
                "reader_addr": reader_addr,
            })),
            kv_import: Some(json!({
                "transfer_id": transfer_id,
                "attempt_id": attempt_id,
                "owner_worker": p_worker.url,
                "owner_addr": owner_addr,
                "expected_tokens": prompt_len,
                "cache_policy": cache_policy,
            })),
        })
    }

    /// Late D choose on symmetric workers (stage 2): the P-leg's
    /// `kv_export {transfer_id, candidates}` (each candidate with its
    /// `cand_idx`, reader worker, reader endpoint and its own attempt id; no
    /// scalar reader fields) and, per candidate in `cand_idx` order, its
    /// `kv_import` with the same attempt id. Engine contract:
    /// `disaggregation/symmetric/marks.py::parse_export_mark`.
    ///
    /// # Outputs
    /// `(kv_export, imports)`, or an endpoint error. Builds metadata only.
    pub fn late_marks(
        &self,
        transfer_id: u64,
        p_worker: &Worker,
        candidates: &[std::sync::Arc<Worker>],
        prompt_len: u64,
    ) -> Result<(Value, Vec<Value>), BackendError> {
        let (owner_addr, _) = self.endpoints(p_worker)?;
        let mut cands = Vec::with_capacity(candidates.len());
        let mut imports = Vec::with_capacity(candidates.len());
        for (cand_idx, d_worker) in candidates.iter().enumerate() {
            let (_, reader_addr) = self.endpoints(d_worker)?;
            let attempt_id = next_attempt_id();
            cands.push(json!({
                "cand_idx": cand_idx,
                "reader_worker": d_worker.url,
                "reader_addr": reader_addr,
                "attempt_id": attempt_id,
            }));
            imports.push(json!({
                "transfer_id": transfer_id,
                "attempt_id": attempt_id,
                "owner_worker": p_worker.url,
                "owner_addr": owner_addr,
                "expected_tokens": prompt_len,
                "cache_policy": self.import_cache_policy_for(d_worker),
            }));
        }
        Ok((
            json!({"transfer_id": transfer_id, "candidates": cands}),
            imports,
        ))
    }

    /// `kv_import.cache_policy` for a D-leg on `d_worker`
    /// (`--symmetric-pd-import-cache-policy`; engine side:
    /// `disaggregation/symmetric/import_queue.py apply_import_cache_policy`).
    /// Choose the SGLang wire value for imported D-side cache retention.
    ///
    /// # Inputs
    /// `self` supplies the configured retention policy; `d_worker` supplies
    /// its current role preference when the configured policy is `Auto`.
    ///
    /// # Outputs
    /// Returns `retain`, `gpu_only` or `drop`. Auto selects `gpu_only` for a
    /// worker with no prefill weight and `retain` otherwise; no state is changed.
    fn import_cache_policy_for(&self, d_worker: &Worker) -> &'static str {
        match self.import_cache_policy {
            SymmetricPdImportCachePolicy::Retain => "retain",
            SymmetricPdImportCachePolicy::GpuOnly => "gpu_only",
            SymmetricPdImportCachePolicy::Drop => "drop",
            SymmetricPdImportCachePolicy::Auto => match crate::backend::registry::role_of(d_worker) {
                // Never a P-leg here: a host copy would only cost write-through
                // bandwidth and host pool space (upstream PD decode keeps the
                // prompt on GPU only).
                Some((_, w)) if w.prefill <= 0.0 => "gpu_only",
                _ => "retain",
            },
        }
    }
}
