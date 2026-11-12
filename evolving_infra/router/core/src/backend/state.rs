//! Authoritative router reservations plus engine observations.
//!
//! An observation is optional: a missing/stale engine is never reported as an
//! idle engine. The runtime uses the aggregate estimate: engine-reported load
//! plus the reservations that no poll covers yet. The per-RID mode
//! ([`BackendState::with_progress`]) replaces each request's initial estimate
//! with its observed remaining work; missing/truncated snapshots preserve
//! estimates.
pub use crate::cost_balance::PrefillProgress;
use crate::scheduler::placement::{WorkerLoad, WorkerLoads};
use parking_lot::Mutex;
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashMap},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::Notify;

const POLL_STALENESS_LIMIT: Duration = Duration::from_secs(5);
const LEDGER_ENTRY_TTL: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct LoadObservation {
    pub uncached_tokens: Option<u64>,
    /// Engine-reported running requests; may include both P and D work.
    pub running: Option<u64>,
    pub waiting: Option<u64>,
}

#[derive(Debug)]
pub struct EngineLoadSample {
    pub observation: LoadObservation,
    pub timestamp: Option<f64>,
    pub prefill_work: Option<PrefillProgress>,
}

#[derive(Debug)]
struct LedgerEntry {
    tokens: u64,
    prompt_tokens: u64,
    rid: Option<String>,
    remaining: AtomicU64,
    seen_progress: AtomicBool,
    reserved_at: Instant,
    dispatched_at: Mutex<Option<Instant>>,
    retired: AtomicBool,
    expires: bool,
    unknown: AtomicBool,
    /// Unknown-entry reconciliation: the send time of the first progress reply
    /// of an unbroken run of replies that did not list this RID.
    absent_since: Mutex<Option<Instant>>,
}
impl LedgerEntry {
    /// Determine whether a reservation still contributes to backend accounting.
    ///
    /// # Inputs
    /// `self` supplies the retirement flag, lease mode and reservation age.
    ///
    /// # Outputs
    /// Returns true unless explicitly retired or an expiring legacy lease has
    /// reached its TTL. Managed requests do not expire solely because time passes.
    fn live(&self) -> bool {
        !self.retired.load(Ordering::Relaxed)
            && (!self.expires || self.reserved_at.elapsed() < LEDGER_ENTRY_TTL)
    }
}

/// Explicitly retired on response/cancellation, not when a streaming handler
/// returns. A cloned handle addresses the same reservation; retirement is idempotent.
#[derive(Debug, Clone)]
pub struct LedgerHandle {
    entry: Arc<LedgerEntry>,
    changed: Arc<Notify>,
}
impl LedgerHandle {
    /// The uncached tokens this reservation counted at placement.
    pub fn tokens(&self) -> u64 {
        self.entry.tokens
    }
    /// Record the first time this reservation was submitted to an engine.
    ///
    /// # Inputs
    /// `self` shares the underlying ledger entry with all cloned handles.
    ///
    /// # Outputs
    /// Returns `()` and initializes `dispatched_at` only if absent. Repeated
    /// calls preserve the original timestamp; they do not retire the reservation.
    pub fn mark_dispatched(&self) {
        let mut at = self.entry.dispatched_at.lock();
        if at.is_none() {
            *at = Some(Instant::now());
        }
    }
    /// Create a rollback guard for a reservation not yet submitted to an engine.
    ///
    /// # Inputs
    /// `self` is the reservation to protect during request construction.
    ///
    /// # Outputs
    /// Returns a guard holding a cloned handle. Dropping the guard retires the
    /// reservation unless `PendingReservation::mark_dispatched` disarmed it; the
    /// caller must disarm this guard when submission begins.
    pub fn pending_guard(&self) -> PendingReservation {
        PendingReservation(Some(self.clone()))
    }
    /// A failed transport is not a terminal engine response. Retain the
    /// reservation and expose uncertainty until an explicit terminal event.
    /// Retain uncertain engine work after an inconclusive transport outcome.
    ///
    /// # Inputs
    /// `self` references the reservation whose completion cannot be confirmed.
    ///
    /// # Outputs
    /// Returns `()` after setting its uncertainty flag. Managed reservations remain
    /// counted until retirement; legacy leases still expire at their deadline.
    /// This flag does not cancel engine work or extend a lease.
    pub fn mark_unknown(&self) {
        self.entry.unknown.store(true, Ordering::Relaxed);
    }
    /// Whether the reservation is unknown (see [`Self::mark_unknown`]).
    pub fn is_unknown(&self) -> bool {
        self.entry.unknown.load(Ordering::Relaxed)
    }
    /// Remove a reservation from future load views after terminal evidence.
    ///
    /// # Inputs
    /// `self` references the shared reservation to retire.
    ///
    /// # Outputs
    /// Returns `()` after atomically marking the entry retired. The first call
    /// notifies scheduling waiters; subsequent calls are harmless. Physical ledger
    /// pruning happens when the worker is next viewed or sampled.
    pub fn retire(&self) {
        if !self.entry.retired.swap(true, Ordering::Relaxed) {
            self.changed.notify_waiters();
        }
    }
    /// Whether the reservation was retired (terminal evidence, rollback or
    /// the unknown reconciliation of [`BackendState::set_unknown_reconcile`]).
    pub fn is_retired(&self) -> bool {
        self.entry.retired.load(Ordering::Relaxed)
    }
}

/// Retirement of unknown P reservations from engine evidence. An unknown entry (transport error, a cancel or an abort without
/// terminal evidence) is retired when the progress replies of its worker have
/// not listed its RID, in an unbroken run of applied, advancing,
/// non-truncated replies, for `empty_queue` while the worker also reported an
/// empty waiting queue the whole time (a queued request is not listed until
/// its first chunk launches), or for `quiet` whatever the queue (`None` =
/// never; the bound of the cost book's `--symmetric-pd-unknown-expire-secs`). A
/// listed RID is never retired; a worker without applied replies keeps its
/// entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownReconcile {
    pub empty_queue: Duration,
    pub quiet: Option<Duration>,
}

/// Default empty-queue absence before an unknown P reservation is retired.
pub const UNKNOWN_RECONCILE_EMPTY_QUEUE: Duration = Duration::from_secs(10);

impl Default for UnknownReconcile {
    fn default() -> Self {
        Self {
            empty_queue: UNKNOWN_RECONCILE_EMPTY_QUEUE,
            quiet: Some(Duration::from_secs(
                crate::config::types::default_symmetric_pd_unknown_expire_secs(),
            )),
        }
    }
}

/// One D-leg this router dispatched to a worker and that has not finished: the
/// decode counterpart of the SGLang Model Gateway's per-worker load (incremented
/// at dispatch, decremented when the request ends), so requests still waiting
/// for KV space on the D are counted. Counted from placement until dropped.
#[derive(Debug)]
pub struct DecodeReservation {
    entry: Arc<LedgerEntry>,
    changed: Arc<Notify>,
}
impl Drop for DecodeReservation {
    /// End this D-leg's contribution to its worker's in-flight count.
    ///
    /// # Inputs
    /// `self` owns the only handle to the decode entry.
    ///
    /// # Outputs
    /// Returns `()` after marking the entry retired and waking placement waiters;
    /// the entry is pruned from the worker's decode ledger on its next view.
    fn drop(&mut self) {
        if !self.entry.retired.swap(true, Ordering::Relaxed) {
            self.changed.notify_waiters();
        }
    }
}

/// Rolls back only while the caller has not submitted any engine request.
/// Once dispatched, dropping a handler must not imply engine completion.
pub struct PendingReservation(Option<LedgerHandle>);
impl PendingReservation {
    /// Disarm automatic rollback when the engine request is submitted.
    ///
    /// # Inputs
    /// `self` is the pending guard still owning an optional reservation handle.
    ///
    /// # Outputs
    /// Returns `()` after recording dispatch and removing the guarded handle.
    /// Dropping this guard afterward does not imply engine completion.
    pub fn mark_dispatched(&mut self) {
        if let Some(handle) = self.0.take() {
            handle.mark_dispatched();
        }
    }
}
impl Drop for PendingReservation {
    /// Roll back a reservation whose submission guard was never disarmed.
    ///
    /// # Inputs
    /// `self` may still contain an undispatched reservation handle.
    ///
    /// # Outputs
    /// Returns `()`; retires that handle if present and otherwise does nothing.
    /// The guard must be disarmed before submission so handler cancellation cannot
    /// incorrectly release running engine work.
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            handle.retire();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationStatus {
    Missing,
    Fresh,
    Stale,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackendLoadView {
    pub observation: Option<LoadObservation>,
    pub status: ObservationStatus,
    pub sample_age_ms: Option<u64>,
    pub reserved_uncached_tokens: u64,
    pub active_prefill_reqs: u64,
    /// Live decode reservations: D-legs dispatched here and not yet finished.
    pub active_decode_reqs: u64,
    pub unknown_prefill_reqs: u64,
    pub progress_observed_reqs: u64,
    pub workload_mode: &'static str,
    /// Unknown P reservations this worker's reconciliation retired so far
    /// (external role control only; see [`UnknownReconcile`]).
    pub unknown_retired_reqs: u64,
}

#[derive(Debug, Default)]
struct WorkerLoadState {
    observation: Option<LoadObservation>,
    sampled_at: Option<Instant>,
    engine_observed_at: Option<Instant>,
    engine_timestamp: Option<f64>,
    progress_cursor: Option<(String, u64)>,
    ledger: Vec<Arc<LedgerEntry>>,
    /// D-legs in flight on this worker (see [`DecodeReservation`]); kept apart
    /// from the prefill ledger so P-side counts and backlog are unchanged.
    decode_ledger: Vec<Arc<LedgerEntry>>,
    /// The send time of the last applied reply that did not report an empty
    /// waiting queue (unknown reconciliation).
    queue_busy_at: Option<Instant>,
    /// Unknown P reservations retired by reconciliation.
    unknown_retired: u64,
}
impl WorkerLoadState {
    /// Build one worker's observable load view and prune dead ledger entries.
    ///
    /// # Inputs
    /// `self` contains engine observations and reservations; `per_request`
    /// selects per-RID remaining work instead of aggregate-plus-reservation mode.
    ///
    /// # Outputs
    /// Returns observation freshness, reserved tokens and active/uncertain
    /// request counts. Per-RID mode counts every live remaining estimate; aggregate
    /// mode counts work not covered by a poll and work with uncertain outcomes.
    /// Retired or expired entries are removed; unobserved work is preserved.
    fn view(&mut self, per_request: bool) -> BackendLoadView {
        self.ledger.retain(|e| e.live());
        self.decode_ledger.retain(|e| e.live());
        let age = self.engine_observed_at.map(|t| t.elapsed());
        BackendLoadView {
            observation: self.observation,
            status: match age {
                None => ObservationStatus::Missing,
                Some(a) if a < POLL_STALENESS_LIMIT => ObservationStatus::Fresh,
                Some(_) => ObservationStatus::Stale,
            },
            sample_age_ms: age.map(|a| a.as_millis() as u64),
            reserved_uncached_tokens: self
                .ledger
                .iter()
                .filter(|e| {
                    // Unsatisfied uploads and uncertain engine requests remain
                    // explicit reservations; an aggregate poll cannot prove admission.
                    per_request
                        || e.unknown.load(Ordering::Relaxed)
                        || e.dispatched_at
                            .lock()
                            .is_none_or(|sent| self.sampled_at.is_none_or(|sample| sent > sample))
                })
                .map(|e| {
                    if per_request {
                        e.remaining.load(Ordering::Relaxed)
                    } else {
                        e.tokens
                    }
                })
                .fold(0u64, u64::saturating_add),
            active_prefill_reqs: self.ledger.len() as u64,
            active_decode_reqs: self.decode_ledger.len() as u64,
            progress_observed_reqs: self
                .ledger
                .iter()
                .filter(|e| e.seen_progress.load(Ordering::Relaxed))
                .count() as u64,
            workload_mode: if per_request {
                "rid_remaining"
            } else {
                "aggregate_estimate"
            },
            unknown_prefill_reqs: self
                .ledger
                .iter()
                .filter(|e| e.unknown.load(Ordering::Relaxed))
                .count() as u64,
            unknown_retired_reqs: self.unknown_retired,
        }
    }

    /// Unknown reconciliation on one applied, advancing progress reply sent
    /// at `sent_at` (see [`UnknownReconcile`]). Retires and logs each due
    /// entry; returns how many it retired.
    fn reconcile_unknown(
        &mut self,
        worker: &str,
        progress: &PrefillProgress,
        waiting: Option<u64>,
        sent_at: Instant,
        rc: UnknownReconcile,
    ) -> usize {
        if waiting != Some(0) {
            self.queue_busy_at = Some(sent_at);
        }
        let mut retired = 0;
        for entry in &self.ledger {
            if !entry.unknown.load(Ordering::Relaxed) || !entry.live() {
                continue;
            }
            let Some(rid) = entry.rid.as_deref() else {
                continue;
            };
            let mut absent = entry.absent_since.lock();
            if progress.truncated || progress.remaining.contains_key(rid) {
                *absent = None;
                continue;
            }
            let since = *absent.get_or_insert(sent_at);
            let quiet = sent_at.saturating_duration_since(since);
            let queue_clear = self.queue_busy_at.is_none_or(|busy| busy < since);
            let reason = if queue_clear && quiet >= rc.empty_queue {
                "unknown_absent_with_empty_queue"
            } else if rc.quiet.is_some_and(|q| quiet >= q) {
                "unknown_absent_quiet"
            } else {
                continue;
            };
            drop(absent);
            entry.retired.store(true, Ordering::Relaxed);
            retired += 1;
            tracing::warn!(target: "symmetric_pd", worker, rid, reason, absent_s = quiet.as_secs_f64(),
                queue_clear, uncached_tokens = entry.tokens,
                "unknown P reservation retired: its worker has not listed the RID; it leaves the ledger");
        }
        self.unknown_retired += retired as u64;
        retired
    }
}

/// Kept under one Backend mutex. Scheduler must hold that mutex from snapshot
/// through selecting and reserving; it must never hold it across network I/O.
#[derive(Debug)]
pub struct BackendState {
    workers: HashMap<String, WorkerLoadState>,
    changed: Arc<Notify>,
    per_request: bool,
    /// Retire unknown P reservations from engine progress; `None` keeps
    /// every unknown entry until terminal evidence.
    unknown_reconcile: Option<UnknownReconcile>,
}
impl BackendState {
    /// Create an empty ledger using aggregate engine-load accounting.
    ///
    /// # Inputs
    /// `changed` is the notification channel used to wake placement waiters.
    ///
    /// # Outputs
    /// Returns an empty `BackendState` with per-request progress disabled.
    /// No network polling is started by this constructor.
    pub fn new(changed: Arc<Notify>) -> Self {
        Self {
            workers: HashMap::new(),
            changed,
            per_request: false,
            unknown_reconcile: Some(UnknownReconcile::default()),
        }
    }

    /// Set the unknown reconciliation (see [`UnknownReconcile`]; on by
    /// default). Its
    /// evidence is the `prefill_work` section of the load polls, which the
    /// refresh task then requests (backend `new`).
    pub fn set_unknown_reconcile(&mut self, rc: Option<UnknownReconcile>) {
        self.unknown_reconcile = rc;
    }

    /// Create an empty ledger with the selected prefill-work accounting mode.
    ///
    /// # Inputs
    /// `changed` wakes placement waiters; `per_request` enables remaining-work
    /// correction by request ID instead of the legacy aggregate estimate.
    ///
    /// # Outputs
    /// Returns an empty state configured for the selected mode. All entries
    /// are still owned here; enabling progress does not create another counter.
    pub fn with_progress(changed: Arc<Notify>, per_request: bool) -> Self {
        let mut state = Self::new(changed);
        state.per_request = per_request;
        state
    }

    /// Read one worker's load and reservation view.
    ///
    /// # Inputs
    /// `self` is the mutable ledger; `worker` is the worker URL or registry key.
    ///
    /// # Outputs
    /// Returns a view, creating a missing-observation entry for an unknown
    /// worker and pruning retired/expired reservations. Makes no engine request.
    pub fn view(&mut self, worker: &str) -> BackendLoadView {
        self.workers
            .entry(worker.to_owned())
            .or_default()
            .view(self.per_request)
    }

    /// Observe the existing ledger without placement, reservations or network I/O.
    /// Pruning already retired entries has the same semantics as the scheduler's
    /// normal view; this does not retire uncertain engine work.
    /// Collect load views for a chosen set of workers.
    ///
    /// # Inputs
    /// `self` is the mutable ledger; `workers` supplies worker identifiers.
    ///
    /// # Outputs
    /// Returns a sorted map from worker ID to load view, creating absent
    /// worker entries and pruning dead reservations through `view`. Unknown
    /// engine observations remain marked missing rather than measured idle.
    pub fn views(
        &mut self,
        workers: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> BTreeMap<String, BackendLoadView> {
        workers
            .into_iter()
            .map(|worker| {
                let worker = worker.as_ref();
                (worker.to_owned(), self.view(worker))
            })
            .collect()
    }

    /// Convert ledger views into the scheduling policy's load representation.
    ///
    /// # Inputs
    /// `self` selects the accounting mode; `workers` names the candidate set.
    ///
    /// # Outputs
    /// Returns `WorkerLoads`. Per-RID mode uses managed remaining-token sums;
    /// aggregate mode adds uncovered reservations to observed uncached tokens.
    /// Absent aggregate fields use the candidate set's maximum available field,
    /// or zero if none exists. Running counts may include both P and D work.
    pub fn loads(&mut self, workers: impl IntoIterator<Item = impl AsRef<str>>) -> WorkerLoads {
        let views: Vec<_> = workers
            .into_iter()
            .map(|w| (w.as_ref().to_owned(), self.view(w.as_ref())))
            .collect();
        // An unknown worker should not look less loaded merely because its
        // poll failed. These are routing estimates, not observed zero values.
        let fallback = LoadObservation {
            uncached_tokens: views
                .iter()
                .filter_map(|(_, v)| v.observation.and_then(|o| o.uncached_tokens))
                .max(),
            running: views
                .iter()
                .filter_map(|(_, v)| v.observation.and_then(|o| o.running))
                .max(),
            waiting: views
                .iter()
                .filter_map(|(_, v)| v.observation.and_then(|o| o.waiting))
                .max(),
        };
        let mut loads = WorkerLoads::new();
        for (worker, view) in views {
            let observed = view.observation.unwrap_or_default();
            loads.insert(
                worker,
                WorkerLoad {
                    backlog_uncached_tokens: if self.per_request {
                        view.reserved_uncached_tokens
                    } else {
                        observed
                            .uncached_tokens
                            .or(fallback.uncached_tokens)
                            .unwrap_or_default()
                            .saturating_add(view.reserved_uncached_tokens)
                    },
                    running_decode: observed.running.or(fallback.running).unwrap_or_default(),
                    waiting: observed.waiting.or(fallback.waiting).unwrap_or_default(),
                    active_prefill_reqs: view.active_prefill_reqs,
                    active_decode_reqs: view.active_decode_reqs,
                },
            );
        }
        loads
    }

    /// Reserve managed prefill work without a request-ID progress association.
    ///
    /// # Inputs
    /// `self` owns the ledger; `worker` selects the target; `tokens` is the
    /// initial uncached-token estimate.
    ///
    /// # Outputs
    /// Returns a shared handle to a newly counted, undispatched reservation.
    /// It has no TTL and must be retired explicitly after terminal evidence.
    pub fn reserve(&mut self, worker: &str, tokens: u64) -> LedgerHandle {
        self.reserve_inner(worker, tokens, None, tokens, false)
    }
    /// Legacy /place has no terminal callback: retain its bounded lease only.
    /// New managed execution must use reserve and explicitly retire its handle.
    /// Create the bounded reservation used by the legacy placement-only API.
    ///
    /// # Inputs
    /// `self` owns the ledger; `worker` is the selected engine; `tokens` is
    /// the uncached-token estimate to reserve.
    ///
    /// # Outputs
    /// Returns a handle with dispatch time initialized immediately and a
    /// 30-second expiry. New managed execution should use a nonexpiring
    /// reservation whose terminal callback explicitly retires it.
    pub fn reserve_lease(&mut self, worker: &str, tokens: u64) -> LedgerHandle {
        self.reserve_inner(worker, tokens, None, tokens, true)
    }
    /// Only observed RIDs can replace a prediction. No RID is an explicit
    /// conservative fallback: initial uncached work remains until P terminates.
    /// Reserve request work that can be corrected by observed prefill progress.
    ///
    /// # Inputs
    /// `worker` selects the engine; `tokens` is the initial uncached estimate;
    /// `rid` optionally identifies the engine request; `prompt_tokens` bounds
    /// valid remaining-token observations. `self` owns the ledger.
    ///
    /// # Outputs
    /// Returns a nonexpiring, undispatched handle. An absent or empty RID
    /// keeps the initial estimate until retirement; valid dispatched RIDs may
    /// receive remaining-work corrections from newer engine samples.
    pub fn reserve_request(
        &mut self,
        worker: &str,
        tokens: u64,
        rid: Option<&str>,
        prompt_tokens: u64,
    ) -> LedgerHandle {
        self.reserve_inner(worker, tokens, rid, prompt_tokens, false)
    }
    /// Insert and initialize one reservation entry in the selected worker ledger.
    ///
    /// # Inputs
    /// `worker`, `tokens`, `rid` and `prompt_tokens` describe target and work.
    /// `expires` enables a bounded legacy lease; `self` supplies shared state
    /// and the notification channel.
    ///
    /// # Outputs
    /// Returns a handle sharing the inserted entry. Empty RIDs are discarded;
    /// initial remaining work equals `tokens`. Legacy leases begin dispatched,
    /// while managed entries await an explicit dispatch acknowledgement.
    fn reserve_inner(
        &mut self,
        worker: &str,
        tokens: u64,
        rid: Option<&str>,
        prompt_tokens: u64,
        expires: bool,
    ) -> LedgerHandle {
        let entry = Arc::new(LedgerEntry {
            tokens,
            prompt_tokens,
            rid: rid.filter(|r| !r.is_empty()).map(str::to_owned),
            remaining: AtomicU64::new(tokens),
            seen_progress: AtomicBool::new(false),
            reserved_at: Instant::now(),
            dispatched_at: Mutex::new(if expires { Some(Instant::now()) } else { None }),
            retired: AtomicBool::new(false),
            expires,
            unknown: AtomicBool::new(false),
            absent_since: Mutex::new(None),
        });
        self.workers
            .entry(worker.to_owned())
            .or_default()
            .ledger
            .push(Arc::clone(&entry));
        LedgerHandle {
            entry,
            changed: Arc::clone(&self.changed),
        }
    }

    /// Count one D-leg dispatched to `worker` until the returned guard drops.
    ///
    /// # Inputs
    /// `self` owns the per-worker ledgers; `worker` is the chosen decode engine.
    ///
    /// # Outputs
    /// Returns a guard whose lifetime is the D-leg's contribution to
    /// `active_decode_reqs`. It never expires on its own and carries no tokens;
    /// the owner drops it when the D-leg completes, fails or is cancelled.
    pub fn reserve_decode(&mut self, worker: &str) -> DecodeReservation {
        let entry = Arc::new(LedgerEntry {
            tokens: 0,
            prompt_tokens: 0,
            rid: None,
            remaining: AtomicU64::new(0),
            seen_progress: AtomicBool::new(false),
            reserved_at: Instant::now(),
            dispatched_at: Mutex::new(None),
            retired: AtomicBool::new(false),
            expires: false,
            unknown: AtomicBool::new(false),
            absent_since: Mutex::new(None),
        });
        self.workers
            .entry(worker.to_owned())
            .or_default()
            .decode_ledger
            .push(Arc::clone(&entry));
        DecodeReservation {
            entry,
            changed: Arc::clone(&self.changed),
        }
    }

    /// Compatibility aggregate sample used by legacy modes and test fixtures.
    /// Apply an aggregate-only observation through the common sample pipeline.
    ///
    /// # Inputs
    /// `self` owns the ledger; `worker` identifies the engine; `observation`
    /// contains measured counts; `sent_at` is the poll's local send time.
    ///
    /// # Outputs
    /// Returns `()` after delegating a sample without engine timestamp or
    /// per-RID progress. Older polls are ignored by `apply_sample`; accepted
    /// samples update observations and notify waiters.
    pub fn apply_poll(&mut self, worker: &str, observation: LoadObservation, sent_at: Instant) {
        self.apply_sample(
            worker,
            EngineLoadSample {
                observation,
                timestamp: None,
                prefill_work: None,
            },
            sent_at,
        );
    }

    /// Reconcile a poll result with engine freshness and owned request work.
    ///
    /// # Inputs
    /// `self` owns worker state; `worker` identifies the engine; `sample`
    /// contains aggregate counts, optional engine timestamp and RID progress;
    /// `sent_at` orders concurrent polls by their local send time.
    ///
    /// # Outputs
    /// Returns `()`. Older polls are ignored. Accepted polls refresh engine age
    /// when their timestamp is absent, their snapshot time advances, or their
    /// progress advances. Updates matching dispatched RIDs only from valid new
    /// progress. Missing RIDs, restarts and impossible
    /// remaining counts never erase owned work. Prunes dead entries and notifies
    /// waiters; a zero remaining count alone does not retire a request.
    pub fn apply_sample(&mut self, worker: &str, sample: EngineLoadSample, sent_at: Instant) {
        let state = self.workers.entry(worker.to_owned()).or_default();
        if state.sampled_at.is_some_and(|at| sent_at < at) {
            return;
        }
        state.sampled_at = Some(sent_at);
        let progress_advanced = sample.prefill_work.as_ref().is_some_and(|p| {
            !p.epoch.is_empty()
                && state
                    .progress_cursor
                    .as_ref()
                    .is_none_or(|(epoch, sequence)| *epoch != p.epoch || p.sequence > *sequence)
        });
        let snapshot_advanced = sample
            .timestamp
            .is_none_or(|ts| state.engine_timestamp.is_none_or(|previous| ts > previous))
            || progress_advanced;
        // /v1/loads reads a published SHM snapshot. A successful HTTP poll of
        // the same snapshot is not evidence of fresh engine state.
        if snapshot_advanced {
            state.engine_observed_at = Some(sent_at);
            state.engine_timestamp = sample.timestamp;
            state.observation = Some(sample.observation);
        }
        state.ledger.retain(|e| e.live());
        state.decode_ledger.retain(|e| e.live());
        if progress_advanced {
            let progress = sample.prefill_work.expect("checked progress");
            if let Some(rc) = self.unknown_reconcile {
                state.reconcile_unknown(worker, &progress, sample.observation.waiting, sent_at, rc);
            }
            // Progress requested only for the reconciliation corrects nothing.
            let entries: &[Arc<LedgerEntry>] = if self.per_request { &state.ledger } else { &[] };
            for entry in entries {
                let Some(rid) = entry.rid.as_ref() else {
                    continue;
                };
                let Some(remaining) = progress.remaining.get(rid) else {
                    continue;
                };
                // Coverage is allowed to rotate. Omission, aggregate zero,
                // engine restart and HTTP disconnect never complete a RID.
                if entry.dispatched_at.lock().is_none() || *remaining > entry.prompt_tokens {
                    continue;
                }
                entry.remaining.store(*remaining, Ordering::Relaxed);
                entry.seen_progress.store(true, Ordering::Relaxed);
            }
            state.progress_cursor = Some((progress.epoch, progress.sequence));
        }
        self.changed.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Create an isolated aggregate-mode ledger for a unit test.
    ///
    /// # Inputs
    /// No arguments; allocates a fresh notification channel.
    ///
    /// # Outputs
    /// Returns an empty `BackendState` without any engine or network tasks.
    fn state() -> BackendState {
        BackendState::new(Arc::new(Notify::new()))
    }
    /// Build a fixed-shape aggregate load sample for ledger tests.
    ///
    /// # Inputs
    /// `tokens` supplies the waiting uncached-token count.
    ///
    /// # Outputs
    /// Returns a complete observation with that token count, one running
    /// request and two waiting requests.
    fn sample(tokens: u64) -> LoadObservation {
        LoadObservation {
            uncached_tokens: Some(tokens),
            running: Some(1),
            waiting: Some(2),
        }
    }

    #[test]
    /// Verify decode reservations count until dropped, apart from prefill work.
    ///
    /// # Inputs
    /// No arguments. Reserve two D-legs on one worker, then drop them in turn.
    ///
    /// # Outputs
    /// Returns `()` if all ledger assertions hold; assertion failure panics.
    fn decode_reservations_count_until_dropped_apart_from_prefill_work() {
        let mut s = state();
        let a = s.reserve_decode("d");
        let b = s.reserve_decode("d");
        let v = s.view("d");
        assert_eq!(
            (
                v.active_decode_reqs,
                v.active_prefill_reqs,
                v.reserved_uncached_tokens
            ),
            (2, 0, 0)
        );
        assert_eq!(s.loads(["d"]).get("d").active_decode_reqs, 2);
        drop(a);
        assert_eq!(s.view("d").active_decode_reqs, 1);
        drop(b);
        assert_eq!(s.loads(["d"]).get("d").active_decode_reqs, 0);
    }
    #[test]
    /// Verify freshness labels and conservative load fallback for unknown workers.
    ///
    /// # Inputs
    /// No arguments. Create a missing worker and a ten-second-old engine sample.
    ///
    /// # Outputs
    /// Returns `()` if all ledger assertions hold; assertion failure panics.
    fn missing_and_stale_observations_are_not_idle_samples() {
        let mut s = state();
        assert_eq!(s.view("unknown").observation, None);
        s.apply_poll(
            "known",
            sample(200),
            Instant::now() - Duration::from_secs(10),
        );
        assert_eq!(s.view("known").status, ObservationStatus::Stale);
        assert_eq!(
            s.loads(["known", "unknown"])
                .get("unknown")
                .backlog_uncached_tokens,
            200
        );
        assert_eq!(s.view("unknown").status, ObservationStatus::Missing);
    }
    #[test]
    /// Verify the older response is ignored and repeated retirement is idempotent.
    ///
    /// # Inputs
    /// No arguments. Create a current sample, a new reservation and an older poll response.
    ///
    /// # Outputs
    /// Returns `()` if all ledger assertions hold; assertion failure panics.
    fn out_of_order_poll_cannot_regress_state_or_hide_new_reservations() {
        let mut s = state();
        let t = Instant::now();
        s.apply_poll("p", sample(300), t);
        let h = s.reserve("p", 50);
        s.apply_poll("p", sample(900), t - Duration::from_millis(1));
        assert_eq!(s.loads(["p"]).get("p").backlog_uncached_tokens, 350);
        h.retire();
        h.retire();
        assert_eq!(s.loads(["p"]).get("p").backlog_uncached_tokens, 300);
    }
    #[test]
    /// Verify aggregate token coverage does not remove live request accounting.
    ///
    /// # Inputs
    /// No arguments. Reserve dispatched work, apply a covering poll, then reserve more work.
    ///
    /// # Outputs
    /// Returns `()` if all ledger assertions hold; assertion failure panics.
    fn poll_coverage_does_not_erase_running_request_count() {
        let mut s = state();
        let old = s.reserve("p", 100);
        old.mark_dispatched();
        s.apply_poll("p", sample(80), Instant::now());
        let new = s.reserve("p", 20);
        let load = s.loads(["p"]).get("p");
        assert_eq!(load.backlog_uncached_tokens, 100);
        assert_eq!(load.active_prefill_reqs, 2);
        old.retire();
        new.retire();
        assert_eq!(s.loads(["p"]).get("p").active_prefill_reqs, 0);
    }
    #[test]
    /// Verify rollback applies only before dispatch and uncertain work stays active.
    ///
    /// # Inputs
    /// No arguments. Drop one pending guard, then disarm and drop another before marking it
    /// unknown.
    ///
    /// # Outputs
    /// Returns `()` if all ledger assertions hold; assertion failure panics.
    fn only_undispatched_guard_rolls_back_without_terminal_evidence() {
        let mut s = state();
        let first = s.reserve("p", 100);
        {
            let _construction = first.pending_guard();
        }
        assert_eq!(s.view("p").active_prefill_reqs, 0);
        let second = s.reserve("p", 200);
        {
            let mut construction = second.pending_guard();
            construction.mark_dispatched();
        }
        second.mark_unknown();
        let view = s.view("p");
        assert_eq!(view.active_prefill_reqs, 1);
        assert_eq!(view.unknown_prefill_reqs, 1);
        second.retire();
        assert_eq!(s.view("p").active_prefill_reqs, 0);
    }
    #[test]
    /// Verify unsubmitted work is added to load and dispatched work can be poll-covered.
    ///
    /// # Inputs
    /// No arguments. Reserve work, poll before submission, then dispatch and poll again.
    ///
    /// # Outputs
    /// Returns `()` if all ledger assertions hold; assertion failure panics.
    fn aggregate_poll_never_subsumes_an_unsubmitted_plan() {
        let mut s = state();
        let handle = s.reserve("p", 1000);
        s.apply_poll("p", sample(20), Instant::now());
        assert_eq!(s.loads(["p"]).get("p").backlog_uncached_tokens, 1020);
        handle.mark_dispatched();
        s.apply_poll("p", sample(900), Instant::now());
        assert_eq!(s.loads(["p"]).get("p").backlog_uncached_tokens, 900);
        assert_eq!(s.view("p").active_prefill_reqs, 1);
    }
    /// Construct a truncated per-RID progress snapshot for ledger tests.
    ///
    /// # Inputs
    /// `epoch` and `sequence` identify the engine snapshot; `remaining` maps
    /// request IDs to remaining tokens; `timestamp` is its engine timestamp.
    ///
    /// # Outputs
    /// Returns a sample with zero aggregate waiting tokens and the supplied
    /// per-request progress, marked truncated to exercise incomplete coverage.
    fn work_sample(
        epoch: &str,
        sequence: u64,
        remaining: &[(&str, u64)],
        timestamp: f64,
    ) -> EngineLoadSample {
        EngineLoadSample {
            observation: sample(0),
            timestamp: Some(timestamp),
            prefill_work: Some(PrefillProgress {
                epoch: epoch.into(),
                sequence,
                remaining: remaining.iter().map(|(r, n)| (r.to_string(), *n)).collect(),
                truncated: true,
            }),
        }
    }
    #[test]
    /// Verify running work is counted and zero remaining tokens do not imply termination.
    ///
    /// # Inputs
    /// No arguments. Use two dispatched RIDs and truncated progress with zero aggregate waiting
    /// work.
    ///
    /// # Outputs
    /// Returns `()` if all ledger assertions hold; assertion failure panics.
    fn running_chunks_are_counted_even_when_waiting_aggregate_is_zero() {
        let mut s = BackendState::with_progress(Arc::new(Notify::new()), true);
        let active = s.reserve_request("p", 8192, Some("running"), 8192);
        active.mark_dispatched();
        let waiting = s.reserve_request("p", 6000, Some("waiting"), 10000);
        waiting.mark_dispatched();
        s.apply_sample(
            "p",
            work_sample("e", 1, &[("running", 8192)], 1.0),
            Instant::now(),
        );
        assert_eq!(s.loads(["p"]).get("p").backlog_uncached_tokens, 14192);
        s.apply_sample(
            "p",
            work_sample("e", 2, &[("running", 0)], 2.0),
            Instant::now(),
        );
        assert_eq!(s.loads(["p"]).get("p").backlog_uncached_tokens, 6000);
        assert_eq!(s.view("p").active_prefill_reqs, 2); // work zero is not HTTP termination
        assert_eq!(s.view("p").progress_observed_reqs, 1);
        waiting.retire();
        active.retire();
    }
    #[test]
    /// Verify valid progress corrects work while incomplete or invalid evidence preserves it.
    ///
    /// # Inputs
    /// No arguments. Apply an upward correction, old sequence, restarted epoch and impossible
    /// count.
    ///
    /// # Outputs
    /// Returns `()` if all ledger assertions hold; assertion failure panics.
    fn missing_rid_old_sequence_and_restart_never_erase_owned_work() {
        let mut s = BackendState::with_progress(Arc::new(Notify::new()), true);
        let h = s.reserve_request("p", 2000, Some("r"), 10000);
        h.mark_dispatched();
        s.apply_sample(
            "p",
            work_sample("e", 3, &[("r", 8000)], 3.0),
            Instant::now(),
        ); // actual miss corrects estimate upward
        s.apply_sample("p", work_sample("e", 2, &[("r", 0)], 2.0), Instant::now());
        assert_eq!(s.loads(["p"]).get("p").backlog_uncached_tokens, 8000);
        s.apply_sample("p", work_sample("new", 1, &[], 4.0), Instant::now());
        assert_eq!(s.loads(["p"]).get("p").backlog_uncached_tokens, 8000);
        s.apply_sample(
            "p",
            work_sample("new", 2, &[("r", 10001)], 5.0),
            Instant::now(),
        ); // impossible value ignored
        assert_eq!(s.loads(["p"]).get("p").backlog_uncached_tokens, 8000);
        h.retire();
    }
    #[test]
    /// Verify only dispatched, identifiable requests receive progress corrections.
    ///
    /// # Inputs
    /// No arguments. Reserve one anonymous request and one unuploaded RID, then report RID
    /// progress.
    ///
    /// # Outputs
    /// Returns `()` if all ledger assertions hold; assertion failure panics.
    fn no_rid_and_unuploaded_requests_keep_their_estimates() {
        let mut s = BackendState::with_progress(Arc::new(Notify::new()), true);
        let no_rid = s.reserve_request("p", 500, None, 1000);
        no_rid.mark_dispatched();
        let unuploaded = s.reserve_request("p", 700, Some("r"), 1000);
        s.apply_sample("p", work_sample("e", 1, &[("r", 0)], 1.0), Instant::now());
        assert_eq!(s.loads(["p"]).get("p").backlog_uncached_tokens, 1200);
        unuploaded.mark_dispatched();
        s.apply_sample("p", work_sample("e", 2, &[("r", 0)], 2.0), Instant::now());
        assert_eq!(s.loads(["p"]).get("p").backlog_uncached_tokens, 500);
    }
    #[test]
    /// Verify repeated HTTP retrieval does not make stale engine state appear fresh.
    ///
    /// # Inputs
    /// No arguments. Read the same stale snapshot twice, then advance its engine sequence and time.
    ///
    /// # Outputs
    /// Returns `()` if all ledger assertions hold; assertion failure panics.
    fn repeated_shared_memory_snapshot_does_not_refresh_engine_age() {
        let mut s = BackendState::with_progress(Arc::new(Notify::new()), true);
        s.apply_sample(
            "p",
            work_sample("e", 1, &[], 1.0),
            Instant::now() - Duration::from_secs(10),
        );
        s.apply_sample("p", work_sample("e", 1, &[], 1.0), Instant::now());
        assert_eq!(s.view("p").status, ObservationStatus::Stale);
        s.apply_sample("p", work_sample("e", 2, &[], 2.0), Instant::now());
        assert_eq!(s.view("p").status, ObservationStatus::Fresh);
    }

    // ---- unknown reconciliation (external role control) ----

    fn reconciling() -> BackendState {
        let mut s = BackendState::new(Arc::new(Notify::new()));
        s.set_unknown_reconcile(Some(UnknownReconcile {
            empty_queue: Duration::from_secs(10),
            quiet: Some(Duration::from_secs(60)),
        }));
        s
    }

    /// One advancing progress reply of worker "p" sent `at_s` after `base`.
    fn reply(
        s: &mut BackendState,
        base: Instant,
        at_s: u64,
        waiting: u64,
        listed: &[&str],
        truncated: bool,
    ) {
        s.apply_sample(
            "p",
            EngineLoadSample {
                observation: LoadObservation {
                    uncached_tokens: Some(0),
                    running: Some(0),
                    waiting: Some(waiting),
                },
                timestamp: None,
                prefill_work: Some(PrefillProgress {
                    epoch: "e".into(),
                    sequence: at_s + 1,
                    remaining: listed.iter().map(|r| (r.to_string(), 100)).collect(),
                    truncated,
                }),
            },
            base + Duration::from_secs(at_s),
        );
    }

    fn unknown_on_p(s: &mut BackendState, rid: &str) -> LedgerHandle {
        let h = s.reserve_request("p", 4096, Some(rid), 8192);
        h.mark_dispatched();
        h.mark_unknown();
        h
    }

    /// An aborted P-leg (unknown, its RID gone from the engine): retired after
    /// 10 s of replies without it while the waiting queue was empty.
    #[test]
    fn unknown_p_reservation_retires_after_absence_with_an_empty_queue() {
        let mut s = reconciling();
        let base = Instant::now();
        let h = unknown_on_p(&mut s, "aborted");
        for at in [0, 5, 9] {
            reply(&mut s, base, at, 0, &[], false);
            assert!(!h.is_retired(), "at {at} s");
        }
        assert_eq!(s.view("p").active_prefill_reqs, 1);
        assert_eq!(s.view("p").unknown_prefill_reqs, 1);
        reply(&mut s, base, 10, 0, &[], false);
        assert!(h.is_retired());
        let v = s.view("p");
        assert_eq!(
            (
                v.active_prefill_reqs,
                v.unknown_prefill_reqs,
                v.unknown_retired_reqs
            ),
            (0, 0, 1)
        );
    }

    /// A busy waiting queue could hide a queued request: only the quiet bound
    /// (60 s) retires it then; a truncated reply restarts the absence clock.
    #[test]
    fn busy_queue_or_truncated_replies_delay_the_retirement() {
        let mut s = reconciling();
        let base = Instant::now();
        let h = unknown_on_p(&mut s, "queued");
        reply(&mut s, base, 0, 3, &[], false);
        reply(&mut s, base, 30, 3, &[], false);
        reply(&mut s, base, 40, 0, &[], true); // truncated: no evidence, clock restarts
        reply(&mut s, base, 45, 3, &[], false);
        reply(&mut s, base, 100, 3, &[], false);
        assert!(!h.is_retired(), "55 s of absence since the truncated reply");
        reply(&mut s, base, 105, 3, &[], false);
        assert!(h.is_retired(), "60 s quiet bound");
    }

    /// A real long-running P (listed by its worker) and a known (not unknown)
    /// reservation are never retired; no reconciliation without the setting.
    #[test]
    fn listed_or_known_reservations_are_never_retired() {
        let mut s = reconciling();
        let base = Instant::now();
        let running = unknown_on_p(&mut s, "running");
        let known = s.reserve_request("p", 4096, Some("known"), 8192);
        known.mark_dispatched();
        for at in (0..=300).step_by(10) {
            reply(&mut s, base, at, 0, &["running"], false);
        }
        assert!(!running.is_retired() && !known.is_retired());
        assert_eq!(s.view("p").active_prefill_reqs, 2);

        let mut off = BackendState::new(Arc::new(Notify::new()));
        off.set_unknown_reconcile(None);
        let base = Instant::now();
        let h = unknown_on_p(&mut off, "aborted");
        for at in (0..=300).step_by(10) {
            reply(&mut off, base, at, 0, &[], false);
        }
        assert!(!h.is_retired());
    }

    /// A worker whose replies stop (dead, partitioned) keeps its unknown
    /// entries: only applied, advancing replies are evidence.
    #[test]
    fn a_worker_without_new_replies_keeps_its_unknown_entries() {
        let mut s = reconciling();
        let base = Instant::now();
        let h = unknown_on_p(&mut s, "aborted");
        reply(&mut s, base, 0, 0, &[], false);
        // The same snapshot again (sequence not advancing) is no evidence.
        for at in [20, 40, 80] {
            s.apply_sample(
                "p",
                EngineLoadSample {
                    observation: LoadObservation::default(),
                    timestamp: None,
                    prefill_work: Some(PrefillProgress {
                        epoch: "e".into(),
                        sequence: 1,
                        remaining: BTreeMap::new(),
                        truncated: false,
                    }),
                },
                base + Duration::from_secs(at),
            );
        }
        assert!(!h.is_retired());
    }

    /// An unknown reservation whose handle is gone (a P-leg task that ended
    /// in a transport error) leaves the ledger once an idle engine has kept
    /// reporting without listing it; no setting is needed.
    #[test]
    fn ownerless_unknown_reservation_is_retired_by_default() {
        let mut s = state();
        {
            let handle = s.reserve_request("p", 5000, Some("rid"), 8000);
            handle.mark_dispatched();
            handle.mark_unknown();
        } // the P-leg task ended: no handle is left
        let base = Instant::now();
        for (i, at) in (0..=3600u64).step_by(10).enumerate() {
            // An idle engine: fresh snapshots, empty queue, no work listed.
            s.apply_sample(
                "p",
                EngineLoadSample {
                    observation: LoadObservation {
                        uncached_tokens: Some(0),
                        running: Some(0),
                        waiting: Some(0),
                    },
                    timestamp: Some(100.0 + at as f64),
                    prefill_work: Some(PrefillProgress {
                        epoch: "e".into(),
                        sequence: i as u64 + 1,
                        remaining: Default::default(),
                        truncated: false,
                    }),
                },
                base + Duration::from_secs(at),
            );
        }
        let view = s.view("p");
        assert_eq!(
            (
                view.active_prefill_reqs,
                view.unknown_prefill_reqs,
                s.loads(["p"]).get("p").backlog_uncached_tokens
            ),
            (0, 0, 0),
            "(P reservations, unknown P reservations, backlog tokens) after one hour of idle engine reports"
        );
    }
}
