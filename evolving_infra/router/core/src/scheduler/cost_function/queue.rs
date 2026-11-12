// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! `--global-queue`: the global waiting queue (global queue) of the symmetric_pd cost
//! function with a measured lead (global queue).
//!
//! ```text
//! free(r)   = h1 - h2 < C       (largest and second largest local hit over
//!                                the pool; C = the engine chunk of the owner
//!                                P from /server_info, 8192 when not reported)
//! bound(r)  -> cost-function at arrival, never held (reserve event queue.class = bound)
//! free(r)   -> held; each step walks the held requests in HRRN order, key
//!              (wait + S) / S, the held work ahead as virtual backlog
//!              (crate::global_queue::walk); the target P = the cost argmin
//! release   iff W(P) <= lead(P): W = queued(P) / R, lead = C / R + T_react(P)
//!              (crate::prefill_work::PrefillWorkBook::queue_step with QueueAutoStep)
//! T_react   = time-weighted mean (tau) of release -> first progress poll that
//!              lists the RID, sampled only for releases to an idle P (no
//!              queued and no launched work, fresh view); poll period + 500 ms
//!              before the first sample of a P
//! guard     = 300 s max wait (released to the cost argmin also without room)
//! ```
//!
//! The queue holds only the request's admission future: no reservation exists
//! while a request is held. A release reserves the request on its P under the
//! work-book lock (one transaction with the step) and hands the reservation
//! to the waiting admission. Release passes run at the arrival (inline), on
//! every applied progress poll, on every terminal status of the work book, on
//! a timer tick while requests are held, and at the oldest request's max-wait
//! deadline. A client that leaves while its request is held takes it out of
//! the queue (drop guard). Trace: `queue` on each reserve event, `queue_drop`,
//! `queue_lead_sample` and the one-second `queue_stats`.

use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde_json::{json, Value};
use tokio::sync::{oneshot, Notify};

use super::{restrict_to_pool, CostFunction, HINT_LOG_POLICY};
use crate::config::GlobalQueueConfig;
use crate::global_queue::{GlobalQueueRoom, QueueRelease, AUTO_DISPATCH_DELAY_MS, CHUNK_TOKENS};
use crate::kv_hint::KvHintInjector;
use crate::prefill_work::{Candidate, QueueAutoStep, QueueHeld, QueueStepArgs, Reservation};
use crate::scheduler::placement_cost::{PlacementCostParams, PrefillWait};
use crate::scheduler::{PlaceInput, SymmetricPdError};
use crate::workers::Worker;

/// The pass period while nothing is held.
const IDLE_TICK: Duration = Duration::from_secs(1);

/// Release triggers (bits), named as the cost function queue names them.
const ARRIVAL: u8 = 1;
const RETIRE: u8 = 2;
const POLL: u8 = 4;
const TICK: u8 = 8;
const STARVATION: u8 = 16;

/// The primary trigger (starvation > poll > retire > arrival > tick) and all
/// triggers joined by `+`.
fn trigger_names(bits: u8) -> (&'static str, String) {
    const ORDER: [(u8, &str); 5] = [
        (STARVATION, "starvation"),
        (POLL, "poll"),
        (RETIRE, "retire"),
        (ARRIVAL, "arrival"),
        (TICK, "tick"),
    ];
    let names: Vec<&str> = ORDER
        .iter()
        .filter(|(bit, _)| bits & bit != 0)
        .map(|(_, name)| *name)
        .collect();
    match names.first() {
        Some(primary) => (*primary, names.join("+")),
        None => ("tick", "tick".to_string()),
    }
}

/// The engine chunk C of `worker` and its source.
pub(super) fn chunk_of(worker: &Worker) -> (u64, &'static str) {
    worker
        .chunk_tokens()
        .map_or((CHUNK_TOKENS, "default"), |c| (c, "server_info"))
}

/// The class of one request: (bound, h1, h2, C, C source, owner URL). The
/// owner is the P with the largest local hit (the first one of the pool on a
/// tie); its chunk decides. A request without a hit is free.
pub(super) fn classify(
    candidates: &[Candidate],
    pool: &[Arc<Worker>],
    tokens: u64,
) -> (bool, u64, u64, u64, &'static str, Option<String>) {
    let hits: Vec<u64> = candidates.iter().map(|c| c.local_hit.min(tokens)).collect();
    let (h1, h2) = crate::global_queue::top_two(&hits);
    let owner = candidates
        .iter()
        .find(|c| h1 > 0 && c.local_hit.min(tokens) == h1)
        .map(|c| c.worker_url.clone());
    let (chunk, source) = owner
        .as_deref()
        .and_then(|url| pool.iter().find(|w| w.url == url))
        .or_else(|| pool.first())
        .map_or((CHUNK_TOKENS, "default"), |w| chunk_of(w));
    (
        crate::global_queue::bound_to_owner(&hits, chunk),
        h1,
        h2,
        chunk,
        source,
        owner,
    )
}

/// The queue state of one cost function.
pub(super) struct GlobalQueue {
    cfg: GlobalQueueConfig,
    max_wait: Duration,
    /// A progress view older than this is stale (fail-open).
    stale_after: Duration,
    /// The pass period while requests are held (the poll period, 10 ms to
    /// 1 s).
    tick: Duration,
    /// T_react of a P before its first sample: poll period + 500 ms.
    pub(super) fallback_react_s: f64,
    state: Mutex<QueueState>,
    notify: Notify,
    pending: AtomicU8,
    passes: AtomicU64,
    /// The healthy P-role pool at each step (set by the runtime at start).
    pool: std::sync::OnceLock<Box<dyn Fn() -> Vec<Arc<Worker>> + Send + Sync>>,
    /// The pool of the last arrival: the step pool without a provider (tests).
    last_pool: Mutex<Vec<Arc<Worker>>>,
    /// `--symmetric-pd-pull-from-any-holder`: the registered workers of any role
    /// (breaker allowing when the breaker is on) at each step, set by the
    /// runtime at start; a held request keeps pull hints from them.
    donors: std::sync::OnceLock<Box<dyn Fn() -> Vec<Arc<Worker>> + Send + Sync>>,
    /// The donors of the last arrival: the step donors without a provider
    /// (tests).
    last_donors: Mutex<Vec<Arc<Worker>>>,
    /// Tests: the pool of the next step only (a step that read the pool
    /// just before a role change).
    #[cfg(test)]
    next_pool: Mutex<Option<Vec<Arc<Worker>>>>,
}

impl std::fmt::Debug for GlobalQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlobalQueue")
            .field("cfg", &self.cfg)
            .field("held", &self.state.lock().entries.len())
            .finish_non_exhaustive()
    }
}

#[derive(Default)]
struct QueueState {
    /// The held requests in arrival order.
    entries: Vec<Held>,
    next_id: u64,
    counters: Counters,
    /// The uncached tokens of the held requests on their plain cost-function choice
    /// at the last step.
    last_queue_tokens: u128,
    /// The step pool of the last pass (URLs): a P that is not in it joined
    /// the pool (role switch, drain end, breaker close).
    pool_urls: Vec<String>,
}

/// What a release hands to the waiting admission.
type Released = (Candidate, Reservation, Value);

/// One held request.
struct Held {
    id: u64,
    arrived: Instant,
    arrived_unix_ms: u64,
    tokens: u64,
    rid: Option<String>,
    /// The session identity and its key ([`crate::scheduler::placement_cost::placement_session_hash`]).
    session: Option<String>,
    session_key: Option<u64>,
    /// Its cost-function candidates over the pool of its arrival; a P that joined
    /// later is priced cold by the step.
    candidates: Vec<Candidate>,
    /// The plain cost-function choice at the first step that saw it.
    p_arrival: Option<String>,
    arrival_pass: u64,
    tx: oneshot::Sender<Released>,
}

/// Cumulative counts (the `queue_stats` event).
#[derive(Debug, Default, Clone, Copy)]
struct Counters {
    arrivals: u64,
    /// Bound to their owner: cost-function at arrival, never held.
    bound: u64,
    /// Free: through the queue.
    free: u64,
    released_at_arrival: u64,
    released_after_hold: u64,
    released_room: u64,
    /// Room `forecast`: the P's predicted drain was within its lead.
    released_forecast: u64,
    released_stale_view: u64,
    released_starved: u64,
    late_bound: u64,
    /// Releases to a P without queued work (a T_react sample at the first
    /// sighting).
    lead_sample_releases: u64,
    dropped_client_gone: u64,
    dropped_wait_budget: u64,
    dropped_reserve_error: u64,
    dropped_admission_stopped: u64,
    refused_rid: u64,
    passes: u64,
    passes_pool_empty: u64,
    /// Released, but the admission had left (the reservation rolled back).
    send_failed: u64,
    /// Releases whose reservation chose another P than the walk (a bug;
    /// must stay 0).
    walk_mismatch: u64,
    /// Releases whose cost-function total tied on two or more P, and those of them
    /// that the rotation decided (the other keys: room, own hit,
    /// outstanding requests).
    tie_breaks: u64,
    tie_rotations: u64,
    /// T_react estimates dropped because their P joined the pool again.
    react_restarts: u64,
    /// Released requests whose P left the P-role pool before the placement
    /// read the workers again: rolled back and walked again.
    rewalks: u64,
}

impl Counters {
    fn value(&self) -> Value {
        json!({
            "arrivals": self.arrivals,
            "bound": self.bound,
            "free": self.free,
            "released_at_arrival": self.released_at_arrival,
            "released_after_hold": self.released_after_hold,
            "released_room": self.released_room,
            "released_forecast": self.released_forecast,
            "released_stale_view": self.released_stale_view,
            "released_starved": self.released_starved,
            "late_bound": self.late_bound,
            "lead_sample_releases": self.lead_sample_releases,
            "dropped_client_gone": self.dropped_client_gone,
            "dropped_wait_budget": self.dropped_wait_budget,
            "dropped_reserve_error": self.dropped_reserve_error,
            "dropped_admission_stopped": self.dropped_admission_stopped,
            "refused_rid": self.refused_rid,
            "passes": self.passes,
            "passes_pool_empty": self.passes_pool_empty,
            "send_failed": self.send_failed,
            "walk_mismatch": self.walk_mismatch,
            "tie_breaks": self.tie_breaks,
            "tie_rotations": self.tie_rotations,
            "react_restarts": self.react_restarts,
            "rewalks": self.rewalks,
        })
    }
}

impl GlobalQueue {
    pub(super) fn new(cfg: GlobalQueueConfig, load_poll_ms: u64) -> Self {
        Self {
            cfg,
            max_wait: Duration::from_millis(cfg.max_wait_ms.max(1)),
            stale_after: crate::global_queue::stale_after(load_poll_ms),
            tick: Duration::from_millis(load_poll_ms.clamp(10, 1000)),
            fallback_react_s: load_poll_ms.saturating_add(AUTO_DISPATCH_DELAY_MS) as f64 / 1000.0,
            state: Mutex::new(QueueState::default()),
            notify: Notify::new(),
            pending: AtomicU8::new(0),
            passes: AtomicU64::new(0),
            pool: std::sync::OnceLock::new(),
            last_pool: Mutex::new(Vec::new()),
            donors: std::sync::OnceLock::new(),
            last_donors: Mutex::new(Vec::new()),
            #[cfg(test)]
            next_pool: Mutex::new(None),
        }
    }

    /// Set the step pool provider (the runtime's healthy P-role pool).
    pub(super) fn set_pool(&self, pool: Box<dyn Fn() -> Vec<Arc<Worker>> + Send + Sync>) {
        let _ = self.pool.set(pool);
    }

    /// Set the step donors provider (`--symmetric-pd-pull-from-any-holder`: the
    /// runtime's registered workers of any role, breaker allowing when the
    /// breaker is on).
    pub(super) fn set_donors(&self, donors: Box<dyn Fn() -> Vec<Arc<Worker>> + Send + Sync>) {
        let _ = self.donors.set(donors);
    }

    /// The URLs of the current donors (the provider, else the donors of the
    /// last arrival).
    fn step_sources(&self) -> Vec<String> {
        let donors = match self.donors.get() {
            Some(donors) => donors(),
            None => self.last_donors.lock().clone(),
        };
        donors.iter().map(|w| w.url.clone()).collect()
    }

    /// The pool of a release step (tests: a one-shot pool first).
    fn step_pool(&self) -> Vec<Arc<Worker>> {
        #[cfg(test)]
        if let Some(pool) = self.next_pool.lock().take() {
            return pool;
        }
        self.current_pool()
    }

    /// The current healthy P-role pool (the provider, else the pool of the
    /// last arrival).
    fn current_pool(&self) -> Vec<Arc<Worker>> {
        match self.pool.get() {
            Some(pool) => pool(),
            None => self.last_pool.lock().clone(),
        }
    }

    fn wake(&self, bits: u8) {
        self.pending.fetch_or(bits, Ordering::AcqRel);
        self.notify.notify_one();
    }
}

/// Takes a held request out of the queue when its admission future is
/// dropped (the client left) before a release.
struct Guard {
    cf: Weak<CostFunction>,
    id: u64,
    armed: bool,
}

impl Drop for Guard {
    fn drop(&mut self) {
        if self.armed {
            if let Some(cf) = self.cf.upgrade() {
                cf.queue_drop(self.id, "client_gone");
            }
        }
    }
}

/// What one release pass did (tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct PassOutcome {
    pub(crate) released: u32,
    /// Held requests when the pass ended.
    pub(crate) held: usize,
}

impl CostFunction {
    /// `--global-queue` is on.
    pub fn queue_enabled(&self) -> bool {
        self.queue.is_some()
    }

    /// A release pass is due (an applied progress poll, a terminal status).
    pub(super) fn queue_wake_poll(&self) {
        if let Some(queue) = &self.queue {
            queue.wake(POLL);
        }
    }

    pub(super) fn queue_wake_retire(&self) {
        if let Some(queue) = &self.queue {
            queue.wake(RETIRE);
        }
    }

    /// The released request `rid` was reserved on `worker`, which left the
    /// P-role pool before the placement read the workers again (a role
    /// switch, a drain or a breaker trip in between): the caller rolls the
    /// reservation back and walks the request again (`attempt` from 1).
    /// Counted (`rewalks`), traced (`queue_rewalk`) and logged.
    pub fn queue_note_rewalk(&self, rid: Option<&str>, worker: &str, attempt: u32) {
        let Some(queue) = self.queue.as_ref() else {
            return;
        };
        queue.state.lock().counters.rewalks += 1;
        tracing::warn!(target: "symmetric_pd", rid = rid.unwrap_or(""), worker, attempt,
            "symmetric_pd global queue: the released P left the P-role pool before the placement; the request \
             is walked again");
        let event = json!({
            "event": "queue_rewalk",
            "rid": rid,
            "worker_url": worker,
            "attempt": attempt,
        });
        if let Err(error) = self.book.record_event(event) {
            tracing::warn!(target: "symmetric_pd", %error, "queue_rewalk event not written");
        }
    }

    /// Select and reserve the P-leg of one request through the queue.
    ///
    /// # Inputs
    /// As [`CostFunction::reserve`] (`donors`: with
    /// `--symmetric-pd-pull-from-any-holder`, the other possible pull sources; the
    /// hints are chosen at arrival, and a held request keeps its arrival
    /// source while that source is in the step pool or still a registered
    /// worker of any role at the release step, breaker allowing when the
    /// breaker is on).
    ///
    /// # Outputs
    /// The selected candidate, the reservation, the placement decision and
    /// whether the request went through the queue (free: the caller reads the
    /// current workers again). A bound request is reserved at once. A free
    /// request waits until a pass releases it (at most the max wait, then the
    /// guard releases it to its cost argmin P); a client that leaves takes
    /// it out of the queue. A missing or reused rid, an empty pool or a
    /// failed trace gives [`SymmetricPdError::CostFunction`].
    #[allow(clippy::too_many_arguments)]
    pub async fn reserve_queue(
        self: &Arc<Self>,
        ix: &KvHintInjector,
        input: &PlaceInput,
        tokens: u64,
        reply: Option<&Value>,
        indexer_ms: f64,
        pool: &[Arc<Worker>],
        donors: &[Arc<Worker>],
    ) -> Result<(Candidate, Reservation, Value, bool), SymmetricPdError> {
        let queue = self.queue.as_ref().expect("--global-queue on");
        *queue.last_pool.lock() = pool.to_vec();
        if self.cfg.pull_from_any_holder {
            *queue.last_donors.lock() = donors.to_vec();
        }
        let full = reply.map(|r| ix.answer_from_reply(r, indexer_ms));
        let answer = full.as_ref().map(|a| restrict_to_pool(a, pool));
        let candidates = if self.cfg.pull_from_any_holder {
            self.candidates_any_holder(ix, full.as_ref(), tokens, pool, donors)
        } else {
            self.candidates(ix, answer.as_ref(), tokens, pool)
        };
        let (bound, h1, h2, chunk, chunk_source, owner) = classify(&candidates, pool, tokens);
        queue.state.lock().counters.arrivals += 1;
        // A rid that the queue holds has no reservation yet: refused here for
        // both classes (the book refuses a used or missing rid itself).
        let refusal = self.book.rid_refusal(input.rid.as_deref()).or_else(|| {
            let st = queue.state.lock();
            input
                .rid
                .as_deref()
                .is_some_and(|r| st.entries.iter().any(|e| e.rid.as_deref() == Some(r)))
                .then_some("uncached balance cannot reuse a rid; the global waiting queue holds it")
        });
        if let Some(error) = refusal {
            queue.state.lock().counters.refused_rid += 1;
            return Err(SymmetricPdError::CostFunction(error.to_string()));
        }
        if bound {
            queue.state.lock().counters.bound += 1;
            let note = json!({
                "rule": "auto",
                "class": "bound",
                "held": false,
                "h1_tokens": h1,
                "h2_tokens": h2,
                "owner": owner,
                "chunk_tokens": chunk,
                "chunk_source": chunk_source,
            });
            // The decision of the cost function's wait (the forecast with
            // its gate under `--prefill-wait forecast`).
            let (selected, reservation, decision) =
                self.reserve_candidates(input, tokens, candidates, Some(note))?;
            self.log_decision(ix, answer.as_ref(), input, &selected, &decision);
            super::log_pull_source(&self.cfg, input, &selected, &decision, pool);
            return Ok((selected, reservation, decision, false));
        }
        let pass_id = queue.passes.fetch_add(1, Ordering::Relaxed) + 1;
        let (tx, rx) = oneshot::channel();
        let id = {
            let mut st = queue.state.lock();
            let id = st.next_id;
            st.next_id += 1;
            st.counters.free += 1;
            st.entries.push(Held {
                id,
                arrived: Instant::now(),
                arrived_unix_ms: crate::prefill_work::unix_ms(),
                tokens,
                rid: input.rid.clone(),
                session: input.placement_session.clone(),
                session_key: input
                    .placement_session
                    .as_deref()
                    .map(crate::scheduler::placement_cost::placement_session_hash),
                candidates,
                p_arrival: None,
                arrival_pass: pass_id,
                tx,
            });
            id
        };
        let mut guard = Guard {
            cf: Arc::downgrade(self),
            id,
            armed: true,
        };
        if self.queue_pass(ARRIVAL, pass_id).held > 0 {
            // Requests wait: the dispatcher re-reads the oldest deadline.
            queue.wake(ARRIVAL);
        }
        // The guard releases every request after the max wait; this bound
        // only trips when even that is impossible (no pool).
        let budget = queue.max_wait * 2 + Duration::from_secs(10);
        let result = tokio::time::timeout(budget, rx).await;
        guard.armed = false;
        match result {
            Ok(Ok((selected, reservation, decision))) => {
                self.log_decision(ix, answer.as_ref(), input, &selected, &decision);
                super::log_pull_source(&self.cfg, input, &selected, &decision, pool);
                Ok((selected, reservation, decision, true))
            }
            // Refused at release (traced there) or the cost function is gone.
            Ok(Err(_)) => Err(SymmetricPdError::CostFunction(
                "the global waiting queue dropped the request".into(),
            )),
            Err(_) => {
                self.queue_drop(id, "wait_budget");
                Err(SymmetricPdError::CostFunction(
                    "the global waiting queue could not release the request".into(),
                ))
            }
        }
    }

    /// One release pass: decision steps ([`crate::prefill_work::PrefillWorkBook::queue_step`]
    /// under the queue lock and the work-book lock) until a step releases
    /// nothing. Each released request is handed to its admission outside
    /// the locks. Requests whose client left are dropped first. `pass_id`
    /// tells the requests whose own arrival started this pass (`held =
    /// false`).
    pub(crate) fn queue_pass(&self, bits: u8, pass_id: u64) -> PassOutcome {
        let mut outcome = PassOutcome::default();
        let Some(queue) = self.queue.as_ref() else {
            return outcome;
        };
        let (trigger, triggers) = trigger_names(bits);
        // The walk and its releases use the token-count terms, also with
        // `--prefill-wait forecast` (room `forecast` replaces the backlog
        // of a P that has a forecast by R x its predicted drain).
        let walk_params = PlacementCostParams {
            wait: PrefillWait::Tokens,
            ..self.cfg.cost_params
        };
        let mut first = true;
        loop {
            let pool = queue.step_pool();
            // `--symmetric-pd-pull-from-any-holder`: a held request keeps a pull
            // hint whose source is still a registered worker of any role
            // (breaker allowing when the breaker is on).
            let sources = if self.cfg.pull_from_any_holder {
                queue.step_sources()
            } else {
                Vec::new()
            };
            // Room `forecast`: the forecast inputs of a cold one-chunk probe per P,
            // before the locks (fits, snapshots, pushed states); only while
            // requests are held.
            let drain_wanted = queue.cfg.room == GlobalQueueRoom::Forecast && !pool.is_empty() && {
                let held = !queue.state.lock().entries.is_empty();
                held
            };
            let forecast = self
                .forecast
                .as_ref()
                .filter(|_| drain_wanted)
                .map(|forecast| {
                    let probes: Vec<Candidate> = pool
                        .iter()
                        .map(|w| crate::prefill_work::queue_probe_candidate(&w.url))
                        .collect();
                    forecast.context(
                        &self.book,
                        &probes,
                        crate::prefill_work::QUEUE_PROBE_TOKENS,
                        None,
                        None,
                    )
                });
            let mut st = queue.state.lock();
            if first {
                st.counters.passes += 1;
                first = false;
            }
            // A P that joined the step pool (again) restarts its T_react:
            // after a role switch back to prefill its earlier samples are
            // from another load (lock order: queue, then work book).
            let now_urls: Vec<String> = pool.iter().map(|w| w.url.clone()).collect();
            if now_urls != st.pool_urls {
                let joined: Vec<String> = now_urls
                    .iter()
                    .filter(|u| !st.pool_urls.contains(u))
                    .cloned()
                    .collect();
                for url in &joined {
                    if let Some(react_s) = self.book.queue_lead_restart(url) {
                        tracing::info!(target: "symmetric_pd", worker = %url, react_s,
                            "symmetric_pd global queue: a P joined the pool again; its T_react restarts");
                        st.counters.react_restarts += 1;
                    }
                }
                st.pool_urls = now_urls;
            }
            let gone = take_closed(&mut st.entries);
            st.counters.dropped_client_gone += gone.len() as u64;
            if st.entries.is_empty() || pool.is_empty() {
                if st.entries.is_empty() {
                    st.last_queue_tokens = 0;
                } else {
                    st.counters.passes_pool_empty += 1;
                }
                outcome.held = st.entries.len();
                drop(st);
                self.queue_trace_drops(&gone, "client_gone", None);
                return outcome;
            }
            let urls: Vec<String> = pool.iter().map(|w| w.url.clone()).collect();
            let chunks: Vec<(u64, &'static str)> = pool.iter().map(|w| chunk_of(w)).collect();
            let depth = vec![0u64; urls.len()];
            let now = Instant::now();
            let step = {
                let held: Vec<QueueHeld<'_>> = st
                    .entries
                    .iter()
                    .map(|e| QueueHeld {
                        input_tokens: e.tokens,
                        rid: e.rid.as_deref(),
                        candidates: e.candidates.as_slice(),
                        arrived: e.arrived,
                        arrived_unix_ms: e.arrived_unix_ms,
                        session: e.session_key,
                        session_label: e.session.as_deref(),
                        p_arrival: e.p_arrival.as_deref(),
                        arrival_pass: e.arrival_pass == pass_id,
                    })
                    .collect();
                self.book.queue_step(
                    walk_params,
                    &held,
                    &QueueStepArgs {
                        urls: &urls,
                        depth: &depth,
                        stale_after: queue.stale_after,
                        max_wait: queue.max_wait,
                        // Unused with `auto`: the running pass of each P is
                        // its own chunk (`chunks`, from /server_info).
                        pass_tokens: CHUNK_TOKENS,
                        auto: Some(QueueAutoStep { chunks: &chunks }),
                        now,
                        pass_id,
                        trigger,
                        triggers: &triggers,
                        forecast: forecast.as_ref(),
                        // Unused: with `auto` each P's lead is C / R + T_react.
                        lead_s: 0.0,
                        sources: &sources,
                    },
                )
            };
            let step = match step {
                Ok(step) => step,
                Err(error) => {
                    // Admission is stopped (placement trace failure): every held
                    // request fails, as every new arrival does.
                    let failed: Vec<Held> = st.entries.drain(..).collect();
                    st.counters.dropped_admission_stopped += failed.len() as u64;
                    st.last_queue_tokens = 0;
                    drop(st);
                    tracing::error!(target: "symmetric_pd", %error, held = failed.len(),
                        "symmetric_pd global queue: cost reservation failed; the queue fails its held requests");
                    self.queue_trace_drops(&gone, "client_gone", None);
                    self.queue_trace_drops(&failed, "admission_stopped", Some(error.as_str()));
                    return outcome;
                }
            };
            for (entry, p) in st.entries.iter_mut().zip(&step.plain) {
                if entry.p_arrival.is_none() {
                    entry.p_arrival = urls.get(*p).cloned();
                }
            }
            st.last_queue_tokens = step.queue_tokens;
            let Some((index, result)) = step.release else {
                outcome.held = st.entries.len();
                drop(st);
                self.queue_trace_drops(&gone, "client_gone", None);
                return outcome;
            };
            let entry = st.entries.remove(index);
            let released = match result {
                Ok(released) => {
                    let counters = &mut st.counters;
                    if entry.arrival_pass == pass_id {
                        counters.released_at_arrival += 1;
                    } else {
                        counters.released_after_hold += 1;
                    }
                    match released.release {
                        QueueRelease::Room => counters.released_room += 1,
                        QueueRelease::Forecast => counters.released_forecast += 1,
                        QueueRelease::StaleView => counters.released_stale_view += 1,
                        QueueRelease::Starved => counters.released_starved += 1,
                    }
                    counters.late_bound += u64::from(released.late_bound);
                    let record = &released.record;
                    counters.lead_sample_releases +=
                        u64::from(record["lead_sample"].as_bool() == Some(true));
                    counters.walk_mismatch += u64::from(
                        record["p_release"].as_str()
                            != Some(released.candidate.worker_url.as_str()),
                    );
                    let tie = &record["tie"];
                    counters.tie_breaks += u64::from(tie["tied"].as_u64().unwrap_or(0) >= 2);
                    counters.tie_rotations += u64::from(tie["decided_by"] == "rotation");
                    released
                }
                Err(error) => {
                    st.counters.dropped_reserve_error += 1;
                    drop(st);
                    tracing::error!(target: "symmetric_pd", %error,
                        rid = entry.rid.as_deref().unwrap_or(""),
                        "symmetric_pd global queue: cost reservation failed for a held request");
                    self.queue_trace_drops(&gone, "client_gone", None);
                    self.queue_trace_drops(
                        std::slice::from_ref(&entry),
                        "reserve_error",
                        Some(error.as_str()),
                    );
                    continue;
                }
            };
            drop(st);
            self.queue_trace_drops(&gone, "client_gone", None);
            let crate::prefill_work::QueueReleased {
                candidate,
                reservation,
                decision,
                record,
                release,
                starved,
                late_bound,
            } = released;
            let hold_ms = now.saturating_duration_since(entry.arrived).as_millis() as u64;
            tracing::info!(target: "symmetric_pd",
                rid = entry.rid.as_deref().unwrap_or(""),
                worker = %candidate.worker_url,
                held = entry.arrival_pass != pass_id,
                hold_ms,
                release = release.as_str(),
                starved,
                late_bound,
                p_arrival = record["p_arrival"].as_str().unwrap_or(""),
                key = record["key"].as_f64().unwrap_or(0.0),
                rank = record["rank"].as_u64().unwrap_or(0),
                queue_len = record["queue_len"].as_u64().unwrap_or(0),
                queued_tokens = record["queued_tokens"].as_str().unwrap_or(""),
                wait_s = record["wait_s"].as_f64().unwrap_or(0.0),
                lead_s = record["lead_s"].as_f64().unwrap_or(0.0),
                react_s = record["react_s"].as_f64().unwrap_or(0.0),
                react_source = record["react_source"].as_str().unwrap_or(""),
                depth_tokens = record["depth_tokens"].as_u64().unwrap_or(0),
                lead_sample = record["lead_sample"].as_bool().unwrap_or(false),
                trigger,
                triggers = triggers.as_str(),
                "symmetric_pd global queue release");
            outcome.released += 1;
            if entry.tx.send((candidate, reservation, decision)).is_err() {
                // The admission left after the step: the unsent reservation
                // rolls back when the value drops here.
                queue.state.lock().counters.send_failed += 1;
            }
        }
    }

    /// Take held request `id` out of the queue (its client left, or its
    /// wait budget ran out) and record the drop. Nothing happens when it has
    /// already left the queue.
    fn queue_drop(&self, id: u64, reason: &'static str) {
        let Some(queue) = self.queue.as_ref() else {
            return;
        };
        let entry = {
            let mut st = queue.state.lock();
            let Some(index) = st.entries.iter().position(|e| e.id == id) else {
                return;
            };
            if reason == "wait_budget" {
                st.counters.dropped_wait_budget += 1;
            } else {
                st.counters.dropped_client_gone += 1;
            }
            st.entries.remove(index)
        };
        self.queue_trace_drops(std::slice::from_ref(&entry), reason, None);
    }

    /// One `queue_drop` trace event and log line per request that left the
    /// queue without a release.
    fn queue_trace_drops(&self, entries: &[Held], reason: &str, error: Option<&str>) {
        let now = Instant::now();
        for entry in entries {
            let hold_s = now.saturating_duration_since(entry.arrived).as_secs_f64();
            tracing::info!(target: "symmetric_pd", rid = entry.rid.as_deref().unwrap_or(""),
                reason, hold_s, "symmetric_pd global queue drop");
            let event = json!({
                "event": "queue_drop",
                "rid": entry.rid,
                "reason": reason,
                "hold_s": hold_s,
                "arrived_unix_ms": entry.arrived_unix_ms,
                "error": error,
            });
            if let Err(error) = self.book.record_event(event) {
                tracing::warn!(target: "symmetric_pd", %error, "queue_drop event not written");
            }
        }
    }

    /// The `queue_stats` trace event (every second) and, with `log`, the
    /// `symmetric_pd queue stats` line.
    fn queue_emit_stats(&self, log: bool) {
        let Some(queue) = self.queue.as_ref() else {
            return;
        };
        let now = Instant::now();
        let pool = queue.current_pool();
        let urls: Vec<String> = pool.iter().map(|w| w.url.clone()).collect();
        let (queue_len, held_input_tokens, held_uncached_tokens, oldest_hold_s, counters) = {
            let st = queue.state.lock();
            (
                st.entries.len(),
                st.entries
                    .iter()
                    .map(|e| u128::from(e.tokens))
                    .sum::<u128>(),
                st.last_queue_tokens,
                st.entries
                    .first()
                    .map(|e| now.saturating_duration_since(e.arrived).as_secs_f64()),
                st.counters,
            )
        };
        let workers: Vec<Value> = pool
            .iter()
            .map(|w| {
                let (chunk, chunk_source) = chunk_of(w);
                let row = self
                    .book
                    .queue_rows(std::slice::from_ref(&w.url), now, chunk)
                    .pop()
                    .expect("one row per URL");
                let fresh = row
                    .view_age_ms
                    .is_some_and(|ms| Duration::from_millis(ms) <= queue.stale_after);
                json!({
                    "worker_url": w.url,
                    "chunk_tokens": chunk,
                    "chunk_source": chunk_source,
                    "queued_tokens": row.queued.to_string(),
                    "unlaunched_tokens": row.unlaunched.to_string(),
                    "launched_tokens": row.launched.to_string(),
                    "view_fresh": fresh,
                    "view_age_ms": row.view_age_ms,
                })
            })
            .collect();
        let event = json!({
            "event": "queue_stats",
            "mode": "auto",
            "queue_len": queue_len,
            "held_input_tokens": held_input_tokens.to_string(),
            "held_uncached_tokens": held_uncached_tokens.to_string(),
            "oldest_hold_s": oldest_hold_s,
            "max_wait_s": queue.max_wait.as_secs_f64(),
            "workers": workers,
            "lead": self.book.queue_lead_rows(&urls),
            "counters": counters.value(),
        });
        if let Err(error) = self.book.record_event(event) {
            tracing::warn!(target: "symmetric_pd", %error, "queue_stats event not written");
        }
        if log {
            tracing::info!(target: "symmetric_pd",
                queue_len,
                oldest_hold_s = oldest_hold_s.unwrap_or(0.0),
                held_uncached_tokens = %held_uncached_tokens,
                counters = %counters.value(),
                "symmetric_pd global queue stats");
        }
    }

    /// Held requests (tests).
    #[cfg(test)]
    pub(crate) fn queue_len(&self) -> usize {
        self.queue
            .as_ref()
            .map_or(0, |g| g.state.lock().entries.len())
    }

    /// The counters of the queue (tests).
    #[cfg(test)]
    pub(crate) fn queue_counters(&self) -> Value {
        self.queue
            .as_ref()
            .map_or(Value::Null, |g| g.state.lock().counters.value())
    }

    /// Tests: the next release step steps over `pool` (one step only), as a
    /// step that read the P-role pool just before a role change.
    #[cfg(test)]
    pub(crate) fn queue_set_next_pool(&self, pool: Vec<Arc<Worker>>) {
        if let Some(queue) = self.queue.as_ref() {
            *queue.next_pool.lock() = Some(pool);
        }
    }

    /// The decision line of one reservation (target `symmetric_pd`) and the
    /// injector's hint line, as [`CostFunction::reserve`] logs them.
    fn log_decision(
        &self,
        ix: &KvHintInjector,
        answer: Option<&crate::kv_hint::IndexerAnswer>,
        input: &PlaceInput,
        selected: &Candidate,
        decision: &Value,
    ) {
        ix.log_decision(
            answer,
            HINT_LOG_POLICY,
            &selected.worker_url,
            selected.hint.as_ref(),
        );
        super::log_decision_line(input, selected, decision);
    }
}

/// Remove the held requests whose admission has left (the receiver is gone).
fn take_closed(entries: &mut Vec<Held>) -> Vec<Held> {
    let mut gone = Vec::new();
    let mut i = 0;
    while i < entries.len() {
        if entries[i].tx.is_closed() {
            gone.push(entries.remove(i));
        } else {
            i += 1;
        }
    }
    gone
}

/// The release passes outside the arrivals: woken by applied progress polls
/// and terminal work-book statuses, by the tick while requests are held,
/// and exactly at the oldest held request's max-wait deadline.
pub(super) async fn dispatcher_loop(weak: Weak<CostFunction>) {
    loop {
        let Some(cf) = weak.upgrade() else { return };
        let Some(queue) = cf.queue.as_ref() else {
            return;
        };
        let (oldest, held) = {
            let st = queue.state.lock();
            (st.entries.first().map(|e| e.arrived), st.entries.len())
        };
        let now = Instant::now();
        let deadline =
            oldest.map(|arrived| (arrived + queue.max_wait).saturating_duration_since(now));
        let (sleep_for, timer_is_deadline) = match deadline {
            // A deadline within one tick: wake exactly then. A deadline in
            // the past means the last pass could not release (no pool).
            Some(until) if !until.is_zero() && until <= queue.tick => (until, true),
            _ if held > 0 => (queue.tick, false),
            _ => (IDLE_TICK, false),
        };
        let notified = queue.notify.notified();
        let timer_fired = tokio::select! {
            _ = notified => false,
            _ = tokio::time::sleep(sleep_for) => true,
        };
        let mut bits = queue.pending.swap(0, Ordering::AcqRel);
        if timer_fired {
            bits |= if timer_is_deadline { STARVATION } else { TICK };
        }
        if bits == 0 {
            bits = TICK;
        }
        let pass_id = queue.passes.fetch_add(1, Ordering::Relaxed) + 1;
        if std::panic::catch_unwind(AssertUnwindSafe(|| cf.queue_pass(bits, pass_id))).is_err() {
            tracing::error!(target: "symmetric_pd", "symmetric_pd global queue pass panicked; the dispatcher continues");
        }
        drop(cf);
    }
}

/// The `queue_stats` trace event every second; the log line every 10 s.
pub(super) async fn stats_loop(weak: Weak<CostFunction>) {
    let mut interval = tokio::time::interval(Duration::from_secs(1));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut ticks = 0u64;
    loop {
        interval.tick().await;
        let Some(cf) = weak.upgrade() else { return };
        cf.queue_emit_stats(ticks.is_multiple_of(10));
        ticks += 1;
        drop(cf);
    }
}
