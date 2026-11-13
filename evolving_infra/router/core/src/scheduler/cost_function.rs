// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! `--symmetric-pd-p-select cost-function`: the placement cost function of the
//! cost-function policy as the symmetric_pd P selection on symmetric workers.
//!
//! The cost function keeps its own [`PrefillWorkBook`] (the work book with the
//! placement cost, lambda and R fixed or auto) and its own per-RID progress poll
//! of the P-role workers: `GET /v1/loads?include=core,queues,prefill_work`
//! every `--prefill-poll-ms`, with one query per worker in flight.
//! It does not read the symmetric_pd Backend load view. Each request makes one
//! work-book transaction: it selects the P-leg among the healthy P-role
//! workers, decides the pull (the P-leg's `kv_pull_hint`) and reserves the
//! request's work. The runtime then leaves that P as the only prefill
//! candidate of the placement; the D-leg follows `--symmetric-pd-d-select`. The
//! routes own the reservation as on the PD path: dispatched before the P-leg
//! send, finished on the P answer, unknown on a transport error, rolled back
//! on a drop before dispatch. An unknown reservation expires and leaves the
//! work book when a progress reply of its P omits its RID for
//! `--symmetric-pd-unknown-expire-secs` (default 60; 0 = never; see
//! [`crate::prefill_work`]); a P without replies keeps it.
//!
//! The price (lambda, R and the placement cost) is computed only behind the
//! work-book API ([`PrefillWorkBook::reserve_cost_session`]). This module builds
//! the candidates, runs the poll and labels the trace. The candidates are
//! the P-role workers only: every hit and every pull source is a P-role
//! worker (unless `--symmetric-pd-pull-from-any-holder`).
//!
//! With `--global-queue` a request that is free to go to any P waits in
//! the global waiting queue of [`global_queue`] until its cost argmin P is about
//! to run out of queued work; a request bound to its cache owner is reserved
//! at arrival as without the queue.
//!
//! With `--prefill-wait forecast` the cost-function decisions price each
//! candidate with the step-time-model P completion forecast and its
//! prediction gate; with `--global-queue-room forecast` the queue also gives
//! a P room while its forecast drain is within its lead. Both read the
//! engine `prefill_routing` snapshot of the progress poll and, with
//! `--forecast-push-port`, the engines' state push ([`forecast`]).
//!
//! With `--symmetric-pd-pull-from-any-holder` the targets and their hits stay the
//! P-role pool's, but the pull source may be any holder of the indexer answer
//! that is a registered worker of any role (breaker allowing when the
//! breaker is on), also a D-role worker (a former P after a P->D role switch).

mod forecast;
mod queue;

use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use serde_json::Value;

use super::placement_cost::{fetch_tokens, PrefillWait, MIN_FETCH_TOKENS};
use super::{PlaceInput, SymmetricPdError};
use crate::config::SymmetricPdCostFunctionConfig;
use crate::kv_hint::{HintCandidate, IndexerAnswer, KvHintConfig, KvHintInjector};
use crate::load_refresh::{fetch_progress_poll, include_rejected};
use crate::load_refresh::{LoadRefresh, Query};
use crate::prefill_work::{Candidate, PrefillWorkBook, Reservation};
use crate::workers::Worker;
use tokio::task::JoinHandle;

/// The `policy` label of every row of the cost-function trace.
pub const TRACE_POLICY: &str = "symmetric_pd_cost_function";

/// The `policy` name of the cost-function line of
/// [`KvHintInjector::log_decision`] (target `kv_hint`).
pub const HINT_LOG_POLICY: &str = "SymmetricPdCostFunction";

/// The refusal of `POST /symmetric_pd/place` with the cost function (HTTP 400):
/// that endpoint reports neither the dispatch nor the end of a request.
pub const PLACE_REFUSAL: &str =
    "/symmetric_pd/place is not available with --symmetric-pd-p-select \
                                 cost-function: the work book needs each request's dispatch \
                                 and end";

/// The placement cost function of one symmetric_pd runtime.
pub struct CostFunction {
    cfg: SymmetricPdCostFunctionConfig,
    book: Arc<PrefillWorkBook>,
    /// The cost-function pull floor: max(`--kv-pull-hint-min-incremental-tokens`, 512).
    fetch_floor: u64,
    /// The progress poll client: 500 ms connect, 800 ms in total.
    poll_client: reqwest::Client,
    /// One progress query per P-role worker in flight; stale sequences dropped.
    progress_refresh: Arc<LoadRefresh>,
    /// `--global-queue` (None: off): the global waiting queue.
    queue: Option<queue::GlobalQueue>,
    /// `--prefill-wait forecast` or `--global-queue-room forecast`
    /// (None: neither): the forecast state.
    forecast: Option<forecast::ForecastState>,
}

impl std::fmt::Debug for CostFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CostFunction")
            .field("cost_function", &self.cfg.cost_params)
            .field("placement_trace", &self.cfg.placement_trace)
            .field("load_poll_ms", &self.cfg.load_poll_ms)
            .field("fetch_floor", &self.fetch_floor)
            .finish_non_exhaustive()
    }
}

impl CostFunction {
    /// Create the work book (a new trace file) and log the cost model.
    ///
    /// # Inputs
    /// `cfg`: the cost-function parameters, the trace path, the poll period and the
    /// unknown expiry;
    /// `kv`: the runtime's indexer config (its incremental gate raises the
    /// pull floor above 512 tokens).
    ///
    /// # Outputs
    /// The cost function without background tasks (see [`Self::start`]).
    /// Panics when a configured trace file exists or cannot be created: a run
    /// that asked for its placement trace must not serve requests without it.
    pub fn new(cfg: SymmetricPdCostFunctionConfig, kv: &KvHintConfig) -> Arc<Self> {
        let (trace, secs) = (cfg.placement_trace.as_deref(), cfg.unknown_expire_secs);
        let expire = (secs > 0).then_some(Duration::from_secs(secs));
        let book = PrefillWorkBook::new(trace, TRACE_POLICY, expire)
            .expect("create new placement trace before serving requests");
        let queue = cfg
            .global_queue
            .map(|queue| queue::GlobalQueue::new(queue, cfg.load_poll_ms));
        if let Some(queue) = &queue {
            book.enable_queue_lead(cfg.cost_params.auto_tau_s, queue.fallback_react_s);
        }
        if cfg.trace_session_moves {
            book.enable_session_moves();
        }
        // The forecast state: the decisions (the forecast wait) or the queue
        // drain (`--global-queue-room forecast`).
        let decides = cfg.cost_params.wait == PrefillWait::Forecast;
        let drain = cfg
            .global_queue
            .is_some_and(|g| g.room == crate::global_queue::GlobalQueueRoom::Forecast);
        let forecast = cfg
            .forecast
            .as_ref()
            .filter(|_| decides || drain)
            .map(|t| forecast::ForecastState::new(Arc::clone(&t.0), decides, &book));
        let fetch_floor = u64::from(kv.min_incremental_tokens).max(MIN_FETCH_TOKENS);
        log_cost_model(&cfg, kv, fetch_floor);
        let poll_client = reqwest::Client::builder()
            .connect_timeout(Duration::from_millis(500))
            .timeout(Duration::from_millis(800))
            .build()
            .expect("build symmetric_pd cost_function poll client");
        let this = Arc::new(Self {
            cfg,
            book,
            fetch_floor,
            poll_client,
            progress_refresh: Arc::new(LoadRefresh::default()),
            queue,
            forecast,
        });
        if this.queue.is_some() {
            // Each terminal status frees work on its P: a release pass.
            let weak = Arc::downgrade(&this);
            this.book.set_terminal_hook(Box::new(move || {
                if let Some(cf) = weak.upgrade() {
                    cf.queue_wake_retire();
                }
            }));
        }
        this
    }

    /// `--symmetric-pd-pull-from-any-holder`: the caller passes the registered
    /// workers of any role (breaker allowing when the breaker is on) as the
    /// possible pull sources.
    pub fn pull_from_any_holder(&self) -> bool {
        self.cfg.pull_from_any_holder
    }

    /// `--symmetric-pd-pull-from-any-holder` with `--global-queue`: the
    /// registered workers of any role (breaker allowing when the breaker is
    /// on) at each queue step (`donors`), so that a held request keeps a pull
    /// hint whose source is still one of them. No-op without the queue.
    pub fn set_queue_donors(&self, donors: impl Fn() -> Vec<Arc<Worker>> + Send + Sync + 'static) {
        if let Some(queue) = &self.queue {
            queue.set_donors(Box::new(donors));
        }
    }

    /// Start the per-RID progress poll of the P-role workers, the one-second
    /// work-book snapshot and, with `--global-queue`, the release
    /// dispatcher and the one-second queue statistics.
    ///
    /// # Inputs
    /// `members`: the poll set at each tick (the P-role workers whatever
    /// their breaker state, so the work book of a tripped P stays current);
    /// `pool`: the healthy P-role pool of each queue step (the pool of the
    /// admissions).
    ///
    /// # Outputs
    /// Unit. The tasks hold a weak reference and stop when the cost function
    /// drops. Without a Tokio runtime (sync unit tests) nothing starts and a
    /// warning is logged.
    pub fn start(
        self: &Arc<Self>,
        members: impl Fn() -> Vec<Arc<Worker>> + Send + Sync + 'static,
        pool: impl Fn() -> Vec<Arc<Worker>> + Send + Sync + 'static,
    ) {
        if let Some(queue) = &self.queue {
            queue.set_pool(Box::new(pool));
        }
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::warn!(
                target: "symmetric_pd",
                "symmetric_pd cost_function: no tokio runtime at construction; progress poll and \
                 snapshot tasks not started"
            );
            return;
        };
        let members: Arc<dyn Fn() -> Vec<Arc<Worker>> + Send + Sync> = Arc::new(members);
        handle.spawn(progress_poller_loop(
            Arc::downgrade(self),
            Arc::clone(&members),
        ));
        if self
            .forecast
            .as_ref()
            .is_some_and(forecast::ForecastState::push_on)
        {
            handle.spawn(forecast::push_supervisor_loop(
                Arc::downgrade(self),
                members,
            ));
        }
        handle.spawn(snapshot_loop(Arc::downgrade(self)));
        if self.queue.is_some() {
            handle.spawn(queue::dispatcher_loop(Arc::downgrade(self)));
            handle.spawn(queue::stats_loop(Arc::downgrade(self)));
        }
    }

    /// True when the cost-function copy price or R comes from the router's
    /// measurements (`auto`); the routes then pass the session identity.
    pub fn is_auto(&self) -> bool {
        self.cfg.cost_params.is_auto()
    }

    /// The routes pass the session identity ([`PlaceInput::placement_session`]):
    /// the copy price or R `auto` (N_active) or
    /// `--symmetric-pd-trace-session-moves`.
    pub fn wants_session(&self) -> bool {
        self.is_auto() || self.cfg.trace_session_moves
    }

    /// `--symmetric-pd-trace-session-moves` is on.
    pub fn traces_session_moves(&self) -> bool {
        self.cfg.trace_session_moves
    }

    /// The work book (tests of the runtime).
    #[cfg(test)]
    pub(crate) fn book_for_test(&self) -> &Arc<PrefillWorkBook> {
        &self.book
    }

    /// The routes pass the forecast session key ([`PlaceInput::forecast_session`], the
    /// host rule of the forecast target): `--prefill-wait forecast`.
    pub fn wants_forecast_session(&self) -> bool {
        self.forecast
            .as_ref()
            .is_some_and(|forecast| forecast.decides)
    }

    /// The client ended `session` (`POST /kv_sessions/end`; the identity of
    /// [`crate::scheduler::placement_cost::placement_session_key`]): it leaves
    /// N_active of the copy price `auto` at once. Returns whether it was
    /// counted (always false with both cost-function modes fixed).
    pub fn session_end(&self, session: &str) -> bool {
        self.is_auto() && self.book.auto_price_end_session(session)
    }

    /// The cost-function candidates of one request: the cost function's rule on the
    /// P-role pool.
    ///
    /// # Inputs
    /// `ix`: the runtime's injector (pull gates); `answer`: the request's
    /// indexer answer, `None` when the query failed; `tokens`: the prompt
    /// length; `pool`: the healthy P-role workers.
    ///
    /// # Outputs
    /// One candidate per pool worker, in pool order. The answer is first
    /// restricted to the pool, so every hit and every pull source is a
    /// P-role worker. A hint whose gain is below the pull floor is dropped:
    /// the P then recomputes those tokens and pays no transfer.
    pub(crate) fn candidates(
        &self,
        ix: &KvHintInjector,
        answer: Option<&IndexerAnswer>,
        tokens: u64,
        pool: &[Arc<Worker>],
    ) -> Vec<Candidate> {
        let answer = answer.map(|a| restrict_to_pool(a, pool));
        pool.iter()
            .map(|worker| {
                let local_hit = answer
                    .as_ref()
                    .map_or(0, |a| {
                        u64::from(KvHintInjector::target_hit_tokens(a, &worker.url))
                    })
                    .min(tokens);
                let mut hint = answer.as_ref().and_then(|a| ix.pick(a, &worker.url));
                let mut remote_hit = hint
                    .as_ref()
                    .and_then(|h| h.get("expected_hit_tokens"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                if fetch_tokens(local_hit, remote_hit, tokens, self.fetch_floor) == 0 {
                    hint = None;
                    remote_hit = 0;
                }
                Candidate {
                    worker_url: worker.url.clone(),
                    local_hit,
                    gpu_hit: answer
                        .as_ref()
                        .and_then(|a| a.observed_gpu_hits.get(&worker.url))
                        .copied()
                        .map(u64::from),
                    expected_hit: local_hit.max(remote_hit).min(tokens),
                    hint,
                }
            })
            .collect()
    }

    /// `--symmetric-pd-pull-from-any-holder`: the candidates of
    /// [`Self::candidates`] with a wider pull source. Kept apart so that the
    /// default path stays the unchanged function above.
    ///
    /// # Inputs
    /// As [`Self::candidates`]; `answer` is the whole indexer answer;
    /// `donors`: the registered workers of any role (breaker allowing when
    /// the breaker is on, draining ones too), the only possible pull sources
    /// besides `pool`.
    ///
    /// # Outputs
    /// One candidate per pool worker, in pool order. The local and GPU hits
    /// are the pool worker's own (the answer restricted to the pool); the
    /// hint is picked from the holders of the answer that are in `pool` or
    /// `donors`, also a worker outside the P-role pool (a D-role worker). A
    /// holder in neither (removed from the registry, only in
    /// `--kv-worker-map`, or breaker-open while the breaker is on) is never a
    /// source. "Healthy" here is the breaker's view only: with
    /// `--disable-circuit-breaker` a crashed or hung worker that is still
    /// registered stays a possible source. The pull floor applies as in
    /// [`Self::candidates`].
    pub(crate) fn candidates_any_holder(
        &self,
        ix: &KvHintInjector,
        answer: Option<&IndexerAnswer>,
        tokens: u64,
        pool: &[Arc<Worker>],
        donors: &[Arc<Worker>],
    ) -> Vec<Candidate> {
        let targets = answer.map(|a| restrict_to_pool(a, pool));
        let sources: Vec<Arc<Worker>> = pool.iter().chain(donors).cloned().collect();
        let holders = answer.map(|a| restrict_to_pool(a, &sources));
        pool.iter()
            .map(|worker| {
                let local_hit = targets
                    .as_ref()
                    .map_or(0, |a| {
                        u64::from(KvHintInjector::target_hit_tokens(a, &worker.url))
                    })
                    .min(tokens);
                let mut hint = holders.as_ref().and_then(|a| ix.pick(a, &worker.url));
                let mut remote_hit = hint
                    .as_ref()
                    .and_then(|h| h.get("expected_hit_tokens"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                if fetch_tokens(local_hit, remote_hit, tokens, self.fetch_floor) == 0 {
                    hint = None;
                    remote_hit = 0;
                }
                Candidate {
                    worker_url: worker.url.clone(),
                    local_hit,
                    gpu_hit: targets
                        .as_ref()
                        .and_then(|a| a.observed_gpu_hits.get(&worker.url))
                        .copied()
                        .map(u64::from),
                    expected_hit: local_hit.max(remote_hit).min(tokens),
                    hint,
                }
            })
            .collect()
    }

    /// Select and reserve the P-leg of one request (one work-book
    /// transaction; no await inside).
    ///
    /// # Inputs
    /// `ix`: the runtime's injector; `input`: the request (its `rid` keys the
    /// engine progress, its `placement_session` counts in N_active of the copy
    /// price `auto`); `tokens`: the prompt length; `reply`/`indexer_ms`: the
    /// raw indexer reply of the request and its latency; `pool`: the healthy
    /// P-role workers; `donors`: with `--symmetric-pd-pull-from-any-holder`, the
    /// registered workers of any role (breaker allowing when the breaker is
    /// on), the only possible pull sources besides `pool`; ignored without
    /// the flag.
    ///
    /// # Outputs
    /// The selected candidate (its `hint` is the P-leg's pull hint), the
    /// unique reservation and the placement decision record. A missing or reused
    /// rid, an empty pool or a failed trace gives
    /// [`SymmetricPdError::CostFunction`]; nothing is reserved then.
    #[allow(clippy::too_many_arguments)]
    pub fn reserve(
        &self,
        ix: &KvHintInjector,
        input: &PlaceInput,
        tokens: u64,
        reply: Option<&Value>,
        indexer_ms: f64,
        pool: &[Arc<Worker>],
        donors: &[Arc<Worker>],
    ) -> Result<(Candidate, Reservation, Value), SymmetricPdError> {
        let full = reply.map(|r| ix.answer_from_reply(r, indexer_ms));
        let answer = full.as_ref().map(|a| restrict_to_pool(a, pool));
        let candidates = if self.cfg.pull_from_any_holder {
            self.candidates_any_holder(ix, full.as_ref(), tokens, pool, donors)
        } else {
            self.candidates(ix, answer.as_ref(), tokens, pool)
        };
        let (selected, reservation, decision) =
            self.reserve_candidates(input, tokens, candidates, None)?;
        ix.log_decision(
            answer.as_ref(),
            HINT_LOG_POLICY,
            &selected.worker_url,
            selected.hint.as_ref(),
        );
        log_decision_line(input, &selected, &decision);
        log_pull_source(&self.cfg, input, &selected, &decision, pool);
        Ok((selected, reservation, decision))
    }

    /// One cost-function reservation of `candidates` (one work-book transaction).
    /// With `--prefill-wait forecast` the forecast inputs are built before the
    /// lock and the routed request is recorded after it (shadow line, online
    /// state); otherwise the token-count or copy-price terms of the parameters.
    /// `queue_note`: the `queue` object of the reserve event (a request that the
    /// queue does not hold).
    pub(super) fn reserve_candidates(
        &self,
        input: &PlaceInput,
        tokens: u64,
        candidates: Vec<Candidate>,
        queue_note: Option<Value>,
    ) -> Result<(Candidate, Reservation, Value), SymmetricPdError> {
        let Some(forecast) = self.forecast.as_ref().filter(|forecast| forecast.decides) else {
            return self
                .book
                .reserve_cost_session_noted(
                    tokens,
                    input.rid.clone(),
                    candidates,
                    self.cfg.cost_params,
                    input.placement_session.as_deref(),
                    queue_note,
                )
                .map_err(SymmetricPdError::CostFunction);
        };
        // Fits, targets, snapshots and pushed states before the work-book lock.
        let ctx = forecast.context(
            &self.book,
            &candidates,
            tokens,
            input.rid.as_deref(),
            input.forecast_session.as_deref(),
        );
        let (selected, reservation, decision) = self
            .book
            .reserve_cost_forecast_session_noted(
                tokens,
                input.rid.clone(),
                candidates,
                self.cfg.cost_params,
                Some(&ctx),
                input.placement_session.as_deref(),
                queue_note,
            )
            .map_err(SymmetricPdError::CostFunction)?;
        forecast.record_route(
            &decision,
            input.rid.as_deref(),
            &selected.worker_url,
            tokens,
            input.forecast_session.clone(),
        );
        Ok((selected, reservation, decision))
    }

    /// One progress query per P-role worker (the cost function's
    /// `refresh_progress`; the `prefill_routing` section only with the forecast
    /// forecast state): apply each reply to the work book; a stale sequence or
    /// a failed query retains the reservations.
    fn refresh_progress(self: &Arc<Self>, members: Vec<Arc<Worker>>) -> Vec<JoinHandle<()>> {
        let age = Duration::from_millis(self.cfg.load_poll_ms);
        members
            .into_iter()
            .filter_map(|worker| {
                let query = self
                    .progress_refresh
                    .claim(&worker.url, Instant::now(), age, true)?;
                let cf = Arc::clone(self);
                Some(tokio::spawn(async move {
                    let result = cf.poll_progress(&worker, &query).await;
                    let elapsed_ms = query.sent_at.elapsed().as_secs_f64() * 1000.0;
                    crate::load_refresh::log_progress_result(&worker.url, &result, elapsed_ms);
                    query.finish(result.ok(), Instant::now());
                }))
            })
            .collect()
    }

    /// One progress query of `worker`, with the `prefill_routing` section
    /// when the forecast state wants it. The reply updates the work book
    /// (and the forecast state). Returns the engine cursor it applied.
    async fn poll_progress(&self, worker: &Worker, query: &Query) -> anyhow::Result<(String, u64)> {
        let mut routing = self
            .forecast
            .as_ref()
            .is_some_and(|forecast| forecast.routing_wanted(&worker.url));
        let fetched = loop {
            match fetch_progress_poll(&self.poll_client, &worker.url, routing).await {
                // An engine without the section rejects the include (HTTP 4xx
                // or 500): poll again without it, so the progress goes on;
                // re-probe later.
                Err(error) if routing && include_rejected(&error) => {
                    if let Some(forecast) = &self.forecast {
                        forecast.note_routing_rejected(&worker.url, &error);
                    }
                    routing = false;
                }
                fetched => break fetched,
            }
        };
        let received = Instant::now();
        let (progress, load, routing_raw) = fetched?;
        if routing {
            if let Some(forecast) = &self.forecast {
                forecast.note_routing_accepted(&worker.url);
            }
        }
        query.check(&progress.epoch, progress.sequence)?;
        let cursor = (progress.epoch.clone(), progress.sequence);
        self.book
            .apply_progress_received(&worker.url, progress, load, query.sent_at, received)
            .map_err(anyhow::Error::msg)?;
        if let (Some(forecast), Some(raw)) = (&self.forecast, routing_raw) {
            // Parsed here, off the work-book lock.
            forecast.observe(&worker.url, &raw, query.sent_at, received);
        }
        // An applied view can give room: a release pass.
        self.queue_wake_poll();
        Ok(cursor)
    }
}

/// The answer of the P-role pool: the candidates and the GPU observations
/// of the pool's workers, keyed by their registry URLs (a trailing `/` is
/// ignored), in the answer's order (best hit first).
fn restrict_to_pool(answer: &IndexerAnswer, pool: &[Arc<Worker>]) -> IndexerAnswer {
    let by_norm: HashMap<&str, &str> = pool
        .iter()
        .map(|w| (w.url.trim_end_matches('/'), w.url.as_str()))
        .collect();
    let url_of = |url: &str| -> Option<String> {
        by_norm
            .get(url.trim_end_matches('/'))
            .map(|registered| (*registered).to_owned())
    };
    IndexerAnswer {
        candidates: answer
            .candidates
            .iter()
            .filter_map(|c| {
                url_of(c.worker_url.as_str()).map(|worker_url| HintCandidate {
                    worker_url,
                    servicer_addr: c.servicer_addr.clone(),
                    expected_hit_tokens: c.expected_hit_tokens,
                })
            })
            .collect(),
        observed_gpu_hits: answer
            .observed_gpu_hits
            .iter()
            .filter_map(|(url, hit)| url_of(url.as_str()).map(|url| (url, *hit)))
            .collect(),
        indexer_ms: answer.indexer_ms,
    }
}

/// The `source_worker_id` of the P-leg's hint, `-` without one.
fn hint_source_worker(selected: &Candidate) -> String {
    selected
        .hint
        .as_ref()
        .and_then(|h| h.get("source_worker_id"))
        .and_then(Value::as_str)
        .unwrap_or("-")
        .to_string()
}

/// The decision line of one reservation (target `symmetric_pd`).
fn log_decision_line(input: &PlaceInput, selected: &Candidate, decision: &Value) {
    // `Value` inside the macro would resolve to tracing's field trait.
    let source_worker = hint_source_worker(selected);
    tracing::info!(
        target: "symmetric_pd",
        rid = input.rid.as_deref().unwrap_or("-"),
        p_worker = %selected.worker_url,
        owner = decision["owner"].as_str().unwrap_or(""),
        detour = decision["detour"].as_bool().unwrap_or(false),
        pulled_tokens = decision["pulled_tokens"].as_u64().unwrap_or(0),
        source_worker = %source_worker,
        expected_hit_tokens = selected.expected_hit,
        local_hit_tokens = selected.local_hit,
        placement_cost = decision["cost"].as_f64().unwrap_or(0.0),
        copy_price = decision["copy_price"].as_f64().unwrap_or(0.0),
        prefill_rate_tokens_per_s = decision["prefill_rate_tokens_per_s"].as_f64().unwrap_or(0.0),
        fallback_reason = decision["fallback_reason"].as_str().unwrap_or("none"),
        "symmetric_pd cost_function decision"
    );
}

/// `--symmetric-pd-pull-from-any-holder` only: the pull-source line of one
/// reservation (target `symmetric_pd`): the source worker of the P-leg's hint and
/// its role class against the healthy P-role `pool` ([`source_role`]).
fn log_pull_source(
    cfg: &SymmetricPdCostFunctionConfig,
    input: &PlaceInput,
    selected: &Candidate,
    decision: &Value,
    pool: &[Arc<Worker>],
) {
    if !cfg.pull_from_any_holder {
        return;
    }
    let source_worker = hint_source_worker(selected);
    tracing::info!(
        target: "symmetric_pd",
        rid = input.rid.as_deref().unwrap_or("-"),
        p_worker = %selected.worker_url,
        source_worker = %source_worker,
        source_role = source_role(&source_worker, pool),
        pulled_tokens = decision["pulled_tokens"].as_u64().unwrap_or(0),
        "symmetric_pd cost_function pull source"
    );
}

/// The role class of a pull source for the decision log: `none` without a
/// hint, `p_pool` for a healthy P-role worker, `outside_p_pool` otherwise (a
/// D-role worker or a P that left the pool).
fn source_role(source: &str, pool: &[Arc<Worker>]) -> &'static str {
    if source == "-" {
        "none"
    } else if pool
        .iter()
        .any(|w| w.url.trim_end_matches('/') == source.trim_end_matches('/'))
    {
        "p_pool"
    } else {
        "outside_p_pool"
    }
}

/// Every `load_poll_ms`: one progress query per P-role worker of `members`.
async fn progress_poller_loop(
    weak: Weak<CostFunction>,
    members: Arc<dyn Fn() -> Vec<Arc<Worker>> + Send + Sync>,
) {
    let Some(cf) = weak.upgrade() else { return };
    // The CLI refuses 0; a config built in code must not panic the task.
    let period = Duration::from_millis(cf.cfg.load_poll_ms.max(1));
    drop(cf);
    let mut interval = tokio::time::interval(period);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        interval.tick().await;
        let Some(cf) = weak.upgrade() else { return };
        let queries = cf.refresh_progress((*members)());
        drop(cf);
        futures::future::join_all(queries).await;
    }
}

/// The one-second work-book snapshot (trace row `snapshot`, with
/// `auto_price` when the copy price or R is `auto`).
async fn snapshot_loop(weak: Weak<CostFunction>) {
    let mut interval = tokio::time::interval(Duration::from_secs(1));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        interval.tick().await;
        let Some(cf) = weak.upgrade() else { break };
        cf.book.snapshot();
    }
}

/// The startup lines of the cost model (target `symmetric_pd`).
fn log_cost_model(cfg: &SymmetricPdCostFunctionConfig, kv: &KvHintConfig, fetch_floor: u64) {
    let cost_params = cfg.cost_params;
    tracing::info!(
        target: "symmetric_pd",
        prefill_rate_tokens_per_s = cost_params.prefill_rate_tokens_per_s,
        fetch_fixed_ms = cost_params.fetch_fixed_ms,
        fetch_ms_per_100k_tokens = cost_params.fetch_ms_per_100k_tokens,
        copy_price = cost_params.copy_price,
        wait = cost_params.wait.as_str(),
        pull_floor_tokens = fetch_floor,
        local_prefix_fallback = kv.local_prefix_fallback,
        "symmetric_pd cost_function: placement cost function on: cost = backlog + uncached \
         + R*max(0, fetch_s - backlog/R) + copy_price*fetched; no fetch cap",
    );
    if cost_params.is_auto() {
        let (lambda, rate) = cost_params.auto_description();
        let (tau, idle) = (cost_params.auto_tau_s, cost_params.auto_session_idle_s);
        tracing::info!(
            target: "symmetric_pd",
            copy_price_mode = cost_params.copy_price_mode.as_str(),
            prefill_rate_mode = cost_params.prefill_rate_mode.as_str(),
            tau_s = tau,
            session_idle_s = idle,
            "symmetric_pd cost_function: copy price auto: {lambda}; {rate}; tau = {tau} s, \
             session idle = {idle} s. Inputs from the prefill work book: busy = time with a \
             request dispatched to the P and its prefill not ended, R_hat = computed \
             prefill tokens per busy second, N_active = sessions with a request on a P \
             or an event in the idle window and no END. The fallback applies for the \
             first tau seconds and when an input is missing",
        );
    }
    if kv.local_prefix_fallback {
        // The CLI rejects this with cost-function; a config built in code can set it.
        tracing::warn!(
            target: "symmetric_pd",
            "symmetric_pd cost_function: the cost function drops engine-side local-prefix fallback hints: a \
             hint is sent only for a pull of at least the floor"
        );
    }
    if cfg.pull_from_any_holder {
        tracing::info!(
            target: "symmetric_pd",
            "symmetric_pd cost_function: pull from any holder (--symmetric-pd-pull-from-any-holder): the \
             pull source may be any holder of the indexer answer that is a healthy registered \
             worker (any role, breaker allowing), also outside the P-role pool (e.g. a former P \
             after a P->D switch); the targets stay the P-role pool"
        );
    }
    let trace = cfg
        .placement_trace
        .as_ref()
        .map_or_else(|| "off".to_string(), |p| p.display().to_string());
    let poll_ms = cfg.load_poll_ms;
    tracing::info!(
        target: "symmetric_pd",
        load_poll_ms = poll_ms,
        placement_trace = %trace,
        "symmetric_pd cost_function: P progress poll every {poll_ms} ms \
         (GET /v1/loads?include=core,queues,prefill_work on each P-role worker); \
         placement trace {trace}",
    );
    if let Some(queue) = cfg.global_queue {
        let fallback_ms = cfg
            .load_poll_ms
            .saturating_add(crate::global_queue::AUTO_DISPATCH_DELAY_MS);
        let (tau, max_wait_s) = (cost_params.auto_tau_s, queue.max_wait_ms as f64 / 1000.0);
        tracing::info!(
            target: "symmetric_pd",
            global_queue = true,
            tau_s = tau,
            max_wait_ms = queue.max_wait_ms,
            fallback_react_ms = fallback_ms,
            "symmetric_pd cost_function: global queue on: a request bound to one \
             P (h1 - h2 >= C, C = the engine chunk from /server_info, 8192 when not reported) \
             is reserved at arrival; a free request waits, in HRRN order with the held work \
             ahead as virtual backlog, until its cheapest P has W <= lead, W = queued / R, \
             lead = C / R + T_react; T_react = time-weighted mean (tau = {tau} s) of release to \
             first sighting, sampled at an empty engine queue ({fallback_ms} ms before the \
             first sample); max wait {max_wait_s} s; exact cost ties of a release \
             go to the P with room, the larger own hit, the fewest outstanding requests, then \
             in rotation; each step reads the current P-role pool",
        );
    }
    log_forecast(cfg);
    if cfg.trace_session_moves {
        tracing::info!(
            target: "symmetric_pd",
            trace_session_moves = true,
            "symmetric_pd cost_function: each reserve event with a session identity has a \
             session_move record (the P of the session's last decision, the hits, the pull)",
        );
    }
    let expire = cfg.unknown_expire_secs;
    if expire > 0 {
        tracing::info!(
            target: "symmetric_pd",
            unknown_expire_secs = expire,
            "symmetric_pd cost_function: unknown P-leg reservations expire after {expire} s \
             without engine progress (--symmetric-pd-unknown-expire-secs {expire}): a transport \
             error or a drop after dispatch keeps the work until a progress reply of its P, \
             sent at least {expire} s after the later of the error and the last listing of its \
             RID, omits the RID; a P that does not answer keeps it",
        );
    } else {
        tracing::info!(
            target: "symmetric_pd",
            unknown_expire_secs = 0,
            "symmetric_pd cost_function: unknown P-leg reservations never expire \
             (--symmetric-pd-unknown-expire-secs 0): their work stays in the work book until a \
             terminal result",
        );
    }
}

/// The startup lines of the forecast state (target `symmetric_pd`): the
/// decisions with `--prefill-wait forecast`, the queue drain with
/// `--global-queue-room forecast`, the state push.
fn log_forecast(cfg: &SymmetricPdCostFunctionConfig) {
    let cost_params = cfg.cost_params;
    let decides = cost_params.wait == PrefillWait::Forecast;
    let drain = cfg
        .global_queue
        .is_some_and(|g| g.room == crate::global_queue::GlobalQueueRoom::Forecast);
    let Some(forecast) = cfg.forecast.as_ref().map(|t| &*t.0) else {
        if decides || drain {
            // The CLI always attaches the forecast state; a config built in code may omit it.
            tracing::error!(
                target: "symmetric_pd",
                "symmetric_pd cost_function: forecast wait or global queue room forecast without the \
                 forecast state: every P uses the token-count terms \
                 (fallback no_context) and the queue keeps the token rule"
            );
        }
        return;
    };
    if !(decides || drain) {
        return;
    }
    let shadow = forecast.shadow_path.as_ref().map_or_else(
        || "placement trace".to_string(),
        |p| p.display().to_string(),
    );
    tracing::info!(
        target: "symmetric_pd",
        error_table = %forecast.bound_path.display(),
        error_table_sha256 = %forecast.bound_sha256,
        bound_mode = %forecast.mode,
        bound_level = %forecast.level,
        rq_margin_s = forecast.rq_margin_s,
        next_pass_lag = forecast.next_pass_lag,
        max_view_age_s = cost_params.max_state_age_s,
        forecast_schema = crate::forecast::SCHEMA,
        shadow = %shadow,
        "symmetric_pd cost_function: forecast state on (step-time-model P completion forecast, \
         online fits from the engine timeline samples): the progress poll also asks for the \
         engine prefill_routing snapshot (GET /v1/loads?include=...,prefill_routing; a symmetric \
         worker needs SGLANG_SYMMETRIC_ROUTING_SNAPSHOT=1); forecast_shadow / forecast_label lines go to the \
         {shadow}",
    );
    let (chunk, page) = crate::prefill_forecast::BOUND_FIT_POLICY;
    tracing::info!(
        target: "symmetric_pd",
        covered_chunk_tokens = chunk,
        covered_page_tokens = page,
        "symmetric_pd cost_function: the forecast error table covers engines with chunk {chunk} / \
         page {page} tokens; a worker with other sizes, or without a routing snapshot, has no \
         forecast and placement uses the token count for it (one warning per worker); if no \
         worker has a forecast the decision is the token-count decision",
    );
    if let Some(frozen) = forecast.frozen_fit.as_ref() {
        tracing::info!(
            target: "symmetric_pd",
            frozen_fit = %frozen.path.display(),
            frozen_sha256 = %frozen.sha256,
            "symmetric_pd cost_function: forecast fit file: every P's GPU model, launch model and \
             calibration constants come from this file, not from the online fits (the online \
             samples give the arrival rates only): {}",
            frozen.summary(),
        );
    }
    if decides {
        tracing::info!(
            target: "symmetric_pd",
            move_margin = cost_params.move_margin,
            "symmetric_pd cost_function: forecast wait: C_w = R*(pred_s + max(0, fetch_s - wait_s)) \
             + copy; leave owner o for best w only if (C_o - C_w)/R - m*sqrt(b_o^2 + b_w^2) > \
             0 (b = error bound at the level, m = {}); a P without a usable forecast keeps the \
             token-count terms and never wins a move",
            cost_params.move_margin,
        );
    }
    if drain {
        tracing::info!(
            target: "symmetric_pd",
            queue_room = "forecast",
            prefill_wait = cost_params.wait.as_str(),
            "symmetric_pd cost_function: global queue room forecast: a P also has room while the forecast drain of \
             a cold {}-token probe (newest pushed or polled state + router rows) is at most its \
             lead C / R + T_react, and the walk prices its wait as min(R x drain, work-book \
             backlog); a P without a usable forecast keeps the token rule; the walk and its \
             releases use the token-count terms ({} wait for the requests reserved at arrival)",
            crate::prefill_work::QUEUE_PROBE_TOKENS,
            cost_params.wait.as_str(),
        );
    }
    if let Some(push) = forecast.push.as_ref() {
        tracing::info!(
            target: "symmetric_pd",
            push_port = push.port,
            push_bound_mode = %push.mode,
            push_bound_level = %push.level,
            offset_window_s = crate::prefill_state_push::OFFSET_WINDOW_S,
            wire_schema = crate::prefill_state_push::WIRE_SCHEMA,
            "symmetric_pd cost_function: forecast state push on: one ZMQ SUB per P-role worker at \
             tcp://<P host>:{} (topic prefill_state, routing rank 0; engine \
             SGLANG_PREFILL_STATE_PUSH_PORT); a P's forecast starts from its newest pushed state \
             when it is fresh, has the polled snapshot's epoch and is usable, else from the polled \
             snapshot; pushed forecasts take the push bound table",
            push.port,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::kv_hint::KvWorkerMapEntry;
    use crate::scheduler::placement_cost::{CopyPriceMode, PlacementCostParams, PrefillWait};
    use serde_json::json;

    const P1: &str = "http://p1:30000";
    const P2: &str = "http://p2:30000";
    const P3: &str = "http://p3:30000";
    const D1: &str = "http://d1:30000";

    fn worker(url: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(url.into()),
            url: url.into(),
            mode: WorkerMode::Symmetric,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: Some(18998),
        }))
    }

    fn pool() -> Vec<Arc<Worker>> {
        vec![worker(P1), worker(P2), worker(P3)]
    }

    /// The indexer config: a closed port, `min_tokens` 256 and the
    /// incremental gate `gate`. Every URL of `urls` has a servicer.
    fn kv(gate: u32, urls: &[&str]) -> KvHintConfig {
        KvHintConfig {
            indexer_url: "http://127.0.0.1:9".into(),
            min_tokens: 256,
            min_incremental_tokens: gate,
            local_prefix_fallback: false,
            indexer_timeout_ms: 20,
            worker_map: urls
                .iter()
                .enumerate()
                .map(|(i, url)| KvWorkerMapEntry {
                    id: i as u64 + 1,
                    url: url.to_string(),
                    servicer: format!("servicer-{i}:8000"),
                })
                .collect(),
        }
    }

    fn cost_function(
        dir: &tempfile::TempDir,
        cost_params: PlacementCostParams,
        kv: &KvHintConfig,
    ) -> Arc<CostFunction> {
        cost_function_with(dir, cost_params, kv, false)
    }

    /// As [`cost_function`], with `--symmetric-pd-pull-from-any-holder` set or not.
    fn cost_function_with(
        dir: &tempfile::TempDir,
        cost_params: PlacementCostParams,
        kv: &KvHintConfig,
        pull_from_any_holder: bool,
    ) -> Arc<CostFunction> {
        CostFunction::new(
            SymmetricPdCostFunctionConfig {
                cost_params,
                placement_trace: Some(dir.path().join("cost.jsonl")),
                load_poll_ms: 100,
                unknown_expire_secs: 60,
                global_queue: None,
                trace_session_moves: false,
                forecast: None,
                pull_from_any_holder,
            },
            kv,
        )
    }

    /// An indexer answer: `(url, tokens)` rows, best first, and the GPU
    /// observations.
    fn answer(rows: &[(&str, u32)], gpu: &[(&str, u32)]) -> IndexerAnswer {
        IndexerAnswer {
            candidates: rows
                .iter()
                .map(|(url, tokens)| HintCandidate {
                    worker_url: url.to_string(),
                    servicer_addr: format!("{url}/servicer"),
                    expected_hit_tokens: *tokens,
                })
                .collect(),
            observed_gpu_hits: gpu.iter().map(|(url, n)| (url.to_string(), *n)).collect(),
            indexer_ms: 1.0,
        }
    }

    fn source(c: &Candidate) -> Option<&str> {
        c.hint.as_ref().and_then(|h| h["source_worker_id"].as_str())
    }

    /// A D-role worker with the longest prefix is not a pull source: the
    /// best source of P2 and P3 is P1, the owner P1 gets no hint.
    #[test]
    fn candidates_restrict_pull_sources_to_the_pool() {
        let dir = tempfile::tempdir().unwrap();
        let kv = kv(512, &[D1, P1, P3]);
        let ix = KvHintInjector::new(&kv);
        let cf = cost_function(&dir, PlacementCostParams::default(), &kv);
        let a = answer(&[(D1, 30_000), (P1, 18_000), (P3, 300)], &[]);
        let c = cf.candidates(&ix, Some(&a), 40_000, &pool());
        let urls: Vec<&str> = c.iter().map(|c| c.worker_url.as_str()).collect();
        assert_eq!(urls, [P1, P2, P3]);
        assert!(c[0].hint.is_none(), "the owner pulls nothing");
        assert_eq!((c[0].local_hit, c[0].expected_hit), (18_000, 18_000));
        for c in &c[1..] {
            assert_eq!(source(c), Some(P1), "{c:?}");
            assert_eq!(c.hint.as_ref().unwrap()["expected_hit_tokens"], 18_000);
            assert_eq!(c.expected_hit, 18_000);
        }
        assert_eq!(c[2].local_hit, 300);
    }

    /// P3 holds 17,600 and P1 18,000: the gain 400 is below the cost-function floor
    /// (512), so P3 recomputes those tokens without a hint. The same holds
    /// when the injector gate is 0 and pick() returns the hint.
    #[test]
    fn sub_floor_gain_sends_no_hint() {
        for gate in [512, 0] {
            let dir = tempfile::tempdir().unwrap();
            let kv = kv(gate, &[P1, P2, P3]);
            let ix = KvHintInjector::new(&kv);
            let cf = cost_function(&dir, PlacementCostParams::default(), &kv);
            let a = answer(&[(P1, 18_000), (P3, 17_600)], &[]);
            assert_eq!(ix.pick(&a, P3).is_some(), gate == 0);
            let c = cf.candidates(&ix, Some(&a), 40_000, &pool());
            assert!(c[2].hint.is_none(), "gate {gate}: {:?}", c[2]);
            assert_eq!((c[2].local_hit, c[2].expected_hit), (17_600, 17_600));
            assert!(c[0].hint.is_none(), "gate {gate}: the owner pulls nothing");
            assert_eq!(source(&c[1]), Some(P1), "gate {gate}");
        }
    }

    /// The cost function rule (`admit_reserved`) on fixed answers: the local
    /// hit is the target's own match (clamped to the prompt), the GPU hit
    /// the observation, the expected hit max(local, pulled source match)
    /// clamped to the prompt. An answer URL with a trailing `/` maps to the
    /// registry URL. Without an answer every hit is 0 and no hint is sent.
    #[test]
    fn candidates_follow_the_cost_function_rule() {
        let dir = tempfile::tempdir().unwrap();
        let kv = kv(512, &[P1, P2, P3]);
        let ix = KvHintInjector::new(&kv);
        let cf = cost_function(&dir, PlacementCostParams::default(), &kv);
        let p2_slash = "http://p2:30000/";
        let a = answer(
            &[(P1, 20_000), (p2_slash, 6_000)],
            &[(P1, 12_000), (p2_slash, 6_000)],
        );
        let c = cf.candidates(&ix, Some(&a), 16_000, &pool());
        let got: Vec<(u64, Option<u64>, u64, Option<&str>)> = c
            .iter()
            .map(|c| (c.local_hit, c.gpu_hit, c.expected_hit, source(c)))
            .collect();
        assert_eq!(
            got,
            [
                (16_000, Some(12_000), 16_000, None),
                (6_000, Some(6_000), 16_000, Some(P1)),
                (0, None, 16_000, Some(P1)),
            ]
        );
        for c in cf.candidates(&ix, None, 16_000, &pool()) {
            assert_eq!((c.local_hit, c.gpu_hit, c.expected_hit), (0, None, 0));
            assert!(c.hint.is_none());
        }
    }

    /// `--symmetric-pd-pull-from-any-holder`: the D-role worker D1 with the longest
    /// prefix (30,000) serves the pulls. The targets and their own hits stay
    /// the pool's (P1 18,000, P2 0, P3 300); every target, the pool owner P1
    /// too (gain 12,000 above the floor), pulls from D1 and expects 30,000.
    #[test]
    fn any_holder_lets_a_d_role_holder_serve_pulls() {
        let dir = tempfile::tempdir().unwrap();
        let kv = kv(512, &[D1, P1, P3]);
        let ix = KvHintInjector::new(&kv);
        let cf = cost_function_with(&dir, PlacementCostParams::default(), &kv, true);
        let a = answer(&[(D1, 30_000), (P1, 18_000), (P3, 300)], &[(P1, 9_000)]);
        let c = cf.candidates_any_holder(&ix, Some(&a), 40_000, &pool(), &[worker(D1)]);
        // (worker, local hit, GPU hit, expected hit, pull source) per candidate.
        type Row<'a> = (&'a str, u64, Option<u64>, u64, Option<&'a str>);
        let got: Vec<Row<'_>> = c
            .iter()
            .map(|c| {
                (
                    c.worker_url.as_str(),
                    c.local_hit,
                    c.gpu_hit,
                    c.expected_hit,
                    source(c),
                )
            })
            .collect();
        assert_eq!(
            got,
            [
                (P1, 18_000, Some(9_000), 30_000, Some(D1)),
                (P2, 0, None, 30_000, Some(D1)),
                (P3, 300, None, 30_000, Some(D1)),
            ]
        );
        for c in &c {
            assert_eq!(c.hint.as_ref().unwrap()["expected_hit_tokens"], 30_000);
            assert_eq!(
                c.hint.as_ref().unwrap()["source_servicer_addr"],
                format!("{D1}/servicer")
            );
        }
        // A holder below the target's own hit plus the floor still sends no hint.
        let a = answer(&[(D1, 18_200), (P1, 18_000)], &[]);
        let c = cf.candidates_any_holder(&ix, Some(&a), 40_000, &pool(), &[worker(D1)]);
        assert!(c[0].hint.is_none(), "{:?}", c[0]);
        assert_eq!(source(&c[1]), Some(D1));
        // Without an answer every hit is 0 and no hint is sent, as in candidates().
        for c in cf.candidates_any_holder(&ix, None, 16_000, &pool(), &[worker(D1)]) {
            assert_eq!((c.local_hit, c.gpu_hit, c.expected_hit), (0, None, 0));
            assert!(c.hint.is_none());
        }
    }

    /// A pool URL with a trailing `/` in the answer maps to the registry URL,
    /// so a target never names itself as its pull source; a holder outside
    /// the pool keeps its own URL.
    #[test]
    fn any_holder_maps_pool_urls_and_never_pulls_from_the_target() {
        let dir = tempfile::tempdir().unwrap();
        let kv = kv(512, &[D1, P1, P2, P3]);
        let ix = KvHintInjector::new(&kv);
        let cf = cost_function_with(&dir, PlacementCostParams::default(), &kv, true);
        let p2_slash = "http://p2:30000/";
        let a = answer(&[(p2_slash, 25_000), (D1, 20_000)], &[(p2_slash, 25_000)]);
        let c = cf.candidates_any_holder(&ix, Some(&a), 40_000, &pool(), &[worker(D1)]);
        assert_eq!((c[1].local_hit, c[1].gpu_hit), (25_000, Some(25_000)));
        assert!(
            c[1].hint.is_none(),
            "P2 owns the longest prefix: {:?}",
            c[1]
        );
        assert_eq!(source(&c[0]), Some(P2), "registry URL of P2");
        assert_eq!(source(&c[2]), Some(P2));
        assert_eq!(source_role(P2, &pool()), "p_pool");
        assert_eq!(source_role(D1, &pool()), "outside_p_pool");
        assert_eq!(source_role("-", &pool()), "none");
        assert_eq!(source_role("http://p3:30000/", &pool()), "p_pool");
    }

    /// A holder that is not a healthy registered worker
    /// (crashed, breaker open, removed, or only in `--kv-worker-map`) is
    /// never a pull source, however long its prefix. The caller passes the
    /// healthy registered workers as `donors` (the runtime: its candidates,
    /// breaker allowing); D1 is absent from them here, D2 is healthy.
    #[test]
    fn any_holder_never_picks_a_holder_that_is_not_a_healthy_registered_worker() {
        const D2: &str = "http://d2:30000";
        let dir = tempfile::tempdir().unwrap();
        let kv = kv(512, &[D1, D2, P1, P2, P3]);
        let ix = KvHintInjector::new(&kv);
        let cf = cost_function_with(&dir, PlacementCostParams::default(), &kv, true);
        let a = answer(&[(D1, 30_000), (D2, 25_000), (P1, 18_000)], &[]);
        // D1 dead: every target pulls from the healthy D2 (P1: 25,000 - 18,000
        // = 7,000 above the floor).
        let c = cf.candidates_any_holder(&ix, Some(&a), 40_000, &pool(), &[worker(D2)]);
        for c in &c {
            assert_eq!(source(c), Some(D2), "{c:?}");
            assert_eq!(c.expected_hit, 25_000, "{c:?}");
        }
        // No healthy D-role donor: the P-role pool rule (P1 owns, no hint;
        // P2 and P3 pull from P1), never the dead D1.
        let c = cf.candidates_any_holder(&ix, Some(&a), 40_000, &pool(), &[]);
        assert!(c[0].hint.is_none(), "{:?}", c[0]);
        assert_eq!((source(&c[1]), source(&c[2])), (Some(P1), Some(P1)));
        // A P-role holder outside the healthy pool (breaker open: not in the
        // pool the runtime passes, not among the donors) is never a source.
        let healthy_pool = vec![worker(P2), worker(P3)];
        let c = cf.candidates_any_holder(&ix, Some(&a), 40_000, &healthy_pool, &[worker(D2)]);
        let urls: Vec<&str> = c.iter().map(|c| c.worker_url.as_str()).collect();
        assert_eq!(urls, [P2, P3]);
        for c in &c {
            assert_eq!(source(c), Some(D2), "{c:?}");
        }
        let c = cf.candidates_any_holder(&ix, Some(&a), 40_000, &healthy_pool, &[]);
        for c in &c {
            assert!(c.hint.is_none(), "P1 is breaker-open and D1 dead: {c:?}");
            assert_eq!(c.expected_hit, 0, "{c:?}");
        }
    }

    /// The indexer reply of `rows` (`(worker id, cpu, gpu)`).
    fn reply(rows: &[(u64, u32, u32)]) -> Value {
        let mut instances = serde_json::Map::new();
        for (id, cpu, gpu) in rows {
            instances.insert(id.to_string(), serde_json::json!({"cpu": cpu, "gpu": gpu}));
        }
        serde_json::json!({ "instances": instances })
    }

    /// The body of `reserve` before the flag existed: the answer
    /// restricted to the pool, `candidates`, one work-book transaction.
    fn reserve_baseline(
        cf: &CostFunction,
        ix: &KvHintInjector,
        input: &PlaceInput,
        tokens: u64,
        reply: Option<&Value>,
        pool: &[Arc<Worker>],
    ) -> (Candidate, Reservation, Value) {
        let answer = reply.map(|r| restrict_to_pool(&ix.answer_from_reply(r, 1.0), pool));
        let candidates = cf.candidates(ix, answer.as_ref(), tokens, pool);
        cf.book
            .reserve_cost_session(
                tokens,
                input.rid.clone(),
                candidates,
                cf.cfg.cost_params,
                input.placement_session.as_deref(),
            )
            .unwrap()
    }

    /// Flag off: the same request sequence (D-role holders, GPU-only and
    /// sub-floor hits, a failed indexer query, a reservation finished midway)
    /// gives byte-identical selections, hints and decisions to that
    /// body, and byte-identical cost-trace rows apart from their clocks. The
    /// healthy D-role donor passed to `reserve` is ignored without the flag.
    #[test]
    fn flag_off_reserve_is_byte_identical_to_baseline() {
        let urls = [D1, P1, P2, P3];
        let kv = kv(512, &urls);
        let ix = KvHintInjector::new(&kv);
        let (dir_new, dir_old) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let new = cost_function_with(&dir_new, PlacementCostParams::default(), &kv, false);
        let old = cost_function_with(&dir_old, PlacementCostParams::default(), &kv, false);
        // Worker ids follow kv(): D1 = 1, P1 = 2, P2 = 3, P3 = 4.
        let replies = [
            Some(reply(&[(1, 30_000, 0), (2, 18_000, 9_000), (4, 300, 0)])),
            Some(reply(&[(1, 0, 40_000), (3, 12_000, 12_000)])),
            Some(reply(&[(2, 17_600, 0), (4, 18_000, 18_000)])),
            None,
            Some(reply(&[(1, 64_000, 64_000)])),
            Some(reply(&[(3, 50_000, 0), (1, 49_000, 0), (2, 2_000, 0)])),
        ];
        let (mut keep_new, mut keep_old) = (Vec::new(), Vec::new());
        for (i, r) in replies.iter().enumerate() {
            let input = PlaceInput {
                prompt_len: 60_000,
                rid: Some(format!("r{i}")),
                placement_session: Some(format!("{i}/agent")),
                ..Default::default()
            };
            let (sel_new, res_new, dec_new) = new
                .reserve(&ix, &input, 60_000, r.as_ref(), 1.0, &pool(), &[worker(D1)])
                .unwrap();
            let (sel_old, res_old, dec_old) =
                reserve_baseline(&old, &ix, &input, 60_000, r.as_ref(), &pool());
            assert_eq!(
                format!("{sel_new:?}"),
                format!("{sel_old:?}"),
                "request {i}"
            );
            assert_eq!(dec_new.to_string(), dec_old.to_string(), "request {i}");
            if i == 2 {
                let (mut res_new, mut res_old) = (res_new, res_old);
                res_new.mark_dispatched().unwrap();
                res_old.mark_dispatched().unwrap();
                res_new.finish("prefill_response_success").unwrap();
                res_old.finish("prefill_response_success").unwrap();
            } else {
                keep_new.push(res_new);
                keep_old.push(res_old);
            }
        }
        drop((keep_new, keep_old));
        let rows = |dir: &tempfile::TempDir| -> Vec<Value> {
            std::fs::read_to_string(dir.path().join("cost.jsonl"))
                .unwrap()
                .lines()
                .map(|line| {
                    let mut row: Value = serde_json::from_str(line).unwrap();
                    for key in ["ts_unix_ms", "ts_unix_s", "elapsed_ms"] {
                        row.as_object_mut().unwrap().remove(key);
                    }
                    if let Some(entry) = row.get_mut("entry").and_then(Value::as_object_mut) {
                        for key in [
                            "assigned_unix_ms",
                            "assigned_elapsed_ms",
                            "dispatched_unix_ms",
                            "terminal_unix_ms",
                        ] {
                            entry.remove(key);
                        }
                    }
                    row
                })
                .collect()
        };
        let (rows_new, rows_old) = (rows(&dir_new), rows(&dir_old));
        assert!(rows_new.len() > replies.len(), "{} rows", rows_new.len());
        assert_eq!(rows_new, rows_old);
    }

    /// Only the copy price auto counts sessions: with lambda fixed the END
    /// of a session is not counted; with lambda auto a reserved session
    /// leaves N_active once.
    #[test]
    fn session_end_counts_only_with_auto() {
        let kv = kv(512, &[P1]);
        let ix = KvHintInjector::new(&kv);
        let auto = PlacementCostParams {
            copy_price: 18.4,
            copy_price_mode: CopyPriceMode::Auto {
                min: 4.6,
                max: 1000.0,
            },
            ..PlacementCostParams::default()
        };
        for (cost_params, counted) in [(PlacementCostParams::default(), false), (auto, true)] {
            let dir = tempfile::tempdir().unwrap();
            let cf = cost_function(&dir, cost_params, &kv);
            assert_eq!(cf.is_auto(), counted);
            let input = PlaceInput {
                prompt_len: 1_000,
                rid: Some("r1".into()),
                placement_session: Some("0/agent".into()),
                ..Default::default()
            };
            let (selected, reservation, _) = cf
                .reserve(&ix, &input, 1_000, None, 0.0, &[worker(P1)], &[])
                .unwrap();
            assert_eq!(selected.worker_url, P1);
            drop(reservation);
            assert_eq!(cf.session_end("0/agent"), counted);
            assert!(!cf.session_end("0/agent"), "a session ends once");
        }
    }

    // ---- --global-queue ----

    fn cand(url: &str, hit: u64) -> Candidate {
        Candidate {
            worker_url: url.into(),
            local_hit: hit,
            gpu_hit: None,
            expected_hit: hit,
            hint: None,
        }
    }

    /// Hold-free classification: bound iff h1 - h2 >= C, with C the engine
    /// chunk of the owner P (from /server_info, 8192 when not reported). A
    /// tie of the two largest hits, or no hit at all, is free.
    #[test]
    fn queue_classify_uses_the_owner_chunk() {
        let pool = pool();
        let c = [cand(P1, 20_000), cand(P2, 5_000), cand(P3, 0)];
        let (bound, h1, h2, chunk, source, owner) = queue::classify(&c, &pool, 40_000);
        assert!(bound, "15,000 >= 8,192");
        assert_eq!((h1, h2, chunk, source), (20_000, 5_000, 8192, "default"));
        assert_eq!(owner.as_deref(), Some(P1));
        pool[0].set_chunk_tokens(16_384);
        let (bound, _, _, chunk, source, _) = queue::classify(&c, &pool, 40_000);
        assert!(!bound, "15,000 < 16,384");
        assert_eq!((chunk, source), (16_384, "server_info"));
        // The hits are clamped to the prompt: 12,000 - 5,000 < 8,192.
        pool[0].set_chunk_tokens(0);
        assert!(!queue::classify(&c, &pool, 12_000).0);
        let tie = [cand(P1, 20_000), cand(P2, 20_000), cand(P3, 0)];
        assert!(!queue::classify(&tie, &pool, 40_000).0);
        let cold = [cand(P1, 0), cand(P2, 0), cand(P3, 0)];
        let (bound, h1, _, _, _, owner) = queue::classify(&cold, &pool, 40_000);
        assert!(!bound && h1 == 0 && owner.is_none());
    }

    /// A cost function with global queue over the pool [P1] (load poll 100 ms,
    /// so T_react falls back to 0.6 s), P1 running a launched request with
    /// `remaining` tokens left.
    fn queue_cost_function(
        dir: &tempfile::TempDir,
        max_wait_ms: u64,
        remaining: u64,
    ) -> (Arc<CostFunction>, KvHintConfig, Reservation) {
        queue_cost_function_with(dir, max_wait_ms, remaining, &[P1, P2], false)
    }

    /// As [`queue_cost_function`], with the indexer worker map `urls` and
    /// `--symmetric-pd-pull-from-any-holder` set or not.
    fn queue_cost_function_with(
        dir: &tempfile::TempDir,
        max_wait_ms: u64,
        remaining: u64,
        urls: &[&str],
        pull_from_any_holder: bool,
    ) -> (Arc<CostFunction>, KvHintConfig, Reservation) {
        let kv = kv(512, urls);
        let cf = CostFunction::new(
            SymmetricPdCostFunctionConfig {
                cost_params: PlacementCostParams {
                    copy_price: 9.2,
                    ..PlacementCostParams::default()
                },
                placement_trace: Some(dir.path().join("cost.jsonl")),
                load_poll_ms: 100,
                unknown_expire_secs: 60,
                global_queue: Some(crate::config::GlobalQueueConfig {
                    max_wait_ms,
                    ..Default::default()
                }),
                trace_session_moves: false,
                forecast: None,
                pull_from_any_holder,
            },
            &kv,
        );
        let mut busy = cf
            .book
            .reserve_cost(
                remaining,
                Some("busy".into()),
                vec![cand(P1, 0)],
                Default::default(),
            )
            .unwrap()
            .1;
        busy.mark_dispatched().unwrap();
        set_remaining(&cf, 1, remaining);
        (cf, kv, busy)
    }

    fn set_remaining(cf: &CostFunction, sequence: u64, remaining: u64) {
        let progress = crate::prefill_work::PrefillProgress {
            epoch: "e".into(),
            sequence,
            remaining: [("busy".to_string(), remaining)].into_iter().collect(),
            truncated: false,
        };
        cf.book
            .apply_progress(P1, progress, json!({}), Instant::now())
            .unwrap();
    }

    fn place(rid: &str, session: Option<&str>) -> PlaceInput {
        PlaceInput {
            rid: Some(rid.into()),
            placement_session: session.map(str::to_string),
            ..PlaceInput::default()
        }
    }

    fn reserve_event(dir: &tempfile::TempDir, rid: &str) -> Option<Value> {
        std::fs::read_to_string(dir.path().join("cost.jsonl"))
            .unwrap_or_default()
            .lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == rid)
    }

    fn trace_events(dir: &tempfile::TempDir, event: &str) -> Vec<Value> {
        std::fs::read_to_string(dir.path().join("cost.jsonl"))
            .unwrap_or_default()
            .lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .filter(|e| e["event"] == event)
            .collect()
    }

    /// P1 has 100,000 tokens left (queued 91,808 > the lead depth 16,293):
    /// a bound request (P1 owns 20,000 of it, P2 none) is reserved at
    /// arrival anyway (`queue.class` bound, not held); a free one is held.
    #[tokio::test]
    async fn queue_bound_request_is_never_held() {
        let dir = tempfile::tempdir().unwrap();
        let (cf, kv, busy) = queue_cost_function(&dir, 300_000, 100_000);
        let ix = KvHintInjector::new(&kv);
        let pool = vec![worker(P1), worker(P2)];
        let reply = json!({"instances": {"1": {"cpu": 20_000}}});
        let (selected, reservation, _, queued) = cf
            .reserve_queue(
                &ix,
                &place("bound-1", None),
                30_000,
                Some(&reply),
                1.0,
                &pool,
                &[],
            )
            .await
            .unwrap();
        assert!(!queued);
        assert_eq!(selected.worker_url, P1);
        let g = reserve_event(&dir, "bound-1").unwrap()["global_queue"].clone();
        assert_eq!(
            (g["class"].clone(), g["held"].clone()),
            (json!("bound"), json!(false))
        );
        assert_eq!(
            (g["h1_tokens"].clone(), g["h2_tokens"].clone()),
            (json!(20_000), json!(0))
        );
        drop((reservation, busy));
    }

    /// The guard: a free request on a P without room waits until the max
    /// wait (200 ms), then goes to its cost argmin P (release `starved`).
    #[tokio::test]
    async fn queue_guard_releases_after_the_max_wait() {
        let dir = tempfile::tempdir().unwrap();
        let (cf, kv, busy) = queue_cost_function(&dir, 200, 100_000);
        tokio::spawn(queue::dispatcher_loop(Arc::downgrade(&cf)));
        let ix = KvHintInjector::new(&kv);
        let pool = vec![worker(P1)];
        let started = Instant::now();
        let (selected, reservation, _, queued) = cf
            .reserve_queue(&ix, &place("free-1", None), 5_000, None, 1.0, &pool, &[])
            .await
            .unwrap();
        let waited = started.elapsed();
        assert!(queued);
        assert_eq!(selected.worker_url, P1);
        assert!(waited >= Duration::from_millis(200), "{waited:?}");
        assert!(waited < Duration::from_secs(3), "{waited:?}");
        let g = reserve_event(&dir, "free-1").unwrap()["global_queue"].clone();
        assert_eq!(
            (g["release"].clone(), g["held"].clone()),
            (json!("starved"), json!(true))
        );
        assert_eq!(g["class"], "free");
        assert_eq!(cf.queue_len(), 0);
        drop((reservation, busy));
    }

    /// `--symmetric-pd-pull-from-any-holder` on the queue's arrival path. Flag on:
    /// a bound request (P1 owns 20,000 of 40,000; reserved at arrival) and a
    /// free one (only the D-role D1 holds its prefix; held until the guard
    /// releases it with its arrival candidates) both pull from the donor D1.
    /// Flag off: the donor is ignored and D1 is never a pull source.
    #[tokio::test]
    async fn queue_any_holder_pulls_from_a_d_role_donor() {
        for flag in [true, false] {
            let dir = tempfile::tempdir().unwrap();
            let (cf, kv, busy) = queue_cost_function_with(&dir, 200, 100_000, &[P1, P2, D1], flag);
            tokio::spawn(queue::dispatcher_loop(Arc::downgrade(&cf)));
            let ix = KvHintInjector::new(&kv);
            let donors = [worker(D1)];
            // Worker ids follow kv(): P1 = 1, P2 = 2, D1 = 3.
            let reply = json!({"instances": {"1": {"cpu": 20_000}, "3": {"cpu": 40_000}}});
            let (selected, reservation, _, queued) = cf
                .reserve_queue(
                    &ix,
                    &place("bound-d", None),
                    40_000,
                    Some(&reply),
                    1.0,
                    &[worker(P1), worker(P2)],
                    &donors,
                )
                .await
                .unwrap();
            assert!(!queued);
            assert_eq!(
                reserve_event(&dir, "bound-d").unwrap()["global_queue"]["class"],
                "bound"
            );
            if flag {
                assert_eq!(source(&selected), Some(D1), "{selected:?}");
            } else {
                assert_ne!(source(&selected), Some(D1), "{selected:?}");
            }
            drop(reservation);
            let reply = json!({"instances": {"3": {"cpu": 30_000}}});
            let (selected, reservation, _, queued) = cf
                .reserve_queue(
                    &ix,
                    &place("free-d", None),
                    40_000,
                    Some(&reply),
                    1.0,
                    &[worker(P1)],
                    &donors,
                )
                .await
                .unwrap();
            assert!(queued);
            assert_eq!(selected.worker_url, P1);
            let g = reserve_event(&dir, "free-d").unwrap()["global_queue"].clone();
            assert_eq!(
                (g["class"].clone(), g["release"].clone()),
                (json!("free"), json!("starved"))
            );
            if flag {
                assert_eq!(source(&selected), Some(D1), "{selected:?}");
                assert_eq!(selected.expected_hit, 30_000);
            } else {
                assert!(selected.hint.is_none(), "{selected:?}");
                assert_eq!(selected.expected_hit, 0);
            }
            drop((reservation, busy));
        }
    }

    /// `--symmetric-pd-pull-from-any-holder` through the donors provider (as the
    /// runtime installs it): a free request held with the donor D1 as its
    /// pull source loses the hint when D1 leaves the provider's list during
    /// the hold (removed from the registry). It is released with no hint, its
    /// own hit only and nothing pulled.
    #[tokio::test]
    async fn queue_any_holder_drops_the_hint_of_a_donor_that_left_during_the_hold() {
        let dir = tempfile::tempdir().unwrap();
        let (cf, kv, busy) = queue_cost_function_with(&dir, 200, 100_000, &[P1, P2, D1], true);
        let listed = Arc::new(parking_lot::Mutex::new(vec![worker(D1)]));
        let provider = Arc::clone(&listed);
        cf.set_queue_donors(move || provider.lock().clone());
        tokio::spawn(queue::dispatcher_loop(Arc::downgrade(&cf)));
        let task = {
            let cf = Arc::clone(&cf);
            tokio::spawn(async move {
                let ix = KvHintInjector::new(&kv);
                // Worker ids follow kv(): P1 = 1, P2 = 2, D1 = 3.
                let reply = json!({"instances": {"3": {"cpu": 30_000}}});
                cf.reserve_queue(
                    &ix,
                    &place("left-d", None),
                    40_000,
                    Some(&reply),
                    1.0,
                    &[worker(P1)],
                    &[worker(D1)],
                )
                .await
            })
        };
        let deadline = Instant::now() + Duration::from_secs(2);
        while cf.queue_len() == 0 && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(cf.queue_len(), 1, "held: P1 has no room");
        listed.lock().clear();
        let (selected, reservation, decision, queued) = task.await.unwrap().unwrap();
        assert!(queued);
        assert_eq!(selected.worker_url, P1);
        assert!(selected.hint.is_none(), "D1 left: {selected:?}");
        assert_eq!(selected.expected_hit, selected.local_hit);
        assert_eq!(decision["pulled_tokens"], 0, "{decision}");
        drop((reservation, busy));
    }

    /// The release rule end to end: held while P1 has no room, released
    /// (`room`) by the pass of the next applied view once P1 has 10,000
    /// left (queued 1,808).
    #[tokio::test]
    async fn queue_holds_until_the_p_has_room() {
        let dir = tempfile::tempdir().unwrap();
        let (cf, kv, busy) = queue_cost_function(&dir, 300_000, 100_000);
        tokio::spawn(queue::dispatcher_loop(Arc::downgrade(&cf)));
        let pool = vec![worker(P1)];
        let task = {
            let cf = Arc::clone(&cf);
            tokio::spawn(async move {
                let ix = KvHintInjector::new(&kv);
                cf.reserve_queue(&ix, &place("free-2", None), 5_000, None, 1.0, &pool, &[])
                    .await
                    .map(|(selected, reservation, _, queued)| {
                        (selected.worker_url, reservation, queued)
                    })
            })
        };
        let deadline = Instant::now() + Duration::from_secs(2);
        while cf.queue_len() == 0 && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(cf.queue_len(), 1, "held: no room");
        assert!(reserve_event(&dir, "free-2").is_none());
        set_remaining(&cf, 2, 10_000);
        cf.queue_wake_poll();
        let (url, reservation, queued) = tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .expect("released")
            .unwrap()
            .unwrap();
        assert!(queued && url == P1);
        let g = reserve_event(&dir, "free-2").unwrap()["global_queue"].clone();
        assert_eq!(
            (g["release"].clone(), g["queued_tokens"].clone()),
            (json!("room"), json!("1808"))
        );
        assert!(g["hold_s"].as_f64().unwrap() >= 0.3, "{g}");
        drop((reservation, busy));
    }

    /// The walk and the release score the step pool. Through the queue: P1
    /// (30,000 left) and P2 (100,000 left) have no room; free-4 (20,000 tokens,
    /// 7,000 cached on P1: free) prefers P1 and is held. P1's breaker trips
    /// (the step pool is P2 only) and P2 gets room (10,000 left): the release
    /// goes to P2 without the pull from P1, with no walk mismatch. Otherwise
    /// the reservation would score P1 too and choose it.
    #[tokio::test]
    async fn queue_release_follows_a_breaker_trip_during_the_hold() {
        let dir = tempfile::tempdir().unwrap();
        let kv = kv(512, &[P1, P2]);
        let cf = CostFunction::new(
            SymmetricPdCostFunctionConfig {
                cost_params: PlacementCostParams::default(),
                placement_trace: Some(dir.path().join("cost.jsonl")),
                load_poll_ms: 100,
                unknown_expire_secs: 60,
                global_queue: Some(crate::config::GlobalQueueConfig {
                    max_wait_ms: 300_000,
                    ..Default::default()
                }),
                trace_session_moves: false,
                forecast: None,
                pull_from_any_holder: false,
            },
            &kv,
        );
        let busy = |url: &str, rid: &str, tokens: u64| {
            let mut r = cf
                .book
                .reserve_cost(
                    tokens,
                    Some(rid.into()),
                    vec![cand(url, 0)],
                    Default::default(),
                )
                .unwrap()
                .1;
            r.mark_dispatched().unwrap();
            r
        };
        let left = |url: &str, rid: &str, sequence: u64, remaining: u64| {
            let progress = crate::prefill_work::PrefillProgress {
                epoch: format!("e-{url}"),
                sequence,
                remaining: [(rid.to_string(), remaining)].into_iter().collect(),
                truncated: false,
            };
            cf.book
                .apply_progress(url, progress, json!({}), Instant::now())
                .unwrap();
        };
        let (b1, b2) = (busy(P1, "busy-1", 30_000), busy(P2, "busy-2", 100_000));
        left(P1, "busy-1", 1, 30_000);
        left(P2, "busy-2", 1, 100_000);
        let pool = Arc::new(parking_lot::Mutex::new(vec![worker(P1), worker(P2)]));
        let provider = Arc::clone(&pool);
        cf.queue
            .as_ref()
            .unwrap()
            .set_pool(Box::new(move || provider.lock().clone()));
        tokio::spawn(queue::dispatcher_loop(Arc::downgrade(&cf)));
        let task = {
            let cf = Arc::clone(&cf);
            let kv = kv.clone();
            tokio::spawn(async move {
                let ix = KvHintInjector::new(&kv);
                let reply = json!({"instances": {"1": {"cpu": 7_000}}});
                let arrival = vec![worker(P1), worker(P2)];
                cf.reserve_queue(
                    &ix,
                    &place("free-4", None),
                    20_000,
                    Some(&reply),
                    1.0,
                    &arrival,
                    &[],
                )
                .await
                .map(|(selected, reservation, decision, _)| (selected, reservation, decision))
            })
        };
        let deadline = Instant::now() + Duration::from_secs(2);
        while cf.queue_len() == 0 && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(cf.queue_len(), 1, "held: no room on P1 or P2");
        *pool.lock() = vec![worker(P2)];
        left(P2, "busy-2", 2, 10_000);
        left(P1, "busy-1", 2, 30_000);
        cf.queue_wake_poll();
        let (selected, reservation, decision) = tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .expect("released")
            .unwrap()
            .unwrap();
        assert_eq!(selected.worker_url, P2, "{decision}");
        assert!(selected.hint.is_none(), "no pull from the tripped P1");
        let g = reserve_event(&dir, "free-4").unwrap()["global_queue"].clone();
        assert!(g.get("walk_mismatch").is_none(), "{g}");
        assert_eq!(
            (g["release"].clone(), g["p_release"].clone()),
            (json!("room"), json!(P2))
        );
        drop((reservation, b1, b2));
    }

    /// The drop guard: a client that leaves while its request is held takes
    /// it out of the queue (`queue_drop`, reason client_gone); nothing is
    /// reserved for it.
    #[tokio::test]
    async fn queue_drop_guard_removes_a_left_request() {
        let dir = tempfile::tempdir().unwrap();
        let (cf, kv, busy) = queue_cost_function(&dir, 300_000, 100_000);
        let ix = KvHintInjector::new(&kv);
        let pool = vec![worker(P1)];
        let left = tokio::time::timeout(
            Duration::from_millis(100),
            cf.reserve_queue(&ix, &place("gone-1", None), 5_000, None, 1.0, &pool, &[]),
        )
        .await;
        assert!(left.is_err(), "still held when the client left");
        assert_eq!(cf.queue_len(), 0);
        let drops = trace_events(&dir, "queue_drop");
        assert_eq!(drops.len(), 1, "{drops:?}");
        assert_eq!(
            (drops[0]["rid"].clone(), drops[0]["reason"].clone()),
            (json!("gone-1"), json!("client_gone"))
        );
        assert!(reserve_event(&dir, "gone-1").is_none());
        drop(busy);
    }

    /// A cost function with global queue (guard 300 s) over `urls`.
    fn queue_auto_cost_function(dir: &tempfile::TempDir, kv: &KvHintConfig) -> Arc<CostFunction> {
        CostFunction::new(
            SymmetricPdCostFunctionConfig {
                cost_params: PlacementCostParams::default(),
                placement_trace: Some(dir.path().join("cost.jsonl")),
                load_poll_ms: 100,
                unknown_expire_secs: 60,
                global_queue: Some(crate::config::GlobalQueueConfig {
                    max_wait_ms: 300_000,
                    ..Default::default()
                }),
                trace_session_moves: false,
                forecast: None,
                pull_from_any_holder: false,
            },
            kv,
        )
    }

    /// A fresh progress view of `url` listing `rids` (rid, remaining).
    fn view(cf: &CostFunction, url: &str, sequence: u64, rids: &[(&str, u64)]) {
        let progress = crate::prefill_work::PrefillProgress {
            epoch: format!("e-{url}"),
            sequence,
            remaining: rids.iter().map(|(r, n)| (r.to_string(), *n)).collect(),
            truncated: false,
        };
        cf.book
            .apply_progress(url, progress, json!({}), Instant::now())
            .unwrap();
    }

    /// The release tie-break: every P idle at
    /// each release (each request ends before the next one arrives). The
    /// plain cost-function tie-break sends every cold request to the smallest URL;
    /// global queue releases them in rotation over the P (`queue.tie`: 3 tied,
    /// decided by the rotation), and the walk and the reservation agree.
    #[tokio::test]
    async fn queue_cold_releases_on_idle_p_spread_over_the_p() {
        let urls = [P1, P2, P3];
        let kv = kv(512, &urls);
        let ix = KvHintInjector::new(&kv);
        let pool = pool();
        let dir = tempfile::tempdir().unwrap();
        let plain = cost_function(&dir, PlacementCostParams::default(), &kv);
        let mut chosen = Vec::new();
        for i in 0..3 {
            let rid = format!("plain-{i}");
            let (selected, mut reservation, _) = plain
                .reserve(&ix, &place(&rid, None), 5_000, None, 1.0, &pool, &[])
                .unwrap();
            chosen.push(selected.worker_url);
            reservation.mark_dispatched().unwrap();
            reservation.finish("prefill_response_success").unwrap();
        }
        assert_eq!(
            chosen,
            [P1, P1, P1],
            "plain cost function: every tie to the smallest URL"
        );
        let dir = tempfile::tempdir().unwrap();
        let cf = queue_auto_cost_function(&dir, &kv);
        let mut chosen = Vec::new();
        for i in 0..6u64 {
            for url in urls {
                view(&cf, url, i + 1, &[]);
            }
            let rid = format!("cold-{i}");
            let (selected, mut reservation, _, queued) = cf
                .reserve_queue(&ix, &place(&rid, None), 5_000, None, 1.0, &pool, &[])
                .await
                .unwrap();
            assert!(queued, "free: released through the queue at arrival");
            chosen.push(selected.worker_url);
            reservation.mark_dispatched().unwrap();
            reservation.finish("prefill_response_success").unwrap();
            let g = reserve_event(&dir, &rid).unwrap()["global_queue"].clone();
            assert_eq!(g["release"], "room", "{g}");
            assert_eq!(
                (g["tie"]["tied"].clone(), g["tie"]["decided_by"].clone()),
                (json!(3), json!("rotation")),
                "{g}"
            );
            assert!(g.get("walk_mismatch").is_none(), "{g}");
        }
        assert_eq!(chosen, [P1, P2, P3, P1, P2, P3], "rotation over the idle P");
        let c = cf.queue_counters();
        assert_eq!(
            (c["tie_breaks"].clone(), c["tie_rotations"].clone()),
            (json!(6), json!(6)),
            "{c}"
        );
    }

    /// A P that joins the step pool again (a role switch back to prefill, a
    /// drain end, a breaker that closed) restarts its T_react: its measured
    /// estimate is dropped and the fallback applies until a new sample.
    #[tokio::test]
    async fn queue_t_react_restarts_when_a_p_joins_the_pool_again() {
        let kv = kv(512, &[P1, P2]);
        let ix = KvHintInjector::new(&kv);
        let dir = tempfile::tempdir().unwrap();
        let cf = queue_auto_cost_function(&dir, &kv);
        let pool = Arc::new(parking_lot::Mutex::new(vec![worker(P2)]));
        let provider = Arc::clone(&pool);
        cf.queue
            .as_ref()
            .unwrap()
            .set_pool(Box::new(move || provider.lock().clone()));
        view(&cf, P2, 1, &[]);
        let (selected, mut reservation, _, _) = cf
            .reserve_queue(
                &ix,
                &place("s-1", None),
                5_000,
                None,
                1.0,
                &[worker(P2)],
                &[],
            )
            .await
            .unwrap();
        assert_eq!(selected.worker_url, P2);
        reservation.mark_dispatched().unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        view(&cf, P2, 2, &[("s-1", 5_000)]);
        let urls = [P2.to_string()];
        let lead = cf.book.queue_lead_rows(&urls);
        assert_eq!(lead["by_p"][P2]["react_source"], "measured", "{lead}");
        // P2 leaves the pool (P1 only), then joins it again.
        *pool.lock() = vec![worker(P1)];
        cf.queue_pass(0, 100);
        let lead = cf.book.queue_lead_rows(&urls);
        assert_eq!(lead["by_p"][P2]["react_source"], "measured", "left: kept");
        *pool.lock() = vec![worker(P1), worker(P2)];
        cf.queue_pass(0, 101);
        let lead = cf.book.queue_lead_rows(&urls);
        assert_eq!(lead["by_p"][P2]["react_source"], "fallback", "{lead}");
        assert_eq!(lead["restarts"], 1, "{lead}");
        drop(reservation);
    }

    /// The forecast state of the cost function from the shipped bound
    /// table (tests/fixtures/forecast).
    fn forecast_state() -> crate::config::SymmetricPdForecast {
        let table = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/forecast/bound-params.json"
        );
        crate::config::SymmetricPdForecast(Arc::new(
            crate::prefill_forecast::ForecastConfig::load(
                std::path::Path::new(table),
                crate::prefill_forecast::DEFAULT_BOUND_MODE,
                None,
                0.0,
                None,
            )
            .unwrap(),
        ))
    }

    /// A recorded `prefill_routing` section (tests/fixtures/forecast).
    fn forecast_snapshot() -> Value {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/forecast/routing-snapshot-recorded.json"
        );
        let rec: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        rec["candidate"]["snapshot"]["raw_prefill_routing"].clone()
    }

    fn forecast_cost_function(
        dir: &tempfile::TempDir,
        cost_params: PlacementCostParams,
        queue_room: Option<crate::global_queue::GlobalQueueRoom>,
        kv: &KvHintConfig,
    ) -> Arc<CostFunction> {
        CostFunction::new(
            SymmetricPdCostFunctionConfig {
                cost_params,
                placement_trace: Some(dir.path().join("cost.jsonl")),
                load_poll_ms: 100,
                unknown_expire_secs: 60,
                global_queue: queue_room.map(|room| crate::config::GlobalQueueConfig {
                    room,
                    ..Default::default()
                }),
                trace_session_moves: false,
                forecast: Some(forecast_state()),
                pull_from_any_holder: false,
            },
            kv,
        )
    }

    /// `--prefill-wait forecast` (gate m 0): the cost-function decision has the
    /// forecast terms, the routed request an `forecast_shadow` line, and the routes are
    /// asked for the forecast session key. With the token-count wait and no predicted
    /// room the bound table alone turns nothing on.
    #[test]
    fn forecast_wait_prices_with_the_forecast() {
        let dir = tempfile::tempdir().unwrap();
        let kv = kv(512, &[P1, P2]);
        let ix = KvHintInjector::new(&kv);
        let cost_params = PlacementCostParams {
            wait: PrefillWait::Forecast,
            move_margin: 0.0,
            ..PlacementCostParams::default()
        };
        let cf = forecast_cost_function(&dir, cost_params, None, &kv);
        assert!(cf.wants_forecast_session());
        let forecast = cf.forecast.as_ref().unwrap();
        assert!(forecast.decides && !forecast.push_on());
        assert!(forecast.routing_wanted(P1));
        let now = Instant::now();
        forecast.observe(P1, &forecast_snapshot(), now, now);
        let input = PlaceInput {
            rid: Some("fc-1".into()),
            forecast_session: Some("0/agent-fc".into()),
            ..PlaceInput::default()
        };
        let (_, reservation, decision) = cf
            .reserve(
                &ix,
                &input,
                5_000,
                None,
                1.0,
                &[worker(P1), worker(P2)],
                &[],
            )
            .unwrap();
        assert!(decision["forecast_decision_s"].is_number(), "{decision}");
        assert_eq!(decision["move_margin"], json!(0.0));
        assert!(decision["forecast_chosen"].is_object(), "{decision}");
        let shadows = trace_events(&dir, "forecast_shadow");
        assert_eq!(shadows.len(), 1, "{shadows:?}");
        assert_eq!(shadows[0]["rid"], "fc-1");
        drop(reservation);
        // The engine rejected the section: no snapshot until the re-probe.
        forecast.note_routing_rejected(P2, &anyhow::anyhow!("HTTP 400"));
        assert!(!forecast.routing_wanted(P2) && forecast.routing_wanted(P1));
        forecast.note_routing_accepted(P2);
        assert!(forecast.routing_wanted(P2));

        let dir = tempfile::tempdir().unwrap();
        let cf = forecast_cost_function(&dir, PlacementCostParams::default(), None, &kv);
        assert!(cf.forecast.is_none() && !cf.wants_forecast_session());
        let (_, reservation, decision) = cf
            .reserve(
                &ix,
                &place("tokens-1", None),
                5_000,
                None,
                1.0,
                &[worker(P1)],
                &[],
            )
            .unwrap();
        assert!(decision.get("forecast_decision_s").is_none(), "{decision}");
        assert!(trace_events(&dir, "forecast_shadow").is_empty());
        drop(reservation);
    }

    /// `--forecast-fit`: P1 and P2 have a polled view but
    /// no timeline sample. With the file both are priced by the forecast
    /// (wait_source forecast, a pred_s, forecast_fit sources frozen); without it both
    /// fall back (no_fit).
    #[test]
    fn forecast_frozen_fit_prices_without_timeline_samples() {
        let kv = kv(512, &[P1, P2]);
        let ix = KvHintInjector::new(&kv);
        let frozen = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/forecast/kimi-k26-fit.json"
        );
        let path = std::path::Path::new(frozen);
        let frozen = Arc::new(crate::prefill_forecast::ForecastFit::load(path).unwrap());
        for with_file in [true, false] {
            let dir = tempfile::tempdir().unwrap();
            let cfg = (*forecast_state().0).clone();
            let cfg = cfg.with_frozen_fit(with_file.then(|| Arc::clone(&frozen)));
            let cost_params = PlacementCostParams {
                wait: PrefillWait::Forecast,
                move_margin: 0.0,
                ..PlacementCostParams::default()
            };
            let cf = CostFunction::new(
                SymmetricPdCostFunctionConfig {
                    cost_params,
                    placement_trace: Some(dir.path().join("cost.jsonl")),
                    load_poll_ms: 100,
                    unknown_expire_secs: 60,
                    global_queue: None,
                    trace_session_moves: false,
                    forecast: Some(crate::config::SymmetricPdForecast(Arc::new(cfg))),
                    pull_from_any_holder: false,
                },
                &kv,
            );
            let forecast = cf.forecast.as_ref().unwrap();
            let now = Instant::now();
            forecast.observe(P1, &forecast_snapshot(), now, now);
            forecast.observe(P2, &forecast_snapshot(), now, now);
            let input = PlaceInput {
                rid: Some("fz-1".into()),
                forecast_session: Some("0/agent-fz".into()),
                ..PlaceInput::default()
            };
            let pool = [worker(P1), worker(P2)];
            let (_, reservation, decision) = cf
                .reserve(&ix, &input, 5_000, None, 1.0, &pool, &[])
                .unwrap();
            let rows = decision["candidates"].as_array().unwrap();
            let source = |key: &str| rows.iter().map(|r| r[key].clone()).collect::<Vec<_>>();
            let fits = trace_events(&dir, "forecast_fit");
            assert_eq!(fits.len(), 2, "{fits:?}");
            if with_file {
                let both_forecast = [json!("forecast"), json!("forecast")];
                assert_eq!(source("wait_source"), both_forecast, "{decision}");
                assert!(
                    decision["forecast_chosen"]["pred_s"].is_number(),
                    "{decision}"
                );
                for fit in &fits {
                    assert_eq!(fit["gpu_large"], "frozen", "{fit}");
                    assert_eq!(fit["launch"], "frozen", "{fit}");
                }
            } else {
                let no_fit = [json!("no_fit"), json!("no_fit")];
                assert_eq!(source("wait_fallback_reason"), no_fit, "{decision}");
            }
            drop(reservation);
        }
    }

    /// `--global-queue-room forecast` with the token-count wait: the forecast state
    /// gives the queue drain only (decisions keep the token-count terms, no forecast
    /// session key). A free request on a P without room is held; the step
    /// records the forecast room rule with each P's drain and fallback.
    /// Without timeline samples no P has a forecast, so the token rule (the
    /// work-conserving floor) releases it once P1 has room.
    #[tokio::test]
    async fn queue_room_forecast_falls_back_to_the_token_rule() {
        let dir = tempfile::tempdir().unwrap();
        let kv = kv(512, &[P1]);
        let cf = forecast_cost_function(
            &dir,
            PlacementCostParams::default(),
            Some(crate::global_queue::GlobalQueueRoom::Forecast),
            &kv,
        );
        assert!(!cf.wants_forecast_session());
        let forecast = cf.forecast.as_ref().unwrap();
        assert!(!forecast.decides);
        let now = Instant::now();
        forecast.observe(P1, &forecast_snapshot(), now, now);
        let mut busy = cf
            .book
            .reserve_cost(
                100_000,
                Some("busy".into()),
                vec![cand(P1, 0)],
                Default::default(),
            )
            .unwrap()
            .1;
        busy.mark_dispatched().unwrap();
        set_remaining(&cf, 1, 100_000);
        tokio::spawn(queue::dispatcher_loop(Arc::downgrade(&cf)));
        let pool = vec![worker(P1)];
        let task = {
            let cf = Arc::clone(&cf);
            tokio::spawn(async move {
                let ix = KvHintInjector::new(&kv);
                cf.reserve_queue(&ix, &place("free-p", None), 5_000, None, 1.0, &pool, &[])
                    .await
                    .map(|(selected, reservation, decision, queued)| {
                        (selected.worker_url, reservation, decision, queued)
                    })
            })
        };
        let deadline = Instant::now() + Duration::from_secs(2);
        while cf.queue_len() == 0 && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(
            cf.queue_len(),
            1,
            "held: no room by tokens or by a forecast drain"
        );
        set_remaining(&cf, 2, 10_000);
        cf.queue_wake_poll();
        let (url, reservation, decision, queued) =
            tokio::time::timeout(Duration::from_secs(2), task)
                .await
                .expect("released")
                .unwrap()
                .unwrap();
        assert!(queued && url == P1);
        assert!(decision.get("forecast_decision_s").is_none(), "{decision}");
        let g = reserve_event(&dir, "free-p").unwrap()["global_queue"].clone();
        assert_eq!(g["room_rule"], "forecast", "{g}");
        // The recorded snapshot's 33 waiting rows and the router's busy row
        // keep any forecast drain above the lead: the token rule releases.
        assert_eq!(g["release"], "room", "{g}");
        let fallbacks = g["drain_fallback_by_p"].as_array().unwrap();
        assert_eq!(fallbacks.len(), 1);
        assert!(
            (g["drain_s"].is_null() && fallbacks[0].is_string())
                || (g["drain_s"].is_number() && fallbacks[0].is_null()),
            "{g}"
        );
        // The released P's lead is global queue's C / R + T_react.
        assert!(g["lead_s"].as_f64().unwrap() > 0.0, "{g}");
        drop((reservation, busy));
    }

    /// Flag off: no queue, no lead book, no session-move book; the trace has
    /// no global queue or session_move key.
    #[test]
    fn queue_off_adds_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let kv = kv(512, &[P1]);
        let ix = KvHintInjector::new(&kv);
        let cf = cost_function(&dir, PlacementCostParams::default(), &kv);
        assert!(!cf.queue_enabled() && !cf.wants_session());
        let (_, reservation, decision) = cf
            .reserve(
                &ix,
                &place("off-1", Some("0/a")),
                5_000,
                None,
                1.0,
                &[worker(P1)],
                &[],
            )
            .unwrap();
        let event = reserve_event(&dir, "off-1").unwrap();
        assert!(
            event.get("global_queue").is_none() && event.get("session_move").is_none(),
            "{event}"
        );
        assert!(decision.get("queue_room").is_none());
        let snapshot = cf.book.snapshot();
        assert!(snapshot.get("session_moves").is_none());
        drop(reservation);
    }
}
