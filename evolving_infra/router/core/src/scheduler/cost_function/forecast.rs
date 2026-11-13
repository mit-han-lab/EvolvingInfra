// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! The step-time-model P completion forecast ([`crate::prefill_forecast`]) on the
//! symmetric_pd cost function, with the cost function's inputs:
//!
//! - `--prefill-wait forecast`: each cost-function decision of the cost function
//!   prices its candidates with the forecast and the prediction gate
//!   ([`crate::scheduler::placement_cost::select_with_forecast`]); the work book does
//!   the forecast under its lock ([`PrefillWorkBook::reserve_cost_forecast_session_noted`]).
//! - `--global-queue-room forecast`: the global waiting queue of
//!   `--global-queue` gives a P room while the forecast drain of a cold
//!   one-chunk probe is within the P's lead ([`PrefillWorkBook::queue_step`]); the
//!   decisions keep their own wait (tokens or forecast).
//!
//! The inputs come from the cost function's progress poll, which also asks for the engine `prefill_routing` snapshot (on a symmetric worker
//! `SGLANG_SYMMETRIC_ROUTING_SNAPSHOT=1`; an engine that rejects the include
//! is polled without it and asked again after [`ROUTING_REPROBE`]); its
//! timeline samples feed the online fits ([`ForecastOnline`]) and its view the
//! forecast. With `--forecast-push-port` one state-push
//! subscriber per P-role worker ([`PushHub`]; the engine side is
//! `SGLANG_PREFILL_STATE_PUSH_PORT`) gives a fresher state. Every routed
//! request with a forecast gets an `forecast_shadow` line, and its completion an
//! `forecast_label` line, in the placement trace (or the file of `ForecastConfig::shadow_path`).

use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use serde_json::{json, Value};

use super::CostFunction;
use crate::prefill_forecast::{
    CandidateTokens, ForecastConfig, ForecastContext, ForecastOnline, ForecastView, RouteRecord,
    ShadowSink,
};
use crate::prefill_state_push::PushHub;
use crate::prefill_work::{Candidate, PrefillWorkBook, RoutingSnapshot};
use crate::workers::Worker;

/// After an engine rejected `include=prefill_routing`, the poll of that
/// worker asks for it again after this long.
pub(super) const ROUTING_REPROBE: Duration = Duration::from_secs(30);

/// The period of the state-push supervisor (one subscriber per P-role
/// worker, and the push statistics).
const PUSH_SUPERVISE: Duration = Duration::from_secs(1);

/// The forecast state of one cost function.
pub(super) struct ForecastState {
    pub(super) cfg: Arc<ForecastConfig>,
    /// `--prefill-wait forecast`: the forecast prices the cost-function
    /// decisions. False: it gives the queue drain only
    /// (`--global-queue-room forecast` with the token-count wait).
    pub(super) decides: bool,
    online: Arc<ForecastOnline>,
    shadow: Option<Arc<ShadowSink>>,
    push: Option<Arc<PushHub>>,
    /// The newest polled snapshot of each P-role worker.
    routing: RwLock<HashMap<String, Arc<RoutingSnapshot>>>,
    /// Worker URL -> the last time its engine rejected the include.
    rejections: Mutex<HashMap<String, Instant>>,
}

impl ForecastState {
    /// The forecast state with its shadow log (a new file, or the cost
    /// trace of `book`) and, with `--forecast-push-port`, the
    /// push hub (subscribers start with [`push_supervisor_loop`]). Panics
    /// when the shadow file exists or cannot be created.
    pub(super) fn new(
        cfg: Arc<ForecastConfig>,
        decides: bool,
        book: &Arc<PrefillWorkBook>,
    ) -> Self {
        let online = ForecastOnline::new(Instant::now());
        let online = online
            .with_next_pass_lag(cfg.next_pass_lag)
            .with_frozen_fit(cfg.frozen_fit.clone());
        let online = Arc::new(if cfg.push.is_some() {
            online.with_label_sources()
        } else {
            online
        });
        let shadow = Some(Arc::new(match &cfg.shadow_path {
            Some(path) => ShadowSink::file(path)
                .expect("create the forecast shadow log before serving requests"),
            None => ShadowSink::Trace(Arc::clone(book)),
        }));
        let push = cfg.push.as_ref().map(|push| {
            PushHub::new(
                push.port,
                Arc::clone(&online),
                shadow.clone(),
                Some(Arc::clone(book)),
            )
        });
        Self {
            cfg,
            decides,
            online,
            shadow,
            push,
            routing: RwLock::new(HashMap::new()),
            rejections: Mutex::new(HashMap::new()),
        }
    }

    /// The state push is on.
    pub(super) fn push_on(&self) -> bool {
        self.push.is_some()
    }

    /// The poll of `worker` asks for the `prefill_routing` snapshot, except
    /// within [`ROUTING_REPROBE`] after its engine rejected it.
    pub(super) fn routing_wanted(&self, worker: &str) -> bool {
        self.rejections
            .lock()
            .get(worker)
            .is_none_or(|at| at.elapsed() >= ROUTING_REPROBE)
    }

    /// The engine of `worker` rejected `include=prefill_routing` and was
    /// polled without it. Its old snapshot is dropped (fallback no_view);
    /// the first rejection is logged as a warning.
    pub(super) fn note_routing_rejected(&self, worker: &str, error: &anyhow::Error) {
        let first = self
            .rejections
            .lock()
            .insert(worker.to_string(), Instant::now())
            .is_none();
        self.routing.write().remove(worker);
        if first {
            tracing::warn!(target: "symmetric_pd", worker, %error,
                reprobe_s = ROUTING_REPROBE.as_secs(),
                "symmetric_pd cost_function: the engine rejected include=prefill_routing (a symmetric \
                 worker needs SGLANG_SYMMETRIC_ROUTING_SNAPSHOT=1 and \
                 SGLANG_PREFILL_ROUTING_SNAPSHOT=1); polling without it, so the forecast of \
                 this P falls back (no_view) until the re-probe");
        } else {
            tracing::debug!(target: "symmetric_pd", worker, %error,
                "engine still rejects include=prefill_routing");
        }
    }

    /// The engine of `worker` answered `include=prefill_routing`.
    pub(super) fn note_routing_accepted(&self, worker: &str) {
        if self.rejections.lock().remove(worker).is_some() {
            tracing::info!(target: "symmetric_pd", worker,
                "symmetric_pd cost_function: engine accepts include=prefill_routing again");
        }
    }

    /// One polled snapshot of `worker` (poll sent at `sent`, reply received
    /// at `received`): its timeline samples go into the online fits (and the
    /// `forecast_label` lines of routed requests whose completion they hold into
    /// the shadow log); the view without samples becomes the P's newest
    /// snapshot. An older snapshot (same engine epoch and a sequence that is
    /// not newer, or an older poll) is dropped; a new epoch replaces it.
    pub(super) fn observe(&self, worker: &str, raw: &Value, sent: Instant, received: Instant) {
        let view = crate::prefill_snapshot::SnapshotView::parse(raw, 1, None);
        let mut forecast = ForecastView::parse(raw);
        let labels = self.online.observe(worker, &forecast, sent, received);
        if let Some(sink) = &self.shadow {
            for label in labels {
                sink.write(label);
            }
        }
        forecast.samples = Vec::new();
        forecast.boots = Vec::new();
        let mut store = self.routing.write();
        if let Some(old) = store.get(worker) {
            let same_epoch = old.view.epoch == view.epoch;
            let not_newer = match (old.view.sequence.as_u64(), view.sequence.as_u64()) {
                (Some(old), Some(new)) => new <= old,
                _ => false,
            };
            if sent < old.poll_sent || (same_epoch && not_newer) {
                return;
            }
        }
        store.insert(
            worker.to_string(),
            Arc::new(RoutingSnapshot {
                view,
                poll_sent: sent,
                poll_received: received,
                forecast: Some(Arc::new(forecast)),
            }),
        );
    }

    /// The forecast inputs of one decision that need no work-book state, built
    /// before its lock: per candidate the newest snapshot, the newest pushed
    /// state, the (P, step) fit (computed here on the first decision of a
    /// step; each new fit goes to the placement trace as an `forecast_fit` event) and
    /// the target tokens with the host rule (`session`: the forecast session key).
    pub(super) fn context(
        &self,
        book: &PrefillWorkBook,
        candidates: &[Candidate],
        tokens_len: u64,
        rid: Option<&str>,
        session: Option<&str>,
    ) -> ForecastContext {
        let now_s = self.online.secs(Instant::now());
        let snapshots: Vec<_> = {
            let store = self.routing.read();
            candidates
                .iter()
                .map(|c| store.get(&c.worker_url).cloned())
                .collect()
        };
        let mut prepared = HashMap::with_capacity(candidates.len());
        for (c, snapshot) in candidates.iter().zip(snapshots) {
            let local = c.local_hit.min(tokens_len);
            let tokens = CandidateTokens {
                input: tokens_len as i64,
                local_hit: local as i64,
                gpu_hit: c.gpu_hit.map(|g| g.min(local) as i64),
                expected_hit: c.expected_hit.min(tokens_len).max(local) as i64,
            };
            let (mut prep, new_fit) = self.online.prepare(
                &c.worker_url,
                snapshot,
                rid.unwrap_or(""),
                tokens,
                session,
                now_s,
            );
            // The P's newest pushed state; the decision checks its age,
            // epoch and view under the work-book lock.
            prep.push = self
                .push
                .as_ref()
                .and_then(|hub| hub.state_of(&c.worker_url));
            if let Some(fit) = new_fit {
                let mut event = json!({
                    "event": "forecast_fit",
                    "worker_url": c.worker_url,
                    "forecast_clock_s": now_s,
                });
                if let (Value::Object(event), Value::Object(record)) = (&mut event, fit.record()) {
                    event.extend(record);
                }
                if let Err(error) = book.emit_event(event) {
                    tracing::warn!(target: "symmetric_pd", %error,
                        "forecast fit not written to the placement trace");
                }
            }
            prepared.insert(c.worker_url.clone(), prep);
        }
        ForecastContext {
            cfg: Arc::clone(&self.cfg),
            anchor: self.online.anchor(),
            unix_at_anchor: self.online.unix_of(0.0),
            candidates: prepared,
        }
    }

    /// After a the forecast wait reservation: remember the routed request in the
    /// online state (receipt line, host rule, arrival rates, labels) and
    /// write its `forecast_shadow` line (chosen P, its `pred_s`, decision time). A
    /// decision without the forecast fields (no forecast state) is skipped.
    pub(super) fn record_route(
        &self,
        decision: &Value,
        rid: Option<&str>,
        worker: &str,
        tokens_len: u64,
        session: Option<String>,
    ) {
        let Some(decided_s) = decision["forecast_decision_s"].as_f64() else {
            return;
        };
        let chosen = &decision["forecast_chosen"];
        let rid = rid.unwrap_or("");
        let decided_unix_s = decision["forecast_decision_unix_s"].as_f64().unwrap_or(0.0);
        self.online.record_route(RouteRecord {
            rid: rid.to_string(),
            worker: worker.to_string(),
            decided_s,
            decided_unix_s,
            input: tokens_len as i64,
            prefix: chosen["target_prefix"].as_i64().unwrap_or(0),
            host: chosen["target_host"].as_i64().unwrap_or(0),
            idle: chosen["idle"].as_bool(),
            pred_s: chosen["pred_s"].as_f64(),
            session,
            session_gap_s: chosen["session_gap_s"].as_f64(),
        });
        if let Some(sink) = &self.shadow {
            let mut line = json!({
                "event": "forecast_shadow",
                "forecast_schema": crate::prefill_forecast::SHADOW_SCHEMA,
                "rid": rid,
                "worker_url": worker,
                "decision_mono_s": decided_s,
                "decision_unix_s": decided_unix_s,
                "pred_s": chosen["pred_s"],
                "wait_s": chosen["wait_s"],
                "bound_s": chosen["bound_s"],
                "idle": chosen["idle"],
                "path": chosen["path"],
                "bucket": chosen["bucket"],
                "fallback": chosen["fallback"],
                "owner": decision["owner"],
                "moved": decision["detour"],
                "gain_s": decision["forecast_gain_s"],
                "margin_s": decision["forecast_margin_s"],
                "protect_reason": decision["forecast_protect_reason"],
            });
            if self.push.is_some() {
                // `push` or `poll`: live MAPE per state source.
                line["state_source"] = chosen["state_source"].clone();
            }
            sink.write(line);
        }
    }
}

/// With `--forecast-push-port`: every [`PUSH_SUPERVISE`], keep
/// one push subscriber per P-role worker of `members` (new workers get one,
/// gone ones lose theirs) and write the push statistics when they are due.
pub(super) async fn push_supervisor_loop(
    weak: Weak<CostFunction>,
    members: Arc<dyn Fn() -> Vec<Arc<Worker>> + Send + Sync>,
) {
    let mut interval = tokio::time::interval(PUSH_SUPERVISE);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        interval.tick().await;
        let Some(cf) = weak.upgrade() else { return };
        let Some(hub) = cf
            .forecast
            .as_ref()
            .and_then(|forecast| forecast.push.clone())
        else {
            return;
        };
        drop(cf);
        let urls: Vec<String> = (*members)().iter().map(|w| w.url.clone()).collect();
        hub.ensure(&urls);
        hub.maybe_emit_stats(Instant::now());
    }
}
