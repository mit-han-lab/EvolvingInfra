// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! The work book of the placement cost function: the uncached prefill work of
//! each request reserved on its P, held until the P request terminates.
//!
//! Each entry starts with the request's uncached tokens on its P and takes
//! the per-RID remaining work that the P's progress poll reports after each
//! chunk. Missing RIDs never imply completed work. A complete P HTTP response
//! (including an error response) is terminal. A transport error is not proof
//! of engine termination: the caller marks the entry unknown, which keeps its
//! work. The reservation moves into the detached P task, independently of the
//! client/D lifetime. An ended entry leaves the work book [`RETIRE_AFTER_S`]
//! seconds after it ended; the snapshot keeps its status count.
//!
//! The unknown expiry (`--symmetric-pd-unknown-expire-secs`) moves an unknown
//! entry to `expired` when its P has answered without its RID for that time:
//! an applied progress reply of its P was sent at least the expiry after the
//! entry's last evidence (the later of the moment it became unknown and the
//! send time of the last poll whose reply listed it). A P without applied
//! replies (dead, partitioned, polls timing out) keeps its unknown entries.
//! Expiry is terminal and releases the entry's work once, as a finish does
//! (trace event `expired`, reason `unknown_expired_no_engine_progress`). An
//! SGLang P aborts a prefill when the router's connection closes, and its
//! progress lists only RIDs whose first chunk has launched; a request still
//! in its queue and a dropped one look alike, and the expiry bounds that
//! error. A reply that lists the RID of an expired entry again measures the
//! error: the first such reply per entry writes the event
//! `expired_rid_listed_again` (the snapshot counts these entries); the work book
//! does not take the work back. The check runs when a progress reply of the
//! entry's P is applied and at each [`PrefillWorkBook::snapshot`], only while unknown
//! entries exist.
//!
//! The placement selects argmin of backlog + own uncached + unhidden pull
//! time + copy price (see [`crate::scheduler::placement_cost::placement_cost`]). The
//! backlog is the P's pending uncached work on this work book.
//!
//! With `--copy-price auto` or `--prefill-rate-tokens-per-s auto` this
//! book also keeps the measurements of
//! [`crate::scheduler::placement_cost::AutoPriceState`], created at the first auto
//! decision. Only the entries that an auto decision reserved feed it. The P of
//! such an entry has one more in-flight request from its dispatch until its
//! prefill ends: the first progress snapshot that shows 0 remaining work for
//! its RID (credited with its computed tokens), the P response
//! (`prefill_response_success` credits too; any other reason does not), a
//! transport error or a drop after dispatch (unknown, no credit). A rollback
//! before dispatch releases only its session. Each end is applied once, so no
//! path leaks an in-flight request. The computed tokens are the larger of the
//! reserved uncached estimate and the largest remaining work that a snapshot
//! showed. Each decision resolves lambda and R from these measurements
//! ([`crate::scheduler::placement_cost::PlacementCostParams::resolved`]) and then prices the
//! candidates with them; the trace event `cost_auto_p_empty` records each
//! busy period that ends, and the one-second snapshot has `auto_price`.
//!
//! With `--prefill-wait forecast` each candidate's wait and service come from the forecast
//! forecast ([`crate::prefill_forecast`]). The fits, the target tokens and the parsed
//! snapshot are prepared before this lock
//! ([`crate::prefill_forecast::ForecastContext`]); the forecast runs under it, because
//! its queue holds the router's own rows of that P that the snapshot does not
//! show (active entries, and entries that finished after the snapshot time,
//! decided after it). The selection is the prediction gate
//! ([`crate::scheduler::placement_cost::select_with_forecast`]).

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Debug, Deserialize)]
pub struct PrefillProgress {
    pub epoch: String,
    pub sequence: u64,
    pub remaining: BTreeMap<String, u64>,
    #[serde(default)]
    pub truncated: bool,
}

struct ProgressCursor {
    epoch: String,
    sequence: u64,
    sent_at: Instant,
}

/// One parsed engine `prefill_routing` snapshot of a P worker and the time
/// the router sent the poll that returned it (the start of the view age).
#[derive(Debug)]
pub struct RoutingSnapshot {
    pub view: crate::prefill_snapshot::SnapshotView,
    pub poll_sent: Instant,
    /// When the router received the reply (the forecast wait: the snapshot time
    /// is this minus the engine's `age_s`).
    pub poll_received: Instant,
    /// the forecast wait: the same snapshot in forecast form (its timeline samples are
    /// already taken out). None for the other wait sources.
    pub forecast: Option<Arc<crate::prefill_forecast::ForecastView>>,
}

/// Finished entries stay visible to the forecast wait router rows this long
/// (s): a request decided after a snapshot and finished before the decision
/// still ran on P after the snapshot time. Longer than any usable view age.
const RECENT_FINISHED_S: f64 = 60.0;
/// An entry leaves the book this many seconds after it ended (finished,
/// rolled back or expired): the forecast reads finished entries for
/// [`RECENT_FINISHED_S`], and a late call on its reservation finds it until
/// then. The snapshot keeps the status counts of the entries that left.
const RETIRE_AFTER_S: f64 = RECENT_FINISHED_S;

/// The forecast of one candidate for the forecast wait, under the lock.
/// `fallback` is None exactly when the candidate is usable: `out` has
/// `pred_s` and `wait_s`, and `bound_s` is set.
#[derive(Default)]
struct ForecastProbe {
    out: Option<crate::forecast::Output>,
    bound_s: Option<f64>,
    bucket: Option<&'static str>,
    fallback: Option<&'static str>,
    error: Option<String>,
    notes: crate::prefill_forecast::BuildNotes,
    /// The degraded flags of the view the forecast started from (sorted;
    /// for example `engine_stale`).
    degraded: Vec<&'static str>,
    /// Decision time minus the snapshot time t0 (s).
    t0_age_s: Option<f64>,
    /// Decision time minus the poll send time, plus the engine's age (s;
    /// the age the limit `max_state_age_s` applies to). A pushed state:
    /// decision time minus its state time.
    view_age_s: Option<f64>,
    view_key: Value,
    /// The prepared inputs (fit, target, host rule) of the candidate.
    prep: Option<crate::prefill_forecast::ForecastPrepared>,
    /// The state push is on (the trace then has the push keys).
    push_on: bool,
    /// `push` or `poll`: the state the forecast started from.
    source: Option<&'static str>,
    /// Push on, the pushed state not used: why (`no_push`,
    /// `push_unverified`, `push_too_old`, `push_unusable`).
    push_fallback: Option<&'static str>,
    /// Decision time minus the state time of the P's newest pushed state.
    push_age_s: Option<f64>,
}

/// The pushed state a decision at `d` uses, or why not: none yet
/// (`no_push`); no polled snapshot of the same engine epoch to check it
/// against (`push_unverified`: the push and the poll read one tracker, so
/// another epoch is a wrong endpoint or a restart between them); older
/// than the view-age limit (`push_too_old`); a view that gives no forecast
/// (`push_unusable`).
fn push_pick<'a>(
    push: Option<&'a crate::prefill_state_push::PushState>,
    poll_epoch: Option<&str>,
    d: f64,
    max_age_s: f64,
) -> Result<&'a crate::prefill_state_push::PushState, &'static str> {
    let p = push.ok_or("no_push")?;
    if poll_epoch != Some(p.epoch.as_str()) {
        return Err("push_unverified");
    }
    let age = d - p.t0_s;
    if !(age.is_finite() && age <= max_age_s) {
        return Err("push_too_old");
    }
    if p.view.unusable.is_some() {
        return Err("push_unusable");
    }
    Ok(p)
}

impl ForecastProbe {
    /// The wait terms of a usable forecast: B = wait_s, S = pred_s - wait_s.
    fn terms(&self) -> Option<crate::scheduler::placement_cost::WaitTerms> {
        if self.fallback.is_some() {
            return None;
        }
        let out = self.out.as_ref()?;
        let (pred, wait) = (out.pred_s?, out.wait_s?);
        Some(crate::scheduler::placement_cost::WaitTerms {
            wait_s: wait,
            service_s: pred - wait,
        })
    }

    fn usable_bound(&self) -> Option<f64> {
        self.terms().and(self.bound_s)
    }

    /// The forecast arrival rate of the candidate's fit (requests per second).
    fn arrival_rate(&self) -> Option<f64> {
        self.prep
            .as_ref()
            .and_then(|p| p.fit.as_ref())
            .and_then(|f| f.arrivals)
            .map(|a| a.rate)
    }

    /// The forecast started from a pushed state.
    fn pushed(&self) -> bool {
        self.source == Some("push")
    }
}

/// The forecast wait keys of one candidate row of the placement decision: the
/// forecast and its bound, the input summary, the target tokens and host
/// rule, and the fit sources of the (P, step).
fn forecast_row_value(
    q: &ForecastProbe,
    cost: &crate::scheduler::placement_cost::CandidateCost,
) -> Value {
    let out = q.out.as_ref();
    let prep = q.prep.as_ref();
    let fit = prep.and_then(|p| p.fit.as_ref());
    // Three json! blocks: one block of all keys exceeds the macro recursion
    // limit of the crate.
    let forecast = json!({
        "wait_source": if q.fallback.is_none() { "forecast" } else { "tokens_fallback" },
        "wait_fallback_reason": q.fallback,
        "wait_s": cost.wait_s,
        "service_s": cost.service_s,
        "view_age_s": q.view_age_s,
        "view_key": q.view_key,
        "forecast_pred_s": out.and_then(|o| o.pred_s),
        "forecast_wait_s": out.and_then(|o| o.wait_s),
        "forecast_bound_s": q.bound_s,
        "forecast_bucket": q.bucket,
        "forecast_idle": out.map(|o| o.idle),
        "forecast_path": out.map(|o| o.path),
    });
    let shape = json!({
        "forecast_receipt_s": out.map(|o| o.receipt_s),
        "forecast_passes_before": out.map(|o| o.passes_before),
        "forecast_x_passes": out.map(|o| o.x_passes),
        "forecast_n_inflight": out.map(|o| o.n_inflight),
        "forecast_partial": out.map(|o| o.partial),
        "forecast_n_queue": out.map(|o| o.n_queue),
        "forecast_n_dispatched": out.map(|o| o.n_dispatched),
        "forecast_n_prefix_unknown": out.map(|o| o.n_prefix_unknown),
        "forecast_tokens_ahead": out.map(|o| o.tokens_ahead),
        "forecast_x_uncached": out.map(|o| o.x_uncached),
        "forecast_x_host": out.map(|o| o.x_host),
        "forecast_u_tokens": out.and_then(|o| o.u_tokens),
        "forecast_n_held": out.map(|o| o.n_held),
        "forecast_x_held": out.map(|o| o.x_held),
        "forecast_f_full_s": out.and_then(|o| o.f_full_s),
        "forecast_f_after_s": out.and_then(|o| o.f_after_s),
    });
    let inputs = json!({
        "forecast_router_rows": q.notes.router_rows,
        "forecast_router_rows_cut": q.notes.router_rows_cut,
        "forecast_bootstrap_cut": q.notes.bootstrap_cut,
        "forecast_null_estimated": q.notes.null_estimated,
        "forecast_null_cold": q.notes.null_cold,
        "forecast_pull_rows": q.notes.pull_rows,
        "forecast_pull_tokens": q.notes.pull_tokens,
        "forecast_degraded": q.degraded,
        "forecast_target_prefix": prep.map(|p| p.target.prefix),
        "forecast_target_host": prep.map(|p| p.target.host),
        "forecast_host_rule": prep.map(|p| p.host_rule),
        "forecast_session_gap_s": prep.and_then(|p| p.session_gap_s),
        "forecast_t0_age_s": q.t0_age_s,
        "forecast_fit_step_s": fit.map(|f| f.step),
        "forecast_gpu_sources": fit.map(|f| json!([f.gpu_sources.small, f.gpu_sources.large])),
        "forecast_launch_source": fit.map(|f| f.launch.source),
        "forecast_rx": prep
            .and_then(|p| p.params.as_ref())
            .and_then(|p| p.rx)
            .map(|rx| json!({"a": rx.a, "b": rx.b, "shift": rx.shift})),
        "forecast_arrival_rate_per_s": q.arrival_rate(),
        "forecast_error": q.error,
    });
    // State push on: the state source and the P's newest pushed state.
    let pushed = q.push_on.then(|| {
        let push = prep.and_then(|p| p.push.as_ref());
        json!({
            "forecast_state_source": q.source,
            "forecast_push_fallback": q.push_fallback,
            "forecast_push_unusable": push.and_then(|p| p.view.unusable),
            "forecast_push_seq": push.map(|p| p.seq),
            "forecast_push_trigger": push.map(|p| p.trigger.as_str()),
            "forecast_push_snapshot_source": push.and_then(|p| p.snapshot_source.as_deref()),
            "forecast_push_age_s": q.push_age_s,
            "forecast_push_superseded_seq": push.and_then(|p| p.superseded_seq),
            "forecast_push_offset_s": push.map(|p| p.offset_s),
        })
    });
    let mut row = serde_json::Map::new();
    for part in [forecast, shape, inputs].into_iter().chain(pushed) {
        if let Value::Object(map) = part {
            row.extend(map);
        }
    }
    Value::Object(row)
}

/// The forecast wait keys of the placement decision: the gate settings and
/// numbers, the bound table, the decision time on the forecast clock, the chosen
/// P's forecast (the shadow log reads it) and the fit step of each P.
fn forecast_decision_value<'a>(
    params: &crate::scheduler::placement_cost::PlacementCostParams,
    selected: &crate::scheduler::placement_cost::CostSelection,
    chosen: Option<&ForecastProbe>,
    forecasts: usize,
    forecast: Option<&crate::prefill_forecast::ForecastContext>,
    decided_at: Instant,
    steps: impl Iterator<Item = (&'a str, Option<f64>)>,
) -> Value {
    let cfg = forecast.map(|ctx| &ctx.cfg);
    let out = chosen.and_then(|q| q.out.as_ref());
    let prep = chosen.and_then(|q| q.prep.as_ref());
    let decision_s =
        forecast.map(|ctx| crate::prefill_forecast::secs_since(ctx.anchor, decided_at));
    let steps: serde_json::Map<String, Value> = steps
        .map(|(url, step)| (url.to_string(), json!(step)))
        .collect();
    json!({
        "move_margin": params.move_margin,
        "forecast_bound_level": cfg.map(|c| c.level.as_str()),
        "forecast_bound_mode": cfg.map(|c| c.mode.as_str()),
        "forecast_bound_sha256": cfg.map(|c| c.bound_sha256.as_str()),
        "forecast_rq_margin_s": cfg.map(|c| c.rq_margin_s),
        "forecast_gain_s": selected.owner_gain_s,
        "forecast_margin_s": selected.margin_s,
        "forecast_protected": selected.margin_protected,
        "forecast_protect_reason": selected.protect_reason,
        "forecast_decision_s": decision_s,
        "forecast_decision_unix_s": forecast.zip(decision_s).map(|(ctx, d)| ctx.unix_at_anchor + d),
        "forecast_forecasts": forecasts,
        "forecast_fit_steps": steps,
        "forecast_chosen": {
            "pred_s": out.and_then(|o| o.pred_s),
            "wait_s": out.and_then(|o| o.wait_s),
            "bound_s": chosen.and_then(|q| q.bound_s),
            "idle": out.map(|o| o.idle),
            "path": out.map(|o| o.path),
            "bucket": chosen.and_then(|q| q.bucket),
            "fallback": chosen.and_then(|q| q.fallback),
            "target_prefix": prep.map(|p| p.target.prefix),
            "target_host": prep.map(|p| p.target.host),
            "session_gap_s": prep.and_then(|p| p.session_gap_s),
        },
    })
}

/// The caller applies the actual pull-hint gate before constructing a candidate.
/// `expected_hit` includes a remote prefix only when `hint` will be forwarded.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub worker_url: String,
    pub local_hit: u64,
    /// Observed GPU prefix; None stays unknown, and known hits clamp to local_hit.
    pub gpu_hit: Option<u64>,
    pub expected_hit: u64,
    pub hint: Option<Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Reserved,
    Dispatched,
    Unknown,
    Finished,
    RolledBack,
    /// Terminal: an unknown entry without engine progress for the expiry
    /// time (see the module docs).
    Expired,
}

/// The reason of every `expired` transition.
const UNKNOWN_EXPIRED_REASON: &str = "unknown_expired_no_engine_progress";

impl Status {
    fn label(self) -> &'static str {
        match self {
            Self::Reserved => "reserved",
            Self::Dispatched => "dispatched",
            Self::Unknown => "unknown",
            Self::Finished => "finished",
            Self::RolledBack => "rolled_back",
            Self::Expired => "expired",
        }
    }

    fn active(self) -> bool {
        matches!(self, Self::Reserved | Self::Dispatched | Self::Unknown)
    }
}

#[derive(Debug)]
struct Entry {
    id: u64,
    rid: Option<String>,
    worker_url: String,
    input_tokens: u64,
    local_hit: u64,
    gpu_hit: Option<u64>,
    expected_hit: u64,
    uncached_tokens: u64,
    seen_engine_progress: bool,
    /// Its RID appeared in an engine progress snapshot of its worker after
    /// dispatch, that is, its first prefill chunk has launched. Until then
    /// (transit, bootstrap queue, waiting queue) it counts in
    /// `pending_unacknowledged_tokens`.
    acknowledged: bool,
    cost: u128,
    hint: Option<Value>,
    status: Status,
    assigned_unix_ms: u64,
    assigned_elapsed_ms: u64,
    dispatched_unix_ms: Option<u64>,
    terminal_unix_ms: Option<u64>,
    reason: Option<String>,
    /// The placement decision time (monotonic).
    decided_at: Instant,
    /// When the router received the progress poll that first listed this
    /// RID (its first prefill chunk had launched by then).
    ack_at: Option<Instant>,
    /// the forecast wait: the router's estimate of the request on its P at the
    /// decision, (prefix, host) tokens: its forecast queue row while the engine
    /// does not show it, and the null-prefix rule.
    forecast_estimate: Option<(i64, i64)>,
    /// cost-function auto bookkeeping; not in the trace.
    auto: AutoTrack,
    /// When it became unknown (the unknown expiry); not in the trace.
    unknown_since: Option<Instant>,
    /// The send time of the last progress poll whose reply listed its RID
    /// after dispatch (the unknown expiry); not in the trace.
    last_progress_seen: Option<Instant>,
    /// `--global-queue`: its release, until its first sighting; not in
    /// the trace.
    queue_lead_mark: Option<QueueLeadMark>,
    /// Its rid left `used_rids` when it rolled back before dispatch, so a
    /// later entry may hold the same rid.
    rid_released: bool,
}

/// The cost-function auto bookkeeping of one entry (see the module docs).
#[derive(Debug, Default)]
struct AutoTrack {
    /// An auto decision reserved this entry: its events feed the auto state.
    tracked: bool,
    /// The request's session key ([`crate::scheduler::placement_cost::placement_session_hash`]).
    session: Option<u64>,
    /// Counted in its P's in-flight requests (dispatched, prefill not ended).
    in_flight: bool,
    /// Its prefill end has been applied (P and session released).
    done: bool,
    /// The computed prefill tokens: the reserved uncached estimate, raised by
    /// any larger remaining work in a progress snapshot.
    peak_tokens: u64,
}

impl Entry {
    fn value(&self) -> Value {
        json!({
            "reservation_id": self.id,
            "rid": self.rid,
            "worker_url": self.worker_url,
            "input_tokens": self.input_tokens,
            "local_hit": self.local_hit,
            "gpu_hit": self.gpu_hit,
            "expected_hit": self.expected_hit,
            "uncached_tokens": self.uncached_tokens,
            "seen_engine_progress": self.seen_engine_progress,
            "initial_uncached_tokens": self.input_tokens - self.expected_hit,
            "cost": self.cost.to_string(),
            "hint": self.hint,
            "status": self.status.label(),
            "assigned_unix_ms": self.assigned_unix_ms,
            "assigned_elapsed_ms": self.assigned_elapsed_ms,
            "dispatched_unix_ms": self.dispatched_unix_ms,
            "terminal_unix_ms": self.terminal_unix_ms,
            "reason": self.reason,
        })
    }
}

#[derive(Debug, Default)]
struct WorkerCost {
    cost: u128,
    pending_uncached_tokens: u128,
    /// The part of `pending_uncached_tokens` held by active entries that no
    /// engine progress snapshot has reported yet (routed, first prefill chunk
    /// not launched: transit, bootstrap queue or engine waiting queue).
    pending_unacknowledged_tokens: u128,
    active_requests: u64,
}

struct State {
    next_id: u64,
    event_sequence: u64,
    entries: BTreeMap<u64, Entry>,
    active_by_rid: BTreeMap<String, u64>,
    used_rids: BTreeSet<String>,
    progress: BTreeMap<String, ProgressCursor>,
    /// cost-function the forecast wait: decisions per fallback reason where a P used the
    /// token-count terms (cumulative; P x decision counts).
    wait_fallbacks: BTreeMap<String, u64>,
    /// cost-function auto (copy price or rate `auto` only): the router's
    /// measurements, created at the first auto decision.
    auto_price: Option<crate::scheduler::placement_cost::AutoPriceState>,
    /// The IDs of the unknown entries (the snapshot's `unknown_requests` is
    /// its size).
    unknown: BTreeSet<u64>,
    /// P-legs closed with [`crate::proxy::READER_GONE_FINISH_REASON`]: the
    /// engine ended them because their reader left (cancelled, not failed).
    prefill_reader_gone: u64,
    /// The unknown expiry: RID -> (ID of the expired entry, progress replies
    /// of its P that listed the RID after the expiry).
    /// The worker URL is kept, so the record outlives the entry's retirement.
    expired_by_rid: BTreeMap<String, (u64, u64, String)>,
    /// The unknown expiry: expired entries whose RID a reply listed again.
    expired_listed_again: u64,
    /// `--global-queue` only (None: off): T_react per P.
    queue_lead: Option<QueueLeadBook>,
    /// `--symmetric-pd-trace-session-moves` only (None: off).
    session_moves: Option<SessionMoveBook>,
    /// Entries that finished in the last [`RECENT_FINISHED_S`] seconds
    /// (finish time, id), oldest first: the forecast wait router rows.
    recent_finished: VecDeque<(Instant, u64)>,
    /// the forecast wait with the state push: candidates per state source (`push`,
    /// `poll`, `none`) and per reason the pushed state was not used
    /// (cumulative; P x decision counts).
    forecast_push_sources: BTreeMap<String, u64>,
    forecast_push_fallbacks: BTreeMap<String, u64>,
    workers: BTreeMap<String, WorkerCost>,
    // Last admission's health snapshot; not a claim of continuous health.
    healthy_candidates: Vec<String>,
    /// The JSONL trace, written and flushed under the work-book lock.
    trace: Option<BufWriter<File>>,
    io_failure: Option<String>,
    /// Ended entries in the order they ended (end time, ID); each leaves
    /// `entries` [`RETIRE_AFTER_S`] after it ended ([`PrefillWorkBook::retire_ended`]).
    ended: VecDeque<(Instant, u64)>,
    /// Status counts of the entries that left `entries`, per worker.
    retired_counts: BTreeMap<String, BTreeMap<&'static str, u64>>,
    /// Entries that left `entries`.
    retired_entries: u64,
}

/// `--global-queue`: T_react of each P, the time from a release of the
/// queue to the first progress poll whose reply lists the released RID. A
/// time-weighted mean with time constant tau (each sample has weight 1 and
/// decays as exp(-age / tau)). Only a release to an idle P (no queued and
/// no launched work: no running pass) gives a sample, so a deeper queue does
/// not raise its own lead.
#[derive(Debug)]
struct QueueLeadBook {
    tau_s: f64,
    /// T_react of a P without a sample (s).
    fallback_s: f64,
    by_p: BTreeMap<String, QueueLeadEstimate>,
    /// The lead of each P at the last step: (lead_s, depth_tokens, R).
    last_step: BTreeMap<String, (f64, u64, f64)>,
    /// Releases marked for a sample, and the others.
    marked: u64,
    unmarked: u64,
    /// First sightings of marked releases (samples taken).
    samples: u64,
    /// The URL of the last release whose cost-function total tied with another P
    /// (the rotation of [`crate::scheduler::placement_cost::break_cost_tie`]).
    tie_last: Option<String>,
    /// T_react estimates dropped because their P joined the step pool again
    /// ([`PrefillWorkBook::queue_lead_restart`]).
    restarts: u64,
}

impl QueueLeadBook {
    /// T_react of `url` now: (seconds, source, samples).
    fn react(&self, url: &str) -> (f64, &'static str, u64) {
        self.by_p
            .get(url)
            .map_or((self.fallback_s, "fallback", 0), |e| {
                (e.sum / e.weight, "measured", e.samples)
            })
    }

    /// One sample of `url` at `at`; returns the new T_react.
    fn observe(&mut self, url: &str, at: Instant, sample_s: f64) -> f64 {
        let tau_s = self.tau_s;
        self.samples += 1;
        let e = self
            .by_p
            .entry(url.to_string())
            .or_insert(QueueLeadEstimate {
                sum: 0.0,
                weight: 0.0,
                samples: 0,
                last_at: at,
                last_sample_s: sample_s,
            });
        let decay = (-at.saturating_duration_since(e.last_at).as_secs_f64() / tau_s).exp();
        e.sum = e.sum * decay + sample_s;
        e.weight = e.weight * decay + 1.0;
        e.samples += 1;
        e.last_at = e.last_at.max(at);
        e.last_sample_s = sample_s;
        e.sum / e.weight
    }
}

/// The T_react estimate of one P ([`QueueLeadBook`]).
#[derive(Debug, Clone, Copy)]
struct QueueLeadEstimate {
    /// Decayed sum of the samples and of their weights, exact at `last_at`.
    sum: f64,
    weight: f64,
    samples: u64,
    last_at: Instant,
    last_sample_s: f64,
}

/// A release of the `--global-queue` queue, kept on its entry until the
/// first sighting of its RID.
#[derive(Debug, Clone, Copy)]
struct QueueLeadMark {
    released_at: Instant,
    /// Its P was idle at the release (no queued and no launched work, fresh
    /// view): the first sighting is a T_react sample.
    sample: bool,
    /// Host-restore tokens (local hit minus GPU hit; all of the local hit
    /// when the GPU hit is unknown) and the pulled tokens of the released
    /// request: the sample includes their time before the first chunk.
    restore_tokens: u64,
    pull_tokens: u64,
}

/// The candidates of a held request over the step pool `urls`: its
/// arrival candidate for each P still in the pool, a cold candidate for a P
/// that joined, none for a P that left. A pull hint whose source is neither
/// in the pool nor among `sources` (`--symmetric-pd-pull-from-any-holder`: the
/// registered workers of any role at this step, breaker allowing when the
/// breaker is on; empty without the flag) is dropped (that P then recomputes
/// those tokens). The walk and the release score this one set.
fn queue_pool_candidates(
    candidates: &[Candidate],
    urls: &[String],
    sources: &[String],
) -> Vec<Candidate> {
    urls.iter()
        .map(|url| {
            let Some(c) = candidates.iter().find(|c| &c.worker_url == url) else {
                return Candidate {
                    worker_url: url.clone(),
                    local_hit: 0,
                    gpu_hit: None,
                    expected_hit: 0,
                    hint: None,
                };
            };
            let source = c
                .hint
                .as_ref()
                .and_then(|h| h.get("source_worker_id"))
                .and_then(Value::as_str);
            if source.is_some_and(|s| !urls.iter().chain(sources).any(|u| u == s)) {
                Candidate {
                    hint: None,
                    expected_hit: c.local_hit,
                    ..c.clone()
                }
            } else {
                c.clone()
            }
        })
        .collect()
}

/// `--symmetric-pd-trace-session-moves`: the P of the last cost-function decision of
/// each session (by session key), for the `session_move` record.
#[derive(Debug, Default)]
struct SessionMoveBook {
    /// Session key -> (P URL, unix ms of the decision).
    last: HashMap<u64, (String, u64)>,
    decisions: u64,
    p_changes: u64,
}

/// Trace-only extras of one reservation ([`PrefillWorkBook::reserve_locked`]).
#[derive(Debug, Default)]
struct ReserveExtras<'a> {
    /// Written as the `queue` object of the event when the reservation is not
    /// a release of the queue (a bound request of `--global-queue`).
    queue_note: Option<Value>,
    /// The session identity of the request, for the `session_move` record.
    session_label: Option<&'a str>,
}

/// `--global-queue` part of one step ([`PrefillWorkBook::queue_step`]): the
/// depth of each P is the lead rule, depth = floor(C + R x T_react) + 1, so
/// that P has room while its queued work W x R <= lead x R, with lead = C / R
/// + T_react (R of the step, T_react of [`QueueLeadBook`]).
#[derive(Debug, Clone, Copy)]
pub struct QueueAutoStep<'a> {
    /// The engine chunk C of each P (pool order) and its source
    /// (`server_info` or `default`).
    pub chunks: &'a [(u64, &'static str)],
}

/// The lead rule of one P in one `--global-queue` step.
#[derive(Debug, Clone, Copy)]
struct QueueLeadRow {
    chunk: u64,
    chunk_source: &'static str,
    react_s: f64,
    react_source: &'static str,
    react_samples: u64,
    /// lead = C / R + T_react (s).
    lead_s: f64,
    /// floor(C + R x T_react) + 1: P has room while queued <= R x lead,
    /// that is W = queued / R <= lead.
    depth: u64,
}

impl QueueLeadRow {
    fn new(
        chunk: u64,
        chunk_source: &'static str,
        react_s: f64,
        react_source: &'static str,
        react_samples: u64,
        rate: f64,
    ) -> Self {
        let react_s = if react_s.is_finite() {
            react_s.max(0.0)
        } else {
            0.0
        };
        let extra = rate * react_s;
        let extra = if extra.is_finite() && extra > 0.0 {
            extra
        } else {
            0.0
        };
        let depth_tokens = chunk as f64 + extra;
        let depth = if depth_tokens < u64::MAX as f64 {
            (depth_tokens.floor() as u64).saturating_add(1)
        } else {
            u64::MAX
        };
        let lead_s = if rate > 0.0 {
            chunk as f64 / rate + react_s
        } else {
            react_s
        };
        Self {
            chunk,
            chunk_source,
            react_s,
            react_source,
            react_samples,
            lead_s,
            depth,
        }
    }
}

pub struct PrefillWorkBook {
    started: Instant,
    /// Written as `policy` on every trace event.
    trace_policy: &'static str,
    /// The unknown expiry (see the module docs); None = never.
    unknown_expire: Option<std::time::Duration>,
    state: Mutex<State>,
    /// Called outside the lock after each terminal status change (the
    /// global waiting queue wakes its release pass). Set at most once.
    on_terminal: std::sync::OnceLock<Box<dyn Fn() + Send + Sync>>,
}

impl std::fmt::Debug for PrefillWorkBook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefillWorkBook").finish_non_exhaustive()
    }
}

struct Evaluation {
    candidate: Candidate,
    cost: u128,
    before: u128,
    after: u128,
    spread: u128,
    active_requests: u64,
    pending_uncached_before: u128,
    pending_uncached_after: u128,
}

impl Evaluation {
    fn value(&self) -> Value {
        json!({
            "worker_url": self.candidate.worker_url,
            "local_hit": self.candidate.local_hit,
            "gpu_hit": self.candidate.gpu_hit,
            "expected_hit": self.candidate.expected_hit,
            "hint": self.candidate.hint,
            "request_cost": self.cost.to_string(),
            "worker_cost_before": self.before.to_string(),
            "worker_cost_after": self.after.to_string(),
            "post_assignment_spread": self.spread.to_string(),
            "active_requests_before": self.active_requests,
            "pending_uncached_before": self.pending_uncached_before.to_string(),
            "pending_uncached_after": self.pending_uncached_after.to_string(),
        })
    }
}

/// The copy price and rate `auto` parts of one cost-function decision: the resolved
/// parameters, how they were resolved, the measurements, and per P (URL, its
/// in-flight requests, its estimates).
type AutoPriceParts = (
    crate::scheduler::placement_cost::PlacementCostParams,
    crate::scheduler::placement_cost::AutoPriceResolution,
    crate::scheduler::placement_cost::AutoPriceInputs,
    Vec<(
        String,
        Option<u64>,
        Option<crate::scheduler::placement_cost::AutoPriceWorkerView>,
    )>,
);

/// The request-independent part of one cost-function decision
/// ([`PrefillWorkBook::cost_ctx`]).
struct CostDecisionCtx {
    decided_at: Instant,
    /// The parameters of this decision (copy price and rate resolved).
    params: crate::scheduler::placement_cost::PlacementCostParams,
    /// Copy price or rate `auto` only.
    auto: Option<AutoPriceParts>,
}

/// One P of a global-waiting-queue release, for the candidate rows.
#[derive(Debug, Clone, Copy)]
struct QueueRowNote {
    /// The held work ahead of the released request on this P.
    virtual_tokens: u128,
    depth: u64,
    room: bool,
    fresh: bool,
    /// `room = forecast` with a usable forecast: the P's backlog term is
    /// R x drain (tokens) instead of the work-book backlog.
    backlog_override: Option<u128>,
    /// `room = forecast`: the P's predicted drain (s), when usable.
    drain_s: Option<f64>,
}

/// A release of the global waiting queue ([`PrefillWorkBook::queue_step`]).
struct QueueReserve {
    ctx: CostDecisionCtx,
    rows: BTreeMap<String, QueueRowNote>,
    /// The `queue` object of the reserve event.
    record: Value,
    /// `--global-queue`: exact cost-function ties are broken as the walk breaks
    /// them ([`crate::scheduler::placement_cost::break_cost_tie`]).
    tie_break: bool,
}

/// One held request of the global waiting queue, as [`PrefillWorkBook::queue_step`]
/// sees it.
#[derive(Debug, Clone, Copy)]
pub struct QueueHeld<'a> {
    pub input_tokens: u64,
    pub rid: Option<&'a str>,
    /// Its cost-function candidates, one per P of the step (pool order).
    pub candidates: &'a [Candidate],
    /// When it entered the queue.
    pub arrived: Instant,
    pub arrived_unix_ms: u64,
    /// Its session key (copy price `auto`).
    pub session: Option<u64>,
    /// Its session identity (`--symmetric-pd-trace-session-moves`); None
    /// otherwise.
    pub session_label: Option<&'a str>,
    /// Its plain cost-function choice at the first step that saw it (None before).
    pub p_arrival: Option<&'a str>,
    /// The step belongs to the pass that the arrival of this request
    /// started: a release now is recorded with `held = false`.
    pub arrival_pass: bool,
}

/// The pool and the settings of one global-waiting-queue step.
#[derive(Debug, Clone, Copy)]
pub struct QueueStepArgs<'a> {
    /// The P workers (pool order).
    pub urls: &'a [String],
    /// The depth of each P in tokens (same order; with `auto` the lead rule
    /// replaces each value, the slice keeps the pool length).
    pub depth: &'a [u64],
    /// A progress view older than this is stale (fail-open).
    pub stale_after: std::time::Duration,
    /// The starvation guard (`sjf`) or safety net (`hrrn`).
    pub max_wait: std::time::Duration,
    /// One engine pass in tokens: the running pass that the queued work
    /// leaves out ([`crate::global_queue::queued_tokens`]).
    pub pass_tokens: u64,
    /// `--global-queue` (None: the depth above and `pass_tokens`): the
    /// lead rule with the engine chunk of each P.
    pub auto: Option<QueueAutoStep<'a>>,
    pub now: Instant,
    pub pass_id: u64,
    pub trigger: &'a str,
    pub triggers: &'a str,
    /// `--global-queue-room forecast`: the forecast inputs of a cold
    /// one-chunk probe per P (built before the lock, as for a the forecast wait
    /// decision). None: the token rule for every P.
    pub forecast: Option<&'a crate::prefill_forecast::ForecastContext>,
    /// `room = forecast`: the lead (s) that a P's predicted drain must not
    /// exceed for room.
    pub lead_s: f64,
    /// `--symmetric-pd-pull-from-any-holder`: the registered workers of any role
    /// at this step (breaker allowing when the breaker is on); a held request
    /// keeps a pull hint whose source is one of them (or in the pool). Empty:
    /// only pool sources are kept.
    pub sources: &'a [String],
}

/// The input tokens of the cold probe of `--global-queue-room
/// forecast`: one engine pass, no cached prefix. Its forecast `wait_s` is the
/// time until the P can start a new pass for a released request.
pub const QUEUE_PROBE_TOKENS: u64 = crate::global_queue::CHUNK_TOKENS;

/// The cold probe candidate of one P ([`global queue_PROBE_TOKENS`]).
pub fn queue_probe_candidate(url: &str) -> Candidate {
    Candidate {
        worker_url: url.to_string(),
        local_hit: 0,
        gpu_hit: None,
        expected_hit: 0,
        hint: None,
    }
}

/// The drain of one P from its probe: the forecast's `wait_s` when the
/// forecast exists (a missing error bound does not matter here: the drain
/// is no move decision). Else None (the token rule) and the reason.
fn queue_probe_drain(probe: &ForecastProbe) -> (Option<f64>, Option<&'static str>) {
    let wait = probe.out.as_ref().and_then(|o| o.wait_s);
    match (probe.fallback, wait) {
        (None | Some("no_bound"), Some(w)) if w.is_finite() => (Some(w.max(0.0)), None),
        (Some(reason), _) => (None, Some(reason)),
        (None, _) => (None, Some("no_forecast")),
    }
}

/// R x `drain_s` in tokens (rounded; 0 for a negative or non-finite value).
fn queue_drain_tokens(drain_s: f64, prefill_rate_tokens_per_s: f64) -> u128 {
    let tokens = (drain_s * prefill_rate_tokens_per_s).round();
    if tokens.is_finite() && tokens > 0.0 {
        tokens as u128
    } else {
        0
    }
}

/// What one [`PrefillWorkBook::queue_step`] did.
#[derive(Debug)]
pub struct QueueStep {
    /// The plain cost-function choice of each held request (index into the URLs).
    pub plain: Vec<usize>,
    /// The uncached tokens of the held requests on their plain choice.
    pub queue_tokens: u128,
    /// None: nothing released. Else the held request (index into `held`)
    /// that left the queue: reserved (`Ok`), or refused by the book (`Err`;
    /// for example a reused RID).
    pub release: Option<(usize, Result<QueueReleased, String>)>,
}

/// A reserved global-waiting-queue release.
#[derive(Debug)]
pub struct QueueReleased {
    pub candidate: Candidate,
    pub reservation: Reservation,
    /// The `placement_decision` of the reserve event.
    pub decision: Value,
    /// The `queue` object of the reserve event.
    pub record: Value,
    pub release: crate::global_queue::QueueRelease,
    pub starved: bool,
    /// The P at release differs from the plain choice at arrival.
    pub late_bound: bool,
}

/// One P of the `queue_stats` event ([`PrefillWorkBook::queue_rows`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueRowStat {
    /// The queued work ([`crate::global_queue::queued_tokens`]).
    pub queued: u128,
    /// Its parts: the not-yet-launched uncached tokens (without the unknown
    /// reservations) and the remaining work of the launched requests.
    pub unlaunched: u128,
    pub launched: u128,
    /// Milliseconds since the poll of the last applied progress snapshot.
    pub view_age_ms: Option<u64>,
}

impl PrefillWorkBook {
    /// A new work book. `trace_path`: a new JSONL trace (an existing path is
    /// never appended or truncated); every trace event carries
    /// `"policy": policy`. `unknown_expire`: an unknown entry expires after
    /// this long without engine progress (see the module docs); None keeps it
    /// until a terminal result.
    pub fn new(
        trace_path: Option<&Path>,
        policy: &'static str,
        unknown_expire: Option<std::time::Duration>,
    ) -> io::Result<Arc<Self>> {
        let trace = trace_path
            .map(|path| OpenOptions::new().write(true).create_new(true).open(path))
            .transpose()?
            .map(BufWriter::new);
        Ok(Arc::new(Self {
            started: Instant::now(),
            trace_policy: policy,
            unknown_expire,
            on_terminal: std::sync::OnceLock::new(),
            state: Mutex::new(State {
                next_id: 1,
                event_sequence: 0,
                entries: BTreeMap::new(),
                active_by_rid: BTreeMap::new(),
                used_rids: BTreeSet::new(),
                progress: BTreeMap::new(),
                wait_fallbacks: BTreeMap::new(),
                auto_price: None,
                unknown: BTreeSet::new(),
                prefill_reader_gone: 0,
                expired_by_rid: BTreeMap::new(),
                expired_listed_again: 0,
                queue_lead: None,
                session_moves: None,
                recent_finished: VecDeque::new(),
                forecast_push_sources: BTreeMap::new(),
                forecast_push_fallbacks: BTreeMap::new(),
                workers: BTreeMap::new(),
                healthy_candidates: Vec::new(),
                trace,
                io_failure: None,
                ended: VecDeque::new(),
                retired_counts: BTreeMap::new(),
                retired_entries: 0,
            }),
        }))
    }

    /// cost-function placement on the work book: argmin over candidates of
    /// backlog + own uncached + unhidden pull time + copy price. The caller
    /// keeps a hint only for a pull of at least the pull floor. Returns the
    /// per-candidate decision record that the reserve trace event also holds.
    pub fn reserve_cost(
        self: &Arc<Self>,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        params: crate::scheduler::placement_cost::PlacementCostParams,
    ) -> Result<(Candidate, Reservation, Value), String> {
        self.reserve_cost_session(input_tokens, rid, candidates, params, None)
    }

    /// [`Self::reserve_cost`] with the request's session identity
    /// ([`crate::scheduler::placement_cost::placement_session_key`]). The session
    /// counts in N_active of the copy price `auto`; with both cost-function modes
    /// fixed it is not used.
    pub fn reserve_cost_session(
        self: &Arc<Self>,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        params: crate::scheduler::placement_cost::PlacementCostParams,
        session: Option<&str>,
    ) -> Result<(Candidate, Reservation, Value), String> {
        self.reserve_cost_session_noted(input_tokens, rid, candidates, params, session, None)
    }

    /// [`Self::reserve_cost_session`] with `queue_note`, written as the `queue`
    /// object of the reserve event (`--global-queue`: a request that the
    /// queue does not hold).
    #[allow(clippy::too_many_arguments)]
    pub fn reserve_cost_session_noted(
        self: &Arc<Self>,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        params: crate::scheduler::placement_cost::PlacementCostParams,
        session: Option<&str>,
        queue_note: Option<Value>,
    ) -> Result<(Candidate, Reservation, Value), String> {
        let (candidate, reservation, decision) = self.reserve_placement_with_decision(
            input_tokens,
            rid,
            candidates,
            params,
            session.map(crate::scheduler::placement_cost::placement_session_hash),
            ReserveExtras {
                queue_note,
                session_label: session,
            },
            None,
        )?;
        Ok((
            candidate,
            reservation,
            decision.expect("a placement records its decision"),
        ))
    }

    /// `--global-queue`: keep T_react of each P (EWMA time constant
    /// `tau_s`; `fallback_s` before the first sample of a P) for the lead of
    /// [`Self::queue_step`] with [`QueueAutoStep`].
    pub fn enable_queue_lead(&self, tau_s: f64, fallback_s: f64) {
        let mut state = self.state.lock();
        if state.queue_lead.is_none() {
            state.queue_lead = Some(QueueLeadBook {
                tau_s,
                fallback_s,
                by_p: BTreeMap::new(),
                last_step: BTreeMap::new(),
                marked: 0,
                unmarked: 0,
                samples: 0,
                tie_last: None,
                restarts: 0,
            });
        }
    }

    /// `--global-queue`: `url` joined the step pool again (a role switch
    /// back to prefill, a drain that ended, a breaker that closed): its
    /// T_react restarts from the fallback, and the releases marked for a
    /// sample on it are unmarked (their release time is from before). Returns
    /// the dropped estimate (s), None when it had none.
    pub fn queue_lead_restart(&self, url: &str) -> Option<f64> {
        let mut state = self.state.lock();
        let State {
            queue_lead,
            entries,
            ..
        } = &mut *state;
        let book = queue_lead.as_mut()?;
        let dropped = book.by_p.remove(url).map(|e| e.sum / e.weight);
        book.last_step.remove(url);
        for entry in entries.values_mut().filter(|e| e.worker_url == url) {
            entry.queue_lead_mark = None;
        }
        if dropped.is_some() {
            book.restarts += 1;
        }
        dropped
    }

    /// `--symmetric-pd-trace-session-moves`: every cost-function reserve event with a
    /// session gets the `session_move` record (the P of the session's last
    /// decision, the hits, the pull).
    pub fn enable_session_moves(&self) {
        let mut state = self.state.lock();
        if state.session_moves.is_none() {
            state.session_moves = Some(SessionMoveBook::default());
        }
    }

    /// The client ended `session` (`POST /kv_sessions/end`): it leaves
    /// N_active of the copy price `auto` at once. Returns whether the auto
    /// state counted it (false without auto decisions).
    pub fn auto_price_end_session(&self, session: &str) -> bool {
        let key = crate::scheduler::placement_cost::placement_session_hash(session);
        self.state
            .lock()
            .auto_price
            .as_mut()
            .is_some_and(|auto| auto.session_end(key))
    }

    /// Every trace row emitted so far is written and flushed.
    pub fn sync_trace(&self) -> io::Result<()> {
        match self.state.lock().trace.as_mut() {
            None => Ok(()),
            Some(writer) => writer.flush(),
        }
    }

    /// [`Self::reserve_cost`] with the inputs of `--prefill-wait
    /// forecast` ([`crate::prefill_forecast::ForecastContext`], built before this lock).
    /// With the forecast wait and no context every P falls back to the work book
    /// terms (reason `no_context`) and the owner is kept.
    pub fn reserve_cost_forecast(
        self: &Arc<Self>,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        params: crate::scheduler::placement_cost::PlacementCostParams,
        forecast: Option<&crate::prefill_forecast::ForecastContext>,
    ) -> Result<(Candidate, Reservation, Value), String> {
        self.reserve_cost_forecast_session(input_tokens, rid, candidates, params, forecast, None)
    }

    /// [`Self::reserve_cost_forecast`] with the request's session identity
    /// ([`crate::scheduler::placement_cost::placement_session_key`]) for N_active
    /// of the copy price `auto` (as [`Self::reserve_cost_session`]).
    pub fn reserve_cost_forecast_session(
        self: &Arc<Self>,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        params: crate::scheduler::placement_cost::PlacementCostParams,
        forecast: Option<&crate::prefill_forecast::ForecastContext>,
        session: Option<&str>,
    ) -> Result<(Candidate, Reservation, Value), String> {
        self.reserve_cost_forecast_session_noted(
            input_tokens,
            rid,
            candidates,
            params,
            forecast,
            session,
            None,
        )
    }

    /// [`Self::reserve_cost_forecast_session`] with `queue_note`, written as the
    /// `queue` object of the reserve event (as
    /// [`Self::reserve_cost_session_noted`]; the symmetric_pd cost function with
    /// `--prefill-wait forecast`).
    #[allow(clippy::too_many_arguments)]
    pub fn reserve_cost_forecast_session_noted(
        self: &Arc<Self>,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        params: crate::scheduler::placement_cost::PlacementCostParams,
        forecast: Option<&crate::prefill_forecast::ForecastContext>,
        session: Option<&str>,
        queue_note: Option<Value>,
    ) -> Result<(Candidate, Reservation, Value), String> {
        let (candidate, reservation, decision) = self.reserve_placement_with_decision(
            input_tokens,
            rid,
            candidates,
            params,
            session.map(crate::scheduler::placement_cost::placement_session_hash),
            ReserveExtras {
                queue_note,
                session_label: session,
            },
            forecast,
        )?;
        Ok((
            candidate,
            reservation,
            decision.expect("a placement records its decision"),
        ))
    }

    /// Write one event (for example a the forecast wait shadow or label line) to
    /// the placement trace, with the trace's sequence number and timestamps.
    pub fn emit_event(&self, event: Value) -> Result<(), String> {
        let mut state = self.state.lock();
        self.emit(&mut state, event)
    }

    /// The forecast wait forecast of one candidate, under the lock.
    ///
    /// The snapshot time is t0 = poll receipt - the engine's `age_s`;
    /// the view age (decision - poll send + `age_s`) must not exceed
    /// `max_state_age_s`. With the state push on, the P's newest pushed
    /// state replaces the polled snapshot when [`push_pick`] accepts it:
    /// t0 = its state time, view age = decision - t0, and the
    /// bound comes from the push table. The queue holds the snapshot's
    /// waiting and bootstrap rows and the router rows: entries of this P
    /// that the snapshot does not show ([`crate::prefill_forecast::ForecastView::shown`])
    /// and that are reserved or dispatched, or finished but decided after
    /// t0. An entry that a progress poll received before t0 listed is left
    /// out: it launched before t0, so a snapshot without it means it is
    /// done. The null-prefix rule takes an entry's own estimate.
    fn probe_forecast(
        state: &State,
        forecast: Option<&crate::prefill_forecast::ForecastContext>,
        candidate: &Candidate,
        decided_at: Instant,
        params: &crate::scheduler::placement_cost::PlacementCostParams,
        ids: &[u64],
    ) -> ForecastProbe {
        let mut probe = ForecastProbe::default();
        let Some(ctx) = forecast else {
            probe.fallback = Some("no_context");
            return probe;
        };
        let Some(prep) = ctx.candidates.get(&candidate.worker_url) else {
            probe.fallback = Some("no_context");
            return probe;
        };
        probe.prep = Some(prep.clone());
        let secs = |at: Instant| crate::prefill_forecast::secs_since(ctx.anchor, at);
        let d = secs(decided_at);
        let polled = prep
            .snapshot
            .as_ref()
            .and_then(|s| s.forecast.as_ref().map(|v| (s, v)));
        let pushed = if ctx.cfg.push.is_some() {
            probe.push_on = true;
            probe.push_age_s = prep.push.as_ref().map(|p| d - p.t0_s);
            let poll_epoch = polled.and_then(|(_, v)| v.epoch.as_deref());
            match push_pick(prep.push.as_deref(), poll_epoch, d, params.max_state_age_s) {
                Ok(p) => Some(p),
                Err(reason) => {
                    probe.push_fallback = Some(reason);
                    None
                }
            }
        } else {
            None
        };
        let (view, t0, view_age) = match pushed {
            Some(p) => {
                probe.source = Some("push");
                probe.view_key = json!([p.epoch, p.seq]);
                (&p.view, p.t0_s, d - p.t0_s)
            }
            None => {
                let Some((snapshot, view)) = polled else {
                    probe.fallback = Some("no_view");
                    return probe;
                };
                probe.source = Some("poll");
                probe.view_key = json!([view.epoch, view.sequence]);
                let Some(age) = view.age_s else {
                    probe.fallback = Some(view.unusable.unwrap_or("age_unknown"));
                    return probe;
                };
                let t0 = secs(snapshot.poll_received) - age;
                (view, t0, d - secs(snapshot.poll_sent) + age)
            }
        };
        probe.t0_age_s = Some(d - t0);
        probe.view_age_s = Some(view_age);
        probe.degraded = view.degraded.iter().copied().collect();
        if let Some(reason) = view.unusable {
            probe.fallback = Some(reason);
            return probe;
        }
        if view_age > params.max_state_age_s {
            probe.fallback = Some("view_too_old");
            return probe;
        }
        // The bound tables were fitted at one engine policy.
        if (view.chunk, view.page) != crate::prefill_forecast::BOUND_FIT_POLICY {
            probe.fallback = Some("policy_not_bound_fit");
            return probe;
        }
        let Some(fit_params) = prep.params.as_ref() else {
            probe.fallback = Some("no_fit");
            return probe;
        };
        // The simulated passes take the engine's budget and page.
        let fparams = &crate::forecast::Params {
            chunk_tokens: Some(view.chunk),
            page_tokens: Some(view.page),
            ..fit_params.clone()
        };
        let rows: Vec<crate::prefill_forecast::RouterRow> = ids
            .iter()
            .filter_map(|id| state.entries.get(id))
            .filter(|e| e.worker_url == candidate.worker_url)
            .filter(|e| match e.status {
                Status::Reserved | Status::Dispatched => true,
                Status::Finished => secs(e.decided_at) > t0,
                Status::Unknown | Status::RolledBack | Status::Expired => false,
            })
            // Listed by a poll received before t0: launched before t0.
            .filter(|e| e.ack_at.is_none_or(|at| secs(at) >= t0))
            .filter_map(|e| {
                let rid = e.rid.as_ref().filter(|rid| !view.shown.contains(*rid))?;
                let input = e.input_tokens as i64;
                let (prefix, host) = e.forecast_estimate.unwrap_or_else(|| {
                    let hit =
                        crate::prefill_snapshot::floor_hit(e.expected_hit as i64, input, view.page);
                    (hit, 0)
                });
                Some(crate::prefill_forecast::RouterRow {
                    rid: rid.clone(),
                    input,
                    prefix,
                    host,
                    decided_s: secs(e.decided_at),
                })
            })
            .collect();
        let estimate = |rid: &str| {
            state
                .active_by_rid
                .get(rid)
                .and_then(|id| state.entries.get(id))
                .and_then(|e| e.forecast_estimate)
        };
        let (input, notes) = crate::prefill_forecast::build_input(
            view,
            t0,
            d,
            prep.target.clone(),
            &rows,
            &estimate,
            fparams,
            Some(ctx.cfg.rq_margin_s),
        );
        probe.notes = notes;
        match crate::forecast::forecast(&input, fparams) {
            Ok(out) => {
                probe.bucket = Some(ctx.cfg.bucket_for(probe.pushed(), &out));
                if out.pred_s.is_none() || out.wait_s.is_none() {
                    probe.fallback = Some("no_forecast");
                } else {
                    probe.bound_s = ctx.cfg.bound_for(probe.pushed(), &out);
                    if probe.bound_s.is_none() {
                        probe.fallback = Some("no_bound");
                    }
                }
                probe.out = Some(out);
            }
            Err(error) => {
                probe.fallback = Some("forecast_error");
                probe.error = Some(error.to_string());
            }
        }
        probe
    }

    /// `session`: the request's session key; used only by an auto cost-function
    /// decision (N_active) and the session-move record.
    #[allow(clippy::too_many_arguments)]
    fn reserve_placement_with_decision(
        self: &Arc<Self>,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        params: crate::scheduler::placement_cost::PlacementCostParams,
        session: Option<u64>,
        extras: ReserveExtras<'_>,
        forecast: Option<&crate::prefill_forecast::ForecastContext>,
    ) -> Result<(Candidate, Reservation, Option<Value>), String> {
        let urls = self.validate_placement(input_tokens, &candidates)?;
        let mut state = self.state.lock();
        self.reserve_locked(
            &mut state,
            input_tokens,
            rid,
            candidates,
            urls,
            params,
            session,
            forecast,
            None,
            extras,
        )
    }

    /// The checks of a placement that need no lock. Returns the candidate URLs.
    fn validate_placement(
        &self,
        input_tokens: u64,
        candidates: &[Candidate],
    ) -> Result<BTreeSet<String>, String> {
        if candidates.is_empty() {
            return Err("cost_balance: no healthy candidates".into());
        }
        let mut urls = BTreeSet::new();
        for candidate in candidates {
            if candidate.worker_url.is_empty() || !urls.insert(candidate.worker_url.clone()) {
                return Err("cost_balance: empty or duplicate candidate URL".into());
            }
            if candidate.expected_hit.min(input_tokens) < candidate.local_hit.min(input_tokens) {
                return Err("cost_balance: expected hit is below the candidate's local hit".into());
            }
            if candidate.hint.is_none()
                && candidate.expected_hit.min(input_tokens) > candidate.local_hit.min(input_tokens)
            {
                return Err(
                    "cost_balance: remote expected hit requires an actual pull hint".into(),
                );
            }
        }
        Ok(urls)
    }

    /// Select and reserve under the lock (the body of
    /// [`Self::reserve_placement_with_decision`]). `queue`: a release of the
    /// global waiting queue, with the decision context of its step (same
    /// lock), the virtual backlog per P and the `queue` trace record. A cost-function
    /// placement then adds the virtual backlog to each candidate's backlog.
    #[allow(clippy::too_many_arguments)]
    fn reserve_locked(
        self: &Arc<Self>,
        state: &mut State,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        urls: BTreeSet<String>,
        configured: crate::scheduler::placement_cost::PlacementCostParams,
        session: Option<u64>,
        forecast: Option<&crate::prefill_forecast::ForecastContext>,
        queue: Option<QueueReserve>,
        extras: ReserveExtras<'_>,
    ) -> Result<(Candidate, Reservation, Option<Value>), String> {
        self.check_admission(state)?;
        {
            let rid = rid
                .as_deref()
                .filter(|r| !r.is_empty())
                .ok_or("uncached balance requires a unique string rid")?;
            if state.used_rids.contains(rid) {
                return Err(
                    "uncached balance cannot reuse a rid; stale engine snapshots may contain it"
                        .into(),
                );
            }
        }
        state.healthy_candidates = urls.into_iter().collect();
        for candidate in &candidates {
            state
                .workers
                .entry(candidate.worker_url.clone())
                .or_default();
        }
        let mut evaluations = Vec::with_capacity(candidates.len());
        for mut candidate in candidates {
            candidate.local_hit = candidate.local_hit.min(input_tokens);
            candidate.gpu_hit = candidate.gpu_hit.map(|hit| hit.min(candidate.local_hit));
            candidate.expected_hit = candidate.expected_hit.min(input_tokens);
            let uncached = input_tokens - candidate.expected_hit;
            let cost = u128::from(uncached);
            let worker = &state.workers[&candidate.worker_url];
            let after = worker.cost.checked_add(cost).ok_or_else(|| {
                format!("cost_balance: cost overflow on {}", candidate.worker_url)
            })?;
            let pending_uncached_after = worker
                .pending_uncached_tokens
                .checked_add(u128::from(input_tokens - candidate.expected_hit))
                .ok_or("cost_balance: pending token overflow")?;
            worker.active_requests.checked_add(1).ok_or_else(|| {
                format!(
                    "cost_balance: request count overflow on {}",
                    candidate.worker_url
                )
            })?;
            let mut minimum = u128::MAX;
            let mut maximum = 0;
            for url in &state.healthy_candidates {
                let value = if *url == candidate.worker_url {
                    after
                } else {
                    state.workers[url].cost
                };
                minimum = minimum.min(value);
                maximum = maximum.max(value);
            }
            evaluations.push(Evaluation {
                candidate,
                cost,
                before: worker.cost,
                after,
                spread: maximum - minimum,
                active_requests: worker.active_requests,
                pending_uncached_before: worker.pending_uncached_tokens,
                pending_uncached_after,
            });
        }
        let placement_decision: Option<Value>;
        let mut wait_fallback_counts = None;
        // Copy price or rate auto: the time of this decision; the entry then
        // feeds the auto state and its session starts at the commit.
        let mut auto_price_at: Option<Instant> = None;
        // Global waiting queue release: the `queue` object of the event.
        let queue_record: Option<Value>;
        // --global-queue: the P of a tie-broken release, the next
        // rotation start once the reservation is recorded.
        let mut queue_tie_commit: Option<String> = None;
        // The decision time of a cost-function placement (the entry's decided_at).
        let cost_decided_at: Option<Instant>;
        // forecast wait: the chosen P's (prefix, host) estimate, for the entry.
        let mut forecast_estimate: Option<(i64, i64)> = None;
        // forecast wait with the state push: the new cumulative source and
        // fallback counts, committed with the decision.
        let mut forecast_push_counts: Option<(BTreeMap<String, u64>, BTreeMap<String, u64>)> = None;
        {
            use crate::scheduler::placement_cost::{CopyPriceMode, PrefillWait};
            // The request-independent part of the decision (copy price and
            // rate auto; [`Self::cost_ctx`]). A global queue release brings
            // the context of its step, which ran under this same lock.
            let (ctx, mut queue, tie_break) = match queue {
                Some(QueueReserve {
                    ctx,
                    rows,
                    record,
                    tie_break,
                }) => (ctx, Some((rows, record)), tie_break),
                None => {
                    let urls: Vec<&str> = evaluations
                        .iter()
                        .map(|e| e.candidate.worker_url.as_str())
                        .collect();
                    (
                        Self::cost_ctx(state, configured, &urls, Instant::now()),
                        None,
                        false,
                    )
                }
            };
            let CostDecisionCtx {
                decided_at,
                params,
                auto,
            } = ctx;
            cost_decided_at = Some(decided_at);
            if auto.is_some() {
                auto_price_at = Some(decided_at);
            }
            let forecast_on = params.wait == PrefillWait::Forecast;
            let mut forecast_ids: Vec<u64> = Vec::new();
            if forecast_on {
                // Finished entries older than the window are no router rows.
                while state.recent_finished.front().is_some_and(|(at, _)| {
                    decided_at.saturating_duration_since(*at).as_secs_f64() > RECENT_FINISHED_S
                }) {
                    state.recent_finished.pop_front();
                }
                forecast_ids = state
                    .active_by_rid
                    .values()
                    .copied()
                    .chain(state.recent_finished.iter().map(|(_, id)| *id))
                    .collect();
                forecast_ids.sort_unstable();
                forecast_ids.dedup();
            }
            let forecast_probes: Vec<Option<ForecastProbe>> = evaluations
                .iter()
                .map(|e| {
                    forecast_on.then(|| {
                        Self::probe_forecast(
                            state,
                            forecast,
                            &e.candidate,
                            decided_at,
                            &params,
                            &forecast_ids,
                        )
                    })
                })
                .collect();
            let inputs: Vec<_> = evaluations
                .iter()
                .zip(&forecast_probes)
                .map(|(e, forecast_probe)| {
                    // A global queue release adds the held work ahead of it (0 otherwise).
                    let note = queue
                        .as_ref()
                        .and_then(|(notes, _)| notes.get(&e.candidate.worker_url));
                    let virtual_tokens = note.map_or(0, |note| note.virtual_tokens);
                    // `room = forecast`: the walk priced this P's wait as
                    // R x drain; the reservation uses the same term.
                    crate::scheduler::placement_cost::CostInput {
                        worker: &e.candidate.worker_url,
                        backlog: note
                            .and_then(|note| note.backlog_override)
                            .unwrap_or(e.pending_uncached_before)
                            .saturating_add(virtual_tokens),
                        local_hit: e.candidate.local_hit,
                        uncached: input_tokens - e.candidate.expected_hit,
                        pulled: e.candidate.expected_hit - e.candidate.local_hit,
                        // The terms of the forecast (None for the work book).
                        wait: forecast_probe.as_ref().and_then(|q| q.terms()),
                        bound_s: forecast_probe
                            .as_ref()
                            .and_then(ForecastProbe::usable_bound),
                    }
                })
                .collect();
            let mut selected = crate::scheduler::placement_cost::select_min_cost(&params, &inputs);
            if let (true, Some((notes, record))) = (tie_break, queue.as_mut()) {
                // --global-queue: the walk's tie-break, with the same
                // room, outstanding requests and rotation of this step.
                let room: Vec<bool> = evaluations
                    .iter()
                    .map(|e| notes.get(&e.candidate.worker_url).is_some_and(|n| n.room))
                    .collect();
                let outstanding: Vec<u64> = evaluations.iter().map(|e| e.active_requests).collect();
                let last = state.queue_lead.as_ref().and_then(|b| b.tie_last.clone());
                let outcome = crate::scheduler::placement_cost::break_cost_tie(
                    &selected,
                    &inputs,
                    &crate::scheduler::placement_cost::TieBreak {
                        room: &room,
                        outstanding: &outstanding,
                        last: last.as_deref(),
                    },
                );
                selected.chosen = outcome.chosen;
                record["tie"] = json!({
                    "tied": outcome.tied,
                    "decided_by": outcome.decided_by,
                    "outstanding": outstanding[outcome.chosen],
                    "last": last,
                });
                if outcome.tied >= 2 {
                    queue_tie_commit =
                        Some(evaluations[outcome.chosen].candidate.worker_url.clone());
                }
            }
            if let Some((_, record)) = queue.as_mut() {
                // The step's walk chose this P with the same inputs under the
                // same lock; a difference is a bug, recorded and logged.
                let chosen = evaluations[selected.chosen].candidate.worker_url.as_str();
                if record["p_release"].as_str() != Some(chosen) {
                    tracing::error!(target: "cost_balance", chosen,
                        expected = %record["p_release"],
                        "global waiting queue: the reservation chose another P than its walk");
                    record["walk_mismatch"] = json!(true);
                    record["p_release"] = json!(chosen);
                }
            }
            let rows: Vec<Value> = inputs
                .iter()
                .zip(&selected.costs)
                .zip(&evaluations)
                .zip(&forecast_probes)
                .map(|(((input, cost), e), forecast_probe)| {
                    let mut row = json!({
                        "worker_url": input.worker,
                        "backlog_tokens": input.backlog.to_string(),
                        "pending_uncached_tokens": e.pending_uncached_before.to_string(),
                        "unacknowledged_tokens": state.workers[input.worker]
                            .pending_unacknowledged_tokens
                            .to_string(),
                        "local_hit": input.local_hit,
                        "pulled_tokens": input.pulled,
                        "uncached_tokens": input.uncached,
                        "pull_s": cost.pull_s,
                        "exposed_pull_s": cost.exposed_pull_s,
                        "exposed_pull_tokens": cost.exposed_pull_tokens,
                        "copy_tokens": cost.copy_tokens,
                        "cost": cost.total,
                    });
                    if let Some(q) = forecast_probe {
                        if let (Value::Object(row), Value::Object(extra)) =
                            (&mut row, forecast_row_value(q, cost))
                        {
                            row.extend(extra);
                        }
                    } else if !params.is_original() {
                        let extra = json!({
                            "wait_source": "tokens",
                            "wait_s": cost.wait_s,
                            "service_s": cost.service_s,
                        });
                        if let (Value::Object(row), Value::Object(extra)) = (&mut row, extra) {
                            row.extend(extra);
                        }
                    }
                    if let Some(note) = queue
                        .as_ref()
                        .and_then(|(notes, _)| notes.get(input.worker))
                    {
                        row["queue_virtual_tokens"] = json!(note.virtual_tokens.to_string());
                        row["queue_depth_tokens"] = json!(note.depth);
                        row["queue_room"] = json!(note.room);
                        row["queue_view_fresh"] = json!(note.fresh);
                        if note.backlog_override.is_some() {
                            row["queue_drain_s"] = json!(note.drain_s);
                            row["queue_pending_backlog_tokens"] =
                                json!(e.pending_uncached_before.to_string());
                        }
                    }
                    if let Some((_, _, _, views)) = &auto {
                        let (in_flight, view) = views
                            .iter()
                            .find(|(url, _, _)| url == input.worker)
                            .map_or((None, None), |(_, in_flight, view)| (*in_flight, *view));
                        row["auto_in_flight"] = json!(in_flight);
                        row["auto_busy"] = json!(view.map(|v| v.busy));
                        row["auto_inflight_mean"] = json!(view.map(|v| v.inflight));
                        row["auto_rate_tokens_per_s"] =
                            json!(view.map(|v| v.prefill_rate_tokens_per_s));
                    }
                    row
                })
                .collect();
            let mut decision = json!({
                "owner": evaluations[selected.owner].candidate.worker_url,
                "has_cached_owner": selected.has_owner,
                "chosen": evaluations[selected.chosen].candidate.worker_url,
                "detour": selected.chosen != selected.owner,
                "pulled_tokens": inputs[selected.chosen].pulled,
                "cost": selected.costs[selected.chosen].total,
                "prefill_rate_tokens_per_s": params.prefill_rate_tokens_per_s,
                "fetch_fixed_ms": params.fetch_fixed_ms,
                "fetch_ms_per_100k_tokens": params.fetch_ms_per_100k_tokens,
                "copy_price": params.copy_price,
                "candidates": rows,
            });
            if !params.is_original() {
                let mut counts = state.wait_fallbacks.clone();
                let mut fallbacks = 0u64;
                let reasons = forecast_probes.iter().flatten().filter_map(|q| q.fallback);
                for reason in reasons {
                    *counts.entry(reason.to_string()).or_default() += 1;
                    fallbacks += 1;
                }
                let extra = json!({
                    "wait": params.wait.as_str(),
                    "max_state_age_s": params.max_state_age_s,
                    "best": evaluations[selected.best].candidate.worker_url,
                    "margin_s": selected.margin_s,
                    "owner_gain_s": selected.owner_gain_s,
                    "margin_protected": selected.margin_protected,
                    "wait_s": selected.costs[selected.chosen].wait_s,
                    "service_s": selected.costs[selected.chosen].service_s,
                    "wait_fallbacks": fallbacks,
                    "wait_fallback_counts": counts,
                });
                if let (Value::Object(decision), Value::Object(extra)) = (&mut decision, extra) {
                    decision.extend(extra);
                }
                wait_fallback_counts = Some(counts);
            }
            // Copy price or rate auto: the measurements and how lambda and R
            // were set. `copy_price` and `prefill_rate_tokens_per_s` above are the
            // values this decision used.
            if let Some((_, resolution, measured, _)) = &auto {
                let (lambda_min, lambda_max) = match configured.copy_price_mode {
                    CopyPriceMode::Auto { min, max } => (Some(min), Some(max)),
                    CopyPriceMode::Fixed => (None, None),
                };
                let extra = json!({
                    "copy_price_mode": resolution.copy_price_mode,
                    "copy_price_used": resolution.copy_price_used,
                    "copy_price_fallback": configured.copy_price,
                    "copy_price_raw": resolution.copy_price_raw,
                    "copy_price_min": lambda_min,
                    "copy_price_max": lambda_max,
                    "q": resolution.q,
                    "busy": measured.busy,
                    "inflight_mean": measured.inflight,
                    "n_active": measured.n_active,
                    "n_p": measured.n_p,
                    "r_hat": measured.r_hat,
                    "rate_mode": resolution.prefill_rate_mode,
                    "rate_used": resolution.rate_used,
                    "rate_fallback": configured.prefill_rate_tokens_per_s,
                    "rate_fallback_reason": resolution.rate_fallback_reason,
                    "fallback_reason": resolution.fallback_reason,
                    "auto_elapsed_s": measured.elapsed_s,
                    "auto_tau_s": configured.auto_tau_s,
                    "session_idle_s": configured.auto_session_idle_s,
                    "session_known": session.is_some(),
                });
                if let (Value::Object(decision), Value::Object(extra)) = (&mut decision, extra) {
                    decision.extend(extra);
                }
            }
            if forecast_on {
                let chosen = forecast_probes[selected.chosen].as_ref();
                let forecasts = forecast_probes
                    .iter()
                    .flatten()
                    .filter(|q| q.fallback.is_none())
                    .count();
                let extra = forecast_decision_value(
                    &params,
                    &selected,
                    chosen,
                    forecasts,
                    forecast,
                    decided_at,
                    evaluations.iter().zip(&forecast_probes).map(|(e, q)| {
                        (
                            e.candidate.worker_url.as_str(),
                            q.as_ref()
                                .and_then(|q| q.prep.as_ref())
                                .and_then(|p| p.fit.as_ref())
                                .map(|f| f.step),
                        )
                    }),
                );
                if let (Value::Object(decision), Value::Object(extra)) = (&mut decision, extra) {
                    decision.extend(extra);
                }
                if let Some(push) = forecast.and_then(|ctx| ctx.cfg.push.as_ref()) {
                    // State push on: the push settings and, cumulative over
                    // decisions, the candidates per state source and per
                    // reason the pushed state was not used.
                    let mut sources = state.forecast_push_sources.clone();
                    let mut fallbacks = state.forecast_push_fallbacks.clone();
                    for q in forecast_probes.iter().flatten() {
                        *sources
                            .entry(q.source.unwrap_or("none").to_string())
                            .or_default() += 1;
                        if let Some(reason) = q.push_fallback {
                            *fallbacks.entry(reason.to_string()).or_default() += 1;
                        }
                    }
                    decision["forecast_push_port"] = json!(push.port);
                    decision["forecast_push_bound_mode"] = json!(push.mode);
                    decision["forecast_push_bound_level"] = json!(push.level);
                    decision["forecast_state_source_counts"] = json!(sources);
                    decision["forecast_push_fallback_counts"] = json!(fallbacks);
                    decision["forecast_chosen"]["state_source"] =
                        json!(chosen.and_then(|q| q.source));
                    forecast_push_counts = Some((sources, fallbacks));
                }
                forecast_estimate = chosen
                    .and_then(|q| q.prep.as_ref())
                    .map(|p| (p.target.prefix, p.target.host));
            }
            drop(inputs);
            evaluations.swap(0, selected.chosen);
            placement_decision = Some(decision);
            queue_record = queue.map(|(_, record)| record);
        }
        // --symmetric-pd-trace-session-moves: the P of the session's last cost-function
        // decision and the hits of this one (the chosen P is first now).
        let session_move = match (&placement_decision, session, state.session_moves.as_ref()) {
            (Some(decision), Some(key), Some(book)) => {
                let chosen = &evaluations[0].candidate;
                let (max_hit, max_p) = evaluations.iter().fold((0u64, None), |(best, p), e| {
                    if e.candidate.local_hit > best {
                        (e.candidate.local_hit, Some(e.candidate.worker_url.as_str()))
                    } else {
                        (best, p)
                    }
                });
                let prev = book.last.get(&key);
                let prev_hit = prev.and_then(|(url, _)| {
                    evaluations
                        .iter()
                        .find(|e| &e.candidate.worker_url == url)
                        .map(|e| e.candidate.local_hit)
                });
                let p_changed = prev.is_some_and(|(url, _)| *url != chosen.worker_url);
                Some((
                    p_changed,
                    json!({
                        "session": extras.session_label,
                        "session_key": key.to_string(),
                        "prev_p": prev.map(|(url, _)| url),
                        "prev_unix_ms": prev.map(|(_, ms)| ms),
                        "p_changed": p_changed,
                        "chosen": chosen.worker_url,
                        "chosen_local_hit": chosen.local_hit,
                        "max_local_hit": max_hit,
                        "max_local_hit_p": max_p,
                        "prev_p_local_hit": prev_hit,
                        "pulled_tokens": decision["pulled_tokens"],
                        "input_tokens": input_tokens,
                    }),
                ))
            }
            _ => None,
        };
        let candidate_scores: Vec<Value> = evaluations.iter().map(Evaluation::value).collect();
        let winner = evaluations.remove(0);
        let id = state.next_id;
        let next_id = id
            .checked_add(1)
            .ok_or("cost_balance: reservation ID overflow")?;
        let entry = Entry {
            id,
            rid,
            worker_url: winner.candidate.worker_url.clone(),
            input_tokens,
            local_hit: winner.candidate.local_hit,
            gpu_hit: winner.candidate.gpu_hit,
            expected_hit: winner.candidate.expected_hit,
            uncached_tokens: input_tokens - winner.candidate.expected_hit,
            seen_engine_progress: false,
            acknowledged: false,
            cost: winner.cost,
            hint: winner.candidate.hint.clone(),
            status: Status::Reserved,
            assigned_unix_ms: unix_ms(),
            assigned_elapsed_ms: self.started.elapsed().as_millis() as u64,
            dispatched_unix_ms: None,
            terminal_unix_ms: None,
            reason: None,
            auto: AutoTrack {
                tracked: auto_price_at.is_some(),
                session: session.filter(|_| auto_price_at.is_some()),
                in_flight: false,
                done: false,
                peak_tokens: input_tokens - winner.candidate.expected_hit,
            },
            unknown_since: None,
            last_progress_seen: None,
            queue_lead_mark: None,
            rid_released: false,
            decided_at: cost_decided_at.unwrap_or_else(Instant::now),
            ack_at: None,
            forecast_estimate,
        };
        // Write before committing: an IO error must not hand the caller a
        // dispatchable reservation that was never recorded. The same lock
        // spans scoring, writing and committing, preventing stale decisions.
        let mut event = json!({
            "event": "reserve",
            "entry": entry.value(),
            "healthy_candidates": state.healthy_candidates,
            "candidate_scores": candidate_scores,
            "selection_policy": "cost_function",
        });
        if let Some(decision) = &placement_decision {
            event["placement"] = decision.clone();
        }
        if let Some(record) = queue_record {
            event["global_queue"] = record;
        } else if let Some(note) = extras.queue_note {
            event["global_queue"] = note;
        }
        let p_changed = session_move.as_ref().map(|(changed, _)| *changed);
        if let Some((_, record)) = session_move {
            event["session_move"] = record;
        }
        self.emit(state, event)?;
        if let Some(counts) = wait_fallback_counts {
            state.wait_fallbacks = counts;
        }
        if let (Some(url), Some(book)) = (queue_tie_commit, state.queue_lead.as_mut()) {
            book.tie_last = Some(url);
        }
        if let Some((sources, fallbacks)) = forecast_push_counts {
            state.forecast_push_sources = sources;
            state.forecast_push_fallbacks = fallbacks;
        }
        let worker = state
            .workers
            .get_mut(&entry.worker_url)
            .expect("candidate registered");
        worker.cost = winner.after;
        worker.pending_uncached_tokens = winner.pending_uncached_after;
        worker.pending_unacknowledged_tokens += u128::from(entry.uncached_tokens);
        worker.active_requests += 1;
        state.next_id = next_id;
        if let (Some(at), Some(key)) = (auto_price_at, entry.auto.session) {
            if let Some(auto) = state.auto_price.as_mut() {
                auto.session_started(key, at);
            }
        }
        let rid = entry.rid.as_ref().expect("validated rid");
        state.active_by_rid.insert(rid.clone(), id);
        state.used_rids.insert(rid.clone());
        if let (Some(changed), Some(key), Some(book)) =
            (p_changed, session, state.session_moves.as_mut())
        {
            book.decisions += 1;
            book.p_changes += u64::from(changed);
            book.last
                .insert(key, (entry.worker_url.clone(), entry.assigned_unix_ms));
        }
        state.entries.insert(id, entry);
        Ok((
            winner.candidate,
            Reservation {
                book: Arc::clone(self),
                id,
            },
            placement_decision,
        ))
    }

    fn check_admission(&self, state: &State) -> Result<(), String> {
        if let Some(reason) = &state.io_failure {
            return Err(format!(
                "cost_balance: trace failed; admission stopped: {reason}"
            ));
        }
        Ok(())
    }

    fn emit(&self, state: &mut State, mut event: Value) -> Result<(), String> {
        if let Some(reason) = &state.io_failure {
            return Err(reason.clone());
        }
        let sequence = state
            .event_sequence
            .checked_add(1)
            .ok_or_else(|| "cost_balance: event sequence overflow".to_string())?;
        event["sequence"] = json!(sequence);
        let timestamp = unix_ms();
        event["schema"] = json!("placement-trace.v1");
        event["policy"] = json!(self.trace_policy);
        event["ts_unix_ms"] = json!(timestamp);
        event["ts_unix_s"] = json!(timestamp as f64 / 1000.0);
        event["elapsed_ms"] = json!(self.started.elapsed().as_millis() as u64);
        event["cost_unit"] = json!("uncached_tokens");
        let result = match &mut state.trace {
            Some(trace) => serde_json::to_writer(&mut *trace, &event)
                .map_err(io::Error::other)
                .and_then(|()| trace.write_all(b"\n"))
                .and_then(|()| trace.flush()),
            None => Ok(()),
        };
        if let Err(error) = result {
            let reason = error.to_string();
            state.io_failure = Some(reason.clone());
            tracing::error!(target: "cost_balance", error = %reason,
                "placement trace failed; further admission is disabled");
            return Err(reason);
        }
        state.event_sequence = sequence;
        Ok(())
    }

    fn change_status(&self, id: u64, target: Status, reason: &str) -> Result<(), String> {
        let mut state = self.state.lock();
        let result = self.change_status_locked(&mut state, id, target, reason, None);
        drop(state);
        // The work of the entry left its P: the global waiting queue (if
        // any) runs a release pass, outside the lock. An unknown expiry
        // runs under the progress poll, whose applied reply wakes the queue.
        if !target.active() {
            if let Some(hook) = self.on_terminal.get() {
                hook();
            }
        }
        result
    }

    /// [`Self::change_status`] under the caller's lock. `detail` (the
    /// unknown expiry only) is written as `expiry` on the trace event.
    fn change_status_locked(
        &self,
        state: &mut State,
        id: u64,
        target: Status,
        reason: &str,
        detail: Option<Value>,
    ) -> Result<(), String> {
        let Some(old) = state.entries.get(&id).map(|e| e.status) else {
            // A retired entry had ended: nothing changes.
            return if id < state.next_id && target != Status::Dispatched {
                Ok(())
            } else if id < state.next_id {
                Err("cost_balance: only a reserved request can be dispatched".into())
            } else {
                Err("cost_balance: unknown reservation ID".into())
            };
        };
        if target == Status::Dispatched && !matches!(old, Status::Reserved | Status::Dispatched) {
            return Err("cost_balance: only a reserved request can be dispatched".into());
        }
        if !old.active() || old == target {
            return Ok(());
        }
        if !target.active() && reason == crate::proxy::READER_GONE_FINISH_REASON {
            state.prefill_reader_gone += 1;
        }
        let mut value = state.entries[&id].value();
        let now = unix_ms();
        value["status"] = json!(target.label());
        value["reason"] = json!(reason);
        if target == Status::Dispatched {
            value["dispatched_unix_ms"] = json!(now);
        }
        if !target.active() {
            value["terminal_unix_ms"] = json!(now);
        }
        let mut event = json!({"event": target.label(), "entry": value});
        if let Some(detail) = detail {
            event["expiry"] = detail;
        }
        let trace_result = self.emit(state, event);
        // A failed dispatch record means the caller must not send the request.
        // Leave it reserved so its unique owner's Drop safely rolls it back.
        if target == Status::Dispatched && trace_result.is_err() {
            return trace_result;
        }
        let entry = state.entries.get_mut(&id).expect("entry checked");
        entry.status = target;
        entry.reason = Some(reason.to_string());
        if target == Status::Dispatched {
            entry.dispatched_unix_ms = Some(now);
        }
        if target == Status::Unknown {
            entry.unknown_since = Some(Instant::now());
        }
        if !target.active() {
            entry.terminal_unix_ms = Some(now);
            let (url, cost, uncached, rid, acknowledged) = (
                entry.worker_url.clone(),
                entry.cost,
                entry.uncached_tokens,
                entry.rid.clone(),
                entry.acknowledged,
            );
            let rid = rid.as_ref().expect("validated rid");
            state.active_by_rid.remove(rid);
            // Never sent to an engine, so no engine snapshot can list it:
            // a replan (e.g. the chosen worker began a role drain before
            // the prepared plan was installed) may reuse the same rid.
            if target == Status::RolledBack && old == Status::Reserved {
                state.used_rids.remove(rid);
                if let Some(entry) = state.entries.get_mut(&id) {
                    entry.rid_released = true;
                }
            }
            let worker = state.workers.get_mut(&url).expect("entry worker exists");
            worker.cost = worker
                .cost
                .checked_sub(cost)
                .expect("cost counter underflow");
            worker.pending_uncached_tokens = worker
                .pending_uncached_tokens
                .checked_sub(u128::from(uncached))
                .expect("pending token counter underflow");
            if !acknowledged {
                worker.pending_unacknowledged_tokens = worker
                    .pending_unacknowledged_tokens
                    .checked_sub(u128::from(uncached))
                    .expect("unacknowledged counter underflow");
            }
            worker.active_requests = worker
                .active_requests
                .checked_sub(1)
                .expect("request counter underflow");
            if target == Status::Finished {
                // forecast wait router rows (bounded: older ones are dropped).
                let at = Instant::now();
                while state.recent_finished.front().is_some_and(|(t, _)| {
                    at.saturating_duration_since(*t).as_secs_f64() > RECENT_FINISHED_S
                }) {
                    state.recent_finished.pop_front();
                }
                state.recent_finished.push_back((at, id));
            }
            let now = Instant::now();
            state.ended.push_back((now, id));
            Self::retire_ended(state, now);
        }
        if target == Status::Unknown {
            state.unknown.insert(id);
        } else if old == Status::Unknown {
            state.unknown.remove(&id);
        }
        // cost-function auto: the P's in-flight requests and the session. An emit
        // error here is recorded like any trace error (admission stops).
        if let Some((url, period)) = auto_status(state, id, target, reason, Instant::now()) {
            let _ = self.emit(state, p_empty_event(&url, period));
        }
        trace_result
    }

    /// The unknown expiry (see the module docs): an unknown entry, of
    /// `worker_url` only when given, expires when the last applied progress
    /// reply of its P was sent after its last engine evidence and at least
    /// the expiry after it (that reply did not list the RID, or the evidence
    /// would be the reply). A P without applied replies keeps its unknown
    /// entries. The move to `expired` goes through
    /// [`Self::change_status_locked`], which releases the work once. `now`
    /// only dates the trace detail. Nothing to do without an expiry or
    /// without unknown entries.
    fn expire_unknown(&self, state: &mut State, worker_url: Option<&str>, now: Instant) {
        let Some(expire) = self.unknown_expire else {
            return;
        };
        if state.unknown.is_empty() {
            return;
        }
        let due: Vec<(u64, Value)> = state
            .unknown
            .iter()
            .filter_map(|id| {
                let entry = &state.entries[id];
                if worker_url.is_some_and(|url| url != entry.worker_url) {
                    return None;
                }
                let since = entry.unknown_since?;
                let evidence = entry.last_progress_seen.map_or(since, |s| s.max(since));
                // The last applied reply of its P; none: no engine evidence.
                let reply = state.progress.get(&entry.worker_url)?.sent_at;
                let due_at = evidence.checked_add(expire)?;
                (reply > evidence && reply >= due_at).then(|| {
                    let age = |at: Instant| reply.saturating_duration_since(at).as_secs_f64();
                    let detail = json!({
                        "expire_s": expire.as_secs_f64(),
                        "quiet_s": age(evidence),
                        "unknown_for_s": age(since),
                        "since_last_progress_s": entry.last_progress_seen.map(age),
                        "reply_age_s": now.saturating_duration_since(reply).as_secs_f64(),
                    });
                    (*id, detail)
                })
            })
            .collect();
        for (id, detail) in due {
            let entry = &state.entries[&id];
            let rid = entry.rid.clone();
            tracing::warn!(target: "cost_balance", reservation_id = id,
                rid = entry.rid.as_deref().unwrap_or("-"), worker = %entry.worker_url,
                uncached_tokens = entry.uncached_tokens, acknowledged = entry.acknowledged,
                quiet_s = detail["quiet_s"].as_f64().unwrap_or(0.0),
                "unknown reservation expired without engine progress; its work leaves the work book");
            let result = self.change_status_locked(
                state,
                id,
                Status::Expired,
                UNKNOWN_EXPIRED_REASON,
                Some(detail),
            );
            if let Err(error) = result {
                tracing::error!(target: "cost_balance", reservation_id = id, %error,
                    "unknown reservation expired and its work released; the trace write failed");
            }
            if let Some(rid) = rid {
                let worker = state.entries[&id].worker_url.clone();
                state.expired_by_rid.insert(rid, (id, 0, worker));
            }
        }
    }

    /// Replace only observed, dispatched RIDs; never subtract an aggregate or
    /// infer completion from an omitted RID (snapshots can rotate coverage).
    /// Each listed entry records `sent_at` as its last engine evidence; then
    /// the unknown expiry of this worker runs (see the module docs).
    /// [`Self::apply_progress_received`] with the receipt time = now.
    pub fn apply_progress(
        &self,
        worker_url: &str,
        progress: PrefillProgress,
        engine_load: Value,
        sent_at: Instant,
    ) -> Result<(), String> {
        self.apply_progress_received(worker_url, progress, engine_load, sent_at, Instant::now())
    }

    /// [`Self::apply_progress`] for a reply received at `received_at`: an
    /// entry that the reply lists for the first time records that time
    /// (the forecast wait leaves it out of the router rows of a snapshot taken
    /// after it).
    pub fn apply_progress_received(
        &self,
        worker_url: &str,
        progress: PrefillProgress,
        engine_load: Value,
        sent_at: Instant,
        received_at: Instant,
    ) -> Result<(), String> {
        if progress.epoch.is_empty() {
            return Err("prefill progress requires an engine epoch".into());
        }
        let mut state = self.state.lock();
        if let Some(previous) = state.progress.get(worker_url) {
            if sent_at <= previous.sent_at
                || (previous.epoch == progress.epoch && progress.sequence <= previous.sequence)
            {
                return Ok(());
            }
        }
        let mut updates = Vec::new();
        // (id, tokens it holds in pending_unacknowledged_tokens) of first sightings.
        let mut acknowledgements = Vec::new();
        // cost-function auto: (id, remaining) of the tracked entries whose prefill has not ended.
        let mut auto_seen = Vec::new();
        // Every listed active entry (unknown ones too): the unknown expiry evidence.
        let mut listed = Vec::new();
        // (rid, id, remaining) of expired entries of this P listed again.
        let mut relisted = Vec::new();
        for (rid, remaining) in &progress.remaining {
            let Some(id) = state.active_by_rid.get(rid) else {
                if let Some((id, _, worker)) = state.expired_by_rid.get(rid) {
                    if worker == worker_url {
                        relisted.push((rid.clone(), *id, *remaining));
                    }
                }
                continue;
            };
            let entry = &state.entries[id];
            if entry.worker_url != worker_url || entry.status == Status::Reserved {
                continue;
            }
            if *remaining > entry.input_tokens {
                return Err(format!("prefill progress for {rid} exceeds input length"));
            }
            listed.push(*id);
            if !entry.acknowledged {
                acknowledgements.push((*id, entry.uncached_tokens));
            }
            if entry.auto.tracked && !entry.auto.done {
                auto_seen.push((*id, *remaining));
            }
            // The first observation may correct an underestimated cache hit,
            // even before the first chunk finishes.
            if *remaining != entry.uncached_tokens {
                updates.push((
                    *id,
                    entry.uncached_tokens,
                    *remaining,
                    entry.cost,
                    u128::from(*remaining),
                ));
            }
        }
        // Validate the entire update before emitting or mutating the work book.
        let mut next_cost = state.workers.get(worker_url).map_or(0, |w| w.cost);
        for (_, _, _, old_cost, cost) in &updates {
            next_cost = next_cost
                .checked_sub(*old_cost)
                .and_then(|v| v.checked_add(*cost))
                .ok_or("prefill progress cost overflow or underflow")?;
        }
        self.emit(
            &mut state,
            json!({
                "event": "prefill_progress",
                "worker_url": worker_url,
                "engine_epoch": progress.epoch,
                "engine_sequence": progress.sequence,
                "snapshot_truncated": progress.truncated,
                "observed_rids": progress.remaining.len(),
                "updates": updates.iter().map(|(id, old, new, old_cost, cost)| json!({
                    "reservation_id": id, "remaining_before": old, "remaining_after": new,
                    "cost_before": old_cost.to_string(), "cost_after": cost.to_string(),
                })).collect::<Vec<_>>(),
                "engine_load": engine_load,
            }),
        )?;
        // The unknown expiry error bound: the first listing of an expired RID.
        for (rid, id, remaining) in relisted {
            let Some(seen) = state.expired_by_rid.get_mut(&rid) else {
                continue;
            };
            seen.1 += 1;
            if seen.1 > 1 {
                continue;
            }
            state.expired_listed_again += 1;
            tracing::warn!(target: "cost_balance", reservation_id = id, rid = %rid,
                worker = %worker_url, remaining_tokens = remaining,
                "an expired reservation is listed again by its P; the expiry released live work");
            let event = json!({
                "event": "expired_rid_listed_again",
                "worker_url": worker_url,
                "rid": rid,
                "reservation_id": id,
                "remaining_tokens": remaining,
                "engine_epoch": progress.epoch,
                "engine_sequence": progress.sequence,
            });
            let _ = self.emit(&mut state, event);
        }
        // Before the updates: an unacknowledged entry holds its pre-update
        // uncached tokens in the unacknowledged counter.
        // --global-queue: (id, rid, T_react sample) of first sightings of
        // releases marked for a sample.
        let mut lead_samples = Vec::new();
        for (id, tokens) in acknowledgements {
            let worker = state.workers.get_mut(worker_url).expect("reserved worker");
            worker.pending_unacknowledged_tokens = worker
                .pending_unacknowledged_tokens
                .checked_sub(u128::from(tokens))
                .expect("unacknowledged counter underflow");
            let entry = state.entries.get_mut(&id).expect("active entry");
            entry.acknowledged = true;
            entry.ack_at = Some(received_at);
            if let Some(mark) = entry.queue_lead_mark.take().filter(|m| m.sample) {
                let sample_s = sent_at
                    .saturating_duration_since(mark.released_at)
                    .as_secs_f64();
                lead_samples.push((id, entry.rid.clone(), sample_s, mark));
            }
        }
        if !lead_samples.is_empty() {
            let mut events = Vec::with_capacity(lead_samples.len());
            if let Some(book) = state.queue_lead.as_mut() {
                for (id, rid, sample_s, mark) in lead_samples {
                    let react_s = book.observe(worker_url, sent_at, sample_s);
                    let samples = book.by_p.get(worker_url).map_or(0, |e| e.samples);
                    events.push(json!({
                        "event": "queue_lead_sample",
                        "worker_url": worker_url,
                        "rid": rid,
                        "reservation_id": id,
                        "sample_s": sample_s,
                        "react_s": react_s,
                        "samples": samples,
                        "restore_tokens": mark.restore_tokens,
                        "pull_tokens": mark.pull_tokens,
                    }));
                }
            }
            for event in events {
                let _ = self.emit(&mut state, event);
            }
        }
        for (id, old, remaining, old_cost, cost) in updates {
            let worker = state.workers.get_mut(worker_url).expect("reserved worker");
            worker.pending_uncached_tokens = worker
                .pending_uncached_tokens
                .checked_sub(u128::from(old))
                .expect("pending token counter underflow")
                + u128::from(remaining);
            worker.cost = worker.cost - old_cost + cost;
            let entry = state.entries.get_mut(&id).expect("active entry");
            entry.uncached_tokens = remaining;
            entry.seen_engine_progress = true;
            entry.cost = cost;
        }
        for id in &listed {
            let entry = state.entries.get_mut(id).expect("active entry");
            entry.last_progress_seen = Some(sent_at);
        }
        state.progress.insert(
            worker_url.into(),
            ProgressCursor {
                epoch: progress.epoch,
                sequence: progress.sequence,
                sent_at,
            },
        );
        // cost-function auto: the largest remaining work seen; 0 ends the prefill.
        let now = Instant::now();
        for (id, remaining) in auto_seen {
            if let Some((url, period)) = auto_progress(&mut state, id, remaining, now) {
                let _ = self.emit(&mut state, p_empty_event(&url, period));
            }
        }
        self.expire_unknown(&mut state, Some(worker_url), now);
        Ok(())
    }

    /// Remove the entries that ended more than [`RETIRE_AFTER_S`] before
    /// `now` from `entries`, count their statuses per worker, and free their
    /// rids (a rid released by a rollback before dispatch was freed then).
    /// The record of an expired rid goes with its entry.
    fn retire_ended(state: &mut State, now: Instant) {
        while let Some(&(ended_at, id)) = state.ended.front() {
            if now.saturating_duration_since(ended_at).as_secs_f64() <= RETIRE_AFTER_S {
                break;
            }
            state.ended.pop_front();
            let Some(entry) = state.entries.remove(&id) else {
                continue;
            };
            if let Some(rid) = &entry.rid {
                if !entry.rid_released {
                    state.used_rids.remove(rid);
                }
                if state.expired_by_rid.get(rid).is_some_and(|e| e.0 == id) {
                    state.expired_by_rid.remove(rid);
                }
            }
            *state
                .retired_counts
                .entry(entry.worker_url)
                .or_default()
                .entry(entry.status.label())
                .or_default() += 1;
            state.retired_entries += 1;
        }
    }

    /// Record a periodic snapshot and return it for the caller's monitoring.
    /// Health is the most recently supplied candidate set, not a new probe.
    /// The unknown expiry of every worker runs first (see the module docs).
    pub fn snapshot(&self) -> Value {
        let mut state = self.state.lock();
        self.expire_unknown(&mut state, None, Instant::now());
        Self::retire_ended(&mut state, Instant::now());
        // Status counts (retired entries included) and the oldest active
        // assigned time of each worker.
        let mut by_worker = HashMap::<&str, (BTreeMap<&str, u64>, Option<u64>)>::new();
        for (url, counts) in &state.retired_counts {
            by_worker.entry(url.as_str()).or_default().0 = counts.clone();
        }
        for entry in state.entries.values() {
            let (counts, oldest_assigned_ms) =
                by_worker.entry(entry.worker_url.as_str()).or_default();
            *counts.entry(entry.status.label()).or_default() += 1;
            if entry.status.active() {
                *oldest_assigned_ms =
                    Some(oldest_assigned_ms.map_or(entry.assigned_unix_ms, |old| {
                        old.min(entry.assigned_unix_ms)
                    }));
            }
        }
        let workers: Vec<Value> = state
            .workers
            .iter()
            .map(|(url, worker)| {
                let (counts, oldest_assigned_ms) =
                    by_worker.remove(url.as_str()).unwrap_or_default();
                json!({
                    "worker_url": url,
                    "cost": worker.cost.to_string(),
                    "pending_uncached_tokens": worker.pending_uncached_tokens.to_string(),
                    "active_requests": worker.active_requests,
                    "status_counts": counts,
                    "oldest_active_assigned_unix_ms": oldest_assigned_ms,
                    "in_last_healthy_candidates": state.healthy_candidates.contains(url),
                })
            })
            .collect();
        let unknown: Vec<Value> = state
            .unknown
            .iter()
            .map(|id| state.entries[id].value())
            .collect();
        let active_requests: u64 = state.workers.values().map(|w| w.active_requests).sum();
        let timestamp = unix_ms();
        let mut value = json!({
            "schema": "placement-trace.v1",
            "event": "snapshot",
            "ts_unix_ms": timestamp,
            "ts_unix_s": timestamp as f64 / 1000.0,
            "elapsed_ms": self.started.elapsed().as_millis() as u64,
            "cost_unit": "uncached_tokens",
            "active_requests": active_requests,
            "unknown_requests": state.unknown.len(),
            "prefill_reader_gone": state.prefill_reader_gone,
            "healthy_candidates": state.healthy_candidates,
            "workers": workers,
            "tracked_entries": state.entries.len(),
            "retired_entries": state.retired_entries,
            "unknown_entries": unknown,
            "fatal": state.io_failure,
            "admission_blocked": self.check_admission(&state).is_err(),
        });
        if let Some(expire) = self.unknown_expire {
            // Only with an unknown expiry, so other traces keep their fields.
            value["unknown_expire_s"] = json!(expire.as_secs_f64());
            value["expired_rid_listed_again"] = json!(state.expired_listed_again);
        }
        if let Some(book) = &state.session_moves {
            // --symmetric-pd-trace-session-moves only: decisions with a session,
            // P changes among them, sessions seen.
            value["session_moves"] = json!({
                "decisions": book.decisions,
                "p_changes": book.p_changes,
                "sessions": book.last.len(),
            });
        }
        if let Some(auto) = state.auto_price.as_ref() {
            // cost-function auto only: the per-P estimates now (N_active itself is
            // in each decision; `sessions_tracked` also holds sessions that
            // wait for their idle expiry).
            let now = Instant::now();
            let workers: Vec<Value> = auto
                .worker_urls()
                .map(|url| {
                    let view = auto.worker(url, now);
                    json!({
                        "worker_url": url,
                        "in_flight": auto.in_flight(url),
                        "busy": view.map(|v| v.busy),
                        "inflight_mean": view.map(|v| v.inflight),
                        "prefill_rate_tokens_per_s": view.map(|v| v.prefill_rate_tokens_per_s),
                    })
                })
                .collect();
            value["auto_price"] = json!({
                "elapsed_s": now.saturating_duration_since(auto.started()).as_secs_f64(),
                "workers": workers,
                "sessions_tracked": auto.tracked_sessions(),
                "sessions_dropped": auto.sessions_dropped,
                "credited": auto.credited,
                "credited_tokens": auto.credited_tokens,
                "uncredited": auto.uncredited,
            });
        }
        if let Err(error) = self.emit(&mut state, value.clone()) {
            value["fatal"] = json!(error);
            value["admission_blocked"] = json!(true);
        }
        value["sequence"] = json!(state.event_sequence);
        value
    }
}

/// The global waiting queue (`--cost-function-global-queue`).
impl PrefillWorkBook {
    /// The request-independent part of one cost-function decision at `decided_at`,
    /// under the lock: the copy price and the rate resolved from the router's
    /// measurements over the P of `urls` (`auto`; the auto state starts at
    /// the first call). Fixed modes resolve to the configured parameters bit
    /// for bit.
    fn cost_ctx(
        state: &mut State,
        configured: crate::scheduler::placement_cost::PlacementCostParams,
        urls: &[&str],
        decided_at: Instant,
    ) -> CostDecisionCtx {
        // Copy price or rate auto: the measurements at this decision give
        // lambda and R of this decision ([`PlacementCostParams::resolved`]); the
        // rest of the parameters is unchanged. Fixed modes skip this, so
        // their decision and trace are the same as before.
        let auto = if configured.is_auto() {
            let book = state.auto_price.get_or_insert_with(|| {
                crate::scheduler::placement_cost::AutoPriceState::new(
                    decided_at,
                    configured.auto_tau_s,
                )
            });
            book.observe_pool(urls.iter().copied(), decided_at);
            let measured = book.inputs(decided_at, urls, configured.auto_session_idle_s);
            let views: Vec<_> = urls
                .iter()
                .map(|url| {
                    let view = book.worker(url, decided_at);
                    (url.to_string(), book.in_flight(url), view)
                })
                .collect();
            let (resolved, resolution) = configured.resolved(&measured);
            Some((resolved, resolution, measured, views))
        } else {
            None
        };
        let params = auto.as_ref().map_or(configured, |a| a.0);
        CostDecisionCtx {
            decided_at,
            params,
            auto,
        }
    }

    /// The unacknowledged tokens of the reservations in the unknown state
    /// (transport error), per P. They are not known to be in an engine
    /// queue, so the queued work of the global waiting queue leaves them out.
    fn unknown_unacknowledged(state: &State) -> BTreeMap<&str, u128> {
        let mut out: BTreeMap<&str, u128> = BTreeMap::new();
        for entry in state.unknown.iter().filter_map(|id| state.entries.get(id)) {
            if entry.status == Status::Unknown && !entry.acknowledged {
                *out.entry(entry.worker_url.as_str()).or_default() +=
                    u128::from(entry.uncached_tokens);
            }
        }
        out
    }

    /// The work of `url` for the global waiting queue: (queued, unlaunched,
    /// launched). Unlaunched: the uncached tokens routed to the P whose first
    /// prefill chunk is not in a progress snapshot yet, without the unknown
    /// reservations. Launched: the remaining work of the requests that a
    /// snapshot has listed. Queued: unlaunched + launched beyond one pass of
    /// `pass_tokens` ([`crate::global_queue::queued_tokens`]).
    fn queue_work(
        state: &State,
        unknown: &BTreeMap<&str, u128>,
        url: &str,
        pass_tokens: u64,
    ) -> (u128, u128, u128) {
        state.workers.get(url).map_or((0, 0, 0), |w| {
            let unlaunched = w
                .pending_unacknowledged_tokens
                .saturating_sub(unknown.get(url).copied().unwrap_or(0));
            let launched = w
                .pending_uncached_tokens
                .saturating_sub(w.pending_unacknowledged_tokens);
            (
                crate::global_queue::queued_tokens(unlaunched, launched, pass_tokens),
                unlaunched,
                launched,
            )
        })
    }

    /// One decision step of the global waiting queue under one lock: the P
    /// rows from the work book (backlog as cost-function uses it, queued work, view
    /// freshness, a_w), the cost-function decision context, the walk
    /// ([`crate::global_queue::walk`]) and, when the walk releases a request,
    /// its reservation with the normal cost-function record, the virtual backlog in
    /// the backlog terms and the `queue` object. `held` is in queue order.
    /// `Err` only when admission is stopped (trace failure).
    pub fn queue_step(
        self: &Arc<Self>,
        configured: crate::scheduler::placement_cost::PlacementCostParams,
        held: &[QueueHeld<'_>],
        args: &QueueStepArgs<'_>,
    ) -> Result<QueueStep, String> {
        use crate::global_queue::{
            terms_of, walk_with, QueueHeldInput, QueueRow, QueueTerms, QueueTie,
        };
        let mut state = self.state.lock();
        self.check_admission(&state)?;
        for url in args.urls {
            state.workers.entry(url.clone()).or_default();
        }
        let urls: Vec<&str> = args.urls.iter().map(String::as_str).collect();
        let ctx = Self::cost_ctx(&mut state, configured, &urls, args.now);
        // --global-queue: the lead rule of each P with R of this step.
        let rate = ctx.params.prefill_rate_tokens_per_s;
        let leads: Option<Vec<QueueLeadRow>> = args.auto.map(|auto| {
            urls.iter()
                .zip(auto.chunks)
                .map(|(url, &(chunk, chunk_source))| {
                    let (react_s, react_source, react_samples) = state
                        .queue_lead
                        .as_ref()
                        .map_or((0.0, "none", 0), |book| book.react(url));
                    QueueLeadRow::new(
                        chunk,
                        chunk_source,
                        react_s,
                        react_source,
                        react_samples,
                        rate,
                    )
                })
                .collect()
        });
        if let (Some(leads), Some(book)) = (leads.as_ref(), state.queue_lead.as_mut()) {
            for (url, lead) in urls.iter().zip(leads) {
                book.last_step
                    .insert(url.to_string(), (lead.lead_s, lead.depth, rate));
            }
        }
        let unknown = Self::unknown_unacknowledged(&state);
        let work: Vec<(u128, u128, u128)> = args
            .urls
            .iter()
            .enumerate()
            .map(|(i, url)| {
                let pass = leads.as_ref().map_or(args.pass_tokens, |l| l[i].chunk);
                Self::queue_work(&state, &unknown, url, pass)
            })
            .collect();
        drop(unknown);
        // `room = forecast`: the forecast drain of each P (a cold one-chunk
        // probe on the newest pushed or polled state, plus the router's own
        // rows that the engine does not show yet), as a the forecast wait decision
        // computes it under this lock.
        let drains: Vec<(Option<f64>, Option<&'static str>)> = match args.forecast {
            Some(forecast) => {
                while state.recent_finished.front().is_some_and(|(at, _)| {
                    args.now.saturating_duration_since(*at).as_secs_f64() > RECENT_FINISHED_S
                }) {
                    state.recent_finished.pop_front();
                }
                let mut ids: Vec<u64> = state
                    .active_by_rid
                    .values()
                    .copied()
                    .chain(state.recent_finished.iter().map(|(_, id)| *id))
                    .collect();
                ids.sort_unstable();
                ids.dedup();
                args.urls
                    .iter()
                    .map(|url| {
                        let probe = Self::probe_forecast(
                            &state,
                            Some(forecast),
                            &queue_probe_candidate(url),
                            args.now,
                            &ctx.params,
                            &ids,
                        );
                        queue_probe_drain(&probe)
                    })
                    .collect()
            }
            None => vec![(None, None); args.urls.len()],
        };
        let rows: Vec<QueueRow> = args
            .urls
            .iter()
            .zip(args.depth)
            .zip(&work)
            .zip(&drains)
            .enumerate()
            .map(|(i, (((url, depth), (queued, _, _)), (drain_s, _)))| {
                let pending = state.workers[url].pending_uncached_tokens;
                QueueRow {
                    // The forecast may make a P look emptier than the
                    // pending work, never fuller (work-conserving floor).
                    backlog: match drain_s {
                        Some(d) => queue_drain_tokens(*d, rate).min(pending),
                        None => pending,
                    },
                    drain_s: *drain_s,
                    // --global-queue: the lead rule of the P (C / R +
                    // T_react), as the token rule's depth.
                    lead_s: leads.as_ref().map_or(args.lead_s, |l| l[i].lead_s),
                    queued: *queued,
                    depth: leads.as_ref().map_or(*depth, |l| l[i].depth),
                    fresh: state.progress.get(url).is_some_and(|cursor| {
                        args.now.saturating_duration_since(cursor.sent_at) <= args.stale_after
                    }),
                }
            })
            .collect();
        // The candidates of each held request over the step pool (a P that
        // joined is cold, a P that left is gone), and their terms.
        let pooled: Vec<Vec<Candidate>> = held
            .iter()
            .map(|h| queue_pool_candidates(h.candidates, args.urls, args.sources))
            .collect();
        let terms: Vec<Vec<QueueTerms>> = held
            .iter()
            .zip(&pooled)
            .map(|(h, cands)| {
                cands
                    .iter()
                    .map(|c| terms_of(c.local_hit, c.expected_hit, h.input_tokens))
                    .collect()
            })
            .collect();
        let inputs: Vec<QueueHeldInput<'_>> = held
            .iter()
            .zip(&terms)
            .map(|(h, terms)| {
                let waited = args.now.saturating_duration_since(h.arrived);
                QueueHeldInput {
                    terms,
                    waited_s: waited.as_secs_f64(),
                    starved: waited >= args.max_wait,
                }
            })
            .collect();
        // --global-queue: exact ties go to the P with room, the larger
        // own hit, the fewest outstanding requests, then the next URL after
        // the last tie (rotation); the release reservation breaks them alike.
        let outstanding: Vec<u64> = args
            .urls
            .iter()
            .map(|url| state.workers[url].active_requests)
            .collect();
        let tie_last = state.queue_lead.as_ref().and_then(|b| b.tie_last.clone());
        let tie = leads.as_ref().map(|_| QueueTie {
            outstanding: &outstanding,
            last: tie_last.as_deref(),
        });
        let walked = walk_with(&ctx.params, args.urls, &rows, &inputs, tie.as_ref());
        let queue_tokens: u128 = walked
            .plain
            .iter()
            .zip(&terms)
            .map(|(p, t)| u128::from(t[*p].uncached))
            .sum();
        let Some(pick) = walked.pick else {
            return Ok(QueueStep {
                plain: walked.plain,
                queue_tokens,
                release: None,
            });
        };
        let h = held[pick.index];
        let p_release = args.urls[pick.worker].as_str();
        let plain_worker = args.urls[pick.plain_worker].as_str();
        let p_arrival = h.p_arrival.unwrap_or(plain_worker);
        let late_bound = p_arrival != p_release;
        let row = &rows[pick.worker];
        let mut record = json!({
            "mode": "hrrn",
            "held": !h.arrival_pass,
            "hold_s": args.now.saturating_duration_since(h.arrived).as_secs_f64(),
            "arrived_unix_ms": h.arrived_unix_ms,
            "release": pick.release.as_str(),
            "starved": pick.starved,
            "key": pick.key,
            "service_s": pick.service_s,
            "plain_worker": plain_worker,
            "p_arrival": p_arrival,
            "p_release": p_release,
            "late_bound": late_bound,
            "rank": pick.rank,
            "queue_len": held.len(),
            "queue_tokens": queue_tokens.to_string(),
            "depth_tokens": row.depth,
            "queued_tokens": row.queued.to_string(),
            "unlaunched_tokens": work[pick.worker].1.to_string(),
            "launched_tokens": work[pick.worker].2.to_string(),
            "view_fresh": row.fresh,
            "virtual_tokens": pick.virtual_tokens[pick.worker].to_string(),
            "room_rule": if args.forecast.is_some() { "forecast" } else { "tokens" },
            "lead_s": args.forecast.map(|_| args.lead_s),
            "drain_s": row.drain_s,
            "drain_s_by_p": args.forecast.map(|_| drains.iter().map(|(d, _)| *d).collect::<Vec<_>>()),
            "drain_fallback_by_p": args.forecast.map(|_| drains.iter().map(|(_, r)| *r).collect::<Vec<_>>()),
            "session_known": h.session.is_some(),
            "pass_id": args.pass_id,
            "trigger": args.trigger,
            "triggers": args.triggers,
        });
        // --global-queue: the lead rule of the released P, the class
        // inputs, and the shadow counters of the working-set hold (ii).
        let lead_sample = row.queued == 0 && work[pick.worker].2 == 0 && row.fresh;
        if let Some(lead) = leads.as_ref().map(|l| &l[pick.worker]) {
            let hits: Vec<u64> = h
                .candidates
                .iter()
                .map(|c| c.local_hit.min(h.input_tokens))
                .collect();
            let (h1, h2) = crate::global_queue::top_two(&hits);
            let extra = json!({
                "rule": "auto",
                "class": "free",
                "h1_tokens": h1,
                "h2_tokens": h2,
                "chunk_tokens": lead.chunk,
                "chunk_source": lead.chunk_source,
                "prefill_rate_tokens_per_s": rate,
                "react_s": lead.react_s,
                "react_source": lead.react_source,
                "react_samples": lead.react_samples,
                "lead_s": lead.lead_s,
                "wait_s": row.queued as f64 / rate,
                "lead_sample": lead_sample,
            });
            if let (Value::Object(record), Value::Object(extra)) = (&mut record, extra) {
                record.extend(extra);
            }
        }
        // --global-queue: the walk's tie-break, as the release
        // reservation records it in the reserve event (the counters of the
        // queue read it from this record).
        if let Some(t) = pick.tie {
            record["tie"] = json!({
                "tied": t.tied,
                "decided_by": t.decided_by,
                "outstanding": outstanding[pick.worker],
                "last": tie_last,
            });
        }
        let notes: BTreeMap<String, QueueRowNote> = args
            .urls
            .iter()
            .zip(&rows)
            .zip(&pick.virtual_tokens)
            .map(|((url, row), virtual_tokens)| {
                (
                    url.clone(),
                    QueueRowNote {
                        virtual_tokens: *virtual_tokens,
                        depth: row.depth,
                        room: row.room().is_some(),
                        fresh: row.fresh,
                        backlog_override: row.drain_s.map(|_| row.backlog),
                        drain_s: row.drain_s,
                    },
                )
            })
            .collect();
        let reserve = QueueReserve {
            ctx,
            rows: notes,
            record: record.clone(),
            tie_break: tie.is_some(),
        };
        let release_candidates = pooled.into_iter().nth(pick.index).unwrap_or_default();
        let (restore_tokens, pull_tokens) = release_candidates
            .iter()
            .find(|c| c.worker_url == p_release)
            .map_or((0, 0), |c| {
                let local = c.local_hit.min(h.input_tokens);
                let gpu = c.gpu_hit.unwrap_or(0).min(local);
                let pulled = if c.hint.is_some() {
                    c.expected_hit.saturating_sub(local)
                } else {
                    0
                };
                (local - gpu, pulled)
            });
        let result = self
            .validate_placement(h.input_tokens, &release_candidates)
            .and_then(|valid| {
                self.reserve_locked(
                    &mut state,
                    h.input_tokens,
                    h.rid.map(str::to_string),
                    release_candidates,
                    valid,
                    configured,
                    h.session,
                    None,
                    Some(reserve),
                    ReserveExtras {
                        queue_note: None,
                        session_label: h.session_label,
                    },
                )
            })
            .map(|(candidate, reservation, decision)| QueueReleased {
                candidate,
                reservation,
                decision: decision.unwrap_or(Value::Null),
                record,
                release: pick.release,
                starved: pick.starved,
                late_bound,
            });
        if let (Ok(released), true) = (&result, leads.is_some()) {
            // The first sighting of the RID gives a T_react sample when P
            // had no queued work (an empty engine queue) and a fresh view.
            if let Some(entry) = state.entries.get_mut(&released.reservation.id) {
                entry.queue_lead_mark = Some(QueueLeadMark {
                    released_at: args.now,
                    sample: lead_sample,
                    restore_tokens,
                    pull_tokens,
                });
            }
            if let Some(book) = state.queue_lead.as_mut() {
                if lead_sample {
                    book.marked += 1;
                } else {
                    book.unmarked += 1;
                }
            }
        }
        Ok(QueueStep {
            plain: walked.plain,
            queue_tokens,
            release: Some((pick.index, result)),
        })
    }

    /// `--global-queue`: T_react and the lead of the last step of each P
    /// of `urls`, and the sample counters (the `queue_stats` event). Null
    /// without [`Self::enable_queue_lead`].
    pub fn queue_lead_rows(&self, urls: &[String]) -> Value {
        let state = self.state.lock();
        let Some(book) = state.queue_lead.as_ref() else {
            return Value::Null;
        };
        let rows: serde_json::Map<String, Value> = urls
            .iter()
            .map(|url| {
                let (react_s, source, samples) = book.react(url);
                let last = book.by_p.get(url.as_str()).map(|e| e.last_sample_s);
                let step = book.last_step.get(url.as_str());
                (
                    url.clone(),
                    json!({
                        "react_s": react_s,
                        "react_source": source,
                        "react_samples": samples,
                        "last_sample_s": last,
                        "lead_s": step.map(|s| s.0),
                        "depth_tokens": step.map(|s| s.1),
                        "prefill_rate_tokens_per_s": step.map(|s| s.2),
                    }),
                )
            })
            .collect();
        json!({
            "by_p": rows,
            "marked": book.marked,
            "unmarked": book.unmarked,
            "samples": book.samples,
            "tau_s": book.tau_s,
            "fallback_s": book.fallback_s,
            "restarts": book.restarts,
            "tie_last": book.tie_last,
        })
    }

    /// Per P of `urls`: the queued work (one pass of `pass_tokens` is the
    /// running pass), its parts and the age of the progress view, for the
    /// `queue_stats` event.
    pub fn queue_rows(&self, urls: &[String], now: Instant, pass_tokens: u64) -> Vec<QueueRowStat> {
        let state = self.state.lock();
        let unknown = Self::unknown_unacknowledged(&state);
        urls.iter()
            .map(|url| {
                let (queued, unlaunched, launched) =
                    Self::queue_work(&state, &unknown, url, pass_tokens);
                QueueRowStat {
                    queued,
                    unlaunched,
                    launched,
                    view_age_ms: state.progress.get(url).map(|cursor| {
                        now.saturating_duration_since(cursor.sent_at).as_millis() as u64
                    }),
                }
            })
            .collect()
    }

    /// Write one more event to the trace (`queue_drop`, `queue_stats`), with the
    /// usual sequence and time keys. A write failure stops admission, as for
    /// every event.
    pub fn record_event(&self, event: Value) -> Result<(), String> {
        let mut state = self.state.lock();
        self.emit(&mut state, event)
    }

    /// Why `rid` cannot be reserved (missing, empty or used), with the same
    /// text as the reservation error; None when it can.
    pub fn rid_refusal(&self, rid: Option<&str>) -> Option<&'static str> {
        match rid.filter(|r| !r.is_empty()) {
            None => Some("uncached balance requires a unique string rid"),
            Some(rid) if self.state.lock().used_rids.contains(rid) => {
                Some("uncached balance cannot reuse a rid; stale engine snapshots may contain it")
            }
            Some(_) => None,
        }
    }

    /// Set the function that runs (outside the lock) after each terminal
    /// status change. Only the first call has an effect.
    pub fn set_terminal_hook(&self, hook: Box<dyn Fn() + Send + Sync>) {
        let _ = self.on_terminal.set(hook);
    }
}

/// Unique task owner. Dropping before HTTP dispatch rolls back; dropping after
/// dispatch without an explicit terminal result retains cost as unknown.
#[derive(Debug)]
pub struct Reservation {
    book: Arc<PrefillWorkBook>,
    id: u64,
}

impl Reservation {
    pub fn mark_dispatched(&mut self) -> Result<(), String> {
        self.book
            .change_status(self.id, Status::Dispatched, "prefill_http_dispatch")
    }

    pub fn finish(&mut self, reason: &str) -> Result<(), String> {
        self.book.change_status(self.id, Status::Finished, reason)
    }

    pub fn mark_unknown(&mut self, reason: &str) -> Result<(), String> {
        self.book.change_status(self.id, Status::Unknown, reason)
    }

    pub fn debuginfo(&self) -> Value {
        let state = self.book.state.lock();
        state.entries.get(&self.id).map_or_else(
            || json!({"reservation_id": self.id, "retired": true}),
            Entry::value,
        )
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        // A retired entry had ended.
        let status = self
            .book
            .state
            .lock()
            .entries
            .get(&self.id)
            .map(|e| e.status);
        let result = match status {
            Some(Status::Reserved) => self.book.change_status(
                self.id,
                Status::RolledBack,
                "owner_dropped_before_dispatch",
            ),
            Some(Status::Dispatched) => self.book.change_status(
                self.id,
                Status::Unknown,
                "dispatched_owner_dropped_without_terminal_result",
            ),
            _ => Ok(()),
        };
        if let Err(error) = result {
            tracing::error!(target: "cost_balance", reservation_id = self.id, %error,
                "failed to record reservation owner teardown");
        }
    }
}

pub(crate) fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// The `finish` reason of a P response with a success status (the routes'
/// literal). Only this terminal reason credits the computed tokens to the
/// cost-function auto rate.
const PREFILL_SUCCESS_REASON: &str = "prefill_response_success";

/// cost-function auto: apply one status change of entry `id` to the auto state.
/// `Dispatched`: one more in-flight request on its P. `Finished`, `Unknown`,
/// `RolledBack` or `Expired` (only after `Unknown`, so it never applies
/// again): the end of its prefill, applied once: its P is released
/// if it was dispatched (tokens credited only for [`PREFILL_SUCCESS_REASON`])
/// and its session in every case. Returns the P and its busy period when the
/// P has no request left. Untracked entries change nothing.
fn auto_status(
    state: &mut State,
    id: u64,
    target: Status,
    reason: &str,
    now: Instant,
) -> Option<(
    String,
    crate::scheduler::placement_cost::AutoPriceBusyPeriod,
)> {
    let entry = state.entries.get_mut(&id)?;
    if !entry.auto.tracked || entry.auto.done {
        return None;
    }
    let auto = state.auto_price.as_mut()?;
    match target {
        Status::Reserved => None,
        Status::Dispatched => {
            if !entry.auto.in_flight {
                auto.dispatched(&entry.worker_url, now);
                entry.auto.in_flight = true;
            }
            None
        }
        Status::Finished | Status::Unknown | Status::RolledBack | Status::Expired => {
            entry.auto.done = true;
            let credit = (target == Status::Finished && reason == PREFILL_SUCCESS_REASON)
                .then_some(entry.auto.peak_tokens);
            let period = if std::mem::take(&mut entry.auto.in_flight) {
                auto.prefill_done(&entry.worker_url, credit, now)
            } else {
                None
            };
            if let Some(session) = entry.auto.session {
                auto.session_finished(session, now);
            }
            period.map(|period| (entry.worker_url.clone(), period))
        }
    }
}

/// cost-function auto: one progress observation of a tracked, dispatched entry. The
/// computed tokens become the larger of their value and `remaining`; 0
/// remaining work ends its prefill (credited, applied once). Returns the P
/// and its busy period when the P has no request left.
fn auto_progress(
    state: &mut State,
    id: u64,
    remaining: u64,
    now: Instant,
) -> Option<(
    String,
    crate::scheduler::placement_cost::AutoPriceBusyPeriod,
)> {
    let entry = state.entries.get_mut(&id)?;
    if !entry.auto.tracked || entry.auto.done || !entry.auto.in_flight {
        return None;
    }
    entry.auto.peak_tokens = entry.auto.peak_tokens.max(remaining);
    if remaining > 0 {
        return None;
    }
    let auto = state.auto_price.as_mut()?;
    entry.auto.done = true;
    entry.auto.in_flight = false;
    let period = auto.prefill_done(&entry.worker_url, Some(entry.auto.peak_tokens), now);
    if let Some(session) = entry.auto.session {
        auto.session_finished(session, now);
    }
    period.map(|period| (entry.worker_url.clone(), period))
}

/// The trace event of a P whose cost-function auto busy period ended.
fn p_empty_event(
    worker_url: &str,
    period: crate::scheduler::placement_cost::AutoPriceBusyPeriod,
) -> Value {
    json!({
        "event": "auto_price_p_empty",
        "worker_url": worker_url,
        "busy_period_s": period.seconds,
        "busy_period_requests": period.requests,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    fn candidate(url: &str, hit: u64) -> Candidate {
        Candidate {
            worker_url: url.into(),
            local_hit: hit,
            gpu_hit: None,
            expected_hit: hit,
            hint: None,
        }
    }

    fn cost(book: &PrefillWorkBook, url: &str) -> u128 {
        book.state.lock().workers[url].cost
    }

    fn uncached(book: &Arc<PrefillWorkBook>, rid: &str, url: &str, tokens: u64) -> Reservation {
        res(book, rid, tokens, vec![candidate(url, 0)]).unwrap().1
    }

    /// A reservation with the default cost-function parameters.
    fn res(
        book: &Arc<PrefillWorkBook>,
        rid: &str,
        tokens: u64,
        candidates: Vec<Candidate>,
    ) -> Result<(Candidate, Reservation), String> {
        book.reserve_cost(tokens, Some(rid.into()), candidates, Default::default())
            .map(|(candidate, reservation, _)| (candidate, reservation))
    }

    fn progress(sequence: u64, remaining: &[(&str, u64)]) -> PrefillProgress {
        PrefillProgress {
            epoch: "engine-1".into(),
            sequence,
            remaining: remaining
                .iter()
                .map(|(rid, n)| (rid.to_string(), *n))
                .collect(),
            truncated: false,
        }
    }

    fn cost_params(lambda: f64) -> crate::scheduler::placement_cost::PlacementCostParams {
        crate::scheduler::placement_cost::PlacementCostParams {
            copy_price: lambda,
            ..Default::default()
        }
    }

    fn pulling(url: &str, local: u64, remote: u64) -> Candidate {
        Candidate {
            worker_url: url.into(),
            local_hit: local,
            gpu_hit: None,
            expected_hit: remote,
            hint: Some(json!({"source_worker_id": "src", "expected_hit_tokens": remote})),
        }
    }

    fn unacknowledged(book: &PrefillWorkBook, url: &str) -> u128 {
        book.state.lock().workers[url].pending_unacknowledged_tokens
    }

    /// Owner `a` holds 100k of 110k with 60k pending; `b` is idle and would
    /// pull 100k. lambda 0 moves (17,425 < 70,000); lambda 9.2 keeps `a`.
    /// The placement decision names the owner, the pull and every candidate.
    #[test]
    fn cost_moves_with_pull_at_lambda_zero_and_keeps_owner_at_large_lambda() {
        for (lambda, chosen, pulled) in [(0.0, "b", 100_000u64), (9.2, "a", 0)] {
            let book = PrefillWorkBook::new(None, "test", None).unwrap();
            let mut busy = uncached(&book, "busy", "a", 60_000);
            busy.mark_dispatched().unwrap();
            book.apply_progress(
                "a",
                progress(1, &[("busy", 60_000)]),
                json!({}),
                Instant::now(),
            )
            .unwrap();
            let (selected, _r, decision) = book
                .reserve_cost(
                    110_000,
                    Some("req".into()),
                    vec![candidate("a", 100_000), pulling("b", 0, 100_000)],
                    cost_params(lambda),
                )
                .unwrap();
            assert_eq!(selected.worker_url, chosen, "lambda {lambda}");
            assert_eq!(decision["owner"], "a");
            assert_eq!(decision["detour"], chosen != "a");
            assert_eq!(decision["pulled_tokens"], pulled);
            assert_eq!(decision["candidates"].as_array().unwrap().len(), 2);
            // The work book keeps uncached-token units: the pulled prefix is not charged.
            assert_eq!(
                cost(&book, chosen),
                if chosen == "a" { 70_000 } else { 10_000 }
            );
            drop(busy);
        }
    }

    fn auto_params(fallback: f64) -> crate::scheduler::placement_cost::PlacementCostParams {
        use crate::scheduler::placement_cost::{CopyPriceMode, PlacementCostParams};
        PlacementCostParams {
            copy_price: fallback,
            copy_price_mode: CopyPriceMode::Auto {
                min: 4.6,
                max: 1000.0,
            },
            ..Default::default()
        }
    }

    fn routing_snapshot(waiting: usize, tokens: u64) -> Value {
        let rows: Vec<Value> = (0..waiting)
            .map(|i| {
                json!({"rid": format!("w{i}"), "phase": "waiting", "input_tokens": tokens,
                "remaining_compute_tokens": tokens, "device_prefix_tokens": 0,
                "host_prefix_tokens": 0, "storage_prefix_tokens": 0,
                "state": {"intent_start_ok": true, "bound": false, "launched_unbound": false,
                    "held": false, "parked": false, "pull_pending": false,
                    "pull_inflight": false, "restore_inflight": false}})
            })
            .collect();
        json!({"epoch": "routing-1", "sequence": 7, "available": true, "age_s": 0.0,
            "stale": false, "truncated": false, "omitted_waiting": 0,
            "inflight_batches_known": true, "inflight_batches": [], "inflight_requests": [],
            "policy": {"page_size": 64, "chunk_tokens": 8192, "schedule_policy": "fcfs"},
            "waiting": rows, "bootstrap": [],
            "admission_resources": {"park_gate_closed_last_scan": false}})
    }

    #[test]
    fn uncached_progress_keeps_unseen_reservations_and_reconciles_actual_miss() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let mut a = uncached(&book, "a", "p", 20000);
        a.mark_dispatched().unwrap();
        let _queued = uncached(&book, "queued", "p", 6000);
        let sent = Instant::now();
        book.apply_progress(
            "p",
            progress(1, &[("a", 8000), ("queued", 0)]),
            json!({}),
            sent,
        )
        .unwrap();
        assert_eq!(cost(&book, "p"), 14000); // un-dispatched queued RID ignored
        book.apply_progress(
            "p",
            progress(2, &[]),
            json!({}),
            sent + std::time::Duration::from_millis(1),
        )
        .unwrap();
        assert_eq!(cost(&book, "p"), 14000); // omitted RID retained
        book.apply_progress(
            "p",
            progress(3, &[("a", 19000)]),
            json!({}),
            sent + std::time::Duration::from_millis(2),
        )
        .unwrap();
        assert_eq!(cost(&book, "p"), 25000); // cache miss/recompute can increase work
        book.apply_progress(
            "p",
            progress(3, &[("a", 0)]),
            json!({}),
            sent + std::time::Duration::from_millis(3),
        )
        .unwrap();
        book.apply_progress("p", progress(4, &[("a", 0)]), json!({}), sent)
            .unwrap();
        assert_eq!(cost(&book, "p"), 25000); // duplicate sequence and old response ignored
        book.apply_progress("other", progress(5, &[("a", 0)]), json!({}), sent)
            .unwrap();
        assert_eq!(cost(&book, "p"), 25000); // never update the wrong worker
        a.finish("http_done").unwrap();
        assert_eq!(cost(&book, "p"), 6000); // subtract only current remaining amount
        book.apply_progress(
            "p",
            progress(6, &[("a", 20000)]),
            json!({}),
            sent + std::time::Duration::from_millis(5),
        )
        .unwrap();
        assert_eq!(cost(&book, "p"), 6000); // cannot resurrect a terminal RID
    }

    #[test]
    fn uncached_chunk_completion_zero_keeps_http_ownership_and_restart_keeps_unknown_work() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let mut a = uncached(&book, "a", "p", 10000);
        let mut b = uncached(&book, "b", "p", 6000);
        a.mark_dispatched().unwrap();
        b.mark_dispatched().unwrap();
        b.mark_unknown("transport_timeout").unwrap();
        let sent = Instant::now();
        book.apply_progress("p", progress(9, &[("a", 0)]), json!({}), sent)
            .unwrap();
        assert_eq!(cost(&book, "p"), 6000);
        assert_eq!(book.state.lock().workers["p"].active_requests, 2);
        let mut restarted = progress(1, &[]);
        restarted.epoch = "engine-2".into();
        book.apply_progress(
            "p",
            restarted,
            json!({}),
            sent + std::time::Duration::from_millis(1),
        )
        .unwrap();
        assert_eq!(cost(&book, "p"), 6000);
        assert!(!book.snapshot()["admission_blocked"].as_bool().unwrap());
        let _c = uncached(&book, "c", "peer", 1);
        a.finish("pd_transfer_complete").unwrap();
        b.finish("abort_confirmed").unwrap();
        assert_eq!(cost(&book, "p"), 0);
        assert!(res(&book, "a", 1, vec![candidate("p", 0)]).is_err());
    }

    #[test]
    fn concurrent_uncached_arrivals_reserve_before_the_next_load_refresh() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let barrier = Arc::new(Barrier::new(18));
        let threads: Vec<_> = (0..18)
            .map(|i| {
                let book = book.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    res(
                        &book,
                        &format!("rid-{i}"),
                        4000,
                        vec![candidate("a", 0), candidate("b", 0), candidate("c", 0)],
                    )
                    .unwrap()
                })
            })
            .collect();
        let owned: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        for worker in ["a", "b", "c"] {
            assert_eq!(cost(&book, worker), 24000);
        }
        drop(owned);
        for worker in ["a", "b", "c"] {
            assert_eq!(cost(&book, worker), 0);
        }
    }

    #[test]
    fn drop_before_dispatch_rolls_back_and_retains_worker() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let (_, request) = res(&book, "r", 12, vec![candidate("a", 0)]).unwrap();
        let id = request.id;
        drop(request);
        assert_eq!(cost(&book, "a"), 0);
        assert_eq!(book.state.lock().entries[&id].status, Status::RolledBack);
        assert_eq!(book.snapshot()["workers"][0]["active_requests"], 0);
    }

    /// A rid whose entry rolled back before any send can be planned again
    /// (no engine snapshot can list it); once dispatched it stays used.
    #[test]
    fn rid_of_an_unsent_rollback_can_be_reused_but_not_after_dispatch() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let reserve = || res(&book, "x", 100, vec![candidate("a", 0)]);
        drop(reserve().unwrap().1);
        let (_, mut sent) = reserve().unwrap();
        sent.mark_dispatched().unwrap();
        sent.finish("prefill_response_success").unwrap();
        assert!(reserve().is_err());
    }

    /// Ended entries leave the book: after many finished requests and the
    /// retire window, no entry and no rid of them is left, the snapshot keeps
    /// their status counts, a late call on their reservation changes nothing,
    /// and a live request keeps its entry and its rid.
    #[test]
    fn ended_entries_leave_the_book_after_the_retire_window() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let mut live = uncached(&book, "live", "a", 100);
        live.mark_dispatched().unwrap();
        let mut ended = Vec::new();
        for i in 0..1000 {
            let mut r = uncached(&book, &format!("r{i}"), "a", 100);
            r.mark_dispatched().unwrap();
            r.finish("prefill_response_success").unwrap();
            ended.push(r);
        }
        drop(uncached(&book, "unsent", "a", 100));
        assert_eq!(book.state.lock().entries.len(), 1002);
        PrefillWorkBook::retire_ended(&mut book.state.lock(), Instant::now() + secs(61));
        {
            let state = book.state.lock();
            assert_eq!(state.entries.len(), 1);
            assert!(state.ended.is_empty());
            assert!(state.used_rids.contains("live") && state.used_rids.len() == 1);
        }
        for mut r in ended.drain(..10) {
            r.finish("prefill_response_success").unwrap();
            assert!(r.mark_dispatched().is_err());
            assert_eq!(r.debuginfo()["retired"], true);
        }
        drop(ended);
        let snapshot = book.snapshot();
        assert_eq!(snapshot["tracked_entries"], 1);
        assert_eq!(snapshot["retired_entries"], 1001);
        let counts = &snapshot["workers"][0]["status_counts"];
        assert_eq!(counts["finished"], 1000);
        assert_eq!(counts["rolled_back"], 1);
        assert_eq!(counts["dispatched"], 1);
        assert_eq!(snapshot["workers"][0]["active_requests"], 1);
        assert!(book.rid_refusal(Some("live")).is_some());
        assert!(book.rid_refusal(Some("r0")).is_none());
    }

    #[test]
    fn drop_after_dispatch_retains_unknown() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let (_, mut request) = res(&book, "r", 12, vec![candidate("a", 0)]).unwrap();
        request.mark_dispatched().unwrap();
        drop(request);
        assert_eq!(cost(&book, "a"), 12);
        assert!(res(&book, "s", 1, vec![candidate("b", 0)]).is_ok());
        assert_eq!(
            book.snapshot()["unknown_entries"].as_array().unwrap().len(),
            1
        );
    }

    #[test]
    fn explicit_unknown_can_only_release_on_explicit_terminal_result() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let (_, mut request) = res(&book, "r", 10, vec![candidate("a", 0)]).unwrap();
        request.mark_dispatched().unwrap();
        request.mark_unknown("transport_timeout").unwrap();
        request.mark_unknown("duplicate").unwrap();
        assert_eq!(cost(&book, "a"), 10);
        request.finish("engine_abort_confirmed").unwrap();
        request.finish("duplicate_terminal").unwrap();
        assert_eq!(cost(&book, "a"), 0);
        assert!(res(&book, "s", 1, vec![candidate("a", 0)]).is_ok());
    }

    #[test]
    fn dispatch_is_idempotent_but_cannot_reopen_unknown_or_terminal_entries() {
        for status in [Status::Unknown, Status::Finished, Status::RolledBack] {
            let book = PrefillWorkBook::new(None, "test", None).unwrap();
            let (_, mut request) = res(&book, "r", 10, vec![candidate("a", 0)]).unwrap();
            request.mark_dispatched().unwrap();
            request.mark_dispatched().unwrap();
            assert_eq!(cost(&book, "a"), 10);
            book.change_status(request.id, status, "test_transition")
                .unwrap();
            assert!(request.mark_dispatched().is_err());
            assert_eq!(request.debuginfo()["status"], status.label());
            if status == Status::Unknown {
                assert_eq!(cost(&book, "a"), 10);
                request.finish("confirmed_engine_terminal").unwrap();
            }
            request.finish("idempotent_terminal").unwrap();
            assert_eq!(cost(&book, "a"), 0);
        }
    }

    #[test]
    fn parallel_reservations_see_each_others_committed_cost() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let barrier = Arc::new(Barrier::new(16));
        let threads: Vec<_> = (0..16)
            .map(|i| {
                let book = Arc::clone(&book);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    let rid = format!("r{i}");
                    res(&book, &rid, 10, vec![candidate("a", 0), candidate("b", 0)]).unwrap()
                })
            })
            .collect();
        let owned: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert_eq!(cost(&book, "a"), 80);
        assert_eq!(cost(&book, "b"), 80);
        assert_eq!(owned.len(), 16);
    }

    #[test]
    fn health_subset_does_not_erase_excluded_workers_entries() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let (_, mut a) = res(&book, "r1", 10, vec![candidate("a", 0)]).unwrap();
        a.mark_dispatched().unwrap();
        let (_, b) = res(&book, "r2", 2, vec![candidate("b", 0)]).unwrap();
        assert_eq!(cost(&book, "a"), 10);
        assert_eq!(book.snapshot()["healthy_candidates"], json!(["b"]));
        assert_eq!(book.snapshot()["workers"].as_array().unwrap().len(), 2);
        let (chosen, c) = res(&book, "r3", 2, vec![candidate("a", 0), candidate("b", 0)]).unwrap();
        assert_eq!(chosen.worker_url, "b");
        a.finish("old_request_finished_after_recovery").unwrap();
        assert_eq!(cost(&book, "a"), 0);
        assert_eq!(cost(&book, "b"), 4);
        drop((b, c));
    }

    #[test]
    fn hit_clamps_and_remote_hit_requires_real_hint() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let (_, request) = res(&book, "r1", 10, vec![candidate("a", 999)]).unwrap();
        assert_eq!(request.debuginfo()["expected_hit"], 10);
        assert_eq!(request.debuginfo()["cost"], "0");
        let invalid = Candidate {
            worker_url: "b".into(),
            local_hit: 0,
            gpu_hit: None,
            expected_hit: 8,
            hint: None,
        };
        assert!(res(&book, "r2", 10, vec![invalid]).is_err());
        let hinted = Candidate {
            worker_url: "b".into(),
            local_hit: 0,
            gpu_hit: None,
            expected_hit: 8,
            hint: Some(json!({"expected_hit_tokens": 8})),
        };
        let (_, remote) = res(&book, "r3", 10, vec![hinted]).unwrap();
        assert_eq!(remote.debuginfo()["cost"], "2");
    }

    #[test]
    fn trace_is_exclusive_ordered_and_costs_are_strings() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cost.jsonl");
        let book = PrefillWorkBook::new(Some(&path), "test", None).unwrap();
        assert!(PrefillWorkBook::new(Some(&path), "test", None).is_err());
        let (_, mut request) = res(&book, "rid", 10, vec![candidate("a", 4)]).unwrap();
        request.mark_dispatched().unwrap();
        book.snapshot();
        request.finish("http_500_complete_response").unwrap();
        let rows: Vec<Value> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(rows.len(), 4);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row["sequence"], i + 1);
        }
        assert_eq!(rows[0]["entry"]["cost"], "6");
        assert_eq!(rows[3]["entry"]["status"], "finished");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn trace_io_failure_blocks_current_and_future_reservations() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        book.state.lock().trace = Some(BufWriter::new(
            OpenOptions::new().write(true).open("/dev/full").unwrap(),
        ));
        assert!(res(&book, "r1", 10, vec![candidate("a", 0)]).is_err());
        assert_eq!(cost(&book, "a"), 0);
        assert_eq!(book.state.lock().entries.len(), 0);
        assert!(res(&book, "r2", 10, vec![candidate("a", 0)]).is_err());
        assert_eq!(book.snapshot()["admission_blocked"], true);
    }

    // ---- cost-function forecast wait ----

    /// A usable (P, step) fit by hand: a pass takes 10 ms + 0.05 ms per new
    /// token, the launch interval is 0.02 s + 0.9 x the GPU s of the pass
    /// before, idle delay 0.03 s, no receipt line (0.044 s).
    fn forecast_fit() -> crate::prefill_forecast::FitBundle {
        let gm = crate::forecast::GpuModel {
            small: Some(vec![10.0, 0.05, 0.0, 0.0, 0.0]),
            large: Some(vec![10.0, 0.05, 0.0, 0.0, 0.0]),
            floor_ms: 1.0,
        };
        crate::prefill_forecast::FitBundle {
            step: 0.0,
            gpu: gm,
            gpu_sources: crate::forecast::GpuSources {
                small: Some("own".into()),
                large: Some("own".into()),
            },
            launch: crate::forecast::LaunchFit {
                coef: vec![0.02, 0.9, 0.0, 0.0, 0.0, 0.0],
                resid_q: None,
                source: crate::forecast::LaunchSource::Constant,
            },
            constants: crate::forecast::Constants {
                idle_delay_s: 0.03,
                phi_s: 0.0,
                boot_s: 0.0,
                lag_none_s: 0.0,
                lag_next: [0.0, 0.0],
            },
            arrivals: None,
            counts: Default::default(),
        }
    }

    /// A the forecast wait context. Per P: the snapshot JSON (None: no snapshot),
    /// its age at the poll, whether its fit has a GPU model, and the target
    /// (prefix, host). The forecast clock starts 100 s ago.
    #[allow(clippy::type_complexity)]
    fn forecast_context(
        rows: &[(&str, Option<Value>, std::time::Duration, bool, i64, i64)],
        input: i64,
    ) -> crate::prefill_forecast::ForecastContext {
        let path = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/forecast/bound-params.json"
        ));
        let cfg = crate::prefill_forecast::ForecastConfig::load(
            path,
            crate::prefill_forecast::DEFAULT_BOUND_MODE,
            None,
            0.0,
            None,
        )
        .unwrap();
        let anchor = Instant::now()
            .checked_sub(std::time::Duration::from_secs(100))
            .unwrap();
        let candidates = rows
            .iter()
            .map(|(url, raw, age, usable, prefix, host)| {
                let snapshot = raw.as_ref().map(|raw| {
                    let sent = Instant::now().checked_sub(*age).unwrap();
                    Arc::new(RoutingSnapshot {
                        view: crate::prefill_snapshot::SnapshotView::parse(raw, 1, None),
                        poll_sent: sent,
                        poll_received: sent,
                        forecast: Some(Arc::new(crate::prefill_forecast::ForecastView::parse(raw))),
                    })
                });
                let fit = forecast_fit();
                let prep = crate::prefill_forecast::ForecastPrepared {
                    snapshot,
                    push: None,
                    params: usable.then(|| fit.params(None, 0.35 * 2_067_584.0)),
                    fit: Some(Arc::new(fit)),
                    target: crate::forecast::Target {
                        rid: "x".into(),
                        input,
                        prefix: *prefix,
                        host: *host,
                    },
                    host_rule: "split",
                    session_gap_s: None,
                };
                (url.to_string(), prep)
            })
            .collect();
        crate::prefill_forecast::ForecastContext {
            cfg: Arc::new(cfg),
            anchor,
            unix_at_anchor: 1_000.0,
            candidates,
        }
    }

    fn forecast_params(lambda: f64) -> crate::scheduler::placement_cost::PlacementCostParams {
        crate::scheduler::placement_cost::PlacementCostParams {
            copy_price: lambda,
            wait: crate::scheduler::placement_cost::PrefillWait::Forecast,
            ..Default::default()
        }
    }

    /// forecast wait. Owner a caches 40,000 of 44,000 tokens behind five
    /// 65,536-token rows (40 passes of 0.42 s): pred above 15 s. b is idle
    /// and pulls the 40,000 tokens (0.49 s, mostly exposed): pred about
    /// 0.3 s. At lambda 0 the gain beats the bound margin and the request
    /// moves to b; the trace has the forecast keys. The next decision sees r1
    /// (reserved on b, not in b's snapshot) as a router row of b. After a
    /// progress poll received before the snapshot lists r1, r1 is left out.
    #[test]
    fn cost_forecast_moves_through_the_gate_and_prices_unseen_router_rows() {
        let before = Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .unwrap();
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let ctx = forecast_context(
            &[
                (
                    "a",
                    Some(routing_snapshot(5, 65_536)),
                    std::time::Duration::ZERO,
                    true,
                    40_000,
                    0,
                ),
                (
                    "b",
                    Some(routing_snapshot(0, 0)),
                    std::time::Duration::ZERO,
                    true,
                    40_000,
                    40_000,
                ),
            ],
            44_000,
        );
        let candidates = || vec![candidate("a", 40_000), pulling("b", 0, 40_000)];
        let (selected, mut r1, decision) = book
            .reserve_cost_forecast(
                44_000,
                Some("r1".into()),
                candidates(),
                forecast_params(0.0),
                Some(&ctx),
            )
            .unwrap();
        assert_eq!(selected.worker_url, "b", "{decision}");
        assert_eq!(decision["wait"], "forecast");
        assert_eq!(decision["detour"], true);
        assert_eq!(decision["forecast_protected"], false);
        assert_eq!(decision["forecast_protect_reason"], Value::Null);
        assert_eq!(decision["forecast_forecasts"], 2);
        assert_eq!(decision["forecast_bound_level"], "0.8");
        assert_eq!(decision["forecast_bound_mode"], "snap+prod");
        assert_eq!(decision["move_margin"], 1.0);
        assert_eq!(decision["wait_fallbacks"], 0);
        let gain = decision["forecast_gain_s"].as_f64().unwrap();
        let margin = decision["forecast_margin_s"].as_f64().unwrap();
        assert!(gain > margin && margin > 0.0, "{gain} {margin}");
        let (a, b) = (&decision["candidates"][0], &decision["candidates"][1]);
        assert_eq!(a["wait_source"], "forecast");
        assert_eq!(a["forecast_bucket"], "busy");
        assert_eq!(a["forecast_n_queue"], 5);
        assert!(a["forecast_pred_s"].as_f64().unwrap() > 15.0, "{a}");
        assert_eq!(b["forecast_idle"], true);
        assert_eq!(b["forecast_bucket"], "idle");
        assert_eq!(b["forecast_router_rows"], 0);
        let (pred_b, wait_b) = (
            b["forecast_pred_s"].as_f64().unwrap(),
            b["forecast_wait_s"].as_f64().unwrap(),
        );
        assert!(pred_b < 1.0 && wait_b < pred_b, "{b}");
        // C_b = R * (pred_s + max(0, pull_s - wait_s)) at lambda 0.
        let rate = 13_500.0;
        let c_b = rate * wait_b + rate * (pred_b - wait_b) + rate * (0.49 - wait_b).max(0.0);
        assert!((b["cost"].as_f64().unwrap() - c_b).abs() < 1e-6, "{b}");
        assert_eq!(decision["forecast_chosen"]["pred_s"], b["forecast_pred_s"]);
        assert_eq!(decision["forecast_chosen"]["target_prefix"], 40_000);
        assert!(decision["forecast_decision_s"].as_f64().unwrap() >= 100.0);
        // The state push is off: no push keys.
        assert!(
            a.get("forecast_state_source").is_none() && a.get("forecast_push_fallback").is_none()
        );
        assert!(decision.get("forecast_push_port").is_none());
        assert!(decision["forecast_chosen"].get("state_source").is_none());

        let (_, r2, decision) = book
            .reserve_cost_forecast(
                44_000,
                Some("r2".into()),
                candidates(),
                forecast_params(0.0),
                Some(&ctx),
            )
            .unwrap();
        let b = &decision["candidates"][1];
        assert_eq!(b["forecast_router_rows"], 1, "{b}");
        assert_eq!(b["forecast_n_dispatched"], 1);
        assert_eq!(b["forecast_idle"], false);
        drop(r2);

        // r1 launched before the snapshot time: the snapshot does not show
        // it, so it is done, not a router row.
        r1.mark_dispatched().unwrap();
        book.apply_progress_received(
            "b",
            progress(1, &[("r1", 4_000)]),
            json!({}),
            before,
            before,
        )
        .unwrap();
        let (_, r3, decision) = book
            .reserve_cost_forecast(
                44_000,
                Some("r3".into()),
                candidates(),
                forecast_params(0.0),
                Some(&ctx),
            )
            .unwrap();
        assert_eq!(decision["candidates"][1]["forecast_router_rows"], 0);
        drop((r1, r3));
    }

    /// Fallbacks: a P without a snapshot (no_view), with an old view
    /// (view_too_old) or without a GPU model (no_fit) keeps the token-count terms
    /// and never wins a move; the trace counts them by reason. An owner
    /// without a forecast is kept (owner_no_forecast); without a context
    /// every P falls back (no_context).
    #[test]
    fn cost_forecast_fallbacks_never_win_a_move_and_are_counted() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let zero = std::time::Duration::ZERO;
        let ctx = forecast_context(
            &[
                (
                    "a",
                    Some(routing_snapshot(5, 65_536)),
                    zero,
                    true,
                    40_000,
                    0,
                ),
                ("b", None, zero, true, 40_000, 40_000),
                (
                    "c",
                    Some(routing_snapshot(0, 0)),
                    std::time::Duration::from_secs(2),
                    true,
                    40_000,
                    40_000,
                ),
            ],
            44_000,
        );
        let candidates = vec![
            candidate("a", 40_000),
            pulling("b", 0, 40_000),
            pulling("c", 0, 40_000),
        ];
        let (selected, r1, decision) = book
            .reserve_cost_forecast(
                44_000,
                Some("f1".into()),
                candidates,
                forecast_params(0.0),
                Some(&ctx),
            )
            .unwrap();
        assert_eq!(selected.worker_url, "a", "{decision}");
        let rows = &decision["candidates"];
        assert_eq!(rows[1]["wait_fallback_reason"], "no_view");
        assert_eq!(rows[1]["wait_source"], "tokens_fallback");
        assert_eq!(rows[1]["forecast_pred_s"], Value::Null);
        assert_eq!(rows[2]["wait_fallback_reason"], "view_too_old");
        assert!(rows[2]["view_age_s"].as_f64().unwrap() > 1.5);
        assert_eq!(decision["wait_fallbacks"], 2);
        assert_eq!(decision["wait_fallback_counts"]["no_view"], 1);
        assert_eq!(decision["forecast_forecasts"], 1);
        assert_eq!(
            decision["forecast_protect_reason"],
            Value::Null,
            "only a has a forecast"
        );
        drop(r1);

        let ctx = forecast_context(
            &[
                (
                    "a",
                    Some(routing_snapshot(5, 65_536)),
                    zero,
                    false,
                    40_000,
                    0,
                ),
                (
                    "b",
                    Some(routing_snapshot(0, 0)),
                    zero,
                    true,
                    40_000,
                    40_000,
                ),
            ],
            44_000,
        );
        let candidates = || vec![candidate("a", 40_000), pulling("b", 0, 40_000)];
        let (selected, r2, decision) = book
            .reserve_cost_forecast(
                44_000,
                Some("f2".into()),
                candidates(),
                forecast_params(0.0),
                Some(&ctx),
            )
            .unwrap();
        assert_eq!(selected.worker_url, "a");
        assert_eq!(decision["candidates"][0]["wait_fallback_reason"], "no_fit");
        assert_eq!(decision["best"], "b");
        assert_eq!(decision["forecast_protect_reason"], "owner_no_forecast");
        assert_eq!(decision["forecast_protected"], true);
        // Counts accumulate over decisions.
        assert_eq!(decision["wait_fallback_counts"]["no_fit"], 1);
        assert_eq!(decision["wait_fallback_counts"]["no_view"], 1);
        drop(r2);

        let (selected, r3, decision) = book
            .reserve_cost_forecast(
                44_000,
                Some("f3".into()),
                candidates(),
                forecast_params(0.0),
                None,
            )
            .unwrap();
        assert_eq!(selected.worker_url, "a");
        assert_eq!(decision["wait_fallback_counts"]["no_context"], 2);
        assert_eq!(decision["forecast_decision_s"], Value::Null);
        drop(r3);
    }

    /// forecast wait and the engine policy: a P whose snapshot has another pass
    /// budget than the bound tables' fit has no forecast
    /// (policy_not_bound_fit). A snapshot that the engine marks stale with a
    /// known age is used (the view-age limit decides); the flag is in
    /// forecast_degraded.
    #[test]
    fn cost_forecast_needs_the_bound_fit_policy_and_ignores_the_engine_stale_flag() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let zero = std::time::Duration::ZERO;
        let mut other = routing_snapshot(0, 0);
        other["policy"]["chunk_tokens"] = json!(16_384);
        let mut stale = routing_snapshot(0, 0);
        stale["stale"] = json!(true);
        let ctx = forecast_context(
            &[
                (
                    "a",
                    Some(routing_snapshot(5, 65_536)),
                    zero,
                    true,
                    40_000,
                    0,
                ),
                ("b", Some(other), zero, true, 40_000, 40_000),
                ("c", Some(stale), zero, true, 40_000, 40_000),
            ],
            44_000,
        );
        let candidates = vec![
            candidate("a", 40_000),
            pulling("b", 0, 40_000),
            pulling("c", 0, 40_000),
        ];
        let (_, r, decision) = book
            .reserve_cost_forecast(
                44_000,
                Some("q1".into()),
                candidates,
                forecast_params(0.0),
                Some(&ctx),
            )
            .unwrap();
        let rows = &decision["candidates"];
        assert_eq!(rows[0]["wait_source"], "forecast", "{decision}");
        assert_eq!(rows[0]["forecast_degraded"], json!([]));
        assert_eq!(rows[0]["forecast_pull_rows"], 0);
        assert_eq!(rows[1]["wait_fallback_reason"], "policy_not_bound_fit");
        assert_eq!(rows[1]["forecast_pred_s"], Value::Null);
        assert_eq!(rows[2]["wait_source"], "forecast", "{decision}");
        assert_eq!(rows[2]["forecast_degraded"], json!(["engine_stale"]));
        assert_eq!(decision["wait_fallback_counts"]["policy_not_bound_fit"], 1);
        drop(r);
    }

    /// A the forecast wait context with the state push on (push table
    /// pushL+prodL): P `a` (polled: five 65,536-token rows) with the pushed
    /// state `a_push` = (snapshot, epoch, its age at the context), P `b`
    /// (polled: idle) without one.
    fn forecast_push_context(
        a_push: Option<(Value, &str, f64)>,
    ) -> crate::prefill_forecast::ForecastContext {
        let zero = std::time::Duration::ZERO;
        let mut ctx = forecast_context(
            &[
                (
                    "a",
                    Some(routing_snapshot(5, 65_536)),
                    zero,
                    true,
                    40_000,
                    0,
                ),
                (
                    "b",
                    Some(routing_snapshot(0, 0)),
                    zero,
                    true,
                    40_000,
                    40_000,
                ),
            ],
            44_000,
        );
        let cfg = (*ctx.cfg)
            .clone()
            .with_push(5600, Some("pushL+prodL"), None)
            .unwrap();
        ctx.cfg = Arc::new(cfg);
        if let Some((raw, epoch, age)) = a_push {
            let t0_s = crate::prefill_forecast::secs_since(ctx.anchor, Instant::now()) - age;
            let state = crate::prefill_state_push::PushState {
                epoch: epoch.to_string(),
                seq: 9,
                trigger: "launch".into(),
                snapshot_source: Some("push_build".into()),
                view: Arc::new(crate::prefill_forecast::ForecastView::parse(&raw)),
                t0_s,
                received_s: t0_s + 0.001,
                offset_s: -5.0,
                superseded_seq: None,
            };
            ctx.candidates.get_mut("a").unwrap().push = Some(Arc::new(state));
        }
        ctx
    }

    /// forecast wait with the state push. P `a` is busy in its polled snapshot
    /// and idle in its pushed state. A fresh pushed state of the polled
    /// epoch is the input (source push, the push table's bound); an old
    /// one, one of another epoch, an unusable one or none keeps the poll
    /// and names the reason. The decision counts the sources and the
    /// reasons over decisions.
    #[test]
    fn cost_forecast_uses_a_fresh_verified_push_and_falls_back_to_the_poll() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let idle = routing_snapshot(0, 0);
        let mut truncated = idle.clone();
        truncated["truncated"] = json!(true);
        let cases = [
            (Some((idle.clone(), "routing-1", 0.01)), None),
            (Some((idle.clone(), "routing-1", 2.0)), Some("push_too_old")),
            (Some((idle, "other", 0.01)), Some("push_unverified")),
            (Some((truncated, "routing-1", 0.01)), Some("push_unusable")),
            (None, Some("no_push")),
        ];
        let candidates = || vec![candidate("a", 40_000), pulling("b", 0, 40_000)];
        let mut last = Value::Null;
        for (i, (push, fallback)) in cases.into_iter().enumerate() {
            let has_push = push.is_some();
            let ctx = forecast_push_context(push);
            let (_, r, decision) = book
                .reserve_cost_forecast(
                    44_000,
                    Some(format!("p{i}")),
                    candidates(),
                    forecast_params(0.0),
                    Some(&ctx),
                )
                .unwrap();
            let (a, b) = (&decision["candidates"][0], &decision["candidates"][1]);
            assert_eq!(a["forecast_push_fallback"], json!(fallback), "{i}: {a}");
            assert_eq!(b["forecast_push_fallback"], "no_push");
            assert_eq!(b["forecast_state_source"], "poll");
            let pushed = fallback.is_none();
            assert_eq!(
                a["forecast_state_source"],
                if pushed { "push" } else { "poll" }
            );
            // The pushed state is idle; the polled one has five rows.
            assert_eq!(
                a["forecast_n_queue"],
                if pushed { 0 } else { 5 },
                "{i}: {a}"
            );
            assert_eq!(
                a["forecast_push_seq"],
                if has_push { json!(9) } else { Value::Null }
            );
            if pushed {
                assert_eq!(a["view_key"], json!(["routing-1", 9]));
                let age = a["view_age_s"].as_f64().unwrap();
                assert!((0.01..0.5).contains(&age), "{age}");
                assert_eq!(a["forecast_bucket"], "idle");
                let view = crate::forecast::BucketView {
                    pred_s: a["forecast_pred_s"].as_f64(),
                    idle: true,
                    wait_s: None,
                    n_inflight: 0,
                    partial: false,
                    n_queue: 0,
                };
                let push = ctx.cfg.push.as_ref().unwrap();
                let want = crate::forecast::error_bound(&view, &push.table, &push.level).unwrap();
                assert_eq!(a["forecast_bound_s"].as_f64(), want);
                let poll =
                    crate::forecast::error_bound(&view, &ctx.cfg.table, &ctx.cfg.level).unwrap();
                assert_ne!(want, poll, "the push table is not the poll table");
            }
            if fallback == Some("push_too_old") {
                assert!(a["forecast_push_age_s"].as_f64().unwrap() >= 2.0);
            }
            if fallback == Some("push_unusable") {
                assert_eq!(a["forecast_push_unusable"], "truncated");
            }
            drop(r);
            last = decision;
        }
        assert_eq!(last["forecast_push_port"], 5600);
        assert_eq!(last["forecast_push_bound_mode"], "pushL+prodL");
        assert_eq!(last["forecast_push_bound_level"], "0.8");
        assert_eq!(last["forecast_chosen"]["state_source"], "poll");
        let sources = &last["forecast_state_source_counts"];
        assert_eq!((&sources["push"], &sources["poll"]), (&json!(1), &json!(9)));
        let reasons = &last["forecast_push_fallback_counts"];
        assert_eq!(reasons["no_push"], 6, "{reasons}");
        for reason in ["push_too_old", "push_unverified", "push_unusable"] {
            assert_eq!(reasons[reason], 1, "{reasons}");
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn dispatch_trace_failure_keeps_undispatched_reservation_rollbackable() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let (_, mut request) = res(&book, "r", 10, vec![candidate("a", 0)]).unwrap();
        book.state.lock().trace = Some(BufWriter::new(
            OpenOptions::new().write(true).open("/dev/full").unwrap(),
        ));
        assert!(request.mark_dispatched().is_err());
        assert_eq!(request.debuginfo()["status"], "reserved");
        drop(request);
        assert_eq!(cost(&book, "a"), 0);
        assert!(book.snapshot()["unknown_entries"]
            .as_array()
            .unwrap()
            .is_empty());
    }

    /// The work book counters of one worker: cost, pending uncached, pending
    /// restore and unacknowledged tokens, active requests.
    fn counters(book: &PrefillWorkBook, url: &str) -> (u128, u128, u128, u64) {
        let st = book.state.lock();
        let w = &st.workers[url];
        (
            w.cost,
            w.pending_uncached_tokens,
            w.pending_unacknowledged_tokens,
            w.active_requests,
        )
    }

    // ---- --global-queue (global queue) ----

    /// The lead rule of one P: depth = floor(C + R x T_react) + 1 and lead =
    /// C / R + T_react, so room (queued < depth) is W = queued / R <= lead.
    /// R 0 leaves lead = T_react and depth C + 1; a non-finite T_react
    /// counts as 0.
    #[test]
    fn queue_lead_row_depth_and_lead() {
        let row = QueueLeadRow::new(8192, "server_info", 0.5, "measured", 3, 13_500.0);
        assert_eq!(row.depth, 8192 + 6750 + 1);
        assert!((row.lead_s - (8192.0 / 13_500.0 + 0.5)).abs() < 1e-12);
        assert_eq!(
            (row.chunk_source, row.react_source, row.react_samples),
            ("server_info", "measured", 3)
        );
        let idle = QueueLeadRow::new(8192, "default", 0.5, "fallback", 0, 0.0);
        assert_eq!((idle.depth, idle.lead_s), (8193, 0.5));
        let nan = QueueLeadRow::new(4096, "default", f64::NAN, "measured", 1, 13_500.0);
        assert_eq!((nan.depth, nan.react_s), (4097, 0.0));
        assert!((nan.lead_s - 4096.0 / 13_500.0).abs() < 1e-12);
    }

    /// T_react of a P: the fallback before the first sample, then the
    /// time-weighted mean (each sample decays as exp(-age / tau)).
    #[test]
    fn queue_lead_book_is_a_time_weighted_mean() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        book.enable_queue_lead(120.0, 0.6);
        let mut state = book.state.lock();
        let lead = state.queue_lead.as_mut().unwrap();
        assert_eq!(lead.react("a"), (0.6, "fallback", 0));
        let t0 = Instant::now();
        assert_eq!(lead.observe("a", t0, 1.0), 1.0);
        let mean = lead.observe("a", t0 + std::time::Duration::from_secs(120), 0.5);
        let decay = (-1.0f64).exp();
        assert!(
            (mean - (decay + 0.5) / (decay + 1.0)).abs() < 1e-9,
            "{mean}"
        );
        let (react_s, source, samples) = lead.react("a");
        assert_eq!((source, samples), ("measured", 2));
        assert!((react_s - mean).abs() < 1e-12);
        assert_eq!(lead.react("b").1, "fallback");
    }

    fn queue_auto_args<'a>(
        urls: &'a [String],
        depth: &'a [u64],
        chunks: &'a [(u64, &'static str)],
    ) -> QueueStepArgs<'a> {
        QueueStepArgs {
            auto: Some(QueueAutoStep { chunks }),
            ..queue_args(urls, depth)
        }
    }

    /// global queue release rule on one P with C 8192, R 13,500 and the
    /// fallback T_react 0.5 s: depth = 8192 + 6750 + 1 = 14,943 (the given
    /// depth is ignored). A launched request with 23,135 left gives queued
    /// 14,943 > lead x R: hold. With 23,134 left, queued 14,942 (W equals
    /// lead up to 1/R): release, with the lead rule in the `queue` object. No
    /// sample (P had queued work) and no shadow data (no host_room).
    #[test]
    fn queue_auto_releases_iff_the_wait_fits_the_lead() {
        use crate::global_queue::QueueRelease;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("queue-a.jsonl");
        let book = PrefillWorkBook::new(Some(&path), "test", None).unwrap();
        book.enable_queue_lead(120.0, 0.5);
        let mut on_a = uncached(&book, "on_a", "a", 30_000);
        on_a.mark_dispatched().unwrap();
        book.apply_progress(
            "a",
            progress(1, &[("on_a", 23_135)]),
            json!({}),
            Instant::now(),
        )
        .unwrap();
        let urls = vec!["a".to_string()];
        let depth = [1u64];
        let chunks = [(8192u64, "default")];
        let cands = vec![candidate("a", 0)];
        let held = [queue_held("r", 5000, &cands, None, false)];
        let step = book
            .queue_step(
                cost_params(9.2),
                &held,
                &queue_auto_args(&urls, &depth, &chunks),
            )
            .unwrap();
        assert!(step.release.is_none(), "queued 14,943 > 14,942: {step:?}");
        book.apply_progress(
            "a",
            progress(2, &[("on_a", 23_134)]),
            json!({}),
            Instant::now(),
        )
        .unwrap();
        let step = book
            .queue_step(
                cost_params(9.2),
                &held,
                &queue_auto_args(&urls, &depth, &chunks),
            )
            .unwrap();
        let (_, result) = step.release.expect("queued 14,942: room");
        let released = result.unwrap();
        assert_eq!(released.release, QueueRelease::Room);
        let g = &released.record;
        assert_eq!(
            (g["rule"].clone(), g["class"].clone()),
            (json!("auto"), json!("free"))
        );
        assert_eq!(g["depth_tokens"], 14_943);
        assert_eq!(g["queued_tokens"], "14942");
        assert_eq!(
            (g["chunk_tokens"].clone(), g["chunk_source"].clone()),
            (json!(8192), json!("default"))
        );
        assert_eq!(
            (g["react_s"].clone(), g["react_source"].clone()),
            (json!(0.5), json!("fallback"))
        );
        let lead = g["lead_s"].as_f64().unwrap();
        assert!((lead - (8192.0 / 13_500.0 + 0.5)).abs() < 1e-12, "{g}");
        let wait = g["wait_s"].as_f64().unwrap();
        assert!(
            (wait - 14_942.0 / 13_500.0).abs() < 1e-12 && wait <= lead + 1e-9,
            "{g}"
        );
        assert_eq!(g["lead_sample"], false);
        let stats = book.queue_lead_rows(&urls);
        assert_eq!(stats["by_p"]["a"]["depth_tokens"], 14_943);
        assert_eq!(
            (stats["marked"].clone(), stats["unmarked"].clone()),
            (json!(0), json!(1))
        );
        let events = trace_lines(&path);
        let reserve = events
            .iter()
            .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == "r")
            .unwrap();
        assert_eq!(reserve["global_queue"]["rule"], "auto");
        drop((on_a, released));
    }

    /// T_react samples come only from releases to an idle P with a fresh
    /// view: r1 (P empty) is sampled at its first sighting (sent 300 ms after
    /// the release); r2 (P running r1) is not.
    #[test]
    fn queue_auto_samples_t_react_only_at_an_empty_queue() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("queue-sample.jsonl");
        let book = PrefillWorkBook::new(Some(&path), "test", None).unwrap();
        book.enable_queue_lead(120.0, 0.5);
        let t0 = Instant::now();
        book.apply_progress("a", progress(1, &[]), json!({}), t0)
            .unwrap();
        let urls = vec!["a".to_string()];
        let depth = [1u64];
        let chunks = [(8192u64, "default")];
        let cands = vec![candidate("a", 0)];
        let release = |rid: &'static str| {
            let held = [queue_held(rid, 20_000, &cands, None, true)];
            let step = book
                .queue_step(
                    cost_params(9.2),
                    &held,
                    &queue_auto_args(&urls, &depth, &chunks),
                )
                .unwrap();
            step.release.expect("room").1.unwrap()
        };
        let mut r1 = release("r1");
        assert_eq!(r1.record["lead_sample"], true);
        r1.reservation.mark_dispatched().unwrap();
        let seen = Instant::now() + std::time::Duration::from_millis(300);
        book.apply_progress("a", progress(2, &[("r1", 20_000)]), json!({}), seen)
            .unwrap();
        let (react_s, source, samples) = book.state.lock().queue_lead.as_ref().unwrap().react("a");
        assert_eq!((source, samples), ("measured", 1));
        assert!((0.3..0.5).contains(&react_s), "{react_s}");
        // r1 launched 20,000: queued 11,808 > 0, so r2 is not sampled.
        let mut r2 = release("r2");
        assert_eq!(r2.record["lead_sample"], false);
        r2.reservation.mark_dispatched().unwrap();
        let later = seen + std::time::Duration::from_millis(100);
        book.apply_progress(
            "a",
            progress(3, &[("r1", 20_000), ("r2", 20_000)]),
            json!({}),
            later,
        )
        .unwrap();
        assert_eq!(
            book.state.lock().queue_lead.as_ref().unwrap().react("a").2,
            1
        );
        let samples: Vec<Value> = trace_lines(&path)
            .into_iter()
            .filter(|e| e["event"] == "queue_lead_sample")
            .collect();
        assert_eq!(samples.len(), 1, "{samples:?}");
        assert_eq!(samples[0]["rid"], "r1");
        assert_eq!(
            (
                samples[0]["restore_tokens"].clone(),
                samples[0]["pull_tokens"].clone()
            ),
            (json!(0), json!(0))
        );
        let stats = book.queue_lead_rows(&urls);
        assert_eq!(
            (
                stats["marked"].clone(),
                stats["unmarked"].clone(),
                stats["samples"].clone()
            ),
            (json!(1), json!(1), json!(1))
        );
        drop((r1, r2));
    }

    /// The walk and the release score the step pool. r is held with
    /// candidates over {a, b} (a holds 9,000 of it; b would pull them from
    /// a). a leaves the pool (breaker) and c joins: the release goes to a P
    /// of {b, c} with no walk mismatch, b's pull from a is dropped, c is
    /// priced cold, and the decision has no row for a.
    #[test]
    fn queue_release_uses_the_step_pool() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("queue-pool.jsonl");
        let book = PrefillWorkBook::new(Some(&path), "test", None).unwrap();
        let from_a = Candidate {
            hint: Some(json!({"source_worker_id": "a", "expected_hit_tokens": 9000})),
            ..pulling("b", 0, 9000)
        };
        let cands = vec![candidate("a", 9000), from_a];
        let held = [queue_held("r", 12_000, &cands, Some("a"), false)];
        let urls = vec!["b".to_string(), "c".to_string()];
        let depth = [1u64 << 40, 1 << 40];
        let step = book
            .queue_step(cost_params(9.2), &held, &queue_args(&urls, &depth))
            .unwrap();
        let (_, result) = step.release.expect("room on b and c");
        let released = result.unwrap();
        assert_ne!(released.candidate.worker_url, "a");
        assert!(released.candidate.hint.is_none(), "the pull source left");
        assert_eq!(released.decision["pulled_tokens"], 0);
        let rows: Vec<&str> = released.decision["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["worker_url"].as_str().unwrap())
            .collect();
        assert_eq!(rows, ["b", "c"]);
        let events = trace_lines(&path);
        let reserve = events
            .iter()
            .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == "r")
            .unwrap();
        assert!(
            reserve["global_queue"].get("walk_mismatch").is_none(),
            "{reserve}"
        );
        assert_eq!(
            reserve["global_queue"]["p_release"].as_str(),
            Some(released.candidate.worker_url.as_str())
        );
        drop(released);
    }

    /// `--symmetric-pd-pull-from-any-holder`: a held request whose pull source d
    /// is outside the step pool keeps its hint while d is among the step's
    /// `sources` (a registered worker of any role) and loses it once d is
    /// not (d was removed from the registry, or its breaker opened while the
    /// breaker is on).
    #[test]
    fn queue_release_keeps_a_pull_source_listed_in_sources() {
        let from_d = Candidate {
            hint: Some(json!({"source_worker_id": "d", "expected_hit_tokens": 9000})),
            ..pulling("b", 0, 9000)
        };
        let cands = vec![from_d];
        let urls = vec!["b".to_string()];
        let depth = [1u64 << 40];
        for (sources, kept) in [(vec!["d".to_string()], true), (Vec::new(), false)] {
            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path().join("queue-sources.jsonl");
            let book = PrefillWorkBook::new(Some(&path), "test", None).unwrap();
            let held = [queue_held("r", 12_000, &cands, Some("b"), false)];
            let args = QueueStepArgs {
                sources: &sources,
                ..queue_args(&urls, &depth)
            };
            let step = book.queue_step(cost_params(9.2), &held, &args).unwrap();
            let (_, result) = step.release.expect("room on b");
            let released = result.unwrap();
            assert_eq!(released.candidate.worker_url, "b");
            assert_eq!(
                released.candidate.hint.is_some(),
                kept,
                "{:?}",
                released.candidate
            );
            let pulled = released.decision["pulled_tokens"].as_u64().unwrap();
            assert_eq!(pulled > 0, kept, "{}", released.decision);
            drop(released);
        }
    }

    /// A released reservation that no admission takes (`send_failed`: the
    /// client left after the step) rolls back when it drops: the P's work book
    /// returns to its value before the release.
    #[test]
    fn queue_unsent_release_rolls_back() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let urls = vec!["a".to_string()];
        let depth = [1u64 << 40];
        let cands = vec![candidate("a", 0)];
        let held = [queue_held("r", 7000, &cands, None, true)];
        let step = book
            .queue_step(cost_params(9.2), &held, &queue_args(&urls, &depth))
            .unwrap();
        let released = step.release.unwrap().1.unwrap();
        let pending = |b: &PrefillWorkBook| b.state.lock().workers["a"].pending_uncached_tokens;
        assert_eq!(pending(&book), 7000);
        drop(released);
        assert_eq!(pending(&book), 0, "rolled back");
        assert_eq!(book.state.lock().workers["a"].active_requests, 0);
    }

    /// Run the unknown expiry of every worker with `at` as the sweep time.
    fn expire_at(book: &PrefillWorkBook, at: Instant) {
        let mut st = book.state.lock();
        book.expire_unknown(&mut st, None, at);
    }

    fn secs(s: u64) -> std::time::Duration {
        std::time::Duration::from_secs(s)
    }

    fn unknown_since(book: &PrefillWorkBook, r: &Reservation) -> Instant {
        book.state.lock().entries[&r.id].unknown_since.unwrap()
    }

    fn trace_rows(path: &Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    // ---- global waiting queue steps ----

    fn queue_args<'a>(urls: &'a [String], depth: &'a [u64]) -> QueueStepArgs<'a> {
        QueueStepArgs {
            urls,
            depth,
            stale_after: std::time::Duration::from_secs(1),
            max_wait: std::time::Duration::from_secs(30),
            pass_tokens: 8192,
            auto: None,
            now: Instant::now(),
            pass_id: 7,
            trigger: "poll",
            triggers: "poll",
            forecast: None,
            lead_s: 0.0,
            sources: &[],
        }
    }

    fn queue_held<'a>(
        rid: &'a str,
        input_tokens: u64,
        candidates: &'a [Candidate],
        p_arrival: Option<&'a str>,
        arrival_pass: bool,
    ) -> QueueHeld<'a> {
        QueueHeld {
            input_tokens,
            rid: Some(rid),
            candidates,
            arrived: Instant::now(),
            arrived_unix_ms: 1,
            session: None,
            session_label: None,
            p_arrival,
            arrival_pass,
        }
    }

    fn trace_lines(path: &Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    /// Apply a progress reply of `url`, sent at `at`, that lists `rows` (the
    /// next engine sequence of that worker); `at` must be newer than its
    /// last applied reply.
    fn reply(book: &PrefillWorkBook, url: &str, at: Instant, rows: &[(&str, u64)]) {
        let sequence = {
            let st = book.state.lock();
            st.progress.get(url).map_or(1, |c| c.sequence + 1)
        };
        book.apply_progress(url, progress(sequence, rows), json!({}), at)
            .unwrap();
        assert_eq!(book.state.lock().progress[url].sent_at, at, "applied");
    }

    /// An unknown entry that its P still lists does not expire, and the
    /// listed work updates it. It expires on the first reply of its P sent at
    /// least the expiry after the last reply that listed it; the release
    /// subtracts its current work.
    #[test]
    fn unknown_still_listed_holds_until_the_engine_stops_listing_it() {
        let book = PrefillWorkBook::new(None, "test", Some(secs(60))).unwrap();
        let mut r = uncached(&book, "slow", "p", 30_000);
        r.mark_dispatched().unwrap();
        r.mark_unknown("prefill_transport_error: upstream timed out")
            .unwrap();
        let since = unknown_since(&book, &r);
        let at = |s: u64| since + secs(s);
        let status = || book.state.lock().entries[&r.id].status;
        reply(&book, "p", at(50), &[("slow", 22_000)]);
        assert_eq!(counters(&book, "p"), (22_000, 22_000, 0, 1));
        reply(&book, "p", at(100), &[("slow", 9_000)]);
        // The P answers without it from 130 s on: the evidence stays at 100 s.
        reply(&book, "p", at(130), &[]);
        reply(&book, "p", at(159), &[]);
        assert_eq!((status(), cost(&book, "p")), (Status::Unknown, 9_000));
        reply(&book, "p", at(160), &[]);
        assert_eq!(status(), Status::Expired);
        assert_eq!(counters(&book, "p"), (0, 0, 0, 0));
    }

    /// A P without applied progress replies (dead, partitioned, or its polls
    /// time out) keeps its unknown work at any age: the expiry needs a reply
    /// of the entry's own P, and replies of another P do not count. The first
    /// reply of its P sent at least the expiry after the evidence expires it.
    #[test]
    fn unknown_on_a_p_without_replies_does_not_expire() {
        let book = PrefillWorkBook::new(None, "test", Some(secs(60))).unwrap();
        let mut r = uncached(&book, "lost", "p", 8_000);
        r.mark_dispatched().unwrap();
        r.mark_unknown("prefill_transport_error: upstream timed out")
            .unwrap();
        let since = unknown_since(&book, &r);
        let status = || book.state.lock().entries[&r.id].status;
        // No reply of `p` at all.
        expire_at(&book, since + secs(10 * 24 * 3600));
        assert_eq!(book.snapshot()["unknown_requests"], 1);
        reply(&book, "q", since + secs(3600), &[]);
        assert_eq!((status(), cost(&book, "p")), (Status::Unknown, 8_000));
        // `p` answers 30 s after the error, then its polls fail for a day.
        reply(&book, "p", since + secs(30), &[]);
        expire_at(&book, since + secs(24 * 3600));
        assert_eq!(book.snapshot()["unknown_requests"], 1);
        assert_eq!((status(), cost(&book, "p")), (Status::Unknown, 8_000));
        // Its first reply after that expires it.
        reply(&book, "p", since + secs(24 * 3600), &[]);
        assert_eq!(status(), Status::Expired);
        assert_eq!(counters(&book, "p"), (0, 0, 0, 0));
    }

    /// The expiry runs when a reply of the entry's own P is applied (not
    /// another P's) and at each snapshot. The snapshot path decides only when
    /// the deciding reply was applied before the entry became unknown, as with
    /// the reply time in the future here.
    #[test]
    fn unknown_expiry_runs_on_its_workers_progress_and_on_snapshots() {
        let book = PrefillWorkBook::new(None, "test", Some(secs(1))).unwrap();
        let mut a = uncached(&book, "ra", "a", 1_000);
        let mut b = uncached(&book, "rb", "b", 2_000);
        // A reply of `a` sent 10 s from now, applied before `ra` fails.
        let later = Instant::now() + secs(10);
        reply(&book, "a", later, &[]);
        for r in [&mut a, &mut b] {
            r.mark_dispatched().unwrap();
            r.mark_unknown("prefill_transport_error: reset").unwrap();
        }
        let status = |r: &Reservation| book.state.lock().entries[&r.id].status;
        reply(&book, "b", later, &[]);
        assert_eq!(status(&a), Status::Unknown);
        assert_eq!(status(&b), Status::Expired);
        assert_eq!((cost(&book, "a"), cost(&book, "b")), (1_000, 0));
        assert_eq!(book.snapshot()["unknown_requests"], 0);
        assert_eq!(status(&a), Status::Expired);
        assert_eq!(cost(&book, "a"), 0);
    }

    /// Without an expiry (every other constructor, and
    /// `--symmetric-pd-unknown-expire-secs 0`), unknown work is held at any age,
    /// through replies that omit it and snapshots, until a terminal result;
    /// the snapshot has no `unknown_expire_s` and the trace no `expired`.
    #[test]
    fn without_an_expiry_unknown_work_is_held_until_a_terminal_result() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("cost.jsonl");
        let book = PrefillWorkBook::new(Some(&path), "symmetric_pd_cost_function", None).unwrap();
        let mut r = uncached(&book, "lost", "a", 12_000);
        r.mark_dispatched().unwrap();
        r.mark_unknown("prefill_transport_error: upstream timed out")
            .unwrap();
        let since = unknown_since(&book, &r);
        expire_at(&book, since + secs(10 * 24 * 3600));
        reply(&book, "a", since + secs(10 * 24 * 3600), &[]);
        let snapshot = book.snapshot();
        assert_eq!(snapshot["unknown_requests"], 1);
        assert!(snapshot.get("unknown_expire_s").is_none());
        assert!(snapshot.get("expired_rid_listed_again").is_none());
        assert_eq!(counters(&book, "a"), (12_000, 12_000, 12_000, 1));
        r.finish("abort_confirmed").unwrap();
        assert_eq!(counters(&book, "a"), (0, 0, 0, 0));
        assert!(trace_rows(&path).iter().all(|e| e["event"] != "expired"));
    }

    /// After an expiry, the owner's late results change nothing and do not
    /// panic: a finish or another unknown is ignored (no trace event), a
    /// dispatch is refused as for any terminal entry, and a drop is a no-op.
    /// A later reply of the same P that lists an expired RID does not take
    /// the work back; the first listing per entry writes one
    /// `expired_rid_listed_again` event and counts in the snapshot. A listing
    /// by another P is not counted.
    #[test]
    fn late_results_after_an_expiry_are_ignored_and_relistings_are_recorded() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("cost.jsonl");
        let expire = Some(secs(60));
        let book = PrefillWorkBook::new(Some(&path), "symmetric_pd_cost_function", expire).unwrap();
        let lose = |rid: &str, tokens: u64| {
            let mut r = uncached(&book, rid, "a", tokens);
            r.mark_dispatched().unwrap();
            r.mark_unknown("prefill_transport_error: upstream timed out")
                .unwrap();
            let since = unknown_since(&book, &r);
            reply(&book, "a", since + secs(60), &[]);
            r
        };
        let (mut f, mut u, d) = (lose("f", 5_000), lose("u", 6_000), lose("d", 7_000));
        assert_eq!(counters(&book, "a"), (0, 0, 0, 0));
        f.finish("prefill_response_success").unwrap();
        u.mark_unknown("prefill_transport_error: late").unwrap();
        assert!(u.mark_dispatched().is_err());
        drop(d);
        // P `a` lists f and u again (twice); P `b` lists d.
        let t = Instant::now() + secs(3600);
        let listed = [("f", 5_000), ("u", 0), ("d", 7_000)];
        reply(&book, "a", t, &listed[..2]);
        reply(&book, "a", t + secs(1), &listed[..2]);
        reply(&book, "b", t, &listed[2..]);
        assert_eq!(counters(&book, "a"), (0, 0, 0, 0));
        let snapshot = book.snapshot();
        assert_eq!(snapshot["unknown_requests"], 0);
        assert_eq!(snapshot["expired_rid_listed_again"], 2);
        assert_eq!(
            book.state.lock().expired_by_rid["f"],
            (f.id, 2, "a".to_string())
        );
        assert_eq!(book.state.lock().expired_by_rid["d"].1, 0);
        for r in [&f, &u] {
            assert_eq!(r.debuginfo()["status"], "expired");
            assert_eq!(r.debuginfo()["reason"], UNKNOWN_EXPIRED_REASON);
        }
        let rows = trace_rows(&path);
        for rid in ["f", "u", "d"] {
            let events: Vec<&str> = rows
                .iter()
                .filter(|e| e["entry"]["rid"] == rid)
                .filter_map(|e| e["event"].as_str())
                .collect();
            let want = ["reserve", "dispatched", "unknown", "expired"];
            assert_eq!(events, want, "{rid}");
        }
        let again: Vec<&Value> = rows
            .iter()
            .filter(|e| e["event"] == "expired_rid_listed_again")
            .collect();
        assert_eq!(again.len(), 2, "{again:?}");
        assert_eq!(again[0]["rid"], "f");
        assert_eq!(again[1]["rid"], "u");
        assert_eq!(again[0]["reservation_id"], f.id);
        assert_eq!(again[0]["worker_url"], "a");
        assert_eq!(again[0]["remaining_tokens"], 5_000);
        assert_eq!(again[1]["remaining_tokens"], 0);
        for e in rows.iter().filter(|e| e["event"] == "prefill_progress") {
            assert_eq!(e["updates"], json!([]), "no update for expired RIDs");
        }
    }

    // ---- async trace ----

    /// `a` has 30k routed and not launched (queued 30k: no room at depth
    /// 9542); `b` runs a launched 40k request (queued 40k - one pass of 8192
    /// = 31,808: no room). A cold 5k request costs 35k on `a` and 45k on `b`:
    /// it waits. When `b` has 9k left (queued 808: room), `b` costs 14k: the
    /// request goes to `b` (late binding), with the cost-function record.
    /// `--global-queue-room forecast`. a has 50,000 and b 6,000
    /// tokens routed and not launched: no room by tokens at depth 1,000, so
    /// the token rule holds a cold request. With the forecast probes, a is predicted
    /// to drain in more than 15 s (five 65,536-token rows in its snapshot plus
    /// its router row) and b within the lead of 1 s (only its router row): the
    /// request goes to b (release `forecast`). The walk and the reservation
    /// price each P as min(R x drain, work-book backlog) (the forecast may make a
    /// P look emptier, never fuller). A P without a usable fit keeps the token
    /// rule (fail-open): with b's fit missing, nothing is released.
    #[test]
    fn queue_step_forecast_room_takes_the_forecast_drain_and_falls_back_per_p() {
        use crate::global_queue::QueueRelease;
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let mut on_a = uncached(&book, "on_a", "a", 50_000);
        on_a.mark_dispatched().unwrap();
        let mut on_b = uncached(&book, "on_b", "b", 6_000);
        on_b.mark_dispatched().unwrap();
        book.apply_progress("a", progress(1, &[]), json!({}), Instant::now())
            .unwrap();
        book.apply_progress("b", progress(1, &[]), json!({}), Instant::now())
            .unwrap();
        let urls = vec!["a".to_string(), "b".to_string()];
        let depth = [1000u64, 1000];
        let cands = vec![candidate("a", 0), candidate("b", 0)];
        let held = [queue_held("r0", 5000, &cands, None, false)];
        let step = book
            .queue_step(cost_params(18.4), &held, &queue_args(&urls, &depth))
            .unwrap();
        assert!(step.release.is_none(), "no room by tokens: {step:?}");

        let zero = std::time::Duration::ZERO;
        let ctx = forecast_context(
            &[
                ("a", Some(routing_snapshot(5, 65_536)), zero, true, 0, 0),
                ("b", Some(routing_snapshot(0, 0)), zero, true, 0, 0),
            ],
            QUEUE_PROBE_TOKENS as i64,
        );
        let args = QueueStepArgs {
            forecast: Some(&ctx),
            lead_s: 1.0,
            ..queue_args(&urls, &depth)
        };
        let held = [queue_held("r1", 5000, &cands, None, false)];
        let step = book.queue_step(cost_params(18.4), &held, &args).unwrap();
        let r1 = step
            .release
            .expect("b is predicted to drain within the lead")
            .1
            .unwrap();
        assert_eq!(r1.candidate.worker_url, "b");
        assert_eq!(r1.release, QueueRelease::Forecast);
        let g = &r1.record;
        assert_eq!(g["room_rule"], "forecast");
        assert_eq!(g["lead_s"], 1.0);
        let drains = g["drain_s_by_p"].as_array().unwrap();
        assert!(drains[0].as_f64().unwrap() > 15.0, "{g}");
        let d_b = drains[1].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&d_b), "{g}");
        assert_eq!(g["drain_s"], drains[1]);
        assert_eq!(r1.decision["chosen"], "b");
        assert_eq!(r1.decision.get("walk_mismatch"), None);
        let row_b = &r1.decision["candidates"][1];
        assert_eq!(row_b["queue_drain_s"], drains[1]);
        assert_eq!(
            row_b["backlog_tokens"],
            queue_drain_tokens(d_b, 13_500.0).min(6_000).to_string()
        );
        assert_eq!(r1.decision["candidates"][0]["backlog_tokens"], "50000");
        drop(r1);

        let ctx = forecast_context(
            &[
                ("a", Some(routing_snapshot(5, 65_536)), zero, true, 0, 0),
                ("b", Some(routing_snapshot(0, 0)), zero, false, 0, 0),
            ],
            QUEUE_PROBE_TOKENS as i64,
        );
        let args = QueueStepArgs {
            forecast: Some(&ctx),
            lead_s: 1.0,
            ..queue_args(&urls, &depth)
        };
        let held = [queue_held("r2", 5000, &cands, None, false)];
        let step = book.queue_step(cost_params(18.4), &held, &args).unwrap();
        assert!(step.release.is_none(), "{step:?}");
        drop((on_a, on_b));
    }

    #[test]
    fn queue_step_holds_without_room_and_binds_the_p_at_release() {
        use crate::global_queue::QueueRelease;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("queue.jsonl");
        let book = PrefillWorkBook::new(Some(&path), "test", None).unwrap();
        let mut on_a = uncached(&book, "on_a", "a", 30_000);
        on_a.mark_dispatched().unwrap();
        let mut on_b = uncached(&book, "on_b", "b", 40_000);
        on_b.mark_dispatched().unwrap();
        book.apply_progress("a", progress(1, &[]), json!({}), Instant::now())
            .unwrap();
        book.apply_progress(
            "b",
            progress(1, &[("on_b", 40_000)]),
            json!({}),
            Instant::now(),
        )
        .unwrap();
        let urls = vec!["a".to_string(), "b".to_string()];
        let depth = [9542u64, 9542];
        let cands = vec![candidate("a", 0), candidate("b", 0)];
        let held = [queue_held("r", 5000, &cands, None, true)];
        let step = book
            .queue_step(cost_params(9.2), &held, &queue_args(&urls, &depth))
            .unwrap();
        assert!(step.release.is_none(), "expected a hold: {step:?}");
        assert_eq!(step.plain, vec![0]);
        assert_eq!(step.queue_tokens, 5000);
        assert_eq!(
            book.state.lock().entries.len(),
            2,
            "no reservation for a hold"
        );
        let rows = book.queue_rows(&urls, Instant::now(), 8192);
        assert_eq!(
            (rows[0].queued, rows[0].unlaunched, rows[0].launched),
            (30_000, 30_000, 0)
        );
        assert_eq!(
            (rows[1].queued, rows[1].unlaunched, rows[1].launched),
            (31_808, 0, 40_000)
        );
        book.apply_progress(
            "b",
            progress(2, &[("on_b", 9_000)]),
            json!({}),
            Instant::now(),
        )
        .unwrap();
        let held = [queue_held("r", 5000, &cands, Some("a"), false)];
        let step = book
            .queue_step(cost_params(9.2), &held, &queue_args(&urls, &depth))
            .unwrap();
        let Some((index, result)) = step.release else {
            panic!("expected a release");
        };
        assert_eq!(index, 0);
        let released = result.unwrap();
        assert_eq!(released.candidate.worker_url, "b");
        assert_eq!(released.release, QueueRelease::Room);
        assert!(released.late_bound && !released.starved);
        assert_eq!(released.decision["chosen"], "b");
        let events = trace_lines(&path);
        let reserve = events
            .iter()
            .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == "r")
            .unwrap();
        assert_eq!(reserve["selection_policy"], "cost_function");
        assert_eq!(reserve["placement"]["chosen"], "b");
        let g = &reserve["global_queue"];
        assert_eq!(g["mode"], "hrrn");
        assert_eq!(g["held"], true);
        assert_eq!(g["release"], "room");
        assert_eq!(g["p_arrival"], "a");
        assert_eq!(g["p_release"], "b");
        assert_eq!(g["plain_worker"], "b");
        assert_eq!(g["late_bound"], true);
        assert_eq!(g["queued_tokens"], "808");
        assert_eq!(g["unlaunched_tokens"], "0");
        assert_eq!(g["launched_tokens"], "9000");
        assert_eq!(g["depth_tokens"], 9542);
        assert_eq!(g["view_fresh"], true);
        assert_eq!(g["virtual_tokens"], "0");
        assert_eq!(g["pass_id"], 7);
        assert_eq!(g["trigger"], "poll");
        assert!(g.get("walk_mismatch").is_none());
        let rows = reserve["placement"]["candidates"].as_array().unwrap();
        assert_eq!(rows[0]["queue_room"], false);
        assert_eq!(rows[1]["queue_room"], true);
        assert_eq!(rows[1]["backlog_tokens"], "9000");
        drop((on_a, on_b, released));
    }

    /// sjf, two equal cold requests: the first prefers `a` (no room) and adds
    /// its 8k to the virtual backlog of `a`; the second then prefers `b`
    /// (room). The decision rows show the virtual tokens in the backlog.
    #[test]
    fn queue_step_charges_the_held_work_ahead_as_virtual_backlog() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("queue-virtual.jsonl");
        let book = PrefillWorkBook::new(Some(&path), "test", None).unwrap();
        let mut on_a = uncached(&book, "on_a", "a", 10_000);
        on_a.mark_dispatched().unwrap();
        let mut on_b = uncached(&book, "on_b", "b", 12_000);
        on_b.mark_dispatched().unwrap();
        book.apply_progress("a", progress(1, &[]), json!({}), Instant::now())
            .unwrap();
        book.apply_progress(
            "b",
            progress(1, &[("on_b", 12_000)]),
            json!({}),
            Instant::now(),
        )
        .unwrap();
        let urls = vec!["a".to_string(), "b".to_string()];
        let depth = [9542u64, 9542];
        let cands = vec![candidate("a", 0), candidate("b", 0)];
        let held = [
            queue_held("first", 8000, &cands, Some("a"), false),
            queue_held("second", 8000, &cands, Some("a"), false),
        ];
        let step = book
            .queue_step(cost_params(9.2), &held, &queue_args(&urls, &depth))
            .unwrap();
        assert_eq!(step.plain, vec![0, 0]);
        let Some((index, result)) = step.release else {
            panic!("expected a release");
        };
        assert_eq!(index, 1);
        let released = result.unwrap();
        assert_eq!(released.candidate.worker_url, "b");
        assert_eq!(released.record["rank"], 1);
        let reserve = trace_lines(&path)
            .into_iter()
            .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == "second")
            .unwrap();
        let rows = reserve["placement"]["candidates"].as_array().unwrap();
        assert_eq!(rows[0]["queue_virtual_tokens"], "8000");
        assert_eq!(rows[0]["backlog_tokens"], "18000");
        assert_eq!(rows[0]["cost"], 26_000.0);
        assert_eq!(rows[1]["queue_virtual_tokens"], "0");
        assert_eq!(rows[1]["cost"], 20_000.0);
        drop((on_a, on_b, released));
    }

    /// A reservation in the unknown state (transport error) is not known to
    /// be in an engine queue: the queued work leaves it out, the cost-function
    /// backlog keeps it.
    #[test]
    fn queue_queued_work_leaves_out_unknown_reservations() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let mut lost = uncached(&book, "lost", "a", 30_000);
        lost.mark_dispatched().unwrap();
        lost.mark_unknown("prefill_transport_error: reset").unwrap();
        book.apply_progress("a", progress(1, &[]), json!({}), Instant::now())
            .unwrap();
        let urls = vec!["a".to_string()];
        let rows = book.queue_rows(&urls, Instant::now(), 8192);
        assert_eq!(rows[0].queued, 0);
        assert!(rows[0].view_age_ms.is_some());
        assert_eq!(unacknowledged(&book, "a"), 30_000);
        let cands = vec![candidate("a", 0)];
        let held = [queue_held("r", 1000, &cands, None, true)];
        let step = book
            .queue_step(cost_params(9.2), &held, &queue_args(&urls, &[9542]))
            .unwrap();
        let Some((_, result)) = step.release else {
            panic!("the unknown work does not block the P");
        };
        let released = result.unwrap();
        assert_eq!(
            released.decision["candidates"][0]["backlog_tokens"],
            "30000"
        );
        drop((lost, released));
    }

    /// The terminal hook runs after a finish and a rollback, not after a
    /// dispatch or an unknown status.
    #[test]
    fn terminal_hook_runs_after_terminal_statuses_only() {
        use std::sync::atomic::{AtomicU64, Ordering};
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let calls = Arc::new(AtomicU64::new(0));
        let counter = Arc::clone(&calls);
        book.set_terminal_hook(Box::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
        }));
        let mut done = uncached(&book, "done", "a", 1000);
        done.mark_dispatched().unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        done.finish("prefill_response_success").unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let mut lost = uncached(&book, "lost", "a", 1000);
        lost.mark_dispatched().unwrap();
        lost.mark_unknown("transport").unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        drop(uncached(&book, "rolled_back", "a", 1000));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        drop((done, lost));
    }

    #[test]
    fn rid_refusal_matches_the_reservation_errors() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        assert!(book
            .rid_refusal(None)
            .unwrap()
            .contains("requires a unique string rid"));
        assert!(book.rid_refusal(Some("")).is_some());
        assert!(book.rid_refusal(Some("fresh")).is_none());
        let _r = uncached(&book, "used", "a", 10);
        assert!(book
            .rid_refusal(Some("used"))
            .unwrap()
            .contains("cannot reuse a rid"));
    }

    /// A book made with a policy label writes `"policy"` on every trace row
    /// (reserve, dispatch, engine progress, finish, snapshot). The other
    /// constructors write no `policy` key.
    #[test]
    fn policy_label_on_every_trace_event() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("labelled.jsonl");
        let book = PrefillWorkBook::new(Some(&path), "symmetric_pd_cost_function", None).unwrap();
        let (_, mut r, _) = book
            .reserve_cost_session(
                10_000,
                Some("r1".into()),
                vec![candidate("a", 0)],
                cost_params(18.4),
                None,
            )
            .unwrap();
        r.mark_dispatched().unwrap();
        book.apply_progress("a", progress(1, &[("r1", 0)]), json!({}), Instant::now())
            .unwrap();
        r.finish("prefill_response_success").unwrap();
        book.snapshot();
        let read = |path: &Path| -> Vec<Value> {
            std::fs::read_to_string(path)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect()
        };
        let rows = read(&path);
        let events: Vec<&str> = rows.iter().filter_map(|e| e["event"].as_str()).collect();
        for event in [
            "reserve",
            "dispatched",
            "prefill_progress",
            "finished",
            "snapshot",
        ] {
            assert!(events.contains(&event), "{event} missing: {events:?}");
        }
        for row in &rows {
            assert_eq!(row["policy"], "symmetric_pd_cost_function", "{row}");
            assert_eq!(row["schema"], "placement-trace.v1", "{row}");
        }
    }

    /// `--symmetric-pd-trace-session-moves`: the second decision of a session
    /// that leaves its last P has p_changed, the previous P and its hit; a
    /// request without a session has no record, and without the flag no
    /// event has one.
    #[test]
    fn session_move_record_names_the_previous_p() {
        for on in [true, false] {
            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path().join("moves.jsonl");
            let book = PrefillWorkBook::new(Some(&path), "test", None).unwrap();
            if on {
                book.enable_session_moves();
            }
            let cands = || vec![candidate("a", 4000), candidate("b", 0)];
            let session = Some("0/agent-1");
            let (_, r1, d1) = book
                .reserve_cost_session(5000, Some("s1".into()), cands(), cost_params(9.2), session)
                .unwrap();
            assert_eq!(d1["chosen"], "a");
            let mut busy = uncached(&book, "busy", "a", 60_000);
            busy.mark_dispatched().unwrap();
            let (_, r2, d2) = book
                .reserve_cost_session(5000, Some("s2".into()), cands(), cost_params(9.2), session)
                .unwrap();
            assert_eq!(d2["chosen"], "b");
            let (_, r3, _) = book
                .reserve_cost_session(5000, Some("s3".into()), cands(), cost_params(9.2), None)
                .unwrap();
            let events = trace_lines(&path);
            let moves: Vec<&Value> = events
                .iter()
                .filter(|e| e["event"] == "reserve")
                .map(|e| &e["session_move"])
                .collect();
            assert_eq!(moves.len(), 4, "s1, busy, s2, s3");
            if !on {
                assert!(moves.iter().all(|m| m.is_null()), "{moves:?}");
                assert!(book.snapshot().get("session_moves").is_none());
                drop((r1, r2, r3, busy));
                continue;
            }
            let move_of = |rid: &str| {
                events
                    .iter()
                    .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == rid)
                    .map(|e| e["session_move"].clone())
                    .unwrap()
            };
            let (first, second) = (move_of("s1"), move_of("s2"));
            assert!(move_of("busy").is_null(), "not a cost-function decision");
            assert_eq!(first["prev_p"], Value::Null);
            assert_eq!(first["p_changed"], false);
            assert_eq!(first["session"], "0/agent-1");
            assert_eq!(
                (
                    first["max_local_hit"].clone(),
                    first["max_local_hit_p"].clone()
                ),
                (json!(4000), json!("a"))
            );
            assert_eq!(second["prev_p"], "a");
            assert_eq!(second["p_changed"], true);
            assert_eq!(second["chosen"], "b");
            assert_eq!(second["prev_p_local_hit"], 4000);
            assert_eq!(second["chosen_local_hit"], 0);
            assert_eq!(second["pulled_tokens"], 0);
            assert_eq!(second["input_tokens"], 5000);
            assert!(move_of("s3").is_null(), "no session");
            assert_eq!(
                book.snapshot()["session_moves"],
                json!({"decisions": 2, "p_changes": 1, "sessions": 1})
            );
            drop((r1, r2, r3, busy));
        }
    }

    /// cost-function auto: the first decision is in start-up (the fallback 18.4 and
    /// every auto key). With a 10-minute history, lambda = clamp(q x R /
    /// R_hat) from the traced q and R_hat, and the candidates pay it.
    #[test]
    fn cost_auto_decision_resolves_lambda_and_traces_its_inputs() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        let decide = |rid: &str| {
            let (_, r, d) = book
                .reserve_cost_session(
                    110_000,
                    Some(rid.into()),
                    vec![candidate("a", 100_000), pulling("b", 0, 100_000)],
                    auto_params(18.4),
                    Some("0/agent-1"),
                )
                .unwrap();
            drop(r);
            d
        };
        let d = decide("s1");
        assert_eq!(d["copy_price_mode"], "auto");
        assert_eq!(d["fallback_reason"], "startup");
        assert_eq!(d["copy_price_used"], 18.4);
        assert_eq!(d["copy_price"], 18.4);
        assert_eq!(d["copy_price_fallback"], 18.4);
        assert!(d["copy_price_raw"].is_null());
        assert_eq!(d["n_p"], 2);
        assert_eq!(d["rate_mode"], "fixed");
        assert_eq!(d["rate_used"], 13_500.0);
        assert_eq!(d["session_known"], true);
        assert_eq!(d["wait"], "tokens", "auto adds the later-option keys");
        for key in [
            "q",
            "busy",
            "inflight_mean",
            "n_active",
            "r_hat",
            "auto_elapsed_s",
            "auto_tau_s",
            "session_idle_s",
            "copy_price_min",
            "copy_price_max",
            "rate_fallback",
            "rate_fallback_reason",
        ] {
            assert!(d.get(key).is_some(), "{key}: {d}");
        }
        for row in d["candidates"].as_array().unwrap() {
            for key in [
                "auto_in_flight",
                "auto_busy",
                "auto_inflight_mean",
                "auto_rate_tokens_per_s",
            ] {
                assert!(row.get(key).is_some(), "{key}: {row}");
            }
        }
        let copy_b = d["candidates"][1]["copy_tokens"].as_f64().unwrap();
        assert!((copy_b - 1_840_000.0).abs() < 1e-6, "{d}");
        assert_eq!(d["chosen"], "a");
        // A 10-minute history: a and b busy 3 s of every 4 s at 10,000
        // tokens per busy second; 60 sessions with an event 5 s ago.
        let Some(t0) = Instant::now().checked_sub(std::time::Duration::from_secs(600)) else {
            return; // the monotonic clock is younger than 10 minutes
        };
        let at = |s: f64| t0 + std::time::Duration::from_secs_f64(s);
        let mut auto = crate::scheduler::placement_cost::AutoPriceState::new(t0, 120.0);
        auto.observe_pool(["a", "b"], t0);
        let mut t = 0.0;
        while t < 590.0 {
            for p in ["a", "b"] {
                auto.dispatched(p, at(t));
                auto.prefill_done(p, Some(30_000), at(t + 3.0));
            }
            t += 4.0;
        }
        for i in 0..60u64 {
            auto.session_started(i, at(594.0));
            auto.session_finished(i, at(595.0));
        }
        book.state.lock().auto_price = Some(auto);
        let d = decide("s2");
        assert!(d["fallback_reason"].is_null(), "{d}");
        assert_eq!(d["n_active"], 60);
        let q = d["q"].as_f64().unwrap();
        let busy = d["busy"].as_f64().unwrap();
        let r_hat = d["r_hat"].as_f64().unwrap();
        let lambda = d["copy_price_used"].as_f64().unwrap();
        assert!((q - 30.0 * busy).abs() < 1e-9, "{d}");
        // The same operations as PlacementCostParams::resolved (mu = 1).
        let raw = q * (13_500.0 / r_hat);
        assert_eq!(d["copy_price_raw"], raw, "{d}");
        assert_eq!(lambda, raw.clamp(4.6, 1000.0));
        assert_eq!(d["copy_price"], lambda);
        assert!(lambda > 15.0 && lambda < 45.0, "{d}");
        let copy_b = d["candidates"][1]["copy_tokens"].as_f64().unwrap();
        assert!((copy_b - lambda * 100_000.0).abs() < 1e-6, "{d}");
    }

    /// Fixed cost-function creates no auto state, key or event. Auto adds the
    /// snapshot summary and one `cost_auto_p_empty` event per busy period,
    /// and END removes a session from N_active at once.
    #[test]
    fn cost_auto_end_snapshot_and_fixed_mode_untouched() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("fixed.jsonl");
        let book = PrefillWorkBook::new(Some(&path), "test", None).unwrap();
        let (_, mut r, d) = book
            .reserve_cost_session(
                10_000,
                Some("f1".into()),
                vec![candidate("a", 0)],
                cost_params(18.4),
                Some("0/agent"),
            )
            .unwrap();
        assert!(
            d.get("copy_price_mode").is_none() && d.get("wait").is_none(),
            "{d}"
        );
        r.mark_dispatched().unwrap();
        book.apply_progress("a", progress(1, &[("f1", 0)]), json!({}), Instant::now())
            .unwrap();
        r.finish("prefill_response_success").unwrap();
        assert!(book.state.lock().auto_price.is_none());
        assert!(book.snapshot().get("auto_price").is_none());
        assert!(!book.auto_price_end_session("0/agent"));
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("auto_price"), "{text}");

        let path = tmp.path().join("auto.jsonl");
        let book = PrefillWorkBook::new(Some(&path), "test", None).unwrap();
        let decide = |rid: &str, session: Option<&str>| {
            book.reserve_cost_session(
                10_000,
                Some(rid.into()),
                vec![candidate("a", 0), candidate("b", 0)],
                auto_params(18.4),
                session,
            )
            .unwrap()
        };
        let (_, r1, _) = decide("s1", Some("0/agent-1"));
        let (_, r2, _) = decide("s2", Some("0/agent-2"));
        drop((r1, r2));
        let (_, r, d) = decide("probe1", None);
        assert_eq!(d["n_active"], 2, "{d}");
        assert_eq!(d["session_known"], false);
        drop(r);
        assert!(book.auto_price_end_session("0/agent-1"));
        assert!(!book.auto_price_end_session("0/agent-1"));
        let (_, r, d) = decide("probe2", None);
        assert_eq!(d["n_active"], 1, "{d}");
        drop(r);
        // A dispatched request ends its P's busy period at the P response.
        let (selected, mut r, _) = decide("busy", None);
        r.mark_dispatched().unwrap();
        r.finish("prefill_response_success").unwrap();
        let snapshot = book.snapshot();
        let auto = &snapshot["auto_price"];
        assert_eq!(auto["credited"], 1, "{snapshot}");
        assert_eq!(auto["credited_tokens"], 10_000);
        assert_eq!(auto["workers"].as_array().unwrap().len(), 2);
        assert_eq!(auto["workers"][0]["in_flight"], 0);
        let text = std::fs::read_to_string(&path).unwrap();
        let empties: Vec<Value> = text
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .filter(|event| event["event"] == "auto_price_p_empty")
            .collect();
        assert_eq!(empties.len(), 1, "{text}");
        assert_eq!(empties[0]["worker_url"], selected.worker_url.as_str());
        assert_eq!(empties[0]["busy_period_requests"], 1);
        assert!(empties[0]["busy_period_s"].as_f64().unwrap() >= 0.0);
    }

    /// cost-function auto: the P of an entry has one more in-flight request from its
    /// dispatch until its prefill ends, released once on every path: 0
    /// remaining work in a snapshot, a success or error response, a
    /// transport error (unknown) with a late finish, a drop after dispatch.
    /// A rollback before dispatch releases only the session. Only a
    /// completed prefill credits tokens: the larger of the estimate and the
    /// largest remaining work seen.
    #[test]
    fn cost_auto_counts_each_request_once_on_every_path() {
        let book = PrefillWorkBook::new(None, "test", None).unwrap();
        // 20,000 tokens with 8,000 cached on a: an estimate of 12,000 uncached.
        let pick = |rid: &str, session: Option<&str>| {
            book.reserve_cost_session(
                20_000,
                Some(rid.into()),
                vec![candidate("a", 8_000)],
                auto_params(18.4),
                session,
            )
            .unwrap()
            .1
        };
        let in_flight = || {
            book.state
                .lock()
                .auto_price
                .as_ref()
                .unwrap()
                .in_flight("a")
        };
        let agent = crate::scheduler::placement_cost::placement_session_hash("0/agent");
        let session = || {
            book.state
                .lock()
                .auto_price
                .as_ref()
                .unwrap()
                .session_in_flight(agent)
        };
        let counts = || {
            let st = book.state.lock();
            let auto = st.auto_price.as_ref().unwrap();
            (auto.credited, auto.credited_tokens, auto.uncredited)
        };
        // Rolled back before dispatch: the P never counts; the session is released.
        let r = pick("r1", Some("0/agent"));
        assert_eq!((in_flight(), session()), (Some(0), Some(1)));
        drop(r);
        assert_eq!((in_flight(), session()), (Some(0), Some(0)));
        // Dispatched (twice: idempotent), 15,000 remaining seen (a cache
        // miss), then 0: released at the 0 and credited 15,000; the later P
        // response changes nothing.
        let t = Instant::now();
        let at = |ms: u64| t + std::time::Duration::from_millis(ms);
        let mut r = pick("r2", Some("0/agent"));
        r.mark_dispatched().unwrap();
        r.mark_dispatched().unwrap();
        assert_eq!((in_flight(), session()), (Some(1), Some(1)));
        book.apply_progress("a", progress(1, &[("r2", 15_000)]), json!({}), at(1))
            .unwrap();
        assert_eq!(in_flight(), Some(1));
        book.apply_progress("a", progress(2, &[("r2", 0)]), json!({}), at(2))
            .unwrap();
        assert_eq!((in_flight(), session()), (Some(0), Some(0)));
        assert_eq!(counts(), (1, 15_000, 0));
        r.finish("prefill_response_success").unwrap();
        assert_eq!((in_flight(), counts()), (Some(0), (1, 15_000, 0)));
        // A success response without a 0 snapshot credits the estimate.
        let mut r = pick("r3", None);
        r.mark_dispatched().unwrap();
        r.finish("prefill_response_success").unwrap();
        assert_eq!((in_flight(), counts()), (Some(0), (2, 27_000, 0)));
        // An error response releases without credit.
        let mut r = pick("r4", None);
        r.mark_dispatched().unwrap();
        r.finish("prefill_response_error").unwrap();
        assert_eq!((in_flight(), counts()), (Some(0), (2, 27_000, 1)));
        // A transport error (unknown) releases; its late finish does not again.
        let mut r = pick("r5", None);
        r.mark_dispatched().unwrap();
        r.mark_unknown("prefill_transport_error: reset").unwrap();
        assert_eq!((in_flight(), counts()), (Some(0), (2, 27_000, 2)));
        r.finish("prefill_response_success").unwrap();
        assert_eq!((in_flight(), counts()), (Some(0), (2, 27_000, 2)));
        // A drop after dispatch (unknown) releases the P and the session.
        let mut r = pick("r6", Some("0/agent"));
        r.mark_dispatched().unwrap();
        assert_eq!((in_flight(), session()), (Some(1), Some(1)));
        drop(r);
        assert_eq!((in_flight(), session()), (Some(0), Some(0)));
        assert_eq!(counts(), (2, 27_000, 3));
        // Two overlapping requests: the P is empty only after both.
        let (mut x, mut y) = (pick("r7", None), pick("r8", None));
        x.mark_dispatched().unwrap();
        y.mark_dispatched().unwrap();
        assert_eq!(in_flight(), Some(2));
        x.finish("prefill_response_success").unwrap();
        assert_eq!(in_flight(), Some(1));
        y.finish("prefill_response_success").unwrap();
        assert_eq!(in_flight(), Some(0));
    }

    /// Regression: a P-leg that the engine never listed gets
    /// a transport error and stays unknown while its P keeps answering. It
    /// holds its work until a reply of its P sent 60 s after it became
    /// unknown, then expires: every counter of its worker returns to its
    /// value before the reservation (another request stays), the RID leaves
    /// the active map but stays used, the cost-function auto end (applied at the
    /// transport error) is not applied again, and the trace has exactly one
    /// `expired` event with the reason and the evidence, also after later
    /// replies and snapshots.
    #[test]
    fn unknown_never_listed_expires_after_the_expiry_and_releases_its_work_once() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("cost.jsonl");
        let expire = Some(secs(60));
        let book = PrefillWorkBook::new(Some(&path), "symmetric_pd_cost_function", expire).unwrap();
        let mut other = uncached(&book, "other", "a", 7_000);
        other.mark_dispatched().unwrap();
        let before = counters(&book, "a");
        assert_eq!(before, (7_000, 7_000, 7_000, 1));
        // 20,000 tokens with 8,000 cached: 12,000 uncached.
        let (_, mut lost, _) = book
            .reserve_cost_session(
                20_000,
                Some("lost".into()),
                vec![candidate("a", 8_000)],
                auto_params(18.4),
                Some("0/agent"),
            )
            .unwrap();
        lost.mark_dispatched().unwrap();
        lost.mark_unknown("prefill_transport_error: upstream timed out")
            .unwrap();
        let held = counters(&book, "a");
        assert_eq!(held, (19_000, 19_000, 19_000, 2));
        let auto = || {
            let st = book.state.lock();
            let auto = st.auto_price.as_ref().unwrap();
            (auto.in_flight("a"), auto.credited, auto.uncredited)
        };
        // The transport error ended the prefill (no credit).
        assert_eq!(auto(), (Some(0), 0, 1));
        let since = unknown_since(&book, &lost);
        let just_before = since + secs(60) - std::time::Duration::from_millis(1);
        reply(&book, "a", just_before, &[]);
        assert_eq!(counters(&book, "a"), held, "held before the expiry");
        assert_eq!(book.snapshot()["unknown_requests"], 1);
        reply(&book, "a", since + secs(60), &[]);
        assert_eq!(counters(&book, "a"), before);
        {
            let st = book.state.lock();
            assert!(st.unknown.is_empty());
            assert!(!st.active_by_rid.contains_key("lost"));
            assert!(st.used_rids.contains("lost"));
            assert_eq!(st.entries[&lost.id].status, Status::Expired);
        }
        assert_eq!(lost.debuginfo()["reason"], UNKNOWN_EXPIRED_REASON);
        assert_eq!(auto(), (Some(0), 0, 1), "auto end applied once");
        assert!(res(&book, "lost", 1, vec![candidate("a", 0)]).is_err());
        // Later replies and snapshots change nothing and write no second event.
        reply(&book, "a", since + secs(600), &[]);
        let snapshot = book.snapshot();
        assert_eq!(snapshot["unknown_requests"], 0);
        assert_eq!(snapshot["unknown_expire_s"], 60.0);
        assert_eq!(snapshot["expired_rid_listed_again"], 0);
        assert_eq!(snapshot["workers"][0]["status_counts"]["expired"], 1);
        assert_eq!(counters(&book, "a"), before);
        let rows = trace_rows(&path);
        let expired: Vec<&Value> = rows.iter().filter(|e| e["event"] == "expired").collect();
        assert_eq!(expired.len(), 1, "{expired:?}");
        let e = expired[0];
        assert_eq!(e["entry"]["rid"], "lost");
        assert_eq!(e["entry"]["status"], "expired");
        assert_eq!(e["entry"]["reason"], UNKNOWN_EXPIRED_REASON);
        assert_eq!(e["entry"]["uncached_tokens"], 12_000);
        assert_eq!(e["policy"], "symmetric_pd_cost_function");
        assert_eq!(e["expiry"]["expire_s"], 60.0);
        assert_eq!(e["expiry"]["quiet_s"], 60.0);
        assert_eq!(e["expiry"]["unknown_for_s"], 60.0);
        assert!(e["expiry"]["since_last_progress_s"].is_null());
        other.finish("prefill_response_success").unwrap();
        assert_eq!(counters(&book, "a"), (0, 0, 0, 0));
    }
}
