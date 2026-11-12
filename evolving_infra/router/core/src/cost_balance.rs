// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Fixed per-request prefill work, held until the owning P request terminates.
//! The original policy balances N×U cost; the optional locality policy retains
//! a cache owner until a detour saves enough pending uncached-token work.
//!
//! The uncached mode replaces estimates with per-RID remaining work reported
//! after engine chunk completion. Missing RIDs never imply completed work.
//! D first byte and timers never retire an entry, with one exception below.
//! A complete P HTTP response (including an error response) is terminal. A
//! transport error is not proof of engine termination: the caller must mark it
//! unknown, which retains its cost and stops further admission. The reservation
//! must move into the detached P task, independently of the client/D lifetime.
//!
//! The exception: a book made with an unknown expiry
//! ([`CostBook::new_uncached_with_policy`], `--symmetric-pd-unknown-expire-secs`)
//! moves an unknown entry to `expired` when its P has answered without its
//! RID for that time: an applied progress reply of its P was sent at least
//! the expiry after the entry's last evidence (the later of the moment it
//! became unknown and the send time of the last poll whose reply listed it).
//! A P without applied replies (dead, partitioned, polls timing out) keeps
//! its unknown entries. Expiry is terminal and releases the entry's work once,
//! as a finish does (trace event `expired`, reason
//! `unknown_expired_no_engine_progress`). An SGLang P aborts a prefill when
//! the router's connection closes, and its progress lists only RIDs whose
//! first chunk has launched; a request still in its queue and a dropped one
//! look alike, and the expiry bounds that error. A reply that lists the RID
//! of an expired entry again measures the error: the first such reply per
//! entry writes the event `expired_rid_listed_again` (the snapshot counts
//! these entries); the ledger does not take the work back. The check runs
//! when a progress reply of the entry's P is applied and at each
//! [`CostBook::snapshot`], only while unknown entries exist.
//!
//! The TRADE placement (`--co-schedule-trade`) uses the same uncached ledger
//! but selects argmin of backlog + own uncached + unhidden pull time + copy
//! price (see [`crate::scheduler::co_schedule::trade_cost`]). Its backlog is
//! the full ledger by default; `--co-schedule-trade-exclude-unlaunched` leaves
//! out work the engine has not yet acknowledged in a progress snapshot. The
//! engine lists a RID only after its first prefill chunk launches, so
//! "unacknowledged" covers transit, the bootstrap queue and the waiting queue.
//!
//! With `--co-schedule-trade-wait eft` the TRADE wait and service terms come
//! from the EFT-0+ closed form ([`crate::eft_cheap`]): B_w = the drain of the
//! P's engine `prefill_routing` snapshot plus the active entries of that P
//! that the snapshot does not show (not in its rows, not in the in-flight
//! pass that R0 prices; unknown entries are left to the snapshot), S_w = the
//! step-model service of the request. The snapshot is parsed off this lock
//! ([`RoutingSnapshot`]); the drain is computed under it, so the outstanding
//! rows and the decision are one transaction. For a P without a usable
//! snapshot, B_w is the step-model drain of all its active entries (the
//! ledger on the same scale); the trade decision counts these fallbacks by
//! reason.
//!
//! With `--co-schedule-trade-hol-weight` above 0 the TRADE cost also has the
//! head-of-line (HOL) term ([`crate::scheduler::co_schedule::trade_hol_tokens`]).
//! Its arrival rate a_w comes from this book: each TRADE decision prunes the
//! placement record ([`crate::scheduler::co_schedule::TradeArrivals`]) to the
//! rate window, reads a_w of every candidate, and records its own placement
//! when it commits the reservation. All of this is under the same lock, so
//! concurrent decisions see each other. A later rollback or failure does not
//! remove a record.
//!
//! With `--co-schedule-trade-copy-lambda auto` or
//! `--co-schedule-trade-rate-tokens-per-s auto` this book also keeps the
//! measurements of [`crate::scheduler::co_schedule::TradeAutoState`], created
//! at the first auto decision. Only the entries that an auto decision
//! reserved feed it. The P of such an entry has one more in-flight request
//! from its dispatch until its prefill ends: the first progress snapshot
//! that shows 0 remaining work for its RID (credited with its computed
//! tokens), the P response (`prefill_response_success` credits too; any
//! other reason does not), a transport error or a drop after dispatch
//! (unknown, no credit). A rollback before dispatch releases only its
//! session. Each end is applied once, so no path leaks an in-flight request.
//! The computed tokens are the larger of the reserved uncached estimate and
//! the largest remaining work that a snapshot showed. Each decision resolves
//! lambda and R from these measurements
//! ([`crate::scheduler::co_schedule::TradeParams::resolved`]) and then
//! prices the candidates with them; the trace event `trade_auto_p_empty`
//! records each busy period that ends, and the one-second snapshot has
//! `trade_auto`.
//!
//! With `--co-schedule-trade-wait olp` (symmetric_pd cost function only) the
//! TRADE wait and service terms come from the online latency predictor
//! ([`crate::olp`]): the structural work of each P's live ledger rows (a
//! per-worker index of live reservation IDs, kept only with OLP) and of the
//! request, scaled online. A decision-level gate gives every P the ledger
//! terms while OLP is cold, stale or orders recent labels worse than the
//! ledger: the decision is then the cost function's, bit for bit. The book
//! feeds OLP from each applied progress reply (the engine's busy counters
//! and the chunk work of the listed entries; a first listing may only
//! correct the cache hit and is not chunk work) and labels each OLP
//! decision when its P-leg is answered with `prefill_response_success`.
//! With `--co-schedule-trade-olp-shadow` OLP is fed and traced and the
//! decision is the ledger's. With OLP the one-second snapshot has `olp` (the
//! gate's warm-up inputs and its results so far). Without OLP state none of
//! this runs and the trace has no OLP key.
//!
//! Faster OLP (each off by default; flag off = the bytes above):
//! `--co-schedule-trade-olp-incremental` keeps, per P, the exact integer
//! sums of the work terms of its live rows, updated at the commit, at each
//! applied progress reply (first chunk launched, remaining work) and at the
//! end of a row, so a decision prices a P in O(1) instead of walking its
//! rows ([`crate::olp::OlpParams::queue_of`]; equal rows, launched tokens
//! and choices, q_model equal up to floating-point rounding).
//! `--co-schedule-trade-olp-trace-compact` writes fewer OLP keys per
//! decision. `--co-schedule-cost-trace-async` moves the trace write and
//! flush to a background thread ([`CostBook::enable_async_trace`]).

//!
//! With `--symmetric-pd-loads-host-room` ([`CostBook::enable_host_room`]) the book
//! keeps the latest `host_room` section of each P's progress reply (the
//! engine's host pool by retention class; [`CostBook::note_host_room`]) and
//! writes it, with its age, into the one-second snapshot and every TRADE
//! decision (`host_room_by_p`). No decision reads it. Without the flag
//! neither key is written.
//!
//! With `--co-schedule-trade-pull-price phi` ([`CostBook::enable_host_room_phi`])
//! the book also keeps phi of each P from the evicted counters of its
//! host_room sections ([`crate::scheduler::co_schedule::TradePhiState`]), and
//! each TRADE decision prices the pull of candidate P at
//! max(min, lambda_q x phi_P) (lambda_q = the copy price auto before the
//! clamp; min = the auto clamp minimum; a P without host_room data uses
//! phi_P = 1). Only pulled tokens are priced: a pull discount, not a price of
//! every landing. The decision rows, the decision and the snapshot then carry
//! the phi keys; with `lambda` (the default) nothing changes.
//!
//! With `--co-schedule-trade-wait e3` each candidate's wait and service come
//! from the E3 forecast ([`crate::trade_e3`]). The fits, the target tokens
//! and the parsed snapshot are prepared before this lock
//! ([`crate::trade_e3::TradeE3Context`]); the forecast runs under it,
//! because its queue holds the router's own rows of that P that the
//! snapshot does not show (active entries, and entries that finished after
//! the snapshot time, decided after it). The selection is the prediction
//! gate ([`crate::scheduler::co_schedule::select_trade_e3`]).

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
    pub view: crate::eft_cheap::SnapshotView,
    pub poll_sent: Instant,
    /// When the router received the reply (`wait = e3`: the snapshot time
    /// is this minus the engine's `age_s`).
    pub poll_received: Instant,
    /// `wait = e3`: the same snapshot in E3 form (its timeline samples are
    /// already taken out). None for the other wait sources.
    pub e3: Option<Arc<crate::trade_e3::E3View>>,
}

/// Finished entries stay visible to the `wait = e3` router rows this long
/// (s): a request decided after a snapshot and finished before the decision
/// still ran on P after the snapshot time. Longer than any usable view age.
const RECENT_FINISHED_S: f64 = 60.0;

/// Inputs of `--co-schedule-trade-wait eft`: the frozen step model, the
/// EFT knobs (jit lead window and pull span of the drain parts) and the
/// latest snapshot of each P, taken before the cost-book lock.
#[derive(Debug, Clone)]
pub struct TradeEftContext {
    pub model: Arc<crate::eft_cheap::EftModel>,
    pub knobs: crate::eft_cheap::EftKnobs,
    pub views: HashMap<String, Arc<RoutingSnapshot>>,
}

/// The wait terms of one candidate for `wait = eft`. `terms` is None only
/// without a step model (every P then uses the ledger terms). `fallback`
/// names why the P has no usable snapshot; its terms then price the
/// router's own rows for that P with the step model.
struct WaitProbe {
    terms: Option<crate::scheduler::co_schedule::TradeWaitTerms>,
    fallback: Option<&'static str>,
    view_age_s: Option<f64>,
    view_key: Value,
    drain: Option<crate::eft_cheap::Drain>,
    outstanding_rows: usize,
    outstanding_tokens: i64,
    /// Active entries left out because the in-flight pass lists them (R0
    /// prices their work).
    pass_rows: usize,
    shape: Option<crate::eft_cheap::Shape>,
    /// Source `eft` only: how the snapshot's own rows were priced.
    row_notes: Option<RowNotes>,
}

/// How the engine rows of a used snapshot were priced, for the trace.
struct RowNotes {
    /// The snapshot's degraded flags (sorted).
    degraded: Vec<String>,
    /// Rows priced at their full input length (no usable prefix data).
    cold_rows: u64,
    cold_tokens: i64,
    /// Rows that the engine start-permission rule left out.
    intent_skipped: u64,
}

/// One TRADE decision's OLP evaluation (mode `olp` or `shadow`).
struct OlpEval {
    /// Mode `olp` with an open gate: the OLP terms price every candidate.
    active: bool,
    /// The terms of every candidate, in candidate order.
    preds: Vec<crate::olp::OlpPrediction>,
    /// `trade_decision.olp`.
    value: Value,
    /// The `olp_step` event of the first decision of a step.
    step_event: Option<Value>,
    /// The decision time (s since the book started).
    t_s: f64,
    /// Incremental queue with `verify` only: every candidate's terms from
    /// the walk of its rows (compared with `preds` by the caller).
    verify_terms: Option<Vec<crate::scheduler::co_schedule::TradeWaitTerms>>,
}

impl WaitProbe {
    fn no_model() -> Self {
        Self {
            terms: None,
            fallback: Some("no_model"),
            view_age_s: None,
            view_key: Value::Null,
            drain: None,
            outstanding_rows: 0,
            outstanding_tokens: 0,
            pass_rows: 0,
            shape: None,
            row_notes: None,
        }
    }
}

/// The E3 forecast of one candidate for `wait = e3`, under the lock.
/// `fallback` is None exactly when the candidate is usable: `out` has
/// `pred_s` and `wait_s`, and `bound_s` is set.
#[derive(Default)]
struct E3Probe {
    out: Option<crate::e3::Output>,
    bound_s: Option<f64>,
    bucket: Option<&'static str>,
    fallback: Option<&'static str>,
    error: Option<String>,
    notes: crate::trade_e3::BuildNotes,
    /// The degraded flags of the view the forecast started from (sorted;
    /// for example `engine_stale`).
    degraded: Vec<&'static str>,
    /// Decision time minus the snapshot time t0 (s).
    t0_age_s: Option<f64>,
    /// Decision time minus the poll send time, plus the engine's age (s;
    /// the age the limit `eft_max_view_age_s` applies to). A pushed state:
    /// decision time minus its state time.
    view_age_s: Option<f64>,
    view_key: Value,
    /// The prepared inputs (fit, target, host rule) of the candidate.
    prep: Option<crate::trade_e3::E3Prepared>,
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
    push: Option<&'a crate::trade_e3_push::PushState>,
    poll_epoch: Option<&str>,
    d: f64,
    max_age_s: f64,
) -> Result<&'a crate::trade_e3_push::PushState, &'static str> {
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

impl E3Probe {
    /// The wait terms of a usable forecast: B = wait_s, S = pred_s - wait_s.
    fn terms(&self) -> Option<crate::scheduler::co_schedule::TradeWaitTerms> {
        if self.fallback.is_some() {
            return None;
        }
        let out = self.out.as_ref()?;
        let (pred, wait) = (out.pred_s?, out.wait_s?);
        Some(crate::scheduler::co_schedule::TradeWaitTerms {
            wait_s: wait,
            service_s: pred - wait,
        })
    }

    fn usable_bound(&self) -> Option<f64> {
        self.terms().and(self.bound_s)
    }

    /// The E3 arrival rate of the candidate's fit (requests per second).
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

/// One active cost-book entry as an outstanding row of the EFT drain. An
/// entry that the engine has acknowledged (its first chunk launched) is
/// priced as started: its host restore and pull are done, and the rest
/// computes after the prefix computed so far. Otherwise: q = expected hit,
/// h = local hit - GPU hit (unknown GPU: all host), s = the hinted pull.
fn outstanding_input(entry: &Entry) -> crate::eft_cheap::OutstandingInput {
    let rid = entry.rid.clone().map_or(Value::Null, Value::String);
    let uncached_tokens = i64::try_from(entry.uncached_tokens).ok();
    if entry.acknowledged {
        let done = entry.input_tokens.saturating_sub(entry.uncached_tokens) as i64;
        return crate::eft_cheap::OutstandingInput {
            rid,
            uncached_tokens,
            expected_hit: done,
            local_hit: done,
            gpu_hit: done,
            hinted: false,
        };
    }
    crate::eft_cheap::OutstandingInput {
        rid,
        uncached_tokens,
        expected_hit: entry.expected_hit as i64,
        local_hit: entry.local_hit as i64,
        gpu_hit: entry.gpu_hit.unwrap_or(0) as i64,
        hinted: entry.hint.is_some(),
    }
}

/// One live entry as an OLP queue row: its remaining (engine-updated)
/// uncached tokens, its input and whether its first chunk has launched.
fn olp_row(entry: &Entry) -> crate::olp::OlpRow {
    crate::olp::OlpRow {
        remaining: entry.uncached_tokens,
        input: entry.input_tokens,
        started: entry.acknowledged,
    }
}

/// The `wait = e3` keys of one candidate row of the trade decision: the
/// forecast and its bound, the input summary, the target tokens and host
/// rule, and the fit sources of the (P, step).
fn e3_row_value(q: &E3Probe, cost: &crate::scheduler::co_schedule::TradeCost) -> Value {
    let out = q.out.as_ref();
    let prep = q.prep.as_ref();
    let fit = prep.and_then(|p| p.fit.as_ref());
    // Three json! blocks: one block of all keys exceeds the macro recursion
    // limit of the crate.
    let forecast = json!({
        "wait_source": if q.fallback.is_none() { "e3" } else { "ledger_fallback" },
        "wait_fallback_reason": q.fallback,
        "wait_s": cost.wait_s,
        "service_s": cost.service_s,
        "dup_tokens": cost.dup_tokens,
        "view_age_s": q.view_age_s,
        "view_key": q.view_key,
        "e3_pred_s": out.and_then(|o| o.pred_s),
        "e3_wait_s": out.and_then(|o| o.wait_s),
        "e3_bound_s": q.bound_s,
        "e3_bucket": q.bucket,
        "e3_idle": out.map(|o| o.idle),
        "e3_path": out.map(|o| o.path),
    });
    let shape = json!({
        "e3_receipt_s": out.map(|o| o.receipt_s),
        "e3_passes_before": out.map(|o| o.passes_before),
        "e3_x_passes": out.map(|o| o.x_passes),
        "e3_n_inflight": out.map(|o| o.n_inflight),
        "e3_partial": out.map(|o| o.partial),
        "e3_n_queue": out.map(|o| o.n_queue),
        "e3_n_dispatched": out.map(|o| o.n_dispatched),
        "e3_n_prefix_unknown": out.map(|o| o.n_prefix_unknown),
        "e3_tokens_ahead": out.map(|o| o.tokens_ahead),
        "e3_x_uncached": out.map(|o| o.x_uncached),
        "e3_x_host": out.map(|o| o.x_host),
        "e3_u_tokens": out.and_then(|o| o.u_tokens),
        "e3_n_held": out.map(|o| o.n_held),
        "e3_x_held": out.map(|o| o.x_held),
        "e3_f_full_s": out.and_then(|o| o.f_full_s),
        "e3_f_after_s": out.and_then(|o| o.f_after_s),
    });
    let inputs = json!({
        "e3_router_rows": q.notes.router_rows,
        "e3_router_rows_cut": q.notes.router_rows_cut,
        "e3_bootstrap_cut": q.notes.bootstrap_cut,
        "e3_null_estimated": q.notes.null_estimated,
        "e3_null_cold": q.notes.null_cold,
        "e3_pull_rows": q.notes.pull_rows,
        "e3_pull_tokens": q.notes.pull_tokens,
        "e3_degraded": q.degraded,
        "e3_target_prefix": prep.map(|p| p.target.prefix),
        "e3_target_host": prep.map(|p| p.target.host),
        "e3_host_rule": prep.map(|p| p.host_rule),
        "e3_session_gap_s": prep.and_then(|p| p.session_gap_s),
        "e3_t0_age_s": q.t0_age_s,
        "e3_fit_step_s": fit.map(|f| f.step),
        "e3_gpu_sources": fit.map(|f| json!([f.gpu_sources.small, f.gpu_sources.large])),
        "e3_launch_source": fit.map(|f| f.launch.source),
        "e3_rx": prep
            .and_then(|p| p.params.as_ref())
            .and_then(|p| p.rx)
            .map(|rx| json!({"a": rx.a, "b": rx.b, "shift": rx.shift})),
        "e3_arrival_rate_per_s": q.arrival_rate(),
        "e3_error": q.error,
    });
    // State push on: the state source and the P's newest pushed state.
    let pushed = q.push_on.then(|| {
        let push = prep.and_then(|p| p.push.as_ref());
        json!({
            "e3_state_source": q.source,
            "e3_push_fallback": q.push_fallback,
            "e3_push_unusable": push.and_then(|p| p.view.unusable),
            "e3_push_seq": push.map(|p| p.seq),
            "e3_push_trigger": push.map(|p| p.trigger.as_str()),
            "e3_push_snapshot_source": push.and_then(|p| p.snapshot_source.as_deref()),
            "e3_push_age_s": q.push_age_s,
            "e3_push_superseded_seq": push.and_then(|p| p.superseded_seq),
            "e3_push_offset_s": push.map(|p| p.offset_s),
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

/// The `wait = e3` keys of the trade decision: the gate settings and
/// numbers, the bound table, the decision time on the E3 clock, the chosen
/// P's forecast (the shadow log reads it) and the fit step of each P.
fn e3_decision_value<'a>(
    params: &crate::scheduler::co_schedule::TradeParams,
    selected: &crate::scheduler::co_schedule::TradeSelection,
    chosen: Option<&E3Probe>,
    forecasts: usize,
    e3: Option<&crate::trade_e3::TradeE3Context>,
    decided_at: Instant,
    steps: impl Iterator<Item = (&'a str, Option<f64>)>,
) -> Value {
    let cfg = e3.map(|ctx| &ctx.cfg);
    let out = chosen.and_then(|q| q.out.as_ref());
    let prep = chosen.and_then(|q| q.prep.as_ref());
    let decision_s = e3.map(|ctx| crate::trade_e3::secs_since(ctx.anchor, decided_at));
    let steps: serde_json::Map<String, Value> = steps
        .map(|(url, step)| (url.to_string(), json!(step)))
        .collect();
    json!({
        "e3_gate_m": params.gate_m,
        "e3_gate_gmin_s": params.gate_gmin_s,
        "e3_bound_level": cfg.map(|c| c.level.as_str()),
        "e3_bound_mode": cfg.map(|c| c.mode.as_str()),
        "e3_bound_sha256": cfg.map(|c| c.bound_sha256.as_str()),
        "e3_rq_margin_s": cfg.map(|c| c.rq_margin_s),
        "e3_gain_s": selected.owner_gain_s,
        "e3_margin_s": selected.margin_s,
        "e3_protected": selected.margin_protected,
        "e3_protect_reason": selected.protect_reason,
        "e3_decision_s": decision_s,
        "e3_decision_unix_s": e3.zip(decision_s).map(|(ctx, d)| ctx.unix_at_anchor + d),
        "e3_forecasts": forecasts,
        "e3_fit_steps": steps,
        "e3_chosen": {
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

#[derive(Clone, Copy)]
enum Placement {
    CostSpread,
    LocalityMargin(u64),
    UncachedThreshold(u64, u64),
    Trade(crate::scheduler::co_schedule::TradeParams),
}

impl Placement {
    fn label(self) -> &'static str {
        match self {
            Self::CostSpread => "cost_spread",
            Self::LocalityMargin(_) => "locality_margin",
            Self::UncachedThreshold(..) => "uncached_threshold",
            Self::Trade(_) => "trade",
        }
    }
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
    restore_tokens: u64,
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
    /// `wait = e3`: the router's estimate of the request on its P at the
    /// decision, (prefix, host) tokens: its E3 queue row while the engine
    /// does not show it, and the null-prefix rule.
    e3_estimate: Option<(i64, i64)>,
    /// TRADE auto bookkeeping; not in the trace.
    auto: AutoTrack,
    /// When it became unknown (the unknown expiry); not in the trace.
    unknown_since: Option<Instant>,
    /// The send time of the last progress poll whose reply listed its RID
    /// after dispatch (the unknown expiry); not in the trace.
    last_progress_seen: Option<Instant>,
    /// `--symmetric-pd-gwq-auto`: its release, until its first sighting; not in
    /// the trace.
    gwq_lead_mark: Option<GwqLeadMark>,
}

/// The TRADE auto bookkeeping of one entry (see the module docs).
#[derive(Debug, Default)]
struct AutoTrack {
    /// An auto decision reserved this entry: its events feed the auto state.
    tracked: bool,
    /// The request's session key ([`crate::scheduler::co_schedule::trade_session_hash`]).
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
            "restore_tokens": self.restore_tokens,
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
    pending_restore_tokens: u128,
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
    /// TRADE `wait = eft`: decisions per fallback reason where a P used the
    /// ledger terms (cumulative; P x decision counts).
    trade_wait_fallbacks: BTreeMap<String, u64>,
    /// TRADE HOL term (`hol_weight > 0` only): the placement times per P
    /// that give the arrival rate a_w.
    trade_arrivals: crate::scheduler::co_schedule::TradeArrivals,
    /// TRADE auto (copy price or rate `auto` only): the router's
    /// measurements, created at the first auto decision.
    trade_auto: Option<crate::scheduler::co_schedule::TradeAutoState>,
    /// OLP (`--co-schedule-trade-wait olp` or the shadow readout): the online
    /// latency predictor and its live index; None (default) without it.
    olp: Option<crate::olp::OlpState>,
    /// The IDs of the unknown entries (the snapshot's `unknown_requests` is
    /// its size).
    unknown: BTreeSet<u64>,
    /// P-legs closed with [`crate::proxy::READER_GONE_FINISH_REASON`]: the
    /// engine ended them because their reader left (cancelled, not failed).
    prefill_reader_gone: u64,
    /// The unknown expiry: RID -> (ID of the expired entry, progress replies
    /// of its P that listed the RID after the expiry).
    expired_by_rid: BTreeMap<String, (u64, u64)>,
    /// The unknown expiry: expired entries whose RID a reply listed again.
    expired_listed_again: u64,
    /// `--symmetric-pd-loads-host-room` only (None: off): the latest `host_room`
    /// section per P.
    host_room: Option<HostRoomBook>,
    /// `--symmetric-pd-gwq-auto` only (None: off): T_react per P.
    gwq_lead: Option<GwqLeadBook>,
    /// `--symmetric-pd-trace-session-moves` only (None: off).
    session_moves: Option<SessionMoveBook>,
    /// Entries that finished in the last [`RECENT_FINISHED_S`] seconds
    /// (finish time, id), oldest first: the `wait = e3` router rows.
    recent_finished: VecDeque<(Instant, u64)>,
    /// `wait = e3` with the state push: candidates per state source (`push`,
    /// `poll`, `none`) and per reason the pushed state was not used
    /// (cumulative; P x decision counts).
    e3_push_sources: BTreeMap<String, u64>,
    e3_push_fallbacks: BTreeMap<String, u64>,
    workers: BTreeMap<String, WorkerCost>,
    // Last admission's health snapshot; not a claim of continuous health.
    healthy_candidates: Vec<String>,
    trace: Option<TraceSink>,
    io_failure: Option<String>,
}

/// Where the cost trace rows go: written and flushed under the cost-book
/// lock (default), or handed in order to a background writer thread
/// (`--co-schedule-cost-trace-async`, [`CostBook::enable_async_trace`]).
enum TraceSink {
    Sync(BufWriter<File>),
    Async(AsyncTrace),
}

/// A message to the background trace writer.
enum TraceMsg {
    /// One event, complete (sequence, schema and times set under the lock).
    Row(Value),
    /// Write and flush every row sent before, then answer.
    Sync(std::sync::mpsc::SyncSender<()>),
}

/// At most this many rows wait for the background writer; a full queue
/// blocks the next event (under the lock), as a slow synchronous write
/// would, so no row is dropped.
const ASYNC_TRACE_QUEUE: usize = 16_384;
/// The background writer flushes at least once per this many rows.
const ASYNC_TRACE_BATCH: usize = 4_096;

/// The background trace writer (`--co-schedule-cost-trace-async`). The
/// cost book sets each event's fields under its lock and sends it over a
/// bounded channel (FIFO, so the file keeps the sequence order); the thread
/// serializes the rows, writes every row waiting in the channel and flushes
/// once per batch, off the decision path. The bytes of each row are those
/// of the synchronous writer. A write error is kept in `failure`: the next
/// event or admission check sees it and admission stops, as with a
/// synchronous error, but a row emitted before that is acknowledged
/// although it may be lost (for example the dispatch row of a request that
/// is then sent). Dropping the cost book drains and flushes the queue.
struct AsyncTrace {
    tx: Option<std::sync::mpsc::SyncSender<TraceMsg>>,
    failure: Arc<Mutex<Option<String>>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl AsyncTrace {
    fn start(file: File) -> io::Result<Self> {
        let (tx, rx) = std::sync::mpsc::sync_channel(ASYNC_TRACE_QUEUE);
        let failure = Arc::new(Mutex::new(None));
        let slot = Arc::clone(&failure);
        let handle = std::thread::Builder::new()
            .name("cost-trace-writer".into())
            .spawn(move || trace_writer(file, rx, slot))?;
        Ok(Self {
            tx: Some(tx),
            failure,
            handle: Some(handle),
        })
    }

    /// The writer's error, if it stopped.
    fn failed(&self) -> Option<String> {
        self.failure.lock().clone()
    }

    fn send(&self, message: TraceMsg) -> io::Result<()> {
        if let Some(reason) = self.failed() {
            return Err(io::Error::other(reason));
        }
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| io::Error::other("cost trace writer closed"))?;
        tx.send(message).map_err(|_| {
            io::Error::other(
                self.failed()
                    .unwrap_or_else(|| "cost trace writer stopped".into()),
            )
        })
    }

    /// Drain, flush and stop the writer.
    fn close(&mut self) {
        drop(self.tx.take());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// The background writer loop: block for a row, write it and every row
/// already waiting (up to [`ASYNC_TRACE_BATCH`]), flush, answer the sync
/// requests of that batch. Ends when the cost book drops the sender (after
/// writing everything received) or at the first write error.
fn trace_writer(
    file: File,
    rx: std::sync::mpsc::Receiver<TraceMsg>,
    failure: Arc<Mutex<Option<String>>>,
) {
    let mut out = BufWriter::with_capacity(1 << 20, file);
    let mut acks = Vec::new();
    while let Ok(first) = rx.recv() {
        let mut next = Some(first);
        let mut written = 0;
        let mut result = Ok(());
        while let Some(message) = next.take() {
            match message {
                TraceMsg::Row(event) => {
                    result = serde_json::to_writer(&mut out, &event)
                        .map_err(io::Error::other)
                        .and_then(|()| out.write_all(b"\n"));
                }
                TraceMsg::Sync(ack) => acks.push(ack),
            }
            if result.is_err() {
                break;
            }
            written += 1;
            if written < ASYNC_TRACE_BATCH {
                next = rx.try_recv().ok();
            }
        }
        if let Err(error) = result.and_then(|()| out.flush()) {
            let reason = error.to_string();
            tracing::error!(target: "cost_balance", error = %reason,
                "async cost trace writer failed; further admission is disabled");
            // The waiting sync requests see the closed answer channel.
            *failure.lock() = Some(reason);
            return;
        }
        for ack in acks.drain(..) {
            let _ = ack.send(());
        }
    }
}

/// The latest engine `host_room` section of each P (`--symmetric-pd-loads-host-room`).
#[derive(Debug, Default)]
struct HostRoomBook {
    /// P URL -> (the section, the send time of the poll that returned it).
    latest: BTreeMap<String, (serde_json::Map<String, Value>, Instant)>,
    /// P URLs whose engine rejected `include=...,host_room` (an engine
    /// without the section); polled without it until a later probe.
    rejected: BTreeSet<String>,
    /// Replies whose `host_room` was not a JSON object (ignored).
    invalid: u64,
    /// P URL -> progress replies that accepted the include but had no
    /// section (an engine without session retention or without a host
    /// pool).
    missing: BTreeMap<String, u64>,
    /// `--co-schedule-trade-pull-price phi` only: phi of each P.
    phi: Option<PhiBook>,
}

/// phi of each P (`--co-schedule-trade-pull-price phi`), from the evicted
/// counters of its host_room sections.
#[derive(Debug)]
struct PhiBook {
    /// tau (s) of the EWMA: the copy price auto tau.
    tau_s: f64,
    states: BTreeMap<String, crate::scheduler::co_schedule::TradePhiState>,
    /// Sections without the two evicted counters (not observed).
    invalid: u64,
    /// The priced candidate rows of the decisions (cumulative), those whose
    /// price the floor raised (lambda_q x phi_P < min), those with price 0,
    /// and the decisions whose chosen P had a floored price.
    rows: u64,
    rows_floored: u64,
    rows_zero: u64,
    decisions: u64,
    chosen_floored: u64,
}

impl PhiBook {
    /// The snapshot row of one P: phi now, its source and the EWMA state.
    fn row(&self, url: &str, now: Instant) -> Value {
        let state = self.states.get(url);
        let view = state.and_then(|s| s.view(now, self.tau_s));
        json!({
            "phi_hat": view.map(|v| v.phi),
            "source": view.map_or("no_host_room", |v| v.source),
            "rate_class0": view.map(|v| v.rate_class0),
            "rate_class1": view.map(|v| v.rate_class1),
            "last_evict_age_s": view.and_then(|v| v.last_evict_age_s),
            "sample_age_s": view.map(|v| v.sample_age_s),
            "samples": state.map_or(0, |s| s.samples),
            "resets": state.map_or(0, |s| s.resets),
        })
    }
}

impl HostRoomBook {
    /// `{url: section + age_s}` for `urls` (null for a P without a section),
    /// the ages at `now`.
    fn by_p<'a>(&self, urls: impl IntoIterator<Item = &'a String>, now: Instant) -> Value {
        let rows: serde_json::Map<String, Value> = urls
            .into_iter()
            .map(|url| {
                let row = self.latest.get(url).map_or(Value::Null, |(room, at)| {
                    let mut row = room.clone();
                    row.insert(
                        "age_s".into(),
                        json!(now.saturating_duration_since(*at).as_secs_f64()),
                    );
                    Value::Object(row)
                });
                (url.clone(), row)
            })
            .collect();
        Value::Object(rows)
    }
}

/// The copy price lambda_P of one candidate under the pull price phi and
/// whether the floor raised it: max(floor, lambda_q x phi_P); a P without
/// host_room data (`view` None) pays max(floor, lambda_q).
fn phi_price(
    base: f64,
    floor: f64,
    view: Option<crate::scheduler::co_schedule::TradePhiView>,
) -> (f64, bool) {
    let raw = view.map_or(base, |v| base * v.phi);
    if raw < floor {
        (floor, true)
    } else {
        (raw, false)
    }
}

/// `--symmetric-pd-gwq-auto`: T_react of each P, the time from a release of the
/// queue to the first progress poll whose reply lists the released RID. A
/// time-weighted mean with time constant tau (each sample has weight 1 and
/// decays as exp(-age / tau)). Only a release to an idle P (no queued and
/// no launched work: no running pass) gives a sample, so a deeper queue does
/// not raise its own lead.
#[derive(Debug)]
struct GwqLeadBook {
    tau_s: f64,
    /// T_react of a P without a sample (s).
    fallback_s: f64,
    by_p: BTreeMap<String, GwqLeadEstimate>,
    /// The lead of each P at the last step: (lead_s, depth_tokens, R).
    last_step: BTreeMap<String, (f64, u64, f64)>,
    /// Releases marked for a sample, and the others.
    marked: u64,
    unmarked: u64,
    /// First sightings of marked releases (samples taken).
    samples: u64,
    /// The URL of the last release whose TRADE total tied with another P
    /// (the rotation of [`crate::scheduler::co_schedule::trade_tie_break`]).
    tie_last: Option<String>,
    /// T_react estimates dropped because their P joined the step pool again
    /// ([`CostBook::gwq_lead_restart`]).
    restarts: u64,
}

impl GwqLeadBook {
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
        let e = self.by_p.entry(url.to_string()).or_insert(GwqLeadEstimate {
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

/// The T_react estimate of one P ([`GwqLeadBook`]).
#[derive(Debug, Clone, Copy)]
struct GwqLeadEstimate {
    /// Decayed sum of the samples and of their weights, exact at `last_at`.
    sum: f64,
    weight: f64,
    samples: u64,
    last_at: Instant,
    last_sample_s: f64,
}

/// A release of the `--symmetric-pd-gwq-auto` queue, kept on its entry until the
/// first sighting of its RID.
#[derive(Debug, Clone, Copy)]
struct GwqLeadMark {
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
fn gwq_pool_candidates(
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

/// `--symmetric-pd-trace-session-moves`: the P of the last TRADE decision of
/// each session (by session key), for the `session_move` record.
#[derive(Debug, Default)]
struct SessionMoveBook {
    /// Session key -> (P URL, unix ms of the decision).
    last: HashMap<u64, (String, u64)>,
    decisions: u64,
    p_changes: u64,
}

/// Trace-only extras of one reservation ([`CostBook::reserve_locked`]).
#[derive(Debug, Default)]
struct ReserveExtras<'a> {
    /// Written as the `gwq` object of the event when the reservation is not
    /// a release of the queue (a bound request of `--symmetric-pd-gwq-auto`).
    gwq_note: Option<Value>,
    /// The session identity of the request, for the `session_move` record.
    session_label: Option<&'a str>,
}

/// `--symmetric-pd-gwq-auto` part of one step ([`CostBook::gwq_step`]): the
/// depth of each P is the lead rule, depth = floor(C + R x T_react) + 1, so
/// that P has room while its queued work W x R <= lead x R, with lead = C / R
/// + T_react (R of the step, T_react of [`GwqLeadBook`]).
#[derive(Debug, Clone, Copy)]
pub struct GwqAutoStep<'a> {
    /// The engine chunk C of each P (pool order) and its source
    /// (`server_info` or `default`).
    pub chunks: &'a [(u64, &'static str)],
}

/// The lead rule of one P in one `--symmetric-pd-gwq-auto` step.
#[derive(Debug, Clone, Copy)]
struct GwqLeadRow {
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

impl GwqLeadRow {
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

pub struct CostBook {
    started: Instant,
    remaining_work: bool,
    restore_cost_divisor: Option<u64>,
    // Written as `policy` on every trace event when set; the co-schedule
    // policy leaves it unset, so its trace keeps the old fields.
    trace_policy: Option<&'static str>,
    /// The unknown expiry (see the module docs); None = never.
    unknown_expire: Option<std::time::Duration>,
    state: Mutex<State>,
    /// Called outside the lock after each terminal status change (the
    /// global waiting queue wakes its release pass). Set at most once.
    on_terminal: std::sync::OnceLock<Box<dyn Fn() + Send + Sync>>,
}

impl std::fmt::Debug for CostBook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CostBook").finish_non_exhaustive()
    }
}

impl Drop for CostBook {
    /// An async trace: write and flush every queued row, then stop the
    /// writer (a synchronous trace is flushed at each row already).
    fn drop(&mut self) {
        if let Some(TraceSink::Async(trace)) = self.state.get_mut().trace.as_mut() {
            trace.close();
        }
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
    restore_tokens: u64,
    pending_restore_before: u128,
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
            "request_restore_tokens": self.restore_tokens,
            "pending_restore_before": self.pending_restore_before.to_string(),
        })
    }
}

/// The copy price and rate `auto` parts of one TRADE decision: the resolved
/// parameters, how they were resolved, the measurements, and per P (URL, its
/// in-flight requests, its estimates).
type TradeAutoParts = (
    crate::scheduler::co_schedule::TradeParams,
    crate::scheduler::co_schedule::TradeAutoResolution,
    crate::scheduler::co_schedule::TradeAutoInputs,
    Vec<(
        String,
        Option<u64>,
        Option<crate::scheduler::co_schedule::TradeAutoWorkerView>,
    )>,
);

/// Pull price phi of one decision ([`CostBook::phi_prices`]): the base
/// lambda_q, its source, tau, the floor min, and phi of each P (pool order).
type PhiPrices = (
    f64,
    &'static str,
    f64,
    f64,
    Vec<Option<crate::scheduler::co_schedule::TradePhiView>>,
);

/// The request-independent part of one TRADE decision
/// ([`CostBook::trade_ctx`]).
struct TradeDecisionCtx {
    decided_at: Instant,
    /// The HOL term is on (its record is pruned to the window).
    hol: bool,
    /// The parameters of this decision (copy price and rate resolved).
    params: crate::scheduler::co_schedule::TradeParams,
    /// Copy price or rate `auto` only.
    auto: Option<TradeAutoParts>,
}

/// One P of a global-waiting-queue release, for the candidate rows.
#[derive(Debug, Clone, Copy)]
struct GwqRowNote {
    /// The held work ahead of the released request on this P.
    virtual_tokens: u128,
    depth: u64,
    room: bool,
    fresh: bool,
    /// `room = predicted` with a usable forecast: the P's backlog term is
    /// R x drain (tokens) instead of the ledger backlog.
    backlog_override: Option<u128>,
    /// `room = predicted`: the P's predicted drain (s), when usable.
    drain_s: Option<f64>,
}

/// A release of the global waiting queue ([`CostBook::gwq_step`]).
struct GwqReserve {
    ctx: TradeDecisionCtx,
    rows: BTreeMap<String, GwqRowNote>,
    /// The `gwq` object of the reserve event.
    record: Value,
    /// `--symmetric-pd-gwq-auto`: exact TRADE ties are broken as the walk breaks
    /// them ([`crate::scheduler::co_schedule::trade_tie_break`]).
    tie_break: bool,
}

/// One held request of the global waiting queue, as [`CostBook::gwq_step`]
/// sees it.
#[derive(Debug, Clone, Copy)]
pub struct GwqHeld<'a> {
    pub input_tokens: u64,
    pub rid: Option<&'a str>,
    /// Its TRADE candidates, one per P of the step (pool order).
    pub candidates: &'a [Candidate],
    /// When it entered the queue.
    pub arrived: Instant,
    pub arrived_unix_ms: u64,
    /// Its session key (copy price `auto`).
    pub session: Option<u64>,
    /// Its session identity (`--symmetric-pd-trace-session-moves`); None
    /// otherwise.
    pub session_label: Option<&'a str>,
    /// Its plain TRADE choice at the first step that saw it (None before).
    pub p_arrival: Option<&'a str>,
    /// The step belongs to the pass that the arrival of this request
    /// started: a release now is recorded with `held = false`.
    pub arrival_pass: bool,
}

/// The pool and the settings of one global-waiting-queue step.
#[derive(Debug, Clone, Copy)]
pub struct GwqStepArgs<'a> {
    pub mode: crate::global_queue::GlobalQueueMode,
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
    /// `--symmetric-pd-gwq-auto` (None: the depth above and `pass_tokens`): the
    /// lead rule with the engine chunk of each P.
    pub auto: Option<GwqAutoStep<'a>>,
    pub now: Instant,
    pub pass_id: u64,
    pub trigger: &'a str,
    pub triggers: &'a str,
    /// `--co-schedule-global-queue-room predicted`: the E3 inputs of a cold
    /// one-chunk probe per P (built before the lock, as for a `wait = e3`
    /// decision). None: the token rule for every P.
    pub e3: Option<&'a crate::trade_e3::TradeE3Context>,
    /// `room = predicted`: the lead (s) that a P's predicted drain must not
    /// exceed for room.
    pub lead_s: f64,
    /// `--symmetric-pd-pull-from-any-holder`: the registered workers of any role
    /// at this step (breaker allowing when the breaker is on); a held request
    /// keeps a pull hint whose source is one of them (or in the pool). Empty:
    /// only pool sources are kept.
    pub sources: &'a [String],
}

/// The input tokens of the cold probe of `--co-schedule-global-queue-room
/// predicted`: one engine pass, no cached prefix. Its E3 `wait_s` is the
/// time until the P can start a new pass for a released request.
pub const GWQ_PROBE_TOKENS: u64 = crate::global_queue::CHUNK_TOKENS;

/// The cold probe candidate of one P ([`GWQ_PROBE_TOKENS`]).
pub fn gwq_probe_candidate(url: &str) -> Candidate {
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
fn gwq_probe_drain(probe: &E3Probe) -> (Option<f64>, Option<&'static str>) {
    let wait = probe.out.as_ref().and_then(|o| o.wait_s);
    match (probe.fallback, wait) {
        (None | Some("no_bound"), Some(w)) if w.is_finite() => (Some(w.max(0.0)), None),
        (Some(reason), _) => (None, Some(reason)),
        (None, _) => (None, Some("no_forecast")),
    }
}

/// R x `drain_s` in tokens (rounded; 0 for a negative or non-finite value).
fn gwq_drain_tokens(drain_s: f64, rate_tokens_per_s: f64) -> u128 {
    let tokens = (drain_s * rate_tokens_per_s).round();
    if tokens.is_finite() && tokens > 0.0 {
        tokens as u128
    } else {
        0
    }
}

/// What one [`CostBook::gwq_step`] did.
#[derive(Debug)]
pub struct GwqStep {
    /// The plain TRADE choice of each held request (index into the URLs).
    pub plain: Vec<usize>,
    /// The uncached tokens of the held requests on their plain choice.
    pub queue_tokens: u128,
    /// None: nothing released. Else the held request (index into `held`)
    /// that left the queue: reserved (`Ok`), or refused by the book (`Err`;
    /// for example a reused RID).
    pub release: Option<(usize, Result<GwqReleased, String>)>,
}

/// A reserved global-waiting-queue release.
#[derive(Debug)]
pub struct GwqReleased {
    pub candidate: Candidate,
    pub reservation: Reservation,
    /// The `trade_decision` of the reserve event.
    pub decision: Value,
    /// The `gwq` object of the reserve event.
    pub record: Value,
    pub release: crate::global_queue::GwqRelease,
    pub starved: bool,
    /// The P at release differs from the plain choice at arrival.
    pub late_bound: bool,
}

/// One P of the `gwq_stats` event ([`CostBook::gwq_rows`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GwqRowStat {
    /// The queued work ([`crate::global_queue::queued_tokens`]).
    pub queued: u128,
    /// Its parts: the not-yet-launched uncached tokens (without the unknown
    /// reservations) and the remaining work of the launched requests.
    pub unlaunched: u128,
    pub launched: u128,
    /// Milliseconds since the poll of the last applied progress snapshot.
    pub view_age_ms: Option<u64>,
}

impl CostBook {
    /// Creates a fresh trace; existing paths are never appended or truncated.
    pub fn new(trace_path: Option<&Path>) -> io::Result<Arc<Self>> {
        Self::create(trace_path, false, None, None, None)
    }

    pub fn new_uncached(trace_path: Option<&Path>) -> io::Result<Arc<Self>> {
        Self::create(trace_path, true, None, None, None)
    }

    /// Same ledger as [`Self::new_uncached`]; every trace event also carries
    /// `"policy": policy`, so a reader can tell which router policy wrote it.
    /// `unknown_expire`: an unknown entry expires after this long without
    /// engine progress (see the module docs); None keeps it until a terminal
    /// result, as the other constructors do.
    pub fn new_uncached_with_policy(
        trace_path: Option<&Path>,
        policy: &'static str,
        unknown_expire: Option<std::time::Duration>,
    ) -> io::Result<Arc<Self>> {
        Self::create(trace_path, true, None, Some(policy), unknown_expire)
    }

    /// Charge D * remaining uncached tokens + outstanding restore tokens.
    /// D is a measured relative cost, not a dispatch threshold or admission cap.
    pub fn new_restore_aware(trace_path: Option<&Path>, divisor: u64) -> io::Result<Arc<Self>> {
        if divisor == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "restore divisor must be positive",
            ));
        }
        Self::create(trace_path, true, Some(divisor), None, None)
    }

    fn create(
        trace_path: Option<&Path>,
        remaining_work: bool,
        restore_cost_divisor: Option<u64>,
        trace_policy: Option<&'static str>,
        unknown_expire: Option<std::time::Duration>,
    ) -> io::Result<Arc<Self>> {
        let trace = trace_path
            .map(|path| OpenOptions::new().write(true).create_new(true).open(path))
            .transpose()?
            .map(|file| TraceSink::Sync(BufWriter::new(file)));
        Ok(Arc::new(Self {
            started: Instant::now(),
            remaining_work,
            restore_cost_divisor,
            trace_policy,
            unknown_expire,
            on_terminal: std::sync::OnceLock::new(),
            state: Mutex::new(State {
                next_id: 1,
                event_sequence: 0,
                entries: BTreeMap::new(),
                active_by_rid: BTreeMap::new(),
                used_rids: BTreeSet::new(),
                progress: BTreeMap::new(),
                trade_wait_fallbacks: BTreeMap::new(),
                trade_arrivals: Default::default(),
                trade_auto: None,
                olp: None,
                unknown: BTreeSet::new(),
                prefill_reader_gone: 0,
                expired_by_rid: BTreeMap::new(),
                expired_listed_again: 0,
                host_room: None,
                gwq_lead: None,
                session_moves: None,
                recent_finished: VecDeque::new(),
                e3_push_sources: BTreeMap::new(),
                e3_push_fallbacks: BTreeMap::new(),
                workers: BTreeMap::new(),
                healthy_candidates: Vec::new(),
                trace,
                io_failure: None,
            }),
        }))
    }

    fn remaining_cost(&self, uncached: u64, restore: u64) -> u128 {
        u128::from(uncached) * u128::from(self.restore_cost_divisor.unwrap_or(1))
            + u128::from(restore)
    }

    fn cost_unit(&self) -> &'static str {
        if self.restore_cost_divisor.is_some() {
            "weighted_uncached_and_restore_tokens"
        } else if self.remaining_work {
            "uncached_tokens"
        } else {
            "token_squared"
        }
    }

    /// Select and reserve atomically against the currently routable candidates.
    pub fn reserve(
        self: &Arc<Self>,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
    ) -> Result<(Candidate, Reservation), String> {
        self.reserve_placement(input_tokens, rid, candidates, Placement::CostSpread)
    }

    /// Keep the highest-hit owner (GPU residency breaks total-hit ties),
    /// unless another worker saves more pending uncached tokens than margin.
    /// This shares the atomic, P-owned lifecycle with the original cost mode.
    /// Tokens measure outstanding work, not predicted service milliseconds.
    pub fn reserve_locality(
        self: &Arc<Self>,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        margin_tokens: u64,
    ) -> Result<(Candidate, Reservation), String> {
        self.reserve_placement(
            input_tokens,
            rid,
            candidates,
            Placement::LocalityMargin(margin_tokens),
        )
    }

    /// The threshold applies to existing owner work, excluding this arrival.
    /// It never limits admission: even an entirely busy pool dispatches now.
    pub fn reserve_uncached(
        self: &Arc<Self>,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        owner_threshold: u64,
    ) -> Result<(Candidate, Reservation), String> {
        self.reserve_uncached_with_margin(input_tokens, rid, candidates, owner_threshold, 0)
    }

    pub fn reserve_uncached_with_margin(
        self: &Arc<Self>,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        owner_threshold: u64,
        detour_margin: u64,
    ) -> Result<(Candidate, Reservation), String> {
        self.reserve_placement(
            input_tokens,
            rid,
            candidates,
            Placement::UncachedThreshold(owner_threshold, detour_margin),
        )
    }

    /// TRADE placement on the uncached ledger: argmin over candidates of
    /// backlog + own uncached + unhidden pull time + copy price. The caller
    /// keeps a hint only for a pull of at least the pull floor. Returns the
    /// per-candidate decision record that the reserve trace event also holds.
    pub fn reserve_trade(
        self: &Arc<Self>,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        params: crate::scheduler::co_schedule::TradeParams,
    ) -> Result<(Candidate, Reservation, Value), String> {
        self.reserve_trade_eft(input_tokens, rid, candidates, params, None)
    }

    /// [`Self::reserve_trade`] with the inputs of `--co-schedule-trade-wait
    /// eft`. `eft` is None for the ledger wait; with `wait = eft` and no
    /// context every P falls back to the ledger (reason `no_model`).
    pub fn reserve_trade_eft(
        self: &Arc<Self>,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        params: crate::scheduler::co_schedule::TradeParams,
        eft: Option<&TradeEftContext>,
    ) -> Result<(Candidate, Reservation, Value), String> {
        self.reserve_trade_session(input_tokens, rid, candidates, params, eft, None)
    }

    /// [`Self::reserve_trade_eft`] with the request's session identity
    /// ([`crate::scheduler::co_schedule::trade_session_key`]). The session
    /// counts in N_active of the copy price `auto`; with both TRADE modes
    /// fixed it is not used.
    pub fn reserve_trade_session(
        self: &Arc<Self>,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        params: crate::scheduler::co_schedule::TradeParams,
        eft: Option<&TradeEftContext>,
        session: Option<&str>,
    ) -> Result<(Candidate, Reservation, Value), String> {
        self.reserve_trade_session_noted(input_tokens, rid, candidates, params, eft, session, None)
    }

    /// [`Self::reserve_trade_session`] with `gwq_note`, written as the `gwq`
    /// object of the reserve event (`--symmetric-pd-gwq-auto`: a request that the
    /// queue does not hold).
    #[allow(clippy::too_many_arguments)]
    pub fn reserve_trade_session_noted(
        self: &Arc<Self>,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        params: crate::scheduler::co_schedule::TradeParams,
        eft: Option<&TradeEftContext>,
        session: Option<&str>,
        gwq_note: Option<Value>,
    ) -> Result<(Candidate, Reservation, Value), String> {
        let (candidate, reservation, decision) = self.reserve_placement_with_decision(
            input_tokens,
            rid,
            candidates,
            Placement::Trade(params),
            eft,
            session.map(crate::scheduler::co_schedule::trade_session_hash),
            ReserveExtras {
                gwq_note,
                session_label: session,
            },
            None,
        )?;
        Ok((
            candidate,
            reservation,
            decision.expect("trade placement records its decision"),
        ))
    }

    /// `--symmetric-pd-gwq-auto`: keep T_react of each P (EWMA time constant
    /// `tau_s`; `fallback_s` before the first sample of a P) for the lead of
    /// [`Self::gwq_step`] with [`GwqAutoStep`].
    pub fn enable_gwq_lead(&self, tau_s: f64, fallback_s: f64) {
        let mut state = self.state.lock();
        if state.gwq_lead.is_none() {
            state.gwq_lead = Some(GwqLeadBook {
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

    /// `--symmetric-pd-gwq-auto`: `url` joined the step pool again (a role switch
    /// back to prefill, a drain that ended, a breaker that closed): its
    /// T_react restarts from the fallback, and the releases marked for a
    /// sample on it are unmarked (their release time is from before). Returns
    /// the dropped estimate (s), None when it had none.
    pub fn gwq_lead_restart(&self, url: &str) -> Option<f64> {
        let mut state = self.state.lock();
        let State {
            gwq_lead, entries, ..
        } = &mut *state;
        let book = gwq_lead.as_mut()?;
        let dropped = book.by_p.remove(url).map(|e| e.sum / e.weight);
        book.last_step.remove(url);
        for entry in entries.values_mut().filter(|e| e.worker_url == url) {
            entry.gwq_lead_mark = None;
        }
        if dropped.is_some() {
            book.restarts += 1;
        }
        dropped
    }

    /// `--symmetric-pd-trace-session-moves`: every TRADE reserve event with a
    /// session gets the `session_move` record (the P of the session's last
    /// decision, the hits, the pull).
    pub fn enable_session_moves(&self) {
        let mut state = self.state.lock();
        if state.session_moves.is_none() {
            state.session_moves = Some(SessionMoveBook::default());
        }
    }

    /// `--symmetric-pd-loads-host-room`: keep each P's latest `host_room` section
    /// and trace it (snapshot and TRADE decisions, key `host_room_by_p`).
    pub fn enable_host_room(&self) {
        let mut state = self.state.lock();
        if state.host_room.is_none() {
            state.host_room = Some(HostRoomBook::default());
        }
    }

    /// `--co-schedule-trade-pull-price phi`: also keep phi of each P from
    /// the evicted counters of its host_room sections, with the EWMA time
    /// constant `tau_s` (the copy price auto tau). Enables the host_room
    /// book; the TRADE decisions of a `phi` placement read it.
    pub fn enable_host_room_phi(&self, tau_s: f64) {
        let mut state = self.state.lock();
        let book = state.host_room.get_or_insert_with(HostRoomBook::default);
        if book.phi.is_none() {
            book.phi = Some(PhiBook {
                tau_s,
                states: BTreeMap::new(),
                invalid: 0,
                rows: 0,
                rows_floored: 0,
                rows_zero: 0,
                decisions: 0,
                chosen_floored: 0,
            });
        }
    }

    /// The `host_room` section of `worker_url`'s progress reply to the poll
    /// sent at `sent_at`. An older reply than the kept one, a value that is
    /// not a JSON object, or a book without [`Self::enable_host_room`]
    /// changes nothing (the invalid value is counted). A section clears the
    /// P's rejected mark.
    pub fn note_host_room(&self, worker_url: &str, room: Value, sent_at: Instant) {
        let mut state = self.state.lock();
        let Some(book) = state.host_room.as_mut() else {
            return;
        };
        let Value::Object(room) = room else {
            book.invalid += 1;
            return;
        };
        book.rejected.remove(worker_url);
        if book
            .latest
            .get(worker_url)
            .is_some_and(|(_, at)| *at >= sent_at)
        {
            return;
        }
        if let Some(phi) = book.phi.as_mut() {
            // The cumulative evicted tokens per class (engine counters).
            let counter = |key: &str| room.get(key).and_then(Value::as_u64);
            let class0 = counter("evicted_class0_tokens");
            let class1 = counter("evicted_class1_tokens");
            match (class0, class1) {
                (Some(class0), Some(class1)) => {
                    let tau_s = phi.tau_s;
                    phi.states
                        .entry(worker_url.to_string())
                        .or_default()
                        .observe(sent_at, class0, class1, tau_s);
                }
                _ => phi.invalid += 1,
            }
        }
        book.latest.insert(worker_url.to_string(), (room, sent_at));
    }

    /// The engine of `worker_url` accepted `include=...,host_room` but sent
    /// no section (no session retention or no host pool). Returns true for
    /// the first such reply of that P (the caller logs it).
    pub fn note_host_room_missing(&self, worker_url: &str) -> bool {
        let mut state = self.state.lock();
        let Some(book) = state.host_room.as_mut() else {
            return false;
        };
        let count = book.missing.entry(worker_url.to_string()).or_default();
        *count += 1;
        *count == 1
    }

    /// The engine of `worker_url` rejected `include=...,host_room`. Returns
    /// true for the first rejection of that P (the caller logs it).
    pub fn note_host_room_rejected(&self, worker_url: &str) -> bool {
        self.state
            .lock()
            .host_room
            .as_mut()
            .is_some_and(|book| book.rejected.insert(worker_url.to_string()))
    }

    /// The client ended `session` (`POST /kv_sessions/end`): it leaves
    /// N_active of the copy price `auto` at once. Returns whether the auto
    /// state counted it (false without auto decisions).
    pub fn trade_auto_end_session(&self, session: &str) -> bool {
        let key = crate::scheduler::co_schedule::trade_session_hash(session);
        self.state
            .lock()
            .trade_auto
            .as_mut()
            .is_some_and(|auto| auto.session_end(key))
    }

    /// `--co-schedule-trade-wait olp` or `--co-schedule-trade-olp-shadow`:
    /// the online latency predictor of this book ([`crate::olp`]), fed by the
    /// applied progress replies and the P-leg answers and used (shadow:
    /// traced only) by the TRADE decisions. Without this call the book has no
    /// OLP state; a `wait = olp` decision then creates one with the defaults.
    pub fn enable_olp(&self, config: crate::olp::OlpConfig) {
        Self::olp_init(&mut self.state.lock(), config);
    }

    /// OLP `verify` (tests and the offline replay): the incremental queue's
    /// comparison with the row walk so far; None without OLP state.
    #[doc(hidden)]
    pub fn olp_verify_stats(&self) -> Option<crate::olp::VerifyStats> {
        self.state.lock().olp.as_ref().map(|o| o.verify_stats())
    }

    /// `--co-schedule-cost-trace-async`: from now on the trace rows go to a
    /// background writer thread (see [`AsyncTrace`]): the events keep their
    /// order and bytes, but the write and flush leave the cost-book lock.
    /// Called before serving (the cost function does it at construction).
    /// Nothing changes without a trace or when it is already async.
    pub fn enable_async_trace(&self) -> io::Result<()> {
        let mut state = self.state.lock();
        if !matches!(state.trace, Some(TraceSink::Sync(_))) {
            return Ok(());
        }
        let Some(TraceSink::Sync(writer)) = state.trace.take() else {
            unreachable!("checked above");
        };
        let started = writer
            .into_inner()
            .map_err(|e| e.into_error())
            .and_then(AsyncTrace::start);
        match started {
            Ok(trace) => {
                state.trace = Some(TraceSink::Async(trace));
                Ok(())
            }
            Err(error) => {
                state.io_failure = Some(error.to_string());
                Err(error)
            }
        }
    }

    /// Every trace row emitted so far is written and flushed (an async
    /// trace waits for its writer; tests and the offline replay).
    pub fn sync_trace(&self) -> io::Result<()> {
        let answer = {
            let mut state = self.state.lock();
            match state.trace.as_mut() {
                None => return Ok(()),
                Some(TraceSink::Sync(writer)) => return writer.flush(),
                Some(TraceSink::Async(trace)) => {
                    let (ack, answer) = std::sync::mpsc::sync_channel(1);
                    trace.send(TraceMsg::Sync(ack))?;
                    answer
                }
            }
        };
        answer
            .recv()
            .map_err(|_| io::Error::other("cost trace writer failed before the sync"))
    }

    /// New OLP state with the live index of the active entries.
    fn olp_init(state: &mut State, config: crate::olp::OlpConfig) {
        let mut olp = crate::olp::OlpState::new(config);
        for entry in state.entries.values().filter(|e| e.status.active()) {
            olp.live_insert(&entry.worker_url, entry.id, olp_row(entry));
        }
        state.olp = Some(olp);
    }

    /// Seconds from the book's start to `at` (the OLP clock; 0 before it).
    fn seconds(&self, at: Instant) -> f64 {
        at.saturating_duration_since(self.started).as_secs_f64()
    }

    /// [`Self::reserve_trade`] with the inputs of `--co-schedule-trade-wait
    /// e3` ([`crate::trade_e3::TradeE3Context`], built before this lock).
    /// With `wait = e3` and no context every P falls back to the ledger
    /// terms (reason `no_context`) and the owner is kept.
    pub fn reserve_trade_e3(
        self: &Arc<Self>,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        params: crate::scheduler::co_schedule::TradeParams,
        e3: Option<&crate::trade_e3::TradeE3Context>,
    ) -> Result<(Candidate, Reservation, Value), String> {
        self.reserve_trade_e3_session(input_tokens, rid, candidates, params, e3, None)
    }

    /// [`Self::reserve_trade_e3`] with the request's session identity
    /// ([`crate::scheduler::co_schedule::trade_session_key`]) for N_active
    /// of the copy price `auto` (as [`Self::reserve_trade_session`]).
    pub fn reserve_trade_e3_session(
        self: &Arc<Self>,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        params: crate::scheduler::co_schedule::TradeParams,
        e3: Option<&crate::trade_e3::TradeE3Context>,
        session: Option<&str>,
    ) -> Result<(Candidate, Reservation, Value), String> {
        self.reserve_trade_e3_session_noted(
            input_tokens,
            rid,
            candidates,
            params,
            e3,
            session,
            None,
        )
    }

    /// [`Self::reserve_trade_e3_session`] with `gwq_note`, written as the
    /// `gwq` object of the reserve event (as
    /// [`Self::reserve_trade_session_noted`]; the symmetric_pd cost function with
    /// `--co-schedule-trade-wait e3`).
    #[allow(clippy::too_many_arguments)]
    pub fn reserve_trade_e3_session_noted(
        self: &Arc<Self>,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        params: crate::scheduler::co_schedule::TradeParams,
        e3: Option<&crate::trade_e3::TradeE3Context>,
        session: Option<&str>,
        gwq_note: Option<Value>,
    ) -> Result<(Candidate, Reservation, Value), String> {
        let (candidate, reservation, decision) = self.reserve_placement_with_decision(
            input_tokens,
            rid,
            candidates,
            Placement::Trade(params),
            None,
            session.map(crate::scheduler::co_schedule::trade_session_hash),
            ReserveExtras {
                gwq_note,
                session_label: session,
            },
            e3,
        )?;
        Ok((
            candidate,
            reservation,
            decision.expect("trade placement records its decision"),
        ))
    }

    /// Write one event (for example a `wait = e3` shadow or label line) to
    /// the cost trace, with the trace's sequence number and timestamps.
    pub fn emit_event(&self, event: Value) -> Result<(), String> {
        let mut state = self.state.lock();
        self.emit(&mut state, event)
    }

    /// The `wait = e3` forecast of one candidate, under the lock.
    ///
    /// The snapshot time is t0 = poll receipt - the engine's `age_s`;
    /// the view age (decision - poll send + `age_s`) must not exceed
    /// `eft_max_view_age_s`. With the state push on, the P's newest pushed
    /// state replaces the polled snapshot when [`push_pick`] accepts it:
    /// t0 = its state time, view age = decision - t0, and the
    /// bound comes from the push table. The queue holds the snapshot's
    /// waiting and bootstrap rows and the router rows: entries of this P
    /// that the snapshot does not show ([`crate::trade_e3::E3View::shown`])
    /// and that are reserved or dispatched, or finished but decided after
    /// t0. An entry that a progress poll received before t0 listed is left
    /// out: it launched before t0, so a snapshot without it means it is
    /// done. The null-prefix rule takes an entry's own estimate.
    fn probe_trade_e3(
        state: &State,
        e3: Option<&crate::trade_e3::TradeE3Context>,
        candidate: &Candidate,
        decided_at: Instant,
        params: &crate::scheduler::co_schedule::TradeParams,
        ids: &[u64],
    ) -> E3Probe {
        let mut probe = E3Probe::default();
        let Some(ctx) = e3 else {
            probe.fallback = Some("no_context");
            return probe;
        };
        let Some(prep) = ctx.candidates.get(&candidate.worker_url) else {
            probe.fallback = Some("no_context");
            return probe;
        };
        probe.prep = Some(prep.clone());
        let secs = |at: Instant| crate::trade_e3::secs_since(ctx.anchor, at);
        let d = secs(decided_at);
        let polled = prep
            .snapshot
            .as_ref()
            .and_then(|s| s.e3.as_ref().map(|v| (s, v)));
        let pushed = if ctx.cfg.push.is_some() {
            probe.push_on = true;
            probe.push_age_s = prep.push.as_ref().map(|p| d - p.t0_s);
            let poll_epoch = polled.and_then(|(_, v)| v.epoch.as_deref());
            match push_pick(
                prep.push.as_deref(),
                poll_epoch,
                d,
                params.eft_max_view_age_s,
            ) {
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
        if view_age > params.eft_max_view_age_s {
            probe.fallback = Some("view_too_old");
            return probe;
        }
        // The bound tables were fitted at one engine policy.
        if (view.chunk, view.page) != crate::trade_e3::BOUND_FIT_POLICY {
            probe.fallback = Some("policy_not_bound_fit");
            return probe;
        }
        let Some(fit_params) = prep.params.as_ref() else {
            probe.fallback = Some("no_fit");
            return probe;
        };
        // The simulated passes take the engine's budget and page.
        let fparams = &crate::e3::Params {
            chunk_tokens: Some(view.chunk),
            page_tokens: Some(view.page),
            ..fit_params.clone()
        };
        let rows: Vec<crate::trade_e3::RouterRow> = ids
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
                let (prefix, host) = e.e3_estimate.unwrap_or_else(|| {
                    let hit = crate::eft_cheap::floor_hit(e.expected_hit as i64, input, view.page);
                    (hit, 0)
                });
                Some(crate::trade_e3::RouterRow {
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
                .and_then(|e| e.e3_estimate)
        };
        let (input, notes) = crate::trade_e3::build_input(
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
        match crate::e3::forecast(&input, fparams) {
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
                probe.fallback = Some("e3_error");
                probe.error = Some(error.to_string());
            }
        }
        probe
    }

    /// B_w and S_w of one candidate for `wait = eft`, under the lock.
    ///
    /// With a usable snapshot (source `eft`): B_w = the drain of the
    /// snapshot plus the outstanding rows. These are the active entries on
    /// the P (FCFS by reservation id) that the snapshot does not show, at
    /// their current remaining uncached tokens. The snapshot shows a RID in
    /// its partial, waiting, bootstrap or in-flight request rows, or in the
    /// in-flight pass that R0 prices: a request whose last chunk is in that
    /// pass is in no row, but R0 already prices its work. An unknown entry
    /// (transport error) is left out: only the engine view tells whether it
    /// still has work, and the view prices it when it does.
    ///
    /// Without a usable snapshot (source `ledger_fallback`, with the reason):
    /// B_w = the step-model drain of all active entries on the P and no
    /// engine rows, that is, the ledger on the scale of the EFT terms. S_w is
    /// the step-model service in both cases (the page and chunk of the
    /// snapshot, else 64 and 8192). Without a model `terms` is None, and every
    /// P uses the ledger terms of the first version.
    fn probe_trade_wait(
        state: &State,
        eft: Option<&TradeEftContext>,
        candidate: &Candidate,
        input_tokens: u64,
        now: Instant,
        params: &crate::scheduler::co_schedule::TradeParams,
    ) -> WaitProbe {
        use crate::eft_cheap::{Shape, SnapshotView, View};
        let Some(ctx) = eft else {
            return WaitProbe::no_model();
        };
        let routing = ctx.views.get(&candidate.worker_url);
        let view = routing.map(|r| {
            View::at(
                &r.view,
                now.saturating_duration_since(r.poll_sent).as_secs_f64(),
            )
        });
        let fallback = match &view {
            None => Some("no_view"),
            Some(view) => view.unusable_reason.or_else(|| {
                view.age_s
                    .is_none_or(|age| age > params.eft_max_view_age_s)
                    .then_some("view_too_old")
            }),
        };
        let view_age_s = view.and_then(|v| v.age_s);
        let view_key = routing.map_or(Value::Null, |r| json!([r.view.epoch, r.view.sequence]));
        let (page, chunk) = routing.map_or(
            (
                crate::eft_cheap::DEFAULT_PAGE,
                crate::eft_cheap::DEFAULT_CHUNK,
            ),
            |r| (r.view.page, r.view.chunk),
        );
        let mut ids: Vec<u64> = state.active_by_rid.values().copied().collect();
        ids.sort_unstable();
        let entries = ids
            .iter()
            .map(|id| &state.entries[id])
            .filter(|e| e.worker_url == candidate.worker_url);
        let empty;
        let (view, rows, pass_rows, row_notes) = match (view, fallback) {
            (Some(view), None) => {
                let snap = view.snap;
                let mut pass_rows = 0usize;
                let inputs: Vec<_> = entries
                    .filter(|e| e.status != Status::Unknown)
                    .filter(|e| {
                        let in_pass = e.rid.as_ref().is_some_and(|rid| {
                            snap.pass_rids.contains(rid) && !snap.rids.contains(rid)
                        });
                        pass_rows += usize::from(in_pass);
                        !in_pass
                    })
                    .map(outstanding_input)
                    .collect();
                let rows = crate::eft_cheap::outstanding_rows(&inputs, &snap.rids);
                let notes = RowNotes {
                    degraded: snap.degraded.iter().cloned().collect(),
                    cold_rows: snap.cold_rows,
                    cold_tokens: snap.cold_tokens,
                    intent_skipped: snap.rows_intent_skipped,
                };
                (view, rows, pass_rows, Some(notes))
            }
            _ => {
                let inputs: Vec<_> = entries.map(outstanding_input).collect();
                empty = SnapshotView::empty(page, chunk);
                let rows = crate::eft_cheap::outstanding_rows(&inputs, &empty.rids);
                (View::at(&empty, 0.0), rows, 0, None)
            }
        };
        let drain = crate::eft_cheap::drain(&view, &rows, &ctx.model, &ctx.knobs);
        let n = input_tokens as i64;
        let pull_hit =
            (candidate.expected_hit > candidate.local_hit).then_some(candidate.expected_hit as i64);
        let (shape, _) = Shape::of(
            n,
            candidate.local_hit as i64,
            candidate.gpu_hit.unwrap_or(0) as i64,
            pull_hit,
            page,
        );
        let service_s = crate::eft_cheap::service_seconds(&ctx.model, shape, page, chunk);
        WaitProbe {
            terms: Some(crate::scheduler::co_schedule::TradeWaitTerms {
                wait_s: drain.d,
                service_s,
            }),
            fallback,
            view_age_s,
            view_key,
            outstanding_rows: rows.len(),
            outstanding_tokens: rows.iter().map(|r| r.u).sum(),
            pass_rows,
            drain: Some(drain),
            shape: Some(shape),
            row_notes,
        }
    }

    /// OLP at one TRADE decision (see [`crate::olp`]): the terms of every
    /// candidate in candidate order, the decision-level gate (applied in mode
    /// `olp`, traced in both modes) and, at the first decision of a step, its
    /// `olp_step` event. A candidate's rows are the live entries of its P
    /// (the per-worker index: reserved, dispatched or unknown) at their
    /// current remaining work; an acknowledged entry (first chunk launched)
    /// is priced as started. O(rows on the candidates + P workers); the
    /// ledger's entries are never scanned. With
    /// `--co-schedule-trade-olp-incremental` a candidate's queue comes from
    /// its integer sums, kept at every row change: O(1) per candidate, no
    /// row is walked (with `verify`, both are computed and compared; the
    /// walk's terms go back in `verify_terms` for the argmin check of the
    /// caller). None without OLP state.
    fn olp_decide(
        state: &mut State,
        mode: crate::olp::OlpMode,
        evaluations: &[Evaluation],
        input_tokens: u64,
        t_s: f64,
    ) -> Option<OlpEval> {
        let State { olp, entries, .. } = state;
        let olp = olp.as_mut()?;
        let entries = &*entries;
        let urls: Vec<&str> = evaluations
            .iter()
            .map(|e| e.candidate.worker_url.as_str())
            .collect();
        let ctx = olp.begin(&urls, t_s);
        let params = *olp.params();
        let incremental = olp.config().incremental;
        let verify = incremental && olp.config().verify;
        let walk = |olp: &crate::olp::OlpState, url: &str| {
            let rows = olp
                .live(url)
                .filter_map(|id| entries.get(&id))
                .filter(|entry| entry.status.active())
                .map(olp_row);
            params.queue(rows)
        };
        let mut walked: Vec<crate::olp::OlpQueue> = Vec::new();
        let preds: Vec<crate::olp::OlpPrediction> = evaluations
            .iter()
            .map(|e| {
                let url = e.candidate.worker_url.as_str();
                let queue = match olp.queue_sums(url) {
                    Some(sums) => {
                        let queue = params.queue_of(&sums);
                        if verify {
                            let reference = walk(olp, url);
                            olp.verify_stats_mut().candidate(&reference, &queue);
                            walked.push(reference);
                        }
                        queue
                    }
                    None => walk(olp, url),
                };
                let uncached = input_tokens - e.candidate.expected_hit;
                olp.predict(&ctx, url, queue, input_tokens, uncached)
            })
            .collect();
        // `verify`: the terms of the walked queues (the caller compares the
        // two OLP argmins).
        let verify_terms = verify.then(|| {
            evaluations
                .iter()
                .zip(walked)
                .map(|(e, queue)| {
                    let url = e.candidate.worker_url.as_str();
                    let uncached = input_tokens - e.candidate.expected_hit;
                    olp.predict(&ctx, url, queue, input_tokens, uncached)
                        .terms()
                })
                .collect()
        });
        let fallback = olp.gate(&ctx, &preds);
        olp.count_gate(fallback);
        let active = mode == crate::olp::OlpMode::Olp && fallback.is_none();
        let value = olp.decision_value(mode, active, fallback, &ctx);
        let step_event = ctx.new_step.then(|| olp.step_event(&ctx));
        Some(OlpEval {
            active,
            preds,
            value,
            step_event,
            t_s,
            verify_terms,
        })
    }

    fn reserve_placement(
        self: &Arc<Self>,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        placement: Placement,
    ) -> Result<(Candidate, Reservation), String> {
        self.reserve_placement_with_decision(
            input_tokens,
            rid,
            candidates,
            placement,
            None,
            None,
            ReserveExtras::default(),
            None,
        )
        .map(|(candidate, reservation, _)| (candidate, reservation))
    }

    /// `session`: the request's session key; used only by an auto TRADE
    /// decision (N_active) and the session-move record.
    #[allow(clippy::too_many_arguments)]
    fn reserve_placement_with_decision(
        self: &Arc<Self>,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        placement: Placement,
        eft: Option<&TradeEftContext>,
        session: Option<u64>,
        extras: ReserveExtras<'_>,
        e3: Option<&crate::trade_e3::TradeE3Context>,
    ) -> Result<(Candidate, Reservation, Option<Value>), String> {
        let urls = self.validate_placement(input_tokens, &candidates, placement)?;
        let mut state = self.state.lock();
        self.reserve_locked(
            &mut state,
            input_tokens,
            rid,
            candidates,
            urls,
            placement,
            eft,
            session,
            e3,
            None,
            extras,
        )
    }

    /// The checks of a placement that need no lock. Returns the candidate URLs.
    fn validate_placement(
        &self,
        input_tokens: u64,
        candidates: &[Candidate],
        placement: Placement,
    ) -> Result<BTreeSet<String>, String> {
        if self.remaining_work
            != matches!(
                placement,
                Placement::UncachedThreshold(..) | Placement::Trade(_)
            )
        {
            return Err("cost_balance: cannot mix cost units in one ledger".into());
        }
        if matches!(placement, Placement::Trade(_)) && self.restore_cost_divisor.is_some() {
            return Err("cost_balance: trade placement does not use a restore divisor".into());
        }
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
    /// [`Self::reserve_placement_with_decision`]). `gwq`: a release of the
    /// global waiting queue, with the decision context of its step (same
    /// lock), the virtual backlog per P and the `gwq` trace record. A TRADE
    /// placement then adds the virtual backlog to each candidate's backlog.
    #[allow(clippy::too_many_arguments)]
    fn reserve_locked(
        self: &Arc<Self>,
        state: &mut State,
        input_tokens: u64,
        rid: Option<String>,
        candidates: Vec<Candidate>,
        urls: BTreeSet<String>,
        placement: Placement,
        eft: Option<&TradeEftContext>,
        session: Option<u64>,
        e3: Option<&crate::trade_e3::TradeE3Context>,
        gwq: Option<GwqReserve>,
        extras: ReserveExtras<'_>,
    ) -> Result<(Candidate, Reservation, Option<Value>), String> {
        self.check_admission(state)?;
        if self.remaining_work {
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
            // Unknown GPU residency is conservatively charged as restore work.
            let restore_tokens = if self.restore_cost_divisor.is_some() {
                candidate
                    .expected_hit
                    .saturating_sub(candidate.gpu_hit.unwrap_or(0))
            } else {
                0
            };
            let uncached = input_tokens - candidate.expected_hit;
            let cost = if self.remaining_work {
                self.remaining_cost(uncached, restore_tokens)
            } else {
                u128::from(input_tokens) * u128::from(uncached)
            };
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
                restore_tokens,
                pending_restore_before: worker.pending_restore_tokens,
            });
        }
        let mut trade_decision = None;
        let mut trade_fallback_counts = None;
        // HOL on: the time of this TRADE placement, recorded at the commit.
        let mut trade_arrival_at: Option<Instant> = None;
        // Copy price or rate auto: the time of this decision; the entry then
        // feeds the auto state and its session starts at the commit.
        let mut trade_auto_at: Option<Instant> = None;
        // OLP: the chosen P's prediction, recorded at the commit for its label.
        let mut olp_label: Option<crate::olp::PendingLabel> = None;
        // Global waiting queue release: the `gwq` object of the event.
        let mut gwq_record: Option<Value> = None;
        // --symmetric-pd-gwq-auto: the P of a tie-broken release, the next
        // rotation start once the reservation is recorded.
        let mut gwq_tie_commit: Option<String> = None;
        // The decision time of a TRADE placement (the entry's decided_at).
        let mut trade_decided_at: Option<Instant> = None;
        // wait = e3: the chosen P's (prefix, host) estimate, for the entry.
        let mut trade_e3_estimate: Option<(i64, i64)> = None;
        // wait = e3 with the state push: the new cumulative source and
        // fallback counts, committed with the decision.
        let mut e3_push_counts: Option<(BTreeMap<String, u64>, BTreeMap<String, u64>)> = None;
        let locality_decision = if let Placement::Trade(configured) = placement {
            use crate::scheduler::co_schedule::{
                TradeLambdaMode, TradeParams, TradePullPrice, TradeWait,
            };
            // The request-independent part of the decision (HOL prune, copy
            // price and rate auto; [`Self::trade_ctx`]). A GWQ release brings
            // the context of its step, which ran under this same lock.
            let (ctx, mut gwq, tie_break) = match gwq {
                Some(GwqReserve {
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
                        Self::trade_ctx(state, configured, &urls, Instant::now()),
                        None,
                        false,
                    )
                }
            };
            let TradeDecisionCtx {
                decided_at,
                hol,
                params,
                auto,
            } = ctx;
            trade_decided_at = Some(decided_at);
            if hol {
                // Each candidate's a_w is read below, and this placement is
                // recorded at the commit.
                trade_arrival_at = Some(decided_at);
            }
            if auto.is_some() {
                trade_auto_at = Some(decided_at);
            }
            // Pull price phi: the base lambda_q (the auto price before the
            // clamp; the fallback price while it falls back) and phi of each
            // candidate (None: no host_room data, the P pays max(min, lambda_q)).
            let phi = {
                let urls: Vec<&str> = evaluations
                    .iter()
                    .map(|e| e.candidate.worker_url.as_str())
                    .collect();
                Self::phi_prices(state, configured, auto.as_ref(), &params, &urls, decided_at)
            };
            let probes: Vec<Option<WaitProbe>> = evaluations
                .iter()
                .map(|e| {
                    (params.wait == TradeWait::Eft).then(|| {
                        Self::probe_trade_wait(
                            &*state,
                            eft,
                            &e.candidate,
                            input_tokens,
                            decided_at,
                            &params,
                        )
                    })
                })
                .collect();
            // OLP (`wait = olp`, or the shadow readout with `wait = ledger`):
            // the terms of every candidate and the decision-level gate.
            // Without OLP state and with `wait != olp` this is None and
            // nothing below changes.
            let olp_mode = if params.wait == TradeWait::Olp {
                if state.olp.is_none() {
                    // A book built in code without `enable_olp`: the defaults.
                    Self::olp_init(state, crate::olp::OlpConfig::default());
                }
                Some(crate::olp::OlpMode::Olp)
            } else if params.wait == TradeWait::Ledger
                && state.olp.as_ref().is_some_and(|o| o.config().shadow)
            {
                Some(crate::olp::OlpMode::Shadow)
            } else {
                None
            };
            let decided_s = self.seconds(decided_at);
            let mut olp = olp_mode.and_then(|mode| {
                Self::olp_decide(state, mode, &evaluations, input_tokens, decided_s)
            });
            if let Some(event) = olp.as_mut().and_then(|o| o.step_event.take()) {
                // An emit error stops the admission at the reserve row below.
                let _ = self.emit(state, event);
            }
            let olp_active = olp.as_ref().is_some_and(|o| o.active);
            // `--co-schedule-trade-olp-trace-compact`: fewer OLP trace keys.
            let olp_compact = state.olp.as_ref().is_some_and(|o| o.config().compact_trace);
            // `wait = olp` with a closed gate: the ledger terms and ties, so the
            // decision is the cost function's, bit for bit.
            let select_params = if params.wait == TradeWait::Olp && !olp_active {
                TradeParams {
                    wait: TradeWait::Ledger,
                    ..params
                }
            } else {
                params
            };
            let e3_on = params.wait == TradeWait::E3;
            let mut e3_ids: Vec<u64> = Vec::new();
            if e3_on {
                // Finished entries older than the window are no router rows.
                while state.recent_finished.front().is_some_and(|(at, _)| {
                    decided_at.saturating_duration_since(*at).as_secs_f64() > RECENT_FINISHED_S
                }) {
                    state.recent_finished.pop_front();
                }
                e3_ids = state
                    .active_by_rid
                    .values()
                    .copied()
                    .chain(state.recent_finished.iter().map(|(_, id)| *id))
                    .collect();
                e3_ids.sort_unstable();
                e3_ids.dedup();
            }
            let e3_probes: Vec<Option<E3Probe>> = evaluations
                .iter()
                .map(|e| {
                    e3_on.then(|| {
                        Self::probe_trade_e3(state, e3, &e.candidate, decided_at, &params, &e3_ids)
                    })
                })
                .collect();
            let inputs: Vec<_> = evaluations
                .iter()
                .zip(&probes)
                .zip(&e3_probes)
                .enumerate()
                .map(|(i, ((e, probe), e3_probe))| {
                    let unacknowledged =
                        state.workers[&e.candidate.worker_url].pending_unacknowledged_tokens;
                    // A GWQ release adds the held work ahead of it (0 otherwise).
                    let note = gwq
                        .as_ref()
                        .and_then(|(notes, _)| notes.get(&e.candidate.worker_url));
                    let virtual_tokens = note.map_or(0, |note| note.virtual_tokens);
                    // `room = predicted`: the walk priced this P's wait as
                    // R x drain; the reservation uses the same term.
                    let ledger = if params.exclude_unlaunched {
                        e.pending_uncached_before.saturating_sub(unacknowledged)
                    } else {
                        e.pending_uncached_before
                    };
                    crate::scheduler::co_schedule::TradeInput {
                        worker: &e.candidate.worker_url,
                        backlog: note
                            .and_then(|note| note.backlog_override)
                            .unwrap_or(ledger)
                            .saturating_add(virtual_tokens),
                        local_hit: e.candidate.local_hit,
                        uncached: input_tokens - e.candidate.expected_hit,
                        pulled: e.candidate.expected_hit - e.candidate.local_hit,
                        // OLP active: its terms; else those of the E3 forecast
                        // (wait = e3) or of the EFT probe (None for the ledger).
                        wait: match (&olp, e3_probe) {
                            (Some(o), _) if o.active => Some(o.preds[i].terms()),
                            (_, Some(q)) => q.terms(),
                            _ => probe.as_ref().and_then(|p| p.terms),
                        },
                        // wait = e3: the E3 arrival-rate fit of the P when
                        // it exists, else the placement rate.
                        arrival_rate_per_s: if hol {
                            e3_probe
                                .as_ref()
                                .and_then(E3Probe::arrival_rate)
                                .unwrap_or_else(|| {
                                    state.trade_arrivals.rate_per_s(
                                        &e.candidate.worker_url,
                                        decided_at,
                                        params.hol_rate_window_s,
                                    )
                                })
                        } else {
                            0.0
                        },
                        copy_lambda: phi
                            .as_ref()
                            .map(|(base, _, _, floor, views)| phi_price(*base, *floor, views[i]).0),
                        bound_s: e3_probe.as_ref().and_then(E3Probe::usable_bound),
                    }
                })
                .collect();
            let mut selected = crate::scheduler::co_schedule::select_trade(&select_params, &inputs);
            if let (true, Some((notes, record))) = (tie_break, gwq.as_mut()) {
                // --symmetric-pd-gwq-auto: the walk's tie-break, with the same
                // room, outstanding requests and rotation of this step.
                let room: Vec<bool> = evaluations
                    .iter()
                    .map(|e| notes.get(&e.candidate.worker_url).is_some_and(|n| n.room))
                    .collect();
                let outstanding: Vec<u64> = evaluations.iter().map(|e| e.active_requests).collect();
                let last = state.gwq_lead.as_ref().and_then(|b| b.tie_last.clone());
                let outcome = crate::scheduler::co_schedule::trade_tie_break(
                    &selected,
                    &inputs,
                    &crate::scheduler::co_schedule::TradeTieBreak {
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
                    gwq_tie_commit = Some(evaluations[outcome.chosen].candidate.worker_url.clone());
                }
            }
            if let Some((_, record)) = gwq.as_mut() {
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
            let mut rows: Vec<Value> = inputs
                .iter()
                .zip(&selected.costs)
                .zip(&evaluations)
                .zip(&probes)
                .zip(&e3_probes)
                .enumerate()
                .map(|(i, ((((input, cost), e), probe), e3_probe))| {
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
                        "trade_cost": cost.total,
                    });
                    if let Some(q) = e3_probe {
                        if let (Value::Object(row), Value::Object(extra)) =
                            (&mut row, e3_row_value(q, cost))
                        {
                            row.extend(extra);
                        }
                        if hol {
                            row["arrival_rate_source"] = json!(if q.arrival_rate().is_some() {
                                "e3"
                            } else {
                                "placements"
                            });
                        }
                    } else if !params.is_original() {
                        // ledger: wait = ledger. eft: the P's usable snapshot.
                        // ledger_fallback: no usable snapshot (the router's
                        // rows priced by the step model), or no model (the
                        // first-version ledger terms).
                        let source = match probe {
                            _ if olp_active => "olp",
                            None => "ledger",
                            Some(p) if p.fallback.is_none() => "eft",
                            Some(_) => "ledger_fallback",
                        };
                        let notes = probe.as_ref().and_then(|p| p.row_notes.as_ref());
                        let extra = json!({
                            "wait_source": source,
                            "wait_fallback_reason": probe.as_ref().and_then(|p| p.fallback),
                            "wait_s": cost.wait_s,
                            "service_s": cost.service_s,
                            "dup_tokens": cost.dup_tokens,
                            "view_age_s": probe.as_ref().and_then(|p| p.view_age_s),
                            "view_key": probe.as_ref().map_or(Value::Null, |p| p.view_key.clone()),
                            "eft_r0_s": probe.as_ref().and_then(|p| p.drain.as_ref()).map(|d| d.r0),
                            "eft_drain_lead_s": probe.as_ref().and_then(|p| p.drain.as_ref()).map(|d| d.d_lead),
                            "eft_pull_backlog_s": probe.as_ref().and_then(|p| p.drain.as_ref()).map(|d| d.pb),
                            "eft_idle": probe.as_ref().and_then(|p| p.drain.as_ref()).map(|d| d.idle),
                            "eft_outstanding_rows": probe.as_ref().map(|p| p.outstanding_rows),
                            "eft_outstanding_tokens": probe.as_ref().map(|p| p.outstanding_tokens),
                            "eft_inflight_pass_rows": probe.as_ref().map(|p| p.pass_rows),
                            "eft_degraded": notes.map(|n| &n.degraded),
                            "eft_cold_rows": notes.map(|n| n.cold_rows),
                            "eft_cold_tokens": notes.map(|n| n.cold_tokens),
                            "eft_rows_intent_skipped": notes.map(|n| n.intent_skipped),
                            "eft_shape": probe.as_ref().and_then(|p| p.shape).map(|s| json!({
                                "u": s.u, "q": s.q, "h": s.h, "s": s.s,
                            })),
                        });
                        if let (Value::Object(row), Value::Object(extra)) = (&mut row, extra) {
                            row.extend(extra);
                        }
                    }
                    // OLP or its shadow readout: this P's terms.
                    if let Some(o) = &olp {
                        let extra = o.preds[i].trace_keys(olp_compact);
                        if let (Value::Object(row), Value::Object(extra)) = (&mut row, extra) {
                            row.extend(extra);
                        }
                    }
                    if hol {
                        row["arrival_rate_per_s"] = json!(cost.arrival_rate_per_s);
                        row["hol_tokens"] = json!(cost.hol_tokens);
                    }
                    if let Some(note) = gwq.as_ref().and_then(|(notes, _)| notes.get(input.worker)) {
                        row["gwq_virtual_tokens"] = json!(note.virtual_tokens.to_string());
                        row["gwq_depth_tokens"] = json!(note.depth);
                        row["gwq_room"] = json!(note.room);
                        row["gwq_view_fresh"] = json!(note.fresh);
                        if note.backlog_override.is_some() {
                            row["gwq_drain_s"] = json!(note.drain_s);
                            row["gwq_ledger_backlog_tokens"] =
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
                        row["auto_rate_tokens_per_s"] = json!(view.map(|v| v.rate_tokens_per_s));
                    }
                    row
                })
                .collect();
            if let Some((base, _, _, floor, views)) = &phi {
                // Pull price phi: each candidate's own copy price and phi.
                for ((row, input), view) in rows.iter_mut().zip(&inputs).zip(views) {
                    let (_, floored) = phi_price(*base, *floor, *view);
                    row["copy_lambda"] = json!(input.copy_lambda);
                    row["phi_floored"] = json!(floored);
                    row["phi_hat"] = json!(view.map(|v| v.phi));
                    row["phi_source"] = json!(view.map_or("no_host_room", |v| v.source));
                    row["phi_rate_class0"] = json!(view.map(|v| v.rate_class0));
                    row["phi_rate_class1"] = json!(view.map(|v| v.rate_class1));
                    row["phi_last_evict_age_s"] = json!(view.and_then(|v| v.last_evict_age_s));
                }
            }
            let mut decision = json!({
                "owner": evaluations[selected.owner].candidate.worker_url,
                "has_cached_owner": selected.has_owner,
                "chosen": evaluations[selected.chosen].candidate.worker_url,
                "detour": selected.chosen != selected.owner,
                "pulled_tokens": inputs[selected.chosen].pulled,
                "trade_cost": selected.costs[selected.chosen].total,
                "rate_tokens_per_s": params.rate_tokens_per_s,
                "pull_fixed_ms": params.pull_fixed_ms,
                "pull_ms_per_100k": params.pull_ms_per_100k,
                "copy_lambda": params.copy_lambda,
                "exclude_unlaunched": params.exclude_unlaunched,
                // Old key, kept for analysis scripts: true = full ledger.
                "count_inflight": !params.exclude_unlaunched,
                "candidates": rows,
            });
            if !params.is_original() {
                let mut counts = state.trade_wait_fallbacks.clone();
                let mut fallbacks = 0u64;
                let reasons = probes
                    .iter()
                    .flatten()
                    .filter_map(|p| p.fallback)
                    .chain(e3_probes.iter().flatten().filter_map(|q| q.fallback));
                for reason in reasons {
                    *counts.entry(reason.to_string()).or_default() += 1;
                    fallbacks += 1;
                }
                let extra = json!({
                    "wait": params.wait.as_str(),
                    "dup_charge": params.dup_charge.as_str(),
                    "margin_noise_k": params.margin_noise_k,
                    "eft_max_view_age_s": params.eft_max_view_age_s,
                    "eft_model_sha256": eft.map(|ctx| ctx.model.sha256.clone()),
                    "best": evaluations[selected.best].candidate.worker_url,
                    "margin_s": selected.margin_s,
                    "owner_gain_s": selected.owner_gain_s,
                    "margin_protected": selected.margin_protected,
                    "wait_s": selected.costs[selected.chosen].wait_s,
                    "service_s": selected.costs[selected.chosen].service_s,
                    "dup_tokens": selected.costs[selected.chosen].dup_tokens,
                    "wait_fallbacks": fallbacks,
                    "wait_fallback_counts": counts,
                });
                if let (Value::Object(decision), Value::Object(extra)) = (&mut decision, extra) {
                    decision.extend(extra);
                }
                trade_fallback_counts = Some(counts);
            }
            if hol {
                decision["hol_weight"] = json!(params.hol_weight);
                decision["hol_rate_window_s"] = json!(params.hol_rate_window_s);
            }
            // Copy price or rate auto: the measurements and how lambda and R
            // were set. `copy_lambda` and `rate_tokens_per_s` above are the
            // values this decision used (with the pull price phi, the price of
            // the chosen P; see below).
            if let Some((_, resolution, measured, _)) = &auto {
                let (mu, lambda_min, lambda_max) = match configured.copy_lambda_mode {
                    TradeLambdaMode::Auto { mu, min, max, .. } => (Some(mu), Some(min), Some(max)),
                    TradeLambdaMode::Fixed => (None, None, None),
                };
                let extra = json!({
                    "lambda_mode": resolution.lambda_mode,
                    "copy_lambda_used": resolution.copy_lambda_used,
                    "copy_lambda_fallback": configured.copy_lambda,
                    "lambda_raw": resolution.lambda_raw,
                    "lambda_mu": mu,
                    "lambda_min": lambda_min,
                    "lambda_max": lambda_max,
                    "q": resolution.q,
                    "q_mode": resolution.q_mode,
                    "busy": measured.busy,
                    "inflight_mean": measured.inflight,
                    "n_active": measured.n_active,
                    "n_p": measured.n_p,
                    "r_hat": measured.r_hat,
                    "rate_mode": resolution.rate_mode,
                    "rate_used": resolution.rate_used,
                    "rate_fallback": configured.rate_tokens_per_s,
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
            if let Some(o) = olp.as_mut() {
                // Both argmins for the A/B readout: the cost function's (no
                // wait terms, ledger ties) and OLP's (its terms and ties,
                // whatever the gate says). The real selection is one of them.
                let active = o.active;
                let alt_inputs: Vec<_> = inputs
                    .iter()
                    .zip(&o.preds)
                    .map(|(input, pred)| crate::scheduler::co_schedule::TradeInput {
                        wait: if active { None } else { Some(pred.terms()) },
                        ..*input
                    })
                    .collect();
                let alt_wait = if active {
                    TradeWait::Ledger
                } else {
                    TradeWait::Olp
                };
                let alt_params = TradeParams {
                    wait: alt_wait,
                    ..params
                };
                let alt = crate::scheduler::co_schedule::select_trade(&alt_params, &alt_inputs);
                let (ledger_i, olp_i) = if active {
                    (alt.chosen, selected.chosen)
                } else {
                    (selected.chosen, alt.chosen)
                };
                let mut value = std::mem::take(&mut o.value);
                value["ledger_chosen"] = json!(inputs[ledger_i].worker);
                value["olp_chosen"] = json!(inputs[olp_i].worker);
                decision["olp"] = value;
                // Incremental queue with `verify` (tests, offline replay):
                // the OLP argmin over the walked queues' terms must be OLP's.
                if let (Some(walked), Some(state_olp)) =
                    (o.verify_terms.as_ref(), state.olp.as_mut())
                {
                    let walk_inputs: Vec<_> = inputs
                        .iter()
                        .zip(walked)
                        .map(|(input, terms)| crate::scheduler::co_schedule::TradeInput {
                            wait: Some(*terms),
                            ..*input
                        })
                        .collect();
                    let olp_params = TradeParams {
                        wait: TradeWait::Olp,
                        ..params
                    };
                    let walk =
                        crate::scheduler::co_schedule::select_trade(&olp_params, &walk_inputs);
                    let stats = state_olp.verify_stats_mut();
                    stats.decisions += 1;
                    stats.olp_choice_mismatch += u64::from(walk.chosen != olp_i);
                }
            }
            // OLP: the chosen P's prediction and the cost function's own
            // latency proxy ((backlog + u) / R + exposed pull), for its label.
            olp_label = olp.as_ref().map(|o| {
                let input = &inputs[selected.chosen];
                let pred = &o.preds[selected.chosen];
                let pull_s = params.pull_seconds(input.pulled);
                let ledger = crate::scheduler::co_schedule::trade_cost_full(
                    &params,
                    input.backlog,
                    input.uncached,
                    input.pulled,
                    0,
                    None,
                    0.0,
                );
                crate::olp::PendingLabel {
                    t_decision: o.t_s,
                    url: input.worker.to_string(),
                    pred_olp_c: pred.completion_s(pull_s),
                    pred_olp_base: pred.completion_base_s(pull_s),
                    pred_cf_c: ledger.wait_s + ledger.service_s + ledger.exposed_pull_s,
                    risk: pred.risk,
                    q_model_s: pred.q_model_s,
                    own_model_s: pred.own_model_s,
                    pulled: input.pulled > 0,
                }
            });
            if let Some((base, base_source, tau_s, floor, views)) = &phi {
                // Pull price phi: the rows carry each candidate's copy price;
                // copy_lambda and copy_lambda_used are the price of the chosen
                // P, copy_lambda_auto the clamped auto lambda.
                let chosen = phi_price(*base, *floor, views[selected.chosen]);
                decision["copy_lambda"] = json!(chosen.0);
                decision["copy_lambda_auto"] = json!(params.copy_lambda);
                if decision.get("copy_lambda_used").is_some() {
                    decision["copy_lambda_used"] = json!(chosen.0);
                }
                decision["pull_price"] = json!(TradePullPrice::Phi.as_str());
                decision["phi_lambda_base"] = json!(base);
                decision["phi_base_source"] = json!(base_source);
                decision["phi_tau_s"] = json!(tau_s);
                decision["phi_lambda_min"] = json!(floor);
                decision["phi_floored"] = json!(chosen.1);
                // S3: count floored and zero prices from the row prices.
                if let Some(book) = state.host_room.as_mut().and_then(|b| b.phi.as_mut()) {
                    book.decisions += 1;
                    book.chosen_floored += u64::from(chosen.1);
                    for (input, view) in inputs.iter().zip(views) {
                        book.rows += 1;
                        book.rows_floored += u64::from(phi_price(*base, *floor, *view).1);
                        book.rows_zero += u64::from(input.copy_lambda == Some(0.0));
                    }
                }
            }
            if let Some(book) = &state.host_room {
                // --symmetric-pd-loads-host-room: the latest engine host_room of
                // every candidate (not a decision input).
                let urls = evaluations.iter().map(|e| &e.candidate.worker_url);
                decision["host_room_by_p"] = book.by_p(urls, decided_at);
            }
            if e3_on {
                let chosen = e3_probes[selected.chosen].as_ref();
                let forecasts = e3_probes
                    .iter()
                    .flatten()
                    .filter(|q| q.fallback.is_none())
                    .count();
                let extra = e3_decision_value(
                    &params,
                    &selected,
                    chosen,
                    forecasts,
                    e3,
                    decided_at,
                    evaluations.iter().zip(&e3_probes).map(|(e, q)| {
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
                if let Some(push) = e3.and_then(|ctx| ctx.cfg.push.as_ref()) {
                    // State push on: the push settings and, cumulative over
                    // decisions, the candidates per state source and per
                    // reason the pushed state was not used.
                    let mut sources = state.e3_push_sources.clone();
                    let mut fallbacks = state.e3_push_fallbacks.clone();
                    for q in e3_probes.iter().flatten() {
                        *sources
                            .entry(q.source.unwrap_or("none").to_string())
                            .or_default() += 1;
                        if let Some(reason) = q.push_fallback {
                            *fallbacks.entry(reason.to_string()).or_default() += 1;
                        }
                    }
                    decision["e3_push_port"] = json!(push.port);
                    decision["e3_push_bound_mode"] = json!(push.mode);
                    decision["e3_push_bound_level"] = json!(push.level);
                    decision["e3_state_source_counts"] = json!(sources);
                    decision["e3_push_fallback_counts"] = json!(fallbacks);
                    decision["e3_chosen"]["state_source"] = json!(chosen.and_then(|q| q.source));
                    e3_push_counts = Some((sources, fallbacks));
                }
                trade_e3_estimate = chosen
                    .and_then(|q| q.prep.as_ref())
                    .map(|p| (p.target.prefix, p.target.host));
            }
            drop(inputs);
            evaluations.swap(0, selected.chosen);
            trade_decision = Some(decision);
            gwq_record = gwq.map(|(_, record)| record);
            None
        } else if let Placement::UncachedThreshold(threshold, margin) = placement {
            let views: Vec<_> = evaluations
                .iter()
                .map(|e| crate::scheduler::co_schedule::Evaluation {
                    worker: &e.candidate.worker_url,
                    local_hit: e.candidate.local_hit,
                    gpu_hit: e.candidate.gpu_hit.unwrap_or(0),
                    before: e.before,
                    after: e.after,
                    spread: e.spread,
                    active_requests: e.active_requests,
                    pending_uncached_before: e.pending_uncached_before,
                })
                .collect();
            let selected = crate::scheduler::co_schedule::select(
                &views,
                threshold,
                self.remaining_cost(margin, 0),
                self.restore_cost_divisor.is_some(),
            );
            let crate::scheduler::co_schedule::Selection {
                owner,
                chosen,
                has_owner,
                keep_owner,
                margin_protected,
                saved_work,
            } = selected;
            let decision = json!({
                "owner": evaluations[owner].candidate.worker_url,
                "has_cached_owner": has_owner,
                "owner_existing_uncached_tokens": evaluations[owner].pending_uncached_before.to_string(),
                "owner_threshold_tokens": threshold,
                "owner_below_threshold": keep_owner,
                "detour_margin_tokens": margin,
                "margin_protected": margin_protected,
                "candidate_saved_work": saved_work.to_string(),
                "detour": chosen != owner,
            });
            evaluations.swap(0, chosen);
            Some(decision)
        } else if let Placement::LocalityMargin(margin) = placement {
            // min_by ordering: better cache first, then lower pending work.
            let owner = evaluations
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    b.candidate
                        .local_hit
                        .cmp(&a.candidate.local_hit)
                        .then_with(|| {
                            b.candidate
                                .gpu_hit
                                .unwrap_or(0)
                                .cmp(&a.candidate.gpu_hit.unwrap_or(0))
                        })
                        .then_with(|| a.pending_uncached_before.cmp(&b.pending_uncached_before))
                        .then_with(|| a.active_requests.cmp(&b.active_requests))
                        .then_with(|| a.candidate.worker_url.cmp(&b.candidate.worker_url))
                })
                .expect("nonempty candidates")
                .0;
            let least_work = evaluations
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.pending_uncached_after
                        .cmp(&b.pending_uncached_after)
                        .then_with(|| {
                            b.candidate
                                .gpu_hit
                                .unwrap_or(0)
                                .cmp(&a.candidate.gpu_hit.unwrap_or(0))
                        })
                        .then_with(|| b.candidate.local_hit.cmp(&a.candidate.local_hit))
                        .then_with(|| a.active_requests.cmp(&b.active_requests))
                        .then_with(|| a.candidate.worker_url.cmp(&b.candidate.worker_url))
                })
                .expect("nonempty candidates")
                .0;
            let saved = evaluations[owner]
                .pending_uncached_after
                .saturating_sub(evaluations[least_work].pending_uncached_after);
            let detour = saved > u128::from(margin);
            let decision = json!({
                "owner": evaluations[owner].candidate.worker_url,
                "least_work_worker": evaluations[least_work].candidate.worker_url,
                "owner_score_tokens": evaluations[owner].pending_uncached_after.to_string(),
                "least_work_score_tokens": evaluations[least_work].pending_uncached_after.to_string(),
                "saved_work_tokens": saved.to_string(),
                "margin_tokens": margin,
                "detour": detour,
            });
            evaluations.swap(0, if detour { least_work } else { owner });
            Some(decision)
        } else {
            evaluations.sort_by(|a, b| {
                a.spread
                    .cmp(&b.spread)
                    .then_with(|| b.candidate.expected_hit.cmp(&a.candidate.expected_hit))
                    .then_with(|| a.before.cmp(&b.before))
                    .then_with(|| a.active_requests.cmp(&b.active_requests))
                    .then_with(|| b.candidate.local_hit.cmp(&a.candidate.local_hit))
                    .then_with(|| a.candidate.worker_url.cmp(&b.candidate.worker_url))
            });
            None
        };
        // --symmetric-pd-trace-session-moves: the P of the session's last TRADE
        // decision and the hits of this one (the chosen P is first now).
        let session_move = match (&trade_decision, session, state.session_moves.as_ref()) {
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
            restore_tokens: winner.restore_tokens,
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
                tracked: trade_auto_at.is_some(),
                session: session.filter(|_| trade_auto_at.is_some()),
                in_flight: false,
                done: false,
                peak_tokens: input_tokens - winner.candidate.expected_hit,
            },
            unknown_since: None,
            last_progress_seen: None,
            gwq_lead_mark: None,
            decided_at: trade_decided_at.unwrap_or_else(Instant::now),
            ack_at: None,
            e3_estimate: trade_e3_estimate,
        };
        // Write before committing: an IO error must not hand the caller a
        // dispatchable reservation that was never recorded. The same lock
        // spans scoring, writing and committing, preventing stale decisions.
        let mut event = json!({
            "event": "reserve",
            "entry": entry.value(),
            "healthy_candidates": state.healthy_candidates,
            "candidate_scores": candidate_scores,
            "selection_policy": if self.restore_cost_divisor.is_some() { "restore_aware_finish" } else { placement.label() },
            "locality_decision": locality_decision,
        });
        if let Some(decision) = &trade_decision {
            event["trade_decision"] = decision.clone();
        }
        if let Some(record) = gwq_record {
            event["gwq"] = record;
        } else if let Some(note) = extras.gwq_note {
            event["gwq"] = note;
        }
        let p_changed = session_move.as_ref().map(|(changed, _)| *changed);
        if let Some((_, record)) = session_move {
            event["session_move"] = record;
        }
        self.emit(state, event)?;
        if let Some(counts) = trade_fallback_counts {
            state.trade_wait_fallbacks = counts;
        }
        if let (Some(url), Some(book)) = (gwq_tie_commit, state.gwq_lead.as_mut()) {
            book.tie_last = Some(url);
        }
        if let Some((sources, fallbacks)) = e3_push_counts {
            state.e3_push_sources = sources;
            state.e3_push_fallbacks = fallbacks;
        }
        let worker = state
            .workers
            .get_mut(&entry.worker_url)
            .expect("candidate registered");
        worker.cost = winner.after;
        worker.pending_uncached_tokens = winner.pending_uncached_after;
        worker.pending_restore_tokens += u128::from(winner.restore_tokens);
        worker.pending_unacknowledged_tokens += u128::from(entry.uncached_tokens);
        worker.active_requests += 1;
        state.next_id = next_id;
        if let Some(at) = trade_arrival_at {
            state.trade_arrivals.record(&entry.worker_url, at);
        }
        if let (Some(at), Some(key)) = (trade_auto_at, entry.auto.session) {
            if let Some(auto) = state.trade_auto.as_mut() {
                auto.session_started(key, at);
            }
        }
        if self.remaining_work {
            let rid = entry.rid.as_ref().expect("validated rid");
            state.active_by_rid.insert(rid.clone(), id);
            state.used_rids.insert(rid.clone());
        }
        if let Some(olp) = state.olp.as_mut() {
            // The live index of the OLP queue; an OLP decision until its label.
            olp.live_insert(&entry.worker_url, id, olp_row(&entry));
            if let Some(label) = olp_label {
                olp.record(id, label);
            }
        }
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
            trade_decision,
        ))
    }

    fn check_admission(&self, state: &State) -> Result<(), String> {
        if let Some(reason) = &state.io_failure {
            return Err(format!(
                "cost_balance: trace failed; admission stopped: {reason}"
            ));
        }
        if let Some(TraceSink::Async(trace)) = &state.trace {
            if let Some(reason) = trace.failed() {
                return Err(format!(
                    "cost_balance: trace failed; admission stopped: {reason}"
                ));
            }
        }
        if !state.unknown.is_empty() && !self.remaining_work {
            return Err(format!(
                "cost_balance: {} unresolved reservations; admission stopped",
                state.unknown.len()
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
        event["schema"] = json!("cost-balance.v1");
        if let Some(policy) = self.trace_policy {
            event["policy"] = json!(policy);
        }
        event["ts_unix_ms"] = json!(timestamp);
        event["ts_unix_s"] = json!(timestamp as f64 / 1000.0);
        event["elapsed_ms"] = json!(self.started.elapsed().as_millis() as u64);
        event["cost_unit"] = json!(self.cost_unit());
        event["restore_cost_divisor"] = json!(self.restore_cost_divisor);
        let result = match &mut state.trace {
            Some(TraceSink::Sync(trace)) => serde_json::to_writer(&mut *trace, &event)
                .map_err(io::Error::other)
                .and_then(|()| trace.write_all(b"\n"))
                .and_then(|()| trace.flush()),
            // Off the lock's critical path: the writer thread serializes,
            // writes and flushes.
            Some(TraceSink::Async(trace)) => trace.send(TraceMsg::Row(event)),
            None => Ok(()),
        };
        if let Err(error) = result {
            let reason = error.to_string();
            state.io_failure = Some(reason.clone());
            tracing::error!(target: "cost_balance", error = %reason,
                "cost trace failed; further admission is disabled");
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
        let old = state
            .entries
            .get(&id)
            .ok_or("cost_balance: unknown reservation ID")?
            .status;
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
            let (url, cost, uncached, restore, rid, acknowledged) = (
                entry.worker_url.clone(),
                entry.cost,
                entry.uncached_tokens,
                entry.restore_tokens,
                entry.rid.clone(),
                entry.acknowledged,
            );
            if self.remaining_work {
                let rid = rid.as_ref().expect("validated rid");
                state.active_by_rid.remove(rid);
                // Never sent to an engine, so no engine snapshot can list it:
                // a replan (e.g. the chosen worker began a role drain before
                // the prepared plan was installed) may reuse the same rid.
                if target == Status::RolledBack && old == Status::Reserved {
                    state.used_rids.remove(rid);
                }
            }
            let worker = state.workers.get_mut(&url).expect("entry worker exists");
            worker.cost = worker
                .cost
                .checked_sub(cost)
                .expect("cost ledger underflow");
            worker.pending_uncached_tokens = worker
                .pending_uncached_tokens
                .checked_sub(u128::from(uncached))
                .expect("pending token ledger underflow");
            worker.pending_restore_tokens = worker
                .pending_restore_tokens
                .checked_sub(u128::from(restore))
                .expect("restore ledger underflow");
            if !acknowledged {
                worker.pending_unacknowledged_tokens = worker
                    .pending_unacknowledged_tokens
                    .checked_sub(u128::from(uncached))
                    .expect("unacknowledged ledger underflow");
            }
            worker.active_requests = worker
                .active_requests
                .checked_sub(1)
                .expect("request ledger underflow");
            if target == Status::Finished {
                // wait = e3 router rows (bounded: older ones are dropped).
                let at = Instant::now();
                while state.recent_finished.front().is_some_and(|(t, _)| {
                    at.saturating_duration_since(*t).as_secs_f64() > RECENT_FINISHED_S
                }) {
                    state.recent_finished.pop_front();
                }
                state.recent_finished.push_back((at, id));
            }
        }
        if target == Status::Unknown {
            state.unknown.insert(id);
        } else if old == Status::Unknown {
            state.unknown.remove(&id);
        }
        // OLP: an ended entry leaves the live index; a P-leg answered with
        // success labels its OLP decision, any other end (also unknown) drops
        // it.
        if target == Status::Unknown || !target.active() {
            let State { olp, entries, .. } = &mut *state;
            if let Some(olp) = olp.as_mut() {
                if !target.active() {
                    olp.live_remove(&entries[&id].worker_url, id);
                }
                let success = target == Status::Finished && reason == PREFILL_SUCCESS_REASON;
                olp.resolve(id, self.seconds(Instant::now()), success);
            }
        }
        // TRADE auto: the P's in-flight requests and the session. An emit
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
                "unknown reservation expired without engine progress; its work leaves the ledger");
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
                state.expired_by_rid.insert(rid, (id, 0));
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
    /// (`wait = e3` leaves it out of the router rows of a snapshot taken
    /// after it).
    pub fn apply_progress_received(
        &self,
        worker_url: &str,
        progress: PrefillProgress,
        engine_load: Value,
        sent_at: Instant,
        received_at: Instant,
    ) -> Result<(), String> {
        if !self.remaining_work || progress.epoch.is_empty() {
            return Err("prefill progress requires uncached mode and an engine epoch".into());
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
        // TRADE auto: (id, remaining) of the tracked entries whose prefill has not ended.
        let mut auto_seen = Vec::new();
        // Every listed active entry (unknown ones too): the unknown expiry evidence.
        let mut listed = Vec::new();
        // (rid, id, remaining) of expired entries of this P listed again.
        let mut relisted = Vec::new();
        for (rid, remaining) in &progress.remaining {
            let Some(id) = state.active_by_rid.get(rid) else {
                if let Some((id, _)) = state.expired_by_rid.get(rid) {
                    if state.entries[id].worker_url == worker_url {
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
            // even before the first chunk finishes. Only a later reduction
            // proves a chunk retired; zero always means all work retired.
            let restore = if *remaining == 0
                || (entry.seen_engine_progress && *remaining < entry.uncached_tokens)
            {
                0
            } else {
                entry.restore_tokens
            };
            if *remaining != entry.uncached_tokens
                || (self.restore_cost_divisor.is_some() && !entry.seen_engine_progress)
            {
                updates.push((
                    *id,
                    entry.uncached_tokens,
                    *remaining,
                    entry.restore_tokens,
                    restore,
                    entry.cost,
                    self.remaining_cost(*remaining, restore),
                ));
            }
        }
        // Validate the entire update before emitting or mutating the ledger.
        let mut next_cost = state.workers.get(worker_url).map_or(0, |w| w.cost);
        for (_, _, _, _, _, old_cost, cost) in &updates {
            next_cost = next_cost
                .checked_sub(*old_cost)
                .and_then(|v| v.checked_add(*cost))
                .ok_or("prefill progress cost overflow or underflow")?;
        }
        // OLP: the chunk work of this reply, priced by the structural model:
        // decreases of entries listed before (a first listing may only
        // correct the cache hit) and first listings at 0 (all computed), at
        // h = input - old, with the restore while the entry had no update.
        let olp_chunks = state.olp.as_ref().map(|olp| {
            let params = olp.params();
            let mut work_s = 0.0;
            let mut tok = 0u64;
            for &(id, old, new, ..) in &updates {
                let entry = &state.entries[&id];
                if new < old && (entry.acknowledged || new == 0) {
                    let before = entry.input_tokens.saturating_sub(old);
                    work_s += params.work(old - new, before, !entry.seen_engine_progress);
                    tok += old - new;
                }
            }
            (work_s, tok)
        });
        self.emit(
            &mut state,
            json!({
                "event": "prefill_progress",
                "worker_url": worker_url,
                "engine_epoch": progress.epoch,
                "engine_sequence": progress.sequence,
                "snapshot_truncated": progress.truncated,
                "observed_rids": progress.remaining.len(),
                "updates": updates.iter().map(|(id, old, new, old_restore, restore, old_cost, cost)| json!({
                    "reservation_id": id, "remaining_before": old, "remaining_after": new,
                    "restore_before": old_restore, "restore_after": restore,
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
        // --symmetric-pd-gwq-auto: (id, rid, T_react sample) of first sightings of
        // releases marked for a sample.
        let mut lead_samples = Vec::new();
        for (id, tokens) in acknowledgements {
            let worker = state.workers.get_mut(worker_url).expect("reserved worker");
            worker.pending_unacknowledged_tokens = worker
                .pending_unacknowledged_tokens
                .checked_sub(u128::from(tokens))
                .expect("unacknowledged ledger underflow");
            let entry = state.entries.get_mut(&id).expect("active entry");
            entry.acknowledged = true;
            entry.ack_at = Some(received_at);
            if let Some(mark) = entry.gwq_lead_mark.take().filter(|m| m.sample) {
                let sample_s = sent_at
                    .saturating_duration_since(mark.released_at)
                    .as_secs_f64();
                lead_samples.push((id, entry.rid.clone(), sample_s, mark));
            }
        }
        if !lead_samples.is_empty() {
            let mut events = Vec::with_capacity(lead_samples.len());
            if let Some(book) = state.gwq_lead.as_mut() {
                for (id, rid, sample_s, mark) in lead_samples {
                    let react_s = book.observe(worker_url, sent_at, sample_s);
                    let samples = book.by_p.get(worker_url).map_or(0, |e| e.samples);
                    events.push(json!({
                        "event": "gwq_lead_sample",
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
        for (id, old, remaining, old_restore, restore, old_cost, cost) in updates {
            let worker = state.workers.get_mut(worker_url).expect("reserved worker");
            worker.pending_uncached_tokens = worker
                .pending_uncached_tokens
                .checked_sub(u128::from(old))
                .expect("pending token ledger underflow")
                + u128::from(remaining);
            worker.pending_restore_tokens = worker
                .pending_restore_tokens
                .checked_sub(u128::from(old_restore))
                .expect("restore ledger underflow")
                + u128::from(restore);
            worker.cost = worker.cost - old_cost + cost;
            let entry = state.entries.get_mut(&id).expect("active entry");
            entry.uncached_tokens = remaining;
            entry.restore_tokens = restore;
            entry.seen_engine_progress = true;
            entry.cost = cost;
        }
        for id in &listed {
            let entry = state.entries.get_mut(id).expect("active entry");
            entry.last_progress_seen = Some(sent_at);
        }
        // OLP incremental queue: every listed entry may have launched its
        // first chunk or changed its remaining work (both only here); its
        // row in its P's sums follows (unchanged rows cost nothing).
        if let State {
            olp: Some(olp),
            entries,
            ..
        } = &mut *state
        {
            if olp.config().incremental {
                for id in &listed {
                    olp.live_update(*id, olp_row(&entries[id]));
                }
            }
        }
        if let (Some((work_s, tok)), Some(olp)) = (olp_chunks, state.olp.as_mut()) {
            let t_s = self.seconds(sent_at);
            olp.observe_poll(worker_url, t_s, &progress.epoch, &engine_load, work_s, tok);
        }
        state.progress.insert(
            worker_url.into(),
            ProgressCursor {
                epoch: progress.epoch,
                sequence: progress.sequence,
                sent_at,
            },
        );
        // TRADE auto: the largest remaining work seen; 0 ends the prefill.
        let now = Instant::now();
        for (id, remaining) in auto_seen {
            if let Some((url, period)) = auto_progress(&mut state, id, remaining, now) {
                let _ = self.emit(&mut state, p_empty_event(&url, period));
            }
        }
        self.expire_unknown(&mut state, Some(worker_url), now);
        Ok(())
    }

    /// Record a periodic snapshot and return it for the caller's monitoring.
    /// Health is the most recently supplied candidate set, not a new probe.
    /// The unknown expiry of every worker runs first (see the module docs).
    pub fn snapshot(&self) -> Value {
        let mut state = self.state.lock();
        self.expire_unknown(&mut state, None, Instant::now());
        // Status counts and the oldest active assigned time of each worker.
        let mut by_worker = HashMap::<&str, (BTreeMap<&str, u64>, Option<u64>)>::new();
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
                    "pending_restore_tokens": worker.pending_restore_tokens.to_string(),
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
            "schema": "cost-balance.v1",
            "event": "snapshot",
            "ts_unix_ms": timestamp,
            "ts_unix_s": timestamp as f64 / 1000.0,
            "elapsed_ms": self.started.elapsed().as_millis() as u64,
            "cost_unit": self.cost_unit(),
            "restore_cost_divisor": self.restore_cost_divisor,
            "active_requests": active_requests,
            "unknown_requests": state.unknown.len(),
            "prefill_reader_gone": state.prefill_reader_gone,
            "healthy_candidates": state.healthy_candidates,
            "workers": workers,
            "tracked_entries": state.entries.len(),
            "unknown_entries": unknown,
            "fatal": state.io_failure,
            "admission_blocked": self.check_admission(&state).is_err(),
        });
        if let Some(expire) = self.unknown_expire {
            // Only with an unknown expiry, so other traces keep their fields.
            value["unknown_expire_s"] = json!(expire.as_secs_f64());
            value["expired_rid_listed_again"] = json!(state.expired_listed_again);
        }
        if let Some(book) = &state.host_room {
            // --symmetric-pd-loads-host-room only: every P with a section, and the
            // P whose engine rejected it.
            value["host_room_by_p"] = book.by_p(book.latest.keys(), Instant::now());
            value["host_room_rejected"] = json!(book.rejected);
            value["host_room_invalid"] = json!(book.invalid);
            value["host_room_missing"] = json!(book.missing);
            if let Some(phi) = &book.phi {
                // --co-schedule-trade-pull-price phi only: phi of every P with
                // a section.
                let now = Instant::now();
                let rows: serde_json::Map<String, Value> = book
                    .latest
                    .keys()
                    .map(|url| (url.clone(), phi.row(url, now)))
                    .collect();
                value["phi_by_p"] = Value::Object(rows);
                value["phi_tau_s"] = json!(phi.tau_s);
                value["phi_invalid"] = json!(phi.invalid);
                value["phi_prices"] = json!({
                    "decisions": phi.decisions,
                    "chosen_floored": phi.chosen_floored,
                    "rows": phi.rows,
                    "rows_floored": phi.rows_floored,
                    "rows_zero": phi.rows_zero,
                });
            }
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
        if let Some(auto) = state.trade_auto.as_ref() {
            // TRADE auto only: the per-P estimates now (N_active itself is
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
                        "rate_tokens_per_s": view.map(|v| v.rate_tokens_per_s),
                    })
                })
                .collect();
            value["trade_auto"] = json!({
                "elapsed_s": now.saturating_duration_since(auto.started()).as_secs_f64(),
                "workers": workers,
                "sessions_tracked": auto.tracked_sessions(),
                "sessions_dropped": auto.sessions_dropped,
                "credited": auto.credited,
                "credited_tokens": auto.credited_tokens,
                "uncredited": auto.uncredited,
            });
        }
        if let Some(olp) = state.olp.as_ref() {
            // OLP or its shadow only: the gate's warm-up inputs and results.
            value["olp"] = olp.snapshot_value(self.seconds(Instant::now()));
        }
        if let Err(error) = self.emit(&mut state, value.clone()) {
            value["fatal"] = json!(error);
            value["admission_blocked"] = json!(true);
        }
        value["sequence"] = json!(state.event_sequence);
        value
    }
}

/// The global waiting queue (`--co-schedule-global-queue`).
impl CostBook {
    /// Pull price phi (`--co-schedule-trade-pull-price phi`) of one decision
    /// at `decided_at`: the base lambda_q (the copy price auto before the
    /// clamp; the fallback price while it falls back), its source, tau, the
    /// floor min of lambda_P (the auto clamp minimum), and phi of each P of
    /// `urls` (None: no host_room data; that P pays max(min, lambda_q)). None
    /// with the lambda price. A GWQ step and its release use the same time,
    /// so the walk and the reservation see the same prices.
    fn phi_prices(
        state: &State,
        configured: crate::scheduler::co_schedule::TradeParams,
        auto: Option<&TradeAutoParts>,
        params: &crate::scheduler::co_schedule::TradeParams,
        urls: &[&str],
        decided_at: Instant,
    ) -> Option<PhiPrices> {
        if configured.pull_price != crate::scheduler::co_schedule::TradePullPrice::Phi {
            return None;
        }
        let (base, base_source) = match auto.map(|a| &a.1) {
            Some(resolution) => match resolution.lambda_raw {
                Some(raw) => (raw, "q"),
                None => (
                    resolution.copy_lambda_used,
                    resolution.fallback_reason.unwrap_or("fallback"),
                ),
            },
            // The CLI requires the copy price auto with phi.
            None => (params.copy_lambda, "fixed"),
        };
        // The floor of lambda_P: the auto clamp minimum
        // (--co-schedule-trade-copy-lambda-min).
        let floor = match configured.copy_lambda_mode {
            crate::scheduler::co_schedule::TradeLambdaMode::Auto { min, .. } => min,
            crate::scheduler::co_schedule::TradeLambdaMode::Fixed => 0.0,
        };
        let book = state.host_room.as_ref().and_then(|b| b.phi.as_ref());
        let tau_s = book.map_or(configured.auto_tau_s, |b| b.tau_s);
        let views = urls
            .iter()
            .map(|url| {
                book.and_then(|b| b.states.get(*url))
                    .and_then(|s| s.view(decided_at, tau_s))
            })
            .collect();
        Some((base, base_source, tau_s, floor, views))
    }

    /// The request-independent part of one TRADE decision at `decided_at`,
    /// under the lock: the HOL record pruned to its window (HOL on), and the
    /// copy price and the rate resolved from the router's measurements over
    /// the P of `urls` (`auto`; the auto state starts at the first call).
    /// Fixed modes resolve to the configured parameters bit for bit.
    fn trade_ctx(
        state: &mut State,
        configured: crate::scheduler::co_schedule::TradeParams,
        urls: &[&str],
        decided_at: Instant,
    ) -> TradeDecisionCtx {
        let hol = configured.hol_weight > 0.0;
        if hol {
            // The lazy prune of the arrival record; each candidate's a_w is
            // read after it.
            state
                .trade_arrivals
                .prune(decided_at, configured.hol_rate_window_s);
        }
        // Copy price or rate auto: the measurements at this decision give
        // lambda and R of this decision ([`TradeParams::resolved`]); the
        // rest of the parameters is unchanged. Fixed modes skip this, so
        // their decision and trace are the same as before.
        let auto = if configured.is_auto() {
            let book = state.trade_auto.get_or_insert_with(|| {
                crate::scheduler::co_schedule::TradeAutoState::new(
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
        TradeDecisionCtx {
            decided_at,
            hol,
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
    fn gwq_work(
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
    /// rows from the ledger (backlog as TRADE uses it, queued work, view
    /// freshness, a_w), the TRADE decision context, the walk
    /// ([`crate::global_queue::walk`]) and, when the walk releases a request,
    /// its reservation with the normal TRADE record, the virtual backlog in
    /// the backlog terms and the `gwq` object. `held` is in queue order.
    /// `Err` only when admission is stopped (trace failure).
    pub fn gwq_step(
        self: &Arc<Self>,
        configured: crate::scheduler::co_schedule::TradeParams,
        held: &[GwqHeld<'_>],
        args: &GwqStepArgs<'_>,
    ) -> Result<GwqStep, String> {
        use crate::global_queue::{terms_of, walk_with, GwqHeldInput, GwqRow, GwqTerms, GwqTie};
        let mut state = self.state.lock();
        self.check_admission(&state)?;
        for url in args.urls {
            state.workers.entry(url.clone()).or_default();
        }
        let urls: Vec<&str> = args.urls.iter().map(String::as_str).collect();
        let ctx = Self::trade_ctx(&mut state, configured, &urls, args.now);
        // --symmetric-pd-gwq-auto: the lead rule of each P with R of this step.
        let rate = ctx.params.rate_tokens_per_s;
        let leads: Option<Vec<GwqLeadRow>> = args.auto.map(|auto| {
            urls.iter()
                .zip(auto.chunks)
                .map(|(url, &(chunk, chunk_source))| {
                    let (react_s, react_source, react_samples) = state
                        .gwq_lead
                        .as_ref()
                        .map_or((0.0, "none", 0), |book| book.react(url));
                    GwqLeadRow::new(
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
        if let (Some(leads), Some(book)) = (leads.as_ref(), state.gwq_lead.as_mut()) {
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
                Self::gwq_work(&state, &unknown, url, pass)
            })
            .collect();
        drop(unknown);
        // Pull price phi: the walk prices each P as the reservation does.
        let phi = Self::phi_prices(
            &state,
            configured,
            ctx.auto.as_ref(),
            &ctx.params,
            &urls,
            args.now,
        );
        // `room = predicted`: the E3 drain of each P (a cold one-chunk
        // probe on the newest pushed or polled state, plus the router's own
        // rows that the engine does not show yet), as a `wait = e3` decision
        // computes it under this lock.
        let drains: Vec<(Option<f64>, Option<&'static str>)> = match args.e3 {
            Some(e3) => {
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
                        let probe = Self::probe_trade_e3(
                            &state,
                            Some(e3),
                            &gwq_probe_candidate(url),
                            args.now,
                            &ctx.params,
                            &ids,
                        );
                        gwq_probe_drain(&probe)
                    })
                    .collect()
            }
            None => vec![(None, None); args.urls.len()],
        };
        let rows: Vec<GwqRow> = args
            .urls
            .iter()
            .zip(args.depth)
            .zip(&work)
            .zip(&drains)
            .enumerate()
            .map(|(i, (((url, depth), (queued, _, _)), (drain_s, _)))| {
                let worker = &state.workers[url];
                let ledger = if ctx.params.exclude_unlaunched {
                    worker
                        .pending_uncached_tokens
                        .saturating_sub(worker.pending_unacknowledged_tokens)
                } else {
                    worker.pending_uncached_tokens
                };
                GwqRow {
                    // The forecast may make a P look emptier than the
                    // ledger, never fuller (work-conserving floor).
                    backlog: match drain_s {
                        Some(d) => gwq_drain_tokens(*d, rate).min(ledger),
                        None => ledger,
                    },
                    drain_s: *drain_s,
                    // --symmetric-pd-gwq-auto: the lead rule of the P (C / R +
                    // T_react), as the token rule's depth.
                    lead_s: leads.as_ref().map_or(args.lead_s, |l| l[i].lead_s),
                    queued: *queued,
                    depth: leads.as_ref().map_or(*depth, |l| l[i].depth),
                    fresh: state.progress.get(url).is_some_and(|cursor| {
                        args.now.saturating_duration_since(cursor.sent_at) <= args.stale_after
                    }),
                    arrival_rate_per_s: if ctx.hol {
                        state
                            .trade_arrivals
                            .rate_per_s(url, args.now, ctx.params.hol_rate_window_s)
                    } else {
                        0.0
                    },
                    copy_lambda: phi
                        .as_ref()
                        .map(|(base, _, _, floor, views)| phi_price(*base, *floor, views[i]).0),
                }
            })
            .collect();
        // The candidates of each held request over the step pool (a P that
        // joined is cold, a P that left is gone), and their terms.
        let pooled: Vec<Vec<Candidate>> = held
            .iter()
            .map(|h| gwq_pool_candidates(h.candidates, args.urls, args.sources))
            .collect();
        let terms: Vec<Vec<GwqTerms>> = held
            .iter()
            .zip(&pooled)
            .map(|(h, cands)| {
                cands
                    .iter()
                    .map(|c| terms_of(c.local_hit, c.expected_hit, h.input_tokens))
                    .collect()
            })
            .collect();
        let inputs: Vec<GwqHeldInput<'_>> = held
            .iter()
            .zip(&terms)
            .map(|(h, terms)| {
                let waited = args.now.saturating_duration_since(h.arrived);
                GwqHeldInput {
                    terms,
                    waited_s: waited.as_secs_f64(),
                    starved: waited >= args.max_wait,
                }
            })
            .collect();
        // --symmetric-pd-gwq-auto: exact ties go to the P with room, the larger
        // own hit, the fewest outstanding requests, then the next URL after
        // the last tie (rotation); the release reservation breaks them alike.
        let outstanding: Vec<u64> = args
            .urls
            .iter()
            .map(|url| state.workers[url].active_requests)
            .collect();
        let tie_last = state.gwq_lead.as_ref().and_then(|b| b.tie_last.clone());
        let tie = leads.as_ref().map(|_| GwqTie {
            outstanding: &outstanding,
            last: tie_last.as_deref(),
        });
        let walked = walk_with(
            args.mode,
            &ctx.params,
            args.urls,
            &rows,
            &inputs,
            tie.as_ref(),
        );
        let queue_tokens: u128 = walked
            .plain
            .iter()
            .zip(&terms)
            .map(|(p, t)| u128::from(t[*p].uncached))
            .sum();
        let Some(pick) = walked.pick else {
            return Ok(GwqStep {
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
            "mode": args.mode.as_str(),
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
            "room_rule": if args.e3.is_some() { "predicted" } else { "tokens" },
            "lead_s": args.e3.map(|_| args.lead_s),
            "drain_s": row.drain_s,
            "drain_s_by_p": args.e3.map(|_| drains.iter().map(|(d, _)| *d).collect::<Vec<_>>()),
            "drain_fallback_by_p": args.e3.map(|_| drains.iter().map(|(_, r)| *r).collect::<Vec<_>>()),
            "session_known": h.session.is_some(),
            "pass_id": args.pass_id,
            "trigger": args.trigger,
            "triggers": args.triggers,
        });
        // --symmetric-pd-gwq-auto: the lead rule of the released P, the class
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
                "rate_tokens_per_s": rate,
                "react_s": lead.react_s,
                "react_source": lead.react_source,
                "react_samples": lead.react_samples,
                "lead_s": lead.lead_s,
                "wait_s": row.queued as f64 / rate,
                "lead_sample": lead_sample,
                "shadow": Self::gwq_shadow(&state, h, p_release, args.now),
            });
            if let (Value::Object(record), Value::Object(extra)) = (&mut record, extra) {
                record.extend(extra);
            }
        }
        // --symmetric-pd-gwq-auto: the walk's tie-break, as the release
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
        let notes: BTreeMap<String, GwqRowNote> = args
            .urls
            .iter()
            .zip(&rows)
            .zip(&pick.virtual_tokens)
            .map(|((url, row), virtual_tokens)| {
                (
                    url.clone(),
                    GwqRowNote {
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
        let placement = Placement::Trade(configured);
        let reserve = GwqReserve {
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
            .validate_placement(h.input_tokens, &release_candidates, placement)
            .and_then(|valid| {
                self.reserve_locked(
                    &mut state,
                    h.input_tokens,
                    h.rid.map(str::to_string),
                    release_candidates,
                    valid,
                    placement,
                    None,
                    h.session,
                    None,
                    Some(reserve),
                    ReserveExtras {
                        gwq_note: None,
                        session_label: h.session_label,
                    },
                )
            })
            .map(|(candidate, reservation, decision)| GwqReleased {
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
                entry.gwq_lead_mark = Some(GwqLeadMark {
                    released_at: args.now,
                    sample: lead_sample,
                    restore_tokens,
                    pull_tokens,
                });
            }
            if let Some(book) = state.gwq_lead.as_mut() {
                if lead_sample {
                    book.marked += 1;
                } else {
                    book.unmarked += 1;
                }
            }
        }
        Ok(GwqStep {
            plain: walked.plain,
            queue_tokens,
            release: Some((pick.index, result)),
        })
    }

    /// The shadow counters of one `--symmetric-pd-gwq-auto` release of `h` to
    /// `url` (no action): phi of the P, its dead (class 0) and free host
    /// tokens, the tokens that the request inserts there (input minus the
    /// P's own hit), and whether the working-set hold (ii) would hold it:
    /// phi > 1/2 and the insert exceeds free + class 0. Null without
    /// host_room data.
    fn gwq_shadow(state: &State, h: GwqHeld<'_>, url: &str, now: Instant) -> Value {
        let book = state.host_room.as_ref();
        let view = book
            .and_then(|b| b.phi.as_ref())
            .and_then(|p| p.states.get(url).and_then(|s| s.view(now, p.tau_s)));
        let section = book.and_then(|b| b.latest.get(url)).map(|(room, _)| room);
        let field = |key: &str| section.and_then(|r| r.get(key)).and_then(Value::as_u64);
        let (class0, free) = (field("class0_tokens"), field("free_tokens"));
        let local = h
            .candidates
            .iter()
            .find(|c| c.worker_url == url)
            .map_or(0, |c| c.local_hit.min(h.input_tokens));
        let insert = h.input_tokens - local;
        let phi = view.map(|v| v.phi);
        let would_hold = match (phi, class0, free) {
            (Some(phi), Some(class0), Some(free)) => {
                Some(phi > 0.5 && u128::from(insert) > u128::from(class0) + u128::from(free))
            }
            _ => None,
        };
        json!({
            "phi": phi,
            "phi_source": view.map_or("no_host_room", |v| v.source),
            "class0_tokens": class0,
            "free_tokens": free,
            "insert_tokens": insert,
            "phi_over_half": phi.map(|p| p > 0.5),
            "ii_would_hold": would_hold,
        })
    }

    /// `--symmetric-pd-gwq-auto`: T_react and the lead of the last step of each P
    /// of `urls`, and the sample counters (the `gwq_stats` event). Null
    /// without [`Self::enable_gwq_lead`].
    pub fn gwq_lead_rows(&self, urls: &[String]) -> Value {
        let state = self.state.lock();
        let Some(book) = state.gwq_lead.as_ref() else {
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
                        "rate_tokens_per_s": step.map(|s| s.2),
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
    /// `gwq_stats` event.
    pub fn gwq_rows(&self, urls: &[String], now: Instant, pass_tokens: u64) -> Vec<GwqRowStat> {
        let state = self.state.lock();
        let unknown = Self::unknown_unacknowledged(&state);
        urls.iter()
            .map(|url| {
                let (queued, unlaunched, launched) =
                    Self::gwq_work(&state, &unknown, url, pass_tokens);
                GwqRowStat {
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

    /// Write one more event to the trace (`gwq_drop`, `gwq_stats`), with the
    /// usual sequence and time keys. A write failure stops admission, as for
    /// every event.
    pub fn record_event(&self, event: Value) -> Result<(), String> {
        let mut state = self.state.lock();
        self.emit(&mut state, event)
    }

    /// Why `rid` cannot be reserved (uncached mode: missing, empty or used),
    /// with the same text as the reservation error; None when it can.
    pub fn rid_refusal(&self, rid: Option<&str>) -> Option<&'static str> {
        if !self.remaining_work {
            return None;
        }
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
    book: Arc<CostBook>,
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
        self.book.state.lock().entries[&self.id].value()
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let status = self.book.state.lock().entries[&self.id].status;
        let result = match status {
            Status::Reserved => self.book.change_status(
                self.id,
                Status::RolledBack,
                "owner_dropped_before_dispatch",
            ),
            Status::Dispatched => self.book.change_status(
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
/// TRADE auto rate.
const PREFILL_SUCCESS_REASON: &str = "prefill_response_success";

/// TRADE auto: apply one status change of entry `id` to the auto state.
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
) -> Option<(String, crate::scheduler::co_schedule::TradeAutoBusyPeriod)> {
    let entry = state.entries.get_mut(&id)?;
    if !entry.auto.tracked || entry.auto.done {
        return None;
    }
    let auto = state.trade_auto.as_mut()?;
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

/// TRADE auto: one progress observation of a tracked, dispatched entry. The
/// computed tokens become the larger of their value and `remaining`; 0
/// remaining work ends its prefill (credited, applied once). Returns the P
/// and its busy period when the P has no request left.
fn auto_progress(
    state: &mut State,
    id: u64,
    remaining: u64,
    now: Instant,
) -> Option<(String, crate::scheduler::co_schedule::TradeAutoBusyPeriod)> {
    let entry = state.entries.get_mut(&id)?;
    if !entry.auto.tracked || entry.auto.done || !entry.auto.in_flight {
        return None;
    }
    entry.auto.peak_tokens = entry.auto.peak_tokens.max(remaining);
    if remaining > 0 {
        return None;
    }
    let auto = state.trade_auto.as_mut()?;
    entry.auto.done = true;
    entry.auto.in_flight = false;
    let period = auto.prefill_done(&entry.worker_url, Some(entry.auto.peak_tokens), now);
    if let Some(session) = entry.auto.session {
        auto.session_finished(session, now);
    }
    period.map(|period| (entry.worker_url.clone(), period))
}

/// The trace event of a P whose TRADE auto busy period ended.
fn p_empty_event(
    worker_url: &str,
    period: crate::scheduler::co_schedule::TradeAutoBusyPeriod,
) -> Value {
    json!({
        "event": "trade_auto_p_empty",
        "worker_url": worker_url,
        "busy_period_s": period.seconds,
        "busy_period_requests": period.requests,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::olp::{strip_olp_keys, OlpConfig, OlpParams, OlpScaleSource};
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

    fn cost(book: &CostBook, url: &str) -> u128 {
        book.state.lock().workers[url].cost
    }

    fn uncached(book: &Arc<CostBook>, rid: &str, url: &str, tokens: u64) -> Reservation {
        book.reserve_uncached(tokens, Some(rid.into()), vec![candidate(url, 0)], 12288)
            .unwrap()
            .1
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

    fn trade(lambda: f64, exclude_unlaunched: bool) -> crate::scheduler::co_schedule::TradeParams {
        crate::scheduler::co_schedule::TradeParams {
            copy_lambda: lambda,
            exclude_unlaunched,
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

    fn unacknowledged(book: &CostBook, url: &str) -> u128 {
        book.state.lock().workers[url].pending_unacknowledged_tokens
    }

    /// Owner `a` holds 100k of 110k with 60k pending; `b` is idle and would
    /// pull 100k. lambda 0 moves (17,425 < 70,000); lambda 9.2 keeps `a`.
    /// The trade decision names the owner, the pull and every candidate.
    #[test]
    fn trade_moves_with_pull_at_lambda_zero_and_keeps_owner_at_large_lambda() {
        for (lambda, chosen, pulled) in [(0.0, "b", 100_000u64), (9.2, "a", 0)] {
            let book = CostBook::new_uncached(None).unwrap();
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
                .reserve_trade(
                    110_000,
                    Some("req".into()),
                    vec![candidate("a", 100_000), pulling("b", 0, 100_000)],
                    trade(lambda, false),
                )
                .unwrap();
            assert_eq!(selected.worker_url, chosen, "lambda {lambda}");
            assert_eq!(decision["owner"], "a");
            assert_eq!(decision["detour"], chosen != "a");
            assert_eq!(decision["pulled_tokens"], pulled);
            assert_eq!(decision["candidates"].as_array().unwrap().len(), 2);
            // The ledger keeps uncached-token units: the pulled prefix is not charged.
            assert_eq!(
                cost(&book, chosen),
                if chosen == "a" { 70_000 } else { 10_000 }
            );
            drop(busy);
        }
    }

    /// Requests routed to a P but not yet in its progress snapshot (the engine
    /// lists a RID only after its first chunk launches, so this includes the
    /// engine waiting queue) count in the default backlog and are left out
    /// only with exclude_unlaunched. Acknowledgement happens once per entry,
    /// also when the reported remaining equals the estimate, and terminal
    /// transitions keep the counter balanced.
    #[test]
    fn trade_backlog_counts_unlaunched_work_unless_excluded() {
        let book = CostBook::new_uncached(None).unwrap();
        let mut a = uncached(&book, "routed", "a", 30_000);
        a.mark_dispatched().unwrap();
        assert_eq!(unacknowledged(&book, "a"), 30_000);
        let pick = |book: &Arc<CostBook>, rid: &str, exclude: bool| {
            let (selected, r, decision) = book
                .reserve_trade(
                    20_000,
                    Some(rid.into()),
                    vec![candidate("a", 0), candidate("b", 0)],
                    trade(9.2, exclude),
                )
                .unwrap();
            drop(r); // rolls back: the next pick sees the same ledger
            (selected.worker_url, decision)
        };
        // Default: the dispatched but unlaunched 30k counts and `b` wins.
        let (worker, decision) = pick(&book, "q1", false);
        assert_eq!(worker, "b");
        assert_eq!(decision["candidates"][0]["backlog_tokens"], "30000");
        assert_eq!(decision["candidates"][0]["unacknowledged_tokens"], "30000");
        assert_eq!(decision["exclude_unlaunched"], false);
        assert_eq!(decision["count_inflight"], true);
        // Excluding unlaunched work, `a` looks idle: tie, smaller URL wins.
        let (worker, decision) = pick(&book, "q2", true);
        assert_eq!(worker, "a");
        assert_eq!(decision["candidates"][0]["backlog_tokens"], "0");
        assert_eq!(
            unacknowledged(&book, "a"),
            30_000,
            "rollbacks keep the counter"
        );
        // The engine reports the RID unchanged: acknowledged, no work update.
        book.apply_progress(
            "a",
            progress(1, &[("routed", 30_000)]),
            json!({}),
            Instant::now(),
        )
        .unwrap();
        assert_eq!(unacknowledged(&book, "a"), 0);
        assert_eq!(cost(&book, "a"), 30_000);
        assert_eq!(pick(&book, "q3", false).0, "b");
        assert_eq!(
            pick(&book, "q3x", true).0,
            "b",
            "acknowledged work always counts"
        );
        a.finish("http_done").unwrap();
        assert_eq!(unacknowledged(&book, "a"), 0);
        assert_eq!(cost(&book, "a"), 0);
        // A finished unacknowledged entry also leaves the counter.
        let mut c = uncached(&book, "never_seen", "b", 5_000);
        c.mark_dispatched().unwrap();
        assert_eq!(unacknowledged(&book, "b"), 5_000);
        c.finish("http_done").unwrap();
        assert_eq!(unacknowledged(&book, "b"), 0);
    }

    #[test]
    fn trade_is_uncached_only_and_records_its_policy() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("trade.jsonl");
        let book = CostBook::new_uncached(Some(&path)).unwrap();
        let (_, _r, _) = book
            .reserve_trade(
                1_000,
                Some("r".into()),
                vec![candidate("a", 0)],
                trade(9.2, false),
            )
            .unwrap();
        let line = std::fs::read_to_string(&path).unwrap();
        let event: Value = serde_json::from_str(line.lines().next().unwrap()).unwrap();
        assert_eq!(event["selection_policy"], "trade");
        assert_eq!(event["trade_decision"]["copy_lambda"], 9.2);
        assert!(event["locality_decision"].is_null());
        // Without the global waiting queue there is no gwq key.
        assert!(event.get("gwq").is_none());
        assert!(event["trade_decision"]["candidates"][0]
            .get("gwq_room")
            .is_none());
        let cost_book = CostBook::new(None).unwrap();
        assert!(cost_book
            .reserve_trade(
                1_000,
                Some("r".into()),
                vec![candidate("a", 0)],
                trade(9.2, false)
            )
            .is_err());
        let restore = CostBook::new_restore_aware(None, 8).unwrap();
        assert!(restore
            .reserve_trade(
                1_000,
                Some("r".into()),
                vec![candidate("a", 0)],
                trade(9.2, false)
            )
            .is_err());
        // The non-trade uncached event carries no trade field.
        let path = tmp.path().join("plain.jsonl");
        let plain = CostBook::new_uncached(Some(&path)).unwrap();
        let _r = plain
            .reserve_uncached(1_000, Some("r".into()), vec![candidate("a", 0)], 12288)
            .unwrap();
        let line = std::fs::read_to_string(&path).unwrap();
        let event: Value = serde_json::from_str(line.lines().next().unwrap()).unwrap();
        assert!(event.get("trade_decision").is_none());
        assert_eq!(event["selection_policy"], "uncached_threshold");
    }

    /// HOL on (mu > 0): each TRADE placement is recorded per P when its
    /// reservation commits, the next decision counts it in a_w, and the trace
    /// has the HOL keys. A rollback keeps the record. With mu = 0 nothing is
    /// recorded and the trace has no new key.
    #[test]
    fn trade_hol_records_placements_and_traces_its_terms() {
        let hol = |mu: f64| crate::scheduler::co_schedule::TradeParams {
            hol_weight: mu,
            ..trade(0.0, false)
        };
        let book = CostBook::new_uncached(None).unwrap();
        // b holds 12,000 tokens from a placement outside TRADE (not recorded).
        let _busy = uncached(&book, "busy", "b", 12_000);
        let pick = |rid: &str, params: crate::scheduler::co_schedule::TradeParams| {
            book.reserve_trade(
                10_000,
                Some(rid.into()),
                vec![candidate("a", 0), candidate("b", 0)],
                params,
            )
            .unwrap()
        };
        let recorded = |url: &str| book.state.lock().trade_arrivals.recorded(url);
        // No history: a_w = 0 and no HOL; a (10,000) beats b (22,000).
        let (selected, r1, decision) = pick("r1", hol(10.0));
        assert_eq!(selected.worker_url, "a");
        assert_eq!(decision["hol_weight"], 10.0);
        assert_eq!(decision["hol_rate_window_s"], 60.0);
        assert_eq!(
            decision["wait"], "ledger",
            "mu > 0 adds the later-option keys"
        );
        for row in decision["candidates"].as_array().unwrap() {
            assert_eq!(row["arrival_rate_per_s"], 0.0, "{row}");
            assert_eq!(row["hol_tokens"], 0.0, "{row}");
        }
        assert_eq!((recorded("a"), recorded("b")), (1, 0));
        // mu = 0: a (10,000 + 10,000) beats b (22,000); no record, no new key.
        let (selected, r, decision) = pick("q0", hol(0.0));
        assert_eq!(selected.worker_url, "a");
        assert!(decision.get("hol_weight").is_none());
        assert!(decision.get("hol_rate_window_s").is_none());
        assert!(
            decision.get("wait").is_none(),
            "mu = 0 keeps the first-version trace"
        );
        assert!(decision["candidates"][0]
            .get("arrival_rate_per_s")
            .is_none());
        assert!(decision["candidates"][0].get("hol_tokens").is_none());
        drop(r);
        assert_eq!((recorded("a"), recorded("b")), (1, 0));
        // mu = 10: r1 gives a_a in (0, 1] (one placement / at least 1 s) and
        // HOL(a) = 10 x a_a x 10,000 x (10,000 + 5,000) / 13,500, which moves
        // the request to b (22,000) for any a_a above 0.02.
        let (selected, r2, decision) = pick("r2", hol(10.0));
        assert_eq!(selected.worker_url, "b");
        let a = &decision["candidates"][0];
        let rate = a["arrival_rate_per_s"].as_f64().unwrap();
        assert!(rate > 0.0 && rate <= 1.0, "{rate}");
        let hol_a = 10.0 * rate * 10_000.0 * 15_000.0 / 13_500.0;
        assert!(
            (a["hol_tokens"].as_f64().unwrap() - hol_a).abs() < 1e-6,
            "{a}"
        );
        assert!(
            (a["trade_cost"].as_f64().unwrap() - (20_000.0 + hol_a)).abs() < 1e-6,
            "{a}"
        );
        let b = &decision["candidates"][1];
        assert_eq!(b["arrival_rate_per_s"], 0.0);
        assert_eq!(b["hol_tokens"], 0.0);
        assert_eq!(b["trade_cost"], 22_000.0);
        // The rollback of r2 keeps its record.
        drop(r2);
        assert_eq!((recorded("a"), recorded("b")), (1, 1));
        drop(r1);
    }

    fn auto_trade(fallback: f64) -> crate::scheduler::co_schedule::TradeParams {
        use crate::scheduler::co_schedule::{TradeAutoQ, TradeLambdaMode, TradeParams};
        TradeParams {
            copy_lambda: fallback,
            copy_lambda_mode: TradeLambdaMode::Auto {
                q: TradeAutoQ::Sat,
                mu: 1.0,
                min: 4.6,
                max: 1000.0,
            },
            ..Default::default()
        }
    }

    /// TRADE auto: the P of an entry has one more in-flight request from its
    /// dispatch until its prefill ends, released once on every path: 0
    /// remaining work in a snapshot, a success or error response, a
    /// transport error (unknown) with a late finish, a drop after dispatch.
    /// A rollback before dispatch releases only the session. Only a
    /// completed prefill credits tokens: the larger of the estimate and the
    /// largest remaining work seen.
    #[test]
    fn trade_auto_ledger_counts_each_request_once_on_every_path() {
        let book = CostBook::new_uncached(None).unwrap();
        // 20,000 tokens with 8,000 cached on a: an estimate of 12,000 uncached.
        let pick = |rid: &str, session: Option<&str>| {
            book.reserve_trade_session(
                20_000,
                Some(rid.into()),
                vec![candidate("a", 8_000)],
                auto_trade(18.4),
                None,
                session,
            )
            .unwrap()
            .1
        };
        let in_flight = || {
            book.state
                .lock()
                .trade_auto
                .as_ref()
                .unwrap()
                .in_flight("a")
        };
        let agent = crate::scheduler::co_schedule::trade_session_hash("0/agent");
        let session = || {
            book.state
                .lock()
                .trade_auto
                .as_ref()
                .unwrap()
                .session_in_flight(agent)
        };
        let counts = || {
            let st = book.state.lock();
            let auto = st.trade_auto.as_ref().unwrap();
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

    /// TRADE auto: the first decision is in start-up (the fallback 18.4 and
    /// every auto key). With a 10-minute history, lambda = clamp(q x R /
    /// R_hat) from the traced q and R_hat, and the candidates pay it.
    #[test]
    fn trade_auto_decision_resolves_lambda_and_traces_its_inputs() {
        let book = CostBook::new_uncached(None).unwrap();
        let decide = |rid: &str| {
            let (_, r, d) = book
                .reserve_trade_session(
                    110_000,
                    Some(rid.into()),
                    vec![candidate("a", 100_000), pulling("b", 0, 100_000)],
                    auto_trade(18.4),
                    None,
                    Some("0/agent-1"),
                )
                .unwrap();
            drop(r);
            d
        };
        let d = decide("s1");
        assert_eq!(d["lambda_mode"], "auto");
        assert_eq!(d["q_mode"], "sat");
        assert_eq!(d["fallback_reason"], "startup");
        assert_eq!(d["copy_lambda_used"], 18.4);
        assert_eq!(d["copy_lambda"], 18.4);
        assert_eq!(d["copy_lambda_fallback"], 18.4);
        assert!(d["lambda_raw"].is_null());
        assert_eq!(d["n_p"], 2);
        assert_eq!(d["rate_mode"], "fixed");
        assert_eq!(d["rate_used"], 13_500.0);
        assert_eq!(d["session_known"], true);
        assert_eq!(d["wait"], "ledger", "auto adds the later-option keys");
        for key in [
            "q",
            "busy",
            "inflight_mean",
            "n_active",
            "r_hat",
            "auto_elapsed_s",
            "auto_tau_s",
            "session_idle_s",
            "lambda_mu",
            "lambda_min",
            "lambda_max",
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
        let mut auto = crate::scheduler::co_schedule::TradeAutoState::new(t0, 120.0);
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
        book.state.lock().trade_auto = Some(auto);
        let d = decide("s2");
        assert!(d["fallback_reason"].is_null(), "{d}");
        assert_eq!(d["n_active"], 60);
        let q = d["q"].as_f64().unwrap();
        let busy = d["busy"].as_f64().unwrap();
        let r_hat = d["r_hat"].as_f64().unwrap();
        let lambda = d["copy_lambda_used"].as_f64().unwrap();
        assert!((q - 30.0 * busy).abs() < 1e-9, "{d}");
        // The same operations as TradeParams::resolved (mu = 1).
        let raw = q * (13_500.0 / r_hat);
        assert_eq!(d["lambda_raw"], raw, "{d}");
        assert_eq!(lambda, raw.clamp(4.6, 1000.0));
        assert_eq!(d["copy_lambda"], lambda);
        assert!(lambda > 15.0 && lambda < 45.0, "{d}");
        let copy_b = d["candidates"][1]["copy_tokens"].as_f64().unwrap();
        assert!((copy_b - lambda * 100_000.0).abs() < 1e-6, "{d}");
    }

    /// Fixed TRADE creates no auto state, key or event. Auto adds the
    /// snapshot summary and one `trade_auto_p_empty` event per busy period,
    /// and END removes a session from N_active at once.
    #[test]
    fn trade_auto_end_snapshot_and_fixed_mode_untouched() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("fixed.jsonl");
        let book = CostBook::new_uncached(Some(&path)).unwrap();
        let (_, mut r, d) = book
            .reserve_trade_session(
                10_000,
                Some("f1".into()),
                vec![candidate("a", 0)],
                trade(18.4, false),
                None,
                Some("0/agent"),
            )
            .unwrap();
        assert!(
            d.get("lambda_mode").is_none() && d.get("wait").is_none(),
            "{d}"
        );
        r.mark_dispatched().unwrap();
        book.apply_progress("a", progress(1, &[("f1", 0)]), json!({}), Instant::now())
            .unwrap();
        r.finish("prefill_response_success").unwrap();
        assert!(book.state.lock().trade_auto.is_none());
        assert!(book.snapshot().get("trade_auto").is_none());
        assert!(!book.trade_auto_end_session("0/agent"));
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("trade_auto"), "{text}");

        let path = tmp.path().join("auto.jsonl");
        let book = CostBook::new_uncached(Some(&path)).unwrap();
        let decide = |rid: &str, session: Option<&str>| {
            book.reserve_trade_session(
                10_000,
                Some(rid.into()),
                vec![candidate("a", 0), candidate("b", 0)],
                auto_trade(18.4),
                None,
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
        assert!(book.trade_auto_end_session("0/agent-1"));
        assert!(!book.trade_auto_end_session("0/agent-1"));
        let (_, r, d) = decide("probe2", None);
        assert_eq!(d["n_active"], 1, "{d}");
        drop(r);
        // A dispatched request ends its P's busy period at the P response.
        let (selected, mut r, _) = decide("busy", None);
        r.mark_dispatched().unwrap();
        r.finish("prefill_response_success").unwrap();
        let snapshot = book.snapshot();
        let auto = &snapshot["trade_auto"];
        assert_eq!(auto["credited"], 1, "{snapshot}");
        assert_eq!(auto["credited_tokens"], 10_000);
        assert_eq!(auto["workers"].as_array().unwrap().len(), 2);
        assert_eq!(auto["workers"][0]["in_flight"], 0);
        let text = std::fs::read_to_string(&path).unwrap();
        let empties: Vec<Value> = text
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .filter(|event| event["event"] == "trade_auto_p_empty")
            .collect();
        assert_eq!(empties.len(), 1, "{text}");
        assert_eq!(empties[0]["worker_url"], selected.worker_url.as_str());
        assert_eq!(empties[0]["busy_period_requests"], 1);
        assert!(empties[0]["busy_period_s"].as_f64().unwrap() >= 0.0);
    }

    fn eft_context(views: &[(&str, Value, std::time::Duration)]) -> TradeEftContext {
        let model = crate::eft_cheap::EftModel::from_json(
            &json!({"models": {"augmented": {"launch_interval": {"global":
                {"coefficients": [0.05, 1e-5, 0.0, 0.004, 1e-7, 0.0]}}}},
                "eft": {"tau_s": 0.02}}),
            Some("model-sha".into()),
        )
        .unwrap();
        TradeEftContext {
            model: Arc::new(model),
            knobs: crate::eft_cheap::EftKnobs::default(),
            views: views
                .iter()
                .map(|(url, raw, age)| {
                    let sent = Instant::now().checked_sub(*age).unwrap();
                    let snapshot = RoutingSnapshot {
                        view: crate::eft_cheap::SnapshotView::parse(raw, 1, None),
                        poll_sent: sent,
                        poll_received: sent,
                        e3: None,
                    };
                    (url.to_string(), Arc::new(snapshot))
                })
                .collect(),
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

    fn eft_trade(lambda: f64) -> crate::scheduler::co_schedule::TradeParams {
        crate::scheduler::co_schedule::TradeParams {
            copy_lambda: lambda,
            wait: crate::scheduler::co_schedule::TradeWait::Eft,
            ..Default::default()
        }
    }

    /// wait = eft. Owner `a` has an empty ledger but its engine queue holds
    /// five 65,536-token rows (D = 5.4368 s); `b` is idle and pulls 18,000
    /// tokens. The ledger keeps `a`; EFT moves to `b`. The next decision
    /// counts the reserved `r1` (not in b's snapshot) in b's drain.
    #[test]
    fn trade_eft_wait_prices_the_engine_queue_and_unseen_router_rows() {
        let book = CostBook::new_uncached(None).unwrap();
        let ctx = eft_context(&[
            ("a", routing_snapshot(5, 65_536), std::time::Duration::ZERO),
            ("b", routing_snapshot(0, 0), std::time::Duration::ZERO),
        ]);
        let candidates = || vec![candidate("a", 18_000), pulling("b", 0, 18_000)];
        // Ledger wait: the owner looks idle.
        let (selected, r0, decision) = book
            .reserve_trade(20_000, Some("l0".into()), candidates(), trade(0.0, false))
            .unwrap();
        assert_eq!(selected.worker_url, "a");
        assert!(
            decision.get("wait").is_none(),
            "ledger trace keeps the first-version keys"
        );
        assert!(decision["candidates"][0].get("wait_s").is_none());
        drop(r0);
        let (selected, r1, decision) = book
            .reserve_trade_eft(
                20_000,
                Some("r1".into()),
                candidates(),
                eft_trade(0.0),
                Some(&ctx),
            )
            .unwrap();
        assert_eq!(selected.worker_url, "b");
        assert_eq!(decision["wait"], "eft");
        assert_eq!(decision["eft_model_sha256"], "model-sha");
        assert_eq!(decision["wait_fallbacks"], 0);
        let a = &decision["candidates"][0];
        assert_eq!(a["wait_source"], "eft");
        let wait = a["wait_s"].as_f64().unwrap();
        assert!(wait > 5.3 && wait <= 5.4368 + 1e-9, "{wait}");
        assert_eq!(a["eft_outstanding_rows"], 0);
        // S_w of a: u 2,016, host restore 17,984, one pass.
        let svc = 1e-5 * 2_016.0 + 1e-7 * 17_984.0 + 0.004 + 0.05;
        assert!((a["service_s"].as_f64().unwrap() - svc).abs() < 1e-12);
        let b = &decision["candidates"][1];
        assert_eq!(b["wait_s"], 0.0);
        assert_eq!(b["pulled_tokens"], 18_000);
        assert_eq!(b["eft_shape"]["s"], 17_984);
        // r1 is reserved on b and not in b's snapshot: it is outstanding.
        let (_, r2, decision) = book
            .reserve_trade_eft(
                20_000,
                Some("r2".into()),
                candidates(),
                eft_trade(0.0),
                Some(&ctx),
            )
            .unwrap();
        let b = &decision["candidates"][1];
        assert_eq!(b["eft_outstanding_rows"], 1);
        assert_eq!(b["eft_outstanding_tokens"], 2_000);
        // W(u 2,000, q 18,000, s 18,000 pulled) + one pass.
        let expected = 1e-5 * 2_000.0 + 2e-7 * 18_000.0 + 0.004 + 0.05;
        assert!(
            (b["wait_s"].as_f64().unwrap() - expected).abs() < 1e-9,
            "{b}"
        );
        drop((r1, r2));
    }

    /// Missing, stale and old snapshots use the ledger terms for that P and
    /// are counted by reason; without a model context every P falls back.
    #[test]
    fn trade_eft_falls_back_to_the_ledger_per_worker_and_counts_it() {
        let book = CostBook::new_uncached(None).unwrap();
        let mut stale = routing_snapshot(0, 0);
        stale["stale"] = json!(true);
        let ctx = eft_context(&[
            ("a", stale, std::time::Duration::ZERO),
            (
                "b",
                routing_snapshot(0, 0),
                std::time::Duration::from_secs(2),
            ),
        ]);
        let pick = |rid: &str, ctx: Option<&TradeEftContext>| {
            let (_, r, decision) = book
                .reserve_trade_eft(
                    1_000,
                    Some(rid.into()),
                    vec![candidate("a", 0), candidate("b", 0), candidate("c", 0)],
                    eft_trade(9.2),
                    ctx,
                )
                .unwrap();
            drop(r);
            decision
        };
        let d = pick("q1", Some(&ctx));
        let reasons: Vec<_> = (0..3)
            .map(|i| d["candidates"][i]["wait_fallback_reason"].clone())
            .collect();
        assert_eq!(
            reasons,
            vec![json!("stale"), json!("view_too_old"), json!("no_view")]
        );
        // The fallback keeps the scale of the EFT terms: S_w is the step
        // model's (1,000 tokens, one pass), not u / R.
        let svc = 1e-5 * 1_000.0 + 0.004 + 0.05;
        for i in 0..3 {
            let row = &d["candidates"][i];
            assert_eq!(row["wait_source"], "ledger_fallback");
            assert_eq!(row["wait_s"], 0.0, "empty ledger");
            assert!(
                (row["service_s"].as_f64().unwrap() - svc).abs() < 1e-12,
                "{row}"
            );
            assert!((row["trade_cost"].as_f64().unwrap() - 13_500.0 * svc).abs() < 1e-6);
        }
        assert_eq!(d["wait_fallbacks"], 3);
        assert_eq!(d["wait_fallback_counts"]["no_view"], 1);
        // Without a model every P uses the first-version ledger terms.
        let d = pick("q2", None);
        assert_eq!(d["candidates"][0]["wait_fallback_reason"], "no_model");
        let service = d["candidates"][0]["service_s"].as_f64().unwrap();
        assert!((service - 1_000.0 / 13_500.0).abs() < 1e-12);
        assert_eq!(d["candidates"][0]["trade_cost"], 1_000.0);
        assert_eq!(d["wait_fallback_counts"]["no_model"], 3);
        assert_eq!(
            d["wait_fallback_counts"]["stale"], 1,
            "counts are cumulative"
        );
        assert!(d["eft_model_sha256"].is_null());
    }

    /// A P without a usable snapshot keeps the scale of the EFT terms: B_w is
    /// the step-model work of all its active router rows (an unknown one too,
    /// as in the ledger), S_w the step-model service.
    #[test]
    fn trade_eft_fallback_prices_the_ledger_rows_with_the_step_model() {
        let book = CostBook::new_uncached(None).unwrap();
        let (_, mut x) = book
            .reserve_uncached(8_192, Some("x".into()), vec![candidate("a", 0)], 12288)
            .unwrap();
        x.mark_dispatched().unwrap();
        let (_, mut y) = book
            .reserve_uncached(1_000, Some("y".into()), vec![candidate("a", 0)], 12288)
            .unwrap();
        y.mark_dispatched().unwrap();
        y.mark_unknown("prefill_transport_error").unwrap();
        // b has a usable, empty snapshot; a has none.
        let ctx = eft_context(&[("b", routing_snapshot(0, 0), std::time::Duration::ZERO)]);
        let (selected, r, decision) = book
            .reserve_trade_eft(
                2_000,
                Some("q".into()),
                vec![candidate("a", 0), candidate("b", 0)],
                eft_trade(0.0),
                Some(&ctx),
            )
            .unwrap();
        let a = &decision["candidates"][0];
        assert_eq!(a["wait_source"], "ledger_fallback");
        assert_eq!(a["wait_fallback_reason"], "no_view");
        assert_eq!(a["eft_outstanding_rows"], 2);
        assert_eq!(a["eft_outstanding_tokens"], 9_192);
        assert!(a["eft_degraded"].is_null());
        // W(8,192) + W(1,000) + two passes (8,192 + 1,024 page-rounded tokens).
        let wait = 1e-5 * 8_192.0 + 0.004 + 1e-5 * 1_000.0 + 0.004 + 0.05 * 2.0;
        assert!((a["wait_s"].as_f64().unwrap() - wait).abs() < 1e-9, "{a}");
        let svc = 1e-5 * 2_000.0 + 0.004 + 0.05;
        assert!(
            (a["service_s"].as_f64().unwrap() - svc).abs() < 1e-12,
            "{a}"
        );
        assert!((a["trade_cost"].as_f64().unwrap() - 13_500.0 * (wait + svc)).abs() < 1e-6);
        let b = &decision["candidates"][1];
        assert_eq!(b["wait_source"], "eft");
        assert_eq!(b["wait_s"], 0.0);
        assert_eq!(b["eft_cold_rows"], 0);
        assert_eq!(selected.worker_url, "b");
        drop((r, x, y));
    }

    /// wait = eft. The engine adds a request to its in-flight request queue
    /// only after the result of its last chunk; while that chunk is in the
    /// in-flight pass the request is in no snapshot row, and the progress
    /// reports the chunk as remaining work. R0 prices that pass, so the
    /// request is not also an outstanding row (unless R0 is unknown). An
    /// unknown entry is left to the snapshot. A request launched after the
    /// snapshot is outstanding and priced as started (no host restore); a
    /// request in transit at the router's estimate.
    #[test]
    fn trade_eft_leaves_out_requests_in_the_in_flight_pass() {
        for (launch_age, rows, pass_rows, tokens) in
            [(json!(0.0), 2, 1, 9_000), (Value::Null, 3, 0, 11_000)]
        {
            let book = CostBook::new_uncached(None).unwrap();
            let reserve = |rid: &str, n: u64, hit: u64| {
                let (_, mut r) = book
                    .reserve_uncached(n, Some(rid.into()), vec![candidate("a", hit)], 12288)
                    .unwrap();
                r.mark_dispatched().unwrap();
                r
            };
            // fin: its last chunk (2,000 tokens) is in the in-flight pass.
            let fin = reserve("fin", 10_000, 0);
            // race: launched after the snapshot; 6,000 of 14,192 tokens left.
            let race = reserve("race", 14_192, 4_096);
            // lost: transport error after its first chunk launched.
            let mut lost = reserve("lost", 4_000, 0);
            // transit: dispatched, not in the engine yet.
            let transit = reserve("transit", 3_000, 0);
            book.apply_progress(
                "a",
                progress(1, &[("fin", 2_000), ("race", 6_000), ("lost", 1_000)]),
                json!({}),
                Instant::now(),
            )
            .unwrap();
            lost.mark_unknown("prefill_transport_error").unwrap();
            let mut raw = routing_snapshot(0, 0);
            raw["inflight_batches"] = json!([{"forward_iter": 9, "launch_age_s": launch_age,
                "status": "pending", "request_ids": ["fin"], "omitted_requests": 0,
                "features": {"new_tokens": 2_000, "attention_pairs": 0, "batch_size": 1,
                    "host_restore_tokens": 0, "storage_restore_tokens": 0}}]);
            let ctx = eft_context(&[("a", raw, std::time::Duration::ZERO)]);
            let (_, r, decision) = book
                .reserve_trade_eft(
                    1_000,
                    Some("new".into()),
                    vec![candidate("a", 0)],
                    eft_trade(0.0),
                    Some(&ctx),
                )
                .unwrap();
            let a = &decision["candidates"][0];
            assert_eq!(a["wait_source"], "eft");
            assert_eq!(a["eft_outstanding_rows"], rows, "{a}");
            assert_eq!(a["eft_inflight_pass_rows"], pass_rows, "{a}");
            assert_eq!(a["eft_outstanding_tokens"], tokens, "{a}");
            // W(u) = 1e-5 u + 0.004 here; race has no restore term (started).
            let w = |u: f64| 1e-5 * u + 0.004;
            let wait = a["wait_s"].as_f64().unwrap();
            if launch_age.is_null() {
                // No R0: fin counts as a row. Two passes of work ahead.
                let expected = w(2_000.0) + w(6_000.0) + w(3_000.0) + 0.05 * 2.0;
                assert!((wait - expected).abs() < 1e-9, "{wait} {expected}");
            } else {
                // R0 = 0.05 + 1e-5 x 2,000 + 0.004 = 0.074 s, less the view age.
                let expected = 0.074 + w(6_000.0) + w(3_000.0) + 0.05 * 2.0;
                assert!(
                    wait <= expected + 1e-9 && wait > expected - 0.05,
                    "{wait} {expected}"
                );
            }
            drop((r, fin, race, lost, transit));
        }
    }

    #[test]
    fn restore_cost_moves_only_when_work_saved_covers_the_new_restore() {
        for (owner_load, peer_load, expected) in [
            (20000, 19000, "owner"),
            (20000, 0, "peer"),
            (12288, 0, "owner"),
        ] {
            let book = CostBook::new_restore_aware(None, 8).unwrap();
            let _owner = uncached(&book, "owner-load", "owner", owner_load);
            let _peer = uncached(&book, "peer-load", "peer", peer_load);
            // With a third heavily loaded node, minimum range can prefer the
            // slower peer merely to raise the least loaded node's score.
            let _third = uncached(&book, "third-load", "third", 100000);
            let mut owner = candidate("owner", 65536);
            owner.gpu_hit = Some(65536);
            let mut peer = candidate("peer", 0);
            peer.expected_hit = 65536;
            peer.hint = Some(json!({"expected_hit_tokens":65536}));
            let (chosen, _reservation) = book
                .reserve_uncached(
                    69633,
                    Some("new".into()),
                    vec![owner, peer, candidate("third", 0)],
                    12288,
                )
                .unwrap();
            assert_eq!(chosen.worker_url, expected);
            assert!(!book.snapshot()["admission_blocked"].as_bool().unwrap());
        }
    }

    #[test]
    fn restore_progress_does_not_mistake_first_cache_correction_for_finished_copy() {
        let book = CostBook::new_restore_aware(None, 8).unwrap();
        let (_, mut request) = book
            .reserve_uncached(100000, Some("r".into()), vec![candidate("p", 90000)], 12288)
            .unwrap();
        assert_eq!(cost(&book, "p"), 170000);
        request.mark_dispatched().unwrap();
        let sent = Instant::now();
        for (sequence, remaining, expected_cost, expected_restore) in [
            (1, 8000, 154000, 90000), // first actual prefix differs from prediction
            (2, 8000, 154000, 90000), // launch/snapshot is not completion
            (3, 7999, 63992, 0),      // later result retires a chunk
            (4, 0, 0, 0),
        ] {
            book.apply_progress(
                "p",
                progress(sequence, &[("r", remaining)]),
                json!({}),
                sent + std::time::Duration::from_millis(sequence),
            )
            .unwrap();
            assert_eq!(cost(&book, "p"), expected_cost);
            assert_eq!(
                book.state.lock().workers["p"].pending_restore_tokens,
                expected_restore
            );
        }
        request
            .finish("P HTTP completed after native work")
            .unwrap();
        assert_eq!(cost(&book, "p"), 0);
        assert_eq!(book.state.lock().workers["p"].active_requests, 0);
    }

    #[test]
    fn restore_reservations_are_charged_before_progress_and_roll_back_exactly() {
        let book = CostBook::new_restore_aware(None, 8).unwrap();
        let (_, first) = book
            .reserve_uncached(100000, Some("first".into()), vec![candidate("a", 90000)], 0)
            .unwrap();
        assert_eq!(book.state.lock().workers["a"].pending_restore_tokens, 90000);
        let (next, second) = book
            .reserve_uncached(
                100000,
                Some("second".into()),
                vec![candidate("a", 90000), candidate("b", 90000)],
                0,
            )
            .unwrap();
        assert_eq!(next.worker_url, "b");
        drop(first);
        drop(second);
        for worker in book.state.lock().workers.values() {
            assert_eq!(worker.cost, 0);
            assert_eq!(worker.pending_uncached_tokens, 0);
            assert_eq!(worker.pending_restore_tokens, 0);
        }
        assert!(CostBook::new_restore_aware(None, 0).is_err());
    }

    #[test]
    fn uncached_threshold_excludes_arrival_and_does_not_hold_a_busy_pool() {
        for load in [12287, 12288, 12289, 50000] {
            let book = CostBook::new_uncached(None).unwrap();
            let _owner = uncached(&book, "busy-a", "a", load);
            let _b = uncached(&book, "busy-b", "b", 15000);
            let _c = uncached(&book, "busy-c", "c", 20000);
            let mut peer = candidate("b", 0);
            peer.expected_hit = 90000;
            peer.hint = Some(json!({"expected_hit_tokens": 90000}));
            let (selected, _request) = book
                .reserve_uncached(
                    100000,
                    Some("new".into()),
                    vec![candidate("a", 90000), peer, candidate("c", 0)],
                    12288,
                )
                .unwrap();
            // At 12289, A+10000 still produces the smallest spread; at 50000
            // every P is above the threshold, but we immediately choose B.
            assert_eq!(selected.worker_url, if load == 50000 { "b" } else { "a" });
        }
    }

    #[test]
    fn detour_requires_savings_after_restore_and_margin() {
        for (owner_load, expected) in [(16000, "a"), (29000, "a"), (50000, "b")] {
            let book = CostBook::new_restore_aware(None, 8).unwrap();
            let _owner = uncached(&book, "owner-work", "a", owner_load);
            let _peer = uncached(&book, "peer-work", "b", 10000);
            let mut owner = candidate("a", 90000);
            owner.gpu_hit = Some(90000);
            let mut peer = candidate("b", 0);
            peer.gpu_hit = Some(0);
            peer.expected_hit = 90000;
            peer.hint = Some(json!({"expected_hit_tokens": 90000}));
            let (selected, _reservation) = book
                .reserve_uncached_with_margin(
                    100000,
                    Some("arrival".into()),
                    vec![owner, peer],
                    16384,
                    16384,
                )
                .unwrap();
            assert_eq!(selected.worker_url, expected);
        }
    }

    #[test]
    fn uncached_threshold_protects_light_cache_owner_even_for_large_arrival() {
        for load in [12288, 12289] {
            let book = CostBook::new_uncached(None).unwrap();
            let _busy = uncached(&book, "busy", "a", load);
            let mut owner = candidate("a", 90000);
            owner.gpu_hit = Some(90000);
            let (selected, _) = book
                .reserve_uncached(
                    100000,
                    Some("new".into()),
                    vec![owner, candidate("b", 90000)],
                    12288,
                )
                .unwrap();
            assert_eq!(selected.worker_url, if load == 12288 { "a" } else { "b" });
        }
    }

    #[test]
    fn uncached_progress_keeps_unseen_reservations_and_reconciles_actual_miss() {
        let book = CostBook::new_uncached(None).unwrap();
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
        let book = CostBook::new_uncached(None).unwrap();
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
        assert!(book
            .reserve_uncached(1, Some("a".into()), vec![candidate("p", 0)], 12288)
            .is_err());
    }

    #[test]
    fn concurrent_uncached_arrivals_reserve_before_the_next_load_refresh() {
        let book = CostBook::new_uncached(None).unwrap();
        let barrier = Arc::new(Barrier::new(18));
        let threads: Vec<_> = (0..18)
            .map(|i| {
                let book = book.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    book.reserve_uncached(
                        4000,
                        Some(format!("rid-{i}")),
                        vec![candidate("a", 0), candidate("b", 0), candidate("c", 0)],
                        12288,
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
    fn locality_retains_gpu_owner_until_savings_strictly_exceed_margin() {
        for backlog in [32767, 32768, 32769] {
            let book = CostBook::new(None).unwrap();
            let (_, _busy) = book
                .reserve(backlog, None, vec![candidate("owner", 0)])
                .unwrap();
            let mut owner = candidate("owner", 8000);
            owner.gpu_hit = Some(8000);
            // Both copies have the same total prefix; only owner has GPU KV.
            let (chosen, request) = book
                .reserve_locality(10000, None, vec![candidate("peer", 8000), owner], 32768)
                .unwrap();
            assert_eq!(
                chosen.worker_url,
                if backlog > 32768 { "peer" } else { "owner" }
            );
            assert_eq!(
                book.state
                    .lock()
                    .workers
                    .values()
                    .map(|w| w.pending_uncached_tokens)
                    .sum::<u128>(),
                u128::from(backlog) + 2000
            );
            drop(request);
            assert_eq!(
                book.state
                    .lock()
                    .workers
                    .values()
                    .map(|w| w.pending_uncached_tokens)
                    .sum::<u128>(),
                u128::from(backlog)
            );
        }
    }

    #[test]
    fn locality_charges_recomputation_when_no_pull_will_be_forwarded() {
        for hinted in [false, true] {
            let book = CostBook::new(None).unwrap();
            let (_, _busy) = book
                .reserve(40000, None, vec![candidate("owner", 0)])
                .unwrap();
            let mut owner = candidate("owner", 90000);
            owner.gpu_hit = Some(90000);
            let mut peer = candidate("peer", 0);
            if hinted {
                peer.expected_hit = 90000;
                peer.hint = Some(json!({"expected_hit_tokens":90000}));
            }
            let (chosen, _) = book
                .reserve_locality(100000, None, vec![owner, peer], 32768)
                .unwrap();
            assert_eq!(chosen.worker_url, if hinted { "peer" } else { "owner" });
            assert_eq!(chosen.hint.is_some(), hinted);
        }
    }

    #[test]
    fn locality_missing_gpu_data_keeps_total_hit_priority_and_clamps_observations() {
        let book = CostBook::new(None).unwrap();
        let (chosen, _) = book
            .reserve_locality(
                100,
                None,
                vec![candidate("less-hit", 40), candidate("more-hit", 80)],
                32768,
            )
            .unwrap();
        assert_eq!(chosen.worker_url, "more-hit");
        let mut a = candidate("a", 80);
        a.gpu_hit = Some(u64::MAX);
        let (chosen, _) = book
            .reserve_locality(100, None, vec![a, candidate("b", 80)], 32768)
            .unwrap();
        assert_eq!(chosen.gpu_hit, Some(80));
    }

    #[test]
    fn concurrent_locality_reservations_account_for_undispatched_work() {
        let book = CostBook::new(None).unwrap();
        let barrier = Arc::new(Barrier::new(16));
        let threads: Vec<_> = (0..16)
            .map(|_| {
                let book = Arc::clone(&book);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    book.reserve_locality(
                        10,
                        None,
                        vec![candidate("a", 0), candidate("b", 0)],
                        32768,
                    )
                    .unwrap()
                })
            })
            .collect();
        let owned: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(book.state.lock().workers["a"].pending_uncached_tokens, 80);
        assert_eq!(book.state.lock().workers["b"].pending_uncached_tokens, 80);
        drop(owned);
        assert!(book
            .state
            .lock()
            .workers
            .values()
            .all(|w| w.pending_uncached_tokens == 0));
    }

    #[test]
    fn minimizes_post_assignment_range_not_current_load() {
        let book = CostBook::new(None).unwrap();
        let (_, mut busy) = book.reserve(10, None, vec![candidate("b", 9)]).unwrap();
        busy.mark_dispatched().unwrap();
        // A is idle but would cost 100. B has cost 10 and adds only 10.
        let (chosen, _) = book
            .reserve(10, None, vec![candidate("a", 0), candidate("b", 9)])
            .unwrap();
        assert_eq!(chosen.worker_url, "b");
        busy.finish("done").unwrap();
    }

    #[test]
    fn fixed_cost_survives_old_age_and_snapshots() {
        let book = CostBook::new(None).unwrap();
        let (_, mut request) = book.reserve(100, None, vec![candidate("a", 20)]).unwrap();
        request.mark_dispatched().unwrap();
        // Timestamp age never participates in membership or cost calculation.
        book.state
            .lock()
            .entries
            .get_mut(&request.id)
            .unwrap()
            .assigned_unix_ms = 0;
        for _ in 0..3 {
            book.snapshot();
        }
        assert_eq!(cost(&book, "a"), 8_000);
        request.finish("prefill_complete").unwrap();
        assert_eq!(cost(&book, "a"), 0);
    }

    #[test]
    fn drop_before_dispatch_rolls_back_and_retains_worker() {
        let book = CostBook::new(None).unwrap();
        let (_, request) = book.reserve(12, None, vec![candidate("a", 0)]).unwrap();
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
        let book = CostBook::new_uncached(None).unwrap();
        let reserve = || book.reserve_uncached(100, Some("x".into()), vec![candidate("a", 0)], 0);
        drop(reserve().unwrap().1);
        let (_, mut sent) = reserve().unwrap();
        sent.mark_dispatched().unwrap();
        sent.finish("prefill_response_success").unwrap();
        assert!(reserve().is_err());
    }

    #[test]
    fn drop_after_dispatch_retains_unknown_and_blocks_admission() {
        let book = CostBook::new(None).unwrap();
        let (_, mut request) = book.reserve(12, None, vec![candidate("a", 0)]).unwrap();
        request.mark_dispatched().unwrap();
        drop(request);
        assert_eq!(cost(&book, "a"), 144);
        assert!(book.reserve(1, None, vec![candidate("b", 0)]).is_err());
        assert_eq!(
            book.snapshot()["unknown_entries"].as_array().unwrap().len(),
            1
        );
    }

    #[test]
    fn explicit_unknown_can_only_release_on_explicit_terminal_result() {
        let book = CostBook::new(None).unwrap();
        let (_, mut request) = book.reserve(10, None, vec![candidate("a", 0)]).unwrap();
        request.mark_dispatched().unwrap();
        request.mark_unknown("transport_timeout").unwrap();
        request.mark_unknown("duplicate").unwrap();
        assert_eq!(cost(&book, "a"), 100);
        request.finish("engine_abort_confirmed").unwrap();
        request.finish("duplicate_terminal").unwrap();
        assert_eq!(cost(&book, "a"), 0);
        assert!(book.reserve(1, None, vec![candidate("a", 0)]).is_ok());
    }

    #[test]
    fn dispatch_is_idempotent_but_cannot_reopen_unknown_or_terminal_entries() {
        for status in [Status::Unknown, Status::Finished, Status::RolledBack] {
            let book = CostBook::new(None).unwrap();
            let (_, mut request) = book.reserve(10, None, vec![candidate("a", 0)]).unwrap();
            request.mark_dispatched().unwrap();
            request.mark_dispatched().unwrap();
            assert_eq!(cost(&book, "a"), 100);
            book.change_status(request.id, status, "test_transition")
                .unwrap();
            assert!(request.mark_dispatched().is_err());
            assert_eq!(request.debuginfo()["status"], status.label());
            if status == Status::Unknown {
                assert_eq!(cost(&book, "a"), 100);
                request.finish("confirmed_engine_terminal").unwrap();
            }
            request.finish("idempotent_terminal").unwrap();
            assert_eq!(cost(&book, "a"), 0);
        }
    }

    #[test]
    fn zero_cost_is_still_owned_and_counts_for_ties() {
        let book = CostBook::new(None).unwrap();
        let (a, first) = book
            .reserve(10, None, vec![candidate("b", 10), candidate("a", 10)])
            .unwrap();
        let (b, second) = book
            .reserve(10, None, vec![candidate("b", 10), candidate("a", 10)])
            .unwrap();
        assert_eq!(a.worker_url, "a");
        assert_eq!(b.worker_url, "b");
        assert_eq!(first.debuginfo()["cost"], "0");
        assert_eq!(book.state.lock().workers["a"].active_requests, 1);
        drop((first, second));
    }

    #[test]
    fn parallel_reservations_see_each_others_committed_cost() {
        let book = CostBook::new(None).unwrap();
        let barrier = Arc::new(Barrier::new(16));
        let threads: Vec<_> = (0..16)
            .map(|_| {
                let book = Arc::clone(&book);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    book.reserve(10, None, vec![candidate("a", 0), candidate("b", 0)])
                        .unwrap()
                })
            })
            .collect();
        let owned: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert_eq!(cost(&book, "a"), 800);
        assert_eq!(cost(&book, "b"), 800);
        assert_eq!(owned.len(), 16);
    }

    #[test]
    fn large_product_uses_u128_and_total_overflow_is_rejected() {
        let book = CostBook::new(None).unwrap();
        let (_, request) = book
            .reserve(u64::MAX, None, vec![candidate("a", 0)])
            .unwrap();
        let expected = u128::from(u64::MAX) * u128::from(u64::MAX);
        assert_eq!(cost(&book, "a"), expected);
        assert_eq!(request.debuginfo()["cost"], expected.to_string());
        assert!(book
            .reserve(u64::MAX, None, vec![candidate("a", 0)])
            .is_err());
        assert_eq!(cost(&book, "a"), expected);
    }

    #[test]
    fn health_subset_does_not_erase_excluded_workers_entries() {
        let book = CostBook::new(None).unwrap();
        let (_, mut a) = book.reserve(10, None, vec![candidate("a", 0)]).unwrap();
        a.mark_dispatched().unwrap();
        let (_, b) = book.reserve(2, None, vec![candidate("b", 0)]).unwrap();
        assert_eq!(cost(&book, "a"), 100);
        assert_eq!(book.snapshot()["healthy_candidates"], json!(["b"]));
        assert_eq!(book.snapshot()["workers"].as_array().unwrap().len(), 2);
        let (chosen, c) = book
            .reserve(2, None, vec![candidate("a", 0), candidate("b", 0)])
            .unwrap();
        assert_eq!(chosen.worker_url, "b");
        a.finish("old_request_finished_after_recovery").unwrap();
        assert_eq!(cost(&book, "a"), 0);
        assert_eq!(cost(&book, "b"), 8);
        drop((b, c));
    }

    #[test]
    fn hit_clamps_and_remote_hit_requires_real_hint() {
        let book = CostBook::new(None).unwrap();
        let (_, request) = book.reserve(10, None, vec![candidate("a", 999)]).unwrap();
        assert_eq!(request.debuginfo()["expected_hit"], 10);
        assert_eq!(request.debuginfo()["cost"], "0");
        let invalid = Candidate {
            worker_url: "b".into(),
            local_hit: 0,
            gpu_hit: None,
            expected_hit: 8,
            hint: None,
        };
        assert!(book.reserve(10, None, vec![invalid]).is_err());
        let hinted = Candidate {
            worker_url: "b".into(),
            local_hit: 0,
            gpu_hit: None,
            expected_hit: 8,
            hint: Some(json!({"expected_hit_tokens": 8})),
        };
        let (_, remote) = book.reserve(10, None, vec![hinted]).unwrap();
        assert_eq!(remote.debuginfo()["cost"], "20");
    }

    #[test]
    fn trace_is_exclusive_ordered_and_costs_are_strings() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cost.jsonl");
        let book = CostBook::new(Some(&path)).unwrap();
        assert!(CostBook::new(Some(&path)).is_err());
        let (_, mut request) = book
            .reserve(10, Some("rid".into()), vec![candidate("a", 4)])
            .unwrap();
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
        assert_eq!(rows[0]["entry"]["cost"], "60");
        assert_eq!(rows[3]["entry"]["status"], "finished");
    }

    /// A book made with a policy label writes `"policy"` on every trace row
    /// (reserve, dispatch, engine progress, finish, snapshot). The other
    /// constructors write no `policy` key.
    #[test]
    fn policy_label_on_every_trace_event() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("labelled.jsonl");
        let book =
            CostBook::new_uncached_with_policy(Some(&path), "symmetric_pd_cost_function", None)
                .unwrap();
        let (_, mut r, _) = book
            .reserve_trade_session(
                10_000,
                Some("r1".into()),
                vec![candidate("a", 0)],
                trade(18.4, false),
                None,
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
            assert_eq!(row["schema"], "cost-balance.v1", "{row}");
        }
        let path = tmp.path().join("plain.jsonl");
        let plain = CostBook::new_uncached(Some(&path)).unwrap();
        let _r = uncached(&plain, "r2", "a", 1_000);
        plain.snapshot();
        let rows = read(&path);
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.get("policy").is_none()), "{rows:?}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn trace_io_failure_blocks_current_and_future_reservations() {
        let book = CostBook::new(None).unwrap();
        book.state.lock().trace = Some(TraceSink::Sync(BufWriter::new(
            OpenOptions::new().write(true).open("/dev/full").unwrap(),
        )));
        assert!(book.reserve(10, None, vec![candidate("a", 0)]).is_err());
        assert_eq!(cost(&book, "a"), 0);
        assert_eq!(book.state.lock().entries.len(), 0);
        assert!(book.reserve(10, None, vec![candidate("a", 0)]).is_err());
        assert_eq!(book.snapshot()["admission_blocked"], true);
    }

    // ---- TRADE wait = e3 ----

    /// A usable (P, step) fit by hand: a pass takes 10 ms + 0.05 ms per new
    /// token, the launch interval is 0.02 s + 0.9 x the GPU s of the pass
    /// before, idle delay 0.03 s, no receipt line (0.044 s).
    fn e3_fit() -> crate::trade_e3::FitBundle {
        let gm = crate::e3::GpuModel {
            small: Some(vec![10.0, 0.05, 0.0, 0.0, 0.0]),
            large: Some(vec![10.0, 0.05, 0.0, 0.0, 0.0]),
            floor_ms: 1.0,
        };
        crate::trade_e3::FitBundle {
            step: 0.0,
            gpu: gm,
            gpu_sources: crate::e3::GpuSources {
                small: Some("own".into()),
                large: Some("own".into()),
            },
            launch: crate::e3::LaunchFit {
                coef: vec![0.02, 0.9, 0.0, 0.0, 0.0, 0.0],
                resid_q: None,
                source: crate::e3::LaunchSource::Constant,
            },
            constants: crate::e3::Constants {
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

    /// A `wait = e3` context. Per P: the snapshot JSON (None: no snapshot),
    /// its age at the poll, whether its fit has a GPU model, and the target
    /// (prefix, host). The E3 clock starts 100 s ago.
    #[allow(clippy::type_complexity)]
    fn e3_context(
        rows: &[(&str, Option<Value>, std::time::Duration, bool, i64, i64)],
        input: i64,
    ) -> crate::trade_e3::TradeE3Context {
        let path = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/e3/bound-params.json"
        ));
        let cfg = crate::trade_e3::TradeE3Config::load(
            path,
            crate::trade_e3::DEFAULT_BOUND_MODE,
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
                        view: crate::eft_cheap::SnapshotView::parse(raw, 1, None),
                        poll_sent: sent,
                        poll_received: sent,
                        e3: Some(Arc::new(crate::trade_e3::E3View::parse(raw))),
                    })
                });
                let fit = e3_fit();
                let prep = crate::trade_e3::E3Prepared {
                    snapshot,
                    push: None,
                    params: usable.then(|| fit.params(None, 0.35 * 2_067_584.0)),
                    fit: Some(Arc::new(fit)),
                    target: crate::e3::Target {
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
        crate::trade_e3::TradeE3Context {
            cfg: Arc::new(cfg),
            anchor,
            unix_at_anchor: 1_000.0,
            candidates,
        }
    }

    fn e3_trade(lambda: f64) -> crate::scheduler::co_schedule::TradeParams {
        crate::scheduler::co_schedule::TradeParams {
            copy_lambda: lambda,
            wait: crate::scheduler::co_schedule::TradeWait::E3,
            ..Default::default()
        }
    }

    /// wait = e3. Owner a caches 40,000 of 44,000 tokens behind five
    /// 65,536-token rows (40 passes of 0.42 s): pred above 15 s. b is idle
    /// and pulls the 40,000 tokens (0.49 s, mostly exposed): pred about
    /// 0.3 s. At lambda 0 the gain beats the bound margin and the request
    /// moves to b; the trace has the e3 keys. The next decision sees r1
    /// (reserved on b, not in b's snapshot) as a router row of b. After a
    /// progress poll received before the snapshot lists r1, r1 is left out.
    #[test]
    fn trade_e3_moves_through_the_gate_and_prices_unseen_router_rows() {
        let before = Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .unwrap();
        let book = CostBook::new_uncached(None).unwrap();
        let ctx = e3_context(
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
            .reserve_trade_e3(
                44_000,
                Some("r1".into()),
                candidates(),
                e3_trade(0.0),
                Some(&ctx),
            )
            .unwrap();
        assert_eq!(selected.worker_url, "b", "{decision}");
        assert_eq!(decision["wait"], "e3");
        assert_eq!(decision["detour"], true);
        assert_eq!(decision["e3_protected"], false);
        assert_eq!(decision["e3_protect_reason"], Value::Null);
        assert_eq!(decision["e3_forecasts"], 2);
        assert_eq!(decision["e3_bound_level"], "0.8");
        assert_eq!(decision["e3_bound_mode"], "snap+prod");
        assert_eq!(decision["e3_gate_m"], 1.0);
        assert_eq!(decision["wait_fallbacks"], 0);
        let gain = decision["e3_gain_s"].as_f64().unwrap();
        let margin = decision["e3_margin_s"].as_f64().unwrap();
        assert!(gain > margin && margin > 0.0, "{gain} {margin}");
        let (a, b) = (&decision["candidates"][0], &decision["candidates"][1]);
        assert_eq!(a["wait_source"], "e3");
        assert_eq!(a["e3_bucket"], "busy");
        assert_eq!(a["e3_n_queue"], 5);
        assert!(a["e3_pred_s"].as_f64().unwrap() > 15.0, "{a}");
        assert_eq!(b["e3_idle"], true);
        assert_eq!(b["e3_bucket"], "idle");
        assert_eq!(b["e3_router_rows"], 0);
        let (pred_b, wait_b) = (
            b["e3_pred_s"].as_f64().unwrap(),
            b["e3_wait_s"].as_f64().unwrap(),
        );
        assert!(pred_b < 1.0 && wait_b < pred_b, "{b}");
        // C_b = R * (pred_s + max(0, pull_s - wait_s)) at lambda 0.
        let rate = 13_500.0;
        let c_b = rate * wait_b + rate * (pred_b - wait_b) + rate * (0.49 - wait_b).max(0.0);
        assert!(
            (b["trade_cost"].as_f64().unwrap() - c_b).abs() < 1e-6,
            "{b}"
        );
        assert_eq!(decision["e3_chosen"]["pred_s"], b["e3_pred_s"]);
        assert_eq!(decision["e3_chosen"]["target_prefix"], 40_000);
        assert!(decision["e3_decision_s"].as_f64().unwrap() >= 100.0);
        // The state push is off: no push keys.
        assert!(a.get("e3_state_source").is_none() && a.get("e3_push_fallback").is_none());
        assert!(decision.get("e3_push_port").is_none());
        assert!(decision["e3_chosen"].get("state_source").is_none());

        let (_, r2, decision) = book
            .reserve_trade_e3(
                44_000,
                Some("r2".into()),
                candidates(),
                e3_trade(0.0),
                Some(&ctx),
            )
            .unwrap();
        let b = &decision["candidates"][1];
        assert_eq!(b["e3_router_rows"], 1, "{b}");
        assert_eq!(b["e3_n_dispatched"], 1);
        assert_eq!(b["e3_idle"], false);
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
            .reserve_trade_e3(
                44_000,
                Some("r3".into()),
                candidates(),
                e3_trade(0.0),
                Some(&ctx),
            )
            .unwrap();
        assert_eq!(decision["candidates"][1]["e3_router_rows"], 0);
        drop((r1, r3));
    }

    /// Fallbacks: a P without a snapshot (no_view), with an old view
    /// (view_too_old) or without a GPU model (no_fit) keeps the ledger terms
    /// and never wins a move; the trace counts them by reason. An owner
    /// without a forecast is kept (owner_no_forecast); without a context
    /// every P falls back (no_context).
    #[test]
    fn trade_e3_fallbacks_never_win_a_move_and_are_counted() {
        let book = CostBook::new_uncached(None).unwrap();
        let zero = std::time::Duration::ZERO;
        let ctx = e3_context(
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
            .reserve_trade_e3(
                44_000,
                Some("f1".into()),
                candidates,
                e3_trade(0.0),
                Some(&ctx),
            )
            .unwrap();
        assert_eq!(selected.worker_url, "a", "{decision}");
        let rows = &decision["candidates"];
        assert_eq!(rows[1]["wait_fallback_reason"], "no_view");
        assert_eq!(rows[1]["wait_source"], "ledger_fallback");
        assert_eq!(rows[1]["e3_pred_s"], Value::Null);
        assert_eq!(rows[2]["wait_fallback_reason"], "view_too_old");
        assert!(rows[2]["view_age_s"].as_f64().unwrap() > 1.5);
        assert_eq!(decision["wait_fallbacks"], 2);
        assert_eq!(decision["wait_fallback_counts"]["no_view"], 1);
        assert_eq!(decision["e3_forecasts"], 1);
        assert_eq!(
            decision["e3_protect_reason"],
            Value::Null,
            "only a has a forecast"
        );
        drop(r1);

        let ctx = e3_context(
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
            .reserve_trade_e3(
                44_000,
                Some("f2".into()),
                candidates(),
                e3_trade(0.0),
                Some(&ctx),
            )
            .unwrap();
        assert_eq!(selected.worker_url, "a");
        assert_eq!(decision["candidates"][0]["wait_fallback_reason"], "no_fit");
        assert_eq!(decision["best"], "b");
        assert_eq!(decision["e3_protect_reason"], "owner_no_forecast");
        assert_eq!(decision["e3_protected"], true);
        // Counts accumulate over decisions.
        assert_eq!(decision["wait_fallback_counts"]["no_fit"], 1);
        assert_eq!(decision["wait_fallback_counts"]["no_view"], 1);
        drop(r2);

        let (selected, r3, decision) = book
            .reserve_trade_e3(44_000, Some("f3".into()), candidates(), e3_trade(0.0), None)
            .unwrap();
        assert_eq!(selected.worker_url, "a");
        assert_eq!(decision["wait_fallback_counts"]["no_context"], 2);
        assert_eq!(decision["e3_decision_s"], Value::Null);
        drop(r3);
    }

    /// wait = e3 and the engine policy: a P whose snapshot has another pass
    /// budget than the bound tables' fit has no forecast
    /// (policy_not_bound_fit). A snapshot that the engine marks stale with a
    /// known age is used (the view-age limit decides); the flag is in
    /// e3_degraded.
    #[test]
    fn trade_e3_needs_the_bound_fit_policy_and_ignores_the_engine_stale_flag() {
        let book = CostBook::new_uncached(None).unwrap();
        let zero = std::time::Duration::ZERO;
        let mut other = routing_snapshot(0, 0);
        other["policy"]["chunk_tokens"] = json!(16_384);
        let mut stale = routing_snapshot(0, 0);
        stale["stale"] = json!(true);
        let ctx = e3_context(
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
            .reserve_trade_e3(
                44_000,
                Some("q1".into()),
                candidates,
                e3_trade(0.0),
                Some(&ctx),
            )
            .unwrap();
        let rows = &decision["candidates"];
        assert_eq!(rows[0]["wait_source"], "e3", "{decision}");
        assert_eq!(rows[0]["e3_degraded"], json!([]));
        assert_eq!(rows[0]["e3_pull_rows"], 0);
        assert_eq!(rows[1]["wait_fallback_reason"], "policy_not_bound_fit");
        assert_eq!(rows[1]["e3_pred_s"], Value::Null);
        assert_eq!(rows[2]["wait_source"], "e3", "{decision}");
        assert_eq!(rows[2]["e3_degraded"], json!(["engine_stale"]));
        assert_eq!(decision["wait_fallback_counts"]["policy_not_bound_fit"], 1);
        drop(r);
    }

    /// A `wait = e3` context with the state push on (push table
    /// pushL+prodL): P `a` (polled: five 65,536-token rows) with the pushed
    /// state `a_push` = (snapshot, epoch, its age at the context), P `b`
    /// (polled: idle) without one.
    fn e3_push_context(a_push: Option<(Value, &str, f64)>) -> crate::trade_e3::TradeE3Context {
        let zero = std::time::Duration::ZERO;
        let mut ctx = e3_context(
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
            let t0_s = crate::trade_e3::secs_since(ctx.anchor, Instant::now()) - age;
            let state = crate::trade_e3_push::PushState {
                epoch: epoch.to_string(),
                seq: 9,
                trigger: "launch".into(),
                snapshot_source: Some("push_build".into()),
                view: Arc::new(crate::trade_e3::E3View::parse(&raw)),
                t0_s,
                received_s: t0_s + 0.001,
                offset_s: -5.0,
                superseded_seq: None,
            };
            ctx.candidates.get_mut("a").unwrap().push = Some(Arc::new(state));
        }
        ctx
    }

    /// wait = e3 with the state push. P `a` is busy in its polled snapshot
    /// and idle in its pushed state. A fresh pushed state of the polled
    /// epoch is the input (source push, the push table's bound); an old
    /// one, one of another epoch, an unusable one or none keeps the poll
    /// and names the reason. The decision counts the sources and the
    /// reasons over decisions.
    #[test]
    fn trade_e3_uses_a_fresh_verified_push_and_falls_back_to_the_poll() {
        let book = CostBook::new_uncached(None).unwrap();
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
            let ctx = e3_push_context(push);
            let (_, r, decision) = book
                .reserve_trade_e3(
                    44_000,
                    Some(format!("p{i}")),
                    candidates(),
                    e3_trade(0.0),
                    Some(&ctx),
                )
                .unwrap();
            let (a, b) = (&decision["candidates"][0], &decision["candidates"][1]);
            assert_eq!(a["e3_push_fallback"], json!(fallback), "{i}: {a}");
            assert_eq!(b["e3_push_fallback"], "no_push");
            assert_eq!(b["e3_state_source"], "poll");
            let pushed = fallback.is_none();
            assert_eq!(a["e3_state_source"], if pushed { "push" } else { "poll" });
            // The pushed state is idle; the polled one has five rows.
            assert_eq!(a["e3_n_queue"], if pushed { 0 } else { 5 }, "{i}: {a}");
            assert_eq!(
                a["e3_push_seq"],
                if has_push { json!(9) } else { Value::Null }
            );
            if pushed {
                assert_eq!(a["view_key"], json!(["routing-1", 9]));
                let age = a["view_age_s"].as_f64().unwrap();
                assert!((0.01..0.5).contains(&age), "{age}");
                assert_eq!(a["e3_bucket"], "idle");
                let view = crate::e3::BucketView {
                    pred_s: a["e3_pred_s"].as_f64(),
                    idle: true,
                    wait_s: None,
                    n_inflight: 0,
                    partial: false,
                    n_queue: 0,
                };
                let push = ctx.cfg.push.as_ref().unwrap();
                let want = crate::e3::error_bound(&view, &push.table, &push.level).unwrap();
                assert_eq!(a["e3_bound_s"].as_f64(), want);
                let poll = crate::e3::error_bound(&view, &ctx.cfg.table, &ctx.cfg.level).unwrap();
                assert_ne!(want, poll, "the push table is not the poll table");
            }
            if fallback == Some("push_too_old") {
                assert!(a["e3_push_age_s"].as_f64().unwrap() >= 2.0);
            }
            if fallback == Some("push_unusable") {
                assert_eq!(a["e3_push_unusable"], "truncated");
            }
            drop(r);
            last = decision;
        }
        assert_eq!(last["e3_push_port"], 5600);
        assert_eq!(last["e3_push_bound_mode"], "pushL+prodL");
        assert_eq!(last["e3_push_bound_level"], "0.8");
        assert_eq!(last["e3_chosen"]["state_source"], "poll");
        let sources = &last["e3_state_source_counts"];
        assert_eq!((&sources["push"], &sources["poll"]), (&json!(1), &json!(9)));
        let reasons = &last["e3_push_fallback_counts"];
        assert_eq!(reasons["no_push"], 6, "{reasons}");
        for reason in ["push_too_old", "push_unverified", "push_unusable"] {
            assert_eq!(reasons[reason], 1, "{reasons}");
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn dispatch_trace_failure_keeps_undispatched_reservation_rollbackable() {
        let book = CostBook::new(None).unwrap();
        let (_, mut request) = book.reserve(10, None, vec![candidate("a", 0)]).unwrap();
        book.state.lock().trace = Some(TraceSink::Sync(BufWriter::new(
            OpenOptions::new().write(true).open("/dev/full").unwrap(),
        )));
        assert!(request.mark_dispatched().is_err());
        assert_eq!(request.debuginfo()["status"], "reserved");
        drop(request);
        assert_eq!(cost(&book, "a"), 0);
        assert!(book.snapshot()["unknown_entries"]
            .as_array()
            .unwrap()
            .is_empty());
    }

    /// The ledger counters of one worker: cost, pending uncached, pending
    /// restore and unacknowledged tokens, active requests.
    fn ledger(book: &CostBook, url: &str) -> (u128, u128, u128, u128, u64) {
        let st = book.state.lock();
        let w = &st.workers[url];
        (
            w.cost,
            w.pending_uncached_tokens,
            w.pending_restore_tokens,
            w.pending_unacknowledged_tokens,
            w.active_requests,
        )
    }

    /// `--symmetric-pd-loads-host-room` is observation only. Without
    /// [`CostBook::enable_host_room`] a section changes nothing and neither
    /// the snapshot nor a decision has a host_room key. With it, every TRADE
    /// decision has the latest section of each candidate with its age (null
    /// without one), and the snapshot has every P with a section, the
    /// rejected P and the count of invalid sections. An older reply does not
    /// replace the kept section, a non-object is counted and ignored, and a
    /// section clears the rejected mark. The decision is otherwise identical.
    #[test]
    fn host_room_is_traced_only_when_enabled_and_never_priced() {
        let room = |free: u64| json!({"free_tokens": free, "class1_tokens": 600});
        let decide = |book: &Arc<CostBook>| -> (String, Value) {
            let (selected, _r, decision) = book
                .reserve_trade(
                    110_000,
                    Some("req".into()),
                    vec![candidate("a", 100_000), pulling("b", 0, 100_000)],
                    trade(9.2, false),
                )
                .unwrap();
            (selected.worker_url, decision)
        };
        let off = CostBook::new_uncached(None).unwrap();
        off.note_host_room("a", room(100), Instant::now());
        assert!(!off.note_host_room_rejected("b"));
        let (chosen_off, decision_off) = decide(&off);
        assert!(decision_off.get("host_room_by_p").is_none());
        let snapshot = off.snapshot();
        for key in ["host_room_by_p", "host_room_rejected", "host_room_invalid"] {
            assert!(snapshot.get(key).is_none(), "{key}");
        }

        let on = CostBook::new_uncached(None).unwrap();
        on.enable_host_room();
        let sent = Instant::now();
        on.note_host_room("a", room(100), sent);
        on.note_host_room("a", room(999), sent - secs(1));
        on.note_host_room("a", json!([1, 2]), sent + secs(1));
        assert!(on.note_host_room_rejected("b"));
        assert!(!on.note_host_room_rejected("b"), "logged once");
        let (chosen_on, decision_on) = decide(&on);
        assert_eq!(chosen_on, chosen_off);
        let rows = &decision_on["host_room_by_p"];
        assert_eq!(rows["a"]["free_tokens"], 100);
        assert_eq!(rows["a"]["class1_tokens"], 600);
        assert!(rows["a"]["age_s"].as_f64().unwrap() >= 0.0);
        assert!(rows.as_object().unwrap().contains_key("b") && rows["b"].is_null());
        let mut without = decision_on.clone();
        without.as_object_mut().unwrap().remove("host_room_by_p");
        assert_eq!(without, decision_off, "the section is not a decision input");
        let snapshot = on.snapshot();
        assert_eq!(snapshot["host_room_by_p"]["a"]["free_tokens"], 100);
        assert!(snapshot["host_room_by_p"].get("b").is_none());
        assert_eq!(snapshot["host_room_rejected"], json!(["b"]));
        assert_eq!(snapshot["host_room_invalid"], 1);

        on.note_host_room("b", room(7), sent);
        on.note_host_room("a", room(50), sent + secs(1));
        let snapshot = on.snapshot();
        assert_eq!(snapshot["host_room_rejected"], json!([]));
        assert_eq!(snapshot["host_room_by_p"]["a"]["free_tokens"], 50);
        assert_eq!(snapshot["host_room_by_p"]["b"]["free_tokens"], 7);
        assert!(on.note_host_room_rejected("b"), "a new rejection is logged");
    }

    /// Pull price phi: candidate P pays lambda_q x phi_P per pulled token.
    /// lambda_q = the copy price auto before the clamp: the fallback 18.4
    /// during start-up, then q (mu 1, R = R_hat). The owner `a` has 60k
    /// pending; `b` evicted 300 class-0 and 100 class-1 host tokens (phi
    /// 0.25), `c` evicted none (phi 0: a free pull), `d` has no host_room
    /// data (it pays lambda_q, reason no_host_room). With the floor min 0
    /// (the lambda = q words) the request moves to `c` with a pull, where the
    /// lambda price keeps it on `a`. The lambda price writes no phi key, and
    /// its copy charge is lambda x pulled. The decision's copy_lambda is the
    /// price of the chosen P.
    #[test]
    fn trade_phi_prices_each_candidate_with_q_times_its_phi() {
        use crate::scheduler::co_schedule::{TradeLambdaMode, TradePullPrice, TradeRateMode};
        let Some(t0) = Instant::now().checked_sub(std::time::Duration::from_secs(600)) else {
            return; // the monotonic clock is younger than 10 minutes
        };
        let at = |s: f64| t0 + std::time::Duration::from_secs_f64(s);
        let lambda_params = crate::scheduler::co_schedule::TradeParams {
            rate_mode: TradeRateMode::Auto,
            ..auto_trade(18.4)
        };
        let TradeLambdaMode::Auto { q, mu, max, .. } = lambda_params.copy_lambda_mode else {
            unreachable!("auto_trade is auto");
        };
        let phi_params = crate::scheduler::co_schedule::TradeParams {
            pull_price: TradePullPrice::Phi,
            copy_lambda_mode: TradeLambdaMode::Auto {
                q,
                mu,
                min: 0.0,
                max,
            },
            ..lambda_params
        };
        let room =
            |c0: u64, c1: u64| json!({"evicted_class0_tokens": c0, "evicted_class1_tokens": c1});
        let candidates = || {
            vec![
                candidate("a", 100_000),
                pulling("b", 0, 100_000),
                pulling("c", 0, 100_000),
                pulling("d", 0, 100_000),
            ]
        };
        let decide = |book: &Arc<CostBook>, rid: &str, params| {
            let (_, r, d) = book
                .reserve_trade_session(
                    110_000,
                    Some(rid.into()),
                    candidates(),
                    params,
                    None,
                    Some("0/agent-1"),
                )
                .unwrap();
            drop(r);
            d
        };
        let setup = |phi: bool| {
            let book = CostBook::new_uncached(None).unwrap();
            book.enable_host_room();
            if phi {
                book.enable_host_room_phi(120.0);
            }
            for (url, first, second) in [("b", (0, 0), (300, 100)), ("c", (50, 50), (50, 50))] {
                book.note_host_room(url, room(first.0, first.1), at(589.0));
                book.note_host_room(url, room(second.0, second.1), at(590.0));
            }
            // A section without the counters: no phi for `a`.
            book.note_host_room("a", json!({"free_tokens": 0}), at(590.0));
            let mut busy = uncached(&book, "busy", "a", 60_000);
            busy.mark_dispatched().unwrap();
            (book, busy)
        };

        // The lambda price: one lambda (the fallback 18.4), no phi key.
        let (book, _busy) = setup(false);
        let d = decide(&book, "l1", lambda_params);
        assert_eq!(d["chosen"], "a", "{d}");
        for key in [
            "pull_price",
            "phi_lambda_base",
            "phi_base_source",
            "phi_tau_s",
        ] {
            assert!(d.get(key).is_none(), "{key}: {d}");
        }
        for row in d["candidates"].as_array().unwrap() {
            assert!(row.get("copy_lambda").is_none() && row.get("phi_hat").is_none());
        }
        let copy_d = d["candidates"][3]["copy_tokens"].as_f64().unwrap();
        assert!((copy_d - 18.4 * 100_000.0).abs() < 1e-6, "{d}");
        assert!(book.snapshot().get("phi_by_p").is_none());

        // The phi price during start-up: lambda_q = the fallback 18.4.
        let (book, _busy) = setup(true);
        let d = decide(&book, "p1", phi_params);
        assert_eq!(d["pull_price"], "phi");
        assert_eq!(d["phi_base_source"], "startup");
        assert_eq!(d["phi_lambda_base"], 18.4);
        assert_eq!(d["phi_tau_s"], 120.0);
        let rows = d["candidates"].as_array().unwrap();
        let col = |key: &str| rows.iter().map(|r| r[key].clone()).collect::<Vec<_>>();
        assert_eq!(
            col("phi_source"),
            [
                json!("no_host_room"),
                json!("phi"),
                json!("no_eviction"),
                json!("no_host_room")
            ]
        );
        let lambda_b = rows[1]["copy_lambda"].as_f64().unwrap();
        assert!(
            (rows[1]["phi_hat"].as_f64().unwrap() - 0.25).abs() < 1e-12,
            "{d}"
        );
        assert!((lambda_b - 18.4 * 0.25).abs() < 1e-9, "{d}");
        assert_eq!(rows[2]["copy_lambda"], 0.0);
        assert_eq!(rows[2]["phi_hat"], 0.0);
        assert_eq!(rows[3]["copy_lambda"], 18.4);
        assert!(rows[3]["phi_hat"].is_null());
        let copy_b = rows[1]["copy_tokens"].as_f64().unwrap();
        assert!((copy_b - lambda_b * 100_000.0).abs() < 1e-6, "{d}");
        assert_eq!(rows[2]["copy_tokens"], 0.0);
        assert_eq!(d["chosen"], "c", "the free pull moves the request: {d}");
        assert_eq!(d["pulled_tokens"], 100_000);
        // The decision reports the price of the chosen P (and the auto one).
        assert_eq!(d["copy_lambda"], 0.0, "{d}");
        assert_eq!(d["copy_lambda_used"], 0.0, "{d}");
        assert_eq!(d["copy_lambda_auto"], 18.4, "{d}");
        assert_eq!(d["phi_lambda_min"], 0.0, "{d}");
        assert_eq!(d["phi_floored"], false, "{d}");
        assert!(rows.iter().all(|r| r["phi_floored"] == false), "{d}");
        let snapshot = book.snapshot();
        let phi_b = &snapshot["phi_by_p"]["b"];
        assert!((phi_b["phi_hat"].as_f64().unwrap() - 0.25).abs() < 1e-12);
        assert_eq!(snapshot["phi_by_p"]["c"]["source"], "no_eviction");
        assert_eq!(snapshot["phi_by_p"]["a"]["source"], "no_host_room");
        assert_eq!(snapshot["phi_invalid"], 1);

        // After tau of auto history: lambda_q = q (mu 1, R = R_hat).
        let mut auto = crate::scheduler::co_schedule::TradeAutoState::new(t0, 120.0);
        auto.observe_pool(["a", "b", "c", "d"], t0);
        let mut t = 0.0;
        while t < 590.0 {
            for p in ["a", "b", "c", "d"] {
                auto.dispatched(p, at(t));
                auto.prefill_done(p, Some(30_000), at(t + 3.0));
            }
            t += 4.0;
        }
        for i in 0..60u64 {
            auto.session_started(i, at(594.0));
            auto.session_finished(i, at(595.0));
        }
        book.state.lock().trade_auto = Some(auto);
        let d = decide(&book, "p2", phi_params);
        assert!(d["fallback_reason"].is_null(), "{d}");
        assert_eq!(d["phi_base_source"], "q");
        let q = d["q"].as_f64().unwrap();
        assert_eq!(d["lambda_raw"], q, "R = R_hat and mu 1: {d}");
        assert_eq!(d["phi_lambda_base"], q);
        let rows = d["candidates"].as_array().unwrap();
        let lambda_b = rows[1]["copy_lambda"].as_f64().unwrap();
        assert!((lambda_b - q * 0.25).abs() < 1e-9, "{d}");
        assert_eq!(rows[3]["copy_lambda"], q);
        assert_eq!(d["chosen"], "c", "{d}");
        assert_eq!(
            book.snapshot()["phi_prices"]["rows_zero"],
            2,
            "c in both decisions"
        );
    }

    /// The floor of the pull price phi: lambda_P = max(min, lambda_q x
    /// phi_P), min = --co-schedule-trade-copy-lambda-min (4.6). `c` (no
    /// eviction, phi 0) pays 4.6 (floored) instead of 0, `b` (phi 0.25)
    /// 18.4 x 0.25 = 4.6 (not floored), `d` (no host_room data) 18.4. The
    /// floored pull of 100k tokens costs 460k: the request stays on the owner.
    /// The snapshot counts the priced rows, the floored rows and the
    /// decisions whose chosen P was floored.
    #[test]
    fn trade_phi_floor_is_the_auto_min() {
        use crate::scheduler::co_schedule::{TradePullPrice, TradeRateMode};
        let Some(t0) = Instant::now().checked_sub(std::time::Duration::from_secs(600)) else {
            return; // the monotonic clock is younger than 10 minutes
        };
        let at = |s: f64| t0 + std::time::Duration::from_secs_f64(s);
        let params = crate::scheduler::co_schedule::TradeParams {
            rate_mode: TradeRateMode::Auto,
            pull_price: TradePullPrice::Phi,
            ..auto_trade(18.4)
        };
        let room =
            |c0: u64, c1: u64| json!({"evicted_class0_tokens": c0, "evicted_class1_tokens": c1});
        let book = CostBook::new_uncached(None).unwrap();
        book.enable_host_room();
        book.enable_host_room_phi(120.0);
        for (url, first, second) in [("b", (0, 0), (300, 100)), ("c", (50, 50), (50, 50))] {
            book.note_host_room(url, room(first.0, first.1), at(589.0));
            book.note_host_room(url, room(second.0, second.1), at(590.0));
        }
        let mut busy = uncached(&book, "busy", "a", 60_000);
        busy.mark_dispatched().unwrap();
        let (_, r, d) = book
            .reserve_trade_session(
                110_000,
                Some("f1".into()),
                vec![
                    candidate("a", 100_000),
                    pulling("b", 0, 100_000),
                    pulling("c", 0, 100_000),
                    pulling("d", 0, 100_000),
                ],
                params,
                None,
                Some("0/agent-1"),
            )
            .unwrap();
        drop(r);
        let rows = d["candidates"].as_array().unwrap();
        let col = |key: &str| rows.iter().map(|r| r[key].clone()).collect::<Vec<_>>();
        assert_eq!(d["phi_lambda_min"], 4.6, "{d}");
        assert_eq!(rows[2]["copy_lambda"], 4.6, "{d}");
        assert_eq!(rows[1]["copy_lambda"], 4.6, "{d}");
        assert_eq!(rows[3]["copy_lambda"], 18.4, "{d}");
        assert_eq!(
            col("phi_floored"),
            [json!(false), json!(false), json!(true), json!(false)]
        );
        let copy_c = rows[2]["copy_tokens"].as_f64().unwrap();
        assert!((copy_c - 460_000.0).abs() < 1e-6, "{d}");
        assert_eq!(d["chosen"], "a", "the floor keeps the owner: {d}");
        assert_eq!(
            d["copy_lambda"], 18.4,
            "the owner's price (no host_room data)"
        );
        assert_eq!(d["phi_floored"], false);
        let prices = &book.snapshot()["phi_prices"];
        assert_eq!(
            (
                prices["decisions"].clone(),
                prices["rows"].clone(),
                prices["rows_floored"].clone(),
                prices["rows_zero"].clone(),
                prices["chosen_floored"].clone()
            ),
            (json!(1), json!(4), json!(1), json!(0), json!(0))
        );
    }

    /// A reply that accepted the include but had no section counts per P;
    /// only the first one of each P asks the caller to log.
    #[test]
    fn host_room_missing_is_counted_per_p() {
        let book = CostBook::new_uncached(None).unwrap();
        assert!(
            !book.note_host_room_missing("a"),
            "no host_room book: nothing"
        );
        book.enable_host_room();
        assert!(book.note_host_room_missing("a"));
        assert!(!book.note_host_room_missing("a"));
        assert!(book.note_host_room_missing("b"));
        assert_eq!(
            book.snapshot()["host_room_missing"],
            json!({"a": 2, "b": 1})
        );
    }

    // ---- --symmetric-pd-gwq-auto (GWQ-AUTO) ----

    /// The lead rule of one P: depth = floor(C + R x T_react) + 1 and lead =
    /// C / R + T_react, so room (queued < depth) is W = queued / R <= lead.
    /// R 0 leaves lead = T_react and depth C + 1; a non-finite T_react
    /// counts as 0.
    #[test]
    fn gwq_lead_row_depth_and_lead() {
        let row = GwqLeadRow::new(8192, "server_info", 0.5, "measured", 3, 13_500.0);
        assert_eq!(row.depth, 8192 + 6750 + 1);
        assert!((row.lead_s - (8192.0 / 13_500.0 + 0.5)).abs() < 1e-12);
        assert_eq!(
            (row.chunk_source, row.react_source, row.react_samples),
            ("server_info", "measured", 3)
        );
        let idle = GwqLeadRow::new(8192, "default", 0.5, "fallback", 0, 0.0);
        assert_eq!((idle.depth, idle.lead_s), (8193, 0.5));
        let nan = GwqLeadRow::new(4096, "default", f64::NAN, "measured", 1, 13_500.0);
        assert_eq!((nan.depth, nan.react_s), (4097, 0.0));
        assert!((nan.lead_s - 4096.0 / 13_500.0).abs() < 1e-12);
    }

    /// T_react of a P: the fallback before the first sample, then the
    /// time-weighted mean (each sample decays as exp(-age / tau)).
    #[test]
    fn gwq_lead_book_is_a_time_weighted_mean() {
        let book = CostBook::new_uncached(None).unwrap();
        book.enable_gwq_lead(120.0, 0.6);
        let mut state = book.state.lock();
        let lead = state.gwq_lead.as_mut().unwrap();
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

    fn gwq_auto_args<'a>(
        urls: &'a [String],
        depth: &'a [u64],
        chunks: &'a [(u64, &'static str)],
    ) -> GwqStepArgs<'a> {
        GwqStepArgs {
            auto: Some(GwqAutoStep { chunks }),
            ..gwq_args(crate::global_queue::GlobalQueueMode::Hrrn, urls, depth)
        }
    }

    /// GWQ-AUTO release rule on one P with C 8192, R 13,500 and the
    /// fallback T_react 0.5 s: depth = 8192 + 6750 + 1 = 14,943 (the given
    /// depth is ignored). A launched request with 23,135 left gives queued
    /// 14,943 > lead x R: hold. With 23,134 left, queued 14,942 (W equals
    /// lead up to 1/R): release, with the lead rule in the `gwq` object. No
    /// sample (P had queued work) and no shadow data (no host_room).
    #[test]
    fn gwq_auto_releases_iff_the_wait_fits_the_lead() {
        use crate::global_queue::GwqRelease;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("gwqa.jsonl");
        let book = CostBook::new_uncached(Some(&path)).unwrap();
        book.enable_gwq_lead(120.0, 0.5);
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
        let held = [gwq_held("r", 5000, &cands, None, false)];
        let step = book
            .gwq_step(
                trade(9.2, false),
                &held,
                &gwq_auto_args(&urls, &depth, &chunks),
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
            .gwq_step(
                trade(9.2, false),
                &held,
                &gwq_auto_args(&urls, &depth, &chunks),
            )
            .unwrap();
        let (_, result) = step.release.expect("queued 14,942: room");
        let released = result.unwrap();
        assert_eq!(released.release, GwqRelease::Room);
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
        assert_eq!(g["shadow"]["ii_would_hold"], Value::Null);
        assert_eq!(g["shadow"]["insert_tokens"], 5000);
        let stats = book.gwq_lead_rows(&urls);
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
        assert_eq!(reserve["gwq"]["rule"], "auto");
        drop((on_a, released));
    }

    /// T_react samples come only from releases to an idle P with a fresh
    /// view: r1 (P empty) is sampled at its first sighting (sent 300 ms after
    /// the release); r2 (P running r1) is not.
    #[test]
    fn gwq_auto_samples_t_react_only_at_an_empty_queue() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("gwqa-sample.jsonl");
        let book = CostBook::new_uncached(Some(&path)).unwrap();
        book.enable_gwq_lead(120.0, 0.5);
        let t0 = Instant::now();
        book.apply_progress("a", progress(1, &[]), json!({}), t0)
            .unwrap();
        let urls = vec!["a".to_string()];
        let depth = [1u64];
        let chunks = [(8192u64, "default")];
        let cands = vec![candidate("a", 0)];
        let release = |rid: &'static str| {
            let held = [gwq_held(rid, 20_000, &cands, None, true)];
            let step = book
                .gwq_step(
                    trade(9.2, false),
                    &held,
                    &gwq_auto_args(&urls, &depth, &chunks),
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
        let (react_s, source, samples) = book.state.lock().gwq_lead.as_ref().unwrap().react("a");
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
        assert_eq!(book.state.lock().gwq_lead.as_ref().unwrap().react("a").2, 1);
        let samples: Vec<Value> = trace_lines(&path)
            .into_iter()
            .filter(|e| e["event"] == "gwq_lead_sample")
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
        let stats = book.gwq_lead_rows(&urls);
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
    fn gwq_release_uses_the_step_pool() {
        use crate::global_queue::GlobalQueueMode;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("gwq-pool.jsonl");
        let book = CostBook::new_uncached(Some(&path)).unwrap();
        let from_a = Candidate {
            hint: Some(json!({"source_worker_id": "a", "expected_hit_tokens": 9000})),
            ..pulling("b", 0, 9000)
        };
        let cands = vec![candidate("a", 9000), from_a];
        let held = [gwq_held("r", 12_000, &cands, Some("a"), false)];
        let urls = vec!["b".to_string(), "c".to_string()];
        let depth = [1u64 << 40, 1 << 40];
        let step = book
            .gwq_step(
                trade(9.2, false),
                &held,
                &gwq_args(GlobalQueueMode::Hrrn, &urls, &depth),
            )
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
        assert!(reserve["gwq"].get("walk_mismatch").is_none(), "{reserve}");
        assert_eq!(
            reserve["gwq"]["p_release"].as_str(),
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
    fn gwq_release_keeps_a_pull_source_listed_in_sources() {
        use crate::global_queue::GlobalQueueMode;
        let from_d = Candidate {
            hint: Some(json!({"source_worker_id": "d", "expected_hit_tokens": 9000})),
            ..pulling("b", 0, 9000)
        };
        let cands = vec![from_d];
        let urls = vec!["b".to_string()];
        let depth = [1u64 << 40];
        for (sources, kept) in [(vec!["d".to_string()], true), (Vec::new(), false)] {
            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path().join("gwq-sources.jsonl");
            let book = CostBook::new_uncached(Some(&path)).unwrap();
            let held = [gwq_held("r", 12_000, &cands, Some("b"), false)];
            let args = GwqStepArgs {
                sources: &sources,
                ..gwq_args(GlobalQueueMode::Hrrn, &urls, &depth)
            };
            let step = book.gwq_step(trade(9.2, false), &held, &args).unwrap();
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
    /// client left after the step) rolls back when it drops: the P ledger
    /// returns to its value before the release.
    #[test]
    fn gwq_unsent_release_rolls_back() {
        use crate::global_queue::GlobalQueueMode;
        let book = CostBook::new_uncached(None).unwrap();
        let urls = vec!["a".to_string()];
        let depth = [1u64 << 40];
        let cands = vec![candidate("a", 0)];
        let held = [gwq_held("r", 7000, &cands, None, true)];
        let step = book
            .gwq_step(
                trade(9.2, false),
                &held,
                &gwq_args(GlobalQueueMode::Hrrn, &urls, &depth),
            )
            .unwrap();
        let released = step.release.unwrap().1.unwrap();
        let pending = |b: &CostBook| b.state.lock().workers["a"].pending_uncached_tokens;
        assert_eq!(pending(&book), 7000);
        drop(released);
        assert_eq!(pending(&book), 0, "rolled back");
        assert_eq!(book.state.lock().workers["a"].active_requests, 0);
    }

    /// The shadow of a release (no action): phi 0.75 > 1/2 and an insert of
    /// 4,000 (5,000 minus the P's own hit 1,000) above free 500 + class 0
    /// 1,000, so the working-set hold (ii) would hold it.
    #[test]
    fn gwq_shadow_reports_the_ii_hold() {
        let Some(t0) = Instant::now().checked_sub(std::time::Duration::from_secs(600)) else {
            return; // the monotonic clock is younger than 10 minutes
        };
        let at = |s: f64| t0 + std::time::Duration::from_secs_f64(s);
        let book = CostBook::new_uncached(None).unwrap();
        book.enable_host_room();
        book.enable_host_room_phi(120.0);
        let room = |c0: u64, c1: u64| {
            json!({"free_tokens": 500, "class0_tokens": 1000,
                "evicted_class0_tokens": c0, "evicted_class1_tokens": c1})
        };
        book.note_host_room("a", room(0, 0), at(589.0));
        book.note_host_room("a", room(100, 300), at(590.0));
        let cands = vec![candidate("a", 1000)];
        let held = gwq_held("r", 5000, &cands, None, false);
        let state = book.state.lock();
        let shadow = CostBook::gwq_shadow(&state, held, "a", at(590.5));
        assert!(
            (shadow["phi"].as_f64().unwrap() - 0.75).abs() < 1e-9,
            "{shadow}"
        );
        assert_eq!(shadow["insert_tokens"], 4000);
        assert_eq!(shadow["phi_over_half"], true);
        assert_eq!(shadow["ii_would_hold"], true);
        let other = CostBook::gwq_shadow(&state, held, "b", at(590.5));
        assert_eq!(
            (other["phi"].clone(), other["ii_would_hold"].clone()),
            (Value::Null, Value::Null)
        );
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
            let book = CostBook::new_uncached(Some(&path)).unwrap();
            if on {
                book.enable_session_moves();
            }
            let cands = || vec![candidate("a", 4000), candidate("b", 0)];
            let session = Some("0/agent-1");
            let (_, r1, d1) = book
                .reserve_trade_session(
                    5000,
                    Some("s1".into()),
                    cands(),
                    trade(9.2, false),
                    None,
                    session,
                )
                .unwrap();
            assert_eq!(d1["chosen"], "a");
            let mut busy = uncached(&book, "busy", "a", 60_000);
            busy.mark_dispatched().unwrap();
            let (_, r2, d2) = book
                .reserve_trade_session(
                    5000,
                    Some("s2".into()),
                    cands(),
                    trade(9.2, false),
                    None,
                    session,
                )
                .unwrap();
            assert_eq!(d2["chosen"], "b");
            let (_, r3, _) = book
                .reserve_trade_session(
                    5000,
                    Some("s3".into()),
                    cands(),
                    trade(9.2, false),
                    None,
                    None,
                )
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
            assert!(move_of("busy").is_null(), "not a TRADE decision");
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

    /// Run the unknown expiry of every worker with `at` as the sweep time.
    fn expire_at(book: &CostBook, at: Instant) {
        let mut st = book.state.lock();
        book.expire_unknown(&mut st, None, at);
    }

    fn secs(s: u64) -> std::time::Duration {
        std::time::Duration::from_secs(s)
    }

    fn unknown_since(book: &CostBook, r: &Reservation) -> Instant {
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

    fn gwq_args<'a>(
        mode: crate::global_queue::GlobalQueueMode,
        urls: &'a [String],
        depth: &'a [u64],
    ) -> GwqStepArgs<'a> {
        GwqStepArgs {
            mode,
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
            e3: None,
            lead_s: 0.0,
            sources: &[],
        }
    }

    fn gwq_held<'a>(
        rid: &'a str,
        input_tokens: u64,
        candidates: &'a [Candidate],
        p_arrival: Option<&'a str>,
        arrival_pass: bool,
    ) -> GwqHeld<'a> {
        GwqHeld {
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
    fn reply(book: &CostBook, url: &str, at: Instant, rows: &[(&str, u64)]) {
        let sequence = {
            let st = book.state.lock();
            st.progress.get(url).map_or(1, |c| c.sequence + 1)
        };
        book.apply_progress(url, progress(sequence, rows), json!({}), at)
            .unwrap();
        assert_eq!(book.state.lock().progress[url].sent_at, at, "applied");
    }

    /// Regression: a P-leg that the engine never listed gets
    /// a transport error and stays unknown while its P keeps answering. It
    /// holds its work until a reply of its P sent 60 s after it became
    /// unknown, then expires: every counter of its worker returns to its
    /// value before the reservation (another request stays), the RID leaves
    /// the active map but stays used, the TRADE auto end (applied at the
    /// transport error) is not applied again, and the trace has exactly one
    /// `expired` event with the reason and the evidence, also after later
    /// replies and snapshots.
    #[test]
    fn unknown_never_listed_expires_after_the_expiry_and_releases_its_work_once() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("cost.jsonl");
        let expire = Some(secs(60));
        let book =
            CostBook::new_uncached_with_policy(Some(&path), "symmetric_pd_cost_function", expire)
                .unwrap();
        let mut other = uncached(&book, "other", "a", 7_000);
        other.mark_dispatched().unwrap();
        let before = ledger(&book, "a");
        assert_eq!(before, (7_000, 7_000, 0, 7_000, 1));
        // 20,000 tokens with 8,000 cached: 12,000 uncached.
        let (_, mut lost, _) = book
            .reserve_trade_session(
                20_000,
                Some("lost".into()),
                vec![candidate("a", 8_000)],
                auto_trade(18.4),
                None,
                Some("0/agent"),
            )
            .unwrap();
        lost.mark_dispatched().unwrap();
        lost.mark_unknown("prefill_transport_error: upstream timed out")
            .unwrap();
        let held = ledger(&book, "a");
        assert_eq!(held, (19_000, 19_000, 0, 19_000, 2));
        let auto = || {
            let st = book.state.lock();
            let auto = st.trade_auto.as_ref().unwrap();
            (auto.in_flight("a"), auto.credited, auto.uncredited)
        };
        // The transport error ended the prefill (no credit).
        assert_eq!(auto(), (Some(0), 0, 1));
        let since = unknown_since(&book, &lost);
        let just_before = since + secs(60) - std::time::Duration::from_millis(1);
        reply(&book, "a", just_before, &[]);
        assert_eq!(ledger(&book, "a"), held, "held before the expiry");
        assert_eq!(book.snapshot()["unknown_requests"], 1);
        reply(&book, "a", since + secs(60), &[]);
        assert_eq!(ledger(&book, "a"), before);
        {
            let st = book.state.lock();
            assert!(st.unknown.is_empty());
            assert!(!st.active_by_rid.contains_key("lost"));
            assert!(st.used_rids.contains("lost"));
            assert_eq!(st.entries[&lost.id].status, Status::Expired);
        }
        assert_eq!(lost.debuginfo()["reason"], UNKNOWN_EXPIRED_REASON);
        assert_eq!(auto(), (Some(0), 0, 1), "auto end applied once");
        assert!(book
            .reserve_uncached(1, Some("lost".into()), vec![candidate("a", 0)], 12288)
            .is_err());
        // Later replies and snapshots change nothing and write no second event.
        reply(&book, "a", since + secs(600), &[]);
        let snapshot = book.snapshot();
        assert_eq!(snapshot["unknown_requests"], 0);
        assert_eq!(snapshot["unknown_expire_s"], 60.0);
        assert_eq!(snapshot["expired_rid_listed_again"], 0);
        assert_eq!(snapshot["workers"][0]["status_counts"]["expired"], 1);
        assert_eq!(ledger(&book, "a"), before);
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
        assert_eq!(ledger(&book, "a"), (0, 0, 0, 0, 0));
    }

    /// An unknown entry that its P still lists does not expire, and the
    /// listed work updates it. It expires on the first reply of its P sent at
    /// least the expiry after the last reply that listed it; the release
    /// subtracts its current work.
    #[test]
    fn unknown_still_listed_holds_until_the_engine_stops_listing_it() {
        let book = CostBook::create(None, true, None, None, Some(secs(60))).unwrap();
        let mut r = uncached(&book, "slow", "p", 30_000);
        r.mark_dispatched().unwrap();
        r.mark_unknown("prefill_transport_error: upstream timed out")
            .unwrap();
        let since = unknown_since(&book, &r);
        let at = |s: u64| since + secs(s);
        let status = || book.state.lock().entries[&r.id].status;
        reply(&book, "p", at(50), &[("slow", 22_000)]);
        assert_eq!(ledger(&book, "p"), (22_000, 22_000, 0, 0, 1));
        reply(&book, "p", at(100), &[("slow", 9_000)]);
        // The P answers without it from 130 s on: the evidence stays at 100 s.
        reply(&book, "p", at(130), &[]);
        reply(&book, "p", at(159), &[]);
        assert_eq!((status(), cost(&book, "p")), (Status::Unknown, 9_000));
        reply(&book, "p", at(160), &[]);
        assert_eq!(status(), Status::Expired);
        assert_eq!(ledger(&book, "p"), (0, 0, 0, 0, 0));
    }

    /// A P without applied progress replies (dead, partitioned, or its polls
    /// time out) keeps its unknown work at any age: the expiry needs a reply
    /// of the entry's own P, and replies of another P do not count. The first
    /// reply of its P sent at least the expiry after the evidence expires it.
    #[test]
    fn unknown_on_a_p_without_replies_does_not_expire() {
        let book = CostBook::create(None, true, None, None, Some(secs(60))).unwrap();
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
        assert_eq!(ledger(&book, "p"), (0, 0, 0, 0, 0));
    }

    /// The expiry runs when a reply of the entry's own P is applied (not
    /// another P's) and at each snapshot. The snapshot path decides only when
    /// the deciding reply was applied before the entry became unknown, as with
    /// the reply time in the future here.
    #[test]
    fn unknown_expiry_runs_on_its_workers_progress_and_on_snapshots() {
        let book = CostBook::create(None, true, None, None, Some(secs(1))).unwrap();
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
        let book =
            CostBook::new_uncached_with_policy(Some(&path), "symmetric_pd_cost_function", None)
                .unwrap();
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
        assert_eq!(ledger(&book, "a"), (12_000, 12_000, 0, 12_000, 1));
        r.finish("abort_confirmed").unwrap();
        assert_eq!(ledger(&book, "a"), (0, 0, 0, 0, 0));
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
        let book =
            CostBook::new_uncached_with_policy(Some(&path), "symmetric_pd_cost_function", expire)
                .unwrap();
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
        assert_eq!(ledger(&book, "a"), (0, 0, 0, 0, 0));
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
        assert_eq!(ledger(&book, "a"), (0, 0, 0, 0, 0));
        let snapshot = book.snapshot();
        assert_eq!(snapshot["unknown_requests"], 0);
        assert_eq!(snapshot["expired_rid_listed_again"], 2);
        assert_eq!(book.state.lock().expired_by_rid["f"], (f.id, 2));
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

    // ---- OLP: --co-schedule-trade-wait olp / --co-schedule-trade-olp-shadow ----

    /// An uncached book with OLP state.
    fn olp_book(params: OlpParams, shadow: bool) -> Arc<CostBook> {
        let book = CostBook::new_uncached(None).unwrap();
        book.enable_olp(OlpConfig {
            params,
            shadow,
            ..OlpConfig::default()
        });
        book
    }

    /// OLP parameters whose gate is open from the first decision (no
    /// warm-up, no label pairs needed) and whose busy scale follows its own
    /// measurement at once (prior weight ~0).
    fn olp_open(scale_source: OlpScaleSource) -> OlpParams {
        OlpParams {
            scale_source,
            warm_busy_s: 0.0,
            warm_labels: 0,
            min_pairs: 0,
            b0_s: 1e-9,
            ..OlpParams::default()
        }
    }

    /// The engine load of a progress reply: the busy counters and the pool.
    fn olp_load(busy_us: u64, tokens: u64) -> Value {
        json!({
            "total_prefill_busy_us": busy_us,
            "total_prefill_uncached_tokens": tokens,
            "max_total_num_tokens": 1_000_000,
        })
    }

    /// `url` computes 8,000 tokens of a dispatched 10,000-token request in
    /// one counter interval whose engine busy time is `slow` x the modelled
    /// time; the request stays live with 2,000 tokens left.
    fn olp_feed(book: &Arc<CostBook>, url: &str, slow: f64, at: Instant) -> Reservation {
        let rid = format!("{url}-busy");
        let mut r = uncached(book, &rid, url, 10_000);
        r.mark_dispatched().unwrap();
        // The first listing, at the estimate: acknowledged; counters start.
        let first = progress(1, &[(rid.as_str(), 10_000)]);
        book.apply_progress(url, first, olp_load(0, 0), at).unwrap();
        let busy_us = (slow * OlpParams::default().work(8_000, 0, true) * 1e6).round() as u64;
        let second = progress(2, &[(rid.as_str(), 2_000)]);
        let later = at + std::time::Duration::from_millis(1);
        book.apply_progress(url, second, olp_load(busy_us, 8_000), later)
            .unwrap();
        r
    }

    /// A cold 4,000-token request on `a` and `b` after `a` (4x slower than
    /// modelled) and `b` (as modelled) each computed 8,000 tokens and kept
    /// 2,000: equal ledger backlogs. Returns the selection and the decision.
    fn olp_pick(
        book: &Arc<CostBook>,
        params: crate::scheduler::co_schedule::TradeParams,
    ) -> (String, Value) {
        let at = Instant::now();
        let _a = olp_feed(book, "a", 4.0, at);
        let _b = olp_feed(book, "b", 1.0, at);
        let candidates = vec![candidate("a", 0), candidate("b", 0)];
        let (selected, _r, decision) = book
            .reserve_trade(4_000, Some("new".into()), candidates, params)
            .unwrap();
        (selected.worker_url, decision)
    }

    /// OLP on, gate open, busy scale: the slower P (engine busy 4x its
    /// modelled time) loses the tie that the ledger gives it (equal
    /// backlogs; the owner rule picks the smaller URL). Every candidate costs
    /// R x (wait + service) with the OLP terms.
    #[test]
    fn olp_active_moves_a_tie_away_from_the_slower_p() {
        use crate::scheduler::co_schedule::{TradeParams, TradeWait};
        let plain = CostBook::new_uncached(None).unwrap();
        let (ledger, decision) = olp_pick(&plain, TradeParams::default());
        assert_eq!(ledger, "a");
        assert!(decision.get("olp").is_none());
        let olp = TradeParams {
            wait: TradeWait::Olp,
            ..TradeParams::default()
        };
        let (chosen, decision) = olp_pick(&olp_book(olp_open(OlpScaleSource::Busy), false), olp);
        assert_eq!(chosen, "b");
        assert_eq!(decision["olp"]["mode"], "olp");
        assert_eq!(decision["olp"]["active"], true);
        assert!(decision["olp"]["fallback"].is_null());
        assert_eq!(decision["olp"]["olp_chosen"], decision["chosen"]);
        assert_eq!(decision["olp"]["ledger_chosen"], "a");
        assert_eq!(decision["wait"], "olp");
        let rows = decision["candidates"].as_array().unwrap();
        let k: Vec<f64> = rows.iter().map(|r| r["olp_k"].as_f64().unwrap()).collect();
        assert!((k[0] - 4.0).abs() < 1e-3, "{k:?}");
        assert!((k[1] - 1.0).abs() < 1e-3, "{k:?}");
        for row in rows {
            assert_eq!(row["wait_source"], "olp");
            assert_eq!(row["olp_rows"], 1);
            let wait = row["olp_wait_s"].as_f64().unwrap();
            let service = row["olp_service_s"].as_f64().unwrap();
            let cost = row["trade_cost"].as_f64().unwrap();
            assert!((cost - 13_500.0 * (wait + service)).abs() < 1e-6, "{row}");
        }
    }

    /// OLP on but cold (the default warm-up): every candidate takes the
    /// ledger terms, so the choice, the owner and every cost equal the ledger
    /// decision; the trace names the fallback and still has the predictions.
    #[test]
    fn olp_cold_gate_is_the_ledger_decision() {
        use crate::scheduler::co_schedule::{TradeParams, TradeWait};
        let plain = CostBook::new_uncached(None).unwrap();
        let (want, ledger) = olp_pick(&plain, TradeParams::default());
        let olp = TradeParams {
            wait: TradeWait::Olp,
            ..TradeParams::default()
        };
        let params = OlpParams {
            scale_source: OlpScaleSource::Busy,
            b0_s: 1e-9,
            ..OlpParams::default()
        };
        let (chosen, decision) = olp_pick(&olp_book(params, false), olp);
        assert_eq!(chosen, want);
        assert_eq!(decision["olp"]["active"], false);
        assert_eq!(decision["olp"]["fallback"], "cold");
        assert_eq!(decision["olp"]["ledger_chosen"], decision["chosen"]);
        assert_eq!(decision["olp"]["olp_chosen"], "b", "would-be choice");
        for key in ["owner", "chosen", "trade_cost", "pulled_tokens"] {
            assert_eq!(decision[key], ledger[key], "{key}");
        }
        let rows = decision["candidates"].as_array().unwrap();
        for (row, plain_row) in rows.iter().zip(ledger["candidates"].as_array().unwrap()) {
            assert_eq!(row["trade_cost"], plain_row["trade_cost"]);
            assert_eq!(row["wait_source"], "ledger");
            assert!(row["olp_wait_s"].is_number(), "the predictions are traced");
        }
    }

    /// The shadow readout (wait ledger): the decision is the ledger decision
    /// key for key; the trace only gains `olp` and the 11 `olp_*` keys of
    /// every candidate.
    #[test]
    fn olp_shadow_traces_without_changing_the_decision() {
        use crate::scheduler::co_schedule::TradeParams;
        let plain = CostBook::new_uncached(None).unwrap();
        let (want, ledger) = olp_pick(&plain, TradeParams::default());
        let book = olp_book(olp_open(OlpScaleSource::Busy), true);
        let (chosen, mut decision) = olp_pick(&book, TradeParams::default());
        assert_eq!(chosen, want);
        assert_eq!(decision["olp"]["mode"], "shadow");
        assert_eq!(decision["olp"]["active"], false);
        assert!(decision["olp"]["fallback"].is_null(), "gate would open");
        assert_eq!(decision["olp"]["ledger_chosen"], decision["chosen"]);
        assert_eq!(decision["olp"]["olp_chosen"], "b", "would-be choice");
        assert_eq!(strip_olp_keys(&mut decision), 1 + 2 * 11);
        assert_eq!(decision, ledger);
    }

    /// Scale source `label` (the default) end to end: P legs answered with
    /// prefill_response_success become the labels of the next step (an
    /// error answer does not), whose weighted fit is `fit`; that step's
    /// decision is active and prices every candidate with the fit: wait =
    /// a + bq * q_model, service = bo * own_model. The snapshot counts the
    /// gate results.
    #[test]
    fn olp_label_fit_from_finished_p_legs_prices_an_active_decision() {
        use crate::scheduler::co_schedule::{TradeParams, TradeWait};
        let params = OlpParams {
            // A step per millisecond; four labels fit, without the prior.
            step_s: 0.001,
            label_min_rows: 4,
            label_n0: 0.0,
            ..olp_open(OlpScaleSource::Label)
        };
        let book = olp_book(params, false);
        let olp = TradeParams {
            wait: TradeWait::Olp,
            ..TradeParams::default()
        };
        // Five P legs on a, each reserved behind the ones before it (q_model
        // grows) with its own size (own_model differs): a full-rank fit.
        let mut legs: Vec<Reservation> = [1_000, 3_000, 2_000, 5_000, 4_000]
            .iter()
            .enumerate()
            .map(|(i, &tokens)| {
                let candidates = vec![candidate("a", 0)];
                book.reserve_trade(tokens, Some(format!("leg{i}")), candidates, olp)
                    .unwrap()
                    .1
            })
            .collect();
        for leg in &mut legs {
            leg.mark_dispatched().unwrap();
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
        let (labelled, failed) = legs.split_at_mut(4);
        for leg in labelled {
            leg.finish(PREFILL_SUCCESS_REASON).unwrap();
        }
        failed[0].finish("prefill_response_error").unwrap();
        // Fresh progress views of both candidates, then a later step.
        for url in ["a", "b"] {
            book.apply_progress(url, progress(1, &[]), olp_load(0, 0), Instant::now())
                .unwrap();
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
        let candidates = vec![candidate("a", 0), candidate("b", 0)];
        let (_, _r, decision) = book
            .reserve_trade(4_000, Some("new".into()), candidates, olp)
            .unwrap();
        let step = &decision["olp"]["step"];
        assert_eq!(step["label_fit"], "fit", "{step}");
        assert_eq!(step["label_n"], 4, "{step}");
        assert_eq!(decision["olp"]["active"], true);
        assert!(decision["olp"]["fallback"].is_null());
        assert_eq!(decision["olp"]["olp_chosen"], decision["chosen"]);
        let fit = |key: &str| step[key].as_f64().unwrap();
        let (a, bq, bo) = (fit("label_a"), fit("label_bq"), fit("label_bo"));
        assert!((0.0..=0.5).contains(&a), "{step}");
        assert!((0.8..=1.25).contains(&bq), "{step}");
        assert!((0.8..=1.25).contains(&bo), "{step}");
        for row in decision["candidates"].as_array().unwrap() {
            assert_eq!(row["wait_source"], "olp");
            let get = |key: &str| row[key].as_f64().unwrap();
            let wait = a + bq * get("olp_q_model_s");
            assert!((get("olp_wait_s") - wait).abs() < 1e-12, "{row}");
            let service = bo * get("olp_own_model_s");
            assert!((get("olp_service_s") - service).abs() < 1e-12, "{row}");
        }
        // Five stale decisions (no progress view of a yet), then this one.
        let snapshot = book.snapshot();
        let gate = &snapshot["olp"]["gate_total"];
        assert_eq!(gate["decisions"], 6);
        assert_eq!(gate["open"], 1);
        assert_eq!(gate["stale"], 5);
        assert_eq!(snapshot["olp"]["pool_workers_with_counters"], 2);
    }

    /// Labels: only a P-leg answered with prefill_response_success labels its
    /// OLP decision; an error, an unknown end and a rollback drop theirs. An
    /// ended entry leaves the live index; an unknown one stays.
    #[test]
    fn olp_labels_only_successful_p_legs() {
        use crate::scheduler::co_schedule::TradeParams;
        let book = olp_book(OlpParams::default(), true);
        let trade = |rid: &str| {
            let candidates = vec![candidate("a", 0)];
            book.reserve_trade(1_000, Some(rid.into()), candidates, TradeParams::default())
                .unwrap()
                .1
        };
        let mut ok = trade("ok");
        let mut err = trade("err");
        let mut lost = trade("lost");
        let rolled = trade("rolled");
        for r in [&mut ok, &mut err, &mut lost] {
            r.mark_dispatched().unwrap();
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
        ok.finish(PREFILL_SUCCESS_REASON).unwrap();
        err.finish("prefill_response_error").unwrap();
        lost.mark_unknown("prefill_transport_error").unwrap();
        drop(rolled);
        let t_s = book.seconds(Instant::now());
        let state = book.state.lock();
        let olp = state.olp.as_ref().unwrap();
        assert_eq!((olp.labels_total(), olp.pending_len()), (1, 0));
        assert_eq!(olp.live("a").collect::<Vec<_>>(), [lost.id]);
        // A later step reads the label (done in the window before it).
        let stats = olp.stats_at(t_s + 120.0);
        assert_eq!(stats.n_labels, 1);
        assert!(stats.mape_olp.is_some() && stats.mape_cf.is_some());
        drop(state);
    }

    /// The OLP feed of a progress reply: a first listing only corrects the
    /// cache hit (no chunk work); later decreases, and a first listing at 0,
    /// are chunk work at h = input - old, with the restore while the entry
    /// had no update. The engine counters of the same replies pair with it.
    #[test]
    fn olp_feed_skips_the_first_listing_correction() {
        let book = olp_book(OlpParams::default(), true);
        let mut x = uncached(&book, "x", "a", 10_000);
        x.mark_dispatched().unwrap();
        let mut y = uncached(&book, "y", "a", 3_000);
        y.mark_dispatched().unwrap();
        let t0 = Instant::now();
        let at = |ms: u64| t0 + std::time::Duration::from_millis(ms);
        // x is first listed at 7,000 (a cache-hit correction of 3,000).
        let first = progress(1, &[("x", 7_000)]);
        book.apply_progress("a", first, olp_load(0, 0), at(1))
            .unwrap();
        // x: 4,000 computed after 3,000; y first listed at 0 (all computed).
        let second = progress(2, &[("x", 3_000), ("y", 0)]);
        book.apply_progress("a", second, olp_load(1_000_000, 7_000), at(2))
            .unwrap();
        // x: the last 3,000 after 7,000.
        let third = progress(3, &[("x", 0)]);
        book.apply_progress("a", third, olp_load(2_000_000, 10_000), at(3))
            .unwrap();
        let p = OlpParams::default();
        let x_work = p.work(4_000, 3_000, false) + p.work(3_000, 7_000, false);
        let want = x_work + p.work(3_000, 0, true);
        let state = book.state.lock();
        let sums = state.olp.as_ref().unwrap().own_sums("a").unwrap();
        drop(state);
        assert_eq!((sums.tok, sums.model_tok), (10_000.0, 10_000.0));
        assert!((sums.model_s - want).abs() < 1e-9, "{sums:?}");
        assert!((sums.busy_s - 2.0).abs() < 1e-9, "{sums:?}");
    }

    /// Flag off (no OLP state): no decision, row or trace event has an OLP
    /// key, and a TRADE decision with the ledger wait allocates no OLP state.
    #[test]
    fn olp_off_writes_no_olp_key() {
        use crate::scheduler::co_schedule::TradeParams;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("trade.jsonl");
        let book = CostBook::new_uncached(Some(&path)).unwrap();
        let _busy = olp_feed(&book, "a", 4.0, Instant::now());
        let candidates = vec![candidate("a", 0), candidate("b", 0)];
        let (_, mut r, decision) = book
            .reserve_trade(
                4_000,
                Some("new".into()),
                candidates,
                TradeParams::default(),
            )
            .unwrap();
        r.mark_dispatched().unwrap();
        r.finish(PREFILL_SUCCESS_REASON).unwrap();
        assert!(book.snapshot().get("olp").is_none());
        assert!(!decision.to_string().contains("olp"), "{decision}");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"event\":\"snapshot\""), "{text}");
        assert!(!text.contains("olp"), "{text}");
        assert!(book.state.lock().olp.is_none());
    }

    // ---- OLP fast path: incremental queue, compact trace, async trace ----

    /// A cost-function book (unknown expiry 60 s) with OLP; `incremental`
    /// also turns `verify` on (the walk is computed and compared too).
    fn olp_fast_book(
        path: Option<&Path>,
        params: OlpParams,
        shadow: bool,
        incremental: bool,
        compact: bool,
    ) -> Arc<CostBook> {
        let book =
            CostBook::new_uncached_with_policy(path, "symmetric_pd_cost_function", Some(secs(60)))
                .unwrap();
        book.enable_olp(OlpConfig {
            params,
            shadow,
            incremental,
            compact_trace: compact,
            verify: incremental,
            ..OlpConfig::default()
        });
        book
    }

    /// Every worker's incremental sums equal the exact sums of its active
    /// entries (a walk of the ledger), and the OLP queue of both is the same
    /// up to rounding.
    fn assert_sums_equal_walk(book: &CostBook, at: &str) {
        let st = book.state.lock();
        let olp = st.olp.as_ref().unwrap();
        let params = *olp.params();
        let mut want: BTreeMap<String, crate::olp::QueueAgg> = BTreeMap::new();
        for entry in st.entries.values().filter(|e| e.status.active()) {
            want.entry(entry.worker_url.clone())
                .or_default()
                .add(&crate::olp::QueueAgg::of_row(olp_row(entry), params.chunk));
        }
        for url in st.workers.keys() {
            let got = olp.queue_sums(url).expect("incremental");
            let exact = want.get(url).copied().unwrap_or_default();
            assert_eq!(got, exact, "{at}: sums of {url}");
            let rows: Vec<_> = st
                .entries
                .values()
                .filter(|e| e.status.active() && &e.worker_url == url)
                .map(olp_row)
                .collect();
            let (walk, fast) = (params.queue(rows), params.queue_of(&got));
            assert_eq!(
                (walk.rows, walk.launched_tokens),
                (fast.rows, fast.launched_tokens)
            );
            let scale = walk.q_model_s.abs().max(1e-300);
            assert!(
                (walk.q_model_s - fast.q_model_s).abs() <= 1e-12 * scale,
                "{at}: {url} walk {} fast {}",
                walk.q_model_s,
                fast.q_model_s
            );
        }
    }

    /// `--co-schedule-trade-olp-incremental`: the per-P sums follow every
    /// ledger event that changes a row: the reservation (a row not started
    /// pays its restore), the first listing at the estimate (launched: no
    /// restore, its input is launched), a first listing below the estimate,
    /// a decrease, a listing at 0 (costs nothing), the end with success, an
    /// unknown answer (the row stays), a rollback, the unknown expiry and a
    /// P that leaves the candidate set and comes back (its rows stay). After
    /// each event the sums are the exact sums of a ledger walk; the verify
    /// counters show no difference at any decision; at the end no sum is
    /// left.
    #[test]
    fn olp_incremental_sums_follow_every_ledger_event() {
        use crate::scheduler::co_schedule::{TradeParams, TradeWait};
        let olp = TradeParams {
            wait: TradeWait::Olp,
            ..TradeParams::default()
        };
        let book = olp_fast_book(None, olp_open(OlpScaleSource::Label), false, true, false);
        let on = |rid: &str, tokens: u64, url: &str, hit: u64| {
            book.reserve_trade(tokens, Some(rid.into()), vec![candidate(url, hit)], olp)
                .unwrap()
                .1
        };
        let mut r1 = on("r1", 20_000, "a", 8_000);
        assert_sums_equal_walk(&book, "reserve r1");
        let mut r2 = on("r2", 9_000, "a", 0);
        let mut r3 = on("r3", 30_000, "b", 4_000);
        let r4 = on("r4", 5_000, "b", 0);
        assert_sums_equal_walk(&book, "reserve r2-r4");
        for r in [&mut r1, &mut r2, &mut r3] {
            r.mark_dispatched().unwrap();
        }
        assert_sums_equal_walk(&book, "dispatch");
        let t0 = Instant::now();
        let at = |ms: u64| t0 + std::time::Duration::from_millis(ms);
        // r1 at its estimate (launched, no update), r2 below it (update).
        reply(&book, "a", at(1), &[("r1", 12_000), ("r2", 7_000)]);
        assert_sums_equal_walk(&book, "first listing a");
        {
            let st = book.state.lock();
            let sums = st.olp.as_ref().unwrap().queue_sums("a").unwrap();
            assert_eq!(sums.rows, 2);
            assert_eq!(sums.restore, 0, "both launched");
            assert_eq!(sums.launched, 29_000);
        }
        reply(&book, "a", at(2), &[("r1", 3_808), ("r2", 0)]);
        assert_sums_equal_walk(&book, "decrease and zero");
        reply(&book, "b", at(3), &[("r3", 26_000)]);
        assert_sums_equal_walk(&book, "first listing b");
        r2.finish(PREFILL_SUCCESS_REASON).unwrap();
        assert_sums_equal_walk(&book, "finish r2");
        r3.mark_unknown("prefill_transport_error: test").unwrap();
        assert_sums_equal_walk(&book, "unknown r3 stays");
        drop(r4);
        assert_sums_equal_walk(&book, "rollback r4");
        // P a leaves the candidate set, then returns: its row stays.
        let mut r5 = book
            .reserve_trade(4_000, Some("r5".into()), vec![candidate("b", 0)], olp)
            .unwrap()
            .1;
        assert_sums_equal_walk(&book, "decision on b only");
        let candidates = vec![candidate("a", 0), candidate("b", 0)];
        let (_, mut r6, decision) = book
            .reserve_trade(6_000, Some("r6".into()), candidates, olp)
            .unwrap();
        assert_eq!(decision["olp"]["active"], true, "{decision}");
        let rows: Vec<u64> = decision["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["olp_rows"].as_u64().unwrap())
            .collect();
        assert_eq!(rows, [1, 2], "a: r1; b: r3 (unknown) and r5, not r4");
        assert_sums_equal_walk(&book, "a back in the pool");
        // The unknown r3 expires at a reply of b sent 61 s later.
        r5.mark_dispatched().unwrap();
        reply(&book, "b", at(61_000), &[("r5", 4_000)]);
        assert_eq!(r3.debuginfo()["status"], "expired");
        assert_sums_equal_walk(&book, "expiry r3");
        for r in [&mut r1, &mut r5] {
            r.finish(PREFILL_SUCCESS_REASON).unwrap();
        }
        drop(r6.finish("prefill_response_error"));
        assert_sums_equal_walk(&book, "all ended");
        let st = book.state.lock();
        let olp = st.olp.as_ref().unwrap();
        for url in ["a", "b"] {
            assert_eq!(olp.queue_sums(url), Some(crate::olp::QueueAgg::default()));
        }
        let verify = olp.verify_stats();
        assert_eq!(verify.decisions, 6, "{verify:?}");
        assert_eq!(verify.candidates, 7, "{verify:?}");
        assert_eq!(verify.rows_mismatch, 0);
        assert_eq!(verify.launched_mismatch, 0);
        assert_eq!(verify.olp_choice_mismatch, 0);
        assert!(verify.q_max_rel_diff <= 1e-12, "{verify:?}");
        drop(st);
        drop((r1, r2, r3, r5, r6));
    }

    /// OLP enabled after entries exist (a book built in code): the sums
    /// start from the active entries.
    #[test]
    fn olp_incremental_starts_from_the_active_entries() {
        let book = CostBook::new_uncached(None).unwrap();
        let mut a = uncached(&book, "x", "a", 10_000);
        a.mark_dispatched().unwrap();
        let _b = uncached(&book, "y", "a", 3_000);
        let mut done = uncached(&book, "z", "b", 3_000);
        done.mark_dispatched().unwrap();
        done.finish(PREFILL_SUCCESS_REASON).unwrap();
        book.enable_olp(OlpConfig {
            incremental: true,
            ..OlpConfig::default()
        });
        assert_sums_equal_walk(&book, "enable");
        let st = book.state.lock();
        let olp = st.olp.as_ref().unwrap();
        assert_eq!(olp.queue_sums("a").unwrap().rows, 2);
        assert_eq!(olp.queue_sums("b").unwrap().rows, 0);
    }

    /// A small deterministic generator for the scripted A/B below.
    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, n: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) % n.max(1)
        }
    }

    /// Incremental versus walk on the same inputs: two books (OLP decides,
    /// gate open, label prior) take the same 400 scripted operations
    /// (decisions over 2-3 of 3 P with cached prefixes, dispatches, replies
    /// with first listings, decreases and zeros, ends, unknown answers,
    /// rollbacks). Every decision chooses the same P with the same owner,
    /// rows and launched tokens; q_model agrees to 1e-12; the verify counters
    /// of the incremental book show no difference.
    #[test]
    fn olp_incremental_decisions_equal_the_walk() {
        use crate::scheduler::co_schedule::{TradeParams, TradeWait};
        let olp = TradeParams {
            wait: TradeWait::Olp,
            ..TradeParams::default()
        };
        let walk = olp_fast_book(None, olp_open(OlpScaleSource::Label), false, false, false);
        let fast = olp_fast_book(None, olp_open(OlpScaleSource::Label), false, true, false);
        let urls = ["a", "b", "c"];
        let mut rng = Lcg(7);
        // Live requests of both books: (rid, url, input, remaining, dispatched).
        let mut live: Vec<(String, String, u64, u64, bool)> = Vec::new();
        let mut handles: Vec<(Reservation, Reservation)> = Vec::new();
        let t0 = Instant::now();
        let mut sequence = 0u64;
        let (mut decisions, mut active) = (0, 0);
        for step in 0..400u64 {
            // Every P answers each step (fresh views keep the gate open).
            for url in urls {
                sequence += 1;
                let at = t0 + std::time::Duration::from_micros(10 * sequence);
                let mut rows = Vec::new();
                for (rid, u, _, remaining, dispatched) in live.iter_mut() {
                    if u == url && *dispatched && rng.below(3) > 0 {
                        let cut = rng.below(*remaining + 1);
                        *remaining -= cut.min(*remaining) * u64::from(rng.below(2) == 0);
                        rows.push((rid.clone(), *remaining));
                    }
                }
                let rows: Vec<(&str, u64)> = rows.iter().map(|(r, n)| (r.as_str(), *n)).collect();
                for book in [&walk, &fast] {
                    book.apply_progress(url, progress(sequence, &rows), json!({}), at)
                        .unwrap();
                }
            }
            match rng.below(6) {
                0..=2 => {
                    let tokens = 1_000 + rng.below(60_000);
                    let skip = rng.below(4) as usize;
                    let candidates: Vec<Candidate> = urls
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != skip)
                        .map(|(_, u)| candidate(u, rng.below(tokens)))
                        .collect();
                    let rid = format!("r{step}");
                    let (s1, r1, d1) = walk
                        .reserve_trade(tokens, Some(rid.clone()), candidates.clone(), olp)
                        .unwrap();
                    let (s2, r2, d2) = fast
                        .reserve_trade(tokens, Some(rid.clone()), candidates, olp)
                        .unwrap();
                    decisions += 1;
                    active += usize::from(d2["olp"]["active"] == true);
                    assert_eq!(s1.worker_url, s2.worker_url, "step {step}");
                    for key in ["chosen", "owner", "best"] {
                        assert_eq!(d1[key], d2[key], "step {step} {key}");
                    }
                    assert_eq!(d1["olp"]["olp_chosen"], d2["olp"]["olp_chosen"]);
                    let (c1, c2) = (
                        d1["candidates"].as_array().unwrap(),
                        d2["candidates"].as_array().unwrap(),
                    );
                    for (x, y) in c1.iter().zip(c2) {
                        for key in ["olp_rows", "olp_launched_tokens", "olp_own_model_s"] {
                            assert_eq!(x[key], y[key], "step {step} {key}");
                        }
                        let (q1, q2) = (
                            x["olp_q_model_s"].as_f64().unwrap(),
                            y["olp_q_model_s"].as_f64().unwrap(),
                        );
                        assert!(
                            (q1 - q2).abs() <= 1e-12 * q1.abs().max(1e-300),
                            "step {step}"
                        );
                    }
                    let input = tokens;
                    let remaining = input - s2.expected_hit;
                    live.push((rid, s2.worker_url.clone(), input, remaining, false));
                    handles.push((r1, r2));
                }
                3 => {
                    // Dispatch one reserved request.
                    if let Some(i) = live.iter().position(|l| !l.4) {
                        live[i].4 = true;
                        handles[i].0.mark_dispatched().unwrap();
                        handles[i].1.mark_dispatched().unwrap();
                    }
                }
                4 => {
                    // End one request: success, error, unknown or rollback.
                    if !live.is_empty() {
                        let i = rng.below(live.len() as u64) as usize;
                        let (_, _, _, _, dispatched) = live.remove(i);
                        let (mut h1, mut h2) = handles.remove(i);
                        match (dispatched, rng.below(3)) {
                            (true, 0) => {
                                h1.finish(PREFILL_SUCCESS_REASON).unwrap();
                                h2.finish(PREFILL_SUCCESS_REASON).unwrap();
                            }
                            (true, 1) => {
                                h1.finish("prefill_response_error").unwrap();
                                h2.finish("prefill_response_error").unwrap();
                            }
                            (true, _) => {
                                h1.mark_unknown("prefill_transport_error").unwrap();
                                h2.mark_unknown("prefill_transport_error").unwrap();
                            }
                            (false, _) => {}
                        }
                        drop((h1, h2));
                    }
                }
                _ => {}
            }
            assert_sums_equal_walk(&fast, &format!("step {step}"));
        }
        assert!(decisions > 100, "{decisions}");
        assert!(
            active * 2 > decisions,
            "OLP decided {active} of {decisions}"
        );
        let verify = fast.olp_verify_stats().unwrap();
        assert_eq!(verify.decisions, decisions as u64);
        assert_eq!(verify.rows_mismatch + verify.launched_mismatch, 0);
        assert_eq!(verify.olp_choice_mismatch, 0);
        assert!(
            walk.olp_verify_stats().unwrap().decisions == 0,
            "walk book has no verify"
        );
        drop(handles);
    }

    /// `--co-schedule-trade-olp-trace-compact`: the decision and its
    /// selection are those of the full trace; the decision's `olp` keeps
    /// mode, active, fallback, k_pool, labels_total and both argmins (no
    /// step statistics) and each candidate has the five core `olp_*` keys.
    #[test]
    fn olp_compact_trace_keeps_the_decision() {
        use crate::scheduler::co_schedule::{TradeParams, TradeWait};
        let olp = TradeParams {
            wait: TradeWait::Olp,
            ..TradeParams::default()
        };
        let full = olp_book(olp_open(OlpScaleSource::Busy), false);
        let compact = olp_fast_book(None, olp_open(OlpScaleSource::Busy), false, false, true);
        let (want, mut d_full) = olp_pick(&full, olp);
        let (got, mut d_compact) = olp_pick(&compact, olp);
        assert_eq!(got, want);
        let mut keys: Vec<&str> = d_compact["olp"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "active",
                "fallback",
                "k_pool",
                "labels_total",
                "ledger_chosen",
                "mode",
                "olp_chosen"
            ]
        );
        for key in [
            "active",
            "fallback",
            "ledger_chosen",
            "olp_chosen",
            "mode",
            "k_pool",
        ] {
            assert_eq!(d_compact["olp"][key], d_full["olp"][key], "{key}");
        }
        for (row, full_row) in d_compact["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .zip(d_full["candidates"].as_array().unwrap())
        {
            for key in [
                "olp_wait_s",
                "olp_service_s",
                "olp_q_model_s",
                "olp_own_model_s",
                "olp_rows",
            ] {
                assert_eq!(row[key], full_row[key], "{key}");
            }
        }
        assert_eq!(strip_olp_keys(&mut d_compact), 1 + 2 * 5);
        assert_eq!(strip_olp_keys(&mut d_full), 1 + 2 * 11);
        assert_eq!(d_compact, d_full);
        let snapshot = compact.snapshot();
        assert_eq!(snapshot["olp"]["trace"], "compact");
        assert!(snapshot["olp"].get("queue").is_none());
        assert!(full.snapshot()["olp"].get("trace").is_none());
    }

    /// The rows of a trace without the wall-clock and elapsed fields.
    fn timeless_rows(path: &Path) -> Vec<Value> {
        fn strip(v: &mut Value) {
            match v {
                Value::Object(map) => {
                    for key in [
                        "ts_unix_ms",
                        "ts_unix_s",
                        "elapsed_ms",
                        "assigned_unix_ms",
                        "assigned_elapsed_ms",
                        "dispatched_unix_ms",
                        "terminal_unix_ms",
                        "oldest_active_assigned_unix_ms",
                    ] {
                        map.remove(key);
                    }
                    map.values_mut().for_each(strip);
                }
                Value::Array(items) => items.iter_mut().for_each(strip),
                _ => {}
            }
        }
        let mut rows = trace_rows(path);
        rows.iter_mut().for_each(strip);
        rows
    }

    /// Reservations, dispatches, replies, ends and snapshots on `book`.
    fn trace_script(book: &Arc<CostBook>) {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + std::time::Duration::from_millis(ms);
        let trade = trade(4.6, false);
        let mut held = Vec::new();
        for i in 0..40u64 {
            let candidates = vec![
                candidate("a", (i * 997) % 5_000),
                candidate("b", (i * 131) % 3_000),
            ];
            let (_, mut r, _) = book
                .reserve_trade(8_000 + i * 100, Some(format!("q{i}")), candidates, trade)
                .unwrap();
            r.mark_dispatched().unwrap();
            if i % 5 == 0 {
                book.snapshot();
            }
            held.push(r);
        }
        reply(book, "a", at(1), &[("q0", 100), ("q1", 0)]);
        reply(book, "b", at(2), &[("q2", 50)]);
        for (i, r) in held.iter_mut().enumerate() {
            match i % 3 {
                0 => r.finish(PREFILL_SUCCESS_REASON).unwrap(),
                1 => r.mark_unknown("prefill_transport_error").unwrap(),
                _ => {}
            }
        }
        book.snapshot();
    }

    /// `--co-schedule-cost-trace-async`: the same operations write the same
    /// rows in the same order (sequence 1..n, no gap) as the synchronous
    /// trace, apart from the clock fields; `sync_trace` waits for the writer;
    /// dropping the book flushes the rest.
    #[test]
    fn async_trace_writes_the_same_rows_in_order() {
        let tmp = tempfile::tempdir().unwrap();
        let (p_sync, p_async) = (
            tmp.path().join("sync.jsonl"),
            tmp.path().join("async.jsonl"),
        );
        let sync =
            CostBook::new_uncached_with_policy(Some(&p_sync), "symmetric_pd_cost_function", None)
                .unwrap();
        let book =
            CostBook::new_uncached_with_policy(Some(&p_async), "symmetric_pd_cost_function", None)
                .unwrap();
        book.enable_async_trace().unwrap();
        book.enable_async_trace().unwrap();
        trace_script(&sync);
        trace_script(&book);
        book.sync_trace().unwrap();
        let rows = timeless_rows(&p_async);
        assert_eq!(rows, timeless_rows(&p_sync));
        let sequences: Vec<u64> = rows
            .iter()
            .map(|r| r["sequence"].as_u64().unwrap())
            .collect();
        assert_eq!(sequences, (1..=rows.len() as u64).collect::<Vec<_>>());
        // A row emitted after the last sync is written when the book drops.
        let before = trace_rows(&p_async).len();
        book.snapshot();
        drop(book);
        drop(sync);
        assert_eq!(trace_rows(&p_async).len(), before + 1);
    }

    /// An async trace that cannot be written: the writer keeps the error,
    /// `sync_trace` fails, the next admission is refused and the snapshot
    /// shows admission blocked.
    #[cfg(target_os = "linux")]
    #[test]
    fn async_trace_failure_stops_admission() {
        let book = CostBook::new_uncached(None).unwrap();
        book.state.lock().trace = Some(TraceSink::Sync(BufWriter::new(
            OpenOptions::new().write(true).open("/dev/full").unwrap(),
        )));
        book.enable_async_trace().unwrap();
        let _first = uncached(&book, "first", "a", 1_000);
        assert!(book.sync_trace().is_err());
        let refused =
            book.reserve_uncached(1_000, Some("next".into()), vec![candidate("a", 0)], 12288);
        assert!(refused.unwrap_err().contains("trace failed"));
        assert_eq!(book.snapshot()["admission_blocked"], true);
    }

    /// `a` has 30k routed and not launched (queued 30k: no room at depth
    /// 9542); `b` runs a launched 40k request (queued 40k - one pass of 8192
    /// = 31,808: no room). A cold 5k request costs 35k on `a` and 45k on `b`:
    /// it waits. When `b` has 9k left (queued 808: room), `b` costs 14k: the
    /// request goes to `b` (late binding), with the TRADE record.
    /// `--co-schedule-global-queue-room predicted`. a has 50,000 and b 6,000
    /// tokens routed and not launched: no room by tokens at depth 1,000, so
    /// the token rule holds a cold request. With the E3 probes, a is predicted
    /// to drain in more than 15 s (five 65,536-token rows in its snapshot plus
    /// its router row) and b within the lead of 1 s (only its router row): the
    /// request goes to b (release `predicted`). The walk and the reservation
    /// price each P as min(R x drain, ledger backlog) (the forecast may make a
    /// P look emptier, never fuller). A P without a usable fit keeps the token
    /// rule (fail-open): with b's fit missing, nothing is released.
    #[test]
    fn gwq_step_predicted_room_takes_the_e3_drain_and_falls_back_per_p() {
        use crate::global_queue::{GlobalQueueMode, GwqRelease};
        let book = CostBook::new_uncached(None).unwrap();
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
        let held = [gwq_held("r0", 5000, &cands, None, false)];
        let step = book
            .gwq_step(
                trade(18.4, false),
                &held,
                &gwq_args(GlobalQueueMode::Hrrn, &urls, &depth),
            )
            .unwrap();
        assert!(step.release.is_none(), "no room by tokens: {step:?}");

        let zero = std::time::Duration::ZERO;
        let ctx = e3_context(
            &[
                ("a", Some(routing_snapshot(5, 65_536)), zero, true, 0, 0),
                ("b", Some(routing_snapshot(0, 0)), zero, true, 0, 0),
            ],
            GWQ_PROBE_TOKENS as i64,
        );
        let args = GwqStepArgs {
            e3: Some(&ctx),
            lead_s: 1.0,
            ..gwq_args(GlobalQueueMode::Hrrn, &urls, &depth)
        };
        let held = [gwq_held("r1", 5000, &cands, None, false)];
        let step = book.gwq_step(trade(18.4, false), &held, &args).unwrap();
        let r1 = step
            .release
            .expect("b is predicted to drain within the lead")
            .1
            .unwrap();
        assert_eq!(r1.candidate.worker_url, "b");
        assert_eq!(r1.release, GwqRelease::Predicted);
        let g = &r1.record;
        assert_eq!(g["room_rule"], "predicted");
        assert_eq!(g["lead_s"], 1.0);
        let drains = g["drain_s_by_p"].as_array().unwrap();
        assert!(drains[0].as_f64().unwrap() > 15.0, "{g}");
        let d_b = drains[1].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&d_b), "{g}");
        assert_eq!(g["drain_s"], drains[1]);
        assert_eq!(r1.decision["chosen"], "b");
        assert_eq!(r1.decision.get("walk_mismatch"), None);
        let row_b = &r1.decision["candidates"][1];
        assert_eq!(row_b["gwq_drain_s"], drains[1]);
        assert_eq!(
            row_b["backlog_tokens"],
            gwq_drain_tokens(d_b, 13_500.0).min(6_000).to_string()
        );
        assert_eq!(r1.decision["candidates"][0]["backlog_tokens"], "50000");
        drop(r1);

        let ctx = e3_context(
            &[
                ("a", Some(routing_snapshot(5, 65_536)), zero, true, 0, 0),
                ("b", Some(routing_snapshot(0, 0)), zero, false, 0, 0),
            ],
            GWQ_PROBE_TOKENS as i64,
        );
        let args = GwqStepArgs {
            e3: Some(&ctx),
            lead_s: 1.0,
            ..gwq_args(GlobalQueueMode::Hrrn, &urls, &depth)
        };
        let held = [gwq_held("r2", 5000, &cands, None, false)];
        let step = book.gwq_step(trade(18.4, false), &held, &args).unwrap();
        assert!(step.release.is_none(), "{step:?}");
        drop((on_a, on_b));
    }

    #[test]
    fn gwq_step_holds_without_room_and_binds_the_p_at_release() {
        use crate::global_queue::{GlobalQueueMode, GwqRelease};
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("gwq.jsonl");
        let book = CostBook::new_uncached(Some(&path)).unwrap();
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
        let held = [gwq_held("r", 5000, &cands, None, true)];
        let step = book
            .gwq_step(
                trade(9.2, false),
                &held,
                &gwq_args(GlobalQueueMode::Hrrn, &urls, &depth),
            )
            .unwrap();
        assert!(step.release.is_none(), "expected a hold: {step:?}");
        assert_eq!(step.plain, vec![0]);
        assert_eq!(step.queue_tokens, 5000);
        assert_eq!(
            book.state.lock().entries.len(),
            2,
            "no reservation for a hold"
        );
        let rows = book.gwq_rows(&urls, Instant::now(), 8192);
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
        let held = [gwq_held("r", 5000, &cands, Some("a"), false)];
        let step = book
            .gwq_step(
                trade(9.2, false),
                &held,
                &gwq_args(GlobalQueueMode::Hrrn, &urls, &depth),
            )
            .unwrap();
        let Some((index, result)) = step.release else {
            panic!("expected a release");
        };
        assert_eq!(index, 0);
        let released = result.unwrap();
        assert_eq!(released.candidate.worker_url, "b");
        assert_eq!(released.release, GwqRelease::Room);
        assert!(released.late_bound && !released.starved);
        assert_eq!(released.decision["chosen"], "b");
        let events = trace_lines(&path);
        let reserve = events
            .iter()
            .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == "r")
            .unwrap();
        assert_eq!(reserve["selection_policy"], "trade");
        assert_eq!(reserve["trade_decision"]["chosen"], "b");
        let g = &reserve["gwq"];
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
        let rows = reserve["trade_decision"]["candidates"].as_array().unwrap();
        assert_eq!(rows[0]["gwq_room"], false);
        assert_eq!(rows[1]["gwq_room"], true);
        assert_eq!(rows[1]["backlog_tokens"], "9000");
        drop((on_a, on_b, released));
    }

    /// sjf, two equal cold requests: the first prefers `a` (no room) and adds
    /// its 8k to the virtual backlog of `a`; the second then prefers `b`
    /// (room). The decision rows show the virtual tokens in the backlog.
    #[test]
    fn gwq_step_charges_the_held_work_ahead_as_virtual_backlog() {
        use crate::global_queue::GlobalQueueMode;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("gwq-virtual.jsonl");
        let book = CostBook::new_uncached(Some(&path)).unwrap();
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
            gwq_held("first", 8000, &cands, Some("a"), false),
            gwq_held("second", 8000, &cands, Some("a"), false),
        ];
        let step = book
            .gwq_step(
                trade(9.2, false),
                &held,
                &gwq_args(GlobalQueueMode::Sjf, &urls, &depth),
            )
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
        let rows = reserve["trade_decision"]["candidates"].as_array().unwrap();
        assert_eq!(rows[0]["gwq_virtual_tokens"], "8000");
        assert_eq!(rows[0]["backlog_tokens"], "18000");
        assert_eq!(rows[0]["trade_cost"], 26_000.0);
        assert_eq!(rows[1]["gwq_virtual_tokens"], "0");
        assert_eq!(rows[1]["trade_cost"], 20_000.0);
        drop((on_a, on_b, released));
    }

    /// A reservation in the unknown state (transport error) is not known to
    /// be in an engine queue: the queued work leaves it out, the TRADE
    /// backlog keeps it.
    #[test]
    fn gwq_queued_work_leaves_out_unknown_reservations() {
        use crate::global_queue::GlobalQueueMode;
        let book = CostBook::new_uncached(None).unwrap();
        let mut lost = uncached(&book, "lost", "a", 30_000);
        lost.mark_dispatched().unwrap();
        lost.mark_unknown("prefill_transport_error: reset").unwrap();
        book.apply_progress("a", progress(1, &[]), json!({}), Instant::now())
            .unwrap();
        let urls = vec!["a".to_string()];
        let rows = book.gwq_rows(&urls, Instant::now(), 8192);
        assert_eq!(rows[0].queued, 0);
        assert!(rows[0].view_age_ms.is_some());
        assert_eq!(unacknowledged(&book, "a"), 30_000);
        let cands = vec![candidate("a", 0)];
        let held = [gwq_held("r", 1000, &cands, None, true)];
        let step = book
            .gwq_step(
                trade(9.2, false),
                &held,
                &gwq_args(GlobalQueueMode::Sjf, &urls, &[9542]),
            )
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
        let book = CostBook::new_uncached(None).unwrap();
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
        let book = CostBook::new_uncached(None).unwrap();
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
        assert!(CostBook::new(None).unwrap().rid_refusal(None).is_none());
    }
}
