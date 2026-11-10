// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Online latency predictor (OLP) of the symmetric_pd cost function:
//! `--co-schedule-trade-wait olp`, or the trace-only readout
//! `--co-schedule-trade-olp-shadow`. Default off: without either flag the
//! cost book has no OLP state, and its decisions and trace do not change.
//!
//! For each P candidate of a TRADE decision OLP predicts, in seconds, the
//! request's wait (until its first prefill pass) and its own service. The
//! cost book passes them to the TRADE cost as
//! [`crate::scheduler::co_schedule::TradeWaitTerms`]; the pull exposure, the
//! copy price and R stay those of the cost function.
//!
//! **Work** (structural per-pass prior, amortized over the passes of a row):
//!
//! ```text
//! work(u, h, first) = c0 * ceil(u / chunk) + c1 * u + c2 * (u * h + u * (u + 1) / 2)
//!                   + (first ? c3 * rr * h : 0)
//! ```
//!
//! u = tokens still to compute, h = tokens before them (u = 0 costs 0). The
//! queue work q_model of a P sums `work` over the router's live ledger rows
//! on it (a row whose first chunk has not launched still pays its restore);
//! the request's own work is own_model = work(u, n - u, true).
//!
//! **Scale** (`scale_source`). `label` (default): wait = a + bq * q_model and
//! service = bo * own_model, (a, bq, bo) refit once per step by weighted
//! least squares on the completion labels of the label window, clamped and
//! shrunk toward (0, 1, 1) ([`label_fit`]). `busy`: both terms times the
//! per-P busy scale k ([`OlpState::busy_scale`]), learned from the engine's
//! cumulative prefill busy counters over the router-attributed chunk work,
//! with a pooled prior, shrinkage, a decayed history across P->D->P and a
//! reset at an engine restart. `none`: the prior as is. The busy scale is
//! fed and traced with every source.
//!
//! **Labels**: a P-leg answered with `prefill_response_success` labels its
//! decision with y = answer - decision in seconds; every other end is
//! censored. Once per step the statistics of the labels done in the window
//! before the step are computed: the APE of OLP and of the cost function's
//! own latency proxy, their pair concordance (decisions close in time on
//! different P, the order that routing uses), the park-gate residual and the
//! label fit.
//!
//! **Gate** (mode `olp`, per decision): OLP prices every candidate only when
//! it is warm, every candidate's progress view is fresh, the step has enough
//! label pairs and OLP's concordance is not below the cost function's
//! (reasons in this order: `cold`, `stale`, `no_pairs`, `not_better`).
//! Otherwise every candidate takes the ledger terms: the cost function's
//! decision, bit for bit. Warm needs the engine busy counters with every
//! scale source (pooled busy seconds of at least `warm_busy_s`), so the gate
//! results ([`GateCounts`]: since the start and per step) and the warm-up
//! inputs are traced in `olp_step` and in the one-second `snapshot` row.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::scheduler::co_schedule::TradeWaitTerms;

/// At most this many OLP decisions wait for their label (the oldest go first).
pub const PENDING_LABELS_MAX: usize = 200_000;
/// A busy-counter interval longer than this (s) is not sampled.
pub const POLL_GAP_MAX_S: f64 = 2.0;
/// Labels are kept for the label window plus this many seconds.
const LABEL_KEEP_EXTRA_S: f64 = 120.0;
/// Two labels form a pair only when they differ by more than this (s).
const PAIR_MIN_LABEL_GAP_S: f64 = 0.05;
/// A Gauss-Jordan pivot of at most this share of the largest normal-matrix
/// entry makes the label fit singular.
const PIVOT_EPS: f64 = 1e-12;

/// What scales the structural work into seconds (`scale_source`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum OlpScaleSource {
    /// `label` (default): wait = a + bq * q_model, service = bo * own_model,
    /// with (a, bq, bo) of the step's label fit ([`label_fit`]).
    #[default]
    #[serde(rename = "label")]
    Label,
    /// `busy`: wait = k * q_model, service = k * own_model, with the P's busy
    /// scale k ([`OlpState::busy_scale`]).
    #[serde(rename = "busy")]
    Busy,
    /// `none`: wait = q_model, service = own_model (the prior as is).
    #[serde(rename = "none")]
    Prior,
}

impl OlpScaleSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Label => "label",
            Self::Busy => "busy",
            Self::Prior => "none",
        }
    }
}

/// The OLP parameters. Defaults: the pooled warmup medians of the GLM P
/// model, the gate chosen offline and the label scale of the offline replay.
/// `--co-schedule-trade-olp-params` overrides them by name ([`load_params`]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OlpParams {
    /// Seconds per prefill pass.
    pub c0: f64,
    /// Seconds per computed token.
    pub c1: f64,
    /// Seconds per attention pair.
    pub c2: f64,
    /// Seconds per restored token.
    pub c3: f64,
    /// Restored share of a cached prefix, in [0, 1].
    pub rr: f64,
    /// Chunked-prefill size: tokens of one row per pass.
    pub chunk: u64,
    /// What scales the work into seconds (default `label`).
    pub scale_source: OlpScaleSource,
    /// Busy scale: the own window (s).
    pub win_k_s: f64,
    /// Busy scale: the pooled window (s).
    pub win_pool_s: f64,
    /// Busy scale: the prior's weight, in busy seconds.
    pub b0_s: f64,
    /// Busy scale: decay time (s) of a P's history after it left the pool.
    pub tau_hist_s: f64,
    /// Busy scale: lower clamp.
    pub k_min: f64,
    /// Busy scale: upper clamp.
    pub k_max: f64,
    /// Label scale: the prior's weight, in label rows.
    pub label_n0: f64,
    /// Label scale: fewer rows (without a pull) keep the prior (0, 1, 1).
    pub label_min_rows: u64,
    /// Label scale: the intercept a is clamped to [0, a_max] seconds.
    pub a_max: f64,
    /// Label scale: lower clamp of bq and bo.
    pub b_min: f64,
    /// Label scale: upper clamp of bq and bo.
    pub b_max: f64,
    /// Gate: every candidate's last applied progress poll at most this old (s).
    pub fresh_s: f64,
    /// Gate: pooled busy seconds in the pooled window needed to be warm.
    pub warm_busy_s: f64,
    /// Gate: labels seen in total needed to be warm.
    pub warm_labels: u64,
    /// The label window before each step (s).
    pub label_window_s: f64,
    /// Pairs: decisions at most this far apart (s).
    pub pair_window_s: f64,
    /// Gate: label pairs needed in the current step.
    pub min_pairs: u64,
    /// Gate: OLP's pair concordance may be this much below the cost function's.
    pub conc_slack: f64,
    /// Step (s) of the label statistics, the label fit and the `olp_step` rows.
    pub step_s: f64,
    /// Park-gate risk: add the residual of the risky labels to a risky P's wait.
    pub risk_residual: bool,
    /// Park-gate risk: the cap is this share of the P's `max_total_num_tokens`.
    pub cap_frac: f64,
    /// Park-gate risk: risky labels needed for a residual.
    pub gate_min_labels: u64,
}

impl Default for OlpParams {
    fn default() -> Self {
        Self {
            c0: 0.01995,
            c1: 3.445e-5,
            c2: 1.539e-10,
            c3: 6.285e-7,
            rr: 0.45,
            chunk: 8192,
            scale_source: OlpScaleSource::Label,
            win_k_s: 120.0,
            win_pool_s: 300.0,
            b0_s: 5.0,
            tau_hist_s: 1800.0,
            k_min: 0.25,
            k_max: 8.0,
            label_n0: 500.0,
            label_min_rows: 50,
            a_max: 0.5,
            b_min: 0.8,
            b_max: 1.25,
            fresh_s: 2.0,
            warm_busy_s: 30.0,
            warm_labels: 50,
            label_window_s: 300.0,
            pair_window_s: 0.2,
            min_pairs: 200,
            conc_slack: 0.01,
            step_s: 60.0,
            risk_residual: false,
            cap_frac: 0.35,
            gate_min_labels: 20,
        }
    }
}

impl OlpParams {
    /// Every value finite; above 0 where a zero breaks the model, a window or
    /// a clamp; `rr` in [0, 1], `chunk` at least 1, ordered clamps.
    pub fn validate(&self) -> Result<(), String> {
        let at_least_zero = [
            ("c0", self.c0),
            ("c2", self.c2),
            ("c3", self.c3),
            ("label_n0", self.label_n0),
            ("a_max", self.a_max),
            ("warm_busy_s", self.warm_busy_s),
            ("conc_slack", self.conc_slack),
        ];
        for (name, value) in at_least_zero {
            if !(value.is_finite() && value >= 0.0) {
                return Err(format!("{name} must be finite and at least 0"));
            }
        }
        let positive = [
            ("c1", self.c1),
            ("win_k_s", self.win_k_s),
            ("win_pool_s", self.win_pool_s),
            ("b0_s", self.b0_s),
            ("tau_hist_s", self.tau_hist_s),
            ("k_min", self.k_min),
            ("k_max", self.k_max),
            ("b_min", self.b_min),
            ("b_max", self.b_max),
            ("fresh_s", self.fresh_s),
            ("label_window_s", self.label_window_s),
            ("pair_window_s", self.pair_window_s),
            ("step_s", self.step_s),
            ("cap_frac", self.cap_frac),
        ];
        for (name, value) in positive {
            if !(value.is_finite() && value > 0.0) {
                return Err(format!("{name} must be finite and greater than 0"));
            }
        }
        if !(0.0..=1.0).contains(&self.rr) {
            return Err("rr must be in [0, 1]".into());
        }
        if self.chunk == 0 {
            return Err("chunk must be at least 1".into());
        }
        if self.k_min > self.k_max {
            return Err("k_min must not be above k_max".into());
        }
        if self.b_min > self.b_max {
            return Err("b_min must not be above b_max".into());
        }
        Ok(())
    }

    /// The structural work (s) of one row: `u` tokens still to compute after
    /// `h` tokens; `first`: the row's first chunk is still ahead, so its
    /// cached prefix h is restored (share `rr`). u = 0 costs 0.
    pub fn work(&self, u: u64, h: u64, first: bool) -> f64 {
        if u == 0 {
            return 0.0;
        }
        let (u_f, h_f) = (u as f64, h as f64);
        let passes = u.div_ceil(self.chunk.max(1)) as f64;
        let pairs = u_f * h_f + u_f * (u_f + 1.0) / 2.0;
        let restore = if first { self.c3 * self.rr * h_f } else { 0.0 };
        self.c0 * passes + self.c1 * u_f + self.c2 * pairs + restore
    }

    /// The queue of one P: the work of each live row (h = input - remaining;
    /// a row that has not started still pays its restore) and the input
    /// tokens of the started rows (they hold P KV until their export).
    pub fn queue(&self, rows: impl IntoIterator<Item = OlpRow>) -> OlpQueue {
        let mut queue = OlpQueue::default();
        for row in rows {
            let done = row.input.saturating_sub(row.remaining);
            queue.q_model_s += self.work(row.remaining, done, !row.started);
            queue.rows += 1;
            if row.started {
                queue.launched_tokens = queue.launched_tokens.saturating_add(row.input);
            }
        }
        queue
    }

    /// The queue of one P from its exact integer sums
    /// (`--co-schedule-trade-olp-incremental`): the same model as
    /// [`Self::queue`] over the same rows, with each coefficient applied once
    /// to the sum of its term:
    ///
    /// ```text
    /// q_model = c0 * Σ passes + c1 * Σ u + c2 * Σ pairs + c3 * rr * Σ h (rows not started, u > 0)
    /// ```
    ///
    /// The rows and launched tokens are equal to [`Self::queue`]'s; q_model
    /// differs from the row-by-row sum only by floating-point rounding (a
    /// few ulp per row).
    pub fn queue_of(&self, agg: &QueueAgg) -> OlpQueue {
        OlpQueue {
            q_model_s: self.c0 * agg.passes as f64
                + self.c1 * agg.tokens as f64
                + self.c2 * agg.pairs as f64
                + self.c3 * self.rr * agg.restore as f64,
            rows: usize::try_from(agg.rows).unwrap_or(usize::MAX),
            launched_tokens: u64::try_from(agg.launched).unwrap_or(u64::MAX),
        }
    }
}

/// The exact integer sums of a P's live rows (or of one row) for the
/// incremental queue ([`OlpParams::queue_of`]). The terms of one row are
/// those of [`OlpParams::work`] with u = remaining and h = input - remaining:
/// passes = ceil(u / chunk), pairs = u * h + u * (u + 1) / 2, restore = h
/// when the row has not started and u > 0 (u = 0 costs nothing); launched =
/// input when the row has started. Adding and removing rows is exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QueueAgg {
    pub rows: u64,
    pub passes: u128,
    pub tokens: u128,
    pub pairs: u128,
    pub restore: u128,
    pub launched: u128,
}

impl QueueAgg {
    /// The sums of one row with chunk size `chunk`.
    pub fn of_row(row: OlpRow, chunk: u64) -> Self {
        let u = u128::from(row.remaining);
        let h = u128::from(row.input.saturating_sub(row.remaining));
        Self {
            rows: 1,
            passes: u128::from(row.remaining.div_ceil(chunk.max(1))),
            tokens: u,
            pairs: u * h + u * (u + 1) / 2,
            restore: if row.started || u == 0 { 0 } else { h },
            launched: if row.started {
                u128::from(row.input)
            } else {
                0
            },
        }
    }

    pub(crate) fn add(&mut self, other: &Self) {
        self.rows += other.rows;
        self.passes += other.passes;
        self.tokens += other.tokens;
        self.pairs += other.pairs;
        self.restore += other.restore;
        self.launched += other.launched;
    }

    /// Remove a row added before (exact; saturating so that a bookkeeping
    /// bug cannot panic the router; the tests check exactness).
    fn sub(&mut self, other: &Self) {
        self.rows = self.rows.saturating_sub(other.rows);
        self.passes = self.passes.saturating_sub(other.passes);
        self.tokens = self.tokens.saturating_sub(other.tokens);
        self.pairs = self.pairs.saturating_sub(other.pairs);
        self.restore = self.restore.saturating_sub(other.restore);
        self.launched = self.launched.saturating_sub(other.launched);
    }
}

/// The incremental queue's self-check (`OlpConfig::verify`, tests and the
/// offline replay only): at every decision both queues of every candidate
/// (the row walk and the integer sums) and both OLP argmins.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct VerifyStats {
    pub decisions: u64,
    pub candidates: u64,
    /// Candidates whose two q_model values are bitwise equal.
    pub q_bitwise_equal: u64,
    pub q_max_abs_diff_s: f64,
    pub q_max_rel_diff: f64,
    pub rows_mismatch: u64,
    pub launched_mismatch: u64,
    /// Decisions whose OLP argmin (wait olp over the OLP terms) differs
    /// between the walk and the integer sums.
    pub olp_choice_mismatch: u64,
}

impl VerifyStats {
    /// Compare one candidate's two queues.
    pub fn candidate(&mut self, walk: &OlpQueue, fast: &OlpQueue) {
        self.candidates += 1;
        let diff = (walk.q_model_s - fast.q_model_s).abs();
        if walk.q_model_s.to_bits() == fast.q_model_s.to_bits() {
            self.q_bitwise_equal += 1;
        }
        self.q_max_abs_diff_s = self.q_max_abs_diff_s.max(diff);
        let scale = walk.q_model_s.abs().max(fast.q_model_s.abs());
        if scale > 0.0 {
            self.q_max_rel_diff = self.q_max_rel_diff.max(diff / scale);
        }
        self.rows_mismatch += u64::from(walk.rows != fast.rows);
        self.launched_mismatch += u64::from(walk.launched_tokens != fast.launched_tokens);
    }

    pub fn value(&self) -> Value {
        json!({
            "decisions": self.decisions,
            "candidates": self.candidates,
            "q_bitwise_equal": self.q_bitwise_equal,
            "q_max_abs_diff_s": self.q_max_abs_diff_s,
            "q_max_rel_diff": self.q_max_rel_diff,
            "rows_mismatch": self.rows_mismatch,
            "launched_mismatch": self.launched_mismatch,
            "olp_choice_mismatch": self.olp_choice_mismatch,
        })
    }
}

/// The parameters of a `--co-schedule-trade-olp-params` file: a JSON object
/// whose keys override [`OlpParams`] fields by name. Unknown keys, wrong
/// types and invalid values are refused.
pub fn params_from_json(data: &[u8]) -> Result<OlpParams, String> {
    let value: Value = serde_json::from_slice(data).map_err(|e| e.to_string())?;
    if !value.is_object() {
        return Err("the OLP parameters must be a JSON object".into());
    }
    let params: OlpParams = serde_json::from_value(value).map_err(|e| e.to_string())?;
    params.validate()?;
    Ok(params)
}

/// Read `--co-schedule-trade-olp-params`: the validated parameters
/// ([`params_from_json`]) and the sha256 (hex) of the file's bytes.
pub fn load_params(path: &Path) -> Result<(OlpParams, String), String> {
    let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let params = params_from_json(&data).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((params, hex_sha256(&data)))
}

fn hex_sha256(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The OLP configuration of a cost function (`SymmetricPdCostFunctionConfig::olp`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OlpConfig {
    /// The defaults or the parameter file's values.
    pub params: OlpParams,
    /// `--co-schedule-trade-olp-shadow`: fed and traced, never used.
    pub shadow: bool,
    /// The sha256 of the parameter file; None with the defaults.
    pub params_sha256: Option<String>,
    /// `--co-schedule-trade-olp-incremental`: each P's queue from exact
    /// integer sums kept at every row change (commit, progress, end) instead
    /// of a walk of its live rows at each decision ([`OlpParams::queue_of`]).
    pub incremental: bool,
    /// `--co-schedule-trade-olp-trace-compact`: fewer OLP keys per decision
    /// (no `step` object and 5 instead of 11 `olp_*` keys per candidate; the
    /// step statistics stay in the `olp_step` rows and the snapshot).
    pub compact_trace: bool,
    /// Tests and the offline replay only (no flag): with `incremental`,
    /// also walk the rows at each decision and count the differences
    /// ([`VerifyStats`], in the snapshot's `olp.verify`).
    pub verify: bool,
}

/// How a decision uses OLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OlpMode {
    /// `wait = olp`: the OLP terms price every candidate when the gate is open.
    Olp,
    /// The shadow readout (`wait = ledger`): traced only.
    Shadow,
}

impl OlpMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Olp => "olp",
            Self::Shadow => "shadow",
        }
    }
}

/// Why a decision of mode `olp` takes the cost function's terms (the gate
/// reasons, checked in this order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OlpFallback {
    /// Pooled busy seconds in the pooled window below `warm_busy_s`, or fewer
    /// than `warm_labels` labels seen.
    Cold,
    /// A candidate without an applied progress poll or engine epoch, or whose
    /// last applied poll is older than `fresh_s`.
    Stale,
    /// Fewer than `min_pairs` label pairs in the current step.
    NoPairs,
    /// OLP's pair concordance below the cost function's minus `conc_slack`.
    NotBetter,
}

impl OlpFallback {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Stale => "stale",
            Self::NoPairs => "no_pairs",
            Self::NotBetter => "not_better",
        }
    }
}

/// Decisions by their gate result: open (mode `olp`: OLP priced the
/// decision; shadow: the gate would have opened) or the fallback reason.
/// Traced in `olp_step` and in the `olp` object of the one-second
/// `snapshot` row, so a live check sees whether OLP decides at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GateCounts {
    pub decisions: u64,
    pub open: u64,
    pub cold: u64,
    pub stale: u64,
    pub no_pairs: u64,
    pub not_better: u64,
}

impl GateCounts {
    /// Count one decision whose gate gave `fallback` (None = open).
    pub fn count(&mut self, fallback: Option<OlpFallback>) {
        self.decisions += 1;
        let slot = match fallback {
            None => &mut self.open,
            Some(OlpFallback::Cold) => &mut self.cold,
            Some(OlpFallback::Stale) => &mut self.stale,
            Some(OlpFallback::NoPairs) => &mut self.no_pairs,
            Some(OlpFallback::NotBetter) => &mut self.not_better,
        };
        *slot += 1;
    }

    /// open / decisions; None without decisions.
    pub fn open_share(&self) -> Option<f64> {
        if self.decisions == 0 {
            None
        } else {
            Some(self.open as f64 / self.decisions as f64)
        }
    }

    /// The trace object: the counts and the open share.
    pub fn value(&self) -> Value {
        json!({
            "decisions": self.decisions,
            "open": self.open,
            "open_share": self.open_share(),
            "cold": self.cold,
            "stale": self.stale,
            "no_pairs": self.no_pairs,
            "not_better": self.not_better,
        })
    }
}

/// One live ledger row of a P (reserved, dispatched or unknown).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OlpRow {
    /// Tokens still to compute (the engine-updated uncached tokens).
    pub remaining: u64,
    /// The request's input tokens.
    pub input: u64,
    /// The engine listed it: its first chunk has launched (restore done).
    pub started: bool,
}

/// The queue of one P ([`OlpParams::queue`]).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct OlpQueue {
    /// Σ work of the live rows (s, unscaled).
    pub q_model_s: f64,
    pub rows: usize,
    /// Σ input tokens of the started rows.
    pub launched_tokens: u64,
}

/// Sums of busy-counter samples: the engine's busy seconds and tokens and the
/// modelled seconds and tokens of the router-attributed chunk work.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Sums {
    pub busy_s: f64,
    pub tok: f64,
    pub model_s: f64,
    pub model_tok: f64,
}

impl Sums {
    fn add(&mut self, d: Sums) {
        self.busy_s += d.busy_s;
        self.tok += d.tok;
        self.model_s += d.model_s;
        self.model_tok += d.model_tok;
    }

    fn sub(&mut self, d: Sums) {
        self.busy_s -= d.busy_s;
        self.tok -= d.tok;
        self.model_s -= d.model_s;
        self.model_tok -= d.model_tok;
    }

    /// Every sum at least 0 (running sums can drift below by rounding).
    fn clamped(self) -> Sums {
        Sums {
            busy_s: self.busy_s.max(0.0),
            tok: self.tok.max(0.0),
            model_s: self.model_s.max(0.0),
            model_tok: self.model_tok.max(0.0),
        }
    }

    /// The modelled seconds of the engine-counted tokens:
    /// (model_s / model_tok) * tok. None without tokens or modelled work.
    fn modelled_s(&self) -> Option<f64> {
        if self.tok > 0.0 && self.model_s > 0.0 && self.model_tok > 0.0 {
            Some(self.model_s / self.model_tok * self.tok)
        } else {
            None
        }
    }

    /// k_obs = busy_s / modelled seconds: measured seconds per token over
    /// modelled seconds per token of the same mix. None without data.
    pub fn ratio(&self) -> Option<f64> {
        let k = self.busy_s / self.modelled_s()?;
        k.is_finite().then_some(k)
    }
}

/// A sliding window of busy-counter samples with running sums: O(1)
/// amortized push and eviction.
#[derive(Debug, Default)]
struct Window {
    samples: VecDeque<(f64, Sums)>,
    sums: Sums,
}

impl Window {
    fn push(&mut self, t_s: f64, d: Sums) {
        self.sums.add(d);
        self.samples.push_back((t_s, d));
    }

    /// Drop the samples older than `span_s` at `now_s`; an empty window has
    /// exact zero sums.
    fn evict(&mut self, now_s: f64, span_s: f64) {
        while let Some(&(t_s, d)) = self.samples.front() {
            if t_s >= now_s - span_s {
                break;
            }
            self.samples.pop_front();
            self.sums.sub(d);
        }
        if self.samples.is_empty() {
            self.sums = Sums::default();
        }
    }

    fn sums(&self) -> Sums {
        self.sums.clamped()
    }
}

/// The OLP state of one worker (keyed by its URL; kept across role changes).
#[derive(Debug, Default)]
struct WorkerOlp {
    epoch: Option<String>,
    /// (poll time s, busy us, tokens) of the last counter reading.
    last_counter: Option<(f64, u64, u64)>,
    /// Samples of the own window (`win_k_s`).
    own: Window,
    /// Samples of the pooled window (`win_pool_s`).
    pool: Window,
    /// Router-attributed chunk work since the last counter reading.
    pending: Sums,
    /// A candidate of the last decision (a P-role worker).
    in_pool: bool,
    /// (k, time it left the pool) of its last P stint.
    k_hist: Option<(f64, f64)>,
    /// Send time (s) of its last applied progress poll.
    last_poll_t_s: Option<f64>,
    /// The engine's `max_total_num_tokens` (latest reported).
    max_total_tokens: Option<u64>,
}

impl WorkerOlp {
    /// An engine restart: counters, windows and history start over.
    fn reset(&mut self) {
        let in_pool = self.in_pool;
        *self = Self {
            in_pool,
            ..Self::default()
        };
    }

    /// The age of its last busy-counter reading at `t_s`; None without one.
    fn counter_age_s(&self, t_s: f64) -> Option<f64> {
        self.last_counter.map(|(at, _, _)| (t_s - at).max(0.0))
    }
}

/// What OLP keeps of one decision until its label.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingLabel {
    /// Decision time (s since the cost book started).
    pub t_decision: f64,
    /// The chosen P.
    pub url: String,
    /// OLP's predicted completion of the chosen P under the active scale:
    /// wait + max(0, pull - wait) + service.
    pub pred_olp_c: f64,
    /// The same without the park-gate residual (the residual's base).
    pub pred_olp_base: f64,
    /// The cost function's latency proxy of the chosen P: (backlog + u) / R
    /// plus the exposed pull time.
    pub pred_cf_c: f64,
    /// The chosen P was at park-gate risk.
    pub risk: bool,
    /// The raw queue work of the chosen P (s, unscaled).
    pub q_model_s: f64,
    /// The raw own work on the chosen P (s, unscaled).
    pub own_model_s: f64,
    /// The chosen P pulls a prefix (such rows stay out of the label fit).
    pub pulled: bool,
}

/// A label: a decision whose P-leg was answered with success.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelRow {
    /// Answer time (s since the cost book started).
    pub t_done: f64,
    /// t_done - t_decision (s), above 0.
    pub y: f64,
    pub decision: PendingLabel,
}

/// The label scale of one step ([`label_fit`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabelFit {
    pub a: f64,
    pub bq: f64,
    pub bo: f64,
    /// Label rows without a pull in the window.
    pub n: usize,
    /// `fit`, or the prior (0, 1, 1): `few_rows`, `singular`.
    pub status: &'static str,
}

impl LabelFit {
    fn prior(n: usize, status: &'static str) -> Self {
        Self {
            a: 0.0,
            bq: 1.0,
            bo: 1.0,
            n,
            status,
        }
    }
}

/// The label statistics of one step, over the labels done in
/// [start_s - label_window_s, start_s).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepStats {
    /// floor(t / step_s).
    pub index: i64,
    /// The start of the step (s since the cost book started).
    pub start_s: f64,
    pub n_labels: usize,
    pub pairs: usize,
    /// Share of the pairs whose OLP order is the label order.
    pub conc_olp: Option<f64>,
    /// The same for the cost function's latency proxy.
    pub conc_cf: Option<f64>,
    pub mape_olp: Option<f64>,
    pub mape_cf: Option<f64>,
    pub p50_olp: Option<f64>,
    pub p50_cf: Option<f64>,
    pub risk_labels: usize,
    /// Median of y - OLP (without the residual) over the risky labels, when
    /// at least `gate_min_labels`; else 0.
    pub gate_residual_s: f64,
    pub fit: LabelFit,
}

impl StepStats {
    /// The `step` object of the trace. `start_s` is the step's start time
    /// (the parameter `step_s` is its length).
    pub fn value(&self) -> Value {
        json!({
            "start_s": self.start_s,
            "n_labels": self.n_labels,
            "pairs": self.pairs,
            "conc_olp": self.conc_olp,
            "conc_cf": self.conc_cf,
            "mape_olp": self.mape_olp,
            "mape_cf": self.mape_cf,
            "p50_olp": self.p50_olp,
            "p50_cf": self.p50_cf,
            "risk_labels": self.risk_labels,
            "gate_residual_s": self.gate_residual_s,
            "label_a": self.fit.a,
            "label_bq": self.fit.bq,
            "label_bo": self.fit.bo,
            "label_n": self.fit.n,
            "label_fit": self.fit.status,
        })
    }
}

/// The decision-level inputs of one OLP decision ([`OlpState::begin`]).
#[derive(Debug, Clone, Copy)]
pub struct OlpContext {
    /// Decision time (s since the cost book started).
    pub t_s: f64,
    /// The pooled busy scale (1 without data).
    pub k_pool: f64,
    /// Busy seconds of every worker in the pooled window.
    pub pool_busy_s: f64,
    /// The statistics of the current step.
    pub step: StepStats,
    /// This decision is the first of its step (its `olp_step` row is due).
    pub new_step: bool,
}

/// OLP's terms of one candidate and how they were made.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct OlpPrediction {
    /// B_w: the wait (s), with the park-gate residual when it applies.
    pub wait_s: f64,
    /// The wait without the residual.
    pub wait_base_s: f64,
    /// S_w: the request's own service (s).
    pub service_s: f64,
    /// The P's busy scale (traced with every scale source).
    pub k: f64,
    /// The P's busy seconds in the own window.
    pub busy_s: f64,
    /// Unscaled work of the P's live rows (s).
    pub q_model_s: f64,
    /// Unscaled own work of the request on the P (s).
    pub own_model_s: f64,
    /// The P's live rows.
    pub rows: usize,
    /// Launched input plus the request's input is above the cap.
    pub risk: bool,
    pub launched_tokens: u64,
    /// cap_frac * max_total_num_tokens; None without an engine report.
    pub cap_tokens: Option<f64>,
    /// Age of the P's last applied progress poll; None without a poll or
    /// engine epoch.
    pub poll_age_s: Option<f64>,
}

/// wait + max(0, pull - wait) + service: the pull runs while the request
/// waits, so only its exposed part delays it.
fn completion(wait_s: f64, service_s: f64, pull_s: f64) -> f64 {
    wait_s + (pull_s - wait_s).max(0.0) + service_s
}

impl OlpPrediction {
    /// The TRADE terms (B_w, S_w).
    pub fn terms(&self) -> TradeWaitTerms {
        TradeWaitTerms {
            wait_s: self.wait_s,
            service_s: self.service_s,
        }
    }

    /// The predicted completion with a pull of `pull_s` seconds.
    pub fn completion_s(&self, pull_s: f64) -> f64 {
        completion(self.wait_s, self.service_s, pull_s)
    }

    /// [`Self::completion_s`] without the park-gate residual.
    pub fn completion_base_s(&self, pull_s: f64) -> f64 {
        completion(self.wait_base_s, self.service_s, pull_s)
    }

    /// The `olp_*` keys of a candidate row; `compact`
    /// (`--co-schedule-trade-olp-trace-compact`): only the terms, the two
    /// raw works and the rows.
    pub fn trace_keys(&self, compact: bool) -> Value {
        if compact {
            return json!({
                "olp_wait_s": self.wait_s,
                "olp_service_s": self.service_s,
                "olp_q_model_s": self.q_model_s,
                "olp_own_model_s": self.own_model_s,
                "olp_rows": self.rows,
            });
        }
        json!({
            "olp_wait_s": self.wait_s,
            "olp_service_s": self.service_s,
            "olp_k": self.k,
            "olp_busy_s": self.busy_s,
            "olp_q_model_s": self.q_model_s,
            "olp_own_model_s": self.own_model_s,
            "olp_rows": self.rows,
            "olp_risk": self.risk,
            "olp_launched_tokens": self.launched_tokens,
            "olp_cap_tokens": self.cap_tokens,
            "olp_poll_age_s": self.poll_age_s,
        })
    }
}

/// The OLP state of one cost book (see the module docs). All times are
/// seconds since the cost book started (monotonic).
#[derive(Debug)]
pub struct OlpState {
    config: OlpConfig,
    /// Per worker URL, created at its first poll or decision.
    workers: BTreeMap<String, WorkerOlp>,
    /// Per worker URL, the reservation IDs of its live entries (reserved,
    /// dispatched or unknown): the queue rows without a scan of the ledger.
    live: HashMap<String, BTreeSet<u64>>,
    /// `incremental`: per worker URL, the integer sums of its live rows.
    agg: HashMap<String, QueueAgg>,
    /// `incremental`: per live reservation ID, its worker URL and the row it
    /// added to that worker's sums (removed exactly at the next change).
    rows: HashMap<u64, (String, OlpRow)>,
    /// `verify`: the walk-versus-sums comparison so far.
    verify: VerifyStats,
    /// Reservation ID -> its OLP decision, until the entry ends.
    pending: BTreeMap<u64, PendingLabel>,
    /// Labels by answer time, kept for the label window plus 120 s.
    labels: VecDeque<LabelRow>,
    labels_total: u64,
    /// The statistics of the latest step.
    step: Option<StepStats>,
    /// Gate results since the start and of the latest step.
    gate_total: GateCounts,
    gate_step: GateCounts,
    /// (start s, gate results) of the last earlier step with decisions.
    gate_last_step: Option<(f64, GateCounts)>,
    /// (t_s, k_pool, pool busy s) of the last decision: what its gate saw.
    last_pool: Option<(f64, f64, f64)>,
}

impl OlpState {
    pub fn new(config: OlpConfig) -> Self {
        Self {
            config,
            workers: BTreeMap::new(),
            live: HashMap::new(),
            agg: HashMap::new(),
            rows: HashMap::new(),
            verify: VerifyStats::default(),
            pending: BTreeMap::new(),
            labels: VecDeque::new(),
            labels_total: 0,
            step: None,
            gate_total: GateCounts::default(),
            gate_step: GateCounts::default(),
            gate_last_step: None,
            last_pool: None,
        }
    }

    pub fn config(&self) -> &OlpConfig {
        &self.config
    }

    pub fn params(&self) -> &OlpParams {
        &self.config.params
    }

    /// Labels seen since the start.
    pub fn labels_total(&self) -> u64 {
        self.labels_total
    }

    /// Decisions that wait for their label.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Gate results since the start.
    pub fn gate_total(&self) -> GateCounts {
        self.gate_total
    }

    /// A new live entry `id` on `url` with its current `row` (with
    /// `incremental`, the row enters the worker's sums).
    pub fn live_insert(&mut self, url: &str, id: u64, row: OlpRow) {
        self.live.entry(url.to_string()).or_default().insert(id);
        if self.config.incremental {
            // A second insert of the same ID replaces its row.
            self.agg_remove(id);
            let chunk = self.config.params.chunk;
            self.agg
                .entry(url.to_string())
                .or_default()
                .add(&QueueAgg::of_row(row, chunk));
            self.rows.insert(id, (url.to_string(), row));
        }
    }

    /// The live entry `id` changed (remaining work or first chunk launched):
    /// with `incremental`, its old row leaves its worker's sums and `row`
    /// enters them. An ID that is not live is ignored.
    pub fn live_update(&mut self, id: u64, row: OlpRow) {
        if !self.config.incremental {
            return;
        }
        let chunk = self.config.params.chunk;
        let Some((url, old)) = self.rows.get_mut(&id) else {
            return;
        };
        if *old == row {
            return;
        }
        let sums = self.agg.entry(url.clone()).or_default();
        sums.sub(&QueueAgg::of_row(*old, chunk));
        sums.add(&QueueAgg::of_row(row, chunk));
        *old = row;
    }

    /// Entry `id` on `url` ended (finished, rolled back or expired).
    pub fn live_remove(&mut self, url: &str, id: u64) {
        if let Some(ids) = self.live.get_mut(url) {
            ids.remove(&id);
            if ids.is_empty() {
                self.live.remove(url);
            }
        }
        self.agg_remove(id);
    }

    /// `incremental`: remove the row of `id` from its worker's sums (a
    /// worker without rows leaves the map).
    fn agg_remove(&mut self, id: u64) {
        let Some((url, old)) = self.rows.remove(&id) else {
            return;
        };
        let chunk = self.config.params.chunk;
        if let Some(sums) = self.agg.get_mut(&url) {
            sums.sub(&QueueAgg::of_row(old, chunk));
            if sums.rows == 0 {
                self.agg.remove(&url);
            }
        }
    }

    /// `incremental`: the integer sums of `url`'s live rows (zero without
    /// rows); None without `incremental`.
    pub fn queue_sums(&self, url: &str) -> Option<QueueAgg> {
        self.config
            .incremental
            .then(|| self.agg.get(url).copied().unwrap_or_default())
    }

    /// `verify`: record one decision's comparison (see [`VerifyStats`]).
    pub fn verify_stats_mut(&mut self) -> &mut VerifyStats {
        &mut self.verify
    }

    /// `verify`: the comparison so far.
    pub fn verify_stats(&self) -> VerifyStats {
        self.verify
    }

    /// The live entries on `url`, by reservation ID.
    pub fn live(&self, url: &str) -> impl Iterator<Item = u64> + '_ {
        self.live.get(url).into_iter().flatten().copied()
    }

    /// One applied progress reply of `url`, sent at `t_s`: the engine epoch,
    /// the engine load (`total_prefill_busy_us`,
    /// `total_prefill_uncached_tokens`, `max_total_num_tokens`) and the
    /// router-attributed chunk work of the reply (modelled seconds, tokens).
    ///
    /// A new epoch resets the worker. The chunk work adds to the pending
    /// sums; a counter reading closes the interval since the last one: when
    /// both counters did not decrease and the interval is at most 2 s, the
    /// sample (busy seconds, tokens, pending work) enters both windows if it
    /// has tokens or pending work. The pending sums then start over.
    pub fn observe_poll(
        &mut self,
        url: &str,
        t_s: f64,
        epoch: &str,
        load: &Value,
        chunk_model_s: f64,
        chunk_tokens: u64,
    ) {
        let params = self.config.params;
        let worker = self.workers.entry(url.to_string()).or_default();
        if worker.epoch.as_deref() != Some(epoch) {
            if worker.epoch.is_some() {
                worker.reset();
            }
            worker.epoch = Some(epoch.to_string());
        }
        worker.last_poll_t_s = Some(t_s);
        if let Some(total) = load.get("max_total_num_tokens").and_then(Value::as_u64) {
            worker.max_total_tokens = Some(total);
        }
        worker.pending.model_s += chunk_model_s;
        worker.pending.model_tok += chunk_tokens as f64;
        let busy = load.get("total_prefill_busy_us").and_then(Value::as_u64);
        let tokens = load
            .get("total_prefill_uncached_tokens")
            .and_then(Value::as_u64);
        let (Some(busy_us), Some(tok)) = (busy, tokens) else {
            // No counters: nothing to pair the chunk work with.
            worker.pending = Sums::default();
            return;
        };
        if let Some((t0, busy0, tok0)) = worker.last_counter {
            let gap = t_s - t0;
            if busy_us >= busy0 && tok >= tok0 && (0.0..=POLL_GAP_MAX_S).contains(&gap) {
                let dtok = tok - tok0;
                if dtok > 0 || worker.pending.model_tok > 0.0 {
                    let sample = Sums {
                        busy_s: (busy_us - busy0) as f64 / 1e6,
                        tok: dtok as f64,
                        model_s: worker.pending.model_s,
                        model_tok: worker.pending.model_tok,
                    };
                    worker.own.push(t_s, sample);
                    worker.pool.push(t_s, sample);
                }
            }
        }
        worker.pending = Sums::default();
        worker.last_counter = Some((t_s, busy_us, tok));
        worker.own.evict(t_s, params.win_k_s);
        worker.pool.evict(t_s, params.win_pool_s);
    }

    /// k_pool = Σ busy / Σ modelled seconds over every worker's pooled window
    /// (1 without data), and the busy seconds of those windows.
    pub fn pooled(&mut self, t_s: f64) -> (f64, f64) {
        let span = self.config.params.win_pool_s;
        let (mut busy, mut num, mut den) = (0.0, 0.0, 0.0);
        for worker in self.workers.values_mut() {
            worker.pool.evict(t_s, span);
            let sums = worker.pool.sums();
            busy += sums.busy_s;
            if let Some(modelled) = sums.modelled_s() {
                num += sums.busy_s;
                den += modelled;
            }
        }
        let k_pool = num / den;
        let k_pool = if den > 0.0 && k_pool.is_finite() {
            k_pool
        } else {
            1.0
        };
        (k_pool, busy)
    }

    /// The busy scale of `url` at `t_s` and its busy seconds in the own window:
    ///
    /// ```text
    /// prior = k_pool, or w * k_hist + (1 - w) * k_pool, w = exp(-(t - t_left) / tau_hist_s)
    /// k     = (B * k_obs + b0_s * prior) / (B + b0_s)    (own data; else k = prior)
    /// ```
    ///
    /// B = own busy seconds, k_obs = [`Sums::ratio`] of the own window; k is
    /// clamped to [k_min, k_max].
    pub fn busy_scale(&mut self, url: &str, t_s: f64, k_pool: f64) -> (f64, f64) {
        let params = self.config.params;
        let (own, k_hist) = match self.workers.get_mut(url) {
            Some(worker) => {
                worker.own.evict(t_s, params.win_k_s);
                (worker.own.sums(), worker.k_hist)
            }
            None => (Sums::default(), None),
        };
        let prior = match k_hist {
            Some((k, left_s)) => {
                let age = (t_s - left_s).max(0.0);
                let w = (-age / params.tau_hist_s).exp();
                w * k + (1.0 - w) * k_pool
            }
            None => k_pool,
        };
        let k = match own.ratio() {
            Some(k_obs) => {
                let b = own.busy_s;
                (b * k_obs + params.b0_s * prior) / (b + params.b0_s)
            }
            None => prior,
        };
        (k.max(params.k_min).min(params.k_max), own.busy_s)
    }

    /// The P pool of a decision: a worker that was in it and is not among
    /// `urls` keeps its busy scale as history (P->D); every URL of `urls` is
    /// in it.
    pub fn sync_pool(&mut self, urls: &[&str], t_s: f64, k_pool: f64) {
        let leaving: Vec<String> = self
            .workers
            .iter()
            .filter(|(url, worker)| worker.in_pool && !urls.contains(&url.as_str()))
            .map(|(url, _)| url.clone())
            .collect();
        for url in leaving {
            let (k, _) = self.busy_scale(&url, t_s, k_pool);
            if let Some(worker) = self.workers.get_mut(&url) {
                worker.k_hist = Some((k, t_s));
                worker.in_pool = false;
            }
        }
        for url in urls {
            self.workers.entry((*url).to_string()).or_default().in_pool = true;
        }
    }

    /// The decision-level inputs at `t_s` for the candidates `urls`: the
    /// pooled scale, the pool sync and the statistics of the current step
    /// (computed at its first decision).
    pub fn begin(&mut self, urls: &[&str], t_s: f64) -> OlpContext {
        let (k_pool, pool_busy_s) = self.pooled(t_s);
        self.last_pool = Some((t_s, k_pool, pool_busy_s));
        self.sync_pool(urls, t_s, k_pool);
        let (step, new_step) = self.refresh_step(t_s);
        OlpContext {
            t_s,
            k_pool,
            pool_busy_s,
            step,
            new_step,
        }
    }

    /// The terms of one candidate `url` with `queue` for a request of
    /// `input_tokens` with `uncached_tokens` to compute on it.
    pub fn predict(
        &mut self,
        ctx: &OlpContext,
        url: &str,
        queue: OlpQueue,
        input_tokens: u64,
        uncached_tokens: u64,
    ) -> OlpPrediction {
        let params = self.config.params;
        let cached = input_tokens.saturating_sub(uncached_tokens);
        let own_model_s = params.work(uncached_tokens, cached, true);
        let (k, busy_s) = self.busy_scale(url, ctx.t_s, ctx.k_pool);
        let fit = ctx.step.fit;
        let (wait_base_s, service_s) = match params.scale_source {
            OlpScaleSource::Label => (fit.a + fit.bq * queue.q_model_s, fit.bo * own_model_s),
            OlpScaleSource::Busy => (k * queue.q_model_s, k * own_model_s),
            OlpScaleSource::Prior => (queue.q_model_s, own_model_s),
        };
        let worker = self.workers.get(url);
        let cap_tokens = worker
            .and_then(|w| w.max_total_tokens)
            .map(|total| params.cap_frac * total as f64);
        let launched = queue.launched_tokens.saturating_add(input_tokens) as f64;
        let risk = cap_tokens.is_some_and(|cap| launched > cap);
        let residual = if params.risk_residual && risk {
            ctx.step.gate_residual_s.max(0.0)
        } else {
            0.0
        };
        let poll_age_s = worker
            .filter(|w| w.epoch.is_some())
            .and_then(|w| w.last_poll_t_s)
            .map(|at| (ctx.t_s - at).max(0.0));
        OlpPrediction {
            wait_s: wait_base_s + residual,
            wait_base_s,
            service_s,
            k,
            busy_s,
            q_model_s: queue.q_model_s,
            own_model_s,
            rows: queue.rows,
            risk,
            launched_tokens: queue.launched_tokens,
            cap_tokens,
            poll_age_s,
        }
    }

    /// The confidence gate of a decision: the first reason that applies, in
    /// the order `cold`, `stale`, `no_pairs`, `not_better`; None = open.
    pub fn gate(&self, ctx: &OlpContext, preds: &[OlpPrediction]) -> Option<OlpFallback> {
        let params = &self.config.params;
        if ctx.pool_busy_s < params.warm_busy_s || self.labels_total < params.warm_labels {
            return Some(OlpFallback::Cold);
        }
        if preds
            .iter()
            .any(|pred| pred.poll_age_s.is_none_or(|age| age > params.fresh_s))
        {
            return Some(OlpFallback::Stale);
        }
        if (ctx.step.pairs as u64) < params.min_pairs {
            return Some(OlpFallback::NoPairs);
        }
        match (ctx.step.conc_olp, ctx.step.conc_cf) {
            (Some(olp), Some(cf)) if olp < cf - params.conc_slack => Some(OlpFallback::NotBetter),
            _ => None,
        }
    }

    /// Count the gate result of one decision ([`GateCounts`]): since the
    /// start and in the step of the last [`Self::begin`].
    pub fn count_gate(&mut self, fallback: Option<OlpFallback>) {
        self.gate_total.count(fallback);
        self.gate_step.count(fallback);
    }

    /// Keep the OLP decision of reservation `id` until its entry ends.
    pub fn record(&mut self, id: u64, label: PendingLabel) {
        self.pending.insert(id, label);
        while self.pending.len() > PENDING_LABELS_MAX {
            self.pending.pop_first();
        }
    }

    /// The entry of reservation `id` ended at `t_s`. A success (the P-leg
    /// answered with `prefill_response_success`) with y > 0 becomes a label;
    /// any other end drops the decision. Returns whether a label was added.
    pub fn resolve(&mut self, id: u64, t_s: f64, success: bool) -> bool {
        let Some(decision) = self.pending.remove(&id) else {
            return false;
        };
        let y = t_s - decision.t_decision;
        if !(success && y.is_finite() && y > 0.0) {
            return false;
        }
        self.labels.push_back(LabelRow {
            t_done: t_s,
            y,
            decision,
        });
        self.labels_total += 1;
        let keep_from = t_s - (self.config.params.label_window_s + LABEL_KEEP_EXTRA_S);
        while self.labels.front().is_some_and(|l| l.t_done < keep_from) {
            self.labels.pop_front();
        }
        true
    }

    fn step_index(&self, t_s: f64) -> i64 {
        (t_s / self.config.params.step_s).floor() as i64
    }

    /// The statistics of step `index`.
    fn stats_for(&self, index: i64) -> StepStats {
        let params = &self.config.params;
        let start_s = index as f64 * params.step_s;
        let from = start_s - params.label_window_s;
        let rows: Vec<&LabelRow> = self
            .labels
            .iter()
            .filter(|l| (from..start_s).contains(&l.t_done))
            .collect();
        step_stats(&rows, params, index, start_s)
    }

    /// The statistics of the step that contains `t_s` (not cached).
    pub fn stats_at(&self, t_s: f64) -> StepStats {
        self.stats_for(self.step_index(t_s))
    }

    /// The statistics of the step of `t_s`, computed once per step; true at
    /// the first call of a step, which also closes the gate results of the
    /// step before.
    fn refresh_step(&mut self, t_s: f64) -> (StepStats, bool) {
        let index = self.step_index(t_s);
        if let Some(step) = self.step.filter(|s| s.index == index) {
            return (step, false);
        }
        let step = self.stats_for(index);
        if let Some(previous) = self.step.replace(step) {
            if self.gate_step.decisions > 0 {
                self.gate_last_step = Some((previous.start_s, self.gate_step));
            }
        }
        self.gate_step = GateCounts::default();
        (step, true)
    }

    /// `trade_decision.olp` of one decision.
    pub fn decision_value(
        &self,
        mode: OlpMode,
        active: bool,
        fallback: Option<OlpFallback>,
        ctx: &OlpContext,
    ) -> Value {
        let params = &self.config.params;
        if self.config.compact_trace {
            // The step statistics are in the step's `olp_step` row and in the
            // snapshot; the parameters in the startup log and `olp_step`.
            return json!({
                "mode": mode.as_str(),
                "active": active,
                "fallback": fallback.map(OlpFallback::as_str),
                "k_pool": ctx.k_pool,
                "labels_total": self.labels_total,
            });
        }
        json!({
            "mode": mode.as_str(),
            "active": active,
            "fallback": fallback.map(OlpFallback::as_str),
            "scale_source": params.scale_source.as_str(),
            "risk_residual": params.risk_residual,
            "k_pool": ctx.k_pool,
            "pool_busy_s": ctx.pool_busy_s,
            "labels_total": self.labels_total,
            "step": ctx.step.value(),
            "params_sha256": self.config.params_sha256,
        })
    }

    /// The `olp_step` event of the step of `ctx`: its statistics, the gate
    /// results of the last earlier step with decisions (`gate_last_step`,
    /// with its `start_s`) and since the start (`gate_total`, this decision
    /// included) and, per worker known to OLP, its pool state, busy scale,
    /// history and the age of its last busy-counter reading.
    pub fn step_event(&mut self, ctx: &OlpContext) -> Value {
        let urls: Vec<String> = self.workers.keys().cloned().collect();
        let mut workers = Vec::with_capacity(urls.len());
        for url in urls {
            let (k, busy_s) = self.busy_scale(&url, ctx.t_s, ctx.k_pool);
            let Some(worker) = self.workers.get(&url) else {
                continue;
            };
            let k_hist = worker
                .k_hist
                .map(|(k_hist, left_s)| json!({"k": k_hist, "left_s": left_s}));
            workers.push(json!({
                "url": url,
                "in_pool": worker.in_pool,
                "k": k,
                "busy_s": busy_s,
                "k_hist": k_hist,
                "epoch": worker.epoch,
                "poll_age_s": worker.last_poll_t_s.map(|at| (ctx.t_s - at).max(0.0)),
                "counter_age_s": worker.counter_age_s(ctx.t_s),
            }));
        }
        let gate_last_step = self.gate_last_step.map(|(start_s, counts)| {
            let mut value = counts.value();
            value["start_s"] = json!(start_s);
            value
        });
        json!({
            "event": "olp_step",
            "step": ctx.step.value(),
            "k_pool": ctx.k_pool,
            "pool_busy_s": ctx.pool_busy_s,
            "gate_last_step": gate_last_step,
            "gate_total": self.gate_total.value(),
            "labels_total": self.labels_total,
            "labels_kept": self.labels.len(),
            "pending_labels": self.pending.len(),
            "workers": workers,
            "params": self.config.params,
            "params_sha256": self.config.params_sha256,
        })
    }

    /// The `olp` object of the cost book's one-second `snapshot` row, at
    /// `t_s` (read only). The gate stays `cold` until the pooled engine busy
    /// seconds reach `warm_busy_s` and `warm_labels` labels were seen, with
    /// every scale source: without engine busy counters every decision is
    /// the cost function's. So the row shows the warm-up inputs of the last
    /// decision (`pool_busy_s`, `labels_total`), the P workers whose last
    /// busy-counter reading is at most `fresh_s` old, and the gate results
    /// of the current step (`gate_step`) and since the start (`gate_total`).
    pub fn snapshot_value(&self, t_s: f64) -> Value {
        let params = &self.config.params;
        let fresh_s = params.fresh_s;
        let pool: Vec<&WorkerOlp> = self.workers.values().filter(|w| w.in_pool).collect();
        let counted = pool
            .iter()
            .filter(|w| w.counter_age_s(t_s).is_some_and(|age| age <= fresh_s))
            .count();
        let (decision_s, k_pool, pool_busy_s) = match self.last_pool {
            Some((at, k_pool, busy_s)) => (Some(at), Some(k_pool), Some(busy_s)),
            None => (None, None, None),
        };
        let mut value = json!({
            "mode": if self.config.shadow { "shadow" } else { "olp" },
            "t_s": t_s,
            "last_decision_s": decision_s,
            "k_pool": k_pool,
            "pool_busy_s": pool_busy_s,
            "warm_busy_s": params.warm_busy_s,
            "labels_total": self.labels_total,
            "warm_labels": params.warm_labels,
            "pending_labels": self.pending.len(),
            "pool_workers": pool.len(),
            "pool_workers_with_counters": counted,
            "step": self.step.as_ref().map(StepStats::value),
            "gate_step": self.gate_step.value(),
            "gate_total": self.gate_total.value(),
        });
        // Only with the new options, so the row is unchanged without them.
        if self.config.incremental {
            value["queue"] = json!("incremental");
            value["queue_workers"] = json!(self.agg.len());
            value["queue_rows"] = json!(self.rows.len());
        }
        if self.config.compact_trace {
            value["trace"] = json!("compact");
        }
        if self.config.verify {
            value["verify"] = self.verify.value();
        }
        value
    }

    /// The own-window sums of `url` (tests).
    #[cfg(test)]
    pub(crate) fn own_sums(&self, url: &str) -> Option<Sums> {
        self.workers.get(url).map(|w| w.own.sums())
    }
}

/// The statistics of one step over `rows` (the labels done in the window
/// before the step): APE of OLP and of the cost function's proxy (mean and
/// median), pair concordance, the park-gate residual and the label fit.
fn step_stats(rows: &[&LabelRow], params: &OlpParams, index: i64, start_s: f64) -> StepStats {
    let mut ape_olp: Vec<f64> = rows
        .iter()
        .map(|l| ape(l.decision.pred_olp_c, l.y))
        .collect();
    let mut ape_cf: Vec<f64> = rows
        .iter()
        .map(|l| ape(l.decision.pred_cf_c, l.y))
        .collect();
    let (pairs, agree_olp, agree_cf) = concordance(rows, params.pair_window_s);
    let share = |agree: usize| {
        if pairs > 0 {
            Some(agree as f64 / pairs as f64)
        } else {
            None
        }
    };
    let mut residuals: Vec<f64> = rows
        .iter()
        .filter(|l| l.decision.risk)
        .map(|l| l.y - l.decision.pred_olp_base)
        .collect();
    let risk_labels = residuals.len();
    let gate_residual_s = if risk_labels > 0 && risk_labels as u64 >= params.gate_min_labels {
        median(&mut residuals).unwrap_or(0.0)
    } else {
        0.0
    };
    StepStats {
        index,
        start_s,
        n_labels: rows.len(),
        pairs,
        conc_olp: share(agree_olp),
        conc_cf: share(agree_cf),
        mape_olp: mean(&ape_olp),
        mape_cf: mean(&ape_cf),
        p50_olp: median(&mut ape_olp),
        p50_cf: median(&mut ape_cf),
        risk_labels,
        gate_residual_s,
        fit: label_fit(rows, params),
    }
}

fn ape(pred: f64, y: f64) -> f64 {
    (pred - y).abs() / y
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    Some(if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    })
}

fn sign(x: f64) -> i8 {
    match x.partial_cmp(&0.0) {
        Some(std::cmp::Ordering::Greater) => 1,
        Some(std::cmp::Ordering::Less) => -1,
        _ => 0,
    }
}

/// The label pairs: decisions at most `window_s` apart on different P whose
/// labels differ by more than 0.05 s. Returns (pairs, pairs that OLP orders
/// like the labels, pairs that the cost function's proxy orders like the
/// labels). Sorted by decision time with a sliding inner loop:
/// O(n log n + n * neighbours).
fn concordance(rows: &[&LabelRow], window_s: f64) -> (usize, usize, usize) {
    let mut by_decision = rows.to_vec();
    by_decision.sort_by(|a, b| a.decision.t_decision.total_cmp(&b.decision.t_decision));
    let (mut pairs, mut agree_olp, mut agree_cf) = (0, 0, 0);
    for (i, a) in by_decision.iter().enumerate() {
        for b in &by_decision[i + 1..] {
            if b.decision.t_decision - a.decision.t_decision > window_s {
                break;
            }
            if a.decision.url == b.decision.url || (a.y - b.y).abs() <= PAIR_MIN_LABEL_GAP_S {
                continue;
            }
            let truth = sign(a.y - b.y);
            let olp_agrees = sign(a.decision.pred_olp_c - b.decision.pred_olp_c) == truth;
            let cf_agrees = sign(a.decision.pred_cf_c - b.decision.pred_cf_c) == truth;
            pairs += 1;
            agree_olp += usize::from(olp_agrees);
            agree_cf += usize::from(cf_agrees);
        }
    }
    (pairs, agree_olp, agree_cf)
}

/// The label scale (scale source `label`) over the step's labels: weighted
/// least squares of y on [1, q_model, own_model] with w = 1 / (y + 1)^2 over
/// the rows without a pull; a is clamped to [0, a_max] and bq, bo to
/// [b_min, b_max], then each is shrunk toward the prior (0, 1, 1) with the
/// prior weight n0 = `label_n0`:
///
/// ```text
/// a' = n * a / (n + n0),  b' = (n * b + n0) / (n + n0)
/// ```
///
/// Fewer than `label_min_rows` rows, or a singular system, keep (0, 1, 1).
pub fn label_fit(rows: &[&LabelRow], params: &OlpParams) -> LabelFit {
    let n = rows.iter().filter(|l| !l.decision.pulled).count();
    if n == 0 || (n as u64) < params.label_min_rows {
        return LabelFit::prior(n, "few_rows");
    }
    // The normal equations [X'WX | X'Wy].
    let mut normal = [[0.0_f64; 4]; 3];
    for l in rows.iter().filter(|l| !l.decision.pulled) {
        let w = 1.0 / ((l.y + 1.0) * (l.y + 1.0));
        let x = [1.0, l.decision.q_model_s, l.decision.own_model_s, l.y];
        for (row, xi) in normal.iter_mut().zip(x) {
            for (cell, xj) in row.iter_mut().zip(x) {
                *cell += w * xi * xj;
            }
        }
    }
    let Some([a, bq, bo]) = solve3(normal) else {
        return LabelFit::prior(n, "singular");
    };
    let (n_f, n0) = (n as f64, params.label_n0);
    let a = a.max(0.0).min(params.a_max);
    let bq = bq.max(params.b_min).min(params.b_max);
    let bo = bo.max(params.b_min).min(params.b_max);
    LabelFit {
        a: n_f * a / (n_f + n0),
        bq: (n_f * bq + n0) / (n_f + n0),
        bo: (n_f * bo + n0) / (n_f + n0),
        n,
        status: "fit",
    }
}

/// Solve the 3x3 system of the augmented matrix `m` = [A | b] by
/// Gauss-Jordan elimination with partial pivoting. None when a pivot is at
/// most [`PIVOT_EPS`] times the largest entry of A (singular) or the result
/// is not finite.
fn solve3(mut m: [[f64; 4]; 3]) -> Option<[f64; 3]> {
    let scale = m
        .iter()
        .flat_map(|row| &row[..3])
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    for col in 0..3 {
        let pivot = (col..3).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))?;
        if m[pivot][col].abs() <= PIVOT_EPS * scale {
            return None;
        }
        m.swap(col, pivot);
        let d = m[col][col];
        let head = m[col].map(|v| v / d);
        m[col] = head;
        for (r, row) in m.iter_mut().enumerate() {
            if r != col {
                let f = row[col];
                for (cell, h) in row.iter_mut().zip(head) {
                    *cell -= f * h;
                }
            }
        }
    }
    let x = [m[0][3], m[1][3], m[2][3]];
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Remove the OLP keys (`olp` and every candidate's `olp_*`) from a trade
/// decision; returns how many were removed (tests: the rest must equal the
/// decision without OLP).
#[cfg(test)]
pub(crate) fn strip_olp_keys(decision: &mut Value) -> usize {
    let mut removed = 0;
    if let Some(object) = decision.as_object_mut() {
        removed += usize::from(object.remove("olp").is_some());
        if let Some(rows) = object.get_mut("candidates").and_then(Value::as_array_mut) {
            for row in rows.iter_mut().filter_map(Value::as_object_mut) {
                let keys: Vec<String> = row
                    .keys()
                    .filter(|k| k.starts_with("olp_"))
                    .cloned()
                    .collect();
                removed += keys.len();
                for key in keys {
                    row.remove(&key);
                }
            }
        }
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
    }

    fn state(params: OlpParams) -> OlpState {
        OlpState::new(OlpConfig {
            params,
            ..OlpConfig::default()
        })
    }

    fn load(busy_us: u64, tokens: u64) -> Value {
        json!({
            "total_prefill_busy_us": busy_us,
            "total_prefill_uncached_tokens": tokens,
            "max_total_num_tokens": 100_000,
        })
    }

    /// Feed `url` `n` one-second counter intervals ending at t = 1..=n, each
    /// with `busy_s` engine seconds for 1,000 tokens whose modelled time is
    /// `model_s` (k_obs = busy_s / model_s).
    fn feed(olp: &mut OlpState, url: &str, n: u64, busy_s: f64, model_s: f64) {
        olp.observe_poll(url, 0.0, "e1", &load(0, 0), 0.0, 0);
        for i in 1..=n {
            let busy_us = (i as f64 * busy_s * 1e6).round() as u64;
            let reading = load(busy_us, i * 1_000);
            olp.observe_poll(url, i as f64, "e1", &reading, model_s, 1_000);
        }
    }

    fn label(
        t_decision: f64,
        url: &str,
        y: f64,
        pred_olp: f64,
        pred_cf: f64,
        risk: bool,
    ) -> LabelRow {
        LabelRow {
            t_done: t_decision + y,
            y,
            decision: PendingLabel {
                t_decision,
                url: url.into(),
                pred_olp_c: pred_olp,
                pred_olp_base: pred_olp,
                pred_cf_c: pred_cf,
                risk,
                q_model_s: 0.0,
                own_model_s: 0.0,
                pulled: false,
            },
        }
    }

    #[test]
    fn work_prices_passes_tokens_pairs_and_the_first_restore() {
        let p = OlpParams::default();
        assert_eq!(p.work(0, 50_000, true), 0.0);
        let one = p.c0 + p.c1 * 8_192.0 + p.c2 * (8_192.0 * 8_193.0 / 2.0);
        assert!(close(p.work(8_192, 0, false), one));
        let (u, h) = (8_193.0, 1_000.0);
        let two = 2.0 * p.c0 + p.c1 * u + p.c2 * (u * h + u * (u + 1.0) / 2.0);
        assert!(close(p.work(8_193, 1_000, false), two));
        let restore = p.c3 * 0.45 * h;
        assert!(close(p.work(8_193, 1_000, true), two + restore));
        // Rows: one not started (pays its restore), one started (launched).
        let queue = p.queue([
            OlpRow {
                remaining: 8_193,
                input: 9_193,
                started: false,
            },
            OlpRow {
                remaining: 8_192,
                input: 8_192,
                started: true,
            },
        ]);
        assert!(close(queue.q_model_s, two + restore + one));
        assert_eq!((queue.rows, queue.launched_tokens), (2, 8_192));
    }

    #[test]
    fn windows_slide_and_reset_when_empty() {
        let mut w = Window::default();
        let sample = |busy_s: f64| Sums {
            busy_s,
            tok: 1.0,
            model_s: 0.5,
            model_tok: 1.0,
        };
        for t in [0.0, 1.0, 2.0] {
            w.push(t, sample(t + 1.0));
        }
        // Span 2 s at t = 2.5: the sample at t = 0 leaves.
        w.evict(2.5, 2.0);
        assert_eq!(w.samples.len(), 2);
        assert!(close(w.sums().busy_s, 5.0));
        assert!(close(w.sums().tok, 2.0));
        w.evict(10.0, 2.0);
        assert!(w.samples.is_empty());
        assert_eq!(w.sums(), Sums::default());
    }

    /// No data: k = k_pool = 1. With data: k = (B k_obs + B0 prior) / (B + B0)
    /// with the pooled prior; a P without own samples takes the prior.
    #[test]
    fn busy_scale_shrinks_from_the_pooled_prior_to_the_observation() {
        let mut olp = state(OlpParams::default());
        let (k_pool, busy) = olp.pooled(0.0);
        assert_eq!((k_pool, busy), (1.0, 0.0));
        assert_eq!(olp.busy_scale("a", 0.0, k_pool), (1.0, 0.0));
        // a: 1.5 busy s per 0.5 modelled s (k_obs 3); b: 0.5 per 0.5 (k_obs 1).
        feed(&mut olp, "a", 50, 1.5, 0.5);
        feed(&mut olp, "b", 50, 0.5, 0.5);
        let (k_pool, busy) = olp.pooled(50.0);
        assert!(close(k_pool, 2.0) && close(busy, 100.0), "{k_pool} {busy}");
        let (k_a, busy_a) = olp.busy_scale("a", 50.0, k_pool);
        assert!(close(busy_a, 75.0), "{busy_a}");
        assert!(close(k_a, (75.0 * 3.0 + 5.0 * 2.0) / 80.0), "{k_a}");
        let (k_b, _) = olp.busy_scale("b", 50.0, k_pool);
        assert!(close(k_b, (25.0 + 5.0 * 2.0) / 30.0), "{k_b}");
        assert_eq!(olp.busy_scale("c", 50.0, k_pool).0, k_pool);
        // The own window (120 s) forgets first; the pooled one (300 s) holds.
        let (k_pool, _) = olp.pooled(200.0);
        assert!(close(k_pool, 2.0));
        assert_eq!(olp.busy_scale("a", 200.0, k_pool), (k_pool, 0.0));
    }

    /// k_pool = Σ busy / Σ modelled: a P with more modelled work weighs more
    /// (a: 30 / 10, b: 10 / 20 -> 40 / 30, not the mean ratio 1.75).
    #[test]
    fn pooled_prior_weights_each_p_by_its_modelled_time() {
        let mut olp = state(OlpParams::default());
        feed(&mut olp, "a", 10, 3.0, 1.0);
        feed(&mut olp, "b", 10, 1.0, 2.0);
        let (k_pool, busy) = olp.pooled(10.0);
        assert!(close(k_pool, 40.0 / 30.0), "{k_pool}");
        assert!(close(busy, 40.0));
    }

    /// P->D: the scale at the switch becomes the history. D->P 1,920 s later
    /// (no samples left anywhere, k_pool = 1): the prior is the decayed
    /// history w * k_hist + (1 - w), w = exp(-1920 / 1800).
    #[test]
    fn history_decays_after_a_p_d_p_round_trip() {
        let p = OlpParams::default();
        let mut olp = state(p);
        feed(&mut olp, "a", 50, 1.5, 0.5);
        feed(&mut olp, "b", 50, 0.5, 0.5);
        let ctx = olp.begin(&["a", "b"], 50.0);
        assert!(olp.workers["a"].in_pool && olp.workers["b"].in_pool);
        let (k_a, _) = olp.busy_scale("a", 50.0, ctx.k_pool);
        olp.begin(&["b"], 60.0);
        assert!(!olp.workers["a"].in_pool);
        let (k_hist, left_s) = olp.workers["a"].k_hist.unwrap();
        assert!(close(k_hist, k_a) && left_s == 60.0, "{k_hist} {left_s}");
        let t = 1_980.0;
        let ctx = olp.begin(&["a", "b"], t);
        assert_eq!(ctx.k_pool, 1.0);
        assert!(olp.workers["a"].in_pool);
        let w = (-(t - 60.0) / p.tau_hist_s).exp();
        let (k, busy) = olp.busy_scale("a", t, ctx.k_pool);
        assert_eq!(busy, 0.0);
        assert!(close(k, w * k_a + (1.0 - w)), "{k}");
    }

    #[test]
    fn epoch_change_resets_and_bad_intervals_are_skipped() {
        let mut olp = state(OlpParams::default());
        feed(&mut olp, "a", 10, 1.0, 0.5);
        olp.begin(&["a"], 10.0);
        olp.begin(&["b"], 10.0);
        assert!(olp.workers["a"].k_hist.is_some());
        let before = olp.workers["a"].own.sums();
        // An interval above 2 s is not sampled (its chunk work is dropped).
        olp.observe_poll("a", 12.5, "e1", &load(13_000_000, 13_000), 0.7, 3_000);
        assert_eq!(olp.workers["a"].own.sums(), before);
        // Counters below the last reading (a reset without a new epoch).
        olp.observe_poll("a", 13.0, "e1", &load(1_000, 100), 0.1, 100);
        assert_eq!(olp.workers["a"].own.sums(), before);
        // The next interval from that reading is sampled again.
        olp.observe_poll("a", 14.0, "e1", &load(501_000, 1_100), 0.25, 1_000);
        let busy = olp.workers["a"].own.sums().busy_s;
        assert!(close(busy, before.busy_s + 0.5));
        // A new engine epoch: windows, counters and history start over.
        olp.observe_poll("a", 15.0, "e2", &load(5, 5), 0.0, 0);
        let a = &olp.workers["a"];
        assert_eq!(a.own.sums(), Sums::default());
        assert_eq!(a.pool.sums(), Sums::default());
        assert!(a.k_hist.is_none());
        assert_eq!(a.last_counter, Some((15.0, 5, 5)));
        assert_eq!(a.epoch.as_deref(), Some("e2"));
        // Without counters only the poll is recorded.
        olp.observe_poll("a", 15.5, "e2", &json!({}), 0.3, 100);
        let a = &olp.workers["a"];
        assert_eq!(a.pending, Sums::default());
        assert_eq!(a.last_poll_t_s, Some(15.5));
        assert_eq!(a.max_total_tokens, Some(100_000));
    }

    #[test]
    fn busy_scale_is_clamped() {
        let p = OlpParams {
            b0_s: 1e-9,
            ..OlpParams::default()
        };
        let mut olp = state(p);
        feed(&mut olp, "fast", 10, 0.001, 1.0);
        feed(&mut olp, "slow", 10, 100.0, 1.0);
        assert_eq!(olp.busy_scale("fast", 10.0, 1.0).0, p.k_min);
        assert_eq!(olp.busy_scale("slow", 10.0, 1.0).0, p.k_max);
    }

    /// Pairs: (L1, L2) and (L2, L3); (L1, L3) share a P and L4 is too late.
    /// OLP orders both pairs like the labels, the proxy neither. The risky
    /// residuals 1 and 2 give a median of 1.5 with 2 risky labels needed.
    #[test]
    fn step_statistics_on_a_hand_made_set() {
        let mut l4 = label(11.0, "b", 4.0, 4.4, 1.0, true);
        l4.decision.pred_olp_base = 2.0;
        let rows = [
            label(10.0, "a", 2.0, 1.8, 3.0, false),
            label(10.1, "b", 5.0, 4.0, 2.0, true),
            label(10.15, "a", 3.0, 3.5, 2.5, false),
            l4,
        ];
        let refs: Vec<&LabelRow> = rows.iter().collect();
        let p = OlpParams {
            gate_min_labels: 2,
            ..OlpParams::default()
        };
        let s = step_stats(&refs, &p, 3, 180.0);
        assert_eq!((s.index, s.start_s, s.n_labels, s.pairs), (3, 180.0, 4, 2));
        assert_eq!((s.conc_olp, s.conc_cf), (Some(1.0), Some(0.0)));
        let mape = (0.1 + 0.2 + 0.5 / 3.0 + 0.1) / 4.0;
        assert!(close(s.mape_olp.unwrap(), mape));
        assert!(close(s.p50_olp.unwrap(), (0.1 + 0.5 / 3.0) / 2.0));
        assert!(
            close(s.p50_cf.unwrap(), (0.5 + 0.6) / 2.0),
            "{:?}",
            s.p50_cf
        );
        assert_eq!(s.risk_labels, 2);
        assert!(close(s.gate_residual_s, 1.5));
        assert_eq!(s.fit.status, "few_rows");
        // The trace names the step's start `start_s` (`step_s` is a length).
        let value = s.value();
        assert_eq!(value["start_s"], 180.0);
        assert!(value.get("step_s").is_none());
        let p3 = OlpParams {
            gate_min_labels: 3,
            ..OlpParams::default()
        };
        assert_eq!(step_stats(&refs, &p3, 3, 180.0).gate_residual_s, 0.0);
        let narrow = OlpParams {
            pair_window_s: 0.06,
            ..p
        };
        assert_eq!(step_stats(&refs, &narrow, 3, 180.0).pairs, 1);
        let empty = step_stats(&[], &p, 0, 0.0);
        assert_eq!(
            (empty.pairs, empty.conc_olp, empty.mape_olp),
            (0, None, None)
        );
    }

    #[test]
    fn label_fit_recovers_clamps_and_shrinks() {
        let p = OlpParams::default();
        let row = |q: f64, o: f64, y: f64, pulled: bool| {
            let mut l = label(0.0, "a", y, 0.0, 0.0, false);
            l.decision.q_model_s = q;
            l.decision.own_model_s = o;
            l.decision.pulled = pulled;
            l
        };
        let make = |a: f64, bq: f64, bo: f64, n: usize| -> Vec<LabelRow> {
            (0..n)
                .map(|i| {
                    let q = (i % 7) as f64 * 0.5;
                    let o = (i % 5) as f64 * 0.3 + 0.1;
                    row(q, o, a + bq * q + bo * o, false)
                })
                .collect()
        };
        // Exact data inside the clamps: (0.3, 1.1, 0.9); with n = n0 = 500
        // the shrinkage halves the step from (0, 1, 1). Pulled rows never
        // enter the fit.
        let mut rows = make(0.3, 1.1, 0.9, 500);
        rows.extend((0..100).map(|_| row(1.0, 1.0, 50.0, true)));
        let fit = label_fit(&rows.iter().collect::<Vec<_>>(), &p);
        assert_eq!((fit.n, fit.status), (500, "fit"));
        assert!(
            close(fit.a, 0.15) && close(fit.bq, 1.05) && close(fit.bo, 0.95),
            "{fit:?}"
        );
        // Clamps before the shrinkage: a -0.2 -> 0, bq 2 -> 1.25.
        let rows = make(-0.2, 2.0, 1.0, 500);
        let fit = label_fit(&rows.iter().collect::<Vec<_>>(), &p);
        assert!(
            close(fit.a, 0.0) && close(fit.bq, 1.125) && close(fit.bo, 1.0),
            "{fit:?}"
        );
        // a 2 -> a_max 0.5, bq 0.5 -> b_min 0.8.
        let rows = make(2.0, 0.5, 1.0, 500);
        let fit = label_fit(&rows.iter().collect::<Vec<_>>(), &p);
        assert!(
            close(fit.a, 0.25) && close(fit.bq, 0.9) && close(fit.bo, 1.0),
            "{fit:?}"
        );
        // Too few rows: the prior.
        let rows = make(0.3, 1.1, 0.9, 49);
        let fit = label_fit(&rows.iter().collect::<Vec<_>>(), &p);
        assert_eq!(fit, LabelFit::prior(49, "few_rows"));
        // No queue on any row: singular, the prior.
        let rows: Vec<LabelRow> = (0..100)
            .map(|i| row(0.0, 0.1 + (i % 5) as f64, 1.0 + i as f64 * 0.01, false))
            .collect();
        let fit = label_fit(&rows.iter().collect::<Vec<_>>(), &p);
        assert_eq!(fit, LabelFit::prior(100, "singular"));
    }

    /// The gate reasons in order: cold (pooled busy, then labels), stale (an
    /// old or missing poll), no_pairs, not_better (beyond the slack).
    #[test]
    fn gate_reasons_come_in_order() {
        let p = OlpParams::default();
        let mut olp = state(p);
        let fresh = OlpPrediction {
            poll_age_s: Some(0.1),
            ..OlpPrediction::default()
        };
        let old = OlpPrediction {
            poll_age_s: Some(p.fresh_s + 0.1),
            ..OlpPrediction::default()
        };
        let never = OlpPrediction::default();
        let mut step = step_stats(&[], &p, 0, 0.0);
        step.pairs = 250;
        step.conc_olp = Some(0.6);
        step.conc_cf = Some(0.6);
        let mut ctx = OlpContext {
            t_s: 0.0,
            k_pool: 1.0,
            pool_busy_s: 0.0,
            step,
            new_step: false,
        };
        assert_eq!(olp.gate(&ctx, &[old]), Some(OlpFallback::Cold));
        ctx.pool_busy_s = p.warm_busy_s;
        olp.labels_total = p.warm_labels - 1;
        assert_eq!(olp.gate(&ctx, &[fresh]), Some(OlpFallback::Cold));
        olp.labels_total = p.warm_labels;
        assert_eq!(olp.gate(&ctx, &[fresh, old]), Some(OlpFallback::Stale));
        assert_eq!(olp.gate(&ctx, &[never]), Some(OlpFallback::Stale));
        ctx.step.pairs = 199;
        assert_eq!(olp.gate(&ctx, &[fresh]), Some(OlpFallback::NoPairs));
        ctx.step.pairs = 200;
        ctx.step.conc_olp = Some(0.58);
        assert_eq!(olp.gate(&ctx, &[fresh]), Some(OlpFallback::NotBetter));
        ctx.step.conc_olp = Some(0.595);
        assert_eq!(olp.gate(&ctx, &[fresh]), None, "within the slack");
    }

    /// wait and service per scale source; the park-gate residual only with
    /// risk_residual and a risky P (30k launched + 20k > 0.35 x 100k).
    #[test]
    fn predictions_follow_the_scale_source_and_the_risk_residual() {
        let queue = OlpQueue {
            q_model_s: 2.0,
            rows: 3,
            launched_tokens: 30_000,
        };
        for source in [
            OlpScaleSource::Label,
            OlpScaleSource::Busy,
            OlpScaleSource::Prior,
        ] {
            for risk_residual in [false, true] {
                let p = OlpParams {
                    scale_source: source,
                    risk_residual,
                    ..OlpParams::default()
                };
                let mut olp = state(p);
                feed(&mut olp, "a", 10, 1.0, 0.5);
                let mut step = step_stats(&[], &p, 0, 0.0);
                step.fit = LabelFit {
                    a: 0.2,
                    bq: 1.1,
                    bo: 0.9,
                    n: 300,
                    status: "fit",
                };
                step.gate_residual_s = 1.5;
                let ctx = OlpContext {
                    t_s: 10.0,
                    k_pool: 1.0,
                    pool_busy_s: 0.0,
                    step,
                    new_step: false,
                };
                let (k, _) = olp.busy_scale("a", 10.0, 1.0);
                let own = p.work(5_000, 15_000, true);
                let pred = olp.predict(&ctx, "a", queue, 20_000, 5_000);
                let (wait, service) = match source {
                    OlpScaleSource::Label => (0.2 + 1.1 * 2.0, 0.9 * own),
                    OlpScaleSource::Busy => (k * 2.0, k * own),
                    OlpScaleSource::Prior => (2.0, own),
                };
                let residual = if risk_residual { 1.5 } else { 0.0 };
                assert!(pred.risk);
                assert_eq!(pred.cap_tokens, Some(35_000.0));
                assert!(close(pred.wait_base_s, wait), "{source:?}");
                assert!(close(pred.wait_s, wait + residual), "{source:?}");
                assert!(close(pred.service_s, service), "{source:?}");
                assert_eq!(pred.poll_age_s, Some(0.0));
                assert_eq!((pred.rows, pred.launched_tokens), (3, 30_000));
                // The pull hides behind the wait; only its excess is exposed.
                assert!(close(pred.completion_s(0.0), pred.wait_s + pred.service_s));
                let long = pred.wait_s + 1.0;
                assert!(close(pred.completion_s(long), long + pred.service_s));
            }
        }
    }

    #[test]
    fn only_successful_answers_label_and_old_labels_leave() {
        let mut olp = state(OlpParams::default());
        let pending = |t_decision: f64| PendingLabel {
            t_decision,
            url: "a".into(),
            pred_olp_c: 1.0,
            pred_olp_base: 1.0,
            pred_cf_c: 1.0,
            risk: false,
            q_model_s: 0.0,
            own_model_s: 0.0,
            pulled: false,
        };
        for id in 1..=3 {
            olp.record(id, pending(10.0));
        }
        assert!(olp.resolve(1, 12.0, true));
        assert!(!olp.resolve(2, 12.0, false), "an error is censored");
        assert!(!olp.resolve(3, 10.0, true), "y = 0 is no label");
        assert!(!olp.resolve(1, 13.0, true), "resolved once");
        assert_eq!((olp.labels_total(), olp.pending_len()), (1, 0));
        assert_eq!(olp.labels[0].y, 2.0);
        // A label older than the window + 120 s leaves at the next label.
        olp.record(4, pending(500.0));
        assert!(olp.resolve(4, 600.0, true));
        assert_eq!((olp.labels.len(), olp.labels_total()), (1, 2));
        // A step reads the labels done in [start - 300, start).
        assert_eq!(olp.stats_at(650.0).n_labels, 0, "done at the start");
        assert_eq!(olp.stats_at(661.0).n_labels, 1);
    }

    #[test]
    fn live_index_tracks_inserts_and_removals() {
        let mut olp = state(OlpParams::default());
        let row = OlpRow {
            remaining: 10,
            input: 20,
            started: false,
        };
        olp.live_insert("a", 3, row);
        olp.live_insert("a", 1, row);
        olp.live_insert("b", 2, row);
        assert_eq!(olp.live("a").collect::<Vec<_>>(), [1, 3]);
        olp.live_remove("a", 1);
        olp.live_remove("a", 3);
        olp.live_remove("c", 9);
        assert_eq!(olp.live("a").count(), 0);
        assert!(!olp.live.contains_key("a"));
        assert_eq!(olp.live("b").collect::<Vec<_>>(), [2]);
    }

    /// One row's integer terms are those of `work`: c0 * passes + c1 * u +
    /// c2 * pairs + c3 * rr * restore equals work(u, h, !started); u = 0
    /// costs nothing (no restore); a started row launches its input.
    #[test]
    fn queue_sums_of_one_row_are_the_work_terms() {
        let p = OlpParams::default();
        for (remaining, input, started) in [
            (0, 50_000, false),
            (0, 50_000, true),
            (1, 1, false),
            (8_192, 8_192, false),
            (8_193, 9_193, false),
            (8_193, 9_193, true),
            (1_000_000, 1_048_576, false),
        ] {
            let row = OlpRow {
                remaining,
                input,
                started,
            };
            let sums = QueueAgg::of_row(row, p.chunk);
            let queue = p.queue_of(&sums);
            let work = p.work(remaining, input - remaining, !started);
            assert!(
                close(queue.q_model_s, work),
                "{row:?}: {} vs {work}",
                queue.q_model_s
            );
            assert_eq!(sums.rows, 1);
            let restore = if started || remaining == 0 {
                0
            } else {
                input - remaining
            };
            assert_eq!(sums.restore, u128::from(restore), "{row:?}");
            assert_eq!(queue.launched_tokens, if started { input } else { 0 });
        }
    }

    /// The incremental sums under random row events (insert, update of the
    /// remaining work and of the launch, removal, a second insert of the
    /// same ID, updates of unknown IDs) on three P with a small and the
    /// default chunk: after every event each P's sums are the exact sums of
    /// its rows, and its queue equals the walk's (rows and launched tokens
    /// exactly, q_model to 1e-12).
    #[test]
    fn queue_sums_equal_the_walk_under_random_row_events() {
        for chunk in [7, 8_192] {
            let params = OlpParams {
                chunk,
                ..OlpParams::default()
            };
            let mut olp = OlpState::new(OlpConfig {
                params,
                incremental: true,
                ..OlpConfig::default()
            });
            let urls = ["a", "b", "c"];
            let mut rows: BTreeMap<u64, (usize, OlpRow)> = BTreeMap::new();
            let mut x: u64 = 12_345;
            let mut next = |n: u64| {
                x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                (x >> 33) % n.max(1)
            };
            for step in 0..5_000u64 {
                match next(5) {
                    0 | 1 => {
                        let id = next(400);
                        let input = 1 + next(1_048_576);
                        let row = OlpRow {
                            remaining: next(input + 1),
                            input,
                            started: next(2) == 0,
                        };
                        let url = next(3) as usize;
                        // A second insert of a live ID replaces its row (and P).
                        if let Some((old_url, _)) = rows.get(&id) {
                            olp.live_remove(urls[*old_url], id);
                        }
                        olp.live_insert(urls[url], id, row);
                        rows.insert(id, (url, row));
                    }
                    2 | 3 => {
                        let id = next(400);
                        let Some((_, row)) = rows.get_mut(&id) else {
                            // Not live: ignored.
                            olp.live_update(
                                id,
                                OlpRow {
                                    remaining: 1,
                                    input: 2,
                                    started: true,
                                },
                            );
                            continue;
                        };
                        row.remaining = next(row.remaining + 1);
                        row.started |= next(3) == 0;
                        olp.live_update(id, *row);
                    }
                    _ => {
                        let id = next(400);
                        if let Some((url, _)) = rows.remove(&id) {
                            olp.live_remove(urls[url], id);
                        }
                    }
                }
                for (i, url) in urls.iter().enumerate() {
                    let mine: Vec<OlpRow> = rows
                        .values()
                        .filter(|(u, _)| *u == i)
                        .map(|(_, r)| *r)
                        .collect();
                    let mut exact = QueueAgg::default();
                    for r in &mine {
                        exact.add(&QueueAgg::of_row(*r, chunk));
                    }
                    let sums = olp.queue_sums(url).unwrap();
                    assert_eq!(sums, exact, "chunk {chunk} step {step} {url}");
                    let (walk, fast) = (params.queue(mine.iter().copied()), params.queue_of(&sums));
                    assert_eq!(walk.rows, fast.rows);
                    assert_eq!(walk.launched_tokens, fast.launched_tokens);
                    let scale = walk.q_model_s.abs().max(1e-300);
                    assert!(
                        (walk.q_model_s - fast.q_model_s).abs() <= 1e-12 * scale,
                        "chunk {chunk} step {step} {url}: {} vs {}",
                        walk.q_model_s,
                        fast.q_model_s
                    );
                    assert_eq!(olp.live(url).count(), mine.len());
                }
            }
            // Everything removed: no P keeps sums.
            for (id, (url, _)) in std::mem::take(&mut rows) {
                olp.live_remove(urls[url], id);
            }
            assert!(olp.agg.is_empty() && olp.rows.is_empty());
        }
    }

    /// Without `incremental` no sums are kept (the walk prices the queue)
    /// and the snapshot has no `queue` key; with it the snapshot names it.
    #[test]
    fn incremental_off_keeps_no_sums() {
        let row = OlpRow {
            remaining: 10,
            input: 20,
            started: false,
        };
        let mut off = state(OlpParams::default());
        off.live_insert("a", 1, row);
        off.live_update(
            1,
            OlpRow {
                remaining: 5,
                ..row
            },
        );
        assert!(off.queue_sums("a").is_none());
        assert!(off.rows.is_empty() && off.agg.is_empty());
        assert!(off.snapshot_value(0.0).get("queue").is_none());
        let mut on = OlpState::new(OlpConfig {
            incremental: true,
            ..OlpConfig::default()
        });
        on.live_insert("a", 1, row);
        assert_eq!(on.queue_sums("a").unwrap().rows, 1);
        assert_eq!(on.queue_sums("b").unwrap(), QueueAgg::default());
        let snap = on.snapshot_value(0.0);
        assert_eq!(snap["queue"], "incremental");
        assert_eq!(snap["queue_rows"], 1);
        assert!(snap.get("verify").is_none());
    }

    /// A step's first decision computes its statistics once (new_step), the
    /// next decisions of the step reuse them.
    #[test]
    fn step_statistics_are_computed_once_per_step() {
        let mut olp = state(OlpParams::default());
        assert!(olp.begin(&["a"], 61.0).new_step);
        assert!(!olp.begin(&["a"], 119.9).new_step);
        let ctx = olp.begin(&["a"], 120.0);
        assert!(ctx.new_step);
        assert_eq!((ctx.step.index, ctx.step.start_s), (2, 120.0));
        let event = olp.step_event(&ctx);
        assert_eq!(event["event"], "olp_step");
        assert_eq!(event["workers"][0]["url"], "a");
        assert_eq!(event["workers"][0]["in_pool"], true);
        assert_eq!(event["params"]["scale_source"], "label");
    }

    /// The gate results: counted since the start and per step; the first
    /// decision of a step closes the step before (`gate_last_step`, with its
    /// start). The snapshot object shows the warm-up inputs of the last
    /// decision, the pool workers with a fresh busy-counter reading and both
    /// counts.
    #[test]
    fn gate_results_are_counted_per_step_and_in_total() {
        let mut olp = state(OlpParams::default());
        assert!(olp.snapshot_value(0.0)["last_decision_s"].is_null());
        assert_eq!(GateCounts::default().open_share(), None);
        // Step 0: an open and a cold decision; a has 1 busy second.
        feed(&mut olp, "a", 2, 0.5, 0.5);
        let ctx = olp.begin(&["a", "b"], 2.5);
        olp.count_gate(None);
        olp.count_gate(Some(OlpFallback::Cold));
        let event = olp.step_event(&ctx);
        assert!(event["gate_last_step"].is_null());
        assert_eq!(event["gate_total"]["decisions"], 2);
        assert_eq!(event["workers"][0]["counter_age_s"], 0.5);
        assert!(event["workers"][1]["counter_age_s"].is_null());
        // Step 1 (60-s steps): b reads its counters; a stale decision.
        olp.observe_poll("b", 69.5, "e1", &load(0, 0), 0.0, 0);
        let ctx = olp.begin(&["a", "b"], 70.0);
        olp.count_gate(Some(OlpFallback::Stale));
        let event = olp.step_event(&ctx);
        let last = &event["gate_last_step"];
        assert_eq!(last["start_s"], 0.0);
        assert_eq!(last["decisions"], 2);
        assert_eq!(last["open"], 1);
        assert_eq!(last["cold"], 1);
        assert_eq!(last["open_share"], 0.5);
        let want = GateCounts {
            decisions: 3,
            open: 1,
            cold: 1,
            stale: 1,
            ..GateCounts::default()
        };
        assert_eq!(olp.gate_total(), want);
        assert!(close(want.open_share().unwrap(), 1.0 / 3.0));
        // a's last counter reading is older than fresh_s (2 s), b's is not.
        let snapshot = olp.snapshot_value(70.5);
        assert_eq!(snapshot["mode"], "olp");
        assert_eq!(snapshot["last_decision_s"], 70.0);
        assert_eq!(snapshot["pool_busy_s"], 1.0);
        assert_eq!(snapshot["pool_workers"], 2);
        assert_eq!(snapshot["pool_workers_with_counters"], 1);
        assert_eq!(snapshot["gate_step"]["decisions"], 1);
        assert_eq!(snapshot["gate_step"]["stale"], 1);
        assert_eq!(snapshot["gate_total"]["decisions"], 3);
        assert_eq!(snapshot["step"]["start_s"], 60.0);
    }

    #[test]
    fn params_file_overrides_by_name_and_refuses_bad_input() {
        let p =
            params_from_json(br#"{"c1": 4e-5, "scale_source": "busy", "min_pairs": 100}"#).unwrap();
        let want = OlpParams {
            c1: 4e-5,
            scale_source: OlpScaleSource::Busy,
            min_pairs: 100,
            ..OlpParams::default()
        };
        assert_eq!(p, want);
        assert_eq!(params_from_json(b"{}").unwrap(), OlpParams::default());
        for (text, want) in [
            (r#"{"c9": 1}"#, "unknown field"),
            ("[1, 2]", "JSON object"),
            ("not json", "line 1"),
            (r#"{"chunk": 0}"#, "chunk"),
            (r#"{"chunk": 8192.5}"#, "invalid type"),
            (r#"{"c1": 0}"#, "c1"),
            (r#"{"c0": -1}"#, "c0"),
            (r#"{"rr": 1.5}"#, "rr"),
            (r#"{"k_min": 9.0}"#, "k_min"),
            (r#"{"b_min": 1.5}"#, "b_min"),
            (r#"{"scale_source": "x"}"#, "unknown variant"),
        ] {
            let err = params_from_json(text.as_bytes()).unwrap_err();
            assert!(err.contains(want), "{text}: {err}");
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("olp.json");
        let text = br#"{"step_s": 30}"#;
        std::fs::write(&path, text).unwrap();
        let (p, sha) = load_params(&path).unwrap();
        assert_eq!(p.step_s, 30.0);
        assert_eq!(sha, hex_sha256(text));
        assert_eq!(sha.len(), 64);
        assert!(load_params(&dir.path().join("missing.json")).is_err());
        // The parameters round-trip through the trace form.
        let back: OlpParams = serde_json::from_value(json!(OlpParams::default())).unwrap();
        assert_eq!(back, OlpParams::default());
    }
}
