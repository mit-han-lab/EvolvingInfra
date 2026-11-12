//! The placement cost function of `--symmetric-pd-p-select cost-function`:
//! each P's predicted time to the request's first token, plus the fetch time
//! that the P's wait does not hide, plus the copy price of the fetched KV.
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::Instant;

/// Default prefill rate R of one P worker, in uncached tokens per second.
pub const DEFAULT_TRADE_RATE_TOKENS_PER_S: f64 = 13_500.0;
/// Default fixed part of one KV pull, in milliseconds.
pub const DEFAULT_TRADE_PULL_FIXED_MS: f64 = 450.0;
/// Default size-dependent part of one KV pull, in milliseconds per 100k tokens.
pub const DEFAULT_TRADE_PULL_MS_PER_100K: f64 = 100.0;
/// TRADE never pulls fewer tokens than this (no pull when the
/// best source match minus the target's own match is below 512). The
/// effective floor is max(--kv-pull-hint-min-incremental-tokens, this).
pub const TRADE_MIN_PULL_TOKENS: u64 = 512;
/// Default copy price, in uncached-token equivalents per pulled token: the
/// value of the paper's runs (half of the simulator estimate, 6.8e-4 s per
/// pulled token x 13,500 tokens/s = 9.18).
pub const DEFAULT_TRADE_COPY_LAMBDA: f64 = 4.6;

/// Where TRADE takes a P's wait and the request's own service time from
/// (`--trade-wait`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TradeWait {
    /// backlog / R and u / R from the router's uncached ledger.
    #[default]
    Ledger,
    /// The E3 P completion forecast ([`crate::e3`], [`crate::trade_e3`]):
    /// B_w = `wait_s` (decision to the GPU start of the request's first
    /// pass on P) and S_w = `pred_s - wait_s` (`pred_s` = decision to P's
    /// prefill completion of the request), so the cost is
    /// R x (pred_s + max(0, pull_s - wait_s)) + copy. The selection is
    /// [`select_trade_e3`]: the request leaves the cache-aware owner only
    /// when the gain beats the forecasts' error bounds (prediction gate).
    /// A P without a usable forecast has the ledger terms and never wins a
    /// move; the trace counts these fallbacks by reason.
    E3,
}

impl TradeWait {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ledger => "ledger",
            Self::E3 => "e3",
        }
    }
}

/// Default age limit of an engine snapshot for `--trade-wait e3`
/// (the view-age upper bound: poll sent to decision plus the engine's age).
pub const DEFAULT_TRADE_EFT_MAX_VIEW_AGE_S: f64 = 1.5;

/// Default floor of the auto copy price (`--trade-copy-lambda-min`),
/// in uncached-token equivalents per pulled token.
pub const DEFAULT_TRADE_AUTO_LAMBDA_MIN: f64 = 4.6;
/// Default cap of the auto copy price (`--trade-copy-lambda-max`).
pub const DEFAULT_TRADE_AUTO_LAMBDA_MAX: f64 = 1000.0;
/// Default time constant tau of the auto inputs, in seconds. It is also the
/// start-up period.
pub const DEFAULT_TRADE_AUTO_TAU_S: f64 = 120.0;
/// Default idle window of a session in N_active, in seconds.
pub const DEFAULT_TRADE_AUTO_SESSION_IDLE_S: f64 = 300.0;
/// [`TradeAutoState`] tracks at most this many sessions. A new session above
/// the limit is not counted (and counted in `sessions_dropped`).
pub const TRADE_AUTO_MAX_SESSIONS: usize = 100_000;

/// Where TRADE takes its copy price lambda from
/// (`--trade-copy-lambda <x|auto>`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TradeLambdaMode {
    /// lambda = [`TradeParams::copy_lambda`].
    #[default]
    Fixed,
    /// lambda = clamp(q * R / R_hat, min, max) from the router's own
    /// measurements ([`TradeParams::resolved`]). `copy_lambda` is the price
    /// during start-up and when an input is missing.
    Auto { min: f64, max: f64 },
}

impl TradeLambdaMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Auto { .. } => "auto",
        }
    }
}

/// Where TRADE takes its rate R from
/// (`--trade-rate-tokens-per-s <x|auto>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TradeRateMode {
    /// R = [`TradeParams::rate_tokens_per_s`].
    #[default]
    Fixed,
    /// R = R_hat, the measured prefill tokens per busy second of all P.
    /// `rate_tokens_per_s` is R during start-up and without R_hat.
    Auto,
}

impl TradeRateMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Auto => "auto",
        }
    }
}
/// Default multiple m of the combined error bound in the `wait = e3`
/// prediction gate (`--trade-gate-m`). With m = 1 the gate uses the bounds
/// of the error table as fitted.
pub const DEFAULT_TRADE_GATE_M: f64 = 1.0;

/// Parameters of the TRADE P-selection cost model (`--trade-*`).
///
/// All work is in uncached-token units; `rate_tokens_per_s` converts seconds
/// to tokens. The CLI validates the values (finite, R > 0, others >= 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TradeParams {
    /// R: uncached prefill tokens one P worker computes per second.
    pub rate_tokens_per_s: f64,
    /// Fixed time of one KV pull (ms).
    pub pull_fixed_ms: f64,
    /// Size-dependent time of one KV pull (ms per 100k pulled tokens).
    pub pull_ms_per_100k: f64,
    /// lambda: copy cost in uncached-token equivalents per pulled token. It
    /// prices the second copy that a pull leaves in the target's host pool.
    pub copy_lambda: f64,
    /// Source of the wait and service terms (default: the ledger).
    pub wait: TradeWait,
    /// `wait = e3`: a snapshot whose view age exceeds this (s) is not used.
    pub eft_max_view_age_s: f64,
    /// `Fixed` (default): lambda = `copy_lambda`. `Auto`: lambda comes from
    /// the router's measurements ([`TradeParams::resolved`]) and
    /// `copy_lambda` is the fallback (start-up, missing input).
    pub copy_lambda_mode: TradeLambdaMode,
    /// `Fixed` (default): R = `rate_tokens_per_s`. `Auto`: R = R_hat and
    /// `rate_tokens_per_s` is the fallback.
    pub rate_mode: TradeRateMode,
    /// tau > 0 (s): the time constant of every auto input
    /// ([`TradeAutoState`]); the auto values apply only after tau seconds
    /// of history (start-up). No effect while both modes are `Fixed`.
    pub auto_tau_s: f64,
    /// Seconds, above 0: a session without a request on a P counts in
    /// N_active for this long after its last router event (the reservation
    /// or the prefill end of its last request). An END from the client
    /// removes it at once. No effect while both modes are `Fixed`.
    pub auto_session_idle_s: f64,
    /// `wait = e3`: m >= 0, the multiple of the combined error bound
    /// sqrt(b_o^2 + b_w^2) in the prediction gate of [`select_trade_e3`].
    /// No effect with the ledger wait.
    pub gate_m: f64,
}

impl Default for TradeParams {
    /// The CLI defaults of the TRADE flags.
    fn default() -> Self {
        Self {
            rate_tokens_per_s: DEFAULT_TRADE_RATE_TOKENS_PER_S,
            pull_fixed_ms: DEFAULT_TRADE_PULL_FIXED_MS,
            pull_ms_per_100k: DEFAULT_TRADE_PULL_MS_PER_100K,
            copy_lambda: DEFAULT_TRADE_COPY_LAMBDA,
            wait: TradeWait::Ledger,
            eft_max_view_age_s: DEFAULT_TRADE_EFT_MAX_VIEW_AGE_S,
            copy_lambda_mode: TradeLambdaMode::Fixed,
            rate_mode: TradeRateMode::Fixed,
            auto_tau_s: DEFAULT_TRADE_AUTO_TAU_S,
            auto_session_idle_s: DEFAULT_TRADE_AUTO_SESSION_IDLE_S,
            gate_m: DEFAULT_TRADE_GATE_M,
        }
    }
}

/// The router's measurements at one TRADE decision, from [`TradeAutoState`]
/// (the inputs of [`TradeParams::resolved`]).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TradeAutoInputs {
    /// Seconds since the first auto decision (the start of the history).
    pub elapsed_s: f64,
    /// n_P: the healthy P workers of this decision.
    pub n_p: usize,
    /// N_active: sessions with a request on a P, or with a router event in
    /// the session idle window and no END.
    pub n_active: usize,
    /// busy: mean over the P with history of the busy fraction (time with at
    /// least one request dispatched and its prefill not ended). None without
    /// history.
    pub busy: Option<f64>,
    /// Mean over the P with history of the time-averaged in-flight requests.
    pub inflight: Option<f64>,
    /// R_hat: computed prefill tokens per busy second, pooled over the P with
    /// history. None without busy time or credited tokens.
    pub r_hat: Option<f64>,
}

impl TradeAutoInputs {
    /// q = N_active / n_P x busy: sessions per P times the P busy fraction,
    /// the E2E seconds that one extra second of P work costs (closed-loop
    /// upper bound). None without the input.
    pub fn q(&self) -> Option<f64> {
        match self.busy {
            Some(busy) if self.n_p > 0 => Some(self.n_active as f64 / self.n_p as f64 * busy),
            _ => None,
        }
    }
}

/// How [`TradeParams::resolved`] set the copy price and the rate of one
/// decision (the trace keys of an auto decision).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TradeAutoResolution {
    pub lambda_mode: &'static str,
    pub q: Option<f64>,
    /// q * R / R_hat before the clamp; None on a fallback.
    pub lambda_raw: Option<f64>,
    /// The copy price of this decision.
    pub copy_lambda_used: f64,
    /// Why lambda is the fallback value; None when the auto inputs were used
    /// (and always None with a fixed lambda).
    pub fallback_reason: Option<&'static str>,
    pub rate_mode: &'static str,
    /// The rate R of this decision.
    pub rate_used: f64,
    /// Why R is the fallback value (rate auto only).
    pub rate_fallback_reason: Option<&'static str>,
}

impl TradeParams {
    /// Predicted wall time of one pull of `pulled` tokens, in seconds.
    /// Zero tokens means no pull, which takes no time.
    pub fn pull_seconds(&self, pulled: u64) -> f64 {
        if pulled == 0 {
            return 0.0;
        }
        (self.pull_fixed_ms + self.pull_ms_per_100k * pulled as f64 / 100_000.0) / 1000.0
    }

    /// True with the ledger wait and a fixed copy price and rate: the trace
    /// then has no forecast or auto keys.
    pub fn is_original(&self) -> bool {
        self.wait == TradeWait::Ledger && !self.is_auto()
    }

    /// True when the copy price or the rate comes from the router's
    /// measurements (`auto`).
    pub fn is_auto(&self) -> bool {
        matches!(self.copy_lambda_mode, TradeLambdaMode::Auto { .. })
            || self.rate_mode == TradeRateMode::Auto
    }

    /// The start-up description of the copy price lambda and of the rate R
    /// (`auto` or fixed), as `(lambda, rate)`.
    pub fn auto_description(&self) -> (String, String) {
        let lambda = match self.copy_lambda_mode {
            TradeLambdaMode::Auto { min, max } => format!(
                "lambda = clamp(q*R/R_hat, min, max) with q = N_active/n_P*busy, \
                 min = {min}, max = {max}, fallback lambda = {}",
                self.copy_lambda,
            ),
            TradeLambdaMode::Fixed => format!("lambda fixed = {}", self.copy_lambda),
        };
        let rate = match self.rate_mode {
            TradeRateMode::Auto => {
                format!("R = R_hat (fallback R = {})", self.rate_tokens_per_s)
            }
            TradeRateMode::Fixed => format!("R fixed = {}", self.rate_tokens_per_s),
        };
        (lambda, rate)
    }

    /// The parameters of one decision, with `copy_lambda` and
    /// `rate_tokens_per_s` resolved from the router's measurements `auto`,
    /// and how they were resolved. [`trade_cost_full`] then uses them
    /// unchanged. With both modes `Fixed` the returned parameters are `self`
    /// (bit for bit).
    ///
    /// ```text
    /// R      = R_hat (rate auto), else rate_tokens_per_s
    /// q      = N_active / n_P * busy
    /// lambda = clamp(q * R / R_hat, min, max)    (lambda auto; R / R_hat = 1 with rate auto)
    /// ```
    ///
    /// lambda is in tokens of P work per pulled token: one pulled token
    /// causes at most one recomputed token (1 / R_hat seconds of P work),
    /// and each second of P work delays the sessions on that P by q seconds
    /// in total; R converts these seconds into the cost's token units.
    ///
    /// Fallback: lambda = `copy_lambda` (not clamped), with the first reason
    /// that applies: `startup` (less than `auto_tau_s` seconds of history),
    /// `no_sessions` (N_active = 0), `no_busy` (no P history), `no_rate`
    /// (no R_hat), `not_finite`. With rate auto, R = `rate_tokens_per_s` for
    /// `startup` and `no_rate`.
    pub fn resolved(&self, auto: &TradeAutoInputs) -> (TradeParams, TradeAutoResolution) {
        let q = auto.q();
        // Start-up until tau seconds of history (a NaN counts as start-up).
        let startup = matches!(
            auto.elapsed_s.partial_cmp(&self.auto_tau_s),
            Some(std::cmp::Ordering::Less) | None
        );
        let r_hat = auto.r_hat.filter(|r| r.is_finite() && *r > 0.0);
        let (rate, rate_fallback_reason) = match (self.rate_mode, startup, r_hat) {
            (TradeRateMode::Fixed, _, _) => (self.rate_tokens_per_s, None),
            (TradeRateMode::Auto, true, _) => (self.rate_tokens_per_s, Some("startup")),
            (TradeRateMode::Auto, false, Some(r)) => (r, None),
            (TradeRateMode::Auto, false, None) => (self.rate_tokens_per_s, Some("no_rate")),
        };
        let mut resolution = TradeAutoResolution {
            lambda_mode: self.copy_lambda_mode.as_str(),
            q,
            lambda_raw: None,
            copy_lambda_used: self.copy_lambda,
            fallback_reason: None,
            rate_mode: self.rate_mode.as_str(),
            rate_used: rate,
            rate_fallback_reason,
        };
        if let TradeLambdaMode::Auto { min, max } = self.copy_lambda_mode {
            let missing = if startup {
                Some("startup")
            } else if auto.n_active == 0 {
                Some("no_sessions")
            } else if q.is_none() {
                Some("no_busy")
            } else if r_hat.is_none() {
                Some("no_rate")
            } else {
                None
            };
            match (missing, q, r_hat) {
                (None, Some(q), Some(r_hat)) => {
                    let raw = q * (rate / r_hat);
                    if raw.is_finite() {
                        resolution.lambda_raw = Some(raw);
                        // max/min, not clamp: no panic on a config built in
                        // code with min > max (the CLI rejects it).
                        resolution.copy_lambda_used = raw.max(min).min(max);
                    } else {
                        resolution.fallback_reason = Some("not_finite");
                    }
                }
                (reason, _, _) => resolution.fallback_reason = reason,
            }
        }
        let params = TradeParams {
            copy_lambda: resolution.copy_lambda_used,
            rate_tokens_per_s: rate,
            ..*self
        };
        (params, resolution)
    }
}

/// The tokens a TRADE candidate pulls from another P: the best source match
/// minus the target's own match, both clamped to the request length. Returns
/// 0 (no pull; the target recomputes these tokens) when the gain is below
/// max(`floor`, [`TRADE_MIN_PULL_TOKENS`]); `floor` is
/// `--kv-pull-hint-min-incremental-tokens`.
pub fn trade_pull_tokens(local_hit: u64, remote_hit: u64, input_tokens: u64, floor: u64) -> u64 {
    let local = local_hit.min(input_tokens);
    let gain = remote_hit.min(input_tokens).saturating_sub(local);
    if gain < floor.max(TRADE_MIN_PULL_TOKENS) {
        0
    } else {
        gain
    }
}

/// The forecast terms of one P (`wait = e3`), in seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TradeWaitTerms {
    /// B_w: predicted time until the P has done the work ahead of the request.
    pub wait_s: f64,
    /// S_w: the request's own prefill service on the P.
    pub service_s: f64,
}

/// The parts of one candidate's TRADE cost, all in uncached-token units
/// except the times.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TradeCost {
    /// Work ahead in tokens: the pending uncached ledger, or R x B_w.
    pub backlog: f64,
    /// This request's own work in tokens: u(P) after the hinted pull, or R x S_w.
    pub uncached: f64,
    /// Predicted pull time (s); 0 without a pull.
    pub pull_s: f64,
    /// Pull time not hidden behind the P's predicted wait (s).
    pub exposed_pull_s: f64,
    /// R x exposed_pull_s.
    pub exposed_pull_tokens: f64,
    /// lambda x pulled tokens.
    pub copy_tokens: f64,
    /// backlog + uncached + exposed_pull_tokens + copy_tokens.
    pub total: f64,
    /// The wait term in seconds: B_w, or backlog / R.
    pub wait_s: f64,
    /// The own service in seconds: S_w, or u / R.
    pub service_s: f64,
}

/// TRADE cost of placing one request on one P worker:
///
/// ```text
/// cost(P) = backlog(P) + u(P)
///         + R * max(0, pull_s(pulled(P)) - backlog(P) / R)
///         + lambda * pulled(P)
/// pull_s(n) = (pull_fixed_ms + pull_ms_per_100k * n / 100000) / 1000,  pull_s(0) = 0
/// ```
///
/// `backlog` = pending uncached tokens on P; `uncached` = u(P), this
/// request's uncached tokens on P after the hinted pull (or without a pull
/// when it is below the floor); `pulled` = tokens the pull brings (0 = no
/// pull). The first three terms equal R x (max(backlog / R, pull_s) + u / R),
/// that is the predicted completion of the request in token units.
pub fn trade_cost(params: &TradeParams, backlog: u128, uncached: u64, pulled: u64) -> TradeCost {
    trade_cost_full(params, backlog, uncached, pulled, None)
}

/// [`trade_cost`] with the forecast terms:
///
/// ```text
/// wait = Some(B_w, S_w):
///   cost(P) = R * B_w + R * S_w + R * max(0, pull_s - B_w) + lambda * pulled
/// wait = None (ledger): the formula of trade_cost
/// ```
pub fn trade_cost_full(
    params: &TradeParams,
    backlog: u128,
    uncached: u64,
    pulled: u64,
    wait: Option<TradeWaitTerms>,
) -> TradeCost {
    let rate = params.rate_tokens_per_s;
    let pull_s = params.pull_seconds(pulled);
    let copy_tokens = params.copy_lambda * pulled as f64;
    let (backlog, own, wait_s, service_s) = match wait {
        None => {
            let backlog = backlog as f64;
            (
                backlog,
                uncached as f64,
                backlog / rate,
                uncached as f64 / rate,
            )
        }
        Some(terms) => (
            rate * terms.wait_s,
            rate * terms.service_s,
            terms.wait_s,
            terms.service_s,
        ),
    };
    let exposed_pull_s = if pulled == 0 {
        0.0
    } else {
        (pull_s - wait_s).max(0.0)
    };
    let exposed_pull_tokens = rate * exposed_pull_s;
    TradeCost {
        backlog,
        uncached: own,
        pull_s,
        exposed_pull_s,
        exposed_pull_tokens,
        copy_tokens,
        total: backlog + own + exposed_pull_tokens + copy_tokens,
        wait_s,
        service_s,
    }
}

/// One P candidate as seen by [`select_trade`].
#[derive(Debug, Clone, Copy)]
pub struct TradeInput<'a> {
    pub worker: &'a str,
    /// Pending uncached tokens on this P.
    pub backlog: u128,
    /// This P's own cached-prefix hit for the request (cache-aware owner key).
    pub local_hit: u64,
    /// u(P): uncached tokens of the request on this P after the hinted pull.
    pub uncached: u64,
    /// Tokens pulled from another P (0 = no pull).
    pub pulled: u64,
    /// `wait = e3`: this P's forecast terms. None: the ledger terms (wait =
    /// ledger, or a P without a usable forecast).
    pub wait: Option<TradeWaitTerms>,
    /// `wait = e3`: the error bound b (s) of this P's forecast at the bound
    /// level. None: no usable forecast (the P then has the ledger terms in
    /// `wait = None`) or another wait source.
    pub bound_s: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct TradeSelection {
    pub chosen: usize,
    /// The cache-aware owner: maximum own hit, then the lowest backlog, then
    /// the URL.
    pub owner: usize,
    pub has_owner: bool,
    /// Cost of every input, in input order.
    pub costs: Vec<TradeCost>,
    /// argmin cost (before the prediction gate).
    pub best: usize,
    /// `wait = e3`: the gate margin m * sqrt(b_o^2 + b_w^2) in seconds, when
    /// the gate was evaluated (an owner exists and the best is another P).
    pub margin_s: Option<f64>,
    /// (cost(owner) - cost(best)) / R in seconds, when the gate was evaluated.
    pub owner_gain_s: Option<f64>,
    /// The prediction gate kept the owner.
    pub margin_protected: bool,
    /// `wait = e3` only: why the owner was kept although another P had the
    /// lower cost (`gate`: the gain did not beat the bound margin;
    /// `owner_no_forecast`: the owner has no usable forecast). None for the
    /// ledger wait.
    pub protect_reason: Option<&'static str>,
}

/// The TRADE cost of every candidate, in the order of `inputs`.
fn trade_costs(params: &TradeParams, inputs: &[TradeInput<'_>]) -> Vec<TradeCost> {
    inputs
        .iter()
        .map(|input| {
            trade_cost_full(
                params,
                input.backlog,
                input.uncached,
                input.pulled,
                input.wait,
            )
        })
        .collect()
}

/// Choose argmin TRADE cost. Equal costs go to the cache-aware owner, then to
/// the higher own hit, then to the lower backlog, then to the smaller URL.
/// With `wait = e3` the selection is [`select_trade_e3`]. Pure: no I/O, no
/// reservation. Panics if `inputs` is empty.
pub fn select_trade(params: &TradeParams, inputs: &[TradeInput<'_>]) -> TradeSelection {
    if params.wait == TradeWait::E3 {
        return select_trade_e3(params, inputs);
    }
    assert!(!inputs.is_empty());
    let costs = trade_costs(params, inputs);
    let owner = inputs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            b.local_hit
                .cmp(&a.local_hit)
                .then_with(|| a.backlog.cmp(&b.backlog))
                .then_with(|| a.worker.cmp(b.worker))
        })
        .unwrap()
        .0;
    let best = inputs
        .iter()
        .enumerate()
        .min_by(|(ia, a), (ib, b)| {
            costs[*ia]
                .total
                .total_cmp(&costs[*ib].total)
                .then_with(|| (*ib == owner).cmp(&(*ia == owner)))
                .then_with(|| b.local_hit.cmp(&a.local_hit))
                .then_with(|| a.backlog.cmp(&b.backlog))
                .then_with(|| a.worker.cmp(b.worker))
        })
        .unwrap()
        .0;
    TradeSelection {
        chosen: best,
        owner,
        has_owner: inputs[owner].local_hit > 0,
        costs,
        best,
        margin_s: None,
        owner_gain_s: None,
        margin_protected: false,
        protect_reason: None,
    }
}

/// `--symmetric-pd-gwq-auto`: the inputs of the release tie-break of the global
/// waiting queue ([`trade_tie_break`]), one entry per [`TradeInput`].
#[derive(Debug, Clone, Copy)]
pub struct TradeTieBreak<'a> {
    /// The P can take a release now (the queue's lead rule; a stale view
    /// counts as room, as in the walk).
    pub room: &'a [bool],
    /// Outstanding requests on the P: the cost book's in-flight count
    /// (reserved, dispatched and unknown entries).
    pub outstanding: &'a [u64],
    /// The URL of the last tie-broken release (None: the rotation starts at
    /// the smallest URL).
    pub last: Option<&'a str>,
}

/// What [`trade_tie_break`] chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TradeTieOutcome {
    /// The chosen input.
    pub chosen: usize,
    /// Inputs whose TRADE total equals the minimum (1: no tie).
    pub tied: usize,
    /// The key that decided: `none` (no tie, or the prediction gate kept
    /// the owner), `room`, `local_hit`, `outstanding` or `rotation`.
    pub decided_by: &'static str,
}

/// `--symmetric-pd-gwq-auto`: the release tie-break of the global waiting queue.
///
/// [`select_trade`] breaks an exact tie of the TRADE total by the owner (for
/// a cold request the smallest backlog, then the smallest URL), the larger
/// own hit, the smaller backlog and the smaller URL. When several P are idle
/// at a release, every such release would go to the same P. Among the
/// inputs whose total equals the minimum bit for bit, this keeps (1) the P
/// with room, (2) the largest own hit, (3) the fewest outstanding requests,
/// then (4) rotates: the first URL after `last` in URL order (cyclic). A
/// choice of the prediction gate stays. Pure; the walk and the release
/// reservation call it with the same inputs, so they agree.
pub fn trade_tie_break(
    selection: &TradeSelection,
    inputs: &[TradeInput<'_>],
    tie: &TradeTieBreak<'_>,
) -> TradeTieOutcome {
    let unchanged = TradeTieOutcome {
        chosen: selection.chosen,
        tied: 1,
        decided_by: "none",
    };
    if selection.margin_protected
        || tie.room.len() != inputs.len()
        || tie.outstanding.len() != inputs.len()
    {
        return unchanged;
    }
    let min = selection.costs[selection.best].total;
    let mut set: Vec<usize> = (0..inputs.len())
        .filter(|&i| selection.costs[i].total.total_cmp(&min) == std::cmp::Ordering::Equal)
        .collect();
    let tied = set.len();
    if tied < 2 {
        return TradeTieOutcome { tied, ..unchanged };
    }
    let decided = |chosen: usize, decided_by: &'static str| TradeTieOutcome {
        chosen,
        tied,
        decided_by,
    };
    if set.iter().any(|&i| tie.room[i]) {
        set.retain(|&i| tie.room[i]);
        if set.len() == 1 {
            return decided(set[0], "room");
        }
    }
    let max_hit = set.iter().map(|&i| inputs[i].local_hit).max().unwrap_or(0);
    set.retain(|&i| inputs[i].local_hit == max_hit);
    if set.len() == 1 {
        return decided(set[0], "local_hit");
    }
    let fewest = set.iter().map(|&i| tie.outstanding[i]).min().unwrap_or(0);
    set.retain(|&i| tie.outstanding[i] == fewest);
    if set.len() == 1 {
        return decided(set[0], "outstanding");
    }
    set.sort_by(|&a, &b| inputs[a].worker.cmp(inputs[b].worker));
    let next = tie
        .last
        .and_then(|last| set.iter().copied().find(|&i| inputs[i].worker > last));
    decided(next.unwrap_or(set[0]), "rotation")
}

/// TRADE selection with `wait = e3`: argmin cost with the prediction gate.
///
/// A candidate is usable when it has a forecast: `wait` (B_w = `wait_s`,
/// S_w = `pred_s - wait_s`) and `bound_s` (b) are both set. Its cost is
/// [`trade_cost_full`] on these terms, that is
/// `C_w = R x (pred_s + max(0, pull_s - wait_s)) + copy`. A
/// candidate without a forecast has the ledger cost.
///
/// - Owner o: the maximum own hit; ties go to a usable P, then to the lower
///   cost, the lower backlog and the smaller URL.
/// - Best w*: argmin cost over the usable P (over all P when none is
///   usable); ties go to the owner, the higher own hit, the shorter wait,
///   the lower backlog and the smaller URL.
/// - No cached owner (own hit 0 on every P): w*.
/// - Otherwise the request leaves o for w* only when both are usable and
///   `gain - margin > 0`, with `gain = (C_o - C_w*) / R` and
///   `margin = m x sqrt(b_o^2 + b_w*^2)` (s; m = `gate_m`). Else it stays
///   on o: `protect_reason` is
///   `owner_no_forecast` (o is not usable) or `gate`. A NaN gain keeps o.
///
/// So a P without a usable forecast never wins a move: it can be chosen
/// only as the owner, or when no P is usable and no P caches the request.
/// `margin_s`, `owner_gain_s` and `margin_protected` hold the gate numbers
/// when the gate was evaluated. Pure; panics if `inputs` is empty.
pub fn select_trade_e3(params: &TradeParams, inputs: &[TradeInput<'_>]) -> TradeSelection {
    assert!(!inputs.is_empty());
    let costs = trade_costs(params, inputs);
    let usable = |i: usize| inputs[i].wait.is_some() && inputs[i].bound_s.is_some();
    let owner = (0..inputs.len())
        .min_by(|&a, &b| {
            inputs[b]
                .local_hit
                .cmp(&inputs[a].local_hit)
                .then_with(|| usable(b).cmp(&usable(a)))
                .then_with(|| costs[a].total.total_cmp(&costs[b].total))
                .then_with(|| inputs[a].backlog.cmp(&inputs[b].backlog))
                .then_with(|| inputs[a].worker.cmp(inputs[b].worker))
        })
        .expect("nonempty inputs");
    let any_usable = (0..inputs.len()).any(usable);
    let best = (0..inputs.len())
        .filter(|&i| !any_usable || usable(i))
        .min_by(|&a, &b| {
            costs[a]
                .total
                .total_cmp(&costs[b].total)
                .then_with(|| (b == owner).cmp(&(a == owner)))
                .then_with(|| inputs[b].local_hit.cmp(&inputs[a].local_hit))
                .then_with(|| costs[a].wait_s.total_cmp(&costs[b].wait_s))
                .then_with(|| inputs[a].backlog.cmp(&inputs[b].backlog))
                .then_with(|| inputs[a].worker.cmp(inputs[b].worker))
        })
        .expect("nonempty inputs");
    let has_owner = inputs[owner].local_hit > 0;
    let mut selection = TradeSelection {
        chosen: best,
        owner,
        has_owner,
        costs,
        best,
        margin_s: None,
        owner_gain_s: None,
        margin_protected: false,
        protect_reason: None,
    };
    if !has_owner || best == owner {
        return selection;
    }
    let (Some(b_o), Some(b_w)) = (inputs[owner].bound_s, inputs[best].bound_s) else {
        // The best P is usable whenever any P is, so this is the owner.
        selection.chosen = owner;
        selection.margin_protected = true;
        selection.protect_reason = Some("owner_no_forecast");
        return selection;
    };
    if !usable(owner) {
        selection.chosen = owner;
        selection.margin_protected = true;
        selection.protect_reason = Some("owner_no_forecast");
        return selection;
    }
    let gain =
        (selection.costs[owner].total - selection.costs[best].total) / params.rate_tokens_per_s;
    let margin = params.gate_m * (b_o * b_o + b_w * b_w).sqrt();
    selection.owner_gain_s = Some(gain);
    selection.margin_s = Some(margin);
    let moves = gain - margin > 0.0;
    if !moves {
        selection.chosen = owner;
        selection.margin_protected = true;
        selection.protect_reason = Some("gate");
    }
    selection
}

/// The session identity string of one agent trajectory for
/// [`TradeAutoState`]: `subagent_id`, a slash, then `agent_id` (one string
/// per pair: the number has no slash).
pub fn trade_session_key(agent_id: &str, subagent_id: u64) -> String {
    format!("{subagent_id}/{agent_id}")
}

/// The key of one session in [`TradeAutoState`]: a 64-bit hash of its
/// identity string ([`trade_session_key`]). Stable within one process.
pub fn trade_session_hash(session: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    session.hash(&mut hasher);
    hasher.finish()
}

/// The estimates of one P worker at one time ([`TradeAutoState::worker`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TradeAutoWorkerView {
    /// Requests dispatched to the P whose prefill has not ended (now).
    pub in_flight: u64,
    /// Busy fraction: the time with `in_flight` > 0 (EWMA, corrected).
    pub busy: f64,
    /// Time-averaged `in_flight` (EWMA, corrected).
    pub inflight: f64,
    /// Credited prefill tokens per second (EWMA, corrected).
    pub rate_tokens_per_s: f64,
}

/// A busy period of one P that has just ended: no request dispatched to the
/// P has a prefill that has not ended.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TradeAutoBusyPeriod {
    pub seconds: f64,
    /// Requests dispatched to the P during the period.
    pub requests: u64,
}

/// One P worker of [`TradeAutoState`].
#[derive(Debug, Clone)]
struct TradeAutoWorker {
    /// The start of this P's history (its EWMAs are 0 there).
    since: Instant,
    /// The EWMAs below are exact at this time.
    updated: Instant,
    /// Requests dispatched to this P whose prefill has not ended.
    in_flight: u64,
    /// EWMA of 1{in_flight > 0}.
    busy: f64,
    /// EWMA of in_flight.
    inflight: f64,
    /// EWMA rate of the credited prefill tokens (tokens per second): a
    /// credit of k tokens adds k / tau.
    rate: f64,
    /// The start of the current busy period and its dispatches.
    busy_since: Option<Instant>,
    period_requests: u64,
}

impl TradeAutoWorker {
    fn new(at: Instant) -> Self {
        Self {
            since: at,
            updated: at,
            in_flight: 0,
            busy: 0.0,
            inflight: 0.0,
            rate: 0.0,
            busy_since: None,
            period_requests: 0,
        }
    }

    /// The three EWMAs at `now`, without the start-up correction. The
    /// signals have not changed since `updated`, so the decay is exact:
    /// m(now) = x + (m - x) * exp(-(now - updated) / tau).
    fn at(&self, now: Instant, tau: f64) -> (f64, f64, f64) {
        let decay = (-now.saturating_duration_since(self.updated).as_secs_f64() / tau).exp();
        let busy = if self.in_flight > 0 { 1.0 } else { 0.0 };
        let n = self.in_flight as f64;
        (
            busy + (self.busy - busy) * decay,
            n + (self.inflight - n) * decay,
            self.rate * decay,
        )
    }

    /// Move the EWMAs to `now`, before `in_flight` changes or a credit.
    fn advance(&mut self, now: Instant, tau: f64) {
        (self.busy, self.inflight, self.rate) = self.at(now, tau);
        self.updated = self.updated.max(now);
    }

    /// The estimates at `now`: each EWMA divided by
    /// 1 - exp(-(now - since) / tau), the weight of the history since
    /// `since`. A short history thus gives its own average, not a value
    /// biased to 0. None while that weight is below 1e-9.
    fn view(&self, now: Instant, tau: f64) -> Option<TradeAutoWorkerView> {
        let age = now.saturating_duration_since(self.since).as_secs_f64();
        let weight = -(-age / tau).exp_m1();
        if weight.is_nan() || weight < 1e-9 {
            return None;
        }
        let (busy, inflight, rate) = self.at(now, tau);
        Some(TradeAutoWorkerView {
            in_flight: self.in_flight,
            busy: (busy / weight).clamp(0.0, 1.0),
            inflight: inflight / weight,
            rate_tokens_per_s: rate / weight,
        })
    }
}

/// One session of [`TradeAutoState`].
#[derive(Debug, Clone, Copy)]
struct TradeAutoSession {
    /// Requests of the session reserved on a P that have not left it.
    in_flight: u64,
    /// The time of the last event of the session.
    last_seen: Instant,
}

/// The router's measurements behind `--trade-copy-lambda auto`
/// and `--trade-rate-tokens-per-s auto`. Every input comes from
/// the router's own ledger (no engine push state):
///
/// - per P: `in_flight` = requests dispatched to the P whose prefill has not
///   ended. The caller reports each dispatch and each end (the engine shows
///   0 remaining work, the P response, a transport error, an unknown end).
///   Three continuous-time EWMAs with time constant tau follow the busy
///   indicator 1{in_flight > 0}, `in_flight`, and the computed prefill
///   tokens credited at each completed prefill (a rate);
/// - per session: a request on a P, or an event (reservation, prefill end)
///   in the last idle window, and no END from the client.
///
/// The cost book owns it under its lock and creates it at the first auto
/// decision. Every time comes from the caller, so the tests use synthetic
/// times.
#[derive(Debug)]
pub struct TradeAutoState {
    started: Instant,
    tau_s: f64,
    workers: BTreeMap<String, TradeAutoWorker>,
    sessions: HashMap<u64, TradeAutoSession>,
    /// (time, session) of each session event, oldest first: the expiry queue.
    touches: VecDeque<(Instant, u64)>,
    /// Prefill ends that credited tokens, and their tokens.
    pub credited: u64,
    pub credited_tokens: u64,
    /// Prefill ends without credit (failure, transport error, unknown end).
    pub uncredited: u64,
    /// New sessions not counted because [`TRADE_AUTO_MAX_SESSIONS`] was reached.
    pub sessions_dropped: u64,
}

impl TradeAutoState {
    /// An empty state whose history starts at `started`; tau in seconds.
    pub fn new(started: Instant, tau_s: f64) -> Self {
        Self {
            started,
            tau_s,
            workers: BTreeMap::new(),
            sessions: HashMap::new(),
            touches: VecDeque::new(),
            credited: 0,
            credited_tokens: 0,
            uncredited: 0,
            sessions_dropped: 0,
        }
    }

    /// tau in seconds, at least 1 ms (the CLI requires tau > 0).
    fn tau(&self) -> f64 {
        self.tau_s.max(1e-3)
    }

    /// The start of the history (the first auto decision).
    pub fn started(&self) -> Instant {
        self.started
    }

    /// Give each P of `pool` a history that starts at `now` if it has none.
    pub fn observe_pool<'a>(&mut self, pool: impl IntoIterator<Item = &'a str>, now: Instant) {
        for url in pool {
            self.workers
                .entry(url.to_string())
                .or_insert_with(|| TradeAutoWorker::new(now));
        }
    }

    /// One request was dispatched to `worker` at `now`.
    pub fn dispatched(&mut self, worker: &str, now: Instant) {
        let tau = self.tau();
        let w = self
            .workers
            .entry(worker.to_string())
            .or_insert_with(|| TradeAutoWorker::new(now));
        w.advance(now, tau);
        if w.in_flight == 0 {
            w.busy_since = Some(now);
            w.period_requests = 0;
        }
        w.in_flight += 1;
        w.period_requests += 1;
    }

    /// The prefill of one request dispatched to `worker` ended at `now`.
    /// `tokens` = its computed prefill tokens when the prefill completed (the
    /// rate credits them); None when it failed or its end is unknown.
    /// Returns the busy period when the P has no request left. A call
    /// without a matching dispatch changes nothing.
    pub fn prefill_done(
        &mut self,
        worker: &str,
        tokens: Option<u64>,
        now: Instant,
    ) -> Option<TradeAutoBusyPeriod> {
        let tau = self.tau();
        let w = self.workers.get_mut(worker)?;
        if w.in_flight == 0 {
            return None;
        }
        w.advance(now, tau);
        match tokens {
            Some(tokens) => {
                w.rate += tokens as f64 / tau;
                self.credited += 1;
                self.credited_tokens = self.credited_tokens.saturating_add(tokens);
            }
            None => self.uncredited += 1,
        }
        w.in_flight -= 1;
        if w.in_flight > 0 {
            return None;
        }
        let since = w.busy_since.take()?;
        Some(TradeAutoBusyPeriod {
            seconds: now.saturating_duration_since(since).as_secs_f64(),
            requests: std::mem::take(&mut w.period_requests),
        })
    }

    /// A request of `session` was reserved on a P at `now`.
    pub fn session_started(&mut self, session: u64, now: Instant) {
        if let Some(s) = self.sessions.get_mut(&session) {
            s.in_flight += 1;
            s.last_seen = s.last_seen.max(now);
        } else if self.sessions.len() >= TRADE_AUTO_MAX_SESSIONS {
            self.sessions_dropped += 1;
            return;
        } else {
            self.sessions.insert(
                session,
                TradeAutoSession {
                    in_flight: 1,
                    last_seen: now,
                },
            );
        }
        self.touches.push_back((now, session));
    }

    /// A request of `session` left its P at `now`: its prefill ended or
    /// failed, or its reservation was rolled back. A session that is not
    /// counted (ended, or above the limit) is ignored.
    pub fn session_finished(&mut self, session: u64, now: Instant) {
        if let Some(s) = self.sessions.get_mut(&session) {
            s.in_flight = s.in_flight.saturating_sub(1);
            s.last_seen = s.last_seen.max(now);
            self.touches.push_back((now, session));
        }
    }

    /// The client ended `session` (END): it no longer counts, also with a
    /// request on a P. Returns whether it was counted.
    pub fn session_end(&mut self, session: u64) -> bool {
        self.sessions.remove(&session).is_some()
    }

    /// N_active at `now`: the sessions with a request on a P or an event in
    /// the last `idle_s` seconds. The expired sessions are dropped.
    pub fn active_sessions(&mut self, now: Instant, idle_s: f64) -> usize {
        while let Some(&(at, session)) = self.touches.front() {
            if now.saturating_duration_since(at).as_secs_f64() <= idle_s {
                break;
            }
            self.touches.pop_front();
            if self
                .sessions
                .get(&session)
                .is_some_and(|s| s.in_flight == 0 && s.last_seen <= at)
            {
                self.sessions.remove(&session);
            }
        }
        self.sessions.len()
    }

    /// The inputs of one decision at `now` over the healthy P of `pool`
    /// (after [`Self::observe_pool`]); `idle_s` = the session idle window.
    /// busy and the in-flight mean are means over the P with history;
    /// R_hat = the sum of their token rates / the sum of their busy fractions.
    pub fn inputs(&mut self, now: Instant, pool: &[&str], idle_s: f64) -> TradeAutoInputs {
        let n_active = self.active_sessions(now, idle_s);
        let tau = self.tau();
        let (mut busy, mut inflight, mut rate, mut n) = (0.0, 0.0, 0.0, 0usize);
        for url in pool {
            if let Some(view) = self.workers.get(*url).and_then(|w| w.view(now, tau)) {
                busy += view.busy;
                inflight += view.inflight;
                rate += view.rate_tokens_per_s;
                n += 1;
            }
        }
        TradeAutoInputs {
            elapsed_s: now.saturating_duration_since(self.started).as_secs_f64(),
            n_p: pool.len(),
            n_active,
            busy: (n > 0).then_some(busy / n as f64),
            inflight: (n > 0).then_some(inflight / n as f64),
            r_hat: (busy > 1e-9 && rate > 0.0).then_some(rate / busy),
        }
    }

    /// The estimates of `worker` at `now`; None without history.
    pub fn worker(&self, worker: &str, now: Instant) -> Option<TradeAutoWorkerView> {
        self.workers
            .get(worker)
            .and_then(|w| w.view(now, self.tau()))
    }

    /// The URLs of the P with a history.
    pub fn worker_urls(&self) -> impl Iterator<Item = &str> {
        self.workers.keys().map(String::as_str)
    }

    /// The in-flight requests of `worker` now; None for an unknown P.
    pub fn in_flight(&self, worker: &str) -> Option<u64> {
        self.workers.get(worker).map(|w| w.in_flight)
    }

    /// The requests of `session` on a P now; None when it is not held.
    pub fn session_in_flight(&self, session: u64) -> Option<u64> {
        self.sessions.get(&session).map(|s| s.in_flight)
    }

    /// The sessions held now (counted or waiting for expiry).
    pub fn tracked_sessions(&self) -> usize {
        self.sessions.len()
    }
}

#[cfg(test)]
mod trade_tests {
    use super::*;

    fn params(lambda: f64) -> TradeParams {
        TradeParams {
            copy_lambda: lambda,
            ..TradeParams::default()
        }
    }

    fn input(
        worker: &str,
        backlog: u128,
        local_hit: u64,
        uncached: u64,
        pulled: u64,
    ) -> TradeInput<'_> {
        TradeInput {
            worker,
            backlog,
            local_hit,
            uncached,
            pulled,
            wait: None,
            bound_s: None,
        }
    }

    fn eft(input: TradeInput<'_>, wait_s: f64, service_s: f64) -> TradeInput<'_> {
        TradeInput {
            wait: Some(TradeWaitTerms { wait_s, service_s }),
            ..input
        }
    }

    /// A `wait = e3` candidate with a forecast: wait_s, pred_s and bound b.
    fn e3(input: TradeInput<'_>, wait_s: f64, pred_s: f64, bound_s: f64) -> TradeInput<'_> {
        TradeInput {
            wait: Some(TradeWaitTerms {
                wait_s,
                service_s: pred_s - wait_s,
            }),
            bound_s: Some(bound_s),
            ..input
        }
    }

    fn e3_params(lambda: f64, gate_m: f64) -> TradeParams {
        TradeParams {
            wait: TradeWait::E3,
            gate_m,
            ..params(lambda)
        }
    }

    /// The prediction gate on hand cases (R = 13,500, lambda = 0). Owner a
    /// caches 40,000 tokens: pred 2.0 s, bound 0.3 s. b is idle and pulls
    /// 40,000 tokens (pull 0.49 s, wait 0.1 s: 0.39 s exposed): pred 0.6 s,
    /// bound 0.2 s. Gain = 2.0 - (0.6 + 0.39) = 1.01 s; margin =
    /// sqrt(0.09 + 0.04) = 0.3606 s at m = 1: b wins. m = 3 (margin 1.08 s)
    /// keeps a (`gate`).
    #[test]
    fn e3_gate_moves_only_when_the_gain_beats_the_bound_margin() {
        let inputs = [
            e3(input("http://a", 0, 40_000, 2_000, 0), 1.5, 2.0, 0.3),
            e3(input("http://b", 0, 0, 2_000, 40_000), 0.1, 0.6, 0.2),
        ];
        let s = select_trade(&e3_params(0.0, 1.0), &inputs);
        assert_eq!((s.owner, s.best, s.chosen), (0, 1, 1));
        assert!(!s.margin_protected);
        assert_eq!(s.protect_reason, None);
        let gain = s.owner_gain_s.unwrap();
        assert!(close(gain, 2.0 - (0.6 + 0.39)), "{gain}");
        assert!(close(s.margin_s.unwrap(), 0.13f64.sqrt()));
        // C_w = R * (pred_s + exposed pull) with lambda 0.
        assert!(close(s.costs[1].total, 13_500.0 * (0.6 + 0.39)));
        assert!(close(s.costs[0].total, 13_500.0 * 2.0));
        assert!(close(s.costs[1].wait_s, 0.1) && close(s.costs[1].service_s, 0.5));
        let s = select_trade(&e3_params(0.0, 3.0), &inputs);
        assert_eq!((s.owner, s.best, s.chosen), (0, 1, 0));
        assert!(s.margin_protected);
        assert_eq!(s.protect_reason, Some("gate"));
        // The copy price is part of C: lambda 9.2 x 40,000 = 368,000 tokens
        // = 27.3 s of R; the owner keeps the request on cost alone.
        let s = select_trade(&e3_params(9.2, 1.0), &inputs);
        assert_eq!((s.best, s.chosen), (0, 0));
        assert_eq!(s.protect_reason, None);
        // m = 0: a move needs only a positive gain.
        let s = select_trade(&e3_params(0.0, 0.0), &inputs);
        assert_eq!(s.chosen, 1);
    }

    /// A P without a usable forecast (no wait or no bound) has the ledger
    /// cost and never wins a move: as best it is skipped while another P has
    /// a forecast; an owner without a forecast is kept (`owner_no_forecast`).
    #[test]
    fn e3_candidates_without_a_forecast_never_win_a_move() {
        let p = e3_params(0.0, 1.0);
        // b has no forecast and the lowest ledger cost (empty backlog).
        let inputs = [
            e3(input("http://a", 500_000, 40_000, 2_000, 0), 1.5, 2.0, 0.3),
            input("http://b", 0, 0, 2_000, 0),
            e3(input("http://c", 0, 0, 2_000, 0), 1.6, 1.9, 0.3),
        ];
        let s = select_trade(&p, &inputs);
        assert!(s.costs[1].total < s.costs[2].total);
        assert_eq!(s.best, 2, "the best P with a forecast");
        assert_eq!(s.chosen, 0, "gain 0.1 s does not beat the margin");
        assert_eq!(s.protect_reason, Some("gate"));
        // The owner has no forecast: it is kept, whatever the others cost.
        let inputs = [
            input("http://a", 900_000, 40_000, 2_000, 0),
            e3(input("http://c", 0, 0, 2_000, 0), 0.0, 0.2, 0.1),
        ];
        let s = select_trade(&p, &inputs);
        assert_eq!((s.owner, s.best, s.chosen), (0, 1, 0));
        assert_eq!(s.protect_reason, Some("owner_no_forecast"));
        assert_eq!(s.owner_gain_s, None);
        // A forecast without a bound is not usable either.
        let no_bound = TradeInput {
            bound_s: None,
            ..e3(input("http://c", 0, 0, 2_000, 0), 0.0, 0.2, 0.1)
        };
        let inputs = [
            e3(input("http://a", 0, 40_000, 2_000, 0), 1.5, 2.0, 0.3),
            no_bound,
        ];
        let s = select_trade(&p, &inputs);
        assert_eq!((s.best, s.chosen), (0, 0));
        // No P is usable: the ledger argmin among all, owner kept.
        let inputs = [
            input("http://a", 90_000, 40_000, 2_000, 0),
            input("http://b", 0, 0, 2_000, 0),
        ];
        let s = select_trade(&p, &inputs);
        assert_eq!((s.owner, s.best, s.chosen), (0, 1, 0));
        assert_eq!(s.protect_reason, Some("owner_no_forecast"));
    }

    /// Without a cached owner the request goes to the best forecast; ties of
    /// the own hit choose the usable owner with the lower cost.
    #[test]
    fn e3_cold_requests_take_the_best_forecast_and_owner_ties_prefer_forecasts() {
        let p = e3_params(0.0, 1.0);
        let cold = [
            e3(input("http://a", 0, 0, 9_000, 0), 1.0, 1.7, 0.4),
            input("http://b", 0, 0, 9_000, 0),
            e3(input("http://c", 0, 0, 9_000, 0), 0.2, 0.9, 0.4),
        ];
        let s = select_trade(&p, &cold);
        assert!(!s.has_owner);
        assert_eq!((s.best, s.chosen), (2, 2));
        assert_eq!(s.margin_s, None, "no gate without an owner");
        let tie = [
            input("http://a", 0, 16_384, 4_000, 0),
            e3(input("http://b", 0, 16_384, 4_000, 0), 0.9, 1.2, 0.2),
            e3(input("http://c", 0, 16_384, 4_000, 0), 0.5, 0.8, 0.2),
        ];
        let s = select_trade(&p, &tie);
        assert_eq!(s.owner, 2, "usable first, then the lower cost");
        assert_eq!((s.best, s.chosen), (2, 2));
        assert_eq!(s.protect_reason, None);
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    /// The GWQ release tie-break: select_trade sends a cold request on four
    /// idle P to the smallest URL every time; trade_tie_break keeps the
    /// exact ties and orders them by room, own hit, outstanding requests,
    /// then the rotation after the last URL (cyclic).
    #[test]
    fn gwq_tie_break_rotates_exact_ties() {
        let p = params(4.6);
        let urls = ["http://p1", "http://p2", "http://p3", "http://p4"];
        let cold: Vec<_> = urls.iter().map(|u| input(u, 0, 0, 5_000, 0)).collect();
        let s = select_trade(&p, &cold);
        assert_eq!(s.chosen, 0, "plain TRADE: the smallest URL");
        let room = [true; 4];
        let idle = [0u64; 4];
        let mut last: Option<&str> = None;
        let mut seen = Vec::new();
        for _ in 0..8 {
            let t = TradeTieBreak {
                room: &room,
                outstanding: &idle,
                last,
            };
            let o = trade_tie_break(&s, &cold, &t);
            assert_eq!((o.tied, o.decided_by), (4, "rotation"));
            last = Some(urls[o.chosen]);
            seen.push(o.chosen);
        }
        assert_eq!(
            seen,
            [0, 1, 2, 3, 0, 1, 2, 3],
            "round robin over the idle P"
        );
        // Room first, then the fewest outstanding requests.
        let room = [false, true, true, true];
        let out = [0u64, 3, 1, 1];
        let t = TradeTieBreak {
            room: &room,
            outstanding: &out,
            last: Some("http://p3"),
        };
        let o = trade_tie_break(&s, &cold, &t);
        assert_eq!(
            (o.chosen, o.decided_by),
            (3, "rotation"),
            "p3 and p4 have 1"
        );
        let out = [0u64, 3, 0, 1];
        let t = TradeTieBreak {
            room: &room,
            outstanding: &out,
            last: None,
        };
        assert_eq!(trade_tie_break(&s, &cold, &t).decided_by, "outstanding");
        assert_eq!(trade_tie_break(&s, &cold, &t).chosen, 2);
        let t = TradeTieBreak {
            room: &[false, true, false, false],
            outstanding: &out,
            last: None,
        };
        assert_eq!(
            (
                trade_tie_break(&s, &cold, &t).chosen,
                trade_tie_break(&s, &cold, &t).decided_by
            ),
            (1, "room")
        );
        // A strict argmin, or a larger own hit among exact ties, is kept.
        let warm = [
            input("http://p1", 0, 0, 5_000, 0),
            input("http://p2", 1_000, 1_000, 4_000, 0),
            input("http://p3", 0, 0, 6_000, 0),
        ];
        let s = select_trade(&p, &warm);
        let t = TradeTieBreak {
            room: &[true; 3],
            outstanding: &[5, 0, 0],
            last: None,
        };
        let o = trade_tie_break(&s, &warm, &t);
        assert_eq!((o.chosen, o.tied, o.decided_by), (1, 2, "local_hit"));
        let busy = [
            input("http://p1", 10, 0, 5_000, 0),
            input("http://p2", 0, 0, 5_000, 0),
        ];
        let s = select_trade(&p, &busy);
        let t = TradeTieBreak {
            room: &[true; 2],
            outstanding: &[0, 9],
            last: None,
        };
        let o = trade_tie_break(&s, &busy, &t);
        assert_eq!((o.chosen, o.tied, o.decided_by), (1, 1, "none"));
    }

    #[test]
    fn defaults_match_the_flag_defaults() {
        let p = TradeParams::default();
        assert_eq!(p.rate_tokens_per_s, 13_500.0);
        assert_eq!(p.pull_fixed_ms, 450.0);
        assert_eq!(p.pull_ms_per_100k, 100.0);
        assert_eq!(p.copy_lambda, 4.6);
        // 0.45 s + 0.10 s per 100k tokens; no pull takes no time.
        assert!(close(p.pull_seconds(100_000), 0.55));
        assert!(close(p.pull_seconds(50_000), 0.50));
        assert_eq!(p.pull_seconds(0), 0.0);
    }

    #[test]
    fn no_pull_cost_is_backlog_plus_own_uncached_only() {
        let c = trade_cost(&params(9.2), 40_000, 12_000, 0);
        assert_eq!(c.pull_s, 0.0);
        assert_eq!(c.exposed_pull_s, 0.0);
        assert_eq!(c.copy_tokens, 0.0);
        assert!(close(c.total, 52_000.0));
    }

    #[test]
    fn pull_on_an_idle_p_pays_the_whole_pull_time_and_the_copy() {
        // Idle P: nothing hides the pull. 100k pulled -> 0.55 s x 13,500 = 7,425.
        let c = trade_cost(&params(0.0), 0, 2_000, 100_000);
        assert!(close(c.exposed_pull_s, 0.55));
        assert!(close(c.exposed_pull_tokens, 7_425.0));
        assert!(close(c.total, 9_425.0));
        let c = trade_cost(&params(9.2), 0, 2_000, 100_000);
        assert!(close(c.copy_tokens, 920_000.0));
        assert!(close(c.total, 929_425.0));
    }

    #[test]
    fn backlog_hides_the_pull_partly_or_completely() {
        // Wait = 5,400 / 13,500 = 0.4 s < 0.55 s pull -> 0.15 s exposed.
        let c = trade_cost(&params(0.0), 5_400, 1_000, 100_000);
        assert!(close(c.exposed_pull_s, 0.15));
        assert!(close(c.total, 5_400.0 + 1_000.0 + 0.15 * 13_500.0));
        // Wait = 2 s >= 0.55 s -> fully hidden; cost = backlog + u.
        let c = trade_cost(&params(0.0), 27_000, 1_000, 100_000);
        assert_eq!(c.exposed_pull_s, 0.0);
        assert!(close(c.total, 28_000.0));
        // Equivalent completion form: R x (max(wait, pull) + u / R).
        for backlog in [0u128, 3_000, 7_425, 20_000] {
            let c = trade_cost(&params(0.0), backlog, 4_000, 100_000);
            let wait = backlog as f64 / 13_500.0;
            assert!(close(
                c.total,
                13_500.0 * (wait.max(0.55) + 4_000.0 / 13_500.0)
            ));
        }
    }

    #[test]
    fn floor_turns_a_small_gain_into_no_pull() {
        // Gain 511 < 512: no pull, the target recomputes.
        assert_eq!(trade_pull_tokens(10_000, 10_511, 50_000, 512), 0);
        // Gain 512 >= 512: pull.
        assert_eq!(trade_pull_tokens(10_000, 10_512, 50_000, 512), 512);
        // The source match is clamped to the request.
        assert_eq!(trade_pull_tokens(10_000, 90_000, 50_000, 512), 40_000);
        // No source better than the target.
        assert_eq!(trade_pull_tokens(10_000, 8_000, 50_000, 512), 0);
        // Floor 0 (gate disabled) or below 512 still uses 512.
        assert_eq!(trade_pull_tokens(10_000, 10_000, 50_000, 0), 0);
        assert_eq!(trade_pull_tokens(10_000, 10_001, 50_000, 0), 0);
        assert_eq!(trade_pull_tokens(10_000, 10_511, 50_000, 1), 0);
        assert_eq!(trade_pull_tokens(10_000, 10_512, 50_000, 0), 512);
        // A floor above 512 is kept.
        assert_eq!(trade_pull_tokens(10_000, 11_000, 50_000, 1_024), 0);
        assert_eq!(trade_pull_tokens(10_000, 11_024, 50_000, 1_024), 1_024);
    }

    /// Owner A holds 100k of a 110k prompt and has 60k backlog. B is idle and
    /// would pull the 100k. lambda = 0: B wins (7,425 + 10k < 60k + 10k).
    /// lambda = 9.2: the copy price keeps the request on A.
    #[test]
    fn lambda_zero_moves_and_large_lambda_keeps_the_owner() {
        let inputs = [
            input("http://a", 60_000, 100_000, 10_000, 0),
            input("http://b", 0, 0, 10_000, 100_000),
        ];
        let s = select_trade(&params(0.0), &inputs);
        assert_eq!(s.owner, 0);
        assert!(s.has_owner);
        assert_eq!(s.chosen, 1);
        assert!(close(s.costs[1].total, 17_425.0));
        let s = select_trade(&params(9.2), &inputs);
        assert_eq!(s.chosen, 0);
        assert!(close(s.costs[0].total, 70_000.0));
    }

    /// A cold request (no hit anywhere, no pull) goes to the least backlog.
    #[test]
    fn cold_request_goes_to_the_least_backlog_without_pull() {
        let inputs = [
            input("http://a", 30_000, 0, 8_000, 0),
            input("http://b", 10_000, 0, 8_000, 0),
            input("http://c", 20_000, 0, 8_000, 0),
        ];
        let s = select_trade(&params(9.2), &inputs);
        assert!(!s.has_owner);
        assert_eq!(s.chosen, 1);
    }

    /// A move without a pull (gain below the floor) pays no copy and no pull
    /// time: a lightly cached owner loses to an idle P when the backlog
    /// difference exceeds the extra recompute.
    #[test]
    fn move_without_pull_is_charged_recompute_only() {
        let inputs = [
            input("http://a", 50_000, 400, 9_600, 0),
            input("http://b", 0, 0, 10_000, 0),
        ];
        let s = select_trade(&params(9.2), &inputs);
        assert_eq!(s.owner, 0);
        assert_eq!(s.chosen, 1);
        assert!(close(s.costs[1].total, 10_000.0));
    }

    #[test]
    fn ties_go_to_the_owner_then_the_higher_hit_then_the_lower_backlog() {
        // Equal totals (20k): the owner (max hit) wins although it is listed last.
        let inputs = [
            input("http://a", 10_000, 0, 10_000, 0),
            input("http://b", 5_000, 5_000, 15_000, 0),
            input("http://c", 0, 20_000, 20_000, 0),
        ];
        let s = select_trade(&params(9.2), &inputs);
        assert_eq!(s.owner, 2);
        assert_eq!(s.chosen, 2);
        // Without the owner in the tie: higher own hit, then lower backlog.
        let inputs = [
            input("http://a", 10_000, 0, 10_000, 0),
            input("http://b", 5_000, 5_000, 15_000, 0),
            input("http://c", 0, 30_000, 30_000, 0),
        ];
        let s = select_trade(&params(9.2), &inputs);
        assert_eq!(s.owner, 2);
        assert_eq!(s.chosen, 1);
        let inputs = [
            input("http://a", 12_000, 0, 8_000, 0),
            input("http://b", 10_000, 0, 10_000, 0),
        ];
        let s = select_trade(&params(9.2), &inputs);
        assert_eq!(s.owner, 1, "no hit anywhere: owner = lowest backlog");
        assert_eq!(s.chosen, 1);
        // Full tie: the smaller URL.
        let inputs = [
            input("http://b", 10_000, 0, 10_000, 0),
            input("http://a", 10_000, 0, 10_000, 0),
        ];
        assert_eq!(select_trade(&params(9.2), &inputs).chosen, 1);
    }

    /// The TRADE cost and selection of the first version,
    /// copied here unchanged so that the defaults are pinned against it.
    mod first_version {
        use crate::scheduler::trade::{TradeInput, TradeParams};

        pub struct Cost {
            pub total: f64,
        }

        pub fn trade_cost(params: &TradeParams, backlog: u128, uncached: u64, pulled: u64) -> Cost {
            let backlog = backlog as f64;
            let rate = params.rate_tokens_per_s;
            let pull_s = params.pull_seconds(pulled);
            let exposed_pull_s = if pulled == 0 {
                0.0
            } else {
                (pull_s - backlog / rate).max(0.0)
            };
            let exposed_pull_tokens = rate * exposed_pull_s;
            let copy_tokens = params.copy_lambda * pulled as f64;
            Cost {
                total: backlog + uncached as f64 + exposed_pull_tokens + copy_tokens,
            }
        }

        /// (chosen, owner, has_owner, costs)
        pub fn select_trade(
            params: &TradeParams,
            inputs: &[TradeInput<'_>],
        ) -> (usize, usize, bool, Vec<Cost>) {
            let costs: Vec<Cost> = inputs
                .iter()
                .map(|i| trade_cost(params, i.backlog, i.uncached, i.pulled))
                .collect();
            let owner = inputs
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    b.local_hit
                        .cmp(&a.local_hit)
                        .then_with(|| a.backlog.cmp(&b.backlog))
                        .then_with(|| a.worker.cmp(b.worker))
                })
                .unwrap()
                .0;
            let chosen = inputs
                .iter()
                .enumerate()
                .min_by(|(ia, a), (ib, b)| {
                    costs[*ia]
                        .total
                        .total_cmp(&costs[*ib].total)
                        .then_with(|| (*ib == owner).cmp(&(*ia == owner)))
                        .then_with(|| b.local_hit.cmp(&a.local_hit))
                        .then_with(|| a.backlog.cmp(&b.backlog))
                        .then_with(|| a.worker.cmp(b.worker))
                })
                .unwrap()
                .0;
            (chosen, owner, inputs[owner].local_hit > 0, costs)
        }
    }

    /// wait = eft: R x B_w and R x S_w replace the ledger backlog and u; the
    /// pull is hidden behind B_w (not backlog / R).
    #[test]
    fn eft_terms_replace_the_ledger_wait_and_service() {
        let p = params(0.0);
        let w = TradeWaitTerms {
            wait_s: 0.2,
            service_s: 0.3,
        };
        // Ledger backlog 27,000 (2 s) would hide the 0.55 s pull; B_w = 0.2 s does not.
        let c = trade_cost_full(&p, 27_000, 1_000, 100_000, Some(w));
        assert!(close(c.backlog, 0.2 * 13_500.0));
        assert!(close(c.uncached, 0.3 * 13_500.0));
        assert!(close(c.exposed_pull_s, 0.35));
        assert!(close(c.total, 13_500.0 * (0.2 + 0.3 + 0.35)));
        assert_eq!(c.wait_s, 0.2);
        assert_eq!(c.service_s, 0.3);
        // A long B_w hides the pull entirely.
        let c = trade_cost_full(
            &p,
            0,
            1_000,
            100_000,
            Some(TradeWaitTerms {
                wait_s: 3.0,
                service_s: 0.3,
            }),
        );
        assert_eq!(c.exposed_pull_s, 0.0);
        // The owner looks idle on the ledger (0 backlog) but its engine
        // queue drains in 5 s; an idle peer that pulls wins.
        let inputs = [
            eft(input("http://a", 0, 18_000, 2_000, 0), 5.0, 0.08),
            eft(input("http://b", 0, 0, 2_000, 18_000), 0.0, 0.08),
        ];
        let s = select_trade(&p, &inputs);
        assert_eq!(s.owner, 0);
        assert_eq!(s.chosen, 1);
        assert_eq!(s.best, 1);
        assert!(!s.margin_protected);
        // The same inputs on the ledger keep the owner.
        let ledger: Vec<_> = inputs
            .iter()
            .map(|i| TradeInput { wait: None, ..*i })
            .collect();
        assert_eq!(select_trade(&p, &ledger).chosen, 0);
        // Mixed: a fallback P (ledger terms) next to an EFT P.
        let mixed = [
            inputs[0],
            TradeInput {
                wait: None,
                ..inputs[1]
            },
        ];
        let s = select_trade(&p, &mixed);
        assert!(close(
            s.costs[1].total,
            2_000.0 + 13_500.0 * p.pull_seconds(18_000)
        ));
        assert_eq!(s.chosen, 1);
    }

    fn auto_lambda() -> TradeParams {
        TradeParams {
            copy_lambda: 18.4,
            copy_lambda_mode: TradeLambdaMode::Auto {
                min: DEFAULT_TRADE_AUTO_LAMBDA_MIN,
                max: DEFAULT_TRADE_AUTO_LAMBDA_MAX,
            },
            ..TradeParams::default()
        }
    }

    /// The measurements of a loaded 3P pool after start-up: 90 sessions,
    /// busy 0.75, 9 in-flight requests per P, R_hat 15,000 tokens/s.
    fn loaded() -> TradeAutoInputs {
        TradeAutoInputs {
            elapsed_s: 600.0,
            n_p: 3,
            n_active: 90,
            busy: Some(0.75),
            inflight: Some(9.0),
            r_hat: Some(15_000.0),
        }
    }

    /// lambda = clamp(q * R / R_hat, min, max); rate auto makes R / R_hat = 1
    /// and sets R = R_hat. The selection then charges the resolved price.
    #[test]
    fn trade_auto_lambda_is_the_clamped_formula() {
        // sat: q = 90 / 3 x 0.75 = 22.5; lambda = 22.5 x 13,500 / 15,000 = 20.25.
        let (p, r) = auto_lambda().resolved(&loaded());
        assert_eq!((r.lambda_mode, r.rate_mode), ("auto", "fixed"));
        assert!(close(r.q.unwrap(), 22.5));
        assert!(close(r.lambda_raw.unwrap(), 20.25));
        assert!(close(p.copy_lambda, 20.25));
        assert_eq!(p.copy_lambda.to_bits(), r.copy_lambda_used.to_bits());
        assert_eq!(r.fallback_reason, None);
        assert_eq!(p.rate_tokens_per_s, 13_500.0);
        assert_eq!(r.rate_used, 13_500.0);
        assert_eq!(p.copy_lambda_mode, auto_lambda().copy_lambda_mode);
        // rate auto: R = R_hat = 15,000, so lambda = q = 22.5.
        let both = TradeParams {
            rate_mode: TradeRateMode::Auto,
            ..auto_lambda()
        };
        let (p, r) = both.resolved(&loaded());
        assert_eq!(r.rate_mode, "auto");
        assert_eq!(p.rate_tokens_per_s, 15_000.0);
        assert_eq!(r.rate_fallback_reason, None);
        assert!(close(p.copy_lambda, 22.5));
        // Clamps: min 4.6 (q = 30 x 0.05 = 1.5 -> raw 1.35), max (raw 20.25 > 20).
        let light = TradeAutoInputs {
            busy: Some(0.05),
            ..loaded()
        };
        let (p, r) = auto_lambda().resolved(&light);
        assert!(close(r.lambda_raw.unwrap(), 1.35));
        assert_eq!(p.copy_lambda, 4.6);
        assert_eq!(r.fallback_reason, None, "a clamp is not a fallback");
        let capped = TradeParams {
            copy_lambda_mode: TradeLambdaMode::Auto {
                min: 4.6,
                max: 20.0,
            },
            ..auto_lambda()
        };
        assert_eq!(capped.resolved(&loaded()).0.copy_lambda, 20.0);
        // An idle pool (busy 0) prices at the floor; it is not a fallback.
        let idle = TradeAutoInputs {
            busy: Some(0.0),
            ..loaded()
        };
        let (p, r) = auto_lambda().resolved(&idle);
        assert_eq!(
            (p.copy_lambda, r.lambda_raw, r.fallback_reason),
            (4.6, Some(0.0), None)
        );
        // The selection charges the resolved price: b pulls 100k at 20.25.
        let inputs = [
            input("http://a", 60_000, 100_000, 10_000, 0),
            input("http://b", 0, 0, 10_000, 100_000),
        ];
        let (p, _) = auto_lambda().resolved(&loaded());
        let s = select_trade(&p, &inputs);
        assert!(close(s.costs[1].copy_tokens, 2_025_000.0));
        assert!(close(s.costs[1].total, 2_025_000.0 + 7_425.0 + 10_000.0));
        assert_eq!(s.chosen, 0);
    }

    /// Start-up (less than tau seconds of history) and each missing input
    /// give the fixed `copy_lambda` (not clamped) and name the reason; rate
    /// auto falls back to `rate_tokens_per_s`.
    #[test]
    fn trade_auto_falls_back_at_startup_and_on_missing_inputs() {
        let sat = auto_lambda();
        let reason = |p: &TradeParams, i: TradeAutoInputs| {
            let (params, r) = p.resolved(&i);
            (params.copy_lambda, r.fallback_reason)
        };
        let early = TradeAutoInputs {
            elapsed_s: 119.9,
            ..loaded()
        };
        assert_eq!(reason(&sat, early), (18.4, Some("startup")));
        let (_, r) = sat.resolved(&early);
        assert!(close(r.q.unwrap(), 22.5), "q is still traced");
        assert!(r.lambda_raw.is_none());
        let at_tau = TradeAutoInputs {
            elapsed_s: 120.0,
            ..loaded()
        };
        assert_eq!(
            reason(&sat, at_tau).1,
            None,
            "tau seconds of history end the start-up"
        );
        let nan = TradeAutoInputs {
            elapsed_s: f64::NAN,
            ..loaded()
        };
        assert_eq!(reason(&sat, nan).1, Some("startup"));
        let no_sessions = TradeAutoInputs {
            n_active: 0,
            ..loaded()
        };
        assert_eq!(reason(&sat, no_sessions), (18.4, Some("no_sessions")));
        let no_history = TradeAutoInputs {
            busy: None,
            inflight: None,
            r_hat: None,
            ..loaded()
        };
        assert_eq!(reason(&sat, no_history), (18.4, Some("no_busy")));
        for r_hat in [
            None,
            Some(0.0),
            Some(-1.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
        ] {
            let i = TradeAutoInputs { r_hat, ..loaded() };
            assert_eq!(reason(&sat, i), (18.4, Some("no_rate")), "{r_hat:?}");
        }
        let huge = TradeAutoInputs {
            n_active: usize::MAX,
            busy: Some(f64::MAX),
            ..loaded()
        };
        assert_eq!(reason(&sat, huge), (18.4, Some("not_finite")));
        // Rate auto: R falls back at start-up and without R_hat.
        let both = TradeParams {
            rate_mode: TradeRateMode::Auto,
            ..sat
        };
        let (p, r) = both.resolved(&early);
        assert_eq!(
            (p.rate_tokens_per_s, r.rate_fallback_reason),
            (13_500.0, Some("startup"))
        );
        let (p, r) = both.resolved(&TradeAutoInputs {
            r_hat: None,
            ..loaded()
        });
        assert_eq!(
            (p.rate_tokens_per_s, r.rate_fallback_reason),
            (13_500.0, Some("no_rate"))
        );
        assert_eq!((p.copy_lambda, r.fallback_reason), (18.4, Some("no_rate")));
        // Fixed lambda with rate auto: the fixed price, never clamped.
        let rate_only = TradeParams {
            copy_lambda: 2.0,
            rate_mode: TradeRateMode::Auto,
            ..TradeParams::default()
        };
        let (p, r) = rate_only.resolved(&loaded());
        assert_eq!(
            (p.copy_lambda, r.fallback_reason, r.lambda_mode),
            (2.0, None, "fixed")
        );
        assert_eq!(p.rate_tokens_per_s, 15_000.0);
    }

    /// The per-P ledger: busy, in-flight and the credited token rate as
    /// continuous-time EWMAs with the start-up correction, the busy period
    /// at an empty P, and no change for an end without a dispatch.
    #[test]
    fn trade_auto_state_measures_busy_inflight_and_rate() {
        let t0 = Instant::now();
        let at = |s: f64| t0 + std::time::Duration::from_secs_f64(s);
        let e = |x: f64| (-x).exp();
        let tau = 120.0;
        let mut st = TradeAutoState::new(t0, tau);
        st.observe_pool(["http://a", "http://b"], t0);
        assert!(
            st.worker("http://a", t0).is_none(),
            "no history at its first instant"
        );
        // a: one request from 0 s to 60 s, 600k tokens credited at 60 s; b idle.
        st.dispatched("http://a", at(0.0));
        assert_eq!(st.worker("http://a", at(1.0)).unwrap().in_flight, 1);
        let period = st
            .prefill_done("http://a", Some(600_000), at(60.0))
            .unwrap();
        assert!(close(period.seconds, 60.0));
        assert_eq!(period.requests, 1);
        // At 120 s: weight 1 - e^-1, busy (e^-0.5 - e^-1) / weight,
        // rate 600k / tau x e^-0.5 / weight.
        let weight = 1.0 - e(1.0);
        let busy_a = (e(0.5) - e(1.0)) / weight;
        let rate_a = 600_000.0 / tau * e(0.5) / weight;
        let v = st.worker("http://a", at(120.0)).unwrap();
        assert_eq!(v.in_flight, 0);
        assert!(close(v.busy, busy_a), "{v:?}");
        assert!(close(v.inflight, busy_a), "{v:?}");
        assert!(close(v.rate_tokens_per_s, rate_a), "{v:?}");
        let b = st.worker("http://b", at(120.0)).unwrap();
        assert_eq!((b.busy, b.inflight, b.rate_tokens_per_s), (0.0, 0.0, 0.0));
        let inputs = st.inputs(at(120.0), &["http://a", "http://b"], 300.0);
        assert_eq!((inputs.n_p, inputs.n_active), (2, 0));
        assert!(close(inputs.elapsed_s, 120.0));
        assert!(close(inputs.busy.unwrap(), busy_a / 2.0));
        assert!(close(inputs.inflight.unwrap(), busy_a / 2.0));
        assert!(close(inputs.r_hat.unwrap(), rate_a / busy_a));
        // Two requests at once: in-flight 2, busy 1 (a constant signal is exact).
        st.observe_pool(["http://c"], at(200.0));
        st.dispatched("http://c", at(200.0));
        st.dispatched("http://c", at(200.0));
        let c = st.worker("http://c", at(260.0)).unwrap();
        assert_eq!(c.in_flight, 2);
        assert!(close(c.busy, 1.0) && close(c.inflight, 2.0), "{c:?}");
        // The first end leaves the P busy; the second closes the period.
        assert!(st.prefill_done("http://c", None, at(270.0)).is_none());
        let period = st.prefill_done("http://c", Some(1_000), at(280.0)).unwrap();
        assert!(close(period.seconds, 80.0));
        assert_eq!(period.requests, 2);
        assert_eq!(
            (st.credited, st.credited_tokens, st.uncredited),
            (2, 601_000, 1)
        );
        // An end without a dispatch, or on an unknown P, changes nothing.
        assert!(st.prefill_done("http://b", Some(5), at(300.0)).is_none());
        assert!(st.prefill_done("http://z", Some(5), at(300.0)).is_none());
        assert_eq!((st.credited, st.uncredited), (2, 1));
        assert_eq!(
            st.worker("http://b", at(300.0)).unwrap().rate_tokens_per_s,
            0.0
        );
        assert_eq!(
            st.worker_urls().collect::<Vec<_>>(),
            ["http://a", "http://b", "http://c"]
        );
        // Steady state: busy 3 s of every 6 s with 30k tokens per period
        // (10,000 per busy second) for 20 tau: busy -> 0.5, R_hat -> 10,000
        // (the EWMA ripple at the measured phase is about 1 %).
        let mut st = TradeAutoState::new(t0, tau);
        st.observe_pool(["http://p"], t0);
        let mut t = 0.0;
        while t < 20.0 * tau {
            st.dispatched("http://p", at(t));
            st.prefill_done("http://p", Some(30_000), at(t + 3.0))
                .unwrap();
            t += 6.0;
        }
        let inputs = st.inputs(at(t), &["http://p"], 300.0);
        let busy = inputs.busy.unwrap();
        let r_hat = inputs.r_hat.unwrap();
        assert!((busy - 0.5).abs() < 0.01, "{busy}");
        assert!((r_hat / 10_000.0 - 1.0).abs() < 0.02, "{r_hat}");
        assert!((inputs.inflight.unwrap() - 0.5).abs() < 0.01);
    }

    /// N_active: a session counts with a request on a P, or within the idle
    /// window after its last event; END removes it at once; a request of an
    /// ended session never underflows; the session limit holds.
    #[test]
    fn trade_auto_sessions_count_in_flight_recent_and_not_ended() {
        let t0 = Instant::now();
        let at = |s: f64| t0 + std::time::Duration::from_secs_f64(s);
        assert_eq!(trade_session_key("agent-1", 0), "0/agent-1");
        assert_ne!(trade_session_key("a", 1), trade_session_key("a", 0));
        let [s1, s2, s3] =
            ["agent-1", "agent-2", "agent-3"].map(|a| trade_session_hash(&trade_session_key(a, 0)));
        assert!(s1 != s2 && s2 != s3 && s1 != s3);
        assert_eq!(
            s1,
            trade_session_hash("0/agent-1"),
            "stable within the process"
        );
        let mut st = TradeAutoState::new(t0, 120.0);
        let idle = 300.0;
        st.session_started(s1, at(0.0));
        st.session_started(s2, at(10.0));
        assert_eq!(st.active_sessions(at(10.0), idle), 2);
        st.session_finished(s1, at(20.0));
        st.session_finished(s2, at(30.0));
        assert_eq!(
            st.active_sessions(at(320.0), idle),
            2,
            "exactly 300 s idle still counts"
        );
        assert_eq!(st.active_sessions(at(321.0), idle), 1, "s1 idle for 301 s");
        // A request on a P never expires.
        st.session_started(s3, at(330.0));
        assert_eq!(st.active_sessions(at(1_000.0), idle), 1);
        st.session_finished(s3, at(1_000.0));
        assert_eq!(st.active_sessions(at(1_299.0), idle), 1);
        assert_eq!(st.active_sessions(at(1_301.0), idle), 0);
        assert_eq!(st.tracked_sessions(), 0);
        // END removes a session at once, also with a request on a P; its
        // later prefill end is ignored.
        st.session_started(s1, at(2_000.0));
        assert_eq!(st.active_sessions(at(2_000.0), idle), 1);
        assert!(st.session_end(s1));
        assert!(!st.session_end(s1));
        assert_eq!(st.active_sessions(at(2_000.0), idle), 0);
        st.session_finished(s1, at(2_001.0));
        assert_eq!(st.tracked_sessions(), 0);
        // Two requests of one session count once; it idles after both left.
        st.session_started(s2, at(3_000.0));
        st.session_started(s2, at(3_001.0));
        assert_eq!(st.active_sessions(at(3_001.0), idle), 1);
        st.session_finished(s2, at(3_002.0));
        assert_eq!(
            st.active_sessions(at(3_500.0), idle),
            1,
            "one request still on a P"
        );
        st.session_finished(s2, at(3_600.0));
        assert_eq!(st.active_sessions(at(3_899.0), idle), 1);
        assert_eq!(st.active_sessions(at(3_901.0), idle), 0);
        // The limit: a new session above it is not counted.
        let mut st = TradeAutoState::new(t0, 120.0);
        for i in 0..TRADE_AUTO_MAX_SESSIONS as u64 {
            st.session_started(i, at(0.0));
        }
        st.session_started(u64::MAX, at(1.0));
        st.session_finished(u64::MAX, at(2.0));
        assert_eq!(st.active_sessions(at(2.0), idle), TRADE_AUTO_MAX_SESSIONS);
        assert_eq!(st.sessions_dropped, 1);
        st.session_started(7, at(3.0));
        assert_eq!(st.sessions_dropped, 1, "a counted session is not new");
    }

    /// From the ledger to the price: 3 P, each busy 3 s of every 4 s with
    /// 10,000 tokens per busy second, and 90 sessions. After 20 tau,
    /// q = 90 / 3 x 0.75 and lambda = q x 13,500 / 10,000 = 30.4 (the EWMA
    /// ripple at the measured phase is below 2 %).
    #[test]
    fn trade_auto_state_feeds_the_resolved_price() {
        let t0 = Instant::now();
        let at = |s: f64| t0 + std::time::Duration::from_secs_f64(s);
        let pool = ["http://a", "http://b", "http://c"];
        let mut st = TradeAutoState::new(t0, 120.0);
        st.observe_pool(pool, t0);
        let mut t = 0.0;
        while t < 2_400.0 {
            for p in pool {
                st.dispatched(p, at(t));
                st.prefill_done(p, Some(30_000), at(t + 3.0));
            }
            t += 4.0;
        }
        for i in 0..90u64 {
            st.session_started(i, at(t - 10.0));
            st.session_finished(i, at(t - 5.0));
        }
        let inputs = st.inputs(at(t), &pool, 300.0);
        assert_eq!((inputs.n_p, inputs.n_active), (3, 90));
        let (p, r) = auto_lambda().resolved(&inputs);
        assert_eq!(r.fallback_reason, None);
        assert!((r.q.unwrap() / 22.5 - 1.0).abs() < 0.01, "{r:?}");
        assert!((p.copy_lambda / 30.375 - 1.0).abs() < 0.03, "{r:?}");
        // Before tau seconds of history the fixed price applies.
        let mut fresh = TradeAutoState::new(at(t), 120.0);
        fresh.observe_pool(pool, at(t));
        let early = fresh.inputs(at(t + 60.0), &pool, 300.0);
        let (p, r) = auto_lambda().resolved(&early);
        assert_eq!((p.copy_lambda, r.fallback_reason), (18.4, Some("startup")));
    }

    /// Fixed copy price and fixed rate: `resolved` returns the parameters
    /// unchanged (bit for bit) for any measurements, so selection and costs
    /// equal the first version; the defaults are fixed and is_original.
    #[test]
    fn trade_auto_fixed_mode_resolves_to_the_same_params_bit_for_bit() {
        let d = TradeParams::default();
        assert_eq!(d.copy_lambda_mode, TradeLambdaMode::Fixed);
        assert_eq!(d.rate_mode, TradeRateMode::Fixed);
        assert_eq!(d.auto_tau_s, 120.0);
        assert_eq!(d.auto_session_idle_s, 300.0);
        assert!(!d.is_auto() && d.is_original());
        assert!(TradeParams {
            auto_tau_s: 5.0,
            auto_session_idle_s: 1.0,
            ..d
        }
        .is_original());
        let lam = auto_lambda();
        assert!(lam.is_auto() && !lam.is_original());
        let rate = TradeParams {
            rate_mode: TradeRateMode::Auto,
            ..d
        };
        assert!(rate.is_auto() && !rate.is_original());
        let measurements = [
            TradeAutoInputs::default(),
            loaded(),
            TradeAutoInputs {
                elapsed_s: 1.0,
                ..loaded()
            },
            TradeAutoInputs {
                busy: None,
                inflight: None,
                r_hat: None,
                ..loaded()
            },
            TradeAutoInputs {
                r_hat: Some(f64::NAN),
                ..loaded()
            },
        ];
        let mut seed = 0x51f1_5eed_0bad_cafe_u64;
        let mut next = |m: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % m
        };
        let workers = ["http://a", "http://b", "http://c", "http://d"];
        for round in 0..2_000 {
            let p = TradeParams {
                copy_lambda: [0.0, 9.2, 18.4][next(3) as usize],
                rate_tokens_per_s: [13_500.0, 20_000.0][next(2) as usize],
                wait: TradeWait::Ledger,
                auto_tau_s: [1.0, 120.0][next(2) as usize],
                ..d
            };
            let (r, res) = p.resolved(&measurements[next(5) as usize]);
            assert_eq!(r, p, "round {round}");
            assert_eq!(r.copy_lambda.to_bits(), p.copy_lambda.to_bits());
            assert_eq!(r.rate_tokens_per_s.to_bits(), p.rate_tokens_per_s.to_bits());
            assert_eq!((res.lambda_mode, res.rate_mode), ("fixed", "fixed"));
            assert_eq!(res.copy_lambda_used.to_bits(), p.copy_lambda.to_bits());
            assert_eq!(
                (res.fallback_reason, res.rate_fallback_reason),
                (None, None)
            );
            assert!(res.lambda_raw.is_none());
            let n = 1 + next(4) as usize;
            let inputs: Vec<TradeInput<'_>> = (0..n)
                .map(|i| TradeInput {
                    worker: workers[i],
                    backlog: u128::from(next(4) * 10_000),
                    local_hit: next(3) * 8_192,
                    uncached: 1 + next(3) * 4_000,
                    pulled: [0, 0, 512, 18_000][next(4) as usize],
                    wait: (next(2) == 1).then_some(TradeWaitTerms {
                        wait_s: next(5) as f64 * 0.75,
                        service_s: 0.05 + next(4) as f64 * 0.25,
                    }),
                    bound_s: None,
                })
                .collect();
            let (a, b) = (select_trade(&r, &inputs), select_trade(&p, &inputs));
            assert_eq!((a.chosen, a.owner, a.best), (b.chosen, b.owner, b.best));
            for (x, y) in a.costs.iter().zip(&b.costs) {
                assert_eq!(x.total.to_bits(), y.total.to_bits(), "round {round}");
                assert_eq!(
                    x.copy_tokens.to_bits(),
                    y.copy_tokens.to_bits(),
                    "round {round}"
                );
            }
            if p.is_original() {
                let (chosen, _, _, costs) = first_version::select_trade(&r, &inputs);
                if inputs.iter().all(|i| i.wait.is_none()) {
                    assert_eq!(a.chosen, chosen, "round {round}");
                    for (x, y) in a.costs.iter().zip(&costs) {
                        assert_eq!(x.total.to_bits(), y.total.to_bits(), "round {round}");
                    }
                }
            }
        }
    }
}
