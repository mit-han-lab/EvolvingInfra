//! Shared uncached-work selection used by legacy CoScheduler and symmetric
//! scheduling. Keep-owner, dispatch-after spread and tie-breaks are identical;
//! the two runtimes may obtain their workload observations differently.
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::Instant;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CoScheduleConfig {
    pub owner_threshold_tokens: u64,
    pub detour_margin_tokens: u64,
}
impl Default for CoScheduleConfig {
    /// Create the default cached-owner protection settings.
    ///
    /// # Inputs
    /// None.
    ///
    /// # Outputs
    /// A configuration with a 12,288-token owner threshold and zero detour margin.
    fn default() -> Self {
        Self {
            owner_threshold_tokens: 12_288,
            detour_margin_tokens: 0,
        }
    }
}

pub struct Evaluation<'a> {
    pub worker: &'a str,
    pub local_hit: u64,
    pub gpu_hit: u64,
    pub before: u128,
    pub after: u128,
    pub spread: u128,
    pub active_requests: u64,
    pub pending_uncached_before: u128,
}
pub struct Selection {
    pub chosen: usize,
    pub owner: usize,
    pub has_owner: bool,
    pub keep_owner: bool,
    pub margin_protected: bool,
    pub saved_work: u128,
}

/// Pure selection; costs have already been evaluated by the caller. `margin`
/// is in the same work units as `after` (plain tokens unless restoration is
/// explicitly enabled by the legacy caller).
/// Choose a prefill destination while protecting a lightly loaded cache owner.
///
/// # Inputs
/// `evaluations`: nonempty candidate costs and hits; `threshold`: owner backlog limit in raw
/// uncached tokens; `margin`: minimum saving in the caller's cost units; `minimize_after`:
/// minimize destination cost instead of cross-worker spread.
///
/// # Outputs
/// Candidate indices and owner-protection diagnostics. The function performs no I/O or
/// reservation and panics if `evaluations` is empty.
pub fn select(
    evaluations: &[Evaluation<'_>],
    threshold: u64,
    margin: u128,
    minimize_after: bool,
) -> Selection {
    assert!(!evaluations.is_empty());
    let owner = evaluations
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            b.local_hit
                .cmp(&a.local_hit)
                .then_with(|| b.gpu_hit.cmp(&a.gpu_hit))
                .then_with(|| a.before.cmp(&b.before))
                .then_with(|| a.active_requests.cmp(&b.active_requests))
                .then_with(|| a.worker.cmp(b.worker))
        })
        .unwrap()
        .0;
    let has_owner = evaluations[owner].local_hit > 0;
    let keep_owner =
        has_owner && evaluations[owner].pending_uncached_before <= u128::from(threshold);
    let mut chosen = if keep_owner {
        owner
    } else {
        evaluations
            .iter()
            .enumerate()
            .min_by(|(ia, a), (ib, b)| {
                (if minimize_after {
                    a.after.cmp(&b.after)
                } else {
                    a.spread.cmp(&b.spread)
                })
                .then_with(|| (*ib == owner).cmp(&(*ia == owner)))
                .then_with(|| b.gpu_hit.cmp(&a.gpu_hit))
                .then_with(|| b.local_hit.cmp(&a.local_hit))
                .then_with(|| a.before.cmp(&b.before))
                .then_with(|| a.active_requests.cmp(&b.active_requests))
                .then_with(|| a.worker.cmp(b.worker))
            })
            .unwrap()
            .0
    };
    let saved_work = evaluations[owner]
        .after
        .saturating_sub(evaluations[chosen].after);
    let margin_protected = has_owner && chosen != owner && margin > 0 && saved_work <= margin;
    if margin_protected {
        chosen = owner;
    }
    Selection {
        chosen,
        owner,
        has_owner,
        keep_owner,
        margin_protected,
        saved_work,
    }
}

// ---------------------------------------------------------------------------
// TRADE: charge KV transfer and copy cost against the waiting a move removes.
// ---------------------------------------------------------------------------

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
    /// backlog / R and u / R from the router's uncached ledger (the first
    /// TRADE version; selection and trace are unchanged).
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

/// What the copy price lambda charges (`--trade-dup-charge`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TradeDupCharge {
    /// lambda x pulled tokens (the first TRADE version).
    #[default]
    Pulled,
    /// lambda x (pulled tokens + the tokens the P recomputes although
    /// another P already caches them): a move without a pull also makes a
    /// second copy.
    All,
}

impl TradeDupCharge {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pulled => "pulled",
            Self::All => "all",
        }
    }
}

/// How TRADE prices the copy that a pull leaves on its target P
/// (`--trade-pull-price`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TradePullPrice {
    /// lambda x pulled tokens, the same lambda for every candidate
    /// ([`TradeParams::copy_lambda`], fixed or from the copy price auto).
    #[default]
    Lambda,
    /// lambda_P x pulled tokens for the target P:
    /// lambda_P = lambda_q x phi_P. lambda_q = mu x q x R / R_hat of the
    /// copy price auto before the clamp (q with mu = 1 and rate auto; the
    /// fallback lambda while the auto price falls back). phi_P = the live
    /// share of the host tokens that P's retention eviction freed in the
    /// last tau seconds ([`TradePhiState`]). A P without host_room data
    /// pays lambda_q (reason `no_host_room`).
    Phi,
}

impl TradePullPrice {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lambda => "lambda",
            Self::Phi => "phi",
        }
    }
}

/// Default age limit of an engine snapshot for `--trade-wait eft`
/// (the view-age upper bound: poll sent to decision plus the engine's age).
pub const DEFAULT_TRADE_EFT_MAX_VIEW_AGE_S: f64 = 1.5;

/// Default window W of the arrival rate a_w of the HOL term, in seconds
/// (`--trade-hol-rate-window-secs`).
pub const DEFAULT_TRADE_HOL_RATE_WINDOW_S: f64 = 60.0;

/// Default floor of the auto copy price (`--trade-copy-lambda-min`),
/// in uncached-token equivalents per pulled token. Safety only: 0.5 x the S6
/// price 9.2.
pub const DEFAULT_TRADE_AUTO_LAMBDA_MIN: f64 = 4.6;
/// Default cap of the auto copy price (`--trade-copy-lambda-max`).
pub const DEFAULT_TRADE_AUTO_LAMBDA_MAX: f64 = 1000.0;
/// Default mu_max of the auto copy price (`--trade-copy-lambda-mu`):
/// recomputed tokens per pulled token, at its upper bound 1.
pub const DEFAULT_TRADE_AUTO_MU: f64 = 1.0;
/// Default time constant tau of the auto inputs, in seconds
/// (`--trade-auto-tau-secs`). It is also the start-up period.
pub const DEFAULT_TRADE_AUTO_TAU_S: f64 = 120.0;
/// Default idle window of a session in N_active, in seconds
/// (`--trade-auto-session-idle-secs`).
pub const DEFAULT_TRADE_AUTO_SESSION_IDLE_S: f64 = 300.0;
/// [`TradeAutoState`] tracks at most this many sessions. A new session above
/// the limit is not counted (and counted in `sessions_dropped`).
pub const TRADE_AUTO_MAX_SESSIONS: usize = 100_000;

/// The queue multiplier q of the auto copy price
/// (`--trade-copy-lambda-q`): the E2E seconds that one extra
/// second of P work costs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TradeAutoQ {
    /// q = N_active / n_P x busy: sessions per P times the P busy fraction,
    /// the closed-loop upper bound of q.
    #[default]
    Sat,
    /// q = 1 + the mean number of in-flight requests per P (router ledger).
    /// Lower than `sat`.
    Inflight,
}

impl TradeAutoQ {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sat => "sat",
            Self::Inflight => "inflight",
        }
    }
}

/// Where TRADE takes its copy price lambda from
/// (`--trade-copy-lambda <x|auto>`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TradeLambdaMode {
    /// lambda = [`TradeParams::copy_lambda`] (the first version).
    #[default]
    Fixed,
    /// lambda = clamp(mu * q * R / R_hat, min, max) from the router's own
    /// measurements ([`TradeParams::resolved`]). `copy_lambda` is the price
    /// during start-up and when an input is missing.
    Auto {
        q: TradeAutoQ,
        mu: f64,
        min: f64,
        max: f64,
    },
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
/// prediction gate (`--trade-gate-m`). With m = 1 the gate is
/// the reroute rule of the E3 bound fit: the bounds come from the
/// shipped bound table at its level, so m = 1 uses them as fitted.
pub const DEFAULT_TRADE_GATE_M: f64 = 1.0;

/// Parameters of the TRADE P-selection cost model (`--co-schedule-trade*`).
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
    /// Leave out of the backlog the requests routed to a P whose first
    /// prefill chunk has not launched yet. The engine's prefill-progress
    /// snapshot lists a request only after that launch, so this leaves out
    /// requests in transit, in the bootstrap queue AND in the waiting queue,
    /// which is most of a loaded P's wait. Off (default): the backlog is the
    /// full uncached ledger, the same as the owner-threshold path.
    pub exclude_unlaunched: bool,
    /// Source of the wait and service terms (default: the ledger).
    pub wait: TradeWait,
    /// What lambda charges (default: pulled tokens only).
    pub dup_charge: TradeDupCharge,
    /// k >= 0 (0 = off): leave the cache-aware owner only if
    /// (cost(owner) - cost(best)) / R > k * sqrt(B_o^2 + B_best^2), with
    /// B the wait terms in seconds. Prediction noise grows with the wait,
    /// so a small predicted gain behind long waits does not move a request.
    pub margin_noise_k: f64,
    /// `wait = eft`: a snapshot whose view age exceeds this (s) is not used.
    pub eft_max_view_age_s: f64,
    /// mu >= 0 (0 = off): weight of the head-of-line (HOL) term. The P
    /// queue is FCFS, so a placement on P delays the requests that arrive at
    /// P later; the cost adds mu * R * a_w * S * (B + S / 2) tokens
    /// ([`trade_hol_tokens`]). That term is a light-load estimate, so mu is
    /// an empirical weight, not a calibrated one.
    pub hol_weight: f64,
    /// W > 0: the window (s) of a_w, the TRADE placements per second on a P
    /// ([`TradeArrivals`]). No effect while `hol_weight` is 0.
    pub hol_rate_window_s: f64,
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
    /// `Lambda` (default): one copy price for every candidate. `Phi`: the
    /// copy price of each candidate is lambda_q x phi of its P (the cost
    /// book sets [`TradeInput::copy_lambda`]; symmetric_pd cost function only).
    pub pull_price: TradePullPrice,
    /// `wait = e3`: m >= 0, the multiple of the combined error bound
    /// sqrt(b_o^2 + b_w^2) in the prediction gate of [`select_trade_e3`].
    /// No effect with the other wait sources.
    pub gate_m: f64,
    /// `wait = e3`: g_min >= 0 (s), the bound-adjusted gain that a move
    /// must exceed. No effect with the other wait sources.
    pub gate_gmin_s: f64,
}

impl Default for TradeParams {
    /// The CLI defaults of the TRADE flags.
    fn default() -> Self {
        Self {
            rate_tokens_per_s: DEFAULT_TRADE_RATE_TOKENS_PER_S,
            pull_fixed_ms: DEFAULT_TRADE_PULL_FIXED_MS,
            pull_ms_per_100k: DEFAULT_TRADE_PULL_MS_PER_100K,
            copy_lambda: DEFAULT_TRADE_COPY_LAMBDA,
            exclude_unlaunched: false,
            wait: TradeWait::Ledger,
            dup_charge: TradeDupCharge::Pulled,
            margin_noise_k: 0.0,
            eft_max_view_age_s: DEFAULT_TRADE_EFT_MAX_VIEW_AGE_S,
            hol_weight: 0.0,
            hol_rate_window_s: DEFAULT_TRADE_HOL_RATE_WINDOW_S,
            copy_lambda_mode: TradeLambdaMode::Fixed,
            rate_mode: TradeRateMode::Fixed,
            auto_tau_s: DEFAULT_TRADE_AUTO_TAU_S,
            auto_session_idle_s: DEFAULT_TRADE_AUTO_SESSION_IDLE_S,
            pull_price: TradePullPrice::Lambda,
            gate_m: DEFAULT_TRADE_GATE_M,
            gate_gmin_s: 0.0,
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
    /// q of `mode`: N_active / n_P x busy (`sat`), or 1 + the mean in-flight
    /// requests per P (`inflight`). None without the input.
    pub fn q(&self, mode: TradeAutoQ) -> Option<f64> {
        match mode {
            TradeAutoQ::Sat => match self.busy {
                Some(busy) if self.n_p > 0 => Some(self.n_active as f64 / self.n_p as f64 * busy),
                _ => None,
            },
            TradeAutoQ::Inflight => self.inflight.map(|mean| 1.0 + mean),
        }
    }
}

/// How [`TradeParams::resolved`] set the copy price and the rate of one
/// decision (the trace keys of an auto decision).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TradeAutoResolution {
    pub lambda_mode: &'static str,
    pub q_mode: &'static str,
    /// q of the configured estimator (`sat` when lambda is fixed).
    pub q: Option<f64>,
    /// mu * q * R / R_hat before the clamp; None on a fallback.
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

    /// True when wait, dup charge, noise margin and HOL weight are at their
    /// defaults and the copy price and the rate are fixed (the HOL window,
    /// tau and the session idle window do not count). Selection and the
    /// trace are then those of the first TRADE version (no new trace keys).
    pub fn is_original(&self) -> bool {
        self.wait == TradeWait::Ledger
            && self.dup_charge == TradeDupCharge::Pulled
            && self.margin_noise_k == 0.0
            && self.hol_weight == 0.0
            && !self.is_auto()
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
            TradeLambdaMode::Auto { q, mu, min, max } => format!(
                "lambda = clamp(mu*q*R/R_hat, min, max) with q {} ({}), mu = {mu}, \
                 min = {min}, max = {max}, fallback lambda = {}",
                q.as_str(),
                match q {
                    TradeAutoQ::Sat => "N_active/n_P*busy",
                    TradeAutoQ::Inflight => "1 + mean in-flight requests per P",
                },
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
    /// q      = N_active / n_P * busy (sat),  or  1 + mean in-flight requests per P (inflight)
    /// lambda = clamp(mu * q * R / R_hat, min, max)    (lambda auto; R / R_hat = 1 with rate auto)
    /// ```
    ///
    /// lambda is in tokens of P work per pulled token: one pulled token
    /// causes at most mu recomputed tokens (mu / R_hat seconds of P work),
    /// and each second of P work delays the sessions on that P by q seconds
    /// in total; R converts these seconds into the cost's token units.
    ///
    /// Fallback: lambda = `copy_lambda` (not clamped), with the first reason
    /// that applies: `startup` (less than `auto_tau_s` seconds of history),
    /// `no_sessions` (q = sat and N_active = 0), `no_busy` / `no_inflight`
    /// (no P history), `no_rate` (no R_hat), `not_finite`. With rate auto,
    /// R = `rate_tokens_per_s` for `startup` and `no_rate`.
    pub fn resolved(&self, auto: &TradeAutoInputs) -> (TradeParams, TradeAutoResolution) {
        let q_mode = match self.copy_lambda_mode {
            TradeLambdaMode::Auto { q, .. } => q,
            TradeLambdaMode::Fixed => TradeAutoQ::default(),
        };
        let q = auto.q(q_mode);
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
            q_mode: q_mode.as_str(),
            q,
            lambda_raw: None,
            copy_lambda_used: self.copy_lambda,
            fallback_reason: None,
            rate_mode: self.rate_mode.as_str(),
            rate_used: rate,
            rate_fallback_reason,
        };
        if let TradeLambdaMode::Auto { mu, min, max, .. } = self.copy_lambda_mode {
            let missing = if startup {
                Some("startup")
            } else if q_mode == TradeAutoQ::Sat && auto.n_active == 0 {
                Some("no_sessions")
            } else if q.is_none() {
                Some(match q_mode {
                    TradeAutoQ::Sat => "no_busy",
                    TradeAutoQ::Inflight => "no_inflight",
                })
            } else if r_hat.is_none() {
                Some("no_rate")
            } else {
                None
            };
            match (missing, q, r_hat) {
                (None, Some(q), Some(r_hat)) => {
                    let raw = mu * q * (rate / r_hat);
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
    /// lambda x pulled tokens; with `dup_charge = all`, lambda x (pulled
    /// tokens + dup_tokens).
    pub copy_tokens: f64,
    /// backlog + uncached + exposed_pull_tokens + copy_tokens + hol_tokens.
    pub total: f64,
    /// The wait term in seconds: B_w, or backlog / R.
    pub wait_s: f64,
    /// The own service in seconds: S_w, or u / R.
    pub service_s: f64,
    /// Tokens of the request this P recomputes although another P caches
    /// them. Always recorded; lambda charges them only with `dup_charge = all`.
    pub dup_tokens: u64,
    /// The HOL term mu * R * a_w * service_s * (wait_s + service_s / 2): mu
    /// times the light-load estimate of the extra wait (in tokens) of the
    /// requests that arrive at P later. 0 when `hol_weight` is 0.
    pub hol_tokens: f64,
    /// a_w: the arrival rate at P that the HOL term used (requests per s).
    pub arrival_rate_per_s: f64,
}

/// The head-of-line (HOL) term of one candidate, in uncached-token units:
///
/// ```text
/// hol = mu * R * a_w * S * (B + S / 2)
/// ```
///
/// mu = `hol_weight`, R = `rate_tokens_per_s`, a_w = `arrival_rate_per_s`
/// (requests per second that arrive at P), S = the request's own service on
/// P (`service_s`), B = P's wait term (`wait_s`; the pull time is not added).
/// The P queue is FCFS: the a_w x B requests that arrive during the wait
/// each wait S longer, and the a_w x S that arrive during the service wait
/// S / 2 longer on average.
///
/// This is a first-order (light-load) estimate. It assumes that the later
/// arrivals do not queue behind each other. Under load, the delay of S also
/// passes to the requests that arrive until P's queue without this request
/// would have drained. In a fluid model where the other requests keep P busy
/// a fraction rho of the time, the sum of the extra waits is
/// a_w * S * (B + S / 2) / (1 - rho) seconds: 5 times the light-load value
/// at rho = 0.8. In the other direction, a_w is the past placement rate on
/// P: after this placement P has more work, so TRADE sends some of the later
/// requests to another P, and fewer of them wait behind this one. So mu is
/// an empirical weight: mu = 1 gives the sum of the extra waits only at
/// light load, and it is not the weight that minimises the total wait.
///
/// With `wait = e3` the terms are the forecast ones: B = `wait_s` (decision
/// to the GPU start of the request's first pass), S = `pred_s - wait_s`
/// (its own passes and result lag), and a_w = the E3 arrival-rate fit of
/// P (requests per second that the router sent to P in the last 600 s;
/// the placement rate above until that fit has 30 requests).
/// Then mu = 1 is the physical quantity: the sum of the extra waits (in
/// R-token units) of the later arrivals, with the same light-load limit
/// (no queueing among the later arrivals; at utilisation rho the fluid
/// value is 1 / (1 - rho) times larger, and the rerouting of later
/// requests makes it smaller). mu stays a flag, with 1 as the value that
/// has a physical meaning.
///
/// Returns 0 when mu or a_w is not above 0.
pub fn trade_hol_tokens(
    params: &TradeParams,
    arrival_rate_per_s: f64,
    wait_s: f64,
    service_s: f64,
) -> f64 {
    if !(params.hol_weight > 0.0 && arrival_rate_per_s > 0.0) {
        return 0.0;
    }
    params.hol_weight
        * params.rate_tokens_per_s
        * arrival_rate_per_s
        * service_s
        * (wait_s + service_s / 2.0)
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
    trade_cost_full(params, backlog, uncached, pulled, 0, None, 0.0)
}

/// [`trade_cost`] with the later options:
///
/// ```text
/// wait = Some(B_w, S_w):
///   cost(P) = R * B_w + R * S_w + R * max(0, pull_s - B_w) + copy + hol
/// wait = None (ledger): the formula of trade_cost, with copy + hol
/// copy = lambda * pulled, or lambda * (pulled + dup) with dup_charge = all
/// hol  = mu * R * a_w * S * (B + S / 2)   (hol_weight mu > 0 only)
/// ```
///
/// `dup_tokens` = the tokens this P recomputes although another P caches
/// them ([`trade_duplicate_tokens`]); it is recorded in every case.
/// `arrival_rate_per_s` = a_w of this P; the HOL term uses the wait and
/// service of the selected wait source ([`trade_hol_tokens`]). With
/// `hol_weight = 0` the HOL term is not added (the cost does not change),
/// and with `dup_charge = pulled` and `wait = None` too, the cost equals
/// the first version bit for bit.
pub fn trade_cost_full(
    params: &TradeParams,
    backlog: u128,
    uncached: u64,
    pulled: u64,
    dup_tokens: u64,
    wait: Option<TradeWaitTerms>,
    arrival_rate_per_s: f64,
) -> TradeCost {
    let rate = params.rate_tokens_per_s;
    let pull_s = params.pull_seconds(pulled);
    let charged = match params.dup_charge {
        TradeDupCharge::Pulled => pulled,
        TradeDupCharge::All => pulled.saturating_add(dup_tokens),
    };
    let copy_tokens = params.copy_lambda * charged as f64;
    let mut cost = match wait {
        None => {
            let backlog = backlog as f64;
            let exposed_pull_s = if pulled == 0 {
                0.0
            } else {
                (pull_s - backlog / rate).max(0.0)
            };
            let exposed_pull_tokens = rate * exposed_pull_s;
            TradeCost {
                backlog,
                uncached: uncached as f64,
                pull_s,
                exposed_pull_s,
                exposed_pull_tokens,
                copy_tokens,
                total: backlog + uncached as f64 + exposed_pull_tokens + copy_tokens,
                wait_s: backlog / rate,
                service_s: uncached as f64 / rate,
                dup_tokens,
                hol_tokens: 0.0,
                arrival_rate_per_s,
            }
        }
        Some(terms) => {
            let backlog = rate * terms.wait_s;
            let own = rate * terms.service_s;
            let exposed_pull_s = if pulled == 0 {
                0.0
            } else {
                (pull_s - terms.wait_s).max(0.0)
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
                wait_s: terms.wait_s,
                service_s: terms.service_s,
                dup_tokens,
                hol_tokens: 0.0,
                arrival_rate_per_s,
            }
        }
    };
    // mu = 0 leaves the total untouched (bit for bit).
    if params.hol_weight > 0.0 {
        cost.hol_tokens = trade_hol_tokens(params, arrival_rate_per_s, cost.wait_s, cost.service_s);
        cost.total += cost.hol_tokens;
    }
    cost
}

/// One P candidate as seen by [`select_trade`].
#[derive(Debug, Clone, Copy)]
pub struct TradeInput<'a> {
    pub worker: &'a str,
    /// Pending uncached tokens on this P (see [`TradeParams::exclude_unlaunched`]).
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
    /// a_w: requests per second that arrive at this P ([`TradeArrivals`];
    /// with `wait = e3` the E3 arrival-rate fit when it exists); used only
    /// with `hol_weight > 0`. 0 when HOL is off.
    pub arrival_rate_per_s: f64,
    /// The copy price of this candidate (`pull_price = phi`). None: the
    /// copy price of the parameters ([`TradeParams::copy_lambda`]).
    pub copy_lambda: Option<f64>,
    /// `wait = e3`: the error bound b (s) of this P's forecast at the bound
    /// level. None: no usable forecast (the P then has the ledger terms in
    /// `wait = None`) or another wait source.
    pub bound_s: Option<f64>,
}

/// Tokens of the request that candidate `i` computes although another
/// candidate already caches them:
/// max(0, min(n, best other own hit) - (own hit + pulled)).
pub fn trade_duplicate_tokens(inputs: &[TradeInput<'_>], i: usize) -> u64 {
    let me = &inputs[i];
    let held = me.local_hit + me.pulled;
    let n = held + me.uncached;
    let elsewhere = inputs
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != i)
        .map(|(_, other)| other.local_hit)
        .max()
        .unwrap_or(0)
        .min(n);
    elsewhere.saturating_sub(held)
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
    /// argmin cost (before the noise margin).
    pub best: usize,
    /// The noise margin k * sqrt(B_o^2 + B_best^2) in seconds, when it was
    /// evaluated (k > 0, an owner exists and the best is another P).
    pub margin_s: Option<f64>,
    /// (cost(owner) - cost(best)) / R in seconds, when the margin was evaluated.
    pub owner_gain_s: Option<f64>,
    /// The noise margin kept the owner.
    pub margin_protected: bool,
    /// `wait = e3` only: why the owner was kept although another P had the
    /// lower cost (`gate`: the gain did not beat the bound margin;
    /// `owner_no_forecast`: the owner has no usable forecast). None for the
    /// other wait sources.
    pub protect_reason: Option<&'static str>,
}

/// The TRADE cost of every candidate, in the order of `inputs`.
fn trade_costs(params: &TradeParams, inputs: &[TradeInput<'_>]) -> Vec<TradeCost> {
    inputs
        .iter()
        .enumerate()
        .map(|(i, input)| {
            // A candidate's own copy price (pull price phi); None keeps the
            // parameters as they are (bit for bit).
            let own;
            let params = match input.copy_lambda {
                Some(copy_lambda) => {
                    own = TradeParams {
                        copy_lambda,
                        ..*params
                    };
                    &own
                }
                None => params,
            };
            trade_cost_full(
                params,
                input.backlog,
                input.uncached,
                input.pulled,
                trade_duplicate_tokens(inputs, i),
                input.wait,
                input.arrival_rate_per_s,
            )
        })
        .collect()
}

/// Choose argmin TRADE cost. Equal costs go to the cache-aware owner, then to
/// the higher own hit, then to the lower backlog, then to the smaller URL.
/// With `margin_noise_k > 0` the
/// owner is kept unless its cost exceeds the best one by more than the noise
/// margin. With `hol_weight > 0` every cost includes the HOL term, so these
/// rules and the gain of the noise margin use it; the margin itself is
/// unchanged. With `wait = e3` the selection is [`select_trade_e3`]; the
/// other wait sources are unchanged by it. Pure: no I/O, no reservation.
/// Panics if `inputs` is empty.
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
    if params.margin_noise_k > 0.0 && has_owner && best != owner {
        let (b_o, b_w) = (selection.costs[owner].wait_s, selection.costs[best].wait_s);
        let margin = params.margin_noise_k * (b_o * b_o + b_w * b_w).sqrt();
        let gain =
            (selection.costs[owner].total - selection.costs[best].total) / params.rate_tokens_per_s;
        selection.margin_s = Some(margin);
        selection.owner_gain_s = Some(gain);
        if gain <= margin {
            selection.chosen = owner;
            selection.margin_protected = true;
        }
    }
    selection
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
    /// The key that decided: `none` (no tie, or the noise margin kept the
    /// owner), `room`, `local_hit`, `outstanding` or `rotation`.
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
/// choice of the noise margin stays. Pure; the walk and the release
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

/// phi_P of `--trade-pull-price phi` for one P worker: the
/// live share of the host tokens that the P's KV session retention evicted,
/// from the cumulative counters of the engine's `/v1/loads` host_room
/// section (`evicted_class0_tokens`: nodes that no session counts;
/// `evicted_class1_tokens`: nodes that a session counts).
///
/// Two EWMA rates with time constant tau (the copy price auto tau) follow
/// the increments of the two counters, like the credited-token rate of
/// [`TradeAutoState`]: an increment of k tokens adds k / tau, and the rates
/// decay as exp(-dt / tau).
///
/// ```text
/// phi = r1 / (r0 + r1)   when the P evicted host tokens in the last tau seconds
/// phi = 0                when it evicted none in the last tau seconds (or never)
/// no view (None)         before the second sample, or when the last sample is
///                        older than tau
/// ```
///
/// A counter below its previous value (an engine restart) starts a new
/// history (counted in `resets`). Every time comes from the caller, so the
/// tests use synthetic times.
#[derive(Debug, Clone, Default)]
pub struct TradePhiState {
    /// The last sample: (time, evicted class 0, evicted class 1).
    last: Option<(Instant, u64, u64)>,
    /// EWMA rates of the class-0 and class-1 increments (tokens per second),
    /// exact at `updated`.
    rate0: f64,
    rate1: f64,
    updated: Option<Instant>,
    /// The last sample with an increment of either counter.
    last_evict: Option<Instant>,
    /// Samples after the first of the current history (each gives the
    /// increments since the previous sample).
    pub samples: u64,
    /// Histories restarted after a counter went down.
    pub resets: u64,
}

/// phi of one P at one time ([`TradePhiState::view`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TradePhiView {
    /// The live share of the evicted host tokens; 0 without an eviction in
    /// the last tau seconds.
    pub phi: f64,
    /// `phi`, or `no_eviction` (no evicted token in the last tau seconds).
    pub source: &'static str,
    /// The two EWMA rates now (evicted tokens per second, not corrected for
    /// a short history).
    pub rate_class0: f64,
    pub rate_class1: f64,
    /// Seconds since the last sample with an increment (None: none yet).
    pub last_evict_age_s: Option<f64>,
    /// Seconds since the last sample.
    pub sample_age_s: f64,
}

impl TradePhiState {
    /// One host_room sample of the P, polled at `at`: the cumulative evicted
    /// tokens of class 0 and class 1. A sample not newer than the last one
    /// changes nothing.
    pub fn observe(&mut self, at: Instant, evicted0: u64, evicted1: u64, tau_s: f64) {
        let Some((last_at, last0, last1)) = self.last else {
            self.last = Some((at, evicted0, evicted1));
            return;
        };
        if at <= last_at {
            return;
        }
        self.last = Some((at, evicted0, evicted1));
        if evicted0 < last0 || evicted1 < last1 {
            // The engine restarted: a new history.
            self.rate0 = 0.0;
            self.rate1 = 0.0;
            self.updated = None;
            self.last_evict = None;
            self.samples = 0;
            self.resets += 1;
            return;
        }
        let (d0, d1) = (evicted0 - last0, evicted1 - last1);
        let decay = self.updated.map_or(0.0, |updated| {
            (-at.saturating_duration_since(updated).as_secs_f64() / tau_s).exp()
        });
        self.rate0 = self.rate0 * decay + d0 as f64 / tau_s;
        self.rate1 = self.rate1 * decay + d1 as f64 / tau_s;
        self.updated = Some(at);
        if d0 > 0 || d1 > 0 {
            self.last_evict = Some(at);
        }
        self.samples += 1;
    }

    /// phi at `now` (see the type). None before the second sample of the
    /// history or when the last sample is older than `tau_s` (no host_room
    /// data).
    pub fn view(&self, now: Instant, tau_s: f64) -> Option<TradePhiView> {
        let (last_at, _, _) = self.last?;
        let sample_age_s = now.saturating_duration_since(last_at).as_secs_f64();
        if self.samples == 0 || sample_age_s > tau_s {
            return None;
        }
        let decay = self.updated.map_or(0.0, |updated| {
            (-now.saturating_duration_since(updated).as_secs_f64() / tau_s).exp()
        });
        let (rate_class0, rate_class1) = (self.rate0 * decay, self.rate1 * decay);
        let last_evict_age_s = self
            .last_evict
            .map(|at| now.saturating_duration_since(at).as_secs_f64());
        let recent = last_evict_age_s.is_some_and(|age| age <= tau_s);
        // The decay is common to both rates, so the share is that of the
        // rates at the last update.
        let total = self.rate0 + self.rate1;
        let (phi, source) = if recent && total > 0.0 {
            (self.rate1 / total, "phi")
        } else {
            (0.0, "no_eviction")
        };
        Some(TradePhiView {
            phi,
            source,
            rate_class0,
            rate_class1,
            last_evict_age_s,
            sample_age_s,
        })
    }
}

/// TRADE selection with `wait = e3`: argmin cost with the prediction gate
/// (the reroute rule of).
///
/// A candidate is usable when it has a forecast: `wait` (B_w = `wait_s`,
/// S_w = `pred_s - wait_s`) and `bound_s` (b) are both set. Its cost is
/// [`trade_cost_full`] on these terms, that is
/// `C_w = R x (pred_s + max(0, pull_s - wait_s)) + copy (+ hol)`. A
/// candidate without a forecast has the ledger cost.
///
/// - Owner o: the maximum own hit; ties go to a usable P, then to the lower
///   cost, the lower backlog and the smaller URL.
/// - Best w*: argmin cost over the usable P (over all P when none is
///   usable); ties go to the owner, the higher own hit, the shorter wait,
///   the lower backlog and the smaller URL.
/// - No cached owner (own hit 0 on every P): w*.
/// - Otherwise the request leaves o for w* only when both are usable and
///   `gain - margin > g_min`, with `gain = (C_o - C_w*) / R` and
///   `margin = m x sqrt(b_o^2 + b_w*^2)` (s; m = `gate_m`, g_min =
///   `gate_gmin_s`). Else it stays on o: `protect_reason` is
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
    let moves = gain - margin > params.gate_gmin_s;
    if !moves {
        selection.chosen = owner;
        selection.margin_protected = true;
        selection.protect_reason = Some("gate");
    }
    selection
}

/// The router's estimate of a_w for the HOL term: the times of its TRADE
/// placements on each P worker. The cost book records every placement (at
/// the reservation, under its lock, so concurrent decisions see each other;
/// a later rollback or failure does not remove it) and prunes the entries
/// older than the window at each decision. Times are monotonic.
#[derive(Debug, Default)]
pub struct TradeArrivals {
    /// The first recorded placement on any P: the start of the history.
    first: Option<Instant>,
    /// Placement times per P URL, oldest first.
    times: BTreeMap<String, VecDeque<Instant>>,
}

impl TradeArrivals {
    /// Record one placement on `worker` at `at`.
    pub fn record(&mut self, worker: &str, at: Instant) {
        if self.first.is_none() {
            self.first = Some(at);
        }
        self.times
            .entry(worker.to_string())
            .or_default()
            .push_back(at);
    }

    /// Drop the placements older than `window_s` seconds at `now`, and the P
    /// left without one. The start of the history does not change.
    pub fn prune(&mut self, now: Instant, window_s: f64) {
        self.times.retain(|_, times| {
            while times
                .front()
                .is_some_and(|t| now.saturating_duration_since(*t).as_secs_f64() > window_s)
            {
                times.pop_front();
            }
            !times.is_empty()
        });
    }

    /// a_w in requests per second: the placements on `worker` in the last
    /// `window_s` seconds divided by min(`window_s`, max(1 s, `now` - the
    /// first placement)). An under-filled window (shortly after the first
    /// placement) divides by the elapsed time, floored at 1 s. A full window
    /// divides by `window_s`, also when `window_s` is below 1 s. 0 without
    /// history.
    pub fn rate_per_s(&self, worker: &str, now: Instant, window_s: f64) -> f64 {
        let Some(first) = self.first else {
            return 0.0;
        };
        let count = self.times.get(worker).map_or(0, |times| {
            times
                .iter()
                .filter(|t| now.saturating_duration_since(**t).as_secs_f64() <= window_s)
                .count()
        });
        if count == 0 {
            return 0.0;
        }
        let elapsed = now.saturating_duration_since(first).as_secs_f64();
        count as f64 / window_s.min(elapsed.max(1.0))
    }

    /// The placements on `worker` that are kept (recorded and not pruned).
    pub fn recorded(&self, worker: &str) -> usize {
        self.times.get(worker).map_or(0, VecDeque::len)
    }
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
            arrival_rate_per_s: 0.0,
            copy_lambda: None,
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

    fn e3_params(lambda: f64, gate_m: f64, gate_gmin_s: f64) -> TradeParams {
        TradeParams {
            wait: TradeWait::E3,
            gate_m,
            gate_gmin_s,
            ..params(lambda)
        }
    }

    /// The prediction gate on hand cases (R = 13,500, lambda = 0). Owner a
    /// caches 40,000 tokens: pred 2.0 s, bound 0.3 s. b is idle and pulls
    /// 40,000 tokens (pull 0.49 s, wait 0.1 s: 0.39 s exposed): pred 0.6 s,
    /// bound 0.2 s. Gain = 2.0 - (0.6 + 0.39) = 1.01 s; margin =
    /// sqrt(0.09 + 0.04) = 0.3606 s at m = 1: b wins. m = 3 (margin 1.08 s)
    /// or g_min = 0.7 s keeps a (`gate`).
    #[test]
    fn e3_gate_moves_only_when_the_gain_beats_the_bound_margin() {
        let inputs = [
            e3(input("http://a", 0, 40_000, 2_000, 0), 1.5, 2.0, 0.3),
            e3(input("http://b", 0, 0, 2_000, 40_000), 0.1, 0.6, 0.2),
        ];
        let s = select_trade(&e3_params(0.0, 1.0, 0.0), &inputs);
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
        for p in [e3_params(0.0, 3.0, 0.0), e3_params(0.0, 1.0, 0.7)] {
            let s = select_trade(&p, &inputs);
            assert_eq!((s.owner, s.best, s.chosen), (0, 1, 0), "{p:?}");
            assert!(s.margin_protected);
            assert_eq!(s.protect_reason, Some("gate"));
        }
        // The copy price is part of C: lambda 9.2 x 40,000 = 368,000 tokens
        // = 27.3 s of R; the owner keeps the request on cost alone.
        let s = select_trade(&e3_params(9.2, 1.0, 0.0), &inputs);
        assert_eq!((s.best, s.chosen), (0, 0));
        assert_eq!(s.protect_reason, None);
        // m = 0 and g_min = 0: a move needs only a positive gain.
        let s = select_trade(&e3_params(0.0, 0.0, 0.0), &inputs);
        assert_eq!(s.chosen, 1);
    }

    /// A P without a usable forecast (no wait or no bound) has the ledger
    /// cost and never wins a move: as best it is skipped while another P has
    /// a forecast; an owner without a forecast is kept (`owner_no_forecast`).
    #[test]
    fn e3_candidates_without_a_forecast_never_win_a_move() {
        let p = e3_params(0.0, 1.0, 0.0);
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
        let p = e3_params(0.0, 1.0, 0.0);
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

    /// `select_trade` as it was before `wait = e3` existed, for
    /// the bit-exactness test of the other wait sources.
    mod pre_e3 {
        use super::super::{
            trade_cost_full, trade_duplicate_tokens, TradeCost, TradeInput, TradeParams,
        };

        pub struct Selection {
            pub chosen: usize,
            pub owner: usize,
            pub has_owner: bool,
            pub costs: Vec<TradeCost>,
            pub best: usize,
            pub margin_s: Option<f64>,
            pub owner_gain_s: Option<f64>,
            pub margin_protected: bool,
        }

        pub fn select_trade(params: &TradeParams, inputs: &[TradeInput<'_>]) -> Selection {
            assert!(!inputs.is_empty());
            let costs: Vec<TradeCost> = inputs
                .iter()
                .enumerate()
                .map(|(i, input)| {
                    trade_cost_full(
                        params,
                        input.backlog,
                        input.uncached,
                        input.pulled,
                        trade_duplicate_tokens(inputs, i),
                        input.wait,
                        input.arrival_rate_per_s,
                    )
                })
                .collect();
            let eft = false;
            let owner = inputs
                .iter()
                .enumerate()
                .min_by(|(ia, a), (ib, b)| {
                    b.local_hit
                        .cmp(&a.local_hit)
                        .then_with(|| {
                            if eft {
                                costs[*ia].total.total_cmp(&costs[*ib].total)
                            } else {
                                std::cmp::Ordering::Equal
                            }
                        })
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
                        .then_with(|| {
                            if eft {
                                costs[*ia].wait_s.total_cmp(&costs[*ib].wait_s)
                            } else {
                                std::cmp::Ordering::Equal
                            }
                        })
                        .then_with(|| a.backlog.cmp(&b.backlog))
                        .then_with(|| a.worker.cmp(b.worker))
                })
                .unwrap()
                .0;
            let has_owner = inputs[owner].local_hit > 0;
            let mut selection = Selection {
                chosen: best,
                owner,
                has_owner,
                costs,
                best,
                margin_s: None,
                owner_gain_s: None,
                margin_protected: false,
            };
            if params.margin_noise_k > 0.0 && has_owner && best != owner {
                let (b_o, b_w) = (selection.costs[owner].wait_s, selection.costs[best].wait_s);
                let margin = params.margin_noise_k * (b_o * b_o + b_w * b_w).sqrt();
                let gain = (selection.costs[owner].total - selection.costs[best].total)
                    / params.rate_tokens_per_s;
                selection.margin_s = Some(margin);
                selection.owner_gain_s = Some(gain);
                if gain <= margin {
                    selection.chosen = owner;
                    selection.margin_protected = true;
                }
            }
            selection
        }
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

    /// Synthetic times: `t0` plus `s` seconds.
    fn secs_after(t0: Instant, s: f64) -> Instant {
        t0 + std::time::Duration::from_secs_f64(s)
    }

    /// phi = r1 / (r0 + r1) of two EWMA rates of the counter increments: an
    /// increment of k tokens adds k / tau, both rates decay as exp(-dt / tau),
    /// and the share does not change while nothing is evicted. A sample not
    /// newer than the last one changes nothing.
    #[test]
    fn phi_state_ewma_math() {
        let tau = 120.0;
        let t0 = Instant::now();
        let at = |s: f64| secs_after(t0, s);
        let mut phi = TradePhiState::default();
        phi.observe(at(0.0), 1_000, 500, tau);
        assert!(phi.view(at(0.0), tau).is_none(), "one sample, no increment");
        // +300 class 0, +100 class 1.
        phi.observe(at(1.0), 1_300, 600, tau);
        let v = phi.view(at(1.0), tau).unwrap();
        assert_eq!(v.source, "phi");
        assert!((v.phi - 0.25).abs() < 1e-12, "{v:?}");
        assert!(close(v.rate_class0, 300.0 / tau));
        assert!(close(v.rate_class1, 100.0 / tau));
        // +300 class 1, 10 s later.
        phi.observe(at(11.0), 1_300, 900, tau);
        let decay = (-10.0f64 / tau).exp();
        let r0 = 300.0 / tau * decay;
        let r1 = 100.0 / tau * decay + 300.0 / tau;
        let v = phi.view(at(11.0), tau).unwrap();
        assert!((v.phi - r1 / (r0 + r1)).abs() < 1e-12, "{v:?}");
        // 60 s without a new sample: the rates decay, the share stays.
        let later = phi.view(at(71.0), tau).unwrap();
        assert!((later.phi - v.phi).abs() < 1e-12);
        let d60 = (-60.0f64 / tau).exp();
        assert!(close(later.rate_class1, r1 * d60));
        assert_eq!(later.last_evict_age_s, Some(60.0));
        assert_eq!(later.sample_age_s, 60.0);
        // An older or an equal sample changes nothing.
        let before = phi.clone();
        phi.observe(at(5.0), 0, 0, tau);
        phi.observe(at(11.0), 9_999, 9_999, tau);
        let now = at(11.0);
        assert_eq!(phi.view(now, tau), before.view(now, tau));
        assert_eq!((phi.samples, phi.resets), (2, 0));
    }

    /// phi = 0 when the P evicted no host token in the last tau seconds,
    /// also when it never evicted one; a new eviction brings the share back.
    #[test]
    fn phi_state_zero_eviction_window() {
        let tau = 120.0;
        let t0 = Instant::now();
        let at = |s: f64| secs_after(t0, s);
        let mut never = TradePhiState::default();
        never.observe(at(0.0), 7, 7, tau);
        never.observe(at(1.0), 7, 7, tau);
        let v = never.view(at(1.0), tau).unwrap();
        assert_eq!((v.phi, v.source), (0.0, "no_eviction"));
        assert_eq!(v.last_evict_age_s, None);
        let mut phi = TradePhiState::default();
        phi.observe(at(0.0), 0, 0, tau);
        // phi 0.75, then polls without an eviction.
        phi.observe(at(1.0), 100, 300, tau);
        for s in [30.0, 60.0, 90.0, 120.0] {
            phi.observe(at(s), 100, 300, tau);
        }
        // 120 s after the eviction: still in the window.
        let v = phi.view(at(121.0), tau).unwrap();
        assert_eq!(v.source, "phi");
        assert!((v.phi - 0.75).abs() < 1e-12, "{v:?}");
        // 120.5 s: out of the window.
        let v = phi.view(at(121.5), tau).unwrap();
        assert_eq!((v.phi, v.source), (0.0, "no_eviction"));
        // A class-0 eviction: back to a share, now below 0.75.
        phi.observe(at(122.0), 200, 300, tau);
        let v = phi.view(at(122.0), tau).unwrap();
        assert_eq!(v.source, "phi");
        assert!(v.phi > 0.0 && v.phi < 0.75, "{v:?}");
    }

    /// No view (no host_room data) without a sample, after one sample, and
    /// when the last sample is older than tau. A counter that goes down (an
    /// engine restart) starts a new history.
    #[test]
    fn phi_state_missing_data_and_restart() {
        let tau = 120.0;
        let t0 = Instant::now();
        let at = |s: f64| secs_after(t0, s);
        let mut phi = TradePhiState::default();
        assert!(phi.view(at(0.0), tau).is_none(), "no sample");
        phi.observe(at(0.0), 10, 10, tau);
        assert!(phi.view(at(0.0), tau).is_none(), "one sample");
        phi.observe(at(1.0), 20, 30, tau);
        assert!(phi.view(at(121.0), tau).is_some());
        let stale = phi.view(at(121.5), tau);
        assert!(stale.is_none(), "the last sample is older than tau");
        // The engine restarted: lower counters, a new history.
        phi.observe(at(130.0), 0, 5, tau);
        assert_eq!((phi.resets, phi.samples), (1, 0));
        let restarted = phi.view(at(130.0), tau);
        assert!(restarted.is_none(), "a new history needs a second sample");
        phi.observe(at(131.0), 4, 5, tau);
        let v = phi.view(at(131.0), tau).unwrap();
        assert_eq!((v.phi, v.source), (0.0, "phi"));
    }

    /// TradeInput::copy_lambda prices each candidate's pull with its own copy
    /// price; None keeps the price of the parameters, bit for bit. Owner `a`
    /// holds 100k of 110k behind 60k pending (cost 70,000); `b` is idle and
    /// pulls 100k (17,425 + lambda x 100,000).
    #[test]
    fn select_trade_prices_each_candidate_with_its_own_lambda() {
        let p = params(9.2);
        let a = input("http://a", 60_000, 100_000, 10_000, 0);
        let b = input("http://b", 0, 0, 10_000, 100_000);
        let s = select_trade(&p, &[a, b]);
        assert_eq!(s.chosen, 0, "lambda 9.2 keeps the owner");
        // b's own price 0 (phi 0: b evicts no live token): the pull is free.
        let free_b = TradeInput {
            copy_lambda: Some(0.0),
            ..b
        };
        let f = select_trade(&p, &[a, free_b]);
        assert_eq!(f.chosen, 1);
        assert_eq!(f.costs[1].copy_tokens, 0.0);
        assert!(close(f.costs[1].total, 17_425.0));
        assert_eq!(f.costs[0], s.costs[0], "the owner keeps its cost");
        // b's own price = the parameters' price: the same costs, bit for bit.
        let same_b = TradeInput {
            copy_lambda: Some(9.2),
            ..b
        };
        assert_eq!(select_trade(&p, &[a, same_b]).costs, s.costs);
        // An own price on the owner: it pulls nothing, so nothing changes.
        let priced_a = TradeInput {
            copy_lambda: Some(1000.0),
            ..a
        };
        assert_eq!(select_trade(&p, &[priced_a, b]).costs, s.costs);
    }

    #[test]
    fn defaults_match_the_flag_defaults() {
        let p = TradeParams::default();
        assert_eq!(p.rate_tokens_per_s, 13_500.0);
        assert_eq!(p.pull_fixed_ms, 450.0);
        assert_eq!(p.pull_ms_per_100k, 100.0);
        assert_eq!(p.copy_lambda, 4.6);
        assert!(!p.exclude_unlaunched);
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
            pub backlog: f64,
            pub uncached: f64,
            pub pull_s: f64,
            pub exposed_pull_s: f64,
            pub exposed_pull_tokens: f64,
            pub copy_tokens: f64,
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
                backlog,
                uncached: uncached as f64,
                pull_s,
                exposed_pull_s,
                exposed_pull_tokens,
                copy_tokens,
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
        let c = trade_cost_full(&p, 27_000, 1_000, 100_000, 0, Some(w), 0.0);
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
            0,
            Some(TradeWaitTerms {
                wait_s: 3.0,
                service_s: 0.3,
            }),
            0.0,
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

    /// dup charge `all`: lambda also prices the tokens a move recomputes
    /// that another P caches; a pull of the same prefix pays only for the pull.
    #[test]
    fn dup_charge_all_prices_recomputed_duplicates() {
        let inputs = [
            input("http://a", 60_000, 30_000, 10_000, 0),
            input("http://b", 0, 0, 40_000, 0),
        ];
        assert_eq!(
            trade_duplicate_tokens(&inputs, 0),
            0,
            "the owner holds the most"
        );
        assert_eq!(trade_duplicate_tokens(&inputs, 1), 30_000);
        let pulled = TradeParams {
            copy_lambda: 1.0,
            ..TradeParams::default()
        };
        let s = select_trade(&pulled, &inputs);
        assert_eq!(s.chosen, 1, "40,000 < 70,000 without the duplicate price");
        assert_eq!(
            s.costs[1].dup_tokens, 30_000,
            "recorded although not charged"
        );
        assert_eq!(s.costs[1].copy_tokens, 0.0);
        assert_eq!(s.costs[0].dup_tokens, 0);
        let all = TradeParams {
            dup_charge: TradeDupCharge::All,
            ..pulled
        };
        let s = select_trade(&all, &inputs);
        assert_eq!(s.costs[1].dup_tokens, 30_000);
        assert!(close(s.costs[1].copy_tokens, 30_000.0));
        assert!(close(s.costs[1].total, 70_000.0));
        assert_eq!(s.chosen, 0, "tie at 70,000 goes to the owner");
        // A pull brings the cached prefix: nothing recomputed is cached elsewhere.
        let pulling = [
            input("http://a", 60_000, 30_000, 10_000, 0),
            input("http://b", 0, 0, 10_000, 30_000),
        ];
        assert_eq!(trade_duplicate_tokens(&pulling, 1), 0);
        let s = select_trade(&all, &pulling);
        assert!(close(s.costs[1].copy_tokens, 30_000.0));
    }

    /// Noise margin: the owner keeps the request unless the gain in seconds
    /// exceeds k * sqrt(B_o^2 + B_best^2).
    #[test]
    fn noise_margin_keeps_the_owner_inside_the_noise() {
        // Owner a: B 4 s, S 0.1 s. Peer b: B 0 s, S 0.3 s, no pull. Gain = 3.8 s.
        let inputs = [
            eft(input("http://a", 0, 30_000, 1_000, 0), 4.0, 0.1),
            eft(input("http://b", 0, 0, 31_000, 0), 0.0, 0.3),
        ];
        let off = params(0.0);
        let s = select_trade(&off, &inputs);
        assert_eq!(s.chosen, 1);
        assert!(s.margin_s.is_none());
        // k = 0.9: margin 3.6 s < 3.8 s -> moves. k = 1.0: 4 s >= 3.8 s -> stays.
        let s = select_trade(
            &TradeParams {
                margin_noise_k: 0.9,
                ..off
            },
            &inputs,
        );
        assert_eq!(s.chosen, 1);
        assert!(close(s.margin_s.unwrap(), 3.6));
        assert!(close(s.owner_gain_s.unwrap(), 3.8));
        assert!(!s.margin_protected);
        let s = select_trade(
            &TradeParams {
                margin_noise_k: 1.0,
                ..off
            },
            &inputs,
        );
        assert_eq!(s.best, 1);
        assert_eq!(s.chosen, 0);
        assert!(s.margin_protected);
        // A cold request has no owner to keep.
        let cold = [
            eft(input("http://a", 0, 0, 31_000, 0), 4.0, 0.3),
            eft(input("http://b", 0, 0, 31_000, 0), 0.0, 0.3),
        ];
        let s = select_trade(
            &TradeParams {
                margin_noise_k: 10.0,
                ..off
            },
            &cold,
        );
        assert!(!s.has_owner);
        assert_eq!(s.chosen, 1);
        assert!(s.margin_s.is_none());
    }

    /// The TRADE cost and selection before the HOL term,
    /// copied here unchanged so that `hol_weight = 0` is pinned against it.
    mod pre_hol {
        use crate::scheduler::trade::{TradeDupCharge, TradeInput, TradeParams, TradeWaitTerms};

        pub struct Cost {
            pub backlog: f64,
            pub uncached: f64,
            pub pull_s: f64,
            pub exposed_pull_s: f64,
            pub exposed_pull_tokens: f64,
            pub copy_tokens: f64,
            pub total: f64,
            pub wait_s: f64,
            pub service_s: f64,
            pub dup_tokens: u64,
        }

        pub fn trade_cost_full(
            params: &TradeParams,
            backlog: u128,
            uncached: u64,
            pulled: u64,
            dup_tokens: u64,
            wait: Option<TradeWaitTerms>,
        ) -> Cost {
            let rate = params.rate_tokens_per_s;
            let pull_s = params.pull_seconds(pulled);
            let charged = match params.dup_charge {
                TradeDupCharge::Pulled => pulled,
                TradeDupCharge::All => pulled.saturating_add(dup_tokens),
            };
            let copy_tokens = params.copy_lambda * charged as f64;
            match wait {
                None => {
                    let backlog = backlog as f64;
                    let exposed_pull_s = if pulled == 0 {
                        0.0
                    } else {
                        (pull_s - backlog / rate).max(0.0)
                    };
                    let exposed_pull_tokens = rate * exposed_pull_s;
                    Cost {
                        backlog,
                        uncached: uncached as f64,
                        pull_s,
                        exposed_pull_s,
                        exposed_pull_tokens,
                        copy_tokens,
                        total: backlog + uncached as f64 + exposed_pull_tokens + copy_tokens,
                        wait_s: backlog / rate,
                        service_s: uncached as f64 / rate,
                        dup_tokens,
                    }
                }
                Some(terms) => {
                    let backlog = rate * terms.wait_s;
                    let own = rate * terms.service_s;
                    let exposed_pull_s = if pulled == 0 {
                        0.0
                    } else {
                        (pull_s - terms.wait_s).max(0.0)
                    };
                    let exposed_pull_tokens = rate * exposed_pull_s;
                    Cost {
                        backlog,
                        uncached: own,
                        pull_s,
                        exposed_pull_s,
                        exposed_pull_tokens,
                        copy_tokens,
                        total: backlog + own + exposed_pull_tokens + copy_tokens,
                        wait_s: terms.wait_s,
                        service_s: terms.service_s,
                        dup_tokens,
                    }
                }
            }
        }

        pub fn trade_duplicate_tokens(inputs: &[TradeInput<'_>], i: usize) -> u64 {
            let me = &inputs[i];
            let held = me.local_hit + me.pulled;
            let n = held + me.uncached;
            let elsewhere = inputs
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, other)| other.local_hit)
                .max()
                .unwrap_or(0)
                .min(n);
            elsewhere.saturating_sub(held)
        }

        pub struct Selection {
            pub chosen: usize,
            pub owner: usize,
            pub has_owner: bool,
            pub costs: Vec<Cost>,
            pub best: usize,
            pub margin_s: Option<f64>,
            pub owner_gain_s: Option<f64>,
            pub margin_protected: bool,
        }

        pub fn select_trade(params: &TradeParams, inputs: &[TradeInput<'_>]) -> Selection {
            let costs: Vec<Cost> = inputs
                .iter()
                .enumerate()
                .map(|(i, input)| {
                    trade_cost_full(
                        params,
                        input.backlog,
                        input.uncached,
                        input.pulled,
                        trade_duplicate_tokens(inputs, i),
                        input.wait,
                    )
                })
                .collect();
            let eft = false;
            let owner = inputs
                .iter()
                .enumerate()
                .min_by(|(ia, a), (ib, b)| {
                    b.local_hit
                        .cmp(&a.local_hit)
                        .then_with(|| {
                            if eft {
                                costs[*ia].total.total_cmp(&costs[*ib].total)
                            } else {
                                std::cmp::Ordering::Equal
                            }
                        })
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
                        .then_with(|| {
                            if eft {
                                costs[*ia].wait_s.total_cmp(&costs[*ib].wait_s)
                            } else {
                                std::cmp::Ordering::Equal
                            }
                        })
                        .then_with(|| a.backlog.cmp(&b.backlog))
                        .then_with(|| a.worker.cmp(b.worker))
                })
                .unwrap()
                .0;
            let has_owner = inputs[owner].local_hit > 0;
            let mut selection = Selection {
                chosen: best,
                owner,
                has_owner,
                costs,
                best,
                margin_s: None,
                owner_gain_s: None,
                margin_protected: false,
            };
            if params.margin_noise_k > 0.0 && has_owner && best != owner {
                let (b_o, b_w) = (selection.costs[owner].wait_s, selection.costs[best].wait_s);
                let margin = params.margin_noise_k * (b_o * b_o + b_w * b_w).sqrt();
                let gain = (selection.costs[owner].total - selection.costs[best].total)
                    / params.rate_tokens_per_s;
                selection.margin_s = Some(margin);
                selection.owner_gain_s = Some(gain);
                if gain <= margin {
                    selection.chosen = owner;
                    selection.margin_protected = true;
                }
            }
            selection
        }
    }

    /// A big cold prefill: u = 64,000, no cached owner, no pull, two P,
    /// R = 13,500, so S = 64,000 / 13,500 = 4.741 s on both.
    /// - P_a: backlog 20,000 (B = 1.481 s), a = 2.0/s:
    ///   HOL = mu x 13,500 x 2.0 x 4.741 x (1.481 + 2.370)
    ///   = mu x 2.0 x 64,000 x 52,000 / 13,500 = mu x 493,037.0
    /// - P_b: backlog 30,000 (B = 2.222 s), a = 0.3/s:
    ///   HOL = mu x 13,500 x 0.3 x 4.741 x (2.222 + 2.370)
    ///   = mu x 0.3 x 64,000 x 62,000 / 13,500 = mu x 88,177.8
    ///
    /// mu = 0: 84,000 (P_a) < 94,000 (P_b) -> P_a, the smaller backlog.
    /// mu = 1: 577,037.0 (P_a) > 182,177.8 (P_b) -> P_b, fewer later arrivals.
    /// The choice changes at mu = 10,000 / (493,037.0 - 88,177.8) = 0.0247.
    #[test]
    fn hol_sends_a_big_cold_prefill_to_the_p_with_fewer_arrivals() {
        let inputs = [
            TradeInput {
                arrival_rate_per_s: 2.0,
                ..input("http://a", 20_000, 0, 64_000, 0)
            },
            TradeInput {
                arrival_rate_per_s: 0.3,
                ..input("http://b", 30_000, 0, 64_000, 0)
            },
        ];
        let off = params(9.2);
        let s = select_trade(&off, &inputs);
        assert!(!s.has_owner);
        assert_eq!(s.chosen, 0);
        assert!(close(s.costs[0].total, 84_000.0));
        assert!(close(s.costs[1].total, 94_000.0));
        assert_eq!(s.costs[0].hol_tokens, 0.0);
        let s = select_trade(
            &TradeParams {
                hol_weight: 1.0,
                ..off
            },
            &inputs,
        );
        assert_eq!(s.chosen, 1);
        let hol_a: f64 = 2.0 * 64_000.0 * 52_000.0 / 13_500.0;
        let hol_b: f64 = 0.3 * 64_000.0 * 62_000.0 / 13_500.0;
        assert!((hol_a - 493_037.037).abs() < 1e-3, "{hol_a}");
        assert!((hol_b - 88_177.778).abs() < 1e-3, "{hol_b}");
        assert!(close(s.costs[0].hol_tokens, hol_a));
        assert!(close(s.costs[1].hol_tokens, hol_b));
        assert!(close(s.costs[0].total, 84_000.0 + hol_a));
        assert!(close(s.costs[1].total, 94_000.0 + hol_b));
        // Around the change at mu = 0.0247.
        assert_eq!(
            select_trade(
                &TradeParams {
                    hol_weight: 0.02,
                    ..off
                },
                &inputs
            )
            .chosen,
            0
        );
        assert_eq!(
            select_trade(
                &TradeParams {
                    hol_weight: 0.03,
                    ..off
                },
                &inputs
            )
            .chosen,
            1
        );
    }

    /// A small warm request: its owner holds 40,000 of its 41,000 tokens
    /// (u = 1,000 there). In this typical state the HOL term is small next
    /// to the other terms, so every mu keeps the mu = 0 choice.
    /// - P_a, owner: backlog 20,000, a = 2.0/s;
    ///   HOL = mu x 2.0 x 1,000 x 20,500 / 13,500 = mu x 3,037.0.
    /// - P_b: backlog 5,000, pulls 40,000 (0.49 s, 0.120 s = 1,615 tokens not
    ///   hidden), a = 0.5/s; HOL = mu x 0.5 x 1,000 x 5,500 / 13,500 = mu x 203.7.
    /// - P_c: idle, no pull (u = 41,000), a = 0.3/s;
    ///   HOL = mu x 0.3 x 41,000 x 20,500 / 13,500 = mu x 18,677.8.
    ///
    /// lambda 9.2: P_a 21,000 + 3,037 mu < P_c 41,000 + 18,678 mu, and P_b
    /// pays 368,000 for the copy -> P_a. lambda 0: P_b 7,615 + 204 mu is
    /// the cheapest -> P_b.
    #[test]
    fn hol_keeps_a_small_warm_request_where_mu_zero_puts_it() {
        let inputs = [
            TradeInput {
                arrival_rate_per_s: 2.0,
                ..input("http://a", 20_000, 40_000, 1_000, 0)
            },
            TradeInput {
                arrival_rate_per_s: 0.5,
                ..input("http://b", 5_000, 0, 1_000, 40_000)
            },
            TradeInput {
                arrival_rate_per_s: 0.3,
                ..input("http://c", 0, 0, 41_000, 0)
            },
        ];
        for (lambda, expected) in [(9.2, 0usize), (0.0, 1)] {
            let off = params(lambda);
            let base = select_trade(&off, &inputs);
            assert_eq!(base.owner, 0);
            assert_eq!(base.chosen, expected, "lambda {lambda}");
            for mu in [0.25, 0.5, 1.0, 2.0, 4.0] {
                let s = select_trade(
                    &TradeParams {
                        hol_weight: mu,
                        ..off
                    },
                    &inputs,
                );
                assert_eq!(s.chosen, base.chosen, "lambda {lambda}, mu {mu}");
                assert!(close(
                    s.costs[0].hol_tokens,
                    mu * 2.0 * 1_000.0 * 20_500.0 / 13_500.0
                ));
                assert!(close(
                    s.costs[1].hol_tokens,
                    mu * 0.5 * 1_000.0 * 5_500.0 / 13_500.0
                ));
                assert!(close(
                    s.costs[2].hol_tokens,
                    mu * 0.3 * 41_000.0 * 20_500.0 / 13_500.0
                ));
            }
        }
    }

    /// a_w = placements in the last W seconds / min(W, max(1 s, time since
    /// the first placement)); entries older than W drop out; 0 without
    /// history. A window below 1 s divides by W.
    #[test]
    fn hol_arrival_rate_uses_the_window_the_elapsed_time_and_a_1_s_floor() {
        let t0 = Instant::now();
        let at = |s: f64| t0 + std::time::Duration::from_secs_f64(s);
        let mut arrivals = TradeArrivals::default();
        // No history: 0 for every P.
        assert_eq!(arrivals.rate_per_s("http://a", at(5.0), 60.0), 0.0);
        arrivals.record("http://a", at(0.0));
        arrivals.record("http://a", at(0.2));
        arrivals.record("http://b", at(0.3));
        assert_eq!(arrivals.recorded("http://a"), 2);
        // 0.5 s after the first placement the elapsed time is floored at 1 s.
        assert!(close(arrivals.rate_per_s("http://a", at(0.5), 60.0), 2.0));
        assert!(close(arrivals.rate_per_s("http://b", at(0.5), 60.0), 1.0));
        assert_eq!(arrivals.rate_per_s("http://c", at(0.5), 60.0), 0.0);
        // A window below 1 s divides by W, not by 1 s (the 1 s floor applies
        // to the elapsed time only): 2 placements in the last 0.5 s / 0.5 s.
        assert!(close(arrivals.rate_per_s("http://a", at(0.25), 0.5), 4.0));
        // Under-filled window: 10 s after the first placement, divide by 10 s.
        assert!(close(arrivals.rate_per_s("http://a", at(10.0), 60.0), 0.2));
        // W = 5 s at 10 s: the placements at 0 s and 0.2 s are older than W.
        assert_eq!(arrivals.rate_per_s("http://a", at(10.0), 5.0), 0.0);
        arrivals.record("http://a", at(8.0));
        assert!(close(
            arrivals.rate_per_s("http://a", at(10.0), 5.0),
            1.0 / 5.0
        ));
        // A full window: 61 s after the first placement, divide by W = 60 s;
        // only the placements at 8 s and 30 s are in it.
        arrivals.record("http://a", at(30.0));
        assert!(close(
            arrivals.rate_per_s("http://a", at(61.0), 60.0),
            2.0 / 60.0
        ));
        assert_eq!(arrivals.rate_per_s("http://b", at(61.0), 60.0), 0.0);
        // The prune drops the old entries (and b, left without one); the
        // rates and the start of the history do not change.
        arrivals.prune(at(61.0), 60.0);
        assert_eq!(arrivals.recorded("http://a"), 2);
        assert_eq!(arrivals.recorded("http://b"), 0);
        assert!(!arrivals.times.contains_key("http://b"));
        assert_eq!(arrivals.first, Some(t0));
        assert!(close(
            arrivals.rate_per_s("http://a", at(61.0), 60.0),
            2.0 / 60.0
        ));
        // The elapsed time counts from the first placement on any P.
        arrivals.record("http://b", at(62.0));
        assert!(close(
            arrivals.rate_per_s("http://b", at(62.5), 60.0),
            1.0 / 60.0
        ));
        // A full window below 1 s: 1 placement in the last 0.5 s is 2/s (a
        // 1 s floor on the window would give 1/s).
        assert!(close(arrivals.rate_per_s("http://b", at(62.4), 0.5), 2.0));
    }

    fn auto_lambda(q: TradeAutoQ, mu: f64) -> TradeParams {
        TradeParams {
            copy_lambda: 18.4,
            copy_lambda_mode: TradeLambdaMode::Auto {
                q,
                mu,
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

    /// lambda = clamp(mu * q * R / R_hat, min, max) with q = sat or
    /// inflight; rate auto makes R / R_hat = 1 and sets R = R_hat. The
    /// selection then charges the resolved price.
    #[test]
    fn trade_auto_lambda_is_the_clamped_formula() {
        // sat: q = 90 / 3 x 0.75 = 22.5; lambda = 22.5 x 13,500 / 15,000 = 20.25.
        let (p, r) = auto_lambda(TradeAutoQ::Sat, 1.0).resolved(&loaded());
        assert_eq!(
            (r.lambda_mode, r.q_mode, r.rate_mode),
            ("auto", "sat", "fixed")
        );
        assert!(close(r.q.unwrap(), 22.5));
        assert!(close(r.lambda_raw.unwrap(), 20.25));
        assert!(close(p.copy_lambda, 20.25));
        assert_eq!(p.copy_lambda.to_bits(), r.copy_lambda_used.to_bits());
        assert_eq!(r.fallback_reason, None);
        assert_eq!(p.rate_tokens_per_s, 13_500.0);
        assert_eq!(r.rate_used, 13_500.0);
        assert_eq!(
            p.copy_lambda_mode,
            auto_lambda(TradeAutoQ::Sat, 1.0).copy_lambda_mode
        );
        // inflight: q = 1 + 9 = 10; lambda = 10 x 0.9 = 9.0. mu 0.5 halves it.
        let (p, r) = auto_lambda(TradeAutoQ::Inflight, 1.0).resolved(&loaded());
        assert_eq!(r.q_mode, "inflight");
        assert!(close(r.q.unwrap(), 10.0));
        assert!(close(p.copy_lambda, 9.0));
        let (p, _) = auto_lambda(TradeAutoQ::Inflight, 0.5).resolved(&loaded());
        assert!(close(p.copy_lambda, 4.6), "4.5 is clamped to the 4.6 floor");
        // rate auto: R = R_hat = 15,000, so lambda = mu x q = 22.5.
        let both = TradeParams {
            rate_mode: TradeRateMode::Auto,
            ..auto_lambda(TradeAutoQ::Sat, 1.0)
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
        let (p, r) = auto_lambda(TradeAutoQ::Sat, 1.0).resolved(&light);
        assert!(close(r.lambda_raw.unwrap(), 1.35));
        assert_eq!(p.copy_lambda, 4.6);
        assert_eq!(r.fallback_reason, None, "a clamp is not a fallback");
        let capped = TradeParams {
            copy_lambda_mode: TradeLambdaMode::Auto {
                q: TradeAutoQ::Sat,
                mu: 1.0,
                min: 4.6,
                max: 20.0,
            },
            ..auto_lambda(TradeAutoQ::Sat, 1.0)
        };
        assert_eq!(capped.resolved(&loaded()).0.copy_lambda, 20.0);
        // An idle pool (busy 0) prices at the floor; it is not a fallback.
        let idle = TradeAutoInputs {
            busy: Some(0.0),
            ..loaded()
        };
        let (p, r) = auto_lambda(TradeAutoQ::Sat, 1.0).resolved(&idle);
        assert_eq!(
            (p.copy_lambda, r.lambda_raw, r.fallback_reason),
            (4.6, Some(0.0), None)
        );
        // The selection charges the resolved price: b pulls 100k at 20.25.
        let inputs = [
            input("http://a", 60_000, 100_000, 10_000, 0),
            input("http://b", 0, 0, 10_000, 100_000),
        ];
        let (p, _) = auto_lambda(TradeAutoQ::Sat, 1.0).resolved(&loaded());
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
        let sat = auto_lambda(TradeAutoQ::Sat, 1.0);
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
        let inflight = auto_lambda(TradeAutoQ::Inflight, 1.0);
        assert_eq!(
            reason(&inflight, no_sessions).1,
            None,
            "inflight needs no sessions"
        );
        let no_history = TradeAutoInputs {
            busy: None,
            inflight: None,
            r_hat: None,
            ..loaded()
        };
        assert_eq!(reason(&sat, no_history), (18.4, Some("no_busy")));
        assert_eq!(reason(&inflight, no_history), (18.4, Some("no_inflight")));
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
        let huge = TradeParams {
            copy_lambda_mode: TradeLambdaMode::Auto {
                q: TradeAutoQ::Sat,
                mu: f64::MAX,
                min: 4.6,
                max: 1000.0,
            },
            ..sat
        };
        assert_eq!(reason(&huge, loaded()), (18.4, Some("not_finite")));
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
        let (p, r) = auto_lambda(TradeAutoQ::Sat, 1.0).resolved(&inputs);
        assert_eq!(r.fallback_reason, None);
        assert!((r.q.unwrap() / 22.5 - 1.0).abs() < 0.01, "{r:?}");
        assert!((p.copy_lambda / 30.375 - 1.0).abs() < 0.03, "{r:?}");
        // Before tau seconds of history the fixed price applies.
        let mut fresh = TradeAutoState::new(at(t), 120.0);
        fresh.observe_pool(pool, at(t));
        let early = fresh.inputs(at(t + 60.0), &pool, 300.0);
        let (p, r) = auto_lambda(TradeAutoQ::Sat, 1.0).resolved(&early);
        assert_eq!((p.copy_lambda, r.fallback_reason), (18.4, Some("startup")));
    }

    /// With the new options at their defaults, every cost term and every
    /// selection equals the first version bit for bit: a grid of costs and
    /// 2,000 pseudo-random candidate sets (coarse values, so ties occur).
    #[test]
    fn defaults_are_the_first_version() {
        let p = TradeParams::default();
        assert_eq!(p.wait, TradeWait::Ledger);
        assert_eq!(p.dup_charge, TradeDupCharge::Pulled);
        assert_eq!(p.margin_noise_k, 0.0);
        assert_eq!(p.eft_max_view_age_s, 1.5);
        assert_eq!(p.hol_weight, 0.0);
        assert_eq!(p.hol_rate_window_s, 60.0);
        assert!(p.is_original());
        assert!(!TradeParams {
            dup_charge: TradeDupCharge::All,
            ..p
        }
        .is_original());
        assert!(!TradeParams {
            margin_noise_k: 1.0,
            ..p
        }
        .is_original());
        assert!(!TradeParams {
            hol_weight: 1.0,
            ..p
        }
        .is_original());
        assert!(TradeParams {
            hol_rate_window_s: 5.0,
            ..p
        }
        .is_original());
        for lambda in [0.0, 1.0, 9.2] {
            let p = params(lambda);
            for backlog in [0u128, 1, 5_400, 7_425, 27_000, 40_000, 1_000_003] {
                for uncached in [0u64, 1, 1_000, 2_000, 12_000] {
                    for pulled in [0u64, 512, 18_000, 100_000] {
                        let new = trade_cost(&p, backlog, uncached, pulled);
                        let old = first_version::trade_cost(&p, backlog, uncached, pulled);
                        let pairs = [
                            ("backlog", new.backlog, old.backlog),
                            ("uncached", new.uncached, old.uncached),
                            ("pull_s", new.pull_s, old.pull_s),
                            ("exposed_s", new.exposed_pull_s, old.exposed_pull_s),
                            ("exposed", new.exposed_pull_tokens, old.exposed_pull_tokens),
                            ("copy", new.copy_tokens, old.copy_tokens),
                            ("total", new.total, old.total),
                        ];
                        for (name, a, b) in pairs {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "{name}: {lambda} {backlog} {uncached} {pulled}"
                            );
                        }
                        assert!(close(new.wait_s, backlog as f64 / 13_500.0));
                        assert!(close(new.service_s, uncached as f64 / 13_500.0));
                    }
                }
            }
        }
        // xorshift64: deterministic candidate sets without a rand dependency.
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = |m: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % m
        };
        let workers = ["http://a", "http://b", "http://c", "http://d"];
        for round in 0..2_000 {
            let n = 1 + next(4) as usize;
            let inputs: Vec<TradeInput<'_>> = (0..n)
                .map(|i| TradeInput {
                    worker: workers[i],
                    backlog: u128::from(next(4) * 10_000),
                    local_hit: next(3) * 8_192,
                    uncached: 1 + next(3) * 4_000,
                    pulled: [0, 0, 512, 18_000][next(4) as usize],
                    wait: None,
                    arrival_rate_per_s: 0.0,
                    copy_lambda: None,
                    bound_s: None,
                })
                .collect();
            let p = params([0.0, 9.2][next(2) as usize]);
            let new = select_trade(&p, &inputs);
            let (chosen, owner, has_owner, costs) = first_version::select_trade(&p, &inputs);
            assert_eq!(
                (new.chosen, new.owner, new.has_owner),
                (chosen, owner, has_owner),
                "round {round}"
            );
            assert_eq!(new.best, chosen);
            assert!(!new.margin_protected && new.margin_s.is_none());
            for (a, b) in new.costs.iter().zip(&costs) {
                assert_eq!(a.total.to_bits(), b.total.to_bits(), "round {round}");
                assert_eq!(
                    a.copy_tokens.to_bits(),
                    b.copy_tokens.to_bits(),
                    "round {round}"
                );
            }
        }
    }

    /// HOL = mu * R * a * S * (B + S / 2) tokens, added to the total; the
    /// wait B is that of the wait source, without the pull time.
    #[test]
    fn hol_term_matches_a_hand_computed_example() {
        // Ledger: backlog 27,000 (B = 2 s), u = 13,500 (S = 1 s), a = 2/s,
        // mu = 0.5: HOL = 0.5 x 13,500 x 2 x 1 x (2 + 0.5) = 33,750;
        // total = 27,000 + 13,500 + 33,750 = 74,250.
        let p = TradeParams {
            hol_weight: 0.5,
            ..params(0.0)
        };
        assert!(close(trade_hol_tokens(&p, 2.0, 2.0, 1.0), 33_750.0));
        let c = trade_cost_full(&p, 27_000, 13_500, 0, 0, None, 2.0);
        assert!(close(c.hol_tokens, 33_750.0));
        assert!(close(c.total, 74_250.0));
        assert_eq!(c.arrival_rate_per_s, 2.0);
        // EFT terms: B_w = 0.4 s, S_w = 0.2 s, a = 3/s, mu = 1:
        // HOL = 13,500 x 3 x 0.2 x (0.4 + 0.1) = 4,050;
        // total = 13,500 x (0.4 + 0.2) + 4,050 = 12,150.
        let p1 = TradeParams {
            hol_weight: 1.0,
            ..params(0.0)
        };
        let w = TradeWaitTerms {
            wait_s: 0.4,
            service_s: 0.2,
        };
        let c = trade_cost_full(&p1, 0, 1_000, 0, 0, Some(w), 3.0);
        assert!(close(c.hol_tokens, 4_050.0));
        assert!(close(c.total, 12_150.0));
        // A pull: B_w = 0.2 s, the 0.55 s pull is 0.35 s exposed, S_w = 0.3 s,
        // a = 1/s: HOL = 13,500 x 1 x 0.3 x (0.2 + 0.15) = 1,417.5 (the pull
        // time is not added to B).
        let w = TradeWaitTerms {
            wait_s: 0.2,
            service_s: 0.3,
        };
        let c = trade_cost_full(&p1, 0, 1_000, 100_000, 0, Some(w), 1.0);
        assert!(close(c.exposed_pull_s, 0.35));
        assert!(close(c.hol_tokens, 1_417.5));
        assert!(close(c.total, 13_500.0 * (0.2 + 0.3 + 0.35) + 1_417.5));
        // Off: mu = 0 or a = 0 adds nothing; a_w is still recorded.
        let c = trade_cost_full(&params(0.0), 27_000, 13_500, 0, 0, None, 2.0);
        assert_eq!(c.hol_tokens, 0.0);
        assert_eq!(c.arrival_rate_per_s, 2.0);
        assert!(close(c.total, 40_500.0));
        let c = trade_cost_full(&p, 27_000, 13_500, 0, 0, None, 0.0);
        assert_eq!(c.hol_tokens, 0.0);
        assert!(close(c.total, 40_500.0));
        // select_trade uses the a_w of each input: a 74,250 > b 53,500.
        let inputs = [
            TradeInput {
                arrival_rate_per_s: 2.0,
                ..input("http://a", 27_000, 0, 13_500, 0)
            },
            input("http://b", 40_000, 0, 13_500, 0),
        ];
        let s = select_trade(&p, &inputs);
        assert!(close(s.costs[0].total, 74_250.0));
        assert_eq!(s.costs[1].hol_tokens, 0.0);
        assert_eq!(s.chosen, 1);
        assert_eq!(
            select_trade(&params(0.0), &inputs).chosen,
            0,
            "40,500 < 53,500"
        );
    }

    /// hol_weight = 0: every cost term and every selection equal the TRADE
    /// before the HOL term (`pre_hol`) bit for bit, with wait ledger and eft
    /// (and eft fallbacks), both dup charges and the noise margin on and off,
    /// although every P has an arrival rate. 4,000 pseudo-random candidate
    /// sets with coarse values, so ties occur.
    #[test]
    fn hol_weight_zero_is_the_selection_without_hol() {
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = |m: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % m
        };
        let workers = ["http://a", "http://b", "http://c", "http://d"];
        let (mut eft_rounds, mut margin_rounds) = (0usize, 0usize);
        for round in 0..4_000 {
            let n = 1 + next(4) as usize;
            let eft = next(2) == 1;
            let inputs: Vec<TradeInput<'_>> = (0..n)
                .map(|i| {
                    // With wait eft, some P use the ledger terms (fallback).
                    let eft_row = eft && next(4) != 0;
                    let wait_s = next(5) as f64 * 0.75;
                    let service_s = 0.05 + next(4) as f64 * 0.25;
                    TradeInput {
                        worker: workers[i],
                        backlog: u128::from(next(4) * 10_000),
                        local_hit: next(3) * 8_192,
                        uncached: 1 + next(3) * 4_000,
                        pulled: [0, 0, 512, 18_000][next(4) as usize],
                        wait: eft_row.then_some(TradeWaitTerms { wait_s, service_s }),
                        arrival_rate_per_s: next(5) as f64 * 0.7,
                        copy_lambda: None,
                        bound_s: None,
                    }
                })
                .collect();
            let p = TradeParams {
                copy_lambda: [0.0, 9.2][next(2) as usize],
                wait: TradeWait::Ledger,
                dup_charge: [TradeDupCharge::Pulled, TradeDupCharge::All][next(2) as usize],
                margin_noise_k: [0.0, 0.5, 1.0][next(3) as usize],
                hol_rate_window_s: [1.0, 60.0][next(2) as usize],
                ..TradeParams::default()
            };
            assert_eq!(p.hol_weight, 0.0);
            let new = select_trade(&p, &inputs);
            let old = pre_hol::select_trade(&p, &inputs);
            assert_eq!(
                (
                    new.chosen,
                    new.owner,
                    new.has_owner,
                    new.best,
                    new.margin_protected
                ),
                (
                    old.chosen,
                    old.owner,
                    old.has_owner,
                    old.best,
                    old.margin_protected
                ),
                "round {round}"
            );
            assert_eq!(
                new.margin_s.map(f64::to_bits),
                old.margin_s.map(f64::to_bits)
            );
            assert_eq!(
                new.owner_gain_s.map(f64::to_bits),
                old.owner_gain_s.map(f64::to_bits)
            );
            eft_rounds += usize::from(eft);
            margin_rounds += usize::from(new.margin_s.is_some());
            assert_eq!(new.costs.len(), old.costs.len());
            for (i, (a, b)) in new.costs.iter().zip(&old.costs).enumerate() {
                let pairs = [
                    ("backlog", a.backlog, b.backlog),
                    ("uncached", a.uncached, b.uncached),
                    ("pull_s", a.pull_s, b.pull_s),
                    ("exposed_s", a.exposed_pull_s, b.exposed_pull_s),
                    ("exposed", a.exposed_pull_tokens, b.exposed_pull_tokens),
                    ("copy", a.copy_tokens, b.copy_tokens),
                    ("total", a.total, b.total),
                    ("wait_s", a.wait_s, b.wait_s),
                    ("service_s", a.service_s, b.service_s),
                ];
                for (name, x, y) in pairs {
                    assert_eq!(x.to_bits(), y.to_bits(), "{name}: round {round}, P {i}");
                }
                assert_eq!(a.dup_tokens, b.dup_tokens, "round {round}, P {i}");
                assert_eq!(a.hol_tokens, 0.0, "round {round}, P {i}");
                assert_eq!(a.arrival_rate_per_s, inputs[i].arrival_rate_per_s);
            }
        }
        assert!(
            eft_rounds > 0 && margin_rounds > 0,
            "{eft_rounds} {margin_rounds}"
        );
    }

    /// The other wait sources select exactly as before wait e3 existed: a
    /// copy of select_trade without the e3 branch ([`pre_e3`]) on random
    /// inputs with every option (ledger and eft waits, dup charge, noise
    /// margin, HOL), and with gate settings that must have no effect.
    #[test]
    fn other_wait_sources_select_as_before_e3() {
        let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = |m: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % m
        };
        let workers = ["http://a", "http://b", "http://c", "http://d"];
        for _ in 0..3_000 {
            let n = 1 + next(4) as usize;
            let eft = next(2) == 1;
            let inputs: Vec<TradeInput<'_>> = (0..n)
                .map(|i| {
                    let eft_row = eft && next(4) != 0;
                    let wait_s = next(5) as f64 * 0.75;
                    let service_s = 0.05 + next(4) as f64 * 0.25;
                    TradeInput {
                        worker: workers[i],
                        backlog: u128::from(next(4) * 10_000),
                        local_hit: next(3) * 8_192,
                        uncached: 1 + next(3) * 4_000,
                        pulled: [0, 0, 512, 18_000][next(4) as usize],
                        wait: eft_row.then_some(TradeWaitTerms { wait_s, service_s }),
                        arrival_rate_per_s: next(5) as f64 * 0.7,
                        copy_lambda: None,
                        bound_s: [None, Some(0.3)][next(2) as usize],
                    }
                })
                .collect();
            let p = TradeParams {
                copy_lambda: [0.0, 9.2][next(2) as usize],
                wait: TradeWait::Ledger,
                dup_charge: [TradeDupCharge::Pulled, TradeDupCharge::All][next(2) as usize],
                margin_noise_k: [0.0, 0.5, 1.0][next(3) as usize],
                hol_weight: [0.0, 0.5, 1.0][next(3) as usize],
                hol_rate_window_s: [1.0, 60.0][next(2) as usize],
                gate_m: [0.0, 1.0, 3.0][next(3) as usize],
                gate_gmin_s: [0.0, 0.5][next(2) as usize],
                ..TradeParams::default()
            };
            let new = select_trade(&p, &inputs);
            let old = pre_e3::select_trade(&p, &inputs);
            assert_eq!(
                (
                    new.chosen,
                    new.owner,
                    new.best,
                    new.has_owner,
                    new.margin_protected
                ),
                (
                    old.chosen,
                    old.owner,
                    old.best,
                    old.has_owner,
                    old.margin_protected
                ),
                "{p:?} {inputs:?}"
            );
            assert_eq!(
                new.margin_s.map(f64::to_bits),
                old.margin_s.map(f64::to_bits)
            );
            assert_eq!(
                new.owner_gain_s.map(f64::to_bits),
                old.owner_gain_s.map(f64::to_bits)
            );
            assert_eq!(new.protect_reason, None);
            for (a, b) in new.costs.iter().zip(&old.costs) {
                assert_eq!(a.total.to_bits(), b.total.to_bits());
                assert_eq!(a.wait_s.to_bits(), b.wait_s.to_bits());
                assert_eq!(a.hol_tokens.to_bits(), b.hol_tokens.to_bits());
            }
        }
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
        let lam = auto_lambda(TradeAutoQ::Sat, 1.0);
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
                dup_charge: [TradeDupCharge::Pulled, TradeDupCharge::All][next(2) as usize],
                margin_noise_k: [0.0, 1.0][next(2) as usize],
                hol_weight: [0.0, 1.0][next(2) as usize],
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
                    arrival_rate_per_s: next(5) as f64 * 0.7,
                    copy_lambda: None,
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
