// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! `place()` - the symmetric P/D leg-placement function.
//!
//! ```text
//! score(p, d) = w_hit      * reuse_gain(p)
//!             - w_backlog  * backlog(p)
//!             - w_role     * (role_pen(p, prefill) + role_pen(d, decode))
//!             - w_xfer     * xfer_cost(p, d)
//!             - w_dload    * decode_load(d)          (router addition, see below)
//! ```
//!
//! evaluated over every `(p, d)` pair of capable workers; the argmax is the
//! placement, and the request is **Local** iff `p == d`.
//!
//! Term definitions (all dimensionless so the weights are comparable):
//!
//! * `reuse_gain(p)`  = `reusable(p) / prompt_len`, where `reusable(p)` is the
//!   indexer's cached-prefix estimate for `p` (`max(gpu, host *
//!   host_hit_factor)`), clipped to `prompt_len`. Cache affinity is the
//!   dominant default weight.
//! * `backlog(p)`     = `backlog_uncached_tokens(p) / backlog_norm_tokens`
//!   (pending uncached prefill tokens).
//! * `role_pen(w, r)` = `1 - weight_r(w)` for a positive weight (clipped to
//!   `[0, 1]`). A weight of **exactly 0 is a hard exclusion**: the worker is
//!   never used for that leg. This is what lets the static 4+4 configuration
//!   reproduce role-locked PD bit-for-bit instead of "mostly".
//! * `xfer_cost(p, d)` = `0` when `p == d`, else
//!   `xfer_fixed_cost + prompt_len / xfer_norm_tokens` (a Split pulls the
//!   whole prompt's KV from the owner).
//! * `decode_load(d)` = `(running_decode(d) + waiting(d)) / decode_norm_reqs`.
//!   Not in the base formula; added so the D-leg is spread by decode
//!   occupancy instead of falling through to the lexical tie-break (the
//!   `loads` input carries `running_decode`/`waiting` for exactly
//!   this). Set `w_dload = 0` to recover the literal formula.
//!
//! Tie-break (deterministic): higher score, then Local over Split, then the
//! lexically smaller `p`, then the lexically smaller `d`.
//!
//! # P-leg / D-leg selection modes
//!
//! Under saturation the additive score degenerates: the backlog term is
//! hundreds of units (millions of pending tokens / `backlog_norm_tokens`)
//! while `reuse_gain <= 1`, so the P-leg becomes least-backlog routing and
//! session affinity collapses (61.8 % of consecutive turns switched prefill
//! worker vs 23 % for `cache_aware_zmq`; cache/prompt 26 % vs 59 %).
//! `PlacementConfig::p_select` therefore defaults to [`PSelect::CacheFirst`],
//! which replicates the `cache_aware_zmq` prefill selection step by step
//! over the prefill-capable candidates:
//!
//! 1. imbalance fast path on `active_prefill_reqs` (in-flight P-legs the
//!    router dispatched, the analogue of `Worker::active_load()`): if
//!    `max - min > balance_abs_reqs && max > min * balance_rel`, pick the
//!    least-loaded worker;
//! 2. else if the best `reuse_gain > cache_threshold`, pick the least-loaded
//!    worker among those with that best hit;
//! 3. else the least-loaded worker.
//!
//! "Least-loaded" = min `(active_prefill_reqs, backlog_uncached_tokens, id)`.
//! The D-leg is then chosen by the score restricted to that P
//! ([`DSelect::Score`]) or by round robin over the decode-capable workers
//! ([`DSelect::RoundRobin`], `transfer_seed % n` - the runtime mints the seed
//! from a monotonic counter - replicating `--pd-decode-policy round-robin`),
//! or by the same three steps over the decode-capable workers with
//! `active_decode_reqs` (D-legs dispatched and not finished, as the SGLang
//! Model Gateway counts load) as the load ([`DSelect::CacheFirst`]).
//! [`PSelect::Score`] / [`DSelect::Score`] keep the literal formula.
//! [`PSelect::CostFunction`] takes the one P that the runtime's placement cost
//! function reserved before the call (the runtime leaves no other prefill
//! candidate). The chosen path is reported as the `p_reason` score term
//! (`P_REASON_*`) and `d_select` (`D_SELECT_*`).
//!
//! `transfer_id` is minted only for a Split placement, from
//! `PlacementRequest::transfer_seed` (the caller supplies a fresh u64; the
//! function itself has no state and no randomness).
//!
//! Placement is pure.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::policy::{CostModel, DefaultCostModel, PairCostInput};

/// Which legs a worker is able to execute ("Capability").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Capabilities {
    pub prefill: bool,
    pub decode: bool,
}

impl Capabilities {
    pub const BOTH: Capabilities = Capabilities {
        prefill: true,
        decode: true,
    };
    pub const PREFILL_ONLY: Capabilities = Capabilities {
        prefill: true,
        decode: false,
    };
    pub const DECODE_ONLY: Capabilities = Capabilities {
        prefill: false,
        decode: true,
    };
}

impl Default for Capabilities {
    /// Default an unqualified worker to both compute capabilities.
    ///
    /// # Inputs
    /// None.
    ///
    /// # Outputs
    /// `Capabilities::BOTH`; role preferences are configured separately.
    fn default() -> Self {
        Self::BOTH
    }
}

/// Primary-role weights of one worker ("Primary role"): the
/// router's preference, not engine state. `0.0` = hard exclusion for that leg.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RoleWeight {
    pub prefill: f64,
    pub decode: f64,
}

impl Default for RoleWeight {
    /// Create equally weighted prefill and decode role preferences.
    ///
    /// # Inputs
    /// None.
    ///
    /// # Outputs
    /// A role weight of `1.0` for each leg; neither role is excluded.
    fn default() -> Self {
        Self {
            prefill: 1.0,
            decode: 1.0,
        }
    }
}

impl RoleWeight {
    /// Today's prefill worker: `{prefill: 1, decode: 0}`.
    pub const PREFILL_LEANING: RoleWeight = RoleWeight {
        prefill: 1.0,
        decode: 0.0,
    };
    /// Today's decode worker: `{prefill: 0, decode: 1}`.
    pub const DECODE_LEANING: RoleWeight = RoleWeight {
        prefill: 0.0,
        decode: 1.0,
    };
    /// A worker draining before it takes its new role: no new leg of either kind.
    pub const DRAINING: RoleWeight = RoleWeight {
        prefill: 0.0,
        decode: 0.0,
    };
}

/// Per-worker weights + capabilities. The key set IS the candidate pool.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RoleWeights {
    pub workers: BTreeMap<String, WorkerRole>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WorkerRole {
    pub weight: RoleWeight,
    #[serde(default)]
    pub capabilities: Capabilities,
}

impl RoleWeights {
    /// Create an empty placement candidate map.
    ///
    /// # Inputs
    /// None.
    ///
    /// # Outputs
    /// A new `RoleWeights` with no workers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a symmetric worker in the candidate map.
    ///
    /// # Inputs
    /// `self`: mutable candidate map; `worker`: identifier/URL; `weight`: P/D role preferences.
    ///
    /// # Outputs
    /// The same mutable map for chaining; the entry is stored with both capabilities.
    pub fn insert(&mut self, worker: impl Into<String>, weight: RoleWeight) -> &mut Self {
        self.workers.insert(
            worker.into(),
            WorkerRole {
                weight,
                capabilities: Capabilities::BOTH,
            },
        );
        self
    }

    /// Insert or replace a worker with explicit compute capabilities.
    ///
    /// # Inputs
    /// `self`: mutable map; `worker`: identifier/URL; `weight`: role preferences;
    /// `capabilities`: permitted compute legs.
    ///
    /// # Outputs
    /// The same mutable map for chaining, after replacing the specified worker entry.
    pub fn insert_with_caps(
        &mut self,
        worker: impl Into<String>,
        weight: RoleWeight,
        capabilities: Capabilities,
    ) -> &mut Self {
        self.workers.insert(
            worker.into(),
            WorkerRole {
                weight,
                capabilities,
            },
        );
        self
    }
}

/// Reusable cached-prefix tokens of one worker for THIS request, by tier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HitTokens {
    pub gpu: u64,
    pub host: u64,
}

/// `worker -> HitTokens` (from the dynamo indexer `/query`; empty = fail-open).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexerHits {
    pub workers: BTreeMap<String, HitTokens>,
    /// Expected usable hit after a validated peer pull, per destination.
    /// An absent entry means local hit only. This never reports actual residency.
    #[serde(default)]
    pub expected_after_pull: BTreeMap<String, u64>,
}

impl IndexerHits {
    /// Create an empty request-specific cache-hit view.
    ///
    /// # Inputs
    /// None.
    ///
    /// # Outputs
    /// An `IndexerHits` with no local hits or expected peer-pull hits.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record or replace local cache-hit estimates for one worker.
    ///
    /// # Inputs
    /// `self`: mutable request hit map; `worker`: identifier/URL; `hit`: GPU and host prefix
    /// lengths in tokens.
    ///
    /// # Outputs
    /// The same mutable map for chaining; expected peer-pull hits are unchanged.
    pub fn insert(&mut self, worker: impl Into<String>, hit: HitTokens) -> &mut Self {
        self.workers.insert(worker.into(), hit);
        self
    }

    /// Read a worker's local cache-hit estimates.
    ///
    /// # Inputs
    /// `self`: request hit map; `worker`: identifier/URL to look up.
    ///
    /// # Outputs
    /// Copied GPU/host hit counts, or zero counts when the worker has no entry.
    pub fn get(&self, worker: &str) -> HitTokens {
        self.workers.get(worker).copied().unwrap_or_default()
    }
}

/// Load view of one worker (`loads`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerLoad {
    /// Pending uncached prefill tokens (engine `/v1/loads`
    /// `num_waiting_uncached_tokens` + router ledger).
    pub backlog_uncached_tokens: u64,
    pub running_decode: u64,
    pub waiting: u64,
    /// P-legs this router dispatched to the worker whose prefill has not
    /// resolved yet (ledger entries alive). The `cache_first` load signal.
    #[serde(default)]
    pub active_prefill_reqs: u64,
    /// D-legs this router dispatched to the worker that have not finished,
    /// including those still waiting for KV space on the D: the
    /// [`DSelect::CacheFirst`] load, counted like the SGLang Model Gateway's
    /// per-worker load (dispatch to end of request).
    #[serde(default)]
    pub active_decode_reqs: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerLoads {
    pub workers: BTreeMap<String, WorkerLoad>,
}

impl WorkerLoads {
    /// Create an empty placement load snapshot.
    ///
    /// # Inputs
    /// None.
    ///
    /// # Outputs
    /// A new `WorkerLoads` with no observations.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record or replace a worker's load in a snapshot.
    ///
    /// # Inputs
    /// `self`: mutable load map; `worker`: identifier/URL; `load`: backlog and request counts.
    ///
    /// # Outputs
    /// The same mutable map for chaining; this does not reserve work in Backend.
    pub fn insert(&mut self, worker: impl Into<String>, load: WorkerLoad) -> &mut Self {
        self.workers.insert(worker.into(), load);
        self
    }

    /// Read a worker's load from the supplied snapshot.
    ///
    /// # Inputs
    /// `self`: load map; `worker`: identifier/URL.
    ///
    /// # Outputs
    /// A copied observation, or an all-zero default when absent; no engine query occurs.
    pub fn get(&self, worker: &str) -> WorkerLoad {
        self.workers.get(worker).copied().unwrap_or_default()
    }
}

/// The request as the placement sees it (`req`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacementRequest {
    /// Page hashes of the prompt (frontend `/prepare`); informational to
    /// `place()` itself - the indexer lookup happens before the call.
    #[serde(default)]
    pub hashes: Vec<u64>,
    pub prompt_len: u64,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub turn: Option<u32>,
    /// Client request id, echoed in logs.
    #[serde(default)]
    pub rid: Option<String>,
    /// Fresh u64 the caller mints (atomic counter or RNG); becomes the
    /// `transfer_id` iff the placement is a Split.
    pub transfer_seed: u64,
    /// Decision replay (`--symmetric-pd-decision-replay`): the `(p_url, d_url)`
    /// pair a reference run chose for this session position. Honoured when
    /// both are eligible candidates (and distinct under `force_split`);
    /// otherwise the configured selection runs as if absent.
    #[serde(default)]
    pub pin: Option<(String, String)>,
}

/// Weights of the score terms.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScoreWeights {
    pub w_hit: f64,
    pub w_backlog: f64,
    pub w_role: f64,
    pub w_xfer: f64,
    pub w_dload: f64,
}

impl Default for ScoreWeights {
    /// Cache affinity dominates: a full-prefix hit (reuse_gain = 1) outweighs
    /// a whole `backlog_norm_tokens` of backlog, a full role penalty on one
    /// leg, and a transfer of one `xfer_norm_tokens` prompt combined.
    /// Create the original additive placement-score coefficients.
    ///
    /// # Inputs
    /// None.
    ///
    /// # Outputs
    /// Weights for hit, backlog, roles, transfer and decode load: `1.0`, `0.3`, `0.3`, `0.2`,
    /// `0.1`.
    fn default() -> Self {
        Self {
            w_hit: 1.0,
            w_backlog: 0.3,
            w_role: 0.3,
            w_xfer: 0.2,
            w_dload: 0.1,
        }
    }
}

/// How the P-leg is chosen (module docs, "selection modes").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PSelect {
    /// Argmax of the additive score over all `(p, d)` pairs.
    Score,
    /// `cache_aware_zmq` semantics: imbalance fast path, then best hit above
    /// `cache_threshold`, then least-loaded.
    #[default]
    CacheFirst,
    /// The runtime's placement cost function (`--symmetric-pd-p-select
    /// cost-function`) reserved one P before placement and left it as the
    /// only prefill candidate; the placement takes that P
    /// ([`P_REASON_COST_FUNCTION`]) and chooses the D-leg by `d_select`.
    CostFunction,
}

/// How the D-leg is chosen once the P-leg is fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DSelect {
    /// Argmax of the score over `(p, d)` for the chosen `p` (Local allowed
    /// unless `force_split`).
    #[default]
    Score,
    /// `transfer_seed % n` over the decode-capable workers (minus `p` under
    /// `force_split`): today's `--pd-decode-policy round-robin`.
    RoundRobin,
    /// The `CacheFirst` rule over the decode-capable workers (minus `p` under
    /// `force_split`), with the router's in-flight D-legs
    /// (`WorkerLoad::active_decode_reqs`) as the load.
    CacheFirst,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlacementConfig {
    #[serde(default)]
    pub weights: ScoreWeights,
    /// Forbid `p == d` (today's PD as a configuration).
    #[serde(default)]
    pub force_split: bool,
    #[serde(default)]
    pub p_select: PSelect,
    #[serde(default)]
    pub d_select: DSelect,
    /// `cache_first`: best `reuse_gain` must exceed this to win on cache
    /// (`cache_aware_zmq --cache-threshold`, default 0.5).
    #[serde(default = "default_cache_threshold")]
    pub cache_threshold: f64,
    /// `cache_first` imbalance fast path: `max - min` of `active_prefill_reqs`
    /// must exceed this (`--balance-abs-threshold`, default 32) ...
    #[serde(default = "default_balance_abs_reqs")]
    pub balance_abs_reqs: u64,
    /// ... and `max > min * balance_rel` (`--balance-rel-threshold`, 1.1).
    #[serde(default = "default_balance_rel")]
    pub balance_rel: f64,
    /// Tokens of backlog that count as one unit of `backlog(p)`.
    #[serde(default = "default_backlog_norm_tokens")]
    pub backlog_norm_tokens: u64,
    /// Prompt tokens that count as one unit of transfer cost.
    #[serde(default = "default_xfer_norm_tokens")]
    pub xfer_norm_tokens: u64,
    /// Constant part of `xfer_cost` for any Split (handshake + READ setup).
    #[serde(default = "default_xfer_fixed_cost")]
    pub xfer_fixed_cost: f64,
    /// Requests (running + waiting) that count as one unit of `decode_load`.
    #[serde(default = "default_decode_norm_reqs")]
    pub decode_norm_reqs: u64,
    /// How much a host-tier hit is worth relative to a GPU-tier hit.
    #[serde(default = "default_host_hit_factor")]
    pub host_hit_factor: f64,
}

/// Supply the default backlog normalization scale.
///
/// # Inputs
/// None; also used by Serde when its configuration field is absent.
///
/// # Outputs
/// 10,240 pending uncached tokens per score unit.
fn default_backlog_norm_tokens() -> u64 {
    10_240
}
/// Supply the default transfer normalization scale.
///
/// # Inputs
/// None; also used by Serde when its configuration field is absent.
///
/// # Outputs
/// 16,384 prompt tokens per transfer-cost unit.
fn default_xfer_norm_tokens() -> u64 {
    16_384
}
/// Supply the default fixed split-transfer penalty.
///
/// # Inputs
/// None; also used by Serde when its configuration field is absent.
///
/// # Outputs
/// A dimensionless penalty of `0.25`.
fn default_xfer_fixed_cost() -> f64 {
    0.25
}
/// Supply the default decode-load normalization scale.
///
/// # Inputs
/// None; also used by Serde when its configuration field is absent.
///
/// # Outputs
/// 64 running/waiting requests per score unit.
fn default_decode_norm_reqs() -> u64 {
    64
}
/// Supply the default weight of host-resident cache hits.
///
/// # Inputs
/// None; also used by Serde when its configuration field is absent.
///
/// # Outputs
/// A host-to-GPU hit multiplier of `1.0`.
fn default_host_hit_factor() -> f64 {
    1.0
}
/// Supply the strict cache-first hit-ratio threshold.
///
/// # Inputs
/// None; also used by Serde when its configuration field is absent.
///
/// # Outputs
/// A hit ratio of `0.5`.
fn default_cache_threshold() -> f64 {
    0.5
}
/// Supply the absolute cache-first imbalance threshold.
///
/// # Inputs
/// None; also used by Serde when its configuration field is absent.
///
/// # Outputs
/// A difference of 32 active prefill requests.
fn default_balance_abs_reqs() -> u64 {
    32
}
/// Supply the relative cache-first imbalance threshold.
///
/// # Inputs
/// None; also used by Serde when its configuration field is absent.
///
/// # Outputs
/// A maximum/minimum active-request multiplier of `1.1`.
fn default_balance_rel() -> f64 {
    1.1
}

impl Default for PlacementConfig {
    /// Build the complete default placement configuration.
    ///
    /// # Inputs
    /// None.
    ///
    /// # Outputs
    /// Cache-first P selection, score-based D selection, Local allowed, and the documented
    /// score, owner, and normalization defaults.
    fn default() -> Self {
        Self {
            weights: ScoreWeights::default(),
            force_split: false,
            p_select: PSelect::default(),
            d_select: DSelect::default(),
            cache_threshold: default_cache_threshold(),
            balance_abs_reqs: default_balance_abs_reqs(),
            balance_rel: default_balance_rel(),
            backlog_norm_tokens: default_backlog_norm_tokens(),
            xfer_norm_tokens: default_xfer_norm_tokens(),
            xfer_fixed_cost: default_xfer_fixed_cost(),
            decode_norm_reqs: default_decode_norm_reqs(),
            host_hit_factor: default_host_hit_factor(),
        }
    }
}

/// The placement decision (output).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Placement {
    pub p_leg: String,
    pub d_leg: String,
    /// `Some` iff Split (`p_leg != d_leg`).
    pub transfer_id: Option<u64>,
    /// Every term of the winning pair's score, raw (unweighted) values plus
    /// the weighted total under `"score"`. Logged per request.
    pub score_terms: Vec<(String, f64)>,
}

impl Placement {
    /// Determine whether both compute legs use the same worker.
    ///
    /// # Inputs
    /// `self`: completed placement.
    ///
    /// # Outputs
    /// `true` when `p_leg == d_leg`, otherwise `false`.
    pub fn is_local(&self) -> bool {
        self.p_leg == self.d_leg
    }

    /// Look up one recorded placement-score term.
    ///
    /// # Inputs
    /// `self`: placement diagnostics; `name`: exact score-term key.
    ///
    /// # Outputs
    /// The first matching value, or `None` if the term was not recorded.
    pub fn term(&self, name: &str) -> Option<f64> {
        self.score_terms
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| *v)
    }
}

/// Why `place()` produced nothing.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PlacementError {
    #[error(
        "symmetric_pd: no worker can run the P-leg (no prefill-capable worker with weight > 0)"
    )]
    NoPrefillCandidate,
    #[error(
        "symmetric_pd: no worker can run the D-leg (no decode-capable worker with weight > 0)"
    )]
    NoDecodeCandidate,
    #[error("symmetric_pd: force_split with a single worker leaves no (p, d) pair")]
    NoPairUnderForceSplit,
    #[error("symmetric_pd: invalid policy output: {0}")]
    InvalidPolicyOutput(String),
}

/// Names of the score terms, in the order they appear in `score_terms`.
pub const TERM_HIT_TOKENS: &str = "hit_tokens";
pub const TERM_EXPECTED_HIT_TOKENS: &str = "expected_hit_tokens";
pub const TERM_REUSE_GAIN: &str = "reuse_gain";
pub const TERM_BACKLOG: &str = "backlog";
pub const TERM_ROLE_PEN_P: &str = "role_pen_p";
pub const TERM_ROLE_PEN_D: &str = "role_pen_d";
pub const TERM_XFER_COST: &str = "xfer_cost";
pub const TERM_DECODE_LOAD: &str = "decode_load";
/// Which `cache_first` branch chose the P-leg (`P_REASON_*` code).
pub const TERM_P_REASON: &str = "p_reason";
/// How the D-leg was chosen (`D_SELECT_*` code).
pub const TERM_D_SELECT: &str = "d_select";
pub const TERM_SCORE: &str = "score";

pub const P_REASON_SCORE: f64 = 0.0;
pub const P_REASON_CACHE: f64 = 1.0;
pub const P_REASON_IMBALANCE: f64 = 2.0;
pub const P_REASON_MIN_LOAD: f64 = 3.0;
/// P-leg pinned by `PlacementRequest::pin` (decision replay).
pub const P_REASON_REPLAY: f64 = 4.0;
/// P-leg reserved by the runtime's placement cost function ([`PSelect::CostFunction`]).
pub const P_REASON_COST_FUNCTION: f64 = 6.0;
pub const D_SELECT_SCORE: f64 = 0.0;
pub const D_SELECT_ROUND_ROBIN: f64 = 1.0;
/// D-leg pinned by `PlacementRequest::pin` (decision replay).
pub const D_SELECT_REPLAY: f64 = 2.0;
/// `DSelect::CacheFirst` branches, as `P_REASON_CACHE` / `_IMBALANCE` / `_MIN_LOAD`.
pub const D_SELECT_CACHE: f64 = 3.0;
pub const D_SELECT_IMBALANCE: f64 = 4.0;
pub const D_SELECT_MIN_LOAD: f64 = 5.0;

/// Human name of a `TERM_P_REASON` value (for logs).
/// Translate a recorded P-selection reason into its log label.
///
/// # Inputs
/// `code`: numeric `TERM_P_REASON` value, cast to an integer for lookup.
///
/// # Outputs
/// A static label (`score`, `cache`, `imbalance`, `min_load`, `replay` or `cost_function`),
/// or `unknown`.
pub fn p_reason_name(code: f64) -> &'static str {
    match code as i64 {
        0 => "score",
        1 => "cache",
        2 => "imbalance",
        3 => "min_load",
        4 => "replay",
        6 => "cost_function",
        _ => "unknown",
    }
}

/// Human name of a `TERM_D_SELECT` value (for logs).
/// Translate a recorded D-selection mode into its log label.
///
/// # Inputs
/// `code`: numeric `TERM_D_SELECT` value, cast to an integer for lookup.
///
/// # Outputs
/// A static `score`, `round_robin`, or `replay` label, or `unknown`.
pub fn d_select_name(code: f64) -> &'static str {
    match code as i64 {
        0 => "score",
        1 => "round_robin",
        2 => "replay",
        3 => "cache",
        4 => "imbalance",
        5 => "min_load",
        _ => "unknown",
    }
}

struct Candidate<'a> {
    id: &'a str,
    role: WorkerRole,
    hit_tokens: f64,
    expected_hit: u64,
    reuse_gain: f64,
    backlog: f64,
    backlog_tokens: u64,
    active_reqs: u64,
    decode_load: f64,
    /// In-flight D-legs (`WorkerLoad::active_decode_reqs`): the
    /// `DSelect::CacheFirst` load signal.
    decode_reqs: u64,
}

impl<'a> Candidate<'a> {
    /// `cache_aware_zmq` "min load" with deterministic tie-breaks.
    /// Build the cache-first least-load ordering key.
    ///
    /// # Inputs
    /// `self`: one evaluated candidate.
    ///
    /// # Outputs
    /// Active prefill count, raw backlog tokens, and worker ID, ordered from most significant
    /// to least significant.
    fn load_key(&self) -> (u64, u64, &'a str) {
        (self.active_reqs, self.backlog_tokens, self.id)
    }
}

/// `cache_aware_zmq` prefill selection over the prefill-capable pool
/// (non-empty). Returns the worker and the `P_REASON_*` code.
/// Apply imbalance protection, cache affinity, then deterministic least-load selection.
///
/// # Inputs
/// `pool`: nonempty eligible P candidates with precomputed hits/loads; `cfg`: absolute/relative
/// imbalance and strict cache-hit thresholds.
///
/// # Outputs
/// The selected borrowed candidate and numeric P reason. No state changes; an empty pool
/// violates the caller contract and panics.
fn select_p_cache_first<'a, 'c>(
    pool: &[&'c Candidate<'a>],
    cfg: &PlacementConfig,
) -> (&'c Candidate<'a>, f64) {
    let min_load = |cands: &[&'c Candidate<'a>]| -> &'c Candidate<'a> {
        cands
            .iter()
            .copied()
            .min_by_key(|c| c.load_key())
            .expect("non-empty pool")
    };
    let (min_l, max_l) = pool.iter().fold((u64::MAX, 0u64), |(mn, mx), c| {
        (mn.min(c.active_reqs), mx.max(c.active_reqs))
    });
    let imbalanced = max_l.saturating_sub(min_l) > cfg.balance_abs_reqs
        && (max_l as f64) > (min_l as f64) * cfg.balance_rel;
    if imbalanced {
        return (min_load(pool), P_REASON_IMBALANCE);
    }
    let best_hit = pool.iter().map(|c| c.hit_tokens).fold(0.0_f64, f64::max);
    let best_gain = pool.iter().map(|c| c.reuse_gain).fold(0.0_f64, f64::max);
    if best_hit > 0.0 && best_gain > cfg.cache_threshold {
        let matched: Vec<&'c Candidate<'a>> = pool
            .iter()
            .copied()
            .filter(|c| c.hit_tokens == best_hit)
            .collect();
        return (min_load(&matched), P_REASON_CACHE);
    }
    (min_load(pool), P_REASON_MIN_LOAD)
}

/// Round robin over the decode-capable pool by `seed % n`; under
/// `force_split` the pool excludes `p`. `None` when nothing is left.
/// Choose a D candidate using a caller-maintained round-robin seed.
///
/// # Inputs
/// `pool`: ordered eligible D candidates; `p`: chosen P; `force_split`: whether to exclude P
/// from D; `seed`: round-robin counter value.
///
/// # Outputs
/// The candidate at `seed % eligible_count`, or `None` if exclusion leaves no D. This function
/// does not advance a counter.
fn select_d_round_robin<'a, 'c>(
    pool: &[&'c Candidate<'a>],
    p: &Candidate<'a>,
    force_split: bool,
    seed: u64,
) -> Option<&'c Candidate<'a>> {
    let eligible: Vec<&'c Candidate<'a>> = pool
        .iter()
        .copied()
        .filter(|d| !(force_split && d.id == p.id))
        .collect();
    if eligible.is_empty() {
        return None;
    }
    Some(eligible[(seed % eligible.len() as u64) as usize])
}

/// [`select_p_cache_first`]'s rule over the decode-capable pool minus
/// `exclude`, by `(decode_reqs, id)`. Returns the worker and the
/// `D_SELECT_*` code; `None` when nothing is left.
fn select_d_cache_first<'a, 'c>(
    pool: &[&'c Candidate<'a>],
    exclude: Option<&str>,
    cfg: &PlacementConfig,
) -> Option<(&'c Candidate<'a>, f64)> {
    let eligible: Vec<&'c Candidate<'a>> = pool
        .iter()
        .copied()
        .filter(|d| Some(d.id) != exclude)
        .collect();
    let min_load =
        |cands: &[&'c Candidate<'a>]| cands.iter().copied().min_by_key(|c| (c.decode_reqs, c.id));
    let (min_l, max_l) = eligible.iter().fold((u64::MAX, 0u64), |(mn, mx), c| {
        (mn.min(c.decode_reqs), mx.max(c.decode_reqs))
    });
    let imbalanced = max_l.saturating_sub(min_l) > cfg.balance_abs_reqs
        && (max_l as f64) > (min_l as f64) * cfg.balance_rel;
    if imbalanced {
        return Some((min_load(&eligible)?, D_SELECT_IMBALANCE));
    }
    let best_hit = eligible
        .iter()
        .map(|c| c.hit_tokens)
        .fold(0.0_f64, f64::max);
    let best_gain = eligible
        .iter()
        .map(|c| c.reuse_gain)
        .fold(0.0_f64, f64::max);
    if best_hit > 0.0 && best_gain > cfg.cache_threshold {
        let matched: Vec<&'c Candidate<'a>> = eligible
            .iter()
            .copied()
            .filter(|c| c.hit_tokens == best_hit)
            .collect();
        return Some((min_load(&matched)?, D_SELECT_CACHE));
    }
    Some((min_load(&eligible)?, D_SELECT_MIN_LOAD))
}

/// Choose the P-leg and D-leg for one request. Pure; see the module docs
/// for the score and tie-break.
/// Choose compute legs with the built-in cost model.
///
/// # Inputs
/// `req`: request size/identity and optional replay pin; `hits`: local/validated pull
/// estimates; `loads`: load snapshot; `weights`: eligible roles/capabilities; `cfg`: selection
/// settings.
///
/// # Outputs
/// `Ok(Placement)` with P/D, optional transfer ID and diagnostics; `Err` if no permitted pair
/// exists. No I/O, reservation, or counter mutation occurs.
pub fn place(
    req: &PlacementRequest,
    hits: &IndexerHits,
    loads: &WorkerLoads,
    weights: &RoleWeights,
    cfg: &PlacementConfig,
) -> Result<Placement, PlacementError> {
    place_with_cost(req, hits, loads, weights, cfg, &DefaultCostModel)
}

/// Existing policy with an injected cost model; refresh and accounting stay in
/// Backend. The model runs on borrowed observations, with no token-body copies.
/// Choose compute legs using existing selection rules and an injected cost model.
///
/// # Inputs
/// `req`, `hits`, `loads`, `weights`, and `cfg`: borrowed request and placement
/// observations/settings; `model`: pure per-P and per-pair cost calculations in consistent
/// units.
///
/// # Outputs
/// `Ok(Placement)` with deterministic tie-breaks and the request seed only for Split; `Err` for
/// missing eligible legs, an impossible Split, or a non-finite pair score. Backend accounting
/// remains unchanged.
pub fn place_with_cost<C: CostModel + ?Sized>(
    req: &PlacementRequest,
    hits: &IndexerHits,
    loads: &WorkerLoads,
    weights: &RoleWeights,
    cfg: &PlacementConfig,
    model: &C,
) -> Result<Placement, PlacementError> {
    let prompt_len = req.prompt_len.max(1) as f64;
    let backlog_norm = cfg.backlog_norm_tokens.max(1) as f64;
    let decode_norm = cfg.decode_norm_reqs.max(1) as f64;
    let xfer_norm = cfg.xfer_norm_tokens.max(1) as f64;

    let candidates: Vec<Candidate> = weights
        .workers
        .iter()
        .map(|(id, role)| {
            let hit = hits.get(id);
            let reusable = (hit.gpu as f64)
                .max(hit.host as f64 * cfg.host_hit_factor)
                .min(prompt_len);
            let load = loads.get(id);
            Candidate {
                id,
                role: *role,
                hit_tokens: reusable,
                expected_hit: hits
                    .expected_after_pull
                    .get(id)
                    .copied()
                    .unwrap_or(reusable as u64)
                    .max(reusable as u64)
                    .min(req.prompt_len),
                reuse_gain: reusable / prompt_len,
                backlog: load.backlog_uncached_tokens as f64 / backlog_norm,
                backlog_tokens: load.backlog_uncached_tokens,
                active_reqs: load.active_prefill_reqs,
                decode_load: (load.running_decode + load.waiting) as f64 / decode_norm,
                decode_reqs: load.active_decode_reqs,
            }
        })
        .collect();

    let p_ok = |c: &Candidate| c.role.capabilities.prefill && c.role.weight.prefill > 0.0;
    let d_ok = |c: &Candidate| c.role.capabilities.decode && c.role.weight.decode > 0.0;
    let p_pool: Vec<&Candidate> = candidates.iter().filter(|c| p_ok(c)).collect();
    let d_pool: Vec<&Candidate> = candidates.iter().filter(|c| d_ok(c)).collect();
    if p_pool.is_empty() {
        return Err(PlacementError::NoPrefillCandidate);
    }
    if d_pool.is_empty() {
        return Err(PlacementError::NoDecodeCandidate);
    }

    // Decision replay: a reference run's (P, D) for this session position
    // wins over both selectors when both legs are eligible here.
    let replay: Option<(&Candidate, &Candidate)> = req.pin.as_ref().and_then(|(pu, du)| {
        let p = p_pool.iter().copied().find(|c| c.id == pu.as_str())?;
        let d = d_pool.iter().copied().find(|c| c.id == du.as_str())?;
        if cfg.force_split && p.id == d.id {
            return None;
        }
        Some((p, d))
    });

    // Pin the P-leg (and possibly the D-leg) before the score loop; the loop
    // then only ranks the remaining free leg (or just reports the terms).
    let (p_pin, p_reason): (Option<&Candidate>, f64) = match (replay, cfg.p_select) {
        (Some((p, _)), _) => (Some(p), P_REASON_REPLAY),
        (None, PSelect::Score) => (None, P_REASON_SCORE),
        (None, PSelect::CacheFirst) => {
            let (p, reason) = select_p_cache_first(&p_pool, cfg);
            (Some(p), reason)
        }
        // The runtime reserved this P in its cost ledger and left it as the
        // only prefill candidate; any other pool is a runtime error.
        (None, PSelect::CostFunction) => match p_pool.as_slice() {
            [p] => (Some(*p), P_REASON_COST_FUNCTION),
            pool => {
                return Err(PlacementError::InvalidPolicyOutput(format!(
                    "cost_function expects exactly one P candidate, got {}",
                    pool.len()
                )))
            }
        },
    };
    let (d_pin, d_select): (Option<&Candidate>, f64) = match (replay, cfg.d_select, p_pin) {
        (Some((_, d)), _, _) => (Some(d), D_SELECT_REPLAY),
        (None, DSelect::Score, _) => (None, D_SELECT_SCORE),
        (None, DSelect::RoundRobin, Some(p)) => (
            Some(
                select_d_round_robin(&d_pool, p, cfg.force_split, req.transfer_seed)
                    .ok_or(PlacementError::NoPairUnderForceSplit)?,
            ),
            D_SELECT_ROUND_ROBIN,
        ),
        // Round robin without a pinned P: the D-leg is drawn from the whole
        // pool; under force_split the loop still skips `p == d`, so a pool of
        // one worker degrades to the score (never to an error).
        (None, DSelect::RoundRobin, None) => {
            let n = d_pool.len() as u64;
            let d = d_pool[(req.transfer_seed % n) as usize];
            if cfg.force_split && d_pool.len() == 1 {
                (None, D_SELECT_SCORE)
            } else {
                (Some(d), D_SELECT_ROUND_ROBIN)
            }
        }
        (None, DSelect::CacheFirst, Some(p)) => {
            let exclude = cfg.force_split.then_some(p.id);
            let (d, code) = select_d_cache_first(&d_pool, exclude, cfg)
                .ok_or(PlacementError::NoPairUnderForceSplit)?;
            (Some(d), code)
        }
        // As round robin without a pinned P: a one-worker pool under
        // force_split degrades to the score.
        (None, DSelect::CacheFirst, None) => {
            if cfg.force_split && d_pool.len() == 1 {
                (None, D_SELECT_SCORE)
            } else {
                let (d, code) = select_d_cache_first(&d_pool, None, cfg)
                    .ok_or(PlacementError::NoDecodeCandidate)?;
                (Some(d), code)
            }
        }
    };
    let same = |pin: Option<&Candidate>, c: &Candidate| pin.is_none_or(|x| x.id == c.id);

    let w = cfg.weights;
    let split_cost = cfg.xfer_fixed_cost + prompt_len / xfer_norm;

    let mut best: Option<(f64, &Candidate, &Candidate, [f64; 6])> = None;
    for p in p_pool.iter().copied().filter(|c| same(p_pin, c)) {
        for d in d_pool.iter().copied().filter(|c| same(d_pin, c)) {
            let local = p.id == d.id;
            if local && cfg.force_split {
                continue;
            }
            let role_pen_p = (1.0 - p.role.weight.prefill).clamp(0.0, 1.0);
            let role_pen_d = (1.0 - d.role.weight.decode).clamp(0.0, 1.0);
            let xfer_cost = if local { 0.0 } else { split_cost };
            let score = model.pair_score(
                PairCostInput {
                    request: req,
                    prefill: p.id,
                    decode: d.id,
                    reuse_gain: p.reuse_gain,
                    backlog: p.backlog,
                    role_penalty_prefill: role_pen_p,
                    role_penalty_decode: role_pen_d,
                    transfer: xfer_cost,
                    decode_load: d.decode_load,
                },
                w,
            );
            if !score.is_finite() {
                return Err(PlacementError::InvalidPolicyOutput(
                    "non-finite pair score".into(),
                ));
            }
            let better = match &best {
                None => true,
                Some((best_score, bp, bd, _)) => {
                    if score != *best_score {
                        score > *best_score
                    } else {
                        let best_local = bp.id == bd.id;
                        if local != best_local {
                            local
                        } else {
                            (p.id, d.id) < (bp.id, bd.id)
                        }
                    }
                }
            };
            if better {
                best = Some((
                    score,
                    p,
                    d,
                    [
                        p.reuse_gain,
                        p.backlog,
                        role_pen_p,
                        role_pen_d,
                        xfer_cost,
                        d.decode_load,
                    ],
                ));
            }
        }
    }

    let Some((score, p, d, terms)) = best else {
        return Err(PlacementError::NoPairUnderForceSplit);
    };
    let local = p.id == d.id;
    let mut placement = Placement {
        p_leg: p.id.to_string(),
        d_leg: d.id.to_string(),
        transfer_id: (!local).then_some(req.transfer_seed),
        score_terms: vec![
            (TERM_HIT_TOKENS.to_string(), p.hit_tokens),
            (TERM_REUSE_GAIN.to_string(), terms[0]),
            (TERM_BACKLOG.to_string(), terms[1]),
            (TERM_ROLE_PEN_P.to_string(), terms[2]),
            (TERM_ROLE_PEN_D.to_string(), terms[3]),
            (TERM_XFER_COST.to_string(), terms[4]),
            (TERM_DECODE_LOAD.to_string(), terms[5]),
            (TERM_P_REASON.to_string(), p_reason),
            (TERM_D_SELECT.to_string(), d_select),
            (TERM_SCORE.to_string(), score),
        ],
    };
    if cfg.p_select == PSelect::CostFunction {
        placement
            .score_terms
            .push((TERM_EXPECTED_HIT_TOKENS.into(), p.expected_hit as f64));
    }
    Ok(placement)
}

#[cfg(test)]
mod tests {
    #[test]
    /// Check the full serialized placement digest matches the pre-extension reference.
    ///
    /// # Inputs
    /// No arguments; evaluates 360 combinations of selection modes, roles, loads, hits, replay
    /// pins and transfer seeds.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn default_policy_frozen_compatibility_matrix() {
        use sha2::{Digest, Sha256};
        let mut digest = Sha256::new();
        for p_select in [PSelect::CacheFirst, PSelect::Score] {
            for d_select in [DSelect::RoundRobin, DSelect::Score] {
                for force_split in [false, true] {
                    for seed in 0..6 {
                        for case in 0..5 {
                            let mut weights = RoleWeights::new();
                            let mut hits = IndexerHits::new();
                            let mut loads = WorkerLoads::new();
                            for i in 0..3 {
                                let id = format!("w{i}");
                                weights.insert(
                                    &id,
                                    RoleWeight {
                                        prefill: if case == 1 && i == 2 { 0.0 } else { 1.0 },
                                        decode: if case == 1 && i == 0 { 0.0 } else { 0.75 },
                                    },
                                );
                                hits.insert(
                                    &id,
                                    HitTokens {
                                        gpu: if case == 0 { 0 } else { i * 2048 },
                                        host: if case == 2 { 8192 } else { 0 },
                                    },
                                );
                                if case == 3 {
                                    hits.expected_after_pull.insert(id.clone(), 6144);
                                }
                                loads.insert(
                                    id,
                                    WorkerLoad {
                                        backlog_uncached_tokens: if case == 0 {
                                            0
                                        } else {
                                            (2 - i) * 16000
                                        },
                                        active_prefill_reqs: if case == 4 { i * 40 } else { i },
                                        running_decode: i * 8,
                                        waiting: 2 - i,
                                        active_decode_reqs: 0,
                                    },
                                );
                            }
                            let cfg = PlacementConfig {
                                p_select,
                                d_select,
                                force_split,
                                ..Default::default()
                            };
                            let request = PlacementRequest {
                                prompt_len: 8192,
                                transfer_seed: seed,
                                pin: (case == 4).then(|| ("w1".into(), "w2".into())),
                                ..Default::default()
                            };
                            let result = place(&request, &hits, &loads, &weights, &cfg).unwrap();
                            digest.update(serde_json::to_vec(&result).unwrap());
                            digest.update(b"\n");
                        }
                    }
                }
            }
        }
        let actual = format!("{:x}", digest.finalize());
        // Captured from place() with the cache-first and score P selections:
        // 240 configurations including RR, replay,
        // local/split, host/pull hits, ties and imbalance. Hash includes every
        // score term and exact floating-point serialization, not just node IDs.
        assert_eq!(
            actual,
            "e47738c1f8cbc13b805f999386e19598ece7fd922a1f26f91d2f304261a7b338"
        );
    }

    use super::*;

    const P: [&str; 4] = ["p0", "p1", "p2", "p3"];
    const D: [&str; 4] = ["d0", "d1", "d2", "d3"];

    /// Construct a minimal placement request for unit tests.
    ///
    /// # Inputs
    /// `prompt_len`: prompt length in tokens.
    ///
    /// # Outputs
    /// A request with that length, fixed seed `0xC0FFEE`, and default optional fields.
    fn req(prompt_len: u64) -> PlacementRequest {
        PlacementRequest {
            prompt_len,
            transfer_seed: 0xC0FFEE,
            ..Default::default()
        }
    }

    /// Build a test fleet whose workers can execute either leg.
    ///
    /// # Inputs
    /// `n`: number of workers.
    ///
    /// # Outputs
    /// Roles for `w0` through `w{n-1}`, all with both capabilities and full P/D weights.
    fn symmetric(n: usize) -> RoleWeights {
        let mut w = RoleWeights::new();
        for i in 0..n {
            w.insert(format!("w{i}"), RoleWeight::default());
        }
        w
    }

    /// 4 prefill-leaning + 4 decode-leaning, force_split.
    /// Build a fixed-role split-placement test fixture.
    ///
    /// # Inputs
    /// None.
    ///
    /// # Outputs
    /// Four P-only role weights, four D-only role weights, and a force-split config;
    /// capabilities remain symmetric.
    fn static_4_plus_4() -> (RoleWeights, PlacementConfig) {
        let mut w = RoleWeights::new();
        for p in P {
            w.insert(p, RoleWeight::PREFILL_LEANING);
        }
        for d in D {
            w.insert(d, RoleWeight::DECODE_LEANING);
        }
        let cfg = PlacementConfig {
            force_split: true,
            ..Default::default()
        };
        (w, cfg)
    }

    #[test]
    /// Check the default policy retains cache affinity and chooses Local execution.
    ///
    /// # Inputs
    /// No arguments; gives w2 a near-full cache hit plus backlog and decode load.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn cache_affinity_dominates_by_default() {
        let weights = symmetric(4);
        let cfg = PlacementConfig::default();
        let mut hits = IndexerHits::new();
        hits.insert("w2", HitTokens { gpu: 8000, host: 0 });
        // w2 also carries a full unit of backlog; the hit must still win.
        let mut loads = WorkerLoads::new();
        loads.insert(
            "w2",
            WorkerLoad {
                backlog_uncached_tokens: cfg.backlog_norm_tokens,
                running_decode: 10,
                waiting: 2,
                ..Default::default()
            },
        );
        let pl = place(&req(8192), &hits, &loads, &weights, &cfg).unwrap();
        assert_eq!(pl.p_leg, "w2");
        assert_eq!(pl.d_leg, "w2", "Local wins: no transfer cost, same worker");
        assert!(pl.is_local());
        assert_eq!(pl.transfer_id, None);
        assert!(pl.term(TERM_REUSE_GAIN).unwrap() > 0.97);
    }

    #[test]
    /// Check the least-backlogged worker receives both legs.
    ///
    /// # Inputs
    /// No arguments; uses three equal-active-count workers with different uncached backlogs and
    /// no hits.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn no_hits_prefers_least_backlog_local() {
        let weights = symmetric(3);
        let cfg = PlacementConfig::default();
        let mut loads = WorkerLoads::new();
        loads.insert(
            "w0",
            WorkerLoad {
                backlog_uncached_tokens: 5000,
                ..Default::default()
            },
        );
        loads.insert(
            "w1",
            WorkerLoad {
                backlog_uncached_tokens: 100,
                ..Default::default()
            },
        );
        loads.insert(
            "w2",
            WorkerLoad {
                backlog_uncached_tokens: 3000,
                ..Default::default()
            },
        );
        let pl = place(&req(1024), &IndexerHits::new(), &loads, &weights, &cfg).unwrap();
        assert_eq!((pl.p_leg.as_str(), pl.d_leg.as_str()), ("w1", "w1"));
    }

    #[test]
    /// Check fixed role exclusions constrain each leg and Split preserves the supplied transfer
    /// seed.
    ///
    /// # Inputs
    /// No arguments; uses four P/four D role groups, a large hit on an excluded D, a valid P
    /// hit, and a busy decoder.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn force_split_static_weights_reproduce_todays_pd() {
        let (weights, cfg) = static_4_plus_4();
        // Even a full hit on a decode-leaning worker must not pull the P-leg
        // there: weight 0 is a hard exclusion.
        let mut hits = IndexerHits::new();
        hits.insert(
            "d1",
            HitTokens {
                gpu: 4096,
                host: 4096,
            },
        );
        // 3072 / 4096 = 0.75 > cache_threshold: p3 wins on cache under
        // `cache_first`; the score mode below sees the same answer.
        hits.insert("p3", HitTokens { gpu: 3072, host: 0 });
        let mut loads = WorkerLoads::new();
        loads.insert(
            "d0",
            WorkerLoad {
                running_decode: 50,
                ..Default::default()
            },
        );
        for seed in 0..20u64 {
            let mut r = req(4096);
            r.transfer_seed = seed;
            let pl = place(&r, &hits, &loads, &weights, &cfg).unwrap();
            assert!(P.contains(&pl.p_leg.as_str()), "P from prefill set: {pl:?}");
            assert!(D.contains(&pl.d_leg.as_str()), "D from decode set: {pl:?}");
            assert!(!pl.is_local());
            assert_eq!(pl.transfer_id, Some(seed), "transfer_id minted for Split");
        }
        let pl = place(&req(4096), &hits, &loads, &weights, &cfg).unwrap();
        assert_eq!(pl.p_leg, "p3", "own-hit prefill worker wins the P-leg");
        assert_ne!(pl.d_leg, "d0", "loaded decode worker loses the D-leg");
    }

    #[test]
    /// Check force-split selects w0 for P and the other worker for D.
    ///
    /// # Inputs
    /// No arguments; uses two symmetric workers with all prompt KV on w0.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn force_split_forbids_local_even_on_symmetric_fleet() {
        let weights = symmetric(2);
        let cfg = PlacementConfig {
            force_split: true,
            ..Default::default()
        };
        let mut hits = IndexerHits::new();
        hits.insert("w0", HitTokens { gpu: 2048, host: 0 });
        let pl = place(&req(2048), &hits, &WorkerLoads::new(), &weights, &cfg).unwrap();
        assert_eq!(pl.p_leg, "w0");
        assert_eq!(pl.d_leg, "w1");
        assert_eq!(pl.transfer_id, Some(0xC0FFEE));
    }

    #[test]
    /// Check all ten default score terms are recorded and Local has zero transfer cost.
    ///
    /// # Inputs
    /// No arguments; places a request on an idle symmetric fleet.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn every_score_term_is_reported() {
        let weights = symmetric(2);
        let pl = place(
            &req(100),
            &IndexerHits::new(),
            &WorkerLoads::new(),
            &weights,
            &PlacementConfig::default(),
        )
        .unwrap();
        for name in [
            TERM_HIT_TOKENS,
            TERM_REUSE_GAIN,
            TERM_BACKLOG,
            TERM_ROLE_PEN_P,
            TERM_ROLE_PEN_D,
            TERM_XFER_COST,
            TERM_DECODE_LOAD,
            TERM_P_REASON,
            TERM_D_SELECT,
            TERM_SCORE,
        ] {
            assert!(pl.term(name).is_some(), "missing score term {name}");
        }
        assert_eq!(pl.score_terms.len(), 10);
        // Local => zero transfer cost.
        assert_eq!(pl.term(TERM_XFER_COST), Some(0.0));
    }

    #[test]
    /// Check only Split placements carry the supplied transfer seed.
    ///
    /// # Inputs
    /// No arguments; compares Local placement with a cached worker whose decode side is heavily
    /// loaded.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn transfer_id_minted_only_for_split() {
        let weights = symmetric(2);
        let cfg = PlacementConfig::default();
        let local = place(
            &req(512),
            &IndexerHits::new(),
            &WorkerLoads::new(),
            &weights,
            &cfg,
        )
        .unwrap();
        assert!(local.is_local());
        assert_eq!(local.transfer_id, None);

        // Make the transfer worth it: huge hit on w1, but w1's decode side is
        // saturated so the D-leg moves to w0.
        let mut hits = IndexerHits::new();
        hits.insert(
            "w1",
            HitTokens {
                gpu: 512,
                host: 512,
            },
        );
        let mut loads = WorkerLoads::new();
        loads.insert(
            "w1",
            WorkerLoad {
                running_decode: 100_000,
                ..Default::default()
            },
        );
        let split = place(&req(512), &hits, &loads, &weights, &cfg).unwrap();
        assert_eq!(split.p_leg, "w1");
        assert_eq!(split.d_leg, "w0");
        assert_eq!(split.transfer_id, Some(0xC0FFEE));
    }

    #[test]
    /// Check equal inputs produce equal results with lexical Local tie-breaking.
    ///
    /// # Inputs
    /// No arguments; repeats placement on three identical idle workers.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn deterministic_tie_break_is_lexical_and_prefers_local() {
        let weights = symmetric(3);
        let cfg = PlacementConfig::default();
        let a = place(
            &req(1),
            &IndexerHits::new(),
            &WorkerLoads::new(),
            &weights,
            &cfg,
        )
        .unwrap();
        let b = place(
            &req(1),
            &IndexerHits::new(),
            &WorkerLoads::new(),
            &weights,
            &cfg,
        )
        .unwrap();
        assert_eq!(a, b);
        assert_eq!((a.p_leg.as_str(), a.d_leg.as_str()), ("w0", "w0"));
    }

    #[test]
    /// Check capabilities restrict leg selection and a missing P produces the typed error.
    ///
    /// # Inputs
    /// No arguments; uses explicit P-only/D-only capabilities, then removes the only P-capable
    /// worker.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn capability_set_gates_legs() {
        let mut w = RoleWeights::new();
        w.insert_with_caps("only_p", RoleWeight::default(), Capabilities::PREFILL_ONLY);
        w.insert_with_caps("only_d", RoleWeight::default(), Capabilities::DECODE_ONLY);
        let pl = place(
            &req(64),
            &IndexerHits::new(),
            &WorkerLoads::new(),
            &w,
            &PlacementConfig::default(),
        )
        .unwrap();
        assert_eq!(pl.p_leg, "only_p");
        assert_eq!(pl.d_leg, "only_d");
        assert!(pl.transfer_id.is_some());

        let mut w = RoleWeights::new();
        w.insert_with_caps("only_d", RoleWeight::default(), Capabilities::DECODE_ONLY);
        assert_eq!(
            place(
                &req(64),
                &IndexerHits::new(),
                &WorkerLoads::new(),
                &w,
                &PlacementConfig::default()
            ),
            Err(PlacementError::NoPrefillCandidate)
        );
    }

    #[test]
    /// Check an impossible distinct-worker pair returns NoPairUnderForceSplit.
    ///
    /// # Inputs
    /// No arguments; forces Split on one symmetric worker.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn force_split_single_worker_is_an_error() {
        let weights = symmetric(1);
        let cfg = PlacementConfig {
            force_split: true,
            ..Default::default()
        };
        assert_eq!(
            place(
                &req(64),
                &IndexerHits::new(),
                &WorkerLoads::new(),
                &weights,
                &cfg
            ),
            Err(PlacementError::NoPairUnderForceSplit)
        );
    }

    #[test]
    /// Check positive role weights affect preference without excluding the worker.
    ///
    /// # Inputs
    /// No arguments; compares a partial decode-role weight before and after adding a full P
    /// cache hit.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn soft_role_weights_bias_but_do_not_lock() {
        // 0.2 decode weight is a penalty, not an exclusion: with nothing else
        // to go on the fully-weighted worker wins, but a big enough hit flips it.
        let mut w = RoleWeights::new();
        w.insert(
            "lean_p",
            RoleWeight {
                prefill: 1.0,
                decode: 0.2,
            },
        );
        w.insert("lean_d", RoleWeight::DECODE_LEANING);
        w.insert("both", RoleWeight::default());
        let cfg = PlacementConfig::default();
        let pl = place(
            &req(4096),
            &IndexerHits::new(),
            &WorkerLoads::new(),
            &w,
            &cfg,
        )
        .unwrap();
        assert_eq!((pl.p_leg.as_str(), pl.d_leg.as_str()), ("both", "both"));

        let mut hits = IndexerHits::new();
        hits.insert("lean_p", HitTokens { gpu: 4096, host: 0 });
        let pl = place(&req(4096), &hits, &WorkerLoads::new(), &w, &cfg).unwrap();
        assert_eq!(pl.p_leg, "lean_p");
    }

    /// Build a four-P load snapshot for selection tests.
    ///
    /// # Inputs
    /// `active`: active-request counts in P order; `backlog`: corresponding uncached-token
    /// counts.
    ///
    /// # Outputs
    /// A `WorkerLoads` for p0 through p3; unspecified load fields are zero.
    fn static_loads(active: [u64; 4], backlog: [u64; 4]) -> WorkerLoads {
        let mut loads = WorkerLoads::new();
        for (i, p) in P.iter().enumerate() {
            loads.insert(
                *p,
                WorkerLoad {
                    backlog_uncached_tokens: backlog[i],
                    active_prefill_reqs: active[i],
                    ..Default::default()
                },
            );
        }
        loads
    }

    /// The attempt-3 failure shape: the hit worker carries millions of
    /// pending tokens; `cache_first` must still follow the cache while the
    /// literal score follows the backlog.
    #[test]
    /// Check cacheFirst retains the high-hit P while Score follows the lower backlog.
    ///
    /// # Inputs
    /// No arguments; compares CacheFirst and Score on millions of pending tokens without count
    /// imbalance.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn cache_first_follows_hit_regardless_of_backlog() {
        let (weights, mut cfg) = static_4_plus_4();
        let mut hits = IndexerHits::new();
        hits.insert(
            "p2",
            HitTokens {
                gpu: 40_000,
                host: 0,
            },
        );
        let loads = static_loads(
            [110, 112, 124, 105],
            [5_000_000, 6_000_000, 9_000_000, 3_000_000],
        );
        let pl = place(&req(59_407), &hits, &loads, &weights, &cfg).unwrap();
        assert_eq!(pl.p_leg, "p2");
        assert_eq!(pl.term(TERM_P_REASON), Some(P_REASON_CACHE));
        assert!(D.contains(&pl.d_leg.as_str()));

        cfg.p_select = PSelect::Score;
        let pl = place(&req(59_407), &hits, &loads, &weights, &cfg).unwrap();
        assert_eq!(pl.p_leg, "p3", "the additive score follows the backlog");
        assert_eq!(pl.term(TERM_P_REASON), Some(P_REASON_SCORE));
    }

    #[test]
    /// Check cache-first load balancing overrides cache only when its imbalance conditions
    /// hold.
    ///
    /// # Inputs
    /// No arguments; varies active-request spreads around the cache-first thresholds.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn cache_first_imbalance_fast_path_overrides_cache() {
        let (weights, cfg) = static_4_plus_4();
        let mut hits = IndexerHits::new();
        hits.insert("p1", HitTokens { gpu: 4096, host: 0 });
        // p0 idle, the rest 40 in flight: 40 - 0 > 32 and 40 > 0 * 1.1.
        let loads = static_loads([0, 40, 40, 40], [0, 0, 0, 0]);
        let pl = place(&req(4096), &hits, &loads, &weights, &cfg).unwrap();
        assert_eq!(pl.p_leg, "p0");
        assert_eq!(pl.term(TERM_P_REASON), Some(P_REASON_IMBALANCE));
        // Spread below the absolute threshold: cache wins again.
        let loads = static_loads([0, 30, 30, 30], [0, 0, 0, 0]);
        let pl = place(&req(4096), &hits, &loads, &weights, &cfg).unwrap();
        assert_eq!(pl.p_leg, "p1");
        assert_eq!(pl.term(TERM_P_REASON), Some(P_REASON_CACHE));
        // Large absolute spread but under the relative threshold (100 vs 95).
        let loads = static_loads([95, 100, 100, 100], [0, 0, 0, 0]);
        let pl = place(&req(4096), &hits, &loads, &weights, &cfg).unwrap();
        assert_eq!(pl.p_leg, "p1");
    }

    #[test]
    /// Check an eligible replay pair wins and reports replay reasons; removing it restores
    /// cache selection.
    ///
    /// # Inputs
    /// No arguments; supplies an eligible P/D replay pin that conflicts with the cached P.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn replay_pin_overrides_cache_first_and_round_robin() {
        let (weights, cfg) = static_4_plus_4();
        let mut hits = IndexerHits::new();
        hits.insert("p1", HitTokens { gpu: 4096, host: 0 }); // cache-first would pick p1
        let loads = static_loads([0, 0, 0, 0], [0, 0, 0, 0]);
        let mut r = req(4096);
        r.pin = Some(("p3".into(), "d2".into()));
        let pl = place(&r, &hits, &loads, &weights, &cfg).unwrap();
        assert_eq!((pl.p_leg.as_str(), pl.d_leg.as_str()), ("p3", "d2"));
        assert_eq!(pl.term(TERM_P_REASON), Some(P_REASON_REPLAY));
        assert_eq!(pl.term(TERM_D_SELECT), Some(D_SELECT_REPLAY));
        assert!(!pl.is_local());
        // Without the pin the same inputs go to the cache owner.
        let pl = place(&req(4096), &hits, &loads, &weights, &cfg).unwrap();
        assert_eq!(pl.p_leg, "p1");
        assert_eq!(pl.term(TERM_P_REASON), Some(P_REASON_CACHE));
    }

    #[test]
    /// Check invalid replay pins fall back to configured placement rather than bypassing
    /// eligibility.
    ///
    /// # Inputs
    /// No arguments; tests unknown, wrong-role, and forbidden Local replay pairs.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn replay_pin_falls_back_when_a_leg_is_not_eligible() {
        let (weights, cfg) = static_4_plus_4();
        let mut hits = IndexerHits::new();
        hits.insert("p1", HitTokens { gpu: 4096, host: 0 });
        let loads = static_loads([0, 0, 0, 0], [0, 0, 0, 0]);
        // Unknown D url: configured selection runs as if no pin.
        let mut r = req(4096);
        r.pin = Some(("p3".into(), "nowhere".into()));
        let pl = place(&r, &hits, &loads, &weights, &cfg).unwrap();
        assert_eq!(pl.p_leg, "p1");
        assert_eq!(pl.term(TERM_P_REASON), Some(P_REASON_CACHE));
        // static_4_plus_4() keeps the default D selection (score), not replay.
        assert_eq!(pl.term(TERM_D_SELECT), Some(D_SELECT_SCORE));
        // A decode-only worker cannot take the P-leg under static 4+4.
        let mut r = req(4096);
        r.pin = Some(("d0".into(), "d1".into()));
        let pl = place(&r, &hits, &loads, &weights, &cfg).unwrap();
        assert_eq!(pl.p_leg, "p1");
        // `p == d` under force_split is not honoured either.
        let mut w = RoleWeights::new();
        for i in 0..2 {
            w.insert(format!("w{i}"), RoleWeight::default());
        }
        let cfg_split = PlacementConfig {
            force_split: true,
            ..Default::default()
        };
        let mut r = req(4096);
        r.pin = Some(("w0".into(), "w0".into()));
        let pl = place(&r, &IndexerHits::new(), &WorkerLoads::new(), &w, &cfg_split).unwrap();
        assert_ne!(pl.p_leg, pl.d_leg);
        assert_ne!(pl.term(TERM_P_REASON), Some(P_REASON_REPLAY));
    }

    #[test]
    /// Check below-threshold cache hits use active count, backlog, then lexical ID tie-breaks.
    ///
    /// # Inputs
    /// No arguments; varies hits around the strict threshold and ties the active
    /// counts/backlogs.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn cache_first_below_threshold_uses_min_load_then_backlog_then_id() {
        let (weights, cfg) = static_4_plus_4();
        let mut hits = IndexerHits::new();
        hits.insert("p1", HitTokens { gpu: 1024, host: 0 }); // 0.25 <= 0.5
        let loads = static_loads([5, 3, 3, 4], [700, 900, 800, 0]);
        let pl = place(&req(4096), &hits, &loads, &weights, &cfg).unwrap();
        assert_eq!(pl.p_leg, "p2", "min active, then min backlog");
        assert_eq!(pl.term(TERM_P_REASON), Some(P_REASON_MIN_LOAD));
        let loads = static_loads([3, 3, 3, 3], [0, 0, 0, 0]);
        let pl = place(&req(4096), &hits, &loads, &weights, &cfg).unwrap();
        assert_eq!(pl.p_leg, "p0", "full tie: lexical");
        // Exactly the threshold does not count (baseline: match_rate > thr).
        hits.insert("p1", HitTokens { gpu: 2048, host: 0 });
        let pl = place(&req(4096), &hits, &loads, &weights, &cfg).unwrap();
        assert_eq!(pl.term(TERM_P_REASON), Some(P_REASON_MIN_LOAD));
        hits.insert("p1", HitTokens { gpu: 2049, host: 0 });
        let pl = place(&req(4096), &hits, &loads, &weights, &cfg).unwrap();
        assert_eq!(pl.p_leg, "p1");
        assert_eq!(pl.term(TERM_P_REASON), Some(P_REASON_CACHE));
    }

    #[test]
    /// Check the lower-load cache owner wins an equal-hit tie.
    ///
    /// # Inputs
    /// No arguments; gives p0 and p3 identical full-prefix hits but different active counts.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn cache_first_ties_on_hit_break_by_load() {
        let (weights, cfg) = static_4_plus_4();
        let mut hits = IndexerHits::new();
        hits.insert("p0", HitTokens { gpu: 4096, host: 0 });
        hits.insert("p3", HitTokens { gpu: 4096, host: 0 });
        let loads = static_loads([9, 0, 0, 2], [0, 0, 0, 0]);
        let pl = place(&req(4096), &hits, &loads, &weights, &cfg).unwrap();
        assert_eq!(pl.p_leg, "p3");
        assert_eq!(pl.term(TERM_P_REASON), Some(P_REASON_CACHE));
    }

    fn decode_loads(running: [u64; 4], waiting: [u64; 4]) -> WorkerLoads {
        let mut loads = WorkerLoads::new();
        for (i, d) in D.iter().enumerate() {
            loads.insert(
                *d,
                WorkerLoad {
                    active_decode_reqs: running[i] + waiting[i],
                    ..Default::default()
                },
            );
        }
        loads
    }

    #[test]
    fn cache_first_d_follows_the_longest_prefix_then_decode_load() {
        let (weights, mut cfg) = static_4_plus_4();
        cfg.d_select = DSelect::CacheFirst;
        let mut hits = IndexerHits::new();
        hits.insert("d2", HitTokens { gpu: 3072, host: 0 });
        hits.insert("d1", HitTokens { gpu: 2560, host: 0 });
        // Spread 32 - 5 is not above 32: the longest prefix wins over load.
        let loads = decode_loads([20, 5, 30, 10], [0, 0, 2, 0]);
        let pl = place(&req(4096), &hits, &loads, &weights, &cfg).unwrap();
        assert_eq!(pl.d_leg, "d2");
        assert_eq!(pl.term(TERM_D_SELECT), Some(D_SELECT_CACHE));
        // 40 - 5 > 32 and 40 > 5 * 1.1: the least-loaded D.
        let loads = decode_loads([40, 5, 30, 10], [0, 0, 0, 0]);
        let pl = place(&req(4096), &hits, &loads, &weights, &cfg).unwrap();
        assert_eq!(pl.d_leg, "d1");
        assert_eq!(pl.term(TERM_D_SELECT), Some(D_SELECT_IMBALANCE));
        // No hit: fewest in-flight D-legs.
        let loads = decode_loads([9, 7, 3, 3], [0, 0, 1, 0]);
        let pl = place(&req(4096), &IndexerHits::new(), &loads, &weights, &cfg).unwrap();
        assert_eq!(pl.d_leg, "d3");
        assert_eq!(pl.term(TERM_D_SELECT), Some(D_SELECT_MIN_LOAD));
    }

    #[test]
    /// The cache-first D load is the router's in-flight D-leg count, not the
    /// engine-reported running/waiting (which misses requests waiting for KV):
    /// a D with few running requests but many dispatched D-legs is imbalanced.
    fn cache_first_d_load_counts_dispatched_d_legs_not_engine_running() {
        let (weights, mut cfg) = static_4_plus_4();
        cfg.d_select = DSelect::CacheFirst;
        let mut hits = IndexerHits::new();
        hits.insert("d2", HitTokens { gpu: 4096, host: 0 });
        let mut loads = WorkerLoads::new();
        for (i, d) in D.iter().enumerate() {
            loads.insert(
                *d,
                WorkerLoad {
                    // Engine view: d2 looks idlest (its D-legs wait for KV).
                    running_decode: if i == 2 { 20 } else { 30 },
                    waiting: 0,
                    active_decode_reqs: if i == 2 { 90 } else { 20 },
                    ..Default::default()
                },
            );
        }
        let pl = place(&req(4096), &hits, &loads, &weights, &cfg).unwrap();
        assert_ne!(pl.d_leg, "d2");
        assert_eq!(pl.term(TERM_D_SELECT), Some(D_SELECT_IMBALANCE));
    }

    #[test]
    fn cache_first_d_skips_p_under_force_split() {
        let cfg = PlacementConfig {
            force_split: true,
            d_select: DSelect::CacheFirst,
            ..Default::default()
        };
        let mut hits = IndexerHits::new();
        hits.insert("w1", HitTokens { gpu: 4096, host: 0 });
        hits.insert("w2", HitTokens { gpu: 3072, host: 0 });
        let pl = place(&req(4096), &hits, &WorkerLoads::new(), &symmetric(3), &cfg).unwrap();
        assert_eq!((pl.p_leg.as_str(), pl.d_leg.as_str()), ("w1", "w2"));
        assert_eq!(pl.term(TERM_D_SELECT), Some(D_SELECT_CACHE));
    }

    #[test]
    /// Check d round robin ignores decode load and excludes the chosen P when Split is
    /// required.
    ///
    /// # Inputs
    /// No arguments; varies seeds on fixed roles and symmetric fleets, including an impossible
    /// one-worker Split.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn round_robin_d_cycles_by_seed_and_skips_p_under_force_split() {
        let (weights, mut cfg) = static_4_plus_4();
        cfg.d_select = DSelect::RoundRobin;
        let mut loads = WorkerLoads::new();
        loads.insert(
            "d0",
            WorkerLoad {
                running_decode: 10_000,
                ..Default::default()
            },
        );
        let mut seen = Vec::new();
        for seed in 0..8u64 {
            let mut r = req(4096);
            r.transfer_seed = seed;
            let pl = place(&r, &IndexerHits::new(), &loads, &weights, &cfg).unwrap();
            assert_eq!(pl.term(TERM_D_SELECT), Some(D_SELECT_ROUND_ROBIN));
            assert_eq!(pl.transfer_id, Some(seed));
            seen.push(pl.d_leg);
        }
        assert_eq!(
            seen,
            ["d0", "d1", "d2", "d3", "d0", "d1", "d2", "d3"],
            "round robin ignores decode load; d0 is not skipped"
        );

        // Symmetric pair under force_split: the pinned P is excluded from
        // the D pool, so the other worker is always the D-leg.
        let weights = symmetric(2);
        let cfg = PlacementConfig {
            force_split: true,
            d_select: DSelect::RoundRobin,
            ..Default::default()
        };
        for seed in 0..4u64 {
            let mut r = req(64);
            r.transfer_seed = seed;
            let pl = place(&r, &IndexerHits::new(), &WorkerLoads::new(), &weights, &cfg).unwrap();
            assert_eq!(pl.p_leg, "w0");
            assert_eq!(pl.d_leg, "w1");
        }
        let cfg = PlacementConfig {
            force_split: true,
            d_select: DSelect::RoundRobin,
            ..Default::default()
        };
        assert_eq!(
            place(
                &req(64),
                &IndexerHits::new(),
                &WorkerLoads::new(),
                &symmetric(1),
                &cfg
            ),
            Err(PlacementError::NoPairUnderForceSplit)
        );
    }

    #[test]
    /// Check cache-first P selection still allows score-based D load selection.
    ///
    /// # Inputs
    /// No arguments; pins P by cache and gives each D a different running count.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn score_mode_d_leg_still_ranked_under_cache_first() {
        // P pinned by cache; D by score => least decode load (not lexical).
        let (weights, cfg) = static_4_plus_4();
        let mut hits = IndexerHits::new();
        hits.insert("p1", HitTokens { gpu: 4096, host: 0 });
        let mut loads = WorkerLoads::new();
        for (d, n) in D.iter().zip([30u64, 20, 5, 40]) {
            loads.insert(
                *d,
                WorkerLoad {
                    running_decode: n,
                    ..Default::default()
                },
            );
        }
        let pl = place(&req(4096), &hits, &loads, &weights, &cfg).unwrap();
        assert_eq!((pl.p_leg.as_str(), pl.d_leg.as_str()), ("p1", "d2"));
        assert_eq!(pl.term(TERM_D_SELECT), Some(D_SELECT_SCORE));
    }

    #[test]
    /// Check mode names use snake_case and omitted fields restore placement defaults.
    ///
    /// # Inputs
    /// No arguments; serializes explicit selection modes and deserializes an empty config.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn selection_modes_serialize_snake_case() {
        let cfg = PlacementConfig {
            p_select: PSelect::CacheFirst,
            d_select: DSelect::RoundRobin,
            ..Default::default()
        };
        let v = serde_json::to_value(&cfg).unwrap();
        assert_eq!(v["p_select"], "cache_first");
        assert_eq!(v["d_select"], "round_robin");
        let back: PlacementConfig = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(back, PlacementConfig::default());
        assert_eq!(back.p_select, PSelect::CacheFirst);
        assert_eq!(back.cache_threshold, 0.5);
        assert_eq!(back.balance_abs_reqs, 32);
        let cost = PlacementConfig {
            p_select: PSelect::CostFunction,
            ..Default::default()
        };
        let v = serde_json::to_value(&cost).unwrap();
        assert_eq!(v["p_select"], "cost_function");
        let back: PlacementConfig = serde_json::from_value(v).unwrap();
        assert_eq!(back.p_select, PSelect::CostFunction);
        assert_eq!(p_reason_name(P_REASON_COST_FUNCTION), "cost_function");
    }

    #[test]
    /// Check the cost-function P selection takes the one prefill candidate that the runtime
    /// left and reports the cost-function expected hit.
    ///
    /// # Inputs
    /// No arguments; three P-role workers of which the runtime left only p1 prefill-capable (p1
    /// has an expected pull hit), and two D-role workers.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn cost_function_takes_the_single_prefill_candidate() {
        let narrowed = Capabilities {
            prefill: false,
            decode: true,
        };
        let mut weights = RoleWeights::new();
        weights
            .insert_with_caps("p0", RoleWeight::PREFILL_LEANING, narrowed)
            .insert("p1", RoleWeight::PREFILL_LEANING)
            .insert_with_caps("p2", RoleWeight::PREFILL_LEANING, narrowed)
            .insert("d0", RoleWeight::DECODE_LEANING)
            .insert("d1", RoleWeight::DECODE_LEANING);
        // p0 has the longest local prefix; the cost function chose p1 anyway.
        let mut hits = IndexerHits::new();
        hits.insert(
            "p0",
            HitTokens {
                gpu: 12000,
                host: 12000,
            },
        );
        hits.expected_after_pull.insert("p1".into(), 6000);
        let cfg = PlacementConfig {
            p_select: PSelect::CostFunction,
            d_select: DSelect::RoundRobin,
            force_split: true,
            ..Default::default()
        };
        for seed in 0..4u64 {
            let mut r = req(16384);
            r.transfer_seed = seed;
            let pl = place(&r, &hits, &WorkerLoads::new(), &weights, &cfg).unwrap();
            assert_eq!(pl.p_leg, "p1");
            assert!(["d0", "d1"].contains(&pl.d_leg.as_str()), "{pl:?}");
            assert_eq!(pl.term(TERM_P_REASON), Some(P_REASON_COST_FUNCTION));
            assert_eq!(p_reason_name(P_REASON_COST_FUNCTION), "cost_function");
            assert_eq!(pl.term(TERM_EXPECTED_HIT_TOKENS), Some(6000.0));
        }
    }

    #[test]
    /// Check the cost-function P selection refuses a pool that the runtime did not narrow.
    ///
    /// # Inputs
    /// No arguments; two prefill-capable P-role workers and one D-role worker.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn cost_function_rejects_two_prefill_candidates() {
        let mut weights = RoleWeights::new();
        weights
            .insert("p0", RoleWeight::PREFILL_LEANING)
            .insert("p1", RoleWeight::PREFILL_LEANING)
            .insert("d0", RoleWeight::DECODE_LEANING);
        let cfg = PlacementConfig {
            p_select: PSelect::CostFunction,
            force_split: true,
            ..Default::default()
        };
        let err = place(
            &req(4096),
            &IndexerHits::new(),
            &WorkerLoads::new(),
            &weights,
            &cfg,
        )
        .unwrap_err();
        let want = "cost_function expects exactly one P candidate, got 2";
        assert_eq!(err, PlacementError::InvalidPolicyOutput(want.into()));
    }
}
