// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! External scheduler orchestration for symmetric P/D engines.
//!
//! Placement belongs to `scheduler::placement`; backend state and SGLang wire
//! operations live in `backend`. Indexer queries run before the short atomic
//! snapshot/select/reserve section. CoSchedule uses cached-owner protection
//! and post-dispatch uncached-work spread, with no router HOLD queue. Existing
//! CacheFirst/Score modes retain their configured HOLD behavior.
//!
//! CoSchedule merges per-RID engine progress into the initial request estimates.
//! Missing RIDs retain their latest estimates; aggregate waiting tokens never
//! replace running work. Legacy modes retain a labelled aggregate estimate.
//! Managed reservations live until explicit completion. The legacy `/symmetric_pd/place` API alone uses an expiring
//! lease because it has no completion callback. New prepared execution owns
//! its reservation until the execution manager receives a terminal callback.

use super::{
    placement,
    policy::{validate_placement, DefaultCostModel, DefaultPlacementPolicy, PlacementPolicy},
};

use std::{
    collections::HashMap,
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

#[cfg(test)]
use bytes::Bytes;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};

use crate::{
    config::SymmetricPdConfig,
    discovery::ModelId,
    kv_hint::{KvHintConfig, KvHintInjector},
    server::error::ApiError,
    workers::{Worker, WorkerRegistry},
};
#[cfg(test)]
use placement::Capabilities;
#[cfg(test)]
use placement::DSelect;
use placement::PSelect;
use placement::{
    d_select_name, p_reason_name, TERM_BACKLOG, TERM_DECODE_LOAD, TERM_D_SELECT,
    TERM_EXPECTED_HIT_TOKENS, TERM_HIT_TOKENS, TERM_P_REASON, TERM_REUSE_GAIN, TERM_ROLE_PEN_D,
    TERM_ROLE_PEN_P, TERM_SCORE, TERM_XFER_COST,
};
use placement::{
    HitTokens, IndexerHits, Placement, PlacementError, PlacementRequest, RoleWeights, WorkerLoads,
};

const HOLD_TICK: Duration = Duration::from_millis(100);
/// `--symmetric-pd-gwq-auto`: how often [`SymmetricPdRuntime::admit`] walks a request
/// again whose released P left the P-role pool before the placement.
const GWQ_REWALKS: u32 = 3;
#[cfg(test)]
use crate::backend::sglang::progress::parse_loads;
pub use crate::backend::sglang::{
    kv::{Marks, KV_EXPORT_KEY, KV_IMPORT_KEY},
    request::{build_leg_bodies, estimate_prompt_len, inject_marks, KV_PULL_HINT_KEY},
};
pub use crate::backend::state::{DecodeReservation, LedgerHandle};
use crate::backend::{Backend, BackendError};

/// What the router hands `place()` for one request, plus what the marks
/// builder needs.
#[derive(Debug, Clone, Default)]
pub struct PlaceInput {
    pub tokens: Option<Vec<u32>>,
    pub hashes: Option<Vec<u64>>,
    /// Prompt length in tokens; estimated by the caller when unknown.
    pub prompt_len: u64,
    pub cache_salt: Option<String>,
    /// Model name forwarded to the indexer (`model_name`); candidates are
    /// always the served model's workers.
    pub model_id: Option<String>,
    pub session_id: Option<String>,
    pub turn: Option<u32>,
    pub rid: Option<String>,
    /// Session routing key from the request header named by
    /// `SymmetricPdConfig::decision_replay_header`; indexes the decision-replay
    /// table. `None` when the header is absent or replay is off.
    pub routing_key: Option<String>,
    /// Milliseconds from the HTTP handler's entry (body received) to
    /// `admit()`: body parse + tokenization + session/rid extraction. Logged
    /// with the placement so the ingress cost is measurable per request.
    pub ingress_ms: f64,
    /// `--symmetric-pd-p-select cost-function` with the copy price or R `auto`:
    /// the request's session identity
    /// ([`crate::agent_identity::trade_session_key`]), counted in N_active.
    /// `None` otherwise.
    pub trade_session: Option<String>,
    /// `--trade-wait e3` on the cost function: the session key
    /// of the E3 host rule ([`crate::trade_e3::session_key`]: the
    /// `x-kv-session-context` agent, else the `x-smg-routing-key` header).
    /// `None` otherwise.
    pub e3_session: Option<String>,
}

/// Result of [`SymmetricPdRuntime::admit`].
#[derive(Debug)]
pub struct Decision {
    pub placement: Placement,
    pub p_worker: Arc<Worker>,
    pub d_worker: Arc<Worker>,
    pub marks: Marks,
    /// Backlog accounting for the P-leg; retire when its response resolves.
    pub ledger: LedgerHandle,
    /// The D-leg's place in its worker's in-flight count (the cache-first D load);
    /// drop it when the D-leg completes, fails or is cancelled.
    pub d_reservation: Option<DecodeReservation>,
    pub queue_wait_ms: u64,
    pub forced: bool,
    pub indexer_ms: f64,
    pub prompt_len: u64,
    /// Peer-kv pull hint for the P-leg (`SymmetricPdConfig::pull_hint`).
    pub pull_hint: Option<Value>,
    /// `--symmetric-pd-p-select cost-function`: the request's unique cost-book
    /// reservation. The route moves it into the detached P-leg task, which
    /// marks it dispatched before the send, finished on the P answer and
    /// unknown on a transport error; a drop before dispatch rolls it back.
    /// `None` for every other P selection.
    pub cost_reservation: Option<crate::cost_balance::Reservation>,
}

impl Decision {
    /// Determine whether this admitted request executes both legs on one worker.
    ///
    /// # Inputs
    /// `self`: admitted decision and its placement.
    ///
    /// # Outputs
    /// `true` for Local placement and `false` for Split; the decision is unchanged.
    pub fn is_local(&self) -> bool {
        self.placement.is_local()
    }

    /// `"local"` / `"split"` as logged on the routing-decision line.
    /// Get the routing branch label used in decision logs.
    ///
    /// # Inputs
    /// `self`: admitted decision.
    ///
    /// # Outputs
    /// The static string `local` or `split`, derived from the selected worker IDs.
    pub fn branch(&self) -> &'static str {
        if self.is_local() {
            "local"
        } else {
            "split"
        }
    }

    /// Score term by name (see `placement::TERM_*`); 0 when absent.
    /// Read a placement diagnostic with a zero fallback.
    ///
    /// # Inputs
    /// `self`: admitted decision; `name`: exact score-term key.
    ///
    /// # Outputs
    /// The recorded numeric value, or `0.0` if absent.
    pub fn term(&self, name: &str) -> f64 {
        self.placement.term(name).unwrap_or(0.0)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SymmetricPdError {
    #[error("symmetric_pd: no available PD worker")]
    NoWorkers,
    #[error("{0}")]
    Placement(#[from] PlacementError),
    #[error("symmetric_pd: cannot resolve KV side-channel endpoints for {worker}: {reason}")]
    Endpoints { worker: String, reason: String },
    /// `--symmetric-pd-p-select cost-function`: the cost book refused the request
    /// (no unique rid, a failed trace) or the request cannot use it.
    #[error("symmetric_pd cost_function: {0}")]
    CostFunction(String),
}

impl From<BackendError> for SymmetricPdError {
    /// Preserve backend endpoint failures at the scheduler boundary.
    ///
    /// # Inputs
    /// `value`: owned `BackendError`.
    ///
    /// # Outputs
    /// A `SymmetricPdError::Endpoints` with the same worker and reason.
    fn from(value: BackendError) -> Self {
        match value {
            BackendError::Endpoints { worker, reason } => Self::Endpoints { worker, reason },
        }
    }
}

impl From<SymmetricPdError> for ApiError {
    /// Translate scheduler failures into the HTTP API error categories.
    ///
    /// # Inputs
    /// `e`: owned no-worker, placement, or endpoint failure.
    ///
    /// # Outputs
    /// A no-placement or endpoint `ApiError` retaining the formatted scheduler error.
    fn from(e: SymmetricPdError) -> Self {
        match e {
            SymmetricPdError::NoWorkers
            | SymmetricPdError::Placement(_)
            | SymmetricPdError::CostFunction(_) => ApiError::SymmetricPdNoPlacement(e.to_string()),
            SymmetricPdError::Endpoints { .. } => ApiError::SymmetricPdEndpoints(e.to_string()),
        }
    }
}

/// Decision replay (routing control). A reference run's
/// per-request (P, D) choices keyed by the sha256 of the session routing key.
/// Two table shapes:
/// - **order-keyed** (v1): `"sessions":{"<hex>":[["<p url>","<d url>"],...]}` —
///   the k-th request the router sees for a key takes the k-th pair (the
///   closed-loop strict replay keeps one request in flight per session).
///   Weakness: if a session's replay cycles end at different event counts in
///   the two runs, later positions are pinned to decisions made for a
///   different prompt (24 % of positions in one measured replay).
/// - **content-keyed** (v2): `"sessions_by_len":{"<hex>":{"<prompt_len>":
///   [["<p url>","<d url>"],...]}}` — the request is matched by its prompt
///   length in tokens (identical across runs and across replay cycles for
///   the same event); repeated occurrences of the same length take
///   successive pairs. A miss on the exact length falls back to the nearest
///   length within `len_tolerance` tokens, then to the configured selection.
///
/// Both shapes may be present; content-keyed wins when it has an entry.
#[derive(Debug)]
pub struct DecisionReplay {
    sessions: HashMap<String, Vec<(String, String)>>,
    by_len: HashMap<String, std::collections::BTreeMap<u64, Vec<(String, String)>>>,
    len_tolerance: u64,
    cursors: Mutex<HashMap<String, usize>>,
    len_cursors: Mutex<HashMap<(String, u64), usize>>,
    hits: AtomicU64,
    unknown: AtomicU64,
    exhausted: AtomicU64,
    no_key: AtomicU64,
}

#[derive(Deserialize)]
struct DecisionReplayFile {
    #[serde(default)]
    key_hash: Option<String>,
    #[serde(default)]
    sessions: HashMap<String, Vec<(String, String)>>,
    #[serde(default)]
    sessions_by_len: HashMap<String, HashMap<String, Vec<(String, String)>>>,
    /// Nearest-length fallback window in tokens (default 0 = exact only).
    #[serde(default)]
    len_tolerance: u64,
}

impl DecisionReplay {
    /// Read and validate a decision-replay table from disk.
    ///
    /// # Inputs
    /// `path`: JSON file containing order-keyed and/or content-length-keyed reference
    /// placements.
    ///
    /// # Outputs
    /// `Ok(DecisionReplay)` with initialized cursors/counters; a descriptive `Err(String)` on
    /// read/parse failure, unsupported key hash, invalid length key, or no recorded decisions.
    /// Successful loading emits a summary log.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let file: DecisionReplayFile =
            serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
        if let Some(h) = &file.key_hash {
            if h != "sha256" {
                return Err(format!(
                    "{}: key_hash must be \"sha256\", got {h:?}",
                    path.display()
                ));
            }
        }
        let total: usize = file.sessions.values().map(Vec::len).sum();
        let mut by_len: HashMap<String, std::collections::BTreeMap<u64, Vec<(String, String)>>> =
            HashMap::new();
        let mut total_len = 0usize;
        for (k, m) in file.sessions_by_len {
            let mut tree = std::collections::BTreeMap::new();
            for (len, v) in m {
                let l: u64 = len.parse().map_err(|_| {
                    format!(
                        "{}: prompt_len key {len:?} is not an integer",
                        path.display()
                    )
                })?;
                total_len += v.len();
                tree.insert(l, v);
            }
            by_len.insert(k, tree);
        }
        if total == 0 && total_len == 0 {
            return Err(format!(
                "{}: table has neither sessions nor sessions_by_len",
                path.display()
            ));
        }
        info!(
            target: "symmetric_pd",
            sessions = file.sessions.len(),
            decisions = total,
            sessions_by_len = by_len.len(),
            decisions_by_len = total_len,
            len_tolerance = file.len_tolerance,
            path = %path.display(),
            "symmetric_pd decision replay table loaded"
        );
        Ok(Self {
            sessions: file.sessions,
            by_len,
            len_tolerance: file.len_tolerance,
            cursors: Mutex::new(HashMap::new()),
            len_cursors: Mutex::new(HashMap::new()),
            hits: AtomicU64::new(0),
            unknown: AtomicU64::new(0),
            exhausted: AtomicU64::new(0),
            no_key: AtomicU64::new(0),
        })
    }

    /// Hash a routing key into the replay table's lookup format.
    ///
    /// # Inputs
    /// `routing_key`: original session or agent routing key.
    ///
    /// # Outputs
    /// Its SHA-256 digest as a lowercase hexadecimal string.
    fn key(routing_key: &str) -> String {
        let mut h = Sha256::new();
        h.update(routing_key.as_bytes());
        format!("{:x}", h.finalize())
    }

    /// Next pinned pair for `routing_key` (and `prompt_len` tokens when the
    /// table is content-keyed), advancing the matching cursor. The
    /// `&'static str` names the outcome for the decision log:
    /// `hit` (order-keyed), `hit_len` (exact length), `hit_len_near` (within
    /// tolerance), `unknown_key`, `unknown_len`, `exhausted`, `no_key`.
    /// Consume a reference placement for the given routing key and prompt length.
    ///
    /// # Inputs
    /// `self`: replay tables and mutable cursor/counter state; `routing_key`: optional nonempty
    /// original key; `prompt_len`: current prompt token count.
    ///
    /// # Outputs
    /// An optional P/D pair and outcome label. Advances matching cursors and counters;
    /// length-keyed lists reuse their last pair after exhaustion, while order-keyed lists
    /// return `None`.
    pub fn next(
        &self,
        routing_key: Option<&str>,
        prompt_len: u64,
    ) -> (Option<(String, String)>, &'static str) {
        let Some(rk) = routing_key.filter(|k| !k.is_empty()) else {
            self.no_key.fetch_add(1, Ordering::Relaxed);
            return (None, "no_key");
        };
        let key = Self::key(rk);
        if let Some(tree) = self.by_len.get(&key) {
            // Exact length, else nearest within tolerance (ties: shorter).
            let pick: Option<(u64, &'static str)> = if tree.contains_key(&prompt_len) {
                Some((prompt_len, "hit_len"))
            } else if self.len_tolerance > 0 {
                let lo = prompt_len.saturating_sub(self.len_tolerance);
                let hi = prompt_len.saturating_add(self.len_tolerance);
                tree.range(lo..=hi)
                    .map(|(l, _)| (*l, l.abs_diff(prompt_len)))
                    .min_by_key(|(l, d)| (*d, *l))
                    .map(|(l, _)| (l, "hit_len_near"))
            } else {
                None
            };
            let Some((len, outcome)) = pick else {
                if !self.sessions.contains_key(&key) {
                    self.unknown.fetch_add(1, Ordering::Relaxed);
                    return (None, "unknown_len");
                }
                // fall through to the order-keyed list below
                return self.next_ordered(&key);
            };
            let list = &tree[&len];
            let idx = {
                let mut cur = self.len_cursors.lock();
                let slot = cur.entry((key.clone(), len)).or_insert(0);
                let i = *slot;
                *slot += 1;
                i
            };
            // Repeated occurrences (replay cycles) take successive pairs; once
            // exhausted, keep the last recorded pair (same content, same
            // placement is the point of the control).
            let pd = list.get(idx).or_else(|| list.last()).cloned();
            self.hits.fetch_add(1, Ordering::Relaxed);
            return (pd, outcome);
        }
        self.next_ordered(&key)
    }

    /// Consume the next order-keyed reference placement.
    ///
    /// # Inputs
    /// `self`: replay tables/counters; `key`: already SHA-256-hashed routing key.
    ///
    /// # Outputs
    /// An optional pair and `hit`, `unknown_key`, or `exhausted`; known keys advance their
    /// cursor and the corresponding outcome counter.
    fn next_ordered(&self, key: &str) -> (Option<(String, String)>, &'static str) {
        let Some(list) = self.sessions.get(key) else {
            self.unknown.fetch_add(1, Ordering::Relaxed);
            return (None, "unknown_key");
        };
        let key = key.to_string();
        let idx = {
            let mut cur = self.cursors.lock();
            let slot = cur.entry(key).or_insert(0);
            let i = *slot;
            *slot += 1;
            i
        };
        match list.get(idx) {
            Some(pd) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                (Some(pd.clone()), "hit")
            }
            None => {
                self.exhausted.fetch_add(1, Ordering::Relaxed);
                (None, "exhausted")
            }
        }
    }

    /// `(hit, unknown_key, exhausted, no_key)` counters.
    /// Read replay outcome counters without advancing any cursor.
    ///
    /// # Inputs
    /// `self`: replay statistics.
    ///
    /// # Outputs
    /// Relaxed atomic snapshots of `(hit, unknown, exhausted, no_key)` counts; `unknown` also
    /// includes unmatched lengths.
    #[cfg(test)]
    pub fn counters(&self) -> (u64, u64, u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.unknown.load(Ordering::Relaxed),
            self.exhausted.load(Ordering::Relaxed),
            self.no_key.load(Ordering::Relaxed),
        )
    }
}

pub struct SymmetricPdRuntime {
    cfg: SymmetricPdConfig,
    /// `--symmetric-pd-decision-replay` table, when configured.
    replay: Option<DecisionReplay>,
    backend: Arc<Backend>,
    policy: Arc<dyn PlacementPolicy>,
    model: ModelId,
    indexer: Option<KvHintInjector>,
    /// Random-seeded so restarts do not reuse ids the engines may still hold.
    transfer_counter: AtomicU64,
    /// `--symmetric-pd-p-select gateway-cache-aware`: the gateway cache-aware
    /// rule and its tree, which pick the P-leg before placement.
    gateway_ca: Option<crate::policies::gateway_cache_aware::GatewayCacheAwarePolicy>,
    /// `--symmetric-pd-p-select cost-function`: the TRADE cost function, its cost
    /// ledger and its P progress poll, which pick and reserve the P-leg
    /// before placement.
    cost_function: Option<Arc<super::cost_function::CostFunction>>,
}

impl std::fmt::Debug for SymmetricPdRuntime {
    /// Format a concise scheduler-runtime debug view.
    ///
    /// # Inputs
    /// `self`: runtime configuration/model/policy; `f`: destination formatter.
    ///
    /// # Outputs
    /// `fmt::Result` from writing selected configuration and optional-component presence; no
    /// scheduling state is changed.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SymmetricPdRuntime")
            .field("model", &self.model)
            .field("force_split", &self.cfg.placement.force_split)
            .field("p_select", &self.cfg.placement.p_select)
            .field("d_select", &self.cfg.placement.d_select)
            .field("policy", &self.policy)
            .field("indexer", &self.indexer.is_some())
            .field("decision_replay", &self.replay.is_some())
            .field("cost_function", &self.cost_function.is_some())
            .finish()
    }
}

impl SymmetricPdRuntime {
    /// Build the runtime and, when inside a Tokio runtime, start the
    /// `/v1/loads` poller. `kv` (indexer URL + worker map) is optional:
    /// without it placement runs on loads and role weights only.
    /// Construct a runtime with the default placement policy and backend.
    ///
    /// # Inputs
    /// `cfg`: scheduler settings; `kv`: optional indexer settings; `registry`: shared worker
    /// registry; `model`: served model identifier.
    ///
    /// # Outputs
    /// A shared `Arc<SymmetricPdRuntime>`. Delegates backend/poller setup to `spawn_with_policy`;
    /// invalid replay or sticky-experiment configuration can panic during construction.
    pub fn spawn(
        cfg: SymmetricPdConfig,
        kv: Option<&KvHintConfig>,
        registry: Arc<WorkerRegistry>,
        model: ModelId,
    ) -> Arc<Self> {
        Self::spawn_with_policy(
            cfg,
            kv,
            registry,
            model,
            Arc::new(DefaultPlacementPolicy::<DefaultCostModel>::default()),
        )
    }

    /// Install a compiled Rust policy. It sees the same authoritative snapshot
    /// and runs inside the existing atomic select/reserve section; it must not
    /// perform I/O, parameter fitting or maintain a second request ledger.
    /// Construct a runtime around a compiled placement-policy extension.
    ///
    /// # Inputs
    /// `cfg`, `kv`, `registry`, `model`: runtime/backend/indexer settings; `policy`:
    /// thread-safe implementation called inside the select-and-reserve lock.
    ///
    /// # Outputs
    /// A shared runtime with Backend, optional indexer/replay and a randomly seeded transfer
    /// counter. Backend starts polling when a Tokio runtime is available; invalid replay or
    /// incompatible sticky settings panic.
    pub fn spawn_with_policy(
        cfg: SymmetricPdConfig,
        kv: Option<&KvHintConfig>,
        registry: Arc<WorkerRegistry>,
        model: ModelId,
        policy: Arc<dyn PlacementPolicy>,
    ) -> Arc<Self> {
        let indexer = kv.map(KvHintInjector::new);
        if indexer.is_none() {
            warn!(
                target: "symmetric_pd",
                "no --kv-indexer-url: symmetric_pd placement runs without cache hits (load/role only)"
            );
        }
        // The CLI checked the path is a file; a malformed table is an
        // operator error worth failing the start for.
        let replay = cfg.decision_replay.as_ref().map(|p| {
            DecisionReplay::load(p)
                .unwrap_or_else(|e| panic!("--symmetric-pd-decision-replay: {e}"))
        });
        assert!(
            !cfg.external_role_control || replay.is_none(),
            "external role control conflicts with decision replay"
        );
        let gateway_ca = cfg
            .gateway_cache_aware
            .clone()
            .map(crate::policies::gateway_cache_aware::GatewayCacheAwarePolicy::new);
        // The CLI requires the indexer with the cost function.
        let cost_function = cfg.cost_function.clone().map(|cost| {
            let kv = kv.expect("--symmetric-pd-p-select cost-function requires --kv-indexer-url");
            super::cost_function::CostFunction::new(cost, kv)
        });
        let this = Arc::new(Self {
            gateway_ca,
            cost_function,
            replay,
            backend: Backend::new(
                registry,
                model.clone(),
                cfg.import_cache_policy,
                cfg.load_poll_ms,
                cfg.placement.p_select == PSelect::CoSchedule,
                cfg.kv_session_retention,
                cfg.kv_session_key_file.as_deref(),
                // Unknown P reservations (a P-leg that ended in a transport
                // error, a cancel without terminal evidence, an aborted P-leg
                // after a drain migrated its D-leg) leave the ledger on engine
                // evidence, so the loads, the hold queue and the controller's
                // transition guard do not count them for ever.
                Some(crate::backend::state::UnknownReconcile {
                    empty_queue: crate::backend::state::UNKNOWN_RECONCILE_EMPTY_QUEUE,
                    quiet: Some(Duration::from_secs(cfg.cost_function.as_ref().map_or(
                        crate::config::types::default_symmetric_pd_unknown_expire_secs(),
                        |c| c.unknown_expire_secs,
                    )))
                    .filter(|d| !d.is_zero()),
                }),
            ),
            cfg,
            model,
            indexer,
            policy,
            transfer_counter: AtomicU64::new(rand::random::<u64>() & (i64::MAX as u64)),
        });
        if let Some(cost_function) = &this.cost_function {
            // The poll: the P-role workers whatever their breaker state (the
            // ledger of a tripped P stays current). The queue steps: the
            // healthy P-role pool of the admissions.
            let weak = Arc::downgrade(&this);
            let pool_weak = Arc::downgrade(&this);
            cost_function.start(
                move || {
                    weak.upgrade()
                        .map(|rt| rt.prefill_members())
                        .unwrap_or_default()
                },
                move || {
                    pool_weak
                        .upgrade()
                        .map(|rt| rt.cost_function_pool())
                        .unwrap_or_default()
                },
            );
            // `--symmetric-pd-pull-from-any-holder`: the queue steps keep a held
            // request's pull hint while its source is still a registered
            // worker of any role among the candidates (breaker allowing when
            // the breaker is on), the same set as the donors of an arrival.
            if cost_function.pull_from_any_holder() {
                let donors_weak = Arc::downgrade(&this);
                cost_function.set_gwq_donors(move || {
                    donors_weak
                        .upgrade()
                        .map(|rt| rt.candidates().1.into_values().collect())
                        .unwrap_or_default()
                });
            }
        }
        this
    }

    /// Expose the immutable runtime configuration.
    ///
    /// # Inputs
    /// `self`: initialized scheduler runtime.
    ///
    /// # Outputs
    /// A borrowed `SymmetricPdConfig`; no configuration is changed.
    /// Input: none. Output: optional backend-owned trajectory registry; no placement side effects.
    pub fn kv_sessions(&self) -> Option<&Arc<crate::backend::kv_sessions::KvSessions>> {
        self.backend.kv_sessions.as_ref()
    }

    pub fn config(&self) -> &SymmetricPdConfig {
        &self.cfg
    }

    /// `--symmetric-pd-p-select cost-function` is on.
    pub fn cost_function_enabled(&self) -> bool {
        self.cost_function.is_some()
    }

    /// The routes pass the session identity ([`PlaceInput::trade_session`]):
    /// the cost function's copy price or R `auto`, or
    /// `--symmetric-pd-trace-session-moves`.
    pub fn cost_function_wants_session(&self) -> bool {
        self.cost_function
            .as_ref()
            .is_some_and(|cf| cf.wants_session())
    }

    /// The cost function (tests).
    #[cfg(test)]
    pub(crate) fn cost_function_for_test(&self) -> &Arc<super::cost_function::CostFunction> {
        self.cost_function
            .as_ref()
            .expect("--symmetric-pd-p-select cost-function")
    }

    /// `--symmetric-pd-trace-session-moves`: the prepared route passes the
    /// session identity for the session-move trace (with the copy price or
    /// R `auto` alone it passes none, as before).
    pub fn cost_function_traces_session_moves(&self) -> bool {
        self.cost_function
            .as_ref()
            .is_some_and(|cf| cf.traces_session_moves())
    }

    /// The routes pass the E3 session key ([`PlaceInput::e3_session`]):
    /// the cost function decides with `--trade-wait e3`.
    pub fn cost_function_wants_e3_session(&self) -> bool {
        self.cost_function
            .as_ref()
            .is_some_and(|cf| cf.wants_e3_session())
    }

    /// The healthy P-role pool of the cost function (the pool of
    /// [`Self::admit`]); the global waiting queue steps over it. Under
    /// external role control it is the current P-role set: a draining
    /// worker (decode to prefill) and a D-role worker have no prefill weight.
    fn cost_function_pool(&self) -> Vec<Arc<Worker>> {
        let (weights, index) = self.candidates();
        prefill_pool(&self.cfg, &weights, &index)
    }

    /// The client ended `session` (`POST /kv_sessions/end`; the identity of
    /// [`crate::scheduler::trade::trade_session_key`]): it leaves
    /// N_active of the cost function's copy price `auto` at once. Returns
    /// whether it was counted.
    pub fn cost_function_session_end(&self, session: &str) -> bool {
        self.cost_function
            .as_ref()
            .is_some_and(|cf| cf.session_end(session))
    }

    /// The header that carries the decision-replay key, when
    /// `--symmetric-pd-decision-replay` is configured.
    pub fn replay_header(&self) -> Option<&str> {
        self.replay
            .as_ref()
            .map(|_| self.cfg.decision_replay_header.as_str())
    }

    /// Fresh transfer id (u64 in `[0, 2^63)`, like `bootstrap_room` was).
    /// Advance the runtime transfer counter used for IDs and D round robin.
    ///
    /// # Inputs
    /// `self`: randomly initialized atomic counter.
    ///
    /// # Outputs
    /// The previous counter value masked into `[0, 2^63)`; advancing it also advances
    /// seed-based D selection.
    pub fn mint_transfer_seed(&self) -> u64 {
        self.transfer_counter.fetch_add(1, Ordering::Relaxed) & (i64::MAX as u64)
    }

    /// Read the currently eligible worker registry view.
    ///
    /// # Inputs
    /// `self`: runtime and its Backend registry.
    ///
    /// # Outputs
    /// A role/capability map and URL-to-shared-worker map from Backend; no requests are placed
    /// or reserved.
    pub fn candidates(&self) -> (RoleWeights, HashMap<String, Arc<Worker>>) {
        self.backend.registry.candidates()
    }

    pub fn role_assignments(&self) -> Result<crate::backend::registry::RoleSnapshot, String> {
        self.backend.registry.assignments()
    }

    /// Replace the external role assignment. With `--symmetric-pd-role-drain-timeout-secs`,
    /// workers moving from decode to prefill drain first (see `scheduler::role_drain`).
    ///
    /// # Outputs
    /// The new snapshot and the workers whose drain it started, identified by
    /// the snapshot's revision.
    pub fn replace_role_assignments(
        &self,
        epoch: &str,
        revision: u64,
        expected: &crate::backend::registry::RoleAssignments,
        desired: &crate::backend::registry::RoleAssignments,
    ) -> Result<(crate::backend::registry::RoleSnapshot, Vec<String>), String> {
        if !self.config().external_role_control {
            return Err("external role control is disabled".into());
        }
        let drain = self.config().role_drain_timeout.is_some();
        if self.kv_sessions().is_some() && !drain {
            return Err("dynamic roles with KV session retention require --symmetric-pd-role-drain-timeout-secs".into());
        }
        let result = self
            .backend
            .registry
            .replace_assignments(epoch, revision, expected, desired, drain)?;
        if let Some(sessions) = self.kv_sessions() {
            // Prefill->decode: keep this worker's current-P prefixes ahead of its
            // new decode imports until their next P is READY.
            for (url, role) in desired {
                if *role == crate::backend::registry::RoleAssignment::Decode
                    && expected.get(url) == Some(&crate::backend::registry::RoleAssignment::Prefill)
                {
                    sessions.begin_worker_handover(url);
                }
            }
        }
        Ok(result)
    }

    /// Whether the drain that assignment `revision` started on `worker` is in progress.
    pub fn is_role_draining(&self, worker: &str, revision: u64) -> bool {
        self.backend.registry.is_draining(worker, revision)
    }

    /// End that drain so `worker` takes prefill legs; false if a later assignment ended it.
    pub fn finish_role_drain(&self, worker: &str, revision: u64) -> bool {
        self.backend.registry.finish_drain_if(worker, revision, || {
            if let Some(sessions) = self.kv_sessions() {
                sessions.begin_worker_clear(worker);
                sessions.finish_worker_clear(worker)
            } else {
                true
            }
        })
    }

    /// Install prepared work atomically with respect to dynamic role changes.
    pub(crate) fn with_current_role_pair<T>(
        &self,
        p: &str,
        d: &str,
        install: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        if self.config().external_role_control {
            self.backend.registry.with_current_pair(p, d, install)
        } else {
            install()
        }
    }

    /// Draining workers, with seconds since each drain started.
    pub fn role_drains(&self) -> std::collections::BTreeMap<String, f64> {
        self.backend.registry.drains()
    }

    /// `/readyz` condition: at least one prefill-capable and one
    /// decode-capable worker (weight > 0, breaker closed) are registered.
    /// Check the basic P/D capability condition for readiness.
    ///
    /// # Inputs
    /// `self`: runtime registry and current eligibility.
    ///
    /// # Outputs
    /// `true` when at least one eligible positive-weight P and one D exist. This is a
    /// capability check, not a guarantee of a distinct pair under force-split.
    pub fn has_capable_pair(&self) -> bool {
        let (weights, _) = self.candidates();
        let p = weights
            .workers
            .values()
            .any(|r| r.capabilities.prefill && r.weight.prefill > 0.0);
        let d = weights
            .workers
            .values()
            .any(|r| r.capabilities.decode && r.weight.decode > 0.0);
        p && d
    }

    /// Read effective work observations for selected workers.
    ///
    /// # Inputs
    /// `self`: Backend state; `workers`: worker identifiers/URLs to query.
    ///
    /// # Outputs
    /// A `WorkerLoads` snapshot obtained under the state lock, merging observations and
    /// reservations according to Backend rules; no network I/O occurs.
    pub fn loads(&self, workers: impl IntoIterator<Item = impl AsRef<str>>) -> WorkerLoads {
        self.backend.state.lock().loads(workers)
    }

    /// Count one D-leg on `worker`'s in-flight decode count (the backend
    /// ledger) until the returned guard drops: a late-D bind moves the
    /// arrival pick's count to the bound candidate.
    pub fn reserve_decode(&self, worker: &str) -> DecodeReservation {
        self.backend.state.lock().reserve_decode(worker)
    }

    /// Read the shared ledger/engine observations without placement, indexer
    /// requests or consumption of transfer IDs (and therefore the D RR counter).
    /// Expose detailed Backend observations without making a routing decision.
    ///
    /// # Inputs
    /// `self`: current Backend registry and shared state.
    ///
    /// # Outputs
    /// An ordered worker-to-load-view map for registered workers. Reads state under its lock
    /// without indexer I/O, reservations, or transfer-seed consumption.
    pub fn backend_load_views(
        &self,
    ) -> std::collections::BTreeMap<String, crate::backend::state::BackendLoadView> {
        let workers = self.backend.registry.workers();
        self.backend
            .state
            .lock()
            .views(workers.iter().map(|w| w.url.as_str()))
    }

    // Accounting is derived from validated directory/pull information, not a
    // custom policy's score/log fields. Changing a cost model cannot forge hit
    // tokens or change the units of Backend's reservation ledger.
    /// Derive the token hit used by authoritative reservation accounting.
    ///
    /// # Inputs
    /// `self`: host-hit weighting and P mode; `worker`: chosen P URL; `prompt_len`: request
    /// tokens; `hits`: directory hits plus validated peer-pull estimates.
    ///
    /// # Outputs
    /// Local usable hit, or the larger validated expected pull hit in CoSchedule mode (and with
    /// the cost function: its TRADE expected hit), bounded by prompt length on normal nonempty
    /// requests. Policy score fields are never trusted for accounting.
    fn accounted_hit(&self, worker: &str, prompt_len: u64, hits: &IndexerHits) -> u64 {
        let hit = hits.get(worker);
        let local = (hit.gpu as f64)
            .max(hit.host as f64 * self.cfg.placement.host_hit_factor)
            .min(prompt_len.max(1) as f64) as u64;
        if matches!(
            self.cfg.placement.p_select,
            PSelect::CoSchedule | PSelect::CostFunction
        ) {
            hits.expected_after_pull
                .get(worker)
                .copied()
                .unwrap_or(local)
                .max(local)
                .min(prompt_len)
        } else {
            local
        }
    }

    /// Indexer hits for this request (empty when no indexer / no key).
    /// Query request-specific cache hits and discard the raw indexer reply.
    ///
    /// # Inputs
    /// `self`: optional indexer client; `input`: token IDs or hashes plus model/cache salt.
    ///
    /// # Outputs
    /// The parsed hit map and indexer duration in milliseconds; empty hits and zero duration if
    /// no usable lookup result exists.
    pub async fn hits(&self, input: &PlaceInput) -> (IndexerHits, f64) {
        let (hits, ms, _) = self.hits_with_reply(input).await;
        (hits, ms)
    }

    /// [`Self::hits`] plus the raw indexer reply (for the pull hint).
    /// Query the external KV directory before acquiring the placement lock.
    ///
    /// # Inputs
    /// `self`: optional indexer and fallback model; `input`: prompt IDs/hashes, salt, and
    /// optional model name.
    ///
    /// # Outputs
    /// Parsed hits, query milliseconds, and optional raw JSON for pull-hint construction.
    /// Missing indexer or unsuccessful lookup returns `(empty, 0.0, None)` so placement can
    /// continue.
    pub async fn hits_with_reply(&self, input: &PlaceInput) -> (IndexerHits, f64, Option<Value>) {
        let Some(ix) = &self.indexer else {
            return (IndexerHits::new(), 0.0, None);
        };
        let model = input.model_id.as_deref().unwrap_or(&self.model.0);
        match ix
            .query_indexer(
                input.tokens.as_deref(),
                input.hashes.as_deref(),
                input.cache_salt.as_deref(),
                Some(model),
            )
            .await
        {
            Some((reply, ms)) => (self.parse_hits(&reply), ms, Some(reply)),
            None => (IndexerHits::new(), 0.0, None),
        }
    }

    /// Construct validated peer-pull hints for eligible P destinations.
    ///
    /// # Inputs
    /// `self`: pull enablement and indexer mapping; `reply`: optional raw query reply;
    /// `indexer_ms`: query latency; `weights`: current candidates; `hits`: mutable request hit
    /// view.
    ///
    /// # Outputs
    /// Destination URL to pull-hint JSON. Adds advertised expected hits to
    /// `hits.expected_after_pull`; returns an empty map if pulls, indexer, or reply are
    /// unavailable. This does not transfer KV.
    fn candidate_pull_hints(
        &self,
        reply: Option<&Value>,
        indexer_ms: f64,
        weights: &RoleWeights,
        hits: &mut IndexerHits,
    ) -> HashMap<String, Value> {
        let mut hints = HashMap::new();
        if !self.cfg.pull_hint {
            return hints;
        }
        let (Some(ix), Some(reply)) = (&self.indexer, reply) else {
            return hints;
        };
        let answer = ix.answer_from_reply(reply, indexer_ms);
        for (url, role) in &weights.workers {
            if !role.capabilities.prefill || role.weight.prefill <= 0.0 {
                continue;
            }
            if let Some(hint) = ix.pick(&answer, url) {
                if let Some(expected) = hint.get("expected_hit_tokens").and_then(Value::as_u64) {
                    hits.expected_after_pull.insert(url.clone(), expected);
                }
                hints.insert(url.clone(), hint);
            }
        }
        hints
    }

    /// Reply shape: `{"instances": {"<id>": {"gpu": t, "cpu": t}}}`, tokens.
    /// Instance ids map to worker URLs through `--kv-worker-map`; keys are
    /// the registered worker URLs (trailing-slash insensitive).
    /// Convert indexer instance summaries into worker-keyed prefix counts.
    ///
    /// # Inputs
    /// `self`: indexer instance mapping and registry; `reply`: JSON with an `instances` object
    /// and GPU/CPU/host token counts.
    ///
    /// # Outputs
    /// Nonzero hit entries keyed by canonical registered URLs where available;
    /// missing/malformed/unmapped instances are skipped and missing tier counts are zero.
    fn parse_hits(&self, reply: &Value) -> IndexerHits {
        let mut hits = IndexerHits::new();
        let (Some(ix), Some(instances)) = (
            &self.indexer,
            reply.get("instances").and_then(Value::as_object),
        ) else {
            return hits;
        };
        let by_norm: HashMap<String, String> = self
            .backend
            .registry
            .workers()
            .iter()
            .map(|w| (w.url.trim_end_matches('/').to_string(), w.url.clone()))
            .collect();
        for (id, summary) in instances {
            let Ok(id) = id.parse::<u64>() else { continue };
            let Some(url) = ix.worker_url_for(id) else {
                debug!(target: "symmetric_pd", worker_id = id, "unmapped indexer instance");
                continue;
            };
            let key = by_norm
                .get(url.trim_end_matches('/'))
                .cloned()
                .unwrap_or_else(|| url.to_string());
            let tier = |k: &str| summary.get(k).and_then(Value::as_u64).unwrap_or(0);
            let hit = HitTokens {
                gpu: tier("gpu"),
                host: tier("cpu").max(tier("host")),
            };
            if hit.gpu > 0 || hit.host > 0 {
                hits.insert(key, hit);
            }
        }
        hits
    }

    /// Is at least one P-leg candidate under the backlog threshold?
    /// Check the legacy HOLD admission threshold across eligible P workers.
    ///
    /// # Inputs
    /// `self`: configured maximum uncached tokens; `weights`: capabilities/role weights;
    /// `loads`: current backlog snapshot.
    ///
    /// # Outputs
    /// `true` if any eligible P has backlog strictly below the threshold; no capacity is
    /// reserved.
    fn has_prefill_capacity(&self, weights: &RoleWeights, loads: &WorkerLoads) -> bool {
        weights
            .workers
            .iter()
            .filter(|(_, r)| r.capabilities.prefill && r.weight.prefill > 0.0)
            .any(|(url, _)| loads.get(url).backlog_uncached_tokens < self.cfg.max_uncached_tokens)
    }

    /// Place one request for dispatch by this router: indexer lookup, hold
    /// queue, `place()`, ledger accounting, marks, decision log.
    /// Query cache state, atomically select/reserve P work, and construct dispatch metadata.
    ///
    /// # Inputs
    /// `self`: runtime, Backend and policy
    /// (unused); `input`: request metadata and optional cache lookup keys.
    ///
    /// # Outputs
    /// `Ok(Decision)` with selected workers, marks, pull hint, timing and an owned ledger
    /// handle; errors for no workers, invalid/impossible placement, or missing KV endpoints.
    /// The caller must retire/hand off the reservation. CoSchedule bypasses HOLD; legacy modes
    /// may wait up to the configured deadline before forced placement.
    pub async fn admit(&self, input: PlaceInput) -> Result<Decision, SymmetricPdError> {
        let arrived = Instant::now();
        let (mut hits, indexer_ms, reply) = self.hits_with_reply(&input).await;
        let deadline = arrived + Duration::from_millis(self.cfg.max_queue_wait_ms);
        let prompt_len = input.prompt_len.max(1);
        let (pin, replay) = match &self.replay {
            Some(r) => r.next(input.routing_key.as_deref(), prompt_len),
            None => (None, "off"),
        };
        let (
            placement,
            p_worker,
            d_worker,
            marks,
            ledger,
            d_reservation,
            pull_hint,
            forced,
            cost_reservation,
        ) = loop {
            let (weights, index) = self.candidates();
            if weights.workers.is_empty() {
                return Err(SymmetricPdError::NoWorkers);
            }
            // `--symmetric-pd-p-select gateway-cache-aware`: the gateway rule
            // picks P first (its tree insert is the gateway's, at selection);
            // the placement then sees that P only.
            let weights = match &self.gateway_ca {
                Some(ca) => gateway_ca_narrow(ca, &self.cfg, &input, weights, &index)?,
                None => weights,
            };
            // `--symmetric-pd-p-select cost-function`: one cost-book transaction
            // picks P and its pull among the healthy P-role workers and
            // reserves the request's work (the cost book has its own lock and
            // no await inside, so concurrent requests see each other's
            // reservations); the placement then sees that P only.
            let (weights, cost, index) = match &self.cost_function {
                Some(cf) => {
                    let ix = self.indexer.as_ref().ok_or_else(|| {
                        SymmetricPdError::CostFunction(
                            "--symmetric-pd-p-select cost-function requires --kv-indexer-url"
                                .into(),
                        )
                    })?;
                    let pool = prefill_pool(&self.cfg, &weights, &index);
                    if pool.is_empty() {
                        return Err(SymmetricPdError::Placement(
                            PlacementError::NoPrefillCandidate,
                        ));
                    }
                    // `--symmetric-pd-gwq-auto`: a free request may wait in the
                    // global waiting queue; after the wait the placement
                    // reads the current workers. A released P that left the
                    // P-role pool in between (a role switch, a drain or a
                    // breaker trip) is not placed: its reservation rolls
                    // back (never sent, so the rid is free again) and the
                    // request is walked again over the current pool.
                    // `--symmetric-pd-pull-from-any-holder`: the pull sources are the
                    // registered workers of any role that the registry offers
                    // as candidates (breaker allowing when the breaker is on;
                    // draining ones too); never a holder removed from the
                    // registry but still in the indexer. With
                    // --disable-circuit-breaker (the CUA deployments) a crashed or hung
                    // worker stays a possible source until it is removed. A
                    // re-walk after a queue wait reads them again with the pool.
                    let (selected, reservation, weights, index) = if cf.gwq_enabled() {
                        let (mut pool, mut weights, mut index) = (pool, weights, index);
                        let mut attempt = 0u32;
                        loop {
                            let donors =
                                any_holder_donors(cf.pull_from_any_holder(), index.values());
                            let (selected, reservation, _, queued) = cf
                                .reserve_gwq(
                                    ix,
                                    &input,
                                    prompt_len,
                                    reply.as_ref(),
                                    indexer_ms,
                                    &pool,
                                    &donors,
                                )
                                .await?;
                            if !queued {
                                break (selected, reservation, weights, index);
                            }
                            let (now_weights, now_index) = self.candidates();
                            let now_pool = prefill_pool(&self.cfg, &now_weights, &now_index);
                            if now_pool.iter().any(|w| w.url == selected.worker_url) {
                                break (selected, reservation, now_weights, now_index);
                            }
                            attempt += 1;
                            cf.gwq_note_rewalk(input.rid.as_deref(), &selected.worker_url, attempt);
                            drop(reservation);
                            if now_pool.is_empty() {
                                return Err(SymmetricPdError::Placement(
                                    PlacementError::NoPrefillCandidate,
                                ));
                            }
                            if attempt > GWQ_REWALKS {
                                return Err(SymmetricPdError::CostFunction(format!(
                                    "the global waiting queue released the request {attempt} times \
                                     to a P that left the P-role pool"
                                )));
                            }
                            (pool, weights, index) = (now_pool, now_weights, now_index);
                        }
                    } else {
                        let donors = any_holder_donors(cf.pull_from_any_holder(), index.values());
                        let (selected, reservation, _) = cf.reserve(
                            ix,
                            &input,
                            prompt_len,
                            reply.as_ref(),
                            indexer_ms,
                            &pool,
                            &donors,
                        )?;
                        (selected, reservation, weights, index)
                    };
                    // The Backend ledger charges prompt - TRADE expected hit.
                    hits.expected_after_pull
                        .insert(selected.worker_url.clone(), selected.expected_hit);
                    let weights = narrow_prefill(weights, &selected.worker_url);
                    (weights, Some((selected, reservation)), index)
                }
                None => (weights, None, index),
            };
            // The cost function's pull is the one it priced (`cost`).
            let pull_hints = if cost.is_some() {
                HashMap::new()
            } else {
                self.candidate_pull_hints(reply.as_ref(), indexer_ms, &weights, &mut hits)
            };
            // Snapshot, choose and reserve are one critical section. No
            // network await is allowed while Backend state is locked.
            let hashes = input.hashes.clone().unwrap_or_default();
            let session_id = input.session_id.clone();
            let rid = input.rid.clone();
            let attempt = {
                let mut state = self.backend.state.lock();
                let loads = state.loads(weights.workers.keys());
                let now = Instant::now();
                // CoSchedule, the gateway cache-aware P selection and the
                // cost function never hold (no hold queue).
                let has_capacity = self.cfg.placement.p_select == PSelect::CoSchedule
                    || self.gateway_ca.is_some()
                    || self.cost_function.is_some()
                    || self.has_prefill_capacity(&weights, &loads);
                if !has_capacity && now < deadline {
                    None
                } else {
                    let transfer_seed = self.mint_transfer_seed();
                    let pin = pin.clone();
                    let req = PlacementRequest {
                        hashes,
                        prompt_len,
                        session_id,
                        turn: input.turn,
                        rid,
                        transfer_seed,
                        pin,
                    };
                    let placement =
                        self.policy
                            .place(&req, &hits, &loads, &weights, &self.cfg.placement)?;
                    validate_placement(&placement, &req, &weights, &self.cfg.placement)?;
                    if let Some((selected, _)) = &cost {
                        if placement.p_leg != selected.worker_url {
                            let reason = format!(
                                "cost_function chose {} but the placement took {}",
                                selected.worker_url, placement.p_leg
                            );
                            return Err(PlacementError::InvalidPolicyOutput(reason).into());
                        }
                    }
                    let p = index.get(&placement.p_leg).expect("candidate P");
                    let d = index.get(&placement.d_leg).expect("candidate D");
                    // Validate protocol endpoints before creating a reservation.
                    let marks = self.marks(&placement, p, d, prompt_len)?;
                    let hit_p = self.accounted_hit(&p.url, prompt_len, &hits);
                    let ledger = state.reserve_request(
                        &p.url,
                        prompt_len.saturating_sub(hit_p),
                        input.rid.as_deref(),
                        prompt_len,
                    );
                    // Counted in the same critical section as the choice, so
                    // concurrent placements see it.
                    let d_reservation = state.reserve_decode(&d.url);
                    let pull_hint = match &cost {
                        Some((selected, _)) => selected.hint.clone(),
                        None => pull_hints.get(&p.url).cloned(),
                    };
                    Some((
                        placement,
                        Arc::clone(p),
                        Arc::clone(d),
                        marks,
                        ledger,
                        Some(d_reservation),
                        pull_hint,
                        !has_capacity,
                        cost.map(|(_, reservation)| reservation),
                    ))
                }
            };
            if let Some(result) = attempt {
                break result;
            }
            let remaining = deadline
                .saturating_duration_since(Instant::now())
                .min(HOLD_TICK);
            let _ = tokio::time::timeout(remaining, self.backend.changed.notified()).await;
        };

        if let Some(h) = &pull_hint {
            // `Value` inside the macro would resolve to tracing's field trait.
            let source_worker = h
                .get("source_worker_id")
                .and_then(Value::as_str)
                .unwrap_or("-")
                .to_string();
            let expected_hit_tokens = h
                .get("expected_hit_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            info!(
                target: "symmetric_pd",
                rid = %input.rid.as_deref().unwrap_or("-"),
                p_worker = %p_worker.url,
                source_worker = %source_worker,
                expected_hit_tokens,
                "symmetric_pd pull hint on P-leg"
            );
        }
        let queue_wait_ms = arrived.elapsed().as_millis() as u64;
        log_decision(
            &placement,
            &input,
            queue_wait_ms,
            forced,
            indexer_ms,
            "dispatch",
            replay,
        );
        Ok(Decision {
            placement,
            p_worker,
            d_worker,
            marks,
            ledger,
            d_reservation,
            queue_wait_ms,
            forced,
            indexer_ms,
            prompt_len,
            pull_hint,
            cost_reservation,
        })
    }

    /// Create a raw-token reservation through the compatibility API.
    ///
    /// # Inputs
    /// `self`: Backend state; `worker`: target URL;
    /// `tokens`: uncached work to reserve.
    ///
    /// # Outputs
    /// A `LedgerHandle` for the newly reserved work; callers manage its dispatch/retirement
    /// lifecycle.
    pub fn account(&self, worker: &str, tokens: u64) -> LedgerHandle {
        self.backend.state.lock().reserve(worker, tokens)
    }

    #[cfg(test)]
    /// Inject an aggregate engine observation for runtime unit tests.
    ///
    /// # Inputs
    /// `self`: runtime Backend; `worker`: URL; `uncached`: waiting token count;
    /// `running`/`waiting`: request counts; `sent_at`: poll start time.
    ///
    /// # Outputs
    /// No return value; Backend applies the observation and reconciles eligible older
    /// reservations.
    fn apply_poll(
        &self,
        worker: &str,
        uncached: u64,
        running: u64,
        waiting: u64,
        sent_at: Instant,
    ) {
        self.backend
            .apply_poll(worker, uncached, running, waiting, sent_at);
    }

    /// Resolve a worker's SGLang KV owner and reader side channels.
    ///
    /// # Inputs
    /// `self`: Backend SGLang adapter; `worker`: registered metadata.
    ///
    /// # Outputs
    /// `Ok((owner_address, reader_address))`, or a scheduler endpoint error when the adapter
    /// cannot resolve them; no network transfer is initiated.
    pub fn endpoints(&self, worker: &Worker) -> Result<(String, String), SymmetricPdError> {
        self.backend.adapter.endpoints(worker).map_err(Into::into)
    }

    /// Build the SGLang KV export/import metadata for a placement.
    ///
    /// # Inputs
    /// `self`: Backend adapter/cache policy; `placement`: selected pair/transfer seed;
    /// `p_worker`/`d_worker`: engine metadata; `prompt_len`: expected KV token count.
    ///
    /// # Outputs
    /// `Ok(Marks)` (empty for Local, matching transfer/attempt metadata for Split), or an
    /// endpoint error. This constructs protocol JSON; it does not move KV.
    pub fn marks(
        &self,
        placement: &Placement,
        p_worker: &Worker,
        d_worker: &Worker,
        prompt_len: u64,
    ) -> Result<Marks, SymmetricPdError> {
        let host_return = self.host_return_for(placement, p_worker, d_worker)?;
        let mut marks = self
            .backend
            .adapter
            .marks(placement, p_worker, d_worker, prompt_len)
            .map_err(SymmetricPdError::from)?;
        if let Some((destination, source, timeout_ms, max_tokens)) = host_return {
            self.backend.adapter.add_host_return(
                &mut marks,
                destination,
                source,
                timeout_ms,
                max_tokens,
            );
        }
        Ok(marks)
    }

    /// The router chooses the original P as the return destination. Engines only
    /// execute this deferred command once generated KV has reached their host pool.
    /// Keep this after placement: no extra directory query, transfer or load ledger.
    fn host_return_for<'a>(
        &'a self,
        placement: &Placement,
        p_worker: &Worker,
        d_worker: &Worker,
    ) -> Result<Option<(&'a str, &'a str, u64, u64)>, SymmetricPdError> {
        if placement.is_local() {
            return Ok(None);
        }
        self.host_return_pair(p_worker, d_worker)
    }

    /// The return directive for one (P, D) pair of a split request, or `None`
    /// when decode host return is off. Shared by the placement marks and the
    /// late-D candidate imports (each candidate D returns to the same P).
    fn host_return_pair<'a>(
        &'a self,
        p_worker: &Worker,
        d_worker: &Worker,
    ) -> Result<Option<(&'a str, &'a str, u64, u64)>, SymmetricPdError> {
        let Some(cfg) = self.cfg.decode_host_return.as_ref() else {
            return Ok(None);
        };
        for worker in [p_worker, d_worker] {
            if worker.mode() != crate::discovery::WorkerMode::Symmetric {
                return Err(SymmetricPdError::Endpoints {
                    worker: worker.url.clone(),
                    reason: "decode host return requires symmetric peer-HiCache engines".into(),
                });
            }
        }
        if self.cfg.import_cache_policy != crate::config::SymmetricPdImportCachePolicy::Retain {
            return Err(SymmetricPdError::Endpoints {
                worker: d_worker.url.clone(),
                reason: "decode host return requires retain cache policy".into(),
            });
        }
        let endpoint = |worker: &Worker| {
            cfg.servicers
                .get(worker.url.trim_end_matches('/'))
                .map(String::as_str)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| SymmetricPdError::Endpoints {
                    worker: worker.url.clone(),
                    reason: "missing configured peer servicer for decode host return".into(),
                })
        };
        let destination = endpoint(p_worker)?;
        let source = endpoint(d_worker)?;
        if destination == source {
            return Err(SymmetricPdError::Endpoints {
                worker: d_worker.url.clone(),
                reason: "distinct workers must have distinct peer servicers".into(),
            });
        }
        Ok(Some((destination, source, cfg.timeout_ms, cfg.max_tokens)))
    }

    /// Late D choose on symmetric workers: the P-leg's candidate
    /// `kv_export` and one `kv_import` per candidate (each with its own
    /// attempt id) for `transfer_id`. Protocol metadata only.
    pub fn late_marks(
        &self,
        transfer_id: u64,
        p_worker: &Worker,
        candidates: &[Arc<Worker>],
        prompt_len: u64,
    ) -> Result<(Value, Vec<Value>), SymmetricPdError> {
        let (export, mut imports) = self
            .backend
            .adapter
            .late_marks(transfer_id, p_worker, candidates, prompt_len)
            .map_err(SymmetricPdError::from)?;
        // Decode host return: every candidate D-leg carries the directive to
        // the same P (the destination); only the bound candidate decodes, so
        // only it returns. A candidate equal to the P has no split to return.
        for (import, d_worker) in imports.iter_mut().zip(candidates) {
            if d_worker.url == p_worker.url {
                continue;
            }
            if let Some((destination, source, timeout_ms, max_tokens)) =
                self.host_return_pair(p_worker, d_worker)?
            {
                import["host_return"] = serde_json::json!({
                    "destination_servicer_addr": destination,
                    "source_servicer_addr": source,
                    "timeout_ms": timeout_ms,
                    "max_tokens": max_tokens,
                });
            }
        }
        Ok((export, imports))
    }

    /// The decode-capable workers of the current roles (decode capability
    /// and weight > 0, breaker allowing), in URL order: the late-D
    /// candidate pool of symmetric workers.
    pub fn decode_workers(&self) -> Vec<Arc<Worker>> {
        role_members(self.candidates(), decode_role)
    }

    /// The decode-capable workers of the current roles whatever their
    /// breaker state, in URL order: the native capacity poll of symmetric
    /// workers. As the PD poll (every decode worker), a breaker trip never
    /// drops a decode's sample while requests that name it are in flight.
    pub fn decode_members(&self) -> Vec<Arc<Worker>> {
        role_members(self.backend.registry.members(), decode_role)
    }

    /// The prefill-capable workers of the current roles whatever their
    /// breaker state, in URL order: the owners of late-D links on symmetric
    /// workers. As the PD links (every prefill worker), a breaker trip never
    /// stops the link of a P whose requests still wait for their decisions.
    pub fn prefill_members(&self) -> Vec<Arc<Worker>> {
        role_members(self.backend.registry.members(), |r| {
            r.capabilities.prefill && r.weight.prefill > 0.0
        })
    }

    /// Every role-capable worker whatever its breaker state or current role
    /// (draining ones too), in URL order: the late-D link owners under
    /// external role control. A role change does not drop a link, so a
    /// P-leg still queued on a worker the controller moved to decode keeps
    /// asking this router for its winner.
    pub fn role_members_all(&self) -> Vec<Arc<Worker>> {
        role_members(self.backend.registry.members(), |r| {
            r.capabilities.prefill || r.capabilities.decode
        })
    }

    /// Placement over HTTP for an external ingress (`POST /symmetric_pd/place`,
    /// `docs/api-reference.md`). No hold queue (the ingress dispatches on its own
    /// schedule); the P-leg's uncached tokens are added to the ledger so
    /// consecutive placements between two polls see each other.
    /// Return a dispatch plan for the legacy external-ingress placement endpoint.
    ///
    /// # Inputs
    /// `self`: runtime/Backend; `req`: input
    /// descriptors and optional force-split, accounting, and engine-route overrides.
    ///
    /// # Outputs
    /// `Ok(PlaceHttpResponse)` with engine targets, marks and diagnostic snapshots; typed
    /// placement/endpoint errors otherwise. Performs indexer I/O and atomic select/reserve
    /// without HOLD; default accounting creates an expiring lease because this API has no
    /// completion callback. It does not dispatch generation.
    pub async fn place_http(
        &self,
        req: PlaceHttpRequest,
    ) -> Result<PlaceHttpResponse, SymmetricPdError> {
        // The cost ledger needs each request's dispatch and end; this API
        // reports neither (the route refuses first).
        if self.cost_function.is_some() {
            return Err(SymmetricPdError::CostFunction(
                super::cost_function::PLACE_REFUSAL.into(),
            ));
        }
        let prompt_len = req
            .prompt_len
            .or_else(|| req.input_ids.as_ref().map(|t| t.len() as u64))
            .unwrap_or(0)
            .max(1);
        let input = PlaceInput {
            tokens: req.input_ids,
            hashes: req.hashes,
            prompt_len,
            cache_salt: req.cache_salt,
            model_id: req.model,
            session_id: req.session_id,
            turn: req.turn,
            rid: req.rid,
            routing_key: None,
            ingress_ms: 0.0,
            trade_session: None,
            e3_session: None,
        };
        let (mut hits, indexer_ms, reply) = self.hits_with_reply(&input).await;
        let (weights, index) = self.candidates();
        if weights.workers.is_empty() {
            return Err(SymmetricPdError::NoWorkers);
        }
        let pull_hints = self.candidate_pull_hints(reply.as_ref(), indexer_ms, &weights, &mut hits);
        let mut state = self.backend.state.lock();
        let loads = state.loads(weights.workers.keys());
        let mut cfg = self.cfg.placement.clone();
        if let Some(force) = req.force_split {
            cfg.force_split = force;
        }
        let placement_req = PlacementRequest {
            hashes: input.hashes.clone().unwrap_or_default(),
            prompt_len,
            session_id: input.session_id.clone(),
            turn: input.turn,
            rid: input.rid.clone(),
            transfer_seed: self.mint_transfer_seed(),
            pin: None,
        };
        let placement = self
            .policy
            .place(&placement_req, &hits, &loads, &weights, &cfg)?;
        validate_placement(&placement, &placement_req, &weights, &cfg)?;
        let (p, d) = (
            index
                .get(&placement.p_leg)
                .expect("p_leg from candidate set"),
            index
                .get(&placement.d_leg)
                .expect("d_leg from candidate set"),
        );
        let marks = self.marks(&placement, p, d, prompt_len)?;
        if req.account.unwrap_or(true) {
            let hit_p = self.accounted_hit(&p.url, prompt_len, &hits);
            let _ = state.reserve_lease(&p.url, prompt_len.saturating_sub(hit_p));
        }
        let observations: HashMap<_, _> = weights
            .workers
            .keys()
            .map(|w| (w.clone(), state.view(w)))
            .collect();
        drop(state);
        log_decision(&placement, &input, 0, false, indexer_ms, "http", "off");

        let route = req.route.unwrap_or_else(|| "/generate".to_string());
        let endpoint = |w: &Arc<Worker>| format!("{}{}", w.url.trim_end_matches('/'), route);
        let targets = if placement.is_local() {
            vec![PlaceTarget {
                leg: "local".to_string(),
                worker: p.url.clone(),
                endpoint: endpoint(p),
                mark: pull_hints
                    .get(&p.url)
                    .map(|hint| json!({KV_PULL_HINT_KEY:hint}))
                    .unwrap_or_else(|| json!({})),
            }]
        } else {
            vec![
                PlaceTarget {
                    leg: "p".to_string(),
                    worker: p.url.clone(),
                    endpoint: endpoint(p),
                    mark: {
                        let mut mark = json!({ KV_EXPORT_KEY: marks.kv_export.clone() });
                        if let Some(hint) = pull_hints.get(&p.url) {
                            mark[KV_PULL_HINT_KEY] = hint.clone();
                        }
                        mark
                    },
                },
                PlaceTarget {
                    leg: "d".to_string(),
                    worker: d.url.clone(),
                    endpoint: endpoint(d),
                    mark: json!({ KV_IMPORT_KEY: marks.kv_import.clone() }),
                },
            ]
        };
        Ok(PlaceHttpResponse {
            p_leg: PlaceLeg {
                worker: p.url.clone(),
                endpoint: endpoint(p),
            },
            d_leg: PlaceLeg {
                worker: d.url.clone(),
                endpoint: endpoint(d),
            },
            local: placement.is_local(),
            transfer_id: placement.transfer_id,
            score_terms: placement.score_terms.clone(),
            marks,
            targets,
            inputs: PlaceInputsEcho {
                prompt_len,
                hits: hits
                    .workers
                    .iter()
                    .map(|(k, v)| (k.clone(), json!({ "gpu": v.gpu, "host": v.host })))
                    .collect(),
                loads: loads
                    .workers
                    .iter()
                    .map(|(k, v)| {
                        (
                            k.clone(),
                            json!({
                                "backlog_uncached_tokens": v.backlog_uncached_tokens,
                                "running_decode": v.running_decode,
                                "waiting": v.waiting,
                                "active_prefill_reqs": v.active_prefill_reqs,
                                "active_decode_reqs": v.active_decode_reqs,
                                "engine_observation": observations.get(k),
                            }),
                        )
                    })
                    .collect(),
                indexer_ms,
            },
        })
    }
}

/// Emit one structured routing-decision record.
///
/// # Inputs
/// `placement`: selected pair and score terms; `input`: request identity/size/ingress timing;
/// `queue_wait_ms`: elapsed admission time; `forced`: HOLD bypass flag; `indexer_ms`: lookup
/// time; `via`/`replay`: path and replay outcome labels.
///
/// # Outputs
/// No return value; emits a `symmetric_pd` tracing event containing selection, cache, load, and
/// timing diagnostics.
fn log_decision(
    placement: &Placement,
    input: &PlaceInput,
    queue_wait_ms: u64,
    forced: bool,
    indexer_ms: f64,
    via: &'static str,
    replay: &'static str,
) {
    let t = |name: &str| placement.term(name).unwrap_or(0.0);
    info!(
        target: "symmetric_pd",
        via,
        replay,
        rid = input.rid.as_deref().unwrap_or("-"),
        session_id = input.session_id.as_deref().unwrap_or("-"),
        turn = input.turn.map(|t| t as i64).unwrap_or(-1),
        transfer_id = placement.transfer_id.map(|t| t as i64).unwrap_or(-1),
        branch = if placement.is_local() { "local" } else { "split" },
        forced,
        p_leg = %placement.p_leg,
        d_leg = %placement.d_leg,
        prompt_len = input.prompt_len,
        hit_tokens = t(TERM_HIT_TOKENS),
        expected_hit_tokens = placement.term(TERM_EXPECTED_HIT_TOKENS).unwrap_or_else(||t(TERM_HIT_TOKENS)),
        reuse_gain = t(TERM_REUSE_GAIN),
        backlog = t(TERM_BACKLOG),
        role_pen_p = t(TERM_ROLE_PEN_P),
        role_pen_d = t(TERM_ROLE_PEN_D),
        xfer_cost = t(TERM_XFER_COST),
        decode_load = t(TERM_DECODE_LOAD),
        p_reason = p_reason_name(t(TERM_P_REASON)),
        d_select = d_select_name(t(TERM_D_SELECT)),
        score = t(TERM_SCORE),
        queue_wait_ms,
        indexer_ms,
        ingress_ms = input.ingress_ms,
        "symmetric_pd placement"
    );
}

/// The workers of a role view whose role passes `keep`, in URL order.
fn role_members(
    (weights, index): (RoleWeights, HashMap<String, Arc<Worker>>),
    keep: impl Fn(&placement::WorkerRole) -> bool,
) -> Vec<Arc<Worker>> {
    weights
        .workers
        .iter()
        .filter(|&(_, r)| keep(r))
        .filter_map(|(url, _)| index.get(url).cloned())
        .collect()
}

/// A decode-capable role with a decode weight.
fn decode_role(r: &placement::WorkerRole) -> bool {
    r.capabilities.decode && r.weight.decode > 0.0
}

/// `--symmetric-pd-p-select gateway-cache-aware`: run the gateway cache-aware
/// rule over the prefill-capable candidates (URL order; under force split
/// only those with a decode-capable peer) on the request's token ids, and
/// leave every other prefill-capable worker without the prefill capability,
/// so the placement takes the gateway's pick. The decode side is unchanged.
/// The pick logs one `prefill policy decision` line (target
/// `prefill_policy`, as `--policy gateway_cache_aware`).
///
/// # Outputs
/// The narrowed roles, or `NoWorkers` when the rule has no healthy candidate.
/// With no prefill-capable candidate the roles are returned unchanged and the
/// placement reports the error.
fn gateway_ca_narrow(
    ca: &crate::policies::gateway_cache_aware::GatewayCacheAwarePolicy,
    cfg: &SymmetricPdConfig,
    input: &PlaceInput,
    weights: RoleWeights,
    index: &HashMap<String, Arc<Worker>>,
) -> Result<RoleWeights, SymmetricPdError> {
    let pool = prefill_pool(cfg, &weights, index);
    if pool.is_empty() {
        return Ok(weights);
    }
    let (chosen, note) =
        ca.select_for_tokens(&pool, input.tokens.as_deref(), input.cache_salt.as_deref());
    let chosen = chosen.ok_or(SymmetricPdError::NoWorkers)?;
    let chosen_url = pool[chosen].url.clone();
    info!(
        target: "prefill_policy",
        rid = input.rid.as_deref().unwrap_or(""),
        prefill_worker = %chosen_url,
        note = %note,
        "prefill policy decision",
    );
    Ok(narrow_prefill(weights, &chosen_url))
}

/// `--symmetric-pd-pull-from-any-holder` (`on`): the possible pull sources
/// besides the P pool, i.e. the registered workers of any role that the
/// registry offers as candidates (`workers`: breaker allowing when the
/// breaker is on, draining ones too). Empty when off.
fn any_holder_donors<'a>(
    on: bool,
    workers: impl Iterator<Item = &'a Arc<Worker>>,
) -> Vec<Arc<Worker>> {
    if on {
        workers.cloned().collect()
    } else {
        Vec::new()
    }
}

/// The P pool of the gateway cache-aware and the cost-function P
/// selections: the prefill-capable candidates with a prefill weight, in URL
/// order; under force split only those with a decode-capable peer.
fn prefill_pool(
    cfg: &SymmetricPdConfig,
    weights: &RoleWeights,
    index: &HashMap<String, Arc<Worker>>,
) -> Vec<Arc<Worker>> {
    weights
        .workers
        .iter()
        .filter(|(_, r)| r.capabilities.prefill && r.weight.prefill > 0.0)
        .filter(|(url, _)| {
            !cfg.placement.force_split
                || weights
                    .workers
                    .iter()
                    .any(|(d, r)| d != *url && decode_role(r))
        })
        .filter_map(|(url, _)| index.get(url).cloned())
        .collect()
}

/// Leave every worker except `chosen` without the prefill capability, so
/// the placement takes `chosen` as the P-leg. The decode side is unchanged.
fn narrow_prefill(mut weights: RoleWeights, chosen: &str) -> RoleWeights {
    for (url, role) in weights.workers.iter_mut() {
        if url != chosen {
            role.capabilities.prefill = false;
        }
    }
    weights
}

/// `POST /symmetric_pd/place` request (see `docs/api-reference.md`).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PlaceHttpRequest {
    #[serde(default)]
    pub hashes: Option<Vec<u64>>,
    #[serde(default)]
    pub input_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub prompt_len: Option<u64>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub turn: Option<u32>,
    #[serde(default)]
    pub rid: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub cache_salt: Option<String>,
    /// Override the configured `force_split` for this call.
    #[serde(default)]
    pub force_split: Option<bool>,
    /// Add the P-leg's uncached tokens to the router's backlog ledger
    /// (default true).
    #[serde(default)]
    pub account: Option<bool>,
    /// Engine route the targets should hit (default `/generate`).
    #[serde(default)]
    pub route: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlaceLeg {
    pub worker: String,
    pub endpoint: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlaceTarget {
    /// `"p"`, `"d"` or `"local"`.
    pub leg: String,
    pub worker: String,
    pub endpoint: String,
    /// Object to merge into the request body sent to `endpoint`
    /// (`{"kv_export": {...}}`, `{"kv_import": {...}}`, or `{}`).
    pub mark: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlaceInputsEcho {
    pub prompt_len: u64,
    pub hits: HashMap<String, Value>,
    pub loads: HashMap<String, Value>,
    pub indexer_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlaceHttpResponse {
    pub p_leg: PlaceLeg,
    pub d_leg: PlaceLeg,
    pub local: bool,
    pub transfer_id: Option<u64>,
    /// Ordered `[["hit_tokens", ..], ["reuse_gain", ..], ..., ["score", ..]]`.
    pub score_terms: Vec<(String, f64)>,
    pub marks: Marks,
    pub targets: Vec<PlaceTarget>,
    pub inputs: PlaceInputsEcho,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{WorkerId, WorkerMode, WorkerSpec};
    use crate::kv_hint::KvWorkerMapEntry;
    use crate::scheduler::placement::RoleWeight;
    use crate::workers::SymmetricPdWorkerInfo;

    #[test]
    /// Check order-keyed replay advances per-session cursors and records outcome
    /// counters/errors.
    ///
    /// # Inputs
    /// No arguments; writes a temporary order-keyed replay file and exercises missing, known,
    /// exhausted and malformed cases.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn decision_replay_advances_per_session_and_names_outcomes() {
        let dir =
            std::env::temp_dir().join(format!("symmetric_pd-replay-rt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("table.json");
        // sha256("s1") keys the session; two recorded pairs.
        let key = {
            let mut h = Sha256::new();
            h.update(b"s1");
            format!("{:x}", h.finalize())
        };
        std::fs::write(
            &path,
            format!(
                r#"{{"key_hash":"sha256","sessions":{{"{key}":[["http://a:38000","http://b:39000"],["http://c:38000","http://a:39000"]]}}}}"#
            ),
        )
        .unwrap();
        let r = DecisionReplay::load(&path).unwrap();
        assert_eq!(r.next(None, 10), (None, "no_key"));
        assert_eq!(r.next(Some(""), 10), (None, "no_key"));
        assert_eq!(r.next(Some("other"), 10), (None, "unknown_key"));
        assert_eq!(
            r.next(Some("s1"), 10),
            (
                Some(("http://a:38000".into(), "http://b:39000".into())),
                "hit"
            )
        );
        assert_eq!(
            r.next(Some("s1"), 10),
            (
                Some(("http://c:38000".into(), "http://a:39000".into())),
                "hit"
            )
        );
        assert_eq!(r.next(Some("s1"), 10), (None, "exhausted"));
        assert_eq!(r.counters(), (2, 1, 1, 2));
        std::fs::write(&path, r#"{"key_hash":"md5","sessions":{}}"#).unwrap();
        assert!(DecisionReplay::load(&path)
            .unwrap_err()
            .contains("key_hash"));
        std::fs::write(&path, r#"{"key_hash":"sha256"}"#).unwrap();
        assert!(DecisionReplay::load(&path).unwrap_err().contains("neither"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    /// Check length-keyed replay uses exact/nearest lengths and retains the last same-content
    /// placement.
    ///
    /// # Inputs
    /// No arguments; writes a temporary length-keyed table with a 16-token tolerance.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn decision_replay_content_keyed_matches_prompt_len_with_tolerance() {
        let dir =
            std::env::temp_dir().join(format!("symmetric_pd-replay-len-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("table.json");
        let key = {
            let mut h = Sha256::new();
            h.update(b"s1");
            format!("{:x}", h.finalize())
        };
        std::fs::write(
            &path,
            format!(
                r#"{{"key_hash":"sha256","len_tolerance":16,"sessions_by_len":{{"{key}":{{"1000":[["http://a:38000","http://b:39000"],["http://c:38000","http://d:39000"]],"5000":[["http://e:38000","http://f:39000"]]}}}}}}"#
            ),
        )
        .unwrap();
        let r = DecisionReplay::load(&path).unwrap();
        let ab = Some(("http://a:38000".to_string(), "http://b:39000".to_string()));
        let cd = Some(("http://c:38000".to_string(), "http://d:39000".to_string()));
        let ef = Some(("http://e:38000".to_string(), "http://f:39000".to_string()));
        assert_eq!(r.next(Some("s1"), 1000), (ab.clone(), "hit_len"));
        // second replay cycle of the same event takes the second recorded pair
        assert_eq!(r.next(Some("s1"), 1000), (cd.clone(), "hit_len"));
        // exhausted lengths keep the last recorded pair (same content -> same placement)
        assert_eq!(r.next(Some("s1"), 1000), (cd, "hit_len"));
        // within tolerance -> nearest length
        assert_eq!(r.next(Some("s1"), 5010), (ef, "hit_len_near"));
        // outside tolerance and no order-keyed list -> unknown_len
        assert_eq!(r.next(Some("s1"), 3000), (None, "unknown_len"));
        assert_eq!(r.next(Some("zz"), 1000), (None, "unknown_key"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    const MODEL: &str = "m";

    /// Create the model identifier used by runtime tests.
    ///
    /// # Inputs
    /// None.
    ///
    /// # Outputs
    /// A `ModelId` wrapping the test constant `m`.
    fn model() -> ModelId {
        ModelId(MODEL.into())
    }

    /// Construct a worker registry for runtime tests.
    ///
    /// # Inputs
    /// `workers`: URL, engine mode, optional bootstrap port and optional symmetric metadata
    /// tuples.
    ///
    /// # Outputs
    /// A shared registry containing those workers for the test model; invalid test
    /// registrations panic through `unwrap`.
    fn registry_with(
        workers: Vec<(&str, WorkerMode, Option<u16>, Option<SymmetricPdWorkerInfo>)>,
    ) -> Arc<WorkerRegistry> {
        let reg = Arc::new(WorkerRegistry::default());
        for (url, mode, port, info) in workers {
            reg.add(WorkerSpec {
                id: WorkerId(url.to_string()),
                url: url.to_string(),
                mode,
                model_ids: vec![model()],
                bootstrap_port: port,
            })
            .unwrap();
            if let Some(info) = info {
                reg.get(&WorkerId(url.to_string()))
                    .unwrap()
                    .set_symmetric_pd(info);
            }
        }
        reg
    }

    /// Build optional symmetric-worker metadata for a role-weight fixture.
    ///
    /// # Inputs
    /// `p` and `d`: desired prefill/decode weights.
    ///
    /// # Outputs
    /// `Some(SymmetricPdWorkerInfo)` with these weights and other metadata at defaults.
    fn sym(p: f64, d: f64) -> Option<SymmetricPdWorkerInfo> {
        Some(SymmetricPdWorkerInfo {
            weight: RoleWeight {
                prefill: p,
                decode: d,
            },
            ..Default::default()
        })
    }

    /// Create deterministic indexer settings for runtime parsing tests.
    ///
    /// # Inputs
    /// None.
    ///
    /// # Outputs
    /// An indexer configuration mapping numeric IDs 1/2 to http://a/http://b with a 20 ms query
    /// timeout.
    fn kv_cfg() -> KvHintConfig {
        KvHintConfig {
            indexer_url: "http://indexer:9001".into(),
            min_tokens: 256,
            min_incremental_tokens: 0,
            local_prefix_fallback: false,
            indexer_timeout_ms: 20,
            worker_map: vec![
                KvWorkerMapEntry {
                    id: 1,
                    url: "http://a".into(),
                    servicer: String::new(),
                },
                KvWorkerMapEntry {
                    id: 2,
                    url: "http://b".into(),
                    servicer: String::new(),
                },
            ],
        }
    }

    /// Late D choose: every candidate D-leg import carries the return
    /// directive to the P of the request, with that candidate as the source;
    /// the P-leg export carries none. Off: no candidate carries it.
    #[tokio::test]
    async fn decode_host_return_rides_on_every_late_d_candidate_import() {
        let reg = registry_with(vec![
            (
                "http://p:30000",
                WorkerMode::Symmetric,
                Some(9000),
                sym(1.0, 0.0),
            ),
            (
                "http://d1:30000",
                WorkerMode::Symmetric,
                Some(9000),
                sym(0.0, 1.0),
            ),
            (
                "http://d2:30000",
                WorkerMode::Symmetric,
                Some(9000),
                sym(0.0, 1.0),
            ),
        ]);
        let cfg = SymmetricPdConfig {
            import_cache_policy: crate::config::SymmetricPdImportCachePolicy::Retain,
            decode_host_return: Some(crate::config::DecodeHostReturnConfig {
                servicers: [
                    ("http://p:30000".into(), "p:18100".into()),
                    ("http://d1:30000".into(), "d1:18100".into()),
                    ("http://d2:30000".into(), "d2:18100".into()),
                ]
                .into(),
                timeout_ms: 10_000,
                max_tokens: 65_536,
            }),
            ..Default::default()
        };
        let rt = SymmetricPdRuntime::spawn(cfg, None, Arc::clone(&reg), model());
        let p = reg.get(&WorkerId("http://p:30000".to_string())).unwrap();
        let candidates = rt.decode_workers();
        assert_eq!(candidates.len(), 2);
        let (export, imports) = rt.late_marks(77, &p, &candidates, 4096).unwrap();
        assert!(export.get("host_return").is_none());
        assert_eq!(imports.len(), 2);
        for (import, d) in imports.iter().zip(&candidates) {
            let source = if d.url == "http://d1:30000" {
                "d1:18100"
            } else {
                "d2:18100"
            };
            assert_eq!(import["cache_policy"], "retain");
            assert_eq!(
                import["host_return"],
                json!({"destination_servicer_addr":"p:18100",
                "source_servicer_addr": source, "timeout_ms":10000, "max_tokens":65536})
            );
        }

        let off = SymmetricPdConfig {
            import_cache_policy: crate::config::SymmetricPdImportCachePolicy::Retain,
            ..Default::default()
        };
        let rt_off = SymmetricPdRuntime::spawn(off, None, Arc::clone(&reg), model());
        let (_, imports) = rt_off
            .late_marks(78, &p, &rt_off.decode_workers(), 4096)
            .unwrap();
        assert!(imports.iter().all(|i| i.get("host_return").is_none()));
    }

    #[tokio::test]
    async fn decode_host_return_uses_selected_pair_and_preserves_request_identity() {
        let reg = registry_with(vec![
            (
                "http://p:30000",
                WorkerMode::Symmetric,
                Some(9000),
                sym(1.0, 0.0),
            ),
            (
                "http://d:30000",
                WorkerMode::Symmetric,
                Some(9000),
                sym(0.0, 1.0),
            ),
        ]);
        let cfg = SymmetricPdConfig {
            import_cache_policy: crate::config::SymmetricPdImportCachePolicy::Retain,
            kv_session_retention: true,
            decode_host_return: Some(crate::config::DecodeHostReturnConfig {
                servicers: [
                    ("http://p:30000".into(), "p:18100".into()),
                    ("http://d:30000".into(), "d:19100".into()),
                ]
                .into(),
                timeout_ms: 10_000,
                max_tokens: 128,
            }),
            ..Default::default()
        };
        let rt = SymmetricPdRuntime::spawn(cfg, None, reg, model());
        // A large prompt must not consume the newly generated suffix budget.
        let decision = rt
            .admit(PlaceInput {
                prompt_len: 131_072,
                rid: Some("stable-request".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        let import = decision.marks.kv_import.as_ref().unwrap();
        assert_eq!(
            import["host_return"],
            json!({"destination_servicer_addr":"p:18100",
            "source_servicer_addr":"d:19100", "timeout_ms":10000, "max_tokens":128})
        );
        assert_eq!(import["cache_policy"], "retain");
        let export = decision.marks.kv_export.as_ref().unwrap();
        assert_eq!(export["attempt_id"], import["attempt_id"]);
        assert_eq!(export["transfer_id"], import["transfer_id"]);
        assert!(export.get("host_return").is_none());
        let identity = crate::agent_identity::RequestIdentity {
            agent_id: "return-session".into(),
            subagent_id: 0,
            request_id: 1,
        };
        let tag = rt
            .kv_sessions()
            .unwrap()
            .register(&identity, &decision.p_worker.url, &decision.d_worker.url)
            .unwrap();
        let (p_body, d_body) = inject_marks(
            &json!({"rid":"stable-request", "cache_salt":"ns", "kv_session": tag}),
            &decision.marks,
        )
        .unwrap();
        assert_eq!(d_body["rid"], p_body["rid"]);
        assert_eq!(d_body["cache_salt"], "ns");
        assert_eq!(d_body["kv_import"]["host_return"], import["host_return"]);
        assert_eq!(p_body["kv_session"], d_body["kv_session"]);
        assert_eq!(d_body["kv_session"], serde_json::to_value(&tag).unwrap());
        assert_eq!(
            rt.kv_sessions()
                .unwrap()
                .register(&identity, &decision.p_worker.url, &decision.d_worker.url)
                .unwrap(),
            tag,
            "return directives do not create a second registration"
        );
        decision.ledger.retire();
    }

    #[tokio::test]
    async fn decode_host_return_missing_destination_does_not_reserve_compute() {
        let reg = registry_with(vec![
            (
                "http://p:30000",
                WorkerMode::Symmetric,
                Some(9000),
                sym(1.0, 0.0),
            ),
            (
                "http://d:30000",
                WorkerMode::Symmetric,
                Some(9000),
                sym(0.0, 1.0),
            ),
        ]);
        let cfg = SymmetricPdConfig {
            import_cache_policy: crate::config::SymmetricPdImportCachePolicy::Retain,
            decode_host_return: Some(crate::config::DecodeHostReturnConfig {
                servicers: [("http://d:30000".into(), "d:18100".into())].into(),
                timeout_ms: 10_000,
                max_tokens: 65_536,
            }),
            ..Default::default()
        };
        let rt = SymmetricPdRuntime::spawn(cfg, None, reg, model());
        let error = rt
            .admit(PlaceInput {
                prompt_len: 4096,
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("missing configured peer servicer"));
        assert_eq!(
            rt.loads(["http://p:30000"])
                .get("http://p:30000")
                .active_prefill_reqs,
            0
        );
    }

    #[test]
    fn decode_host_return_skips_local_and_explicit_cache_optouts_are_preserved() {
        let reg = registry_with(vec![
            (
                "http://p:30000",
                WorkerMode::Symmetric,
                Some(9000),
                sym(1.0, 0.0),
            ),
            (
                "http://d:30000",
                WorkerMode::Symmetric,
                Some(9000),
                sym(0.0, 1.0),
            ),
        ]);
        let p = reg.get(&WorkerId("http://p:30000".into())).unwrap();
        let d = reg.get(&WorkerId("http://d:30000".into())).unwrap();
        for (policy, wire) in [
            (
                crate::config::SymmetricPdImportCachePolicy::Auto,
                "gpu_only",
            ),
            (
                crate::config::SymmetricPdImportCachePolicy::GpuOnly,
                "gpu_only",
            ),
            (crate::config::SymmetricPdImportCachePolicy::Drop, "drop"),
        ] {
            let cfg = SymmetricPdConfig {
                import_cache_policy: policy,
                ..Default::default()
            };
            let rt = SymmetricPdRuntime::spawn(cfg, None, Arc::clone(&reg), model());
            let placement = Placement {
                p_leg: p.url.clone(),
                d_leg: d.url.clone(),
                transfer_id: Some(7),
                score_terms: vec![],
            };
            let marks = rt.marks(&placement, &p, &d, 5).unwrap();
            assert_eq!(marks.kv_import.unwrap()["cache_policy"], wire);
        }
        let cfg = SymmetricPdConfig {
            decode_host_return: Some(crate::config::DecodeHostReturnConfig {
                servicers: Default::default(),
                timeout_ms: 10_000,
                max_tokens: 65_536,
            }),
            ..Default::default()
        };
        let rt = SymmetricPdRuntime::spawn(cfg, None, reg, model());
        let local = Placement {
            p_leg: p.url.clone(),
            d_leg: p.url.clone(),
            transfer_id: None,
            score_terms: vec![],
        };
        assert_eq!(rt.marks(&local, &p, &p, 5).unwrap(), Marks::default());
    }

    #[tokio::test]
    /// Check candidate capabilities and role weights reflect each engine mode without losing
    /// symmetric metadata.
    ///
    /// # Inputs
    /// No arguments; registers one legacy P, one legacy D and one symmetric worker.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    async fn candidates_map_legacy_and_symmetric_workers() {
        let reg = registry_with(vec![
            ("http://p:1", WorkerMode::Prefill, Some(9000), None),
            ("http://d:1", WorkerMode::Decode, None, None),
            (
                "http://s:1",
                WorkerMode::Symmetric,
                Some(9100),
                sym(0.7, 1.0),
            ),
        ]);
        let rt = SymmetricPdRuntime::spawn(SymmetricPdConfig::default(), None, reg, model());
        let (weights, index) = rt.candidates();
        assert_eq!(index.len(), 3);
        let p = weights.workers["http://p:1"];
        assert_eq!(p.capabilities, Capabilities::PREFILL_ONLY);
        assert_eq!(p.weight, RoleWeight::PREFILL_LEANING);
        let d = weights.workers["http://d:1"];
        assert_eq!(d.capabilities, Capabilities::DECODE_ONLY);
        assert_eq!(d.weight, RoleWeight::DECODE_LEANING);
        let s = weights.workers["http://s:1"];
        assert_eq!(s.capabilities, Capabilities::BOTH);
        assert_eq!(s.weight.prefill, 0.7);
        assert!(rt.has_capable_pair());
    }

    #[tokio::test]
    /// Check readiness excludes plain workers and requires both P and D capabilities.
    ///
    /// # Inputs
    /// No arguments; constructs a plain-only registry and then a prefill-only registry.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    async fn plain_workers_and_prefill_only_fleets_are_not_a_capable_pair() {
        let reg = registry_with(vec![("http://w:1", WorkerMode::Plain, None, None)]);
        let rt = SymmetricPdRuntime::spawn(SymmetricPdConfig::default(), None, reg, model());
        assert!(rt.candidates().1.is_empty());
        assert!(!rt.has_capable_pair());

        let reg = registry_with(vec![("http://p:1", WorkerMode::Prefill, Some(1), None)]);
        let rt = SymmetricPdRuntime::spawn(SymmetricPdConfig::default(), None, reg, model());
        assert!(!rt.has_capable_pair());
    }

    #[tokio::test]
    /// Check sGLang endpoint resolution prefers advertised addresses and fails when no fallback
    /// exists.
    ///
    /// # Inputs
    /// No arguments; compares explicit owner/reader endpoints, bootstrap-derived endpoints and
    /// missing endpoint metadata.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    async fn endpoints_prefer_server_info_then_bootstrap_convention() {
        let advertised = Some(SymmetricPdWorkerInfo {
            owner_addr: Some("tcp://10.0.0.5:7000".into()),
            reader_addr: Some("tcp://10.0.0.5:7001".into()),
            ..Default::default()
        });
        let reg = registry_with(vec![
            (
                "http://10.0.0.5:30000",
                WorkerMode::Symmetric,
                Some(8998),
                advertised,
            ),
            (
                "http://10.0.0.6:30000",
                WorkerMode::Symmetric,
                Some(8998),
                None,
            ),
            ("http://10.0.0.7:30000", WorkerMode::Symmetric, None, None),
        ]);
        let rt = SymmetricPdRuntime::spawn(
            SymmetricPdConfig::default(),
            None,
            Arc::clone(&reg),
            model(),
        );
        let by_url = |u: &str| reg.get(&WorkerId(u.into())).unwrap();
        assert_eq!(
            rt.endpoints(&by_url("http://10.0.0.5:30000")).unwrap(),
            (
                "tcp://10.0.0.5:7000".to_string(),
                "tcp://10.0.0.5:7001".to_string()
            )
        );
        assert_eq!(
            rt.endpoints(&by_url("http://10.0.0.6:30000")).unwrap(),
            (
                "tcp://10.0.0.6:8998".to_string(),
                "tcp://10.0.0.6:8999".to_string()
            )
        );
        assert!(matches!(
            rt.endpoints(&by_url("http://10.0.0.7:30000")),
            Err(SymmetricPdError::Endpoints { .. })
        ));
    }

    #[tokio::test]
    /// Check split marks carry matching unique attempt IDs, correct endpoints and a retireable
    /// P reservation.
    ///
    /// # Inputs
    /// No arguments; admits a 4,096-token Split request and repeatedly rebuilds its marks.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    async fn admit_split_builds_marks_and_accounts_ledger() {
        let reg = registry_with(vec![
            (
                "http://10.0.0.1:30000",
                WorkerMode::Symmetric,
                Some(9000),
                sym(1.0, 0.0),
            ),
            (
                "http://10.0.0.2:30000",
                WorkerMode::Symmetric,
                Some(9000),
                sym(0.0, 1.0),
            ),
        ]);
        let mut cfg = SymmetricPdConfig::default();
        cfg.placement.force_split = true;
        let rt = SymmetricPdRuntime::spawn(cfg, None, reg, model());
        let input = PlaceInput {
            prompt_len: 4096,
            rid: Some("r1".into()),
            ..Default::default()
        };
        let d = rt.admit(input).await.unwrap();
        assert_eq!(d.p_worker.url, "http://10.0.0.1:30000");
        assert_eq!(d.d_worker.url, "http://10.0.0.2:30000");
        assert_eq!(d.branch(), "split");
        let tid = d.placement.transfer_id.unwrap();
        let export = d.marks.kv_export.clone().unwrap();
        let import = d.marks.kv_import.clone().unwrap();
        let attempt = export["attempt_id"].as_u64().unwrap();
        assert_ne!(attempt, 0);
        assert_eq!(import["attempt_id"].as_u64(), Some(attempt));
        assert_eq!(export["transfer_id"].as_u64(), Some(tid));
        assert_eq!(export["reader_worker"], "http://10.0.0.2:30000");
        assert_eq!(export["reader_addr"], "tcp://10.0.0.2:9001");
        assert_eq!(import["transfer_id"].as_u64(), Some(tid));
        assert_eq!(import["owner_worker"], "http://10.0.0.1:30000");
        // Without host return, dedicated D preserves the original Auto behavior.
        assert_eq!(import["cache_policy"], "gpu_only");
        assert!(import.get("host_return").is_none());
        assert_eq!(import["owner_addr"], "tcp://10.0.0.1:9000");
        assert_eq!(import["expected_tokens"].as_u64(), Some(4096));
        assert!(!d.forced);

        // A new transfer attempt is distinct even when retrying the exact same
        // placement/transfer ID; rebuilding must never retire the retry as stale.
        let mut attempts = std::collections::HashSet::from([attempt]);
        for _ in 0..128 {
            let retry = rt
                .marks(&d.placement, &d.p_worker, &d.d_worker, 4096)
                .unwrap();
            let p = retry.kv_export.unwrap();
            let dd = retry.kv_import.unwrap();
            let id = p["attempt_id"].as_u64().unwrap();
            assert_ne!(id, 0);
            assert!(attempts.insert(id));
            assert_eq!(dd["attempt_id"].as_u64(), Some(id));
            assert_eq!(p["transfer_id"].as_u64(), Some(tid));
            assert_eq!(dd["transfer_id"].as_u64(), Some(tid));
        }

        // Ledger: the P-leg now carries 4096 uncached tokens until retired.
        let loads = rt.loads(["http://10.0.0.1:30000"]);
        assert_eq!(
            loads.get("http://10.0.0.1:30000").backlog_uncached_tokens,
            4096
        );
        d.ledger.retire();
        let loads = rt.loads(["http://10.0.0.1:30000"]);
        assert_eq!(
            loads.get("http://10.0.0.1:30000").backlog_uncached_tokens,
            0
        );

        // Marks land on the right leg bodies, never both, bootstrap fields removed.
        let body = json!({"text": "hi", "bootstrap_room": 7, "stream": true});
        let (p_body, d_body) = inject_marks(&body, &d.marks).unwrap();
        assert!(p_body.get(KV_EXPORT_KEY).is_some() && p_body.get(KV_IMPORT_KEY).is_none());
        assert!(d_body.get(KV_IMPORT_KEY).is_some() && d_body.get(KV_EXPORT_KEY).is_none());
        assert!(p_body.get("bootstrap_room").is_none() && d_body.get("bootstrap_room").is_none());
        assert_eq!(p_body[KV_EXPORT_KEY], export);
        assert_eq!(d_body[KV_IMPORT_KEY], import);
        assert_eq!(p_body["stream"], true);
    }

    #[tokio::test]
    /// Check local admission has no transfer ID or KV marks and leaves body injection
    /// unchanged.
    ///
    /// # Inputs
    /// No arguments; admits a request on one fully symmetric worker.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    async fn admit_local_has_no_marks_and_no_transfer_id() {
        let reg = registry_with(vec![(
            "http://10.0.0.1:30000",
            WorkerMode::Symmetric,
            Some(9000),
            sym(1.0, 1.0),
        )]);
        let rt = SymmetricPdRuntime::spawn(SymmetricPdConfig::default(), None, reg, model());
        let d = rt
            .admit(PlaceInput {
                prompt_len: 100,
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(d.is_local());
        assert_eq!(d.placement.transfer_id, None);
        assert_eq!(d.marks, Marks::default());
        assert_eq!(serde_json::to_value(&d.marks).unwrap(), json!({}));
        let body = json!({"text": "hi"});
        let (p, dd) = inject_marks(&body, &d.marks).unwrap();
        assert_eq!(p, body);
        assert_eq!(dd, body);
    }

    /// Legacy Prefill/Decode workers pair up as {1,0}/{0,1}; a legacy
    /// Decode worker advertises no side channel, so the Split fails on the
    /// reader endpoint rather than silently sending a mark without one.
    #[tokio::test]
    /// Check split admission fails with an endpoint error instead of creating unusable transfer
    /// marks.
    ///
    /// # Inputs
    /// No arguments; pairs a legacy P with a D lacking the required side channels.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    async fn split_without_decode_endpoints_is_an_endpoints_error() {
        let reg = registry_with(vec![
            ("http://p:30000", WorkerMode::Prefill, Some(8998), None),
            ("http://d:30000", WorkerMode::Decode, None, None),
        ]);
        let rt = SymmetricPdRuntime::spawn(SymmetricPdConfig::default(), None, reg, model());
        let err = rt
            .admit(PlaceInput {
                prompt_len: 64,
                ..Default::default()
            })
            .await
            .err();
        assert!(
            matches!(err, Some(SymmetricPdError::Endpoints { .. })),
            "{err:?}"
        );
    }

    #[tokio::test]
    /// Check legacy admission waits near its configured cap then reports forced placement.
    ///
    /// # Inputs
    /// No arguments; saturates the sole P candidate under a 250 ms HOLD limit.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    async fn hold_queue_caps_wait_and_force_places() {
        let reg = registry_with(vec![(
            "http://10.0.0.1:30000",
            WorkerMode::Symmetric,
            Some(9000),
            sym(1.0, 1.0),
        )]);
        let cfg = SymmetricPdConfig {
            max_queue_wait_ms: 250,
            max_uncached_tokens: 1000,
            ..Default::default()
        };
        let rt = SymmetricPdRuntime::spawn(cfg, None, reg, model());
        // Saturate the only P-leg candidate.
        let _h = rt.account("http://10.0.0.1:30000", 5000);
        let start = Instant::now();
        let d = rt
            .admit(PlaceInput {
                prompt_len: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        let waited = start.elapsed();
        assert!(d.forced, "no capacity ever appeared -> forced");
        assert!(waited >= Duration::from_millis(200), "waited {waited:?}");
        assert!(
            waited < Duration::from_millis(1500),
            "cap respected: {waited:?}"
        );
        assert!(d.queue_wait_ms >= 200);
    }

    #[tokio::test]
    /// Check retiring work wakes HOLD admission before its maximum wait without forcing.
    ///
    /// # Inputs
    /// No arguments; retires a saturating reservation after a short delay while admission
    /// waits.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    async fn hold_queue_wakes_on_ledger_retire() {
        let reg = registry_with(vec![(
            "http://10.0.0.1:30000",
            WorkerMode::Symmetric,
            Some(9000),
            sym(1.0, 1.0),
        )]);
        let cfg = SymmetricPdConfig {
            max_queue_wait_ms: 5000,
            max_uncached_tokens: 1000,
            ..Default::default()
        };
        let rt = SymmetricPdRuntime::spawn(cfg, None, reg, model());
        let h = rt.account("http://10.0.0.1:30000", 5000);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(150)).await;
            h.retire();
        });
        let start = Instant::now();
        let d = rt
            .admit(PlaceInput {
                prompt_len: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(!d.forced);
        let waited = start.elapsed();
        assert!(
            waited >= Duration::from_millis(100) && waited < Duration::from_millis(2000),
            "{waited:?}"
        );
    }

    #[tokio::test]
    /// Check aggregate reconciliation replaces older accounted work while retaining newer work
    /// and decode counts.
    ///
    /// # Inputs
    /// No arguments; injects an aggregate poll between old dispatched and new reservations.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    async fn poll_subsumes_older_ledger_entries_and_feeds_decode_load() {
        let reg = registry_with(vec![(
            "http://10.0.0.1:30000",
            WorkerMode::Symmetric,
            Some(9000),
            sym(1.0, 1.0),
        )]);
        let rt = SymmetricPdRuntime::spawn(SymmetricPdConfig::default(), None, reg, model());
        let old = rt.account("http://10.0.0.1:30000", 700);
        old.mark_dispatched();
        let sent_at = Instant::now();
        let _new = rt.account("http://10.0.0.1:30000", 50);
        rt.apply_poll("http://10.0.0.1:30000", 300, 7, 2, sent_at);
        let load = rt
            .loads(["http://10.0.0.1:30000"])
            .get("http://10.0.0.1:30000");
        assert_eq!(
            load.backlog_uncached_tokens, 350,
            "baseline + post-poll entry"
        );
        assert_eq!(load.running_decode, 7);
        assert_eq!(load.waiting, 2);
    }

    #[tokio::test]
    /// Check indexer parsing returns canonical nonzero worker hits and skips unusable
    /// instances.
    ///
    /// # Inputs
    /// No arguments; parses GPU/CPU hits for mapped, zero-hit and unknown instances.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    async fn indexer_reply_parsing_maps_instances_to_workers() {
        let reg = registry_with(vec![
            ("http://a", WorkerMode::Symmetric, Some(1), sym(1.0, 1.0)),
            ("http://b", WorkerMode::Symmetric, Some(1), sym(1.0, 1.0)),
        ]);
        let rt =
            SymmetricPdRuntime::spawn(SymmetricPdConfig::default(), Some(&kv_cfg()), reg, model());
        let hits = rt.parse_hits(&json!({
            "instances": {"1": {"gpu": 1024, "cpu": 4096}, "2": {"cpu": 0}, "99": {"gpu": 7}}
        }));
        assert_eq!(
            hits.get("http://a"),
            HitTokens {
                gpu: 1024,
                host: 4096
            }
        );
        assert_eq!(hits.get("http://b"), HitTokens::default());
        assert_eq!(hits.workers.len(), 1);
    }

    #[test]
    /// Check prompt length estimation follows its token-first fallback order.
    ///
    /// # Inputs
    /// No arguments; passes explicit tokens, body token fields, text, messages and an empty
    /// body.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn prompt_len_estimate_prefers_tokens_then_ids_then_text() {
        assert_eq!(
            estimate_prompt_len(&json!({"text": "abcdefgh"}), Some(&[1, 2, 3])),
            3
        );
        assert_eq!(
            estimate_prompt_len(&json!({"input_ids": [1, 2, 3, 4]}), None),
            4
        );
        assert_eq!(estimate_prompt_len(&json!({"prompt": [5, 6]}), None), 2);
        assert_eq!(estimate_prompt_len(&json!({"text": "abcdefgh"}), None), 2);
        assert_eq!(
            estimate_prompt_len(&json!({"messages": [{"content": "abcdefgh"}]}), None),
            2
        );
        assert_eq!(estimate_prompt_len(&json!({}), None), 1);
    }

    #[test]
    /// Check load parsing sums supported rank fields while retaining missing observations as
    /// unknown.
    ///
    /// # Inputs
    /// No arguments; parses mixed top-level/core rank observations and incomplete payloads.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn parse_loads_sums_ranks_and_reads_core_block() {
        let v = json!({"loads": [
            {"num_waiting_uncached_tokens": 100, "num_running_reqs": 3, "num_waiting_reqs": 1},
            {"core": {"num_waiting_uncached_tokens": 50, "num_running_reqs": 2}},
        ]});
        assert_eq!(
            parse_loads(&v).unwrap(),
            crate::backend::state::LoadObservation {
                uncached_tokens: Some(150),
                running: Some(5),
                waiting: None
            }
        );
        assert!(parse_loads(&json!({"loads": [{"num_running_reqs": 1}]})).is_err());
        assert_eq!(
            parse_loads(&json!({"num_waiting_uncached_tokens": 9})).unwrap(),
            crate::backend::state::LoadObservation {
                uncached_tokens: Some(9),
                running: None,
                waiting: None
            }
        );
    }

    #[test]
    /// Check p/D body construction isolates export/import marks and adds peer-pull hints only
    /// to the executing P body.
    ///
    /// # Inputs
    /// No arguments; builds Local/Split bodies with token overrides, bootstrap fields and an
    /// optional peer-pull hint.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn build_leg_bodies_injects_marks_and_input_ids() {
        let body = Bytes::from_static(br#"{"model":"m","prompt":[1,2],"bootstrap_room":3}"#);
        let value: Value = serde_json::from_slice(&body).unwrap();
        let marks = Marks {
            kv_export: Some(json!({"transfer_id": 1})),
            kv_import: Some(json!({"transfer_id": 1})),
        };
        let (p, d) =
            build_leg_bodies(&body, Some(value.clone()), Some(&[7, 8]), &marks, None).unwrap();
        let p: Value = serde_json::from_slice(&p).unwrap();
        let d: Value = serde_json::from_slice(&d).unwrap();
        assert_eq!(p["input_ids"], json!([7, 8]));
        assert!(p.get(KV_EXPORT_KEY).is_some() && p.get(KV_IMPORT_KEY).is_none());
        assert!(d.get(KV_IMPORT_KEY).is_some() && d.get(KV_EXPORT_KEY).is_none());
        assert!(p.get("bootstrap_room").is_none() && d.get("bootstrap_room").is_none());
        assert!(p.get(KV_PULL_HINT_KEY).is_none() && d.get(KV_PULL_HINT_KEY).is_none());

        // Local, nothing to inject: original bytes for both legs.
        let (p, d) =
            build_leg_bodies(&body, Some(value.clone()), None, &Marks::default(), None).unwrap();
        assert_eq!(p, body);
        assert_eq!(d, body);

        // Pull hint: Split -> P-leg only; Local -> the single executing body.
        let hint = json!({"source_worker_id": "http://o:38000", "source_servicer_addr": "o:28000", "expected_hit_tokens": 4096});
        let (p, d) =
            build_leg_bodies(&body, Some(value.clone()), None, &marks, Some(&hint)).unwrap();
        let p: Value = serde_json::from_slice(&p).unwrap();
        let d: Value = serde_json::from_slice(&d).unwrap();
        assert_eq!(p[KV_PULL_HINT_KEY], hint);
        assert!(d.get(KV_PULL_HINT_KEY).is_none() && p.get(KV_EXPORT_KEY).is_some());
        let (p, _) =
            build_leg_bodies(&body, Some(value), None, &Marks::default(), Some(&hint)).unwrap();
        let p: Value = serde_json::from_slice(&p).unwrap();
        assert_eq!(p[KV_PULL_HINT_KEY], hint);
    }

    #[tokio::test]
    /// Check hTTP placement returns matching engine targets/marks and accounts the P token
    /// lease.
    ///
    /// # Inputs
    /// No arguments; requests legacy HTTP placement for three input tokens across fixed logical
    /// roles.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    async fn place_http_returns_targets_ready_to_embed() {
        let reg = registry_with(vec![
            (
                "http://10.0.0.1:30000",
                WorkerMode::Symmetric,
                Some(9000),
                sym(1.0, 0.0),
            ),
            (
                "http://10.0.0.2:30000",
                WorkerMode::Symmetric,
                Some(9000),
                sym(0.0, 1.0),
            ),
        ]);
        let rt = SymmetricPdRuntime::spawn(SymmetricPdConfig::default(), None, reg, model());
        let resp = rt
            .place_http(PlaceHttpRequest {
                input_ids: Some(vec![1, 2, 3]),
                session_id: Some("s".into()),
                turn: Some(2),
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(!resp.local);
        assert_eq!(resp.targets.len(), 2);
        assert_eq!(resp.targets[0].leg, "p");
        assert_eq!(resp.targets[0].endpoint, "http://10.0.0.1:30000/generate");
        assert!(resp.targets[0].mark.get(KV_EXPORT_KEY).is_some());
        assert_eq!(resp.targets[1].leg, "d");
        assert!(resp.targets[1].mark.get(KV_IMPORT_KEY).is_some());
        assert_eq!(
            resp.transfer_id,
            resp.marks.kv_export.as_ref().unwrap()["transfer_id"].as_u64()
        );
        assert_eq!(resp.inputs.prompt_len, 3);
        assert_eq!(resp.score_terms.len(), 10);
        assert_eq!(resp.score_terms.last().unwrap().0, "score");
        // The P-leg's uncached tokens were accounted.
        assert_eq!(
            rt.loads(["http://10.0.0.1:30000"])
                .get("http://10.0.0.1:30000")
                .backlog_uncached_tokens,
            3
        );
    }

    #[tokio::test]
    /// Check per-call HTTP placement overrides choose Split and preserve the requested endpoint
    /// path.
    ///
    /// # Inputs
    /// No arguments; requests Local placement without accounting then overrides force-split and
    /// the engine route.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    async fn place_http_local_and_force_split_override() {
        let reg = registry_with(vec![
            (
                "http://10.0.0.1:30000",
                WorkerMode::Symmetric,
                Some(9000),
                sym(1.0, 1.0),
            ),
            (
                "http://10.0.0.2:30000",
                WorkerMode::Symmetric,
                Some(9000),
                sym(1.0, 1.0),
            ),
        ]);
        let rt = SymmetricPdRuntime::spawn(SymmetricPdConfig::default(), None, reg, model());
        let resp = rt
            .place_http(PlaceHttpRequest {
                prompt_len: Some(10),
                account: Some(false),
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(resp.local);
        assert_eq!(resp.transfer_id, None);
        assert_eq!(resp.targets.len(), 1);
        assert_eq!(resp.targets[0].leg, "local");
        assert_eq!(resp.marks, Marks::default());
        let resp = rt
            .place_http(PlaceHttpRequest {
                prompt_len: Some(10),
                force_split: Some(true),
                route: Some("/v1/completions".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(!resp.local);
        assert!(resp.targets[0].endpoint.ends_with("/v1/completions"));
    }
    #[test]
    /// Check atomic select/reserve balances simultaneous arrivals and D round robin without
    /// using HOLD.
    ///
    /// # Inputs
    /// No arguments; admits 256 requests from eight synchronized threads across two P and three
    /// D workers.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn concurrent_co_schedule_arrivals_reserve_before_next_selection() {
        let reg = registry_with(vec![
            (
                "http://p0",
                WorkerMode::Symmetric,
                Some(9000),
                sym(1.0, 0.0),
            ),
            (
                "http://p1",
                WorkerMode::Symmetric,
                Some(9000),
                sym(1.0, 0.0),
            ),
            (
                "http://d0",
                WorkerMode::Symmetric,
                Some(9000),
                sym(0.0, 1.0),
            ),
            (
                "http://d1",
                WorkerMode::Symmetric,
                Some(9000),
                sym(0.0, 1.0),
            ),
            (
                "http://d2",
                WorkerMode::Symmetric,
                Some(9000),
                sym(0.0, 1.0),
            ),
        ]);
        let mut cfg = SymmetricPdConfig::default();
        cfg.placement.p_select = PSelect::CoSchedule;
        cfg.placement.d_select = DSelect::RoundRobin;
        cfg.placement.force_split = true;
        cfg.max_uncached_tokens = 1; // new policy must never use the old HOLD gate
        let rt = SymmetricPdRuntime::spawn(cfg, None, reg, model());
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let rt = Arc::clone(&rt);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .unwrap();
                    barrier.wait();
                    runtime.block_on(async {
                        let mut decisions = Vec::new();
                        for _ in 0..32 {
                            decisions.push(
                                rt.admit(PlaceInput {
                                    prompt_len: 1024,
                                    ..Default::default()
                                })
                                .await
                                .unwrap(),
                            );
                        }
                        decisions
                    })
                })
            })
            .collect();
        let decisions: Vec<_> = threads
            .into_iter()
            .flat_map(|t| t.join().unwrap())
            .collect();
        let loads = rt.loads(["http://p0", "http://p1"]);
        assert_eq!(loads.get("http://p0").active_prefill_reqs, 128);
        assert_eq!(loads.get("http://p1").active_prefill_reqs, 128);
        assert!(decisions.iter().all(|d| !d.forced));
        let counts: Vec<_> = ["http://d0", "http://d1", "http://d2"]
            .iter()
            .map(|u| decisions.iter().filter(|d| d.d_worker.url == *u).count())
            .collect();
        assert!(counts.iter().max().unwrap() - counts.iter().min().unwrap() <= 1);
        for decision in decisions {
            decision.ledger.retire();
        }
        assert_eq!(
            rt.loads(["http://p0", "http://p1"])
                .get("http://p0")
                .active_prefill_reqs,
            0
        );
    }
    #[derive(Debug)]
    struct LeastReservedPolicy;
    impl PlacementPolicy for LeastReservedPolicy {
        /// Select the least-reserved P for shared-ledger extension tests.
        ///
        /// # Inputs
        /// `req`: runtime seed; `loads`: active P reservations; `roles`: candidate role
        /// weights; cache hits and config are deliberately unused; `self` is stateless.
        ///
        /// # Outputs
        /// `Ok(Placement)` with least active P (reverse-lexical tie), fixed d0, and an
        /// intentionally forged hit log field. Panics if the fixture has no P; the runtime must
        /// ignore score terms for accounting.
        fn place(
            &self,
            req: &PlacementRequest,
            _: &IndexerHits,
            loads: &WorkerLoads,
            roles: &RoleWeights,
            _: &placement::PlacementConfig,
        ) -> Result<Placement, PlacementError> {
            let p = roles
                .workers
                .iter()
                .filter(|(_, r)| r.weight.prefill > 0.0)
                .min_by_key(|(id, _)| (loads.get(id).active_prefill_reqs, std::cmp::Reverse(*id)))
                .unwrap()
                .0;
            Ok(Placement {
                p_leg: p.clone(),
                d_leg: "http://d0".into(),
                transfer_id: Some(req.transfer_seed),
                // Malicious/incorrect logging must NOT reduce authoritative work.
                score_terms: vec![(TERM_HIT_TOKENS.into(), u64::MAX as f64)],
            })
        }
    }
    /// Build the symmetric-engine fleet used by extension tests.
    ///
    /// # Inputs
    /// None.
    ///
    /// # Outputs
    /// A shared test registry with two logical P workers and one logical D, each with bootstrap
    /// metadata.
    fn extension_registry() -> Arc<WorkerRegistry> {
        registry_with(vec![
            (
                "http://p0",
                WorkerMode::Symmetric,
                Some(9000),
                sym(1.0, 0.0),
            ),
            (
                "http://p1",
                WorkerMode::Symmetric,
                Some(9000),
                sym(1.0, 0.0),
            ),
            (
                "http://d0",
                WorkerMode::Symmetric,
                Some(9000),
                sym(0.0, 1.0),
            ),
        ])
    }
    /// Configure extension tests for co-schedule P and round-robin D.
    ///
    /// # Inputs
    /// None.
    ///
    /// # Outputs
    /// A default-derived force-split `SymmetricPdConfig` with CoSchedule and D RoundRobin selected.
    fn extension_config() -> SymmetricPdConfig {
        let mut cfg = SymmetricPdConfig::default();
        cfg.placement.p_select = PSelect::CoSchedule;
        cfg.placement.d_select = DSelect::RoundRobin;
        cfg.placement.force_split = true;
        cfg
    }
    #[test]
    /// Check both runtime entrypoints use the injected policy and reserve true raw work despite
    /// forged score fields.
    ///
    /// # Inputs
    /// No arguments; injects a custom policy, admits concurrent requests, then invokes legacy
    /// HTTP placement.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn injected_policy_uses_atomic_shared_ledger_on_both_runtime_entrypoints() {
        // Construct outside Tokio: no background network poller is started.
        let rt = SymmetricPdRuntime::spawn_with_policy(
            extension_config(),
            None,
            extension_registry(),
            model(),
            Arc::new(LeastReservedPolicy),
        );
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let rt = Arc::clone(&rt);
                std::thread::spawn(move || {
                    tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .unwrap()
                        .block_on(async {
                            let mut decisions = Vec::new();
                            for _ in 0..16 {
                                decisions.push(
                                    rt.admit(PlaceInput {
                                        prompt_len: 1024,
                                        ..Default::default()
                                    })
                                    .await
                                    .unwrap(),
                                );
                            }
                            decisions
                        })
                })
            })
            .collect();
        let decisions: Vec<_> = threads
            .into_iter()
            .flat_map(|t| t.join().unwrap())
            .collect();
        for p in ["http://p0", "http://p1"] {
            let load = rt.loads([p]).get(p);
            assert_eq!(load.active_prefill_reqs, 32);
            assert_eq!(
                load.backlog_uncached_tokens,
                32 * 1024,
                "policy score terms cannot forge reservation hit"
            );
        }
        for decision in decisions {
            decision.ledger.retire();
        }
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let reply = runtime
            .block_on(rt.place_http(PlaceHttpRequest {
                prompt_len: Some(1024),
                ..Default::default()
            }))
            .unwrap();
        assert_eq!(
            reply.p_leg.worker, "http://p1",
            "custom tie rule reached place_http too"
        );
        assert_eq!(
            rt.loads(["http://p1"])
                .get("http://p1")
                .backlog_uncached_tokens,
            1024
        );
    }
    #[derive(Debug)]
    struct InvalidPolicy;
    impl PlacementPolicy for InvalidPolicy {
        /// Return a deliberately invalid destination for policy-validation tests.
        ///
        /// # Inputs
        /// `req`: transfer seed; all observations/configuration and `self` are unused.
        ///
        /// # Outputs
        /// `Ok(Placement)` naming absent P and fixed d0. The runtime must reject this otherwise
        /// well-shaped result before reserving work.
        fn place(
            &self,
            req: &PlacementRequest,
            _: &IndexerHits,
            _: &WorkerLoads,
            _: &RoleWeights,
            _: &placement::PlacementConfig,
        ) -> Result<Placement, PlacementError> {
            Ok(Placement {
                p_leg: "absent".into(),
                d_leg: "http://d0".into(),
                transfer_id: Some(req.transfer_seed),
                score_terms: vec![],
            })
        }
    }
    #[test]
    /// Check invalid custom placements produce a typed error and leave both P reservation
    /// counts zero.
    ///
    /// # Inputs
    /// No arguments; injects a policy returning an absent P and attempts admission.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn invalid_injected_policy_returns_error_without_reservation_or_panic() {
        let rt = SymmetricPdRuntime::spawn_with_policy(
            extension_config(),
            None,
            extension_registry(),
            model(),
            Arc::new(InvalidPolicy),
        );
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = runtime.block_on(rt.admit(PlaceInput {
            prompt_len: 1024,
            ..Default::default()
        }));
        assert!(matches!(
            result,
            Err(SymmetricPdError::Placement(
                PlacementError::InvalidPolicyOutput(_)
            ))
        ));
        assert_eq!(
            rt.loads(["http://p0"]).get("http://p0").active_prefill_reqs,
            0
        );
        assert_eq!(
            rt.loads(["http://p1"]).get("http://p1").active_prefill_reqs,
            0
        );
    }

    #[derive(Debug)]
    struct PreferIdleP1Cost;
    impl super::super::policy::CostModel for PreferIdleP1Cost {
        /// Add a p0-specific overhead in the runtime cost-extension fixture.
        ///
        /// # Inputs
        /// `i`: candidate URL, outstanding backlog, prompt length and expected hit; `self` is
        /// stateless.
        ///
        /// # Outputs
        /// Before/after costs with a 1,000-unit overhead on p0; raw Backend reservations must
        /// still use uncached tokens only.
        fn prefill_cost(
            &self,
            i: super::super::policy::PrefillCostInput<'_>,
        ) -> super::super::policy::PrefillCost {
            // Fixed 1000-unit overhead on p0; every request still reserves raw
            // uncached tokens. A second arrival must see the first reservation.
            let before = u128::from(i.load.backlog_uncached_tokens)
                + if i.worker == "http://p0" { 1000 } else { 0 };
            super::super::policy::PrefillCost {
                before,
                after: before + u128::from(i.request.prompt_len.saturating_sub(i.expected_hit)),
            }
        }
        /// Keep the runtime test model's detour margin in token-scale units.
        ///
        /// # Inputs
        /// `tokens`: configured token margin; `self` is stateless.
        ///
        /// # Outputs
        /// The same value widened to `u128`.
        fn detour_margin(&self, tokens: u64) -> u128 {
            u128::from(tokens)
        }
    }
    #[test]
    /// Check custom costs affect P choice while each arrival observes prior reservations and
    /// reserves only raw tokens.
    ///
    /// # Inputs
    /// No arguments; admits two 1,024-token requests through a runtime using the p0-overhead
    /// cost model.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn injected_cost_uses_actual_runtime_and_keeps_raw_token_reservations() {
        let rt = SymmetricPdRuntime::spawn_with_policy(
            extension_config(),
            None,
            extension_registry(),
            model(),
            Arc::new(DefaultPlacementPolicy {
                cost: PreferIdleP1Cost,
            }),
        );
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let first = runtime
            .block_on(rt.admit(PlaceInput {
                prompt_len: 1024,
                ..Default::default()
            }))
            .unwrap();
        assert_eq!(first.p_worker.url, "http://p1");
        let second = runtime
            .block_on(rt.admit(PlaceInput {
                prompt_len: 1024,
                ..Default::default()
            }))
            .unwrap();
        assert_eq!(
            second.p_worker.url, "http://p0",
            "second choice sees first reservation through the shared ledger"
        );
        for p in ["http://p0", "http://p1"] {
            assert_eq!(rt.loads([p]).get(p).backlog_uncached_tokens, 1024);
        }
        first.ledger.retire();
        second.ledger.retire();
    }

    /// A 3P + 2D symmetric fleet (pinned roles, bootstrap ports set).
    fn ca_registry() -> Arc<WorkerRegistry> {
        registry_with(vec![
            (
                "http://10.0.0.1:30000",
                WorkerMode::Symmetric,
                Some(9000),
                sym(1.0, 0.0),
            ),
            (
                "http://10.0.0.2:30000",
                WorkerMode::Symmetric,
                Some(9000),
                sym(1.0, 0.0),
            ),
            (
                "http://10.0.0.3:30000",
                WorkerMode::Symmetric,
                Some(9000),
                sym(1.0, 0.0),
            ),
            (
                "http://10.0.0.4:30000",
                WorkerMode::Symmetric,
                Some(9000),
                sym(0.0, 1.0),
            ),
            (
                "http://10.0.0.5:30000",
                WorkerMode::Symmetric,
                Some(9000),
                sym(0.0, 1.0),
            ),
        ])
    }

    fn ca_config() -> SymmetricPdConfig {
        let mut cfg = SymmetricPdConfig {
            max_queue_wait_ms: 2_000,
            max_uncached_tokens: 1,
            gateway_cache_aware: Some(crate::config::GatewayCacheAwareConfig::default()),
            ..Default::default()
        };
        cfg.placement.force_split = true;
        cfg.placement.d_select = DSelect::RoundRobin;
        cfg
    }

    /// `--symmetric-pd-p-select gateway-cache-aware`: P is the gateway rule's pick
    /// (the same pick as the gateway policy for the same tree and loads), a
    /// repeated prefix follows its tree insert, D stays round robin, and the
    /// hold queue never holds (every P is over the cap here).
    #[tokio::test]
    async fn gateway_cache_aware_p_select_follows_the_gateway_rule_without_holding() {
        let reg = ca_registry();
        let rt = SymmetricPdRuntime::spawn(ca_config(), None, Arc::clone(&reg), model());
        let p_urls = [
            "http://10.0.0.1:30000",
            "http://10.0.0.2:30000",
            "http://10.0.0.3:30000",
        ];
        let _held: Vec<LedgerHandle> = p_urls.iter().map(|p| rt.account(p, 1_000_000)).collect();
        let tokens: Vec<u32> = (0..200).collect();
        let start = Instant::now();
        let first = rt
            .admit(PlaceInput {
                tokens: Some(tokens.clone()),
                prompt_len: 200,
                rid: Some("r1".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(start.elapsed() < Duration::from_millis(500), "no hold");
        assert!(!first.forced);
        assert!(!first.is_local());
        assert!(p_urls.contains(&first.p_worker.url.as_str()));
        // The reference: the gateway rule alone over the same P pool, with
        // the same first insert.
        let reference = crate::policies::gateway_cache_aware::GatewayCacheAwarePolicy::new(
            crate::config::GatewayCacheAwareConfig::default(),
        );
        let pool: Vec<Arc<Worker>> = p_urls
            .iter()
            .map(|u| reg.get(&WorkerId(u.to_string())).unwrap())
            .collect();
        reference.init_workers(p_urls.iter().copied());
        let first_idx = pool
            .iter()
            .position(|w| w.url == first.p_worker.url)
            .unwrap();
        let (seeded, _) = reference.select_for_tokens(
            &pool[first_idx..=first_idx],
            Some(tokens.as_slice()),
            None,
        );
        assert_eq!(seeded, Some(0));
        // A longer prompt with the same prefix: a cache hit in both.
        let longer: Vec<u32> = (0..240).collect();
        let second = rt
            .admit(PlaceInput {
                tokens: Some(longer.clone()),
                prompt_len: 240,
                rid: Some("r2".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(
            second.p_worker.url, first.p_worker.url,
            "cache hit on the tree"
        );
        let (reference_pick, note) =
            reference.select_for_tokens(&pool, Some(longer.as_slice()), None);
        assert_eq!(pool[reference_pick.unwrap()].url, second.p_worker.url);
        assert_eq!(note["branch"], "cache_hit", "{note}");
        // D round robin: consecutive transfer seeds alternate the two Ds.
        assert_ne!(first.d_worker.url, second.d_worker.url);
        first.ledger.retire();
        second.ledger.retire();
    }

    /// Without the selector the hold queue behaves as before (forced after
    /// its cap when every P is over the threshold).
    #[tokio::test]
    async fn cache_first_still_holds_when_the_gateway_selector_is_off() {
        let mut cfg = ca_config();
        cfg.gateway_cache_aware = None;
        cfg.max_queue_wait_ms = 150;
        let rt = SymmetricPdRuntime::spawn(cfg, None, ca_registry(), model());
        let _held: Vec<LedgerHandle> = [
            "http://10.0.0.1:30000",
            "http://10.0.0.2:30000",
            "http://10.0.0.3:30000",
        ]
        .iter()
        .map(|p| rt.account(p, 1_000_000))
        .collect();
        let d = rt
            .admit(PlaceInput {
                prompt_len: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(d.forced);
    }

    /// Late D on symmetric workers: the link owners and the capacity poll
    /// follow the roles whatever the breakers say (as the PD links and poll
    /// cover every prefill / decode worker); the late-D candidate pool and
    /// the placement candidates still skip a breaker that refuses.
    #[tokio::test]
    async fn late_d_members_ignore_the_breaker_and_candidates_do_not() {
        let reg = ca_registry();
        let rt = SymmetricPdRuntime::spawn(
            SymmetricPdConfig::default(),
            None,
            Arc::clone(&reg),
            model(),
        );
        let urls = |ws: Vec<Arc<Worker>>| ws.iter().map(|w| w.url.clone()).collect::<Vec<_>>();
        for url in ["http://10.0.0.1:30000", "http://10.0.0.4:30000"] {
            let w = reg.get(&WorkerId(url.to_string())).unwrap();
            for _ in 0..3 {
                w.breaker.record_failure();
            }
            assert!(!w.breaker.would_allow());
        }
        assert_eq!(
            urls(rt.prefill_members()),
            [
                "http://10.0.0.1:30000",
                "http://10.0.0.2:30000",
                "http://10.0.0.3:30000",
            ]
        );
        assert_eq!(
            urls(rt.decode_members()),
            ["http://10.0.0.4:30000", "http://10.0.0.5:30000"]
        );
        assert_eq!(urls(rt.decode_workers()), ["http://10.0.0.5:30000"]);
        let (weights, index) = rt.candidates();
        assert!(!weights.workers.contains_key("http://10.0.0.1:30000"));
        assert!(!index.contains_key("http://10.0.0.4:30000"));
        assert_eq!(index.len(), 3);
    }

    const CF_P1: &str = "http://10.0.0.1:30000";
    const CF_P2: &str = "http://10.0.0.2:30000";
    const CF_D: &str = "http://10.0.0.3:30000";
    /// A closed port: the indexer query fails, so no request has a hit.
    const DEAD_INDEXER: &str = "http://127.0.0.1:9";

    /// `--symmetric-pd-p-select cost-function` on the P-role workers P1 and P2
    /// and the D-role worker D. The indexer maps ids 1 and 2 to P1 and P2;
    /// the cost trace is `dir/cost.jsonl`. `tune` edits the config last.
    fn cost_function_runtime(
        dir: &Path,
        indexer_url: &str,
        trade: crate::scheduler::trade::TradeParams,
        tune: impl FnOnce(&mut SymmetricPdConfig),
    ) -> Arc<SymmetricPdRuntime> {
        let reg = registry_with(vec![
            (CF_P1, WorkerMode::Symmetric, Some(9000), sym(1.0, 0.0)),
            (CF_P2, WorkerMode::Symmetric, Some(9000), sym(1.0, 0.0)),
            (CF_D, WorkerMode::Symmetric, Some(9000), sym(0.0, 1.0)),
        ]);
        let kv = KvHintConfig {
            indexer_url: indexer_url.into(),
            min_tokens: 256,
            min_incremental_tokens: 512,
            local_prefix_fallback: false,
            indexer_timeout_ms: 2000,
            worker_map: [CF_P1, CF_P2]
                .iter()
                .enumerate()
                .map(|(i, url)| KvWorkerMapEntry {
                    id: i as u64 + 1,
                    url: url.to_string(),
                    servicer: format!("10.0.0.{}:8000", i + 1),
                })
                .collect(),
        };
        let mut cfg = SymmetricPdConfig::default();
        cfg.placement.force_split = true;
        cfg.placement.p_select = PSelect::CostFunction;
        cfg.pull_hint = true;
        cfg.cost_function = Some(crate::config::SymmetricPdCostFunctionConfig {
            trade,
            cost_trace: Some(dir.join("cost.jsonl")),
            load_poll_ms: 100,
            unknown_expire_secs: 60,
            loads_host_room: false,
            gwq_auto: None,
            trace_session_moves: false,
            trade_e3: None,
            pull_from_any_holder: false,
            trace_async: false,
        });
        tune(&mut cfg);
        SymmetricPdRuntime::spawn(cfg, Some(&kv), reg, model())
    }

    /// The rows of the cost trace in `dir` (empty before the first row).
    fn cost_rows(dir: &Path) -> Vec<Value> {
        std::fs::read_to_string(dir.join("cost.jsonl"))
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    /// The rids of the trace rows with this `event`, in trace order.
    fn cost_rids(dir: &Path, event: &str) -> Vec<String> {
        cost_rows(dir)
            .iter()
            .filter(|e| e["event"] == event)
            .map(|e| e["entry"]["rid"].as_str().unwrap_or("").to_string())
            .collect()
    }

    /// Admit `rid` with an `n`-token prompt (token ids, so the indexer is
    /// queried).
    async fn admit_tokens(
        rt: &Arc<SymmetricPdRuntime>,
        rid: Option<&str>,
        n: usize,
    ) -> Result<Decision, SymmetricPdError> {
        let input = PlaceInput {
            tokens: Some(vec![7; n]),
            prompt_len: n as u64,
            rid: rid.map(str::to_string),
            ..Default::default()
        };
        rt.admit(input).await
    }

    #[tokio::test]
    /// Check the cost function never holds a request.
    ///
    /// # Inputs
    /// No arguments; every P backlog in the Backend ledger is above
    /// `max_uncached_tokens` and the hold deadline is 10 s.
    ///
    /// # Outputs
    /// No return value; each admit returns at once, unforced.
    async fn cost_function_never_holds() {
        use crate::scheduler::trade::TradeParams;
        let dir = tempfile::tempdir().unwrap();
        let rt = cost_function_runtime(dir.path(), DEAD_INDEXER, TradeParams::default(), |cfg| {
            cfg.max_uncached_tokens = 1;
            cfg.max_queue_wait_ms = 10_000;
        });
        let mut decisions = Vec::new();
        for i in 0..4 {
            let rid = format!("r{i}");
            let admit = admit_tokens(&rt, Some(rid.as_str()), 4096);
            let d = tokio::time::timeout(Duration::from_secs(2), admit)
                .await
                .expect("the cost function never holds")
                .unwrap();
            assert!(!d.forced);
            decisions.push(d);
        }
        let loads = rt.loads([CF_P1, CF_P2]);
        assert_eq!(loads.get(CF_P1).backlog_uncached_tokens, 2 * 4096);
        assert_eq!(loads.get(CF_P2).backlog_uncached_tokens, 2 * 4096);
    }

    #[tokio::test]
    /// Check each request makes one cost-book reservation and the Decision
    /// carries it; the next request sees the reserved work.
    ///
    /// # Inputs
    /// No arguments; two admits without indexer hits.
    ///
    /// # Outputs
    /// No return value; `a` takes P1 (idle tie, lowest URL), `b` takes P2,
    /// and the trace has one labelled `reserve` row per request.
    async fn cost_function_reserves_once_per_request() {
        use crate::scheduler::trade::TradeParams;
        let dir = tempfile::tempdir().unwrap();
        let rt = cost_function_runtime(dir.path(), DEAD_INDEXER, TradeParams::default(), |_| {});
        let a = admit_tokens(&rt, Some("a"), 4096).await.unwrap();
        let b = admit_tokens(&rt, Some("b"), 4096).await.unwrap();
        assert_eq!(
            (a.p_worker.url.as_str(), b.p_worker.url.as_str()),
            (CF_P1, CF_P2)
        );
        for d in [&a, &b] {
            assert!(d.cost_reservation.is_some());
            assert!(d.pull_hint.is_none());
            assert_eq!(d.d_worker.url, CF_D);
            assert_eq!(d.branch(), "split");
            assert_eq!(d.placement.term(TERM_P_REASON), Some(6.0));
        }
        assert_eq!(cost_rids(dir.path(), "reserve"), ["a", "b"]);
        let rows = cost_rows(dir.path());
        assert!(rows
            .iter()
            .all(|e| e["policy"] == "symmetric_pd_cost_function"));
        let reserve = rows.iter().find(|e| e["event"] == "reserve").unwrap();
        assert_eq!(reserve["selection_policy"], "trade");
        assert_eq!(reserve["entry"]["worker_url"], CF_P1);
    }

    #[tokio::test]
    /// Check a Decision dropped before its P-leg dispatch rolls its cost
    /// reservation back.
    ///
    /// # Inputs
    /// No arguments; one admit, then the Decision is dropped.
    ///
    /// # Outputs
    /// No return value; the trace has a `rolled_back` row for the rid.
    async fn cost_function_rolls_back_when_dropped_before_dispatch() {
        use crate::scheduler::trade::TradeParams;
        let dir = tempfile::tempdir().unwrap();
        let rt = cost_function_runtime(dir.path(), DEAD_INDEXER, TradeParams::default(), |_| {});
        let d = admit_tokens(&rt, Some("gone"), 4096).await.unwrap();
        assert!(cost_rids(dir.path(), "rolled_back").is_empty());
        drop(d);
        assert_eq!(cost_rids(dir.path(), "rolled_back"), ["gone"]);
        // The rolled-back work no longer counts: the next request takes P1.
        let next = admit_tokens(&rt, Some("next"), 4096).await.unwrap();
        assert_eq!(next.p_worker.url, CF_P1);
    }

    #[tokio::test]
    /// Check the cost function refuses a request without a rid (the engine
    /// progress is keyed by it) and reserves nothing.
    ///
    /// # Inputs
    /// No arguments; one admit with `rid: None`.
    ///
    /// # Outputs
    /// No return value; `SymmetricPdError::CostFunction` names the rid rule.
    async fn cost_function_refuses_a_missing_rid() {
        use crate::scheduler::trade::TradeParams;
        let dir = tempfile::tempdir().unwrap();
        let rt = cost_function_runtime(dir.path(), DEAD_INDEXER, TradeParams::default(), |_| {});
        match admit_tokens(&rt, None, 4096).await {
            Err(SymmetricPdError::CostFunction(error)) => {
                assert!(
                    error.contains("uncached balance requires a unique string rid"),
                    "{error}"
                );
            }
            other => panic!("expected a cost-function refusal, got {other:?}"),
        }
        let loads = rt.loads([CF_P1, CF_P2]);
        assert_eq!(loads.get(CF_P1).backlog_uncached_tokens, 0);
        assert_eq!(loads.get(CF_P2).backlog_uncached_tokens, 0);
        assert!(cost_rids(dir.path(), "reserve").is_empty());
    }

    #[tokio::test]
    /// Check the Backend ledger charges prompt minus the TRADE expected hit.
    ///
    /// # Inputs
    /// No arguments; the indexer reports 3,000 cached tokens on P1. `busy`
    /// (60,000 tokens) stays on P1. With lambda 0, `pull` (8,192 tokens)
    /// moves to P2 and pulls P1's 3,000 tokens (P2 costs 5,192 + 13,500 *
    /// 0.453 s; P1 costs 57,000 + 5,192).
    ///
    /// # Outputs
    /// No return value; P2 is charged 5,192 tokens, not its local 8,192.
    async fn cost_function_accounts_the_trade_expected_hit() {
        use crate::scheduler::trade::TradeParams;
        let reply = json!({"instances": {"1": {"cpu": 3000}}});
        let app = axum::Router::new().route(
            "/query",
            axum::routing::post(move |_body: Bytes| {
                let reply = reply.clone();
                async move { axum::Json(reply) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let indexer = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let dir = tempfile::tempdir().unwrap();
        let lambda0 = TradeParams {
            copy_lambda: 0.0,
            ..TradeParams::default()
        };
        let rt = cost_function_runtime(dir.path(), &indexer, lambda0, |_| {});
        let busy = admit_tokens(&rt, Some("busy"), 60_000).await.unwrap();
        assert_eq!(busy.p_worker.url, CF_P1);
        assert!(busy.pull_hint.is_none());
        let loads = rt.loads([CF_P1]);
        assert_eq!(loads.get(CF_P1).backlog_uncached_tokens, 57_000);
        let pull = admit_tokens(&rt, Some("pull"), 8192).await.unwrap();
        assert_eq!(pull.p_worker.url, CF_P2);
        let hint = pull.pull_hint.as_ref().expect("P2 pulls from P1");
        assert_eq!(hint["source_worker_id"], CF_P1);
        assert_eq!(hint["expected_hit_tokens"], 3000);
        assert_eq!(pull.placement.term(TERM_EXPECTED_HIT_TOKENS), Some(3000.0));
        let loads = rt.loads([CF_P2]);
        assert_eq!(loads.get(CF_P2).backlog_uncached_tokens, 8192 - 3000);
        server.abort();
    }

    // ---- --symmetric-pd-gwq-auto under external role control ----

    const CF_D2: &str = "http://10.0.0.4:30000";

    /// GWQ-AUTO with external role control (5 s drain, ended by hand here):
    /// P1 and P2 P-role, D and D2 D-role, every request cold (closed indexer
    /// port), the 300-s guard. The test steps the queue itself
    /// ([`super::super::cost_function::CostFunction::gwq_pass`]).
    fn gwq_role_runtime(dir: &Path) -> Arc<SymmetricPdRuntime> {
        let reg = registry_with(vec![
            (CF_P1, WorkerMode::Symmetric, Some(9000), sym(1.0, 0.0)),
            (CF_P2, WorkerMode::Symmetric, Some(9000), sym(1.0, 0.0)),
            (CF_D, WorkerMode::Symmetric, Some(9000), sym(0.0, 1.0)),
            (CF_D2, WorkerMode::Symmetric, Some(9000), sym(0.0, 1.0)),
        ]);
        let kv = KvHintConfig {
            indexer_url: DEAD_INDEXER.into(),
            min_tokens: 256,
            min_incremental_tokens: 512,
            local_prefix_fallback: false,
            indexer_timeout_ms: 200,
            worker_map: [CF_P1, CF_P2, CF_D, CF_D2]
                .iter()
                .enumerate()
                .map(|(i, url)| KvWorkerMapEntry {
                    id: i as u64 + 1,
                    url: url.to_string(),
                    servicer: format!("10.0.0.{}:8000", i + 1),
                })
                .collect(),
        };
        let mut cfg = SymmetricPdConfig::default();
        cfg.placement.force_split = true;
        cfg.placement.p_select = PSelect::CostFunction;
        cfg.pull_hint = true;
        cfg.external_role_control = true;
        cfg.role_drain_timeout = Some(Duration::from_secs(5));
        cfg.cost_function = Some(crate::config::SymmetricPdCostFunctionConfig {
            trade: crate::scheduler::trade::TradeParams::default(),
            cost_trace: Some(dir.join("cost.jsonl")),
            load_poll_ms: 100,
            unknown_expire_secs: 60,
            loads_host_room: false,
            gwq_auto: Some(crate::config::SymmetricPdGwqAutoConfig {
                max_wait_ms: 300_000,
                ..Default::default()
            }),
            trace_session_moves: false,
            trade_e3: None,
            pull_from_any_holder: false,
            trace_async: false,
        });
        SymmetricPdRuntime::spawn(cfg, Some(&kv), reg, model())
    }

    /// The cost function with `--trade-wait e3` (gate m 0, the
    /// shipped bound table of tests/fixtures/e3) under external role control:
    /// P1 and P2 P-role, D and D2 D-role, every request cold (closed indexer
    /// port), no state push and no engine snapshot (each P keeps the ledger
    /// terms, fallback `no_view`).
    fn e3_role_runtime(dir: &Path) -> Arc<SymmetricPdRuntime> {
        use crate::scheduler::trade::{TradeParams, TradeWait};
        let table = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/e3/bound-params.json"
        );
        let e3 = crate::trade_e3::TradeE3Config::load(
            Path::new(table),
            crate::trade_e3::DEFAULT_BOUND_MODE,
            None,
            0.0,
            None,
        )
        .unwrap();
        let reg = registry_with(vec![
            (CF_P1, WorkerMode::Symmetric, Some(9000), sym(1.0, 0.0)),
            (CF_P2, WorkerMode::Symmetric, Some(9000), sym(1.0, 0.0)),
            (CF_D, WorkerMode::Symmetric, Some(9000), sym(0.0, 1.0)),
            (CF_D2, WorkerMode::Symmetric, Some(9000), sym(0.0, 1.0)),
        ]);
        let kv = KvHintConfig {
            indexer_url: DEAD_INDEXER.into(),
            min_tokens: 256,
            min_incremental_tokens: 512,
            local_prefix_fallback: false,
            indexer_timeout_ms: 200,
            worker_map: [CF_P1, CF_P2, CF_D, CF_D2]
                .iter()
                .enumerate()
                .map(|(i, url)| KvWorkerMapEntry {
                    id: i as u64 + 1,
                    url: url.to_string(),
                    servicer: format!("10.0.0.{}:8000", i + 1),
                })
                .collect(),
        };
        let mut cfg = SymmetricPdConfig::default();
        cfg.placement.force_split = true;
        cfg.placement.p_select = PSelect::CostFunction;
        cfg.pull_hint = true;
        cfg.external_role_control = true;
        cfg.role_drain_timeout = Some(Duration::from_secs(5));
        cfg.cost_function = Some(crate::config::SymmetricPdCostFunctionConfig {
            trade: TradeParams {
                wait: TradeWait::E3,
                gate_m: 0.0,
                ..TradeParams::default()
            },
            cost_trace: Some(dir.join("cost.jsonl")),
            load_poll_ms: 100,
            unknown_expire_secs: 60,
            loads_host_room: false,
            gwq_auto: None,
            trace_session_moves: false,
            trade_e3: Some(crate::config::SymmetricPdTradeE3(Arc::new(e3))),
            pull_from_any_holder: false,
            trace_async: false,
        });
        SymmetricPdRuntime::spawn(cfg, Some(&kv), reg, model())
    }

    /// One cold 5,000-token admission with the E3 session key `session`.
    async fn admit_e3(rt: &Arc<SymmetricPdRuntime>, rid: &str, session: &str) -> Decision {
        let input = PlaceInput {
            tokens: Some(vec![7; 5_000]),
            prompt_len: 5_000,
            rid: Some(rid.into()),
            e3_session: Some(session.into()),
            ..Default::default()
        };
        rt.admit(input).await.unwrap()
    }

    /// The trade decision of the reserve event of `rid`.
    fn e3_decision(dir: &Path, rid: &str) -> Value {
        cost_rows(dir)
            .into_iter()
            .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == rid)
            .map(|e| e["trade_decision"].clone())
            .unwrap_or_else(|| panic!("no reserve event of {rid}"))
    }

    /// The candidate URLs of a trade decision, sorted.
    fn e3_candidate_urls(decision: &Value) -> Vec<String> {
        let mut urls: Vec<String> = decision["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["worker_url"].as_str().unwrap().to_string())
            .collect();
        urls.sort();
        urls
    }

    #[tokio::test]
    /// `--trade-wait e3` under external role control (the CUA
    /// dynamic router): every decision carries the e3 terms and one e3 row
    /// per candidate, the candidates are the current P-role pool, the E3
    /// session key reaches the host rule (the second request of a session
    /// sees its gap on the P of the first), and each routed request has an
    /// `e3_shadow` line. After the controller moves P1 to decode and D2 to
    /// prefill (drain ended), the candidates are P2 and D2.
    async fn e3_prices_the_current_p_role_pool_under_external_roles() {
        use crate::backend::registry::RoleAssignment;
        let dir = tempfile::tempdir().unwrap();
        let rt = e3_role_runtime(dir.path());
        assert!(rt.cost_function_wants_e3_session());
        let first = admit_e3(&rt, "e3-1", "agent-a/0").await;
        let second = admit_e3(&rt, "e3-2", "agent-a/0").await;
        let d1 = e3_decision(dir.path(), "e3-1");
        assert_eq!(d1["wait"], "e3", "{d1}");
        assert_eq!(d1["e3_gate_m"], json!(0.0));
        assert!(d1["e3_decision_s"].is_number(), "{d1}");
        assert!(d1["e3_chosen"].is_object(), "{d1}");
        assert_eq!(e3_candidate_urls(&d1), [CF_P1, CF_P2]);
        for c in d1["candidates"].as_array().unwrap() {
            assert_eq!(c["wait_source"], "ledger_fallback", "{c}");
            assert_eq!(c["wait_fallback_reason"], "no_view", "{c}");
            assert_eq!(c["e3_host_rule"], "cold", "{c}");
        }
        // The session went to the first P: the second decision sees its gap
        // there (and none on the other P).
        let p1 = first.p_worker.url.clone();
        let d2 = e3_decision(dir.path(), "e3-2");
        for c in d2["candidates"].as_array().unwrap() {
            let gap = &c["e3_session_gap_s"];
            if c["worker_url"] == p1.as_str() {
                assert!(gap.as_f64().is_some_and(|g| g >= 0.0), "{c}");
            } else {
                assert!(gap.is_null(), "{c}");
            }
        }
        // P1 goes to decode, D2 to prefill (a drain, then ended).
        let now = rt.role_assignments().unwrap();
        let mut desired = now.roles.clone();
        desired.insert(CF_P1.into(), RoleAssignment::Decode);
        desired.insert(CF_D2.into(), RoleAssignment::Prefill);
        let (after, started) = rt
            .replace_role_assignments(&now.epoch, now.revision, &now.roles, &desired)
            .unwrap();
        assert_eq!(started, [CF_D2]);
        assert!(rt.finish_role_drain(CF_D2, after.revision));
        let third = admit_e3(&rt, "e3-3", "agent-b/0").await;
        let d3 = e3_decision(dir.path(), "e3-3");
        assert_eq!(d3["wait"], "e3", "{d3}");
        assert_eq!(e3_candidate_urls(&d3), [CF_P2, CF_D2]);
        assert!([CF_P2, CF_D2].contains(&third.p_worker.url.as_str()));
        assert!(
            [CF_P1, CF_D].contains(&third.d_worker.url.as_str()),
            "D-leg {} on a D-role worker",
            third.d_worker.url
        );
        let shadows: Vec<String> = cost_rows(dir.path())
            .iter()
            .filter(|e| e["event"] == "e3_shadow")
            .map(|e| e["rid"].as_str().unwrap_or("").to_string())
            .collect();
        assert_eq!(shadows, ["e3-1", "e3-2", "e3-3"]);
        drop((first, second, third));
    }

    /// A dispatched 100,000-token reservation `rid` on `url` (listed by
    /// [`gwq_views`], its queued work 91,808 is above the lead depth 16,293).
    fn gwq_busy(
        book: &Arc<crate::cost_balance::CostBook>,
        url: &str,
        rid: &str,
    ) -> crate::cost_balance::Reservation {
        let candidate = crate::cost_balance::Candidate {
            worker_url: url.into(),
            local_hit: 0,
            gpu_hit: None,
            expected_hit: 0,
            hint: None,
        };
        let (_, mut reservation) = book
            .reserve_uncached(100_000, Some(rid.into()), vec![candidate], 12288)
            .unwrap();
        reservation.mark_dispatched().unwrap();
        reservation
    }

    /// The progress views that [`gwq_views`] applies every 100 ms: per URL
    /// the listed (rid, remaining tokens). A P's view is stale (fail-open)
    /// 1 s after its last applied snapshot.
    type GwqViews = Arc<Mutex<Vec<(&'static str, Vec<(&'static str, u64)>)>>>;

    /// Apply `views` every 100 ms until `stop` (the poll of the test).
    fn gwq_views(
        book: &Arc<crate::cost_balance::CostBook>,
        views: &GwqViews,
        stop: &Arc<std::sync::atomic::AtomicBool>,
    ) -> tokio::task::JoinHandle<()> {
        let (book, views, stop) = (Arc::clone(book), Arc::clone(views), Arc::clone(stop));
        tokio::spawn(async move {
            let mut sequence = 0u64;
            while !stop.load(Ordering::Relaxed) {
                sequence += 1;
                let snapshot = views.lock().clone();
                for (url, rids) in snapshot {
                    let progress = crate::cost_balance::PrefillProgress {
                        epoch: format!("e-{url}"),
                        sequence,
                        remaining: rids.iter().map(|(r, n)| (r.to_string(), *n)).collect(),
                        truncated: false,
                    };
                    book.apply_progress(url, progress, json!({}), Instant::now())
                        .unwrap();
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
    }

    /// Wait up to 3 s for `pred`.
    async fn gwq_wait(pred: impl Fn() -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(3);
        while !pred() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        pred()
    }

    #[tokio::test]
    /// GWQ-AUTO steps over the current P-role pool only. A cold request is
    /// held while P1 and P2 are busy. The controller moves P1 to decode (its
    /// queue then empties) and D2 to prefill (a drain starts): no release
    /// goes to P1 (a D-role worker with room) or to the draining D2. When the
    /// drain ends, D2 is a P without a progress view yet (fail-open) and the
    /// request is released and placed there; its D-leg is a D-role worker.
    async fn gwq_auto_releases_only_to_the_current_p_role_pool() {
        use crate::backend::registry::RoleAssignment;
        let dir = tempfile::tempdir().unwrap();
        let rt = gwq_role_runtime(dir.path());
        let cf = Arc::clone(rt.cost_function_for_test());
        let book = Arc::clone(cf.book_for_test());
        let mut b1 = gwq_busy(&book, CF_P1, "busy-1");
        let b2 = gwq_busy(&book, CF_P2, "busy-2");
        let views: GwqViews = Arc::new(Mutex::new(vec![
            (CF_P1, vec![("busy-1", 100_000)]),
            (CF_P2, vec![("busy-2", 100_000)]),
        ]));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let poller = gwq_views(&book, &views, &stop);
        tokio::time::sleep(Duration::from_millis(150)).await;
        let task = {
            let rt = Arc::clone(&rt);
            tokio::spawn(async move { admit_tokens(&rt, Some("held-1"), 5_000).await })
        };
        assert!(
            gwq_wait(|| cf.gwq_len() == 1).await,
            "held: P1 and P2 are busy"
        );
        let now = rt.role_assignments().unwrap();
        let mut desired = now.roles.clone();
        desired.insert(CF_P1.into(), RoleAssignment::Decode);
        desired.insert(CF_D2.into(), RoleAssignment::Prefill);
        let (after, started) = rt
            .replace_role_assignments(&now.epoch, now.revision, &now.roles, &desired)
            .unwrap();
        assert_eq!(started, [CF_D2]);
        // P1 (now decode) finished its work; D2 drains; P2 stays busy.
        b1.finish("prefill_response_success").unwrap();
        views.lock()[0].1.clear();
        cf.gwq_pass(0, 1_000_000);
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert_eq!(
            cf.gwq_len(),
            1,
            "no release to the D-role P1 or the draining D2"
        );
        assert!(!task.is_finished());
        assert!(cost_rids(dir.path(), "reserve")
            .iter()
            .all(|rid| rid != "held-1"));
        assert!(rt.finish_role_drain(CF_D2, after.revision));
        cf.gwq_pass(0, 1_000_001);
        let d = tokio::time::timeout(Duration::from_secs(3), task)
            .await
            .expect("released once D2 is a P")
            .unwrap()
            .unwrap();
        assert_eq!(d.p_worker.url, CF_D2);
        assert!(
            [CF_P1, CF_D].contains(&d.d_worker.url.as_str()),
            "D-leg {} on a D-role worker",
            d.d_worker.url
        );
        let row = cost_rows(dir.path())
            .into_iter()
            .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == "held-1")
            .unwrap();
        assert_eq!(row["gwq"]["p_release"], CF_D2, "{row}");
        assert_eq!(row["gwq"]["release"], "stale_view", "{row}");
        assert!(row["gwq"].get("walk_mismatch").is_none(), "{row}");
        stop.store(true, Ordering::Relaxed);
        poller.await.unwrap();
        drop((d, b1, b2));
    }

    #[tokio::test]
    /// The released P left the P-role pool between the release step and the
    /// placement (the step read the pool just before the controller moved P1
    /// to decode): the reservation rolls back (its rid is free again), the
    /// request is walked again over the current pool (gwq_rewalk, counter
    /// rewalks) and placed on P2 once P2 has room.
    async fn gwq_auto_walks_again_when_the_released_p_left_the_pool() {
        use crate::backend::registry::RoleAssignment;
        let dir = tempfile::tempdir().unwrap();
        let rt = gwq_role_runtime(dir.path());
        let cf = Arc::clone(rt.cost_function_for_test());
        let book = Arc::clone(cf.book_for_test());
        let mut b1 = gwq_busy(&book, CF_P1, "busy-1");
        let b2 = gwq_busy(&book, CF_P2, "busy-2");
        let views: GwqViews = Arc::new(Mutex::new(vec![
            (CF_P1, vec![("busy-1", 100_000)]),
            (CF_P2, vec![("busy-2", 100_000)]),
        ]));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let poller = gwq_views(&book, &views, &stop);
        tokio::time::sleep(Duration::from_millis(150)).await;
        let task = {
            let rt = Arc::clone(&rt);
            tokio::spawn(async move { admit_tokens(&rt, Some("held-2"), 5_000).await })
        };
        assert!(
            gwq_wait(|| cf.gwq_len() == 1).await,
            "held: P1 and P2 are busy"
        );
        let now = rt.role_assignments().unwrap();
        let mut desired = now.roles.clone();
        desired.insert(CF_P1.into(), RoleAssignment::Decode);
        rt.replace_role_assignments(&now.epoch, now.revision, &now.roles, &desired)
            .unwrap();
        // P1 (now decode) finished its work: it would have room as a P.
        b1.finish("prefill_response_success").unwrap();
        views.lock()[0].1.clear();
        tokio::time::sleep(Duration::from_millis(150)).await;
        let p1 = rt
            .role_members_all()
            .into_iter()
            .find(|w| w.url == CF_P1)
            .unwrap();
        cf.gwq_set_next_pool(vec![p1]);
        cf.gwq_pass(0, 2_000_000);
        assert!(
            gwq_wait(|| cf.gwq_counters()["rewalks"] == 1 && cf.gwq_len() == 1).await,
            "walked again and held: {}",
            cf.gwq_counters()
        );
        assert_eq!(cost_rids(dir.path(), "rolled_back"), ["held-2"]);
        let rewalk = cost_rows(dir.path())
            .into_iter()
            .find(|e| e["event"] == "gwq_rewalk")
            .unwrap();
        assert_eq!(
            (
                rewalk["rid"].clone(),
                rewalk["worker_url"].clone(),
                rewalk["attempt"].clone()
            ),
            (json!("held-2"), json!(CF_P1), json!(1))
        );
        // P2 gets room (10,000 left: queued 1,808).
        views.lock()[1].1 = vec![("busy-2", 10_000)];
        tokio::time::sleep(Duration::from_millis(150)).await;
        cf.gwq_pass(0, 2_000_001);
        let d = tokio::time::timeout(Duration::from_secs(3), task)
            .await
            .expect("released to P2")
            .unwrap()
            .unwrap();
        assert_eq!(d.p_worker.url, CF_P2);
        let reserves: Vec<Value> = cost_rows(dir.path())
            .into_iter()
            .filter(|e| e["event"] == "reserve" && e["entry"]["rid"] == "held-2")
            .collect();
        assert_eq!(reserves.len(), 2, "P1 (rolled back), then P2");
        assert_eq!(reserves[0]["entry"]["worker_url"], CF_P1);
        assert_eq!(reserves[1]["entry"]["worker_url"], CF_P2);
        stop.store(true, Ordering::Relaxed);
        poller.await.unwrap();
        drop((d, b1, b2));
    }
}
