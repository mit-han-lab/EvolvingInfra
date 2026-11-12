// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `--policy gateway_cache_aware`: the prefill selection of sgl-model-gateway
//! `--prefill-policy cache_aware`, a port of SGLang's
//! `sgl-model-gateway/src/policies/cache_aware.rs`. The tree is
//! [`super::gateway_ca_tree`], the gateway's `tree.rs`.
//!
//! # Decision rule (per request, over the prefill candidates)
//!
//! 1. `min` / `max` = the smallest / largest load over all candidates. The
//!    load is [`Worker::active_load`]: the in-flight prefill requests this
//!    router counts (taken after selection, released when the P leg
//!    answers), as the gateway's `Worker::load()` (`WorkerLoadGuard` around
//!    the P HTTP leg). The engine load poll (`--load-check-interval-secs`)
//!    feeds only the gateway's `power_of_two` decode policy, not this rule.
//! 2. Imbalanced when `max - min > abs` and `max > min * rel` (f32): the
//!    least-loaded healthy candidate (uniform random tie), and the prompt is
//!    inserted into the tree for it.
//! 3. Otherwise the prompt is matched against the tree of all workers. The
//!    match rate is `matched tokens / prompt tokens` (complete tokens only;
//!    the salt segment is not counted). Rate `> cache_threshold`: the tenant
//!    of the deepest matched node (the node's creator while it still holds
//!    it). Else the least-loaded healthy candidate (random tie). The prompt
//!    is inserted for the chosen worker.
//! 4. A matched tenant that is not a healthy candidate is removed from the
//!    tree and the first healthy candidate is taken (no insert).
//!
//! Prompts: a token-id prompt is 8 lowercase hex characters per token
//! (`{token:08x}`), in a tree separate from text prompts. A `cache_salt`
//! puts the prompt under `"\0" + 16 hex characters of sha256(salt)`.
//! Every `eviction_interval_secs` a background thread evicts each worker's
//! least recently used leaves until its tree characters are at most
//! `max_tree_size`.
//!
//! # Parameters (gateway flag, gateway CLI default, this policy's flag)
//!
//! | gateway flag | default | here |
//! |---|---|---|
//! | `--cache-threshold` | 0.3 | `--gateway-ca-cache-threshold` |
//! | `--balance-abs-threshold` | 64 | `--gateway-ca-balance-abs-threshold` |
//! | `--balance-rel-threshold` | 1.5 | `--gateway-ca-balance-rel-threshold` |
//! | `--eviction-interval` | 120 s | `--gateway-ca-eviction-interval-secs` |
//! | `--max-tree-size` | 67108864 | `--gateway-ca-max-tree-size` |
//!
//! # Known differences from the gateway
//!
//! - Workers enter the trees (an empty insert, as the gateway's
//!   `init_workers`) at their first selection, not at registration; a
//!   worker removed from the registry stays a tenant until a match finds it
//!   unavailable. The LRU epoch counter starts at 0 per policy (see the tree
//!   module).
//! - Text prompts: `prompt` (a string, or the first string of an array) or
//!   `text`. The salt applies to a text prompt only on the gateway's
//!   extended completions route (a body with `forced_output_token_ids` or
//!   `return_token_ids`). Chat requests use their token ids when the ingress
//!   has them (the gateway uses the chat text). A batched token prompt
//!   (nested arrays) routes as an empty text prompt (the gateway uses the
//!   first prompt); late D choose refuses batches in both routers.
//! - `cache_salt` comes from the ingress (`kv_hint::extract_cache_salt`: a
//!   non-empty string); the gateway also takes the first string of an array.
//! - No mesh synchronization.

use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use rand::seq::IteratorRandom;
use serde_json::{json, Value};

use super::gateway_ca_tree::{EpochCounter, Tree};
use super::{Policy, SelectionContext};
use crate::config::GatewayCacheAwareConfig;
use crate::workers::Worker;

/// Tracing target of the periodic stats line and the eviction log.
pub const LOG_TARGET: &str = "gateway_ca";

/// Hex characters per token id in the token tree (gateway
/// `TOKEN_CACHE_HEX_WIDTH`).
pub const TOKEN_CACHE_HEX_WIDTH: usize = 8;

// A salted request can never share engine KV with any other salt, so its
// affinity entries live under a per-salt namespace segment prepended to the
// tree text. Fixed width guarantees no salt segment is a prefix of another,
// and the NUL sentinel keeps raw request text from colliding with a segment.
// The segment chars are excluded from match-rate accounting
// (CacheLookupKey::matched_unit_count), so cache_threshold semantics are
// unchanged.
const SALT_SEG_HEX_WIDTH: usize = 16;
/// Characters of a salt segment (NUL + 16 hex characters).
pub const SALT_SEG_CHARS: usize = 1 + SALT_SEG_HEX_WIDTH;

/// The tree prefix of a salted request: NUL, then the first 8 bytes of
/// sha256(salt) as 16 lowercase hex characters (gateway `salt_segment`).
pub fn salt_segment(salt: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(salt.as_bytes());
    let mut segment = String::with_capacity(SALT_SEG_CHARS);
    segment.push('\0');
    for byte in &digest[..SALT_SEG_HEX_WIDTH / 2] {
        write!(&mut segment, "{byte:02x}").expect("writing to String cannot fail");
    }
    segment
}

/// Text and token prompts live in separate trees (gateway
/// `CacheKeyNamespace`).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CacheKeyNamespace {
    Text,
    Tokens,
}

impl CacheKeyNamespace {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Tokens => "tokens",
        }
    }
}

/// The tree key of one request (gateway `CacheLookupKey`).
#[derive(Debug)]
pub struct CacheLookupKey<'a> {
    tree_text: Cow<'a, str>,
    namespace: CacheKeyNamespace,
    input_unit_count: usize,
    unit_width: usize,
    should_update: bool,
    /// Chars of the per-salt namespace segment leading `tree_text` (0 when
    /// unsalted). Excluded from match accounting: the segment is routing
    /// metadata, not request payload.
    salt_prefix_chars: usize,
}

impl<'a> CacheLookupKey<'a> {
    /// Gateway `CacheLookupKey::from_info`: `tokens` (non-empty) wins over
    /// `request_text`; an empty salt is no salt.
    pub fn new(
        tokens: Option<&[u32]>,
        request_text: Option<&'a str>,
        cache_salt: Option<&str>,
    ) -> Self {
        let salt_seg = cache_salt.filter(|salt| !salt.is_empty()).map(salt_segment);

        if let Some(tokens) = tokens.filter(|tokens| !tokens.is_empty()) {
            let salt_prefix_chars = salt_seg.as_ref().map_or(0, String::len);
            let mut encoded =
                String::with_capacity(salt_prefix_chars + tokens.len() * TOKEN_CACHE_HEX_WIDTH);
            if let Some(segment) = &salt_seg {
                encoded.push_str(segment);
            }
            for token in tokens {
                write!(&mut encoded, "{token:08x}").expect("writing to String cannot fail");
            }
            return Self {
                tree_text: Cow::Owned(encoded),
                namespace: CacheKeyNamespace::Tokens,
                input_unit_count: tokens.len(),
                unit_width: TOKEN_CACHE_HEX_WIDTH,
                should_update: true,
                salt_prefix_chars,
            };
        }

        let text = request_text.unwrap_or("");
        // Only salt actual payload: empty text must keep today's no-op
        // semantics (worker registration inserts "" without salt).
        match salt_seg.filter(|_| !text.is_empty()) {
            Some(segment) => Self {
                salt_prefix_chars: segment.len(),
                tree_text: Cow::Owned(format!("{segment}{text}")),
                namespace: CacheKeyNamespace::Text,
                input_unit_count: text.chars().count(),
                unit_width: 1,
                should_update: request_text.is_some(),
            },
            None => Self {
                tree_text: Cow::Borrowed(text),
                namespace: CacheKeyNamespace::Text,
                input_unit_count: text.chars().count(),
                unit_width: 1,
                should_update: request_text.is_some(),
                salt_prefix_chars: 0,
            },
        }
    }

    pub fn matched_unit_count(&self, matched_char_count: usize) -> usize {
        matched_char_count.saturating_sub(self.salt_prefix_chars) / self.unit_width
    }

    pub fn match_rate(&self, matched_char_count: usize) -> f32 {
        match_rate_units(
            self.matched_unit_count(matched_char_count),
            self.input_unit_count,
        )
    }

    fn tree_text_for_update(&self) -> Option<&str> {
        self.should_update.then_some(self.tree_text.as_ref())
    }

    pub fn tree_text(&self) -> &str {
        self.tree_text.as_ref()
    }

    pub fn namespace(&self) -> CacheKeyNamespace {
        self.namespace
    }

    pub fn input_unit_count(&self) -> usize {
        self.input_unit_count
    }

    pub fn salt_prefix_chars(&self) -> usize {
        self.salt_prefix_chars
    }
}

/// What the rule reads from a candidate. Read live at the same points as the
/// gateway (the load twice: the min/max pass, then the least-loaded pass).
pub trait CandidateView {
    fn url(&self) -> &str;
    fn load(&self) -> usize;
    fn is_healthy(&self) -> bool;
}

impl CandidateView for Arc<Worker> {
    fn url(&self) -> &str {
        &self.url
    }
    fn load(&self) -> usize {
        self.active_load()
    }
    fn is_healthy(&self) -> bool {
        self.breaker.would_allow()
    }
}

/// Chooses among tied least-loaded candidates (indices into the candidate
/// list, in list order). Production: uniform random, as the gateway.
pub trait TieBreak {
    fn pick(&mut self, tied: &[usize]) -> Option<usize>;
}

/// The gateway's tie rule: a uniform random choice.
pub struct RandomTie;

impl TieBreak for RandomTie {
    fn pick(&mut self, tied: &[usize]) -> Option<usize> {
        tied.iter().copied().choose(&mut rand::thread_rng())
    }
}

/// Which part of the rule chose the worker.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Branch {
    /// Load imbalance: least loaded, tree updated.
    Imbalanced,
    /// Match rate above the threshold: the matched tenant.
    CacheHit,
    /// Match rate at or below the threshold: least loaded.
    LowMatch,
    /// The matched tenant is no healthy candidate: tenant removed, first
    /// healthy candidate.
    StaleTenant,
    /// No healthy candidate (no choice).
    NoHealthy,
}

impl Branch {
    pub const ALL: [Branch; 5] = [
        Branch::Imbalanced,
        Branch::CacheHit,
        Branch::LowMatch,
        Branch::StaleTenant,
        Branch::NoHealthy,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Branch::Imbalanced => "imbalanced",
            Branch::CacheHit => "cache_hit",
            Branch::LowMatch => "low_match",
            Branch::StaleTenant => "stale_tenant",
            Branch::NoHealthy => "no_healthy",
        }
    }

    fn index(self) -> usize {
        match self {
            Branch::Imbalanced => 0,
            Branch::CacheHit => 1,
            Branch::LowMatch => 2,
            Branch::StaleTenant => 3,
            Branch::NoHealthy => 4,
        }
    }
}

/// One selection: the branch, the choice and the inputs the rule read.
#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    pub branch: Branch,
    /// Index into the candidate list; `None` = no selection.
    pub chosen: Option<usize>,
    pub namespace: CacheKeyNamespace,
    pub input_units: usize,
    /// Matched complete tokens (characters for text) of the tree match;
    /// `None` on the imbalanced branch (no match is made).
    pub matched_units: Option<usize>,
    pub match_rate: Option<f32>,
    /// The tenant of the deepest matched node.
    pub matched_tenant: Option<String>,
    pub min_load: usize,
    pub max_load: usize,
    /// Candidates tied at the least load on a least-loaded branch (0 else).
    pub tied: usize,
}

impl Decision {
    fn new(key: &CacheLookupKey<'_>) -> Self {
        Self {
            branch: Branch::NoHealthy,
            chosen: None,
            namespace: key.namespace,
            input_units: key.input_unit_count,
            matched_units: None,
            match_rate: None,
            matched_tenant: None,
            min_load: 0,
            max_load: 0,
            tied: 0,
        }
    }

    /// The per-request note of the routing log.
    pub fn note(&self) -> Value {
        json!({
            "rule": "gateway_cache_aware",
            "branch": self.branch.as_str(),
            "namespace": self.namespace.as_str(),
            "input_units": self.input_units,
            "matched_units": self.matched_units,
            "match_rate": self.match_rate,
            "min_load": self.min_load,
            "max_load": self.max_load,
            "tied": self.tied,
        })
    }
}

/// Gateway imbalance test (`select_worker`): f32 relative comparison as the
/// gateway.
pub fn is_imbalanced(min_load: usize, max_load: usize, config: &GatewayCacheAwareConfig) -> bool {
    max_load.saturating_sub(min_load) > config.balance_abs_threshold
        && (max_load as f32) > (min_load as f32 * config.balance_rel_threshold)
}

/// Gateway match rate: matched units over input units (f32; 0 for an empty
/// prompt).
pub fn match_rate_units(matched_units: usize, input_units: usize) -> f32 {
    if input_units == 0 {
        0.0
    } else {
        matched_units as f32 / input_units as f32
    }
}

/// The rule applied to recorded inputs (the gateway route-time observer:
/// per-candidate load, health and history units before the selection), for
/// offline replay. Returns the branch and the candidates the gateway may
/// choose: the least-loaded ties, or every candidate whose history equals
/// the deepest match (the gateway takes that node's creator, which the
/// observer does not record). A matched candidate that is unhealthy adds
/// the stale path's first healthy candidate.
pub fn decide_from_history(
    loads: &[usize],
    healthy: &[bool],
    history_units: &[usize],
    input_units: usize,
    config: &GatewayCacheAwareConfig,
) -> (Branch, Vec<usize>) {
    let healthy_indices: Vec<usize> = (0..loads.len()).filter(|&i| healthy[i]).collect();
    let Some(&first_healthy) = healthy_indices.first() else {
        return (Branch::NoHealthy, Vec::new());
    };
    let min_load = loads.iter().copied().min().unwrap_or(0);
    let max_load = loads.iter().copied().max().unwrap_or(0);
    let least_loaded = || {
        let pairs: Vec<(usize, usize)> = healthy_indices.iter().map(|&i| (i, loads[i])).collect();
        min_load_ties(&pairs)
    };
    if is_imbalanced(min_load, max_load, config) {
        return (Branch::Imbalanced, least_loaded());
    }
    let matched = history_units.iter().copied().max().unwrap_or(0);
    if match_rate_units(matched, input_units) > config.cache_threshold {
        let deepest: Vec<usize> = (0..history_units.len())
            .filter(|&i| history_units[i] == matched)
            .collect();
        let mut allowed: Vec<usize> = deepest.iter().copied().filter(|&i| healthy[i]).collect();
        if allowed.len() < deepest.len() && !allowed.contains(&first_healthy) {
            allowed.push(first_healthy);
        }
        return (Branch::CacheHit, allowed);
    }
    (Branch::LowMatch, least_loaded())
}

/// The least-loaded candidates of `(index, load)` pairs, in input order.
pub fn min_load_ties(loads: &[(usize, usize)]) -> Vec<usize> {
    let Some(min) = loads.iter().map(|&(_, load)| load).min() else {
        return Vec::new();
    };
    loads
        .iter()
        .filter(|&&(_, load)| load == min)
        .map(|&(idx, _)| idx)
        .collect()
}

#[derive(Default)]
struct Stats {
    branch: [AtomicU64; 5],
    tied_picks: AtomicU64,
    evictions: AtomicU64,
}

/// Background eviction thread (gateway `utils::PeriodicTask`: sleeps the
/// interval in 100 ms steps, then runs; stops on drop).
struct PeriodicTask {
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl PeriodicTask {
    fn spawn<F>(interval_secs: u64, task: F) -> Self
    where
        F: Fn() + Send + 'static,
    {
        let shutdown = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&shutdown);
        let handle = thread::Builder::new()
            .name("gateway-ca-eviction".into())
            .spawn(move || {
                let check_interval_ms = 100u64;
                let total_sleep_ms = interval_secs * 1000;
                loop {
                    let mut slept_ms = 0u64;
                    while slept_ms < total_sleep_ms {
                        if flag.load(Ordering::Relaxed) {
                            return;
                        }
                        thread::sleep(Duration::from_millis(check_interval_ms));
                        slept_ms += check_interval_ms;
                    }
                    if flag.load(Ordering::Relaxed) {
                        return;
                    }
                    task();
                }
            })
            .expect("spawn the gateway_cache_aware eviction thread");
        Self {
            shutdown,
            handle: Some(handle),
        }
    }
}

impl Drop for PeriodicTask {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl std::fmt::Debug for PeriodicTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PeriodicTask")
    }
}

struct Shared {
    config: GatewayCacheAwareConfig,
    text: Arc<Tree>,
    tokens: Arc<Tree>,
    stats: Stats,
}

impl Shared {
    fn tree(&self, namespace: CacheKeyNamespace) -> &Arc<Tree> {
        match namespace {
            CacheKeyNamespace::Text => &self.text,
            CacheKeyNamespace::Tokens => &self.tokens,
        }
    }

    /// Gateway eviction task body: each tree, `evict_tenant_by_size`.
    fn evict(&self, reason: &str) {
        let started = Instant::now();
        let before = sorted_counts(&self.tokens);
        self.text.evict_tenant_by_size(self.config.max_tree_size);
        self.tokens.evict_tenant_by_size(self.config.max_tree_size);
        self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        let after = sorted_counts(&self.tokens);
        let eviction_ms = started.elapsed().as_secs_f64() * 1e3;
        let s = &self.stats;
        let branches: serde_json::Map<String, Value> = Branch::ALL
            .iter()
            .map(|b| {
                (
                    b.as_str().to_string(),
                    json!(s.branch[b.index()].load(Ordering::Relaxed)),
                )
            })
            .collect();
        let before = json!(before);
        let after = json!(after);
        let branches = Value::Object(branches);
        tracing::info!(
            target: LOG_TARGET,
            reason,
            max_tree_size = self.config.max_tree_size,
            token_tree_chars_before = %before,
            token_tree_chars_after = %after,
            eviction_ms,
            evictions = s.evictions.load(Ordering::Relaxed),
            branches = %branches,
            tied_picks = s.tied_picks.load(Ordering::Relaxed),
            "GATEWAY_CA_STATS",
        );
    }
}

fn sorted_counts(tree: &Tree) -> Vec<(String, usize)> {
    let mut counts: Vec<(String, usize)> = tree.get_tenant_char_count().into_iter().collect();
    counts.sort();
    counts
}

/// The ported gateway cache-aware prefill policy.
pub struct GatewayCacheAwarePolicy {
    shared: Arc<Shared>,
    /// Worker URLs already inserted ("" in both trees, gateway
    /// `init_workers`).
    registered: Mutex<HashSet<String>>,
    _eviction: Option<PeriodicTask>,
}

impl std::fmt::Debug for GatewayCacheAwarePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatewayCacheAwarePolicy")
            .field("config", &self.shared.config)
            .finish_non_exhaustive()
    }
}

impl GatewayCacheAwarePolicy {
    pub fn new(config: GatewayCacheAwareConfig) -> Self {
        Self::with_options(config, Arc::new(AtomicU64::new(0)), false)
    }

    /// `epoch` feeds both trees; `deterministic_ties` replaces the
    /// arbitrary tree choices with the smallest tenant (parity tests only).
    pub fn with_options(
        config: GatewayCacheAwareConfig,
        epoch: EpochCounter,
        deterministic_ties: bool,
    ) -> Self {
        let shared = Arc::new(Shared {
            text: Arc::new(Tree::with_options(Arc::clone(&epoch), deterministic_ties)),
            tokens: Arc::new(Tree::with_options(epoch, deterministic_ties)),
            stats: Stats::default(),
            config,
        });
        let eviction = (shared.config.eviction_interval_secs > 0).then(|| {
            let task_shared = Arc::clone(&shared);
            PeriodicTask::spawn(shared.config.eviction_interval_secs, move || {
                task_shared.evict("interval")
            })
        });
        Self {
            shared,
            registered: Mutex::new(HashSet::new()),
            _eviction: eviction,
        }
    }

    pub fn config(&self) -> &GatewayCacheAwareConfig {
        &self.shared.config
    }

    /// Gateway `init_workers`: per worker, insert "" into the text tree,
    /// then the token tree. Workers already inserted are skipped.
    pub fn init_workers<'u>(&self, urls: impl IntoIterator<Item = &'u str>) {
        let mut registered = self.registered.lock();
        for url in urls {
            if registered.insert(url.to_string()) {
                self.shared.text.insert("", url);
                self.shared.tokens.insert("", url);
            }
        }
    }

    fn ensure_registered<W: CandidateView>(&self, workers: &[W]) {
        {
            let registered = self.registered.lock();
            if workers.iter().all(|w| registered.contains(w.url())) {
                return;
            }
        }
        self.init_workers(workers.iter().map(|w| w.url()));
    }

    /// Run one eviction pass now (gateway `evict_cache` with the configured
    /// size).
    pub fn evict_now(&self) {
        self.shared.evict("explicit");
    }

    /// Tree characters per worker (token tree or text tree).
    pub fn tenant_char_counts(&self, namespace: CacheKeyNamespace) -> Vec<(String, usize)> {
        sorted_counts(self.shared.tree(namespace))
    }

    /// Per-candidate matched units of `key` without touching the LRU state
    /// (gateway `observe_request_history`).
    pub fn observe_history<W: CandidateView>(
        &self,
        workers: &[W],
        key: &CacheLookupKey<'_>,
    ) -> Vec<usize> {
        let tree = self.shared.tree(key.namespace);
        workers
            .iter()
            .map(|w| {
                key.matched_unit_count(
                    tree.prefix_match_tenant_chars_read_only(key.tree_text(), w.url()),
                )
            })
            .collect()
    }

    /// Gateway `CacheAwarePolicy::select_worker` over `workers`, with the
    /// least-load ties broken by `tie`.
    pub fn select_with<W: CandidateView>(
        &self,
        workers: &[W],
        key: &CacheLookupKey<'_>,
        tie: &mut dyn TieBreak,
    ) -> Decision {
        let config = &self.shared.config;
        let mut decision = Decision::new(key);
        let healthy_indices: Vec<usize> = workers
            .iter()
            .enumerate()
            .filter(|(_, w)| w.is_healthy())
            .map(|(idx, _)| idx)
            .collect();
        if healthy_indices.is_empty() {
            return decision;
        }
        let tree = self.shared.tree(key.namespace);

        // Get current load statistics - compute min/max in single pass without allocation
        let (min_load, max_load) = workers.iter().fold((usize::MAX, 0usize), |(min, max), w| {
            let load = w.load();
            (min.min(load), max.max(load))
        });
        let min_load = if min_load == usize::MAX { 0 } else { min_load };
        decision.min_load = min_load;
        decision.max_load = max_load;

        if is_imbalanced(min_load, max_load, config) {
            // Gateway select_worker_min_load: snapshot load(), shortest
            // queue, random tie, and update the tree even when imbalanced.
            let loads: Vec<(usize, usize)> = healthy_indices
                .iter()
                .map(|&idx| (idx, workers[idx].load()))
                .collect();
            let tied = min_load_ties(&loads);
            decision.branch = Branch::Imbalanced;
            decision.tied = tied.len();
            let Some(idx) = tie.pick(&tied) else {
                return decision;
            };
            if let Some(text) = key.tree_text_for_update() {
                tree.insert(text, workers[idx].url());
            }
            decision.chosen = Some(idx);
            return decision;
        }

        // Use cache-aware routing when balanced
        let text = key.tree_text();
        let result = tree.prefix_match_with_counts(text);
        let match_rate = key.match_rate(result.matched_char_count);
        decision.matched_units = Some(key.matched_unit_count(result.matched_char_count));
        decision.match_rate = Some(match_rate);
        decision.matched_tenant = Some(result.tenant.to_string());
        let hit = match_rate > config.cache_threshold;

        let selected_idx = if hit {
            let tenant_url: &str = &result.tenant;
            decision.branch = Branch::CacheHit;
            workers
                .iter()
                .position(|w| w.url() == tenant_url)
                .filter(|&idx| workers[idx].is_healthy())
        } else {
            let loads: Vec<(usize, usize)> = healthy_indices
                .iter()
                .map(|&idx| (idx, workers[idx].load()))
                .collect();
            let tied = min_load_ties(&loads);
            decision.branch = Branch::LowMatch;
            decision.tied = tied.len();
            tie.pick(&tied)
        };

        if let Some(idx) = selected_idx {
            tree.insert(text, workers[idx].url());
            decision.chosen = Some(idx);
            return decision;
        }

        // Selected worker no longer exists or unhealthy, remove stale tenant from tree
        if hit {
            tree.remove_tenant(&result.tenant);
            tracing::debug!(
                target: LOG_TARGET,
                tenant = %result.tenant,
                "removed a stale worker from the cache tree"
            );
        }
        decision.branch = Branch::StaleTenant;
        decision.chosen = healthy_indices.first().copied();
        decision
    }

    fn record(&self, decision: &Decision) {
        let s = &self.shared.stats;
        s.branch[decision.branch.index()].fetch_add(1, Ordering::Relaxed);
        if decision.tied > 1 {
            s.tied_picks.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The selection of [`Policy::select_noted`] for a request known by its
    /// token ids (`--policy symmetric_pd --symmetric-pd-p-select gateway-cache-aware`):
    /// the same registration, key, rule, random tie and statistics. `None`
    /// tokens route as an empty text prompt. Returns the chosen index into
    /// `workers` and the decision note.
    pub fn select_for_tokens(
        &self,
        workers: &[Arc<Worker>],
        tokens: Option<&[u32]>,
        cache_salt: Option<&str>,
    ) -> (Option<usize>, Value) {
        if workers.is_empty() {
            return (None, Value::Null);
        }
        self.ensure_registered(workers);
        let tokens = tokens.filter(|ids| !ids.is_empty());
        let key = CacheLookupKey::new(tokens, None, cache_salt.filter(|_| tokens.is_some()));
        let decision = self.select_with(workers, &key, &mut RandomTie);
        self.record(&decision);
        (decision.chosen, decision.note())
    }
}

/// The request's text prompt for the text tree (see the module docs).
fn text_prompt(value: &Value) -> Option<&str> {
    match value.get("prompt") {
        Some(Value::String(text)) => Some(text.as_str()),
        Some(Value::Array(items)) => items.first().and_then(Value::as_str),
        _ => value.get("text").and_then(Value::as_str),
    }
}

/// The gateway salts a text prompt only on its extended completions route.
fn text_route_uses_salt(value: &Value) -> bool {
    value.get("forced_output_token_ids").is_some() || value.get("return_token_ids").is_some()
}

impl Policy for GatewayCacheAwarePolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        self.select_noted(workers, ctx).0
    }

    fn select_noted(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> (Option<Arc<Worker>>, Option<Value>) {
        if workers.is_empty() {
            return (None, None);
        }
        self.ensure_registered(workers);
        let tokens = ctx.request_tokens().filter(|ids| !ids.is_empty());
        let parsed: Option<Value> = match tokens {
            Some(_) => None,
            None => ctx
                .request_body()
                .and_then(|body| serde_json::from_slice(body).ok()),
        };
        let (text, salt) = match parsed.as_ref() {
            Some(value) => (
                text_prompt(value),
                ctx.cache_salt().filter(|_| text_route_uses_salt(value)),
            ),
            None => (None, ctx.cache_salt()),
        };
        let key = CacheLookupKey::new(tokens, text, salt);
        let decision = self.select_with(workers, &key, &mut RandomTie);
        self.record(&decision);
        let chosen = decision.chosen.map(|idx| Arc::clone(&workers[idx]));
        (chosen, Some(decision.note()))
    }

    /// The completions route parses the body (token ids) for this policy.
    fn needs_request_tokens(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// A candidate with a settable load and health.
    struct Fake {
        url: String,
        load: Cell<usize>,
        healthy: Cell<bool>,
    }

    impl Fake {
        fn new(url: &str) -> Self {
            Self {
                url: url.into(),
                load: Cell::new(0),
                healthy: Cell::new(true),
            }
        }
    }

    impl CandidateView for Fake {
        fn url(&self) -> &str {
            &self.url
        }
        fn load(&self) -> usize {
            self.load.get()
        }
        fn is_healthy(&self) -> bool {
            self.healthy.get()
        }
    }

    fn cfg(threshold: f32, abs: usize, rel: f32, max_tree_size: usize) -> GatewayCacheAwareConfig {
        GatewayCacheAwareConfig {
            cache_threshold: threshold,
            balance_abs_threshold: abs,
            balance_rel_threshold: rel,
            eviction_interval_secs: 0,
            max_tree_size,
            ..GatewayCacheAwareConfig::default()
        }
    }

    fn policy(config: GatewayCacheAwareConfig) -> GatewayCacheAwarePolicy {
        GatewayCacheAwarePolicy::new(config)
    }

    fn pick(p: &GatewayCacheAwarePolicy, w: &[Fake], key: &CacheLookupKey<'_>) -> usize {
        p.ensure_registered(w);
        p.select_with(w, key, &mut RandomTie).chosen.unwrap()
    }

    fn text(t: &str) -> CacheLookupKey<'_> {
        CacheLookupKey::new(None, Some(t), None)
    }

    #[test]
    fn defaults_are_the_gateway_cli_defaults() {
        let d = GatewayCacheAwareConfig::default();
        assert_eq!(d.cache_threshold, 0.3);
        assert_eq!(d.balance_abs_threshold, 64);
        assert_eq!(d.balance_rel_threshold, 1.5);
        assert_eq!(d.eviction_interval_secs, 120);
        assert_eq!(d.max_tree_size, 67_108_864);
    }

    #[test]
    fn token_key_is_eight_hex_chars_per_token_after_the_salt_segment() {
        let key = CacheLookupKey::new(Some(&[1, 0xabcdef12]), None, None);
        assert_eq!(key.tree_text(), "00000001abcdef12");
        assert_eq!(key.namespace(), CacheKeyNamespace::Tokens);
        let salted = CacheLookupKey::new(Some(&[1]), None, Some("s"));
        assert_eq!(salted.salt_prefix_chars(), SALT_SEG_CHARS);
        assert!(salted.tree_text().starts_with('\0'));
        assert!(salted.tree_text().ends_with("00000001"));
        // sha256("s") = 043a718774c572bd8a25adbeb1bfcd5c0256ae11cecf9f9c3f925d0e52beaf89
        assert_eq!(&salted.tree_text()[1..SALT_SEG_CHARS], "043a718774c572bd");
        // An empty salt is no salt.
        assert_eq!(
            CacheLookupKey::new(Some(&[1]), None, Some("")).salt_prefix_chars(),
            0
        );
    }

    // Gateway test `observer_records_history_before_the_real_selection_inserts`.
    #[test]
    fn observer_records_history_before_the_real_selection_inserts() {
        let p = policy(cfg(0.5, 32, 1.1, 10000));
        let w = vec![Fake::new("http://prefill:8000")];
        let tokens = [1, 2, 3];
        let key = CacheLookupKey::new(Some(&tokens), None, Some("session-a"));
        p.ensure_registered(&w);
        assert_eq!(p.observe_history(&w, &key), vec![0]);
        assert_eq!(pick(&p, &w, &key), 0);
        assert_eq!(p.observe_history(&w, &key), vec![3]);
        let other = CacheLookupKey::new(Some(&tokens), None, Some("session-b"));
        assert_eq!(p.observe_history(&w, &other), vec![0]);
    }

    // Gateway test `token_cache_matches_only_complete_token_boundaries`.
    #[test]
    fn token_cache_matches_only_complete_token_boundaries() {
        let stored = CacheLookupKey::new(Some(&[1, 2]), None, None);
        let tree = Tree::new();
        tree.insert(stored.tree_text(), "worker");
        let different = CacheLookupKey::new(Some(&[3, 2]), None, None);
        let result = tree.prefix_match_with_counts(different.tree_text());
        assert_eq!(different.matched_unit_count(result.matched_char_count), 0);
        let common = CacheLookupKey::new(Some(&[1, 9]), None, None);
        let result = tree.prefix_match_with_counts(common.tree_text());
        // 0x00000009 shares 7 characters with 0x00000002: one complete token.
        assert_eq!(result.matched_char_count, 15);
        assert_eq!(common.matched_unit_count(result.matched_char_count), 1);
    }

    // Gateway test `cache_salt_isolates_identical_text`.
    #[test]
    fn cache_salt_isolates_identical_text() {
        let key_a = CacheLookupKey::new(None, Some("shared prefix payload"), Some("salt-a"));
        assert_eq!(key_a.salt_prefix_chars(), SALT_SEG_CHARS);
        let tree = Tree::new();
        tree.insert(key_a.tree_text(), "worker-a");
        let result = tree.prefix_match_with_counts(key_a.tree_text());
        assert_eq!(
            key_a.matched_unit_count(result.matched_char_count),
            key_a.input_unit_count()
        );
        assert!(key_a.match_rate(result.matched_char_count) > 0.99);
        let key_b = CacheLookupKey::new(None, Some("shared prefix payload"), Some("salt-b"));
        let result = tree.prefix_match_with_counts(key_b.tree_text());
        assert_eq!(key_b.matched_unit_count(result.matched_char_count), 0);
        let plain = CacheLookupKey::new(None, Some("shared prefix payload"), None);
        let result = tree.prefix_match_with_counts(plain.tree_text());
        assert_eq!(plain.matched_unit_count(result.matched_char_count), 0);
    }

    // Gateway test `absent_salt_keeps_borrowed_text_and_legacy_accounting`.
    #[test]
    fn absent_salt_keeps_borrowed_text_and_legacy_accounting() {
        let key = text("plain");
        assert!(matches!(key.tree_text, Cow::Borrowed(_)));
        assert_eq!(key.salt_prefix_chars(), 0);
        assert_eq!(key.matched_unit_count(5), 5);
        let empty = CacheLookupKey::new(None, Some(""), Some("salt"));
        assert_eq!(empty.tree_text(), "");
        assert_eq!(empty.salt_prefix_chars(), 0);
    }

    // Gateway test `test_cache_aware_with_balanced_load` (CacheAwareConfig::default()).
    #[test]
    fn balanced_load_follows_the_cache() {
        let p = policy(cfg(0.5, 32, 1.1, 10000));
        let w = vec![Fake::new("http://w1:8000"), Fake::new("http://w2:8000")];
        let first = pick(&p, &w, &text("hello world"));
        assert_eq!(pick(&p, &w, &text("hello world")), first);
        assert_eq!(pick(&p, &w, &text("hello")), first);
    }

    // Gateway test `test_cache_aware_with_imbalanced_load`.
    #[test]
    fn imbalanced_load_takes_the_least_loaded() {
        let p = policy(cfg(0.5, 5, 2.0, 10000));
        let w = vec![Fake::new("http://w1:8000"), Fake::new("http://w2:8000")];
        w[0].load.set(20);
        for _ in 0..5 {
            assert_eq!(pick(&p, &w, &text("test")), 1);
        }
    }

    // Gateway test `test_cache_aware_imbalanced_random_tie_break`.
    #[test]
    fn imbalanced_ties_spread() {
        let p = policy(cfg(0.5, 5, 2.0, 10000));
        let w: Vec<Fake> = (0..5)
            .map(|j| Fake::new(&format!("http://w{j}:8000")))
            .collect();
        w[0].load.set(50);
        let mut counts = [0; 5];
        for _ in 0..100 {
            counts[pick(&p, &w, &text("same_prompt"))] += 1;
        }
        assert_eq!(counts[0], 0, "{counts:?}");
        assert!(counts[1..].iter().all(|&c| c > 0), "{counts:?}");
    }

    // Gateway test `test_cache_aware_random_tie_break_cold_start`.
    #[test]
    fn cold_start_misses_spread() {
        let p = policy(cfg(0.5, 32, 1.1, 10000));
        let w: Vec<Fake> = (0..5)
            .map(|j| Fake::new(&format!("http://w{j}:8000")))
            .collect();
        let mut counts = [0; 5];
        for i in 0..100 {
            let prompt = format!("{i}_unique_request");
            counts[pick(&p, &w, &text(&prompt))] += 1;
        }
        assert!(counts.iter().all(|&c| c > 0), "{counts:?}");
    }

    // Gateway test `test_cache_aware_worker_removal`: a matched tenant that
    // is no longer healthy is removed and the first healthy worker serves.
    #[test]
    fn stale_tenant_is_removed_and_first_healthy_serves() {
        let p = policy(cfg(0.5, 32, 1.1, 10000));
        let w = vec![Fake::new("http://w1:8000"), Fake::new("http://w2:8000")];
        let a = pick(&p, &w, &text("test1"));
        w[a].healthy.set(false);
        p.ensure_registered(&w);
        let d = p.select_with(&w, &text("test1"), &mut RandomTie);
        assert_eq!(d.branch, Branch::StaleTenant);
        assert_eq!(d.chosen, Some(1 - a));
        // The tenant is gone from the tree.
        assert_eq!(
            p.observe_history(&w, &text("test1"))[a],
            0,
            "stale tenant still matches"
        );
    }

    #[test]
    fn threshold_is_strictly_greater() {
        // 3 of 10 tokens match: rate 0.3 is not > 0.3.
        let p = policy(cfg(0.3, 64, 1.5, 1 << 26));
        let w = vec![Fake::new("http://a"), Fake::new("http://b")];
        let seed: Vec<u32> = (0..3).collect();
        p.ensure_registered(&w);
        let d = p.select_with(
            &w,
            &CacheLookupKey::new(Some(&seed), None, None),
            &mut RandomTie,
        );
        let owner = d.chosen.unwrap();
        w[owner].load.set(1);
        let probe: Vec<u32> = (0..3).chain(100..107).collect();
        let d = p.select_with(
            &w,
            &CacheLookupKey::new(Some(&probe), None, None),
            &mut RandomTie,
        );
        assert_eq!(d.matched_units, Some(3));
        assert_eq!(d.branch, Branch::LowMatch);
        assert_eq!(d.chosen, Some(1 - owner));
    }

    #[test]
    fn decide_from_history_matches_select_with() {
        // The replay rule and the live rule agree on the same inputs.
        let c = cfg(0.3, 64, 1.5, 1 << 26);
        assert_eq!(
            decide_from_history(&[70, 0, 3], &[true; 3], &[50, 0, 0], 50, &c),
            (Branch::Imbalanced, vec![1])
        );
        assert_eq!(
            decide_from_history(&[5, 0, 3], &[true; 3], &[50, 50, 0], 60, &c),
            (Branch::CacheHit, vec![0, 1])
        );
        assert_eq!(
            decide_from_history(&[5, 0, 0], &[true; 3], &[18, 0, 0], 60, &c),
            (Branch::LowMatch, vec![1, 2])
        );
        assert_eq!(
            decide_from_history(&[0, 0], &[false, true], &[9, 0], 10, &c),
            (Branch::CacheHit, vec![1])
        );
    }

    #[test]
    fn imbalance_needs_both_thresholds() {
        let c = cfg(0.3, 64, 1.5, 1);
        assert!(!is_imbalanced(100, 164, &c)); // spread 64 is not > 64
        assert!(is_imbalanced(100, 165, &c)); // 165 > 150
        assert!(!is_imbalanced(200, 290, &c)); // 290 is not > 300
        assert!(is_imbalanced(0, 65, &c));
    }

    #[test]
    fn eviction_keeps_each_worker_at_the_size_cap() {
        let p = policy(cfg(0.3, 64, 1.5, 80));
        let w = vec![Fake::new("http://a"), Fake::new("http://b")];
        for s in 0..20u32 {
            let ids: Vec<u32> = (0..6).map(|t| s * 100 + t).collect();
            pick(&p, &w, &CacheLookupKey::new(Some(&ids), None, None));
        }
        p.evict_now();
        for (_, chars) in p.tenant_char_counts(CacheKeyNamespace::Tokens) {
            assert!(chars <= 80, "{chars}");
        }
    }

    /// Golden vectors from the real gateway: the same generated steps
    /// (`gateway_ca_golden_gen`, byte-identical in both crates) driven
    /// through sgl-model-gateway's `CacheAwarePolicy::select_worker`
    /// (test build, deterministic tree ties) produced
    /// `tests/fixtures/gateway_ca_golden.json`.
    mod golden {
        use super::*;
        use crate::policies::gateway_ca_golden_gen as gen;
        use serde::Deserialize;

        #[derive(Deserialize)]
        struct Golden {
            source: String,
            scenarios: Vec<GoldenScenario>,
        }

        #[derive(Deserialize)]
        struct GoldenScenario {
            name: String,
            /// The gateway's process-wide epoch counter at the scenario
            /// start and end (the port's counter starts at the same value,
            /// so the 1-in-8 LRU refresh hits the same matches).
            epoch_start: u64,
            epoch_end: u64,
            steps: Vec<GoldenStep>,
        }

        #[derive(Deserialize)]
        struct GoldenStep {
            /// Digest of the step's inputs.
            d: String,
            /// Per-candidate history units before the selection.
            #[serde(default)]
            h: Vec<usize>,
            /// The gateway's choice (index into the candidates).
            #[serde(default)]
            c: Option<usize>,
            /// Digest of the tree characters per worker after the step.
            x: String,
            /// Full tree characters (some steps only).
            #[serde(default)]
            tok: Option<Vec<(String, usize)>>,
            #[serde(default)]
            txt: Option<Vec<(String, usize)>>,
        }

        /// The gateway's choice when it is one of the tied candidates.
        struct Recorded(Option<usize>);

        impl TieBreak for Recorded {
            fn pick(&mut self, tied: &[usize]) -> Option<usize> {
                assert!(
                    self.0.is_none_or(|c| tied.contains(&c)),
                    "the gateway chose {:?}, outside the least-loaded set {tied:?}",
                    self.0
                );
                self.0
            }
        }

        fn run(golden: &Golden) -> (usize, usize) {
            assert!(
                golden
                    .source
                    .contains("sgl-model-gateway cache-aware policy"),
                "{}",
                golden.source
            );
            let (mut requests, mut evictions) = (0, 0);
            for spec in gen::scenarios() {
                let sc = golden
                    .scenarios
                    .iter()
                    .find(|g| g.name == spec.name)
                    .unwrap_or_else(|| panic!("golden has no scenario {}", spec.name));
                let config = GatewayCacheAwareConfig {
                    cache_threshold: spec.cache_threshold,
                    balance_abs_threshold: spec.balance_abs_threshold,
                    balance_rel_threshold: spec.balance_rel_threshold,
                    eviction_interval_secs: 0,
                    max_tree_size: spec.max_tree_size,
                    ..GatewayCacheAwareConfig::default()
                };
                let epoch = Arc::new(AtomicU64::new(sc.epoch_start));
                let p = GatewayCacheAwarePolicy::with_options(config, Arc::clone(&epoch), true);
                let urls: Vec<String> = (0..spec.workers).map(gen::worker_url).collect();
                p.init_workers(urls.iter().map(String::as_str));
                let steps = gen::generate(&spec);
                assert_eq!(steps.len(), sc.steps.len(), "{}: step count", spec.name);
                for (i, (step, want)) in steps.iter().zip(&sc.steps).enumerate() {
                    let at = format!("{} step {i}", spec.name);
                    assert_eq!(gen::step_digest(step), want.d, "{at}: inputs differ");
                    match step {
                        gen::GenStep::Evict => {
                            p.evict_now();
                            evictions += 1;
                        }
                        gen::GenStep::Request {
                            tokens,
                            text,
                            salt,
                            candidates,
                            loads,
                            healthy,
                        } => {
                            let w: Vec<Fake> = candidates
                                .iter()
                                .enumerate()
                                .map(|(j, &k)| {
                                    let f = Fake::new(&urls[k]);
                                    f.load.set(loads[j]);
                                    f.healthy.set(healthy[j]);
                                    f
                                })
                                .collect();
                            let key = CacheLookupKey::new(
                                tokens.as_deref(),
                                text.as_deref(),
                                salt.as_deref(),
                            );
                            assert_eq!(p.observe_history(&w, &key), want.h, "{at}: history");
                            let d = p.select_with(&w, &key, &mut Recorded(want.c));
                            assert_eq!(d.chosen, want.c, "{at}: choice ({:?})", d.branch);
                            requests += 1;
                        }
                    }
                    let tok = p.tenant_char_counts(CacheKeyNamespace::Tokens);
                    let txt = p.tenant_char_counts(CacheKeyNamespace::Text);
                    if let Some(want_tok) = &want.tok {
                        assert_eq!(&tok, want_tok, "{at}: token tree");
                    }
                    if let Some(want_txt) = &want.txt {
                        assert_eq!(&txt, want_txt, "{at}: text tree");
                    }
                    assert_eq!(
                        gen::chars_digest(&tok, &txt),
                        want.x,
                        "{at}: tree characters (port: tokens {tok:?}, text {txt:?})"
                    );
                }
                assert_eq!(
                    epoch.load(Ordering::Relaxed),
                    sc.epoch_end,
                    "{}: epochs drawn",
                    spec.name
                );
            }
            (requests, evictions)
        }

        /// Every step: the same inputs, the same per-candidate history
        /// before the step, the same choice (a random tie pick of the
        /// gateway must be in this port's least-loaded set and is then
        /// used), and the same tree characters per worker after the step.
        #[test]
        fn port_reproduces_the_gateway_golden_vectors() {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/gateway_ca_golden.json");
            let golden: Golden = serde_json::from_slice(&std::fs::read(path).expect("read golden"))
                .expect("parse golden");
            let (requests, evictions) = run(&golden);
            assert_eq!(requests, 4800);
            assert!(evictions > 100, "{evictions}");
        }
    }
}
