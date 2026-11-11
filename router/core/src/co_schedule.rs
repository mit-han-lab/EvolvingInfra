// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Co-scheduled request routing and KV movement (`--policy co_schedule`).
//!
//! For every request the scheduler co-decides WHERE it runs and WHETHER its
//! KV prefix should move there first:
//!
//! 1. `A` = the routable worker with the highest cached-prefix hit for this
//!    request (per-worker hits come from the standalone dynamo indexer, the
//!    same salt-aware answer the kv-pull-hint path uses).
//! 2. If `backlog(A) < max_uncached_tokens`, dispatch to `A`.
//! 3. Otherwise, if ANY routable worker is under the cap, dispatch to the
//!    one with the highest OWN hit among them (ties: lowest backlog) — the
//!    KV moves to it via the existing `kv_pull_hint` mechanism.
//! 4. Otherwise the request WAITS in a router-side queue — unless the pool
//!    is badly skewed: when every worker is over the cap but the backlog
//!    spread (max − min) exceeds `imbalance_spread_tokens`, the queue HEAD
//!    is dispatched to the minimum-backlog worker anyway
//!    ([`Branch::Imbalance`], hint through the normal pull gate). This is
//!    the imbalance escape: it trades a cap violation on the least-loaded
//!    worker against letting one hot worker stall the whole queue.
//!
//! **Backlog** (积压) of a worker = its pending uncached prefill tokens:
//! the engine-reported `/v1/loads` `num_waiting_uncached_tokens` as a
//! baseline, plus a router-side ledger of tokens dispatched AFTER that poll
//! and not yet prefill-complete (covers poll staleness and the PD
//! bootstrap-queue blind spot, where just-dispatched requests are invisible
//! to the engine metric).
//!
//! **Queue** semantics: arrival order with cache-affinity reordering — on
//! every capacity change the scheduler dispatches the FIRST queued request
//! whose own `A` is under the cap (zero-transfer); if none qualifies, the
//! HEAD request goes to the best under-cap worker with a transfer; if no
//! worker is under the cap, everything holds (subject to the imbalance
//! escape above). A request queued longer than `max_queue_wait_ms` gets
//! absolute priority and is force-dispatched to the minimum-backlog worker
//! (starvation guard).
//!
//! **Triggers**: a dispatch pass runs on every arrival (inline, on the
//! arriving request's task), on every ledger retire, on every applied
//! `/v1/loads` sample, exactly at the oldest queued entry's starvation
//! deadline, and on a `tick_ms` fallback timer. Wakes coalesce (a
//! [`Notify`] stores one permit); the labels accumulate in a bitset so
//! every dispatch line carries `trigger` (the primary wake source of its
//! pass) and `triggers` (the full coalesced set).
//!
//! **Poll-on-hold**: a pass that ends in HOLD signals the `/v1/loads`
//! poller, which re-polls the pool `poll_on_hold_ms` after the previous
//! round instead of waiting a full `load_poll_ms` period — a freed worker
//! is seen within ~`poll_on_hold_ms` while anything is queued.
//!
//! **Pull rule** is decoupled from routing (uniform): on ANY dispatch to
//! target `X`, if `best_source_hit − hit(X) ≥ min_incremental`, attach the
//! hint. `--kv-pull-scope detour-only` restricts pulls to non-direct
//! dispatches (the strict three-step rule, kept for ablation).
//!
//! Everything is fail-open: indexer unreachable → hits are empty and the
//! policy degrades to min-backlog dispatch; `/v1/loads` stale (>5 s) → the
//! ledger alone is trusted. Both degradations are logged.
//!
//! One structured decision line per dispatch (INFO, target `co_sched`),
//! carrying the branch (`direct|transfer|forced|imbalance`), the trigger
//! labels, `pass_id`, `spread_tokens`, and `poll_age_ms`.
//!
//! **TRADE** (`--co-schedule-trade`, with `--co-schedule-uncached-balance`):
//! on the uncached ledger, each P candidate gets
//!
//! ```text
//! cost(P) = backlog(P) + u(P) + R * max(0, pull_s - backlog(P) / R) + lambda * pulled(P)
//! pull_s  = 0.45 s + 0.10 s per 100k pulled tokens (flags), 0 without a pull
//! ```
//!
//! The pull floor is max(`--kv-pull-hint-min-incremental-tokens`, 512); the
//! CLI rejects a smaller explicit value (absent = 512) and the engine-side
//! local-prefix fallback with TRADE.
//!
//! `backlog(P)` = pending uncached tokens of P from the router ledger with
//! the engine's per-RID `/v1/loads` progress, the same ledger as the
//! owner-threshold path. `--co-schedule-trade-exclude-unlaunched` leaves out
//! requests whose first prefill chunk has not launched (the engine reports a
//! RID only after that launch, so this drops the transit, bootstrap-queue
//! and waiting-queue work; diagnostic only). `pulled(P)` = best source match − P's own match when
//! that is at least `--kv-pull-hint-min-incremental-tokens`, else 0 (no
//! hint; P recomputes). `u(P)` = input − own match − pulled. There is no pull
//! cap: the transfer time and the copy price are charged instead. The request
//! goes to argmin cost; ties go to the cache-aware owner (max own hit), then
//! to the lower backlog. See [`crate::scheduler::co_schedule::trade_cost`].
//!
//! `--co-schedule-trade-wait eft` replaces backlog / R and u / R by the EFT-0+
//! drain forecast B_w and the step-model service S_w
//! ([`crate::scheduler::co_schedule::trade_cost_full`]). The progress poll then
//! also asks for the engine `prefill_routing` snapshot, parses it off
//! the cost-book lock, keeps the newest one per P and strips it from the
//! `engine_load` of the trace. An engine that rejects the section (an older
//! engine) is polled without it; its P then uses the fallback (`no_view`), and the
//! router asks again every 30 s. `--co-schedule-trade-dup-charge all`,
//! `--co-schedule-trade-margin-noise` and `--co-schedule-trade-hol-weight` are
//! the other later options. The last one adds mu * R * a_w * S * (B + S / 2),
//! a light-load estimate of the extra wait that the placement causes for the
//! later arrivals at the P ([`crate::scheduler::co_schedule::trade_hol_tokens`];
//! a_w is counted in the cost book; mu is an empirical weight). With all four
//! at their defaults, selection and trace are those of the first TRADE.
//!
//! `--co-schedule-trade-copy-lambda auto` takes lambda from the router's own
//! measurements at each decision: lambda = clamp(mu * q * R / R_hat, min,
//! max), with q = N_active / n_P * busy (`sat`, default) or 1 + the mean
//! in-flight requests per P (`inflight`), R_hat = computed prefill tokens per
//! busy second; `--co-schedule-trade-rate-tokens-per-s auto` sets R = R_hat
//! ([`crate::scheduler::co_schedule::TradeParams::resolved`],
//! [`crate::scheduler::co_schedule::TradeAutoState`] in the cost book). The
//! routes pass the session identity (`x-kv-session-context`, else
//! `x-agent-context`) to [`CoScheduler::admit_session`], and
//! `POST /kv_sessions/end` ends a session ([`CoScheduler::trade_auto_session_end`]).
//! The fixed `--co-schedule-trade-copy-lambda-fallback` applies for the first
//! tau seconds and when an input is missing.
//!
//! **Reserved placement**: cost balancing and the optional locality-margin
//! policy bypass this cap/queue/poll path. Both retain each request's work
//! until its P operation terminates. Locality placement ranks total hit, then
//! GPU residency, and detours only for a pending-uncached-work saving larger
//! than the declared margin. It has no router admission wait. See `cost_balance`.
//!
//! **Global waiting queue** (`--co-schedule-global-queue sjf|hrrn`, TRADE
//! only): TRADE gets its own router queue,
//! separate from the cap queue above. A P has room while its queued work on
//! the cost-book ledger (the work beyond its running pass) is below the
//! depth (or its progress view is stale: fail-open). A request whose TRADE
//! argmin P has no room waits; each release step
//! ([`crate::cost_balance::CostBook::gwq_step`]) walks the held requests in
//! `sjf` or `hrrn` order and releases the first one whose TRADE argmin (with
//! the held work ahead of it as a virtual backlog) has room, to that P (late
//! P binding). Passes run on arrival, on every applied progress poll, on
//! every terminal cost-book status, on a tick while requests are held, and
//! at the oldest request's max-wait deadline. With the flag off, nothing of
//! this runs.

use std::collections::{HashMap, VecDeque};
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde_json::Value;
use tokio::sync::{oneshot, Notify};

use crate::config::{CoScheduleConfig, KvPullScope};
use crate::discovery::ModelId;
use crate::global_queue::{
    GlobalQueueConfig, GlobalQueueDepth, GlobalQueueHold, GlobalQueueMode, GlobalQueueRoom,
    GwqRelease,
};
use crate::kv_hint::{IndexerAnswer, KvHintConfig, KvHintInjector};
use crate::policies::registry::PdPoolResolver;
use crate::workers::{Worker, WorkerRegistry};

/// Polled `/v1/loads` baselines older than this are distrusted and the
/// ledger alone is used (agreed fail-open for a wedged poller/engine).
const POLL_STALENESS_LIMIT: Duration = Duration::from_secs(5);

/// A ledger entry that was never explicitly retired (errored request,
/// dropped client) stops counting after this long, bounding backlog
/// inflation even when the `/v1/loads` poller is down. A prefill that has
/// not started producing in 30 s is not predictable near-term work anyway.
const LEDGER_ENTRY_TTL: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Branch {
    /// Step 2: the max-hit worker was under the cap.
    Direct,
    /// Step 3: detoured to an under-cap worker.
    Transfer,
    /// Starvation guard fired.
    Forced,
    /// Imbalance escape: every worker was over the cap but the backlog
    /// spread (max − min) exceeded `imbalance_spread_tokens`, so the queue
    /// head went to the min-backlog worker anyway.
    Imbalance,
    /// Minimize post-dispatch input-times-uncached cost spread.
    CostBalanced,
    /// Retain cache locality unless pending-work savings exceed a margin.
    Locality,
    /// Immediately balance remaining uncached work, protecting light owners.
    UncachedBalanced,
}

impl Branch {
    pub fn as_str(self) -> &'static str {
        match self {
            Branch::Direct => "direct",
            Branch::Transfer => "transfer",
            Branch::Forced => "forced",
            Branch::Imbalance => "imbalance",
            Branch::CostBalanced => "cost",
            Branch::Locality => "locality",
            Branch::UncachedBalanced => "uncached",
        }
    }
}

/// Why a dispatch pass ran. Bits accumulate in `CoScheduler::pending`
/// between dispatcher wakes (Notify coalesces permits, so one pass may
/// serve several events).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Triggers(u8);

impl Triggers {
    /// `enqueue` pushed a request (the inline arrival pass uses this too).
    pub const ARRIVAL: Triggers = Triggers(1);
    /// `LedgerHandle::retire` freed ledger tokens.
    pub const RETIRE: Triggers = Triggers(2);
    /// A `/v1/loads` sample was applied (periodic or poll-on-hold).
    pub const POLL: Triggers = Triggers(4);
    /// The `tick_ms` fallback timer (or a spurious permit).
    pub const TICK: Triggers = Triggers(8);
    /// The oldest entry's starvation-deadline timer.
    pub const STARVATION: Triggers = Triggers(16);

    fn is_empty(self) -> bool {
        self.0 == 0
    }

    fn or(self, o: Triggers) -> Triggers {
        Triggers(self.0 | o.0)
    }

    fn contains(self, o: Triggers) -> bool {
        self.0 & o.0 != 0
    }

    /// Most specific member, fixed order used by the `trigger` log field:
    /// starvation > poll > retire > arrival > tick.
    pub fn primary(self) -> Trigger {
        if self.contains(Self::STARVATION) {
            Trigger::Starvation
        } else if self.contains(Self::POLL) {
            Trigger::Poll
        } else if self.contains(Self::RETIRE) {
            Trigger::Retire
        } else if self.contains(Self::ARRIVAL) {
            Trigger::Arrival
        } else {
            Trigger::Tick
        }
    }
}

/// `+`-joined members in priority order (e.g. `poll+arrival`); `tick` when
/// empty.
impl std::fmt::Display for Triggers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const ORDER: [(Triggers, Trigger); 5] = [
            (Triggers::STARVATION, Trigger::Starvation),
            (Triggers::POLL, Trigger::Poll),
            (Triggers::RETIRE, Trigger::Retire),
            (Triggers::ARRIVAL, Trigger::Arrival),
            (Triggers::TICK, Trigger::Tick),
        ];
        if self.is_empty() {
            return f.write_str("tick");
        }
        let mut first = true;
        for (bit, t) in ORDER {
            if self.contains(bit) {
                if !first {
                    f.write_str("+")?;
                }
                f.write_str(t.as_str())?;
                first = false;
            }
        }
        Ok(())
    }
}

/// Primary wake source of the pass that dispatched a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trigger {
    Arrival,
    Retire,
    Poll,
    Tick,
    Starvation,
}

impl Trigger {
    pub fn as_str(self) -> &'static str {
        match self {
            Trigger::Arrival => "arrival",
            Trigger::Retire => "retire",
            Trigger::Poll => "poll",
            Trigger::Tick => "tick",
            Trigger::Starvation => "starvation",
        }
    }
}

/// What one `drain` pass did; returned so tests can assert synchronously.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct PassOutcome {
    pub(crate) dispatched: u32,
    /// Queue depth when the pass ended (0 unless it ended in HOLD/NoPool).
    pub(crate) held: usize,
}

/// Per-pass context stamped on every dispatch line of the pass.
#[derive(Clone, Copy)]
struct PassCtx {
    triggers: Triggers,
    pass_id: u64,
    /// max − min backlog over the pool at the start of this iteration.
    spread_tokens: u64,
}

/// One dispatched request's contribution to its worker's backlog. Created
/// at dispatch, retired when the prefill is known to be done (PD: the
/// prefill-side request resolving; plain: first streamed byte) or when a
/// fresher `/v1/loads` poll subsumes it.
#[derive(Debug)]
struct LedgerEntry {
    tokens: u64,
    dispatched_at: Instant,
    retired: AtomicBool,
}

/// RAII handle the route holds for a dispatched request. `retire()` is
/// idempotent; dropping the last clone retires as a fallback so an errored
/// request can never pin backlog forever.
#[derive(Debug, Clone)]
pub struct LedgerHandle {
    entry: Arc<LedgerEntry>,
    sched: Weak<CoScheduler>,
}

impl LedgerHandle {
    /// Mark the prefill work done and wake the dispatcher (capacity freed).
    pub fn retire(&self) {
        if !self.entry.retired.swap(true, Ordering::Relaxed) {
            if let Some(s) = self.sched.upgrade() {
                s.wake(Triggers::RETIRE);
            }
        }
    }
}

// NOTE: deliberately NO retire-on-drop. Handles are cloned into places
// whose drop timing is wrong for "prefill done" (a streaming handler
// returns before the first body byte), so dropping must be a no-op.
// Missed explicit retires are bounded twice over: the next successful
// `/v1/loads` poll subsumes every entry dispatched before it, and
// [`LEDGER_ENTRY_TTL`] expires stragglers even with the poller down.

/// The dispatch decision handed back to the route.
pub struct Dispatch {
    /// Unique ownership moves into the detached P task in cost mode.
    pub cost_reservation: Option<crate::cost_balance::Reservation>,
    pub worker: Arc<Worker>,
    pub hint: Option<Value>,
    pub ledger: LedgerHandle,
    pub branch: Branch,
    pub queue_wait_ms: u64,
    /// Primary wake source of the pass that dispatched this request
    /// (same value as the `trigger` log field).
    pub trigger: Trigger,
}

#[derive(Debug, thiserror::Error)]
pub enum AdmitError {
    /// The forced-dispatch guard could not fire either (e.g. zero healthy
    /// workers for longer than the wait cap) — surfaced as 503 upstream.
    #[error("co_schedule: no dispatchable worker within the wait budget")]
    Wedged,
}

struct QueueEntry {
    arrived: Instant,
    /// Uncached-token estimate BEFORE any per-worker hit correction.
    tokens_len: u64,
    /// Per-worker cached-prefix hits (url → tokens) at arrival time.
    hits: HashMap<String, u32>,
    /// Populated only for the optional GPU-residency tie-break.
    gpu_hits: HashMap<String, u32>,
    /// The raw indexer answer, kept for the pull-hint pick at dispatch.
    answer: Option<IndexerAnswer>,
    /// Client request id (trie sends `rid` in the body) — logged on the
    /// dispatch line so per-request records join across router and engines.
    rid: Option<String>,
    tx: oneshot::Sender<Dispatch>,
}

#[derive(Default)]
struct WorkerLoadState {
    polled_uncached: u64,
    polled_at: Option<Instant>,
    /// `sent_at` of the newest applied sample; older samples are ignored.
    polled_sent_at: Option<Instant>,
    ledger: Vec<Arc<LedgerEntry>>,
}

impl WorkerLoadState {
    /// Age of the polled baseline in ms for logging; -1 when never polled.
    fn poll_age_ms(&self) -> i64 {
        self.polled_at
            .map_or(-1, |t| t.elapsed().as_millis() as i64)
    }

    fn backlog(&self) -> u64 {
        let base = match self.polled_at {
            Some(t) if t.elapsed() < POLL_STALENESS_LIMIT => self.polled_uncached,
            _ => 0,
        };
        let live: u64 = self
            .ledger
            .iter()
            .filter(|e| Self::entry_live(e))
            .map(|e| e.tokens)
            .sum();
        base + live
    }

    fn entry_live(e: &LedgerEntry) -> bool {
        !e.retired.load(Ordering::Relaxed) && e.dispatched_at.elapsed() < LEDGER_ENTRY_TTL
    }

    fn prune(&mut self) {
        self.ledger.retain(|e| Self::entry_live(e));
    }
}

#[derive(Default)]
struct SchedState {
    queue: VecDeque<QueueEntry>,
    workers: HashMap<String, WorkerLoadState>,
}

pub struct CoScheduler {
    cfg: CoScheduleConfig,
    cost_book: Option<Arc<crate::cost_balance::CostBook>>,
    injector: KvHintInjector,
    registry: Arc<WorkerRegistry>,
    model: ModelId,
    state: Mutex<SchedState>,
    notify: Notify,
    /// Coalesced [`Triggers`] bits for the next dispatcher wake.
    pending: AtomicU8,
    /// Monotonic pass counter (`pass_id` on dispatch lines).
    pass_counter: AtomicU64,
    /// Signalled by a HOLD pass; the poller re-polls after `poll_on_hold_ms`.
    /// `Arc` so the poller can wait on it without holding the scheduler.
    poll_now: Arc<Notify>,
    /// Count of HOLD passes that signalled `poll_now` (tests/metrics).
    poll_now_signals: AtomicU64,
    poll_client: reqwest::Client,
    progress_refresh: Arc<crate::load_refresh::LoadRefresh>,
    /// TRADE `wait = eft`: the newest parsed `prefill_routing` snapshot per P.
    routing: parking_lot::RwLock<HashMap<String, Arc<crate::cost_balance::RoutingSnapshot>>>,
    /// TRADE `wait = eft`: P workers whose engine rejected the
    /// `prefill_routing` section, and when. They are polled without it until
    /// [`ROUTING_REPROBE`] has passed.
    routing_rejections: Mutex<HashMap<String, Instant>>,
    /// TRADE `wait = e3`: the online fits, receipt lines, routed requests
    /// and host rule ([`crate::trade_e3::E3Online`]).
    e3_online: Option<Arc<crate::trade_e3::E3Online>>,
    /// TRADE `wait = e3`: where the shadow and label lines go.
    e3_shadow: Option<Arc<crate::trade_e3::ShadowSink>>,
    /// TRADE `wait = e3` with `--co-schedule-trade-e3-push-port`: the state
    /// push subscribers and the newest pushed state per P
    /// ([`crate::trade_e3_push::PushHub`]).
    e3_push: Option<Arc<crate::trade_e3_push::PushHub>>,
    /// `--co-schedule-global-queue sjf|hrrn` (TRADE only); None = off.
    gwq: Option<Gwq>,
}

/// The pass period of the global waiting queue while nothing is held.
const GWQ_IDLE_TICK: Duration = Duration::from_secs(1);

/// The global waiting queue (`--co-schedule-global-queue`): the held
/// requests, the settings and the counters.
struct Gwq {
    cfg: GlobalQueueConfig,
    /// The depth of every P in tokens.
    depth_tokens: u64,
    /// `room = predicted`: the lead in seconds (depth / R).
    lead_s: f64,
    /// A progress view older than this is stale (fail-open).
    stale_after: Duration,
    max_wait: Duration,
    /// The pass period while requests are held (the progress poll
    /// interval, 10 ms to 1 s).
    tick: Duration,
    state: Mutex<GwqState>,
}

impl Gwq {
    fn new(cfg: GlobalQueueConfig, rate_tokens_per_s: f64, load_poll_ms: u64) -> Self {
        Self {
            cfg,
            depth_tokens: cfg.depth_tokens(rate_tokens_per_s, load_poll_ms),
            lead_s: crate::global_queue::lead_seconds(
                cfg.depth_tokens(rate_tokens_per_s, load_poll_ms),
                rate_tokens_per_s,
            ),
            stale_after: crate::global_queue::stale_after(load_poll_ms),
            max_wait: Duration::from_millis(cfg.max_wait_ms),
            tick: Duration::from_millis(load_poll_ms.clamp(10, 1000)),
            state: Mutex::new(GwqState::default()),
        }
    }
}

#[derive(Default)]
struct GwqState {
    /// The held requests in arrival order.
    entries: Vec<GwqEntry>,
    next_id: u64,
    counters: GwqCounters,
    /// The uncached tokens of the held requests on their plain TRADE choice
    /// at the last step (0 when the queue is empty).
    last_queue_tokens: u128,
}

/// One held request.
struct GwqEntry {
    id: u64,
    arrived: Instant,
    arrived_unix_ms: u64,
    tokens_len: u64,
    answer: Option<IndexerAnswer>,
    rid: Option<String>,
    /// Its session key (copy price `auto`).
    session: Option<u64>,
    /// Its `wait = e3` session key (host rule of the target).
    e3_session: Option<String>,
    /// Its TRADE candidates for the pool `cands_pool` (rebuilt when the pool
    /// changes).
    cands: Vec<crate::cost_balance::Candidate>,
    cands_pool: Vec<String>,
    /// The plain TRADE choice at the first step that saw it.
    p_arrival: Option<String>,
    /// The pass that its arrival started.
    arrival_pass: u64,
    tx: oneshot::Sender<Dispatch>,
}

/// Cumulative counts of the global waiting queue (the `gwq_stats` event).
#[derive(Debug, Default, Clone, Copy)]
struct GwqCounters {
    arrivals: u64,
    /// Released by the pass of their own arrival (`held = false`).
    released_at_arrival: u64,
    /// Released by a later pass (`held = true`).
    released_after_hold: u64,
    released_room: u64,
    /// Fail-open: the P had no fresh progress view.
    released_stale_view: u64,
    /// The max wait released them without room.
    released_starved: u64,
    /// The P at release differs from the plain choice at arrival.
    late_bound: u64,
    dropped_client_gone: u64,
    dropped_wait_budget: u64,
    dropped_reserve_error: u64,
    dropped_admission_stopped: u64,
    /// Arrivals refused at once (as the TRADE path refuses them).
    refused_pool: u64,
    refused_rid: u64,
    passes: u64,
    /// Passes that held requests without a healthy PD prefill pool.
    passes_pool_unusable: u64,
    /// Released, but the client had left (the reservation rolled back).
    send_failed: u64,
    /// `--co-schedule-on-demand-load-refresh`: refreshes while holding.
    on_demand_refreshes: u64,
    /// `hold = free`: arrivals bound to their cache owner, dispatched at
    /// once by the TRADE path (never held; not in `arrivals`).
    bypassed_bound: u64,
    /// `room = predicted`: releases on a predicted drain within the lead.
    released_predicted: u64,
}

impl GwqCounters {
    fn value(&self) -> Value {
        serde_json::json!({
            "arrivals": self.arrivals,
            "released_at_arrival": self.released_at_arrival,
            "released_after_hold": self.released_after_hold,
            "released_room": self.released_room,
            "released_stale_view": self.released_stale_view,
            "released_starved": self.released_starved,
            "late_bound": self.late_bound,
            "dropped_client_gone": self.dropped_client_gone,
            "dropped_wait_budget": self.dropped_wait_budget,
            "dropped_reserve_error": self.dropped_reserve_error,
            "dropped_admission_stopped": self.dropped_admission_stopped,
            "refused_pool": self.refused_pool,
            "refused_rid": self.refused_rid,
            "passes": self.passes,
            "passes_pool_unusable": self.passes_pool_unusable,
            "send_failed": self.send_failed,
            "on_demand_refreshes": self.on_demand_refreshes,
            "bypassed_bound": self.bypassed_bound,
            "released_predicted": self.released_predicted,
        })
    }
}

/// What one release pass did (tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct GwqPassOutcome {
    pub(crate) released: u32,
    /// Held requests when the pass ended.
    pub(crate) held: usize,
}

/// Takes a held request out of the queue when its admission future is
/// dropped (the client left) before a release.
struct GwqGuard {
    sched: Weak<CoScheduler>,
    id: u64,
    armed: bool,
}

impl Drop for GwqGuard {
    fn drop(&mut self) {
        if self.armed {
            if let Some(sched) = self.sched.upgrade() {
                sched.gwq_drop(self.id, "client_gone");
            }
        }
    }
}

/// How often the pool's push subscribers are checked (started for new P
/// workers, stopped for gone ones) and the push stats are due.
const PUSH_SUPERVISE: Duration = Duration::from_secs(1);

/// How long a P whose engine rejected `include=prefill_routing` is polled
/// without the section before the router asks for it again.
const ROUTING_REPROBE: Duration = Duration::from_secs(30);

impl std::fmt::Debug for CoScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = self.state.lock();
        f.debug_struct("CoScheduler")
            .field("model", &self.model)
            .field("queue_depth", &st.queue.len())
            .field("max_uncached_tokens", &self.cfg.max_uncached_tokens)
            .finish()
    }
}

impl CoScheduler {
    /// Build the scheduler and, when a Tokio runtime is available, spawn
    /// the dispatcher and `/v1/loads` poller tasks. Without a runtime
    /// (plain-sync tests) the scheduler still works: `admit` drains
    /// inline, only the starvation timer and poller need the tasks.
    ///
    /// `kv` is the same indexer/worker-map config the pull-hint feature
    /// uses — `Cli::into_config` guarantees it is present whenever
    /// `--policy co_schedule` is selected.
    pub fn spawn(
        cfg: CoScheduleConfig,
        kv: &KvHintConfig,
        registry: Arc<WorkerRegistry>,
        model: ModelId,
    ) -> Arc<Self> {
        let injector = KvHintInjector::new(kv);
        if let Some(trade) = cfg.trade {
            // Empty with hol_weight 0: the line is then the same as before.
            let hol = if trade.hol_weight > 0.0 {
                format!(
                    " + mu*R*a_w*S*(B + S/2) (HOL: extra wait of the later arrivals at P; \
                     mu = {}, a_w = TRADE placements per s on P over the last {} s)",
                    trade.hol_weight, trade.hol_rate_window_s,
                )
            } else {
                String::new()
            };
            tracing::info!(
                target: "co_sched",
                rate_tokens_per_s = trade.rate_tokens_per_s,
                pull_fixed_ms = trade.pull_fixed_ms,
                pull_ms_per_100k = trade.pull_ms_per_100k,
                copy_lambda = trade.copy_lambda,
                exclude_unlaunched = trade.exclude_unlaunched,
                pull_floor_tokens = u64::from(kv.min_incremental_tokens)
                    .max(crate::scheduler::co_schedule::TRADE_MIN_PULL_TOKENS),
                local_prefix_fallback = kv.local_prefix_fallback,
                "co_schedule TRADE cost model enabled: cost = backlog + uncached \
                 + R*max(0, pull_s - backlog/R) + lambda*pulled{hol}; no pull cap",
            );
            if !trade.is_original() {
                tracing::info!(
                    target: "co_sched",
                    wait = trade.wait.as_str(),
                    dup_charge = trade.dup_charge.as_str(),
                    margin_noise_k = trade.margin_noise_k,
                    eft_max_view_age_s = trade.eft_max_view_age_s,
                    eft_model_sha256 = cfg.trade_eft_model.as_ref().map_or("", |m| m.sha256.as_str()),
                    "co_schedule TRADE options: wait eft = R*B_w + R*S_w \
                     + R*max(0, pull_s - B_w); dup all = lambda also on recomputed copies; \
                     noise margin = keep owner unless gain_s > k*sqrt(B_o^2 + B_best^2)",
                );
            }
            if trade.is_auto() {
                let (lambda, rate) = trade.auto_description();
                let (tau, idle) = (trade.auto_tau_s, trade.auto_session_idle_s);
                tracing::info!(
                    target: "co_sched",
                    lambda_mode = trade.copy_lambda_mode.as_str(),
                    rate_mode = trade.rate_mode.as_str(),
                    tau_s = tau,
                    session_idle_s = idle,
                    "co_schedule TRADE copy price auto: {lambda}; {rate}; tau = {tau} s, \
                     session idle = {idle} s. Inputs from the router ledger: busy = time with a \
                     request dispatched to the P and its prefill not ended, R_hat = computed \
                     prefill tokens per busy second, N_active = sessions with a request on a P \
                     or an event in the idle window and no END. The fallback applies for the \
                     first tau seconds and when an input is missing",
                );
            }
            if trade.wait != crate::scheduler::co_schedule::TradeWait::E3
                && cfg
                    .global_queue
                    .is_some_and(|g| g.room == GlobalQueueRoom::Predicted)
            {
                if let Some(e3) = &cfg.trade_e3 {
                    tracing::info!(
                        target: "co_sched",
                        bound_table = %e3.bound_path.display(),
                        bound_sha256 = %e3.bound_sha256,
                        push_port = e3.push.as_ref().map_or(0, |p| p.port),
                        max_view_age_s = trade.eft_max_view_age_s,
                        e3_schema = crate::e3::SCHEMA,
                        "co_schedule GWQ room predicted: E3 forecast state on for the drain of \
                         the global waiting queue only; the TRADE decisions keep the {} wait \
                         (no e3 gate); state push {}",
                        trade.wait.as_str(),
                        if e3.push.is_some() { "on" } else { "off (polled snapshots)" },
                    );
                }
            }
            if trade.wait == crate::scheduler::co_schedule::TradeWait::E3 {
                match &cfg.trade_e3 {
                    Some(e3) => tracing::info!(
                        target: "co_sched",
                        bound_table = %e3.bound_path.display(),
                        bound_sha256 = %e3.bound_sha256,
                        bound_schema = %e3.bound_schema,
                        bound_tables_sha256 = e3.file_tables_sha256.as_deref().unwrap_or(""),
                        bound_mode = %e3.mode,
                        bound_level = %e3.level,
                        file_level = e3.file_level.as_deref().unwrap_or(""),
                        file_g_min_s = e3.file_g_min_s.unwrap_or(f64::NAN),
                        gate_m = trade.gate_m,
                        gate_gmin_s = trade.gate_gmin_s,
                        rq_margin_s = e3.rq_margin_s,
                        next_pass_lag = e3.next_pass_lag,
                        max_view_age_s = trade.eft_max_view_age_s,
                        hol_weight = trade.hol_weight,
                        shadow = %e3.shadow_path.as_ref().map_or_else(
                            || "cost trace".to_string(),
                            |p| p.display().to_string()
                        ),
                        e3_schema = crate::e3::SCHEMA,
                        reference = "e3 reference r5",
                        "co_schedule TRADE wait e3: C_w = R*(pred_s + max(0, pull_s - wait_s)) \
                         + copy; leave owner o for best w only if (C_o - C_w)/R \
                         - m*sqrt(b_o^2 + b_w^2) > g_min (b = error bound at the level); \
                         a P without a usable forecast keeps the ledger terms and never wins a move",
                    ),
                    None => tracing::error!(
                        target: "co_sched",
                        "TRADE wait = e3 without a bound table: every P uses the ledger terms \
                         (fallback reason no_context) and the owner is kept"
                    ),
                }
                if let Some(push) = cfg.trade_e3.as_ref().and_then(|e3| e3.push.as_ref()) {
                    tracing::info!(
                        target: "co_sched",
                        push_port = push.port,
                        push_bound_mode = %push.mode,
                        push_bound_level = %push.level,
                        push_file_level = push.file_level.as_deref().unwrap_or(""),
                        max_view_age_s = trade.eft_max_view_age_s,
                        offset_window_s = crate::trade_e3_push::OFFSET_WINDOW_S,
                        wire_schema = crate::trade_e3_push::WIRE_SCHEMA,
                        "co_schedule TRADE wait e3 state push on: one ZMQ SUB per P at \
                         tcp://<P host>:PORT (topic prefill_state, routing rank 0); a P's \
                         forecast starts from its newest pushed state (t0 = sampled_monotonic_s \
                         + min over 30 s of (receive - sent_mono_s)) when d - t0 <= the view-age \
                         limit, its epoch is the polled snapshot's and it is usable, else from \
                         the polled snapshot; pushed forecasts take the push bound table",
                    );
                }
            }
            if let Some(frozen) = cfg.trade_e3.as_ref().and_then(|e3| e3.frozen_fit.as_ref()) {
                tracing::info!(
                    target: "co_sched",
                    frozen_fit = %frozen.path.display(),
                    frozen_sha256 = %frozen.sha256,
                    "co_schedule TRADE e3 frozen fit: every P's GPU model, launch model and \
                     calibration constants come from this file, not from the online fits (the \
                     online samples give the arrival rates only): {}",
                    frozen.summary(),
                );
            }
            if trade.wait == crate::scheduler::co_schedule::TradeWait::Eft
                && cfg.trade_eft_model.is_none()
            {
                // The CLI requires the model; a config built in code may omit it.
                tracing::error!(
                    target: "co_sched",
                    "TRADE wait = eft without a step model: every P uses the ledger terms \
                     (fallback reason no_model)"
                );
            }
            if kv.local_prefix_fallback {
                // The CLI rejects this; a config built in code can still set it.
                tracing::warn!(
                    target: "co_sched",
                    "TRADE drops engine-side local-prefix fallback hints: a hint is sent \
                     only for a pull of at least the floor"
                );
            }
        }
        // The global waiting queue: TRADE with the ledger wait only (the CLI
        // refuses the other cases; a config built in code is logged).
        let gwq = match (cfg.global_queue, cfg.trade) {
            (Some(g), Some(trade))
                if cfg.uncached_balance
                    && (trade.wait == crate::scheduler::co_schedule::TradeWait::Ledger
                        || trade.wait == crate::scheduler::co_schedule::TradeWait::E3)
                    && (g.room == GlobalQueueRoom::Tokens || cfg.trade_e3.is_some()) =>
            {
                let gwq = Gwq::new(g, trade.rate_tokens_per_s, cfg.load_poll_ms);
                let (order, guard) = match g.mode {
                    GlobalQueueMode::Sjf => (
                        "smallest service u/R on the plain TRADE choice first",
                        "the sjf starvation guard",
                    ),
                    GlobalQueueMode::Hrrn => (
                        "highest (wait + S) / S first",
                        "only a safety net with hrrn",
                    ),
                };
                let mode = g.mode.as_str();
                let depth = gwq.depth_tokens;
                let reaction_ms = crate::global_queue::auto_reaction_ms(cfg.load_poll_ms);
                let depth_rule = match g.depth {
                    GlobalQueueDepth::Auto => format!(
                        "auto = chunk {} + R {} x reaction {} ms (poll {} ms + dispatch {} ms)",
                        crate::global_queue::CHUNK_TOKENS,
                        trade.rate_tokens_per_s,
                        reaction_ms,
                        cfg.load_poll_ms,
                        crate::global_queue::AUTO_DISPATCH_DELAY_MS,
                    ),
                    GlobalQueueDepth::Tokens(_) => "fixed".to_string(),
                };
                let stale_ms = gwq.stale_after.as_millis() as u64;
                let max_wait_ms = g.max_wait_ms;
                let chunk = crate::global_queue::CHUNK_TOKENS;
                let hold = g.hold.as_str();
                let room = g.room.as_str();
                let lead_s = gwq.lead_s;
                let hold_rule = match g.hold {
                    GlobalQueueHold::All => "every arrival may wait".to_string(),
                    GlobalQueueHold::Free => format!(
                        "only free arrivals wait (largest local hit minus second largest below \
                         {chunk} tokens); a bound arrival takes the TRADE decision at once ({} \
                         wait) and waits in the engine queue",
                        trade.wait.as_str()
                    ),
                };
                let room_rule = match g.room {
                    GlobalQueueRoom::Tokens => "tokens: queued work below the depth".to_string(),
                    GlobalQueueRoom::Predicted => format!(
                        "predicted: the E3 drain of a cold {chunk}-token probe (newest pushed or \
                         polled state + router rows) at most lead {lead_s:.4} s = depth / R, or the \
                         token rule (work-conserving floor: the forecast may make a P look emptier \
                         than the ledger, never fuller), and the walk prices the P's wait as \
                         min(R x drain, ledger backlog); a P without a usable forecast keeps the \
                         token rule"
                    ),
                };
                tracing::info!(
                    target: "co_sched",
                    mode,
                    hold,
                    room,
                    lead_s,
                    trade_wait = trade.wait.as_str(),
                    copy_lambda = trade.copy_lambda,
                    copy_lambda_mode = trade.copy_lambda_mode.as_str(),
                    depth_tokens = depth,
                    depth_mode = if g.depth == GlobalQueueDepth::Auto { "auto" } else { "fixed" },
                    chunk_tokens = crate::global_queue::CHUNK_TOKENS,
                    rate_tokens_per_s = trade.rate_tokens_per_s,
                    reaction_ms,
                    poll_ms = cfg.load_poll_ms,
                    dispatch_delay_ms = crate::global_queue::AUTO_DISPATCH_DELAY_MS,
                    stale_after_ms = stale_ms,
                    max_wait_ms,
                    tick_ms = gwq.tick.as_millis() as u64,
                    "co_schedule global waiting queue on: mode {mode} (release order: {order}); \
                     depth {depth} tokens per P ({depth_rule}); a P has room while its queued \
                     work on the router ledger is below the depth (queued = uncached tokens routed \
                     and not yet in a progress snapshot + remaining work of its launched requests \
                     beyond one pass of {chunk} tokens); a request goes to its TRADE argmin P when \
                     that P has room (late P binding: the argmin is taken at release, with the \
                     held work ahead as a virtual backlog), else it waits in one router queue; a P \
                     whose progress view is missing or older than {stale_ms} ms has room \
                     (fail-open); a request held {max_wait_ms} ms goes to its TRADE argmin at once \
                     ({guard}); hold {hold}: {hold_rule}; room {room}: {room_rule}; the walk and \
                     its releases use the ledger terms",
                );
                Some(gwq)
            }
            (Some(_), _) => {
                tracing::error!(
                    target: "co_sched",
                    "the global waiting queue needs TRADE with the ledger or e3 wait \
                     (--co-schedule-trade, --co-schedule-uncached-balance; room predicted \
                     needs the E3 forecast state); it stays off"
                );
                None
            }
            (None, _) => None,
        };
        let poll_client = reqwest::Client::builder()
            .connect_timeout(Duration::from_millis(500))
            .timeout(Duration::from_millis(800))
            .build()
            .expect("build co_schedule poll client");
        let cost_book = if cfg.uncached_balance {
            Some(
                match cfg.restore_cost_divisor {
                    Some(divisor) => crate::cost_balance::CostBook::new_restore_aware(
                        cfg.cost_trace.as_deref(),
                        divisor,
                    ),
                    None => crate::cost_balance::CostBook::new_uncached(cfg.cost_trace.as_deref()),
                }
                .expect("create new uncached trace before serving requests"),
            )
        } else {
            (cfg.cost_balance || cfg.locality_margin_tokens.is_some()).then(|| {
                crate::cost_balance::CostBook::new(cfg.cost_trace.as_deref())
                    .expect("create new cost trace before serving requests")
            })
        };
        // The E3 forecast state: `wait = e3`, or the GWQ room `predicted`
        // (the drain only, also with the ledger wait).
        let e3_on = cfg
            .trade
            .is_some_and(|t| t.wait == crate::scheduler::co_schedule::TradeWait::E3)
            || (cfg.trade_e3.is_some()
                && cfg
                    .global_queue
                    .is_some_and(|g| g.room == GlobalQueueRoom::Predicted));
        let push = cfg
            .trade_e3
            .as_ref()
            .and_then(|e3| e3.push.as_ref())
            .filter(|_| e3_on);
        let e3_online = e3_on.then(|| {
            let next_pass_lag = cfg.trade_e3.as_ref().is_some_and(|e3| e3.next_pass_lag);
            let frozen = cfg.trade_e3.as_ref().and_then(|e3| e3.frozen_fit.clone());
            let online = crate::trade_e3::E3Online::new(Instant::now());
            let online = online
                .with_next_pass_lag(next_pass_lag)
                .with_frozen_fit(frozen);
            Arc::new(if push.is_some() {
                online.with_label_sources()
            } else {
                online
            })
        });
        let e3_shadow = match (&cfg.trade_e3, &cost_book) {
            (Some(e3), Some(book)) if e3_on => Some(Arc::new(match &e3.shadow_path {
                Some(path) => crate::trade_e3::ShadowSink::file(path)
                    .expect("create new e3 shadow log before serving requests"),
                None => crate::trade_e3::ShadowSink::Trace(Arc::clone(book)),
            })),
            _ => None,
        };
        let e3_push = match (push, &e3_online) {
            (Some(push), Some(online)) => Some(crate::trade_e3_push::PushHub::new(
                push.port,
                Arc::clone(online),
                e3_shadow.clone(),
                cost_book.clone(),
            )),
            _ => None,
        };
        let this = Arc::new(Self {
            cfg,
            cost_book,
            injector,
            registry,
            model,
            state: Mutex::new(SchedState::default()),
            notify: Notify::new(),
            pending: AtomicU8::new(0),
            pass_counter: AtomicU64::new(0),
            poll_now: Arc::new(Notify::new()),
            poll_now_signals: AtomicU64::new(0),
            poll_client,
            progress_refresh: Arc::new(crate::load_refresh::LoadRefresh::default()),
            routing: parking_lot::RwLock::new(HashMap::new()),
            routing_rejections: Mutex::new(HashMap::new()),
            gwq,
            e3_online,
            e3_shadow,
            e3_push,
        });
        if this.gwq.is_some() {
            // A terminal status frees P work: run a release pass.
            if let Some(book) = &this.cost_book {
                let weak = Arc::downgrade(&this);
                book.set_terminal_hook(Box::new(move || {
                    if let Some(sched) = weak.upgrade() {
                        sched.wake(Triggers::RETIRE);
                    }
                }));
            }
        }
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let weak = Arc::downgrade(&this);
            if this.gwq.is_some() {
                handle.spawn(gwq_dispatcher_loop(weak.clone()));
                handle.spawn(gwq_stats_loop(weak.clone()));
            }
            if this.cfg.uncached_balance && !this.cfg.on_demand_load_refresh {
                handle.spawn(progress_poller_loop(weak.clone()));
            }
            if this.e3_push.is_some() {
                handle.spawn(push_supervisor_loop(weak.clone()));
            }
            if let Some(book) = &this.cost_book {
                let weak_book = Arc::downgrade(book);
                handle.spawn(async move {
                    let mut interval = tokio::time::interval(Duration::from_secs(1));
                    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                    loop {
                        interval.tick().await;
                        let Some(book) = weak_book.upgrade() else {
                            break;
                        };
                        book.snapshot();
                    }
                });
            } else {
                handle.spawn(dispatcher_loop(weak.clone()));
                handle.spawn(poller_loop(weak));
            }
        } else {
            tracing::warn!(
                target: "co_sched",
                "no tokio runtime at construction; dispatcher/poller tasks not started"
            );
        }
        this
    }

    /// Admit one request: query the indexer for per-worker hits, queue it,
    /// and wait until the co-scheduler dispatches it. `tokens` drive both
    /// the hit lookup and the backlog ledger; when absent (`None` — e.g. a
    /// text prompt the ingress could not tokenize) the request degrades to
    /// hit-less min-backlog dispatch using `est_tokens` for the ledger.
    pub async fn admit(
        &self,
        self_arc: &Arc<Self>,
        tokens: Option<&[u32]>,
        est_tokens: usize,
        cache_salt: Option<&str>,
        rid: Option<String>,
    ) -> Result<Dispatch, AdmitError> {
        self.admit_session(self_arc, tokens, est_tokens, cache_salt, rid, None, None)
            .await
    }

    /// True when TRADE takes its copy price or R from the router's
    /// measurements (`auto`); the routes then pass the session identity.
    pub fn trade_auto(&self) -> bool {
        self.cfg.trade.is_some_and(|t| t.is_auto())
    }

    /// The client ended `session` (`POST /kv_sessions/end`; the identity of
    /// [`crate::scheduler::co_schedule::trade_session_key`]): it leaves
    /// N_active of the copy price `auto` at once. Returns whether it was
    /// counted.
    pub fn trade_auto_session_end(&self, session: &str) -> bool {
        self.trade_auto()
            && self
                .cost_book
                .as_ref()
                .is_some_and(|book| book.trade_auto_end_session(session))
    }

    /// [`Self::admit`] with the request's session identity
    /// ([`crate::scheduler::co_schedule::trade_session_key`]). Only the TRADE
    /// copy price `auto` uses it (N_active); it does not change routing
    /// otherwise. `e3_session`: the session key that TRADE `wait = e3` uses
    /// for the host rule of the target ([`crate::trade_e3::session_key`];
    /// see [`Self::wants_session`]).
    #[allow(clippy::too_many_arguments)]
    pub async fn admit_session(
        &self,
        self_arc: &Arc<Self>,
        tokens: Option<&[u32]>,
        est_tokens: usize,
        cache_salt: Option<&str>,
        rid: Option<String>,
        session: Option<String>,
        e3_session: Option<String>,
    ) -> Result<Dispatch, AdmitError> {
        if self.cfg.on_demand_load_refresh {
            // Start before the cache lookup so the two network operations overlap.
            self_arc.refresh_progress(false);
        }
        let answer = match tokens {
            Some(t) if !t.is_empty() => self.injector.prepare(t, cache_salt, &self.model.0).await,
            _ => None,
        };
        let tokens_len = tokens.map(|t| t.len()).unwrap_or(est_tokens).max(1) as u64;
        if self.cfg.on_demand_load_refresh {
            self.wait_for_initial_progress().await?;
        }
        if self.cost_book.is_some() {
            if self.gwq.is_some() {
                return self
                    .admit_gwq(self_arc, tokens_len, answer, rid, session, e3_session)
                    .await;
            }
            self.admit_reserved(self_arc, tokens_len, answer, rid, session, e3_session)
        } else {
            self.enqueue(self_arc, tokens_len, answer, rid).await
        }
    }

    /// One slot per worker, including retries and periodic-mode triggers.
    fn refresh_progress(self: &Arc<Self>, force: bool) -> Vec<tokio::task::JoinHandle<()>> {
        let age = Duration::from_millis(self.cfg.load_poll_ms);
        self.pool().into_iter().filter_map(|worker| {
            let query = self.progress_refresh.claim(&worker.url, Instant::now(), age, force)?;
            let sched = Arc::clone(self);
            Some(tokio::spawn(async move {
                let result = async {
                    let routing_wanted = sched.routing_wanted(&worker.url);
                    let fetched = fetch_prefill_progress(
                        &sched.poll_client, &worker.url, routing_wanted).await;
                    let (progress, load, routing) = match fetched {
                        // An engine without the section (an older
                        // engine) rejects the include. Poll again without it:
                        // progress goes on, and this P has no view (fallback
                        // no_view) until a later probe.
                        Err(error) if routing_wanted && include_rejected(&error) => {
                            let fetched = fetch_prefill_progress(
                                &sched.poll_client, &worker.url, false).await?;
                            sched.note_routing_rejected(&worker.url, &error);
                            fetched
                        }
                        fetched => {
                            let fetched = fetched?;
                            if routing_wanted {
                                sched.note_routing_accepted(&worker.url);
                            }
                            fetched
                        }
                    };
                    let received = Instant::now();
                    query.check(&progress.epoch, progress.sequence)?;
                    let cursor = (progress.epoch.clone(), progress.sequence);
                    sched.cost_book.as_ref().expect("uncached book")
                        .apply_progress_received(&worker.url, progress, load, query.sent_at, received)
                        .map_err(anyhow::Error::msg)?;
                    if let Some(raw) = routing {
                        // Parsed here, off the cost-book lock.
                        let view = crate::eft_cheap::SnapshotView::parse(&raw, 1, None);
                        let e3 = sched.observe_e3(&worker.url, &raw, query.sent_at, received);
                        sched.publish_routing(&worker.url, view, query.sent_at, received, e3);
                    }
                    Ok::<_, anyhow::Error>(cursor)
                }.await;
                let elapsed_ms = query.sent_at.elapsed().as_secs_f64() * 1000.0;
                crate::load_refresh::log_progress_result(&worker.url, &result, elapsed_ms);
                let applied = result.is_ok();
                query.finish(result.ok(), Instant::now());
                if applied && sched.gwq.is_some() {
                    // New progress: the queued work of this P may have room.
                    sched.wake(Triggers::POLL);
                }
            }))
        }).collect()
    }

    /// TRADE `wait = eft` and `wait = e3` ask the engine for its
    /// `prefill_routing` snapshot.
    fn wants_routing(&self) -> bool {
        self.e3_online.is_some()
            || self.cfg.trade.is_some_and(|t| {
                matches!(
                    t.wait,
                    crate::scheduler::co_schedule::TradeWait::Eft
                        | crate::scheduler::co_schedule::TradeWait::E3
                )
            })
    }

    /// TRADE `wait = e3` needs the request's session key (host rule).
    pub fn wants_session(&self) -> bool {
        self.e3_online.is_some()
    }

    /// TRADE `wait = e3`: the snapshot in E3 form. Its timeline samples go
    /// into the online fits first (and the `e3_label` lines of routed
    /// requests whose completion they hold into the shadow log); the stored
    /// view keeps no samples. None for the other wait sources.
    fn observe_e3(
        &self,
        worker: &str,
        raw: &Value,
        sent: Instant,
        received: Instant,
    ) -> Option<Arc<crate::trade_e3::E3View>> {
        let online = self.e3_online.as_ref()?;
        let mut view = crate::trade_e3::E3View::parse(raw);
        let labels = online.observe(worker, &view, sent, received);
        if let Some(sink) = &self.e3_shadow {
            for label in labels {
                sink.write(label);
            }
        }
        view.samples = Vec::new();
        view.boots = Vec::new();
        Some(Arc::new(view))
    }

    /// [`Self::wants_routing`], except for a worker whose engine rejected the
    /// section less than [`ROUTING_REPROBE`] ago.
    fn routing_wanted(&self, worker: &str) -> bool {
        self.wants_routing()
            && self
                .routing_rejections
                .lock()
                .get(worker)
                .is_none_or(|at| at.elapsed() >= ROUTING_REPROBE)
    }

    /// The engine of `worker` rejected `include=prefill_routing` and answered
    /// without it. Its old snapshot is dropped (fallback reason no_view); the
    /// first rejection is logged as a warning.
    fn note_routing_rejected(&self, worker: &str, error: &anyhow::Error) {
        let first = self
            .routing_rejections
            .lock()
            .insert(worker.to_string(), Instant::now())
            .is_none();
        self.routing.write().remove(worker);
        if first {
            tracing::warn!(target: "co_sched", worker, %error,
                "engine rejected include=prefill_routing (needs an engine with the prefill_routing load section); \
                 polling without it, so TRADE wait eft uses the fallback for this P (no_view); \
                 asking again every 30 s");
        } else {
            tracing::debug!(target: "co_sched", worker, %error,
                "engine still rejects include=prefill_routing");
        }
    }

    /// The engine of `worker` answered `include=prefill_routing`.
    fn note_routing_accepted(&self, worker: &str) {
        if self.routing_rejections.lock().remove(worker).is_some() {
            tracing::info!(target: "co_sched", worker,
                "engine accepts include=prefill_routing again");
        }
    }

    /// Keep the newest snapshot of `worker`: a snapshot of the same engine
    /// epoch with a sequence that is not newer, or from an older poll, is
    /// dropped. A new epoch (engine restart) replaces the old one.
    fn publish_routing(
        &self,
        worker: &str,
        view: crate::eft_cheap::SnapshotView,
        poll_sent: Instant,
        poll_received: Instant,
        e3: Option<Arc<crate::trade_e3::E3View>>,
    ) {
        let mut store = self.routing.write();
        if let Some(old) = store.get(worker) {
            let same_epoch = old.view.epoch == view.epoch;
            let not_newer = match (old.view.sequence.as_u64(), view.sequence.as_u64()) {
                (Some(old), Some(new)) => new <= old,
                _ => false,
            };
            if poll_sent < old.poll_sent || (same_epoch && not_newer) {
                return;
            }
        }
        store.insert(
            worker.to_string(),
            Arc::new(crate::cost_balance::RoutingSnapshot {
                view,
                poll_sent,
                poll_received,
                e3,
            }),
        );
    }

    /// TRADE `wait = e3`: the inputs of one decision that need no cost-book
    /// state, built before its lock: per candidate the newest snapshot, the
    /// (P, step) fit (computed here on the first decision of a step; each
    /// new fit is written to the cost trace as an `e3_fit` event), the
    /// receipt line, the park cap and the target tokens with the host rule.
    fn trade_e3_context(
        &self,
        candidates: &[crate::cost_balance::Candidate],
        tokens_len: u64,
        rid: Option<&str>,
        session: Option<&str>,
    ) -> Option<crate::trade_e3::TradeE3Context> {
        let online = self.e3_online.as_ref()?;
        let cfg = self.cfg.trade_e3.clone()?;
        let now_s = online.secs(Instant::now());
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
            let tokens = crate::trade_e3::CandidateTokens {
                input: tokens_len as i64,
                local_hit: local as i64,
                gpu_hit: c.gpu_hit.map(|g| g.min(local) as i64),
                expected_hit: c.expected_hit.min(tokens_len).max(local) as i64,
            };
            let (mut prep, new_fit) = online.prepare(
                &c.worker_url,
                snapshot,
                rid.unwrap_or(""),
                tokens,
                session,
                now_s,
            );
            // State push on: the P's newest pushed state; the decision
            // checks its age, epoch and view under the cost-book lock.
            prep.push = self
                .e3_push
                .as_ref()
                .and_then(|hub| hub.state_of(&c.worker_url));
            if let (Some(fit), Some(book)) = (new_fit, self.cost_book.as_ref()) {
                let mut event = serde_json::json!({
                    "event": "e3_fit",
                    "worker_url": c.worker_url,
                    "e3_clock_s": now_s,
                });
                if let (Value::Object(event), Value::Object(record)) = (&mut event, fit.record()) {
                    event.extend(record);
                }
                if let Err(error) = book.emit_event(event) {
                    tracing::warn!(target: "co_sched", %error, "e3 fit not written to the cost trace");
                }
            }
            prepared.insert(c.worker_url.clone(), prep);
        }
        Some(crate::trade_e3::TradeE3Context {
            cfg,
            anchor: online.anchor(),
            unix_at_anchor: online.unix_of(0.0),
            candidates: prepared,
        })
    }

    /// TRADE `wait = e3`, after the reservation: remember the routed request
    /// in the online state (receipt line, host rule, arrival rates, labels)
    /// and write its `e3_shadow` line (chosen P, its `pred_s`, decision time).
    fn record_e3_route(
        &self,
        decision: &Value,
        rid: Option<&str>,
        worker: &str,
        tokens_len: u64,
        session: Option<String>,
    ) {
        let Some(online) = &self.e3_online else {
            return;
        };
        let Some(decided_s) = decision["e3_decision_s"].as_f64() else {
            return;
        };
        let chosen = &decision["e3_chosen"];
        let rid = rid.unwrap_or("");
        let decided_unix_s = decision["e3_decision_unix_s"].as_f64().unwrap_or(0.0);
        online.record_route(crate::trade_e3::RouteRecord {
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
        if let Some(sink) = &self.e3_shadow {
            let mut line = serde_json::json!({
                "event": "e3_shadow",
                "e3_schema": crate::trade_e3::SHADOW_SCHEMA,
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
                "gain_s": decision["e3_gain_s"],
                "margin_s": decision["e3_margin_s"],
                "protect_reason": decision["e3_protect_reason"],
            });
            if self.e3_push.is_some() {
                // `push` or `poll`: live MAPE per state source.
                line["state_source"] = chosen["state_source"].clone();
            }
            sink.write(line);
        }
    }

    /// The EFT inputs of one decision: the model and the newest snapshot of
    /// each pool worker, cloned before the cost-book lock.
    fn trade_eft_context(&self, pool: &[Arc<Worker>]) -> Option<crate::cost_balance::TradeEftContext> {
        let model = self.cfg.trade_eft_model.clone()?;
        let store = self.routing.read();
        Some(crate::cost_balance::TradeEftContext {
            model,
            knobs: crate::eft_cheap::EftKnobs::default(),
            views: pool
                .iter()
                .filter_map(|w| store.get(&w.url).map(|v| (w.url.clone(), Arc::clone(v))))
                .collect(),
        })
    }

    /// Only startup waits for a first usable snapshot. Once initialized,
    /// stale information triggers a background query and never gates admission.
    async fn wait_for_initial_progress(&self) -> Result<(), AdmitError> {
        let deadline = tokio::time::Instant::now() + Duration::from_millis(950);
        loop {
            let notified = self.progress_refresh.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.pool().iter().any(|w| self.progress_refresh.initialized(&w.url)) {
                return Ok(());
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return Err(AdmitError::Wedged);
            }
        }
    }

    /// Query is finished before this call. Selection and reservation are one
    /// CostBook transaction. Uncached mode uses RID progress; it never waits
    /// for a load poll or for a dispatch timer. `session` reaches only the
    /// TRADE copy price `auto`.
    fn admit_reserved(
        &self,
        self_arc: &Arc<Self>,
        tokens_len: u64,
        answer: Option<IndexerAnswer>,
        rid: Option<String>,
        session: Option<String>,
        e3_session: Option<String>,
    ) -> Result<Dispatch, AdmitError> {
        let decision_started = Instant::now();
        let pool = self.reserved_pool();
        if !Self::reserved_pool_usable(&pool) {
            tracing::error!(target: "cost_balance", "reserved placement requires healthy PD prefill workers");
            return Err(AdmitError::Wedged);
        }
        let candidates = self.reserved_candidates(&pool, answer.as_ref(), tokens_len);
        let book = self.cost_book.as_ref().expect("reserved placement");
        let mut trade_decision = None;
        let reserved = if let Some(params) = self.cfg.trade {
            let reserved = if params.wait == crate::scheduler::co_schedule::TradeWait::E3 {
                // Fits, targets and snapshots before the cost-book lock.
                let e3 = self.trade_e3_context(
                    &candidates,
                    tokens_len,
                    rid.as_deref(),
                    e3_session.as_deref(),
                );
                book.reserve_trade_e3_session(
                    tokens_len,
                    rid.clone(),
                    candidates,
                    params,
                    e3.as_ref(),
                    session.as_deref(),
                )
            } else {
                let eft = if params.wait == crate::scheduler::co_schedule::TradeWait::Eft {
                    self.trade_eft_context(&pool)
                } else {
                    None
                };
                book.reserve_trade_session(
                    tokens_len,
                    rid.clone(),
                    candidates,
                    params,
                    eft.as_ref(),
                    session.as_deref(),
                )
            };
            reserved.map(|(selected, reservation, decision)| {
                trade_decision = Some(decision);
                (selected, reservation)
            })
        } else if self.cfg.uncached_balance {
            book.reserve_uncached_with_margin(
                tokens_len,
                rid.clone(),
                candidates,
                self.cfg.owner_threshold_tokens,
                self.cfg.detour_margin_tokens,
            )
        } else {
            match self.cfg.locality_margin_tokens {
                Some(margin) => book.reserve_locality(tokens_len, rid.clone(), candidates, margin),
                None => book.reserve(tokens_len, rid.clone(), candidates),
            }
        };
        let (selected, reservation) = reserved.map_err(|error| {
            tracing::error!(target: "cost_balance", %error, "cost reservation failed");
            AdmitError::Wedged
        })?;
        let worker = pool
            .into_iter()
            .find(|w| w.url == selected.worker_url)
            .expect("reserved candidate belongs to pool");
        Ok(self.finish_reserved(
            self_arc,
            worker,
            selected,
            reservation,
            trade_decision,
            answer.as_ref(),
            rid.as_deref(),
            tokens_len,
            decision_started,
            0,
            Triggers::ARRIVAL,
            0,
            e3_session,
        ))
    }

    /// The pool of a reserved placement: the PD prefill candidates (with
    /// on-demand refresh, only those with a first progress snapshot).
    fn reserved_pool(&self) -> Vec<Arc<Worker>> {
        self.pool()
            .into_iter()
            .filter(|w| {
                !self.cfg.on_demand_load_refresh || self.progress_refresh.initialized(&w.url)
            })
            .collect()
    }

    /// A reserved placement needs a non-empty pool of PD prefill workers.
    fn reserved_pool_usable(pool: &[Arc<Worker>]) -> bool {
        !pool.is_empty()
            && pool
                .iter()
                .all(|w| w.mode() == crate::discovery::WorkerMode::Prefill)
    }

    /// The candidates of a reserved placement on `pool`, in pool order: each
    /// P's own hit, its GPU hit and the pull hint that it would get.
    fn reserved_candidates(
        &self,
        pool: &[Arc<Worker>],
        answer: Option<&IndexerAnswer>,
        tokens_len: u64,
    ) -> Vec<crate::cost_balance::Candidate> {
        // TRADE sends a hint only for a pull of at least the floor (never
        // below 512). A smaller gain is recomputed on the target and costs no
        // transfer. pick() already applies the flag's gate; this also drops
        // the owner's hints and sub-floor hints when that gate is lower
        // (a config built in code) or the local-prefix fallback is on.
        let trade_floor = u64::from(self.injector.min_incremental_tokens())
            .max(crate::scheduler::co_schedule::TRADE_MIN_PULL_TOKENS);
        pool.iter()
            .map(|worker| {
                let local_hit = answer
                    .map_or(0, |a| {
                        KvHintInjector::target_hit_tokens(a, &worker.url) as u64
                    })
                    .min(tokens_len);
                let mut hint = answer.and_then(|a| self.injector.pick(a, &worker.url));
                let mut remote_hit = hint
                    .as_ref()
                    .and_then(|h| h.get("expected_hit_tokens"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                if self.cfg.trade.is_some()
                    && crate::scheduler::co_schedule::trade_pull_tokens(
                        local_hit,
                        remote_hit,
                        tokens_len,
                        trade_floor,
                    ) == 0
                {
                    hint = None;
                    remote_hit = 0;
                }
                crate::cost_balance::Candidate {
                    worker_url: worker.url.clone(),
                    local_hit,
                    gpu_hit: answer
                        .and_then(|a| a.observed_gpu_hits.get(&worker.url))
                        .copied()
                        .map(u64::from),
                    expected_hit: local_hit.max(remote_hit).min(tokens_len),
                    hint,
                }
            })
            .collect()
    }

    /// The log lines and the dispatch of one reserved placement (also of a
    /// release of the global waiting queue): `queue_wait_ms` is the time in
    /// a router queue, `triggers` the wake sources of the pass and
    /// `queue_depth` the requests left in the queue.
    #[allow(clippy::too_many_arguments)]
    fn finish_reserved(
        &self,
        self_arc: &Arc<Self>,
        worker: Arc<Worker>,
        selected: crate::cost_balance::Candidate,
        reservation: crate::cost_balance::Reservation,
        trade_decision: Option<Value>,
        answer: Option<&IndexerAnswer>,
        rid: Option<&str>,
        tokens_len: u64,
        decision_started: Instant,
        queue_wait_ms: u64,
        triggers: Triggers,
        queue_depth: usize,
        e3_session: Option<String>,
    ) -> Dispatch {
        let trigger = triggers.primary();
        let triggers_text = triggers.to_string();
        let best_source = answer.and_then(|a| a.candidates.first());
        let best_hit = best_source.map_or(0, |c| c.expected_hit_tokens);
        let a_worker = best_source.map_or("", |c| c.worker_url.as_str());
        let source = selected
            .hint
            .as_ref()
            .and_then(|h| h.get("source_worker_id"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let branch = if self.cfg.uncached_balance {
            Branch::UncachedBalanced
        } else if self.cfg.locality_margin_tokens.is_some() {
            Branch::Locality
        } else {
            Branch::CostBalanced
        };
        // Keep the existing trace contract for throughput/cache-hit analysis.
        tracing::info!(target: "co_sched", rid = rid.unwrap_or(""),
            branch = branch.as_str(), trigger = trigger.as_str(), triggers = triggers_text.as_str(),
            worker = %worker.url, a_worker, source, hit_here = selected.local_hit, best_hit,
            gpu_hit_here = selected.gpu_hit,
            input_tokens = tokens_len, expected_hit = selected.expected_hit,
            expected_uncached_tokens = tokens_len - selected.expected_hit,
            uncached_tokens = tokens_len - selected.local_hit,
            injected = selected.hint.is_some(), queue_wait_ms,
            decision_us = decision_started.elapsed().as_micros() as u64,
            queue_depth, cost = %reservation.debuginfo(),
            "co_schedule dispatch");
        if let Some(decision) = &trade_decision {
            // The full per-candidate record is in the cost trace.
            tracing::info!(target: "co_sched", rid = rid.unwrap_or(""),
                worker = %worker.url,
                owner = decision["owner"].as_str().unwrap_or(""),
                detour = decision["detour"].as_bool().unwrap_or(false),
                pulled_tokens = decision["pulled_tokens"].as_u64().unwrap_or(0),
                trade_cost = decision["trade_cost"].as_f64().unwrap_or(0.0),
                "co_schedule trade decision");
            if decision.get("wait").is_some() {
                tracing::info!(target: "co_sched", rid = rid.unwrap_or(""),
                    worker = %worker.url,
                    wait = decision["wait"].as_str().unwrap_or(""),
                    wait_s = decision["wait_s"].as_f64().unwrap_or(0.0),
                    service_s = decision["service_s"].as_f64().unwrap_or(0.0),
                    best = decision["best"].as_str().unwrap_or(""),
                    margin_protected = decision["margin_protected"].as_bool().unwrap_or(false),
                    wait_fallbacks = decision["wait_fallbacks"].as_u64().unwrap_or(0),
                    "co_schedule trade terms");
            }
            if decision.get("lambda_mode").is_some() {
                tracing::info!(target: "co_sched", rid = rid.unwrap_or(""),
                    worker = %worker.url,
                    copy_lambda_used = decision["copy_lambda_used"].as_f64().unwrap_or(0.0),
                    rate_used = decision["rate_used"].as_f64().unwrap_or(0.0),
                    q = %decision["q"],
                    busy = %decision["busy"],
                    n_active = decision["n_active"].as_u64().unwrap_or(0),
                    r_hat = %decision["r_hat"],
                    fallback_reason = decision["fallback_reason"].as_str().unwrap_or("none"),
                    "co_schedule trade copy price");
            }
            self.record_e3_route(decision, rid, &worker.url, tokens_len, e3_session);
        }
        self.injector.log_decision(
            answer,
            if self.cfg.trade.is_some() {
                "CoScheduleTrade"
            } else if self.cfg.uncached_balance {
                "CoScheduleUncached"
            } else if self.cfg.locality_margin_tokens.is_some() {
                "CoScheduleLocality"
            } else {
                "CoScheduleCost"
            },
            &worker.url,
            selected.hint.as_ref(),
        );
        // Legacy first-byte hooks may retire this inert handle. Only the
        // unique reservation held by the P task controls the cost ledger.
        let ledger = LedgerHandle {
            entry: Arc::new(LedgerEntry {
                tokens: 0,
                dispatched_at: Instant::now(),
                retired: AtomicBool::new(true),
            }),
            sched: Arc::downgrade(self_arc),
        };
        Dispatch {
            worker,
            hint: selected.hint,
            ledger,
            cost_reservation: Some(reservation),
            branch,
            queue_wait_ms,
            trigger,
        }
    }

    /// Admission through the global waiting queue: the request goes to the tail of the
    /// queue and a release pass runs at once on this task. With an empty
    /// queue this is the TRADE decision of [`Self::admit_reserved`], and the
    /// request is dispatched at once when its TRADE choice has room. The
    /// arrival is refused at once, as the TRADE path refuses it, without a
    /// usable pool or with a missing or reused RID.
    async fn admit_gwq(
        &self,
        self_arc: &Arc<Self>,
        tokens_len: u64,
        answer: Option<IndexerAnswer>,
        rid: Option<String>,
        session: Option<String>,
        e3_session: Option<String>,
    ) -> Result<Dispatch, AdmitError> {
        let gwq = self.gwq.as_ref().expect("global waiting queue on");
        let book = self.cost_book.as_ref().expect("TRADE has a cost book");
        let pool = self.reserved_pool();
        if !Self::reserved_pool_usable(&pool) {
            gwq.state.lock().counters.refused_pool += 1;
            tracing::error!(target: "cost_balance", "reserved placement requires healthy PD prefill workers");
            return Err(AdmitError::Wedged);
        }
        let duplicate = |st: &GwqState| {
            rid.as_deref()
                .is_some_and(|r| st.entries.iter().any(|e| e.rid.as_deref() == Some(r)))
        };
        let refusal = book.rid_refusal(rid.as_deref()).or_else(|| {
            duplicate(&gwq.state.lock())
                .then_some("uncached balance cannot reuse a rid; the global waiting queue holds it")
        });
        if let Some(error) = refusal {
            gwq.state.lock().counters.refused_rid += 1;
            tracing::error!(target: "cost_balance", %error, "cost reservation failed");
            return Err(AdmitError::Wedged);
        }
        let cands = self.reserved_candidates(&pool, answer.as_ref(), tokens_len);
        if gwq.cfg.hold == GlobalQueueHold::Free {
            let hits: Vec<u64> = cands.iter().map(|c| c.local_hit).collect();
            if crate::global_queue::bound_to_owner(&hits, crate::global_queue::CHUNK_TOKENS) {
                // Bound to its cache owner: the TRADE decision now (the
                // configured wait), and the engine queue holds it.
                gwq.state.lock().counters.bypassed_bound += 1;
                return self.admit_reserved(self_arc, tokens_len, answer, rid, session, e3_session);
            }
        }
        let cands_pool: Vec<String> = pool.iter().map(|w| w.url.clone()).collect();
        let session = session
            .as_deref()
            .map(crate::scheduler::co_schedule::trade_session_hash);
        let pass_id = self.pass_counter.fetch_add(1, Ordering::Relaxed) + 1;
        let (tx, rx) = oneshot::channel();
        let id = {
            let mut st = gwq.state.lock();
            let id = st.next_id;
            st.next_id += 1;
            st.counters.arrivals += 1;
            st.entries.push(GwqEntry {
                id,
                arrived: Instant::now(),
                arrived_unix_ms: crate::cost_balance::unix_ms(),
                tokens_len,
                answer,
                rid,
                session,
                e3_session,
                cands,
                cands_pool,
                p_arrival: None,
                arrival_pass: pass_id,
                tx,
            });
            id
        };
        // Takes the request out of the queue if this future is dropped
        // (the client left) before a release.
        let mut guard = GwqGuard {
            sched: Arc::downgrade(self_arc),
            id,
            armed: true,
        };
        if self.gwq_pass(self_arc, Triggers::ARRIVAL, pass_id).held > 0 {
            // Requests wait: the dispatcher re-reads the oldest deadline (it
            // may sleep for the idle tick).
            self.wake(Triggers::ARRIVAL);
        }
        // The max wait releases every request (also without room); this
        // bound only trips when even that is impossible (no usable pool).
        let budget = gwq.max_wait * 2 + Duration::from_secs(10);
        let result = tokio::time::timeout(budget, rx).await;
        guard.armed = false;
        match result {
            Ok(Ok(dispatch)) => Ok(dispatch),
            // Refused at release (logged there) or the scheduler is gone.
            Ok(Err(_)) => Err(AdmitError::Wedged),
            Err(_) => {
                self.gwq_drop(id, "wait_budget");
                Err(AdmitError::Wedged)
            }
        }
    }

    /// One release pass of the global waiting queue: decision steps
    /// ([`crate::cost_balance::CostBook::gwq_step`], under the queue lock and
    /// the cost-book lock) until a step releases nothing. Each released
    /// request is handed to its route outside the locks. Requests whose
    /// client left are dropped first. `pass_id` tells the requests whose own
    /// arrival started this pass (`held = false`).
    pub(crate) fn gwq_pass(
        &self,
        self_arc: &Arc<Self>,
        triggers: Triggers,
        pass_id: u64,
    ) -> GwqPassOutcome {
        let mut outcome = GwqPassOutcome::default();
        let (Some(gwq), Some(book), Some(params)) =
            (self.gwq.as_ref(), self.cost_book.as_ref(), self.cfg.trade)
        else {
            return outcome;
        };
        // The walk and its releases use the ledger terms (with `wait = e3`
        // too; `room = predicted` replaces the backlog of a P that has a
        // forecast by R x its predicted drain).
        let params = crate::scheduler::co_schedule::TradeParams {
            wait: crate::scheduler::co_schedule::TradeWait::Ledger,
            ..params
        };
        let trigger = triggers.primary();
        let triggers_text = triggers.to_string();
        let mut first = true;
        loop {
            let pool = self.reserved_pool();
            let usable = Self::reserved_pool_usable(&pool);
            let step_started = Instant::now();
            // `room = predicted`: the E3 inputs of a cold one-chunk probe per
            // P, before the locks (fits, snapshots, pushed states).
            let e3 = (gwq.cfg.room == GlobalQueueRoom::Predicted && usable)
                .then(|| {
                    let probes: Vec<crate::cost_balance::Candidate> = pool
                        .iter()
                        .map(|w| crate::cost_balance::gwq_probe_candidate(&w.url))
                        .collect();
                    self.trade_e3_context(
                        &probes,
                        crate::cost_balance::GWQ_PROBE_TOKENS,
                        None,
                        None,
                    )
                })
                .flatten();
            let mut st = gwq.state.lock();
            if first {
                st.counters.passes += 1;
                first = false;
            }
            let gone = take_closed(&mut st.entries);
            st.counters.dropped_client_gone += gone.len() as u64;
            if st.entries.is_empty() || !usable {
                if st.entries.is_empty() {
                    st.last_queue_tokens = 0;
                } else {
                    st.counters.passes_pool_unusable += 1;
                }
                outcome.held = st.entries.len();
                drop(st);
                self.gwq_trace_drops(&gone, "client_gone", None);
                return outcome;
            }
            let urls: Vec<String> = pool.iter().map(|w| w.url.clone()).collect();
            for entry in st.entries.iter_mut() {
                if entry.cands_pool != urls {
                    entry.cands =
                        self.reserved_candidates(&pool, entry.answer.as_ref(), entry.tokens_len);
                    entry.cands_pool = urls.clone();
                }
            }
            let depth = vec![gwq.depth_tokens; urls.len()];
            let now = Instant::now();
            let step = {
                let held: Vec<crate::cost_balance::GwqHeld<'_>> = st
                    .entries
                    .iter()
                    .map(|e| crate::cost_balance::GwqHeld {
                        input_tokens: e.tokens_len,
                        rid: e.rid.as_deref(),
                        candidates: e.cands.as_slice(),
                        arrived: e.arrived,
                        arrived_unix_ms: e.arrived_unix_ms,
                        session: e.session,
                        session_label: None,
                        p_arrival: e.p_arrival.as_deref(),
                        arrival_pass: e.arrival_pass == pass_id,
                    })
                    .collect();
                book.gwq_step(
                    params,
                    &held,
                    &crate::cost_balance::GwqStepArgs {
                        mode: gwq.cfg.mode,
                        urls: &urls,
                        depth: &depth,
                        stale_after: gwq.stale_after,
                        max_wait: gwq.max_wait,
                        pass_tokens: crate::global_queue::CHUNK_TOKENS,
                        auto: None,
                        now,
                        pass_id,
                        trigger: trigger.as_str(),
                        triggers: &triggers_text,
                        e3: e3.as_ref(),
                        lead_s: gwq.lead_s,
                        sources: &[],
                    },
                )
            };
            let step = match step {
                Ok(step) => step,
                Err(error) => {
                    // Admission is stopped (cost trace failure): every held
                    // request fails, as every new arrival does.
                    let failed: Vec<GwqEntry> = st.entries.drain(..).collect();
                    st.counters.dropped_admission_stopped += failed.len() as u64;
                    st.last_queue_tokens = 0;
                    drop(st);
                    tracing::error!(target: "cost_balance", %error, held = failed.len(),
                        "cost reservation failed; the global waiting queue fails its held requests");
                    self.gwq_trace_drops(&gone, "client_gone", None);
                    self.gwq_trace_drops(&failed, "admission_stopped", Some(error.as_str()));
                    return outcome;
                }
            };
            for (entry, p) in st.entries.iter_mut().zip(&step.plain) {
                if entry.p_arrival.is_none() {
                    entry.p_arrival = urls.get(*p).cloned();
                }
            }
            st.last_queue_tokens = step.queue_tokens;
            let Some((index, result)) = step.release else {
                outcome.held = st.entries.len();
                drop(st);
                self.gwq_trace_drops(&gone, "client_gone", None);
                return outcome;
            };
            let entry = st.entries.remove(index);
            let queue_depth = st.entries.len();
            let released = match result {
                Ok(released) => {
                    let counters = &mut st.counters;
                    if entry.arrival_pass == pass_id {
                        counters.released_at_arrival += 1;
                    } else {
                        counters.released_after_hold += 1;
                    }
                    match released.release {
                        GwqRelease::Room => counters.released_room += 1,
                        GwqRelease::StaleView => counters.released_stale_view += 1,
                        GwqRelease::Starved => counters.released_starved += 1,
                        GwqRelease::Predicted => counters.released_predicted += 1,
                    }
                    if released.late_bound {
                        counters.late_bound += 1;
                    }
                    released
                }
                Err(error) => {
                    st.counters.dropped_reserve_error += 1;
                    drop(st);
                    tracing::error!(target: "cost_balance", %error,
                        rid = entry.rid.as_deref().unwrap_or(""),
                        "cost reservation failed for a request of the global waiting queue");
                    self.gwq_trace_drops(&gone, "client_gone", None);
                    self.gwq_trace_drops(
                        std::slice::from_ref(&entry),
                        "reserve_error",
                        Some(error.as_str()),
                    );
                    continue;
                }
            };
            drop(st);
            self.gwq_trace_drops(&gone, "client_gone", None);
            let crate::cost_balance::GwqReleased {
                candidate,
                reservation,
                decision,
                record,
                release,
                starved,
                late_bound,
            } = released;
            let Some(worker) = pool.iter().find(|w| w.url == candidate.worker_url).cloned() else {
                // The step reserves only candidates of this pool.
                tracing::error!(target: "co_sched", worker = %candidate.worker_url,
                    "global waiting queue: the released P is not in the pool; the request fails");
                continue;
            };
            let hold_ms = now.saturating_duration_since(entry.arrived).as_millis() as u64;
            tracing::info!(target: "co_sched",
                rid = entry.rid.as_deref().unwrap_or(""),
                worker = %worker.url,
                held = entry.arrival_pass != pass_id,
                hold_ms,
                release = release.as_str(),
                starved,
                late_bound,
                p_arrival = record["p_arrival"].as_str().unwrap_or(""),
                key = record["key"].as_f64().unwrap_or(0.0),
                service_s = record["service_s"].as_f64().unwrap_or(0.0),
                rank = record["rank"].as_u64().unwrap_or(0),
                queue_len = record["queue_len"].as_u64().unwrap_or(0),
                queued_tokens = record["queued_tokens"].as_str().unwrap_or(""),
                virtual_tokens = record["virtual_tokens"].as_str().unwrap_or(""),
                depth_tokens = gwq.depth_tokens,
                mode = gwq.cfg.mode.as_str(),
                trigger = trigger.as_str(),
                triggers = triggers_text.as_str(),
                "co_schedule gwq release");
            // A request released by the pass of its own arrival was not held
            // (as TRADE without the queue: 0); the trace has the exact time.
            let queue_wait_ms = if entry.arrival_pass == pass_id {
                0
            } else {
                hold_ms
            };
            let dispatch = self.finish_reserved(
                self_arc,
                worker,
                candidate,
                reservation,
                Some(decision),
                entry.answer.as_ref(),
                entry.rid.as_deref(),
                entry.tokens_len,
                step_started,
                queue_wait_ms,
                triggers,
                queue_depth,
                entry.e3_session.clone(),
            );
            outcome.released += 1;
            if entry.tx.send(dispatch).is_err() {
                // The client left after the step: the unsent reservation
                // rolls back when the dispatch drops here.
                gwq.state.lock().counters.send_failed += 1;
            }
        }
    }

    /// Take held request `id` out of the queue (its client left, or its wait
    /// budget ran out) and record the drop. Nothing happens when it has
    /// already left the queue.
    fn gwq_drop(&self, id: u64, reason: &'static str) {
        let Some(gwq) = self.gwq.as_ref() else { return };
        let entry = {
            let mut st = gwq.state.lock();
            let Some(index) = st.entries.iter().position(|e| e.id == id) else {
                return;
            };
            if reason == "wait_budget" {
                st.counters.dropped_wait_budget += 1;
            } else {
                st.counters.dropped_client_gone += 1;
            }
            st.entries.remove(index)
        };
        self.gwq_trace_drops(std::slice::from_ref(&entry), reason, None);
    }

    /// One `gwq_drop` trace event and log line per request that left the
    /// queue without a release.
    fn gwq_trace_drops(&self, entries: &[GwqEntry], reason: &str, error: Option<&str>) {
        let Some(book) = self.cost_book.as_ref() else {
            return;
        };
        let now = Instant::now();
        for entry in entries {
            let hold_s = now.saturating_duration_since(entry.arrived).as_secs_f64();
            tracing::info!(target: "co_sched", rid = entry.rid.as_deref().unwrap_or(""),
                reason, hold_s, "co_schedule gwq drop");
            let event = serde_json::json!({
                "event": "gwq_drop",
                "rid": entry.rid,
                "reason": reason,
                "hold_s": hold_s,
                "arrived_unix_ms": entry.arrived_unix_ms,
                "error": error,
            });
            if let Err(error) = book.record_event(event) {
                tracing::warn!(target: "co_sched", %error, "gwq_drop event not written");
            }
        }
    }

    /// The `gwq_stats` trace event (every second) and, with `log`, the
    /// `co_schedule gwq stats` line.
    fn gwq_emit_stats(&self, log: bool) {
        let (Some(gwq), Some(book)) = (self.gwq.as_ref(), self.cost_book.as_ref()) else {
            return;
        };
        let now = Instant::now();
        let urls: Vec<String> = self.reserved_pool().iter().map(|w| w.url.clone()).collect();
        let (queue_len, held_input_tokens, held_uncached_tokens, oldest_hold_s, counters) = {
            let st = gwq.state.lock();
            (
                st.entries.len(),
                st.entries
                    .iter()
                    .map(|e| u128::from(e.tokens_len))
                    .sum::<u128>(),
                st.last_queue_tokens,
                st.entries
                    .first()
                    .map(|e| now.saturating_duration_since(e.arrived).as_secs_f64()),
                st.counters,
            )
        };
        let workers: Vec<Value> = urls
            .iter()
            .zip(book.gwq_rows(&urls, now, crate::global_queue::CHUNK_TOKENS))
            .map(|(url, row)| {
                let fresh = row
                    .view_age_ms
                    .is_some_and(|ms| Duration::from_millis(ms) <= gwq.stale_after);
                serde_json::json!({
                    "worker_url": url,
                    "queued_tokens": row.queued.to_string(),
                    "unlaunched_tokens": row.unlaunched.to_string(),
                    "launched_tokens": row.launched.to_string(),
                    "room": !fresh || row.queued < u128::from(gwq.depth_tokens),
                    "view_fresh": fresh,
                    "view_age_ms": row.view_age_ms,
                })
            })
            .collect();
        let event = serde_json::json!({
            "event": "gwq_stats",
            "mode": gwq.cfg.mode.as_str(),
            "queue_len": queue_len,
            "held_input_tokens": held_input_tokens.to_string(),
            "held_uncached_tokens": held_uncached_tokens.to_string(),
            "oldest_hold_s": oldest_hold_s,
            "depth_tokens": gwq.depth_tokens,
            "workers": workers,
            "counters": counters.value(),
        });
        if let Err(error) = book.record_event(event) {
            tracing::warn!(target: "co_sched", %error, "gwq_stats event not written");
        }
        if log {
            tracing::info!(target: "co_sched",
                mode = gwq.cfg.mode.as_str(),
                queue_len,
                oldest_hold_s = oldest_hold_s.unwrap_or(0.0),
                held_uncached_tokens = %held_uncached_tokens,
                counters = %counters.value(),
                "co_schedule gwq stats");
        }
    }

    /// Held requests (tests).
    #[cfg(test)]
    fn gwq_len(&self) -> usize {
        self.gwq
            .as_ref()
            .map_or(0, |g| g.state.lock().entries.len())
    }

    /// Queue + wait, with a prebuilt indexer answer. Split from [`admit`]
    /// so tests can drive the scheduler without a live indexer.
    pub async fn enqueue(
        &self,
        self_arc: &Arc<Self>,
        tokens_len: u64,
        answer: Option<IndexerAnswer>,
        rid: Option<String>,
    ) -> Result<Dispatch, AdmitError> {
        let hits: HashMap<String, u32> = answer
            .as_ref()
            .map(|a| {
                a.candidates
                    .iter()
                    .map(|c| (c.worker_url.clone(), c.expected_hit_tokens))
                    .collect()
            })
            .unwrap_or_default();
        let (tx, rx) = oneshot::channel();
        let gpu_hits = if self.cfg.prefer_gpu_hit {
            answer
                .as_ref()
                .map(|a| a.observed_gpu_hits.clone())
                .unwrap_or_default()
        } else {
            HashMap::new()
        };
        {
            let mut st = self.state.lock();
            st.queue.push_back(QueueEntry {
                arrived: Instant::now(),
                tokens_len,
                hits,
                gpu_hits,
                answer,
                rid,
                tx,
            });
        }
        // The arrival pass runs inline, immediately, on the arriving
        // request's task (unchanged behavior, now labeled). The dispatcher
        // is also woken as a safety net.
        self.drain(self_arc, Triggers::ARRIVAL);
        self.wake(Triggers::ARRIVAL);

        // The starvation guard fires at max_queue_wait_ms; this outer bound
        // only trips when even forced dispatch is impossible (no healthy
        // workers at all) — treat that as a wedge, not a hang.
        let budget = Duration::from_millis(self.cfg.max_queue_wait_ms * 2 + 10_000);
        match tokio::time::timeout(budget, rx).await {
            Ok(Ok(d)) => Ok(d),
            // Sender dropped without dispatch (scheduler torn down).
            Ok(Err(_)) => Err(AdmitError::Wedged),
            Err(_) => {
                // Best effort: the entry stays queued; it will be dropped
                // whenever it reaches the front and its receiver is gone.
                Err(AdmitError::Wedged)
            }
        }
    }

    /// The routable pool right now (prefill pool in PD mode, plain workers
    /// otherwise) — same resolution the routes use.
    fn pool(&self) -> Vec<Arc<Worker>> {
        PdPoolResolver::new(Arc::clone(&self.registry))
            .prefill_candidates(&self.model)
            .unwrap_or_default()
    }

    /// Record why the dispatcher should run and wake it. Bits accumulate
    /// until the dispatcher swaps them out, so coalesced events keep every
    /// label. Touches no `state` → safe to call with the state lock held.
    fn wake(&self, t: Triggers) {
        self.pending.fetch_or(t.0, Ordering::AcqRel);
        self.notify.notify_one();
    }

    fn backlog_of(st: &mut SchedState, url: &str) -> u64 {
        st.workers.get_mut(url).map_or(0, |w| {
            w.prune();
            w.backlog()
        })
    }

    /// Run the dispatch rules until nothing more can be dispatched. Every
    /// dispatch of this pass is stamped with `triggers`. Returns what the
    /// pass did so tests (and callers) can assert synchronously.
    pub(crate) fn drain(&self, self_arc: &Arc<Self>, triggers: Triggers) -> PassOutcome {
        let pool = self.pool();
        if pool.is_empty() {
            return PassOutcome {
                dispatched: 0,
                held: self.state.lock().queue.len(),
            };
        }
        let cap = self.cfg.max_uncached_tokens;
        let max_wait = Duration::from_millis(self.cfg.max_queue_wait_ms);
        let pass_id = self.pass_counter.fetch_add(1, Ordering::Relaxed) + 1;
        let mut dispatched = 0u32;

        loop {
            let mut st = self.state.lock();
            if st.queue.is_empty() {
                return PassOutcome {
                    dispatched,
                    held: 0,
                };
            }

            let backlogs: HashMap<String, u64> = pool
                .iter()
                .map(|w| (w.url.clone(), Self::backlog_of(&mut st, &w.url)))
                .collect();
            let min_backlog_worker = pool
                .iter()
                .min_by_key(|w| backlogs.get(&w.url).copied().unwrap_or(0))
                .cloned();
            // Imbalance metric at this iteration's start, logged per
            // dispatch and driving the Step 3 imbalance escape.
            let spread_tokens = backlogs
                .values()
                .max()
                .copied()
                .unwrap_or(0)
                .saturating_sub(backlogs.values().min().copied().unwrap_or(0));
            let ctx = PassCtx {
                triggers,
                pass_id,
                spread_tokens,
            };

            // Starvation guard first: the oldest over-budget entry gets the
            // min-backlog worker unconditionally.
            let forced_idx = st
                .queue
                .iter()
                .position(|e| e.arrived.elapsed() >= max_wait);
            if let Some(idx) = forced_idx {
                let worker = min_backlog_worker.expect("pool non-empty");
                let entry = st.queue.remove(idx).expect("index valid");
                drop(st);
                self.dispatch(self_arc, entry, worker, Branch::Forced, &backlogs, ctx);
                dispatched += 1;
                continue;
            }

            // Cache-affinity scan: first entry whose own A is under the cap.
            let direct = st.queue.iter().enumerate().find_map(|(i, e)| {
                let a = Self::best_hit_worker(&pool, &e.hits, &e.gpu_hits, &backlogs)?;
                (backlogs.get(&a.url).copied().unwrap_or(0) < cap).then_some((i, a))
            });
            if let Some((idx, worker)) = direct {
                let entry = st.queue.remove(idx).expect("index valid");
                drop(st);
                self.dispatch(self_arc, entry, worker, Branch::Direct, &backlogs, ctx);
                dispatched += 1;
                continue;
            }

            // Head transfer: any under-cap worker takes the head request,
            // preferring the one that already holds the most of ITS prefix.
            let head_hits = st.queue.front().map(|e| e.hits.clone()).unwrap_or_default();
            let head_gpu_hits = st
                .queue
                .front()
                .map(|e| e.gpu_hits.clone())
                .unwrap_or_default();
            let eligible: Vec<Arc<Worker>> = pool
                .iter()
                .filter(|w| backlogs.get(&w.url).copied().unwrap_or(0) < cap)
                .cloned()
                .collect();
            if !eligible.is_empty() {
                let worker =
                    Self::best_hit_worker(&eligible, &head_hits, &head_gpu_hits, &backlogs)
                        .expect("eligible non-empty");
                let entry = st.queue.pop_front().expect("queue non-empty");
                drop(st);
                self.dispatch(self_arc, entry, worker, Branch::Transfer, &backlogs, ctx);
                dispatched += 1;
                continue;
            }

            // Imbalance escape: every worker is over the cap, but if the
            // pool is skewed beyond the spread threshold the head goes to
            // the min-backlog worker anyway instead of holding (0 = off).
            let spread_threshold = self.cfg.imbalance_spread_tokens;
            if spread_threshold > 0 && spread_tokens > spread_threshold {
                let worker = min_backlog_worker.expect("pool non-empty");
                let entry = st.queue.pop_front().expect("queue non-empty");
                drop(st);
                self.dispatch(self_arc, entry, worker, Branch::Imbalance, &backlogs, ctx);
                dispatched += 1;
                continue;
            }

            // Everyone over the cap: hold. The dispatcher task retries on
            // ledger retires, poll updates (periodic, or every
            // `poll_on_hold_ms` while held), arrivals, the oldest entry's
            // starvation deadline, and the `tick_ms` fallback.
            let held = st.queue.len();
            drop(st);
            if self.cfg.poll_on_hold_ms > 0 {
                // Fresh load data for the next pass: ask the poller for an
                // early /v1/loads round (rate-limited there).
                self.poll_now_signals.fetch_add(1, Ordering::Relaxed);
                self.poll_now.notify_one();
            }
            return PassOutcome { dispatched, held };
        }
    }

    /// Highest-hit worker in `pool` (ties and the all-zero-hits degenerate
    /// case resolve to the lowest backlog). An enabled GPU observation breaks
    /// equal total-hit ties before backlog. `None` only for an empty pool.
    fn best_hit_worker(
        pool: &[Arc<Worker>],
        hits: &HashMap<String, u32>,
        gpu_hits: &HashMap<String, u32>,
        backlogs: &HashMap<String, u64>,
    ) -> Option<Arc<Worker>> {
        pool.iter()
            .max_by(|x, y| {
                let hx = hits.get(&x.url).copied().unwrap_or(0);
                let hy = hits.get(&y.url).copied().unwrap_or(0);
                let gx = gpu_hits.get(&x.url).copied().unwrap_or(0).min(hx);
                let gy = gpu_hits.get(&y.url).copied().unwrap_or(0).min(hy);
                hx.cmp(&hy).then_with(|| gx.cmp(&gy)).then_with(|| {
                    // Lower backlog wins the tie → reverse the ordering.
                    let bx = backlogs.get(&x.url).copied().unwrap_or(0);
                    let by = backlogs.get(&y.url).copied().unwrap_or(0);
                    by.cmp(&bx)
                })
            })
            .cloned()
    }

    fn dispatch(
        &self,
        self_arc: &Arc<Self>,
        entry: QueueEntry,
        worker: Arc<Worker>,
        branch: Branch,
        backlogs: &HashMap<String, u64>,
        ctx: PassCtx,
    ) {
        let hit_here = entry.hits.get(&worker.url).copied().unwrap_or(0);
        let hint = match (self.cfg.pull_scope, branch) {
            (KvPullScope::DetourOnly, Branch::Direct) => None,
            _ => entry
                .answer
                .as_ref()
                .and_then(|a| self.injector.pick(a, &worker.url)),
        };
        let uncached = entry.tokens_len.saturating_sub(hit_here as u64);
        let ledger_entry = Arc::new(LedgerEntry {
            tokens: uncached,
            dispatched_at: Instant::now(),
            retired: AtomicBool::new(false),
        });
        let poll_age_ms = {
            let mut st = self.state.lock();
            let ws = st.workers.entry(worker.url.clone()).or_default();
            ws.ledger.push(Arc::clone(&ledger_entry));
            ws.poll_age_ms()
        };
        let queue_wait_ms = entry.arrived.elapsed().as_millis() as u64;
        let handle = LedgerHandle {
            entry: ledger_entry,
            sched: Arc::downgrade(self_arc),
        };
        let (a_worker, best_hit) = entry
            .hits
            .iter()
            .max_by_key(|(_, t)| **t)
            .map(|(u, t)| (u.clone(), *t))
            .unwrap_or_default();
        let trigger = ctx.triggers.primary();
        tracing::info!(
            target: "co_sched",
            rid = entry.rid.as_deref().unwrap_or(""),
            branch = branch.as_str(),
            trigger = trigger.as_str(),
            triggers = %ctx.triggers,
            pass_id = ctx.pass_id,
            worker = %worker.url,
            a_worker = %a_worker,
            hit_here = hit_here,
            gpu_hit_here = entry.gpu_hits.get(&worker.url).copied(),
            best_hit = best_hit,
            uncached_tokens = uncached,
            backlog_before = backlogs.get(&worker.url).copied().unwrap_or(0),
            spread_tokens = ctx.spread_tokens,
            poll_age_ms = poll_age_ms,
            injected = hint.is_some(),
            source = hint
                .as_ref()
                .and_then(|h| h.get("source_worker_id"))
                .and_then(|v| v.as_str())
                .unwrap_or(""),
            queue_wait_ms = queue_wait_ms,
            queue_depth = self.state.lock().queue.len(),
            candidates = entry.answer.as_ref().map(|a| a.candidates.len()).unwrap_or(0),
            "co_schedule dispatch",
        );
        // Also emit the standard `kv_pull_hint decision` line so the
        // existing probe/canary log parsers work unchanged on this policy.
        self.injector.log_decision(
            entry.answer.as_ref(),
            "CoSchedule",
            &worker.url,
            hint.as_ref(),
        );
        let dispatch = Dispatch {
            cost_reservation: None,
            worker,
            hint,
            ledger: handle,
            branch,
            queue_wait_ms,
            trigger,
        };
        if let Err(dead) = entry.tx.send(dispatch) {
            // Client went away while queued — retire the just-created
            // ledger entry so the phantom request frees its backlog.
            dead.ledger.retire();
            tracing::debug!(target: "co_sched", "queued client gone before dispatch");
        }
    }

    /// Apply one `/v1/loads` poll result: reset the baseline and retire
    /// ledger entries the engine has had a chance to observe (dispatched
    /// before the poll was sent). Periodic and poll-on-hold rounds
    /// interleave, so a sample older than the applied baseline is dropped.
    fn apply_poll(&self, url: &str, uncached: u64, sent_at: Instant) {
        let mut st = self.state.lock();
        let ws = st.workers.entry(url.to_string()).or_default();
        if ws.polled_sent_at.is_some_and(|prev| sent_at < prev) {
            tracing::debug!(
                target: "co_sched",
                worker = url,
                "out-of-order /v1/loads sample ignored",
            );
            return;
        }
        ws.polled_sent_at = Some(sent_at);
        ws.polled_uncached = uncached;
        ws.polled_at = Some(Instant::now());
        for e in &ws.ledger {
            if e.dispatched_at <= sent_at {
                e.retired.store(true, Ordering::Relaxed);
            }
        }
        ws.prune();
    }

    #[cfg(test)]
    fn queue_depth(&self) -> usize {
        self.state.lock().queue.len()
    }
}

/// Remove the held requests whose client has left (their admission future,
/// and so the receiver, is gone).
fn take_closed(entries: &mut Vec<GwqEntry>) -> Vec<GwqEntry> {
    let mut gone = Vec::new();
    let mut i = 0;
    while i < entries.len() {
        if entries[i].tx.is_closed() {
            gone.push(entries.remove(i));
        } else {
            i += 1;
        }
    }
    gone
}

/// The release passes of the global waiting queue outside the arrivals:
/// woken by applied progress polls and terminal cost-book statuses (the
/// coalesced [`Triggers`]), by the tick while requests are held, and exactly
/// at the oldest held request's max-wait deadline.
async fn gwq_dispatcher_loop(weak: Weak<CoScheduler>) {
    loop {
        let Some(sched) = weak.upgrade() else { return };
        let Some(gwq) = sched.gwq.as_ref() else {
            return;
        };
        let (oldest, held) = {
            let st = gwq.state.lock();
            (st.entries.first().map(|e| e.arrived), st.entries.len())
        };
        let now = Instant::now();
        let deadline =
            oldest.map(|arrived| (arrived + gwq.max_wait).saturating_duration_since(now));
        let (sleep_for, timer_is_deadline) = match deadline {
            // A deadline within one tick: wake exactly then. A deadline in
            // the past means the last pass could not release (no usable
            // pool): fall back to the tick.
            Some(until) if !until.is_zero() && until <= gwq.tick => (until, true),
            _ if held > 0 => (gwq.tick, false),
            _ => (GWQ_IDLE_TICK, false),
        };
        let notified = sched.notify.notified();
        let timer_fired = tokio::select! {
            _ = notified => false,
            _ = tokio::time::sleep(sleep_for) => true,
        };
        let mut bits = Triggers(sched.pending.swap(0, Ordering::AcqRel));
        if timer_fired {
            bits = bits.or(if timer_is_deadline {
                Triggers::STARVATION
            } else {
                Triggers::TICK
            });
        }
        if bits.is_empty() {
            bits = Triggers::TICK;
        }
        let pass_id = sched.pass_counter.fetch_add(1, Ordering::Relaxed) + 1;
        let held_after = match std::panic::catch_unwind(AssertUnwindSafe(|| {
            sched.gwq_pass(&sched, bits, pass_id)
        })) {
            Ok(outcome) => outcome.held,
            Err(_) => {
                tracing::error!(target: "co_sched",
                    "global waiting queue pass panicked; the dispatcher continues");
                0
            }
        };
        if held_after > 0 && sched.cfg.on_demand_load_refresh {
            // No periodic poller in this mode: refresh while holding.
            if let Some(gwq) = sched.gwq.as_ref() {
                gwq.state.lock().counters.on_demand_refreshes += 1;
            }
            drop(sched.refresh_progress(false));
        }
        drop(sched);
    }
}

/// The `gwq_stats` trace event every second; the log line every 10 s.
async fn gwq_stats_loop(weak: Weak<CoScheduler>) {
    let mut interval = tokio::time::interval(Duration::from_secs(1));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut ticks = 0u64;
    loop {
        interval.tick().await;
        let Some(sched) = weak.upgrade() else { return };
        sched.gwq_emit_stats(ticks.is_multiple_of(10));
        ticks += 1;
        drop(sched);
    }
}

async fn dispatcher_loop(weak: Weak<CoScheduler>) {
    loop {
        let Some(sched) = weak.upgrade() else { return };
        let tick = Duration::from_millis(sched.cfg.tick_ms);
        let max_wait = Duration::from_millis(sched.cfg.max_queue_wait_ms);
        // push_back-only queue (removals never reorder): the front is the
        // oldest entry, so its deadline is the earliest starvation deadline.
        let deadline = sched
            .state
            .lock()
            .queue
            .front()
            .map(|e| e.arrived + max_wait);
        let now = Instant::now();
        let (sleep_for, timer_is_deadline) =
            match deadline.map(|d| d.saturating_duration_since(now)) {
                // Deadline within one tick: wake exactly then (the starvation
                // guard must not be quantized by the tick). A deadline already
                // in the past means the pass that just ran could not
                // force-dispatch (empty pool) — fall back to the tick rather
                // than spinning.
                Some(u) if !u.is_zero() && u <= tick => (u, true),
                _ => (tick, false),
            };
        let notified = sched.notify.notified();
        let timer_fired = tokio::select! {
            _ = notified => false,
            _ = tokio::time::sleep(sleep_for) => true,
        };
        let mut bits = Triggers(sched.pending.swap(0, Ordering::AcqRel));
        if timer_fired {
            bits = bits.or(if timer_is_deadline {
                Triggers::STARVATION
            } else {
                Triggers::TICK
            });
        }
        if bits.is_empty() {
            bits = Triggers::TICK; // late/spurious permit: still a full pass
        }
        // The tick is the only fallback; a panicking pass must not silently
        // kill the dispatcher (parking_lot does not poison).
        if std::panic::catch_unwind(AssertUnwindSafe(|| sched.drain(&sched, bits))).is_err() {
            tracing::error!(
                target: "co_sched",
                "co_schedule drain panicked; dispatcher continues",
            );
        }
        // Drop the strong ref each iteration so context teardown frees us.
        drop(sched);
    }
}

async fn poller_loop(weak: Weak<CoScheduler>) {
    loop {
        let Some(sched) = weak.upgrade() else { return };
        let period = Duration::from_millis(sched.cfg.load_poll_ms.max(100));
        let on_hold = Duration::from_millis(sched.cfg.poll_on_hold_ms);
        let round_started = Instant::now();
        for w in sched.pool() {
            let sched = Arc::clone(&sched);
            tokio::spawn(async move {
                let sent_at = Instant::now();
                match fetch_waiting_uncached(&sched.poll_client, &w.url).await {
                    Ok(v) => {
                        sched.apply_poll(&w.url, v, sent_at);
                        sched.wake(Triggers::POLL);
                    }
                    Err(e) => tracing::debug!(
                        target: "co_sched",
                        worker = %w.url,
                        error = %e,
                        "/v1/loads poll failed; ledger-only until it recovers",
                    ),
                }
            });
        }
        // `poll_now` is an Arc so the strong scheduler ref can be dropped
        // while waiting (teardown semantics unchanged) — a `Notified`
        // borrowed from `sched` would not compile past `drop(sched)`.
        let poll_now = Arc::clone(&sched.poll_now);
        drop(sched);
        tokio::select! {
            _ = tokio::time::sleep(period) => {}
            // A HOLD pass asked for fresh loads: start the next round
            // `poll_on_hold_ms` after this one started (rate limit). A
            // permit stored while we were mid-round is consumed here, so a
            // HOLD signalled during the fan-out is not lost.
            _ = poll_now.notified(), if !on_hold.is_zero() => {
                tokio::time::sleep_until((round_started + on_hold).into()).await;
            }
        }
    }
}

/// Progress updates only: no HOLD queue, dispatcher wake or admission timer.
/// Each worker has at most one outstanding request in a polling round.
async fn progress_poller_loop(weak: Weak<CoScheduler>) {
    let Some(sched) = weak.upgrade() else { return };
    let mut interval = tokio::time::interval(Duration::from_millis(sched.cfg.load_poll_ms));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    drop(sched);
    loop {
        interval.tick().await;
        let Some(sched) = weak.upgrade() else { return };
        futures::future::join_all(sched.refresh_progress(true)).await;
    }
}

/// TRADE `wait = e3` with the state push: every [`PUSH_SUPERVISE`], keep
/// one push subscriber per P of the routable pool (new P workers get one,
/// gone ones lose theirs) and write the push stats when they are due.
async fn push_supervisor_loop(weak: Weak<CoScheduler>) {
    let mut interval = tokio::time::interval(PUSH_SUPERVISE);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        interval.tick().await;
        let Some(sched) = weak.upgrade() else { return };
        let Some(hub) = sched.e3_push.clone() else {
            return;
        };
        let urls: Vec<String> = sched.pool().iter().map(|w| w.url.clone()).collect();
        drop(sched);
        hub.ensure(&urls);
        hub.maybe_emit_stats(Instant::now());
    }
}

/// A `/v1/loads` poll that the engine answered with HTTP 4xx or 500. An
/// engine without the `prefill_routing` section (an older
/// engine) answers `include=...,prefill_routing` this way: its load-snapshot
/// section check raises ValueError (400 or 500). The same holds for an
/// engine without the `host_room` section.
pub(crate) fn include_rejected(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<reqwest::Error>()
        .and_then(reqwest::Error::status)
        .is_some_and(|status| {
            status.is_client_error() || status == reqwest::StatusCode::INTERNAL_SERVER_ERROR
        })
}

/// The fields of one `/v1/loads` progress poll ([`fetch_progress_poll`]):
/// the prefill-work progress, the rest of the load row (the trace's
/// `engine_load`), the raw `prefill_routing` snapshot and the raw `host_room`
/// section.
pub(crate) type ProgressPoll = (
    crate::cost_balance::PrefillProgress,
    Value,
    Option<Value>,
    Option<Value>,
);

/// One `/v1/loads` poll: the prefill-work progress, the rest of the load
/// row for the trace and, with `routing`, the raw `prefill_routing` snapshot.
/// The snapshot is removed from the load row in every case, so the trace's
/// `engine_load` keeps its size.
pub(crate) async fn fetch_prefill_progress(
    client: &reqwest::Client,
    base: &str,
    routing: bool,
) -> anyhow::Result<(crate::cost_balance::PrefillProgress, Value, Option<Value>)> {
    let (progress, load, snapshot, _) = fetch_progress_poll(client, base, routing, false).await?;
    Ok((progress, load, snapshot))
}

/// [`fetch_prefill_progress`] that, with `host_room`, also asks for the
/// engine's `host_room` section (its host pool by retention class), removes
/// it from the load row and returns it. Without `host_room` the query and
/// the load row are those of [`fetch_prefill_progress`].
pub(crate) async fn fetch_progress_poll(
    client: &reqwest::Client,
    base: &str,
    routing: bool,
    host_room: bool,
) -> anyhow::Result<ProgressPoll> {
    let include = match (routing, host_room) {
        (false, false) => "core,queues,prefill_work",
        (true, false) => "core,queues,prefill_work,prefill_routing",
        (false, true) => "core,queues,prefill_work,host_room",
        (true, true) => "core,queues,prefill_work,prefill_routing,host_room",
    };
    let mut value: Value = client
        .get(format!("{}/v1/loads", base.trim_end_matches('/')))
        .query(&[("include", include)])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let loads = value
        .get_mut("loads")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| anyhow::anyhow!("missing loads[]"))?;
    // TP shares one scheduler RID namespace. Supporting DP requires explicit
    // per-rank cursors and routing; never silently take an arbitrary rank.
    anyhow::ensure!(
        loads.len() == 1,
        "uncached balance requires DP=1 per worker"
    );
    let mut load = loads.swap_remove(0);
    let progress = load
        .as_object_mut()
        .and_then(|row| row.remove("prefill_work"))
        .ok_or_else(|| anyhow::anyhow!("engine must enable --enable-prefill-work-snapshot"))?;
    let snapshot = load
        .as_object_mut()
        .and_then(|row| row.remove("prefill_routing"))
        .filter(|_| routing);
    let room = if host_room {
        load.as_object_mut().and_then(|row| row.remove("host_room"))
    } else {
        None
    };
    Ok((serde_json::from_value(progress)?, load, snapshot, room))
}

/// GET `{worker}/v1/loads` and sum `num_waiting_uncached_tokens` across DP
/// ranks. The field is live queue depth net of prefix-cache hits (see
/// sglang `SchedulerLoadInquirer.get_num_waiting_uncached_tokens`).
async fn fetch_waiting_uncached(client: &reqwest::Client, base: &str) -> anyhow::Result<u64> {
    let url = format!("{}/v1/loads", base.trim_end_matches('/'));
    let v: Value = client
        .get(&url)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let loads = v
        .get("loads")
        .and_then(|l| l.as_array())
        .ok_or_else(|| anyhow::anyhow!("missing loads[]"))?;
    let values: Vec<u64> = loads
        .iter()
        .filter_map(|l| {
            l.get("num_waiting_uncached_tokens")
                .and_then(|n| n.as_u64())
        })
        .collect();
    if values.is_empty() {
        // Empty array / field absent on every rank: treat as malformed
        // rather than a real 0 — the caller keeps the previous baseline.
        anyhow::bail!("no num_waiting_uncached_tokens in loads[]");
    }
    Ok(values.iter().sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{WorkerId, WorkerMode, WorkerSpec};
    use crate::kv_hint::{HintCandidate, KvWorkerMapEntry};

    /// Tests pin the RULE, not the CLI defaults: cap 10_240 and tick 100 ms
    /// keep the pre-existing tests' semantics and wall-clock times, and
    /// poll-on-hold / imbalance are off unless a test opts in.
    fn test_cfg(pull_scope: KvPullScope, max_wait_ms: u64) -> CoScheduleConfig {
        CoScheduleConfig {
            on_demand_load_refresh: false,
            uncached_balance: false,
            owner_threshold_tokens: 12288,
            detour_margin_tokens: 0,
            restore_cost_divisor: None,
            trade: None,
            trade_eft_model: None,
            global_queue: None,
            trade_e3: None,
            prefer_gpu_hit: false,
            locality_margin_tokens: None,
            cost_balance: false,
            cost_trace: None,
            max_uncached_tokens: 10_240,
            max_queue_wait_ms: max_wait_ms,
            load_poll_ms: 3_600_000, // effectively off in tests
            tick_ms: 100,
            poll_on_hold_ms: 0,
            imbalance_spread_tokens: 0,
            pull_scope,
            kv_session_retention: false,
            kv_session_key_file: None,
        }
    }

    fn test_cfg_with(f: impl FnOnce(&mut CoScheduleConfig)) -> CoScheduleConfig {
        let mut c = test_cfg(KvPullScope::Any, 30_000);
        f(&mut c);
        c
    }

    #[tokio::test]
    async fn demand_queries_are_shared_and_stale_refresh_does_not_gate_admission() {
        use axum::{routing::get, Json, Router};
        use serde_json::json;
        let calls = Arc::new(AtomicU64::new(0));
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let app = Router::new().route("/v1/loads", get({
            let calls = calls.clone(); let started = started.clone(); let release = release.clone();
            move || {
                let calls = calls.clone(); let started = started.clone(); let release = release.clone();
                async move {
                    let sequence = calls.fetch_add(1, Ordering::SeqCst) + 1;
                    started.notify_one();
                    release.notified().await;
                    Json(json!({"loads":[{"prefill_work":{"epoch":"boot1", "sequence":sequence,
                                                       "remaining":{},"truncated":false}}]}))
                }
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
        let tmp = tempfile::tempdir().unwrap();
        let sched = CoScheduler::spawn(test_cfg_with(|cfg| {
            cfg.uncached_balance = true; cfg.on_demand_load_refresh = true;
            cfg.cost_trace = Some(tmp.path().join("cost.jsonl")); cfg.load_poll_ms = 100;
        }), &test_kv(), registry_with_prefills(&[&url]), ModelId("m".into()));
        // No unconditional query even with a ready registered worker.
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let requests: Vec<_> = (0..16).map(|i| {
            let sched = sched.clone();
            tokio::spawn(async move { sched.admit(&sched, None, 64, None, Some(format!("step{i}"))).await })
        }).collect();
        tokio::time::timeout(Duration::from_secs(1), started.notified()).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        release.notify_one();
        let first: Vec<_> = futures::future::join_all(requests).await.into_iter()
            .map(|r| r.unwrap().unwrap()).collect();
        assert_eq!(first.len(), 16);
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // Query 2 is held at the server. Admission still returns using the
        // initialized ledger; it does not wait for that query's response.
        let next = tokio::time::timeout(Duration::from_millis(100),
            sched.admit(&sched, None, 64, None, Some("next".into()))).await.unwrap().unwrap();
        tokio::time::timeout(Duration::from_secs(1), started.notified()).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let another = sched.admit(&sched, None, 64, None, Some("another".into())).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        release.notify_one();
        drop((first, next, another));
        server.abort();
    }

    fn test_kv() -> KvHintConfig {
        KvHintConfig {
            indexer_url: "http://indexer:8090".into(),
            min_tokens: 256,
            min_incremental_tokens: 2048,
            local_prefix_fallback: false,
            indexer_timeout_ms: 20,
            worker_map: vec![
                KvWorkerMapEntry {
                    id: 1,
                    url: "http://p0:30000".into(),
                    servicer: "p0:18100".into(),
                },
                KvWorkerMapEntry {
                    id: 2,
                    url: "http://p1:30000".into(),
                    servicer: "p1:18100".into(),
                },
            ],
        }
    }

    fn registry_with_prefills(urls: &[&str]) -> Arc<WorkerRegistry> {
        let reg = Arc::new(WorkerRegistry::default());
        for (i, url) in urls.iter().enumerate() {
            let _ = reg.add(WorkerSpec {
                id: WorkerId(format!("p{i}")),
                url: url.to_string(),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("m".into())],
                bootstrap_port: Some(8998),
            });
            let _ = reg.add(WorkerSpec {
                id: WorkerId(format!("d{i}")),
                url: format!("http://d{i}:30000"),
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId("m".into())],
                bootstrap_port: None,
            });
        }
        reg
    }

    fn answer(hits: &[(&str, u32)]) -> IndexerAnswer {
        let mut candidates: Vec<HintCandidate> = hits
            .iter()
            .map(|(url, t)| HintCandidate {
                worker_url: url.to_string(),
                servicer_addr: format!("{}:18100", url.trim_start_matches("http://")),
                expected_hit_tokens: *t,
            })
            .collect();
        candidates.sort_by(|a, b| b.expected_hit_tokens.cmp(&a.expected_hit_tokens));
        IndexerAnswer {
            candidates,
            observed_gpu_hits: HashMap::new(),
            indexer_ms: 0.5,
        }
    }

    fn set_polled(s: &Arc<CoScheduler>, url: &str, uncached: u64) {
        s.apply_poll(url, uncached, Instant::now());
    }

    #[tokio::test]
    async fn gpu_tie_break_is_opt_in_and_still_respects_the_cap() {
        for (enabled, hot_backlog, expected) in [
            (false, 8192, "http://p1:30000"),
            (true, 8192, "http://p0:30000"),
            (true, 10_240, "http://p1:30000"),
        ] {
            let cfg = test_cfg_with(|c| c.prefer_gpu_hit = enabled);
            let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
            let s = CoScheduler::spawn(cfg, &test_kv(), reg, ModelId("m".into()));
            set_polled(&s, "http://p0:30000", hot_backlog);
            set_polled(&s, "http://p1:30000", 0);
            let mut a = answer(&[("http://p0:30000", 8192), ("http://p1:30000", 8192)]);
            a.observed_gpu_hits = HashMap::from([
                ("http://p0:30000".into(), 8192),
                ("http://p1:30000".into(), 0),
            ]);
            let dispatch = s.enqueue(&s, 9000, Some(a), None).await.unwrap();
            assert_eq!(dispatch.worker.url, expected);
            // The CPU-available prefix is unchanged: no unnecessary peer pull.
            assert!(dispatch.hint.is_none());
        }
    }

    #[tokio::test]
    async fn gpu_tie_break_preserves_total_hit_priority_and_missing_data_fallback() {
        for (hot_total, observe_gpu) in [(4096, true), (8192, false)] {
            let cfg = test_cfg_with(|c| c.prefer_gpu_hit = true);
            let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
            let s = CoScheduler::spawn(cfg, &test_kv(), reg, ModelId("m".into()));
            set_polled(&s, "http://p0:30000", 8192);
            set_polled(&s, "http://p1:30000", 0);
            let mut a = answer(&[("http://p0:30000", hot_total), ("http://p1:30000", 8192)]);
            if observe_gpu {
                a.observed_gpu_hits
                    .insert("http://p0:30000".into(), hot_total);
            }
            let dispatch = s.enqueue(&s, 9000, Some(a), None).await.unwrap();
            assert_eq!(dispatch.worker.url, "http://p1:30000");
        }
    }

    #[test]
    fn cost_balance_uses_the_same_incremental_gate_as_the_forwarded_hint() {
        for (local_hit, expected_hit, hinted) in [(2049, 2049, false), (2048, 4096, true)] {
            let cfg = test_cfg_with(|c| c.cost_balance = true);
            let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
            let s = CoScheduler::spawn(cfg, &test_kv(), reg, ModelId("m".into()));
            let (_, _busy) = s
                .cost_book
                .as_ref()
                .unwrap()
                .reserve(
                    1_000_000,
                    None,
                    vec![crate::cost_balance::Candidate {
                        worker_url: "http://p0:30000".into(),
                        local_hit: 0,
                        gpu_hit: None,
                        expected_hit: 0,
                        hint: None,
                    }],
                )
                .unwrap();
            let d = s
                .admit_reserved(
                    &s,
                    8192,
                    Some(answer(&[
                        ("http://p0:30000", 4096),
                        ("http://p1:30000", local_hit),
                    ])),
                    Some("gate-test".into()),
                    None,
                    None,
                )
                .unwrap();
            assert_eq!(d.worker.url, "http://p1:30000");
            assert_eq!(d.hint.is_some(), hinted);
            let details = d.cost_reservation.as_ref().unwrap().debuginfo();
            assert_eq!(details["expected_hit"], expected_hit);
            assert_eq!(
                details["cost"],
                (8192u128 * (8192 - expected_hit) as u128).to_string()
            );
        }
    }

    #[test]
    fn cost_reservation_survives_legacy_poll_and_first_byte_retire() {
        for locality in [false, true] {
            let cfg = test_cfg_with(|c| {
                c.cost_balance = !locality;
                c.locality_margin_tokens = locality.then_some(32768);
                c.max_uncached_tokens = 1;
            });
            let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
            let s = CoScheduler::spawn(cfg, &test_kv(), reg, ModelId("m".into()));
            set_polled(&s, "http://p0:30000", u64::MAX);
            set_polled(&s, "http://p1:30000", u64::MAX);
            let mut d = s
                .admit_reserved(&s, 8192, None, Some("bootstrap-test".into()), None, None)
                .unwrap();
            assert_eq!(d.queue_wait_ms, 0);
            assert_eq!(s.queue_depth(), 0);
            assert_eq!(
                d.branch,
                if locality {
                    Branch::Locality
                } else {
                    Branch::CostBalanced
                }
            );
            d.cost_reservation
                .as_mut()
                .unwrap()
                .mark_dispatched()
                .unwrap();
            set_polled(&s, &d.worker.url, 0);
            d.ledger.retire();
            let book = s.cost_book.as_ref().unwrap();
            let snapshot = book.snapshot();
            let row = snapshot["workers"]
                .as_array()
                .unwrap()
                .iter()
                .find(|v| v["worker_url"] == d.worker.url)
                .unwrap();
            assert_eq!(row["active_requests"], 1);
            assert_eq!(row["cost"], "67108864");
            assert_eq!(row["pending_uncached_tokens"], "8192");
            d.cost_reservation
                .as_mut()
                .unwrap()
                .finish("test_prefill_complete")
                .unwrap();
            let snapshot = book.snapshot();
            let row = snapshot["workers"]
                .as_array()
                .unwrap()
                .iter()
                .find(|v| v["worker_url"] == d.worker.url)
                .unwrap();
            assert_eq!(row["active_requests"], 0);
            assert_eq!(row["cost"], "0");
            assert_eq!(row["pending_uncached_tokens"], "0");
        }
    }

    #[tokio::test]
    async fn direct_dispatch_to_max_hit_under_cap() {
        let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
        let s = CoScheduler::spawn(
            test_cfg(KvPullScope::Any, 30_000),
            &test_kv(),
            reg,
            ModelId("m".into()),
        );
        let d = s
            .enqueue(
                &s,
                8_000,
                Some(answer(&[
                    ("http://p0:30000", 6_000),
                    ("http://p1:30000", 100),
                ])),
                None,
            )
            .await
            .unwrap();
        assert_eq!(d.worker.url, "http://p0:30000");
        assert_eq!(d.branch, Branch::Direct);
        // The inline arrival pass dispatched it.
        assert_eq!(d.trigger, Trigger::Arrival);
        // Uniform pull rule: p0 is the best source itself → no hint.
        assert!(d.hint.is_none());
        // Ledger charged with prompt − own hit.
        assert_eq!(
            s.state.lock().workers["http://p0:30000"].backlog(),
            8_000 - 6_000
        );
    }

    #[tokio::test]
    async fn transfer_to_under_cap_worker_with_hint_when_a_is_full() {
        let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
        let s = CoScheduler::spawn(
            test_cfg(KvPullScope::Any, 30_000),
            &test_kv(),
            reg,
            ModelId("m".into()),
        );
        // p0 (the max-hit worker) is over the cap; p1 is idle.
        set_polled(&s, "http://p0:30000", 50_000);
        set_polled(&s, "http://p1:30000", 0);
        let d = s
            .enqueue(
                &s,
                9_000,
                Some(answer(&[
                    ("http://p0:30000", 8_192),
                    ("http://p1:30000", 0),
                ])),
                None,
            )
            .await
            .unwrap();
        assert_eq!(d.worker.url, "http://p1:30000");
        assert_eq!(d.branch, Branch::Transfer);
        let hint = d.hint.expect("transfer must carry the pull hint");
        assert_eq!(hint["source_worker_id"], "http://p0:30000");
        assert_eq!(hint["expected_hit_tokens"], 8_192);
    }

    #[tokio::test]
    async fn detour_only_scope_suppresses_direct_branch_hints() {
        let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
        let s = CoScheduler::spawn(
            test_cfg(KvPullScope::DetourOnly, 30_000),
            &test_kv(),
            reg,
            ModelId("m".into()),
        );
        // p1 has the max hit and is idle → direct; p0 holds MORE tokens is
        // false here — make p0 the better source while routing lands on p0
        // itself? Instead: route direct to p1 (max hit), best other source
        // p0 clears the incremental gate — Any would inject, DetourOnly
        // must not.
        let d = s
            .enqueue(
                &s,
                12_000,
                Some(answer(&[
                    ("http://p1:30000", 4_096),
                    ("http://p0:30000", 100),
                ])),
                None,
            )
            .await
            .unwrap();
        assert_eq!(d.branch, Branch::Direct);
        assert_eq!(d.worker.url, "http://p1:30000");
        assert!(d.hint.is_none(), "detour-only must not hint on direct");
    }

    #[tokio::test]
    async fn uniform_scope_hints_on_direct_when_better_source_exists() {
        let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
        let s = CoScheduler::spawn(
            test_cfg(KvPullScope::Any, 30_000),
            &test_kv(),
            reg,
            ModelId("m".into()),
        );
        // Both under cap. p1 max hit → direct to p1; p0 holds MORE (stale
        // index ordering can produce this; with decode sources it is the
        // common case). Any-scope injects source=p0.
        // NOTE: hits map has p0 > p1, so A = p0 actually. Build the case
        // where A is p1 (max hit) but the ANSWER's best candidate is p0
        // via equal hits is impossible — instead assert on the transfer-
        // free case: A=p0 direct, best other p1 below gate → no hint; then
        // A=p0 direct with p1 8000 vs p0 100 → A=p1... Keep it simple: the
        // realistic stale-index shape is hits{p1: 9000} while the pool also
        // contains p0 with 0 — direct to p1, no better source, no hint;
        // uniform-vs-detour difference is already pinned by the two tests
        // above plus the incremental-gate unit tests in kv_hint.
        let d = s
            .enqueue(
                &s,
                12_000,
                Some(answer(&[("http://p1:30000", 9_000)])),
                None,
            )
            .await
            .unwrap();
        assert_eq!(d.branch, Branch::Direct);
        assert!(d.hint.is_none());
    }

    #[tokio::test]
    async fn holds_when_all_over_cap_then_dispatches_on_retire() {
        let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
        let s = CoScheduler::spawn(
            test_cfg(KvPullScope::Any, 30_000),
            &test_kv(),
            reg,
            ModelId("m".into()),
        );
        set_polled(&s, "http://p0:30000", 50_000);
        set_polled(&s, "http://p1:30000", 50_000);
        let s2 = Arc::clone(&s);
        let waiter = tokio::spawn(async move {
            s2.enqueue(
                &s2,
                4_000,
                Some(answer(&[("http://p0:30000", 3_000)])),
                None,
            )
            .await
        });
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(s.queue_depth(), 1, "must hold while everyone is over cap");
        // Capacity frees on p1 via a fresh poll.
        set_polled(&s, "http://p1:30000", 100);
        s.wake(Triggers::POLL);
        let d = tokio::time::timeout(Duration::from_secs(2), waiter)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(d.worker.url, "http://p1:30000");
        assert_eq!(d.branch, Branch::Transfer);
        assert_eq!(d.trigger, Trigger::Poll);
    }

    #[tokio::test]
    async fn forced_dispatch_fires_after_max_wait() {
        let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
        let s = CoScheduler::spawn(
            test_cfg(KvPullScope::Any, 200),
            &test_kv(),
            reg,
            ModelId("m".into()),
        );
        set_polled(&s, "http://p0:30000", 50_000);
        set_polled(&s, "http://p1:30000", 60_000);
        let start = Instant::now();
        let d = s
            .enqueue(&s, 4_000, Some(answer(&[("http://p0:30000", 3_000)])), None)
            .await
            .unwrap();
        assert_eq!(d.branch, Branch::Forced);
        // min backlog = p0 (50k < 60k)
        assert_eq!(d.worker.url, "http://p0:30000");
        assert!(start.elapsed() >= Duration::from_millis(200));
        assert!(d.queue_wait_ms >= 200);
    }

    #[tokio::test]
    async fn cache_affinity_reorder_skips_blocked_head() {
        let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
        let s = CoScheduler::spawn(
            test_cfg(KvPullScope::Any, 30_000),
            &test_kv(),
            reg,
            ModelId("m".into()),
        );
        // Head's A (p0) is over cap; second entry's A (p1) is free. The
        // second must dispatch FIRST (direct), then the head detours to p1
        // once... p1 then has backlog 4k < cap so head transfers to p1 too.
        set_polled(&s, "http://p0:30000", 50_000);
        set_polled(&s, "http://p1:30000", 0);
        let s_head = Arc::clone(&s);
        let head = tokio::spawn(async move {
            s_head
                .enqueue(
                    &s_head,
                    6_000,
                    Some(answer(&[("http://p0:30000", 5_000)])),
                    None,
                )
                .await
        });
        tokio::time::sleep(Duration::from_millis(120)).await;
        let d2 = s
            .enqueue(&s, 4_096, Some(answer(&[("http://p1:30000", 4_000)])), None)
            .await
            .unwrap();
        assert_eq!(d2.branch, Branch::Direct);
        assert_eq!(d2.worker.url, "http://p1:30000");
        let d1 = tokio::time::timeout(Duration::from_secs(2), head)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(d1.branch, Branch::Transfer);
        assert_eq!(d1.worker.url, "http://p1:30000");
        let hint = d1.hint.expect("head transfer pulls from p0");
        assert_eq!(hint["source_worker_id"], "http://p0:30000");
    }

    #[tokio::test]
    async fn poll_baseline_plus_ledger_fusion_and_poll_retire() {
        let reg = registry_with_prefills(&["http://p0:30000"]);
        let s = CoScheduler::spawn(
            test_cfg(KvPullScope::Any, 30_000),
            &test_kv(),
            reg,
            ModelId("m".into()),
        );
        set_polled(&s, "http://p0:30000", 1_000);
        let d = s
            .enqueue(&s, 3_000, Some(answer(&[("http://p0:30000", 1_000)])), None)
            .await
            .unwrap();
        // baseline 1000 + ledger (3000 − 1000 hit) = 3000
        assert_eq!(s.state.lock().workers["http://p0:30000"].backlog(), 3_000);
        // First-byte retire drops the ledger part.
        d.ledger.retire();
        assert_eq!(s.state.lock().workers["http://p0:30000"].backlog(), 1_000);

        // A fresh poll subsumes entries dispatched before it was sent.
        let d2 = s
            .enqueue(&s, 2_000, Some(answer(&[("http://p0:30000", 0)])), None)
            .await
            .unwrap();
        assert_eq!(s.state.lock().workers["http://p0:30000"].backlog(), 3_000);
        s.apply_poll("http://p0:30000", 2_500, Instant::now());
        assert_eq!(
            s.state.lock().workers["http://p0:30000"].backlog(),
            2_500,
            "post-poll: baseline replaces the pre-poll ledger entry"
        );
        drop(d2);
    }

    #[tokio::test]
    async fn degrades_to_min_backlog_without_indexer_answer() {
        let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
        let s = CoScheduler::spawn(
            test_cfg(KvPullScope::Any, 30_000),
            &test_kv(),
            reg,
            ModelId("m".into()),
        );
        set_polled(&s, "http://p0:30000", 5_000);
        set_polled(&s, "http://p1:30000", 200);
        let d = s.enqueue(&s, 4_000, None, None).await.unwrap();
        assert_eq!(d.worker.url, "http://p1:30000");
        assert!(d.hint.is_none());
    }

    #[tokio::test]
    async fn retire_release_is_labeled_retire() {
        let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
        let s = CoScheduler::spawn(
            test_cfg(KvPullScope::Any, 30_000),
            &test_kv(),
            reg,
            ModelId("m".into()),
        );
        set_polled(&s, "http://p0:30000", 0);
        set_polled(&s, "http://p1:30000", 50_000);
        // d1 fills p0 past the cap via its ledger entry (12 000 > 10 240).
        let d1 = s
            .enqueue(&s, 12_000, Some(answer(&[("http://p0:30000", 0)])), None)
            .await
            .unwrap();
        assert_eq!(d1.worker.url, "http://p0:30000");
        let s2 = Arc::clone(&s);
        let waiter = tokio::spawn(async move {
            s2.enqueue(
                &s2,
                4_000,
                Some(answer(&[("http://p0:30000", 3_000)])),
                None,
            )
            .await
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(s.queue_depth(), 1, "must hold while everyone is over cap");
        // First-byte retire frees p0; the retire wake dispatches d2.
        d1.ledger.retire();
        let d2 = tokio::time::timeout(Duration::from_secs(2), waiter)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(d2.worker.url, "http://p0:30000");
        assert_eq!(d2.branch, Branch::Direct);
        assert_eq!(d2.trigger, Trigger::Retire);
    }

    #[tokio::test]
    async fn starvation_fires_at_deadline_not_at_tick() {
        let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
        let s = CoScheduler::spawn(
            test_cfg_with(|c| {
                c.max_queue_wait_ms = 200;
                c.tick_ms = 5_000; // a naive tick would take ~5 s
            }),
            &test_kv(),
            reg,
            ModelId("m".into()),
        );
        set_polled(&s, "http://p0:30000", 50_000);
        set_polled(&s, "http://p1:30000", 60_000);
        let start = Instant::now();
        let d = s
            .enqueue(&s, 4_000, Some(answer(&[("http://p0:30000", 3_000)])), None)
            .await
            .unwrap();
        assert_eq!(d.branch, Branch::Forced);
        // min backlog = p0 (50k < 60k)
        assert_eq!(d.worker.url, "http://p0:30000");
        assert_eq!(d.trigger, Trigger::Starvation);
        assert!(start.elapsed() >= Duration::from_millis(200));
        assert!(
            start.elapsed() < Duration::from_millis(1_000),
            "deadline sleep must not be quantized by the 5 s tick (took {:?})",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn tick_fallback_is_labeled_tick() {
        let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
        let s = CoScheduler::spawn(
            test_cfg_with(|c| c.tick_ms = 50),
            &test_kv(),
            reg,
            ModelId("m".into()),
        );
        set_polled(&s, "http://p0:30000", 50_000);
        set_polled(&s, "http://p1:30000", 50_000);
        let s2 = Arc::clone(&s);
        let waiter = tokio::spawn(async move {
            s2.enqueue(
                &s2,
                4_000,
                Some(answer(&[("http://p0:30000", 3_000)])),
                None,
            )
            .await
        });
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(s.queue_depth(), 1);
        // Free p1 WITHOUT any wake: only the tick fallback can see it.
        s.state
            .lock()
            .workers
            .get_mut("http://p1:30000")
            .unwrap()
            .polled_uncached = 100;
        let d = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(d.worker.url, "http://p1:30000");
        assert_eq!(d.branch, Branch::Transfer);
        assert_eq!(d.trigger, Trigger::Tick);
    }

    #[tokio::test]
    async fn hold_signals_poll_on_hold_counter() {
        // Opted in: every HOLD pass signals the poller.
        let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
        let s = CoScheduler::spawn(
            test_cfg_with(|c| c.poll_on_hold_ms = 100),
            &test_kv(),
            reg,
            ModelId("m".into()),
        );
        set_polled(&s, "http://p0:30000", 50_000);
        set_polled(&s, "http://p1:30000", 50_000);
        let s2 = Arc::clone(&s);
        let _waiter = tokio::spawn(async move {
            s2.enqueue(
                &s2,
                4_000,
                Some(answer(&[("http://p0:30000", 3_000)])),
                None,
            )
            .await
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            s.poll_now_signals.load(Ordering::Relaxed) >= 1,
            "the inline arrival HOLD (and the dispatcher HOLD) must signal"
        );

        // Disabled (0, the test_cfg default): no signal ever.
        let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
        let s = CoScheduler::spawn(
            test_cfg(KvPullScope::Any, 30_000),
            &test_kv(),
            reg,
            ModelId("m".into()),
        );
        set_polled(&s, "http://p0:30000", 50_000);
        set_polled(&s, "http://p1:30000", 50_000);
        let s2 = Arc::clone(&s);
        let _waiter = tokio::spawn(async move {
            s2.enqueue(
                &s2,
                4_000,
                Some(answer(&[("http://p0:30000", 3_000)])),
                None,
            )
            .await
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(s.poll_now_signals.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn apply_poll_ignores_out_of_order_sample() {
        let reg = registry_with_prefills(&["http://p0:30000"]);
        let s = CoScheduler::spawn(
            test_cfg(KvPullScope::Any, 30_000),
            &test_kv(),
            reg,
            ModelId("m".into()),
        );
        let backlog = |s: &Arc<CoScheduler>| s.state.lock().workers["http://p0:30000"].backlog();
        let t0 = Instant::now();
        tokio::time::sleep(Duration::from_millis(5)).await;
        let t1 = Instant::now();
        s.apply_poll("http://p0:30000", 5_000, t1);
        assert_eq!(backlog(&s), 5_000);
        // Strictly older sample (periodic round overtaken by poll-on-hold):
        // ignored.
        s.apply_poll("http://p0:30000", 50_000, t0);
        assert_eq!(backlog(&s), 5_000);
        // Equal sent_at is NOT older: applied.
        s.apply_poll("http://p0:30000", 7_000, t1);
        assert_eq!(backlog(&s), 7_000);
    }

    #[test]
    fn trigger_priority_and_label_pure() {
        assert_eq!(
            Triggers::POLL.or(Triggers::ARRIVAL).primary(),
            Trigger::Poll
        );
        assert_eq!(
            Triggers::STARVATION.or(Triggers::POLL).primary(),
            Trigger::Starvation
        );
        assert_eq!(
            Triggers::RETIRE.or(Triggers::ARRIVAL).primary(),
            Trigger::Retire
        );
        assert_eq!(Triggers::ARRIVAL.primary(), Trigger::Arrival);
        assert_eq!(Triggers(0).primary(), Trigger::Tick);
        assert_eq!(
            Triggers::POLL.or(Triggers::ARRIVAL).to_string(),
            "poll+arrival"
        );
        assert_eq!(
            Triggers::STARVATION
                .or(Triggers::POLL)
                .or(Triggers::TICK)
                .to_string(),
            "starvation+poll+tick"
        );
        assert_eq!(Triggers(0).to_string(), "tick");
        for t in [
            Trigger::Arrival,
            Trigger::Retire,
            Trigger::Poll,
            Trigger::Tick,
            Trigger::Starvation,
        ] {
            assert!(["arrival", "retire", "poll", "tick", "starvation"].contains(&t.as_str()));
        }
    }

    #[tokio::test]
    async fn drain_returns_pass_outcome_synchronously() {
        let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
        let s = CoScheduler::spawn(
            test_cfg_with(|c| c.tick_ms = 60_000), // dispatcher effectively off
            &test_kv(),
            reg,
            ModelId("m".into()),
        );
        set_polled(&s, "http://p0:30000", 50_000);
        set_polled(&s, "http://p1:30000", 50_000);
        let s2 = Arc::clone(&s);
        let waiter = tokio::spawn(async move {
            s2.enqueue(
                &s2,
                4_000,
                Some(answer(&[("http://p0:30000", 3_000)])),
                None,
            )
            .await
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(
            s.drain(&s, Triggers::TICK),
            PassOutcome {
                dispatched: 0,
                held: 1
            }
        );
        s.state
            .lock()
            .workers
            .get_mut("http://p1:30000")
            .unwrap()
            .polled_uncached = 100;
        assert_eq!(
            s.drain(&s, Triggers::TICK),
            PassOutcome {
                dispatched: 1,
                held: 0
            }
        );
        let d = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(d.worker.url, "http://p1:30000");
        assert_eq!(d.branch, Branch::Transfer);
        assert_eq!(d.trigger, Trigger::Tick);

        // NoPool path: empty registry → the pass holds everything queued.
        let s = CoScheduler::spawn(
            test_cfg(KvPullScope::Any, 30_000),
            &test_kv(),
            registry_with_prefills(&[]), // no workers at all
            ModelId("m".into()),
        );
        let s2 = Arc::clone(&s);
        let _waiter = tokio::spawn(async move { s2.enqueue(&s2, 4_000, None, None).await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(
            s.drain(&s, Triggers::TICK),
            PassOutcome {
                dispatched: 0,
                held: 1
            }
        );
    }

    #[tokio::test]
    async fn imbalance_dispatches_head_to_min_backlog_when_all_over_cap() {
        let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
        let s = CoScheduler::spawn(
            test_cfg_with(|c| c.imbalance_spread_tokens = 131_072),
            &test_kv(),
            reg,
            ModelId("m".into()),
        );
        // Both over the cap (10 240), spread 180 000 > 131 072.
        set_polled(&s, "http://p0:30000", 200_000);
        set_polled(&s, "http://p1:30000", 20_000);
        let d = s
            .enqueue(&s, 4_000, Some(answer(&[("http://p0:30000", 3_000)])), None)
            .await
            .unwrap();
        // The inline arrival pass takes the imbalance escape immediately.
        assert_eq!(d.branch, Branch::Imbalance);
        assert_eq!(d.worker.url, "http://p1:30000");
        assert_eq!(d.trigger, Trigger::Arrival);
        // Hint goes through the normal pull gate, same as Transfer.
        let hint = d.hint.expect("imbalance dispatch pulls from p0");
        assert_eq!(hint["source_worker_id"], "http://p0:30000");
    }

    #[tokio::test]
    async fn imbalance_holds_when_spread_below_threshold() {
        let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
        let s = CoScheduler::spawn(
            test_cfg_with(|c| c.imbalance_spread_tokens = 131_072),
            &test_kv(),
            reg,
            ModelId("m".into()),
        );
        // Both over the cap but spread 40 000 <= 131 072: hold.
        set_polled(&s, "http://p0:30000", 60_000);
        set_polled(&s, "http://p1:30000", 20_000);
        let s2 = Arc::clone(&s);
        let waiter = tokio::spawn(async move {
            s2.enqueue(
                &s2,
                4_000,
                Some(answer(&[("http://p0:30000", 3_000)])),
                None,
            )
            .await
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(s.queue_depth(), 1, "sub-threshold skew must not dispatch");
        // A fresh poll frees p1 → ordinary Transfer, not Imbalance.
        s.apply_poll("http://p1:30000", 0, Instant::now());
        s.wake(Triggers::POLL);
        let d = tokio::time::timeout(Duration::from_secs(2), waiter)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(d.worker.url, "http://p1:30000");
        assert_eq!(d.branch, Branch::Transfer);
        assert_eq!(d.trigger, Trigger::Poll);
    }

    #[tokio::test]
    async fn imbalance_disabled_holds() {
        let reg = registry_with_prefills(&["http://p0:30000", "http://p1:30000"]);
        let s = CoScheduler::spawn(
            test_cfg(KvPullScope::Any, 30_000), // imbalance_spread_tokens: 0
            &test_kv(),
            reg,
            ModelId("m".into()),
        );
        set_polled(&s, "http://p0:30000", 200_000);
        set_polled(&s, "http://p1:30000", 20_000);
        let s2 = Arc::clone(&s);
        let _waiter = tokio::spawn(async move {
            s2.enqueue(
                &s2,
                4_000,
                Some(answer(&[("http://p0:30000", 3_000)])),
                None,
            )
            .await
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(
            s.queue_depth(),
            1,
            "threshold 0 disables the escape even at spread 180 000"
        );
        assert_eq!(
            s.drain(&s, Triggers::TICK),
            PassOutcome {
                dispatched: 0,
                held: 1
            }
        );
    }

    // ---- global waiting queue (`--co-schedule-global-queue`) ----

    const GWQ_P: &str = "http://p0:30000";

    /// TRADE (lambda 9.2, ledger wait) with the queue at a fixed depth. The
    /// test config polls nothing (load_poll_ms 1 h): the tests apply the
    /// progress snapshots themselves, and a view never goes stale.
    fn gwq_sched(
        tmp: &tempfile::TempDir,
        queue: Option<(GlobalQueueMode, u64, u64)>,
        urls: &[&str],
    ) -> Arc<CoScheduler> {
        gwq_sched_hold(tmp, queue, None, urls)
    }

    /// [`gwq_sched`] with `--co-schedule-global-queue-hold`.
    fn gwq_sched_hold(
        tmp: &tempfile::TempDir,
        queue: Option<(GlobalQueueMode, u64, u64)>,
        hold: Option<GlobalQueueHold>,
        urls: &[&str],
    ) -> Arc<CoScheduler> {
        let cfg = test_cfg_with(|c| {
            c.uncached_balance = true;
            c.trade = Some(crate::scheduler::co_schedule::TradeParams::default());
            c.cost_trace = Some(tmp.path().join("cost.jsonl"));
            c.global_queue = queue.map(|(mode, depth, max_wait_ms)| GlobalQueueConfig {
                mode,
                depth: GlobalQueueDepth::Tokens(depth),
                max_wait_ms,
                hold: hold.unwrap_or_default(),
                room: GlobalQueueRoom::Tokens,
            });
        });
        CoScheduler::spawn(
            cfg,
            &test_kv(),
            registry_with_prefills(urls),
            ModelId("m".into()),
        )
    }

    fn progress_of(
        sequence: u64,
        remaining: &[(&str, u64)],
    ) -> crate::cost_balance::PrefillProgress {
        crate::cost_balance::PrefillProgress {
            epoch: "engine-1".into(),
            sequence,
            remaining: remaining
                .iter()
                .map(|(rid, n)| (rid.to_string(), *n))
                .collect(),
            truncated: false,
        }
    }

    fn apply(s: &Arc<CoScheduler>, url: &str, sequence: u64, remaining: &[(&str, u64)]) {
        s.cost_book
            .as_ref()
            .unwrap()
            .apply_progress(
                url,
                progress_of(sequence, remaining),
                serde_json::json!({}),
                Instant::now(),
            )
            .unwrap();
    }

    fn trace_events(tmp: &tempfile::TempDir) -> Vec<Value> {
        std::fs::read_to_string(tmp.path().join("cost.jsonl"))
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    fn reserve_event(tmp: &tempfile::TempDir, rid: &str) -> Option<Value> {
        trace_events(tmp)
            .into_iter()
            .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == rid)
    }

    async fn wait_for(what: &str, check: impl Fn() -> bool) {
        tokio::time::timeout(Duration::from_secs(3), async {
            while !check() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {what}"));
    }

    /// The P has a fresh view and `busy` (5000 tokens, dispatched, first chunk
    /// not launched): its queued work is above the depth 1000.
    async fn gwq_block(s: &Arc<CoScheduler>) -> Dispatch {
        apply(s, GWQ_P, 1, &[]);
        let mut busy = s
            .admit(s, None, 5000, None, Some("busy".into()))
            .await
            .unwrap();
        assert_eq!(
            busy.queue_wait_ms, 0,
            "an idle P with room takes it at once"
        );
        busy.cost_reservation
            .as_mut()
            .unwrap()
            .mark_dispatched()
            .unwrap();
        busy
    }

    fn spawn_admit(
        s: &Arc<CoScheduler>,
        rid: &'static str,
        tokens: usize,
    ) -> tokio::task::JoinHandle<Result<Dispatch, AdmitError>> {
        let s = Arc::clone(s);
        tokio::spawn(async move { s.admit(&s, None, tokens, None, Some(rid.into())).await })
    }

    #[tokio::test]
    async fn gwq_holds_while_the_trade_choice_has_no_room_and_releases_on_progress() {
        let tmp = tempfile::tempdir().unwrap();
        let s = gwq_sched(&tmp, Some((GlobalQueueMode::Hrrn, 1000, 30_000)), &[GWQ_P]);
        let busy = gwq_block(&s).await;
        let small = spawn_admit(&s, "small", 800);
        wait_for("small held", || s.gwq_len() == 1).await;
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert!(
            !small.is_finished(),
            "5000 queued tokens are above the depth 1000"
        );
        assert!(
            reserve_event(&tmp, "small").is_none(),
            "no reservation while held"
        );
        // The engine lists `busy`: its first chunk launched, the queued work is 0.
        apply(&s, GWQ_P, 2, &[("busy", 5000)]);
        s.wake(Triggers::POLL);
        let small = tokio::time::timeout(Duration::from_secs(3), small)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(small.worker.url, GWQ_P);
        assert!(
            matches!(small.trigger, Trigger::Poll | Trigger::Tick),
            "{:?}",
            small.trigger
        );
        assert!(small.queue_wait_ms >= 60, "{}", small.queue_wait_ms);
        assert_eq!(small.branch, Branch::UncachedBalanced);
        assert_eq!(s.gwq_len(), 0);
        let b = reserve_event(&tmp, "busy").unwrap();
        assert_eq!(b["gwq"]["held"], false);
        assert_eq!(b["gwq"]["release"], "room");
        assert_eq!(b["gwq"]["queued_tokens"], "0");
        let r = reserve_event(&tmp, "small").unwrap();
        let g = &r["gwq"];
        assert_eq!(g["mode"], "hrrn");
        assert_eq!(g["held"], true);
        assert_eq!(g["release"], "room");
        assert_eq!(g["starved"], false);
        assert_eq!(g["depth_tokens"], 1000);
        assert_eq!(g["queued_tokens"], "0");
        assert_eq!(g["p_arrival"], GWQ_P);
        assert_eq!(g["p_release"], GWQ_P);
        assert_eq!(g["late_bound"], false);
        assert_eq!(g["queue_len"], 1);
        assert!(g["hold_s"].as_f64().unwrap() >= 0.06, "{g}");
        assert_eq!(r["trade_decision"]["chosen"], GWQ_P);
        let row = &r["trade_decision"]["candidates"][0];
        assert_eq!(row["gwq_depth_tokens"], 1000);
        assert_eq!(row["gwq_room"], true);
        assert_eq!(row["gwq_virtual_tokens"], "0");
        let counters = s.gwq.as_ref().unwrap().state.lock().counters;
        assert_eq!(counters.arrivals, 2);
        assert_eq!(counters.released_at_arrival, 1);
        assert_eq!(counters.released_after_hold, 1);
        assert_eq!(counters.released_room, 2);
        drop((busy, small));
    }

    /// sjf with one P: each release fills the depth again, so the next
    /// release waits for the next launch, and the order is by service.
    #[tokio::test]
    async fn gwq_sjf_releases_the_smallest_held_request_first() {
        let tmp = tempfile::tempdir().unwrap();
        let s = gwq_sched(&tmp, Some((GlobalQueueMode::Sjf, 1000, 30_000)), &[GWQ_P]);
        let busy = gwq_block(&s).await;
        let mut waiters = Vec::new();
        for (rid, tokens) in [("big", 4000usize), ("small", 1500), ("medium", 2000)] {
            waiters.push((rid, spawn_admit(&s, rid, tokens)));
            let n = waiters.len();
            wait_for("held", || s.gwq_len() == n).await;
        }
        let mut dispatches = vec![busy];
        let mut order = Vec::new();
        let mut launched = "busy";
        for sequence in 2..5 {
            apply(&s, GWQ_P, sequence, &[(launched, 1)]);
            s.wake(Triggers::POLL);
            let deadline = Instant::now() + Duration::from_secs(3);
            let i = loop {
                if let Some(i) = waiters.iter().position(|(_, h)| h.is_finished()) {
                    break i;
                }
                assert!(
                    Instant::now() < deadline,
                    "no release after {launched} launched"
                );
                tokio::time::sleep(Duration::from_millis(5)).await;
            };
            let (rid, handle) = waiters.remove(i);
            let mut d = handle.await.unwrap().unwrap();
            d.cost_reservation
                .as_mut()
                .unwrap()
                .mark_dispatched()
                .unwrap();
            dispatches.push(d);
            order.push(rid);
            launched = rid;
            tokio::time::sleep(Duration::from_millis(30)).await;
            assert!(
                waiters.iter().all(|(_, h)| !h.is_finished()),
                "one release per launch: the released request fills the depth"
            );
        }
        assert_eq!(order, vec!["small", "medium", "big"]);
        let g = reserve_event(&tmp, "small").unwrap()["gwq"].clone();
        assert_eq!(g["mode"], "sjf");
        assert_eq!(g["rank"], 0);
        assert_eq!(g["queue_len"], 3);
        assert!(
            (g["key"].as_f64().unwrap() - 1500.0 / 13_500.0).abs() < 1e-9,
            "{g}"
        );
        drop(dispatches);
    }

    #[tokio::test]
    async fn gwq_drops_a_held_request_whose_client_left() {
        let tmp = tempfile::tempdir().unwrap();
        let s = gwq_sched(&tmp, Some((GlobalQueueMode::Hrrn, 1000, 30_000)), &[GWQ_P]);
        let busy = gwq_block(&s).await;
        let gone = spawn_admit(&s, "gone", 800);
        wait_for("gone held", || s.gwq_len() == 1).await;
        gone.abort();
        wait_for("gone dropped", || s.gwq_len() == 0).await;
        let counters = s.gwq.as_ref().unwrap().state.lock().counters;
        assert_eq!(counters.dropped_client_gone, 1);
        wait_for("gwq_drop event", || {
            trace_events(&tmp).iter().any(|e| {
                e["event"] == "gwq_drop" && e["rid"] == "gone" && e["reason"] == "client_gone"
            })
        })
        .await;
        // Room comes later: nothing is reserved for the client that left.
        apply(&s, GWQ_P, 2, &[("busy", 5000)]);
        s.wake(Triggers::POLL);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(reserve_event(&tmp, "gone").is_none());
        drop(busy);
    }

    /// Fail-open: without a progress view of the P, the P has room, and a
    /// request goes at once even with queued work above the depth.
    #[tokio::test]
    async fn gwq_fails_open_without_a_progress_view() {
        let tmp = tempfile::tempdir().unwrap();
        let s = gwq_sched(&tmp, Some((GlobalQueueMode::Sjf, 1000, 30_000)), &[GWQ_P]);
        let mut busy = s
            .admit(&s, None, 5000, None, Some("busy".into()))
            .await
            .unwrap();
        busy.cost_reservation
            .as_mut()
            .unwrap()
            .mark_dispatched()
            .unwrap();
        let small = tokio::time::timeout(
            Duration::from_millis(500),
            s.admit(&s, None, 800, None, Some("small".into())),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(small.queue_wait_ms, 0);
        let g = reserve_event(&tmp, "small").unwrap()["gwq"].clone();
        assert_eq!(g["release"], "stale_view");
        assert_eq!(g["view_fresh"], false);
        assert_eq!(g["held"], false);
        assert_eq!(g["queued_tokens"], "5000");
        let counters = s.gwq.as_ref().unwrap().state.lock().counters;
        assert_eq!(counters.released_stale_view, 2);
        drop((busy, small));
    }

    /// sjf: a request held for the max wait goes to its TRADE argmin at
    /// once, without room, exactly at its deadline.
    #[tokio::test]
    async fn gwq_sjf_starvation_guard_releases_without_room() {
        let tmp = tempfile::tempdir().unwrap();
        let s = gwq_sched(&tmp, Some((GlobalQueueMode::Sjf, 1000, 200)), &[GWQ_P]);
        let busy = gwq_block(&s).await;
        let started = Instant::now();
        let late = tokio::time::timeout(
            Duration::from_secs(3),
            s.admit(&s, None, 800, None, Some("late".into())),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(started.elapsed() >= Duration::from_millis(200));
        assert_eq!(late.trigger, Trigger::Starvation);
        assert!(late.queue_wait_ms >= 200, "{}", late.queue_wait_ms);
        let g = reserve_event(&tmp, "late").unwrap()["gwq"].clone();
        assert_eq!(g["release"], "starved");
        assert_eq!(g["starved"], true);
        assert_eq!(g["held"], true);
        assert_eq!(g["queued_tokens"], "5000", "released without room");
        let counters = s.gwq.as_ref().unwrap().state.lock().counters;
        assert_eq!(counters.released_starved, 1);
        drop((busy, late));
    }

    /// Default off: TRADE dispatches at arrival whatever the queued work,
    /// and the trace has no queue keys or events.
    #[tokio::test]
    async fn gwq_off_dispatches_every_trade_request_at_arrival() {
        let tmp = tempfile::tempdir().unwrap();
        let s = gwq_sched(&tmp, None, &[GWQ_P]);
        assert!(s.gwq.is_none());
        let busy = gwq_block(&s).await;
        let small = tokio::time::timeout(
            Duration::from_millis(500),
            s.admit(&s, None, 800, None, Some("small".into())),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(small.queue_wait_ms, 0);
        assert_eq!(small.trigger, Trigger::Arrival);
        tokio::time::sleep(Duration::from_millis(50)).await;
        let events = trace_events(&tmp);
        for rid in ["busy", "small"] {
            let r = reserve_event(&tmp, rid).unwrap();
            assert!(r.get("gwq").is_none(), "{r}");
            assert!(r["trade_decision"]["candidates"][0]
                .get("gwq_room")
                .is_none());
        }
        assert!(!events
            .iter()
            .any(|e| e["event"] == "gwq_stats" || e["event"] == "gwq_drop"));
        drop((busy, small));
    }

    /// Late P binding through the scheduler: P0 has no room and is the
    /// TRADE choice at arrival; when P1's launched work computes, P1 becomes
    /// the TRADE argmin and takes the request.
    #[tokio::test]
    async fn gwq_binds_the_p_at_release() {
        let tmp = tempfile::tempdir().unwrap();
        let (p0, p1) = ("http://p0:30000", "http://p1:30000");
        let s = gwq_sched(&tmp, Some((GlobalQueueMode::Hrrn, 1000, 30_000)), &[p0, p1]);
        apply(&s, p0, 1, &[]);
        apply(&s, p1, 1, &[]);
        // p0: 30k routed, not launched (no room). p1: a launched 40k request
        // (queued = 40k - one pass of 8192: no room).
        let mut on_p0 = s
            .admit(&s, None, 30_000, None, Some("on_p0".into()))
            .await
            .unwrap();
        assert_eq!(on_p0.worker.url, p0);
        on_p0
            .cost_reservation
            .as_mut()
            .unwrap()
            .mark_dispatched()
            .unwrap();
        let mut on_p1 = s
            .admit(&s, None, 40_000, None, Some("on_p1".into()))
            .await
            .unwrap();
        assert_eq!(on_p1.worker.url, p1);
        on_p1
            .cost_reservation
            .as_mut()
            .unwrap()
            .mark_dispatched()
            .unwrap();
        apply(&s, p1, 2, &[("on_p1", 40_000)]);
        // p0 = 35k < p1 = 45k, and p0 has no room: the request waits.
        let r = spawn_admit(&s, "r", 5000);
        wait_for("r held", || s.gwq_len() == 1).await;
        // p1 has 9k left: p1 = 14k < p0 = 35k, and p1 has room (queued 808).
        apply(&s, p1, 3, &[("on_p1", 9_000)]);
        s.wake(Triggers::POLL);
        let r = tokio::time::timeout(Duration::from_secs(3), r)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(r.worker.url, p1);
        let g = reserve_event(&tmp, "r").unwrap()["gwq"].clone();
        assert_eq!(g["p_arrival"], p0);
        assert_eq!(g["p_release"], p1);
        assert_eq!(g["late_bound"], true);
        assert_eq!(s.gwq.as_ref().unwrap().state.lock().counters.late_bound, 1);
        drop((on_p0, on_p1, r));
    }

    /// Predicted-drain fill (`--co-schedule-global-queue-hold free`): with
    /// both P full, a request bound to its owner (cached prefix one chunk
    /// above every other P) is dispatched at once by the TRADE path (no
    /// `gwq` record, counted as bypassed), while a free request (the same
    /// prefix on both P within one chunk) waits. With `hold = all` the bound
    /// request waits too.
    #[tokio::test]
    async fn gwq_hold_free_dispatches_bound_requests_at_once_and_holds_free_ones() {
        let (p0, p1) = ("http://p0:30000", "http://p1:30000");
        for hold in [GlobalQueueHold::Free, GlobalQueueHold::All] {
            let tmp = tempfile::tempdir().unwrap();
            let s = gwq_sched_hold(
                &tmp,
                Some((GlobalQueueMode::Hrrn, 1000, 30_000)),
                Some(hold),
                &[p0, p1],
            );
            apply(&s, p0, 1, &[]);
            apply(&s, p1, 1, &[]);
            let mut full = Vec::new();
            for rid in ["fill0", "fill1"] {
                let mut d = s
                    .admit(&s, None, 30_000, None, Some(rid.into()))
                    .await
                    .unwrap();
                d.cost_reservation
                    .as_mut()
                    .unwrap()
                    .mark_dispatched()
                    .unwrap();
                full.push(d);
            }
            assert_ne!(full[0].worker.url, full[1].worker.url, "one fill per P");
            let bound = {
                let s2 = Arc::clone(&s);
                tokio::spawn(async move {
                    s2.admit_gwq(
                        &s2,
                        60_000,
                        Some(answer(&[(p0, 50_000), (p1, 0)])),
                        Some("bound".into()),
                        None,
                        None,
                    )
                    .await
                })
            };
            match hold {
                GlobalQueueHold::Free => {
                    let d = tokio::time::timeout(Duration::from_millis(500), bound)
                        .await
                        .expect("a bound request is not held")
                        .unwrap()
                        .unwrap();
                    assert_eq!(d.worker.url, p0, "TRADE keeps the owner");
                    assert_eq!(d.queue_wait_ms, 0);
                    let r = reserve_event(&tmp, "bound").unwrap();
                    assert!(r.get("gwq").is_none(), "{r}");
                    let free = {
                        let s2 = Arc::clone(&s);
                        tokio::spawn(async move {
                            s2.admit_gwq(
                                &s2,
                                60_000,
                                Some(answer(&[(p0, 50_000), (p1, 45_000)])),
                                Some("free".into()),
                                None,
                                None,
                            )
                            .await
                        })
                    };
                    wait_for("free held", || s.gwq_len() == 1).await;
                    assert!(!free.is_finished());
                    let counters = s.gwq.as_ref().unwrap().state.lock().counters;
                    assert_eq!(counters.bypassed_bound, 1);
                    assert_eq!(counters.arrivals, 3, "two fills and the free request");
                    free.abort();
                    drop(d);
                }
                GlobalQueueHold::All => {
                    wait_for("bound held", || s.gwq_len() == 1).await;
                    assert!(!bound.is_finished());
                    assert_eq!(
                        s.gwq.as_ref().unwrap().state.lock().counters.bypassed_bound,
                        0
                    );
                    bound.abort();
                }
            }
            drop(full);
        }
    }
}
