// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `--pd-decode-policy power-of-two-smg`: the decode pick of
//! sgl-model-gateway's `--decode-policy power_of_two`, so this router and
//! sgl-model-gateway (SMG) use the same decode rule. The code is a
//! port of these SMG parts (paths under `sgl-model-gateway/src/`):
//!
//! * load monitor (`core/worker_manager.rs` `LoadMonitor::monitor_loop`,
//!   `get_all_worker_loads`, `parse_load_response`, `parse_load_value`):
//!   every `--pd-decode-load-check-interval-secs` (SMG
//!   `--load-check-interval-secs`, default 5) GET
//!   `{worker}/v1/loads?include=core` from EVERY registered worker at once
//!   (5 s timeout each); a worker's load is the sum of
//!   `loads[].num_total_tokens` (all ranks present and non-negative), or
//!   `aggregate.total_tokens` when there is no `loads` key; a worker without
//!   a value is left out, and each round REPLACES the whole snapshot (an
//!   empty round clears it);
//! * selection (`policies/power_of_two.rs` `select_worker`): two distinct
//!   samples over the healthy decodes (`idx1` uniform,
//!   `idx2 = (idx1 + 1 + U[0, n-1)) % n`); compare token loads when BOTH
//!   samples have one, else compare the request counts of BOTH; the first
//!   sample wins ties (`load1 <= load2`); one healthy decode is returned
//!   without sampling;
//! * request count (`core/worker.rs` `WorkerLoadGuard`, used by
//!   `routers/http/pd_router.rs`): +1 when a decode leg starts, -1 when it
//!   ends. A non-streaming leg counts from dispatch
//!   (`execute_dual_dispatch_internal`); a streaming leg counts from the
//!   decode response headers (`create_streaming_response`) to the end of the
//!   stream. SGLang sends streaming headers with the first output chunk, so
//!   a streaming request counts on its decode from its first token. This is
//!   [`SmgLoadGuard`]; it is separate from [`Worker::active_load`], which
//!   counts from dispatch in both modes and is left unchanged.
//!
//! Observability: per-round INFO line (`target: decode_po2_smg`) with the
//! cumulative counters, and `/metrics` series `sgl_router_decode_po2_smg_*`
//! (picks by signal: `tokens` / `requests` (fallback) / `single`; poll rounds,
//! empty rounds and per-worker poll failures; the snapshot and the request
//! counts).

use crate::server::metrics::escape_label;
use crate::workers::{Worker, WorkerRegistry};
use dashmap::DashMap;
use rand::Rng;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, Weak};
use std::time::{Duration, Instant};

const LOG_TARGET: &str = "decode_po2_smg";

/// SMG `worker_manager.rs` `REQUEST_TIMEOUT`: per-worker `/v1/loads` timeout.
const LOAD_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// Load signal one decode pick compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// Both samples had a polled token load.
    Tokens,
    /// Fallback: at least one sample had no token load, so both request
    /// counts were compared.
    Requests,
    /// One healthy decode: returned without a comparison.
    Single,
}

#[derive(Debug, Default)]
struct Stats {
    selections_tokens: AtomicU64,
    selections_requests: AtomicU64,
    selections_single: AtomicU64,
    poll_rounds: AtomicU64,
    poll_empty_rounds: AtomicU64,
    poll_worker_ok: AtomicU64,
    poll_worker_failures: AtomicU64,
    counted_at_dispatch: AtomicU64,
    counted_at_headers: AtomicU64,
}

/// Point-in-time copy of the cumulative counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StatsSnapshot {
    pub selections_tokens: u64,
    pub selections_requests: u64,
    pub selections_single: u64,
    pub poll_rounds: u64,
    pub poll_empty_rounds: u64,
    pub poll_worker_ok: u64,
    pub poll_worker_failures: u64,
    pub counted_at_dispatch: u64,
    pub counted_at_headers: u64,
}

/// Result of one polling round.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RoundSummary {
    pub polled: usize,
    pub ok: usize,
    pub failed: usize,
}

/// See the module docs.
#[derive(Debug, Default)]
pub struct SmgPowerOfTwo {
    /// SMG `PowerOfTwoPolicy::cached_loads`: worker URL -> token load.
    cached_loads: RwLock<HashMap<String, isize>>,
    /// SMG `Worker::load()` of each decode worker, by URL ([`SmgLoadGuard`]).
    request_loads: DashMap<String, Arc<AtomicUsize>>,
    stats: Arc<Stats>,
    /// Late D choose (`--pd-late-decode choose`): poll
    /// `/v1/loads?include=core,disagg` and read each rank's
    /// `late_d.bound_total_tokens` (num_total_tokens without the DEFERRED
    /// candidates, which hold no KV) when present. Off: the SMG poll as is.
    bound_view: AtomicBool,
    /// When the last poll round replaced the snapshot.
    last_round: std::sync::Mutex<Option<Instant>>,
}

impl SmgPowerOfTwo {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the policy and, when a Tokio runtime is available, start its
    /// load monitor over every worker in `registry`. The monitor stops when
    /// the policy is dropped. `interval_secs` 0 is treated as 1 (the CLI
    /// rejects 0, as SMG's config validation does).
    pub fn spawn(registry: Arc<WorkerRegistry>, interval_secs: u64) -> Arc<Self> {
        let this = Arc::new(Self::new());
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                let interval = Duration::from_secs(interval_secs.max(1));
                tracing::info!(
                    target: LOG_TARGET,
                    interval_secs = interval.as_secs(),
                    "decode power-of-two-smg: load monitor started (GET /v1/loads?include=core on every worker)"
                );
                handle.spawn(monitor_loop(
                    Arc::downgrade(&this),
                    registry,
                    load_client(),
                    interval,
                ));
            }
            Err(_) => tracing::warn!(
                target: LOG_TARGET,
                "no tokio runtime at construction; decode load monitor not started"
            ),
        }
        this
    }

    /// Late D choose: poll and compare the bound view (see the field doc).
    /// Call before the first poll round; the monitor reads it every round.
    pub fn set_bound_view(&self, on: bool) {
        self.bound_view.store(on, Ordering::Relaxed);
    }

    pub fn bound_view(&self) -> bool {
        self.bound_view.load(Ordering::Relaxed)
    }

    /// Age of the token-load snapshot (since the last poll round); `None`
    /// before the first round.
    pub fn loads_age(&self) -> Option<Duration> {
        self.last_round
            .lock()
            .ok()
            .and_then(|t| t.map(|t| t.elapsed()))
    }

    /// Late D `po2` rule: the SMG power-of-two pick among
    /// `candidates[set[..]]` (the eligible candidates of one bind): two
    /// distinct uniform draws, [`Self::choose`] between them. One index is
    /// returned without a comparison; `None` for an empty set. Returns the
    /// index into `candidates` and the signal compared.
    pub fn pick_among(&self, candidates: &[Arc<Worker>], set: &[usize]) -> Option<(usize, Signal)> {
        match set.len() {
            0 => None,
            1 => Some((set[0], Signal::Single)),
            n => {
                let mut rng = rand::thread_rng();
                let r1 = rng.gen_range(0..n);
                let r2 = rng.gen_range(0..n - 1);
                let (i1, i2) = sample_pair(n, r1, r2);
                let (a, b) = (set[i1], set[i2]);
                let (picked, signal) = self.choose(&candidates[a], &candidates[b]);
                let winner = if Arc::ptr_eq(picked, &candidates[a]) {
                    a
                } else {
                    b
                };
                Some((winner, signal))
            }
        }
    }

    /// Pick a decode from `candidates` (SMG `select_worker`). `None` when no
    /// candidate is healthy (SMG: no available worker -> the pick fails).
    pub fn select(&self, candidates: &[Arc<Worker>]) -> Option<Arc<Worker>> {
        // SMG get_healthy_worker_indices: healthy and the breaker admits.
        let healthy_indices: Vec<usize> = candidates
            .iter()
            .enumerate()
            .filter(|(_, w)| w.breaker.would_allow())
            .map(|(idx, _)| idx)
            .collect();

        if healthy_indices.is_empty() {
            return None;
        }

        if healthy_indices.len() == 1 {
            self.stats.selections_single.fetch_add(1, Ordering::Relaxed);
            return Some(Arc::clone(&candidates[healthy_indices[0]]));
        }

        let mut rng = rand::thread_rng();
        let r1 = rng.gen_range(0..healthy_indices.len());
        let r2 = rng.gen_range(0..healthy_indices.len() - 1);
        let (idx1, idx2) = sample_pair(healthy_indices.len(), r1, r2);

        let worker1 = &candidates[healthy_indices[idx1]];
        let worker2 = &candidates[healthy_indices[idx2]];
        let (selected, _) = self.choose(worker1, worker2);
        Some(Arc::clone(selected))
    }

    /// The comparison of two sampled decodes (SMG `select_worker` after the
    /// sampling): token loads when both have one, else both request counts;
    /// `worker1` wins ties.
    pub fn choose<'a>(
        &self,
        worker1: &'a Arc<Worker>,
        worker2: &'a Arc<Worker>,
    ) -> (&'a Arc<Worker>, Signal) {
        let loads_guard = self.cached_loads.read().ok();

        let load1_tokens = loads_guard
            .as_ref()
            .and_then(|m| m.get(worker1.url.as_str()).copied());
        let load2_tokens = loads_guard
            .as_ref()
            .and_then(|m| m.get(worker2.url.as_str()).copied());

        // If either worker is missing token data, compare request counts
        // for BOTH (SMG: "degrade BOTH to request counts to ensure fairness").
        let (load1, load2, signal) = match (load1_tokens, load2_tokens) {
            (Some(t1), Some(t2)) => (t1, t2, Signal::Tokens),
            _ => (
                self.request_load(&worker1.url) as isize,
                self.request_load(&worker2.url) as isize,
                Signal::Requests,
            ),
        };

        let selected = if load1 <= load2 { worker1 } else { worker2 };

        match signal {
            Signal::Tokens => self.stats.selections_tokens.fetch_add(1, Ordering::Relaxed),
            Signal::Requests => self
                .stats
                .selections_requests
                .fetch_add(1, Ordering::Relaxed),
            Signal::Single => self.stats.selections_single.fetch_add(1, Ordering::Relaxed),
        };

        tracing::debug!(
            target: LOG_TARGET,
            "Power-of-two selection: {}={} vs {}={} ({:?}) -> selected {}",
            worker1.url,
            load1,
            worker2.url,
            load2,
            signal,
            selected.url
        );

        (selected, signal)
    }

    /// Replace the whole token-load snapshot (SMG `update_loads`): only
    /// non-negative loads are kept; workers not in `loads` become absent.
    pub fn update_loads(&self, loads: &HashMap<String, isize>) {
        if let Ok(mut cached) = self.cached_loads.write() {
            *cached = loads
                .iter()
                .filter(|(_, load)| **load >= 0)
                .map(|(worker, load)| (worker.clone(), *load))
                .collect();
        }
    }

    /// Token load of `url` in the current snapshot.
    pub fn token_load(&self, url: &str) -> Option<isize> {
        self.cached_loads
            .read()
            .ok()
            .and_then(|m| m.get(url).copied())
    }

    /// SMG request count (`Worker::load()`) of the decode at `url`.
    pub fn request_load(&self, url: &str) -> usize {
        self.request_loads
            .get(url)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    fn counter(&self, url: &str) -> Arc<AtomicUsize> {
        if let Some(c) = self.request_loads.get(url) {
            return Arc::clone(c.value());
        }
        Arc::clone(
            self.request_loads
                .entry(url.to_owned())
                .or_insert_with(|| Arc::new(AtomicUsize::new(0)))
                .value(),
        )
    }

    /// Request-count guard for a non-streaming decode leg: counted from now
    /// until the guard drops (SMG `_decode_guard` in
    /// `execute_dual_dispatch_internal`).
    pub fn count_from_dispatch(&self, url: &str) -> SmgLoadGuard {
        let counter = self.counter(url);
        counter.fetch_add(1, Ordering::Relaxed);
        self.stats
            .counted_at_dispatch
            .fetch_add(1, Ordering::Relaxed);
        SmgLoadGuard {
            counter,
            state: Arc::new(AtomicU8::new(ARMED)),
        }
    }

    /// Request-count guard for a streaming decode leg: counted only after
    /// [`SmgLoadArm::arm`] (call it when the decode response headers arrive,
    /// SMG `create_streaming_response`) and until the guard drops (end of the
    /// stream). If the guard drops first, a later `arm` counts nothing.
    pub fn count_from_headers(&self, url: &str) -> (SmgLoadGuard, SmgLoadArm) {
        let counter = self.counter(url);
        let state = Arc::new(AtomicU8::new(PENDING));
        (
            SmgLoadGuard {
                counter: Arc::clone(&counter),
                state: Arc::clone(&state),
            },
            SmgLoadArm {
                counter,
                state,
                stats: Arc::clone(&self.stats),
            },
        )
    }

    pub fn stats(&self) -> StatsSnapshot {
        let s = &self.stats;
        StatsSnapshot {
            selections_tokens: s.selections_tokens.load(Ordering::Relaxed),
            selections_requests: s.selections_requests.load(Ordering::Relaxed),
            selections_single: s.selections_single.load(Ordering::Relaxed),
            poll_rounds: s.poll_rounds.load(Ordering::Relaxed),
            poll_empty_rounds: s.poll_empty_rounds.load(Ordering::Relaxed),
            poll_worker_ok: s.poll_worker_ok.load(Ordering::Relaxed),
            poll_worker_failures: s.poll_worker_failures.load(Ordering::Relaxed),
            counted_at_dispatch: s.counted_at_dispatch.load(Ordering::Relaxed),
            counted_at_headers: s.counted_at_headers.load(Ordering::Relaxed),
        }
    }

    /// Prometheus text for `/metrics`.
    pub fn render_metrics(&self) -> String {
        let s = self.stats();
        let mut out = String::new();
        out.push_str("# HELP sgl_router_decode_po2_smg_selections_total Decode picks of --pd-decode-policy power-of-two-smg by the signal compared: tokens (both samples had a polled /v1/loads token load), requests (fallback: at least one had none, both request counts compared), single (one healthy decode).\n");
        out.push_str("# TYPE sgl_router_decode_po2_smg_selections_total counter\n");
        for (signal, v) in [
            ("tokens", s.selections_tokens),
            ("requests", s.selections_requests),
            ("single", s.selections_single),
        ] {
            out.push_str(&format!(
                "sgl_router_decode_po2_smg_selections_total{{signal=\"{signal}\"}} {v}\n"
            ));
        }
        for (name, help, v) in [
            (
                "sgl_router_decode_po2_smg_poll_rounds_total",
                "Load-monitor rounds (GET /v1/loads?include=core on every worker).",
                s.poll_rounds,
            ),
            (
                "sgl_router_decode_po2_smg_poll_empty_rounds_total",
                "Rounds that fetched no load (SMG warns \"No loads fetched from workers\"); the snapshot is then empty.",
                s.poll_empty_rounds,
            ),
            (
                "sgl_router_decode_po2_smg_poll_worker_failures_total",
                "Per-worker fetches with no value (transport error, timeout, non-2xx or unusable body).",
                s.poll_worker_failures,
            ),
        ] {
            out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} counter\n{name} {v}\n"));
        }
        let mut loads: Vec<(String, isize)> = self
            .cached_loads
            .read()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect())
            .unwrap_or_default();
        loads.sort();
        out.push_str("# HELP sgl_router_decode_po2_smg_token_load Token load of each worker in the current snapshot (absent: no value in the last round).\n");
        out.push_str("# TYPE sgl_router_decode_po2_smg_token_load gauge\n");
        for (url, v) in &loads {
            out.push_str(&format!(
                "sgl_router_decode_po2_smg_token_load{{worker_url=\"{}\"}} {v}\n",
                escape_label(url)
            ));
        }
        let mut counts: Vec<(String, usize)> = self
            .request_loads
            .iter()
            .map(|e| (e.key().clone(), e.value().load(Ordering::Relaxed)))
            .collect();
        counts.sort();
        out.push_str("# HELP sgl_router_decode_po2_smg_requests SMG request count of each decode worker (+1 at dispatch, or at the response headers for a streaming leg; -1 at the end).\n");
        out.push_str("# TYPE sgl_router_decode_po2_smg_requests gauge\n");
        for (url, v) in &counts {
            out.push_str(&format!(
                "sgl_router_decode_po2_smg_requests{{worker_url=\"{}\"}} {v}\n",
                escape_label(url)
            ));
        }
        out
    }
}

/// SMG sampling: `idx1 = r1`, `idx2 = (idx1 + 1 + r2) % n` with `r1` in
/// `0..n` and `r2` in `0..n-1`, so `idx2` is uniform over the other `n-1`.
pub fn sample_pair(n: usize, r1: usize, r2: usize) -> (usize, usize) {
    (r1, (r1 + 1 + r2) % n)
}

const PENDING: u8 = 0;
const ARMED: u8 = 1;
const DONE: u8 = 2;

/// One decode leg's share of the SMG request count (SMG `WorkerLoadGuard`):
/// -1 on drop if it was counted.
#[must_use = "SmgLoadGuard must be held for the decode leg's lifetime"]
#[derive(Debug)]
pub struct SmgLoadGuard {
    counter: Arc<AtomicUsize>,
    state: Arc<AtomicU8>,
}

impl Drop for SmgLoadGuard {
    fn drop(&mut self) {
        if self.state.swap(DONE, Ordering::AcqRel) == ARMED {
            decrement(&self.counter);
        }
    }
}

/// Starts the count of a deferred [`SmgLoadGuard`]
/// ([`SmgPowerOfTwo::count_from_headers`]).
#[derive(Debug)]
pub struct SmgLoadArm {
    counter: Arc<AtomicUsize>,
    state: Arc<AtomicU8>,
    stats: Arc<Stats>,
}

impl SmgLoadArm {
    /// +1 now, unless the guard has already dropped (the stream ended
    /// first): then nothing stays counted.
    pub fn arm(self) {
        // Count first: a guard drop that sees ARMED must find the +1 applied.
        self.counter.fetch_add(1, Ordering::Relaxed);
        if self
            .state
            .compare_exchange(PENDING, ARMED, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.stats
                .counted_at_headers
                .fetch_add(1, Ordering::Relaxed);
        } else {
            decrement(&self.counter);
        }
    }
}

/// SMG `decrement_load`: never below 0.
fn decrement(counter: &AtomicUsize) {
    if counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |c| c.checked_sub(1))
        .is_err()
    {
        tracing::warn!(
            target: LOG_TARGET,
            "Attempted to decrement a decode request count that is already at 0"
        );
    }
}

/// HTTP client for the `/v1/loads` polls: sgl-model-gateway's client
/// defaults (`config/types.rs`: connect timeout 10 s, TCP
/// keepalive 30 s, nodelay), but no idle connection is kept, so every poll
/// opens a new connection. Each request also gets SMG's 5 s
/// `REQUEST_TIMEOUT`.
///
/// Why no idle pool: this client carries only the polls, so a pooled
/// connection would be reused after ~5 s idle (the poll interval), right
/// when the worker's 5 s uvicorn keep-alive closes it; the poll then fails
/// at once (observed: 20 immediate failures in 180 rounds). SMG's monitor
/// shares its pool with the request traffic, so its polls mostly reuse a
/// connection idle for less than 5 s. The proxy client avoids the same race
/// with a zero idle pool.
fn load_client() -> reqwest::Client {
    reqwest::Client::builder()
        .pool_max_idle_per_host(0)
        .connect_timeout(Duration::from_secs(10))
        .tcp_nodelay(true)
        .tcp_keepalive(Some(Duration::from_secs(30)))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

/// SMG `parse_load_value`.
pub fn parse_load_value(json: &Value) -> Option<isize> {
    fn nonnegative_isize(value: &Value) -> Option<isize> {
        isize::try_from(value.as_u64()?).ok()
    }

    match json.get("loads") {
        Some(loads) => {
            let loads = loads.as_array()?;
            if loads.is_empty() {
                return None;
            }
            loads.iter().try_fold(0isize, |total, load| {
                let rank_load = nonnegative_isize(load.get("num_total_tokens")?)?;
                total.checked_add(rank_load)
            })
        }
        None => nonnegative_isize(json.get("aggregate")?.get("total_tokens")?),
    }
}

/// Late D choose: [`parse_load_value`] on the bound view: per rank
/// `late_d.bound_total_tokens` when present, else `num_total_tokens` (equal
/// when nothing is DEFERRED); the legacy aggregate as before.
pub fn parse_bound_load_value(json: &Value) -> Option<isize> {
    fn nonnegative_isize(value: &Value) -> Option<isize> {
        isize::try_from(value.as_u64()?).ok()
    }

    match json.get("loads") {
        Some(loads) => {
            let loads = loads.as_array()?;
            if loads.is_empty() {
                return None;
            }
            loads.iter().try_fold(0isize, |total, load| {
                let bound = load
                    .get("late_d")
                    .and_then(|late_d| late_d.get("bound_total_tokens"))
                    .filter(|v| !v.is_null());
                let rank_load = nonnegative_isize(bound.or_else(|| load.get("num_total_tokens"))?)?;
                total.checked_add(rank_load)
            })
        }
        None => nonnegative_isize(json.get("aggregate")?.get("total_tokens")?),
    }
}

/// SMG `parse_load_response`: the worker's load, or why there is none.
/// SMG maps every failure to -1 (no value); so does `poll_round`.
async fn fetch_load(client: &reqwest::Client, url: &str) -> Result<isize, String> {
    fetch_load_with_timeout(client, url, LOAD_REQUEST_TIMEOUT).await
}

async fn fetch_load_with_timeout(
    client: &reqwest::Client,
    url: &str,
    timeout: Duration,
) -> Result<isize, String> {
    fetch_load_view(client, url, timeout, false).await
}

/// One load fetch; `bound_view` (late D choose) adds the `disagg` section
/// and reads the bound view ([`parse_bound_load_value`]).
async fn fetch_load_view(
    client: &reqwest::Client,
    url: &str,
    timeout: Duration,
    bound_view: bool,
) -> Result<isize, String> {
    let load_url = if bound_view {
        format!("{}/v1/loads?include=core,disagg", url)
    } else {
        format!("{}/v1/loads?include=core", url)
    };
    let response = client
        .get(&load_url)
        .timeout(timeout)
        .send()
        .await
        .map_err(|e| poll_error_kind(&e))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("status {}", status.as_u16()));
    }
    let json = response
        .json::<Value>()
        .await
        .map_err(|e| poll_error_kind(&e))?;
    let value = if bound_view {
        parse_bound_load_value(&json)
    } else {
        parse_load_value(&json)
    };
    value.ok_or_else(|| "parse (no load value)".to_string())
}

/// Error-chain fragments of a connection the worker closed or reset.
const CLOSED_HINTS: [&str; 4] = [
    "connection closed",
    "connection reset",
    "broken pipe",
    "incompletemessage",
];

/// Short cause of a failed `/v1/loads` request: `timeout`, `connect`,
/// `closed` (connection closed or reset by the worker), `parse` (body is
/// not JSON) or `request`, followed by the error chain.
fn poll_error_kind(e: &reqwest::Error) -> String {
    let mut chain = Vec::new();
    let mut closed = false;
    let mut source = std::error::Error::source(e);
    while let Some(s) = source {
        closed |= is_closed_io(s);
        chain.push(s.to_string());
        source = s.source();
    }
    let detail = chain.join(": ");
    let lower = detail.to_ascii_lowercase();
    closed |= CLOSED_HINTS.iter().any(|hint| lower.contains(hint));
    let kind = if e.is_timeout() {
        "timeout"
    } else if e.is_connect() {
        "connect"
    } else if closed {
        "closed"
    } else if e.is_decode() {
        "parse"
    } else {
        "request"
    };
    let detail: String = detail.chars().take(200).collect();
    if detail.is_empty() {
        kind.to_string()
    } else {
        format!("{kind} ({detail})")
    }
}

/// An I/O error of a connection the worker closed or reset.
fn is_closed_io(source: &(dyn std::error::Error + 'static)) -> bool {
    use std::io::ErrorKind::*;
    let Some(io) = source.downcast_ref::<std::io::Error>() else {
        return false;
    };
    matches!(
        io.kind(),
        ConnectionReset | ConnectionAborted | BrokenPipe | UnexpectedEof
    )
}

/// One SMG load-monitor round: fetch every registered worker at once, keep
/// the non-negative loads, replace the snapshot (also when it is empty).
pub async fn poll_round(
    policy: &SmgPowerOfTwo,
    registry: &WorkerRegistry,
    client: &reqwest::Client,
) -> RoundSummary {
    let urls: Vec<String> = registry.all().into_iter().map(|w| w.url.clone()).collect();
    let bound_view = policy.bound_view();
    let results: Vec<(String, Result<isize, String>)> =
        futures::future::join_all(urls.into_iter().map(|url| async move {
            let load = if bound_view {
                fetch_load_view(client, &url, LOAD_REQUEST_TIMEOUT, true).await
            } else {
                fetch_load(client, &url).await
            };
            (url, load)
        }))
        .await;

    let loads: HashMap<String, isize> = results
        .iter()
        .filter_map(|(url, load)| match load {
            Ok(load) if *load >= 0 => Some((url.clone(), *load)),
            _ => None,
        })
        .collect();
    let failed: Vec<&str> = results
        .iter()
        .filter(|(url, _)| !loads.contains_key(url))
        .map(|(url, _)| url.as_str())
        .collect();
    for (url, load) in &results {
        if let Err(error) = load {
            tracing::warn!(
                target: LOG_TARGET,
                worker = %url,
                error = %error,
                "/v1/loads poll failed"
            );
        }
    }
    let summary = RoundSummary {
        polled: results.len(),
        ok: loads.len(),
        failed: failed.len(),
    };

    let stats = &policy.stats;
    stats.poll_rounds.fetch_add(1, Ordering::Relaxed);
    stats
        .poll_worker_ok
        .fetch_add(summary.ok as u64, Ordering::Relaxed);
    stats
        .poll_worker_failures
        .fetch_add(summary.failed as u64, Ordering::Relaxed);
    if loads.is_empty() {
        stats.poll_empty_rounds.fetch_add(1, Ordering::Relaxed);
        tracing::warn!(target: LOG_TARGET, "No loads fetched from workers");
    } else if !failed.is_empty() {
        tracing::warn!(
            target: LOG_TARGET,
            failed = ?failed,
            "no /v1/loads value from some workers; they are absent from this round's snapshot"
        );
    }

    // Replace the whole snapshot, including with an empty map, so stale
    // token loads never outlive a failed round (SMG monitor_loop).
    policy.update_loads(&loads);
    if let Ok(mut last) = policy.last_round.lock() {
        *last = Some(Instant::now());
    }
    summary
}

async fn monitor_loop(
    weak: Weak<SmgPowerOfTwo>,
    registry: Arc<WorkerRegistry>,
    client: reqwest::Client,
    interval: Duration,
) {
    let mut interval_timer = tokio::time::interval(interval);
    loop {
        interval_timer.tick().await;
        let Some(policy) = weak.upgrade() else {
            return;
        };
        let summary = poll_round(&policy, &registry, &client).await;
        let s = policy.stats();
        tracing::info!(
            target: LOG_TARGET,
            round = s.poll_rounds,
            polled = summary.polled,
            ok = summary.ok,
            failed = summary.failed,
            sel_tokens = s.selections_tokens,
            sel_requests = s.selections_requests,
            sel_single = s.selections_single,
            poll_failures = s.poll_worker_failures,
            empty_rounds = s.poll_empty_rounds,
            "decode power-of-two-smg load round"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PdDecodePolicy;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::registry::PdPoolResolver;
    use serde_json::json;
    use std::collections::HashSet;

    fn spec(url: &str) -> WorkerSpec {
        WorkerSpec {
            id: WorkerId(url.into()),
            url: url.into(),
            mode: WorkerMode::Decode,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
        }
    }

    fn decode(url: &str) -> Arc<Worker> {
        Arc::new(Worker::new(spec(url)))
    }

    /// `n` request-count guards on `url` (SMG tests: `increment_load` n times).
    fn hold(p: &SmgPowerOfTwo, url: &str, n: usize) -> Vec<SmgLoadGuard> {
        (0..n).map(|_| p.count_from_dispatch(url)).collect()
    }

    fn loads(pairs: &[(&str, isize)]) -> HashMap<String, isize> {
        pairs.iter().map(|(u, l)| (u.to_string(), *l)).collect()
    }

    /// `choose` as (URL of the pick, signal).
    fn pick(p: &SmgPowerOfTwo, w1: &Arc<Worker>, w2: &Arc<Worker>) -> (String, Signal) {
        let (w, s) = p.choose(w1, w2);
        (w.url.clone(), s)
    }

    // ---- parse: SMG worker_manager.rs tests, same fixtures ----

    #[test]
    fn parse_rank_loads_sums_total_tokens() {
        let value = json!({
            "loads": [
                {"dp_rank": 0, "num_total_tokens": 12},
                {"dp_rank": 1, "num_total_tokens": 30}
            ]
        });
        assert_eq!(parse_load_value(&value), Some(42));
    }

    #[test]
    fn parse_rank_loads_takes_precedence_over_legacy_aggregate() {
        let value = json!({
            "loads": [{"dp_rank": 0, "num_total_tokens": 7}],
            "aggregate": {"total_tokens": 999}
        });
        assert_eq!(parse_load_value(&value), Some(7));
    }

    #[test]
    fn parse_legacy_aggregate_remains_supported() {
        let value = json!({"aggregate": {"total_tokens": 17}});
        assert_eq!(parse_load_value(&value), Some(17));
    }

    #[test]
    fn parse_rank_loads_rejects_empty_or_malformed_snapshots() {
        for value in [
            json!({"loads": []}),
            json!({"loads": {"num_total_tokens": 1}}),
            json!({"loads": [{"dp_rank": 0}]}),
            json!({"loads": [{"num_total_tokens": -1}]}),
            json!({"loads": [{"num_total_tokens": "1"}]}),
        ] {
            assert_eq!(parse_load_value(&value), None);
        }
    }

    #[test]
    fn parse_rank_loads_rejects_overflow_without_legacy_fallback() {
        let value = json!({
            "loads": [
                {"num_total_tokens": isize::MAX as u64},
                {"num_total_tokens": 1}
            ],
            "aggregate": {"total_tokens": 3}
        });
        assert_eq!(parse_load_value(&value), None);
    }

    /// The engine's own shape (one rank, extra fields, no `aggregate`), and a
    /// `loads` key that is not an array never falls back to `aggregate`.
    #[test]
    fn parse_engine_shape_and_non_array_loads() {
        let value = json!({
            "timestamp": "2026-10-09T13:50:00Z", "version": "0.5.19",
            "accelerator": "NVIDIA GB300", "num_accelerators": 4,
            "loads": [{"dp_rank": 0, "num_running_reqs": 3, "num_waiting_reqs": 0,
                       "num_used_tokens": 1183000, "num_total_tokens": 1183872,
                       "max_total_num_tokens": 2067584, "token_usage": 0.57}]
        });
        assert_eq!(parse_load_value(&value), Some(1_183_872));
        let value = json!({"loads": null, "aggregate": {"total_tokens": 5}});
        assert_eq!(parse_load_value(&value), None);
    }

    // ---- sampling ----

    #[test]
    fn sample_pair_covers_every_ordered_pair_exactly_once() {
        for n in 2..=7 {
            let mut seen = HashSet::new();
            for r1 in 0..n {
                for r2 in 0..n - 1 {
                    let (a, b) = sample_pair(n, r1, r2);
                    assert!(a < n && b < n && a != b, "n={n} r1={r1} r2={r2}");
                    assert!(seen.insert((a, b)), "pair ({a}, {b}) drawn twice for n={n}");
                }
            }
            assert_eq!(seen.len(), n * (n - 1));
        }
    }

    // ---- comparison ----

    /// SMG test_power_of_two_edge_cases, scenario 1: both have token loads,
    /// so tokens decide even against the request counts.
    #[test]
    fn choose_compares_tokens_when_both_samples_have_one() {
        let p = SmgPowerOfTwo::new();
        let (a, b) = (decode("http://a:8000"), decode("http://b:8000"));
        let _ga = hold(&p, "http://a:8000", 10);
        let _gb = hold(&p, "http://b:8000", 2);
        p.update_loads(&loads(&[
            ("http://a:8000", 1_000),
            ("http://b:8000", 100_000),
        ]));
        assert_eq!(pick(&p, &a, &b), (a.url.clone(), Signal::Tokens));
        assert_eq!(pick(&p, &b, &a), (a.url.clone(), Signal::Tokens));
        assert_eq!(p.stats().selections_tokens, 2);
        assert_eq!(p.stats().selections_requests, 0);
    }

    /// SMG scenarios 2 and 3 and test_reproduce_incompatible_metric_bug: one
    /// sample without a token load sends BOTH to request counts.
    #[test]
    fn choose_falls_back_to_request_counts_for_both_when_one_is_missing() {
        let p = SmgPowerOfTwo::new();
        let (c, d) = (decode("http://c:8000"), decode("http://d:8000"));
        let _gc = hold(&p, "http://c:8000", 10);
        let _gd = hold(&p, "http://d:8000", 2);
        p.update_loads(&loads(&[("http://c:8000", 1_000)]));
        assert_eq!(pick(&p, &c, &d), (d.url.clone(), Signal::Requests));
        assert_eq!(pick(&p, &d, &c), (d.url.clone(), Signal::Requests));

        let (e, f) = (decode("http://e:8000"), decode("http://f:8000"));
        let _ge = hold(&p, "http://e:8000", 2);
        let _gf = hold(&p, "http://f:8000", 10);
        p.update_loads(&loads(&[("http://f:8000", 1_000)]));
        assert_eq!(pick(&p, &e, &f), (e.url.clone(), Signal::Requests));

        // Idle A with a big token load vs busy B without one: counts (0 vs 5).
        let (wa, wb) = (
            decode("http://worker_a:8000"),
            decode("http://worker_b:8000"),
        );
        let _gb = hold(&p, "http://worker_b:8000", 5);
        p.update_loads(&loads(&[("http://worker_a:8000", 50_000)]));
        assert_eq!(pick(&p, &wa, &wb), (wa.url.clone(), Signal::Requests));
        assert_eq!(p.stats().selections_requests, 4);
        assert_eq!(p.stats().selections_tokens, 0);
    }

    /// SMG scenario 4 and test_negative_token_load_is_treated_as_missing.
    #[test]
    fn choose_uses_request_counts_when_both_are_missing_or_negative() {
        let p = SmgPowerOfTwo::new();
        let (g, h) = (decode("http://g:8000"), decode("http://h:8000"));
        let _gg = hold(&p, "http://g:8000", 5);
        let _gh = hold(&p, "http://h:8000", 3);
        p.update_loads(&HashMap::new());
        assert_eq!(pick(&p, &g, &h), (h.url.clone(), Signal::Requests));

        let (a, b) = (decode("http://a:8000"), decode("http://b:8000"));
        let _gb = hold(&p, "http://b:8000", 5);
        p.update_loads(&loads(&[("http://a:8000", 100), ("http://b:8000", -1)]));
        assert_eq!(p.token_load("http://b:8000"), None);
        assert_eq!(pick(&p, &a, &b), (a.url.clone(), Signal::Requests));
    }

    /// `load1 <= load2` keeps the first sample, for tokens and for counts.
    #[test]
    fn choose_tie_goes_to_the_first_sample() {
        let p = SmgPowerOfTwo::new();
        let (a, b) = (decode("http://a:8000"), decode("http://b:8000"));
        p.update_loads(&loads(&[("http://a:8000", 7), ("http://b:8000", 7)]));
        assert_eq!(pick(&p, &a, &b), (a.url.clone(), Signal::Tokens));
        assert_eq!(pick(&p, &b, &a), (b.url.clone(), Signal::Tokens));
        p.update_loads(&HashMap::new());
        let _ga = hold(&p, "http://a:8000", 2);
        let _gb = hold(&p, "http://b:8000", 2);
        assert_eq!(pick(&p, &a, &b), (a.url.clone(), Signal::Requests));
        assert_eq!(pick(&p, &b, &a), (b.url.clone(), Signal::Requests));
    }

    // ---- selection ----

    #[test]
    fn select_single_healthy_worker_and_empty_pool() {
        let p = SmgPowerOfTwo::new();
        assert!(p.select(&[]).is_none());
        let w = decode("http://w1:8000");
        // SMG test_power_of_two_single_worker.
        assert_eq!(p.select(&[Arc::clone(&w)]).unwrap().url, "http://w1:8000");
        // Two candidates, one breaker open: the healthy one, no sampling.
        let open = decode("http://w2:8000");
        for _ in 0..3 {
            open.breaker.record_failure();
        }
        assert!(!open.breaker.would_allow());
        for _ in 0..20 {
            let got = p.select(&[Arc::clone(&open), Arc::clone(&w)]).unwrap();
            assert_eq!(got.url, "http://w1:8000");
        }
        // Every breaker open: no pick (SMG: no available worker).
        assert!(p.select(&[Arc::clone(&open)]).is_none());
        assert_eq!(p.stats().selections_single, 21);
        assert_eq!(
            p.stats().selections_tokens + p.stats().selections_requests,
            0
        );
    }

    /// SMG test_power_of_two_with_cached_loads: two decodes are always both
    /// sampled, so the lower token load always wins.
    #[test]
    fn select_two_decodes_takes_the_lower_token_load() {
        let p = SmgPowerOfTwo::new();
        let ws = vec![decode("http://w1:8000"), decode("http://w2:8000")];
        p.update_loads(&loads(&[("http://w1:8000", 100), ("http://w2:8000", 10)]));
        for _ in 0..50 {
            assert_eq!(p.select(&ws).unwrap().url, "http://w2:8000");
        }
        assert_eq!(p.stats().selections_tokens, 50);
    }

    /// SMG test_power_of_two_selection (request counts 10 / 5 / 0).
    #[test]
    fn select_three_decodes_by_request_count() {
        let p = SmgPowerOfTwo::new();
        let ws = vec![
            decode("http://w1:8000"),
            decode("http://w2:8000"),
            decode("http://w3:8000"),
        ];
        let _g1 = hold(&p, "http://w1:8000", 10);
        let _g2 = hold(&p, "http://w2:8000", 5);
        let mut counts = [0usize; 3];
        for _ in 0..100 {
            let got = p.select(&ws).unwrap();
            counts[ws.iter().position(|w| w.url == got.url).unwrap()] += 1;
        }
        assert!(counts[2] > counts[1], "{counts:?}");
        assert!(counts[1] > counts[0], "{counts:?}");
        assert_eq!(
            counts[0], 0,
            "the busiest decode never wins a pair: {counts:?}"
        );
    }

    /// `decode_select` (the stateless resolver path) never serves this
    /// policy; `AppContext::select_decode` does.
    #[test]
    fn resolver_decode_select_refuses_power_of_two_smg() {
        let r = Arc::new(WorkerRegistry::default());
        r.add(spec("http://d1:8000")).unwrap();
        let resolver = PdPoolResolver::new(r);
        assert!(resolver
            .decode_select(
                &ModelId("m".into()),
                "http://p:1",
                PdDecodePolicy::PowerOfTwoSmg
            )
            .is_err());
    }

    // ---- snapshot ----

    #[test]
    fn update_loads_replaces_the_whole_snapshot() {
        let p = SmgPowerOfTwo::new();
        p.update_loads(&loads(&[("http://a", 1), ("http://b", 2)]));
        assert_eq!(p.token_load("http://a"), Some(1));
        assert_eq!(p.token_load("http://b"), Some(2));
        p.update_loads(&loads(&[("http://a", 3)]));
        assert_eq!(p.token_load("http://a"), Some(3));
        assert_eq!(p.token_load("http://b"), None, "absent from the new round");
        p.update_loads(&loads(&[("http://a", -1)]));
        assert_eq!(p.token_load("http://a"), None, "negative loads are dropped");
        p.update_loads(&loads(&[("http://a", 4)]));
        p.update_loads(&HashMap::new());
        assert_eq!(
            p.token_load("http://a"),
            None,
            "an empty round clears the snapshot"
        );
    }

    // ---- request count ----

    #[test]
    fn request_count_from_dispatch_and_from_headers() {
        let p = SmgPowerOfTwo::new();
        let u = "http://d:1";
        assert_eq!(p.request_load(u), 0);
        let g = p.count_from_dispatch(u);
        assert_eq!(p.request_load(u), 1);
        drop(g);
        assert_eq!(p.request_load(u), 0);

        // Streaming: nothing until the headers arm it, then until the drop.
        let (g, arm) = p.count_from_headers(u);
        assert_eq!(p.request_load(u), 0);
        arm.arm();
        assert_eq!(p.request_load(u), 1);
        drop(g);
        assert_eq!(p.request_load(u), 0);

        // The stream ended (guard dropped) before the arm: nothing counted.
        let (g, arm) = p.count_from_headers(u);
        drop(g);
        arm.arm();
        assert_eq!(p.request_load(u), 0);

        // Never armed (transport error / stale cancel): nothing counted.
        let (g, arm) = p.count_from_headers(u);
        drop(arm);
        drop(g);
        assert_eq!(p.request_load(u), 0);

        let s = p.stats();
        assert_eq!((s.counted_at_dispatch, s.counted_at_headers), (1, 1));
    }

    #[test]
    fn request_count_arm_and_drop_race_never_leaks() {
        let p = Arc::new(SmgPowerOfTwo::new());
        let u = "http://d:1";
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let p = Arc::clone(&p);
                std::thread::spawn(move || {
                    for i in 0..500 {
                        let (g, arm) = p.count_from_headers(u);
                        if i % 2 == 0 {
                            let t = std::thread::spawn(move || arm.arm());
                            drop(g);
                            t.join().unwrap();
                        } else {
                            let t = std::thread::spawn(move || drop(g));
                            arm.arm();
                            t.join().unwrap();
                        }
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(p.request_load(u), 0);
    }

    // ---- load monitor ----

    async fn loads_server(
        status: u16,
        body: &'static str,
    ) -> (String, Arc<std::sync::Mutex<Vec<Option<String>>>>) {
        let seen: Arc<std::sync::Mutex<Vec<Option<String>>>> = Arc::default();
        let seen_in = Arc::clone(&seen);
        let app = axum::Router::new().route(
            "/v1/loads",
            axum::routing::get(move |axum::extract::RawQuery(q): axum::extract::RawQuery| {
                let seen = Arc::clone(&seen_in);
                async move {
                    seen.lock().unwrap().push(q);
                    (
                        axum::http::StatusCode::from_u16(status).unwrap(),
                        [(axum::http::header::CONTENT_TYPE, "application/json")],
                        body,
                    )
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (url, seen)
    }

    async fn refused_url() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        url
    }

    const GOOD: &str = r#"{"version":"0.5.19","loads":[{"dp_rank":0,"num_total_tokens":1183872},{"dp_rank":1,"num_total_tokens":128}]}"#;
    const NO_TOKENS: &str = r#"{"loads":[{"dp_rank":0,"num_used_tokens":5}]}"#;

    #[tokio::test]
    async fn poll_round_fetches_every_worker_and_replaces_the_snapshot() {
        let (good, seen) = loads_server(200, GOOD).await;
        let (bad_status, _) = loads_server(500, GOOD).await;
        let (bad_body, _) = loads_server(200, NO_TOKENS).await;
        let dead = refused_url().await;
        let registry = WorkerRegistry::default();
        for (i, url) in [&good, &bad_status, &bad_body, &dead].iter().enumerate() {
            let mut s = spec(url);
            // Prefill workers are polled too (SMG polls every registered worker).
            if i == 0 {
                s.mode = WorkerMode::Prefill;
            }
            registry.add(s).unwrap();
        }
        let p = SmgPowerOfTwo::new();
        // A stale value for a worker that fails this round must disappear.
        p.update_loads(&loads(&[(dead.as_str(), 9), (bad_body.as_str(), 9)]));
        let client = load_client();
        let round = poll_round(&p, &registry, &client).await;
        assert_eq!(
            round,
            RoundSummary {
                polled: 4,
                ok: 1,
                failed: 3
            }
        );
        assert_eq!(p.token_load(&good), Some(1_183_872 + 128));
        assert_eq!(p.token_load(&dead), None);
        assert_eq!(p.token_load(&bad_body), None);
        assert_eq!(p.token_load(&bad_status), None);
        assert_eq!(
            seen.lock().unwrap().as_slice(),
            &[Some("include=core".to_string())]
        );
        let s = p.stats();
        assert_eq!(
            (s.poll_rounds, s.poll_worker_ok, s.poll_worker_failures),
            (1, 1, 3)
        );
        assert_eq!(s.poll_empty_rounds, 0);

        // A round with no value at all empties the snapshot.
        let registry = WorkerRegistry::default();
        registry.add(spec(&dead)).unwrap();
        let round = poll_round(&p, &registry, &client).await;
        assert_eq!(
            round,
            RoundSummary {
                polled: 1,
                ok: 0,
                failed: 1
            }
        );
        assert_eq!(p.token_load(&good), None);
        assert_eq!(p.stats().poll_empty_rounds, 1);
    }

    // ---- late D choose: bound view and the po2 rule among a set ----

    const BOUND: &str = r#"{"loads":[{"dp_rank":0,"num_total_tokens":5000,"late_d":{"deferred_reqs":2,"deferred_tokens":3000,"bound_total_tokens":2000}},{"dp_rank":1,"num_total_tokens":700}]}"#;

    #[test]
    fn parse_bound_load_value_reads_the_bound_view() {
        let v: Value = serde_json::from_str(BOUND).unwrap();
        assert_eq!(parse_bound_load_value(&v), Some(2000 + 700));
        // The plain parse keeps reading num_total_tokens.
        assert_eq!(parse_load_value(&v), Some(5000 + 700));
        // A null bound value falls back to num_total_tokens.
        let v =
            json!({"loads": [{"num_total_tokens": 10, "late_d": {"bound_total_tokens": null}}]});
        assert_eq!(parse_bound_load_value(&v), Some(10));
        let v = json!({"loads": [{"num_total_tokens": 10, "late_d": null}]});
        assert_eq!(parse_bound_load_value(&v), Some(10));
        // Malformed or empty snapshots and the legacy aggregate, as before.
        assert_eq!(parse_bound_load_value(&json!({"loads": []})), None);
        assert_eq!(
            parse_bound_load_value(&json!({"loads": [{"late_d": {"bound_total_tokens": -1}}]})),
            None
        );
        assert_eq!(
            parse_bound_load_value(&json!({"aggregate": {"total_tokens": 42}})),
            Some(42)
        );
        assert_eq!(parse_bound_load_value(&json!({"loads": 5})), None);
    }

    #[tokio::test]
    async fn poll_round_bound_view_asks_for_disagg_and_reads_bound_tokens() {
        let (url, seen) = loads_server(200, BOUND).await;
        let registry = WorkerRegistry::default();
        registry.add(spec(&url)).unwrap();
        let p = SmgPowerOfTwo::new();
        assert!(!p.bound_view());
        assert_eq!(p.loads_age(), None);
        p.set_bound_view(true);
        let client = load_client();
        let round = poll_round(&p, &registry, &client).await;
        assert_eq!(round.ok, 1);
        assert_eq!(p.token_load(&url), Some(2000 + 700));
        assert!(p
            .loads_age()
            .is_some_and(|age| age < Duration::from_secs(5)));
        // Off again: the plain query and the plain value.
        p.set_bound_view(false);
        poll_round(&p, &registry, &client).await;
        assert_eq!(p.token_load(&url), Some(5000 + 700));
        assert_eq!(
            seen.lock().unwrap().as_slice(),
            &[
                Some("include=core,disagg".to_string()),
                Some("include=core".to_string())
            ]
        );
    }

    #[test]
    fn pick_among_compares_two_of_the_set() {
        let p = SmgPowerOfTwo::new();
        let ws: Vec<Arc<Worker>> = (0..4).map(|i| decode(&format!("http://d{i}:1"))).collect();
        p.update_loads(&loads(&[
            ("http://d0:1", 10),
            ("http://d1:1", 900),
            ("http://d2:1", 5),
            ("http://d3:1", 300),
        ]));
        for _ in 0..50 {
            // d0 and d2 are lighter but not in the set: d3 wins.
            assert_eq!(p.pick_among(&ws, &[1, 3]), Some((3, Signal::Tokens)));
        }
        assert_eq!(p.pick_among(&ws, &[2]), Some((2, Signal::Single)));
        assert_eq!(p.pick_among(&ws, &[]), None);
        // Three in the set: the heaviest never wins.
        let mut seen = HashSet::new();
        for _ in 0..200 {
            let (w, _) = p.pick_among(&ws, &[0, 1, 3]).unwrap();
            assert_ne!(w, 1);
            seen.insert(w);
        }
        assert_eq!(seen, HashSet::from([0, 3]));
        // Without polled loads: SMG request counts.
        let p = SmgPowerOfTwo::new();
        let _busy = hold(&p, "http://d1:1", 3);
        assert_eq!(p.pick_among(&ws, &[1, 2]), Some((2, Signal::Requests)));
    }

    #[derive(Clone, Copy)]
    enum RawMode {
        /// Read the request; never answer.
        Silent,
        /// Read the request; close the connection without an answer.
        Close,
        /// Answer every request with `GOOD`; keep the connection open.
        KeepAlive,
    }

    /// Raw TCP worker stub: its URL and its count of accepted connections.
    async fn raw_server(mode: RawMode) -> (String, Arc<AtomicUsize>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let accepts = Arc::new(AtomicUsize::new(0));
        let accepts_in = Arc::clone(&accepts);
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                accepts_in.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    loop {
                        let mut head = Vec::new();
                        while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                            match sock.read(&mut buf).await {
                                Ok(0) | Err(_) => return,
                                Ok(n) => head.extend_from_slice(&buf[..n]),
                            }
                        }
                        match mode {
                            RawMode::Silent => {
                                tokio::time::sleep(Duration::from_secs(30)).await;
                                return;
                            }
                            RawMode::Close => return,
                            RawMode::KeepAlive => {
                                let response = format!(
                                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                                     content-length: {}\r\n\r\n{GOOD}",
                                    GOOD.len()
                                );
                                if sock.write_all(response.as_bytes()).await.is_err() {
                                    return;
                                }
                            }
                        }
                    }
                });
            }
        });
        (url, accepts)
    }

    #[tokio::test]
    async fn fetch_load_reports_the_error_kind() {
        let client = load_client();
        let (good, _) = loads_server(200, GOOD).await;
        assert_eq!(fetch_load(&client, &good).await, Ok(1_183_872 + 128));
        let (bad_status, _) = loads_server(503, GOOD).await;
        assert_eq!(
            fetch_load(&client, &bad_status).await,
            Err("status 503".to_string())
        );
        let (bad_body, _) = loads_server(200, NO_TOKENS).await;
        assert_eq!(
            fetch_load(&client, &bad_body).await,
            Err("parse (no load value)".to_string())
        );
        let (not_json, _) = loads_server(200, "not json").await;
        let e = fetch_load(&client, &not_json).await.unwrap_err();
        assert!(e.starts_with("parse"), "{e}");
        let dead = refused_url().await;
        let e = fetch_load(&client, &dead).await.unwrap_err();
        assert!(e.starts_with("connect"), "{e}");
        let (silent, _) = raw_server(RawMode::Silent).await;
        let e = fetch_load_with_timeout(&client, &silent, Duration::from_millis(300))
            .await
            .unwrap_err();
        assert!(e.starts_with("timeout"), "{e}");
        let (closer, _) = raw_server(RawMode::Close).await;
        let e = fetch_load(&client, &closer).await.unwrap_err();
        assert!(e.starts_with("closed"), "{e}");
    }

    /// The poll client keeps no idle connection: every poll opens a new
    /// one, so no poll reuses a connection the worker's keep-alive timer is
    /// closing. The control shows the stub would allow reuse.
    #[tokio::test]
    async fn poll_client_opens_a_new_connection_for_every_poll() {
        let (url, accepts) = raw_server(RawMode::KeepAlive).await;
        let client = load_client();
        for _ in 0..3 {
            assert_eq!(fetch_load(&client, &url).await, Ok(1_183_872 + 128));
        }
        assert_eq!(accepts.load(Ordering::SeqCst), 3, "one connection per poll");

        let (url, accepts) = raw_server(RawMode::KeepAlive).await;
        let pooling = reqwest::Client::new();
        for _ in 0..3 {
            assert_eq!(fetch_load(&pooling, &url).await, Ok(1_183_872 + 128));
        }
        assert_eq!(
            accepts.load(Ordering::SeqCst),
            1,
            "control: a pooling client reuses the stub's connection"
        );
    }

    #[tokio::test]
    async fn spawned_monitor_polls_at_start_and_then_every_interval() {
        let (good, seen) = loads_server(200, GOOD).await;
        let registry = Arc::new(WorkerRegistry::default());
        registry.add(spec(&good)).unwrap();
        let p = SmgPowerOfTwo::spawn(Arc::clone(&registry), 1);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        while p.stats().poll_rounds < 2 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "monitor did not poll twice"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert_eq!(p.token_load(&good), Some(1_183_872 + 128));
        assert!(seen.lock().unwrap().len() >= 2);
        let m = p.render_metrics();
        assert!(
            m.contains("sgl_router_decode_po2_smg_poll_rounds_total"),
            "{m}"
        );
        assert!(
            m.contains(&format!(
                "sgl_router_decode_po2_smg_token_load{{worker_url=\"{good}\"}} 1184000"
            )),
            "{m}"
        );
        // The monitor stops with the policy.
        let rounds = p.stats().poll_rounds;
        drop(p);
        tokio::time::sleep(Duration::from_millis(2_500)).await;
        assert!(seen.lock().unwrap().len() as u64 <= rounds + 1);
    }

    #[test]
    fn render_metrics_reports_picks_by_signal() {
        let p = SmgPowerOfTwo::new();
        let (a, b) = (decode("http://a:1"), decode("http://b:1"));
        p.update_loads(&loads(&[("http://a:1", 1), ("http://b:1", 2)]));
        let _ = p.choose(&a, &b);
        p.update_loads(&HashMap::new());
        let _ = p.choose(&a, &b);
        let _ = p.choose(&a, &b);
        let _g = p.count_from_dispatch("http://b:1");
        let m = p.render_metrics();
        assert!(
            m.contains("sgl_router_decode_po2_smg_selections_total{signal=\"tokens\"} 1\n"),
            "{m}"
        );
        assert!(
            m.contains("sgl_router_decode_po2_smg_selections_total{signal=\"requests\"} 2\n"),
            "{m}"
        );
        assert!(
            m.contains("sgl_router_decode_po2_smg_selections_total{signal=\"single\"} 0\n"),
            "{m}"
        );
        assert!(
            m.contains("sgl_router_decode_po2_smg_requests{worker_url=\"http://b:1\"} 1\n"),
            "{m}"
        );
    }
}
