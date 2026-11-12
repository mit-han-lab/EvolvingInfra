// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Late D `fit_req` on observed decode capacity
//! (`--pd-late-decode-fit-native`, `--pd-native-decode-state`).
//!
//! Port of sgl-model-gateway
//! `policies/decode_capacity.rs` and the native part of
//! `policies/min_inflight_tokens.rs` (`select_native_fit_and_charge`,
//! `charge_native_fallback`, the fit reservations of `DecodeTokenCharge`),
//! on this router's token ledger ([`DecodeTokenBalancer`]).
//!
//! * [`DecodeCapacity`]: one decode worker's observed allocation state from
//!   `/v1/loads?include=core,disagg` (`loads[0].late_d`): allocated requests
//!   and their context tokens, the allocatable and free GPU tokens, the
//!   output reserve, the page size and the allocated bootstrap rooms. DP 1
//!   and page size 64 only (as the gateway).
//! * [`start_collector`]: every 100 ms, GET that view from every decode
//!   worker (400 ms timeout, 128 KiB body cap) and replace the snapshot. It
//!   never touches the `power-of-two-smg` load poll.
//! * [`select_native_fit_and_charge`]: among the candidates whose
//!   page-rounded input plus the output reserve fits in the allocatable
//!   tokens (less the router reservations the worker has not reported yet),
//!   the fewest allocated requests, then context tokens, then the most free
//!   GPU tokens, then URL. With a missing or stale sample, or when none
//!   fits, the `d1` order of the token ledger. The winner gets a local
//!   reservation for the room until the worker reports the room (or the
//!   charge ends).
//! * Symmetric workers (`--policy symmetric_pd` with late D choose,
//!   [`start_collector_over`]): the same poll and parse over the decode-role
//!   symmetric workers. The room is the `transfer_id`, so a symmetric engine
//!   must list the transfer ids of its committed imports in
//!   `allocated_rooms`. A worker whose `late_d` has no capacity fields gives
//!   no sample, and the bind uses the `d1` order
//!   (`native_capacity_missing_or_stale`). A symmetric engine fills them
//!   only with `SGLANG_DECODE_SELECTION_METRICS=1` (default off), so this
//!   collector logs a WARN, at most once a minute, naming each worker that
//!   answered without a usable sample and why ([`unusable_reason`]).

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use serde_json::Value;

use crate::discovery::WorkerMode;
use crate::policies::decode_tokens::DecodeTokenBalancer;
use crate::workers::{DecodeTokenCharge, Worker, WorkerRegistry};

const LOG_TARGET: &str = "native_decode_capacity";
/// Poll period of the collector.
const POLL_INTERVAL: Duration = Duration::from_millis(100);
/// Per-worker request timeout of the collector.
const POLL_TIMEOUT: Duration = Duration::from_millis(400);
/// Largest `/v1/loads` body the collector reads.
const MAX_BODY_BYTES: usize = 131_072;
/// The collector polls nobody above this many decode workers (128 covers a
/// 90-worker symmetric fleet; was 64).
const MAX_WORKERS: usize = 128;
/// Rooms a `/v1/loads` sample may list.
const MAX_ROOMS: usize = 1024;
/// The only page size the capacity arithmetic supports.
const PAGE_SIZE: u64 = 64;
/// Least time between two WARN lines of a collector that reports answers
/// without a usable sample ([`start_collector_over`] `warn_unusable`).
const UNUSABLE_WARN_INTERVAL: Duration = Duration::from_secs(60);
/// Workers one such WARN line names.
const UNUSABLE_WARN_MAX_WORKERS: usize = 8;

/// Observed decode allocation state, distinct from router token charges.
#[derive(Clone, Debug)]
pub struct DecodeCapacity {
    pub allocated_requests: u64,
    pub context_tokens: u64,
    pub allocatable_tokens: u64,
    pub free_gpu_tokens: u64,
    pub output_reserve_tokens: u64,
    pub page_size: u64,
    pub allocated_rooms: BTreeSet<u64>,
    pub received: Instant,
    pub round_trip: Duration,
    pub sample_age: Duration,
}

impl DecodeCapacity {
    /// One `/v1/loads?include=core,disagg` answer. DP 1 only: TP copies
    /// must not be summed as independent capacity. `None` for anything
    /// incomplete, truncated, duplicated or with another page size.
    pub fn parse(value: &Value, round_trip: Duration) -> Option<Self> {
        let loads = value.get("loads")?.as_array()?;
        if loads.len() != 1 {
            return None;
        }
        // Both timestamps come from the decode host; never subtract the
        // router's wall clock.
        let response_time = chrono::DateTime::parse_from_rfc3339(value.get("timestamp")?.as_str()?)
            .ok()?
            .timestamp_micros() as f64
            / 1e6;
        let sampled = loads[0].get("timestamp")?.as_f64()?;
        let age = response_time - sampled;
        if !(-0.001..=86_400.0).contains(&age) {
            return None;
        }
        let sample_age = Duration::from_secs_f64(age.max(0.0));
        let d = loads[0].get("late_d")?;
        if d.get("capacity_available")?.as_bool() != Some(true)
            || d.get("allocated_rooms_truncated")?.as_bool() != Some(false)
            || d.get("allocated_room_ids_complete")?.as_bool() != Some(true)
        {
            return None;
        }
        let ids = d.get("allocated_rooms")?.as_array()?;
        if ids.len() > MAX_ROOMS {
            return None;
        }
        let mut rooms = BTreeSet::new();
        for x in ids {
            if !rooms.insert(x.as_u64()?) {
                return None;
            }
        }
        let n = d.get("allocated_requests")?.as_u64()?;
        if n != rooms.len() as u64 {
            return None;
        }
        let page = d.get("page_size")?.as_u64()?;
        if page != PAGE_SIZE {
            return None;
        }
        Some(Self {
            allocated_requests: n,
            context_tokens: d.get("allocated_context_tokens")?.as_u64()?,
            allocatable_tokens: d.get("allocatable_tokens")?.as_u64()?,
            free_gpu_tokens: d.get("free_gpu_tokens")?.as_u64()?,
            output_reserve_tokens: d.get("output_reserve_tokens")?.as_u64()?,
            page_size: page,
            allocated_rooms: rooms,
            received: Instant::now(),
            round_trip,
            sample_age,
        })
    }

    pub fn fresh(&self, max_age: Duration) -> bool {
        self.age() <= max_age
    }

    /// Time since the engine sampled it: the engine-side sample age, the
    /// HTTP round trip and the time since the router received it.
    pub fn age(&self) -> Duration {
        self.received
            .elapsed()
            .saturating_add(self.round_trip)
            .saturating_add(self.sample_age)
    }

    /// Tokens a request of `input` prompt tokens needs: the input rounded
    /// up to whole pages, plus the output reserve.
    pub fn required(&self, input: u64) -> Option<u64> {
        input
            .checked_add(self.page_size - 1)?
            .checked_div(self.page_size)?
            .checked_mul(self.page_size)?
            .checked_add(self.output_reserve_tokens)
    }
}

/// Why an answered `/v1/loads` body gives no [`DecodeCapacity::parse`]
/// sample, for the WARN of the symmetric collector. Coarse: one sufficient
/// cause, the most actionable first. A symmetric engine without
/// `SGLANG_DECODE_SELECTION_METRICS=1` answers a `late_d` block with only
/// `deferred_reqs`, `deferred_tokens` and `bound_total_tokens`.
pub fn unusable_reason(value: &Value) -> String {
    let Some(loads) = value.get("loads").and_then(Value::as_array) else {
        return "no loads array".to_string();
    };
    if loads.len() != 1 {
        return format!("{} loads entries (DP 1 only)", loads.len());
    }
    let Some(d) = loads[0].get("late_d") else {
        return "no late_d block (late D off on the engine)".to_string();
    };
    match d.get("capacity_available").and_then(Value::as_bool) {
        None => {
            return "late_d has no capacity fields (the engine needs \
                    SGLANG_DECODE_SELECTION_METRICS=1)"
                .to_string();
        }
        Some(false) => {
            let why: String = d
                .get("capacity_unsupported_reason")
                .and_then(Value::as_str)
                .unwrap_or("no reason given")
                .chars()
                .take(64)
                .collect();
            return format!("capacity_available=false ({why})");
        }
        Some(true) => {}
    }
    if let Some(page) = d
        .get("page_size")
        .and_then(Value::as_u64)
        .filter(|p| *p != PAGE_SIZE)
    {
        return format!("page_size {page} (only {PAGE_SIZE})");
    }
    if d.get("allocated_rooms_truncated").and_then(Value::as_bool) != Some(false)
        || d.get("allocated_room_ids_complete")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return "allocated rooms truncated or incomplete".to_string();
    }
    "timestamps, room ids or counts invalid".to_string()
}

/// The latest observed state of every decode worker, by URL.
#[derive(Debug, Default)]
pub struct DecodeCapacityStore {
    states: RwLock<Arc<HashMap<String, DecodeCapacity>>>,
}

impl DecodeCapacityStore {
    /// Replace the whole snapshot (one collector round).
    pub fn replace(&self, states: HashMap<String, DecodeCapacity>) {
        *self.states.write() = Arc::new(states);
    }

    pub fn snapshot(&self) -> Arc<HashMap<String, DecodeCapacity>> {
        self.states.read().clone()
    }

    pub fn len(&self) -> usize {
        self.states.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.read().is_empty()
    }
}

/// The decode workers one collector round polls.
pub type DecodeWorkerList = Arc<dyn Fn(&WorkerRegistry) -> Vec<Arc<Worker>> + Send + Sync>;

/// Start the 100 ms collector over every registered decode worker when a
/// Tokio runtime is available. It stops when the registry is dropped.
pub fn start_collector(registry: &Arc<WorkerRegistry>, store: Arc<DecodeCapacityStore>) {
    start_collector_over(
        registry,
        store,
        Arc::new(|registry: &WorkerRegistry| {
            registry
                .all()
                .into_iter()
                .filter(|w| w.mode() == WorkerMode::Decode)
                .collect()
        }),
        "every decode worker",
        false,
    );
}

/// [`start_collector`] over the workers `select` returns each round (for
/// example the decode-role symmetric workers of `--policy symmetric_pd`); `what`
/// names them in the start log line. Same poll, parse and limits. With
/// `warn_unusable`, a round in which a worker answered without a usable
/// sample logs a WARN naming such workers and the reason
/// ([`unusable_reason`]), at most once a minute; without it the collector
/// computes no reason and logs no WARN.
pub fn start_collector_over(
    registry: &Arc<WorkerRegistry>,
    store: Arc<DecodeCapacityStore>,
    select: DecodeWorkerList,
    what: &'static str,
    warn_unusable: bool,
) {
    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        tracing::warn!(target: LOG_TARGET, "no tokio runtime; native decode capacity poll not started");
        return;
    };
    let weak = Arc::downgrade(registry);
    tracing::info!(
        target: LOG_TARGET,
        interval_ms = POLL_INTERVAL.as_millis() as u64,
        workers = what,
        "native decode capacity poll started (GET /v1/loads?include=core,disagg on every decode worker)"
    );
    handle.spawn(async move {
        let Ok(client) = reqwest::Client::builder().timeout(POLL_TIMEOUT).build() else {
            tracing::error!(target: LOG_TARGET, "native decode capacity poll: HTTP client failed to build");
            return;
        };
        let mut rounds = 0u64;
        let mut last_unusable_warn: Option<Instant> = None;
        let mut timer = tokio::time::interval(POLL_INTERVAL);
        timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            timer.tick().await;
            let Some(registry) = weak.upgrade() else {
                return;
            };
            let workers: Vec<Arc<Worker>> = select(&*registry);
            drop(registry);
            if workers.len() > MAX_WORKERS {
                store.replace(HashMap::new());
                continue;
            }
            let results = futures::future::join_all(workers.iter().map(|w| {
                let client = &client;
                async move {
                    let started = Instant::now();
                    let url = w.url.clone();
                    let state = fetch(client, &url, started, warn_unusable).await;
                    (url, state)
                }
            }))
            .await;
            let mut states: HashMap<String, DecodeCapacity> = HashMap::new();
            let mut unusable: Vec<(String, String)> = Vec::new();
            for (url, state) in results {
                match state {
                    Ok(s) => {
                        states.insert(url, s);
                    }
                    Err(Some(why)) => unusable.push((url, why)),
                    Err(None) => {}
                }
            }
            rounds += 1;
            if rounds == 1 || rounds.is_multiple_of(100) {
                tracing::info!(
                    target: LOG_TARGET,
                    rounds,
                    workers = workers.len(),
                    available = states.len(),
                    "native allocation state poll"
                );
            }
            if !unusable.is_empty()
                && last_unusable_warn.is_none_or(|t| t.elapsed() >= UNUSABLE_WARN_INTERVAL)
            {
                last_unusable_warn = Some(Instant::now());
                unusable.sort();
                let named: Vec<String> = unusable
                    .iter()
                    .take(UNUSABLE_WARN_MAX_WORKERS)
                    .map(|(url, why)| format!("{url}: {why}"))
                    .collect();
                tracing::warn!(
                    target: LOG_TARGET,
                    rounds,
                    workers = workers.len(),
                    available = states.len(),
                    unusable = unusable.len(),
                    first = ?named,
                    "native allocation state: workers answered /v1/loads without a usable late_d \
                     capacity sample; fit_req binds by the d1 order while a candidate has none \
                     (native_capacity_missing_or_stale)"
                );
            }
            store.replace(states);
        }
    });
}

/// One worker's sample. `Err(Some(reason))`: with `diagnose`, the worker
/// answered but gave no usable sample ([`unusable_reason`]). `Err(None)`: no
/// usable answer (transport, HTTP status, body size, JSON), or an unusable
/// one without `diagnose`.
async fn fetch(
    client: &reqwest::Client,
    url: &str,
    started: Instant,
    diagnose: bool,
) -> Result<DecodeCapacity, Option<String>> {
    let Some(value) = fetch_loads(client, url).await else {
        return Err(None);
    };
    DecodeCapacity::parse(&value, started.elapsed())
        .ok_or_else(|| diagnose.then(|| unusable_reason(&value)))
}

/// GET one worker's `/v1/loads?include=core,disagg` as JSON (body capped at
/// [`MAX_BODY_BYTES`]).
async fn fetch_loads(client: &reqwest::Client, url: &str) -> Option<Value> {
    let mut response = client
        .get(format!(
            "{}/v1/loads?include=core,disagg",
            url.trim_end_matches('/')
        ))
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?;
    let mut bytes = Vec::new();
    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                if bytes.len() + chunk.len() > MAX_BODY_BYTES {
                    return None;
                }
                bytes.extend_from_slice(&chunk);
            }
            Ok(None) => break,
            Err(_) => return None,
        }
    }
    serde_json::from_slice::<Value>(&bytes).ok()
}

// ---------------------------------------------------------------------------
// Router reservations and the charge that carries one.
// ---------------------------------------------------------------------------

/// A room the router chose (or saw admitted) on a decode worker, until that
/// worker reports the room or the request's charge ends.
#[derive(Debug, Clone)]
pub struct FitReservation {
    pub worker: String,
    pub capacity_tokens: u64,
    pub context_tokens: u64,
    /// The worker reported the room in a fresh sample: its own counts
    /// include the request, so the reservation no longer counts.
    pub acknowledged: bool,
    /// `capacity_tokens` came from an observed sample (not the charge).
    pub capacity_known: bool,
}

/// The router reservations, by bootstrap room.
pub type FitReservations = Arc<Mutex<HashMap<u64, FitReservation>>>;

/// Removes its room's reservation when dropped.
#[derive(Debug)]
pub struct FitGuard {
    room: u64,
    map: FitReservations,
}

impl FitGuard {
    /// The reservation moved to `worker_url`: the new worker has not
    /// reported it.
    fn retarget(&self, worker_url: &str) {
        if let Some(r) = self.map.lock().get_mut(&self.room) {
            r.worker = worker_url.to_string();
            r.acknowledged = false;
        }
    }
}

impl Drop for FitGuard {
    fn drop(&mut self) {
        self.map.lock().remove(&self.room);
    }
}

/// A late-D request's charge on its decode worker: the token-ledger charge
/// and, under `fit_native`, the room's reservation. Dropping it releases
/// both.
#[must_use = "the charge is released when this guard is dropped"]
#[derive(Debug)]
pub struct LateCharge {
    worker_url: String,
    charge: DecodeTokenCharge,
    fit: Option<FitGuard>,
}

impl LateCharge {
    /// A ledger charge without a reservation.
    pub fn plain(worker: &Worker, charge: DecodeTokenCharge) -> Self {
        Self {
            worker_url: worker.url.clone(),
            charge,
            fit: None,
        }
    }

    pub fn worker_url(&self) -> &str {
        &self.worker_url
    }

    pub fn tokens(&self) -> u64 {
        self.charge.tokens()
    }

    /// Move the charge (and the reservation) to `to`: the prefill admitted
    /// another candidate than the one charged. A move to the same worker
    /// changes nothing.
    pub fn move_to(self, ledger: &DecodeTokenBalancer, to: &Worker) -> Self {
        if self.worker_url == to.url {
            return self;
        }
        let LateCharge { charge, fit, .. } = self;
        let charge = ledger.move_charge(charge, to);
        if let Some(f) = &fit {
            f.retarget(&to.url);
        }
        Self {
            worker_url: to.url.clone(),
            charge,
            fit,
        }
    }
}

/// The outcome of the observed-capacity fit: the winner (index into the
/// candidates), its charge, whether it fit, and the decision-log signal.
pub type NativeFit = (usize, LateCharge, bool, String);

/// sgl-model-gateway `select_native_fit_and_charge`: pick among
/// `candidates[set[..]]` on the observed capacity `states` and charge the
/// winner `charge_tokens`, with a reservation for `room` of the winner's
/// observed requirement for `input_tokens`. Every sample must be fresh
/// (`max_age`) and every unacknowledged reservation's requirement known,
/// else the `d1` order. `None` when `set` is empty, a requirement overflows
/// or `room` already has a live reservation.
#[allow(clippy::too_many_arguments)]
pub fn select_native_fit_and_charge(
    ledger: &DecodeTokenBalancer,
    reservations: &FitReservations,
    candidates: &[Arc<Worker>],
    set: &[usize],
    charge_tokens: u64,
    room: u64,
    input_tokens: u64,
    states: &HashMap<String, DecodeCapacity>,
    max_age: Duration,
) -> Option<NativeFit> {
    if set.is_empty() {
        return None;
    }
    let _pick = ledger.lock();
    let mut pending = reservations.lock();
    // A confirmed allocation never becomes an unseen incoming reservation
    // again.
    for (key, r) in pending.iter_mut() {
        if states
            .get(&r.worker)
            .is_some_and(|s| s.fresh(max_age) && s.allocated_rooms.contains(key))
        {
            r.acknowledged = true;
        }
    }
    let url = |i: usize| candidates[i].url.as_str();
    let complete = input_tokens > 0
        && pending.values().all(|p| p.acknowledged || p.capacity_known)
        && set
            .iter()
            .all(|&i| states.get(url(i)).is_some_and(|s| s.fresh(max_age)));
    // (idx, count, context, free, budget, required, age_ms)
    let mut rows: Vec<(usize, u64, u64, u64, u64, u64, u128)> = Vec::new();
    if complete {
        for &i in set {
            let s = &states[url(i)];
            let required = s.required(input_tokens)?;
            let unseen: Vec<&FitReservation> = pending
                .values()
                .filter(|r| r.worker == url(i) && !r.acknowledged)
                .collect();
            let reserved = unseen
                .iter()
                .fold(0u64, |a, r| a.saturating_add(r.capacity_tokens));
            let context = unseen
                .iter()
                .fold(s.context_tokens, |a, r| a.saturating_add(r.context_tokens));
            let count = s.allocated_requests.saturating_add(unseen.len() as u64);
            let free = s.free_gpu_tokens.saturating_sub(reserved);
            let budget = s.allocatable_tokens.saturating_sub(reserved);
            rows.push((
                i,
                count,
                context,
                free,
                budget,
                required,
                s.age().as_millis(),
            ));
        }
    }
    let chosen = rows.iter().filter(|r| r.5 <= r.4).min_by(|a, b| {
        a.1.cmp(&b.1)
            .then(a.2.cmp(&b.2))
            .then(b.3.cmp(&a.3))
            .then_with(|| url(a.0).cmp(url(b.0)))
    });
    let fit = chosen.is_some();
    let winner = match chosen {
        Some(r) => r.0,
        None => ledger.d1_index_locked(candidates, set, charge_tokens)?,
    };
    let signal = if complete {
        format!("native_fit={fit}; idx/count/context/free/budget/required/age_ms={rows:?}")
    } else {
        "native_fit=false; native_capacity_missing_or_stale; fallback=d1".to_string()
    };
    let observed_required = states.get(url(winner)).and_then(|s| {
        s.required(if input_tokens > 0 {
            input_tokens
        } else {
            charge_tokens
        })
    });
    let capacity_tokens = observed_required.unwrap_or(charge_tokens);
    // Refuse a duplicate room rather than replace a live reservation.
    if pending.contains_key(&room) {
        return None;
    }
    pending.insert(
        room,
        FitReservation {
            worker: url(winner).to_string(),
            capacity_tokens,
            context_tokens: input_tokens,
            acknowledged: false,
            capacity_known: observed_required.is_some(),
        },
    );
    drop(pending);
    let worker = &candidates[winner];
    let charge = LateCharge {
        worker_url: worker.url.clone(),
        charge: worker.charge_decode_tokens(charge_tokens),
        fit: Some(FitGuard {
            room,
            map: Arc::clone(reservations),
        }),
    };
    Some((winner, charge, fit, signal))
}

/// sgl-model-gateway `charge_native_fallback`: a default or admitted winner
/// seen without a decision (the prefill timed out, or the link was down)
/// still needs a reservation for its room before the next sample shows it.
/// `required` is the winner's observed requirement, when a sample exists.
pub fn charge_native_fallback(
    ledger: &DecodeTokenBalancer,
    reservations: &FitReservations,
    worker: &Worker,
    tokens: u64,
    room: u64,
    required: Option<u64>,
) -> LateCharge {
    let _pick = ledger.lock();
    reservations.lock().insert(
        room,
        FitReservation {
            worker: worker.url.clone(),
            capacity_tokens: required.unwrap_or(tokens),
            context_tokens: tokens,
            acknowledged: false,
            capacity_known: required.is_some(),
        },
    );
    LateCharge {
        worker_url: worker.url.clone(),
        charge: worker.charge_decode_tokens(tokens),
        fit: Some(FitGuard {
            room,
            map: Arc::clone(reservations),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PdDecodeTokensConfig;
    use crate::discovery::{ModelId, WorkerId, WorkerSpec};
    use crate::policies::decode_tokens::TokenRule;
    use serde_json::json;

    fn decode(url: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(url.into()),
            url: url.into(),
            mode: WorkerMode::Decode,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
        }))
    }

    /// Deliberately not in URL order: selection must not depend on slice
    /// order.
    fn pool(n: usize) -> Vec<Arc<Worker>> {
        (0..n)
            .rev()
            .map(|i| decode(&format!("http://d{i}")))
            .collect()
    }

    fn ledger() -> DecodeTokenBalancer {
        DecodeTokenBalancer::new(
            TokenRule::MinInflightTokens,
            PdDecodeTokensConfig::default(),
        )
    }

    fn reservations() -> FitReservations {
        Arc::new(Mutex::new(HashMap::new()))
    }

    fn capacity(n: u64, context: u64, budget: u64, free: u64) -> DecodeCapacity {
        DecodeCapacity {
            allocated_requests: n,
            context_tokens: context,
            allocatable_tokens: budget,
            free_gpu_tokens: free,
            output_reserve_tokens: 512,
            page_size: 64,
            allocated_rooms: (0..n).collect(),
            received: Instant::now(),
            round_trip: Duration::ZERO,
            sample_age: Duration::ZERO,
        }
    }

    fn sample() -> Value {
        let at = chrono::DateTime::parse_from_rfc3339("2026-10-14T00:00:10+00:00").unwrap();
        json!({"timestamp": at.to_rfc3339(), "loads": [{"timestamp": at.timestamp() as f64,
            "late_d": {"capacity_available": true, "allocated_rooms_truncated": false,
                "allocated_room_ids_complete": true, "allocated_rooms": [11, 12],
                "allocated_requests": 2, "allocated_context_tokens": 1000,
                "allocatable_tokens": 9000, "free_gpu_tokens": 10000,
                "output_reserve_tokens": 512, "page_size": 64}}]})
    }

    fn url_of(ws: &[Arc<Worker>], i: usize) -> &str {
        ws[i].url.as_str()
    }

    #[test]
    fn actual_budget_not_charged_tokens_and_fixed_output_reserve() {
        let s = DecodeCapacity::parse(&sample(), Duration::ZERO).unwrap();
        assert_eq!(s.required(65), Some(640));
        assert_eq!(s.allocated_requests, 2);
        assert!(s.fresh(Duration::from_secs(1)));
    }

    #[test]
    fn unknown_truncated_duplicate_or_multirank_capacity_is_unavailable() {
        let mut s = sample();
        s["loads"][0]["late_d"]["allocated_rooms_truncated"] = true.into();
        assert!(DecodeCapacity::parse(&s, Duration::ZERO).is_none());
        s = sample();
        s["loads"][0]["late_d"]["allocated_rooms"] = json!([11, 11]);
        assert!(DecodeCapacity::parse(&s, Duration::ZERO).is_none());
        s = sample();
        let row = s["loads"][0].clone();
        s["loads"].as_array_mut().unwrap().push(row);
        assert!(DecodeCapacity::parse(&s, Duration::ZERO).is_none());
        s = sample();
        s["loads"][0]["late_d"]["page_size"] = 32.into();
        assert!(DecodeCapacity::parse(&s, Duration::ZERO).is_none());
        s = sample();
        s["loads"][0]["late_d"]["capacity_available"] = false.into();
        assert!(DecodeCapacity::parse(&s, Duration::ZERO).is_none());
        assert!(!DecodeCapacity::parse(&sample(), Duration::from_secs(10))
            .unwrap()
            .fresh(Duration::from_secs(6)));
    }

    #[test]
    fn fresh_http_does_not_refresh_stale_engine_shared_memory() {
        let mut value = sample();
        let old = value["loads"][0]["timestamp"].as_f64().unwrap() - 2.0;
        value["loads"][0]["timestamp"] = json!(old);
        let state = DecodeCapacity::parse(&value, Duration::ZERO).unwrap();
        assert!(!state.fresh(Duration::from_secs(1)));
        assert!(state.sample_age >= Duration::from_secs(2));
        value["loads"][0]["timestamp"] = Value::Null;
        assert!(DecodeCapacity::parse(&value, Duration::ZERO).is_none());
    }

    #[test]
    fn native_fit_uses_request_count_then_context_then_free_and_real_budget() {
        let l = ledger();
        let res = reservations();
        let ws = pool(3);
        let set = vec![0, 1, 2];
        let states = HashMap::from([
            ("http://d0".to_string(), capacity(2, 100, 9000, 10000)),
            // Does not fit 128 + 512 despite few requests.
            ("http://d1".to_string(), capacity(1, 200, 600, 10000)),
            ("http://d2".to_string(), capacity(2, 100, 9000, 11000)),
        ]);
        let (i, g, fit, _) = select_native_fit_and_charge(
            &l,
            &res,
            &ws,
            &set,
            66,
            100,
            65,
            &states,
            Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(url_of(&ws, i), "http://d2");
        assert!(fit);
        assert_eq!(g.tokens(), 66);
        assert_eq!(res.lock().get(&100).unwrap().capacity_tokens, 640);
        drop(g);
        assert!(res.lock().is_empty());
        let mut states = states;
        states.get_mut("http://d0").unwrap().context_tokens = 99;
        let (i, _g, fit, _) = select_native_fit_and_charge(
            &l,
            &res,
            &ws,
            &set,
            66,
            101,
            65,
            &states,
            Duration::from_secs(1),
        )
        .unwrap();
        assert!(fit);
        assert_eq!(url_of(&ws, i), "http://d0");
    }

    #[test]
    fn native_reservations_prevent_repeated_pick_and_do_not_double_charge_acknowledged_rooms() {
        let l = ledger();
        let res = reservations();
        let ws = pool(2);
        let set = vec![0, 1];
        let mut states = HashMap::from([
            ("http://d0".to_string(), capacity(0, 0, 9000, 10000)),
            ("http://d1".to_string(), capacity(0, 0, 9000, 10000)),
        ]);
        let (a, ga, _, _) = select_native_fit_and_charge(
            &l,
            &res,
            &ws,
            &set,
            66,
            100,
            65,
            &states,
            Duration::from_secs(1),
        )
        .unwrap();
        let (b, gb, _, _) = select_native_fit_and_charge(
            &l,
            &res,
            &ws,
            &set,
            66,
            101,
            65,
            &states,
            Duration::from_secs(1),
        )
        .unwrap();
        assert_ne!(a, b);
        assert_eq!(url_of(&ws, a), "http://d0");
        // A duplicate room is refused while its reservation lives.
        assert!(select_native_fit_and_charge(
            &l,
            &res,
            &ws,
            &set,
            66,
            100,
            65,
            &states,
            Duration::from_secs(1),
        )
        .is_none());
        let state = states.get_mut("http://d0").unwrap();
        state.allocated_rooms.insert(100);
        state.allocated_requests = 1;
        state.context_tokens = 65;
        state.allocatable_tokens -= 640;
        state.free_gpu_tokens -= 640;
        let (_, gc, _, signal) = select_native_fit_and_charge(
            &l,
            &res,
            &ws,
            &set,
            66,
            102,
            65,
            &states,
            Duration::from_secs(1),
        )
        .unwrap();
        assert!(res.lock().get(&100).unwrap().acknowledged);
        assert!(signal.contains("native_fit=true"), "{signal}");
        let d1 = Arc::clone(&ws[b]);
        let ga = ga.move_to(&l, &d1);
        assert_eq!(ga.worker_url(), "http://d1");
        assert!(!res.lock().get(&100).unwrap().acknowledged);
        assert_eq!(res.lock().get(&100).unwrap().worker, "http://d1");
        drop((ga, gb, gc));
        assert!(res.lock().is_empty());
        for w in &ws {
            assert_eq!((w.decode_charged_tokens(), w.decode_charged_reqs()), (0, 0));
        }
    }

    #[test]
    fn missing_stale_or_no_fit_uses_original_min_tokens_order() {
        let l = ledger();
        let res = reservations();
        let ws = pool(2);
        let set = vec![0, 1];
        let d0 = ws.iter().find(|w| w.url == "http://d0").unwrap();
        let _busy = l.charge(d0, 1000);
        let states = HashMap::from([
            ("http://d0".to_string(), capacity(0, 0, 1, 1)),
            ("http://d1".to_string(), capacity(10, 100, 1, 1)),
        ]);
        for (key, map) in [(100, states), (101, HashMap::new())] {
            let (i, g, fit, _) = select_native_fit_and_charge(
                &l,
                &res,
                &ws,
                &set,
                66,
                key,
                65,
                &map,
                Duration::from_secs(1),
            )
            .unwrap();
            assert!(!fit);
            assert_eq!(url_of(&ws, i), "http://d1");
            drop(g);
        }
        let mut state = capacity(0, 0, 10000, 10000);
        state.round_trip = Duration::from_secs(2);
        let stale = HashMap::from([
            ("http://d0".to_string(), state.clone()),
            ("http://d1".to_string(), state),
        ]);
        let (i, _g, fit, why) = select_native_fit_and_charge(
            &l,
            &res,
            &ws,
            &set,
            66,
            102,
            65,
            &stale,
            Duration::from_secs(1),
        )
        .unwrap();
        assert!(!fit);
        assert_eq!(url_of(&ws, i), "http://d1");
        assert!(why.contains("stale"), "{why}");
    }

    #[test]
    fn native_fallback_reserves_the_observed_requirement() {
        let l = ledger();
        let res = reservations();
        let w = decode("http://d0");
        let c = charge_native_fallback(&l, &res, &w, 1000, 7, Some(1600));
        let r = res.lock().get(&7).cloned().unwrap();
        assert_eq!(
            (r.capacity_tokens, r.capacity_known, r.acknowledged),
            (1600, true, false)
        );
        assert_eq!(
            (w.decode_charged_tokens(), w.decode_charged_reqs()),
            (1000, 1)
        );
        // Same worker: nothing moves.
        let c = c.move_to(&l, &w);
        assert_eq!(c.worker_url(), "http://d0");
        drop(c);
        assert!(res.lock().is_empty());
        assert_eq!((w.decode_charged_tokens(), w.decode_charged_reqs()), (0, 0));
        let c = charge_native_fallback(&l, &res, &w, 1000, 8, None);
        let r = res.lock().get(&8).cloned().unwrap();
        assert_eq!((r.capacity_tokens, r.capacity_known), (1000, false));
        drop(c);
    }

    /// The symmetric collector's WARN reason. The `late_d` block of an
    /// engine without `SGLANG_DECODE_SELECTION_METRICS` (only the deferred
    /// and bound fields, `LoadSnapshot.to_dict`) gives no sample and names
    /// the flag; the other causes are named too.
    #[test]
    fn unusable_reason_names_the_missing_engine_flag_and_other_causes() {
        let mut s = sample();
        assert!(DecodeCapacity::parse(&s, Duration::ZERO).is_some());
        s["loads"][0]["late_d"] =
            json!({"deferred_reqs": 0, "deferred_tokens": 0, "bound_total_tokens": 4096});
        assert!(DecodeCapacity::parse(&s, Duration::ZERO).is_none());
        let why = unusable_reason(&s);
        assert!(why.contains("SGLANG_DECODE_SELECTION_METRICS=1"), "{why}");
        s["loads"][0].as_object_mut().unwrap().remove("late_d");
        assert!(unusable_reason(&s).starts_with("no late_d block"));

        let mut s = sample();
        s["loads"][0]["late_d"]["capacity_available"] = false.into();
        s["loads"][0]["late_d"]["capacity_unsupported_reason"] =
            "requires_dense_fixed_reserve".into();
        assert_eq!(
            unusable_reason(&s),
            "capacity_available=false (requires_dense_fixed_reserve)"
        );
        let mut s = sample();
        s["loads"][0]["late_d"]["page_size"] = 256.into();
        assert_eq!(unusable_reason(&s), "page_size 256 (only 64)");
        let mut s = sample();
        let row = s["loads"][0].clone();
        s["loads"].as_array_mut().unwrap().push(row);
        assert_eq!(unusable_reason(&s), "2 loads entries (DP 1 only)");
        let mut s = sample();
        s["loads"][0]["late_d"]["allocated_room_ids_complete"] = false.into();
        assert_eq!(
            unusable_reason(&s),
            "allocated rooms truncated or incomplete"
        );
        let mut s = sample();
        s["loads"][0]["timestamp"] = Value::Null;
        assert!(DecodeCapacity::parse(&s, Duration::ZERO).is_none());
        assert_eq!(
            unusable_reason(&s),
            "timestamps, room ids or counts invalid"
        );
        assert_eq!(unusable_reason(&json!({})), "no loads array");
    }
}
