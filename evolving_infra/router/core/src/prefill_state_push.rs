// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! cost-function the forecast wait: the subscriber of the P engines' state push
//! (`--forecast-push-port PORT`).
//!
//! The P engine (DecoupledEngine prefill state push, wire contract v1 in
//! `prefill_state_push.py`) publishes its `prefill_routing`
//! state on a ZMQ PUB socket of its attention TP0 rank, `tcp://*:PORT+rank`
//! (`SGLANG_PREFILL_STATE_PUSH_PORT`): at a pass launch while the previous
//! pass still runs on the GPU, at the end of each loop iteration that
//! launched a pass or processed a result, and as a heartbeat after
//! `SGLANG_PREFILL_STATE_PUSH_HEARTBEAT_S` (default 1 s) without a message.
//! A message has three frames: `b"prefill_state"`, the sequence number (8
//! bytes, big-endian) and one msgpack map ([`PushPayload`]) whose `snapshot`
//! has the fields of `/v1/loads?include=prefill_routing`.
//!
//! # What the router does
//!
//! - One SUB socket per P worker of the routable pool, connected to
//!   `tcp://<host of the worker URL>:PORT` (routing rank 0, so one P per host
//!   and port). P workers whose URLs share a host get no subscriber (their
//!   messages would be indistinguishable: a message does not name its host)
//!   and keep the poll. A background task per P reads every message when it
//!   arrives and decodes it ([`PushHub::on_message`]); the newest state with
//!   a snapshot is kept per P ([`PushState`]).
//! - Clock: for each message read, `x = router receive time -
//!   sent_mono_s`; the offset is the minimum `x` of the messages read in the
//!   last [`OFFSET_WINDOW_S`] of the current epoch; the state time on the
//!   router clock is `t0 = snapshot.sampled_monotonic_s + offset`, and the
//!   state's age at a decision `d` is `d - t0`. The age includes the time in
//!   queues; its error is the smallest transit time in the window plus the
//!   clock drift over it. No synchronized wall clocks are needed.
//! - Sequence: within one epoch, a jump of the sequence by more
//!   than one counts the lost messages (`gaps`, `lost`); a repeated or older
//!   sequence is dropped (`repeats`); a new epoch (engine restart) starts the
//!   sequence, the offset window and the state again.
//! - The pushed snapshot parses into the same [`ForecastView`] as a polled one. Its
//!   timeline samples feed the online fits ([`ForecastOnline::observe_push`];
//!   samples are deduplicated with the polled ones by their sequence), and
//!   the `forecast_label` of a routed request whose completion it holds is written.
//! - The message's `inflight` list is compared with the snapshot's pending
//!   passes (`inflight_mismatch`) and its `events` are counted. The forecast
//!   reads the in-flight passes from the snapshot: a launch message builds
//!   the snapshot with the launched pass in flight, with its features, which
//!   the `inflight` list does not carry. The GPU fields are not used.
//! - A message without a snapshot (`over_max_bytes`, `build_error:*`) keeps
//!   the previous state, marked as superseded by its sequence.
//!
//! The decision ([`crate::prefill_work`]) uses a P's pushed state instead of
//! its polled snapshot when the state is at most
//! `--max-state-age-ms` old, has the epoch of the P's
//! polled snapshot (the push and the poll read one tracker, so another epoch
//! means a wrong endpoint or a restart between them) and is usable;
//! otherwise it uses the polled snapshot and names the reason.
//!
//! Counters per P ([`PushCounts`]) go to an `forecast_push_stats` event in the cost
//! trace every [`STATS_INTERVAL`] (the interval of the engine's own push
//! summary log).

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;
use zeromq::{Socket, SocketRecv, SubSocket};

use crate::prefill_forecast::{ForecastOnline, ForecastView, ShadowSink};

/// Frame 1 of every message; the SUB subscribes to it.
pub const TOPIC: &[u8] = b"prefill_state";
/// `schema` and `schema_version` of a v1 payload.
pub const WIRE_SCHEMA: &str = "sglang.prefill_state_push";
pub const WIRE_SCHEMA_VERSION: u64 = 1;
/// The offset is the minimum over the messages read in this many
/// seconds (the engine's GPU clock uses the same window; it bounds the clock
/// drift that enters the minimum).
pub const OFFSET_WINDOW_S: f64 = 30.0;
/// `forecast_push_stats` interval (the engine logs its push summary every 60 s).
pub const STATS_INTERVAL: Duration = Duration::from_secs(60);
/// `forecast_schema` of the `forecast_push_stats` events.
pub const STATS_SCHEMA: &str = "state-push-stats.v1";
/// Wait between connection attempts after a failed one (doubling up to the
/// maximum). zeromq itself retries a refused connection for up to 30 s per
/// attempt and reconnects a dropped one.
const RECONNECT_MIN: Duration = Duration::from_millis(100);
const RECONNECT_MAX: Duration = Duration::from_secs(2);
/// Consecutive receive errors after which the socket is opened again.
const RECV_ERROR_CEILING: u32 = 64;
/// Pause after a receive error (no busy loop on a broken socket).
const RECV_ERROR_PAUSE: Duration = Duration::from_millis(10);

// ---------------------------------------------------------------------------
// Wire format.
// ---------------------------------------------------------------------------

/// One pass at one observation point (`PassEvent`). Times are on
/// the engine's monotonic clock.
#[derive(Debug, Default, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct PassEvent {
    /// `launch`, `result` or `inflight`.
    pub kind: String,
    pub forward_iter: Option<i64>,
    pub launch_mono_s: Option<f64>,
    pub observed_mono_s: Option<f64>,
    pub gpu_started: Option<bool>,
    pub gpu_ended: Option<bool>,
    pub gpu_ms: Option<f64>,
    pub gpu_gap_before_ms: Option<f64>,
    pub gpu_start_mono_s: Option<f64>,
    pub gpu_end_mono_s: Option<f64>,
}

/// The payload map (`PrefillStatePush`). Unknown fields are
/// ignored; absent ones take their default.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct PushPayload {
    pub schema: String,
    pub schema_version: u64,
    pub epoch: String,
    pub seq: u64,
    pub rank: u64,
    /// `launch`, `result` or `heartbeat`.
    pub trigger: String,
    pub sent_mono_s: Option<f64>,
    pub sent_unix_s: Option<f64>,
    pub mono_to_unix_s: Option<f64>,
    pub clock_domain_id: Option<String>,
    pub events: Vec<PassEvent>,
    pub inflight: Vec<PassEvent>,
    pub gpu_timing: Option<String>,
    pub gpu_clock_samples: u64,
    pub dropped_total: u64,
    pub errors_total: u64,
    pub oversize_total: u64,
    pub build_us: Option<f64>,
    pub convert_us: Option<f64>,
    pub prev_encode_us: Option<f64>,
    pub prev_send_us: Option<f64>,
    pub prev_bytes: Option<u64>,
    pub prev_flush_us: Option<f64>,
    /// `push_build` or `v1_loads_capture`; None without a snapshot.
    pub snapshot_source: Option<String>,
    /// The `prefill_routing` object, or None (`snapshot_omitted` says why).
    pub snapshot: Option<Value>,
    pub snapshot_omitted: Option<String>,
}

/// A message that passed the frame and schema checks.
#[derive(Debug, Clone)]
pub struct PushMessage {
    /// The sequence of frame 2 (equal to `payload.seq`).
    pub seq: u64,
    /// Payload bytes.
    pub bytes: usize,
    pub payload: PushPayload,
}

/// Why a message was dropped before it reached the state (the reason is a
/// counter key; the detail is logged once per P and reason).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushDrop {
    pub reason: &'static str,
    pub detail: String,
}

fn dropped(reason: &'static str, detail: impl Into<String>) -> PushDrop {
    PushDrop {
        reason,
        detail: detail.into(),
    }
}

/// Decode the three frames of one message: the topic, the
/// big-endian sequence and a v1 payload of routing rank 0 whose `seq` is the
/// frame's and whose send time is finite.
pub fn decode_frames(frames: &[&[u8]]) -> Result<PushMessage, PushDrop> {
    let [topic, seq, payload] = frames else {
        return Err(dropped("frame_count", format!("{} frames", frames.len())));
    };
    if *topic != TOPIC {
        return Err(dropped(
            "topic",
            String::from_utf8_lossy(&topic[..topic.len().min(64)]).into_owned(),
        ));
    }
    let seq = u64::from_be_bytes(
        <[u8; 8]>::try_from(*seq)
            .map_err(|_| dropped("seq_frame", format!("{} bytes", seq.len())))?,
    );
    let body: PushPayload =
        rmp_serde::from_slice(payload).map_err(|e| dropped("decode", e.to_string()))?;
    if body.schema != WIRE_SCHEMA || body.schema_version != WIRE_SCHEMA_VERSION {
        return Err(dropped(
            "schema",
            format!("{} v{}", body.schema, body.schema_version),
        ));
    }
    if body.seq != seq {
        return Err(dropped(
            "seq_mismatch",
            format!("frame {seq}, payload {}", body.seq),
        ));
    }
    if body.rank != 0 {
        return Err(dropped("rank", format!("rank {}", body.rank)));
    }
    if body.epoch.is_empty() {
        return Err(dropped("no_epoch", ""));
    }
    if !body.sent_mono_s.is_some_and(f64::is_finite) {
        return Err(dropped("no_send_time", ""));
    }
    Ok(PushMessage {
        seq,
        bytes: payload.len(),
        payload: body,
    })
}

/// The SUB endpoint of a P worker: `tcp://<host of its URL>:port`.
pub fn endpoint_of(worker_url: &str, port: u16) -> Result<String, &'static str> {
    let url = url::Url::parse(worker_url).map_err(|_| "bad_url")?;
    let host = url.host_str().filter(|h| !h.is_empty()).ok_or("no_host")?;
    Ok(format!("tcp://{host}:{port}"))
}

// ---------------------------------------------------------------------------
// State per P.
// ---------------------------------------------------------------------------

/// The newest pushed state of one P: the decision's input when it is fresh,
/// verified and usable.
#[derive(Debug, Clone)]
pub struct PushState {
    pub epoch: String,
    pub seq: u64,
    pub trigger: String,
    pub snapshot_source: Option<String>,
    /// The snapshot in forecast form (its timeline samples are already ingested
    /// and taken out).
    pub view: Arc<ForecastView>,
    /// The state time on the router (forecast) clock: `sampled_monotonic_s` +
    /// the offset.
    pub t0_s: f64,
    /// When the router read the message (forecast clock).
    pub received_s: f64,
    /// The offset used: router clock minus engine monotonic clock (s).
    pub offset_s: f64,
    /// A newer message of the same epoch had no snapshot: its sequence.
    pub superseded_seq: Option<u64>,
}

/// Sliding-window minimum of `x`: (read time, x), x increasing.
#[derive(Debug, Default)]
struct OffsetWindow {
    q: VecDeque<(f64, f64)>,
}

impl OffsetWindow {
    /// Add `x` read at router time `at`; the minimum over the window.
    fn add(&mut self, at: f64, x: f64) -> f64 {
        while self.q.back().is_some_and(|b| b.1 >= x) {
            self.q.pop_back();
        }
        self.q.push_back((at, x));
        while self.q.front().is_some_and(|f| f.0 < at - OFFSET_WINDOW_S) {
            self.q.pop_front();
        }
        self.q.front().map_or(x, |f| f.1)
    }
}

/// Counters of one P's push (cumulative; the `forecast_push_stats` event).
#[derive(Debug, Default, Clone, Serialize)]
pub struct PushCounts {
    pub connects: u64,
    pub connect_failures: u64,
    pub recv_errors: u64,
    /// Messages that passed the checks and were in sequence.
    pub messages: u64,
    pub bytes: u64,
    pub max_bytes: u64,
    /// Messages per `trigger`.
    pub triggers: BTreeMap<String, u64>,
    /// Sequence jumps within an epoch, and the messages they lost.
    pub gaps: u64,
    pub lost: u64,
    /// Messages with a sequence that was not newer (dropped).
    pub repeats: u64,
    pub epoch_changes: u64,
    /// Messages dropped by the frame and schema checks, per reason.
    pub dropped: BTreeMap<String, u64>,
    /// Messages with a snapshot that became the state.
    pub states: u64,
    /// Messages without a usable state time, per `snapshot_omitted` reason
    /// (`no_sampled_monotonic_s` when the snapshot has no state time).
    pub snapshot_omitted: BTreeMap<String, u64>,
    /// States whose view cannot give a forecast, per reason.
    pub unusable: BTreeMap<String, u64>,
    /// States whose pending passes differ from the message's `inflight`.
    pub inflight_mismatch: u64,
    pub launch_events: u64,
    pub result_events: u64,
    /// `forecast_label` lines from pushed timeline samples.
    pub labels: u64,
    /// Receive delay above the window minimum (`x - offset`), sum and max
    /// (s): how much later than the fastest message a message was read.
    pub excess_sum_s: f64,
    pub excess_max_s: f64,
    /// The engine's own totals in the newest message.
    pub engine_dropped_total: u64,
    pub engine_errors_total: u64,
    pub engine_oversize_total: u64,
    /// `prev_flush_us` (the engine's whole push call), newest and maximum.
    pub prev_flush_us_last: Option<f64>,
    pub prev_flush_us_max: f64,
}

/// The wire state of one P.
#[derive(Debug, Default)]
struct PushStream {
    endpoint: Option<String>,
    /// Another pool URL has the same endpoint: no subscriber.
    conflict: bool,
    epoch: Option<String>,
    last_seq: Option<u64>,
    offset: OffsetWindow,
    offset_s: Option<f64>,
    last_received_s: Option<f64>,
    counts: PushCounts,
    /// Reasons already logged (once per P each).
    warned: BTreeSet<&'static str>,
}

impl PushStream {
    fn warn_once(&mut self, reason: &'static str) -> bool {
        self.warned.insert(reason)
    }
}

/// What one applied message changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Applied {
    pub seq: u64,
    /// The message's snapshot became the P's state.
    pub state: bool,
    pub labels: usize,
}

struct PushTask {
    endpoint: String,
    cancel: CancellationToken,
}

/// The subscribers of all P workers and their newest states.
pub struct PushHub {
    port: u16,
    online: Arc<ForecastOnline>,
    shadow: Option<Arc<ShadowSink>>,
    book: Option<Arc<crate::prefill_work::PrefillWorkBook>>,
    // Lock order: tasks, then streams, then states.
    tasks: Mutex<HashMap<String, PushTask>>,
    streams: Mutex<BTreeMap<String, PushStream>>,
    states: RwLock<HashMap<String, Arc<PushState>>>,
    stats_at: Mutex<Option<Instant>>,
}

impl std::fmt::Debug for PushHub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PushHub")
            .field("port", &self.port)
            .field("subscribers", &self.tasks.lock().len())
            .finish_non_exhaustive()
    }
}

impl Drop for PushHub {
    fn drop(&mut self) {
        for task in self.tasks.get_mut().values() {
            task.cancel.cancel();
        }
    }
}

impl PushHub {
    /// `online` takes the pushed timeline samples, `shadow` the labels and
    /// `book` the `forecast_push_stats` events.
    pub fn new(
        port: u16,
        online: Arc<ForecastOnline>,
        shadow: Option<Arc<ShadowSink>>,
        book: Option<Arc<crate::prefill_work::PrefillWorkBook>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            port,
            online,
            shadow,
            book,
            tasks: Mutex::new(HashMap::new()),
            streams: Mutex::new(BTreeMap::new()),
            states: RwLock::new(HashMap::new()),
            stats_at: Mutex::new(None),
        })
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    /// The newest pushed state of `worker`.
    pub fn state_of(&self, worker: &str) -> Option<Arc<PushState>> {
        self.states.read().get(worker).cloned()
    }

    /// Keep one subscriber per P worker of `urls` (the routable pool): start
    /// the missing ones and stop those of workers that left the pool or
    /// whose endpoint another pool URL shares. Needs a Tokio runtime to start
    /// a subscriber.
    pub fn ensure(self: &Arc<Self>, urls: &[String]) {
        let mut by_endpoint: BTreeMap<String, Vec<&String>> = BTreeMap::new();
        let mut bad: Vec<(&String, &'static str)> = Vec::new();
        for url in urls {
            match endpoint_of(url, self.port) {
                Ok(endpoint) => by_endpoint.entry(endpoint).or_default().push(url),
                Err(reason) => bad.push((url, reason)),
            }
        }
        let mut wanted: HashMap<String, String> = HashMap::new();
        let mut conflicts: Vec<(&String, &String)> = Vec::new();
        for (endpoint, group) in &by_endpoint {
            match group.as_slice() {
                [url] => {
                    wanted.insert((*url).clone(), endpoint.clone());
                }
                group => conflicts.extend(group.iter().map(|url| (*url, endpoint))),
            }
        }
        let mut tasks = self.tasks.lock();
        tasks.retain(|url, task| {
            let keep = wanted.get(url) == Some(&task.endpoint);
            if !keep {
                task.cancel.cancel();
                self.states.write().remove(url);
                tracing::info!(target: "symmetric_pd", worker = %url, endpoint = %task.endpoint,
                    "forecast state push: subscriber stopped (the worker left the pool or shares \
                     its host)");
            }
            keep
        });
        {
            let mut streams = self.streams.lock();
            for (url, endpoint) in &conflicts {
                let st = streams.entry((*url).clone()).or_default();
                st.endpoint = Some((*endpoint).clone());
                st.conflict = true;
                if st.warn_once("endpoint_conflict") {
                    tracing::error!(target: "symmetric_pd", worker = %url, endpoint = %endpoint,
                        "forecast state push: another P of the pool has the same host, and the push \
                         port is one per router; this P keeps the polled snapshot");
                }
            }
            for (url, reason) in &bad {
                let st = streams.entry((*url).clone()).or_default();
                if st.warn_once("bad_url") {
                    tracing::error!(target: "symmetric_pd", worker = %url, reason,
                        "forecast state push: no endpoint for this worker URL; it keeps the polled \
                         snapshot");
                }
            }
            for (url, endpoint) in &wanted {
                let st = streams.entry(url.clone()).or_default();
                st.endpoint = Some(endpoint.clone());
                st.conflict = false;
            }
        }
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        for (url, endpoint) in wanted {
            if tasks.contains_key(&url) {
                continue;
            }
            let cancel = CancellationToken::new();
            handle.spawn(run_subscriber(
                Arc::downgrade(self),
                url.clone(),
                endpoint.clone(),
                cancel.clone(),
            ));
            tracing::info!(target: "symmetric_pd", worker = %url, %endpoint,
                "forecast state push: subscriber started");
            tasks.insert(url, PushTask { endpoint, cancel });
        }
    }

    /// Apply one message of `worker`, read at `received`: the checks, the
    /// sequence and offset bookkeeping, then (with a snapshot) the timeline
    /// samples and the new state. Messages of one P must come in read order.
    pub fn on_message(
        &self,
        worker: &str,
        frames: &[&[u8]],
        received: Instant,
    ) -> Result<Applied, &'static str> {
        let received_s = self.online.secs(received);
        let msg = match decode_frames(frames) {
            Ok(msg) => msg,
            Err(bad) => {
                let mut streams = self.streams.lock();
                let st = streams.entry(worker.to_string()).or_default();
                *st.counts.dropped.entry(bad.reason.to_string()).or_default() += 1;
                if st.warn_once(bad.reason) {
                    tracing::warn!(target: "symmetric_pd", worker, reason = bad.reason,
                        detail = %bad.detail,
                        "forecast state push: message dropped (further ones counted)");
                }
                return Err(bad.reason);
            }
        };
        let p = &msg.payload;
        // Parsed off every lock.
        let mut view = p.snapshot.as_ref().map(ForecastView::parse);
        let sampled = view.as_ref().and_then(|v| v.sampled_mono_s);
        let (offset, new_epoch) = {
            let mut streams = self.streams.lock();
            let st = streams.entry(worker.to_string()).or_default();
            let new_epoch = st.epoch.as_deref() != Some(p.epoch.as_str());
            if new_epoch {
                if st.epoch.is_some() {
                    st.counts.epoch_changes += 1;
                }
                st.epoch = Some(p.epoch.clone());
                st.last_seq = None;
                st.offset = OffsetWindow::default();
            }
            if let Some(last) = st.last_seq {
                if msg.seq <= last {
                    st.counts.repeats += 1;
                    return Err("repeat");
                }
                if msg.seq > last + 1 {
                    st.counts.gaps += 1;
                    st.counts.lost += msg.seq - last - 1;
                }
            }
            st.last_seq = Some(msg.seq);
            let x = received_s - p.sent_mono_s.unwrap_or(f64::NAN);
            let offset = st.offset.add(received_s, x);
            st.offset_s = Some(offset);
            st.last_received_s = Some(received_s);
            let c = &mut st.counts;
            c.messages += 1;
            c.bytes += msg.bytes as u64;
            c.max_bytes = c.max_bytes.max(msg.bytes as u64);
            *c.triggers.entry(p.trigger.clone()).or_default() += 1;
            c.excess_sum_s += x - offset;
            c.excess_max_s = c.excess_max_s.max(x - offset);
            for e in &p.events {
                match e.kind.as_str() {
                    "launch" => c.launch_events += 1,
                    "result" => c.result_events += 1,
                    _ => {}
                }
            }
            c.engine_dropped_total = p.dropped_total;
            c.engine_errors_total = p.errors_total;
            c.engine_oversize_total = p.oversize_total;
            if let Some(us) = p.prev_flush_us.filter(|us| us.is_finite()) {
                c.prev_flush_us_last = Some(us);
                c.prev_flush_us_max = c.prev_flush_us_max.max(us);
            }
            match (&view, sampled) {
                (Some(v), Some(_)) => {
                    c.states += 1;
                    if let Some(reason) = v.unusable {
                        *c.unusable.entry(reason.to_string()).or_default() += 1;
                    }
                    if inflight_differs(v, &p.inflight) {
                        c.inflight_mismatch += 1;
                    }
                }
                (Some(_), None) => {
                    *c.snapshot_omitted
                        .entry("no_sampled_monotonic_s".to_string())
                        .or_default() += 1;
                }
                (None, _) => {
                    let reason = p.snapshot_omitted.as_deref().unwrap_or("unknown");
                    *c.snapshot_omitted.entry(reason.to_string()).or_default() += 1;
                }
            }
            (offset, new_epoch)
        };
        if new_epoch {
            // The state of the previous engine process is not this one's.
            self.states.write().remove(worker);
        }
        let (Some(v), Some(sampled)) = (view.as_mut(), sampled) else {
            self.supersede(worker, &p.epoch, msg.seq);
            return Ok(Applied {
                seq: msg.seq,
                state: false,
                labels: 0,
            });
        };
        let t0_s = sampled + offset;
        let labels = self.online.observe_push(worker, v, t0_s, received_s);
        let n_labels = labels.len();
        if let Some(sink) = &self.shadow {
            for label in labels {
                sink.write(label);
            }
        }
        if n_labels > 0 {
            if let Some(st) = self.streams.lock().get_mut(worker) {
                st.counts.labels += n_labels as u64;
            }
        }
        v.samples = Vec::new();
        v.boots = Vec::new();
        let state = Arc::new(PushState {
            epoch: p.epoch.clone(),
            seq: msg.seq,
            trigger: p.trigger.clone(),
            snapshot_source: p.snapshot_source.clone(),
            view: Arc::new(view.take().expect("checked above")),
            t0_s,
            received_s,
            offset_s: offset,
            superseded_seq: None,
        });
        let mut states = self.states.write();
        let newer = states
            .get(worker)
            .is_some_and(|old| old.epoch == state.epoch && old.seq > state.seq);
        if !newer {
            states.insert(worker.to_string(), state);
        }
        Ok(Applied {
            seq: msg.seq,
            state: !newer,
            labels: n_labels,
        })
    }

    /// A newer message of the same epoch without a state: the stored state
    /// stays (it is the newest full state) and names that message.
    fn supersede(&self, worker: &str, epoch: &str, seq: u64) {
        let mut states = self.states.write();
        if let Some(old) = states.get(worker) {
            if old.epoch == epoch && old.seq < seq {
                let mut state = (**old).clone();
                state.superseded_seq = Some(seq);
                states.insert(worker.to_string(), Arc::new(state));
            }
        }
    }

    fn note_connect(&self, worker: &str, endpoint: &str, result: Result<(), &str>) {
        let mut streams = self.streams.lock();
        let st = streams.entry(worker.to_string()).or_default();
        match result {
            Ok(()) => {
                st.counts.connects += 1;
                tracing::info!(target: "symmetric_pd", worker, endpoint,
                    connects = st.counts.connects, "forecast state push: connected and subscribed");
            }
            Err(error) => {
                st.counts.connect_failures += 1;
                if st.warn_once("connect") {
                    tracing::warn!(target: "symmetric_pd", worker, endpoint, error,
                        "forecast state push: cannot connect yet (retrying; the P keeps the polled \
                         snapshot meanwhile; further failures counted)");
                } else {
                    tracing::debug!(target: "symmetric_pd", worker, endpoint, error,
                        "forecast state push: connect failed");
                }
            }
        }
    }

    fn note_recv_error(&self, worker: &str, error: &str) {
        let mut streams = self.streams.lock();
        let st = streams.entry(worker.to_string()).or_default();
        st.counts.recv_errors += 1;
        if st.warn_once("recv") {
            tracing::warn!(target: "symmetric_pd", worker, error,
                "forecast state push: receive error (zeromq reconnects; further errors counted)");
        }
    }

    /// Counters and state per P, at router time now.
    pub fn stats(&self) -> Value {
        let now_s = self.online.secs(Instant::now());
        let streams = self.streams.lock();
        let states = self.states.read();
        let workers: serde_json::Map<String, Value> = streams
            .iter()
            .map(|(url, st)| {
                let state = states.get(url);
                let value = json!({
                    "endpoint": st.endpoint,
                    "endpoint_conflict": st.conflict,
                    "epoch": st.epoch,
                    "last_seq": st.last_seq,
                    "offset_s": st.offset_s,
                    "since_last_message_s": st.last_received_s.map(|r| now_s - r),
                    "state_seq": state.map(|s| s.seq),
                    "state_age_s": state.map(|s| now_s - s.t0_s),
                    "state_superseded_seq": state.and_then(|s| s.superseded_seq),
                    "state_unusable": state.and_then(|s| s.view.unusable),
                    "counts": serde_json::to_value(&st.counts).unwrap_or(Value::Null),
                });
                (url.clone(), value)
            })
            .collect();
        Value::Object(workers)
    }

    /// Every [`STATS_INTERVAL`]: one `forecast_push_stats` event in the placement trace
    /// and one log line per P.
    pub fn maybe_emit_stats(&self, now: Instant) {
        {
            let mut at = self.stats_at.lock();
            let last: Option<Instant> = *at;
            if last.is_some_and(|l| now.saturating_duration_since(l) < STATS_INTERVAL) {
                return;
            }
            *at = Some(now);
            if last.is_none() {
                // The first interval starts with the first call.
                return;
            }
        }
        let stats = self.stats();
        if let Value::Object(workers) = &stats {
            for (url, s) in workers {
                let c = &s["counts"];
                tracing::info!(target: "symmetric_pd", worker = %url,
                    messages = c["messages"].as_u64().unwrap_or(0),
                    states = c["states"].as_u64().unwrap_or(0),
                    lost = c["lost"].as_u64().unwrap_or(0),
                    gaps = c["gaps"].as_u64().unwrap_or(0),
                    repeats = c["repeats"].as_u64().unwrap_or(0),
                    epoch_changes = c["epoch_changes"].as_u64().unwrap_or(0),
                    inflight_mismatch = c["inflight_mismatch"].as_u64().unwrap_or(0),
                    offset_s = s["offset_s"].as_f64().unwrap_or(f64::NAN),
                    state_age_s = s["state_age_s"].as_f64().unwrap_or(f64::NAN),
                    "forecast state push stats");
            }
        }
        if let Some(book) = &self.book {
            let event = json!({
                "event": "forecast_push_stats",
                "forecast_schema": STATS_SCHEMA,
                "port": self.port,
                "forecast_clock_s": self.online.secs(now),
                "workers": stats,
            });
            if let Err(error) = book.emit_event(event) {
                tracing::warn!(target: "symmetric_pd", %error,
                    "state push stats not written to the placement trace");
            }
        }
    }
}

/// The snapshot's pending passes and the message's `inflight` list name
/// different passes (only when both name all of theirs).
fn inflight_differs(view: &ForecastView, inflight: &[PassEvent]) -> bool {
    let snap: Option<BTreeSet<i64>> = view.inflight.iter().map(|p| p.forward_iter).collect();
    let msg: Option<BTreeSet<i64>> = inflight.iter().map(|e| e.forward_iter).collect();
    matches!((snap, msg), (Some(a), Some(b)) if a != b)
}

/// One P's subscriber: connect (retrying), subscribe to [`TOPIC`], and hand
/// every message to the hub with the time it was read. Ends when cancelled
/// or when the hub is gone.
async fn run_subscriber(
    hub: Weak<PushHub>,
    worker: String,
    endpoint: String,
    cancel: CancellationToken,
) {
    let topic = std::str::from_utf8(TOPIC).expect("ASCII topic");
    let mut delay = RECONNECT_MIN;
    loop {
        let mut sub = SubSocket::new();
        let connected = tokio::select! {
            biased;
            _ = cancel.cancelled() => return,
            result = sub.connect(&endpoint) => result,
        };
        let subscribed = match connected {
            Ok(()) => sub.subscribe(topic).await,
            Err(error) => Err(error),
        };
        let Some(h) = hub.upgrade() else { return };
        if let Err(error) = subscribed {
            let error = error.to_string();
            h.note_connect(&worker, &endpoint, Err(error.as_str()));
            drop(h);
            tokio::select! {
                biased;
                _ = cancel.cancelled() => return,
                _ = tokio::time::sleep(delay) => {}
            }
            delay = (delay * 2).min(RECONNECT_MAX);
            continue;
        }
        h.note_connect(&worker, &endpoint, Ok(()));
        drop(h);
        delay = RECONNECT_MIN;
        let mut errors = 0u32;
        loop {
            let got = tokio::select! {
                biased;
                _ = cancel.cancelled() => return,
                result = sub.recv() => result,
            };
            // The read time of the message (just after the read).
            let received = Instant::now();
            let Some(h) = hub.upgrade() else { return };
            match got {
                Ok(msg) => {
                    errors = 0;
                    let frames = msg.into_vec();
                    let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_ref()).collect();
                    // Drops are counted by the hub.
                    let _ = h.on_message(&worker, &refs, received);
                }
                Err(error) => {
                    errors += 1;
                    h.note_recv_error(&worker, &error.to_string());
                    drop(h);
                    if errors >= RECV_ERROR_CEILING {
                        break;
                    }
                    tokio::select! {
                        biased;
                        _ = cancel.cancelled() => return,
                        _ = tokio::time::sleep(RECV_ERROR_PAUSE) => {}
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/forecast")
            .join(name)
    }

    /// The recorded `prefill_routing` reply (PROVENANCE.md).
    fn recorded_snapshot() -> Value {
        let text = std::fs::read_to_string(fixture("routing-snapshot-recorded.json")).unwrap();
        let rec: Value = serde_json::from_str(&text).unwrap();
        rec["candidate"]["snapshot"]["raw_prefill_routing"].clone()
    }

    /// A v1 payload with `snapshot` (None: omitted over the
    /// byte cap), sent at engine time `sent_mono_s`, and one field that a
    /// v1 decoder does not know.
    fn payload(epoch: &str, seq: u64, snapshot: Option<Value>, sent_mono_s: f64) -> Value {
        let omitted = snapshot.is_none().then_some("over_max_bytes");
        let source = snapshot.is_some().then_some("push_build");
        json!({
            "schema": WIRE_SCHEMA, "schema_version": 1, "epoch": epoch, "seq": seq, "rank": 0,
            "trigger": "result", "sent_mono_s": sent_mono_s, "sent_unix_s": 1.7e9 + sent_mono_s,
            "mono_to_unix_s": 1.7e9, "clock_domain_id": "boot-1",
            "events": [{"kind": "result", "forward_iter": 7, "launch_mono_s": sent_mono_s - 0.3,
                        "observed_mono_s": sent_mono_s - 0.002, "gpu_started": null,
                        "gpu_ended": null, "gpu_ms": null, "gpu_gap_before_ms": null,
                        "gpu_start_mono_s": null, "gpu_end_mono_s": null}],
            "inflight": [], "gpu_timing": "unavailable", "gpu_clock_samples": 0,
            "dropped_total": 0, "errors_total": 0, "oversize_total": 0, "build_us": 400.0,
            "convert_us": 120.0, "prev_encode_us": 200.0, "prev_send_us": 30.0,
            "prev_bytes": 20000, "prev_flush_us": 800.0, "snapshot_source": source,
            "snapshot": snapshot, "snapshot_omitted": omitted,
            "a_field_from_a_later_engine": [1, 2, 3],
        })
    }

    /// A small usable snapshot (an idle P) with its state time.
    fn small_snapshot(epoch: &str, sampled: f64) -> Value {
        json!({"epoch": epoch, "sequence": 1, "available": true, "stale": false,
            "truncated": false, "age_s": 0.0002, "sampled_monotonic_s": sampled,
            "inflight_batches_known": true, "inflight_batches": [], "waiting": [],
            "bootstrap": [], "inflight_requests": [],
            "policy": {"page_size": 64, "chunk_tokens": 8192, "schedule_policy": "fcfs",
                       "share_enabled": false},
            "admission_resources": {"token_pool_capacity": 1000000}})
    }

    fn frames(seq: u64, payload: &Value) -> Vec<Vec<u8>> {
        vec![
            TOPIC.to_vec(),
            seq.to_be_bytes().to_vec(),
            rmp_serde::to_vec_named(payload).unwrap(),
        ]
    }

    fn refs(frames: &[Vec<u8>]) -> Vec<&[u8]> {
        frames.iter().map(Vec::as_slice).collect()
    }

    /// Send `msg` (its own seq in frame 2) to the hub as read at `read`.
    fn apply(hub: &PushHub, w: &str, msg: &Value, read: Instant) -> Result<Applied, &'static str> {
        let seq = msg["seq"].as_u64().unwrap();
        hub.on_message(w, &refs(&frames(seq, msg)), read)
    }

    /// A message of epoch `epoch` read at router time `read` with
    /// x = read - sent; its state is sampled 0.2 ms before the
    /// send.
    fn timed(epoch: &str, seq: u64, read: f64, x: f64) -> Value {
        let sent = read - x;
        payload(epoch, seq, Some(small_snapshot(epoch, sent - 0.0002)), sent)
    }

    fn hub() -> (Arc<PushHub>, Instant) {
        let anchor = Instant::now();
        let online = Arc::new(ForecastOnline::new(anchor).with_label_sources());
        (PushHub::new(5600, online, None, None), anchor)
    }

    fn at(anchor: Instant, s: f64) -> Instant {
        anchor + Duration::from_secs_f64(s)
    }

    /// The recorded snapshot in a v1 payload: the parsed snapshot
    /// becomes the state, its state time is sampled_monotonic_s + offset
    /// (offset = read - sent of the first message), and its 64 timeline
    /// samples are ingested and taken out of the stored view. The message's
    /// inflight list is checked against the snapshot's pending pass.
    #[test]
    fn a_pushed_snapshot_becomes_the_state_with_the_offset_clock() {
        let (hub, anchor) = hub();
        let w = "http://p:1";
        let epoch = "8e884f52591543248367e6693fba8b72";
        let snap = recorded_snapshot();
        let sampled = snap["sampled_monotonic_s"].as_f64().unwrap();
        let sent = sampled + 0.0005;
        let mut msg = payload(epoch, 41, Some(snap), sent);
        // As the engine sends it: the inflight list names the pending pass.
        msg["inflight"] = json!([{"kind": "inflight", "forward_iter": 2051}]);
        let applied = apply(&hub, w, &msg, at(anchor, 1_000.0)).unwrap();
        let want = Applied {
            seq: 41,
            state: true,
            labels: 0,
        };
        assert_eq!(applied, want);
        let s = hub.state_of(w).unwrap();
        assert_eq!((s.seq, s.trigger.as_str()), (41, "result"));
        assert_eq!(s.snapshot_source.as_deref(), Some("push_build"));
        assert_eq!(s.view.unusable, None);
        assert_eq!(s.view.inflight.len(), 1);
        assert_eq!(s.view.inflight[0].forward_iter, Some(2051));
        assert!(s.view.samples.is_empty() && s.view.boots.is_empty());
        let received = hub.online.secs(at(anchor, 1_000.0));
        assert_eq!(s.offset_s, received - sent);
        assert_eq!(s.t0_s, sampled + (received - sent));
        let age = received - s.t0_s;
        assert!((age - 0.0005).abs() < 1e-6, "{age}");
        let stats = hub.stats();
        let c = &stats[w]["counts"];
        assert_eq!(c["messages"], 1);
        assert_eq!(c["states"], 1);
        assert_eq!(c["result_events"], 1);
        assert_eq!(c["inflight_mismatch"], 0);
        assert_eq!(stats[w]["epoch"], epoch);
        let online = hub.online.counts();
        assert_eq!(online[w]["launch_samples"], 13, "{online}");
        // A message whose inflight list leaves out the pending pass counts.
        let other = payload(epoch, 42, Some(recorded_snapshot()), sent + 0.01);
        apply(&hub, w, &other, at(anchor, 1_000.01)).unwrap();
        assert_eq!(hub.stats()[w]["counts"]["inflight_mismatch"], 1);
    }

    /// Sequence gaps count the lost messages; a repeat is dropped; the
    /// offset is the minimum read - sent of the last 30 s; a new epoch
    /// restarts the sequence, the offset window and the state.
    #[test]
    fn sequence_gaps_repeats_epochs_and_the_offset_window() {
        let (hub, anchor) = hub();
        let w = "http://p:1";
        for (seq, read, x) in [(1, 100.0, 10.010), (2, 101.0, 10.004), (5, 102.0, 10.020)] {
            apply(&hub, w, &timed("e1", seq, read, x), at(anchor, read)).unwrap();
        }
        let s = hub.state_of(w).unwrap();
        assert_eq!(s.seq, 5);
        assert!((s.offset_s - 10.004).abs() < 1e-9, "{}", s.offset_s);
        let stats = hub.stats();
        let c = &stats[w]["counts"];
        assert_eq!((c["gaps"].as_u64(), c["lost"].as_u64()), (Some(1), Some(2)));
        // A repeat of seq 5 is dropped.
        let again = apply(&hub, w, &timed("e1", 5, 103.0, 10.0), at(anchor, 103.0));
        assert_eq!(again, Err("repeat"));
        assert_eq!(hub.stats()[w]["counts"]["repeats"], 1);
        // 40 s later the 10.004 sample has left the window.
        apply(&hub, w, &timed("e1", 6, 143.0, 10.030), at(anchor, 143.0)).unwrap();
        assert!((hub.state_of(w).unwrap().offset_s - 10.030).abs() < 1e-9);
        // A new epoch (an engine restart) starts at seq 1: no loss, a new
        // offset window.
        apply(&hub, w, &timed("e2", 1, 150.0, 140.0), at(anchor, 150.0)).unwrap();
        let s = hub.state_of(w).unwrap();
        assert_eq!((s.epoch.as_str(), s.seq), ("e2", 1));
        assert!((s.offset_s - 140.0).abs() < 1e-9, "{}", s.offset_s);
        let stats = hub.stats();
        let c = &stats[w]["counts"];
        assert_eq!(c["epoch_changes"], 1);
        assert_eq!(c["lost"], 2);
        assert_eq!(c["messages"], 5);
    }

    /// Frames and payloads that break the contract are dropped and counted
    /// by reason; an unknown payload field is ignored.
    #[test]
    fn bad_messages_are_dropped_and_counted() {
        let (hub, anchor) = hub();
        let w = "http://p:1";
        let now = at(anchor, 10.0);
        let good = payload("e1", 3, Some(small_snapshot("e1", 5.0)), 5.001);
        let f = frames(3, &good);
        let check = |frames: &[&[u8]], reason: &'static str| {
            assert_eq!(hub.on_message(w, frames, now), Err(reason));
        };
        let (topic, seq, body) = (f[0].as_slice(), f[1].as_slice(), f[2].as_slice());
        check(&[topic, body], "frame_count");
        check(&[b"other".as_slice(), seq, body], "topic");
        check(&[topic, b"\x00\x01".as_slice(), body], "seq_frame");
        check(&[topic, seq, b"\xc1".as_slice()], "decode");
        check(
            &[topic, 4u64.to_be_bytes().as_slice(), body],
            "seq_mismatch",
        );
        for (key, value, reason) in [
            ("schema_version", json!(2), "schema"),
            ("rank", json!(1), "rank"),
            ("epoch", json!(""), "no_epoch"),
            ("sent_mono_s", Value::Null, "no_send_time"),
        ] {
            let mut bad = good.clone();
            bad[key] = value;
            check(&refs(&frames(3, &bad)), reason);
        }
        assert!(hub.state_of(w).is_none());
        // The good message (with its unknown field) applies.
        assert!(hub.on_message(w, &refs(&f), now).unwrap().state);
        let stats = hub.stats();
        let dropped = &stats[w]["counts"]["dropped"];
        for reason in [
            "frame_count",
            "topic",
            "seq_frame",
            "decode",
            "seq_mismatch",
            "schema",
            "rank",
            "no_epoch",
            "no_send_time",
        ] {
            assert_eq!(dropped[reason], 1, "{reason}: {dropped}");
        }
        assert_eq!(stats[w]["counts"]["messages"], 1);
    }

    /// A message without a snapshot keeps the newest state and marks it
    /// superseded; an unusable snapshot still becomes the state (the
    /// decision then keeps the poll) and is counted.
    #[test]
    fn snapshot_less_and_unusable_messages() {
        let (hub, anchor) = hub();
        let w = "http://p:1";
        apply(&hub, w, &timed("e1", 1, 10.0, 5.0), at(anchor, 10.0)).unwrap();
        let omitted = payload("e1", 2, None, 5.2);
        assert!(!apply(&hub, w, &omitted, at(anchor, 10.2)).unwrap().state);
        let s = hub.state_of(w).unwrap();
        assert_eq!((s.seq, s.superseded_seq), (1, Some(2)));
        let mut truncated = small_snapshot("e1", 5.4);
        truncated["truncated"] = json!(true);
        let truncated = payload("e1", 3, Some(truncated), 5.401);
        apply(&hub, w, &truncated, at(anchor, 10.4)).unwrap();
        let s = hub.state_of(w).unwrap();
        assert_eq!(
            (s.seq, s.superseded_seq, s.view.unusable),
            (3, None, Some("truncated"))
        );
        let stats = hub.stats();
        let c = &stats[w]["counts"];
        assert_eq!(c["snapshot_omitted"]["over_max_bytes"], 1);
        assert_eq!(c["unusable"]["truncated"], 1);
        assert_eq!(c["states"], 2);
    }

    /// The pushed timeline samples feed the online fits and label routed
    /// requests (`label_source` push, a point interval); the engine clock may
    /// be far from the router's (here +400 s).
    #[test]
    fn pushed_timeline_samples_feed_the_fits_and_label_routed_requests() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shadow.jsonl");
        let anchor = Instant::now();
        let online = Arc::new(ForecastOnline::new(anchor).with_label_sources());
        let sink = Arc::new(ShadowSink::file(&path).unwrap());
        let hub = PushHub::new(5600, online, Some(sink), None);
        let w = "http://p:1";
        hub.online
            .record_route(crate::prefill_forecast::RouteRecord {
                rid: "r1".into(),
                worker: w.into(),
                decided_s: 50.0,
                decided_unix_s: 1_000.0,
                input: 2_000,
                prefix: 0,
                host: 0,
                idle: Some(true),
                pred_s: Some(0.25),
                session: None,
                session_gap_s: None,
            });
        let mut snap = small_snapshot("e1", 500.0);
        snap["timeline"] = json!({"samples": [{"sequence": 1, "kind": "request_complete",
            "forward_iter": 3, "rid": "r1", "duration_s": 0.25, "completed_age_s": 49.5,
            "features": {"input_tokens": 2000, "receive_to_entry_s": 0.002,
                         "cached_tokens_device": 0, "cached_tokens_host": 0,
                         "cached_tokens_storage": 0, "cache_breakdown_computed": true}}]});
        // Read at router time 100.0005, sent at engine time 500.0005.
        let msg = payload("e1", 1, Some(snap), 500.0005);
        let applied = apply(&hub, w, &msg, at(anchor, 100.0005)).unwrap();
        assert_eq!(applied.labels, 1);
        let t0 = hub.state_of(w).unwrap().t0_s;
        assert!((t0 - (hub.online.secs(at(anchor, 100.0005)) - 0.0005)).abs() < 1e-9);
        assert_eq!(hub.stats()[w]["counts"]["labels"], 1);
        // done = t0 - 49.5 = 50.5, receipt = 50.25: 0.25 s after the decision.
        assert_eq!(hub.online.counts()[w]["rx_samples"], 1);
        let text = std::fs::read_to_string(&path).unwrap();
        let label: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(label["event"], "forecast_label");
        assert_eq!(label["label_source"], "push");
        assert_eq!(label["label_lo_s"], label["label_hi_s"]);
        assert!(
            (label["label_s"].as_f64().unwrap() - 0.5).abs() < 1e-6,
            "{label}"
        );
    }

    /// Endpoints use the URL host and the push port. Two pool URLs on one
    /// host get no subscriber (their messages could not be told apart); a
    /// worker that leaves the pool loses its subscriber.
    #[tokio::test]
    async fn workers_that_share_a_host_keep_the_poll() {
        let ep = |url: &str| endpoint_of(url, 5600);
        assert_eq!(
            ep("http://10.0.0.1:30000"),
            Ok("tcp://10.0.0.1:5600".to_string())
        );
        assert_eq!(ep("http://[::1]:30000"), Ok("tcp://[::1]:5600".to_string()));
        assert!(ep("not a url").is_err());
        let (hub, _) = hub();
        hub.ensure(&[
            "http://10.0.0.1:30000".to_string(),
            "http://10.0.0.1:30001".to_string(),
            "http://10.0.0.2:30000".to_string(),
        ]);
        {
            let tasks = hub.tasks.lock();
            assert_eq!(tasks.len(), 1);
            assert_eq!(
                tasks["http://10.0.0.2:30000"].endpoint,
                "tcp://10.0.0.2:5600"
            );
        }
        let stats = hub.stats();
        assert_eq!(stats["http://10.0.0.1:30000"]["endpoint_conflict"], true);
        assert_eq!(stats["http://10.0.0.1:30001"]["endpoint_conflict"], true);
        assert_eq!(stats["http://10.0.0.2:30000"]["endpoint_conflict"], false);
        hub.ensure(&["http://10.0.0.1:30000".to_string()]);
        let tasks = hub.tasks.lock();
        assert_eq!(tasks.len(), 1);
        assert!(tasks.contains_key("http://10.0.0.1:30000"));
    }

    /// A real PUB socket (the engine side): the subscriber connects to
    /// tcp://<host>:PORT, subscribes to the topic and a message becomes the
    /// state. Messages are sent until one arrives (a SUB joins late).
    #[tokio::test]
    async fn the_subscriber_reads_a_pub_socket() {
        use zeromq::{Endpoint, PubSocket, SocketSend, ZmqMessage};
        let mut publisher = PubSocket::new();
        let port = match publisher.bind("tcp://127.0.0.1:0").await.unwrap() {
            Endpoint::Tcp(_, port) => port,
            other => panic!("unexpected endpoint {other:?}"),
        };
        let online = Arc::new(ForecastOnline::new(Instant::now()).with_label_sources());
        let hub = PushHub::new(port, online, None, None);
        let worker = "http://127.0.0.1:30000".to_string();
        hub.ensure(std::slice::from_ref(&worker));
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut seq = 0u64;
        while hub.state_of(&worker).is_none() {
            assert!(Instant::now() < deadline, "no message arrived");
            seq += 1;
            // The engine clock is the router clock + 1,000 s.
            let sampled = hub.online.secs(Instant::now()) + 1_000.0;
            let msg = payload(
                "e1",
                seq,
                Some(small_snapshot("e1", sampled)),
                sampled + 3e-4,
            );
            let mut out = ZmqMessage::from(bytes::Bytes::from_static(TOPIC));
            out.push_back(bytes::Bytes::copy_from_slice(&seq.to_be_bytes()));
            out.push_back(bytes::Bytes::from(rmp_serde::to_vec_named(&msg).unwrap()));
            publisher.send(out).await.unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let s = hub.state_of(&worker).unwrap();
        assert_eq!(s.epoch, "e1");
        assert!(s.view.unusable.is_none());
        assert!((s.offset_s + 1_000.0).abs() < 1.0, "{}", s.offset_s);
        let stats = hub.stats();
        assert!(stats[&worker]["counts"]["connects"].as_u64().unwrap() >= 1);
    }

    /// The engine's own encoding (msgspec and the PrefillStatePush struct
    /// of the engine) of the recorded snapshot as an
    /// end-of-iteration message that reuses the /v1/loads sample
    /// (PROVENANCE.md): every field decodes, the snapshot equals the
    /// recorded reply with the three send-time fields, and it gives the
    /// same forecast rows. The same message without its snapshot decodes too.
    #[test]
    fn the_engine_encoded_payload_decodes() {
        let text = std::fs::read_to_string(fixture("prefill-state-push-recorded.json")).unwrap();
        let meta: Value = serde_json::from_str(&text).unwrap();
        let expected = &meta["expected"];
        let seq = expected["seq"].as_u64().unwrap();
        let seq_frame = seq.to_be_bytes();
        let bytes = std::fs::read(fixture("prefill-state-push-recorded.msgpack")).unwrap();
        let msg = decode_frames(&[TOPIC, seq_frame.as_slice(), bytes.as_slice()]).unwrap();
        let p = &msg.payload;
        assert_eq!(p.epoch, expected["epoch"].as_str().unwrap());
        assert_eq!(p.trigger, "result");
        assert_eq!(p.sent_mono_s, expected["sent_mono_s"].as_f64());
        assert_eq!(p.snapshot_source.as_deref(), Some("v1_loads_capture"));
        assert_eq!(p.gpu_timing.as_deref(), Some("unavailable"));
        let event = |v: &Value| serde_json::from_value::<PassEvent>(v.clone()).unwrap();
        assert_eq!(p.events, vec![event(&expected["events"][0])]);
        assert_eq!(p.inflight, vec![event(&expected["inflight"][0])]);
        let mut snap = recorded_snapshot();
        snap["sequence"] = json!(seq);
        snap["age_s"] = expected["snapshot_age_s"].clone();
        snap["stale"] = expected["snapshot_stale"].clone();
        assert_eq!(p.snapshot.as_ref(), Some(&snap));
        let pushed = ForecastView::parse(p.snapshot.as_ref().unwrap());
        let polled = ForecastView::parse(&recorded_snapshot());
        assert_eq!(pushed.inflight, polled.inflight);
        assert_eq!(pushed.partial, polled.partial);
        assert_eq!(pushed.waiting, polled.waiting);
        assert_eq!(pushed.samples.len(), 64);
        assert!(!inflight_differs(&pushed, &p.inflight));
        let oversize = std::fs::read(fixture("prefill-state-push-oversize.msgpack")).unwrap();
        let msg = decode_frames(&[TOPIC, seq_frame.as_slice(), oversize.as_slice()]).unwrap();
        assert!(msg.payload.snapshot.is_none());
        assert_eq!(
            msg.payload.snapshot_omitted.as_deref(),
            Some("over_max_bytes")
        );
        assert_eq!(msg.payload.oversize_total, 1);
    }
}
