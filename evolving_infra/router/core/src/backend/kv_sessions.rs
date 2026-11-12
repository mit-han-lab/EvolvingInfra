// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! Versioned trajectory retention, independent of placement and cache hashes.
//!
//! The request path only updates bounded local metadata. One background task
//! batches engine status/release/end controls; only a complete, current target
//! registration authorizes retirement of obsolete P/D registrations. Missing replies
//! retain extra copies. Engine pins and physical KV ownership remain in SGLang.
use crate::agent_identity::RequestIdentity;
use futures::{stream, StreamExt};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::atomic::{AtomicU64, Ordering},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::Notify;

/// Sessions the registry holds at once: live branches, and ended ones kept
/// so that a late request of an ended branch is refused. At this size the
/// ended branches every engine has acknowledged leave, oldest scan first.
const MAX_SESSIONS: usize = 100_000;
/// At capacity, the registry is scanned for finished sessions at most this often.
const FINISHED_SCAN_PERIOD: Duration = Duration::from_secs(1);
const MAX_WORKERS_PER_SESSION: usize = 128;
const BATCH: usize = 128;

/// Engine registration identity. Input: controller identity/version; output: wire metadata.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct KvSession {
    pub session_id: String,
    pub version: u64,
}

struct Session {
    step: u64,
    version: u64,
    p: String,
    d: String,
    workers: HashMap<String, u64>,
    retire: HashMap<String, u64>,
    /// When each pending `retire` entry was queued. Read only by
    /// [`KvSessions::retire_backlog`]; a stale entry for a worker that left
    /// `retire` is ignored there.
    retire_since: HashMap<String, Instant>,
    /// Former prefill workers, now decode, whose registration of this version
    /// must outrank ordinary references until the next P is READY.
    protect: HashMap<String, u64>,
    /// Readiness is valid only for this observed target worker/cache generation.
    ready_boot: Option<String>,
    /// A role drain released a local registration, so even a retry needs a new version.
    needs_registration: bool,
    ended: bool,
    /// Late D: decodes that may have received the current version's tag in
    /// an ADMIT frame ([`KvSessions::note_candidate`]) and are not bound.
    /// A bind of this version releases them; a new version queues each one
    /// for release at its own version and clears the set. Empty without
    /// late D.
    maybe: HashSet<String>,
}
#[derive(Default)]
struct State {
    sessions: HashMap<String, Session>,
    queue: VecDeque<String>,
    queued: HashSet<String>,
    boots: HashMap<String, String>,
    /// Drained workers awaiting exact release/END acknowledgements. Admission
    /// stays closed until the role registry finishes the corresponding drain.
    clearing: HashMap<String, HashMap<String, u64>>,
    clear_turn: bool,
    /// The last scan for finished sessions at capacity.
    finished_scan: Option<Instant>,
}
impl State {
    /// At capacity: drop the ended sessions that no engine references any
    /// more (every END acknowledged, nothing queued). Their identities were
    /// refused as ended; after this a request with one of them would start a
    /// new session, which the protocol forbids the client to send.
    fn drop_finished(&mut self) {
        if self
            .finished_scan
            .is_some_and(|at| at.elapsed() < FINISHED_SCAN_PERIOD)
        {
            return;
        }
        self.finished_scan = Some(Instant::now());
        let queued = &self.queued;
        self.sessions.retain(|sid, s| {
            !(s.ended && s.workers.is_empty() && s.maybe.is_empty() && !queued.contains(sid))
        });
    }

    /// Input: owned session key. Output: at most one queued reconciliation entry.
    fn enqueue(&mut self, sid: &str) {
        if self.queued.insert(sid.to_owned()) {
            self.queue.push_back(sid.to_owned());
        }
    }
}

/// One backend-owned registry; no token arrays or cache hashes are retained here.
pub struct KvSessions {
    namespace: String,
    state: Mutex<State>,
    changed: Notify,
    stats: KvSessionStats,
}

/// Cumulative counters of one registry, for the router's
/// `KV_SESSION_ROUTER_STATS` line. Observation only: no counter changes a
/// registration, a control or a retirement.
#[derive(Debug, Default)]
pub struct KvSessionStats {
    /// Requests whose prefill-leg body got a `kv_session` tag.
    pub tagged: AtomicU64,
    /// Requests with an identity that took a non-PD dispatch (sent untagged).
    pub untagged_plain: AtomicU64,
    /// Failed registrations by reason; each such request was sent untagged.
    pub register_stale: AtomicU64,
    pub register_ended: AtomicU64,
    pub register_capacity: AtomicU64,
    pub register_clearing: AtomicU64,
    pub register_invalid: AtomicU64,
    pub register_other: AtomicU64,
    /// Controls put in a batch for the engines, by operation.
    pub status_sent: AtomicU64,
    pub release_sent: AtomicU64,
    pub end_sent: AtomicU64,
    /// `status` replies that proved the current prefill holds the full context.
    pub ready: AtomicU64,
    /// `release` replies with status `released`.
    pub release_acked: AtomicU64,
    /// `end` replies with status `ended`.
    pub end_acked: AtomicU64,
    /// Control batches without a usable reply (transport error, non-2xx,
    /// oversized or unparsable body). The work stays queued for a retry.
    pub control_errors: AtomicU64,
    /// Old registrations queued for retirement (a P or D change).
    pub retire_queued: AtomicU64,
    /// Queued retirements that an engine reply resolved (released, stale or ended).
    pub retire_released: AtomicU64,
    /// `POST /kv_sessions/end` requests accepted / refused (unknown session).
    pub end_requests: AtomicU64,
    pub end_unknown: AtomicU64,
}

/// Add one to a [`KvSessionStats`] counter.
pub fn inc(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

/// Pending retirements at one instant (see [`KvSessions::retire_backlog`]).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RetireBacklog {
    /// Sessions in the registry; ended ones stay as END tombstones.
    pub sessions: u64,
    /// Old registrations that wait for the new prefill's READY or for a release reply.
    pub pending: u64,
    /// Age of the oldest pending retirement.
    pub max_age: Duration,
}
#[derive(Clone, Debug, Serialize)]
struct Control {
    op: &'static str,
    session_id: String,
    version: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_boot_id: Option<String>,
}
#[derive(Deserialize)]
struct Reply {
    boot_id: String,
    results: Vec<Outcome>,
}
#[derive(Deserialize)]
struct Outcome {
    session_id: String,
    version: u64,
    status: String,
    #[serde(default)]
    ready_tokens: u64,
    #[serde(default)]
    required_tokens: u64,
}
impl KvSessions {
    /// Input: optional engine API/admin key file. Output: isolated registry and bounded background worker.
    pub fn spawn(key_file: Option<&std::path::Path>) -> Arc<Self> {
        let this = Arc::new(Self::new(uuid::Uuid::new_v4().to_string()));
        let weak = Arc::downgrade(&this);
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(path) = key_file {
            let key = std::fs::read_to_string(path).expect("read KV session API key file");
            let key = key.trim();
            assert!(
                !key.is_empty() && !key.chars().any(char::is_whitespace),
                "invalid KV session API key"
            );
            let mut bearer = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
                .expect("valid KV session API key header");
            bearer.set_sensitive(true);
            headers.insert(reqwest::header::AUTHORIZATION, bearer);
        }
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(2))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("KV control client");
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                loop {
                    let Some(owner) = weak.upgrade() else { break };
                    tokio::select! {
                        _ = owner.changed.notified() => {},
                        _ = tokio::time::sleep(Duration::from_millis(100)) => {},
                    }
                    let batches = owner.batches();
                    stream::iter(batches)
                        .map(|(worker, controls)| {
                            let client = client.clone();
                            async move {
                                let reply = async {
                                    let body = json!({"operations": controls});
                                    let response = client
                                        .post(format!(
                                            "{}/kv_session/control",
                                            worker.trim_end_matches('/')
                                        ))
                                        .json(&body)
                                        .send()
                                        .await?
                                        .error_for_status()?;
                                    // Bound untrusted engine replies independently of session-table capacity.
                                    let mut response = response;
                                    let mut bytes = Vec::new();
                                    while let Some(chunk) = response.chunk().await? {
                                        if bytes.len() + chunk.len() > 1_048_576 {
                                            return Ok(None);
                                        }
                                        bytes.extend_from_slice(&chunk);
                                    }
                                    Ok::<_, reqwest::Error>(
                                        serde_json::from_slice::<Reply>(&bytes).ok(),
                                    )
                                }
                                .await
                                .ok()
                                .flatten();
                                (worker, controls, reply)
                            }
                        })
                        .buffer_unordered(8)
                        .for_each(|(worker, controls, reply)| {
                            if let Some(reply) = reply {
                                owner.apply(&worker, &controls, reply);
                            } else {
                                inc(&owner.stats.control_errors);
                            }
                            owner.requeue(&controls);
                            std::future::ready(())
                        })
                        .await;
                    // Retries are bounded even while admission continuously notifies the task.
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            });
        }
        this
    }

    /// Input: unique controller namespace. Output: empty registry, without spawning IO.
    fn new(namespace: String) -> Self {
        Self {
            namespace,
            state: Mutex::default(),
            changed: Notify::new(),
            stats: KvSessionStats::default(),
        }
    }

    /// Input: none. Output: this registry's cumulative counters.
    pub fn stats(&self) -> &KvSessionStats {
        &self.stats
    }

    /// Input: none. Output: the session count, the pending retirements and
    /// the age of the oldest one, from one scan of the table under its lock.
    /// For the periodic stats line only, never on the request path.
    pub fn retire_backlog(&self) -> RetireBacklog {
        let now = Instant::now();
        let state = self.state.lock();
        let mut backlog = RetireBacklog {
            sessions: state.sessions.len() as u64,
            ..RetireBacklog::default()
        };
        for s in state.sessions.values() {
            for worker in s.retire.keys() {
                backlog.pending += 1;
                if let Some(since) = s.retire_since.get(worker) {
                    backlog.max_age = backlog.max_age.max(now.saturating_duration_since(*since));
                }
            }
        }
        backlog
    }

    /// Input: frontend trajectory identity. Output: bounded opaque lifetime key; no cache salt changes.
    fn sid(&self, identity: &RequestIdentity) -> String {
        let mut digest = Sha256::new();
        digest.update((identity.agent_id.len() as u64).to_be_bytes());
        digest.update(identity.agent_id.as_bytes());
        digest.update(identity.subagent_id.to_be_bytes());
        format!("{}:{:x}", self.namespace, digest.finalize())
    }

    /// Input: logical request and selected P/D. Output: stable retry metadata or a fail-closed error.
    /// New requests/placements advance the version; obsolete P/D placements queue retirement.
    pub fn register(
        &self,
        identity: &RequestIdentity,
        p: &str,
        d: &str,
    ) -> Result<KvSession, String> {
        self.register_inner(identity, p, Some(d))
    }

    /// Late D: register the P-leg only. The
    /// version advances on a new request, a P change or a forced
    /// re-registration, never on a decode change; the decode is bound later
    /// by [`Self::bind_decode`] (it receives the P-leg's tag in the ADMIT
    /// frame, never in its candidate body). The current decode home and its
    /// pending retirement are left as they are. A new version queues the
    /// release of every unbound possible ADMIT target of the old version
    /// ([`Self::note_candidate`]) at that version, behind the new P's READY.
    pub fn register_late_prefill(
        &self,
        identity: &RequestIdentity,
        p: &str,
    ) -> Result<KvSession, String> {
        self.register_inner(identity, p, None)
    }

    /// Input: identity, selected P and, outside late D, the selected D.
    /// Output: the tag; `d = None` leaves the decode side unchanged.
    fn register_inner(
        &self,
        identity: &RequestIdentity,
        p: &str,
        d: Option<&str>,
    ) -> Result<KvSession, String> {
        identity
            .engine_rid()
            .map_err(|_| "invalid trajectory identity")?;
        let sid = self.sid(identity);
        let mut state = self.state.lock();
        if state.clearing.contains_key(p) || d.is_some_and(|d| state.clearing.contains_key(d)) {
            return Err(
                "worker is clearing session registrations; replan after its role drain".into(),
            );
        }
        if !state.sessions.contains_key(&sid) && state.sessions.len() >= MAX_SESSIONS {
            state.drop_finished();
        }
        if !state.sessions.contains_key(&sid) && state.sessions.len() >= MAX_SESSIONS {
            return Err(
                "KV session registry at capacity: 100,000 branches are live or have an END no engine has acknowledged".into(),
            );
        }
        let s = state
            .sessions
            .entry(sid.clone())
            .or_insert_with(|| Session {
                step: identity.request_id,
                version: 1,
                p: p.into(),
                // Late D: no decode home until the first bind.
                d: d.unwrap_or_default().into(),
                workers: HashMap::new(),
                retire: HashMap::new(),
                retire_since: HashMap::new(),
                protect: HashMap::new(),
                ready_boot: None,
                needs_registration: false,
                ended: false,
                maybe: HashSet::new(),
            });
        if s.ended {
            return Err("trajectory has ended; use a new agent/subagent identity".into());
        }
        if identity.request_id < s.step {
            return Err("stale trajectory request".into());
        }
        let added = std::iter::once(p)
            .chain(d)
            .collect::<HashSet<_>>()
            .into_iter()
            .filter(|w| !s.workers.contains_key(*w))
            .count();
        if s.workers.len() + added > MAX_WORKERS_PER_SESSION {
            return Err("KV session worker history at capacity".into());
        }
        // Stats only: the workers this call newly queues for retirement.
        let mut newly_retired: Vec<String> = Vec::new();
        let d_changed = d.is_some_and(|d| s.d != d);
        if identity.request_id != s.step || s.p != p || d_changed || s.needs_registration {
            let version = s
                .version
                .checked_add(1)
                .filter(|v| *v <= i64::MAX as u64)
                .ok_or("KV session version exhausted")?;
            if s.p != p {
                if let Some(old) = s.workers.get(&s.p) {
                    if s.retire.insert(s.p.clone(), *old).is_none() {
                        newly_retired.push(s.p.clone());
                    }
                }
            }
            if d_changed {
                if let Some(old) = s.workers.get(&s.d) {
                    if s.retire.insert(s.d.clone(), *old).is_none() {
                        newly_retired.push(s.d.clone());
                    }
                }
            }
            // Late D: a possible ADMIT target of
            // the old version that never bound (its leg failed after ADMIT,
            // or the router never saw its 2xx) may hold a live registration
            // at that version. Queue its release at that exact version
            // behind the new P's READY, as for an old home. The new P and D
            // keep theirs (removed from `retire` below); the current home is
            // never in `maybe`. Empty without late D.
            for w in std::mem::take(&mut s.maybe) {
                if w == p || d == Some(w.as_str()) || w == s.d {
                    continue;
                }
                if let Some(old) = s.workers.get(&w) {
                    if s.retire.insert(w.clone(), *old).is_none() {
                        newly_retired.push(w);
                    }
                }
            }
            s.version = version;
            s.step = identity.request_id;
            s.p = p.into();
            if let Some(d) = d {
                s.d = d.into();
            }
            s.ready_boot = None;
            s.needs_registration = false;
        }
        s.workers.insert(p.into(), s.version);
        if let Some(d) = d {
            s.workers.insert(d.into(), s.version);
        }
        // Either current role gets a newer registration. Returning workers and
        // P/D role crossovers invalidate their older pending retirements.
        s.retire.remove(p);
        if let Some(d) = d {
            s.retire.remove(d);
        }
        for worker in newly_retired {
            if s.retire.contains_key(&worker) {
                s.retire_since.insert(worker, Instant::now());
                inc(&self.stats.retire_queued);
            }
        }
        let version = s.version;
        let pending = !s.retire.is_empty();
        if pending {
            state.enqueue(&sid);
        }
        drop(state);
        if pending {
            self.changed.notify_one();
        }
        Ok(KvSession {
            session_id: sid,
            version,
        })
    }

    /// Late D: `worker` may receive `tag` in an ADMIT frame (the arrival
    /// pick, or a decided winner). Record it in the session history before
    /// that can happen, so END and the retirement of an unused copy reach
    /// it. A tag of an older version, or of an ended
    /// session, is recorded too (END must reach it).
    pub fn note_candidate(&self, tag: &KvSession, worker: &str) {
        let mut state = self.state.lock();
        let Some(s) = state.sessions.get_mut(&tag.session_id) else {
            return;
        };
        let known = s.workers.get(worker).copied();
        if known.is_some_and(|v| v >= tag.version) {
            if s.version == tag.version && s.d != worker && s.p != worker {
                s.maybe.insert(worker.to_string());
            }
            return;
        }
        if known.is_none() && s.workers.len() >= MAX_WORKERS_PER_SESSION {
            return;
        }
        s.workers.insert(worker.to_string(), tag.version);
        if s.version == tag.version && s.d != worker && s.p != worker {
            s.maybe.insert(worker.to_string());
        }
        if s.ended {
            state.enqueue(&tag.session_id);
            drop(state);
            self.changed.notify_one();
        }
    }

    /// Late D: the decode `worker` took the stream of the request tagged
    /// `tag`, so P admitted it with that tag. It becomes the decode home
    /// (one decode copy per agent): the previous home and every other
    /// possible ADMIT target of this version are released once the current
    /// P reports the full context ready (the same gate as a P change). A
    /// bind of an older version only records the worker and queues its own
    /// release; a bind on an ended session queues END to it.
    pub fn bind_decode(&self, tag: &KvSession, worker: &str) {
        let mut state = self.state.lock();
        let Some(s) = state.sessions.get_mut(&tag.session_id) else {
            return;
        };
        let known = s.workers.get(worker).copied();
        if known.is_none() && s.workers.len() >= MAX_WORKERS_PER_SESSION {
            return;
        }
        if known.is_none_or(|v| v < tag.version) {
            s.workers.insert(worker.to_string(), tag.version);
        }
        let mut newly: Vec<String> = Vec::new();
        if s.ended {
            // END reaches every recorded worker, this one included.
        } else if s.version != tag.version {
            // A newer request already registered: this copy is not the home.
            if worker != s.p
                && worker != s.d
                && s.workers.get(worker) == Some(&tag.version)
                && s.retire.insert(worker.to_string(), tag.version).is_none()
            {
                newly.push(worker.to_string());
            }
        } else {
            let old = std::mem::replace(&mut s.d, worker.to_string());
            s.retire.remove(worker);
            s.maybe.remove(worker);
            let mut release: Vec<String> = s.maybe.drain().collect();
            if !old.is_empty() && old != worker {
                release.push(old);
            }
            for w in release {
                if w == s.p || w == s.d {
                    continue;
                }
                if let Some(v) = s.workers.get(&w).copied() {
                    if s.retire.insert(w.clone(), v).is_none() {
                        newly.push(w);
                    }
                }
            }
        }
        for w in newly {
            s.retire_since.insert(w, Instant::now());
            inc(&self.stats.retire_queued);
        }
        let pending = s.ended || !s.retire.is_empty();
        if pending {
            state.enqueue(&tag.session_id);
        }
        drop(state);
        if pending {
            self.changed.notify_one();
        }
    }

    /// Input: existing frontend branch identity. Output: END queued once for every recorded engine.
    /// Request number is ignored; END closes the lifetime, not one step. Unknown IDs are rejected.
    pub fn end(&self, identity: &RequestIdentity) -> Result<(), String> {
        let sid = self.sid(identity);
        let mut state = self.state.lock();
        let s = state.sessions.get_mut(&sid).ok_or("unknown KV session")?;
        s.ended = true;
        s.retire.clear();
        s.protect.clear();
        state.enqueue(&sid);
        drop(state);
        self.changed.notify_one();
        Ok(())
    }

    /// Called once under the role registry's drain lock after all old legs end.
    /// Reuse versioned releases: clear local references, never physical KV or LRU.
    /// The one table scan is off the admission/transfer paths.
    pub(crate) fn begin_worker_clear(&self, worker: &str) {
        let mut state = self.state.lock();
        if state.clearing.contains_key(worker) {
            return;
        }
        let pending = state
            .sessions
            .iter()
            .filter_map(|(sid, s)| s.workers.get(worker).map(|version| (sid.clone(), *version)))
            .collect();
        state.clearing.insert(worker.into(), pending);
        drop(state);
        self.changed.notify_one();
    }

    /// Called once when `worker` changes from prefill to decode. Sessions whose
    /// current P is `worker` get a versioned PROTECT there, so their prefix
    /// outranks new decode imports until the ordinary release after the next
    /// P is READY. Never moves KV; the one table scan is off the request path.
    pub(crate) fn begin_worker_handover(&self, worker: &str) {
        let mut state = self.state.lock();
        let mut queued = Vec::new();
        for (sid, s) in state.sessions.iter_mut() {
            if s.ended || s.p != worker {
                continue;
            }
            if let Some(version) = s.workers.get(worker) {
                s.protect.insert(worker.into(), *version);
                queued.push(sid.clone());
            }
        }
        for sid in &queued {
            state.enqueue(sid);
        }
        drop(state);
        if !queued.is_empty() {
            self.changed.notify_one();
        }
    }

    /// Remove the admission fence only when every recorded registration was
    /// acknowledged. The role registry calls this before exposing the new P.
    pub(crate) fn finish_worker_clear(&self, worker: &str) -> bool {
        let mut state = self.state.lock();
        if !state.clearing.get(worker).is_some_and(HashMap::is_empty) {
            return false;
        }
        state.clearing.remove(worker);
        true
    }

    /// Input: local pending work. Output: bounded per-worker batches, without touching request execution.
    fn batches(&self) -> Vec<(String, Vec<Control>)> {
        let mut state = self.state.lock();
        let mut batches: HashMap<String, Vec<Control>> = HashMap::new();
        // Bound the number of sessions, operations and simultaneous workers each cycle.
        let mut remaining = BATCH;
        // Alternate full budgets: ordinary END may need all 128 operations for
        // one session, and must keep progressing even if a clear cannot finish.
        state.clear_turn = !state.clear_turn;
        let per_worker = (BATCH / state.clearing.len().max(1)).max(1);
        for (worker, pending) in state.clearing.iter().filter(|_| state.clear_turn) {
            for (sid, version) in pending.iter().take(per_worker.min(remaining)) {
                let op = if !state.boots.contains_key(worker) {
                    "status"
                } else if state.sessions[sid].ended {
                    "end"
                } else {
                    "release"
                };
                batches.entry(worker.clone()).or_default().push(Control {
                    op,
                    session_id: sid.clone(),
                    version: *version,
                    expected_boot_id: (op != "status").then(|| state.boots[worker].clone()),
                });
                remaining -= 1;
            }
        }
        for _ in 0..state.queue.len().min(BATCH) {
            let sid = state.queue.pop_front().unwrap();
            state.queued.remove(&sid);
            // A pending release of the same worker supersedes its protection;
            // sending both for one (session, version) would make the reply ambiguous.
            let s = state.sessions.get_mut(&sid).unwrap();
            let Session {
                retire, protect, ..
            } = s;
            protect.retain(|worker, _| !retire.contains_key(worker));
            let s = &state.sessions[&sid];
            let mut commands = Vec::new();
            if s.ended {
                for (worker, version) in &s.workers {
                    commands.push((worker.clone(), "end", *version));
                }
            } else if !s.retire.is_empty() {
                if s.ready_boot
                    .as_ref()
                    .is_some_and(|boot| state.boots.get(&s.p) == Some(boot))
                {
                    for (worker, version) in &s.retire {
                        commands.push((worker.clone(), "release", *version));
                    }
                } else {
                    commands.push((s.p.clone(), "status", s.version));
                }
            }
            if !s.ended {
                for (worker, version) in &s.protect {
                    commands.push((worker.clone(), "protect", *version));
                }
            }
            // Clearing uses the same RPC but overrides the ordinary READY gate.
            // Never send the same (session, version) twice in one worker batch.
            commands.retain(|(worker, _, _)| !state.clearing.contains_key(worker));
            if commands.is_empty()
                && ((s.ended && !s.workers.is_empty())
                    || !s.retire.is_empty()
                    || !s.protect.is_empty())
            {
                // No sent control will requeue this session while its current P
                // is clearing. Keep its ordinary work reachable after the drain.
                state.enqueue(&sid);
                continue;
            }
            if commands.len() > remaining {
                // Keep the unsent group at the front. Moving it behind the
                // just-sent retries can starve it forever when batch sizes do
                // not divide the per-session control count (for example 128/3).
                state.queued.insert(sid.clone());
                state.queue.push_front(sid);
                break;
            }
            remaining -= commands.len();
            for (worker, op, version) in commands {
                let op = if op != "status" && !state.boots.contains_key(&worker) {
                    "status"
                } else {
                    op
                };
                let expected_boot_id = (op != "status").then(|| state.boots[&worker].clone());
                batches.entry(worker).or_default().push(Control {
                    op,
                    session_id: sid.clone(),
                    version,
                    expected_boot_id,
                });
            }
        }
        drop(state);
        for control in batches.values().flatten() {
            match control.op {
                "status" => inc(&self.stats.status_sent),
                "release" => inc(&self.stats.release_sent),
                "end" => inc(&self.stats.end_sent),
                _ => {}
            }
        }
        batches.into_iter().collect()
    }

    /// Input: exact sent controls and one response. Output: apply only still-current, boot-valid outcomes.
    fn apply(&self, worker: &str, controls: &[Control], reply: Reply) {
        if reply.boot_id.is_empty() || reply.boot_id.len() > 256 {
            return;
        }
        let mut state = self.state.lock();
        let changed_boot = state
            .boots
            .insert(worker.into(), reply.boot_id.clone())
            .is_some_and(|old| old != reply.boot_id);
        // A changed service/cache generation invalidates the complete reply,
        // including release/end outcomes. Reconciliation retries against it.
        if changed_boot {
            return;
        }
        for control in controls {
            if control
                .expected_boot_id
                .as_ref()
                .is_some_and(|boot| boot != &reply.boot_id)
            {
                continue;
            }
            let mut matching = reply
                .results
                .iter()
                .filter(|r| r.session_id == control.session_id && r.version == control.version);
            let Some(result) = matching.next() else {
                continue;
            };
            if matching.next().is_some() {
                continue;
            } // Ambiguous replies are not evidence.
            match (control.op, result.status.as_str()) {
                ("release", "released") => inc(&self.stats.release_acked),
                ("end", "ended") => inc(&self.stats.end_acked),
                _ => {}
            }
            let clearing = state
                .clearing
                .get(worker)
                .is_some_and(|pending| pending.get(&control.session_id) == Some(&control.version));
            let Some(s) = state.sessions.get_mut(&control.session_id) else {
                continue;
            };
            if clearing {
                let cleared = (control.op == "end" && result.status == "ended")
                    || (!s.ended
                        && control.op == "release"
                        && matches!(result.status.as_str(), "released" | "ended"));
                if cleared {
                    // Keep worker history so a later END still closes its lifetime.
                    if s.retire.get(worker) == Some(&control.version) {
                        s.retire.remove(worker);
                    }
                    s.needs_registration |= s.version == control.version;
                    if s.p == worker {
                        s.ready_boot = None;
                    }
                    if s.ended {
                        s.workers.remove(worker);
                    }
                    state
                        .clearing
                        .get_mut(worker)
                        .unwrap()
                        .remove(&control.session_id);
                }
                // Missing/stale/unknown and partial TP replies never prove zero.
                continue;
            }
            match control.op {
                "status" if !s.ended && s.p == worker && s.version == control.version => {
                    s.ready_boot = (result.status == "ready"
                        && result.required_tokens > 0
                        && result.ready_tokens == result.required_tokens)
                        .then(|| reply.boot_id.clone());
                    if s.ready_boot.is_some() {
                        inc(&self.stats.ready);
                    }
                }
                "release" if !s.ended && s.retire.get(worker) == Some(&control.version) => {
                    if matches!(result.status.as_str(), "released" | "stale" | "ended") {
                        s.retire.remove(worker);
                        inc(&self.stats.retire_released);
                    }
                }
                "protect" if s.protect.get(worker) == Some(&control.version) => {
                    // Any definite state ends the request: protected now, or the
                    // registration was already replaced/retired so nothing remains.
                    if matches!(
                        result.status.as_str(),
                        "protected" | "released" | "stale" | "ended" | "missing"
                    ) {
                        s.protect.remove(worker);
                    }
                }
                "end" if s.ended && s.workers.get(worker) == Some(&control.version) => {
                    // A missing session is not an END tombstone and cannot fence late admission.
                    if result.status == "ended" {
                        s.workers.remove(worker);
                    }
                }
                _ => {}
            }
        }
    }

    /// Input: attempted controls. Output: retry only unresolved current work, including transport failure.
    fn requeue(&self, controls: &[Control]) {
        let mut state = self.state.lock();
        for control in controls {
            if state.sessions.get(&control.session_id).is_some_and(|s| {
                (s.ended && !s.workers.is_empty())
                    || (!s.ended && (!s.retire.is_empty() || !s.protect.is_empty()))
            }) {
                state.enqueue(&control.session_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn identity(step: u64) -> RequestIdentity {
        RequestIdentity {
            agent_id: "a".into(),
            subagent_id: 2,
            request_id: step,
        }
    }
    fn ready(tag: &KvSession, boot: &str, tokens: u64) -> Reply {
        Reply {
            boot_id: boot.into(),
            results: vec![Outcome {
                session_id: tag.session_id.clone(),
                version: tag.version,
                status: "ready".into(),
                ready_tokens: tokens,
                required_tokens: 32,
            }],
        }
    }
    #[tokio::test]
    async fn http_control_batches_wait_for_ready_and_use_boot_fences() {
        use axum::{
            extract::{Path, State},
            http::HeaderMap,
            routing::post,
            Json, Router,
        };
        use std::sync::atomic::{AtomicBool, Ordering};
        #[derive(Clone)]
        struct Mock {
            ready: Arc<AtomicBool>,
            seen: Arc<Mutex<Vec<(String, serde_json::Value)>>>,
        }
        async fn control(
            Path(worker): Path<String>,
            State(state): State<Mock>,
            headers: HeaderMap,
            Json(body): Json<serde_json::Value>,
        ) -> Json<serde_json::Value> {
            assert_eq!(headers["authorization"], "Bearer test-retention-key");
            assert!(body.get("expected_boot_id").is_none());
            state.seen.lock().push((worker.clone(), body.clone()));
            let boot = format!("boot-{worker}");
            let results: Vec<_> = body["operations"]
                .as_array()
                .unwrap()
                .iter()
                .map(|op| {
                    let status = match op["op"].as_str().unwrap() {
                        "status" => {
                            assert!(op.get("expected_boot_id").is_none());
                            if state.ready.load(Ordering::SeqCst) {
                                "ready"
                            } else {
                                "pending"
                            }
                        }
                        "release" => {
                            assert_eq!(op["expected_boot_id"], boot);
                            "released"
                        }
                        "end" => {
                            assert_eq!(op["expected_boot_id"], boot);
                            "ended"
                        }
                        _ => panic!("unexpected control"),
                    };
                    json!({"session_id":op["session_id"],"version":op["version"],"status":status,
                       "ready_tokens":if status == "ready" {32} else {0},"required_tokens":32})
                })
                .collect();
            Json(json!({"boot_id":boot,"results":results}))
        }
        let state = Mock {
            ready: Arc::new(AtomicBool::new(false)),
            seen: Arc::new(Mutex::new(Vec::new())),
        };
        let app = Router::new()
            .route("/{worker}/kv_session/control", post(control))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let key = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(key.path(), "test-retention-key").unwrap();
        let sessions = KvSessions::spawn(Some(key.path()));
        sessions
            .register(&identity(0), &format!("{base}/p1"), &format!("{base}/d"))
            .unwrap();
        sessions
            .register(&identity(1), &format!("{base}/p2"), &format!("{base}/d"))
            .unwrap();
        tokio::time::sleep(Duration::from_millis(350)).await;
        assert!(!state.seen.lock().is_empty());
        assert!(state.seen.lock().iter().all(|(_, b)| b["operations"]
            .as_array()
            .unwrap()
            .iter()
            .all(|o| o["op"] == "status")));
        state.ready.store(true, Ordering::SeqCst);
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if state.seen.lock().iter().any(|(w, b)| {
                    w == "p1"
                        && b["operations"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|o| o["op"] == "release")
                }) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert!(state.seen.lock().iter().all(|(w, b)| w != "d"
            || b["operations"]
                .as_array()
                .unwrap()
                .iter()
                .all(|o| o["op"] != "release")));
        sessions.end(&identity(1)).unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if sessions
                    .state
                    .lock()
                    .sessions
                    .values()
                    .all(|s| s.workers.is_empty())
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        server.abort();
    }

    #[test]
    fn retry_and_new_step_versions_are_distinct() {
        let owner = KvSessions::new("controller".into());
        let one = owner.register(&identity(0), "p1", "d").unwrap();
        assert_eq!(one, owner.register(&identity(0), "p1", "d").unwrap());
        assert_eq!(owner.register(&identity(1), "p1", "d").unwrap().version, 2);
        assert!(owner.register(&identity(0), "p1", "d").is_err());
        assert!(owner.batches().is_empty());
    }
    #[test]
    fn migration_waits_for_exact_full_readiness() {
        let owner = KvSessions::new("controller".into());
        owner.register(&identity(0), "p1", "d").unwrap();
        owner
            .state
            .lock()
            .boots
            .insert("p1".into(), "sourceboot".into());
        let next = owner.register(&identity(1), "p2", "d").unwrap();
        let (worker, controls) = owner.batches().pop().unwrap();
        assert_eq!(worker, "p2");
        assert_eq!(controls[0].op, "status");
        owner.apply(&worker, &controls, ready(&next, "boot1", 16));
        owner.requeue(&controls);
        let (_, again) = owner.batches().pop().unwrap();
        assert_eq!(again[0].op, "status");
        owner.apply(&worker, &again, ready(&next, "boot1", 32));
        owner.requeue(&again);
        let (worker, release) = owner.batches().pop().unwrap();
        assert_eq!(worker, "p1");
        assert_eq!(release[0].op, "release");
        assert_eq!(release[0].version, 1);
    }

    #[test]
    fn decode_migration_waits_for_current_prefill_context() {
        let owner = KvSessions::new("controller".into());
        let old = owner.register(&identity(0), "p0", "d0").unwrap();
        owner
            .state
            .lock()
            .boots
            .insert("d0".into(), "d0-boot".into());
        let current = owner.register(&identity(1), "p0", "d1").unwrap();
        let (worker, controls) = owner.batches().pop().unwrap();
        assert_eq!(worker, "p0", "D retirement checks the current P context");
        assert_eq!(controls[0].op, "status");
        assert_eq!(controls[0].version, current.version);

        // An older READY from the unchanged P does not cover the new input.
        owner.apply(&worker, &controls, ready(&old, "p0-boot", 32));
        owner.requeue(&controls);
        let (_, controls) = owner.batches().pop().unwrap();
        assert_eq!(controls[0].op, "status");
        owner.apply(&worker, &controls, ready(&current, "p0-boot", 16));
        owner.requeue(&controls);
        let (_, controls) = owner.batches().pop().unwrap();
        assert_eq!(controls[0].op, "status");
        owner.apply(&worker, &controls, ready(&current, "p0-boot", 32));
        owner.requeue(&controls);

        let batches = owner.batches();
        assert_eq!(batches.len(), 1);
        let (worker, controls) = &batches[0];
        assert_eq!(worker, "d0");
        assert_eq!(controls[0].op, "release");
        assert_eq!(controls[0].version, old.version);
        assert_eq!(controls[0].expected_boot_id.as_deref(), Some("d0-boot"));
    }

    #[test]
    fn compacted_context_requires_new_readiness_before_old_decode_retirement() {
        let owner = KvSessions::new("controller".into());
        owner.register(&identity(0), "p0", "d0").unwrap();
        owner
            .state
            .lock()
            .boots
            .insert("d0".into(), "d0-boot".into());
        let previous = owner.register(&identity(1), "p0", "d1").unwrap();
        let (worker, controls) = owner.batches().pop().unwrap();
        owner.apply(&worker, &controls, ready(&previous, "p0-boot", 32));
        owner.requeue(&controls);

        // The next step can compact or rewrite its input. Only the engine knows
        // its required range; the controller must discard the earlier proof.
        let current = owner.register(&identity(2), "p0", "d1").unwrap();
        let (worker, controls) = owner.batches().pop().unwrap();
        assert_eq!(worker, "p0");
        assert_eq!(controls[0].op, "status");
        assert_eq!(controls[0].version, current.version);
        let mut partial = ready(&current, "p0-boot", 4);
        partial.results[0].required_tokens = 8;
        owner.apply(&worker, &controls, partial);
        owner.requeue(&controls);
        let (_, controls) = owner.batches().pop().unwrap();
        assert_eq!(controls[0].op, "status");
        let mut complete = ready(&current, "p0-boot", 8);
        complete.results[0].required_tokens = 8;
        owner.apply(&worker, &controls, complete);
        owner.requeue(&controls);
        let (worker, controls) = owner.batches().pop().unwrap();
        assert_eq!(worker, "d0");
        assert_eq!(controls[0].op, "release");
        assert_eq!(controls[0].version, 1);
    }

    #[test]
    fn placement_role_crossovers_preserve_both_current_workers() {
        for (p, d, expected_retired) in [
            ("p1", "d1", vec!["p0", "d0"]),
            ("p0", "d1", vec!["d0"]),
            ("d0", "d1", vec!["p0"]),
            ("p1", "p0", vec!["d0"]),
            ("d0", "p0", vec![]),
            ("p0", "d0", vec![]),
            ("p0", "p0", vec!["d0"]),
            ("d0", "d0", vec!["p0"]),
            ("p1", "p1", vec!["p0", "d0"]),
        ] {
            let owner = KvSessions::new("controller".into());
            owner.register(&identity(0), "p0", "d0").unwrap();
            let current = owner.register(&identity(1), p, d).unwrap();
            {
                let state = owner.state.lock();
                let session = &state.sessions[&current.session_id];
                assert_eq!(
                    session.retire,
                    expected_retired
                        .iter()
                        .map(|w| (w.to_string(), 1))
                        .collect(),
                    "current P={p}, D={d}"
                );
                assert_eq!(session.workers[p], current.version);
                assert_eq!(session.workers[d], current.version);
            }
            let batches = owner.batches();
            if expected_retired.is_empty() {
                assert!(batches.is_empty());
                continue;
            }
            assert_eq!(batches.len(), 1);
            let (worker, controls) = &batches[0];
            assert_eq!(worker, p);
            assert_eq!(controls[0].op, "status");
            owner.apply(worker, controls, ready(&current, "current-boot", 32));
            owner.requeue(controls);
            owner.state.lock().boots.extend(
                expected_retired
                    .iter()
                    .map(|w| (w.to_string(), format!("{w}-boot"))),
            );
            let releases = owner.batches();
            assert_eq!(releases.len(), expected_retired.len());
            assert!(releases.iter().all(|(worker, controls)| {
                worker != p && worker != d && controls[0].op == "release"
            }));
        }
    }

    #[test]
    fn returning_decode_worker_ignores_older_release_ack() {
        let owner = KvSessions::new("controller".into());
        let old = owner.register(&identity(0), "p0", "d0").unwrap();
        owner
            .state
            .lock()
            .boots
            .insert("d0".into(), "d0-boot".into());
        let second = owner.register(&identity(1), "p0", "d1").unwrap();
        let (worker, controls) = owner.batches().pop().unwrap();
        owner.apply(&worker, &controls, ready(&second, "p0-boot", 32));
        owner.requeue(&controls);
        let (worker, old_release) = owner.batches().pop().unwrap();
        assert_eq!(worker, "d0");
        assert_eq!(old_release[0].op, "release");
        let returned = owner.register(&identity(2), "p0", "d0").unwrap();
        assert!(!owner.state.lock().sessions[&returned.session_id]
            .retire
            .contains_key("d0"));
        owner.register(&identity(3), "p0", "d2").unwrap();
        let mut ack = ready(&old, "d0-boot", 0);
        ack.results[0].status = "released".into();
        owner.apply(&worker, &old_release, ack);
        assert_eq!(
            owner.state.lock().sessions[&returned.session_id]
                .retire
                .get("d0"),
            Some(&returned.version),
            "the old ACK must not discard retirement of the newer D registration"
        );
    }

    #[test]
    fn late_ready_and_engine_restart_cannot_retire_new_placement() {
        let owner = KvSessions::new("controller".into());
        owner.register(&identity(0), "p1", "d").unwrap();
        let old = owner.register(&identity(1), "p2", "d").unwrap();
        let (worker, controls) = owner.batches().pop().unwrap();
        owner.register(&identity(2), "p1", "d").unwrap();
        owner.apply(&worker, &controls, ready(&old, "boot1", 32));
        owner.requeue(&controls);
        assert!(owner
            .batches()
            .iter()
            .all(|(_, c)| c.iter().all(|c| c.op == "status")));
        let new = owner.register(&identity(3), "p2", "d").unwrap();
        let (worker, controls) = owner.batches().pop().unwrap();
        owner.apply(&worker, &controls, ready(&new, "boot2", 32));
        owner.requeue(&controls);
        assert!(owner
            .batches()
            .iter()
            .all(|(_, c)| c.iter().all(|c| c.op == "status")));
    }
    #[test]
    fn failed_multi_worker_groups_do_not_starve_at_batch_boundary() {
        let owner = KvSessions::new("controller".into());
        for i in 0..85 {
            let mut id = identity(0);
            id.agent_id = format!("trajectory-{i}");
            owner.register(&id, "p1", "d").unwrap();
            id.request_id = 1;
            owner.register(&id, "p2", "d").unwrap();
            owner.end(&id).unwrap();
        }
        owner.state.lock().boots.extend([
            ("p1".into(), "p1-boot".into()),
            ("p2".into(), "p2-boot".into()),
            ("d".into(), "d-boot".into()),
        ]);
        let mut seen = HashSet::new();
        for _ in 0..3 {
            let batches = owner.batches();
            assert!(batches.iter().map(|(_, ops)| ops.len()).sum::<usize>() <= BATCH);
            let mut counts = HashMap::new();
            for (_, controls) in &batches {
                for op in controls {
                    assert_eq!(op.op, "end");
                    seen.insert(op.session_id.clone());
                    *counts.entry(op.session_id.clone()).or_insert(0) += 1;
                }
                // Model complete transport failure: every sent operation stays
                // pending and returns to the queue behind untouched work.
                owner.requeue(controls);
            }
            assert!(counts.values().all(|&n| n == 3));
        }
        assert_eq!(
            seen.len(),
            85,
            "every group must be attempted within three cycles"
        );
    }

    #[test]
    fn another_sessions_restart_observation_invalidates_cached_ready() {
        let owner = KvSessions::new("controller".into());
        owner.register(&identity(0), "p1", "d").unwrap();
        let a = owner.register(&identity(1), "p2", "d").unwrap();
        let (worker, controls) = owner.batches().pop().unwrap();
        owner.apply(&worker, &controls, ready(&a, "old-p2-boot", 32));
        owner.requeue(&controls);
        owner
            .state
            .lock()
            .boots
            .insert("p1".into(), "p1-boot".into());
        let mut other = identity(0);
        other.agent_id = "another-trajectory".into();
        owner.register(&other, "p3", "d").unwrap();
        other.request_id = 1;
        let b = owner.register(&other, "p2", "d").unwrap();
        let probe = Control {
            op: "status",
            session_id: b.session_id.clone(),
            version: b.version,
            expected_boot_id: None,
        };
        owner.apply("p2", &[probe], ready(&b, "new-p2-boot", 32));
        let batches = owner.batches();
        let a_ops: Vec<_> = batches
            .iter()
            .flat_map(|(worker, ops)| {
                ops.iter()
                    .filter(|op| op.session_id == a.session_id)
                    .map(move |op| (worker, op))
            })
            .collect();
        assert_eq!(a_ops.len(), 1);
        assert_eq!(a_ops[0].0, "p2");
        assert_eq!(
            a_ops[0].1.op, "status",
            "old READY must not authorize source retirement"
        );
        for (worker, controls) in &batches {
            owner.apply(worker, controls, ready(&a, "new-p2-boot", 32));
            owner.requeue(controls);
        }
        assert!(owner.batches().iter().any(|(worker, ops)| worker == "p1"
            && ops
                .iter()
                .any(|op| op.session_id == a.session_id && op.op == "release")));
    }

    #[test]
    fn missing_release_is_not_a_tombstone_and_boot_mismatch_is_not_success() {
        let owner = KvSessions::new("controller".into());
        let first = owner.register(&identity(0), "p1", "d").unwrap();
        owner.register(&identity(1), "p2", "d").unwrap();
        owner.state.lock().boots.insert("p1".into(), "boot1".into());
        let control = Control {
            op: "release",
            session_id: first.session_id.clone(),
            version: 1,
            expected_boot_id: Some("boot1".into()),
        };
        let mut reply = ready(&first, "boot1", 0);
        reply.results[0].status = "missing".into();
        owner.apply("p1", std::slice::from_ref(&control), reply);
        assert_eq!(
            owner.state.lock().sessions[&first.session_id]
                .retire
                .get("p1"),
            Some(&1)
        );
        let mut reply = ready(&first, "newboot", 0);
        reply.results[0].status = "released".into();
        owner.apply("p1", std::slice::from_ref(&control), reply);
        assert_eq!(
            owner.state.lock().sessions[&first.session_id]
                .retire
                .get("p1"),
            Some(&1)
        );
        let mut reply = ready(&first, "newboot", 0);
        reply.results[0].status = "released".into();
        let current = Control {
            expected_boot_id: Some("newboot".into()),
            ..control
        };
        owner.apply("p1", &[current], reply);
        assert!(!owner.state.lock().sessions[&first.session_id]
            .retire
            .contains_key("p1"));
    }

    #[test]
    fn end_is_explicit_idempotent_and_fences_reuse() {
        let owner = KvSessions::new("controller".into());
        owner.register(&identity(0), "p1", "d").unwrap();
        owner
            .state
            .lock()
            .boots
            .extend([("p1".into(), "pboot".into()), ("d".into(), "dboot".into())]);
        owner.end(&identity(0)).unwrap();
        owner.end(&identity(0)).unwrap();
        assert!(owner.register(&identity(1), "p1", "d").is_err());
        let batches = owner.batches();
        assert_eq!(batches.len(), 2);
        assert!(batches.iter().all(|(_, c)| c[0].op == "end"));
        let other = KvSessions::new("new-controller".into());
        assert_ne!(owner.sid(&identity(0)), other.sid(&identity(0)));
    }

    fn clear_reply(controls: &[Control], boot: &str, status: &str) -> Reply {
        Reply {
            boot_id: boot.into(),
            results: controls
                .iter()
                .map(|c| Outcome {
                    session_id: c.session_id.clone(),
                    version: c.version,
                    status: status.into(),
                    ready_tokens: 0,
                    required_tokens: 32,
                })
                .collect(),
        }
    }

    fn next_worker_controls(owner: &KvSessions, target: &str) -> Vec<Control> {
        // Clear and normal work alternate budgets. Retain unrelated work as
        // the background task does after an unsuccessful transport attempt.
        for _ in 0..4 {
            let mut found = None;
            for (worker, controls) in owner.batches() {
                if worker == target {
                    found = Some(controls);
                } else {
                    owner.requeue(&controls);
                }
            }
            if let Some(controls) = found {
                return controls;
            }
        }
        panic!("no pending controls for {target}");
    }

    #[test]
    fn worker_clear_releases_without_ready_and_advances_retry_version_once() {
        let owner = KvSessions::new("controller".into());
        let original = owner.register(&identity(0), "p", "d").unwrap();
        owner.begin_worker_clear("d");
        owner.begin_worker_clear("d");
        assert!(owner.register(&identity(0), "p", "d").is_err());
        assert!(!owner.finish_worker_clear("d"));

        let probe = next_worker_controls(&owner, "d");
        assert_eq!(probe.len(), 1, "repeated begin must not duplicate releases");
        assert_eq!(probe[0].op, "status");
        assert!(probe[0].expected_boot_id.is_none());
        owner.apply("d", &probe, clear_reply(&probe, "d-boot", "pending"));
        let releases = next_worker_controls(&owner, "d");
        assert_eq!(releases[0].op, "release");
        assert_eq!(releases[0].version, original.version);
        assert_eq!(releases[0].expected_boot_id.as_deref(), Some("d-boot"));
        assert!(owner.state.lock().sessions[&original.session_id]
            .ready_boot
            .is_none());
        owner.apply("d", &releases, clear_reply(&releases, "d-boot", "released"));
        owner.apply("d", &releases, clear_reply(&releases, "d-boot", "released"));
        assert!(owner.finish_worker_clear("d"));
        assert!(!owner.finish_worker_clear("d"));
        assert!(
            owner.state.lock().sessions[&original.session_id]
                .workers
                .contains_key("d"),
            "END must still reach the old worker's lifetime fence"
        );

        let renewed = owner.register(&identity(0), "p", "d").unwrap();
        assert_eq!(renewed.version, original.version + 1);
        owner.apply("d", &releases, clear_reply(&releases, "d-boot", "released"));
        assert_eq!(owner.register(&identity(0), "p", "d").unwrap(), renewed);
    }

    #[test]
    fn worker_clear_requires_complete_unambiguous_current_boot_acknowledgements() {
        let owner = KvSessions::new("controller".into());
        owner.register(&identity(0), "p", "d").unwrap();
        let mut other = identity(0);
        other.agent_id = "other".into();
        owner.register(&other, "p", "d").unwrap();
        owner
            .state
            .lock()
            .boots
            .insert("d".into(), "old-boot".into());
        owner.begin_worker_clear("d");
        let controls = next_worker_controls(&owner, "d");
        assert_eq!(controls.len(), 2);
        for status in ["missing", "stale", "unknown", "pending", "ready"] {
            owner.apply("d", &controls, clear_reply(&controls, "old-boot", status));
            assert_eq!(owner.state.lock().clearing["d"].len(), 2, "{status}");
        }
        let mut wrong_version = clear_reply(&controls, "old-boot", "released");
        for result in &mut wrong_version.results {
            result.version += 1;
        }
        owner.apply("d", &controls, wrong_version);
        assert_eq!(owner.state.lock().clearing["d"].len(), 2);

        // One acknowledged session never establishes a complete node clear.
        owner.apply(
            "d",
            &controls,
            clear_reply(&controls[..1], "old-boot", "released"),
        );
        assert!(!owner.finish_worker_clear("d"));
        let remaining = &controls[1..];
        let mut ambiguous = clear_reply(remaining, "old-boot", "released");
        ambiguous
            .results
            .extend(clear_reply(remaining, "old-boot", "released").results);
        owner.apply("d", remaining, ambiguous);
        assert!(!owner.finish_worker_clear("d"));

        // Learning a new cache generation invalidates the whole old reply;
        // even a second reply cannot acknowledge a release sent to old-boot.
        owner.apply(
            "d",
            remaining,
            clear_reply(remaining, "new-boot", "released"),
        );
        owner.apply(
            "d",
            remaining,
            clear_reply(remaining, "new-boot", "released"),
        );
        assert!(!owner.finish_worker_clear("d"));
        let current = next_worker_controls(&owner, "d");
        assert_eq!(current.len(), 1);
        assert_eq!(current[0].expected_boot_id.as_deref(), Some("new-boot"));
        owner.apply("d", &current, clear_reply(&current, "new-boot", "released"));
        assert!(owner.finish_worker_clear("d"));
    }

    #[test]
    fn end_during_worker_clear_waits_for_end_tombstone_not_inflight_release() {
        let owner = KvSessions::new("controller".into());
        let tag = owner.register(&identity(0), "p", "d").unwrap();
        owner
            .state
            .lock()
            .boots
            .extend([("p".into(), "p-boot".into()), ("d".into(), "d-boot".into())]);
        owner.begin_worker_clear("d");
        let releases = next_worker_controls(&owner, "d");
        assert_eq!(releases[0].op, "release");
        owner.end(&identity(0)).unwrap();
        owner.apply("d", &releases, clear_reply(&releases, "d-boot", "released"));
        assert!(!owner.finish_worker_clear("d"));
        let ends = next_worker_controls(&owner, "d");
        assert_eq!(ends[0].op, "end");
        owner.apply("d", &ends, clear_reply(&ends, "d-boot", "ended"));
        owner.requeue(&ends);
        assert!(owner.finish_worker_clear("d"));
        assert!(!owner.state.lock().sessions[&tag.session_id]
            .workers
            .contains_key("d"));
        let p_end = next_worker_controls(&owner, "p");
        assert_eq!(p_end[0].op, "end");
        owner.apply("p", &p_end, clear_reply(&p_end, "p-boot", "ended"));
        assert!(owner.state.lock().sessions[&tag.session_id]
            .workers
            .is_empty());
        assert!(owner.register(&identity(1), "p", "d").is_err());
    }

    #[test]
    fn clearing_current_prefill_keeps_ordinary_retirement_queued() {
        let owner = KvSessions::new("controller".into());
        owner.register(&identity(0), "p0", "d").unwrap();
        let tag = owner.register(&identity(1), "p1", "d").unwrap();
        owner
            .state
            .lock()
            .boots
            .insert("p1".into(), "p1-boot".into());
        owner.begin_worker_clear("p1");
        let releases = next_worker_controls(&owner, "p1");
        assert_eq!(releases[0].op, "release");
        // The ordinary READY probe is filtered during the clear and therefore
        // has no HTTP completion that could requeue this retirement itself.
        assert!(owner.state.lock().queued.contains(&tag.session_id));
        owner.apply(
            "p1",
            &releases,
            clear_reply(&releases, "p1-boot", "released"),
        );
        assert!(owner.finish_worker_clear("p1"));
        let pending = next_worker_controls(&owner, "p1");
        assert_eq!(pending[0].op, "status");
        assert_eq!(pending[0].session_id, tag.session_id);
        assert_eq!(
            owner.state.lock().sessions[&tag.session_id]
                .retire
                .get("p0"),
            Some(&1)
        );
    }

    #[test]
    fn unsuccessful_clear_does_not_starve_maximum_worker_end_group() {
        let owner = KvSessions::new("controller".into());
        let mut history = identity(0);
        let tag = owner.register(&history, "p0", "fixed-d").unwrap();
        for step in 1..MAX_WORKERS_PER_SESSION - 1 {
            history.request_id = step as u64;
            owner
                .register(&history, &format!("p{step}"), "fixed-d")
                .unwrap();
        }
        owner.end(&history).unwrap();
        let mut other = identity(0);
        other.agent_id = "clearing".into();
        owner.register(&other, "other-p", "clearing-d").unwrap();
        owner.begin_worker_clear("clearing-d");
        {
            let mut state = owner.state.lock();
            let workers: Vec<_> = state.sessions[&tag.session_id]
                .workers
                .keys()
                .cloned()
                .collect();
            assert_eq!(workers.len(), BATCH);
            state
                .boots
                .extend(workers.into_iter().map(|worker| (worker, "boot".into())));
            state.boots.insert("clearing-d".into(), "boot".into());
        }
        let mut clear_attempts = 0;
        let mut normal_attempts = 0;
        for _ in 0..4 {
            let batches = owner.batches();
            assert!(
                batches
                    .iter()
                    .map(|(_, controls)| controls.len())
                    .sum::<usize>()
                    <= BATCH
            );
            let mut group = 0;
            for (worker, controls) in &batches {
                if worker == "clearing-d" {
                    clear_attempts += 1;
                }
                group += controls
                    .iter()
                    .filter(|c| c.session_id == tag.session_id && c.op == "end")
                    .count();
                owner.requeue(controls); // Every attempt fails; both kinds must retry.
            }
            if group > 0 {
                assert_eq!(group, BATCH, "the 128-worker END group stays intact");
                normal_attempts += 1;
            }
        }
        assert_eq!(clear_attempts, 2);
        assert_eq!(normal_attempts, 2);
        assert!(!owner.finish_worker_clear("clearing-d"));
    }

    /// A P-only registration (P, P) puts no decode worker into the
    /// registry. A P move queues only the old P, and its release goes out
    /// only after the new P reports the full context ready. The counters
    /// follow each step.
    #[test]
    fn prefill_only_registration_releases_the_old_prefill_after_ready() {
        let owner = KvSessions::new("controller".into());
        let get = |c: &AtomicU64| c.load(Ordering::Relaxed);
        owner.register(&identity(0), "p1", "p1").unwrap();
        owner
            .state
            .lock()
            .boots
            .insert("p1".into(), "p1-boot".into());
        // Same P, next step: a new version and nothing to retire.
        assert_eq!(owner.register(&identity(1), "p1", "p1").unwrap().version, 2);
        assert!(owner.batches().is_empty());
        assert_eq!(get(&owner.stats().retire_queued), 0);

        // The P moves: p1 (version 2) waits behind the READY of p2.
        let moved = owner.register(&identity(2), "p2", "p2").unwrap();
        assert_eq!(moved.version, 3);
        assert_eq!(get(&owner.stats().retire_queued), 1);
        let backlog = owner.retire_backlog();
        assert_eq!((backlog.sessions, backlog.pending), (1, 1));
        let (worker, controls) = owner.batches().pop().unwrap();
        assert_eq!(worker, "p2");
        assert_eq!(controls.len(), 1);
        assert_eq!((controls[0].op, controls[0].version), ("status", 3));
        owner.apply(&worker, &controls, ready(&moved, "p2-boot", 16));
        owner.requeue(&controls);
        let (worker, controls) = owner.batches().pop().unwrap();
        assert_eq!(
            (worker.as_str(), controls[0].op),
            ("p2", "status"),
            "a partial READY releases nothing"
        );
        owner.apply(&worker, &controls, ready(&moved, "p2-boot", 32));
        owner.requeue(&controls);
        let (worker, controls) = owner.batches().pop().unwrap();
        assert_eq!(worker, "p1");
        assert_eq!((controls[0].op, controls[0].version), ("release", 2));
        assert_eq!(controls[0].expected_boot_id.as_deref(), Some("p1-boot"));
        let mut released = ready(&moved, "p1-boot", 0);
        released.results[0].version = 2;
        released.results[0].status = "released".into();
        owner.apply(&worker, &controls, released);
        owner.requeue(&controls);
        assert!(owner.batches().is_empty());

        let stats = owner.stats();
        assert_eq!(get(&stats.status_sent), 2);
        assert_eq!(get(&stats.ready), 1);
        assert_eq!(get(&stats.release_sent), 1);
        assert_eq!(get(&stats.release_acked), 1);
        assert_eq!(get(&stats.retire_released), 1);
        assert_eq!(get(&stats.end_sent), 0);
        assert_eq!(owner.retire_backlog().pending, 0);
    }

    // -- late D ------------------------------------------------------------

    /// Every worker of each control batch, in batch order.
    fn batch_ops(owner: &KvSessions) -> Vec<(String, &'static str, u64)> {
        let mut out: Vec<(String, &'static str, u64)> = owner
            .batches()
            .into_iter()
            .flat_map(|(w, cs)| cs.into_iter().map(move |c| (w.clone(), c.op, c.version)))
            .collect();
        out.sort();
        out
    }

    /// A late-D P registration leaves the decode side alone: a decode change
    /// never bumps the version, and the first bind sets the decode home.
    #[test]
    fn late_prefill_registration_never_versions_a_decode_change() {
        let owner = KvSessions::new("controller".into());
        let t0 = owner.register_late_prefill(&identity(0), "p0").unwrap();
        assert_eq!(t0.version, 1);
        // Retry of the same request: same version.
        assert_eq!(owner.register_late_prefill(&identity(0), "p0").unwrap(), t0);
        owner.note_candidate(&t0, "d0");
        owner.bind_decode(&t0, "d0");
        {
            let state = owner.state.lock();
            let s = &state.sessions[&t0.session_id];
            assert_eq!((s.p.as_str(), s.d.as_str()), ("p0", "d0"));
            assert_eq!(s.workers.get("d0"), Some(&1));
            assert!(s.retire.is_empty());
        }
        // Next turn, same P: one new version, the decode home unchanged.
        let t1 = owner.register_late_prefill(&identity(1), "p0").unwrap();
        assert_eq!(t1.version, 2);
        let state = owner.state.lock();
        let s = &state.sessions[&t1.session_id];
        assert_eq!(s.d, "d0");
        assert!(s.retire.is_empty(), "a new turn retires nothing by itself");
    }

    /// One decode copy per agent: binding another decode retires the old home
    /// (at its own version) and the unused ADMIT candidates of this version,
    /// after the current P reports the full context ready.
    #[test]
    fn late_bind_retires_the_old_home_and_unused_candidates_after_ready() {
        let owner = KvSessions::new("controller".into());
        let t0 = owner.register_late_prefill(&identity(0), "p0").unwrap();
        owner.note_candidate(&t0, "d0");
        owner.bind_decode(&t0, "d0");
        let t1 = owner.register_late_prefill(&identity(1), "p0").unwrap();
        // The arrival pick d2 and the decided winner d1; d1 took the stream.
        owner.note_candidate(&t1, "d2");
        owner.note_candidate(&t1, "d1");
        owner.bind_decode(&t1, "d1");
        {
            let state = owner.state.lock();
            let s = &state.sessions[&t1.session_id];
            assert_eq!(s.d, "d1");
            assert_eq!(s.retire.get("d0"), Some(&t0.version));
            assert_eq!(s.retire.get("d2"), Some(&t1.version));
            assert!(!s.retire.contains_key("d1"));
            assert!(s.maybe.is_empty());
        }
        assert_eq!(owner.stats().retire_queued.load(Ordering::Relaxed), 2);
        // Gate: the current P's status first.
        let (worker, controls) = owner.batches().pop().unwrap();
        assert_eq!((worker.as_str(), controls[0].op), ("p0", "status"));
        assert_eq!(controls[0].version, t1.version);
        {
            let mut state = owner.state.lock();
            state.boots.insert("d0".into(), "d0-boot".into());
            state.boots.insert("d2".into(), "d2-boot".into());
        }
        owner.apply(&worker, &controls, ready(&t1, "p0-boot", 32));
        owner.requeue(&controls);
        assert_eq!(
            batch_ops(&owner),
            vec![
                ("d0".to_string(), "release", t0.version),
                ("d2".to_string(), "release", t1.version),
            ]
        );
    }

    /// The same decode again: nothing is retired. A bind of an older version
    /// (a newer request already registered) queues only its own release.
    #[test]
    fn late_bind_of_the_same_home_or_an_older_version() {
        let owner = KvSessions::new("controller".into());
        let t0 = owner.register_late_prefill(&identity(0), "p0").unwrap();
        owner.note_candidate(&t0, "d0");
        owner.bind_decode(&t0, "d0");
        let t1 = owner.register_late_prefill(&identity(1), "p0").unwrap();
        owner.note_candidate(&t1, "d0");
        owner.bind_decode(&t1, "d0");
        {
            let state = owner.state.lock();
            let s = &state.sessions[&t1.session_id];
            assert!(s.retire.is_empty());
            assert_eq!(s.workers.get("d0"), Some(&t1.version));
        }
        let t2 = owner.register_late_prefill(&identity(2), "p0").unwrap();
        // A late bind of t1 on d5 after t2 registered.
        owner.bind_decode(&t1, "d5");
        let state = owner.state.lock();
        let s = &state.sessions[&t2.session_id];
        assert_eq!(s.d, "d0", "an older bind never moves the home");
        assert_eq!(s.retire.get("d5"), Some(&t1.version));
    }

    /// A P change under late D retires the old P (as before) and keeps the
    /// decode home; END reaches every recorded candidate, also one recorded
    /// after END.
    #[test]
    fn late_p_change_and_end_reach_every_recorded_worker() {
        let owner = KvSessions::new("controller".into());
        let t0 = owner.register_late_prefill(&identity(0), "p0").unwrap();
        owner.note_candidate(&t0, "d0");
        owner.bind_decode(&t0, "d0");
        let t1 = owner.register_late_prefill(&identity(1), "p1").unwrap();
        {
            let state = owner.state.lock();
            let s = &state.sessions[&t1.session_id];
            assert_eq!(s.retire.get("p0"), Some(&t0.version));
            assert_eq!(s.d, "d0");
        }
        owner.note_candidate(&t1, "d3");
        owner.end(&identity(0)).unwrap();
        let ops = batch_ops(&owner);
        let workers: Vec<&str> = ops.iter().map(|(w, _, _)| w.as_str()).collect();
        assert_eq!(workers, vec!["d0", "d3", "p0", "p1"]);
        // A decode recorded after END also gets END.
        owner.note_candidate(&t1, "d4");
        let ops = batch_ops(&owner);
        assert!(ops.iter().any(|(w, _, _)| w == "d4"), "{ops:?}");
    }

    /// A possible ADMIT target that never bound (its leg failed after ADMIT)
    /// is queued for release at its own version when the next version
    /// registers, behind the new P's READY (one decode copy per
    /// agent). The decode home stays. A late bind of the old
    /// version keeps that release and counts nothing new.
    #[test]
    fn late_unbound_admit_target_is_released_at_its_version_on_the_next_version() {
        let owner = KvSessions::new("controller".into());
        let get = |c: &AtomicU64| c.load(Ordering::Relaxed);
        let t1 = owner.register_late_prefill(&identity(0), "p0").unwrap();
        owner.note_candidate(&t1, "d0");
        owner.bind_decode(&t1, "d0");
        // Turn 2: dx may receive ADMIT, but its leg never answers 2xx.
        let t2 = owner.register_late_prefill(&identity(1), "p0").unwrap();
        owner.note_candidate(&t2, "dx");
        // Turn 3 registers: dx is queued at version 2; the home d0 is not.
        let t3 = owner.register_late_prefill(&identity(2), "p0").unwrap();
        {
            let state = owner.state.lock();
            let s = &state.sessions[&t3.session_id];
            assert_eq!(s.retire.get("dx"), Some(&t2.version));
            assert!(!s.retire.contains_key("d0"));
            assert_eq!(s.d, "d0");
            assert!(s.maybe.is_empty());
        }
        assert_eq!(get(&owner.stats().retire_queued), 1);
        assert_eq!(owner.retire_backlog().pending, 1);
        // Turn 3 binds dy: the old home d0 is queued too; dx keeps version 2.
        owner.note_candidate(&t3, "dy");
        owner.bind_decode(&t3, "dy");
        {
            let state = owner.state.lock();
            let s = &state.sessions[&t3.session_id];
            assert_eq!(s.d, "dy");
            assert_eq!(s.retire.get("dx"), Some(&t2.version));
            assert_eq!(s.retire.get("d0"), Some(&t1.version));
        }
        assert_eq!(get(&owner.stats().retire_queued), 2);
        // Gate: the current P's status first, then each release at its own
        // version.
        let (worker, controls) = owner.batches().pop().unwrap();
        assert_eq!(
            (worker.as_str(), controls[0].op, controls[0].version),
            ("p0", "status", t3.version)
        );
        {
            let mut state = owner.state.lock();
            state.boots.insert("d0".into(), "d0-boot".into());
            state.boots.insert("dx".into(), "dx-boot".into());
        }
        owner.apply(&worker, &controls, ready(&t3, "p0-boot", 32));
        owner.requeue(&controls);
        assert_eq!(
            batch_ops(&owner),
            vec![
                ("d0".to_string(), "release", t1.version),
                ("dx".to_string(), "release", t2.version),
            ]
        );
        // dx's 2xx arrives after turn 3 registered: an older-version bind.
        owner.bind_decode(&t2, "dx");
        {
            let state = owner.state.lock();
            let s = &state.sessions[&t3.session_id];
            assert_eq!(s.retire.get("dx"), Some(&t2.version));
            assert_eq!(s.d, "dy");
        }
        assert_eq!(get(&owner.stats().retire_queued), 2);
    }

    /// A possible ADMIT target that becomes the next P is not released: the
    /// P-leg registers it at the new version. The old P is queued as before.
    #[test]
    fn late_unbound_admit_target_that_becomes_the_p_keeps_its_registration() {
        let owner = KvSessions::new("controller".into());
        let t1 = owner.register_late_prefill(&identity(0), "p0").unwrap();
        owner.note_candidate(&t1, "w1");
        let t2 = owner.register_late_prefill(&identity(1), "w1").unwrap();
        let state = owner.state.lock();
        let s = &state.sessions[&t2.session_id];
        assert_eq!(s.retire.get("p0"), Some(&t1.version));
        assert!(!s.retire.contains_key("w1"));
        assert_eq!(s.workers.get("w1"), Some(&t2.version));
        drop(state);
        assert_eq!(owner.stats().retire_queued.load(Ordering::Relaxed), 1);
    }

    fn protects(batches: &[(String, Vec<Control>)], worker: &str) -> Vec<Control> {
        batches
            .iter()
            .filter(|(w, _)| w == worker)
            .flat_map(|(_, controls)| controls.iter().filter(|c| c.op == "protect").cloned())
            .collect()
    }

    #[test]
    fn handover_protects_current_prefill_sessions_until_acknowledged() {
        let owner = KvSessions::new("controller".into());
        let on_p = owner.register(&identity(0), "p", "d").unwrap();
        let other = RequestIdentity {
            agent_id: "b".into(),
            subagent_id: 2,
            request_id: 0,
        };
        owner.register(&other, "p2", "p").unwrap(); // `p` is only its D.
        owner
            .state
            .lock()
            .boots
            .extend([("p".into(), "pboot".into()), ("p2".into(), "p2boot".into())]);
        owner.begin_worker_handover("p");

        let batches = owner.batches();
        let sent = protects(&batches, "p");
        assert_eq!(sent.len(), 1, "only sessions whose current P is the worker");
        assert_eq!(sent[0].session_id, on_p.session_id);
        assert_eq!(sent[0].version, on_p.version);
        assert_eq!(sent[0].expected_boot_id.as_deref(), Some("pboot"));

        owner.requeue(&sent); // A failed transport retries the same protection.
        let retried = protects(&owner.batches(), "p");
        assert_eq!(retried.len(), 1);
        owner.apply("p", &retried, clear_reply(&retried, "pboot", "protected"));
        owner.requeue(&retried);
        assert!(protects(&owner.batches(), "p").is_empty());
    }

    #[test]
    fn pending_release_supersedes_handover_protection() {
        let owner = KvSessions::new("controller".into());
        owner.register(&identity(0), "p", "d").unwrap();
        owner
            .state
            .lock()
            .boots
            .extend([("p".into(), "pboot".into()), ("p2".into(), "p2boot".into())]);
        owner.begin_worker_handover("p");
        // The next step already moved to p2: `p` awaits ordinary retirement.
        owner.register(&identity(1), "p2", "d").unwrap();
        let batches = owner.batches();
        assert!(protects(&batches, "p").is_empty());
        let state = owner.state.lock();
        let s = state.sessions.values().next().unwrap();
        assert!(s.protect.is_empty());
        assert!(s.retire.contains_key("p"));
    }

    #[test]
    fn stale_protection_reply_is_final() {
        let owner = KvSessions::new("controller".into());
        owner.register(&identity(0), "p", "d").unwrap();
        owner.state.lock().boots.insert("p".into(), "pboot".into());
        owner.begin_worker_handover("p");
        let sent = protects(&owner.batches(), "p");
        owner.apply("p", &sent, clear_reply(&sent, "pboot", "stale"));
        owner.requeue(&sent);
        assert!(protects(&owner.batches(), "p").is_empty());
    }

    // ---------------------------------------------------------------- REPRO
    fn agent(i: usize) -> RequestIdentity {
        RequestIdentity {
            agent_id: format!("agent-{i}"),
            subagent_id: 0,
            request_id: 0,
        }
    }
    /// Answer every pending control as a healthy engine does (`status` of an
    /// unknown boot first, then `end` -> `ended`, `release` -> `released`,
    /// `status` -> `ready`), except for `dead`, which never answers.
    fn run_engines(owner: &KvSessions, dead: Option<&str>) -> usize {
        let mut cycles = 0;
        loop {
            if owner.state.lock().queue.is_empty() {
                return cycles;
            }
            let batches = owner.batches();
            cycles += 1;
            let mut progressed = batches.is_empty();
            for (worker, controls) in batches {
                if dead != Some(worker.as_str()) {
                    let reply = Reply {
                        boot_id: format!("boot-{worker}"),
                        results: controls
                            .iter()
                            .map(|c| Outcome {
                                session_id: c.session_id.clone(),
                                version: c.version,
                                status: match c.op {
                                    "end" => "ended",
                                    "release" => "released",
                                    "protect" => "protected",
                                    _ => "ready",
                                }
                                .into(),
                                ready_tokens: 32,
                                required_tokens: 32,
                            })
                            .collect(),
                    };
                    owner.apply(&worker, &controls, reply);
                    progressed = true;
                }
                owner.requeue(&controls);
            }
            if !progressed || cycles > 100_000 {
                return cycles;
            }
        }
    }

    /// Every agent branch is ended with `POST /kv_sessions/end` and both
    /// engines acknowledge every END: the 100,001st branch is admitted, and a
    /// registry full of live branches still refuses a new one.
    #[test]
    fn ended_and_acknowledged_sessions_leave_a_full_registry() {
        let owner = KvSessions::new("controller".into());
        for i in 0..MAX_SESSIONS {
            owner.register(&agent(i), "p", "d").unwrap();
            owner.end(&agent(i)).unwrap();
        }
        run_engines(&owner, None);
        {
            let state = owner.state.lock();
            assert!(state.queue.is_empty());
            assert!(
                state
                    .sessions
                    .values()
                    .all(|s| s.ended && s.workers.is_empty()),
                "every END was acknowledged by every engine"
            );
        }
        assert_eq!(
            owner
                .register(&agent(MAX_SESSIONS), "p", "d")
                .map(|t| t.version),
            Ok(1),
            "a new agent branch after 100,000 ended ones"
        );
        assert_eq!(owner.state.lock().sessions.len(), 1);

        let live = KvSessions::new("controller".into());
        for i in 0..MAX_SESSIONS {
            live.register(&agent(i), "p", "d").unwrap();
        }
        assert!(live
            .register(&agent(MAX_SESSIONS), "p", "d")
            .unwrap_err()
            .contains("at capacity"));
        assert_eq!(live.state.lock().sessions.len(), MAX_SESSIONS);
    }

    /// Control cycles a decode->prefill clear needs when N live sessions
    /// recorded the worker and every engine answers at once: clears run on
    /// every other cycle, one `status` round learns the boot, then one
    /// `release` round per `BATCH` sessions.
    #[test]
    fn worker_clear_cycles_grow_with_batches() {
        for n in [1_000usize, 10_000] {
            let owner = KvSessions::new("controller".into());
            for i in 0..n {
                owner.register(&agent(i), "p", "d").unwrap();
            }
            owner.begin_worker_clear("d");
            let mut cycles = 0;
            while !owner.finish_worker_clear("d") {
                cycles += 1;
                for (worker, controls) in owner.batches() {
                    let reply = Reply {
                        boot_id: format!("boot-{worker}"),
                        results: controls
                            .iter()
                            .map(|c| Outcome {
                                session_id: c.session_id.clone(),
                                version: c.version,
                                status: if c.op == "release" {
                                    "released"
                                } else {
                                    "ready"
                                }
                                .into(),
                                ready_tokens: 32,
                                required_tokens: 32,
                            })
                            .collect(),
                    };
                    owner.apply(&worker, &controls, reply);
                    owner.requeue(&controls);
                }
                assert!(cycles < 1_000_000);
            }
            assert_eq!(cycles, 1 + 2 * n.div_ceil(BATCH), "sessions={n}");
        }
    }
}
