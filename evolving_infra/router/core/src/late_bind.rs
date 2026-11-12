// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Late D stage 2 (`--pd-late-decode choose`): the late-bind table, the link
//! to each prefill, and the bind-time decode choice.
//!
//! Port of the PD part of sgl-model-gateway's `routers/http/late_bind.rs`.
//! Symmetric (symmetric_pd) owners reuse this code; see the last paragraph.
//! Their LOCAL winner is not ported.
//! The engine side is `engine/decoupled_engine/srt/disaggregation/late_d*.py`;
//! every wire field below is the engine's.
//!
//! In choose mode the router sends a request to one prefill (P) and to K
//! decode (D) candidates at once. The P-leg carries `late_d {v, mode,
//! router_epoch, k, default_idx}`, each D-leg `late_d {v, mode,
//! router_epoch, k, cand_idx}`. A candidate allocates no KV until P
//! admits it. When P is about to prefill the request, its rank 0 asks this
//! router over the link (`bind_ready`), the router answers with the winner
//! (`decision`), and P sends ADMIT to the winner and RELEASE to the others;
//! a released candidate answers HTTP 409 with `X-Late-D-Reason: released`.
//! P then reports what it admitted (`final`). With no answer within
//! the engine's choice timeout, or with the link down, P commits the
//! default (the arrival pick) itself.
//!
//! * Table: one [`LateEntry`] per request, keyed by the bootstrap room and
//!   checked against the P link, from dispatch until the dispatch returns.
//! * Link: one task per P on a dedicated runtime thread: a ZMQ DEALER to the
//!   P rank-0 ROUTER at `late_d.control_port` from `/server_info`; hello,
//!   then a ping every second, a reconnect after 3 s of silence; frames are
//!   received in `tokio::task::unconstrained` (a stalled receive side
//!   livelocks the bridge).
//!   Messages are msgpack maps under 1 KB.
//! * Choice: among the candidates P names eligible that the router still
//!   holds alive and healthy, `keep` takes the arrival pick, `po2` the SMG
//!   power-of-two pick on the polled bound loads, `d1` the fewest
//!   router-charged tokens (min-inflight-tokens), `fit_req` the fewest
//!   requests among the candidates that fit (a fixed capacity on the router
//!   charges, or the observed capacity of [`crate::late_fit`]), `d1` when
//!   none fits. `fit_req` is a later sgl-model-gateway rule, ported too.
//! * Charge: every rule charges the winner (prompt + capped output tokens,
//!   the min-inflight-tokens charge) at the decision; `final` moves it when
//!   P admitted another candidate; the leg that answers 2xx takes it and
//!   holds it until its stream ends. With the observed capacity the charge
//!   also holds a reservation for the room until the decode reports it.
//!
//! Symmetric owners (`--policy symmetric_pd`, opt-in with `--pd-late-decode
//! choose`; [`LateBind::new_symmetric`]): the same table, links, rules and
//! charges serve symmetric workers. The owner is the symmetric_pd P-leg worker,
//! its link port is the server argument `disaggregation_late_d_control_port`
//! (a symmetric `late_d` block names no port), and its `hello_ack` says mode
//! `symmetric`. The table key is the `transfer_id` instead of the bootstrap
//! room. Each D-leg carries its own `kv_import` (with its own `attempt_id`)
//! next to its `late_d.cand_idx` ([`LegBodies::with_imports`]). LOCAL (the
//! owner as the winner) is not offered: `local_ok` is false.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytes::{Bytes, BytesMut};
use dashmap::DashMap;
use parking_lot::Mutex;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use zeromq::{DealerSendHalf, DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

use crate::config::{LateDecodeConfig, LateDecodeRule, PdDecodeTokensConfig};
use crate::discovery::WorkerMode;
use crate::health::circuit_breaker::CircuitBreaker;
use crate::late_fit::{DecodeCapacityStore, FitReservations, LateCharge};
use crate::policies::decode_po2_smg::SmgPowerOfTwo;
use crate::policies::decode_tokens::{DecodeTokenBalancer, TokenRule};
use crate::server::metrics::escape_label;
use crate::workers::{LateDWorkerInfo, Worker, WorkerRegistry};

/// Link protocol version (`v` in every message and in the `late_d` fields).
pub const LINK_VERSION: u64 = 1;
/// `winner_idx` for a key the router does not know: P commits its fallback
/// at once.
pub const WINNER_UNKNOWN: u8 = 255;
/// Every link message stays under 1 KB; a larger one is refused.
pub const MAX_MESSAGE_BYTES: usize = 1024;
/// Items per `decision` message; 8 keep it under 1 KB.
pub const MAX_ITEMS: usize = 8;
/// The engine's limit on candidates per request (the eligible mask is a u64).
pub const MAX_CANDIDATES: usize = 64;
/// A loser leg still open this long after the bind is dropped.
pub const LOSER_DRAIN: Duration = Duration::from_secs(30);
/// The engine's reason header on a late-D terminal answer.
pub const LATE_D_REASON_HEADER: &str = "x-late-d-reason";
/// The reason of a released candidate (HTTP 409).
pub const RELEASED_REASON: &str = "released";
/// The engine's `--disaggregation-late-d` value for stage 2.
pub const CHOOSE_MODE: &str = "choose";
/// Tracing target of every late-D line.
pub const LOG_TARGET: &str = "late_d";

const PING_INTERVAL: Duration = Duration::from_secs(1);
const LINK_SILENT: Duration = Duration::from_secs(3);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
const SEND_TIMEOUT: Duration = Duration::from_secs(1);
const RECONNECT_MIN: Duration = Duration::from_millis(100);
const RECONNECT_MAX: Duration = Duration::from_secs(5);
const SUPERVISE_INTERVAL: Duration = Duration::from_secs(1);

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

fn mask_of(k: usize) -> u64 {
    if k >= 64 {
        u64::MAX
    } else {
        (1u64 << k) - 1
    }
}

fn bits(mask: u64) -> impl Iterator<Item = usize> {
    (0..64usize).filter(move |i| (mask >> i) & 1 == 1)
}

// ---------------------------------------------------------------------------
// Link codec: one ZMQ frame per message, a msgpack map with v and t.
// ---------------------------------------------------------------------------

/// One link message, any type: every field the protocol defines, defaulted
/// when absent, so one decoder serves every type.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct LinkMessage {
    pub v: u64,
    pub t: String,
    pub router_epoch: Option<u64>,
    pub router_id: Option<String>,
    pub rules: Vec<String>,
    pub engine_boot_ns: u64,
    pub mode: Option<String>,
    pub kv_ip: Option<String>,
    pub ts_ns: u64,
    pub p_seq: u64,
    pub t_send_ns: u64,
    pub items: Vec<LinkItem>,
}

/// One item of `bind_ready` (L4), `decision` (L5) or `final` (L6).
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct LinkItem {
    pub key: u64,
    pub bind_seq: u64,
    pub eligible: u64,
    pub ready: u64,
    pub input_len: u64,
    /// When the prefill started the request's first chunk (wall clock, ns);
    /// absent when the engine does not send it.
    pub first_chunk_start_unix_ns: Option<u64>,
    pub default_idx: u64,
    pub local_ok: bool,
    pub local_load: u64,
    pub winner_idx: u64,
    pub rule: Option<String>,
    pub t_recv_ns: u64,
    pub t_decide_ns: u64,
    pub admitted_idx: u64,
    pub reason: Option<String>,
    pub t_admit_ns: u64,
}

/// A link frame as a message; `Err` for anything that is not a version-1
/// map with a type, or that is larger than [`MAX_MESSAGE_BYTES`].
pub fn decode_message(frame: &[u8]) -> Result<LinkMessage, String> {
    if frame.len() > MAX_MESSAGE_BYTES {
        return Err(format!(
            "{} bytes, over the {MAX_MESSAGE_BYTES}-byte limit",
            frame.len()
        ));
    }
    let msg: LinkMessage =
        rmp_serde::from_slice(frame).map_err(|e| format!("undecodable link message: {e}"))?;
    if msg.v != LINK_VERSION {
        return Err(format!("not a version-{LINK_VERSION} link message"));
    }
    if msg.t.is_empty() {
        return Err("link message without a type".to_string());
    }
    Ok(msg)
}

#[derive(Serialize)]
struct HelloMsg<'a> {
    v: u64,
    t: &'static str,
    router_epoch: u32,
    router_id: &'a str,
    rules: [&'static str; 1],
}

#[derive(Serialize)]
struct PingMsg {
    v: u64,
    t: &'static str,
    ts_ns: u64,
}

/// One answer in a `decision` message (L5).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DecisionItem {
    pub key: u64,
    pub bind_seq: u64,
    pub winner_idx: u8,
    pub rule: &'static str,
    pub t_recv_ns: u64,
    pub t_decide_ns: u64,
}

#[derive(Serialize)]
struct DecisionMsg<'a> {
    v: u64,
    t: &'static str,
    router_epoch: u32,
    items: &'a [DecisionItem],
}

fn encode<T: Serialize>(msg: &T) -> Vec<u8> {
    rmp_serde::to_vec_named(msg).unwrap_or_else(|e| {
        tracing::error!(target: LOG_TARGET, "late_d: link message failed to encode: {e}");
        Vec::new()
    })
}

fn ping_frame(ts_ns: u64) -> Vec<u8> {
    encode(&PingMsg {
        v: LINK_VERSION,
        t: "ping",
        ts_ns,
    })
}

// ---------------------------------------------------------------------------
// Worker registration (D15: refuse a worker without late D choose).
// ---------------------------------------------------------------------------

/// Why choose mode refuses a worker at registration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// `/server_info` has no `late_d` block (late D off, or an older engine).
    NoLateD,
    /// `late_d.mode` is not `choose` (for example an `alloc` engine).
    Mode(String),
    /// A PD prefill without `late_d.control_port` (the router link).
    NoControlPort,
}

impl Refusal {
    /// The `reason` label of the refusal counter and log line.
    pub fn reason(&self) -> &'static str {
        match self {
            Self::NoLateD => "no_late_d",
            Self::Mode(_) => "mode",
            Self::NoControlPort => "no_control_port",
        }
    }

    fn slot(&self) -> usize {
        match self {
            Self::NoLateD => 0,
            Self::Mode(_) => 1,
            Self::NoControlPort => 2,
        }
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoLateD => write!(f, "/server_info has no late_d block"),
            Self::Mode(mode) => write!(f, "late_d.mode is {mode}, not {CHOOSE_MODE}"),
            Self::NoControlPort => write!(f, "the prefill's late_d has no control_port"),
        }
    }
}

/// The registration verdict for one worker in choose mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistrationCheck {
    /// A prefill or decode worker that runs late D choose.
    Pass,
    /// Not a PD prefill or decode worker: no late-D leg goes to it.
    NotChecked,
    /// `/server_info` could not be read: the worker must take no traffic
    /// until a later read (the manager registers it without a model).
    Unread,
    Refuse(Refusal),
}

/// Choose-mode registration rule for a worker of `mode` whose
/// `/server_info` was read (`server_info_read`) with the `late_d` block
/// `late_d`: (a) no block, (b) a mode other than `choose`, (c) a prefill
/// without a control port are refused.
pub fn registration_check(
    mode: WorkerMode,
    server_info_read: bool,
    late_d: Option<&LateDWorkerInfo>,
) -> RegistrationCheck {
    if !server_info_read {
        return RegistrationCheck::Unread;
    }
    let prefill = match mode {
        WorkerMode::Prefill => true,
        WorkerMode::Decode => false,
        WorkerMode::Plain | WorkerMode::Symmetric => return RegistrationCheck::NotChecked,
    };
    let Some(info) = late_d else {
        return RegistrationCheck::Refuse(Refusal::NoLateD);
    };
    if info.mode != CHOOSE_MODE {
        return RegistrationCheck::Refuse(Refusal::Mode(info.mode.clone()));
    }
    if prefill && info.control_port.is_none() {
        return RegistrationCheck::Refuse(Refusal::NoControlPort);
    }
    RegistrationCheck::Pass
}

static REFUSED: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static PACKED_PROMPTS: AtomicU64 = AtomicU64::new(0);

/// Count one registration refusal (rendered by [`render_static_metrics`]).
pub fn count_refusal(refusal: &Refusal) {
    REFUSED[refusal.slot()].fetch_add(1, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Request-side helpers: worker capability, bodies, packed prompt, headers.
// ---------------------------------------------------------------------------

/// The P link endpoint of a prefill worker that runs late D choose
/// (`late_d.mode` and `late_d.control_port` from `/server_info`), on the
/// worker's host.
pub fn link_endpoint(worker: &Worker) -> Option<String> {
    let info = worker.late_d()?;
    if info.mode != CHOOSE_MODE {
        return None;
    }
    let port = info.control_port.filter(|p| *p != 0)?;
    let host = worker.bootstrap_host();
    Some(if host.contains(':') && !host.starts_with('[') {
        format!("tcp://[{host}]:{port}")
    } else {
        format!("tcp://{host}:{port}")
    })
}

/// A decode worker that runs late D choose (can take a candidate leg).
pub fn is_choose_decode(worker: &Worker) -> bool {
    worker.late_d().is_some_and(|i| i.mode == CHOOSE_MODE)
}

/// The P-leg `late_d` field.
pub fn prefill_field(router_epoch: u32, k: usize, default_idx: usize) -> Value {
    json!({
        "v": LINK_VERSION,
        "mode": CHOOSE_MODE,
        "router_epoch": router_epoch,
        "k": k,
        "default_idx": default_idx,
    })
}

/// The P-leg `late_d` field of a symmetric owner:
/// the PD field plus `local_ok: false` (LOCAL is not offered).
pub fn sym_prefill_field(router_epoch: u32, k: usize, default_idx: usize) -> Value {
    let mut field = prefill_field(router_epoch, k, default_idx);
    if let Some(obj) = field.as_object_mut() {
        obj.insert("local_ok".to_string(), Value::Bool(false));
    }
    field
}

/// A D-leg `late_d` field.
pub fn decode_field(router_epoch: u32, k: usize, cand_idx: usize) -> Value {
    json!({
        "v": LINK_VERSION,
        "mode": CHOOSE_MODE,
        "router_epoch": router_epoch,
        "k": k,
        "cand_idx": cand_idx,
    })
}

/// The D-leg bodies of one request: the shared body serialized once, and
/// per candidate a tail that adds `late_d` with its `cand_idx`.
#[derive(Debug, Clone)]
pub struct LegBodies {
    /// The shared body without its closing brace.
    prefix: Bytes,
    empty: bool,
    router_epoch: u32,
    k: usize,
    /// Symmetric owners: per candidate (cand_idx order) the serialized
    /// `kv_import` object; `None` on the PD path.
    imports: Option<Arc<[Bytes]>>,
}

fn trim_end_ascii(bytes: &[u8]) -> &[u8] {
    let end = bytes
        .iter()
        .rposition(|b| !b.is_ascii_whitespace())
        .map_or(0, |i| i + 1);
    &bytes[..end]
}

impl LegBodies {
    /// `body` must be a serialized JSON object without a `late_d` field.
    pub fn from_object_bytes(body: &[u8], router_epoch: u32, k: usize) -> Result<Self, String> {
        let body = trim_end_ascii(body);
        let start = body
            .iter()
            .position(|b| !b.is_ascii_whitespace())
            .unwrap_or(body.len());
        let body = &body[start..];
        let not_object = || "decode body is not a JSON object".to_string();
        if body.first() != Some(&b'{') {
            return Err(not_object());
        }
        let Some((&b'}', inner)) = body.split_last() else {
            return Err(not_object());
        };
        let inner = trim_end_ascii(inner);
        // In a JSON object only the empty one has `{` before its closing
        // brace (else a value ends there).
        let empty = inner.last() == Some(&b'{');
        Ok(Self {
            prefix: Bytes::copy_from_slice(inner),
            empty,
            router_epoch,
            k,
            imports: None,
        })
    }

    /// Symmetric owners: each candidate leg also carries its own
    /// `kv_import` object (`imports[cand_idx]`). The shared body must carry
    /// no `kv_import`. Exactly `k` objects are required.
    pub fn with_imports(mut self, imports: &[Value]) -> Result<Self, String> {
        if imports.len() != self.k {
            return Err(format!(
                "{} kv_import objects for {} candidates",
                imports.len(),
                self.k
            ));
        }
        let mut out = Vec::with_capacity(imports.len());
        for import in imports {
            if !import.is_object() {
                return Err("kv_import is not a JSON object".to_string());
            }
            out.push(Bytes::from(
                serde_json::to_vec(import).map_err(|e| e.to_string())?,
            ));
        }
        self.imports = Some(out.into());
        Ok(self)
    }

    /// `base` must be a JSON object without a `late_d` field.
    pub fn new(base: &Value, router_epoch: u32, k: usize) -> Result<Self, String> {
        if !base.is_object() {
            return Err("decode body is not a JSON object".to_string());
        }
        let bytes = serde_json::to_vec(base).map_err(|e| e.to_string())?;
        Self::from_object_bytes(&bytes, router_epoch, k)
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// The body of candidate `cand_idx`.
    pub fn leg(&self, cand_idx: usize) -> Bytes {
        let import = self.imports.as_ref().and_then(|i| i.get(cand_idx));
        let tail = format!(
            "\"late_d\":{{\"v\":{LINK_VERSION},\"mode\":\"{CHOOSE_MODE}\",\
             \"router_epoch\":{},\"k\":{},\"cand_idx\":{cand_idx}}}}}",
            self.router_epoch, self.k,
        );
        let import_len = import.map_or(0, |b| b.len() + 13);
        let mut out = BytesMut::with_capacity(self.prefix.len() + import_len + tail.len() + 1);
        out.extend_from_slice(&self.prefix);
        if !self.empty {
            out.extend_from_slice(b",");
        }
        if let Some(import) = import {
            out.extend_from_slice(b"\"kv_import\":");
            out.extend_from_slice(import);
            out.extend_from_slice(b",");
        }
        out.extend_from_slice(tail.as_bytes());
        out.freeze()
    }
}

const B64_ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 (RFC 4648 alphabet, `=` padding), the encoding Python's
/// `base64.b64decode` reads. Hand-rolled: the router adds no dependency.
pub fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(B64_ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(B64_ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            B64_ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64_ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Move a single token prompt into `prompt_ids_b64` (base64 of
/// little-endian int32 ids) and `prompt_len`, and drop `prompt`, when the
/// engine takes it that way: one prompt of ids within int32, no echo, n = 1,
/// no `strict_replay_deadline_audit` (its check needs the JSON ids), not
/// packed already. For /v1/completions decode-leg bodies only. Returns
/// whether it packed.
pub fn pack_prompt(body: &mut Value) -> bool {
    let Some(obj) = body.as_object_mut() else {
        return false;
    };
    let flag = |k: &str| obj.get(k).and_then(Value::as_bool).unwrap_or(false);
    if obj.contains_key("prompt_ids_b64")
        || flag("echo")
        || flag("strict_replay_deadline_audit")
        || obj
            .get("n")
            .is_some_and(|n| !n.is_null() && n.as_u64() != Some(1))
    {
        return false;
    }
    let Some(ids) = obj.get("prompt").and_then(Value::as_array) else {
        return false;
    };
    if ids.is_empty() {
        return false;
    }
    let mut packed = Vec::with_capacity(ids.len() * 4);
    for id in ids {
        let Some(id) = id.as_u64().and_then(|v| i32::try_from(v).ok()) else {
            return false;
        };
        packed.extend_from_slice(&id.to_le_bytes());
    }
    let len = ids.len();
    obj.remove("prompt");
    obj.insert(
        "prompt_ids_b64".to_string(),
        Value::String(base64_encode(&packed)),
    );
    obj.insert("prompt_len".to_string(), Value::from(len as u64));
    PACKED_PROMPTS.fetch_add(1, Ordering::Relaxed);
    true
}

/// A batch prompt (an array of prompts): choose mode binds one request at
/// a time and refuses it.
pub fn is_batch_prompt(body: &Value) -> bool {
    body.get("prompt")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .is_some_and(|first| first.is_array() || first.is_string())
}

/// The `x-late-d-reason` of an engine answer, if any.
pub fn late_reason(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get(LATE_D_REASON_HEADER)?
        .to_str()
        .ok()
        .map(str::to_string)
}

// ---------------------------------------------------------------------------
// Counters.
// ---------------------------------------------------------------------------

/// Cumulative late-D counters of one router (`/metrics`).
#[derive(Debug, Default)]
pub struct LateStats {
    pub requests_choose: AtomicU64,
    pub requests_plain: AtomicU64,
    pub decisions_decided: AtomicU64,
    pub decisions_repeat: AtomicU64,
    pub decisions_bound: AtomicU64,
    pub decisions_unknown_key: AtomicU64,
    pub decisions_no_candidate: AtomicU64,
    pub rebinds: AtomicU64,
    pub finals_decided: AtomicU64,
    pub finals_timeout: AtomicU64,
    pub finals_no_router: AtomicU64,
    pub finals_unknown_key: AtomicU64,
    pub finals_other: AtomicU64,
    pub final_mismatch: AtomicU64,
    pub bind_mismatch: AtomicU64,
    pub bind_undecided: AtomicU64,
    pub legs_winner: AtomicU64,
    pub legs_released: AtomicU64,
    pub legs_dead: AtomicU64,
    pub legs_conflict: AtomicU64,
    pub legs_not_sent: AtomicU64,
    /// `fit_req` decisions where no eligible candidate fit (the `d1` order
    /// decided).
    pub fit_none: AtomicU64,
    /// External role control: decisions whose eligible candidates included
    /// one no longer in the decode role (left out of the choice).
    pub role_filtered: AtomicU64,
    /// External role control: decisions where no eligible candidate was
    /// still in the decode role (the choice fell back to every eligible one).
    pub role_fallback: AtomicU64,
    pub loser_drain_timeouts: AtomicU64,
    pub no_winner: AtomicU64,
    /// R8: the P-leg failed before any D-leg bound (the client got P's error).
    pub prefill_failed: AtomicU64,
    /// Every P-leg that failed, before or after the bind.
    pub prefill_leg_failed: AtomicU64,
    /// P-legs that the engine ended because their reader left
    /// (`symmetric_reader_gone`): cancelled, not failed (no R8).
    pub prefill_reader_gone: AtomicU64,
    pub stale_epoch: AtomicU64,
    pub bad_messages: AtomicU64,
}

/// Add one to a [`LateStats`] counter.
pub fn inc(c: &AtomicU64) {
    c.fetch_add(1, Ordering::Relaxed);
}

/// R11: an answer with `X-Late-D-Reason` ticks no circuit breaker. A
/// half-open breaker is closed, though: the engine answered by the
/// protocol, and the probe slot the leg claimed would otherwise stay
/// claimed for good (the worker would never be picked again).
pub fn settle_late_answer(breaker: &CircuitBreaker) {
    if breaker.snapshot().state_code == 2 {
        breaker.record_success();
    }
}

/// A breaker claimed for a leg that was never sent: a half-open breaker
/// gets its probe slot back as a failure (it opens again and probes after
/// its cool-down); a closed breaker is left alone.
pub fn settle_unused_probe(breaker: &CircuitBreaker) {
    if breaker.snapshot().state_code == 2 {
        breaker.record_failure();
    }
}

fn get(c: &AtomicU64) -> u64 {
    c.load(Ordering::Relaxed)
}

/// Process-wide late-D series that exist without a [`LateBind`] (packed
/// prompts with late D off; registration refusals in the worker manager).
pub fn render_static_metrics() -> String {
    let mut out = String::new();
    out.push_str("# HELP sgl_router_late_d_packed_prompts_total Requests whose decode-leg token prompt was sent as prompt_ids_b64 (--pd-late-decode-packed-prompt); in choose mode every decode candidate leg of the request carries it.\n");
    out.push_str("# TYPE sgl_router_late_d_packed_prompts_total counter\n");
    out.push_str(&format!(
        "sgl_router_late_d_packed_prompts_total {}\n",
        get(&PACKED_PROMPTS)
    ));
    out.push_str("# HELP sgl_router_late_d_worker_refused_total Workers refused at registration in choose mode, by reason.\n");
    out.push_str("# TYPE sgl_router_late_d_worker_refused_total counter\n");
    for (slot, reason) in ["no_late_d", "mode", "no_control_port"].iter().enumerate() {
        out.push_str(&format!(
            "sgl_router_late_d_worker_refused_total{{reason=\"{reason}\"}} {}\n",
            get(&REFUSED[slot])
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// The late-bind table.
// ---------------------------------------------------------------------------

/// Session KV retention on a late-D request:
/// the router records every decode that may receive the request's session
/// tag in the P's ADMIT frame before it can receive it, and binds the
/// decode whose leg took the stream. `None` on requests without a tag.
pub trait LateSessionHook: Send + Sync {
    /// `worker_url` may receive ADMIT for this request: the arrival pick at
    /// registration, and each decided winner before the answer leaves.
    fn candidate(&self, worker_url: &str);
    /// The leg on `worker_url` answered 2xx: P admitted that decode.
    fn bound(&self, worker_url: &str);
}

/// One choose-mode request, from dispatch until the dispatch returns.
pub struct LateEntry {
    /// The P link endpoint (the table checks link messages against it).
    pub owner: String,
    pub prefill_url: String,
    /// The bootstrap room (the link's `key`; the `transfer_id` for a
    /// symmetric owner).
    pub key: u64,
    pub rid: Option<String>,
    pub candidates: Vec<Arc<Worker>>,
    pub default_idx: usize,
    /// The d1 charge of the request (prompt + capped output tokens).
    pub charge_tokens: u64,
    created: Instant,
    /// Candidates whose leg ended with an error (bit i = cand_idx i).
    dead: AtomicU64,
    state: Mutex<EntryState>,
    /// Session KV retention of this request ([`LateBind::register_with`]).
    session: Option<Arc<dyn LateSessionHook>>,
}

#[derive(Default)]
struct EntryState {
    decided: Option<Decided>,
    /// The charge of the decided (or, after a mismatched `final`, the
    /// admitted) candidate and its index, until a leg binds the stream.
    charge: Option<(usize, LateCharge)>,
    /// The candidate P reported admitted (`final`).
    admitted: Option<usize>,
    /// The candidate whose leg answered 2xx; the charge left with it.
    bound: Option<usize>,
    /// The request's decode leg ended (prepared executions,
    /// [`LateBind::drop_pending_charge`]): a later decision or `final`
    /// charges nothing.
    ended: bool,
}

#[derive(Debug, Clone, Copy)]
struct Decided {
    bind_seq: u64,
    winner: usize,
    t_decide_ns: u64,
}

impl LateEntry {
    pub fn k(&self) -> usize {
        self.candidates.len()
    }

    /// Candidates whose leg ended with an error.
    pub fn dead_mask(&self) -> u64 {
        self.dead.load(Ordering::Relaxed)
    }

    /// The decided winner, if the router answered a `bind_ready`.
    pub fn decided(&self) -> Option<usize> {
        self.state.lock().decided.map(|d| d.winner)
    }

    /// The candidate P reported admitted, if a `final` came.
    pub fn admitted(&self) -> Option<usize> {
        self.state.lock().admitted
    }

    /// The candidate expected to hold the decode: the bound one, else the
    /// one P reported admitted, else the decided winner.
    pub fn decode_target(&self) -> Option<usize> {
        let st = self.state.lock();
        st.bound.or(st.admitted).or(st.decided.map(|d| d.winner))
    }

    pub fn age(&self) -> Duration {
        self.created.elapsed()
    }
}

impl fmt::Debug for LateEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LateEntry")
            .field("owner", &self.owner)
            .field("key", &self.key)
            .field("k", &self.candidates.len())
            .field("default_idx", &self.default_idx)
            .finish()
    }
}

/// What the dispatch knows about one choose-mode request.
#[derive(Debug)]
pub struct EntrySpec {
    /// The P link endpoint ([`link_endpoint`]).
    pub owner: String,
    pub prefill_url: String,
    /// The bootstrap room.
    pub key: u64,
    pub rid: Option<String>,
    pub candidates: Vec<Arc<Worker>>,
    /// The arrival pick's index among the candidates.
    pub default_idx: usize,
    pub charge_tokens: u64,
}

/// Removes its entry from the table when dropped (the dispatch returned, or
/// its future was dropped on a client disconnect).
pub struct EntryGuard {
    late: Arc<LateBind>,
    entry: Arc<LateEntry>,
}

impl EntryGuard {
    pub fn entry(&self) -> &Arc<LateEntry> {
        &self.entry
    }
}

impl Drop for EntryGuard {
    fn drop(&mut self) {
        self.late
            .entries
            .remove_if(&self.entry.key, |_, e| Arc::ptr_eq(e, &self.entry));
    }
}

/// The state of one P link, for logs and tests.
#[derive(Debug, Clone, Default)]
pub struct LinkStatus {
    pub up: bool,
    pub engine_boot_ns: u64,
    pub kv_ip: Option<String>,
    pub hello_acks: u64,
    pub decisions: u64,
    pub finals: u64,
    pub bad_messages: u64,
}

/// Late D stage 2: the table, the links and the choice.
pub struct LateBind {
    config: LateDecodeConfig,
    rule: LateDecodeRule,
    router_epoch: u32,
    router_id: String,
    entries: DashMap<u64, Arc<LateEntry>>,
    /// The min-inflight-tokens ledger (charged at the decision in every
    /// rule; the `d1` rule picks on it).
    ledger: DecodeTokenBalancer,
    /// The `power-of-two-smg` decode policy (polled bound loads); `None`
    /// only in tests, where `po2_fallback` stands in.
    po2: Option<Arc<SmgPowerOfTwo>>,
    po2_fallback: SmgPowerOfTwo,
    registry: Arc<WorkerRegistry>,
    links: DashMap<String, LinkStatus>,
    started: AtomicBool,
    stats: LateStats,
    /// Observed decode capacity (`--pd-native-decode-state`), read by
    /// `fit_req` with `fit_native`. Filled by
    /// [`crate::late_fit::start_collector`].
    capacity: Arc<DecodeCapacityStore>,
    /// Router reservations of `fit_req` with `fit_native`, by room.
    reservations: FitReservations,
    /// Which engines own the links: PD prefills, or symmetric owners.
    owners: LateOwners,
    /// External role control (symmetric only): the workers currently in the
    /// decode role and not draining. A bind chooses among these; set once
    /// with [`Self::set_decode_role_gate`]. Unset = no role filter.
    decode_roles: std::sync::OnceLock<SymOwnerList>,
}

/// The P-capable symmetric workers a symmetric table links to (the symmetric_pd
/// roles at the time of the call). Not filtered by circuit breaker, as the
/// PD owners: a worker missing from the list loses its link.
pub type SymOwnerList = Arc<dyn Fn() -> Vec<Arc<Worker>> + Send + Sync>;

/// The engines that own late-D links.
#[derive(Clone)]
enum LateOwners {
    /// Every registered PD prefill with a late-D control port.
    Pd,
    /// Symmetric (symmetric_pd) owners: the listed workers that run late D choose
    /// and name a control port.
    Symmetric(SymOwnerList),
}

impl fmt::Debug for LateBind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LateBind")
            .field("config", &self.config)
            .field("router_epoch", &self.router_epoch)
            .field("pending", &self.entries.len())
            .field("links", &self.links.len())
            .field("symmetric", &self.symmetric())
            .finish()
    }
}

/// The outcome of one choice (before the charge is stored).
struct Choice {
    winner: usize,
    charge: LateCharge,
    considered: u64,
    signal: String,
}

impl LateBind {
    /// `config` must be choose mode (its rule is required). `tokens` tunes
    /// the min-inflight-tokens charge (`--pd-decode-tokens-*`); `po2` is
    /// the `power-of-two-smg` decode policy.
    pub fn new(
        config: LateDecodeConfig,
        tokens: PdDecodeTokensConfig,
        registry: Arc<WorkerRegistry>,
        po2: Option<Arc<SmgPowerOfTwo>>,
    ) -> Arc<Self> {
        let router_epoch = rand::thread_rng().gen_range(1..=u32::MAX);
        let rule = config.rule.unwrap_or(LateDecodeRule::Keep);
        Arc::new(Self {
            router_id: format!("sgl-router-{}-{router_epoch:08x}", std::process::id()),
            rule,
            router_epoch,
            config,
            entries: DashMap::new(),
            ledger: DecodeTokenBalancer::new(TokenRule::MinInflightTokens, tokens),
            po2,
            po2_fallback: SmgPowerOfTwo::new(),
            registry,
            links: DashMap::new(),
            started: AtomicBool::new(false),
            stats: LateStats::default(),
            capacity: Arc::new(DecodeCapacityStore::default()),
            reservations: Arc::new(Mutex::new(HashMap::new())),
            owners: LateOwners::Pd,
            decode_roles: std::sync::OnceLock::new(),
        })
    }

    /// Late D choose on symmetric workers (`--policy symmetric_pd`): the same
    /// table, rules and charges as [`Self::new`], with links to the
    /// symmetric owners `owners` returns (each with a late-D choose block and
    /// a control port) instead of the PD prefills, and no `power-of-two-smg`
    /// policy (the arrival pick is the symmetric_pd D selection; `po2` is refused
    /// at the CLI).
    pub fn new_symmetric(
        config: LateDecodeConfig,
        tokens: PdDecodeTokensConfig,
        registry: Arc<WorkerRegistry>,
        owners: SymOwnerList,
    ) -> Arc<Self> {
        let router_epoch = rand::thread_rng().gen_range(1..=u32::MAX);
        let rule = config.rule.unwrap_or(LateDecodeRule::Keep);
        Arc::new(Self {
            router_id: format!("sgl-router-sym-{}-{router_epoch:08x}", std::process::id()),
            rule,
            router_epoch,
            config,
            entries: DashMap::new(),
            ledger: DecodeTokenBalancer::new(TokenRule::MinInflightTokens, tokens),
            po2: None,
            po2_fallback: SmgPowerOfTwo::new(),
            registry,
            links: DashMap::new(),
            started: AtomicBool::new(false),
            stats: LateStats::default(),
            capacity: Arc::new(DecodeCapacityStore::default()),
            reservations: Arc::new(Mutex::new(HashMap::new())),
            owners: LateOwners::Symmetric(owners),
            decode_roles: std::sync::OnceLock::new(),
        })
    }

    /// External role control on symmetric workers: `roles` returns the
    /// workers currently in the decode role (draining ones excluded). A bind
    /// then chooses only among candidates still in that list; when none is,
    /// it falls back to every eligible candidate (counted, logged). Only the
    /// first call takes effect; without a call nothing is filtered.
    pub fn set_decode_role_gate(&self, roles: SymOwnerList) {
        let _ = self.decode_roles.set(roles);
    }

    /// The candidates of `entry` currently in the decode role (bit i =
    /// cand_idx i), or `None` without a role gate.
    fn decode_role_mask(&self, entry: &LateEntry) -> Option<u64> {
        let roles = self.decode_roles.get()?;
        let current: Vec<Arc<Worker>> = roles();
        Some(
            entry
                .candidates
                .iter()
                .enumerate()
                .filter(|(_, c)| current.iter().any(|w| w.url == c.url))
                .fold(0u64, |m, (i, _)| m | (1 << i)),
        )
    }

    /// Release the charge a decision (or a mismatched `final`) left on the
    /// entry before any leg bound: the request's decode leg ended without a
    /// bind (prepared executions; see `backend/execution.rs`).
    /// Later decisions and `final`s of the entry charge nothing.
    pub fn drop_pending_charge(&self, entry: &LateEntry) {
        let mut st = entry.state.lock();
        st.charge = None;
        st.ended = true;
    }

    /// Answer one `bind_ready` (all candidates eligible) as a link message
    /// would, for tests outside this module. Returns the winner index.
    #[cfg(test)]
    pub(crate) fn decide_for_test(&self, owner: &str, key: u64, bind_seq: u64) -> u8 {
        let item = LinkItem {
            key,
            bind_seq,
            eligible: u64::MAX,
            ready: 2,
            input_len: 100,
            ..LinkItem::default()
        };
        self.decide_item(owner, &item, 1, 0).winner_idx
    }

    /// The links go to symmetric (symmetric_pd) owners ([`Self::new_symmetric`]).
    pub fn symmetric(&self) -> bool {
        matches!(self.owners, LateOwners::Symmetric(_))
    }

    /// The link mode the owners answer in `hello_ack` (L2).
    fn owner_mode(&self) -> &'static str {
        if self.symmetric() {
            "symmetric"
        } else {
            "pd"
        }
    }

    /// The observed decode capacity this table reads (the collector fills
    /// it under `--pd-native-decode-state`).
    pub fn capacity_store(&self) -> &Arc<DecodeCapacityStore> {
        &self.capacity
    }

    /// Rooms with a live `fit_native` reservation (tests and logs).
    pub fn reserved_rooms(&self) -> usize {
        self.reservations.lock().len()
    }

    /// The charge of a winner seen without a decision, or admitted against
    /// one (gateway `charge_default_winner`): with `fit_native`, also a
    /// reservation for the room of the winner's observed requirement (the
    /// request's charge is an upper bound on its prompt).
    fn charge_default_winner(&self, entry: &LateEntry, idx: usize) -> LateCharge {
        let worker = &entry.candidates[idx];
        if self.config.fit_native {
            let required = self
                .capacity
                .snapshot()
                .get(&worker.url)
                .and_then(|s| s.required(entry.charge_tokens));
            crate::late_fit::charge_native_fallback(
                &self.ledger,
                &self.reservations,
                worker,
                entry.charge_tokens,
                entry.key,
                required,
            )
        } else {
            LateCharge::plain(worker, self.ledger.charge(worker, entry.charge_tokens))
        }
    }

    pub fn config(&self) -> &LateDecodeConfig {
        &self.config
    }

    pub fn router_epoch(&self) -> u32 {
        self.router_epoch
    }

    pub fn router_id(&self) -> &str {
        &self.router_id
    }

    pub fn rule(&self) -> LateDecodeRule {
        self.rule
    }

    pub fn stats(&self) -> &LateStats {
        &self.stats
    }

    pub fn pending(&self) -> usize {
        self.entries.len()
    }

    pub fn link_status(&self, endpoint: &str) -> Option<LinkStatus> {
        self.links.get(endpoint).map(|s| s.value().clone())
    }

    /// The d1 charge of a request: prompt tokens + the capped output budget.
    pub fn request_charge(&self, prompt_tokens: u64, body: Option<&Value>) -> u64 {
        self.ledger.request_charge(prompt_tokens, body)
    }

    fn po2(&self) -> &SmgPowerOfTwo {
        self.po2.as_deref().unwrap_or(&self.po2_fallback)
    }

    /// The candidate set for one request from the choose-capable decode
    /// workers `pool` and the arrival pick `pool[d0]`: all of them, or the
    /// pick and `candidates - 1` others drawn uniformly; pool order is kept.
    /// Returns the candidates and the pick's index among them.
    pub fn candidates(&self, pool: &[Arc<Worker>], d0: usize) -> (Vec<Arc<Worker>>, usize) {
        let k = self
            .config
            .candidates
            .unwrap_or(pool.len())
            .clamp(1, MAX_CANDIDATES);
        if k >= pool.len() {
            return (pool.to_vec(), d0);
        }
        let mut chosen: Vec<usize> = (0..pool.len()).filter(|&i| i != d0).collect();
        chosen.shuffle(&mut rand::thread_rng());
        chosen.truncate(k - 1);
        chosen.push(d0);
        chosen.sort_unstable();
        let default_idx = chosen.iter().position(|&i| i == d0).unwrap_or(0);
        (
            chosen.iter().map(|&i| Arc::clone(&pool[i])).collect(),
            default_idx,
        )
    }

    /// Add a request to the table; the guard removes it.
    pub fn register(self: &Arc<Self>, spec: EntrySpec) -> EntryGuard {
        self.register_with(spec, None)
    }

    /// [`Self::register`] with the request's session KV retention hook:
    /// the arrival pick is recorded as a possible ADMIT target before the
    /// entry enters the table (before any leg leaves).
    pub fn register_with(
        self: &Arc<Self>,
        spec: EntrySpec,
        session: Option<Arc<dyn LateSessionHook>>,
    ) -> EntryGuard {
        let key = spec.key;
        if let (Some(hook), Some(d0)) = (&session, spec.candidates.get(spec.default_idx)) {
            hook.candidate(&d0.url);
        }
        let entry = Arc::new(LateEntry {
            owner: spec.owner,
            prefill_url: spec.prefill_url,
            key,
            rid: spec.rid,
            candidates: spec.candidates,
            default_idx: spec.default_idx,
            charge_tokens: spec.charge_tokens,
            created: Instant::now(),
            dead: AtomicU64::new(0),
            state: Mutex::new(EntryState::default()),
            session,
        });
        if self.entries.insert(key, Arc::clone(&entry)).is_some() {
            tracing::warn!(
                target: LOG_TARGET,
                room = key,
                "late_d: bootstrap room reused; the older entry is replaced"
            );
        }
        inc(&self.stats.requests_choose);
        EntryGuard {
            late: Arc::clone(self),
            entry,
        }
    }

    /// Count a choose-mode request that took the plain two-leg dispatch.
    pub fn count_plain(&self) {
        inc(&self.stats.requests_plain);
    }

    fn entry(&self, owner: &str, key: u64) -> Option<Arc<LateEntry>> {
        let entry = self.entries.get(&key)?;
        (entry.owner == owner).then(|| Arc::clone(entry.value()))
    }

    /// The leg of candidate `idx` ended with an error.
    pub fn mark_dead(&self, entry: &LateEntry, idx: usize) {
        if idx < 64 {
            entry.dead.fetch_or(1u64 << idx, Ordering::Relaxed);
        }
    }

    /// The leg of candidate `idx` answered 2xx and is the stream. Returns
    /// the request's charge, on that candidate: the decision's (moved if P
    /// admitted another candidate) or a new one (no decision: the link was
    /// down or P timed out).
    pub fn bind_charge(&self, entry: &LateEntry, idx: usize) -> LateCharge {
        let worker = &entry.candidates[idx];
        let mut st = entry.state.lock();
        st.bound = Some(idx);
        // P's report wins over the decision (a mismatched final is counted
        // on its own).
        match st.admitted.or(st.decided.map(|d| d.winner)) {
            Some(expected) if expected != idx => inc(&self.stats.bind_mismatch),
            None => inc(&self.stats.bind_undecided),
            _ => {}
        }
        let charge = match st.charge.take() {
            Some((ci, charge)) if ci == idx => charge,
            Some((_, charge)) => charge.move_to(&self.ledger, worker),
            None => self.charge_default_winner(entry, idx),
        };
        drop(st);
        if let Some(hook) = &entry.session {
            hook.bound(&worker.url);
        }
        charge
    }

    /// `(tokens, requests)` charged to `worker` (the shared per-worker
    /// min-inflight-tokens counters).
    pub fn charged(worker: &Worker) -> (u64, usize) {
        (worker.decode_charged_tokens(), worker.decode_charged_reqs())
    }

    // -- decision side ------------------------------------------------------

    fn hello_frame(&self) -> Vec<u8> {
        encode(&HelloMsg {
            v: LINK_VERSION,
            t: "hello",
            router_epoch: self.router_epoch,
            router_id: &self.router_id,
            rules: [self.rule.as_str()],
        })
    }

    /// One frame from the P at `owner`; returns the frames to send back.
    /// `up` turns true on a valid `hello_ack`.
    pub fn on_frame(&self, owner: &str, frame: &[u8], up: &mut bool) -> Vec<Vec<u8>> {
        let msg = match decode_message(frame) {
            Ok(msg) => msg,
            Err(e) => {
                self.bad_message(owner, &e);
                return Vec::new();
            }
        };
        match msg.t.as_str() {
            "hello_ack" => {
                self.on_hello_ack(owner, &msg, up);
                Vec::new()
            }
            "pong" => {
                let rtt_ms = now_ns().saturating_sub(msg.ts_ns) as f64 / 1e6;
                tracing::trace!(target: LOG_TARGET, p = %owner, rtt_ms, "late_d: pong");
                Vec::new()
            }
            "bind_ready" if msg.router_epoch == Some(u64::from(self.router_epoch)) => {
                let items = self.decide(owner, &msg);
                items
                    .chunks(MAX_ITEMS)
                    .map(|chunk| {
                        encode(&DecisionMsg {
                            v: LINK_VERSION,
                            t: "decision",
                            router_epoch: self.router_epoch,
                            items: chunk,
                        })
                    })
                    .collect()
            }
            "final" if msg.router_epoch == Some(u64::from(self.router_epoch)) => {
                for item in &msg.items {
                    self.on_final(owner, item);
                }
                Vec::new()
            }
            "bind_ready" | "final" => {
                inc(&self.stats.stale_epoch);
                tracing::debug!(
                    target: LOG_TARGET,
                    p = %owner,
                    t = %msg.t,
                    epoch = ?msg.router_epoch,
                    "late_d: message of another router epoch dropped"
                );
                Vec::new()
            }
            other => {
                self.bad_message(owner, &format!("unexpected type {other}"));
                Vec::new()
            }
        }
    }

    fn bad_message(&self, owner: &str, why: &str) {
        inc(&self.stats.bad_messages);
        if let Some(mut status) = self.links.get_mut(owner) {
            status.bad_messages += 1;
        }
        tracing::warn!(target: LOG_TARGET, p = %owner, "late_d: link message dropped: {why}");
    }

    fn on_hello_ack(&self, owner: &str, msg: &LinkMessage, up: &mut bool) {
        if msg.router_epoch != Some(u64::from(self.router_epoch)) {
            self.bad_message(owner, "hello_ack for another router epoch");
            return;
        }
        if msg.mode.as_deref() != Some(self.owner_mode()) {
            // A PD table serves PD prefills only, a symmetric table
            // symmetric owners only (L2: a symmetric owner answers mode
            // "symmetric").
            tracing::error!(
                target: LOG_TARGET,
                p = %owner,
                mode = ?msg.mode,
                expected = self.owner_mode(),
                "late_d: link refused: the owner runs another late-D mode"
            );
            return;
        }
        let mut status = self.links.entry(owner.to_string()).or_default();
        if status.hello_acks > 0 && status.engine_boot_ns != msg.engine_boot_ns {
            tracing::warn!(
                target: LOG_TARGET,
                p = %owner,
                "late_d: the prefill restarted (new engine_boot_ns)"
            );
        }
        status.up = true;
        status.hello_acks += 1;
        status.engine_boot_ns = msg.engine_boot_ns;
        status.kv_ip = msg.kv_ip.clone();
        drop(status);
        if !*up {
            tracing::info!(
                target: LOG_TARGET,
                p = %owner,
                kv_ip = ?msg.kv_ip,
                router_epoch = self.router_epoch,
                "late_d: link up"
            );
        }
        *up = true;
    }

    fn set_link_down(&self, owner: &str) {
        if let Some(mut status) = self.links.get_mut(owner) {
            status.up = false;
        }
    }

    /// L4 -> L5: one answer per item, in order.
    fn decide(&self, owner: &str, msg: &LinkMessage) -> Vec<DecisionItem> {
        let t_recv_ns = now_ns();
        let items = msg
            .items
            .iter()
            .map(|item| self.decide_item(owner, item, t_recv_ns, msg.t_send_ns))
            .collect::<Vec<_>>();
        if let Some(mut status) = self.links.get_mut(owner) {
            status.decisions += items.len() as u64;
        }
        items
    }

    fn decide_item(
        &self,
        owner: &str,
        item: &LinkItem,
        t_recv_ns: u64,
        p_t_send_ns: u64,
    ) -> DecisionItem {
        let rule = self.rule.as_str();
        let mut answer = DecisionItem {
            key: item.key,
            bind_seq: item.bind_seq,
            winner_idx: WINNER_UNKNOWN,
            rule,
            t_recv_ns,
            t_decide_ns: 0,
        };
        let Some(entry) = self.entry(owner, item.key) else {
            inc(&self.stats.decisions_unknown_key);
            tracing::debug!(
                target: LOG_TARGET,
                p = %owner,
                room = item.key,
                "late_d: bind_ready for an unknown key"
            );
            answer.t_decide_ns = now_ns();
            return answer;
        };
        let mut st = entry.state.lock();
        if let Some(d) = st.decided.filter(|d| d.bind_seq == item.bind_seq) {
            // L4 repeat: the same answer.
            inc(&self.stats.decisions_repeat);
            answer.winner_idx = d.winner as u8;
            answer.t_decide_ns = d.t_decide_ns;
            return answer;
        }
        if let Some(bound) = st.bound {
            // The stream is bound already: nothing left to choose.
            inc(&self.stats.decisions_bound);
            answer.winner_idx = bound as u8;
            answer.t_decide_ns = now_ns();
            return answer;
        }
        // A new bind_seq replaces an older decision and its charge.
        st.charge = None;
        let Some(choice) = self.choose(&entry, item.eligible, item.input_len) else {
            inc(&self.stats.decisions_no_candidate);
            tracing::warn!(
                target: LOG_TARGET,
                p = %owner,
                room = item.key,
                eligible = item.eligible,
                "late_d: bind_ready names no candidate"
            );
            answer.t_decide_ns = now_ns();
            return answer;
        };
        let t_decide_ns = now_ns();
        // A decode leg that already ended keeps no charge (the choice's
        // charge and fit reservation drop here).
        if !st.ended {
            st.charge = Some((choice.winner, choice.charge));
        }
        st.decided = Some(Decided {
            bind_seq: item.bind_seq,
            winner: choice.winner,
            t_decide_ns,
        });
        drop(st);
        // Session retention: the winner may receive ADMIT with the tag once
        // this answer leaves; record it first so END reaches it.
        if let Some(hook) = &entry.session {
            hook.candidate(&entry.candidates[choice.winner].url);
        }
        inc(&self.stats.decisions_decided);
        if choice.winner != entry.default_idx {
            inc(&self.stats.rebinds);
        }
        let signal_age_ms = match self.rule {
            LateDecodeRule::D1 => 0,
            _ => self
                .po2()
                .loads_age()
                .map_or(-1, |age| age.as_millis() as i64),
        };
        tracing::info!(
            target: LOG_TARGET,
            rid = entry.rid.as_deref().unwrap_or("-"),
            room = item.key,
            p = %entry.prefill_url,
            rule,
            native_capacity_fit = self.config.fit_native,
            k = entry.k(),
            bind_seq = item.bind_seq,
            ready = item.ready,
            input_len = item.input_len,
            first_chunk_start_unix_ns = item.first_chunk_start_unix_ns.unwrap_or(0),
            eligible = %format!("{:#x}", item.eligible),
            considered = %format!("{:#x}", choice.considered),
            default = entry.default_idx,
            winner = choice.winner,
            winner_url = %entry.candidates[choice.winner].url,
            signal = %choice.signal,
            signal_age_ms,
            t_recv_ns,
            t_decide_ns,
            p_t_send_ns,
            wait_ms = entry.age().as_millis() as u64,
            "late_d decision"
        );
        answer.winner_idx = choice.winner as u8;
        answer.t_decide_ns = t_decide_ns;
        answer
    }

    /// The bind-time choice among the eligible candidates the router holds
    /// alive and healthy (else alive, else eligible), with its charge on the
    /// winner. `None` when P named no candidate.
    fn choose(&self, entry: &LateEntry, eligible: u64, input_tokens: u64) -> Option<Choice> {
        let mut eligible = eligible & mask_of(entry.k());
        if eligible == 0 {
            return None;
        }
        // External role control: only candidates still in the decode role.
        if let Some(roles) = self.decode_role_mask(entry) {
            if eligible & !roles != 0 {
                inc(&self.stats.role_filtered);
                if eligible & roles == 0 {
                    inc(&self.stats.role_fallback);
                    tracing::warn!(
                        target: LOG_TARGET,
                        rid = entry.rid.as_deref().unwrap_or("-"),
                        room = entry.key,
                        eligible = %format!("{eligible:#x}"),
                        "late_d: no eligible candidate is still in the decode role; choosing among all eligible"
                    );
                } else {
                    eligible &= roles;
                }
            }
        }
        let alive = eligible & !entry.dead_mask();
        let healthy = bits(alive)
            .filter(|&i| entry.candidates[i].breaker.would_allow())
            .fold(0u64, |m, i| m | (1 << i));
        let considered = [healthy, alive, eligible]
            .into_iter()
            .find(|m| *m != 0)
            .unwrap_or(eligible);
        let set: Vec<usize> = bits(considered).collect();
        let tokens = entry.charge_tokens;
        let po2 = self.po2();
        let polled = |i: usize| {
            let url = entry.candidates[i].url.as_str();
            match po2.token_load(url) {
                Some(load) => load.to_string(),
                None => format!("-/r{}", po2.request_load(url)),
            }
        };
        let plain = |winner: usize, charge| LateCharge::plain(&entry.candidates[winner], charge);
        let (signal, winner, charge) = match self.rule {
            LateDecodeRule::Keep => {
                let signal: Vec<String> =
                    set.iter().map(|&i| format!("{i}:{}", polled(i))).collect();
                let winner = if (considered >> entry.default_idx) & 1 == 1 {
                    entry.default_idx
                } else {
                    set[0]
                };
                let charge = self.ledger.charge(&entry.candidates[winner], tokens);
                (signal, winner, plain(winner, charge))
            }
            LateDecodeRule::Po2 => {
                let signal: Vec<String> =
                    set.iter().map(|&i| format!("{i}:{}", polled(i))).collect();
                let (winner, _) = po2.pick_among(&entry.candidates, &set)?;
                let charge = self.ledger.charge(&entry.candidates[winner], tokens);
                (signal, winner, plain(winner, charge))
            }
            LateDecodeRule::D1 => {
                let signal: Vec<String> = set
                    .iter()
                    .map(|&i| {
                        let (t, r) = Self::charged(&entry.candidates[i]);
                        format!("{i}:{t}/{r}")
                    })
                    .collect();
                let (winner, charge) = self.ledger.select_among(&entry.candidates, &set, tokens)?;
                (signal, winner, plain(winner, charge))
            }
            LateDecodeRule::FitReq if self.config.fit_native => {
                let (winner, charge, fit, signal) = crate::late_fit::select_native_fit_and_charge(
                    &self.ledger,
                    &self.reservations,
                    &entry.candidates,
                    &set,
                    tokens,
                    entry.key,
                    input_tokens,
                    &self.capacity.snapshot(),
                    Duration::from_millis(self.config.fit_state_max_age_ms),
                )?;
                if !fit {
                    inc(&self.stats.fit_none);
                }
                (vec![signal], winner, charge)
            }
            LateDecodeRule::FitReq => {
                // A fixed capacity on the router charges.
                let cap = self.config.fit_capacity_tokens.unwrap_or(u64::MAX);
                let signal: Vec<String> = set
                    .iter()
                    .map(|&i| {
                        let (t, r) = Self::charged(&entry.candidates[i]);
                        let mut s = format!("{i}:{t}/{r}");
                        if t.saturating_add(tokens) > cap {
                            s.push('!');
                        }
                        s
                    })
                    .collect();
                let (winner, charge, fit) =
                    self.ledger
                        .select_fit_fewest_requests(&entry.candidates, &set, tokens, cap)?;
                if !fit {
                    inc(&self.stats.fit_none);
                }
                (signal, winner, plain(winner, charge))
            }
        };
        Some(Choice {
            winner,
            charge,
            considered,
            signal: signal.join(","),
        })
    }

    /// L6: P admitted `admitted_idx`. A mismatch with the decision (P timed
    /// out, or had no answer) moves the charge before a leg binds.
    fn on_final(&self, owner: &str, item: &LinkItem) {
        let reason = item.reason.clone().unwrap_or_default();
        match reason.as_str() {
            "decided" => inc(&self.stats.finals_decided),
            "timeout" => inc(&self.stats.finals_timeout),
            "no_router" => inc(&self.stats.finals_no_router),
            "unknown_key" => inc(&self.stats.finals_unknown_key),
            _ => inc(&self.stats.finals_other),
        }
        if let Some(mut status) = self.links.get_mut(owner) {
            status.finals += 1;
        }
        let Some(entry) = self.entry(owner, item.key) else {
            tracing::debug!(
                target: LOG_TARGET,
                p = %owner,
                room = item.key,
                %reason,
                "late_d: final for a key that left the table"
            );
            return;
        };
        let admitted = item.admitted_idx as usize;
        if admitted >= entry.k() {
            tracing::warn!(
                target: LOG_TARGET,
                p = %owner,
                room = item.key,
                admitted,
                "late_d: final names no candidate"
            );
            return;
        }
        let mut st = entry.state.lock();
        st.admitted = Some(admitted);
        let decided = st.decided.map(|d| d.winner);
        let mismatch = decided != Some(admitted);
        if mismatch {
            inc(&self.stats.final_mismatch);
            if st.bound.is_none() && !st.ended {
                let worker = &entry.candidates[admitted];
                st.charge = Some(match st.charge.take() {
                    Some((_, charge)) => (admitted, charge.move_to(&self.ledger, worker)),
                    None => (admitted, self.charge_default_winner(&entry, admitted)),
                });
            }
        }
        drop(st);
        // Session retention: P admitted (and sent the tag to) a candidate
        // that was not the decision; record it so END and its release reach
        // it even if its leg fails.
        if let Some(hook) = entry.session.as_ref().filter(|_| mismatch) {
            hook.candidate(&entry.candidates[admitted].url);
        }
        tracing::info!(
            target: LOG_TARGET,
            rid = entry.rid.as_deref().unwrap_or("-"),
            room = item.key,
            p = %entry.prefill_url,
            bind_seq = item.bind_seq,
            admitted,
            admitted_url = %entry.candidates[admitted].url,
            decided = ?decided,
            reason = %reason,
            mismatch,
            t_admit_ns = item.t_admit_ns,
            "late_d final"
        );
    }

    // -- links --------------------------------------------------------------

    /// Start the link supervisor on its own runtime thread (once). Every
    /// second it starts a link task for each registered prefill whose
    /// `/server_info` names a late-D control port, and stops the tasks of
    /// prefills that left.
    pub fn start(self: &Arc<Self>) {
        if !self.config.choose() || self.started.swap(true, Ordering::SeqCst) {
            return;
        }
        let late = Arc::clone(self);
        let spawned = std::thread::Builder::new()
            .name("late-d-link".to_string())
            .spawn(move || {
                match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime.block_on(late.supervise()),
                    Err(e) => tracing::error!(
                        target: LOG_TARGET,
                        "late_d: the link runtime failed to start: {e}"
                    ),
                }
            });
        if let Err(e) = spawned {
            tracing::error!(target: LOG_TARGET, "late_d: the link thread failed to start: {e}");
        }
    }

    async fn supervise(self: Arc<Self>) {
        let mut tasks: HashMap<String, JoinHandle<()>> = HashMap::new();
        let mut ticker = tokio::time::interval(SUPERVISE_INTERVAL);
        loop {
            ticker.tick().await;
            let mut want: HashMap<String, String> = HashMap::new();
            let owners: Vec<Arc<Worker>> = match &self.owners {
                LateOwners::Pd => self
                    .registry
                    .all()
                    .into_iter()
                    .filter(|w| w.mode() == WorkerMode::Prefill)
                    .collect(),
                LateOwners::Symmetric(list) => list()
                    .into_iter()
                    .filter(|w| w.mode() == WorkerMode::Symmetric)
                    .collect(),
            };
            for worker in owners {
                if let Some(endpoint) = link_endpoint(&worker) {
                    want.entry(endpoint).or_insert_with(|| worker.url.clone());
                }
            }
            tasks.retain(|endpoint, handle| {
                let keep = want.contains_key(endpoint);
                if !keep {
                    handle.abort();
                    self.set_link_down(endpoint);
                    self.links.remove(endpoint);
                    tracing::info!(
                        target: LOG_TARGET,
                        endpoint = %endpoint,
                        "late_d: link stopped (the prefill left)"
                    );
                }
                keep
            });
            for (endpoint, url) in want {
                if tasks.contains_key(&endpoint) {
                    continue;
                }
                tracing::info!(target: LOG_TARGET, p = %url, endpoint = %endpoint, "late_d: link started");
                self.links.entry(endpoint.clone()).or_default();
                let handle = tokio::spawn(run_link(Arc::clone(&self), endpoint.clone()));
                tasks.insert(endpoint, handle);
            }
        }
    }

    /// One link session: connect, hello, then serve until the link fails or
    /// goes silent. Returns whether the link came up, and why it ended.
    async fn link_session(&self, endpoint: &str) -> (bool, String) {
        let mut socket = DealerSocket::new();
        match tokio::time::timeout(CONNECT_TIMEOUT, socket.connect(endpoint)).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return (false, format!("connect: {e}")),
            Err(_) => return (false, "connect: timed out".to_string()),
        }
        let (mut tx, mut rx) = socket.split();
        let (frames_tx, mut frames) = mpsc::channel::<Bytes>(256);
        // Unconstrained: the tokio budget must never stall the receive side
        // (a stalled receive side livelocks the bridge).
        let reader = tokio::spawn(tokio::task::unconstrained(async move {
            while let Ok(msg) = rx.recv().await {
                let Some(frame) = msg.into_vec().pop() else {
                    continue;
                };
                if frames_tx.send(frame).await.is_err() {
                    break;
                }
            }
        }));
        let mut up = false;
        let why = self
            .serve_link(endpoint, &mut tx, &mut frames, &mut up)
            .await;
        reader.abort();
        (up, why)
    }

    /// Serve one connected link; returns why it ended.
    async fn serve_link(
        &self,
        endpoint: &str,
        tx: &mut DealerSendHalf,
        frames: &mut mpsc::Receiver<Bytes>,
        up: &mut bool,
    ) -> String {
        let mut last_heard = Instant::now();
        if let Err(e) = send_frame(tx, self.hello_frame()).await {
            return e;
        }
        let mut ticker = tokio::time::interval(PING_INTERVAL);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
        ticker.tick().await;
        loop {
            tokio::select! {
                frame = frames.recv() => {
                    let Some(frame) = frame else {
                        return "the socket closed".to_string();
                    };
                    last_heard = Instant::now();
                    for reply in self.on_frame(endpoint, &frame, up) {
                        if let Err(e) = send_frame(tx, reply).await {
                            return e;
                        }
                    }
                }
                _ = ticker.tick() => {
                    if last_heard.elapsed() >= LINK_SILENT {
                        return if *up {
                            "silent for 3 s".to_string()
                        } else {
                            "no hello_ack".to_string()
                        };
                    }
                    let frame = if *up { ping_frame(now_ns()) } else { self.hello_frame() };
                    if let Err(e) = send_frame(tx, frame).await {
                        return e;
                    }
                }
            }
        }
    }

    /// Prometheus text for `/metrics` (choose mode).
    pub fn render_metrics(&self) -> String {
        let s = &self.stats;
        let mut out = String::new();
        let mut family = |name: &str, kind: &str, help: &str, rows: &[(&str, u64)]| {
            out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} {kind}\n"));
            for (labels, v) in rows {
                if labels.is_empty() {
                    out.push_str(&format!("{name} {v}\n"));
                } else {
                    out.push_str(&format!("{name}{{{labels}}} {v}\n"));
                }
            }
        };
        family(
            "sgl_router_late_d_requests_total",
            "counter",
            "Choose-mode PD requests by dispatch path (choose = K candidate legs; plain = no link or no choose decode).",
            &[
                ("path=\"choose\"", get(&s.requests_choose)),
                ("path=\"plain\"", get(&s.requests_plain)),
            ],
        );
        family(
            "sgl_router_late_d_decisions_total",
            "counter",
            "bind_ready items answered, by outcome.",
            &[
                ("outcome=\"decided\"", get(&s.decisions_decided)),
                ("outcome=\"repeat\"", get(&s.decisions_repeat)),
                ("outcome=\"bound\"", get(&s.decisions_bound)),
                ("outcome=\"unknown_key\"", get(&s.decisions_unknown_key)),
                ("outcome=\"no_candidate\"", get(&s.decisions_no_candidate)),
            ],
        );
        family(
            "sgl_router_late_d_role_gate_total",
            "counter",
            "External role control: decisions that left out candidates no longer in the decode role (filtered), and decisions where none was left (fallback to every eligible candidate).",
            &[
                ("outcome=\"filtered\"", get(&s.role_filtered)),
                ("outcome=\"fallback\"", get(&s.role_fallback)),
            ],
        );
        family(
            "sgl_router_late_d_rebinds_total",
            "counter",
            "Decisions whose winner is not the arrival pick.",
            &[("", get(&s.rebinds))],
        );
        family(
            "sgl_router_late_d_fit_none_total",
            "counter",
            "fit_req decisions where no eligible candidate fit (or the observed capacity was missing or stale), so the d1 order decided.",
            &[("", get(&s.fit_none))],
        );
        family(
            "sgl_router_late_d_finals_total",
            "counter",
            "final items from the prefills, by reason.",
            &[
                ("reason=\"decided\"", get(&s.finals_decided)),
                ("reason=\"timeout\"", get(&s.finals_timeout)),
                ("reason=\"no_router\"", get(&s.finals_no_router)),
                ("reason=\"unknown_key\"", get(&s.finals_unknown_key)),
                ("reason=\"other\"", get(&s.finals_other)),
            ],
        );
        family(
            "sgl_router_late_d_mismatch_total",
            "counter",
            "Bind outcomes that differ from the decision: final = P admitted another candidate; bind = the 2xx leg is not the one P admitted (without a final: not the decided one); undecided = a leg bound with no decision and no final.",
            &[
                ("kind=\"final\"", get(&s.final_mismatch)),
                ("kind=\"bind\"", get(&s.bind_mismatch)),
                ("kind=\"undecided\"", get(&s.bind_undecided)),
            ],
        );
        family(
            "sgl_router_late_d_legs_total",
            "counter",
            "Decode candidate legs by outcome (winner = the 2xx stream; released = 409 late_d_released; dead = another failure; conflict = a second 2xx; not_sent = breaker refused).",
            &[
                ("outcome=\"winner\"", get(&s.legs_winner)),
                ("outcome=\"released\"", get(&s.legs_released)),
                ("outcome=\"dead\"", get(&s.legs_dead)),
                ("outcome=\"conflict\"", get(&s.legs_conflict)),
                ("outcome=\"not_sent\"", get(&s.legs_not_sent)),
            ],
        );
        family(
            "sgl_router_late_d_failures_total",
            "counter",
            "Choose-mode failures: no_winner = every candidate ended without 2xx; prefill = the P leg failed before any bind (the client got its error); prefill_leg = every failed P leg, before or after the bind; prefill_reader_gone = a P leg the engine ended because its reader left (not a failure); loser_drain = a loser leg still open 30 s after the bind.",
            &[
                ("kind=\"no_winner\"", get(&s.no_winner)),
                ("kind=\"prefill\"", get(&s.prefill_failed)),
                ("kind=\"prefill_leg\"", get(&s.prefill_leg_failed)),
                ("kind=\"prefill_reader_gone\"", get(&s.prefill_reader_gone)),
                ("kind=\"loser_drain\"", get(&s.loser_drain_timeouts)),
            ],
        );
        family(
            "sgl_router_late_d_link_messages_dropped_total",
            "counter",
            "Link messages dropped: stale_epoch = another router epoch; bad = undecodable or unexpected.",
            &[
                ("kind=\"stale_epoch\"", get(&s.stale_epoch)),
                ("kind=\"bad\"", get(&s.bad_messages)),
            ],
        );
        family(
            "sgl_router_late_d_pending",
            "gauge",
            "Choose-mode requests in the late-bind table.",
            &[("", self.entries.len() as u64)],
        );
        let mut links: Vec<(String, bool)> = self
            .links
            .iter()
            .map(|e| (e.key().clone(), e.value().up))
            .collect();
        links.sort();
        let rows: Vec<(String, u64)> = links
            .iter()
            .map(|(p, up)| (format!("p=\"{}\"", escape_label(p)), u64::from(*up)))
            .collect();
        let rows: Vec<(&str, u64)> = rows.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        family(
            "sgl_router_late_d_link_up",
            "gauge",
            "1 while the link to the prefill rank 0 is up (hello_ack received).",
            &rows,
        );
        out
    }
}

async fn send_frame(tx: &mut DealerSendHalf, frame: Vec<u8>) -> Result<(), String> {
    match tokio::time::timeout(SEND_TIMEOUT, tx.send(ZmqMessage::from(frame))).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(format!("send: {e}")),
        Err(_) => Err("send: timed out".to_string()),
    }
}

/// The link task of one P: sessions with a reconnect backoff, until aborted.
pub async fn run_link(late: Arc<LateBind>, endpoint: String) {
    let mut backoff = RECONNECT_MIN;
    let mut failures = 0u64;
    loop {
        let (was_up, why) = late.link_session(&endpoint).await;
        late.set_link_down(&endpoint);
        if was_up {
            tracing::warn!(
                target: LOG_TARGET,
                p = %endpoint,
                "late_d: link lost ({why}); reconnecting"
            );
            backoff = RECONNECT_MIN;
            failures = 0;
        } else {
            failures += 1;
            if failures == 1 || failures.is_multiple_of(60) {
                tracing::warn!(
                    target: LOG_TARGET,
                    p = %endpoint,
                    failures,
                    "late_d: link not up: {why}"
                );
            } else {
                tracing::debug!(
                    target: LOG_TARGET,
                    p = %endpoint,
                    failures,
                    "late_d: link not up: {why}"
                );
            }
        }
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(RECONNECT_MAX);
    }
}

/// The end of a loser leg after the bind. A release (409 with the reason
/// header) is the normal end; a second 2xx is a bug (counted and dropped);
/// another failure ticks the breaker unless it carries a late-D reason.
pub async fn account_loser(
    late: &LateBind,
    worker: &Worker,
    result: Result<reqwest::Response, String>,
) {
    let stats = late.stats();
    match result {
        Ok(res) if res.status().is_success() => {
            inc(&stats.legs_conflict);
            tracing::error!(
                target: LOG_TARGET,
                decode_url = %worker.url,
                "late_d: a second decode leg answered 2xx; dropped"
            );
        }
        Ok(res) => {
            let status = res.status();
            let reason = late_reason(res.headers());
            if status == reqwest::StatusCode::CONFLICT && reason.as_deref() == Some(RELEASED_REASON)
            {
                inc(&stats.legs_released);
                settle_late_answer(&worker.breaker);
            } else {
                inc(&stats.legs_dead);
                match reason {
                    Some(_) => settle_late_answer(&worker.breaker),
                    None if status.is_server_error() => worker.breaker.record_failure(),
                    None => worker.breaker.record_success(),
                }
                tracing::warn!(
                    target: LOG_TARGET,
                    decode_url = %worker.url,
                    %status,
                    reason = ?reason,
                    "late_d: loser leg failed"
                );
            }
            let _ = tokio::time::timeout(Duration::from_secs(5), res.bytes()).await;
        }
        Err(e) => {
            inc(&stats.legs_dead);
            worker.breaker.record_failure();
            tracing::warn!(
                target: LOG_TARGET,
                decode_url = %worker.url,
                error = %e,
                "late_d: loser leg failed"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use zeromq::{Endpoint, RouterSendHalf, RouterSocket};

    use super::*;
    use crate::config::LateDecodeMode;
    use crate::discovery::{ModelId, WorkerId, WorkerSpec};

    fn worker(url: &str, mode: WorkerMode, late: Option<LateDWorkerInfo>) -> Arc<Worker> {
        let w = Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(url.into()),
            url: url.into(),
            mode,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: (mode == WorkerMode::Prefill).then_some(8998),
        }));
        w.set_late_d(late);
        w
    }

    fn choose_info(port: Option<u16>) -> LateDWorkerInfo {
        LateDWorkerInfo {
            mode: CHOOSE_MODE.into(),
            control_port: port,
            kv_ip: Some("10.0.0.9".into()),
        }
    }

    fn decode_worker(url: &str) -> Arc<Worker> {
        worker(url, WorkerMode::Decode, Some(choose_info(None)))
    }

    fn late_cfg(rule: LateDecodeRule, candidates: Option<usize>) -> LateDecodeConfig {
        LateDecodeConfig {
            mode: LateDecodeMode::Choose,
            rule: Some(rule),
            candidates,
            ..LateDecodeConfig::default()
        }
    }

    fn late(rule: LateDecodeRule, candidates: Option<usize>) -> Arc<LateBind> {
        LateBind::new(
            late_cfg(rule, candidates),
            PdDecodeTokensConfig::default(),
            Arc::new(WorkerRegistry::default()),
            None,
        )
    }

    fn pool(n: usize) -> Vec<Arc<Worker>> {
        (0..n)
            .map(|i| decode_worker(&format!("http://d{i}:8000")))
            .collect()
    }

    fn item(key: u64, bind_seq: u64, eligible: u64) -> LinkItem {
        LinkItem {
            key,
            bind_seq,
            eligible,
            ready: 2,
            input_len: 100,
            ..Default::default()
        }
    }

    fn msg(value: Value) -> Vec<u8> {
        rmp_serde::to_vec_named(&value).unwrap()
    }

    #[test]
    fn codec_round_trip_and_limits() {
        let l = late(LateDecodeRule::Keep, None);
        let hello = decode_message(&l.hello_frame()).unwrap();
        assert_eq!(hello.t, "hello");
        assert_eq!(hello.router_epoch, Some(u64::from(l.router_epoch())));
        assert_eq!(hello.rules, vec!["keep".to_string()]);
        assert!(hello.router_id.unwrap().starts_with("sgl-router-"));
        let ping = decode_message(&ping_frame(42)).unwrap();
        assert_eq!((ping.t.as_str(), ping.ts_ns), ("ping", 42));

        // What the engine sends (msgspec maps, Python field names).
        let ready = msg(json!({
            "v": 1, "t": "bind_ready", "router_epoch": 7, "p_seq": 3, "t_send_ns": 99,
            "items": [{"key": u64::MAX >> 1, "bind_seq": 2, "eligible": 5u64, "ready": 2,
                       "input_len": 1000, "default_idx": 2, "local_ok": false, "local_load": 0}],
        }));
        let ready = decode_message(&ready).unwrap();
        assert_eq!(ready.items.len(), 1);
        assert_eq!(ready.items[0].key, u64::MAX >> 1);
        assert_eq!(ready.items[0].eligible, 5);
        assert_eq!(ready.t_send_ns, 99);
        let ack = decode_message(&msg(json!({
            "v": 1, "t": "hello_ack", "router_epoch": 7, "engine_boot_ns": 5,
            "mode": "pd", "kv_ip": null,
        })))
        .unwrap();
        assert_eq!((ack.mode.as_deref(), ack.kv_ip), (Some("pd"), None));
        // A full 64-bit eligible mask survives.
        let wide = decode_message(&msg(json!({
            "v": 1, "t": "bind_ready", "router_epoch": 7,
            "items": [{"key": 1, "bind_seq": 1, "eligible": u64::MAX}],
        })))
        .unwrap();
        assert_eq!(wide.items[0].eligible, u64::MAX);

        assert!(decode_message(&msg(json!({"v": 2, "t": "ping"}))).is_err());
        assert!(decode_message(&msg(json!({"v": 1}))).is_err());
        assert!(decode_message(&msg(json!([1, 2]))).is_err());
        assert!(decode_message(b"\xc1").is_err());
        let big = msg(json!({"v": 1, "t": "ping", "pad": "x".repeat(MAX_MESSAGE_BYTES)}));
        assert!(decode_message(&big).unwrap_err().contains("limit"));

        // A full decision batch stays under the limit and reads back with
        // the engine's field names.
        let items: Vec<DecisionItem> = (0..MAX_ITEMS as u64)
            .map(|i| DecisionItem {
                key: u64::MAX >> 1,
                bind_seq: u64::from(u32::MAX) + i,
                winner_idx: 63,
                rule: "keep",
                t_recv_ns: u64::MAX,
                t_decide_ns: u64::MAX,
            })
            .collect();
        let frame = encode(&DecisionMsg {
            v: LINK_VERSION,
            t: "decision",
            router_epoch: u32::MAX,
            items: &items,
        });
        assert!(frame.len() < MAX_MESSAGE_BYTES, "{} bytes", frame.len());
        let back = decode_message(&frame).unwrap();
        assert_eq!(back.items.len(), MAX_ITEMS);
        assert_eq!(back.items[7].winner_idx, 63);
        assert_eq!(back.items[7].rule.as_deref(), Some("keep"));
        // The map keys are exactly the engine's (msgspec reads them by name).
        let raw: Value = rmp_serde::from_slice(&frame).unwrap();
        let keys: HashSet<&str> = raw["items"][0]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            HashSet::from([
                "key",
                "bind_seq",
                "winner_idx",
                "rule",
                "t_recv_ns",
                "t_decide_ns"
            ])
        );
        assert_eq!(raw["t"], "decision");
        assert_eq!(raw["v"], 1);
    }

    #[test]
    fn leg_bodies_add_the_candidate_field() {
        let base = json!({"model": "m", "prompt": [1, 2, 3], "stream": true, "bootstrap_room": 9});
        let legs = LegBodies::new(&base, 77, 3).unwrap();
        assert_eq!(legs.k(), 3);
        for idx in 0..3 {
            let body: Value = serde_json::from_slice(&legs.leg(idx)).unwrap();
            let mut want = base.clone();
            want["late_d"] = decode_field(77, 3, idx);
            assert_eq!(body, want);
        }
        let empty = LegBodies::new(&json!({}), 1, 1).unwrap();
        let body: Value = serde_json::from_slice(&empty.leg(0)).unwrap();
        assert_eq!(body, json!({"late_d": decode_field(1, 1, 0)}));
        assert!(LegBodies::new(&json!([1]), 1, 1).is_err());
        assert!(LegBodies::from_object_bytes(b"[1]", 1, 1).is_err());
        assert!(LegBodies::from_object_bytes(b"", 1, 1).is_err());
        // From bytes, with trailing whitespace and a nested empty object last.
        let legs = LegBodies::from_object_bytes(b"{\"a\":{}} \n", 5, 2).unwrap();
        let body: Value = serde_json::from_slice(&legs.leg(1)).unwrap();
        assert_eq!(body, json!({"a": {}, "late_d": decode_field(5, 2, 1)}));
        assert!(LegBodies::from_object_bytes(b"}", 1, 1).is_err());
        assert!(LegBodies::from_object_bytes(b"\"{}\"", 1, 1).is_err());
        let legs = LegBodies::from_object_bytes(b" {\"b\":[1]}", 5, 2).unwrap();
        let body: Value = serde_json::from_slice(&legs.leg(0)).unwrap();
        assert_eq!(body, json!({"b": [1], "late_d": decode_field(5, 2, 0)}));
        let legs = LegBodies::from_object_bytes(b"{ }", 5, 2).unwrap();
        let body: Value = serde_json::from_slice(&legs.leg(0)).unwrap();
        assert_eq!(body, json!({"late_d": decode_field(5, 2, 0)}));
        assert_eq!(
            prefill_field(5, 4, 2),
            json!({"v": 1, "mode": "choose", "router_epoch": 5, "k": 4, "default_idx": 2})
        );
    }

    #[test]
    fn base64_matches_rfc_4648_vectors() {
        for (input, want) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64_encode(input.as_bytes()), want, "{input}");
        }
        assert_eq!(base64_encode(&[0xff, 0xfe, 0xfd, 0x00]), "//79AA==");
        let all: Vec<u8> = (0..=255).collect();
        let enc = base64_encode(&all);
        assert_eq!(enc.len(), 344);
        assert!(enc.starts_with("AAECAwQFBgcICQoL"));
        assert!(enc.ends_with("+/w=="));
    }

    #[test]
    fn packed_prompt_equals_the_json_ids() {
        let ids: Vec<u64> = vec![0, 1, 151_643, i32::MAX as u64];
        let mut body = json!({"prompt": ids, "max_tokens": 4, "stream": true});
        assert!(pack_prompt(&mut body));
        assert!(body.get("prompt").is_none());
        assert_eq!(body["prompt_len"], json!(4));
        // Little-endian int32: 0, 1, 151643 (0x0002_505b), i32::MAX; the
        // string Python's base64.b64encode(struct.pack("<4i", ...)) gives.
        assert_eq!(
            body["prompt_ids_b64"],
            json!(base64_encode(&[
                0, 0, 0, 0, 1, 0, 0, 0, 0x5b, 0x50, 0x02, 0, 0xff, 0xff, 0xff, 0x7f
            ]))
        );
        assert_eq!(body["prompt_ids_b64"], "AAAAAAEAAABbUAIA////fw==");
        // Packed once only.
        assert!(!pack_prompt(&mut body));

        for refused in [
            json!({"prompt": "text"}),
            json!({"prompt": []}),
            json!({"prompt": [[1, 2], [3]]}),
            json!({"prompt": [1, 2147483648u64]}),
            json!({"prompt": [1, 2], "echo": true}),
            json!({"prompt": [1, 2], "n": 2}),
            json!({"prompt": [1, 2], "strict_replay_deadline_audit": true}),
        ] {
            let mut body = refused.clone();
            assert!(!pack_prompt(&mut body), "{refused}");
            assert_eq!(body, refused);
        }
        let mut n1 = json!({"prompt": [5], "n": 1, "echo": false});
        assert!(pack_prompt(&mut n1));
        assert!(is_batch_prompt(&json!({"prompt": [[1], [2]]})));
        assert!(is_batch_prompt(&json!({"prompt": ["a", "b"]})));
        assert!(!is_batch_prompt(&json!({"prompt": [1, 2]})));
        assert!(!is_batch_prompt(&json!({"prompt": "a"})));
    }

    #[test]
    fn link_endpoint_and_capability() {
        let p = worker(
            "http://10.0.0.1:30000",
            WorkerMode::Prefill,
            Some(choose_info(Some(31000))),
        );
        assert_eq!(link_endpoint(&p).as_deref(), Some("tcp://10.0.0.1:31000"));
        let alloc = worker(
            "http://10.0.0.1:30000",
            WorkerMode::Prefill,
            Some(LateDWorkerInfo {
                mode: "alloc".into(),
                control_port: Some(31000),
                kv_ip: None,
            }),
        );
        assert_eq!(link_endpoint(&alloc), None);
        let no_port = worker(
            "http://10.0.0.1:30000",
            WorkerMode::Prefill,
            Some(choose_info(None)),
        );
        assert_eq!(link_endpoint(&no_port), None);
        let v6 = worker(
            "http://[fd00::1]:30000",
            WorkerMode::Prefill,
            Some(choose_info(Some(31000))),
        );
        assert_eq!(link_endpoint(&v6).as_deref(), Some("tcp://[fd00::1]:31000"));
        assert!(is_choose_decode(&decode_worker("http://d:1")));
        assert!(!is_choose_decode(&worker(
            "http://d:1",
            WorkerMode::Decode,
            None
        )));
    }

    #[test]
    fn registration_rules() {
        use RegistrationCheck::*;
        let choose_p = choose_info(Some(31000));
        let choose_d = choose_info(None);
        let alloc = LateDWorkerInfo {
            mode: "alloc".into(),
            control_port: Some(1),
            kv_ip: None,
        };
        assert_eq!(
            registration_check(WorkerMode::Prefill, true, Some(&choose_p)),
            Pass
        );
        assert_eq!(
            registration_check(WorkerMode::Decode, true, Some(&choose_d)),
            Pass
        );
        assert_eq!(
            registration_check(WorkerMode::Prefill, true, Some(&choose_d)),
            Refuse(Refusal::NoControlPort)
        );
        assert_eq!(
            registration_check(WorkerMode::Decode, true, None),
            Refuse(Refusal::NoLateD)
        );
        assert_eq!(
            registration_check(WorkerMode::Decode, true, Some(&alloc)),
            Refuse(Refusal::Mode("alloc".into()))
        );
        assert_eq!(
            registration_check(WorkerMode::Plain, true, None),
            NotChecked
        );
        assert_eq!(
            registration_check(WorkerMode::Symmetric, true, None),
            NotChecked
        );
        assert_eq!(
            registration_check(WorkerMode::Prefill, false, Some(&choose_p)),
            Unread
        );
        assert_eq!(Refusal::Mode("alloc".into()).reason(), "mode");
        assert_eq!(
            Refusal::NoControlPort.to_string(),
            "the prefill's late_d has no control_port"
        );
        count_refusal(&Refusal::NoLateD);
        assert!(render_static_metrics()
            .contains("sgl_router_late_d_worker_refused_total{reason=\"no_late_d\"}"));
    }

    #[test]
    fn candidates_keep_the_arrival_pick() {
        let all = late(LateDecodeRule::Keep, None);
        let ws = pool(5);
        let (c, d) = all.candidates(&ws, 3);
        assert_eq!((c.len(), d), (5, 3));
        let two = late(LateDecodeRule::Keep, Some(2));
        let mut seen = HashSet::new();
        for _ in 0..200 {
            let (c, d) = two.candidates(&ws, 3);
            assert_eq!(c.len(), 2);
            assert_eq!(c[d].url, "http://d3:8000");
            assert!(c[0].url < c[1].url, "pool order kept");
            seen.insert(c[1 - d].url.clone());
        }
        assert_eq!(seen.len(), 4, "the other candidate is drawn uniformly");
        let one = late(LateDecodeRule::Keep, Some(1));
        let (c, d) = one.candidates(&ws, 4);
        assert_eq!((c.len(), d, c[0].url.as_str()), (1, 0, "http://d4:8000"));
    }

    fn register_on(
        l: &Arc<LateBind>,
        key: u64,
        candidates: Vec<Arc<Worker>>,
        default_idx: usize,
    ) -> EntryGuard {
        l.register(EntrySpec {
            owner: "tcp://p:1".to_string(),
            prefill_url: "http://p:1".to_string(),
            key,
            rid: Some(format!("rid-{key}")),
            candidates,
            default_idx,
            charge_tokens: 1000,
        })
    }

    #[test]
    fn keep_decides_once_and_repeats() {
        let l = late(LateDecodeRule::Keep, None);
        let ws = pool(3);
        let guard = register_on(&l, 11, ws.clone(), 1);
        let entry = Arc::clone(guard.entry());
        let a = l.decide_item("tcp://p:1", &item(11, 1, 0b111), 1, 0);
        assert_eq!((a.winner_idx, a.rule), (1, "keep"));
        assert_eq!(LateBind::charged(&ws[1]).0, 1000);
        // L4 repeat: the same answer, no second charge.
        let b = l.decide_item("tcp://p:1", &item(11, 1, 0b111), 2, 0);
        assert_eq!((b.winner_idx, b.t_decide_ns), (1, a.t_decide_ns));
        assert_eq!(LateBind::charged(&ws[1]).1, 1);
        // Another link, or an unknown key: 255.
        assert_eq!(
            l.decide_item("tcp://q:1", &item(11, 1, 0b111), 3, 0)
                .winner_idx,
            WINNER_UNKNOWN
        );
        assert_eq!(
            l.decide_item("tcp://p:1", &item(12, 1, 0b111), 3, 0)
                .winner_idx,
            WINNER_UNKNOWN
        );
        // The default is not eligible: the lowest eligible.
        let w2 = pool(3);
        let g2 = register_on(&l, 13, w2, 0);
        assert_eq!(
            l.decide_item("tcp://p:1", &item(13, 1, 0b110), 4, 0)
                .winner_idx,
            1
        );
        // A dead candidate is skipped while another is alive.
        let w3 = pool(3);
        let g3 = register_on(&l, 14, w3, 2);
        l.mark_dead(g3.entry(), 2);
        assert_eq!(
            l.decide_item("tcp://p:1", &item(14, 1, 0b101), 5, 0)
                .winner_idx,
            0
        );
        // No eligible candidate (bit beyond k): 255.
        let g4 = register_on(&l, 15, pool(2), 0);
        assert_eq!(
            l.decide_item("tcp://p:1", &item(15, 1, 0b100), 6, 0)
                .winner_idx,
            WINNER_UNKNOWN
        );
        drop((g2, g3, g4));
        assert_eq!(entry.decided(), Some(1));
        // Dropping the guard removes the entry and releases the charge.
        drop(guard);
        drop(entry);
        assert_eq!(l.pending(), 0);
        assert_eq!(LateBind::charged(&ws[1]), (0, 0));
        assert_eq!(get(&l.stats().decisions_decided), 3);
        assert_eq!(get(&l.stats().decisions_repeat), 1);
        assert_eq!(get(&l.stats().decisions_unknown_key), 2);
        assert_eq!(get(&l.stats().decisions_no_candidate), 1);
        assert_eq!(get(&l.stats().rebinds), 2);
    }

    #[test]
    fn d1_picks_the_fewest_charged_tokens_among_the_eligible() {
        let l = late(LateDecodeRule::D1, None);
        let ws = pool(3);
        let busy = ws[0].charge_decode_tokens(50_000);
        let lighter = ws[2].charge_decode_tokens(10);
        let g = register_on(&l, 21, ws.clone(), 0);
        // d1 is the least charged, but not eligible: d2 beats d0.
        let a = l.decide_item("tcp://p:1", &item(21, 1, 0b101), 1, 0);
        assert_eq!((a.winner_idx, a.rule), (2, "d1"));
        assert_eq!(LateBind::charged(&ws[2]).0, 1010);
        // A new bind_seq re-decides; the old charge goes first.
        let b = l.decide_item("tcp://p:1", &item(21, 2, 0b111), 2, 0);
        assert_eq!(b.winner_idx, 1);
        assert_eq!(LateBind::charged(&ws[2]).0, 10);
        assert_eq!(LateBind::charged(&ws[1]).0, 1000);
        drop((g, busy, lighter));
        assert_eq!(LateBind::charged(&ws[1]).0, 0);
        // Ties: fewer requests, then URL order.
        let ws = pool(3);
        let _a = ws[0].charge_decode_tokens(500);
        let _b1 = ws[1].charge_decode_tokens(250);
        let _b2 = ws[1].charge_decode_tokens(250);
        let _c = ws[2].charge_decode_tokens(500);
        let g = register_on(&l, 22, ws.clone(), 2);
        // d0 and d2 tie on tokens and requests: URL order -> d0; d1 holds two.
        assert_eq!(
            l.decide_item("tcp://p:1", &item(22, 1, 0b111), 1, 0)
                .winner_idx,
            0
        );
        drop(g);
        // A dead candidate is excluded even when it is the lightest.
        let ws = pool(2);
        let _heavy = ws[1].charge_decode_tokens(9_999);
        let g = register_on(&l, 23, ws.clone(), 1);
        l.mark_dead(g.entry(), 0);
        assert_eq!(
            l.decide_item("tcp://p:1", &item(23, 1, 0b11), 1, 0)
                .winner_idx,
            1
        );
        drop(g);
    }

    #[test]
    fn d1_spreads_concurrent_binds() {
        // Four decisions in a row with nothing bound: each sees the charges
        // of the decisions before it.
        let l = late(LateDecodeRule::D1, None);
        let ws = pool(2);
        let mut guards = Vec::new();
        let mut winners = Vec::new();
        for key in 0..4u64 {
            let g = register_on(&l, 100 + key, ws.clone(), 0);
            winners.push(
                l.decide_item("tcp://p:1", &item(100 + key, 1, 0b11), 1, 0)
                    .winner_idx,
            );
            guards.push(g);
        }
        assert_eq!(winners, vec![0, 1, 0, 1]);
        assert_eq!(LateBind::charged(&ws[0]), (2000, 2));
        assert_eq!(LateBind::charged(&ws[1]), (2000, 2));
        drop(guards);
        assert_eq!(LateBind::charged(&ws[0]), (0, 0));
    }

    #[test]
    fn po2_reads_the_polled_loads_of_the_eligible() {
        let po2 = Arc::new(SmgPowerOfTwo::new());
        let l = LateBind::new(
            late_cfg(LateDecodeRule::Po2, None),
            PdDecodeTokensConfig::default(),
            Arc::new(WorkerRegistry::default()),
            Some(Arc::clone(&po2)),
        );
        po2.update_loads(&HashMap::from([
            ("http://d0:8000".to_string(), 900),
            ("http://d1:8000".to_string(), 100),
            ("http://d2:8000".to_string(), 500),
        ]));
        for key in 0..40u64 {
            let g = register_on(&l, 100 + key, pool(3), 0);
            // Eligible d0 and d2: d2 is lighter whichever draw comes first.
            let a = l.decide_item("tcp://p:1", &item(100 + key, 1, 0b101), 1, 0);
            assert_eq!((a.winner_idx, a.rule), (2, "po2"));
            drop(g);
        }
        // One eligible candidate wins alone.
        let g = register_on(&l, 200, pool(3), 0);
        assert_eq!(
            l.decide_item("tcp://p:1", &item(200, 1, 0b010), 1, 0)
                .winner_idx,
            1
        );
        drop(g);
    }

    /// `fit_req` with `capacity` tokens per decode worker.
    fn late_fit(capacity: u64) -> Arc<LateBind> {
        LateBind::new(
            LateDecodeConfig {
                fit_capacity_tokens: Some(capacity),
                ..late_cfg(LateDecodeRule::FitReq, None)
            },
            PdDecodeTokensConfig::default(),
            Arc::new(WorkerRegistry::default()),
            None,
        )
    }

    /// `n` held charges of `tokens` each on `w`.
    fn hold(
        l: &LateBind,
        w: &Worker,
        n: usize,
        tokens: u64,
    ) -> Vec<crate::workers::DecodeTokenCharge> {
        (0..n).map(|_| l.ledger.charge(w, tokens)).collect()
    }

    #[test]
    fn fit_req_picks_the_fewest_requests_among_the_fitting() {
        // 10_000 tokens per decode worker; the request charges 1000.
        let l = late_fit(10_000);
        let ws = pool(3);
        // d0: three light requests; d1: two medium ones; d2: one heavy one,
        // the fewest requests but over capacity (9_500 + 1000).
        let d0 = hold(&l, &ws[0], 3, 1_000);
        let d1 = hold(&l, &ws[1], 2, 4_000);
        let d2 = hold(&l, &ws[2], 1, 9_500);
        let g = register_on(&l, 51, ws.clone(), 0);
        // The signal marks the candidate that does not fit.
        let choice = l.choose(g.entry(), 0b111, 100).unwrap();
        assert_eq!(choice.signal, "0:3000/3,1:8000/2,2:9500/1!");
        drop(choice);
        // d1: d2 does not fit, and d1 has fewer requests than d0 (d1 would
        // take d0, the fewest tokens).
        let a = l.decide_item("tcp://p:1", &item(51, 1, 0b111), 1, 0);
        assert_eq!((a.winner_idx, a.rule), (1, "fit_req"));
        assert_eq!(LateBind::charged(&ws[1]), (9_000, 3));
        // Only d0 and d2 eligible: d0, the one that fits.
        let g2 = register_on(&l, 52, ws.clone(), 2);
        assert_eq!(
            l.decide_item("tcp://p:1", &item(52, 1, 0b101), 2, 0)
                .winner_idx,
            0
        );
        drop((g, g2));
        assert_eq!(LateBind::charged(&ws[0]).1, 3);
        assert_eq!(LateBind::charged(&ws[1]).1, 2);
        drop((d0, d1, d2));
        assert_eq!(get(&l.stats().fit_none), 0);
    }

    #[test]
    fn fit_req_falls_back_to_the_d1_order_when_none_fits() {
        let l = late_fit(5_000);
        let ws = pool(3);
        // With the 1000-token charge none fits: d0 4_500 + 1000, d1 4_200 +
        // 1000, d2 4_800 + 1000.
        let d0 = hold(&l, &ws[0], 1, 4_500);
        let d1 = hold(&l, &ws[1], 2, 2_100);
        let d2 = hold(&l, &ws[2], 1, 4_800);
        let g = register_on(&l, 53, ws.clone(), 0);
        let choice = l.choose(g.entry(), 0b111, 100).unwrap();
        assert_eq!(choice.signal, "0:4500/1!,1:4200/2!,2:4800/1!");
        drop(choice);
        // The fewest tokens (d1), not the fewest requests (d0, d2).
        let a = l.decide_item("tcp://p:1", &item(53, 1, 0b111), 1, 0);
        assert_eq!((a.winner_idx, a.rule), (1, "fit_req"));
        assert_eq!(LateBind::charged(&ws[1]).0, 5_200);
        drop((g, d0, d1, d2));
        assert_eq!(LateBind::charged(&ws[1]).1, 0);
        assert_eq!(get(&l.stats().fit_none), 2);
        assert!(l
            .render_metrics()
            .contains("sgl_router_late_d_fit_none_total 2"));
    }

    fn late_native() -> Arc<LateBind> {
        LateBind::new(
            LateDecodeConfig {
                fit_native: true,
                native_state_poll: true,
                ..late_cfg(LateDecodeRule::FitReq, None)
            },
            PdDecodeTokensConfig::default(),
            Arc::new(WorkerRegistry::default()),
            None,
        )
    }

    fn observed(n: u64, budget: u64, rooms: &[u64]) -> crate::late_fit::DecodeCapacity {
        crate::late_fit::DecodeCapacity {
            allocated_requests: n,
            context_tokens: 0,
            allocatable_tokens: budget,
            free_gpu_tokens: budget,
            output_reserve_tokens: 512,
            page_size: 64,
            allocated_rooms: rooms.iter().copied().collect(),
            received: Instant::now(),
            round_trip: Duration::ZERO,
            sample_age: Duration::ZERO,
        }
    }

    /// `fit_req` on observed capacity: the fewest allocated requests among
    /// the candidates that fit; the room stays reserved with the charge
    /// (moved by a mismatched `final`, kept by the bound leg); a bind with
    /// no decision reserves the default winner's room; no sample: the `d1`
    /// order.
    #[test]
    fn fit_req_native_reserves_the_room_and_reads_the_observed_state() {
        let l = late_native();
        let ws = pool(3);
        let store = Arc::clone(l.capacity_store());
        store.replace(HashMap::from([
            (ws[0].url.clone(), observed(2, 9_000, &[1, 2])),
            // 100 input tokens need 128 + 512 = 640 > 600.
            (ws[1].url.clone(), observed(1, 600, &[3])),
            (ws[2].url.clone(), observed(1, 9_000, &[4])),
        ]));
        let g = register_on(&l, 61, ws.clone(), 0);
        let a = l.decide_item("tcp://p:1", &item(61, 1, 0b111), 1, 0);
        assert_eq!((a.winner_idx, a.rule), (2, "fit_req"));
        assert_eq!(l.reserved_rooms(), 1);
        assert_eq!(LateBind::charged(&ws[2]), (1000, 1));
        // The unreported room counts on d2 (1 + 1 requests, 100 context
        // tokens): the next request takes d0 (2 requests, 0 context).
        let g2 = register_on(&l, 62, ws.clone(), 2);
        assert_eq!(
            l.decide_item("tcp://p:1", &item(62, 1, 0b111), 2, 0)
                .winner_idx,
            0
        );
        assert_eq!(l.reserved_rooms(), 2);
        // P timed out on room 61 and admitted d1: charge and room move.
        l.on_final(
            "tcp://p:1",
            &LinkItem {
                key: 61,
                bind_seq: 1,
                admitted_idx: 1,
                reason: Some("timeout".to_string()),
                ..Default::default()
            },
        );
        assert_eq!(LateBind::charged(&ws[2]), (0, 0));
        assert_eq!(LateBind::charged(&ws[1]), (1000, 1));
        let charge = l.bind_charge(g.entry(), 1);
        drop(g);
        assert_eq!(l.reserved_rooms(), 2, "the bound charge keeps room 61");
        drop(charge);
        assert_eq!(l.reserved_rooms(), 1);
        drop(g2);
        assert_eq!(l.reserved_rooms(), 0);
        // No decision (the link was down): the bound leg reserves its room.
        let g3 = register_on(&l, 63, ws.clone(), 0);
        let c3 = l.bind_charge(g3.entry(), 0);
        assert_eq!((l.reserved_rooms(), c3.tokens()), (1, 1000));
        drop((g3, c3));
        assert_eq!(l.reserved_rooms(), 0);
        // No observed state: the d1 order (the fewest charged tokens).
        store.replace(HashMap::new());
        let busy = l.ledger.charge(&ws[0], 50_000);
        let g4 = register_on(&l, 64, ws.clone(), 0);
        let choice = l.choose(g4.entry(), 0b011, 100).unwrap();
        assert_eq!(choice.winner, 1);
        assert!(
            choice.signal.contains("missing_or_stale"),
            "{}",
            choice.signal
        );
        drop((choice, g4, busy));
        assert_eq!(get(&l.stats().fit_none), 1);
        assert_eq!(l.reserved_rooms(), 0);
        for w in &ws {
            assert_eq!(LateBind::charged(w), (0, 0));
        }
    }

    #[test]
    fn bind_ready_reads_the_first_chunk_start() {
        let ready = msg(json!({
            "v": 1, "t": "bind_ready", "router_epoch": 7, "p_seq": 1, "t_send_ns": 5,
            "items": [{"key": 3, "bind_seq": 1, "eligible": 3, "ready": 2, "input_len": 10,
                       "default_idx": 0, "local_ok": false, "local_load": 0,
                       "first_chunk_start_unix_ns": 123_456_789u64}],
        }));
        let m = decode_message(&ready).unwrap();
        assert_eq!(m.items[0].first_chunk_start_unix_ns, Some(123_456_789));
        let old = msg(json!({
            "v": 1, "t": "bind_ready", "router_epoch": 7,
            "items": [{"key": 3, "bind_seq": 1, "eligible": 3}],
        }));
        assert_eq!(
            decode_message(&old).unwrap().items[0].first_chunk_start_unix_ns,
            None
        );
    }

    #[test]
    fn final_moves_the_charge_and_bind_takes_it() {
        let l = late(LateDecodeRule::Keep, None);
        let ws = pool(3);
        let g = register_on(&l, 31, ws.clone(), 0);
        let a = l.decide_item("tcp://p:1", &item(31, 1, 0b111), 1, 0);
        assert_eq!(a.winner_idx, 0);
        // P timed out and admitted candidate 2: the charge moves.
        l.on_final(
            "tcp://p:1",
            &LinkItem {
                key: 31,
                bind_seq: 1,
                admitted_idx: 2,
                reason: Some("timeout".to_string()),
                ..Default::default()
            },
        );
        assert_eq!(g.entry().admitted(), Some(2));
        assert_eq!(LateBind::charged(&ws[0]).1, 0);
        assert_eq!(LateBind::charged(&ws[2]).1, 1);
        // The 2xx leg of candidate 2 takes the charge with it.
        let charge = l.bind_charge(g.entry(), 2);
        drop(g);
        assert_eq!(LateBind::charged(&ws[2]), (1000, 1));
        drop(charge);
        assert_eq!(LateBind::charged(&ws[2]), (0, 0));
        assert_eq!(get(&l.stats().final_mismatch), 1);
        assert_eq!(get(&l.stats().finals_timeout), 1);

        // No decision at all (link down): the bound leg gets a new charge.
        let ws = pool(2);
        let g = register_on(&l, 32, ws.clone(), 0);
        let charge = l.bind_charge(g.entry(), 1);
        assert_eq!(
            (charge.tokens(), LateBind::charged(&ws[1])),
            (1000, (1000, 1))
        );
        // A decision for a key bound already answers the bound leg.
        assert_eq!(
            l.decide_item("tcp://p:1", &item(32, 1, 0b11), 1, 0)
                .winner_idx,
            1
        );
        // The bound leg differs from the decision: the charge moves to it.
        let ws3 = pool(2);
        let g3 = register_on(&l, 33, ws3.clone(), 0);
        assert_eq!(
            l.decide_item("tcp://p:1", &item(33, 1, 0b11), 1, 0)
                .winner_idx,
            0
        );
        let moved = l.bind_charge(g3.entry(), 1);
        assert_eq!(LateBind::charged(&ws3[0]), (0, 0));
        assert_eq!(LateBind::charged(&ws3[1]), (1000, 1));
        drop((g, charge, g3, moved));
        assert_eq!(LateBind::charged(&ws[1]), (0, 0));
        assert_eq!(LateBind::charged(&ws3[1]), (0, 0));
        assert_eq!(get(&l.stats().bind_undecided), 1);
        assert_eq!(get(&l.stats().bind_mismatch), 1);
        assert_eq!(get(&l.stats().decisions_bound), 1);
        assert!(l
            .render_metrics()
            .contains("sgl_router_late_d_mismatch_total{kind=\"bind\"} 1"));
    }

    #[test]
    fn frames_of_another_epoch_or_type_are_dropped() {
        let l = late(LateDecodeRule::Keep, None);
        let g = register_on(&l, 41, pool(2), 0);
        let mut up = false;
        let epoch = l.router_epoch();
        let stale = msg(json!({
            "v": 1, "t": "bind_ready", "router_epoch": u64::from(epoch) + 1,
            "items": [{"key": 41, "bind_seq": 1, "eligible": 3}],
        }));
        assert!(l.on_frame("tcp://p:1", &stale, &mut up).is_empty());
        assert_eq!(g.entry().decided(), None);
        let ready = msg(json!({
            "v": 1, "t": "bind_ready", "router_epoch": epoch,
            "items": [{"key": 41, "bind_seq": 1, "eligible": 3}],
        }));
        let replies = l.on_frame("tcp://p:1", &ready, &mut up);
        assert_eq!(replies.len(), 1);
        let decision = decode_message(&replies[0]).unwrap();
        assert_eq!(decision.t, "decision");
        assert_eq!(decision.router_epoch, Some(u64::from(epoch)));
        assert_eq!(decision.items[0].winner_idx, 0);
        // An ack of another mode never brings the link up.
        let sym = msg(json!({
            "v": 1, "t": "hello_ack", "router_epoch": epoch, "engine_boot_ns": 1,
            "mode": "symmetric",
        }));
        assert!(l.on_frame("tcp://p:1", &sym, &mut up).is_empty());
        assert!(!up);
        assert!(l
            .on_frame("tcp://p:1", &msg(json!({"v": 1, "t": "nonsense"})), &mut up)
            .is_empty());
        assert_eq!(get(&l.stats().stale_epoch), 1);
        assert_eq!(get(&l.stats().bad_messages), 1);
        // Nine items answer in two decision frames (8 + 1).
        let mut guards = Vec::new();
        let mut items = Vec::new();
        for key in 0..9u64 {
            guards.push(register_on(&l, 500 + key, pool(2), 1));
            items.push(json!({"key": 500 + key, "bind_seq": 1, "eligible": 3}));
        }
        let many = msg(json!({"v": 1, "t": "bind_ready", "router_epoch": epoch, "items": items}));
        let replies = l.on_frame("tcp://p:1", &many, &mut up);
        assert_eq!(replies.len(), 2);
        assert_eq!(decode_message(&replies[0]).unwrap().items.len(), MAX_ITEMS);
        assert_eq!(decode_message(&replies[1]).unwrap().items.len(), 1);
        drop((g, guards));
    }

    // -- link tests against an in-test owner (a zeromq ROUTER) --------------

    struct MockOwner {
        tx: RouterSendHalf,
        rx: mpsc::Receiver<(Bytes, LinkMessage)>,
        reader: JoinHandle<()>,
        endpoint: String,
        ident: Option<Bytes>,
    }

    impl Drop for MockOwner {
        fn drop(&mut self) {
            self.reader.abort();
        }
    }

    impl MockOwner {
        /// A ROUTER on 127.0.0.1:`port` (0 = any); a port just freed may
        /// take a moment to bind again.
        async fn bind(port: u16) -> Self {
            let mut tries = 0;
            let (socket, bound) = loop {
                let mut socket = RouterSocket::new();
                match socket.bind(&format!("tcp://127.0.0.1:{port}")).await {
                    Ok(bound) => break (socket, bound),
                    Err(e) if tries < 50 => {
                        tries += 1;
                        tracing::debug!("mock owner bind retry: {e}");
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                    Err(e) => panic!("mock owner bind: {e}"),
                }
            };
            let Endpoint::Tcp(_, port) = bound else {
                panic!("tcp endpoint expected")
            };
            let (tx, mut rx) = socket.split();
            let (in_tx, in_rx) = mpsc::channel(64);
            let reader = tokio::spawn(async move {
                while let Ok(msg) = rx.recv().await {
                    let mut frames = msg.into_vec();
                    let payload = frames.pop().unwrap();
                    let ident = frames.remove(0);
                    let decoded = decode_message(&payload).unwrap();
                    if in_tx.send((ident, decoded)).await.is_err() {
                        break;
                    }
                }
            });
            Self {
                tx,
                rx: in_rx,
                reader,
                endpoint: format!("tcp://127.0.0.1:{port}"),
                ident: None,
            }
        }

        /// The next message of type `t` (others, e.g. pings, are skipped).
        async fn expect(&mut self, t: &str) -> LinkMessage {
            loop {
                let (ident, m) = tokio::time::timeout(Duration::from_secs(5), self.rx.recv())
                    .await
                    .unwrap_or_else(|_| panic!("no {t} within 5 s"))
                    .unwrap();
                self.ident = Some(ident);
                if m.t == t {
                    return m;
                }
            }
        }

        async fn send(&mut self, value: Value) {
            let mut frame = ZmqMessage::from(Bytes::from(msg(value)));
            frame.push_front(self.ident.clone().unwrap());
            self.tx.send(frame).await.unwrap();
        }

        async fn handshake(&mut self, boot: u64) -> u64 {
            let hello = self.expect("hello").await;
            let epoch = hello.router_epoch.unwrap();
            self.send(json!({
                "v": 1, "t": "hello_ack", "router_epoch": epoch,
                "engine_boot_ns": boot, "mode": "pd", "kv_ip": "10.0.0.1",
            }))
            .await;
            epoch
        }
    }

    async fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
        for _ in 0..1000 {
            if cond() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("timed out waiting for {what}");
    }

    #[tokio::test]
    async fn link_hello_decision_final_and_ping() {
        let l = late(LateDecodeRule::D1, None);
        let mut owner = MockOwner::bind(0).await;
        let endpoint = owner.endpoint.clone();
        let link = tokio::spawn(run_link(Arc::clone(&l), endpoint.clone()));
        let epoch = owner.handshake(1).await;
        wait_until("link up", || l.link_status(&endpoint).is_some_and(|s| s.up)).await;

        let ws = pool(3);
        let _busy = ws[0].charge_decode_tokens(10_000);
        let _busy2 = ws[2].charge_decode_tokens(10_000);
        let g = l.register(EntrySpec {
            owner: endpoint.clone(),
            prefill_url: "http://p:1".to_string(),
            key: 41,
            rid: None,
            candidates: ws.clone(),
            default_idx: 2,
            charge_tokens: 500,
        });
        owner
            .send(json!({
                "v": 1, "t": "bind_ready", "router_epoch": epoch, "p_seq": 1, "t_send_ns": 5,
                "items": [
                    {"key": 41, "bind_seq": 1, "eligible": 7, "ready": 3, "input_len": 10,
                     "default_idx": 2, "local_ok": false, "local_load": 0},
                    {"key": 999, "bind_seq": 1, "eligible": 1, "ready": 1, "input_len": 10,
                     "default_idx": 0, "local_ok": false, "local_load": 0},
                ],
            }))
            .await;
        let decision = owner.expect("decision").await;
        assert_eq!(decision.router_epoch, Some(epoch));
        let winners: Vec<(u64, u64)> = decision
            .items
            .iter()
            .map(|i| (i.key, i.winner_idx))
            .collect();
        // d1: candidate 1 is the least charged.
        assert_eq!(winners, vec![(41, 1), (999, u64::from(WINNER_UNKNOWN))]);
        assert_eq!(decision.items[0].rule.as_deref(), Some("d1"));
        assert!(decision.items[0].t_decide_ns >= decision.items[0].t_recv_ns);

        // A bind_ready of another router epoch gets no answer.
        owner
            .send(json!({
                "v": 1, "t": "bind_ready", "router_epoch": epoch ^ 1, "p_seq": 2,
                "items": [{"key": 41, "bind_seq": 9, "eligible": 1}],
            }))
            .await;
        owner
            .send(json!({
                "v": 1, "t": "final", "router_epoch": epoch,
                "items": [{"key": 41, "bind_seq": 1, "admitted_idx": 0, "reason": "timeout", "t_admit_ns": 1}],
            }))
            .await;
        wait_until("final moved the charge", || {
            LateBind::charged(&ws[0]).1 == 2
        })
        .await;
        assert_eq!(LateBind::charged(&ws[1]).1, 0);

        // Pings carry the time; the router answers nothing to a pong.
        let ping = owner.expect("ping").await;
        assert!(ping.ts_ns > 0);
        owner
            .send(json!({"v": 1, "t": "pong", "ts_ns": ping.ts_ns}))
            .await;
        // A bad message is dropped and counted; the link stays up.
        owner.send(json!({"v": 1, "t": "nonsense"})).await;
        wait_until("bad message counted", || {
            l.link_status(&endpoint)
                .is_some_and(|s| s.bad_messages == 1)
        })
        .await;
        assert!(l.link_status(&endpoint).unwrap().up);
        assert_eq!(l.link_status(&endpoint).unwrap().finals, 1);
        assert!(l
            .render_metrics()
            .contains(&format!("sgl_router_late_d_link_up{{p=\"{endpoint}\"}} 1")));
        drop(g);
        link.abort();
    }

    #[tokio::test]
    async fn link_reconnects_after_the_owner_restarts() {
        let l = late(LateDecodeRule::Keep, None);
        let mut owner = MockOwner::bind(0).await;
        let endpoint = owner.endpoint.clone();
        let port: u16 = endpoint.rsplit(':').next().unwrap().parse().unwrap();
        let link = tokio::spawn(run_link(Arc::clone(&l), endpoint.clone()));
        owner.handshake(1).await;
        wait_until("link up", || l.link_status(&endpoint).is_some_and(|s| s.up)).await;
        // The owner dies; the link notices within 3 s of silence (or at its
        // next send) and goes down.
        drop(owner);
        wait_until("link down", || {
            l.link_status(&endpoint).is_some_and(|s| !s.up)
        })
        .await;
        // The owner comes back on the same port; the link says hello again.
        let mut owner = MockOwner::bind(port).await;
        owner.handshake(2).await;
        wait_until("link up again", || {
            l.link_status(&endpoint)
                .is_some_and(|s| s.up && s.engine_boot_ns == 2 && s.hello_acks == 2)
        })
        .await;
        link.abort();
    }

    #[tokio::test]
    async fn link_refuses_a_symmetric_owner() {
        let l = late(LateDecodeRule::Keep, None);
        let mut owner = MockOwner::bind(0).await;
        let endpoint = owner.endpoint.clone();
        let link = tokio::spawn(run_link(Arc::clone(&l), endpoint.clone()));
        let hello = owner.expect("hello").await;
        owner
            .send(json!({
                "v": 1, "t": "hello_ack", "router_epoch": hello.router_epoch,
                "engine_boot_ns": 1, "mode": "symmetric", "owner_addr": "tcp://x:1",
            }))
            .await;
        // The router keeps saying hello: the link never comes up.
        owner.expect("hello").await;
        assert!(!l.link_status(&endpoint).is_some_and(|s| s.up));
        link.abort();
    }

    #[tokio::test]
    async fn supervisor_links_registered_choose_prefills_only() {
        let mut owner = MockOwner::bind(0).await;
        let port: u16 = owner.endpoint.rsplit(':').next().unwrap().parse().unwrap();
        let registry = Arc::new(WorkerRegistry::default());
        for (id, url, mode, info) in [
            (
                "p0",
                "http://127.0.0.1:30000",
                WorkerMode::Prefill,
                Some(choose_info(Some(port))),
            ),
            // Not a choose prefill (no port): no link.
            (
                "p1",
                "http://127.0.0.2:30000",
                WorkerMode::Prefill,
                Some(choose_info(None)),
            ),
            (
                "d0",
                "http://127.0.0.3:30000",
                WorkerMode::Decode,
                Some(choose_info(None)),
            ),
        ] {
            registry
                .add(WorkerSpec {
                    id: WorkerId(id.into()),
                    url: url.into(),
                    mode,
                    model_ids: vec![ModelId("m".into())],
                    bootstrap_port: (mode == WorkerMode::Prefill).then_some(8998),
                })
                .unwrap();
            registry.get(&WorkerId(id.into())).unwrap().set_late_d(info);
        }
        let l = LateBind::new(
            late_cfg(LateDecodeRule::Keep, None),
            PdDecodeTokensConfig::default(),
            Arc::clone(&registry),
            None,
        );
        l.start();
        l.start(); // once only
        owner.handshake(7).await;
        let endpoint = format!("tcp://127.0.0.1:{port}");
        wait_until("supervised link up", || {
            l.link_status(&endpoint)
                .is_some_and(|s| s.up && s.engine_boot_ns == 7)
        })
        .await;
        assert_eq!(l.links.len(), 1);
    }

    // -- symmetric owners ---------------------------------------------------

    fn sym_worker(url: &str, late: Option<LateDWorkerInfo>) -> Arc<Worker> {
        let w = Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(url.into()),
            url: url.into(),
            mode: WorkerMode::Symmetric,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: Some(18998),
        }));
        w.set_late_d(late);
        w
    }

    fn late_sym(rule: LateDecodeRule, owners: Vec<Arc<Worker>>) -> Arc<LateBind> {
        LateBind::new_symmetric(
            late_cfg(rule, None),
            PdDecodeTokensConfig::default(),
            Arc::new(WorkerRegistry::default()),
            Arc::new(move || owners.clone()),
        )
    }

    /// Records every hook call, in order.
    #[derive(Default)]
    struct Hook(Mutex<Vec<String>>);

    impl LateSessionHook for Hook {
        fn candidate(&self, worker_url: &str) {
            self.0.lock().push(format!("candidate {worker_url}"));
        }
        fn bound(&self, worker_url: &str) {
            self.0.lock().push(format!("bound {worker_url}"));
        }
    }

    /// A symmetric `fit_req` table (fixed capacity) whose decode-role gate
    /// returns the URLs in `roles` (changed by the test like a role update).
    fn late_sym_gated(roles: Arc<Mutex<Vec<Arc<Worker>>>>) -> Arc<LateBind> {
        let l = LateBind::new_symmetric(
            LateDecodeConfig {
                fit_capacity_tokens: Some(1_000_000),
                ..late_cfg(LateDecodeRule::FitReq, None)
            },
            PdDecodeTokensConfig::default(),
            Arc::new(WorkerRegistry::default()),
            Arc::new(Vec::new),
        );
        l.set_decode_role_gate(Arc::new(move || roles.lock().clone()));
        l
    }

    fn sym_pool(n: usize) -> Vec<Arc<Worker>> {
        (0..n)
            .map(|i| sym_worker(&format!("http://w{i}:8000"), Some(choose_info(None))))
            .collect()
    }

    /// External role control: a candidate the controller moved out of the
    /// decode role (or that is draining) is left out of the bind, even when
    /// it has the fewest requests; the arrival pick among them too.
    #[test]
    fn role_gate_leaves_out_candidates_that_left_the_decode_role() {
        let ws = sym_pool(3);
        let roles = Arc::new(Mutex::new(ws.clone()));
        let l = late_sym_gated(Arc::clone(&roles));
        // w0 and w2 carry load; w1 is empty (the fit_req winner).
        let h0 = hold(&l, &ws[0], 2, 1_000);
        let h2 = hold(&l, &ws[2], 1, 1_000);
        let g = register_on(&l, 61, ws.clone(), 1);
        // The controller moves w1 (the arrival pick, the emptiest) to prefill.
        *roles.lock() = vec![Arc::clone(&ws[0]), Arc::clone(&ws[2])];
        let a = l.decide_item("tcp://p:1", &item(61, 1, 0b111), 1, 0);
        assert_eq!(
            a.winner_idx, 2,
            "w2: fewest requests among those still decode"
        );
        assert_eq!(get(&l.stats().role_filtered), 1);
        assert_eq!(get(&l.stats().role_fallback), 0);
        assert_eq!(LateBind::charged(&ws[1]).1, 0);
        // Every candidate in the decode role: no filter.
        let g2 = register_on(&l, 62, ws.clone(), 0);
        *roles.lock() = ws.clone();
        assert_eq!(
            l.decide_item("tcp://p:1", &item(62, 1, 0b111), 1, 0)
                .winner_idx,
            1
        );
        assert_eq!(get(&l.stats().role_filtered), 1);
        drop((g, g2, h0, h2));
    }

    /// A decode added after the request was planned is not one of its
    /// candidates; when every eligible candidate left the decode role (all
    /// of them draining toward prefill), the bind falls back to all eligible
    /// (counted) instead of naming none.
    #[test]
    fn role_gate_with_no_candidate_left_falls_back_to_the_eligible() {
        let ws = sym_pool(3);
        let added = sym_worker("http://w9:8000", Some(choose_info(None)));
        let roles = Arc::new(Mutex::new(vec![Arc::clone(&ws[0]), Arc::clone(&ws[1])]));
        let l = late_sym_gated(Arc::clone(&roles));
        let h0 = hold(&l, &ws[0], 3, 1_000);
        let g = register_on(&l, 71, vec![Arc::clone(&ws[0]), Arc::clone(&ws[1])], 0);
        // Both candidates drain toward prefill; w2 and the new w9 are decode.
        *roles.lock() = vec![Arc::clone(&ws[2]), Arc::clone(&added)];
        let a = l.decide_item("tcp://p:1", &item(71, 1, 0b11), 1, 0);
        assert!(
            a.winner_idx < 2,
            "a candidate of the request, not a new decode"
        );
        assert_eq!(a.winner_idx, 1, "fit_req among both: w1 has fewer requests");
        assert_eq!(get(&l.stats().role_filtered), 1);
        assert_eq!(get(&l.stats().role_fallback), 1);
        drop((g, h0));
    }

    /// The decision's charge stays on the entry until a leg binds; a D-leg
    /// that ends without a bind releases it at once (no leaked charge while
    /// the execution lingers on an unknown P-leg).
    #[test]
    fn drop_pending_charge_releases_an_unbound_decision() {
        let ws = sym_pool(2);
        let l = late_sym_gated(Arc::new(Mutex::new(ws.clone())));
        let g = register_on(&l, 81, ws.clone(), 0);
        let a = l.decide_item("tcp://p:1", &item(81, 1, 0b11), 1, 0);
        let w = a.winner_idx as usize;
        assert_eq!(LateBind::charged(&ws[w]).1, 1);
        l.drop_pending_charge(g.entry());
        assert_eq!(LateBind::charged(&ws[w]).1, 0);
        // A later decision (new bind_seq) after the D-leg ended still
        // answers the prefill but charges nothing.
        let again = l.decide_item("tcp://p:1", &item(81, 2, 0b11), 1, 0);
        assert!((again.winner_idx as usize) < 2);
        assert_eq!(g.entry().decode_target(), Some(again.winner_idx as usize));
        for w in &ws {
            assert_eq!(LateBind::charged(w).1, 0);
        }
        // A bind after that charges afresh and the guard holds it.
        let charge = l.bind_charge(g.entry(), w);
        assert_eq!(LateBind::charged(&ws[w]).1, 1);
        drop(charge);
        assert_eq!(LateBind::charged(&ws[w]).1, 0);
        drop(g);
    }

    #[test]
    fn sym_prefill_field_offers_no_local() {
        assert_eq!(
            sym_prefill_field(5, 3, 1),
            json!({"v": 1, "mode": "choose", "router_epoch": 5, "k": 3,
                   "default_idx": 1, "local_ok": false})
        );
        // The PD field is unchanged.
        assert!(prefill_field(5, 3, 1).get("local_ok").is_none());
    }

    /// Each symmetric candidate leg carries its own `kv_import` next to its
    /// `late_d.cand_idx`; without imports the bytes are the PD bytes.
    #[test]
    fn leg_bodies_with_imports_carry_one_import_each() {
        let base = json!({"model": "m", "prompt": [1, 2, 3], "rid": "r"});
        let pd = LegBodies::new(&base, 9, 2).unwrap();
        let imports = vec![
            json!({"transfer_id": 7, "attempt_id": 11, "owner_worker": "http://p",
                   "owner_addr": "tcp://p:1", "expected_tokens": 3, "cache_policy": "retain"}),
            json!({"transfer_id": 7, "attempt_id": 12, "owner_worker": "http://p",
                   "owner_addr": "tcp://p:1", "expected_tokens": 3, "cache_policy": "retain"}),
        ];
        let sym = LegBodies::new(&base, 9, 2)
            .unwrap()
            .with_imports(&imports)
            .unwrap();
        for (i, import) in imports.iter().enumerate() {
            let v: Value = serde_json::from_slice(&sym.leg(i)).unwrap();
            assert_eq!(v["kv_import"], *import);
            assert_eq!(v["late_d"]["cand_idx"], i);
            assert_eq!(v["late_d"]["k"], 2);
            assert_eq!(v["prompt"], json!([1, 2, 3]));
            let plain: Value = serde_json::from_slice(&pd.leg(i)).unwrap();
            assert!(plain.get("kv_import").is_none());
            let mut stripped = v.clone();
            stripped.as_object_mut().unwrap().remove("kv_import");
            assert_eq!(stripped, plain);
        }
        // An empty shared body still gives valid JSON.
        let empty = LegBodies::new(&json!({}), 9, 1)
            .unwrap()
            .with_imports(&imports[..1])
            .unwrap();
        let v: Value = serde_json::from_slice(&empty.leg(0)).unwrap();
        assert_eq!(v["kv_import"]["attempt_id"], 11);
        // The import count must match k, and each must be an object.
        assert!(LegBodies::new(&base, 9, 2)
            .unwrap()
            .with_imports(&imports[..1])
            .is_err());
        assert!(LegBodies::new(&base, 9, 1)
            .unwrap()
            .with_imports(&[json!(1)])
            .is_err());
    }

    /// The session hook sees the arrival pick at registration, the decided
    /// winner before the answer, and the bound leg; a request without a hook
    /// behaves as before.
    #[test]
    fn session_hook_sees_the_default_the_decided_winner_and_the_bind() {
        let l = late(LateDecodeRule::D1, None);
        let ws = pool(3);
        let _busy = ws[0].charge_decode_tokens(10_000);
        let hook = Arc::new(Hook::default());
        let g = l.register_with(
            EntrySpec {
                owner: "tcp://p:1".to_string(),
                prefill_url: "http://p:1".to_string(),
                key: 71,
                rid: None,
                candidates: ws.clone(),
                default_idx: 0,
                charge_tokens: 100,
            },
            Some(Arc::clone(&hook) as Arc<dyn LateSessionHook>),
        );
        assert_eq!(*hook.0.lock(), vec![format!("candidate {}", ws[0].url)]);
        let a = l.decide_item("tcp://p:1", &item(71, 1, 0b111), 1, 0);
        assert_eq!(a.winner_idx, 1);
        // A repeated bind_ready records nothing new.
        l.decide_item("tcp://p:1", &item(71, 1, 0b111), 2, 0);
        let charge = l.bind_charge(g.entry(), 1);
        assert_eq!(
            *hook.0.lock(),
            vec![
                format!("candidate {}", ws[0].url),
                format!("candidate {}", ws[1].url),
                format!("bound {}", ws[1].url),
            ]
        );
        drop((charge, g));
    }

    /// A `final` that names another candidate than the decision records that
    /// candidate (P sent it the tag); a matching `final` records nothing.
    #[test]
    fn session_hook_records_a_mismatched_admit() {
        let l = late(LateDecodeRule::D1, None);
        let ws = pool(3);
        let _busy = ws[0].charge_decode_tokens(10_000);
        let hook = Arc::new(Hook::default());
        let g = l.register_with(
            EntrySpec {
                owner: "tcp://p:1".to_string(),
                prefill_url: "http://p:1".to_string(),
                key: 72,
                rid: None,
                candidates: ws.clone(),
                default_idx: 0,
                charge_tokens: 100,
            },
            Some(Arc::clone(&hook) as Arc<dyn LateSessionHook>),
        );
        let a = l.decide_item("tcp://p:1", &item(72, 1, 0b111), 1, 0);
        assert_eq!(a.winner_idx, 1);
        let fin = |admitted_idx: u64| LinkItem {
            key: 72,
            bind_seq: 1,
            admitted_idx,
            reason: Some("timeout".to_string()),
            ..Default::default()
        };
        l.on_final("tcp://p:1", &fin(1));
        assert_eq!(hook.0.lock().len(), 2, "a matching final records nothing");
        l.on_final("tcp://p:1", &fin(2));
        assert_eq!(
            hook.0.lock().last().cloned(),
            Some(format!("candidate {}", ws[2].url))
        );
        drop(g);
    }

    /// A symmetric table answers `bind_ready` of a symmetric owner (key =
    /// transfer id) and refuses a PD prefill's `hello_ack`.
    #[tokio::test]
    async fn symmetric_link_serves_a_symmetric_owner_only() {
        let l = late_sym(LateDecodeRule::D1, Vec::new());
        assert!(l.symmetric());
        let mut owner = MockOwner::bind(0).await;
        let endpoint = owner.endpoint.clone();
        let link = tokio::spawn(run_link(Arc::clone(&l), endpoint.clone()));
        // A PD prefill: the link never comes up.
        let hello = owner.expect("hello").await;
        owner
            .send(json!({
                "v": 1, "t": "hello_ack", "router_epoch": hello.router_epoch,
                "engine_boot_ns": 1, "mode": "pd", "kv_ip": "10.0.0.1",
            }))
            .await;
        let hello = owner.expect("hello").await;
        assert!(!l.link_status(&endpoint).is_some_and(|s| s.up));
        // A symmetric owner: up, and its bind_ready is answered.
        let epoch = hello.router_epoch.unwrap();
        owner
            .send(json!({
                "v": 1, "t": "hello_ack", "router_epoch": epoch,
                "engine_boot_ns": 3, "mode": "symmetric", "kv_ip": "10.0.0.1",
            }))
            .await;
        wait_until("symmetric link up", || {
            l.link_status(&endpoint).is_some_and(|s| s.up)
        })
        .await;
        let ws: Vec<Arc<Worker>> = (0..2)
            .map(|i| sym_worker(&format!("http://s{i}:8000"), Some(choose_info(None))))
            .collect();
        let _busy = ws[0].charge_decode_tokens(5_000);
        let transfer_id = (1u64 << 62) + 5;
        let g = l.register(EntrySpec {
            owner: endpoint.clone(),
            prefill_url: "http://owner:8000".to_string(),
            key: transfer_id,
            rid: Some("r1".to_string()),
            candidates: ws.clone(),
            default_idx: 0,
            charge_tokens: 100,
        });
        owner
            .send(json!({
                "v": 1, "t": "bind_ready", "router_epoch": epoch, "p_seq": 1, "t_send_ns": 5,
                "items": [{"key": transfer_id, "bind_seq": 1, "eligible": 3, "ready": 2,
                           "input_len": 10, "default_idx": 0, "local_ok": false,
                           "local_load": 0}],
            }))
            .await;
        let decision = owner.expect("decision").await;
        assert_eq!(decision.items[0].key, transfer_id);
        assert_eq!(
            decision.items[0].winner_idx, 1,
            "d1: the uncharged candidate"
        );
        drop(g);
        link.abort();
    }

    /// The symmetric supervisor links the listed symmetric owners that run
    /// late D choose with a control port, and nothing else.
    #[tokio::test]
    async fn symmetric_supervisor_links_listed_choose_owners_only() {
        let mut owner = MockOwner::bind(0).await;
        let port: u16 = owner.endpoint.rsplit(':').next().unwrap().parse().unwrap();
        let owners = vec![
            sym_worker("http://127.0.0.1:30000", Some(choose_info(Some(port)))),
            // No control port: no link.
            sym_worker("http://127.0.0.2:30000", Some(choose_info(None))),
            // Late D off: no link.
            sym_worker("http://127.0.0.3:30000", None),
            // A PD prefill in the list is not a symmetric owner.
            worker(
                "http://127.0.0.4:30000",
                WorkerMode::Prefill,
                Some(choose_info(Some(port.wrapping_add(1)))),
            ),
        ];
        let l = late_sym(LateDecodeRule::Keep, owners);
        l.start();
        let epoch = {
            let hello = owner.expect("hello").await;
            hello.router_epoch.unwrap()
        };
        owner
            .send(json!({
                "v": 1, "t": "hello_ack", "router_epoch": epoch,
                "engine_boot_ns": 9, "mode": "symmetric", "kv_ip": "10.0.0.1",
            }))
            .await;
        let endpoint = format!("tcp://127.0.0.1:{port}");
        wait_until("supervised symmetric link up", || {
            l.link_status(&endpoint)
                .is_some_and(|s| s.up && s.engine_boot_ns == 9)
        })
        .await;
        assert_eq!(l.links.len(), 1);
    }

    /// `fit_req` on observed capacity with symmetric candidates: the
    /// reservation is keyed by the transfer id and is acknowledged when the
    /// decode lists that transfer id among its allocated rooms.
    #[test]
    fn fit_req_native_on_symmetric_candidates_keys_the_transfer_id() {
        let l = LateBind::new_symmetric(
            LateDecodeConfig {
                fit_native: true,
                native_state_poll: true,
                ..late_cfg(LateDecodeRule::FitReq, None)
            },
            PdDecodeTokensConfig::default(),
            Arc::new(WorkerRegistry::default()),
            Arc::new(Vec::<Arc<Worker>>::new),
        );
        let ws: Vec<Arc<Worker>> = (0..3)
            .map(|i| sym_worker(&format!("http://s{i}:8000"), Some(choose_info(None))))
            .collect();
        let tid = 1u64 << 40;
        l.capacity_store().replace(HashMap::from([
            (ws[0].url.clone(), observed(3, 9_000, &[1, 2, 3])),
            (ws[1].url.clone(), observed(1, 600, &[4])),
            (ws[2].url.clone(), observed(2, 9_000, &[5, 6])),
        ]));
        let g = register_on(&l, tid, ws.clone(), 0);
        let a = l.decide_item("tcp://p:1", &item(tid, 1, 0b111), 1, 0);
        // d1 does not fit (640 > 600); d2 has fewer requests than d0.
        assert_eq!((a.winner_idx, a.rule), (2, "fit_req"));
        assert_eq!(l.reserved_rooms(), 1);
        assert!(l.reservations.lock().contains_key(&tid));
        // The decode reports the transfer id: the reservation is
        // acknowledged (no longer counted) at the next choice.
        l.capacity_store().replace(HashMap::from([
            (ws[0].url.clone(), observed(3, 9_000, &[1, 2, 3])),
            (ws[1].url.clone(), observed(1, 600, &[4])),
            (ws[2].url.clone(), observed(3, 9_000, &[5, 6, tid])),
        ]));
        let g2 = register_on(&l, tid + 1, ws.clone(), 0);
        l.decide_item("tcp://p:1", &item(tid + 1, 1, 0b101), 2, 0);
        assert!(l.reservations.lock()[&tid].acknowledged);
        drop((g, g2));
    }
}
