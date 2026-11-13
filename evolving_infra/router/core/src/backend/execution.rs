// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! Lifecycle for externally forwarded prepared inputs. This owns handles to the
//! existing Backend ledger (the P-leg's reservation and the D-leg's in-flight
//! decode count), never a separate load counter. Uncertain transport
//! failures stay uncertain. Dispatched P work requires terminal evidence before
//! retirement; work known not to have been submitted can be released earlier.
//!
//! Validates metadata-only prepared inputs and manages reservation, claim,
//! upload and terminal callbacks for split P/D plans. Authenticated tombstones
//! make late callbacks idempotent; uncertain requests continue consuming capacity.
use crate::{
    backend::sglang::request::prepared_target,
    backend::state::{DecodeReservation, LedgerHandle},
    backend::workload::{CompletedWorkload, RequestSample},
    scheduler::{Decision, PlaceInput},
};
// Compatibility export; the wire protocol itself belongs to the SGLang adapter.
pub use crate::backend::sglang::request::cancel_engines;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HashContract {
    pub page_size: u64,
    pub is_eagle: bool,
    pub model_path: String,
    pub cache_namespace: Option<String>,
    /// Frozen image preprocessing fields; absent for text-only deployments.
    #[serde(default)]
    pub image_config: Option<Value>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedConfig {
    pub callback_url: String,
    pub max_inflight: usize,
    /// Tokenizer config_id -> the frozen query hash contract.
    pub contracts: HashMap<String, HashContract>,
    /// Safety net for a lost D-leg confirmation (see
    /// [`Executions::with_decode_unresolved_expiry`]): a claimed execution
    /// whose D-leg was reported `failed`/`cancelled` without terminal engine
    /// evidence, and that received no callback for this many seconds, ends its
    /// D-leg. 0 = never (the earlier behaviour). Default 300.
    #[serde(default = "default_decode_unresolved_expire_secs")]
    pub decode_unresolved_expire_secs: u64,
    /// A claimed execution that received no callback for this many seconds
    /// is treated as abandoned by its tokenizer (see
    /// [`Executions::with_silent_expiry`]). It must exceed the tokenizer's
    /// `request_timeout_seconds` (default 7200): a running request sends no
    /// callback between its upload and its end. 0 = never. Default 7500.
    #[serde(default = "default_silent_expire_secs")]
    pub silent_expire_secs: u64,
}
/// Default of [`PreparedConfig::decode_unresolved_expire_secs`]: ten times the
/// tokenizer's 30 s cancellation grace, so a slow but live cleanup always wins.
pub const DEFAULT_DECODE_UNRESOLVED_EXPIRE_SECS: u64 = 300;
fn default_decode_unresolved_expire_secs() -> u64 {
    DEFAULT_DECODE_UNRESOLVED_EXPIRE_SECS
}
/// Default of [`PreparedConfig::silent_expire_secs`].
pub const DEFAULT_SILENT_EXPIRE_SECS: u64 = 7500;
fn default_silent_expire_secs() -> u64 {
    DEFAULT_SILENT_EXPIRE_SECS
}
impl PreparedConfig {
    /// The D-leg expiry of [`Self::decode_unresolved_expire_secs`]; `None` = never.
    pub fn decode_unresolved_expiry(&self) -> Option<Duration> {
        (self.decode_unresolved_expire_secs > 0)
            .then(|| Duration::from_secs(self.decode_unresolved_expire_secs))
    }
    /// The silent-execution expiry of [`Self::silent_expire_secs`]; `None` = never.
    pub fn silent_expiry(&self) -> Option<Duration> {
        (self.silent_expire_secs > 0).then(|| Duration::from_secs(self.silent_expire_secs))
    }
    /// Load and validate the deployment contract for prepared-input forwarding.
    ///
    /// # Inputs
    /// No arguments. Reads `EVOLVING_INFRA_PREPARED_CONFIG` as a JSON file path
    /// from the process environment.
    ///
    /// # Outputs
    /// Returns `Ok(None)` when the feature is unconfigured, or the parsed
    /// configuration when callback URL, capacity and hash contracts are valid.
    /// File, JSON, URL and contract validation failures return an error string;
    /// no service is started.
    pub fn from_env() -> Result<Option<Self>, String> {
        let Some(path) = std::env::var_os("EVOLVING_INFRA_PREPARED_CONFIG") else {
            return Ok(None);
        };
        let data = std::fs::read(path).map_err(|e| e.to_string())?;
        let cfg: Self = serde_json::from_slice(&data).map_err(|e| e.to_string())?;
        let url = reqwest::Url::parse(&cfg.callback_url).map_err(|e| e.to_string())?;
        if !matches!(url.scheme(), "http" | "https")
            || cfg.max_inflight == 0
            || cfg.contracts.is_empty()
            || cfg.contracts.values().any(|c| c.page_size == 0)
        {
            return Err("invalid prepared-input configuration".into());
        }
        Ok(Some(cfg))
    }
    /// Validate a prepared descriptor and convert its metadata into placement input.
    ///
    /// # Inputs
    /// `self` supplies the frozen tokenizer/hash contracts; `p` is the
    /// metadata-only descriptor; `model` identifies the model being scheduled.
    ///
    /// # Outputs
    /// Returns a `PlaceInput` carrying request ID, prompt length, cache namespace
    /// and bit-preserved hash values, with no token array. Rejects unexpected
    /// fields, mismatched hash specifications, invalid IDs/counts, partial blocks
    /// and unsupported protocol versions with an error string. No KV query or
    /// reservation is performed.
    pub fn input(&self, p: &Value, model: &str) -> Result<PlaceInput, String> {
        let object = p
            .as_object()
            .ok_or("prepared descriptor must be an object")?;
        let allowed = [
            "protocol_version",
            "node_id",
            "request_id",
            "ref",
            "config_id",
            "prompt_tokens",
            "max_new_tokens",
            "content",
            "expires_in_ms",
        ];
        if object.keys().any(|key| !allowed.contains(&key.as_str())) {
            return Err("prepared descriptor accepts metadata only".into());
        }
        let content = p["content"].as_object().ok_or("missing content")?;
        if content
            .keys()
            .any(|k| !matches!(k.as_str(), "namespace" | "hash_spec" | "blocks"))
        {
            return Err("content accepts hashes only".into());
        }
        let id = required_str(p, "config_id")?;
        let c = self
            .contracts
            .get(id)
            .ok_or("unknown tokenizer config_id")?;
        let rid = required_str(p, "request_id")?;
        required_str(p, "ref")?;
        let n = p["prompt_tokens"]
            .as_u64()
            .filter(|n| *n > 0)
            .ok_or("invalid prompt_tokens")?;
        if p["protocol_version"] != 1 {
            return Err("unsupported prepared protocol".into());
        }
        let h = p["content"]["hash_spec"]
            .as_object()
            .ok_or("missing hash specification")?;
        let hash_fields = [
            "mode",
            "encoding",
            "implementation",
            "page_size",
            "is_eagle",
            "model",
            "cache_namespace",
            "partial_blocks",
            "window_tokens",
        ];
        if h.len() != hash_fields.len() + usize::from(c.image_config.is_some())
            || h.keys().any(|key| {
                !(hash_fields.contains(&key.as_str())
                    || (key == "image_config" && c.image_config.is_some()))
            })
        {
            return Err("hash specification does not match the tokenizer contract".into());
        }
        if h["mode"] != "dynamo"
            || h["encoding"] != "signed_i64"
            || h["implementation"] != "dynamo.llm.compute_block_hash_for_seq"
            || h["page_size"].as_u64() != Some(c.page_size)
            || h["is_eagle"].as_bool() != Some(c.is_eagle)
            || h["model"].as_str() != Some(c.model_path.as_str())
            || h["cache_namespace"] != json!(c.cache_namespace)
            || h["partial_blocks"] != false
            || h["window_tokens"].as_u64() != c.page_size.checked_add(u64::from(c.is_eagle))
            || h.get("image_config") != c.image_config.as_ref()
        {
            return Err("prepared hash contract does not match deployment".into());
        }
        let blocks = p["content"]["blocks"].as_array().ok_or("missing blocks")?;
        let count = n.saturating_sub(u64::from(c.is_eagle)) / c.page_size;
        if blocks.len() as u64 != count {
            return Err("wrong number of complete hash blocks".into());
        }
        let hashes = blocks
            .iter()
            .map(|b| {
                let fields = b.as_object().ok_or("invalid hash block")?;
                if fields.len() != 2
                    || fields
                        .keys()
                        .any(|key| !matches!(key.as_str(), "hash" | "valid_tokens"))
                {
                    return Err("block accepts only hash and valid_tokens".to_string());
                }
                if b["valid_tokens"].as_u64() != Some(c.page_size) {
                    return Err("partial block".to_string());
                }
                // Bit-preserving wire conversion: Dynamo uses i64, router uses u64.
                b["hash"]
                    .as_i64()
                    .map(|v| v as u64)
                    .ok_or("invalid signed hash".into())
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(PlaceInput {
            tokens: None,
            hashes: Some(hashes),
            prompt_len: n,
            cache_salt: c.cache_namespace.clone(),
            model_id: Some(model.into()),
            session_id: None,
            turn: None,
            rid: Some(rid.into()),
            routing_key: None,
            ingress_ms: 0.0,
            placement_session: None,
            forecast_session: None,
        })
    }
}
/// Read a bounded, nonempty string field from a JSON descriptor.
///
/// # Inputs
/// `v` is the containing JSON value; `key` identifies the required field.
///
/// # Outputs
/// Returns a borrowed string of 1 through 512 bytes, or an error naming
/// the field when its type, presence or length is invalid.
fn required_str<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    v[key]
        .as_str()
        .filter(|s| !s.is_empty() && s.len() <= 512)
        .ok_or_else(|| format!("invalid {key}"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Leg {
    Prefill,
    Decode,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LegState {
    Reserved,
    Claimed,
    Uploaded,
    Completed,
    Failed,
    Cancelled,
    /// Late D choose only, with `candidate`: that candidate's D-leg answered
    /// 2xx, so the prefill admitted it and it is the request's D-leg.
    Bound,
}
impl LegState {
    /// Expose the canonical event-ID component for an execution-leg state.
    ///
    /// # Inputs
    /// `self` is the state being serialized into an event identity.
    ///
    /// # Outputs
    /// Returns its static lowercase wire name without modifying any record.
    fn name(self) -> &'static str {
        match self {
            Self::Reserved => "reserved",
            Self::Claimed => "claimed",
            Self::Uploaded => "uploaded",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Bound => "bound",
        }
    }
}
#[derive(Debug, Deserialize)]
pub struct ExecutionEvent {
    pub protocol_version: u32,
    pub request_id: String,
    pub reservation_id: String,
    pub transfer_id: u64,
    pub leg: Leg,
    pub state: LegState,
    pub event_id: String,
    pub http_status: Option<u16>,
    #[serde(default)]
    pub terminal_response: bool,
    #[serde(default)]
    pub not_submitted: bool,
    /// Engine token counts reported with `completed`: `cached_tokens` from P,
    /// `completion_tokens` / `spec_verify_ct` from D.
    #[serde(default)]
    pub usage: Option<LegUsage>,
    /// Late D choose: the decode candidate (`cand_idx`) this event is about.
    /// Only `bound` and `failed` are reported per candidate; the request's
    /// logical D-leg (the bound candidate) keeps the plain decode events.
    #[serde(default)]
    pub candidate: Option<usize>,
    /// The engine's abort message of a failed leg (bounded), when its error
    /// answer names one, e.g. a symmetric prefill abort answered HTTP 503
    /// "aborted by client" (see [`is_terminal_engine_abort`]).
    #[serde(default)]
    pub engine_abort: Option<String>,
}

/// Engine abort messages that end a symmetric prefill: SGLang answers such an
/// abort with HTTP 503 (late D) where an ordinary prefill abort is HTTP 200
/// with `finish_reason` abort.
pub const TERMINAL_PREFILL_ABORT_MESSAGES: [&str; 2] =
    ["aborted by client", "compute-before-allocation aborted"];

/// Whether a failed P-leg answer is terminal engine evidence although it is
/// an HTTP error: HTTP 503 whose body names one of
/// [`TERMINAL_PREFILL_ABORT_MESSAGES`] or a `finish_reason` abort (the
/// tokenizer reports it as `finish_reason abort: <message>`). Other 5xx
/// answers stay uncertain (real errors after possible admission).
pub fn is_terminal_engine_abort(http_status: Option<u16>, engine_abort: Option<&str>) -> bool {
    let Some(message) = engine_abort.filter(|_| http_status == Some(503)) else {
        return false;
    };
    let message = message.to_ascii_lowercase();
    message.starts_with("finish_reason abort")
        || TERMINAL_PREFILL_ABORT_MESSAGES
            .iter()
            .any(|m| message.contains(m))
}
#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub struct LegUsage {
    #[serde(default)]
    pub vision_matmul_flops: Option<u64>,
    #[serde(default)]
    pub image_attention_pairs: Option<u64>,
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub cached_tokens: Option<u64>,
    #[serde(default)]
    pub completion_tokens: Option<u64>,
    #[serde(default)]
    pub spec_verify_ct: Option<u64>,
}
#[derive(Debug)]
struct Execution {
    rid: String,
    token: String,
    transfer: u64,
    workers: [String; 2],
    states: [LegState; 2],
    terminal: [bool; 2],
    // Preserve historical upload evidence when current state becomes cancelled.
    ever_uploaded: [bool; 2],
    not_submitted: [bool; 2],
    ledger: LedgerHandle,
    /// The D-leg's place in its worker's in-flight count; dropped when the
    /// D-leg completes, fails or is cancelled (or the execution is removed).
    d_reservation: Option<DecodeReservation>,
    created: Instant,
    /// The last authenticated callback (or Backend cancel) for this
    /// execution: the quiet period of the unresolved D-leg expiry starts here.
    last_event: Instant,
    claimed: bool,
    prompt_tokens: u64,
    prefill_usage: Option<LegUsage>,
    decode_usage: Option<LegUsage>,
    /// The D-leg was aborted to move it off a worker draining toward prefill.
    migrating: bool,
    /// The P-leg's placement cost entry (`--symmetric-pd-p-select cost-function`). It
    /// follows the P-leg like `ledger`: dispatched on upload, finished on
    /// terminal P evidence, unknown on an uncertain failure. Dropped with the
    /// execution (before dispatch: rolled back; after: unknown).
    cost: Option<crate::prefill_work::Reservation>,
    /// Late D choose (`--pd-late-decode choose`): the candidates and the
    /// late-bind table entry. `workers[1]` is the arrival pick until a
    /// candidate binds, then the bound candidate.
    late: Option<LateLeg>,
}

/// What a prepared plan with late D choose adds ([`Executions::install`]):
/// the late-bind table, the request's entry (registered before the plan
/// leaves; its guard removes it with the execution), the P-leg `kv_export`
/// naming every candidate and one `kv_import` per candidate.
pub struct PreparedLate {
    pub late: Arc<crate::late_bind::LateBind>,
    pub guard: crate::late_bind::EntryGuard,
    pub kv_export: Value,
    pub imports: Vec<Value>,
    /// Counts a D-leg on a worker's in-flight decode count (the backend
    /// ledger): the bound candidate takes the arrival pick's place.
    pub reserve_decode: ReserveDecode,
}

/// See [`PreparedLate::reserve_decode`].
pub type ReserveDecode = Arc<dyn Fn(&str) -> DecodeReservation + Send + Sync>;

/// The late-D state of one prepared execution.
struct LateLeg {
    late: Arc<crate::late_bind::LateBind>,
    /// Keeps the table entry until the execution is removed.
    guard: crate::late_bind::EntryGuard,
    /// Candidate URLs in `cand_idx` order.
    candidates: Vec<String>,
    /// The candidate whose D-leg answered 2xx.
    bound: Option<usize>,
    /// The bound leg's decode charge (late-bind ledger, and the fit
    /// reservation with `fit_native`). Released when the D-leg ends in any
    /// way (completed, failed, cancelled), as the D in-flight count.
    charge: Option<crate::late_fit::LateCharge>,
    /// Moves the D in-flight count to the bound candidate.
    reserve: ReserveDecode,
}
impl std::fmt::Debug for LateLeg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LateLeg")
            .field("candidates", &self.candidates)
            .field("bound", &self.bound)
            .field("charged", &self.charge.is_some())
            .finish()
    }
}
impl LateLeg {
    /// The D-leg ended in some way: drop its charge, bound or still pending
    /// on the entry (a decision whose leg never bound). Idempotent.
    fn end_decode(&mut self) {
        self.charge.take();
        self.late.drop_pending_charge(self.guard.entry());
    }
}
/// Apply one prepared P-leg observation to the placement cost entry, if any.
/// A failed trace write is logged; the engine request is already out.
fn observe_cost(
    cost: &mut Option<crate::prefill_work::Reservation>,
    rid: &str,
    f: impl FnOnce(&mut crate::prefill_work::Reservation) -> Result<(), String>,
) {
    if let Some(entry) = cost.as_mut() {
        if let Err(error) = f(entry) {
            tracing::error!(target: "prefill_work", rid, %error,
                "prepared cost lifecycle observation failed");
        }
    }
}
impl Execution {
    /// `(worker URL, RID)` of every engine that may hold this request: the P
    /// and the D-leg; for an unbound late-D request every candidate (each
    /// holds a waiting D-leg until the prefill admits or releases it).
    fn targets(&self) -> Vec<(String, String)> {
        let mut out = vec![(self.workers[0].clone(), self.rid.clone())];
        match self.late.as_ref().filter(|l| l.bound.is_none()) {
            Some(late) => {
                for url in &late.candidates {
                    if !out.iter().any(|(w, _)| w == url) {
                        out.push((url.clone(), self.rid.clone()));
                    }
                }
            }
            None => out.push((self.workers[1].clone(), self.rid.clone())),
        }
        out
    }
    /// Whether the D-leg is placed on `workers[1]`: always, except a late-D
    /// request before a candidate bound (its D-leg has no worker yet).
    fn decode_placed(&self) -> bool {
        self.late.as_ref().is_none_or(|l| l.bound.is_some())
    }
    /// The worker that owes this request's D-leg, if any yet: `workers[1]`
    /// once placed; for an unbound late-D request the candidate P reported
    /// admitted or the router decided (its D-leg may already run there
    /// before the tokenizer reports the bind), else none.
    fn decode_owner(&self) -> Option<&str> {
        match self.late.as_ref().filter(|l| l.bound.is_none()) {
            None => Some(self.workers[1].as_str()),
            Some(late) => late
                .guard
                .entry()
                .decode_target()
                .and_then(|i| late.candidates.get(i))
                .map(String::as_str),
        }
    }
    /// The D-leg ended in some way: it stops counting toward its worker
    /// (in-flight count, late-D charge).
    fn end_decode(&mut self) {
        self.d_reservation.take();
        if let Some(late) = self.late.as_mut() {
            late.end_decode();
        }
    }
    /// A per-candidate late-D event (`e.candidate = Some(idx)`).
    ///
    /// `bound`: the candidate's D-leg answered 2xx. The first one becomes
    /// the D-leg (`workers[1]`), takes the request's decode charge (moved
    /// from the decision if needed) and binds the session; a repeat is
    /// idempotent, another candidate is an error. `failed`: an unbound
    /// candidate's leg failed, so a later decision leaves it out.
    fn candidate_event(&mut self, idx: usize, state: LegState) -> Result<(), String> {
        let ended = matches!(
            self.states[1],
            LegState::Completed | LegState::Failed | LegState::Cancelled
        );
        let late = self
            .late
            .as_mut()
            .ok_or("candidate event on a plan without late D")?;
        if idx >= late.candidates.len() {
            return Err("unknown decode candidate".into());
        }
        match state {
            LegState::Bound => match late.bound {
                Some(b) if b == idx => Ok(()),
                Some(_) => Err("another decode candidate is already bound".into()),
                None => {
                    let charge = late.late.bind_charge(late.guard.entry(), idx);
                    late.bound = Some(idx);
                    // A D-leg that already ended keeps no charge and no
                    // in-flight count; otherwise both move to the winner.
                    late.charge = (!ended).then_some(charge);
                    if ended {
                        late.late.drop_pending_charge(late.guard.entry());
                    }
                    let winner = late.candidates[idx].clone();
                    let moved = (!ended).then(|| (late.reserve)(winner.as_str()));
                    self.d_reservation = moved;
                    self.workers[1] = winner;
                    Ok(())
                }
            },
            LegState::Failed => {
                if late.bound != Some(idx) {
                    late.late.mark_dead(late.guard.entry(), idx);
                }
                Ok(())
            }
            _ => Err("a decode candidate reports only bound or failed".into()),
        }
    }
    /// Both legs completed with their counts; failed or cancelled requests are not samples.
    fn sample(&self) -> Option<RequestSample> {
        if self.states != [LegState::Completed; 2] {
            return None;
        }
        let d = self.decode_usage?;
        Some(RequestSample {
            input_tokens: self.prompt_tokens,
            cached_tokens: self.prefill_usage?.cached_tokens?,
            output_tokens: d.completion_tokens?,
            verify_steps: d.spec_verify_ct,
            vision_matmul_flops: self.prefill_usage?.vision_matmul_flops.unwrap_or(0),
            image_attention_pairs: self.prefill_usage?.image_attention_pairs.unwrap_or(0),
        })
    }
}
#[derive(Debug)]
struct Tombstone {
    reservation: String,
    rid: String,
    token: String,
    transfer: u64,
    migrating: bool,
}
#[derive(Default)]
struct Records {
    active: HashMap<String, Execution>,
    pending: HashMap<String, String>,
    done: VecDeque<Tombstone>,
    seen_rids: HashSet<String>,
    reserved_expiry: VecDeque<(Instant, String)>,
    /// See [`Executions::with_decode_unresolved_expiry`]; `None` = never.
    decode_unresolved_expire: Option<Duration>,
    /// D-legs ended by that expiry, and the executions it retired (both legs
    /// then ended), since start.
    decode_unresolved_expired: u64,
    executions_expired: u64,
    /// See [`Executions::with_silent_expiry`]; `None` = never.
    silent_expire: Option<Duration>,
    /// Executions abandoned by their tokenizer, since start.
    silent_expired: u64,
}
impl Records {
    /// Keep bounded identity evidence for a retired execution.
    ///
    /// # Inputs
    /// `self` owns the lifecycle records; `id` is the reservation ID; `x` is
    /// the removed execution whose identity and credentials must be remembered.
    ///
    /// # Outputs
    /// Returns `()` after adding a tombstone. Keeps at most 4,096 tombstones
    /// and removes evicted request IDs from duplicate detection. This function
    /// does not retire the ledger handle; callers establish retirement first.
    fn remember(&mut self, id: String, x: Execution) {
        self.done.push_back(Tombstone {
            reservation: id,
            rid: x.rid,
            token: x.token,
            transfer: x.transfer,
            migrating: x.migrating,
        });
        while self.done.len() > 4096 {
            if let Some(old) = self.done.pop_front() {
                self.seen_rids.remove(&old.rid);
            }
        }
    }
    /// Release plans that remained unclaimed past their five-minute deadline.
    ///
    /// # Inputs
    /// `self` owns the ordered expiry queue and active execution records.
    ///
    /// # Outputs
    /// Returns `()` after removing due queue entries. Still-unclaimed active
    /// plans are retired and tombstoned; claimed plans retain their reservations
    /// because elapsed time does not prove engine termination.
    fn expire_unclaimed(&mut self) {
        while self
            .reserved_expiry
            .front()
            .is_some_and(|(at, _)| at.elapsed() >= Duration::from_secs(300))
        {
            let (_, id) = self.reserved_expiry.pop_front().unwrap();
            if self.active.get(&id).is_some_and(|x| !x.claimed) {
                let x = self.active.remove(&id).unwrap();
                x.ledger.retire();
                self.remember(id, x);
            }
        }
        self.expire_silent();
        self.sweep_reconciled();
        self.expire_unresolved_decode();
    }
    /// A claimed execution whose tokenizer stopped reporting (the process
    /// died, or its callbacks are lost): nothing else would end its legs, so
    /// it would hold its execution slot, its P reservation, its D in-flight
    /// count and a leg owed by each worker for ever, and a role drain of
    /// either worker would never end. After `silent_expire` without a
    /// callback, its D-leg ends by time, as an unresolved D-leg does. Its
    /// P-leg is not ended by time: the reservation becomes unknown, and the
    /// unknown reconciliation retires it once its worker has stopped listing
    /// the RID ([`Self::sweep_reconciled`] then ends the leg).
    fn expire_silent(&mut self) {
        let Some(after) = self.silent_expire else {
            return;
        };
        let mut expired = 0;
        let done: Vec<String> = self
            .active
            .iter_mut()
            .filter(|(_, x)| {
                x.claimed
                    && x.last_event.elapsed() >= after
                    && (!x.terminal[0] && !x.ledger.is_unknown()
                        || !x.terminal[1] && !x.not_submitted[1])
            })
            .filter_map(|(id, x)| {
                expired += 1;
                tracing::warn!(target: "symmetric_pd", rid = %x.rid, reservation = %id,
                    p_worker = %x.workers[0],
                    d_worker = %x.decode_owner().unwrap_or(x.workers[1].as_str()),
                    p_state = x.states[0].name(), d_state = x.states[1].name(),
                    quiet_s = x.last_event.elapsed().as_secs_f64(),
                    reason = "silent_execution_expired",
                    "prepared execution abandoned by its tokenizer: no callback; its D-leg ends                      and its P reservation waits for engine evidence");
                if !x.terminal[0] {
                    x.ledger.mark_unknown();
                }
                if !x.terminal[1] && !x.not_submitted[1] {
                    x.terminal[1] = true;
                    x.end_decode();
                }
                x.terminal[0].then(|| id.clone())
            })
            .collect();
        self.silent_expired += expired;
        for id in done {
            let x = self.active.remove(&id).unwrap();
            self.executions_expired += 1;
            self.remember(id, x);
        }
    }
    /// Safety net for a lost D-leg confirmation. The tokenizer reports a
    /// client-cancelled or failed D-leg first as `cancelled`/`failed` without
    /// terminal evidence and then normally confirms it (the engine's abort
    /// event, a completion). If that confirmation never comes, nothing else
    /// would ever end the D-leg: there is no engine-side reconciliation for
    /// D-legs (the P-leg has the unknown reconciliation), so the execution
    /// would count forever against the prepared-execution capacity, as an
    /// `unknown` D request in `decode_population`, and as a leg its D worker
    /// owes to a role drain.
    ///
    /// A claimed execution whose D-leg is `failed`/`cancelled` without
    /// terminal evidence and which received no callback for
    /// `decode_unresolved_expire` ends that D-leg here (logged, counted). Its
    /// D in-flight count already ended at the `failed`/`cancelled` event and
    /// the Backend already asked the engines to abort. A P-leg is never ended
    /// by time (an unknown P-leg its worker still lists stays owed); when the
    /// P-leg has already ended, the execution leaves the registry (tombstoned,
    /// so late callbacks stay idempotent; no workload sample). Executions
    /// whose D-leg is running, completed or confirmed failed are untouched.
    fn expire_unresolved_decode(&mut self) {
        let Some(after) = self.decode_unresolved_expire else {
            return;
        };
        let mut expired = 0;
        let done: Vec<String> = self
            .active
            .iter_mut()
            .filter(|(_, x)| {
                x.claimed
                    && !x.terminal[1]
                    && !x.not_submitted[1]
                    && matches!(x.states[1], LegState::Failed | LegState::Cancelled)
                    && x.last_event.elapsed() >= after
            })
            .filter_map(|(id, x)| {
                x.terminal[1] = true;
                x.end_decode();
                expired += 1;
                tracing::warn!(target: "symmetric_pd", rid = %x.rid, reservation = %id,
                    worker = %x.decode_owner().unwrap_or(x.workers[1].as_str()),
                    d_state = x.states[1].name(), p_state = x.states[0].name(),
                    p_terminal = x.terminal[0], quiet_s = x.last_event.elapsed().as_secs_f64(),
                    reason = "decode_unresolved_expired",
                    "prepared D-leg expired: failed/cancelled without terminal evidence and no callback; \
                     it no longer counts as unresolved, owed or in decode_population");
                x.terminal[0].then(|| id.clone())
            })
            .collect();
        self.decode_unresolved_expired += expired;
        for id in done {
            let x = self.active.remove(&id).unwrap();
            self.executions_expired += 1;
            self.remember(id, x);
        }
    }
    /// A claimed P-leg without terminal evidence whose reservation the
    /// Backend retired on engine evidence (the unknown reconciliation of
    /// external role control: its worker stopped listing the RID) is ended
    /// here too: it is no longer owed to its worker's drain, and an
    /// execution whose two legs have ended leaves the registry (tombstoned,
    /// so late callbacks stay idempotent; no workload sample).
    fn sweep_reconciled(&mut self) {
        let done: Vec<String> = self
            .active
            .iter_mut()
            .filter(|(_, x)| x.claimed && !x.terminal[0] && x.ledger.is_retired())
            .filter_map(|(id, x)| {
                x.terminal[0] = true;
                tracing::info!(target: "symmetric_pd", rid = %x.rid, worker = %x.workers[0],
                    "prepared P-leg ended by the unknown reconciliation");
                x.terminal[1].then(|| id.clone())
            })
            .collect();
        for id in done {
            let x = self.active.remove(&id).unwrap();
            self.remember(id, x);
        }
    }
}
#[derive(Default)]
pub struct Executions {
    records: Mutex<Records>,
    workload: Option<Arc<CompletedWorkload>>,
}
impl Executions {
    /// Add every request whose two legs complete to `workload`.
    pub fn with_workload(workload: Arc<CompletedWorkload>) -> Self {
        Self {
            records: Mutex::default(),
            workload: Some(workload),
        }
    }
    /// End a claimed execution's D-leg that was reported `failed`/`cancelled`
    /// without terminal evidence once no callback came for `after` (`None` =
    /// never, the default). See [`PreparedConfig::decode_unresolved_expire_secs`].
    pub fn with_decode_unresolved_expiry(self, after: Option<Duration>) -> Self {
        self.records.lock().decode_unresolved_expire = after;
        self
    }
    /// Treat a claimed execution without a callback for `after` as abandoned
    /// by its tokenizer (`None` = never, the default). See
    /// [`PreparedConfig::silent_expire_secs`].
    pub fn with_silent_expiry(self, after: Option<Duration>) -> Self {
        self.records.lock().silent_expire = after;
        self
    }
    /// Reserve request identity BEFORE awaiting placement, preventing two plans
    /// for the same request. The returned ID also identifies this execution attempt.
    /// Reserve a unique request identity before asynchronous placement begins.
    ///
    /// # Inputs
    /// `self` owns lifecycle records; `rid` is the request ID; `limit` bounds
    /// pending plus active executions, including unresolved engine work.
    ///
    /// # Outputs
    /// Returns a new reservation ID and records a pending plan. Expires old
    /// unclaimed plans first; returns an error if the RID was already planned
    /// or capacity is exhausted. Does not reserve worker load yet.
    pub fn begin(&self, rid: &str, limit: usize) -> Result<String, String> {
        let mut r = self.records.lock();
        r.expire_unclaimed();
        if r.seen_rids.contains(rid) {
            return Err("request already planned; do not repeat generation".into());
        }
        if r.pending.len() + r.active.len() >= limit {
            return Err("prepared executions at capacity (including unresolved requests)".into());
        }
        let id = uuid::Uuid::new_v4().to_string();
        r.seen_rids.insert(rid.into());
        r.pending.insert(rid.into(), id.clone());
        Ok(id)
    }
    /// Release an identity reservation when placement never became an execution.
    ///
    /// # Inputs
    /// `self` owns lifecycle records; `rid` and `reservation` must identify
    /// the same pending placement attempt.
    ///
    /// # Outputs
    /// Returns `()` after removing matching pending/duplicate-ID entries,
    /// or leaves records unchanged when they do not match. Does not cancel
    /// active engine work or alter an installed execution.
    pub fn abandon_unsubmitted(&self, rid: &str, reservation: &str) {
        let mut r = self.records.lock();
        if r.pending.get(rid).is_some_and(|id| id == reservation) {
            r.pending.remove(rid);
            r.seen_rids.remove(rid);
        }
    }
    /// Turn a reserved identity and split placement into a forwardable P/D plan.
    ///
    /// # Inputs
    /// `self` owns execution records; `cfg` supplies the callback endpoint;
    /// `prepared` supplies request/configuration IDs and input reference;
    /// `reservation` identifies the pending attempt; `d` owns the placement,
    /// worker handles, KV marks and backend load reservation.
    ///
    /// # Outputs
    /// Returns plan JSON with P/D targets, KV metadata and a callback token,
    /// and stores the active unclaimed execution. Returns an error for malformed
    /// fields, missing transfer ID, Local placement or a removed pending attempt.
    /// Local/cancelled placements explicitly retire the supplied load handle;
    /// other early validation errors do not retire it here. No engine upload occurs.
    pub fn install(
        &self,
        cfg: &PreparedConfig,
        prepared: &Value,
        reservation: &str,
        d: Decision,
        late: Option<PreparedLate>,
    ) -> Result<Value, String> {
        let rid = required_str(prepared, "request_id")?;
        // Ordinary /generate remains available for Local; prepared forwarding
        // currently guarantees the two-leg Split lifecycle only.
        if d.is_local() {
            d.ledger.retire();
            return Err(
                "prepared PD requires split placement; use the local generation path".into(),
            );
        }
        let transfer = d.placement.transfer_id.ok_or("split transfer id missing")?;
        let token = uuid::Uuid::new_v4().to_string();
        let config_id = required_str(prepared, "config_id")?;
        let mut plan = json!({"protocol_version":1,"request_id":rid,"config_id":config_id,
            "ref":prepared["ref"],"reservation_id":reservation,"transfer_id":transfer,
            "prefill":{"target":prepared_target(&d.p_worker.url, config_id),"kv_export":d.marks.kv_export,"pull_hint":d.pull_hint},
            "decode":{"target":prepared_target(&d.d_worker.url, config_id),"kv_import":d.marks.kv_import},
            "callback":{"endpoint":cfg.callback_url,"token":token}});
        // Late D choose: the P-leg exports to every candidate and carries the
        // symmetric `late_d` field; the tokenizer sends one D-leg per
        // candidate (`decode_candidates`, each with its own `kv_import` and
        // `late_d.cand_idx`). `decode` stays the arrival pick, the candidate
        // the prefill admits when no decision comes in time.
        let late = match late {
            Some(l) => {
                let entry = l.guard.entry();
                let k = entry.candidates.len();
                if k == 0 || l.imports.len() != k || entry.default_idx >= k {
                    d.ledger.retire();
                    return Err("late D plan without matching candidates".into());
                }
                let epoch = l.late.router_epoch();
                plan["prefill"]["kv_export"] = l.kv_export.clone();
                plan["prefill"]["late_d"] =
                    crate::late_bind::sym_prefill_field(epoch, k, entry.default_idx);
                plan["decode"] = json!({
                    "target": prepared_target(&entry.candidates[entry.default_idx].url, config_id),
                    "kv_import": l.imports[entry.default_idx],
                });
                plan["decode_candidates"] = Value::Array(
                    entry
                        .candidates
                        .iter()
                        .zip(&l.imports)
                        .enumerate()
                        .map(|(i, (w, import))| {
                            json!({
                                "target": prepared_target(&w.url, config_id),
                                "kv_import": import,
                                "late_d": crate::late_bind::decode_field(epoch, k, i),
                            })
                        })
                        .collect(),
                );
                let candidates = entry.candidates.iter().map(|w| w.url.clone()).collect();
                Some(LateLeg {
                    late: l.late,
                    guard: l.guard,
                    candidates,
                    bound: None,
                    charge: None,
                    reserve: l.reserve_decode,
                })
            }
            None => None,
        };
        let d_url = match &late {
            Some(l) => l.candidates[l.guard.entry().default_idx].clone(),
            None => d.d_worker.url.clone(),
        };
        let mut r = self.records.lock();
        if r.pending.get(rid).map(String::as_str) != Some(reservation) {
            d.ledger.retire();
            return Err("placement was cancelled before dispatch".into());
        }
        r.pending.remove(rid);
        r.active.insert(
            reservation.into(),
            Execution {
                rid: rid.into(),
                token,
                transfer,
                workers: [d.p_worker.url.clone(), d_url],
                states: [LegState::Reserved; 2],
                terminal: [false; 2],
                ever_uploaded: [false; 2],
                not_submitted: [false; 2],
                ledger: d.ledger,
                d_reservation: d.d_reservation,
                created: Instant::now(),
                last_event: Instant::now(),
                claimed: false,
                prompt_tokens: prepared["prompt_tokens"].as_u64().unwrap_or(0),
                prefill_usage: None,
                decode_usage: None,
                migrating: false,
                cost: d.cost_reservation,
                late,
            },
        );
        r.reserved_expiry
            .push_back((Instant::now(), reservation.into()));
        Ok(plan)
    }
    /// Returns engines to cancel for uncertain/failed exchanges. An HTTP abort
    /// acknowledgement is NOT confirmation that engine work has terminated.
    /// Authenticate a forwarding callback and advance one execution lifecycle.
    ///
    /// # Inputs
    /// `self` owns lifecycle records; `e` identifies the request, reservation,
    /// transfer, leg and event evidence; `token` authenticates the callback.
    ///
    /// # Outputs
    /// Returns `(worker URL, RID)` pairs requiring best-effort abort, or an
    /// error for invalid protocol, credentials, identity or transition evidence.
    /// Claims authorize dispatch; terminal P evidence retires its load reservation.
    /// Uncertain failures retain work, late uploads cannot undo cancellation,
    /// and completed executions become tombstones for idempotent callbacks.
    /// This method does not send the returned abort requests itself.
    pub fn event(&self, e: &ExecutionEvent, token: &str) -> Result<Vec<(String, String)>, String> {
        if e.protocol_version != 1
            || e.state == LegState::Reserved
            || (e.not_submitted && e.state != LegState::Failed)
            || (e.state == LegState::Bound && e.candidate.is_none())
            || (e.candidate.is_some()
                && (e.leg != Leg::Decode
                    || e.not_submitted
                    || !matches!(e.state, LegState::Bound | LegState::Failed)))
        {
            return Err("invalid event".into());
        }
        let leg = match e.leg {
            Leg::Prefill => 0,
            Leg::Decode => 1,
        };
        let name = match (leg, e.candidate) {
            (0, _) => "prefill".to_string(),
            (_, None) => "decode".to_string(),
            (_, Some(i)) => format!("decode#{i}"),
        };
        if e.event_id != format!("{}:{}:{}", e.reservation_id, name, e.state.name()) {
            return Err("event identity mismatch".into());
        }
        let mut r = self.records.lock();
        r.expire_unclaimed();
        let Some(x) = r.active.get_mut(&e.reservation_id) else {
            if e.state == LegState::Claimed {
                return Err("expired or unknown plan; generation must not start".into());
            }
            return if r.done.iter().any(|x| {
                x.reservation == e.reservation_id
                    && x.rid == e.request_id
                    && x.transfer == e.transfer_id
                    && x.token == token
            }) {
                Ok(vec![])
            } else {
                Err("unknown execution or invalid credentials".into())
            };
        };
        if x.rid != e.request_id || x.transfer != e.transfer_id || x.token != token {
            return Err("execution identity mismatch".into());
        }
        x.last_event = Instant::now();
        if e.state == LegState::Claimed {
            if e.leg != Leg::Prefill {
                return Err("claim must identify the prefill leg".into());
            }
            if x.states
                .iter()
                .any(|s| matches!(s, LegState::Failed | LegState::Cancelled))
                || x.terminal.iter().any(|v| *v)
            {
                return Err("cancelled or terminal plan; generation must not start".into());
            }
            if !x.claimed {
                x.claimed = true;
                x.states = [LegState::Claimed; 2];
                // The claim authorizes the P send (as the proxy path marks the
                // entry just before its send); the later upload is idempotent.
                observe_cost(&mut x.cost, &x.rid, |c| c.mark_dispatched());
            }
            return Ok(vec![]);
        }
        if !x.claimed {
            return Err("plan must be claimed before dispatch".into());
        }
        if let Some(idx) = e.candidate {
            x.candidate_event(idx, e.state)?;
            return Ok(vec![]);
        }
        if e.not_submitted && x.ever_uploaded[leg] {
            return Err("not_submitted contradicts upload acknowledgement".into());
        }
        if e.state == LegState::Uploaded {
            x.ever_uploaded[leg] = true;
        }
        if x.terminal[leg] {
            return Ok(vec![]);
        }
        // An HTTP error can follow engine admission. It is diagnostic, not
        // evidence that computation or retained export ownership has ended,
        // except a P-leg engine abort answered HTTP 503 (symmetric prefill
        // with late D): that request is gone from the engine.
        let engine_abort = leg == 0
            && e.state == LegState::Failed
            && !x.terminal[0]
            && is_terminal_engine_abort(e.http_status, e.engine_abort.as_deref());
        if engine_abort {
            tracing::warn!(worker=%x.workers[0], rid=%x.rid, http_status=?e.http_status,
                engine_abort=%e.engine_abort.as_deref().unwrap_or(""),
                "prefill engine abort is terminal: P reservation retired");
        }
        let terminal_evidence = e.terminal_response || e.not_submitted || engine_abort;
        if let Some(status) = e.http_status {
            tracing::debug!(worker=%x.workers[leg],rid=%x.rid,http_status=status,
                terminal_response=e.terminal_response,"prepared engine HTTP response observed");
        }
        if x.states[leg] == e.state && !(e.state == LegState::Failed && terminal_evidence) {
            return Ok(vec![]);
        }
        // A late upload cannot undo cancellation. Repeat the idempotent abort
        // because the earlier abort may have arrived before this submission.
        if e.state == LegState::Uploaded
            && matches!(x.states[leg], LegState::Failed | LegState::Cancelled)
        {
            if leg == 0 {
                x.ledger.mark_unknown();
                observe_cost(&mut x.cost, &x.rid, |c| {
                    c.mark_unknown("prepared_upload_after_cancel")
                });
            }
            return Ok(x.targets());
        }
        if leg == 0 && e.state == LegState::Uploaded {
            x.ledger.mark_dispatched();
            observe_cost(&mut x.cost, &x.rid, |c| c.mark_dispatched());
        }
        if leg == 0
            && matches!(e.state, LegState::Failed | LegState::Cancelled)
            && !terminal_evidence
        {
            x.ledger.mark_unknown();
            let reason = format!(
                "prepared_prefill_{}_without_terminal_evidence",
                e.state.name()
            );
            observe_cost(&mut x.cost, &x.rid, |c| c.mark_unknown(&reason));
        }
        x.not_submitted[leg] |= e.not_submitted;
        x.states[leg] = e.state;
        x.terminal[leg] =
            e.state == LegState::Completed || (e.state == LegState::Failed && terminal_evidence);
        if e.state == LegState::Completed {
            if leg == 0 {
                x.prefill_usage = e.usage;
            } else {
                x.decode_usage = e.usage;
            }
        }
        if leg == 0 && x.terminal[0] {
            x.ledger.retire();
            // Only a completed P credits its tokens to the cost-function auto rate.
            let reason = if e.state == LegState::Completed {
                "prefill_response_success"
            } else {
                "prefill_response_error"
            };
            observe_cost(&mut x.cost, &x.rid, |c| c.finish(reason));
        }
        // Like the gateway's per-request load, the D-leg stops counting toward
        // its worker once it ends in any way, even if P is still unresolved.
        if leg == 1
            && matches!(
                e.state,
                LegState::Completed | LegState::Failed | LegState::Cancelled
            )
        {
            x.end_decode();
        }
        // P compute completion does not release exported KV ownership. If D
        // fails later, still ask both engines to cancel the exchange.
        let cancel = if matches!(e.state, LegState::Failed | LegState::Cancelled)
            && !x.not_submitted.iter().all(|v| *v)
        {
            x.targets()
        } else {
            vec![]
        };
        if x.terminal.iter().all(|v| *v) {
            let x = r.active.remove(&e.reservation_id).unwrap();
            if let (Some(w), Some(s)) = (&self.workload, x.sample()) {
                w.add(s);
            }
            r.remember(e.reservation_id.clone(), x);
        }
        Ok(cancel)
    }
    /// Cancel an execution without mistaking abort acknowledgement for completion.
    ///
    /// # Inputs
    /// `self` owns lifecycle records; `reservation` identifies the plan;
    /// `token` must match its callback credential.
    ///
    /// # Outputs
    /// Returns no targets for matching tombstones or unclaimed plans, retiring
    /// an unclaimed plan immediately. For claimed plans, marks unresolved legs
    /// cancelled and returns both engine URL/RID pairs while retaining uncertain
    /// work. Unknown plans or wrong credentials return an error; abort delivery
    /// and terminal confirmation remain separate operations.
    pub fn cancel(&self, reservation: &str, token: &str) -> Result<Vec<(String, String)>, String> {
        let mut r = self.records.lock();
        r.expire_unclaimed();
        if r.done
            .iter()
            .any(|x| x.reservation == reservation && x.token == token)
        {
            return Ok(vec![]);
        }
        let x = r.active.get_mut(reservation).ok_or("unknown execution")?;
        if x.token != token {
            return Err("invalid credentials".into());
        }
        if !x.claimed {
            let x = r.active.remove(reservation).unwrap();
            x.ledger.retire();
            r.remember(reservation.into(), x);
            return Ok(vec![]);
        }
        x.last_event = Instant::now();
        x.ledger.mark_unknown();
        if !x.terminal[0] {
            observe_cost(&mut x.cost, &x.rid, |c| c.mark_unknown("prepared_cancel"));
        }
        x.end_decode();
        for i in 0..2 {
            if !x.terminal[i] {
                x.states[i] = LegState::Cancelled;
            }
        }
        Ok(x.targets())
    }
    /// Legs `worker` still owes under existing plans, claimed or not: every leg
    /// placed on it that is neither terminal nor known never to have been sent.
    pub fn legs_owed(&self, worker: &str) -> usize {
        let mut records = self.records.lock();
        records.expire_unclaimed();
        let mut owed = 0;
        for x in records.active.values() {
            for leg in 0..2 {
                // A late-D request's waiting candidates are not owed: the
                // router never binds a worker that left the decode role
                // while another candidate remains, and the prefill releases
                // them at its bind. The decided (or admitted) candidate is
                // owed before the tokenizer reports the bind.
                let owner = if leg == 0 {
                    Some(x.workers[0].as_str())
                } else {
                    x.decode_owner()
                };
                if owner == Some(worker) && !x.terminal[leg] && !x.not_submitted[leg] {
                    owed += 1;
                }
            }
        }
        owed
    }
    /// Move the D-legs still running on `worker` to other workers.
    ///
    /// # Outputs
    /// Marks each claimed execution whose D-leg on `worker` has not ended as
    /// migrating and returns that D-leg's `(worker URL, request ID)` for the
    /// caller to abort. The tokenizer streaming the request sees the abort,
    /// confirms it through [`Self::migration_status`] and resumes the request on a
    /// newly placed pair. A D-leg still waiting for its P-leg moves too; its
    /// failure report then aborts that P-leg. Unclaimed plans are not touched.
    pub fn migrate_decode(&self, worker: &str) -> Vec<(String, String)> {
        let mut records = self.records.lock();
        let mut targets = Vec::new();
        for x in records.active.values_mut() {
            if x.claimed
                && x.decode_placed()
                && x.workers[1] == worker
                && !x.terminal[1]
                && !x.not_submitted[1]
            {
                x.migrating = true;
                targets.push((x.workers[1].clone(), x.rid.clone()));
            }
        }
        targets
    }
    /// Whether an execution's D-leg was aborted by [`Self::migrate_decode`].
    ///
    /// # Inputs
    /// `reservation` identifies the execution; `token` is its callback credential.
    ///
    /// # Outputs
    /// The flag of the active execution or of its tombstone; an error for an
    /// unknown execution or a wrong credential.
    pub fn migration_status(&self, reservation: &str, token: &str) -> Result<bool, String> {
        let records = self.records.lock();
        if let Some(x) = records.active.get(reservation) {
            return if x.token == token {
                Ok(x.migrating)
            } else {
                Err("invalid credentials".into())
            };
        }
        records
            .done
            .iter()
            .find(|t| t.reservation == reservation && t.token == token)
            .map(|t| t.migrating)
            .ok_or_else(|| "unknown execution or invalid credentials".into())
    }
    #[cfg(test)]
    /// Install a claimed plan with P on `workers[0]` and D on `workers[1]`, for
    /// tests elsewhere in the crate that need legs on a worker.
    pub(crate) fn install_claimed_for_test(
        &self,
        reservation: &str,
        token: &str,
        workers: [&str; 2],
        ledger: LedgerHandle,
    ) {
        self.records.lock().active.insert(
            reservation.into(),
            Execution {
                rid: format!("rid-{reservation}"),
                token: token.into(),
                transfer: 1,
                workers: workers.map(String::from),
                states: [LegState::Claimed; 2],
                terminal: [false; 2],
                ever_uploaded: [false; 2],
                not_submitted: [false; 2],
                ledger,
                d_reservation: None,
                created: Instant::now(),
                last_event: Instant::now(),
                claimed: true,
                prompt_tokens: 1,
                prefill_usage: None,
                decode_usage: None,
                migrating: false,
                cost: None,
                late: None,
            },
        );
    }
    #[cfg(test)]
    /// Pretend `reservation` had no callback for `secs` seconds (D-leg expiry
    /// tests elsewhere in the crate).
    pub(crate) fn age_for_test(&self, reservation: &str, secs: u64) {
        self.records
            .lock()
            .active
            .get_mut(reservation)
            .unwrap()
            .last_event = Instant::now() - Duration::from_secs(secs);
    }
    /// Nonterminal claimed D requests, counted once from the authoritative ledger.
    /// P-not-completed work is potential supply, not evidence of runnable decode.
    /// Unknown requests are reported separately and must not be assumed runnable.
    pub fn decode_population(&self) -> Value {
        let mut records = self.records.lock();
        records.expire_silent();
        records.sweep_reconciled();
        records.expire_unresolved_decode();
        let mut workers = std::collections::BTreeMap::<String, [u64; 3]>::new();
        for e in records.active.values() {
            if !e.claimed || e.terminal[1] || e.not_submitted[1] {
                continue;
            }
            // An unbound late-D request counts on its decided candidate,
            // else on its arrival pick.
            let worker = e.decode_owner().unwrap_or(&e.workers[1]).to_string();
            let counts = workers.entry(worker).or_default();
            let unknown = e
                .states
                .iter()
                .any(|s| matches!(s, LegState::Failed | LegState::Cancelled));
            let index = if unknown {
                2
            } else if e.states[0] == LegState::Completed {
                0
            } else {
                1
            };
            counts[index] += 1;
        }
        json!({"fields": ["p_completed", "p_pending", "unknown"], "workers": workers})
    }

    /// Report lifecycle counts for prepared-input observability.
    ///
    /// # Inputs
    /// `self` supplies the synchronized pending and active execution records.
    ///
    /// # Outputs
    /// Returns JSON counts for pending, active, unresolved and older-than-300s
    /// executions. Expires due unclaimed plans and unresolved D-legs before
    /// counting; claimed uncertain work is otherwise retained and included in
    /// capacity. `decode_unresolved_expired` / `executions_expired` count the
    /// D-legs and executions the D-leg expiry ended since start;
    /// `decode_unresolved_expire_s` is its setting (null = never).
    pub fn summary(&self) -> Value {
        let mut r = self.records.lock();
        r.expire_unclaimed();
        json!({"pending":r.pending.len(),"active":r.active.len(),
            "unresolved":r.active.values().filter(|e|e.states.iter().any(|s|matches!(s,LegState::Failed|LegState::Cancelled))).count(),
            "older_than_300s":r.active.values().filter(|e|e.created.elapsed()>Duration::from_secs(300)).count(),
            "decode_unresolved_expired":r.decode_unresolved_expired,
            "silent_expired":r.silent_expired,
            "executions_expired":r.executions_expired,
            "decode_unresolved_expire_s":r.decode_unresolved_expire.map(|d| d.as_secs_f64())})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::state::BackendState;
    /// Create a split-plan lifecycle fixture with a matching prefill reservation.
    ///
    /// # Inputs
    /// No arguments. Uses fixed request/transfer IDs, callback token and
    /// placeholder P/D URLs; starts no HTTP or engine services.
    ///
    /// # Outputs
    /// Returns `(Executions, BackendState)` sharing one 4,096-token reservation
    /// and an active, unclaimed two-leg plan for lifecycle assertions.
    fn fixture() -> (Executions, BackendState) {
        fixture_with(Executions::default())
    }
    /// [`fixture`] with the caller's `Executions` (e.g. one adding to a workload).
    fn fixture_with(executions: Executions) -> (Executions, BackendState) {
        let mut state = BackendState::new(Arc::new(tokio::sync::Notify::new()));
        executions.records.lock().active.insert(
            "r".into(),
            Execution {
                rid: "req".into(),
                token: "secret".into(),
                transfer: 1,
                workers: ["http://p".into(), "http://d".into()],
                states: [LegState::Reserved; 2],
                terminal: [false; 2],
                ever_uploaded: [false; 2],
                not_submitted: [false; 2],
                ledger: state.reserve("http://p", 4096),
                d_reservation: Some(state.reserve_decode("http://d")),
                created: Instant::now(),
                last_event: Instant::now(),
                claimed: false,
                prompt_tokens: 120_000,
                prefill_usage: None,
                decode_usage: None,
                migrating: false,
                cost: None,
                late: None,
            },
        );
        (executions, state)
    }
    /// The fixture plan with a placement cost entry for its P-leg (`http://p`).
    fn fixture_with_cost() -> (
        Executions,
        BackendState,
        Arc<crate::prefill_work::PrefillWorkBook>,
    ) {
        let (e, s) = fixture();
        let book = crate::prefill_work::PrefillWorkBook::new(None, "test", None).unwrap();
        let candidate = crate::prefill_work::Candidate {
            worker_url: "http://p".into(),
            local_hit: 0,
            gpu_hit: None,
            expected_hit: 0,
            hint: None,
        };
        let (_, reservation, _) = book
            .reserve_cost(
                4096,
                Some("req".into()),
                vec![candidate],
                Default::default(),
            )
            .unwrap();
        e.records.lock().active.get_mut("r").unwrap().cost = Some(reservation);
        (e, s, book)
    }
    fn cost_status(e: &Executions) -> Value {
        e.records.lock().active["r"]
            .cost
            .as_ref()
            .unwrap()
            .debuginfo()["status"]
            .clone()
    }
    /// The prepared (tokenizer) path drives the P-leg cost entry as the proxy
    /// path does: reserved until the claim, dispatched on claim, finished on
    /// terminal P evidence; an uncertain failure or a claimed cancel leaves it
    /// unknown until terminal evidence arrives.
    #[test]
    fn prepared_p_leg_events_drive_the_cost_entry() {
        let (e, _s, _book) = fixture_with_cost();
        assert_eq!(cost_status(&e), "reserved");
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        assert_eq!(cost_status(&e), "dispatched");
        e.event(&ev(Leg::Prefill, LegState::Uploaded), "secret")
            .unwrap();
        assert_eq!(cost_status(&e), "dispatched");
        e.event(&ev(Leg::Prefill, LegState::Completed), "secret")
            .unwrap();
        assert_eq!(cost_status(&e), "finished");
        assert_eq!(cost_reason(&e), "prefill_response_success");

        let (e, _s, _book) = fixture_with_cost();
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        e.event(&ev(Leg::Prefill, LegState::Uploaded), "secret")
            .unwrap();
        e.event(&ev(Leg::Prefill, LegState::Failed), "secret")
            .unwrap();
        assert_eq!(cost_status(&e), "unknown");
        let mut confirmed = ev(Leg::Prefill, LegState::Failed);
        confirmed.terminal_response = true;
        e.event(&confirmed, "secret").unwrap();
        assert_eq!(cost_status(&e), "finished");
        assert_eq!(cost_reason(&e), "prefill_response_error");

        let (e, _s, _book) = fixture_with_cost();
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        e.event(&ev(Leg::Prefill, LegState::Uploaded), "secret")
            .unwrap();
        e.cancel("r", "secret").unwrap();
        assert_eq!(cost_status(&e), "unknown");
        assert_eq!(cost_reason(&e), "prepared_cancel");
    }
    fn cost_reason(e: &Executions) -> Value {
        e.records.lock().active["r"]
            .cost
            .as_ref()
            .unwrap()
            .debuginfo()["reason"]
            .clone()
    }
    #[test]
    fn legs_owed_count_unclaimed_plans_until_each_leg_ends() {
        let (e, _s) = fixture();
        // The fixture plan (P on http://p, D on http://d) is placed but not claimed.
        assert_eq!((e.legs_owed("http://p"), e.legs_owed("http://d")), (1, 1));
        e.records.lock().active.get_mut("r").unwrap().terminal[0] = true;
        assert_eq!((e.legs_owed("http://p"), e.legs_owed("http://d")), (0, 1));
        e.records.lock().active.get_mut("r").unwrap().not_submitted[1] = true;
        assert_eq!(e.legs_owed("http://d"), 0);
    }
    #[test]
    fn drain_poll_expires_unclaimed_plans_but_never_claimed_work() {
        for claimed in [false, true] {
            let (e, mut state) = fixture();
            if claimed {
                e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
                    .unwrap();
            }
            e.records
                .lock()
                .reserved_expiry
                .push_back((Instant::now() - Duration::from_secs(301), "r".into()));
            let expected = usize::from(claimed);
            assert_eq!(e.legs_owed("http://p"), expected);
            assert_eq!(e.legs_owed("http://d"), expected);
            assert_eq!(state.view("http://p").active_prefill_reqs, expected as u64);
            if !claimed {
                assert!(e
                    .event(&ev(Leg::Prefill, LegState::Claimed), "secret")
                    .is_err());
            }
        }
    }
    #[test]
    fn migrate_decode_marks_running_d_legs_and_the_mark_outlives_the_execution() {
        let (e, _s) = fixture();
        assert!(
            e.migrate_decode("http://d").is_empty(),
            "an unclaimed plan is not moved"
        );
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        assert!(e.migrate_decode("http://p").is_empty(), "only D-legs move");
        assert_eq!(e.migration_status("r", "secret"), Ok(false));
        assert_eq!(
            e.migrate_decode("http://d"),
            vec![("http://d".to_string(), "req".to_string())]
        );
        assert_eq!(e.migration_status("r", "secret"), Ok(true));
        assert!(e.migration_status("r", "wrong").is_err());
        // The aborted D-leg's failure report aborts the P-leg still running.
        let mut aborted = ev(Leg::Decode, LegState::Failed);
        aborted.terminal_response = true;
        let cancel = e.event(&aborted, "secret").unwrap();
        assert!(cancel.contains(&("http://p".to_string(), "req".to_string())));
        assert_eq!(e.legs_owed("http://d"), 0);
        let mut p_aborted = ev(Leg::Prefill, LegState::Failed);
        p_aborted.terminal_response = true;
        e.event(&p_aborted, "secret").unwrap();
        assert_eq!(e.summary()["active"], 0);
        assert_eq!(
            e.migration_status("r", "secret"),
            Ok(true),
            "kept by the tombstone"
        );
        assert!(e.migration_status("other", "secret").is_err());
    }
    #[test]
    fn decode_population_separates_p_supply_unknown_and_unclaimed() {
        let (e, _state) = fixture();
        assert_eq!(e.decode_population()["workers"], json!({}));
        {
            let mut r = e.records.lock();
            r.active.get_mut("r").unwrap().claimed = true;
        }
        assert_eq!(
            e.decode_population()["workers"]["http://d"],
            json!([0, 1, 0])
        );
        {
            let mut r = e.records.lock();
            r.active.get_mut("r").unwrap().states[0] = LegState::Completed;
        }
        assert_eq!(
            e.decode_population()["workers"]["http://d"],
            json!([1, 0, 0])
        );
        e.cancel("r", "secret").unwrap();
        assert_eq!(
            e.decode_population()["workers"]["http://d"],
            json!([0, 0, 1])
        );
        e.records.lock().active.get_mut("r").unwrap().terminal[1] = true;
        assert_eq!(e.decode_population()["workers"], json!({}));
    }

    /// Build a valid callback envelope for the fixed lifecycle fixture.
    ///
    /// # Inputs
    /// `leg` selects P or D; `state` chooses the reported lifecycle transition.
    ///
    /// # Outputs
    /// Returns an event with matching fixture IDs and deterministic event ID.
    /// Terminal-response and not-submitted evidence are initially false so tests
    /// must explicitly add them when needed.
    fn ev(leg: Leg, state: LegState) -> ExecutionEvent {
        let name = if leg == Leg::Prefill {
            "prefill"
        } else {
            "decode"
        };
        ExecutionEvent {
            protocol_version: 1,
            request_id: "req".into(),
            reservation_id: "r".into(),
            transfer_id: 1,
            leg,
            state,
            event_id: format!("r:{name}:{}", state.name()),
            http_status: None,
            terminal_response: false,
            not_submitted: false,
            usage: None,
            candidate: None,
            engine_abort: None,
        }
    }
    fn completed(leg: Leg, usage: LegUsage) -> ExecutionEvent {
        ExecutionEvent {
            usage: Some(usage),
            ..ev(leg, LegState::Completed)
        }
    }
    #[test]
    fn completed_requests_add_both_legs_counts_in_either_order() {
        let p = LegUsage {
            prompt_tokens: Some(120_000),
            cached_tokens: Some(115_000),
            vision_matmul_flops: Some(136),
            image_attention_pairs: Some(6),
            ..Default::default()
        };
        let d = LegUsage {
            completion_tokens: Some(1_000),
            spec_verify_ct: Some(250),
            ..Default::default()
        };
        for decode_first in [false, true] {
            let workload = Arc::new(CompletedWorkload::default());
            let (e, _s) = fixture_with(Executions::with_workload(Arc::clone(&workload)));
            e.begin("other", 10).unwrap();
            e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
                .unwrap();
            let order = if decode_first {
                [completed(Leg::Decode, d), completed(Leg::Prefill, p)]
            } else {
                [completed(Leg::Prefill, p), completed(Leg::Decode, d)]
            };
            for event in &order {
                e.event(event, "secret").unwrap();
            }
            let sums = workload.totals();
            assert_eq!(sums.vision_matmul_flops, 136.0);
            assert_eq!(sums.image_attention_pairs, 6.0);
            assert_eq!(sums.requests, 1);
            assert_eq!((sums.input_tokens, sums.uncached_tokens), (120_000, 5_000));
            assert_eq!((sums.output_tokens, sums.verify_steps), (1_000, 250));
        }
    }
    #[test]
    fn failed_or_uncounted_requests_are_not_added() {
        let p = LegUsage {
            cached_tokens: Some(115_000),
            ..Default::default()
        };
        let workload = Arc::new(CompletedWorkload::default());
        let (e, _s) = fixture_with(Executions::with_workload(Arc::clone(&workload)));
        e.begin("other", 10).unwrap();
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        e.event(&completed(Leg::Prefill, p), "secret").unwrap();
        let mut failed = ev(Leg::Decode, LegState::Failed);
        failed.terminal_response = true;
        e.event(&failed, "secret").unwrap();
        assert_eq!(workload.totals().requests, 0);
        // A completion without counts (an older tokenizer) is not a sample either.
        let (e, _s) = fixture_with(Executions::with_workload(Arc::clone(&workload)));
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        e.event(&completed(Leg::Prefill, p), "secret").unwrap();
        e.event(&ev(Leg::Decode, LegState::Completed), "secret")
            .unwrap();
        assert_eq!(workload.totals().requests, 0);
    }
    #[test]
    /// Verify P load retirement, retained D lifecycle and idempotent late callbacks.
    ///
    /// # Inputs
    /// No arguments. Claim and complete P, send a late upload, then complete D twice.
    ///
    /// # Outputs
    /// Returns `()` if all lifecycle/contract assertions hold; assertion failure panics.
    fn p_completion_retires_only_p_and_late_events_do_not_revive_it() {
        let (e, mut s) = fixture();
        assert_eq!(s.view("http://d").active_decode_reqs, 1);
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        e.event(&ev(Leg::Prefill, LegState::Completed), "secret")
            .unwrap();
        assert_eq!(s.view("http://p").active_prefill_reqs, 0);
        assert_eq!(s.view("http://d").active_decode_reqs, 1);
        assert_eq!(e.summary()["active"], 1);
        e.event(&ev(Leg::Prefill, LegState::Uploaded), "secret")
            .unwrap();
        e.event(&ev(Leg::Decode, LegState::Completed), "secret")
            .unwrap();
        assert_eq!(s.view("http://d").active_decode_reqs, 0);
        assert_eq!(e.summary()["active"], 0);
        e.event(&ev(Leg::Decode, LegState::Completed), "secret")
            .unwrap();
        assert!(e.cancel("r", "secret").unwrap().is_empty());
    }
    #[test]
    /// Verify uncertain failure requests two aborts and preserves P load until proof.
    ///
    /// # Inputs
    /// No arguments. Claim the plan, fail D transport, then confirm P termination.
    ///
    /// # Outputs
    /// Returns `()` if all lifecycle/contract assertions hold; assertion failure panics.
    fn failed_transport_keeps_work_until_terminal_engine_evidence() {
        let (e, mut s) = fixture();
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        assert_eq!(
            e.event(&ev(Leg::Decode, LegState::Failed), "secret")
                .unwrap()
                .len(),
            2
        );
        // The failed D-leg no longer counts toward its worker; P work stays.
        assert_eq!(s.view("http://d").active_decode_reqs, 0);
        assert_eq!(s.view("http://p").active_prefill_reqs, 1);
        assert_eq!(e.summary()["unresolved"], 1);
        let mut aborted = ev(Leg::Prefill, LegState::Failed);
        aborted.terminal_response = true;
        e.event(&aborted, "secret").unwrap();
        assert_eq!(s.view("http://p").active_prefill_reqs, 0);
    }
    #[test]
    /// Verify expired or unauthorized plans cannot proceed to dispatch.
    ///
    /// # Inputs
    /// No arguments. Expire one unclaimed plan and send bad credentials/unclaimed upload on
    /// another.
    ///
    /// # Outputs
    /// Returns `()` if all lifecycle/contract assertions hold; assertion failure panics.
    fn unclaimed_expiry_rejects_late_dispatch_and_wrong_identity() {
        let (e, mut s) = fixture();
        e.records
            .lock()
            .reserved_expiry
            .push_back((Instant::now() - Duration::from_secs(301), "r".into()));
        assert!(e
            .event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .is_err());
        assert_eq!(s.view("http://p").active_prefill_reqs, 0);
        let (e, _) = fixture();
        assert!(e
            .event(&ev(Leg::Prefill, LegState::Claimed), "wrong")
            .is_err());
        assert!(e
            .event(&ev(Leg::Prefill, LegState::Uploaded), "secret")
            .is_err());
    }
    #[test]
    /// Verify cancellation blocks dispatch authorization but accepts stronger terminal evidence.
    ///
    /// # Inputs
    /// No arguments. Cancel a claimed plan, retry claim/upload, then upgrade failure with
    /// terminal proof.
    ///
    /// # Outputs
    /// Returns `()` if all lifecycle/contract assertions hold; assertion failure panics.
    fn cancelled_claim_cannot_reauthorize_dispatch_and_failure_evidence_can_upgrade() {
        let (e, mut s) = fixture();
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        e.cancel("r", "secret").unwrap();
        assert!(e
            .event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .is_err());
        assert_eq!(
            e.event(&ev(Leg::Prefill, LegState::Uploaded), "secret")
                .unwrap()
                .len(),
            2
        );
        let uncertain = ev(Leg::Prefill, LegState::Failed);
        e.event(&uncertain, "secret").unwrap();
        assert_eq!(s.view("http://p").active_prefill_reqs, 1);
        let mut confirmed = ev(Leg::Prefill, LegState::Failed);
        confirmed.terminal_response = true;
        e.event(&confirmed, "secret").unwrap();
        assert_eq!(s.view("http://p").active_prefill_reqs, 0);
    }
    #[test]
    /// Verify HTTP error alone retains uncertain P work and requests aborts for both peers.
    ///
    /// # Inputs
    /// No arguments. Upload P, report HTTP 500 without proof, then confirm termination.
    ///
    /// # Outputs
    /// Returns `()` if all lifecycle/contract assertions hold; assertion failure panics.
    fn http_500_keeps_p_work_and_aborts_both_until_explicit_terminal_evidence() {
        let (e, mut s) = fixture();
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        e.event(&ev(Leg::Prefill, LegState::Uploaded), "secret")
            .unwrap();
        let mut failed = ev(Leg::Prefill, LegState::Failed);
        failed.http_status = Some(500);
        let targets = e.event(&failed, "secret").unwrap();
        assert_eq!(
            targets,
            vec![
                ("http://p".into(), "req".into()),
                ("http://d".into(), "req".into())
            ]
        );
        assert_eq!(s.view("http://p").active_prefill_reqs, 1);
        assert_eq!(s.view("http://p").unknown_prefill_reqs, 1);
        assert!(!e.records.lock().active["r"].terminal[0]);
        failed.terminal_response = true;
        e.event(&failed, "secret").unwrap();
        assert_eq!(s.view("http://p").active_prefill_reqs, 0);
        assert!(e.records.lock().active["r"].terminal[0]);
    }
    #[test]
    /// Verify contradictory not-submitted evidence is rejected and capacity remains reserved.
    ///
    /// # Inputs
    /// No arguments. For each leg, acknowledge upload, cancel, then claim it was never submitted.
    ///
    /// # Outputs
    /// Returns `()` if all lifecycle/contract assertions hold; assertion failure panics.
    fn cancelled_uploaded_leg_cannot_claim_it_was_never_submitted() {
        for leg in [Leg::Prefill, Leg::Decode] {
            let (e, mut s) = fixture();
            e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
                .unwrap();
            e.event(&ev(leg, LegState::Uploaded), "secret").unwrap();
            e.cancel("r", "secret").unwrap();
            let mut not_submitted = ev(leg, LegState::Failed);
            not_submitted.not_submitted = true;
            assert!(e.event(&not_submitted, "secret").is_err());
            assert_eq!(s.view("http://p").active_prefill_reqs, 1);
        }
    }
    #[test]
    /// Verify upload history survives cancellation and prevents false unsent proof.
    ///
    /// # Inputs
    /// No arguments. Cancel a claimed plan, acknowledge delayed P upload, then report P as unsent.
    ///
    /// # Outputs
    /// Returns `()` if all lifecycle/contract assertions hold; assertion failure panics.
    fn delayed_upload_after_cancel_still_blocks_not_submitted_proof() {
        let (e, mut s) = fixture();
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        e.cancel("r", "secret").unwrap();
        assert_eq!(
            e.event(&ev(Leg::Prefill, LegState::Uploaded), "secret")
                .unwrap()
                .len(),
            2
        );
        let mut failed = ev(Leg::Prefill, LegState::Failed);
        failed.not_submitted = true;
        assert!(e.event(&failed, "secret").is_err());
        assert_eq!(s.view("http://p").active_prefill_reqs, 1);
    }
    #[test]
    /// Verify unsent terminal evidence releases all capacity without authorizing another claim.
    ///
    /// # Inputs
    /// No arguments. Repeat a claim, then prove both legs were never submitted with duplicate
    /// callbacks.
    ///
    /// # Outputs
    /// Returns `()` if all lifecycle/contract assertions hold; assertion failure panics.
    fn lost_claim_ack_with_both_legs_definitely_unsent_releases_capacity() {
        let (e, mut s) = fixture();
        // Both claim acknowledgements are lost; T creates no engine tasks.
        let claim = ev(Leg::Prefill, LegState::Claimed);
        e.event(&claim, "secret").unwrap();
        e.event(&claim, "secret").unwrap();
        for (i, leg) in [Leg::Prefill, Leg::Decode].into_iter().enumerate() {
            let mut unsent = ev(leg, LegState::Failed);
            unsent.not_submitted = true;
            let targets = e.event(&unsent, "secret").unwrap();
            if i == 0 {
                assert_eq!(e.summary()["active"], 1);
            } else {
                assert!(targets.is_empty());
            }
            // Terminal callbacks are idempotent after each leg and tombstoning.
            assert!(e.event(&unsent, "secret").unwrap().is_empty());
        }
        assert_eq!(e.summary()["active"], 0);
        assert_eq!(e.summary()["unresolved"], 0);
        assert_eq!(s.view("http://p").active_prefill_reqs, 0);
        assert!(e.event(&claim, "secret").is_err());
    }
    #[test]
    /// Verify both engines are asked to abort to release retained KV ownership.
    ///
    /// # Inputs
    /// No arguments. Complete P, then separately exercise D failure and client cancellation.
    ///
    /// # Outputs
    /// Returns `()` if all lifecycle/contract assertions hold; assertion failure panics.
    fn completed_p_is_still_aborted_when_d_later_fails_or_client_cancels() {
        for use_cancel in [false, true] {
            let (e, mut s) = fixture();
            e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
                .unwrap();
            e.event(&ev(Leg::Prefill, LegState::Completed), "secret")
                .unwrap();
            assert_eq!(s.view("http://p").active_prefill_reqs, 0);
            let targets = if use_cancel {
                e.cancel("r", "secret").unwrap()
            } else {
                e.event(&ev(Leg::Decode, LegState::Failed), "secret")
                    .unwrap()
            };
            assert_eq!(
                targets,
                vec![
                    ("http://p".into(), "req".into()),
                    ("http://d".into(), "req".into())
                ]
            );
            assert_eq!(s.view("http://p").active_prefill_reqs, 0);
            assert_eq!(e.summary()["active"], 1);
        }
    }
    #[test]
    /// Verify signed hash bits are preserved and incompatible metadata is rejected.
    ///
    /// # Inputs
    /// No arguments. Build a valid EAGLE hash descriptor and variants with extra fields or
    /// wrong contracts.
    ///
    /// # Outputs
    /// Returns `()` if all lifecycle/contract assertions hold; assertion failure panics.
    fn prepared_hash_contract_preserves_signed_bits_and_rejects_native_hash() {
        let mut cfg = PreparedConfig {
            callback_url: "http://router/symmetric_pd/events".into(),
            max_inflight: 10,
            contracts: HashMap::from([(
                "glm-eagle".into(),
                HashContract {
                    page_size: 64,
                    is_eagle: true,
                    model_path: "/glm".into(),
                    cache_namespace: None,
                    image_config: None,
                },
            )]),
            decode_unresolved_expire_secs: DEFAULT_DECODE_UNRESOLVED_EXPIRE_SECS,
            silent_expire_secs: DEFAULT_SILENT_EXPIRE_SECS,
        };
        let mut p = json!({"protocol_version":1,"config_id":"glm-eagle","request_id":"req","ref":"input-ref","prompt_tokens":65,
            "content":{"hash_spec":{"mode":"dynamo","encoding":"signed_i64","implementation":"dynamo.llm.compute_block_hash_for_seq","page_size":64,"is_eagle":true,"model":"/glm","cache_namespace":null,"partial_blocks":false,"window_tokens":65},"blocks":[{"hash":-1,"valid_tokens":64}]}});
        let input = cfg.input(&p, "glm").unwrap();
        assert!(input.tokens.is_none());
        assert_eq!(input.hashes, Some(vec![u64::MAX]));
        let mut unknown_spec = p.clone();
        unknown_spec["content"]["hash_spec"]["input_ids"] = json!([1, 2]);
        assert!(cfg.input(&unknown_spec, "glm").is_err());
        let mut unknown_block = p.clone();
        unknown_block["content"]["blocks"][0]["tokens"] = json!([1, 2]);
        assert!(cfg.input(&unknown_block, "glm").is_err());
        let mut wrong_window = p.clone();
        wrong_window["content"]["hash_spec"]["window_tokens"] = json!(64);
        assert!(cfg.input(&wrong_window, "glm").is_err());
        p["content"]["hash_spec"]["mode"] = json!("sglang_native");
        assert!(cfg.input(&p, "glm").is_err());
        p["content"]["hash_spec"]["mode"] = json!("dynamo");
        let image = json!({"image_token_id":100,"vision_patch_size":14,
            "vision_downsample_ratio":3,"vision_max_n_token":384,
            "vision_min_pixels":147456,"vision_max_wh_ratio":8});
        cfg.contracts.get_mut("glm-eagle").unwrap().image_config = Some(image.clone());
        assert!(
            cfg.input(&p, "glm").is_err(),
            "image configuration must be advertised"
        );
        p["content"]["hash_spec"]["image_config"] = image;
        assert!(cfg.input(&p, "glm").is_ok());
        p["content"]["hash_spec"]["image_config"]["vision_patch_size"] = json!(16);
        assert!(
            cfg.input(&p, "glm").is_err(),
            "different layouts must not share a hash contract"
        );
    }
    #[test]
    /// Verify the D-leg leaves its worker's in-flight count when D completes before P,
    /// and that cancelling a claimed plan or expiring an unclaimed one also releases it.
    ///
    /// # Inputs
    /// No arguments. Three fixtures: D completes first; a claimed plan is cancelled;
    /// an unclaimed plan expires.
    ///
    /// # Outputs
    /// Returns `()` if all decode-count assertions hold; assertion failure panics.
    fn decode_count_follows_the_d_leg_lifecycle() {
        let (e, mut s) = fixture();
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        e.event(&ev(Leg::Decode, LegState::Completed), "secret")
            .unwrap();
        assert_eq!(s.view("http://d").active_decode_reqs, 0);
        assert_eq!(s.view("http://p").active_prefill_reqs, 1);

        let (e, mut s) = fixture();
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        assert_eq!(s.view("http://d").active_decode_reqs, 1);
        assert_eq!(e.cancel("r", "secret").unwrap().len(), 2);
        assert_eq!(s.view("http://d").active_decode_reqs, 0);

        let (e, mut s) = fixture();
        e.records
            .lock()
            .reserved_expiry
            .push_back((Instant::now() - Duration::from_secs(301), "r".into()));
        assert_eq!(e.summary()["active"], 0);
        assert_eq!(s.view("http://d").active_decode_reqs, 0);
    }

    // ---- late D choose on prepared executions ----

    use crate::config::{LateDecodeConfig, LateDecodeMode, LateDecodeRule, PdDecodeTokensConfig};
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::late_bind::{EntrySpec, LateBind};
    use crate::workers::{Worker, WorkerRegistry};

    fn sym(url: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(url.into()),
            url: url.into(),
            mode: WorkerMode::Symmetric,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: Some(8998),
        }))
    }

    /// The fixture plan as a late-D request: candidates d0, d1, d2 (arrival
    /// pick d0) in a symmetric `fit_req` table; `workers[1]` = d0. The
    /// backend ledger is shared with the bind's in-flight move.
    #[allow(clippy::type_complexity)]
    fn late_fixture() -> (
        Executions,
        Arc<Mutex<BackendState>>,
        Arc<LateBind>,
        Vec<Arc<Worker>>,
    ) {
        let (e, s) = fixture();
        let s = Arc::new(Mutex::new(s));
        let shared = Arc::clone(&s);
        let reserve: ReserveDecode = Arc::new(move |w: &str| shared.lock().reserve_decode(w));
        let ws: Vec<Arc<Worker>> = (0..3).map(|i| sym(&format!("http://d{i}"))).collect();
        let late = LateBind::new_symmetric(
            LateDecodeConfig {
                mode: LateDecodeMode::Choose,
                rule: Some(LateDecodeRule::FitReq),
                fit_capacity_tokens: Some(1_000_000),
                ..LateDecodeConfig::default()
            },
            PdDecodeTokensConfig::default(),
            Arc::new(WorkerRegistry::default()),
            Arc::new(Vec::new),
        );
        let guard = late.register(EntrySpec {
            owner: "tcp://p:1".into(),
            prefill_url: "http://p".into(),
            key: 1,
            rid: Some("req".into()),
            candidates: ws.clone(),
            default_idx: 0,
            charge_tokens: 1000,
        });
        {
            let mut r = e.records.lock();
            let x = r.active.get_mut("r").unwrap();
            x.workers[1] = ws[0].url.clone();
            x.d_reservation = Some(s.lock().reserve_decode(&ws[0].url));
            x.late = Some(LateLeg {
                late: Arc::clone(&late),
                guard,
                candidates: ws.iter().map(|w| w.url.clone()).collect(),
                bound: None,
                charge: None,
                reserve,
            });
        }
        (e, s, late, ws)
    }

    fn cand(idx: usize, state: LegState) -> ExecutionEvent {
        ExecutionEvent {
            candidate: Some(idx),
            event_id: format!("r:decode#{idx}:{}", state.name()),
            ..ev(Leg::Decode, state)
        }
    }

    fn charged(w: &Worker) -> usize {
        LateBind::charged(w).1
    }

    /// The candidate that answers 2xx becomes the D-leg: it is owed on its
    /// worker (drain), holds the request's decode charge until the D-leg
    /// ends, and a second candidate cannot bind.
    #[test]
    fn late_bound_candidate_becomes_the_d_leg_and_holds_the_charge() {
        let (e, _s, _late, ws) = late_fixture();
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        assert_eq!(
            e.legs_owed("http://d0"),
            0,
            "an unbound D-leg is owed nowhere"
        );
        e.event(&cand(1, LegState::Bound), "secret").unwrap();
        assert_eq!(e.records.lock().active["r"].workers[1], "http://d1");
        assert_eq!(charged(&ws[1]), 1);
        assert_eq!(e.legs_owed("http://d1"), 1);
        assert_eq!(e.legs_owed("http://d0"), 0);
        // A repeat is idempotent; another candidate is refused.
        e.event(&cand(1, LegState::Bound), "secret").unwrap();
        assert!(e.event(&cand(2, LegState::Bound), "secret").is_err());
        assert_eq!(charged(&ws[1]), 1);
        e.event(&ev(Leg::Decode, LegState::Uploaded), "secret")
            .unwrap();
        e.event(&ev(Leg::Decode, LegState::Completed), "secret")
            .unwrap();
        assert_eq!(
            charged(&ws[1]),
            0,
            "the D-leg ended: its charge is released"
        );
        assert_eq!(e.legs_owed("http://d1"), 0);
    }

    /// Before a bind a cancel reaches the P and every candidate (each holds
    /// a waiting D-leg); a drain moves no unbound leg.
    #[test]
    fn unbound_late_request_cancel_reaches_every_candidate() {
        let (e, _s, _late, _ws) = late_fixture();
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        assert!(e.migrate_decode("http://d0").is_empty());
        let mut targets = e.cancel("r", "secret").unwrap();
        targets.sort();
        let want: Vec<(String, String)> = ["http://d0", "http://d1", "http://d2", "http://p"]
            .iter()
            .map(|w| (w.to_string(), "req".to_string()))
            .collect();
        assert_eq!(targets, want);
    }

    /// A failed unbound candidate is left out of later decisions; the bound
    /// candidate's failure is the D-leg's, not a candidate mark.
    #[test]
    fn failed_candidate_is_marked_dead_until_bound() {
        let (e, _s, late, _ws) = late_fixture();
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        e.event(&cand(2, LegState::Failed), "secret").unwrap();
        let entry = |e: &Executions| {
            Arc::clone(
                e.records.lock().active["r"]
                    .late
                    .as_ref()
                    .unwrap()
                    .guard
                    .entry(),
            )
        };
        assert_eq!(entry(&e).dead_mask(), 0b100);
        e.event(&cand(0, LegState::Bound), "secret").unwrap();
        e.event(&cand(0, LegState::Failed), "secret").unwrap();
        assert_eq!(entry(&e).dead_mask(), 0b100);
        assert_eq!(late.pending(), 1, "the entry lives with the execution");
    }

    /// Candidate events are checked: only decode, only bound / failed, only
    /// on a late plan, with a candidate-scoped event id; `bound` needs one.
    #[test]
    fn candidate_events_are_validated() {
        let (e, _s, _late, _ws) = late_fixture();
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        assert!(e
            .event(&ev(Leg::Decode, LegState::Bound), "secret")
            .is_err());
        let mut on_prefill = cand(0, LegState::Bound);
        on_prefill.leg = Leg::Prefill;
        assert!(e.event(&on_prefill, "secret").is_err());
        assert!(e.event(&cand(0, LegState::Completed), "secret").is_err());
        let mut wrong_id = cand(0, LegState::Bound);
        wrong_id.event_id = "r:decode:bound".into();
        assert!(e.event(&wrong_id, "secret").is_err());
        assert!(e.event(&cand(3, LegState::Bound), "secret").is_err());
        // A plan without late D refuses candidate events.
        let (plain, _s) = fixture();
        plain
            .event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        assert!(plain.event(&cand(0, LegState::Bound), "secret").is_err());
    }

    /// The controller moves the bound D to prefill: the drain owes its leg
    /// and migrates it; the D-leg's end releases the charge. A request still
    /// unbound when its arrival pick leaves the decode role is not moved.
    #[test]
    fn bound_late_leg_drains_and_migrates_with_its_worker() {
        let (e, _s, _late, ws) = late_fixture();
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        e.event(&cand(2, LegState::Bound), "secret").unwrap();
        assert_eq!(e.legs_owed("http://d2"), 1);
        assert_eq!(
            e.migrate_decode("http://d2"),
            vec![("http://d2".to_string(), "req".to_string())]
        );
        assert_eq!(e.migration_status("r", "secret"), Ok(true));
        e.event(&ev(Leg::Decode, LegState::Failed), "secret")
            .unwrap();
        assert_eq!(charged(&ws[2]), 0);
        let pop = e.decode_population();
        assert!(pop["workers"].get("http://d0").is_none());
    }

    /// A decision charge left on the entry (the D-leg never bound) is
    /// released when the D-leg ends in any way, so an execution that lingers
    /// on an unknown P-leg holds no decode charge (else the charge leaks).
    #[test]
    fn unbound_decision_charge_is_released_when_the_d_leg_ends() {
        let (e, _s, late, ws) = late_fixture();
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        // The prefill asked for its winner: the decision charges it on the
        // entry, then the P-leg fails without terminal evidence (unknown).
        let w = late.decide_for_test("tcp://p:1", 1, 1) as usize;
        assert_eq!(charged(&ws[w]), 1);
        e.event(&ev(Leg::Prefill, LegState::Uploaded), "secret")
            .unwrap();
        e.event(&ev(Leg::Prefill, LegState::Failed), "secret")
            .unwrap();
        e.event(&ev(Leg::Decode, LegState::Cancelled), "secret")
            .unwrap();
        assert_eq!(late.pending(), 1, "the unknown P keeps the execution");
        for w in &ws {
            assert_eq!(charged(w), 0);
        }
    }

    /// Between the router's decision and the tokenizer's `bound` report the
    /// D-leg may already run on the decided candidate: a drain of that
    /// worker owes it, no other candidate does, and the
    /// decode population counts it there.
    #[test]
    fn decided_candidate_owes_the_d_leg_before_the_bind() {
        let (e, _s, late, ws) = late_fixture();
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        for w in &ws {
            assert_eq!(e.legs_owed(&w.url), 0, "undecided: owed nowhere");
        }
        let w = late.decide_for_test("tcp://p:1", 1, 1) as usize;
        for (i, x) in ws.iter().enumerate() {
            assert_eq!(e.legs_owed(&x.url), usize::from(i == w));
        }
        let pop = e.decode_population();
        assert_eq!(pop["workers"][&ws[w].url], json!([0, 1, 0]));
        // Unbound legs are still not migrated (the tokenizer has no stream).
        assert!(e.migrate_decode(&ws[w].url).is_empty());
        e.event(&cand(w, LegState::Bound), "secret").unwrap();
        assert_eq!(e.migrate_decode(&ws[w].url).len(), 1);
    }

    /// The bound candidate takes the D in-flight count from the arrival
    /// pick; the count leaves with the D-leg's end, and
    /// a bind reported after the D-leg ended counts nowhere.
    #[test]
    fn bind_moves_the_in_flight_decode_count_to_the_winner() {
        let (e, s, _late, ws) = late_fixture();
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        let active = |u: &str| s.lock().view(u).active_decode_reqs;
        assert_eq!((active(&ws[0].url), active(&ws[2].url)), (1, 0));
        e.event(&cand(2, LegState::Bound), "secret").unwrap();
        assert_eq!((active(&ws[0].url), active(&ws[2].url)), (0, 1));
        e.event(&ev(Leg::Decode, LegState::Uploaded), "secret")
            .unwrap();
        e.event(&ev(Leg::Decode, LegState::Completed), "secret")
            .unwrap();
        assert_eq!(active(&ws[2].url), 0);

        let (e, s, _late, ws) = late_fixture();
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        e.event(&ev(Leg::Decode, LegState::Cancelled), "secret")
            .unwrap();
        e.event(&cand(1, LegState::Bound), "secret").unwrap();
        for w in &ws {
            assert_eq!(s.lock().view(&w.url).active_decode_reqs, 0);
            assert_eq!(charged(w), 0);
        }
    }

    /// A symmetric prefill abort answered HTTP 503 with one of the engine's
    /// abort messages (or a finish_reason abort) is terminal for the P-leg:
    /// its reservation retires at once (a retained one freezes the router).
    /// Any other 5xx, or those messages on another status, stays
    /// unknown (the reconciliation is the backstop).
    #[test]
    fn prefill_engine_abort_503_is_terminal_and_other_errors_are_not() {
        for (status, message, terminal) in [
            (503, "aborted by client", true),
            (503, "Compute-before-allocation aborted", true),
            (503, "finish_reason abort: request 7 aborted", true),
            (503, "service unavailable", false),
            (500, "aborted by client", false),
            (502, "finish_reason abort: x", false),
        ] {
            let (e, mut s) = fixture();
            e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
                .unwrap();
            e.event(&ev(Leg::Prefill, LegState::Uploaded), "secret")
                .unwrap();
            let failed = ExecutionEvent {
                http_status: Some(status),
                engine_abort: Some(message.into()),
                ..ev(Leg::Prefill, LegState::Failed)
            };
            e.event(&failed, "secret").unwrap();
            let v = s.view("http://p");
            assert_eq!(
                (v.active_prefill_reqs, v.unknown_prefill_reqs),
                if terminal { (0, 0) } else { (1, 1) },
                "{status} {message}"
            );
            assert_eq!(
                e.legs_owed("http://p"),
                usize::from(!terminal),
                "{status} {message}"
            );
        }
        // The rule itself.
        assert!(is_terminal_engine_abort(
            Some(503),
            Some("aborted by client")
        ));
        assert!(is_terminal_engine_abort(
            Some(503),
            Some("compute-before-allocation aborted")
        ));
        assert!(!is_terminal_engine_abort(Some(503), None));
        assert!(!is_terminal_engine_abort(None, Some("aborted by client")));
        assert!(!is_terminal_engine_abort(
            Some(504),
            Some("aborted by client")
        ));
    }

    // ---- unknown reconciliation of an aborted P-leg ----

    /// A claimed plan "r" (rid "rid-r", P "http://p", D "http://d") whose P
    /// reservation carries its RID, in a reconciling Backend ledger.
    fn migrating_fixture() -> (Executions, BackendState) {
        let mut state = BackendState::new(Arc::new(tokio::sync::Notify::new()));
        state.set_unknown_reconcile(Some(crate::backend::state::UnknownReconcile {
            empty_queue: Duration::from_secs(10),
            quiet: Some(Duration::from_secs(60)),
        }));
        let e = Executions::default();
        let ledger = state.reserve_request("http://p", 4096, Some("rid-r"), 8192);
        e.install_claimed_for_test("r", "secret", ["http://p", "http://d"], ledger);
        (e, state)
    }

    fn rid_ev(leg: Leg, state: LegState) -> ExecutionEvent {
        ExecutionEvent {
            request_id: "rid-r".into(),
            ..ev(leg, state)
        }
    }

    fn p_reply(s: &mut BackendState, base: Instant, at_s: u64, listed: &[&str]) {
        s.apply_sample(
            "http://p",
            crate::backend::state::EngineLoadSample {
                observation: crate::backend::state::LoadObservation {
                    uncached_tokens: Some(0),
                    running: Some(0),
                    waiting: Some(0),
                },
                timestamp: None,
                prefill_work: Some(crate::backend::state::PrefillProgress {
                    epoch: "e".into(),
                    sequence: at_s + 1,
                    remaining: listed.iter().map(|r| (r.to_string(), 100)).collect(),
                    truncated: false,
                }),
            },
            base + Duration::from_secs(at_s),
        );
    }

    /// A migrated D-leg with an aborted P-leg does not stay owed: a D->P
    /// drain migrates the D-leg, the router aborts the queued P-leg, the tokenizer reports the P-leg only
    /// `cancelled` (unknown). The worker stops listing the RID; after 10 s
    /// with an empty queue the reservation is retired, the P-leg is no longer
    /// owed, the execution leaves the registry and a late callback is
    /// idempotent.
    #[test]
    fn migrated_then_aborted_p_leg_is_retired_by_reconciliation() {
        let (e, mut s) = migrating_fixture();
        e.event(&rid_ev(Leg::Prefill, LegState::Uploaded), "secret")
            .unwrap();
        e.event(&rid_ev(Leg::Decode, LegState::Uploaded), "secret")
            .unwrap();
        assert_eq!(e.migrate_decode("http://d").len(), 1);
        let mut migrated = rid_ev(Leg::Decode, LegState::Failed);
        migrated.terminal_response = true;
        let aborts = e.event(&migrated, "secret").unwrap();
        assert!(
            aborts.iter().any(|(w, _)| w == "http://p"),
            "the P-leg is aborted too"
        );
        e.event(&rid_ev(Leg::Prefill, LegState::Cancelled), "secret")
            .unwrap();
        assert_eq!(s.view("http://p").unknown_prefill_reqs, 1);
        assert_eq!(e.legs_owed("http://p"), 1);
        let base = Instant::now();
        p_reply(&mut s, base, 0, &[]);
        p_reply(&mut s, base, 5, &[]);
        assert_eq!(e.legs_owed("http://p"), 1, "5 s is not enough evidence");
        p_reply(&mut s, base, 10, &[]);
        let v = s.view("http://p");
        assert_eq!((v.active_prefill_reqs, v.unknown_retired_reqs), (0, 1));
        assert_eq!(e.legs_owed("http://p"), 0);
        assert_eq!(e.summary()["active"], 0);
        let mut late = rid_ev(Leg::Prefill, LegState::Failed);
        late.terminal_response = true;
        assert_eq!(e.event(&late, "secret").unwrap(), vec![]);
        assert_eq!(e.migration_status("r", "secret"), Ok(true));
    }

    /// A real long-running P-leg reported unknown (transport error) but still
    /// listed by its worker stays reserved and owed.
    #[test]
    fn listed_unknown_p_leg_stays_owed() {
        let (e, mut s) = migrating_fixture();
        e.event(&rid_ev(Leg::Prefill, LegState::Uploaded), "secret")
            .unwrap();
        e.event(&rid_ev(Leg::Prefill, LegState::Failed), "secret")
            .unwrap();
        assert_eq!(s.view("http://p").unknown_prefill_reqs, 1);
        let base = Instant::now();
        for at in (0..=200).step_by(10) {
            p_reply(&mut s, base, at, &["rid-r"]);
        }
        assert_eq!(s.view("http://p").active_prefill_reqs, 1);
        assert_eq!(e.legs_owed("http://p"), 1);
        assert_eq!(e.summary()["active"], 1);
    }

    // ---- unresolved D-leg expiry (execution-ledger leak) ----

    /// The fixture plan, claimed, with the D-leg expiry `after`.
    fn expiring(after: Option<Duration>) -> (Executions, BackendState) {
        let (e, s) = fixture();
        let e = e.with_decode_unresolved_expiry(after);
        e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
            .unwrap();
        e.event(&ev(Leg::Prefill, LegState::Uploaded), "secret")
            .unwrap();
        e.event(&ev(Leg::Decode, LegState::Uploaded), "secret")
            .unwrap();
        (e, s)
    }
    /// No callback for the fixture execution in the last `secs` seconds.
    fn quiet_for(e: &Executions, secs: u64) {
        e.records.lock().active.get_mut("r").unwrap().last_event =
            Instant::now() - Duration::from_secs(secs);
    }
    const EXPIRY: Option<Duration> = Some(Duration::from_secs(300));

    /// Seen in an open-loop run: P completed; a client cancel during D
    /// streaming; the tokenizer reported the D-leg `cancelled`, then `failed`
    /// without terminal evidence, and its confirmation never came. Without
    /// the expiry (the earlier behaviour, `None`) the execution holds a
    /// prepared-execution slot, an `unknown` D request and a leg owed by its
    /// D worker forever. With it, 300 s after the last callback the D-leg
    /// ends and the execution leaves the registry; a late confirmation is
    /// idempotent.
    #[test]
    fn lost_decode_confirmation_expires_after_the_quiet_period() {
        for after in [None, EXPIRY] {
            let (e, mut s) = expiring(after);
            e.event(&ev(Leg::Prefill, LegState::Completed), "secret")
                .unwrap();
            e.event(&ev(Leg::Decode, LegState::Cancelled), "secret")
                .unwrap();
            e.event(&ev(Leg::Decode, LegState::Failed), "secret")
                .unwrap();
            assert_eq!(
                s.view("http://d").active_decode_reqs,
                0,
                "ended at the cancel"
            );
            for quiet in [0, 299] {
                quiet_for(&e, quiet);
                let summary = e.summary();
                assert_eq!(
                    (&summary["active"], &summary["unresolved"]),
                    (&json!(1), &json!(1))
                );
                assert_eq!(
                    e.decode_population()["workers"]["http://d"],
                    json!([0, 0, 1])
                );
                assert_eq!(e.legs_owed("http://d"), 1);
                assert!(e.begin("next", 1).is_err(), "at capacity");
            }
            quiet_for(&e, 301);
            let summary = e.summary();
            if after.is_none() {
                assert_eq!(
                    (&summary["active"], &summary["unresolved"]),
                    (&json!(1), &json!(1))
                );
                assert_eq!(
                    e.decode_population()["workers"]["http://d"],
                    json!([0, 0, 1])
                );
                assert_eq!(e.legs_owed("http://d"), 1);
                assert!(e.begin("next", 1).is_err(), "the leak: never released");
                assert_eq!(summary["decode_unresolved_expired"], 0);
                assert_eq!(summary["decode_unresolved_expire_s"], Value::Null);
                continue;
            }
            assert_eq!(
                (&summary["active"], &summary["unresolved"]),
                (&json!(0), &json!(0))
            );
            assert_eq!(summary["decode_unresolved_expired"], 1);
            assert_eq!(summary["executions_expired"], 1);
            assert_eq!(summary["decode_unresolved_expire_s"], 300.0);
            assert_eq!(e.decode_population()["workers"], json!({}));
            assert_eq!(e.legs_owed("http://d"), 0);
            assert!(e.begin("next", 1).is_ok(), "the slot is free again");
            let mut late = ev(Leg::Decode, LegState::Failed);
            late.terminal_response = true;
            assert_eq!(e.event(&late, "secret").unwrap(), vec![], "tombstoned");
            assert_eq!(e.summary()["decode_unresolved_expired"], 1);
        }
    }

    /// The expiry never ends a running, completed or confirmed D-leg, never
    /// ends a P-leg, and every callback restarts its quiet period.
    #[test]
    fn decode_expiry_leaves_healthy_legs_and_p_legs_alone() {
        // A long healthy decode: no callback for an hour.
        let (e, _s) = expiring(EXPIRY);
        e.event(&ev(Leg::Prefill, LegState::Completed), "secret")
            .unwrap();
        quiet_for(&e, 3600);
        assert_eq!(e.summary()["active"], 1);
        assert_eq!(
            e.decode_population()["workers"]["http://d"],
            json!([1, 0, 0])
        );
        assert_eq!(e.legs_owed("http://d"), 1);
        e.event(&ev(Leg::Decode, LegState::Completed), "secret")
            .unwrap();
        let summary = e.summary();
        assert_eq!(
            (&summary["active"], &summary["decode_unresolved_expired"]),
            (&json!(0), &json!(0))
        );

        // A confirmed D failure retires the execution by itself (no expiry).
        let (e, _s) = expiring(EXPIRY);
        e.event(&ev(Leg::Prefill, LegState::Completed), "secret")
            .unwrap();
        let mut failed = ev(Leg::Decode, LegState::Failed);
        failed.terminal_response = true;
        e.event(&failed, "secret").unwrap();
        let summary = e.summary();
        assert_eq!(
            (&summary["active"], &summary["decode_unresolved_expired"]),
            (&json!(0), &json!(0))
        );

        // An unknown P-leg (cancelled without evidence) with a completed
        // D-leg: P-legs end only on engine evidence (reconciliation).
        let (e, _s) = expiring(EXPIRY);
        e.event(&ev(Leg::Decode, LegState::Completed), "secret")
            .unwrap();
        e.event(&ev(Leg::Prefill, LegState::Cancelled), "secret")
            .unwrap();
        quiet_for(&e, 3600);
        assert_eq!(e.summary()["active"], 1);
        assert_eq!(e.legs_owed("http://p"), 1);
        assert_eq!(e.summary()["decode_unresolved_expired"], 0);

        // D cancelled while P still prefills: only the D-leg ends; the
        // execution stays until the P-leg's own evidence.
        let (e, mut s) = expiring(EXPIRY);
        e.event(&ev(Leg::Decode, LegState::Cancelled), "secret")
            .unwrap();
        quiet_for(&e, 301);
        assert_eq!((e.legs_owed("http://d"), e.legs_owed("http://p")), (0, 1));
        let summary = e.summary();
        assert_eq!(
            (&summary["active"], &summary["decode_unresolved_expired"]),
            (&json!(1), &json!(1))
        );
        assert_eq!(summary["executions_expired"], 0);
        assert_eq!(
            s.view("http://p").active_prefill_reqs,
            1,
            "P load untouched"
        );
        e.event(&ev(Leg::Prefill, LegState::Completed), "secret")
            .unwrap();
        assert_eq!(e.summary()["active"], 0);
        assert_eq!(s.view("http://p").active_prefill_reqs, 0);

        // A callback restarts the quiet period.
        let (e, _s) = expiring(EXPIRY);
        e.event(&ev(Leg::Prefill, LegState::Completed), "secret")
            .unwrap();
        e.event(&ev(Leg::Decode, LegState::Cancelled), "secret")
            .unwrap();
        quiet_for(&e, 200);
        e.event(&ev(Leg::Decode, LegState::Failed), "secret")
            .unwrap();
        assert!(e.records.lock().active["r"].last_event.elapsed() < Duration::from_secs(60));
        assert_eq!(e.summary()["active"], 1);

        // A Backend cancel restarts it too, and an unclaimed plan is left to
        // the unclaimed-plan expiry.
        let (e, _s) = expiring(EXPIRY);
        quiet_for(&e, 200);
        e.cancel("r", "secret").unwrap();
        assert!(e.records.lock().active["r"].last_event.elapsed() < Duration::from_secs(60));
        let (e, _s) = fixture();
        let e = e.with_decode_unresolved_expiry(EXPIRY);
        e.records.lock().active.get_mut("r").unwrap().states[1] = LegState::Cancelled;
        quiet_for(&e, 301);
        assert_eq!(e.summary()["active"], 1);
        assert_eq!(e.summary()["decode_unresolved_expired"], 0);
    }

    /// `decode_unresolved_expire_secs` defaults to 300 s; 0 turns it off;
    /// unknown fields stay rejected.
    #[test]
    fn prepared_config_decode_unresolved_expiry_defaults_to_300_s() {
        let cfg = json!({"callback_url": "http://router/symmetric_pd/events", "max_inflight": 4,
            "contracts": {"c": {"page_size": 64, "is_eagle": false, "model_path": "/m",
                                "cache_namespace": null}}});
        let parse = |v: &Value| serde_json::from_value::<PreparedConfig>(v.clone());
        assert_eq!(
            parse(&cfg).unwrap().decode_unresolved_expiry(),
            Some(Duration::from_secs(300))
        );
        for (secs, want) in [(0, None), (120, Some(Duration::from_secs(120)))] {
            let mut v = cfg.clone();
            v["decode_unresolved_expire_secs"] = json!(secs);
            assert_eq!(parse(&v).unwrap().decode_unresolved_expiry(), want);
        }
        let mut v = cfg.clone();
        v["decode_unresolved_expiry"] = json!(1);
        assert!(parse(&v).is_err());
    }

    /// A D-leg the role drain marked migrating, then cancelled without
    /// terminal evidence, expires; its tombstone still answers /migration.
    #[test]
    fn migrating_d_leg_expires_and_tombstone_keeps_flag() {
        let (e, _s) = expiring(EXPIRY);
        e.event(&ev(Leg::Prefill, LegState::Completed), "secret")
            .unwrap();
        assert_eq!(e.migrate_decode("http://d").len(), 1);
        e.event(&ev(Leg::Decode, LegState::Cancelled), "secret")
            .unwrap();
        quiet_for(&e, 301);
        let summary = e.summary();
        assert_eq!(summary["active"], 0);
        assert_eq!(summary["decode_unresolved_expired"], 1);
        assert_eq!(summary["executions_expired"], 1);
        assert_eq!(e.migration_status("r", "secret"), Ok(true));
        assert_eq!(e.legs_owed("http://d"), 0);
    }

    /// A D upload acknowledged after the expiry: without the expiry the
    /// Backend repeats the abort (late upload after a cancel); after the
    /// expiry the leg is terminal and no abort is repeated (no abort resend).
    #[test]
    fn late_d_upload_after_expiry_no_longer_repeats_abort() {
        for after in [None, EXPIRY] {
            let (e, _s) = fixture();
            let e = e.with_decode_unresolved_expiry(after);
            e.event(&ev(Leg::Prefill, LegState::Claimed), "secret")
                .unwrap();
            e.event(&ev(Leg::Prefill, LegState::Uploaded), "secret")
                .unwrap();
            e.event(&ev(Leg::Decode, LegState::Cancelled), "secret")
                .unwrap();
            quiet_for(&e, 301);
            let _ = e.summary();
            let targets = e
                .event(&ev(Leg::Decode, LegState::Uploaded), "secret")
                .unwrap();
            assert_eq!(
                targets.is_empty(),
                after.is_some(),
                "{after:?}: {targets:?}"
            );
        }
    }

    /// A terminal confirmation that arrives after the quiet period is
    /// processed after the expiry: it is answered idempotently, and the
    /// counter counts the leg as expired (read the WARN `quiet_s` with it).
    #[test]
    fn late_terminal_confirmation_after_quiet_period_counts_as_expired() {
        let (e, _s) = expiring(EXPIRY);
        e.event(&ev(Leg::Prefill, LegState::Completed), "secret")
            .unwrap();
        e.event(&ev(Leg::Decode, LegState::Cancelled), "secret")
            .unwrap();
        quiet_for(&e, 301);
        let mut late = ev(Leg::Decode, LegState::Failed);
        late.terminal_response = true;
        assert_eq!(e.event(&late, "secret").unwrap(), vec![]);
        assert_eq!(e.summary()["decode_unresolved_expired"], 1);
    }

    /// The tokenizer stops reporting (process died, callbacks lost) right
    /// after it claimed the plan and uploaded both legs. Both engines drop
    /// the request; the P worker answers every progress poll and never lists
    /// the RID. Once the execution has been silent for the expiry, its D-leg
    /// ends, its P reservation becomes unknown and is retired on the engine's
    /// evidence, and the execution leaves the registry. Before the expiry
    /// nothing changes.
    #[test]
    fn silent_claimed_execution_is_reclaimed_after_the_expiry() {
        let mut state = BackendState::new(Arc::new(tokio::sync::Notify::new()));
        let e = Executions::default().with_silent_expiry(Some(Duration::from_secs(1800)));
        let ledger = state.reserve_request("http://p", 4096, Some("rid-r"), 8192);
        e.install_claimed_for_test("r", "secret", ["http://p", "http://d"], ledger);
        e.records.lock().active.get_mut("r").unwrap().d_reservation =
            Some(state.reserve_decode("http://d"));
        e.event(&rid_ev(Leg::Prefill, LegState::Uploaded), "secret")
            .unwrap();
        e.event(&rid_ev(Leg::Decode, LegState::Uploaded), "secret")
            .unwrap();
        let held = |e: &Executions, state: &mut BackendState| {
            let (p, d) = (state.view("http://p"), state.view("http://d"));
            (
                e.summary()["active"].as_u64().unwrap(),
                e.legs_owed("http://p"),
                e.legs_owed("http://d"),
                p.active_prefill_reqs,
                d.active_decode_reqs,
            )
        };
        // Quiet for less than the expiry: a long request looks the same.
        e.age_for_test("r", 1700);
        e.decode_population();
        let base = Instant::now();
        for at in (0..=600).step_by(10) {
            p_reply(&mut state, base, at, &[]);
        }
        assert_eq!(held(&e, &mut state), (1, 1, 1, 1, 1));
        // Past it: the D-leg ends at once, the P-leg on engine evidence.
        e.age_for_test("r", 1900);
        e.decode_population();
        assert_eq!(held(&e, &mut state), (1, 1, 0, 1, 0));
        for at in (610..=1200).step_by(10) {
            p_reply(&mut state, base, at, &[]);
        }
        assert_eq!(
            held(&e, &mut state),
            (0, 0, 0, 0, 0),
            "(active executions, legs owed by P, legs owed by D, P reservations, D in-flight)"
        );
        assert_eq!(e.summary()["silent_expired"], 1);
    }

    /// A client cancel before the prefill answered, on a router without
    /// external role control. The tokenizer reports both legs `cancelled`
    /// and its cleanup grace ends without terminal P evidence. The D-leg
    /// expires after 300 s; the P-leg ends once its worker has stopped
    /// listing the RID, and the execution leaves the registry.
    #[test]
    fn cancelled_p_leg_without_terminal_evidence_is_reclaimed_without_role_control() {
        let mut s = BackendState::new(Arc::new(tokio::sync::Notify::new()));
        let e = Executions::default().with_decode_unresolved_expiry(EXPIRY);
        let ledger = s.reserve_request("http://p", 4096, Some("rid-r"), 8192);
        e.install_claimed_for_test("r", "secret", ["http://p", "http://d"], ledger);
        for (leg, state) in [
            (Leg::Prefill, LegState::Uploaded),
            (Leg::Decode, LegState::Uploaded),
            (Leg::Prefill, LegState::Cancelled),
            (Leg::Decode, LegState::Cancelled),
        ] {
            e.event(&rid_ev(leg, state), "secret").unwrap();
        }
        e.age_for_test("r", 3600);
        let base = Instant::now();
        for at in (0..=3600).step_by(10) {
            p_reply(&mut s, base, at, &[]);
        }
        let p = s.view("http://p");
        let leaked = (
            e.summary()["active"].as_u64().unwrap(),
            e.legs_owed("http://p"),
            p.active_prefill_reqs,
            p.unknown_prefill_reqs,
        );
        assert_eq!(
            leaked,
            (0, 0, 0, 0),
            "(active executions, legs owed by P, P reservations, unknown P reservations) one hour after the cancel"
        );
    }
}
