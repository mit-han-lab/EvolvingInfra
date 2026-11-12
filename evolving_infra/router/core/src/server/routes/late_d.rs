// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Late D choose (`--pd-late-decode choose`) on POST /v1/completions and
//! POST /v1/chat/completions: the
//! arrival pick, the leg bodies and the dispatch of one P-leg and K D-legs.
//! The table, the prefill links and the bind-time choice are in
//! [`crate::late_bind`].
//!
//! Stream rules (PD column):
//! the router sends the client nothing before a D-leg answers 2xx (R1);
//! the first 2xx D-leg is the stream (R2); a 409 with `X-Late-D-Reason:
//! released` is no error (R4); another failure marks its candidate dead
//! (R5); when every D-leg ended without 2xx the client gets the last
//! failure (R6); a P-leg failure before any D-leg bound answers the client
//! with the prefill's error and drops every D-leg (R8); a client disconnect
//! drops every D-leg (R9); loser legs drain for 30 s after the bind (R10);
//! an answer with `X-Late-D-Reason` ticks no breaker (R11).
//!
//! Differences from sgl-model-gateway's `execute_late_bind`, both kept from
//! this router's plain PD dispatch: the P-leg is a detached task that
//! outlives the client connection (see the chat route; R9 drops the D-legs
//! only), and the winner's stream does not wait for the prefill's answer.
//! The dispatch learns the P outcome over a channel. A P failure after the
//! bind is logged and counted; with `--pd-abort-decode-on-prefill-failure`
//! it also cuts the stream, as in the plain dispatch.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response, StatusCode};
use bytes::Bytes;
use futures::stream::{FuturesUnordered, StreamExt};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::chat::{build_outgoing_body, BootstrapFields, RecordDurationOnDrop, X_SGL_DECODE_URL};
use super::decision_log::AdmissionLedger;
use crate::discovery::ModelId;
use crate::late_bind::{self, EntrySpec, LateBind, LegBodies, LOG_TARGET};
use crate::policies::registry::{PdPoolResolver, PdResolveError};
use crate::proxy::{prefill_failed, prefill_failure_status, PrefillFailure};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::workers::Worker;

/// Read limit for the body of a failed D-leg that is passed to the client.
const ERROR_BODY_WAIT: Duration = Duration::from_secs(2);

/// Choose mode, before the arrival pick: a batch prompt is refused (400
/// `late_d_batch_unsupported`) and a client `late_d` field is dropped (the
/// router owns it).
pub(crate) fn prepare_request(value: Option<&mut Value>) -> Result<(), ApiError> {
    let Some(value) = value else {
        return Ok(());
    };
    if late_bind::is_batch_prompt(value) {
        return Err(ApiError::LateDBatchUnsupported);
    }
    if let Some(obj) = value.as_object_mut() {
        obj.remove("late_d");
    }
    Ok(())
}

/// A choose-mode request after the arrival pick: its candidates.
pub(crate) struct Choose {
    pub(crate) late: Arc<LateBind>,
    /// The prefill's link endpoint.
    pub(crate) owner: String,
    /// One D-leg each, in URL order.
    pub(crate) candidates: Vec<Arc<Worker>>,
    /// The arrival pick's index: the candidate the prefill admits when no
    /// decision comes in time.
    pub(crate) default_idx: usize,
}

/// The arrival pick of a choose-mode PD request and, when the request takes
/// the choose dispatch, its candidates.
pub(crate) struct Pick {
    /// The `power-of-two-smg` pick (the decode of a plain dispatch).
    pub(crate) arrival: Arc<Worker>,
    /// `None`: the prefill has no late-D link, or no healthy decode runs
    /// late D choose; the request takes the plain two-leg dispatch.
    pub(crate) choose: Option<Choose>,
}

/// Choose mode: the arrival pick by `power-of-two-smg` among the healthy
/// decode workers that run late D choose (all healthy decode workers when
/// the request cannot take the choose dispatch), and the candidates (all of
/// them, or the pick and N - 1 others; `--pd-late-decode-candidates`).
pub(crate) fn pick(
    ctx: &AppContext,
    late: &Arc<LateBind>,
    resolver: &PdPoolResolver,
    model: &ModelId,
    prefill: &Worker,
) -> Result<Pick, PdResolveError> {
    let available = resolver.decode_candidates(model)?;
    let owner = late_bind::link_endpoint(prefill);
    let mut capable: Vec<Arc<Worker>> = if owner.is_some() {
        available
            .iter()
            .filter(|w| late_bind::is_choose_decode(w))
            .cloned()
            .collect()
    } else {
        Vec::new()
    };
    // The registry order is a HashSet's; URL order keeps cand_idx stable.
    capable.sort_by(|a, b| a.url.cmp(&b.url));
    let pool = if capable.is_empty() {
        &available
    } else {
        &capable
    };
    let arrival = match &ctx.decode_po2_smg {
        Some(policy) => policy.select(pool),
        // The CLI requires power-of-two-smg in choose mode.
        None => pool.first().cloned(),
    }
    .ok_or(PdResolveError::NoDecodeWorkersAvailable)?;
    let choose = match owner {
        Some(owner) if !capable.is_empty() => {
            let d0 = capable
                .iter()
                .position(|w| Arc::ptr_eq(w, &arrival))
                .unwrap_or(0);
            let (candidates, default_idx) = late.candidates(&capable, d0);
            Some(Choose {
                late: Arc::clone(late),
                owner,
                candidates,
                default_idx,
            })
        }
        _ => {
            late.count_plain();
            None
        }
    };
    Ok(Pick { arrival, choose })
}

/// The P-leg body (`late_d`, the bootstrap fields, the pull hint, the
/// `kv_session` tag of `--kv-session-retention`) and the D-leg bodies (the
/// bootstrap fields, the packed prompt when enabled; each leg adds its
/// `late_d.cand_idx`) of a choose-mode request. No D-leg body carries
/// the hint or the session tag.
pub(crate) fn build_bodies(
    body: &Bytes,
    value: Option<Value>,
    bootstrap: &BootstrapFields,
    hint: Option<&Value>,
    kv_session: Option<&Value>,
    choose: &Choose,
) -> Result<(Bytes, LegBodies), ApiError> {
    let value = match value {
        Some(v) => v,
        None => serde_json::from_slice(body).map_err(|_| {
            ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
        })?,
    };
    let late = &choose.late;
    let k = choose.candidates.len();
    let mut prefill_value = value.clone();
    if let Some(obj) = prefill_value.as_object_mut() {
        obj.insert(
            "late_d".to_string(),
            late_bind::prefill_field(late.router_epoch(), k, choose.default_idx),
        );
        // `--kv-session-retention`: the P leg alone carries the session tag.
        if let Some(tag) = kv_session {
            obj.insert(super::kv_session::KV_SESSION_FIELD.to_string(), tag.clone());
        }
    }
    let prefill_body = build_outgoing_body(body, Some(prefill_value), None, Some(bootstrap), hint)?;
    let mut decode_value = value;
    if late.config().packed_prompt {
        late_bind::pack_prompt(&mut decode_value);
    }
    let decode_base = build_outgoing_body(body, Some(decode_value), None, Some(bootstrap), None)?;
    let legs = LegBodies::from_object_bytes(&decode_base, late.router_epoch(), k)
        .map_err(|e| ApiError::Internal(anyhow::anyhow!("late_d leg bodies: {e}")))?;
    Ok((prefill_body, legs))
}

/// `--pd-late-decode-packed-prompt` with late D off: the decode body with
/// its token prompt packed; unchanged when it cannot be packed.
pub(crate) fn pack_decode_body(body: Bytes) -> Bytes {
    let Ok(mut value) = serde_json::from_slice::<Value>(&body) else {
        return body;
    };
    if !late_bind::pack_prompt(&mut value) {
        return body;
    }
    serde_json::to_vec(&value).map(Bytes::from).unwrap_or(body)
}

// ---------------------------------------------------------------------------
// Late D choose on symmetric workers (`--policy symmetric_pd`).
// ---------------------------------------------------------------------------

/// A symmetric_pd Split under late D choose: the candidates (the `Choose`) and
/// the marks. The P-leg names every candidate in `kv_export.candidates`;
/// each D-leg carries its own `kv_import`.
pub(crate) struct FlexLate {
    pub(crate) transfer_id: u64,
    pub(crate) kv_export: Value,
    /// Per candidate, in `cand_idx` order.
    pub(crate) imports: Vec<Value>,
}

/// Late D choose on a symmetric_pd Split: the owner is the placement's P (its
/// link from `disaggregation_late_d_control_port`), the candidates are the
/// decode-capable workers other than P that run late D choose (URL order;
/// all of them, or the arrival pick and N - 1 others), and the arrival pick
/// is the placement's D (`--symmetric-pd-d-select`), which P admits when no
/// decision comes in time. `None`: a Local placement (no transfer; not
/// counted), or, counted `plain`, an owner without a link or no candidate;
/// the request then takes the plain symmetric_pd dispatch.
pub(crate) fn flex_pick(
    late: &Arc<LateBind>,
    flex: &crate::scheduler::SymmetricPdRuntime,
    d: &crate::scheduler::Decision,
) -> Result<Option<(Choose, FlexLate)>, ApiError> {
    let Some(transfer_id) = d.placement.transfer_id.filter(|_| !d.is_local()) else {
        return Ok(None);
    };
    let owner = late_bind::link_endpoint(&d.p_worker)
        .filter(|_| d.p_worker.mode() == crate::discovery::WorkerMode::Symmetric);
    let capable: Vec<Arc<Worker>> = flex
        .decode_workers()
        .into_iter()
        .filter(|w| w.url != d.p_worker.url && late_bind::is_choose_decode(w))
        .collect();
    let Some(owner) = owner.filter(|_| !capable.is_empty()) else {
        late.count_plain();
        return Ok(None);
    };
    let d0 = capable
        .iter()
        .position(|w| w.url == d.d_worker.url)
        .unwrap_or(0);
    let (candidates, default_idx) = late.candidates(&capable, d0);
    let (kv_export, imports) = flex
        .late_marks(transfer_id, &d.p_worker, &candidates, d.prompt_len)
        .map_err(ApiError::from)?;
    Ok(Some((
        Choose {
            late: Arc::clone(late),
            owner,
            candidates,
            default_idx,
        },
        FlexLate {
            transfer_id,
            kv_export,
            imports,
        },
    )))
}

/// The P-leg body (`kv_export` with the candidates, the symmetric `late_d`
/// field, the pull hint, the session tag) and the D-leg bodies (each with
/// its own `kv_import` and `late_d.cand_idx`; the packed prompt when
/// enabled) of a symmetric_pd late-D request. No D-leg body carries the hint or
/// the session tag (the P forwards the tag to the admitted decode in the
/// ADMIT frame).
///
/// `d_image_source` (`--late-d-image-source-url`): the D-leg bodies of a
/// request with a prepared image layout drop `image_data` and carry this
/// `prepared_image_source` instead ([`crate::image_source`]); the removed
/// `image_data` list is returned for the router's image store. The P-leg
/// keeps the pixels.
pub(crate) fn build_flex_bodies(
    body: &Bytes,
    value: Option<Value>,
    choose: &Choose,
    marks: &FlexLate,
    hint: Option<&Value>,
    kv_session: Option<&Value>,
    d_image_source: Option<&Value>,
) -> Result<(Bytes, LegBodies, Option<Vec<Value>>), ApiError> {
    use crate::scheduler::{KV_EXPORT_KEY, KV_IMPORT_KEY, KV_PULL_HINT_KEY};
    let mut value = match value {
        Some(v) => v,
        None => serde_json::from_slice(body).map_err(|_| {
            ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
        })?,
    };
    let obj = value.as_object_mut().ok_or_else(|| {
        ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
    })?;
    for k in [
        "bootstrap_host",
        "bootstrap_port",
        "bootstrap_room",
        KV_EXPORT_KEY,
        KV_IMPORT_KEY,
        KV_PULL_HINT_KEY,
        "late_d",
        super::kv_session::KV_SESSION_FIELD,
    ] {
        obj.remove(k);
    }
    let late = &choose.late;
    let k = choose.candidates.len();
    let mut prefill_value = value.clone();
    if let Some(obj) = prefill_value.as_object_mut() {
        obj.insert(KV_EXPORT_KEY.to_string(), marks.kv_export.clone());
        obj.insert(
            "late_d".to_string(),
            late_bind::sym_prefill_field(late.router_epoch(), k, choose.default_idx),
        );
        if let Some(h) = hint {
            obj.insert(KV_PULL_HINT_KEY.to_string(), h.clone());
        }
        if let Some(tag) = kv_session {
            obj.insert(super::kv_session::KV_SESSION_FIELD.to_string(), tag.clone());
        }
    }
    let to_bytes = |v: &Value| -> Result<Bytes, ApiError> {
        serde_json::to_vec(v)
            .map(Bytes::from)
            .map_err(|e| ApiError::Internal(anyhow::Error::new(e).context("serialize leg body")))
    };
    let prefill_body = to_bytes(&prefill_value)?;
    // `--late-d-image-source-url`: the D-legs leave the pixels behind.
    let stripped = match (d_image_source, value.as_object_mut()) {
        (Some(source), Some(obj)) => crate::image_source::strip_images(obj, source),
        _ => None,
    };
    if late.config().packed_prompt {
        late_bind::pack_prompt(&mut value);
    }
    let legs = LegBodies::from_object_bytes(&to_bytes(&value)?, late.router_epoch(), k)
        .and_then(|l| l.with_imports(&marks.imports))
        .map_err(|e| ApiError::Internal(anyhow::anyhow!("late_d leg bodies: {e}")))?;
    Ok((prefill_body, legs, stripped))
}

/// Session KV retention of one symmetric_pd late-D request: the registry and the
/// P-leg's tag. Candidates are recorded in the session history before they
/// can receive the tag; the bound decode becomes the decode home.
pub(crate) struct FlexLateSession {
    pub(crate) sessions: Arc<crate::backend::kv_sessions::KvSessions>,
    pub(crate) tag: crate::backend::kv_sessions::KvSession,
}

impl late_bind::LateSessionHook for FlexLateSession {
    fn candidate(&self, worker_url: &str) {
        self.sessions.note_candidate(&self.tag, worker_url);
    }

    fn bound(&self, worker_url: &str) {
        self.sessions.bind_decode(&self.tag, worker_url);
    }
}

/// Everything the choose-mode dispatch needs from the completions route.
pub(crate) struct ChooseDispatch {
    /// Session KV retention of a symmetric_pd late-D request; `None` otherwise.
    pub(crate) session: Option<Arc<dyn late_bind::LateSessionHook>>,
    pub(crate) choose: Choose,
    pub(crate) prefill: Arc<Worker>,
    /// The P-leg's load guards; the P task holds them until the P call ends.
    pub(crate) prefill_holds: Box<dyn Send + 'static>,
    /// The engines' route of every leg: the route the request arrived on.
    pub(crate) path: &'static str,
    pub(crate) headers: HeaderMap,
    pub(crate) prefill_body: Bytes,
    pub(crate) legs: LegBodies,
    pub(crate) streaming: bool,
    pub(crate) rid: Option<String>,
    pub(crate) room: u64,
    /// The d1 charge (prompt + capped output tokens).
    pub(crate) charge_tokens: u64,
    /// The co-schedule ledger entry; retired when the P call ends.
    pub(crate) ledger: Option<AdmissionLedger>,
    pub(crate) cost: Option<crate::cost_balance::Reservation>,
    /// The stale-request token of the P-leg's active-load entry.
    pub(crate) stale: CancellationToken,
    pub(crate) model: String,
    /// Streaming only: held by the stream, as in the plain dispatch.
    pub(crate) duration_guard: Option<RecordDurationOnDrop>,
    pub(crate) ttft_hook: Option<Box<dyn FnOnce() + Send + 'static>>,
    /// `--late-d-image-source-url`: the request's images in the router's
    /// store (the D-legs went without pixels). Held until the request
    /// ends: by the stream when streaming, else until the answer is read.
    pub(crate) image_hold: Option<crate::image_source::Hold>,
}

/// The breaker slot a D-leg claimed with `allow()`. A leg dropped without an
/// outcome (the request went stale, the prefill failed before the bind, a
/// loser still open after its drain, the client left) would keep a half-open
/// breaker's probe slot for good, and its worker would never be a candidate
/// again; dropping the claim gives the slot back as a failed probe.
struct ProbeClaim {
    breaker: Arc<crate::health::circuit_breaker::CircuitBreaker>,
    armed: bool,
}

impl ProbeClaim {
    fn new(breaker: &Arc<crate::health::circuit_breaker::CircuitBreaker>) -> Self {
        Self {
            breaker: Arc::clone(breaker),
            armed: true,
        }
    }
    /// The leg's outcome is recorded elsewhere.
    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for ProbeClaim {
    fn drop(&mut self) {
        if self.armed {
            late_bind::settle_unused_probe(&self.breaker);
        }
    }
}

/// The end of a P-leg.
#[derive(Debug, Clone, Default)]
struct PrefillOutcome {
    /// The prefill's status; `None` when it never answered.
    status: Option<StatusCode>,
    /// `X-Late-D-Reason` of the answer.
    reason: Option<String>,
    /// A transport error, or a body that failed after the status.
    error: Option<String>,
    /// A 5xx answer naming `symmetric_reader_gone`: the engine finished the
    /// P-leg because its reader left. Cancelled, not failed: no R8, no
    /// breaker tick; the D-legs own the outcome.
    reader_gone: bool,
}

impl PrefillOutcome {
    /// `(client status, detail)` of a failed P-leg (statuses mapped as the
    /// plain dispatch maps them); `None` for a 2xx answer read to the end.
    fn failure(&self) -> Option<(StatusCode, String)> {
        if self.reader_gone {
            return None;
        }
        let reason = self
            .reason
            .as_deref()
            .map(|r| format!(" (X-Late-D-Reason: {r})"))
            .unwrap_or_default();
        match (self.status, &self.error) {
            (Some(s), None) if s.is_success() => None,
            (Some(s), None) => Some((
                prefill_failure_status(s),
                format!("prefill answered {s}{reason}"),
            )),
            (Some(s), Some(e)) => Some((
                StatusCode::BAD_GATEWAY,
                format!("prefill answered {s}, then its body failed: {e}"),
            )),
            (None, e) => Some((
                StatusCode::BAD_GATEWAY,
                e.clone()
                    .unwrap_or_else(|| "prefill leg failed".to_string()),
            )),
        }
    }
}

/// A D-leg's error answer as the client's answer: its status, its body (up
/// to [`ERROR_BODY_WAIT`]), its content type and `X-Late-D-Reason`.
async fn passthrough_error(res: reqwest::Response, reason: Option<&str>) -> Response<Body> {
    let status = res.status();
    let content_type = res.headers().get(reqwest::header::CONTENT_TYPE).cloned();
    let body = match tokio::time::timeout(ERROR_BODY_WAIT, res.bytes()).await {
        Ok(Ok(body)) => body,
        _ => Bytes::new(),
    };
    let mut out = Response::new(Body::from(body));
    *out.status_mut() = status;
    out.headers_mut().insert(
        HeaderName::from_static("content-type"),
        content_type.unwrap_or_else(|| HeaderValue::from_static("application/json")),
    );
    if let Some(v) = reason.and_then(|r| HeaderValue::from_str(r).ok()) {
        out.headers_mut()
            .insert(HeaderName::from_static(late_bind::LATE_D_REASON_HEADER), v);
    }
    out
}

/// The choose-mode dispatch: the P-leg and one D-leg per candidate at once;
/// the first 2xx D-leg is the client's answer. See the module docs.
pub(crate) async fn dispatch(
    ctx: &Arc<AppContext>,
    d: ChooseDispatch,
) -> Result<Response<Body>, ApiError> {
    let ChooseDispatch {
        session,
        choose,
        prefill,
        prefill_holds,
        path,
        headers,
        prefill_body,
        legs: bodies,
        streaming,
        rid,
        room,
        charge_tokens,
        ledger,
        mut cost,
        stale,
        model,
        duration_guard,
        ttft_hook,
        image_hold,
    } = d;
    let Choose {
        late,
        owner,
        candidates,
        default_idx,
    } = choose;
    let started = Instant::now();
    let stats = late.stats();
    let rid_log = rid.clone().unwrap_or_default();

    // Breakers first: no leg leaves for a refused prefill; a refused
    // candidate gets no leg and is dead for the choice.
    // No leg leaves on either refusal, so the P work the admission counted
    // is retired here: nothing else would end it.
    if !prefill.breaker.allow() {
        if let Some(ledger) = &ledger {
            ledger.retire();
        }
        return Err(ApiError::BreakerOpen {
            worker: prefill.url.clone(),
        });
    }
    let sendable: Vec<bool> = candidates.iter().map(|w| w.breaker.allow()).collect();
    if !sendable.iter().any(|&ok| ok) {
        late_bind::settle_unused_probe(&prefill.breaker);
        if let Some(ledger) = &ledger {
            ledger.retire();
        }
        return Err(ApiError::NoDecodeWorkersAvailable { model });
    }

    // The entry is in the table before any leg leaves (the prefill may ask
    // for the winner as soon as it holds the request).
    let entry_guard = late.register_with(
        EntrySpec {
            owner,
            prefill_url: prefill.url.clone(),
            key: room,
            rid: rid.clone(),
            candidates: candidates.clone(),
            default_idx,
            charge_tokens,
        },
        session,
    );
    let entry = Arc::clone(entry_guard.entry());

    // P-leg: detached, as in the plain PD dispatch.
    let headers = Arc::new(headers);
    let prefill_failure = PrefillFailure::new();
    let (p_tx, mut p_rx) = tokio::sync::oneshot::channel::<PrefillOutcome>();
    {
        let proxy = Arc::clone(&ctx.proxy);
        let late = Arc::clone(&late);
        let headers = Arc::clone(&headers);
        let prefill = Arc::clone(&prefill);
        let failure = prefill_failure.clone();
        let rid = rid_log.clone();
        tokio::spawn(async move {
            let _hold = prefill_holds;
            let outcome = match cost.as_mut().map(|c| c.mark_dispatched()) {
                Some(Err(error)) => {
                    tracing::error!(target: "cost_balance", %error, "cost trace failed before P dispatch");
                    PrefillOutcome {
                        status: Some(StatusCode::INTERNAL_SERVER_ERROR),
                        reason: None,
                        error: Some("prefill leg not dispatched: cost trace failed".to_string()),
                        reader_gone: false,
                    }
                }
                _ => {
                    let result = proxy
                        .send_late_leg(&prefill.url, path, &headers, prefill_body, false)
                        .await;
                    let outcome = match result {
                        Ok(res) => {
                            let status = res.status();
                            let reason = late_bind::late_reason(res.headers());
                            let (error, reader_gone) = match res.bytes().await {
                                Ok(body) => {
                                    (None, crate::proxy::body_is_reader_gone(status, &body))
                                }
                                Err(e) => (Some(e.to_string()), false),
                            };
                            PrefillOutcome {
                                status: Some(status),
                                reason,
                                error,
                                reader_gone,
                            }
                        }
                        Err(e) => PrefillOutcome {
                            status: None,
                            reason: None,
                            error: Some(e.to_string()),
                            reader_gone: false,
                        },
                    };
                    // R11: an answer with a late-D reason ticks no breaker;
                    // neither does a reader-gone end.
                    match (&outcome.status, &outcome.error, &outcome.reason) {
                        _ if outcome.reader_gone => late_bind::settle_late_answer(&prefill.breaker),
                        (Some(_), None, Some(_)) => late_bind::settle_late_answer(&prefill.breaker),
                        (Some(s), None, None) if !s.is_server_error() => {
                            prefill.breaker.record_success()
                        }
                        _ => prefill.breaker.record_failure(),
                    }
                    if let Some(cost) = cost.as_mut() {
                        let accounted = match (&outcome.status, &outcome.error) {
                            (Some(s), None) => cost.finish(if s.is_success() {
                                "prefill_response_success"
                            } else if outcome.reader_gone {
                                crate::proxy::READER_GONE_FINISH_REASON
                            } else {
                                "prefill_response_error"
                            }),
                            (_, e) => cost.mark_unknown(&format!(
                                "prefill_transport_error: {}",
                                e.as_deref().unwrap_or("-")
                            )),
                        };
                        if let Err(error) = accounted {
                            tracing::error!(target: "cost_balance", %error, "cost lifecycle observation failed");
                        }
                    }
                    outcome
                }
            };
            match outcome.failure() {
                Some((status, detail)) => {
                    late_bind::inc(&late.stats().prefill_leg_failed);
                    tracing::warn!(
                        target: LOG_TARGET,
                        rid = %rid,
                        bootstrap_room = room,
                        prefill_url = %prefill.url,
                        status = status.as_u16(),
                        reason = ?outcome.reason,
                        error = %detail,
                        "late_d: prefill leg failed",
                    );
                    failure.fail(status, detail);
                }
                None if outcome.reader_gone => {
                    late_bind::inc(&late.stats().prefill_reader_gone);
                    tracing::warn!(
                        target: LOG_TARGET,
                        rid = %rid,
                        bootstrap_room = room,
                        prefill_url = %prefill.url,
                        "late_d: prefill leg ended: its reader left (symmetric_reader_gone); the \
                         decode legs own the request",
                    );
                }
                None => tracing::debug!(prefill_url = %prefill.url, "prefill side completed"),
            }
            // The prefill call resolving is the authoritative "prefill done".
            // A symmetric_pd P-leg without an answer keeps its work as unknown,
            // as in the plain symmetric_pd Split.
            match (ledger, outcome.status) {
                (Some(AdmissionLedger::Flex(handle)), None) => handle.mark_unknown(),
                (Some(l), _) => l.retire(),
                (None, _) => {}
            }
            let _ = p_tx.send(outcome);
        });
    }

    // D-legs: one per candidate, at once.
    let mut legs = FuturesUnordered::new();
    for (idx, worker) in candidates.iter().enumerate() {
        if !sendable[idx] {
            late.mark_dead(&entry, idx);
            late_bind::inc(&stats.legs_not_sent);
            tracing::warn!(
                target: LOG_TARGET,
                rid = %rid_log,
                bootstrap_room = room,
                decode_url = %worker.url,
                cand_idx = idx,
                "late_d: decode candidate not sent (circuit breaker open)",
            );
            continue;
        }
        let proxy = Arc::clone(&ctx.proxy);
        let headers = Arc::clone(&headers);
        let url = worker.url.clone();
        let body = bodies.leg(idx);
        let probe = ProbeClaim::new(&worker.breaker);
        legs.push(async move {
            let result = proxy
                .send_late_leg(&url, path, &headers, body, streaming)
                .await;
            (idx, result, probe)
        });
    }
    let mut open = legs.len();

    let mut p_answer: Option<PrefillOutcome> = None;
    let mut last_failure: Option<Result<Response<Body>, ApiError>> = None;
    let mut released = 0usize;
    let mut all_released = false;
    let released_wait = tokio::time::sleep(late_bind::LOSER_DRAIN);
    tokio::pin!(released_wait);
    let no_winner = |detail: String| {
        late_bind::inc(&stats.no_winner);
        tracing::warn!(
            target: LOG_TARGET,
            rid = %rid_log,
            bootstrap_room = room,
            prefill_url = %prefill.url,
            k = candidates.len(),
            error = %detail,
            "late_d: no decode candidate bound the request",
        );
    };
    let (winner_idx, winner) = loop {
        tokio::select! {
            biased;
            _ = stale.cancelled() => {
                return Err(ApiError::StaleRequestExpired { model });
            }
            outcome = &mut p_rx, if p_answer.is_none() => {
                let outcome = outcome.unwrap_or_else(|_| PrefillOutcome {
                    error: Some("the prefill task ended without an outcome".to_string()),
                    ..PrefillOutcome::default()
                });
                if let Some((status, detail)) = outcome.failure() {
                    // R8: the client gets the prefill's error; returning
                    // drops every D-leg (their connections close).
                    late_bind::inc(&stats.prefill_failed);
                    tracing::warn!(
                        target: LOG_TARGET,
                        rid = %rid_log,
                        bootstrap_room = room,
                        prefill_url = %prefill.url,
                        status = status.as_u16(),
                        open_legs = open,
                        "late_d: decode legs dropped: the prefill leg failed before any bind",
                    );
                    return Err(ApiError::PrefillFailed { status, detail });
                }
                if open == 0 {
                    // R6: every D-leg was released and P answered.
                    let detail = format!("all {released} decode candidates released");
                    no_winner(detail.clone());
                    return Err(ApiError::LateDNoWinner { detail });
                }
                p_answer = Some(outcome);
            }
            Some((idx, result, probe)) = legs.next(), if open > 0 => {
                open -= 1;
                let worker = &candidates[idx];
                match result {
                    Ok(res) if res.status().is_success() => {
                        // The winner's outcome is recorded when its answer ends.
                        probe.disarm();
                        break (idx, res);
                    }
                    Ok(res) => {
                        let status = res.status();
                        let reason = late_bind::late_reason(res.headers());
                        if status == StatusCode::CONFLICT
                            && reason.as_deref() == Some(late_bind::RELEASED_REASON)
                        {
                            // R4: released, not an error.
                            released += 1;
                            late_bind::inc(&stats.legs_released);
                            late_bind::settle_late_answer(&worker.breaker);
                            tokio::spawn(async move {
                                let _ = tokio::time::timeout(late_bind::LOSER_DRAIN, res.bytes()).await;
                            });
                        } else {
                            // R5 (R11: no breaker tick with a late-D reason).
                            late.mark_dead(&entry, idx);
                            late_bind::inc(&stats.legs_dead);
                            match reason {
                                Some(_) => late_bind::settle_late_answer(&worker.breaker),
                                None if status.is_server_error() => worker.breaker.record_failure(),
                                None => worker.breaker.record_success(),
                            }
                            tracing::warn!(
                                target: LOG_TARGET,
                                rid = %rid_log,
                                bootstrap_room = room,
                                decode_url = %worker.url,
                                cand_idx = idx,
                                status = status.as_u16(),
                                reason = ?reason,
                                "late_d: decode candidate failed",
                            );
                            last_failure = Some(Ok(passthrough_error(res, reason.as_deref()).await));
                        }
                    }
                    Err(e) => {
                        late.mark_dead(&entry, idx);
                        late_bind::inc(&stats.legs_dead);
                        worker.breaker.record_failure();
                        tracing::warn!(
                            target: LOG_TARGET,
                            rid = %rid_log,
                            bootstrap_room = room,
                            decode_url = %worker.url,
                            cand_idx = idx,
                            error = %e,
                            "late_d: decode candidate failed",
                        );
                        last_failure = Some(Err(e));
                    }
                }
                if open == 0 {
                    // R6: every D-leg ended without 2xx.
                    if let Some(failure) = last_failure.take() {
                        no_winner(format!(
                            "{released} released, {} failed",
                            entry.dead_mask().count_ones()
                        ));
                        return failure;
                    }
                    if p_answer.is_some() {
                        let detail = format!("all {released} decode candidates released");
                        no_winner(detail.clone());
                        return Err(ApiError::LateDNoWinner { detail });
                    }
                    // All released while P has not answered: its answer
                    // tells why, for up to 30 s.
                    all_released = true;
                    released_wait
                        .as_mut()
                        .reset(tokio::time::Instant::now() + late_bind::LOSER_DRAIN);
                }
            }
            _ = &mut released_wait, if all_released => {
                let detail = format!(
                    "all {released} decode candidates released; the prefill did not answer within {} s",
                    late_bind::LOSER_DRAIN.as_secs()
                );
                no_winner(detail.clone());
                return Err(ApiError::LateDNoWinner { detail });
            }
        }
    };
    late_bind::inc(&stats.legs_winner);

    // R10: legs still open drain for up to 30 s.
    if !legs.is_empty() {
        let late = Arc::clone(&late);
        let candidates = candidates.clone();
        tokio::spawn(async move {
            let deadline = tokio::time::sleep(late_bind::LOSER_DRAIN);
            tokio::pin!(deadline);
            loop {
                tokio::select! {
                    _ = &mut deadline => {
                        late_bind::inc(&late.stats().loser_drain_timeouts);
                        tracing::warn!(
                            target: LOG_TARGET,
                            open = legs.len(),
                            "late_d: loser legs still open 30 s after the bind; dropped",
                        );
                        break;
                    }
                    next = legs.next() => {
                        let Some((idx, result, _probe)) = next else { break };
                        late_bind::account_loser(
                            &late,
                            &candidates[idx],
                            result.map_err(|e| e.to_string()),
                        )
                        .await;
                    }
                }
            }
        });
    }

    // Bind: the request's charge moves to the winner and rides with its
    // stream; the entry leaves the table when this returns.
    let decode = Arc::clone(&candidates[winner_idx]);
    let charge = late.bind_charge(&entry, winner_idx);
    let status = winner.status();
    tracing::info!(
        target: LOG_TARGET,
        rid = %rid_log,
        bootstrap_room = room,
        prefill_worker = %prefill.url,
        decode_worker = %decode.url,
        cand_idx = winner_idx,
        default_idx,
        k = candidates.len(),
        decided = ?entry.decided(),
        admitted = ?entry.admitted(),
        released,
        dead = entry.dead_mask().count_ones(),
        prefill_answered = p_answer.is_some(),
        charge_tokens,
        bind_ms = started.elapsed().as_millis() as u64,
        status = status.as_u16(),
        "late_d bound",
    );
    let (smg_load, smg_arm) = ctx.decode_smg_load(&decode.url, streaming);
    if let Some(arm) = smg_arm {
        arm.arm();
    }
    let decode_guard = (decode.load_guard(), charge, smg_load);
    let abort_flag = ctx.config.proxy.abort_decode_on_prefill_failure;
    let decode_url_header = HeaderValue::from_str(&decode.url).ok();
    let result = if streaming {
        let breaker = Arc::clone(&decode.breaker);
        let on_complete: Box<dyn FnOnce(bool) + Send + 'static> = Box::new(move |ok| {
            if ok {
                breaker.record_success();
            } else {
                breaker.record_failure();
            }
        });
        let stream_guards: Box<dyn Send + 'static> =
            Box::new((decode_guard, duration_guard, image_hold));
        let body = crate::proxy::sse::bytes_stream_to_body_abortable(
            winner.bytes_stream(),
            Some(stream_guards),
            Some(on_complete),
            ttft_hook,
            abort_flag.then(|| prefill_failure.token()),
        );
        let mut out = Response::new(body);
        *out.status_mut() = status;
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("text/event-stream"),
        );
        Ok(out)
    } else {
        let _hold = (decode_guard, image_hold);
        let read = tokio::select! {
            biased;
            error = prefill_failed(abort_flag.then_some(&prefill_failure)) => return Err(error),
            _ = stale.cancelled() => return Err(ApiError::StaleRequestExpired { model }),
            read = winner.bytes() => read,
        };
        match read {
            Ok(bytes) => {
                decode.breaker.record_success();
                let mut out = Response::new(Body::from(bytes));
                *out.status_mut() = status;
                out.headers_mut().insert(
                    HeaderName::from_static("content-type"),
                    HeaderValue::from_static("application/json"),
                );
                Ok(out)
            }
            Err(e) => {
                decode.breaker.record_failure();
                tracing::warn!(
                    target: LOG_TARGET,
                    rid = %rid_log,
                    decode_url = %decode.url,
                    %status,
                    error = %e,
                    "late_d: the bound decode dropped its body",
                );
                Err(ApiError::UpstreamStatus { status })
            }
        }
    };
    drop(entry_guard);
    result.map(|mut out| {
        if let Some(v) = decode_url_header {
            out.headers_mut().insert(X_SGL_DECODE_URL, v);
        }
        out
    })
}
