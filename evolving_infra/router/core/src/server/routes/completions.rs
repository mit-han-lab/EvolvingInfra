// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! POST /v1/completions and SGLang-native POST /generate — proxies with the
//! same worker selection / PD dispatch / kv-pull-hint machinery as the chat
//! route. Each request goes to the same path on the chosen workers.
//!
//! Exists primarily for token-replay clients: strict-replay traffic sends
//! `prompt` (native: `input_ids`) as a flat token-id array (plus engine
//! extensions such as `forced_output_token_ids` and `cache_salt`), which
//! never touches a chat encoder. Such an array feeds cache-aware routing and
//! the kv-pull-hint query directly — no router-side tokenizer involved.
//! Text prompts are forwarded verbatim and route without token context
//! (load-based policies work; cache-aware falls back to its no-token
//! behavior).
//!
//! The body is forwarded byte-faithfully except for the injected fields
//! (`bootstrap_*` in PD mode, `kv_pull_hint` on the executing worker,
//! `kv_export` / `kv_import` under `--policy symmetric_pd`), so engine extensions
//! survive untouched.
//!
//! symmetric_pd: the placer sees the token-id prompt as
//! `prompt_len` / indexer key, chooses a P-leg and D-leg; a Local placement
//! is one plain request, a Split fans out exactly like PD below with the
//! marks in place of the bootstrap fields.
//!
//! Late D choose (`--pd-late-decode choose`): a PD request goes to its
//! prefill and to every decode candidate at once; the prefill asks the
//! router for the winner when it is about to prefill ([`super::late_d`]).

use super::chat::{
    build_outgoing_body, estimate_prefill_tokens, generate_room_id, parse_probe, BootstrapFields,
    RecordDurationOnDrop, X_SGL_DECODE_URL,
};
use super::decision_log::{
    log_routing_decision, settle_prefill_leg, AdmissionLedger, LateDispatchLog, RoutingDecision,
};
use crate::discovery::{ModelId, WorkerMode};
use crate::policies::registry::PdPoolResolver;
use crate::policies::SelectionContext;
use crate::proxy::{prefill_failed, PdLegs, PrefillFailure};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::{RequestOutcome, StaleRequestOutcome, WorkerModeLabel};
use crate::workers::{LoadGuard, Worker};
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, Response, StatusCode};
use bytes::Bytes;
use std::sync::Arc;

/// The route a request arrived on; the upstream request uses the same path.
#[derive(Clone, Copy, Debug)]
enum Ingress {
    Completions,
    /// Native bodies may omit `model`; the router's single model is implied.
    Generate,
    /// A chat request under late D choose (`routes/chat.rs` sends it here):
    /// its token ids come from the router's chat encoder.
    Chat,
}

impl Ingress {
    fn path(self) -> &'static str {
        match self {
            Ingress::Completions => "/v1/completions",
            Ingress::Generate => "/generate",
            Ingress::Chat => "/v1/chat/completions",
        }
    }

    /// The token-id prompt: array-form `prompt`, or native `input_ids`.
    fn token_ids(self, value: &serde_json::Value) -> Option<Vec<u32>> {
        let field = match self {
            Ingress::Completions => "prompt",
            Ingress::Generate => "input_ids",
            Ingress::Chat => return None,
        };
        flat_token_ids(value.get(field)?)
    }

    /// Whether late D choose may dispatch this request. Its legs go to the
    /// engines' `/v1/completions` or `/v1/chat/completions`, which answer a
    /// released candidate with 409; a native `/generate` request takes the
    /// plain two-leg dispatch, and the engines serve a request without
    /// `late_d` as late D alloc.
    fn late_d_choose(self) -> bool {
        matches!(self, Ingress::Completions | Ingress::Chat)
    }
}

/// A single flat array of token ids. Text prompts, string arrays and nested
/// (batched) arrays return `None` — the engine handles those shapes; the
/// router just loses token-aware routing for them.
fn flat_token_ids(value: &serde_json::Value) -> Option<Vec<u32>> {
    let arr = value.as_array()?;
    if arr.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        out.push(u32::try_from(v.as_u64()?).ok()?);
    }
    Some(out)
}

/// POST /v1/completions — select a worker (prefill pool in PD mode), inject
/// bootstrap / pull-hint fields, then proxy buffered or SSE.
pub async fn completions(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    dispatch(ctx, headers, body, Ingress::Completions).await
}

/// POST /generate — the SGLang-native request, dispatched like completions.
pub async fn generate(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    dispatch(ctx, headers, body, Ingress::Generate).await
}

/// A chat request under `--pd-late-decode choose`, dispatched like
/// completions with the legs on the engines' chat route.
pub(crate) async fn chat(
    ctx: Arc<AppContext>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    dispatch(ctx, headers, body, Ingress::Chat).await
}

async fn dispatch(
    ctx: Arc<AppContext>,
    headers: HeaderMap,
    mut body: Bytes,
    ingress: Ingress,
) -> Result<Response<Body>, ApiError> {
    let start = std::time::Instant::now();
    let probe = parse_probe(&body)?;
    let streaming = probe.stream.unwrap_or(false);
    let model_str = match (probe.model, ingress) {
        (Some(model), _) => model,
        (None, Ingress::Generate) => ctx.config.model.id.clone(),
        (None, Ingress::Completions | Ingress::Chat) => {
            return Err(ApiError::BadRequest("missing `model` field".into()))
        }
    };
    let model_id = ModelId(model_str.clone());

    let resolver = PdPoolResolver::new(Arc::clone(&ctx.registry));
    // symmetric_pd builds its own candidate pool (every PD worker, both legs);
    // the prefill-pool resolution is the legacy path's.
    let workers = if ctx.symmetric_pd.is_some() {
        Vec::new()
    } else {
        resolver
            .prefill_candidates(&model_id)
            .map_err(|e| ApiError::from_pd_resolve(e, &model_str))?
    };

    let policy = ctx
        .policies
        .get(&model_id)
        .ok_or_else(|| ApiError::ModelNotFound(model_str.clone()))?;

    // Parse the body when anything downstream wants to look inside it:
    // token-aware routing, the kv-pull-hint query (token ids + cache_salt),
    // symmetric P/D placement (hits + backlog need the ids), or PD bootstrap
    // injection. Load-only plain-mode deployments skip the parse entirely,
    // mirroring the chat route.
    let want_value = headers.contains_key(crate::agent_identity::CONTEXT_HEADER)
        || policy.needs_request_tokens()
        || ctx.decode_tokens.is_some()
        || ctx.kv_hint.is_some()
        || ctx.symmetric_pd.is_some()
        || ctx.late_bind.is_some()
        || ctx.kv_sessions.is_some()
        || ctx.config.proxy.image_layout.is_some();
    let mut request_value: Option<serde_json::Value> = if want_value {
        Some(serde_json::from_slice(&body).map_err(|_| {
            ApiError::BadRequest("invalid request: body must be a JSON object".into())
        })?)
    } else {
        None
    };
    if crate::agent_identity::apply(&headers, request_value.as_mut())? {
        // Plain-mode forwarding can keep the original bytes, so synchronize
        // them when a header-only client needed its RID inserted.
        body = Bytes::from(
            serde_json::to_vec(&request_value).map_err(|e| ApiError::Internal(e.into()))?,
        );
    }
    ensure_progress_rid(&ctx, request_value.as_mut())?;
    // External role control: the controller's workload sums count this
    // request from its replies (see `backend::plain_usage`). A native
    // `/generate` reply has no OpenAI usage and is not counted.
    let usage_asked = match (ctx.external_workload.as_ref(), request_value.as_mut()) {
        (Some(_), Some(value))
            if ctx.symmetric_pd.is_some() && !matches!(ingress, Ingress::Generate) =>
        {
            let asked = crate::backend::plain_usage::ask(value, streaming);
            body =
                Bytes::from(serde_json::to_vec(value).map_err(|e| ApiError::Internal(e.into()))?);
            Some(asked)
        }
        _ => None,
    };
    if ctx.late_bind.is_some() {
        super::late_d::prepare_request(request_value.as_mut())?;
    }
    // `--kv-session-retention`: read the `x-kv-session-context` identity and
    // refuse a client `kv_session`, before admission. Never touches the rid.
    let kv_session_identity = if ctx.kv_sessions.is_some() {
        super::kv_session::ingress(&headers, request_value.as_ref())?
    } else {
        None
    };

    // Token ids come straight from an array-form `prompt`; unlike chat there
    // is no router-side tokenization and the ids are engine-equivalent by
    // definition (the engine reads the same array). They are NOT re-injected
    // as `input_ids` — the forwarded `prompt` already carries them.
    let mut prompt_ids: Option<Vec<u32>> =
        request_value.as_ref().and_then(|v| ingress.token_ids(v));
    if let (Ingress::Chat, Some(value)) = (ingress, request_value.as_mut()) {
        // Late D choose binds one request: `n` choices are one engine batch.
        if value.get("n").and_then(serde_json::Value::as_u64) > Some(1) {
            return Err(ApiError::LateDBatchUnsupported);
        }
        let tokens = super::chat::chat_request_tokens(&ctx, &model_id, value, &workers).await;
        // Engine-equivalent ids ride on every leg as `input_ids`, so the
        // prefill and each decode candidate skip tokenizing the same prompt.
        if let (Some(t), Some(obj)) = (tokens.as_ref(), value.as_object_mut()) {
            if t.engine_equivalent
                && super::chat::input_ids_safe_to_forward(&serde_json::Value::Object(obj.clone()))
            {
                obj.insert("input_ids".to_string(), serde_json::json!(t.ids));
                body = Bytes::from(
                    serde_json::to_vec(&*obj).map_err(|e| ApiError::Internal(e.into()))?,
                );
            }
        }
        prompt_ids = tokens.map(|t| t.ids);
    }
    // `--image-layout-config`: a token-id prompt with uploaded screenshots
    // routes on the engine's radix view (placeholders expanded to the
    // images' layout pad values), and the forwarded body carries the
    // `prepared_image_layout` that makes the engine use the same pad values.
    if let (Some(layout), Some(ids)) = (ctx.config.proxy.image_layout, prompt_ids.as_ref()) {
        if request_value
            .as_ref()
            .is_some_and(|v| v.get("image_data").is_some_and(|d| !d.is_null()))
        {
            if let Some(value) = request_value.take() {
                let ids = ids.clone();
                let t0 = std::time::Instant::now();
                // Base64 decode + sha256 of every screenshot: off the async workers.
                let (value, outcome) = tokio::task::spawn_blocking(move || {
                    let outcome = layout.prepare(&value, &ids);
                    (value, outcome)
                })
                .await
                .map_err(|e| ApiError::Internal(e.into()))?;
                crate::image_layout::record(&outcome, t0.elapsed().as_micros() as u64);
                let mut value = value;
                match outcome {
                    Ok(Some(prepared)) => {
                        if let Some(inject) = prepared.inject {
                            if let Some(obj) = value.as_object_mut() {
                                obj.insert("prepared_image_layout".to_string(), inject);
                            }
                            // Every leg (including the untouched-bytes paths) forwards it.
                            body = Bytes::from(
                                serde_json::to_vec(&value)
                                    .map_err(|e| ApiError::Internal(e.into()))?,
                            );
                        }
                        prompt_ids = Some(prepared.ids);
                    }
                    Ok(None) => {}
                    Err(skip) => {
                        tracing::debug!(
                            outcome = crate::image_layout::skip_label(skip),
                            "image layout skipped; routing on the placeholder ids"
                        );
                    }
                }
                request_value = Some(value);
            }
        }
    }

    let routing_key = crate::policies::sticky::routing_key(&ctx.config.model, &headers);
    // Salted requests publish KV events under a salt-seeded hash chain; the
    // cache-aware policy needs the salt to hash a matching query. Extracted
    // here (whenever the body was parsed) so the policy never re-parses.
    let route_cache_salt = request_value
        .as_ref()
        .and_then(crate::kv_hint::extract_cache_salt);
    let external_prefix = if let Some(index) = &ctx.dynamo_prefix {
        Some(
            crate::dynamo_prefix::query(
                index,
                prompt_ids.as_deref(),
                route_cache_salt,
                &model_str,
                ctx.block_size_oracle.get(),
                ctx.block_size_oracle.is_bigram(),
            )
            .await,
        )
    } else {
        None
    };
    let selection_ctx = SelectionContext::with_routing_key(&model_id, Some(&body), routing_key)
        .with_request_tokens(prompt_ids.as_deref())
        .with_external_prefix(external_prefix.as_ref())
        .with_cache_salt(route_cache_salt);
    // Worker selection: symmetric_pd admission (router-side queue, backlog
    // cap and the pull decided with the placement) when configured,
    // `Policy::select` otherwise. The admission may HOLD the request here
    // until capacity frees or its starvation guard fires.
    let mut spd_decision: Option<crate::scheduler::Decision> = None;
    let mut pd_replay_decode: Option<Arc<Worker>> = None;
    // A policy's per-request note (`--policy gateway_cache_aware`), logged
    // with the rid below.
    let mut policy_note: Option<serde_json::Value> = None;
    let worker = if let Some(spd) = ctx.symmetric_pd.as_ref() {
        // symmetric_pd placement: the token-id prompt IS the prompt length and
        // the indexer key (the text-prompt fallback estimates from bytes).
        let prompt_len = spd_prompt_len(
            request_value.as_ref(),
            prompt_ids.as_deref(),
            &body,
            spd.cost_function_enabled(),
        );
        let (session_id, turn) = extract_session(request_value.as_ref());
        let d = spd
            .admit(crate::scheduler::PlaceInput {
                tokens: prompt_ids.clone(),
                hashes: None,
                prompt_len,
                cache_salt: route_cache_salt.map(str::to_string),
                model_id: Some(model_str.clone()),
                session_id,
                turn,
                rid: extract_rid(request_value.as_ref()),
                routing_key: replay_routing_key(spd.replay_header(), &headers),
                ingress_ms: start.elapsed().as_secs_f64() * 1e3,
                // The cost function's copy price auto (N_active) and
                // the session-move trace.
                placement_session: if spd.cost_function_wants_session() {
                    crate::agent_identity::placement_session_key(&headers)
                } else {
                    None
                },
                // The cost function's wait forecast: the session key of the
                // target host rule.
                forecast_session: if spd.cost_function_wants_forecast_session() {
                    crate::prefill_forecast::session_key(&headers, kv_session_identity.as_ref())
                } else {
                    None
                },
            })
            .await?;
        let w = Arc::clone(&d.p_worker);
        spd_decision = Some(d);
        w
    } else {
        let pinned = ctx.pd_replay.as_ref().and_then(|replay| {
            let prompt_len = request_value
                .as_ref()
                .map(|v| crate::scheduler::estimate_prompt_len(v, prompt_ids.as_deref()))
                .unwrap_or_else(|| estimate_prefill_tokens(&body) as u64);
            let rid = extract_rid(request_value.as_ref());
            super::pd_replay::pick(
                &ctx,
                replay,
                &headers,
                prompt_len,
                &workers,
                &model_id,
                rid.as_deref(),
            )
        });
        match pinned {
            Some(p) => {
                pd_replay_decode = Some(p.decode);
                p.prefill
            }
            None => {
                let (picked, note) = policy.select_noted(&workers, &selection_ctx);
                policy_note = note;
                picked.ok_or_else(|| ApiError::PolicySelectionFailed {
                    model: model_str.clone(),
                })?
            }
        }
    };

    // PD decode-peer resolution — identical semantics to the chat route.
    // symmetric_pd: the D-leg is the placement's (Split); Local has no peer.
    // Token-charged decode policies: charge = prompt ids (else the chars/4
    // estimate) + capped output, taken at selection, released with the
    // decode LoadGuard (see the chat route).
    let decode_charge_tokens = ctx.decode_charge_tokens(
        || {
            prompt_ids
                .as_ref()
                .map(|ids| ids.len() as u64)
                .or_else(|| {
                    request_value
                        .as_ref()
                        .map(|v| crate::scheduler::estimate_prompt_len(v, None))
                })
                .unwrap_or_else(|| estimate_prefill_tokens(&body) as u64)
        },
        request_value.as_ref(),
    );
    let mut decode_charge: Option<crate::workers::DecodeTokenCharge> = None;
    let mut decode_token_log: Option<crate::policies::decode_tokens::DecodeTokenLog> = None;
    // Late D choose: the decode is chosen when the prefill binds; this is
    // the arrival pick (and the candidates) instead.
    let mut late_choose: Option<super::late_d::Choose> = None;
    let mut pending_spd_reservation = spd_decision.as_ref().map(|d| d.ledger.pending_guard());
    let decode_peer: Option<Arc<Worker>> = if let Some(d) = spd_decision.as_ref() {
        (!d.is_local()).then(|| Arc::clone(&d.d_worker))
    } else if let Some(d) = pd_replay_decode.take() {
        (decode_charge, decode_token_log) =
            ctx.charge_pinned_decode(&resolver, &model_id, &d, decode_charge_tokens);
        Some(d)
    } else if let Some(late) = ctx
        .late_bind
        .as_ref()
        .filter(|_| ingress.late_d_choose() && worker.mode() == WorkerMode::Prefill)
    {
        let pick = super::late_d::pick(&ctx, late, &resolver, &model_id, &worker)
            .map_err(|e| ApiError::from_pd_resolve(e, &model_str))?;
        late_choose = pick.choose;
        Some(pick.arrival)
    } else if worker.mode() == WorkerMode::Prefill {
        let selection = ctx
            .select_decode(&resolver, &model_id, &worker.url, decode_charge_tokens)
            .map_err(|e| ApiError::from_pd_resolve(e, &model_str))?;
        decode_charge = selection.charge;
        decode_token_log = selection.token_log;
        Some(selection.worker)
    } else {
        None
    };
    // Late D choose on symmetric workers (symmetric_pd Split): the candidates
    // and the marks; the placement's D is the arrival pick.
    let mut spd_late: Option<super::late_d::SymmetricLate> = None;
    if let (Some(late), Some(d), Some(spd)) = (
        ctx.late_bind
            .as_ref()
            .filter(|l| ingress.late_d_choose() && l.symmetric()),
        spd_decision.as_ref(),
        ctx.symmetric_pd.as_ref(),
    ) {
        if let Some((choose, marks)) = super::late_d::spd_pick(late, spd, d)? {
            late_choose = Some(choose);
            spd_late = Some(marks);
        }
    }
    let mut request_headers = headers;
    if let Some(decode) = decode_peer.as_ref().filter(|_| late_choose.is_none()) {
        if let Ok(v) = HeaderValue::from_str(&decode.url) {
            request_headers.insert(X_SGL_DECODE_URL, v);
        }
    }
    let headers = request_headers;

    let plain_usage = match (usage_asked, ctx.external_workload.as_ref()) {
        (Some(_), Some(workload)) => Some(crate::backend::plain_usage::PlainUsage::new(
            Arc::clone(workload),
            spd_decision
                .as_ref()
                .map_or(0, |d| d.prompt_len.saturating_sub(d.ledger.tokens())),
        )),
        _ => None,
    };
    let guard = worker.load_guard();
    let prefill_load = prompt_ids
        .as_ref()
        .map(|ids| ids.len().max(1))
        .unwrap_or_else(|| estimate_prefill_tokens(&body));
    let active_guard =
        ctx.active_load
            .register(worker.id.clone(), worker.url.clone(), prefill_load, 0);
    let stale_token = active_guard.cancel_token().clone();

    let metrics_worker_url = worker.url.clone();
    let metrics_mode = WorkerModeLabel::from(worker.mode());
    let metrics_model = model_str.clone();
    let make_duration_guard = || RecordDurationOnDrop {
        metrics: Arc::clone(&ctx.metrics),
        model: metrics_model.clone(),
        start,
    };

    // symmetric_pd Splits carry marks, never the bootstrap fields.
    let bootstrap = if spd_decision.is_some() {
        None
    } else {
        decode_peer.as_ref().map(|_| BootstrapFields {
            host: worker.bootstrap_host().to_string(),
            port: worker.bootstrap_port(),
            room: generate_room_id(),
        })
    };

    let admission_queue_wait_ms = spd_decision.as_ref().map_or(0, |d| d.queue_wait_ms);
    let admission_branch = spd_decision.as_ref().map_or("policy", |d| d.branch());

    // kv-p2p pull hint. symmetric_pd requests already carry the hint decided
    // with the placement (one indexer answer drove BOTH the routing and the pull — they can
    // never disagree about the owner); otherwise same post-selection
    // semantics as the chat route: fail-open, and in PD mode the hint rides
    // on the PREFILL request only.
    let (kv_pull_hint, mut co_ledger, mut cost_reservation): (
        Option<serde_json::Value>,
        Option<AdmissionLedger>,
        Option<_>,
    ) = if let Some(d) = spd_decision.as_mut() {
        // symmetric_pd: the hint (if any) was decided at admit from the placement
        // indexer reply and rides on the P-leg body only. With the
        // cost function the P-leg task owns the cost reservation.
        (
            d.pull_hint.clone(),
            Some(AdmissionLedger::Symmetric(d.ledger.clone())),
            d.cost_reservation.take(),
        )
    } else {
        let hint = match (ctx.kv_hint.as_ref(), prompt_ids.as_ref()) {
            (Some(injector), Some(ids)) => {
                let cache_salt = request_value
                    .as_ref()
                    .and_then(crate::kv_hint::extract_cache_salt);
                let answer = injector.prepare(ids, cache_salt, &model_str).await;
                let hint = answer.as_ref().and_then(|a| injector.pick(a, &worker.url));
                injector.log_decision(
                    answer.as_ref(),
                    &format!("{:?}", ctx.config.model.policy),
                    &worker.url,
                    hint.as_ref(),
                );
                hint
            }
            _ => None,
        };
        (hint, None, None)
    };

    // symmetric_pd session retention. Late D: the P registers alone and its tag
    // rides on the P-leg body only; the admitted decode gets it in the
    // ADMIT frame.
    let mut spd_late_session: Option<Arc<super::late_d::SymmetricLateSession>> = None;
    if let Some(spd) = &ctx.symmetric_pd {
        if spd_late.is_some() {
            if let (Some(sessions), Some(tag)) = (
                spd.kv_sessions(),
                crate::backend::sglang::request::late_kv_session(
                    spd.kv_sessions().map(Arc::as_ref),
                    &headers,
                    &worker.url,
                    request_value.as_ref(),
                )?,
            ) {
                spd_late_session = Some(Arc::new(super::late_d::SymmetricLateSession {
                    sessions: Arc::clone(sessions),
                    tag,
                }));
            }
        } else {
            let d_url = spd_decision
                .as_ref()
                .map(|d| d.d_worker.url.as_str())
                .or_else(|| decode_peer.as_ref().map(|d| d.url.as_str()))
                .unwrap_or(&worker.url);
            crate::backend::sglang::request::attach_kv_session(
                spd.kv_sessions().map(Arc::as_ref),
                &headers,
                &worker.url,
                d_url,
                &mut request_value,
            )?;
        }
    }

    // One common RID-keyed record for every arm, before upstream dispatch.
    // queue_wait excludes ingress parsing/indexer lookup; elapsed includes them.
    let rid = extract_rid(request_value.as_ref());
    // `--kv-session-retention`, after P selection: register (P, P) for a PD
    // dispatch and keep the tag for the P-leg body only (below). A
    // registration error sends the request untagged.
    let kv_session_tag = match (ctx.kv_sessions.as_ref(), kv_session_identity.as_ref()) {
        (Some(sessions), Some(identity)) => super::kv_session::register_prefill(
            sessions,
            identity,
            &worker.url,
            bootstrap.is_some(),
            rid.as_deref(),
        ),
        _ => None,
    };
    // The late-D key: the bootstrap room (PD) or the transfer id (symmetric_pd).
    let late_key = spd_late
        .as_ref()
        .map(|f| f.transfer_id)
        .or_else(|| bootstrap.as_ref().map(|b| b.room));
    let late_log = late_choose
        .as_ref()
        .zip(late_key)
        .map(|(c, key)| LateDispatchLog {
            rule: c.late.rule().as_str(),
            k: c.candidates.len(),
            default_worker: c.candidates[c.default_idx].url.as_str(),
            bootstrap_room: key,
        });
    log_routing_decision(&RoutingDecision {
        rid: rid.as_deref().unwrap_or(""),
        request_id: headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or(""),
        policy: ctx.config.model.policy,
        prefill_worker: &worker.url,
        decode_worker: spd_decision
            .as_ref()
            .filter(|_| spd_late.is_none())
            .map(|d| d.d_worker.url.as_str())
            .or_else(|| {
                decode_peer
                    .as_ref()
                    .filter(|_| late_choose.is_none())
                    .map(|d| d.url.as_str())
            })
            .unwrap_or(""),
        queue_wait_ms: admission_queue_wait_ms,
        branch: admission_branch,
        ingress_to_dispatch_ms: start.elapsed().as_millis() as u64,
        owner_hint: kv_pull_hint.as_ref(),
        spd: spd_decision.as_ref(),
        decode_tokens: decode_token_log.as_ref(),
        late_d: late_log.as_ref(),
    });
    if let Some(note) = policy_note.as_ref() {
        tracing::info!(
            target: "prefill_policy",
            rid = rid.as_deref().unwrap_or(""),
            prefill_worker = %worker.url,
            note = %note,
            "prefill policy decision",
        );
    }

    // First streamed byte = the decode (or plain worker) started producing,
    // which implies this request's prefill work is done — the cost-function
    // ledger entry stops counting toward the worker's backlog then. The
    // factory hands each dispatch path its own clone (retire is idempotent;
    // a clone that never fires is subsumed by the next /v1/loads poll).
    let ledger_for_hook = co_ledger.clone();
    let make_ttft_hook = || -> Box<dyn FnOnce() + Send + 'static> {
        let metrics = Arc::clone(&ctx.metrics);
        let model = metrics_model.clone();
        let started = start;
        let ledger = ledger_for_hook.clone();
        Box::new(move || {
            metrics.observe_ttft(&model, started.elapsed().as_secs_f64());
            if let Some(l) = ledger {
                l.retire();
            }
        })
    };

    // Late D choose: the d1 charge (prompt ids, else the estimate, plus the
    // capped output budget, as the token-charged decode policies charge);
    // the P-leg body carries `late_d` (and the hint), the D-leg bodies add
    // each candidate's index to one shared body.
    let late_charge_tokens = late_choose.as_ref().map(|c| {
        c.late.request_charge(
            prompt_ids
                .as_ref()
                .map(|ids| ids.len() as u64)
                .or_else(|| {
                    request_value
                        .as_ref()
                        .map(|v| crate::scheduler::estimate_prompt_len(v, None))
                })
                .unwrap_or_else(|| estimate_prefill_tokens(&body) as u64),
            request_value.as_ref(),
        )
    });
    // `--late-d-image-source-url` (symmetric_pd late D): this request's images in
    // the router's store while its D-legs go without pixels.
    let mut late_image_hold: Option<crate::image_source::Hold> = None;
    let late_bodies = match (late_choose.as_ref(), bootstrap.as_ref(), spd_late.as_ref()) {
        (Some(choose), Some(b), _) => Some(super::late_d::build_bodies(
            &body,
            request_value.take(),
            b,
            kv_pull_hint.as_ref(),
            kv_session_tag.as_ref(),
            choose,
        )?),
        // symmetric_pd: the candidate marks, the symmetric `late_d`, the pull
        // hint and the session tag on the P-leg; one `kv_import` per D-leg.
        (Some(choose), None, Some(marks)) => {
            let tag = spd_late_session
                .as_ref()
                .map(|s| serde_json::to_value(&s.tag))
                .transpose()
                .map_err(|e| ApiError::Internal(e.into()))?;
            let has_images = crate::image_source::has_prepared_images(request_value.as_ref());
            let hold = match (ctx.config.proxy.late_d_image_source_url, rid.as_deref()) {
                (Some(_), Some(token)) if has_images => crate::image_source::Hold::reserve(token),
                _ => None,
            };
            let source = match (ctx.config.proxy.late_d_image_source_url, hold.as_ref()) {
                (Some(url), Some(h)) => Some(crate::image_source::source_value(url, h.token())),
                _ => None,
            };
            let (prefill_body, legs, stripped) = super::late_d::build_spd_bodies(
                &body,
                request_value.take(),
                choose,
                marks,
                kv_pull_hint.as_ref(),
                tag.as_ref(),
                source.as_ref(),
            )?;
            if let (Some(h), Some(images)) = (hold, stripped) {
                h.fill(images);
                late_image_hold = Some(h);
            }
            Some((prefill_body, legs))
        }
        _ => None,
    };

    // `outgoing_body` stays clean (PD decode side); `executing_body` carries
    // the hint and the `--kv-session-retention` tag. `input_ids` is never
    // injected on this route (see above).
    // symmetric_pd: `executing_body` is the P-leg (`kv_export`), `outgoing_body`
    // the D-leg (`kv_import`); Local leaves the bytes untouched.
    let (outgoing_body, executing_body) = if late_bodies.is_some() {
        // Late D choose (PD or symmetric_pd): built above.
        (Bytes::new(), Bytes::new())
    } else if let Some(d) = spd_decision.as_ref() {
        let (p_body, d_body) = crate::scheduler::build_leg_bodies(
            &body,
            request_value,
            None,
            &d.marks,
            d.pull_hint.as_ref(),
        )?;
        (d_body, p_body)
    } else if let Some(hint) = &kv_pull_hint {
        let clean =
            build_outgoing_body(&body, request_value.clone(), None, bootstrap.as_ref(), None)?;
        let p_value = super::kv_session::tag_prefill(request_value, kv_session_tag.as_ref());
        let hinted = build_outgoing_body(&body, p_value, None, bootstrap.as_ref(), Some(hint))?;
        (clean, hinted)
    } else if kv_session_tag.is_some() {
        // `--kv-session-retention` (PD only): a separate P-leg body carries
        // the tag; the D leg keeps the clean body.
        let clean =
            build_outgoing_body(&body, request_value.clone(), None, bootstrap.as_ref(), None)?;
        let p_value = super::kv_session::tag_prefill(request_value, kv_session_tag.as_ref());
        let executing = build_outgoing_body(&body, p_value, None, bootstrap.as_ref(), None)?;
        (clean, executing)
    } else {
        let clean = build_outgoing_body(&body, request_value, None, bootstrap.as_ref(), None)?;
        let executing = clean.clone();
        (clean, executing)
    };
    // `--pd-late-decode-packed-prompt` with late D off: the PD decode leg
    // carries the token prompt packed.
    let outgoing_body = if ctx.config.proxy.late_decode.packed_prompt
        && late_bodies.is_none()
        && spd_decision.is_none()
        && bootstrap.is_some()
    {
        super::late_d::pack_decode_body(outgoing_body)
    } else {
        outgoing_body
    };

    // Body construction and validation above have not submitted anything.
    // From here P may execute independently of the client's connection.
    if let Some(pending) = pending_spd_reservation.as_mut() {
        pending.mark_dispatched();
    }
    let result = if let (Some(choose), Some((prefill_body, legs))) = (late_choose, late_bodies) {
        super::late_d::dispatch(
            &ctx,
            super::late_d::ChooseDispatch {
                session: spd_late_session.map(|s| s as Arc<dyn crate::late_bind::LateSessionHook>),
                choose,
                prefill: Arc::clone(&worker),
                prefill_holds: Box::new((guard, active_guard)),
                path: ingress.path(),
                headers: headers.clone(),
                prefill_body,
                legs,
                streaming,
                rid: rid.clone(),
                room: late_key.unwrap_or(0),
                charge_tokens: late_charge_tokens.unwrap_or(1),
                ledger: co_ledger.take(),
                cost: cost_reservation.take(),
                stale: stale_token.clone(),
                model: model_str,
                duration_guard: streaming.then(make_duration_guard),
                ttft_hook: streaming.then(make_ttft_hook),
                image_hold: late_image_hold.take(),
            },
        )
        .await
    } else if let Some(decode_worker) = decode_peer {
        // PD dispatch: spawn-and-forget prefill (see the chat route for the
        // full rationale — the prefill request must outlive the client
        // connection), await decode for the client-facing response.
        let prefill_url = worker.url.clone();
        let prefill_breaker = Arc::clone(&worker.breaker);
        let prefill_headers = headers.clone();
        let prefill_body = executing_body.clone();
        let prefill_proxy = Arc::clone(&ctx.proxy);
        let prefill_holds: (LoadGuard, _) = (guard, active_guard);
        let prefill_ledger = co_ledger.take();
        let mut prefill_cost = cost_reservation.take();
        // `--pd-abort-decode-on-prefill-failure`: the prefill task's verdict,
        // shared with the decode leg below (sgl-model-gateway drops the
        // decode request when its prefill leg fails).
        let prefill_failure = ctx
            .config
            .proxy
            .abort_decode_on_prefill_failure
            .then(PrefillFailure::new);
        let prefill_failure_p = prefill_failure.clone();
        let usage_p = plain_usage.clone();
        let decode_url_p = decode_worker.url.clone();
        let rid_p = rid.clone();
        let body_for_rid = prefill_failure.as_ref().map(|_| prefill_body.clone());
        let bootstrap_room_p = bootstrap.as_ref().map(|b| b.room);
        tokio::spawn(async move {
            let _hold = prefill_holds;
            let legs = PdLegs {
                rid: rid_p.as_deref(),
                bootstrap_room: bootstrap_room_p,
                prefill_url: &prefill_url,
                decode_url: &decode_url_p,
                body: body_for_rid.as_deref(),
            };
            if let Some(cost) = prefill_cost.as_mut() {
                if let Err(error) = cost.mark_dispatched() {
                    tracing::error!(target: "prefill_work", %error, "placement trace failed before P dispatch");
                    if let Some(failure) = &prefill_failure_p {
                        failure.fail_and_log(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "prefill leg not dispatched: placement trace failed".to_string(),
                            &legs,
                        );
                    }
                    return;
                }
            }
            let mut prefill_result = prefill_proxy
                .forward_json_to(
                    &prefill_url,
                    &prefill_breaker,
                    ingress.path(),
                    &prefill_headers,
                    prefill_body,
                )
                .await;
            if let Some(failure) = &prefill_failure_p {
                if let Some((status, detail)) = PrefillFailure::verdict(&prefill_result) {
                    failure.fail_and_log(status, detail, &legs);
                }
            }
            settle_prefill_leg(prefill_cost.as_mut(), prefill_ledger, &prefill_result);
            crate::backend::plain_usage::read_prefill(usage_p.as_ref(), &mut prefill_result).await;
            match prefill_result {
                // Cancelled, not failed: the engine finished the P-leg
                // because its reader left; the D-leg owns the outcome.
                ref r if crate::proxy::is_reader_gone(r) => tracing::warn!(
                    prefill_url = %prefill_url,
                    "prefill leg ended: its reader left (symmetric_reader_gone); the decode leg \
                     owns the request",
                ),
                Ok(_) => tracing::debug!(prefill_url = %prefill_url, "prefill side completed"),
                Err(e) => tracing::warn!(
                    prefill_url = %prefill_url,
                    error = %e,
                    "prefill request failed; decode will time out on bootstrap_room",
                ),
            }
        });

        // Token charge rides with the decode LoadGuard (see the chat route).
        // SMG request count (`power-of-two-smg`), see the chat route.
        let (smg_load, smg_arm) = ctx.decode_smg_load(&decode_worker.url, streaming);
        // symmetric_pd: the D-leg's place in its worker's in-flight count
        // lasts as long as the leg (the stream, when streaming).
        let d_reservation = spd_decision.as_mut().and_then(|d| d.d_reservation.take());
        let decode_guard = (
            decode_worker.load_guard(),
            decode_charge.take(),
            smg_load,
            d_reservation,
        );
        if streaming {
            let stream_guards: Box<dyn Send + 'static> =
                Box::new((decode_guard, make_duration_guard()));
            // A prefill failure after the decode headers reached the client
            // ends the stream inside the SSE pump (the token below).
            let fetch = ctx.proxy.forward_streaming_to_abortable(
                &decode_worker.url,
                &decode_worker.breaker,
                ingress.path(),
                &headers,
                outgoing_body,
                Some(stream_guards),
                Some(make_ttft_hook()),
                prefill_failure.as_ref().map(PrefillFailure::token),
            );
            let r = tokio::select! {
                biased;
                error = prefill_failed(prefill_failure.as_ref()) => Err(error),
                r = fetch => r,
                _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
            };
            if let (Ok(_), Some(arm)) = (&r, smg_arm) {
                arm.arm();
            }
            r
        } else {
            let _decode_hold = decode_guard;
            let fetch = ctx.proxy.forward_json_to(
                &decode_worker.url,
                &decode_worker.breaker,
                ingress.path(),
                &headers,
                outgoing_body,
            );
            tokio::select! {
                biased;
                error = prefill_failed(prefill_failure.as_ref()) => Err(error),
                r = fetch => r,
                _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
            }
        }
    } else if streaming {
        let stream_guards: Box<dyn Send + 'static> =
            Box::new((guard, active_guard, make_duration_guard()));
        let fetch = ctx.proxy.forward_streaming_to(
            &worker.url,
            &worker.breaker,
            ingress.path(),
            &headers,
            executing_body,
            Some(stream_guards),
            Some(make_ttft_hook()),
        );
        tokio::select! {
            biased;
            r = fetch => r,
            _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
        }
    } else {
        let _holds: (LoadGuard, _) = (guard, active_guard);
        let fetch = ctx.proxy.forward_json_to(
            &worker.url,
            &worker.breaker,
            ingress.path(),
            &headers,
            executing_body,
        );
        let r = tokio::select! {
            biased;
            r = fetch => r,
            _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
        };
        // Non-streaming plain: the full generation is done here — retire
        // the cost-function ledger entry (later than "prefill done", but the
        // only completion signal this path has; the poller subsumes it
        // within a poll interval regardless).
        if let Some(l) = co_ledger.take() {
            l.retire();
        }
        r
    };

    let outcome = match &result {
        Ok(_) => RequestOutcome::Success,
        Err(ApiError::StaleRequestExpired { .. }) => {
            ctx.metrics
                .record_stale_request(StaleRequestOutcome::Expired);
            RequestOutcome::Cancelled
        }
        Err(_) => RequestOutcome::Error,
    };
    ctx.metrics
        .record_worker_request(&metrics_worker_url, &metrics_model, metrics_mode, outcome);

    let request_id = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-");
    let http_status = match &result {
        Ok(resp) => resp.status().as_u16(),
        Err(e) => e.status_code().as_u16(),
    };
    tracing::info!(
        target: "access",
        route = ingress.path(),
        model = %metrics_model,
        worker = %metrics_worker_url,
        status = http_status,
        stream = streaming,
        request_id = request_id,
        elapsed_ms = start.elapsed().as_millis() as u64,
        "completions request",
    );

    crate::backend::plain_usage::count(result, plain_usage, usage_asked)
}

/// Client request id from the body (`rid` — the strict-replay client sends
/// one per request). Also correlates engine progress in uncached balance mode.
pub(crate) fn extract_rid(v: Option<&serde_json::Value>) -> Option<String> {
    v.and_then(|b| b.get("rid"))
        .and_then(|r| r.as_str())
        .map(String::from)
}

/// The decision-replay routing key: the value of the replay header, when
/// decision replay is on (`header` is `Some`) and the request carries it as
/// valid UTF-8.
pub(crate) fn replay_routing_key(header: Option<&str>, headers: &HeaderMap) -> Option<String> {
    header
        .and_then(|h| headers.get(h))
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

/// Prompt length of a symmetric_pd placement (the completions and chat routes).
///
/// # Inputs
/// `value`: the parsed body; `ids`: the prompt token ids (an array-form
/// prompt, or the chat route's router-side tokens); `body`: the raw body;
/// `cost_function`: `--symmetric-pd-p-select cost-function` is on.
///
/// # Outputs
/// The token-id count when there are ids. Without ids, the cost function
/// uses the estimate of the cost function cost-function path (the whole body's bytes /
/// 4), so both cost-function paths reserve the same backlog for the same request.
/// The other P selections estimate from the prompt or message text.
pub(crate) fn spd_prompt_len(
    value: Option<&serde_json::Value>,
    ids: Option<&[u32]>,
    body: &Bytes,
    cost_function: bool,
) -> u64 {
    if ids.is_none() && cost_function {
        return estimate_prefill_tokens(body) as u64;
    }
    value
        .map(|v| crate::scheduler::estimate_prompt_len(v, ids))
        .unwrap_or_else(|| estimate_prefill_tokens(body) as u64)
}

/// `(session_id, turn)` from the body; echoed on the symmetric_pd placement
/// line only (no effect on the score).
pub(crate) fn extract_session(v: Option<&serde_json::Value>) -> (Option<String>, Option<u32>) {
    let session_id = v
        .and_then(|b| b.get("session_id"))
        .and_then(|s| s.as_str())
        .map(String::from);
    let turn = v
        .and_then(|b| b.get("turn"))
        .and_then(|t| t.as_u64())
        .and_then(|t| u32::try_from(t).ok());
    (session_id, turn)
}

/// PD forwarding preserves this RID on both sides. Ordinary clients need not
/// supply an engine RID; strict replay keeps its original correlation ID.
/// The uncached cost ledger (`--symmetric-pd-p-select cost-function`) keys
/// engine progress by it, and under `--policy symmetric_pd` the Backend ledger
/// needs it to retire the reservation of a P-leg that ended without terminal
/// evidence.
pub(crate) fn ensure_progress_rid(
    ctx: &AppContext,
    value: Option<&mut serde_json::Value>,
) -> Result<(), ApiError> {
    let uncached_ledger = ctx.symmetric_pd.is_some();
    if !uncached_ledger {
        return Ok(());
    }
    let object = value
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| ApiError::BadRequest("uncached routing requires a JSON object".into()))?;
    match object.get("rid") {
        None | Some(serde_json::Value::Null) => {
            object.insert(
                "rid".into(),
                serde_json::json!(format!("rid_{}", uuid::Uuid::new_v4().simple())),
            );
        }
        Some(serde_json::Value::String(rid)) if !rid.is_empty() => {}
        _ => return Err(ApiError::BadRequest("rid must be a nonempty string".into())),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn prompt_token_ids(v: &serde_json::Value) -> Option<Vec<u32>> {
        Ingress::Completions.token_ids(v)
    }

    #[test]
    fn generate_token_ids_read_input_ids_only() {
        let v = json!({"input_ids": [4, 5], "prompt": [1, 2]});
        assert_eq!(Ingress::Generate.token_ids(&v), Some(vec![4, 5]));
        assert_eq!(Ingress::Generate.token_ids(&json!({"text": "hello"})), None);
    }

    #[test]
    fn prompt_token_ids_reads_flat_array() {
        let v = json!({"prompt": [1, 2, 3]});
        assert_eq!(prompt_token_ids(&v), Some(vec![1, 2, 3]));
    }

    #[test]
    fn prompt_token_ids_rejects_other_shapes() {
        assert_eq!(prompt_token_ids(&json!({"prompt": "hello"})), None);
        assert_eq!(prompt_token_ids(&json!({"prompt": ["a", "b"]})), None);
        assert_eq!(prompt_token_ids(&json!({"prompt": [[1, 2], [3]]})), None);
        assert_eq!(prompt_token_ids(&json!({"prompt": []})), None);
        assert_eq!(prompt_token_ids(&json!({"prompt": [1, -5]})), None);
        assert_eq!(prompt_token_ids(&json!({"other": 1})), None);
    }

    /// A text prompt: the cost function takes the cost function cost-function
    /// estimate (body bytes / 4); the other P selections keep the prompt
    /// text estimate. Token ids: the id count in both cases.
    #[test]
    fn spd_prompt_len_uses_the_cost_estimate_for_the_cost_function() {
        let v = json!({"model": "m", "prompt": "abcdefgh".repeat(100), "max_tokens": 4});
        let body = Bytes::from(serde_json::to_vec(&v).unwrap());
        let cost_params = estimate_prefill_tokens(&body) as u64;
        assert!(
            cost_params > 200,
            "the body estimate must differ from the text estimate"
        );
        assert_eq!(spd_prompt_len(Some(&v), None, &body, true), cost_params);
        assert_eq!(spd_prompt_len(Some(&v), None, &body, false), 200);
        let tv = json!({"prompt": [1, 2, 3]});
        let tbody = Bytes::from(serde_json::to_vec(&tv).unwrap());
        let ids = [1_u32, 2, 3];
        assert_eq!(spd_prompt_len(Some(&tv), Some(&ids), &tbody, true), 3);
        assert_eq!(spd_prompt_len(Some(&tv), Some(&ids), &tbody, false), 3);
    }
}
