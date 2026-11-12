// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::decision_log::{log_routing_decision, AdmissionLedger, RoutingDecision};
use crate::discovery::{ModelId, WorkerMode};
use crate::policies::kv_events::{compute_block_hashes, compute_block_hashes_bigram};
use crate::policies::registry::PdPoolResolver;
use crate::policies::{request_tokens_for, ExternalPrefixSignal, RequestTokens, SelectionContext};
use crate::proxy::{prefill_failed, PdLegs, PrefillFailure};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::{
    MetricsRegistry, RequestOutcome, StaleRequestOutcome, WorkerModeLabel,
};
use crate::workers::{LoadGuard, Worker};
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response, StatusCode};
use bytes::Bytes;
use serde::de::IgnoredAny;
use serde::Deserialize;
use sgl_kv_indexer::PrefixIndex;
use std::collections::HashMap;
use std::sync::Arc;

/// Observability header carrying the decode-pool URL selected via host
/// affinity for a PD-disaggregated request. The router fans the
/// bootstrap-injected request body to BOTH the prefill and the decode
/// worker concurrently; this header lets the prefill log the chosen
/// peer, and is mirrored onto the response so sidecars / tests can
/// observe affinity without sniffing the proxy hop. The `x-sgl-`
/// prefix matches `x-sgl-router-error-code` so router-emitted metadata
/// stays grouped.
pub(crate) const X_SGL_DECODE_URL: HeaderName = HeaderName::from_static("x-sgl-decode-url");

/// Coarse char-count → token-count divisor used to estimate prefill load
/// from the request body when no real tokenizer count is available. Four
/// bytes per token is the standard SGLang upstream estimate; it
/// overcounts ASCII and undercounts CJK but stays within an order of
/// magnitude of the real token count, which is plenty for load
/// scoring. The active-load counters' role is relative ordering across
/// workers — not absolute accuracy — so the estimate is fit for
/// purpose.
const CHARS_PER_TOKEN_ESTIMATE: usize = 4;

/// Per-route body-size cap on `/v1/chat/completions`. 5 MiB accommodates a
/// long context — a ~1 M-token context tokenized as JSON fits under this —
/// while preventing a hostile client from forcing the router to
/// heap-allocate hundreds of MiB before forwarding. The cap is wired in
/// `crate::server::app::build_router` as a route-level `DefaultBodyLimit`
/// layer; axum's `Bytes` extractor enforces it and returns 413
/// PAYLOAD_TOO_LARGE before this handler runs.
pub const MAX_CHAT_BODY_BYTES: usize = 5 << 20;

/// Minimal probe over the request body — we only need the `stream` field
/// and the `model` field to decide between buffered vs SSE forwarding and
/// to select a worker. Deserializing into this struct (vs `serde_json::Value`)
/// does two things:
///
/// 1. Avoids the per-field heap allocation of `Value` for a multi-MiB body.
/// 2. Pins the contract: the body MUST be a JSON object. Degenerate
///    shapes (`null`, `[]`, `"hi"`) fail at this step rather than being
///    silently forwarded with `stream=false`.
///
/// All other fields are ignored — the worker is authoritative for the
/// full request schema.
#[derive(Debug, Deserialize)]
pub(crate) struct RequestProbe {
    #[serde(default)]
    pub(crate) stream: Option<bool>,
    #[serde(default)]
    pub(crate) model: Option<String>,
}

/// RAII guard that records `sgl_router_request_duration_seconds` when
/// dropped. For streaming requests the handler returns at response-headers
/// time, so recording end-to-end latency at the dispatch site would capture
/// only time-to-headers (≈ TTFT). Instead this guard is packed into the SSE
/// pump's `stream_guards`, so it drops — and records — when the stream
/// completes (or the client disconnects), yielding true end-to-end latency.
/// Non-streaming requests record at the dispatch site directly (the body is
/// already buffered there) and do not use this guard.
pub(crate) struct RecordDurationOnDrop {
    pub(crate) metrics: Arc<MetricsRegistry>,
    pub(crate) model: String,
    pub(crate) start: std::time::Instant,
}

impl Drop for RecordDurationOnDrop {
    fn drop(&mut self) {
        self.metrics
            .observe_request_duration(&self.model, self.start.elapsed().as_secs_f64());
    }
}

/// POST /v1/chat/completions — parse model from body, select a healthy
/// worker via the per-model policy, then proxy the request. If the
/// request opts into streaming (`stream: true`), we pipe SSE bytes back;
/// otherwise buffer.
pub async fn chat_completions(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    mut body: Bytes,
) -> Result<Response<Body>, ApiError> {
    // `--pd-late-decode choose`: the completions dispatch serves the request
    // (one decode leg per candidate; a released or failed candidate never
    // reaches the client).
    if ctx.late_bind.is_some() {
        return crate::server::routes::completions::chat(ctx, headers, body).await;
    }
    let start = std::time::Instant::now();
    let probe = parse_probe(&body)?;
    let streaming = probe.stream.unwrap_or(false);
    let model_str = probe
        .model
        .ok_or_else(|| ApiError::BadRequest("missing `model` field".into()))?;
    let model_id = ModelId(model_str.clone());

    // PD pool isolation: for PD-mode deployments, prefill traffic
    // selects from the prefill pool only. Plain-mode deployments fall
    // through to the full candidate set. Partial-failure errors
    // (`no_prefill_workers_available`) are surfaced as 503 with a
    // distinct error code so operators can alert independently.
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

    // Tokenize once at ingress whenever it can pay off — decoupled from the
    // routing policy, because forwarding `input_ids` is a property of the
    // MODEL (does it have a chat encoder so the router can produce
    // engine-equivalent tokens?), not of how we pick the worker. Two gates:
    //
    //   * `has_chat_encoder` → a chat request on this model yields
    //     engine-equivalent ids we can forward as `input_ids` so the engine
    //     skips re-tokenizing. This enables the offload for EVERY policy —
    //     sticky and round-robin included — not just cache-aware.
    //   * `needs_request_tokens()` → the cache-aware policy ALSO wants the
    //     raw-prompt path tokenized for tree matching even on a model with no
    //     chat encoder (`/v1/completions` / `text`), which the first gate
    //     alone wouldn't trigger.
    //
    // When neither holds, `parse_probe`'s minimal probe is enough, so we keep
    // avoiding the full `serde_json::Value` allocation over a (up to 1 MiB)
    // body. When parsed, this single value is reused for the routing
    // tokenization and the outgoing-body injection below (and PD bootstrap
    // injection). `parse_probe` already validated the object shape.
    // The kv-pull-hint injector also needs the request's token ids (its
    // indexer query is keyed on them), independent of the routing policy.
    let want_tokens = headers.contains_key(crate::agent_identity::CONTEXT_HEADER)
        || ctx.tokenizers.has_chat_encoder(&model_str)
        || policy.needs_request_tokens()
        || ctx.kv_hint.is_some()
        || ctx.symmetric_pd.is_some();
    let mut request_value: Option<serde_json::Value> = if want_tokens {
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
    crate::server::routes::completions::ensure_progress_rid(&ctx, request_value.as_mut())?;
    // External role control: the controller's workload sums count this
    // request from its replies (see `backend::plain_usage`).
    let usage_asked = match (ctx.external_workload.as_ref(), request_value.as_mut()) {
        (Some(_), Some(value)) if ctx.symmetric_pd.is_some() => {
            let asked = crate::backend::plain_usage::ask(value, streaming);
            body =
                Bytes::from(serde_json::to_vec(value).map_err(|e| ApiError::Internal(e.into()))?);
            Some(asked)
        }
        _ => None,
    };

    // The ids feed both the routing decision (cache-aware consumes them; other
    // policies ignore them) and — when engine-equivalent — the engine itself,
    // forwarded as `input_ids` so it skips re-tokenizing the same prompt. The
    // ingress owns the tokenize via the shared registry, so the choice of
    // policy never changes whether we tokenize.
    let request_tokens = match request_value.as_ref() {
        Some(v) => chat_request_tokens(&ctx, &model_id, v, &workers).await,
        None => None,
    };
    let external_prefix = if let Some(index) = &ctx.dynamo_prefix {
        Some(
            crate::dynamo_prefix::query(
                index,
                request_tokens.as_ref().map(|t| t.ids.as_slice()),
                request_value
                    .as_ref()
                    .and_then(crate::kv_hint::extract_cache_salt),
                &model_str,
                ctx.block_size_oracle.get(),
                ctx.block_size_oracle.is_bigram(),
            )
            .await,
        )
    } else {
        match (
            ctx.prefix_index.as_ref(),
            request_tokens.as_ref(),
            ctx.block_size_oracle.get(),
        ) {
            (Some(index), Some(tokens), Some(block_size)) => {
                let hashes = if ctx.block_size_oracle.is_bigram() {
                    compute_block_hashes_bigram(&tokens.ids, block_size as usize)
                } else {
                    compute_block_hashes(&tokens.ids, block_size as usize)
                };
                let query_blocks = hashes.len();
                let outcome = if hashes.is_empty() {
                    sgl_kv_indexer::PrefixOutcome::Empty
                } else {
                    resolve_prefix_query(index.match_prefix(hashes).await, &model_str)?
                };
                Some(ExternalPrefixSignal {
                    outcome,
                    query_blocks,
                })
            }
            (Some(_), _, _) => Some(ExternalPrefixSignal {
                outcome: sgl_kv_indexer::PrefixOutcome::Empty,
                query_blocks: 0,
            }),
            _ => None,
        }
    };

    // Sticky-session routing key. When the sticky policy is configured,
    // read the routing key from the operator-chosen header into the
    // selection context; the policy pins it to a worker. Other policies
    // leave `routing_key` `None` and ignore it.
    let routing_key = crate::policies::sticky::routing_key(&ctx.config.model, &headers);
    // Salted requests publish KV events under a salt-seeded hash chain; the
    // cache-aware policy needs the salt to hash a matching query. Extracted
    // here (whenever the body was parsed) so the policy never re-parses.
    let route_cache_salt = request_value
        .as_ref()
        .and_then(crate::kv_hint::extract_cache_salt);
    let selection_ctx = SelectionContext::with_routing_key(&model_id, Some(&body), routing_key)
        .with_request_tokens(request_tokens.as_ref().map(|t| t.ids.as_slice()))
        .with_external_prefix(external_prefix.as_ref())
        .with_cache_salt(route_cache_salt);
    // Worker selection: symmetric_pd admission (router-side queue, backlog
    // cap and the pull decided with the placement) when configured,
    // `Policy::select` otherwise. The admission may HOLD the request here
    // until capacity frees or its starvation guard fires.
    let mut spd_decision: Option<crate::scheduler::Decision> = None;
    // Traditional-PD decision replay: the pinned decode peer (routes/pd_replay.rs).
    let mut pd_replay_decode: Option<Arc<Worker>> = None;
    let worker = if let Some(spd) = ctx.symmetric_pd.as_ref() {
        // symmetric_pd placement: the ingress tokens are the prompt length and
        // the indexer key (text fallback estimates from bytes).
        let ids = request_tokens.as_ref().map(|t| t.ids.as_slice());
        let prompt_len = crate::server::routes::completions::spd_prompt_len(
            request_value.as_ref(),
            ids,
            &body,
            spd.cost_function_enabled(),
        );
        let (session_id, turn) =
            crate::server::routes::completions::extract_session(request_value.as_ref());
        let d = spd
            .admit(crate::scheduler::PlaceInput {
                tokens: ids.map(<[u32]>::to_vec),
                hashes: None,
                prompt_len,
                cache_salt: route_cache_salt.map(str::to_string),
                model_id: Some(model_str.clone()),
                session_id,
                turn,
                rid: crate::server::routes::completions::extract_rid(request_value.as_ref()),
                routing_key: crate::server::routes::completions::replay_routing_key(
                    spd.replay_header(),
                    &headers,
                ),
                ingress_ms: start.elapsed().as_secs_f64() * 1e3,
                // The cost function's copy price auto (N_active) and the
                // session-move trace.
                placement_session: if spd.cost_function_wants_session() {
                    crate::agent_identity::placement_session_key(&headers)
                } else {
                    None
                },
                // The cost function's wait forecast: the session key of the
                // target host rule (the header's identity).
                forecast_session: if spd.cost_function_wants_forecast_session() {
                    crate::prefill_forecast::session_key(&headers, None)
                } else {
                    None
                },
            })
            .await?;
        // Role control: the placement was made outside the role lock. A
        // pair with a worker that has left its role since gets no leg
        // (503: the client retries and is placed again). The reservations
        // the placement made are already counted, so a drain that starts
        // after this check waits for this request's D-leg.
        if spd.config().external_role_control && !d.is_local() {
            let current = spd.with_current_role_pair(&d.p_worker.url, &d.d_worker.url, || Ok(()));
            if let Err(reason) = current {
                d.ledger.retire();
                return Err(ApiError::SymmetricPdNoPlacement(reason));
            }
        }
        let w = Arc::clone(&d.p_worker);
        spd_decision = Some(d);
        w
    } else {
        let pinned = ctx.pd_replay.as_ref().and_then(|replay| {
            let ids = request_tokens.as_ref().map(|t| t.ids.as_slice());
            let prompt_len = request_value
                .as_ref()
                .map(|v| crate::scheduler::estimate_prompt_len(v, ids))
                .unwrap_or_else(|| estimate_prefill_tokens(&body) as u64);
            let rid = crate::server::routes::completions::extract_rid(request_value.as_ref());
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
            None => policy.select(&workers, &selection_ctx).ok_or_else(|| {
                ApiError::PolicySelectionFailed {
                    model: model_str.clone(),
                }
            })?,
        }
    };

    // PD-mode decoder affinity. When the selected prefill worker is
    // part of a PD-disagg deployment, also resolve the matching decode
    // peer (same host where possible, falling back to min-load via
    // `select_decode_with_affinity`). Both workers receive the SAME
    // request body — augmented with the three flat `bootstrap_*`
    // fields below — so the SGLang engine can match incoming KV
    // transfers via `bootstrap_room`.
    //
    // Plain-mode workers skip the decode resolution entirely (no
    // decode peer to find). PD-mode requests that fail to resolve a
    // decode peer (`NoDecodeWorkersAvailable`) bubble up as 503 so
    // operators can alert on prefill-vs-decode pool imbalance.
    // symmetric_pd: the D-leg is the placement's (Split); Local has no peer.
    //
    // Token-charged decode policies: the charge (prompt tokens, else the
    // chars/4 estimate, + capped output) is taken at selection and released
    // when `decode_charge` drops — it rides with the decode `LoadGuard`.
    let decode_charge_tokens = ctx.decode_charge_tokens(
        || {
            request_tokens
                .as_ref()
                .map(|t| t.ids.len() as u64)
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
    let mut pending_spd_reservation = spd_decision.as_ref().map(|d| d.ledger.pending_guard());
    let decode_peer: Option<Arc<Worker>> = if let Some(d) = spd_decision.as_ref() {
        (!d.is_local()).then(|| Arc::clone(&d.d_worker))
    } else if let Some(d) = pd_replay_decode.take() {
        (decode_charge, decode_token_log) =
            ctx.charge_pinned_decode(&resolver, &model_id, &d, decode_charge_tokens);
        Some(d)
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
    let decode_hint_url: Option<String> = decode_peer.as_ref().map(|d| d.url.clone());
    let mut request_headers = headers;
    if let Some(url) = &decode_hint_url {
        match HeaderValue::from_str(url) {
            Ok(v) => {
                request_headers.insert(X_SGL_DECODE_URL, v);
            }
            Err(e) => {
                // Discovery emits URLs the proxy has already used; a
                // header-value parse failure here means the URL
                // contains a control character (e.g. CR / LF) — drop
                // the header but keep the request: bootstrap injection
                // below carries the host/port the engine actually
                // needs; the header is purely observability.
                tracing::warn!(
                    decode_url = %url,
                    error = %e,
                    "decode worker URL rejected by header parser; sending request without decode hint",
                );
            }
        }
    }
    let headers = request_headers;

    // Per-worker `active_requests` guard. The `ActiveLoadGuard` below
    // sits beside this one: both track in-flight load, but the
    // ActiveLoadGuard entry is per-request (with timeout-based janitor)
    // while the worker-scoped counter is what the cache-aware policy
    // reads. Both must drop at the same time — when the response stream
    // ends, the client disconnects, or the handler returns an error. In
    // PD mode the pair moves into the spawned prefill task so prefill
    // load is tracked for the full duration of the KV transfer; in plain
    // mode the pair stays in this handler. Decode-load contribution is
    // 0 here: the active-load registry's decode axis is reserved for a
    // future decode-side scheduler — current decode selection is
    // host-affinity only.
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
    // Use the exact token count from the ingress tokenization when available;
    // fall back to the byte-count heuristic for load-only policies that don't
    // tokenize. The exact count makes the cache-aware load-imbalance fast-path
    // accurate rather than off by the char/token ratio.
    let prefill_load = request_tokens
        .as_ref()
        .map(|t| t.ids.len().max(1))
        .unwrap_or_else(|| estimate_prefill_tokens(&body));
    let active_guard =
        ctx.active_load
            .register(worker.id.clone(), worker.url.clone(), prefill_load, 0);
    // Snapshot the stale-request cancel token BEFORE moving the guard
    // into the spawned prefill task / streaming pump / response future.
    // The token is cheap to clone (it's an `Arc<...>` internally) and
    // the chat handler races the client-facing fetch against
    // `token.cancelled()` to surface a 504 `stale_request_expired` if
    // the janitor expires the request mid-flight.
    let stale_token = active_guard.cancel_token().clone();

    // Snapshot the labels we need for metrics BEFORE moving the worker
    // / model_str values into the per-branch fetch futures.
    let metrics_worker_url = worker.url.clone();
    let metrics_mode = WorkerModeLabel::from(worker.mode());
    let metrics_model = model_str.clone();

    // Builds the time-to-first-token hook the SSE pump fires when the first
    // upstream chunk lands. Installed only on the streaming arms below —
    // non-streaming "first token" equals total latency, already captured by
    // `sgl_router_request_duration_seconds`. The proxy drops the hook for
    // non-2xx responses so error bodies don't pollute TTFT.
    // (The TTFT hook factory is built AFTER the kv-pull-hint block below —
    // it also retires the cost-function backlog ledger entry at first byte.)

    // Builds the end-to-end-latency guard for streaming requests. Packed into
    // `stream_guards` so it records when the SSE pump finishes (stream end or
    // client disconnect), not at response-headers time. Non-streaming records
    // at the dispatch site instead (see below).
    let make_duration_guard = || RecordDurationOnDrop {
        metrics: Arc::clone(&ctx.metrics),
        model: metrics_model.clone(),
        start,
    };

    // Forward the router-computed tokens to the engine as `input_ids` so it
    // skips re-tokenizing the same prompt — but only when they are
    // engine-equivalent (chat-encoder path) AND the request contains nothing
    // the router's encoder didn't replicate (see `input_ids_safe_to_forward`).
    // Otherwise omit them and the engine tokenizes from `messages` as usual —
    // a transparent, always-correct fallback (`messages` are always retained
    // in the forwarded body). `forward_input_ids` is `Some` only when
    // `request_value` is `Some` (a model the ingress tokenized for), so the
    // predicate always has a parsed body to inspect.
    let forward_input_ids: Option<&[u32]> = match (request_tokens.as_ref(), request_value.as_ref())
    {
        (Some(t), Some(v)) if t.engine_equivalent && input_ids_safe_to_forward(v) => {
            Some(t.ids.as_slice())
        }
        _ => None,
    };

    // Surface a broken offload: when the encoder SHOULD have produced
    // engine-equivalent ids but didn't, the chat request silently fell back to
    // engine-side tokenization. Count only that case (see
    // `ingress_tokenize_offload_failed`); successful forwards and expected
    // omissions are not problems.
    if ingress_tokenize_offload_failed(
        ctx.tokenizers.has_chat_encoder(&model_str),
        request_value.as_ref(),
        request_tokens.as_ref(),
    ) {
        ctx.metrics.record_ingress_tokenize_error(&metrics_model);
    }

    // PD-disagg bootstrap fields (prefill worker address + a per-request
    // room). Present only when a decode peer was resolved; symmetric_pd Splits
    // carry marks instead.
    let bootstrap = if spd_decision.is_some() {
        None
    } else {
        decode_peer.as_ref().map(|_| BootstrapFields {
            host: worker.bootstrap_host().to_string(),
            port: worker.bootstrap_port(),
            room: generate_room_id(),
        })
    };
    let bootstrap_room = bootstrap.as_ref().map(|b| b.room);

    let admission_queue_wait_ms = spd_decision.as_ref().map_or(0, |d| d.queue_wait_ms);
    let admission_branch = spd_decision.as_ref().map_or("policy", |d| d.branch());

    // kv-p2p pull hint: ask the indexer which peers own this request's
    // token prefix, then (below the gates) hint the EXECUTING worker to
    // RDMA-pull the matched pages from that peer instead of recomputing.
    // Fail-open at every step; `None` when the feature is off, the request
    // carries no tokens, or no peer beats the gates. In PD mode the
    // executing worker is the prefill side — decode never pulls.
    let (kv_pull_hint, mut co_ledger, mut cost_reservation): (
        Option<serde_json::Value>,
        Option<AdmissionLedger>,
        Option<_>,
    ) = if let Some(d) = spd_decision.as_mut() {
        // symmetric_pd: hint decided at admit, P-leg body only. With the
        // cost function the P-leg task owns the cost reservation.
        (
            d.pull_hint.clone(),
            Some(AdmissionLedger::Symmetric(d.ledger.clone())),
            d.cost_reservation.take(),
        )
    } else {
        let hint = match (ctx.kv_hint.as_ref(), request_tokens.as_ref()) {
            (Some(injector), Some(tokens)) => {
                let cache_salt = request_value
                    .as_ref()
                    .and_then(crate::kv_hint::extract_cache_salt);
                let answer = injector.prepare(&tokens.ids, cache_salt, &model_str).await;
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

    if let Some(spd) = &ctx.symmetric_pd {
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

    // One common RID-keyed record for every arm, before upstream dispatch.
    // queue_wait excludes ingress tokenization/indexer lookup; elapsed includes them.
    let rid = crate::server::routes::completions::extract_rid(request_value.as_ref());
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
            .map(|d| d.d_worker.url.as_str())
            .or_else(|| decode_peer.as_ref().map(|d| d.url.as_str()))
            .unwrap_or(""),
        queue_wait_ms: admission_queue_wait_ms,
        branch: admission_branch,
        ingress_to_dispatch_ms: start.elapsed().as_millis() as u64,
        owner_hint: kv_pull_hint.as_ref(),
        spd: spd_decision.as_ref(),
        decode_tokens: decode_token_log.as_ref(),
        late_d: None,
    });

    // First streamed byte = the decode (or plain worker) started producing,
    // which implies this request's prefill work is done — the cost-function
    // ledger entry stops counting toward the worker's backlog then. Each
    // dispatch path gets its own clone (retire is idempotent; a clone that
    // never fires is subsumed by the next /v1/loads poll).
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

    // Build the bodies forwarded to the engine(s) — injecting the
    // `input_ids`, bootstrap fields and/or pull hint, or forwarding the
    // original bytes untouched when nothing applies. `outgoing_body` stays
    // CLEAN of the pull hint (it is what a PD decode worker receives);
    // `executing_body` carries the hint and goes to whichever worker runs
    // the prefill (the prefill side in PD mode, the sole worker otherwise).
    // symmetric_pd: `executing_body` is the P-leg (`kv_export`), `outgoing_body`
    // the D-leg (`kv_import`); Local leaves only `input_ids` injected.
    let (outgoing_body, executing_body) = if let Some(d) = spd_decision.as_ref() {
        let (p_body, d_body) = crate::scheduler::build_leg_bodies(
            &body,
            request_value,
            forward_input_ids,
            &d.marks,
            d.pull_hint.as_ref(),
        )?;
        (d_body, p_body)
    } else if let Some(hint) = &kv_pull_hint {
        let clean = build_outgoing_body(
            &body,
            request_value.clone(),
            forward_input_ids,
            bootstrap.as_ref(),
            None,
        )?;
        let hinted = build_outgoing_body(
            &body,
            request_value,
            forward_input_ids,
            bootstrap.as_ref(),
            Some(hint),
        )?;
        (clean, hinted)
    } else {
        let clean = build_outgoing_body(
            &body,
            request_value,
            forward_input_ids,
            bootstrap.as_ref(),
            None,
        )?;
        // `Bytes` clone is a refcount bump; both names refer to the same buffer.
        let executing = clean.clone();
        (clean, executing)
    };

    // Body construction and validation above have not submitted anything.
    // From here P may execute independently of the client's connection.
    if let Some(pending) = pending_spd_reservation.as_mut() {
        pending.mark_dispatched();
    }
    let result = if let Some(decode_worker) = decode_peer {
        // PD-disagg dispatch (Pattern B — spawn prefill, await decode).
        //
        // SGLang's HTTP-mode disagg-prefill requires three flat
        // top-level fields on the request body: `bootstrap_host`,
        // `bootstrap_port` (the prefill worker's bootstrap-server
        // address) and `bootstrap_room` (a per-request 63-bit u64 ID
        // used by both sides to pair up the KV transfer). We inject
        // these here and fan the same modified body to both the
        // prefill and decode workers concurrently.
        //
        // **Why spawn-and-forget for prefill instead of
        // `tokio::join!`?** All three peer SGLang-HTTP-PD routers
        // (Dynamo / llm-d / aibrix) converged on this shape: the
        // prefill request must outlive the client connection because
        // tying prefill to the client future opens a cancel-race
        // window where the engine's NIXL RPC teardown can leak KV
        // block refs (NVBugs 5969206 in Dynamo). The detached task
        // also keeps the LoadGuard + ActiveLoadGuard alive for the full
        // prefill duration — KV transfer can run for tens of seconds
        // even when the client gave up.
        //
        // No watchdog for fail-fast on prefill 5xx: llm-d / aibrix both
        // ship without one. On prefill failure the client experiences
        // the SGLang decode-side bootstrap_room timeout (~30–60 s by
        // default) instead of an immediate 502. A follow-up can wire a
        // `tokio::sync::watch` channel if telemetry shows it matters.
        //
        // **Scope of the "detached" guarantee.** The spawn protects
        // against client disconnect — the handler future being dropped
        // does NOT cancel the prefill HTTP request. It does NOT protect
        // against router shutdown: when `AppContext` tears down, the
        // tokio runtime cancels all unfinished tasks including this
        // one. A future follow-up could thread a `TaskTracker` /
        // `JoinSet` through `AppContext` for graceful shutdown drain;
        // the current implementation ships without one (matching SMG's
        // shutdown behaviour).
        //
        // `bootstrap_room` is `None` for a symmetric_pd Split (the join key is
        // the `transfer_id` inside the marks).
        let prefill_url = worker.url.clone();
        let prefill_breaker = Arc::clone(&worker.breaker);
        let prefill_headers = headers.clone();
        // The prefill side gets the hint-carrying variant; the decode
        // dispatches below keep the clean `outgoing_body`.
        let prefill_body = executing_body.clone();
        let prefill_proxy = Arc::clone(&ctx.proxy);
        let prefill_holds: (LoadGuard, _) = (guard, active_guard);
        let prefill_ledger = co_ledger.take();
        let mut prefill_cost = cost_reservation.take();
        // `--pd-abort-decode-on-prefill-failure` (the follow-up the comment
        // above anticipates): the prefill task's verdict, shared with the
        // decode leg below, so a failed prefill drops its decode request
        // as sgl-model-gateway does. Off: unchanged (no channel back).
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
        tokio::spawn(async move {
            // The tuple binding extends both guards' lifetime to the
            // end of this async block, which lasts until the prefill
            // HTTP request returns (success / error / engine-side
            // bootstrap_room timeout). The result is logged and
            // swallowed — no channel back to the client. See the big
            // comment above for the rationale.
            let _hold = prefill_holds;
            let legs = PdLegs {
                rid: rid_p.as_deref(),
                bootstrap_room,
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
                    "/v1/chat/completions",
                    &prefill_headers,
                    prefill_body,
                )
                .await;
            if let Some(failure) = &prefill_failure_p {
                if let Some((status, detail)) = PrefillFailure::verdict(&prefill_result) {
                    failure.fail_and_log(status, detail, &legs);
                }
            }
            super::decision_log::settle_prefill_leg(
                prefill_cost.as_mut(),
                prefill_ledger,
                &prefill_result,
            );
            crate::backend::plain_usage::read_prefill(usage_p.as_ref(), &mut prefill_result).await;
            match prefill_result {
                // Cancelled, not failed: the engine finished the P-leg
                // because its reader left; the D-leg owns the outcome.
                ref r if crate::proxy::is_reader_gone(r) => tracing::warn!(
                    prefill_url = %prefill_url,
                    "prefill leg ended: its reader left (symmetric_reader_gone); the decode leg \
                     owns the request",
                ),
                Ok(_) => tracing::debug!(
                    prefill_url = %prefill_url,
                    bootstrap_room = ?bootstrap_room,
                    "prefill side completed",
                ),
                Err(e) => tracing::warn!(
                    prefill_url = %prefill_url,
                    bootstrap_room = ?bootstrap_room,
                    error = %e,
                    "prefill request failed; decode will time out waiting for the KV transfer",
                ),
            }
        });

        // Synchronously await the decode worker. Its response is what
        // the client sees. The decode side gets its own LoadGuard so
        // per-worker `active_requests` reflects decode-pool load for
        // cache-aware-zmq decisions on the decode side.
        // The token charge (token-charged decode policies) is released with
        // the decode LoadGuard: at stream end / client disconnect inside the
        // SSE pump, or when this handler returns or is dropped.
        // `--pd-decode-policy power-of-two-smg`: the SMG request count rides
        // with the decode LoadGuard too (non-streaming: counted from here;
        // streaming: counted from the decode response headers).
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
                "/v1/chat/completions",
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
                "/v1/chat/completions",
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
        // Plain mode, streaming. Both guards ride the SSE pump until
        // the body completes — see the matching comment in the
        // non-streaming arm.
        let stream_guards: Box<dyn Send + 'static> =
            Box::new((guard, active_guard, make_duration_guard()));
        let fetch = ctx.proxy.forward_streaming_to(
            &worker.url,
            &worker.breaker,
            "/v1/chat/completions",
            &headers,
            executing_body,
            Some(stream_guards),
            Some(make_ttft_hook()),
        );
        // Bias `fetch` over the cancellation branch: a successful
        // response that completes in the same poll as the token firing
        // MUST win (returning 504 for a request that already has
        // headers is a correctness regression). The cancellation
        // branch only matters when fetch is still pending — at that
        // point biasing the order is a wash.
        tokio::select! {
            biased;
            r = fetch => r,
            _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
        }
    } else {
        // Plain mode, non-streaming. The handler awaits the full
        // buffered response, so both guards live correctly in this
        // scope. The tuple binding exists only to extend the guards'
        // lifetime to the end of the function — the `forward_json_to`
        // future does not need them (it does not return until the
        // body is buffered).
        let _holds: (LoadGuard, _) = (guard, active_guard);
        let fetch = ctx.proxy.forward_json_to(
            &worker.url,
            &worker.breaker,
            "/v1/chat/completions",
            &headers,
            executing_body,
        );
        // Same `biased` order as the streaming arm.
        let r = tokio::select! {
            biased;
            r = fetch => r,
            _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
        };
        // Non-streaming plain: full generation done — retire the
        // cost-function ledger entry (only completion signal on this path).
        if let Some(l) = co_ledger.take() {
            l.retire();
        }
        r
    };

    // Record the dispatch outcome AFTER we know whether the upstream
    // accepted the request. A 504 from the stale-request branch counts as
    // `cancelled` — semantically distinct from upstream errors that bubble
    // through as `error`. The metric is per-worker so convergence tests
    // can scrape `/metrics` and assert that ≥N requests landed on a
    // single prefill worker.
    let outcome = match &result {
        Ok(_) => RequestOutcome::Success,
        Err(ApiError::StaleRequestExpired { .. }) => {
            // The janitor fired the stale-cancel and we observed it
            // user-side; record both the per-request `cancelled` outcome
            // AND the global `expired` count. The two views are useful for
            // different alerts: per-worker request_total{cancelled} flags a
            // worker that's hanging, while stale_requests_total{expired}
            // tracks the global health of the janitor.
            ctx.metrics
                .record_stale_request(StaleRequestOutcome::Expired);
            RequestOutcome::Cancelled
        }
        Err(_) => RequestOutcome::Error,
    };
    ctx.metrics
        .record_worker_request(&metrics_worker_url, &metrics_model, metrics_mode, outcome);

    // Per-request access log — always on at INFO so incoming traffic and its
    // status are visible without DEBUG. `request_id` is the client/gateway
    // X-Request-Id (echoed end-to-end); `worker` is the engine the policy
    // selected. The cache-aware routing rationale is logged separately at
    // DEBUG by the policy.
    let request_id = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-");
    let http_status = match &result {
        Ok(resp) => resp.status().as_u16(),
        Err(e) => e.status_code().as_u16(),
    };

    // Record end-to-end latency now that the outcome is known. Non-streaming:
    // body is buffered here, so `start.elapsed()` is true e2e — record directly.
    // Streaming: body still pending, so the `RecordDurationOnDrop` guard in
    // `stream_guards` records it at stream completion (not just time-to-headers).
    // `elapsed` still feeds the access-log `latency_ms` for both.
    //
    // HTTP status is counted into `responses_total` by the edge middleware
    // (app.rs), not here — the old per-handler site skipped early exits.
    let elapsed = start.elapsed();
    if !streaming {
        ctx.metrics
            .observe_request_duration(&metrics_model, elapsed.as_secs_f64());
    }
    let outcome_str = match outcome {
        RequestOutcome::Success => "success",
        RequestOutcome::Error => "error",
        RequestOutcome::Cancelled => "cancelled",
    };
    tracing::info!(
        request_id = %request_id,
        method = "POST",
        path = "/v1/chat/completions",
        model = %metrics_model,
        worker = %metrics_worker_url,
        outcome = outcome_str,
        http_status,
        stream = streaming,
        latency_ms = elapsed.as_millis() as u64,
        "chat_completions",
    );

    // Mirror the upstream `x-sgl-decode-url` hint onto the response so
    // external tests / sidecars can observe PD decode affinity without
    // sniffing the proxy hop. The request-side header was set above for
    // the prefill worker; copying it here makes the affinity observable
    // end-to-end. Plain-mode requests skip this (no decode peer was
    // resolved). A malformed URL was already rejected at the
    // request-side parse — we only reach this branch when the URL was
    // header-valid, so the second parse is safe.
    let result = crate::backend::plain_usage::count(result, plain_usage, usage_asked);
    match (result, decode_hint_url) {
        (Ok(mut response), Some(url)) => {
            match HeaderValue::from_str(&url) {
                Ok(v) => {
                    response.headers_mut().insert(X_SGL_DECODE_URL, v);
                }
                Err(e) => {
                    // Already-validated upstream; defensive log only.
                    tracing::warn!(
                        decode_url = %url,
                        error = %e,
                        "decode worker URL rejected by header parser on response; omitting response-side hint",
                    );
                }
            }
            Ok(response)
        }
        (other, _) => other,
    }
}

fn resolve_prefix_query(
    result: Result<sgl_kv_indexer::PrefixOutcome, sgl_kv_indexer::PrefixIndexError>,
    model: &str,
) -> Result<sgl_kv_indexer::PrefixOutcome, ApiError> {
    use sgl_kv_indexer::PrefixIndexError;
    match result {
        Ok(outcome) => Ok(outcome),
        // The prefix hit only improves worker choice, so an indexer that is
        // shedding, slow, or down costs cache affinity — not availability.
        Err(
            error @ (PrefixIndexError::Overloaded
            | PrefixIndexError::Timeout
            | PrefixIndexError::Unreachable),
        ) => {
            tracing::warn!(%model, error = %error, "KV Indexer unavailable; falling back to min-load routing");
            Ok(sgl_kv_indexer::PrefixOutcome::Empty)
        }
        // A prompt too long to fit one gRPC message is still a prompt a worker
        // can serve, so it costs cache affinity like the cases above. Logged
        // separately because the remedy is operational — raise the indexer's
        // message limit — rather than waiting for the indexer to recover.
        Err(error @ PrefixIndexError::QueryTooLarge) => {
            tracing::warn!(%model, error = %error, "prompt exceeds the KV Indexer query size limit; falling back to min-load routing");
            Ok(sgl_kv_indexer::PrefixOutcome::Empty)
        }
        // A rejection means the router and the indexer disagree on the request
        // contract; degrading would hide that from every request.
        Err(error) => {
            tracing::warn!(%model, error = %error, "KV Indexer rejected the query");
            Err(ApiError::PolicySelectionFailed {
                model: model.to_string(),
            })
        }
    }
}

/// Estimate prefill-token count from the raw request body for use as
/// the active-load `prefill_load` counter. Returns 1 at minimum so
/// a registered request always shows up as "load > 0" — under-counting
/// to zero would hide the request from the cache-aware policy's
/// load-imbalance fast-path.
///
/// This is a coarse approximation: we count the body length in bytes
/// and divide by [`CHARS_PER_TOKEN_ESTIMATE`]. A future improvement is
/// to thread the tokenizer's actual token count through (the
/// cache-aware-zmq policy already tokenizes the prompt for tree
/// matching — that count could be reused here).
pub(crate) fn estimate_prefill_tokens(body: &Bytes) -> usize {
    (body.len() / CHARS_PER_TOKEN_ESTIMATE).max(1)
}

/// Mint a fresh `bootstrap_room` for a PD-disagg request.
///
/// SGLang's disagg-prefill stores the room as a signed `i64` internally
/// (see `python/sglang/srt/disaggregation/utils.py` — `bootstrap_room`
/// metadata buffer is allocated as `torch.int64`). Generating in
/// `[0, i64::MAX]` keeps the value safely positive when reinterpreted
/// signed. Mirrors SMG's `pd_types::generate_room_id`, Dynamo's
/// `rand::random_range(0..=i64::MAX.cast_unsigned())`, and SGLang's
/// own Python-side `random.randint(0, 2**63 - 1)`.
pub(crate) fn generate_room_id() -> u64 {
    rand::random::<u64>() & (i64::MAX as u64)
}

/// PD-disagg bootstrap fields injected into the body forwarded to both the
/// prefill and decode workers. SGLang's HTTP disagg-prefill validator
/// requires all three as flat top-level fields:
///
/// * `host` → `bootstrap_host` — the prefill worker's hostname; decode
///   connects here for the KV transfer.
/// * `port` → `bootstrap_port` — the prefill worker's bootstrap-server port
///   (`null` when the worker is misconfigured; the engine rejects with a
///   clear error). Emitted as JSON `null`, not omitted — SGLang's validator
///   distinguishes missing from null.
/// * `room` → `bootstrap_room` — a per-request 63-bit `u64` identifying this
///   request on both prefill and decode sides.
pub(crate) struct BootstrapFields {
    pub(crate) host: String,
    pub(crate) port: Option<u16>,
    pub(crate) room: u64,
}

/// Build the body forwarded to the engine, injecting (when present) the
/// precomputed `input_ids` and/or the PD `bootstrap_*` fields into the
/// already-parsed request object and serializing once. When neither is
/// needed, returns the original bytes unchanged (no re-serialize).
///
/// `input_ids`: the router-computed prompt tokens. When set, the engine skips
/// its own chat-template tokenization; `messages` are retained in the body so
/// the engine still derives stop tokens / tool-call constraint and the OpenAI
/// response shape. The caller sets this only when the tokens are
/// engine-equivalent and `input_ids_safe_to_forward` held.
///
/// `value` is the already-parsed request body when one is on hand (the
/// cache-aware path parses once at ingress); it is consumed so the mutation
/// reuses that parse. It is `None` only for a load-only policy in PD mode — a
/// path that never parses at ingress — so the bootstrap injection re-parses
/// the bytes here (matching the pre-refactor behavior). The body shape was
/// validated by `parse_probe`; the non-object arm defends against a TOCTOU
/// regression rather than panicking.
pub(crate) fn build_outgoing_body(
    body: &Bytes,
    value: Option<serde_json::Value>,
    input_ids: Option<&[u32]>,
    bootstrap: Option<&BootstrapFields>,
    kv_pull_hint: Option<&serde_json::Value>,
) -> Result<Bytes, ApiError> {
    if input_ids.is_none() && bootstrap.is_none() && kv_pull_hint.is_none() {
        // Nothing to inject — forward the original bytes (cheap Arc clone).
        return Ok(body.clone());
    }
    let parsed = match value {
        Some(v) => v,
        // Load-only + PD: the ingress skipped the parse, so re-parse for the
        // bootstrap injection (input_ids is never set on this path).
        None => serde_json::from_slice(body).map_err(|_| {
            ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
        })?,
    };
    let mut obj = match parsed {
        serde_json::Value::Object(map) => map,
        _ => {
            return Err(ApiError::BadRequest(
                "invalid request: body must be a JSON object".to_string(),
            ))
        }
    };
    if let Some(ids) = input_ids {
        obj.insert(
            "input_ids".to_string(),
            serde_json::Value::Array(
                ids.iter()
                    .map(|&i| serde_json::Value::Number(i.into()))
                    .collect(),
            ),
        );
    }
    if let Some(b) = bootstrap {
        obj.insert(
            "bootstrap_host".to_string(),
            serde_json::Value::String(b.host.clone()),
        );
        obj.insert(
            "bootstrap_port".to_string(),
            match b.port {
                Some(p) => serde_json::Value::Number(p.into()),
                None => serde_json::Value::Null,
            },
        );
        obj.insert(
            "bootstrap_room".to_string(),
            serde_json::Value::Number(b.room.into()),
        );
    }
    if let Some(hint) = kv_pull_hint {
        obj.insert("kv_pull_hint".to_string(), hint.clone());
    }
    let bytes = serde_json::to_vec(&obj).map_err(|e| {
        ApiError::Internal(anyhow::Error::new(e).context("re-serialize injected request body"))
    })?;
    Ok(Bytes::from(bytes))
}

/// The token ids of chat request `value` for placement. The router's chat
/// encoder renders the messages alone; with `--chat-tools-tokenize`, the ids
/// of a request with tools come from an engine instead (placement only,
/// never forwarded). `workers` are the engines outside symmetric_pd.
pub(crate) async fn chat_request_tokens(
    ctx: &AppContext,
    model_id: &ModelId,
    value: &serde_json::Value,
    workers: &[Arc<Worker>],
) -> Option<RequestTokens> {
    let tokens = request_tokens_for(&ctx.tokenizers, model_id, value);
    if !ctx.config.proxy.chat_tools_tokenize
        || !request_has_tools(value)
        || request_is_multimodal(value)
    {
        return tokens;
    }
    let engines = match ctx.symmetric_pd.as_ref() {
        Some(spd) => spd.role_members_all(),
        None => workers.to_vec(),
    };
    match crate::engine_tokenize::chat_ids(&engines, value).await {
        Some(ids) => Some(RequestTokens {
            ids,
            engine_equivalent: false,
        }),
        None => tokens,
    }
}

/// Whether the router's `input_ids` may be forwarded for this request.
///
/// We forward only when the engine, fed `input_ids`, would have produced the
/// SAME prompt the router tokenized. When `input_ids` is present the engine
/// uses it verbatim and ignores everything that would otherwise steer its
/// `messages`-side tokenization (only stop tokens / tool-call constraint are
/// still taken from `messages`). So any request field that changes that
/// tokenization but which the router's chat encoder does not replicate makes
/// the forwarded ids wrong. This predicate is conservative by construction —
/// any such signal returns `false` and the engine tokenizes from `messages`
/// (always correct).
///
/// Replicated-and-safe: plain text `messages` with a string `content`.
/// Not replicated → omit:
///   * `tools` / `functions` — the encoder doesn't render tool schemas.
///   * multimodal (array) `content` — a text tokenizer can't represent images.
///   * `chat_template` — an OpenAI-compatible per-request template override
///     (e.g. vLLM); the router renders with the model's default template, so a
///     custom one would diverge. (SGLang ignores it today, but block it so the
///     offload stays correct across engines / future versions.)
///   * `chat_template_kwargs` (carries `enable_thinking`/`thinking`),
///     `reasoning` / `reasoning_effort`, `task` — thinking/mode toggles the
///     encoder renders in the engine's default mode only.
///   * `continue_final_message: true`, or a trailing `assistant` message — the
///     engine rewrites/strips the final assistant turn; the encoder renders it
///     verbatim.
///
/// NOTE: the router's chat encoder renders in the engine's default
/// (non-thinking) mode. Current sglang derives thinking from the request
/// (`chat_template_kwargs`), which this guard already omits, so a plain request
/// the router rendered matches the engine. The only way to diverge is an engine
/// build that applies a non-default thinking mode the router can't observe from
/// the request — the same router↔engine tokenization-parity assumption that
/// cache-aware routing already depends on. The same assumption covers
/// `add_special_tokens`: the router renders specials via the chat template, which
/// matches the engine on tokenizers that auto-add them (the common case); a
/// tokenizer that does not would diverge by a leading special, again undetectable
/// from the request.
pub(crate) fn input_ids_safe_to_forward(value: &serde_json::Value) -> bool {
    if request_has_tools(value) || request_is_multimodal(value) {
        return false;
    }
    // Fields that steer the engine's template tokenization but which the
    // router's encoder does not thread through.
    for key in [
        "chat_template",
        "chat_template_kwargs",
        "reasoning",
        "reasoning_effort",
        "task",
    ] {
        if value.get(key).is_some_and(|v| !v.is_null()) {
            return false;
        }
    }
    if value
        .get("continue_final_message")
        .and_then(|v| v.as_bool())
        == Some(true)
    {
        return false;
    }
    !last_message_is_assistant(value)
}

/// Whether the ingress tokenization offload was expected to fire but failed —
/// the condition behind `sgl_router_ingress_tokenize_errors_total`.
///
/// True only when ALL of:
///   * the model has a chat encoder (`has_chat_encoder`), so a chat request
///     on it SHOULD have produced engine-equivalent ids;
///   * the request is a chat request (`messages` array present);
///   * the tokens are absent OR not engine-equivalent — i.e. `encode_chat`
///     render/encode failed and the request silently fell back to engine-side
///     tokenization.
///
/// Non-chat-encoder / non-`messages` requests never expected the offload, so
/// they are not failures. A tools / multimodal / thinking request on a
/// chat-encoder model still gets engine-equivalent ids (`encode_chat`
/// succeeded; the safe-predicate withholds forwarding for other reasons), so it
/// is an expected omission, not a failure.
fn ingress_tokenize_offload_failed(
    has_chat_encoder: bool,
    request_value: Option<&serde_json::Value>,
    request_tokens: Option<&RequestTokens>,
) -> bool {
    if !has_chat_encoder {
        return false;
    }
    let chat_request =
        request_value.is_some_and(|v| v.get("messages").is_some_and(|m| m.is_array()));
    if !chat_request {
        return false;
    }
    !request_tokens.is_some_and(|t| t.engine_equivalent)
}

/// Whether the final chat message has `role: "assistant"` (a prefix /
/// continuation turn the engine's template path special-cases).
fn last_message_is_assistant(value: &serde_json::Value) -> bool {
    value
        .get("messages")
        .and_then(|m| m.as_array())
        .and_then(|msgs| msgs.last())
        .and_then(|m| m.get("role"))
        .and_then(|r| r.as_str())
        == Some("assistant")
}

/// Whether the request carries tool / function definitions. The router's chat
/// encoder renders only `messages`, so its `input_ids` would omit the tool
/// schemas the engine's template injects into the prompt — the caller must let
/// the engine tokenize these itself.
fn request_has_tools(value: &serde_json::Value) -> bool {
    let nonempty = |key: &str| {
        value.get(key).is_some_and(|v| match v {
            serde_json::Value::Array(a) => !a.is_empty(),
            serde_json::Value::Null => false,
            _ => true,
        })
    };
    nonempty("tools") || nonempty("functions")
}

/// Whether any message carries non-string (array / multimodal) content. A text
/// tokenizer cannot represent image content, so the router's `input_ids` would
/// drop it — the caller must let the engine handle these requests.
fn request_is_multimodal(value: &serde_json::Value) -> bool {
    value
        .get("messages")
        .and_then(|m| m.as_array())
        .is_some_and(|msgs| {
            msgs.iter()
                .any(|m| matches!(m.get("content"), Some(serde_json::Value::Array(_))))
        })
}

pub(crate) fn parse_probe(body: &Bytes) -> Result<RequestProbe, ApiError> {
    // We deliberately do NOT echo the serde error into the client-visible
    // message — that risks leaking field-level detail and is also of little
    // help to a real client (which already has its own JSON validator).
    // Server-side, the full error is logged with `tracing::debug!` for
    // operator triage.
    //
    // Two-step deserialize:
    //   1. `Map<String, IgnoredAny>` *anchors* the shape to a JSON object.
    //      This rejects `null` / `[]` / `"hi"` (all valid JSON but not
    //      request shape) without walking the full value into a
    //      `serde_json::Value` per field.
    //   2. `RequestProbe` (struct of `Option<bool>` + `Option<String>`)
    //      lifts out only the fields we care about — `stream` and `model`.
    //      Other fields are ignored; the worker is authoritative for the
    //      rest of the schema.
    let _: HashMap<String, IgnoredAny> = serde_json::from_slice(body).map_err(|e| {
        tracing::debug!(error = %e, "chat-completions body rejected as non-object JSON");
        ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
    })?;
    let probe: RequestProbe = serde_json::from_slice(body).map_err(|e| {
        tracing::debug!(error = %e, "chat-completions request-probe deserialize failed");
        ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
    })?;
    Ok(probe)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An unavailable indexer must never fail a request that min-load routing
    /// can still serve. `QueryTooLarge` belongs here too: a prompt that outgrows
    /// the query's message limit loses cache affinity, not availability.
    #[test]
    fn unavailable_indexer_degrades_to_empty_prefix_signal() {
        for error in [
            sgl_kv_indexer::PrefixIndexError::Overloaded,
            sgl_kv_indexer::PrefixIndexError::Timeout,
            sgl_kv_indexer::PrefixIndexError::Unreachable,
            sgl_kv_indexer::PrefixIndexError::QueryTooLarge,
        ] {
            assert_eq!(
                resolve_prefix_query(Err(error.clone()), "tiny").unwrap(),
                sgl_kv_indexer::PrefixOutcome::Empty,
                "{error} should degrade"
            );
        }
    }

    #[test]
    fn rejected_indexer_query_still_fails_selection() {
        assert!(matches!(
            resolve_prefix_query(
                Err(sgl_kv_indexer::PrefixIndexError::Rejected(
                    sgl_kv_indexer::RpcCode::InvalidArgument
                )),
                "tiny"
            ),
            Err(ApiError::PolicySelectionFailed { .. })
        ));
    }

    /// `generate_room_id` MUST return values in `[0, i64::MAX]`. The
    /// SGLang prefill stores `bootstrap_room` as `torch.int64`; a u64
    /// with the top bit set would wrap negative on the engine side.
    /// Sample many times to defend against future refactors of the
    /// mask (e.g. someone "simplifying" to plain `rand::random::<u64>()`).
    #[test]
    fn generate_room_id_stays_in_63_bit_range() {
        for _ in 0..10_000 {
            let r = generate_room_id();
            assert!(
                r <= i64::MAX as u64,
                "generate_room_id() returned {r} > i64::MAX; would wrap negative as torch.int64",
            );
        }
    }

    /// When the prefill worker has no `bootstrap_port` configured
    /// (a misconfiguration the engine will reject loudly), the
    /// injected field MUST be JSON `null` — not omitted, not 0.
    /// SGLang's validator distinguishes "missing field" from
    /// "null field" in some code paths.
    #[test]
    fn build_outgoing_body_emits_null_for_missing_port() {
        let body = Bytes::from_static(br#"{"model":"x","messages":[]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let bootstrap = BootstrapFields {
            host: "host".into(),
            port: None,
            room: 42,
        };
        let injected =
            build_outgoing_body(&body, Some(value), None, Some(&bootstrap), None).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&injected).unwrap();
        assert_eq!(parsed.get("bootstrap_port"), Some(&serde_json::Value::Null));
        assert_eq!(
            parsed.get("bootstrap_host"),
            Some(&serde_json::Value::String("host".into()))
        );
        assert_eq!(
            parsed.get("bootstrap_room"),
            Some(&serde_json::Value::Number(42.into()))
        );
    }

    /// `input_ids` are injected and `messages` retained (the engine still
    /// needs them for stop tokens / tool-call constraint / response shape).
    #[test]
    fn build_outgoing_body_injects_input_ids_and_keeps_messages() {
        let body =
            Bytes::from_static(br#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let ids = [1u32, 2, 3];
        let out = build_outgoing_body(&body, Some(value), Some(&ids), None, None).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(parsed.get("input_ids"), Some(&serde_json::json!([1, 2, 3])));
        assert!(
            parsed.get("messages").is_some(),
            "messages must be retained alongside input_ids"
        );
    }

    /// With nothing to inject, the original bytes are forwarded unchanged
    /// (no re-serialize) — the transparent no-op fallback.
    #[test]
    fn build_outgoing_body_no_injection_returns_original_bytes() {
        let body = Bytes::from_static(br#"{"model":"x","messages":[]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let out = build_outgoing_body(&body, Some(value), None, None, None).unwrap();
        assert_eq!(
            out, body,
            "no injection must forward the original bytes unchanged"
        );
    }

    /// PD + forwarding: both `input_ids` and the bootstrap fields land in one
    /// serialized body.
    #[test]
    fn build_outgoing_body_injects_both_input_ids_and_bootstrap() {
        let body =
            Bytes::from_static(br#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let ids = [7u32, 8];
        let bootstrap = BootstrapFields {
            host: "h".into(),
            port: Some(9),
            room: 5,
        };
        let out =
            build_outgoing_body(&body, Some(value), Some(&ids), Some(&bootstrap), None).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(parsed.get("input_ids"), Some(&serde_json::json!([7, 8])));
        assert_eq!(
            parsed.get("bootstrap_room"),
            Some(&serde_json::Value::Number(5.into()))
        );
        assert_eq!(
            parsed.get("bootstrap_port"),
            Some(&serde_json::Value::Number(9.into()))
        );
    }

    /// Tool / function requests are detected so the caller omits `input_ids`
    /// (the router's encoder doesn't render tools).
    #[test]
    fn request_has_tools_detects_tools_and_functions() {
        assert!(request_has_tools(
            &serde_json::json!({"tools":[{"type":"function"}]})
        ));
        assert!(request_has_tools(
            &serde_json::json!({"functions":[{"name":"f"}]})
        ));
        assert!(!request_has_tools(&serde_json::json!({"tools":[]})));
        assert!(!request_has_tools(&serde_json::json!({"messages":[]})));
    }

    /// Array (multimodal) message content is detected so the caller omits
    /// `input_ids` (a text tokenizer can't represent image content).
    #[test]
    fn request_is_multimodal_detects_array_content() {
        assert!(request_is_multimodal(&serde_json::json!({
            "messages":[{"role":"user","content":[{"type":"image_url","image_url":"x"}]}]
        })));
        assert!(!request_is_multimodal(&serde_json::json!({
            "messages":[{"role":"user","content":"hello"}]
        })));
    }

    /// Plain text chat with nothing unreplicated → input_ids may be forwarded.
    #[test]
    fn input_ids_safe_to_forward_allows_plain_text_chat() {
        assert!(input_ids_safe_to_forward(&serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}]
        })));
    }

    /// Every field the engine honors on the `messages` path but which the
    /// router's encoder does not replicate must block forwarding — otherwise
    /// the engine uses the router's ids verbatim and silently runs a different
    /// prompt than the request asked for.
    #[test]
    fn input_ids_safe_to_forward_blocks_unreplicated_signals() {
        let blockers = [
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function"}]}),
            serde_json::json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":"x"}]}]}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"chat_template":"{{ custom }}"}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":true}}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"reasoning":{"enabled":true}}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"task":"generate"}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"continue_final_message":true}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"partial"}]}),
        ];
        for b in blockers {
            assert!(
                !input_ids_safe_to_forward(&b),
                "must NOT forward input_ids for: {b}"
            );
        }
    }

    /// Null / false-valued fields do not block (absent ≡ null ≡ default).
    #[test]
    fn input_ids_safe_to_forward_ignores_null_and_false_fields() {
        assert!(input_ids_safe_to_forward(&serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template": null,
            "reasoning_effort": null,
            "chat_template_kwargs": null,
            "continue_final_message": false
        })));
    }

    /// Load-only + PD: `build_outgoing_body` is handed `None` for the value
    /// (the ingress skipped the parse for a load-only policy) and re-parses the
    /// bytes to inject the bootstrap fields. `input_ids` is never set here.
    #[test]
    fn build_outgoing_body_reparses_when_value_absent() {
        let body = Bytes::from_static(br#"{"model":"x","messages":[]}"#);
        let bootstrap = BootstrapFields {
            host: "h".into(),
            port: Some(1),
            room: 2,
        };
        let out = build_outgoing_body(&body, None, None, Some(&bootstrap), None).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(
            parsed.get("bootstrap_room"),
            Some(&serde_json::Value::Number(2.into()))
        );
        assert!(parsed.get("input_ids").is_none());
    }

    /// A chat request on a chat-encoder model that yields engine-equivalent
    /// ids (encode succeeded) is NOT a failure — the offload worked.
    #[test]
    fn offload_failed_false_when_tokens_engine_equivalent() {
        let value = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        let tokens = RequestTokens {
            ids: vec![1, 2, 3],
            engine_equivalent: true,
        };
        assert!(!ingress_tokenize_offload_failed(
            true,
            Some(&value),
            Some(&tokens)
        ));
    }

    /// A chat request on a chat-encoder model whose tokenization yielded NO
    /// tokens (encode_chat returned None → request_tokens None) IS a failure:
    /// the encoder should have fired but didn't.
    #[test]
    fn offload_failed_true_when_chat_encoder_request_has_no_tokens() {
        let value = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        assert!(ingress_tokenize_offload_failed(true, Some(&value), None));
    }

    /// Encode produced ids but NOT via the chat encoder (raw fallback,
    /// `engine_equivalent = false`) on a chat-encoder model + chat request →
    /// the chat-encode render/encode failed and fell through to the raw path.
    #[test]
    fn offload_failed_true_when_tokens_not_engine_equivalent() {
        let value = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        let tokens = RequestTokens {
            ids: vec![1, 2, 3],
            engine_equivalent: false,
        };
        assert!(ingress_tokenize_offload_failed(
            true,
            Some(&value),
            Some(&tokens)
        ));
    }

    /// Non-chat-encoder models never expected the offload → not a failure even
    /// with no tokens.
    #[test]
    fn offload_failed_false_without_chat_encoder() {
        let value = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        assert!(!ingress_tokenize_offload_failed(false, Some(&value), None));
    }

    /// A non-chat (no `messages`) request on a chat-encoder model — e.g.
    /// `/v1/completions` `prompt` — never expected the chat-encode offload, so
    /// the absence of engine-equivalent ids is not a failure.
    #[test]
    fn offload_failed_false_for_non_messages_request() {
        let value = serde_json::json!({"prompt":"hi"});
        assert!(!ingress_tokenize_offload_failed(true, Some(&value), None));
    }

    #[test]
    fn parse_probe_reads_stream_bool_from_object() {
        let b = Bytes::from_static(br#"{"stream": true, "model": "tiny"}"#);
        assert_eq!(parse_probe(&b).unwrap().stream, Some(true));
        let b = Bytes::from_static(br#"{"stream": false, "model": "tiny"}"#);
        assert_eq!(parse_probe(&b).unwrap().stream, Some(false));
    }

    #[test]
    fn parse_probe_defaults_when_stream_absent() {
        // Existing happy-path contract: well-formed object missing `stream`
        // must default to None (caller picks false). The minimal `RequestProbe`
        // (Option<bool> + #[serde(default)]) must NOT break this.
        let b = Bytes::from_static(br#"{"model": "tiny", "messages": []}"#);
        let p = parse_probe(&b).unwrap();
        assert_eq!(p.stream, None);
        assert_eq!(p.model.as_deref(), Some("tiny"));
    }

    #[test]
    fn parse_probe_rejects_non_object_shapes() {
        // Pin the contract: degenerate JSON (valid JSON but wrong shape)
        // must be rejected, not silently forwarded with `stream=false`.
        for bad in [&b"null"[..], &b"[]"[..], &b"\"hi\""[..], &b"42"[..]] {
            let b = Bytes::copy_from_slice(bad);
            let err = parse_probe(&b).unwrap_err();
            match err {
                ApiError::BadRequest(_) => {}
                other => panic!("expected BadRequest for {bad:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn parse_probe_rejects_malformed_json() {
        let b = Bytes::from_static(b"{not json}");
        let err = parse_probe(&b).unwrap_err();
        assert!(matches!(err, ApiError::BadRequest(_)));
    }

    #[test]
    fn parse_probe_handles_nested_messages_with_stream_true() {
        // Well-formed object with nested arrays/objects (real chat-completions
        // payloads carry `messages: [{role, content: [{type, text}]}]`). The
        // two-step deserialize must not balk on this — only the top-level
        // object shape and the `stream`/`model` fields matter.
        let b = Bytes::from_static(
            br#"{
              "model": "x",
              "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
              "stream": true
            }"#,
        );
        assert_eq!(parse_probe(&b).unwrap().stream, Some(true));
    }

    #[test]
    fn parse_probe_handles_nested_messages_with_stream_false() {
        let b = Bytes::from_static(
            br#"{
              "model": "x",
              "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
              "stream": false
            }"#,
        );
        assert_eq!(parse_probe(&b).unwrap().stream, Some(false));
    }

    #[test]
    fn parse_probe_handles_duplicate_stream_keys() {
        // RFC 8259 says "names within an object SHOULD be unique" but a
        // parser MAY accept duplicates. Step 1 (HashMap) silently
        // last-wins, but step 2 deserializes into the typed `RequestProbe`
        // struct, and `serde_json`'s `#[derive(Deserialize)]` REJECTS
        // duplicate fields with a `duplicate field` error.
        //
        // We map that to `BadRequest` (same path as other malformed input).
        // Pinning "reject" rather than "last-wins" is intentional —
        // ambiguous bodies should fail loudly at the edge, not silently
        // route based on which copy serde happened to see last.
        let b = Bytes::from_static(br#"{"stream": true, "stream": false}"#);
        let err = parse_probe(&b).unwrap_err();
        match err {
            ApiError::BadRequest(_) => {}
            other => panic!("expected BadRequest on duplicate `stream` key, got {other:?}"),
        }
    }

    #[test]
    fn parse_probe_bad_request_message_does_not_leak_serde_detail() {
        // Info-leak guard: the client-visible message must be a fixed
        // string, not the serde error (which can contain line/column
        // detail or hint at field shape).
        let b = Bytes::from_static(br#"{"stream": "not-a-bool"}"#);
        let err = parse_probe(&b).unwrap_err();
        match err {
            ApiError::BadRequest(msg) => assert_eq!(
                msg, "invalid request: body must be a JSON object",
                "client-visible message must be fixed; got: {msg}"
            ),
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }
}
