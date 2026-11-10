// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use thiserror::Error;

pub const X_ROUTER_ERROR_CODE: HeaderName = HeaderName::from_static("x-router-error-code");

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("bad request: {0}")]
    BadRequest(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// Could not reach the upstream worker (connect refused, DNS, TLS, request
    /// build error). `source` captures the full anyhow chain for server-side
    /// logging; clients see a generic message.
    ///
    /// `worker` is the typed `reqwest::Url` so we don't re-stringify a value
    /// that is already a `Url` at the construction site. Rendering goes
    /// through `Display`, which produces the same canonical form as
    /// `Url::as_str()`.
    #[error("upstream unreachable: worker {worker}")]
    UpstreamUnreachable {
        worker: reqwest::Url,
        #[source]
        source: anyhow::Error,
    },

    /// The worker started a response (status + headers received) but failed
    /// to deliver the full body — mid-body socket drop, framing error, etc.
    /// Distinct from `UpstreamUnreachable` (no reply at all) and from a
    /// well-formed non-2xx (which `Proxy` forwards verbatim with the worker's
    /// own body).
    #[error("upstream returned status {status}")]
    UpstreamStatus { status: StatusCode },

    /// Wall-clock timeout exceeded while waiting for the upstream worker's
    /// response (per-request `request_timeout`).
    ///
    /// `worker` is the typed `reqwest::Url` for the same reason as
    /// `UpstreamUnreachable`.
    #[error("upstream timed out: worker {worker}")]
    UpstreamTimeout { worker: reqwest::Url },

    /// PD mode with `--pd-abort-decode-on-prefill-failure`: the request's
    /// prefill leg failed (transport error or non-2xx), so the router
    /// dropped its decode leg. `status` follows sgl-model-gateway: 502 for
    /// a transport error; a prefill 400 / 404 / 500 / 502 / 503 passes
    /// through, any other non-2xx becomes 500. `detail` is logged, never
    /// sent to the client (it can carry worker URLs).
    #[error("prefill failed ({status}): {detail}")]
    PrefillFailed { status: StatusCode, detail: String },

    /// No healthy worker is available for `model`: either none were ever
    /// registered, or every candidate's circuit breaker is open.  Clients
    /// should retry; operators should check discovery + worker health.
    #[error("no healthy workers for model {model}")]
    NoHealthyWorkers { model: String },

    /// PD-mode deployment whose prefill pool has zero healthy workers.
    /// Distinct from `NoHealthyWorkers` because the decode pool may
    /// still be healthy — the failure is pool-specific, and surfacing
    /// the distinct code lets operators alert on prefill-fleet outages
    /// independently of full-model outages.
    #[error("no prefill workers available for model {model}")]
    NoPrefillWorkersAvailable { model: String },

    /// PD-mode deployment whose decode pool has zero healthy workers.
    /// Mirror of [`Self::NoPrefillWorkersAvailable`].
    #[error("no decode workers available for model {model}")]
    NoDecodeWorkersAvailable { model: String },

    /// A request whose lifetime exceeded `stale_request_timeout` — the
    /// active-load janitor force-expired the in-flight bookkeeping
    /// AND fired the per-request cancellation token, which the chat
    /// handler `select!`-races against the upstream fetch.  When the
    /// token wins, the handler returns this variant → HTTP 504 →
    /// client sees `stale_request_expired`.
    ///
    /// Mapped to 504 (not 503) because the failure is a router-side
    /// gateway timeout from the client's perspective: the upstream
    /// worker is still potentially fine, the router gave up because
    /// the per-request budget elapsed.
    #[error("stale request expired for model {model}")]
    StaleRequestExpired { model: String },

    /// The per-model policy returned `None` despite the candidate set
    /// being non-empty.  Almost always a router bug or an unsupported
    /// policy state; surfaced as 503 (not 500) so retry-on-failure clients
    /// can drain through a rotation rather than fail-fast on internal_error.
    #[error("policy selected no worker for model {model}")]
    PolicySelectionFailed { model: String },

    /// The worker's circuit breaker was open at the moment of dispatch.
    /// Surfaced post-policy-selection (race with `healthy_workers_for`);
    /// the next selection will skip this worker.
    #[error("worker circuit breaker open: {worker}")]
    BreakerOpen { worker: String },

    /// The worker URL emitted by discovery failed to parse.  Always a
    /// config / discovery-backend bug, not a transient infra issue — but
    /// from the client's perspective the worker is unreachable, so 503.
    /// The forwarder trips the circuit breaker before returning so the
    /// malformed worker drops out of subsequent selection.
    #[error("worker misconfigured: {worker}")]
    WorkerMisconfigured {
        worker: String,
        #[source]
        source: anyhow::Error,
    },

    /// `POST /symmetric_pd/place` on a router not started with `--policy
    /// symmetric_pd` (contract: 404 `symmetric_pd_disabled`).
    #[error("symmetric_pd placement is not enabled on this router")]
    SymmetricPdDisabled,

    /// No `(p, d)` pair could be placed (no available PD worker, no
    /// prefill-capable worker with weight > 0, `force_split` with one
    /// worker). 503 `symmetric_pd_no_placement`.
    #[error("{0}")]
    SymmetricPdNoPlacement(String),

    /// Split chosen but a leg has neither a `/server_info` `symmetric_pd`
    /// section nor a `bootstrap_port` to derive the ZMQ endpoints from.
    /// 500 `symmetric_pd_endpoints`.
    #[error("{0}")]
    SymmetricPdEndpoints(String),

    /// `--pd-late-decode choose` serves only `/v1/completions`: on another
    /// route a released or failed candidate would reach the client as a
    /// 2xx stream. 501
    /// `late_d_route_unsupported`.
    #[error("late decode (choose) serves only /v1/completions")]
    LateDRouteUnsupported,

    /// `--pd-late-decode choose` binds one request at a time; a batch
    /// prompt is refused. 400 `late_d_batch_unsupported`.
    #[error(
        "late decode (choose) binds one request at a time; send batch prompts as separate requests"
    )]
    LateDBatchUnsupported,

    /// Late D choose: every decode candidate was released (HTTP 409) or no
    /// candidate answered 2xx, and no candidate failure explains it. 503
    /// `late_d_no_winner`. `detail` is logged, never sent to the client.
    #[error("late decode: no decode candidate bound the request ({detail})")]
    LateDNoWinner { detail: String },

    #[error("internal: {0}")]
    Internal(#[from] anyhow::Error),
}

impl ApiError {
    fn status_and_code(&self) -> (StatusCode, &'static str) {
        match self {
            ApiError::BadRequest(_) => (StatusCode::BAD_REQUEST, "bad_request"),
            ApiError::ModelNotFound(_) => (StatusCode::NOT_FOUND, "model_not_found"),
            ApiError::UpstreamUnreachable { .. } => {
                (StatusCode::BAD_GATEWAY, "upstream_unreachable")
            }
            ApiError::UpstreamStatus { .. } => (StatusCode::BAD_GATEWAY, "upstream_status"),
            ApiError::UpstreamTimeout { .. } => (StatusCode::BAD_GATEWAY, "upstream_timeout"),
            ApiError::PrefillFailed { status, .. } => (*status, "prefill_failed"),
            ApiError::NoHealthyWorkers { .. } => {
                (StatusCode::SERVICE_UNAVAILABLE, "no_healthy_workers")
            }
            ApiError::NoPrefillWorkersAvailable { .. } => (
                StatusCode::SERVICE_UNAVAILABLE,
                "no_prefill_workers_available",
            ),
            ApiError::NoDecodeWorkersAvailable { .. } => (
                StatusCode::SERVICE_UNAVAILABLE,
                "no_decode_workers_available",
            ),
            ApiError::StaleRequestExpired { .. } => {
                (StatusCode::GATEWAY_TIMEOUT, "stale_request_expired")
            }
            ApiError::PolicySelectionFailed { .. } => {
                (StatusCode::SERVICE_UNAVAILABLE, "policy_selection_failed")
            }
            ApiError::BreakerOpen { .. } => (StatusCode::SERVICE_UNAVAILABLE, "breaker_open"),
            ApiError::WorkerMisconfigured { .. } => {
                (StatusCode::SERVICE_UNAVAILABLE, "worker_misconfigured")
            }
            ApiError::SymmetricPdDisabled => (StatusCode::NOT_FOUND, "symmetric_pd_disabled"),
            ApiError::SymmetricPdNoPlacement(_) => {
                (StatusCode::SERVICE_UNAVAILABLE, "symmetric_pd_no_placement")
            }
            ApiError::SymmetricPdEndpoints(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "symmetric_pd_endpoints")
            }
            ApiError::LateDRouteUnsupported => {
                (StatusCode::NOT_IMPLEMENTED, "late_d_route_unsupported")
            }
            ApiError::LateDBatchUnsupported => {
                (StatusCode::BAD_REQUEST, "late_d_batch_unsupported")
            }
            ApiError::LateDNoWinner { .. } => (StatusCode::SERVICE_UNAVAILABLE, "late_d_no_winner"),
            ApiError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
        }
    }

    /// The HTTP status this error maps to — same value the client receives via
    /// `into_response`. Exposed so the access log records the real status
    /// (e.g. 502/503/504) instead of a sentinel.
    pub fn status_code(&self) -> StatusCode {
        self.status_and_code().0
    }
}

#[derive(Serialize)]
struct ErrorEnvelope<'a> {
    error: ErrorBody<'a>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    #[serde(rename = "type")]
    typ: &'static str,
    code: &'a str,
    message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code) = self.status_and_code();
        let typ = match status.as_u16() {
            400..=499 => "invalid_request_error",
            _ => "server_error",
        };
        // Pick a client-facing message that NEVER leaks worker URLs or raw
        // source chains; full structured details are logged server-side.
        let message = match &self {
            ApiError::Internal(e) => {
                // `{:#}` prints the anyhow chain (top error + sources) — `?e`
                // would only show the outermost message.
                tracing::error!("internal error serving request: {e:#}");
                "internal error".to_string()
            }
            ApiError::UpstreamUnreachable { worker, source } => {
                tracing::warn!(
                    upstream = %worker,
                    error = %format_args!("{source:#}"),
                    "upstream worker unreachable",
                );
                "upstream unavailable".to_string()
            }
            ApiError::UpstreamStatus { status } => {
                tracing::warn!(
                    upstream_status = %status,
                    "upstream returned an error status",
                );
                "upstream returned an error status".to_string()
            }
            ApiError::UpstreamTimeout { worker } => {
                tracing::warn!(upstream = %worker, "upstream request timed out");
                "upstream request timed out".to_string()
            }
            // The WARN "decode leg dropped: its prefill leg failed" is
            // logged where the prefill leg failed (with rid and URLs).
            ApiError::PrefillFailed { detail, .. } => {
                tracing::debug!(error = %detail, "prefill failed; answering the client");
                "prefill failed; the decode request was dropped".to_string()
            }
            ApiError::NoHealthyWorkers { model } => {
                tracing::warn!(model = %model, reason = "no_healthy_workers", "service unavailable");
                "no healthy workers for the requested model".to_string()
            }
            ApiError::NoPrefillWorkersAvailable { model } => {
                tracing::warn!(
                    model = %model,
                    reason = "no_prefill_workers_available",
                    "service unavailable",
                );
                "no prefill workers available for the requested model".to_string()
            }
            ApiError::NoDecodeWorkersAvailable { model } => {
                tracing::warn!(
                    model = %model,
                    reason = "no_decode_workers_available",
                    "service unavailable",
                );
                "no decode workers available for the requested model".to_string()
            }
            ApiError::StaleRequestExpired { model } => {
                tracing::warn!(
                    model = %model,
                    reason = "stale_request_expired",
                    "stale-request janitor expired in-flight request",
                );
                "request expired before completion".to_string()
            }
            ApiError::PolicySelectionFailed { model } => {
                tracing::warn!(model = %model, reason = "policy_selection_failed", "service unavailable");
                "service unavailable".to_string()
            }
            ApiError::BreakerOpen { worker } => {
                tracing::warn!(upstream = %worker, reason = "breaker_open", "service unavailable");
                "service unavailable".to_string()
            }
            ApiError::WorkerMisconfigured { worker, source } => {
                tracing::error!(
                    upstream = %worker,
                    error = %format_args!("{source:#}"),
                    "worker URL emitted by discovery is malformed",
                );
                "service unavailable".to_string()
            }
            // symmetric_pd errors are consumed by the external ingress; the
            // message is the contract's diagnostic (docs/api-reference.md).
            ApiError::SymmetricPdNoPlacement(_) | ApiError::SymmetricPdEndpoints(_) => {
                tracing::warn!(error = %self, "symmetric_pd placement failed");
                self.to_string()
            }
            // The WARN with rid and URLs is logged where the request failed.
            ApiError::LateDNoWinner { detail } => {
                tracing::debug!(error = %detail, "late decode: no winner; answering the client");
                "no decode candidate bound the request".to_string()
            }
            ApiError::BadRequest(_)
            | ApiError::ModelNotFound(_)
            | ApiError::SymmetricPdDisabled
            | ApiError::LateDRouteUnsupported
            | ApiError::LateDBatchUnsupported => self.to_string(),
        };
        let mut resp = (
            status,
            Json(ErrorEnvelope {
                error: ErrorBody { typ, code, message },
            }),
        )
            .into_response();
        resp.headers_mut()
            .insert(X_ROUTER_ERROR_CODE, HeaderValue::from_static(code));
        resp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use http_body_util::BodyExt;
    use serde::Deserialize;

    fn collect_body(resp: Response) -> String {
        let bytes = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async { BodyExt::collect(resp.into_body()).await.unwrap().to_bytes() });
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Pin the exact JSON envelope shape that clients see. Renaming any of
    /// these fields (or removing one) breaks every downstream consumer
    /// silently, so we deserialize into a fixed struct rather than
    /// regex-matching the rendered JSON.
    #[derive(Deserialize)]
    struct ErrEnv {
        error: ErrField,
    }

    #[derive(Deserialize)]
    struct ErrField {
        #[serde(rename = "type")]
        typ: String,
        code: String,
        message: String,
    }

    fn parse_envelope(resp: Response) -> (StatusCode, Option<String>, ErrEnv) {
        let status = resp.status();
        let code_header = resp
            .headers()
            .get("x-router-error-code")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let body_str = collect_body(resp);
        let env: ErrEnv = serde_json::from_str(&body_str)
            .unwrap_or_else(|e| panic!("envelope did not match expected shape: {e}: {body_str}"));
        (status, code_header, env)
    }

    #[test]
    fn upstream_unreachable_envelope_has_code_and_no_leak() {
        let worker_str = "http://10.0.0.42:30000/";
        let worker = reqwest::Url::parse(worker_str).unwrap();
        let secret = "TLS_HANDSHAKE_FAILED at /etc/secret_ca.pem";
        let err = ApiError::UpstreamUnreachable {
            worker: worker.clone(),
            source: anyhow::anyhow!("{secret}"),
        };
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            resp.headers()
                .get("x-router-error-code")
                .and_then(|v| v.to_str().ok()),
            Some("upstream_unreachable"),
        );
        let body = collect_body(resp);
        assert!(body.contains("\"code\":\"upstream_unreachable\""), "{body}");
        assert!(body.contains("\"type\":\"server_error\""), "{body}");
        assert!(
            !body.contains(worker_str) && !body.contains(secret),
            "client body must NOT leak worker URL or reqwest source chain; got: {body}",
        );
    }

    #[test]
    fn upstream_status_envelope_has_code() {
        let err = ApiError::UpstreamStatus {
            status: StatusCode::INTERNAL_SERVER_ERROR,
        };
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            resp.headers()
                .get("x-router-error-code")
                .and_then(|v| v.to_str().ok()),
            Some("upstream_status"),
        );
        let body = collect_body(resp);
        assert!(body.contains("\"code\":\"upstream_status\""), "{body}");
    }

    #[test]
    fn upstream_timeout_envelope_has_code_and_no_leak() {
        let worker_str = "http://10.0.0.42:30000/";
        let worker = reqwest::Url::parse(worker_str).unwrap();
        let err = ApiError::UpstreamTimeout {
            worker: worker.clone(),
        };
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            resp.headers()
                .get("x-router-error-code")
                .and_then(|v| v.to_str().ok()),
            Some("upstream_timeout"),
        );
        let body = collect_body(resp);
        assert!(body.contains("\"code\":\"upstream_timeout\""), "{body}");
        assert!(
            !body.contains(worker_str),
            "client body must NOT leak worker URL; got: {body}",
        );
    }

    #[test]
    fn prefill_failed_envelope_keeps_the_status_and_hides_the_detail() {
        let detail = "upstream timed out: worker http://10.0.0.42:30000/";
        for status in [StatusCode::BAD_GATEWAY, StatusCode::BAD_REQUEST] {
            let err = ApiError::PrefillFailed {
                status,
                detail: detail.to_string(),
            };
            let (got, code_header, env) = parse_envelope(err.into_response());
            assert_eq!(got, status);
            assert_eq!(code_header.as_deref(), Some("prefill_failed"));
            assert_eq!(env.error.code, "prefill_failed");
            assert!(
                !env.error.message.contains("10.0.0.42"),
                "client body must NOT leak the prefill URL: {}",
                env.error.message
            );
        }
    }

    #[test]
    fn late_d_envelopes_have_their_codes() {
        for (err, status, code) in [
            (
                ApiError::LateDRouteUnsupported,
                StatusCode::NOT_IMPLEMENTED,
                "late_d_route_unsupported",
            ),
            (
                ApiError::LateDBatchUnsupported,
                StatusCode::BAD_REQUEST,
                "late_d_batch_unsupported",
            ),
            (
                ApiError::LateDNoWinner {
                    detail: "released by http://10.0.0.42:30000".into(),
                },
                StatusCode::SERVICE_UNAVAILABLE,
                "late_d_no_winner",
            ),
        ] {
            let (got, code_header, env) = parse_envelope(err.into_response());
            assert_eq!(got, status);
            assert_eq!(code_header.as_deref(), Some(code));
            assert_eq!(env.error.code, code);
            assert!(
                !env.error.message.contains("10.0.0.42"),
                "{}",
                env.error.message
            );
        }
    }

    #[test]
    fn bad_request_envelope_has_expected_shape() {
        let msg = "invalid_request: body must be an object";
        let err = ApiError::BadRequest(msg.into());
        let resp = err.into_response();
        let (status, code_header, env) = parse_envelope(resp);

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(code_header.as_deref(), Some("bad_request"));
        assert_eq!(env.error.code, "bad_request");
        assert_eq!(env.error.typ, "invalid_request_error");
        assert!(
            !env.error.message.is_empty(),
            "message must not be empty: {:?}",
            env.error.message,
        );
        assert_ne!(env.error.code, "internal_error");
        assert_ne!(env.error.code, "model_not_found");
    }

    #[test]
    fn model_not_found_envelope_has_expected_shape() {
        let err = ApiError::ModelNotFound("ghost-7b".into());
        let resp = err.into_response();
        let (status, code_header, env) = parse_envelope(resp);

        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(code_header.as_deref(), Some("model_not_found"));
        assert_eq!(env.error.code, "model_not_found");
        assert_eq!(env.error.typ, "invalid_request_error");
        assert!(
            !env.error.message.is_empty(),
            "message must not be empty: {:?}",
            env.error.message,
        );
        assert_ne!(env.error.code, "internal_error");
        assert_ne!(env.error.code, "bad_request");
    }

    #[test]
    fn internal_error_response_sanitizes_anyhow_chain() {
        let secret_msg = "internal /opt/secret/credential.json missing";
        let err = ApiError::Internal(anyhow::anyhow!("{secret_msg}"));
        let resp = err.into_response();
        let body_str = collect_body(resp);
        // Generic to client:
        assert!(
            body_str.contains("\"code\":\"internal_error\""),
            "body: {body_str}"
        );
        assert!(
            body_str.contains("\"type\":\"server_error\""),
            "body: {body_str}"
        );
        // No leak of the original anyhow message:
        assert!(
            !body_str.contains(secret_msg),
            "ApiError::Internal must not leak anyhow chain to client; got: {body_str}"
        );
    }
}
