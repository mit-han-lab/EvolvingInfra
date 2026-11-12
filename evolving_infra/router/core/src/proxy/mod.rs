// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! HTTP proxy — forwards requests to the upstream SGLang worker.

pub mod sse;

use crate::health::circuit_breaker::CircuitBreaker;
use crate::server::error::ApiError;
use crate::server::header_utils::should_forward_request_header;
use anyhow::Context;
use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response, StatusCode};
use bytes::Bytes;
use reqwest::{Client, Url};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// The engine `err_type` of a symmetric P-leg that it finished because its
/// reader (the D-leg) left (FINISH_ABORT, HTTP 500). The D-leg may still
/// serve the client by local recompute, so the router treats such a P-leg
/// as cancelled, not failed: it does not drop the D-leg(s), ticks no
/// circuit breaker, and releases the P-leg's ledgers once as finished.
/// The marker is found anywhere in a 5xx body (the error `type`, the
/// abort `err_type` or the message).
pub const READER_GONE_ERR_TYPE: &str = "symmetric_reader_gone";

/// The reason that closes the work-book reservation of such a P-leg.
pub const READER_GONE_FINISH_REASON: &str = "prefill_reader_gone";

/// Response extension: the answer is a [`READER_GONE_ERR_TYPE`] P-leg end.
#[derive(Debug, Clone, Copy)]
pub struct ReaderGone;

/// P-leg answers with [`READER_GONE_ERR_TYPE`] seen by this process (all
/// routes).
static PREFILL_READER_GONE: AtomicU64 = AtomicU64::new(0);

/// The number of [`READER_GONE_ERR_TYPE`] P-leg answers so far.
pub fn prefill_reader_gone_total() -> u64 {
    PREFILL_READER_GONE.load(Ordering::Relaxed)
}

/// A 5xx answer whose body (at most 64 KiB) names
/// [`READER_GONE_ERR_TYPE`]. Counts it.
pub fn body_is_reader_gone(status: StatusCode, body: &[u8]) -> bool {
    let needle = READER_GONE_ERR_TYPE.as_bytes();
    let found = status.is_server_error()
        && body.len() <= 64 * 1024
        && body.windows(needle.len()).any(|w| w == needle);
    if found {
        PREFILL_READER_GONE.fetch_add(1, Ordering::Relaxed);
    }
    found
}

/// The prefill result is a reader-gone end ([`ReaderGone`]).
pub fn is_reader_gone(result: &Result<Response<Body>, ApiError>) -> bool {
    matches!(result, Ok(response) if response.extensions().get::<ReaderGone>().is_some())
}

/// `--pd-abort-decode-on-prefill-failure`: the verdict of a PD request's
/// detached prefill leg, shared with the decode leg of the same request.
///
/// The prefill task calls [`PrefillFailure::fail`] on a transport error or
/// a non-2xx answer. The decode leg then stops waiting: before the decode
/// response headers through a `select!` arm on [`prefill_failed`], after
/// them through the SSE pump watching [`PrefillFailure::token`]. Either way
/// the decode connection is closed, which makes the engine abort the
/// request and free its KV (sgl-model-gateway drops its decode future for
/// the same reason).
#[derive(Clone, Debug, Default)]
pub struct PrefillFailure {
    token: CancellationToken,
    reason: Arc<parking_lot::Mutex<Option<(StatusCode, String)>>>,
}

impl PrefillFailure {
    pub fn new() -> Self {
        Self::default()
    }

    /// The token the decode SSE pump watches.
    pub fn token(&self) -> CancellationToken {
        self.token.clone()
    }

    /// Record why the prefill leg failed (first call wins) and fire the
    /// token.
    pub fn fail(&self, status: StatusCode, detail: impl Into<String>) {
        {
            let mut reason = self.reason.lock();
            if reason.is_none() {
                *reason = Some((status, detail.into()));
            }
        }
        self.token.cancel();
    }

    /// Why a finished prefill call failed, or `None` when it answered
    /// 2xx or ended because its reader left ([`ReaderGone`]: the D-leg
    /// owns the outcome): any transport-level error is 502, a non-2xx
    /// answer maps by [`prefill_failure_status`].
    pub fn verdict(result: &Result<Response<Body>, ApiError>) -> Option<(StatusCode, String)> {
        match result {
            Ok(response) if response.status().is_success() => None,
            Ok(response) if response.extensions().get::<ReaderGone>().is_some() => None,
            Ok(response) => Some((
                prefill_failure_status(response.status()),
                format!("prefill answered {}", response.status()),
            )),
            Err(error) => Some((StatusCode::BAD_GATEWAY, error.to_string())),
        }
    }

    /// Log the WARN "decode leg dropped: its prefill leg failed" and fire
    /// the token.
    pub fn fail_and_log(&self, status: StatusCode, detail: String, legs: &PdLegs<'_>) {
        // Policies that never parse the body have no rid: read it from the
        // prefill body then (only on this failure path).
        let rid = legs
            .rid
            .map(str::to_owned)
            .or_else(|| legs.body.and_then(rid_from_json_body))
            .unwrap_or_default();
        tracing::warn!(
            rid = %rid,
            bootstrap_room = ?legs.bootstrap_room,
            prefill_url = %legs.prefill_url,
            decode_url = %legs.decode_url,
            status = status.as_u16(),
            error = %detail,
            "decode leg dropped: its prefill leg failed",
        );
        self.fail(status, detail);
    }

    /// The client-facing error for a failed prefill leg.
    pub fn error(&self) -> ApiError {
        let (status, detail) = self
            .reason
            .lock()
            .clone()
            .unwrap_or((StatusCode::BAD_GATEWAY, "prefill leg failed".to_string()));
        ApiError::PrefillFailed { status, detail }
    }
}

/// Identity of one PD request's legs, for the drop WARN.
#[derive(Debug, Clone, Copy)]
pub struct PdLegs<'a> {
    pub rid: Option<&'a str>,
    pub bootstrap_room: Option<u64>,
    pub prefill_url: &'a str,
    pub decode_url: &'a str,
    /// The prefill request body: the rid source when `rid` is `None`.
    pub body: Option<&'a [u8]>,
}

/// The `rid` string of a JSON request body, if any.
pub fn rid_from_json_body(body: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    value
        .get("rid")?
        .as_str()
        .filter(|rid| !rid.is_empty())
        .map(str::to_owned)
}

/// Resolves with the client-facing error once the prefill leg failed.
/// Never resolves when `failure` is `None` (flag off) or while the prefill
/// leg has not failed.
pub async fn prefill_failed(failure: Option<&PrefillFailure>) -> ApiError {
    match failure {
        Some(failure) => {
            failure.token.cancelled().await;
            failure.error()
        }
        None => std::future::pending().await,
    }
}

/// Client status for a prefill leg that answered a non-2xx `status`, as
/// sgl-model-gateway maps it (`process_prefill_response`): 400, 404, 500,
/// 502 and 503 pass through; any other status becomes 500.
pub fn prefill_failure_status(status: StatusCode) -> StatusCode {
    match status {
        StatusCode::BAD_REQUEST
        | StatusCode::NOT_FOUND
        | StatusCode::INTERNAL_SERVER_ERROR
        | StatusCode::BAD_GATEWAY
        | StatusCode::SERVICE_UNAVAILABLE => status,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// Parse a worker URL emitted by discovery.  On failure, trip the worker's
/// circuit breaker so the malformed worker drops out of subsequent
/// `healthy_workers_for(...)` selection, then surface the error as
/// `ApiError::WorkerMisconfigured`.
fn parse_worker_url(worker_url: &str, breaker: &CircuitBreaker) -> Result<Url, ApiError> {
    Url::parse(worker_url).map_err(|e| {
        breaker.record_failure();
        ApiError::WorkerMisconfigured {
            worker: worker_url.to_string(),
            source: anyhow::Error::new(e).context("parse worker URL"),
        }
    })
}

#[derive(Debug)]
pub struct Proxy {
    pub client: Client,
    /// Wall-clock timeout applied to non-streaming upstream requests. Streaming
    /// requests deliberately do not use this (long generations are valid).
    pub request_timeout: Duration,
}

impl Proxy {
    /// Build a proxy. `request_timeout` is the per-request wall-clock budget for
    /// non-streaming forwards. Connect timeout is hard-coded to 5 s — even a
    /// streaming request fails fast at TCP setup if the worker is unreachable.
    pub fn new(request_timeout: Duration) -> Result<Self, anyhow::Error> {
        Self::with_tcp_keepalive(request_timeout, None)
    }

    /// [`Proxy::new`] with an explicit TCP keepalive idle time
    /// (`TCP_KEEPIDLE`) for every upstream socket
    /// (`--proxy-tcp-keepalive-secs`). `None` keeps reqwest's default of
    /// 15 s. The probe interval (15 s), probe count (3) and
    /// `TCP_USER_TIMEOUT` (30 s) stay at the reqwest defaults, which
    /// sgl-model-gateway also keeps; its idle time defaults to 30 s.
    ///
    /// Why it matters: with `TCP_USER_TIMEOUT` set, Linux resets a
    /// connection that waits for response headers at the first keepalive
    /// tick that is at least 30 s after the peer's last ACK while a probe
    /// is unanswered (`ETIMEDOUT`, surfaced as `upstream timed out`). A
    /// busy worker's kernel still answers probes, so only lost probes or
    /// ACKs trigger it; the idle time sets when the first probe goes out.
    pub fn with_tcp_keepalive(
        request_timeout: Duration,
        tcp_keepalive: Option<Duration>,
    ) -> Result<Self, anyhow::Error> {
        let mut builder = Client::builder()
            // Zero idle pool: every request opens a fresh connection. The
            // reuse race against the workers' 5 s uvicorn keep-alive cannot
            // be fully closed by a short idle timeout (observed: synchronized
            // SendRequest resets with 3 s idle); the stock gateway needs a
            // connection-level targeted retry for the same reason. At the
            // strict-replay request rates (<= ~15 req/s) fresh TCP setup is
            // noise, and streaming connections are per-request anyway.
            .pool_max_idle_per_host(0)
            // Below the workers' 5 s uvicorn keep-alive: reqwest's default
            // (90 s) reuses connections the worker already closed, producing
            // sporadic 502s that void zero-failure points (a gateway
            // in front of the same workers needs --pool-idle-timeout-secs 3
            // for the same reason).
            .pool_idle_timeout(Duration::from_secs(3))
            .connect_timeout(Duration::from_secs(5));
        if let Some(idle) = tcp_keepalive {
            builder = builder.tcp_keepalive(Some(idle));
        }
        let client = builder.build().context("build reqwest client")?;
        Ok(Self {
            client,
            request_timeout,
        })
    }

    /// Classify a reqwest error into the right `ApiError` variant, given an
    /// explicit worker URL. Called from the breaker-gated `forward_*_to`
    /// methods, which carry per-request worker URLs (not a single proxy-level
    /// URL).
    ///
    /// Walks the full source chain to detect timeouts, because reqwest wraps
    /// hyper which wraps `std::io::Error` — a top-level `is_timeout()` check
    /// misses both the wrapped reqwest timeout and the `io::ErrorKind::TimedOut`
    /// cases.
    fn classify_reqwest_error_for(worker: Url, e: reqwest::Error, path: &str) -> ApiError {
        let source = anyhow::Error::new(e).context(format!("worker {worker}: post {path}"));
        let is_timeout = source.chain().any(|c| {
            c.downcast_ref::<reqwest::Error>()
                .is_some_and(|r| r.is_timeout())
        }) || source.chain().any(|c| {
            c.downcast_ref::<std::io::Error>()
                .is_some_and(|io| io.kind() == std::io::ErrorKind::TimedOut)
        });
        if is_timeout {
            ApiError::UpstreamTimeout { worker }
        } else {
            ApiError::UpstreamUnreachable { worker, source }
        }
    }

    /// Send with a connection-level targeted retry: connect-class failures
    /// (SYN loss, the hard-coded 5 s connect timeout, refused/reset before
    /// any byte of the request was written) are retried up to 2 extra
    /// times. Zero request bytes have been sent when `is_connect()` holds,
    /// so this is safe even under strict replay (`max_retries=0` on the
    /// client governs REQUEST-level retries, not TCP setup). The stock
    /// gateway carries the same fix ("connection-level targeted retry").
    async fn send_with_connect_retry(
        req: reqwest::RequestBuilder,
        url: &Url,
    ) -> Result<reqwest::Response, reqwest::Error> {
        let mut attempt = 0u32;
        loop {
            let cloned = req
                .try_clone()
                .expect("proxy bodies are Bytes; RequestBuilder is cloneable");
            match cloned.send().await {
                Ok(resp) => return Ok(resp),
                Err(e) if e.is_connect() && attempt < 2 => {
                    attempt += 1;
                    tracing::warn!(
                        upstream = %url,
                        attempt,
                        error = %e,
                        "connect failed before any byte was sent; retrying",
                    );
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Breaker-gated JSON POST: checks `breaker.allow()` first, records
    /// success/failure based on response status, and returns
    /// `ApiError::BreakerOpen` immediately when the breaker is Open.
    ///
    /// `worker_url` is the discovery-emitted worker URL string. It's parsed
    /// to [`reqwest::Url`] internally so we can use [`Url::join`] for clean
    /// path concatenation (no double-slash) and pass a typed URL to the
    /// split error variants (`UpstreamUnreachable` / `UpstreamTimeout` /
    /// `UpstreamStatus`).
    pub async fn forward_json_to(
        &self,
        worker_url: &str,
        breaker: &CircuitBreaker,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
    ) -> Result<Response<Body>, ApiError> {
        if !breaker.allow() {
            return Err(ApiError::BreakerOpen {
                worker: worker_url.to_string(),
            });
        }
        let worker_url = parse_worker_url(worker_url, breaker)?;
        let url = worker_url.join(path).map_err(|e| {
            ApiError::Internal(anyhow::Error::new(e).context(format!("join worker path {path}")))
        })?;
        let mut req = self.client.post(url.clone()).body(body);
        for (k, v) in headers {
            if should_forward_request_header(k) {
                req = req.header(k, v);
            }
        }
        req = req
            .header("content-type", "application/json")
            .timeout(self.request_timeout);
        let resp = Self::send_with_connect_retry(req, &url)
            .await
            .map_err(|e| {
                breaker.record_failure();
                Self::classify_reqwest_error_for(worker_url.clone(), e, path)
            })?;
        let status = resp.status();
        // Defer breaker recording until after the body completes — a
        // worker that returns 2xx headers and then drops mid-body is
        // still failing the request, and crediting it as healthy lets
        // a misbehaving worker stay eligible. For 5xx the early bail is
        // safe (no body to consume meaningfully), but we still wait
        // until after the read attempt to record exactly once.
        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    upstream = %url,
                    status = %status,
                    error = ?e,
                    "upstream dropped connection mid-body",
                );
                breaker.record_failure();
                return Err(ApiError::UpstreamStatus { status });
            }
        };
        // A P-leg that ended because its reader left ticks no breaker; a
        // half-open breaker is closed, so its probe slot is not lost.
        let reader_gone = body_is_reader_gone(status, &bytes);
        if reader_gone {
            crate::late_bind::settle_late_answer(breaker);
        } else if status.is_server_error() {
            breaker.record_failure();
        } else {
            breaker.record_success();
        }
        let mut out = Response::new(Body::from(bytes));
        *out.status_mut() = status;
        if reader_gone {
            out.extensions_mut().insert(ReaderGone);
        }
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("application/json"),
        );
        Ok(out)
    }

    /// Breaker-gated streaming POST: checks `breaker.allow()` first, records
    /// success/failure, and returns `ApiError::BreakerOpen` when Open.
    ///
    /// `stream_guards` — when `Some`, the value is threaded into the SSE
    /// pump task and held for the entire body lifetime (headers → last byte
    /// / client disconnect).  The proxy does not inspect the boxed value; it
    /// relies entirely on `Drop` semantics, so callers typically pack
    /// `(LoadGuard, ActiveLoadGuard)` here. This keeps both the per-worker
    /// `active_requests` counter and the per-request active-load entry alive
    /// for the full streaming lifetime — without which a long-running SSE
    /// response would under-report load.
    // Each parameter is a distinct, required input to a single upstream
    // forward (target, breaker, path, headers, body, plus the two
    // streaming-lifetime callbacks). Bundling them into a struct purely to
    // satisfy the arg-count heuristic would add indirection without clarity.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_streaming_to(
        &self,
        worker_url: &str,
        breaker: &Arc<CircuitBreaker>,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
        stream_guards: Option<Box<dyn Send + 'static>>,
        on_first_byte: Option<Box<dyn FnOnce() + Send + 'static>>,
    ) -> Result<Response<Body>, ApiError> {
        self.forward_streaming_to_abortable(
            worker_url,
            breaker,
            path,
            headers,
            body,
            stream_guards,
            on_first_byte,
            None,
        )
        .await
    }

    /// [`Proxy::forward_streaming_to`] whose SSE pump also stops when
    /// `abort` fires (`--pd-abort-decode-on-prefill-failure`): it drops the
    /// upstream stream, which closes the upstream connection, ends the
    /// client body with an error, and records no breaker outcome (the
    /// upstream is not at fault).
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_streaming_to_abortable(
        &self,
        worker_url: &str,
        breaker: &Arc<CircuitBreaker>,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
        stream_guards: Option<Box<dyn Send + 'static>>,
        on_first_byte: Option<Box<dyn FnOnce() + Send + 'static>>,
        abort: Option<CancellationToken>,
    ) -> Result<Response<Body>, ApiError> {
        if !breaker.allow() {
            return Err(ApiError::BreakerOpen {
                worker: worker_url.to_string(),
            });
        }
        let worker_url = parse_worker_url(worker_url, breaker)?;
        let url = worker_url.join(path).map_err(|e| {
            ApiError::Internal(anyhow::Error::new(e).context(format!("join worker path {path}")))
        })?;
        let mut req = self.client.post(url.clone()).body(body);
        for (k, v) in headers {
            if should_forward_request_header(k) {
                req = req.header(k, v);
            }
        }
        req = req
            .header("content-type", "application/json")
            .header("accept", "text/event-stream");
        let resp = Self::send_with_connect_retry(req, &url)
            .await
            .map_err(|e| {
                breaker.record_failure();
                Self::classify_reqwest_error_for(worker_url.clone(), e, path)
            })?;
        let status = resp.status();
        let upstream_ct = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/json")
            .to_string();
        let content_type = if status.is_success() {
            "text/event-stream".to_string()
        } else {
            upstream_ct
        };
        // Breaker recording is deferred to the pump's completion hook so
        // an upstream that returns 2xx headers and then drops mid-stream
        // is recorded as a failure. For 5xx headers we record_failure
        // up front and skip the pump hook (the body we surface is the
        // error response — its stream completing is not a worker win).
        let on_complete: Option<Box<dyn FnOnce(bool) + Send + 'static>> =
            if status.is_server_error() {
                breaker.record_failure();
                None
            } else {
                let breaker_for_hook = Arc::clone(breaker);
                Some(Box::new(move |ok| {
                    if ok {
                        breaker_for_hook.record_success();
                    } else {
                        breaker_for_hook.record_failure();
                    }
                }))
            };
        // Only record TTFT for successful streams — a 4xx/5xx error body
        // streaming back is not a generated token, so drop the hook for
        // non-2xx responses.
        let first_byte_hook = if status.is_success() {
            on_first_byte
        } else {
            None
        };
        let body = sse::bytes_stream_to_body_abortable(
            resp.bytes_stream(),
            stream_guards,
            on_complete,
            first_byte_hook,
            abort,
        );
        let mut out = Response::new(body);
        *out.status_mut() = status;
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_str(&content_type)
                .unwrap_or_else(|_| HeaderValue::from_static("application/json")),
        );
        Ok(out)
    }

    /// One leg of a late-D choose request (`--pd-late-decode choose`): a
    /// JSON POST with the connection-level retry and the forwarded request
    /// headers, but without circuit-breaker gating or recording. The caller
    /// claims the breaker before and records the outcome after by the
    /// late-D rules (a released or otherwise late-D answer, with
    /// `X-Late-D-Reason`, ticks no breaker). `sse` asks for an event stream
    /// and leaves the exchange without a deadline, as
    /// [`Proxy::forward_streaming_to`] does; otherwise the request timeout
    /// bounds the whole exchange, body included, as
    /// [`Proxy::forward_json_to`] does. Returns the upstream response as is.
    pub async fn send_late_leg(
        &self,
        worker_url: &str,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
        sse: bool,
    ) -> Result<reqwest::Response, ApiError> {
        let worker = Url::parse(worker_url).map_err(|e| ApiError::WorkerMisconfigured {
            worker: worker_url.to_string(),
            source: anyhow::Error::new(e).context("parse worker URL"),
        })?;
        let url = worker.join(path).map_err(|e| {
            ApiError::Internal(anyhow::Error::new(e).context(format!("join worker path {path}")))
        })?;
        let mut req = self.client.post(url.clone()).body(body);
        for (k, v) in headers {
            if should_forward_request_header(k) {
                req = req.header(k, v);
            }
        }
        req = req.header("content-type", "application/json");
        req = if sse {
            req.header("accept", "text/event-stream")
        } else {
            req.timeout(self.request_timeout)
        };
        Self::send_with_connect_retry(req, &url)
            .await
            .map_err(|e| Self::classify_reqwest_error_for(worker, e, path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// The reader-gone marker counts only in a 5xx body; the verdict of
    /// such an answer is "not failed".
    #[test]
    fn reader_gone_is_a_5xx_body_marker() {
        let body = br#"{"object":"error","message":"x","type":"symmetric_reader_gone","code":500}"#;
        assert!(body_is_reader_gone(StatusCode::INTERNAL_SERVER_ERROR, body));
        assert!(!body_is_reader_gone(StatusCode::BAD_REQUEST, body));
        assert!(!body_is_reader_gone(
            StatusCode::INTERNAL_SERVER_ERROR,
            b"{\"type\":\"500\"}"
        ));
        let mut gone = Response::new(Body::empty());
        *gone.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
        gone.extensions_mut().insert(ReaderGone);
        let gone: Result<Response<Body>, ApiError> = Ok(gone);
        assert!(is_reader_gone(&gone) && PrefillFailure::verdict(&gone).is_none());
        let mut failed = Response::new(Body::empty());
        *failed.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
        let failed: Result<Response<Body>, ApiError> = Ok(failed);
        assert!(!is_reader_gone(&failed) && PrefillFailure::verdict(&failed).is_some());
    }

    #[test]
    fn rid_from_json_body_reads_a_nonempty_string_rid() {
        assert_eq!(
            rid_from_json_body(br#"{"rid":"r-1","prompt":"x"}"#).as_deref(),
            Some("r-1")
        );
        assert_eq!(rid_from_json_body(br#"{"rid":""}"#), None);
        assert_eq!(rid_from_json_body(br#"{"rid":7}"#), None);
        assert_eq!(rid_from_json_body(br#"{"prompt":"x"}"#), None);
        assert_eq!(rid_from_json_body(b"not json"), None);
    }

    #[tokio::test]
    async fn new_returns_result_not_panic() {
        let p = Proxy::new(Duration::from_secs(5)).unwrap();
        assert_eq!(p.request_timeout, Duration::from_secs(5));
    }

    /// `--proxy-tcp-keepalive-secs`: the kernel keepalive timer of an
    /// upstream socket that waits for response headers runs on the
    /// configured idle time (30 s, as in sgl-model-gateway) instead of
    /// reqwest's 15 s default. Reads the client socket's timer from
    /// `/proc/net/tcp`.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn tcp_keepalive_sets_the_kernel_keepalive_idle_time() {
        let default_secs = waiting_socket_keepalive_secs(None).await;
        let smg_secs = waiting_socket_keepalive_secs(Some(Duration::from_secs(30))).await;
        eprintln!("keepalive timer: default {default_secs:.2} s, with 30 s flag {smg_secs:.2} s");
        assert!(
            (12.0..=17.5).contains(&default_secs),
            "default keepalive timer {default_secs} s, want about 15 s"
        );
        assert!(
            (27.0..=32.5).contains(&smg_secs),
            "--proxy-tcp-keepalive-secs 30: timer {smg_secs} s, want about 30 s"
        );
    }

    /// Remaining keepalive time (s) of the proxy's socket while the stub
    /// worker holds the response; then releases the stub and checks the
    /// request still completes.
    #[cfg(target_os = "linux")]
    async fn waiting_socket_keepalive_secs(tcp_keepalive: Option<Duration>) -> f64 {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
        let worker = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut head = Vec::new();
            let mut buf = [0u8; 1024];
            while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                let n = sock.read(&mut buf).await.unwrap();
                assert!(n > 0, "client closed before sending the request");
                head.extend_from_slice(&buf[..n]);
            }
            let _ = release_rx.await;
            sock.write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\n{}")
                .await
                .unwrap();
        });
        let proxy = Proxy::with_tcp_keepalive(Duration::from_secs(30), tcp_keepalive).unwrap();
        let client = proxy.client.clone();
        let request = tokio::spawn(async move {
            client
                .get(format!("http://127.0.0.1:{port}/hold"))
                .send()
                .await
                .map(|r| r.status().as_u16())
        });
        let mut timer = None;
        for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            timer = client_keepalive_timer_secs(port);
            if timer.is_some() {
                break;
            }
        }
        let _ = release_tx.send(());
        assert_eq!(request.await.unwrap().unwrap(), 200);
        worker.await.unwrap();
        timer.expect("an ESTABLISHED client socket with a keepalive timer in /proc/net/tcp")
    }

    /// The keepalive timer (s) of the ESTABLISHED IPv4 socket whose remote
    /// end is 127.0.0.1:`server_port` (the client side of the test
    /// connection); `None` while no keepalive timer (`tr` 02) is pending.
    /// `/proc/net/tcp` reports `tm->when` in clock ticks (USER_HZ = 100).
    #[cfg(target_os = "linux")]
    fn client_keepalive_timer_secs(server_port: u16) -> Option<f64> {
        let table = std::fs::read_to_string("/proc/net/tcp").ok()?;
        let remote = format!("0100007F:{server_port:04X}");
        for line in table.lines().skip(1) {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 6 || f[2] != remote || f[3] != "01" {
                continue;
            }
            let (tr, when) = f[5].split_once(':')?;
            if tr != "02" {
                return None;
            }
            return Some(u64::from_str_radix(when, 16).ok()? as f64 / 100.0);
        }
        None
    }
}
