// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `--pd-abort-decode-on-prefill-failure` over raw-socket mock workers.
//!
//! The prefill mock fails (an HTTP error status, or a connection closed
//! without an answer) or answers 200. The decode mock holds its response
//! the way a decode engine waits for KV that never comes (SSE headers then
//! silence, or no answer at all) and records when the router closes its
//! connection; that close is what makes a real engine abort the request
//! and free its KV.
//!
//! With the flag the router closes the decode connection and fails the
//! client request: an HTTP error status (code `prefill_failed`) while the
//! decode has not answered, a cut stream once the decode headers reached
//! the client. Without the flag the decode leg keeps waiting (old
//! behaviour). A successful prefill leaves the request unchanged.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::json;
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig, PdDecodePolicy,
    PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tower::ServiceExt;

const CHAT: &str = "/v1/chat/completions";
const COMPLETIONS: &str = "/v1/completions";

#[derive(Clone, Copy)]
enum PMode {
    /// Answer this HTTP status.
    Status(u16),
    /// Close the connection without an answer (a transport error).
    Close,
    /// HTTP 500 with the engine error type `symmetric_reader_gone` (the
    /// symmetric owner ended the P-leg because its reader left).
    ReaderGone,
}

#[derive(Clone, Copy)]
enum DMode {
    /// SSE response headers at once, then silence.
    HoldSse,
    /// No answer at all.
    HoldNoHeaders,
    /// SSE headers at once, after `Duration` one event and `[DONE]`.
    StreamDone(Duration),
}

#[derive(Default)]
struct DSeen {
    requests: usize,
    closed_at: Option<Instant>,
}

/// Read one HTTP/1.1 request (head and content-length body); `false` when
/// the peer closed first.
async fn read_request(sock: &mut TcpStream) -> bool {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let head_end = loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        match sock.read(&mut tmp).await {
            Ok(0) | Err(_) => return false,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_ascii_lowercase();
    let len = head
        .lines()
        .find_map(|l| l.strip_prefix("content-length:"))
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0);
    while buf.len() < head_end + len {
        match sock.read(&mut tmp).await {
            Ok(0) | Err(_) => return false,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
        }
    }
    true
}

/// Resolve when the peer closes the connection.
async fn peer_closed(sock: &mut TcpStream) {
    let mut tmp = [0u8; 1024];
    loop {
        match sock.read(&mut tmp).await {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
    }
}

fn chunk(data: &str) -> String {
    format!("{:x}\r\n{data}\r\n", data.len())
}

const SSE_HEAD: &str =
    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n";

async fn start_prefill(mode: PMode, delay: Duration) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        while let Ok((mut sock, _)) = listener.accept().await {
            tokio::spawn(async move {
                if !read_request(&mut sock).await {
                    return;
                }
                tokio::time::sleep(delay).await;
                let answer = match mode {
                    PMode::Close => return,
                    PMode::ReaderGone => {
                        let body = r#"{"object":"error","message":"reader left","type":"symmetric_reader_gone","param":null,"code":500}"#;
                        format!(
                            "HTTP/1.1 500 Internal Server Error\r\ncontent-type: application/json\r\n\
                             content-length: {}\r\nconnection: close\r\n\r\n{body}",
                            body.len()
                        )
                    }
                    PMode::Status(code) => {
                        let body = r#"{"error":"mock prefill failure"}"#;
                        format!(
                            "HTTP/1.1 {code} Mock\r\ncontent-type: application/json\r\n\
                             content-length: {}\r\nconnection: close\r\n\r\n{body}",
                            body.len()
                        )
                    }
                };
                let _ = sock.write_all(answer.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    url
}

async fn start_decode(mode: DMode) -> (String, Arc<Mutex<DSeen>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(DSeen::default()));
    let seen_in = Arc::clone(&seen);
    tokio::spawn(async move {
        while let Ok((mut sock, _)) = listener.accept().await {
            let seen = Arc::clone(&seen_in);
            tokio::spawn(async move {
                if !read_request(&mut sock).await {
                    return;
                }
                seen.lock().unwrap().requests += 1;
                match mode {
                    DMode::HoldNoHeaders => {}
                    DMode::HoldSse => {
                        let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
                    }
                    DMode::StreamDone(after) => {
                        let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
                        tokio::time::sleep(after).await;
                        let body = format!(
                            "{}{}0\r\n\r\n",
                            chunk("data: {\"choices\":[{\"index\":0,\"text\":\"hi\"}]}\n\n"),
                            chunk("data: [DONE]\n\n")
                        );
                        let _ = sock.write_all(body.as_bytes()).await;
                    }
                }
                peer_closed(&mut sock).await;
                seen.lock().unwrap().closed_at = Some(Instant::now());
            });
        }
    });
    (url, seen)
}

fn build_ctx(prefill_url: &str, decode_url: &str, abort: bool) -> Arc<AppContext> {
    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: PolicyKind::RoundRobin,
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            kv_hint: None,
            pd_decode_policy: PdDecodePolicy::AffinityMinLoad,
            pd_decode_tokens: None,
            pd_decision_replay: None,
            symmetric_pd: None,
            gateway_cache_aware: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig {
            abort_decode_on_prefill_failure: abort,
            ..ProxyConfig::default()
        },
        active_load: ActiveLoadConfig::default(),
    };
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::from_config(&cfg));
    for (id, url, mode, port) in [
        ("p0", prefill_url, WorkerMode::Prefill, Some(8997)),
        ("d0", decode_url, WorkerMode::Decode, None),
    ] {
        registry
            .add(WorkerSpec {
                id: WorkerId(id.into()),
                url: url.into(),
                mode,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: port,
            })
            .unwrap();
    }
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

fn request(route: &str, stream: bool) -> Request<Body> {
    let body = if route == CHAT {
        json!({
            "model": "tiny",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "stream": stream,
        })
    } else {
        json!({"model": "tiny", "prompt": "hello", "max_tokens": 4, "stream": stream})
    };
    Request::builder()
        .method("POST")
        .uri(route)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

/// When the decode mock saw its connection closed, polling up to `limit`.
async fn decode_closed_within(seen: &Mutex<DSeen>, limit: Duration) -> Option<Instant> {
    let deadline = Instant::now() + limit;
    loop {
        if let Some(at) = seen.lock().unwrap().closed_at {
            return Some(at);
        }
        if Instant::now() >= deadline {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

fn error_code(resp: &axum::response::Response) -> Option<String> {
    resp.headers()
        .get("x-router-error-code")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
}

/// The decode headers reach the client first; the prefill fails 300 ms
/// later. With the flag the stream is cut and the decode connection closed.
#[tokio::test]
async fn flag_cuts_the_stream_and_closes_the_decode_when_the_prefill_fails_later() {
    for route in [CHAT, COMPLETIONS] {
        let p = start_prefill(PMode::Status(500), Duration::from_millis(300)).await;
        let (d, seen) = start_decode(DMode::HoldSse).await;
        let app = build_router(build_ctx(&p, &d, true));
        let t0 = Instant::now();
        let resp = app.oneshot(request(route, true)).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "{route}: decode headers first"
        );
        let body = tokio::time::timeout(Duration::from_secs(5), resp.into_body().collect())
            .await
            .unwrap_or_else(|_| panic!("{route}: the stream must end after the prefill fails"));
        assert!(body.is_err(), "{route}: a cut stream, not a clean end");
        let closed = decode_closed_within(&seen, Duration::from_secs(5)).await;
        let closed = closed.unwrap_or_else(|| panic!("{route}: decode connection still open"));
        assert!(
            closed.duration_since(t0) < Duration::from_secs(3),
            "{route}: decode closed {:?} after dispatch",
            closed.duration_since(t0)
        );
        assert_eq!(seen.lock().unwrap().requests, 1);
    }
}

/// The decode never answers; the prefill answers 503 after 100 ms. With the
/// flag the client gets the prefill's status (as sgl-model-gateway maps it)
/// with code `prefill_failed`, and the decode connection is closed.
#[tokio::test]
async fn flag_returns_the_prefill_status_before_the_decode_answers() {
    for (route, stream) in [(COMPLETIONS, false), (CHAT, false), (COMPLETIONS, true)] {
        let p = start_prefill(PMode::Status(503), Duration::from_millis(100)).await;
        let (d, seen) = start_decode(DMode::HoldNoHeaders).await;
        let app = build_router(build_ctx(&p, &d, true));
        let resp =
            tokio::time::timeout(Duration::from_secs(5), app.oneshot(request(route, stream)))
                .await
                .unwrap_or_else(|_| panic!("{route} stream={stream}: no answer"))
                .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "{route} stream={stream}"
        );
        assert_eq!(error_code(&resp).as_deref(), Some("prefill_failed"));
        assert!(
            decode_closed_within(&seen, Duration::from_secs(5))
                .await
                .is_some(),
            "{route} stream={stream}: decode connection still open"
        );
    }
}

/// A prefill connection closed without an answer is a transport error:
/// 502 `prefill_failed`.
#[tokio::test]
async fn flag_maps_a_closed_prefill_connection_to_502() {
    let p = start_prefill(PMode::Close, Duration::from_millis(100)).await;
    let (d, seen) = start_decode(DMode::HoldNoHeaders).await;
    let app = build_router(build_ctx(&p, &d, true));
    let resp = tokio::time::timeout(
        Duration::from_secs(5),
        app.oneshot(request(COMPLETIONS, true)),
    )
    .await
    .expect("no answer")
    .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(error_code(&resp).as_deref(), Some("prefill_failed"));
    assert!(decode_closed_within(&seen, Duration::from_secs(5))
        .await
        .is_some());
}

/// Default (no flag): the decode leg keeps waiting after the prefill fails.
#[tokio::test]
async fn without_the_flag_the_decode_keeps_waiting() {
    for route in [CHAT, COMPLETIONS] {
        let p = start_prefill(PMode::Status(500), Duration::from_millis(100)).await;
        let (d, seen) = start_decode(DMode::HoldSse).await;
        let app = build_router(build_ctx(&p, &d, false));
        let resp = app.oneshot(request(route, true)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let pending =
            tokio::time::timeout(Duration::from_millis(1500), resp.into_body().collect()).await;
        assert!(pending.is_err(), "{route}: the stream is still open");
        assert!(
            seen.lock().unwrap().closed_at.is_none(),
            "{route}: the decode connection is still open"
        );
    }
}

/// A prefill 500 with `symmetric_reader_gone` is cancelled, not failed:
/// with the flag the decode leg (local recompute) still answers the client
/// (200 and [DONE]) on both routes, streaming and not, and the prefill
/// breaker is not ticked: four such answers in a row (the breaker opens
/// after three failures) all succeed.
#[tokio::test]
async fn flag_keeps_the_decode_when_the_prefill_reader_left() {
    let before = sgl_router::proxy::prefill_reader_gone_total();
    let p = start_prefill(PMode::ReaderGone, Duration::from_millis(50)).await;
    let (d, seen) = start_decode(DMode::StreamDone(Duration::from_millis(300))).await;
    let app = build_router(build_ctx(&p, &d, true));
    for (route, stream) in [
        (COMPLETIONS, true),
        (CHAT, true),
        (COMPLETIONS, true),
        (CHAT, true),
    ] {
        let resp = tokio::time::timeout(
            Duration::from_secs(5),
            app.clone().oneshot(request(route, stream)),
        )
        .await
        .unwrap_or_else(|_| panic!("{route}: no answer"))
        .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{route}");
        let body = tokio::time::timeout(Duration::from_secs(5), resp.into_body().collect())
            .await
            .expect("stream ends")
            .expect("clean end")
            .to_bytes();
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("[DONE]"), "{route}: {text}");
    }
    assert_eq!(seen.lock().unwrap().requests, 4);
    assert!(sgl_router::proxy::prefill_reader_gone_total() >= before + 4);
}

/// With the flag a successful prefill leaves the request unchanged.
#[tokio::test]
async fn flag_keeps_a_successful_request_unchanged() {
    for route in [CHAT, COMPLETIONS] {
        let p = start_prefill(PMode::Status(200), Duration::from_millis(100)).await;
        let (d, _seen) = start_decode(DMode::StreamDone(Duration::from_millis(300))).await;
        let app = build_router(build_ctx(&p, &d, true));
        let resp = app.oneshot(request(route, true)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = tokio::time::timeout(Duration::from_secs(5), resp.into_body().collect())
            .await
            .expect("stream ends")
            .expect("clean end")
            .to_bytes();
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("[DONE]"), "{route}: {text}");
    }
}
