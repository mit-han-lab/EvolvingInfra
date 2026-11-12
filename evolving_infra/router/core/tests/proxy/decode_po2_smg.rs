// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `--pd-decode-policy power-of-two-smg` end to end over MockWorkers (one
//! prefill, two decodes). The mocks serve no `/v1/loads`, so every pick uses
//! the SMG request-count fallback. A streaming decode leg counts on its
//! decode from the response headers to the end of the stream, a
//! non-streaming leg from dispatch to the response; every count returns to
//! 0. Other decode policies build no `power-of-two-smg` state.

use crate::common::mock_worker::MockWorker;
use axum::body::Body;
use axum::http::{Request, StatusCode};
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
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

fn build_ctx(prefill_url: &str, decode_urls: &[&str], policy: PdDecodePolicy) -> Arc<AppContext> {
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
            pd_decode_policy: policy,
            pd_decode_tokens: None,
            pd_decision_replay: None,
            symmetric_pd: None,
            gateway_cache_aware: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    };
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    registry
        .add(WorkerSpec {
            id: WorkerId("p0".into()),
            url: prefill_url.into(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: Some(8997),
        })
        .unwrap();
    for (i, url) in decode_urls.iter().enumerate() {
        registry
            .add(WorkerSpec {
                id: WorkerId(format!("d{i}")),
                url: (*url).into(),
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: None,
            })
            .unwrap();
    }
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

fn request(path: &str, stream: bool) -> Request<Body> {
    let body = if path == "/v1/chat/completions" {
        json!({"model": "tiny", "messages": [{"role": "user", "content": "hi"}], "stream": stream})
    } else {
        json!({"model": "tiny", "prompt": [1, 2, 3, 4], "max_tokens": 1, "stream": stream})
    };
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

/// Streaming and non-streaming legs on chat and completions: while a slow
/// stream is open its decode counts 1; after every response all counts are 0.
#[tokio::test]
async fn smg_request_count_follows_each_decode_leg() {
    let p = MockWorker::start(vec!["data: [DONE]\n\n"]).await;
    let chunks = vec![
        "data: {\"x\":1}\n\n",
        "data: {\"x\":2}\n\n",
        "data: [DONE]\n\n",
    ];
    let d0 = MockWorker::start_slow_stream(chunks.clone(), Duration::from_millis(250)).await;
    let d1 = MockWorker::start_slow_stream(chunks, Duration::from_millis(250)).await;
    let ctx = build_ctx(&p.url, &[&d0.url, &d1.url], PdDecodePolicy::PowerOfTwoSmg);
    let policy = Arc::clone(ctx.decode_po2_smg.as_ref().expect("power-of-two-smg state"));
    assert!(ctx.decode_tokens.is_none());
    let app = build_router(Arc::clone(&ctx));
    let total = || policy.request_load(&d0.url) + policy.request_load(&d1.url);

    for path in ["/v1/chat/completions", "/v1/completions"] {
        // Streaming: headers are back (first chunk follows in 250 ms) ->
        // counted; the body is read to the end -> released.
        let resp = app.clone().oneshot(request(path, true)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{path}");
        assert_eq!(total(), 1, "{path}: an open stream counts on its decode");
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("[DONE]"), "{path}");
        assert_eq!(total(), 0, "{path}: released at stream end");

        // Non-streaming: counted for the call, released with the response.
        let resp = app.clone().oneshot(request(path, false)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{path}");
        assert_eq!(total(), 0, "{path}: released with the response");
    }

    let s = policy.stats();
    assert_eq!(s.counted_at_headers, 2, "{s:?}");
    assert_eq!(s.counted_at_dispatch, 2, "{s:?}");
    // No /v1/loads on the mocks: every two-decode pick is the fallback.
    assert_eq!(s.selections_requests, 4, "{s:?}");
    assert_eq!(s.selections_tokens, 0, "{s:?}");

    // The monitor polls at start; every mock fails /v1/loads (404).
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while policy.stats().poll_rounds < 1 {
        assert!(tokio::time::Instant::now() < deadline, "no poll round");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let s = policy.stats();
    assert!(s.poll_worker_failures >= 3, "{s:?}");
    assert!(s.poll_empty_rounds >= 1, "{s:?}");

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let text = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&text);
    assert!(
        text.contains("sgl_router_decode_po2_smg_selections_total{signal=\"requests\"} 4"),
        "{text}"
    );
}

/// Other decode policies keep their behavior: no power-of-two-smg state, no
/// extra `/metrics` series.
#[tokio::test]
async fn other_decode_policies_build_no_smg_state() {
    let p = MockWorker::start(vec!["data: [DONE]\n\n"]).await;
    let d0 = MockWorker::start(vec!["data: [DONE]\n\n"]).await;
    for policy in [PdDecodePolicy::RoundRobin, PdDecodePolicy::PowerOfTwo] {
        let ctx = build_ctx(&p.url, &[&d0.url], policy);
        assert!(ctx.decode_po2_smg.is_none());
        let app = build_router(Arc::clone(&ctx));
        let resp = app
            .clone()
            .oneshot(request("/v1/chat/completions", true))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let _ = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let text = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(!String::from_utf8_lossy(&text).contains("decode_po2_smg"));
    }
}
