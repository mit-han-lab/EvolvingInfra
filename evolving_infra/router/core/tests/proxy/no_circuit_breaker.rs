// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! `--disable-circuit-breaker` over MockWorkers (one prefill, one decode
//! that always answers HTTP 500). With the flag the decode stays in the
//! pool and receives every request; with the default breaker (3 consecutive
//! failures) it leaves the pool and the router answers 503.

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

fn build_ctx(prefill_url: &str, decode_url: &str, disable: bool) -> Arc<AppContext> {
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
            disable_circuit_breaker: disable,
            ..ProxyConfig::default()
        },
        active_load: ActiveLoadConfig::default(),
    };
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    // The registry main.rs builds for this config.
    let registry = Arc::new(WorkerRegistry::from_config(&cfg));
    assert_eq!(registry.circuit_breakers_disabled(), disable);
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

/// Six non-streaming chat requests; returns each status and whether the
/// decode received it.
async fn run(disable: bool) -> Vec<(StatusCode, bool)> {
    let p = MockWorker::start(vec![]).await;
    let d = MockWorker::start_returning_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        json!({"error": "decode failure"}),
    )
    .await;
    let app = build_router(build_ctx(&p.url, &d.url, disable));
    let mut out = Vec::new();
    for i in 0..6 {
        d.captured.lock().unwrap().last_body = None;
        let body = json!({
            "model": "tiny",
            "messages": [{"role": "user", "content": format!("hi {i}")}],
            "stream": false,
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let status = app.clone().oneshot(req).await.unwrap().status();
        let reached = d.captured.lock().unwrap().last_body.is_some();
        out.push((status, reached));
    }
    out
}

#[tokio::test]
async fn disabled_breaker_keeps_a_failing_decode_in_the_pool() {
    let got = run(true).await;
    let want = (StatusCode::INTERNAL_SERVER_ERROR, true);
    assert_eq!(got, vec![want; 6], "all reach the failing decode");
}

#[tokio::test]
async fn default_breaker_drops_the_decode_after_three_failures() {
    let got = run(false).await;
    let mut want = vec![(StatusCode::INTERNAL_SERVER_ERROR, true); 3];
    want.extend([(StatusCode::SERVICE_UNAVAILABLE, false); 3]);
    assert_eq!(got, want, "3 failures, then no decode candidate");
}
