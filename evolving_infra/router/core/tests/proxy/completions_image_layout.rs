// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! `/v1/completions` with uploaded screenshots (Kimi K2.6 image requests):
//! `--max-completions-body-mib` lifts the 5 MiB cap, and
//! `--image-layout-config` forwards `prepared_image_layout` (the engine then
//! keys its cache with the router's pad values) while the token-id prompt
//! and every other field stay as the client sent them.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::image_layout::ImageLayoutConfig;
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

use crate::common::mock_worker::MockWorker;

const MODEL: &str = "tiny";

fn golden() -> Value {
    serde_json::from_str(include_str!("../fixtures/kimi_image_layout_golden.json")).unwrap()
}

fn build_ctx(url: String, body_mib: usize, layout: bool) -> Arc<AppContext> {
    let image_layout: Option<&'static ImageLayoutConfig> = layout.then(|| {
        &*Box::leak(Box::new(
            ImageLayoutConfig::from_value(golden()["config"].clone()).unwrap(),
        ))
    });
    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: MODEL.into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: PolicyKind::RoundRobin,
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            kv_hint: None,
            pd_decode_policy: sgl_router::config::PdDecodePolicy::AffinityMinLoad,
            pd_decision_replay: None,
            pd_decode_tokens: None,
            symmetric_pd: None,
            gateway_cache_aware: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig {
            max_completions_body_bytes: body_mib << 20,
            image_layout,
            ..ProxyConfig::default()
        },
        active_load: ActiveLoadConfig::default(),
    };
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(WorkerSpec {
        id: WorkerId(url.clone()),
        url,
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId(MODEL.into())],
        bootstrap_port: None,
    });
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(10)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

async fn post(ctx: Arc<AppContext>, uri: &str, body: Vec<u8>) -> (StatusCode, String) {
    let app = build_router(ctx);
    let req = Request::builder()
        .method(if body.is_empty() { "GET" } else { "POST" })
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

fn captured(mock: &MockWorker) -> Option<Value> {
    let b = mock.captured.lock().unwrap().last_body.clone()?;
    Some(serde_json::from_slice(&b).expect("captured body is valid JSON"))
}

fn big_body(bytes: usize) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "model": MODEL,
        "prompt": [1, 2, 3],
        "max_tokens": 1,
        "pad": "x".repeat(bytes),
    }))
    .unwrap()
}

#[tokio::test]
async fn default_cap_rejects_6_mib_and_the_flag_admits_it() {
    let mock = MockWorker::start(vec![]).await;
    let (status, _) = post(
        build_ctx(mock.url.clone(), 5, false),
        "/v1/completions",
        big_body(6 << 20),
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "default 5 MiB cap");
    assert!(captured(&mock).is_none(), "the 413 happens before dispatch");

    let (status, _) = post(
        build_ctx(mock.url.clone(), 64, false),
        "/v1/completions",
        big_body(40 << 20),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "--max-completions-body-mib 64 admits 40 MiB"
    );
    let fwd = captured(&mock).expect("forwarded");
    assert_eq!(
        fwd["pad"].as_str().map(str::len),
        Some(40 << 20),
        "body forwarded intact"
    );
}

#[tokio::test]
async fn image_layout_is_injected_and_the_prompt_is_unchanged() {
    let mock = MockWorker::start(vec![]).await;
    let g = golden();
    let uris: Vec<Value> = g["images"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["data_uri"].clone())
        .collect();
    let prompt = g["prompt"].clone();
    let body = json!({
        "model": MODEL,
        "prompt": prompt,
        "image_data": uris,
        "max_tokens": 1,
        "cache_salt": "s",
        "rid": "rid_0123456789abcdef0123456789abcdef",
    });
    let ctx = build_ctx(mock.url.clone(), 64, true);
    let (status, _) = post(
        ctx.clone(),
        "/v1/completions",
        serde_json::to_vec(&body).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let fwd = captured(&mock).expect("forwarded");
    assert_eq!(
        fwd["prepared_image_layout"], g["layout"],
        "router layout = Python layout"
    );
    for key in [
        "prompt",
        "image_data",
        "cache_salt",
        "rid",
        "max_tokens",
        "model",
    ] {
        assert_eq!(fwd[key], body[key], "{key} forwarded unchanged");
    }
    let (status, text) = post(ctx, "/metrics", Vec::new()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        text.contains("sgl_router_image_layout_total{outcome=\"expanded_injected\"}"),
        "{text}"
    );
}

#[tokio::test]
async fn unreadable_images_forward_the_original_body() {
    let mock = MockWorker::start(vec![]).await;
    let body = json!({
        "model": MODEL,
        "prompt": [1, 163605, 2],
        "image_data": ["https://example.invalid/a.png"],
        "max_tokens": 1,
    });
    let ctx = build_ctx(mock.url.clone(), 64, true);
    let (status, _) = post(ctx, "/v1/completions", serde_json::to_vec(&body).unwrap()).await;
    assert_eq!(status, StatusCode::OK);
    let fwd = captured(&mock).expect("forwarded");
    assert!(fwd.get("prepared_image_layout").is_none());
    assert_eq!(fwd, body, "skipped requests are forwarded as sent");
}
