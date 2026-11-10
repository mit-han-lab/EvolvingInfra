// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `--pd-decode-policy min-inflight-tokens` end to end over MockWorkers:
//! decode workers are chosen by charged tokens (not request count), the
//! charge is visible in `/metrics`, and it is released on every path that
//! ends the decode leg.

use crate::common::mock_worker::MockWorker;
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
use sgl_router::workers::{Worker, WorkerRegistry};
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

fn config(policy: PdDecodePolicy) -> Config {
    Config {
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
            co_schedule: None,
            pd_decode_policy: policy,
            // None with a token-charged policy = defaults (output cap 8192).
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
    }
}

/// One prefill (bootstrap port 8997) + the given decode URLs.
fn build_ctx(prefill_url: &str, decode_urls: &[&str]) -> Arc<AppContext> {
    let cfg = config(PdDecodePolicy::MinInflightTokens);
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
    let ctx = Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies));
    assert!(ctx.decode_tokens.is_some());
    ctx
}

fn decode_worker(ctx: &AppContext, url: &str) -> Arc<Worker> {
    ctx.registry
        .workers_for(&ModelId("tiny".into()))
        .into_iter()
        .find(|w| w.url == url)
        .unwrap()
}

fn completions(prompt_len: usize, stream: bool) -> Request<Body> {
    let prompt: Vec<u32> = (0..prompt_len as u32).map(|i| 1 + i % 50).collect();
    Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny", "prompt": prompt, "max_tokens": 1, "stream": stream,
            }))
            .unwrap(),
        ))
        .unwrap()
}

fn chat(content: &str, stream: bool) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": content}],
                "stream": stream,
            }))
            .unwrap(),
        ))
        .unwrap()
}

async fn wait_released(w: &Worker, label: &str) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while w.decode_charged_tokens() != 0 || w.decode_charged_reqs() != 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "{label}: charge not released ({} tokens / {} reqs)",
            w.decode_charged_tokens(),
            w.decode_charged_reqs(),
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(w.active_load(), 0, "{label}: decode LoadGuard leaked");
}

async fn wait_charged(w: &Worker, label: &str) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while w.decode_charged_reqs() == 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "{label}: never charged"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// H, L, L, L, H with streams held open: the first heavy prompt's decode
/// gets no light request, and the second heavy prompt goes to the other
/// decode — 1 vs 4 requests, ~equal tokens. A request-count policy would
/// have alternated. `/metrics` reports the per-decode charges; all of them
/// return to 0 when the streams end.
#[tokio::test]
async fn heavy_prompts_spread_by_tokens_not_count() {
    let p = MockWorker::start(vec!["data: [DONE]\n\n"]).await;
    let d0 = MockWorker::start_slow_stream(
        vec!["data: {}\n\n", "data: [DONE]\n\n"],
        Duration::from_millis(1500),
    )
    .await;
    let d1 = MockWorker::start_slow_stream(
        vec!["data: {}\n\n", "data: [DONE]\n\n"],
        Duration::from_millis(1500),
    )
    .await;
    let ctx = build_ctx(&p.url, &[&d0.url, &d1.url]);
    let app = build_router(Arc::clone(&ctx));
    let decodes = [&d0, &d1];

    const HEAVY: usize = 20_000;
    const LIGHT: usize = 10;
    let mut held = Vec::new();
    let mut placed = Vec::new();
    for len in [HEAVY, LIGHT, LIGHT, LIGHT, HEAVY] {
        for d in decodes {
            d.captured.lock().unwrap().last_body = None;
        }
        let response = app.clone().oneshot(completions(len, true)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let hit: Vec<&str> = decodes
            .iter()
            .filter(|d| d.captured.lock().unwrap().last_body.is_some())
            .map(|d| d.url.as_str())
            .collect();
        assert_eq!(hit.len(), 1, "exactly one decode receives each request");
        placed.push(hit[0].to_string());
        held.push(response);
    }
    let first_heavy = placed[0].clone();
    let other = if first_heavy == d0.url {
        &d1.url
    } else {
        &d0.url
    };
    assert!(
        placed[1..].iter().all(|u| u == other),
        "light requests and the 2nd heavy one avoid the 1st heavy one's decode: {placed:?}"
    );

    // Charges: prompt tokens + min(max_tokens=1, cap).
    let wa = decode_worker(&ctx, &first_heavy);
    let wb = decode_worker(&ctx, other);
    assert_eq!(
        (wa.decode_charged_tokens(), wa.decode_charged_reqs()),
        (20_001, 1)
    );
    assert_eq!(
        (wb.decode_charged_tokens(), wb.decode_charged_reqs()),
        (20_034, 4)
    );

    let metrics = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let text = String::from_utf8(
        metrics
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap();
    assert!(text.contains(&format!(
        "sgl_router_decode_charged_tokens{{worker_url=\"{first_heavy}\"}} 20001"
    )));
    assert!(text.contains(&format!(
        "sgl_router_decode_charged_tokens{{worker_url=\"{other}\"}} 20034"
    )));
    assert!(text.contains(&format!(
        "sgl_router_decode_charged_requests{{worker_url=\"{other}\"}} 4"
    )));
    // Prefill workers are not decode candidates and carry no series.
    assert!(!text.contains(&format!(
        "sgl_router_decode_charged_tokens{{worker_url=\"{}\"}}",
        p.url
    )));

    // Streams end (2 × 1.5 s) → every charge released.
    for r in held {
        r.into_body().collect().await.unwrap();
    }
    wait_released(&wa, "heavy decode").await;
    wait_released(&wb, "light decode").await;
}

/// Every way a decode leg ends releases its charge: streaming and
/// non-streaming success, upstream 5xx, transport error, client abort
/// before response headers, and client disconnect mid-stream.
#[tokio::test]
async fn decode_charge_released_on_every_completion_path() {
    let p = MockWorker::start(vec!["data: [DONE]\n\n"]).await;

    // Success (chat route: no tokenizer chat encoder → chars/4 estimate +
    // the output cap, since the unparsed body has no visible max_tokens).
    let ok = MockWorker::start(vec!["data: {}\n\n", "data: [DONE]\n\n"]).await;
    let ctx = build_ctx(&p.url, &[&ok.url]);
    let app = build_router(Arc::clone(&ctx));
    let w = decode_worker(&ctx, &ok.url);
    let r = app.clone().oneshot(chat("hello", false)).await.unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    // Non-streaming: the charge drops before the handler returns.
    assert_eq!((w.decode_charged_tokens(), w.decode_charged_reqs()), (0, 0));
    r.into_body().collect().await.unwrap();
    let r = app.clone().oneshot(chat("hello", true)).await.unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    r.into_body().collect().await.unwrap();
    wait_released(&w, "streaming success").await;
    let r = app.clone().oneshot(completions(64, true)).await.unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    r.into_body().collect().await.unwrap();
    wait_released(&w, "completions streaming success").await;

    // Upstream 5xx, both modes.
    let err = MockWorker::start_returning_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        json!({"error": "boom"}),
    )
    .await;
    let ctx = build_ctx(&p.url, &[&err.url]);
    let app = build_router(Arc::clone(&ctx));
    let w = decode_worker(&ctx, &err.url);
    for stream in [false, true] {
        let r = app.clone().oneshot(chat("hello", stream)).await.unwrap();
        assert_eq!(r.status(), StatusCode::INTERNAL_SERVER_ERROR);
        r.into_body().collect().await.unwrap();
        wait_released(&w, "upstream 500").await;
    }

    // Transport error: nothing listens on port 1.
    let dead = "http://127.0.0.1:1";
    let ctx = build_ctx(&p.url, &[dead]);
    let app = build_router(Arc::clone(&ctx));
    let w = decode_worker(&ctx, dead);
    for stream in [false, true] {
        let r = app.clone().oneshot(chat("hello", stream)).await.unwrap();
        assert!(r.status().is_server_error(), "{}", r.status());
        wait_released(&w, "transport error").await;
    }

    // Client abort before response headers: the handler future is dropped
    // while the decode hangs.
    let hang = MockWorker::start_hanging(Duration::from_secs(30)).await;
    let ctx = build_ctx(&p.url, &[&hang.url]);
    let app = build_router(Arc::clone(&ctx));
    let w = decode_worker(&ctx, &hang.url);
    for stream in [false, true] {
        let task = tokio::spawn(app.clone().oneshot(chat("hello", stream)));
        wait_charged(&w, "hanging decode").await;
        assert!(w.decode_charged_tokens() > 8192, "chars/4 + output cap");
        task.abort();
        let _ = task.await;
        wait_released(&w, "client abort before headers").await;
    }

    // Client disconnect mid-stream: read one chunk, drop the body.
    let slow =
        MockWorker::start_slow_stream(vec!["data: {}\n\n"; 50], Duration::from_millis(100)).await;
    let ctx = build_ctx(&p.url, &[&slow.url]);
    let app = build_router(Arc::clone(&ctx));
    let w = decode_worker(&ctx, &slow.url);
    let r = app.clone().oneshot(chat("hello", true)).await.unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    let mut body = r.into_body();
    let first = body.frame().await.unwrap().unwrap();
    assert!(first.is_data());
    assert_eq!(w.decode_charged_reqs(), 1, "held while streaming");
    drop(body);
    // The pump notices on its next send (≤ one chunk interval).
    wait_released(&w, "client disconnect mid-stream").await;
}

/// The charge comes from the request: prompt token ids exactly, plus
/// `min(max_tokens, cap)` (the cap when absent); an unparsed chat body
/// charges its chars/4 estimate plus the cap.
#[tokio::test]
async fn charge_counts_prompt_ids_and_capped_max_tokens() {
    let p = MockWorker::start(vec!["data: [DONE]\n\n"]).await;
    let hang = MockWorker::start_hanging(Duration::from_secs(30)).await;
    let ctx = build_ctx(&p.url, &[&hang.url]);
    let app = build_router(Arc::clone(&ctx));
    let w = decode_worker(&ctx, &hang.url);
    let completions_with = |max_tokens: Option<u64>| {
        let prompt: Vec<u32> = (1..=1000).collect();
        let mut body = json!({"model": "tiny", "prompt": prompt});
        if let Some(m) = max_tokens {
            body["max_tokens"] = json!(m);
        }
        Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap()
    };
    for (max_tokens, expected) in [
        (Some(5), 1_005),
        (Some(1_000_000), 1_000 + 8_192),
        (None, 1_000 + 8_192),
    ] {
        let task = tokio::spawn(app.clone().oneshot(completions_with(max_tokens)));
        wait_charged(&w, "completions").await;
        assert_eq!(
            w.decode_charged_tokens(),
            expected,
            "max_tokens={max_tokens:?}"
        );
        task.abort();
        let _ = task.await;
        wait_released(&w, "completions abort").await;
    }

    let request = chat(&"y".repeat(40_000), false);
    let task = tokio::spawn(app.clone().oneshot(request));
    wait_charged(&w, "chat").await;
    let charged = w.decode_charged_tokens();
    // Body bytes / 4 (≥ 10k for 40k chars of content) + the 8192 cap.
    assert!(
        (10_000 + 8_192..10_100 + 8_192).contains(&charged),
        "{charged}"
    );
    task.abort();
    let _ = task.await;
    wait_released(&w, "chat abort").await;
}
