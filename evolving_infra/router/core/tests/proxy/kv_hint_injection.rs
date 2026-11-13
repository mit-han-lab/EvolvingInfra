// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! KV pull-hint injection — end-to-end at the HTTP layer with MockWorkers
//! and a mock dynamo indexer.
//!
//! Asserts the router-side contract of `--inject-kv-pull-hint`:
//!
//! * The hint (`kv_pull_hint` = `{source_worker_id, source_servicer_addr,
//!   expected_hit_tokens}`) is injected into the body of the EXECUTING
//!   worker: the prefill side in PD mode, the sole worker in plain mode.
//! * The PD decode body never carries a hint (decode receives KV via the
//!   bootstrap transfer, not peer pulls) while still carrying the
//!   bootstrap fields.
//! * Indexer failures fail OPEN: the request proxies normally, unhinted.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::post;
use axum::Json;
use bytes::Bytes;
use serde_json::{json, Value};
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::kv_hint::{KvHintConfig, KvWorkerMapEntry};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;
use tower::ServiceExt;

/// Minimal mock of the dynamo standalone indexer: answers every
/// POST /query with a fixed reply. Shuts down on Drop.
struct MockIndexer {
    url: String,
    _shutdown: oneshot::Sender<()>,
}

impl MockIndexer {
    async fn start(reply: Value) -> Self {
        let app = axum::Router::new().route(
            "/query",
            post(move |_body: Bytes| {
                let reply = reply.clone();
                async move { Json(reply) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (tx, rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = rx.await;
                })
                .await
                .unwrap();
        });
        Self { url, _shutdown: tx }
    }
}

fn config_with_hint(indexer_url: &str, worker_map: Vec<KvWorkerMapEntry>) -> Config {
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
            kv_hint: Some(KvHintConfig {
                indexer_url: indexer_url.into(),
                min_tokens: 256,
                min_incremental_tokens: 0,
                local_prefix_fallback: false,
                // Generous for a same-process mock; production default is 20.
                indexer_timeout_ms: 500,
                worker_map,
            }),
            pd_decode_policy: sgl_router::config::PdDecodePolicy::AffinityMinLoad,
            pd_decision_replay: None,
            pd_decode_tokens: None,
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

fn build_ctx(cfg: Config, specs: Vec<WorkerSpec>) -> Arc<AppContext> {
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for s in specs {
        let _ = registry.add(s);
    }
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

fn completions_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny",
                "prompt": [11, 12, 13, 14, 15, 16, 17, 18],
                "cache_salt": "sess-1",
                "max_tokens": 1,
            }))
            .unwrap(),
        ))
        .unwrap()
}

async fn await_captured_body(
    mock: &crate::common::mock_worker::MockWorker,
    timeout: Duration,
    label: &str,
) -> Bytes {
    let start = std::time::Instant::now();
    loop {
        let captured = mock.captured.lock().unwrap().last_body.clone();
        if let Some(b) = captured {
            return b;
        }
        if start.elapsed() > timeout {
            panic!("{label}: no request body captured within {timeout:?}");
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

fn parse_body(b: &Bytes) -> Value {
    serde_json::from_slice(b).expect("body must be valid JSON")
}

/// PD mode: the hint lands on the prefill body only; the decode body keeps
/// the bootstrap fields but never a hint.
#[tokio::test]
async fn pd_mode_hint_lands_on_prefill_body_only() {
    check_pd_hint(false).await;
}

#[tokio::test]
async fn pd_mode_equal_prefix_fallback_lands_on_prefill_body_only() {
    check_pd_hint(true).await;
}

async fn check_pd_hint(local_prefix_fallback: bool) {
    let prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    // The peer owner (indexer instance 2) is a worker the policy can never
    // select — only its map entry exists — so the hint decision is
    // deterministic regardless of selection.
    let indexer = MockIndexer::start(json!({
        "instances": {
            "1": {"gpu": 0, "cpu": if local_prefix_fallback {8192} else {0}},
            "2": {"gpu": 0, "cpu": 8192}
        }
    }))
    .await;
    let mut cfg = config_with_hint(
        &indexer.url,
        vec![
            KvWorkerMapEntry {
                id: 1,
                url: prefill.url.clone(),
                servicer: "p1:18100".into(),
            },
            KvWorkerMapEntry {
                id: 2,
                url: "http://owner:30000".into(),
                servicer: "owner:18100".into(),
            },
        ],
    );
    if local_prefix_fallback {
        let hint = cfg.model.kv_hint.as_mut().unwrap();
        hint.local_prefix_fallback = true;
        hint.min_incremental_tokens = 2048;
    }
    let ctx = build_ctx(
        cfg,
        vec![
            WorkerSpec {
                id: WorkerId("p1".into()),
                url: prefill.url.clone(),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: Some(8997),
            },
            WorkerSpec {
                id: WorkerId("d1".into()),
                url: decode.url.clone(),
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: None,
            },
        ],
    );
    let app = build_router(ctx);

    let res = app.oneshot(completions_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK, "decode side should 200");

    let prefill_body = await_captured_body(&prefill, Duration::from_secs(2), "prefill").await;
    let decode_body = await_captured_body(&decode, Duration::from_secs(2), "decode").await;
    let pj = parse_body(&prefill_body);
    let dj = parse_body(&decode_body);

    let hint = pj
        .get("kv_pull_hint")
        .expect("prefill body must carry kv_pull_hint");
    assert_eq!(hint["source_worker_id"], "http://owner:30000");
    if local_prefix_fallback {
        assert_eq!(hint["min_local_deficit_tokens"], 2048);
    } else {
        assert!(hint.get("min_local_deficit_tokens").is_none());
    }
    assert_eq!(hint["source_servicer_addr"], "owner:18100");
    assert_eq!(hint["expected_hit_tokens"], 8192);

    assert!(
        dj.get("kv_pull_hint").is_none(),
        "decode body must never carry a kv_pull_hint"
    );
    // Bootstrap fields still ride on both sides.
    assert!(pj.get("bootstrap_room").is_some());
    assert!(dj.get("bootstrap_room").is_some());
    // The prompt is forwarded untouched on both sides.
    assert_eq!(pj["prompt"], json!([11, 12, 13, 14, 15, 16, 17, 18]));
    assert_eq!(dj["prompt"], json!([11, 12, 13, 14, 15, 16, 17, 18]));
}

/// Plain mode: the sole worker executes the prefill, so its body carries
/// the hint.
#[tokio::test]
async fn plain_mode_hint_lands_on_single_body() {
    let plain = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let indexer = MockIndexer::start(json!({
        "instances": {"2": {"cpu": 4096}}
    }))
    .await;
    let cfg = config_with_hint(
        &indexer.url,
        vec![
            KvWorkerMapEntry {
                id: 1,
                url: plain.url.clone(),
                servicer: "w1:18100".into(),
            },
            KvWorkerMapEntry {
                id: 2,
                url: "http://owner:30000".into(),
                servicer: "owner:18100".into(),
            },
        ],
    );
    let ctx = build_ctx(
        cfg,
        vec![WorkerSpec {
            id: WorkerId("w1".into()),
            url: plain.url.clone(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }],
    );
    let app = build_router(ctx);

    let res = app.oneshot(completions_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let body = await_captured_body(&plain, Duration::from_secs(2), "plain").await;
    let j = parse_body(&body);
    let hint = j
        .get("kv_pull_hint")
        .expect("plain-mode body must carry kv_pull_hint");
    assert_eq!(hint["source_worker_id"], "http://owner:30000");
    // No PD fields in plain mode.
    assert!(j.get("bootstrap_room").is_none());
}

/// Indexer unreachable → fail open: the request proxies normally with no
/// hint injected.
#[tokio::test]
async fn unreachable_indexer_fails_open() {
    let plain = crate::common::mock_worker::MockWorker::start(vec![]).await;
    // A bound, never-listening port — a refused TCP destination for the indexer.
    let (_dead, dead_addr) = crate::common::refused_addr();
    let dead_url = format!("http://{dead_addr}");
    let cfg = config_with_hint(
        &dead_url,
        vec![KvWorkerMapEntry {
            id: 1,
            url: plain.url.clone(),
            servicer: "w1:18100".into(),
        }],
    );
    let ctx = build_ctx(
        cfg,
        vec![WorkerSpec {
            id: WorkerId("w1".into()),
            url: plain.url.clone(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }],
    );
    let app = build_router(ctx);

    let res = app.oneshot(completions_request()).await.unwrap();
    assert_eq!(
        res.status(),
        StatusCode::OK,
        "hint failure must not break the request"
    );

    let body = await_captured_body(&plain, Duration::from_secs(2), "plain").await;
    let j = parse_body(&body);
    assert!(j.get("kv_pull_hint").is_none());
    assert_eq!(j["prompt"], json!([11, 12, 13, 14, 15, 16, 17, 18]));
}

/// A Dynamo match affects placement only, and the Python identity survives both PD legs.
#[tokio::test]
async fn dynamo_cache_aware_routes_to_owner_without_cross_p_hint() {
    let p0 = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let p1 = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let d = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let index =
        MockIndexer::start(json!({"instances":{"1":{"gpu":0},"2":{"cpu":128,"gpu":64}}})).await;
    let mut cfg = config_with_hint(
        &index.url,
        vec![
            KvWorkerMapEntry {
                id: 1,
                url: p0.url.clone(),
                servicer: String::new(),
            },
            KvWorkerMapEntry {
                id: 2,
                url: p1.url.clone(),
                servicer: String::new(),
            },
        ],
    );
    cfg.model.policy = PolicyKind::CacheAwareZmq;
    cfg.model.cache_aware = Some(sgl_router::config::CacheAwareConfig {
        dynamo: cfg.model.kv_hint.take(),
        ..Default::default()
    });
    cfg.model.pd_decode_policy = sgl_router::config::PdDecodePolicy::RoundRobin;
    let ctx = build_ctx(
        cfg,
        vec![
            WorkerSpec {
                id: WorkerId("p0".into()),
                url: p0.url.clone(),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: Some(8997),
            },
            WorkerSpec {
                id: WorkerId("p1".into()),
                url: p1.url.clone(),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: Some(8998),
            },
            WorkerSpec {
                id: WorkerId("d".into()),
                url: d.url.clone(),
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: None,
            },
        ],
    );
    ctx.block_size_oracle.try_set(64).unwrap();
    ctx.block_size_oracle.set_bigram(true);
    assert!(ctx.kv_hint.is_none() && ctx.symmetric_pd.is_none());
    let request = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .header(
            "x-agent-context",
            r#"{"agent_id":"question/\u4f60\u597d","subagent_id":2,"request_id":7}"#,
        )
        .body(Body::from(
            serde_json::to_vec(&json!({"model":"tiny","prompt":vec![11;129],
            "max_tokens":1,"cache_salt":"shared"}))
            .unwrap(),
        ))
        .unwrap();
    let response = build_router(ctx).oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    for (worker, name) in [(&p1, "owner"), (&d, "decode")] {
        let body = parse_body(&await_captured_body(worker, Duration::from_secs(2), name).await);
        assert!(body.get("kv_pull_hint").is_none());
        assert_eq!(
            body["rid"],
            "agent_e471509b10f0c301d8c86761c428ff065faf2d6d9ad0e8939059aa277dda0759"
        );
        assert_eq!(body["cache_salt"], "shared");
    }
    assert!(p0.captured.lock().unwrap().last_body.is_none());
}

#[tokio::test]
async fn header_only_identity_is_forwarded_in_plain_mode() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let mut cfg = config_with_hint("http://unused:1", vec![]);
    cfg.model.kv_hint = None;
    let ctx = build_ctx(
        cfg,
        vec![WorkerSpec {
            id: WorkerId("plain".into()),
            url: worker.url.clone(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }],
    );
    let request = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .header(
            "x-agent-context",
            r#"{"agent_id":"q","subagent_id":0,"request_id":0}"#,
        )
        .body(Body::from(
            r#"{"model":"tiny","prompt":[1,2],"max_tokens":1}"#,
        ))
        .unwrap();
    assert_eq!(
        build_router(ctx).oneshot(request).await.unwrap().status(),
        StatusCode::OK
    );
    let body = parse_body(&await_captured_body(&worker, Duration::from_secs(2), "plain").await);
    assert!(body["rid"].as_str().unwrap().starts_with("agent_"));
}
