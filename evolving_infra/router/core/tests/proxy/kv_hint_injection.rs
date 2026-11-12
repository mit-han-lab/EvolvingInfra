// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
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
            co_schedule: None,
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

/// The D HTTP response must not release P work. A real HTTP load snapshot
/// updates that reservation, and the next arrival must see the updated value.
#[tokio::test]
async fn uncached_progress_updates_p_owned_work_without_waiting_for_p_http_end() {
    verify_p_owned_progress(None).await;
}

#[tokio::test]
async fn restore_cost_progress_updates_without_a_dispatch_gate() {
    verify_p_owned_progress(Some(8)).await;
}

async fn verify_p_owned_progress(restore_cost_divisor: Option<u64>) {
    use sgl_router::config::{CoScheduleConfig, KvPullScope};
    use std::sync::atomic::{AtomicU64, Ordering};
    let release = Arc::new(tokio::sync::Notify::new());
    let remaining = Arc::new(AtomicU64::new(20000));
    let sequence = Arc::new(AtomicU64::new(0));
    let worker_app = axum::Router::new()
        .route("/v1/completions", post({
            let release = release.clone();
            move |Json(body): Json<Value>| {
                let release = release.clone();
                async move {
                    assert!(body["rid"].as_str().is_some_and(|rid| !rid.is_empty()));
                    if body["rid"] == "first" { release.notified().await; }
                    Json(json!({"choices": []}))
                }
            }
        }))
        .route("/v1/loads", axum::routing::get({
            let remaining = remaining.clone();
            move || {
                let remaining = remaining.clone();
                let sequence = sequence.clone();
                async move { Json(json!({"loads": [{
                    "num_running_reqs": 1,
                    "prefill_work": {"epoch": "p1", "sequence": sequence.fetch_add(1, Ordering::SeqCst),
                        "remaining": {"first": remaining.load(Ordering::SeqCst)}}
                }]})) }
            }
        }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let p_url = format!("http://{}", listener.local_addr().unwrap());
    let p_task = tokio::spawn(async move {
        axum::serve(listener, worker_app).await.unwrap();
    });
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let indexer = MockIndexer::start(json!({"instances": {}})).await;
    let tmp = tempfile::tempdir().unwrap();
    let trace = tmp.path().join("work.jsonl");
    let mut cfg = config_with_hint(
        &indexer.url,
        vec![KvWorkerMapEntry {
            id: 1,
            url: p_url.clone(),
            servicer: "p1:18100".into(),
        }],
    );
    cfg.model.policy = PolicyKind::CoSchedule;
    cfg.model.co_schedule = Some(CoScheduleConfig {
        on_demand_load_refresh: false,
        uncached_balance: true,
        owner_threshold_tokens: 12288,
        detour_margin_tokens: 0,
        restore_cost_divisor,
        trade: None,
        trade_eft_model: None,
        global_queue: None,
        trade_e3: None,
        prefer_gpu_hit: false,
        locality_margin_tokens: None,
        cost_balance: false,
        cost_trace: Some(trace.clone()),
        max_uncached_tokens: 1,
        max_queue_wait_ms: 30000,
        load_poll_ms: 10,
        tick_ms: 30000,
        poll_on_hold_ms: 0,
        imbalance_spread_tokens: 0,
        pull_scope: KvPullScope::Any,
        kv_session_retention: false,
        kv_session_key_file: None,
    });
    let app = build_router(build_ctx(
        cfg,
        vec![
            WorkerSpec {
                id: WorkerId("p1".into()),
                url: p_url,
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
    ));
    let request = |rid: &str| {
        Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&json!({
                    "model": "tiny", "prompt": vec![11; 20000], "max_tokens": 1, "rid": rid,
                }))
                .unwrap(),
            ))
            .unwrap()
    };
    let first = tokio::time::timeout(
        Duration::from_secs(2),
        app.clone().oneshot(request("first")),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(first.status(), StatusCode::OK);
    let read_events = || {
        std::fs::read_to_string(&trace)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>()
    };
    assert!(!read_events()
        .iter()
        .any(|e| e["event"] == "finished" && e["entry"]["rid"] == "first"));
    remaining.store(8000, Ordering::SeqCst);
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if read_events().iter().any(|e| {
                e["event"] == "prefill_progress"
                    && e["updates"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|u| u["remaining_after"] == 8000)
            }) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    let second = tokio::time::timeout(
        Duration::from_secs(2),
        app.clone().oneshot(request("second")),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(second.status(), StatusCode::OK);
    let events = read_events();
    let reservation = events
        .iter()
        .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == "second")
        .unwrap();
    assert_eq!(
        reservation["candidate_scores"][0]["worker_cost_before"],
        (8000 * restore_cost_divisor.unwrap_or(1)).to_string()
    );
    assert_eq!(
        reservation["cost_unit"],
        if restore_cost_divisor.is_some() {
            "weighted_uncached_and_restore_tokens"
        } else {
            "uncached_tokens"
        }
    );
    assert_eq!(
        reservation["restore_cost_divisor"],
        json!(restore_cost_divisor)
    );
    release.notify_one();
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if read_events()
                .iter()
                .any(|e| e["event"] == "finished" && e["entry"]["rid"] == "first")
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    let generated = app.clone().oneshot(completions_request()).await.unwrap();
    assert_eq!(generated.status(), StatusCode::OK);
    let decode_body =
        await_captured_body(&decode, Duration::from_secs(2), "decode generated RID").await;
    let generated_body = parse_body(&decode_body);
    let generated_rid = generated_body["rid"].as_str().unwrap();
    assert!(generated_rid.starts_with("rid_") && generated_rid.len() == 36);
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if read_events()
                .iter()
                .any(|e| e["event"] == "finished" && e["entry"]["rid"] == generated_rid)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    p_task.abort();
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
    // Bind a port, drop it — a refused TCP destination for the indexer.
    let dead = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let dead_url = format!("http://{}", dead.local_addr().unwrap());
    drop(dead);
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
    assert!(ctx.kv_hint.is_none() && ctx.co_schedule.is_none() && ctx.symmetric_pd.is_none());
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

/// One mock P engine for the TRADE test: records every P-leg body, holds a
/// request whose rid is `busy` until released (reporting 20,000 remaining
/// uncached tokens for it in `/v1/loads`), and answers others at once.
struct TradeP {
    url: String,
    bodies: Arc<std::sync::Mutex<Vec<Value>>>,
    release: Arc<tokio::sync::Notify>,
}

impl TradeP {
    async fn start(name: &'static str) -> Self {
        use std::collections::BTreeMap;
        use std::sync::atomic::{AtomicU64, Ordering};
        let bodies: Arc<std::sync::Mutex<Vec<Value>>> = Arc::default();
        let release = Arc::new(tokio::sync::Notify::new());
        let held: Arc<std::sync::Mutex<BTreeMap<String, u64>>> = Arc::default();
        let sequence = Arc::new(AtomicU64::new(0));
        let app = axum::Router::new()
            .route(
                "/v1/completions",
                post({
                    let (bodies, release, held) = (bodies.clone(), release.clone(), held.clone());
                    move |Json(body): Json<Value>| {
                        let (bodies, release, held) =
                            (bodies.clone(), release.clone(), held.clone());
                        async move {
                            let rid = body["rid"].as_str().unwrap_or("").to_string();
                            bodies.lock().unwrap().push(body);
                            if rid == "busy" {
                                held.lock().unwrap().insert(rid.clone(), 20_000);
                                release.notified().await;
                                held.lock().unwrap().remove(&rid);
                            }
                            Json(json!({"choices": []}))
                        }
                    }
                }),
            )
            .route(
                "/v1/loads",
                axum::routing::get(move || {
                    let held = held.lock().unwrap().clone();
                    let n = sequence.fetch_add(1, Ordering::SeqCst);
                    async move {
                        Json(json!({"loads": [{"num_running_reqs": held.len(),
                            "prefill_work": {"epoch": name, "sequence": n, "remaining": held}}]}))
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self {
            url,
            bodies,
            release,
        }
    }

    fn body(&self, rid: &str) -> Option<Value> {
        self.bodies
            .lock()
            .unwrap()
            .iter()
            .find(|b| b["rid"] == rid)
            .cloned()
    }
}

/// Indexer whose reply the test can change between requests.
async fn switchable_indexer(reply: Arc<std::sync::Mutex<Value>>) -> String {
    let app = axum::Router::new().route(
        "/query",
        post(move |_b: Bytes| {
            let reply = reply.lock().unwrap().clone();
            async move { Json(reply) }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    url
}

/// TRADE through the HTTP path with three P engines (P1, P2, P3) and a
/// 20,000-token prompt. The indexer reports P1 = 18,000 cached tokens and
/// P3 = 300 (P2 none); the pull floor is 512.
///
/// 1. `busy`: all idle -> P1 (cost 2,000, no hint). P1 holds it and reports
///    20,000 remaining, so P1's backlog is 20,000.
/// 2. `pull`: P1 costs 22,000. lambda 0: P3 pulls 17,700 tokens from P1
///    (2,000 + 13,500 x 0.4677 s = 8,314 < P2's 8,318) and its P body
///    carries the hint. lambda 9.2: the copy price keeps `pull` on P1, unhinted.
/// 3. (lambda 0 only) The indexer now reports P3 = 17,600: the gain from P1
///    is 400 < 512, so `small` moves to P3 WITHOUT a hint (cost 2,400).
///
/// `gate` is the injector's `min_incremental_tokens`. With 0, pick() itself
/// returns hints for the owner (source P3, 300 < own 18,000) and for the
/// 400-token gain; only TRADE's own floor (512) drops them, so the `busy`
/// and `small` bodies prove that filter.
#[tokio::test]
async fn trade_three_p_moves_with_a_pull_only_when_it_pays() {
    trade_three_p(0.0, 512).await;
}

#[tokio::test]
async fn trade_three_p_copy_price_keeps_the_loaded_owner() {
    trade_three_p(9.2, 512).await;
}

#[tokio::test]
async fn trade_three_p_drops_sub_floor_hints_when_the_injector_gate_is_off() {
    trade_three_p(0.0, 0).await;
}

async fn trade_three_p(lambda: f64, gate: u32) {
    use sgl_router::config::{CoScheduleConfig, KvPullScope};
    use sgl_router::scheduler::co_schedule::TradeParams;
    let ps = [
        TradeP::start("p1").await,
        TradeP::start("p2").await,
        TradeP::start("p3").await,
    ];
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let reply = Arc::new(std::sync::Mutex::new(
        json!({"instances": {"1": {"cpu": 18000}, "3": {"cpu": 300}}}),
    ));
    let indexer = switchable_indexer(reply.clone()).await;
    let tmp = tempfile::tempdir().unwrap();
    let trace = tmp.path().join("trade.jsonl");
    let map = ps
        .iter()
        .enumerate()
        .map(|(i, p)| KvWorkerMapEntry {
            id: i as u64 + 1,
            url: p.url.clone(),
            servicer: format!("p{}:18100", i + 1),
        })
        .collect();
    let mut cfg = config_with_hint(&indexer, map);
    cfg.model.kv_hint.as_mut().unwrap().min_incremental_tokens = gate;
    cfg.model.policy = PolicyKind::CoSchedule;
    cfg.model.co_schedule = Some(CoScheduleConfig {
        on_demand_load_refresh: false,
        uncached_balance: true,
        owner_threshold_tokens: 12288,
        detour_margin_tokens: 0,
        restore_cost_divisor: None,
        trade: Some(TradeParams {
            copy_lambda: lambda,
            ..TradeParams::default()
        }),
        trade_eft_model: None,
        global_queue: None,
        trade_e3: None,
        prefer_gpu_hit: false,
        locality_margin_tokens: None,
        cost_balance: false,
        cost_trace: Some(trace.clone()),
        max_uncached_tokens: 1,
        max_queue_wait_ms: 30000,
        load_poll_ms: 10,
        tick_ms: 30000,
        poll_on_hold_ms: 0,
        imbalance_spread_tokens: 0,
        pull_scope: KvPullScope::Any,
        kv_session_retention: false,
        kv_session_key_file: None,
    });
    let mut specs: Vec<WorkerSpec> = ps
        .iter()
        .enumerate()
        .map(|(i, p)| WorkerSpec {
            id: WorkerId(format!("p{}", i + 1)),
            url: p.url.clone(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: Some(8997),
        })
        .collect();
    specs.push(WorkerSpec {
        id: WorkerId("d1".into()),
        url: decode.url.clone(),
        mode: WorkerMode::Decode,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: None,
    });
    let app = build_router(build_ctx(cfg, specs));
    let read_events = || -> Vec<Value> {
        std::fs::read_to_string(&trace)
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    };
    let wait_for = |what: &'static str, check: Box<dyn Fn(&[Value]) -> bool>| {
        let read_events = &read_events;
        async move {
            tokio::time::timeout(Duration::from_secs(3), async {
                while !check(&read_events()) {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .unwrap_or_else(|_| panic!("timed out waiting for {what}"));
        }
    };
    let send = |rid: &'static str| {
        let app = app.clone();
        async move {
            let request = Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "model": "tiny", "prompt": vec![11; 20000], "max_tokens": 1, "rid": rid,
                    }))
                    .unwrap(),
                ))
                .unwrap();
            let res = tokio::time::timeout(Duration::from_secs(3), app.oneshot(request))
                .await
                .unwrap()
                .unwrap();
            assert_eq!(res.status(), StatusCode::OK, "{rid}");
        }
    };
    let decision = |rid: &str| -> Value {
        read_events()
            .into_iter()
            .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == rid)
            .unwrap_or_else(|| panic!("no reserve event for {rid}"))["trade_decision"]
            .clone()
    };
    let landed_on = |rid: &'static str| {
        let ps = &ps;
        async move {
            tokio::time::timeout(Duration::from_secs(3), async {
                loop {
                    let hits: Vec<usize> = (0..3).filter(|i| ps[*i].body(rid).is_some()).collect();
                    if !hits.is_empty() {
                        assert_eq!(hits.len(), 1, "{rid} reached several P");
                        return (hits[0], ps[hits[0]].body(rid).unwrap());
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .unwrap_or_else(|_| panic!("{rid} reached no P"))
        }
    };

    // 1. Idle pool: the owner P1 takes `busy` without a hint.
    send("busy").await;
    let (p, body) = landed_on("busy").await;
    assert_eq!(p, 0);
    assert!(
        body.get("kv_pull_hint").is_none(),
        "the owner never gets a hint (gate {gate})"
    );
    let d = decision("busy");
    assert_eq!(d["chosen"], ps[0].url.as_str());
    assert_eq!(d["owner"], ps[0].url.as_str());
    assert_eq!(d["pulled_tokens"], 0);
    // P1 reports 20,000 remaining for `busy`: its acknowledged backlog.
    wait_for(
        "busy progress",
        Box::new(|events: &[Value]| {
            events.iter().any(|e| {
                e["event"] == "prefill_progress"
                    && e["updates"]
                        .as_array()
                        .is_some_and(|u| u.iter().any(|u| u["remaining_after"] == 20000))
            })
        }),
    )
    .await;

    // 2. The owner is loaded.
    send("pull").await;
    let (p, body) = landed_on("pull").await;
    let d = decision("pull");
    assert_eq!(d["owner"], ps[0].url.as_str());
    let row = |d: &Value, i: usize| -> Value {
        d["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["worker_url"] == ps[i].url.as_str())
            .cloned()
            .unwrap()
    };
    assert_eq!(row(&d, 0)["backlog_tokens"], "20000");
    assert_eq!(row(&d, 0)["trade_cost"], 22000.0);
    assert_eq!(row(&d, 2)["pulled_tokens"], 17700);
    assert_eq!(row(&d, 1)["pulled_tokens"], 18000);
    if lambda == 0.0 {
        assert_eq!(p, 2, "P3 pulls: 8,314 < P2 8,318 < P1 22,000");
        assert_eq!(d["detour"], true);
        assert_eq!(d["pulled_tokens"], 17700);
        let hint = body.get("kv_pull_hint").expect("P3 body carries the hint");
        assert_eq!(hint["source_worker_id"], ps[0].url.as_str());
        assert_eq!(hint["expected_hit_tokens"], 18000);
    } else {
        assert_eq!(p, 0, "the copy price keeps the request on the owner");
        assert_eq!(d["detour"], false);
        assert_eq!(d["pulled_tokens"], 0);
        assert!(body.get("kv_pull_hint").is_none());
    }
    // The decode body never carries a hint.
    let decode_body = await_captured_body(&decode, Duration::from_secs(2), "decode").await;
    assert!(parse_body(&decode_body).get("kv_pull_hint").is_none());

    if lambda == 0.0 {
        // 3. Below the floor: move without a pull.
        wait_for(
            "pull finished",
            Box::new(|events: &[Value]| {
                events
                    .iter()
                    .any(|e| e["event"] == "finished" && e["entry"]["rid"] == "pull")
            }),
        )
        .await;
        *reply.lock().unwrap() = json!({"instances": {"1": {"cpu": 18000}, "3": {"cpu": 17600}}});
        send("small").await;
        let (p, body) = landed_on("small").await;
        assert_eq!(p, 2, "P3 recomputes 2,400 tokens instead of waiting on P1");
        assert!(
            body.get("kv_pull_hint").is_none(),
            "gain 400 < floor 512 (gate {gate})"
        );
        let d = decision("small");
        assert_eq!(d["detour"], true);
        assert_eq!(d["pulled_tokens"], 0);
        assert_eq!(row(&d, 2)["trade_cost"], 2400.0);
    }
    ps[0].release.notify_one();
    wait_for(
        "busy finished",
        Box::new(|events: &[Value]| {
            events
                .iter()
                .any(|e| e["event"] == "finished" && e["entry"]["rid"] == "busy")
        }),
    )
    .await;
}

/// One mock P engine for the TRADE wait-eft test. `/v1/loads` returns an
/// empty prefill-work ledger (the router ledger sees no backlog) and, only
/// when the poll asks for it, the `prefill_routing` snapshot the test sets.
/// Each poll gets a new sequence; the include strings are recorded.
struct EftP {
    url: String,
    bodies: Arc<std::sync::Mutex<Vec<Value>>>,
    snapshot: Arc<std::sync::Mutex<Value>>,
    includes: Arc<std::sync::Mutex<Vec<String>>>,
}

impl EftP {
    async fn start(name: &'static str, snapshot: Value) -> Self {
        Self::start_with(name, snapshot, false).await
    }

    /// `reject_routing`: an engine without the `prefill_routing` section
    /// (an older engine). A poll that asks for the section gets
    /// HTTP 500, as from that engine's load-snapshot section check.
    async fn start_with(name: &'static str, snapshot: Value, reject_routing: bool) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        let bodies: Arc<std::sync::Mutex<Vec<Value>>> = Arc::default();
        let snapshot = Arc::new(std::sync::Mutex::new(snapshot));
        let includes: Arc<std::sync::Mutex<Vec<String>>> = Arc::default();
        let sequence = Arc::new(AtomicU64::new(0));
        let app = axum::Router::new()
            .route(
                "/v1/completions",
                post({
                    let bodies = bodies.clone();
                    move |Json(body): Json<Value>| {
                        let bodies = bodies.clone();
                        async move {
                            bodies.lock().unwrap().push(body);
                            Json(json!({"choices": []}))
                        }
                    }
                }),
            )
            .route(
                "/v1/loads",
                axum::routing::get({
                    let (snapshot, includes) = (snapshot.clone(), includes.clone());
                    move |axum::extract::Query(query): axum::extract::Query<
                        std::collections::HashMap<String, String>,
                    >| {
                        let include = query.get("include").cloned().unwrap_or_default();
                        includes.lock().unwrap().push(include.clone());
                        let asks = include.split(',').any(|s| s == "prefill_routing");
                        let n = sequence.fetch_add(1, Ordering::SeqCst) + 1;
                        let mut routing = snapshot.lock().unwrap().clone();
                        routing["sequence"] = json!(n);
                        async move {
                            if reject_routing && asks {
                                return axum::response::IntoResponse::into_response((
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    "Invalid include sections: {'prefill_routing'}",
                                ));
                            }
                            let mut load = json!({"num_running_reqs": 0,
                                "prefill_work": {"epoch": name, "sequence": n, "remaining": {}}});
                            if asks {
                                load["prefill_routing"] = routing;
                            }
                            axum::response::IntoResponse::into_response(Json(
                                json!({"loads": [load]}),
                            ))
                        }
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self {
            url,
            bodies,
            snapshot,
            includes,
        }
    }

    fn body(&self, rid: &str) -> Option<Value> {
        self.bodies
            .lock()
            .unwrap()
            .iter()
            .find(|b| b["rid"] == rid)
            .cloned()
    }
}

/// A `prefill_routing` snapshot with `rows` waiting rows of
/// `tokens` uncached tokens each.
fn routing_snapshot(epoch: &str, rows: usize, tokens: u64) -> Value {
    let waiting: Vec<Value> = (0..rows)
        .map(|i| {
            json!({"rid": format!("{epoch}-w{i}"), "phase": "waiting", "input_tokens": tokens,
                "remaining_compute_tokens": tokens, "device_prefix_tokens": 0,
                "host_prefix_tokens": 0, "storage_prefix_tokens": 0,
                "plan": {"pull_tokens": null},
                "state": {"intent_start_ok": true, "intent_received": true, "bound": false,
                    "launched_unbound": false, "gate_blocked": false, "held": false,
                    "parked": false, "pull_pending": false, "pull_inflight": false,
                    "restore_inflight": false}})
        })
        .collect();
    json!({"schema_version": 1, "epoch": epoch, "sequence": 0, "available": true,
        "reason": null, "age_s": 0.0, "stale": false, "max_age_s": 1.0,
        "truncated": false, "total_waiting": rows, "omitted_waiting": 0,
        "engine_revision": "mock",
        "policy": {"page_size": 64, "chunk_tokens": 8192, "max_reqs": 256,
            "schedule_policy": "fcfs", "share_enabled": false},
        "partial": null, "waiting": waiting, "bootstrap": [],
        "inflight_batches_known": true, "inflight_batches": [], "inflight_requests": [],
        "admission_resources": {"token_pool_capacity": 2067584,
            "unbound_prompt_tokens_last_scan": 0, "park_gate_closed_last_scan": false}})
}

/// TRADE `--co-schedule-trade-wait eft` through the HTTP path with three
/// mock P engines and a 20,000-token prompt; P1 caches 18,000 tokens; the
/// copy price is 0.
///
/// 1. P1's ledger is empty, but its engine snapshot holds five waiting rows
///    of 65,536 tokens (B_w = 5.4368 s). EFT prices P1 at about
///    13,500 x 5.51 s = 74,400 and an idle peer that pulls 17,984 tokens at
///    13,500 x (0.078 + 0.468) s = 7,368, so `eft1` moves with a hint. The
///    ledger wait would keep it on P1 (2,000).
/// 2. P1's snapshot turns old (engine age 2 s > 1.5 s): P1 falls back to
///    the router's own rows priced by the step model (none, so B_w = 0) and
///    its step-model S_w (13,500 x 0.076 s = 1,025 < 7,368), `eft2` stays on
///    P1, and the trace names and counts the fallback. The mock engine's
///    queue is not router traffic, so the fallback does not see it.
///
/// The polls ask for `prefill_routing`, and the trace's `engine_load` does
/// not carry it.
#[tokio::test]
async fn trade_wait_eft_prices_the_engine_queue_and_falls_back_on_an_old_snapshot() {
    use sgl_router::config::{CoScheduleConfig, KvPullScope};
    use sgl_router::scheduler::co_schedule::{TradeParams, TradeWait};
    let ps = [
        EftP::start("p1", routing_snapshot("r1", 5, 65_536)).await,
        EftP::start("p2", routing_snapshot("r2", 0, 0)).await,
        EftP::start("p3", routing_snapshot("r3", 0, 0)).await,
    ];
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let indexer = MockIndexer::start(json!({"instances": {"1": {"cpu": 18000}}})).await;
    let tmp = tempfile::tempdir().unwrap();
    let trace = tmp.path().join("trade-eft.jsonl");
    let map = ps
        .iter()
        .enumerate()
        .map(|(i, p)| KvWorkerMapEntry {
            id: i as u64 + 1,
            url: p.url.clone(),
            servicer: format!("p{}:18100", i + 1),
        })
        .collect();
    let model = sgl_router::eft_cheap::EftModel::from_json(
        &json!({"models": {"augmented": {"launch_interval": {"global":
            {"coefficients": [0.05, 1e-5, 0.0, 0.004, 1e-7, 0.0]}}}},
            "eft": {"tau_s": 0.02}}),
        Some("proxy-model".into()),
    )
    .unwrap();
    let mut cfg = config_with_hint(&indexer.url, map);
    cfg.model.kv_hint.as_mut().unwrap().min_incremental_tokens = 512;
    cfg.model.policy = PolicyKind::CoSchedule;
    cfg.model.co_schedule = Some(CoScheduleConfig {
        on_demand_load_refresh: false,
        uncached_balance: true,
        owner_threshold_tokens: 12288,
        detour_margin_tokens: 0,
        restore_cost_divisor: None,
        trade: Some(TradeParams {
            copy_lambda: 0.0,
            wait: TradeWait::Eft,
            ..TradeParams::default()
        }),
        trade_eft_model: Some(Arc::new(model)),
        global_queue: None,
        trade_e3: None,
        prefer_gpu_hit: false,
        locality_margin_tokens: None,
        cost_balance: false,
        cost_trace: Some(trace.clone()),
        max_uncached_tokens: 1,
        max_queue_wait_ms: 30000,
        load_poll_ms: 10,
        tick_ms: 30000,
        poll_on_hold_ms: 0,
        imbalance_spread_tokens: 0,
        pull_scope: KvPullScope::Any,
        kv_session_retention: false,
        kv_session_key_file: None,
    });
    let mut specs: Vec<WorkerSpec> = ps
        .iter()
        .enumerate()
        .map(|(i, p)| WorkerSpec {
            id: WorkerId(format!("p{}", i + 1)),
            url: p.url.clone(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: Some(8997),
        })
        .collect();
    specs.push(WorkerSpec {
        id: WorkerId("d1".into()),
        url: decode.url.clone(),
        mode: WorkerMode::Decode,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: None,
    });
    let app = build_router(build_ctx(cfg, specs));
    let read_events = || -> Vec<Value> {
        std::fs::read_to_string(&trace)
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    };
    let progress_count = |url: &str| {
        read_events()
            .iter()
            .filter(|e| e["event"] == "prefill_progress" && e["worker_url"] == url)
            .count()
    };
    // Two applied polls per P: the first poll's snapshot is published.
    let wait_for_polls = |min: Vec<usize>| {
        let ps = &ps;
        let progress_count = &progress_count;
        async move {
            tokio::time::timeout(Duration::from_secs(3), async {
                while (0..3).any(|i| progress_count(&ps[i].url) < min[i]) {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .expect("progress polls");
        }
    };
    wait_for_polls(vec![2, 2, 2]).await;
    let send = |rid: &'static str| {
        let app = app.clone();
        async move {
            let request = Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "model": "tiny", "prompt": vec![11; 20000], "max_tokens": 1, "rid": rid,
                    }))
                    .unwrap(),
                ))
                .unwrap();
            let res = tokio::time::timeout(Duration::from_secs(3), app.oneshot(request))
                .await
                .unwrap()
                .unwrap();
            assert_eq!(res.status(), StatusCode::OK, "{rid}");
        }
    };
    let decision = |rid: &str| -> Value {
        read_events()
            .into_iter()
            .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == rid)
            .unwrap_or_else(|| panic!("no reserve event for {rid}"))["trade_decision"]
            .clone()
    };
    let row = |d: &Value, i: usize| -> Value {
        d["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["worker_url"] == ps[i].url.as_str())
            .cloned()
            .unwrap()
    };
    let landed_on = |rid: &'static str| {
        let ps = &ps;
        async move {
            tokio::time::timeout(Duration::from_secs(3), async {
                loop {
                    let hits: Vec<usize> = (0..3).filter(|i| ps[*i].body(rid).is_some()).collect();
                    if !hits.is_empty() {
                        assert_eq!(hits.len(), 1, "{rid} reached several P");
                        return (hits[0], ps[hits[0]].body(rid).unwrap());
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .unwrap_or_else(|_| panic!("{rid} reached no P"))
        }
    };

    // The polls ask for the snapshot; the trace does not store it.
    for p in &ps {
        let includes = p.includes.lock().unwrap().clone();
        let progress: Vec<_> = includes
            .iter()
            .filter(|i| i.contains("prefill_work"))
            .collect();
        assert!(!progress.is_empty());
        assert!(
            progress.iter().all(|i| i.contains("prefill_routing")),
            "every progress poll asks for prefill_routing: {includes:?}"
        );
    }
    for e in read_events()
        .iter()
        .filter(|e| e["event"] == "prefill_progress")
    {
        assert!(e["engine_load"].get("prefill_routing").is_none(), "{e}");
        assert!(e["engine_load"].get("num_running_reqs").is_some());
    }

    // 1. P1's engine queue is long although its ledger is empty: move with a pull.
    send("eft1").await;
    let (p, body) = landed_on("eft1").await;
    assert_ne!(
        p, 0,
        "an idle peer pulls instead of waiting behind P1's queue"
    );
    let hint = body
        .get("kv_pull_hint")
        .expect("the pulling P carries the hint");
    assert_eq!(hint["source_worker_id"], ps[0].url.as_str());
    let d = decision("eft1");
    assert_eq!(d["wait"], "eft");
    assert_eq!(d["owner"], ps[0].url.as_str());
    assert_eq!(d["detour"], true);
    assert_eq!(d["wait_fallbacks"], 0);
    assert_eq!(d["eft_model_sha256"], "proxy-model");
    let p1 = row(&d, 0);
    assert_eq!(p1["wait_source"], "eft");
    assert_eq!(p1["backlog_tokens"], "0", "the ledger sees no backlog");
    let wait = p1["wait_s"].as_f64().unwrap();
    assert!(wait > 5.0 && wait <= 5.4368 + 1e-9, "B_w of P1: {wait}");
    let chosen = row(&d, p);
    assert_eq!(chosen["wait_source"], "eft");
    assert_eq!(chosen["wait_s"], 0.0);
    assert_eq!(chosen["pulled_tokens"], 18000);
    assert!(chosen["exposed_pull_s"].as_f64().unwrap() > 0.4);

    // 2. P1's snapshot is 2 s old at the engine: P1 uses the fallback.
    {
        let mut snapshot = ps[0].snapshot.lock().unwrap();
        snapshot["age_s"] = json!(2.0);
    }
    // +3 applied polls: the third started after the change and after the
    // second poll's publish (each round waits for the previous one).
    let seen: Vec<usize> = ps.iter().map(|p| progress_count(&p.url) + 3).collect();
    wait_for_polls(seen).await;
    send("eft2").await;
    let (p, body) = landed_on("eft2").await;
    assert_eq!(p, 0, "the fallback prices P1 at its own service only");
    assert!(body.get("kv_pull_hint").is_none());
    let d = decision("eft2");
    let p1 = row(&d, 0);
    assert_eq!(p1["wait_source"], "ledger_fallback");
    assert_eq!(p1["wait_fallback_reason"], "view_too_old");
    assert!(p1["view_age_s"].as_f64().unwrap() > 1.5);
    assert_eq!(p1["wait_s"], 0.0, "no router rows on P1");
    // The step-model S_w of P1 (u 2,016, host restore 17,984, one pass), on
    // the same scale as the peers' terms (not u / R = 2,000 tokens).
    let svc = 1e-5 * 2_016.0 + 1e-7 * 17_984.0 + 0.004 + 0.05;
    assert!(
        (p1["service_s"].as_f64().unwrap() - svc).abs() < 1e-12,
        "{p1}"
    );
    assert!(
        (p1["trade_cost"].as_f64().unwrap() - 13_500.0 * svc).abs() < 1e-6,
        "{p1}"
    );
    assert_eq!(d["wait_fallbacks"], 1);
    assert!(d["wait_fallback_counts"]["view_too_old"].as_u64().unwrap() >= 1);
    assert_eq!(row(&d, 1)["wait_source"], "eft");
}

/// TRADE wait eft next to an engine without the `prefill_routing` section
/// (an older engine): it answers a poll that asks for the section with
/// HTTP 500. The router polls it again without the section, so its progress
/// goes on and it stays a candidate; its P falls back with `no_view`. Later
/// polls (within 30 s) do not ask it for the section again.
#[tokio::test]
async fn trade_wait_eft_polls_an_engine_without_the_section_without_it() {
    use sgl_router::config::{CoScheduleConfig, KvPullScope};
    use sgl_router::scheduler::co_schedule::{TradeParams, TradeWait};
    let ps = [
        EftP::start_with("old", routing_snapshot("r1", 0, 0), true).await,
        EftP::start("new", routing_snapshot("r2", 0, 0)).await,
    ];
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let indexer = MockIndexer::start(json!({"instances": {}})).await;
    let tmp = tempfile::tempdir().unwrap();
    let trace = tmp.path().join("trade-eft-old.jsonl");
    let map = ps
        .iter()
        .enumerate()
        .map(|(i, p)| KvWorkerMapEntry {
            id: i as u64 + 1,
            url: p.url.clone(),
            servicer: format!("p{}:18100", i + 1),
        })
        .collect();
    let model = sgl_router::eft_cheap::EftModel::from_json(
        &json!({"models": {"augmented": {"launch_interval": {"global":
            {"coefficients": [0.05, 1e-5, 0.0, 0.004, 1e-7, 0.0]}}}},
            "eft": {"tau_s": 0.02}}),
        Some("proxy-model".into()),
    )
    .unwrap();
    let mut cfg = config_with_hint(&indexer.url, map);
    cfg.model.kv_hint.as_mut().unwrap().min_incremental_tokens = 512;
    cfg.model.policy = PolicyKind::CoSchedule;
    cfg.model.co_schedule = Some(CoScheduleConfig {
        on_demand_load_refresh: false,
        uncached_balance: true,
        owner_threshold_tokens: 12288,
        detour_margin_tokens: 0,
        restore_cost_divisor: None,
        trade: Some(TradeParams {
            copy_lambda: 0.0,
            wait: TradeWait::Eft,
            ..TradeParams::default()
        }),
        trade_eft_model: Some(Arc::new(model)),
        global_queue: None,
        trade_e3: None,
        prefer_gpu_hit: false,
        locality_margin_tokens: None,
        cost_balance: false,
        cost_trace: Some(trace.clone()),
        max_uncached_tokens: 1,
        max_queue_wait_ms: 30000,
        load_poll_ms: 10,
        tick_ms: 30000,
        poll_on_hold_ms: 0,
        imbalance_spread_tokens: 0,
        pull_scope: KvPullScope::Any,
        kv_session_retention: false,
        kv_session_key_file: None,
    });
    let mut specs: Vec<WorkerSpec> = ps
        .iter()
        .enumerate()
        .map(|(i, p)| WorkerSpec {
            id: WorkerId(format!("p{}", i + 1)),
            url: p.url.clone(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: Some(8997),
        })
        .collect();
    specs.push(WorkerSpec {
        id: WorkerId("d1".into()),
        url: decode.url.clone(),
        mode: WorkerMode::Decode,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: None,
    });
    let app = build_router(build_ctx(cfg, specs));
    let read_events = || -> Vec<Value> {
        std::fs::read_to_string(&trace)
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    };
    let progress_count = |url: &str| {
        read_events()
            .iter()
            .filter(|e| e["event"] == "prefill_progress" && e["worker_url"] == url)
            .count()
    };
    // Both engines report progress, the old one through the second poll.
    tokio::time::timeout(Duration::from_secs(3), async {
        while ps.iter().any(|p| progress_count(&p.url) < 3) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("progress polls of both engines");
    // The progress polls of the old engine: the first asks for the section
    // and is rejected; its retry and the later polls do not ask for it.
    let includes: Vec<String> = ps[0]
        .includes
        .lock()
        .unwrap()
        .iter()
        .filter(|i| i.contains("prefill_work"))
        .cloned()
        .collect();
    let asked = includes
        .iter()
        .filter(|i| i.split(',').any(|s| s == "prefill_routing"))
        .count();
    assert_eq!(asked, 1, "asked once, then polled without it: {includes:?}");
    assert!(includes[0].contains("prefill_routing"), "{includes:?}");
    assert!(includes.len() >= 4, "{includes:?}");
    // The new engine is asked every time.
    let new_includes = ps[1].includes.lock().unwrap().clone();
    assert!(new_includes
        .iter()
        .filter(|i| i.contains("prefill_work"))
        .all(|i| i.contains("prefill_routing")));

    let request = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny", "prompt": vec![11; 2000], "max_tokens": 1, "rid": "old1",
            }))
            .unwrap(),
        ))
        .unwrap();
    let res = tokio::time::timeout(Duration::from_secs(3), app.oneshot(request))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let decision = read_events()
        .into_iter()
        .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == "old1")
        .expect("reserve event")["trade_decision"]
        .clone();
    let row = |url: &str| -> Value {
        decision["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["worker_url"] == url)
            .cloned()
            .unwrap_or_else(|| panic!("{url} is not a candidate: {decision}"))
    };
    let old = row(&ps[0].url);
    assert_eq!(old["wait_source"], "ledger_fallback");
    assert_eq!(old["wait_fallback_reason"], "no_view");
    assert_eq!(row(&ps[1].url)["wait_source"], "eft");
}

/// TRADE copy price `auto` through the HTTP path: two P engines, session KV
/// retention off. Each request names its session in `x-kv-session-context`.
/// The decisions trace the auto inputs (start-up: the fallback 18.4),
/// N_active counts the earlier sessions, `POST /kv_sessions/end` removes a
/// session at once (its reply is still the retention-off 400), each P busy
/// period ends with a `trade_auto_p_empty` event after the P response, and
/// the one-second snapshot carries `trade_auto`.
#[tokio::test]
async fn trade_copy_lambda_auto_counts_sessions_and_traces_its_inputs() {
    use sgl_router::config::{CoScheduleConfig, KvPullScope};
    use sgl_router::scheduler::co_schedule::{TradeAutoQ, TradeLambdaMode, TradeParams};
    let ps = [
        EftP::start("p1", json!({})).await,
        EftP::start("p2", json!({})).await,
    ];
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let indexer = MockIndexer::start(json!({"instances": {}})).await;
    let tmp = tempfile::tempdir().unwrap();
    let trace = tmp.path().join("trade-auto.jsonl");
    let map = ps
        .iter()
        .enumerate()
        .map(|(i, p)| KvWorkerMapEntry {
            id: i as u64 + 1,
            url: p.url.clone(),
            servicer: format!("p{}:18100", i + 1),
        })
        .collect();
    let mut cfg = config_with_hint(&indexer.url, map);
    cfg.model.kv_hint.as_mut().unwrap().min_incremental_tokens = 512;
    cfg.model.policy = PolicyKind::CoSchedule;
    cfg.model.co_schedule = Some(CoScheduleConfig {
        on_demand_load_refresh: false,
        uncached_balance: true,
        owner_threshold_tokens: 12288,
        detour_margin_tokens: 0,
        restore_cost_divisor: None,
        trade: Some(TradeParams {
            copy_lambda: 18.4,
            copy_lambda_mode: TradeLambdaMode::Auto {
                q: TradeAutoQ::Sat,
                mu: 1.0,
                min: 4.6,
                max: 1000.0,
            },
            ..TradeParams::default()
        }),
        trade_eft_model: None,
        global_queue: None,
        trade_e3: None,
        prefer_gpu_hit: false,
        locality_margin_tokens: None,
        cost_balance: false,
        cost_trace: Some(trace.clone()),
        max_uncached_tokens: 1,
        max_queue_wait_ms: 30000,
        load_poll_ms: 10,
        tick_ms: 30000,
        poll_on_hold_ms: 0,
        imbalance_spread_tokens: 0,
        pull_scope: KvPullScope::Any,
        kv_session_retention: false,
        kv_session_key_file: None,
    });
    let mut specs: Vec<WorkerSpec> = ps
        .iter()
        .enumerate()
        .map(|(i, p)| WorkerSpec {
            id: WorkerId(format!("p{}", i + 1)),
            url: p.url.clone(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: Some(8997),
        })
        .collect();
    specs.push(WorkerSpec {
        id: WorkerId("d1".into()),
        url: decode.url.clone(),
        mode: WorkerMode::Decode,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: None,
    });
    let app = build_router(build_ctx(cfg, specs));
    let read_events = || -> Vec<Value> {
        std::fs::read_to_string(&trace)
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    };
    type Check = Box<dyn Fn(&[Value]) -> bool>;
    let wait_for = |what: &'static str, check: Check| {
        let read_events = &read_events;
        async move {
            tokio::time::timeout(Duration::from_secs(5), async {
                while !check(&read_events()) {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .unwrap_or_else(|_| panic!("timed out waiting for {what}"));
        }
    };
    let send = |rid: &'static str, agent: &'static str| {
        let app = app.clone();
        async move {
            let request = Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .header(
                    "x-kv-session-context",
                    format!(r#"{{"agent_id":"{agent}","subagent_id":0,"request_id":1}}"#),
                )
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "model": "tiny", "prompt": vec![11; 2000], "max_tokens": 1, "rid": rid,
                    }))
                    .unwrap(),
                ))
                .unwrap();
            let res = tokio::time::timeout(Duration::from_secs(3), app.oneshot(request))
                .await
                .unwrap()
                .unwrap();
            assert_eq!(res.status(), StatusCode::OK, "{rid}");
        }
    };
    let decision = |rid: &str| -> Value {
        read_events()
            .into_iter()
            .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == rid)
            .unwrap_or_else(|| panic!("no reserve event for {rid}"))["trade_decision"]
            .clone()
    };
    send("a1", "agent-a").await;
    let d = decision("a1");
    assert_eq!(d["lambda_mode"], "auto", "{d}");
    assert_eq!(d["q_mode"], "sat");
    assert_eq!(d["fallback_reason"], "startup");
    assert_eq!(d["copy_lambda_used"], 18.4);
    assert_eq!(d["copy_lambda"], 18.4);
    assert_eq!(d["session_known"], true);
    assert_eq!(d["n_p"], 2);
    assert_eq!(d["n_active"], 0, "a session counts from its own commit on");
    for row in d["candidates"].as_array().unwrap() {
        assert!(row.get("auto_in_flight").is_some(), "{row}");
    }
    send("b1", "agent-b").await;
    assert_eq!(decision("b1")["n_active"], 1);
    send("a2", "agent-a").await;
    assert_eq!(decision("a2")["n_active"], 2);
    // END agent-a: the reply is the retention-off 400, but N_active drops.
    let end = Request::builder()
        .method("POST")
        .uri("/kv_sessions/end")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"agent_id":"agent-a","subagent_id":0}"#))
        .unwrap();
    let res = tokio::time::timeout(Duration::from_secs(3), app.clone().oneshot(end))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    send("c1", "agent-c").await;
    assert_eq!(decision("c1")["n_active"], 1, "agent-b only");
    // Each P response ends a busy period; the snapshot summarises the state.
    wait_for(
        "busy-period events",
        Box::new(|events: &[Value]| {
            let empties: Vec<&Value> = events
                .iter()
                .filter(|e| e["event"] == "trade_auto_p_empty")
                .collect();
            let requests: u64 = empties
                .iter()
                .map(|e| e["busy_period_requests"].as_u64().unwrap_or(0))
                .sum();
            requests == 4
        }),
    )
    .await;
    wait_for(
        "an auto snapshot with 4 credited prefills",
        Box::new(|events: &[Value]| {
            events.iter().any(|e| {
                e["event"] == "snapshot" && e["trade_auto"]["credited"].as_u64() == Some(4)
            })
        }),
    )
    .await;
    let snapshot = read_events()
        .into_iter()
        .rev()
        .find(|e| e["event"] == "snapshot" && e.get("trade_auto").is_some())
        .unwrap();
    assert_eq!(
        snapshot["trade_auto"]["workers"].as_array().unwrap().len(),
        2
    );
    assert_eq!(snapshot["trade_auto"]["credited_tokens"], 8000);
}
