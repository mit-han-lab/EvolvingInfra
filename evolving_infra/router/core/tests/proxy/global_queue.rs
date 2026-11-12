// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! The global waiting queue of the TRADE placement through the HTTP PD path
//! (`--co-schedule-global-queue`). A request whose P already has the depth of
//! queued work (routed and not launched, or launched beyond the running pass)
//! waits in the router: neither leg is sent. When the engine's progress
//! snapshot shows that work launched and in its last pass, the router
//! releases the request; its P leg and its D leg then go out as without the
//! queue (the D leg has the bootstrap fields and never a pull hint).

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::{get, post};
use axum::Json;
use bytes::Bytes;
use serde_json::{json, Value};
use sgl_router::config::{
    ActiveLoadConfig, CoScheduleConfig, Config, DiscoveryBackend, KvPullScope, ModelConfig,
    ObservabilityConfig, PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::global_queue::{GlobalQueueConfig, GlobalQueueDepth, GlobalQueueMode};
use sgl_router::kv_hint::{KvHintConfig, KvWorkerMapEntry};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::scheduler::co_schedule::TradeParams;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tower::ServiceExt;

/// One mock P engine. It records every P-leg body and holds the request
/// `busy` until the test releases it. Its progress snapshot (`/v1/loads`
/// with `prefill_work`) lists `busy` only after the test sets `launched`:
/// before that, `busy` is routed but its first chunk has not launched.
/// After it, `busy` has 6,000 tokens left: its last pass (below 8,192).
struct GwqP {
    url: String,
    bodies: Arc<Mutex<Vec<Value>>>,
    launched: Arc<AtomicBool>,
    release: Arc<tokio::sync::Notify>,
}

impl GwqP {
    async fn start() -> Self {
        let bodies: Arc<Mutex<Vec<Value>>> = Arc::default();
        let launched = Arc::new(AtomicBool::new(false));
        let release = Arc::new(tokio::sync::Notify::new());
        let sequence = Arc::new(AtomicU64::new(0));
        let app = axum::Router::new()
            .route(
                "/v1/completions",
                post({
                    let (bodies, release) = (bodies.clone(), release.clone());
                    move |Json(body): Json<Value>| {
                        let (bodies, release) = (bodies.clone(), release.clone());
                        async move {
                            let rid = body["rid"].as_str().unwrap_or("").to_string();
                            bodies.lock().unwrap().push(body);
                            if rid == "busy" {
                                release.notified().await;
                            }
                            Json(json!({"choices": []}))
                        }
                    }
                }),
            )
            .route(
                "/v1/loads",
                get({
                    let launched = launched.clone();
                    move || {
                        let n = sequence.fetch_add(1, Ordering::SeqCst) + 1;
                        let mut remaining = BTreeMap::new();
                        if launched.load(Ordering::SeqCst) {
                            remaining.insert("busy".to_string(), 6_000u64);
                        }
                        async move {
                            Json(json!({"loads": [{"num_running_reqs": 1,
                                "prefill_work": {"epoch": "gwq-p", "sequence": n,
                                                 "remaining": remaining}}]}))
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
            launched,
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

/// A dynamo indexer mock without any cached prefix.
async fn empty_indexer() -> String {
    let app = axum::Router::new().route(
        "/query",
        post(|_b: Bytes| async { Json(json!({"instances": {}})) }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    url
}

fn gwq_config(indexer: &str, p_url: &str, trace: std::path::PathBuf) -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: PolicyKind::CoSchedule,
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            kv_hint: Some(KvHintConfig {
                indexer_url: indexer.into(),
                min_tokens: 256,
                min_incremental_tokens: 512,
                local_prefix_fallback: false,
                indexer_timeout_ms: 500,
                worker_map: vec![KvWorkerMapEntry {
                    id: 1,
                    url: p_url.into(),
                    servicer: "p1:18100".into(),
                }],
            }),
            co_schedule: Some(CoScheduleConfig {
                on_demand_load_refresh: false,
                uncached_balance: true,
                owner_threshold_tokens: 12288,
                detour_margin_tokens: 0,
                restore_cost_divisor: None,
                trade: Some(TradeParams {
                    copy_lambda: 18.4,
                    ..TradeParams::default()
                }),
                trade_eft_model: None,
                global_queue: Some(GlobalQueueConfig {
                    mode: GlobalQueueMode::Hrrn,
                    depth: GlobalQueueDepth::Tokens(1000),
                    max_wait_ms: 30_000,
                    hold: sgl_router::global_queue::GlobalQueueHold::All,
                    room: sgl_router::global_queue::GlobalQueueRoom::Tokens,
                }),
                trade_e3: None,
                prefer_gpu_hit: false,
                locality_margin_tokens: None,
                cost_balance: false,
                cost_trace: Some(trace),
                max_uncached_tokens: 1,
                max_queue_wait_ms: 30000,
                load_poll_ms: 10,
                tick_ms: 30000,
                poll_on_hold_ms: 0,
                imbalance_spread_tokens: 0,
                pull_scope: KvPullScope::Any,
                kv_session_retention: false,
                kv_session_key_file: None,
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

/// HRRN queue with a fixed depth of 1000 tokens and one P (progress polled
/// every 10 ms).
///
/// 1. `busy` (20,000 tokens) goes at once: the P has no queued work. The P
///    holds it, and its snapshot does not list it: 20,000 routed tokens have
///    not launched.
/// 2. `small` (500 tokens) waits: no P leg, no D leg, no reservation.
/// 3. The snapshot lists `busy` with 6,000 tokens left (launched, in its
///    last pass): the queued work is 0, the next applied poll releases
///    `small`. Its P leg and D leg go out with one bootstrap room; the D leg
///    has no pull hint and the same prompt. The trace records the hold.
#[tokio::test]
async fn trade_gwq_holds_the_p_leg_and_releases_it_when_the_p_launches_its_work() {
    let p = GwqP::start().await;
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let indexer = empty_indexer().await;
    let tmp = tempfile::tempdir().unwrap();
    let trace = tmp.path().join("gwq.jsonl");
    let cfg = gwq_config(&indexer, &p.url, trace.clone());
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for spec in [
        WorkerSpec {
            id: WorkerId("p1".into()),
            url: p.url.clone(),
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
    ] {
        let _ = registry.add(spec);
    }
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let app = build_router(Arc::new(AppContext::new(
        cfg, tokenizers, proxy, registry, policies,
    )));
    let read_events = || -> Vec<Value> {
        std::fs::read_to_string(&trace)
            .unwrap_or_default()
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()
    };
    let reserve = |rid: &str| -> Option<Value> {
        read_events()
            .into_iter()
            .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == rid)
    };
    let send = |rid: &'static str, tokens: usize| {
        let app = app.clone();
        async move {
            let request = Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "model": "tiny", "prompt": vec![11; tokens], "max_tokens": 1, "rid": rid,
                    }))
                    .unwrap(),
                ))
                .unwrap();
            let res = tokio::time::timeout(Duration::from_secs(10), app.oneshot(request))
                .await
                .unwrap()
                .unwrap();
            assert_eq!(res.status(), StatusCode::OK, "{rid}");
        }
    };
    async fn wait_until(what: &str, check: impl Fn() -> bool) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while !check() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {what}"));
    }

    // A first progress snapshot: the P has a fresh view.
    wait_until("a progress snapshot", || {
        read_events()
            .iter()
            .any(|e| e["event"] == "prefill_progress")
    })
    .await;
    // 1. `busy` goes at once; the D leg answers, the P keeps it.
    send("busy", 20_000).await;
    wait_until("busy on P", || p.body("busy").is_some()).await;
    let b = reserve("busy").expect("busy reserved");
    assert_eq!(b["gwq"]["held"], false);
    assert_eq!(b["gwq"]["release"], "room");

    // 2. `small` waits in the router.
    let small = tokio::spawn(send("small", 500));
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        !small.is_finished(),
        "small must wait: 20,000 queued tokens"
    );
    assert!(p.body("small").is_none(), "no P leg while held");
    assert!(reserve("small").is_none(), "no reservation while held");
    let decode_body = decode.captured.lock().unwrap().last_body.clone();
    let decode_rid = decode_body
        .as_ref()
        .and_then(|b| serde_json::from_slice::<Value>(b).ok())
        .map(|v| v["rid"].clone());
    assert_ne!(decode_rid, Some(json!("small")), "no D leg while held");

    // 3. `busy` launches: the next poll releases `small`.
    let launched_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as f64;
    p.launched.store(true, Ordering::SeqCst);
    tokio::time::timeout(Duration::from_secs(5), small)
        .await
        .expect("small released")
        .unwrap();
    let p_body = p.body("small").expect("P leg sent after the release");
    assert!(p_body.get("kv_pull_hint").is_none(), "no cache: no pull");
    let decode_body = decode.captured.lock().unwrap().last_body.clone().unwrap();
    let d_body: Value = serde_json::from_slice(&decode_body).unwrap();
    assert_eq!(d_body["rid"], "small");
    assert!(
        d_body.get("kv_pull_hint").is_none(),
        "the D leg never has a hint"
    );
    assert!(d_body.get("bootstrap_room").is_some());
    assert_eq!(d_body["bootstrap_room"], p_body["bootstrap_room"]);
    assert_eq!(d_body["prompt"], json!(vec![11; 500]));
    assert_eq!(p_body["prompt"], json!(vec![11; 500]));
    let r = reserve("small").expect("small reserved at release");
    let g = &r["gwq"];
    assert_eq!(g["mode"], "hrrn");
    assert_eq!(g["held"], true);
    assert_eq!(g["release"], "room");
    assert_eq!(g["p_arrival"], p.url.as_str());
    assert_eq!(g["p_release"], p.url.as_str());
    assert_eq!(g["depth_tokens"], 1000);
    assert_eq!(g["queued_tokens"], "0");
    assert_eq!(g["launched_tokens"], "6000");
    // Held from its arrival in the router until at least the launch (1 ms
    // for the millisecond truncation of both clocks).
    let held_until_launch_s =
        (launched_unix_ms - g["arrived_unix_ms"].as_f64().unwrap() - 1.0) / 1000.0;
    assert!(g["hold_s"].as_f64().unwrap() >= held_until_launch_s, "{g}");
    assert_eq!(r["trade_decision"]["chosen"], p.url.as_str());
    wait_until("a gwq_stats event", || {
        read_events().iter().any(|e| e["event"] == "gwq_stats")
    })
    .await;
    p.release.notify_one();
    wait_until("busy finished", || {
        read_events()
            .iter()
            .any(|e| e["event"] == "finished" && e["entry"]["rid"] == "busy")
    })
    .await;
}
