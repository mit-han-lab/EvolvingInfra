// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `--symmetric-pd-p-select cost-function` end to end over mock symmetric
//! engines: the TRADE cost model of the co-schedule policy picks the P-leg
//! among the P-role workers and decides its pull (the P-leg's
//! `kv_pull_hint`); the D-leg follows the symmetric_pd D selection and never
//! carries the hint.
//!
//! These are the co-schedule TRADE proxy tests of `kv_hint_injection.rs` on
//! symmetric_pd, with the same prompts, indexer replies, expected costs and
//! choices. Every cost-trace row also carries
//! `"policy": "symmetric_pd_cost_function"`.

use axum::body::Body;
use axum::extract::Query;
use axum::http::{Request, StatusCode};
use axum::routing::{get, post};
use axum::Json;
use bytes::Bytes;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, SymmetricPdConfig, SymmetricPdCostFunctionConfig,
    SymmetricPdGwqAutoConfig, ModelConfig, ObservabilityConfig, PdDecodePolicy, PolicyKind, ProxyConfig,
    ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::symmetric_pd::{PSelect, RoleWeight};
use sgl_router::kv_hint::{KvHintConfig, KvWorkerMapEntry};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::scheduler::co_schedule::TradeParams;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{SymmetricPdWorkerInfo, WorkerRegistry};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;
use tower::ServiceExt;

/// One mock symmetric engine. It records every `/v1/completions` body and
/// answers with one choice. A P-leg (`kv_export`) whose rid starts with
/// `busy` waits for `release` (one waiter per notification); while it waits,
/// `/v1/loads` lists 20,000 remaining uncached tokens for it in
/// `prefill_work`. A P-leg whose rid starts with `gone` answers 500 with
/// the engine error type `symmetric_reader_gone`. A P-leg whose rid starts with
/// `lost` never gets an answer and is never listed (a dropped request).
/// `/v1/loads` records the `include` value of each query and answers
/// `include=...,host_room` as its [`HostRoom`] says.
struct Engine {
    url: String,
    bodies: Arc<Mutex<Vec<Value>>>,
    release: Arc<Notify>,
    includes: Arc<Mutex<Vec<String>>>,
}

/// What a mock engine answers to `GET /v1/loads?include=...,host_room`.
#[derive(Clone)]
enum HostRoom {
    /// An engine before the section: HTTP 400 (its load-snapshot section
    /// check rejects the unknown section).
    Rejected,
    /// An engine with the section but without session retention: 200
    /// without the section.
    Omitted,
    /// 200 with this section.
    Section(Value),
}

impl Engine {
    async fn start(name: &'static str) -> Self {
        Self::start_with(name, HostRoom::Rejected).await
    }

    async fn start_with(name: &'static str, room: HostRoom) -> Self {
        let bodies: Arc<Mutex<Vec<Value>>> = Arc::default();
        let release = Arc::new(Notify::new());
        let held: Arc<Mutex<BTreeMap<String, u64>>> = Arc::default();
        let sequence = Arc::new(AtomicU64::new(0));
        let includes: Arc<Mutex<Vec<String>>> = Arc::default();
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
                            let hold = rid.starts_with("busy") && body.get("kv_export").is_some();
                            let lost = rid.starts_with("lost") && body.get("kv_export").is_some();
                            let gone = rid.starts_with("gone") && body.get("kv_export").is_some();
                            bodies.lock().unwrap().push(body);
                            if lost {
                                std::future::pending::<()>().await;
                            }
                            if hold {
                                held.lock().unwrap().insert(rid.clone(), 20_000);
                                release.notified().await;
                                held.lock().unwrap().remove(&rid);
                            }
                            if gone {
                                // The symmetric owner ended the P-leg: its
                                // reader left (the D-leg recomputes).
                                let error = json!({"object": "error", "message": "reader left",
                                    "type": "symmetric_reader_gone", "code": 500});
                                return (StatusCode::INTERNAL_SERVER_ERROR, Json(error));
                            }
                            (
                                StatusCode::OK,
                                Json(json!({"choices": [{"index": 0, "text": "x"}]})),
                            )
                        }
                    }
                }),
            )
            .route(
                "/v1/loads",
                get({
                    let includes = includes.clone();
                    move |Query(query): Query<BTreeMap<String, String>>| {
                        let held = held.lock().unwrap().clone();
                        let n = sequence.fetch_add(1, Ordering::SeqCst);
                        let include = query.get("include").cloned().unwrap_or_default();
                        includes.lock().unwrap().push(include.clone());
                        let room = room.clone();
                        async move {
                            let mut load = json!({"num_running_reqs": held.len(),
                                "prefill_work": {"epoch": name, "sequence": n, "remaining": held}});
                            if include.split(',').any(|section| section == "host_room") {
                                match room {
                                    HostRoom::Rejected => return Err(StatusCode::BAD_REQUEST),
                                    HostRoom::Omitted => {}
                                    HostRoom::Section(room) => load["host_room"] = room,
                                }
                            }
                            Ok(Json(json!({"loads": [load]})))
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
            release,
            includes,
        }
    }

    /// The first body with this rid.
    fn body(&self, rid: &str) -> Option<Value> {
        self.bodies
            .lock()
            .unwrap()
            .iter()
            .find(|b| b["rid"] == rid)
            .cloned()
    }

    fn bodies(&self) -> Vec<Value> {
        self.bodies.lock().unwrap().clone()
    }

    /// The `include` value of every `/v1/loads` query so far, in order.
    fn includes(&self) -> Vec<String> {
        self.includes.lock().unwrap().clone()
    }
}

/// Indexer whose reply the test can change between requests.
async fn switchable_indexer(reply: Arc<Mutex<Value>>) -> String {
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

/// Register `url` as a symmetric worker with the P role (weights {1, 0})
/// or the D role ({0, 1}).
fn register(registry: &WorkerRegistry, url: &str, prefill: bool, bootstrap_port: u16) {
    let id = WorkerId(url.to_string());
    registry
        .add(WorkerSpec {
            id: id.clone(),
            url: url.to_string(),
            mode: WorkerMode::Symmetric,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: Some(bootstrap_port),
        })
        .unwrap();
    registry.get(&id).unwrap().set_symmetric_pd(SymmetricPdWorkerInfo {
        weight: RoleWeight {
            prefill: if prefill { 1.0 } else { 0.0 },
            decode: if prefill { 0.0 } else { 1.0 },
        },
        ..Default::default()
    });
}

struct Fixture {
    app: axum::Router,
    /// The P-role engines P1, P2, ... (indexer ids 1, 2, ...).
    ps: Vec<Engine>,
    /// The D-role engine.
    d: Engine,
    /// The router's worker registry (breaker state of the engines).
    registry: Arc<WorkerRegistry>,
    reply: Arc<Mutex<Value>>,
    trace: PathBuf,
    _tmp: tempfile::TempDir,
}

/// `--policy symmetric_pd --symmetric-pd-force-split --symmetric-pd-pull-hint
/// --symmetric-pd-p-select cost-function` with `trade` (load poll 10 ms) over
/// `n_p` P-role engines and one D-role engine. `gate` is the injector's
/// `min_incremental_tokens`; `reply` is the first indexer reply. The unknown
/// expiry is 60 s and the upstream request timeout 5 s.
async fn fixture(n_p: usize, trade: TradeParams, gate: u32, reply: Value) -> Fixture {
    fixture_with(n_p, trade, gate, reply, 60, Duration::from_secs(5), None).await
}

/// [`fixture`] with the unknown expiry `unknown_expire_secs` and the
/// upstream request timeout `timeout`. With `rooms` (one per P),
/// `--symmetric-pd-loads-host-room` is on and P `i` answers the `host_room`
/// include as `rooms[i]` says; without, every engine rejects it.
async fn fixture_with(
    n_p: usize,
    trade: TradeParams,
    gate: u32,
    reply: Value,
    unknown_expire_secs: u64,
    timeout: Duration,
    rooms: Option<Vec<HostRoom>>,
) -> Fixture {
    let extra = (None, false);
    fixture_full(
        n_p,
        trade,
        gate,
        reply,
        unknown_expire_secs,
        timeout,
        rooms,
        extra,
        None,
    )
    .await
}

/// [`fixture_with`] with `extra` = (`--symmetric-pd-gwq-auto` settings,
/// `--symmetric-pd-trace-session-moves`) and `any_holder`: `Some(flag)` also maps
/// the D-role engine in `--kv-worker-map` (indexer id `n_p + 1`, servicer
/// `d1:18100`), as a dynamic-role deployment maps every worker, plus a worker that
/// is mapped but never registered (id `n_p + 2`, `http://127.0.0.1:9`: a
/// removed worker whose blocks are still in the indexer), and sets
/// `--symmetric-pd-pull-from-any-holder` to `flag`. `None`: only the P-role
/// engines are mapped and the flag is off.
#[allow(clippy::too_many_arguments)]
async fn fixture_full(
    n_p: usize,
    trade: TradeParams,
    gate: u32,
    reply: Value,
    unknown_expire_secs: u64,
    timeout: Duration,
    rooms: Option<Vec<HostRoom>>,
    extra: (Option<SymmetricPdGwqAutoConfig>, bool),
    any_holder: Option<bool>,
) -> Fixture {
    let mut ps = Vec::new();
    for (i, name) in ["p1", "p2", "p3"].into_iter().take(n_p).enumerate() {
        let room = rooms
            .as_ref()
            .map_or(HostRoom::Rejected, |rooms| rooms[i].clone());
        ps.push(Engine::start_with(name, room).await);
    }
    let d = Engine::start("d1").await;
    let reply = Arc::new(Mutex::new(reply));
    let indexer = switchable_indexer(reply.clone()).await;
    let tmp = tempfile::tempdir().unwrap();
    let trace = tmp.path().join("cost.jsonl");
    let registry = Arc::new(WorkerRegistry::default());
    for (i, p) in ps.iter().enumerate() {
        register(&registry, &p.url, true, 20_000 + 2 * i as u16);
    }
    register(&registry, &d.url, false, 20_100);
    let mut worker_map: Vec<KvWorkerMapEntry> = ps
        .iter()
        .enumerate()
        .map(|(i, p)| KvWorkerMapEntry {
            id: i as u64 + 1,
            url: p.url.clone(),
            servicer: format!("p{}:18100", i + 1),
        })
        .collect();
    if any_holder.is_some() {
        worker_map.push(KvWorkerMapEntry {
            id: n_p as u64 + 1,
            url: d.url.clone(),
            servicer: "d1:18100".into(),
        });
        worker_map.push(KvWorkerMapEntry {
            id: n_p as u64 + 2,
            url: "http://127.0.0.1:9".into(),
            servicer: "ghost:18100".into(),
        });
    }
    let mut flex = SymmetricPdConfig {
        max_queue_wait_ms: 0,
        pull_hint: true,
        cost_function: Some(SymmetricPdCostFunctionConfig {
            trade,
            cost_trace: Some(trace.clone()),
            load_poll_ms: 10,
            unknown_expire_secs,
            loads_host_room: rooms.is_some(),
            gwq_auto: extra.0,
            trace_session_moves: extra.1,
            trade_e3: None,
            pull_from_any_holder: any_holder.unwrap_or(false),
            olp: None,
            trace_async: false,
        }),
        ..SymmetricPdConfig::default()
    };
    flex.placement.force_split = true;
    flex.placement.p_select = PSelect::CostFunction;
    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: PolicyKind::SymmetricPd,
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            kv_hint: Some(KvHintConfig {
                indexer_url: indexer,
                min_tokens: 256,
                min_incremental_tokens: gate,
                local_prefix_fallback: false,
                // Generous for a same-process mock; production default is 20.
                indexer_timeout_ms: 500,
                worker_map,
            }),
            co_schedule: None,
            pd_decode_policy: PdDecodePolicy::AffinityMinLoad,
            pd_decision_replay: None,
            pd_decode_tokens: None,
            symmetric_pd: Some(flex),
            gateway_cache_aware: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    };
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(timeout).unwrap());
    let ctx = Arc::new(AppContext::new(
        cfg,
        tokenizers,
        proxy,
        Arc::clone(&registry),
        policies,
    ));
    Fixture {
        app: build_router(ctx),
        ps,
        d,
        registry,
        reply,
        trace,
        _tmp: tmp,
    }
}

impl Fixture {
    fn events(&self) -> Vec<Value> {
        std::fs::read_to_string(&self.trace)
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    async fn wait_for(&self, what: &str, check: impl Fn(&[Value]) -> bool) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while !check(&self.events()) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {what}"));
    }

    /// POST `/v1/completions` with an `n`-token prompt and the rid; with
    /// `agent`, also the `x-kv-session-context` header. Expects 200.
    async fn send(&self, rid: &str, n: usize, agent: Option<&str>) {
        let mut request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json");
        if let Some(agent) = agent {
            request = request.header(
                "x-kv-session-context",
                format!(r#"{{"agent_id":"{agent}","subagent_id":0,"request_id":1}}"#),
            );
        }
        let body = json!({"model": "tiny", "prompt": vec![11; n], "max_tokens": 1, "rid": rid});
        let request = request
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let res = tokio::time::timeout(Duration::from_secs(5), self.app.clone().oneshot(request))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK, "{rid}");
    }

    /// The TRADE decision of the `reserve` row of `rid`.
    fn decision(&self, rid: &str) -> Value {
        self.events()
            .into_iter()
            .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == rid)
            .unwrap_or_else(|| panic!("no reserve event for {rid}"))["trade_decision"]
            .clone()
    }

    /// The index of the P-role engine that got the P-leg of `rid`, and
    /// that body.
    async fn landed_on(&self, rid: &str) -> (usize, Value) {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let hits: Vec<usize> = (0..self.ps.len())
                    .filter(|i| self.ps[*i].body(rid).is_some())
                    .collect();
                if !hits.is_empty() {
                    assert_eq!(hits.len(), 1, "{rid} reached several P");
                    return (hits[0], self.ps[hits[0]].body(rid).unwrap());
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("{rid} reached no P"))
    }
}

/// The candidate row of P `i` in a TRADE decision.
fn row(f: &Fixture, d: &Value, i: usize) -> Value {
    d["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["worker_url"] == f.ps[i].url.as_str())
        .cloned()
        .unwrap()
}

/// TRADE (`kv_hint_injection.rs` `trade_three_p`) on symmetric_pd: three P-role
/// engines (P1, P2, P3), one D-role engine and a 20,000-token prompt. The
/// indexer reports P1 = 18,000 cached tokens and P3 = 300 (P2 none); the
/// pull floor is 512.
///
/// 1. `busy`: all idle -> P1 (cost 2,000, no hint). P1 holds it and reports
///    20,000 remaining, so P1's backlog is 20,000.
/// 2. `pull`: P1 costs 22,000. lambda 0: P3 pulls 17,700 tokens from P1
///    (2,000 + 13,500 x 0.4677 s = 8,314 < P2's 8,318) and its P-leg
///    carries the hint. lambda 9.2: the copy price keeps `pull` on P1,
///    unhinted.
/// 3. (lambda 0 only) The indexer now reports P3 = 17,600: the gain from P1
///    is 400 < 512, so `small` moves to P3 WITHOUT a hint (cost 2,400).
///
/// `gate` is the injector's `min_incremental_tokens`. With 0, pick() itself
/// returns hints for the owner and for the 400-token gain; only the TRADE
/// floor (512) drops them.
#[tokio::test]
async fn flex_cost_function_moves_with_a_pull_only_when_it_pays() {
    flex_trade_three_p(0.0, 512).await;
}

#[tokio::test]
async fn flex_cost_function_copy_price_keeps_the_loaded_owner() {
    flex_trade_three_p(9.2, 512).await;
}

#[tokio::test]
async fn flex_cost_function_drops_sub_floor_hints_when_the_injector_gate_is_off() {
    flex_trade_three_p(0.0, 0).await;
}

async fn flex_trade_three_p(lambda: f64, gate: u32) {
    let trade = TradeParams {
        copy_lambda: lambda,
        ..TradeParams::default()
    };
    let reply = json!({"instances": {"1": {"cpu": 18000}, "3": {"cpu": 300}}});
    let f = fixture(3, trade, gate, reply).await;

    // 1. Idle pool: the owner P1 takes `busy` without a hint.
    f.send("busy", 20_000, None).await;
    let (p, body) = f.landed_on("busy").await;
    assert_eq!(p, 0);
    assert!(body.get("kv_export").is_some(), "{body}");
    assert!(
        body.get("kv_pull_hint").is_none(),
        "the owner never gets a hint (gate {gate})"
    );
    let d = f.decision("busy");
    assert_eq!(d["chosen"], f.ps[0].url.as_str());
    assert_eq!(d["owner"], f.ps[0].url.as_str());
    assert_eq!(d["pulled_tokens"], 0);
    // P1 reports 20,000 remaining for `busy`: its acknowledged backlog.
    f.wait_for("busy progress", |events| {
        events.iter().any(|e| {
            e["event"] == "prefill_progress"
                && e["updates"]
                    .as_array()
                    .is_some_and(|u| u.iter().any(|u| u["remaining_after"] == 20000))
        })
    })
    .await;

    // 2. The owner is loaded.
    f.send("pull", 20_000, None).await;
    let (p, body) = f.landed_on("pull").await;
    let d = f.decision("pull");
    assert_eq!(d["owner"], f.ps[0].url.as_str());
    assert_eq!(row(&f, &d, 0)["backlog_tokens"], "20000");
    assert_eq!(row(&f, &d, 0)["trade_cost"], 22000.0);
    assert_eq!(row(&f, &d, 2)["pulled_tokens"], 17700);
    assert_eq!(row(&f, &d, 1)["pulled_tokens"], 18000);
    if lambda == 0.0 {
        assert_eq!(p, 2, "P3 pulls: 8,314 < P2 8,318 < P1 22,000");
        assert_eq!(d["detour"], true);
        assert_eq!(d["pulled_tokens"], 17700);
        let hint = body.get("kv_pull_hint").expect("P3's P-leg has the hint");
        assert_eq!(hint["source_worker_id"], f.ps[0].url.as_str());
        assert_eq!(hint["expected_hit_tokens"], 18000);
    } else {
        assert_eq!(p, 0, "the copy price keeps the request on the owner");
        assert_eq!(d["detour"], false);
        assert_eq!(d["pulled_tokens"], 0);
        assert!(body.get("kv_pull_hint").is_none());
    }
    // The D-legs carry `kv_import` and never a hint.
    let d_legs = f.d.bodies();
    assert_eq!(d_legs.len(), 2);
    for leg in &d_legs {
        assert!(leg.get("kv_import").is_some(), "{leg}");
        assert!(leg.get("kv_pull_hint").is_none(), "{leg}");
    }

    if lambda == 0.0 {
        // 3. Below the floor: move without a pull.
        f.wait_for("pull finished", |events| {
            events
                .iter()
                .any(|e| e["event"] == "finished" && e["entry"]["rid"] == "pull")
        })
        .await;
        *f.reply.lock().unwrap() = json!({"instances": {"1": {"cpu": 18000}, "3": {"cpu": 17600}}});
        f.send("small", 20_000, None).await;
        let (p, body) = f.landed_on("small").await;
        assert_eq!(p, 2, "P3 recomputes 2,400 tokens instead of waiting on P1");
        assert!(
            body.get("kv_pull_hint").is_none(),
            "gain 400 < floor 512 (gate {gate})"
        );
        let d = f.decision("small");
        assert_eq!(d["detour"], true);
        assert_eq!(d["pulled_tokens"], 0);
        assert_eq!(row(&f, &d, 2)["trade_cost"], 2400.0);
    }
    f.ps[0].release.notify_one();
    f.wait_for("busy finished", |events| {
        events
            .iter()
            .any(|e| e["event"] == "finished" && e["entry"]["rid"] == "busy")
    })
    .await;
    let rows = f.events();
    assert!(
        rows.iter().all(|e| e["policy"] == "symmetric_pd_cost_function"),
        "every cost-trace row is labelled"
    );
}

/// `--co-schedule-trade-copy-lambda auto` on symmetric_pd
/// (`kv_hint_injection.rs`
/// `trade_copy_lambda_auto_counts_sessions_and_traces_its_inputs`): the
/// fallback lambda before tau, N_active from the session header, and
/// `POST /kv_sessions/end` removes a session at once (the reply is the
/// retention-off 400).
#[tokio::test]
async fn flex_cost_function_copy_lambda_auto_counts_sessions() {
    use sgl_router::scheduler::co_schedule::{TradeAutoQ, TradeLambdaMode};
    let trade = TradeParams {
        copy_lambda: 18.4,
        copy_lambda_mode: TradeLambdaMode::Auto {
            q: TradeAutoQ::Sat,
            mu: 1.0,
            min: 4.6,
            max: 1000.0,
        },
        ..TradeParams::default()
    };
    let f = fixture(2, trade, 512, json!({"instances": {}})).await;
    f.send("a1", 2000, Some("agent-a")).await;
    let d = f.decision("a1");
    assert_eq!(d["lambda_mode"], "auto", "{d}");
    assert_eq!(d["q_mode"], "sat");
    assert_eq!(d["fallback_reason"], "startup");
    assert_eq!(d["copy_lambda_used"], 18.4);
    assert_eq!(d["copy_lambda"], 18.4);
    assert_eq!(d["session_known"], true);
    assert_eq!(d["n_p"], 2);
    assert_eq!(d["n_active"], 0, "a session counts from its own commit on");
    for candidate in d["candidates"].as_array().unwrap() {
        assert!(candidate.get("auto_in_flight").is_some(), "{candidate}");
    }
    f.send("b1", 2000, Some("agent-b")).await;
    assert_eq!(f.decision("b1")["n_active"], 1);
    f.send("a2", 2000, Some("agent-a")).await;
    assert_eq!(f.decision("a2")["n_active"], 2);
    // END agent-a: the reply is the retention-off 400, but N_active drops.
    let end = Request::builder()
        .method("POST")
        .uri("/kv_sessions/end")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"agent_id":"agent-a","subagent_id":0}"#))
        .unwrap();
    let res = tokio::time::timeout(Duration::from_secs(3), f.app.clone().oneshot(end))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    f.send("c1", 2000, Some("agent-c")).await;
    assert_eq!(f.decision("c1")["n_active"], 1, "agent-b only");
    // Each P answer ends a busy period; the snapshot summarises the state.
    f.wait_for("busy-period events", |events| {
        let requests: u64 = events
            .iter()
            .filter(|e| e["event"] == "trade_auto_p_empty")
            .map(|e| e["busy_period_requests"].as_u64().unwrap_or(0))
            .sum();
        requests == 4
    })
    .await;
    f.wait_for("an auto snapshot with 4 credited prefills", |events| {
        events
            .iter()
            .any(|e| e["event"] == "snapshot" && e["trade_auto"]["credited"].as_u64() == Some(4))
    })
    .await;
    let snapshot = f
        .events()
        .into_iter()
        .rev()
        .find(|e| e["event"] == "snapshot" && e.get("trade_auto").is_some())
        .unwrap();
    assert_eq!(
        snapshot["trade_auto"]["workers"].as_array().unwrap().len(),
        2
    );
    assert_eq!(snapshot["trade_auto"]["credited_tokens"], 8000);
    assert_eq!(snapshot["policy"], "symmetric_pd_cost_function");
}

/// A P-leg without an answer (the upstream
/// request timeout of 300 ms here: a transport error) leaves its reservation
/// unknown, and its P never lists the RID. With
/// `--symmetric-pd-unknown-expire-secs 1` the reservation expires about 1 s later
/// (one `expired` event, reason `unknown_expired_no_engine_progress`) and the
/// next decision sees that P without the work. With 0 the work stays (the
/// behaviour before the flag).
#[tokio::test]
async fn flex_cost_function_unknown_p_leg_expires_without_engine_progress() {
    for expire_secs in [1u64, 0] {
        let timeout = Duration::from_millis(300);
        let reply = json!({"instances": {}});
        let f = fixture_with(
            2,
            TradeParams::default(),
            512,
            reply,
            expire_secs,
            timeout,
            None,
        )
        .await;
        f.send("lost", 4_000, None).await;
        let (p, _) = f.landed_on("lost").await;
        f.wait_for("the unknown row of lost", |events| {
            events.iter().any(|e| {
                e["event"] == "unknown"
                    && e["entry"]["rid"] == "lost"
                    && e["entry"]["reason"]
                        .as_str()
                        .is_some_and(|r| r.starts_with("prefill_transport_error"))
            })
        })
        .await;
        let expired = |events: &[Value]| -> Vec<Value> {
            events
                .iter()
                .filter(|e| e["event"] == "expired")
                .cloned()
                .collect()
        };
        if expire_secs > 0 {
            f.wait_for("lost expired", |events| !expired(events).is_empty())
                .await;
            let rows = expired(&f.events());
            assert_eq!(rows.len(), 1, "{rows:?}");
            assert_eq!(rows[0]["entry"]["rid"], "lost");
            let reason = "unknown_expired_no_engine_progress";
            assert_eq!(rows[0]["entry"]["reason"], reason);
            assert!(rows[0]["expiry"]["quiet_s"].as_f64().unwrap() >= 1.0);
            assert!(rows[0]["expiry"]["since_last_progress_s"].is_null());
        } else {
            tokio::time::sleep(Duration::from_millis(1_500)).await;
            assert!(expired(&f.events()).is_empty());
        }
        f.send("next", 4_000, None).await;
        let d = f.decision("next");
        let held = if expire_secs > 0 { "0" } else { "4000" };
        let lost_p = row(&f, &d, p);
        assert_eq!(lost_p["pending_uncached_tokens"], held, "{expire_secs}");
        assert_eq!(lost_p["unacknowledged_tokens"], held, "{expire_secs}");
        // The one-second snapshot agrees (the latest one may predate the expiry).
        let unknown = u64::from(expire_secs == 0);
        f.wait_for("a snapshot with the unknown count", |events| {
            events
                .iter()
                .rev()
                .find(|e| e["event"] == "snapshot")
                .is_some_and(|e| e["unknown_requests"] == unknown)
        })
        .await;
    }
}

/// `--symmetric-pd-loads-host-room` (observation only). P1 answers the
/// `host_room` section, P2 is an engine before the section (HTTP 400 for the
/// include) and P3 has the section but no session retention (200 without
/// it). With the flag every P is first asked for
/// `core,queues,prefill_work,host_room`; P2 is asked again at once without
/// it and not asked for the section again before the 30-s re-probe. The
/// one-second snapshot has P1's section with its age and lists P2 as
/// rejected; the decision has the section of P1 and null for P2 and P3.
/// Without the flag every progress query is `core,queues,prefill_work`, no
/// query asks for the section and no trace row has a `host_room` key. The P
/// choice and its cost are the same. (The symmetric_pd backend also polls every
/// worker, with `core` or `core,prefill_work`; only the progress queries,
/// the ones with `queues`, are checked in order.)
#[tokio::test]
async fn flex_cost_function_traces_the_engine_host_room_only_with_the_flag() {
    let room = json!({"total_tokens": 8_268_864u64, "free_tokens": 1_000,
        "class0_tokens": 6_000_000u64, "class1_tokens": 2_267_864u64,
        "evicted_class0_tokens": 300, "evicted_class1_tokens": 0});
    let plain = "core,queues,prefill_work";
    let asked = "core,queues,prefill_work,host_room";
    let mut picks = Vec::new();
    for flag in [false, true] {
        let rooms = flag.then(|| {
            vec![
                HostRoom::Section(room.clone()),
                HostRoom::Rejected,
                HostRoom::Omitted,
            ]
        });
        let reply = json!({"instances": {"1": {"cpu": 3000}}});
        let timeout = Duration::from_secs(5);
        let f = fixture_with(3, TradeParams::default(), 512, reply, 60, timeout, rooms).await;
        let (p1, p2, p3) = (&f.ps[0].url, &f.ps[1].url, &f.ps[2].url);
        if flag {
            f.wait_for("a snapshot with P1's host_room and P2 rejected", |events| {
                events.iter().any(|e| {
                    e["event"] == "snapshot"
                        && e["host_room_by_p"][p1.as_str()]["free_tokens"] == 1_000
                        && e["host_room_rejected"] == json!([p2])
                })
            })
            .await;
        } else {
            let any_snapshot = |events: &[Value]| events.iter().any(|e| e["event"] == "snapshot");
            f.wait_for("a snapshot", any_snapshot).await;
        }
        f.send("r1", 4_000, None).await;
        let (p, _) = f.landed_on("r1").await;
        let d = f.decision("r1");
        picks.push((p, d["trade_cost"].clone(), d["pulled_tokens"].clone()));
        let polls = |i: usize| -> Vec<String> {
            let includes = f.ps[i].includes().into_iter();
            includes
                .filter(|include| include.contains("queues"))
                .collect()
        };
        if flag {
            let rows = &d["host_room_by_p"];
            assert_eq!(rows[p1.as_str()]["class1_tokens"], 2_267_864u64, "{d}");
            assert_eq!(rows[p1.as_str()]["evicted_class0_tokens"], 300, "{d}");
            assert!(rows[p1.as_str()]["age_s"].as_f64().unwrap() >= 0.0, "{d}");
            assert!(
                rows[p2.as_str()].is_null() && rows[p3.as_str()].is_null(),
                "{d}"
            );
            let snapshot = f
                .events()
                .into_iter()
                .rev()
                .find(|e| e["event"] == "snapshot")
                .unwrap();
            assert_eq!(snapshot["host_room_invalid"], 0);
            assert!(snapshot["host_room_by_p"].get(p3.as_str()).is_none());
            for i in [0, 2] {
                let polls = polls(i);
                assert!(
                    !polls.is_empty() && polls.iter().all(|s| s == asked),
                    "{polls:?}"
                );
            }
            // P2: one rejected probe, then the plain poll at once and after.
            let polls = polls(1);
            assert_eq!(polls.first().map(String::as_str), Some(asked), "{polls:?}");
            assert_eq!(polls.get(1).map(String::as_str), Some(plain), "{polls:?}");
            assert_eq!(polls.iter().filter(|s| *s == asked).count(), 1, "{polls:?}");
            assert!(polls.len() > 2, "P2 is still polled: {polls:?}");
        } else {
            assert!(d.get("host_room_by_p").is_none(), "{d}");
            for (i, engine) in f.ps.iter().enumerate() {
                let polls = polls(i);
                assert!(
                    !polls.is_empty() && polls.iter().all(|s| s == plain),
                    "{polls:?}"
                );
                assert!(engine.includes().iter().all(|s| !s.contains("host_room")));
            }
            let keys = |e: &Value| -> Vec<String> {
                e.as_object()
                    .map(|row| row.keys().filter(|k| k.starts_with("host_room")).cloned())
                    .into_iter()
                    .flatten()
                    .collect()
            };
            for event in f.events() {
                assert!(keys(&event).is_empty(), "{event}");
                assert!(keys(&event["trade_decision"]).is_empty(), "{event}");
            }
        }
    }
    assert_eq!(picks[0], picks[1], "the section never changes the choice");
    assert_eq!(picks[0].0, 0, "P1 holds 3,000 of the 4,000 tokens");
}

/// `--co-schedule-trade-pull-price phi` end to end, with the lambda = q
/// words (copy price auto, mu 1, no clamp, rate auto) during start-up, so
/// lambda_q = the fallback 18.4. P1 owns 18,000 of the 20,000-token prompt
/// and holds `busy` (20,000 tokens of backlog). P1 and P2 answer a
/// host_room section with constant evicted counters (phi 0: no eviction in
/// the window); P3 has no section (it pays lambda_q, reason no_host_room).
/// `pull`: with the lambda price every pull costs 18.4 per token, so the
/// request stays on P1 (22,000 < P2's 339,518); with phi, P2 pulls 18,000
/// tokens from P1 and pays only the pull time (8,318 < 22,000). The
/// decision's copy_lambda is the price of the chosen P2 (0), and P3's
/// omitted section is counted (host_room_missing).
#[tokio::test]
async fn flex_cost_function_phi_price_moves_off_a_loaded_owner() {
    use sgl_router::scheduler::co_schedule::{
        TradeAutoQ, TradeLambdaMode, TradePullPrice, TradeRateMode,
    };
    let room = json!({"total_tokens": 1_000, "free_tokens": 0, "class0_tokens": 600,
        "class1_tokens": 400, "evicted_class0_tokens": 300, "evicted_class1_tokens": 0});
    for pull_price in [TradePullPrice::Lambda, TradePullPrice::Phi] {
        let trade = TradeParams {
            copy_lambda: 18.4,
            copy_lambda_mode: TradeLambdaMode::Auto {
                q: TradeAutoQ::Sat,
                mu: 1.0,
                min: 0.0,
                max: 1e9,
            },
            rate_mode: TradeRateMode::Auto,
            pull_price,
            ..TradeParams::default()
        };
        let rooms = vec![
            HostRoom::Section(room.clone()),
            HostRoom::Section(room.clone()),
            HostRoom::Omitted,
        ];
        let reply = json!({"instances": {"1": {"cpu": 18000}, "3": {"cpu": 300}}});
        let timeout = Duration::from_secs(5);
        let f = fixture_with(3, trade, 512, reply, 60, timeout, Some(rooms)).await;
        let (p1, p2) = (f.ps[0].url.clone(), f.ps[1].url.clone());
        let phi = pull_price == TradePullPrice::Phi;
        if phi {
            // Two samples of P1 and of P2: phi 0 (no increment).
            f.wait_for("phi of P1 and P2", |events| {
                events.iter().any(|e| {
                    e["event"] == "snapshot"
                        && e["phi_by_p"][p1.as_str()]["source"] == "no_eviction"
                        && e["phi_by_p"][p2.as_str()]["source"] == "no_eviction"
                })
            })
            .await;
        }
        f.send("busy", 20_000, None).await;
        let (p, _) = f.landed_on("busy").await;
        assert_eq!(p, 0, "the owner P1 takes the first request");
        f.wait_for("busy progress", |events| {
            events.iter().any(|e| {
                e["event"] == "prefill_progress"
                    && e["updates"]
                        .as_array()
                        .is_some_and(|u| u.iter().any(|u| u["remaining_after"] == 20000))
            })
        })
        .await;
        f.send("pull", 20_000, None).await;
        let (p, body) = f.landed_on("pull").await;
        let d = f.decision("pull");
        if phi {
            assert_eq!(p, 1, "P2 pulls without a copy charge: {d}");
            assert_eq!(d["pull_price"], "phi");
            assert_eq!(d["phi_base_source"], "startup");
            assert_eq!(d["phi_lambda_base"], 18.4);
            assert_eq!(d["pulled_tokens"], 18000);
            let p2_row = row(&f, &d, 1);
            assert_eq!(p2_row["copy_lambda"], 0.0);
            assert_eq!(p2_row["phi_source"], "no_eviction");
            let p3_row = row(&f, &d, 2);
            assert_eq!(p3_row["copy_lambda"], 18.4);
            assert_eq!(p3_row["phi_source"], "no_host_room");
            assert_eq!(d["copy_lambda"], 0.0, "the chosen P2's price: {d}");
            assert_eq!(d["copy_lambda_auto"], 18.4, "{d}");
            assert_eq!(d["phi_floored"], false, "min 0: {d}");
            let p3 = f.ps[2].url.clone();
            f.wait_for("P3 counted in host_room_missing", |events| {
                events.iter().any(|e| {
                    e["event"] == "snapshot"
                        && e["host_room_missing"][p3.as_str()]
                            .as_u64()
                            .is_some_and(|n| n >= 1)
                        && e["host_room_missing"].get(p1.as_str()).is_none()
                })
            })
            .await;
            let hint = body.get("kv_pull_hint").expect("P2's P-leg has the hint");
            assert_eq!(hint["source_worker_id"], p1.as_str());
            assert_eq!(hint["expected_hit_tokens"], 18000);
        } else {
            assert_eq!(p, 0, "the copy price keeps the request on P1: {d}");
            assert!(d.get("pull_price").is_none(), "{d}");
            assert!(row(&f, &d, 1).get("copy_lambda").is_none(), "{d}");
            assert!(body.get("kv_pull_hint").is_none());
        }
        f.ps[0].release.notify_one();
        f.wait_for("busy finished", |events| {
            events
                .iter()
                .any(|e| e["event"] == "finished" && e["entry"]["rid"] == "busy")
        })
        .await;
    }
}

/// `POST /symmetric_pd/place` cannot report a request's dispatch and end to the
/// cost ledger, so it is refused (400) and reserves nothing.
#[tokio::test]
async fn flex_place_endpoint_is_refused_with_cost_function() {
    let f = fixture(2, TradeParams::default(), 512, json!({"instances": {}})).await;
    let request = Request::builder()
        .method("POST")
        .uri("/symmetric_pd/place")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"prompt_len": 64, "rid": "place-1"}"#))
        .unwrap();
    let res = f.app.clone().oneshot(request).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    assert!(
        text.contains(sgl_router::scheduler::cost_function::PLACE_REFUSAL),
        "{text}"
    );
    assert!(f.events().iter().all(|e| e["event"] != "reserve"));
}

/// `--symmetric-pd-gwq-auto` end to end on one P-role engine (C 8192: no
/// /server_info chunk; R 13,500). `busy` (20,000 tokens) goes at arrival to
/// the empty P1, and its first sighting is a T_react sample (tens of ms).
/// With busy launched, P1 has 11,808 queued tokens, above C + R x T_react:
/// `free` (5,000 tokens) is held and does not reach P1. When busy ends, P1
/// has room and the queue releases free there (`room`). With
/// `--symmetric-pd-trace-session-moves` busy (with a session) has a
/// `session_move` record.
#[tokio::test]
async fn flex_cost_function_gwq_auto_holds_a_free_request_until_room() {
    let gwq = Some(SymmetricPdGwqAutoConfig {
        max_wait_ms: 300_000,
        ..Default::default()
    });
    let f = fixture_full(
        1,
        TradeParams::default(),
        512,
        json!({"instances": {}}),
        60,
        Duration::from_secs(5),
        None,
        (gwq, true),
        None,
    )
    .await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    f.send("busy", 20_000, Some("agent-b")).await;
    f.landed_on("busy").await;
    f.wait_for("the T_react sample of busy", |events| {
        events
            .iter()
            .any(|e| e["event"] == "gwq_lead_sample" && e["rid"] == "busy")
    })
    .await;
    let checks = async {
        tokio::time::sleep(Duration::from_millis(300)).await;
        let held = f.ps[0].body("free").is_none()
            && !f
                .events()
                .iter()
                .any(|e| e["event"] == "reserve" && e["entry"]["rid"] == "free");
        f.ps[0].release.notify_one();
        held
    };
    let ((), held) = tokio::join!(f.send("free", 5_000, None), checks);
    assert!(held, "free waits while P1 has no room");
    f.landed_on("free").await;
    let events = f.events();
    let gwq_of = |rid: &str| {
        events
            .iter()
            .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == rid)
            .map(|e| e["gwq"].clone())
            .unwrap()
    };
    let busy = gwq_of("busy");
    assert_eq!(busy["held"], false, "{busy}");
    assert_eq!(busy["lead_sample"], true, "{busy}");
    let free = gwq_of("free");
    assert_eq!(free["held"], true, "{free}");
    assert_eq!(free["release"], "room", "{free}");
    assert_eq!(
        (free["rule"].clone(), free["class"].clone()),
        (json!("auto"), json!("free"))
    );
    assert_eq!(free["react_source"], "measured", "{free}");
    // Held for most of the 300-ms check window (it arrives just after it
    // starts).
    assert!(free["hold_s"].as_f64().unwrap() >= 0.2, "{free}");
    let moved = events
        .iter()
        .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == "busy")
        .map(|e| e["session_move"].clone())
        .unwrap();
    assert_eq!(
        (moved["p_changed"].clone(), moved["prev_p"].clone()),
        (json!(false), Value::Null)
    );
    f.wait_for("gwq_stats with the release after a hold", |events| {
        events.iter().any(|e| {
            e["event"] == "gwq_stats"
                && e["counters"]["released_after_hold"]
                    .as_u64()
                    .is_some_and(|n| n >= 1)
        })
    })
    .await;
}

/// A P-leg answer 500 `symmetric_reader_gone` (the reader left; the D-leg
/// recomputes) is cancelled, not failed: the client gets the D-leg's 200,
/// the reservation finishes once with reason `prefill_reader_gone` (not
/// unknown, not an error), and the snapshot counts it.
#[tokio::test]
async fn flex_cost_function_reader_gone_prefill_end_is_not_a_failure() {
    let f = fixture(1, TradeParams::default(), 512, json!({"instances": {}})).await;
    f.send("gone-1", 2_000, None).await;
    f.wait_for("the P-leg ends", |events| {
        events
            .iter()
            .any(|e| e["event"] == "finished" && e["entry"]["rid"] == "gone-1")
    })
    .await;
    let events = f.events();
    let ends: Vec<&Value> = events
        .iter()
        .filter(|e| {
            e["entry"]["rid"] == "gone-1" && e["event"] != "reserve" && e["event"] != "dispatched"
        })
        .collect();
    assert_eq!(ends.len(), 1, "one terminal event: {ends:?}");
    assert_eq!(
        ends[0]["entry"]["reason"], "prefill_reader_gone",
        "{}",
        ends[0]
    );
    f.wait_for("the snapshot counts it", |events| {
        events
            .iter()
            .any(|e| e["event"] == "snapshot" && e["prefill_reader_gone"] == 1)
    })
    .await;
}

/// `--symmetric-pd-pull-from-any-holder`: after a P->D role switch the former P
/// (here the D-role engine D1, mapped in `--kv-worker-map`) still holds the
/// prefix. Three idle P-role engines, a 20,000-token prompt and an indexer
/// that reports only D1 = 18,000 cached tokens (lambda 0, floor 512).
///
/// Flag on: the P-leg pulls 18,000 tokens from D1 (its `kv_pull_hint` names
/// D1 and servicer `d1:18100`); the D-leg carries no hint. Flag off (D1 still
/// mapped): the answer is restricted to the P-role pool, so the same request
/// is a cold prefill without a hint (the behaviour before the flag). Both
/// also with `--symmetric-pd-gwq-auto`: the cold request is free, goes through
/// the global waiting queue and keeps the pull source of its arrival.
#[tokio::test]
async fn flex_cost_function_any_holder_pulls_from_a_d_role_holder() {
    for (flag, gwq) in [(true, false), (false, false), (true, true), (false, true)] {
        let trade = TradeParams {
            copy_lambda: 0.0,
            ..TradeParams::default()
        };
        let gwq_cfg = gwq.then(|| SymmetricPdGwqAutoConfig {
            max_wait_ms: 2_000,
            ..Default::default()
        });
        let reply = json!({"instances": {"4": {"cpu": 18000}}});
        let f = fixture_full(
            3,
            trade,
            512,
            reply,
            60,
            Duration::from_secs(5),
            None,
            (gwq_cfg, false),
            Some(flag),
        )
        .await;
        f.send("from-d", 20_000, None).await;
        let (_, body) = f.landed_on("from-d").await;
        let d = f.decision("from-d");
        let reserve = f
            .events()
            .into_iter()
            .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == "from-d")
            .unwrap();
        if gwq {
            assert_eq!(reserve["gwq"]["class"], "free", "{reserve}");
        } else {
            assert!(reserve.get("gwq").is_none(), "{reserve}");
        }
        if flag {
            let hint = body.get("kv_pull_hint").expect("the P-leg pulls from D1");
            assert_eq!(hint["source_worker_id"], f.d.url.as_str());
            assert_eq!(hint["source_servicer_addr"], "d1:18100");
            assert_eq!(hint["expected_hit_tokens"], 18000);
            assert_eq!(d["pulled_tokens"], 18000, "{d}");
        } else {
            assert!(
                body.get("kv_pull_hint").is_none(),
                "flag off: no D-role source {body}"
            );
            assert_eq!(d["pulled_tokens"], 0, "{d}");
        }
        let d_legs = f.d.bodies();
        assert_eq!(d_legs.len(), 1);
        assert!(d_legs[0].get("kv_import").is_some(), "{}", d_legs[0]);
        assert!(d_legs[0].get("kv_pull_hint").is_none(), "{}", d_legs[0]);
        f.wait_for("from-d finished", |events| {
            events
                .iter()
                .any(|e| e["event"] == "finished" && e["entry"]["rid"] == "from-d")
        })
        .await;
    }
}

/// End to end: with `--symmetric-pd-pull-from-any-holder` a
/// holder that is only in `--kv-worker-map` (a removed worker, here never
/// registered: 40,000 tokens) or whose breaker is open (P3: 30,000) is never
/// the pull source, however long its prefix; the healthy D-role D1 (18,000)
/// serves the pull, and P3 is not a target either.
#[tokio::test]
async fn flex_cost_function_any_holder_skips_unregistered_and_breaker_open_holders() {
    let trade = TradeParams {
        copy_lambda: 0.0,
        ..TradeParams::default()
    };
    let reply = json!({"instances": {
        "5": {"cpu": 40000},
        "3": {"cpu": 30000},
        "4": {"cpu": 18000}
    }});
    let f = fixture_full(
        3,
        trade,
        512,
        reply,
        60,
        Duration::from_secs(5),
        None,
        (None, false),
        Some(true),
    )
    .await;
    let p3 = f.registry.get(&WorkerId(f.ps[2].url.clone())).unwrap();
    for _ in 0..3 {
        p3.breaker.record_failure();
    }
    assert!(!p3.breaker.would_allow());
    f.send("healthy-only", 20_000, None).await;
    let (p, body) = f.landed_on("healthy-only").await;
    assert_ne!(p, 2, "P3's breaker is open: not a target");
    let hint = body.get("kv_pull_hint").expect("the P-leg pulls from D1");
    assert_eq!(hint["source_worker_id"], f.d.url.as_str());
    assert_eq!(hint["source_servicer_addr"], "d1:18100");
    assert_eq!(hint["expected_hit_tokens"], 18000);
    assert_eq!(f.decision("healthy-only")["pulled_tokens"], 18000);
    f.wait_for("healthy-only finished", |events| {
        events
            .iter()
            .any(|e| e["event"] == "finished" && e["entry"]["rid"] == "healthy-only")
    })
    .await;
}
