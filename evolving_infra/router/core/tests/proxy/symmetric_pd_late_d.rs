// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `--policy symmetric_pd` on symmetric workers end to end over mock engines:
//!
//! * late D choose (`--pd-late-decode choose`): the P-leg names every
//!   candidate in `kv_export.candidates` and carries the symmetric `late_d`
//!   field; each D-leg carries its own `kv_import` and `late_d.cand_idx`; the
//!   P asks the router over its link (mode `symmetric`, key = transfer id);
//!   `fit_req` on observed capacity reads each decode's `/v1/loads`;
//! * session retention: the P-leg alone carries `kv_session`; the bound
//!   decode becomes the decode home (the old home is queued for release); the
//!   `x-kv-session-context` alias works on the plain Split too;
//! * `--symmetric-pd-p-select gateway-cache-aware`: the gateway rule picks
//!   P, a repeated prefix follows its tree;
//! * `--symmetric-pd-p-select cost-function` with late D and session
//!   retention: the TRADE cost function picks P and its pull; the P-leg
//!   alone carries the pull hint and the session tag.
//!
//! The mock owner holds its P-leg until every candidate leg reached its
//! decode, sends BIND_READY, takes the router's DECISION (or its default),
//! and reports FINAL. A mock decode holds its leg until the transfer is
//! bound, then answers 200 (the winner) or 409 `released`.

use axum::body::Body;
use axum::http::{HeaderValue, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Json;
use bytes::Bytes;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, GatewayCacheAwareConfig, LateDecodeConfig,
    LateDecodeMode, LateDecodeRule, ModelConfig, ObservabilityConfig, PdDecodePolicy,
    PdDecodeTokensConfig, PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
    SymmetricPdConfig, SymmetricPdCostFunctionConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::kv_hint::{KvHintConfig, KvWorkerMapEntry};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::scheduler::co_schedule::TradeParams;
use sgl_router::scheduler::{DSelect, PSelect, RoleWeight};
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{LateDWorkerInfo, SymmetricPdWorkerInfo, WorkerRegistry};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tower::ServiceExt;
use zeromq::{Endpoint, RouterSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

const COMPLETIONS: &str = "/v1/completions";
const GENERATE: &str = "/generate";
const CHOICE_TIMEOUT: Duration = Duration::from_millis(500);

// ---------------------------------------------------------------------------
// Shared transfer state and the owner's router link.
// ---------------------------------------------------------------------------

#[derive(Default, Clone, Copy)]
struct Transfer {
    arrived: u64,
    winner: Option<u64>,
}

#[derive(Default)]
struct Transfers(Mutex<HashMap<u64, Transfer>>);

impl Transfers {
    fn update(&self, id: u64, f: impl FnOnce(&mut Transfer)) {
        f(self.0.lock().unwrap().entry(id).or_default());
    }

    fn get(&self, id: u64) -> Transfer {
        self.0.lock().unwrap().get(&id).copied().unwrap_or_default()
    }

    async fn wait_for<T>(
        &self,
        id: u64,
        limit: Duration,
        f: impl Fn(&Transfer) -> Option<T>,
    ) -> Option<T> {
        let deadline = Instant::now() + limit;
        loop {
            if let Some(v) = f(&self.get(id)) {
                return Some(v);
            }
            if Instant::now() >= deadline {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
}

#[derive(Default)]
struct LinkState {
    epoch: Option<u64>,
    waiters: HashMap<(u64, u64), oneshot::Sender<u64>>,
}

/// The symmetric owner's ROUTER socket (its late-D link).
struct Link {
    port: u16,
    out: mpsc::Sender<Vec<u8>>,
    state: Arc<Mutex<LinkState>>,
}

fn frame(v: Value) -> Vec<u8> {
    rmp_serde::to_vec_named(&v).unwrap()
}

impl Link {
    async fn start() -> Arc<Self> {
        let mut socket = RouterSocket::new();
        let bound = socket.bind("tcp://127.0.0.1:0").await.unwrap();
        let Endpoint::Tcp(_, port) = bound else {
            panic!("tcp endpoint expected")
        };
        let (mut tx, mut rx) = socket.split();
        let (out, mut out_rx) = mpsc::channel::<Vec<u8>>(64);
        let state: Arc<Mutex<LinkState>> = Arc::default();
        let st = Arc::clone(&state);
        tokio::spawn(async move {
            let mut ident: Option<Bytes> = None;
            loop {
                tokio::select! {
                    msg = rx.recv() => {
                        let Ok(msg) = msg else { break };
                        let mut frames = msg.into_vec();
                        let payload = frames.pop().unwrap();
                        let id = frames.remove(0);
                        let m: Value = rmp_serde::from_slice(&payload).unwrap();
                        let reply = match m["t"].as_str().unwrap() {
                            "hello" => {
                                ident = Some(id);
                                st.lock().unwrap().epoch = m["router_epoch"].as_u64();
                                Some(frame(json!({"v": 1, "t": "hello_ack",
                                    "router_epoch": m["router_epoch"],
                                    "engine_boot_ns": 7, "mode": "symmetric",
                                    "kv_ip": "127.0.0.1"})))
                            }
                            "ping" => Some(frame(json!({"v": 1, "t": "pong", "ts_ns": m["ts_ns"]}))),
                            "decision" => {
                                let mut s = st.lock().unwrap();
                                for d in m["items"].as_array().unwrap() {
                                    let key = (d["key"].as_u64().unwrap(), d["bind_seq"].as_u64().unwrap());
                                    if let Some(w) = s.waiters.remove(&key) {
                                        let _ = w.send(d["winner_idx"].as_u64().unwrap());
                                    }
                                }
                                None
                            }
                            _ => None,
                        };
                        if let (Some(reply), Some(id)) = (reply, ident.clone()) {
                            let mut msg = ZmqMessage::from(Bytes::from(reply));
                            msg.push_front(id);
                            let _ = tx.send(msg).await;
                        }
                    }
                    f = out_rx.recv() => {
                        let Some(f) = f else { break };
                        if let Some(id) = ident.clone() {
                            let mut msg = ZmqMessage::from(Bytes::from(f));
                            msg.push_front(id);
                            let _ = tx.send(msg).await;
                        }
                    }
                }
            }
        });
        Arc::new(Self { port, out, state })
    }

    async fn ask(&self, key: u64, eligible: u64, default_idx: u64) -> Option<u64> {
        let (tx, rx) = oneshot::channel();
        let epoch = {
            let mut s = self.state.lock().unwrap();
            s.epoch?;
            s.waiters.insert((key, 1), tx);
            s.epoch
        };
        self.out
            .send(frame(
                json!({"v": 1, "t": "bind_ready", "router_epoch": epoch,
                "p_seq": 1, "t_send_ns": 1u64,
                "items": [{"key": key, "bind_seq": 1, "eligible": eligible,
                           "ready": eligible.count_ones(), "input_len": 4,
                           "default_idx": default_idx, "local_ok": false, "local_load": 0}]}),
            ))
            .await
            .unwrap();
        tokio::time::timeout(CHOICE_TIMEOUT, rx).await.ok()?.ok()
    }

    async fn final_(&self, key: u64, admitted: u64, reason: &str) {
        let epoch = self.state.lock().unwrap().epoch;
        if epoch.is_none() {
            return;
        }
        self.out
            .send(frame(json!({"v": 1, "t": "final", "router_epoch": epoch,
                "items": [{"key": key, "bind_seq": 1, "admitted_idx": admitted,
                           "reason": reason, "t_admit_ns": 2u64}]})))
            .await
            .unwrap();
    }
}

// ---------------------------------------------------------------------------
// Mock symmetric engines.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Seen {
    /// (worker URL, body) of every `/v1/completions` call.
    bodies: Mutex<Vec<(String, Value)>>,
}

impl Seen {
    fn of(&self, pred: impl Fn(&Value) -> bool) -> Vec<(String, Value)> {
        self.bodies
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, b)| pred(b))
            .cloned()
            .collect()
    }
}

/// One symmetric engine: P-legs (`kv_export`), D-legs (`kv_import`) and
/// plain requests on `/v1/completions`; `/v1/loads` reports `capacity`
/// (allocated requests, allocatable tokens) when set, or the per-RID
/// prefill progress when `cost` is set.
#[derive(Clone)]
struct Engine {
    me: String,
    link: Option<Arc<Link>>,
    transfers: Arc<Transfers>,
    seen: Arc<Seen>,
    capacity: Option<(u64, u64)>,
    cost: Option<Arc<CostEngine>>,
}

/// The P-role engine state of the cost-function test: the P-leg of `busy`
/// waits for `release`, and while it waits `/v1/loads` lists 20,000
/// remaining uncached tokens for it in `prefill_work`.
#[derive(Default)]
struct CostEngine {
    held: Mutex<BTreeMap<String, u64>>,
    release: tokio::sync::Notify,
    sequence: AtomicU64,
}

fn released() -> Response {
    let mut r = (
        StatusCode::CONFLICT,
        Json(json!({"object": "error", "message": "late_d released", "code": 409})),
    )
        .into_response();
    r.headers_mut()
        .insert("x-late-d-reason", HeaderValue::from_static("released"));
    r
}

fn ok_answer() -> Response {
    Json(json!({"choices": [{"index": 0, "text": "x"}]})).into_response()
}

async fn engine_completions(e: Engine, body: Value) -> Response {
    e.seen
        .bodies
        .lock()
        .unwrap()
        .push((e.me.clone(), body.clone()));
    let hold = body["rid"] == "busy" && body.get("kv_export").is_some();
    if let Some(cost) = e.cost.as_ref().filter(|_| hold) {
        cost.held.lock().unwrap().insert("busy".into(), 20_000);
        cost.release.notified().await;
        cost.held.lock().unwrap().remove("busy");
    }
    if let (Some(export), Some(late)) = (body.get("kv_export"), body.get("late_d")) {
        let id = export["transfer_id"].as_u64().unwrap();
        let k = late["k"].as_u64().unwrap();
        let default_idx = late["default_idx"].as_u64().unwrap();
        let full = (1u64 << k) - 1;
        let eligible = e
            .transfers
            .wait_for(id, Duration::from_secs(3), |t| {
                (t.arrived == full).then_some(t.arrived)
            })
            .await
            .unwrap_or_else(|| e.transfers.get(id).arrived);
        let link = e.link.as_ref().expect("an owner with a link");
        let (winner, reason) = match link.ask(id, eligible, default_idx).await {
            Some(w) if w < 64 && eligible >> w & 1 == 1 => (w, "decided"),
            _ => (default_idx, "timeout"),
        };
        e.transfers.update(id, |t| t.winner = Some(winner));
        link.final_(id, winner, reason).await;
        return Json(json!({"ok": true})).into_response();
    }
    if let (Some(import), Some(idx)) = (
        body.get("kv_import"),
        body.pointer("/late_d/cand_idx").and_then(Value::as_u64),
    ) {
        let id = import["transfer_id"].as_u64().unwrap();
        e.transfers.update(id, |t| t.arrived |= 1 << idx);
        return match e
            .transfers
            .wait_for(id, Duration::from_secs(10), |t| t.winner)
            .await
        {
            Some(w) if w == idx => ok_answer(),
            _ => released(),
        };
    }
    // A plain Split leg or a Local request.
    ok_answer()
}

async fn engine_loads(e: Engine) -> Response {
    if let Some(cost) = &e.cost {
        let held = cost.held.lock().unwrap().clone();
        let n = cost.sequence.fetch_add(1, Ordering::SeqCst);
        let loads = json!({"loads": [{"num_running_reqs": held.len(),
            "prefill_work": {"epoch": e.me, "sequence": n, "remaining": held}}]});
        return Json(loads).into_response();
    }
    let Some((requests, allocatable)) = e.capacity else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let rooms: Vec<u64> = (1..=requests).map(|r| 9_000_000 + r).collect();
    Json(json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "loads": [{
            "timestamp": now - 0.0005,
            "late_d": {
                "capacity_available": true,
                "allocated_rooms_truncated": false,
                "allocated_room_ids_complete": true,
                "allocated_rooms": rooms,
                "allocated_requests": requests,
                "allocated_context_tokens": 0,
                "allocatable_tokens": allocatable,
                "free_gpu_tokens": allocatable,
                "output_reserve_tokens": 512,
                "page_size": 64,
            },
        }],
    }))
    .into_response()
}

async fn serve_engine(e: Engine, listener: tokio::net::TcpListener) {
    let (ec, eg, el) = (e.clone(), e.clone(), e);
    let app = axum::Router::new()
        .route(
            COMPLETIONS,
            post(move |Json(body): Json<Value>| engine_completions(ec.clone(), body)),
        )
        // Native requests, marked so a test can tell the two routes apart.
        .route(
            GENERATE,
            post(move |Json(mut body): Json<Value>| {
                body["mock_route"] = json!(GENERATE);
                engine_completions(eg.clone(), body)
            }),
        )
        .route("/v1/loads", get(move || engine_loads(el.clone())));
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
}

// ---------------------------------------------------------------------------
// The router under test.
// ---------------------------------------------------------------------------

struct Fixture {
    app: axum::Router,
    ctx: Arc<AppContext>,
    /// Prefill-role workers, then decode-role workers, each in URL order.
    p_urls: Vec<String>,
    d_urls: Vec<String>,
    seen: Arc<Seen>,
    transfers: Arc<Transfers>,
    /// The `/v1/loads` capacity each decode reports, by URL.
    d_capacity: HashMap<String, Option<(u64, u64)>>,
    /// `Opts::cost_function`: the indexer reply (ids 1, 2, ... are the
    /// prefill-role workers in URL order), the cost trace and each
    /// prefill-role engine's hold state, by URL.
    indexer: Option<Arc<Mutex<Value>>>,
    trace: Option<std::path::PathBuf>,
    costs: HashMap<String, Arc<CostEngine>>,
    _tmp: Option<tempfile::TempDir>,
}

struct Opts {
    late: Option<LateDecodeRule>,
    fit_native: bool,
    n_p: usize,
    /// Per decode worker: `(allocated requests, allocatable tokens)`.
    d_capacity: Vec<Option<(u64, u64)>>,
    kv_sessions: bool,
    gateway_ca: bool,
    /// `--symmetric-pd-p-select cost-function` (TRADE, lambda 0) with the pull
    /// hint, an indexer mock and the prefill progress of the P-role engines.
    cost_function: bool,
}

async fn fixture(opts: Opts) -> Fixture {
    let transfers: Arc<Transfers> = Arc::default();
    let seen: Arc<Seen> = Arc::default();
    let registry = Arc::new(WorkerRegistry::default());
    let mut p_urls = Vec::new();
    let mut d_urls = Vec::new();
    let mut links = Vec::new();
    let mut d_capacity = HashMap::new();
    let mut costs = HashMap::new();
    let n_d = opts.d_capacity.len();
    for i in 0..opts.n_p + n_d {
        let prefill = i < opts.n_p;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let url = format!("http://127.0.0.1:{port}");
        let link = if prefill && opts.late.is_some() {
            Some(Link::start().await)
        } else {
            None
        };
        let cost = (prefill && opts.cost_function).then(|| Arc::new(CostEngine::default()));
        if let Some(cost) = &cost {
            costs.insert(url.clone(), Arc::clone(cost));
        }
        serve_engine(
            Engine {
                me: url.clone(),
                link: link.clone(),
                transfers: Arc::clone(&transfers),
                seen: Arc::clone(&seen),
                capacity: if prefill {
                    None
                } else {
                    opts.d_capacity[i - opts.n_p]
                },
                cost,
            },
            listener,
        )
        .await;
        let id = WorkerId(url.clone());
        registry
            .add(WorkerSpec {
                id: id.clone(),
                url: url.clone(),
                mode: WorkerMode::Symmetric,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: Some(20_000 + 2 * i as u16),
            })
            .unwrap();
        let w = registry.get(&id).unwrap();
        w.set_symmetric_pd(SymmetricPdWorkerInfo {
            weight: RoleWeight {
                prefill: if prefill { 1.0 } else { 0.0 },
                decode: if prefill { 0.0 } else { 1.0 },
            },
            ..Default::default()
        });
        w.set_late_d(opts.late.map(|_| LateDWorkerInfo {
            mode: "choose".into(),
            control_port: link.as_ref().map(|l| l.port),
            kv_ip: Some("127.0.0.1".into()),
        }));
        if let Some(l) = &link {
            links.push(format!("tcp://127.0.0.1:{}", l.port));
        }
        if prefill {
            p_urls.push(url);
        } else {
            d_capacity.insert(url.clone(), opts.d_capacity[i - opts.n_p]);
            d_urls.push(url);
        }
    }
    p_urls.sort();
    d_urls.sort();
    let mut flex = SymmetricPdConfig {
        max_queue_wait_ms: 0,
        kv_session_retention: opts.kv_sessions,
        gateway_cache_aware: opts.gateway_ca.then(|| GatewayCacheAwareConfig {
            max_tree_size: 66_163_200,
            ..GatewayCacheAwareConfig::default()
        }),
        ..SymmetricPdConfig::default()
    };
    flex.placement.force_split = true;
    flex.placement.d_select = DSelect::RoundRobin;
    let (kv_hint, indexer, trace, tmp) = if opts.cost_function {
        let reply = Arc::new(Mutex::new(json!({"instances": {}})));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cost.jsonl");
        flex.pull_hint = true;
        flex.placement.p_select = PSelect::CostFunction;
        flex.cost_function = Some(SymmetricPdCostFunctionConfig {
            trade: TradeParams {
                copy_lambda: 0.0,
                ..TradeParams::default()
            },
            cost_trace: Some(path.clone()),
            load_poll_ms: 10,
            unknown_expire_secs: 60,
            loads_host_room: false,
            gwq_auto: None,
            trace_session_moves: false,
            trade_e3: None,
            pull_from_any_holder: false,
            olp: None,
            trace_async: false,
        });
        let kv = KvHintConfig {
            indexer_url: switchable_indexer(Arc::clone(&reply)).await,
            min_tokens: 256,
            min_incremental_tokens: 512,
            local_prefix_fallback: false,
            indexer_timeout_ms: 500,
            worker_map: p_urls
                .iter()
                .enumerate()
                .map(|(i, url)| KvWorkerMapEntry {
                    id: i as u64 + 1,
                    url: url.clone(),
                    servicer: format!("p{}:18100", i + 1),
                })
                .collect(),
        };
        (Some(kv), Some(reply), Some(path), Some(dir))
    } else {
        (None, None, None, None)
    };
    let late = match opts.late {
        Some(rule) => LateDecodeConfig {
            mode: LateDecodeMode::Choose,
            rule: Some(rule),
            fit_native: opts.fit_native,
            native_state_poll: opts.fit_native,
            ..LateDecodeConfig::default()
        },
        None => LateDecodeConfig::default(),
    };
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
            kv_hint,
            co_schedule: None,
            pd_decode_policy: PdDecodePolicy::AffinityMinLoad,
            pd_decode_tokens: Some(PdDecodeTokensConfig::default()),
            pd_decision_replay: None,
            symmetric_pd: Some(flex),
            gateway_cache_aware: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig {
            late_decode: late,
            ..ProxyConfig::default()
        },
        active_load: ActiveLoadConfig::default(),
    };
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(20)).unwrap());
    let ctx = Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies));
    if let Some(late) = ctx.late_bind.as_ref() {
        assert!(late.symmetric());
        for endpoint in &links {
            assert!(
                wait_until(Duration::from_secs(10), || {
                    late.link_status(endpoint).is_some_and(|s| s.up)
                })
                .await,
                "the router link to the mock owner {endpoint} never came up"
            );
        }
        if opts.fit_native {
            let want = opts.d_capacity.iter().filter(|c| c.is_some()).count();
            assert!(
                wait_until(Duration::from_secs(10), || late.capacity_store().len()
                    == want)
                .await,
                "the capacity poll never saw every decode"
            );
        }
    }
    Fixture {
        app: build_router(Arc::clone(&ctx)),
        ctx,
        p_urls,
        d_urls,
        seen,
        transfers,
        d_capacity,
        indexer,
        trace,
        costs,
        _tmp: tmp,
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

async fn wait_until(limit: Duration, mut cond: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + limit;
    while Instant::now() < deadline {
        if cond() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    cond()
}

impl Fixture {
    async fn post(&self, body: Value, header: Option<(&str, &str)>) -> Response {
        self.post_to(COMPLETIONS, body, header).await
    }

    async fn post_to(&self, uri: &str, body: Value, header: Option<(&str, &str)>) -> Response {
        let mut req = Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json");
        if let Some((name, value)) = header {
            req = req.header(name, value);
        }
        self.app
            .clone()
            .oneshot(
                req.body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }
}

fn request(rid: &str, prompt: Vec<u32>) -> Value {
    json!({"model": "tiny", "prompt": prompt, "max_tokens": 4, "rid": rid})
}

async fn body_json(res: Response) -> Value {
    let bytes = tokio::time::timeout(Duration::from_secs(10), res.into_body().collect())
        .await
        .expect("body within 10 s")
        .expect("complete body")
        .to_bytes();
    serde_json::from_slice(&bytes).unwrap_or(Value::Null)
}

fn opts(late: Option<LateDecodeRule>, d_capacity: Vec<Option<(u64, u64)>>) -> Opts {
    Opts {
        late,
        fit_native: false,
        n_p: 1,
        d_capacity,
        kv_sessions: false,
        gateway_ca: false,
        cost_function: false,
    }
}

// ---------------------------------------------------------------------------
// Late D choose on symmetric workers.
// ---------------------------------------------------------------------------

/// `fit_req` on observed capacity over symmetric decodes: the P-leg names
/// every decode in `kv_export.candidates`, each D-leg carries its own
/// `kv_import` (the matching attempt) and `late_d.cand_idx`, and the router
/// binds the decode with the fewest allocated requests among those that fit,
/// whatever the arrival pick.
#[tokio::test]
async fn flex_late_d_fit_req_native_binds_the_fewest_requests_that_fit() {
    let f = fixture(Opts {
        fit_native: true,
        ..opts(
            Some(LateDecodeRule::FitReq),
            // d0: 5 requests; d1: does not fit (64 + 512 > 100); d2: 2 requests.
            vec![Some((5, 100_000)), Some((1, 100)), Some((2, 100_000))],
        )
    })
    .await;
    let res = f.post(request("r-fit", (0..64).collect()), None).await;
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(body_json(res).await["choices"][0]["text"], "x");

    let p = f.seen.of(|b| b.get("kv_export").is_some());
    assert_eq!(p.len(), 1);
    let (p_url, p_body) = &p[0];
    assert_eq!(p_url, &f.p_urls[0]);
    assert!(p_body.get("kv_import").is_none());
    assert!(p_body.get("kv_session").is_none());
    let export = &p_body["kv_export"];
    let id = export["transfer_id"].as_u64().unwrap();
    assert!(export.get("reader_worker").is_none() && export.get("attempt_id").is_none());
    let cands = export["candidates"].as_array().unwrap();
    assert_eq!(cands.len(), 3);
    let late = &p_body["late_d"];
    assert_eq!(late["k"], 3);
    assert_eq!(late["mode"], "choose");
    assert_eq!(late["local_ok"], false);
    for (i, c) in cands.iter().enumerate() {
        assert_eq!(c["cand_idx"], i);
        assert_eq!(c["reader_worker"], f.d_urls[i]);
        assert_ne!(c["attempt_id"].as_u64().unwrap(), 0);
    }

    let mut legs = f.seen.of(|b| b.get("kv_import").is_some());
    legs.sort_by_key(|(_, b)| b["late_d"]["cand_idx"].as_u64());
    assert_eq!(legs.len(), 3);
    for (i, (url, b)) in legs.iter().enumerate() {
        assert_eq!(url, &f.d_urls[i]);
        assert!(b.get("kv_export").is_none());
        assert!(b.get("kv_pull_hint").is_none());
        assert_eq!(b["kv_import"]["transfer_id"].as_u64(), Some(id));
        assert_eq!(b["kv_import"]["owner_worker"], f.p_urls[0]);
        assert_eq!(b["kv_import"]["attempt_id"], cands[i]["attempt_id"]);
        assert_eq!(b["late_d"]["cand_idx"], i);
        assert_eq!(b["prompt"], json!((0..64).collect::<Vec<u32>>()));
        assert_eq!(b["rid"], "r-fit");
    }
    let late = f.ctx.late_bind.as_ref().unwrap();
    assert_eq!(late.stats().legs_winner.load(Ordering::Relaxed), 1);
    assert_eq!(late.stats().decisions_decided.load(Ordering::Relaxed), 1);
    assert_eq!(late.stats().fit_none.load(Ordering::Relaxed), 0);
    let metrics = late.render_metrics();
    assert!(
        metrics.contains("sgl_router_late_d_requests_total{path=\"choose\"} 1"),
        "{metrics}"
    );
    // The decode with 2 requests won: one has 5, one does not fit.
    let want = f
        .d_urls
        .iter()
        .position(|u| f.d_capacity[u] == Some((2, 100_000)))
        .unwrap() as u64;
    assert_eq!(f.transfers.get(id).winner, Some(want));
    assert!(
        wait_until(Duration::from_secs(5), || {
            late.stats().legs_released.load(Ordering::Relaxed) == 2
        })
        .await
    );
}

/// A native `/generate` request under late D choose takes the plain Split:
/// both legs go to the engines' `/generate`, no leg carries `late_d`, and
/// one decode gets the request. A `/v1/completions` request after it still
/// takes the choose dispatch.
#[tokio::test]
async fn flex_late_d_generate_takes_the_plain_split() {
    let f = fixture(opts(Some(LateDecodeRule::FitReq), vec![None, None])).await;
    let res = f
        .post_to(
            GENERATE,
            json!({"input_ids": [1, 2, 3, 4], "sampling_params": {"max_new_tokens": 4},
                   "rid": "r-gen"}),
            None,
        )
        .await;
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(body_json(res).await["choices"][0]["text"], "x");
    let legs = f.seen.of(|b| b["rid"] == "r-gen");
    assert_eq!(legs.len(), 2, "{legs:?}");
    for (_, b) in &legs {
        assert_eq!(b["mock_route"], GENERATE);
        assert!(b.get("late_d").is_none());
    }
    let p = f
        .seen
        .of(|b| b["rid"] == "r-gen" && b.get("kv_export").is_some());
    assert_eq!(p.len(), 1);
    assert!(p[0].1["kv_export"].get("candidates").is_none());
    assert_eq!(
        f.seen
            .of(|b| b["rid"] == "r-gen" && b.get("kv_import").is_some())
            .len(),
        1
    );
    let late = f.ctx.late_bind.as_ref().unwrap();
    assert_eq!(late.stats().legs_winner.load(Ordering::Relaxed), 0);

    let res = f.post(request("r-choose", vec![1, 2, 3, 4]), None).await;
    assert_eq!(res.status(), StatusCode::OK);
    body_json(res).await;
    let p = f
        .seen
        .of(|b| b["rid"] == "r-choose" && b.get("kv_export").is_some());
    assert_eq!(p.len(), 1);
    assert_eq!(p[0].1["late_d"]["mode"], "choose");
    assert_eq!(late.stats().legs_winner.load(Ordering::Relaxed), 1);
}

/// Without observed capacity, `fit_req` native falls back to `d1` and
/// counts it; every request still binds exactly one decode.
#[tokio::test]
async fn flex_late_d_without_capacity_falls_back_to_d1() {
    let f = fixture(Opts {
        fit_native: true,
        ..opts(Some(LateDecodeRule::FitReq), vec![None, None])
    })
    .await;
    for i in 0..2 {
        let res = f
            .post(request(&format!("r{i}"), vec![1, 2, 3, 4]), None)
            .await;
        assert_eq!(res.status(), StatusCode::OK);
        body_json(res).await;
    }
    let late = f.ctx.late_bind.as_ref().unwrap();
    assert_eq!(late.stats().legs_winner.load(Ordering::Relaxed), 2);
    assert_eq!(late.stats().fit_none.load(Ordering::Relaxed), 2);
}

// ---------------------------------------------------------------------------
// Session retention on symmetric_pd.
// ---------------------------------------------------------------------------

/// Late D: the P-leg alone carries `kv_session` (from the `x-kv-session-context` alias header,
/// rid unchanged); no candidate body does; a decode change between turns
/// queues the old decode home for release (one decode copy per agent).
#[tokio::test]
async fn flex_late_d_session_tag_rides_on_the_p_leg_and_moves_the_decode_home() {
    let f = fixture(Opts {
        kv_sessions: true,
        ..opts(Some(LateDecodeRule::Keep), vec![None, None, None])
    })
    .await;
    for step in 1..=2u64 {
        let header = format!(r#"{{"agent_id":"c1","subagent_id":0,"request_id":{step}}}"#);
        let res = f
            .post(
                request(&format!("client-{step}"), vec![1, 2, 3, 4]),
                Some(("x-kv-session-context", header.as_str())),
            )
            .await;
        assert_eq!(res.status(), StatusCode::OK);
        body_json(res).await;
    }
    let p = f.seen.of(|b| b.get("kv_export").is_some());
    assert_eq!(p.len(), 2);
    for (i, (_, b)) in p.iter().enumerate() {
        assert_eq!(b["kv_session"]["version"], i as u64 + 1, "{b}");
        assert_eq!(b["rid"], format!("client-{}", i + 1));
    }
    let legs = f.seen.of(|b| b.get("kv_import").is_some());
    assert_eq!(legs.len(), 6);
    assert!(legs.iter().all(|(_, b)| b.get("kv_session").is_none()));
    let sessions = f.ctx.symmetric_pd.as_ref().unwrap().kv_sessions().unwrap();
    assert_eq!(sessions.stats().tagged.load(Ordering::Relaxed), 2);
    // Keep binds the round-robin arrival pick: two different decodes, so
    // the first home is queued for release (gated on the P's READY, which
    // the mock never answers).
    assert_eq!(sessions.stats().retire_queued.load(Ordering::Relaxed), 1);
    assert_eq!(sessions.retire_backlog().pending, 1);
}

/// The plain symmetric_pd Split reads the `x-kv-session-context` alias: both legs
/// carry the tag and the body rid is untouched; a malformed alias is 400.
#[tokio::test]
async fn flex_split_reads_the_kv_session_context_alias() {
    let f = fixture(Opts {
        kv_sessions: true,
        ..opts(None, vec![None])
    })
    .await;
    let header = r#"{"agent_id":"c1","subagent_id":0,"request_id":1}"#;
    let res = f
        .post(
            request("client-rid", vec![1, 2, 3]),
            Some(("x-kv-session-context", header)),
        )
        .await;
    assert_eq!(res.status(), StatusCode::OK);
    body_json(res).await;
    assert!(
        wait_until(Duration::from_secs(5), || f
            .seen
            .bodies
            .lock()
            .unwrap()
            .len()
            == 2)
        .await
    );
    let bodies = f.seen.of(|_| true);
    let tags: Vec<&Value> = bodies.iter().map(|(_, b)| &b["kv_session"]).collect();
    assert_eq!(tags[0], tags[1]);
    assert_eq!(tags[0]["version"], 1);
    assert!(bodies.iter().all(|(_, b)| b["rid"] == "client-rid"));
    let res = f
        .post(
            request("bad", vec![1, 2, 3]),
            Some(("x-kv-session-context", r#"{"agent_id":"c1"}"#)),
        )
        .await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

// ---------------------------------------------------------------------------
// --symmetric-pd-p-select gateway-cache-aware.
// ---------------------------------------------------------------------------

/// The gateway rule picks P among the prefill-role workers; a longer prompt
/// with the same prefix goes to the same P (its tree), with or without late
/// D; D stays the configured selection.
#[tokio::test]
async fn flex_gateway_cache_aware_p_select_follows_its_tree() {
    for late in [None, Some(LateDecodeRule::D1)] {
        let f = fixture(Opts {
            n_p: 3,
            gateway_ca: true,
            ..opts(late, vec![None, None])
        })
        .await;
        let first: Vec<u32> = (100..300).collect();
        let res = f.post(request("ca-1", first.clone()), None).await;
        assert_eq!(res.status(), StatusCode::OK);
        body_json(res).await;
        let longer: Vec<u32> = (100..340).collect();
        let res = f.post(request("ca-2", longer), None).await;
        assert_eq!(res.status(), StatusCode::OK);
        body_json(res).await;
        assert!(
            wait_until(Duration::from_secs(5), || {
                f.seen.of(|b| b.get("kv_export").is_some()).len() == 2
            })
            .await
        );
        let p = f.seen.of(|b| b.get("kv_export").is_some());
        assert_eq!(
            p[0].0, p[1].0,
            "late={late:?}: the second prompt follows the tree"
        );
        assert!(f.p_urls.contains(&p[0].0));
    }
}

// ---------------------------------------------------------------------------
// --symmetric-pd-p-select cost-function with late D and session retention.
// ---------------------------------------------------------------------------

/// The cost function (TRADE, lambda 0) picks the P-leg and its pull; late D
/// picks the decode. `busy` has no cached prefix: every P costs 20,000, so
/// the tie goes to the lowest URL (P1), which holds it and reports 20,000
/// remaining tokens (the reserved estimate: the progress row acknowledges
/// it without an update). Then the indexer reports P1 = 18,000 cached tokens:
/// `pull` (20,000 tokens) moves to P2 and pulls 18,000 tokens from P1
/// (8,318 < P1's 22,000; P2 and P3 tie, the lower URL wins). The P-leg
/// alone carries the pull hint and the session tag, the late-D winner
/// answers, and the cost trace has the reservation, the dispatch and the
/// P answer of `pull`, each with the policy label.
#[tokio::test]
async fn late_d_choose_with_cost_function_and_session_retention() {
    let f = fixture(Opts {
        n_p: 3,
        kv_sessions: true,
        cost_function: true,
        ..opts(Some(LateDecodeRule::Keep), vec![None, None])
    })
    .await;
    let trace = f.trace.clone().unwrap();
    let events = || -> Vec<Value> {
        std::fs::read_to_string(&trace)
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    };
    let p1 = Arc::clone(&f.costs[&f.p_urls[0]]);

    // 1. `busy` goes to P1, which holds its P-leg.
    let app = f.app.clone();
    let busy = tokio::spawn(async move {
        let body = serde_json::to_vec(&request("busy", vec![11; 20_000])).unwrap();
        let req = Request::builder()
            .method("POST")
            .uri(COMPLETIONS)
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        app.oneshot(req).await.unwrap()
    });
    assert!(
        wait_until(Duration::from_secs(5), || {
            p1.held.lock().unwrap().contains_key("busy")
        })
        .await,
        "the P-leg of busy never reached P1"
    );
    // P1 reports busy. Its 20,000 remaining tokens equal the reserved
    // estimate (no cached prefix), so the row acknowledges the entry and
    // has no update.
    let p1_url = f.p_urls[0].clone();
    assert!(
        wait_until(Duration::from_secs(5), || {
            events().iter().any(|e| {
                e["event"] == "prefill_progress"
                    && e["worker_url"] == p1_url.as_str()
                    && e["observed_rids"].as_u64() == Some(1)
            })
        })
        .await,
        "P1 never reported the progress of busy"
    );

    // 2. P1 has 18,000 cached tokens; `pull` moves to P2 and pulls them.
    *f.indexer.as_ref().unwrap().lock().unwrap() = json!({"instances": {"1": {"cpu": 18000}}});
    let header = r#"{"agent_id":"c1","subagent_id":0,"request_id":1}"#;
    let res = f
        .post(
            request("pull", vec![11; 20_000]),
            Some(("x-kv-session-context", header)),
        )
        .await;
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(body_json(res).await["choices"][0]["text"], "x");

    let p_leg = |b: &Value| b["rid"] == "pull" && b.get("kv_export").is_some();
    let d_leg = |b: &Value| b["rid"] == "pull" && b.get("kv_import").is_some();
    let p = f.seen.of(p_leg);
    assert_eq!(p.len(), 1);
    let (p_url, p_body) = &p[0];
    assert_eq!(p_url, &f.p_urls[1], "P2 pulls: 8,318 < P1 22,000");
    assert!(p_body.get("late_d").is_some(), "{p_body}");
    assert_eq!(p_body["kv_pull_hint"]["source_worker_id"], f.p_urls[0]);
    assert_eq!(p_body["kv_pull_hint"]["expected_hit_tokens"], 18000);
    assert_eq!(p_body["kv_session"]["version"], 1, "{p_body}");
    let id = p_body["kv_export"]["transfer_id"].as_u64().unwrap();
    let legs = f.seen.of(d_leg);
    assert_eq!(legs.len(), 2);
    for (_, b) in &legs {
        assert!(b.get("kv_pull_hint").is_none(), "{b}");
        assert!(b.get("kv_session").is_none(), "{b}");
        assert_eq!(b["kv_import"]["owner_worker"], f.p_urls[1]);
    }
    assert!(f.transfers.get(id).winner.is_some());
    let sessions = f.ctx.symmetric_pd.as_ref().unwrap().kv_sessions().unwrap();
    assert_eq!(sessions.stats().tagged.load(Ordering::Relaxed), 1);

    // 3. The cost trace of `pull`: reserve, dispatch, P answer.
    assert!(
        wait_until(Duration::from_secs(5), || {
            events()
                .iter()
                .any(|e| e["event"] == "finished" && e["entry"]["rid"] == "pull")
        })
        .await,
        "the P answer of pull is not in the cost trace"
    );
    let rows = events();
    let pull: Vec<&str> = rows
        .iter()
        .filter(|e| e["entry"]["rid"] == "pull")
        .filter_map(|e| e["event"].as_str())
        .collect();
    assert_eq!(pull, ["reserve", "dispatched", "finished"]);
    let reserve = rows
        .iter()
        .find(|e| e["event"] == "reserve" && e["entry"]["rid"] == "pull")
        .unwrap();
    assert_eq!(reserve["trade_decision"]["chosen"], f.p_urls[1]);
    assert_eq!(reserve["trade_decision"]["pulled_tokens"], 18000);
    assert!(rows
        .iter()
        .all(|e| e["policy"] == "symmetric_pd_cost_function"));

    // 4. Release `busy`: its P-leg binds a decode and the client gets 200.
    p1.release.notify_one();
    let res = tokio::time::timeout(Duration::from_secs(10), busy)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    body_json(res).await;
}
