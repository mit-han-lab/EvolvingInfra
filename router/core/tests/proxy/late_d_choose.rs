// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `--pd-late-decode choose` end to end over mock engines that speak the
//! late-D stage-2 protocol.
//!
//! The mock prefill owns a ZMQ ROUTER (its router link). When a P-leg
//! arrives it waits until every candidate leg reached its decode, sends
//! BIND_READY, takes the router's DECISION (or, without one, its default),
//! admits the winner and releases the others (shared room state), and
//! reports FINAL. A mock decode holds its leg until the room is bound, then
//! answers 200 (the winner) or 409 with `X-Late-D-Reason: released`.

use axum::body::Body;
use axum::http::{HeaderValue, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Json;
use bytes::Bytes;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use sgl_router::backend::kv_sessions::KvSessions;
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, GatewayCacheAwareConfig, LateDecodeConfig,
    LateDecodeMode, LateDecodeRule, ModelConfig, ObservabilityConfig, PdDecodePolicy,
    PdDecodeTokensConfig, PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::health::circuit_breaker::CircuitBreakerConfig;
use sgl_router::kv_hint::{KvHintConfig, KvWorkerMapEntry};
use sgl_router::late_bind::LateBind;
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{LateDWorkerInfo, Worker, WorkerRegistry};
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tower::ServiceExt;
use zeromq::{Endpoint, RouterSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

const COMPLETIONS: &str = "/v1/completions";
const GENERATE: &str = "/generate";
const CHOICE_TIMEOUT: Duration = Duration::from_millis(500);

// ---------------------------------------------------------------------------
// Shared room state: which candidates reached their decode, and the winner.
// ---------------------------------------------------------------------------

#[derive(Default, Clone, Copy)]
struct Room {
    /// Candidates whose leg reached their decode (bit = cand_idx).
    arrived: u64,
    /// Candidates whose decode failed the leg before registering.
    gone: u64,
    /// The admitted candidate (ADMIT to it, RELEASE to the others).
    winner: Option<u64>,
}

#[derive(Default)]
struct Rooms(Mutex<HashMap<u64, Room>>);

impl Rooms {
    fn update(&self, room: u64, f: impl FnOnce(&mut Room)) {
        f(self.0.lock().unwrap().entry(room).or_default());
    }

    fn get(&self, room: u64) -> Room {
        self.0
            .lock()
            .unwrap()
            .get(&room)
            .copied()
            .unwrap_or_default()
    }

    async fn wait_for<T>(
        &self,
        room: u64,
        timeout: Duration,
        f: impl Fn(&Room) -> Option<T>,
    ) -> Option<T> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(v) = f(&self.get(room)) {
                return Some(v);
            }
            if Instant::now() >= deadline {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
}

// ---------------------------------------------------------------------------
// The prefill's router link: a ZMQ ROUTER, as the engine's LateDLink.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct LinkState {
    epoch: Option<u64>,
    has_router: bool,
    waiters: HashMap<(u64, u64), oneshot::Sender<u64>>,
    hellos: u64,
    finals: u64,
}

struct Link {
    port: u16,
    out: mpsc::Sender<Vec<u8>>,
    state: Arc<Mutex<LinkState>>,
    /// Answer no hello (the router never sees the link up).
    mute: Arc<AtomicBool>,
    p_seq: AtomicU64,
}

fn frame(v: Value) -> Vec<u8> {
    rmp_serde::to_vec_named(&v).unwrap()
}

impl Link {
    async fn start(mute: bool) -> Arc<Self> {
        let mut socket = RouterSocket::new();
        let bound = socket.bind("tcp://127.0.0.1:0").await.unwrap();
        let Endpoint::Tcp(_, port) = bound else {
            panic!("tcp endpoint expected")
        };
        let (mut tx, mut rx) = socket.split();
        let (out, mut out_rx) = mpsc::channel::<Vec<u8>>(64);
        let state: Arc<Mutex<LinkState>> = Arc::default();
        let mute = Arc::new(AtomicBool::new(mute));
        let (st, muted) = (Arc::clone(&state), Arc::clone(&mute));
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
                        assert_eq!(m["v"], 1, "link version");
                        let reply = match m["t"].as_str().unwrap() {
                            "hello" if !muted.load(Ordering::SeqCst) => {
                                ident = Some(id);
                                let mut s = st.lock().unwrap();
                                s.epoch = m["router_epoch"].as_u64();
                                s.has_router = true;
                                s.hellos += 1;
                                assert!(m["router_id"].as_str().is_some());
                                assert!(m["rules"].is_array());
                                Some(frame(json!({"v": 1, "t": "hello_ack",
                                    "router_epoch": m["router_epoch"],
                                    "engine_boot_ns": 7, "mode": "pd", "kv_ip": "127.0.0.1"})))
                            }
                            "ping" => Some(frame(json!({"v": 1, "t": "pong", "ts_ns": m["ts_ns"]}))),
                            "decision" => {
                                let mut s = st.lock().unwrap();
                                assert_eq!(m["router_epoch"].as_u64(), s.epoch);
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
        Arc::new(Self {
            port,
            out,
            state,
            mute,
            p_seq: AtomicU64::new(0),
        })
    }

    /// BIND_READY for one room; the router's winner, or `None` without a
    /// router or without an answer within the choice timeout.
    async fn ask(&self, key: u64, eligible: u64, default_idx: u64) -> Option<u64> {
        let (tx, rx) = oneshot::channel();
        let epoch = {
            let mut s = self.state.lock().unwrap();
            if !s.has_router || self.mute.load(Ordering::SeqCst) {
                return None;
            }
            s.waiters.insert((key, 1), tx);
            s.epoch
        };
        let p_seq = self.p_seq.fetch_add(1, Ordering::SeqCst) + 1;
        self.out
            .send(frame(
                json!({"v": 1, "t": "bind_ready", "router_epoch": epoch,
                "p_seq": p_seq, "t_send_ns": 1u64,
                "items": [{"key": key, "bind_seq": 1, "eligible": eligible,
                           "ready": eligible.count_ones(), "input_len": 4,
                           "default_idx": default_idx, "local_ok": false, "local_load": 0}]}),
            ))
            .await
            .unwrap();
        tokio::time::timeout(CHOICE_TIMEOUT, rx).await.ok()?.ok()
    }

    async fn final_(&self, key: u64, admitted: u64, reason: &str) {
        let epoch = {
            let mut s = self.state.lock().unwrap();
            if !s.has_router {
                return;
            }
            s.finals += 1;
            s.epoch
        };
        self.out
            .send(frame(json!({"v": 1, "t": "final", "router_epoch": epoch,
                "items": [{"key": key, "bind_seq": 1, "admitted_idx": admitted,
                           "reason": reason, "t_admit_ns": 2u64}]})))
            .await
            .unwrap();
    }
}

// ---------------------------------------------------------------------------
// Mock engines.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Seen {
    prefill: Mutex<Vec<Value>>,
    decode: Mutex<Vec<(String, Value)>>,
}

fn in_list(body: &Value, key: &str, idx: u64) -> bool {
    body.get(key)
        .and_then(Value::as_array)
        .is_some_and(|a| a.iter().any(|v| v.as_u64() == Some(idx)))
}

fn late_reason(status: StatusCode, reason: &'static str) -> Response {
    let mut r = (
        status,
        Json(json!({"object": "error", "message": format!("late_d {reason}"), "code": status.as_u16()})),
    )
        .into_response();
    r.headers_mut()
        .insert("x-late-d-reason", HeaderValue::from_static(reason));
    r
}

fn ok_answer(body: &Value) -> Response {
    if !body.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        return Json(json!({"choices": [{"index": 0, "text": "x"}]})).into_response();
    }
    let chunks = [
        Ok::<_, std::io::Error>(Bytes::from_static(
            b"data: {\"choices\":[{\"index\":0,\"text\":\"x\"}]}\n\n",
        )),
        Ok(Bytes::from_static(b"data: [DONE]\n\n")),
    ];
    let mut r = Response::new(Body::from_stream(futures::stream::iter(chunks)));
    r.headers_mut().insert(
        "content-type",
        HeaderValue::from_static("text/event-stream"),
    );
    r
}

async fn prefill_handler(
    link: Arc<Link>,
    rooms: Arc<Rooms>,
    seen: Arc<Seen>,
    body: Value,
) -> Response {
    seen.prefill.lock().unwrap().push(body.clone());
    if body["mock_p"] == "fail" {
        return late_reason(StatusCode::SERVICE_UNAVAILABLE, "bind_timeout");
    }
    if body["mock_p"] == "reader_gone" {
        // The symmetric owner ends its P-leg because its reader left; that
        // reader (the default candidate) serves by local recompute.
        let room = body["bootstrap_room"].as_u64().unwrap();
        let default_idx = body["late_d"]["default_idx"].as_u64().unwrap();
        rooms.update(room, |r| r.winner = Some(default_idx));
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"object": "error", "message": "reader left",
                "type": "symmetric_reader_gone", "code": 500})),
        )
            .into_response();
    }
    let Some(late) = body.get("late_d") else {
        return Json(json!({"ok": true})).into_response();
    };
    let room = body["bootstrap_room"].as_u64().unwrap();
    let k = late["k"].as_u64().unwrap();
    let default_idx = late["default_idx"].as_u64().unwrap();
    let full = if k >= 64 { u64::MAX } else { (1u64 << k) - 1 };
    // Every candidate registers (or fails) first.
    let eligible = rooms
        .wait_for(room, Duration::from_secs(3), |r| {
            ((r.arrived | r.gone) == full).then_some(r.arrived)
        })
        .await
        .unwrap_or_else(|| rooms.get(room).arrived);
    if eligible == 0 {
        // Nothing to bind: the decode legs carry the failure.
        tokio::time::sleep(Duration::from_secs(1)).await;
        return Json(json!({"ok": true})).into_response();
    }
    let fallback = if eligible >> default_idx & 1 == 1 {
        default_idx
    } else {
        u64::from(eligible.trailing_zeros())
    };
    let (winner, reason) = match link.ask(room, eligible, default_idx).await {
        Some(w) if w < 64 && eligible >> w & 1 == 1 => (w, "decided"),
        Some(255) => (fallback, "unknown_key"),
        Some(w) => panic!("decision names a candidate that is not eligible: {w}"),
        None => (fallback, "timeout"),
    };
    rooms.update(room, |r| r.winner = Some(winner));
    link.final_(room, winner, reason).await;
    Json(json!({"ok": true})).into_response()
}

async fn decode_handler(me: String, rooms: Arc<Rooms>, seen: Arc<Seen>, body: Value) -> Response {
    seen.decode.lock().unwrap().push((me, body.clone()));
    let Some(idx) = body.pointer("/late_d/cand_idx").and_then(Value::as_u64) else {
        return ok_answer(&body);
    };
    let room = body["bootstrap_room"].as_u64().unwrap();
    if in_list(&body, "mock_dead", idx) {
        rooms.update(room, |r| r.gone |= 1 << idx);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": {"message": format!("candidate {idx} failed")}})),
        )
            .into_response();
    }
    rooms.update(room, |r| r.arrived |= 1 << idx);
    if in_list(&body, "mock_hold", idx) {
        tokio::time::sleep(Duration::from_secs(60)).await;
    }
    match rooms
        .wait_for(room, Duration::from_secs(10), |r| r.winner)
        .await
    {
        Some(w) if w == idx => {
            tokio::time::sleep(Duration::from_millis(20)).await;
            ok_answer(&body)
        }
        Some(_) => late_reason(StatusCode::CONFLICT, "released"),
        None => late_reason(StatusCode::SERVICE_UNAVAILABLE, "ttl"),
    }
}

async fn serve(app: axum::Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    url
}

/// Minimal dynamo indexer: every POST /query gets `reply`.
async fn mock_indexer(reply: Value) -> String {
    serve(axum::Router::new().route(
        "/query",
        post(move |_b: Bytes| {
            let reply = reply.clone();
            async move { Json(reply) }
        }),
    ))
    .await
}

// ---------------------------------------------------------------------------
// The router under test.
// ---------------------------------------------------------------------------

struct Fixture {
    app: axum::Router,
    ctx: Arc<AppContext>,
    link: Arc<Link>,
    prefill: Arc<Worker>,
    /// Decode workers in URL order (= cand_idx order with K = N).
    decode: Vec<Arc<Worker>>,
    seen: Arc<Seen>,
}

#[derive(Clone, Copy)]
struct Opts {
    late: LateDecodeConfig,
    mute_link: bool,
    /// Decode engines that advertise late D choose.
    decode_choose: bool,
    kv_hint: bool,
    /// The context gets a session KV registry, as `--kv-session-retention`.
    kv_sessions: bool,
    /// `--policy gateway_cache_aware` (with `kv_sessions`: its config turns
    /// the registry on) instead of `round_robin`.
    gateway_ca: bool,
}

fn choose(rule: LateDecodeRule) -> Opts {
    Opts {
        late: LateDecodeConfig {
            mode: LateDecodeMode::Choose,
            rule: Some(rule),
            candidates: None,
            ..LateDecodeConfig::default()
        },
        mute_link: false,
        decode_choose: true,
        kv_hint: false,
        kv_sessions: false,
        gateway_ca: false,
    }
}

async fn fixture(n: usize, opts: Opts) -> Fixture {
    let link = Link::start(opts.mute_link).await;
    let rooms: Arc<Rooms> = Arc::default();
    let seen: Arc<Seen> = Arc::default();
    let p_url = {
        let (link, rooms, seen) = (Arc::clone(&link), Arc::clone(&rooms), Arc::clone(&seen));
        let (glink, grooms, gseen) = (Arc::clone(&link), Arc::clone(&rooms), Arc::clone(&seen));
        serve(
            axum::Router::new()
                .route(
                    COMPLETIONS,
                    post(move |Json(body): Json<Value>| {
                        prefill_handler(
                            Arc::clone(&link),
                            Arc::clone(&rooms),
                            Arc::clone(&seen),
                            body,
                        )
                    }),
                )
                .route(
                    GENERATE,
                    post(move |Json(mut body): Json<Value>| {
                        body["mock_route"] = json!(GENERATE);
                        prefill_handler(
                            Arc::clone(&glink),
                            Arc::clone(&grooms),
                            Arc::clone(&gseen),
                            body,
                        )
                    }),
                ),
        )
        .await
    };
    let mut d_urls = Vec::new();
    for _ in 0..n {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (me, rooms, seen) = (url.clone(), Arc::clone(&rooms), Arc::clone(&seen));
        let (gme, grooms, gseen) = (me.clone(), Arc::clone(&rooms), Arc::clone(&seen));
        let app = axum::Router::new()
            .route(
                COMPLETIONS,
                post(move |Json(body): Json<Value>| {
                    decode_handler(me.clone(), Arc::clone(&rooms), Arc::clone(&seen), body)
                }),
            )
            .route(
                GENERATE,
                post(move |Json(mut body): Json<Value>| {
                    body["mock_route"] = json!(GENERATE);
                    decode_handler(gme.clone(), Arc::clone(&grooms), Arc::clone(&gseen), body)
                }),
            );
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        d_urls.push(url);
    }
    let kv_hint = if opts.kv_hint {
        let indexer = mock_indexer(json!({"instances": {"2": {"gpu": 0, "cpu": 8192}}})).await;
        Some(KvHintConfig {
            indexer_url: indexer,
            min_tokens: 256,
            min_incremental_tokens: 0,
            local_prefix_fallback: false,
            indexer_timeout_ms: 500,
            worker_map: vec![
                KvWorkerMapEntry {
                    id: 1,
                    url: p_url.clone(),
                    servicer: "p1:18100".into(),
                },
                KvWorkerMapEntry {
                    id: 2,
                    url: "http://owner:30000".into(),
                    servicer: "owner:18100".into(),
                },
            ],
        })
    } else {
        None
    };
    let d1 = opts.late.rule == Some(LateDecodeRule::D1);
    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: if opts.gateway_ca {
                PolicyKind::GatewayCacheAware
            } else {
                PolicyKind::RoundRobin
            },
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            kv_hint,
            co_schedule: None,
            pd_decode_policy: PdDecodePolicy::PowerOfTwoSmg,
            pd_decode_tokens: d1.then(PdDecodeTokensConfig::default),
            pd_decision_replay: None,
            symmetric_pd: None,
            gateway_cache_aware: opts.gateway_ca.then(|| GatewayCacheAwareConfig {
                max_tree_size: 66_163_200,
                kv_session_retention: opts.kv_sessions,
                ..GatewayCacheAwareConfig::default()
            }),
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig {
            late_decode: opts.late,
            ..ProxyConfig::default()
        },
        active_load: ActiveLoadConfig::default(),
    };
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    // One failure opens a breaker: every breaker tick is visible.
    let cb = || {
        Some(CircuitBreakerConfig {
            threshold: NonZeroU32::new(1).unwrap(),
            cool_down: Duration::from_secs(600),
        })
    };
    registry
        .add_with_cb(
            WorkerSpec {
                id: WorkerId("p0".into()),
                url: p_url.clone(),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: Some(8998),
            },
            cb(),
        )
        .unwrap();
    for (i, url) in d_urls.iter().enumerate() {
        registry
            .add_with_cb(
                WorkerSpec {
                    id: WorkerId(format!("d{i}")),
                    url: url.clone(),
                    mode: WorkerMode::Decode,
                    model_ids: vec![ModelId("tiny".into())],
                    bootstrap_port: None,
                },
                cb(),
            )
            .unwrap();
    }
    let info = |port: Option<u16>| LateDWorkerInfo {
        mode: "choose".into(),
        control_port: port,
        kv_ip: Some("127.0.0.1".into()),
    };
    let prefill = registry.get(&WorkerId("p0".into())).unwrap();
    prefill.set_late_d(Some(info(Some(link.port))));
    let mut decode: Vec<Arc<Worker>> = (0..n)
        .map(|i| registry.get(&WorkerId(format!("d{i}"))).unwrap())
        .collect();
    for d in &decode {
        d.set_late_d(opts.decode_choose.then(|| info(None)));
    }
    decode.sort_by(|a, b| a.url.cmp(&b.url));
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(20)).unwrap());
    let mut ctx = AppContext::new(cfg, tokenizers, proxy, registry, policies);
    if opts.gateway_ca {
        // The policy config alone builds the registry.
        assert_eq!(ctx.kv_sessions.is_some(), opts.kv_sessions);
    } else if opts.kv_sessions {
        ctx.kv_sessions = Some(KvSessions::spawn(None));
    }
    let ctx = Arc::new(ctx);
    if opts.late.choose() && !opts.mute_link {
        let late = Arc::clone(ctx.late_bind.as_ref().unwrap());
        let endpoint = format!("tcp://127.0.0.1:{}", link.port);
        let up = wait_until(Duration::from_secs(10), || {
            late.link_status(&endpoint).is_some_and(|s| s.up)
        })
        .await;
        assert!(up, "the router link to the mock prefill never came up");
    }
    Fixture {
        app: build_router(Arc::clone(&ctx)),
        ctx,
        link,
        prefill,
        decode,
        seen,
    }
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
    fn late(&self) -> &Arc<LateBind> {
        self.ctx.late_bind.as_ref().unwrap()
    }

    fn stat(&self, f: impl Fn(&sgl_router::late_bind::LateStats) -> &AtomicU64) -> u64 {
        f(self.late().stats()).load(Ordering::Relaxed)
    }

    async fn post(&self, route: &str, body: Value) -> Response {
        self.app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(route)
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    async fn post_header(&self, route: &str, body: Value, name: &str, value: &str) -> Response {
        self.app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(route)
                    .header("content-type", "application/json")
                    .header(name, value)
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    fn prefill_bodies(&self) -> Vec<Value> {
        self.seen.prefill.lock().unwrap().clone()
    }

    /// Decode-leg bodies of the room, by `cand_idx`.
    fn legs_of(&self, room: u64) -> Vec<(String, Value)> {
        let mut legs: Vec<(String, Value)> = self
            .seen
            .decode
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, b)| b["bootstrap_room"].as_u64() == Some(room))
            .cloned()
            .collect();
        legs.sort_by_key(|(_, b)| b.pointer("/late_d/cand_idx").and_then(Value::as_u64));
        legs
    }
}

fn request(stream: bool, extra: Value) -> Value {
    let mut body = json!({
        "model": "tiny",
        "prompt": [11, 12, 13, 14],
        "max_tokens": 4,
        "stream": stream,
        "rid": "rid-late",
    });
    for (k, v) in extra.as_object().unwrap() {
        body[k] = v.clone();
    }
    body
}

async fn body_text(res: Response) -> String {
    let bytes = tokio::time::timeout(Duration::from_secs(10), res.into_body().collect())
        .await
        .expect("body within 10 s")
        .expect("complete body")
        .to_bytes();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn decode_url_header(res: &Response) -> Option<String> {
    res.headers()
        .get("x-sgl-decode-url")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

fn error_code(res: &Response) -> Option<String> {
    res.headers()
        .get("x-router-error-code")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

/// Choose + d1 over all decodes. The least-charged candidate
/// wins at the bind; the others are released (409) and tick no breaker;
/// the stream, the charge and the protocol fields are right.
#[tokio::test]
async fn choose_d1_binds_the_least_charged_candidate() {
    let f = fixture(3, choose(LateDecodeRule::D1)).await;
    // Other traffic charged d0 and d2: d1 has the fewest tokens.
    let _c0 = f.decode[0].charge_decode_tokens(50_000);
    let _c2 = f.decode[2].charge_decode_tokens(40_000);
    let mut rooms = Vec::new();
    for (round, stream) in [true, false].into_iter().enumerate() {
        let res = f.post(COMPLETIONS, request(stream, json!({}))).await;
        assert_eq!(res.status(), StatusCode::OK, "stream={stream}");
        assert_eq!(
            decode_url_header(&res).as_deref(),
            Some(f.decode[1].url.as_str())
        );
        let text = body_text(res).await;
        if stream {
            assert!(text.contains("[DONE]"), "{text}");
        } else {
            assert_eq!(
                serde_json::from_str::<Value>(&text).unwrap()["choices"][0]["text"],
                "x"
            );
        }
        // Charge released with the stream; the entry left the table.
        assert!(
            wait_until(Duration::from_secs(5), || f.decode[1]
                .decode_charged_tokens()
                == 0)
            .await
        );
        assert_eq!(f.late().pending(), 0);

        // The P-leg names K, the arrival pick and the router epoch.
        let p = f.prefill_bodies()[round].clone();
        let late = &p["late_d"];
        assert_eq!(late["v"], 1);
        assert_eq!(late["mode"], "choose");
        assert_eq!(late["k"], 3);
        assert_eq!(late["router_epoch"], f.late().router_epoch());
        assert!(late["default_idx"].as_u64().unwrap() < 3);
        assert!(late.get("cand_idx").is_none());
        assert_eq!(p["prompt"], json!([11, 12, 13, 14]));
        let room = p["bootstrap_room"].as_u64().unwrap();
        rooms.push(room);
        // One D-leg per candidate, URL order, same room and bootstrap.
        let legs = f.legs_of(room);
        assert_eq!(legs.len(), 3);
        for (i, (url, leg)) in legs.iter().enumerate() {
            assert_eq!(
                url, &f.decode[i].url,
                "cand_idx {i} goes to the i-th decode by URL"
            );
            assert_eq!(leg["late_d"]["cand_idx"], i as u64);
            assert_eq!(leg["late_d"]["k"], 3);
            assert_eq!(leg["late_d"]["router_epoch"], late["router_epoch"]);
            assert!(leg["late_d"].get("default_idx").is_none());
            assert_eq!(leg["bootstrap_host"], p["bootstrap_host"]);
            assert_eq!(leg["bootstrap_port"], p["bootstrap_port"]);
            assert_eq!(
                leg["prompt"],
                json!([11, 12, 13, 14]),
                "not packed by default"
            );
            assert!(leg.get("kv_pull_hint").is_none());
        }
    }
    assert_ne!(rooms[0], rooms[1]);
    // FINAL arrives over the link (reason decided, no mismatch).
    assert!(
        wait_until(Duration::from_secs(5), || f.stat(|s| &s.finals_decided)
            == 2)
        .await
    );
    assert_eq!(f.stat(|s| &s.decisions_decided), 2);
    assert_eq!(f.stat(|s| &s.legs_winner), 2);
    assert!(wait_until(Duration::from_secs(5), || f.stat(|s| &s.legs_released) == 4).await);
    assert_eq!(f.stat(|s| &s.final_mismatch), 0);
    assert_eq!(f.stat(|s| &s.bind_mismatch), 0);
    assert_eq!(f.stat(|s| &s.legs_dead), 0);
    // R4 + R11: released legs tick no breaker.
    for w in f.decode.iter().chain([&f.prefill]) {
        assert_eq!(w.breaker.snapshot().state_code, 0, "{}", w.url);
    }
    // /metrics carries the late-D families and the charges.
    let metrics = body_text(
        f.app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap(),
    )
    .await;
    assert!(
        metrics.contains("sgl_router_late_d_decisions_total{outcome=\"decided\"} 2"),
        "{metrics}"
    );
    assert!(metrics.contains("sgl_router_late_d_requests_total{path=\"choose\"} 2"));
    assert!(metrics.contains("sgl_router_late_d_link_up{p=\"tcp://127.0.0.1:"));
}

/// `fit_req` with a fixed capacity (the gateway rule): among the candidates
/// whose charged tokens plus the request's charge (4 prompt + 4 output
/// tokens) fit, the fewest charged requests wins at the bind, where `d1`
/// would take the fewest tokens; a candidate over capacity is skipped.
#[tokio::test]
async fn choose_fit_req_binds_the_fewest_requests_that_fit() {
    let mut opts = choose(LateDecodeRule::FitReq);
    opts.late.fit_capacity_tokens = Some(10_000);
    let f = fixture(3, opts).await;
    // d0: three light requests; d1: one medium one; d2: one heavy one that
    // leaves no room for the request.
    let _c0: Vec<_> = (0..3)
        .map(|_| f.decode[0].charge_decode_tokens(1_000))
        .collect();
    let _c1 = f.decode[1].charge_decode_tokens(5_000);
    let _c2 = f.decode[2].charge_decode_tokens(9_995);
    let res = f.post(COMPLETIONS, request(false, json!({}))).await;
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(
        decode_url_header(&res).as_deref(),
        Some(f.decode[1].url.as_str())
    );
    let _ = body_text(res).await;
    assert_eq!(f.stat(|s| &s.decisions_decided), 1);
    assert_eq!(f.stat(|s| &s.fit_none), 0);
    // The charge left with the response.
    assert!(
        wait_until(Duration::from_secs(5), || f.decode[1]
            .decode_charged_tokens()
            == 5_000)
        .await
    );
}

/// Without the router link (the prefill never acks the hello) the prefill
/// commits its default, the arrival pick, after its choice timeout; the
/// request completes on that candidate.
#[tokio::test]
async fn choose_without_a_link_binds_the_arrival_pick() {
    let mut opts = choose(LateDecodeRule::D1);
    opts.mute_link = true;
    let f = fixture(2, opts).await;
    let res = f.post(COMPLETIONS, request(true, json!({}))).await;
    assert_eq!(res.status(), StatusCode::OK);
    let p = f.prefill_bodies()[0].clone();
    let default_idx = p["late_d"]["default_idx"].as_u64().unwrap() as usize;
    assert_eq!(
        decode_url_header(&res).as_deref(),
        Some(f.decode[default_idx].url.as_str())
    );
    assert!(body_text(res).await.contains("[DONE]"));
    assert_eq!(f.stat(|s| &s.decisions_decided), 0);
    assert_eq!(f.stat(|s| &s.bind_undecided), 1);
    assert_eq!(f.link.state.lock().unwrap().hellos, 0);
}

/// R8 + R11: the prefill fails (503 with `X-Late-D-Reason`) while every
/// candidate still holds: the client gets the prefill's status at once,
/// the legs are dropped, and the prefill's breaker is not ticked.
#[tokio::test]
async fn choose_prefill_failure_answers_at_once_and_ticks_no_breaker() {
    let f = fixture(2, choose(LateDecodeRule::D1)).await;
    let started = Instant::now();
    let res = f
        .post(
            COMPLETIONS,
            request(true, json!({"mock_p": "fail", "mock_hold": [0, 1]})),
        )
        .await;
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(error_code(&res).as_deref(), Some("prefill_failed"));
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "{:?}",
        started.elapsed()
    );
    assert_eq!(
        f.prefill.breaker.snapshot().state_code,
        0,
        "R11: no breaker tick"
    );
    assert_eq!(f.stat(|s| &s.prefill_failed), 1);
    assert_eq!(f.stat(|s| &s.prefill_leg_failed), 1);
    assert_eq!(f.late().pending(), 0);
    for d in &f.decode {
        assert_eq!(d.decode_charged_reqs(), 0);
    }
}

/// A P-leg that the engine ended because its reader left
/// (`symmetric_reader_gone`, HTTP 500) is cancelled, not failed: no R8, the
/// D-leg that recomputes answers the client (200), the prefill breaker is
/// not ticked (3 such answers would open it), and the counters say so.
#[tokio::test]
async fn choose_reader_gone_prefill_end_keeps_the_decode_legs() {
    let f = fixture(2, choose(LateDecodeRule::D1)).await;
    for i in 0..3 {
        let res = f
            .post(
                COMPLETIONS,
                request(i % 2 == 0, json!({"mock_p": "reader_gone"})),
            )
            .await;
        assert_eq!(res.status(), StatusCode::OK, "request {i}");
        let text = body_text(res).await;
        assert!(text.contains("\"text\""), "request {i}: {text}");
    }
    assert_eq!(
        f.prefill.breaker.snapshot().state_code,
        0,
        "no breaker tick"
    );
    assert_eq!(f.stat(|s| &s.prefill_reader_gone), 3);
    assert_eq!(f.stat(|s| &s.prefill_failed), 0);
    assert_eq!(f.stat(|s| &s.prefill_leg_failed), 0);
    assert!(wait_until(Duration::from_secs(5), || f.late().pending() == 0).await);
}

/// R5: a failed candidate is dead (excluded from the choice, its breaker
/// ticked); the request completes on another. R6: when every candidate
/// fails, the client gets the last failure.
#[tokio::test]
async fn choose_dead_candidates_and_no_winner() {
    let f = fixture(3, choose(LateDecodeRule::D1)).await;
    // d0 would win on load, but it fails its leg.
    let _c1 = f.decode[1].charge_decode_tokens(9_000);
    let _c2 = f.decode[2].charge_decode_tokens(5_000);
    let res = f
        .post(COMPLETIONS, request(false, json!({"mock_dead": [0]})))
        .await;
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(
        decode_url_header(&res).as_deref(),
        Some(f.decode[2].url.as_str())
    );
    assert_eq!(f.stat(|s| &s.legs_dead), 1);
    assert_eq!(
        f.decode[0].breaker.snapshot().state_code,
        1,
        "R5: d0 ticked"
    );
    assert_eq!(f.decode[2].breaker.snapshot().state_code, 0);

    let f = fixture(2, choose(LateDecodeRule::Keep)).await;
    let res = f
        .post(COMPLETIONS, request(true, json!({"mock_dead": [0, 1]})))
        .await;
    assert_eq!(res.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert!(body_text(res).await.contains("failed"));
    assert_eq!(f.stat(|s| &s.no_winner), 1);
    assert_eq!(f.late().pending(), 0);
}

/// Choose mode serves /v1/completions only (chat: 501) and one request at
/// a time (a batch prompt: 400).
#[tokio::test]
async fn choose_refuses_chat_and_batches() {
    let f = fixture(2, choose(LateDecodeRule::Po2)).await;
    let res = f
        .post(
            "/v1/chat/completions",
            json!({"model": "tiny", "messages": [{"role": "user", "content": "hi"}]}),
        )
        .await;
    assert_eq!(res.status(), StatusCode::NOT_IMPLEMENTED);
    assert_eq!(
        error_code(&res).as_deref(),
        Some("late_d_route_unsupported")
    );
    let res = f
        .post(
            COMPLETIONS,
            json!({"model": "tiny", "prompt": [[1, 2], [3]], "max_tokens": 1}),
        )
        .await;
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        error_code(&res).as_deref(),
        Some("late_d_batch_unsupported")
    );
    assert!(f.prefill_bodies().is_empty());
    // A po2 bind still completes (no polled loads: request counts).
    let res = f
        .post(COMPLETIONS, request(true, json!({"late_d": {"bogus": 1}})))
        .await;
    assert_eq!(res.status(), StatusCode::OK);
    assert!(body_text(res).await.contains("[DONE]"));
    // The client's late_d field never reaches an engine.
    assert!(f.prefill_bodies()[0]["late_d"].get("bogus").is_none());
    assert_eq!(f.stat(|s| &s.decisions_decided), 1);
}

/// A native `/generate` request takes the plain two-leg dispatch: both legs
/// go to the engines' `/generate` without `late_d`, one decode gets it.
#[tokio::test]
async fn choose_sends_generate_through_the_plain_dispatch() {
    let f = fixture(2, choose(LateDecodeRule::D1)).await;
    let res = f
        .post(
            GENERATE,
            json!({"input_ids": [11, 12, 13, 14], "sampling_params": {"max_new_tokens": 4},
                   "rid": "rid-gen"}),
        )
        .await;
    assert_eq!(res.status(), StatusCode::OK);
    assert!(body_text(res).await.contains("\"x\""));
    let p = f.prefill_bodies()[0].clone();
    assert_eq!(p["mock_route"], GENERATE);
    assert!(p.get("late_d").is_none());
    let legs = f.legs_of(p["bootstrap_room"].as_u64().unwrap());
    assert_eq!(legs.len(), 1);
    assert_eq!(legs[0].1["mock_route"], GENERATE);
    assert!(legs[0].1.get("late_d").is_none());
    assert_eq!(f.stat(|s| &s.requests_choose), 0);
}

/// The pull hint rides on the P-leg only, as in the plain dispatch; packed
/// prompts ride on the D-legs only.
#[tokio::test]
async fn choose_keeps_the_pull_hint_on_the_prefill_and_packs_decode_prompts() {
    let mut opts = choose(LateDecodeRule::Keep);
    opts.kv_hint = true;
    opts.late.packed_prompt = true;
    let f = fixture(2, opts).await;
    let res = f.post(COMPLETIONS, request(true, json!({}))).await;
    assert_eq!(res.status(), StatusCode::OK);
    assert!(body_text(res).await.contains("[DONE]"));
    let p = f.prefill_bodies()[0].clone();
    assert_eq!(p["kv_pull_hint"]["source_worker_id"], "http://owner:30000");
    assert_eq!(p["prompt"], json!([11, 12, 13, 14]));
    let legs = f.legs_of(p["bootstrap_room"].as_u64().unwrap());
    assert_eq!(legs.len(), 2);
    for (_, leg) in &legs {
        assert!(leg.get("kv_pull_hint").is_none());
        assert!(leg.get("prompt").is_none());
        assert_eq!(leg["prompt_len"], 4);
        // base64 of int32 LE 11, 12, 13, 14.
        assert_eq!(leg["prompt_ids_b64"], "CwAAAAwAAAANAAAADgAAAA==");
    }
}

/// Late D off: the same engines get one decode leg chosen at arrival and
/// no `late_d` field (with packed prompts as the only change when that
/// flag is set). Choose mode with decodes that do not run late D choose:
/// the plain two-leg dispatch.
#[tokio::test]
async fn off_and_plain_fallback_send_one_decode_leg() {
    for packed in [false, true] {
        let mut opts = choose(LateDecodeRule::D1);
        opts.late = LateDecodeConfig {
            packed_prompt: packed,
            ..LateDecodeConfig::default()
        };
        let f = fixture(2, opts).await;
        assert!(f.ctx.late_bind.is_none());
        let res = f.post(COMPLETIONS, request(false, json!({}))).await;
        assert_eq!(res.status(), StatusCode::OK);
        let p = f.prefill_bodies()[0].clone();
        assert!(p.get("late_d").is_none());
        let legs = f.legs_of(p["bootstrap_room"].as_u64().unwrap());
        assert_eq!(legs.len(), 1, "packed={packed}");
        assert!(legs[0].1.get("late_d").is_none());
        assert_eq!(legs[0].1.get("prompt").is_none(), packed);
        assert_eq!(p["prompt"], json!([11, 12, 13, 14]));
    }
    let mut opts = choose(LateDecodeRule::D1);
    opts.decode_choose = false;
    let f = fixture(2, opts).await;
    let res = f.post(COMPLETIONS, request(true, json!({}))).await;
    assert_eq!(res.status(), StatusCode::OK);
    assert!(body_text(res).await.contains("[DONE]"));
    let p = f.prefill_bodies()[0].clone();
    assert!(p.get("late_d").is_none());
    assert_eq!(f.legs_of(p["bootstrap_room"].as_u64().unwrap()).len(), 1);
    assert_eq!(f.stat(|s| &s.requests_plain), 1);
    assert_eq!(f.stat(|s| &s.requests_choose), 0);
}

/// `--kv-session-retention`: the session tag rides on the P-leg only (with
/// `late_d` and the pull hint); no D-leg carries it, and the client rid
/// reaches every leg unchanged. The plain fallback (decodes without late D
/// choose) tags only its P-leg too.
#[tokio::test]
async fn kv_session_tag_rides_on_the_prefill_leg_only() {
    let identity = r#"{"agent_id":"agent-1","subagent_id":0,"request_id":0}"#;
    for decode_choose in [true, false] {
        let mut opts = choose(LateDecodeRule::Keep);
        opts.kv_hint = true;
        opts.kv_sessions = true;
        opts.decode_choose = decode_choose;
        let f = fixture(2, opts).await;
        let body = request(true, json!({}));
        let header = sgl_router::agent_identity::KV_SESSION_HEADER;
        let res = f.post_header(COMPLETIONS, body, header, identity).await;
        assert_eq!(res.status(), StatusCode::OK, "{decode_choose}");
        assert!(body_text(res).await.contains("[DONE]"));
        let p = f.prefill_bodies()[0].clone();
        assert_eq!(p["kv_session"]["version"], 1, "{decode_choose}");
        assert!(p["kv_session"]["session_id"].is_string());
        assert_eq!(p["kv_pull_hint"]["source_worker_id"], "http://owner:30000");
        assert_eq!(p.get("late_d").is_some(), decode_choose);
        assert_eq!(p["rid"], "rid-late");
        let legs = f.legs_of(p["bootstrap_room"].as_u64().unwrap());
        assert_eq!(legs.len(), if decode_choose { 2 } else { 1 });
        for (_, leg) in &legs {
            assert!(leg.get("kv_session").is_none(), "D-leg tagged");
            assert!(leg.get("kv_pull_hint").is_none());
            assert_eq!(leg["rid"], "rid-late");
        }
        let stats = f.ctx.kv_sessions.as_ref().unwrap().stats();
        assert_eq!(stats.tagged.load(Ordering::Relaxed), 1);
        assert_eq!(stats.untagged_plain.load(Ordering::Relaxed), 0);
    }
}

/// `--policy gateway_cache_aware` + late D choose `fit_req` +
/// `--kv-session-retention`: the P is chosen by the ported gateway rule (the
/// only P here), the P-leg carries `late_d` and the session tag, the D is
/// bound at the prefill's BIND_READY by `fit_req` exactly as with the other
/// P policies, and no D-leg is tagged. A second step of the same session
/// advances the version on the same P.
#[tokio::test]
async fn gateway_cache_aware_composes_with_choose_fit_req_and_kv_session() {
    let mut opts = choose(LateDecodeRule::FitReq);
    opts.late.fit_capacity_tokens = Some(10_000);
    opts.kv_sessions = true;
    opts.gateway_ca = true;
    let f = fixture(2, opts).await;
    assert!(
        format!("{:?}", f.ctx.policies.get(&ModelId("tiny".into())).unwrap())
            .contains("GatewayCacheAwarePolicy")
    );
    let _busy = f.decode[0].charge_decode_tokens(9_990);
    let header = sgl_router::agent_identity::KV_SESSION_HEADER;
    for step in 0..2u64 {
        let identity = format!(r#"{{"agent_id":"agent-g","subagent_id":0,"request_id":{step}}}"#);
        let mut prompt: Vec<u64> = (100..140).collect();
        prompt.extend(1000 + 10 * step..1000 + 10 * step + 10);
        let body = request(false, json!({"prompt": prompt, "cache_salt": "sess-g"}));
        let res = f.post_header(COMPLETIONS, body, header, &identity).await;
        assert_eq!(res.status(), StatusCode::OK, "step {step}");
        assert_eq!(
            decode_url_header(&res).as_deref(),
            Some(f.decode[1].url.as_str()),
            "fit_req binds the decode with room"
        );
        let _ = body_text(res).await;
        let p = f.prefill_bodies()[step as usize].clone();
        assert!(p.get("late_d").is_some());
        assert_eq!(p["kv_session"]["version"], step + 1);
        let legs = f.legs_of(p["bootstrap_room"].as_u64().unwrap());
        assert_eq!(legs.len(), 2);
        for (_, leg) in &legs {
            assert!(leg.get("kv_session").is_none(), "D-leg tagged");
        }
    }
    let stats = f.ctx.kv_sessions.as_ref().unwrap().stats();
    assert_eq!(stats.tagged.load(Ordering::Relaxed), 2);
    assert_eq!(stats.retire_queued.load(Ordering::Relaxed), 0);
}
