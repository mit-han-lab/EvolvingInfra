// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! `--kv-session-retention` on the PD path (the session protocol), end to
//! end at the HTTP layer with mock
//! engines. The choose-mode (late D) case is in `late_d_choose.rs`.
//!
//! * Only the prefill-leg body carries `kv_session`; a decode body never.
//! * The client `rid` reaches both legs unchanged: the identity comes from
//!   the `x-kv-session-context` header, and the router never rewrites the
//!   rid for it. The header itself never reaches an engine.
//! * A client `kv_session` field or a malformed header gets 400 before any
//!   leg leaves. A registration error sends the request untagged.
//! * With the flag off the header is ignored and the bodies do not change.
//! * `POST /kv_sessions/end` works with the flag and is refused without it.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::post;
use axum::Json;
use bytes::Bytes;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use sgl_router::agent_identity::KV_SESSION_HEADER;
use sgl_router::backend::kv_sessions::{KvSessionStats, KvSessions};
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig, PdDecodePolicy,
    PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::kv_hint::{KvHintConfig, KvWorkerMapEntry};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tower::ServiceExt;

use crate::common::mock_worker::MockWorker;

const COMPLETIONS: &str = "/v1/completions";

fn config(kv_hint: Option<KvHintConfig>) -> Config {
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
            kv_hint,
            pd_decode_policy: PdDecodePolicy::AffinityMinLoad,
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

fn hint_config(indexer_url: String, worker_map: Vec<KvWorkerMapEntry>) -> KvHintConfig {
    KvHintConfig {
        indexer_url,
        min_tokens: 256,
        min_incremental_tokens: 0,
        local_prefix_fallback: false,
        indexer_timeout_ms: 500,
        worker_map,
    }
}

fn pd_specs(prefill: &str, decode: &str) -> Vec<WorkerSpec> {
    vec![
        WorkerSpec {
            id: WorkerId("p1".into()),
            url: prefill.into(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: Some(8997),
        },
        WorkerSpec {
            id: WorkerId("d1".into()),
            url: decode.into(),
            mode: WorkerMode::Decode,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        },
    ]
}

/// The router under test. `flag` gives the context a session registry as
/// `--kv-session-retention` does; the route reads only that field, not the
/// policy.
fn router(cfg: Config, specs: Vec<WorkerSpec>, flag: bool) -> (axum::Router, Arc<AppContext>) {
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for spec in specs {
        let _ = registry.add(spec);
    }
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let mut ctx = AppContext::new(cfg, tokenizers, proxy, registry, policies);
    if flag {
        ctx.kv_sessions = Some(KvSessions::spawn(None));
    }
    let ctx = Arc::new(ctx);
    (build_router(Arc::clone(&ctx)), ctx)
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

fn identity(agent: &str, step: u64) -> String {
    let value = json!({"agent_id": agent, "subagent_id": 0, "request_id": step});
    value.to_string()
}

fn completion(rid: &str) -> Value {
    json!({
        "model": "tiny",
        "prompt": [11, 12, 13, 14, 15, 16, 17, 18],
        "cache_salt": "sess-1",
        "max_tokens": 1,
        "rid": rid,
    })
}

fn post_to(route: &str, header: Option<&str>, body: &Value) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri(route)
        .header("content-type", "application/json");
    if let Some(identity) = header {
        builder = builder.header(KV_SESSION_HEADER, identity);
    }
    builder
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap()
}

/// The next request body a mock engine receives (taken, so the next call
/// waits for a new one).
async fn next_body(mock: &MockWorker, label: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let body = mock.captured.lock().unwrap().last_body.take();
        if let Some(body) = body {
            return serde_json::from_slice(&body).unwrap();
        }
        assert!(Instant::now() < deadline, "{label}: no body");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

async fn text(res: axum::response::Response) -> String {
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn stat(ctx: &AppContext, f: impl Fn(&KvSessionStats) -> &AtomicU64) -> u64 {
    let stats = ctx.kv_sessions.as_ref().unwrap().stats();
    f(stats).load(Ordering::Relaxed)
}

/// Plain PD dispatch: the P body carries the tag, the D body does not, and
/// both keep the client rid. The same agent on the same P advances the
/// version without a retirement.
#[tokio::test]
async fn pd_path_tags_the_prefill_body_only_and_keeps_the_rid() {
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let (app, ctx) = router(config(None), pd_specs(&prefill.url, &decode.url), true);
    for (step, version) in [(0, 1), (1, 2)] {
        let header = identity("agent-1", step);
        let req = post_to(COMPLETIONS, Some(&header), &completion("client-rid"));
        let res = app.clone().oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let p = next_body(&prefill, "prefill").await;
        let d = next_body(&decode, "decode").await;
        assert_eq!(p["kv_session"]["version"], version);
        assert!(p["kv_session"]["session_id"].is_string());
        assert!(d.get("kv_session").is_none(), "the D body is untagged");
        assert_eq!(p["rid"], "client-rid");
        assert_eq!(d["rid"], "client-rid");
        assert_eq!(p["bootstrap_room"], d["bootstrap_room"]);
        assert_eq!(p["prompt"], d["prompt"]);
    }
    for mock in [&prefill, &decode] {
        let captured = mock.captured.lock().unwrap();
        assert!(!captured.headers.contains_key(KV_SESSION_HEADER));
    }
    assert_eq!(stat(&ctx, |s| &s.tagged), 2);
    assert_eq!(stat(&ctx, |s| &s.retire_queued), 0);
}

/// Hinted PD dispatch: the P body carries the pull hint and the tag; the D
/// body carries neither.
#[tokio::test]
async fn pd_hinted_path_tags_the_prefill_body_only() {
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let indexer = mock_indexer(json!({
        "instances": {"1": {"gpu": 0, "cpu": 0}, "2": {"gpu": 0, "cpu": 8192}}
    }))
    .await;
    let map = vec![
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
    ];
    let cfg = config(Some(hint_config(indexer, map)));
    let (app, ctx) = router(cfg, pd_specs(&prefill.url, &decode.url), true);
    let header = identity("agent-2", 0);
    let req = post_to(COMPLETIONS, Some(&header), &completion("hinted"));
    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let p = next_body(&prefill, "prefill").await;
    let d = next_body(&decode, "decode").await;
    assert_eq!(p["kv_pull_hint"]["source_worker_id"], "http://owner:30000");
    assert_eq!(p["kv_session"]["version"], 1);
    assert!(d.get("kv_pull_hint").is_none());
    assert!(d.get("kv_session").is_none());
    assert_eq!(d["rid"], "hinted");
    assert_eq!(stat(&ctx, |s| &s.tagged), 1);
}

/// A client `kv_session` and a malformed identity header get 400 with a
/// clear message, and no leg leaves.
#[tokio::test]
async fn client_tag_and_malformed_header_get_400_before_any_leg() {
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let (app, ctx) = router(config(None), pd_specs(&prefill.url, &decode.url), true);
    let mut forged = completion("forged");
    forged["kv_session"] = json!({"session_id": "x", "version": 9});
    let header = identity("agent-3", 0);
    let res = app
        .clone()
        .oneshot(post_to(COMPLETIONS, Some(&header), &forged))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let message = text(res).await;
    assert!(
        message.contains("kv_session is assigned by the router"),
        "{message}"
    );
    for bad in ["{not json", r#"{"agent_id":"a","subagent_id":0}"#] {
        let req = post_to(COMPLETIONS, Some(bad), &completion("bad"));
        let res = app.clone().oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST, "{bad}");
        let message = text(res).await;
        assert!(
            message.contains("invalid x-kv-session-context header"),
            "{message}"
        );
        assert!(message.contains("expected {"), "{message}");
    }
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(prefill.captured.lock().unwrap().last_body.is_none());
    assert!(decode.captured.lock().unwrap().last_body.is_none());
    assert_eq!(stat(&ctx, |s| &s.tagged), 0);
}

/// Flag off: the header (valid or not) is ignored, no body is tagged, the
/// rid is unchanged, and the bodies equal those of a request without the
/// header (the bootstrap room aside). END answers 400.
#[tokio::test]
async fn flag_off_ignores_the_header_and_keeps_the_bodies() {
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let (app, ctx) = router(config(None), pd_specs(&prefill.url, &decode.url), false);
    assert!(ctx.kv_sessions.is_none());
    let valid = identity("agent-4", 0);
    let mut seen = Vec::new();
    for header in [None, Some(valid.as_str()), Some("{not json")] {
        let req = post_to(COMPLETIONS, header, &completion("client-rid"));
        let res = app.clone().oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK, "{header:?}");
        let mut p = next_body(&prefill, "prefill").await;
        let mut d = next_body(&decode, "decode").await;
        for body in [&mut p, &mut d] {
            assert!(body.get("kv_session").is_none());
            assert_eq!(body["rid"], "client-rid");
            body.as_object_mut().unwrap().remove("bootstrap_room");
        }
        seen.push((p, d));
    }
    assert_eq!(seen[0], seen[1]);
    assert_eq!(seen[0], seen[2]);
    let end = json!({"agent_id": "agent-4", "subagent_id": 0});
    let req = post_to("/kv_sessions/end", None, &end);
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let message = text(res).await;
    assert!(
        message.contains("KV session retention is disabled"),
        "{message}"
    );
}

/// A stale step (an older request_id of the same agent) fails to register:
/// the request still succeeds, untagged, and the error is counted.
#[tokio::test]
async fn stale_registration_goes_out_untagged_and_is_counted() {
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let (app, ctx) = router(config(None), pd_specs(&prefill.url, &decode.url), true);
    for (step, rid, tagged) in [(5, "r5", true), (3, "r3", false)] {
        let header = identity("agent-5", step);
        let req = post_to(COMPLETIONS, Some(&header), &completion(rid));
        let res = app.clone().oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK, "{rid}");
        let p = next_body(&prefill, "prefill").await;
        let d = next_body(&decode, "decode").await;
        assert_eq!(p.get("kv_session").is_some(), tagged, "{rid}");
        assert!(d.get("kv_session").is_none());
        assert_eq!(p["rid"], rid);
        assert_eq!(d["rid"], rid);
    }
    assert_eq!(stat(&ctx, |s| &s.tagged), 1);
    assert_eq!(stat(&ctx, |s| &s.register_stale), 1);
}

/// END under the flag: 200 for a known agent (its next request goes out
/// untagged), 400 for an unknown one; both are counted.
#[tokio::test]
async fn end_route_ends_a_known_session_and_refuses_an_unknown_one() {
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let (app, ctx) = router(config(None), pd_specs(&prefill.url, &decode.url), true);
    let header = identity("agent-6", 0);
    let req = post_to(COMPLETIONS, Some(&header), &completion("before-end"));
    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let p = next_body(&prefill, "prefill").await;
    assert!(p["kv_session"].is_object());
    next_body(&decode, "decode").await;

    let end = |agent: &str| {
        let body = json!({"agent_id": agent, "subagent_id": 0});
        post_to("/kv_sessions/end", None, &body)
    };
    let res = app.clone().oneshot(end("agent-6")).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let reply: Value = serde_json::from_str(&text(res).await).unwrap();
    assert_eq!(reply["accepted"], true);
    let res = app.clone().oneshot(end("nobody")).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let message = text(res).await;
    assert!(message.contains("unknown KV session"), "{message}");
    assert_eq!(stat(&ctx, |s| &s.end_requests), 1);
    assert_eq!(stat(&ctx, |s| &s.end_unknown), 1);

    let header = identity("agent-6", 1);
    let req = post_to(COMPLETIONS, Some(&header), &completion("after-end"));
    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let p = next_body(&prefill, "prefill").await;
    assert!(p.get("kv_session").is_none(), "ended");
    next_body(&decode, "decode").await;
    assert_eq!(stat(&ctx, |s| &s.register_ended), 1);
}
