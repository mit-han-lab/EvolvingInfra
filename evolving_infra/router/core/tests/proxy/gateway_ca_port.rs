// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `--policy gateway_cache_aware` (the ported sgl-model-gateway cache-aware
//! prefill rule) end to end at the HTTP layer with mock engines: plain PD
//! dispatch over three prefill workers and one decode worker.
//!
//! * A later step of the same session (same salt, longer prompt) goes to the
//!   prefill that served the first step (match rate above 0.3).
//! * The load signal is the router's in-flight prefill count
//!   (`Worker::active_requests`): when that prefill holds 70 more requests
//!   than the others (imbalanced: 70 > 64 and 70 > 1.5 * 0), the request
//!   goes to the least-loaded prefill.
//! * With `--kv-session-retention` in the policy config, the context builds
//!   the session registry and only the P body carries `kv_session`; a P
//!   change (here forced by imbalance) queues the old P for retirement.
//!
//! The choose-mode (late D) composition is in `late_d_choose.rs`.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use sgl_router::agent_identity::KV_SESSION_HEADER;
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, GatewayCacheAwareConfig, ModelConfig,
    ObservabilityConfig, PdDecodePolicy, PolicyKind, ProxyConfig, ServerConfig,
    StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tower::ServiceExt;

use crate::common::mock_worker::MockWorker;

const COMPLETIONS: &str = "/v1/completions";

fn config(kv_session_retention: bool) -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: PolicyKind::GatewayCacheAware,
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            kv_hint: None,
            pd_decode_policy: PdDecodePolicy::AffinityMinLoad,
            pd_decision_replay: None,
            pd_decode_tokens: None,
            symmetric_pd: None,
            gateway_cache_aware: Some(GatewayCacheAwareConfig {
                max_tree_size: 66_163_200,
                kv_session_retention,
                ..GatewayCacheAwareConfig::default()
            }),
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    }
}

struct Fixture {
    app: axum::Router,
    ctx: Arc<AppContext>,
    registry: Arc<WorkerRegistry>,
    prefill: Vec<MockWorker>,
    decode: MockWorker,
}

async fn fixture(kv_session_retention: bool) -> Fixture {
    let mut prefill = Vec::new();
    for _ in 0..3 {
        prefill.push(MockWorker::start(vec![]).await);
    }
    let decode = MockWorker::start(vec![]).await;
    let cfg = config(kv_session_retention);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for (i, p) in prefill.iter().enumerate() {
        registry
            .add(WorkerSpec {
                id: WorkerId(format!("p{i}")),
                url: p.url.clone(),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: Some(8997),
            })
            .unwrap();
    }
    registry
        .add(WorkerSpec {
            id: WorkerId("d0".into()),
            url: decode.url.clone(),
            mode: WorkerMode::Decode,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        })
        .unwrap();
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let ctx = Arc::new(AppContext::new(
        cfg,
        tokenizers,
        proxy,
        Arc::clone(&registry),
        policies,
    ));
    Fixture {
        app: build_router(Arc::clone(&ctx)),
        ctx,
        registry,
        prefill,
        decode,
    }
}

fn completion(session: &str, prompt: &[u32], rid: &str) -> Value {
    json!({
        "model": "tiny",
        "prompt": prompt,
        "cache_salt": session,
        "max_tokens": 1,
        "rid": rid,
    })
}

fn post(body: &Value, identity: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri(COMPLETIONS)
        .header("content-type", "application/json");
    if let Some(identity) = identity {
        builder = builder.header(KV_SESSION_HEADER, identity);
    }
    builder
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap()
}

impl Fixture {
    /// Send one request; return the index of the prefill mock that got it
    /// and its body. The decode body is consumed too.
    async fn send(&self, body: &Value, identity: Option<&str>) -> (usize, Value) {
        let res = self
            .app
            .clone()
            .oneshot(post(body, identity))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            for (i, p) in self.prefill.iter().enumerate() {
                let taken = p.captured.lock().unwrap().last_body.take();
                if let Some(bytes) = taken {
                    let _ = self.decode.captured.lock().unwrap().last_body.take();
                    return (i, serde_json::from_slice(&bytes).unwrap());
                }
            }
            assert!(Instant::now() < deadline, "no prefill body");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Wait until no prefill holds a request (the P legs are detached).
    async fn settle(&self) {
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            let busy: usize = (0..3)
                .map(|i| {
                    self.registry
                        .get(&WorkerId(format!("p{i}")))
                        .unwrap()
                        .active_load()
                })
                .sum();
            if busy == 0 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("prefill loads did not drain");
    }
}

#[tokio::test]
async fn same_session_follows_the_cache_and_imbalance_takes_the_least_loaded() {
    let f = fixture(false).await;
    assert!(f.ctx.kv_sessions.is_none());
    let base: Vec<u32> = (100..164).collect();
    let (first, _) = f.send(&completion("sess-a", &base, "a0"), None).await;
    f.settle().await;
    // Steps 2..4: the prompt grows; 64 of at most 94 tokens match (> 0.3).
    for step in 1..4u32 {
        let mut prompt = base.clone();
        prompt.extend(1000 * step..1000 * step + 10 * step);
        let (p, body) = f.send(&completion("sess-a", &prompt, "a"), None).await;
        assert_eq!(p, first, "step {step} left the cached prefill");
        assert_eq!(body["prompt"], json!(prompt));
        assert!(body.get("kv_session").is_none());
        f.settle().await;
    }
    // 70 requests held on the cached prefill: imbalanced.
    let hot = f.registry.get(&WorkerId(format!("p{first}"))).unwrap();
    let held: Vec<_> = (0..70).map(|_| hot.load_guard()).collect();
    let (p, _) = f.send(&completion("sess-a", &base, "a-hot"), None).await;
    assert_ne!(p, first, "the overloaded prefill was chosen");
    drop(held);
    f.settle().await;
    // The imbalanced pick inserted the prompt for that prefill, so with the
    // loads balanced again the exact prompt now matches there too; the
    // deepest node still belongs to its creator (`first`).
    let (p, _) = f.send(&completion("sess-a", &base, "a-back"), None).await;
    assert_eq!(p, first);
    // Another salt never matches: a cold miss is placed by load.
    let (_, body) = f.send(&completion("sess-b", &base, "b0"), None).await;
    assert_eq!(body["cache_salt"], "sess-b");
}

#[tokio::test]
async fn kv_session_retention_in_the_policy_config_tags_the_p_body_only() {
    let f = fixture(true).await;
    assert!(
        f.ctx.kv_sessions.is_some(),
        "the policy config builds the registry"
    );
    let base: Vec<u32> = (500..600).collect();
    let mut first = None;
    for step in 0..3u64 {
        let identity = format!(r#"{{"agent_id":"agent-k","subagent_id":0,"request_id":{step}}}"#);
        let mut prompt = base.clone();
        prompt.extend((0..step as u32).map(|t| 7000 + t));
        let (p, body) = f
            .send(&completion("sess-k", &prompt, "k"), Some(&identity))
            .await;
        assert_eq!(*first.get_or_insert(p), p, "same session, same prefill");
        assert_eq!(body["kv_session"]["version"], step + 1);
        assert_eq!(body["rid"], "k");
        f.settle().await;
    }
    let stats = f.ctx.kv_sessions.as_ref().unwrap().stats();
    assert_eq!(stats.tagged.load(Ordering::Relaxed), 3);
    assert_eq!(stats.retire_queued.load(Ordering::Relaxed), 0);
    // Imbalance moves step 3 to another prefill: the registry queues the
    // old copy for release after the new prefill reports READY.
    let first = first.unwrap();
    let hot = f.registry.get(&WorkerId(format!("p{first}"))).unwrap();
    let held: Vec<_> = (0..70).map(|_| hot.load_guard()).collect();
    let identity = r#"{"agent_id":"agent-k","subagent_id":0,"request_id":3}"#;
    let mut prompt = base.clone();
    prompt.extend(7000..7003);
    let (p, body) = f
        .send(&completion("sess-k", &prompt, "k"), Some(identity))
        .await;
    drop(held);
    assert_ne!(p, first, "the overloaded prefill was chosen");
    assert_eq!(body["kv_session"]["version"], 4);
    assert_eq!(stats.tagged.load(Ordering::Relaxed), 4);
    assert_eq!(stats.retire_queued.load(Ordering::Relaxed), 1);
}
