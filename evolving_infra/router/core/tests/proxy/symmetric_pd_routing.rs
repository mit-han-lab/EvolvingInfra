// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! symmetric P/D placement end-to-end at the HTTP layer
//! (`--policy symmetric_pd`).
//!
//! Pins the router-side symmetric P/D contract:
//!
//! * A Split fans out to the P-leg and the D-leg simultaneously; the P-leg
//!   body carries `kv_export`, the D-leg body `kv_import`, both share the
//!   `transfer_id`, neither carries the other mark nor any `bootstrap_*`.
//! * A Local placement is ONE plain request: no marks, no bootstrap fields.
//! * `/v1/completions` with a token-id `prompt` feeds the placer the exact
//!   token count (`kv_import.expected_tokens`).
//! * `POST /symmetric_pd/place` answers the placement contract; 404
//!   `symmetric_pd_disabled` when the policy is not selected.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use bytes::Bytes;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig, SymmetricPdConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::scheduler::RoleWeight;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{SymmetricPdWorkerInfo, WorkerRegistry};
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

use crate::common::mock_worker::MockWorker;

fn config(flex: Option<SymmetricPdConfig>) -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: if flex.is_some() {
                PolicyKind::SymmetricPd
            } else {
                PolicyKind::RoundRobin
            },
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            kv_hint: None,
            pd_decode_policy: sgl_router::config::PdDecodePolicy::AffinityMinLoad,
            pd_decision_replay: None,
            pd_decode_tokens: None,
            symmetric_pd: flex,
            gateway_cache_aware: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    }
}

/// `(url, prefill_weight, decode_weight, bootstrap_port)`.
fn build_ctx(
    flex: Option<SymmetricPdConfig>,
    workers: &[(&str, f64, f64, u16)],
) -> Arc<AppContext> {
    let cfg = config(flex);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for (url, p, d, port) in workers {
        let id = WorkerId(url.to_string());
        registry
            .add(WorkerSpec {
                id: id.clone(),
                url: url.to_string(),
                mode: WorkerMode::Symmetric,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: Some(*port),
            })
            .unwrap();
        registry
            .get(&id)
            .unwrap()
            .set_symmetric_pd(SymmetricPdWorkerInfo {
                weight: RoleWeight {
                    prefill: *p,
                    decode: *d,
                },
                ..Default::default()
            });
    }
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

fn post(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .header("x-request-id", "req-1")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

async fn await_captured_body(mock: &MockWorker, label: &str) -> Bytes {
    let start = std::time::Instant::now();
    loop {
        let captured = mock.captured.lock().unwrap().last_body.clone();
        if let Some(b) = captured {
            return b;
        }
        if start.elapsed() > Duration::from_secs(2) {
            panic!("{label}: no request body captured within 2s");
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

fn parse(b: &Bytes) -> Value {
    serde_json::from_slice(b).expect("body must be valid JSON")
}

async fn read_json(res: axum::response::Response) -> Value {
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

fn assert_no_bootstrap(v: &Value) {
    for k in ["bootstrap_host", "bootstrap_port", "bootstrap_room"] {
        assert!(v.get(k).is_none(), "{k} must not be present: {v}");
    }
}

#[tokio::test]
async fn completions_decode_host_return_is_preserved_on_decode_only() {
    let p = MockWorker::start(vec![]).await;
    let d = MockWorker::start(vec![]).await;
    let flex = SymmetricPdConfig {
        import_cache_policy: sgl_router::config::SymmetricPdImportCachePolicy::Retain,
        decode_host_return: Some(sgl_router::config::DecodeHostReturnConfig {
            servicers: [
                (p.url.clone(), "p:18100".into()),
                (d.url.clone(), "d:19100".into()),
            ]
            .into(),
            timeout_ms: 10_000,
            max_tokens: 65_536,
        }),
        ..Default::default()
    };
    let app = build_router(build_ctx(
        Some(flex),
        &[(&p.url, 1.0, 0.0, 9000), (&d.url, 0.0, 1.0, 9100)],
    ));
    let response = app.oneshot(post("/v1/completions", json!({
        "model":"tiny", "prompt":[11,12,13], "rid":"return-request", "cache_salt":"namespace"
    }))).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let pj = parse(&await_captured_body(&p, "P").await);
    let dj = parse(&await_captured_body(&d, "D").await);
    assert_eq!(
        dj["kv_import"]["host_return"],
        json!({
            "destination_servicer_addr":"p:18100", "source_servicer_addr":"d:19100",
            "timeout_ms":10000, "max_tokens":65536,
        })
    );
    assert_eq!(dj["kv_import"]["cache_policy"], "retain");
    assert_eq!(dj["rid"], "return-request");
    assert_eq!(dj["cache_salt"], "namespace");
    assert!(pj["kv_export"].get("host_return").is_none());
    assert!(pj.get("kv_import").is_none());
}

/// Static 4+4-style weights ({1,0} / {0,1}) force a Split; the token-id
/// prompt gives the placer the exact prompt length.
#[tokio::test]
async fn completions_split_injects_marks_per_leg() {
    let p = MockWorker::start(vec![]).await;
    let d = MockWorker::start(vec![]).await;
    let ctx = build_ctx(
        Some(SymmetricPdConfig::default()),
        &[(&p.url, 1.0, 0.0, 9000), (&d.url, 0.0, 1.0, 9100)],
    );
    let app = build_router(ctx);

    let res = app
        .oneshot(post(
            "/v1/completions",
            json!({"model": "tiny", "prompt": [11, 12, 13, 14, 15], "rid": "r-split",
                   "bootstrap_room": 42}),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK, "D-leg answers the client");

    let pj = parse(&await_captured_body(&p, "p-leg").await);
    let dj = parse(&await_captured_body(&d, "d-leg").await);

    let export = pj.get("kv_export").expect("P-leg carries kv_export");
    assert!(
        pj.get("kv_import").is_none(),
        "P-leg never carries kv_import"
    );
    let import = dj.get("kv_import").expect("D-leg carries kv_import");
    assert!(
        dj.get("kv_export").is_none(),
        "D-leg never carries kv_export"
    );
    assert_no_bootstrap(&pj);
    assert_no_bootstrap(&dj);

    let tid = export["transfer_id"].as_u64().expect("transfer_id");
    assert_eq!(import["transfer_id"].as_u64(), Some(tid));
    let attempt = export["attempt_id"].as_u64().expect("attempt_id");
    assert_ne!(attempt, 0);
    assert_eq!(import["attempt_id"].as_u64(), Some(attempt));
    assert!(tid <= i64::MAX as u64);
    assert_eq!(export["reader_worker"], d.url);
    assert_eq!(export["reader_addr"], "tcp://127.0.0.1:9101");
    assert_eq!(import["owner_worker"], p.url);
    assert_eq!(import["owner_addr"], "tcp://127.0.0.1:9000");
    assert_eq!(
        import["expected_tokens"], 5,
        "token-id prompt length reaches the placer"
    );
    // The prompt itself is forwarded untouched on both legs.
    assert_eq!(pj["prompt"], json!([11, 12, 13, 14, 15]));
    assert_eq!(dj["prompt"], json!([11, 12, 13, 14, 15]));
}

/// Two fully symmetric workers and no hits: Local wins (zero transfer
/// cost). Exactly one plain request, no marks, no bootstrap fields.
#[tokio::test]
async fn completions_local_is_one_plain_request() {
    let a = MockWorker::start(vec![]).await;
    let b = MockWorker::start(vec![]).await;
    let ctx = build_ctx(
        Some(SymmetricPdConfig::default()),
        &[(&a.url, 1.0, 1.0, 9000), (&b.url, 1.0, 1.0, 9000)],
    );
    let app = build_router(ctx);

    let res = app
        .oneshot(post(
            "/v1/completions",
            json!({"model": "tiny", "prompt": [1, 2, 3], "rid": "r-local"}),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // Tie-break: lexically smaller URL takes the Local request.
    let (hit, idle) = if a.url < b.url { (&a, &b) } else { (&b, &a) };
    let v = parse(&await_captured_body(hit, "local").await);
    assert!(v.get("kv_export").is_none() && v.get("kv_import").is_none());
    assert_no_bootstrap(&v);
    assert_eq!(v["prompt"], json!([1, 2, 3]));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        idle.captured.lock().unwrap().last_body.is_none(),
        "Local must not touch the other worker"
    );
}

/// Chat route: same leg-body contract, `messages` retained on both legs.
#[tokio::test]
async fn chat_split_injects_marks_per_leg() {
    let p = MockWorker::start(vec![]).await;
    let d = MockWorker::start(vec![]).await;
    let mut cfg = SymmetricPdConfig::default();
    cfg.placement.force_split = true;
    let ctx = build_ctx(
        Some(cfg),
        &[(&p.url, 1.0, 1.0, 9000), (&d.url, 1.0, 1.0, 9000)],
    );
    let app = build_router(ctx);

    let res = app
        .oneshot(post(
            "/v1/chat/completions",
            json!({"model": "tiny", "messages": [{"role": "user", "content": "hi"}]}),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let (pj, dj) = {
        // force_split + equal weights: lexical tie-break picks the smaller
        // URL as P and the other as D.
        let (pw, dw) = if p.url < d.url { (&p, &d) } else { (&d, &p) };
        (
            parse(&await_captured_body(pw, "p-leg").await),
            parse(&await_captured_body(dw, "d-leg").await),
        )
    };
    assert!(pj.get("kv_export").is_some() && pj.get("kv_import").is_none());
    assert!(dj.get("kv_import").is_some() && dj.get("kv_export").is_none());
    assert_no_bootstrap(&pj);
    assert_no_bootstrap(&dj);
    assert!(pj.get("messages").is_some() && dj.get("messages").is_some());
    assert_eq!(
        pj["kv_export"]["transfer_id"], dj["kv_import"]["transfer_id"],
        "both legs share the transfer_id"
    );
}

/// The placement contract (`docs/api-reference.md`): `p_leg`, `d_leg`, `local`,
/// `transfer_id`, eight ordered `score_terms`, `marks`, `targets`, `inputs`.
#[tokio::test]
async fn symmetric_pd_place_answers_the_contract() {
    let ctx = build_ctx(
        Some(SymmetricPdConfig::default()),
        &[
            ("http://10.0.0.1:30000", 1.0, 0.0, 8998),
            ("http://10.0.0.5:30000", 0.0, 1.0, 8998),
        ],
    );
    let app = build_router(ctx);

    let res = app
        .clone()
        .oneshot(post(
            "/symmetric_pd/place",
            json!({"input_ids": [1, 2, 3, 4], "session_id": "s", "turn": 3, "rid": "req-abc"}),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let v = read_json(res).await;
    assert_eq!(v["p_leg"]["worker"], "http://10.0.0.1:30000");
    assert_eq!(v["p_leg"]["endpoint"], "http://10.0.0.1:30000/generate");
    assert_eq!(v["d_leg"]["worker"], "http://10.0.0.5:30000");
    assert_eq!(v["local"], false);
    let tid = v["transfer_id"].as_u64().expect("transfer_id for a Split");
    let terms = v["score_terms"].as_array().unwrap();
    let names: Vec<&str> = terms.iter().map(|t| t[0].as_str().unwrap()).collect();
    assert_eq!(
        names,
        [
            "hit_tokens",
            "reuse_gain",
            "backlog",
            "role_pen_p",
            "role_pen_d",
            "xfer_cost",
            "decode_load",
            "p_reason",
            "d_select",
            "score"
        ]
    );
    assert_eq!(v["marks"]["kv_export"]["transfer_id"], tid);
    assert_eq!(
        v["marks"]["kv_export"]["reader_addr"],
        "tcp://10.0.0.5:8999"
    );
    assert_eq!(v["marks"]["kv_import"]["owner_addr"], "tcp://10.0.0.1:8998");
    assert_eq!(v["marks"]["kv_import"]["expected_tokens"], 4);
    let attempt = v["marks"]["kv_export"]["attempt_id"].as_u64().unwrap();
    assert_ne!(attempt, 0);
    assert_eq!(
        v["marks"]["kv_import"]["attempt_id"].as_u64(),
        Some(attempt)
    );
    let targets = v["targets"].as_array().unwrap();
    assert_eq!(targets.len(), 2);
    assert_eq!(targets[0]["leg"], "p");
    assert!(targets[0]["mark"].get("kv_export").is_some());
    assert_eq!(targets[1]["leg"], "d");
    assert!(targets[1]["mark"].get("kv_import").is_some());
    assert_eq!(targets[0]["mark"]["kv_export"], v["marks"]["kv_export"]);
    assert_eq!(targets[1]["mark"]["kv_import"], v["marks"]["kv_import"]);
    assert_eq!(v["inputs"]["prompt_len"], 4);
    assert!(v["inputs"].get("loads").is_some());

    // Local answer: `transfer_id` null, `marks` {}, one `local` target.
    let res = app
        .clone()
        .oneshot(post(
            "/symmetric_pd/place",
            json!({"prompt_len": 100, "force_split": false, "account": false}),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let v = read_json(res).await;
    // Weights {1,0}/{0,1} leave no Local pair: still a Split here.
    assert_eq!(v["local"], false);

    // Malformed body -> 400 in the router envelope.
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/symmetric_pd/place")
                .header("content-type", "application/json")
                .body(Body::from("{not json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn symmetric_pd_place_local_when_symmetric_pair_allows_it() {
    let ctx = build_ctx(
        Some(SymmetricPdConfig::default()),
        &[
            ("http://10.0.0.1:30000", 1.0, 1.0, 8998),
            ("http://10.0.0.2:30000", 1.0, 1.0, 8998),
        ],
    );
    let app = build_router(ctx);
    let res = app
        .oneshot(post("/symmetric_pd/place", json!({"prompt_len": 100})))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let v = read_json(res).await;
    assert_eq!(v["local"], true);
    assert!(v["transfer_id"].is_null());
    assert_eq!(v["marks"], json!({}));
    assert_eq!(v["targets"].as_array().unwrap().len(), 1);
    assert_eq!(v["targets"][0]["leg"], "local");
    assert_eq!(v["targets"][0]["mark"], json!({}));
}

#[tokio::test]
async fn symmetric_pd_place_is_404_without_the_policy() {
    let ctx = build_ctx(None, &[]);
    let app = build_router(ctx);
    let res = app
        .oneshot(post("/symmetric_pd/place", json!({"prompt_len": 10})))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        res.headers()
            .get("x-router-error-code")
            .and_then(|v| v.to_str().ok()),
        Some("symmetric_pd_disabled")
    );
    let v = read_json(res).await;
    assert_eq!(v["error"]["code"], "symmetric_pd_disabled");
}

#[tokio::test]
async fn symmetric_pd_place_503_when_no_pair_can_be_placed() {
    // Decode-only fleet: no prefill-capable worker with weight > 0.
    let ctx = build_ctx(
        Some(SymmetricPdConfig::default()),
        &[("http://10.0.0.1:30000", 0.0, 1.0, 8998)],
    );
    let app = build_router(ctx);
    let res = app
        .oneshot(post("/symmetric_pd/place", json!({"prompt_len": 10})))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    let v = read_json(res).await;
    assert_eq!(v["error"]["code"], "symmetric_pd_no_placement");
}

// ---------------------------------------------------------------------------
// on the direct routes the D-leg's in-flight count (the load of
// `--symmetric-pd-d-select cache-first`) ends when the stream starts.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn decode_reservation_lasts_as_long_as_the_stream() {
    let p = MockWorker::start(vec![]).await;
    // The decode streams 6 chunks, 300 ms apart (1.8 s of decoding).
    let d = MockWorker::start_slow_stream(
        vec![
            "data: a\n\n",
            "data: b\n\n",
            "data: c\n\n",
            "data: d\n\n",
            "data: e\n\n",
            "data: [DONE]\n\n",
        ],
        Duration::from_millis(300),
    )
    .await;
    let ctx = build_ctx(
        Some(SymmetricPdConfig::default()),
        &[(&p.url, 1.0, 0.0, 9000), (&d.url, 0.0, 1.0, 9100)],
    );
    let app = build_router(Arc::clone(&ctx));
    let flex = ctx.symmetric_pd.as_ref().unwrap();
    let res = app
        .oneshot(post(
            "/v1/completions",
            json!({"model": "tiny", "prompt": [1, 2, 3], "rid": "r-stream", "stream": true}),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    // The handler returned the response head; the decode has sent nothing yet.
    let during = flex.backend_load_views()[&d.url].active_decode_reqs;
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let after = flex.backend_load_views()[&d.url].active_decode_reqs;
    eprintln!(
        "active_decode_reqs while the decode streams={during} after the stream={after} \
         stream_bytes={}",
        body.len()
    );
    assert_eq!(
        during, 1,
        "the D-leg is decoding: it must count in its worker's in-flight decodes"
    );
}
