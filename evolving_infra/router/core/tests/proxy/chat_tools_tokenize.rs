// SPDX-License-Identifier: Apache-2.0

//! `--chat-tools-tokenize`: a chat request with tools gets its placement ids
//! from the engine's `/v1/tokenize`, and the engine still tokenizes the
//! request it serves (no `input_ids` in the forwarded body).

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry;
use sgl_router::policies::kv_events::{BlockSizeOracle, HashTree};
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{Worker, WorkerRegistry};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tower::ServiceExt;

use crate::common::cache_aware_fixture::{config, MODEL};

#[derive(Default)]
struct Seen {
    tokenize: Vec<Value>,
    chat: Vec<Value>,
}

/// An engine that answers `/v1/tokenize` with `tokens` and serves chat.
async fn engine(tokens: Value) -> (String, Arc<Mutex<Seen>>) {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let tokenize = move |State(seen): State<Arc<Mutex<Seen>>>, Json(body): Json<Value>| {
        let tokens = tokens.clone();
        async move {
            seen.lock().unwrap().tokenize.push(body);
            Json(json!({"tokens": tokens, "count": 3, "max_model_len": 4096}))
        }
    };
    let chat = |State(seen): State<Arc<Mutex<Seen>>>, Json(body): Json<Value>| async move {
        seen.lock().unwrap().chat.push(body);
        Json(json!({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}))
    };
    let app = Router::new()
        .route("/v1/tokenize", post(tokenize))
        .route("/v1/chat/completions", post(chat))
        .with_state(Arc::clone(&seen));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, seen)
}

fn ctx(url: String, tools_tokenize: bool) -> Arc<AppContext> {
    let mut cfg = config();
    cfg.proxy.chat_tools_tokenize = tools_tokenize;
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(WorkerSpec {
        id: WorkerId(url.clone()),
        url,
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId(MODEL.into())],
        bootstrap_port: None,
    });
    let policies = Arc::new(
        build_registry(
            &cfg,
            Arc::new(HashTree::new()),
            Arc::clone(&tokenizers),
            BlockSizeOracle::new(),
        )
        .unwrap(),
    );
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

async fn send(ctx: Arc<AppContext>, body: Value) -> StatusCode {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    build_router(ctx).oneshot(req).await.unwrap().status()
}

fn tool_request() -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "weather in Paris?"}],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "tool_choice": "auto",
        "max_tokens": 16,
    })
}

#[tokio::test]
async fn tool_request_is_tokenized_by_the_engine_and_served_without_input_ids() {
    let (url, seen) = engine(json!([5, 6, 7])).await;
    assert_eq!(send(ctx(url, true), tool_request()).await, StatusCode::OK);
    let seen = seen.lock().unwrap();
    assert_eq!(
        seen.tokenize,
        vec![json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "tool_choice": "auto",
        })]
    );
    assert_eq!(seen.chat.len(), 1);
    assert!(seen.chat[0].get("input_ids").is_none(), "{}", seen.chat[0]);
    assert!(seen.chat[0].get("tools").is_some());
}

#[tokio::test]
async fn without_the_flag_or_without_tools_no_engine_tokenize() {
    let (url, seen) = engine(json!([5, 6, 7])).await;
    assert_eq!(
        send(ctx(url.clone(), false), tool_request()).await,
        StatusCode::OK
    );
    let mut plain = tool_request();
    plain.as_object_mut().unwrap().remove("tools");
    assert_eq!(send(ctx(url, true), plain).await, StatusCode::OK);
    let seen = seen.lock().unwrap();
    assert!(seen.tokenize.is_empty(), "{:?}", seen.tokenize);
    assert_eq!(seen.chat.len(), 2);
}

#[tokio::test]
async fn chat_ids_returns_the_engine_ids_and_none_on_failure() {
    let worker = |url: String| {
        vec![Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(url.clone()),
            url,
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId(MODEL.into())],
            bootstrap_port: None,
        }))]
    };
    let (url, _) = engine(json!([5, 6, 7])).await;
    assert_eq!(
        sgl_router::engine_tokenize::chat_ids(&worker(url), &tool_request()).await,
        Some(vec![5, 6, 7])
    );
    // A reply that is not one flat id array, and an engine that is not there.
    let (url, _) = engine(json!([[5], [6]])).await;
    assert_eq!(
        sgl_router::engine_tokenize::chat_ids(&worker(url), &tool_request()).await,
        None
    );
    let gone = worker("http://127.0.0.1:1".into());
    assert_eq!(
        sgl_router::engine_tokenize::chat_ids(&gone, &tool_request()).await,
        None
    );
    assert_eq!(
        sgl_router::engine_tokenize::chat_ids(&[], &tool_request()).await,
        None
    );
}
