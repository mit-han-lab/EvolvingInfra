// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Assembles the Axum HTTP routes for serving, prepared execution, health and
//! observability. Applies route-specific body limits plus ingress/response metrics
//! and payload-rejection logging; request placement remains in route handlers.

use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::routes::chat::MAX_CHAT_BODY_BYTES;
use axum::extract::{DefaultBodyLimit, MatchedPath, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, MethodRouter};
use axum::Router;
use std::sync::Arc;

/// Edge counters: `requests_total{route,method}` at entry (true intake, incl.
/// requests parked/shed/cancelled before dispatch), `responses_total{...,
/// status_code}` on exit (incl. early-exit 400/413/503). Their difference =
/// received-but-not-answered, invisible to post-dispatch `worker_requests_total`.
/// `route` is the matched template (not raw URI) to bound label cardinality.
/// Count request intake and completed responses around the handler chain.
///
/// # Inputs
/// `ctx` supplies metrics; `req` supplies the matched route template and
/// HTTP method; `next` is the remaining middleware/handler chain.
///
/// # Outputs
/// Returns the downstream response unchanged after recording its status.
/// Records intake before awaiting the handler; cancellation before a response
/// therefore leaves an unmatched intake count. Uses bounded route templates
/// rather than raw URIs as metric labels.
async fn count_requests(State(ctx): State<Arc<AppContext>>, req: Request, next: Next) -> Response {
    let method = req.method().as_str().to_owned();
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|m| m.as_str().to_owned())
        .unwrap_or_else(|| "unmatched".to_owned());
    ctx.metrics.record_ingress(&route, &method);
    let resp = next.run(req).await;
    ctx.metrics
        .record_response(&route, &method, resp.status().as_u16());
    resp
}

/// Middleware: log 413 PAYLOAD_TOO_LARGE responses with the request method
/// and URI so an operator investigating "client X gets 413s" has a
/// server-side breadcrumb. The 413 is produced by axum's `DefaultBodyLimit`
/// layer BEFORE the handler runs, so without this we would have no record
/// of which request was rejected.
/// Record oversized-payload rejections that can occur before a route handler.
///
/// # Inputs
/// `req` supplies method and URI; `next` runs the remaining handler chain.
///
/// # Outputs
/// Returns the response unchanged. Logs a warning when the resulting
/// status is HTTP 413; it does not change body limits or retry the request.
async fn log_413(req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let resp = next.run(req).await;
    if resp.status() == StatusCode::PAYLOAD_TOO_LARGE {
        tracing::warn!(
            %method,
            %uri,
            "request rejected with 413 PAYLOAD_TOO_LARGE (body exceeded route limit)",
        );
    }
    resp
}

/// Construct the HTTP route table and attach shared state and middleware.
///
/// # Inputs
/// `ctx` is the shared application context retained by routes and metrics
/// middleware.
///
/// # Outputs
/// Returns a configured Axum router with serving, prepared-execution,
/// cache, health and metrics endpoints. Chat/completions use explicit body
/// limits and HTTP 413 logging. Does not bind a socket or start serving.
pub fn build_router(ctx: Arc<AppContext>) -> Router {
    // With external role control and KV session retention, a role change
    // releases a worker's session references only after its old legs ended.
    // The router sees the end of a prepared leg (lifecycle callbacks) and of a
    // leg it proxies itself, and a decode->prefill drain waits for both
    // (`scheduler::role_drain`). A placement handed to an external ingress
    // has no end the router sees, so `/symmetric_pd/place` is refused there.
    let no_lifecycle = ctx.symmetric_pd.as_ref().is_some_and(|flex| {
        flex.config().external_role_control && flex.config().kv_session_retention
    });
    let tracked = |route: MethodRouter<Arc<AppContext>>| {
        if no_lifecycle {
            route.layer(middleware::from_fn(|_: Request, _: Next| async {
                ApiError::BadRequest(
                    "dynamic role switching with KV session retention cannot serve /symmetric_pd/place: the router does not see the end of a placed request".into(),
                ).into_response()
            }))
        } else {
            route
        }
    };
    Router::new()
        .route("/healthz", get(crate::server::routes::health::healthz))
        .route("/readyz", get(crate::server::routes::health::readyz))
        .route("/metrics", get(crate::server::routes::metrics::metrics))
        .route(
            "/v1/models",
            get(crate::server::routes::models::list_models),
        )
        .route(
            "/v1/tokenize",
            post(crate::server::routes::tokenize::tokenize),
        )
        .route(
            "/v1/detokenize",
            post(crate::server::routes::tokenize::detokenize),
        )
        .route(
            "/v1/chat/completions",
            post(crate::server::routes::chat::chat_completions)
                .layer(DefaultBodyLimit::max(MAX_CHAT_BODY_BYTES))
                .layer(middleware::from_fn(log_413)),
        )
        .route(
            "/v1/completions",
            post(crate::server::routes::completions::completions)
                .layer(DefaultBodyLimit::max(
                    ctx.config.proxy.max_completions_body_bytes,
                ))
                .layer(middleware::from_fn(log_413)),
        )
        .route(
            "/generate",
            post(crate::server::routes::completions::generate)
                .layer(DefaultBodyLimit::max(
                    ctx.config.proxy.max_completions_body_bytes,
                ))
                .layer(middleware::from_fn(log_413)),
        )
        .route(
            "/kv_sessions/end",
            post(crate::server::routes::cache::end_kv_session),
        )
        // `--late-d-image-source-url`: original image bytes for a decode that
        // recomputes a pixel-free late-D leg (404 when the request ended).
        .route(
            crate::image_source::FETCH_PATH,
            post(crate::image_source::fetch),
        )
        .route(
            "/flush_cache",
            post(crate::server::routes::cache::flush_cache),
        )
        .route(
            "/symmetric_pd/place",
            tracked(post(crate::server::routes::symmetric_pd::place)),
        )
        .route(
            "/symmetric_pd/prepared",
            post(crate::server::routes::prepared::plan),
        )
        .route(
            "/symmetric_pd/events",
            post(crate::server::routes::prepared::event),
        )
        .route(
            "/symmetric_pd/cancel",
            post(crate::server::routes::prepared::cancel),
        )
        .route(
            "/symmetric_pd/executions",
            get(crate::server::routes::prepared::summary),
        )
        .route(
            "/symmetric_pd/backend/loads",
            get(crate::server::routes::prepared::backend_loads),
        )
        .route(
            "/symmetric_pd/roles",
            get(crate::server::routes::prepared::roles)
                .put(crate::server::routes::prepared::replace_roles),
        )
        .route(
            "/symmetric_pd/migration",
            post(crate::server::routes::prepared::migration),
        )
        // After routing, so MatchedPath is set for every route.
        .layer(middleware::from_fn_with_state(ctx.clone(), count_requests))
        .with_state(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::SymmetricPdConfig, discovery::ModelId, scheduler::SymmetricPdRuntime};
    use axum::{body::Body, http::Request as HttpRequest};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[tokio::test]
    async fn dynamic_retention_requires_tracked_execution_routes() {
        for (external_role_control, kv_session_retention) in
            [(false, false), (false, true), (true, false), (true, true)]
        {
            let mut ctx = AppContext::stub();
            let registry = Arc::clone(&ctx.registry);
            // No serving or background tasks: exercise the real ingress router.
            ctx.symmetric_pd = Some(
                std::thread::spawn(move || {
                    SymmetricPdRuntime::spawn(
                        SymmetricPdConfig {
                            external_role_control,
                            kv_session_retention,
                            ..Default::default()
                        },
                        None,
                        registry,
                        ModelId("stub-model".into()),
                    )
                })
                .join()
                .unwrap(),
            );
            let app = build_router(Arc::new(ctx));
            for route in [
                "/v1/chat/completions",
                "/v1/completions",
                "/symmetric_pd/place",
                "/symmetric_pd/prepared",
            ] {
                let response = app
                    .clone()
                    .oneshot(
                        HttpRequest::builder()
                            .method("POST")
                            .uri(route)
                            .header("content-type", "application/json")
                            .body(Body::from("invalid json"))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                let body = response.into_body().collect().await.unwrap().to_bytes();
                let body = std::str::from_utf8(&body).unwrap();
                assert_eq!(
                    body.contains("cannot serve /symmetric_pd/place"),
                    external_role_control && kv_session_retention && route == "/symmetric_pd/place",
                    "{route}: {body}",
                );
            }
        }
    }
}
