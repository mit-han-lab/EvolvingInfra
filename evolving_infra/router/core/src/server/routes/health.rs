// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::app_context::AppContext;
use axum::extract::State;
use axum::http::StatusCode;
use std::sync::Arc;

/// Always returns 200 — liveness probe.
pub async fn healthz() -> StatusCode {
    StatusCode::OK
}

/// Readiness probe — 200 only when the pod can actually serve traffic.
///
/// Requires BOTH:
/// 1. `AppContext::mark_ready()` was called by main (process bootstrap
///    finished — config loaded, tokenizers built, server bound), AND
/// 2. At least one worker is registered. Without this second check,
///    `/readyz` flips green before the first `DiscoveryEvent::Added`
///    has been processed — the Service starts sending traffic to a
///    pod whose registry is empty, and every request returns 503
///    `no_healthy_workers`.
///
/// Under `--policy symmetric_pd` the second check becomes "at least one
/// prefill-capable and one decode-capable worker are up" (weight > 0,
/// breaker closed): a fleet that can only run one leg cannot place a
/// Split, and a decode-only fleet cannot place anything.
pub async fn readyz(State(ctx): State<Arc<AppContext>>) -> StatusCode {
    let workers_ok = match ctx.symmetric_pd.as_ref() {
        Some(flex) => flex.has_capable_pair(),
        None => !ctx.registry.is_empty(),
    };
    if ctx.is_ready() && workers_ok {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn healthz_always_200() {
        let app = crate::server::app::build_router(test_ctx(false, false));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn readyz_503_when_not_ready() {
        let app = crate::server::app::build_router(test_ctx(false, true));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn readyz_503_when_ready_but_registry_empty() {
        // Regression: `/readyz` previously returned 200 the moment
        // `mark_ready()` was called, even with an empty worker
        // registry. The Service would route traffic to a pod that
        // could only return 503 no_healthy_workers.
        let app = crate::server::app::build_router(test_ctx(true, false));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            res.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "ready=true + empty registry must still be 503"
        );
    }

    #[tokio::test]
    async fn readyz_200_when_ready_and_worker_registered() {
        let app = crate::server::app::build_router(test_ctx(true, true));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    /// symmetric_pd readiness needs a prefill-capable AND a decode-capable
    /// worker; one prefill-leaning worker alone stays 503.
    #[tokio::test]
    async fn readyz_symmetric_pd_requires_a_capable_pair() {
        use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
        use crate::scheduler::RoleWeight;
        use crate::workers::SymmetricPdWorkerInfo;
        let mut ctx = AppContext::stub();
        ctx.config.model.policy = crate::config::PolicyKind::SymmetricPd;
        ctx.config.model.symmetric_pd = Some(crate::config::SymmetricPdConfig::default());
        ctx.symmetric_pd = Some(crate::scheduler::SymmetricPdRuntime::spawn(
            crate::config::SymmetricPdConfig::default(),
            None,
            Arc::clone(&ctx.registry),
            ModelId("stub-model".into()),
        ));
        ctx.mark_ready();
        let add = |ctx: &AppContext, id: &str, weight: RoleWeight| {
            ctx.registry
                .add(WorkerSpec {
                    id: WorkerId(id.into()),
                    url: format!("http://{id}:30000"),
                    mode: WorkerMode::Symmetric,
                    model_ids: vec![ModelId("stub-model".into())],
                    bootstrap_port: Some(8998),
                })
                .unwrap();
            ctx.registry
                .get(&WorkerId(id.into()))
                .unwrap()
                .set_symmetric_pd(SymmetricPdWorkerInfo {
                    weight,
                    ..Default::default()
                });
        };
        add(&ctx, "p", RoleWeight::PREFILL_LEANING);
        let ctx = Arc::new(ctx);
        let probe = |ctx: Arc<AppContext>| async move {
            crate::server::app::build_router(ctx)
                .oneshot(
                    Request::builder()
                        .uri("/readyz")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap()
                .status()
        };
        assert_eq!(
            probe(Arc::clone(&ctx)).await,
            StatusCode::SERVICE_UNAVAILABLE,
            "prefill-only fleet is not ready under symmetric_pd"
        );
        add(&ctx, "d", RoleWeight::DECODE_LEANING);
        assert_eq!(probe(ctx).await, StatusCode::OK);
    }

    fn test_ctx(ready: bool, with_worker: bool) -> Arc<AppContext> {
        use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
        let ctx = AppContext::stub();
        if ready {
            ctx.mark_ready();
        }
        if with_worker {
            ctx.registry
                .add(WorkerSpec {
                    id: WorkerId("test-w".into()),
                    url: "http://test:30000".into(),
                    mode: WorkerMode::Plain,
                    model_ids: vec![ModelId("test".into())],
                    bootstrap_port: None,
                })
                .expect("test worker accepted");
        }
        Arc::new(ctx)
    }
}
