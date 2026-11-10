// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `/metrics` endpoint — Prometheus 0.0.4 exposition.
//!
//! Returns the live snapshot of [`crate::server::metrics::MetricsRegistry`].
//! Plain-text body; charset is utf-8. We deliberately don't gate this on
//! readiness — scrapers should be able to read the metrics surface even
//! while the router is warming up so the "router started but no workers
//! discovered" failure mode is observable.

use crate::discovery::WorkerMode;
use crate::server::app_context::AppContext;
use crate::server::metrics::{render_decode_token_charges, DecodeChargeSnapshot, WorkerSnapshot};
use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use std::sync::Arc;

/// Content-Type per Prometheus exposition format spec.
const PROMETHEUS_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

pub async fn metrics(State(ctx): State<Arc<AppContext>>) -> impl IntoResponse {
    // Sample the live registry into a snapshot for the worker gauges. These
    // are pull-on-scrape (not pushed) so removed workers stop emitting series
    // immediately; see `MetricsRegistry::render_with_workers`.
    let workers: Vec<WorkerSnapshot> = ctx
        .registry
        .all()
        .into_iter()
        .map(|w| {
            // One lock acquisition for both health + state so the two gauges
            // can't report a torn (self-contradictory) pair for one scrape.
            let cb = w.breaker.snapshot();
            WorkerSnapshot {
                worker_url: w.url.clone(),
                mode: match w.mode() {
                    WorkerMode::Plain => "plain",
                    WorkerMode::Prefill => "prefill",
                    WorkerMode::Decode => "decode",
                    WorkerMode::Symmetric => "symmetric",
                },
                healthy: cb.admit,
                cb_state: cb.state_code,
                // Saturating rather than `as i64`: a guard-accounting
                // underflow would wrap usize and render as a nonsensical
                // negative gauge; clamp to a large positive ceiling instead.
                inflight: i64::try_from(w.active_load()).unwrap_or(i64::MAX),
            }
        })
        .collect();
    let mut body = ctx.metrics.render_with_workers(&workers);
    // `--image-layout-config`: screenshot-aware token ids, by outcome.
    if ctx.config.proxy.image_layout.is_some() {
        body.push_str(&crate::image_layout::render_metrics());
    }
    // `--late-d-image-source-url`: pixel-free late-D decode legs.
    if ctx.config.proxy.late_d_image_source_url.is_some() {
        body.push_str(&crate::image_source::render_metrics());
    }
    // Token-charged decode policies (and late D choose, which charges every
    // bound request): per-decode charged tokens / requests.
    if ctx.decode_tokens.is_some() || ctx.late_bind.is_some() {
        let rows: Vec<DecodeChargeSnapshot> = ctx
            .registry
            .all()
            .into_iter()
            .filter(|w| w.mode() == WorkerMode::Decode)
            .map(|w| DecodeChargeSnapshot {
                worker_url: w.url.clone(),
                charged_tokens: w.decode_charged_tokens(),
                charged_reqs: w.decode_charged_reqs() as u64,
            })
            .collect();
        body.push_str(&render_decode_token_charges(&rows));
    }
    // `power-of-two-smg`: picks by load signal, poll health, snapshot.
    if let Some(policy) = &ctx.decode_po2_smg {
        body.push_str(&policy.render_metrics());
    }
    // Late D: the choose-mode table, links and choices; packed prompts and
    // registration refusals (only when a late-D flag is set).
    if let Some(late) = &ctx.late_bind {
        body.push_str(&late.render_metrics());
    }
    if ctx.late_bind.is_some() || ctx.config.proxy.late_decode.packed_prompt {
        body.push_str(&crate::late_bind::render_static_metrics());
    }
    (
        StatusCode::OK,
        [(CONTENT_TYPE, PROMETHEUS_CONTENT_TYPE)],
        body,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::metrics::{RequestOutcome, WorkerModeLabel};
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[tokio::test]
    async fn metrics_endpoint_returns_prometheus_text() {
        let ctx = Arc::new(AppContext::stub());
        let app = crate::server::app::build_router(ctx.clone());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let content_type = res
            .headers()
            .get(CONTENT_TYPE)
            .expect("content-type header")
            .to_str()
            .unwrap()
            .to_owned();
        assert!(
            content_type.starts_with("text/plain"),
            "expected text/plain, got {content_type}",
        );
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&body).unwrap();
        // Every metric family should at least carry its HELP/TYPE lines.
        assert!(body.contains("# TYPE sgl_router_requests_total counter"));
        assert!(body.contains("# TYPE sgl_router_overlap_blocks histogram"));
        assert!(body.contains("# TYPE sgl_router_active_load gauge"));
    }

    #[tokio::test]
    async fn metrics_endpoint_reflects_recorded_counters() {
        let ctx = Arc::new(AppContext::stub());
        ctx.metrics.record_worker_request(
            "http://w-test:30000",
            "tiny",
            WorkerModeLabel::Prefill,
            RequestOutcome::Success,
        );
        let app = crate::server::app::build_router(ctx.clone());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&body).unwrap();
        assert!(
            body.contains(r#"worker_url="http://w-test:30000""#),
            "metrics did not include the recorded worker_url; got:\n{body}",
        );
    }

    #[tokio::test]
    async fn metrics_endpoint_samples_worker_gauges_from_registry() {
        use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};

        let ctx = Arc::new(AppContext::stub());
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId("p0".into()),
                url: "http://p0:30000".into(),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("m".into())],
                bootstrap_port: None,
            })
            .unwrap();
        let app = crate::server::app::build_router(ctx.clone());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&body).unwrap();
        // Pool size reflects the registered prefill worker, and the per-worker
        // gauges are sampled (fresh breaker => healthy, closed, 0 inflight).
        assert!(
            body.contains(r#"sgl_router_workers{mode="prefill"} 1"#),
            "got:\n{body}"
        );
        assert!(body.contains(r#"sgl_router_worker_health{worker_url="http://p0:30000"} 1"#));
        assert!(body.contains(r#"sgl_router_worker_cb_state{worker_url="http://p0:30000"} 0"#));
        assert!(
            body.contains(r#"sgl_router_worker_inflight_requests{worker_url="http://p0:30000"} 0"#)
        );
    }
}
