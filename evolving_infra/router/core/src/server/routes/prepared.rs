//! Prepared input planning and Backend lifecycle callbacks. Full token arrays
//! stay at the tokenizer; these endpoints handle only descriptors and events.
use crate::{
    backend::{execution::ExecutionEvent, sglang::request::cancel_engines},
    server::{app_context::AppContext, error::ApiError},
};
use axum::{extract::State, http::HeaderMap, Json};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedRequest {
    prepared: Value,
}
/// Extract a callback credential without logging or copying its contents.
///
/// # Inputs
/// `headers`: request headers containing a nonempty `Authorization: Bearer ...`.
///
/// # Outputs
/// Borrowed credential string, or `BadRequest` for absent/malformed credentials.
/// Token validity is checked separately by the execution registry.
fn bearer(headers: &HeaderMap) -> Result<&str, ApiError> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .filter(|v| !v.is_empty())
        .ok_or_else(|| ApiError::BadRequest("missing callback credentials".into()))
}
/// Reserve a request identity and return a prepared P/D execution plan.
///
/// # Inputs
/// `ctx`: shared configuration, scheduler and execution registry.
/// `req`: compact tokenizer descriptor; full input tokens remain at the tokenizer.
///
/// # Outputs
/// JSON plan with targets, KV marks and callback credentials, or a configuration,
/// validation/admission error. Registers the plan and its load reservation but
/// does not submit engine requests; the tokenizer must claim and dispatch it.
pub async fn plan(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    Json(req): Json<PreparedRequest>,
) -> Result<Json<Value>, ApiError> {
    let cfg = ctx
        .prepared_config
        .as_ref()
        .ok_or_else(|| ApiError::BadRequest("prepared inputs not configured".into()))?;
    let flex = ctx.symmetric_pd.as_ref().ok_or(ApiError::SymmetricPdDisabled)?;
    let mut input = cfg
        .input(&req.prepared, &ctx.config.model.id)
        .map_err(ApiError::BadRequest)?;
    // `--symmetric-pd-trace-session-moves`: the session identity of the
    // session-move trace, as the completions route passes it.
    if flex.cost_function_traces_session_moves() {
        input.trade_session = crate::agent_identity::trade_session_key(&headers);
    }
    let rid = input.rid.as_ref().unwrap().clone();
    let identity = crate::agent_identity::from_headers(&headers)?;
    if identity
        .as_ref()
        .is_some_and(|id| id.engine_rid().ok().as_deref() != Some(rid.as_str()))
    {
        return Err(ApiError::BadRequest(
            "prepared request_id conflicts with trajectory identity".into(),
        ));
    }
    // `--co-schedule-trade-wait e3`: the session key of the forecast target's
    // host rule (the agent identity, else `x-smg-routing-key`), as the
    // completions route passes it.
    if flex.cost_function_wants_e3_session() {
        input.e3_session = crate::trade_e3::session_key(&headers, identity.as_ref());
    }

    let reservation = ctx
        .executions
        .begin(&rid, cfg.max_inflight)
        .map_err(ApiError::BadRequest)?;
    // If the HTTP task is cancelled during placement, no engine has received
    // anything. Drop clears only this pending identity reservation.
    struct Pending {
        executions: Arc<crate::backend::execution::Executions>,
        rid: String,
        id: String,
    }
    impl Drop for Pending {
        /// Clean up a pending identity if planning exits before installation.
        ///
        /// # Inputs
        /// `self`: execution registry, request ID and matching reservation ID.
        ///
        /// # Outputs
        /// Unit. Abandons only this matching uninstalled reservation; an installed
        /// execution remains managed by its callbacks and cancellation lifecycle.
        fn drop(&mut self) {
            self.executions.abandon_unsubmitted(&self.rid, &self.id)
        }
    }
    let _pending = Pending {
        executions: Arc::clone(&ctx.executions),
        rid: rid.clone(),
        id: reservation.clone(),
    };
    let decision = flex.admit(input).await?;
    let ledger = decision.ledger.clone();
    let (p, d) = (decision.p_worker.url.clone(), decision.d_worker.url.clone());
    // Late D choose on symmetric workers (`--pd-late-decode choose`) under
    // external role control: the candidates are the decode-role
    // workers now, `d` (the arrival pick) among them; a bind later chooses
    // among those still in the decode role. Without external role control
    // prepared plans stay single-D as before.
    let late = match ctx
        .late_bind
        .as_ref()
        .filter(|l| l.symmetric() && flex.config().external_role_control)
    {
        Some(late) => super::late_d::flex_pick(late, flex, &decision).inspect_err(|_| {
            ledger.retire();
        })?,
        None => None,
    };
    let charge_body =
        json!({"sampling_params": {"max_new_tokens": req.prepared["max_new_tokens"]}});
    let prompt_tokens = req.prepared["prompt_tokens"].as_u64().unwrap_or(0);
    let installed = flex.with_current_role_pair(&p, &d, || {
        let Some((choose, marks)) = late else {
            let tag =
                if let (Some(sessions), Some(identity)) = (flex.kv_sessions(), identity.as_ref()) {
                    Some(sessions.register(identity, &p, &d)?)
                } else {
                    None
                };
            return ctx
                .executions
                .install(cfg, &req.prepared, &reservation, decision, None)
                .map(|result| (result, tag, false));
        };
        // Late D: the P registers alone; its tag rides on the P-leg only and
        // the prefill forwards it to the admitted decode in the ADMIT frame.
        // The entry enters the table before the
        // plan leaves, with the session hook recording every ADMIT target.
        let session = match (flex.kv_sessions(), identity.as_ref()) {
            (Some(sessions), Some(identity)) => Some(Arc::new(super::late_d::FlexLateSession {
                sessions: Arc::clone(sessions),
                tag: sessions.register_late_prefill(identity, &p)?,
            })),
            _ => None,
        };
        let tag = session.as_ref().map(|s| s.tag.clone());
        let charge_tokens = choose
            .late
            .request_charge(prompt_tokens, Some(&charge_body));
        let guard = choose.late.register_with(
            crate::late_bind::EntrySpec {
                owner: choose.owner,
                prefill_url: p.clone(),
                key: marks.transfer_id,
                rid: Some(rid.clone()),
                candidates: choose.candidates,
                default_idx: choose.default_idx,
                charge_tokens,
            },
            session.map(|s| s as Arc<dyn crate::late_bind::LateSessionHook>),
        );
        let late = crate::backend::execution::PreparedLate {
            late: choose.late,
            guard,
            kv_export: marks.kv_export,
            imports: marks.imports,
            reserve_decode: {
                let flex = Arc::clone(flex);
                Arc::new(move |w: &str| flex.reserve_decode(w))
            },
        };
        ctx.executions
            .install(cfg, &req.prepared, &reservation, decision, Some(late))
            .map(|result| (result, tag, true))
    });
    let (mut result, tag, late) = installed.map_err(|error| {
        ledger.retire();
        ApiError::BadRequest(error)
    })?;
    if let Some(tag) = tag {
        let tag = serde_json::to_value(tag).map_err(|e| ApiError::Internal(e.into()))?;
        result["prefill"]["kv_session"] = tag.clone();
        if !late {
            result["decode"]["kv_session"] = tag;
        }
    }
    Ok(Json(result))
}
/// Apply an authenticated tokenizer event to its registered execution.
///
/// # Inputs
/// `ctx`: shared execution registry and engine HTTP client.
/// `headers`: callback bearer credential.
/// `event`: reservation/leg identity and lifecycle event from the tokenizer.
///
/// # Outputs
/// JSON `accepted: true` after the event is recorded, or `BadRequest` on invalid
/// identity, credential or transition. May spawn best-effort engine cancellation;
/// acceptance alone does not confirm that either engine has terminated.
pub async fn event(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    Json(event): Json<ExecutionEvent>,
) -> Result<Json<Value>, ApiError> {
    let token = bearer(&headers)?;
    let targets = ctx
        .executions
        .event(&event, token)
        .map_err(ApiError::BadRequest)?;
    if !targets.is_empty() {
        tokio::spawn(cancel_engines(ctx.proxy.client.clone(), targets));
    }
    Ok(Json(json!({"accepted":true})))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CancelRequest {
    reservation_id: String,
}
/// Request cancellation of a prepared execution and dispatch engine aborts.
///
/// # Inputs
/// `ctx`: execution registry and engine HTTP client.
/// `headers`: bearer credential belonging to this execution.
/// `req`: reservation ID to cancel.
///
/// # Outputs
/// JSON acknowledging the cancellation request, or `BadRequest` for an invalid
/// reservation/credential. Aborts run asynchronously; the response explicitly
/// does not confirm engine termination or KV resource reclamation.
pub async fn cancel(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    Json(req): Json<CancelRequest>,
) -> Result<Json<Value>, ApiError> {
    let targets = ctx
        .executions
        .cancel(&req.reservation_id, bearer(&headers)?)
        .map_err(ApiError::BadRequest)?;
    tokio::spawn(cancel_engines(ctx.proxy.client.clone(), targets));
    Ok(Json(
        json!({"cancellation_requested":true,"engine_termination_confirmed":false}),
    ))
}
/// Read counts and states of registered prepared executions.
///
/// # Inputs
/// `ctx`: application context containing the shared execution registry.
///
/// # Outputs
/// JSON summary of local lifecycle records after expiring unclaimed plans older
/// than 300 seconds. Their reservations are retired; claimed work is retained.
/// Does not query engines.
pub async fn summary(State(ctx): State<Arc<AppContext>>) -> Json<Value> {
    Json(ctx.executions.summary())
}

/// Read merged Backend observations without performing a hypothetical placement.
/// In particular, observing load must not consume the D round-robin transfer seed.
///
/// # Inputs
/// `ctx`: context whose symmetric P/D runtime owns the merged Backend state.
///
/// # Outputs
/// JSON `workers` map of load views, or `SymmetricPdDisabled`. Includes freshness and
/// unknown-state information; does not trigger a new engine poll or placement.
pub async fn backend_loads(State(ctx): State<Arc<AppContext>>) -> Result<Json<Value>, ApiError> {
    let flex = ctx.symmetric_pd.as_ref().ok_or(ApiError::SymmetricPdDisabled)?;
    Ok(Json(json!({"workers": flex.backend_load_views()})))
}

/// External controller operations are opt-in and leave native routing untouched.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplaceRoles {
    epoch: String,
    revision: u64,
    #[serde(deserialize_with = "unique_roles")]
    expected: crate::backend::registry::RoleAssignments,
    #[serde(deserialize_with = "unique_roles")]
    desired: crate::backend::registry::RoleAssignments,
}
fn unique_roles<'de, D: serde::Deserializer<'de>>(d: D)
    -> Result<crate::backend::registry::RoleAssignments, D::Error> {
    struct RolesVisitor;
    impl<'de> serde::de::Visitor<'de> for RolesVisitor {
        type Value = crate::backend::registry::RoleAssignments;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("unique worker URL to role map")
        }
        fn visit_map<M: serde::de::MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
            let mut roles = Self::Value::new();
            while let Some((url, role)) = map.next_entry::<String, crate::backend::registry::RoleAssignment>()? {
                if roles.insert(url, role).is_some() {
                    return Err(serde::de::Error::custom("duplicate worker URL"));
                }
            }
            Ok(roles)
        }
    }
    d.deserialize_map(RolesVisitor)
}
fn external_credential(ctx: &AppContext, headers: &HeaderMap) -> Result<(), ApiError> {
    let key = ctx.external_role_key.as_deref()
        .ok_or_else(|| ApiError::BadRequest("external role control is disabled".into()))?;
    if bearer(headers)? != key {
        return Err(ApiError::BadRequest("invalid controller credentials".into()));
    }
    Ok(())
}
pub async fn roles(State(ctx): State<Arc<AppContext>>, headers: HeaderMap) -> Result<Json<Value>, ApiError> {
    external_credential(&ctx, &headers)?;
    let flex = ctx.symmetric_pd.as_ref().ok_or(ApiError::SymmetricPdDisabled)?;
    if !flex.config().external_role_control {
        return Err(ApiError::BadRequest("external role control is disabled".into()));
    }
    Ok(Json(json!({"assignment": flex.role_assignments().map_err(ApiError::BadRequest)?,
        "completed_workload": ctx.external_workload.as_ref().map(|w| w.totals()),
        "decode_population": ctx.executions.decode_population(),
        "draining": flex.role_drains()})))
}
pub async fn replace_roles(
    State(ctx): State<Arc<AppContext>>, headers: HeaderMap, Json(req): Json<ReplaceRoles>,
) -> Result<Json<Value>, ApiError> {
    let flex = ctx.symmetric_pd.as_ref().ok_or(ApiError::SymmetricPdDisabled)?;
    external_credential(&ctx, &headers)?;
    let (roles, draining) = flex
        .replace_role_assignments(&req.epoch, req.revision, &req.expected, &req.desired)
        .map_err(ApiError::BadRequest)?;
    if let Some(timeout) = flex.config().role_drain_timeout {
        crate::scheduler::role_drain::spawn(
            flex,
            &ctx.executions,
            &ctx.proxy.client,
            draining.clone(),
            roles.revision,
            timeout,
        );
    }
    Ok(Json(json!({"assignment": roles, "draining": draining})))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MigrationQuery {
    reservation_id: String,
}
/// Tell a tokenizer whether a D-leg it saw aborted was moved off a draining worker.
///
/// # Inputs
/// `headers` carry the execution's callback credential; `req` names its reservation.
///
/// # Outputs
/// `{"migrating": true}` when a decode->prefill drain aborted the D-leg, so the
/// tokenizer resumes the request on a newly placed pair; otherwise false.
/// `BadRequest` for an unknown reservation or a wrong credential.
pub async fn migration(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    Json(req): Json<MigrationQuery>,
) -> Result<Json<Value>, ApiError> {
    let migrating = ctx
        .executions
        .migration_status(&req.reservation_id, bearer(&headers)?)
        .map_err(ApiError::BadRequest)?;
    Ok(Json(json!({"migrating": migrating})))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::SymmetricPdConfig,
        discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec},
        scheduler::{DSelect, SymmetricPdRuntime, PSelect, PlaceHttpRequest, RoleWeight},
        workers::SymmetricPdWorkerInfo,
    };
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[test]
    fn role_update_rejects_duplicate_urls_and_requires_controller_key() {
        let request = r#"{"epoch":"e","revision":0,"expected":{"u":"prefill","u":"decode"},"desired":{}}"#;
        assert!(serde_json::from_str::<ReplaceRoles>(request).is_err());
        let mut ctx = AppContext::stub();
        let mut headers = HeaderMap::new();
        assert!(external_credential(&ctx, &headers).is_err());
        ctx.external_role_key = Some("key".into());
        headers.insert("authorization", "Bearer incorrect".parse().unwrap());
        assert!(external_credential(&ctx, &headers).is_err());
        headers.insert("authorization", "Bearer key".parse().unwrap());
        assert!(external_credential(&ctx, &headers).is_ok());
    }

    /// Verify that load inspection reads the shared ledger without advancing RR.
    ///
    /// # Inputs
    /// None. Builds one logical P, three logical D workers and an unknown reservation.
    ///
    /// # Outputs
    /// Unit; assertions check load fields, reservation retention and the next D
    /// selection. No engine requests or background load polls are needed.
    #[tokio::test]
    async fn backend_load_route_reads_shared_state_without_consuming_round_robin() {
        let mut ctx = AppContext::stub();
        for (id, weight) in [
            ("p", RoleWeight::PREFILL_LEANING),
            ("d0", RoleWeight::DECODE_LEANING),
            ("d1", RoleWeight::DECODE_LEANING),
            ("d2", RoleWeight::DECODE_LEANING),
        ] {
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
        }
        let registry = Arc::clone(&ctx.registry);
        // Build outside Tokio so automatic background polling cannot affect this
        // route-only test. No engine is needed to observe a local reservation.
        let flex = std::thread::spawn(move || {
            let mut cfg = SymmetricPdConfig::default();
            cfg.placement.force_split = true;
            cfg.placement.p_select = PSelect::CoSchedule;
            cfg.placement.d_select = DSelect::RoundRobin;
            SymmetricPdRuntime::spawn(cfg, None, registry, ModelId("stub-model".into()))
        })
        .join()
        .unwrap();
        ctx.symmetric_pd = Some(Arc::clone(&flex));
        let reservation = flex.account("http://p:30000", 8192);
        reservation.mark_unknown();
        let app = crate::server::app::build_router(Arc::new(ctx));
        let probe = PlaceHttpRequest {
            account: Some(false),
            prompt_len: Some(8),
            ..Default::default()
        };
        let before = flex.place_http(probe.clone()).await.unwrap();
        for _ in 0..4 {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/symmetric_pd/backend/loads")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let value: Value =
                serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                    .unwrap();
            assert_eq!(value["workers"].as_object().unwrap().len(), 4);
            let p = &value["workers"]["http://p:30000"];
            assert_eq!(p["reserved_uncached_tokens"], 8192);
            assert_eq!(p["active_prefill_reqs"], 1);
            assert_eq!(p["unknown_prefill_reqs"], 1);
            assert_eq!(p["workload_mode"], "rid_remaining");
            assert_eq!(p["status"], "missing");
            assert!(p["observation"].is_null());
            assert!(p["sample_age_ms"].is_null());
        }
        let after = flex.place_http(probe).await.unwrap();
        assert_eq!(
            after.transfer_id.unwrap(),
            (before.transfer_id.unwrap() + 1) & i64::MAX as u64
        );
        let next_d = format!("http://d{}:30000", after.transfer_id.unwrap() % 3);
        assert_eq!(after.d_leg.worker, next_d);
        assert_ne!(before.d_leg.worker, after.d_leg.worker);
        // Queries must not retire work; only its authoritative handle can do so.
        assert_eq!(
            flex.backend_load_views()["http://p:30000"].active_prefill_reqs,
            1
        );
        reservation.retire();
        assert_eq!(
            flex.backend_load_views()["http://p:30000"].active_prefill_reqs,
            0
        );
    }

    /// Verify that load inspection rejects a context without symmetric P/D enabled.
    ///
    /// # Inputs
    /// None. Builds a stub application and sends an in-process load request.
    ///
    /// # Outputs
    /// Unit; asserts HTTP 404 for the unavailable Backend endpoint.
    #[tokio::test]
    async fn backend_load_route_requires_symmetric_pd() {
        let app = crate::server::app::build_router(Arc::new(AppContext::stub()));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/symmetric_pd/backend/loads")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn migration_status_requires_the_execution_credential() {
        let ctx = AppContext::stub();
        let mut state =
            crate::backend::state::BackendState::new(Arc::new(tokio::sync::Notify::new()));
        ctx.executions.install_claimed_for_test(
            "r",
            "secret",
            ["http://p", "http://d"],
            state.reserve("http://p", 1),
        );
        let app = crate::server::app::build_router(Arc::new(ctx));
        let ask = |token: Option<&str>| {
            let mut request = Request::builder()
                .method("POST")
                .uri("/symmetric_pd/migration")
                .header("content-type", "application/json");
            if let Some(token) = token {
                request = request.header("authorization", format!("Bearer {token}"));
            }
            request
                .body(Body::from(r#"{"reservation_id":"r"}"#))
                .unwrap()
        };
        for token in [None, Some("wrong")] {
            let response = app.clone().oneshot(ask(token)).await.unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
        let response = app.oneshot(ask(Some("secret"))).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap(),
            json!({"migrating": false})
        );
    }
}
