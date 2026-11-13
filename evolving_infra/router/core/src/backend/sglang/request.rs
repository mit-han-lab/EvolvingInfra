// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! SGLang request bodies. No placement or load accounting lives here.
//!
//! Builds prepared-input targets and per-leg JSON payloads, estimates prompt
//! length when tokens are unavailable, and sends best-effort abort requests.
//! Successful abort delivery alone never proves that engine work has stopped.
use super::kv::{Marks, BOOTSTRAP_KEYS, KV_EXPORT_KEY, KV_IMPORT_KEY};
use crate::server::error::ApiError;
use bytes::Bytes;
use serde_json::{json, Value};
use std::time::Duration;

/// Input: ingress identity, selected engines, and already-parsed request JSON.
/// Output: optional session metadata inserted without another prompt serialization or network IO.
pub fn attach_kv_session(
    sessions: Option<&crate::backend::kv_sessions::KvSessions>,
    headers: &axum::http::HeaderMap,
    p: &str,
    d: &str,
    value: &mut Option<Value>,
) -> Result<(), ApiError> {
    let Some(sessions) = sessions else {
        return Ok(());
    };
    if value
        .as_ref()
        .is_some_and(|v| v.get("kv_session").is_some())
    {
        return Err(ApiError::BadRequest(
            "kv_session is assigned by the router, not the client".into(),
        ));
    }
    let Some(identity) = session_identity(headers)? else {
        return Ok(());
    };
    let object = value
        .as_mut()
        .and_then(Value::as_object_mut)
        .ok_or_else(|| ApiError::BadRequest("KV retention requires an object request".into()))?;
    let tag = sessions
        .register(&identity, p, d)
        .map_err(ApiError::BadRequest)?;
    object.insert(
        "kv_session".into(),
        serde_json::to_value(tag).map_err(|e| ApiError::Internal(e.into()))?,
    );
    Ok(())
}

/// The session identity of a symmetric_pd request: `x-agent-context`, else
/// its alias `x-kv-session-context` (the C1 replay client's header; the
/// router never touches the body `rid` for it). Malformed: 400.
pub fn session_identity(
    headers: &axum::http::HeaderMap,
) -> Result<Option<crate::agent_identity::RequestIdentity>, ApiError> {
    match crate::agent_identity::from_headers(headers)? {
        Some(identity) => Ok(Some(identity)),
        None => {
            crate::agent_identity::from_header(headers, crate::agent_identity::KV_SESSION_HEADER)
        }
    }
}

/// Late D on symmetric_pd: register the P-leg only
/// ([`crate::backend::kv_sessions::KvSessions::register_late_prefill`]) and
/// return its tag. The caller puts it on the P-leg body only; no decode
/// candidate body carries it (the P forwards it in the ADMIT frame).
/// `Ok(None)`: retention off or no identity. A client `kv_session`, a
/// malformed identity or a registration error gives 400, as
/// [`attach_kv_session`].
pub fn late_kv_session(
    sessions: Option<&crate::backend::kv_sessions::KvSessions>,
    headers: &axum::http::HeaderMap,
    p: &str,
    value: Option<&Value>,
) -> Result<Option<crate::backend::kv_sessions::KvSession>, ApiError> {
    let Some(sessions) = sessions else {
        return Ok(None);
    };
    if value.is_some_and(|v| v.get("kv_session").is_some()) {
        return Err(ApiError::BadRequest(
            "kv_session is assigned by the router, not the client".into(),
        ));
    }
    let Some(identity) = session_identity(headers)? else {
        return Ok(None);
    };
    if !value.is_some_and(Value::is_object) {
        return Err(ApiError::BadRequest(
            "KV retention requires an object request".into(),
        ));
    }
    let tag = sessions
        .register_late_prefill(&identity, p)
        .map_err(ApiError::BadRequest)?;
    crate::backend::kv_sessions::inc(&sessions.stats().tagged);
    Ok(Some(tag))
}

/// Prepared-input transport understood by T's SGLang adapter.
/// Describe how the input service should submit a prepared SGLang leg.
///
/// # Inputs
/// `worker_url` is the engine base URL; `config_id` selects its prepared
/// input/tokenizer configuration.
///
/// # Outputs
/// Returns JSON containing the SGLang transport, normalized `/generate`
/// endpoint and configuration ID. Does not contact the engine.
pub fn prepared_target(worker_url: &str, config_id: &str) -> Value {
    json!({
        "transport": "sglang_generate",
        "endpoint": format!("{}/generate", worker_url.trim_end_matches('/')),
        "config_id": config_id,
    })
}

/// Best-effort engine abort. An HTTP ACK never confirms termination or KV release;
/// the Backend execution record remains responsible for that distinction.
/// Send concurrent, best-effort aborts for a set of engine requests.
///
/// # Inputs
/// `client` is a reusable HTTP client; `targets` contains `(worker URL,
/// request ID)` pairs whose requests may still own compute or KV resources.
///
/// # Outputs
/// Returns `()` after all abort attempts finish, each bounded by three
/// seconds. Logs unacknowledged requests without retrying. Does not retire
/// reservations or treat HTTP acknowledgement as terminal engine evidence.
pub async fn cancel_engines(client: reqwest::Client, targets: Vec<(String, String)>) {
    futures::future::join_all(targets.into_iter().map(|(url, rid)| {
        let client = client.clone();
        async move {
            let result = client
                .post(format!("{}/abort_request", url.trim_end_matches('/')))
                .timeout(Duration::from_secs(3))
                .json(&json!({"rid": rid}))
                .send()
                .await;
            if !result.as_ref().is_ok_and(|r| r.status().is_success()) {
                tracing::warn!(worker=%url,"abort request was not acknowledged; execution remains unresolved");
            }
        }
    }))
    .await;
}

/// Insert the P-leg / D-leg marks into copies of the request body.
/// Returns `(p_body, d_body)`; for Local both are the untouched body. A
/// request never carries both marks nor the retired
/// bootstrap fields.
/// Create separate P and D request objects with their respective KV marks.
///
/// # Inputs
/// `body` is the original request JSON; `marks` contains matching export
/// and import metadata for split execution.
///
/// # Outputs
/// Returns `(p_body, d_body)`, removing opposite-leg and legacy bootstrap
/// fields for a split request. Returns cloned input when either mark is
/// absent, or an error if a split request body is not a JSON object.
pub fn inject_marks(body: &Value, marks: &Marks) -> Result<(Value, Value), String> {
    let (Some(export), Some(import)) = (&marks.kv_export, &marks.kv_import) else {
        return Ok((body.clone(), body.clone()));
    };
    let mut p_body = body.clone();
    let mut d_body = body.clone();
    let p_obj = p_body
        .as_object_mut()
        .ok_or_else(|| "Request must be a JSON object".to_string())?;
    for k in BOOTSTRAP_KEYS.iter().chain([KV_IMPORT_KEY].iter()) {
        p_obj.remove(*k);
    }
    p_obj.insert(KV_EXPORT_KEY.to_string(), export.clone());
    let d_obj = d_body
        .as_object_mut()
        .ok_or_else(|| "Request must be a JSON object".to_string())?;
    for k in BOOTSTRAP_KEYS.iter().chain([KV_EXPORT_KEY].iter()) {
        d_obj.remove(*k);
    }
    d_obj.insert(KV_IMPORT_KEY.to_string(), import.clone());
    Ok((p_body, d_body))
}

/// Serialized leg bodies for the routes: `(p_body, d_body)`. Parses `body`
/// once (or reuses `value`), injects router `input_ids` when given, the
/// peer-kv `kv_pull_hint` (P-leg only), then the marks. For Local (no
/// marks) both are the same bytes and, when nothing is injected, the
/// original buffer.
/// Serialize engine request bodies with prepared tokens and transfer metadata.
///
/// # Inputs
/// `body` is the original wire body; `value` optionally supplies its parsed
/// JSON. `input_ids` overrides token input, `marks` defines split execution,
/// and `pull_hint` supplies optional P-side peer-cache metadata.
///
/// # Outputs
/// Returns `(P bytes, D bytes)`, sharing the original buffer when no
/// changes are needed. Local execution reuses one body; split execution
/// adds the pull hint only to P. Invalid JSON/object bodies yield a bad-request
/// error; serialization failure yields an internal error. No upload occurs.
pub fn build_leg_bodies(
    body: &Bytes,
    value: Option<Value>,
    input_ids: Option<&[u32]>,
    marks: &Marks,
    pull_hint: Option<&Value>,
) -> Result<(Bytes, Bytes), ApiError> {
    let split = marks.kv_export.is_some() && marks.kv_import.is_some();
    if !split
        && input_ids.is_none()
        && pull_hint.is_none()
        && value.as_ref().is_none_or(|v| v.get("kv_session").is_none())
    {
        return Ok((body.clone(), body.clone()));
    }
    let mut parsed = match value {
        Some(v) => v,
        None => serde_json::from_slice(body).map_err(|_| {
            ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
        })?,
    };
    let obj = parsed.as_object_mut().ok_or_else(|| {
        ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
    })?;
    if let Some(ids) = input_ids {
        obj.insert(
            "input_ids".to_string(),
            Value::Array(ids.iter().map(|&i| Value::Number(i.into())).collect()),
        );
    }
    let to_bytes = |v: &Value| -> Result<Bytes, ApiError> {
        serde_json::to_vec(v)
            .map(Bytes::from)
            .map_err(|e| ApiError::Internal(anyhow::Error::new(e).context("serialize leg body")))
    };
    if !split {
        // Local: the single executing engine is the P-leg; the hint rides on it.
        if let Some(h) = pull_hint {
            obj.insert(KV_PULL_HINT_KEY.to_string(), h.clone());
        }
        let b = to_bytes(&parsed)?;
        return Ok((b.clone(), b));
    }
    let (mut p, d) = inject_marks(&parsed, marks).map_err(ApiError::BadRequest)?;
    if let Some(h) = pull_hint {
        if let Some(o) = p.as_object_mut() {
            o.insert(KV_PULL_HINT_KEY.to_string(), h.clone());
        }
    }
    Ok((to_bytes(&p)?, to_bytes(&d)?))
}

/// Engine contract for the peer-kv pull hint (same key the PD routes inject).
pub const KV_PULL_HINT_KEY: &str = "kv_pull_hint";

/// Best-effort prompt length of a raw generate/completions/chat body.
/// Estimate prompt tokens for relative scheduling and transfer costs.
///
/// # Inputs
/// `body` is a generate/completions/chat request; `tokens` is an optional
/// authoritative token slice prepared by the router.
///
/// # Outputs
/// Returns the supplied token count, otherwise an input-ID count or a
/// text estimate of ceil(UTF-8 bytes / 4). Text/numeric-prompt fallbacks are
/// at least one token; an explicit empty token/input-ID array may yield zero.
pub fn estimate_prompt_len(body: &Value, tokens: Option<&[u32]>) -> u64 {
    if let Some(t) = tokens {
        return t.len() as u64;
    }
    if let Some(ids) = body.get("input_ids").and_then(Value::as_array) {
        return ids.len() as u64;
    }
    if let Some(ids) = body.get("prompt").and_then(Value::as_array) {
        if ids.iter().all(Value::is_number) {
            return (ids.len() as u64).max(1);
        }
    }
    let text_len = body
        .get("text")
        .or_else(|| body.get("prompt"))
        .and_then(Value::as_str)
        .map(|s| s.len())
        .or_else(|| {
            body.get("messages").and_then(Value::as_array).map(|msgs| {
                msgs.iter()
                    .filter_map(|m| m.get("content").and_then(Value::as_str))
                    .map(str::len)
                    .sum()
            })
        })
        .unwrap_or(0);
    // ~4 chars / token; only feeds the relative backlog/transfer terms.
    (text_len as u64).div_ceil(4).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
    use std::sync::{Arc, Mutex};

    #[tokio::test]
    async fn session_tags_reach_split_and_local_without_changing_namespace() {
        let sessions = crate::backend::kv_sessions::KvSessions::spawn(None);
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            crate::agent_identity::CONTEXT_HEADER,
            r#"{"agent_id":"lifetime","subagent_id":0,"request_id":1}"#
                .parse()
                .unwrap(),
        );
        let bytes = Bytes::from_static(br#"{"prompt":[1,2],"cache_salt":"shared"}"#);
        let mut value = Some(serde_json::from_slice(&bytes).unwrap());
        attach_kv_session(Some(&sessions), &headers, "p", "d", &mut value).unwrap();
        let marks = Marks {
            kv_export: Some(json!({"transfer_id":1})),
            kv_import: Some(json!({"transfer_id":1})),
        };
        let (p, d) = build_leg_bodies(&bytes, value.clone(), None, &marks, None).unwrap();
        let p: Value = serde_json::from_slice(&p).unwrap();
        let d: Value = serde_json::from_slice(&d).unwrap();
        assert_eq!(p["kv_session"], d["kv_session"]);
        assert_eq!(p["cache_salt"], "shared");
        let local = Marks {
            kv_export: None,
            kv_import: None,
        };
        let (p, _) = build_leg_bodies(&bytes, value, None, &local, None).unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&p).unwrap()["kv_session"],
            d["kv_session"]
        );
        let mut value = Some(json!({"prompt":[1,2]}));
        attach_kv_session(None, &headers, "p", "d", &mut value).unwrap();
        assert!(value.unwrap().get("kv_session").is_none());
    }

    /// The C1 client's `x-kv-session-context` registers (P, D) and tags both
    /// legs like `x-agent-context`, without touching the body `rid`;
    /// with both headers `x-agent-context` wins; a malformed alias is 400.
    #[tokio::test]
    async fn kv_session_context_alias_tags_both_legs_and_keeps_the_rid() {
        let sessions = crate::backend::kv_sessions::KvSessions::spawn(None);
        let alias = r#"{"agent_id":"c1","subagent_id":0,"request_id":3}"#;
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            crate::agent_identity::KV_SESSION_HEADER,
            alias.parse().unwrap(),
        );
        let bytes = Bytes::from_static(br#"{"prompt":[1,2],"rid":"client-rid"}"#);
        let mut value = Some(serde_json::from_slice(&bytes).unwrap());
        attach_kv_session(Some(&sessions), &headers, "p", "d", &mut value).unwrap();
        let marks = Marks {
            kv_export: Some(json!({"transfer_id":1})),
            kv_import: Some(json!({"transfer_id":1})),
        };
        let (p, d) = build_leg_bodies(&bytes, value, None, &marks, None).unwrap();
        let p: Value = serde_json::from_slice(&p).unwrap();
        let d: Value = serde_json::from_slice(&d).unwrap();
        assert_eq!(p["rid"], "client-rid");
        assert_eq!(d["rid"], "client-rid");
        assert_eq!(p["kv_session"], d["kv_session"]);
        assert_eq!(p["kv_session"]["version"], 1);

        // Both headers: x-agent-context names the session.
        let mut both = headers.clone();
        both.insert(
            crate::agent_identity::CONTEXT_HEADER,
            r#"{"agent_id":"other","subagent_id":0,"request_id":3}"#
                .parse()
                .unwrap(),
        );
        let alias_sid = p["kv_session"]["session_id"].clone();
        let mut value = Some(json!({"prompt":[1,2]}));
        attach_kv_session(Some(&sessions), &both, "p", "d", &mut value).unwrap();
        assert_ne!(value.unwrap()["kv_session"]["session_id"], alias_sid);

        // A malformed alias: 400 before any registration.
        let mut bad = axum::http::HeaderMap::new();
        bad.insert(
            crate::agent_identity::KV_SESSION_HEADER,
            r#"{"agent_id":"c1"}"#.parse().unwrap(),
        );
        let mut value = Some(json!({"prompt":[1,2]}));
        assert!(matches!(
            attach_kv_session(Some(&sessions), &bad, "p", "d", &mut value),
            Err(ApiError::BadRequest(_))
        ));
        assert!(matches!(
            late_kv_session(Some(&sessions), &bad, "p", Some(&json!({}))),
            Err(ApiError::BadRequest(_))
        ));
    }

    /// Late D: only the P registers; the tag is returned, not inserted; a
    /// client tag is refused; no identity or no registry gives `None`.
    #[tokio::test]
    async fn late_kv_session_registers_the_prefill_only() {
        let sessions = crate::backend::kv_sessions::KvSessions::spawn(None);
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            crate::agent_identity::KV_SESSION_HEADER,
            r#"{"agent_id":"c1","subagent_id":0,"request_id":1}"#
                .parse()
                .unwrap(),
        );
        let body = json!({"prompt":[1,2],"rid":"r"});
        let tag = late_kv_session(Some(&sessions), &headers, "p", Some(&body))
            .unwrap()
            .unwrap();
        assert_eq!(tag.version, 1);
        assert_eq!(
            sessions
                .stats()
                .tagged
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        let tagged = json!({"prompt":[1],"kv_session":{"session_id":"x","version":1}});
        assert!(late_kv_session(Some(&sessions), &headers, "p", Some(&tagged)).is_err());
        assert!(late_kv_session(None, &headers, "p", Some(&body))
            .unwrap()
            .is_none());
        assert!(late_kv_session(
            Some(&sessions),
            &axum::http::HeaderMap::new(),
            "p",
            Some(&body)
        )
        .unwrap()
        .is_none());
    }

    #[tokio::test]
    /// Verify prepared target and abort payloads against a local mock engine.
    ///
    /// # Inputs
    /// No arguments; starts a temporary HTTP server accepting abort requests
    /// and uses one successful and one failed request ID.
    ///
    /// # Outputs
    /// Returns `()` after asserting endpoint/configuration metadata and one
    /// abort per ID, then stopping the mock server. Assertion failure panics.
    async fn prepared_target_and_abort_keep_native_sglang_wire_contract() {
        /// Record an abort payload and emulate an engine acknowledgement.
        ///
        /// # Inputs
        /// `seen` is shared test storage; `body` is the received JSON abort body.
        ///
        /// # Outputs
        /// Appends the body to `seen` and returns HTTP 500 for RID `failed`, or
        /// HTTP 200 for other IDs. Used only by the local wire-contract test.
        async fn abort(
            State(seen): State<Arc<Mutex<Vec<Value>>>>,
            Json(body): Json<Value>,
        ) -> StatusCode {
            seen.lock().unwrap().push(body.clone());
            if body["rid"] == "failed" {
                StatusCode::INTERNAL_SERVER_ERROR
            } else {
                StatusCode::OK
            }
        }
        let seen = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route("/abort_request", post(abort))
            .with_state(Arc::clone(&seen));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let target = prepared_target(&(base.clone() + "/"), "glm-test");
        assert_eq!(
            target,
            json!({"transport":"sglang_generate","endpoint":format!("{base}/generate"),"config_id":"glm-test"})
        );
        cancel_engines(
            reqwest::Client::new(),
            vec![
                (base.clone() + "/", "finished-p".into()),
                (base, "failed".into()),
            ],
        )
        .await;
        let mut bodies = seen.lock().unwrap().clone();
        bodies.sort_by_key(|v| v["rid"].as_str().unwrap().to_owned());
        assert_eq!(
            bodies,
            vec![json!({"rid":"failed"}), json!({"rid":"finished-p"})]
        );
        // A non-2xx acknowledgement is logged, not retried or called terminal.
        assert_eq!(seen.lock().unwrap().len(), 2);
        server.abort();
    }
}
