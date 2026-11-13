// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! The Python frontend's problem / branch / step identity at HTTP ingress.
//! Identity is independent of routing affinity, cache namespace and retries.
//!
//! Validates the SDK identity header and derives a deterministic engine RID using
//! the same byte layout as Python. Existing cache salt, affinity and prompt fields
//! remain independent; conflicting engine IDs are rejected at ingress.
use axum::http::HeaderMap;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::server::error::ApiError;

pub const CONTEXT_HEADER: &str = "x-agent-context";

/// Identity header of session KV retention on the cost function PD path
/// (`--kv-session-retention`). Same JSON as [`CONTEXT_HEADER`], but the
/// router never derives, inserts or checks a body `rid` from it, so the
/// client's `rid` reaches the engines unchanged. The proxy header whitelist
/// keeps it away from the engines.
pub const KV_SESSION_HEADER: &str = "x-kv-session-context";

/// Largest identity header value, in bytes.
const MAX_IDENTITY_HEADER_BYTES: usize = 4096;

/// The identity JSON, as the 400 messages of [`from_header`] show it.
const IDENTITY_JSON: &str =
    r#"{"agent_id":"<1..512 UTF-8 bytes>","subagent_id":<u64>,"request_id":<u64>}"#;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestIdentity {
    pub agent_id: String,
    pub subagent_id: u64,
    pub request_id: u64,
}

impl RequestIdentity {
    /// Derive the stable engine request ID shared with the Python frontend.
    ///
    /// # Inputs
    /// `self` supplies an agent ID of 1..512 UTF-8 bytes and unsigned subagent
    /// and request IDs. Integer values and the agent byte length use big-endian
    /// encoding before hashing.
    ///
    /// # Outputs
    /// Returns `agent_` plus the SHA-256 digest of the identity tuple, or
    /// a bad-request error for an empty/oversized agent ID. Does not select a
    /// worker or alter the cache namespace.
    pub fn engine_rid(&self) -> Result<String, ApiError> {
        if self.agent_id.is_empty() || self.agent_id.len() > 512 {
            return Err(ApiError::BadRequest(
                "agent_id must contain 1..512 UTF-8 bytes".into(),
            ));
        }
        let mut hash = Sha256::new();
        hash.update((self.agent_id.len() as u64).to_be_bytes());
        hash.update(self.agent_id.as_bytes());
        hash.update(self.subagent_id.to_be_bytes());
        hash.update(self.request_id.to_be_bytes());
        Ok(format!("agent_{:x}", hash.finalize()))
    }
}

/// Input: request headers. Output: validated optional identity; no routing or KV mutation.
pub fn from_headers(headers: &HeaderMap) -> Result<Option<RequestIdentity>, ApiError> {
    let Some(context) = headers.get(CONTEXT_HEADER) else {
        return Ok(None);
    };
    if context.as_bytes().len() > 4096 {
        return Err(ApiError::BadRequest("request identity is too large".into()));
    }
    let identity: RequestIdentity = serde_json::from_slice(context.as_bytes())
        .map_err(|_| ApiError::BadRequest("invalid request identity".into()))?;
    identity.engine_rid()?;
    Ok(Some(identity))
}

/// Read and validate the identity JSON of one header.
///
/// # Inputs
/// `headers` are the request headers; `name` is the identity header (for
/// example [`KV_SESSION_HEADER`]).
///
/// # Outputs
/// Returns `None` without the header, else the validated identity. An
/// oversized value, invalid JSON, a missing or negative field, an unknown
/// field or an empty or oversized `agent_id` gives a bad-request error that
/// names the header and the expected JSON. Unlike [`apply`], it never reads
/// or changes the request body.
pub fn from_header(headers: &HeaderMap, name: &str) -> Result<Option<RequestIdentity>, ApiError> {
    let Some(value) = headers.get(name) else {
        return Ok(None);
    };
    if value.as_bytes().len() > MAX_IDENTITY_HEADER_BYTES {
        return Err(invalid_header(name, "larger than 4096 bytes"));
    }
    let identity: RequestIdentity = serde_json::from_slice(value.as_bytes())
        .map_err(|e| invalid_header(name, &e.to_string()))?;
    if identity.engine_rid().is_err() {
        return Err(invalid_header(name, "agent_id needs 1..512 UTF-8 bytes"));
    }
    Ok(Some(identity))
}

/// The session identity of a request for the cost-function copy price `auto`
/// (N_active): the identity of [`KV_SESSION_HEADER`], else of
/// [`CONTEXT_HEADER`], as
/// [`crate::scheduler::placement_cost::placement_session_key`]. A missing or
/// invalid header gives None; this never rejects a request (the routes
/// validate the headers they use themselves).
pub fn placement_session_key(headers: &HeaderMap) -> Option<String> {
    [KV_SESSION_HEADER, CONTEXT_HEADER]
        .into_iter()
        .find_map(|name| from_header(headers, name).ok().flatten())
        .map(|identity| {
            crate::scheduler::placement_cost::placement_session_key(
                &identity.agent_id,
                identity.subagent_id,
            )
        })
}

/// Input: the header name and the reason. Output: the bad-request error of
/// [`from_header`], with the expected JSON.
fn invalid_header(name: &str, reason: &str) -> ApiError {
    ApiError::BadRequest(format!(
        "invalid {name} header: {reason}; expected {IDENTITY_JSON}"
    ))
}

/// Validate an agent identity header and ensure the request carries its RID.
///
/// # Inputs
/// `headers` may contain `x-agent-context`; `body` is an optional
/// mutable parsed JSON request body.
///
/// # Outputs
/// Returns `true` when a derived RID was inserted, or `false` when no
/// header exists or the same RID was already present. Malformed/oversized
/// identity, missing object body or a conflicting RID returns a bad-request
/// error. Other body fields are unchanged; accepted identities are logged.
pub fn apply(headers: &HeaderMap, body: Option<&mut Value>) -> Result<bool, ApiError> {
    let Some(identity) = from_headers(headers)? else {
        return Ok(false);
    };
    let rid = identity.engine_rid()?;
    let object = body.and_then(Value::as_object_mut).ok_or_else(|| {
        ApiError::BadRequest("requests with an identity header require a JSON object".into())
    })?;
    let inserted = match object.get("rid") {
        None | Some(Value::Null) => {
            object.insert("rid".into(), Value::String(rid.clone()));
            true
        }
        Some(Value::String(existing)) if *existing == rid => false,
        _ => {
            return Err(ApiError::BadRequest(
                "rid conflicts with the request identity".into(),
            ));
        }
    };
    tracing::info!(target: "agent_request", agent_id = %identity.agent_id,
                   subagent_id = identity.subagent_id, request_id = identity.request_id,
                   rid, "request identity accepted");
    Ok(inserted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    /// Verify deterministic RID compatibility, unchanged payload/namespace and conflict rejection.
    ///
    /// # Inputs
    /// No arguments. Use the Python golden identity, a cache salt and an existing prompt.
    ///
    /// # Outputs
    /// Returns `()` when the stated invariant holds; failed assertions panic.
    fn python_wire_identity_and_namespace_are_preserved() {
        let mut headers = HeaderMap::new();
        // Header JSON is ASCII on the Python wire.
        headers.insert(
            CONTEXT_HEADER,
            r#"{"agent_id":"question/\u4f60\u597d","subagent_id":2,"request_id":7}"#
                .parse()
                .unwrap(),
        );
        let expected = "agent_e471509b10f0c301d8c86761c428ff065faf2d6d9ad0e8939059aa277dda0759"; // Python SDK golden wire identity
        let mut body = json!({"model":"test", "prompt":[1,2], "cache_salt":"shared"});
        apply(&headers, Some(&mut body)).unwrap();
        assert_eq!(body["rid"], expected);
        assert_eq!(body["cache_salt"], "shared");
        assert_eq!(body["prompt"], json!([1, 2]));
        apply(&headers, Some(&mut body)).unwrap();
        body["rid"] = json!("different-step");
        assert!(apply(&headers, Some(&mut body)).is_err());
    }

    #[test]
    /// Verify malformed SDK identity is rejected at HTTP ingress.
    ///
    /// # Inputs
    /// No arguments. Supply missing fields, a negative branch ID and an empty agent ID.
    ///
    /// # Outputs
    /// Returns `()` when the stated invariant holds; failed assertions panic.
    fn partial_or_negative_identity_is_rejected() {
        for raw in [
            r#"{"agent_id":"q","subagent_id":1}"#,
            r#"{"agent_id":"q","subagent_id":-1,"request_id":0}"#,
            r#"{"agent_id":"","subagent_id":0,"request_id":0}"#,
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(CONTEXT_HEADER, raw.parse().unwrap());
            assert!(apply(&headers, Some(&mut json!({}))).is_err());
        }
    }

    #[test]
    /// Verify ordinary clients retain their existing request ID without new requirements.
    ///
    /// # Inputs
    /// No arguments. Submit no identity header, with and without a request body.
    ///
    /// # Outputs
    /// Returns `()` when the stated invariant holds; failed assertions panic.
    fn unmarked_clients_remain_unchanged() {
        assert!(apply(&HeaderMap::new(), None).is_ok());
        let mut body = json!({"rid":"original"});
        apply(&HeaderMap::new(), Some(&mut body)).unwrap();
        assert_eq!(body["rid"], "original");
    }

    #[test]
    /// Verify the session KV header parses on its own and leaves the body rid unchanged.
    ///
    /// # Inputs
    /// No arguments. Use one `x-kv-session-context` header and a client rid.
    ///
    /// # Outputs
    /// Returns `()` when the stated invariant holds; failed assertions panic.
    fn kv_session_header_parses_without_touching_the_body() {
        let mut headers = HeaderMap::new();
        assert!(from_header(&headers, KV_SESSION_HEADER).unwrap().is_none());
        headers.insert(
            KV_SESSION_HEADER,
            r#"{"agent_id":"task-1/agent-3","subagent_id":0,"request_id":4}"#
                .parse()
                .unwrap(),
        );
        let identity = from_header(&headers, KV_SESSION_HEADER).unwrap().unwrap();
        assert_eq!(identity.agent_id, "task-1/agent-3");
        assert_eq!((identity.subagent_id, identity.request_id), (0, 4));
        // `apply` reads only x-agent-context: the client rid stays.
        let mut body = json!({"rid": "client-rid"});
        assert!(!apply(&headers, Some(&mut body)).unwrap());
        assert_eq!(body["rid"], "client-rid");
        assert!(from_headers(&headers).unwrap().is_none());
    }

    #[test]
    /// Verify a malformed session KV header gives a 400 that names the header and the JSON.
    ///
    /// # Inputs
    /// No arguments. Supply invalid JSON, a missing, negative and unknown field,
    /// an empty agent ID and an oversized value.
    ///
    /// # Outputs
    /// Returns `()` when the stated invariant holds; failed assertions panic.
    fn malformed_kv_session_header_gives_a_clear_bad_request() {
        let missing = r#"{"agent_id":"a","subagent_id":0}"#;
        let negative = r#"{"agent_id":"a","subagent_id":-1,"request_id":0}"#;
        let unknown = r#"{"agent_id":"a","subagent_id":0,"request_id":0,"x":1}"#;
        let empty = r#"{"agent_id":"","subagent_id":0,"request_id":0}"#;
        let oversized = "a".repeat(5000);
        for (raw, needle) in [
            ("not json", "line 1 column 2"),
            (missing, "missing field `request_id`"),
            (negative, "invalid value"),
            (unknown, "unknown field `x`"),
            (empty, "agent_id needs 1..512"),
            (oversized.as_str(), "larger than 4096 bytes"),
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(KV_SESSION_HEADER, raw.parse().unwrap());
            let error = from_header(&headers, KV_SESSION_HEADER).unwrap_err();
            assert!(matches!(error, ApiError::BadRequest(_)), "{error}");
            let message = error.to_string();
            assert!(
                message.contains("invalid x-kv-session-context header"),
                "{message}"
            );
            assert!(message.contains(needle), "{raw}: {message}");
            assert!(message.contains(r#""request_id":<u64>"#), "{message}");
        }
    }

    /// The cost-function auto session key: the kv-session identity first, then the
    /// agent identity; the request step does not change it; a missing
    /// or invalid header gives None instead of an error.
    #[test]
    fn cost_auto_session_key_prefers_the_kv_session_header() {
        let mut headers = HeaderMap::new();
        assert_eq!(placement_session_key(&headers), None);
        headers.insert(
            CONTEXT_HEADER,
            r#"{"agent_id":"m","subagent_id":2,"request_id":5}"#
                .parse()
                .unwrap(),
        );
        assert_eq!(placement_session_key(&headers).as_deref(), Some("2/m"));
        headers.insert(
            KV_SESSION_HEADER,
            r#"{"agent_id":"k","subagent_id":0,"request_id":9}"#
                .parse()
                .unwrap(),
        );
        assert_eq!(placement_session_key(&headers).as_deref(), Some("0/k"));
        headers.insert(
            KV_SESSION_HEADER,
            r#"{"agent_id":"k","subagent_id":0,"request_id":10}"#
                .parse()
                .unwrap(),
        );
        assert_eq!(placement_session_key(&headers).as_deref(), Some("0/k"));
        headers.insert(KV_SESSION_HEADER, "not json".parse().unwrap());
        assert_eq!(placement_session_key(&headers).as_deref(), Some("2/m"));
        headers.remove(CONTEXT_HEADER);
        assert_eq!(placement_session_key(&headers), None);
    }
}
