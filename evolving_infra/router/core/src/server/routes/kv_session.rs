// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Session KV retention on the PD path of `--policy cost function`
//! (`--kv-session-retention`).
//!
//! - Identity: the `x-kv-session-context` header (the same JSON as
//!   `x-agent-context`). The router never rewrites the body `rid` for it.
//! - Registration: right after the P of a PD request is known, the route
//!   registers (P, P) with [`KvSessions`]. With D = P, a decode change never
//!   bumps the version or retires anything; no decode worker enters the
//!   registry.
//! - Tag: only the P-leg body gets `kv_session`. No D-leg body carries it.
//! - Retirement: a P change queues the old P. The background task of
//!   [`KvSessions`] sends its release only after the new P reports the full
//!   context ready. `POST /kv_sessions/end` queues END for every engine of
//!   the session.
//! - Failures: a client `kv_session` field or a malformed header gets 400
//!   before admission. A registration error fails open: the request goes out
//!   untagged and the error is counted by reason.
//!
//! With the flag off, `AppContext::kv_sessions` is `None` and no code here
//! runs.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::http::HeaderMap;
use serde_json::Value;

use crate::agent_identity::{RequestIdentity, KV_SESSION_HEADER};
use crate::backend::kv_sessions::{inc, KvSessions};
use crate::server::error::ApiError;

/// Tracing target of this module.
pub const LOG_TARGET: &str = "kv_session";
/// The request-body field of the session tag (the engine protocol).
pub const KV_SESSION_FIELD: &str = "kv_session";
/// Period of the `KV_SESSION_ROUTER_STATS` line.
pub const STATS_PERIOD: Duration = Duration::from_secs(30);

/// Ingress, before admission: refuse a client `kv_session` field and read
/// the identity header.
///
/// # Inputs
/// `headers` are the request headers; `value` is the parsed request body.
///
/// # Outputs
/// Returns the identity, or `None` without the header. A client
/// `kv_session` field, a malformed header, or an identity on a body that is
/// not a JSON object gives a bad-request error.
pub(crate) fn ingress(
    headers: &HeaderMap,
    value: Option<&Value>,
) -> Result<Option<RequestIdentity>, ApiError> {
    if value.is_some_and(|v| v.get(KV_SESSION_FIELD).is_some()) {
        return Err(ApiError::BadRequest(
            "kv_session is assigned by the router, not the client".into(),
        ));
    }
    let identity = crate::agent_identity::from_header(headers, KV_SESSION_HEADER)?;
    if identity.is_some() && !value.is_some_and(Value::is_object) {
        return Err(ApiError::BadRequest(
            "x-kv-session-context requires a JSON object body".into(),
        ));
    }
    Ok(identity)
}

/// After P selection: register (P, P) and return the tag for the P-leg body.
///
/// # Inputs
/// `sessions` is the registry; `identity` comes from [`ingress`];
/// `prefill_url` is the selected P; `pd` is true when the request takes a PD
/// dispatch; `rid` is the client rid (for the log line only).
///
/// # Outputs
/// Returns the `kv_session` value for the P-leg body. Returns `None` for a
/// non-PD dispatch or a failed registration; both are counted, and a failed
/// registration is logged. Never fails the request.
pub(crate) fn register_prefill(
    sessions: &KvSessions,
    identity: &RequestIdentity,
    prefill_url: &str,
    pd: bool,
    rid: Option<&str>,
) -> Option<Value> {
    let stats = sessions.stats();
    if !pd {
        inc(&stats.untagged_plain);
        return None;
    }
    let error = match sessions.register(identity, prefill_url, prefill_url) {
        Ok(tag) => match serde_json::to_value(tag) {
            Ok(tag) => {
                inc(&stats.tagged);
                return Some(tag);
            }
            Err(e) => e.to_string(),
        },
        Err(e) => e,
    };
    let reason = register_error_reason(&error);
    inc(match reason {
        "stale" => &stats.register_stale,
        "ended" => &stats.register_ended,
        "capacity" => &stats.register_capacity,
        "clearing" => &stats.register_clearing,
        "invalid" => &stats.register_invalid,
        _ => &stats.register_other,
    });
    tracing::warn!(
        target: LOG_TARGET,
        rid = rid.unwrap_or("-"),
        agent_id = %identity.agent_id,
        subagent_id = identity.subagent_id,
        request_id = identity.request_id,
        prefill_url,
        reason,
        error = %error,
        "kv_session: registration failed; the request goes out untagged",
    );
    None
}

/// Input: a [`KvSessions::register`] error. Output: its counter reason.
pub(crate) fn register_error_reason(error: &str) -> &'static str {
    if error.starts_with("stale trajectory request") {
        "stale"
    } else if error.starts_with("trajectory has ended") {
        "ended"
    } else if error.contains("capacity") {
        "capacity"
    } else if error.contains("clearing") {
        "clearing"
    } else if error.contains("invalid trajectory identity") {
        "invalid"
    } else {
        "other"
    }
}

/// Input: a P-leg request value and the optional tag. Output: the value with
/// `kv_session` set when both exist and the value is a JSON object;
/// otherwise the value unchanged.
pub(crate) fn tag_prefill(mut value: Option<Value>, tag: Option<&Value>) -> Option<Value> {
    if let (Some(object), Some(tag)) = (value.as_mut().and_then(Value::as_object_mut), tag) {
        object.insert(KV_SESSION_FIELD.to_string(), tag.clone());
    }
    value
}

/// Input: one counter. Output: its value.
fn get(counter: &AtomicU64) -> u64 {
    counter.load(Ordering::Relaxed)
}

/// Write one `KV_SESSION_ROUTER_STATS` info line.
///
/// # Inputs
/// `sessions` is the registry; `reason` is `periodic` or `shutdown`.
///
/// # Outputs
/// Returns `()`. Reads the counters and scans the session table once under
/// its lock (for the pending retirements and their maximum age).
pub fn log_stats(sessions: &KvSessions, reason: &str) {
    let s = sessions.stats();
    let backlog = sessions.retire_backlog();
    let by_reason = [
        ("stale", get(&s.register_stale)),
        ("ended", get(&s.register_ended)),
        ("capacity", get(&s.register_capacity)),
        ("clearing", get(&s.register_clearing)),
        ("invalid", get(&s.register_invalid)),
        ("other", get(&s.register_other)),
    ];
    let register_errors: u64 = by_reason.iter().map(|(_, n)| n).sum();
    let register_by_reason = by_reason
        .iter()
        .map(|(reason, n)| format!("{reason}={n}"))
        .collect::<Vec<_>>()
        .join(",");
    tracing::info!(
        target: LOG_TARGET,
        reason,
        tagged = get(&s.tagged),
        untagged_plain = get(&s.untagged_plain),
        register_errors,
        register_by_reason = %register_by_reason,
        status_sent = get(&s.status_sent),
        ready = get(&s.ready),
        release_sent = get(&s.release_sent),
        release_acked = get(&s.release_acked),
        end_sent = get(&s.end_sent),
        end_acked = get(&s.end_acked),
        control_errors = get(&s.control_errors),
        retire_queued = get(&s.retire_queued),
        retire_released = get(&s.retire_released),
        retire_pending = backlog.pending,
        pending_retire_max_age_s = backlog.max_age.as_secs_f64(),
        sessions = backlog.sessions,
        end_requests = get(&s.end_requests),
        end_unknown = get(&s.end_unknown),
        "KV_SESSION_ROUTER_STATS",
    );
}

/// Start the periodic stats line (every [`STATS_PERIOD`]) on the current
/// Tokio runtime. The task ends when the registry is dropped. Without a
/// runtime (plain unit tests) nothing starts.
pub fn spawn_stats_log(sessions: &Arc<KvSessions>) {
    let weak = Arc::downgrade(sessions);
    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        return;
    };
    handle.spawn(async move {
        let mut ticker = tokio::time::interval(STATS_PERIOD);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // The first tick completes at once; the first line comes one period later.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            let Some(sessions) = weak.upgrade() else {
                break;
            };
            log_stats(&sessions, "periodic");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn identity(agent: &str, step: u64) -> RequestIdentity {
        RequestIdentity {
            agent_id: agent.into(),
            subagent_id: 0,
            request_id: step,
        }
    }

    /// Input: headers and a body that [`ingress`] must refuse. Output: the error text.
    fn refused(headers: &HeaderMap, value: &Value) -> String {
        ingress(headers, Some(value)).unwrap_err().to_string()
    }

    /// A client `kv_session` and a malformed header are refused before
    /// admission with a clear message; no header means no identity.
    #[test]
    fn ingress_refuses_a_client_tag_and_reports_header_errors() {
        let body = json!({"prompt": [1, 2], "rid": "client"});
        let mut headers = HeaderMap::new();
        assert!(ingress(&headers, Some(&body)).unwrap().is_none());

        let client_tag = json!({"prompt": [1], "kv_session": {"session_id": "x", "version": 1}});
        let error = refused(&headers, &client_tag);
        assert!(
            error.contains("kv_session is assigned by the router"),
            "{error}"
        );

        headers.insert(KV_SESSION_HEADER, "{bad".parse().unwrap());
        let error = refused(&headers, &body);
        assert!(
            error.contains("invalid x-kv-session-context header"),
            "{error}"
        );
        assert!(error.contains("expected"), "{error}");

        headers.insert(
            KV_SESSION_HEADER,
            r#"{"agent_id":"a","subagent_id":0,"request_id":3}"#
                .parse()
                .unwrap(),
        );
        let identity = ingress(&headers, Some(&body)).unwrap().unwrap();
        assert_eq!(identity.agent_id, "a");
        assert_eq!(identity.request_id, 3);
        let error = refused(&headers, &json!([1, 2]));
        assert!(error.contains("requires a JSON object body"), "{error}");
    }

    /// Registration tags the P leg; a non-PD dispatch and each registry
    /// error go out untagged; every outcome is counted.
    #[test]
    fn register_prefill_tags_counts_and_fails_open() {
        let sessions = KvSessions::spawn(None);
        let stats = sessions.stats();
        let p1 = "http://p1";
        let tag = register_prefill(&sessions, &identity("a", 5), p1, true, Some("r1"));
        let tag = tag.expect("a PD request is tagged");
        assert_eq!(tag["version"], 1);
        assert!(tag["session_id"].as_str().is_some_and(|s| !s.is_empty()));
        assert_eq!(get(&stats.tagged), 1);

        // Not a PD dispatch: untagged, nothing registered.
        let plain = register_prefill(&sessions, &identity("b", 0), p1, false, None);
        assert!(plain.is_none());
        assert_eq!(get(&stats.untagged_plain), 1);
        assert_eq!(sessions.retire_backlog().sessions, 1);

        // An older step of the same agent is stale.
        let stale = register_prefill(&sessions, &identity("a", 4), p1, true, None);
        assert!(stale.is_none());
        assert_eq!(get(&stats.register_stale), 1);

        // An ended agent cannot register again.
        sessions.end(&identity("a", 0)).unwrap();
        let ended = register_prefill(&sessions, &identity("a", 6), p1, true, None);
        assert!(ended.is_none());
        assert_eq!(get(&stats.register_ended), 1);
        assert_eq!(get(&stats.tagged), 1);
    }

    /// Every error text of [`KvSessions::register`] maps to its reason.
    #[test]
    fn register_error_reasons_cover_the_registry_messages() {
        let sessions = KvSessions::spawn(None);
        let mut errors = Vec::new();
        sessions.register(&identity("a", 2), "p", "p").unwrap();
        errors.push(sessions.register(&identity("a", 1), "p", "p").unwrap_err());
        // One session holds at most 128 workers.
        let mut capacity = None;
        for step in 3..200 {
            let p = format!("p{step}");
            if let Err(e) = sessions.register(&identity("a", step), &p, &p) {
                capacity = Some(e);
                break;
            }
        }
        errors.push(capacity.expect("the worker history fills up"));
        sessions.begin_worker_clear("q");
        errors.push(sessions.register(&identity("c", 0), "q", "q").unwrap_err());
        errors.push(sessions.register(&identity("", 0), "p", "p").unwrap_err());
        sessions.end(&identity("a", 0)).unwrap();
        errors.push(sessions.register(&identity("a", 0), "p", "p").unwrap_err());
        let reasons: Vec<_> = errors
            .iter()
            .map(String::as_str)
            .map(register_error_reason)
            .collect();
        assert_eq!(
            reasons,
            ["stale", "capacity", "clearing", "invalid", "ended"]
        );
        assert_eq!(
            register_error_reason("KV session version exhausted"),
            "other"
        );
    }

    /// The tag goes into the given value only; without a tag the value is
    /// returned unchanged.
    #[test]
    fn tag_prefill_sets_only_the_given_value() {
        let shared = json!({"prompt": [1, 2], "rid": "client"});
        let tag = json!({"session_id": "s", "version": 2});
        let tagged = tag_prefill(Some(shared.clone()), Some(&tag)).unwrap();
        assert_eq!(tagged[KV_SESSION_FIELD], tag);
        assert_eq!(tagged["rid"], "client");
        assert!(shared.get(KV_SESSION_FIELD).is_none());
        assert_eq!(tag_prefill(Some(shared.clone()), None), Some(shared));
        assert_eq!(tag_prefill(None, Some(&tag)), None);
    }

    /// A P move queues one retirement, and the stats line reads the counters
    /// without a runtime.
    #[test]
    fn stats_line_reads_counters() {
        let sessions = KvSessions::spawn(None);
        let first = register_prefill(&sessions, &identity("a", 0), "p1", true, None);
        let moved = register_prefill(&sessions, &identity("a", 1), "p2", true, None);
        assert_eq!(moved.unwrap()["version"], 2);
        assert_eq!(first.unwrap()["version"], 1);
        let backlog = sessions.retire_backlog();
        assert_eq!((backlog.sessions, backlog.pending), (1, 1));
        assert_eq!(get(&sessions.stats().retire_queued), 1);
        log_stats(&sessions, "test");
        spawn_stats_log(&sessions);
    }
}
