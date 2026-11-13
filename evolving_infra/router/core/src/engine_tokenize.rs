// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! `--chat-tools-tokenize`: token ids of a chat request with tools, from an
//! engine.
//!
//! The router's chat encoder renders `messages` only, so the ids it computes
//! for a request with tools differ from the engine's from the tool schemas on
//! and match no cached prefix. The engine's `/v1/tokenize` renders messages
//! and tools with the code its chat route uses, so its ids are the prompt the
//! request will run. They feed placement (prompt length, KV directory query,
//! pull hint) and are never forwarded as `input_ids`: the engine still
//! tokenizes the request it serves.

use crate::workers::Worker;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

/// A request waits at most this long for the ids; after it, the request is
/// placed as before this option existed.
pub const TIMEOUT: Duration = Duration::from_secs(3);

/// Request fields the engine's chat template reads.
const FIELDS: [&str; 7] = [
    "model",
    "messages",
    "tools",
    "tool_choice",
    "chat_template_kwargs",
    "continue_final_message",
    "reasoning_effort",
];

static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
static NEXT: AtomicUsize = AtomicUsize::new(0);

/// The body of the engine's `/v1/tokenize` for chat request `value`.
pub fn tokenize_body(value: &serde_json::Value) -> serde_json::Value {
    let mut body = serde_json::Map::new();
    for field in FIELDS {
        if let Some(v) = value.get(field).filter(|v| !v.is_null()) {
            body.insert(field.to_string(), v.clone());
        }
    }
    serde_json::Value::Object(body)
}

/// The `tokens` of an engine `/v1/tokenize` reply: one flat array of ids.
pub fn reply_ids(reply: &serde_json::Value) -> Option<Vec<u32>> {
    let ids = reply
        .get("tokens")?
        .as_array()?
        .iter()
        .map(|v| v.as_u64().and_then(|id| u32::try_from(id).ok()))
        .collect::<Option<Vec<u32>>>()?;
    (!ids.is_empty()).then_some(ids)
}

/// Ask one of `workers` (taken in turn) for the token ids of chat request
/// `value`. `None` on any failure: the caller keeps the ids it had.
pub async fn chat_ids(workers: &[Arc<Worker>], value: &serde_json::Value) -> Option<Vec<u32>> {
    if workers.is_empty() {
        return None;
    }
    let worker = &workers[NEXT.fetch_add(1, Ordering::Relaxed) % workers.len()];
    let url = format!("{}/v1/tokenize", worker.url.trim_end_matches('/'));
    let client = CLIENT.get_or_init(reqwest::Client::new);
    let reply = client
        .post(&url)
        .timeout(TIMEOUT)
        .json(&tokenize_body(value))
        .send()
        .await
        .and_then(reqwest::Response::error_for_status);
    let reply = match reply {
        Ok(reply) => reply.json::<serde_json::Value>().await,
        Err(e) => Err(e),
    };
    match reply {
        Ok(reply) => reply_ids(&reply),
        Err(error) => {
            tracing::warn!(worker = %worker.url, %error, "engine tokenize of a chat request with tools failed");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn body_keeps_only_what_the_template_reads() {
        let body = tokenize_body(&json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": "auto", "chat_template_kwargs": null,
            "stream": true, "max_tokens": 7, "rid": "r",
        }));
        assert_eq!(
            body,
            json!({
                "model": "m", "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "f"}}],
                "tool_choice": "auto",
            })
        );
    }

    #[test]
    fn reply_ids_reads_a_flat_array_only() {
        assert_eq!(
            reply_ids(&json!({"tokens": [1, 2, 3], "count": 3})),
            Some(vec![1, 2, 3])
        );
        assert_eq!(reply_ids(&json!({"tokens": []})), None);
        assert_eq!(reply_ids(&json!({"tokens": [[1], [2]]})), None);
        assert_eq!(reply_ids(&json!({"tokens": [1, -2]})), None);
        assert_eq!(reply_ids(&json!({"count": 3})), None);
    }
}
