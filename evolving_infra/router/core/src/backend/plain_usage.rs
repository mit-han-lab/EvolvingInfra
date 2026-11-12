// SPDX-License-Identifier: Apache-2.0

//! Completed work of requests the router proxies itself (the plain routes),
//! for the external role controller.
//!
//! The controller's workload sums (`GET /symmetric_pd/roles`) came only from
//! the prepared path, where the tokenizer reports each leg's usage. A plain
//! request is counted here from the replies the router already forwards:
//!
//! * the prefill leg's reply gives the prompt tokens the prefill engine reused
//!   (`usage.prompt_tokens_details.cached_tokens`; absent means none);
//! * the decode leg's reply, which is the client's, gives the prompt length,
//!   the output tokens and the speculative verify steps
//!   (`usage`, `sglext.spec_tokens_details`).
//!
//! The engines send usage on a stream only when asked, and the verify steps
//! only with `return_spec_tokens_details`, so [`ask`] sets both on the request
//! and [`PlainUsage::tap`] removes from the client's reply what the client
//! did not ask for: the chunks that carry only that, and the `sglext` field
//! of the other chunks and of a buffered reply. A request is counted once, when its decode reply
//! ends complete; a failed or cut reply counts nothing.

use crate::backend::workload::{CompletedWorkload, RequestSample};
use axum::body::Body;
use axum::http::Response;
use bytes::BytesMut;
use futures::StreamExt;
use parking_lot::Mutex;
use serde_json::Value;
use std::sync::Arc;

/// A buffered (non-streaming) reply larger than this is forwarded uncounted.
const MAX_BUFFERED_REPLY: usize = 64 << 20;

/// What the client itself asked for; the rest of what [`ask`] turned on is
/// removed from its reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClientAsked {
    pub usage: bool,
    pub spec_details: bool,
}

/// Turn on usage reporting in request `value` (both legs carry it) and return
/// what the client had asked for.
pub fn ask(value: &mut Value, streaming: bool) -> ClientAsked {
    let Some(obj) = value.as_object_mut() else {
        return ClientAsked {
            usage: true,
            spec_details: true,
        };
    };
    let spec_details = obj
        .get("return_spec_tokens_details")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    obj.insert("return_spec_tokens_details".into(), Value::Bool(true));
    let mut usage = !streaming;
    if streaming {
        let options = obj
            .entry("stream_options")
            .or_insert_with(|| Value::Object(Default::default()));
        if !options.is_object() {
            *options = Value::Object(Default::default());
        }
        let options = options.as_object_mut().expect("object");
        usage = options
            .get("include_usage")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        options.insert("include_usage".into(), Value::Bool(true));
    }
    ClientAsked {
        usage,
        spec_details,
    }
}

#[derive(Default)]
struct Seen {
    prompt_tokens: Option<u64>,
    /// From the prefill leg's reply.
    cached_tokens: Option<u64>,
    output_tokens: Option<u64>,
    verify_steps: Option<u64>,
    counted: bool,
}

/// One plain request's usage, shared by its prefill task and its decode reply.
#[derive(Clone)]
pub struct PlainUsage {
    workload: Arc<CompletedWorkload>,
    seen: Arc<Mutex<Seen>>,
    /// Cached prompt tokens the placement expected on the prefill worker;
    /// used when the prefill reply carried no usage.
    expected_cached: u64,
}

impl PlainUsage {
    pub fn new(workload: Arc<CompletedWorkload>, expected_cached: u64) -> Self {
        Self {
            workload,
            seen: Arc::default(),
            expected_cached,
        }
    }

    /// The prefill leg's complete reply (JSON, or an event stream).
    pub fn prefill_reply(&self, body: &[u8]) {
        let mut found = Seen::default();
        for event in events(body) {
            read(&event, &mut found);
        }
        if found.prompt_tokens.is_some() {
            self.seen.lock().cached_tokens = Some(found.cached_tokens.unwrap_or(0));
        }
    }

    fn observe(&self, event: &Value) {
        let mut seen = self.seen.lock();
        let cached = seen.cached_tokens;
        read(event, &mut seen);
        // The decode leg imported the prompt's KV: its own cached count says
        // nothing about the prefill's work.
        seen.cached_tokens = cached;
    }

    /// The decode reply ended complete: count the request once.
    fn finish(&self) {
        let mut seen = self.seen.lock();
        if seen.counted {
            return;
        }
        let (Some(input_tokens), Some(output_tokens)) = (seen.prompt_tokens, seen.output_tokens)
        else {
            return;
        };
        seen.counted = true;
        self.workload.add(RequestSample {
            input_tokens,
            cached_tokens: seen.cached_tokens.unwrap_or(self.expected_cached),
            output_tokens,
            verify_steps: seen.verify_steps,
            vision_matmul_flops: 0,
            image_attention_pairs: 0,
        });
    }

    /// Count the request from the client's reply `response` as it passes,
    /// removing what only [`ask`] asked for: the chunks that carry only that,
    /// and `sglext` when the client did not ask for it (a buffered reply is
    /// then held until its end, up to [`MAX_BUFFERED_REPLY`]). A non-2xx
    /// reply passes untouched.
    pub fn tap(self, response: Response<Body>, asked: ClientAsked) -> Response<Body> {
        if !response.status().is_success() {
            return response;
        }
        let sse = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.starts_with("text/event-stream"));
        let (mut parts, body) = response.into_parts();
        if sse || !asked.spec_details {
            // The forwarded bytes may differ from the engine's.
            parts.headers.remove(axum::http::header::CONTENT_LENGTH);
        }
        struct Tap<S> {
            stream: S,
            pending: BytesMut,
            usage: PlainUsage,
            sse: bool,
            asked: ClientAsked,
            complete: bool,
            done: bool,
        }
        let tap = Tap {
            stream: body.into_data_stream(),
            pending: BytesMut::new(),
            usage: self,
            sse,
            asked,
            complete: true,
            done: false,
        };
        let out = futures::stream::unfold(tap, |mut tap| async move {
            loop {
                if tap.done {
                    return None;
                }
                match tap.stream.next().await {
                    Some(Err(error)) => {
                        tap.complete = false;
                        tap.done = true;
                        return Some((Err(error), tap));
                    }
                    Some(Ok(chunk)) if !tap.sse => {
                        // A buffered reply: read at its end. It is forwarded
                        // as it comes when the client asked for `sglext`,
                        // else held to remove `sglext` at its end.
                        let hold = !tap.asked.spec_details;
                        if tap.complete && tap.pending.len() + chunk.len() <= MAX_BUFFERED_REPLY {
                            tap.pending.extend_from_slice(&chunk);
                            if hold {
                                continue;
                            }
                            return Some((Ok(chunk), tap));
                        }
                        // Too large to read: forwarded uncounted and as is.
                        tap.complete = false;
                        if hold && !tap.pending.is_empty() {
                            let mut held = tap.pending.split();
                            held.extend_from_slice(&chunk);
                            return Some((Ok(held.freeze()), tap));
                        }
                        return Some((Ok(chunk), tap));
                    }
                    Some(Ok(chunk)) => {
                        tap.pending.extend_from_slice(&chunk);
                        let mut forward = BytesMut::new();
                        while let Some(end) = find(&tap.pending, b"\n\n") {
                            let event = tap.pending.split_to(end + 2);
                            match data(&event) {
                                Some(mut value) => {
                                    tap.usage.observe(&value);
                                    if only_asked_by_router(&value, tap.asked) {
                                        continue;
                                    }
                                    if strip_unasked(&mut value, tap.asked) {
                                        forward.extend_from_slice(b"data: ");
                                        forward.extend_from_slice(
                                            &serde_json::to_vec(&value).unwrap_or_default(),
                                        );
                                        forward.extend_from_slice(b"\n\n");
                                    } else {
                                        forward.extend_from_slice(&event);
                                    }
                                }
                                None => forward.extend_from_slice(&event),
                            }
                        }
                        if !forward.is_empty() {
                            return Some((Ok(forward.freeze()), tap));
                        }
                    }
                    None => {
                        tap.done = true;
                        let rest = tap.pending.split();
                        let held = !tap.sse && !tap.asked.spec_details;
                        let mut stripped = None;
                        if !tap.sse && tap.complete {
                            if let Ok(mut value) = serde_json::from_slice::<Value>(&rest) {
                                tap.usage.observe(&value);
                                if held && strip_unasked(&mut value, tap.asked) {
                                    stripped = serde_json::to_vec(&value).ok();
                                }
                            }
                        }
                        if tap.complete {
                            tap.usage.finish();
                        }
                        if let Some(bytes) = stripped {
                            return Some((Ok(bytes.into()), tap));
                        }
                        if (tap.sse || held) && !rest.is_empty() {
                            return Some((Ok(rest.freeze()), tap));
                        }
                        return None;
                    }
                }
            }
        });
        Response::from_parts(parts, Body::from_stream(out))
    }
}

/// A prefill reply larger than this is not read.
const MAX_PREFILL_REPLY: usize = 16 << 20;

/// Read the prefill leg's reply for `usage`, when the leg answered 2xx. Takes
/// the reply's body: nothing else reads it.
pub async fn read_prefill(
    usage: Option<&PlainUsage>,
    reply: &mut Result<Response<Body>, crate::server::error::ApiError>,
) {
    let (Some(usage), Ok(reply)) = (usage, reply) else {
        return;
    };
    if !reply.status().is_success() {
        return;
    }
    let body = std::mem::take(reply.body_mut());
    if let Ok(bytes) = axum::body::to_bytes(body, MAX_PREFILL_REPLY).await {
        usage.prefill_reply(&bytes);
    }
}

/// [`PlainUsage::tap`] on a handler's result, when the request is counted.
pub fn count(
    result: Result<Response<Body>, crate::server::error::ApiError>,
    usage: Option<PlainUsage>,
    asked: Option<ClientAsked>,
) -> Result<Response<Body>, crate::server::error::ApiError> {
    match (result, usage, asked) {
        (Ok(response), Some(usage), Some(asked)) => Ok(usage.tap(response, asked)),
        (result, ..) => result,
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// The JSON of an SSE `data:` event that may carry usage; `None` for other
/// events (most token chunks are not parsed at all).
fn data(event: &[u8]) -> Option<Value> {
    let payload = event.strip_prefix(b"data:")?;
    if find(payload, b"\"usage\"").is_none() && find(payload, b"\"sglext\"").is_none() {
        return None;
    }
    serde_json::from_slice(payload).ok()
}

/// Every JSON object of a complete reply: the body itself, or its `data:` events.
fn events(body: &[u8]) -> Vec<Value> {
    if let Ok(value) = serde_json::from_slice::<Value>(body) {
        return vec![value];
    }
    body.split(|b| *b == b'\n').filter_map(data).collect()
}

fn read(event: &Value, seen: &mut Seen) {
    if let Some(usage) = event.get("usage").filter(|u| u.is_object()) {
        if let Some(n) = usage.get("prompt_tokens").and_then(Value::as_u64) {
            seen.prompt_tokens = Some(n);
            seen.cached_tokens = Some(
                usage
                    .pointer("/prompt_tokens_details/cached_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
            );
        }
        if let Some(n) = usage.get("completion_tokens").and_then(Value::as_u64) {
            seen.output_tokens = Some(n);
        }
    }
    let details = event.pointer("/sglext/spec_tokens_details");
    let details = match details {
        Some(Value::Array(items)) => items.first(),
        other => other,
    };
    if let Some(n) = details
        .and_then(|d| d.get("spec_verify_ct"))
        .and_then(Value::as_u64)
    {
        seen.verify_steps = Some(n);
    }
}

/// Remove `sglext` from a chunk or reply the client did not ask it of;
/// true when it was there.
fn strip_unasked(event: &mut Value, asked: ClientAsked) -> bool {
    if asked.spec_details {
        return false;
    }
    event
        .as_object_mut()
        .and_then(|obj| obj.remove("sglext"))
        .is_some()
}

/// A chunk without choices that carries only what the router asked for.
fn only_asked_by_router(event: &Value, asked: ClientAsked) -> bool {
    let no_choices = event
        .get("choices")
        .and_then(Value::as_array)
        .is_some_and(Vec::is_empty);
    let has = |key: &str| event.get(key).is_some_and(|v| !v.is_null());
    no_choices
        && (has("usage") || has("sglext"))
        && (!has("usage") || !asked.usage)
        && (!has("sglext") || !asked.spec_details)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use http_body_util::BodyExt;
    use serde_json::json;

    fn usage() -> (PlainUsage, Arc<CompletedWorkload>) {
        let workload = Arc::new(CompletedWorkload::default());
        (PlainUsage::new(Arc::clone(&workload), 100), workload)
    }

    async fn through(
        u: PlainUsage,
        content_type: &str,
        chunks: Vec<&'static str>,
        asked: ClientAsked,
    ) -> String {
        let stream = futures::stream::iter(
            chunks
                .into_iter()
                .map(|c| Ok::<_, std::io::Error>(Bytes::from_static(c.as_bytes()))),
        );
        let response = Response::builder()
            .header("content-type", content_type)
            .body(Body::from_stream(stream))
            .unwrap();
        let body = u.tap(response, asked).into_body().collect().await.unwrap();
        String::from_utf8(body.to_bytes().to_vec()).unwrap()
    }

    const NOTHING: ClientAsked = ClientAsked {
        usage: false,
        spec_details: false,
    };

    #[test]
    fn ask_turns_reporting_on_and_remembers_the_client() {
        let mut v = json!({"model": "m", "stream": true});
        assert_eq!(ask(&mut v, true), NOTHING);
        assert_eq!(v["stream_options"]["include_usage"], true);
        assert_eq!(v["return_spec_tokens_details"], true);

        let mut v =
            json!({"stream_options": {"include_usage": true}, "return_spec_tokens_details": true});
        assert_eq!(
            ask(&mut v, true),
            ClientAsked {
                usage: true,
                spec_details: true
            }
        );
        // A buffered reply always has usage; no stream option is added.
        let mut v = json!({"model": "m"});
        assert!(ask(&mut v, false).usage);
        assert!(v.get("stream_options").is_none());
    }

    #[tokio::test]
    async fn stream_is_counted_and_router_only_chunks_are_removed() {
        let (u, workload) = usage();
        u.prefill_reply(
            br#"{"usage": {"prompt_tokens": 1000, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 768}}}"#,
        );
        // Events split across network chunks.
        let out = through(
            u,
            "text/event-stream",
            vec![
                "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\ndata: {\"choi",
                "ces\":[],\"sglext\":{\"spec_tokens_details\":{\"spec_verify_ct\":4}}}\n\n",
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1000,\"completion_tokens\":12}}\n\n",
                "data: [DONE]\n\n",
            ],
            NOTHING,
        )
        .await;
        assert_eq!(
            out,
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\ndata: [DONE]\n\n"
        );
        let sums = workload.totals();
        assert_eq!(
            (
                sums.requests,
                sums.input_tokens,
                sums.uncached_tokens,
                sums.output_tokens,
                sums.verify_steps
            ),
            (1, 1000, 232, 12, 4)
        );
    }

    #[tokio::test]
    async fn what_the_client_asked_for_stays() {
        let (u, workload) = usage();
        let chunks = vec![
            "data: {\"choices\":[],\"sglext\":{\"spec_tokens_details\":{\"spec_verify_ct\":4}}}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2}}\n\n",
        ];
        let out = through(
            u,
            "text/event-stream",
            chunks,
            ClientAsked {
                usage: true,
                spec_details: false,
            },
        )
        .await;
        assert!(
            out.contains("\"usage\"") && !out.contains("sglext"),
            "{out}"
        );
        // No prefill reply was seen: the placement's expectation is used,
        // bounded by the prompt.
        assert_eq!(workload.totals().uncached_tokens, 0);
        assert_eq!(workload.totals().requests, 1);
    }

    /// `sglext` that only the router asked for leaves the chunks that carry
    /// choices and a buffered reply; it is still counted.
    #[tokio::test]
    async fn unasked_sglext_is_removed_from_every_reply() {
        let (u, workload) = usage();
        let out = through(
            u,
            "text/event-stream",
            vec![
                "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}],\"sglext\":{\"spec_tokens_details\":{\"spec_verify_ct\":3}}}\n\n",
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2}}\n\n",
                "data: [DONE]\n\n",
            ],
            NOTHING,
        )
        .await;
        assert_eq!(
            out,
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\ndata: [DONE]\n\n"
        );
        assert_eq!(workload.totals().requests, 1);
        let (u, workload) = usage();
        let body = vec![
            r#"{"choices":[{"message":{"content":"x"}}],"usage":{"prompt_tokens":30,"#,
            r#""completion_tokens":4},"sglext":{"spec_tokens_details":{"spec_verify_ct":2}}}"#,
        ];
        let out = through(u, "application/json", body, NOTHING).await;
        assert_eq!(
            out,
            r#"{"choices":[{"message":{"content":"x"}}],"usage":{"prompt_tokens":30,"completion_tokens":4}}"#
        );
        assert_eq!(workload.totals().requests, 1);
        let (u, _) = usage();
        let asked = ClientAsked {
            usage: true,
            spec_details: true,
        };
        let body = r#"{"choices":[],"sglext":{"spec_tokens_details":{"spec_verify_ct":2}}}"#;
        assert_eq!(
            through(u, "application/json", vec![body], asked).await,
            body
        );
    }

    #[tokio::test]
    async fn buffered_reply_is_counted_and_unchanged() {
        let (u, workload) = usage();
        u.prefill_reply(br#"{"usage": {"prompt_tokens": 300, "prompt_tokens_details": null}}"#);
        let body = r#"{"choices":[{"message":{"content":"x"}}],"usage":{"prompt_tokens":300,"completion_tokens":45}}"#;
        let out = through(u, "application/json", vec![body], NOTHING).await;
        assert_eq!(out, body);
        let sums = workload.totals();
        assert_eq!(
            (sums.requests, sums.uncached_tokens, sums.output_tokens),
            (1, 300, 45)
        );
    }

    #[tokio::test]
    async fn failed_or_cut_replies_count_nothing() {
        let (u, workload) = usage();
        let response = Response::builder()
            .status(500)
            .body(Body::from(
                r#"{"usage":{"prompt_tokens":1,"completion_tokens":1}}"#,
            ))
            .unwrap();
        let _ = u.clone().tap(response, NOTHING).into_body().collect().await;
        // A stream that ends in a transport error after its usage chunk.
        let stream = futures::stream::iter(vec![
            Ok(Bytes::from_static(
                b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2}}\n\n",
            )),
            Err(std::io::Error::other("cut")),
        ]);
        let response = Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from_stream(stream))
            .unwrap();
        let _ = u.tap(response, NOTHING).into_body().collect().await;
        assert_eq!(workload.totals().requests, 0);
    }
}
