// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Introspection for newly-discovered workers.
//!
//! Two concurrent requests, because the worker answers two different
//! questions on two different endpoints: `/model_info` reports the identity
//! the worker currently serves under (a weight update moves it), while
//! `/server_info` reports its launch configuration — kv-event publisher and
//! disaggregation role. The result is dispatched by the manager: registry
//! consumes `served_model_name`, the optional `KvEventIndex` consumes the
//! resolved `EventConfig`.
//!
//! `served_model_name` is taken from `/model_info`, falling back to
//! `/server_info` for workers that predate the field there.
//!
//! # Failure semantics
//!
//! `fetch` is **infallible** — any error (network, non-2xx, JSON parse,
//! invalid worker URL) is logged and yields `None` for whatever that request
//! was carrying, so the caller can register the worker with empty `model_ids`
//! and no kv-events attachment. The two requests fail independently. Workers
//! that need accuracy around publisher availability use
//! `kv_events::discovery::fetch_event_config` directly (it returns
//! `Result<Option<EventConfig>>`); the manager intentionally doesn't.

use std::time::Duration;

use serde::Deserialize;
use tracing::warn;
use url::Url;

use crate::policies::kv_events::EventConfig;

/// Default timeout for `/server_info`. Conservative for a small JSON
/// payload served by SGLang's HTTP server.
const SERVER_INFO_TIMEOUT: Duration = Duration::from_secs(2);

/// Retry budget for transient `/server_info` failures (connect/timeout/5xx).
/// 4xx + JSON-parse errors short-circuit — they're authoritative.
/// EndpointSlice can flip ready=true before the worker's HTTP server is
/// actually serving; without retry, that race lands a worker in the
/// registry with empty model_ids and chat dispatch fails with 502.
const FETCH_MAX_ATTEMPTS: u32 = 3;
const FETCH_BACKOFF_BASE: Duration = Duration::from_millis(100);

/// Resolved per-worker bootstrap state.
///
/// `served_model_name` populates the registry; `event_config` is handed
/// to `KvEventIndex::add_worker` (skipping its own fetch);
/// `disaggregation_role` lets the worker manager override the discovery
/// backend's PD classification (and fill in `WorkerSpec.bootstrap_port`
/// for prefill workers) — see `manager::register_one`.
#[derive(Debug, Clone, Default)]
pub struct ServerInfo {
    pub served_model_name: Option<String>,
    pub event_config: Option<EventConfig>,
    pub disaggregation_role: Option<DisaggregationRole>,
    /// symmetric P/D rank-0 `(owner_addr, reader_addr)` ZMQ endpoints from the
    /// `/server_info` `symmetric_pd` section. `None`
    /// on engines without the section; the router then derives them from
    /// `bootstrap_port` (owner = `tcp://host:port`, reader = `+1`).
    pub symmetric_pd_endpoints: Option<(String, String)>,
    /// KV pool size in tokens (`max_total_num_tokens`, top level or first scheduler).
    /// Also read by the token-charged decode policies with
    /// `--pd-decode-tokens-normalize-capacity`.
    pub kv_capacity_tokens: Option<u64>,
    /// The engine chunk in tokens (`ServerArgs.chunked_prefill_size`, top
    /// level); None when absent, 0, negative (chunking off) or not an
    /// integer. Read by `--symmetric-pd-gwq-auto`.
    pub chunk_tokens: Option<u64>,
    /// `/server_info` `late_d` block (late D engines); `None` when absent.
    pub late_d: Option<crate::workers::LateDWorkerInfo>,
    /// `/server_info` answered with a JSON body. `false` when every attempt
    /// failed: then nothing above that comes from `/server_info` is known
    /// (late-D choose mode does not let such a worker take traffic).
    pub server_info_read: bool,
}

/// PD classification derived from a worker's `/server_info` response.
///
/// `Some(_)` means the worker self-disclosed its role and we should trust
/// it over the discovery backend's classification. `None` (the
/// `ServerInfo::disaggregation_role` value, not a variant here) means the
/// worker didn't tell us — older SGLang, missing field, or a partial
/// response — and the backend's classification wins. See the resolution
/// table in `resolve_disaggregation_role`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisaggregationRole {
    Plain,
    Prefill {
        bootstrap_port: u16,
    },
    Decode,
    /// symmetric P/D engine (`disaggregation_mode = "symmetric"`): runs either
    /// leg. `bootstrap_port` is the base of its ZMQ side channels when
    /// advertised.
    Symmetric {
        bootstrap_port: Option<u16>,
    },
}

/// Performs the two round-trips concurrently and projects the responses into
/// `ServerInfo`. Cheap to clone — wraps a `reqwest::Client` (which is
/// internally `Arc`-backed).
#[derive(Clone)]
pub struct WorkerIntrospector {
    client: reqwest::Client,
}

impl WorkerIntrospector {
    /// Build with a private `reqwest::Client` carrying the supplied
    /// request timeout.  Production callers pass `SERVER_INFO_TIMEOUT`
    /// via `default()`; tests may pass shorter timeouts.
    pub fn new(timeout: Duration) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .expect("introspector http client builds");
        Self { client }
    }

    /// Fetch `/model_info` and `/server_info` for the worker, concurrently.
    /// Never returns an error: any failure is logged at `warn!` and yields
    /// `None` for the fields that request carried. Callers register the
    /// worker with empty model IDs and no event subscription on the
    /// failure path; future re-discovery will retry.
    ///
    /// Transient failures (network errors, 5xx) are retried up to
    /// `FETCH_MAX_ATTEMPTS` times with exponential backoff. 4xx
    /// responses and JSON-parse errors short-circuit immediately —
    /// the worker answered authoritatively, retrying won't help.
    pub async fn fetch(&self, worker_url: &str) -> ServerInfo {
        let base = worker_url.trim_end_matches('/');
        let server_info_url = format!("{base}/server_info");
        let model_info_url = format!("{base}/model_info");
        let (parsed, model_info) = tokio::join!(
            Self::fetch_with_retry::<ServerInfoBody>(&self.client, &server_info_url, worker_url),
            Self::fetch_with_retry::<ModelInfoBody>(&self.client, &model_info_url, worker_url),
        );
        // A worker that answers one endpoint and not the other still gets
        // registered with whatever did answer.
        let server_info_read = parsed.is_some();
        let parsed = parsed.unwrap_or_default();

        // `/model_info` is the effective identity; `/server_info` is the launch
        // record, kept as the fallback for workers that predate the field
        // there. An empty string is the same as absent on either.
        let non_empty = |name: String| Some(name).filter(|name| !name.is_empty());
        let served_model_name = model_info
            .and_then(|body| body.served_model_name)
            .and_then(non_empty)
            .or_else(|| parsed.served_model_name.and_then(non_empty));
        if served_model_name.is_none() {
            warn!(
                worker_url = %worker_url,
                "introspect: no `served_model_name` on /model_info or /server_info; registering worker with empty model_ids"
            );
        }

        // EAGLE-family speculative decoding ⇒ the worker hashes KV blocks over
        // token bigrams; the router must mirror that on the selection side.
        let is_bigram = crate::policies::kv_events::classify_bigram(
            parsed.speculative_algorithm.as_deref(),
            worker_url,
        );
        let event_config = parsed
            .kv_events
            .map(|block| resolve_event_config(block, worker_url, is_bigram));

        let disaggregation_role = resolve_disaggregation_role(
            parsed.disaggregation_mode.as_deref(),
            parsed.disaggregation_bootstrap_port,
            worker_url,
        );
        let symmetric_pd_endpoints = parsed
            .symmetric_pd
            .as_ref()
            .and_then(|section| symmetric_pd_endpoints_from_server_info(section, worker_url))
            .or_else(|| {
                // SGLang aggregates scheduler telemetry under internal_states.
                // Use the advertised rank-0 endpoints verbatim: the transfer
                // protocol keys owner registrations by this exact address, so
                // a hostname-derived fallback is not interchangeable with IP.
                parsed
                    .internal_states
                    .as_deref()
                    .unwrap_or_default()
                    .iter()
                    .filter_map(|state| state.symmetric_pd.as_ref())
                    .filter(|section| section.get("tp_rank").and_then(|v| v.as_u64()) == Some(0))
                    .find_map(|section| symmetric_pd_endpoints_from_server_info(section, worker_url))
            });
        // Lenient: an absent, zero or non-integer value is "not reported",
        // never a failed parse of the rest of the body.
        let capacity = |v: &Option<serde_json::Value>| {
            v.as_ref()
                .and_then(serde_json::Value::as_u64)
                .filter(|&c| c > 0)
        };
        let kv_capacity_tokens = capacity(&parsed.max_total_num_tokens).or_else(|| {
            parsed
                .internal_states
                .as_deref()
                .unwrap_or_default()
                .iter()
                .find_map(|state| capacity(&state.max_total_num_tokens))
        });

        let mut late_d = parsed
            .late_d
            .as_ref()
            .and_then(late_d_info_from_server_info);
        // A symmetric worker's `late_d` block names no link port (the engine
        // adds it on a PD prefill only); its server argument does.
        if parsed.disaggregation_mode.as_deref() == Some("symmetric") {
            if let Some(info) = late_d.as_mut().filter(|i| i.control_port.is_none()) {
                info.control_port = parsed
                    .disaggregation_late_d_control_port
                    .as_ref()
                    .and_then(serde_json::Value::as_u64)
                    .filter(|p| (1..=u64::from(u16::MAX)).contains(p))
                    .map(|p| p as u16);
            }
        }

        let chunk_tokens = capacity(&parsed.chunked_prefill_size);
        ServerInfo {
            served_model_name,
            event_config,
            disaggregation_role,
            symmetric_pd_endpoints,
            kv_capacity_tokens,
            chunk_tokens,
            late_d,
            server_info_read,
        }
    }

    /// Issue one introspection GET with bounded retry on transient errors.
    /// Returns `Some(body)` on success, `None` after exhausting retries or on
    /// an authoritative answer (4xx, unparsable JSON). The caller decides what
    /// a missing body costs — each endpoint carries different fields.
    async fn fetch_with_retry<T: serde::de::DeserializeOwned>(
        client: &reqwest::Client,
        url: &str,
        worker_url: &str,
    ) -> Option<T> {
        let mut delay = FETCH_BACKOFF_BASE;
        for attempt in 1..=FETCH_MAX_ATTEMPTS {
            match client.get(url).send().await {
                Err(e) => {
                    warn!(
                        worker_url = %worker_url,
                        url = %url,
                        attempt,
                        error = %e,
                        "introspect: request failed; will retry"
                    );
                }
                Ok(resp) if resp.status().is_server_error() => {
                    warn!(
                        worker_url = %worker_url,
                        url = %url,
                        attempt,
                        status = %resp.status(),
                        "introspect: returned 5xx; will retry"
                    );
                }
                Ok(resp) if !resp.status().is_success() => {
                    warn!(
                        worker_url = %worker_url,
                        url = %url,
                        status = %resp.status(),
                        "introspect: returned non-2xx; treating the endpoint as absent"
                    );
                    return None;
                }
                Ok(resp) => match resp.json::<T>().await {
                    Ok(body) => return Some(body),
                    Err(e) => {
                        warn!(
                            worker_url = %worker_url,
                            url = %url,
                            error = %e,
                            "introspect: JSON parse failed; treating the endpoint as absent"
                        );
                        return None;
                    }
                },
            }
            if attempt < FETCH_MAX_ATTEMPTS {
                tokio::time::sleep(delay).await;
                delay *= 2;
            }
        }
        warn!(
            worker_url = %worker_url,
            url = %url,
            attempts = FETCH_MAX_ATTEMPTS,
            "introspect: failed after retries; treating the endpoint as absent"
        );
        None
    }
}

/// Map the two `disaggregation_*` fields from `/server_info` into a
/// `DisaggregationRole`. Returns `None` when the worker hasn't told us
/// enough to be useful — the caller treats that as "defer to the
/// discovery backend's classification" instead of forcing Plain, which
/// preserves backwards compatibility with SGLang versions that predate
/// the field.
///
/// Resolution table:
///
/// | `disaggregation_mode`        | `disaggregation_bootstrap_port` | Result                              |
/// |------------------------------|----------------------------------|-------------------------------------|
/// | `None` (older SGLang)        | _any_                            | `None` — defer to backend           |
/// | `Some("null")`               | _any_                            | `Some(Plain)`                       |
/// | `Some("prefill")`            | `Some(p)`                        | `Some(Prefill { bootstrap_port: p })` |
/// | `Some("prefill")`            | `None`                           | warn + `None` — defer to backend    |
/// | `Some("decode")`             | _any_                            | `Some(Decode)`                      |
/// | `Some("symmetric")`          | `p` (optional)                   | `Some(Symmetric { bootstrap_port: p })` |
/// | `Some(other)`                | _any_                            | warn + `None`                       |
fn resolve_disaggregation_role(
    mode: Option<&str>,
    bootstrap_port: Option<u16>,
    worker_url: &str,
) -> Option<DisaggregationRole> {
    match mode {
        None => None,
        Some("null") => Some(DisaggregationRole::Plain),
        Some("prefill") => match bootstrap_port {
            Some(p) => Some(DisaggregationRole::Prefill { bootstrap_port: p }),
            None => {
                warn!(
                    worker_url = %worker_url,
                    "introspect: /server_info reports disaggregation_mode=\"prefill\" but \
                     disaggregation_bootstrap_port is missing; deferring to the discovery \
                     backend's classification"
                );
                None
            }
        },
        Some("decode") => Some(DisaggregationRole::Decode),
        Some("symmetric") => Some(DisaggregationRole::Symmetric { bootstrap_port }),
        Some(other) => {
            warn!(
                worker_url = %worker_url,
                disaggregation_mode = %other,
                "introspect: /server_info has unknown disaggregation_mode value; \
                 deferring to the discovery backend's classification"
            );
            None
        }
    }
}

impl Default for WorkerIntrospector {
    fn default() -> Self {
        Self::new(SERVER_INFO_TIMEOUT)
    }
}

/// Extract `(owner_addr, reader_addr)` from a `/server_info` `symmetric_pd`
/// section. Accepts full endpoints (`owner_addr` / `reader_addr`, the
/// engine's `symmetric_info()` shape) or bare ports (`owner_port` /
/// `reader_port` combined with `host`, defaulting to the worker URL's
/// host). `None` when neither form is present.
pub fn symmetric_pd_endpoints_from_server_info(
    section: &serde_json::Value,
    worker_url: &str,
) -> Option<(String, String)> {
    let obj = section.as_object()?;
    let addr = |k: &str| {
        obj.get(k)
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
    };
    if let (Some(owner), Some(reader)) = (addr("owner_addr"), addr("reader_addr")) {
        return Some((owner.to_string(), reader.to_string()));
    }
    let port = |k: &str| obj.get(k).and_then(|v| v.as_u64());
    let host = addr("host")
        .map(str::to_string)
        .or_else(|| Url::parse(worker_url).ok()?.host_str().map(str::to_string))?;
    let (owner, reader) = (port("owner_port")?, port("reader_port")?);
    Some((
        format!("tcp://{host}:{owner}"),
        format!("tcp://{host}:{reader}"),
    ))
}

/// The late-D view of a `/server_info` `late_d` block (`{"mode": "alloc" |
/// "choose", "kv_ip": str | null, "control_port": int}`, the port on a
/// prefill only). `None` for a block that is not an object or has no mode;
/// a malformed port or KV IP is left out.
pub fn late_d_info_from_server_info(
    section: &serde_json::Value,
) -> Option<crate::workers::LateDWorkerInfo> {
    let obj = section.as_object()?;
    let text = |k: &str| {
        obj.get(k)
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let mode = text("mode")?;
    let control_port = obj
        .get("control_port")
        .and_then(|v| v.as_u64())
        .filter(|p| (1..=u64::from(u16::MAX)).contains(p))
        .map(|p| p as u16);
    Some(crate::workers::LateDWorkerInfo {
        mode,
        control_port,
        kv_ip: text("kv_ip"),
    })
}

/// Substitute a wildcard bind host (`*`, `0.0.0.0`, `::`, `[::]`) with
/// the host parsed from the worker URL — the gateway has to connect to
/// a routable address.  An unparsable worker URL leaves the host
/// unchanged: the subsequent ZMQ connect will fail visibly with the
/// wildcard literal, which is the same observable failure mode that
/// would occur today if the bind/connect were skipped.
pub(crate) fn resolve_event_config(
    block: KvEventsBlock,
    worker_url: &str,
    is_bigram: bool,
) -> EventConfig {
    let host = if matches!(
        block.endpoint_host.as_str(),
        "*" | "0.0.0.0" | "::" | "[::]"
    ) {
        match Url::parse(worker_url)
            .ok()
            .and_then(|u| u.host_str().map(|s| s.to_owned()))
        {
            Some(h) => h,
            None => {
                warn!(
                    worker_url = %worker_url,
                    "introspect: cannot parse worker_url for wildcard substitution; keeping advertised host"
                );
                block.endpoint_host
            }
        }
    } else {
        block.endpoint_host
    };
    EventConfig {
        host,
        port_base: block.endpoint_port_base,
        topic: block.topic,
        block_size: block.block_size,
        dp_size: block.dp_size,
        is_bigram,
    }
}

/// Projection of `/model_info` used by the introspector: the identity the
/// worker currently serves under, which a weight update moves.
#[derive(Debug, Default, Deserialize)]
struct ModelInfoBody {
    #[serde(default)]
    served_model_name: Option<String>,
}

/// Projection of `/server_info` used by the introspector. Every field is
/// `#[serde(default)]` so a worker that exposes only some of them still
/// deserialises; downstream callers handle `None` as "absent".
#[derive(Debug, Default, Deserialize)]
struct ServerInfoBody {
    /// The launch record's value, and the fallback for a worker whose
    /// `/model_info` predates the field.
    #[serde(default)]
    served_model_name: Option<String>,
    #[serde(default)]
    kv_events: Option<KvEventsBlock>,
    /// Top-level `speculative_algorithm`. EAGLE-family values
    /// (EAGLE / EAGLE3 / FROZEN_KV_MTP) ⇒ the worker hashes KV blocks over
    /// token bigrams. Absent on workers without speculative decoding.
    #[serde(default)]
    speculative_algorithm: Option<String>,
    /// Carries the value of `ServerArgs.disaggregation_mode`
    /// (`"null"` | `"prefill"` | `"decode"`). Absent on older SGLang
    /// versions that predate the field.
    #[serde(default)]
    disaggregation_mode: Option<String>,
    /// `ServerArgs.disaggregation_bootstrap_port`. Meaningful only when
    /// `disaggregation_mode == "prefill"`; the prefill server's
    /// bootstrap server binds to exactly this port (no internal offset).
    #[serde(default)]
    disaggregation_bootstrap_port: Option<u16>,
    /// symmetric P/D section (`scheduler_mixin.symmetric_info()`): rank-0
    /// `owner_addr` / `reader_addr` plus telemetry the router ignores.
    #[serde(default)]
    symmetric_pd: Option<serde_json::Value>,
    /// Scheduler KV capacity in tokens (from `scheduler_info`). Kept as a raw
    /// value so an unexpected type never fails the whole body.
    #[serde(default)]
    max_total_num_tokens: Option<serde_json::Value>,
    /// `ServerArgs.chunked_prefill_size` (the engine chunk; -1 = off). Raw
    /// for leniency.
    #[serde(default)]
    chunked_prefill_size: Option<serde_json::Value>,
    #[serde(default)]
    internal_states: Option<Vec<SchedulerState>>,
    /// Late D block (`describe_late_d`): mode, KV IP and, on a prefill, the
    /// router link port; `null` when late D is off. Raw for leniency.
    #[serde(default)]
    late_d: Option<serde_json::Value>,
    /// `ServerArgs.disaggregation_late_d_control_port`: the late-D link port
    /// of a symmetric worker, whose `late_d` block names none. Raw for
    /// leniency.
    #[serde(default)]
    disaggregation_late_d_control_port: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct SchedulerState {
    #[serde(default)]
    symmetric_pd: Option<serde_json::Value>,
    #[serde(default)]
    max_total_num_tokens: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct KvEventsBlock {
    // Forward-compatibility: the only publisher implementation
    // supported on the gateway side is ZMQ. Keeping the field optional
    // means a future SGLang that adds a non-ZMQ publisher string won't
    // fail deserialize; the resulting subscriber will still try to open
    // a ZMQ connection and fail visibly.
    #[allow(dead_code)]
    #[serde(default)]
    publisher: Option<String>,
    pub endpoint_host: String,
    pub endpoint_port_base: u16,
    #[serde(default)]
    pub topic: String,
    pub block_size: u32,
    pub dp_size: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{routing::get, Json, Router};
    use serde_json::{json, Value};
    use std::sync::Arc;
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    /// A worker that serves `/server_info` only: an SGLang that predates
    /// `served_model_name` on `/model_info`.
    async fn spawn_fake_worker(body: Value) -> (String, oneshot::Sender<()>) {
        spawn_fake_worker_with_model_info(body, None).await
    }

    async fn spawn_fake_worker_with_model_info(
        server_info: Value,
        model_info: Option<Value>,
    ) -> (String, oneshot::Sender<()>) {
        let body = Arc::new(server_info);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let mut app = Router::new().route(
            "/server_info",
            get(move || {
                let body = body.clone();
                async move { Json((*body).clone()) }
            }),
        );
        if let Some(model_info) = model_info {
            let model_info = Arc::new(model_info);
            app = app.route(
                "/model_info",
                get(move || {
                    let model_info = model_info.clone();
                    async move { Json((*model_info).clone()) }
                }),
            );
        }
        let (tx, rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = rx.await;
                })
                .await;
        });
        (format!("http://127.0.0.1:{port}"), tx)
    }

    fn fast_introspector() -> WorkerIntrospector {
        WorkerIntrospector::new(Duration::from_millis(500))
    }

    /// The PRIMARY `/server_info` path (the introspector, not the discovery.rs
    /// fallback) must flag `is_bigram` for an EAGLE worker so the policy picks
    /// the bigram hasher. Regression guard for the duplicated parse + the
    /// `resolve_event_config(.., is_bigram)` threading.
    #[tokio::test]
    async fn fetch_sets_is_bigram_for_eagle_worker() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "speculative_algorithm": "EAGLE",
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "*",
                "endpoint_port_base": 5557,
                "topic": "",
                "block_size": 64,
                "dp_size": 1,
            }
        }))
        .await;
        let cfg = fast_introspector()
            .fetch(&url)
            .await
            .event_config
            .expect("kv_events present");
        assert!(
            cfg.is_bigram,
            "EAGLE worker via the introspector must set is_bigram"
        );
    }

    /// A non-speculative worker (no `speculative_algorithm`) must NOT be bigram.
    #[tokio::test]
    async fn fetch_no_bigram_without_speculative_algorithm() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "*",
                "endpoint_port_base": 5557,
                "topic": "",
                "block_size": 64,
                "dp_size": 1,
            }
        }))
        .await;
        let cfg = fast_introspector()
            .fetch(&url)
            .await
            .event_config
            .expect("kv_events present");
        assert!(!cfg.is_bigram, "non-speculative worker must not be bigram");
    }

    #[tokio::test]
    async fn fetch_returns_both_served_model_name_and_event_config() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "Qwen3-0.6B",
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "10.1.2.3",
                "endpoint_port_base": 6000,
                "topic": "kv",
                "block_size": 64,
                "dp_size": 2,
            }
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert_eq!(got.served_model_name.as_deref(), Some("Qwen3-0.6B"));
        let cfg = got.event_config.expect("kv_events present");
        assert_eq!(cfg.host, "10.1.2.3");
        assert_eq!(cfg.port_base, 6000);
        assert_eq!(cfg.topic, "kv");
        assert_eq!(cfg.block_size, 64);
        assert_eq!(cfg.dp_size, 2);
    }

    #[tokio::test]
    async fn fetch_substitutes_wildcard_host() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "*",
                "endpoint_port_base": 5557,
                "topic": "kv",
                "block_size": 64,
                "dp_size": 1,
            }
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        let cfg = got.event_config.expect("kv_events present");
        assert_eq!(cfg.host, "127.0.0.1");
    }

    #[tokio::test]
    async fn fetch_returns_empty_on_connection_refused() {
        // Port 1 is reserved; bind a temp listener to reserve a free
        // port then drop it so the connect fails fast.
        let temp = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = temp.local_addr().unwrap().port();
        drop(temp);
        let url = format!("http://127.0.0.1:{port}");
        let got = fast_introspector().fetch(&url).await;
        assert!(
            got.served_model_name.is_none(),
            "served_model_name must be None on connection refused"
        );
        assert!(
            got.event_config.is_none(),
            "event_config must be None on connection refused"
        );
    }

    #[tokio::test]
    async fn fetch_prefers_model_info_over_server_info_for_the_name() {
        let (url, _shutdown) = spawn_fake_worker_with_model_info(
            json!({"served_model_name": "at-launch"}),
            Some(json!({"served_model_name": "after-weight-update"})),
        )
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert_eq!(
            got.served_model_name.as_deref(),
            Some("after-weight-update")
        );
    }

    #[tokio::test]
    async fn fetch_falls_back_to_server_info_when_model_info_lacks_the_name() {
        let (url, _shutdown) = spawn_fake_worker_with_model_info(
            json!({"served_model_name": "at-launch"}),
            Some(json!({"model_path": "/models/m"})),
        )
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert_eq!(got.served_model_name.as_deref(), Some("at-launch"));
    }

    /// An empty name on either surface is the same as absent.
    #[tokio::test]
    async fn fetch_treats_an_empty_name_as_absent_on_both() {
        let (url, _shutdown) = spawn_fake_worker_with_model_info(
            json!({"served_model_name": "at-launch"}),
            Some(json!({"served_model_name": ""})),
        )
        .await;
        assert_eq!(
            fast_introspector()
                .fetch(&url)
                .await
                .served_model_name
                .as_deref(),
            Some("at-launch"),
        );

        let (url, _shutdown) =
            spawn_fake_worker_with_model_info(json!({"served_model_name": ""}), None).await;
        assert!(fast_introspector()
            .fetch(&url)
            .await
            .served_model_name
            .is_none());
    }

    #[tokio::test]
    async fn fetch_keeps_the_name_when_server_info_is_absent() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = Router::new().route(
            "/model_info",
            get(|| async { Json(json!({"served_model_name": "m"})) }),
        );
        let (tx, rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = rx.await;
                })
                .await;
        });
        let _shutdown = tx;
        let got = fast_introspector()
            .fetch(&format!("http://127.0.0.1:{port}"))
            .await;
        assert_eq!(got.served_model_name.as_deref(), Some("m"));
        assert!(got.event_config.is_none());
        assert!(got.disaggregation_role.is_none());
        assert!(!got.server_info_read, "no /server_info answer");
        assert!(got.late_d.is_none());
    }

    #[tokio::test]
    async fn fetch_reads_the_late_d_block() {
        use crate::workers::LateDWorkerInfo;
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "late_d": {"mode": "choose", "kv_ip": "10.0.0.7", "control_port": 31000},
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert!(got.server_info_read);
        assert_eq!(
            got.late_d,
            Some(LateDWorkerInfo {
                mode: "choose".into(),
                control_port: Some(31000),
                kv_ip: Some("10.0.0.7".into()),
            })
        );
        // Late D off: a null block (or none at all).
        for body in [
            json!({"served_model_name": "m", "late_d": null}),
            json!({"served_model_name": "m"}),
        ] {
            let (url, _shutdown) = spawn_fake_worker(body).await;
            let got = fast_introspector().fetch(&url).await;
            assert!(got.server_info_read);
            assert_eq!(got.late_d, None);
        }
    }

    /// A symmetric worker's link port comes from its server argument; a
    /// decode worker's does not (its block keeps no port).
    #[tokio::test]
    async fn fetch_reads_the_symmetric_late_d_control_port() {
        use crate::workers::LateDWorkerInfo;
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "disaggregation_mode": "symmetric",
            "disaggregation_bootstrap_port": 18998,
            "disaggregation_late_d_control_port": 21776,
            "late_d": {"mode": "choose", "kv_ip": "10.0.0.7"},
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert_eq!(
            got.late_d,
            Some(LateDWorkerInfo {
                mode: "choose".into(),
                control_port: Some(21776),
                kv_ip: Some("10.0.0.7".into()),
            })
        );
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "disaggregation_mode": "decode",
            "disaggregation_late_d_control_port": 21776,
            "late_d": {"mode": "choose", "kv_ip": "10.0.0.7"},
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert_eq!(got.late_d.unwrap().control_port, None);
    }

    #[test]
    fn late_d_block_parse_is_lenient() {
        use crate::workers::LateDWorkerInfo;
        assert_eq!(
            late_d_info_from_server_info(&json!({"mode": "alloc", "kv_ip": null})),
            Some(LateDWorkerInfo {
                mode: "alloc".into(),
                control_port: None,
                kv_ip: None,
            })
        );
        for port in [
            json!(70000),
            json!("31000"),
            json!(0),
            json!(-1),
            json!(null),
        ] {
            let info =
                late_d_info_from_server_info(&json!({"mode": "choose", "control_port": port}))
                    .unwrap();
            assert_eq!(info.control_port, None, "{port}");
        }
        let info = late_d_info_from_server_info(&json!({"mode": "choose", "kv_ip": ""})).unwrap();
        assert_eq!(info.kv_ip, None);
        assert_eq!(late_d_info_from_server_info(&json!({"kv_ip": "x"})), None);
        assert_eq!(late_d_info_from_server_info(&json!({"mode": ""})), None);
        assert_eq!(late_d_info_from_server_info(&json!("choose")), None);
    }

    #[tokio::test]
    async fn fetch_only_served_model_name_when_kv_events_absent() {
        let (url, _shutdown) = spawn_fake_worker(json!({"served_model_name": "m"})).await;
        let got = fast_introspector().fetch(&url).await;
        assert_eq!(got.served_model_name.as_deref(), Some("m"));
        assert!(got.event_config.is_none());
    }

    #[tokio::test]
    async fn fetch_only_event_config_when_served_model_name_absent() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "127.0.0.1",
                "endpoint_port_base": 5557,
                "topic": "",
                "block_size": 64,
                "dp_size": 1,
            }
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert!(got.served_model_name.is_none());
        let cfg = got.event_config.expect("kv_events present");
        assert_eq!(cfg.port_base, 5557);
    }

    /// `disaggregation_mode=prefill` + a bootstrap port → manager should
    /// see the worker as a prefill peer with the supplied port. This is
    /// the happy path that lets PD-on-K8s skip pod annotations entirely.
    #[tokio::test]
    async fn fetch_resolves_prefill_role_with_bootstrap_port() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "disaggregation_mode": "prefill",
            "disaggregation_bootstrap_port": 8998,
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert_eq!(
            got.disaggregation_role,
            Some(DisaggregationRole::Prefill {
                bootstrap_port: 8998
            }),
        );
    }

    /// `disaggregation_mode=decode` → role is Decode regardless of any
    /// bootstrap-port field value (decode workers don't bind one).
    #[tokio::test]
    async fn fetch_resolves_decode_role() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "disaggregation_mode": "decode",
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert_eq!(got.disaggregation_role, Some(DisaggregationRole::Decode));
    }

    /// `max_total_num_tokens` (scheduler_info, top level) becomes the KV
    /// capacity; absent, zero or a non-integer value is "not reported" and
    /// never fails the rest of the body.
    #[tokio::test]
    async fn fetch_reads_kv_capacity_leniently() {
        let (url, _s1) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "disaggregation_mode": "decode",
            "max_total_num_tokens": 1_234_560,
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert_eq!(got.kv_capacity_tokens, Some(1_234_560));
        assert_eq!(got.disaggregation_role, Some(DisaggregationRole::Decode));
        for bad in [json!("lots"), json!(0), json!(1.5), json!(null)] {
            let (url, _s) = spawn_fake_worker(json!({
                "served_model_name": "m",
                "disaggregation_mode": "decode",
                "max_total_num_tokens": bad,
            }))
            .await;
            let got = fast_introspector().fetch(&url).await;
            assert_eq!(got.kv_capacity_tokens, None);
            assert_eq!(got.served_model_name.as_deref(), Some("m"));
            assert_eq!(got.disaggregation_role, Some(DisaggregationRole::Decode));
        }
    }

    /// `disaggregation_mode="null"` is SGLang's explicit "not
    /// disaggregated" value — we trust it and force the worker to Plain
    /// even if the discovery backend mistakenly classified it as
    /// prefill/decode.
    #[tokio::test]
    async fn fetch_resolves_plain_role_when_mode_is_null() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "disaggregation_mode": "null",
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert_eq!(got.disaggregation_role, Some(DisaggregationRole::Plain));
    }

    /// Partial data (`prefill` mode with no bootstrap port) returns
    /// `None` so the manager keeps the discovery backend's
    /// classification. The alternative — forcing Plain — would silently
    /// demote a misconfigured prefill worker to plain dispatch.
    #[tokio::test]
    async fn fetch_defers_to_backend_when_prefill_mode_lacks_bootstrap_port() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "disaggregation_mode": "prefill",
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert!(
            got.disaggregation_role.is_none(),
            "prefill with no bootstrap port must defer to backend, got {:?}",
            got.disaggregation_role,
        );
    }

    /// A symmetric P/D engine reports `disaggregation_mode="symmetric"` and its
    /// rank-0 side-channel endpoints; both must reach the manager.
    #[tokio::test]
    async fn fetch_resolves_symmetric_role_and_symmetric_pd_endpoints() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "disaggregation_mode": "symmetric",
            "disaggregation_bootstrap_port": 8998,
            "symmetric_pd": {
                "mode": "symmetric",
                "owner_addr": "tcp://10.0.0.5:8998",
                "reader_addr": "tcp://10.0.0.5:8999",
                "export_table": {"entries": 0},
            }
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert_eq!(
            got.disaggregation_role,
            Some(DisaggregationRole::Symmetric {
                bootstrap_port: Some(8998)
            }),
        );
        assert_eq!(
            got.symmetric_pd_endpoints,
            Some((
                "tcp://10.0.0.5:8998".to_string(),
                "tcp://10.0.0.5:8999".to_string()
            )),
        );
    }

    #[tokio::test]
    async fn fetch_uses_advertised_rank_zero_endpoints_from_internal_states() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "disaggregation_mode": "symmetric",
            "disaggregation_bootstrap_port": 8998,
            "symmetric_pd": null,
            "internal_states": [
                {"symmetric_pd": {"tp_rank": 1, "owner_addr": "tcp://10.0.0.5:9000", "reader_addr": "tcp://10.0.0.5:9001"}},
                {"max_total_num_tokens": 1000},
                {"symmetric_pd": {"mode": "symmetric", "tp_rank": 0,
                    "owner_addr": "tcp://10.0.0.5:8998", "reader_addr": "tcp://10.0.0.5:8999"}}
            ]
        })).await;
        let got = fast_introspector().fetch(&url).await;
        assert_eq!(
            got.symmetric_pd_endpoints,
            Some(("tcp://10.0.0.5:8998".into(), "tcp://10.0.0.5:8999".into()))
        );
    }

    #[tokio::test]
    async fn fetch_does_not_use_another_rank_as_rank_zero() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "disaggregation_mode": "symmetric",
            "internal_states": [{"symmetric_pd": {"tp_rank": 1,
                "owner_addr": "tcp://10.0.0.5:9000", "reader_addr": "tcp://10.0.0.5:9001"}}]
        }))
        .await;
        assert!(fast_introspector()
            .fetch(&url)
            .await
            .symmetric_pd_endpoints
            .is_none());
    }

    #[tokio::test]
    async fn fetch_preserves_top_level_endpoint_precedence() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "symmetric_pd": {"owner_addr": "tcp://10.0.0.5:8998", "reader_addr": "tcp://10.0.0.5:8999"},
            "internal_states": [{"symmetric_pd": {"tp_rank": 0,
                "owner_addr": "tcp://10.0.0.6:9000", "reader_addr": "tcp://10.0.0.6:9001"}}]
        }))
        .await;
        assert_eq!(
            fast_introspector().fetch(&url).await.symmetric_pd_endpoints,
            Some(("tcp://10.0.0.5:8998".into(), "tcp://10.0.0.5:8999".into()))
        );
    }

    #[test]
    fn symmetric_pd_endpoints_accept_ports_with_host_fallback() {
        let section = json!({"owner_port": 7000, "reader_port": 7001});
        assert_eq!(
            symmetric_pd_endpoints_from_server_info(&section, "http://10.0.0.9:30000"),
            Some(("tcp://10.0.0.9:7000".into(), "tcp://10.0.0.9:7001".into())),
        );
        let section = json!({"owner_port": 7000, "reader_port": 7001, "host": "node-a"});
        assert_eq!(
            symmetric_pd_endpoints_from_server_info(&section, "http://10.0.0.9:30000"),
            Some(("tcp://node-a:7000".into(), "tcp://node-a:7001".into())),
        );
        assert_eq!(
            symmetric_pd_endpoints_from_server_info(&json!({"mode": "symmetric"}), "http://x"),
            None
        );
    }

    /// Older SGLang doesn't expose `disaggregation_mode`. The
    /// introspector must not invent a classification — the discovery
    /// backend's seed (K8s labels, static-urls Plain default) still
    /// drives mode for these workers.
    #[tokio::test]
    async fn fetch_defers_to_backend_when_mode_field_is_absent() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert!(got.disaggregation_role.is_none());
    }

    /// Unknown `disaggregation_mode` value (future SGLang adds a new
    /// disaggregation flavor, network garbled the field, etc.) → defer
    /// to backend rather than guessing.
    #[tokio::test]
    async fn fetch_defers_to_backend_when_mode_is_unrecognized() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "disaggregation_mode": "encode_only",
            "disaggregation_bootstrap_port": 8998,
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert!(got.disaggregation_role.is_none());
    }
}
