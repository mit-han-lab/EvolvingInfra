// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! KV pull-hint injection — external-indexer mode.
//!
//! Queries a standalone Dynamo KV indexer for request-prefix observations,
//! reusable during placement and after a worker is selected. When a peer match
//! passes configured gates, the caller can add the returned `kv_pull_hint`
//! object to the forwarded body. The engine's storage backend then RDMA-pulls the
//! matched pages from the owner's pinned host pool instead of recomputing
//! them (the owner-side LOOKUP remains the truth moment; everything here is
//! advisory and fail-open).
//!
//! PD deployments: the hint rides on the PREFILL request only. Decode
//! workers receive their KV through the configured P/D transfer protocol and
//! must never start a peer prefetch of their own through this hint.
//!
//! Salted traffic: the request's top-level `cache_salt` is forwarded
//! verbatim in the indexer query (the indexer namespaces matches by salt).
//! The hint payload itself carries no salt — the reader engine derives its
//! servicer LOOKUP keys from its own storage hash chain, which is
//! salt-independent.
//!
//! The wire format originates from sgl-model-gateway's
//! `routers/http/kv_pull_hint.rs`. This module retains the external-indexer
//! boundary; colocating the indexer process with the router does not make
//! cache ownership or physical KV transfer part of this client.
//!
//! DP attention: a worker whose map entries name two or more DP ranks
//! (`dp_rank`) is a set of prefill slots, one per rank, each with its own
//! directory instance and its own servicer. Hits, pull sources and pull
//! targets are then per slot ([`slot_key`]); a worker without ranks in the map
//! is one slot, keyed by its URL.

use std::collections::HashMap;
use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};

/// One `--kv-worker-map` entry: indexer instance id → worker base URL, its
/// DP rank and that rank's peer-KV servicer address (`host:port`; an engine
/// with attention TP inside the rank serves TP shard i at this port + i).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct KvWorkerMapEntry {
    pub id: u64,
    pub url: String,
    /// Peer-KV servicer address; only the pull-hint path needs it, so the
    /// symmetric_pd placer (which maps hits to worker URLs only) may omit it.
    /// An instance without one is never a pull source.
    #[serde(default)]
    pub servicer: String,
    /// The DP rank of this instance on a DP-attention worker. A worker whose
    /// entries name two or more ranks is placed per rank.
    #[serde(default)]
    pub dp_rank: Option<u32>,
}

/// Separator of a slot key: `<worker url>#dp<rank>`.
const SLOT_RANK_SEPARATOR: &str = "#dp";

/// The key of one prefill slot: the worker URL for a worker placed as a
/// whole, `<url>#dp<rank>` for DP rank `rank` of a worker placed per rank.
/// The key names the slot in the cost function's work book, the trace and
/// the logs; requests go to the URL, with the rank as `routed_dp_rank`.
pub fn slot_key(url: &str, dp_rank: Option<u32>) -> String {
    match dp_rank {
        Some(rank) => format!("{url}{SLOT_RANK_SEPARATOR}{rank}"),
        None => url.to_string(),
    }
}

/// The worker URL and DP rank of a slot key (see [`slot_key`]); a plain URL
/// has no rank.
pub fn split_slot(key: &str) -> (&str, Option<u32>) {
    key.rsplit_once(SLOT_RANK_SEPARATOR)
        .and_then(|(url, rank)| rank.parse::<u32>().ok().map(|rank| (url, Some(rank))))
        .unwrap_or((key, None))
}

/// One directory instance: its worker, its DP rank when the worker is placed
/// per rank, and its servicer.
#[derive(Debug, Clone)]
struct Instance {
    url: String,
    dp_rank: Option<u32>,
    servicer: String,
}

/// The workers placed per rank (URL without a trailing `/`): those whose map
/// entries name two or more distinct DP ranks, with their ranks ascending.
pub(crate) fn ranks_by_url(map: &[KvWorkerMapEntry]) -> HashMap<String, Vec<u32>> {
    let mut ranks: HashMap<String, Vec<u32>> = HashMap::new();
    for entry in map {
        if let Some(rank) = entry.dp_rank {
            ranks
                .entry(entry.url.trim_end_matches('/').to_string())
                .or_default()
                .push(rank);
        }
    }
    ranks.retain(|_, list| {
        list.sort_unstable();
        list.dedup();
        list.len() > 1
    });
    ranks
}

/// Parsed configuration for the injector (see CLI flags in `config::cli`).
#[derive(Debug, Clone)]
pub struct KvHintConfig {
    pub indexer_url: String,
    pub min_tokens: u32,
    pub min_incremental_tokens: u32,
    pub local_prefix_fallback: bool,
    pub indexer_timeout_ms: u64,
    pub worker_map: Vec<KvWorkerMapEntry>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct HintCandidate {
    pub worker_url: String,
    /// The holder's DP rank on a worker placed per rank; None otherwise.
    pub dp_rank: Option<u32>,
    pub servicer_addr: String,
    pub expected_hit_tokens: u32,
}

impl HintCandidate {
    /// The holder's slot key ([`slot_key`]).
    pub fn slot(&self) -> String {
        slot_key(&self.worker_url, self.dp_rank)
    }

    /// Whether this holder is the target `selected`: the same slot, or, for a
    /// plain URL of a worker placed per rank, any rank of that worker.
    fn is_target(&self, selected: &str) -> bool {
        match split_slot(selected) {
            (url, Some(rank)) => self.worker_url == url && self.dp_rank == Some(rank),
            (url, None) => self.worker_url == url,
        }
    }
}

/// One indexer answer, reused across retry attempts of the same request.
#[derive(Debug, Clone)]
pub struct IndexerAnswer {
    /// Sorted by `expected_hit_tokens`, descending.
    pub candidates: Vec<HintCandidate>,
    /// Residency is distinct from CPU availability; absent observations stay
    /// absent. Keyed by slot ([`slot_key`]).
    pub observed_gpu_hits: HashMap<String, u32>,
    pub indexer_ms: f64,
}

#[derive(Debug)]
pub struct KvHintInjector {
    client: reqwest::Client,
    query_url: String,
    min_tokens: u32,
    min_incremental_tokens: u32,
    local_prefix_fallback: bool,
    timeout: Duration,
    instances: HashMap<u64, Instance>,
    /// The DP ranks of each worker placed per rank (URL without a trailing
    /// `/`), ascending.
    ranks_by_url: HashMap<String, Vec<u32>>,
}

impl KvHintInjector {
    /// Infallible by design: `--kv-worker-map` shape and non-emptiness are
    /// validated at CLI-parse time, and the dedicated reqwest client build
    /// cannot fail for this configuration (no TLS/proxy options involved).
    /// Configure a dedicated external-indexer client and worker-address lookup.
    ///
    /// # Inputs
    /// `cfg` supplies indexer URL, query timeout in milliseconds, minimum
    /// prefix/incremental hits in tokens, fallback mode and instance-to-worker
    /// URL/servicer mappings. CLI validation is expected to run first.
    ///
    /// # Outputs
    /// Returns an injector with connection pooling disabled and a two-second
    /// connect timeout; each query also uses the configured request timeout.
    /// Does not query the indexer. Panics if HTTP-client construction fails.
    pub fn new(cfg: &KvHintConfig) -> Self {
        // Dedicated client with connection pooling disabled: the indexer's
        // HTTP server closes idle keep-alive connections sooner than a
        // pooled client's idle timeout, and the first query after an idle
        // gap then dies on the half-closed socket ("error sending
        // request"). A fresh connect costs sub-ms on the pod network,
        // negligible against the query itself.
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(0)
            .connect_timeout(Duration::from_secs(2))
            .build()
            .expect("build kv-hint reqwest client");
        let ranks_by_url = ranks_by_url(&cfg.worker_map);
        let instances = cfg
            .worker_map
            .iter()
            .map(|entry| {
                // A rank counts only on a worker placed per rank.
                let per_rank = ranks_by_url.contains_key(entry.url.trim_end_matches('/'));
                let instance = Instance {
                    url: entry.url.clone(),
                    dp_rank: entry.dp_rank.filter(|_| per_rank),
                    servicer: entry.servicer.clone(),
                };
                (entry.id, instance)
            })
            .collect();
        Self {
            client,
            query_url: format!("{}/query", cfg.indexer_url.trim_end_matches('/')),
            min_tokens: cfg.min_tokens,
            min_incremental_tokens: cfg.min_incremental_tokens,
            local_prefix_fallback: cfg.local_prefix_fallback,
            timeout: Duration::from_millis(cfg.indexer_timeout_ms),
            instances,
            ranks_by_url,
        }
    }

    /// Worker URL of an indexer instance id (`--kv-worker-map`).
    /// Resolve an external indexer instance ID into the runtime worker URL.
    ///
    /// # Inputs
    /// `self` owns the configured mapping; `id` is the indexer instance ID.
    ///
    /// # Outputs
    /// Returns a borrowed worker URL when mapped, otherwise `None`. Performs
    /// no discovery lookup or network request.
    pub fn worker_url_for(&self, id: u64) -> Option<&str> {
        self.instances.get(&id).map(|i| i.url.as_str())
    }

    /// The prefill slots of a worker: one per DP rank (ascending) for a worker
    /// whose map entries name two or more ranks, else the worker itself
    /// (`[None]`). `url` may end with `/`.
    pub fn slot_ranks(&self, url: &str) -> Vec<Option<u32>> {
        match self.ranks_by_url.get(url.trim_end_matches('/')) {
            Some(ranks) => ranks.iter().copied().map(Some).collect(),
            None => vec![None],
        }
    }

    /// One indexer query per request; fail-open on any error. The answer is
    /// computed once and can be re-scored against whichever worker each
    /// retry attempt selects.
    /// Fetch reusable prefix observations for one tokenized request.
    ///
    /// # Inputs
    /// `self` supplies indexer transport/mappings; `token_ids` contains input
    /// tokens; `cache_salt` identifies an optional namespace; `model_name` scopes
    /// the query to the serving model.
    ///
    /// # Outputs
    /// Returns candidates sorted by decreasing max(CPU hit, GPU hit), explicit
    /// GPU-residency observations and query latency, or `None` on unavailable/
    /// invalid transport results. Lookup is advisory; it does not transfer KV,
    /// and `query_indexer` may retry a send failure once.
    pub async fn prepare(
        &self,
        token_ids: &[u32],
        cache_salt: Option<&str>,
        model_name: &str,
    ) -> Option<IndexerAnswer> {
        let (value, indexer_ms) = self
            .query_indexer(Some(token_ids), None, cache_salt, Some(model_name))
            .await?;
        let mut candidates = self.parse_candidates(&value);
        candidates.sort_by(|a, b| b.expected_hit_tokens.cmp(&a.expected_hit_tokens));
        Some(IndexerAnswer {
            candidates,
            observed_gpu_hits: self.parse_gpu_observations(&value),
            indexer_ms,
        })
    }

    /// Build the answer from a reply the caller already fetched (symmetric_pd
    /// queries the indexer once for placement and reuses the reply for the
    /// pull hint), same sorting as [`Self::prepare`].
    /// Reuse an existing indexer reply for candidate and residency decisions.
    ///
    /// # Inputs
    /// `self` supplies worker mappings; `reply` is indexer JSON; `indexer_ms`
    /// is the caller-recorded query latency in milliseconds.
    ///
    /// # Outputs
    /// Returns a sorted `IndexerAnswer` without another query. Malformed or
    /// unmapped instance entries are ignored; absent GPU observations stay absent.
    pub fn answer_from_reply(&self, reply: &Value, indexer_ms: f64) -> IndexerAnswer {
        let mut candidates = self.parse_candidates(reply);
        candidates.sort_by(|a, b| b.expected_hit_tokens.cmp(&a.expected_hit_tokens));
        IndexerAnswer {
            candidates,
            observed_gpu_hits: self.parse_gpu_observations(reply),
            indexer_ms,
        }
    }

    /// Raw `POST {indexer}/query`: `token_ids` when present, else
    /// `block_hashes` (page hashes an ingress already holds; the indexer
    /// contract, fail-open when unsupported). Returns the reply and the
    /// query latency in ms; `None` on any error or when both keys are empty.
    /// Issue the advisory token-or-hash query understood by the external indexer.
    ///
    /// # Inputs
    /// `self` supplies the endpoint and timeout; nonempty `token_ids` take
    /// precedence over `block_hashes`. Optional nonempty `cache_salt` and
    /// `model_name` are copied into the query payload.
    ///
    /// # Outputs
    /// Returns parsed JSON and elapsed milliseconds through receiving response
    /// headers, including a possible single retry after a send error. JSON body
    /// decoding occurs after that timing sample. Returns `None` for empty input,
    /// transport/status/JSON failures; failures are logged and no KV is moved.
    pub async fn query_indexer(
        &self,
        token_ids: Option<&[u32]>,
        block_hashes: Option<&[u64]>,
        cache_salt: Option<&str>,
        model_name: Option<&str>,
    ) -> Option<(Value, f64)> {
        let mut payload = match (token_ids, block_hashes) {
            (Some(t), _) if !t.is_empty() => json!({ "token_ids": t }),
            (_, Some(h)) if !h.is_empty() => json!({ "block_hashes": h }),
            _ => return None,
        };
        if let Some(model) = model_name.filter(|m| !m.is_empty()) {
            payload["model_name"] = Value::String(model.to_string());
        }
        if let Some(salt) = cache_salt.filter(|s| !s.is_empty()) {
            payload["cache_salt"] = Value::String(salt.to_string());
        }
        let start = std::time::Instant::now();
        let mut resp = self
            .client
            .post(&self.query_url)
            .timeout(self.timeout)
            .json(&payload)
            .send()
            .await;
        if resp.is_err() {
            // One immediate retry on a fresh connection: /query is
            // semantically idempotent and a send error is almost always a
            // transient connection-level failure.
            resp = self
                .client
                .post(&self.query_url)
                .timeout(self.timeout)
                .json(&payload)
                .send()
                .await;
        }
        let indexer_ms = start.elapsed().as_secs_f64() * 1e3;
        let value: Value = match resp {
            Ok(r) if r.status().is_success() => match r.json().await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(error = %e, "kv_hint: indexer reply decode failed; fail-open");
                    return None;
                }
            },
            Ok(r) => {
                tracing::warn!(status = %r.status(), "kv_hint: indexer query failed; fail-open");
                return None;
            }
            Err(e) => {
                tracing::warn!(error = %e, "kv_hint: indexer query failed; fail-open");
                return None;
            }
        };
        Some((value, indexer_ms))
    }

    /// Parse the indexer `/query` reply's per-instance summaries.
    ///
    /// Reply shape (dynamo standalone indexer): `{"instances":
    /// {"<worker_id>": {"gpu": t, "cpu": t, ...}}}` — counts in TOKENS. The
    /// servable tier for peer pulls is the owner's pinned host pool, so
    /// `expected_hit_tokens = max(cpu, gpu)`. CPU copies can outlive GPU
    /// residency; preserve that distinction separately for local routing.
    /// The owner's LOOKUP remains the truth moment for a peer pull.
    /// Map positive CPU/GPU prefix observations into potential peer owners.
    ///
    /// # Inputs
    /// `self` supplies instance-to-worker and servicer mappings; `reply`
    /// contains per-instance `cpu`/`gpu` prefix lengths measured in tokens.
    ///
    /// # Outputs
    /// Returns unsorted candidates using max(CPU, GPU), never their sum.
    /// Unknown IDs, unmapped instances and zero matches are skipped; missing,
    /// negative or out-of-range tier counts contribute zero. An advisory match
    /// does not prove pages will still exist at owner-side lookup time.
    fn parse_candidates(&self, reply: &Value) -> Vec<HintCandidate> {
        let Some(instances) = reply.get("instances").and_then(|v| v.as_object()) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for (id_str, summary) in instances {
            let Ok(id) = id_str.parse::<u64>() else {
                continue;
            };
            let Some(instance) = self.instances.get(&id) else {
                tracing::debug!(worker_id = id, "kv_hint: unmapped indexer instance");
                continue;
            };
            let tier = |k: &str| {
                summary
                    .get(k)
                    .and_then(|v| v.as_u64())
                    .and_then(|n| u32::try_from(n).ok())
                    .unwrap_or(0)
            };
            let expected = tier("cpu").max(tier("gpu"));
            if expected == 0 {
                continue;
            }
            out.push(HintCandidate {
                worker_url: instance.url.clone(),
                dp_rank: instance.dp_rank,
                servicer_addr: instance.servicer.clone(),
                expected_hit_tokens: expected,
            });
        }
        out
    }

    /// Preserve explicit GPU-resident prefix measurements independently of CPU hits.
    ///
    /// # Inputs
    /// `self` maps indexer IDs to slots; `reply` contains tier summaries.
    ///
    /// # Outputs
    /// Returns slot key ([`slot_key`]) to GPU-hit-token mappings, including
    /// valid zeroes. Missing, negative, noninteger, out-of-range or unmapped
    /// values are omitted rather than reported as measured zero.
    fn parse_gpu_observations(&self, reply: &Value) -> HashMap<String, u32> {
        let Some(instances) = reply.get("instances").and_then(Value::as_object) else {
            return HashMap::new();
        };
        instances
            .iter()
            .filter_map(|(id, summary)| {
                let instance = self.instances.get(&id.parse::<u64>().ok()?)?;
                let gpu = u32::try_from(summary.get("gpu")?.as_u64()?).ok()?;
                Some((slot_key(&instance.url, instance.dp_rank), gpu))
            })
            .collect()
    }

    /// Pick the best owner != the selected slot; `None` below the gates.
    /// The payload is the engine contract — field names must not change.
    /// Choose a peer prefix hint after the destination worker has been selected.
    ///
    /// # Inputs
    /// `self` supplies minimum-hit and incremental-hit gates; `answer` must
    /// have candidates sorted by decreasing expected hit tokens;
    /// `selected_worker_url` is the target that would fetch peer KV: a slot
    /// key ([`slot_key`]), or a worker URL (then no rank of that worker is a
    /// source).
    ///
    /// # Outputs
    /// Returns the highest-ranked other slot's engine-compatible hint if
    /// its gates pass, otherwise `None`. Optional local-prefix fallback adds an
    /// engine-side minimum-deficit gate so complete local matches can be checked
    /// at execution time. This function neither changes placement nor starts transfer.
    pub fn pick(&self, answer: &IndexerAnswer, selected_worker_url: &str) -> Option<Value> {
        let best = self
            .sources(answer, selected_worker_url)
            .into_iter()
            .next()?;
        Some(self.hint(best))
    }

    /// The holders of `answer` that may serve a pull to `selected` (a slot
    /// key or a worker URL, as in [`Self::pick`]), best hit first: every other
    /// slot with a servicer whose hit passes the minimum-hit gate and the
    /// incremental gate against the target's own hit.
    pub fn sources<'a>(&self, answer: &'a IndexerAnswer, selected: &str) -> Vec<&'a HintCandidate> {
        let target = Self::target_hit_tokens(answer, selected);
        answer
            .candidates
            .iter()
            .filter(|c| !c.is_target(selected) && !c.servicer_addr.trim().is_empty())
            .filter(|c| c.expected_hit_tokens >= self.min_tokens)
            .filter(|c| {
                // FULL-prefix ownership does not guarantee the target has the
                // SWA/state checkpoint for this endpoint. An opt-in fallback
                // lets the engine apply the gate to its complete local match.
                self.min_incremental_tokens == 0
                    || c.expected_hit_tokens.saturating_sub(target) >= self.min_incremental_tokens
                    || (self.local_prefix_fallback
                        && c.expected_hit_tokens >= self.min_incremental_tokens)
            })
            .collect()
    }

    /// The engine's `kv_pull_hint` for a pull from `source`: its worker, the
    /// servicer of its slot and its hit.
    pub fn hint(&self, source: &HintCandidate) -> Value {
        let mut hint = json!({
            "source_worker_id": source.worker_url,
            "source_servicer_addr": source.servicer_addr,
            "expected_hit_tokens": source.expected_hit_tokens,
        });
        if self.local_prefix_fallback {
            hint["min_local_deficit_tokens"] = json!(self.min_incremental_tokens.max(1));
        }
        hint
    }

    /// `--kv-pull-hint-min-incremental-tokens`: the pull floor (0 = gate off).
    pub fn min_incremental_tokens(&self) -> u32 {
        self.min_incremental_tokens
    }

    /// The selected slot's own matched tokens in this answer (0 if absent).
    /// Read the selected worker's advisory prefix hit from a shared answer.
    ///
    /// # Inputs
    /// `answer` contains mapped candidates; `selected_worker_url` identifies
    /// the chosen execution destination: a slot key, or a worker URL (then
    /// its best rank's hit, the rank its DP controller would choose).
    ///
    /// # Outputs
    /// Returns the matching candidate's expected hit-token count, or
    /// zero when absent. This combines CPU/GPU availability and is not a GPU-only
    /// residency measurement.
    pub fn target_hit_tokens(answer: &IndexerAnswer, selected_worker_url: &str) -> u32 {
        answer
            .candidates
            .iter()
            .filter(|c| c.is_target(selected_worker_url))
            .map(|c| c.expected_hit_tokens)
            .max()
            .unwrap_or(0)
    }

    /// Structured per-request decision line (INFO, target `kv_hint`): one
    /// line per hint-eligible request so canaries can parse routing + pull
    /// decisions without DEBUG.
    /// Emit one structured explanation of a selected worker and optional KV hint.
    ///
    /// # Inputs
    /// `self` is the injector; `answer` may contain lookup evidence; `policy`
    /// names the selector; `worker_url` is the chosen target; `hint` is the
    /// optional payload that will be injected.
    ///
    /// # Outputs
    /// Returns `()` after an INFO event under `kv_hint`, containing hit and
    /// incremental token counts, source, candidate count and lookup milliseconds.
    /// Missing lookup time is logged as -1; this function changes no routing state.
    pub fn log_decision(
        &self,
        answer: Option<&IndexerAnswer>,
        policy: &str,
        worker_url: &str,
        hint: Option<&Value>,
    ) {
        let (indexer_ms, num_candidates) = match answer {
            Some(a) => (a.indexer_ms, a.candidates.len()),
            None => (-1.0, 0),
        };
        let (best_other, target_hit) = match answer {
            Some(a) => (
                a.candidates
                    .iter()
                    .find(|c| !c.is_target(worker_url))
                    .map(|c| c.expected_hit_tokens)
                    .unwrap_or(0),
                Self::target_hit_tokens(a, worker_url),
            ),
            None => (0, 0),
        };
        tracing::info!(
            target: "kv_hint",
            policy = policy,
            worker = worker_url,
            injected = hint.is_some(),
            source = hint
                .and_then(|h| h.get("source_worker_id"))
                .and_then(|v| v.as_str())
                .unwrap_or(""),
            expected_hit_tokens = hint
                .and_then(|h| h.get("expected_hit_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            target_hit_tokens = target_hit,
            incremental_tokens = best_other.saturating_sub(target_hit),
            indexer_ms = indexer_ms,
            candidates = num_candidates,
            "kv_pull_hint decision"
        );
    }
}

/// Extract the request's top-level `cache_salt` (engine KV namespace).
/// Read an optional cache namespace from the top-level request object.
///
/// # Inputs
/// `body` is the parsed incoming request JSON.
///
/// # Outputs
/// Returns a borrowed nonempty `cache_salt` string, or `None` when absent,
/// empty or of another type. Does not derive a salt from agent identity.
pub fn extract_cache_salt(body: &Value) -> Option<&str> {
    match body.get("cache_salt")? {
        Value::String(s) if !s.is_empty() => Some(s),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Construct an offline hint-selection fixture with two mapped workers.
    ///
    /// # Inputs
    /// `min_tokens` and `min_incremental` are token-count thresholds for
    /// accepting a peer prefix.
    ///
    /// # Outputs
    /// Returns an injector with local-prefix fallback disabled and a 20 ms
    /// query timeout. The fixture construction makes no network request.
    fn injector(min_tokens: u32, min_incremental: u32) -> KvHintInjector {
        KvHintInjector::new(&KvHintConfig {
            indexer_url: "http://indexer:8090".into(),
            min_tokens,
            min_incremental_tokens: min_incremental,
            local_prefix_fallback: false,
            indexer_timeout_ms: 20,
            worker_map: vec![
                KvWorkerMapEntry {
                    id: 1,
                    url: "http://a:30000".into(),
                    servicer: "a:18100".into(),
                    dp_rank: None,
                },
                KvWorkerMapEntry {
                    id: 2,
                    url: "http://b:30000".into(),
                    servicer: "b:18100".into(),
                    dp_rank: None,
                },
            ],
        })
    }

    /// Two DP-attention workers `a` and `b` of two ranks each: instance
    /// `20001 + worker + 1000 * rank` (the launcher's ids), servicer
    /// `<host>:18100 + rank`; worker `c` is one instance without a rank.
    fn dp_injector(min_tokens: u32, min_incremental: u32) -> KvHintInjector {
        let mut worker_map = Vec::new();
        for (w, host) in ["a", "b"].iter().enumerate() {
            for rank in 0..2u32 {
                worker_map.push(KvWorkerMapEntry {
                    id: 20001 + w as u64 + 1000 * u64::from(rank),
                    url: format!("http://{host}:30000"),
                    servicer: format!("{host}:{}", 18100 + rank),
                    dp_rank: Some(rank),
                });
            }
        }
        worker_map.push(KvWorkerMapEntry {
            id: 20003,
            url: "http://c:30000".into(),
            servicer: "c:18100".into(),
            dp_rank: Some(0),
        });
        KvHintInjector::new(&KvHintConfig {
            indexer_url: "http://indexer:8090".into(),
            min_tokens,
            min_incremental_tokens: min_incremental,
            local_prefix_fallback: false,
            indexer_timeout_ms: 20,
            worker_map,
        })
    }

    /// Build a deterministic indexer answer from test JSON without HTTP.
    ///
    /// # Inputs
    /// `injector` supplies the worker mapping; `reply` supplies tier summaries.
    ///
    /// # Outputs
    /// Returns sorted candidates and parsed GPU observations with a fixed
    /// 1 ms query latency for assertions.
    fn answer(injector: &KvHintInjector, reply: Value) -> IndexerAnswer {
        let mut candidates = injector.parse_candidates(&reply);
        candidates.sort_by(|a, b| b.expected_hit_tokens.cmp(&a.expected_hit_tokens));
        IndexerAnswer {
            candidates,
            observed_gpu_hits: injector.parse_gpu_observations(&reply),
            indexer_ms: 1.0,
        }
    }

    #[test]
    /// Verify the best peer is chosen while GPU residency remains separately recorded.
    ///
    /// # Inputs
    /// No arguments. Create two workers with differing CPU/GPU hits and select the weaker worker.
    ///
    /// # Outputs
    /// Returns `()` when the stated invariant holds; failed assertions panic.
    fn parses_candidates_and_picks_best_other_worker() {
        let inj = injector(256, 0);
        let ans = answer(
            &inj,
            json!({"instances": {"1": {"gpu": 1024, "cpu": 4096}, "2": {"gpu": 512, "cpu": 512}}}),
        );
        // Selected worker is b — best other is a with max(cpu, gpu) = 4096.
        let hint = inj.pick(&ans, "http://b:30000").expect("hint");
        assert_eq!(hint["source_worker_id"], "http://a:30000");
        assert_eq!(hint["source_servicer_addr"], "a:18100");
        assert_eq!(hint["expected_hit_tokens"], 4096);
        assert_eq!(ans.observed_gpu_hits.get("http://a:30000"), Some(&1024));
        assert_eq!(ans.observed_gpu_hits.get("http://b:30000"), Some(&512));
    }

    #[test]
    /// Verify only valid GPU observations are retained without losing valid CPU candidates.
    ///
    /// # Inputs
    /// No arguments. Supply a zero GPU hit, absent GPU hit and several invalid GPU counts.
    ///
    /// # Outputs
    /// Returns `()` when the stated invariant holds; failed assertions panic.
    fn gpu_observations_distinguish_zero_from_missing_or_invalid() {
        let inj = injector(256, 0);
        let ans = answer(
            &inj,
            json!({"instances": {
                "1": {"gpu": 0, "cpu": 4096}, "2": {"cpu": 4096}
            }}),
        );
        assert_eq!(ans.observed_gpu_hits.get("http://a:30000"), Some(&0));
        assert!(!ans.observed_gpu_hits.contains_key("http://b:30000"));
        for invalid in [json!(-1), json!("4096"), json!(u64::MAX)] {
            let ans = answer(
                &inj,
                json!({"instances": {"1": {"gpu": invalid, "cpu": 4096}}}),
            );
            assert!(ans.observed_gpu_hits.is_empty());
            assert_eq!(ans.candidates[0].expected_hit_tokens, 4096);
        }
    }

    #[test]
    /// Verify no request is instructed to fetch KV from itself.
    ///
    /// # Inputs
    /// No arguments. Use an answer whose sole owner is the selected target.
    ///
    /// # Outputs
    /// Returns `()` when the stated invariant holds; failed assertions panic.
    fn no_hint_when_selected_worker_is_the_owner() {
        let inj = injector(256, 0);
        let ans = answer(&inj, json!({"instances": {"1": {"cpu": 4096}}}));
        assert!(inj.pick(&ans, "http://a:30000").is_none());
    }

    #[test]
    /// Verify the absolute minimum-prefix gate rejects the hint.
    ///
    /// # Inputs
    /// No arguments. Use a peer match below the absolute token threshold.
    ///
    /// # Outputs
    /// Returns `()` when the stated invariant holds; failed assertions panic.
    fn min_tokens_gate_blocks_small_matches() {
        let inj = injector(1024, 0);
        let ans = answer(&inj, json!({"instances": {"1": {"cpu": 512}}}));
        assert!(inj.pick(&ans, "http://b:30000").is_none());
    }

    #[test]
    /// Verify hint eligibility uses additional tokens beyond the target's own hit.
    ///
    /// # Inputs
    /// No arguments. Vary the target's hit while keeping the source's longer prefix fixed.
    ///
    /// # Outputs
    /// Returns `()` when the stated invariant holds; failed assertions panic.
    fn incremental_gate_compares_against_target_hit() {
        let inj = injector(256, 2048);
        // Owner a has 8192; selected b already holds 7168 → incremental
        // 1024 < 2048 → no hint.
        let ans = answer(
            &inj,
            json!({"instances": {"1": {"cpu": 8192}, "2": {"cpu": 7168}}}),
        );
        assert!(inj.pick(&ans, "http://b:30000").is_none());
        // Selected b holds only 4096 → incremental 4096 ≥ 2048 → hint.
        let ans = answer(
            &inj,
            json!({"instances": {"1": {"cpu": 8192}, "2": {"cpu": 4096}}}),
        );
        assert!(inj.pick(&ans, "http://b:30000").is_some());
    }

    #[test]
    /// Verify conditional hints defer complete-local-match checks to the engine.
    ///
    /// # Inputs
    /// No arguments. Enable fallback and vary target prefix length around a peer match.
    ///
    /// # Outputs
    /// Returns `()` when the stated invariant holds; failed assertions panic.
    fn complete_local_match_gate_can_recover_an_equal_or_shorter_peer_prefix() {
        let mut inj = injector(256, 2048);
        inj.local_prefix_fallback = true;
        for target in [4096, 8192, 16384] {
            let ans = answer(
                &inj,
                json!({"instances": {"1": {"cpu": 8192}, "2": {"cpu": target}}}),
            );
            let hint = inj.pick(&ans, "http://b:30000").expect("conditional hint");
            assert_eq!(hint["source_servicer_addr"], "a:18100");
            assert_eq!(hint["expected_hit_tokens"], 8192);
            assert_eq!(hint["min_local_deficit_tokens"], 2048);
        }
    }

    #[test]
    /// Verify fallback preserves minimum-hit gates and uses a positive local-deficit threshold.
    ///
    /// # Inputs
    /// No arguments. Exercise absolute/incremental gates and self-owner exclusion with fallback
    /// enabled.
    ///
    /// # Outputs
    /// Returns `()` when the stated invariant holds; failed assertions panic.
    fn local_prefix_fallback_preserves_disabled_and_minimum_hit_gates() {
        for (minimum, incremental) in [(16384, 2048), (256, u32::MAX)] {
            let mut inj = injector(minimum, incremental);
            inj.local_prefix_fallback = true;
            let ans = answer(
                &inj,
                json!({"instances": {"1": {"cpu": 8192}, "2": {"cpu": 8192}}}),
            );
            assert!(inj.pick(&ans, "http://b:30000").is_none());
        }
        let mut inj = injector(256, 0);
        inj.local_prefix_fallback = true;
        let ans = answer(&inj, json!({"instances": {"1": {"cpu": 8192}}}));
        assert!(inj.pick(&ans, "http://a:30000").is_none());
        assert_eq!(
            inj.pick(&ans, "http://b:30000").unwrap()["min_local_deficit_tokens"],
            1
        );
    }

    #[test]
    /// Verify unmapped indexer instances do not become routable candidates.
    ///
    /// # Inputs
    /// No arguments. Supply a large hit for an instance absent from the worker map.
    ///
    /// # Outputs
    /// Returns `()` when the stated invariant holds; failed assertions panic.
    fn unmapped_instances_are_skipped() {
        let inj = injector(256, 0);
        let ans = answer(&inj, json!({"instances": {"99": {"cpu": 65536}}}));
        assert!(ans.candidates.is_empty());
    }

    #[test]
    /// A map entry without a servicer (the ranks of a static split's decode
    /// worker, which run none) keeps its hits for placement but is never a
    /// pull source, even with the longest prefix.
    fn a_holder_without_a_servicer_is_no_pull_source() {
        let mut inj = injector(256, 0);
        inj.instances.insert(
            3,
            Instance {
                url: "http://d:30000".into(),
                dp_rank: None,
                servicer: String::new(),
            },
        );
        let ans = answer(
            &inj,
            json!({"instances": {"3": {"gpu": 16384, "cpu": 16384}, "1": {"cpu": 4096}}}),
        );
        assert_eq!(ans.candidates[0].worker_url, "http://d:30000");
        assert_eq!(ans.observed_gpu_hits.get("http://d:30000"), Some(&16384));
        let sources: Vec<&str> = inj
            .sources(&ans, "http://b:30000")
            .iter()
            .map(|c| c.worker_url.as_str())
            .collect();
        assert_eq!(sources, ["http://a:30000"]);
        let hint = inj.pick(&ans, "http://b:30000").expect("hint from a");
        assert_eq!(hint["source_servicer_addr"], "a:18100");
    }

    #[test]
    /// A worker whose map names two or more DP ranks is one slot per rank: each
    /// instance keeps its own hit and servicer, a hint names the holder rank's
    /// servicer, and a sibling rank of the target's worker is a valid source.
    fn dp_ranks_are_slots_with_their_own_hits_and_servicers() {
        let inj = dp_injector(256, 512);
        assert_eq!(inj.slot_ranks("http://a:30000/"), vec![Some(0), Some(1)]);
        // One instance with a rank is a worker placed as a whole.
        assert_eq!(inj.slot_ranks("http://c:30000"), vec![None]);
        assert_eq!(slot_key("http://a:30000", Some(1)), "http://a:30000#dp1");
        assert_eq!(
            split_slot("http://a:30000#dp1"),
            ("http://a:30000", Some(1))
        );
        assert_eq!(split_slot("http://a:30000"), ("http://a:30000", None));
        let ans = answer(
            &inj,
            json!({"instances": {
                "21001": {"gpu": 8192, "cpu": 0},
                "20001": {"gpu": 1024, "cpu": 0},
                "20002": {"cpu": 4096},
                "20003": {"gpu": 2048}
            }}),
        );
        let slots: Vec<String> = ans.candidates.iter().map(HintCandidate::slot).collect();
        assert_eq!(
            slots,
            [
                "http://a:30000#dp1",
                "http://b:30000#dp0",
                "http://c:30000",
                "http://a:30000#dp0"
            ]
        );
        assert_eq!(ans.observed_gpu_hits.get("http://a:30000#dp1"), Some(&8192));
        assert_eq!(ans.observed_gpu_hits.get("http://a:30000#dp0"), Some(&1024));
        assert_eq!(
            KvHintInjector::target_hit_tokens(&ans, "http://a:30000#dp0"),
            1024
        );
        // A plain worker URL reads its best rank (the rank its DP controller picks).
        assert_eq!(
            KvHintInjector::target_hit_tokens(&ans, "http://a:30000"),
            8192
        );
        // Rank 0 of a pulls from rank 1 of the same worker, at that rank's servicer.
        let hint = inj
            .pick(&ans, "http://a:30000#dp0")
            .expect("sibling rank source");
        assert_eq!(hint["source_worker_id"], "http://a:30000");
        assert_eq!(hint["source_servicer_addr"], "a:18101");
        assert_eq!(hint["expected_hit_tokens"], 8192);
        // Rank 1 holds the most itself: its best source is b rank 0, whose 4096
        // tokens add nothing over its own 8192 (incremental gate).
        assert!(inj.pick(&ans, "http://a:30000#dp1").is_none());
        // b rank 1 holds nothing: a rank 1 is its source.
        let hint = inj
            .pick(&ans, "http://b:30000#dp1")
            .expect("cross-worker source");
        assert_eq!(hint["source_servicer_addr"], "a:18101");
        // A worker-level target never takes one of its own ranks as the source,
        // and its hit is its best rank's: nothing to gain from b or c.
        assert!(inj.pick(&ans, "http://a:30000").is_none());
        let hint = inj.pick(&ans, "http://c:30000").expect("other worker");
        assert_eq!(hint["source_servicer_addr"], "a:18101");
        let sources: Vec<String> = inj
            .sources(&ans, "http://b:30000#dp1")
            .iter()
            .map(|c| c.slot())
            .collect();
        assert_eq!(
            sources,
            [
                "http://a:30000#dp1",
                "http://b:30000#dp0",
                "http://c:30000",
                "http://a:30000#dp0"
            ]
        );
    }

    #[test]
    /// Verify only a nonempty string is returned as the request cache namespace.
    ///
    /// # Inputs
    /// No arguments. Supply valid, empty, nonstring and absent top-level cache salt values.
    ///
    /// # Outputs
    /// Returns `()` when the stated invariant holds; failed assertions panic.
    fn extract_cache_salt_reads_top_level_string() {
        assert_eq!(extract_cache_salt(&json!({"cache_salt": "s1"})), Some("s1"));
        assert_eq!(extract_cache_salt(&json!({"cache_salt": ""})), None);
        assert_eq!(extract_cache_salt(&json!({"cache_salt": 5})), None);
        assert_eq!(extract_cache_salt(&json!({"prompt": "x"})), None);
    }
}
