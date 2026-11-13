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

use std::collections::HashMap;
use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};

/// One `--kv-worker-map` entry: indexer instance id → worker base URL and
/// its peer-KV servicer address (`host:port` of rank 0; the engine fans out
/// per-rank by port offset).
#[derive(Debug, Clone, Deserialize)]
pub struct KvWorkerMapEntry {
    pub id: u64,
    pub url: String,
    /// Peer-KV servicer address; only the pull-hint path needs it, so the
    /// symmetric_pd placer (which maps hits to worker URLs only) may omit it.
    #[serde(default)]
    pub servicer: String,
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

#[derive(Debug, Clone, PartialEq)]
pub struct HintCandidate {
    pub worker_url: String,
    pub servicer_addr: String,
    pub expected_hit_tokens: u32,
}

/// One indexer answer, reused across retry attempts of the same request.
#[derive(Debug, Clone)]
pub struct IndexerAnswer {
    /// Sorted by `expected_hit_tokens`, descending.
    pub candidates: Vec<HintCandidate>,
    /// Residency is distinct from CPU availability; absent observations stay absent.
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
    url_by_id: HashMap<u64, String>,
    servicer_by_id: HashMap<u64, String>,
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
        let mut url_by_id = HashMap::new();
        let mut servicer_by_id = HashMap::new();
        for entry in &cfg.worker_map {
            url_by_id.insert(entry.id, entry.url.clone());
            servicer_by_id.insert(entry.id, entry.servicer.clone());
        }
        Self {
            client,
            query_url: format!("{}/query", cfg.indexer_url.trim_end_matches('/')),
            min_tokens: cfg.min_tokens,
            min_incremental_tokens: cfg.min_incremental_tokens,
            local_prefix_fallback: cfg.local_prefix_fallback,
            timeout: Duration::from_millis(cfg.indexer_timeout_ms),
            url_by_id,
            servicer_by_id,
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
        self.url_by_id.get(&id).map(String::as_str)
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
            let (Some(url), Some(servicer)) =
                (self.url_by_id.get(&id), self.servicer_by_id.get(&id))
            else {
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
                worker_url: url.clone(),
                servicer_addr: servicer.clone(),
                expected_hit_tokens: expected,
            });
        }
        out
    }

    /// Preserve explicit GPU-resident prefix measurements independently of CPU hits.
    ///
    /// # Inputs
    /// `self` maps indexer IDs to worker URLs; `reply` contains tier summaries.
    ///
    /// # Outputs
    /// Returns worker URL to GPU-hit-token mappings, including valid zeroes.
    /// Missing, negative, noninteger, out-of-range or unmapped values are omitted
    /// rather than reported as measured zero.
    fn parse_gpu_observations(&self, reply: &Value) -> HashMap<String, u32> {
        let Some(instances) = reply.get("instances").and_then(Value::as_object) else {
            return HashMap::new();
        };
        instances
            .iter()
            .filter_map(|(id, summary)| {
                let url = self.url_by_id.get(&id.parse::<u64>().ok()?)?;
                let gpu = u32::try_from(summary.get("gpu")?.as_u64()?).ok()?;
                Some((url.clone(), gpu))
            })
            .collect()
    }

    /// Pick the best owner != the selected worker; `None` below the gates.
    /// The payload is the engine contract — field names must not change.
    /// Choose a peer prefix hint after the destination worker has been selected.
    ///
    /// # Inputs
    /// `self` supplies minimum-hit and incremental-hit gates; `answer` must
    /// have candidates sorted by decreasing expected hit tokens;
    /// `selected_worker_url` is the target that would fetch peer KV.
    ///
    /// # Outputs
    /// Returns the highest-ranked other worker's engine-compatible hint if
    /// its gates pass, otherwise `None`. Optional local-prefix fallback adds an
    /// engine-side minimum-deficit gate so complete local matches can be checked
    /// at execution time. This function neither changes placement nor starts transfer.
    pub fn pick(&self, answer: &IndexerAnswer, selected_worker_url: &str) -> Option<Value> {
        let best = answer
            .candidates
            .iter()
            .find(|c| c.worker_url != selected_worker_url)?;
        if best.expected_hit_tokens < self.min_tokens {
            return None;
        }
        if self.min_incremental_tokens > 0 {
            let target = Self::target_hit_tokens(answer, selected_worker_url);
            let incremental = best.expected_hit_tokens.saturating_sub(target);
            // FULL-prefix ownership does not guarantee the target has the
            // SWA/state checkpoint for this endpoint. An opt-in fallback
            // lets the engine apply the gate to its complete local match.
            if incremental < self.min_incremental_tokens
                && (!self.local_prefix_fallback
                    || best.expected_hit_tokens < self.min_incremental_tokens)
            {
                return None;
            }
        }
        let mut hint = json!({
            "source_worker_id": best.worker_url,
            "source_servicer_addr": best.servicer_addr,
            "expected_hit_tokens": best.expected_hit_tokens,
        });
        if self.local_prefix_fallback {
            hint["min_local_deficit_tokens"] = json!(self.min_incremental_tokens.max(1));
        }
        Some(hint)
    }

    /// `--kv-pull-hint-min-incremental-tokens`: the pull floor (0 = gate off).
    pub fn min_incremental_tokens(&self) -> u32 {
        self.min_incremental_tokens
    }

    /// The selected worker's own matched tokens in this answer (0 if absent).
    /// Read the selected worker's advisory prefix hit from a shared answer.
    ///
    /// # Inputs
    /// `answer` contains mapped candidates; `selected_worker_url` identifies
    /// the chosen execution destination.
    ///
    /// # Outputs
    /// Returns the first matching candidate's expected hit-token count, or
    /// zero when absent. This combines CPU/GPU availability and is not a GPU-only
    /// residency measurement.
    pub fn target_hit_tokens(answer: &IndexerAnswer, selected_worker_url: &str) -> u32 {
        answer
            .candidates
            .iter()
            .find(|c| c.worker_url == selected_worker_url)
            .map(|c| c.expected_hit_tokens)
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
                    .find(|c| c.worker_url != worker_url)
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
                },
                KvWorkerMapEntry {
                    id: 2,
                    url: "http://b:30000".into(),
                    servicer: "b:18100".into(),
                },
            ],
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
