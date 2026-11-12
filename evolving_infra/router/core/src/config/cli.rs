// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Command-line interface. The router is configured entirely through
//! flags — there is no config file. [`Cli::into_config`] resolves the
//! flags into a validated [`Config`].

use anyhow::{anyhow, Result};
use clap::Parser;
use std::num::NonZeroU32;

use crate::config::{
    default_cb_cool_down, default_proxy_request_timeout_secs, default_stale_request_timeout_secs,
    default_symmetric_pd_load_poll_ms, default_symmetric_pd_max_queue_wait_ms,
    default_symmetric_pd_max_uncached_tokens, resolve_mode, ActiveLoadConfig, CacheAwareConfig,
    CircuitBreakerConfig, Config, DiscoveryBackend, GatewayCacheAwareConfig, K8sDiscoveryConfig,
    KvIndexerEndpointConfig, LogFormat, ModelConfig, ObservabilityConfig, PdDecisionReplayConfig,
    PdDecodePolicy, PdDecodeTokensConfig, PolicyKind, ProxyConfig, ServerConfig,
    StaticUrlsDiscoveryConfig, StickyConfig, SymmetricPdConfig, SymmetricPdCostFunctionConfig,
    SymmetricPdDSelect, SymmetricPdImportCachePolicy, SymmetricPdPSelect, SymmetricPdPreset,
    SymmetricPdWorkerSpec, SYMMETRIC_PD_MAX_QUEUE_WAIT_MS_CAP,
};

const DEFAULT_KV_INDEXER_QUERY_TIMEOUT_MS: u64 = 100;
/// Decode->prefill drain bound when external role control sets no explicit value.
const DEFAULT_ROLE_DRAIN_TIMEOUT_SECS: u64 = 5;
const DEFAULT_KV_INDEXER_QUERY_MAX_INFLIGHT: usize = sgl_kv_indexer::DEFAULT_QUERY_MAX_INFLIGHT;
/// Per-RID progress poll period of the P-role workers under
/// `--symmetric-pd-p-select cost-function` when `--prefill-poll-ms` is unset.
const DEFAULT_COST_FUNCTION_LOAD_POLL_MS: u64 = 100;
/// KV hint indexer query timeout when `--kv-indexer-timeout-ms` is unset.
const DEFAULT_KV_HINT_INDEXER_TIMEOUT_MS: u64 = 20;
/// KV hint indexer query timeout set by `--symmetric-pd-preset paper`.
const PAPER_PRESET_KV_HINT_INDEXER_TIMEOUT_MS: u64 = 200;
/// `--symmetric-pd-preset paper` with `--prefill-wait forecast`: the
/// `--max-state-age-ms`.
const PAPER_PRESET_MAX_STATE_AGE_MS: f64 = 5000.0;
/// Minimum expected hit (one page) when `--kv-pull-hint-min-tokens` is unset.
const DEFAULT_KV_PULL_HINT_MIN_TOKENS: u32 = 256;

/// Trimmed decision-replay routing-key header named by `flag`, or the default
/// header when the flag is unset. An empty value is an error.
fn resolve_decision_replay_header(header: Option<&str>, flag: &str) -> Result<String> {
    match header {
        Some(h) if h.trim().is_empty() => Err(anyhow!("{flag} must not be empty")),
        Some(h) => Ok(h.trim().to_string()),
        None => Ok(crate::config::types::default_symmetric_pd_decision_replay_header()),
    }
}

fn parse_kv_worker_map(raw: &str) -> Result<Vec<crate::kv_hint::KvWorkerMapEntry>> {
    let entries: Vec<crate::kv_hint::KvWorkerMapEntry> =
        serde_json::from_str(raw).map_err(|e| {
            anyhow!("--kv-worker-map is not a valid JSON array of {{id, url, servicer}}: {e}")
        })?;
    if entries.is_empty() {
        return Err(anyhow!("--kv-worker-map must not be empty"));
    }
    Ok(entries)
}

/// Match SGLang's HostReturn endpoint grammar and canonical identity check.
/// Peer URLs are rank-0 host:port addresses, not HTTP URLs.
fn normalize_host_return_endpoint(raw: &str) -> Result<String> {
    let invalid = || anyhow!("host return peer servicer must be host:port with port in 1..65535");
    if raw.is_empty() || raw.trim() != raw {
        return Err(invalid());
    }
    // PeerKVBackend and PeerKVServicer currently use IPv4 transport. Do not
    // accept an IPv6 directive that can pass validation but never execute.
    let (host, port_text) = raw.rsplit_once(':').ok_or_else(invalid)?;
    let host = host.strip_suffix('.').unwrap_or(host).to_ascii_lowercase();
    if host.len() > 253
        || !host.split('.').all(|label| {
            !label.is_empty()
                && label.len() <= 63
                && label.as_bytes()[0].is_ascii_alphanumeric()
                && label.as_bytes()[label.len() - 1].is_ascii_alphanumeric()
                && label
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'-')
        })
    {
        return Err(invalid());
    }
    if port_text.is_empty() || port_text.len() > 5 || !port_text.bytes().all(|c| c.is_ascii_digit())
    {
        return Err(invalid());
    }
    let port = port_text.parse::<u16>().map_err(|_| invalid())?;
    if port == 0 {
        return Err(invalid());
    }
    Ok(format!("{host}:{port}"))
}

/// `sgl-router` — slim KV-aware OpenAI-compatible router for SGLang workers.
///
/// Discovery is mutually exclusive: pass `--worker-urls` for a static
/// worker list, or `--service-discovery` for Kubernetes EndpointSlice
/// discovery — exactly one is required.
#[derive(Parser, Debug)]
#[command(
    name = "sgl-router",
    version,
    about = "Slim KV-aware OpenAI-compatible router for SGLang workers"
)]
pub struct Cli {
    // ---- server ----
    /// Address to bind the HTTP server to.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// Port to bind the HTTP server to.
    #[arg(long, default_value_t = 30000)]
    pub port: u16,

    // ---- model (exactly one) ----
    /// Model id this router serves (the OpenAI `model` field).
    #[arg(long)]
    pub model_id: String,
    /// Tokenizer source: a local `tokenizer.json` path, or a HuggingFace
    /// repo id to download from. When omitted, falls back to `--model-id`
    /// as the repo id (download honors `HF_TOKEN` / `HF_HOME`).
    #[arg(long)]
    pub tokenizer_path: Option<String>,
    /// Routing policy.
    #[arg(long, value_enum, default_value = "round_robin")]
    pub policy: PolicyKind,

    // ---- circuit breaker (opt-in via --cb-threshold) ----
    /// Consecutive upstream failures before the circuit breaker opens.
    /// Setting this enables the circuit breaker; `0` is rejected.
    #[arg(long)]
    pub cb_threshold: Option<NonZeroU32>,
    /// Circuit-breaker cool-down in seconds. Only meaningful with
    /// `--cb-threshold`; defaults to 30 when the breaker is enabled.
    #[arg(long)]
    pub cb_cool_down_secs: Option<u64>,
    /// Disable the per-worker circuit breaker: every worker always admits
    /// and never leaves the candidate sets on upstream failures (the
    /// counterpart of sgl-model-gateway's `--disable-circuit-breaker`).
    /// Without this flag and without `--cb-threshold`, every worker still
    /// has the default breaker (3 consecutive failures open it for 30 s).
    /// Conflicts with `--cb-threshold` / `--cb-cool-down-secs`.
    #[arg(long)]
    pub disable_circuit_breaker: bool,

    // ---- cache-aware-zmq tuning (only used by that policy) ----
    /// Min `matched_blocks / total_blocks` for a cache match to win.
    #[arg(long)]
    pub cache_threshold: Option<f32>,
    /// Absolute load spread above which the cache check is skipped.
    #[arg(long)]
    pub balance_abs_threshold: Option<usize>,
    /// Multiplicative load spread gating the absolute balance check.
    #[arg(long)]
    pub balance_rel_threshold: Option<f32>,
    /// External KV indexer gRPC endpoint used as the authoritative cache signal.
    /// Needs an explicit scheme, e.g. `http://10.0.0.1:50051`.
    #[arg(long)]
    pub kv_indexer_endpoint: Option<String>,
    /// KV Indexer query timeout in milliseconds. Requires
    /// `--kv-indexer-endpoint`; defaults to 100.
    #[arg(long)]
    pub kv_indexer_query_timeout_ms: Option<u64>,
    /// Maximum concurrent KV Indexer queries issued by this Router. Requires
    /// `--kv-indexer-endpoint`; defaults to 32.
    #[arg(long)]
    pub kv_indexer_query_max_inflight: Option<usize>,

    // ---- gateway_cache_aware tuning (`--policy gateway_cache_aware`, or
    // `--policy symmetric_pd --symmetric-pd-p-select gateway-cache-aware`) ----
    /// sgl-model-gateway `--cache-threshold` (default 0.3, range 0..=1):
    /// the matched worker wins when `matched tokens / prompt tokens` is
    /// greater than this; else the least-loaded prefill.
    #[arg(long, hide = true)]
    pub gateway_ca_cache_threshold: Option<f32>,
    /// sgl-model-gateway `--balance-abs-threshold` (default 64): imbalanced
    /// when `max - min` in-flight prefill requests is greater than this (and
    /// the relative test holds).
    #[arg(long, hide = true)]
    pub gateway_ca_balance_abs_threshold: Option<usize>,
    /// sgl-model-gateway `--balance-rel-threshold` (default 1.5, at least
    /// 1): imbalanced when `max > min * this` (and the absolute test holds).
    #[arg(long, hide = true)]
    pub gateway_ca_balance_rel_threshold: Option<f32>,
    /// sgl-model-gateway `--eviction-interval` in seconds (default 120,
    /// positive): period of the tree size eviction.
    #[arg(long, hide = true)]
    pub gateway_ca_eviction_interval_secs: Option<u64>,
    /// sgl-model-gateway `--max-tree-size` (default 67108864, positive): tree
    /// characters kept per prefill worker by each eviction (8 per token).
    #[arg(long, hide = true)]
    pub gateway_ca_max_tree_size: Option<usize>,

    // ---- sticky-session policy (only used by `--policy sticky`) ----
    /// Request header carrying the routing key for sticky-session routing.
    /// Defaults to `x-sgl-routing-key` when `--policy sticky` is set.
    #[arg(long)]
    pub routing_key_header: Option<String>,
    /// Policy used to select a worker for requests with no routing key, and
    /// to pick the initial worker when a new key is first seen. One of
    /// `round_robin` / `random` / `power_of_two` / `load_based`. Defaults
    /// to `round_robin`.
    #[arg(long, value_enum)]
    pub sticky_fallback_policy: Option<PolicyKind>,
    /// Evict a sticky assignment after it has been idle (unreferenced) this
    /// many seconds. Defaults to 600.
    #[arg(long)]
    pub sticky_idle_secs: Option<u64>,
    /// Wall-clock cadence of the sticky idle-eviction sweep, in seconds.
    /// Defaults to 60.
    #[arg(long)]
    pub sticky_eviction_interval_secs: Option<u64>,

    // ---- KV pull-hint injection (peer KV pulls between workers) ----
    /// Inject an engine-opaque `kv_pull_hint` into forwarded bodies after
    /// the policy selects a worker (PREFILL request only in PD mode;
    /// fail-open on every indexer error).
    #[arg(long)]
    pub inject_kv_pull_hint: bool,
    /// Use Dynamo CPU/GPU prefix observations for cache-aware placement only.
    /// Requires --policy cache_aware_zmq, --kv-indexer-url and --kv-worker-map.
    #[arg(long)]
    pub dynamo_cache_aware: bool,
    /// Standalone dynamo KV indexer base URL (e.g. `http://indexer:8090`).
    /// Required with `--inject-kv-pull-hint`.
    #[arg(long)]
    pub kv_indexer_url: Option<String>,
    /// Minimum expected hit (tokens) before a hint is injected. Gates the
    /// source peer's TOTAL matched prefix, which long sessions always
    /// exceed — prefer `--kv-pull-hint-min-incremental-tokens`. Defaults
    /// to 256 (one page).
    #[arg(long)]
    pub kv_pull_hint_min_tokens: Option<u32>,
    /// Inject only when `best_source_match − routed_target_match ≥ N`
    /// tokens (both from the same indexer answer) — gates on the recompute
    /// the pull actually saves. 0 disables the incremental gate.
    #[arg(long)]
    pub kv_pull_hint_min_incremental_tokens: Option<u32>,
    /// Let the engine apply the incremental-hit gate to its complete local
    /// match, including SWA/state checkpoints. Requires engine support for
    /// the min_local_deficit_tokens hint field.
    #[arg(long)]
    pub kv_pull_hint_local_prefix_fallback: bool,
    /// Indexer query timeout in milliseconds; fail-open on expiry.
    /// Defaults to 20.
    #[arg(long)]
    pub kv_indexer_timeout_ms: Option<u64>,
    /// JSON array mapping indexer instance ids to workers and their peer-KV
    /// servicer addresses:
    /// `[{"id":1,"url":"http://a:30000","servicer":"a:18100"}]`.
    /// Required with `--inject-kv-pull-hint`; list PREFILL engines only in
    /// PD deployments.
    #[arg(long)]
    pub kv_worker_map: Option<String>,

    // ---- the placement cost function of `--symmetric-pd-p-select
    // cost-function` (docs/placement.md): cost(P) = R x (wait(P) + exposed
    // fetch time) + uncached tokens + copy price x fetched tokens ----
    /// Placement cost function: R, the uncached prefill tokens one prefill
    /// worker computes per second. It converts wait and fetch time into
    /// tokens. Default 13500. `auto`: the router measures R as the computed
    /// prefill tokens per busy second of all prefill workers (13500 for the
    /// first 120 s and while no measurement exists).
    #[arg(long, value_parser = parse_number_or_auto)]
    pub prefill_rate_tokens_per_s: Option<NumberOrAuto>,
    /// Placement cost function: fixed time of one KV fetch in milliseconds.
    /// Default 450.
    #[arg(long)]
    pub fetch_fixed_ms: Option<f64>,
    /// Placement cost function: KV fetch time in milliseconds per 100k
    /// fetched tokens. Default 100.
    #[arg(long)]
    pub fetch_ms_per_100k_tokens: Option<f64>,
    /// Placement cost function: the copy price, in uncached-token
    /// equivalents per fetched token. It charges the second copy that a
    /// fetch leaves in the host pool of the target worker. Default 4.6.
    /// 0 charges the fetch time only. `auto`: at each decision the price is
    /// q x R / R_hat (q: active requests per prefill worker times their
    /// busy share; R_hat: the measured prefill rate), clamped to
    /// --copy-price-min and --copy-price-max.
    #[arg(long, value_parser = parse_number_or_auto)]
    pub copy_price: Option<NumberOrAuto>,
    /// Placement cost function: how the wait of a prefill worker is
    /// estimated. `tokens`: the uncached tokens the router has sent to the
    /// worker and not seen finish, divided by R. `forecast`: the prefill
    /// completion forecast, a step-time model run on the scheduler state
    /// that the engine reports (the engine needs
    /// SGLANG_PREFILL_ROUTING_SNAPSHOT=1; a symmetric worker also
    /// SGLANG_SYMMETRIC_ROUTING_SNAPSHOT=1). With `forecast`, a request
    /// leaves its cache-aware owner only when the forecast saving exceeds
    /// the forecast error (--move-margin). A worker without a usable
    /// forecast keeps the token-count wait and is never chosen in place of
    /// the owner. Default `forecast` with --symmetric-pd-preset paper,
    /// else `tokens`.
    #[arg(long, value_parser = ["tokens", "forecast"])]
    pub prefill_wait: Option<String>,
    /// Forecast: a worker state that is older than this many milliseconds
    /// at the decision is not used. Default 5000 with
    /// --symmetric-pd-preset paper, else 1500.
    #[arg(long)]
    pub max_state_age_ms: Option<f64>,
    /// Forecast: the error-table file (schema `forecast-error-table.v2`),
    /// the error bound of the forecast per deployment mode. Default: the
    /// table built into the router (`data/forecast-error-table.json`). The
    /// tables were fitted for engines with 8192-token chunks and 64-token
    /// pages; a worker with another chunk or page size gets no forecast.
    /// The table's sha256 is in the startup log and the placement trace.
    #[arg(long)]
    pub forecast_error_table: Option<std::path::PathBuf>,
    /// Forecast: the margin m of the move rule. A request leaves its
    /// cache-aware owner o for the cheapest worker w only if
    /// (C_o - C_w) / R - m x sqrt(b_o^2 + b_w^2) > 0, with C the cost and
    /// b the error bound of each forecast. Default 0 with
    /// --symmetric-pd-preset paper, else 1.
    #[arg(long)]
    pub move_margin: Option<f64>,
    /// Forecast: subscribe to the state push of each prefill engine (one
    /// ZMQ SUB per worker at tcp://<host>:PORT; the engine needs
    /// SGLANG_PREFILL_STATE_PUSH_PORT=PORT and
    /// SGLANG_PREFILL_ROUTING_SNAPSHOT=1). A pushed state replaces the
    /// polled one when it is newer, at most --max-state-age-ms old and from
    /// the same engine start. Workers that share a host keep the poll.
    /// Default off.
    #[arg(long)]
    pub forecast_push_port: Option<u16>,
    /// Forecast: when the last pass of a request leaves no known work on a
    /// busy worker, also count the next pass that the expected arrivals
    /// form (the engine handles the results of a pass after it launches
    /// the next one). `--forecast-next-pass-lag` or `=true` turns it on,
    /// `=false` off. Default on with --symmetric-pd-preset paper when a
    /// forecast is used, else off.
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true")]
    pub forecast_next_pass_lag: Option<bool>,
    /// Forecast: a JSON file of offline step-model constants that replace
    /// the online fits of every prefill worker (the online samples still
    /// give the arrival rates). It helps when a worker is never idle: the
    /// online fit takes only passes launched after an idle gap, so a
    /// saturated worker has no fit and its forecast falls back. The file's
    /// sha256 is in the startup log. Default off.
    #[arg(long)]
    pub forecast_fit: Option<std::path::PathBuf>,
    /// Placement cost function, `--copy-price auto`: the lowest copy price.
    /// Default 4.6.
    #[arg(long)]
    pub copy_price_min: Option<f64>,
    /// Placement cost function, `--copy-price auto`: the highest copy
    /// price. Default 1000.
    #[arg(long)]
    pub copy_price_max: Option<f64>,
    /// Placement cost function, `--copy-price auto`: the copy price for the
    /// first seconds and while a measurement is missing. Default 4.6.
    #[arg(long)]
    pub copy_price_fallback: Option<f64>,
    /// Placement cost function: write a new JSONL file (schema
    /// `placement-trace.v1`) with each placement and its costs, the request
    /// lifecycle events and 1 s snapshots. Default off.
    #[arg(long)]
    pub placement_trace: Option<std::path::PathBuf>,
    /// Placement cost function: interval in milliseconds of the progress
    /// poll (`/v1/loads`) of each prefill worker. Default 100.
    #[arg(long)]
    pub prefill_poll_ms: Option<u64>,
    /// Session KV retention on the PD path of
    /// `--policy gateway_cache_aware` (needs engines with session
    /// retention). A request with an
    /// `x-kv-session-context` header (the JSON of `x-agent-context`)
    /// registers its prefill worker; only the prefill-leg body carries
    /// `kv_session`, and the body `rid` is not changed. When the prefill
    /// worker changes, the old copy is released after the new prefill
    /// reports the full context ready. `POST /kv_sessions/end` ends a
    /// session. `--policy symmetric_pd` uses `--symmetric-pd-kv-session-retention`.
    #[arg(long)]
    pub kv_session_retention: bool,
    /// File with the engine API/admin key for the background controls of
    /// `--kv-session-retention`.
    #[arg(long, requires = "kv_session_retention")]
    pub kv_session_key_file: Option<std::path::PathBuf>,

    // ---- symmetric P/D placement (only used by `--policy symmetric_pd`) ----
    /// symmetric P/D worker `URL,BOOTSTRAP_PORT[,PREFILL_WEIGHT[,DECODE_WEIGHT]]`
    /// (repeatable), e.g. `http://10.0.0.1:30000,8998,1,0`. The URL is also
    /// registered for static discovery; weights default to 1, and `0` is a
    /// hard exclusion for that leg (static 4+4).
    #[arg(long = "symmetric-pd-worker", action = clap::ArgAction::Append)]
    pub symmetric_pd_worker: Vec<String>,
    /// JSON array of symmetric P/D workers: `[{"url", "bootstrap_port",
    /// "prefill_weight", "decode_weight", "capabilities": ["prefill","decode"]}]`.
    #[arg(long)]
    pub symmetric_pd_workers_file: Option<String>,
    /// Forbid Local placements (`p == d`): role-locked PD as a configuration.
    #[arg(long)]
    pub symmetric_pd_force_split: bool,
    /// Inject the peer-kv `kv_pull_hint` into the P-leg when the
    /// indexer's best host-tier owner is another engine (needs
    /// `--kv-worker-map` entries with `servicer`; same gates as the PD hint).
    #[arg(long)]
    pub symmetric_pd_pull_hint: bool,
    /// Hold-queue cap in ms (default 2000, max 30000): a request waits at
    /// most this long for a P-leg candidate under
    /// `--symmetric-pd-max-uncached-tokens`, then is force-placed. Zero disables holding.
    #[arg(long)]
    pub symmetric_pd_max_queue_wait_ms: Option<u64>,
    /// Pending uncached prefill tokens above which a P-leg candidate counts
    /// as overloaded (engine `/v1/loads` + router ledger). Default 10240.
    #[arg(long)]
    pub symmetric_pd_max_uncached_tokens: Option<u64>,
    /// `/v1/loads` poll period per symmetric P/D worker in ms. Default 1000.
    #[arg(long)]
    pub symmetric_pd_load_poll_ms: Option<u64>,
    /// Score weight of `reuse_gain(p)` (default 1.0).
    #[arg(long, hide = true)]
    pub symmetric_pd_w_hit: Option<f64>,
    /// Score weight of `backlog(p)` (default 0.3).
    #[arg(long, hide = true)]
    pub symmetric_pd_w_backlog: Option<f64>,
    /// Score weight of `role_pen(p) + role_pen(d)` (default 0.3).
    #[arg(long, hide = true)]
    pub symmetric_pd_w_role: Option<f64>,
    /// Score weight of `xfer_cost(p, d)` (default 0.2).
    #[arg(long, hide = true)]
    pub symmetric_pd_w_xfer: Option<f64>,
    /// Score weight of `decode_load(d)` (default 0.1; 0 recovers the plain
    /// additive formula).
    #[arg(long, hide = true)]
    pub symmetric_pd_w_dload: Option<f64>,
    /// Prefill selection. `cache-first` (default without the paper preset):
    /// the cache-aware rule (imbalance fast path, best hit above the cache
    /// threshold, else least loaded). `score`: argmax of the additive score.
    /// `gateway-cache-aware`: the sgl-model-gateway cache-aware prefill
    /// rule; never holds requests. `cost-function`: the placement cost
    /// function (docs/placement.md) over the prefill workers; it picks the
    /// prefill worker and the KV fetch of its `kv_pull_hint`.
    /// `cost-function` needs `--symmetric-pd-force-split`,
    /// `--symmetric-pd-pull-hint`, `--kv-indexer-url` and a
    /// `--kv-worker-map` entry with a servicer for each prefill worker.
    #[arg(long, value_enum)]
    pub symmetric_pd_p_select: Option<SymmetricPdPSelect>,
    /// `--policy symmetric_pd` only: what fills the flags left unset.
    /// `paper` (default): force split, round-robin decode selection, 100 ms
    /// load polls, late decode choice by `fit_req` over every decode on
    /// observed capacity, KV session retention, and, with `--kv-indexer-url`
    /// and a `--kv-worker-map` that names servicers, the placement cost
    /// function with KV fetches from any holder. With the cost function it
    /// also sets `--prefill-wait forecast`, `--global-queue`,
    /// `--global-queue-room forecast`, `--forecast-next-pass-lag`,
    /// `--move-margin 0` and `--max-state-age-ms 5000`. The paper's runs
    /// used `--prefill-wait tokens --global-queue=false`. `none`: the flags
    /// as given.
    #[arg(long, value_enum, default_value = "paper")]
    pub symmetric_pd_preset: SymmetricPdPreset,
    /// `--symmetric-pd-p-select cost-function` only: an unknown P-leg reservation
    /// (a transport error such as `upstream timed out`, or a drop after
    /// dispatch) leaves the work book when its P answers a progress poll
    /// sent at least this many seconds after the later of the error and the
    /// last listing of its RID, and the answer does not list the RID. A P
    /// that does not answer keeps the work. A P aborts a prefill whose router
    /// connection closed, and lists only RIDs whose first chunk launched, so
    /// a still-queued request and a dropped one look alike; this bounds that
    /// error. Default 60; 0 = never expire (the work is held until a terminal
    /// result).
    #[arg(long)]
    pub symmetric_pd_unknown_expire_secs: Option<u64>,
    /// Placement cost function: hold requests in one router-wide queue and
    /// send each to its cheapest prefill worker when that worker is about
    /// to have room. A request whose local cache hit binds it to one worker
    /// (its largest hit exceeds the second largest by at least one engine
    /// chunk) is placed at arrival. Other requests wait in HRRN order until
    /// the queued work of their cheapest worker, divided by R, is within
    /// that worker's lead time (one chunk / R plus the measured time from a
    /// release to the worker's first report of the request). A request is
    /// held at most 300 s. `--global-queue` or `=true` turns it on, `=false`
    /// off. Default on with --symmetric-pd-preset paper, else off.
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true")]
    pub global_queue: Option<bool>,
    /// Global queue: when a prefill worker has room. `tokens`: its queued
    /// work divided by R is within the lead time. `forecast`: also when the
    /// forecast time to drain its queue is within the lead time; a worker
    /// without a usable forecast keeps the token rule. `forecast` uses the
    /// forecast state (the --forecast-* flags) also with
    /// `--prefill-wait tokens`. Default `forecast` with
    /// --symmetric-pd-preset paper, else `tokens`.
    #[arg(long, value_parser = ["tokens", "forecast"])]
    pub global_queue_room: Option<String>,
    /// `--symmetric-pd-p-select cost-function` only: each cost-function reserve event of
    /// a request with a session identity gets a `session_move` record (the
    /// session, the P of its last decision, the chosen P, the largest local
    /// hit over the P, the local hit on the last P, the pulled tokens), so
    /// the share of moved sessions comes from the router trace alone. Off by
    /// default: the trace is unchanged.
    #[arg(long, hide = true)]
    pub symmetric_pd_trace_session_moves: bool,
    /// With `--symmetric-pd-p-select cost-function`: the pull source of a P-leg may
    /// be any holder of the indexer answer that is a healthy registered
    /// worker of any role (breaker allowing) with a `--kv-worker-map` entry,
    /// also a D-role worker such as a former P after a P->D role switch. The
    /// P-leg targets stay the healthy P-role workers. Off (default): every
    /// hit and every pull source is a P-role worker.
    #[arg(long)]
    pub symmetric_pd_pull_from_any_holder: bool,
    /// D-leg selection once P is fixed: `score` (default) or `round-robin`
    /// (the `--pd-decode-policy round-robin` equivalent).
    #[arg(long, value_enum)]
    pub symmetric_pd_d_select: Option<SymmetricPdDSelect>,
    /// `cache-first`: best `reuse_gain` must exceed this to win on cache
    /// (default 0.5, like `--cache-threshold`).
    #[arg(long)]
    pub symmetric_pd_cache_threshold: Option<f64>,
    /// `cache-first` imbalance fast path: spread of in-flight P-legs
    /// (`max - min`) above which the least-loaded worker is taken
    /// regardless of cache (default 32, like `--balance-abs-threshold`).
    #[arg(long)]
    pub symmetric_pd_balance_abs_reqs: Option<u64>,
    /// `cache-first` imbalance fast path: `max > min * this` must also hold
    /// (default 1.1, like `--balance-rel-threshold`).
    #[arg(long)]
    pub symmetric_pd_balance_rel: Option<f64>,
    /// Worth of a host-tier (CPU) indexer hit relative to a GPU-tier hit in
    /// `reuse_gain` (default 1.0). 0 counts the GPU tier only; a value below 1
    /// lets a host-resident prefix compete with, but not outrank, a
    /// GPU-resident one.
    #[arg(long)]
    pub symmetric_pd_host_hit_factor: Option<f64>,
    /// Routing control: replay a reference run's
    /// per-request (P, D) choices. JSON with `"key_hash":"sha256"` and either
    /// `"sessions":{"<sha256 hex of the routing key>":[["<p url>","<d url>"],
    /// ...]}` (the k-th request of a routing key takes its k-th pair) or
    /// `"sessions_by_len":{"<hex>":{"<prompt tokens>":[[p,d],...]}}` with
    /// optional `"len_tolerance"` (the request is matched by prompt length;
    /// repeated occurrences take successive pairs). Unknown keys/lengths,
    /// exhausted sessions and ineligible pairs fall back to the configured
    /// selection; every decision logs `replay=hit|hit_len|hit_len_near|
    /// unknown_key|unknown_len|exhausted|no_key|off`.
    #[arg(long, hide = true)]
    pub symmetric_pd_decision_replay: Option<std::path::PathBuf>,
    /// Header carrying the routing key for `--symmetric-pd-decision-replay`
    /// (default `X-SMG-Routing-Key`, what the strict-replay client sends).
    #[arg(long, hide = true)]
    pub symmetric_pd_decision_replay_header: Option<String>,
    /// What the D-leg engine does with the imported prompt KV at finish
    /// (`kv_import.cache_policy`): `auto` (default: `gpu-only` for D-leg
    /// workers with prefill weight 0, `retain` otherwise), `retain`,
    /// `gpu-only` (insert into the radix tree, no host write-through — what
    /// upstream PD decode does), `drop`.
    #[arg(long, value_enum)]
    pub symmetric_pd_import_cache_policy: Option<SymmetricPdImportCachePolicy>,
    /// Return generated D-side host KV to this request's selected P after D2H
    /// completes. Defaults an unspecified import policy to retain; requires
    /// retain and explicit peer servicers in --kv-worker-map;
    /// all workers must enable compatible peer HiCache. No indexer is required.
    #[arg(long)]
    pub symmetric_pd_decode_host_return: bool,
    #[arg(long, requires = "symmetric_pd_decode_host_return")]
    pub symmetric_pd_decode_host_return_timeout_ms: Option<u64>,
    /// Bound newly generated suffix tokens, not full request tokens.
    #[arg(long, requires = "symmetric_pd_decode_host_return")]
    pub symmetric_pd_decode_host_return_max_tokens: Option<u64>,
    /// Let an external controller replace symmetric workers' roles (`/symmetric_pd/roles`).
    #[arg(long)]
    pub symmetric_pd_external_role_control: bool,

    /// Enable versioned trajectory CPU KV retention (requires matching engine support).
    #[arg(long, default_value_t = false)]
    pub symmetric_pd_kv_session_retention: bool,
    /// Keep KV session retention off under `--symmetric-pd-preset paper`, for engines whose
    /// model does not support it (a cache tree that is not FULL-only, e.g. DeepSeek-V4).
    #[arg(long, conflicts_with = "symmetric_pd_kv_session_retention")]
    pub symmetric_pd_no_kv_session_retention: bool,

    /// File containing the engine API/admin key for background KV controls.
    #[arg(long, requires = "symmetric_pd_kv_session_retention")]
    pub symmetric_pd_kv_session_key_file: Option<std::path::PathBuf>,
    /// With `--symmetric-pd-external-role-control`: a worker reassigned from decode
    /// to prefill takes no new legs for up to this many seconds while its D-legs
    /// finish, then aborts the rest (its tokenizers resume them on other workers)
    /// and takes prefill legs. Defaults to 5 with external role control.
    #[arg(long, requires = "symmetric_pd_external_role_control")]
    pub symmetric_pd_role_drain_timeout_secs: Option<u64>,

    /// Decode-peer selection for PD-mode dispatches (orthogonal to
    /// `--policy`, which only selects the prefill worker):
    /// `affinity-min-load` (default — same-host preference, then min
    /// active-load) or `po2` (per-request power-of-two over healthy
    /// decodes), `round-robin`, `min-inflight-tokens` (fewest charged
    /// tokens: prompt + capped output of every dispatched, unfinished
    /// request on that decode; ties fewest requests, then URL), `po2-tokens`
    /// (two healthy samples, fewer charged tokens) or `power-of-two-smg`
    /// (sgl-model-gateway `power_of_two`: two healthy samples, fewer polled
    /// `/v1/loads` tokens when both have a value, else fewer requests).
    /// Applies in plain PD.
    #[arg(long, value_enum)]
    pub pd_decode_policy: Option<PdDecodePolicy>,
    /// `--pd-decode-policy power-of-two-smg` only: seconds between its
    /// `/v1/loads?include=core` polls of every worker (sgl-model-gateway
    /// `--load-check-interval-secs`; default 5; must be > 0).
    #[arg(long)]
    pub pd_decode_load_check_interval_secs: Option<u64>,
    /// Token-charged decode policies: output tokens charged per request,
    /// `min(request max_tokens, N)`, or `N` when the request has no limit
    /// (default 8192).
    #[arg(long)]
    pub pd_decode_tokens_output_cap: Option<u64>,
    /// Token-charged decode policies: compare `charged / max_total_num_tokens`
    /// (each decode's `/server_info` at registration) instead of raw charged
    /// tokens, when every candidate reported a capacity (else equal capacity).
    #[arg(long)]
    pub pd_decode_tokens_normalize_capacity: bool,
    /// Traditional-PD per-request decision replay: JSON table in the
    /// `--symmetric-pd-decision-replay` format (`sessions` / `sessions_by_len`
    /// keyed by sha256(routing key) with `[prefill_url, decode_url]` pairs).
    /// Each request's `--pd-decision-replay-header` value + prompt length
    /// picks its prefill worker and decode peer; unknown keys / lengths fall
    /// back to `--policy` + `--pd-decode-policy`. Every decision logs
    /// `replay=hit|hit_len|hit_len_near|unknown_key|unknown_len|exhausted|
    /// no_key` (target `pd_replay`). Not for `--policy symmetric_pd`.
    #[arg(long, hide = true)]
    pub pd_decision_replay: Option<std::path::PathBuf>,
    /// Header carrying the routing key for `--pd-decision-replay`
    /// (default `X-SMG-Routing-Key`).
    #[arg(long, hide = true)]
    pub pd_decision_replay_header: Option<String>,

    // ---- discovery: static ----
    /// Static worker URLs (space-separated or repeated). Mutually
    /// exclusive with `--service-discovery`.
    #[arg(long, num_args = 1..)]
    pub worker_urls: Vec<String>,

    // ---- discovery: kubernetes ----
    /// Enable Kubernetes EndpointSlice discovery.
    #[arg(long)]
    pub service_discovery: bool,
    /// Namespace to watch. Unset/empty watches all namespaces (requires
    /// cluster-wide RBAC).
    #[arg(long)]
    pub service_discovery_namespace: Option<String>,
    /// Plain-mode label selector terms, e.g. `app=engines-qwen3`
    /// (space-separated or repeated `key=value`, AND-joined). Mutually
    /// exclusive with the prefill/decode selectors.
    #[arg(long, num_args = 1..)]
    pub selector: Vec<String>,
    /// PD-mode prefill label selector terms. Requires `--decode-selector`.
    #[arg(long, num_args = 1..)]
    pub prefill_selector: Vec<String>,
    /// PD-mode decode label selector terms. Requires `--prefill-selector`.
    #[arg(long, num_args = 1..)]
    pub decode_selector: Vec<String>,

    // ---- proxy / active-load ----
    /// Per-request upstream timeout in seconds.
    #[arg(long, default_value_t = default_proxy_request_timeout_secs())]
    pub request_timeout_secs: u64,
    /// TCP keepalive idle time in seconds (`TCP_KEEPIDLE`) for every
    /// upstream socket the proxy opens (prefill and decode legs). Unset:
    /// reqwest's default of 15 s. `30` matches sgl-model-gateway (its
    /// `--tcp-keepalive-secs` default): idle 30 s, probe interval 15 s,
    /// 3 probes, `TCP_USER_TIMEOUT` 30 s. With `TCP_USER_TIMEOUT` set, the
    /// kernel resets a waiting connection at the first keepalive tick that
    /// is at least 30 s after the peer's last ACK while a probe is
    /// unanswered; a larger idle time sends the first probe later.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    pub proxy_tcp_keepalive_secs: Option<u64>,
    /// PD mode: when a request's prefill leg fails (transport error or
    /// non-2xx), drop its decode leg (the decode connection is closed, so
    /// the engine aborts the request and frees its KV) and fail the client
    /// request, as sgl-model-gateway does. Unset: the decode leg waits for
    /// the engine's own KV-wait timeout. Streaming requests whose decode
    /// headers already reached the client end with a cut stream instead of
    /// an HTTP error status.
    #[arg(long)]
    pub pd_abort_decode_on_prefill_failure: bool,
    /// `/v1/completions` request-body cap in MiB (larger bodies get HTTP
    /// 413 before the handler runs). Default 5 (the chat route keeps 5).
    /// Token-id requests with uploaded images can need more (around
    /// 100 MiB for long multi-image requests).
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u64).range(1..=4096))]
    pub max_completions_body_mib: u64,
    /// Screenshot-aware token ids (Kimi K2.5/K2.6): PATH = the engine's
    /// image configuration as JSON (the trace's `metadata.image_config`:
    /// layout_type kimi_k25, image_token_id, patch_size, merge_kernel_size,
    /// in_patch_limit, patch_limit_on_one_side, fixed_output_tokens,
    /// image_mean, image_std). A token-id `/v1/completions` request with
    /// `image_data` data URIs then routes on the engine's radix view (each
    /// image placeholder expanded to its MoonViT token count of its layout
    /// pad value), and the router adds `prepared_image_layout` to the
    /// forwarded body so the engine keys its cache with the same pad
    /// values (engine support: decoupled_engine/image_layout.py, prepared-image replay).
    /// Requests it cannot read route as before (counted in
    /// sgl_router_image_layout_total).
    #[arg(long)]
    pub image_layout_config: Option<std::path::PathBuf>,
    /// Pixel-free late-D decode legs for image requests (needs `--image-layout-config`
    /// and `--pd-late-decode choose` with `--policy symmetric_pd`): URL = this
    /// router's `/internal/prepared_image` as the decode workers reach it
    /// (e.g. `http://10.0.0.1:25800/internal/prepared_image`). The decode
    /// legs of a request with a prepared image layout then carry no
    /// `image_data`, only `prepared_image_source = {endpoint: URL, token:
    /// rid}` (the engine builds pixel-free image items from the layout); a
    /// decode that must recompute fetches the original image bytes from URL
    /// while the request is in flight. Prefill legs are unchanged. Default:
    /// off (decode legs carry the pixels). Counted in
    /// sgl_router_late_d_image_source_total.
    #[arg(long)]
    pub late_d_image_source_url: Option<String>,
    /// Chat requests with `tools`: ask an engine's `/v1/tokenize` for the
    /// token ids and place the request on them (prompt length, KV directory
    /// query, pull hint). The router's chat encoder renders messages only, so
    /// without this such a request matches no cached prefix. The ids are not
    /// forwarded; a failed or slow (3 s) tokenize leaves the request placed
    /// as without this flag. Default: off.
    #[arg(long)]
    pub chat_tools_tokenize: bool,
    /// Late D (engines run `--disaggregation-late-d choose`): `off` (default)
    /// = one decode leg chosen at arrival; `choose` = the request goes to the
    /// prefill and to every decode candidate at once, and the prefill asks
    /// this router for the winner (link to its `late_d.control_port`) when it
    /// is about to prefill. It serves `/v1/completions` and
    /// `/v1/chat/completions`; a native `/generate` request gets its decode
    /// at arrival.
    /// Requires `--pd-decode-policy power-of-two-smg` (the arrival pick that
    /// the prefill admits on a choice timeout) and `--pd-late-decode-rule`.
    /// With `--policy symmetric_pd` (symmetric workers; needs
    /// `--symmetric-pd-force-split`): the P-leg's owner links over its
    /// `disaggregation_late_d_control_port`, the arrival pick is the symmetric_pd
    /// D selection, and the rule is keep, d1 or fit_req.
    #[arg(long, value_enum)]
    pub pd_late_decode: Option<crate::config::LateDecodeMode>,
    /// `--pd-late-decode choose`: the bind-time rule. `d1` =
    /// min-inflight-tokens among the eligible candidates (router-charged
    /// prompt + min(max_tokens, `--pd-decode-tokens-output-cap`) tokens);
    /// `po2` = upstream SMG power_of_two among them on the polled
    /// `late_d.bound_total_tokens`; `keep` = the arrival pick (control);
    /// `fit_req` = the fewest requests among the candidates that fit
    /// (`--pd-late-decode-fit-capacity-tokens`, or the observed capacity
    /// with `--pd-late-decode-fit-native`), `d1` when none fits.
    #[arg(long, value_enum)]
    pub pd_late_decode_rule: Option<crate::config::LateDecodeRule>,
    /// `--pd-late-decode choose`: decode candidates per request, `all`
    /// (default; K = N) or a count N in 1..=64 (the arrival pick and N - 1
    /// others drawn uniformly).
    #[arg(long)]
    pub pd_late_decode_candidates: Option<String>,
    /// `/v1/completions` decode legs carry a token-id prompt as
    /// `prompt_ids_b64` (base64 of little-endian int32 ids) + `prompt_len`
    /// instead of the JSON `prompt` array (not with echo, n > 1,
    /// `strict_replay_deadline_audit` or a batch). Valid with late D off too.
    #[arg(long)]
    pub pd_late_decode_packed_prompt: bool,
    /// `--pd-late-decode-rule fit_req` with a fixed capacity: the KV
    /// capacity of one decode worker in tokens; a candidate fits when its
    /// router-charged tokens plus the request's charge are at most this.
    /// Required by `fit_req` without `--pd-late-decode-fit-native`.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    pub pd_late_decode_fit_capacity_tokens: Option<u64>,
    /// `--pd-late-decode-rule fit_req` on the capacity each decode worker
    /// reports: the fewest allocated requests, then context tokens, then
    /// the most free GPU tokens. Needs `--pd-native-decode-state`.
    #[arg(long)]
    pub pd_late_decode_fit_native: bool,
    /// Poll every decode worker's observed allocation state
    /// (`/v1/loads?include=core,disagg`) every 100 ms. Read-only; it does
    /// not change the `power-of-two-smg` load poll. Valid with late D off.
    #[arg(long)]
    pub pd_native_decode_state: bool,
    /// Maximum age in milliseconds of an observed capacity sample, HTTP
    /// round trip included; an older sample makes the bind use `d1`.
    #[arg(long, default_value_t = crate::config::DEFAULT_LATE_DECODE_FIT_STATE_MAX_AGE_MS)]
    pub pd_late_decode_fit_state_max_age_ms: u64,
    /// sgl-model-gateway `--pd-late-decode-local`: only `off` is accepted
    /// (the owner-local winner of symmetric workers is not ported).
    #[arg(long, value_parser = ["off", "on"], hide = true)]
    pub pd_late_decode_local: Option<String>,
    /// Max lifetime of an in-flight request entry before the janitor
    /// reaps it (returns 504 `stale_request_expired`).
    #[arg(long, default_value_t = default_stale_request_timeout_secs())]
    pub stale_request_timeout_secs: u64,

    // ---- observability ----
    /// Default tracing level (overridden by `RUST_LOG`).
    #[arg(long, default_value = "info")]
    pub log_level: String,
    /// Log output format.
    #[arg(long, value_enum, default_value = "text")]
    pub log_format: LogFormat,
}

impl Cli {
    /// Resolve the `--pd-late-decode*` flags. Off by default; orphan knobs
    /// are rejected. Choose mode needs a rule, a candidate count in 1..=64
    /// (the engine's limit), the `power-of-two-smg` decode policy (the
    /// arrival pick, which the prefill admits when no decision comes in time,
    /// and the load monitor whose `bound_total_tokens` the `po2` rule reads),
    /// and no decision replay. symmetric_pd (symmetric owners) needs force split,
    /// no `po2` rule and no symmetric decision replay (external role control is
    /// accepted), and no `power-of-two-smg` (its arrival pick is the symmetric_pd
    /// D selection).
    fn build_late_decode(
        &self,
        pd_decode_policy: PdDecodePolicy,
    ) -> Result<crate::config::LateDecodeConfig> {
        use crate::config::{LateDecodeConfig, LateDecodeMode, LateDecodeRule};
        let mode = self.pd_late_decode.unwrap_or_default();
        let packed_prompt = self.pd_late_decode_packed_prompt;
        if self.pd_late_decode_local.as_deref() == Some("on") {
            return Err(anyhow!(
                "--pd-late-decode-local on is not available (the owner-local winner of symmetric \
                 workers is not ported); use off"
            ));
        }
        if mode != LateDecodeMode::Choose {
            if self.pd_late_decode_rule.is_some()
                || self.pd_late_decode_candidates.is_some()
                || self.pd_late_decode_fit_capacity_tokens.is_some()
                || self.pd_late_decode_fit_native
            {
                return Err(anyhow!(
                    "--pd-late-decode-rule / --pd-late-decode-candidates / \
                     --pd-late-decode-fit-capacity-tokens / --pd-late-decode-fit-native require \
                     --pd-late-decode choose"
                ));
            }
            return Ok(LateDecodeConfig {
                packed_prompt,
                native_state_poll: self.pd_native_decode_state,
                fit_state_max_age_ms: self.pd_late_decode_fit_state_max_age_ms,
                ..LateDecodeConfig::default()
            });
        }
        let rule = self
            .pd_late_decode_rule
            .ok_or_else(|| anyhow!("--pd-late-decode choose requires --pd-late-decode-rule"))?;
        let candidates = match self.pd_late_decode_candidates.as_deref().map(str::trim) {
            None | Some("all") => None,
            Some(n) => match n.parse::<usize>() {
                Ok(k) if (1..=crate::late_bind::MAX_CANDIDATES).contains(&k) => Some(k),
                _ => {
                    return Err(anyhow!(
                        "--pd-late-decode-candidates must be all or a count in 1..={}, got {n:?}",
                        crate::late_bind::MAX_CANDIDATES
                    ))
                }
            },
        };
        let symmetric_pd = self.policy == PolicyKind::SymmetricPd;
        if symmetric_pd {
            // Symmetric owners (symmetric_pd): the arrival pick is the symmetric_pd
            // D selection; LOCAL is not offered, so every request splits.
            if !self.symmetric_pd_force_split {
                return Err(anyhow!(
                    "--pd-late-decode choose with --policy symmetric_pd requires --symmetric-pd-force-split \
                     (the owner-local winner is not ported)"
                ));
            }
            if rule == LateDecodeRule::Po2 {
                return Err(anyhow!(
                    "--pd-late-decode-rule po2 is not available with --policy symmetric_pd (no polled \
                     bound view of symmetric workers); use keep, d1 or fit_req"
                ));
            }
            // External role control is accepted: the candidates are the
            // decode-role workers at admission, a bind chooses among those
            // still in the decode role, and prepared executions track the
            // candidate legs (backend/execution.rs).
            if self.symmetric_pd_decision_replay.is_some() {
                return Err(anyhow!(
                    "--pd-late-decode choose with --policy symmetric_pd conflicts with \
                     --symmetric-pd-decision-replay"
                ));
            }
        }
        if self.pd_decision_replay.is_some() {
            return Err(anyhow!(
                "--pd-late-decode choose conflicts with --pd-decision-replay (the replay pins the decode peer)"
            ));
        }
        if !symmetric_pd && pd_decode_policy != PdDecodePolicy::PowerOfTwoSmg {
            return Err(anyhow!(
                "--pd-late-decode choose requires --pd-decode-policy power-of-two-smg (the arrival pick \
                 the prefill admits when no decision comes in time)"
            ));
        }
        // sgl-model-gateway `validate_fit_capacity`: the observed-capacity
        // fit needs fit_req, the capacity poll, a positive sample age and no
        // fixed capacity; the fixed capacity goes with fit_req only.
        let fit_req = rule == LateDecodeRule::FitReq;
        if self.pd_late_decode_fit_native {
            if !fit_req
                || !self.pd_native_decode_state
                || self.pd_late_decode_fit_capacity_tokens.is_some()
                || self.pd_late_decode_fit_state_max_age_ms == 0
            {
                return Err(anyhow!(
                    "--pd-late-decode-fit-native needs --pd-late-decode-rule fit_req, \
                     --pd-native-decode-state, a positive --pd-late-decode-fit-state-max-age-ms \
                     and no --pd-late-decode-fit-capacity-tokens"
                ));
            }
        } else {
            match (self.pd_late_decode_fit_capacity_tokens, fit_req) {
                (Some(_), false) => {
                    return Err(anyhow!(
                        "--pd-late-decode-fit-capacity-tokens is used only by \
                         --pd-late-decode-rule fit_req"
                    ))
                }
                (None, true) => {
                    return Err(anyhow!(
                    "--pd-late-decode-rule fit_req requires --pd-late-decode-fit-capacity-tokens \
                         (fixed capacity) or --pd-late-decode-fit-native (observed capacity)"
                ))
                }
                _ => {}
            }
        }
        Ok(LateDecodeConfig {
            mode,
            rule: Some(rule),
            candidates,
            packed_prompt,
            fit_capacity_tokens: self.pd_late_decode_fit_capacity_tokens,
            fit_native: self.pd_late_decode_fit_native,
            native_state_poll: self.pd_native_decode_state,
            fit_state_max_age_ms: self.pd_late_decode_fit_state_max_age_ms,
        })
    }

    /// True when any flag of the placement cost function (its model, the
    /// forecast, the placement trace and the progress poll) was given.
    fn cost_flag_given(&self) -> bool {
        self.prefill_rate_tokens_per_s.is_some()
            || self.fetch_fixed_ms.is_some()
            || self.fetch_ms_per_100k_tokens.is_some()
            || self.copy_price.is_some()
            || self.prefill_wait.is_some()
            || self.max_state_age_ms.is_some()
            || self.forecast_error_table.is_some()
            || self.move_margin.is_some()
            || self.forecast_push_port.is_some()
            || self.forecast_next_pass_lag.is_some()
            || self.forecast_fit.is_some()
            || self.copy_price_min.is_some()
            || self.copy_price_max.is_some()
            || self.copy_price_fallback.is_some()
            || self.placement_trace.is_some()
            || self.prefill_poll_ms.is_some()
    }

    /// `--symmetric-pd-preset paper`: set every flag of the paper's
    /// configuration that is still unset. Features whose inputs are missing
    /// stay off: the cost function needs the KV directory and the servicers.
    fn apply_symmetric_pd_preset(&mut self) {
        use crate::config::{LateDecodeMode, LateDecodeRule};
        if self.policy != PolicyKind::SymmetricPd
            || self.symmetric_pd_preset != SymmetricPdPreset::Paper
        {
            return;
        }
        self.symmetric_pd_force_split = true;
        self.symmetric_pd_d_select
            .get_or_insert(SymmetricPdDSelect::RoundRobin);
        self.symmetric_pd_load_poll_ms.get_or_insert(100);
        if self.pd_late_decode.is_none() && self.symmetric_pd_decision_replay.is_none() {
            self.pd_late_decode = Some(LateDecodeMode::Choose);
            let rule = *self
                .pd_late_decode_rule
                .get_or_insert(LateDecodeRule::FitReq);
            self.pd_late_decode_candidates
                .get_or_insert_with(|| "all".into());
            self.pd_late_decode_local
                .get_or_insert_with(|| "off".into());
            if rule == LateDecodeRule::FitReq && self.pd_late_decode_fit_capacity_tokens.is_none() {
                self.pd_late_decode_fit_native = true;
                self.pd_native_decode_state = true;
            }
        }
        self.symmetric_pd_kv_session_retention = !self.symmetric_pd_no_kv_session_retention;
        // A worker that serves decode now may serve prefill after a role
        // change, so imported KV keeps the engine's host write policy.
        self.symmetric_pd_import_cache_policy
            .get_or_insert(SymmetricPdImportCachePolicy::Retain);
        if self.symmetric_pd_external_role_control {
            self.symmetric_pd_role_drain_timeout_secs
                .get_or_insert(DEFAULT_ROLE_DRAIN_TIMEOUT_SECS);
        }
        if self.kv_indexer_url.is_some() {
            self.kv_indexer_timeout_ms
                .get_or_insert(PAPER_PRESET_KV_HINT_INDEXER_TIMEOUT_MS);
        }
        let servicers = self
            .kv_worker_map
            .as_deref()
            .is_some_and(|map| map.contains("servicer"));
        if self.kv_indexer_url.is_some() && servicers && self.symmetric_pd_p_select.is_none() {
            self.symmetric_pd_p_select = Some(SymmetricPdPSelect::CostFunction);
            self.symmetric_pd_pull_hint = true;
            self.symmetric_pd_pull_from_any_holder = true;
            self.prefill_poll_ms
                .get_or_insert(DEFAULT_COST_FUNCTION_LOAD_POLL_MS);
            self.kv_pull_hint_min_incremental_tokens
                .get_or_insert(crate::scheduler::placement_cost::MIN_FETCH_TOKENS as u32);
        }
        // The cost function waits on the forecast and holds requests in the
        // global queue; the paper's runs pinned `--prefill-wait tokens` and
        // `--global-queue=false`.
        if self.symmetric_pd_p_select == Some(SymmetricPdPSelect::CostFunction) {
            let forecast_wait =
                self.prefill_wait.get_or_insert_with(|| "forecast".into()) == "forecast";
            let room_forecast = *self.global_queue.get_or_insert(true)
                && self
                    .global_queue_room
                    .get_or_insert_with(|| "forecast".into())
                    == "forecast";
            if forecast_wait || room_forecast {
                self.forecast_next_pass_lag.get_or_insert(true);
            }
            if forecast_wait {
                self.move_margin.get_or_insert(0.0);
                self.max_state_age_ms
                    .get_or_insert(PAPER_PRESET_MAX_STATE_AGE_MS);
            }
        }
    }

    /// Resolve parsed flags into a validated [`Config`].
    ///
    /// Builds the [`DiscoveryBackend`] (enforcing static-vs-k8s mutual
    /// exclusivity and resolving the k8s selector grammar via
    /// [`resolve_mode`]), assembles the single [`ModelConfig`], then runs
    /// [`Config::validate`] for the remaining value-level invariants
    /// (model id, static worker URLs).
    pub fn into_config(mut self) -> Result<Config> {
        self.apply_symmetric_pd_preset();
        let mut symmetric_pd = self.build_symmetric_pd()?;
        // `--symmetric-pd-p-select cost-function`: the cost-function flags of the
        // cost function configure the symmetric_pd P selection.
        let spd_cost_function = symmetric_pd.is_some()
            && self.symmetric_pd_p_select == Some(SymmetricPdPSelect::CostFunction);
        let gateway_cache_aware = self.build_gateway_cache_aware()?;
        let cost_flag_given = self.cost_flag_given();
        let pd_decision_replay = match &self.pd_decision_replay {
            Some(p) => {
                if self.policy == PolicyKind::SymmetricPd {
                    return Err(anyhow!(
                        "--pd-decision-replay is for the traditional PD path; use --symmetric-pd-decision-replay with --policy symmetric_pd"
                    ));
                }
                if !p.is_file() {
                    return Err(anyhow!(
                        "--pd-decision-replay {}: not a readable file",
                        p.display()
                    ));
                }
                let header = resolve_decision_replay_header(
                    self.pd_decision_replay_header.as_deref(),
                    "--pd-decision-replay-header",
                )?;
                Some(PdDecisionReplayConfig {
                    table: p.clone(),
                    header,
                })
            }
            None => {
                if self.pd_decision_replay_header.is_some() {
                    return Err(anyhow!(
                        "--pd-decision-replay-header requires --pd-decision-replay"
                    ));
                }
                None
            }
        };
        let discovery = self.build_discovery(symmetric_pd.as_ref())?;
        let pd_decode_policy = self
            .pd_decode_policy
            .unwrap_or(PdDecodePolicy::AffinityMinLoad);
        let late_decode = self.build_late_decode(pd_decode_policy)?;
        // `--image-layout-config`: validated once, lives for the process.
        let image_layout: Option<&'static crate::image_layout::ImageLayoutConfig> =
            match &self.image_layout_config {
                Some(path) => Some(Box::leak(Box::new(
                    crate::image_layout::ImageLayoutConfig::load(path)?,
                ))),
                None => None,
            };
        // `--late-d-image-source-url`: validated once, leaked (lives for the process).
        let late_d_image_source_url: Option<&'static str> = match &self.late_d_image_source_url {
            Some(url) => {
                if image_layout.is_none() {
                    return Err(anyhow!(
                        "--late-d-image-source-url requires --image-layout-config"
                    ));
                }
                if !late_decode.choose() || self.policy != PolicyKind::SymmetricPd {
                    return Err(anyhow!(
                        "--late-d-image-source-url requires --pd-late-decode choose with --policy symmetric_pd"
                    ));
                }
                if !(url.starts_with("http://") || url.starts_with("https://")) {
                    return Err(anyhow!(
                        "--late-d-image-source-url must be an http(s) URL, got {url:?}"
                    ));
                }
                Some(Box::leak(url.clone().into_boxed_str()))
            }
            None => None,
        };
        // The late-D `d1` and `fit_req` rules pick on the token-charged
        // decode ledger, so its tuning flags configure that charge too.
        let late_d1 = matches!(
            late_decode.rule,
            Some(crate::config::LateDecodeRule::D1 | crate::config::LateDecodeRule::FitReq)
        );
        let pd_decode_tokens = if pd_decode_policy.is_token_charged() || late_d1 {
            Some(PdDecodeTokensConfig {
                output_charge_cap: self
                    .pd_decode_tokens_output_cap
                    .unwrap_or(crate::config::types::DEFAULT_PD_DECODE_OUTPUT_CHARGE_CAP),
                normalize_by_capacity: self.pd_decode_tokens_normalize_capacity,
            })
        } else {
            if self.pd_decode_tokens_output_cap.is_some()
                || self.pd_decode_tokens_normalize_capacity
            {
                return Err(anyhow!(
                    "--pd-decode-tokens-* flags require --pd-decode-policy min-inflight-tokens or po2-tokens \
                     (or --pd-late-decode-rule d1 or fit_req)"
                ));
            }
            None
        };
        let pd_decode_load_check_interval_secs = match self.pd_decode_load_check_interval_secs {
            Some(_) if pd_decode_policy != PdDecodePolicy::PowerOfTwoSmg => {
                return Err(anyhow!(
                    "--pd-decode-load-check-interval-secs requires --pd-decode-policy power-of-two-smg"
                ));
            }
            // sgl-model-gateway config validation: load_check_interval_secs "Must be > 0".
            Some(0) => {
                return Err(anyhow!("--pd-decode-load-check-interval-secs must be > 0"));
            }
            Some(secs) => secs,
            None => crate::config::types::DEFAULT_PD_DECODE_LOAD_CHECK_INTERVAL_SECS,
        };

        // Reject knobs that only take effect alongside another flag, rather
        // than silently dropping them — mirrors the discovery mutual-exclusion
        // checks. Otherwise an operator believes they tuned something that has
        // no effect.
        if self.disable_circuit_breaker
            && (self.cb_threshold.is_some() || self.cb_cool_down_secs.is_some())
        {
            return Err(anyhow!(
                "--disable-circuit-breaker conflicts with --cb-threshold / --cb-cool-down-secs"
            ));
        }
        if self.cb_cool_down_secs.is_some() && self.cb_threshold.is_none() {
            return Err(anyhow!(
                "--cb-cool-down-secs requires --cb-threshold (the circuit breaker is \
                 enabled by --cb-threshold)"
            ));
        }
        let tuned_cache_aware = self.dynamo_cache_aware
            || self.cache_threshold.is_some()
            || self.balance_abs_threshold.is_some()
            || self.balance_rel_threshold.is_some()
            || self.kv_indexer_endpoint.is_some()
            || self.kv_indexer_query_timeout_ms.is_some()
            || self.kv_indexer_query_max_inflight.is_some();
        if tuned_cache_aware && self.policy != PolicyKind::CacheAwareZmq {
            return Err(anyhow!(
                "cache-aware tuning flags require --policy cache_aware_zmq"
            ));
        }
        if self.dynamo_cache_aware
            && (self.inject_kv_pull_hint || self.kv_indexer_endpoint.is_some())
        {
            return Err(anyhow!(
                "Dynamo cache-aware uses one indexer and does not inject cross-P pulls"
            ));
        }
        if self.kv_indexer_query_timeout_ms == Some(0) {
            return Err(anyhow!(
                "--kv-indexer-query-timeout-ms must be greater than zero"
            ));
        }
        if self.kv_indexer_query_timeout_ms.is_some() && self.kv_indexer_endpoint.is_none() {
            return Err(anyhow!(
                "--kv-indexer-query-timeout-ms requires --kv-indexer-endpoint"
            ));
        }
        if self.kv_indexer_query_max_inflight == Some(0) {
            return Err(anyhow!(
                "--kv-indexer-query-max-inflight must be greater than zero"
            ));
        }
        if self.kv_indexer_query_max_inflight.is_some() && self.kv_indexer_endpoint.is_none() {
            return Err(anyhow!(
                "--kv-indexer-query-max-inflight requires --kv-indexer-endpoint"
            ));
        }
        let tuned_sticky = self.routing_key_header.is_some()
            || self.sticky_fallback_policy.is_some()
            || self.sticky_idle_secs.is_some()
            || self.sticky_eviction_interval_secs.is_some();
        if tuned_sticky && self.policy != PolicyKind::Sticky {
            return Err(anyhow!(
                "--routing-key-header / --sticky-fallback-policy / --sticky-idle-secs / \
                 --sticky-eviction-interval-secs require --policy sticky"
            ));
        }

        // Build (and validate) the sticky config exactly when the sticky
        // policy is selected. The header name must parse as an HTTP header
        // name so a typo fails at startup rather than silently never
        // matching any request header; the fallback must be a
        // dependency-free policy the factory can build standalone.
        let sticky = if self.policy == PolicyKind::Sticky {
            let d = StickyConfig::default();
            let header_name = self.routing_key_header.unwrap_or(d.header_name);
            axum::http::HeaderName::try_from(header_name.as_str()).map_err(|e| {
                anyhow!("--routing-key-header {header_name:?} is not a valid HTTP header name: {e}")
            })?;
            let fallback_policy = self.sticky_fallback_policy.unwrap_or(d.fallback_policy);
            if matches!(
                fallback_policy,
                PolicyKind::Sticky
                    | PolicyKind::CacheAwareZmq
                    | PolicyKind::GatewayCacheAware
                    | PolicyKind::SymmetricPd
            ) {
                return Err(anyhow!(
                    "--sticky-fallback-policy must be one of round_robin / random / \
                     power_of_two / load_based"
                ));
            }
            let idle_secs = self.sticky_idle_secs.unwrap_or(d.idle_secs);
            let eviction_interval_secs = self
                .sticky_eviction_interval_secs
                .unwrap_or(d.eviction_interval_secs);
            // Reject zero durations: `--sticky-eviction-interval-secs 0` would
            // panic `tokio::time::interval` at startup, and `--sticky-idle-secs
            // 0` would evict every assignment on the next sweep (defeating
            // stickiness entirely). Fail fast with a clear message instead.
            if eviction_interval_secs == 0 {
                return Err(anyhow!(
                    "--sticky-eviction-interval-secs must be greater than 0"
                ));
            }
            if idle_secs == 0 {
                return Err(anyhow!(
                    "--sticky-idle-secs must be greater than 0 (0 would evict every \
                     assignment immediately, defeating sticky routing)"
                ));
            }
            Some(StickyConfig {
                header_name,
                fallback_policy,
                idle_secs,
                eviction_interval_secs,
            })
        } else {
            None
        };

        // The flags of the placement cost function configure
        // `--symmetric-pd-p-select cost-function`.
        if cost_flag_given && !spd_cost_function {
            return Err(anyhow!(
                "the cost-function flags (--copy-price, --prefill-wait, --forecast-*, ...) \
                 require --policy symmetric_pd with --symmetric-pd-p-select cost-function"
            ));
        }
        // Session KV retention on the gateway_cache_aware PD path. symmetric_pd
        // keeps its own registry and flag (--symmetric-pd-kv-session-retention).
        let gateway_ca_selected = self.policy == PolicyKind::GatewayCacheAware;
        if self.kv_session_retention && !gateway_ca_selected {
            return Err(if self.policy == PolicyKind::SymmetricPd {
                anyhow!(
                    "--kv-session-retention is not available with --policy symmetric_pd; \
                     use --symmetric-pd-kv-session-retention"
                )
            } else {
                anyhow!("--kv-session-retention requires --policy gateway_cache_aware")
            });
        }
        let mut forecast = None;
        let cost_params = if spd_cost_function {
            let floor = crate::scheduler::placement_cost::MIN_FETCH_TOKENS;
            if self
                .kv_pull_hint_min_incremental_tokens
                .is_some_and(|n| u64::from(n) < floor)
            {
                return Err(anyhow!(
                    "the cost function never pulls fewer than {floor} tokens: \
                     --kv-pull-hint-min-incremental-tokens must be at least {floor} \
                     (absent = {floor})"
                ));
            }
            if self.kv_pull_hint_local_prefix_fallback {
                return Err(anyhow!(
                    "the cost function sends a hint only for a pull of at least the floor; \
                     it cannot be combined with --kv-pull-hint-local-prefix-fallback"
                ));
            }
            let defaults = crate::scheduler::placement_cost::PlacementCostParams::default();
            let lambda_auto = self.copy_price == Some(NumberOrAuto::Auto);
            let rate_auto = self.prefill_rate_tokens_per_s == Some(NumberOrAuto::Auto);
            if !lambda_auto
                && (self.copy_price_min.is_some()
                    || self.copy_price_max.is_some()
                    || self.copy_price_fallback.is_some())
            {
                return Err(anyhow!(
                    "--copy-price-min, -max and -fallback need --copy-price auto"
                ));
            }
            let auto = (
                self.copy_price_min
                    .unwrap_or(crate::scheduler::placement_cost::DEFAULT_COPY_PRICE_MIN),
                self.copy_price_max
                    .unwrap_or(crate::scheduler::placement_cost::DEFAULT_COPY_PRICE_MAX),
            );
            for (name, value) in [
                ("--copy-price-min", auto.0),
                ("--copy-price-max", auto.1),
                (
                    "--copy-price-fallback",
                    self.copy_price_fallback.unwrap_or(defaults.copy_price),
                ),
            ] {
                if !(value.is_finite() && value >= 0.0) {
                    return Err(anyhow!("{name} must be finite and at least 0"));
                }
            }
            if auto.0 > auto.1 {
                return Err(anyhow!(
                    "--copy-price-min must not be above --copy-price-max"
                ));
            }
            let params = crate::scheduler::placement_cost::PlacementCostParams {
                // auto: the default R is the fallback.
                prefill_rate_tokens_per_s: match self.prefill_rate_tokens_per_s {
                    Some(NumberOrAuto::Number(rate)) => rate,
                    _ => defaults.prefill_rate_tokens_per_s,
                },
                fetch_fixed_ms: self.fetch_fixed_ms.unwrap_or(defaults.fetch_fixed_ms),
                fetch_ms_per_100k_tokens: self
                    .fetch_ms_per_100k_tokens
                    .unwrap_or(defaults.fetch_ms_per_100k_tokens),
                // auto: the fallback price.
                copy_price: match self.copy_price {
                    Some(NumberOrAuto::Number(price)) => price,
                    Some(NumberOrAuto::Auto) => {
                        self.copy_price_fallback.unwrap_or(defaults.copy_price)
                    }
                    None => defaults.copy_price,
                },
                copy_price_mode: if lambda_auto {
                    crate::scheduler::placement_cost::CopyPriceMode::Auto {
                        min: auto.0,
                        max: auto.1,
                    }
                } else {
                    crate::scheduler::placement_cost::CopyPriceMode::Fixed
                },
                prefill_rate_mode: if rate_auto {
                    crate::scheduler::placement_cost::PrefillRateMode::Auto
                } else {
                    crate::scheduler::placement_cost::PrefillRateMode::Fixed
                },
                wait: match self.prefill_wait.as_deref() {
                    Some("forecast") => crate::scheduler::placement_cost::PrefillWait::Forecast,
                    _ => crate::scheduler::placement_cost::PrefillWait::Tokens,
                },
                max_state_age_s: self
                    .max_state_age_ms
                    .map_or(defaults.max_state_age_s, |ms| ms / 1000.0),
                move_margin: self.move_margin.unwrap_or(defaults.move_margin),
                ..defaults
            };
            if !(params.prefill_rate_tokens_per_s.is_finite()
                && params.prefill_rate_tokens_per_s > 0.0)
            {
                return Err(anyhow!(
                    "--prefill-rate-tokens-per-s must be finite and greater than 0"
                ));
            }
            for (name, value) in [
                ("--fetch-fixed-ms", params.fetch_fixed_ms),
                (
                    "--fetch-ms-per-100k-tokens",
                    params.fetch_ms_per_100k_tokens,
                ),
                ("--copy-price", params.copy_price),
                ("--move-margin", params.move_margin),
            ] {
                if !(value.is_finite() && value >= 0.0) {
                    return Err(anyhow!("{name} must be finite and at least 0"));
                }
            }
            let forecast_wait =
                params.wait == crate::scheduler::placement_cost::PrefillWait::Forecast;
            if !(params.max_state_age_s.is_finite() && params.max_state_age_s > 0.0) {
                return Err(anyhow!(
                    "--max-state-age-ms must be finite and greater than 0"
                ));
            }
            if !forecast_wait && self.max_state_age_ms.is_some() {
                return Err(anyhow!("--max-state-age-ms needs --prefill-wait forecast"));
            }
            let forecast_flags = self.forecast_error_table.is_some()
                || self.move_margin.is_some()
                || self.forecast_push_port.is_some()
                || self.forecast_next_pass_lag == Some(true)
                || self.forecast_fit.is_some();
            // `--global-queue-room forecast` needs the forecast state also
            // with the token-count wait (the drain only; no move rule).
            let room_forecast = self.global_queue == Some(true)
                && self.global_queue_room.as_deref() == Some("forecast");
            if !forecast_wait && self.move_margin.is_some() {
                return Err(anyhow!("--move-margin needs --prefill-wait forecast"));
            }
            if forecast_wait || room_forecast {
                let path = self
                    .forecast_error_table
                    .as_deref()
                    .unwrap_or(std::path::Path::new(
                        crate::prefill_forecast::BUILTIN_ERROR_TABLE_PATH,
                    ));
                let config = crate::prefill_forecast::ForecastConfig::load(
                    path,
                    crate::prefill_forecast::DEFAULT_BOUND_MODE,
                    None,
                    0.0,
                    None,
                )
                .map_err(|e| anyhow!("--forecast-error-table: {e}"))?;
                let config = match self.forecast_push_port {
                    Some(port) => config
                        .with_push(port, None, None)
                        .map_err(|e| anyhow!("--forecast-push-port: {e}"))?,
                    None => config,
                };
                let config = config.with_next_pass_lag(self.forecast_next_pass_lag == Some(true));
                let frozen = self
                    .forecast_fit
                    .as_deref()
                    .map(crate::prefill_forecast::ForecastFit::load)
                    .transpose()
                    .map_err(|e| anyhow!("--forecast-fit: {e}"))?
                    .map(std::sync::Arc::new);
                forecast = Some(std::sync::Arc::new(config.with_frozen_fit(frozen)));
            } else if forecast_flags {
                return Err(anyhow!(
                    "the --forecast-* flags need --prefill-wait forecast or \
                     --global-queue-room forecast"
                ));
            }
            Some(params)
        } else {
            None
        };
        // The cost-function parameters, the placement trace and the progress
        // poll period go to the symmetric_pd config.
        if let (Some(cost_params), Some(spd)) = (cost_params, symmetric_pd.as_mut()) {
            let load_poll_ms = self
                .prefill_poll_ms
                .unwrap_or(DEFAULT_COST_FUNCTION_LOAD_POLL_MS);
            if load_poll_ms == 0 {
                return Err(anyhow!("--prefill-poll-ms must be greater than 0"));
            }
            spd.cost_function = Some(SymmetricPdCostFunctionConfig {
                cost_params,
                placement_trace: self.placement_trace.clone(),
                load_poll_ms,
                unknown_expire_secs: self
                    .symmetric_pd_unknown_expire_secs
                    .unwrap_or_else(crate::config::types::default_symmetric_pd_unknown_expire_secs),
                global_queue: (self.global_queue == Some(true)).then(|| {
                    crate::config::GlobalQueueConfig {
                        room: if self.global_queue_room.as_deref() == Some("forecast") {
                            crate::global_queue::GlobalQueueRoom::Forecast
                        } else {
                            crate::global_queue::GlobalQueueRoom::Tokens
                        },
                        ..Default::default()
                    }
                }),
                trace_session_moves: self.symmetric_pd_trace_session_moves,
                forecast: forecast.clone().map(crate::config::SymmetricPdForecast),
                pull_from_any_holder: self.symmetric_pd_pull_from_any_holder,
            });
        }

        // KV pull-hint injection: reject orphan tuning knobs (mirrors the
        // sticky / cache-aware checks above), then validate the required
        // pieces at startup so a malformed --kv-worker-map fails fast
        // instead of silently never injecting.
        let tuned_kv_hint = self.kv_indexer_url.is_some()
            || self.kv_pull_hint_min_tokens.is_some()
            || self.kv_pull_hint_min_incremental_tokens.is_some()
            || self.kv_pull_hint_local_prefix_fallback
            || self.kv_indexer_timeout_ms.is_some()
            || self.kv_worker_map.is_some();
        // symmetric_pd: the indexer is optional (placement degrades to load /
        // role only, logged), so the flags are accepted but not required.
        let spd_selected = symmetric_pd.is_some();
        if tuned_kv_hint && !self.inject_kv_pull_hint && !spd_selected && !self.dynamo_cache_aware {
            return Err(anyhow!(
                "--kv-indexer-url / --kv-pull-hint-min-tokens / \
                 --kv-pull-hint-min-incremental-tokens / --kv-pull-hint-local-prefix-fallback / --kv-indexer-timeout-ms / \
                 --kv-worker-map require --inject-kv-pull-hint or --policy symmetric_pd"
            ));
        }
        let spd_wants_indexer = spd_selected && self.kv_indexer_url.is_some();
        let mut kv_hint =
            if self.inject_kv_pull_hint || spd_wants_indexer || self.dynamo_cache_aware {
                let requirer = if self.inject_kv_pull_hint {
                    "--inject-kv-pull-hint"
                } else if self.dynamo_cache_aware {
                    "--dynamo-cache-aware"
                } else {
                    "--policy symmetric_pd with --kv-indexer-url"
                };
                let indexer_url = self
                    .kv_indexer_url
                    .ok_or_else(|| anyhow!("{requirer} requires --kv-indexer-url"))?;
                let map_json = self
                    .kv_worker_map
                    .ok_or_else(|| anyhow!("{requirer} requires --kv-worker-map"))?;
                let worker_map = parse_kv_worker_map(&map_json)?;
                let timeout_ms = self
                    .kv_indexer_timeout_ms
                    .unwrap_or(DEFAULT_KV_HINT_INDEXER_TIMEOUT_MS);
                if timeout_ms == 0 {
                    return Err(anyhow!("--kv-indexer-timeout-ms must be greater than 0"));
                }
                Some(crate::kv_hint::KvHintConfig {
                    indexer_url,
                    min_tokens: self
                        .kv_pull_hint_min_tokens
                        .unwrap_or(DEFAULT_KV_PULL_HINT_MIN_TOKENS),
                    // cost-function never pulls below 512, so its gate defaults to 512.
                    min_incremental_tokens: self.kv_pull_hint_min_incremental_tokens.unwrap_or(
                        if spd_cost_function {
                            crate::scheduler::placement_cost::MIN_FETCH_TOKENS as u32
                        } else {
                            0
                        },
                    ),
                    local_prefix_fallback: self.kv_pull_hint_local_prefix_fallback,
                    indexer_timeout_ms: timeout_ms,
                    worker_map,
                })
            } else {
                None
            };
        // `--symmetric-pd-p-select cost-function`: the cost function reads the
        // indexer, and every P-role worker can be a pull source, so each needs
        // its servicer. D-role workers may stay out of the map.
        if let Some(spd) = symmetric_pd.as_ref().filter(|_| spd_cost_function) {
            let kv = kv_hint.as_ref().ok_or_else(|| {
                anyhow!(
                    "--symmetric-pd-p-select cost-function requires --kv-indexer-url and \
                     --kv-worker-map"
                )
            })?;
            let p_roles: Vec<&SymmetricPdWorkerSpec> = spd
                .workers
                .iter()
                .filter(|w| w.prefill_weight > 0.0 && w.capabilities().prefill)
                .collect();
            if p_roles.is_empty() {
                return Err(anyhow!(
                    "--symmetric-pd-p-select cost-function needs the P-role workers (prefill weight \
                     > 0) in --symmetric-pd-worker or --symmetric-pd-workers-file"
                ));
            }
            for worker in p_roles {
                let url = worker.normalized_url();
                let mapped = kv.worker_map.iter().any(|entry| {
                    entry.url.trim().trim_end_matches('/') == url
                        && !entry.servicer.trim().is_empty()
                });
                if !mapped {
                    return Err(anyhow!(
                        "--symmetric-pd-p-select cost-function: P-role worker {url} needs a \
                         --kv-worker-map entry with a servicer"
                    ));
                }
            }
        }

        let circuit_breaker = self.cb_threshold.map(|threshold| CircuitBreakerConfig {
            threshold,
            cool_down_secs: self.cb_cool_down_secs.unwrap_or_else(default_cb_cool_down),
        });

        // Only build a CacheAwareConfig when the operator tuned at least
        // one knob; otherwise leave it None so the policy uses its own
        // defaults. Unset knobs fall back to the per-field defaults.
        let kv_indexer_query_timeout_ms = self
            .kv_indexer_query_timeout_ms
            .unwrap_or(DEFAULT_KV_INDEXER_QUERY_TIMEOUT_MS);
        let kv_indexer_query_max_inflight = self
            .kv_indexer_query_max_inflight
            .unwrap_or(DEFAULT_KV_INDEXER_QUERY_MAX_INFLIGHT);
        let cache_aware = if tuned_cache_aware {
            let d = CacheAwareConfig::default();
            Some(CacheAwareConfig {
                dynamo: if self.dynamo_cache_aware {
                    kv_hint.take()
                } else {
                    None
                },
                cache_threshold: self.cache_threshold.unwrap_or(d.cache_threshold),
                balance_abs_threshold: self
                    .balance_abs_threshold
                    .unwrap_or(d.balance_abs_threshold),
                balance_rel_threshold: self
                    .balance_rel_threshold
                    .unwrap_or(d.balance_rel_threshold),
                kv_indexer_endpoint: self.kv_indexer_endpoint.map(|url| KvIndexerEndpointConfig {
                    url,
                    query_timeout_ms: kv_indexer_query_timeout_ms,
                    query_max_inflight: kv_indexer_query_max_inflight,
                }),
            })
        } else {
            None
        };

        let config = Config {
            server: ServerConfig {
                host: self.host,
                port: self.port,
            },
            observability: ObservabilityConfig {
                log_level: self.log_level,
                log_format: self.log_format,
            },
            model: ModelConfig {
                // Default the tokenizer source to the model id (treated as a
                // HuggingFace repo id) when --tokenizer-path is omitted.
                tokenizer_path: self.tokenizer_path.unwrap_or_else(|| self.model_id.clone()),
                id: self.model_id,
                policy: self.policy,
                circuit_breaker,
                cache_aware,
                sticky,
                kv_hint,
                pd_decode_policy,
                pd_decode_tokens,
                pd_decision_replay,
                symmetric_pd,
                gateway_cache_aware,
            },
            discovery,
            proxy: ProxyConfig {
                request_timeout_secs: self.request_timeout_secs,
                disable_circuit_breaker: self.disable_circuit_breaker,
                tcp_keepalive_secs: self.proxy_tcp_keepalive_secs,
                abort_decode_on_prefill_failure: self.pd_abort_decode_on_prefill_failure,
                late_decode,
                max_completions_body_bytes: (self.max_completions_body_mib as usize) << 20,
                image_layout,
                late_d_image_source_url,
                chat_tools_tokenize: self.chat_tools_tokenize,
            },
            active_load: ActiveLoadConfig {
                stale_request_timeout_secs: self.stale_request_timeout_secs,
                pd_decode_load_check_interval_secs,
            },
        };
        config.validate()?;
        Ok(config)
    }

    /// Build the `--policy gateway_cache_aware` config exactly when that
    /// policy is selected (sgl-model-gateway CLI defaults and validation);
    /// `--gateway-ca-*` flags without it are refused.
    fn build_gateway_cache_aware(&self) -> Result<Option<GatewayCacheAwareConfig>> {
        let tuned = self.gateway_ca_cache_threshold.is_some()
            || self.gateway_ca_balance_abs_threshold.is_some()
            || self.gateway_ca_balance_rel_threshold.is_some()
            || self.gateway_ca_eviction_interval_secs.is_some()
            || self.gateway_ca_max_tree_size.is_some();
        let spd_ca = self.policy == PolicyKind::SymmetricPd
            && self.symmetric_pd_p_select == Some(SymmetricPdPSelect::GatewayCacheAware);
        if self.policy != PolicyKind::GatewayCacheAware {
            if tuned && !spd_ca {
                return Err(anyhow!(
                    "--gateway-ca-* flags require --policy gateway_cache_aware \
                     (or --policy symmetric_pd --symmetric-pd-p-select gateway-cache-aware)"
                ));
            }
            return Ok(None);
        }
        self.gateway_ca_values().map(Some)
    }

    /// The `--gateway-ca-*` values with the sgl-model-gateway CLI defaults,
    /// validated as the gateway does. The session retention fields are the
    /// PD-path flags (`--kv-session-retention`); symmetric_pd keeps its own.
    fn gateway_ca_values(&self) -> Result<GatewayCacheAwareConfig> {
        let d = GatewayCacheAwareConfig::default();
        let config = GatewayCacheAwareConfig {
            cache_threshold: self.gateway_ca_cache_threshold.unwrap_or(d.cache_threshold),
            balance_abs_threshold: self
                .gateway_ca_balance_abs_threshold
                .unwrap_or(d.balance_abs_threshold),
            balance_rel_threshold: self
                .gateway_ca_balance_rel_threshold
                .unwrap_or(d.balance_rel_threshold),
            eviction_interval_secs: self
                .gateway_ca_eviction_interval_secs
                .unwrap_or(d.eviction_interval_secs),
            max_tree_size: self.gateway_ca_max_tree_size.unwrap_or(d.max_tree_size),
            kv_session_retention: self.kv_session_retention,
            kv_session_key_file: self.kv_session_key_file.clone(),
        };
        // sgl-model-gateway config validation of the cache-aware policy.
        if !(0.0..=1.0).contains(&config.cache_threshold) {
            return Err(anyhow!(
                "--gateway-ca-cache-threshold must be in 0..=1, got {}",
                config.cache_threshold
            ));
        }
        if config.balance_rel_threshold.is_nan() || config.balance_rel_threshold < 1.0 {
            return Err(anyhow!(
                "--gateway-ca-balance-rel-threshold must be at least 1, got {}",
                config.balance_rel_threshold
            ));
        }
        if config.eviction_interval_secs == 0 {
            return Err(anyhow!("--gateway-ca-eviction-interval-secs must be > 0"));
        }
        if config.max_tree_size == 0 {
            return Err(anyhow!("--gateway-ca-max-tree-size must be > 0"));
        }
        Ok(config)
    }

    /// Build the symmetric P/D config exactly when `--policy symmetric_pd` is
    /// selected; reject orphan `--symmetric-pd-*` knobs otherwise.
    fn build_symmetric_pd(&self) -> Result<Option<SymmetricPdConfig>> {
        let selected = self.policy == PolicyKind::SymmetricPd;
        let tuned = !self.symmetric_pd_worker.is_empty()
            || self.symmetric_pd_workers_file.is_some()
            || self.symmetric_pd_force_split
            || self.symmetric_pd_pull_hint
            || self.symmetric_pd_max_queue_wait_ms.is_some()
            || self.symmetric_pd_max_uncached_tokens.is_some()
            || self.symmetric_pd_load_poll_ms.is_some()
            || self.symmetric_pd_w_hit.is_some()
            || self.symmetric_pd_w_backlog.is_some()
            || self.symmetric_pd_w_role.is_some()
            || self.symmetric_pd_w_xfer.is_some()
            || self.symmetric_pd_w_dload.is_some()
            || self.symmetric_pd_p_select.is_some()
            || self.symmetric_pd_d_select.is_some()
            || self.symmetric_pd_cache_threshold.is_some()
            || self.symmetric_pd_balance_abs_reqs.is_some()
            || self.symmetric_pd_balance_rel.is_some()
            || self.symmetric_pd_host_hit_factor.is_some()
            || self.symmetric_pd_decision_replay.is_some()
            || self.symmetric_pd_decision_replay_header.is_some()
            || self.symmetric_pd_import_cache_policy.is_some()
            || self.symmetric_pd_decode_host_return
            || self.symmetric_pd_decode_host_return_timeout_ms.is_some()
            || self.symmetric_pd_decode_host_return_max_tokens.is_some()
            || self.symmetric_pd_external_role_control
            || self.symmetric_pd_kv_session_retention
            || self.symmetric_pd_kv_session_key_file.is_some()
            || self.symmetric_pd_role_drain_timeout_secs.is_some()
            || self.symmetric_pd_unknown_expire_secs.is_some()
            || self.global_queue == Some(true)
            || self.global_queue_room.is_some()
            || self.symmetric_pd_trace_session_moves
            || self.symmetric_pd_pull_from_any_holder;
        if tuned && !selected {
            return Err(anyhow!(
                "--symmetric-pd-* flags require --policy symmetric_pd"
            ));
        }
        if !selected {
            return Ok(None);
        }
        let mut workers: Vec<SymmetricPdWorkerSpec> = Vec::new();
        for spec in &self.symmetric_pd_worker {
            workers.push(SymmetricPdWorkerSpec::parse_cli(spec).map_err(|e| anyhow!(e))?);
        }
        if let Some(path) = &self.symmetric_pd_workers_file {
            let text = std::fs::read_to_string(path)
                .map_err(|e| anyhow!("--symmetric-pd-workers-file {path:?}: {e}"))?;
            let file_workers: Vec<SymmetricPdWorkerSpec> =
                serde_json::from_str(&text).map_err(|e| {
                    anyhow!(
                        "--symmetric-pd-workers-file {path:?} is not a JSON array of workers: {e}"
                    )
                })?;
            workers.extend(file_workers);
        }
        let mut seen = std::collections::HashSet::new();
        for w in &workers {
            if w.url.trim().is_empty() {
                return Err(anyhow!("--symmetric-pd-worker: empty URL"));
            }
            if !(w.prefill_weight.is_finite()
                && w.prefill_weight >= 0.0
                && w.decode_weight.is_finite()
                && w.decode_weight >= 0.0)
            {
                return Err(anyhow!(
                    "symmetric-pd worker {:?}: weights must be finite and >= 0",
                    w.url
                ));
            }
            if !seen.insert(w.normalized_url()) {
                return Err(anyhow!("symmetric-pd worker {:?} listed twice", w.url));
            }
        }
        let max_queue_wait_ms = self
            .symmetric_pd_max_queue_wait_ms
            .unwrap_or_else(default_symmetric_pd_max_queue_wait_ms);
        if max_queue_wait_ms > SYMMETRIC_PD_MAX_QUEUE_WAIT_MS_CAP {
            return Err(anyhow!(
                "--symmetric-pd-max-queue-wait-ms must be in 0..={SYMMETRIC_PD_MAX_QUEUE_WAIT_MS_CAP} \
                 (seconds-level hold cap)"
            ));
        }
        let max_uncached_tokens = self
            .symmetric_pd_max_uncached_tokens
            .unwrap_or_else(default_symmetric_pd_max_uncached_tokens);
        let load_poll_ms = self
            .symmetric_pd_load_poll_ms
            .unwrap_or_else(default_symmetric_pd_load_poll_ms);
        if max_uncached_tokens == 0 || load_poll_ms == 0 {
            return Err(anyhow!(
                "--symmetric-pd-max-uncached-tokens / --symmetric-pd-load-poll-ms must be greater than 0"
            ));
        }
        let mut placement = crate::scheduler::placement::PlacementConfig {
            force_split: self.symmetric_pd_force_split,
            ..Default::default()
        };
        if let Some(v) = self.symmetric_pd_p_select {
            placement.p_select = v.into();
        }
        if placement.p_select == crate::scheduler::placement::PSelect::CostFunction {
            self.check_cost_function()?;
        } else {
            let cost_function_only = [
                (
                    self.symmetric_pd_unknown_expire_secs.is_some(),
                    "--symmetric-pd-unknown-expire-secs",
                ),
                (self.global_queue == Some(true), "--global-queue"),
                (self.global_queue_room.is_some(), "--global-queue-room"),
                (
                    self.symmetric_pd_trace_session_moves,
                    "--symmetric-pd-trace-session-moves",
                ),
                (
                    self.symmetric_pd_pull_from_any_holder,
                    "--symmetric-pd-pull-from-any-holder",
                ),
            ];
            if let Some((_, flag)) = cost_function_only.iter().find(|(set, _)| *set) {
                return Err(anyhow!(
                    "{flag} requires --symmetric-pd-p-select cost-function"
                ));
            }
        }
        if let Some(v) = self.symmetric_pd_d_select {
            placement.d_select = v.into();
        }
        if let Some(v) = self.symmetric_pd_cache_threshold {
            if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                return Err(anyhow!("--symmetric-pd-cache-threshold must be in 0..=1"));
            }
            placement.cache_threshold = v;
        }
        if let Some(v) = self.symmetric_pd_balance_abs_reqs {
            placement.balance_abs_reqs = v;
        }
        if let Some(v) = self.symmetric_pd_balance_rel {
            if !v.is_finite() || v < 1.0 {
                return Err(anyhow!(
                    "--symmetric-pd-balance-rel must be finite and >= 1"
                ));
            }
            placement.balance_rel = v;
        }
        if let Some(v) = self.symmetric_pd_host_hit_factor {
            if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                return Err(anyhow!("--symmetric-pd-host-hit-factor must be in 0..=1"));
            }
            placement.host_hit_factor = v;
        }
        let w = &mut placement.weights;
        for (name, value, slot) in [
            ("hit", self.symmetric_pd_w_hit, &mut w.w_hit),
            ("backlog", self.symmetric_pd_w_backlog, &mut w.w_backlog),
            ("role", self.symmetric_pd_w_role, &mut w.w_role),
            ("xfer", self.symmetric_pd_w_xfer, &mut w.w_xfer),
            ("dload", self.symmetric_pd_w_dload, &mut w.w_dload),
        ] {
            if let Some(v) = value {
                if !v.is_finite() || v < 0.0 {
                    return Err(anyhow!("--symmetric-pd-w-{name} must be finite and >= 0"));
                }
                *slot = v;
            }
        }
        if let Some(p) = &self.symmetric_pd_decision_replay {
            if !p.is_file() {
                return Err(anyhow!(
                    "--symmetric-pd-decision-replay {}: not a readable file",
                    p.display()
                ));
            }
        } else if self.symmetric_pd_decision_replay_header.is_some() {
            return Err(anyhow!(
                "--symmetric-pd-decision-replay-header requires --symmetric-pd-decision-replay"
            ));
        }
        let decision_replay_header = resolve_decision_replay_header(
            self.symmetric_pd_decision_replay_header.as_deref(),
            "--symmetric-pd-decision-replay-header",
        )?;
        if self.symmetric_pd_external_role_control && self.symmetric_pd_decision_replay.is_some() {
            return Err(anyhow!(
                "external role control conflicts with --symmetric-pd-decision-replay"
            ));
        }
        if self.symmetric_pd_role_drain_timeout_secs == Some(0) {
            return Err(anyhow!(
                "--symmetric-pd-role-drain-timeout-secs must be positive"
            ));
        }
        // `--symmetric-pd-p-select gateway-cache-aware`: the gateway rule and its
        // flags pick P; the cache-first knobs do not apply.
        let gateway_cache_aware = if self.symmetric_pd_p_select
            == Some(SymmetricPdPSelect::GatewayCacheAware)
        {
            if self.symmetric_pd_cache_threshold.is_some()
                || self.symmetric_pd_balance_abs_reqs.is_some()
                || self.symmetric_pd_balance_rel.is_some()
            {
                return Err(anyhow!(
                    "--symmetric-pd-p-select gateway-cache-aware takes the --gateway-ca-* flags, \
                     not --symmetric-pd-cache-threshold / --symmetric-pd-balance-*"
                ));
            }
            if self.symmetric_pd_decision_replay.is_some() {
                return Err(anyhow!(
                    "--symmetric-pd-p-select gateway-cache-aware conflicts with --symmetric-pd-decision-replay"
                ));
            }
            let mut ca = self.gateway_ca_values()?;
            // symmetric_pd keeps its own retention flag (--symmetric-pd-kv-session-retention).
            ca.kv_session_retention = false;
            ca.kv_session_key_file = None;
            Some(ca)
        } else {
            None
        };
        let import_cache_policy = self.symmetric_pd_import_cache_policy.unwrap_or(
            if self.symmetric_pd_decode_host_return {
                SymmetricPdImportCachePolicy::Retain
            } else {
                SymmetricPdImportCachePolicy::Auto
            },
        );
        let decode_host_return = if self.symmetric_pd_decode_host_return {
            if import_cache_policy != SymmetricPdImportCachePolicy::Retain {
                return Err(anyhow!(
                    "--symmetric-pd-decode-host-return requires --symmetric-pd-import-cache-policy retain"
                ));
            }
            let map_json = self.kv_worker_map.as_deref().ok_or_else(|| {
                anyhow!("--symmetric-pd-decode-host-return requires --kv-worker-map with peer servicers")
            })?;
            let worker_map = parse_kv_worker_map(map_json)?;
            let mut servicers = std::collections::BTreeMap::new();
            for entry in worker_map {
                let worker = entry.url.trim_end_matches('/').to_string();
                let worker_url = url::Url::parse(&worker)
                    .map_err(|_| anyhow!("invalid peer worker URL {worker}"))?;
                if !matches!(worker_url.scheme(), "http" | "https")
                    || worker_url.host_str().is_none()
                {
                    return Err(anyhow!("invalid peer worker URL {worker}"));
                }
                let address = normalize_host_return_endpoint(&entry.servicer)
                    .map_err(|e| anyhow!("invalid peer servicer for {worker}: {e}"))?;
                if servicers.insert(worker.clone(), address).is_some() {
                    return Err(anyhow!("duplicate peer worker {worker}"));
                }
            }
            if servicers
                .values()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != servicers.len()
            {
                return Err(anyhow!("peer workers require distinct servicer addresses"));
            }
            for worker in &workers {
                if !servicers.contains_key(worker.url.trim_end_matches('/')) {
                    return Err(anyhow!("missing peer servicer for {}", worker.url));
                }
            }
            let timeout_ms = self
                .symmetric_pd_decode_host_return_timeout_ms
                .unwrap_or(10_000);
            let max_tokens = self
                .symmetric_pd_decode_host_return_max_tokens
                .unwrap_or(65_536);
            if !(1..=60_000).contains(&timeout_ms) || !(1..=1_048_576).contains(&max_tokens) {
                return Err(anyhow!(
                    "decode host return timeout must be in 1..=60000 ms and max tokens in 1..=1048576"
                ));
            }
            Some(crate::config::types::DecodeHostReturnConfig {
                servicers,
                timeout_ms,
                max_tokens,
            })
        } else {
            None
        };
        Ok(Some(SymmetricPdConfig {
            gateway_cache_aware,
            placement,
            workers,
            max_queue_wait_ms,
            max_uncached_tokens,
            load_poll_ms,
            pull_hint: self.symmetric_pd_pull_hint,
            decision_replay: self.symmetric_pd_decision_replay.clone(),
            decision_replay_header,
            import_cache_policy,
            decode_host_return,
            external_role_control: self.symmetric_pd_external_role_control,
            kv_session_retention: self.symmetric_pd_kv_session_retention,
            kv_session_key_file: self.symmetric_pd_kv_session_key_file.clone(),
            role_drain_timeout: self
                .symmetric_pd_role_drain_timeout_secs
                .or(self
                    .symmetric_pd_external_role_control
                    .then_some(DEFAULT_ROLE_DRAIN_TIMEOUT_SECS))
                .map(std::time::Duration::from_secs),
            // `into_config` attaches it from the cost-function flags.
            cost_function: None,
        }))
    }

    /// The `--symmetric-pd-*` flags that `--symmetric-pd-p-select cost-function`
    /// needs or refuses. The cost-function pull rides on the P-leg as the pull hint;
    /// the cost function never uses the symmetric_pd hold queue (its own waiting
    /// queue is `--global-queue`). External role control is accepted:
    /// the candidate pool, the pull sources, the progress poll and the
    /// waiting-queue steps follow the live P-role set at each request.
    fn check_cost_function(&self) -> Result<()> {
        if !self.symmetric_pd_force_split {
            return Err(anyhow!(
                "--symmetric-pd-p-select cost-function requires --symmetric-pd-force-split"
            ));
        }
        if !self.symmetric_pd_pull_hint {
            return Err(anyhow!(
                "--symmetric-pd-p-select cost-function requires --symmetric-pd-pull-hint: the cost-function pull \
                 rides on the P-leg as the pull hint"
            ));
        }
        if self.symmetric_pd_max_queue_wait_ms.is_some()
            || self.symmetric_pd_max_uncached_tokens.is_some()
        {
            return Err(anyhow!(
                "--symmetric-pd-p-select cost-function never holds a request: \
                 --symmetric-pd-max-queue-wait-ms and --symmetric-pd-max-uncached-tokens do not apply"
            ));
        }
        if self.symmetric_pd_decision_replay.is_some() {
            return Err(anyhow!(
                "--symmetric-pd-p-select cost-function conflicts with --symmetric-pd-decision-replay"
            ));
        }
        if self.global_queue_room.is_some() && self.global_queue != Some(true) {
            return Err(anyhow!(
                "--global-queue-room requires --global-queue (the room rule of its queue)"
            ));
        }
        let cache_first_knobs = self.symmetric_pd_cache_threshold.is_some()
            || self.symmetric_pd_balance_abs_reqs.is_some()
            || self.symmetric_pd_balance_rel.is_some();
        if cache_first_knobs && self.symmetric_pd_d_select != Some(SymmetricPdDSelect::CacheFirst) {
            return Err(anyhow!(
                "--symmetric-pd-cache-threshold and --symmetric-pd-balance-* set only the cache-first D \
                 selection when --symmetric-pd-p-select is cost-function"
            ));
        }
        Ok(())
    }

    /// Resolve the discovery flags into a [`DiscoveryBackend`].
    ///
    /// `--worker-urls` (static) and `--service-discovery` (k8s) are
    /// mutually exclusive and exactly one is required. K8s-only flags
    /// passed without `--service-discovery` are rejected so a typo can't
    /// silently fall back to the static (empty) path. The k8s selector
    /// grammar (plain vs PD) is validated eagerly here by [`resolve_mode`]
    /// before the `K8sDiscoveryConfig` is constructed, so an invalid
    /// combination is never stored.
    ///
    /// `--symmetric-pd-worker` URLs join the static list (they are the
    /// fleet), so a symmetric P/D deployment needs no separate
    /// `--worker-urls`; with `--service-discovery` they only carry weights.
    fn build_discovery(
        &self,
        symmetric_pd: Option<&SymmetricPdConfig>,
    ) -> Result<DiscoveryBackend> {
        let mut static_urls = self.worker_urls.clone();
        if let Some(spd) = symmetric_pd {
            if !self.service_discovery {
                let mut seen: std::collections::HashSet<String> = static_urls
                    .iter()
                    .map(|u| u.trim().trim_end_matches('/').to_string())
                    .collect();
                for w in &spd.workers {
                    if seen.insert(w.normalized_url()) {
                        static_urls.push(w.url.trim().to_string());
                    }
                }
            }
        }
        let has_static = !static_urls.is_empty();
        let backend = match (has_static, self.service_discovery) {
            (true, true) => {
                return Err(anyhow!(
                    "--worker-urls and --service-discovery are mutually exclusive; pass exactly one"
                ))
            }
            (false, false) => {
                return Err(anyhow!(
                    "no discovery backend selected; pass --worker-urls <URL...> (static) \
                     or --service-discovery (kubernetes)"
                ))
            }
            (true, false) => {
                if self.service_discovery_namespace.is_some()
                    || !self.selector.is_empty()
                    || !self.prefill_selector.is_empty()
                    || !self.decode_selector.is_empty()
                {
                    return Err(anyhow!(
                        "--service-discovery-namespace / --selector / --prefill-selector / \
                         --decode-selector require --service-discovery"
                    ));
                }
                DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig { urls: static_urls })
            }
            (false, true) => {
                // Resolve (and validate) the selector flags into a
                // K8sDiscoveryMode here, so an invalid combination can't be
                // stored. Surfaces ConfigError as anyhow for the CLI.
                let mode = resolve_mode(
                    join_selector(&self.selector).as_deref(),
                    join_selector(&self.prefill_selector).as_deref(),
                    join_selector(&self.decode_selector).as_deref(),
                )
                .map_err(|e| anyhow!("{e}"))?;
                DiscoveryBackend::K8s(K8sDiscoveryConfig {
                    namespace: self.service_discovery_namespace.clone().unwrap_or_default(),
                    mode,
                })
            }
        };
        Ok(backend)
    }
}

/// Join space/repeated `key=value` selector terms into the single
/// comma-joined string the k8s backend's `labels_match_selector`
/// expects. `None` for an empty term list so [`resolve_mode`] can apply
/// its plain-vs-PD rules (and surface `NoSelector`).
fn join_selector(terms: &[String]) -> Option<String> {
    if terms.is_empty() {
        None
    } else {
        Some(terms.join(","))
    }
}

/// The value of a cost-function flag that takes a number or `auto`
/// (`--copy-price`, `--prefill-rate-tokens-per-s`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NumberOrAuto {
    Number(f64),
    Auto,
}

/// Parse `auto` (any case) or a number; [`Cli::into_config`] checks the
/// range of the number.
fn parse_number_or_auto(value: &str) -> std::result::Result<NumberOrAuto, String> {
    let value = value.trim();
    if value.eq_ignore_ascii_case("auto") {
        return Ok(NumberOrAuto::Auto);
    }
    value
        .parse::<f64>()
        .map(NumberOrAuto::Number)
        .map_err(|e| format!("expected a number or `auto`: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DiscoveryBackend, K8sDiscoveryMode};

    /// Parse argv (without the leading binary name) into a `Config`.
    /// Without `--symmetric-pd-preset` in `args` the flags are taken as given
    /// (`none`); the preset has its own tests.
    fn into_config(args: &[&str]) -> Result<Config> {
        let preset: &[&str] = if args.contains(&"--symmetric-pd-preset") {
            &[]
        } else {
            &["--symmetric-pd-preset", "none"]
        };
        let argv = std::iter::once("sgl-router")
            .chain(args.iter().copied())
            .chain(preset.iter().copied());
        let cli = Cli::try_parse_from(argv).map_err(|e| anyhow!("{e}"))?;
        cli.into_config()
    }

    const MODEL_ARGS: &[&str] = &[
        "--model-id",
        "qwen3-0.6b",
        "--tokenizer-path",
        "/tmp/qwen.json",
    ];

    fn with_model(extra: &[&str]) -> Vec<String> {
        MODEL_ARGS
            .iter()
            .chain(extra.iter())
            .map(|s| s.to_string())
            .collect()
    }

    /// `--policy symmetric_pd` defaults to the paper's configuration: late
    /// decode choice by fit_req over every decode, retention, force split and
    /// round-robin D; with the KV directory and servicers also the cost
    /// function with peer pulls, the copy price 4.6, the forecast wait, the
    /// global queue and no placement trace. Flags that are given win, and
    /// `none` takes the flags as given.
    #[test]
    fn symmetric_pd_preset_paper_is_the_default() {
        use crate::config::{LateDecodeMode, LateDecodeRule};
        let workers = [
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://10.0.0.1:30000,8998,1,0",
            "--symmetric-pd-worker",
            "http://10.0.0.2:30000,8998,0,1",
            "--symmetric-pd-preset",
            "paper",
        ];
        let c = into_config_owned(with_model(&workers)).unwrap();
        let spd = c.model.symmetric_pd.as_ref().unwrap();
        assert!(spd.placement.force_split);
        assert!(spd.kv_session_retention);
        assert_eq!(format!("{:?}", spd.placement.d_select), "RoundRobin");
        assert_eq!(
            format!("{:?}", spd.placement.p_select),
            "CacheFirst",
            "no KV directory"
        );
        assert!(spd.cost_function.is_none());
        let late = &c.proxy.late_decode;
        assert_eq!(late.mode, LateDecodeMode::Choose);
        assert_eq!(late.rule, Some(LateDecodeRule::FitReq));
        assert!(late.fit_native && late.native_state_poll);

        let mut no_retention = workers.to_vec();
        no_retention.push("--symmetric-pd-no-kv-session-retention");
        let c = into_config_owned(with_model(&no_retention)).unwrap();
        let spd = c.model.symmetric_pd.as_ref().unwrap();
        assert!(
            !spd.kv_session_retention && spd.placement.force_split,
            "only retention is off"
        );

        let mut with_kv = workers.to_vec();
        with_kv.extend([
            "--kv-indexer-url",
            "http://indexer:8091",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://10.0.0.1:30000","servicer":"10.0.0.1:18100"},{"id":2,"url":"http://10.0.0.2:30000","servicer":"10.0.0.2:18100"}]"#,
        ]);
        let c = into_config_owned(with_model(&with_kv)).unwrap();
        let spd = c.model.symmetric_pd.as_ref().unwrap();
        assert_eq!(format!("{:?}", spd.placement.p_select), "CostFunction");
        assert!(spd.pull_hint);
        let cost = spd.cost_function.as_ref().unwrap();
        assert_eq!(cost.cost_params.copy_price, 4.6);
        assert_eq!(cost.cost_params.prefill_rate_tokens_per_s, 13_500.0);
        assert_eq!(
            cost.cost_params.wait,
            crate::scheduler::placement_cost::PrefillWait::Forecast
        );
        assert!(cost.global_queue.is_some() && cost.forecast.is_some());
        assert!(cost.placement_trace.is_none());
        assert_eq!(
            spd.import_cache_policy,
            SymmetricPdImportCachePolicy::Retain
        );
        assert_eq!(c.model.kv_hint.as_ref().unwrap().indexer_timeout_ms, 200);
        assert!(spd.role_drain_timeout.is_none(), "no external role control");

        let mut roles = with_kv.clone();
        roles.push("--symmetric-pd-external-role-control");
        let c = into_config_owned(with_model(&roles)).unwrap();
        let spd = c.model.symmetric_pd.as_ref().unwrap();
        assert_eq!(
            spd.role_drain_timeout,
            Some(std::time::Duration::from_secs(5))
        );

        let mut given = with_kv.clone();
        given.extend([
            "--symmetric-pd-p-select",
            "cache-first",
            "--pd-late-decode-rule",
            "d1",
        ]);
        let c = into_config_owned(with_model(&given)).unwrap();
        let spd = c.model.symmetric_pd.as_ref().unwrap();
        assert_eq!(format!("{:?}", spd.placement.p_select), "CacheFirst");
        assert_eq!(c.proxy.late_decode.rule, Some(LateDecodeRule::D1));

        let mut none = with_kv.clone();
        let at = none.iter().position(|a| *a == "paper").unwrap();
        none[at] = "none";
        let c = into_config_owned(with_model(&none)).unwrap();
        let spd = c.model.symmetric_pd.as_ref().unwrap();
        assert!(!spd.kv_session_retention && !spd.pull_hint);
        assert_eq!(c.proxy.late_decode.mode, LateDecodeMode::Off);
    }

    fn into_config_owned(args: Vec<String>) -> Result<Config> {
        let refs: Vec<&str> = args.iter().map(String::as_str).collect();
        into_config(&refs)
    }

    #[test]
    fn dynamo_cache_aware_is_a_single_cache_source_without_transfer() {
        let args = with_model(&[
            "--worker-urls",
            "http://p:1",
            "--policy",
            "cache_aware_zmq",
            "--dynamo-cache-aware",
            "--kv-indexer-url",
            "http://index:2",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://p:1","servicer":"p:3"}]"#,
            "--pd-decode-policy",
            "round-robin",
        ]);
        let cfg = into_config_owned(args.clone()).unwrap();
        assert!(cfg.model.cache_aware.unwrap().dynamo.is_some());
        assert!(cfg.model.kv_hint.is_none());
        assert!(cfg.model.symmetric_pd.is_none());
        for extra in [
            vec!["--inject-kv-pull-hint"],
            vec!["--kv-indexer-endpoint", "http://grpc:4"],
        ] {
            let mut bad = args.clone();
            bad.extend(extra.into_iter().map(str::to_owned));
            assert!(into_config_owned(bad).is_err());
        }
    }

    #[test]
    fn token_charged_decode_policies_parse_and_validate() {
        let parse = |extra: &[&str]| {
            let mut args = vec!["--worker-urls", "http://10.0.0.1:30000"];
            args.extend(extra);
            into_config_owned(with_model(&args)).map(|c| c.model)
        };
        // Default unchanged: affinity-min-load, no token config.
        let m = parse(&[]).unwrap();
        assert_eq!(m.pd_decode_policy, PdDecodePolicy::AffinityMinLoad);
        assert!(m.pd_decode_tokens.is_none());
        let m = parse(&["--pd-decode-policy", "round-robin"]).unwrap();
        assert!(m.pd_decode_tokens.is_none());

        let m = parse(&["--pd-decode-policy", "min-inflight-tokens"]).unwrap();
        assert_eq!(m.pd_decode_policy, PdDecodePolicy::MinInflightTokens);
        assert_eq!(m.pd_decode_tokens, Some(PdDecodeTokensConfig::default()));
        assert_eq!(m.pd_decode_tokens.unwrap().output_charge_cap, 8192);

        let m = parse(&[
            "--pd-decode-policy",
            "po2-tokens",
            "--pd-decode-tokens-output-cap",
            "512",
            "--pd-decode-tokens-normalize-capacity",
        ])
        .unwrap();
        assert_eq!(m.pd_decode_policy, PdDecodePolicy::PowerOfTwoTokens);
        assert_eq!(
            m.pd_decode_tokens,
            Some(PdDecodeTokensConfig {
                output_charge_cap: 512,
                normalize_by_capacity: true,
            })
        );

        // Orphan knobs are rejected.
        for extra in [
            &["--pd-decode-tokens-output-cap", "512"][..],
            &["--pd-decode-tokens-normalize-capacity"][..],
            &[
                "--pd-decode-policy",
                "po2",
                "--pd-decode-tokens-normalize-capacity",
            ][..],
        ] {
            let err = parse(extra).unwrap_err().to_string();
            assert!(err.contains("--pd-decode-tokens-*"), "{err}");
        }
        assert!(parse(&["--pd-decode-policy", "min-tokens"]).is_err());
    }

    #[test]
    fn power_of_two_smg_decode_policy_parses_with_the_smg_default_interval() {
        let parse = |extra: &[&str]| {
            let mut args = vec!["--worker-urls", "http://10.0.0.1:30000"];
            args.extend(extra);
            into_config_owned(with_model(&args))
        };
        // Defaults unchanged: affinity-min-load, interval 5 (unused).
        let c = parse(&[]).unwrap();
        assert_eq!(c.model.pd_decode_policy, PdDecodePolicy::AffinityMinLoad);
        assert_eq!(c.active_load.pd_decode_load_check_interval_secs, 5);
        // The existing `po2` stays a different policy.
        let c = parse(&["--pd-decode-policy", "po2"]).unwrap();
        assert_eq!(c.model.pd_decode_policy, PdDecodePolicy::PowerOfTwo);

        let c = parse(&["--pd-decode-policy", "power-of-two-smg"]).unwrap();
        assert_eq!(c.model.pd_decode_policy, PdDecodePolicy::PowerOfTwoSmg);
        assert_eq!(
            c.active_load.pd_decode_load_check_interval_secs, 5,
            "SMG --load-check-interval-secs default"
        );
        assert!(c.model.pd_decode_tokens.is_none());
        let c = parse(&[
            "--pd-decode-policy",
            "power-of-two-smg",
            "--pd-decode-load-check-interval-secs",
            "2",
        ])
        .unwrap();
        assert_eq!(c.active_load.pd_decode_load_check_interval_secs, 2);

        // 0 is rejected (SMG: "Must be > 0"); the knob needs the policy.
        let err = parse(&[
            "--pd-decode-policy",
            "power-of-two-smg",
            "--pd-decode-load-check-interval-secs",
            "0",
        ])
        .unwrap_err()
        .to_string();
        assert!(err.contains("must be > 0"), "{err}");
        for extra in [
            &["--pd-decode-load-check-interval-secs", "5"][..],
            &[
                "--pd-decode-policy",
                "po2",
                "--pd-decode-load-check-interval-secs",
                "5",
            ][..],
        ] {
            let err = parse(extra).unwrap_err().to_string();
            assert!(
                err.contains("requires --pd-decode-policy power-of-two-smg"),
                "{err}"
            );
        }
        // Token knobs stay rejected for this non-token policy.
        let err = parse(&[
            "--pd-decode-policy",
            "power-of-two-smg",
            "--pd-decode-tokens-output-cap",
            "512",
        ])
        .unwrap_err()
        .to_string();
        assert!(err.contains("--pd-decode-tokens-*"), "{err}");
    }

    #[test]
    fn defaults_host_port_and_policy() {
        let c = into_config_owned(with_model(&["--worker-urls", "http://10.0.0.1:30000"])).unwrap();
        assert_eq!(c.server.host, "127.0.0.1");
        assert_eq!(c.server.port, 30000);
        assert_eq!(c.model.policy, PolicyKind::RoundRobin);
        assert_eq!(c.model.id, "qwen3-0.6b");
        assert_eq!(c.proxy.request_timeout_secs, 300);
        assert_eq!(c.active_load.stale_request_timeout_secs, 600);
    }

    /// With `--tokenizer-path` omitted, the tokenizer source defaults to the
    /// model id (treated as an HF repo id at load time).
    #[test]
    fn tokenizer_path_defaults_to_model_id_when_omitted() {
        let c = into_config(&[
            "--model-id",
            "Qwen/Qwen3-0.6B",
            "--worker-urls",
            "http://x:30000",
        ])
        .unwrap();
        assert_eq!(c.model.id, "Qwen/Qwen3-0.6B");
        assert_eq!(c.model.tokenizer_path, "Qwen/Qwen3-0.6B");
    }

    #[test]
    fn explicit_tokenizer_path_is_used() {
        let c = into_config(&[
            "--model-id",
            "qwen3",
            "--tokenizer-path",
            "/models/qwen3/tokenizer.json",
            "--worker-urls",
            "http://x:30000",
        ])
        .unwrap();
        assert_eq!(c.model.tokenizer_path, "/models/qwen3/tokenizer.json");
    }

    #[test]
    fn static_urls_backend() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://10.0.0.1:30000",
            "http://10.0.0.2:30000",
        ]))
        .unwrap();
        match &c.discovery {
            DiscoveryBackend::StaticUrls(s) => assert_eq!(
                s.urls,
                vec![
                    "http://10.0.0.1:30000".to_string(),
                    "http://10.0.0.2:30000".to_string()
                ]
            ),
            _ => panic!("expected static_urls backend"),
        }
    }

    #[test]
    fn rejects_no_discovery_backend() {
        let err = into_config_owned(with_model(&[])).unwrap_err().to_string();
        assert!(err.contains("no discovery backend"), "got: {err}");
    }

    #[test]
    fn rejects_both_discovery_backends() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--service-discovery",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("mutually exclusive"), "got: {err}");
    }

    #[test]
    fn rejects_k8s_flags_without_service_discovery() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--selector",
            "app=sglang",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("require --service-discovery"), "got: {err}");
    }

    #[test]
    fn rejects_static_urls_duplicate() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "http://x:30000",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("duplicate"), "got: {err}");
    }

    #[test]
    fn rejects_static_urls_schemeless() {
        let err = into_config_owned(with_model(&["--worker-urls", "10.0.0.1:30000"]))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("not a valid URL") || err.contains("unsupported scheme"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_static_urls_non_http_scheme() {
        let err = into_config_owned(with_model(&["--worker-urls", "ws://x:30000"]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported scheme"), "got: {err}");
    }

    #[test]
    fn k8s_plain_backend() {
        let c = into_config_owned(with_model(&[
            "--service-discovery",
            "--service-discovery-namespace",
            "prod",
            "--selector",
            "app=engines-qwen3",
        ]))
        .unwrap();
        match &c.discovery {
            DiscoveryBackend::K8s(k) => {
                assert_eq!(k.namespace, "prod");
                assert_eq!(
                    k.mode,
                    K8sDiscoveryMode::Plain {
                        label_selector: "app=engines-qwen3".to_string()
                    }
                );
            }
            _ => panic!("expected k8s backend"),
        }
    }

    /// Multiple `--selector` terms AND-join into one comma-separated
    /// label selector (matches the Python router's space-separated form).
    #[test]
    fn k8s_plain_selector_joins_multiple_terms() {
        let c = into_config_owned(with_model(&[
            "--service-discovery",
            "--selector",
            "app=sglang",
            "zone=us-east",
        ]))
        .unwrap();
        match &c.discovery {
            DiscoveryBackend::K8s(k) => assert_eq!(
                k.mode,
                K8sDiscoveryMode::Plain {
                    label_selector: "app=sglang,zone=us-east".to_string()
                }
            ),
            _ => panic!("expected k8s backend"),
        }
    }

    /// Empty namespace is intentional — it triggers a cluster-wide watch.
    #[test]
    fn k8s_empty_namespace_watches_all() {
        let c = into_config_owned(with_model(&[
            "--service-discovery",
            "--selector",
            "app=sglang",
        ]))
        .unwrap();
        match &c.discovery {
            DiscoveryBackend::K8s(k) => assert_eq!(k.namespace, ""),
            _ => panic!("expected k8s backend"),
        }
    }

    #[test]
    fn k8s_pd_backend() {
        let c = into_config_owned(with_model(&[
            "--service-discovery",
            "--service-discovery-namespace",
            "default",
            "--prefill-selector",
            "app=sglang,role=prefill",
            "--decode-selector",
            "app=sglang,role=decode",
        ]))
        .unwrap();
        match &c.discovery {
            DiscoveryBackend::K8s(k) => assert_eq!(
                k.mode,
                K8sDiscoveryMode::PdDisaggregation {
                    prefill_selector: "app=sglang,role=prefill".to_string(),
                    decode_selector: "app=sglang,role=decode".to_string(),
                }
            ),
            _ => panic!("expected k8s backend"),
        }
    }

    /// `--service-discovery` with no selector at all fails `resolve_mode`
    /// validation with the `NoSelector` wording.
    #[test]
    fn rejects_k8s_without_selector() {
        let err = into_config_owned(with_model(&["--service-discovery"]))
            .unwrap_err()
            .to_string()
            .to_lowercase();
        assert!(err.contains("none were set"), "got: {err}");
    }

    /// `--prefill-selector` without `--decode-selector` is rejected through
    /// the full CLI path — pins that `build_discovery` feeds the right
    /// selectors into `resolve_mode` (a positional mix-up would surface a
    /// different error or none).
    #[test]
    fn rejects_k8s_partial_pd_selectors() {
        let err = into_config_owned(with_model(&[
            "--service-discovery",
            "--prefill-selector",
            "app=sglang,role=prefill",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("PD mode requires BOTH"),
            "expected PartialPdSelectors wording, got: {err}"
        );
    }

    /// Identical prefill/decode selectors are rejected through the full CLI
    /// path (would silently leave the decode pool empty at runtime).
    #[test]
    fn rejects_k8s_identical_pd_selectors() {
        let err = into_config_owned(with_model(&[
            "--service-discovery",
            "--prefill-selector",
            "app=sglang",
            "--decode-selector",
            "app=sglang",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("must differ"),
            "expected IdenticalPdSelectors wording, got: {err}"
        );
    }

    /// clap rejects an unknown `--policy` value at parse time.
    #[test]
    fn rejects_unknown_policy() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "bogus_policy",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("bogus_policy") || err.contains("policy"),
            "got: {err}"
        );
    }

    /// `--policy load_based` parses to the load-based selector.
    #[test]
    fn parses_load_based_policy() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://10.0.0.1:30000",
            "--policy",
            "load_based",
        ]))
        .unwrap();
        assert_eq!(c.model.policy, PolicyKind::LoadBased);
    }

    /// clap rejects `--cb-threshold 0` because the field is `NonZeroU32`.
    #[test]
    fn rejects_zero_cb_threshold() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--cb-threshold",
            "0",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("cb-threshold"), "got: {err}");
    }

    #[test]
    fn cb_threshold_enables_circuit_breaker_with_default_cool_down() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--cb-threshold",
            "5",
        ]))
        .unwrap();
        let cb = c.model.circuit_breaker.expect("cb enabled");
        assert_eq!(cb.threshold.get(), 5);
        assert_eq!(cb.cool_down_secs, 30);
    }

    #[test]
    fn cb_cool_down_honors_explicit_override() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--cb-threshold",
            "3",
            "--cb-cool-down-secs",
            "10",
        ]))
        .unwrap();
        let cb = c.model.circuit_breaker.expect("cb enabled");
        assert_eq!(cb.cool_down_secs, 10);
    }

    #[test]
    fn disable_circuit_breaker_parses_and_conflicts_with_cb_knobs() {
        let base = ["--worker-urls", "http://10.0.0.1:30000"];
        let c = into_config_owned(with_model(&base)).unwrap();
        assert!(!c.proxy.disable_circuit_breaker, "default: enabled");
        assert!(c.model.circuit_breaker.is_none());

        let mut args = base.to_vec();
        args.push("--disable-circuit-breaker");
        let c = into_config_owned(with_model(&args)).unwrap();
        assert!(c.proxy.disable_circuit_breaker);
        assert!(c.model.circuit_breaker.is_none());

        for extra in [
            &["--cb-threshold", "3"][..],
            &["--cb-cool-down-secs", "5"][..],
            &["--cb-threshold", "3", "--cb-cool-down-secs", "5"][..],
        ] {
            let mut args = args.clone();
            args.extend(extra);
            let err = into_config_owned(with_model(&args)).unwrap_err();
            assert!(
                err.to_string()
                    .contains("--disable-circuit-breaker conflicts"),
                "{err}"
            );
        }
    }

    #[test]
    fn pd_abort_decode_on_prefill_failure_parses_and_defaults_off() {
        let base = ["--worker-urls", "http://10.0.0.1:30000"];
        let c = into_config_owned(with_model(&base)).unwrap();
        assert!(!c.proxy.abort_decode_on_prefill_failure, "default: off");

        let mut args = base.to_vec();
        args.push("--pd-abort-decode-on-prefill-failure");
        let c = into_config_owned(with_model(&args)).unwrap();
        assert!(c.proxy.abort_decode_on_prefill_failure);
    }

    #[test]
    fn pd_late_decode_flags_parse_and_validate() {
        use crate::config::{LateDecodeConfig, LateDecodeMode, LateDecodeRule};
        let parse = |extra: &[&str]| {
            let mut args = vec!["--worker-urls", "http://10.0.0.1:30000"];
            args.extend(extra);
            into_config_owned(with_model(&args))
        };
        let po2 = ["--pd-decode-policy", "power-of-two-smg"];
        // Default: off, nothing packed, no token config.
        let c = parse(&[]).unwrap();
        assert_eq!(c.proxy.late_decode, LateDecodeConfig::default());
        assert!(!c.proxy.late_decode.choose());
        assert!(c.model.pd_decode_tokens.is_none());

        // Choose + d1 over all candidates.
        let mut args = po2.to_vec();
        args.extend([
            "--pd-late-decode",
            "choose",
            "--pd-late-decode-rule",
            "d1",
            "--pd-late-decode-candidates",
            "all",
        ]);
        let c = parse(&args).unwrap();
        assert_eq!(
            c.proxy.late_decode,
            LateDecodeConfig {
                mode: LateDecodeMode::Choose,
                rule: Some(LateDecodeRule::D1),
                candidates: None,
                packed_prompt: false,
                ..LateDecodeConfig::default()
            }
        );
        // d1 charges with the token ledger: its tuning is built (defaults).
        let tokens = c.model.pd_decode_tokens.clone().expect("d1 token config");
        assert_eq!(tokens.output_charge_cap, 8192);
        assert_eq!(c.model.pd_decode_policy, PdDecodePolicy::PowerOfTwoSmg);
        // ... and takes the token flags.
        let mut capped = args.clone();
        capped.extend(["--pd-decode-tokens-output-cap", "1024"]);
        let c = parse(&capped).unwrap();
        assert_eq!(c.model.pd_decode_tokens.unwrap().output_charge_cap, 1024);

        // po2 / keep, a candidate count, packed prompts.
        for (rule, want) in [("po2", LateDecodeRule::Po2), ("keep", LateDecodeRule::Keep)] {
            let mut args = po2.to_vec();
            args.extend([
                "--pd-late-decode",
                "choose",
                "--pd-late-decode-rule",
                rule,
                "--pd-late-decode-candidates",
                "2",
                "--pd-late-decode-packed-prompt",
            ]);
            let c = parse(&args).unwrap();
            assert_eq!(c.proxy.late_decode.rule, Some(want));
            assert_eq!(c.proxy.late_decode.candidates, Some(2));
            assert!(c.proxy.late_decode.packed_prompt);
            assert!(
                c.model.pd_decode_tokens.is_none(),
                "{rule}: no token config"
            );
        }
        // Explicit off; packed prompts alone are valid with late D off.
        let c = parse(&["--pd-late-decode", "off", "--pd-late-decode-packed-prompt"]).unwrap();
        assert!(!c.proxy.late_decode.choose());
        assert!(c.proxy.late_decode.packed_prompt);

        // Refused combinations.
        let expect_err = |extra: &[&str], needle: &str| {
            let err = parse(extra).unwrap_err().to_string();
            assert!(err.contains(needle), "{extra:?}: {err}");
        };
        expect_err(
            &["--pd-late-decode-rule", "d1"],
            "require --pd-late-decode choose",
        );
        expect_err(
            &["--pd-late-decode-candidates", "2"],
            "require --pd-late-decode choose",
        );
        expect_err(
            &[
                "--pd-decode-policy",
                "power-of-two-smg",
                "--pd-late-decode",
                "choose",
            ],
            "requires --pd-late-decode-rule",
        );
        for bad in ["0", "65", "x"] {
            let mut args = po2.to_vec();
            args.extend([
                "--pd-late-decode",
                "choose",
                "--pd-late-decode-rule",
                "d1",
                "--pd-late-decode-candidates",
                bad,
            ]);
            expect_err(&args, "--pd-late-decode-candidates must be all or a count");
        }
        expect_err(
            &["--pd-late-decode", "choose", "--pd-late-decode-rule", "d1"],
            "requires --pd-decode-policy power-of-two-smg",
        );
        expect_err(
            &[
                "--pd-decode-policy",
                "min-inflight-tokens",
                "--pd-late-decode",
                "choose",
                "--pd-late-decode-rule",
                "d1",
            ],
            "requires --pd-decode-policy power-of-two-smg",
        );
        expect_err(&["--pd-late-decode", "maybe"], "invalid value");
        // The token flags still need a token rule.
        let mut args = po2.to_vec();
        args.extend([
            "--pd-late-decode",
            "choose",
            "--pd-late-decode-rule",
            "po2",
            "--pd-decode-tokens-output-cap",
            "1024",
        ]);
        expect_err(&args, "--pd-decode-tokens-* flags require");
        // symmetric_pd without force split is refused (LOCAL is not ported).
        let err = into_config_owned(with_model(&[
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://10.0.0.1:30000,8998,1,1",
            "--pd-decode-policy",
            "power-of-two-smg",
            "--pd-late-decode",
            "choose",
            "--pd-late-decode-rule",
            "d1",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("requires --symmetric-pd-force-split"), "{err}");
    }

    /// Late D choose on symmetric workers (`--policy symmetric_pd`): opt-in,
    /// needs force split, takes keep / d1 / fit_req (native too) without the
    /// `power-of-two-smg` decode policy, refuses po2 and the decision replay,
    /// accepts external role control; off by default.
    #[test]
    fn symmetric_pd_late_decode_choose_is_opt_in() {
        let base = [
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-force-split",
            "--symmetric-pd-worker",
            "http://10.0.0.1:30000,8998,1,0",
            "--symmetric-pd-worker",
            "http://10.0.0.2:30000,8998,0,1",
            "--symmetric-pd-d-select",
            "round-robin",
        ];
        let off = into_config_owned(with_model(&base)).unwrap();
        assert!(!off.proxy.late_decode.choose());

        let mut args = base.to_vec();
        args.extend([
            "--pd-late-decode",
            "choose",
            "--pd-late-decode-rule",
            "fit_req",
            "--pd-late-decode-fit-native",
            "--pd-native-decode-state",
        ]);
        let on = into_config_owned(with_model(&args)).unwrap();
        assert!(on.proxy.late_decode.choose());
        assert!(on.proxy.late_decode.fit_native);
        assert!(on.model.symmetric_pd.is_some());
        // fit_req charges on the token ledger.
        assert!(on.model.pd_decode_tokens.is_some());

        let mut args = base.to_vec();
        args.extend(["--pd-late-decode", "choose", "--pd-late-decode-rule", "po2"]);
        let err = into_config_owned(with_model(&args))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("po2 is not available with --policy symmetric_pd"),
            "{err}"
        );

        // External role control (dynamic roles) is accepted.
        let mut args = base.to_vec();
        args.extend([
            "--symmetric-pd-external-role-control",
            "--pd-late-decode",
            "choose",
            "--pd-late-decode-rule",
            "fit_req",
            "--pd-late-decode-fit-native",
            "--pd-native-decode-state",
        ]);
        let on = into_config_owned(with_model(&args)).unwrap();
        assert!(on.proxy.late_decode.choose());
        assert!(
            on.model
                .symmetric_pd
                .as_ref()
                .unwrap()
                .external_role_control
        );

        let table =
            std::env::temp_dir().join(format!("late-d-replay-{}.jsonl", std::process::id()));
        std::fs::write(&table, "").unwrap();
        let table_arg = table.to_string_lossy().into_owned();
        let mut args = base.to_vec();
        args.extend([
            "--symmetric-pd-decision-replay",
            table_arg.as_str(),
            "--pd-late-decode",
            "choose",
            "--pd-late-decode-rule",
            "d1",
        ]);
        let err = into_config_owned(with_model(&args))
            .unwrap_err()
            .to_string();
        let _ = std::fs::remove_file(&table);
        assert!(
            err.contains("conflicts with --symmetric-pd-decision-replay"),
            "{err}"
        );
        assert!(!err.contains("not a readable file"), "{err}");
    }

    /// `--symmetric-pd-p-select gateway-cache-aware` takes the `--gateway-ca-*`
    /// flags and defaults, keeps the model-level gateway config off, and
    /// refuses the cache-first knobs.
    #[test]
    fn symmetric_pd_gateway_cache_aware_p_select() {
        let base = [
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-force-split",
            "--symmetric-pd-worker",
            "http://10.0.0.1:30000,8998,1,0",
            "--symmetric-pd-worker",
            "http://10.0.0.2:30000,8998,0,1",
            "--symmetric-pd-p-select",
            "gateway-cache-aware",
        ];
        let cfg = into_config_owned(with_model(&base)).unwrap();
        assert!(cfg.model.gateway_cache_aware.is_none());
        let spd = cfg.model.symmetric_pd.unwrap();
        assert_eq!(
            spd.gateway_cache_aware,
            Some(GatewayCacheAwareConfig::default())
        );
        assert_eq!(
            spd.placement.p_select,
            crate::scheduler::placement::PSelect::CacheFirst
        );

        let mut args = base.to_vec();
        args.extend([
            "--gateway-ca-cache-threshold",
            "0.3",
            "--gateway-ca-balance-abs-threshold",
            "64",
            "--gateway-ca-balance-rel-threshold",
            "1.5",
            "--gateway-ca-eviction-interval-secs",
            "120",
            "--gateway-ca-max-tree-size",
            "66163200",
        ]);
        let ca = into_config_owned(with_model(&args))
            .unwrap()
            .model
            .symmetric_pd
            .unwrap()
            .gateway_cache_aware
            .unwrap();
        assert_eq!(ca.max_tree_size, 66_163_200);
        assert!(!ca.kv_session_retention);

        let mut args = base.to_vec();
        args.extend(["--symmetric-pd-cache-threshold", "0.5"]);
        let err = into_config_owned(with_model(&args))
            .unwrap_err()
            .to_string();
        assert!(err.contains("takes the --gateway-ca-* flags"), "{err}");

        // The --gateway-ca-* flags with another symmetric_pd P selection: refused.
        let err = into_config_owned(with_model(&[
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://10.0.0.1:30000,8998,1,1",
            "--gateway-ca-max-tree-size",
            "64",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("require --policy gateway_cache_aware"),
            "{err}"
        );
    }

    /// sgl-model-gateway `fit_req` flags: the fixed capacity, the observed
    /// capacity (needs the capacity poll), the default sample age, the
    /// capacity poll alone with late D off, and the refused combinations.
    #[test]
    fn pd_late_decode_fit_req_flags_parse_and_validate() {
        use crate::config::{LateDecodeConfig, LateDecodeMode, LateDecodeRule};
        let parse = |extra: &[&str]| {
            let mut args = vec![
                "--worker-urls",
                "http://10.0.0.1:30000",
                "--pd-decode-policy",
                "power-of-two-smg",
            ];
            args.extend(extra);
            into_config_owned(with_model(&args))
        };
        let choose = [
            "--pd-late-decode",
            "choose",
            "--pd-late-decode-rule",
            "fit_req",
        ];
        // Fixed capacity.
        let mut args = choose.to_vec();
        args.extend(["--pd-late-decode-fit-capacity-tokens", "1900000"]);
        let c = parse(&args).unwrap();
        assert_eq!(
            c.proxy.late_decode,
            LateDecodeConfig {
                mode: LateDecodeMode::Choose,
                rule: Some(LateDecodeRule::FitReq),
                candidates: None,
                packed_prompt: false,
                fit_capacity_tokens: Some(1_900_000),
                fit_native: false,
                native_state_poll: false,
                fit_state_max_age_ms: 1000,
            }
        );
        assert_eq!(LateDecodeRule::FitReq.as_str(), "fit_req");
        // fit_req picks on the token ledger: its tuning is built.
        assert_eq!(c.model.pd_decode_tokens.unwrap().output_charge_cap, 8192);

        // The decode flags of the gateway control: observed capacity, local
        // off, packed prompts, no circuit breaker.
        let mut args = choose.to_vec();
        args.extend([
            "--pd-late-decode-candidates",
            "all",
            "--pd-late-decode-local",
            "off",
            "--pd-native-decode-state",
            "--pd-late-decode-fit-native",
            "--pd-late-decode-fit-state-max-age-ms",
            "1000",
            "--pd-late-decode-packed-prompt",
            "--disable-circuit-breaker",
        ]);
        let c = parse(&args).unwrap();
        let l = c.proxy.late_decode;
        assert!(l.choose() && l.fit_native && l.native_state_poll && l.packed_prompt);
        assert_eq!(l.candidates, None);
        assert_eq!(
            (l.fit_capacity_tokens, l.fit_state_max_age_ms),
            (None, 1000)
        );
        assert!(c.proxy.disable_circuit_breaker);

        // The capacity poll alone is valid with late D off.
        let c = parse(&["--pd-native-decode-state"]).unwrap();
        assert!(c.proxy.late_decode.native_state_poll);
        assert!(!c.proxy.late_decode.choose());

        let expect_err = |extra: &[&str], needle: &str| {
            let err = parse(extra).unwrap_err().to_string();
            assert!(err.contains(needle), "{extra:?}: {err}");
        };
        // fit_req needs a capacity source.
        expect_err(&choose, "requires --pd-late-decode-fit-capacity-tokens");
        // The fixed capacity needs fit_req.
        expect_err(
            &[
                "--pd-late-decode",
                "choose",
                "--pd-late-decode-rule",
                "d1",
                "--pd-late-decode-fit-capacity-tokens",
                "1000",
            ],
            "used only by",
        );
        // The observed capacity needs the poll, a positive age and no fixed
        // capacity.
        let mut no_poll = choose.to_vec();
        no_poll.push("--pd-late-decode-fit-native");
        expect_err(&no_poll, "--pd-late-decode-fit-native needs");
        let mut both = no_poll.clone();
        both.extend([
            "--pd-native-decode-state",
            "--pd-late-decode-fit-capacity-tokens",
            "1000",
        ]);
        expect_err(&both, "--pd-late-decode-fit-native needs");
        let mut zero_age = no_poll.clone();
        zero_age.extend([
            "--pd-native-decode-state",
            "--pd-late-decode-fit-state-max-age-ms",
            "0",
        ]);
        expect_err(&zero_age, "--pd-late-decode-fit-native needs");
        // Orphan fit flags with late D off; the owner-local winner.
        expect_err(
            &["--pd-late-decode-fit-native"],
            "require --pd-late-decode choose",
        );
        expect_err(&["--pd-late-decode-local", "on"], "not available");
        expect_err(
            &["--pd-late-decode-fit-capacity-tokens", "0"],
            "invalid value",
        );
    }
    #[test]
    fn proxy_tcp_keepalive_secs_parses_and_rejects_zero() {
        let base = ["--worker-urls", "http://10.0.0.1:30000"];
        let c = into_config_owned(with_model(&base)).unwrap();
        assert_eq!(c.proxy.tcp_keepalive_secs, None, "default: reqwest's 15 s");

        let mut args = base.to_vec();
        args.extend(["--proxy-tcp-keepalive-secs", "30"]);
        let c = into_config_owned(with_model(&args)).unwrap();
        assert_eq!(c.proxy.tcp_keepalive_secs, Some(30));

        for bad in ["0", "-1", "x"] {
            let mut args = base.to_vec();
            args.extend(["--proxy-tcp-keepalive-secs", bad]);
            assert!(
                into_config_owned(with_model(&args)).is_err(),
                "--proxy-tcp-keepalive-secs {bad} must be rejected"
            );
        }
    }

    #[test]
    fn rejects_cb_cool_down_without_threshold() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--cb-cool-down-secs",
            "10",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("--cb-cool-down-secs requires --cb-threshold"),
            "got: {err}"
        );
    }

    #[test]
    fn cache_aware_knob_builds_partial_config() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
            "--cache-threshold",
            "0.7",
        ]))
        .unwrap();
        let ca = c.model.cache_aware.expect("cache_aware set");
        assert_eq!(ca.cache_threshold, 0.7);
        // Untouched knobs fall back to defaults.
        assert_eq!(ca.balance_abs_threshold, 32);
    }

    #[test]
    fn kv_indexer_reuses_cache_aware_policy_config() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
            "--kv-indexer-endpoint",
            "http://indexer:50051",
            "--kv-indexer-query-timeout-ms",
            "75",
            "--kv-indexer-query-max-inflight",
            "17",
        ]))
        .unwrap();
        let cache = c.model.cache_aware.expect("cache-aware config");
        let indexer = cache.kv_indexer_endpoint.expect("Indexer config");
        assert_eq!(indexer.url, "http://indexer:50051");
        assert_eq!(indexer.query_timeout_ms, 75);
        assert_eq!(indexer.query_max_inflight, 17);
    }

    #[test]
    fn kv_indexer_uses_safe_query_defaults() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
            "--kv-indexer-endpoint",
            "http://indexer:50051",
        ]))
        .unwrap();
        let indexer = c
            .model
            .cache_aware
            .expect("cache-aware config")
            .kv_indexer_endpoint
            .expect("Indexer config");
        assert_eq!(indexer.query_timeout_ms, 100);
        assert_eq!(indexer.query_max_inflight, 32);
    }

    #[test]
    fn kv_indexer_requires_cache_aware_policy() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--kv-indexer-endpoint",
            "http://indexer:50051",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("require --policy cache_aware_zmq"));
    }

    #[test]
    fn kv_indexer_timeout_requires_endpoint() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
            "--kv-indexer-query-timeout-ms",
            "75",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("requires --kv-indexer-endpoint"));
    }

    #[test]
    fn kv_indexer_max_inflight_requires_endpoint() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
            "--kv-indexer-query-max-inflight",
            "17",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("requires --kv-indexer-endpoint"));
    }

    #[test]
    fn kv_indexer_max_inflight_must_be_positive() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
            "--kv-indexer-endpoint",
            "http://indexer:50051",
            "--kv-indexer-query-max-inflight",
            "0",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("must be greater than zero"));
    }

    #[test]
    fn no_cache_aware_flags_leaves_none() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
        ]))
        .unwrap();
        assert!(c.model.cache_aware.is_none());
    }

    #[test]
    fn rejects_cache_aware_knob_without_cache_aware_policy() {
        // Default policy is round_robin, so a cache knob has no effect —
        // reject rather than silently ignore it.
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--cache-threshold",
            "0.7",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("require --policy cache_aware_zmq"),
            "got: {err}"
        );
    }

    #[test]
    fn log_format_parses_json() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--log-format",
            "json",
        ]))
        .unwrap();
        assert_eq!(c.observability.log_format, LogFormat::Json);
    }

    /// Pins that the two timeout overrides land in the right fields — they
    /// are adjacent `u64`s with similar names, so a copy-paste swap would
    /// otherwise go unnoticed (and `stale` must sit above `proxy`).
    #[test]
    fn timeout_overrides_land_in_distinct_fields() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--request-timeout-secs",
            "120",
            "--stale-request-timeout-secs",
            "240",
        ]))
        .unwrap();
        assert_eq!(c.proxy.request_timeout_secs, 120);
        assert_eq!(c.active_load.stale_request_timeout_secs, 240);
    }

    #[test]
    fn sticky_policy_defaults_header_and_tuning() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "sticky",
        ]))
        .unwrap();
        assert_eq!(c.model.policy, PolicyKind::Sticky);
        let s = c.model.sticky.expect("sticky config built");
        assert_eq!(s.header_name, "x-sgl-routing-key");
        assert_eq!(s.fallback_policy, PolicyKind::RoundRobin);
        assert_eq!(s.idle_secs, 600);
        assert_eq!(s.eviction_interval_secs, 60);
    }

    #[test]
    fn sticky_flags_override_defaults() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "sticky",
            "--routing-key-header",
            "x-session-id",
            "--sticky-fallback-policy",
            "load_based",
            "--sticky-idle-secs",
            "120",
            "--sticky-eviction-interval-secs",
            "15",
        ]))
        .unwrap();
        let s = c.model.sticky.expect("sticky config built");
        assert_eq!(s.header_name, "x-session-id");
        assert_eq!(s.fallback_policy, PolicyKind::LoadBased);
        assert_eq!(s.idle_secs, 120);
        assert_eq!(s.eviction_interval_secs, 15);
    }

    #[test]
    fn non_sticky_policy_leaves_sticky_none() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "round_robin",
        ]))
        .unwrap();
        assert!(c.model.sticky.is_none());
    }

    #[test]
    fn rejects_sticky_flags_without_sticky_policy() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--routing-key-header",
            "x-session-id",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("require --policy sticky"), "got: {err}");
    }

    #[test]
    fn rejects_invalid_routing_key_header() {
        // A space is not a legal HTTP header-name character.
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "sticky",
            "--routing-key-header",
            "bad header",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("not a valid HTTP header name"), "got: {err}");
    }

    #[test]
    fn rejects_cache_aware_zmq_as_sticky_fallback() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "sticky",
            "--sticky-fallback-policy",
            "cache_aware_zmq",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("--sticky-fallback-policy must be one of"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_sticky_as_sticky_fallback() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "sticky",
            "--sticky-fallback-policy",
            "sticky",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("--sticky-fallback-policy must be one of"),
            "got: {err}"
        );
    }

    /// A zero eviction interval would panic `tokio::time::interval` at
    /// startup — reject it at config-build time with a clear message.
    #[test]
    fn rejects_zero_sticky_eviction_interval() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "sticky",
            "--sticky-eviction-interval-secs",
            "0",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("--sticky-eviction-interval-secs must be greater than 0"),
            "got: {err}"
        );
    }

    #[test]
    fn local_prefix_fallback_is_opt_in_and_requires_hint_configuration() {
        let flags = [
            "--worker-urls",
            "http://x:30000",
            "--inject-kv-pull-hint",
            "--kv-indexer-url",
            "http://indexer:8090",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://x:30000","servicer":"x:18100"}]"#,
        ];
        assert!(
            !into_config_owned(with_model(&flags))
                .unwrap()
                .model
                .kv_hint
                .unwrap()
                .local_prefix_fallback
        );
        let mut enabled = flags.to_vec();
        enabled.push("--kv-pull-hint-local-prefix-fallback");
        assert!(
            into_config_owned(with_model(&enabled))
                .unwrap()
                .model
                .kv_hint
                .unwrap()
                .local_prefix_fallback
        );
        assert!(into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--kv-pull-hint-local-prefix-fallback",
        ]))
        .is_err());
    }

    #[test]
    fn rejects_zero_sticky_idle() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "sticky",
            "--sticky-idle-secs",
            "0",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("--sticky-idle-secs must be greater than 0"),
            "got: {err}"
        );
    }

    // ---- symmetric_pd ----

    #[test]
    fn symmetric_pd_zero_hold_is_valid_without_changing_default() {
        let base = [
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://p:30000,8998,1,1",
        ];
        let default = into_config_owned(with_model(&base)).unwrap();
        assert_eq!(default.model.symmetric_pd.unwrap().max_queue_wait_ms, 2000);
        let mut args = base.to_vec();
        args.extend(["--symmetric-pd-max-queue-wait-ms", "0"]);
        let disabled = into_config_owned(with_model(&args)).unwrap();
        assert_eq!(disabled.model.symmetric_pd.unwrap().max_queue_wait_ms, 0);
        *args.last_mut().unwrap() = "30001";
        assert!(into_config_owned(with_model(&args)).is_err());
    }

    #[test]
    fn symmetric_pd_workers_build_config_and_join_static_discovery() {
        let c = into_config_owned(with_model(&[
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://10.0.0.1:30000,8998,1,0",
            "--symmetric-pd-worker",
            "http://10.0.0.5:30000/,8998,0,1",
            "--symmetric-pd-force-split",
            "--symmetric-pd-max-queue-wait-ms",
            "1500",
            "--symmetric-pd-w-dload",
            "0",
        ]))
        .unwrap();
        assert_eq!(c.model.policy, PolicyKind::SymmetricPd);
        let spd = c.model.symmetric_pd.as_ref().expect("symmetric_pd config");
        assert!(spd.placement.force_split);
        assert_eq!(spd.max_queue_wait_ms, 1500);
        assert_eq!(spd.placement.weights.w_dload, 0.0);
        assert_eq!(spd.placement.weights.w_hit, 1.0);
        assert_eq!(spd.workers.len(), 2);
        assert_eq!(spd.workers[0].bootstrap_port, Some(8998));
        assert_eq!(spd.workers[0].prefill_weight, 1.0);
        assert_eq!(spd.workers[0].decode_weight, 0.0);
        assert!(spd
            .worker_spec_for("http://10.0.0.5:30000")
            .is_some_and(|w| w.decode_weight == 1.0));
        // No --worker-urls: the symmetric P/D workers ARE the static fleet.
        match &c.discovery {
            DiscoveryBackend::StaticUrls(s) => assert_eq!(
                s.urls,
                vec![
                    "http://10.0.0.1:30000".to_string(),
                    "http://10.0.0.5:30000/".to_string()
                ]
            ),
            _ => panic!("expected static_urls backend"),
        }
        assert!(
            c.model.kv_hint.is_none(),
            "indexer is optional under symmetric_pd"
        );
    }

    #[test]
    fn symmetric_pd_import_cache_policy_flag_maps() {
        let base = [
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://10.0.0.1:30000,8998,1,1",
        ];
        let c = into_config_owned(with_model(&base)).unwrap();
        assert_eq!(
            c.model.symmetric_pd.as_ref().unwrap().import_cache_policy,
            SymmetricPdImportCachePolicy::Auto
        );
        let mut args = base.to_vec();
        args.extend(["--symmetric-pd-import-cache-policy", "gpu-only"]);
        let c = into_config_owned(with_model(&args)).unwrap();
        assert_eq!(
            c.model.symmetric_pd.as_ref().unwrap().import_cache_policy,
            SymmetricPdImportCachePolicy::GpuOnly
        );
        let args = ["--symmetric-pd-import-cache-policy", "drop"];
        let err = into_config_owned(with_model(&args))
            .unwrap_err()
            .to_string();
        assert!(err.contains("require --policy symmetric_pd"), "{err}");
    }

    #[test]
    fn decode_host_return_config_is_opt_in_and_does_not_require_indexer() {
        let base = [
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://p:30000/,8998,1,0",
            "--symmetric-pd-worker",
            "http://d:30000,8998,0,1",
        ];
        let default = into_config_owned(with_model(&base)).unwrap();
        let default = default.model.symmetric_pd.unwrap();
        assert!(default.decode_host_return.is_none());
        assert_eq!(
            default.import_cache_policy,
            SymmetricPdImportCachePolicy::Auto
        );
        let mapping = r#"[{"id":1,"url":"http://p:30000","servicer":"p:18100"},{"id":2,"url":"http://d:30000/","servicer":"d:18100"}]"#;
        let mut args = base.to_vec();
        args.extend([
            "--symmetric-pd-decode-host-return",
            "--kv-worker-map",
            mapping,
        ]);
        let config = into_config_owned(with_model(&args)).unwrap();
        assert!(config.model.kv_hint.is_none());
        let spd = config.model.symmetric_pd.unwrap();
        assert_eq!(
            spd.import_cache_policy,
            SymmetricPdImportCachePolicy::Retain
        );
        let returns = spd.decode_host_return.unwrap();
        assert_eq!(returns.servicers["http://p:30000"], "p:18100");
        assert_eq!(returns.servicers["http://d:30000"], "d:18100");
        assert_eq!((returns.timeout_ms, returns.max_tokens), (10_000, 65_536));
        for policy in ["auto", "gpu-only", "drop"] {
            let mut incompatible = args.clone();
            incompatible.extend(["--symmetric-pd-import-cache-policy", policy]);
            assert!(into_config_owned(with_model(&incompatible))
                .unwrap_err()
                .to_string()
                .contains("requires --symmetric-pd-import-cache-policy retain"));
        }
        args.extend([
            "--symmetric-pd-decode-host-return-max-tokens",
            "128",
            "--symmetric-pd-decode-host-return-timeout-ms",
            "2000",
        ]);
        let config = into_config_owned(with_model(&args)).unwrap();
        let returns = config
            .model
            .symmetric_pd
            .unwrap()
            .decode_host_return
            .unwrap();
        assert_eq!((returns.timeout_ms, returns.max_tokens), (2000, 128));
    }

    #[test]
    fn decode_host_return_rejects_missing_or_ambiguous_peer_configuration() {
        let base = [
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://p:30000,8998,1,0",
            "--symmetric-pd-worker",
            "http://d:30000,8998,0,1",
            "--symmetric-pd-decode-host-return",
        ];
        assert!(into_config_owned(with_model(&base))
            .unwrap_err()
            .to_string()
            .contains("requires --kv-worker-map"));
        for mapping in [
            "[]",
            r#"[{"id":1,"url":"http://p:30000","servicer":"p:18100"}]"#,
            r#"[{"id":1,"url":"http://p:30000","servicer":""}]"#,
            r#"[{"id":1,"url":"http://p:30000","servicer":"p:0"}]"#,
            r#"[{"id":1,"url":"http://p:30000","servicer":"p:18100/x"}]"#,
            r#"[{"id":1,"url":"http://p:30000","servicer":"p:18100/"}]"#,
            r#"[{"id":1,"url":"http://p:30000","servicer":"p:18100"},{"id":2,"url":"http://p:30000/","servicer":"d:18100"}]"#,
            r#"[{"id":1,"url":"http://p:30000","servicer":"p:18100"},{"id":2,"url":"http://d:30000","servicer":"p:18100"}]"#,
        ] {
            let mut args = base.to_vec();
            args.extend(["--kv-worker-map", mapping]);
            assert!(
                into_config_owned(with_model(&args)).is_err(),
                "accepted {mapping}"
            );
        }
        let args = [
            "--worker-urls",
            "http://p:30000",
            "--symmetric-pd-decode-host-return",
        ];
        assert!(into_config_owned(with_model(&args))
            .unwrap_err()
            .to_string()
            .contains("require --policy symmetric_pd"));
    }

    #[test]
    fn decode_host_return_endpoint_matches_engine_grammar() {
        for (raw, normalized) in [
            ("Peer.EXAMPLE.:01810", "peer.example:1810"),
            ("127.0.0.1:65535", "127.0.0.1:65535"),
        ] {
            assert_eq!(normalize_host_return_endpoint(raw).unwrap(), normalized);
        }
        for raw in [
            "",
            " peer:18100",
            "peer:18100 ",
            "peer:0",
            "peer:65536",
            "peer:000001",
            "peer:1/x",
            "peer:1/",
            "peer:1?x",
            "peer:1#x",
            "tcp://peer:1",
            "a@peer:1",
            "peer_name:1",
            "-peer:1",
            "peer-:1",
            "a..b:1",
            "[127.0.0.1]:1",
            "[fe80::1%eth0]:1",
            "::1:18100",
            "[::1]:+1",
            "[2001:db8::1]:18100",
            "[::1]:18100",
        ] {
            assert!(
                normalize_host_return_endpoint(raw).is_err(),
                "accepted {raw}"
            );
        }
    }

    #[test]
    fn decode_host_return_rejects_endpoint_aliases_and_engine_limit_overflow() {
        let base = [
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://p:30000,8998,1,0",
            "--symmetric-pd-worker",
            "http://d:30000,8998,0,1",
            "--symmetric-pd-decode-host-return",
            "--kv-worker-map",
        ];
        for pair in [("Peer.:01810", "peer:1810"), ("P:00080", "p:80")] {
            let mapping = serde_json::json!([
                {"id":1, "url":"http://p:30000", "servicer":pair.0},
                {"id":2, "url":"http://d:30000", "servicer":pair.1},
            ])
            .to_string();
            let mut args = base.to_vec();
            args.push(&mapping);
            assert!(into_config_owned(with_model(&args))
                .unwrap_err()
                .to_string()
                .contains("distinct servicer addresses"));
        }
        let mapping = r#"[{"id":1,"url":"http://p:30000","servicer":"p:18100"},{"id":2,"url":"http://d:30000","servicer":"d:18100"}]"#;
        for (flag, value) in [
            ("--symmetric-pd-decode-host-return-timeout-ms", "0"),
            ("--symmetric-pd-decode-host-return-timeout-ms", "60001"),
            ("--symmetric-pd-decode-host-return-max-tokens", "0"),
            ("--symmetric-pd-decode-host-return-max-tokens", "1048577"),
        ] {
            let mut args = base.to_vec();
            args.extend([mapping, flag, value]);
            assert!(into_config_owned(with_model(&args))
                .unwrap_err()
                .to_string()
                .contains("timeout must be in 1..=60000"));
        }
        let mut args = base.to_vec();
        args.extend([
            mapping,
            "--symmetric-pd-decode-host-return-timeout-ms",
            "60000",
            "--symmetric-pd-decode-host-return-max-tokens",
            "1048576",
        ]);
        assert!(into_config_owned(with_model(&args)).is_ok());
    }

    #[test]
    fn pd_decision_replay_flag_maps_and_validates() {
        let base = [
            "--policy",
            "cache_aware_zmq",
            "--worker-urls",
            "http://10.0.0.1:30000",
        ];
        let c = into_config_owned(with_model(&base)).unwrap();
        assert!(c.model.pd_decision_replay.is_none());
        let mut args = base.to_vec();
        args.extend(["--pd-decision-replay-header", "X-Key"]);
        let err = into_config_owned(with_model(&args))
            .unwrap_err()
            .to_string();
        assert!(err.contains("requires --pd-decision-replay"), "{err}");
        let mut args = base.to_vec();
        args.extend(["--pd-decision-replay", "/nonexistent/replay.json"]);
        let err = into_config_owned(with_model(&args))
            .unwrap_err()
            .to_string();
        assert!(err.contains("not a readable file"), "{err}");
        let dir = std::env::temp_dir().join(format!("pd-replay-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("replay.json");
        std::fs::write(&path, r#"{"key_hash":"sha256","sessions":{}}"#).unwrap();
        let path_s = path.to_string_lossy().to_string();
        let mut args = base.to_vec();
        args.extend(["--pd-decision-replay", path_s.as_str()]);
        let c = into_config_owned(with_model(&args)).unwrap();
        let r = c.model.pd_decision_replay.as_ref().unwrap();
        assert_eq!(r.table, path);
        assert_eq!(r.header, "X-SMG-Routing-Key");
        let mut args = base.to_vec();
        args.extend([
            "--pd-decision-replay",
            path_s.as_str(),
            "--pd-decision-replay-header",
            " X-Key ",
        ]);
        let c = into_config_owned(with_model(&args)).unwrap();
        assert_eq!(c.model.pd_decision_replay.as_ref().unwrap().header, "X-Key");
        // symmetric_pd has its own replay flag.
        let args = [
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://10.0.0.1:30000,8998,1,1",
            "--symmetric-pd-worker",
            "http://10.0.0.5:30000,8998,1,1",
            "--pd-decision-replay",
            path_s.as_str(),
        ];
        let err = into_config_owned(with_model(&args))
            .unwrap_err()
            .to_string();
        assert!(err.contains("traditional PD path"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn symmetric_pd_decision_replay_flag_maps_and_validates() {
        let base = [
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://10.0.0.1:30000,8998,1,1",
            "--symmetric-pd-worker",
            "http://10.0.0.5:30000,8998,1,1",
        ];
        let c = into_config_owned(with_model(&base)).unwrap();
        let f = c.model.symmetric_pd.as_ref().unwrap();
        assert!(f.decision_replay.is_none());
        assert_eq!(f.decision_replay_header, "X-SMG-Routing-Key");
        // Header without a table is an orphan flag.
        let mut args = base.to_vec();
        args.extend(["--symmetric-pd-decision-replay-header", "X-Key"]);
        let err = into_config_owned(with_model(&args))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("requires --symmetric-pd-decision-replay"),
            "{err}"
        );
        // Missing file is rejected at config time.
        let mut args = base.to_vec();
        args.extend(["--symmetric-pd-decision-replay", "/nonexistent/replay.json"]);
        let err = into_config_owned(with_model(&args))
            .unwrap_err()
            .to_string();
        assert!(err.contains("not a readable file"), "{err}");
        // An existing file maps through, with a custom header.
        let dir = std::env::temp_dir().join(format!("symmetric_pd-replay-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("replay.json");
        std::fs::write(&path, r#"{"key_hash":"sha256","sessions":{}}"#).unwrap();
        let path_s = path.to_string_lossy().to_string();
        let mut args = base.to_vec();
        args.extend([
            "--symmetric-pd-decision-replay",
            path_s.as_str(),
            "--symmetric-pd-decision-replay-header",
            " X-Key ",
        ]);
        let c = into_config_owned(with_model(&args)).unwrap();
        let f = c.model.symmetric_pd.as_ref().unwrap();
        assert_eq!(f.decision_replay.as_deref(), Some(path.as_path()));
        assert_eq!(f.decision_replay_header, "X-Key");
        // Orphan without --policy symmetric_pd.
        let args = ["--symmetric-pd-decision-replay", path_s.as_str()];
        let err = into_config_owned(with_model(&args))
            .unwrap_err()
            .to_string();
        assert!(err.contains("require --policy symmetric_pd"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn role_drain_timeout_requires_external_role_control_and_is_positive() {
        let base = [
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://10.0.0.1:30000,8998,1,0",
            "--symmetric-pd-worker",
            "http://10.0.0.5:30000,8998,0,1",
        ];
        let c = into_config_owned(with_model(&base)).unwrap();
        assert!(c
            .model
            .symmetric_pd
            .as_ref()
            .unwrap()
            .role_drain_timeout
            .is_none());
        let mut args = base.to_vec();
        args.extend(["--symmetric-pd-role-drain-timeout-secs", "30"]);
        assert!(into_config_owned(with_model(&args)).is_err());
        args.push("--symmetric-pd-external-role-control");
        let c = into_config_owned(with_model(&args)).unwrap();
        let spd = c.model.symmetric_pd.as_ref().unwrap();
        assert!(spd.external_role_control);
        assert_eq!(
            spd.role_drain_timeout,
            Some(std::time::Duration::from_secs(30))
        );
        let mut zero = base.to_vec();
        zero.extend([
            "--symmetric-pd-external-role-control",
            "--symmetric-pd-role-drain-timeout-secs",
            "0",
        ]);
        assert!(into_config_owned(with_model(&zero)).is_err());
        let mut default = base.to_vec();
        default.push("--symmetric-pd-external-role-control");
        let c = into_config_owned(with_model(&default)).unwrap();
        assert_eq!(
            c.model.symmetric_pd.as_ref().unwrap().role_drain_timeout,
            Some(std::time::Duration::from_secs(5))
        );
    }

    #[test]
    fn symmetric_pd_host_hit_factor_flag_maps_and_validates() {
        let base = [
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://10.0.0.1:30000,8998,1,1",
            "--symmetric-pd-worker",
            "http://10.0.0.5:30000,8998,1,1",
        ];
        let c = into_config_owned(with_model(&base)).unwrap();
        assert_eq!(
            c.model
                .symmetric_pd
                .as_ref()
                .unwrap()
                .placement
                .host_hit_factor,
            1.0,
            "default"
        );
        let mut args = base.to_vec();
        args.extend(["--symmetric-pd-host-hit-factor", "0"]);
        let c = into_config_owned(with_model(&args)).unwrap();
        assert_eq!(
            c.model
                .symmetric_pd
                .as_ref()
                .unwrap()
                .placement
                .host_hit_factor,
            0.0
        );
        let mut args = base.to_vec();
        args.extend(["--symmetric-pd-pull-hint"]);
        let c = into_config_owned(with_model(&args)).unwrap();
        assert!(c.model.symmetric_pd.as_ref().unwrap().pull_hint);
        let mut args = base.to_vec();
        args.extend(["--symmetric-pd-host-hit-factor", "1.5"]);
        let err = into_config_owned(with_model(&args))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--symmetric-pd-host-hit-factor"), "{err}");
    }

    #[test]
    fn kv_session_retention_is_opt_in_and_does_not_change_placement() {
        let base = [
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://p:30000,8998,1,0",
            "--symmetric-pd-worker",
            "http://d:30000,8998,0,1",
        ];
        let default = into_config_owned(with_model(&base))
            .unwrap()
            .model
            .symmetric_pd
            .unwrap();
        assert!(!default.kv_session_retention);
        let mut args = base.to_vec();
        args.extend([
            "--symmetric-pd-kv-session-retention",
            "--symmetric-pd-kv-session-key-file",
            "/tmp/api-key",
        ]);
        let enabled = into_config_owned(with_model(&args))
            .unwrap()
            .model
            .symmetric_pd
            .unwrap();
        assert!(enabled.kv_session_retention);
        assert_eq!(enabled.placement, default.placement);
        assert_eq!(
            enabled.kv_session_key_file.unwrap(),
            std::path::PathBuf::from("/tmp/api-key")
        );
    }

    #[test]
    fn gateway_cache_aware_defaults_g1_values_and_refusals() {
        let base = [
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "gateway_cache_aware",
        ];
        // Defaults = the sgl-model-gateway CLI defaults.
        let cfg = into_config_owned(with_model(&base)).unwrap();
        assert_eq!(cfg.model.policy, PolicyKind::GatewayCacheAware);
        assert_eq!(
            cfg.model.gateway_cache_aware.clone().unwrap(),
            GatewayCacheAwareConfig::default()
        );
        assert!(cfg.model.kv_hint.is_none());
        assert!(cfg.model.pd_kv_session_retention().is_none());

        // Baseline values, late D choose and session retention.
        let mut args = base.to_vec();
        args.extend([
            "--gateway-ca-cache-threshold",
            "0.3",
            "--gateway-ca-balance-abs-threshold",
            "64",
            "--gateway-ca-balance-rel-threshold",
            "1.5",
            "--gateway-ca-eviction-interval-secs",
            "120",
            "--gateway-ca-max-tree-size",
            "66163200",
            "--pd-decode-policy",
            "power-of-two-smg",
            "--pd-decode-load-check-interval-secs",
            "5",
            "--pd-native-decode-state",
            "--pd-late-decode",
            "choose",
            "--pd-late-decode-rule",
            "fit_req",
            "--pd-late-decode-candidates",
            "all",
            "--pd-late-decode-local",
            "off",
            "--pd-late-decode-fit-native",
            "--pd-late-decode-fit-state-max-age-ms",
            "1000",
            "--pd-late-decode-packed-prompt",
            "--kv-session-retention",
            "--kv-session-key-file",
            "/tmp/api-key",
        ]);
        let cfg = into_config_owned(with_model(&args)).unwrap();
        let g = cfg.model.gateway_cache_aware.clone().unwrap();
        assert_eq!(g.cache_threshold, 0.3);
        assert_eq!(g.balance_abs_threshold, 64);
        assert_eq!(g.balance_rel_threshold, 1.5);
        assert_eq!(g.eviction_interval_secs, 120);
        assert_eq!(g.max_tree_size, 66_163_200);
        assert!(g.kv_session_retention);
        assert!(cfg.proxy.late_decode.choose());
        assert_eq!(
            cfg.model.pd_kv_session_retention(),
            Some(Some(std::path::Path::new("/tmp/api-key")))
        );

        // --gateway-ca-* needs the policy.
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--gateway-ca-max-tree-size",
            "1",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("require --policy gateway_cache_aware"),
            "{err}"
        );

        // The gateway's value checks.
        for (flag, value, want) in [
            ("--gateway-ca-cache-threshold", "1.5", "0..=1"),
            ("--gateway-ca-balance-rel-threshold", "0.9", "at least 1"),
            ("--gateway-ca-eviction-interval-secs", "0", "must be > 0"),
            ("--gateway-ca-max-tree-size", "0", "must be > 0"),
        ] {
            let mut args = base.to_vec();
            args.extend([flag, value]);
            let err = into_config_owned(with_model(&args))
                .unwrap_err()
                .to_string();
            assert!(err.contains(want), "{flag} {value}: {err}");
        }
    }

    #[test]
    fn symmetric_pd_selection_flags_map_into_placement_config() {
        use crate::scheduler::placement::{DSelect, PSelect};
        let c = into_config_owned(with_model(&[
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://10.0.0.1:30000,8998,1,0",
            "--symmetric-pd-worker",
            "http://10.0.0.5:30000,8998,0,1",
        ]))
        .unwrap();
        let spd = c.model.symmetric_pd.as_ref().unwrap();
        assert_eq!(spd.placement.p_select, PSelect::CacheFirst, "default");
        assert_eq!(spd.placement.d_select, DSelect::Score, "default");
        assert_eq!(spd.placement.cache_threshold, 0.5);
        assert_eq!(spd.placement.balance_abs_reqs, 32);
        assert_eq!(spd.placement.balance_rel, 1.1);

        let c = into_config_owned(with_model(&[
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://10.0.0.1:30000,8998,1,0",
            "--symmetric-pd-worker",
            "http://10.0.0.5:30000,8998,0,1",
            "--symmetric-pd-p-select",
            "score",
            "--symmetric-pd-d-select",
            "round-robin",
            "--symmetric-pd-cache-threshold",
            "0.25",
            "--symmetric-pd-balance-abs-reqs",
            "8",
            "--symmetric-pd-balance-rel",
            "1.5",
        ]))
        .unwrap();
        let spd = c.model.symmetric_pd.as_ref().unwrap();
        assert_eq!(spd.placement.p_select, PSelect::Score);
        assert_eq!(spd.placement.d_select, DSelect::RoundRobin);
        assert_eq!(spd.placement.cache_threshold, 0.25);
        assert_eq!(spd.placement.balance_abs_reqs, 8);
        assert_eq!(spd.placement.balance_rel, 1.5);

        let c = into_config_owned(with_model(&[
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://10.0.0.1:30000,8998,1,0",
            "--symmetric-pd-worker",
            "http://10.0.0.5:30000,8998,0,1",
            "--symmetric-pd-d-select",
            "cache-first",
        ]))
        .unwrap();
        let spd = c.model.symmetric_pd.as_ref().unwrap();
        assert_eq!(spd.placement.d_select, DSelect::CacheFirst);

        for bad in [
            ["--symmetric-pd-cache-threshold", "1.5"],
            ["--symmetric-pd-balance-rel", "0.9"],
        ] {
            let mut args = vec![
                "--policy",
                "symmetric_pd",
                "--symmetric-pd-worker",
                "http://10.0.0.1:30000,8998,1,1",
            ];
            args.extend(bad);
            assert!(into_config_owned(with_model(&args)).is_err(), "{bad:?}");
        }
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--symmetric-pd-d-select",
            "round-robin",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("require --policy symmetric_pd"), "got: {err}");
    }

    #[test]
    fn symmetric_pd_flags_require_the_policy() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--symmetric-pd-force-split",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("require --policy symmetric_pd"), "got: {err}");
    }

    #[test]
    fn symmetric_pd_queue_wait_is_capped_at_30s() {
        let err = into_config_owned(with_model(&[
            "--policy",
            "symmetric_pd",
            "--worker-urls",
            "http://x:30000",
            "--symmetric-pd-max-queue-wait-ms",
            "30001",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("--symmetric-pd-max-queue-wait-ms"),
            "got: {err}"
        );
        let c = into_config_owned(with_model(&[
            "--policy",
            "symmetric_pd",
            "--worker-urls",
            "http://x:30000",
        ]))
        .unwrap();
        assert_eq!(c.model.symmetric_pd.unwrap().max_queue_wait_ms, 2000);
    }

    #[test]
    fn symmetric_pd_accepts_indexer_flags_without_inject_and_without_servicer() {
        let c = into_config_owned(with_model(&[
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://a:30000,8998",
            "--kv-indexer-url",
            "http://indexer:8090",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://a:30000"}]"#,
        ]))
        .unwrap();
        let kv = c
            .model
            .kv_hint
            .expect("indexer config built for symmetric_pd");
        assert_eq!(kv.indexer_url, "http://indexer:8090");
        assert_eq!(kv.worker_map[0].url, "http://a:30000");
        assert!(kv.worker_map[0].servicer.is_empty());
    }

    #[test]
    fn symmetric_pd_worker_spec_parse_cli_shapes() {
        let w = SymmetricPdWorkerSpec::parse_cli("http://a:30000,8998,1,0").unwrap();
        assert_eq!(
            (w.bootstrap_port, w.prefill_weight, w.decode_weight),
            (Some(8998), 1.0, 0.0)
        );
        let w = SymmetricPdWorkerSpec::parse_cli("http://a:30000").unwrap();
        assert_eq!(
            (w.bootstrap_port, w.prefill_weight, w.decode_weight),
            (None, 1.0, 1.0)
        );
        let w = SymmetricPdWorkerSpec::parse_cli("http://a:30000,none,0.5").unwrap();
        assert_eq!((w.bootstrap_port, w.prefill_weight), (None, 0.5));
        assert!(SymmetricPdWorkerSpec::parse_cli("").is_err());
        assert!(SymmetricPdWorkerSpec::parse_cli("http://a,notaport").is_err());
        assert!(SymmetricPdWorkerSpec::parse_cli("http://a,1,-1").is_err());
        assert!(SymmetricPdWorkerSpec::parse_cli("http://a,1,1,1,1").is_err());
        let dup = into_config_owned(with_model(&[
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://a:30000,8998",
            "--symmetric-pd-worker",
            "http://a:30000/,8998",
        ]))
        .unwrap_err()
        .to_string();
        assert!(dup.contains("listed twice"), "got: {dup}");
    }

    #[test]
    fn symmetric_pd_workers_file_is_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workers.json");
        std::fs::write(
            &path,
            r#"[{"url":"http://f:30000","bootstrap_port":9000,"prefill_weight":0.2,
                 "capabilities":["prefill"]}]"#,
        )
        .unwrap();
        let c = into_config_owned(with_model(&[
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-workers-file",
            path.to_str().unwrap(),
        ]))
        .unwrap();
        let spd = c.model.symmetric_pd.unwrap();
        assert_eq!(spd.workers.len(), 1);
        let w = &spd.workers[0];
        assert_eq!(w.prefill_weight, 0.2);
        assert_eq!(w.decode_weight, 1.0);
        assert!(w.capabilities().prefill && !w.capabilities().decode);
    }

    /// The `--kv-worker-map` of the cost-function argv: the three P-role
    /// workers with their servicers. D-role workers are not in the map.
    const COST_FUNCTION_MAP: &str = r#"[
        {"id":11,"url":"http://10.0.0.1:18080","servicer":"10.0.0.1:8000"},
        {"id":12,"url":"http://10.0.0.2:18080","servicer":"10.0.0.2:8000"},
        {"id":13,"url":"http://10.0.0.3:18080","servicer":"10.0.0.3:8000"}]"#;

    /// A cost-function router argv with
    /// dummy URLs: 3 P-role and 6 D-role workers. One flag and its value
    /// per group.
    fn cost_function_argv() -> Vec<Vec<&'static str>> {
        vec![
            vec!["--host", "127.0.0.1"],
            vec!["--port", "30080"],
            vec!["--symmetric-pd-worker", "http://10.0.0.1:18080,18998,1,0"],
            vec!["--symmetric-pd-worker", "http://10.0.0.2:18080,18998,1,0"],
            vec!["--symmetric-pd-worker", "http://10.0.0.3:18080,18998,1,0"],
            vec!["--symmetric-pd-worker", "http://10.0.0.4:18081,18998,0,1"],
            vec!["--symmetric-pd-worker", "http://10.0.0.5:18081,18998,0,1"],
            vec!["--symmetric-pd-worker", "http://10.0.0.6:18081,18998,0,1"],
            vec!["--symmetric-pd-worker", "http://10.0.0.7:18081,18998,0,1"],
            vec!["--symmetric-pd-worker", "http://10.0.0.8:18081,18998,0,1"],
            vec!["--symmetric-pd-worker", "http://10.0.0.9:18081,18998,0,1"],
            vec!["--request-timeout-secs", "3600"],
            vec!["--stale-request-timeout-secs", "3600"],
            vec!["--policy", "symmetric_pd"],
            vec!["--symmetric-pd-force-split"],
            vec!["--symmetric-pd-p-select", "cost-function"],
            vec!["--symmetric-pd-pull-hint"],
            vec!["--copy-price", "auto"],
            vec!["--copy-price-fallback", "18.4"],
            vec!["--prefill-poll-ms", "100"],
            vec!["--placement-trace", "/tmp/placement-trace.jsonl"],
            vec!["--kv-indexer-url", "http://127.0.0.1:9100"],
            vec!["--kv-worker-map", COST_FUNCTION_MAP],
            vec!["--kv-pull-hint-min-tokens", "256"],
            vec!["--kv-pull-hint-min-incremental-tokens", "512"],
            vec!["--kv-indexer-timeout-ms", "200"],
            vec!["--symmetric-pd-d-select", "round-robin"],
            vec!["--symmetric-pd-import-cache-policy", "retain"],
            vec!["--pd-native-decode-state"],
            vec!["--pd-late-decode", "choose"],
            vec!["--pd-late-decode-rule", "fit_req"],
            vec!["--pd-late-decode-candidates", "all"],
            vec!["--pd-late-decode-local", "off"],
            vec!["--pd-late-decode-fit-native"],
            vec!["--pd-late-decode-fit-state-max-age-ms", "1000"],
            vec!["--pd-late-decode-packed-prompt"],
            vec!["--symmetric-pd-kv-session-retention"],
            vec!["--disable-circuit-breaker"],
            vec!["--proxy-tcp-keepalive-secs", "30"],
            vec!["--pd-abort-decode-on-prefill-failure"],
            vec!["--log-level", "info"],
        ]
    }

    /// Parse the cost-function argv without each group whose text (flag and
    /// value, space separated) starts with an item of `drop`, plus `add`.
    fn cost_function_config(drop: &[&str], add: &[&str]) -> Result<Config> {
        let mut args: Vec<&str> = cost_function_argv()
            .into_iter()
            .filter(|group| !drop.iter().any(|d| group.join(" ").starts_with(d)))
            .flatten()
            .collect();
        args.extend(add);
        into_config_owned(with_model(&args))
    }

    #[test]
    fn symmetric_pd_cost_function_accepts_the_startup_argv() {
        use crate::config::{LateDecodeMode, LateDecodeRule};
        use crate::scheduler::placement::{DSelect, PSelect};
        use crate::scheduler::placement_cost::{CopyPriceMode, PrefillRateMode};
        let cfg = cost_function_config(&[], &[]).unwrap();
        let kv = cfg.model.kv_hint.as_ref().unwrap();
        assert_eq!((kv.min_tokens, kv.min_incremental_tokens), (256, 512));
        assert_eq!(kv.worker_map.len(), 3);
        let spd = cfg.model.symmetric_pd.as_ref().unwrap();
        assert_eq!(spd.workers.len(), 9);
        assert_eq!(spd.placement.p_select, PSelect::CostFunction);
        assert_eq!(spd.placement.d_select, DSelect::RoundRobin);
        assert!(spd.placement.force_split && spd.pull_hint && spd.kv_session_retention);
        let cost = spd.cost_function.as_ref().unwrap();
        assert_eq!(cost.load_poll_ms, 100);
        assert_eq!(cost.unknown_expire_secs, 60, "the default unknown expiry");
        assert_eq!(
            cost.placement_trace.as_deref().and_then(|p| p.to_str()),
            Some("/tmp/placement-trace.jsonl")
        );
        let cost_params = cost.cost_params;
        assert_eq!(
            cost_params.copy_price_mode,
            CopyPriceMode::Auto {
                min: 4.6,
                max: 1000.0,
            }
        );
        assert_eq!(cost_params.copy_price, 18.4);
        assert_eq!(
            (
                cost_params.prefill_rate_mode,
                cost_params.prefill_rate_tokens_per_s
            ),
            (PrefillRateMode::Fixed, 13_500.0)
        );
        assert!(cost_params.is_auto());
        let late = &cfg.proxy.late_decode;
        assert_eq!(late.mode, LateDecodeMode::Choose);
        assert_eq!(late.rule, Some(LateDecodeRule::FitReq));
        assert!(late.fit_native && late.native_state_poll && late.packed_prompt);
    }

    /// `--prefill-wait forecast` and `--global-queue-room forecast` on the
    /// cost function. The paper preset turns both on with the built-in error
    /// table; the paper's runs pin `--prefill-wait tokens
    /// --global-queue=false`; each flag that needs the forecast is refused
    /// without it.
    #[test]
    fn symmetric_pd_cost_function_forecast_and_queue_room() {
        use crate::global_queue::GlobalQueueRoom;
        use crate::scheduler::placement_cost::PrefillWait;
        let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/forecast");
        let table = format!("{fixtures}/bound-params.json");
        let cost = |add: &[&str]| {
            cost_function_config(&[], add)
                .map(|c| c.model.symmetric_pd.unwrap().cost_function.unwrap())
        };
        let paper = |add: &[&str]| {
            let mut args = vec!["--symmetric-pd-preset", "paper"];
            args.extend(add);
            cost(&args)
        };
        let with_pins = |add: &[&str]| {
            let mut args = vec!["--prefill-wait", "tokens", "--global-queue=false"];
            args.extend(add);
            paper(&args)
        };
        // The paper preset: forecast wait, the global queue with the
        // forecast room, the next-pass lag, m = 0, a 5 s state age and the
        // built-in error table.
        let preset = paper(&[]).unwrap();
        assert_eq!(preset.cost_params.wait, PrefillWait::Forecast);
        assert_eq!(preset.cost_params.move_margin, 0.0);
        assert_eq!(preset.cost_params.max_state_age_s, 5.0);
        assert_eq!(preset.global_queue.unwrap().room, GlobalQueueRoom::Forecast);
        let state = &preset.forecast.as_ref().unwrap().0;
        assert!(state.next_pass_lag && state.push.is_none() && state.frozen_fit.is_none());
        assert_eq!(
            state.bound_path,
            std::path::Path::new(crate::prefill_forecast::BUILTIN_ERROR_TABLE_PATH)
        );
        // The built-in table is the fixture: the same state as the file.
        assert_eq!(
            preset.forecast,
            paper(&["--forecast-error-table", &table]).unwrap().forecast
        );
        // The paper's runs: the token-count wait, no queue, no forecast state.
        let pinned = with_pins(&[]).unwrap();
        assert_eq!(pinned.cost_params.wait, PrefillWait::Tokens);
        assert_eq!(pinned.cost_params.move_margin, 1.0);
        assert!(pinned.global_queue.is_none() && pinned.forecast.is_none());
        // `--symmetric-pd-preset none`: the same as the pins.
        let none = cost(&[]).unwrap();
        assert_eq!(none.cost_params, pinned.cost_params);
        assert!(none.global_queue.is_none() && none.forecast.is_none());
        // Given flags win over the preset.
        let given = paper(&[
            "--move-margin",
            "1",
            "--max-state-age-ms",
            "1500",
            "--forecast-next-pass-lag=false",
            "--forecast-push-port",
            "26648",
            "--global-queue-room",
            "tokens",
        ])
        .unwrap();
        assert_eq!(given.cost_params.move_margin, 1.0);
        assert_eq!(given.cost_params.max_state_age_s, 1.5);
        assert_eq!(given.global_queue.unwrap().room, GlobalQueueRoom::Tokens);
        let state = &given.forecast.as_ref().unwrap().0;
        assert!(!state.next_pass_lag);
        assert_eq!(state.push.as_ref().map(|p| p.port), Some(26648));
        // The queue drain with the token-count wait.
        let drain = paper(&["--prefill-wait", "tokens"]).unwrap();
        assert_eq!(drain.cost_params.wait, PrefillWait::Tokens);
        assert_eq!(drain.global_queue.unwrap().room, GlobalQueueRoom::Forecast);
        assert!(drain.forecast.as_ref().unwrap().0.next_pass_lag);
        // Without the preset each piece is given.
        let explicit = cost(&[
            "--prefill-wait",
            "forecast",
            "--global-queue",
            "--global-queue-room",
            "forecast",
        ])
        .unwrap();
        assert_eq!(explicit.cost_params.wait, PrefillWait::Forecast);
        assert_eq!(explicit.cost_params.move_margin, 1.0);
        assert_eq!(
            explicit.global_queue.unwrap().room,
            GlobalQueueRoom::Forecast
        );
        assert!(!explicit.forecast.as_ref().unwrap().0.next_pass_lag);
        // The Kimi argv: a fit file on top of the preset.
        let fit = format!("{fixtures}/kimi-k26-fit.json");
        let kimi = paper(&["--forecast-fit", fit.as_str()]).unwrap();
        let state = &kimi.forecast.as_ref().unwrap().0;
        assert_eq!(state.frozen_fit.as_ref().unwrap().sha256.len(), 64);
        assert_ne!(kimi.forecast, preset.forecast, "the fit is in the state");
        // A file that is not a fit (the error table) is refused.
        let err = paper(&["--forecast-fit", table.as_str()])
            .unwrap_err()
            .to_string();
        assert!(err.contains("--forecast-fit"), "{err}");
        let refused = |add: &[&str], want: &str| {
            let err = with_pins(add).unwrap_err().to_string();
            assert!(err.contains(want), "{add:?}: {err}");
        };
        refused(
            &["--global-queue-room", "forecast"],
            "--global-queue-room requires --global-queue",
        );
        refused(
            &["--move-margin", "0"],
            "--move-margin needs --prefill-wait forecast",
        );
        refused(
            &["--max-state-age-ms", "5000"],
            "--max-state-age-ms needs --prefill-wait forecast",
        );
        let needs = "need --prefill-wait forecast or --global-queue-room forecast";
        refused(&["--forecast-error-table", &table], needs);
        refused(&["--forecast-next-pass-lag"], needs);
        refused(&["--forecast-fit", &fit], needs);
        let err = into_config_owned(with_model(&[
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://10.0.0.1:18080,18998,1,0",
            "--global-queue-room",
            "tokens",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("--global-queue-room requires --symmetric-pd-p-select cost-function"),
            "{err}"
        );
    }

    #[test]
    fn symmetric_pd_cost_function_fixed_lambda() {
        use crate::scheduler::placement_cost::CopyPriceMode;
        let cfg = cost_function_config(&["--copy-price"], &["--copy-price", "18.4"]).unwrap();
        let cost_params = cfg
            .model
            .symmetric_pd
            .unwrap()
            .cost_function
            .unwrap()
            .cost_params;
        assert_eq!(cost_params.copy_price_mode, CopyPriceMode::Fixed);
        assert_eq!(cost_params.copy_price, 18.4);
        assert!(!cost_params.is_auto());
    }

    /// `--symmetric-pd-unknown-expire-secs` sets the unknown expiry of the cost
    /// function (0 = never). It needs `--symmetric-pd-p-select cost-function` and,
    /// as every `--symmetric-pd-*` flag, `--policy symmetric_pd`.
    #[test]
    fn completions_body_cap_and_image_layout_flags() {
        let cfg = into_config(&["--model-id", "m", "--worker-urls", "http://h:1"]).unwrap();
        assert_eq!(
            cfg.proxy.max_completions_body_bytes,
            5 << 20,
            "default unchanged"
        );
        assert!(cfg.proxy.image_layout.is_none());
        let cfg = into_config(&[
            "--model-id",
            "m",
            "--worker-urls",
            "http://h:1",
            "--max-completions-body-mib",
            "256",
        ])
        .unwrap();
        assert_eq!(cfg.proxy.max_completions_body_bytes, 256 << 20);
        assert!(into_config(&[
            "--model-id",
            "m",
            "--worker-urls",
            "http://h:1",
            "--max-completions-body-mib",
            "0",
        ])
        .is_err());
        let golden: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/kimi_image_layout_golden.json"
        ))
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image_config.json");
        std::fs::write(&path, golden["config"].to_string()).unwrap();
        let p = path.to_str().unwrap();
        let cfg = into_config(&[
            "--model-id",
            "m",
            "--worker-urls",
            "http://h:1",
            "--image-layout-config",
            p,
        ])
        .unwrap();
        assert_eq!(cfg.proxy.image_layout.unwrap().image_token_id, 163_605);
        std::fs::write(&path, "{}").unwrap();
        assert!(into_config(&[
            "--model-id",
            "m",
            "--worker-urls",
            "http://h:1",
            "--image-layout-config",
            p,
        ])
        .is_err());
    }

    #[test]
    fn symmetric_pd_unknown_expire_secs() {
        for (value, want) in [("0", 0), ("1", 1), ("120", 120)] {
            let cfg =
                cost_function_config(&[], &["--symmetric-pd-unknown-expire-secs", value]).unwrap();
            let cost = cfg.model.symmetric_pd.unwrap().cost_function.unwrap();
            assert_eq!(cost.unknown_expire_secs, want, "{value}");
        }
        let err = into_config_owned(with_model(&[
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://10.0.0.1:18080,18998,1,0",
            "--symmetric-pd-unknown-expire-secs",
            "60",
        ]))
        .unwrap_err()
        .to_string();
        let want =
            "--symmetric-pd-unknown-expire-secs requires --symmetric-pd-p-select cost-function";
        assert!(err.contains(want), "{err}");
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--symmetric-pd-unknown-expire-secs",
            "60",
        ]))
        .unwrap_err()
        .to_string();
        let want = "--symmetric-pd-* flags require --policy symmetric_pd";
        assert!(err.contains(want), "{err}");
        let err = cost_function_config(&[], &["--symmetric-pd-unknown-expire-secs", "x"])
            .unwrap_err()
            .to_string();
        assert!(err.contains("--symmetric-pd-unknown-expire-secs"), "{err}");
    }

    /// `--symmetric-pd-pull-from-any-holder` is off by default and sets the cost
    /// function's wider pull source; it needs `--symmetric-pd-p-select
    /// cost-function` and, as every `--symmetric-pd-*` flag, `--policy symmetric_pd`.
    #[test]
    fn symmetric_pd_pull_from_any_holder() {
        for (extra, want) in [
            (&[][..], false),
            (&["--symmetric-pd-pull-from-any-holder"][..], true),
        ] {
            let cfg = cost_function_config(&[], extra).unwrap();
            let cost = cfg.model.symmetric_pd.unwrap().cost_function.unwrap();
            assert_eq!(cost.pull_from_any_holder, want, "{extra:?}");
        }
        let err = into_config_owned(with_model(&[
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://10.0.0.1:18080,18998,1,0",
            "--symmetric-pd-pull-from-any-holder",
        ]))
        .unwrap_err()
        .to_string();
        let want =
            "--symmetric-pd-pull-from-any-holder requires --symmetric-pd-p-select cost-function";
        assert!(err.contains(want), "{err}");
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--symmetric-pd-pull-from-any-holder",
        ]))
        .unwrap_err()
        .to_string();
        let want = "--symmetric-pd-* flags require --policy symmetric_pd";
        assert!(err.contains(want), "{err}");
    }

    /// The dynamic-role argv: the cost function under
    /// `--symmetric-pd-external-role-control` (roles from the external controller,
    /// 5 s default drain), with or without the router late-D choice (the
    /// full argv keeps `--pd-late-decode choose --pd-late-decode-rule
    /// fit_req` native; without it the D-leg follows `--symmetric-pd-d-select` at
    /// admission).
    #[test]
    fn symmetric_pd_cost_function_accepts_external_role_control() {
        for drop in [&["--pd-native-decode-state", "--pd-late-decode"][..], &[]] {
            let c = cost_function_config(drop, &["--symmetric-pd-external-role-control"]).unwrap();
            let spd = c.model.symmetric_pd.as_ref().unwrap();
            assert!(spd.external_role_control);
            assert!(spd.cost_function.is_some());
            assert_eq!(
                spd.role_drain_timeout,
                Some(std::time::Duration::from_secs(5))
            );
            assert_eq!(c.proxy.late_decode.choose(), drop.is_empty());
        }
    }

    /// global queue on the dynamic argv: `--global-queue` and
    /// `--symmetric-pd-trace-session-moves` are accepted with
    /// `--symmetric-pd-external-role-control`, with and without the router late-D
    /// choice (the queue steps over the current P-role pool).
    #[test]
    fn symmetric_pd_global_queue_accepts_external_role_control() {
        for drop in [&["--pd-native-decode-state", "--pd-late-decode"][..], &[]] {
            let c = cost_function_config(
                drop,
                &[
                    "--symmetric-pd-external-role-control",
                    "--global-queue",
                    "--symmetric-pd-trace-session-moves",
                ],
            )
            .unwrap();
            let spd = c.model.symmetric_pd.as_ref().unwrap();
            assert!(spd.external_role_control);
            let cost = spd.cost_function.as_ref().unwrap();
            assert_eq!(
                cost.global_queue,
                Some(crate::config::GlobalQueueConfig {
                    max_wait_ms: 300_000,
                    room: crate::global_queue::GlobalQueueRoom::Tokens,
                })
            );
            assert!(cost.trace_session_moves);
            assert_eq!(c.proxy.late_decode.choose(), drop.is_empty());
        }
    }

    /// `--global-queue` (on with the paper preset) and
    /// `--symmetric-pd-trace-session-moves` (default off): only with
    /// `--symmetric-pd-p-select cost-function`; the queue has the 300-s guard.
    #[test]
    fn symmetric_pd_global_queue_flags() {
        let cost = |cfg: Config| cfg.model.symmetric_pd.unwrap().cost_function.unwrap();
        let off = cost(cost_function_config(&[], &[]).unwrap());
        assert!(off.global_queue.is_none() && !off.trace_session_moves);
        let on = cost(
            cost_function_config(
                &[],
                &["--global-queue", "--symmetric-pd-trace-session-moves"],
            )
            .unwrap(),
        );
        assert_eq!(
            on.global_queue,
            Some(crate::config::GlobalQueueConfig {
                max_wait_ms: 300_000,
                room: crate::global_queue::GlobalQueueRoom::Tokens,
            })
        );
        assert!(on.trace_session_moves);
        for (flag, want) in [
            (
                "--global-queue",
                "--global-queue requires --symmetric-pd-p-select cost-function",
            ),
            (
                "--symmetric-pd-trace-session-moves",
                "--symmetric-pd-trace-session-moves requires --symmetric-pd-p-select cost-function",
            ),
        ] {
            let err = into_config_owned(with_model(&[
                "--policy",
                "symmetric_pd",
                "--symmetric-pd-worker",
                "http://10.0.0.1:18080,18998,1,0",
                flag,
            ]))
            .unwrap_err()
            .to_string();
            assert!(err.contains(want), "{err}");
        }
    }

    /// One row for each refusal of `--symmetric-pd-p-select cost-function` (new
    /// checks, and existing checks that now also apply to symmetric_pd). Each
    /// row removes or adds one item of the accepted argv.
    #[test]
    fn symmetric_pd_cost_function_refusals() {
        let map_without_w2 = r#"[
            {"id":11,"url":"http://10.0.0.1:18080","servicer":"10.0.0.1:8000"},
            {"id":12,"url":"http://10.0.0.2:18080","servicer":"10.0.0.2:8000"}]"#;
        // W1 is listed with a trailing slash and a blank servicer.
        let map_empty_servicer = r#"[
            {"id":11,"url":"http://10.0.0.1:18080","servicer":"10.0.0.1:8000"},
            {"id":12,"url":"http://10.0.0.2:18080/","servicer":" "},
            {"id":13,"url":"http://10.0.0.3:18080","servicer":"10.0.0.3:8000"}]"#;
        let replay = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
        let rows: &[(&str, &[&str], &[&str], &str)] = &[
            (
                "N1",
                &["--symmetric-pd-p-select"],
                &[],
                "require --policy symmetric_pd with --symmetric-pd-p-select cost-function",
            ),
            (
                "N5",
                &["--prefill-poll-ms"],
                &["--prefill-poll-ms", "0"],
                "--prefill-poll-ms must be greater than 0",
            ),
            (
                "N6",
                &["--symmetric-pd-force-split"],
                &[],
                "--symmetric-pd-p-select cost-function requires --symmetric-pd-force-split",
            ),
            (
                "N7",
                &["--symmetric-pd-pull-hint"],
                &[],
                "--symmetric-pd-p-select cost-function requires --symmetric-pd-pull-hint: the cost-function pull \
                 rides on the P-leg as the pull hint",
            ),
            (
                "N8 queue wait",
                &[],
                &["--symmetric-pd-max-queue-wait-ms", "100"],
                "--symmetric-pd-p-select cost-function never holds a request: \
                 --symmetric-pd-max-queue-wait-ms and --symmetric-pd-max-uncached-tokens do not apply",
            ),
            (
                "N8 uncached cap",
                &[],
                &["--symmetric-pd-max-uncached-tokens", "1"],
                "--symmetric-pd-p-select cost-function never holds a request",
            ),
            (
                "N9",
                &[],
                &["--symmetric-pd-decision-replay", replay],
                "--symmetric-pd-p-select cost-function conflicts with --symmetric-pd-decision-replay",
            ),
            (
                "N11",
                &[],
                &["--symmetric-pd-cache-threshold", "0.5"],
                "--symmetric-pd-cache-threshold and --symmetric-pd-balance-* set only the cache-first D \
                 selection when --symmetric-pd-p-select is cost-function",
            ),
            (
                "N12",
                &["--kv-indexer-url"],
                &[],
                "--symmetric-pd-p-select cost-function requires --kv-indexer-url and --kv-worker-map",
            ),
            (
                "N13",
                &[
                    "--symmetric-pd-worker http://10.0.0.1:",
                    "--symmetric-pd-worker http://10.0.0.2:",
                    "--symmetric-pd-worker http://10.0.0.3:",
                ],
                &[],
                "--symmetric-pd-p-select cost-function needs the P-role workers (prefill weight > 0) \
                 in --symmetric-pd-worker or --symmetric-pd-workers-file",
            ),
            (
                "N14 not mapped",
                &["--kv-worker-map"],
                &["--kv-worker-map", map_without_w2],
                "--symmetric-pd-p-select cost-function: P-role worker http://10.0.0.3:18080 needs a \
                 --kv-worker-map entry with a servicer",
            ),
            (
                "N14 no servicer",
                &["--kv-worker-map"],
                &["--kv-worker-map", map_empty_servicer],
                "--symmetric-pd-p-select cost-function: P-role worker http://10.0.0.2:18080 needs a \
                 --kv-worker-map entry with a servicer",
            ),
            (
                "E8",
                &["--kv-pull-hint-min-incremental-tokens"],
                &["--kv-pull-hint-min-incremental-tokens", "256"],
                "the cost function never pulls fewer than 512 tokens",
            ),
            (
                "E9",
                &[],
                &["--kv-pull-hint-local-prefix-fallback"],
                "it cannot be combined with --kv-pull-hint-local-prefix-fallback",
            ),
            (
                "E10",
                &[],
                &["--kv-session-retention"],
                "--kv-session-retention is not available with --policy symmetric_pd; use \
                 --symmetric-pd-kv-session-retention",
            ),
            (
                "N15",
                &["--symmetric-pd-p-select"],
                &["--symmetric-pd-unknown-expire-secs", "60"],
                "--symmetric-pd-unknown-expire-secs requires --symmetric-pd-p-select cost-function",
            ),
        ];
        for (id, drop, add, want) in rows {
            let err = cost_function_config(drop, add).unwrap_err().to_string();
            assert!(err.contains(want), "{id}: got {err:?}, want {want:?}");
        }
    }

    /// The cost-function flags are refused under the other policies.
    #[test]
    fn cost_flags_need_the_cost_function() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "round_robin",
            "--prefill-poll-ms",
            "100",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("require --policy symmetric_pd"), "{err}");
    }

    /// The cost function never pulls below 512 tokens: an absent gate becomes
    /// 512, a smaller explicit gate and the local-prefix fallback are rejected.
    #[test]
    fn cost_function_pull_floor_is_at_least_512() {
        let gate = |add: &[&str]| {
            cost_function_config(&["--kv-pull-hint-min-incremental-tokens"], add)
                .map(|c| c.model.kv_hint.unwrap().min_incremental_tokens)
        };
        assert_eq!(gate(&[]).unwrap(), 512);
        assert_eq!(
            gate(&["--kv-pull-hint-min-incremental-tokens", "2048"]).unwrap(),
            2048
        );
        for low in ["0", "1", "511"] {
            let err = gate(&["--kv-pull-hint-min-incremental-tokens", low])
                .unwrap_err()
                .to_string();
            assert!(err.contains("at least 512"), "{low}: {err}");
        }
        let err = gate(&["--kv-pull-hint-local-prefix-fallback"])
            .unwrap_err()
            .to_string();
        assert!(err.contains("local-prefix-fallback"), "{err}");
    }
}
