// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Command-line interface. The router is configured entirely through
//! flags — there is no config file. [`Cli::into_config`] resolves the
//! flags into a validated [`Config`].

use anyhow::{anyhow, Result};
use clap::Parser;
use std::num::NonZeroU32;

use crate::config::{
    default_cb_cool_down, default_symmetric_pd_load_poll_ms, default_symmetric_pd_max_queue_wait_ms,
    default_symmetric_pd_max_uncached_tokens, default_proxy_request_timeout_secs,
    default_stale_request_timeout_secs, resolve_mode, ActiveLoadConfig, CacheAwareConfig,
    CircuitBreakerConfig, CoScheduleConfig, Config, DiscoveryBackend, SymmetricPdConfig,
    SymmetricPdCostFunctionConfig, SymmetricPdDSelect, SymmetricPdImportCachePolicy, SymmetricPdPSelect,
    SymmetricPdPreset,
    SymmetricPdWorkerSpec, GatewayCacheAwareConfig, K8sDiscoveryConfig, KvIndexerEndpointConfig,
    KvPullScope, LogFormat, ModelConfig, ObservabilityConfig, PdDecisionReplayConfig,
    PdDecodePolicy, PdDecodeTokensConfig, PolicyKind, ProxyConfig, ServerConfig,
    StaticUrlsDiscoveryConfig, StickyConfig, SYMMETRIC_PD_MAX_QUEUE_WAIT_MS_CAP,
};

const DEFAULT_KV_INDEXER_QUERY_TIMEOUT_MS: u64 = 100;
/// Decode->prefill drain bound when external role control sets no explicit value.
const DEFAULT_ROLE_DRAIN_TIMEOUT_SECS: u64 = 5;
const DEFAULT_KV_INDEXER_QUERY_MAX_INFLIGHT: usize = sgl_kv_indexer::DEFAULT_QUERY_MAX_INFLIGHT;

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
    #[arg(long)]
    pub gateway_ca_cache_threshold: Option<f32>,
    /// sgl-model-gateway `--balance-abs-threshold` (default 64): imbalanced
    /// when `max - min` in-flight prefill requests is greater than this (and
    /// the relative test holds).
    #[arg(long)]
    pub gateway_ca_balance_abs_threshold: Option<usize>,
    /// sgl-model-gateway `--balance-rel-threshold` (default 1.5, at least
    /// 1): imbalanced when `max > min * this` (and the absolute test holds).
    #[arg(long)]
    pub gateway_ca_balance_rel_threshold: Option<f32>,
    /// sgl-model-gateway `--eviction-interval` in seconds (default 120,
    /// positive): period of the tree size eviction.
    #[arg(long)]
    pub gateway_ca_eviction_interval_secs: Option<u64>,
    /// sgl-model-gateway `--max-tree-size` (default 67108864, positive): tree
    /// characters kept per prefill worker by each eviction (8 per token).
    #[arg(long)]
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

    // ---- co-schedule policy (`--policy co_schedule`; the TRADE flags, the
    // uncached balance, the cost trace, the load poll and the pull scope
    // also with `--policy symmetric_pd --symmetric-pd-p-select cost-function`) ----
    /// Immediately balance remaining uncached tokens using engine RID progress.
    /// Requires --co-schedule-cost-trace and engines with --enable-prefill-work-snapshot.
    #[arg(long)]
    pub co_schedule_uncached_balance: bool,
    /// Keep the cache-aware owner when its existing work is at most this many
    /// uncached tokens. Default 12288. This never delays request dispatch.
    #[arg(long)]
    pub co_schedule_owner_threshold_tokens: Option<u64>,
    /// Extra uncached-token work a detour must save after restoration cost.
    /// Requires uncached balance and a restore cost divisor; default 0.
    #[arg(long)]
    pub co_schedule_detour_margin_tokens: Option<u64>,
    /// With uncached balance, choose the lowest post-dispatch work using
    /// cost = D * remaining uncached tokens + tokens needing GPU restoration.
    /// D must be positive. GPU-unknown prefixes count as needing restoration.
    #[arg(long)]
    pub co_schedule_restore_cost_divisor: Option<u64>,
    /// TRADE P selection on the uncached ledger: choose the P with the least
    /// backlog + own uncached tokens after the hinted pull + R x pull time
    /// not hidden behind the P's wait + lambda x pulled tokens. No pull cap;
    /// a pull below --kv-pull-hint-min-incremental-tokens is not sent (the
    /// target recomputes); that floor is at least 512 (absent = 512, a
    /// smaller value or --kv-pull-hint-local-prefix-fallback is rejected).
    /// Ties go to the cache-aware owner. Requires
    /// --co-schedule-uncached-balance; excludes the owner threshold, the
    /// restore divisor and the detour margin. Off by default.
    #[arg(long)]
    pub co_schedule_trade: bool,
    /// TRADE: R, uncached prefill tokens one P computes per second. Converts
    /// pull time into tokens. Default 13500. `auto`: R = R_hat, the computed
    /// prefill tokens per busy second of all P, measured by the router at
    /// each decision (see --co-schedule-trade-auto-tau-secs); 13500 during
    /// the first tau seconds and without a measurement.
    #[arg(long, requires = "co_schedule_trade", value_parser = parse_number_or_auto)]
    pub co_schedule_trade_rate_tokens_per_s: Option<NumberOrAuto>,
    /// TRADE: fixed time of one KV pull in milliseconds. Default 450.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_pull_fixed_ms: Option<f64>,
    /// TRADE: pull time in milliseconds per 100k pulled tokens. Default 100.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_pull_ms_per_100k: Option<f64>,
    /// TRADE: copy price lambda, in uncached-token equivalents per pulled
    /// token (the second copy a pull leaves in the target host pool).
    /// Default 4.6, the value of the paper's runs. 0 = transfer time only.
    /// `auto`: at each decision lambda = clamp(mu * q * R / R_hat, min, max)
    /// from the router's own measurements (see the
    /// --co-schedule-trade-copy-lambda-* and --co-schedule-trade-auto-* flags).
    #[arg(long, requires = "co_schedule_trade", value_parser = parse_number_or_auto)]
    pub co_schedule_trade_copy_lambda: Option<NumberOrAuto>,
    /// TRADE: count all work routed to a P in its backlog, also the requests
    /// whose first prefill chunk has not launched. This is the default; the
    /// flag is accepted for older launch scripts.
    #[arg(
        long,
        requires = "co_schedule_trade",
        conflicts_with = "co_schedule_trade_exclude_unlaunched"
    )]
    pub co_schedule_trade_count_inflight: bool,
    /// TRADE (diagnosis only): leave out of the backlog the requests whose
    /// first prefill chunk has not launched. The engine reports a request
    /// only after that launch, so this drops the requests in transit, in the
    /// bootstrap queue AND in the engine waiting queue. Off by default.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_exclude_unlaunched: bool,
    /// TRADE: source of the P wait B and the request's own service S.
    /// `ledger` (default): B = backlog / R and S = u / R from the router
    /// ledger (unchanged selection and trace). `eft`: B_w = the EFT-0+ drain
    /// forecast of the P's engine prefill_routing snapshot (P needs
    /// SGLANG_PREFILL_ROUTING_SNAPSHOT=1) plus the routed requests that the
    /// snapshot does not show yet, each at its step-model work; S_w = the
    /// step-model service of the request. cost = R*B_w + R*S_w +
    /// R*max(0, pull_s - B_w) + lambda*pulled. For a P whose snapshot is
    /// missing, unusable or older than --co-schedule-trade-eft-max-view-age-ms,
    /// B_w is the step-model work of the router's active requests on that P
    /// alone (the ledger on the same scale; counted in the trace). Requires
    /// --co-schedule-trade-eft-model. `olp` (--symmetric-pd-p-select cost-function
    /// only): the online latency predictor: B_w and S_w in seconds from the
    /// structural work of the P's live ledger rows and of the request, scaled
    /// online (completion labels by default; see
    /// --co-schedule-trade-olp-params). A per-decision confidence gate keeps
    /// the ledger decision while the predictor is cold, a P's progress view is
    /// stale or it orders recent label pairs worse than the ledger. `e3`: the E3 P completion forecast of
    /// the P's snapshot, the router's own unseen rows and online fits from
    /// the engine timeline samples: B_w = wait_s, S_w = pred_s - wait_s, so
    /// cost = R*(pred_s + max(0, pull_s - wait_s)) + lambda*pulled; the
    /// request leaves the cache-aware owner only through the prediction
    /// gate (--co-schedule-trade-gate-m, --co-schedule-trade-gate-gmin-ms).
    /// A P without a usable forecast keeps the ledger terms and never wins
    /// a move. Requires --co-schedule-trade-e3-bound-table.
    #[arg(long, requires = "co_schedule_trade", value_parser = ["ledger", "eft", "olp", "e3"])]
    pub co_schedule_trade_wait: Option<String>,
    /// TRADE wait eft: the frozen step-time model JSON (the
    /// fit_prefill_completion_model layout; its sha256 goes into the trace).
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_eft_model: Option<std::path::PathBuf>,
    /// TRADE wait eft or e3: a snapshot whose view age (poll sent to
    /// decision plus the engine's own snapshot age) is above this many
    /// milliseconds is not used. Default 1500.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_eft_max_view_age_ms: Option<f64>,
    /// TRADE wait olp (and the shadow readout): a JSON object whose keys
    /// override the online latency predictor's parameters by name (the
    /// `OlpParams` fields: c0..c3, rr, chunk, scale_source, the busy-scale,
    /// label-scale, gate and park-gate keys); an unknown key or an invalid
    /// value stops the start. Its sha256 goes into the startup log and the
    /// cost trace. Needs --co-schedule-trade-wait olp or
    /// --co-schedule-trade-olp-shadow.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_olp_params: Option<std::path::PathBuf>,
    /// TRADE with --symmetric-pd-p-select cost-function and
    /// --co-schedule-trade-wait ledger: feed the online latency predictor and
    /// write its predictions to the cost trace (`olp_*` candidate keys,
    /// `trade_decision.olp`, `olp_step` rows) without using them: every
    /// decision stays the cost function's. Off by default.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_olp_shadow: bool,
    /// TRADE wait olp (and the shadow readout): price each P's queue from
    /// exact integer sums of its live rows' work terms, kept at every row
    /// change (reservation, progress reply, end), instead of walking the
    /// rows at each decision: O(1) per candidate. The rows, launched tokens
    /// and choices are those of the walk; q_model differs only by
    /// floating-point rounding. Off by default.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_olp_incremental: bool,
    /// TRADE wait olp (and the shadow readout): fewer OLP keys in the cost
    /// trace: each decision's `olp` object without the step statistics
    /// (they stay in the `olp_step` rows and the snapshot) and five
    /// `olp_*` keys per candidate (wait, service, q_model, own_model, rows)
    /// instead of eleven. Off by default.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_olp_trace_compact: bool,
    /// --symmetric-pd-p-select cost-function: write the cost trace from a
    /// background thread: each row's content and order are unchanged, but
    /// the write and flush leave the cost-book lock (rows are flushed per
    /// batch of waiting rows). A write error stops admission at the next
    /// event instead of at the failing one. Off by default (each row is
    /// written and flushed under the lock).
    #[arg(long, requires = "co_schedule_cost_trace")]
    pub co_schedule_cost_trace_async: bool,
    /// TRADE wait e3: the error-bound file. Either bound-params.json
    /// (schema router-e3-bound-params.v2: per deploy mode a table and its
    /// level) or bound-tables.json (mode -> table; then
    /// --co-schedule-trade-e3-bound-level is required). Its sha256 goes into
    /// the startup log and the trace. Required with wait e3.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_e3_bound_table: Option<std::path::PathBuf>,
    /// TRADE wait e3: the deploy mode of the bound table of poll forecasts:
    /// snap+prod (default: snapshot poll, router calibration, the production
    /// rules) or snap+prodL (the lag-corrected calibration). Other modes
    /// (push, gpu, split, fresh, r3) are refused.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_e3_bound_mode: Option<String>,
    /// TRADE wait e3: the bound level (nominal coverage of |actual - pred|).
    /// Default: the level that bound-params.json ships for the mode.
    #[arg(long, requires = "co_schedule_trade", value_parser = ["0.8", "0.9"])]
    pub co_schedule_trade_e3_bound_level: Option<String>,
    /// TRADE wait e3: m of the prediction gate. The request leaves the
    /// cache-aware owner o for the cheapest P w only if
    /// (C_o - C_w)/R - m*sqrt(b_o^2 + b_w^2) > g_min, with b the error bound
    /// of each forecast. Default 1 (the bounds as fitted).
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_gate_m: Option<f64>,
    /// TRADE wait e3: g_min of the prediction gate, in milliseconds.
    /// Default 0.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_gate_gmin_ms: Option<f64>,
    /// TRADE wait e3: margin m of the in-transit rule (rq), in
    /// milliseconds. A bootstrap row or a routed row that the snapshot does
    /// not show is in the forecast queue only when its ready time is at most
    /// the request's own ready time minus m. Default 0.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_e3_rq_margin_ms: Option<f64>,
    /// TRADE wait e3: write the shadow log (one e3_shadow line per routed
    /// request with its rid, chosen P, pred_s and decision time; one e3_label
    /// line when its completion is observed) to this new file instead of the
    /// cost trace. An existing file is refused.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_e3_shadow: Option<std::path::PathBuf>,
    /// TRADE wait e3: subscribe to the state push of each P engine
    /// (DecoupledEngine prefill state push; the engine needs
    /// SGLANG_PREFILL_STATE_PUSH_PORT=PORT and
    /// SGLANG_PREFILL_ROUTING_SNAPSHOT=1). One ZMQ SUB per P at
    /// tcp://<P host>:PORT (routing rank 0). A P's forecast starts from its
    /// newest pushed state when that state is at most
    /// --co-schedule-trade-eft-max-view-age-ms old (state time = the
    /// snapshot's sampled_monotonic_s + the minimum receive - send offset of
    /// the last 30 s), has the epoch of the P's polled snapshot and is
    /// usable; otherwise from the polled snapshot. P workers that share a
    /// host keep the poll. Default off.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_e3_push_port: Option<u16>,
    /// TRADE wait e3 with the state push: the deploy mode of the bound table
    /// for forecasts that start from a pushed state (same file and level
    /// rule as --co-schedule-trade-e3-bound-mode): snap, push, pushL or
    /// pushslow with +prod or +prodL (gpu and split modes are refused).
    /// Default: the mode of --co-schedule-trade-e3-bound-mode.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_e3_push_bound_mode: Option<String>,
    /// TRADE wait e3 or the predicted GWQ room: the forecast of a request
    /// whose last pass leaves no known work on a busy P prices the next pass
    /// that the expected arrivals form (the engine processes a pass's
    /// results after it launches the next pass). Default off: the E3
    /// reference (the bare result lag). Offline on Kimi-K2.6 TP4 at
    /// concurrency 128: APE 17.8 % -> 13.5 %, mean signed error -3.7 s ->
    /// -0.6 s.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_e3_next_pass_lag: bool,
    /// TRADE wait e3 or the predicted GWQ room: FILE holds offline warmup
    /// constants (the e3_constants of an offline step model: the model JSON
    /// or that object) that replace the online GPU model, launch model and
    /// calibration constants of every P; the online samples still give the
    /// arrival rates. For a P that is never idle: the online GPU fit takes
    /// only single passes launched after an idle gap (20 per class), so a
    /// saturated P keeps no fit and its forecast falls back (no_fit). The
    /// file's sha256 goes into the startup log. Default off (the online
    /// fits).
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_e3_frozen_fit: Option<std::path::PathBuf>,
    /// TRADE: what the copy price lambda charges. `pulled` (default): the
    /// pulled tokens. `all`: also the tokens a P recomputes although another
    /// P already caches them (a move without a pull also makes a copy). When
    /// any of the later TRADE options is on, the trace records those tokens
    /// (dup_tokens) with either value.
    #[arg(long, requires = "co_schedule_trade", value_parser = ["pulled", "all"])]
    pub co_schedule_trade_dup_charge: Option<String>,
    /// TRADE: the price of the copy that a pull leaves on its target P.
    /// `lambda` (default): lambda x pulled tokens, the same lambda for every
    /// candidate. `phi`: lambda_P x pulled tokens with lambda_P = q x phi_P
    /// for the target P. q: the copy price auto before the clamp (mu x q x
    /// R / R_hat with mu 1 and R = R_hat; the fallback lambda while the auto
    /// price falls back, for example during the first tau seconds). phi_P:
    /// the live share (class 1) of the host tokens that P's KV session
    /// retention evicted, an EWMA with time constant tau
    /// (--co-schedule-trade-auto-tau-secs) of the increments of the engine
    /// host_room counters; 0 when P evicted no host token in the last tau
    /// seconds. A P without host_room data (no section yet, or none in the
    /// last tau seconds) uses phi_P = 1 (trace reason no_host_room). Floor:
    /// lambda_P = max(min, q x phi_P) with min =
    /// --co-schedule-trade-copy-lambda-min (default 4.6); the max does not
    /// apply. Only the pulled tokens are priced (a pull discount, not a
    /// price of every landing). Needs --symmetric-pd-p-select cost-function,
    /// --symmetric-pd-loads-host-room, --co-schedule-trade-copy-lambda auto,
    /// --co-schedule-trade-rate-tokens-per-s auto and mu 1; not with
    /// --co-schedule-trade-dup-charge all.
    #[arg(long, requires = "co_schedule_trade", value_parser = ["lambda", "phi"])]
    pub co_schedule_trade_pull_price: Option<String>,
    /// TRADE: noise margin k (default 0 = off). The request leaves the
    /// cache-aware owner (highest local hit) only if
    /// (cost(owner) - cost(best)) / R > k * sqrt(B_o^2 + B_best^2) seconds,
    /// with B the wait terms of the owner and of the best P.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_margin_noise: Option<f64>,
    /// TRADE: weight mu of the head-of-line (HOL) term (default 0 = off).
    /// The P queue is FCFS, so a request placed on P makes the requests that
    /// arrive at P later wait longer. The cost of P adds
    /// mu x R x a_w x S x (B + S/2) tokens: S = the request's own service on
    /// P (u/R, or S_w with wait eft), B = P's wait term (backlog/R, or B_w;
    /// no pull time), a_w = TRADE placements per second on P (see
    /// --co-schedule-trade-hol-rate-window-secs). The term is a light-load
    /// estimate of those extra waits (mu = 1 gives their sum only at light
    /// load), so mu is an empirical weight to tune.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_hol_weight: Option<f64>,
    /// TRADE: window W in seconds of a_w, the arrival rate of the HOL term:
    /// TRADE placements on P in the last W seconds divided by
    /// min(W, max(1 s, time since the first TRADE placement)). Default 60.
    /// No effect while --co-schedule-trade-hol-weight is 0.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_hol_rate_window_secs: Option<f64>,
    /// TRADE copy price auto: the queue multiplier q (E2E seconds per second
    /// of extra P work). `sat` (default): N_active / n_P x busy, sessions
    /// per P times the P busy fraction (the closed-loop upper bound).
    /// `inflight`: 1 + the mean number of requests dispatched to a P whose
    /// prefill has not ended (lower). Needs --co-schedule-trade-copy-lambda auto.
    #[arg(long, requires = "co_schedule_trade", value_parser = ["sat", "inflight"])]
    pub co_schedule_trade_copy_lambda_q: Option<String>,
    /// TRADE copy price auto: mu, recomputed tokens per pulled token. Default
    /// 1 (the upper bound). Needs --co-schedule-trade-copy-lambda auto.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_copy_lambda_mu: Option<f64>,
    /// TRADE copy price auto: the floor of lambda. Default 4.6 (safety only).
    /// Needs --co-schedule-trade-copy-lambda auto.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_copy_lambda_min: Option<f64>,
    /// TRADE copy price auto: the cap of lambda. Default 1000. Needs
    /// --co-schedule-trade-copy-lambda auto.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_copy_lambda_max: Option<f64>,
    /// TRADE copy price auto: the fixed lambda for the first tau seconds and
    /// when an input is missing (no session, no P history, no R_hat).
    /// Default 4.6 (the fixed default). Needs --co-schedule-trade-copy-lambda auto.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_copy_lambda_fallback: Option<f64>,
    /// TRADE auto (copy price or rate): the time constant tau in seconds of
    /// the router's measurements (P busy fraction, in-flight requests,
    /// computed prefill tokens per second); also the start-up period of the
    /// fallback. Default 120.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_auto_tau_secs: Option<f64>,
    /// TRADE auto (copy price or rate): a session (identity of
    /// x-kv-session-context, else x-agent-context) counts in N_active
    /// while it has a request on a P and for this many seconds after its last
    /// router event; POST /kv_sessions/end removes it at once. Default 300.
    #[arg(long, requires = "co_schedule_trade")]
    pub co_schedule_trade_auto_session_idle_secs: Option<f64>,
    /// TRADE global waiting queue (GWQ) in the router. `off` (default): TRADE
    /// dispatches every request at arrival. `sjf` / `hrrn`: a request whose
    /// TRADE choice P already has the depth of queued work (see
    /// --co-schedule-global-queue-depth-tokens) waits in one router queue.
    /// When a P has room, the router releases held requests, each to its
    /// TRADE argmin P at release (late P binding): `sjf` = the smallest
    /// predicted service u/R first (starvation guard
    /// --co-schedule-global-queue-max-wait-ms), `hrrn` = the highest
    /// (wait + service) / service first. Every P engine stays FCFS. A P
    /// without a fresh progress view counts as having room (fail-open).
    /// Needs --co-schedule-trade; not with --co-schedule-trade-wait eft.
    /// With --co-schedule-trade-wait e3 the walk and its releases use the
    /// ledger terms (and the E3 drain with
    /// --co-schedule-global-queue-room predicted); requests that the queue
    /// does not hold (--co-schedule-global-queue-hold free) keep e3.
    #[arg(long, value_parser = ["off", "sjf", "hrrn"])]
    pub co_schedule_global_queue: Option<String>,
    /// GWQ: the queued work per P that the router keeps, in tokens (at least
    /// 1), or `auto` (default) = 8192 (one chunk) + ceil(R x
    /// (--co-schedule-load-poll-ms + 500 ms dispatch delay)), with R =
    /// --co-schedule-trade-rate-tokens-per-s (its fixed value or fallback,
    /// 13500): 16292 with a 100 ms poll. Queued work = the work of the P
    /// beyond its running pass: the uncached tokens routed to it whose first
    /// prefill chunk is not yet in a progress snapshot (transit, bootstrap
    /// queue, engine waiting queue) + the remaining work of its launched
    /// requests minus one pass of 8192 (look-ahead batch, rest of a chunked
    /// request).
    #[arg(long, requires = "co_schedule_global_queue", value_parser = parse_tokens_or_auto)]
    pub co_schedule_global_queue_depth_tokens: Option<TokensOrAuto>,
    /// GWQ: a request held this many milliseconds goes at once to its TRADE
    /// argmin P, also without room. With sjf it is the starvation guard
    /// (default 30000); with hrrn only a safety net (default 300000; the
    /// order bounds the wait by itself).
    #[arg(long, requires = "co_schedule_global_queue")]
    pub co_schedule_global_queue_max_wait_ms: Option<u64>,
    /// GWQ: which arrivals may wait in the router queue. `all` (default):
    /// every arrival. `free` (predicted-drain fill): only a request that is
    /// free to go to any P; a request bound to its cache owner (the owner's
    /// cached prefix exceeds every other P's by at least one engine chunk,
    /// 8192 tokens) takes the TRADE decision at arrival and is dispatched at
    /// once, so it waits in the engine queue (prefix pinned, host prefetch).
    #[arg(long, requires = "co_schedule_global_queue", value_parser = ["all", "free"])]
    pub co_schedule_global_queue_hold: Option<String>,
    /// GWQ: the room rule. `tokens` (default): queued tokens on the router
    /// ledger below the depth. `predicted`: room also while the P's
    /// predicted drain time (E3 forecast of a cold one-chunk probe on the
    /// newest engine state) is at most the lead = depth / R (the forecast
    /// may only release earlier than the token rule), and the walk prices
    /// the P's wait as min(R x drain, ledger backlog); a P without a usable
    /// forecast keeps the token rule. Needs the
    /// E3 forecast state: --co-schedule-trade-e3-bound-table (and best
    /// --co-schedule-trade-e3-push-port), with --co-schedule-trade-wait
    /// ledger (the drain only) or e3.
    #[arg(long, requires = "co_schedule_global_queue", value_parser = ["tokens", "predicted"])]
    pub co_schedule_global_queue_room: Option<String>,
    /// Prefer GPU-resident prefixes when total cache hits tie. Capped mode only.
    #[arg(long)]
    pub co_schedule_prefer_gpu_hit: bool,
    /// Keep cache locality unless a detour saves more pending uncached tokens
    /// than this margin. Bypasses the cap and uses P-completion reservations.
    /// Requires --co-schedule-cost-trace; exclusive with --co-schedule-cost-balance.
    #[arg(long)]
    pub co_schedule_locality_margin_tokens: Option<u64>,
    /// Balance input length times expected uncached tokens across P nodes.
    /// Disables the legacy cap and holds each cost until its P response ends.
    #[arg(long)]
    pub co_schedule_cost_balance: bool,
    /// New JSONL file for request costs, lifecycle events and 1s snapshots.
    #[arg(long)]
    pub co_schedule_cost_trace: Option<std::path::PathBuf>,
    /// Dispatch to a worker only while its pending uncached-prefill tokens
    /// (engine `/v1/loads` baseline + router in-flight ledger) stay below
    /// this. Defaults to 32768 (128 pages of 256).
    #[arg(long)]
    pub co_schedule_max_uncached_tokens: Option<u64>,
    /// Force-dispatch a request queued longer than this many milliseconds
    /// to the min-backlog worker (starvation guard). Defaults to 30000.
    #[arg(long)]
    pub co_schedule_max_queue_wait_ms: Option<u64>,
    /// Interval in milliseconds for polling each mapped worker's
    /// `/v1/loads`. Defaults to 1000; 100 with
    /// `--co-schedule-uncached-balance` and with
    /// `--symmetric-pd-p-select cost-function` (the per-RID progress poll of each
    /// P-role worker).
    #[arg(long)]
    pub co_schedule_load_poll_ms: Option<u64>,
    /// Refresh stale progress at request arrival; share one query per worker.
    /// Requires uncached balance. load-poll-ms becomes the age/retry threshold.
    #[arg(long)]
    pub co_schedule_on_demand_load_refresh: bool,
    /// Fallback period in milliseconds for the dispatcher to re-check the
    /// waiting queue when no event (arrival, ledger retire, poll) woke it;
    /// the sleep is also cut short at the oldest queued request's
    /// starvation deadline. Must be > 0. Defaults to 500.
    #[arg(long)]
    pub co_schedule_tick_ms: Option<u64>,
    /// While the queue is holding (every worker over the cap), re-poll
    /// `/v1/loads` on the whole pool this many milliseconds after the
    /// previous poll round instead of waiting `--co-schedule-load-poll-ms`.
    /// 0 disables (ledger-only between periodic polls). Defaults to 100.
    #[arg(long)]
    pub co_schedule_poll_on_hold_ms: Option<u64>,
    /// When every pool worker is over the cap but the backlog spread
    /// (max − min, in tokens) exceeds this, dispatch the queue head to the
    /// min-backlog worker anyway (branch `imbalance`) instead of holding.
    /// 0 disables the escape. Defaults to 131072.
    #[arg(long)]
    pub co_schedule_imbalance_spread_tokens: Option<u64>,
    /// When a co-schedule dispatch may carry a `kv_pull_hint`: `any`
    /// (default — whenever the incremental gate passes, regardless of which
    /// worker was routed) or `detour-only` (only when routed away from the
    /// max-hit worker; the literal three-step rule, for ablation).
    #[arg(long, value_enum)]
    pub kv_pull_scope: Option<KvPullScope>,
    /// Session KV retention on the PD path of `--policy co_schedule` or
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
    #[arg(long)]
    pub symmetric_pd_w_hit: Option<f64>,
    /// Score weight of `backlog(p)` (default 0.3).
    #[arg(long)]
    pub symmetric_pd_w_backlog: Option<f64>,
    /// Score weight of `role_pen(p) + role_pen(d)` (default 0.3).
    #[arg(long)]
    pub symmetric_pd_w_role: Option<f64>,
    /// Score weight of `xfer_cost(p, d)` (default 0.2).
    #[arg(long)]
    pub symmetric_pd_w_xfer: Option<f64>,
    /// Score weight of `decode_load(d)` (default 0.1; 0 recovers the plain
    /// additive formula).
    #[arg(long)]
    pub symmetric_pd_w_dload: Option<f64>,
    /// P-leg selection: `cache-first` (default; replicates
    /// `cache_aware_zmq`: imbalance fast path, best hit above the cache
    /// threshold, else least-loaded) or `score` (argmax of the additive
    /// score, which under saturation degenerates to least-backlog),
    /// `co-schedule`, `gateway-cache-aware` (the sgl-model-gateway
    /// cache-aware prefill rule with the `--gateway-ca-*` flags and
    /// defaults; never holds requests), or `cost-function` (the TRADE cost
    /// model of `--co-schedule-trade*`, lambda fixed or auto, over the
    /// P-role workers: it picks the P-leg and the pull of the P-leg's
    /// `kv_pull_hint`; never holds requests). `cost-function` needs
    /// `--symmetric-pd-force-split`, `--symmetric-pd-pull-hint`, `--co-schedule-trade`,
    /// `--co-schedule-uncached-balance`, a new `--co-schedule-cost-trace`
    /// file, `--kv-indexer-url` and a `--kv-worker-map` entry with a
    /// servicer for each P-role worker.
    #[arg(long, value_enum)]
    pub symmetric_pd_p_select: Option<SymmetricPdPSelect>,
    /// `--policy symmetric_pd` only: what fills the flags left unset.
    /// `paper` (default) is the configuration of the paper's runs: force
    /// split, round-robin D selection, 100 ms load polls, late decode choice
    /// by `fit_req` over every decode on observed capacity, KV session
    /// retention, and, with `--kv-indexer-url` and a `--kv-worker-map` that
    /// names servicers, the cost-function P selection (TRADE, uncached
    /// balance) with peer pulls from any holder. `none`: the flags as given.
    #[arg(long, value_enum, default_value = "paper")]
    pub symmetric_pd_preset: SymmetricPdPreset,
    /// `--symmetric-pd-p-select cost-function` only: an unknown P-leg reservation
    /// (a transport error such as `upstream timed out`, or a drop after
    /// dispatch) leaves the cost ledger when its P answers a progress poll
    /// sent at least this many seconds after the later of the error and the
    /// last listing of its RID, and the answer does not list the RID. A P
    /// that does not answer keeps the work. A P aborts a prefill whose router
    /// connection closed, and lists only RIDs whose first chunk launched, so
    /// a still-queued request and a dropped one look alike; this bounds that
    /// error. Default 60; 0 = never expire (the work is held until a terminal
    /// result).
    #[arg(long)]
    pub symmetric_pd_unknown_expire_secs: Option<u64>,
    /// `--symmetric-pd-p-select cost-function` only: the per-RID progress poll of
    /// each P-role worker also asks for the engine's `host_room` section
    /// (`GET /v1/loads?include=core,queues,prefill_work,host_room`: the
    /// HiCache host pool by retention class, from engines with session
    /// retention and the section). The cost trace records the latest section
    /// of each P with its age (`host_room_by_p` in the one-second snapshot
    /// and in every decision). No decision uses it. An engine that rejects
    /// the section is polled without it (re-probed every 30 s). Off by
    /// default: the poll and the trace are unchanged.
    #[arg(long)]
    pub symmetric_pd_loads_host_room: bool,
    /// `--symmetric-pd-p-select cost-function` only: the global waiting queue of
    /// the TRADE P choice with a measured lead (GWQ-AUTO). A request bound to
    /// one P (largest local hit minus the second largest >= C, the engine
    /// chunk from /server_info, 8192 when not reported) takes TRADE at
    /// arrival and is never held. A free request waits in one router queue,
    /// in HRRN order (the held work ahead counts as virtual backlog), until
    /// its TRADE argmin P has W(P) <= lead(P): W = queued work / R, lead =
    /// C / R + T_react(P). T_react = the time from a release to the first
    /// progress poll that lists the RID (time-weighted mean with tau =
    /// --co-schedule-trade-auto-tau-secs; only releases to an idle P (no
    /// queued and no launched work) count; poll period + 500 ms before the
    /// first sample). A
    /// request held 300 s goes at once. With --symmetric-pd-loads-host-room each
    /// release also records shadow counters (phi, class-0 stock, the (ii)
    /// hold); no memory rule acts. Off by default: no request is held and
    /// the trace is unchanged.
    #[arg(long)]
    pub symmetric_pd_gwq_auto: bool,
    /// `--symmetric-pd-gwq-auto` only: how the queue decides that a P has room.
    /// `tokens` (default): W(P) = queued work / R <= lead(P). `predicted`:
    /// room also while the P's predicted drain (the E3 step-time-model
    /// forecast of a cold one-chunk probe on the P's newest engine state,
    /// the state push or the polled prefill_routing snapshot, plus the
    /// router's own rows) is at most lead(P) = C / R + T_react(P); the walk
    /// then prices the P's wait as min(R x drain, ledger backlog). A P
    /// without a usable forecast keeps the token rule (the forecast may only
    /// release earlier). Needs the E3 forecast state:
    /// --co-schedule-trade-e3-bound-table (and best
    /// --co-schedule-trade-e3-push-port), with --co-schedule-trade-wait
    /// ledger (the drain only) or e3.
    #[arg(long, value_parser = ["tokens", "predicted"])]
    pub symmetric_pd_gwq_room: Option<String>,
    /// `--symmetric-pd-p-select cost-function` only: each TRADE reserve event of
    /// a request with a session identity gets a `session_move` record (the
    /// session, the P of its last decision, the chosen P, the largest local
    /// hit over the P, the local hit on the last P, the pulled tokens), so
    /// the share of moved sessions comes from the router trace alone. Off by
    /// default: the trace is unchanged.
    #[arg(long)]
    pub symmetric_pd_trace_session_moves: bool,
    /// With `--symmetric-pd-p-select cost-function`: the pull source of a P-leg may
    /// be any holder of the indexer answer that is a healthy registered
    /// worker of any role (breaker allowing) with a `--kv-worker-map` entry,
    /// also a D-role worker such as a former P after a P->D role switch. The
    /// P-leg targets stay the healthy P-role workers. Off (default): every
    /// hit and every pull source is a P-role worker.
    #[arg(long)]
    pub symmetric_pd_pull_from_any_holder: bool,
    /// Co-schedule: preserve cache owners below this existing uncached workload (default 12288).
    #[arg(long)]
    pub symmetric_pd_owner_threshold_tokens: Option<u64>,
    /// Co-schedule: a detour must save more work than this margin (default 0).
    #[arg(long)]
    pub symmetric_pd_detour_margin_tokens: Option<u64>,
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
    #[arg(long)]
    pub symmetric_pd_decision_replay: Option<std::path::PathBuf>,
    /// Header carrying the routing key for `--symmetric-pd-decision-replay`
    /// (default `X-SMG-Routing-Key`, what the strict-replay client sends).
    #[arg(long)]
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
    /// Applies in plain PD and co_schedule.
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
    #[arg(long)]
    pub pd_decision_replay: Option<std::path::PathBuf>,
    /// Header carrying the routing key for `--pd-decision-replay`
    /// (default `X-SMG-Routing-Key`).
    #[arg(long)]
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
    /// Token-id replay with uploaded screenshots needs more: the Kimi CUA
    /// OSWorld2 trace reaches 102 MiB per request (p95 38 MiB).
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
    /// Pixel-free late-D decode legs (Kimi CUA; needs `--image-layout-config`
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
    /// Late D (engines run `--disaggregation-late-d choose`): `off` (default)
    /// = one decode leg chosen at arrival; `choose` = the request goes to the
    /// prefill and to every decode candidate at once, and the prefill asks
    /// this router for the winner (link to its `late_d.control_port`) when it
    /// is about to prefill. `/v1/completions` only (chat answers 501).
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
    #[arg(long, value_parser = ["off", "on"])]
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
    /// no `po2` rule and no flex decision replay (external role control is
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
                (None, true) => return Err(anyhow!(
                    "--pd-late-decode-rule fit_req requires --pd-late-decode-fit-capacity-tokens \
                         (fixed capacity) or --pd-late-decode-fit-native (observed capacity)"
                )),
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

    /// Resolve parsed flags into a validated [`Config`].
    ///
    /// Builds the [`DiscoveryBackend`] (enforcing static-vs-k8s mutual
    /// exclusivity and resolving the k8s selector grammar via
    /// [`resolve_mode`]), assembles the single [`ModelConfig`], then runs
    /// [`Config::validate`] for the remaining value-level invariants
    /// (model id, static worker URLs).
    /// `--symmetric-pd-preset paper`: set every flag of the paper's
    /// configuration that is still unset. Features whose inputs are missing
    /// stay off: the cost function needs the KV directory and the servicers.
    fn apply_symmetric_pd_preset(&mut self) {
        use crate::config::{LateDecodeMode, LateDecodeRule};
        if self.policy != PolicyKind::SymmetricPd || self.symmetric_pd_preset != SymmetricPdPreset::Paper {
            return;
        }
        self.symmetric_pd_force_split = true;
        self.symmetric_pd_d_select.get_or_insert(SymmetricPdDSelect::RoundRobin);
        self.symmetric_pd_load_poll_ms.get_or_insert(100);
        if self.pd_late_decode.is_none() && self.symmetric_pd_decision_replay.is_none() {
            self.pd_late_decode = Some(LateDecodeMode::Choose);
            let rule = *self.pd_late_decode_rule.get_or_insert(LateDecodeRule::FitReq);
            self.pd_late_decode_candidates.get_or_insert_with(|| "all".into());
            self.pd_late_decode_local.get_or_insert_with(|| "off".into());
            if rule == LateDecodeRule::FitReq && self.pd_late_decode_fit_capacity_tokens.is_none() {
                self.pd_late_decode_fit_native = true;
                self.pd_native_decode_state = true;
            }
        }
        self.symmetric_pd_kv_session_retention = true;
        let servicers = self.kv_worker_map.as_deref().is_some_and(|map| map.contains("servicer"));
        if self.kv_indexer_url.is_some() && servicers && self.symmetric_pd_p_select.is_none() {
            self.symmetric_pd_p_select = Some(SymmetricPdPSelect::CostFunction);
            self.symmetric_pd_pull_hint = true;
            self.symmetric_pd_pull_from_any_holder = true;
            self.kv_pull_scope.get_or_insert(KvPullScope::Any);
            self.co_schedule_trade = true;
            self.co_schedule_uncached_balance = true;
            self.co_schedule_load_poll_ms.get_or_insert(100);
            self.kv_pull_hint_min_incremental_tokens.get_or_insert(512);
        }
    }

    pub fn into_config(mut self) -> Result<Config> {
        self.apply_symmetric_pd_preset();
        let mut symmetric_pd = self.build_symmetric_pd()?;
        // `--symmetric-pd-p-select cost-function`: the TRADE flags of the
        // co-schedule policy configure the symmetric_pd P selection.
        let flex_cost_function =
            symmetric_pd.is_some() && self.symmetric_pd_p_select == Some(SymmetricPdPSelect::CostFunction);
        let gateway_cache_aware = self.build_gateway_cache_aware()?;
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
                let header = match &self.pd_decision_replay_header {
                    Some(h) if h.trim().is_empty() => {
                        return Err(anyhow!("--pd-decision-replay-header must not be empty"))
                    }
                    Some(h) => h.trim().to_string(),
                    None => crate::config::types::default_symmetric_pd_decision_replay_header(),
                };
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
        let tuned_cache_aware = self.dynamo_cache_aware || self.cache_threshold.is_some()
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
        if self.dynamo_cache_aware && (self.inject_kv_pull_hint || self.kv_indexer_endpoint.is_some()) {
            return Err(anyhow!("Dynamo cache-aware uses one indexer and does not inject cross-P pulls"));
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
                    | PolicyKind::ExperimentalAgentSticky
                    | PolicyKind::GatewayCacheAware
            ) {
                return Err(anyhow!(
                    "--sticky-fallback-policy must be one of round_robin / random / \
                     power_of_two / load_based; cache_aware_zmq and sticky are not allowed"
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

        // Co-schedule policy: reject orphan tuning knobs, then build the
        // config with per-field defaults. The policy consumes the kv-hint
        // machinery (indexer for per-worker hits, worker map for sources),
        // so `--policy co_schedule` requires those flags even without
        // `--inject-kv-pull-hint`.
        let co_sched_selected = self.policy == PolicyKind::CoSchedule;
        let tuned_co_sched = self.co_schedule_on_demand_load_refresh
            || self.co_schedule_prefer_gpu_hit
            || self.co_schedule_uncached_balance
            || self.co_schedule_owner_threshold_tokens.is_some()
            || self.co_schedule_detour_margin_tokens.is_some()
            || self.co_schedule_restore_cost_divisor.is_some()
            || self.co_schedule_trade
            || self.co_schedule_locality_margin_tokens.is_some()
            || self.co_schedule_cost_balance
            || self.co_schedule_cost_trace.is_some()
            || self.co_schedule_max_uncached_tokens.is_some()
            || self.co_schedule_max_queue_wait_ms.is_some()
            || self.co_schedule_load_poll_ms.is_some()
            || self.co_schedule_tick_ms.is_some()
            || self.co_schedule_poll_on_hold_ms.is_some()
            || self.co_schedule_imbalance_spread_tokens.is_some()
            || self
                .co_schedule_global_queue
                .as_deref()
                .is_some_and(|mode| mode != "off")
            || self.kv_pull_scope.is_some();
        if tuned_co_sched && !co_sched_selected && !flex_cost_function {
            return Err(anyhow!(
                "--co-schedule-max-uncached-tokens / --co-schedule-max-queue-wait-ms / \
                 --co-schedule-load-poll-ms / --co-schedule-tick-ms / \
                 --co-schedule-poll-on-hold-ms / --co-schedule-imbalance-spread-tokens / \
                 --co-schedule-global-queue / --kv-pull-scope require --policy co_schedule, or \
                 --policy symmetric_pd with --symmetric-pd-p-select cost-function"
            ));
        }
        // The on-demand refresh belongs to the CoScheduler admission; the
        // symmetric_pd cost function polls the P-role workers on a timer.
        if flex_cost_function && self.co_schedule_on_demand_load_refresh {
            return Err(anyhow!(
                "--co-schedule-on-demand-load-refresh is not available with \
                 --symmetric-pd-p-select cost-function: the P progress is polled every \
                 --co-schedule-load-poll-ms"
            ));
        }
        if self.co_schedule_cost_trace_async && !flex_cost_function {
            return Err(anyhow!(
                "--co-schedule-cost-trace-async requires --policy symmetric_pd --symmetric-pd-p-select \
                 cost-function"
            ));
        }
        // Session KV retention: the co_schedule and gateway_cache_aware PD
        // paths. symmetric_pd keeps its own registry and flag
        // (--symmetric-pd-kv-session-retention).
        let gateway_ca_selected = self.policy == PolicyKind::GatewayCacheAware;
        if self.kv_session_retention && !co_sched_selected && !gateway_ca_selected {
            return Err(if self.policy == PolicyKind::SymmetricPd {
                anyhow!(
                    "--kv-session-retention is not available with --policy symmetric_pd; \
                     use --symmetric-pd-kv-session-retention"
                )
            } else {
                anyhow!(
                    "--kv-session-retention requires --policy co_schedule or --policy \
                     gateway_cache_aware"
                )
            });
        }
        let mode_count = u8::from(self.co_schedule_cost_balance)
            + u8::from(self.co_schedule_locality_margin_tokens.is_some())
            + u8::from(self.co_schedule_uncached_balance);
        let reserved_mode = mode_count > 0;
        if self.co_schedule_on_demand_load_refresh && !self.co_schedule_uncached_balance {
            return Err(anyhow!("on-demand load refresh requires --co-schedule-uncached-balance"));
        }
        if mode_count > 1 {
            return Err(anyhow!(
                "cost balance, locality margin and uncached balance are mutually exclusive"
            ));
        }
        if self.co_schedule_owner_threshold_tokens.is_some() && !self.co_schedule_uncached_balance {
            return Err(anyhow!(
                "owner threshold requires --co-schedule-uncached-balance"
            ));
        }
        if self.co_schedule_restore_cost_divisor.is_some()
            && (!self.co_schedule_uncached_balance
                || self.co_schedule_restore_cost_divisor == Some(0))
        {
            return Err(anyhow!(
                "restore cost divisor must be positive and requires --co-schedule-uncached-balance"
            ));
        }
        if self.co_schedule_detour_margin_tokens.is_some()
            && (!self.co_schedule_uncached_balance
                || self.co_schedule_restore_cost_divisor.is_none())
        {
            return Err(anyhow!(
                "detour margin requires uncached balance and a restore cost divisor"
            ));
        }
        let mut trade_eft_model = None;
        let mut trade_olp: Option<crate::olp::OlpConfig> = None;
        let mut trade_e3 = None;
        let trade = if self.co_schedule_trade {
            if !self.co_schedule_uncached_balance {
                return Err(anyhow!(
                    "--co-schedule-trade requires --co-schedule-uncached-balance"
                ));
            }
            if self.co_schedule_owner_threshold_tokens.is_some()
                || self.co_schedule_restore_cost_divisor.is_some()
                || self.co_schedule_detour_margin_tokens.is_some()
            {
                return Err(anyhow!(
                    "--co-schedule-trade replaces the owner threshold, restore divisor and \
                     detour margin"
                ));
            }
            let floor = crate::scheduler::co_schedule::TRADE_MIN_PULL_TOKENS;
            if self
                .kv_pull_hint_min_incremental_tokens
                .is_some_and(|n| u64::from(n) < floor)
            {
                return Err(anyhow!(
                    "--co-schedule-trade never pulls fewer than {floor} tokens: \
                     --kv-pull-hint-min-incremental-tokens must be at least {floor} \
                     (absent = {floor})"
                ));
            }
            if self.kv_pull_hint_local_prefix_fallback {
                return Err(anyhow!(
                    "--co-schedule-trade sends a hint only for a pull of at least the floor; \
                     it cannot be combined with --kv-pull-hint-local-prefix-fallback"
                ));
            }
            let defaults = crate::scheduler::co_schedule::TradeParams::default();
            let lambda_auto = self.co_schedule_trade_copy_lambda == Some(NumberOrAuto::Auto);
            let rate_auto = self.co_schedule_trade_rate_tokens_per_s == Some(NumberOrAuto::Auto);
            if !lambda_auto
                && (self.co_schedule_trade_copy_lambda_q.is_some()
                    || self.co_schedule_trade_copy_lambda_mu.is_some()
                    || self.co_schedule_trade_copy_lambda_min.is_some()
                    || self.co_schedule_trade_copy_lambda_max.is_some()
                    || self.co_schedule_trade_copy_lambda_fallback.is_some())
            {
                return Err(anyhow!(
                    "--co-schedule-trade-copy-lambda-q, -mu, -min, -max and -fallback need \
                     --co-schedule-trade-copy-lambda auto"
                ));
            }
            // Pull price phi = q x phi_P: q of the copy price auto with mu 1
            // and R = R_hat, phi_P from the engine host_room counters of the
            // symmetric_pd cost function poll.
            let phi = self.co_schedule_trade_pull_price.as_deref() == Some("phi");
            if phi && !flex_cost_function {
                return Err(anyhow!(
                    "--co-schedule-trade-pull-price phi requires --policy symmetric_pd with \
                     --symmetric-pd-p-select cost-function"
                ));
            }
            if phi && !self.symmetric_pd_loads_host_room {
                return Err(anyhow!(
                    "--co-schedule-trade-pull-price phi requires --symmetric-pd-loads-host-room \
                     (phi comes from the engine host_room counters)"
                ));
            }
            if phi && !lambda_auto {
                return Err(anyhow!(
                    "--co-schedule-trade-pull-price phi requires \
                     --co-schedule-trade-copy-lambda auto (q comes from the copy price auto)"
                ));
            }
            if phi && !rate_auto {
                return Err(anyhow!(
                    "--co-schedule-trade-pull-price phi requires \
                     --co-schedule-trade-rate-tokens-per-s auto (R = R_hat)"
                ));
            }
            // The duplicate charge would use the per-P price (0 at phi 0
            // without a floor): refused rather than mixed.
            if phi && self.co_schedule_trade_dup_charge.as_deref() == Some("all") {
                return Err(anyhow!(
                    "--co-schedule-trade-pull-price phi does not take \
                     --co-schedule-trade-dup-charge all (the duplicate charge would use the \
                     per-P price)"
                ));
            }
            if !(lambda_auto || rate_auto)
                && (self.co_schedule_trade_auto_tau_secs.is_some()
                    || self.co_schedule_trade_auto_session_idle_secs.is_some())
            {
                return Err(anyhow!(
                    "--co-schedule-trade-auto-tau-secs and \
                     --co-schedule-trade-auto-session-idle-secs need \
                     --co-schedule-trade-copy-lambda auto or \
                     --co-schedule-trade-rate-tokens-per-s auto"
                ));
            }
            let auto = (
                self.co_schedule_trade_copy_lambda_mu
                    .unwrap_or(crate::scheduler::co_schedule::DEFAULT_TRADE_AUTO_MU),
                self.co_schedule_trade_copy_lambda_min
                    .unwrap_or(crate::scheduler::co_schedule::DEFAULT_TRADE_AUTO_LAMBDA_MIN),
                self.co_schedule_trade_copy_lambda_max
                    .unwrap_or(crate::scheduler::co_schedule::DEFAULT_TRADE_AUTO_LAMBDA_MAX),
            );
            for (name, value) in [
                ("--co-schedule-trade-copy-lambda-mu", auto.0),
                ("--co-schedule-trade-copy-lambda-min", auto.1),
                ("--co-schedule-trade-copy-lambda-max", auto.2),
                (
                    "--co-schedule-trade-copy-lambda-fallback",
                    self.co_schedule_trade_copy_lambda_fallback
                        .unwrap_or(defaults.copy_lambda),
                ),
            ] {
                if !(value.is_finite() && value >= 0.0) {
                    return Err(anyhow!("{name} must be finite and at least 0"));
                }
            }
            if auto.1 > auto.2 {
                return Err(anyhow!(
                    "--co-schedule-trade-copy-lambda-min must not be above \
                     --co-schedule-trade-copy-lambda-max"
                ));
            }
            if phi && auto.0 != 1.0 {
                return Err(anyhow!(
                    "--co-schedule-trade-pull-price phi requires \
                     --co-schedule-trade-copy-lambda-mu 1"
                ));
            }
            for (name, value) in [
                (
                    "--co-schedule-trade-auto-tau-secs",
                    self.co_schedule_trade_auto_tau_secs
                        .unwrap_or(defaults.auto_tau_s),
                ),
                (
                    "--co-schedule-trade-auto-session-idle-secs",
                    self.co_schedule_trade_auto_session_idle_secs
                        .unwrap_or(defaults.auto_session_idle_s),
                ),
            ] {
                if !(value.is_finite() && value > 0.0) {
                    return Err(anyhow!("{name} must be finite and greater than 0"));
                }
            }
            let params = crate::scheduler::co_schedule::TradeParams {
                // auto: the default R is the fallback.
                rate_tokens_per_s: match self.co_schedule_trade_rate_tokens_per_s {
                    Some(NumberOrAuto::Number(rate)) => rate,
                    _ => defaults.rate_tokens_per_s,
                },
                pull_fixed_ms: self
                    .co_schedule_trade_pull_fixed_ms
                    .unwrap_or(defaults.pull_fixed_ms),
                pull_ms_per_100k: self
                    .co_schedule_trade_pull_ms_per_100k
                    .unwrap_or(defaults.pull_ms_per_100k),
                // auto: the fallback lambda.
                copy_lambda: match self.co_schedule_trade_copy_lambda {
                    Some(NumberOrAuto::Number(lambda)) => lambda,
                    Some(NumberOrAuto::Auto) => self
                        .co_schedule_trade_copy_lambda_fallback
                        .unwrap_or(defaults.copy_lambda),
                    None => defaults.copy_lambda,
                },
                copy_lambda_mode: if lambda_auto {
                    crate::scheduler::co_schedule::TradeLambdaMode::Auto {
                        q: match self.co_schedule_trade_copy_lambda_q.as_deref() {
                            Some("inflight") => crate::scheduler::co_schedule::TradeAutoQ::Inflight,
                            _ => crate::scheduler::co_schedule::TradeAutoQ::Sat,
                        },
                        mu: auto.0,
                        min: auto.1,
                        max: auto.2,
                    }
                } else {
                    crate::scheduler::co_schedule::TradeLambdaMode::Fixed
                },
                rate_mode: if rate_auto {
                    crate::scheduler::co_schedule::TradeRateMode::Auto
                } else {
                    crate::scheduler::co_schedule::TradeRateMode::Fixed
                },
                auto_tau_s: self
                    .co_schedule_trade_auto_tau_secs
                    .unwrap_or(defaults.auto_tau_s),
                auto_session_idle_s: self
                    .co_schedule_trade_auto_session_idle_secs
                    .unwrap_or(defaults.auto_session_idle_s),
                exclude_unlaunched: self.co_schedule_trade_exclude_unlaunched,
                wait: match self.co_schedule_trade_wait.as_deref() {
                    Some("eft") => crate::scheduler::co_schedule::TradeWait::Eft,
                    Some("olp") => crate::scheduler::co_schedule::TradeWait::Olp,
                    Some("e3") => crate::scheduler::co_schedule::TradeWait::E3,
                    _ => crate::scheduler::co_schedule::TradeWait::Ledger,
                },
                dup_charge: match self.co_schedule_trade_dup_charge.as_deref() {
                    Some("all") => crate::scheduler::co_schedule::TradeDupCharge::All,
                    _ => crate::scheduler::co_schedule::TradeDupCharge::Pulled,
                },
                margin_noise_k: self
                    .co_schedule_trade_margin_noise
                    .unwrap_or(defaults.margin_noise_k),
                eft_max_view_age_s: self
                    .co_schedule_trade_eft_max_view_age_ms
                    .map_or(defaults.eft_max_view_age_s, |ms| ms / 1000.0),
                hol_weight: self
                    .co_schedule_trade_hol_weight
                    .unwrap_or(defaults.hol_weight),
                hol_rate_window_s: self
                    .co_schedule_trade_hol_rate_window_secs
                    .unwrap_or(defaults.hol_rate_window_s),
                pull_price: match self.co_schedule_trade_pull_price.as_deref() {
                    Some("phi") => crate::scheduler::co_schedule::TradePullPrice::Phi,
                    _ => crate::scheduler::co_schedule::TradePullPrice::Lambda,
                },
                gate_m: self.co_schedule_trade_gate_m.unwrap_or(defaults.gate_m),
                gate_gmin_s: self
                    .co_schedule_trade_gate_gmin_ms
                    .map_or(defaults.gate_gmin_s, |ms| ms / 1000.0),
            };
            if !(params.rate_tokens_per_s.is_finite() && params.rate_tokens_per_s > 0.0) {
                return Err(anyhow!(
                    "--co-schedule-trade-rate-tokens-per-s must be finite and greater than 0"
                ));
            }
            for (name, value) in [
                ("--co-schedule-trade-pull-fixed-ms", params.pull_fixed_ms),
                ("--co-schedule-trade-pull-ms-per-100k", params.pull_ms_per_100k),
                ("--co-schedule-trade-copy-lambda", params.copy_lambda),
                ("--co-schedule-trade-margin-noise", params.margin_noise_k),
                ("--co-schedule-trade-hol-weight", params.hol_weight),
                ("--co-schedule-trade-gate-m", params.gate_m),
                ("--co-schedule-trade-gate-gmin-ms", params.gate_gmin_s),
            ] {
                if !(value.is_finite() && value >= 0.0) {
                    return Err(anyhow!("{name} must be finite and at least 0"));
                }
            }
            if !(params.hol_rate_window_s.is_finite() && params.hol_rate_window_s > 0.0) {
                return Err(anyhow!(
                    "--co-schedule-trade-hol-rate-window-secs must be finite and greater than 0"
                ));
            }
            let eft = params.wait == crate::scheduler::co_schedule::TradeWait::Eft;
            // `wait eft` (EFT-0+) is not wired into the symmetric_pd cost function;
            // its step-time-model successor `wait e3` is.
            if flex_cost_function && eft {
                return Err(anyhow!(
                    "--co-schedule-trade-wait eft is not available with --symmetric-pd-p-select \
                     cost-function; use --co-schedule-trade-wait e3 (the step-time-model predictor)"
                ));
            }
            let e3 = params.wait == crate::scheduler::co_schedule::TradeWait::E3;
            if !(params.eft_max_view_age_s.is_finite() && params.eft_max_view_age_s > 0.0) {
                return Err(anyhow!(
                    "--co-schedule-trade-eft-max-view-age-ms must be finite and greater than 0"
                ));
            }
            match (eft, self.co_schedule_trade_eft_model.as_deref()) {
                (true, Some(path)) => {
                    let model = crate::eft_cheap::load_model(path)
                        .map_err(|e| anyhow!("--co-schedule-trade-eft-model: {e}"))?;
                    trade_eft_model = Some(std::sync::Arc::new(model));
                }
                (true, None) => {
                    return Err(anyhow!(
                        "--co-schedule-trade-wait eft requires --co-schedule-trade-eft-model"
                    ));
                }
                (false, _)
                    if self.co_schedule_trade_eft_model.is_some()
                        || (!e3 && self.co_schedule_trade_eft_max_view_age_ms.is_some()) =>
                {
                    return Err(anyhow!(
                        "--co-schedule-trade-eft-model and --co-schedule-trade-eft-max-view-age-ms \
                         need --co-schedule-trade-wait eft (the age limit also applies to e3)"
                    ));
                }
                (false, _) => {}
            }
            // `wait olp` and the shadow readout: the online latency
            // predictor of the symmetric_pd cost function only (it reads the cost
            // book's own ledger rows and progress poll).
            let olp = params.wait == crate::scheduler::co_schedule::TradeWait::Olp;
            let shadow = self.co_schedule_trade_olp_shadow;
            if olp && !flex_cost_function {
                return Err(anyhow!(
                    "--co-schedule-trade-wait olp requires --policy symmetric_pd --symmetric-pd-p-select \
                     cost-function"
                ));
            }
            if shadow && !flex_cost_function {
                return Err(anyhow!(
                    "--co-schedule-trade-olp-shadow requires --policy symmetric_pd \
                     --symmetric-pd-p-select cost-function"
                ));
            }
            if shadow && params.wait != crate::scheduler::co_schedule::TradeWait::Ledger {
                return Err(anyhow!(
                    "--co-schedule-trade-olp-shadow needs --co-schedule-trade-wait ledger: the \
                     shadow readout never changes a decision"
                ));
            }
            if olp || shadow {
                let (olp_params, params_sha256) = match &self.co_schedule_trade_olp_params {
                    Some(path) => {
                        let (olp_params, sha256) = crate::olp::load_params(path)
                            .map_err(|e| anyhow!("--co-schedule-trade-olp-params {e}"))?;
                        (olp_params, Some(sha256))
                    }
                    None => (crate::olp::OlpParams::default(), None),
                };
                trade_olp = Some(crate::olp::OlpConfig {
                    params: olp_params,
                    shadow,
                    params_sha256,
                    incremental: self.co_schedule_trade_olp_incremental,
                    compact_trace: self.co_schedule_trade_olp_trace_compact,
                    verify: false,
                });
            } else if self.co_schedule_trade_olp_params.is_some() {
                return Err(anyhow!(
                    "--co-schedule-trade-olp-params needs --co-schedule-trade-wait olp or \
                     --co-schedule-trade-olp-shadow"
                ));
            } else if self.co_schedule_trade_olp_incremental {
                return Err(anyhow!(
                    "--co-schedule-trade-olp-incremental needs --co-schedule-trade-wait olp or \
                     --co-schedule-trade-olp-shadow"
                ));
            } else if self.co_schedule_trade_olp_trace_compact {
                return Err(anyhow!(
                    "--co-schedule-trade-olp-trace-compact needs --co-schedule-trade-wait olp or \
                     --co-schedule-trade-olp-shadow"
                ));
            }
            let e3_flags = self.co_schedule_trade_e3_bound_table.is_some()
                || self.co_schedule_trade_e3_bound_mode.is_some()
                || self.co_schedule_trade_e3_bound_level.is_some()
                || self.co_schedule_trade_gate_m.is_some()
                || self.co_schedule_trade_gate_gmin_ms.is_some()
                || self.co_schedule_trade_e3_rq_margin_ms.is_some()
                || self.co_schedule_trade_e3_shadow.is_some()
                || self.co_schedule_trade_e3_push_port.is_some()
                || self.co_schedule_trade_e3_push_bound_mode.is_some()
                || self.co_schedule_trade_e3_next_pass_lag
                || self.co_schedule_trade_e3_frozen_fit.is_some();
            // `--co-schedule-global-queue-room predicted` needs the E3 forecast
            // state also with the ledger wait (the drain only; no gate).
            let room_predicted = (self.co_schedule_global_queue_room.as_deref() == Some("predicted")
                && matches!(
                    self.co_schedule_global_queue.as_deref(),
                    Some("sjf" | "hrrn")
                ))
                // The symmetric_pd GWQ-AUTO queue (`--symmetric-pd-gwq-room predicted`).
                || (flex_cost_function
                    && self.symmetric_pd_gwq_auto
                    && self.symmetric_pd_gwq_room.as_deref() == Some("predicted"));
            if !e3
                && room_predicted
                && (self.co_schedule_trade_gate_m.is_some()
                    || self.co_schedule_trade_gate_gmin_ms.is_some())
            {
                return Err(anyhow!(
                    "--co-schedule-trade-gate-* need --co-schedule-trade-wait e3 (the GWQ drain \
                     of --co-schedule-global-queue-room predicted or --symmetric-pd-gwq-room \
                     predicted has no gate)"
                ));
            }
            if e3 || room_predicted {
                if e3 && params.margin_noise_k > 0.0 {
                    return Err(anyhow!(
                        "--co-schedule-trade-wait e3 uses the prediction gate \
                         (--co-schedule-trade-gate-m); --co-schedule-trade-margin-noise must be 0"
                    ));
                }
                let path = self.co_schedule_trade_e3_bound_table.as_deref().ok_or_else(|| {
                    if e3 {
                        anyhow!("--co-schedule-trade-wait e3 requires --co-schedule-trade-e3-bound-table")
                    } else {
                        let flag = if flex_cost_function {
                            "--symmetric-pd-gwq-room"
                        } else {
                            "--co-schedule-global-queue-room"
                        };
                        anyhow!(
                            "{flag} predicted requires --co-schedule-trade-e3-bound-table (the E3 \
                             forecast state)"
                        )
                    }
                })?;
                let rq_margin_ms = self.co_schedule_trade_e3_rq_margin_ms.unwrap_or(0.0);
                if !(rq_margin_ms.is_finite() && rq_margin_ms >= 0.0) {
                    return Err(anyhow!(
                        "--co-schedule-trade-e3-rq-margin-ms must be finite and at least 0"
                    ));
                }
                let config = crate::trade_e3::TradeE3Config::load(
                    path,
                    self.co_schedule_trade_e3_bound_mode
                        .as_deref()
                        .unwrap_or(crate::trade_e3::DEFAULT_BOUND_MODE),
                    self.co_schedule_trade_e3_bound_level.as_deref(),
                    rq_margin_ms / 1000.0,
                    self.co_schedule_trade_e3_shadow.clone(),
                )
                .map_err(|e| anyhow!("--co-schedule-trade-e3-bound-table: {e}"))?;
                let config = match self.co_schedule_trade_e3_push_port {
                    Some(port) => config
                        .with_push(
                            port,
                            self.co_schedule_trade_e3_push_bound_mode.as_deref(),
                            self.co_schedule_trade_e3_bound_level.as_deref(),
                        )
                        .map_err(|e| anyhow!("--co-schedule-trade-e3-push-port: {e}"))?,
                    None if self.co_schedule_trade_e3_push_bound_mode.is_some() => {
                        return Err(anyhow!(
                            "--co-schedule-trade-e3-push-bound-mode needs \
                             --co-schedule-trade-e3-push-port"
                        ));
                    }
                    None => config,
                };
                let config = config.with_next_pass_lag(self.co_schedule_trade_e3_next_pass_lag);
                let frozen = self
                    .co_schedule_trade_e3_frozen_fit
                    .as_deref()
                    .map(crate::trade_e3::FrozenFit::load)
                    .transpose()
                    .map_err(|e| anyhow!("--co-schedule-trade-e3-frozen-fit: {e}"))?
                    .map(std::sync::Arc::new);
                trade_e3 = Some(std::sync::Arc::new(config.with_frozen_fit(frozen)));
            } else if e3_flags {
                return Err(anyhow!(
                    "the --co-schedule-trade-e3-* and --co-schedule-trade-gate-* flags \
                     need --co-schedule-trade-wait e3 (the e3 flags also \
                     --co-schedule-global-queue-room predicted or --symmetric-pd-gwq-room predicted)"
                ));
            }
            Some(params)
        } else {
            None
        };
        // The global waiting queue: TRADE with the ledger wait only.
        let global_queue = match self.co_schedule_global_queue.as_deref() {
            None | Some("off") => {
                if self.co_schedule_global_queue_depth_tokens.is_some()
                    || self.co_schedule_global_queue_max_wait_ms.is_some()
                    || self.co_schedule_global_queue_hold.is_some()
                    || self.co_schedule_global_queue_room.is_some()
                {
                    return Err(anyhow!(
                        "--co-schedule-global-queue-depth-tokens, \
                         --co-schedule-global-queue-max-wait-ms, \
                         --co-schedule-global-queue-hold and \
                         --co-schedule-global-queue-room need \
                         --co-schedule-global-queue sjf or hrrn"
                    ));
                }
                None
            }
            Some(mode) => {
                let Some(params) = trade.as_ref() else {
                    return Err(anyhow!(
                        "--co-schedule-global-queue {mode} needs --co-schedule-trade \
                         (the TRADE cost-balance path)"
                    ));
                };
                if params.wait == crate::scheduler::co_schedule::TradeWait::Eft {
                    return Err(anyhow!(
                        "--co-schedule-global-queue is not supported with \
                         --co-schedule-trade-wait eft"
                    ));
                }
                let depth = match self.co_schedule_global_queue_depth_tokens {
                    None | Some(TokensOrAuto::Auto) => crate::global_queue::GlobalQueueDepth::Auto,
                    Some(TokensOrAuto::Tokens(0)) => {
                        return Err(anyhow!(
                            "--co-schedule-global-queue-depth-tokens must be at least 1"
                        ));
                    }
                    Some(TokensOrAuto::Tokens(n)) => {
                        crate::global_queue::GlobalQueueDepth::Tokens(n)
                    }
                };
                let mode = if mode == "sjf" {
                    crate::global_queue::GlobalQueueMode::Sjf
                } else {
                    crate::global_queue::GlobalQueueMode::Hrrn
                };
                let max_wait_ms = self
                    .co_schedule_global_queue_max_wait_ms
                    .unwrap_or(mode.default_max_wait_ms());
                if max_wait_ms == 0 {
                    return Err(anyhow!(
                        "--co-schedule-global-queue-max-wait-ms must be greater than 0"
                    ));
                }
                let hold = match self.co_schedule_global_queue_hold.as_deref() {
                    Some("free") => crate::global_queue::GlobalQueueHold::Free,
                    _ => crate::global_queue::GlobalQueueHold::All,
                };
                let room = match self.co_schedule_global_queue_room.as_deref() {
                    Some("predicted") => {
                        if trade_e3.is_none() {
                            return Err(anyhow!(
                                "--co-schedule-global-queue-room predicted needs the E3 \
                                 forecast state (--co-schedule-trade-e3-bound-table)"
                            ));
                        }
                        crate::global_queue::GlobalQueueRoom::Predicted
                    }
                    _ => crate::global_queue::GlobalQueueRoom::Tokens,
                };
                Some(crate::global_queue::GlobalQueueConfig {
                    mode,
                    depth,
                    max_wait_ms,
                    hold,
                    room,
                })
            }
        };
        if self.co_schedule_uncached_balance
            && (self.co_schedule_max_uncached_tokens.is_some()
                || self.co_schedule_max_queue_wait_ms.is_some()
                || self.co_schedule_tick_ms.is_some()
                || self.co_schedule_poll_on_hold_ms.is_some()
                || self.co_schedule_imbalance_spread_tokens.is_some())
        {
            return Err(anyhow!(
                "uncached balance has no admission cap, HOLD queue or dispatch timer"
            ));
        }
        // The uncached balance writes the cost trace only when one is given;
        // the other reserved modes need it.
        let trace = self.co_schedule_cost_trace.is_some();
        if (reserved_mode && !trace && !self.co_schedule_uncached_balance) || (trace && !reserved_mode) {
            return Err(anyhow!(
                "reserved placement requires --co-schedule-cost-trace, and vice versa"
            ));
        }
        if self.co_schedule_prefer_gpu_hit && reserved_mode {
            return Err(anyhow!(
                "--co-schedule-prefer-gpu-hit requires capped co-schedule mode"
            ));
        }
        if reserved_mode && self.kv_pull_scope == Some(KvPullScope::DetourOnly) {
            return Err(anyhow!("reserved placement requires --kv-pull-scope any"));
        }
        // `--symmetric-pd-p-select cost-function`: the TRADE parameters, the cost
        // trace and the P progress poll period go to the symmetric_pd config.
        if flex_cost_function {
            let trade = trade.ok_or_else(|| {
                anyhow!(
                    "--symmetric-pd-p-select cost-function requires --co-schedule-trade (the TRADE \
                     cost model)"
                )
            })?;
            let load_poll_ms = self.co_schedule_load_poll_ms.unwrap_or(100);
            if load_poll_ms == 0 {
                return Err(anyhow!("--co-schedule-load-poll-ms must be greater than 0"));
            }
            let cost_trace = self.co_schedule_cost_trace.clone();
            let unknown_expire_secs = self
                .symmetric_pd_unknown_expire_secs
                .unwrap_or_else(crate::config::types::default_symmetric_pd_unknown_expire_secs);
            if let Some(flex) = symmetric_pd.as_mut() {
                flex.cost_function = Some(SymmetricPdCostFunctionConfig {
                    trade,
                    cost_trace,
                    load_poll_ms,
                    unknown_expire_secs,
                    loads_host_room: self.symmetric_pd_loads_host_room,
                    gwq_auto: self
                        .symmetric_pd_gwq_auto
                        .then(|| crate::config::SymmetricPdGwqAutoConfig {
                            room: if self.symmetric_pd_gwq_room.as_deref() == Some("predicted") {
                                crate::global_queue::GlobalQueueRoom::Predicted
                            } else {
                                crate::global_queue::GlobalQueueRoom::Tokens
                            },
                            ..Default::default()
                        }),
                    trace_session_moves: self.symmetric_pd_trace_session_moves,
                    trade_e3: trade_e3.clone().map(crate::config::SymmetricPdTradeE3),
                    pull_from_any_holder: self.symmetric_pd_pull_from_any_holder,
                    olp: trade_olp,
                    trace_async: self.co_schedule_cost_trace_async,
                });
            }
        }
        let co_schedule = if co_sched_selected {
            let max_uncached_tokens = self.co_schedule_max_uncached_tokens.unwrap_or(32768);
            let max_queue_wait_ms = self.co_schedule_max_queue_wait_ms.unwrap_or(30_000);
            let load_poll_ms =
                self.co_schedule_load_poll_ms
                    .unwrap_or(if self.co_schedule_uncached_balance {
                        100
                    } else {
                        1000
                    });
            // poll_on_hold and imbalance_spread legitimately accept 0
            // (= feature disabled); the others must be positive.
            let tick_ms = self.co_schedule_tick_ms.unwrap_or(500);
            let poll_on_hold_ms = self.co_schedule_poll_on_hold_ms.unwrap_or(100);
            let imbalance_spread_tokens =
                self.co_schedule_imbalance_spread_tokens.unwrap_or(131_072);
            if max_uncached_tokens == 0
                || max_queue_wait_ms == 0
                || load_poll_ms == 0
                || tick_ms == 0
            {
                return Err(anyhow!(
                    "--co-schedule-max-uncached-tokens / --co-schedule-max-queue-wait-ms / \
                     --co-schedule-load-poll-ms / --co-schedule-tick-ms must be greater than 0"
                ));
            }
            Some(CoScheduleConfig {
                on_demand_load_refresh: self.co_schedule_on_demand_load_refresh,
                uncached_balance: self.co_schedule_uncached_balance,
                owner_threshold_tokens: self.co_schedule_owner_threshold_tokens.unwrap_or(12288),
                detour_margin_tokens: self.co_schedule_detour_margin_tokens.unwrap_or(0),
                restore_cost_divisor: self.co_schedule_restore_cost_divisor,
                trade,
                trade_eft_model,
                global_queue,
                trade_e3,
                prefer_gpu_hit: self.co_schedule_prefer_gpu_hit,
                locality_margin_tokens: self.co_schedule_locality_margin_tokens,
                cost_balance: self.co_schedule_cost_balance,
                cost_trace: self.co_schedule_cost_trace,
                max_uncached_tokens,
                max_queue_wait_ms,
                load_poll_ms,
                tick_ms,
                poll_on_hold_ms,
                imbalance_spread_tokens,
                pull_scope: self.kv_pull_scope.unwrap_or(KvPullScope::Any),
                kv_session_retention: self.kv_session_retention,
                kv_session_key_file: self.kv_session_key_file,
            })
        } else {
            None
        };

        // KV pull-hint injection: reject orphan tuning knobs (mirrors the
        // sticky / cache-aware checks above), then validate the required
        // pieces at startup so a malformed --kv-worker-map fails fast
        // instead of silently never injecting. `--policy co_schedule`
        // implies the machinery even without `--inject-kv-pull-hint`.
        let tuned_kv_hint = self.kv_indexer_url.is_some()
            || self.kv_pull_hint_min_tokens.is_some()
            || self.kv_pull_hint_min_incremental_tokens.is_some()
            || self.kv_pull_hint_local_prefix_fallback
            || self.kv_indexer_timeout_ms.is_some()
            || self.kv_worker_map.is_some();
        // symmetric_pd: the indexer is optional (placement degrades to load /
        // role only, logged), so the flags are accepted but not required.
        let flex_selected = symmetric_pd.is_some();
        if tuned_kv_hint && !self.inject_kv_pull_hint && !co_sched_selected && !flex_selected && !self.dynamo_cache_aware {
            return Err(anyhow!(
                "--kv-indexer-url / --kv-pull-hint-min-tokens / \
                 --kv-pull-hint-min-incremental-tokens / --kv-pull-hint-local-prefix-fallback / --kv-indexer-timeout-ms / \
                 --kv-worker-map require --inject-kv-pull-hint, --policy co_schedule \
                 or --policy symmetric_pd"
            ));
        }
        let flex_wants_indexer = flex_selected && self.kv_indexer_url.is_some();
        let mut kv_hint = if self.inject_kv_pull_hint || co_sched_selected || flex_wants_indexer || self.dynamo_cache_aware {
            let requirer = if self.inject_kv_pull_hint {
                "--inject-kv-pull-hint"
            } else if co_sched_selected {
                "--policy co_schedule"
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
            let timeout_ms = self.kv_indexer_timeout_ms.unwrap_or(20);
            if timeout_ms == 0 {
                return Err(anyhow!("--kv-indexer-timeout-ms must be greater than 0"));
            }
            Some(crate::kv_hint::KvHintConfig {
                indexer_url,
                min_tokens: self.kv_pull_hint_min_tokens.unwrap_or(256),
                // TRADE never pulls below 512, so its gate defaults to 512.
                min_incremental_tokens: self.kv_pull_hint_min_incremental_tokens.unwrap_or(
                    if self.co_schedule_trade {
                        crate::scheduler::co_schedule::TRADE_MIN_PULL_TOKENS as u32
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
        if let Some(flex) = symmetric_pd.as_ref().filter(|_| flex_cost_function) {
            let kv = kv_hint.as_ref().ok_or_else(|| {
                anyhow!(
                    "--symmetric-pd-p-select cost-function requires --kv-indexer-url and \
                     --kv-worker-map"
                )
            })?;
            let p_roles: Vec<&SymmetricPdWorkerSpec> = flex
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
                dynamo: if self.dynamo_cache_aware { kv_hint.take() } else { None },
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
                co_schedule,
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
        let flex_ca = self.policy == PolicyKind::SymmetricPd
            && self.symmetric_pd_p_select == Some(SymmetricPdPSelect::GatewayCacheAware);
        if self.policy != PolicyKind::GatewayCacheAware {
            if tuned && !flex_ca {
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
            || self.symmetric_pd_owner_threshold_tokens.is_some()
            || self.symmetric_pd_detour_margin_tokens.is_some()
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
            || self.symmetric_pd_loads_host_room
            || self.symmetric_pd_gwq_auto
            || self.symmetric_pd_gwq_room.is_some()
            || self.symmetric_pd_trace_session_moves
            || self.symmetric_pd_pull_from_any_holder;
        if tuned && !selected {
            return Err(anyhow!("--symmetric-pd-* flags require --policy symmetric_pd"));
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
            let file_workers: Vec<SymmetricPdWorkerSpec> = serde_json::from_str(&text).map_err(|e| {
                anyhow!("--symmetric-pd-workers-file {path:?} is not a JSON array of workers: {e}")
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
        let mut placement = crate::symmetric_pd::placement::PlacementConfig {
            force_split: self.symmetric_pd_force_split,
            ..Default::default()
        };
        if let Some(v) = self.symmetric_pd_p_select {
            placement.p_select = v.into();
        }
        if placement.p_select == crate::symmetric_pd::placement::PSelect::CostFunction {
            self.check_flex_cost_function()?;
        } else if self.symmetric_pd_unknown_expire_secs.is_some() {
            return Err(anyhow!(
                "--symmetric-pd-unknown-expire-secs requires --symmetric-pd-p-select cost-function"
            ));
        } else if self.symmetric_pd_loads_host_room {
            return Err(anyhow!(
                "--symmetric-pd-loads-host-room requires --symmetric-pd-p-select cost-function"
            ));
        } else if self.symmetric_pd_gwq_auto {
            return Err(anyhow!(
                "--symmetric-pd-gwq-auto requires --symmetric-pd-p-select cost-function"
            ));
        } else if self.symmetric_pd_gwq_room.is_some() {
            return Err(anyhow!(
                "--symmetric-pd-gwq-room requires --symmetric-pd-p-select cost-function"
            ));
        } else if self.symmetric_pd_trace_session_moves {
            return Err(anyhow!(
                "--symmetric-pd-trace-session-moves requires --symmetric-pd-p-select cost-function"
            ));
        } else if self.symmetric_pd_pull_from_any_holder {
            return Err(anyhow!(
                "--symmetric-pd-pull-from-any-holder requires --symmetric-pd-p-select cost-function"
            ));
        }
        if self.symmetric_pd_owner_threshold_tokens.is_some() || self.symmetric_pd_detour_margin_tokens.is_some() {
            if placement.p_select != crate::symmetric_pd::placement::PSelect::CoSchedule {
                return Err(anyhow!("--symmetric-pd-owner-threshold-tokens / --symmetric-pd-detour-margin-tokens require --symmetric-pd-p-select co-schedule"));
            }
        }
        if let Some(v) = self.symmetric_pd_owner_threshold_tokens { placement.co_schedule.owner_threshold_tokens = v; }
        if let Some(v) = self.symmetric_pd_detour_margin_tokens { placement.co_schedule.detour_margin_tokens = v; }
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
                return Err(anyhow!("--symmetric-pd-balance-rel must be finite and >= 1"));
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
        let decision_replay_header = match &self.symmetric_pd_decision_replay_header {
            Some(h) if h.trim().is_empty() => {
                return Err(anyhow!("--symmetric-pd-decision-replay-header must not be empty"))
            }
            Some(h) => h.trim().to_string(),
            None => crate::config::types::default_symmetric_pd_decision_replay_header(),
        };
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
        let import_cache_policy =
            self.symmetric_pd_import_cache_policy
                .unwrap_or(if self.symmetric_pd_decode_host_return {
                    SymmetricPdImportCachePolicy::Retain
                } else {
                    SymmetricPdImportCachePolicy::Auto
                });
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
            let timeout_ms = self.symmetric_pd_decode_host_return_timeout_ms.unwrap_or(10_000);
            let max_tokens = self.symmetric_pd_decode_host_return_max_tokens.unwrap_or(65_536);
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
            // `into_config` attaches it from the `--co-schedule-trade*` flags.
            cost_function: None,
        }))
    }

    /// The `--symmetric-pd-*` flags that `--symmetric-pd-p-select cost-function`
    /// needs or refuses. The TRADE pull rides on the P-leg as the pull hint;
    /// the cost function never uses the symmetric_pd hold queue (its own waiting
    /// queue is `--symmetric-pd-gwq-auto`). External role control is accepted:
    /// the candidate pool, the pull sources, the progress poll and the
    /// waiting-queue steps follow the live P-role set at each request.
    fn check_flex_cost_function(&self) -> Result<()> {
        if !self.symmetric_pd_force_split {
            return Err(anyhow!(
                "--symmetric-pd-p-select cost-function requires --symmetric-pd-force-split"
            ));
        }
        if !self.symmetric_pd_pull_hint {
            return Err(anyhow!(
                "--symmetric-pd-p-select cost-function requires --symmetric-pd-pull-hint: the TRADE pull \
                 rides on the P-leg as the pull hint"
            ));
        }
        if self.symmetric_pd_max_queue_wait_ms.is_some() || self.symmetric_pd_max_uncached_tokens.is_some() {
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
        // The co_schedule queue is not wired into symmetric_pd; its queue is
        // --symmetric-pd-gwq-auto.
        if self
            .co_schedule_global_queue
            .as_deref()
            .is_some_and(|mode| mode != "off")
        {
            return Err(anyhow!(
                "--co-schedule-global-queue is not available with --symmetric-pd-p-select \
                 cost-function: its global waiting queue is --symmetric-pd-gwq-auto"
            ));
        }
        if self.symmetric_pd_gwq_room.is_some() && !self.symmetric_pd_gwq_auto {
            return Err(anyhow!(
                "--symmetric-pd-gwq-room requires --symmetric-pd-gwq-auto (the room rule of its queue)"
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
    fn build_discovery(&self, symmetric_pd: Option<&SymmetricPdConfig>) -> Result<DiscoveryBackend> {
        let mut static_urls = self.worker_urls.clone();
        if let Some(flex) = symmetric_pd {
            if !self.service_discovery {
                let mut seen: std::collections::HashSet<String> = static_urls
                    .iter()
                    .map(|u| u.trim().trim_end_matches('/').to_string())
                    .collect();
                for w in &flex.workers {
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

/// The value of a TRADE flag that takes a number or `auto`
/// (`--co-schedule-trade-copy-lambda`, `--co-schedule-trade-rate-tokens-per-s`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NumberOrAuto {
    Number(f64),
    Auto,
}

/// The value of `--co-schedule-global-queue-depth-tokens`: a whole number of
/// tokens or `auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokensOrAuto {
    Tokens(u64),
    Auto,
}

/// Parse `auto` (any case) or a whole number of tokens; [`Cli::into_config`]
/// checks the range.
fn parse_tokens_or_auto(value: &str) -> std::result::Result<TokensOrAuto, String> {
    let value = value.trim();
    if value.eq_ignore_ascii_case("auto") {
        return Ok(TokensOrAuto::Auto);
    }
    value
        .parse::<u64>()
        .map(TokensOrAuto::Tokens)
        .map_err(|e| format!("expected a whole number of tokens or `auto`: {e}"))
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
    /// function with peer pulls, the copy price 4.6 and no cost trace. Flags
    /// that are given win, and `none` takes the flags as given.
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
        let flex = c.model.symmetric_pd.as_ref().unwrap();
        assert!(flex.placement.force_split);
        assert!(flex.kv_session_retention);
        assert_eq!(format!("{:?}", flex.placement.d_select), "RoundRobin");
        assert_eq!(format!("{:?}", flex.placement.p_select), "CacheFirst", "no KV directory");
        assert!(flex.cost_function.is_none());
        let late = &c.proxy.late_decode;
        assert_eq!(late.mode, LateDecodeMode::Choose);
        assert_eq!(late.rule, Some(LateDecodeRule::FitReq));
        assert!(late.fit_native && late.native_state_poll);

        let mut with_kv = workers.to_vec();
        with_kv.extend([
            "--kv-indexer-url",
            "http://indexer:8091",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://10.0.0.1:30000","servicer":"10.0.0.1:18100"},{"id":2,"url":"http://10.0.0.2:30000","servicer":"10.0.0.2:18100"}]"#,
        ]);
        let c = into_config_owned(with_model(&with_kv)).unwrap();
        let flex = c.model.symmetric_pd.as_ref().unwrap();
        assert_eq!(format!("{:?}", flex.placement.p_select), "CostFunction");
        assert!(flex.pull_hint);
        let cost = flex.cost_function.as_ref().unwrap();
        assert_eq!(cost.trade.copy_lambda, 4.6);
        assert!(cost.cost_trace.is_none());

        let mut given = with_kv.clone();
        given.extend(["--symmetric-pd-p-select", "cache-first", "--pd-late-decode-rule", "d1"]);
        let c = into_config_owned(with_model(&given)).unwrap();
        let flex = c.model.symmetric_pd.as_ref().unwrap();
        assert_eq!(format!("{:?}", flex.placement.p_select), "CacheFirst");
        assert_eq!(c.proxy.late_decode.rule, Some(LateDecodeRule::D1));

        let mut none = with_kv.clone();
        let at = none.iter().position(|a| *a == "paper").unwrap();
        none[at] = "none";
        let c = into_config_owned(with_model(&none)).unwrap();
        let flex = c.model.symmetric_pd.as_ref().unwrap();
        assert!(!flex.kv_session_retention && !flex.pull_hint);
        assert_eq!(c.proxy.late_decode.mode, LateDecodeMode::Off);
    }

    fn into_config_owned(args: Vec<String>) -> Result<Config> {
        let refs: Vec<&str> = args.iter().map(String::as_str).collect();
        into_config(&refs)
    }

    #[test]
    fn dynamo_cache_aware_is_a_single_cache_source_without_transfer() {
        let args = with_model(&["--worker-urls", "http://p:1", "--policy", "cache_aware_zmq",
            "--dynamo-cache-aware", "--kv-indexer-url", "http://index:2",
            "--kv-worker-map", r#"[{"id":1,"url":"http://p:1","servicer":"p:3"}]"#,
            "--pd-decode-policy", "round-robin"]);
        let cfg = into_config_owned(args.clone()).unwrap();
        assert!(cfg.model.cache_aware.unwrap().dynamo.is_some());
        assert!(cfg.model.kv_hint.is_none());
        assert!(cfg.model.co_schedule.is_none());
        assert!(cfg.model.symmetric_pd.is_none());
        for extra in [vec!["--inject-kv-pull-hint"], vec!["--kv-indexer-endpoint", "http://grpc:4"]] {
            let mut bad = args.clone(); bad.extend(extra.into_iter().map(str::to_owned));
            assert!(into_config_owned(bad).is_err());
        }
        assert!(into_config_owned(with_model(&["--worker-urls", "http://p:1",
            "--co-schedule-on-demand-load-refresh"])).is_err());
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
        let err = into_config_owned(with_model(&args)).unwrap_err().to_string();
        assert!(err.contains("po2 is not available with --policy symmetric_pd"), "{err}");

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
        assert!(on.model.symmetric_pd.as_ref().unwrap().external_role_control);

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
        let err = into_config_owned(with_model(&args)).unwrap_err().to_string();
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
        let flex = cfg.model.symmetric_pd.unwrap();
        assert_eq!(
            flex.gateway_cache_aware,
            Some(GatewayCacheAwareConfig::default())
        );
        assert_eq!(
            flex.placement.p_select,
            crate::symmetric_pd::placement::PSelect::CacheFirst
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
        let err = into_config_owned(with_model(&args)).unwrap_err().to_string();
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
        assert!(err.contains("require --policy gateway_cache_aware"), "{err}");
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
    fn co_schedule_requires_indexer_flags() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "co_schedule",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("--policy co_schedule requires --kv-indexer-url"),
            "got: {err}"
        );
    }

    #[test]
    fn co_schedule_builds_defaults_and_implies_kv_hint() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "co_schedule",
            "--kv-indexer-url",
            "http://indexer:8090",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://x:30000","servicer":"x:18100"}]"#,
        ]))
        .unwrap();
        let cs = c.model.co_schedule.expect("co_schedule config built");
        assert!(!cs.prefer_gpu_hit);
        assert_eq!(cs.locality_margin_tokens, None);
        assert_eq!(cs.max_uncached_tokens, 32768);
        assert_eq!(cs.max_queue_wait_ms, 30_000);
        assert_eq!(cs.load_poll_ms, 1000);
        assert_eq!(cs.tick_ms, 500);
        assert_eq!(cs.poll_on_hold_ms, 100);
        assert_eq!(cs.imbalance_spread_tokens, 131_072);
        assert_eq!(cs.pull_scope, KvPullScope::Any);
        // The kv-hint config is implied WITHOUT --inject-kv-pull-hint.
        assert!(c.model.kv_hint.is_some());
    }

    #[test]
    fn rejects_co_schedule_knobs_without_the_policy() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--co-schedule-max-uncached-tokens",
            "4096",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("require --policy co_schedule"), "got: {err}");
    }

    #[test]
    fn gpu_hit_preference_requires_capped_co_schedule() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--co-schedule-prefer-gpu-hit",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("require --policy co_schedule"));
        let flags = [
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "co_schedule",
            "--kv-indexer-url",
            "http://indexer:8090",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://x:30000","servicer":"x:18100"}]"#,
            "--co-schedule-prefer-gpu-hit",
        ];
        let c = into_config_owned(with_model(&flags)).unwrap();
        assert!(c.model.co_schedule.unwrap().prefer_gpu_hit);
        let mut cost_flags = flags.to_vec();
        cost_flags.extend([
            "--co-schedule-cost-balance",
            "--co-schedule-cost-trace",
            "/tmp/gpu-tie-test.jsonl",
        ]);
        let err = into_config_owned(with_model(&cost_flags))
            .unwrap_err()
            .to_string();
        assert!(err.contains("requires capped co-schedule mode"));
    }

    #[test]
    fn uncached_balance_rejects_gates_and_uses_progress_poll_defaults() {
        let flags = [
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "co_schedule",
            "--kv-indexer-url",
            "http://indexer:8090",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://x:30000","servicer":"x:18100"}]"#,
            "--co-schedule-uncached-balance",
            "--co-schedule-cost-trace",
            "/tmp/uncached-test.jsonl",
        ];
        let config = into_config_owned(with_model(&flags))
            .unwrap()
            .model
            .co_schedule
            .unwrap();
        assert!(config.uncached_balance);
        assert_eq!(config.owner_threshold_tokens, 12288);
        assert_eq!(config.load_poll_ms, 100);
        assert_eq!(config.restore_cost_divisor, None);
        let mut restore = flags.to_vec();
        restore.extend(["--co-schedule-restore-cost-divisor", "8"]);
        assert_eq!(
            into_config_owned(with_model(&restore))
                .unwrap()
                .model
                .co_schedule
                .unwrap()
                .restore_cost_divisor,
            Some(8)
        );
        *restore.last_mut().unwrap() = "0";
        assert!(into_config_owned(with_model(&restore)).is_err());
        let orphan: Vec<_> = restore
            .into_iter()
            .filter(|s| *s != "--co-schedule-uncached-balance")
            .collect();
        assert!(into_config_owned(with_model(&orphan)).is_err());
        for extra in [
            vec!["--co-schedule-cost-balance"],
            vec!["--co-schedule-locality-margin-tokens", "100"],
            vec!["--co-schedule-max-uncached-tokens", "16384"],
            vec!["--co-schedule-max-queue-wait-ms", "100"],
            vec!["--co-schedule-tick-ms", "100"],
            vec!["--co-schedule-poll-on-hold-ms", "100"],
            vec!["--co-schedule-imbalance-spread-tokens", "100"],
        ] {
            let mut conflict = flags.to_vec();
            conflict.extend(extra);
            assert!(into_config_owned(with_model(&conflict)).is_err());
        }
    }

    #[test]
    fn trade_is_opt_in_with_defaults_and_validated() {
        let flags = [
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "co_schedule",
            "--kv-indexer-url",
            "http://indexer:8090",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://x:30000","servicer":"x:18100"}]"#,
            "--kv-pull-hint-min-incremental-tokens",
            "512",
            "--co-schedule-uncached-balance",
            "--co-schedule-cost-trace",
            "/tmp/trade-test.jsonl",
        ];
        let off = into_config_owned(with_model(&flags)).unwrap();
        assert!(off.model.co_schedule.unwrap().trade.is_none());

        let mut on = flags.to_vec();
        on.push("--co-schedule-trade");
        let trade = into_config_owned(with_model(&on))
            .unwrap()
            .model
            .co_schedule
            .unwrap()
            .trade
            .unwrap();
        assert_eq!(trade, crate::scheduler::co_schedule::TradeParams::default());
        assert_eq!(trade.rate_tokens_per_s, 13_500.0);
        assert_eq!(trade.pull_fixed_ms, 450.0);
        assert_eq!(trade.pull_ms_per_100k, 100.0);
        assert_eq!(trade.copy_lambda, 4.6);
        assert!(!trade.exclude_unlaunched, "the backlog is the full ledger by default");

        let mut tuned = on.clone();
        tuned.extend([
            "--co-schedule-trade-rate-tokens-per-s",
            "20000",
            "--co-schedule-trade-pull-fixed-ms",
            "300",
            "--co-schedule-trade-pull-ms-per-100k",
            "50",
            "--co-schedule-trade-copy-lambda",
            "0",
            "--co-schedule-trade-exclude-unlaunched",
        ]);
        let trade = into_config_owned(with_model(&tuned))
            .unwrap()
            .model
            .co_schedule
            .unwrap()
            .trade
            .unwrap();
        assert_eq!(trade.rate_tokens_per_s, 20_000.0);
        assert_eq!(trade.pull_fixed_ms, 300.0);
        assert_eq!(trade.pull_ms_per_100k, 50.0);
        assert_eq!(trade.copy_lambda, 0.0);
        assert!(trade.exclude_unlaunched);
        // The old flag is the default and still parses.
        let mut compat = on.clone();
        compat.push("--co-schedule-trade-count-inflight");
        let trade = into_config_owned(with_model(&compat))
            .unwrap()
            .model
            .co_schedule
            .unwrap()
            .trade
            .unwrap();
        assert!(!trade.exclude_unlaunched);
        let mut both = compat.clone();
        both.push("--co-schedule-trade-exclude-unlaunched");
        assert!(into_config_owned(with_model(&both)).is_err());

        // Tuning flags need --co-schedule-trade.
        for extra in [
            vec!["--co-schedule-trade-rate-tokens-per-s", "20000"],
            vec!["--co-schedule-trade-copy-lambda", "1"],
            vec!["--co-schedule-trade-count-inflight"],
            vec!["--co-schedule-trade-exclude-unlaunched"],
        ] {
            let mut orphan = flags.to_vec();
            orphan.extend(extra);
            assert!(into_config_owned(with_model(&orphan)).is_err());
        }
        // Invalid values and replaced uncached-balance knobs.
        for extra in [
            vec!["--co-schedule-trade-rate-tokens-per-s", "0"],
            vec!["--co-schedule-trade-rate-tokens-per-s", "inf"],
            vec!["--co-schedule-trade-pull-fixed-ms=-1"],
            vec!["--co-schedule-trade-pull-ms-per-100k", "NaN"],
            vec!["--co-schedule-trade-copy-lambda=-0.5"],
            vec!["--co-schedule-owner-threshold-tokens", "4096"],
            vec!["--co-schedule-restore-cost-divisor", "8"],
        ] {
            let mut bad = on.clone();
            bad.extend(extra);
            assert!(into_config_owned(with_model(&bad)).is_err(), "{bad:?}");
        }
        // Uncached balance is required; so is the policy.
        let no_balance: Vec<_> = on
            .iter()
            .copied()
            .filter(|s| *s != "--co-schedule-uncached-balance" && *s != "--co-schedule-cost-trace"
                && *s != "/tmp/trade-test.jsonl")
            .collect();
        let err = into_config_owned(with_model(&no_balance)).unwrap_err().to_string();
        assert!(err.contains("requires --co-schedule-uncached-balance"), "{err}");
        let err = into_config_owned(with_model(&["--worker-urls", "http://x:30000",
            "--co-schedule-trade"])).unwrap_err().to_string();
        assert!(err.contains("require --policy co_schedule"), "{err}");
    }

    /// TRADE never pulls below 512 tokens: an absent gate becomes 512, a
    /// smaller explicit gate and the local-prefix fallback are rejected.
    /// Without TRADE the gate keeps its default of 0.
    #[test]
    fn trade_pull_floor_is_at_least_512() {
        let base = [
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "co_schedule",
            "--kv-indexer-url",
            "http://indexer:8090",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://x:30000","servicer":"x:18100"}]"#,
            "--co-schedule-uncached-balance",
            "--co-schedule-cost-trace",
            "/tmp/trade-floor-test.jsonl",
        ];
        let gate = |extra: &[&str]| {
            let args: Vec<String> = base
                .iter()
                .map(|s| s.to_string())
                .chain(extra.iter().map(|s| s.to_string()))
                .collect();
            let refs: Vec<&str> = args.iter().map(String::as_str).collect();
            into_config_owned(with_model(&refs))
                .map(|c| c.model.kv_hint.unwrap().min_incremental_tokens)
        };
        assert_eq!(gate(&[]).unwrap(), 0, "no TRADE: gate off by default");
        assert_eq!(gate(&["--co-schedule-trade"]).unwrap(), 512);
        assert_eq!(
            gate(&["--co-schedule-trade", "--kv-pull-hint-min-incremental-tokens", "512"]).unwrap(),
            512
        );
        assert_eq!(
            gate(&["--co-schedule-trade", "--kv-pull-hint-min-incremental-tokens", "2048"])
                .unwrap(),
            2048
        );
        for low in ["0", "1", "511"] {
            let err = gate(&["--co-schedule-trade", "--kv-pull-hint-min-incremental-tokens", low])
                .unwrap_err()
                .to_string();
            assert!(err.contains("at least 512"), "{low}: {err}");
        }
        let err = gate(&["--co-schedule-trade", "--kv-pull-hint-local-prefix-fallback"])
            .unwrap_err()
            .to_string();
        assert!(err.contains("local-prefix-fallback"), "{err}");
        // Without TRADE both stay allowed.
        assert_eq!(gate(&["--kv-pull-hint-min-incremental-tokens", "1"]).unwrap(), 1);
        assert!(gate(&[
            "--kv-pull-hint-min-incremental-tokens",
            "512",
            "--kv-pull-hint-local-prefix-fallback",
        ])
        .is_ok());
    }

    /// The later TRADE options: wait eft (with its model and age limit),
    /// dup charge all and the noise margin. Defaults keep the original behavior.
    #[test]
    fn trade_wait_dup_charge_and_noise_margin_flags() {
        use crate::scheduler::co_schedule::{TradeDupCharge, TradeWait};
        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("model.json");
        std::fs::write(
            &model,
            r#"{"models":{"augmented":{"launch_interval":{"global":
                {"coefficients":[0.05,1e-5,0.0,0.004,1e-7,0.0]}}}},"eft":{"tau_s":0.02}}"#,
        )
        .unwrap();
        let model_path = model.to_str().unwrap().to_string();
        let base = [
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "co_schedule",
            "--kv-indexer-url",
            "http://indexer:8090",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://x:30000","servicer":"x:18100"}]"#,
            "--co-schedule-uncached-balance",
            "--co-schedule-cost-trace",
            "/tmp/trade-eft-test.jsonl",
        ];
        let build = |extra: &[&str]| {
            let args: Vec<String> = base
                .iter()
                .map(|s| s.to_string())
                .chain(extra.iter().map(|s| s.to_string()))
                .collect();
            let refs: Vec<&str> = args.iter().map(String::as_str).collect();
            into_config_owned(with_model(&refs)).map(|c| c.model.co_schedule.unwrap())
        };
        let cfg = build(&["--co-schedule-trade"]).unwrap();
        let trade = cfg.trade.unwrap();
        assert_eq!(trade.wait, TradeWait::Ledger);
        assert_eq!(trade.dup_charge, TradeDupCharge::Pulled);
        assert_eq!(trade.margin_noise_k, 0.0);
        assert_eq!(trade.eft_max_view_age_s, 1.5);
        assert!(trade.is_original());
        assert!(cfg.trade_eft_model.is_none());
        let explicit = build(&[
            "--co-schedule-trade",
            "--co-schedule-trade-wait",
            "ledger",
            "--co-schedule-trade-dup-charge",
            "pulled",
            "--co-schedule-trade-margin-noise",
            "0",
        ])
        .unwrap();
        assert!(explicit.trade.unwrap().is_original());

        let cfg = build(&[
            "--co-schedule-trade",
            "--co-schedule-trade-wait",
            "eft",
            "--co-schedule-trade-eft-model",
            &model_path,
            "--co-schedule-trade-eft-max-view-age-ms",
            "800",
            "--co-schedule-trade-dup-charge",
            "all",
            "--co-schedule-trade-margin-noise",
            "0.5",
        ])
        .unwrap();
        let trade = cfg.trade.unwrap();
        assert_eq!(trade.wait, TradeWait::Eft);
        assert_eq!(trade.dup_charge, TradeDupCharge::All);
        assert_eq!(trade.margin_noise_k, 0.5);
        assert_eq!(trade.eft_max_view_age_s, 0.8);
        let model = cfg.trade_eft_model.unwrap();
        assert_eq!(model.b0, 0.05);
        assert_eq!(model.sha256.len(), 64);
        assert_eq!(model.sources["sha256"], "file");

        for (extra, message) in [
            (vec!["--co-schedule-trade-wait", "eft"], "requires --co-schedule-trade-eft-model"),
            (
                vec!["--co-schedule-trade-eft-model", model_path.as_str()],
                "need --co-schedule-trade-wait eft",
            ),
            (
                vec!["--co-schedule-trade-eft-max-view-age-ms", "800"],
                "need --co-schedule-trade-wait eft",
            ),
            (
                vec![
                    "--co-schedule-trade-wait",
                    "eft",
                    "--co-schedule-trade-eft-model",
                    "/nonexistent/model.json",
                ],
                "--co-schedule-trade-eft-model",
            ),
            (
                vec![
                    "--co-schedule-trade-wait",
                    "eft",
                    "--co-schedule-trade-eft-model",
                    model_path.as_str(),
                    "--co-schedule-trade-eft-max-view-age-ms",
                    "0",
                ],
                "max-view-age-ms",
            ),
            (vec!["--co-schedule-trade-margin-noise=-1"], "margin-noise"),
            (vec!["--co-schedule-trade-margin-noise", "inf"], "margin-noise"),
        ] {
            let mut args = vec!["--co-schedule-trade"];
            args.extend(extra);
            let err = build(&args).unwrap_err().to_string();
            assert!(err.contains(message), "{args:?}: {err}");
        }
        // Unknown values and the flags without --co-schedule-trade.
        for extra in [
            vec!["--co-schedule-trade", "--co-schedule-trade-wait", "simulated"],
            vec!["--co-schedule-trade", "--co-schedule-trade-dup-charge", "none"],
            vec!["--co-schedule-trade-wait", "eft"],
            vec!["--co-schedule-trade-dup-charge", "all"],
            vec!["--co-schedule-trade-margin-noise", "1"],
        ] {
            assert!(build(&extra).is_err(), "{extra:?}");
        }
    }

    /// TRADE wait e3: the bound table (bound-params.json at its shipped
    /// level, or bound-tables.json with an explicit level), the gate m and
    /// g_min, the rq margin, the shadow path and the view-age limit. The e3
    /// and gate flags need wait e3; wait e3 refuses the EFT model and the
    /// noise margin.
    #[test]
    fn trade_wait_e3_flags() {
        use crate::scheduler::co_schedule::TradeWait;
        let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/e3");
        let params_path = format!("{fixtures}/bound-params.json");
        let tables_path = format!("{fixtures}/bound-tables.json");
        let base = [
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "co_schedule",
            "--kv-indexer-url",
            "http://indexer:8090",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://x:30000","servicer":"x:18100"}]"#,
            "--co-schedule-uncached-balance",
            "--co-schedule-cost-trace",
            "/tmp/trade-e3-test.jsonl",
        ];
        let build = |extra: &[&str]| {
            let args: Vec<String> = base
                .iter()
                .map(|s| s.to_string())
                .chain(extra.iter().map(|s| s.to_string()))
                .collect();
            let refs: Vec<&str> = args.iter().map(String::as_str).collect();
            into_config_owned(with_model(&refs)).map(|c| c.model.co_schedule.unwrap())
        };
        let cfg = build(&[
            "--co-schedule-trade",
            "--co-schedule-trade-wait",
            "e3",
            "--co-schedule-trade-e3-bound-table",
            &params_path,
        ])
        .unwrap();
        let trade = cfg.trade.unwrap();
        assert_eq!(trade.wait, TradeWait::E3);
        assert_eq!((trade.gate_m, trade.gate_gmin_s), (1.0, 0.0));
        assert_eq!(trade.eft_max_view_age_s, 1.5);
        assert!(!trade.is_original());
        assert!(cfg.trade_eft_model.is_none());
        let e3 = cfg.trade_e3.unwrap();
        assert_eq!((e3.mode.as_str(), e3.level.as_str()), ("snap+prod", "0.8"));
        assert_eq!(e3.file_level.as_deref(), Some("0.8"));
        assert_eq!(e3.rq_margin_s, 0.0);
        assert!(e3.shadow_path.is_none());
        assert_eq!(e3.bound_sha256.len(), 64);

        let cfg = build(&[
            "--co-schedule-trade",
            "--co-schedule-trade-wait",
            "e3",
            "--co-schedule-trade-e3-bound-table",
            &tables_path,
            "--co-schedule-trade-e3-bound-mode",
            "snap+prodL",
            "--co-schedule-trade-e3-bound-level",
            "0.9",
            "--co-schedule-trade-gate-m",
            "1.5",
            "--co-schedule-trade-gate-gmin-ms",
            "100",
            "--co-schedule-trade-e3-rq-margin-ms",
            "20",
            "--co-schedule-trade-e3-shadow",
            "/tmp/e3-shadow-test.jsonl",
            "--co-schedule-trade-eft-max-view-age-ms",
            "800",
            "--co-schedule-trade-hol-weight",
            "1",
        ])
        .unwrap();
        let trade = cfg.trade.unwrap();
        assert_eq!(
            (
                trade.gate_m,
                trade.gate_gmin_s,
                trade.eft_max_view_age_s,
                trade.hol_weight
            ),
            (1.5, 0.1, 0.8, 1.0)
        );
        let e3 = cfg.trade_e3.unwrap();
        assert_eq!((e3.mode.as_str(), e3.level.as_str()), ("snap+prodL", "0.9"));
        assert_eq!(e3.bound_schema, "bound-tables");
        assert_eq!(e3.file_level, None);
        assert!((e3.rq_margin_s - 0.02).abs() < 1e-12);
        assert_eq!(
            e3.shadow_path.as_deref(),
            Some(std::path::Path::new("/tmp/e3-shadow-test.jsonl"))
        );

        let e3_args = |extra: &[&'static str]| {
            let mut v = vec![
                "--co-schedule-trade-wait",
                "e3",
                "--co-schedule-trade-e3-bound-table",
                params_path.as_str(),
            ];
            v.extend_from_slice(extra);
            v
        };
        for (extra, message) in [
            (
                vec!["--co-schedule-trade-wait", "e3"],
                "requires --co-schedule-trade-e3-bound-table",
            ),
            (
                vec![
                    "--co-schedule-trade-wait",
                    "e3",
                    "--co-schedule-trade-e3-bound-table",
                    tables_path.as_str(),
                ],
                "--co-schedule-trade-e3-bound-level",
            ),
            (
                vec![
                    "--co-schedule-trade-wait",
                    "e3",
                    "--co-schedule-trade-e3-bound-table",
                    "/nonexistent/bounds.json",
                ],
                "--co-schedule-trade-e3-bound-table",
            ),
            (
                e3_args(&["--co-schedule-trade-e3-bound-mode", "rfng"]),
                "no deploy mode rfng",
            ),
            (
                e3_args(&["--co-schedule-trade-e3-bound-mode", "pushL+prodL"]),
                "does not match the router's poll forecasts",
            ),
            (
                e3_args(&["--co-schedule-trade-margin-noise", "1"]),
                "margin-noise must be 0",
            ),
            (
                e3_args(&["--co-schedule-trade-gate-m=-1"]),
                "--co-schedule-trade-gate-m",
            ),
            (
                e3_args(&["--co-schedule-trade-gate-gmin-ms", "inf"]),
                "--co-schedule-trade-gate-gmin-ms",
            ),
            (
                e3_args(&["--co-schedule-trade-e3-rq-margin-ms=-5"]),
                "rq-margin-ms",
            ),
            (
                e3_args(&["--co-schedule-trade-eft-model", "/tmp/model.json"]),
                "need --co-schedule-trade-wait eft",
            ),
            (
                vec!["--co-schedule-trade-e3-bound-table", params_path.as_str()],
                "need --co-schedule-trade-wait e3",
            ),
            (
                vec!["--co-schedule-trade-gate-m", "2"],
                "need --co-schedule-trade-wait e3",
            ),
            (
                vec!["--co-schedule-trade-e3-shadow", "/tmp/x.jsonl"],
                "need --co-schedule-trade-wait e3",
            ),
        ] {
            let mut args = vec!["--co-schedule-trade"];
            args.extend(extra);
            let err = build(&args).unwrap_err().to_string();
            assert!(err.contains(message), "{args:?}: {err}");
        }
        // Clap refuses an unknown level, and the flags without the TRADE flag.
        let mut args = vec!["--co-schedule-trade"];
        args.extend(e3_args(&["--co-schedule-trade-e3-bound-level", "0.7"]));
        assert!(build(&args).is_err());
        assert!(build(&["--co-schedule-trade-wait", "e3"]).is_err());
        assert!(build(&["--co-schedule-trade-gate-m", "1"]).is_err());
    }

    /// TRADE wait e3 with the state push: the port and the push bound table
    /// (default: the poll mode's; the level rule of the poll table). Off
    /// without the port. The push flags need wait e3, the push bound mode
    /// needs the port, and port 0 or an unknown mode are refused.
    #[test]
    fn trade_wait_e3_push_flags() {
        let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/e3");
        let params_path = format!("{fixtures}/bound-params.json");
        let base = [
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "co_schedule",
            "--kv-indexer-url",
            "http://indexer:8090",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://x:30000","servicer":"x:18100"}]"#,
            "--co-schedule-uncached-balance",
            "--co-schedule-cost-trace",
            "/tmp/trade-e3-push-test.jsonl",
        ];
        let build = |extra: &[&str]| {
            let args: Vec<String> = base
                .iter()
                .map(|s| s.to_string())
                .chain(extra.iter().map(|s| s.to_string()))
                .collect();
            let refs: Vec<&str> = args.iter().map(String::as_str).collect();
            into_config_owned(with_model(&refs)).map(|c| c.model.co_schedule.unwrap())
        };
        let e3 = |extra: &[&str]| {
            let mut args = vec![
                "--co-schedule-trade",
                "--co-schedule-trade-wait",
                "e3",
                "--co-schedule-trade-e3-bound-table",
                params_path.as_str(),
            ];
            args.extend_from_slice(extra);
            build(&args)
        };
        assert!(e3(&[]).unwrap().trade_e3.unwrap().push.is_none());
        let port = ["--co-schedule-trade-e3-push-port", "5600"];
        let cfg = e3(&port).unwrap().trade_e3.unwrap();
        let push = cfg.push.as_ref().unwrap();
        assert_eq!(
            (push.port, push.mode.as_str(), push.level.as_str()),
            (5600, "snap+prod", "0.8")
        );
        assert_eq!(push.table, cfg.table);
        let mode = ["--co-schedule-trade-e3-push-bound-mode", "pushL+prodL"];
        let cfg = e3(&[port.as_slice(), mode.as_slice()].concat())
            .unwrap()
            .trade_e3
            .unwrap();
        let push = cfg.push.as_ref().unwrap();
        assert_eq!(
            (push.mode.as_str(), push.level.as_str()),
            ("pushL+prodL", "0.8")
        );
        assert_eq!(push.file_level.as_deref(), Some("0.8"));
        assert_ne!(push.table, cfg.table);
        assert_eq!(
            (cfg.mode.as_str(), cfg.level.as_str()),
            ("snap+prod", "0.8")
        );
        let level = ["--co-schedule-trade-e3-bound-level", "0.9"];
        let cfg = e3(&[port.as_slice(), mode.as_slice(), level.as_slice()].concat())
            .unwrap()
            .trade_e3
            .unwrap();
        let push = cfg.push.as_ref().unwrap();
        assert_eq!((cfg.level.as_str(), push.level.as_str()), ("0.9", "0.9"));
        let unknown = ["--co-schedule-trade-e3-push-bound-mode", "rfng"];
        let gpu = ["--co-schedule-trade-e3-push-bound-mode", "pushL+gpu+prod"];
        for (extra, message) in [
            (
                vec!["--co-schedule-trade-e3-push-port", "0"],
                "push port must be 1 to 65535",
            ),
            (mode.to_vec(), "needs --co-schedule-trade-e3-push-port"),
            (
                [port.as_slice(), unknown.as_slice()].concat(),
                "no deploy mode rfng",
            ),
            (
                [port.as_slice(), gpu.as_slice()].concat(),
                "does not match the router's pushed-state forecasts",
            ),
        ] {
            let err = e3(&extra).unwrap_err().to_string();
            assert!(err.contains(message), "{extra:?}: {err}");
        }
        let err = build(&["--co-schedule-trade", port[0], port[1]])
            .unwrap_err()
            .to_string();
        assert!(err.contains("need --co-schedule-trade-wait e3"), "{err}");
        // Clap: not a TCP port.
        assert!(e3(&["--co-schedule-trade-e3-push-port", "70000"]).is_err());
    }

    /// The TRADE head-of-line term: weight mu (default 0 = off, the first
    /// version) and the rate window W (default 60 s). Both need
    /// --co-schedule-trade; mu must be finite and at least 0, W finite and
    /// above 0.
    #[test]
    fn trade_hol_weight_and_rate_window_flags() {
        let base = [
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "co_schedule",
            "--kv-indexer-url",
            "http://indexer:8090",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://x:30000","servicer":"x:18100"}]"#,
            "--co-schedule-uncached-balance",
            "--co-schedule-cost-trace",
            "/tmp/trade-hol-test.jsonl",
        ];
        let build = |extra: &[&str]| {
            let args: Vec<String> = base
                .iter()
                .map(|s| s.to_string())
                .chain(extra.iter().map(|s| s.to_string()))
                .collect();
            let refs: Vec<&str> = args.iter().map(String::as_str).collect();
            into_config_owned(with_model(&refs)).map(|c| c.model.co_schedule.unwrap())
        };
        let trade = build(&["--co-schedule-trade"]).unwrap().trade.unwrap();
        assert_eq!(trade.hol_weight, 0.0);
        assert_eq!(trade.hol_rate_window_s, 60.0);
        assert!(trade.is_original());

        let trade = build(&[
            "--co-schedule-trade",
            "--co-schedule-trade-hol-weight",
            "1.5",
            "--co-schedule-trade-hol-rate-window-secs",
            "30",
        ])
        .unwrap()
        .trade
        .unwrap();
        assert_eq!(trade.hol_weight, 1.5);
        assert_eq!(trade.hol_rate_window_s, 30.0);
        assert!(!trade.is_original());
        // The other options keep their defaults.
        assert_eq!(trade.margin_noise_k, 0.0);
        assert_eq!(trade.copy_lambda, 4.6);
        // mu = 0 is the original behavior; the window alone changes nothing.
        let trade = build(&[
            "--co-schedule-trade",
            "--co-schedule-trade-hol-weight",
            "0",
            "--co-schedule-trade-hol-rate-window-secs",
            "0.5",
        ])
        .unwrap()
        .trade
        .unwrap();
        assert_eq!(trade.hol_rate_window_s, 0.5);
        assert!(trade.is_original());

        for (extra, message) in [
            (vec!["--co-schedule-trade-hol-weight=-1"], "--co-schedule-trade-hol-weight"),
            (vec!["--co-schedule-trade-hol-weight", "NaN"], "--co-schedule-trade-hol-weight"),
            (vec!["--co-schedule-trade-hol-weight", "inf"], "--co-schedule-trade-hol-weight"),
            (
                vec!["--co-schedule-trade-hol-rate-window-secs", "0"],
                "--co-schedule-trade-hol-rate-window-secs",
            ),
            (
                vec!["--co-schedule-trade-hol-rate-window-secs=-5"],
                "--co-schedule-trade-hol-rate-window-secs",
            ),
            (
                vec!["--co-schedule-trade-hol-rate-window-secs", "NaN"],
                "--co-schedule-trade-hol-rate-window-secs",
            ),
            (
                vec!["--co-schedule-trade-hol-rate-window-secs", "inf"],
                "--co-schedule-trade-hol-rate-window-secs",
            ),
        ] {
            let mut args = vec!["--co-schedule-trade"];
            args.extend(extra);
            let err = build(&args).unwrap_err().to_string();
            assert!(err.contains(message), "{args:?}: {err}");
            assert!(err.contains("must be finite"), "{args:?}: {err}");
        }
        // The flags without --co-schedule-trade.
        for extra in [
            vec!["--co-schedule-trade-hol-weight", "1"],
            vec!["--co-schedule-trade-hol-rate-window-secs", "30"],
        ] {
            assert!(build(&extra).is_err(), "{extra:?}");
        }
    }

    /// The TRADE copy price and rate `auto`: `auto` (any case) or a number
    /// for both flags; the auto defaults (q sat, mu 1, min 4.6, max 1000,
    /// fallback 4.6, tau 120 s, idle 300 s); the fallback lambda in
    /// `copy_lambda`; validation (finite, at least 0, min <= max, tau and
    /// idle above 0; the auto-only flags need an auto mode).
    #[test]
    fn trade_auto_copy_lambda_and_rate_flags() {
        use crate::scheduler::co_schedule::{TradeAutoQ, TradeLambdaMode, TradeRateMode};
        let base = [
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "co_schedule",
            "--kv-indexer-url",
            "http://indexer:8090",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://x:30000","servicer":"x:18100"}]"#,
            "--co-schedule-uncached-balance",
            "--co-schedule-cost-trace",
            "/tmp/trade-auto-test.jsonl",
        ];
        let build = |extra: &[&str]| {
            let args: Vec<String> = base
                .iter()
                .map(|s| s.to_string())
                .chain(extra.iter().map(|s| s.to_string()))
                .collect();
            let refs: Vec<&str> = args.iter().map(String::as_str).collect();
            into_config_owned(with_model(&refs)).map(|c| c.model.co_schedule.unwrap())
        };
        assert_eq!(parse_number_or_auto("auto"), Ok(NumberOrAuto::Auto));
        assert_eq!(parse_number_or_auto(" AUTO "), Ok(NumberOrAuto::Auto));
        assert_eq!(parse_number_or_auto("18.4"), Ok(NumberOrAuto::Number(18.4)));
        assert!(parse_number_or_auto("autox").is_err());
        assert!(parse_number_or_auto("").is_err());

        // Without the flags, and with numbers: fixed, as before.
        let trade = build(&["--co-schedule-trade"]).unwrap().trade.unwrap();
        assert_eq!(trade.copy_lambda_mode, TradeLambdaMode::Fixed);
        assert_eq!(trade.rate_mode, TradeRateMode::Fixed);
        assert!(!trade.is_auto() && trade.is_original());
        let fixed = build(&[
            "--co-schedule-trade",
            "--co-schedule-trade-copy-lambda",
            "18.4",
        ])
        .unwrap()
        .trade
        .unwrap();
        assert_eq!(fixed.copy_lambda, 18.4);
        assert_eq!(fixed.copy_lambda_mode, TradeLambdaMode::Fixed);
        assert!(fixed.is_original());
        let fixed = build(&[
            "--co-schedule-trade",
            "--co-schedule-trade-rate-tokens-per-s",
            "20000",
        ])
        .unwrap()
        .trade
        .unwrap();
        assert_eq!(
            (fixed.rate_tokens_per_s, fixed.rate_mode),
            (20_000.0, TradeRateMode::Fixed)
        );

        // lambda auto with the defaults: the fallback is the fixed default.
        let trade = build(&[
            "--co-schedule-trade",
            "--co-schedule-trade-copy-lambda",
            "auto",
        ])
        .unwrap()
        .trade
        .unwrap();
        assert_eq!(
            trade.copy_lambda_mode,
            TradeLambdaMode::Auto {
                q: TradeAutoQ::Sat,
                mu: 1.0,
                min: 4.6,
                max: 1000.0,
            }
        );
        assert_eq!(trade.copy_lambda, 4.6);
        assert_eq!(
            (trade.rate_mode, trade.rate_tokens_per_s),
            (TradeRateMode::Fixed, 13_500.0)
        );
        assert_eq!(
            (trade.auto_tau_s, trade.auto_session_idle_s),
            (120.0, 300.0)
        );
        assert!(trade.is_auto() && !trade.is_original());

        // Every auto flag, and rate auto (its fallback is the default R).
        let trade = build(&[
            "--co-schedule-trade",
            "--co-schedule-trade-copy-lambda",
            "AUTO",
            "--co-schedule-trade-copy-lambda-q",
            "inflight",
            "--co-schedule-trade-copy-lambda-mu",
            "0.5",
            "--co-schedule-trade-copy-lambda-min",
            "2",
            "--co-schedule-trade-copy-lambda-max",
            "50",
            "--co-schedule-trade-copy-lambda-fallback",
            "18.4",
            "--co-schedule-trade-auto-tau-secs",
            "60",
            "--co-schedule-trade-auto-session-idle-secs",
            "90",
            "--co-schedule-trade-rate-tokens-per-s",
            "auto",
        ])
        .unwrap()
        .trade
        .unwrap();
        assert_eq!(
            trade.copy_lambda_mode,
            TradeLambdaMode::Auto {
                q: TradeAutoQ::Inflight,
                mu: 0.5,
                min: 2.0,
                max: 50.0,
            }
        );
        assert_eq!(trade.copy_lambda, 18.4);
        assert_eq!(
            (trade.rate_mode, trade.rate_tokens_per_s),
            (TradeRateMode::Auto, 13_500.0)
        );
        assert_eq!((trade.auto_tau_s, trade.auto_session_idle_s), (60.0, 90.0));
        // min = max is allowed.
        assert!(build(&[
            "--co-schedule-trade",
            "--co-schedule-trade-copy-lambda",
            "auto",
            "--co-schedule-trade-copy-lambda-min",
            "20",
            "--co-schedule-trade-copy-lambda-max",
            "20",
        ])
        .is_ok());
        // Rate auto alone: lambda stays fixed; tau is accepted.
        let trade = build(&[
            "--co-schedule-trade",
            "--co-schedule-trade-rate-tokens-per-s",
            "auto",
            "--co-schedule-trade-auto-tau-secs",
            "30",
        ])
        .unwrap()
        .trade
        .unwrap();
        assert_eq!(trade.copy_lambda_mode, TradeLambdaMode::Fixed);
        assert_eq!(
            (trade.rate_mode, trade.auto_tau_s),
            (TradeRateMode::Auto, 30.0)
        );
        assert!(trade.is_auto() && !trade.is_original());

        // Invalid values, each with lambda auto.
        for (extra, message) in [
            (
                vec!["--co-schedule-trade-copy-lambda-mu=-1"],
                "--co-schedule-trade-copy-lambda-mu must be finite and at least 0",
            ),
            (
                vec!["--co-schedule-trade-copy-lambda-mu", "NaN"],
                "--co-schedule-trade-copy-lambda-mu must be finite",
            ),
            (
                vec!["--co-schedule-trade-copy-lambda-min=-0.1"],
                "--co-schedule-trade-copy-lambda-min must be finite",
            ),
            (
                vec!["--co-schedule-trade-copy-lambda-max", "inf"],
                "--co-schedule-trade-copy-lambda-max must be finite",
            ),
            (
                vec!["--co-schedule-trade-copy-lambda-fallback=-2"],
                "--co-schedule-trade-copy-lambda-fallback must be finite",
            ),
            (
                vec![
                    "--co-schedule-trade-copy-lambda-min",
                    "50",
                    "--co-schedule-trade-copy-lambda-max",
                    "10",
                ],
                "must not be above",
            ),
            (
                vec!["--co-schedule-trade-auto-tau-secs", "0"],
                "--co-schedule-trade-auto-tau-secs must be finite and greater than 0",
            ),
            (
                vec!["--co-schedule-trade-auto-tau-secs", "inf"],
                "--co-schedule-trade-auto-tau-secs must be finite",
            ),
            (
                vec!["--co-schedule-trade-auto-session-idle-secs=-1"],
                "--co-schedule-trade-auto-session-idle-secs must be finite",
            ),
            (
                vec!["--co-schedule-trade-rate-tokens-per-s", "0"],
                "--co-schedule-trade-rate-tokens-per-s must be finite and greater than 0",
            ),
        ] {
            let mut args = vec![
                "--co-schedule-trade",
                "--co-schedule-trade-copy-lambda",
                "auto",
            ];
            args.extend(extra);
            let err = build(&args).unwrap_err().to_string();
            assert!(err.contains(message), "{args:?}: {err}");
        }
        // Parse errors and an unknown q.
        let err = build(&[
            "--co-schedule-trade",
            "--co-schedule-trade-copy-lambda",
            "autox",
        ])
        .unwrap_err()
        .to_string();
        assert!(err.contains("expected a number or `auto`"), "{err}");
        for extra in [
            vec!["--co-schedule-trade-rate-tokens-per-s", "fast"],
            vec![
                "--co-schedule-trade-copy-lambda",
                "auto",
                "--co-schedule-trade-copy-lambda-q",
                "bpsat",
            ],
        ] {
            let mut args = vec!["--co-schedule-trade"];
            args.extend(extra);
            assert!(build(&args).is_err(), "{args:?}");
        }
        // The auto-only flags without an auto mode.
        let need_auto = "need --co-schedule-trade-copy-lambda auto";
        for (extra, message) in [
            (vec!["--co-schedule-trade-copy-lambda-q", "sat"], need_auto),
            (vec!["--co-schedule-trade-copy-lambda-mu", "1"], need_auto),
            (vec!["--co-schedule-trade-copy-lambda-min", "1"], need_auto),
            (vec!["--co-schedule-trade-copy-lambda-max", "1"], need_auto),
            (
                vec![
                    "--co-schedule-trade-copy-lambda",
                    "18.4",
                    "--co-schedule-trade-copy-lambda-fallback",
                    "18.4",
                ],
                need_auto,
            ),
            (
                vec!["--co-schedule-trade-auto-tau-secs", "60"],
                "or --co-schedule-trade-rate-tokens-per-s auto",
            ),
            (
                vec!["--co-schedule-trade-auto-session-idle-secs", "60"],
                "or --co-schedule-trade-rate-tokens-per-s auto",
            ),
        ] {
            let mut args = vec!["--co-schedule-trade"];
            args.extend(extra);
            let err = build(&args).unwrap_err().to_string();
            assert!(err.contains(message), "{args:?}: {err}");
        }
        // Every new flag needs --co-schedule-trade.
        for extra in [
            vec!["--co-schedule-trade-copy-lambda", "auto"],
            vec!["--co-schedule-trade-copy-lambda-q", "sat"],
            vec!["--co-schedule-trade-copy-lambda-fallback", "18.4"],
            vec!["--co-schedule-trade-auto-tau-secs", "60"],
            vec!["--co-schedule-trade-auto-session-idle-secs", "60"],
            vec!["--co-schedule-trade-rate-tokens-per-s", "auto"],
        ] {
            assert!(build(&extra).is_err(), "{extra:?}");
        }
    }

    /// The global waiting queue: off by default (also `off` given), sjf and
    /// hrrn on TRADE with the depth (auto or tokens) and the max wait; refused
    /// without TRADE, with wait eft, without the policy, with 0 values and
    /// with tuning flags while it is off.
    #[test]
    fn global_queue_flags_are_trade_only_and_validated() {
        use crate::global_queue::{
            GlobalQueueConfig, GlobalQueueDepth, GlobalQueueHold, GlobalQueueMode, GlobalQueueRoom,
        };
        let base = [
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "co_schedule",
            "--kv-indexer-url",
            "http://indexer:8090",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://x:30000","servicer":"x:18100"}]"#,
            "--co-schedule-uncached-balance",
            "--co-schedule-cost-trace",
            "/tmp/gwq-test.jsonl",
        ];
        let build = |extra: &[&str]| {
            let args: Vec<String> = base
                .iter()
                .map(|s| s.to_string())
                .chain(extra.iter().map(|s| s.to_string()))
                .collect();
            let refs: Vec<&str> = args.iter().map(String::as_str).collect();
            into_config_owned(with_model(&refs)).map(|c| c.model.co_schedule.unwrap())
        };
        assert_eq!(parse_tokens_or_auto("auto"), Ok(TokensOrAuto::Auto));
        assert_eq!(parse_tokens_or_auto(" Auto "), Ok(TokensOrAuto::Auto));
        assert_eq!(parse_tokens_or_auto("9542"), Ok(TokensOrAuto::Tokens(9542)));
        assert!(parse_tokens_or_auto("1.5").is_err());
        assert!(parse_tokens_or_auto("-1").is_err());

        // Default and explicit off: no queue, TRADE unchanged.
        assert!(build(&["--co-schedule-trade"])
            .unwrap()
            .global_queue
            .is_none());
        let off = build(&["--co-schedule-trade", "--co-schedule-global-queue", "off"]).unwrap();
        assert!(off.global_queue.is_none());
        assert_eq!(off.trade, build(&["--co-schedule-trade"]).unwrap().trade);

        // On: the default depth is auto; the default max wait is 300 s with
        // hrrn (a safety net) and 30 s with sjf (the starvation guard).
        let hrrn = build(&["--co-schedule-trade", "--co-schedule-global-queue", "hrrn"])
            .unwrap()
            .global_queue
            .unwrap();
        assert_eq!(
            hrrn,
            GlobalQueueConfig {
                mode: GlobalQueueMode::Hrrn,
                depth: GlobalQueueDepth::Auto,
                max_wait_ms: 300_000,
                hold: GlobalQueueHold::All,
                room: GlobalQueueRoom::Tokens,
            }
        );
        // R 13500, poll 100 ms + dispatch 500 ms: 8192 + 8100.
        assert_eq!(hrrn.depth_tokens(13_500.0, 100), 16_292);
        let sjf = build(&[
            "--co-schedule-trade",
            "--co-schedule-trade-copy-lambda",
            "auto",
            "--co-schedule-trade-copy-lambda-fallback",
            "18.4",
            "--co-schedule-global-queue",
            "sjf",
            "--co-schedule-global-queue-depth-tokens",
            "4096",
            "--co-schedule-global-queue-max-wait-ms",
            "5000",
        ])
        .unwrap()
        .global_queue
        .unwrap();
        assert_eq!(sjf.mode, GlobalQueueMode::Sjf);
        assert_eq!(sjf.depth, GlobalQueueDepth::Tokens(4096));
        assert_eq!(sjf.max_wait_ms, 5000);
        let auto = build(&[
            "--co-schedule-trade",
            "--co-schedule-global-queue",
            "sjf",
            "--co-schedule-global-queue-depth-tokens",
            "auto",
        ])
        .unwrap()
        .global_queue
        .unwrap();
        assert_eq!(auto.depth, GlobalQueueDepth::Auto);
        assert_eq!(auto.max_wait_ms, 30_000);

        // Refusals.
        let err = build(&["--co-schedule-global-queue", "sjf"])
            .unwrap_err()
            .to_string();
        assert!(err.contains("needs --co-schedule-trade"), "{err}");
        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("model.json");
        std::fs::write(
            &model,
            r#"{"models":{"augmented":{"launch_interval":{"global":
                {"coefficients":[0.05,1e-5,0.0,0.004,1e-7,0.0]}}}},"eft":{"tau_s":0.02}}"#,
        )
        .unwrap();
        let model_path = model.to_str().unwrap().to_string();
        let eft = [
            "--co-schedule-trade",
            "--co-schedule-trade-wait",
            "eft",
            "--co-schedule-trade-eft-model",
            model_path.as_str(),
        ];
        assert!(build(&eft).unwrap().global_queue.is_none());
        let mut eft_gwq = eft.to_vec();
        eft_gwq.extend(["--co-schedule-global-queue", "hrrn"]);
        let err = build(&eft_gwq).unwrap_err().to_string();
        assert!(
            err.contains("not supported with --co-schedule-trade-wait eft"),
            "{err}"
        );
        for extra in [
            vec!["--co-schedule-trade", "--co-schedule-global-queue", "fifo"],
            vec![
                "--co-schedule-trade",
                "--co-schedule-global-queue",
                "sjf",
                "--co-schedule-global-queue-depth-tokens",
                "0",
            ],
            vec![
                "--co-schedule-trade",
                "--co-schedule-global-queue",
                "hrrn",
                "--co-schedule-global-queue-max-wait-ms",
                "0",
            ],
            // Tuning flags without the flag, or with the queue off.
            vec![
                "--co-schedule-trade",
                "--co-schedule-global-queue-depth-tokens",
                "4096",
            ],
            vec![
                "--co-schedule-trade",
                "--co-schedule-global-queue-max-wait-ms",
                "100",
            ],
            vec![
                "--co-schedule-trade",
                "--co-schedule-global-queue",
                "off",
                "--co-schedule-global-queue-depth-tokens",
                "4096",
            ],
        ] {
            assert!(build(&extra).is_err(), "{extra:?}");
        }
        // Another policy refuses the queue.
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--co-schedule-global-queue",
            "hrrn",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("require --policy co_schedule"), "{err}");
    }

    /// Predicted-drain fill: `--co-schedule-global-queue-hold free` and
    /// `--co-schedule-global-queue-room predicted` (default `all` / `tokens`,
    /// so existing GWQ command lines parse as before).
    #[test]
    fn global_queue_hold_and_room_flags() {
        use crate::global_queue::{GlobalQueueHold, GlobalQueueRoom};
        use crate::scheduler::co_schedule::TradeWait;
        let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/e3");
        let params_path = format!("{fixtures}/bound-params.json");
        let base = [
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "co_schedule",
            "--kv-indexer-url",
            "http://indexer:8090",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://x:30000","servicer":"x:18100"}]"#,
            "--co-schedule-uncached-balance",
            "--co-schedule-cost-trace",
            "/tmp/gwq-pdf-test.jsonl",
        ];
        let build = |extra: &[&str]| {
            let args: Vec<String> = base
                .iter()
                .map(|s| s.to_string())
                .chain(extra.iter().map(|s| s.to_string()))
                .collect();
            let refs: Vec<&str> = args.iter().map(String::as_str).collect();
            into_config_owned(with_model(&refs)).map(|c| c.model.co_schedule.unwrap())
        };
        let first = build(&["--co-schedule-trade", "--co-schedule-global-queue", "hrrn"])
            .unwrap()
            .global_queue
            .unwrap();
        assert_eq!(
            (first.hold, first.room),
            (GlobalQueueHold::All, GlobalQueueRoom::Tokens)
        );
        let free = build(&[
            "--co-schedule-trade",
            "--co-schedule-global-queue",
            "hrrn",
            "--co-schedule-global-queue-hold",
            "free",
        ])
        .unwrap()
        .global_queue
        .unwrap();
        assert_eq!(
            (free.hold, free.room),
            (GlobalQueueHold::Free, GlobalQueueRoom::Tokens)
        );
        // Predicted room with the e3 forecast state; the ledger wait refuses it.
        let e3 = [
            "--co-schedule-trade",
            "--co-schedule-trade-wait",
            "e3",
            "--co-schedule-trade-e3-bound-table",
            params_path.as_str(),
            "--co-schedule-global-queue",
            "hrrn",
            "--co-schedule-global-queue-hold",
            "free",
            "--co-schedule-global-queue-room",
            "predicted",
        ];
        let cfg = build(&e3).unwrap();
        assert_eq!(cfg.trade.unwrap().wait, TradeWait::E3);
        let pdf = cfg.global_queue.unwrap();
        assert_eq!(
            (pdf.hold, pdf.room),
            (GlobalQueueHold::Free, GlobalQueueRoom::Predicted)
        );
        // Predicted room with the ledger wait: the E3 state for the drain only.
        let ledger = [
            "--co-schedule-trade",
            "--co-schedule-trade-e3-bound-table",
            params_path.as_str(),
            "--co-schedule-global-queue",
            "hrrn",
            "--co-schedule-global-queue-hold",
            "free",
            "--co-schedule-global-queue-room",
            "predicted",
        ];
        let cfg = build(&ledger).unwrap();
        assert_eq!(cfg.trade.unwrap().wait, TradeWait::Ledger);
        assert!(cfg.trade_e3.is_some());
        assert_eq!(cfg.global_queue.unwrap().room, GlobalQueueRoom::Predicted);
        let mut gated = ledger.to_vec();
        gated.extend(["--co-schedule-trade-gate-gmin-ms", "100"]);
        let err = build(&gated).unwrap_err().to_string();
        assert!(err.contains("need --co-schedule-trade-wait e3"), "{err}");
        let err = build(&[
            "--co-schedule-trade",
            "--co-schedule-global-queue",
            "hrrn",
            "--co-schedule-global-queue-room",
            "predicted",
        ])
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("requires --co-schedule-trade-e3-bound-table"),
            "{err}"
        );
        // The e3 flags without wait e3 and without room predicted are refused.
        let err = build(&[
            "--co-schedule-trade",
            "--co-schedule-trade-e3-bound-table",
            params_path.as_str(),
            "--co-schedule-global-queue",
            "hrrn",
        ])
        .unwrap_err()
        .to_string();
        assert!(err.contains("need --co-schedule-trade-wait e3"), "{err}");
        for extra in [
            vec![
                "--co-schedule-trade",
                "--co-schedule-global-queue-hold",
                "free",
            ],
            vec![
                "--co-schedule-trade",
                "--co-schedule-global-queue-room",
                "tokens",
            ],
            vec![
                "--co-schedule-trade",
                "--co-schedule-global-queue",
                "off",
                "--co-schedule-global-queue-hold",
                "free",
            ],
            vec![
                "--co-schedule-trade",
                "--co-schedule-global-queue",
                "hrrn",
                "--co-schedule-global-queue-hold",
                "bound",
            ],
            vec![
                "--co-schedule-trade",
                "--co-schedule-global-queue",
                "hrrn",
                "--co-schedule-global-queue-room",
                "e3",
            ],
        ] {
            assert!(build(&extra).is_err(), "{extra:?}");
        }
    }

    #[test]
    fn locality_margin_requires_a_trace_and_rejects_conflicting_modes() {
        let flags = [
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "co_schedule",
            "--kv-indexer-url",
            "http://indexer:8090",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://x:30000","servicer":"x:18100"}]"#,
            "--co-schedule-locality-margin-tokens",
            "32768",
        ];
        assert!(into_config_owned(with_model(&flags))
            .unwrap_err()
            .to_string()
            .contains("requires --co-schedule-cost-trace"));
        let mut valid = flags.to_vec();
        valid.extend(["--co-schedule-cost-trace", "/tmp/locality-test.jsonl"]);
        let config = into_config_owned(with_model(&valid))
            .unwrap()
            .model
            .co_schedule
            .unwrap();
        assert_eq!(config.locality_margin_tokens, Some(32768));
        assert!(!config.cost_balance);
        for extra in [
            vec!["--co-schedule-cost-balance"],
            vec!["--co-schedule-prefer-gpu-hit"],
            vec!["--kv-pull-scope", "detour-only"],
        ] {
            let mut conflicting = valid.clone();
            conflicting.extend(extra);
            assert!(into_config_owned(with_model(&conflicting)).is_err());
        }
        let orphan = [
            "--worker-urls",
            "http://x:30000",
            "--co-schedule-locality-margin-tokens",
            "0",
        ];
        assert!(into_config_owned(with_model(&orphan))
            .unwrap_err()
            .to_string()
            .contains("require --policy co_schedule"));
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
    fn rejects_co_schedule_tick_without_the_policy() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--co-schedule-tick-ms",
            "500",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("require --policy co_schedule"), "got: {err}");
    }

    #[test]
    fn rejects_zero_co_schedule_tick() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "co_schedule",
            "--kv-indexer-url",
            "http://indexer:8090",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://x:30000","servicer":"x:18100"}]"#,
            "--co-schedule-tick-ms",
            "0",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("--co-schedule-tick-ms must be greater than 0"),
            "got: {err}"
        );
    }

    #[test]
    fn accepts_zero_poll_on_hold_and_imbalance_as_disabled() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "co_schedule",
            "--kv-indexer-url",
            "http://indexer:8090",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://x:30000","servicer":"x:18100"}]"#,
            "--co-schedule-poll-on-hold-ms",
            "0",
            "--co-schedule-imbalance-spread-tokens",
            "0",
        ]))
        .unwrap();
        let cs = c.model.co_schedule.expect("co_schedule config built");
        assert_eq!(cs.poll_on_hold_ms, 0);
        assert_eq!(cs.imbalance_spread_tokens, 0);
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
        let base = ["--policy", "symmetric_pd", "--symmetric-pd-worker", "http://p:30000,8998,1,1"];
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
        let flex = c.model.symmetric_pd.as_ref().expect("symmetric_pd config");
        assert!(flex.placement.force_split);
        assert_eq!(flex.max_queue_wait_ms, 1500);
        assert_eq!(flex.placement.weights.w_dload, 0.0);
        assert_eq!(flex.placement.weights.w_hit, 1.0);
        assert_eq!(flex.workers.len(), 2);
        assert_eq!(flex.workers[0].bootstrap_port, Some(8998));
        assert_eq!(flex.workers[0].prefill_weight, 1.0);
        assert_eq!(flex.workers[0].decode_weight, 0.0);
        assert!(flex
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
        let err = into_config_owned(with_model(&args)).unwrap_err().to_string();
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
        assert_eq!(default.import_cache_policy, SymmetricPdImportCachePolicy::Auto);
        let mapping = r#"[{"id":1,"url":"http://p:30000","servicer":"p:18100"},{"id":2,"url":"http://d:30000/","servicer":"d:18100"}]"#;
        let mut args = base.to_vec();
        args.extend(["--symmetric-pd-decode-host-return", "--kv-worker-map", mapping]);
        let config = into_config_owned(with_model(&args)).unwrap();
        assert!(config.model.kv_hint.is_none());
        let flex = config.model.symmetric_pd.unwrap();
        assert_eq!(flex.import_cache_policy, SymmetricPdImportCachePolicy::Retain);
        let returns = flex.decode_host_return.unwrap();
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
        let returns = config.model.symmetric_pd.unwrap().decode_host_return.unwrap();
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
        let base = ["--policy", "cache_aware_zmq", "--worker-urls", "http://10.0.0.1:30000"];
        let c = into_config_owned(with_model(&base)).unwrap();
        assert!(c.model.pd_decision_replay.is_none());
        let mut args = base.to_vec();
        args.extend(["--pd-decision-replay-header", "X-Key"]);
        let err = into_config_owned(with_model(&args)).unwrap_err().to_string();
        assert!(err.contains("requires --pd-decision-replay"), "{err}");
        let mut args = base.to_vec();
        args.extend(["--pd-decision-replay", "/nonexistent/replay.json"]);
        let err = into_config_owned(with_model(&args)).unwrap_err().to_string();
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
        args.extend(["--pd-decision-replay", path_s.as_str(), "--pd-decision-replay-header", " X-Key "]);
        let c = into_config_owned(with_model(&args)).unwrap();
        assert_eq!(c.model.pd_decision_replay.as_ref().unwrap().header, "X-Key");
        // symmetric_pd has its own replay flag.
        let args = [
            "--policy", "symmetric_pd", "--symmetric-pd-worker", "http://10.0.0.1:30000,8998,1,1",
            "--symmetric-pd-worker", "http://10.0.0.5:30000,8998,1,1", "--pd-decision-replay", path_s.as_str(),
        ];
        let err = into_config_owned(with_model(&args)).unwrap_err().to_string();
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
        let err = into_config_owned(with_model(&args)).unwrap_err().to_string();
        assert!(err.contains("requires --symmetric-pd-decision-replay"), "{err}");
        // Missing file is rejected at config time.
        let mut args = base.to_vec();
        args.extend(["--symmetric-pd-decision-replay", "/nonexistent/replay.json"]);
        let err = into_config_owned(with_model(&args)).unwrap_err().to_string();
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
        let err = into_config_owned(with_model(&args)).unwrap_err().to_string();
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
        let flex = c.model.symmetric_pd.as_ref().unwrap();
        assert!(flex.external_role_control);
        assert_eq!(
            flex.role_drain_timeout,
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
            c.model.symmetric_pd.as_ref().unwrap().placement.host_hit_factor,
            1.0,
            "default"
        );
        let mut args = base.to_vec();
        args.extend(["--symmetric-pd-host-hit-factor", "0"]);
        let c = into_config_owned(with_model(&args)).unwrap();
        assert_eq!(c.model.symmetric_pd.as_ref().unwrap().placement.host_hit_factor, 0.0);
        let mut args = base.to_vec();
        args.extend(["--symmetric-pd-pull-hint"]);
        let c = into_config_owned(with_model(&args)).unwrap();
        assert!(c.model.symmetric_pd.as_ref().unwrap().pull_hint);
        let mut args = base.to_vec();
        args.extend(["--symmetric-pd-host-hit-factor", "1.5"]);
        let err = into_config_owned(with_model(&args)).unwrap_err().to_string();
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
        assert!(cfg.model.co_schedule.is_none());
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
    fn co_schedule_kv_session_retention_is_opt_in_and_refused_elsewhere() {
        let base = [
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "co_schedule",
            "--kv-indexer-url",
            "http://indexer:8090",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://x:30000","servicer":"x:18100"}]"#,
        ];
        let off = into_config_owned(with_model(&base)).unwrap();
        let cs = off.model.co_schedule.unwrap();
        assert!(!cs.kv_session_retention);
        assert!(cs.kv_session_key_file.is_none());

        // On, with the late decode choice and a key file.
        let mut args = base.to_vec();
        args.extend([
            "--kv-session-retention",
            "--kv-session-key-file",
            "/tmp/api-key",
            "--pd-decode-policy",
            "power-of-two-smg",
            "--pd-late-decode",
            "choose",
            "--pd-late-decode-rule",
            "d1",
        ]);
        let on = into_config_owned(with_model(&args)).unwrap();
        assert!(on.proxy.late_decode.choose());
        let cs = on.model.co_schedule.unwrap();
        assert!(cs.kv_session_retention);
        assert_eq!(
            cs.kv_session_key_file.unwrap(),
            std::path::PathBuf::from("/tmp/api-key")
        );

        // symmetric_pd keeps its own flag.
        let err = into_config_owned(with_model(&[
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://10.0.0.1:30000,8998,1,1",
            "--kv-session-retention",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("not available with --policy symmetric_pd"), "{err}");
        assert!(err.contains("--symmetric-pd-kv-session-retention"), "{err}");

        // Every other policy refuses it.
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--kv-session-retention",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("requires --policy co_schedule"), "{err}");

        // The key file needs the flag.
        let mut args = base.to_vec();
        args.extend(["--kv-session-key-file", "/tmp/api-key"]);
        assert!(into_config_owned(with_model(&args)).is_err());
    }

    #[test]
    fn symmetric_pd_co_schedule_is_explicit_and_keeps_native_defaults() {
        let base = [
            "--policy", "symmetric_pd",
            "--symmetric-pd-worker", "http://10.0.0.1:30000,8998,1,0",
            "--symmetric-pd-worker", "http://10.0.0.5:30000,8998,0,1",
        ];
        let mut args = base.to_vec();
        args.extend([
            "--symmetric-pd-p-select", "co-schedule",
            "--symmetric-pd-d-select", "round-robin",
            "--symmetric-pd-owner-threshold-tokens", "8192",
            "--symmetric-pd-detour-margin-tokens", "1024",
        ]);
        let config = into_config_owned(with_model(&args)).unwrap();
        let flex = config.model.symmetric_pd.unwrap();
        assert_eq!(flex.placement.p_select, crate::symmetric_pd::placement::PSelect::CoSchedule);
        assert_eq!(flex.placement.d_select, crate::symmetric_pd::placement::DSelect::RoundRobin);
        assert_eq!(flex.placement.co_schedule.owner_threshold_tokens, 8192);
        assert_eq!(flex.placement.co_schedule.detour_margin_tokens, 1024);
        let default = into_config_owned(with_model(&base)).unwrap();
        assert_eq!(default.model.symmetric_pd.unwrap().placement.p_select, crate::symmetric_pd::placement::PSelect::CacheFirst);
        let mut invalid = base.to_vec();
        invalid.extend(["--symmetric-pd-owner-threshold-tokens", "8192"]);
        assert!(into_config_owned(with_model(&invalid)).is_err());
    }

    #[test]
    fn symmetric_pd_selection_flags_map_into_placement_config() {
        use crate::symmetric_pd::placement::{DSelect, PSelect};
        let c = into_config_owned(with_model(&[
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://10.0.0.1:30000,8998,1,0",
            "--symmetric-pd-worker",
            "http://10.0.0.5:30000,8998,0,1",
        ]))
        .unwrap();
        let flex = c.model.symmetric_pd.as_ref().unwrap();
        assert_eq!(flex.placement.p_select, PSelect::CacheFirst, "default");
        assert_eq!(flex.placement.d_select, DSelect::Score, "default");
        assert_eq!(flex.placement.cache_threshold, 0.5);
        assert_eq!(flex.placement.balance_abs_reqs, 32);
        assert_eq!(flex.placement.balance_rel, 1.1);

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
        let flex = c.model.symmetric_pd.as_ref().unwrap();
        assert_eq!(flex.placement.p_select, PSelect::Score);
        assert_eq!(flex.placement.d_select, DSelect::RoundRobin);
        assert_eq!(flex.placement.cache_threshold, 0.25);
        assert_eq!(flex.placement.balance_abs_reqs, 8);
        assert_eq!(flex.placement.balance_rel, 1.5);

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
        let flex = c.model.symmetric_pd.as_ref().unwrap();
        assert_eq!(flex.placement.d_select, DSelect::CacheFirst);

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
        assert!(err.contains("--symmetric-pd-max-queue-wait-ms"), "got: {err}");
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
        let kv = c.model.kv_hint.expect("indexer config built for symmetric_pd");
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
        let flex = c.model.symmetric_pd.unwrap();
        assert_eq!(flex.workers.len(), 1);
        let w = &flex.workers[0];
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
            vec!["--co-schedule-trade"],
            vec!["--co-schedule-trade-copy-lambda", "auto"],
            vec!["--co-schedule-trade-copy-lambda-q", "sat"],
            vec!["--co-schedule-trade-copy-lambda-mu", "1"],
            vec!["--co-schedule-trade-copy-lambda-fallback", "18.4"],
            vec!["--co-schedule-trade-count-inflight"],
            vec!["--co-schedule-uncached-balance"],
            vec!["--co-schedule-load-poll-ms", "100"],
            vec!["--co-schedule-cost-trace", "/tmp/flex-cost.jsonl"],
            vec!["--kv-indexer-url", "http://127.0.0.1:9100"],
            vec!["--kv-worker-map", COST_FUNCTION_MAP],
            vec!["--kv-pull-hint-min-tokens", "256"],
            vec!["--kv-pull-hint-min-incremental-tokens", "512"],
            vec!["--kv-indexer-timeout-ms", "200"],
            vec!["--kv-pull-scope", "any"],
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
        use crate::symmetric_pd::placement::{DSelect, PSelect};
        use crate::scheduler::co_schedule::{TradeAutoQ, TradeLambdaMode, TradeRateMode};
        let cfg = cost_function_config(&[], &[]).unwrap();
        // The co-schedule policy is not built: its flags configure symmetric_pd.
        assert!(cfg.model.co_schedule.is_none());
        let kv = cfg.model.kv_hint.as_ref().unwrap();
        assert_eq!((kv.min_tokens, kv.min_incremental_tokens), (256, 512));
        assert_eq!(kv.worker_map.len(), 3);
        let flex = cfg.model.symmetric_pd.as_ref().unwrap();
        assert_eq!(flex.workers.len(), 9);
        assert_eq!(flex.placement.p_select, PSelect::CostFunction);
        assert_eq!(flex.placement.d_select, DSelect::RoundRobin);
        assert!(flex.placement.force_split && flex.pull_hint && flex.kv_session_retention);
        let cost = flex.cost_function.as_ref().unwrap();
        assert_eq!(cost.load_poll_ms, 100);
        assert_eq!(cost.unknown_expire_secs, 60, "the default unknown expiry");
        assert!(
            !cost.loads_host_room,
            "the host_room poll is off by default"
        );
        assert_eq!(cost.cost_trace.as_deref().and_then(|p| p.to_str()), Some("/tmp/flex-cost.jsonl"));
        let trade = cost.trade;
        assert_eq!(
            trade.copy_lambda_mode,
            TradeLambdaMode::Auto {
                q: TradeAutoQ::Sat,
                mu: 1.0,
                min: 4.6,
                max: 1000.0,
            }
        );
        assert_eq!(trade.copy_lambda, 18.4);
        assert_eq!(
            (trade.rate_mode, trade.rate_tokens_per_s),
            (TradeRateMode::Fixed, 13_500.0)
        );
        assert!(trade.is_auto() && !trade.exclude_unlaunched);
        let late = &cfg.proxy.late_decode;
        assert_eq!(late.mode, LateDecodeMode::Choose);
        assert_eq!(late.rule, Some(LateDecodeRule::FitReq));
        assert!(late.fit_native && late.native_state_poll && late.packed_prompt);
    }

    /// `--co-schedule-trade-wait e3` and `--symmetric-pd-gwq-room predicted` on
    /// the cost function: the E3 forecast state reaches the symmetric_pd config
    /// (decisions, or the queue drain with the ledger wait), and each missing
    /// piece is refused.
    #[test]
    fn symmetric_pd_cost_function_e3_and_gwq_room() {
        use crate::global_queue::GlobalQueueRoom;
        use crate::scheduler::co_schedule::TradeWait;
        let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/e3");
        let table = format!("{fixtures}/bound-params.json");
        let cost = |add: &[&str]| {
            cost_function_config(&[], add).map(|c| c.model.symmetric_pd.unwrap().cost_function.unwrap())
        };
        // Off by default: no E3 state, the token rule.
        let off = cost(&["--symmetric-pd-gwq-auto"]).unwrap();
        assert!(off.trade_e3.is_none());
        assert_eq!(off.gwq_auto.unwrap().room, GlobalQueueRoom::Tokens);
        // wait e3 with the gate at m = 0 and the state push.
        let e3 = cost(&[
            "--co-schedule-trade-wait",
            "e3",
            "--co-schedule-trade-e3-bound-table",
            &table,
            "--co-schedule-trade-gate-m",
            "0",
            "--co-schedule-trade-e3-push-port",
            "26648",
        ])
        .unwrap();
        assert_eq!(e3.trade.wait, TradeWait::E3);
        assert_eq!((e3.trade.gate_m, e3.trade.gate_gmin_s), (0.0, 0.0));
        let state = e3.trade_e3.as_ref().unwrap();
        assert_eq!(state.0.push.as_ref().map(|p| p.port), Some(26648));
        assert_eq!(state.0.bound_sha256.len(), 64);
        assert!(e3.gwq_auto.is_none());
        // The same file parsed twice compares equal (the config is PartialEq).
        assert_eq!(
            e3.trade_e3,
            cost(&[
                "--co-schedule-trade-wait",
                "e3",
                "--co-schedule-trade-e3-bound-table",
                &table,
                "--co-schedule-trade-e3-push-port",
                "26648",
            ])
            .unwrap()
            .trade_e3
        );
        // The queue drain with the ledger wait.
        let drain = cost(&[
            "--symmetric-pd-gwq-auto",
            "--symmetric-pd-gwq-room",
            "predicted",
            "--co-schedule-trade-e3-bound-table",
            &table,
        ])
        .unwrap();
        assert_eq!(drain.trade.wait, TradeWait::Ledger);
        assert_eq!(drain.gwq_auto.unwrap().room, GlobalQueueRoom::Predicted);
        assert!(drain.trade_e3.is_some());
        // Both: wait e3 and the predicted room.
        let both = cost(&[
            "--co-schedule-trade-wait",
            "e3",
            "--symmetric-pd-gwq-auto",
            "--symmetric-pd-gwq-room",
            "predicted",
            "--co-schedule-trade-e3-bound-table",
            &table,
        ])
        .unwrap();
        assert_eq!(both.trade.wait, TradeWait::E3);
        assert_eq!(both.gwq_auto.unwrap().room, GlobalQueueRoom::Predicted);
        assert!(!both.trade_e3.as_ref().unwrap().0.next_pass_lag);
        // The Kimi argv: wait e3, gate m 0, push, predicted room, next-pass lag.
        let kimi_args = [
            "--co-schedule-trade-wait",
            "e3",
            "--co-schedule-trade-e3-bound-table",
            table.as_str(),
            "--co-schedule-trade-gate-m",
            "0",
            "--co-schedule-trade-e3-push-port",
            "26648",
            "--symmetric-pd-gwq-auto",
            "--symmetric-pd-gwq-room",
            "predicted",
        ];
        let kimi_off = cost(&kimi_args).unwrap();
        let mut with_lag = kimi_args.to_vec();
        with_lag.push("--co-schedule-trade-e3-next-pass-lag");
        let kimi = cost(&with_lag).unwrap();
        assert!(kimi.trade_e3.as_ref().unwrap().0.next_pass_lag);
        assert!(!kimi_off.trade_e3.as_ref().unwrap().0.next_pass_lag);
        assert_ne!(kimi.trade_e3, kimi_off.trade_e3, "switch in the state");
        // The frozen fit: the Kimi fixture constants, with their sha256.
        let frozen = format!("{fixtures}/kimi-frozen-fit.json");
        let mut with_frozen = with_lag.clone();
        with_frozen.extend(["--co-schedule-trade-e3-frozen-fit", frozen.as_str()]);
        let kimi_frozen = cost(&with_frozen).unwrap();
        let state = kimi_frozen.trade_e3.as_ref().unwrap();
        let fit = state.0.frozen_fit.as_ref().unwrap();
        assert_eq!(fit.sha256.len(), 64);
        assert!(state.0.next_pass_lag);
        assert!(kimi.trade_e3.as_ref().unwrap().0.frozen_fit.is_none());
        assert_ne!(kimi_frozen.trade_e3, kimi.trade_e3, "in the state");
        // A file that is not a frozen fit (the bound table) is refused.
        let mut bad = with_lag.clone();
        bad.extend(["--co-schedule-trade-e3-frozen-fit", table.as_str()]);
        let err = cost(&bad).unwrap_err().to_string();
        assert!(err.contains("--co-schedule-trade-e3-frozen-fit"), "{err}");
        let refused = |add: &[&str], want: &str| {
            let err = cost(add).unwrap_err().to_string();
            assert!(err.contains(want), "{add:?}: {err}");
        };
        refused(
            &["--symmetric-pd-gwq-auto", "--symmetric-pd-gwq-room", "predicted"],
            "--symmetric-pd-gwq-room predicted requires --co-schedule-trade-e3-bound-table",
        );
        refused(
            &["--symmetric-pd-gwq-room", "predicted"],
            "--symmetric-pd-gwq-room requires --symmetric-pd-gwq-auto",
        );
        refused(
            &["--co-schedule-trade-wait", "e3"],
            "--co-schedule-trade-wait e3 requires --co-schedule-trade-e3-bound-table",
        );
        refused(
            &[
                "--symmetric-pd-gwq-auto",
                "--symmetric-pd-gwq-room",
                "predicted",
                "--co-schedule-trade-e3-bound-table",
                &table,
                "--co-schedule-trade-gate-m",
                "0",
            ],
            "--co-schedule-trade-gate-* need --co-schedule-trade-wait e3",
        );
        refused(
            &["--co-schedule-trade-e3-bound-table", &table],
            "need --co-schedule-trade-wait e3",
        );
        refused(
            &["--co-schedule-trade-e3-next-pass-lag"],
            "need --co-schedule-trade-wait e3",
        );
        refused(
            &["--co-schedule-trade-e3-frozen-fit", &frozen],
            "need --co-schedule-trade-wait e3",
        );
        let err = into_config_owned(with_model(&[
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://10.0.0.1:18080,18998,1,0",
            "--symmetric-pd-gwq-room",
            "tokens",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("--symmetric-pd-gwq-room requires --symmetric-pd-p-select cost-function"),
            "{err}"
        );
    }

    #[test]
    fn symmetric_pd_cost_function_fixed_lambda() {
        use crate::scheduler::co_schedule::TradeLambdaMode;
        let cfg = cost_function_config(
            &["--co-schedule-trade-copy-lambda"],
            &["--co-schedule-trade-copy-lambda", "18.4"],
        )
        .unwrap();
        let trade = cfg.model.symmetric_pd.unwrap().cost_function.unwrap().trade;
        assert_eq!(trade.copy_lambda_mode, TradeLambdaMode::Fixed);
        assert_eq!(trade.copy_lambda, 18.4);
        assert!(!trade.is_auto());
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
            let cfg = cost_function_config(&[], &["--symmetric-pd-unknown-expire-secs", value]).unwrap();
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
        let want = "--symmetric-pd-unknown-expire-secs requires --symmetric-pd-p-select cost-function";
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
        let want = "--symmetric-pd-pull-from-any-holder requires --symmetric-pd-p-select cost-function";
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

    /// `--co-schedule-trade-wait olp`, `--co-schedule-trade-olp-params` and
    /// `--co-schedule-trade-olp-shadow`: off by default; the parameter file
    /// overrides by name and is hashed; unknown keys and invalid values stop
    /// the start; the shadow readout needs the ledger wait; both need the
    /// cost function (refused with `--policy co_schedule`).
    #[test]
    fn symmetric_pd_cost_function_olp_flags() {
        use crate::olp::{OlpParams, OlpScaleSource};
        use crate::scheduler::co_schedule::{TradeParams, TradeWait};
        let cost = |add: &[&str]| {
            cost_function_config(&[], add).map(|c| c.model.symmetric_pd.unwrap().cost_function.unwrap())
        };
        // Off by default.
        let off = cost(&[]).unwrap();
        assert!(off.olp.is_none());
        assert_eq!(off.trade.wait, TradeWait::Ledger);
        // wait olp with the default parameters.
        let on = cost(&["--co-schedule-trade-wait", "olp"]).unwrap();
        assert_eq!(on.trade.wait, TradeWait::Olp);
        // The wait is the only TRADE parameter that OLP changes.
        let ledger = TradeParams {
            wait: TradeWait::Ledger,
            ..on.trade
        };
        assert_eq!(ledger, off.trade);
        let olp = on.olp.unwrap();
        assert_eq!(olp.params, OlpParams::default());
        assert!(!olp.shadow && olp.params_sha256.is_none());
        // A parameter file overrides by name and is hashed.
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, text: &str| {
            let path = dir.path().join(name);
            std::fs::write(&path, text).unwrap();
            path.to_str().unwrap().to_string()
        };
        let good = write("olp.json", r#"{"c1": 4e-5, "scale_source": "busy"}"#);
        let with_file = |path: &str| {
            cost(&[
                "--co-schedule-trade-wait",
                "olp",
                "--co-schedule-trade-olp-params",
                path,
            ])
        };
        let olp = with_file(good.as_str()).unwrap().olp.unwrap();
        let want = OlpParams {
            c1: 4e-5,
            scale_source: OlpScaleSource::Busy,
            ..OlpParams::default()
        };
        assert_eq!(olp.params, want);
        assert!(!olp.shadow);
        assert_eq!(olp.params_sha256.as_deref().map(str::len), Some(64));
        // Unknown keys, invalid values, a non-object and a missing file.
        for (text, message) in [
            (r#"{"c9": 1}"#, "unknown field"),
            (r#"{"chunk": 0}"#, "chunk must be at least 1"),
            (r#"{"c1": -1}"#, "c1 must be finite and greater than 0"),
            ("[1, 2]", "must be a JSON object"),
        ] {
            let path = write("bad.json", text);
            let err = with_file(path.as_str()).unwrap_err().to_string();
            assert!(err.contains("--co-schedule-trade-olp-params"), "{err}");
            assert!(err.contains(message), "{text}: {err}");
        }
        let missing = dir.path().join("missing.json");
        let err = with_file(missing.to_str().unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("--co-schedule-trade-olp-params"), "{err}");
        // The shadow readout: the ledger wait; OLP is fed and traced only.
        let shadow = cost(&["--co-schedule-trade-olp-shadow"]).unwrap();
        assert_eq!(shadow.trade.wait, TradeWait::Ledger);
        assert_eq!(shadow.trade, off.trade);
        let olp = shadow.olp.unwrap();
        assert!(olp.shadow && olp.params_sha256.is_none());
        let shadow_file = [
            "--co-schedule-trade-olp-shadow",
            "--co-schedule-trade-olp-params",
            good.as_str(),
        ];
        let olp = cost(&shadow_file).unwrap().olp.unwrap();
        assert!(olp.shadow);
        assert_eq!(olp.params, want);
        // Refusals: shadow with wait olp; the file without olp or shadow.
        let both = [
            "--co-schedule-trade-wait",
            "olp",
            "--co-schedule-trade-olp-shadow",
        ];
        let err = cost(&both).unwrap_err().to_string();
        let want = "--co-schedule-trade-olp-shadow needs --co-schedule-trade-wait ledger";
        assert!(err.contains(want), "{err}");
        let err = cost(&["--co-schedule-trade-olp-params", good.as_str()])
            .unwrap_err()
            .to_string();
        let want = "--co-schedule-trade-olp-params needs --co-schedule-trade-wait olp or \
                    --co-schedule-trade-olp-shadow";
        assert!(err.contains(want), "{err}");
        // The co_schedule policy has no OLP.
        let co_schedule = |extra: &[&str]| {
            let mut args = vec![
                "--worker-urls",
                "http://x:30000",
                "--policy",
                "co_schedule",
                "--kv-indexer-url",
                "http://indexer:8090",
                "--kv-worker-map",
                r#"[{"id":1,"url":"http://x:30000","servicer":"x:18100"}]"#,
                "--co-schedule-uncached-balance",
                "--co-schedule-cost-trace",
                "/tmp/trade-olp-test.jsonl",
                "--co-schedule-trade",
            ];
            args.extend_from_slice(extra);
            into_config_owned(with_model(&args))
                .unwrap_err()
                .to_string()
        };
        let err = co_schedule(&["--co-schedule-trade-wait", "olp"]);
        let want = "--co-schedule-trade-wait olp requires --policy symmetric_pd --symmetric-pd-p-select \
                    cost-function";
        assert!(err.contains(want), "{err}");
        let err = co_schedule(&["--co-schedule-trade-olp-shadow"]);
        let want = "--co-schedule-trade-olp-shadow requires --policy symmetric_pd";
        assert!(err.contains(want), "{err}");
    }

    /// `--co-schedule-trade-olp-incremental`, `--co-schedule-trade-olp-trace-compact`
    /// and `--co-schedule-cost-trace-async`: off by default; the OLP options
    /// need wait olp or the shadow readout and change nothing else in the
    /// config; the async trace needs the symmetric_pd cost function and its
    /// trace and changes nothing else.
    #[test]
    fn symmetric_pd_cost_function_olp_fast_flags() {
        let cost = |add: &[&str]| {
            cost_function_config(&[], add).map(|c| c.model.symmetric_pd.unwrap().cost_function.unwrap())
        };
        let off = cost(&["--co-schedule-trade-wait", "olp"]).unwrap();
        let olp_off = off.olp.clone().unwrap();
        assert!(!olp_off.incremental && !olp_off.compact_trace && !olp_off.verify);
        assert!(!off.trace_async);
        let on = cost(&[
            "--co-schedule-trade-wait",
            "olp",
            "--co-schedule-trade-olp-incremental",
            "--co-schedule-trade-olp-trace-compact",
            "--co-schedule-cost-trace-async",
        ])
        .unwrap();
        let olp_on = on.olp.clone().unwrap();
        assert!(olp_on.incremental && olp_on.compact_trace && !olp_on.verify);
        assert!(on.trace_async);
        assert_eq!(
            crate::olp::OlpConfig {
                incremental: false,
                compact_trace: false,
                ..olp_on
            },
            olp_off
        );
        assert_eq!(on.trade, off.trade);
        let shadow = cost(&[
            "--co-schedule-trade-olp-shadow",
            "--co-schedule-trade-olp-incremental",
        ])
        .unwrap();
        let olp = shadow.olp.unwrap();
        assert!(olp.shadow && olp.incremental && !olp.compact_trace);
        // The async trace alone (no OLP).
        let plain = cost(&["--co-schedule-cost-trace-async"]).unwrap();
        assert!(plain.trace_async && plain.olp.is_none());
        // Refusals: the OLP options without OLP; async without the cost function.
        for flag in [
            "--co-schedule-trade-olp-incremental",
            "--co-schedule-trade-olp-trace-compact",
        ] {
            let err = cost(&[flag]).unwrap_err().to_string();
            let want = format!(
                "{flag} needs --co-schedule-trade-wait olp or --co-schedule-trade-olp-shadow"
            );
            assert!(err.contains(&want), "{err}");
        }
        let co_schedule = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "co_schedule",
            "--kv-indexer-url",
            "http://indexer:8090",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://x:30000","servicer":"x:18100"}]"#,
            "--co-schedule-uncached-balance",
            "--co-schedule-cost-trace",
            "/tmp/trade-async-test.jsonl",
            "--co-schedule-cost-trace-async",
        ]))
        .unwrap_err()
        .to_string();
        let want = "--co-schedule-cost-trace-async requires --policy symmetric_pd --symmetric-pd-p-select \
                    cost-function";
        assert!(co_schedule.contains(want), "{co_schedule}");
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
            let flex = c.model.symmetric_pd.as_ref().unwrap();
            assert!(flex.external_role_control);
            assert!(flex.cost_function.is_some());
            assert_eq!(
                flex.role_drain_timeout,
                Some(std::time::Duration::from_secs(5))
            );
            assert_eq!(c.proxy.late_decode.choose(), drop.is_empty());
        }
    }

    /// GWQ-AUTO on the dynamic argv: `--symmetric-pd-gwq-auto` and
    /// `--symmetric-pd-trace-session-moves` are accepted with
    /// `--symmetric-pd-external-role-control`, with and without the router late-D
    /// choice (the queue steps over the current P-role pool).
    #[test]
    fn symmetric_pd_gwq_auto_accepts_external_role_control() {
        for drop in [&["--pd-native-decode-state", "--pd-late-decode"][..], &[]] {
            let c = cost_function_config(
                drop,
                &[
                    "--symmetric-pd-external-role-control",
                    "--symmetric-pd-gwq-auto",
                    "--symmetric-pd-trace-session-moves",
                ],
            )
            .unwrap();
            let flex = c.model.symmetric_pd.as_ref().unwrap();
            assert!(flex.external_role_control);
            let cost = flex.cost_function.as_ref().unwrap();
            assert_eq!(
                cost.gwq_auto,
                Some(crate::config::SymmetricPdGwqAutoConfig {
                    max_wait_ms: 300_000,
                    ..Default::default()
                })
            );
            assert!(cost.trace_session_moves);
            assert_eq!(c.proxy.late_decode.choose(), drop.is_empty());
        }
    }

    /// `--symmetric-pd-loads-host-room` turns on the host_room observation of the
    /// cost function. It needs `--symmetric-pd-p-select cost-function` and, as
    /// every `--symmetric-pd-*` flag, `--policy symmetric_pd`.
    #[test]
    fn symmetric_pd_loads_host_room() {
        let cfg = cost_function_config(&[], &["--symmetric-pd-loads-host-room"]).unwrap();
        assert!(
            cfg.model
                .symmetric_pd
                .unwrap()
                .cost_function
                .unwrap()
                .loads_host_room
        );
        let err = into_config_owned(with_model(&[
            "--policy",
            "symmetric_pd",
            "--symmetric-pd-worker",
            "http://10.0.0.1:18080,18998,1,0",
            "--symmetric-pd-loads-host-room",
        ]))
        .unwrap_err()
        .to_string();
        let want = "--symmetric-pd-loads-host-room requires --symmetric-pd-p-select cost-function";
        assert!(err.contains(want), "{err}");
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--symmetric-pd-loads-host-room",
        ]))
        .unwrap_err()
        .to_string();
        let want = "--symmetric-pd-* flags require --policy symmetric_pd";
        assert!(err.contains(want), "{err}");
    }

    /// `--symmetric-pd-gwq-auto` and `--symmetric-pd-trace-session-moves` (default
    /// off): only with `--symmetric-pd-p-select cost-function`; the queue has the
    /// 300-s guard; the co_schedule queue flag is refused with the cost
    /// function (its queue is --symmetric-pd-gwq-auto).
    #[test]
    fn symmetric_pd_gwq_auto_flags() {
        let cost = |cfg: Config| cfg.model.symmetric_pd.unwrap().cost_function.unwrap();
        let off = cost(cost_function_config(&[], &[]).unwrap());
        assert!(off.gwq_auto.is_none() && !off.trace_session_moves);
        let on = cost(
            cost_function_config(
                &[],
                &["--symmetric-pd-gwq-auto", "--symmetric-pd-trace-session-moves"],
            )
            .unwrap(),
        );
        assert_eq!(
            on.gwq_auto,
            Some(crate::config::SymmetricPdGwqAutoConfig {
                max_wait_ms: 300_000,
                ..Default::default()
            })
        );
        assert!(on.trace_session_moves);
        for (flag, want) in [
            (
                "--symmetric-pd-gwq-auto",
                "--symmetric-pd-gwq-auto requires --symmetric-pd-p-select cost-function",
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
        let err = cost_function_config(&[], &["--co-schedule-global-queue", "hrrn"])
            .unwrap_err()
            .to_string();
        let want = "--co-schedule-global-queue is not available with --symmetric-pd-p-select \
                    cost-function: its global waiting queue is --symmetric-pd-gwq-auto";
        assert!(err.contains(want), "{err}");
    }

    /// `--co-schedule-trade-pull-price` (default `lambda`). `phi` is accepted
    /// with the lambda = q words (copy price auto, mu 1, rate auto) and
    /// --symmetric-pd-loads-host-room on the symmetric_pd cost function. Each missing
    /// requirement is refused, and the co_schedule policy refuses `phi`.
    #[test]
    fn trade_pull_price_phi_flag() {
        use crate::scheduler::co_schedule::TradePullPrice;
        let trade = |cfg: Config| {
            let flex = cfg.model.symmetric_pd.unwrap();
            flex.cost_function.unwrap().trade
        };
        let rate_auto = ["--co-schedule-trade-rate-tokens-per-s", "auto"];
        let phi = ["--co-schedule-trade-pull-price", "phi"];
        let host_room = ["--symmetric-pd-loads-host-room"];
        let default = trade(cost_function_config(&[], &[]).unwrap());
        assert_eq!(default.pull_price, TradePullPrice::Lambda);
        let lambda = ["--co-schedule-trade-pull-price", "lambda"];
        let lambda = trade(cost_function_config(&[], &lambda).unwrap());
        assert_eq!(lambda.pull_price, TradePullPrice::Lambda);
        let on = [&rate_auto[..], &phi[..], &host_room[..]].concat();
        let on = trade(cost_function_config(&[], &on).unwrap());
        assert_eq!(on.pull_price, TradePullPrice::Phi);
        assert!(on.is_auto());
        let refused = |drop: &[&str], add: Vec<&str>, want: &str| {
            let err = cost_function_config(drop, &add).unwrap_err().to_string();
            assert!(err.contains(want), "{want}: got {err}");
        };
        let need = "--co-schedule-trade-pull-price phi requires";
        refused(
            &[],
            [&rate_auto[..], &phi[..]].concat(),
            &format!("{need} --symmetric-pd-loads-host-room"),
        );
        refused(
            &[],
            [&phi[..], &host_room[..]].concat(),
            &format!("{need} --co-schedule-trade-rate-tokens-per-s auto"),
        );
        let fixed = ["--co-schedule-trade-copy-lambda", "18.4"];
        refused(
            &["--co-schedule-trade-copy-lambda"],
            [&fixed[..], &rate_auto[..], &phi[..], &host_room[..]].concat(),
            &format!("{need} --co-schedule-trade-copy-lambda auto"),
        );
        let mu = ["--co-schedule-trade-copy-lambda-mu", "2"];
        refused(
            &["--co-schedule-trade-copy-lambda-mu"],
            [&mu[..], &rate_auto[..], &phi[..], &host_room[..]].concat(),
            &format!("{need} --co-schedule-trade-copy-lambda-mu 1"),
        );
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "co_schedule",
            "--kv-indexer-url",
            "http://indexer:8090",
            "--kv-worker-map",
            r#"[{"id":1,"url":"http://x:30000","servicer":"x:18100"}]"#,
            "--co-schedule-uncached-balance",
            "--co-schedule-cost-trace",
            "/tmp/trade-phi-test.jsonl",
            "--co-schedule-trade",
            "--co-schedule-trade-copy-lambda",
            "auto",
            "--co-schedule-trade-rate-tokens-per-s",
            "auto",
            "--co-schedule-trade-pull-price",
            "phi",
        ]))
        .unwrap_err()
        .to_string();
        let want = format!("{need} --policy symmetric_pd with --symmetric-pd-p-select cost-function");
        assert!(err.contains(&want), "{err}");
        let psi = ["--co-schedule-trade-pull-price", "psi"];
        let err = cost_function_config(&[], &psi).unwrap_err().to_string();
        assert!(err.contains("psi"), "{err}");
        let dup_all = ["--co-schedule-trade-dup-charge", "all"];
        refused(
            &[],
            [&rate_auto[..], &phi[..], &host_room[..], &dup_all[..]].concat(),
            "--co-schedule-trade-pull-price phi does not take --co-schedule-trade-dup-charge all",
        );
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
                "--kv-pull-scope require --policy co_schedule, or --policy symmetric_pd with \
                 --symmetric-pd-p-select cost-function",
            ),
            (
                "N2",
                &["--co-schedule-trade"],
                &[],
                "--symmetric-pd-p-select cost-function requires --co-schedule-trade (the TRADE cost \
                 model)",
            ),
            (
                "N3",
                &[],
                &[
                    "--co-schedule-trade-wait",
                    "eft",
                    "--co-schedule-trade-eft-model",
                    "/nonexistent/eft-model.json",
                ],
                "--co-schedule-trade-wait eft is not available with --symmetric-pd-p-select \
                 cost-function; use --co-schedule-trade-wait e3 (the step-time-model predictor)",
            ),
            (
                "N4",
                &[],
                &["--co-schedule-on-demand-load-refresh"],
                "--co-schedule-on-demand-load-refresh is not available with --symmetric-pd-p-select \
                 cost-function: the P progress is polled every --co-schedule-load-poll-ms",
            ),
            (
                "N5",
                &["--co-schedule-load-poll-ms"],
                &["--co-schedule-load-poll-ms", "0"],
                "--co-schedule-load-poll-ms must be greater than 0",
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
                "--symmetric-pd-p-select cost-function requires --symmetric-pd-pull-hint: the TRADE pull \
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
                "E1",
                &["--co-schedule-uncached-balance"],
                &[],
                "--co-schedule-trade requires --co-schedule-uncached-balance",
            ),
            (
                "E3",
                &[],
                &["--co-schedule-max-uncached-tokens", "4096"],
                "uncached balance has no admission cap, HOLD queue or dispatch timer",
            ),
            (
                "E4",
                &[],
                &["--co-schedule-cost-balance"],
                "cost balance, locality margin and uncached balance are mutually exclusive",
            ),
            (
                "E5",
                &[],
                &["--co-schedule-owner-threshold-tokens", "1"],
                "--co-schedule-trade replaces the owner threshold, restore divisor and detour \
                 margin",
            ),
            (
                "E6",
                &[],
                &["--co-schedule-prefer-gpu-hit"],
                "--co-schedule-prefer-gpu-hit requires capped co-schedule mode",
            ),
            (
                "E7",
                &["--kv-pull-scope"],
                &["--kv-pull-scope", "detour-only"],
                "reserved placement requires --kv-pull-scope any",
            ),
            (
                "E8",
                &["--kv-pull-hint-min-incremental-tokens"],
                &["--kv-pull-hint-min-incremental-tokens", "256"],
                "--co-schedule-trade never pulls fewer than 512 tokens",
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
                "E11",
                &[],
                &["--symmetric-pd-owner-threshold-tokens", "1"],
                "require --symmetric-pd-p-select co-schedule",
            ),
            (
                "N15",
                &["--symmetric-pd-p-select"],
                &["--symmetric-pd-unknown-expire-secs", "60"],
                "--symmetric-pd-unknown-expire-secs requires --symmetric-pd-p-select cost-function",
            ),
            (
                "N16",
                &["--symmetric-pd-p-select"],
                &["--symmetric-pd-loads-host-room"],
                "--symmetric-pd-loads-host-room requires --symmetric-pd-p-select cost-function",
            ),
        ];
        for (id, drop, add, want) in rows {
            let err = cost_function_config(drop, add).unwrap_err().to_string();
            assert!(err.contains(want), "{id}: got {err:?}, want {want:?}");
        }
    }

    /// The co-schedule gate keeps its message for the other policies.
    #[test]
    fn co_schedule_flags_keep_their_message_without_symmetric_pd() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "round_robin",
            "--co-schedule-uncached-balance",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("require --policy co_schedule"), "{err}");
    }
}
