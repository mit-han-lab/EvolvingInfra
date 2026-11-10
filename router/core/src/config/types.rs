use std::num::NonZeroU32;

/// In-memory router configuration, built from CLI flags by
/// [`crate::config::cli::Cli::into_config`] and validated by
/// [`Config::validate`]. The router serves exactly one model.
#[derive(Debug, Clone)]
pub struct Config {
    pub server: ServerConfig,
    pub observability: ObservabilityConfig,
    pub model: ModelConfig,
    /// Selected discovery backend. Built from CLI flags by
    /// [`crate::config::cli::Cli::into_config`]: the static-vs-k8s choice
    /// and the k8s selector grammar are resolved there (the latter via
    /// [`resolve_mode`]); static worker-URL validity is checked by
    /// [`Config::validate`].
    pub discovery: DiscoveryBackend,
    pub proxy: ProxyConfig,
    pub active_load: ActiveLoadConfig,
}

/// Outbound proxy tuning. Default mirrors SGLang's typical prefill /
/// decode latency budget; e2e tests lower it so per-request failures
/// trip the circuit breaker within the test's wall-time.
#[derive(Debug, Clone, Copy)]
pub struct ProxyConfig {
    /// Maximum time to wait for a single upstream HTTP request to
    /// return headers + body. Default 300 s. The circuit breaker
    /// records a failure when this fires.
    pub request_timeout_secs: u64,
    /// `--disable-circuit-breaker`: every worker's breaker always admits
    /// and never opens (no worker leaves the candidate sets on upstream
    /// failures). Default `false`: each worker gets the default breaker
    /// (3 consecutive failures open it for 30 s) or the `--cb-threshold`
    /// one.
    pub disable_circuit_breaker: bool,
    /// `--proxy-tcp-keepalive-secs`: TCP keepalive idle time
    /// (`TCP_KEEPIDLE`) of every upstream socket the proxy opens. `None`
    /// keeps reqwest's default (15 s). `Some(30)` gives the sockets the
    /// same options as sgl-model-gateway's client (its
    /// `--tcp-keepalive-secs` default): idle 30 s, probe interval 15 s,
    /// 3 probes, `TCP_USER_TIMEOUT` 30 s (the last three are reqwest
    /// defaults in both routers).
    pub tcp_keepalive_secs: Option<u64>,
    /// `--pd-abort-decode-on-prefill-failure`: when a PD request's prefill
    /// leg fails (transport error or non-2xx), drop its decode leg (close
    /// the decode connection, so the engine aborts the request) and fail
    /// the client request, as sgl-model-gateway's PD router does. Default
    /// `false`: the decode leg is left to the engine's own KV-wait timeout.
    pub abort_decode_on_prefill_failure: bool,
    /// Late D (`--pd-late-decode*`): bind-time decode choice and packed
    /// decode prompts. Default: both off, the router dispatches as before.
    pub late_decode: LateDecodeConfig,
    /// `--max-completions-body-mib`: the `/v1/completions` request-body cap
    /// in bytes (axum `DefaultBodyLimit`; larger bodies get HTTP 413).
    /// Default 5 MiB (= the chat route's cap). Token-id replay with uploaded
    /// screenshots (Kimi CUA) needs ~100 MiB.
    pub max_completions_body_bytes: usize,
    /// `--image-layout-config`: screenshot-aware token ids for token-id
    /// `/v1/completions` requests with `image_data` (Kimi), see
    /// [`crate::image_layout`]. Loaded once at startup (leaked: lives for
    /// the process). `None`: requests route on the placeholder ids.
    pub image_layout: Option<&'static crate::image_layout::ImageLayoutConfig>,
    /// `--late-d-image-source-url`: late-D decode legs of a request with a
    /// prepared image layout carry no `image_data`, only a
    /// `prepared_image_source` pointing at this URL (this router's
    /// `/internal/prepared_image`), see [`crate::image_source`]. Leaked:
    /// lives for the process. `None`: decode legs carry the pixels.
    pub late_d_image_source_url: Option<&'static str>,
}

/// Default `/v1/completions` body cap (5 MiB).
pub const DEFAULT_MAX_COMPLETIONS_BODY_BYTES: usize = 5 << 20;

pub fn default_proxy_request_timeout_secs() -> u64 {
    300
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            request_timeout_secs: default_proxy_request_timeout_secs(),
            disable_circuit_breaker: false,
            tcp_keepalive_secs: None,
            abort_decode_on_prefill_failure: false,
            late_decode: LateDecodeConfig::default(),
            max_completions_body_bytes: DEFAULT_MAX_COMPLETIONS_BODY_BYTES,
            image_layout: None,
            late_d_image_source_url: None,
        }
    }
}

/// Late D stage 2 (port of sgl-model-gateway's
/// `routers/http/late_bind.rs`): with `choose`, a PD request goes to one
/// prefill and to K decode candidates at once; the prefill asks the router
/// for the winner when it is about to prefill ([`crate::late_bind`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LateDecodeConfig {
    pub mode: LateDecodeMode,
    /// The bind-time rule; `Some` exactly when `mode` is `Choose`.
    pub rule: Option<LateDecodeRule>,
    /// Decode candidates per request; `None` = every decode worker (K = N).
    pub candidates: Option<usize>,
    /// `/v1/completions` decode legs carry a token prompt as
    /// `prompt_ids_b64` + `prompt_len` (in either mode).
    pub packed_prompt: bool,
    /// `--pd-late-decode-fit-capacity-tokens`: the KV capacity of one
    /// decode worker in tokens for the `fit_req` fit test with a fixed
    /// capacity (router-charged tokens + the request's charge <= capacity).
    /// Required by `fit_req` without `fit_native`; refused otherwise.
    pub fit_capacity_tokens: Option<u64>,
    /// `--pd-late-decode-fit-native`: `fit_req` reads the capacity each
    /// decode worker reports (see [`crate::late_fit`]) instead of a fixed
    /// capacity. Needs `native_state_poll`.
    pub fit_native: bool,
    /// `--pd-native-decode-state`: poll every decode worker's observed
    /// allocation state every 100 ms. Read-only; valid in any mode.
    pub native_state_poll: bool,
    /// `--pd-late-decode-fit-state-max-age-ms`: the oldest capacity sample
    /// (HTTP round trip and engine sample age included) that `fit_native`
    /// uses; an older one makes the bind fall back to `d1`. Default 1000.
    pub fit_state_max_age_ms: u64,
}

/// Default `--pd-late-decode-fit-state-max-age-ms` (sgl-model-gateway).
pub const DEFAULT_LATE_DECODE_FIT_STATE_MAX_AGE_MS: u64 = 1000;

impl Default for LateDecodeConfig {
    fn default() -> Self {
        Self {
            mode: LateDecodeMode::Off,
            rule: None,
            candidates: None,
            packed_prompt: false,
            fit_capacity_tokens: None,
            fit_native: false,
            native_state_poll: false,
            fit_state_max_age_ms: DEFAULT_LATE_DECODE_FIT_STATE_MAX_AGE_MS,
        }
    }
}

impl LateDecodeConfig {
    /// Choose mode is on (K decode legs per PD request, bind-time choice).
    pub fn choose(&self) -> bool {
        self.mode == LateDecodeMode::Choose
    }
}

/// `--pd-late-decode`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum LateDecodeMode {
    /// One decode leg, chosen at arrival (the router's behaviour without
    /// late D).
    #[default]
    #[value(name = "off")]
    Off,
    /// One decode leg per candidate; the prefill binds one of them.
    #[value(name = "choose")]
    Choose,
}

/// `--pd-late-decode-rule`: how the router picks the winner at the bind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum LateDecodeRule {
    /// min-inflight-tokens over the eligible candidates: the fewest
    /// router-charged tokens (prompt + capped output of every bound,
    /// unfinished request), then the fewest charged requests, then URL.
    #[value(name = "d1")]
    D1,
    /// Upstream SMG power_of_two over the eligible candidates: two distinct
    /// uniform draws, the lower polled `late_d.bound_total_tokens` wins
    /// (request counts when a draw has no polled value; ties to the first).
    #[value(name = "po2")]
    Po2,
    /// The arrival pick (the baseline).
    #[value(name = "keep")]
    Keep,
    /// Memory first, then step time. Among the eligible candidates that
    /// fit, the one with the fewest requests; `d1` when none fits. With a
    /// fixed capacity (`fit_capacity_tokens`) a candidate fits when its
    /// router-charged tokens plus the request's charge are at most the
    /// capacity, and the order is the fewest router-charged requests, then
    /// tokens, then URL. With `fit_native` a candidate fits when the
    /// request's page-rounded input plus the output reserve is at most the
    /// worker's reported allocatable tokens (less the router reservations
    /// it has not reported yet), and the order is the fewest allocated
    /// requests, then context tokens, then the most free GPU tokens, then
    /// URL (sgl-model-gateway `fit_req`).
    #[value(name = "fit_req")]
    FitReq,
}

impl LateDecodeRule {
    pub fn as_str(self) -> &'static str {
        match self {
            LateDecodeRule::D1 => "d1",
            LateDecodeRule::Po2 => "po2",
            LateDecodeRule::Keep => "keep",
            LateDecodeRule::FitReq => "fit_req",
        }
    }
}

/// Active-load (per-request) tracking. Production default (10 min)
/// sits above `proxy.request_timeout_secs` so the proxy timeout is the
/// one users hit first for normal slow upstreams; tests lower it to
/// let the janitor fire within their wall-time budget.
#[derive(Debug, Clone, Copy)]
pub struct ActiveLoadConfig {
    /// How long a request entry can live in the registry before the
    /// janitor fires its `cancel_token` and the chat handler returns
    /// 504 `stale_request_expired`. Default 600 s.
    pub stale_request_timeout_secs: u64,
    /// `--pd-decode-policy power-of-two-smg` only: seconds between the
    /// `/v1/loads` polling rounds of its load monitor (sgl-model-gateway
    /// `--load-check-interval-secs`). Default 5, as in SMG.
    pub pd_decode_load_check_interval_secs: u64,
}

pub fn default_stale_request_timeout_secs() -> u64 {
    600
}

/// sgl-model-gateway's `--load-check-interval-secs` default.
pub const DEFAULT_PD_DECODE_LOAD_CHECK_INTERVAL_SECS: u64 = 5;

impl Default for ActiveLoadConfig {
    fn default() -> Self {
        Self {
            stale_request_timeout_secs: default_stale_request_timeout_secs(),
            pd_decode_load_check_interval_secs: DEFAULT_PD_DECODE_LOAD_CHECK_INTERVAL_SECS,
        }
    }
}

/// Routing policy selector — the enum form lets `clap` reject unknown
/// values at parse time and removes the runtime string match in the
/// policy factory.
///
/// Accepted on the CLI (`--policy`) as `round_robin` / `random` /
/// `power_of_two` / `load_based` / `cache_aware_zmq` / `sticky`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum PolicyKind {
    #[default]
    #[value(name = "round_robin")]
    RoundRobin,
    #[value(name = "random")]
    Random,
    #[value(name = "power_of_two")]
    PowerOfTwo,
    /// Selects the currently least-loaded worker.
    #[value(name = "load_based")]
    LoadBased,
    /// Cache-aware routing fed by SGLang's ZMQ KV-cache event publisher.
    /// Requires the model to have a tokenizer loaded; cache_aware tuning
    /// lives on `ModelConfig::cache_aware`.
    #[value(name = "cache_aware_zmq")]
    CacheAwareZmq,
    /// Sticky-session routing: pins a routing key (read from a
    /// configurable request header) to a worker via an in-memory map, so
    /// stateful sessions land on the same backend. Tuning — header name,
    /// keyless-fallback policy, and TTL eviction — lives on
    /// `ModelConfig::sticky`.
    #[value(name = "sticky")]
    Sticky,
    /// Fixed-fleet comparison only: agent key -> observed SMG HTTP PD worker order.
    #[value(name = "experimental_agent_sticky")]
    ExperimentalAgentSticky,
    /// Co-scheduled routing + KV movement: prefer the max-cache-hit worker
    /// unless its pending uncached-prefill backlog exceeds a threshold, in
    /// which case either detour to an under-threshold worker (pulling the
    /// KV from the owner via `kv_pull_hint`) or hold the request in a
    /// router-side admission queue until capacity frees. Tuning lives on
    /// `ModelConfig::co_schedule`; requires the kv-hint indexer flags
    /// (`--kv-indexer-url`, `--kv-worker-map`).
    #[value(name = "co_schedule")]
    CoSchedule,
    /// symmetric P/D placement: every request goes through
    /// `crate::symmetric_pd::placement::place()` over the PD workers (legacy
    /// Prefill/Decode plus `Symmetric` engines with weights); Split
    /// requests carry `kv_export` / `kv_import` marks instead of the
    /// `bootstrap_*` fields, Local requests are one plain request. Tuning
    /// lives on `ModelConfig::symmetric_pd`; the indexer feed reuses
    /// `--kv-indexer-url` / `--kv-worker-map`.
    #[value(name = "symmetric_pd")]
    SymmetricPd,
    /// The prefill selection of sgl-model-gateway's `--prefill-policy
    /// cache_aware`, ported: an approximate radix
    /// tree of the routed prompts per worker (token ids, 8 hex characters per
    /// token), the match-rate threshold, the imbalance rule on the in-flight
    /// prefill request count, and size eviction on a timer. Tuning lives on
    /// `ModelConfig::gateway_cache_aware` (`--gateway-ca-*`). Not the
    /// KV-event policy `cache_aware_zmq`.
    #[value(name = "gateway_cache_aware")]
    GatewayCacheAware,
}

/// When the co-schedule policy may attach a `kv_pull_hint` to a dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum KvPullScope {
    /// Attach on ANY dispatch whose best source beats the target by the
    /// incremental gate (routing and pulling fully decoupled).
    #[value(name = "any")]
    Any,
    /// Attach only on the step-3 detour (routed away from the max-hit
    /// worker) — the literal three-step rule, kept for ablation.
    #[value(name = "detour-only")]
    DetourOnly,
}

/// Decode-peer selection rule for PD-mode dispatches. Orthogonal to
/// `--policy`, which only ever selects over the prefill pool — this knob
/// governs the second pick after the prefill worker is chosen. Runs
/// compared with each other must share one value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum PdDecodePolicy {
    /// Same-host affinity, then min active-load among healthy decodes
    /// (the historical behavior; see
    /// [`crate::policies::registry::select_decode_with_affinity`]).
    #[value(name = "affinity-min-load")]
    AffinityMinLoad,
    /// Per-request power-of-two: sample two distinct healthy decodes,
    /// dispatch to the lower `active_load`.
    #[value(name = "po2")]
    PowerOfTwo,
    /// Per-request round robin over the healthy decode pool (atomic
    /// counter, same semantics as sgl-model-gateway's `round_robin`
    /// decode policy).
    #[value(name = "round-robin")]
    RoundRobin,
    /// Minimum in-flight charged tokens: the router charges every request
    /// dispatched to a decode worker its prompt length plus a capped output
    /// budget, releases the charge when the decode leg ends (success, error,
    /// abort or client disconnect), and picks the healthy decode with the
    /// fewest charged tokens (ties: fewest charged requests, then URL order).
    /// See [`crate::policies::decode_tokens`].
    #[value(name = "min-inflight-tokens")]
    MinInflightTokens,
    /// Power-of-two over the same token charge: sample two distinct healthy
    /// decodes, take the one with fewer charged tokens.
    #[value(name = "po2-tokens")]
    PowerOfTwoTokens,
    /// sgl-model-gateway `--decode-policy power_of_two`: two distinct healthy
    /// samples; compare the token loads polled from every worker's
    /// `/v1/loads?include=core` (every `--pd-decode-load-check-interval-secs`)
    /// when both samples have one, else both SMG request counts; ties to the
    /// first sample. See [`crate::policies::decode_po2_smg`].
    #[value(name = "power-of-two-smg")]
    PowerOfTwoSmg,
}

impl PdDecodePolicy {
    /// The token-charged decode policies (stateful; selected through
    /// `AppContext::select_decode`, never through `PdPoolResolver::decode_select`).
    pub fn is_token_charged(self) -> bool {
        matches!(self, Self::MinInflightTokens | Self::PowerOfTwoTokens)
    }
}

/// Tuning for the token-charged decode policies
/// (`--pd-decode-policy min-inflight-tokens | po2-tokens`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdDecodeTokensConfig {
    /// Output tokens charged per request: `min(request max_tokens, cap)`,
    /// or `cap` when the request carries no output limit.
    pub output_charge_cap: u64,
    /// Compare `charged / max_total_num_tokens` (from each decode worker's
    /// `/server_info` at registration) instead of raw charged tokens. Only
    /// applied when every candidate reported a capacity; otherwise all
    /// decodes are treated as equal capacity.
    pub normalize_by_capacity: bool,
}

/// Default output-token charge cap for the token-charged decode policies.
pub const DEFAULT_PD_DECODE_OUTPUT_CHARGE_CAP: u64 = 8192;

impl Default for PdDecodeTokensConfig {
    fn default() -> Self {
        Self {
            output_charge_cap: DEFAULT_PD_DECODE_OUTPUT_CHARGE_CAP,
            normalize_by_capacity: false,
        }
    }
}

/// Tuning for the co-schedule policy. Ignored unless
/// `policy = "co_schedule"`.
#[derive(Debug, Clone)]
pub struct CoScheduleConfig {
    /// Query stale progress on request arrival instead of unconditional polling.
    /// Only supported with the request-level uncached-work ledger.
    pub on_demand_load_refresh: bool,
    /// Immediately balance remaining uncached work, with engine RID progress.
    pub uncached_balance: bool,
    /// Retain a cache owner at or below this existing workload (not a cap).
    pub owner_threshold_tokens: u64,
    pub detour_margin_tokens: u64,
    /// Optional measured ratio D: cost = D * uncached + pending restore tokens.
    /// Enables immediate minimum-finish-work placement; never creates a gate.
    pub restore_cost_divisor: Option<u64>,
    /// `--co-schedule-trade`: the TRADE P-selection cost model on the
    /// uncached ledger (`Some` only with `uncached_balance`). `None` keeps
    /// the owner-threshold selection unchanged.
    pub trade: Option<crate::scheduler::co_schedule::TradeParams>,
    /// `--co-schedule-trade-eft-model`: the frozen step-time model of
    /// `--co-schedule-trade-wait eft` (`Some` only then).
    pub trade_eft_model: Option<std::sync::Arc<crate::eft_cheap::EftModel>>,
    /// `--co-schedule-global-queue sjf|hrrn` (`Some` only with `trade`, wait
    /// ledger): the router-side global waiting queue of the TRADE placement.
    /// `None` (default, `off`): TRADE
    /// dispatches every request at arrival, unchanged.
    pub global_queue: Option<crate::global_queue::GlobalQueueConfig>,
    /// `--co-schedule-trade-e3-bound-table` and the other `wait = e3`
    /// settings that are not in [`crate::scheduler::co_schedule::TradeParams`]
    /// (`Some` only with `--co-schedule-trade-wait e3`).
    pub trade_e3: Option<std::sync::Arc<crate::trade_e3::TradeE3Config>>,
    /// Prefer GPU-resident prefixes when total cache hits tie in capped mode.
    pub prefer_gpu_hit: bool,
    /// Bypass the cap, keeping the GPU-local owner unless a detour reduces
    /// outstanding expected uncached work by more than this token margin.
    pub locality_margin_tokens: Option<u64>,
    /// Minimize the post-dispatch spread of input_tokens * expected_uncached.
    /// This mode bypasses the legacy uncached-token admission cap.
    pub cost_balance: bool,
    /// Append-only per-request cost events and one-second snapshots.
    pub cost_trace: Option<std::path::PathBuf>,
    /// Dispatch to a worker only while its pending uncached-prefill tokens
    /// (engine-reported baseline + router in-flight ledger) stay below this
    /// (CLI default 32768).
    pub max_uncached_tokens: u64,
    /// A request queued longer than this is force-dispatched to the
    /// min-backlog worker (starvation guard).
    pub max_queue_wait_ms: u64,
    /// Interval for polling each mapped worker's `/v1/loads`.
    pub load_poll_ms: u64,
    /// Dispatcher fallback wake period (ms). The loop also wakes on
    /// arrival, ledger retire, poll, and exactly at the oldest queued
    /// entry's starvation deadline. Must be > 0. CLI default 500.
    pub tick_ms: u64,
    /// While the queue is holding, re-poll `/v1/loads` this many ms after
    /// the previous round instead of waiting `load_poll_ms`; 0 disables
    /// (ledger-only between periodic polls). CLI default 100.
    pub poll_on_hold_ms: u64,
    /// When every pool worker is over the cap but the backlog spread
    /// (max − min, tokens) exceeds this, the queue head is dispatched to
    /// the min-backlog worker anyway (branch `imbalance`) instead of
    /// holding; 0 disables. CLI default 131072.
    pub imbalance_spread_tokens: u64,
    /// See [`KvPullScope`].
    pub pull_scope: KvPullScope,
    /// `--kv-session-retention`: session KV retention on the PD path (the
    /// engine session protocol). A PD request with an
    /// `x-kv-session-context` header registers its prefill worker, and only
    /// the prefill-leg body carries `kv_session`. Does not change placement.
    pub kv_session_retention: bool,
    /// `--kv-session-key-file`: the engine API/admin key file for the
    /// background retention controls.
    pub kv_session_key_file: Option<std::path::PathBuf>,
}

/// Tuning for `--policy gateway_cache_aware`. `Some` exactly when that policy
/// is selected. The defaults are the sgl-model-gateway CLI defaults (not its
/// `CacheAwareConfig::default()`, which the CLI never uses).
#[derive(Debug, Clone, PartialEq)]
pub struct GatewayCacheAwareConfig {
    /// `--gateway-ca-cache-threshold` (gateway `--cache-threshold`, default
    /// 0.3): the cache branch needs `matched / input > threshold`.
    pub cache_threshold: f32,
    /// `--gateway-ca-balance-abs-threshold` (gateway
    /// `--balance-abs-threshold`, default 64).
    pub balance_abs_threshold: usize,
    /// `--gateway-ca-balance-rel-threshold` (gateway
    /// `--balance-rel-threshold`, default 1.5). Imbalanced when
    /// `max - min > abs` and `max > min * rel` (loads = in-flight prefill
    /// requests counted by this router).
    pub balance_rel_threshold: f32,
    /// `--gateway-ca-eviction-interval-secs` (gateway `--eviction-interval`,
    /// default 120); 0 = no eviction thread.
    pub eviction_interval_secs: u64,
    /// `--gateway-ca-max-tree-size` (gateway `--max-tree-size`, default
    /// 67108864): tree characters per worker kept by each eviction. A token
    /// is 8 characters, so the default is 8388608 tokens per worker.
    pub max_tree_size: usize,
    /// `--kv-session-retention` with this policy (see
    /// [`CoScheduleConfig::kv_session_retention`]).
    pub kv_session_retention: bool,
    /// `--kv-session-key-file` with this policy.
    pub kv_session_key_file: Option<std::path::PathBuf>,
}

/// sgl-model-gateway CLI default of `--cache-threshold`.
pub const DEFAULT_GATEWAY_CA_CACHE_THRESHOLD: f32 = 0.3;
/// sgl-model-gateway CLI default of `--balance-abs-threshold`.
pub const DEFAULT_GATEWAY_CA_BALANCE_ABS_THRESHOLD: usize = 64;
/// sgl-model-gateway CLI default of `--balance-rel-threshold`.
pub const DEFAULT_GATEWAY_CA_BALANCE_REL_THRESHOLD: f32 = 1.5;
/// sgl-model-gateway CLI default of `--eviction-interval`.
pub const DEFAULT_GATEWAY_CA_EVICTION_INTERVAL_SECS: u64 = 120;
/// sgl-model-gateway CLI default of `--max-tree-size`.
pub const DEFAULT_GATEWAY_CA_MAX_TREE_SIZE: usize = 67_108_864;

impl Default for GatewayCacheAwareConfig {
    fn default() -> Self {
        Self {
            cache_threshold: DEFAULT_GATEWAY_CA_CACHE_THRESHOLD,
            balance_abs_threshold: DEFAULT_GATEWAY_CA_BALANCE_ABS_THRESHOLD,
            balance_rel_threshold: DEFAULT_GATEWAY_CA_BALANCE_REL_THRESHOLD,
            eviction_interval_secs: DEFAULT_GATEWAY_CA_EVICTION_INTERVAL_SECS,
            max_tree_size: DEFAULT_GATEWAY_CA_MAX_TREE_SIZE,
            kv_session_retention: false,
            kv_session_key_file: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone)]
pub struct ObservabilityConfig {
    pub log_level: String,
    /// Selects the tracing-subscriber output format. `clap` rejects
    /// unrecognized values at parse time (`--log-format jsonl` and
    /// similar typos surface as an error instead of silently degrading
    /// to text).
    pub log_format: LogFormat,
}

/// `text` for human-readable dev output, `json` for one-line-per-record
/// JSON suitable for k8s log aggregators (fluent-bit / vector / Loki).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum LogFormat {
    #[default]
    #[value(name = "text")]
    Text,
    #[value(name = "json")]
    Json,
}

fn default_log_level() -> String {
    "info".to_string()
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            log_level: default_log_level(),
            log_format: LogFormat::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub id: String,
    /// Tokenizer source: a local `tokenizer.json` path or a HuggingFace repo
    /// id (downloaded on demand). Defaults to `id` when `--tokenizer-path`
    /// is omitted. Resolved by [`crate::tokenizer::adapter::load`].
    pub tokenizer_path: String,
    pub policy: PolicyKind,
    pub circuit_breaker: Option<CircuitBreakerConfig>,
    /// Tuning for cache-aware routing. Ignored unless
    /// `policy = "cache_aware_zmq"`. `None` falls back to defaults at
    /// policy construction time.
    pub cache_aware: Option<CacheAwareConfig>,
    /// Tuning for the sticky-session policy. `Some` exactly when
    /// `policy = "sticky"` (built by [`crate::config::cli::Cli::into_config`]).
    /// The chat handler reads `sticky.header_name` to populate
    /// [`crate::policies::SelectionContext::routing_key`].
    pub sticky: Option<StickyConfig>,
    /// KV pull-hint injection (peer KV pulls between workers). `Some`
    /// exactly when `--inject-kv-pull-hint` is set; the [`AppContext`]
    /// builds a [`crate::kv_hint::KvHintInjector`] from it at startup.
    ///
    /// [`AppContext`]: crate::server::app_context::AppContext
    pub kv_hint: Option<crate::kv_hint::KvHintConfig>,
    /// Co-schedule tuning. `Some` exactly when `policy = "co_schedule"`;
    /// `kv_hint` is also always `Some` then (the policy needs the indexer
    /// for per-worker cache hits and reuses the hint machinery for pulls).
    pub co_schedule: Option<CoScheduleConfig>,
    /// Decode-peer selection for PD-mode dispatches. See [`PdDecodePolicy`].
    pub pd_decode_policy: PdDecodePolicy,
    /// Token-charged decode tuning. `Some` exactly when `pd_decode_policy`
    /// is token-charged or the late-D rule is `d1` (built by the CLI); a
    /// hand-built config with a token-charged policy and `None` here uses
    /// the defaults.
    pub pd_decode_tokens: Option<PdDecodeTokensConfig>,
    /// Traditional-PD per-request decision replay (`--pd-decision-replay`):
    /// the prefill worker and its decode peer are taken from a table
    /// recorded by a reference run (same file format as
    /// `--symmetric-pd-decision-replay`, see [`crate::symmetric_pd::DecisionReplay`]),
    /// so a traditional-PD run can be compared with a symmetric P/D run under
    /// identical per-request placement. `None` = live selection. Unknown
    /// keys / lengths fall back to `policy` + `pd_decode_policy`.
    pub pd_decision_replay: Option<PdDecisionReplayConfig>,
    /// symmetric P/D tuning. `Some` exactly when `policy = "symmetric_pd"`.
    pub symmetric_pd: Option<SymmetricPdConfig>,
    /// `--policy gateway_cache_aware` tuning. `Some` exactly when that policy
    /// is selected.
    pub gateway_cache_aware: Option<GatewayCacheAwareConfig>,
}

impl ModelConfig {
    /// Session KV retention on the PD path (`--kv-session-retention`, with
    /// `--policy co_schedule` or `--policy gateway_cache_aware`): `Some(key
    /// file)` when on, `None` when off. symmetric_pd keeps its own flag.
    pub fn pd_kv_session_retention(&self) -> Option<Option<&std::path::Path>> {
        if let Some(c) = self.co_schedule.as_ref().filter(|c| c.kv_session_retention) {
            return Some(c.kv_session_key_file.as_deref());
        }
        self.gateway_cache_aware
            .as_ref()
            .filter(|g| g.kv_session_retention)
            .map(|g| g.kv_session_key_file.as_deref())
    }
}

/// See [`ModelConfig::pd_decision_replay`].
#[derive(Debug, Clone, PartialEq)]
pub struct PdDecisionReplayConfig {
    pub table: std::path::PathBuf,
    /// Request header carrying the routing key (default `X-SMG-Routing-Key`).
    pub header: String,
}

/// One statically configured symmetric P/D worker (`--symmetric-pd-worker` /
/// `--symmetric-pd-workers-file`). The URL is also fed to
/// static discovery; the manager marks the worker `Symmetric` and applies
/// these weights after `/server_info` introspection.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SymmetricPdWorkerSpec {
    pub url: String,
    /// Base port of the per-rank ZMQ side channels (owner = port + 2*tp_rank,
    /// reader = owner + 1); the engine's `--disaggregation-bootstrap-port`.
    #[serde(default)]
    pub bootstrap_port: Option<u16>,
    /// Primary-role weights; `0` = never used for that leg.
    #[serde(default = "default_role_weight")]
    pub prefill_weight: f64,
    #[serde(default = "default_role_weight")]
    pub decode_weight: f64,
    /// Capability set (`"prefill"` / `"decode"`); default both.
    #[serde(default = "default_capabilities")]
    pub capabilities: Vec<String>,
}

fn default_role_weight() -> f64 {
    1.0
}

fn default_capabilities() -> Vec<String> {
    vec!["prefill".to_string(), "decode".to_string()]
}

impl SymmetricPdWorkerSpec {
    /// Parse the CLI shorthand `URL[,BOOTSTRAP_PORT[,PREFILL_WEIGHT[,DECODE_WEIGHT]]]`,
    /// e.g. `http://10.0.0.1:30000,8998,1,0` for a prefill-leaning worker.
    pub fn parse_cli(spec: &str) -> Result<Self, String> {
        let mut parts = spec.split(',').map(str::trim);
        let url = parts
            .next()
            .filter(|u| !u.is_empty())
            .ok_or_else(|| format!("--symmetric-pd-worker '{spec}': missing URL"))?
            .to_string();
        let bootstrap_port = match parts.next() {
            None | Some("") | Some("none") | Some("None") => None,
            Some(p) => Some(
                p.parse::<u16>()
                    .map_err(|e| format!("--symmetric-pd-worker '{spec}': bad bootstrap port: {e}"))?,
            ),
        };
        let weight = |s: Option<&str>, name: &str| -> Result<f64, String> {
            match s {
                None | Some("") => Ok(1.0),
                Some(w) => w
                    .parse::<f64>()
                    .map_err(|e| format!("--symmetric-pd-worker '{spec}': bad {name} weight: {e}"))
                    .and_then(|w| {
                        if w.is_finite() && w >= 0.0 {
                            Ok(w)
                        } else {
                            Err(format!(
                                "--symmetric-pd-worker '{spec}': {name} weight must be >= 0"
                            ))
                        }
                    }),
            }
        };
        let prefill_weight = weight(parts.next(), "prefill")?;
        let decode_weight = weight(parts.next(), "decode")?;
        if parts.next().is_some() {
            return Err(format!("--symmetric-pd-worker '{spec}': too many fields"));
        }
        Ok(Self {
            url,
            bootstrap_port,
            prefill_weight,
            decode_weight,
            capabilities: default_capabilities(),
        })
    }

    /// URL with a trailing `/` stripped, the form worker ids use.
    pub fn normalized_url(&self) -> String {
        self.url.trim().trim_end_matches('/').to_string()
    }

    /// Capability set as the placement sees it.
    pub fn capabilities(&self) -> crate::symmetric_pd::placement::Capabilities {
        let has = |name: &str| {
            self.capabilities
                .iter()
                .any(|c| c.eq_ignore_ascii_case(name))
        };
        crate::symmetric_pd::placement::Capabilities {
            prefill: has("prefill"),
            decode: has("decode"),
        }
    }

    pub fn weight(&self) -> crate::symmetric_pd::placement::RoleWeight {
        crate::symmetric_pd::placement::RoleWeight {
            prefill: self.prefill_weight,
            decode: self.decode_weight,
        }
    }
}

/// `--symmetric-pd-p-select`: how the symmetric P/D P-leg is chosen (see
/// `symmetric_pd::placement` module docs, "selection modes").
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SymmetricPdPSelect {
    /// Argmax of the additive score over all `(p, d)` pairs.
    #[value(name = "score")]
    Score,
    /// `cache_aware_zmq` semantics over the prefill-capable pool: imbalance
    /// fast path, then best hit above `--symmetric-pd-cache-threshold`, then
    /// least-loaded (default).
    #[value(name = "cache-first")]
    CacheFirst,
    /// Co-schedule prefill placement and optional KV movement; D is configured separately.
    #[value(name = "co-schedule")]
    CoSchedule,
    /// sgl-model-gateway `--prefill-policy cache_aware` (the
    /// `--policy gateway_cache_aware` rule, its tree and its `--gateway-ca-*`
    /// flags and defaults) picks the P-leg among the prefill-capable
    /// workers; the D-leg follows `--symmetric-pd-d-select`. No hold queue.
    #[value(name = "gateway-cache-aware")]
    GatewayCacheAware,
    /// The TRADE cost model (`--co-schedule-trade*`, lambda fixed or auto)
    /// picks the P-leg among the P-role workers and decides the pull. The
    /// D-leg follows `--symmetric-pd-d-select`. No hold queue.
    #[value(name = "cost-function")]
    CostFunction,
}

impl From<SymmetricPdPSelect> for crate::symmetric_pd::placement::PSelect {
    fn from(v: SymmetricPdPSelect) -> Self {
        match v {
            SymmetricPdPSelect::Score => Self::Score,
            SymmetricPdPSelect::CacheFirst => Self::CacheFirst,
            SymmetricPdPSelect::CoSchedule => Self::CoSchedule,
            // The runtime narrows the prefill pool to the gateway's pick
            // (`SymmetricPdConfig::gateway_cache_aware`); the placement then
            // takes that single P.
            SymmetricPdPSelect::GatewayCacheAware => Self::CacheFirst,
            // The runtime narrows the prefill pool to the P that the cost
            // function reserved (`SymmetricPdConfig::cost_function`).
            SymmetricPdPSelect::CostFunction => Self::CostFunction,
        }
    }
}

/// `--symmetric-pd-p-select cost-function`: the TRADE cost model on the P-role
/// workers of `--policy symmetric_pd`. The runtime keeps its own cost ledger
/// ([`crate::cost_balance::CostBook`]) and its own per-RID progress poll of
/// the P-role workers; the price (lambda, R, the TRADE cost) comes from the
/// cost book only.
#[derive(Debug, Clone, PartialEq)]
pub struct SymmetricPdCostFunctionConfig {
    /// The `--co-schedule-trade*` parameters (lambda fixed or auto).
    pub trade: crate::scheduler::co_schedule::TradeParams,
    /// `--co-schedule-cost-trace`: a new JSONL file (an existing file stops
    /// the start). Every row has `"policy": "symmetric_pd_cost_function"`.
    pub cost_trace: std::path::PathBuf,
    /// `--co-schedule-load-poll-ms` (default 100): the period of the
    /// per-RID progress poll of each P-role worker.
    pub load_poll_ms: u64,
    /// `--symmetric-pd-unknown-expire-secs` (default 60): an unknown reservation
    /// (a P-leg transport error or a drop after dispatch) leaves the ledger
    /// when a progress reply of its P, sent at least this many seconds after
    /// the later of the moment it became unknown and the last listing of its
    /// RID, omits the RID. A P without replies keeps it. 0 = never: it is
    /// held until a terminal result.
    pub unknown_expire_secs: u64,
    /// `--symmetric-pd-loads-host-room` (default off): the progress poll also asks
    /// each P for its `host_room` section (host pool by retention class), and
    /// the cost trace records the latest one per P (`host_room_by_p` in the
    /// snapshot and in every decision). Observation only.
    pub loads_host_room: bool,
    /// `--symmetric-pd-gwq-auto` (default off = None): the global waiting queue
    /// of the TRADE P choice with the measured lead.
    pub gwq_auto: Option<SymmetricPdGwqAutoConfig>,
    /// `--symmetric-pd-trace-session-moves` (default off): each TRADE reserve
    /// event with a session identity gets a `session_move` record (the P of
    /// the session's last decision, the hits, the pull). Trace only.
    pub trace_session_moves: bool,
    /// The E3 forecast state (`--co-schedule-trade-e3-*`): `Some` with
    /// `--co-schedule-trade-wait e3` (the step-time-model predictor in the P
    /// selection, with the prediction gate) or `--symmetric-pd-gwq-room
    /// predicted` (the drain of the global waiting queue). None otherwise.
    pub trade_e3: Option<SymmetricPdTradeE3>,
    /// `--symmetric-pd-pull-from-any-holder` (default off): the pull source of the
    /// P-leg (`kv_pull_hint`) may be any holder of the indexer answer that is
    /// a healthy registered worker of any role (breaker allowing) with a
    /// `--kv-worker-map` entry, also a D-role worker (a former P after a P->D
    /// role switch). The targets stay the healthy P-role workers. Off: every
    /// hit and every pull source is a P-role worker.
    pub pull_from_any_holder: bool,
    /// `--co-schedule-trade-wait olp` or `--co-schedule-trade-olp-shadow`:
    /// the online latency predictor of the cost book ([`crate::olp`]), with
    /// its parameters (the defaults or `--co-schedule-trade-olp-params`),
    /// whether it only traces (shadow) and the parameter file's sha256.
    /// None (default): no OLP state; decisions and trace are unchanged.
    pub olp: Option<crate::olp::OlpConfig>,
    /// `--co-schedule-cost-trace-async`: the cost trace is written by a
    /// background thread ([`crate::cost_balance::CostBook::enable_async_trace`]).
    /// Off (default): each row is written and flushed under the cost-book
    /// lock.
    pub trace_async: bool,
}

/// The E3 forecast settings of the symmetric_pd cost function, shared with the
/// co_schedule path ([`crate::trade_e3::TradeE3Config`]). Two values are
/// equal when they hold the same bound file (sha256) with the same mode,
/// level, rq margin, shadow path and push settings.
#[derive(Debug, Clone)]
pub struct SymmetricPdTradeE3(pub std::sync::Arc<crate::trade_e3::TradeE3Config>);

impl PartialEq for SymmetricPdTradeE3 {
    fn eq(&self, other: &Self) -> bool {
        let (a, b) = (&*self.0, &*other.0);
        let push = |c: &crate::trade_e3::TradeE3Config| {
            c.push
                .as_ref()
                .map(|p| (p.port, p.mode.clone(), p.level.clone()))
        };
        std::sync::Arc::ptr_eq(&self.0, &other.0)
            || (a.bound_sha256 == b.bound_sha256
                && a.mode == b.mode
                && a.level == b.level
                && a.rq_margin_s == b.rq_margin_s
                && a.shadow_path == b.shadow_path
                && a.next_pass_lag == b.next_pass_lag
                && a.frozen_fit == b.frozen_fit
                && push(a) == push(b))
    }
}

/// `--symmetric-pd-gwq-auto`: the settings of the symmetric_pd global waiting queue.
/// The CLI sets only the default; tests build shorter guards in code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymmetricPdGwqAutoConfig {
    /// A request held this long (ms) goes to its TRADE argmin P, also
    /// without room: a safety bound (300000), not a tuned value.
    pub max_wait_ms: u64,
    /// `--symmetric-pd-gwq-room` (default `tokens`): `predicted` also gives a P
    /// room while its E3 drain is at most its lead C / R + T_react (needs
    /// [`SymmetricPdCostFunctionConfig::trade_e3`]).
    pub room: crate::global_queue::GlobalQueueRoom,
}

impl Default for SymmetricPdGwqAutoConfig {
    fn default() -> Self {
        Self {
            max_wait_ms: crate::global_queue::DEFAULT_HRRN_MAX_WAIT_MS,
            room: crate::global_queue::GlobalQueueRoom::Tokens,
        }
    }
}

/// `--symmetric-pd-import-cache-policy`: what the D-leg engine does with the
/// imported prompt KV when the request finishes (`kv_import.cache_policy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SymmetricPdImportCachePolicy {
    /// `gpu_only` when the D-leg worker has prefill weight 0, `retain`
    /// otherwise. Default when decode host return is disabled.
    #[value(name = "auto")]
    Auto,
    /// Insert into the radix tree and follow the engine's hicache write policy.
    #[value(name = "retain")]
    Retain,
    /// Insert into the radix tree, skip the host write-through.
    #[value(name = "gpu-only")]
    GpuOnly,
    /// Do not insert; free the imported pages at finish.
    #[value(name = "drop")]
    Drop,
}

/// `--symmetric-pd-d-select`: how the symmetric P/D D-leg is chosen once P is fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SymmetricPdDSelect {
    /// Score over `(p, d)` for the chosen `p` (default).
    #[value(name = "score")]
    Score,
    /// Round robin over the decode-capable workers (`--pd-decode-policy
    /// round-robin` equivalent).
    #[value(name = "round-robin")]
    RoundRobin,
    /// Longest cached prefix among the decode-capable workers, with the
    /// `cache-first` imbalance and threshold fallbacks on decode load.
    #[value(name = "cache-first")]
    CacheFirst,
}

impl From<SymmetricPdDSelect> for crate::symmetric_pd::placement::DSelect {
    fn from(v: SymmetricPdDSelect) -> Self {
        match v {
            SymmetricPdDSelect::Score => Self::Score,
            SymmetricPdDSelect::RoundRobin => Self::RoundRobin,
            SymmetricPdDSelect::CacheFirst => Self::CacheFirst,
        }
    }
}

/// symmetric P/D router configuration. `Some` on
/// `ModelConfig::symmetric_pd` exactly when `--policy symmetric_pd`.
#[derive(Debug, Clone, PartialEq)]
pub struct SymmetricPdConfig {
    /// Score weights / normalisation of the pure `place()` function.
    pub placement: crate::symmetric_pd::placement::PlacementConfig,
    /// Statically configured symmetric workers (e.g. 4 prefill-leaning + 4
    /// decode-leaning). Legacy Prefill / Decode workers also take part
    /// with `{1,0}` / `{0,1}` weights.
    pub workers: Vec<SymmetricPdWorkerSpec>,
    /// Hold-queue cap (ms). A request waits at most this long for a P-leg
    /// candidate whose backlog drops under `max_uncached_tokens`; then it is
    /// force-placed. Seconds-level by design: default 2000,
    /// never above [`SYMMETRIC_PD_MAX_QUEUE_WAIT_MS_CAP`].
    pub max_queue_wait_ms: u64,
    /// Backlog (pending uncached prefill tokens) above which every P-leg
    /// candidate counts as overloaded and the request holds.
    pub max_uncached_tokens: u64,
    /// `/v1/loads` poll period per worker (ms).
    pub load_poll_ms: u64,
    /// After placement, inject the peer-kv `kv_pull_hint` into the
    /// P-leg body when the indexer's best owner (host tier, gated by the
    /// `--kv-*` hint thresholds) is a different engine than the P-leg.
    pub pull_hint: bool,
    /// Routing control: replay a reference run's
    /// per-request (P, D) choices from this JSON table (see
    /// `symmetric_pd::DecisionReplay`). `None` = live selection.
    pub decision_replay: Option<std::path::PathBuf>,
    /// Request header carrying the session routing key that indexes the
    /// replay table (the strict-replay client sends `X-SMG-Routing-Key`).
    pub decision_replay_header: String,
    /// `kv_import.cache_policy` sent on the D-leg.
    pub import_cache_policy: SymmetricPdImportCachePolicy,
    /// Optional deferred D-host to selected-P-host return command. Independent
    /// of directory lookup; every participating engine must enable peer HiCache.
    pub decode_host_return: Option<DecodeHostReturnConfig>,
    /// Allow an external controller to atomically replace symmetric role assignments.
    pub external_role_control: bool,
    /// Track versioned trajectory CPU KV retention without changing placement.
    pub kv_session_retention: bool,
    /// Optional engine API/admin key file for background retention controls.
    pub kv_session_key_file: Option<std::path::PathBuf>,
    /// With external role control: a worker reassigned from decode to prefill
    /// takes no new legs for up to this long while its D-legs finish, then has
    /// the rest migrated and takes prefill legs. `None`: it takes them at once.
    pub role_drain_timeout: Option<std::time::Duration>,
    /// `--symmetric-pd-p-select gateway-cache-aware`: the gateway cache-aware
    /// rule (`--gateway-ca-*` flags) picks the P-leg before placement.
    /// `None` for every other P selection.
    pub gateway_cache_aware: Option<GatewayCacheAwareConfig>,
    /// `--symmetric-pd-p-select cost-function`: the TRADE cost function picks the
    /// P-leg (and the pull) before placement. `None` for every other P
    /// selection.
    pub cost_function: Option<SymmetricPdCostFunctionConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeHostReturnConfig {
    /// Canonical worker URL -> explicitly configured rank-0 peer servicer.
    pub servicers: std::collections::BTreeMap<String, String>,
    pub timeout_ms: u64,
    /// Maximum newly generated suffix tokens, not the full prompt length.
    pub max_tokens: u64,
}

pub fn default_symmetric_pd_decision_replay_header() -> String {
    "X-SMG-Routing-Key".to_string()
}

pub const SYMMETRIC_PD_MAX_QUEUE_WAIT_MS_CAP: u64 = 30_000;

pub fn default_symmetric_pd_max_queue_wait_ms() -> u64 {
    2_000
}
pub fn default_symmetric_pd_max_uncached_tokens() -> u64 {
    10_240
}
pub fn default_symmetric_pd_load_poll_ms() -> u64 {
    1_000
}
pub fn default_symmetric_pd_unknown_expire_secs() -> u64 {
    60
}

impl Default for SymmetricPdConfig {
    fn default() -> Self {
        Self {
            placement: Default::default(),
            workers: Vec::new(),
            max_queue_wait_ms: default_symmetric_pd_max_queue_wait_ms(),
            max_uncached_tokens: default_symmetric_pd_max_uncached_tokens(),
            load_poll_ms: default_symmetric_pd_load_poll_ms(),
            pull_hint: false,
            decision_replay: None,
            decision_replay_header: default_symmetric_pd_decision_replay_header(),
            import_cache_policy: SymmetricPdImportCachePolicy::Auto,
            decode_host_return: None,
            external_role_control: false,
            kv_session_retention: false,
            kv_session_key_file: None,
            role_drain_timeout: None,
            gateway_cache_aware: None,
            cost_function: None,
        }
    }
}

impl SymmetricPdConfig {
    /// The static spec for `url`, if the operator listed it.
    pub fn worker_spec_for(&self, url: &str) -> Option<&SymmetricPdWorkerSpec> {
        let want = url.trim().trim_end_matches('/');
        self.workers.iter().find(|w| w.normalized_url() == want)
    }
}

/// External KV Indexer client settings.
#[derive(Debug, Clone)]
pub struct KvIndexerEndpointConfig {
    pub url: String,
    pub query_timeout_ms: u64,
    pub query_max_inflight: usize,
}

/// Per-model cache-aware tuning.
#[derive(Debug, Clone)]
pub struct CacheAwareConfig {
    /// Optional Dynamo HTTP prefix source; no KV pull-hint injection.
    pub dynamo: Option<crate::kv_hint::KvHintConfig>,
    /// Lower bound on `matched_blocks / total_blocks` for the tree match
    /// to win the selection. Below this, the policy falls back to
    /// min-load. Default 0.5 — a half-cached prompt is still a strong
    /// signal but not so weak that random hash collisions could trigger
    /// affinity to an arbitrary worker.
    pub cache_threshold: f32,
    /// Absolute load spread (`max - min`) above which the cache check is
    /// skipped in favour of min-load. Default 32 — picked to dominate
    /// over typical batch-of-8 effect.
    pub balance_abs_threshold: usize,
    /// Multiplicative load spread (`max > min * balance_rel_threshold`)
    /// that the absolute check is gated on. Default 1.1 — 10 % relative
    /// difference triggers re-balancing.
    pub balance_rel_threshold: f32,
    /// Optional external KV Indexer client configuration. When configured, it
    /// replaces the local ZMQ radix tree as the cache signal.
    pub kv_indexer_endpoint: Option<KvIndexerEndpointConfig>,
}

impl Default for CacheAwareConfig {
    fn default() -> Self {
        Self {
            cache_threshold: default_cache_threshold(),
            dynamo: None,
            balance_abs_threshold: default_balance_abs(),
            balance_rel_threshold: default_balance_rel(),
            kv_indexer_endpoint: None,
        }
    }
}

fn default_cache_threshold() -> f32 {
    0.5
}
fn default_balance_abs() -> usize {
    32
}
fn default_balance_rel() -> f32 {
    1.1
}

/// Default routing-key header for the sticky policy. The `x-sgl-` prefix
/// matches the router's other emitted/consumed metadata headers
/// (`x-sgl-decode-url`, `x-sgl-router-error-code`).
pub const DEFAULT_STICKY_HEADER: &str = "x-sgl-routing-key";

/// Per-model sticky-session tuning. Built from the `--routing-key-header`
/// / `--sticky-*` flags by [`crate::config::cli::Cli::into_config`], which
/// also validates that `header_name` parses as an HTTP header name and
/// that `fallback_policy` is one of the dependency-free policies.
#[derive(Debug, Clone)]
pub struct StickyConfig {
    /// Request header carrying the routing key. Validated to parse as a
    /// `http::HeaderName` at config-build time.
    pub header_name: String,
    /// Policy used to pick a worker when a request has no routing key, and
    /// to pick the initial worker when a new key is first seen. One of
    /// `round_robin` / `random` / `power_of_two` / `load_based` — the
    /// dependency-free policies the factory can build standalone (no
    /// `HashTree` / tokenizer / ZMQ feed). `cache_aware_zmq` and `sticky`
    /// are rejected at config-build time.
    pub fallback_policy: PolicyKind,
    /// Evict an assignment after it has been idle (unreferenced) this many
    /// seconds. Bounds the map against unbounded routing-key cardinality.
    pub idle_secs: u64,
    /// Wall-clock cadence of the background eviction sweep.
    pub eviction_interval_secs: u64,
}

pub fn default_sticky_idle_secs() -> u64 {
    600
}
pub fn default_sticky_eviction_interval_secs() -> u64 {
    60
}

impl Default for StickyConfig {
    fn default() -> Self {
        Self {
            header_name: DEFAULT_STICKY_HEADER.to_string(),
            fallback_policy: PolicyKind::RoundRobin,
            idle_secs: default_sticky_idle_secs(),
            eviction_interval_secs: default_sticky_eviction_interval_secs(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Consecutive failures required before the breaker opens. Encoded
    /// as `NonZeroU32` so `--cb-threshold 0` (which would open the
    /// breaker before any failure) is rejected at CLI-parse time rather
    /// than silently behaving as "always open".
    pub threshold: NonZeroU32,
    pub cool_down_secs: u64,
}

/// Default circuit-breaker cool-down, applied when `--cb-threshold` is
/// set without an explicit `--cb-cool-down-secs`.
pub fn default_cb_cool_down() -> u64 {
    30
}

#[derive(Debug, Clone)]
pub enum DiscoveryBackend {
    StaticUrls(StaticUrlsDiscoveryConfig),
    K8s(K8sDiscoveryConfig),
}

/// Fixed list of worker URLs. Each URL is registered once at startup;
/// `mode`, `model_ids`, and `bootstrap_port` are resolved per-worker
/// from `/server_info` (see [`crate::workers::introspect`]).
///
/// No file watcher, no hot-reload: topology change requires a restart.
#[derive(Debug, Clone)]
pub struct StaticUrlsDiscoveryConfig {
    pub urls: Vec<String>,
}

/// Configuration for the Kubernetes `EndpointSlice` discovery backend.
/// Built from the `--service-discovery*` / `--selector` / `--prefill-selector`
/// / `--decode-selector` flags by [`crate::config::cli::Cli::build_discovery`].
///
/// Two operating modes, distinguished by which selector flags are set:
///
/// 1. **Plain** — all matched workers share the same role:
///    `--service-discovery-namespace default --selector app=sglang`
///
/// 2. **PD disaggregation** — prefill and decode workers are separated by
///    different selectors:
///    `--service-discovery-namespace default
///    --prefill-selector app=sglang,role=prefill
///    --decode-selector app=sglang,role=decode`
///
/// In PD mode, the selectors drive **slice-classification** (which
/// EndpointSlices feed the prefill pool vs the decode pool). The actual
/// `WorkerMode` and `bootstrap_port` for each worker are filled in by
/// the worker manager from each worker's `/server_info` introspection,
/// so PD works without any pod-level annotations — see
/// [`crate::workers::introspect`] for the `disaggregation_mode` and
/// `disaggregation_bootstrap_port` extraction.
///
/// [`resolve_mode`] validates the selector flags and produces the
/// resolved [`K8sDiscoveryMode`] once, at construction in
/// [`crate::config::cli::Cli::build_discovery`] — so an invalid selector
/// combination is unrepresentable here.
#[derive(Debug, Clone)]
pub struct K8sDiscoveryConfig {
    pub namespace: String,
    /// Resolved + validated selector mode (plain vs PD).
    pub mode: K8sDiscoveryMode,
}

/// Resolved discovery mode, produced by [`resolve_mode`] from the CLI
/// selector flags and stored on [`K8sDiscoveryConfig`].
///
/// The discovery backend uses this to:
/// * pick the server-side `LIST` label selector (Plain: the single selector;
///   PD: empty, with classification done client-side per slice), and
/// * assign each `EndpointSlice` a [`crate::discovery::WorkerMode`] in
///   `extract_workers`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum K8sDiscoveryMode {
    /// One global label selector; every matched EndpointSlice becomes a
    /// `WorkerMode::Plain` worker.
    Plain { label_selector: String },
    /// Two label selectors; an EndpointSlice's labels are matched against
    /// each to classify it as `WorkerMode::Prefill` or `WorkerMode::Decode`.
    PdDisaggregation {
        prefill_selector: String,
        decode_selector: String,
    },
}

/// Error returned by [`resolve_mode`] when the selector combination is
/// invalid.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("discovery.k8s requires either `label_selector` (plain) or both `prefill_selector` and `decode_selector` (PD); none were set")]
    NoSelector,
    #[error("discovery.k8s: `label_selector` (plain) and `prefill_selector`/`decode_selector` (PD) are mutually exclusive — set one or the other, not both")]
    MixedModes,
    #[error("discovery.k8s: PD mode requires BOTH `prefill_selector` and `decode_selector`")]
    PartialPdSelectors,
    #[error(
        "discovery.k8s: {selector}_selector `{value}` uses unsupported syntax — \
         only equality terms (`key=value` or `key==value`) joined by `,` are accepted. \
         Set-based operators (`in`, `notin`), presence tests, and `!=` silently match \
         zero endpoints at runtime and are rejected at startup."
    )]
    UnsupportedSelectorGrammar {
        selector: &'static str,
        value: String,
    },
    #[error(
        "discovery.k8s: PD `{selector}_selector` is empty (or only whitespace/commas) — \
         it would match every EndpointSlice, and since classify_mode checks prefill before \
         decode, the opposite role's pool would stay empty. Set non-empty equality terms \
         distinguishing the two roles."
    )]
    EmptyPdSelector { selector: &'static str },
    #[error(
        "discovery.k8s: `prefill_selector` and `decode_selector` resolve to the same set \
         of equality terms — classify_mode would tag every matching slice as Prefill and \
         leave the decode pool empty. The two selectors must differ."
    )]
    IdenticalPdSelectors,
}

/// Returns `true` when `selector` has zero non-empty terms after
/// trimming and splitting on `,`. `labels_match_selector` then returns
/// `true` for every label set, which is the "matches everything"
/// degenerate case PD mode must reject.
fn is_selector_empty(selector: &str) -> bool {
    selector.split(',').all(|t| t.trim().is_empty())
}

/// Canonicalize a comma-separated equality selector to a sorted list of
/// parsed `(key, value)` tuples. Comparison happens at the parsed-term
/// level — *not* the raw string level — because `labels_match_selector`
/// already strips whitespace and treats `key=value` and `key==value` as
/// the same equality test. Comparing raw strings would let
/// `"app=sglang"` vs `"app==sglang"` (and `"app = sglang"` vs
/// `"app=sglang"`) past the identical-selector check, even though
/// `classify_mode` would treat them identically at runtime — exactly
/// the silent decode-pool-empty failure mode this check exists to
/// prevent.
///
/// Returns an empty `Vec` for selectors with no parseable terms
/// (whitespace-only, comma-only, or any term that doesn't match the
/// `key=value` / `key==value` grammar). Callers must run
/// [`is_equality_selector`] before this to surface malformed
/// selectors as `UnsupportedSelectorGrammar`.
fn canonical_selector(selector: &str) -> Vec<(String, String)> {
    let mut terms: Vec<(String, String)> = selector
        .split(',')
        .filter_map(|raw| {
            let term = raw.trim();
            if term.is_empty() {
                return None;
            }
            // Mirror `labels_match_selector`: prefer the `==` alias so a
            // term like `key==value` parses to `(key, value)` instead of
            // `(key, =value)`.
            let (k, v) = term.split_once("==").or_else(|| term.split_once('='))?;
            Some((k.trim().to_string(), v.trim().to_string()))
        })
        .collect();
    terms.sort();
    terms
}

/// Returns `true` when `selector` parses as a comma-separated equality
/// selector — every term has the shape `key=value` or `key==value`.
/// See [`ConfigError::UnsupportedSelectorGrammar`] for rationale.
fn is_equality_selector(selector: &str) -> bool {
    for term in selector.split(',') {
        let term = term.trim();
        if term.is_empty() {
            // Treat lone trailing commas / whitespace as fine; the runtime
            // splitter ignores empty terms.
            continue;
        }
        if let Some((k, _)) = term.split_once("==") {
            if k.trim().is_empty() {
                return false;
            }
            continue;
        }
        if let Some((k, _value)) = term.split_once('=') {
            // Reject `!=` (rendered as `key!` + `=value` by split_once).
            // Empty value is legal in K8s — `label_selector = "tier="`
            // matches pods with `tier=""` — so we don't constrain it.
            if k.trim().is_empty() || k.trim().ends_with('!') {
                return false;
            }
            continue;
        }
        // No `=` at all → set-based operator, presence test, or garbage.
        return false;
    }
    true
}

/// Validate the selector combination and return the resolved
/// [`K8sDiscoveryMode`]. Called once at construction by
/// [`crate::config::cli::Cli::build_discovery`], so an invalid
/// combination can never be stored on a [`K8sDiscoveryConfig`].
pub fn resolve_mode(
    label_selector: Option<&str>,
    prefill_selector: Option<&str>,
    decode_selector: Option<&str>,
) -> Result<K8sDiscoveryMode, ConfigError> {
    match (label_selector, prefill_selector, decode_selector) {
        (Some(label), None, None) => {
            // Plain mode pushes `label` to the K8s API as the
            // server-side `labelSelector` of the EndpointSlice
            // watcher (`watcher::Config::default().labels(&label)`
            // in `discovery::k8s::spawn`). K8s itself parses the
            // full label-selector grammar — equality, set-based
            // (`in` / `notin`), presence (`key` / `!key`), and
            // `!=` — and rejects malformed selectors at
            // watch-start time. So we don't grammar-check `label`
            // here and let the K8s API be the syntax authority. PD
            // mode, in contrast, evaluates selectors client-side via
            // `labels_match_selector` which only understands
            // equality — so PD selectors are still grammar-checked
            // below.
            Ok(K8sDiscoveryMode::Plain {
                label_selector: label.to_string(),
            })
        }
        (None, Some(prefill), Some(decode)) => {
            // Both selectors validated individually so the operator
            // sees which one is malformed. WorkerMode + bootstrap_port
            // for each prefill pod are filled in by the worker
            // manager from each worker's `/server_info` — these
            // selectors only drive client-side classification per
            // EndpointSlice (see `classify_mode` in discovery/k8s.rs).
            if !is_equality_selector(prefill) {
                return Err(ConfigError::UnsupportedSelectorGrammar {
                    selector: "prefill",
                    value: prefill.to_string(),
                });
            }
            if !is_equality_selector(decode) {
                return Err(ConfigError::UnsupportedSelectorGrammar {
                    selector: "decode",
                    value: decode.to_string(),
                });
            }
            // Empty PD selector matches every EndpointSlice at
            // runtime; combined with classify_mode's prefill-first
            // ordering, an empty selector would silently funnel all
            // workers into one role. Reject up front.
            if is_selector_empty(prefill) {
                return Err(ConfigError::EmptyPdSelector {
                    selector: "prefill",
                });
            }
            if is_selector_empty(decode) {
                return Err(ConfigError::EmptyPdSelector { selector: "decode" });
            }
            // Identical selectors degrade the same way as an empty
            // one: every slice matches both, prefill wins, decode
            // stays empty.
            if canonical_selector(prefill) == canonical_selector(decode) {
                return Err(ConfigError::IdenticalPdSelectors);
            }
            Ok(K8sDiscoveryMode::PdDisaggregation {
                prefill_selector: prefill.to_string(),
                decode_selector: decode.to_string(),
            })
        }
        (None, None, None) => Err(ConfigError::NoSelector),
        (None, Some(_), None) | (None, None, Some(_)) => Err(ConfigError::PartialPdSelectors),
        (Some(_), _, _) => Err(ConfigError::MixedModes),
    }
}

#[cfg(test)]
mod k8s_discovery_config_tests {
    use super::*;

    #[test]
    fn mode_constructs_pd_disaggregation_from_prefill_and_decode_selectors() {
        // K8s PD now works without per-pod annotations: each worker's
        // `/server_info` carries `disaggregation_bootstrap_port`, and the
        // worker manager applies it post-discovery. The K8s config layer's
        // job is just to validate the selector combination.
        let m = resolve_mode(None, Some("app=sglang,role=p"), Some("app=sglang,role=d"))
            .expect("PD mode is now valid");
        assert_eq!(
            m,
            K8sDiscoveryMode::PdDisaggregation {
                prefill_selector: "app=sglang,role=p".to_string(),
                decode_selector: "app=sglang,role=d".to_string(),
            }
        );
    }

    #[test]
    fn mode_pd_rejects_set_based_prefill_selector() {
        // Both PD selectors get the same equality-only grammar check as
        // the plain label_selector. A set-based prefill selector would
        // silently match zero pods at runtime → fail-fast at load.
        let err =
            resolve_mode(None, Some("app in (sglang, vllm)"), Some("app=sglang")).unwrap_err();
        assert!(
            matches!(
                err,
                ConfigError::UnsupportedSelectorGrammar {
                    selector: "prefill",
                    ..
                },
            ),
            "expected UnsupportedSelectorGrammar(prefill), got {err:?}",
        );
    }

    #[test]
    fn mode_pd_rejects_set_based_decode_selector() {
        let err =
            resolve_mode(None, Some("app=sglang"), Some("app in (sglang, vllm)")).unwrap_err();
        assert!(
            matches!(
                err,
                ConfigError::UnsupportedSelectorGrammar {
                    selector: "decode",
                    ..
                },
            ),
            "expected UnsupportedSelectorGrammar(decode), got {err:?}",
        );
    }

    #[test]
    fn mode_accepts_plain_with_equality_selector() {
        let m = resolve_mode(Some("app=sglang"), None, None).unwrap();
        assert_eq!(
            m,
            K8sDiscoveryMode::Plain {
                label_selector: "app=sglang".to_string()
            }
        );
    }

    /// Plain mode pushes its selector to the K8s API server-side
    /// (`watcher::Config::default().labels(&selector)` in
    /// `discovery::k8s::spawn`), so the full K8s label-selector grammar
    /// — including set-based operators like `app in (a,b)` — is
    /// supported and must not be grammar-checked at startup. PD mode
    /// (checked client-side) is the opposite; see the PD tests below.
    #[test]
    fn mode_accepts_set_based_selector_in_plain_mode() {
        let m = resolve_mode(Some("app in (sglang,sglang-small)"), None, None)
            .expect("plain mode must accept set-based selectors");
        assert_eq!(
            m,
            K8sDiscoveryMode::Plain {
                label_selector: "app in (sglang,sglang-small)".to_string(),
            }
        );
    }

    /// `notin`, presence (`key`), absence (`!key`), and inequality (`!=`)
    /// are all valid K8s server-side selector grammar — plain mode must
    /// pass them through.
    #[test]
    fn mode_accepts_other_set_based_forms_in_plain_mode() {
        for raw in [
            "app notin (vllm,trtllm)",
            "tier",
            "!deprecated",
            "tier!=canary",
        ] {
            let m = resolve_mode(Some(raw), None, None)
                .unwrap_or_else(|e| panic!("plain mode must accept `{raw}`, got {e:?}"));
            assert_eq!(
                m,
                K8sDiscoveryMode::Plain {
                    label_selector: raw.to_string(),
                },
                "selector roundtrip mismatch for `{raw}`",
            );
        }
    }

    /// PD mode evaluates selectors *client-side* via
    /// `labels_match_selector`, which only handles equality. A set-based
    /// PD selector would silently match zero pods → fail-fast at load.
    /// Pins the plain-server-side / PD-client-side asymmetry: relaxing
    /// the grammar check for plain (see `mode_accepts_set_based_*`
    /// above) must not accidentally relax it for PD selectors. Uses
    /// `notin` so this test covers a different set-based form than
    /// `mode_pd_rejects_set_based_prefill_selector` (which uses `in`)
    /// — both must keep failing.
    #[test]
    fn mode_pd_rejects_notin_prefill_selector() {
        let err =
            resolve_mode(None, Some("app notin (vllm, trtllm)"), Some("app=sglang")).unwrap_err();
        assert!(
            matches!(
                err,
                ConfigError::UnsupportedSelectorGrammar {
                    selector: "prefill",
                    ..
                },
            ),
            "expected UnsupportedSelectorGrammar(prefill), got {err:?}",
        );
    }

    #[test]
    fn mode_accepts_comma_separated_equality_terms() {
        // The canonical Plain-mode selector form: `key1=v1,key2=v2`.
        let m = resolve_mode(Some("app=sglang,zone=us-east"), None, None).unwrap();
        assert_eq!(
            m,
            K8sDiscoveryMode::Plain {
                label_selector: "app=sglang,zone=us-east".to_string()
            }
        );
    }

    #[test]
    fn mode_rejects_when_no_selector_is_set() {
        let err = resolve_mode(None, None, None).unwrap_err();
        assert!(matches!(err, ConfigError::NoSelector), "got {err:?}");
    }

    #[test]
    fn mode_rejects_mixed_plain_and_pd_selectors() {
        let err = resolve_mode(
            Some("app=sglang"),
            Some("role=prefill"),
            Some("role=decode"),
        )
        .unwrap_err();
        assert!(matches!(err, ConfigError::MixedModes), "got {err:?}");
    }

    #[test]
    fn mode_rejects_partial_pd_selectors() {
        let err = resolve_mode(None, Some("role=prefill"), None).unwrap_err();
        assert!(
            matches!(err, ConfigError::PartialPdSelectors),
            "got {err:?}"
        );
        let err = resolve_mode(None, None, Some("role=decode")).unwrap_err();
        assert!(
            matches!(err, ConfigError::PartialPdSelectors),
            "got {err:?}"
        );
    }

    /// Empty plain `label_selector` is valid — matches every
    /// EndpointSlice in the namespace (documented K8s behavior; the
    /// operator opts in by setting plain mode at all).
    #[test]
    fn mode_accepts_empty_plain_label_selector() {
        let m = resolve_mode(Some(""), None, None).unwrap();
        assert_eq!(
            m,
            K8sDiscoveryMode::Plain {
                label_selector: String::new()
            }
        );
    }

    /// PD mode is the *opposite* of plain: an empty selector would match
    /// every EndpointSlice, and since `classify_mode` checks prefill
    /// before decode, an empty `prefill_selector` would classify
    /// everything as Prefill — decode pool stays empty and the resolver
    /// surfaces the wrong `no_decode_workers_available` error. Fail-fast
    /// at config load.
    #[test]
    fn mode_pd_rejects_empty_prefill_selector() {
        let err = resolve_mode(None, Some(""), Some("role=decode")).unwrap_err();
        assert!(
            matches!(
                err,
                ConfigError::EmptyPdSelector {
                    selector: "prefill"
                },
            ),
            "expected EmptyPdSelector(prefill), got {err:?}",
        );
    }

    #[test]
    fn mode_pd_rejects_empty_decode_selector() {
        let err = resolve_mode(None, Some("role=prefill"), Some("")).unwrap_err();
        assert!(
            matches!(err, ConfigError::EmptyPdSelector { selector: "decode" },),
            "expected EmptyPdSelector(decode), got {err:?}",
        );
    }

    /// Whitespace-only / comma-only PD selector parses to zero terms in
    /// `labels_match_selector` and matches every slice at runtime — same
    /// failure mode as a literal empty string.
    #[test]
    fn mode_pd_rejects_whitespace_only_prefill_selector() {
        let err = resolve_mode(None, Some("  ,  "), Some("role=decode")).unwrap_err();
        assert!(
            matches!(
                err,
                ConfigError::EmptyPdSelector {
                    selector: "prefill"
                },
            ),
            "expected EmptyPdSelector(prefill), got {err:?}",
        );
    }

    /// Identical prefill and decode selectors degrade silently: every
    /// slice matches both, but `classify_mode` returns `Prefill` first,
    /// so the decode pool stays empty.
    #[test]
    fn mode_pd_rejects_identical_prefill_and_decode_selectors() {
        let err = resolve_mode(None, Some("app=sglang"), Some("app=sglang")).unwrap_err();
        assert!(
            matches!(err, ConfigError::IdenticalPdSelectors),
            "expected IdenticalPdSelectors, got {err:?}",
        );
    }

    /// Trailing whitespace must not be a loophole that bypasses the
    /// identical-selector check.
    #[test]
    fn mode_pd_rejects_identical_selectors_under_whitespace_normalization() {
        let err = resolve_mode(None, Some("app=sglang"), Some("  app=sglang  ")).unwrap_err();
        assert!(
            matches!(err, ConfigError::IdenticalPdSelectors),
            "expected IdenticalPdSelectors, got {err:?}",
        );
    }

    /// `labels_match_selector` accepts both `key=value` and `key==value`
    /// for equality and parses them to the same `(key, value)` tuple.
    /// Two selectors that differ only in this alias choice are runtime-
    /// equivalent — they'd match the same EndpointSlices, then
    /// `classify_mode`'s prefill-first ordering would funnel every slice
    /// into Prefill, leaving decode empty. The check must canonicalize
    /// at the term level (parsed `(key, value)` tuples), not the raw
    /// string level.
    #[test]
    fn mode_pd_rejects_identical_selectors_under_eq_alias() {
        let err = resolve_mode(None, Some("app=sglang"), Some("app==sglang")).unwrap_err();
        assert!(
            matches!(err, ConfigError::IdenticalPdSelectors),
            "expected IdenticalPdSelectors, got {err:?}",
        );
    }

    /// Inner whitespace inside a term (`"app = sglang"`) is the same
    /// label as no whitespace (`"app=sglang"`) — the runtime
    /// `labels_match_selector` trims key and value independently
    /// (see `key.trim()` / `expected.trim()` in `k8s.rs`). Canonical
    /// form must agree.
    #[test]
    fn mode_pd_rejects_identical_selectors_under_inner_whitespace() {
        let err = resolve_mode(None, Some("app=sglang"), Some("app =  sglang")).unwrap_err();
        assert!(
            matches!(err, ConfigError::IdenticalPdSelectors),
            "expected IdenticalPdSelectors, got {err:?}",
        );
    }

    /// Term order doesn't matter for label matching, so `"a=1,b=2"` and
    /// `"b=2,a=1"` must be treated as identical. (Implied by the sort
    /// in `canonical_selector`, but pinned explicitly so a future
    /// "preserve user order for diagnostics" refactor can't silently
    /// reintroduce the silent-failure bug.)
    #[test]
    fn mode_pd_rejects_identical_selectors_under_term_order_permutation() {
        let err =
            resolve_mode(None, Some("role=p,app=sglang"), Some("app=sglang,role=p")).unwrap_err();
        assert!(
            matches!(err, ConfigError::IdenticalPdSelectors),
            "expected IdenticalPdSelectors, got {err:?}",
        );
    }

    /// Sanity: two selectors that genuinely differ at the term level
    /// must still pass validation — the canonicalizer must not be so
    /// aggressive that it false-positives on legitimate PD configs.
    #[test]
    fn mode_pd_accepts_truly_distinct_selectors() {
        let m = resolve_mode(
            None,
            Some("app=sglang,role=prefill"),
            Some("app=sglang,role=decode"),
        )
        .expect("distinct selectors must validate");
        assert!(matches!(m, K8sDiscoveryMode::PdDisaggregation { .. }));
    }
}
