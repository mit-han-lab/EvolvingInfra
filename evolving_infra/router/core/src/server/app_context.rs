// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Wires shared dependencies for HTTP handlers: configuration, tokenizers, proxy,
//! workers, routing policies, cache lookup, metrics and optional managed execution.
//! Constructors attach observability and start configured scheduler background
//! services; the readiness flag reports local startup status only.

use crate::backend::kv_sessions::KvSessions;
use crate::config::Config;
use crate::config::PdDecodePolicy;
use crate::discovery::ModelId;
use crate::policies::decode_po2_smg::{SmgLoadArm, SmgLoadGuard, SmgPowerOfTwo};
use crate::policies::decode_tokens::{DecodeTokenBalancer, DecodeTokenLog};
use crate::policies::registry::{PdPoolResolver, PdResolveError};
use crate::workers::{DecodeTokenCharge, Worker};

use crate::policies::active_load::ActiveLoadRegistry;
use crate::policies::kv_events::BlockSizeOracle;
use crate::policies::PolicyRegistry;
use crate::proxy::Proxy;
use crate::server::metrics::MetricsRegistry;
use crate::tokenizer::TokenizerRegistry;
use crate::workers::WorkerRegistry;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub struct AppContext {
    pub config: Config,
    pub tokenizers: Arc<TokenizerRegistry>,
    pub proxy: Arc<Proxy>,
    pub registry: Arc<WorkerRegistry>,
    pub policies: Arc<PolicyRegistry>,
    /// Per-worker active-load bookkeeping. Shared between the proxy
    /// (which mints guards on the request hot path), the cache-aware
    /// policy (which reads per-worker load when scoring candidates), and
    /// the stale-request janitor (which sweeps expired entries).
    pub active_load: Arc<ActiveLoadRegistry>,
    /// Lightweight Prometheus-format metrics registry served via
    /// `/metrics`. Shared with the chat handler (requests_total),
    /// cache-aware-zmq policy (overlap_blocks), active-load registry
    /// (active_load gauge + stale_requests_total), and PD resolver
    /// (decode_affinity_total).
    pub metrics: Arc<MetricsRegistry>,
    pub prefix_index: Option<Arc<sgl_kv_indexer::GrpcPrefixIndex>>,
    pub dynamo_prefix: Option<Arc<crate::kv_hint::KvHintInjector>>,
    pub block_size_oracle: Arc<BlockSizeOracle>,
    /// KV pull-hint injector; `Some` exactly when `--inject-kv-pull-hint`
    /// is configured. Built here (not in config) so the config stays plain
    /// data. Request handlers consult it after worker selection.
    pub kv_hint: Option<Arc<crate::kv_hint::KvHintInjector>>,
    /// symmetric P/D placement runtime (indexer feed, `/v1/loads` poller + ledger,
    /// hold queue, marks); `Some` exactly when `--policy symmetric_pd`. The
    /// chat/completions routes place through `SymmetricPdRuntime::admit` and
    /// `POST /symmetric_pd/place` serves an external ingress.
    pub symmetric_pd: Option<Arc<crate::scheduler::SymmetricPdRuntime>>,
    /// Traditional-PD per-request decision replay table; `Some` exactly when
    /// `--pd-decision-replay` is configured (`ModelConfig::pd_decision_replay`).
    /// The chat/completions routes pin (prefill, decode) from it before
    /// falling back to `Policy::select` + `decode_select`.
    pub pd_replay: Option<Arc<crate::scheduler::DecisionReplay>>,
    pub prepared_config: Option<crate::backend::execution::PreparedConfig>,
    pub executions: Arc<crate::backend::execution::Executions>,
    /// Completed-request totals for the external role controller; `Some`
    /// exactly when `--symmetric-pd-external-role-control` is set.
    pub external_workload: Option<Arc<crate::backend::workload::CompletedWorkload>>,
    pub external_role_key: Option<String>,
    /// Token-charged decode selection; `Some` exactly when
    /// `--pd-decode-policy` is `min-inflight-tokens` or `po2-tokens`.
    pub decode_tokens: Option<Arc<DecodeTokenBalancer>>,
    /// sgl-model-gateway `power_of_two` decode selection (load monitor,
    /// token-load snapshot, SMG request counts); `Some` exactly when
    /// `--pd-decode-policy power-of-two-smg`.
    pub decode_po2_smg: Option<Arc<SmgPowerOfTwo>>,
    /// Late D stage 2 (late-bind table, prefill links, bind-time choice);
    /// `Some` exactly when `--pd-late-decode choose`. The completions route
    /// sends a PD request to its prefill and to every decode candidate, and
    /// the prefill asks this for the winner when it is about to prefill.
    pub late_bind: Option<Arc<crate::late_bind::LateBind>>,
    /// Session KV retention on the PD path (`--kv-session-retention`, with
    /// `--policy gateway_cache_aware`); `Some` exactly then. The completions route
    /// registers each PD request with an `x-kv-session-context` header
    /// (P, P) and tags only the P-leg body; the registry's task sends the
    /// engine controls. `None`: the route takes no retention branch.
    /// symmetric_pd keeps its own registry in its Backend.
    pub kv_sessions: Option<Arc<KvSessions>>,
    ready: AtomicBool,
}

/// A decode pick. Under a token-charged policy it also carries the live
/// charge (hold it for the decode leg's lifetime, beside the decode
/// `LoadGuard`) and the decision-log record; both are `None` otherwise.
pub struct DecodeSelection {
    pub worker: Arc<Worker>,
    pub charge: Option<DecodeTokenCharge>,
    pub token_log: Option<DecodeTokenLog>,
}

impl AppContext {
    /// Construct HTTP handler dependencies with the default active-load registry.
    ///
    /// # Inputs
    /// `config` selects services and policies; `tokenizers`, `proxy`, `registry`
    /// and `policies` are shared serving dependencies supplied by the caller.
    ///
    /// # Outputs
    /// Returns a context initially not ready, with a default five-minute
    /// active-load timeout. Delegates to `with_active_load`, including its
    /// background-task startup and configuration-validation panic behavior.
    pub fn new(
        config: Config,
        tokenizers: Arc<TokenizerRegistry>,
        proxy: Arc<Proxy>,
        registry: Arc<WorkerRegistry>,
        policies: Arc<PolicyRegistry>,
    ) -> Self {
        Self::with_active_load(
            config,
            tokenizers,
            proxy,
            registry,
            policies,
            ActiveLoadRegistry::with_defaults(),
        )
    }

    /// Construct an [`AppContext`] with an explicit [`ActiveLoadRegistry`].
    /// Production wires the default (5-minute timeout, SystemTimeClock)
    /// via [`Self::new`]; tests that exercise the janitor pass a registry
    /// built with a `MockClock`.
    /// Wire shared serving dependencies and configured scheduler integrations.
    ///
    /// # Inputs
    /// `config` selects services/policies; `tokenizers`, `proxy`, `registry`
    /// and `policies` are shared dependencies. `active_load` supplies the caller
    /// chosen clock and expiry behavior for traditional routing request guards.
    ///
    /// # Outputs
    /// Returns a not-ready context with metrics attached to load/policy state,
    /// optional indexer clients and scheduler runtimes, and an empty execution
    /// registry. Configured schedulers may start background tasks. Panics on an
    /// invalid decision-replay table, prepared-input environment configuration
    /// or a dependency constructor failure; does not bind HTTP listeners.
    pub fn with_active_load(
        config: Config,
        tokenizers: Arc<TokenizerRegistry>,
        proxy: Arc<Proxy>,
        registry: Arc<WorkerRegistry>,
        policies: Arc<PolicyRegistry>,
        active_load: Arc<ActiveLoadRegistry>,
    ) -> Self {
        let prepared_config = crate::backend::execution::PreparedConfig::from_env()
            .expect("valid EVOLVING_INFRA_PREPARED_CONFIG");
        Self::with_role_control_inputs(
            config,
            tokenizers,
            proxy,
            registry,
            policies,
            active_load,
            prepared_config,
            None,
        )
    }

    /// [`Self::with_active_load`] with the inputs that production reads from
    /// the environment given explicitly: the prepared-input configuration
    /// (`EVOLVING_INFRA_PREPARED_CONFIG`) and, with external role control,
    /// the role controller's credential (`None`: read from
    /// `EVOLVING_INFRA_ROLE_CONTROL_TOKEN_FILE`). Tests use it to run role
    /// control without process-wide environment variables.
    #[allow(clippy::too_many_arguments)]
    pub fn with_role_control_inputs(
        config: Config,
        tokenizers: Arc<TokenizerRegistry>,
        proxy: Arc<Proxy>,
        registry: Arc<WorkerRegistry>,
        policies: Arc<PolicyRegistry>,
        active_load: Arc<ActiveLoadRegistry>,
        prepared_config: Option<crate::backend::execution::PreparedConfig>,
        role_key: Option<String>,
    ) -> Self {
        let metrics = MetricsRegistry::new();
        // Wire the per-worker active-load gauge so `sgl_router_active_load`
        // mirrors the live counter on every register / drop / sweep.
        // Without this, the metric is permanently 0 in production even
        // though the chat handler is faithfully calling `register`.
        active_load.attach_metrics(Arc::clone(&metrics));
        // Same rationale for the cache-aware-zmq policy's
        // `sgl_router_overlap_blocks`: the metrics registry is built here,
        // after the policy registry, so inject it now. No-op for policies
        // that don't emit metrics.
        policies.attach_metrics(Arc::clone(&metrics));
        // KV pull-hint injector: built once from config; handlers read it
        // per request. `None` (feature off) is the common case.
        let kv_hint = config
            .model
            .kv_hint
            .as_ref()
            .map(|cfg| Arc::new(crate::kv_hint::KvHintInjector::new(cfg)));
        // symmetric P/D: the indexer is optional (placement degrades to
        // load/role only); the runtime reuses the kv-hint indexer config.
        let symmetric_pd = config.model.symmetric_pd.as_ref().map(|f| {
            crate::scheduler::SymmetricPdRuntime::spawn(
                f.clone(),
                config.model.kv_hint.as_ref(),
                Arc::clone(&registry),
                ModelId(config.model.id.clone()),
            )
        });
        // Traditional-PD decision replay: a bad table is a configuration
        // error for a controlled experiment, so fail loudly at startup.
        let pd_replay = config.model.pd_decision_replay.as_ref().map(|r| {
            Arc::new(
                crate::scheduler::DecisionReplay::load(&r.table)
                    .unwrap_or_else(|e| panic!("--pd-decision-replay: {e}")),
            )
        });
        let dynamo_prefix = config
            .model
            .cache_aware
            .as_ref()
            .and_then(|c| c.dynamo.as_ref())
            .map(|c| Arc::new(crate::kv_hint::KvHintInjector::new(c)));
        // The cost function reserves at admit. With prepared inputs the
        // execution registry owns the reservation and applies the tokenizer's
        // P-leg callbacks (upload = dispatch, terminal P = finish, uncertain
        // failure or cancel = unknown); see backend/execution.rs.
        let external = symmetric_pd
            .as_ref()
            .is_some_and(|f| f.config().external_role_control);
        assert!(
            !external || prepared_config.is_some(),
            "external role control requires prepared usage callbacks"
        );
        let external_role_key = external.then(|| {
            let key = role_key.unwrap_or_else(|| {
                let path = std::env::var_os("EVOLVING_INFRA_ROLE_CONTROL_TOKEN_FILE").expect(
                    "external role control requires EVOLVING_INFRA_ROLE_CONTROL_TOKEN_FILE",
                );
                std::fs::read_to_string(path)
                    .expect("read role controller credential")
                    .trim()
                    .to_string()
            });
            assert!(
                key.len() >= 32 && !key.chars().any(char::is_whitespace),
                "invalid role control credential"
            );
            key
        });
        let external_workload =
            external.then(|| Arc::new(crate::backend::workload::CompletedWorkload::default()));
        let decode_unresolved_expiry = prepared_config
            .as_ref()
            .and_then(crate::backend::execution::PreparedConfig::decode_unresolved_expiry);
        if prepared_config.is_some() {
            tracing::info!(target: "symmetric_pd",
                decode_unresolved_expire_s = ?decode_unresolved_expiry.map(|d| d.as_secs()),
                "prepared executions: unresolved D-leg expiry (none = never)");
        }
        let executions = Arc::new(
            match &external_workload {
                Some(workload) => {
                    crate::backend::execution::Executions::with_workload(Arc::clone(workload))
                }
                None => crate::backend::execution::Executions::default(),
            }
            .with_decode_unresolved_expiry(decode_unresolved_expiry)
            .with_silent_expiry(
                prepared_config
                    .as_ref()
                    .and_then(crate::backend::execution::PreparedConfig::silent_expiry),
            ),
        );
        let decode_tokens = DecodeTokenBalancer::for_policy(
            config.model.pd_decode_policy,
            config.model.pd_decode_tokens.clone().unwrap_or_default(),
        )
        .map(Arc::new);
        let decode_po2_smg =
            (config.model.pd_decode_policy == PdDecodePolicy::PowerOfTwoSmg).then(|| {
                SmgPowerOfTwo::spawn(
                    Arc::clone(&registry),
                    config.active_load.pd_decode_load_check_interval_secs,
                )
            });
        // Late D choose: the po2 rule and the arrival pick read the bound
        // view of the decode loads (DEFERRED candidates hold no KV).
        let late_bind = config.proxy.late_decode.choose().then(|| {
            // symmetric_pd: the owners are the prefill-capable symmetric workers
            // of the current roles, whatever their breaker state (as the PD
            // links); no power-of-two-smg policy is involved.
            if let Some(spd) = &symmetric_pd {
                let weak = Arc::downgrade(spd);
                // External role control: link every worker (a P-leg queued
                // on a worker moved to decode still binds through its link),
                // and bind only to candidates still in the decode role.
                let external = spd.config().external_role_control;
                let late = crate::late_bind::LateBind::new_symmetric(
                    config.proxy.late_decode,
                    config.model.pd_decode_tokens.clone().unwrap_or_default(),
                    Arc::clone(&registry),
                    Arc::new(move || {
                        weak.upgrade()
                            .map(|f| {
                                if external {
                                    f.role_members_all()
                                } else {
                                    f.prefill_members()
                                }
                            })
                            .unwrap_or_default()
                    }),
                );
                if external {
                    let weak = Arc::downgrade(spd);
                    late.set_decode_role_gate(Arc::new(move || {
                        weak.upgrade()
                            .map(|f| f.decode_members())
                            .unwrap_or_default()
                    }));
                }
                late.start();
                return late;
            }
            if let Some(policy) = &decode_po2_smg {
                policy.set_bound_view(true);
            }
            let late = crate::late_bind::LateBind::new(
                config.proxy.late_decode,
                config.model.pd_decode_tokens.clone().unwrap_or_default(),
                Arc::clone(&registry),
                decode_po2_smg.clone(),
            );
            late.start();
            late
        });
        // `--pd-native-decode-state`: the 100 ms poll of each decode worker's
        // observed allocation state. In choose mode it fills the late-bind
        // table's store (`fit_req` with `--pd-late-decode-fit-native` reads
        // it); with late D off it only polls and logs, as the gateway does.
        // symmetric_pd with late D choose: the decode-capable symmetric workers of
        // the current roles, whatever their breaker state (as the PD poll).
        if config.proxy.late_decode.native_state_poll {
            let store = late_bind
                .as_ref()
                .map(|l| Arc::clone(l.capacity_store()))
                .unwrap_or_default();
            match symmetric_pd
                .as_ref()
                .filter(|_| late_bind.as_ref().is_some_and(|l| l.symmetric()))
            {
                Some(spd) => {
                    let weak = Arc::downgrade(spd);
                    crate::late_fit::start_collector_over(
                        &registry,
                        store,
                        Arc::new(move |_: &WorkerRegistry| {
                            weak.upgrade()
                                .map(|f| f.decode_members())
                                .unwrap_or_default()
                        }),
                        "every decode-capable symmetric_pd worker",
                        // A symmetric engine fills the capacity fields only
                        // with SGLANG_DECODE_SELECTION_METRICS=1: say so.
                        true,
                    );
                }
                None => crate::late_fit::start_collector(&registry, store),
            }
        }
        // Session KV retention (gateway_cache_aware PD path):
        // the registry, its control task and the periodic stats line.
        let kv_sessions = config.model.pd_kv_session_retention().map(|key_file| {
            let sessions = KvSessions::spawn(key_file);
            crate::server::routes::kv_session::spawn_stats_log(&sessions);
            sessions
        });
        Self {
            config,
            tokenizers,
            proxy,
            registry,
            policies,
            active_load,
            metrics,
            prefix_index: None,
            dynamo_prefix,
            block_size_oracle: BlockSizeOracle::new(),
            kv_hint,
            symmetric_pd,
            pd_replay,
            prepared_config,
            executions,
            external_workload,
            external_role_key,
            decode_tokens,
            decode_po2_smg,
            late_bind,
            kv_sessions,
            ready: AtomicBool::new(false),
        }
    }

    /// Pick the decode peer per `--pd-decode-policy`. `charge_tokens` is
    /// this request's token charge ([`Self::decode_charge_tokens`]); only
    /// the token-charged policies read it.
    pub fn select_decode(
        &self,
        resolver: &PdPoolResolver,
        model: &ModelId,
        prefill_url: &str,
        charge_tokens: Option<u64>,
    ) -> Result<DecodeSelection, PdResolveError> {
        let plain = |worker| DecodeSelection {
            worker,
            charge: None,
            token_log: None,
        };
        if let Some(balancer) = &self.decode_tokens {
            let candidates = resolver.decode_candidates(model)?;
            let pick = balancer
                .select(&candidates, charge_tokens.unwrap_or(1))
                .ok_or(PdResolveError::NoDecodeWorkersAvailable)?;
            Ok(DecodeSelection {
                worker: pick.worker,
                charge: Some(pick.charge),
                token_log: Some(pick.log),
            })
        } else if let Some(policy) = &self.decode_po2_smg {
            let candidates = resolver.decode_candidates(model)?;
            policy
                .select(&candidates)
                .map(plain)
                .ok_or(PdResolveError::NoDecodeWorkersAvailable)
        } else {
            resolver
                .decode_select(model, prefill_url, self.config.model.pd_decode_policy)
                .map(plain)
        }
    }

    /// The SMG request-count guard for one decode leg under
    /// `--pd-decode-policy power-of-two-smg`; `(None, None)` otherwise. A
    /// non-streaming leg counts from now (SMG `execute_dual_dispatch_internal`).
    /// A streaming leg counts once the returned arm is armed — call
    /// [`SmgLoadArm::arm`] when the decode response headers arrive (SMG
    /// `create_streaming_response`). Either way the count ends when the
    /// guard drops, so hold it with the decode leg's `LoadGuard`.
    pub fn decode_smg_load(
        &self,
        decode_url: &str,
        streaming: bool,
    ) -> (Option<SmgLoadGuard>, Option<SmgLoadArm>) {
        match &self.decode_po2_smg {
            None => (None, None),
            Some(policy) if streaming => {
                let (guard, arm) = policy.count_from_headers(decode_url);
                (Some(guard), Some(arm))
            }
            Some(policy) => (Some(policy.count_from_dispatch(decode_url)), None),
        }
    }

    /// This request's decode token charge under a token-charged policy
    /// (`None` otherwise): `prompt_tokens()` plus the capped output budget
    /// from the parsed body.
    pub fn decode_charge_tokens(
        &self,
        prompt_tokens: impl FnOnce() -> u64,
        body: Option<&serde_json::Value>,
    ) -> Option<u64> {
        self.decode_tokens
            .as_ref()
            .map(|b| b.request_charge(prompt_tokens(), body))
    }

    /// Charge a decode peer pinned by `--pd-decision-replay`, so the
    /// token-charged per-decode sums still cover every dispatched request.
    pub fn charge_pinned_decode(
        &self,
        resolver: &PdPoolResolver,
        model: &ModelId,
        worker: &Arc<Worker>,
        charge_tokens: Option<u64>,
    ) -> (Option<DecodeTokenCharge>, Option<DecodeTokenLog>) {
        match (&self.decode_tokens, charge_tokens) {
            (Some(balancer), Some(tokens)) => {
                let candidates = resolver.decode_candidates(model).unwrap_or_default();
                let (charge, log) = balancer.charge_pinned(worker, &candidates, tokens);
                (Some(charge), Some(log))
            }
            _ => (None, None),
        }
    }

    /// Mark local application startup as complete.
    ///
    /// # Inputs
    /// `self` owns the atomic readiness flag.
    ///
    /// # Outputs
    /// Returns `()` after setting the flag true with relaxed ordering. This
    /// flag neither synchronizes other state nor proves worker/engine health.
    pub fn mark_ready(&self) {
        // Relaxed: this flag does not synchronize other state; readers only
        // care about eventual visibility, not happens-before with surrounding ops.
        self.ready.store(true, Ordering::Relaxed);
    }

    /// Read the local startup readiness flag.
    ///
    /// # Inputs
    /// `self` owns the atomic readiness flag.
    ///
    /// # Outputs
    /// Returns its current boolean value using relaxed ordering; performs no
    /// engine probe and does not check whether placement candidates exist.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    /// Build a minimal application context for isolated HTTP/unit tests.
    ///
    /// # Inputs
    /// No arguments. Uses placeholder model/worker configuration, empty
    /// registries and a proxy with a 60-second request timeout.
    ///
    /// # Outputs
    /// Returns a not-ready context with optional schedulers/indexers disabled.
    /// Starts no discovery or serving tasks; panics if the test proxy client
    /// cannot be constructed.
    pub fn stub() -> Self {
        Self {
            config: Config {
                server: crate::config::ServerConfig {
                    host: "x".into(),
                    port: 0,
                },
                observability: Default::default(),
                model: crate::config::ModelConfig {
                    id: "stub-model".into(),
                    tokenizer_path: "stub".into(),
                    policy: crate::config::PolicyKind::RoundRobin,
                    circuit_breaker: None,
                    cache_aware: None,
                    sticky: None,
                    kv_hint: None,
                    pd_decode_policy: PdDecodePolicy::AffinityMinLoad,
                    pd_decision_replay: None,
                    pd_decode_tokens: None,
                    symmetric_pd: None,
                    gateway_cache_aware: None,
                },
                discovery: crate::config::DiscoveryBackend::StaticUrls(
                    crate::config::StaticUrlsDiscoveryConfig {
                        urls: vec!["http://placeholder:0".into()],
                    },
                ),
                proxy: crate::config::ProxyConfig::default(),
                active_load: crate::config::ActiveLoadConfig::default(),
            },
            tokenizers: Arc::new(TokenizerRegistry::default()),
            proxy: Arc::new(Proxy::new(std::time::Duration::from_secs(60)).expect("stub proxy")),
            registry: Arc::new(WorkerRegistry::default()),
            policies: Arc::new(PolicyRegistry::default()),
            active_load: ActiveLoadRegistry::with_defaults(),
            metrics: MetricsRegistry::new(),
            prefix_index: None,
            dynamo_prefix: None,
            block_size_oracle: BlockSizeOracle::new(),
            kv_hint: None,
            symmetric_pd: None,
            pd_replay: None,
            prepared_config: None,
            executions: Arc::new(crate::backend::execution::Executions::default()),
            external_workload: None,
            external_role_key: None,
            decode_tokens: None,
            decode_po2_smg: None,
            late_bind: None,
            kv_sessions: None,
            ready: AtomicBool::new(false),
        }
    }
}
