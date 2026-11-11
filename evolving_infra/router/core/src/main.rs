// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Executable entry point for the Evolving Infra router. Resolves CLI
//! configuration, loads tokenizers, builds cache lookup and routing services,
//! starts worker discovery and HTTP serving, then coordinates shutdown.
//! The reusable scheduler, backend and HTTP modules are exported by `lib.rs`.

use anyhow::{Context, Result};
use clap::Parser;
use sgl_router::config::{Cli, LogFormat};
use std::sync::Arc;
use tokio::signal::unix::{signal, Signal, SignalKind};

/// Install the global tracing subscriber.
///
/// Idempotent: a second call returns `Ok` without panicking. When
/// `try_init` errors, some other code has already installed a subscriber,
/// so the `tracing::debug!` below is delivered through THAT subscriber —
/// no recursive init.
///
/// `format` selects the output shape: `Json` emits one JSON record per
/// line (target for production / k8s log aggregators), `Text` is the
/// human-readable default. The `RUST_LOG` environment variable always
/// wins over `default_level`.
/// Attempt to install the configured global logging subscriber.
///
/// # Inputs
/// `default_level` is the fallback filter when `RUST_LOG` is invalid or
/// absent; `format` selects JSON or text output for a successful installation.
///
/// # Outputs
/// Returns `Ok(())` even when another subscriber is already installed,
/// logging that condition through the existing subscriber. Does not replace
/// or reconfigure an existing global subscriber.
fn init_tracing(default_level: &str, format: LogFormat) -> Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_level));
    let install_result = match format {
        LogFormat::Json => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .json()
            .try_init(),
        LogFormat::Text => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .try_init(),
    };
    if let Err(e) = install_result {
        // A second install attempt; the existing subscriber is fine.
        // Surface the attempted default level so an operator can see
        // what we tried.
        tracing::debug!(
            default_level = %default_level,
            ?format,
            error = %e,
            "tracing subscriber already installed; continuing"
        );
    }
    Ok(())
}

/// Install a minimal text-format subscriber BEFORE config resolution so a
/// config-resolution error has somewhere to surface. The real subscriber
/// (driven by `Config.observability`) is installed after; the second
/// `try_init` is a no-op because a subscriber is already present.
/// The bootstrap subscriber respects `RUST_LOG` so an operator can
/// debug startup with `RUST_LOG=debug` even when configuration resolution
/// fails.
/// Provide logging before command-line configuration is resolved.
///
/// # Inputs
/// No arguments. Reads `RUST_LOG`, falling back to the `info` filter.
///
/// # Outputs
/// Returns `()` after attempting to install a text subscriber. Ignores
/// installation failure when one already exists; a successful bootstrap
/// installation remains active when later `try_init` calls are ignored.
fn install_bootstrap_subscriber() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init();
}

/// Install SIGTERM and SIGINT handlers up front so a failure here surfaces
/// before `axum::serve` starts. If installation fails (rare: container
/// without signal capability, seccomp policy), we return an error and the
/// process exits cleanly rather than running deaf to k8s termination.
/// Register Unix termination notifications before HTTP serving begins.
///
/// # Inputs
/// No arguments. Requires a Tokio runtime supporting Unix signals.
///
/// # Outputs
/// Returns `(SIGTERM stream, SIGINT stream)`, or a contextual error when
/// either handler cannot be installed. No signal is consumed here.
fn install_signal_handlers() -> Result<(Signal, Signal)> {
    let sigterm = signal(SignalKind::terminate()).context("install SIGTERM handler")?;
    let sigint = signal(SignalKind::interrupt()).context("install SIGINT handler")?;
    Ok((sigterm, sigint))
}

#[tokio::main]
/// Initialize the runtime, serve HTTP requests and clean up on shutdown.
///
/// # Inputs
/// No function arguments. Reads process CLI/environment configuration,
/// tokenizer files and optional replay/prepared-input configuration; connects
/// to configured discovery, indexer and worker services.
///
/// # Outputs
/// Returns `Ok(())` after graceful HTTP shutdown, or a contextual startup/
/// serving error. Starts discovery, worker management, stale-load cleanup and
/// configured scheduler tasks. After serving returns, aborts discovery/manager
/// and joins the cleanup task. Dependency constructors may panic on invalid
/// configuration; argument parsing can exit before initialization completes.
async fn main() -> Result<()> {
    let cli = Cli::parse();
    // Bootstrap subscriber so a config-resolution error has structured
    // output. The configured-format subscriber installs after this and
    // becomes a no-op via try_init's idempotency.
    install_bootstrap_subscriber();
    let cfg = cli
        .into_config()
        .context("resolve configuration from CLI flags")?;

    init_tracing(&cfg.observability.log_level, cfg.observability.log_format)?;

    tracing::info!(
        "sgl-router {} starting on {}:{}",
        env!("CARGO_PKG_VERSION"),
        cfg.server.host,
        cfg.server.port
    );

    let tokenizers = Arc::new(
        sgl_router::tokenizer::TokenizerRegistry::load_from_config(&cfg)
            .context("load tokenizers")?,
    );

    // `--disable-circuit-breaker` is applied here, before discovery adds any
    // worker: every worker then gets a breaker that always admits.
    let registry = Arc::new(sgl_router::workers::WorkerRegistry::from_config(&cfg));
    if registry.circuit_breakers_disabled() {
        tracing::info!("circuit breaker disabled: every worker always admits");
    }
    let prefix_index = cfg
        .model
        .cache_aware
        .as_ref()
        .and_then(|cache| cache.kv_indexer_endpoint.as_ref())
        .map(|indexer| {
            let config = sgl_kv_indexer::PrefixIndexConfig {
                endpoint: indexer.url.clone(),
                query_deadline: std::time::Duration::from_millis(indexer.query_timeout_ms),
                max_inflight: indexer.query_max_inflight,
            };
            sgl_kv_indexer::GrpcPrefixIndex::new(config)
                .map(Arc::new)
                .context("configure KV Indexer client")
        })
        .transpose()?;

    // Build the KV-event index up front so the cache-aware-zmq policy can
    // share its `HashTree` handle + `BlockSizeOracle`. An external Indexer makes
    // the local tree irrelevant to routing, so only discover hash metadata rather
    // than duplicating every KV event.
    let block_size_oracle = sgl_router::policies::kv_events::BlockSizeOracle::new();
    let kv_event_http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .expect("default http client builds");
    let kv_index = if prefix_index.is_some() || cfg.model.cache_aware.as_ref().is_some_and(|c| c.dynamo.is_some()) {
        sgl_router::policies::kv_events::KvEventIndex::new_metadata_only_with_http_and_oracle(
            kv_event_http,
            Arc::clone(&block_size_oracle),
        )
    } else {
        sgl_router::policies::kv_events::KvEventIndex::new_with_http_and_oracle(
            kv_event_http,
            Arc::clone(&block_size_oracle),
        )
    };
    let policies = Arc::new(
        sgl_router::policies::factory::build_registry(
            &cfg,
            kv_index.tree(),
            Arc::clone(&tokenizers),
            Arc::clone(&block_size_oracle),
        )
        .context("build policy registry")?,
    );

    // Shared ActiveLoadRegistry + janitor task. The janitor reaps
    // request entries whose lifetime exceeded `stale_request_timeout`,
    // so a leaked guard (proxy task panic, etc.) does not inflate a
    // worker's load forever. The registry is built BEFORE the manager
    // is spawned so the manager can call `forget_worker` on
    // `DiscoveryEvent::Removed`.
    let stale_timeout = std::time::Duration::from_secs(cfg.active_load.stale_request_timeout_secs);
    let active_load = sgl_router::policies::active_load::ActiveLoadRegistry::new(
        Arc::new(sgl_router::policies::active_load::SystemTimeClock),
        stale_timeout,
    );
    // Sweep cadence is 1/10 of the configured timeout, clamped to
    // [1 s, 60 s]. A short timeout (test setting) needs frequent
    // sweeps to fire within the test's window; a long timeout
    // (production) doesn't need sub-minute checks.
    let sweep_interval = std::time::Duration::from_secs(
        (cfg.active_load.stale_request_timeout_secs / 10).clamp(1, 60),
    );
    let janitor_handle =
        sgl_router::policies::active_load::spawn_janitor(Arc::clone(&active_load), sweep_interval);

    // Spawn discovery + manager tasks.
    let (event_rx, discovery_handle) = sgl_router::discovery::spawn_discovery(&cfg)
        .await
        .context("spawn discovery")?;
    let kv_index_opt: Option<Arc<sgl_router::policies::kv_events::KvEventIndex>> =
        Some(Arc::clone(&kv_index));
    let manager_handle = tokio::spawn(sgl_router::workers::manager::run_with_config(
        event_rx,
        registry.clone(),
        Some(Arc::new(cfg.clone())),
        kv_index_opt,
        Some(Arc::clone(&active_load)),
    ));

    let proxy_tcp_keepalive = cfg
        .proxy
        .tcp_keepalive_secs
        .map(std::time::Duration::from_secs);
    if let Some(idle) = proxy_tcp_keepalive {
        tracing::info!(
            idle_secs = idle.as_secs(),
            "proxy TCP keepalive idle time set; probe interval 15 s, 3 probes and \
             TCP_USER_TIMEOUT 30 s stay at the reqwest defaults"
        );
    }
    if cfg.proxy.abort_decode_on_prefill_failure {
        tracing::info!(
            "PD: a failed prefill leg drops its decode leg \
             (--pd-abort-decode-on-prefill-failure)"
        );
    }
    let late_decode = cfg.proxy.late_decode;
    if late_decode.choose() {
        tracing::info!(
            target: "late_d",
            rule = late_decode.rule.map_or("-", |r| r.as_str()),
            candidates = %late_decode
                .candidates
                .map_or_else(|| "all".to_string(), |k| k.to_string()),
            packed_prompt = late_decode.packed_prompt,
            fit_capacity_tokens = late_decode.fit_capacity_tokens.unwrap_or(0),
            native_capacity_fit = late_decode.fit_native,
            native_state_poll = late_decode.native_state_poll,
            native_capacity_max_age_ms = late_decode.fit_state_max_age_ms,
            request_timeout_secs = cfg.proxy.request_timeout_secs,
            stale_request_timeout_secs = cfg.active_load.stale_request_timeout_secs,
            "late D choose: each PD request goes to its prefill and to every decode \
             candidate; the prefill binds one over its router link (--pd-late-decode choose; \
             /v1/completions only)"
        );
    } else if late_decode.packed_prompt {
        tracing::info!(
            target: "late_d",
            "late D off; PD decode legs carry packed token prompts \
             (--pd-late-decode-packed-prompt)"
        );
    }
    if late_decode.native_state_poll && !late_decode.choose() {
        tracing::info!(
            target: "late_d",
            "late D off; the decode capacity poll runs for observation only \
             (--pd-native-decode-state)"
        );
    }
    if let Some(g) = cfg.model.gateway_cache_aware.as_ref() {
        tracing::info!(
            target: sgl_router::policies::gateway_cache_aware::LOG_TARGET,
            cache_threshold = g.cache_threshold,
            balance_abs_threshold = g.balance_abs_threshold,
            balance_rel_threshold = g.balance_rel_threshold,
            eviction_interval_secs = g.eviction_interval_secs,
            max_tree_size = g.max_tree_size,
            "gateway cache-aware prefill policy on: sgl-model-gateway --prefill-policy \
             cache_aware (sgl-model-gateway cache-aware policy), load = in-flight \
             prefill requests of this router"
        );
    }
    if let Some(key_file) = cfg.model.pd_kv_session_retention() {
        tracing::info!(
            target: "kv_session",
            identity_header = sgl_router::agent_identity::KV_SESSION_HEADER,
            key_file = key_file.is_some(),
            stats_period_secs = sgl_router::server::routes::kv_session::STATS_PERIOD.as_secs(),
            "session KV retention on: a PD request with the identity header registers its \
             prefill worker and only the prefill leg carries kv_session (--kv-session-retention)"
        );
    }
    let proxy = Arc::new(
        sgl_router::proxy::Proxy::with_tcp_keepalive(
            std::time::Duration::from_secs(cfg.proxy.request_timeout_secs),
            proxy_tcp_keepalive,
        )
        .context("build proxy client")?,
    );

    let mut app_ctx = sgl_router::server::app_context::AppContext::with_active_load(
        cfg.clone(),
        tokenizers,
        proxy,
        registry,
        policies,
        active_load,
    );
    app_ctx.prefix_index = prefix_index;
    app_ctx.block_size_oracle = block_size_oracle;
    let ctx = Arc::new(app_ctx);
    ctx.mark_ready();

    let app = sgl_router::server::app::build_router(ctx.clone());

    let bind = format!("{}:{}", cfg.server.host, cfg.server.port);
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .with_context(|| format!("bind {bind}"))?;
    tracing::info!("listening on {bind}");

    let (sigterm, sigint) = install_signal_handlers()?;

    let serve = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal(sigterm, sigint));
    let server_result = serve.await.context("axum serve");

    // Best-effort: cancel discovery + manager + janitor on shutdown.
    // The janitor handle's drop signals cancellation; we additionally
    // await `shutdown` so the task joins cleanly before the process
    // exits — useful for tracing tail logs.
    discovery_handle.abort();
    manager_handle.abort();
    janitor_handle.shutdown().await;
    if let Some(sessions) = &ctx.kv_sessions {
        sgl_router::server::routes::kv_session::log_stats(sessions, "shutdown");
    }
    server_result
}

/// Wait for either supported termination signal to release HTTP shutdown.
///
/// # Inputs
/// `sigterm` and `sigint` are registered Unix signal streams.
///
/// # Outputs
/// Returns `()` when either stream yields (or closes), logging the
/// selected signal branch. Axum uses this completion to begin graceful
/// shutdown; this function does not terminate worker engines.
async fn shutdown_signal(mut sigterm: Signal, mut sigint: Signal) {
    tokio::select! {
        _ = sigterm.recv() => tracing::info!("got SIGTERM, shutting down"),
        _ = sigint.recv()  => tracing::info!("got SIGINT, shutting down"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    /// Verify signal setup succeeds in a signal-capable runtime.
    ///
    /// # Inputs
    /// No arguments. Run under Tokio and register both Unix termination signal handlers.
    ///
    /// # Outputs
    /// Returns `()` when the stated invariant holds; failed assertions panic.
    async fn install_signal_handlers_returns_both() {
        // Pins the contract that handler installation works on a standard
        // tokio runtime. If this fails on a sandboxed runner, the real
        // service would also fail to install — which is the point.
        assert!(install_signal_handlers().is_ok());
    }

    #[test]
    /// Verify repeated initialization does not panic.
    ///
    /// # Inputs
    /// No arguments. Attempt global text-subscriber initialization twice.
    ///
    /// # Outputs
    /// Returns `()` when the stated invariant holds; failed assertions panic.
    fn init_tracing_is_idempotent() {
        let _ = init_tracing("info", LogFormat::Text);
        let _ = init_tracing("info", LogFormat::Text);
    }

    #[test]
    /// Verify JSON initialization returns successfully with or without an existing subscriber.
    ///
    /// # Inputs
    /// No arguments. Request JSON logging, allowing another test to have installed a subscriber.
    ///
    /// # Outputs
    /// Returns `()` when the stated invariant holds; failed assertions panic.
    fn init_tracing_accepts_json_format() {
        // Doesn't matter whether we win or lose the race against another
        // subscriber install — the function must return Ok either way.
        assert!(init_tracing("info", LogFormat::Json).is_ok());
    }
}
