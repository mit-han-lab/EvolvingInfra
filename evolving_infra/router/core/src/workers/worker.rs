// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::{ModelId, WorkerId, WorkerMode};
use crate::scheduler::placement::{Capabilities, RoleWeight};
use crate::health::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

/// symmetric P/D view of one worker (capability set +
/// primary-role weights + the rank-0 KV side-channel endpoints). Set by the
/// worker manager from `--symmetric-pd-worker` specs and the engine's
/// `/server_info` `symmetric_pd` section; read by `crate::symmetric_pd`.
#[derive(Debug, Clone, PartialEq)]
pub struct SymmetricPdWorkerInfo {
    pub capabilities: Capabilities,
    pub weight: RoleWeight,
    /// Rank-0 owner (export) ZMQ endpoint, `tcp://host:port`; `None` =
    /// derive from `bootstrap_port`.
    pub owner_addr: Option<String>,
    /// Rank-0 reader (import) ZMQ endpoint; `None` = `owner + 1`.
    pub reader_addr: Option<String>,
    /// `/server_info` `max_total_num_tokens` (KV pool tokens); used by the P/D ratio controller.
    pub kv_capacity_tokens: Option<u64>,
}

impl Default for SymmetricPdWorkerInfo {
    fn default() -> Self {
        Self {
            capabilities: Capabilities::BOTH,
            weight: RoleWeight::default(),
            owner_addr: None,
            reader_addr: None,
            kv_capacity_tokens: None,
        }
    }
}

/// Late-D view of one worker: the `/server_info` `late_d` block of an
/// engine that runs `--disaggregation-late-d`. `None` on the worker when the
/// block is absent (late D off, or an
/// older engine).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LateDWorkerInfo {
    /// The engine's `--disaggregation-late-d` mode (`alloc`, `choose`).
    pub mode: String,
    /// A prefill's router-link port (`--disaggregation-late-d-control-port`);
    /// the router connects a ZMQ DEALER to `tcp://<bootstrap_host>:<port>`.
    pub control_port: Option<u16>,
    /// The address the engine's KV connector puts in its bootstrap messages.
    pub kv_ip: Option<String>,
}

/// Parse a host from a worker URL. Matches SMG's `worker_builder.rs`
/// fallback chain: parse as-is, retry with `http://` prefix if missing,
/// fall back to `"localhost"` if both fail. The fallback is defensive —
/// discovery code should never emit an unparsable URL — but a panic
/// here would crash the whole router on a single bad config entry.
fn parse_bootstrap_host(url: &str) -> String {
    if let Ok(parsed) = url::Url::parse(url) {
        if let Some(h) = parsed.host_str() {
            return h.to_string();
        }
    }
    if !url.contains("://") {
        if let Ok(parsed) = url::Url::parse(&format!("http://{url}")) {
            if let Some(h) = parsed.host_str() {
                return h.to_string();
            }
        }
    }
    tracing::warn!(
        worker_url = %url,
        "Failed to parse worker URL for bootstrap_host; defaulting to 'localhost'"
    );
    "localhost".to_string()
}

/// RAII guard that increments `active_requests` on construction and
/// decrements on drop.  Obtain via [`Worker::load_guard`].
///
/// `#[must_use]`: a statement-form call like `worker.load_guard();` would
/// drop the guard on the same line, so the counter would never see the
/// in-flight request.  The compile-time warning catches that misuse.
#[must_use = "LoadGuard must be held for the request's lifetime; dropping it immediately decrements active_requests"]
pub struct LoadGuard {
    counter: Arc<AtomicUsize>,
}

impl LoadGuard {
    pub(crate) fn new(counter: Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::Relaxed);
        Self { counter }
    }
}

impl Drop for LoadGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Relaxed);
    }
}

/// RAII token charge on a decode worker, minted by the token-charged
/// `--pd-decode-policy` variants ([`crate::policies::decode_tokens`]) at
/// selection time and released on drop. The request routes hold it beside
/// the decode leg's [`LoadGuard`] (streaming: inside the SSE pump's
/// `stream_guards`; non-streaming: in the handler scope), so every path
/// that ends the decode leg — stream end, upstream error, transport error,
/// stale-request cancel, client disconnect, or the handler future being
/// dropped before dispatch — releases the charge exactly once.
#[must_use = "DecodeTokenCharge must be held for the decode leg's lifetime; dropping it releases the charge"]
pub struct DecodeTokenCharge {
    tokens: u64,
    charged_tokens: Arc<AtomicU64>,
    charged_reqs: Arc<AtomicUsize>,
}

impl DecodeTokenCharge {
    fn new(tokens: u64, charged_tokens: Arc<AtomicU64>, charged_reqs: Arc<AtomicUsize>) -> Self {
        charged_tokens.fetch_add(tokens, Ordering::Relaxed);
        charged_reqs.fetch_add(1, Ordering::Relaxed);
        Self {
            tokens,
            charged_tokens,
            charged_reqs,
        }
    }

    /// Tokens this charge holds on its worker.
    pub fn tokens(&self) -> u64 {
        self.tokens
    }
}

impl Drop for DecodeTokenCharge {
    fn drop(&mut self) {
        self.charged_tokens
            .fetch_sub(self.tokens, Ordering::Relaxed);
        self.charged_reqs.fetch_sub(1, Ordering::Relaxed);
    }
}

impl std::fmt::Debug for DecodeTokenCharge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodeTokenCharge")
            .field("tokens", &self.tokens)
            .finish()
    }
}

impl WorkerMode {
    fn as_u8(self) -> u8 {
        match self {
            WorkerMode::Plain => 0,
            WorkerMode::Prefill => 1,
            WorkerMode::Decode => 2,
            WorkerMode::Symmetric => 3,
        }
    }

    /// Inverse of [`Self::as_u8`].  The only writers of the underlying
    /// `AtomicU8` are `as_u8`-derived values, so any out-of-range byte
    /// indicates memory corruption or a stale store from an
    /// incompatible build — fail loudly rather than silently mislabel
    /// the worker as `Decode`.
    fn from_u8(v: u8) -> Self {
        match v {
            0 => WorkerMode::Plain,
            1 => WorkerMode::Prefill,
            2 => WorkerMode::Decode,
            3 => WorkerMode::Symmetric,
            other => unreachable!("invalid WorkerMode discriminant {other}"),
        }
    }
}

pub struct Worker {
    pub id: WorkerId,
    pub url: String,
    /// Interior-mutable mode so `ModeChanged` can update in place without
    /// dropping the Worker (which would reset `active_requests` + breaker).
    mode: AtomicU8,
    pub model_ids: Vec<ModelId>,
    pub breaker: Arc<CircuitBreaker>,
    pub active_requests: Arc<AtomicUsize>,
    /// Hostname parsed from `url` at construction time and cached.
    /// Used as the `bootstrap_host` field on PD-disagg requests so the
    /// prefill engine can match incoming KV-transfer requests from
    /// decode peers. Falls back to `"localhost"` if the URL fails to
    /// parse — a misconfigured worker will fail the prefill request
    /// downstream rather than panic here.
    bootstrap_host: String,
    /// SGLang bootstrap server port for prefill workers (`None` for
    /// decode and plain). Set via `--disaggregation-bootstrap-port` at
    /// worker startup; carried from `WorkerSpec`. For `Symmetric` workers
    /// it is the base port of the per-rank ZMQ side channels.
    bootstrap_port: Option<u16>,
    /// symmetric P/D capabilities / weights / endpoints (see [`SymmetricPdWorkerInfo`]).
    /// Defaults to `{prefill: 1, decode: 1}`, both capabilities, no
    /// advertised endpoints; the manager overwrites it after registration.
    symmetric_pd: RwLock<SymmetricPdWorkerInfo>,
    /// Sum of [`DecodeTokenCharge`]s currently held on this worker (token-
    /// charged decode policies only; 0 otherwise).
    decode_charged_tokens: Arc<AtomicU64>,
    /// Number of [`DecodeTokenCharge`]s currently held on this worker.
    decode_charged_reqs: Arc<AtomicUsize>,
    /// KV capacity in tokens (`/server_info` `max_total_num_tokens`); 0 =
    /// not reported.
    kv_capacity_tokens: AtomicU64,
    /// Engine chunk in tokens (`/server_info` `chunked_prefill_size`); 0 =
    /// not reported.
    chunk_tokens: AtomicU64,
    /// `/server_info` `late_d` block (see [`LateDWorkerInfo`]).
    late_d: RwLock<Option<LateDWorkerInfo>>,
}

impl Worker {
    pub fn new(spec: crate::discovery::WorkerSpec) -> Self {
        Self::with_cb_config(spec, None)
    }

    /// Construct a worker with an explicit circuit-breaker configuration.
    /// Pass `None` to use the default config (threshold = 3, cool_down = 30 s).
    pub fn with_cb_config(
        spec: crate::discovery::WorkerSpec,
        cb: Option<CircuitBreakerConfig>,
    ) -> Self {
        let breaker = match cb {
            Some(cfg) => Arc::new(CircuitBreaker::with_config(cfg)),
            None => Arc::new(CircuitBreaker::new()),
        };
        Self::with_breaker(spec, breaker)
    }

    /// Construct a worker around an existing breaker (e.g.
    /// [`CircuitBreaker::disabled`] under `--disable-circuit-breaker`).
    pub fn with_breaker(spec: crate::discovery::WorkerSpec, breaker: Arc<CircuitBreaker>) -> Self {
        let bootstrap_host = parse_bootstrap_host(&spec.url);
        Self {
            id: spec.id,
            url: spec.url,
            mode: AtomicU8::new(spec.mode.as_u8()),
            model_ids: spec.model_ids,
            breaker,
            active_requests: Arc::new(AtomicUsize::new(0)),
            bootstrap_host,
            bootstrap_port: spec.bootstrap_port,
            symmetric_pd: RwLock::new(SymmetricPdWorkerInfo::default()),
            decode_charged_tokens: Arc::new(AtomicU64::new(0)),
            decode_charged_reqs: Arc::new(AtomicUsize::new(0)),
            kv_capacity_tokens: AtomicU64::new(0),
            chunk_tokens: AtomicU64::new(0),
            late_d: RwLock::new(None),
        }
    }

    /// Snapshot of the symmetric P/D capability / weight / endpoint view.
    pub fn symmetric_pd(&self) -> SymmetricPdWorkerInfo {
        self.symmetric_pd.read().clone()
    }

    /// Replace the symmetric P/D view (manager, after `/server_info` + config).
    pub fn set_symmetric_pd(&self, info: SymmetricPdWorkerInfo) {
        *self.symmetric_pd.write() = info;
    }

    /// Hostname carried on PD-disagg request bodies as `bootstrap_host`.
    pub fn bootstrap_host(&self) -> &str {
        &self.bootstrap_host
    }

    /// SGLang bootstrap server port. `None` for decode / plain workers.
    pub fn bootstrap_port(&self) -> Option<u16> {
        self.bootstrap_port
    }

    /// Returns the current [`WorkerMode`] of this worker.
    ///
    /// Uses `Relaxed` ordering: mode changes are rare discovery events and do
    /// not need to synchronise with any other memory access.
    pub fn mode(&self) -> WorkerMode {
        WorkerMode::from_u8(self.mode.load(Ordering::Relaxed))
    }

    /// Update the worker's mode in place.
    ///
    /// Preserves `active_requests` and `breaker` state — the same `Arc<Worker>`
    /// identity survives the mode transition.
    pub fn set_mode(&self, m: WorkerMode) {
        self.mode.store(m.as_u8(), Ordering::Relaxed);
    }

    pub fn active_load(&self) -> usize {
        self.active_requests.load(Ordering::Relaxed)
    }

    /// Returns a RAII guard that increments `active_requests` now and
    /// decrements when the guard is dropped.
    pub fn load_guard(&self) -> LoadGuard {
        LoadGuard::new(self.active_requests.clone())
    }

    /// Charge `tokens` to this worker until the returned guard drops.
    pub fn charge_decode_tokens(&self, tokens: u64) -> DecodeTokenCharge {
        DecodeTokenCharge::new(
            tokens,
            Arc::clone(&self.decode_charged_tokens),
            Arc::clone(&self.decode_charged_reqs),
        )
    }

    /// Tokens currently charged by live [`DecodeTokenCharge`]s.
    pub fn decode_charged_tokens(&self) -> u64 {
        self.decode_charged_tokens.load(Ordering::Relaxed)
    }

    /// Live [`DecodeTokenCharge`] count.
    pub fn decode_charged_reqs(&self) -> usize {
        self.decode_charged_reqs.load(Ordering::Relaxed)
    }

    /// KV capacity in tokens as reported by `/server_info`
    /// (`max_total_num_tokens`), if any.
    pub fn kv_capacity_tokens(&self) -> Option<u64> {
        Some(self.kv_capacity_tokens.load(Ordering::Relaxed)).filter(|&c| c > 0)
    }

    /// Record the `/server_info` KV capacity (0 clears it).
    pub fn set_kv_capacity_tokens(&self, tokens: u64) {
        self.kv_capacity_tokens.store(tokens, Ordering::Relaxed);
    }

    /// The engine chunk in tokens as reported by `/server_info`
    /// (`chunked_prefill_size`), if any.
    pub fn chunk_tokens(&self) -> Option<u64> {
        Some(self.chunk_tokens.load(Ordering::Relaxed)).filter(|&c| c > 0)
    }

    /// Record the `/server_info` engine chunk (0 clears it).
    pub fn set_chunk_tokens(&self, tokens: u64) {
        self.chunk_tokens.store(tokens, Ordering::Relaxed);
    }

    /// The `/server_info` `late_d` block, if the engine runs late D.
    pub fn late_d(&self) -> Option<LateDWorkerInfo> {
        self.late_d.read().clone()
    }

    /// Record the `/server_info` `late_d` block (`None` clears it).
    pub fn set_late_d(&self, info: Option<LateDWorkerInfo>) {
        *self.late_d.write() = info;
    }
}

impl std::fmt::Debug for Worker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Worker")
            .field("id", &self.id)
            .field("url", &self.url)
            .field("mode", &self.mode())
            .field("active_load", &self.active_load())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};

    #[test]
    fn load_guard_increments_and_decrements() {
        let w = Worker::new(WorkerSpec {
            id: WorkerId("w".into()),
            url: "http://x".into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
        });
        assert_eq!(w.active_load(), 0);
        let g = w.load_guard();
        assert_eq!(w.active_load(), 1);
        let g2 = w.load_guard();
        assert_eq!(w.active_load(), 2);
        drop(g);
        assert_eq!(w.active_load(), 1);
        drop(g2);
        assert_eq!(w.active_load(), 0);
    }

    #[test]
    fn decode_token_charge_releases_on_drop_including_unwind() {
        let w = Arc::new(Worker::new(WorkerSpec {
            id: WorkerId("d".into()),
            url: "http://d".into(),
            mode: WorkerMode::Decode,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
        }));
        assert_eq!((w.decode_charged_tokens(), w.decode_charged_reqs()), (0, 0));
        let a = w.charge_decode_tokens(1000);
        let b = w.charge_decode_tokens(24);
        assert_eq!(a.tokens(), 1000);
        assert_eq!(
            (w.decode_charged_tokens(), w.decode_charged_reqs()),
            (1024, 2)
        );
        drop(a);
        assert_eq!(
            (w.decode_charged_tokens(), w.decode_charged_reqs()),
            (24, 1)
        );
        // Boxed as an opaque `stream_guards` payload (the SSE pump path).
        let boxed: Box<dyn Send + 'static> = Box::new((w.load_guard(), b));
        assert_eq!(w.active_load(), 1);
        drop(boxed);
        assert_eq!((w.decode_charged_tokens(), w.decode_charged_reqs()), (0, 0));
        assert_eq!(w.active_load(), 0);
        // A panic while the charge is held still releases it.
        let w2 = Arc::clone(&w);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _c = w2.charge_decode_tokens(7);
            if w2.decode_charged_reqs() == 1 {
                panic!("boom");
            }
        }));
        assert!(r.is_err());
        assert_eq!((w.decode_charged_tokens(), w.decode_charged_reqs()), (0, 0));
        // Capacity: 0 means "not reported".
        assert_eq!(w.kv_capacity_tokens(), None);
        w.set_kv_capacity_tokens(500_000);
        assert_eq!(w.kv_capacity_tokens(), Some(500_000));
    }

    #[test]
    fn mode_accessor_round_trips_all_variants() {
        for m in [
            WorkerMode::Plain,
            WorkerMode::Prefill,
            WorkerMode::Decode,
            WorkerMode::Symmetric,
        ] {
            let w = Worker::new(WorkerSpec {
                id: WorkerId("w".into()),
                url: "http://x".into(),
                mode: m,
                model_ids: vec![],
                bootstrap_port: None,
            });
            assert_eq!(w.mode(), m);
        }
    }

    #[test]
    fn set_mode_updates_in_place() {
        let w = Worker::new(WorkerSpec {
            id: WorkerId("w".into()),
            url: "http://x".into(),
            mode: WorkerMode::Prefill,
            model_ids: vec![],
            bootstrap_port: None,
        });
        assert_eq!(w.mode(), WorkerMode::Prefill);
        w.set_mode(WorkerMode::Decode);
        assert_eq!(w.mode(), WorkerMode::Decode);
        w.set_mode(WorkerMode::Plain);
        assert_eq!(w.mode(), WorkerMode::Plain);
    }

    #[test]
    fn symmetric_pd_info_defaults_and_updates_in_place() {
        let w = Worker::new(WorkerSpec {
            id: WorkerId("s".into()),
            url: "http://10.0.0.1:30000".into(),
            mode: WorkerMode::Symmetric,
            model_ids: vec![],
            bootstrap_port: Some(8998),
        });
        assert_eq!(w.symmetric_pd(), SymmetricPdWorkerInfo::default());
        w.set_symmetric_pd(SymmetricPdWorkerInfo {
            weight: RoleWeight::PREFILL_LEANING,
            owner_addr: Some("tcp://10.0.0.1:7000".into()),
            ..Default::default()
        });
        let info = w.symmetric_pd();
        assert_eq!(info.weight, RoleWeight::PREFILL_LEANING);
        assert_eq!(info.owner_addr.as_deref(), Some("tcp://10.0.0.1:7000"));
        assert_eq!(w.mode(), WorkerMode::Symmetric);
    }

    #[test]
    fn bootstrap_port_returns_spec_value_for_prefill() {
        let w = Worker::new(WorkerSpec {
            id: WorkerId("p1".into()),
            url: "http://10.0.0.1:30000".into(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: Some(8997),
        });
        assert_eq!(w.bootstrap_port(), Some(8997));
    }

    #[test]
    fn bootstrap_port_defaults_to_none() {
        let w = Worker::new(WorkerSpec {
            id: WorkerId("w".into()),
            url: "http://10.0.0.1:30000".into(),
            mode: WorkerMode::Plain,
            model_ids: vec![],
            bootstrap_port: None,
        });
        assert_eq!(w.bootstrap_port(), None);
    }

    #[test]
    fn bootstrap_host_parses_ipv4_from_url() {
        let w = Worker::new(WorkerSpec {
            id: WorkerId("p1".into()),
            url: "http://10.0.0.1:30000".into(),
            mode: WorkerMode::Prefill,
            model_ids: vec![],
            bootstrap_port: Some(8997),
        });
        assert_eq!(w.bootstrap_host(), "10.0.0.1");
    }

    #[test]
    fn bootstrap_host_parses_dns_name_from_url() {
        let w = Worker::new(WorkerSpec {
            id: WorkerId("p1".into()),
            url: "http://prefill-0.svc.cluster.local:30000".into(),
            mode: WorkerMode::Prefill,
            model_ids: vec![],
            bootstrap_port: Some(8997),
        });
        assert_eq!(w.bootstrap_host(), "prefill-0.svc.cluster.local");
    }

    #[test]
    fn bootstrap_host_falls_back_to_localhost_for_unparsable_url() {
        // An empty / invalid URL is not expected from discovery, but the
        // accessor must return a usable string rather than panic — the
        // prefill worker will reject the request body-side if the host
        // really is unreachable.
        let w = Worker::new(WorkerSpec {
            id: WorkerId("p1".into()),
            url: "not a url".into(),
            mode: WorkerMode::Prefill,
            model_ids: vec![],
            bootstrap_port: Some(8997),
        });
        assert_eq!(w.bootstrap_host(), "localhost");
    }
}
