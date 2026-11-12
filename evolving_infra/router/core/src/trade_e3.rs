// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! TRADE `--co-schedule-trade-wait e3`: the E3 P completion forecast
//! ([`crate::e3`], a port of the Python reference `router_e3.py` r5) in the
//! TRADE P selection, with the prediction gate.
//!
//! # What the router does per decision
//!
//! For each P candidate `w` the router builds the E3 input
//! from `w`'s latest polled `prefill_routing` snapshot ([`E3View`]) and from
//! its own cost book, and forecasts `pred_s` (decision to `w`'s
//! `prefill_finished_time` of the request) and `wait_s` (decision to the GPU
//! start of the request's first pass). The deployed mode is
//! `snap+prod` (poll, router calibration `rcal`, the production rules):
//!
//! - `t0` = poll receipt - the snapshot's `age_s`. The replay's
//!   `snap` mode used `d - effective_age_upper_s`, which counts from the poll
//!   SEND time; the two differ by the poll round trip (7.6 ms in the cut
//!   fixture).
//! - Engine policy: the pass budget and the page of the simulated
//!   passes are the snapshot's `policy.chunk_tokens` and `policy.page_size`.
//!   The shipped bound tables were fitted at [`BOUND_FIT_POLICY`]; a P with
//!   another policy has no forecast (`policy_not_bound_fit`).
//! - In-flight passes: the snapshot's pending `inflight_batches` (launch =
//!   `t0 - launch_age_s`). The engine publishes the gated
//!   `host_restore_tokens`; the forecast wants the ungated provenance, which
//!   no engine publishes yet.
//! - The partial request: `remaining_compute_tokens` (floor 1) and
//!   `committed_prefix_tokens + inflight_new_tokens`.
//! - Queue rows (rule `rq`): the waiting rows in snapshot order;
//!   then the bootstrap rows (ready = receipt + `boot_s`, rule `bootest`) and
//!   the router's own rows that the snapshot does not show, sorted by ready
//!   time (ties by decision time). A row is kept only when its ready time is
//!   at most the target's ready time minus the margin
//!   (`--co-schedule-trade-e3-rq-margin-ms`, default 0).
//! - A snapshot row with a pending pull (`plan.pull_tokens`): the pulled
//!   tokens count as cached and restored from host (the engine's
//!   `remaining_compute_tokens` still counts them as compute until the pull
//!   lands), as for the target and the router rows.
//! - Null-prefix rule: a snapshot row without a published prefix
//!   takes the router's own dispatch-time estimate of that RID, else it is
//!   cold (prefix 0).
//! - The router rows: active cost-book entries of `w` whose RID the snapshot
//!   does not show (rows, the partial request, the transfer queue, the
//!   pending passes), except an entry that a progress poll received before
//!   `t0` already listed (it launched before the snapshot); plus finished
//!   entries decided after `t0`. Each row carries the router's estimate at
//!   its decision (prefix and host, as for the target).
//! - The target: prefix = page-floored own hit plus the pulled tokens when
//!   `w` pulls; host = own hit minus the indexer's observed GPU hit when the
//!   indexer reports one (an approximate split), else the session host rule:
//!   the learned share of host-restored cached
//!   tokens in the bin of (own prefix, seconds since this session last went
//!   to `w`) decides between the whole prefix and none.
//!
//! # Online fits (rcal)
//!
//! [`E3Online`] keeps the engine timeline samples (`timeline.samples` of
//! each snapshot, deduplicated by engine epoch and sequence) per P, as
//! router-clock time-keyed windows. It drops samples older than
//! `CAL_S + WINDOW_S` (1,500 s), except the calibration-window copies and
//! the service samples, which move to a per-class history
//! ([`SERVICE_HISTORY_PER_CLASS`]). It refits once per (P, 60-s step),
//! lazily on the first decision of a step, from copies of the windows taken
//! under its own lock and fitted outside it (never under the cost-book
//! lock). The fits:
//!
//! - GPU (service) model per size class: samples of single-pass, batch-size-1
//!   requests on a pass launched after an idle gap and not followed by a
//!   busy launch (`last_launch_to_done`, `y_ms = (done - launch) x 1000`,
//!   the result lag included, `lag_none_s = 0`); chain own P window, all P
//!   window, own P calibration window, every earlier sample of the own P,
//!   of all P.
//! - Launch model (G1) on that GPU model from `launch_interval` samples
//!   (window, else calibration rows, else the median launch interval).
//! - Calibration constants from the first 900 s of samples of each P (the
//!   router does not know the warmup boundary): the window opens at the
//!   first sample that completed after the router's first snapshot of the
//!   P; the samples of that first snapshot (the engine's startup warmup,
//!   traffic before a router restart) feed the windowed fits only. The
//!   constants: `boot_s` (waiting rows: `recv_age_s - queue_age_s`),
//!   `phi_s` (receipt to the next busy launch), `idle_delay_s` (receipt to
//!   the first launch of requests whose first pass followed an idle gap,
//!   minus `boot_s`), the GPU floor and the next-aware result lag.
//! - Receipt line `rx2`: requests whose chosen forecast was idle; the sample
//!   (input tokens, receipt - decision) is added when the router receives
//!   the request's `request_complete` sample.
//! - Arrival rates from the router's dispatch log of the P.
//!
//! Without a GPU model (no service sample yet) a P has no forecast
//! (`no_fit`): the selection keeps the cache-aware owner.
//!
//! With `--co-schedule-trade-e3-frozen-fit FILE` ([`FrozenFit`]) every fit
//! takes the GPU model, the launch model and the calibration constants of
//! FILE (offline warmup constants, source `frozen` in the `e3_fit` event);
//! the online samples still give the arrival rates. A P that is never idle
//! (Kimi-K2.6 under CUA load) has almost no service samples, so without
//! FILE it keeps `no_fit`.
//!
//! # Shadow log and live MAPE
//!
//! Every routed request gets one `e3_shadow` line (rid, chosen P, its
//! `pred_s`, decision time) in the cost trace, or in
//! `--co-schedule-trade-e3-shadow PATH`. When the router later receives the
//! request's `request_complete` timeline sample from the chosen P, it writes
//! an `e3_label` line: `done` on the router clock is within
//! `[poll sent, poll received] - age_s - completed_age_s`, so
//! `label_lo_s <= label <= label_hi_s` and `label_s` is the midpoint (the
//! evaluator's convention). Live MAPE = mean over `e3_label` lines of
//! `|pred_s - label_s| / label_s`; coverage = labels / shadows. The same
//! rows join the engine's `prefill_finished_time` (req_trace) by `rid` with
//! the evaluator's clock-offset interval (`decision_unix_s` is the router
//! wall clock at the decision).
//!
//! # State push (`--co-schedule-trade-e3-push-port`)
//!
//! With the push on, [`crate::trade_e3_push`] subscribes to each P's state
//! push (engine wire contract v1, `prefill_state_push.py`). A pushed
//! message carries the same `prefill_routing` snapshot, so it parses into the
//! same [`E3View`]; its state time is `sampled_monotonic_s + offset` (the
//! offset per connection). A candidate's forecast starts from the
//! newest pushed state when that state is fresh, has the epoch of the P's
//! polled snapshot and is usable; otherwise from the polled snapshot as
//! above. Its error bound is the table of its state source
//! ([`TradeE3Config::bound_for`]). The pushed timeline samples feed the same
//! online fits (deduplicated by sequence with the polled ones); a label from
//! a pushed sample has a zero-width interval (`label_source` `push`).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::e3::numeric::{py_cmp, py_max, quantile, sort_py};
use crate::e3::{
    arrival_rates, error_bound, features, fit_gpu_classes, fit_launch_intervals, make_params,
    receipt_delays, resolve_gpu, router_floor_ms, router_lag_next, row_ready, step_of, ArrivalRow,
    Arrivals, BoundParams, BoundTable, BucketView, Constants, Features, GpuClasses, GpuModel,
    GpuSample, GpuSources, InflightPass, Input, LagSample, LaunchAnchor, LaunchFit, LaunchSample,
    Output, Params, Partial, QueueRow, RowSource, RxLine, RxParams, Target,
};
use crate::eft_cheap::{is_true, nonneg_int};

/// The deploy mode of the router's poll path: snapshot poll,
/// router calibration without GPU times, the production rules.
pub const DEFAULT_BOUND_MODE: &str = "snap+prod";
/// `e3_schema` of the `e3_shadow` and `e3_label` lines.
pub const SHADOW_SCHEMA: &str = "trade-e3-shadow.v1";
/// The longest window any fit reads: the calibration window plus the refit
/// window. Samples, routes and dispatch rows older than
/// this are dropped.
pub const RETAIN_S: f64 = crate::e3::CAL_S + crate::e3::WINDOW_S;
/// Size bins of the session host rule
/// (`prefill_completion.HOST_SIZE_EDGES`): cached prefixes up to
/// 32k, up to 128k and above.
pub const HOST_SIZE_EDGES: [i64; 2] = [32_768, 131_072];
/// Session-gap bins of the host rule in seconds
/// (`HOST_GAP_EDGES`); a session's first route to the P is its own bin.
pub const HOST_GAP_EDGES: [f64; 4] = [2.0, 5.0, 10.0, 30.0];
/// A bin whose learned host share is at least this restores the whole
/// prefix from host, else none (`HOST_RULE_SHARE`: the median
/// of a two-point choice).
pub const HOST_RULE_SHARE: f64 = 0.5;
/// Completions needed per host-share bin (online
/// `HOST_MIN_SAMPLES`).
pub const HOST_MIN_SAMPLES: usize = 20;
/// Timeline bookkeeping per P covers this many passes (bounded memory; a
/// sample that refers to an older pass is not classified).
pub const PASS_MEMORY: i64 = 8192;
/// Busy launch intervals kept for the `phi_s` samples (s): a request whose
/// prefill completes later than this after its receipt gives no sample.
pub const BUSY_KEEP_S: f64 = 300.0;
/// Router records of routed requests and their sessions kept at most this
/// many entries (each is also dropped after [`RETAIN_S`]).
pub const MAX_ROUTES: usize = 200_000;
/// Service samples older than [`RETAIN_S`] kept per P and size class for
/// the `own_all` / `pooled_all` steps of the GPU model chain (the reference uses
/// every earlier sample). A memory bound only (about 0.4 MB per class and
/// P): it binds after this many older service samples of one class on one
/// P, and then the newest ones are kept.
pub const SERVICE_HISTORY_PER_CLASS: usize = 8192;
/// The engine policy (pass budget `chunk_tokens`, page size) of the
/// forecasts that fitted the shipped bound tables: the replay simulated
/// with the reference values ([`crate::e3::CHUNK_TOKENS`],
/// [`crate::e3::PAGE_TOKENS`]), the settings of the replayed GB300 GLM-5.2 P
/// engines (`--chunked-prefill-size 8192`, page 64; the recorded
/// snapshot). A bound file does not record it. A P whose snapshot has
/// another policy gets no forecast (`policy_not_bound_fit`): its error
/// bound is not calibrated.
pub const BOUND_FIT_POLICY: (i64, i64) = (crate::e3::CHUNK_TOKENS, crate::e3::PAGE_TOKENS);

/// Unix seconds of a [`SystemTime`].
fn unix_s(at: SystemTime) -> f64 {
    at.duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

fn positive_int(v: Option<&Value>) -> Option<i64> {
    nonneg_int(v).filter(|x| *x > 0)
}

/// A finite JSON number >= 0.
fn nonneg_f64(v: Option<&Value>) -> Option<f64> {
    v.and_then(Value::as_f64)
        .filter(|x| x.is_finite() && *x >= 0.0)
}

// ---------------------------------------------------------------------------
// Configuration: the error-bound table.
// ---------------------------------------------------------------------------

/// `--co-schedule-trade-wait e3` settings that are not `Copy`
/// ([`crate::scheduler::co_schedule::TradeParams`] holds the gate numbers).
#[derive(Debug, Clone)]
pub struct TradeE3Config {
    /// Error-bound table of [`Self::mode`] (table v2).
    pub table: BoundTable,
    /// Bound level used by the gate (`"0.8"` or `"0.9"`).
    pub level: String,
    /// Deploy mode whose table is used.
    pub mode: String,
    /// The bound file and the sha256 of its bytes.
    pub bound_path: PathBuf,
    pub bound_sha256: String,
    /// `router-e3-bound-params.v2` (bound-params.json) or `bound-tables`
    /// (the map of mode to table of bound-tables.json).
    pub bound_schema: String,
    /// The level that bound-params.json ships for the mode (None for a
    /// bound-tables.json file).
    pub file_level: Option<String>,
    /// `g_min_s` recorded in bound-params.json (the threshold of its pair
    /// statistics; the gate uses `--co-schedule-trade-gate-gmin-ms`).
    pub file_g_min_s: Option<f64>,
    /// sha256 of the bound-tables.json that bound-params.json was cut from.
    pub file_tables_sha256: Option<String>,
    /// rq margin m (s): a router or bootstrap row is kept only when its
    /// ready time is at most the target's ready time minus m.
    pub rq_margin_s: f64,
    /// `--co-schedule-trade-e3-shadow`: the shadow log; None = the cost trace.
    pub shadow_path: Option<PathBuf>,
    /// `--co-schedule-trade-e3-push-port`: the state push; None = poll only.
    pub push: Option<PushConfig>,
    /// `--co-schedule-trade-e3-next-pass-lag`: forecasts take
    /// [`crate::e3::Params::next_pass_lag`] (off = the reference).
    pub next_pass_lag: bool,
    /// `--co-schedule-trade-e3-frozen-fit`: offline warmup constants in
    /// place of the online GPU, launch and lag fits (None: the online fits).
    pub frozen_fit: Option<Arc<FrozenFit>>,
}

/// `--co-schedule-trade-e3-push-port`: the port of each P's state push and
/// the error-bound table of the forecasts that start from a pushed state.
#[derive(Debug, Clone)]
pub struct PushConfig {
    /// The router connects to `tcp://<P host>:port` (routing rank 0).
    pub port: u16,
    /// Deploy mode of the push table (`--co-schedule-trade-e3-push-bound-mode`;
    /// default: the poll mode).
    pub mode: String,
    pub level: String,
    pub table: BoundTable,
    /// The level that bound-params.json ships for the mode.
    pub file_level: Option<String>,
}

/// One deploy mode's table read from a bound file.
struct ModeTable {
    table: BoundTable,
    file_level: Option<String>,
    file_g_min_s: Option<f64>,
    tables_sha256: Option<String>,
    schema: String,
}

/// The table of `mode` in bound-params.json or bound-tables.json.
fn mode_table(value: &Value, path: &Path, mode: &str) -> Result<ModeTable, String> {
    let params_schema =
        value.get("schema").and_then(Value::as_str) == Some(crate::e3::BOUND_PARAMS_SCHEMA);
    if params_schema {
        let params: BoundParams = serde_json::from_value(value.clone())
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let m = params.modes.get(mode).ok_or_else(|| {
            format!(
                "{}: no deploy mode {mode} (modes: {})",
                path.display(),
                params.modes.keys().cloned().collect::<Vec<_>>().join(", ")
            )
        })?;
        Ok(ModeTable {
            table: m.table.clone(),
            file_level: Some(m.level.clone()),
            file_g_min_s: Some(params.g_min_s),
            tables_sha256: Some(params.bound_tables_sha256.clone()),
            schema: crate::e3::BOUND_PARAMS_SCHEMA.to_string(),
        })
    } else {
        let mut tables: BTreeMap<String, BoundTable> = serde_json::from_value(value.clone())
            .map_err(|e| {
                format!(
                    "{}: not bound-params.json or bound-tables.json: {e}",
                    path.display()
                )
            })?;
        let names = tables.keys().cloned().collect::<Vec<_>>().join(", ");
        let table = tables
            .remove(mode)
            .ok_or_else(|| format!("{}: no mode {mode} (modes: {names})", path.display()))?;
        Ok(ModeTable {
            table,
            file_level: None,
            file_g_min_s: None,
            tables_sha256: None,
            schema: "bound-tables".to_string(),
        })
    }
}

/// The level of a table: the explicit one, else the file's level of the mode.
fn resolve_level(
    path: &Path,
    level: Option<&str>,
    file_level: Option<&str>,
) -> Result<String, String> {
    match (level, file_level) {
        (Some(l), _) => Ok(l.to_string()),
        (None, Some(l)) => Ok(l.to_string()),
        (None, None) => Err(format!(
            "{}: a bound-tables.json file has no level; give \
             --co-schedule-trade-e3-bound-level",
            path.display()
        )),
    }
}

/// The deploy modes whose inputs match a forecast of the router:
/// `<state>+prod` (the router calibration `rcal` and the production rules) or `<state>+prodL` (the lag-corrected calibration `rcalL`, the
/// nearest table when the state model has no `prod` one). A poll forecast
/// takes the state `snap`; a forecast from a pushed state also `push`,
/// `pushL` or `pushslow` (the push as built, with launch messages, with 60 ms
/// delivery). Refused: the `gpu` and `split` modes (inputs the router does
/// not have: GPU times, an exact zero-lag indexer split; their bounds are
/// up to 1.9x narrower), `fresh` (a zero-age bound) and the r3 modes.
fn check_mode(mode: &str, pushed: bool) -> Result<(), String> {
    let states: &[&str] = if pushed {
        &["snap", "push", "pushL", "pushslow"]
    } else {
        &["snap"]
    };
    let kind = if pushed { "pushed-state" } else { "poll" };
    match mode.split_once('+') {
        Some((state, "prod" | "prodL")) if states.contains(&state) => Ok(()),
        _ => Err(format!(
            "mode {mode} does not match the router's {kind} forecasts: use \
             <state>+prod or <state>+prodL with <state> one of {} (the router has \
             no GPU times and no exact indexer split; fresh and the r3 modes are \
             offline references)",
            states.join(", ")
        )),
    }
}

impl TradeE3Config {
    /// Read the bound file and check that the table of `mode` gives a bound
    /// at `level` (or at the file's level) for every bucket.
    ///
    /// The file is either `bound-params.json` (schema
    /// `router-e3-bound-params.v2`: per deploy mode a level and a table) or
    /// `bound-tables.json` (a map of mode to table, without a level; then
    /// `level` is required). `mode` must be a poll mode (`snap+prod` or
    /// `snap+prodL`; `check_mode`).
    pub fn load(
        path: &Path,
        mode: &str,
        level: Option<&str>,
        rq_margin_s: f64,
        shadow_path: Option<PathBuf>,
    ) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let value: Value =
            serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        let sha = crate::sha256_hex(&bytes);
        let ModeTable {
            table,
            file_level,
            file_g_min_s,
            tables_sha256: tables_sha,
            schema,
        } = mode_table(&value, path, mode)?;
        check_mode(mode, false).map_err(|e| format!("{}: {e}", path.display()))?;
        let level = resolve_level(path, level, file_level.as_deref())?;
        check_table(&table, &level).map_err(|e| format!("{} mode {mode}: {e}", path.display()))?;
        if !(rq_margin_s.is_finite() && rq_margin_s >= 0.0) {
            return Err("the rq margin must be finite and at least 0".into());
        }
        Ok(Self {
            table,
            level,
            mode: mode.to_string(),
            bound_path: path.to_path_buf(),
            bound_sha256: sha,
            bound_schema: schema,
            file_level,
            file_g_min_s,
            file_tables_sha256: tables_sha,
            rq_margin_s,
            shadow_path,
            push: None,
            next_pass_lag: false,
            frozen_fit: None,
        })
    }

    /// `--co-schedule-trade-e3-frozen-fit`: every (P, step) fit takes the
    /// GPU, launch and lag parameters of `frozen` ([`FrozenFit::apply`]).
    pub fn with_frozen_fit(mut self, frozen: Option<Arc<FrozenFit>>) -> Self {
        self.frozen_fit = frozen;
        self
    }

    /// `--co-schedule-trade-e3-next-pass-lag`: on a busy P, the result of a
    /// request whose last pass leaves no known work waits for the next pass
    /// of the expected arrivals ([`crate::e3::Params::next_pass_lag`]).
    pub fn with_next_pass_lag(mut self, on: bool) -> Self {
        self.next_pass_lag = on;
        self
    }

    /// Turn the state push on: the router subscribes to `tcp://<P host>:port`
    /// of each P, and a forecast from a pushed state takes the bound table
    /// of `mode` (None: the poll mode) at `level` (None: the file's level of
    /// that mode). The table comes from the same bound file, which must
    /// still have the sha256 read at [`Self::load`]. `mode` must be a
    /// pushed-state mode (state `snap`, `push`, `pushL` or `pushslow`, rules
    /// `prod` or `prodL`).
    pub fn with_push(
        mut self,
        port: u16,
        mode: Option<&str>,
        level: Option<&str>,
    ) -> Result<Self, String> {
        if port == 0 {
            return Err("the state push port must be 1 to 65535".into());
        }
        let path = self.bound_path.clone();
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if crate::sha256_hex(&bytes) != self.bound_sha256 {
            return Err(format!(
                "{}: the bound file changed while it was read",
                path.display()
            ));
        }
        let value: Value =
            serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        let mode = mode.unwrap_or(self.mode.as_str()).to_string();
        let t = mode_table(&value, &path, &mode)?;
        check_mode(&mode, true).map_err(|e| format!("{}: {e}", path.display()))?;
        let level = resolve_level(&path, level, t.file_level.as_deref())?;
        check_table(&t.table, &level)
            .map_err(|e| format!("{} mode {mode}: {e}", path.display()))?;
        self.push = Some(PushConfig {
            port,
            mode,
            level,
            table: t.table,
            file_level: t.file_level,
        });
        Ok(self)
    }

    /// Error bound of a forecast at the configured level (None without a
    /// forecast).
    pub fn bound(&self, out: &Output) -> Option<f64> {
        error_bound(&BucketView::from(out), &self.table, &self.level)
            .ok()
            .flatten()
    }

    /// The bucket of a forecast in the table's scheme.
    pub fn bucket(&self, out: &Output) -> &'static str {
        crate::e3::bucket_of(&BucketView::from(out), self.table.scheme)
    }

    /// The (table, level, mode) of a forecast's state source: the push
    /// table when `pushed` and the push is on, else the poll table.
    fn source_table(&self, pushed: bool) -> (&BoundTable, &str, &str) {
        match (&self.push, pushed) {
            (Some(p), true) => (&p.table, p.level.as_str(), p.mode.as_str()),
            _ => (&self.table, self.level.as_str(), self.mode.as_str()),
        }
    }

    /// [`Self::bound`] with the table of the forecast's state source.
    pub fn bound_for(&self, pushed: bool, out: &Output) -> Option<f64> {
        let (table, level, _) = self.source_table(pushed);
        error_bound(&BucketView::from(out), table, level)
            .ok()
            .flatten()
    }

    /// [`Self::bucket`] with the table of the forecast's state source.
    pub fn bucket_for(&self, pushed: bool, out: &Output) -> &'static str {
        crate::e3::bucket_of(&BucketView::from(out), self.source_table(pushed).0.scheme)
    }

    /// The deploy mode whose table bounds a forecast of this state source.
    pub fn mode_for(&self, pushed: bool) -> &str {
        self.source_table(pushed).2
    }
}

/// The source label of a parameter from `--co-schedule-trade-e3-frozen-fit`
/// (the GPU sources of the `e3_fit` trace event).
pub const FROZEN_SOURCE: &str = "frozen";

/// `--co-schedule-trade-e3-frozen-fit FILE`: offline warmup constants (the
/// `e3_constants` of the offline step model) that replace the online GPU,
/// launch and lag fits of every P; the online state still gives the arrival
/// rates. For a P that is never idle: the online GPU fit takes only single
/// passes launched after an idle gap ([`crate::e3::GPU_FIT_MIN`] per class),
/// so a saturated P (Kimi-K2.6 under CUA load) keeps no fit and every
/// forecast falls back (`no_fit`).
#[derive(Debug, Clone, PartialEq)]
pub struct FrozenFit {
    /// The file and the sha256 of its bytes.
    pub path: PathBuf,
    pub sha256: String,
    /// One GPU model for idle and busy passes (`gpu_busy`; a `gpu_idle` in
    /// the file must equal it).
    pub gpu: GpuModel,
    /// The launch model (anchor `launch`, 6 coefficients) and its residual
    /// quantiles.
    pub launch_coef: Vec<f64>,
    pub launch_resid_q: Option<Vec<f64>>,
    pub constants: Constants,
}

/// The `e3_constants` object of the offline step model (other keys, such as
/// `units` and `notes`, are ignored).
#[derive(serde::Deserialize)]
struct FrozenFitFile {
    gpu_busy: GpuModel,
    #[serde(default)]
    gpu_idle: Option<GpuModel>,
    launch_coef: Vec<f64>,
    #[serde(default)]
    launch_anchor: Option<String>,
    #[serde(default)]
    launch_resid_q: Option<Vec<f64>>,
    lag_next: [f64; 2],
    lag_none_s: f64,
    boot_s: f64,
    idle_delay_s: f64,
    #[serde(default)]
    phi_s: f64,
    #[serde(default)]
    chunk_tokens: Option<i64>,
    #[serde(default)]
    page_tokens: Option<i64>,
    #[serde(default)]
    graph_tokens: Option<i64>,
}

impl FrozenFit {
    /// Read `path`: the offline step model (its `kimi.e3_constants` or
    /// `e3_constants`) or the `e3_constants` object itself. Refused: no GPU
    /// class, a class without 5 finite coefficients, a floor that is not
    /// finite and at least 0, a `gpu_idle` other than `gpu_busy`, a launch
    /// model without 6 finite coefficients or with an anchor other than
    /// `launch`, a constant that is not finite, and a pass budget, page or
    /// graph size other than the router's.
    pub fn load(path: &Path) -> Result<Self, String> {
        use crate::e3::{CHUNK_TOKENS, GRAPH_TOKENS, PAGE_TOKENS};
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let value: Value =
            serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        let body = value
            .pointer("/kimi/e3_constants")
            .or_else(|| value.get("e3_constants"))
            .unwrap_or(&value)
            .clone();
        let file: FrozenFitFile =
            serde_json::from_value(body).map_err(|e| format!("{}: {e}", path.display()))?;
        let refuse =
            |what: &str| -> Result<Self, String> { Err(format!("{}: {what}", path.display())) };
        let finite = |xs: &[f64]| xs.iter().all(|x| x.is_finite());
        let gpu = file.gpu_busy;
        if gpu.small.is_none() && gpu.large.is_none() {
            return refuse("gpu_busy has neither a small nor a large class");
        }
        for (name, class) in [("small", &gpu.small), ("large", &gpu.large)] {
            if class.as_ref().is_some_and(|c| c.len() != 5 || !finite(c)) {
                return refuse(&format!(
                    "gpu_busy.{name} needs 5 finite coefficients (ms over [1, new tokens, \
                     attention pairs / 1e6, host tokens, batch size])"
                ));
            }
        }
        if !(gpu.floor_ms.is_finite() && gpu.floor_ms >= 0.0) {
            return refuse("gpu_busy.floor_ms must be finite and at least 0");
        }
        if file.gpu_idle.as_ref().is_some_and(|idle| *idle != gpu) {
            return refuse(
                "gpu_idle differs from gpu_busy (the router prices idle and busy passes with one \
                 GPU model)",
            );
        }
        if file.launch_coef.len() != 6 || !finite(&file.launch_coef) {
            return refuse("launch_coef needs 6 finite coefficients");
        }
        if file.launch_anchor.is_some_and(|a| a != "launch") {
            return refuse("launch_anchor must be launch (the router's launch model)");
        }
        if file.launch_resid_q.as_ref().is_some_and(|q| !finite(q)) {
            return refuse("launch_resid_q must be finite");
        }
        let constants = Constants {
            idle_delay_s: file.idle_delay_s,
            phi_s: file.phi_s,
            boot_s: file.boot_s,
            lag_none_s: file.lag_none_s,
            lag_next: file.lag_next,
        };
        let c = constants;
        let scalars = [c.idle_delay_s, c.phi_s, c.boot_s, c.lag_none_s];
        if !finite(&scalars) || !finite(&c.lag_next) {
            return refuse("idle_delay_s, phi_s, boot_s, lag_none_s and lag_next must be finite");
        }
        for (name, got, want) in [
            ("chunk_tokens", file.chunk_tokens, CHUNK_TOKENS),
            ("page_tokens", file.page_tokens, PAGE_TOKENS),
            ("graph_tokens", file.graph_tokens, GRAPH_TOKENS),
        ] {
            if let Some(got) = got.filter(|got| *got != want) {
                return refuse(&format!("{name} {got} is not the router's {want}"));
            }
        }
        Ok(Self {
            path: path.to_path_buf(),
            sha256: crate::sha256_hex(&bytes),
            gpu,
            launch_coef: file.launch_coef,
            launch_resid_q: file.launch_resid_q,
            constants,
        })
    }

    /// Replace the GPU model, the launch model and the warmup constants of
    /// an online fit (its arrival rates and sample counts stay).
    pub fn apply(&self, fit: &mut FitBundle) {
        let source = |class: &Option<Vec<f64>>| class.as_ref().map(|_| FROZEN_SOURCE.to_string());
        fit.gpu_sources = GpuSources {
            small: source(&self.gpu.small),
            large: source(&self.gpu.large),
        };
        fit.gpu = self.gpu.clone();
        fit.launch = LaunchFit {
            coef: self.launch_coef.clone(),
            resid_q: self.launch_resid_q.clone(),
            source: crate::e3::LaunchSource::Frozen,
        };
        fit.constants = self.constants;
    }

    /// The values in one line (the start-up log).
    pub fn summary(&self) -> String {
        let class = |c: &Option<Vec<f64>>| match c {
            Some(v) => format!("{v:?}"),
            None => "none".to_string(),
        };
        let c = &self.constants;
        format!(
            "gpu small {} large {} floor {} ms; launch {:?}; lag_next {:?}; lag_none {} s; \
             boot {} s; idle_delay {} s; phi {} s",
            class(&self.gpu.small),
            class(&self.gpu.large),
            self.gpu.floor_ms,
            self.launch_coef,
            c.lag_next,
            c.lag_none_s,
            c.boot_s,
            c.idle_delay_s,
            c.phi_s,
        )
    }
}

/// Every bucket of the table's scheme must resolve to a valid curve at
/// `level` (the bucket's own curve or `all`): knots finite, positive and
/// strictly increasing; bounds finite, at least 0 and non-decreasing.
fn check_table(table: &BoundTable, level: &str) -> Result<(), String> {
    let curves = table
        .levels
        .get(level)
        .ok_or_else(|| format!("no level {level}"))?;
    let buckets: &[&str] = match table.scheme {
        crate::e3::BucketScheme::Idle => &["idle", "busy"],
        crate::e3::BucketScheme::Wait => &["idle", "busy_w0", "busy_w1", "busy_w2"],
        crate::e3::BucketScheme::Ahead => &["idle", "busy_a0", "busy_a1", "busy_a2"],
    };
    for bucket in buckets {
        let curve = curves
            .get(*bucket)
            .or_else(|| curves.get("all"))
            .ok_or_else(|| format!("level {level}: no curve {bucket} or all"))?;
        let (k, b) = (&curve.knots_s, &curve.b_s);
        let ok = !k.is_empty()
            && k.len() == b.len()
            && k.iter().all(|x| x.is_finite() && *x > 0.0)
            && b.iter().all(|x| x.is_finite() && *x >= 0.0)
            && k.windows(2).all(|w| w[1] > w[0])
            && b.windows(2).all(|w| w[1] >= w[0]);
        if !ok {
            return Err(format!(
                "level {level}: curve {bucket} is not a valid table v2 curve"
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The engine snapshot as E3 inputs.
// ---------------------------------------------------------------------------

/// A pending in-flight pass of the snapshot, relative to its capture.
#[derive(Debug, Clone, PartialEq)]
pub struct RelPass {
    /// The pass number (`forward_iter`), when the snapshot gives it.
    pub forward_iter: Option<i64>,
    pub launch_age_s: f64,
    pub new_tokens: i64,
    pub attention_pairs: i64,
    pub host_tokens: i64,
    pub batch_size: i64,
    pub finishing: i64,
}

/// A waiting or bootstrap row of the snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapRow {
    pub rid: String,
    pub input: i64,
    /// `input_tokens - remaining_compute_tokens`; None when the engine
    /// publishes no prefix for the row (rows outside the lookahead plan).
    pub prefix: Option<i64>,
    pub host: Option<i64>,
    pub recv_age_s: Option<f64>,
    /// Tokens of a pending pull of the row (`plan.pull_tokens`; 0 without
    /// one). The engine's `remaining_compute_tokens` counts them as compute
    /// until the pull lands in the host pool.
    pub pull: i64,
}

/// Kinds of the engine timeline samples (`prefill_timeline.py`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleKind {
    /// `launch(k + 1) - launch(k)` of two passes without an idle gap
    /// between them (`forward_iter` = k).
    LaunchInterval,
    /// Batch formation to launch of a request's first pass (`forward_iter`
    /// = that pass).
    EntryToLaunch,
    /// Launch of a request's last pass to its completion.
    LastLaunchToDone,
    /// Receipt to completion of a request, with its cache tiers.
    RequestComplete,
}

/// One parsed timeline sample; times are relative to the capture.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TimelineRow {
    pub sequence: u64,
    pub kind: Option<SampleKind>,
    pub forward_iter: Option<i64>,
    pub rid: Option<String>,
    pub duration_s: Option<f64>,
    pub completed_age_s: Option<f64>,
    pub new_tokens: Option<i64>,
    pub attention_pairs: Option<i64>,
    pub batch_size: Option<i64>,
    pub host_tokens: Option<i64>,
    pub finishing: Option<i64>,
    pub next_host_tokens: Option<i64>,
    pub input_tokens: Option<i64>,
    pub receive_to_entry_s: Option<f64>,
    pub cached_device: Option<i64>,
    pub cached_host: Option<i64>,
    pub cached_storage: Option<i64>,
    pub breakdown: bool,
}

impl TimelineRow {
    /// None without an integer sequence (the row cannot be ordered).
    pub fn parse(v: &Value) -> Option<Self> {
        let sequence = v.get("sequence").and_then(Value::as_u64)?;
        let f = v.get("features");
        let fi = |k: &str| nonneg_int(f.and_then(|f| f.get(k)));
        let ff = |k: &str| nonneg_f64(f.and_then(|f| f.get(k)));
        let kind = match v.get("kind").and_then(Value::as_str) {
            Some("launch_interval") => Some(SampleKind::LaunchInterval),
            Some("entry_to_launch") => Some(SampleKind::EntryToLaunch),
            Some("last_launch_to_done") => Some(SampleKind::LastLaunchToDone),
            Some("request_complete") => Some(SampleKind::RequestComplete),
            _ => None,
        };
        Some(Self {
            sequence,
            kind,
            forward_iter: v
                .get("forward_iter")
                .and_then(Value::as_i64)
                .filter(|k| *k >= 0),
            rid: v.get("rid").and_then(Value::as_str).map(str::to_string),
            duration_s: nonneg_f64(v.get("duration_s")),
            completed_age_s: nonneg_f64(v.get("completed_age_s")),
            new_tokens: fi("new_tokens"),
            attention_pairs: fi("attention_pairs"),
            batch_size: fi("batch_size"),
            host_tokens: fi("host_restore_tokens"),
            finishing: fi("finishing_requests"),
            next_host_tokens: fi("next_host_restore_tokens"),
            input_tokens: fi("input_tokens"),
            receive_to_entry_s: ff("receive_to_entry_s"),
            cached_device: fi("cached_tokens_device"),
            cached_host: fi("cached_tokens_host"),
            cached_storage: fi("cached_tokens_storage"),
            breakdown: is_true(f.and_then(|f| f.get("cache_breakdown_computed"))),
        })
    }

    /// Pass features `[1, new, pairs / 1e6, host, bs]` of the sample's pass
    /// (the gated `host_restore_tokens`).
    pub fn pass_features(&self) -> Option<Features> {
        Some(features(
            self.new_tokens? as f64,
            self.attention_pairs? as f64,
            self.host_tokens? as f64,
            self.batch_size? as f64,
        ))
    }
}

/// One parsed `prefill_routing` snapshot in E3 form. Parsed once
/// per poll, off the cost-book lock.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct E3View {
    pub epoch: Option<String>,
    pub sequence: Option<u64>,
    /// The engine's snapshot age at the reply (`age_s`).
    pub age_s: Option<f64>,
    /// The state time on the engine's monotonic clock
    /// (`sampled_monotonic_s`; the state push maps it to the router clock).
    pub sampled_mono_s: Option<f64>,
    /// Engine policy: the KV page size and the pass budget
    /// (`chunked_prefill_size`) of the simulated passes, and
    /// `max_prefill_tokens` (`max_input_tokens`; None when not published).
    /// A view without page or chunk is unusable (`policy_unknown`).
    pub page: i64,
    pub chunk: i64,
    pub max_input: Option<i64>,
    pub inflight: Vec<RelPass>,
    pub partial: Option<Partial>,
    pub waiting: Vec<SnapRow>,
    pub bootstrap: Vec<SnapRow>,
    /// Every RID the snapshot shows: partial, waiting, bootstrap and
    /// transfer-queue rows and the members of the pending passes.
    pub shown: HashSet<String>,
    /// U of the late-D park gate; None when unknown.
    pub unbound_tokens: Option<i64>,
    /// `admission_resources.token_pool_capacity`.
    pub pool_tokens: Option<i64>,
    /// The first reason the view cannot give a forecast.
    pub unusable: Option<&'static str>,
    /// Values that the parse replaced (for the trace).
    pub degraded: BTreeSet<&'static str>,
    /// Waiting rows: (rid, receipt to waiting-queue entry in s).
    pub boots: Vec<(String, f64)>,
    /// Timeline samples in sequence order (cleared after ingestion).
    pub samples: Vec<TimelineRow>,
}

impl E3View {
    /// Parse one `loads[0].prefill_routing` value.
    pub fn parse(raw: &Value) -> Self {
        let mut v = Self {
            page: crate::e3::PAGE_TOKENS,
            chunk: crate::e3::CHUNK_TOKENS,
            ..Self::default()
        };
        if !raw.is_object() {
            v.unusable = Some("not_an_object");
            return v;
        }
        v.epoch = raw.get("epoch").and_then(Value::as_str).map(str::to_string);
        v.sequence = raw.get("sequence").and_then(Value::as_u64);
        v.age_s = nonneg_f64(raw.get("age_s"));
        v.sampled_mono_s = nonneg_f64(raw.get("sampled_monotonic_s"));
        let policy = raw.get("policy");
        let page = positive_int(policy.and_then(|p| p.get("page_size")));
        let chunk = positive_int(policy.and_then(|p| p.get("chunk_tokens")));
        v.page = page.unwrap_or(crate::e3::PAGE_TOKENS);
        v.chunk = chunk.unwrap_or(crate::e3::CHUNK_TOKENS);
        v.max_input = positive_int(policy.and_then(|p| p.get("max_input_tokens")));
        // The engine marks a snapshot stale when its age at the reply is
        // above the engine's own limit (SGLANG_PREFILL_ROUTING_SNAPSHOT_MAX_AGE_S)
        // or unknown. With a known age the router's limit on the view age
        // decides (--co-schedule-trade-eft-max-view-age-ms; the bound tables
        // were fitted on every decision, whatever the snapshot age): the flag
        // is only noted.
        let stale = is_true(raw.get("stale"));
        if stale && v.age_s.is_some() {
            v.degraded.insert("engine_stale");
        }
        // The usability rules of the EFT view, strict on any
        // omitted record, and the plain FCFS pass formation that E3 models.
        v.unusable = if !is_true(raw.get("available")) {
            Some("unavailable")
        } else if stale && v.age_s.is_none() {
            Some("stale")
        } else if raw.get("truncated").and_then(Value::as_bool) != Some(false) {
            Some("truncated")
        } else if !is_true(raw.get("inflight_batches_known")) {
            Some("inflight_unknown")
        } else if policy
            .and_then(|p| p.get("schedule_policy"))
            .and_then(Value::as_str)
            != Some("fcfs")
        {
            Some("policy_not_fcfs")
        } else if is_true(policy.and_then(|p| p.get("share_enabled"))) {
            Some("policy_share_enabled")
        } else if page.is_none() || chunk.is_none() {
            // The simulated passes need the engine's budget and page.
            Some("policy_unknown")
        } else if v.max_input.is_some_and(|m| m < v.chunk) {
            // The engine stops adding rows to a pass once they reach
            // max_prefill_tokens, before the chunk budget that E3 simulates.
            Some("policy_max_input_below_chunk")
        } else {
            None
        };

        // In-flight passes, in result-queue (launch) order.
        if let Some(batches) = raw.get("inflight_batches").and_then(Value::as_array) {
            for b in batches {
                if let Some(ids) = b.get("request_ids").and_then(Value::as_array) {
                    v.shown
                        .extend(ids.iter().filter_map(Value::as_str).map(str::to_string));
                }
                if b.get("status").and_then(Value::as_str) != Some("pending") {
                    continue;
                }
                let f = b.get("features");
                let get = |k: &str| nonneg_int(f.and_then(|f| f.get(k)));
                let (age, new, pairs, bs) = (
                    nonneg_f64(b.get("launch_age_s")),
                    get("new_tokens"),
                    get("attention_pairs"),
                    get("batch_size"),
                );
                let host = get("host_restore_tokens");
                let finishing = nonneg_int(b.get("finishing_requests"));
                match (age, new, pairs, bs) {
                    (Some(age), Some(new), Some(pairs), Some(bs)) => {
                        if host.is_none() {
                            v.degraded.insert("inflight_host_unknown");
                        }
                        if finishing.is_none() {
                            v.degraded.insert("inflight_finishing_unknown");
                        }
                        v.inflight.push(RelPass {
                            forward_iter: b.get("forward_iter").and_then(Value::as_i64),
                            launch_age_s: age,
                            new_tokens: new,
                            attention_pairs: pairs,
                            host_tokens: host.unwrap_or(0),
                            batch_size: bs,
                            finishing: finishing.unwrap_or(0),
                        });
                    }
                    _ => {
                        v.unusable = v.unusable.or(Some("inflight_features_unknown"));
                    }
                }
            }
        }

        // The partial (chunked) request.
        if let Some(p) = raw.get("partial").filter(|p| p.is_object()) {
            let rid = p.get("rid").and_then(Value::as_str);
            if let Some(rid) = rid {
                v.shown.insert(rid.to_string());
            }
            match (
                rid,
                nonneg_int(p.get("remaining_compute_tokens")),
                nonneg_int(p.get("committed_prefix_tokens")),
                nonneg_int(p.get("inflight_new_tokens")),
            ) {
                (Some(rid), Some(rem), Some(committed), Some(inflight)) => {
                    v.partial = Some(Partial {
                        rid: rid.to_string(),
                        remaining: rem.max(1),
                        prefix: committed + inflight,
                    });
                }
                _ => v.unusable = v.unusable.or(Some("partial_unknown")),
            }
        }

        // Waiting rows (stored scheduler order), then bootstrap rows.
        for (name, bootstrap) in [("waiting", false), ("bootstrap", true)] {
            let Some(rows) = raw.get(name).and_then(Value::as_array) else {
                continue;
            };
            for row in rows {
                let rid = row.get("rid").and_then(Value::as_str);
                if let Some(rid) = rid {
                    v.shown.insert(rid.to_string());
                }
                let input = nonneg_int(row.get("input_tokens"));
                let (Some(rid), Some(input)) = (rid, input) else {
                    v.unusable = v.unusable.or(Some("row_unknown"));
                    continue;
                };
                let remaining = nonneg_int(row.get("remaining_compute_tokens"));
                let prefix = match remaining {
                    Some(rem) if rem <= input => Some(input - rem),
                    Some(_) => {
                        v.degraded.insert("row_prefix_invalid");
                        None
                    }
                    None => None,
                };
                let recv_age_s = nonneg_f64(row.get("recv_age_s"));
                if !bootstrap {
                    if let (Some(recv), Some(queue)) =
                        (recv_age_s, nonneg_f64(row.get("queue_age_s")))
                    {
                        if recv >= queue {
                            v.boots.push((rid.to_string(), recv - queue));
                        }
                    }
                }
                let snap = SnapRow {
                    rid: rid.to_string(),
                    input,
                    prefix,
                    host: prefix.and(nonneg_int(row.get("host_prefix_tokens"))),
                    recv_age_s,
                    pull: nonneg_int(row.get("plan").and_then(|p| p.get("pull_tokens")))
                        .unwrap_or(0),
                };
                if bootstrap {
                    if snap.recv_age_s.is_none() {
                        v.degraded.insert("bootstrap_receipt_unknown");
                    }
                    v.bootstrap.push(snap);
                } else {
                    v.waiting.push(snap);
                }
            }
        }
        if let Some(rows) = raw.get("inflight_requests").and_then(Value::as_array) {
            v.shown.extend(
                rows.iter()
                    .filter_map(|r| r.get("rid").and_then(Value::as_str))
                    .map(str::to_string),
            );
        }

        let adm = raw.get("admission_resources");
        v.pool_tokens = positive_int(adm.and_then(|a| a.get("token_pool_capacity")));
        v.unbound_tokens = nonneg_int(adm.and_then(|a| a.get("unbound_prompt_tokens_last_scan")))
            .or_else(|| unbound_from_rows(raw));

        if let Some(samples) = raw
            .get("timeline")
            .and_then(|t| t.get("samples"))
            .and_then(Value::as_array)
        {
            v.samples = samples.iter().filter_map(TimelineRow::parse).collect();
            v.samples.sort_by_key(|r| r.sequence);
        }
        v
    }

    /// The late-D park cap of the P: `PARK_CAP x token pool`,
    /// with the replay's default pool when the snapshot has none.
    pub fn cap_tokens(&self) -> f64 {
        crate::e3::PARK_CAP * self.pool_tokens.unwrap_or(crate::e3::DEFAULT_POOL_TOKENS) as f64
    }
}

/// Fallback of U: the input tokens of the rows launched without a
/// decode worker and not bound; None when no row says.
fn unbound_from_rows(raw: &Value) -> Option<i64> {
    let mut rows: Vec<&Value> = Vec::new();
    if let Some(p) = raw.get("partial").filter(|p| p.is_object()) {
        rows.push(p);
    }
    for name in ["waiting", "bootstrap", "inflight_requests"] {
        if let Some(list) = raw.get(name).and_then(Value::as_array) {
            rows.extend(list.iter());
        }
    }
    let mut known = false;
    let mut total = 0i64;
    for row in rows {
        let state = row.get("state");
        match state
            .and_then(|s| s.get("launched_unbound"))
            .and_then(Value::as_bool)
        {
            Some(true) => {
                known = true;
                if !is_true(state.and_then(|s| s.get("bound"))) {
                    total += nonneg_int(row.get("input_tokens")).unwrap_or(0);
                }
            }
            Some(false) => known = true,
            None => {}
        }
    }
    known.then_some(total)
}

// ---------------------------------------------------------------------------
// Input building.
// ---------------------------------------------------------------------------

/// A request the router sent to the P that the snapshot does not show, with
/// the router's estimate at its decision.
#[derive(Debug, Clone, PartialEq)]
pub struct RouterRow {
    pub rid: String,
    pub input: i64,
    pub prefix: i64,
    pub host: i64,
    /// Decision time on the router clock (s).
    pub decided_s: f64,
}

/// Counts of the input building, for the trace.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BuildNotes {
    /// Router rows given (before the rq cut).
    pub router_rows: usize,
    /// Router rows left out because they are ready after the target.
    pub router_rows_cut: usize,
    /// Bootstrap rows left out by the same rule.
    pub bootstrap_cut: usize,
    /// Snapshot rows without a published prefix that took the router's
    /// estimate.
    pub null_estimated: usize,
    /// Snapshot rows without a published prefix and without an estimate
    /// (cold).
    pub null_cold: usize,
    /// Snapshot rows with a pending pull whose tokens were moved from
    /// compute to the cached prefix restored from host, and those tokens.
    pub pull_rows: usize,
    pub pull_tokens: i64,
}

/// The E3 input of one candidate.
///
/// `t0` = poll receipt - `age_s`; `d` = the decision time (both router clock
/// seconds). `estimate(rid)` gives the router's (prefix, host) of a RID that
/// it routed, for the null-prefix rule. A snapshot row with a pending pull
/// has the pulled tokens in its prefix and host (capped at its input).
/// `rq_margin_s` None keeps every router and bootstrap row (`rqall`).
#[allow(clippy::too_many_arguments)]
pub fn build_input(
    view: &E3View,
    t0: f64,
    d: f64,
    target: Target,
    router_rows: &[RouterRow],
    estimate: &dyn Fn(&str) -> Option<(i64, i64)>,
    params: &Params,
    rq_margin_s: Option<f64>,
) -> (Input, BuildNotes) {
    let mut notes = BuildNotes {
        router_rows: router_rows.len(),
        ..BuildNotes::default()
    };
    let inflight: Vec<InflightPass> = view
        .inflight
        .iter()
        .map(|p| InflightPass {
            launch: t0 - p.launch_age_s,
            new_tokens: p.new_tokens,
            attention_pairs: p.attention_pairs,
            host_tokens: p.host_tokens,
            batch_size: p.batch_size,
            finishing: p.finishing,
        })
        .collect();
    let mut fill = |row: &SnapRow| -> (Option<i64>, Option<i64>) {
        match row.prefix {
            // A pending pull: its tokens are cached, restored from host, when
            // the row runs (as for the target and the router rows). The
            // engine's remaining compute still counts them.
            Some(prefix) if row.pull > 0 && prefix < row.input => {
                let pulled = row.pull.min(row.input - prefix);
                notes.pull_rows += 1;
                notes.pull_tokens += pulled;
                (Some(prefix + pulled), Some(row.host.unwrap_or(0) + pulled))
            }
            Some(prefix) => (Some(prefix), row.host),
            None => match estimate(row.rid.as_str()) {
                Some((prefix, host)) => {
                    notes.null_estimated += 1;
                    (Some(prefix), Some(host))
                }
                None => {
                    notes.null_cold += 1;
                    (None, None)
                }
            },
        }
    };
    let mut queue: Vec<QueueRow> =
        Vec::with_capacity(view.waiting.len() + view.bootstrap.len() + router_rows.len());
    for row in &view.waiting {
        let (prefix, host) = fill(row);
        queue.push(QueueRow {
            rid: row.rid.clone(),
            source: RowSource::Snapshot,
            input: row.input,
            prefix,
            host,
            ready: None,
            decision_time: None,
        });
    }
    let boot = params.boot_s;
    // (ready, decision time, is a router row, row)
    let mut later: Vec<(f64, f64, bool, QueueRow)> = Vec::new();
    for row in &view.bootstrap {
        let (prefix, host) = fill(row);
        // Rule `bootest`: ready = receipt + boot_s (receipt = t0 - recv_age).
        let ready = t0 - row.recv_age_s.unwrap_or(0.0) + boot;
        later.push((
            ready,
            f64::NEG_INFINITY,
            false,
            QueueRow {
                rid: row.rid.clone(),
                source: RowSource::Snapshot,
                input: row.input,
                prefix,
                host,
                ready: Some(ready),
                decision_time: None,
            },
        ));
    }
    for r in router_rows {
        let row = QueueRow {
            rid: r.rid.clone(),
            source: RowSource::Dispatched,
            input: r.input,
            prefix: Some(r.prefix),
            host: Some(r.host),
            ready: None,
            decision_time: Some(r.decided_s),
        };
        // decision + rxd(input) + boot_s; never an error here.
        let ready = row_ready(&row, params)
            .ok()
            .flatten()
            .unwrap_or(f64::NEG_INFINITY);
        later.push((ready, r.decided_s, true, row));
    }
    if let Some(margin) = rq_margin_s {
        let (line, _) = receipt_delays(params, target.input);
        let x_ready = d + line + boot;
        later.retain(|(ready, _, router, _)| {
            // Python: `ready > x_ready - margin` leaves the row out.
            let cut = *ready > x_ready - margin;
            if cut {
                if *router {
                    notes.router_rows_cut += 1;
                } else {
                    notes.bootstrap_cut += 1;
                }
            }
            !cut
        });
    }
    // Python `sort(key=(ready, decision_time))`: stable, -0.0 == 0.0.
    later.sort_by(|a, b| py_cmp(&a.0, &b.0).then_with(|| py_cmp(&a.1, &b.1)));
    queue.extend(later.into_iter().map(|(_, _, _, row)| row));
    (
        Input {
            d,
            t0,
            inflight,
            partial: view.partial.clone(),
            queue,
            unbound_tokens: view.unbound_tokens,
            target,
        },
        notes,
    )
}

// ---------------------------------------------------------------------------
// Fits per (P, step).
// ---------------------------------------------------------------------------

/// Copies of the sample windows of one (P, step) fit; every row has a time
/// key below the step (causal).
#[derive(Debug, Clone, Default)]
pub struct FitData {
    /// Service samples of the P in `[step - WINDOW_S, step)`.
    pub own_window: Vec<GpuSample>,
    /// The same window of every P (P in URL order, each in time order).
    pub pooled_window: Vec<GpuSample>,
    /// Service samples of the P's calibration window.
    pub own_cal: Vec<GpuSample>,
    /// The calibration-window samples of every P.
    pub pooled_cal: Vec<GpuSample>,
    /// Every earlier service sample of the P / of every P: the per-class
    /// history ([`SERVICE_HISTORY_PER_CLASS`]) and the retained window.
    pub own_all: Vec<GpuSample>,
    pub pooled_all: Vec<GpuSample>,
    /// Launch-interval samples of the P in the window and in its
    /// calibration window, and the durations of all its retained ones.
    pub launch_window: Vec<LaunchSample>,
    pub launch_cal: Vec<LaunchSample>,
    pub launch_ys: Vec<f64>,
    /// Next-aware lag samples of the calibration window.
    pub lag_cal: Vec<LagSample>,
    /// Calibration-window `boot_s`, `phi_s` and idle-delay samples (s).
    pub boots: Vec<f64>,
    pub phis: Vec<f64>,
    pub idles: Vec<f64>,
    /// The router's dispatch log of the P in the window (time order).
    pub dispatch: Vec<ArrivalRow>,
}

/// Sample counts of a fit, for the trace.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FitCounts {
    pub service_window: usize,
    pub service_pooled: usize,
    pub service_cal: usize,
    pub launch_window: usize,
    pub launch_cal: usize,
    pub lag_cal: usize,
    pub boots: usize,
    pub phis: usize,
    pub idles: usize,
    pub dispatch: usize,
}

/// The fitted parameters of one (P, step).
#[derive(Debug, Clone)]
pub struct FitBundle {
    pub step: f64,
    pub gpu: GpuModel,
    pub gpu_sources: GpuSources,
    pub launch: LaunchFit,
    pub constants: Constants,
    pub arrivals: Option<Arrivals>,
    pub counts: FitCounts,
}

impl FitBundle {
    /// A forecast needs a GPU model with at least one class.
    pub fn usable(&self) -> bool {
        self.gpu.small.is_some() || self.gpu.large.is_some()
    }

    /// Forecast parameters with the receipt line and park cap of the
    /// decision (`gpu_idle` = `gpu_busy` in rcal).
    pub fn params(&self, rx: Option<RxParams>, cap_tokens: f64) -> Params {
        make_params(
            &self.constants,
            self.gpu.clone(),
            self.gpu.clone(),
            self.launch.coef.clone(),
            self.launch.resid_q.clone(),
            LaunchAnchor::Launch,
            rx,
            self.arrivals,
            cap_tokens,
        )
    }

    /// The fit sources for the trace.
    pub fn record(&self) -> Value {
        json!({
            "step_s": self.step,
            "gpu_small": self.gpu_sources.small,
            "gpu_large": self.gpu_sources.large,
            "launch": self.launch.source,
            "arrivals": self.arrivals.is_some(),
            "idle_delay_s": self.constants.idle_delay_s,
            "phi_s": self.constants.phi_s,
            "boot_s": self.constants.boot_s,
            "lag_next": self.constants.lag_next,
            "floor_ms": self.gpu.floor_ms,
            "samples": {
                "service_window": self.counts.service_window,
                "service_pooled": self.counts.service_pooled,
                "service_cal": self.counts.service_cal,
                "launch_window": self.counts.launch_window,
                "launch_cal": self.counts.launch_cal,
                "lag_cal": self.counts.lag_cal,
                "boots": self.counts.boots,
                "phis": self.counts.phis,
                "idles": self.counts.idles,
                "dispatch": self.counts.dispatch,
            },
        })
    }
}

fn median_or(xs: &[f64], default: f64) -> f64 {
    if xs.is_empty() {
        default
    } else {
        quantile(xs, 0.5).unwrap_or(default)
    }
}

/// The router calibration of one (P, step) (the replay's
/// `router_chain` and `params_for` with gpu = `router`).
pub fn compute_fit(step: f64, data: &FitData) -> FitBundle {
    use crate::e3::{GPU_FIT_MIN, GRAPH_TOKENS};
    let fit = |rows: &[GpuSample]| fit_gpu_classes(rows, GRAPH_TOKENS, GPU_FIT_MIN);
    let own_cal_ms: Vec<f64> = data.own_cal.iter().map(|s| s.y_ms).collect();
    let pooled_cal_ms: Vec<f64> = data.pooled_cal.iter().map(|s| s.y_ms).collect();
    // Python `own or pooled`: the own list when it is not empty.
    let floor = router_floor_ms(if own_cal_ms.is_empty() {
        &pooled_cal_ms
    } else {
        &own_cal_ms
    });
    let cal_fit: GpuClasses = fit(data.own_cal.as_slice());
    let mut chain: Vec<(String, Option<GpuClasses>)> = vec![
        ("own".into(), Some(fit(data.own_window.as_slice()))),
        ("pooled".into(), Some(fit(data.pooled_window.as_slice()))),
        ("cal".into(), Some(cal_fit.clone())),
    ];
    let (mut gpu, mut sources) = resolve_gpu(&chain, floor);
    if gpu.small.is_none() || gpu.large.is_none() {
        chain.push(("own_all".into(), Some(fit(data.own_all.as_slice()))));
        (gpu, sources) = resolve_gpu(&chain, floor);
    }
    if gpu.small.is_none() || gpu.large.is_none() {
        chain.push(("pooled_all".into(), Some(fit(data.pooled_all.as_slice()))));
        (gpu, sources) = resolve_gpu(&chain, floor);
    }
    // lag_next: the calibration-window service model (own P, else all P).
    let (gpu_cal, _) = resolve_gpu(
        &[
            ("own".to_string(), Some(cal_fit)),
            ("pooled".to_string(), Some(fit(data.pooled_cal.as_slice()))),
        ],
        floor,
    );
    let lag_next = router_lag_next(&data.lag_cal, &gpu_cal, GRAPH_TOKENS);
    // The replay's constant interval: sorted(ys)[len // 2].
    let mut ys = data.launch_ys.clone();
    sort_py(&mut ys);
    let const_median = ys.get(ys.len() / 2).copied();
    let launch = fit_launch_intervals(
        &data.launch_window,
        &gpu,
        Some(data.launch_cal.as_slice()),
        const_median,
    );
    let boot_s = median_or(&data.boots, 0.0);
    let phi_s = median_or(&data.phis, 0.0);
    // Formula: median(first launch - receipt) - boot.
    let idle_delay_s = if data.idles.is_empty() {
        crate::e3::DEFAULT_IDLE_DELAY_S
    } else {
        py_max(0.0, median_or(&data.idles, 0.0) - boot_s)
    };
    FitBundle {
        step,
        gpu,
        gpu_sources: sources,
        launch,
        constants: Constants {
            idle_delay_s,
            phi_s,
            boot_s,
            lag_none_s: 0.0,
            lag_next,
        },
        arrivals: arrival_rates(
            &data.dispatch,
            step,
            crate::e3::WINDOW_S,
            crate::e3::ARRIVAL_MIN,
        ),
        counts: FitCounts {
            service_window: data.own_window.len(),
            service_pooled: data.pooled_window.len(),
            service_cal: data.own_cal.len(),
            launch_window: data.launch_window.len(),
            launch_cal: data.launch_cal.len(),
            lag_cal: data.lag_cal.len(),
            boots: data.boots.len(),
            phis: data.phis.len(),
            idles: data.idles.len(),
            dispatch: data.dispatch.len(),
        },
    }
}

// ---------------------------------------------------------------------------
// Online state: timeline ingestion, routes, fits.
// ---------------------------------------------------------------------------

/// A routed request, kept until its `request_complete` sample arrives.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteRecord {
    pub rid: String,
    pub worker: String,
    /// Decision time: router clock and unix seconds.
    pub decided_s: f64,
    pub decided_unix_s: f64,
    pub input: i64,
    /// The router's estimate on the chosen P.
    pub prefix: i64,
    pub host: i64,
    /// The chosen P's forecast: idle (receipt line `rx2`) and `pred_s`;
    /// None without a forecast.
    pub idle: Option<bool>,
    pub pred_s: Option<f64>,
    pub session: Option<String>,
    /// Seconds since the session last went to the chosen P (None: never).
    pub session_gap_s: Option<f64>,
}

#[derive(Debug, Clone)]
struct PendingService {
    k: i64,
    rid: String,
    done: f64,
    new_tokens: i64,
    pairs: i64,
    bs: i64,
    y_ms: f64,
    host: Option<i64>,
}

#[derive(Debug, Clone, Copy)]
struct PendingLag {
    k_next: i64,
    done: f64,
    y: f64,
}

/// One completion's cache split, for the session host rule.
#[derive(Debug, Clone, Copy)]
struct HostSample {
    t: f64,
    bin: HostBin,
    host: f64,
    cached: f64,
}

/// (size bin, gap bin; None = the session's first route to the P).
type HostBin = (usize, Option<usize>);

fn host_bin(cached: i64, gap: Option<f64>) -> HostBin {
    let size = HOST_SIZE_EDGES.partition_point(|e| *e <= cached);
    let gap = gap.map(|g| HOST_GAP_EDGES.partition_point(|e| *e <= g));
    (size, gap)
}

/// Timeline state and samples of one P.
#[derive(Debug, Default)]
struct WorkerStream {
    epoch: Option<String>,
    last_seq: Option<u64>,
    /// The smallest pass whose launch-time row (an entry_to_launch row of
    /// the pass, or the launch_interval into it) the current contiguous
    /// run has: every launch-time row of a later pass is in the run.
    run_first_launch_pass: Option<i64>,
    max_pass: Option<i64>,
    /// Pass k launched after an idle gap (true) or busy (false); absent
    /// while unknown.
    after_idle: BTreeMap<i64, bool>,
    /// Duration of launch_interval k (k -> k + 1, busy).
    li_dur: BTreeMap<i64, f64>,
    pass_feat: BTreeMap<i64, Features>,
    /// rid -> (first pass, its entry_to_launch duration).
    first_pass: HashMap<String, (i64, f64)>,
    pending_service: Vec<PendingService>,
    pending_lag: Vec<PendingLag>,
    /// Busy launch intervals (launch k, launch k + 1), launch order.
    busy: VecDeque<(f64, f64)>,
    /// Capture time (router clock) of the router's first snapshot of this P:
    /// samples that completed before it do not open the calibration window.
    first_seen: Option<f64>,
    /// Start of the calibration window: the time key of the first sample
    /// that completed after `first_seen`.
    cal0: Option<f64>,
    service: Vec<(f64, GpuSample)>,
    /// Service samples older than [`RETAIN_S`] per size class (small, large;
    /// time order), at most [`SERVICE_HISTORY_PER_CLASS`] each.
    service_old: [VecDeque<GpuSample>; 2],
    launch: Vec<LaunchSample>,
    cal_service: Vec<(f64, GpuSample)>,
    cal_launch: Vec<LaunchSample>,
    cal_lag: Vec<(f64, LagSample)>,
    cal_boots: Vec<(f64, f64)>,
    cal_phis: Vec<(f64, f64)>,
    cal_idles: Vec<(f64, f64)>,
    boot_seen: HashSet<String>,
    boot_order: VecDeque<String>,
    rx: RxLine,
    dispatch: VecDeque<ArrivalRow>,
    counts: BTreeMap<&'static str, u64>,
}

/// Insert keeping the time order (rows arrive almost sorted).
fn insert_sorted<T>(rows: &mut Vec<T>, t: f64, row: T, key: impl Fn(&T) -> f64) {
    let at = rows.partition_point(|r| key(r) <= t);
    rows.insert(at, row);
}

impl WorkerStream {
    fn count(&mut self, name: &'static str) {
        *self.counts.entry(name).or_default() += 1;
    }

    fn in_cal(&self, t: f64) -> bool {
        self.cal0
            .is_some_and(|c| t >= c && t < c + crate::e3::CAL_S)
    }

    /// A gap in the sequence: classifications that need every row are reset.
    fn on_gap(&mut self) {
        self.pending_service.clear();
        self.run_first_launch_pass = None;
    }

    fn note_launch_pass(&mut self, pass: i64) {
        self.run_first_launch_pass = Some(self.run_first_launch_pass.map_or(pass, |p| p.min(pass)));
        self.max_pass = Some(self.max_pass.map_or(pass, |m| m.max(pass)));
    }

    fn add_service(&mut self, t: f64, sample: GpuSample) {
        if self.in_cal(t) {
            insert_sorted(&mut self.cal_service, t, (t, sample.clone()), |r| r.0);
        }
        insert_sorted(&mut self.service, t, (t, sample), |r| r.0);
        self.count("service_samples");
    }

    fn add_lag(&mut self, t: f64, sample: LagSample) {
        if self.in_cal(t) {
            insert_sorted(&mut self.cal_lag, t, (t, sample), |r| r.0);
            self.count("lag_samples");
        }
    }

    fn set_pass_features(&mut self, k: i64, f: Features) {
        self.pass_feat.insert(k, f);
        // Lag samples that waited for the features of this pass.
        let ready: Vec<PendingLag> = self
            .pending_lag
            .iter()
            .copied()
            .filter(|p| p.k_next == k)
            .collect();
        if !ready.is_empty() {
            self.pending_lag.retain(|p| p.k_next != k);
            for p in ready {
                self.add_lag(p.done, LagSample { f, y: p.y });
            }
        }
    }

    /// Resolve the pending service samples by a row of pass `fi`: a launch
    /// interval out of pass k (`fi == k`) rejects k; any other row of a pass
    /// after k shows that pass k + 1 has launched without one.
    fn resolve_pending(&mut self, kind: SampleKind, fi: i64) {
        if self.pending_service.is_empty() {
            return;
        }
        let mut accepted: Vec<PendingService> = Vec::new();
        let li_dur = &self.li_dur;
        let mut rejected = 0u64;
        self.pending_service.retain(|p| {
            if kind == SampleKind::LaunchInterval && fi == p.k {
                rejected += 1;
                return false;
            }
            if fi > p.k {
                if !li_dur.contains_key(&p.k) && p.host.is_some() {
                    accepted.push(p.clone());
                } else {
                    rejected += 1;
                }
                return false;
            }
            true
        });
        for _ in 0..rejected {
            self.count("service_candidates_rejected");
        }
        for p in accepted {
            let f = features(
                p.new_tokens as f64,
                p.pairs as f64,
                p.host.unwrap_or(0) as f64,
                p.bs as f64,
            );
            self.add_service(p.done, GpuSample { f, y_ms: p.y_ms });
        }
    }

    /// The busy launch interval that holds `receipt`: phi = next launch -
    /// receipt.
    fn phi_of(&self, receipt: f64) -> Option<f64> {
        let i = self.busy.partition_point(|(start, _)| *start <= receipt);
        let (start, end) = *self.busy.get(i.checked_sub(1)?)?;
        (start <= receipt && receipt < end).then_some(end - receipt)
    }

    /// Drop the dispatch rows that no later fit window holds.
    fn prune_dispatch(&mut self, now: f64) {
        while self
            .dispatch
            .front()
            .is_some_and(|r| r.t < now - crate::e3::WINDOW_S - crate::e3::STEP_S)
        {
            self.dispatch.pop_front();
        }
    }

    fn prune(&mut self, now: f64) {
        let horizon = now - RETAIN_S;
        let n = self.service.partition_point(|r| r.0 < horizon);
        // Service samples leave the window for the per-class history (the
        // own_all / pooled_all steps use every earlier sample).
        for (_, sample) in self.service.drain(..n) {
            let class = usize::from(sample.f[1] > crate::e3::GRAPH_TOKENS as f64);
            let old = &mut self.service_old[class];
            old.push_back(sample);
            if old.len() > SERVICE_HISTORY_PER_CLASS {
                old.pop_front();
            }
        }
        let n = self.launch.partition_point(|r| r.t < horizon);
        self.launch.drain(..n);
        while self
            .busy
            .front()
            .is_some_and(|(_, end)| *end < now - BUSY_KEEP_S)
        {
            self.busy.pop_front();
        }
        self.prune_dispatch(now);
        if let Some(max) = self.max_pass {
            let floor = max - PASS_MEMORY;
            self.after_idle = self.after_idle.split_off(&floor);
            self.li_dur = self.li_dur.split_off(&floor);
            self.pass_feat = self.pass_feat.split_off(&floor);
            self.pending_lag.retain(|p| p.k_next >= floor);
            if self.first_pass.len() > PASS_MEMORY as usize {
                self.first_pass.retain(|_, (k, _)| *k >= floor);
            }
        }
        while self.boot_order.len() > PASS_MEMORY as usize {
            if let Some(rid) = self.boot_order.pop_front() {
                self.boot_seen.remove(&rid);
            }
        }
    }
}

struct Ingest<'a> {
    worker: &'a str,
    /// Router time of the capture: [poll sent, poll received] - age_s. A
    /// pushed state: both are its mapped state time.
    base_lo: f64,
    base: f64,
    received_s: f64,
    unix_offset: f64,
    /// `label_source` of the labels (`poll` or `push`; None leaves the key
    /// out, as without the state push).
    source: Option<&'static str>,
}

#[derive(Default)]
struct Online {
    workers: BTreeMap<String, WorkerStream>,
    fits: HashMap<String, Arc<FitBundle>>,
    routes: HashMap<String, RouteRecord>,
    route_order: VecDeque<(f64, String)>,
    last_route: HashMap<(String, String), f64>,
    host_samples: VecDeque<HostSample>,
    host_table: Option<(f64, HashMap<HostBin, f64>)>,
}

/// The router's online E3 state (fits, receipt lines, routes, host rule).
#[derive(Debug)]
pub struct E3Online {
    anchor: Instant,
    unix_at_anchor: f64,
    /// The state push is on: labels name their source (`label_source`).
    label_sources: bool,
    /// Forecast parameters take [`crate::e3::Params::next_pass_lag`].
    next_pass_lag: bool,
    /// `--co-schedule-trade-e3-frozen-fit`: replaces the GPU, launch and
    /// lag parts of every fit.
    frozen: Option<Arc<FrozenFit>>,
    state: Mutex<OnlineDebug>,
}

/// `Online` behind a Debug wrapper (the state is large).
struct OnlineDebug(Online);

impl std::fmt::Debug for OnlineDebug {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("E3OnlineState")
            .field("workers", &self.0.workers.len())
            .field("routes", &self.0.routes.len())
            .finish_non_exhaustive()
    }
}

impl E3Online {
    /// `anchor` is the zero of the router clock (seconds) of every E3 time.
    pub fn new(anchor: Instant) -> Self {
        let unix_at_anchor = unix_s(SystemTime::now())
            - Instant::now()
                .saturating_duration_since(anchor)
                .as_secs_f64();
        Self {
            anchor,
            unix_at_anchor,
            label_sources: false,
            next_pass_lag: false,
            frozen: None,
            state: Mutex::new(OnlineDebug(Online::default())),
        }
    }

    /// `--co-schedule-trade-e3-frozen-fit`: every (P, step) fit takes the
    /// GPU model, the launch model and the warmup constants of `frozen`;
    /// the online samples still give its arrival rates.
    pub fn with_frozen_fit(mut self, frozen: Option<Arc<FrozenFit>>) -> Self {
        self.frozen = frozen;
        self
    }

    /// Forecasts price the next pass of the expected arrivals into the
    /// result of a request whose last pass leaves no known work on a busy P
    /// (`--co-schedule-trade-e3-next-pass-lag`).
    pub fn with_next_pass_lag(mut self, on: bool) -> Self {
        self.next_pass_lag = on;
        self
    }

    /// With the state push: every `e3_label` line names its source (`poll`
    /// or `push`).
    pub fn with_label_sources(mut self) -> Self {
        self.label_sources = true;
        self
    }

    pub fn anchor(&self) -> Instant {
        self.anchor
    }

    /// Router clock seconds of an instant.
    pub fn secs(&self, at: Instant) -> f64 {
        secs_since(self.anchor, at)
    }

    /// Unix seconds of a router clock time.
    pub fn unix_of(&self, secs: f64) -> f64 {
        self.unix_at_anchor + secs
    }

    /// Take the timeline samples and waiting-row receipts of one snapshot
    /// of `worker`, polled at `sent` and received at `received`. Returns the
    /// `e3_label` events of routed requests whose completion arrived.
    pub fn observe(
        &self,
        worker: &str,
        view: &E3View,
        sent: Instant,
        received: Instant,
    ) -> Vec<Value> {
        let Some(age) = view.age_s else {
            return Vec::new();
        };
        let received_s = self.secs(received);
        let ing = Ingest {
            worker,
            base_lo: self.secs(sent) - age,
            base: received_s - age,
            received_s,
            unix_offset: self.unix_at_anchor,
            source: self.label_sources.then_some("poll"),
        };
        self.observe_at(view, &ing)
    }

    /// Take the timeline samples and waiting-row receipts of one pushed
    /// state of `worker` ([`crate::trade_e3_push`]): `t0_s` is its state time
    /// on the router clock (`sampled_monotonic_s` + the offset) and
    /// `received_s` when the router read the message. A label's interval is
    /// the point `t0_s - completed_age_s` (the offset's error, the smallest
    /// transit time in its window, is left out). Returns the `e3_label`
    /// events.
    pub fn observe_push(
        &self,
        worker: &str,
        view: &E3View,
        t0_s: f64,
        received_s: f64,
    ) -> Vec<Value> {
        let ing = Ingest {
            worker,
            base_lo: t0_s,
            base: t0_s,
            received_s,
            unix_offset: self.unix_at_anchor,
            source: Some("push"),
        };
        self.observe_at(view, &ing)
    }

    fn observe_at(&self, view: &E3View, ing: &Ingest<'_>) -> Vec<Value> {
        let (worker, received_s) = (ing.worker, ing.received_s);
        let mut labels = Vec::new();
        let mut guard = self.state.lock();
        let Online {
            workers,
            routes,
            host_samples,
            ..
        } = &mut guard.0;
        let ws = workers.entry(worker.to_string()).or_default();
        ingest(ws, view, ing, routes, host_samples, &mut labels);
        ws.prune(received_s);
        while host_samples
            .front()
            .is_some_and(|s| s.t < received_s - RETAIN_S)
        {
            host_samples.pop_front();
        }
        labels
    }

    /// The (P, step) fit of the decision at `now_s`, computed on the first
    /// call of the step from copies of the windows (the fit itself runs
    /// outside this lock). The flag is true when this call computed it.
    pub fn fit_for(&self, worker: &str, now_s: f64) -> (Arc<FitBundle>, bool) {
        let step = step_of(now_s, crate::e3::STEP_S);
        let data = {
            let guard = self.state.lock();
            let st = &guard.0;
            if let Some(fit) = st.fits.get(worker).filter(|f| f.step == step) {
                return (Arc::clone(fit), false);
            }
            collect_fit_data(st, worker, step)
        };
        let mut fit = compute_fit(step, &data);
        if let Some(frozen) = &self.frozen {
            frozen.apply(&mut fit);
        }
        let fit = Arc::new(fit);
        let mut guard = self.state.lock();
        let slot = guard
            .0
            .fits
            .entry(worker.to_string())
            .or_insert_with(|| Arc::clone(&fit));
        if slot.step < step {
            *slot = Arc::clone(&fit);
        }
        (Arc::clone(slot), Arc::ptr_eq(slot, &fit))
    }

    /// The E3 inputs of one candidate that need no cost-book state: the
    /// (P, step) fit, the receipt line, the park cap, the target tokens and
    /// the host rule. `tokens` are the router's: own hit, indexer GPU hit
    /// and expected hit (own hit + the hinted pull). The second value is a
    /// fit that this call computed (for the `e3_fit` trace event).
    pub fn prepare(
        &self,
        worker: &str,
        snapshot: Option<Arc<crate::cost_balance::RoutingSnapshot>>,
        rid: &str,
        tokens: CandidateTokens,
        session: Option<&str>,
        now_s: f64,
    ) -> (E3Prepared, Option<Arc<FitBundle>>) {
        let view = snapshot.as_ref().and_then(|s| s.e3.as_ref());
        let page = view.map_or(crate::e3::PAGE_TOKENS, |v| v.page);
        let n = tokens.input;
        let q_own = crate::eft_cheap::floor_hit(tokens.local_hit, n, page);
        let prefix = crate::eft_cheap::floor_hit(tokens.expected_hit, n, page).max(q_own);
        let gap = self.session_gap(session, worker, now_s);
        let (host, host_rule) = match tokens.gpu_hit {
            // Indexer split: the cached tokens that are not on the GPU,
            // pulled tokens included, are restored from host.
            Some(gpu) => (
                prefix - gpu.div_euclid(page).saturating_mul(page).clamp(0, prefix),
                "split",
            ),
            None => {
                let (own_host, rule) = self.host_rule(gap, q_own, now_s);
                // Pulled tokens arrive through the target's host pool.
                (own_host + (prefix - q_own), rule)
            }
        };
        let (fit, new) = self.fit_for(worker, now_s);
        let rx = self.rx_params(worker);
        let cap = view.map_or(
            crate::e3::PARK_CAP * crate::e3::DEFAULT_POOL_TOKENS as f64,
            |v| v.cap_tokens(),
        );
        let params = fit.usable().then(|| Params {
            next_pass_lag: self.next_pass_lag,
            ..fit.params(rx, cap)
        });
        let prep = E3Prepared {
            snapshot,
            push: None,
            fit: Some(Arc::clone(&fit)),
            params,
            target: Target {
                rid: rid.to_string(),
                input: n,
                prefix,
                host,
            },
            host_rule,
            session_gap_s: gap,
        };
        (prep, new.then_some(fit))
    }

    /// Seconds since `session` last went to `worker` (None: never, or no
    /// session).
    pub fn session_gap(&self, session: Option<&str>, worker: &str, now_s: f64) -> Option<f64> {
        let session = session?;
        let guard = self.state.lock();
        guard
            .0
            .last_route
            .get(&(session.to_string(), worker.to_string()))
            .map(|t| py_max(0.0, now_s - t))
    }

    /// The receipt line of `worker` (rule `rx2`), None below its minimum.
    pub fn rx_params(&self, worker: &str) -> Option<RxParams> {
        self.state
            .lock()
            .0
            .workers
            .get(worker)
            .and_then(|w| w.rx.params())
    }

    /// The session host rule (`timing.host_share`):
    /// (host tokens, rule). `q_own` is the page-floored own prefix of the
    /// target and `gap` the seconds since its session last went to this P
    /// ([`Self::session_gap`]; None = the first route). The learned share of
    /// host-restored cached tokens of the completions in the bin of (q_own,
    /// gap) over the refit window decides between the whole prefix
    /// (`share_host`) and none (`share_device`); a bin with fewer than
    /// [`HOST_MIN_SAMPLES`] completions gives none (`no_share`).
    pub fn host_rule(&self, gap: Option<f64>, q_own: i64, now_s: f64) -> (i64, &'static str) {
        if q_own <= 0 {
            return (0, "cold");
        }
        let mut guard = self.state.lock();
        let st = &mut guard.0;
        let step = step_of(now_s, crate::e3::STEP_S);
        if st.host_table.as_ref().is_none_or(|(s, _)| *s != step) {
            let mut sums: HashMap<HostBin, (f64, f64, usize)> = HashMap::new();
            for s in st
                .host_samples
                .iter()
                .filter(|s| s.t >= step - crate::e3::WINDOW_S && s.t < step)
            {
                let acc = sums.entry(s.bin).or_insert((0.0, 0.0, 0));
                acc.0 += s.host;
                acc.1 += s.cached;
                acc.2 += 1;
            }
            let table = sums
                .into_iter()
                .filter(|(_, (_, cached, n))| *n >= HOST_MIN_SAMPLES && *cached > 0.0)
                .map(|(bin, (host, cached, _))| (bin, host / cached))
                .collect();
            st.host_table = Some((step, table));
        }
        let share = st
            .host_table
            .as_ref()
            .and_then(|(_, t)| t.get(&host_bin(q_own, gap)).copied());
        match share {
            Some(s) if s >= HOST_RULE_SHARE => (q_own, "share_host"),
            Some(_) => (0, "share_device"),
            None => (0, "no_share"),
        }
    }

    /// Remember a routed request: its completion trains the receipt line,
    /// the host rule and gives the label; its tokens enter the arrival
    /// rates; its session's last route to the P moves.
    pub fn record_route(&self, rec: RouteRecord) {
        let mut guard = self.state.lock();
        let st = &mut guard.0;
        let ws = st.workers.entry(rec.worker.clone()).or_default();
        ws.dispatch.push_back(ArrivalRow {
            t: rec.decided_s,
            uncached: (rec.input - rec.prefix).max(0),
            cached: rec.prefix,
        });
        // Also here: a P whose snapshots are never ingested (no
        // prefill_routing section, failing polls) is never pruned by observe.
        ws.prune_dispatch(rec.decided_s);
        if let Some(session) = &rec.session {
            st.last_route
                .insert((session.clone(), rec.worker.clone()), rec.decided_s);
        }
        let horizon = rec.decided_s - RETAIN_S;
        while st
            .route_order
            .front()
            .is_some_and(|(t, _)| *t < horizon || st.routes.len() > MAX_ROUTES)
        {
            if let Some((_, rid)) = st.route_order.pop_front() {
                st.routes.remove(&rid);
            }
        }
        if st.last_route.len() > MAX_ROUTES {
            st.last_route.retain(|_, t| *t >= horizon);
        }
        st.route_order.push_back((rec.decided_s, rec.rid.clone()));
        st.routes.insert(rec.rid.clone(), rec);
    }

    /// Counters per P (for the startup and debug logs).
    pub fn counts(&self) -> Value {
        let guard = self.state.lock();
        json!(guard
            .0
            .workers
            .iter()
            .map(|(url, w)| (url.clone(), json!(w.counts)))
            .collect::<serde_json::Map<String, Value>>())
    }
}

/// Seconds from `anchor` to `at` (negative when `at` is earlier).
pub fn secs_since(anchor: Instant, at: Instant) -> f64 {
    match at.checked_duration_since(anchor) {
        Some(d) => d.as_secs_f64(),
        None => -anchor.saturating_duration_since(at).as_secs_f64(),
    }
}

fn collect_fit_data(st: &Online, worker: &str, step: f64) -> FitData {
    let lo = step - crate::e3::WINDOW_S;
    let mut data = FitData::default();
    let own = st.workers.get(worker);
    let window_of = |rows: &[(f64, GpuSample)], lo: f64, hi: f64| -> Vec<GpuSample> {
        crate::e3::window(rows, lo, hi, |r| r.0)
            .iter()
            .map(|r| r.1.clone())
            .collect()
    };
    for (url, w) in &st.workers {
        let cal_hi = w.cal0.map_or(step, |c| (c + crate::e3::CAL_S).min(step));
        let cal_lo = w.cal0.unwrap_or(f64::INFINITY);
        let window = window_of(w.service.as_slice(), lo, step);
        let cal = window_of(w.cal_service.as_slice(), cal_lo, cal_hi);
        // Every earlier service sample: the history before the
        // retained window (per class, time order), then the retained ones.
        // Each class's rows stay in time order.
        let all: Vec<GpuSample> = w
            .service_old
            .iter()
            .flatten()
            .cloned()
            .chain(window_of(w.service.as_slice(), f64::NEG_INFINITY, step))
            .collect();
        if url == worker {
            data.own_window = window.clone();
            data.own_cal = cal.clone();
            data.own_all = all.clone();
        }
        data.pooled_window.extend(window);
        data.pooled_cal.extend(cal);
        data.pooled_all.extend(all);
    }
    if let Some(w) = own {
        let cal_hi = w.cal0.map_or(step, |c| (c + crate::e3::CAL_S).min(step));
        let cal_lo = w.cal0.unwrap_or(f64::INFINITY);
        data.launch_window = crate::e3::window(&w.launch, lo, step, |r| r.t).to_vec();
        data.launch_cal = crate::e3::window(&w.cal_launch, cal_lo, cal_hi, |r| r.t).to_vec();
        data.launch_ys = w
            .launch
            .iter()
            .filter(|r| r.t < step)
            .map(|r| r.y)
            .collect();
        data.lag_cal = crate::e3::window(&w.cal_lag, cal_lo, cal_hi, |r| r.0)
            .iter()
            .map(|r| r.1.clone())
            .collect();
        let cal_values = |rows: &[(f64, f64)]| -> Vec<f64> {
            rows.iter()
                .filter(|(t, _)| *t >= cal_lo && *t < cal_hi)
                .map(|(_, v)| *v)
                .collect()
        };
        data.boots = cal_values(w.cal_boots.as_slice());
        data.phis = cal_values(w.cal_phis.as_slice());
        data.idles = cal_values(w.cal_idles.as_slice());
        data.dispatch = w
            .dispatch
            .iter()
            .filter(|r| r.t >= lo && r.t < step)
            .copied()
            .collect();
    }
    data
}

/// Ingest the new timeline rows (sequence above the last one) of one P.
fn ingest(
    ws: &mut WorkerStream,
    view: &E3View,
    ing: &Ingest<'_>,
    routes: &mut HashMap<String, RouteRecord>,
    host_samples: &mut VecDeque<HostSample>,
    labels: &mut Vec<Value>,
) {
    if view.epoch != ws.epoch {
        if ws.epoch.is_some() {
            // An engine restart: pass numbers restart; samples stay.
            ws.count("epoch_changes");
            ws.last_seq = None;
            ws.after_idle.clear();
            ws.li_dur.clear();
            ws.pass_feat.clear();
            ws.first_pass.clear();
            ws.pending_lag.clear();
            ws.max_pass = None;
            ws.on_gap();
        }
        ws.epoch = view.epoch.clone();
    }
    let new_rows: Vec<&TimelineRow> = view
        .samples
        .iter()
        .filter(|r| ws.last_seq.is_none_or(|last| r.sequence > last))
        .collect();
    // The calibration window opens at the first sample that completed after
    // the router's first snapshot of this P. The rows of that snapshot
    // completed before the router watched the P (the engine's startup
    // warmup, a smoke run, traffic before a router restart; the engine keeps
    // its last 64 rows however old): they feed the windowed fits only.
    let seen = *ws.first_seen.get_or_insert(ing.base);
    if ws.cal0.is_none() {
        let first = new_rows
            .iter()
            .filter_map(|r| Some(ing.base - r.completed_age_s?))
            .filter(|t| *t > seen)
            .fold(f64::INFINITY, f64::min);
        if first.is_finite() {
            ws.cal0 = Some(first);
        } else if !view.boots.is_empty() {
            ws.cal0 = Some(ing.received_s);
        }
    }
    for (rid, value) in &view.boots {
        if ws.boot_seen.insert(rid.clone()) {
            ws.boot_order.push_back(rid.clone());
            if ws.in_cal(ing.received_s) {
                ws.cal_boots.push((ing.received_s, *value));
            }
        }
    }
    for row in new_rows {
        let contiguous = ws.last_seq.is_some_and(|last| row.sequence == last + 1);
        if !contiguous {
            if ws.last_seq.is_some() {
                ws.count("timeline_gaps");
            }
            ws.on_gap();
        }
        ws.last_seq = Some(row.sequence);
        let (Some(kind), Some(dur), Some(age)) = (row.kind, row.duration_s, row.completed_age_s)
        else {
            continue;
        };
        let end = ing.base - age;
        let start = end - dur;
        let Some(k) = row.forward_iter else {
            continue;
        };
        match kind {
            SampleKind::LaunchInterval => {
                ws.li_dur.insert(k, dur);
                ws.after_idle.insert(k + 1, false);
                if let Some(f) = row.pass_features() {
                    ws.set_pass_features(k, f);
                }
                ws.note_launch_pass(k + 1);
                ws.busy.push_back((start, end));
                if (0.0..crate::e3::LAUNCH_Y_MAX_S).contains(&dur) {
                    if let (Some(f), Some(fin), Some(next_host)) =
                        (row.pass_features(), row.finishing, row.next_host_tokens)
                    {
                        let sample = LaunchSample {
                            t: end,
                            f,
                            fin: fin as f64,
                            host_next: next_host as f64,
                            y: dur,
                        };
                        if ws.in_cal(end) {
                            insert_sorted(&mut ws.cal_launch, end, sample.clone(), |r| r.t);
                        }
                        insert_sorted(&mut ws.launch, end, sample, |r| r.t);
                        ws.count("launch_samples");
                    }
                }
            }
            SampleKind::EntryToLaunch => {
                if let Some(f) = row.pass_features() {
                    ws.set_pass_features(k, f);
                }
                // Every launch-time row of pass k is in this run when the run
                // has a launch-time row of an earlier pass; the launch
                // interval into k comes before the entry rows of k.
                if !ws.after_idle.contains_key(&k)
                    && ws.run_first_launch_pass.is_some_and(|p| p < k)
                {
                    let busy = ws.li_dur.contains_key(&(k - 1));
                    ws.after_idle.insert(k, !busy);
                }
                ws.note_launch_pass(k);
                if let Some(rid) = &row.rid {
                    ws.first_pass.insert(rid.clone(), (k, dur));
                }
            }
            SampleKind::LastLaunchToDone => {
                if let Some(f) = row.pass_features() {
                    ws.set_pass_features(k, f);
                }
                let done = end;
                // Service sample candidate: one pass, alone,
                // launched after an idle gap; pass k + 1 decides it.
                let single = row
                    .rid
                    .as_ref()
                    .and_then(|rid| ws.first_pass.get(rid))
                    .is_some_and(|(first, _)| *first == k);
                if single
                    && row.batch_size == Some(1)
                    && ws.after_idle.get(&k) == Some(&true)
                    && !ws.li_dur.contains_key(&k)
                {
                    if let (Some(rid), Some(new), Some(pairs)) =
                        (row.rid.clone(), row.new_tokens, row.attention_pairs)
                    {
                        ws.pending_service.push(PendingService {
                            k,
                            rid,
                            done,
                            new_tokens: new,
                            pairs,
                            bs: 1,
                            y_ms: dur * 1000.0,
                            host: None,
                        });
                    }
                }
                // Next-aware lag: pass k + 1 launched before done.
                if let Some(&li) = ws.li_dur.get(&k) {
                    if dur > li {
                        let y = dur - li;
                        match ws.pass_feat.get(&(k + 1)).copied() {
                            Some(f) => ws.add_lag(done, LagSample { f, y }),
                            None => ws.pending_lag.push(PendingLag {
                                k_next: k + 1,
                                done,
                                y,
                            }),
                        }
                    }
                }
            }
            SampleKind::RequestComplete => {
                let receipt = start;
                let done = end;
                let Some(rid) = row.rid.as_deref() else {
                    continue;
                };
                // The host feature of a service sample: the request's
                // ungated cache provenance.
                let host = if row.breakdown { row.cached_host } else { None };
                match host {
                    Some(h) => ws
                        .pending_service
                        .iter_mut()
                        .filter(|p| p.rid == rid)
                        .for_each(|p| p.host = Some(h)),
                    None => ws.pending_service.retain(|p| p.rid != rid),
                }
                if let Some(phi) = ws.phi_of(receipt) {
                    if ws.in_cal(done) {
                        ws.cal_phis.push((done, phi));
                    }
                }
                if let (Some(&(first, etl)), Some(wait)) =
                    (ws.first_pass.get(rid), row.receive_to_entry_s)
                {
                    if ws.after_idle.get(&first) == Some(&true) && ws.in_cal(done) {
                        ws.cal_idles.push((done, wait + etl));
                    }
                }
                if routes.get(rid).is_some_and(|r| r.worker == ing.worker) {
                    let rec = routes.remove(rid).expect("checked");
                    if rec.idle == Some(true) {
                        let y = receipt - rec.decided_s;
                        if y >= 0.0 {
                            ws.rx.add(rec.input as f64, y);
                            ws.count("rx_samples");
                        }
                    }
                    if row.breakdown {
                        if let (Some(device), Some(h)) = (row.cached_device, row.cached_host) {
                            let cached = device + h + row.cached_storage.unwrap_or(0);
                            if cached > 0 {
                                host_samples.push_back(HostSample {
                                    t: done,
                                    bin: host_bin(cached, rec.session_gap_s),
                                    host: h as f64,
                                    cached: cached as f64,
                                });
                            }
                        }
                    }
                    let lo = ing.base_lo - age - rec.decided_s;
                    let hi = ing.base - age - rec.decided_s;
                    let label = (lo + hi) / 2.0;
                    let mut line = json!({
                        "event": "e3_label",
                        "e3_schema": SHADOW_SCHEMA,
                        "rid": rec.rid,
                        "worker_url": rec.worker,
                        "decision_mono_s": rec.decided_s,
                        "decision_unix_s": rec.decided_unix_s,
                        "done_mono_s": done,
                        "done_unix_s": ing.unix_offset + done,
                        "label_s": label,
                        "label_lo_s": lo,
                        "label_hi_s": hi,
                        "pred_s": rec.pred_s,
                        "ape": rec.pred_s.filter(|_| label > 0.0).map(|p| (p - label).abs() / label),
                        "idle": rec.idle,
                    });
                    if let Some(source) = ing.source {
                        line["label_source"] = json!(source);
                    }
                    labels.push(line);
                    ws.count("labels");
                }
            }
        }
        ws.resolve_pending(kind, k);
    }
}

// ---------------------------------------------------------------------------
// Decision context (built before the cost-book lock) and shadow log.
// ---------------------------------------------------------------------------

/// The router's token view of one candidate P for the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateTokens {
    pub input: i64,
    /// The P's own cached prefix (indexer).
    pub local_hit: i64,
    /// The GPU-resident part of it, when the indexer reports one.
    pub gpu_hit: Option<i64>,
    /// Own hit plus the tokens of the hinted pull (= own hit without one).
    pub expected_hit: i64,
}

/// The E3 inputs of one candidate that do not need the cost book.
#[derive(Debug, Clone)]
pub struct E3Prepared {
    pub snapshot: Option<Arc<crate::cost_balance::RoutingSnapshot>>,
    /// The P's newest pushed state (state push on; None: no push yet or the
    /// push is off). The decision uses it instead of `snapshot` when it is
    /// fresh, verified and usable.
    pub push: Option<Arc<crate::trade_e3_push::PushState>>,
    pub fit: Option<Arc<FitBundle>>,
    /// The forecast parameters; None when the fit has no GPU model.
    pub params: Option<Params>,
    pub target: Target,
    /// `split` (indexer GPU hit), `share_host`, `share_device`, `no_share`
    /// or `cold`.
    pub host_rule: &'static str,
    pub session_gap_s: Option<f64>,
}

/// The inputs of one `wait = e3` decision, built before the cost-book lock.
#[derive(Debug, Clone)]
pub struct TradeE3Context {
    pub cfg: Arc<TradeE3Config>,
    /// Zero of the router clock (s) of every E3 time.
    pub anchor: Instant,
    /// Unix seconds at `anchor` (the decision's wall-clock time in the
    /// trace, for the join with engine times).
    pub unix_at_anchor: f64,
    pub candidates: HashMap<String, E3Prepared>,
}

/// The session key of a request for the host rule: the
/// `x-kv-session-context` agent (`agent_id/subagent_id`) when the header
/// is valid, else the `x-smg-routing-key` header (the trace client's agent
/// routing key), else None (no session gap: the rule then uses the bin of a
/// first route).
pub fn session_key(
    headers: &axum::http::HeaderMap,
    identity: Option<&crate::agent_identity::RequestIdentity>,
) -> Option<String> {
    let parsed;
    let identity = match identity {
        Some(id) => Some(id),
        None => {
            parsed = crate::agent_identity::from_header(
                headers,
                crate::agent_identity::KV_SESSION_HEADER,
            )
            .ok()
            .flatten();
            parsed.as_ref()
        }
    };
    if let Some(id) = identity {
        return Some(format!("{}/{}", id.agent_id, id.subagent_id));
    }
    headers
        .get("x-smg-routing-key")
        .and_then(|v| v.to_str().ok())
        .filter(|k| !k.is_empty())
        .map(|k| format!("routing-key/{k}"))
}

/// Where the `e3_shadow` and `e3_label` lines go.
pub enum ShadowSink {
    /// The cost trace (its sequence numbers and timestamps).
    Trace(Arc<crate::cost_balance::CostBook>),
    /// `--co-schedule-trade-e3-shadow PATH` (a new file).
    File(Mutex<Option<BufWriter<File>>>),
}

impl std::fmt::Debug for ShadowSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Trace(_) => "ShadowSink::Trace",
            Self::File(_) => "ShadowSink::File",
        })
    }
}

impl ShadowSink {
    /// A new shadow file; an existing path is never appended or truncated.
    pub fn file(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        Ok(Self::File(Mutex::new(Some(BufWriter::new(file)))))
    }

    /// Write one line. A file error stops the file log (logged once); a
    /// cost-trace error is the cost book's (it stops admission).
    pub fn write(&self, mut event: Value) {
        match self {
            Self::Trace(book) => {
                if let Err(error) = book.emit_event(event) {
                    tracing::error!(target: "co_sched", %error, "e3 shadow line not written to the cost trace");
                }
            }
            Self::File(slot) => {
                let mut slot = slot.lock();
                let Some(writer) = slot.as_mut() else {
                    return;
                };
                event["ts_unix_s"] = json!(unix_s(SystemTime::now()));
                let result = serde_json::to_writer(&mut *writer, &event)
                    .map_err(std::io::Error::other)
                    .and_then(|()| writer.write_all(b"\n"))
                    .and_then(|()| writer.flush());
                if let Err(error) = result {
                    tracing::error!(target: "co_sched", %error, "e3 shadow log failed; no further lines");
                    *slot = None;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::e3::{forecast, ForecastPath};

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/e3")
            .join(name)
    }

    /// The real reply cut from the recorded observer (PROVENANCE.md).
    fn recorded() -> Value {
        let text = std::fs::read_to_string(fixture("routing-snapshot-recorded.json")).unwrap();
        serde_json::from_str(&text).unwrap()
    }

    fn gm() -> GpuModel {
        GpuModel {
            small: Some(vec![20.0, 0.02, 0.5, 0.0, 1.0]),
            large: Some(vec![30.0, 0.03, 0.4, 0.0, 1.0]),
            floor_ms: 1.0,
        }
    }

    fn params() -> Params {
        make_params(
            &Constants {
                idle_delay_s: 0.03,
                phi_s: 0.01,
                boot_s: 0.002,
                lag_none_s: 0.0,
                lag_next: [0.01, 0.5],
            },
            gm(),
            gm(),
            vec![0.02, 0.9, 0.0, 0.0, 0.0, 0.0],
            None,
            LaunchAnchor::Launch,
            None,
            None,
            0.35 * 2_067_584.0,
        )
    }

    /// The recorded reply in E3 form: one pending pass, the
    /// chunked request, 33 waiting rows (the last one has no published
    /// prefix: it is past the lookahead plan), U = 0 and the pool.
    #[test]
    fn the_recorded_snapshot_parses_into_e3_rows() {
        let rec = recorded();
        let raw = &rec["candidate"]["snapshot"]["raw_prefill_routing"];
        let v = E3View::parse(raw);
        assert_eq!(v.unusable, None, "{:?}", v.degraded);
        assert!(v.degraded.is_empty(), "{:?}", v.degraded);
        assert_eq!(v.epoch.as_deref(), Some("8e884f52591543248367e6693fba8b72"));
        assert_eq!((v.page, v.chunk, v.max_input), (64, 8192, Some(8192)));
        assert_eq!((v.chunk, v.page), BOUND_FIT_POLICY);
        assert_eq!(v.inflight.len(), 1);
        let p = &v.inflight[0];
        assert_eq!(p.launch_age_s, 0.3081744050141424);
        assert_eq!(
            (p.new_tokens, p.attention_pairs, p.batch_size),
            (8192, 448_270_336, 1)
        );
        assert_eq!((p.host_tokens, p.finishing), (0, 0));
        let partial = v.partial.clone().unwrap();
        assert_eq!(partial.prefix, 50_624 + 8_192);
        assert_eq!(partial.remaining, 79_424);
        assert_eq!(v.waiting.len(), 33);
        assert_eq!(v.waiting.iter().filter(|r| r.prefix.is_none()).count(), 1);
        let first = &v.waiting[0];
        assert_eq!(
            (first.input, first.prefix, first.host),
            (32_448, Some(32_448 - 832), Some(7_936))
        );
        assert!(v.waiting[32].prefix.is_none() && v.waiting[32].host.is_none());
        assert!(v.waiting.iter().all(|r| r.pull == 0), "no pending pull");
        assert!(v.bootstrap.is_empty());
        assert!(v.shown.contains(&partial.rid));
        assert_eq!(
            v.shown.len(),
            34,
            "partial (also the pass member) + 33 waiting"
        );
        assert_eq!(v.unbound_tokens, Some(0));
        assert_eq!(v.pool_tokens, Some(2_067_584));
        assert_eq!(v.cap_tokens(), 0.35 * 2_067_584.0);
        assert_eq!(v.boots.len(), 33);
        assert_eq!(v.samples.len(), 64);
        assert!(v
            .samples
            .windows(2)
            .all(|w| w[1].sequence == w[0].sequence + 1));
        let kinds: BTreeSet<_> = v
            .samples
            .iter()
            .filter_map(|s| s.kind)
            .map(|k| k as u8)
            .collect();
        assert_eq!(kinds.len(), 4);
    }

    /// The recorded timeline (64 samples; the P is busy, every pass has a
    /// busy launch before it): 13 launch samples, 17 next-aware lag samples
    /// (all in the calibration window), no service sample, 33 boot samples
    /// from the waiting rows. Observing the same snapshot again adds nothing.
    /// The router saw the P 30 s earlier (a snapshot without samples), so
    /// the recorded samples open the calibration window.
    #[test]
    fn the_recorded_timeline_feeds_the_fits() {
        let rec = recorded();
        let env = &rec["candidate"]["snapshot"];
        let v = E3View::parse(&env["raw_prefill_routing"]);
        let t0 = Instant::now();
        let online = E3Online::new(t0);
        let at = |s: f64| t0 + std::time::Duration::from_secs_f64(s);
        let sent = env["poll_sent_mono_s"].as_f64().unwrap();
        let received = env["poll_received_mono_s"].as_f64().unwrap();
        let mut first = v.clone();
        first.samples.clear();
        first.boots.clear();
        assert!(online
            .observe("p", &first, at(sent - 30.0), at(received - 30.0))
            .is_empty());
        for _ in 0..2 {
            assert!(online.observe("p", &v, at(sent), at(received)).is_empty());
        }
        let guard = online.state.lock();
        let w = &guard.0.workers["p"];
        assert_eq!(w.launch.len(), 13);
        assert_eq!(w.cal_launch.len(), 13);
        // Launch intervals 0.29 s to 1.17 s, time-ordered by their launch.
        assert!(
            w.launch.iter().all(|r| r.y > 0.25 && r.y < 1.2),
            "{:?}",
            w.launch
        );
        assert!(w.launch.windows(2).all(|p| p[0].t <= p[1].t));
        assert_eq!(w.cal_lag.len(), 17);
        assert!(w.pending_lag.is_empty());
        assert!(w.service.is_empty() && w.pending_service.is_empty());
        assert_eq!(w.cal_boots.len(), 33);
        assert_eq!(w.last_seq, Some(6909));
        assert_eq!(w.counts.get("timeline_gaps"), None);
        drop(guard);
        // The fit of the step of the snapshot does not see its samples (the
        // window ends at the step start); the next step's fit does. No
        // service sample on any P yet: no GPU model, no forecast.
        let (fit, _) = online.fit_for("p", received);
        assert_eq!(
            (fit.step, fit.counts.launch_window, fit.counts.boots),
            (1080.0, 0, 0)
        );
        let (fit, _) = online.fit_for("p", received + 60.0);
        assert!(!fit.usable());
        assert_eq!((fit.counts.launch_window, fit.counts.launch_cal), (13, 13));
        assert_eq!((fit.counts.lag_cal, fit.counts.boots), (17, 33));
    }

    /// The input of the recorded decision: t0 = poll receipt - age_s, the
    /// pending pass at t0 - launch_age_s, the waiting rows in snapshot order,
    /// then the two rows the router had sent that the snapshot does not show
    /// (the observer's `dispatched_not_in_snapshot`), by ready time. The
    /// null-prefix row takes the router's estimate when it has one.
    #[test]
    fn the_recorded_snapshot_builds_the_forecast_input() {
        let rec = recorded();
        let env = &rec["candidate"]["snapshot"];
        let raw = &env["raw_prefill_routing"];
        let v = E3View::parse(raw);
        let received = env["poll_received_mono_s"].as_f64().unwrap();
        let d = rec["decision"]["decision_mono_s"].as_f64().unwrap();
        let t0 = received - v.age_s.unwrap();
        let dispatched: Vec<RouterRow> = rec["candidate"]["dispatched_not_in_snapshot"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| RouterRow {
                rid: r["rid"].as_str().unwrap().to_string(),
                input: r["input_tokens"].as_i64().unwrap(),
                prefix: 0,
                host: 0,
                decided_s: r["dispatch_mono_s"].as_f64().unwrap(),
            })
            .collect();
        assert_eq!(dispatched.len(), 2);
        let target = Target {
            rid: rec["decision"]["rid"].as_str().unwrap().to_string(),
            input: 88_704,
            prefix: 81_920,
            host: 0,
        };
        let null_rid = v.waiting[32].rid.clone();
        let estimate = |rid: &str| (rid == null_rid).then_some((40_000, 0));
        let p = params();
        let (inp, notes) = build_input(
            &v,
            t0,
            d,
            target.clone(),
            &dispatched,
            &estimate,
            &p,
            Some(0.0),
        );
        assert_eq!(inp.t0, t0);
        assert_eq!(inp.inflight.len(), 1);
        assert_eq!(inp.inflight[0].launch, t0 - 0.3081744050141424);
        assert_eq!(inp.queue.len(), 35);
        for (row, snap) in inp.queue.iter().zip(&v.waiting) {
            assert_eq!(row.rid, snap.rid);
            assert_eq!(row.source, RowSource::Snapshot);
        }
        assert_eq!(inp.queue[32].prefix, Some(40_000));
        // Ready = decision + 0.044 (no receipt line) + boot; the earlier
        // decision first.
        assert_eq!(inp.queue[33].decision_time, Some(1098.656756167));
        assert_eq!(inp.queue[34].decision_time, Some(1098.765526879));
        assert_eq!(notes.router_rows, 2);
        assert_eq!(
            (notes.router_rows_cut, notes.null_estimated, notes.null_cold),
            (0, 1, 0)
        );
        assert_eq!(inp.unbound_tokens, Some(0));
        let out = forecast(&inp, &p).unwrap();
        assert!(!out.idle);
        assert_eq!(out.n_inflight, 1);
        assert!(out.partial);
        assert_eq!(out.n_queue, 35);
        assert_eq!(out.n_dispatched, 2);
        assert_eq!(out.n_prefix_unknown, 0);
        let pred = out.pred_s.unwrap();
        let wait = out.wait_s.unwrap();
        assert!(pred > wait && wait > 0.0, "{pred} {wait}");
        // Without an estimate the row is cold; a row decided after the
        // target is ready is cut (margin 0).
        let none = |_: &str| None;
        let late = vec![RouterRow {
            rid: "late".into(),
            input: 1_000,
            prefix: 0,
            host: 0,
            decided_s: d + 1.0,
        }];
        let (inp, notes) = build_input(&v, t0, d, target.clone(), &late, &none, &p, Some(0.0));
        assert_eq!((notes.router_rows_cut, notes.null_cold), (1, 1));
        assert_eq!(inp.queue.len(), 33);
        assert_eq!(forecast(&inp, &p).unwrap().n_prefix_unknown, 1);
        // rqall (None) keeps it, after the waiting rows.
        let (inp, notes) = build_input(&v, t0, d, target, &late, &none, &p, None);
        assert_eq!((notes.router_rows_cut, inp.queue.len()), (0, 34));
        assert_eq!(inp.queue[33].rid, "late");
    }

    /// An idle P: no pass, partial or row; the forecast takes the idle path.
    #[test]
    fn an_empty_snapshot_is_an_idle_p() {
        let raw = json!({"epoch": "e", "sequence": 1, "available": true, "stale": false,
            "truncated": false, "age_s": 0.05, "inflight_batches_known": true,
            "inflight_batches": [], "waiting": [], "bootstrap": [], "inflight_requests": [],
            "policy": {"page_size": 64, "chunk_tokens": 8192, "schedule_policy": "fcfs",
                       "share_enabled": false},
            "admission_resources": {"token_pool_capacity": 1000000,
                                    "unbound_prompt_tokens_last_scan": null}});
        let v = E3View::parse(&raw);
        assert_eq!(v.unusable, None);
        assert_eq!(v.unbound_tokens, None, "no scan value and no row");
        assert!((v.cap_tokens() - 350_000.0).abs() < 1e-6);
        let target = Target {
            rid: "x".into(),
            input: 1_000,
            prefix: 0,
            host: 0,
        };
        let none = |_: &str| None;
        let (inp, _) = build_input(&v, 9.95, 10.0, target, &[], &none, &params(), Some(0.0));
        let out = forecast(&inp, &params()).unwrap();
        assert!(out.idle);
        assert_eq!(out.path, ForecastPath::Idle);
        assert!(out.pred_s.unwrap() > 0.044);
    }

    /// A waiting row with a pending pull (`plan.pull_tokens`): the engine's
    /// remaining compute still holds the pulled tokens; the input moves them
    /// to the row's cached prefix and its host part (at most the input).
    #[test]
    fn a_pending_pull_of_a_row_counts_as_cached_host_tokens() {
        let row = |rid: &str, input: i64, remaining: i64, host: i64, pull: i64| {
            json!({"rid": rid, "phase": "waiting", "input_tokens": input,
                   "remaining_compute_tokens": remaining, "host_prefix_tokens": host,
                   "recv_age_s": 0.5, "queue_age_s": 0.4,
                   "plan": {"pull_tokens": pull, "uncached_tokens": 1, "skip_reason": "pull"}})
        };
        let raw = json!({"epoch": "e", "sequence": 1, "available": true, "stale": false,
            "truncated": false, "age_s": 0.0, "inflight_batches_known": true,
            "inflight_batches": [], "bootstrap": [], "inflight_requests": [],
            "policy": {"page_size": 64, "chunk_tokens": 8192, "schedule_policy": "fcfs"},
            "waiting": [row("a", 50_000, 42_000, 8_000, 30_000),
                        row("b", 10_000, 9_000, 0, 20_000),
                        row("c", 4_096, 4_096, 0, 0)]});
        let v = E3View::parse(&raw);
        assert_eq!(v.unusable, None);
        let pulls: Vec<i64> = v.waiting.iter().map(|r| r.pull).collect();
        assert_eq!(pulls, vec![30_000, 20_000, 0]);
        let target = Target {
            rid: "x".into(),
            input: 1_000,
            prefix: 0,
            host: 0,
        };
        let none = |_: &str| None;
        let (inp, notes) = build_input(&v, 9.95, 10.0, target, &[], &none, &params(), Some(0.0));
        let tokens: Vec<_> = inp.queue.iter().map(|r| (r.prefix, r.host)).collect();
        // a: 8,000 cached (host) + 30,000 pulled; b: 1,000 cached on the GPU,
        // the pull capped at the 9,000 tokens left; c: no pull.
        assert_eq!(
            tokens,
            vec![
                (Some(38_000), Some(38_000)),
                (Some(10_000), Some(9_000)),
                (Some(0), Some(0))
            ]
        );
        assert_eq!((notes.pull_rows, notes.pull_tokens), (2, 39_000));
    }

    /// Unusable views name their reason. The engine's stale flag counts only
    /// without a known age (the router's view-age limit decides otherwise);
    /// the engine policy must give the pass budget and page, and a
    /// max_prefill_tokens below the budget is not E3's pass formation.
    #[test]
    fn unusable_snapshots_name_the_reason() {
        let base = json!({"epoch": "e", "available": true, "stale": false, "truncated": false,
            "age_s": 0.0, "inflight_batches_known": true,
            "policy": {"page_size": 64, "chunk_tokens": 8192, "max_input_tokens": 8192,
                       "schedule_policy": "fcfs", "share_enabled": false}});
        let with = |k: &str, v: Value| {
            let mut raw = base.clone();
            raw[k] = v;
            E3View::parse(&raw).unusable
        };
        assert_eq!(E3View::parse(&base).unusable, None);
        assert!(E3View::parse(&base).degraded.is_empty());
        assert_eq!(with("available", json!(false)), Some("unavailable"));
        let mut stale = base.clone();
        stale["stale"] = json!(true);
        let v = E3View::parse(&stale);
        assert_eq!(v.unusable, None, "a known age: the router's limit decides");
        assert!(v.degraded.contains("engine_stale"));
        stale["age_s"] = Value::Null;
        assert_eq!(E3View::parse(&stale).unusable, Some("stale"));
        assert_eq!(with("truncated", json!(true)), Some("truncated"));
        assert_eq!(
            with("inflight_batches_known", json!(false)),
            Some("inflight_unknown")
        );
        assert_eq!(
            with("policy", json!({"schedule_policy": "lpm"})),
            Some("policy_not_fcfs")
        );
        assert_eq!(
            with(
                "policy",
                json!({"schedule_policy": "fcfs", "share_enabled": true})
            ),
            Some("policy_share_enabled")
        );
        assert_eq!(
            with(
                "policy",
                json!({"schedule_policy": "fcfs", "share_enabled": false, "page_size": 64})
            ),
            Some("policy_unknown")
        );
        assert_eq!(
            with(
                "policy",
                json!({"schedule_policy": "fcfs", "chunk_tokens": 8192, "page_size": 0})
            ),
            Some("policy_unknown")
        );
        assert_eq!(
            with(
                "policy",
                json!({"schedule_policy": "fcfs", "page_size": 64, "chunk_tokens": 8192,
                       "max_input_tokens": 4096})
            ),
            Some("policy_max_input_below_chunk")
        );
        // Without max_input_tokens, or with a larger one, the chunk is the
        // budget; another chunk or page parses (the decision checks it
        // against the bound tables' policy).
        let v = E3View::parse(&json!({"epoch": "e", "available": true, "stale": false,
            "truncated": false, "age_s": 0.0, "inflight_batches_known": true,
            "policy": {"page_size": 32, "chunk_tokens": 16384, "max_input_tokens": 16384,
                       "schedule_policy": "fcfs"}}));
        assert_eq!(v.unusable, None);
        assert_eq!((v.chunk, v.page, v.max_input), (16384, 32, Some(16384)));
        assert_ne!((v.chunk, v.page), BOUND_FIT_POLICY);
        assert_eq!(
            with(
                "inflight_batches",
                json!([{"status": "pending", "launch_age_s": null,
                "features": {"new_tokens": 1, "attention_pairs": 1, "batch_size": 1}}])
            ),
            Some("inflight_features_unknown")
        );
        assert_eq!(
            with(
                "partial",
                json!({"rid": "p", "remaining_compute_tokens": null})
            ),
            Some("partial_unknown")
        );
        assert_eq!(with("waiting", json!([{"rid": "w"}])), Some("row_unknown"));
        assert_eq!(E3View::parse(&json!([])).unusable, Some("not_an_object"));
    }

    fn row(seq: u64, kind: &str, k: i64, rid: Option<&str>, dur: f64, age: f64, f: Value) -> Value {
        json!({"sequence": seq, "kind": kind, "forward_iter": k, "rid": rid,
               "duration_s": dur, "completed_age_s": age, "features": f})
    }

    fn pass(n: i64, pairs: i64, bs: i64) -> Value {
        json!({"new_tokens": n, "attention_pairs": pairs, "batch_size": bs,
               "host_restore_tokens": 0, "storage_restore_tokens": 0, "finishing_requests": 1,
               "next_host_restore_tokens": 0})
    }

    fn complete(input: i64, host: i64) -> Value {
        json!({"input_tokens": input, "receive_to_entry_s": 0.002, "cached_tokens_device": 0,
               "cached_tokens_host": host, "cached_tokens_storage": 0,
               "cache_breakdown_computed": true})
    }

    fn view_of(rows: Vec<Value>) -> E3View {
        E3View::parse(&json!({"epoch": "e1", "available": true, "stale": false,
            "truncated": false, "age_s": 0.0, "inflight_batches_known": true,
            "policy": {"schedule_policy": "fcfs", "share_enabled": false},
            "timeline": {"samples": rows}}))
    }

    /// Timeline classification. Pass 10 is busy (launch interval 9 -> 10);
    /// pass 12 follows an idle gap: its only request (r12, one pass, alone)
    /// is a service sample once pass 13 shows no busy launch out of 12. Pass
    /// 20's request is alone too, but pass 21 launches busy: no sample.
    /// The launch intervals are launch samples; the request of pass 10 that
    /// finished after pass 11 launched gives a next-aware lag sample. The
    /// router saw the P at 50 s (a snapshot without samples), so these
    /// samples open the calibration window.
    #[test]
    fn timeline_rows_give_service_launch_and_lag_samples() {
        let t0 = Instant::now();
        let online = E3Online::new(t0);
        let at = |s: f64| t0 + std::time::Duration::from_secs_f64(s);
        assert!(online
            .observe("http://p", &view_of(Vec::new()), at(50.0), at(50.0))
            .is_empty());
        let rows = vec![
            row(
                1,
                "entry_to_launch",
                9,
                Some("r9"),
                0.01,
                9.0,
                pass(4096, 4096, 1),
            ),
            row(
                2,
                "launch_interval",
                9,
                None,
                0.10,
                8.9,
                pass(4096, 4096, 1),
            ),
            row(
                3,
                "entry_to_launch",
                10,
                Some("r10"),
                0.01,
                8.9,
                pass(2048, 2048, 1),
            ),
            row(
                4,
                "launch_interval",
                10,
                None,
                0.12,
                8.78,
                pass(2048, 2048, 1),
            ),
            row(
                5,
                "last_launch_to_done",
                10,
                Some("r10"),
                0.15,
                8.75,
                pass(2048, 2048, 1),
            ),
            row(
                6,
                "request_complete",
                10,
                Some("r10"),
                0.2,
                8.75,
                complete(2048, 0),
            ),
            row(
                7,
                "entry_to_launch",
                12,
                Some("r12"),
                0.01,
                5.0,
                pass(1000, 500_500, 1),
            ),
            row(
                8,
                "last_launch_to_done",
                12,
                Some("r12"),
                0.08,
                4.92,
                pass(1000, 500_500, 1),
            ),
            row(
                9,
                "request_complete",
                12,
                Some("r12"),
                0.1,
                4.92,
                complete(5000, 4000),
            ),
            row(
                10,
                "entry_to_launch",
                13,
                Some("r13"),
                0.01,
                3.0,
                pass(3000, 3000, 1),
            ),
            row(
                11,
                "entry_to_launch",
                20,
                Some("r20"),
                0.01,
                2.0,
                pass(500, 500, 1),
            ),
            row(
                12,
                "launch_interval",
                20,
                None,
                0.05,
                1.95,
                pass(500, 500, 1),
            ),
            row(
                13,
                "entry_to_launch",
                21,
                Some("r21"),
                0.01,
                1.95,
                pass(800, 800, 1),
            ),
            row(
                14,
                "last_launch_to_done",
                20,
                Some("r20"),
                0.07,
                1.93,
                pass(500, 500, 1),
            ),
            row(
                15,
                "request_complete",
                20,
                Some("r20"),
                0.1,
                1.93,
                complete(500, 0),
            ),
        ];
        let v = view_of(rows);
        let labels = online.observe("http://p", &v, at(99.99), at(100.0));
        assert!(labels.is_empty());
        let guard = online.state.lock();
        let w = &guard.0.workers["http://p"];
        assert_eq!(w.after_idle.get(&10), Some(&false));
        assert_eq!(w.after_idle.get(&12), Some(&true));
        assert_eq!(w.after_idle.get(&9), None, "the run starts at pass 9");
        assert_eq!(w.service.len(), 1, "{:?}", w.counts);
        let (t, s) = &w.service[0];
        assert!((t - (100.0 - 4.92)).abs() < 1e-9);
        assert!((s.y_ms - 80.0).abs() < 1e-9);
        assert_eq!(s.f, features(1000.0, 500_500.0, 4000.0, 1.0));
        assert_eq!(w.launch.len(), 3);
        assert!((w.launch[0].y - 0.10).abs() < 1e-12);
        // r10: last pass 10, launch interval 10 (0.12 s) < 0.15 s, so pass
        // 11 launched before done; its features never arrive here. r20:
        // pass 21 (features known) launched 0.02 s before done.
        assert_eq!(w.pending_lag.len(), 1);
        assert_eq!(w.pending_lag[0].k_next, 11);
        assert!((w.pending_lag[0].y - 0.03).abs() < 1e-12);
        assert_eq!(w.cal_lag.len(), 1);
        assert!((w.cal_lag[0].1.y - 0.02).abs() < 1e-12);
        assert_eq!(w.cal_lag[0].1.f, features(800.0, 800.0, 0.0, 1.0));
        assert!(w.pending_service.is_empty());
        assert_eq!(w.cal0, Some(100.0 - 9.0));
        // phi: r10 was received inside the busy interval of passes 9 -> 10.
        assert_eq!(w.cal_phis.len(), 1);
        assert!((w.cal_phis[0].1 - 0.05).abs() < 1e-9);
        // idle delay: r12 and r20 (first pass after an idle gap): receipt
        // to entry 0.002 s + entry to launch 0.01 s.
        assert_eq!(w.cal_idles.len(), 2);
        assert!(w.cal_idles.iter().all(|(_, v)| (v - 0.012).abs() < 1e-12));
    }

    /// The engine keeps its last timeline rows however old. Rows that the
    /// router's first snapshot of a P already holds (here completed 600 s
    /// before it, for example the engine's warmup) feed the windowed fits
    /// but do not open the calibration window; the first row that completed
    /// after that snapshot does.
    #[test]
    fn rows_from_before_the_first_snapshot_do_not_open_the_calibration_window() {
        let t0 = Instant::now();
        let online = E3Online::new(t0);
        let at = |s: f64| t0 + std::time::Duration::from_secs_f64(s);
        let old = vec![
            row(
                1,
                "entry_to_launch",
                3,
                Some("w"),
                0.01,
                600.0,
                pass(4096, 4096, 1),
            ),
            row(
                2,
                "launch_interval",
                3,
                None,
                0.10,
                599.9,
                pass(4096, 4096, 1),
            ),
        ];
        online.observe("p", &view_of(old.clone()), at(1_000.0), at(1_000.0));
        {
            let guard = online.state.lock();
            let w = &guard.0.workers["p"];
            assert_eq!(w.first_seen, Some(1_000.0));
            assert_eq!(w.cal0, None, "rows of the first snapshot");
            assert_eq!(w.launch.len(), 1, "they feed the windowed fits");
            assert!(w.cal_launch.is_empty());
        }
        let mut new = old;
        new.push(row(
            3,
            "entry_to_launch",
            4,
            Some("n"),
            0.01,
            20.0,
            pass(100, 100, 1),
        ));
        new.push(row(
            4,
            "launch_interval",
            4,
            None,
            0.20,
            19.8,
            pass(100, 100, 1),
        ));
        online.observe("p", &view_of(new), at(1_100.0), at(1_100.0));
        let guard = online.state.lock();
        let w = &guard.0.workers["p"];
        assert_eq!(w.cal0, Some(1_080.0));
        assert_eq!((w.launch.len(), w.cal_launch.len()), (2, 1));
    }

    /// Service samples older than the retained 1,500 s move to the history
    /// of their size class (at most SERVICE_HISTORY_PER_CLASS, the newest
    /// kept). The own_all / pooled_all fits read the history and the
    /// retained samples, each class in time order (every earlier
    /// sample).
    #[test]
    fn old_service_samples_stay_for_the_all_sample_fallback() {
        let sample = |n: f64, y: f64| GpuSample {
            f: features(n, n, 0.0, 1.0),
            y_ms: y,
        };
        let mut st = Online::default();
        let ws = st.workers.entry("p".to_string()).or_default();
        for i in 0..SERVICE_HISTORY_PER_CLASS + 2 {
            ws.service.push((i as f64 * 0.01, sample(100.0, i as f64)));
        }
        ws.service.push((90.0, sample(4_000.0, -1.0)));
        ws.service.push((1_990.0, sample(200.0, -2.0)));
        ws.prune(2_000.0);
        assert_eq!(ws.service.len(), 1, "the retained 1,500 s");
        assert_eq!(ws.service_old[0].len(), SERVICE_HISTORY_PER_CLASS);
        assert_eq!(ws.service_old[0].front().map(|s| s.y_ms), Some(2.0));
        assert_eq!(ws.service_old[1].len(), 1);
        let data = collect_fit_data(&st, "p", 2_040.0);
        assert_eq!(data.own_window.len(), 1);
        assert_eq!(data.own_all.len(), SERVICE_HISTORY_PER_CLASS + 2);
        assert_eq!(data.pooled_all.len(), data.own_all.len());
        let small: Vec<f64> = data
            .own_all
            .iter()
            .filter(|s| s.f[1] <= crate::e3::GRAPH_TOKENS as f64)
            .map(|s| s.y_ms)
            .collect();
        assert_eq!(small.len(), SERVICE_HISTORY_PER_CLASS + 1);
        assert_eq!((small[0], small[small.len() - 1]), (2.0, -2.0));
        assert!(small[..small.len() - 1].windows(2).all(|w| w[0] < w[1]));
    }

    /// The dispatch log of a P is pruned when a route is recorded, also for
    /// a P whose snapshots the router never ingests.
    #[test]
    fn the_dispatch_log_is_pruned_when_routes_are_recorded() {
        let online = E3Online::new(Instant::now());
        for i in 0..100 {
            online.record_route(RouteRecord {
                rid: format!("r{i}"),
                worker: "q".into(),
                decided_s: f64::from(i) * 30.0,
                decided_unix_s: 0.0,
                input: 1_000,
                prefix: 0,
                host: 0,
                idle: None,
                pred_s: None,
                session: None,
                session_gap_s: None,
            });
        }
        let guard = online.state.lock();
        let w = &guard.0.workers["q"];
        // Decisions every 30 s up to 2,970 s; a fit window needs 660 s.
        assert_eq!(w.dispatch.len(), 23);
        assert!(w.dispatch.iter().all(|r| r.t >= 2_970.0 - 660.0));
    }

    /// A sequence gap drops the pending service candidates; a row seen again
    /// is not counted twice.
    #[test]
    fn sequence_gaps_and_repeats_are_handled() {
        let t0 = Instant::now();
        let online = E3Online::new(t0);
        let at = |s: f64| t0 + std::time::Duration::from_secs_f64(s);
        let first = vec![
            row(
                1,
                "entry_to_launch",
                11,
                Some("a"),
                0.01,
                5.0,
                pass(100, 100, 1),
            ),
            row(
                2,
                "entry_to_launch",
                12,
                Some("r12"),
                0.01,
                4.0,
                pass(1000, 500_500, 1),
            ),
            row(
                3,
                "last_launch_to_done",
                12,
                Some("r12"),
                0.08,
                3.92,
                pass(1000, 500_500, 1),
            ),
            row(
                4,
                "request_complete",
                12,
                Some("r12"),
                0.1,
                3.92,
                complete(1000, 0),
            ),
        ];
        online.observe("p", &view_of(first.clone()), at(9.9), at(10.0));
        // The same rows again, then rows 6.. (row 5 missing): the pending
        // sample of pass 12 is dropped although pass 13 shows up.
        let mut second = first;
        second.push(row(
            6,
            "entry_to_launch",
            13,
            Some("b"),
            0.01,
            1.0,
            pass(100, 100, 1),
        ));
        online.observe("p", &view_of(second), at(10.9), at(11.0));
        let guard = online.state.lock();
        let w = &guard.0.workers["p"];
        assert!(w.service.is_empty());
        assert!(w.pending_service.is_empty());
        assert_eq!(w.counts.get("timeline_gaps"), Some(&1));
        assert_eq!(w.last_seq, Some(6));
    }

    /// Routed requests: an idle forecast's completion trains the receipt
    /// line (rx2) and every routed completion gives an `e3_label` line whose
    /// interval holds the poll round trip.
    #[test]
    fn completions_of_routed_requests_give_receipt_samples_and_labels() {
        let t0 = Instant::now();
        let online = E3Online::new(t0);
        let at = |s: f64| t0 + std::time::Duration::from_secs_f64(s);
        online.record_route(RouteRecord {
            rid: "r1".into(),
            worker: "p".into(),
            decided_s: 50.0,
            decided_unix_s: 1_000.0,
            input: 2_000,
            prefix: 0,
            host: 0,
            idle: Some(true),
            pred_s: Some(0.25),
            session: Some("s".into()),
            session_gap_s: None,
        });
        let rows = vec![row(
            1,
            "request_complete",
            3,
            Some("r1"),
            0.25,
            49.5,
            complete(2_000, 0),
        )];
        // Poll sent at 99.875, received at 100.0, engine age 0.125: the
        // capture is in [99.75, 99.875]; done = capture - 49.5.
        let mut v = view_of(rows);
        v.age_s = Some(0.125);
        let labels = online.observe("p", &v, at(99.875), at(100.0));
        assert_eq!(labels.len(), 1);
        let l = &labels[0];
        assert_eq!(l["event"], "e3_label");
        assert_eq!(
            (l["label_lo_s"].as_f64(), l["label_hi_s"].as_f64()),
            (Some(0.25), Some(0.375))
        );
        assert_eq!(l["label_s"].as_f64(), Some(0.3125));
        assert_eq!(l["ape"].as_f64(), Some(0.2), "{l}");
        assert_eq!(l["done_mono_s"].as_f64(), Some(50.375));
        let guard = online.state.lock();
        let w = &guard.0.workers["p"];
        // receipt = done - 0.25 = 50.125: y = 0.125 s after the decision.
        assert_eq!(w.rx.len(), 1);
        assert!(guard.0.routes.is_empty(), "a label is written once");
        assert_eq!(guard.0.host_samples.len(), 0, "no cached tokens");
        assert_eq!(w.dispatch.len(), 1);
        assert_eq!(
            guard.0.last_route.get(&("s".to_string(), "p".to_string())),
            Some(&50.0)
        );
    }

    /// The session host rule: per (size, gap) bin, the share of cached
    /// tokens that came from host decides between the whole prefix and none.
    #[test]
    fn the_session_host_rule_learns_the_share_per_bin() {
        let online = E3Online::new(Instant::now());
        {
            let mut guard = online.state.lock();
            let st = &mut guard.0;
            for i in 0..HOST_MIN_SAMPLES {
                // Gap bin 4 (>= 30 s): host; gap bin 0 (< 2 s): device.
                st.host_samples.push_back(HostSample {
                    t: 100.0 + i as f64,
                    bin: host_bin(40_000, Some(60.0)),
                    host: 40_000.0,
                    cached: 40_000.0,
                });
                st.host_samples.push_back(HostSample {
                    t: 100.0 + i as f64,
                    bin: host_bin(40_000, Some(1.0)),
                    host: 0.0,
                    cached: 40_000.0,
                });
            }
            st.last_route.insert(("s".into(), "p".into()), 160.0);
            st.last_route.insert(("old".into(), "p".into()), 100.0);
        }
        // now = 190 s: step 180; samples at 100..119 are in [-420, 180).
        let gap = online.session_gap(Some("s"), "p", 190.0);
        assert_eq!(gap, Some(30.0));
        assert_eq!(online.host_rule(gap, 40_000, 190.0), (40_000, "share_host"));
        let gap = online.session_gap(Some("old"), "p", 190.0);
        assert_eq!(gap, Some(90.0));
        assert_eq!(online.host_rule(gap, 40_000, 190.0), (40_000, "share_host"));
        let gap = online.session_gap(Some("s"), "p", 161.0);
        assert_eq!(online.host_rule(gap, 40_000, 161.0), (0, "share_device"));
        assert_eq!(online.session_gap(None, "p", 190.0), None);
        assert_eq!(online.session_gap(Some("s"), "q", 190.0), None);
        assert_eq!(online.host_rule(None, 40_000, 190.0), (0, "no_share"));
        assert_eq!(online.host_rule(Some(30.0), 0, 190.0), (0, "cold"));
        assert_eq!(host_bin(32_767, None), (0, None));
        assert_eq!(host_bin(32_768, Some(2.0)), (1, Some(1)));
        assert_eq!(host_bin(131_072, Some(30.0)), (2, Some(4)));
    }

    /// The (P, step) fit: the GPU model comes from the window when it has
    /// enough service samples per class, the launch model from the window
    /// rows, the constants from the calibration samples, the arrival rates
    /// from the dispatch log; one fit per step.
    #[test]
    fn the_step_fit_uses_the_windows_and_is_cached_per_step() {
        let sample = |n: f64, pairs: f64| GpuSample {
            f: features(n, pairs, 0.0, 1.0),
            y_ms: 5.0 + 0.02 * n + 3.0 * pairs / 1e6,
        };
        let mut data = FitData::default();
        for i in 0..40 {
            let n = 100.0 + 50.0 * i as f64;
            let pairs = ((i * 7919) % 1000) as f64 * 1e3;
            data.own_window.push(sample(n, pairs));
            data.own_window.push(sample(n + 3000.0, pairs));
        }
        data.pooled_window = data.own_window.clone();
        for i in 0..40 {
            data.launch_window.push(LaunchSample {
                t: i as f64,
                f: features(4096.0, 1e6, 0.0, 1.0),
                fin: (i % 2) as f64,
                host_next: 0.0,
                y: 0.05 + 0.01 * (i % 2) as f64,
            });
        }
        data.boots = vec![0.001, 0.002, 0.003];
        data.phis = vec![0.125, 0.375];
        data.idles = vec![0.05, 0.07];
        let fit = compute_fit(600.0, &data);
        assert!(fit.usable());
        assert_eq!(fit.gpu_sources.small.as_deref(), Some("own"));
        assert_eq!(fit.gpu_sources.large.as_deref(), Some("own"));
        let small = fit.gpu.small.clone().unwrap();
        assert!(
            (small[0] - 5.0).abs() < 1e-2 && (small[1] - 0.02).abs() < 1e-5,
            "{small:?}"
        );
        assert_eq!(fit.launch.source, crate::e3::LaunchSource::Window);
        assert_eq!(fit.constants.boot_s, 0.002);
        assert_eq!(fit.constants.phi_s, 0.25);
        assert!((fit.constants.idle_delay_s - 0.058).abs() < 1e-12);
        assert_eq!(fit.constants.lag_none_s, 0.0);
        assert_eq!(fit.constants.lag_next, [0.0, 0.0], "no lag samples");
        assert!(fit.arrivals.is_none());
        let p = fit.params(None, 1e6);
        assert_eq!(p.gpu_idle, p.gpu_busy);
        assert!(!p.next_pass_lag, "the reference result lag");
        // `--co-schedule-trade-e3-next-pass-lag`: the prepared forecast
        // parameters carry the switch (the fit of step 600 s serves 610 s).
        let usable = Arc::new(fit.clone());
        for on in [false, true] {
            let online = E3Online::new(Instant::now()).with_next_pass_lag(on);
            online
                .state
                .lock()
                .0
                .fits
                .insert("p".into(), Arc::clone(&usable));
            let tokens = CandidateTokens {
                input: 5_000,
                local_hit: 0,
                gpu_hit: None,
                expected_hit: 0,
            };
            let (prep, new_fit) = online.prepare("p", None, "r", tokens, None, 610.0);
            assert!(new_fit.is_none(), "the fit of the step is reused");
            assert_eq!(prep.params.map(|p| p.next_pass_lag), Some(on));
        }
        // No service sample: no GPU model, no forecast.
        let empty = compute_fit(600.0, &FitData::default());
        assert!(!empty.usable());
        assert_eq!(
            empty.constants.idle_delay_s,
            crate::e3::DEFAULT_IDLE_DELAY_S
        );
        assert_eq!(empty.launch.coef[0], crate::e3::DEFAULT_LAUNCH_INTERVAL_S);
        // One fit per (P, step): the second call of the step returns it.
        let online = E3Online::new(Instant::now());
        let (a, new_a) = online.fit_for("p", 125.0);
        let (b, new_b) = online.fit_for("p", 179.0);
        assert!(Arc::ptr_eq(&a, &b));
        assert!(new_a && !new_b);
        assert_eq!(a.step, 120.0);
        let (c, new_c) = online.fit_for("p", 180.0);
        assert_eq!(c.step, 180.0);
        assert!(new_c);
    }

    /// `--co-schedule-trade-e3-frozen-fit`: the recorded Kimi constants load
    /// from the e3_constants object and from a step model that wraps it;
    /// bad files are refused with the reason.
    #[test]
    fn frozen_fit_loads_the_offline_constants() {
        let path = fixture("kimi-frozen-fit.json");
        let frozen = FrozenFit::load(&path).unwrap();
        assert_eq!(frozen.sha256.len(), 64);
        assert!(frozen.gpu.small.is_none());
        let large = frozen.gpu.large.clone().unwrap();
        assert_eq!(large.len(), 5);
        assert!((large[0] - 346.128).abs() < 1e-3);
        assert!((large[4] - 120.382).abs() < 1e-3);
        assert!((frozen.gpu.floor_ms - 711.724).abs() < 1e-3);
        assert_eq!(frozen.launch_coef, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(frozen.launch_resid_q.is_none());
        let c = frozen.constants;
        assert!((c.lag_next[0] + 0.050288).abs() < 1e-6);
        assert!((c.lag_next[1] - 0.949088).abs() < 1e-6);
        assert!((c.lag_none_s + 0.080503).abs() < 1e-6);
        assert!((c.boot_s - 0.128301).abs() < 1e-6);
        assert_eq!(c.phi_s, 0.0);
        let text = std::fs::read_to_string(&path).unwrap();
        let body: Value = serde_json::from_str(&text).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, value: &Value| {
            let p = dir.path().join(name);
            let bytes = serde_json::to_vec(value).unwrap();
            std::fs::write(&p, bytes).unwrap();
            p
        };
        // The step model that wraps the object (kimi.e3_constants).
        let model = json!({"kimi": {"e3_constants": body}});
        let wrapped = FrozenFit::load(&write("model.json", &model)).unwrap();
        assert_eq!(wrapped.gpu, frozen.gpu);
        assert_eq!(wrapped.launch_coef, frozen.launch_coef);
        assert_eq!(wrapped.constants, frozen.constants);
        let refused = |pointer: &str, value: Value, what: &str| {
            let mut v = body.clone();
            *v.pointer_mut(pointer).unwrap() = value;
            let err = FrozenFit::load(&write("bad.json", &v)).unwrap_err();
            assert!(err.contains(what), "{err}");
        };
        refused("/gpu_busy/large", Value::Null, "neither a small");
        refused("/gpu_busy/large", json!([1.0, 2.0]), "needs 5 finite");
        refused("/gpu_busy/floor_ms", json!(-1.0), "floor_ms must be");
        refused("/gpu_idle/floor_ms", json!(1.0), "gpu_idle differs");
        refused("/launch_coef", json!([0.0, 1.0]), "needs 6 finite");
        refused("/launch_anchor", json!("start"), "must be launch");
        refused("/chunk_tokens", json!(4096), "4096 is not the router's");
    }

    /// With the frozen fit a P without any service sample has a forecast:
    /// the fit takes the file's GPU model, launch model and constants
    /// (sources `frozen`); without it the same P has none (`no_fit`).
    #[test]
    fn frozen_fit_gives_a_forecast_without_service_samples() {
        let path = fixture("kimi-frozen-fit.json");
        let frozen = Arc::new(FrozenFit::load(&path).unwrap());
        let online = E3Online::new(Instant::now());
        let online = online.with_frozen_fit(Some(Arc::clone(&frozen)));
        let (fit, new) = online.fit_for("p", 125.0);
        assert!(new && fit.usable());
        assert_eq!(fit.gpu, frozen.gpu);
        assert_eq!(fit.gpu_sources.small, None);
        assert_eq!(fit.gpu_sources.large.as_deref(), Some(FROZEN_SOURCE));
        assert_eq!(fit.launch.source, crate::e3::LaunchSource::Frozen);
        assert_eq!(fit.launch.coef, frozen.launch_coef);
        assert_eq!(fit.constants, frozen.constants);
        let record = fit.record();
        assert_eq!(record["gpu_large"], "frozen");
        assert_eq!(record["launch"], "frozen");
        let p = fit.params(None, 1e6);
        assert_eq!(p.gpu_idle, frozen.gpu);
        assert_eq!(p.gpu_busy, frozen.gpu);
        assert_eq!(p.lag_next, frozen.constants.lag_next);
        assert_eq!(p.lag_none_s, frozen.constants.lag_none_s);
        let tokens = CandidateTokens {
            input: 5_000,
            local_hit: 0,
            gpu_hit: None,
            expected_hit: 0,
        };
        let online = E3Online::new(Instant::now())
            .with_next_pass_lag(true)
            .with_frozen_fit(Some(frozen));
        let (prep, new_fit) = online.prepare("p", None, "r", tokens, None, 610.0);
        assert!(new_fit.is_some());
        let params = prep.params.expect("a forecast without service samples");
        assert!(params.next_pass_lag);
        assert_eq!(params.launch_coef, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let online = E3Online::new(Instant::now());
        let (prep, _) = online.prepare("p", None, "r", tokens, None, 610.0);
        assert!(prep.params.is_none(), "no service sample: no_fit");
    }

    /// The shipped bound file: snap+prod at its level 0.8; bound-tables.json
    /// needs an explicit level; unknown modes and levels are refused.
    #[test]
    fn bound_files_load_with_their_level() {
        let params = fixture("bound-params.json");
        let cfg = TradeE3Config::load(&params, DEFAULT_BOUND_MODE, None, 0.0, None).unwrap();
        assert_eq!(cfg.level, "0.8");
        assert_eq!(cfg.file_level.as_deref(), Some("0.8"));
        assert_eq!(cfg.file_g_min_s, Some(0.1));
        assert_eq!(cfg.bound_schema, crate::e3::BOUND_PARAMS_SCHEMA);
        assert_eq!(
            cfg.bound_sha256,
            "11455495bef2612b5de3dfef5226ffd9c9a3b5ac87bd9e5af02784a6c2e2feb5"
        );
        let cfg9 =
            TradeE3Config::load(&params, DEFAULT_BOUND_MODE, Some("0.9"), 0.0, None).unwrap();
        assert_eq!(cfg9.level, "0.9");
        let tables = fixture("bound-tables.json");
        assert!(
            TradeE3Config::load(&tables, DEFAULT_BOUND_MODE, None, 0.0, None)
                .unwrap_err()
                .contains("--co-schedule-trade-e3-bound-level")
        );
        let t = TradeE3Config::load(&tables, DEFAULT_BOUND_MODE, Some("0.8"), 0.0, None).unwrap();
        assert_eq!(t.bound_schema, "bound-tables");
        assert_eq!(t.table, cfg.table);
        assert!(TradeE3Config::load(&params, "rfng", None, 0.0, None)
            .unwrap_err()
            .contains("no deploy mode rfng"));
        assert!(
            TradeE3Config::load(&params, DEFAULT_BOUND_MODE, Some("0.7"), 0.0, None)
                .unwrap_err()
                .contains("no level 0.7")
        );
        assert!(TradeE3Config::load(&params, DEFAULT_BOUND_MODE, None, -1.0, None).is_err());
        // Only modes with the router's inputs: snap for the poll table;
        // snap, push, pushL or pushslow with prod or prodL for the push one.
        let snap_l = TradeE3Config::load(&params, "snap+prodL", None, 0.0, None).unwrap();
        assert_eq!(
            (snap_l.mode.as_str(), snap_l.level.as_str()),
            ("snap+prodL", "0.8")
        );
        assert!(TradeE3Config::load(&params, "push+prodL", None, 0.0, None)
            .unwrap_err()
            .contains("does not match the router's poll forecasts"));
        for mode in ["fresh+prodL", "rfng+rq+rcal", "split", "pushslow+prodL"] {
            let err = TradeE3Config::load(&tables, mode, Some("0.8"), 0.0, None).unwrap_err();
            assert!(
                err.contains("does not match the router's poll"),
                "{mode}: {err}"
            );
        }
        let push = cfg
            .clone()
            .with_push(5600, Some("pushL+prodL"), None)
            .unwrap();
        assert_eq!(push.mode_for(true), "pushL+prodL");
        assert!(cfg.clone().with_push(5600, None, None).is_ok(), "snap+prod");
        for mode in ["pushL+gpu+prod", "push+split+prodL"] {
            let err = cfg.clone().with_push(5600, Some(mode), None).unwrap_err();
            assert!(
                err.contains("does not match the router's pushed-state forecasts"),
                "{mode}: {err}"
            );
        }
        // A bound grows with the forecast and the idle bucket has its curve.
        let out = |pred: f64, idle: bool| Output {
            schema: crate::e3::SCHEMA.into(),
            pred_s: Some(pred),
            idle,
            n_inflight: 0,
            partial: false,
            n_queue: 0,
            n_dispatched: 0,
            n_prefix_unknown: 0,
            tokens_ahead: 0,
            x_uncached: 1,
            x_host: 0,
            receipt_s: 0.0,
            path: ForecastPath::Full,
            f_full_s: None,
            f_after_s: None,
            n_held: 0,
            x_held: false,
            u_tokens: None,
            wait_s: Some(0.0),
            x_passes: 1,
            passes_before: 0,
        };
        let b1 = cfg.bound(&out(0.5, false)).unwrap();
        let b2 = cfg.bound(&out(5.0, false)).unwrap();
        assert!(b2 > b1 && b1 > 0.0);
        assert_eq!(cfg.bucket(&out(0.5, true)), "idle");
        assert_eq!(cfg.bucket(&out(0.5, false)), "busy");
    }

    #[test]
    fn the_shadow_file_is_new_and_line_per_event() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shadow.jsonl");
        let sink = ShadowSink::file(&path).unwrap();
        assert!(ShadowSink::file(&path).is_err(), "never appended");
        sink.write(json!({"event": "e3_shadow", "rid": "a"}));
        sink.write(json!({"event": "e3_label", "rid": "a"}));
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1]["event"], "e3_label");
        assert!(lines[0]["ts_unix_s"].as_f64().unwrap() > 0.0);
    }

    /// The target of a candidate: page-floored own prefix plus the pulled
    /// tokens; host = the non-GPU part when the indexer reports the GPU hit
    /// (`split`), else the session rule (no samples: `no_share`) plus the
    /// pulled tokens. Without a service sample there is no forecast model.
    #[test]
    fn candidates_are_prepared_with_the_split_or_the_session_rule() {
        let online = E3Online::new(Instant::now());
        let tokens = CandidateTokens {
            input: 50_000,
            local_hit: 20_000,
            gpu_hit: Some(8_010),
            expected_hit: 30_000,
        };
        let (prep, new_fit) = online.prepare("p", None, "r", tokens, Some("s"), 10.0);
        // floor_hit: 20,000 -> 19,968; 30,000 -> 29,952; GPU 8,010 -> 8,000.
        assert_eq!(prep.target.prefix, 29_952);
        assert_eq!(prep.target.host, 29_952 - 8_000);
        assert_eq!(prep.host_rule, "split");
        assert_eq!(prep.session_gap_s, None);
        assert!(prep.params.is_none(), "no service sample, no GPU model");
        assert!(new_fit.is_some());
        let (prep, new_fit) = online.prepare(
            "p",
            None,
            "r",
            CandidateTokens {
                gpu_hit: None,
                ..tokens
            },
            Some("s"),
            11.0,
        );
        assert_eq!(prep.target.host, 29_952 - 19_968, "pulled tokens only");
        assert_eq!(prep.host_rule, "no_share");
        assert!(new_fit.is_none(), "same step");
        // A request that would cache everything keeps one page to compute.
        let full = CandidateTokens {
            input: 4_096,
            local_hit: 4_096,
            gpu_hit: Some(4_096),
            expected_hit: 4_096,
        };
        let (prep, _) = online.prepare("p", None, "r", full, None, 12.0);
        assert_eq!((prep.target.prefix, prep.target.host), (4_032, 0));
    }

    #[test]
    fn the_session_key_prefers_the_kv_session_agent() {
        let mut headers = axum::http::HeaderMap::new();
        assert_eq!(session_key(&headers, None), None);
        headers.insert("x-smg-routing-key", "agent-7".parse().unwrap());
        assert_eq!(
            session_key(&headers, None).as_deref(),
            Some("routing-key/agent-7")
        );
        headers.insert(
            crate::agent_identity::KV_SESSION_HEADER,
            r#"{"agent_id":"a1","subagent_id":2,"request_id":5}"#
                .parse()
                .unwrap(),
        );
        assert_eq!(session_key(&headers, None).as_deref(), Some("a1/2"));
        let other = crate::agent_identity::RequestIdentity {
            agent_id: "b".into(),
            subagent_id: 0,
            request_id: 1,
        };
        assert_eq!(session_key(&headers, Some(&other)).as_deref(), Some("b/0"));
    }

    #[test]
    fn router_clock_seconds_can_be_negative() {
        let anchor = Instant::now() + std::time::Duration::from_secs(5);
        let now = Instant::now();
        assert!(secs_since(anchor, now) < -4.0);
        assert!((secs_since(now, now + std::time::Duration::from_millis(1500)) - 1.5).abs() < 1e-9);
    }
}
