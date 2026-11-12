// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! forecast: router-side forecast of the completion time of a prefill (P) worker.
//!
//! This module is a Rust port of the Python reference `router_forecast.py` (code
//! version r5). The reference is the source of truth. The port is
//! self-contained: it reads no router state and changes no router behaviour.
//! The routing policy calls it from [`crate::prefill_forecast`].
//!
//! What it computes: `pred_s` = seconds from the routing decision `d`
//! to the selected P worker's `prefill_finished_time` of the routed request.
//! The forecast simulates the FCFS chunked passes of the P worker from its
//! last known state (in-flight passes, the started chunked request, the queue
//! rows and the rows the router sent that the state does not show), then the
//! target's own passes and the result lag.
//!
//! Files:
//! - [`numeric`]: `quantile`, `solve`, `lstsq`, `nonneg_fit`,
//!   `resid_quantiles` and the Python arithmetic helpers.
//! - [`forecast`]: inputs, parameters and outputs;
//!   `features`, `gpu_s`, `receipt_delays`, `row_ready`, `simulate`,
//!   `forecast`.
//! - [`fit`]: the online fits and the warmup / router calibration.
//! - [`bound`]: the error bound table v2 and the reroute gain.
//!
//! # Exactness (with one correction)
//!
//! All real values are `f64`, token counts are `i64`, and every float
//! expression is evaluated in the order of the reference. The golden tests
//! (`tests/fixtures/forecast/`, see `PROVENANCE.md` there) compare the port with
//! the reference outputs.
//!
//! Sums: the reference runs
//! on CPython 3.12.3 (x86-64 and aarch64 hosts). Since Python 3.12 the
//! built-in `sum()` of floats is a Neumaier compensated sum, not a
//! left-to-right sum. [`numeric::py_sum`] implements the CPython algorithm,
//! and the port uses it at every `sum()` call of the reference. Explicit
//! running sums (`+=`, for example the receipt line accumulators and the pass
//! formation) stay plain additions. With left-to-right sums the fits differ
//! from the goldens (calibrate by up to 2.4e-8, the GPU fits by 3.9e-10).
//! The reference computes squares with `x ** 2` (C `pow`); the port uses
//! `x * x`. On all 9,392 r5 goldens the two give equal results (checked with
//! a variant of the reference).
//!
//! # Constants and their sources
//!
//! Every remaining constant names its source. Engine policy values:
//! [`CHUNK_TOKENS`], [`PAGE_TOKENS`], [`PARK_CAP`]. Fitted or calibrated values
//! are not constants: they come from [`fit`] (GPU model, launch model, receipt
//! line, arrival rates, warmup constants) and from the bound tables
//! (`bound-params.json`). The other constants below are fixed design choices of
//! the forecast reference (`estate_replay_kv.py` / `router_forecast.py`). They were not
//! tuned on the forecast error in this port; each one says what it controls.

pub mod error_table;
pub mod fit;
pub mod model;
pub mod numeric;

#[cfg(test)]
mod golden_tests;

pub use error_table::{
    bound_curve, bucket_of, error_bound, reroute_gain, BoundCurve, BoundModeParams, BoundTable,
    BucketScheme, BucketView, ErrorTable,
};
pub use fit::{
    arrival_rates, calibrate, fit_gpu_classes, fit_launch_intervals, fit_launch_rows,
    gpu_chain_from_rows, launch_interval_design, make_params, resolve_gpu, router_floor_ms,
    router_lag_next, start_anchored_pairs, step_of, window, AnchoredPair, ArrivalRow, CalIter,
    CalReq, Calibration, Constants, DesignRow, GpuChain, GpuClasses, GpuSample, GpuSources,
    LagSample, LaunchFit, LaunchSample, LaunchSource, RxLine,
};
pub use model::{
    features, forecast, gpu_s, receipt_delays, row_ready, simulate, Arrivals, Features,
    ForecastPath, GpuModel, InflightPass, Input, LaunchAnchor, Output, Params, Partial, QueueRow,
    RowSource, RxParams, SimRow, SimState, Target, Trace,
};

/// Output schema of [`forecast()`] (`forecast.v2`).
pub const SCHEMA: &str = "forecast.v2";
/// Schema of an error bound table (table v2).
pub const BOUND_TABLE_SCHEMA: &str = "forecast-error-bound.v2";
/// Schema of `bound-params.json` (the shipped bound tables and levels).
pub const ERROR_TABLE_SCHEMA: &str = "forecast-error-table.v2";

/// Engine policy: tokens per prefill pass (`chunked_prefill_size`); default
/// of `Params::chunk_tokens`. The router sets the P's own value from its
/// snapshot; the replay that fitted the shipped bound tables used this one.
pub const CHUNK_TOKENS: i64 = 8192;
/// Engine policy: KV page size in tokens; a chunk uses its page-rounded size
/// of the pass budget. Default of `Params::page_tokens`; the router sets the
/// P's own value from its snapshot.
pub const PAGE_TOKENS: i64 = 64;
/// GPU model size classes: a pass with at most this many new tokens uses the
/// `small` coefficients, a larger pass the `large` ones (forecast reference).
pub const GRAPH_TOKENS: i64 = 2048;
/// Receipt delay (s) of a request while the receipt line has fewer than
/// [`RX_MIN`] samples (forecast reference value; replaced by the fitted line).
pub const DISPATCH_S: f64 = 0.044;
/// Park-gate choice: the forecast without the held rows is used when its
/// share `F_after / (F_after + F_full)` is at most this value
/// (`estate_replay_kv.LO14_P_AFTER`).
pub const P_AFTER: f64 = 0.2;
/// Engine setting `SGLANG_LATE_D_PARK_CAP`: `cap_tokens` = this share of the
/// KV token pool.
pub const PARK_CAP: f64 = 0.35;
/// Token pool of the replay when the engine server info has none.
pub const DEFAULT_POOL_TOKENS: i64 = 2_067_584;
/// Receipt line: minimum number of samples before the line is used.
pub const RX_MIN: u64 = 30;
/// Receipt line: the idle shift is the median residual of this many latest
/// samples.
pub const RX_SHIFT_SAMPLES: usize = 500;
/// Online refit grid (s): one fit per (P worker, step of this length).
pub const STEP_S: f64 = 60.0;
/// Online fits use the samples of this window (s) before the step start.
pub const WINDOW_S: f64 = 600.0;
/// Calibration window (s) before the first scored decision (production: the
/// last 900 s of the warmup).
pub const CAL_S: f64 = 900.0;
/// Minimum samples per size class of a GPU model fit, and of the
/// next-aware result lag fit.
pub const GPU_FIT_MIN: usize = 20;
/// Minimum samples of a launch model fit.
pub const LAUNCH_FIT_MIN: usize = 30;
/// Minimum requests in the window for the arrival rates.
pub const ARRIVAL_MIN: usize = 30;
/// Minimum consecutive busy pass pairs for the warmup launch line.
pub const LAUNCH_LINE_MIN: usize = 20;
/// Launch samples are kept only when `0 <= y < LAUNCH_Y_MAX_S` (s).
pub const LAUNCH_Y_MAX_S: f64 = 5.0;
/// Simulation guard: no forecast after this many simulated passes.
pub const MAX_PASSES: usize = 100_000;
/// Ridge term of [`numeric::lstsq`] on unit-norm columns (numerical
/// conditioning only).
pub const RIDGE: f64 = 1e-6;
/// [`numeric::solve`] reports a singular system below this pivot size.
pub const SINGULAR_PIVOT: f64 = 1e-300;
/// GPU seconds of a pass when the GPU model has no coefficients.
pub const NO_MODEL_GPU_S: f64 = 0.3;
/// GPU model floor (ms) when no GPU sample gives one.
pub const DEFAULT_FLOOR_MS: f64 = 1.0;
/// Constant launch interval (s) of the G1 launch fit without window rows,
/// calibration rows or a median of the P's launch samples.
pub const DEFAULT_LAUNCH_INTERVAL_S: f64 = 0.1;
/// Warmup fallbacks (no samples in the calibration window): idle launch
/// delay (s), result lag (s) and the launch line `[a s, b s/s]`.
pub const DEFAULT_IDLE_DELAY_S: f64 = 0.03;
/// See [`DEFAULT_IDLE_DELAY_S`].
pub const DEFAULT_LAG_S: f64 = 0.06;
/// See [`DEFAULT_IDLE_DELAY_S`].
pub const DEFAULT_LAUNCH_LINE: [f64; 2] = [0.06, 0.75];

/// Errors where the Python reference raises an exception (invalid
/// parameters or inputs). A forecast without a result is not an error: it is
/// `Output::pred_s == None`.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ForecastError {
    #[error("router_e3: invalid parameters: {0}")]
    Params(&'static str),
    #[error("router_e3: invalid input: {0}")]
    Input(&'static str),
    #[error("router_e3: singular system in {0}")]
    Singular(&'static str),
    #[error("router_e3: bound table: {0}")]
    Bound(String),
}
