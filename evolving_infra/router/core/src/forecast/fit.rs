// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! Online fits from router-visible samples and the warmup / router
//! calibration: refit grid, GPU model per size class, launch
//! models, receipt line, arrival rates, `calibrate`, the router floor and
//! next-aware lag, and the parameter assembly.
//!
//! The sample types serialize as the JSON arrays of the reference's fit
//! goldens (for example a launch sample is `[t, f, fin, host_next, y]`).

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use super::model::{gpu_s, Arrivals, Features, GpuModel, LaunchAnchor, Params, RxParams};
use super::numeric::{
    finite, lstsq, nonneg_fit, py_max, py_min, py_sum, quantile, resid_quantiles, sort_py,
    truthy_or,
};
use super::{
    ForecastError, CHUNK_TOKENS, DEFAULT_FLOOR_MS, DEFAULT_IDLE_DELAY_S, DEFAULT_LAG_S,
    DEFAULT_LAUNCH_INTERVAL_S, DEFAULT_LAUNCH_LINE, DISPATCH_S, GPU_FIT_MIN, GRAPH_TOKENS,
    LAUNCH_FIT_MIN, LAUNCH_LINE_MIN, LAUNCH_Y_MAX_S, PAGE_TOKENS, P_AFTER, RIDGE, RX_MIN,
    RX_SHIFT_SAMPLES, SCHEMA,
};

/// GPU model fit per size class (ms coefficients; None below the sample
/// minimum or on a singular system).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GpuClasses {
    #[serde(default)]
    pub small: Option<Vec<f64>>,
    #[serde(default)]
    pub large: Option<Vec<f64>>,
}

/// Chain entry name that gave each class in [`resolve_gpu`] (None: no fit).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuSources {
    pub small: Option<String>,
    pub large: Option<String>,
}

/// A GPU (service) sample: pass features and the GPU / service time in ms
/// (JSON `[f, y_ms]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "(Features, f64)", into = "(Features, f64)")]
pub struct GpuSample {
    pub f: Features,
    pub y_ms: f64,
}

impl From<(Features, f64)> for GpuSample {
    fn from((f, y_ms): (Features, f64)) -> Self {
        Self { f, y_ms }
    }
}

impl From<GpuSample> for (Features, f64) {
    fn from(s: GpuSample) -> Self {
        (s.f, s.y_ms)
    }
}

/// A design row `(x, y)` of a launch model (JSON `[x, y]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "(Vec<f64>, f64)", into = "(Vec<f64>, f64)")]
pub struct DesignRow {
    pub x: Vec<f64>,
    pub y: f64,
}

impl From<(Vec<f64>, f64)> for DesignRow {
    fn from((x, y): (Vec<f64>, f64)) -> Self {
        Self { x, y }
    }
}

impl From<DesignRow> for (Vec<f64>, f64) {
    fn from(r: DesignRow) -> Self {
        (r.x, r.y)
    }
}

/// A launch-interval sample of consecutive passes k, k + 1 (JSON
/// `[t, f, fin, host_next, y]`): time key `t` = launch of k + 1, features
/// `f` and `fin` (finishing requests) of pass k, host tokens of k + 1,
/// `y = launch(k + 1) - launch(k)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    from = "(f64, Features, f64, f64, f64)",
    into = "(f64, Features, f64, f64, f64)"
)]
pub struct LaunchSample {
    pub t: f64,
    pub f: Features,
    pub fin: f64,
    pub host_next: f64,
    pub y: f64,
}

impl From<(f64, Features, f64, f64, f64)> for LaunchSample {
    fn from((t, f, fin, host_next, y): (f64, Features, f64, f64, f64)) -> Self {
        Self {
            t,
            f,
            fin,
            host_next,
            y,
        }
    }
}

impl From<LaunchSample> for (f64, Features, f64, f64, f64) {
    fn from(s: LaunchSample) -> Self {
        (s.t, s.f, s.fin, s.host_next, s.y)
    }
}

/// A request received by the P worker (JSON `[t, u, c]`): receipt
/// time, uncached and cached tokens.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(from = "(f64, i64, i64)", into = "(f64, i64, i64)")]
pub struct ArrivalRow {
    pub t: f64,
    pub uncached: i64,
    pub cached: i64,
}

impl From<(f64, i64, i64)> for ArrivalRow {
    fn from((t, uncached, cached): (f64, i64, i64)) -> Self {
        Self {
            t,
            uncached,
            cached,
        }
    }
}

impl From<ArrivalRow> for (f64, i64, i64) {
    fn from(r: ArrivalRow) -> Self {
        (r.t, r.uncached, r.cached)
    }
}

/// A next-aware result lag sample (JSON `[f, y]`): features of
/// pass k + 1 and `done - launch(k + 1)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "(Features, f64)", into = "(Features, f64)")]
pub struct LagSample {
    pub f: Features,
    pub y: f64,
}

impl From<(Features, f64)> for LagSample {
    fn from((f, y): (Features, f64)) -> Self {
        Self { f, y }
    }
}

impl From<LagSample> for (Features, f64) {
    fn from(s: LagSample) -> Self {
        (s.f, s.y)
    }
}

/// A pass of the calibration window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalIter {
    /// `forward_iter` index.
    pub k: i64,
    #[serde(default)]
    pub launch: Option<f64>,
    #[serde(default)]
    pub gpu_ms: Option<f64>,
    /// The pass was launched after an idle gap.
    #[serde(default)]
    pub after_idle: bool,
    /// Requests whose last chunk is in the pass.
    pub fin: f64,
    pub f: Features,
}

/// A request done in the calibration window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalReq {
    /// Passes of the request (`forward_iter` indices, in order).
    pub its: Vec<i64>,
    #[serde(default)]
    pub recv: Option<f64>,
    /// Waiting-queue entry.
    #[serde(default)]
    pub entry: Option<f64>,
    #[serde(default)]
    pub first_launch: Option<f64>,
    /// The first pass was launched after an idle gap.
    #[serde(default)]
    pub after_idle: bool,
    pub done: f64,
    #[serde(default)]
    pub last_gpu_end: Option<f64>,
}

/// GPU chain: `forward_iter` -> (GPU start, GPU end).
pub type GpuChain = HashMap<i64, (Option<f64>, Option<f64>)>;

/// The GPU chain from `[k, start, end]` rows (a later row of the same k
/// replaces an earlier one, as the reference's dict).
pub fn gpu_chain_from_rows(rows: &[(i64, Option<f64>, Option<f64>)]) -> GpuChain {
    rows.iter().map(|&(k, s, e)| (k, (s, e))).collect()
}

/// A row of the launch model anchored on the previous GPU start.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnchoredPair {
    pub x: Vec<f64>,
    pub y: f64,
    /// Time key: launch of pass k + 1.
    pub t: f64,
}

/// Where a G1 launch fit came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LaunchSource {
    Window,
    Calibration,
    Constant,
    /// `--forecast-fit`: offline warmup constants.
    Frozen,
}

/// A G1 launch fit: coefficients, residual quantiles and source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaunchFit {
    pub coef: Vec<f64>,
    pub resid_q: Option<Vec<f64>>,
    pub source: LaunchSource,
}

/// Warmup constants of one P worker (reference `calibrate`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    /// Per-class GPU coefficients (ms) from the passes with GPU times.
    pub coef: GpuClasses,
    pub floor_ms: f64,
    /// Launch line `[a, b]` over consecutive busy pairs; None only when its
    /// fit was singular.
    pub launch_coef: Option<Vec<f64>>,
    /// Start-anchored launch model (6 coefficients).
    pub launch2_coef: Vec<f64>,
    pub idle_delay_s: f64,
    pub lag_s: f64,
    pub lag_none_s: f64,
    pub lag_next: [f64; 2],
    pub phi_s: f64,
    pub boot_s: f64,
}

/// The warmup constants that enter the forecast parameters (the router calibrations replace `lag_none_s` and `lag_next`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Constants {
    pub idle_delay_s: f64,
    pub phi_s: f64,
    pub boot_s: f64,
    pub lag_none_s: f64,
    pub lag_next: [f64; 2],
}

impl From<&Calibration> for Constants {
    fn from(c: &Calibration) -> Self {
        Self {
            idle_delay_s: c.idle_delay_s,
            phi_s: c.phi_s,
            boot_s: c.boot_s,
            lag_none_s: c.lag_none_s,
            lag_next: c.lag_next,
        }
    }
}

/// Rows (sorted by `key`) with `lo_t <= key < hi_t` (`bisect_left`
/// on both ends).
pub fn window<T>(rows: &[T], lo_t: f64, hi_t: f64, key: impl Fn(&T) -> f64) -> &[T] {
    let lo = rows.partition_point(|r| key(r) < lo_t);
    let hi = rows.partition_point(|r| key(r) < hi_t);
    if hi > lo {
        &rows[lo..hi]
    } else {
        &rows[..0]
    }
}

/// Start of the refit step of a decision: `floor(d / step_s) * step_s`.
pub fn step_of(d: f64, step_s: f64) -> f64 {
    (d / step_s).floor() * step_s
}

/// GPU ms model per size class: [`nonneg_fit`] on the class's
/// rows (`small`: `f[1] <= graph_tokens`) when it has at least `min_n`.
pub fn fit_gpu_classes(rows: &[GpuSample], graph_tokens: i64, min_n: usize) -> GpuClasses {
    let g = graph_tokens as f64;
    let fit = |small: bool| -> Option<Vec<f64>> {
        let sel: Vec<&GpuSample> = rows
            .iter()
            .filter(|r| if small { r.f[1] <= g } else { r.f[1] > g })
            .collect();
        if sel.len() < min_n {
            return None;
        }
        let xs: Vec<&[f64]> = sel.iter().map(|r| &r.f[..]).collect();
        let ys: Vec<f64> = sel.iter().map(|r| r.y_ms).collect();
        nonneg_fit(&xs, &ys)
    };
    GpuClasses {
        small: fit(true),
        large: fit(false),
    }
}

/// GPU model whose class coefficients are the first existing fit in `chain`
/// (for example own P window, pooled window, own calibration window, then a
/// fallback), and the name that gave each class.
pub fn resolve_gpu(
    chain: &[(String, Option<GpuClasses>)],
    floor_ms: f64,
) -> (GpuModel, GpuSources) {
    let pick = |small: bool| -> (Option<Vec<f64>>, Option<String>) {
        for (name, fit) in chain {
            if let Some(fit) = fit {
                let c = if small { &fit.small } else { &fit.large };
                if let Some(c) = c {
                    return (Some(c.clone()), Some(name.clone()));
                }
            }
        }
        (None, None)
    };
    let (small, src_small) = pick(true);
    let (large, src_large) = pick(false);
    (
        GpuModel {
            small,
            large,
            floor_ms,
        },
        GpuSources {
            small: src_small,
            large: src_large,
        },
    )
}

/// Composition launch model from design rows: [`nonneg_fit`] and
/// the 19 residual quantiles when there are at least `min_n` rows.
pub fn fit_launch_rows(xy: &[DesignRow], min_n: usize) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
    if xy.len() < min_n {
        return (None, None);
    }
    let xs: Vec<&[f64]> = xy.iter().map(|r| r.x.as_slice()).collect();
    let ys: Vec<f64> = xy.iter().map(|r| r.y).collect();
    let c = match nonneg_fit(&xs, &ys) {
        Some(c) if !c.is_empty() => c,
        _ => return (None, None),
    };
    let mut rs: Vec<f64> = xy
        .iter()
        .map(|r| r.y - py_sum(c.iter().zip(&r.x).map(|(ci, xi)| ci * xi)))
        .collect();
    sort_py(&mut rs);
    let q = resid_quantiles(&rs);
    (Some(c), Some(q))
}

/// Design rows of the G1 launch model without GPU times (anchor = previous
/// launch): `x = [1, modelled GPU s of k, batch size of k,
/// finishing of k, 1 if k + 1 restores host KV, host tokens of k + 1 / 1e3]`.
pub fn launch_interval_design(
    rows: &[LaunchSample],
    gm: &GpuModel,
    graph_tokens: i64,
) -> Vec<DesignRow> {
    rows.iter()
        .map(|r| DesignRow {
            x: vec![
                1.0,
                gpu_s(gm, &r.f, graph_tokens),
                r.f[4],
                r.fin,
                if r.host_next > 0.0 { 1.0 } else { 0.0 },
                r.host_next / 1000.0,
            ],
            y: r.y,
        })
        .collect()
}

/// G1 launch fit: the window rows, else the calibration rows,
/// else the constant `[const_median or 0.1, 0, 0, 0, 0, 0]` without
/// quantiles.
pub fn fit_launch_intervals(
    rows: &[LaunchSample],
    gm: &GpuModel,
    fallback_rows: Option<&[LaunchSample]>,
    const_median: Option<f64>,
) -> LaunchFit {
    if let (Some(coef), resid_q) = fit_launch_rows(
        &launch_interval_design(rows, gm, GRAPH_TOKENS),
        LAUNCH_FIT_MIN,
    ) {
        return LaunchFit {
            coef,
            resid_q,
            source: LaunchSource::Window,
        };
    }
    if let Some(fb) = fallback_rows {
        if let (Some(coef), resid_q) = fit_launch_rows(
            &launch_interval_design(fb, gm, GRAPH_TOKENS),
            LAUNCH_FIT_MIN,
        ) {
            return LaunchFit {
                coef,
                resid_q,
                source: LaunchSource::Calibration,
            };
        }
    }
    LaunchFit {
        coef: vec![
            const_median.unwrap_or(DEFAULT_LAUNCH_INTERVAL_S),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        resid_q: None,
        source: LaunchSource::Constant,
    }
}

/// Receipt line: delay = a + b x input tokens (b >= 0), least
/// squares over the running sums of all samples, and the idle shift = the
/// median residual of the latest [`RX_SHIFT_SAMPLES`] samples. The caller
/// adds samples in their usable order (production rule `rx2`).
#[derive(Debug, Clone, Default)]
pub struct RxLine {
    n: u64,
    sx: f64,
    sy: f64,
    sxx: f64,
    sxy: f64,
    /// The latest samples (the reference keeps all; only these are read).
    recent: VecDeque<(f64, f64)>,
}

impl RxLine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a sample (`x` = input tokens, `y` = receipt - decision time).
    pub fn add(&mut self, x: f64, y: f64) {
        self.n += 1;
        self.sx += x;
        self.sy += y;
        self.sxx += x * x;
        self.sxy += x * y;
        if self.recent.len() == RX_SHIFT_SAMPLES {
            self.recent.pop_front();
        }
        self.recent.push_back((x, y));
    }

    /// Number of samples added.
    pub fn len(&self) -> u64 {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The line with at least [`RX_MIN`] samples and some spread of `x`.
    pub fn params(&self) -> Option<RxParams> {
        self.params_min(RX_MIN)
    }

    /// The line with at least `min_n` samples.
    pub fn params_min(&self, min_n: u64) -> Option<RxParams> {
        let n = self.n as f64;
        let den = n * self.sxx - self.sx * self.sx;
        if self.n >= min_n && den > 0.0 {
            let b = py_max(0.0, (n * self.sxy - self.sx * self.sy) / den);
            let a = (self.sy - b * self.sx) / n;
            let mut rs: Vec<f64> = self.recent.iter().map(|&(x, y)| y - (a + b * x)).collect();
            sort_py(&mut rs);
            let shift = *rs.get(rs.len() / 2)?;
            Some(RxParams { a, b, shift })
        } else {
            None
        }
    }
}

/// Arrival rates from the requests received in `[step - window_s, step)` in
/// receipt order: None below `min_n` requests or without uncached
/// tokens; `span = max(1, min(window_s, step - first receipt))`,
/// `lam = sum u / span`, `rate = n / span`, `unc = sum u / n`,
/// `pf = sum(u c + u^2 / 2) / sum u`.
pub fn arrival_rates(
    rows: &[ArrivalRow],
    step: f64,
    window_s: f64,
    min_n: usize,
) -> Option<Arrivals> {
    let su: i64 = rows.iter().map(|r| r.uncached).sum();
    if rows.len() < min_n || su <= 0 {
        return None;
    }
    let span = py_max(1.0, py_min(window_s, step - rows.first()?.t));
    let (n, suf) = (rows.len() as f64, su as f64);
    let pf = py_sum(
        rows.iter()
            .map(|r| (r.uncached * r.cached) as f64 + (r.uncached * r.uncached) as f64 / 2.0),
    ) / suf;
    Some(Arrivals {
        lam: suf / span,
        rate: n / span,
        unc: suf / n,
        pf,
    })
}

/// Rows of the launch model anchored on the previous GPU start (needs GPU
/// times): consecutive passes k, k + 1 with k + 1 not after an
/// idle gap and `0 <= launch(k + 1) - start(k) < y_max`;
/// `x = [1, GPU s of k, batch size of k, finishing of k, 1 if k + 1
/// restores host KV, host of k + 1 / 1e3]`, `y = launch(k + 1) - start(k)`.
pub fn start_anchored_pairs(
    passes: &[CalIter],
    gpu_chain: &GpuChain,
    y_max: f64,
) -> Vec<AnchoredPair> {
    let mut rows = Vec::new();
    for w in passes.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        if b.k != a.k + 1 || b.after_idle {
            continue;
        }
        let sa = gpu_chain.get(&a.k).and_then(|c| c.0);
        if let (Some(sa), Some(bl), Some(ms)) = (finite(sa), finite(b.launch), finite(a.gpu_ms)) {
            let y = bl - sa;
            if (0.0..y_max).contains(&y) {
                let host = b.f[3];
                rows.push(AnchoredPair {
                    x: vec![
                        1.0,
                        ms / 1000.0,
                        a.f[4],
                        a.fin,
                        if host > 0.0 { 1.0 } else { 0.0 },
                        host / 1000.0,
                    ],
                    y,
                    t: bl,
                });
            }
        }
    }
    rows
}

/// Warmup constants of one P worker (port of
/// `estate_replay_kv.Model.__init__`): GPU model per class and floor,
/// launch models, idle launch delay, result lags, loop-top cut `phi` and
/// receipt-to-entry `boot`. Errors where the reference raises: a request
/// without passes, or a singular launch line with no start-anchored model.
pub fn calibrate(
    cal_iters: &[CalIter],
    cal_reqs: &[CalReq],
    gpu_chain: &GpuChain,
    graph_tokens: i64,
) -> Result<Calibration, ForecastError> {
    let g = graph_tokens as f64;
    let class_fit = |small: bool| -> Option<Vec<f64>> {
        let mut xs: Vec<&[f64]> = Vec::new();
        let mut ys: Vec<f64> = Vec::new();
        for it in cal_iters {
            let sel = if small { it.f[1] <= g } else { it.f[1] > g };
            if let Some(ms) = finite(it.gpu_ms) {
                if sel && ms > 0.0 {
                    xs.push(&it.f[..]);
                    ys.push(ms);
                }
            }
        }
        if xs.len() >= GPU_FIT_MIN {
            nonneg_fit(&xs, &ys)
        } else {
            None
        }
    };
    let coef = GpuClasses {
        small: class_fit(true),
        large: class_fit(false),
    };
    let gpu_ms: Vec<f64> = cal_iters
        .iter()
        .filter_map(|it| finite(it.gpu_ms))
        .collect();
    let floor_ms = truthy_or(quantile(&gpu_ms, 0.01), DEFAULT_FLOOR_MS);

    // Launch line over consecutive busy pairs: launch(k + 1) - start(k).
    let mut line_x: Vec<[f64; 2]> = Vec::new();
    let mut line_y: Vec<f64> = Vec::new();
    for w in cal_iters.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        if b.k != a.k + 1 || b.after_idle {
            continue;
        }
        let sa = gpu_chain.get(&a.k).and_then(|c| c.0);
        if let (Some(sa), Some(bl), Some(ms)) = (finite(sa), finite(b.launch), finite(a.gpu_ms)) {
            line_x.push([1.0, ms / 1000.0]);
            line_y.push(bl - sa);
        }
    }
    let launch_coef = if line_x.len() >= LAUNCH_LINE_MIN {
        lstsq(&line_x, &line_y, RIDGE)
    } else {
        Some(DEFAULT_LAUNCH_LINE.to_vec())
    };
    let pairs = start_anchored_pairs(cal_iters, gpu_chain, LAUNCH_Y_MAX_S);
    let lc2 = if pairs.len() >= LAUNCH_FIT_MIN {
        let xs: Vec<&[f64]> = pairs.iter().map(|p| p.x.as_slice()).collect();
        let ys: Vec<f64> = pairs.iter().map(|p| p.y).collect();
        nonneg_fit(&xs, &ys)
    } else {
        None
    };
    let launch2_coef = match lc2 {
        Some(c) if !c.is_empty() => c,
        _ => {
            let lc = launch_coef
                .as_ref()
                .ok_or(ForecastError::Singular("calibrate: launch line"))?;
            vec![lc[0], lc[1], 0.0, 0.0, 0.0, 0.0]
        }
    };

    // Idle launch delay: first launch - max(receipt, entry) after an idle gap.
    let idle: Vec<f64> = cal_reqs
        .iter()
        .filter_map(|r| {
            if !r.after_idle {
                return None;
            }
            let first = finite(r.first_launch)?;
            let start = py_max(finite(r.recv)?, r.entry?);
            (first >= start).then_some(first - start)
        })
        .collect();
    let idle_delay_s = if idle.is_empty() {
        DEFAULT_IDLE_DELAY_S
    } else {
        quantile(&idle, 0.5).unwrap_or(DEFAULT_IDLE_DELAY_S)
    };

    // Result lags: done - GPU end of the last pass.
    let lags: Vec<f64> = cal_reqs
        .iter()
        .filter_map(|r| {
            let e = finite(r.last_gpu_end)?;
            (r.done >= e).then_some(r.done - e)
        })
        .collect();
    let lag = if lags.is_empty() {
        DEFAULT_LAG_S
    } else {
        quantile(&lags, 0.5).unwrap_or(DEFAULT_LAG_S)
    };
    // The last iteration of each k wins, as the reference's dict.
    let by_k: HashMap<i64, &CalIter> = cal_iters.iter().map(|it| (it.k, it)).collect();
    let mut next_x: Vec<[f64; 2]> = Vec::new();
    let mut next_y: Vec<f64> = Vec::new();
    let mut none_lags: Vec<f64> = Vec::new();
    for r in cal_reqs {
        let k = *r
            .its
            .last()
            .ok_or(ForecastError::Input("calibrate: request without passes"))?;
        let Some(e) = finite(gpu_chain.get(&k).and_then(|c| c.1)) else {
            continue;
        };
        let next = by_k
            .get(&(k + 1))
            .and_then(|b| match (finite(b.launch), finite(b.gpu_ms)) {
                (Some(launch), Some(ms)) if launch < r.done => Some((launch, ms)),
                _ => None,
            });
        if let Some((launch, ms)) = next {
            next_x.push([1.0, ms / 1000.0]);
            next_y.push(r.done - launch);
        } else if r.done >= e {
            none_lags.push(r.done - e);
        }
    }
    let lag_none_s = if none_lags.is_empty() {
        lag
    } else {
        quantile(&none_lags, 0.5).unwrap_or(lag)
    };
    let lag_next = if next_x.len() >= GPU_FIT_MIN {
        lstsq(&next_x, &next_y, RIDGE)
    } else {
        None
    };
    let lag_next = match lag_next {
        Some(c) => [c[0], c[1]],
        None => [lag, 0.0],
    };

    // phi: first launch at or after the receipt - receipt (busy requests).
    let mut launches: Vec<f64> = cal_iters
        .iter()
        .filter_map(|it| finite(it.launch))
        .collect();
    sort_py(&mut launches);
    let mut phis: Vec<f64> = Vec::new();
    for r in cal_reqs {
        let Some(recv) = finite(r.recv) else {
            continue;
        };
        let i = launches.partition_point(|&l| l < recv);
        if i < launches.len() && !r.after_idle {
            phis.push(launches[i] - recv);
        }
    }
    let phi_s = if phis.is_empty() {
        0.0
    } else {
        quantile(&phis, 0.5).unwrap_or(0.0)
    };
    let boots: Vec<f64> = cal_reqs
        .iter()
        .filter_map(|r| {
            let (entry, recv) = (finite(r.entry)?, finite(r.recv)?);
            (entry >= recv).then_some(entry - recv)
        })
        .collect();
    let boot_s = if boots.is_empty() {
        0.0
    } else {
        quantile(&boots, 0.5).unwrap_or(0.0)
    };
    Ok(Calibration {
        coef,
        floor_ms,
        launch_coef,
        launch2_coef,
        idle_delay_s,
        lag_s: lag,
        lag_none_s,
        lag_next,
        phi_s,
        boot_s,
    })
}

/// Router calibration without GPU times: GPU model floor = the
/// 0.01 quantile of the service samples (ms), 1.0 without samples.
pub fn router_floor_ms(service_ms: &[f64]) -> f64 {
    truthy_or(quantile(service_ms, 0.01), DEFAULT_FLOOR_MS)
}

/// Router calibration of the next-aware result lag:
/// `done = launch(k + 1) + a + b x modelled GPU s of pass k + 1`, least
/// squares with at least [`GPU_FIT_MIN`] samples, else `[median y, 0]`
/// (`[0, 0]` without samples).
pub fn router_lag_next(samples: &[LagSample], gm: &GpuModel, graph_tokens: i64) -> [f64; 2] {
    let ys: Vec<f64> = samples.iter().map(|s| s.y).collect();
    if samples.len() >= GPU_FIT_MIN {
        let xs: Vec<[f64; 2]> = samples
            .iter()
            .map(|s| [1.0, gpu_s(gm, &s.f, graph_tokens)])
            .collect();
        if let Some(c) = lstsq(&xs, &ys, RIDGE) {
            return [c[0], c[1]];
        }
    }
    let median = if ys.is_empty() {
        0.0
    } else {
        quantile(&ys, 0.5).unwrap_or(0.0)
    };
    [median, 0.0]
}

/// Assemble the forecast parameters (reference `make_params`). An empty
/// `launch_resid_q` becomes None.
#[allow(clippy::too_many_arguments)]
pub fn make_params(
    cal: &Constants,
    gpu_idle: GpuModel,
    gpu_busy: GpuModel,
    launch_coef: Vec<f64>,
    launch_resid_q: Option<Vec<f64>>,
    launch_anchor: LaunchAnchor,
    rx: Option<RxParams>,
    arrivals: Option<Arrivals>,
    cap_tokens: f64,
) -> Params {
    Params {
        schema: Some(SCHEMA.to_string()),
        chunk_tokens: Some(CHUNK_TOKENS),
        page_tokens: Some(PAGE_TOKENS),
        graph_tokens: Some(GRAPH_TOKENS),
        dispatch_s: Some(DISPATCH_S),
        p_after: Some(P_AFTER),
        cap_tokens,
        gpu_idle,
        gpu_busy,
        launch_coef,
        launch_resid_q: launch_resid_q.filter(|q| !q.is_empty()),
        launch_anchor: Some(launch_anchor),
        idle_delay_s: cal.idle_delay_s,
        phi_s: cal.phi_s,
        boot_s: cal.boot_s,
        lag_none_s: cal.lag_none_s,
        lag_next: cal.lag_next,
        rx,
        arrivals,
        next_pass_lag: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecast::model::features;
    use crate::forecast::{ARRIVAL_MIN, STEP_S, WINDOW_S};

    fn gm() -> GpuModel {
        GpuModel {
            small: Some(vec![50.0, 0.03, 2.0, 0.0, 0.0]),
            large: Some(vec![50.0, 0.03, 2.0, 0.0, 0.0]),
            floor_ms: 1.0,
        }
    }

    #[test]
    fn window_is_left_closed_right_open() {
        let rows = [1.0, 2.0, 3.0, 4.0, 5.0];
        let key = |r: &f64| *r;
        assert_eq!(window(&rows, 2.0, 4.0, key), &[2.0, 3.0]);
        assert_eq!(window(&rows, 0.0, 1.0, key), &[] as &[f64]);
        assert_eq!(window(&rows, 5.0, 5.0, key), &[] as &[f64]);
        assert_eq!(window(&rows, 4.0, 2.0, key), &[] as &[f64]);
        assert_eq!(window(&rows, f64::NEG_INFINITY, 3.0, key), &[1.0, 2.0]);
        assert_eq!(window(&rows, 4.5, 99.0, key), &[5.0]);
    }

    #[test]
    fn step_of_floors_to_the_grid() {
        assert_eq!(step_of(125.0, STEP_S), 120.0);
        assert_eq!(step_of(60.0, STEP_S), 60.0);
        assert_eq!(step_of(59.999, STEP_S), 0.0);
        assert_eq!(step_of(-1.0, STEP_S), -60.0);
    }

    #[test]
    fn gpu_classes_recover_a_known_model() {
        // Reference self-test rows: y = 5 + 0.02 n + 3 x pairs / 1e6.
        let rows: Vec<GpuSample> = (100..2100)
            .step_by(50)
            .map(|n: i64| {
                let q = ((n * 7919) % 1000) as f64;
                GpuSample {
                    f: features(n as f64, q * 1e3, 0.0, 1.0),
                    y_ms: 5.0 + 0.02 * n as f64 + 3.0 * q * 1e-3,
                }
            })
            .collect();
        let c = fit_gpu_classes(&rows, GRAPH_TOKENS, GPU_FIT_MIN);
        // n = 2050 is the only large row: below the minimum.
        assert_eq!(c.large, None);
        assert_eq!(
            c.small,
            Some(vec![
                5.0001157875513265,
                0.01999991252090918,
                2.9999374819807065,
                0.0,
                0.0
            ])
        );
    }

    #[test]
    fn resolve_gpu_takes_the_first_fit_per_class() {
        let chain = vec![
            ("own".to_string(), Some(GpuClasses::default())),
            (
                "pooled".to_string(),
                Some(GpuClasses {
                    small: None,
                    large: Some(vec![60.0, 0.04, 2.0, 0.0, 0.0]),
                }),
            ),
            ("cal".to_string(), None),
            (
                "own_all".to_string(),
                Some(GpuClasses {
                    small: Some(vec![50.0, 0.03, 2.0, 0.0, 1.0]),
                    large: None,
                }),
            ),
            (
                "pooled_all".to_string(),
                Some(GpuClasses {
                    small: Some(vec![1.0; 5]),
                    large: Some(vec![2.0; 5]),
                }),
            ),
        ];
        let (gm, src) = resolve_gpu(&chain, 4.0);
        assert_eq!(gm.small, Some(vec![50.0, 0.03, 2.0, 0.0, 1.0]));
        assert_eq!(gm.large, Some(vec![60.0, 0.04, 2.0, 0.0, 0.0]));
        assert_eq!(gm.floor_ms, 4.0);
        assert_eq!(src.small.as_deref(), Some("own_all"));
        assert_eq!(src.large.as_deref(), Some("pooled"));
        let (none, src) = resolve_gpu(&chain[..1], 1.0);
        assert_eq!((none.small, none.large), (None, None));
        assert_eq!(src, GpuSources::default());
    }

    fn launch_samples(n: usize) -> Vec<LaunchSample> {
        (0..n)
            .map(|i| {
                let nt = 500.0 + 37.0 * i as f64;
                let hb = if i % 3 == 0 { 0.0 } else { 1000.0 * i as f64 };
                let f = features(nt, nt * 2000.0, 0.0, 1.0 + (i % 4) as f64);
                let y = 0.02
                    + 0.9 * gpu_s(&gm(), &f, GRAPH_TOKENS)
                    + 0.002 * (i % 3) as f64
                    + hb * 2e-7;
                LaunchSample {
                    t: 1000.0 + i as f64,
                    f,
                    fin: (i % 3) as f64,
                    host_next: hb,
                    y,
                }
            })
            .collect()
    }

    #[test]
    fn launch_interval_fallbacks() {
        let g = gm();
        let win = launch_samples(40);
        let fit = fit_launch_intervals(&win, &g, None, None);
        assert_eq!(fit.source, LaunchSource::Window);
        assert_eq!(fit.coef.len(), 6);
        assert_eq!(fit.resid_q.as_ref().map(Vec::len), Some(19));
        let cal = fit_launch_intervals(&win[..29], &g, Some(&win), Some(0.3));
        assert_eq!(cal.source, LaunchSource::Calibration);
        assert_eq!(cal.coef, fit.coef);
        let cst = fit_launch_intervals(&win[..29], &g, Some(&win[..5]), Some(0.3));
        assert_eq!(cst.source, LaunchSource::Constant);
        assert_eq!(cst.coef, [0.3, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(cst.resid_q, None);
        let dflt = fit_launch_intervals(&win[..3], &g, None, None);
        assert_eq!(
            dflt.coef,
            [DEFAULT_LAUNCH_INTERVAL_S, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
        // The design row of one sample.
        let d = launch_interval_design(&win[1..2], &g, GRAPH_TOKENS);
        let s = &win[1];
        assert_eq!(
            d[0].x,
            [
                1.0,
                gpu_s(&g, &s.f, GRAPH_TOKENS),
                s.f[4],
                s.fin,
                1.0,
                s.host_next / 1000.0
            ]
        );
        assert_eq!(d[0].y, s.y);
        assert_eq!(fit_launch_rows(&d, LAUNCH_FIT_MIN), (None, None));
    }

    #[test]
    fn receipt_line_matches_the_reference() {
        // Reference self-test: rx.add(1000. * i, 0.04 + 1e-6 * 1000. * i).
        let mut rx = RxLine::new();
        for i in 0..40 {
            let fi = f64::from(i);
            rx.add(1000.0 * fi, 0.04 + 1e-6 * 1000.0 * fi);
        }
        assert_eq!(rx.len(), 40);
        let p = rx.params().unwrap();
        assert_eq!(
            (p.a, p.b, p.shift),
            (
                0.04000000000000003,
                9.99999999999999e-07,
                -6.938893903907228e-18
            )
        );
        // Too few samples, and no spread of x.
        let mut few = RxLine::new();
        for i in 0..29 {
            few.add(f64::from(i), 0.04);
        }
        assert_eq!(few.params(), None);
        let mut flat = RxLine::new();
        for i in 0..40 {
            flat.add(5000.0, 0.04 + 0.001 * f64::from(i));
        }
        assert_eq!(flat.params(), None);
        assert!(RxLine::new().is_empty());
    }

    #[test]
    fn arrival_rates_hand_case() {
        let rows: Vec<ArrivalRow> = (0..30)
            .map(|j| ArrivalRow {
                t: 900.0 + 10.0 * f64::from(j),
                uncached: 100,
                cached: 1000,
            })
            .collect();
        // span = max(1, min(600, 1320 - 900)) = 420.
        let a = arrival_rates(&rows, 1320.0, WINDOW_S, ARRIVAL_MIN).unwrap();
        assert_eq!(
            a,
            Arrivals {
                lam: 3000.0 / 420.0,
                rate: 30.0 / 420.0,
                unc: 100.0,
                pf: 1050.0
            }
        );
        assert_eq!(
            arrival_rates(&rows[..29], 1320.0, WINDOW_S, ARRIVAL_MIN),
            None
        );
        let cold: Vec<ArrivalRow> = rows
            .iter()
            .map(|r| ArrivalRow { uncached: 0, ..*r })
            .collect();
        assert_eq!(arrival_rates(&cold, 1320.0, WINDOW_S, ARRIVAL_MIN), None);
        // A first receipt less than 1 s before the step: span 1.
        let late: Vec<ArrivalRow> = rows
            .iter()
            .map(|r| ArrivalRow { t: 1319.5, ..*r })
            .collect();
        assert_eq!(
            arrival_rates(&late, 1320.0, WINDOW_S, ARRIVAL_MIN)
                .unwrap()
                .lam,
            3000.0
        );
    }

    fn cal_case() -> (Vec<CalIter>, Vec<CalReq>, GpuChain) {
        let iters = vec![
            CalIter {
                k: 1,
                launch: Some(10.0),
                gpu_ms: Some(5.0),
                after_idle: true,
                fin: 1.0,
                f: [1.0, 100.0, 0.005, 0.0, 1.0],
            },
            CalIter {
                k: 2,
                launch: Some(10.006),
                gpu_ms: Some(4.0),
                after_idle: false,
                fin: 1.0,
                f: [1.0, 200.0, 0.02, 3000.0, 2.0],
            },
        ];
        let reqs = vec![
            CalReq {
                its: vec![1],
                recv: Some(9.99),
                entry: Some(9.995),
                first_launch: Some(10.0),
                after_idle: true,
                done: 10.012,
                last_gpu_end: Some(10.005),
            },
            CalReq {
                its: vec![2],
                recv: Some(10.001),
                entry: Some(10.0015),
                first_launch: Some(10.006),
                after_idle: false,
                done: 10.02,
                last_gpu_end: Some(10.010),
            },
        ];
        let chain = gpu_chain_from_rows(&[
            (1, Some(10.0), Some(10.005)),
            (2, Some(10.006), Some(10.010)),
        ]);
        (iters, reqs, chain)
    }

    #[test]
    fn calibrate_matches_the_reference_on_a_hand_case() {
        let (iters, reqs, chain) = cal_case();
        let c = calibrate(&iters, &reqs, &chain, GRAPH_TOKENS).unwrap();
        assert_eq!(c.coef, GpuClasses::default());
        assert_eq!(c.floor_ms, 4.01);
        assert_eq!(c.launch_coef, Some(vec![0.06, 0.75]));
        assert_eq!(c.launch2_coef, [0.06, 0.75, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(c.idle_delay_s, 0.005000000000000782);
        assert_eq!(c.lag_s, 0.00849999999999973);
        assert_eq!(c.lag_none_s, 0.009999999999999787);
        assert_eq!(c.lag_next, [0.00849999999999973, 0.0]);
        assert_eq!(c.phi_s, 0.005000000000000782);
        assert_eq!(c.boot_s, 0.002749999999999808);
        let pairs = start_anchored_pairs(&iters, &chain, LAUNCH_Y_MAX_S);
        assert_eq!(
            pairs,
            [AnchoredPair {
                x: vec![1.0, 0.005, 1.0, 1.0, 1.0, 3.0],
                y: 0.006000000000000227,
                t: 10.006
            }]
        );
        // A request without passes is an error, as in the reference.
        let mut bad = reqs.clone();
        bad[0].its.clear();
        assert!(calibrate(&iters, &bad, &chain, GRAPH_TOKENS).is_err());
    }

    #[test]
    fn calibrate_defaults_without_samples() {
        let c = calibrate(&[], &[], &GpuChain::new(), GRAPH_TOKENS).unwrap();
        assert_eq!(c.coef, GpuClasses::default());
        assert_eq!(c.floor_ms, DEFAULT_FLOOR_MS);
        assert_eq!(c.launch_coef, Some(DEFAULT_LAUNCH_LINE.to_vec()));
        assert_eq!(c.launch2_coef, [0.06, 0.75, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!((c.idle_delay_s, c.lag_s, c.lag_none_s), (0.03, 0.06, 0.06));
        assert_eq!(c.lag_next, [0.06, 0.0]);
        assert_eq!((c.phi_s, c.boot_s), (0.0, 0.0));
    }

    #[test]
    fn router_lag_next_and_floor() {
        let g = gm();
        let samples: Vec<LagSample> = (100..3000)
            .step_by(100)
            .map(|n: i64| {
                let f = features(n as f64, (n * 10) as f64, 0.0, 1.0);
                LagSample {
                    f,
                    y: 0.01 + gpu_s(&g, &f, GRAPH_TOKENS),
                }
            })
            .collect();
        assert_eq!(
            router_lag_next(&samples, &g, GRAPH_TOKENS),
            [0.010001302224371065, 0.9999861914587678]
        );
        // Fewer than 20 samples: [median y, 0]; none: [0, 0].
        let few = &samples[..3];
        let mut ys: Vec<f64> = few.iter().map(|s| s.y).collect();
        sort_py(&mut ys);
        assert_eq!(router_lag_next(few, &g, GRAPH_TOKENS), [ys[1], 0.0]);
        assert_eq!(router_lag_next(&[], &g, GRAPH_TOKENS), [0.0, 0.0]);
        assert_eq!(router_floor_ms(&[]), 1.0);
        assert_eq!(router_floor_ms(&[0.0, 0.0]), 1.0);
        assert_eq!(router_floor_ms(&[4.0, 5.0]), 4.01);
    }

    #[test]
    fn make_params_matches_the_reference() {
        let cal = Constants {
            idle_delay_s: 0.03,
            phi_s: 0.01,
            boot_s: 0.005,
            lag_none_s: 0.02,
            lag_next: [0.05, 0.5],
        };
        let p = make_params(
            &cal,
            gm(),
            gm(),
            vec![0.02, 0.5, 0.0, 0.0, 0.0, 0.0],
            Some(vec![]),
            LaunchAnchor::Launch,
            None,
            None,
            1e9,
        );
        let want = serde_json::json!({"schema": "forecast.v2", "chunk_tokens": 8192, "page_tokens": 64,
            "graph_tokens": 2048, "dispatch_s": 0.044, "p_after": 0.2, "cap_tokens": 1000000000.0,
            "gpu_idle": {"small": [50.0, 0.03, 2.0, 0.0, 0.0], "large": [50.0, 0.03, 2.0, 0.0, 0.0], "floor_ms": 1.0},
            "gpu_busy": {"small": [50.0, 0.03, 2.0, 0.0, 0.0], "large": [50.0, 0.03, 2.0, 0.0, 0.0], "floor_ms": 1.0},
            "launch_coef": [0.02, 0.5, 0.0, 0.0, 0.0, 0.0], "launch_resid_q": null, "launch_anchor": "launch",
            "idle_delay_s": 0.03, "phi_s": 0.01, "boot_s": 0.005, "lag_none_s": 0.02, "lag_next": [0.05, 0.5],
            "rx": null, "arrivals": null});
        assert_eq!(serde_json::to_value(&p).unwrap(), want);
        let c = Calibration {
            coef: GpuClasses::default(),
            floor_ms: 1.0,
            launch_coef: None,
            launch2_coef: vec![],
            idle_delay_s: 0.03,
            lag_s: 0.06,
            lag_none_s: 0.02,
            lag_next: [0.05, 0.5],
            phi_s: 0.01,
            boot_s: 0.005,
        };
        assert_eq!(Constants::from(&c), cal);
    }

    #[test]
    fn samples_parse_from_the_golden_json_arrays() {
        let s: LaunchSample =
            serde_json::from_str("[1.5, [1.0, 2.0, 3.0, 4.0, 5.0], 1, 2000, 0.25]").unwrap();
        assert_eq!(
            (s.t, s.f[4], s.fin, s.host_next, s.y),
            (1.5, 5.0, 1.0, 2000.0, 0.25)
        );
        let a: ArrivalRow = serde_json::from_str("[900.0, 10, 20]").unwrap();
        assert_eq!(
            a,
            ArrivalRow {
                t: 900.0,
                uncached: 10,
                cached: 20
            }
        );
        let g: GpuSample = serde_json::from_str("[[1.0, 2.0, 3.0, 4.0, 5.0], 7.5]").unwrap();
        assert_eq!(g.y_ms, 7.5);
        assert_eq!(
            serde_json::to_string(&g).unwrap(),
            "[[1.0,2.0,3.0,4.0,5.0],7.5]"
        );
        let d: DesignRow = serde_json::from_str("[[1.0, 0.5], 0.25]").unwrap();
        assert_eq!(
            d,
            DesignRow {
                x: vec![1.0, 0.5],
                y: 0.25
            }
        );
        let chain: Vec<(i64, Option<f64>, Option<f64>)> =
            serde_json::from_str("[[3, 1.0, 2.0], [3, 5.0, null]]").unwrap();
        assert_eq!(
            gpu_chain_from_rows(&chain).get(&3),
            Some(&(Some(5.0), None))
        );
    }
}
