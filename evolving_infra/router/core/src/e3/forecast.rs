// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! The E3 forecast: inputs, parameters, the pass
//! simulator and the park-gate choice.
//!
//! All times are seconds on one monotonic clock (in the router: the router
//! clock). Token counts are integers.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::numeric::{ceil_page, py_max, py_min, py_sum};
use super::{
    E3Error, CHUNK_TOKENS, DISPATCH_S, GRAPH_TOKENS, MAX_PASSES, NO_MODEL_GPU_S, PAGE_TOKENS,
    P_AFTER, SCHEMA,
};

/// Pass features `[1, new tokens, attention pairs / 1e6, host restore
/// tokens, batch size]`.
pub type Features = [f64; 5];

/// A pass launched at or before `t0` whose result was not processed at `t0`
/// (`inflight[]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InflightPass {
    /// Launch time (s).
    pub launch: f64,
    pub new_tokens: i64,
    pub attention_pairs: i64,
    /// Host KV tokens restored for the first-chunk members (the ungated
    /// provenance).
    pub host_tokens: i64,
    pub batch_size: i64,
    /// Members whose last chunk is in this pass.
    pub finishing: i64,
}

/// The started (chunked) request of the P state (`partial`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Partial {
    pub rid: String,
    /// Tokens still to compute after every launched chunk (>= 1).
    pub remaining: i64,
    /// Cached + launched tokens.
    pub prefix: i64,
}

/// Where a queue row comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RowSource {
    /// A row of the P state; `ready` = when it can join a pass.
    Snapshot,
    /// A row the router sent before `d` that the state does not show;
    /// `decision_time` = the router's decision time of that row.
    Dispatched,
}

/// A row ahead of the target, in FCFS order (`queue[]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueueRow {
    pub rid: String,
    pub source: RowSource,
    /// Full input tokens.
    pub input: i64,
    /// Cached tokens; None = no published prefix and no router estimate: the
    /// row is cold (prefix 0, host 0, 12.1).
    #[serde(default)]
    pub prefix: Option<i64>,
    /// Part of the prefix restored from host memory; None = 0.
    #[serde(default)]
    pub host: Option<i64>,
    /// Snapshot rows: ready time; None or a value <= `t0` = ready at `t0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ready: Option<f64>,
    /// Dispatched rows: the router's decision time of the row (required).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision_time: Option<f64>,
}

/// The routed request (`target`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Target {
    pub rid: String,
    pub input: i64,
    /// Cached tokens as the router knows them.
    pub prefix: i64,
    /// Part of the prefix restored from host memory.
    pub host: i64,
}

/// Forecast input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Input {
    /// Decision time.
    pub d: f64,
    /// State time of the P state.
    pub t0: f64,
    /// In-flight passes in launch order.
    #[serde(default)]
    pub inflight: Vec<InflightPass>,
    #[serde(default)]
    pub partial: Option<Partial>,
    /// Rows ahead of the target in FCFS order.
    #[serde(default)]
    pub queue: Vec<QueueRow>,
    /// U at `t0` (late-D park gate); None skips the park-gate choice.
    #[serde(default)]
    pub unbound_tokens: Option<i64>,
    pub target: Target,
}

/// GPU model: coefficients (ms) per size class and a floor (ms).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuModel {
    #[serde(default)]
    pub small: Option<Vec<f64>>,
    #[serde(default)]
    pub large: Option<Vec<f64>>,
    pub floor_ms: f64,
}

/// Receipt line `max(0, a + b x input)` and its idle shift.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RxParams {
    pub a: f64,
    pub b: f64,
    pub shift: f64,
}

/// Expected arrivals after the target.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Arrivals {
    /// Uncached tokens per second.
    pub lam: f64,
    /// Requests per second.
    pub rate: f64,
    /// Uncached tokens per request.
    pub unc: f64,
    /// Attention pairs per uncached token.
    pub pf: f64,
}

/// Anchor of the busy launch model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LaunchAnchor {
    /// The previous pass's launch (G1, RFNG).
    Launch,
    /// The previous pass's GPU start (`gpul`).
    Start,
}

fn default_anchor() -> Option<LaunchAnchor> {
    Some(LaunchAnchor::Launch)
}

/// Forecast parameters. The optional fields fall back to the
/// module constants when missing or null, as in the reference (`_p`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Params {
    #[serde(default)]
    pub schema: Option<String>,
    #[serde(default)]
    pub chunk_tokens: Option<i64>,
    #[serde(default)]
    pub page_tokens: Option<i64>,
    #[serde(default)]
    pub graph_tokens: Option<i64>,
    #[serde(default)]
    pub dispatch_s: Option<f64>,
    #[serde(default)]
    pub p_after: Option<f64>,
    /// 0.35 x the KV token pool (late-D park gate).
    pub cap_tokens: f64,
    pub gpu_idle: GpuModel,
    pub gpu_busy: GpuModel,
    /// Launch model `[c0 s, c1 s/s, c2 s, c3 s, c4 s, c5 s/1e3 tok]` (+ an
    /// optional c6 s per first chunk).
    pub launch_coef: Vec<f64>,
    /// 19 residual quantiles of the launch fit, or None.
    #[serde(default)]
    pub launch_resid_q: Option<Vec<f64>>,
    /// Missing = `launch`; null = not `launch` (reference:
    /// `params.get('launch_anchor', 'launch') == 'launch'`).
    #[serde(default = "default_anchor")]
    pub launch_anchor: Option<LaunchAnchor>,
    pub idle_delay_s: f64,
    pub phi_s: f64,
    pub boot_s: f64,
    pub lag_none_s: f64,
    /// Next-aware result lag `[a s, b s/s]`.
    pub lag_next: [f64; 2],
    #[serde(default)]
    pub rx: Option<RxParams>,
    #[serde(default)]
    pub arrivals: Option<Arrivals>,
    /// Result of a target whose last pass leaves no known work on a busy P.
    /// Off (the reference): the bare result lag `lag_none_s`. On
    /// (`--trade-e3-next-pass-lag`, fitted on Kimi-K2.6): the
    /// expected arrivals form the next pass, and the target's result is
    /// processed after that pass launches (`lag_next`), as on an engine that
    /// processes a pass's results after launching the next one.
    #[serde(default, skip_serializing_if = "is_false")]
    pub next_pass_lag: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl Params {
    pub fn chunk(&self) -> i64 {
        self.chunk_tokens.unwrap_or(CHUNK_TOKENS)
    }

    pub fn page(&self) -> i64 {
        self.page_tokens.unwrap_or(PAGE_TOKENS)
    }

    pub fn graph(&self) -> i64 {
        self.graph_tokens.unwrap_or(GRAPH_TOKENS)
    }

    pub fn dispatch(&self) -> f64 {
        self.dispatch_s.unwrap_or(DISPATCH_S)
    }

    pub fn p_after_share(&self) -> f64 {
        self.p_after.unwrap_or(P_AFTER)
    }

    pub fn anchor_is_launch(&self) -> bool {
        self.launch_anchor == Some(LaunchAnchor::Launch)
    }

    /// Parameters for which the reference raises an exception: a page size
    /// below 1 (division by zero) and fewer than 6 launch coefficients.
    pub fn check(&self) -> Result<(), E3Error> {
        if self.page() < 1 {
            return Err(E3Error::Params("page_tokens must be at least 1"));
        }
        if self.launch_coef.len() < 6 {
            return Err(E3Error::Params("launch_coef needs at least 6 coefficients"));
        }
        Ok(())
    }
}

/// Which simulation gave `pred_s`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ForecastPath {
    Idle,
    Full,
    After,
}

/// Forecast output (schema `router-e3.v2`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Output {
    pub schema: String,
    /// Seconds from `d` to the target's completion; None = no forecast.
    pub pred_s: Option<f64>,
    /// No in-flight pass, no partial request and no queue row.
    pub idle: bool,
    pub n_inflight: usize,
    pub partial: bool,
    pub n_queue: usize,
    pub n_dispatched: usize,
    /// Queue rows without a prefix (taken as cold).
    pub n_prefix_unknown: usize,
    /// Uncached tokens of the queue rows and the partial request.
    pub tokens_ahead: i64,
    pub x_uncached: i64,
    pub x_host: i64,
    /// Target receipt delay + `boot_s` (= x_arrival - d).
    pub receipt_s: f64,
    pub path: ForecastPath,
    pub f_full_s: Option<f64>,
    pub f_after_s: Option<f64>,
    pub n_held: usize,
    pub x_held: bool,
    pub u_tokens: Option<i64>,
    /// GPU start of the target's first pass - d.
    pub wait_s: Option<f64>,
    /// Passes that contain the target's chunks.
    pub x_passes: u64,
    /// Simulated passes before the target's first chunk.
    pub passes_before: u64,
}

/// Pass features.
pub fn features(n: f64, pairs: f64, host: f64, bs: f64) -> Features {
    [1.0, n, pairs / 1e6, host, bs]
}

/// Modelled GPU seconds of a pass: class `small` when
/// `f[1] <= graph_tokens`, else `large`; a missing (or empty) class falls
/// back to `large`, then to `small`; without coefficients 0.3 s; else
/// `max(floor_ms, sum_i c_i f_i) / 1000`.
pub fn gpu_s(gm: &GpuModel, f: &Features, graph_tokens: i64) -> f64 {
    let nonempty = |c: &Option<Vec<f64>>| c.as_ref().is_some_and(|v| !v.is_empty());
    let primary = if f[1] <= graph_tokens as f64 {
        &gm.small
    } else {
        &gm.large
    };
    // Python `gm[cls] or gm['large'] or gm['small']`.
    let c = if nonempty(primary) {
        primary
    } else if nonempty(&gm.large) {
        &gm.large
    } else {
        &gm.small
    };
    match c {
        None => NO_MODEL_GPU_S,
        Some(c) => {
            py_max(
                gm.floor_ms,
                py_sum(c.iter().zip(f.iter()).map(|(a, b)| a * b)),
            ) / 1000.0
        }
    }
}

/// Receipt delays `(line, idle)` of a request with `n_input` tokens:
/// `line = max(0, a + b x input)`, `idle = max(0, line + shift)`;
/// both `dispatch_s` without a receipt line.
pub fn receipt_delays(params: &Params, n_input: i64) -> (f64, f64) {
    match &params.rx {
        Some(rx) => {
            let delay = py_max(0.0, rx.a + rx.b * n_input as f64);
            (delay, py_max(0.0, delay + rx.shift))
        }
        None => {
            let ds = params.dispatch();
            (ds, ds)
        }
    }
}

/// Time at which a queue row can join a pass, before the loop-top cut
/// : dispatched rows `decision_time + rxd(input) + boot_s`;
/// snapshot rows their `ready` (None = ready at or before `t0`).
pub fn row_ready(row: &QueueRow, params: &Params) -> Result<Option<f64>, E3Error> {
    match row.source {
        RowSource::Dispatched => {
            let decision_time = row
                .decision_time
                .ok_or(E3Error::Input("dispatched queue row without decision_time"))?;
            let rxd = match &params.rx {
                Some(rx) => py_max(0.0, rx.a + rx.b * row.input as f64),
                None => params.dispatch(),
            };
            Ok(Some(decision_time + rxd + params.boot_s))
        }
        RowSource::Snapshot => Ok(row.ready),
    }
}

/// A work row of the simulation.
#[derive(Debug, Clone, PartialEq)]
pub struct SimRow<'a> {
    pub rid: &'a str,
    /// Earliest loop top at which the row can join a pass (-inf = now).
    pub arrival: f64,
    pub remaining: i64,
    pub prefix: i64,
    pub host: i64,
    /// The row has launched a chunk (its host load is already counted).
    pub started: bool,
}

/// One simulation: the state at `t0` and the work rows in order
/// (partial request, queue rows, target).
#[derive(Debug, Clone)]
pub struct SimState<'a> {
    pub t0: f64,
    pub inflight: &'a [InflightPass],
    pub rows: Vec<SimRow<'a>>,
    pub x_rid: &'a str,
    pub x_arrival: f64,
    /// Expected arrivals after the target (busy P only).
    pub arrivals: Option<&'a Arrivals>,
}

/// Per-simulation record of the target's passes (`wait_s`,
/// `x_passes`, `passes_before`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Trace {
    pub first_start: Option<f64>,
    pub x_passes: u64,
    pub passes_before: u64,
}

struct Work {
    /// Index of the row's rid (rows with the same rid share it).
    gid: usize,
    arrival: f64,
    remaining: i64,
    prefix: i64,
    host: i64,
}

/// Index of a rid (rows with the same rid share one `started` flag).
fn rid_index<'a>(gids: &mut HashMap<&'a str, usize>, rid: &'a str) -> usize {
    let next = gids.len();
    *gids.entry(rid).or_insert(next)
}

/// Work rows of a busy simulation: the partial request, `rows`, the target.
fn busy_rows<'a>(
    partial: Option<&SimRow<'a>>,
    rows: &[SimRow<'a>],
    target: &SimRow<'a>,
) -> Vec<SimRow<'a>> {
    let mut v = Vec::with_capacity(rows.len() + 2);
    v.extend(partial.cloned());
    v.extend(rows.iter().cloned());
    v.push(target.clone());
    v
}

/// Minimum arrival over the work rows (Python `min`).
fn min_arrival(work: &[Work]) -> Option<f64> {
    let mut it = work.iter();
    let mut m = it.next()?.arrival;
    for w in it {
        if w.arrival < m {
            m = w.arrival;
        }
    }
    Some(m)
}

/// FCFS chunked passes from `t0` (exact order of operations).
/// Returns the target's completion time on the same clock, or None (no
/// forecast: a pass without tokens, or more than `MAX_PASSES` passes).
pub fn simulate(
    gm: &GpuModel,
    params: &Params,
    sim: SimState<'_>,
    trace: &mut Trace,
) -> Option<f64> {
    let SimState {
        t0,
        inflight,
        rows,
        x_rid,
        x_arrival,
        arrivals,
    } = sim;
    let chunk = params.chunk();
    let page = params.page();
    let graph = params.graph();
    let lc = &params.launch_coef;
    let lres = params.launch_resid_q.as_deref().filter(|q| !q.is_empty());
    let anchor_launch = params.anchor_is_launch();
    let (idle_delay, phi, lag_none) = (params.idle_delay_s, params.phi_s, params.lag_none_s);
    let [la, lb] = params.lag_next;

    // 1. In-flight passes, in launch order.
    let (mut ps, mut pe, mut pg): (Option<f64>, Option<f64>, f64) = (None, None, 0.0);
    let (mut pbs, mut pfin) = (0.0_f64, 0_i64);
    let mut pl: Option<f64> = None;
    for pas in inflight {
        let launch = pas.launch;
        let f = features(
            pas.new_tokens as f64,
            pas.attention_pairs as f64,
            pas.host_tokens as f64,
            pas.batch_size as f64,
        );
        let s = match pe {
            None => launch,
            Some(e) => py_max(launch, e),
        };
        let g = gpu_s(gm, &f, graph);
        let e = s + g;
        ps = Some(s);
        pe = Some(e);
        pg = g;
        pbs = f[4];
        pfin = pas.finishing;
        pl = Some(launch);
    }

    // 2. Work rows; `started` is keyed by rid, as the reference's set.
    let mut gids: HashMap<&str, usize> = HashMap::new();
    let x_gid = rid_index(&mut gids, x_rid);
    let mut work: Vec<Work> = Vec::with_capacity(rows.len());
    let mut started_rids: Vec<usize> = Vec::new();
    for r in &rows {
        let gid = rid_index(&mut gids, r.rid);
        if r.started {
            started_rids.push(gid);
        }
        work.push(Work {
            gid,
            arrival: r.arrival,
            remaining: r.remaining,
            prefix: r.prefix,
            host: r.host,
        });
    }
    let mut started = vec![false; gids.len()];
    for gid in started_rids {
        started[gid] = true;
    }
    let mut x_end: Option<f64> = None;
    let tail = lag_none;
    let (mut jt, mut jtime) = (0.0_f64, x_arrival);
    let mut eligible: Vec<usize> = Vec::new();
    let mut finished: Vec<usize> = Vec::new();

    // 3. Passes.
    for _ in 0..MAX_PASSES {
        if let Some(xe) = x_end {
            if work.is_empty() {
                let next = NextPass {
                    t0,
                    x_end: xe,
                    prev_start: ps,
                    prev_end: pe,
                    prev_launch: pl,
                    prev_gpu: pg,
                    prev_bs: pbs,
                    prev_finishing: pfin,
                    arrivals,
                    jt,
                    jtime,
                };
                return Some(next_pass_result(gm, params, next).unwrap_or(xe + tail));
            }
        }
        let busy_launch = matches!((ps, pe), (Some(_), Some(e)) if e >= t0);
        let mut launch = if busy_launch {
            let anchor = match (anchor_launch, pl, ps) {
                (true, Some(l), _) => l,
                (_, _, Some(s)) => s,
                _ => return None, // unreachable: busy implies a previous pass
            };
            anchor + lc[0] + lc[1] * pg + lc[2] * pbs + lc[3] * pfin as f64
        } else {
            match min_arrival(&work) {
                None => return x_end.map(|xe| xe + tail),
                Some(nxt) => py_max(py_max(t0, nxt), pe.unwrap_or(f64::NEG_INFINITY)) + idle_delay,
            }
        };
        let cut = if busy_launch { launch - phi } else { launch };
        eligible.clear();
        eligible.extend((0..work.len()).filter(|&i| work[i].arrival <= cut));
        let mut idle_formed = !busy_launch;
        if eligible.is_empty() {
            // Nothing can join the busy launch: the pass forms after an idle
            // wait. `busy_launch` keeps its value (launch load, expected
            // start), as in the E3 scripts.
            idle_formed = true;
            let nxt = min_arrival(&work)?; // the reference raises on empty work
            launch = py_max(nxt, pe.unwrap_or(f64::NEG_INFINITY)) + idle_delay;
            eligible.extend((0..work.len()).filter(|&i| work[i].arrival <= launch));
        }

        // 4. Pass formation (integer tokens; one partial row per pass).
        let mut budget: i64 = chunk;
        let (mut n_tok, mut bs_rows, mut host, mut firsts): (i64, i64, i64, i64) = (0, 0, 0, 0);
        let mut pairs = 0.0_f64;
        finished.clear();
        for &i in &eligible {
            if budget <= 0 {
                break;
            }
            let w = &mut work[i];
            let take = w.remaining.min(budget);
            if !started[w.gid] {
                host += w.host;
                firsts += 1;
                started[w.gid] = true;
            }
            pairs += (take * w.prefix) as f64 + (take * (take + 1)) as f64 / 2.0;
            n_tok += take;
            bs_rows += 1;
            budget -= ceil_page(take, page);
            w.prefix += take;
            w.remaining -= take;
            if w.remaining <= 0 {
                finished.push(i);
            } else {
                break;
            }
        }
        let mut n = n_tok as f64;
        let mut bs = bs_rows as f64;

        // 5. Expected arrivals after the target (they fill the rest of the
        // budget once the target has started).
        if let Some(arr) = arrivals {
            if cut > jtime {
                jt += arr.lam * (cut - jtime);
                jtime = cut;
            }
            if started[x_gid] && budget > 0 && jt > 0.0 {
                // Python `min(budget, jt)`: jt only when it is smaller.
                let j = if jt < budget as f64 {
                    jt
                } else {
                    budget as f64
                };
                n += j;
                pairs += j * arr.pf;
                bs += j / arr.unc;
                jt -= j;
                // The reference also lowers `budget` by j; it is not read
                // again in this pass.
            }
        }

        // 6. The target finished in the previous pass: its result time.
        if let Some(xe) = x_end {
            if !idle_formed || launch < xe + lag_none {
                let g_next = gpu_s(gm, &features(n, pairs, host as f64, bs), graph);
                return Some(py_max(xe, launch + la + lb * g_next));
            }
            return Some(xe + tail);
        }
        // 7. A pass without tokens: no forecast.
        if n == 0.0 {
            return None;
        }
        // 8. Host restore load of a busy launch.
        if busy_launch {
            let host_f = host as f64;
            let c4 = if host > 0 { lc[4] } else { 0.0 };
            let c6 = if lc.len() > 6 {
                lc[6] * firsts as f64
            } else {
                0.0
            };
            launch += c4 + lc[5] * host_f / 1000.0 + c6;
        }
        // 9. GPU time and the (expected) GPU start.
        let g = gpu_s(gm, &features(n, pairs, host as f64, bs), graph);
        let s = match (lres, busy_launch, pe) {
            (Some(q), true, Some(e_prev)) => {
                e_prev
                    + py_sum(q.iter().map(|&r| py_max(0.0, launch + r - e_prev))) / q.len() as f64
            }
            (_, _, Some(e_prev)) => py_max(launch, e_prev),
            (_, _, None) => launch,
        };
        let e = s + g;
        if started[x_gid] {
            if trace.first_start.is_none() {
                trace.first_start = Some(s);
            }
            trace.x_passes += 1;
        } else {
            trace.passes_before += 1;
        }
        // 10. Remove the finished rows (indices ascend in work order).
        for &i in finished.iter().rev() {
            if work[i].gid == x_gid {
                x_end = Some(e);
            }
            work.remove(i);
        }
        // 11. The previous pass.
        ps = Some(s);
        pe = Some(e);
        pg = g;
        pbs = bs;
        pfin = finished.len() as i64;
        pl = Some(launch);
    }
    None
}

/// The state of [`simulate`] when the target's last pass leaves no known
/// work (input of [`next_pass_result`]).
struct NextPass<'a> {
    t0: f64,
    /// GPU end of the target's last pass.
    x_end: f64,
    /// The previous pass (the target's last): GPU start and end, launch,
    /// GPU seconds, batch size and finishing rows.
    prev_start: Option<f64>,
    prev_end: Option<f64>,
    prev_launch: Option<f64>,
    prev_gpu: f64,
    prev_bs: f64,
    prev_finishing: i64,
    arrivals: Option<&'a Arrivals>,
    /// Expected arrival tokens after the target not yet placed in a pass,
    /// counted up to `jtime`.
    jt: f64,
    jtime: f64,
}

/// [`Params::next_pass_lag`]: on a busy P with expected arrivals, the
/// target's result time is `max(x_end, launch + lag_next[0] + lag_next[1] x
/// g_next)`, where `launch` is the busy launch after the target's pass and
/// `g_next` the GPU time of one pass of the arrivals expected by its cut
/// (at most one chunk). None (the reference's `x_end + lag_none_s`) when the
/// switch is off, without arrivals, on an idle P or when no arrival is
/// expected by the cut.
fn next_pass_result(gm: &GpuModel, params: &Params, st: NextPass<'_>) -> Option<f64> {
    if !params.next_pass_lag {
        return None;
    }
    let arr = st.arrivals?;
    let (s_prev, e_prev) = (st.prev_start?, st.prev_end?);
    if e_prev < st.t0 {
        return None;
    }
    let lc = &params.launch_coef;
    let anchor = match (params.anchor_is_launch(), st.prev_launch) {
        (true, Some(l)) => l,
        _ => s_prev,
    };
    let launch = anchor
        + lc[0]
        + lc[1] * st.prev_gpu
        + lc[2] * st.prev_bs
        + lc[3] * st.prev_finishing as f64;
    let cut = launch - params.phi_s;
    let jt = if cut > st.jtime {
        st.jt + arr.lam * (cut - st.jtime)
    } else {
        st.jt
    };
    if jt <= 0.0 {
        return None;
    }
    let j = py_min(params.chunk() as f64, jt);
    let f = features(j, j * arr.pf, 0.0, j / arr.unc);
    let g_next = gpu_s(gm, &f, params.graph());
    let [la, lb] = params.lag_next;
    Some(py_max(st.x_end, launch + la + lb * g_next))
}

fn finish(out: &mut Output, comp: Option<f64>, trace: &Trace, d: f64) {
    out.pred_s = comp.map(|c| c - d);
    out.wait_s = trace.first_start.map(|s| s - d);
    out.x_passes = trace.x_passes;
    out.passes_before = trace.passes_before;
}

/// Forecast of the target's completion (seconds from `d`) and the features
/// that select its error bound.
///
/// Idle P (no in-flight pass, partial request or queue row): one simulation
/// with the idle GPU model and the idle-shifted receipt. Busy P: a
/// simulation with every queue row (`F_full`) and, when the park gate holds
/// queue rows at `t0` (`U > 0` and `U + prefix + remaining > cap_tokens`),
/// one without them (`F_after`); the forecast is `F_after` when the target
/// is not held and `F_after / (F_after + F_full) <= p_after`, else `F_full`.
pub fn forecast(inp: &Input, params: &Params) -> Result<Output, E3Error> {
    params.check()?;
    let (d, t0) = (inp.d, inp.t0);
    let (phi, boot) = (params.phi_s, params.boot_s);

    // 5.1 Rows.
    let mut queue: Vec<SimRow<'_>> = Vec::with_capacity(inp.queue.len());
    let (mut n_dispatched, mut n_unknown) = (0_usize, 0_usize);
    for r in &inp.queue {
        let ready = row_ready(r, params)?;
        if r.source == RowSource::Dispatched {
            n_dispatched += 1;
        }
        let (prefix, host) = match (r.prefix, r.host) {
            (None, _) => {
                n_unknown += 1;
                (0, 0)
            }
            (Some(p), h) => (p, h.unwrap_or(0)),
        };
        let arrival = match ready {
            Some(rd) if rd > t0 => rd - phi,
            _ => f64::NEG_INFINITY,
        };
        queue.push(SimRow {
            rid: &r.rid,
            arrival,
            remaining: (r.input - prefix).max(1),
            prefix,
            host,
            started: false,
        });
    }
    let partial = inp.partial.as_ref().map(|p| SimRow {
        rid: &p.rid,
        arrival: f64::NEG_INFINITY,
        remaining: p.remaining,
        prefix: p.prefix,
        host: 0,
        started: true,
    });
    let x = &inp.target;
    let infl = &inp.inflight[..];
    let idle = infl.is_empty() && partial.is_none() && queue.is_empty();

    // 5.2 Receipt delay of the target (no phi shift).
    let (delay, delay_idle) = receipt_delays(params, x.input);
    let xq = SimRow {
        rid: &x.rid,
        arrival: d + (if idle { delay_idle } else { delay }) + boot,
        remaining: (x.input - x.prefix).max(1),
        prefix: x.prefix,
        host: x.host,
        started: false,
    };
    let tokens_ahead = queue.iter().map(|w| w.remaining).sum::<i64>()
        + partial.as_ref().map_or(0, |p| p.remaining);
    let mut out = Output {
        schema: SCHEMA.to_string(),
        pred_s: None,
        idle,
        n_inflight: infl.len(),
        partial: partial.is_some(),
        n_queue: queue.len(),
        n_dispatched,
        n_prefix_unknown: n_unknown,
        tokens_ahead,
        x_uncached: xq.remaining,
        x_host: xq.host,
        receipt_s: xq.arrival - d,
        path: if idle {
            ForecastPath::Idle
        } else {
            ForecastPath::Full
        },
        f_full_s: None,
        f_after_s: None,
        n_held: 0,
        x_held: false,
        u_tokens: inp.unbound_tokens,
        wait_s: None,
        x_passes: 0,
        passes_before: 0,
    };

    // 5.3 Idle P.
    if idle {
        let mut tr = Trace::default();
        let sim = SimState {
            t0,
            inflight: infl,
            rows: vec![xq.clone()],
            x_rid: &x.rid,
            x_arrival: xq.arrival,
            arrivals: None,
        };
        let comp = simulate(&params.gpu_idle, params, sim, &mut tr);
        finish(&mut out, comp, &tr, d);
        return Ok(out);
    }

    // Busy P: F_full.
    let arr = params.arrivals.as_ref();
    let mut tr = Trace::default();
    let sim = SimState {
        t0,
        inflight: infl,
        rows: busy_rows(partial.as_ref(), &queue, &xq),
        x_rid: &x.rid,
        x_arrival: xq.arrival,
        arrivals: arr,
    };
    let comp = simulate(&params.gpu_busy, params, sim, &mut tr);
    let fk = comp.map(|c| c - d);
    out.f_full_s = fk;
    let u = match inp.unbound_tokens {
        None => {
            finish(&mut out, comp, &tr, d);
            return Ok(out);
        }
        Some(u) => u,
    };

    // 5.6 Park-gate choice (row values before the simulation).
    let cap = params.cap_tokens;
    // Python compares the int sum with the float cap exactly; token sums are
    // far below 2^53, so the f64 conversion is exact.
    let held = |prefix: i64, remaining: i64| u > 0 && ((u + prefix + remaining) as f64) > cap;
    let qa0: Vec<SimRow<'_>> = queue
        .iter()
        .filter(|w| !held(w.prefix, w.remaining))
        .cloned()
        .collect();
    out.n_held = queue.len() - qa0.len();
    let (mut fd, mut tr_d, mut comp_d) = (fk, tr, comp);
    if qa0.len() != queue.len() {
        tr_d = Trace::default();
        let sim = SimState {
            t0,
            inflight: infl,
            rows: busy_rows(partial.as_ref(), &qa0, &xq),
            x_rid: &x.rid,
            x_arrival: xq.arrival,
            arrivals: arr,
        };
        comp_d = simulate(&params.gpu_busy, params, sim, &mut tr_d);
        fd = comp_d.map(|c| c - d);
    }
    out.f_after_s = fd;
    let x_held = held(xq.prefix, xq.remaining);
    out.x_held = x_held;
    let p_after = params.p_after_share();
    if let (false, Some(full), Some(after)) = (x_held, fk, fd) {
        if after + full > 0.0 && p_after >= after / (after + full) {
            out.path = if out.n_held > 0 {
                ForecastPath::After
            } else {
                ForecastPath::Full
            };
            finish(&mut out, comp_d, &tr_d, d);
            return Ok(out);
        }
    }
    finish(&mut out, comp, &tr, d);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gm() -> GpuModel {
        GpuModel {
            small: Some(vec![50.0, 0.03, 2.0, 0.0, 0.0]),
            large: Some(vec![50.0, 0.03, 2.0, 0.0, 0.0]),
            floor_ms: 1.0,
        }
    }

    /// The parameters of the reference self-test (`router_e3._selftest`).
    fn params() -> Params {
        Params {
            schema: Some(SCHEMA.to_string()),
            chunk_tokens: Some(CHUNK_TOKENS),
            page_tokens: Some(PAGE_TOKENS),
            graph_tokens: Some(GRAPH_TOKENS),
            dispatch_s: Some(DISPATCH_S),
            p_after: Some(P_AFTER),
            cap_tokens: 1e9,
            gpu_idle: gm(),
            gpu_busy: gm(),
            launch_coef: vec![0.02, 0.5, 0.0, 0.0, 0.0, 0.0],
            launch_resid_q: None,
            launch_anchor: Some(LaunchAnchor::Launch),
            idle_delay_s: 0.03,
            phi_s: 0.01,
            boot_s: 0.005,
            lag_none_s: 0.02,
            lag_next: [0.05, 0.5],
            rx: None,
            arrivals: None,
            next_pass_lag: false,
        }
    }

    fn idle_input() -> Input {
        Input {
            d: 100.0,
            t0: 99.8,
            inflight: vec![],
            partial: None,
            queue: vec![],
            unbound_tokens: Some(0),
            target: Target {
                rid: "x".into(),
                input: 1000,
                prefix: 0,
                host: 0,
            },
        }
    }

    fn pairs_1000() -> f64 {
        1000.0 * 1001.0 / 2.0
    }

    fn busy_input() -> Input {
        Input {
            t0: 99.96,
            inflight: vec![InflightPass {
                launch: 99.95,
                new_tokens: 1000,
                attention_pairs: 500_500,
                host_tokens: 0,
                batch_size: 1,
                finishing: 1,
            }],
            ..idle_input()
        }
    }

    fn snapshot_row(rid: &str, input: i64, prefix: Option<i64>, host: Option<i64>) -> QueueRow {
        QueueRow {
            rid: rid.into(),
            source: RowSource::Snapshot,
            input,
            prefix,
            host,
            ready: None,
            decision_time: None,
        }
    }

    #[test]
    fn features_and_gpu_model_classes() {
        let f = features(1000.0, 500_500.0, 0.0, 1.0);
        assert_eq!(f, [1.0, 1000.0, 0.5005, 0.0, 1.0]);
        let g = gm();
        // (50 + 0.03 x 1000 + 2 x 0.5005) / 1000.
        assert_eq!(gpu_s(&g, &f, GRAPH_TOKENS), (50.0 + 30.0 + 1.001) / 1000.0);
        // A missing large class falls back to small; floor applies.
        let small_only = GpuModel {
            small: Some(vec![5.0, 0.02, 1.0, 0.0, 0.0]),
            large: None,
            floor_ms: 2.0,
        };
        let big = [1.0, 8000.0, 400.0, 0.0, 1.0];
        assert_eq!(
            gpu_s(&small_only, &big, GRAPH_TOKENS),
            py_sum([5.0, 160.0, 400.0, 0.0, 0.0]) / 1000.0
        );
        let floor = GpuModel {
            small: Some(vec![0.1, 0.0, 0.0, 0.0, 0.0]),
            large: None,
            floor_ms: 3.0,
        };
        assert_eq!(
            gpu_s(&floor, &[1.0, 10.0, 0.0, 0.0, 1.0], GRAPH_TOKENS),
            0.003
        );
        // Small class at exactly graph_tokens; large above it.
        let split = GpuModel {
            small: Some(vec![1.0, 0.0, 0.0, 0.0, 0.0]),
            large: Some(vec![2.0, 0.0, 0.0, 0.0, 0.0]),
            floor_ms: 0.0,
        };
        assert_eq!(gpu_s(&split, &[1.0, 2048.0, 0.0, 0.0, 1.0], 2048), 0.001);
        assert_eq!(gpu_s(&split, &[1.0, 2049.0, 0.0, 0.0, 1.0], 2048), 0.002);
        // An empty class is falsy in the reference: fall back to large.
        let empty_small = GpuModel {
            small: Some(vec![]),
            large: Some(vec![4.0, 0.0, 0.0, 0.0, 0.0]),
            floor_ms: 0.0,
        };
        assert_eq!(
            gpu_s(&empty_small, &[1.0, 10.0, 0.0, 0.0, 1.0], 2048),
            0.004
        );
        let none = GpuModel {
            small: None,
            large: None,
            floor_ms: 1.0,
        };
        assert_eq!(gpu_s(&none, &f, GRAPH_TOKENS), NO_MODEL_GPU_S);
    }

    #[test]
    fn receipt_delays_and_row_ready() {
        let mut p = params();
        assert_eq!(receipt_delays(&p, 5000), (DISPATCH_S, DISPATCH_S));
        p.rx = Some(RxParams {
            a: 0.04,
            b: 1e-6,
            shift: -0.01,
        });
        let (line, idle) = receipt_delays(&p, 5000);
        assert_eq!(line, 0.04 + 1e-6 * 5000.0);
        assert_eq!(idle, line - 0.01);
        // Both delays are clamped at 0.
        p.rx = Some(RxParams {
            a: -0.5,
            b: 0.0,
            shift: -0.1,
        });
        assert_eq!(receipt_delays(&p, 10), (0.0, 0.0));
        p.rx = Some(RxParams {
            a: 0.04,
            b: 1e-6,
            shift: -0.01,
        });
        let mut row = snapshot_row("a", 100, Some(0), Some(0));
        assert_eq!(row_ready(&row, &p), Ok(None));
        row.ready = Some(101.5);
        assert_eq!(row_ready(&row, &p), Ok(Some(101.5)));
        row.source = RowSource::Dispatched;
        assert!(row_ready(&row, &p).is_err());
        row.decision_time = Some(100.0);
        assert_eq!(
            row_ready(&row, &p),
            Ok(Some(100.0 + (0.04 + 1e-6 * 100.0) + p.boot_s))
        );
    }

    #[test]
    fn idle_p_matches_the_reference_self_test() {
        let out = forecast(&idle_input(), &params()).unwrap();
        let g = (50.0 + 0.03 * 1000.0 + 2.0 * pairs_1000() / 1e6) / 1000.0;
        let want = (100.0 + 0.044 + 0.005 + 0.03 + g + 0.02) - 100.0;
        assert!((out.pred_s.unwrap() - want).abs() < 1e-12, "{out:?}");
        // Reference output, bit for bit.
        assert_eq!(out.pred_s, Some(0.18000099999999009));
        assert_eq!(out.wait_s, Some(0.07899999999999352));
        assert!(out.idle && out.path == ForecastPath::Idle);
        assert_eq!((out.x_passes, out.passes_before), (1, 0));
        assert_eq!(out.f_full_s, None);
        assert_eq!(out.schema, SCHEMA);
        assert!((out.receipt_s - (0.044 + 0.005)).abs() < 1e-12);
        assert_eq!(out.u_tokens, Some(0));
    }

    #[test]
    fn busy_p_forms_the_target_pass_after_an_idle_wait() {
        // The next busy launch 99.95 + 0.02 + 0.5 x gpu has its cut before the
        // target's arrival (100.049): the pass forms after an idle wait,
        // launch = 100.049 + idle_delay (busy_launch keeps its value).
        let out = forecast(&busy_input(), &params()).unwrap();
        let g = (50.0 + 0.03 * 1000.0 + 2.0 * pairs_1000() / 1e6) / 1000.0;
        let want = (100.049 + 0.03) + g + 0.02 - 100.0;
        assert!((out.pred_s.unwrap() - want).abs() < 1e-9, "{out:?}");
        assert_eq!(
            (out.pred_s, out.f_full_s),
            (Some(0.18000099999999009), Some(0.18000099999999009))
        );
        assert!(!out.idle && out.path == ForecastPath::Full && out.passes_before == 0);
        assert_eq!(out.n_inflight, 1);
    }

    #[test]
    fn busy_p_joins_the_busy_launch() {
        // A longer launch interval (0.1 s) puts the target into the busy launch
        // 99.95 + 0.1 + 0.5 x gpu; it starts at max(launch, previous GPU end).
        let mut p = params();
        p.launch_coef = vec![0.1, 0.5, 0.0, 0.0, 0.0, 0.0];
        let out = forecast(&busy_input(), &p).unwrap();
        let g = (50.0 + 0.03 * 1000.0 + 2.0 * pairs_1000() / 1e6) / 1000.0;
        let launch = 99.95 + 0.1 + 0.5 * g;
        assert!(launch - 0.01 >= 100.049);
        let want = py_max(launch, 99.95 + g) + g + 0.02 - 100.0;
        assert!((out.pred_s.unwrap() - want).abs() < 1e-12, "{out:?}");
    }

    #[test]
    fn park_gate_choice_with_a_held_row() {
        let mut p = params();
        p.cap_tokens = 50_000.0;
        let mut inp = busy_input();
        inp.unbound_tokens = Some(40_000);
        inp.queue = vec![snapshot_row("big", 20_000, Some(0), Some(0))];
        let out = forecast(&inp, &p).unwrap();
        assert_eq!(out.n_held, 1);
        assert!(!out.x_held);
        let (full, after) = (out.f_full_s.unwrap(), out.f_after_s.unwrap());
        assert!(after < full);
        let want = if after / (after + full) <= 0.2 {
            ForecastPath::After
        } else {
            ForecastPath::Full
        };
        assert_eq!(out.path, want);
        // Reference output: F_full 1.2320220000000006, F_after 0.18000099999999009, path after.
        assert_eq!(
            (full, after, out.path),
            (1.2320220000000006, 0.18000099999999009, ForecastPath::After)
        );
        assert_eq!(out.pred_s, Some(after));
        // Without U the park-gate choice is skipped.
        inp.unbound_tokens = None;
        let out2 = forecast(&inp, &p).unwrap();
        assert_eq!(
            (out2.n_held, out2.f_after_s, out2.path),
            (0, None, ForecastPath::Full)
        );
        assert_eq!(out2.pred_s, out2.f_full_s);
        // A held target takes F_full.
        inp.unbound_tokens = Some(49_500);
        let out3 = forecast(&inp, &p).unwrap();
        assert!(out3.x_held);
        assert_eq!(out3.pred_s, out3.f_full_s);
        assert_eq!(
            (out3.wait_s, out3.passes_before),
            (Some(0.8909728400000034), 2)
        );
    }

    #[test]
    fn a_null_prefix_row_is_cold() {
        let mut a = busy_input();
        a.queue = vec![snapshot_row("big", 20_000, Some(0), Some(0))];
        let mut b = busy_input();
        b.queue = vec![snapshot_row("big", 20_000, None, None)];
        let (oa, ob) = (
            forecast(&a, &params()).unwrap(),
            forecast(&b, &params()).unwrap(),
        );
        assert_eq!(oa.pred_s, ob.pred_s);
        assert_eq!((oa.n_prefix_unknown, ob.n_prefix_unknown), (0, 1));
        assert_eq!(ob.tokens_ahead, 20_000);
        // A given prefix with a null host takes host 0.
        let mut c = busy_input();
        c.queue = vec![snapshot_row("big", 20_000, Some(5_000), None)];
        let oc = forecast(&c, &params()).unwrap();
        assert_eq!((oc.n_prefix_unknown, oc.tokens_ahead), (0, 15_000));
    }

    #[test]
    fn passes_without_tokens_and_the_pass_limit_give_no_forecast() {
        let mut p = params();
        p.chunk_tokens = Some(0);
        let out = forecast(&idle_input(), &p).unwrap();
        assert_eq!(out.pred_s, None);
        assert_eq!(out.wait_s, None);
        let mut p2 = params();
        p2.chunk_tokens = Some(1);
        p2.page_tokens = Some(1);
        let mut inp = idle_input();
        inp.target.input = 150_000;
        let out2 = forecast(&inp, &p2).unwrap();
        assert_eq!(out2.pred_s, None);
        assert_eq!(out2.x_passes as usize, MAX_PASSES);
    }

    #[test]
    fn invalid_parameters_are_errors() {
        let mut p = params();
        p.page_tokens = Some(0);
        assert!(forecast(&idle_input(), &p).is_err());
        let mut p2 = params();
        p2.launch_coef = vec![0.1, 0.5];
        assert!(forecast(&idle_input(), &p2).is_err());
    }

    #[test]
    fn launch_anchor_missing_null_and_start() {
        let mut v = serde_json::to_value(params()).unwrap();
        v.as_object_mut().unwrap().remove("launch_anchor");
        let p: Params = serde_json::from_value(v.clone()).unwrap();
        assert!(p.anchor_is_launch());
        v["launch_anchor"] = serde_json::Value::Null;
        let p: Params = serde_json::from_value(v.clone()).unwrap();
        assert!(!p.anchor_is_launch());
        v["launch_anchor"] = "start".into();
        let p: Params = serde_json::from_value(v).unwrap();
        assert_eq!(p.launch_anchor, Some(LaunchAnchor::Start));
        assert!(!p.anchor_is_launch());
    }

    #[test]
    fn dispatched_rows_partial_and_expected_arrivals() {
        let mut p = params();
        p.rx = Some(RxParams {
            a: 0.04,
            b: 1e-7,
            shift: -0.01,
        });
        p.arrivals = Some(Arrivals {
            lam: 20_000.0,
            rate: 3.0,
            unc: 6_000.0,
            pf: 50_000.0,
        });
        p.launch_resid_q = Some((0..19).map(|j| 0.001 * f64::from(j) - 0.01).collect());
        let mut inp = busy_input();
        inp.partial = Some(Partial {
            rid: "p".into(),
            remaining: 9_000,
            prefix: 40_000,
        });
        inp.queue = vec![QueueRow {
            rid: "q".into(),
            source: RowSource::Dispatched,
            input: 15_000,
            prefix: Some(8_000),
            host: Some(8_000),
            ready: None,
            decision_time: Some(99.98),
        }];
        inp.target = Target {
            rid: "x".into(),
            input: 30_000,
            prefix: 12_000,
            host: 4_000,
        };
        let out = forecast(&inp, &p).unwrap();
        assert_eq!((out.n_dispatched, out.n_queue, out.partial), (1, 1, true));
        // Reference output, bit for bit.
        assert_eq!(
            (out.pred_s, out.wait_s),
            (Some(3.92927499999999), Some(1.0492380560000072))
        );
        assert_eq!(
            (out.x_passes, out.passes_before, out.receipt_s),
            (4, 1, 0.04800000000000182)
        );
        assert_eq!(out.tokens_ahead, 9_000 + 7_000);
        assert_eq!((out.x_uncached, out.x_host), (18_000, 4_000));
        let pred = out.pred_s.unwrap();
        let wait = out.wait_s.unwrap();
        assert!(pred > wait && wait > 0.0, "{out:?}");
        // 18,000 target tokens need at least 3 passes of 8,192.
        assert!(out.x_passes >= 3, "{out:?}");
        // Without expected arrivals the result lag uses only the known work.
        let mut p2 = p.clone();
        p2.arrivals = None;
        let out2 = forecast(&inp, &p2).unwrap();
        assert!(out2.pred_s.unwrap() <= pred + 1e-12, "{out2:?} vs {out:?}");
        assert_eq!(out2.pred_s, Some(3.0390349999999984));
        // `next_pass_lag`: the target is the last row, so its result takes the
        // next pass of the expected arrivals, here launched before its GPU end
        // (result = x_end, the bare lag of 0.02 s goes). The patched reference.
        let p3 = Params {
            next_pass_lag: true,
            ..p.clone()
        };
        let out3 = forecast(&inp, &p3).unwrap();
        assert_eq!(
            (out3.pred_s, out3.wait_s, out3.x_passes),
            (Some(3.909274999999994), Some(1.0492380560000072), 4)
        );
    }

    /// `next_pass_lag` (the Kimi-K2.6 fix of the E3 result lag) against the
    /// reference router_e3.py patched with the same fix, bit for bit. Busy P:
    /// one in-flight pass, the target (1,000 tokens) alone in the queue.
    #[test]
    fn next_pass_lag_prices_the_pass_of_the_expected_arrivals() {
        let arr = Arrivals {
            lam: 20_000.0,
            rate: 3.0,
            unc: 6_000.0,
            pf: 50_000.0,
        };
        let run = |arrivals: Option<Arrivals>, on: bool| {
            let p = Params {
                arrivals,
                next_pass_lag: on,
                ..params()
            };
            let out = forecast(&busy_input(), &p).unwrap();
            (out.pred_s, out.wait_s)
        };
        let reference = (Some(0.18000099999999009), Some(0.07899999999999352));
        // Off: the reference, x_end + lag_none_s (0.02 s), arrivals or not.
        assert_eq!(run(Some(arr), false), reference);
        assert_eq!(run(None, false), reference);
        // On: launch of the next busy pass + lag_next over the GPU time of the
        // arrivals expected by its cut. The wait (GPU start) is unchanged.
        assert_eq!(
            run(Some(arr), true),
            (Some(0.31915114999995353), reference.1)
        );
        // More than one chunk expected: the next pass is one chunk.
        let flood = Arrivals { lam: 2.0e6, ..arr };
        assert_eq!(run(Some(flood), true).0, Some(0.7469804999999781));
        // No arrivals model, or no arrival expected by the cut: the reference.
        assert_eq!(run(None, true), reference);
        let none = Arrivals { lam: 0.0, ..arr };
        assert_eq!(run(Some(none), true), reference);
        // The idle path never applies it (no arrivals there).
        let idle = |on: bool| {
            let p = Params {
                arrivals: Some(arr),
                next_pass_lag: on,
                ..params()
            };
            forecast(&idle_input(), &p).unwrap().pred_s
        };
        assert_eq!(idle(true), idle(false));
    }

    #[test]
    fn next_pass_lag_is_off_unless_set_and_kept_out_of_the_reference_json() {
        let mut v = serde_json::to_value(params()).unwrap();
        assert!(v.get("next_pass_lag").is_none(), "{v}");
        let p: Params = serde_json::from_value(v.clone()).unwrap();
        assert!(!p.next_pass_lag);
        v["next_pass_lag"] = true.into();
        let p: Params = serde_json::from_value(v).unwrap();
        assert!(p.next_pass_lag);
        assert_eq!(serde_json::to_value(&p).unwrap()["next_pass_lag"], true);
    }
}
