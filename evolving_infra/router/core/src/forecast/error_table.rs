// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! Error bound table v2 and the reroute rule.
//!
//! A bound table gives, per bucket (`idle` | `busy` in the shipped scheme)
//! and level (`"0.8"`, `"0.9"`), a continuous, non-decreasing curve
//! `b(pred)`. The tables are fitted offline (`router_forecast_bounds.py`) and
//! shipped in `bound-params.json` with the level of each deploy mode and
//! `g_min_s`. Table v1 (r3, `pred_edges_s`) is not ported: it is not
//! monotone and must not be deployed; a v1 table does not parse.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::model::Output;
use super::numeric::py_max;
use super::ForecastError;

/// Bucket scheme of a bound table (the shipped scheme is `idle`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BucketScheme {
    /// `idle` | `busy`.
    Idle,
    /// `idle` | `busy_w0` | `busy_w1` | `busy_w2` by predicted wait share
    /// `wait_s / pred_s` (< 0.25, < 0.6, else).
    Wait,
    /// `idle` | `busy_a0` | `busy_a1` | `busy_a2` by work rows ahead
    /// (in-flight passes + partial + queue rows: <= 1, <= 4, more).
    Ahead,
}

/// The forecast fields that select a bound (a subset of [`Output`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct BucketView {
    #[serde(default)]
    pub pred_s: Option<f64>,
    pub idle: bool,
    #[serde(default)]
    pub wait_s: Option<f64>,
    #[serde(default)]
    pub n_inflight: usize,
    #[serde(default)]
    pub partial: bool,
    #[serde(default)]
    pub n_queue: usize,
}

impl From<&Output> for BucketView {
    fn from(o: &Output) -> Self {
        Self {
            pred_s: o.pred_s,
            idle: o.idle,
            wait_s: o.wait_s,
            n_inflight: o.n_inflight,
            partial: o.partial,
            n_queue: o.n_queue,
        }
    }
}

/// One bound curve: knots (median pred of each bin, increasing) and bounds
/// (the bin's level-quantile of |label - pred|, non-decreasing), seconds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundCurve {
    pub knots_s: Vec<f64>,
    pub b_s: Vec<f64>,
    /// Fit forecasts per bin (informational).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<Vec<u64>>,
}

/// A bound table v2 (schema `forecast-error-bound.v2`). Fields of the fit report
/// that the bound does not use (`min_n`, `init_edges_s`, `n_fit`) are
/// ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundTable {
    #[serde(default)]
    pub schema: Option<String>,
    pub scheme: BucketScheme,
    #[serde(default)]
    pub min_bound_s: Option<f64>,
    /// level -> bucket -> curve; every level has the curve `all`.
    pub levels: BTreeMap<String, BTreeMap<String, BoundCurve>>,
}

/// Bucket of a forecast for the bound table (reference `bucket_of`).
pub fn bucket_of(view: &BucketView, scheme: BucketScheme) -> &'static str {
    if view.idle {
        return "idle";
    }
    match scheme {
        BucketScheme::Idle => "busy",
        BucketScheme::Wait => {
            let share = match (view.pred_s, view.wait_s) {
                (Some(p), Some(w)) if p != 0.0 && p > 0.0 => w / p,
                _ => 0.0,
            };
            if share < 0.25 {
                "busy_w0"
            } else if share < 0.6 {
                "busy_w1"
            } else {
                "busy_w2"
            }
        }
        BucketScheme::Ahead => {
            let ahead = view.n_inflight + usize::from(view.partial) + view.n_queue;
            if ahead <= 1 {
                "busy_a0"
            } else if ahead <= 4 {
                "busy_a1"
            } else {
                "busy_a2"
            }
        }
    }
}

/// Continuous, non-decreasing bound curve through `(knots[i], values[i])`
/// (11): `values[0]` at and below `knots[0]`; linear between knots
/// (`bisect_right`); `values[-1] x p / knots[-1]` (a constant relative
/// bound) at and above `knots[-1]`. None for empty or unequal inputs.
pub fn bound_curve(knots: &[f64], values: &[f64], p: f64) -> Option<f64> {
    if knots.is_empty() || knots.len() != values.len() {
        return None;
    }
    let last = knots.len() - 1;
    if p <= knots[0] {
        return Some(values[0]);
    }
    if p >= knots[last] {
        return Some(values[last] * p / knots[last]);
    }
    // knots[i - 1] <= p < knots[i]; 1 <= i <= last here (a NaN p or knots
    // out of order fall outside: no bound).
    let i = knots.partition_point(|&k| k <= p);
    if i == 0 || i > last {
        return None;
    }
    let (x0, x1, y0, y1) = (knots[i - 1], knots[i], values[i - 1], values[i]);
    Some(y0 + (y1 - y0) * (p - x0) / (x1 - x0))
}

/// Bound (s) of |actual - forecast| at the nominal coverage `level` (`"0.8"`
/// or `"0.9"`); `Ok(None)` without a forecast. The bucket's curve, else the
/// curve `all`; the result is at least `min_bound_s`. Errors for a level or
/// curve the table does not have.
pub fn error_bound(
    view: &BucketView,
    table: &BoundTable,
    level: &str,
) -> Result<Option<f64>, ForecastError> {
    let Some(p) = view.pred_s else {
        return Ok(None);
    };
    let curves = table
        .levels
        .get(level)
        .ok_or_else(|| ForecastError::Bound(format!("no level {level}")))?;
    let bucket = bucket_of(view, table.scheme);
    let curve = curves
        .get(bucket)
        .or_else(|| curves.get("all"))
        .ok_or_else(|| ForecastError::Bound(format!("level {level}: no curve {bucket} or all")))?;
    let b = bound_curve(&curve.knots_s, &curve.b_s, p)
        .ok_or_else(|| ForecastError::Bound(format!("level {level}: bad curve {bucket}")))?;
    Ok(Some(py_max(table.min_bound_s.unwrap_or(0.0), b)))
}

/// Bound-adjusted gain of moving a request from its owner P (forecast
/// `c_owner`, bound `b_owner`) to another P: `(C_o - C_w) - sqrt(b_o^2 +
/// b_w^2)`. Reroute only when it exceeds `g_min_s`; the KV transfer
/// and duplication costs are charged separately by the routing policy.
pub fn reroute_gain(c_owner: f64, b_owner: f64, c_other: f64, b_other: f64) -> f64 {
    (c_owner - c_other) - (b_owner * b_owner + b_other * b_other).sqrt()
}

/// The shipped bound of one deploy mode (`bound-params.json`, `modes.<mode>`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundModeParams {
    /// Level used by the reroute rule (the lower level whose held-out
    /// firing exceedance meets the nominal value).
    pub level: String,
    #[serde(default)]
    pub level_meets_nominal: Option<bool>,
    pub table: BoundTable,
}

/// `bound-params.json` (schema `forecast-error-table.v2`): the shipped
/// tables of the deploy modes, their levels and `g_min_s`. The held-out
/// statistics of the file are not parsed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorTable {
    pub schema: String,
    /// sha256 of the `bound-tables.json` the tables were copied from.
    pub bound_tables_sha256: String,
    #[serde(default)]
    pub fit_cells: Vec<String>,
    /// Minimum bound-adjusted gain (s) of a reroute.
    pub g_min_s: f64,
    pub modes: BTreeMap<String, BoundModeParams>,
}

impl ErrorTable {
    /// Bound of a forecast in `mode` at the mode's shipped level.
    pub fn bound(&self, mode: &str, view: &BucketView) -> Result<Option<f64>, ForecastError> {
        let m = self
            .modes
            .get(mode)
            .ok_or_else(|| ForecastError::Bound(format!("no deploy mode {mode}")))?;
        error_bound(view, &m.table, &m.level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> BoundTable {
        serde_json::from_value(serde_json::json!({
            "schema": "forecast-error-bound.v2", "scheme": "idle", "min_bound_s": 0.0,
            "levels": {"0.8": {"busy": {"knots_s": [0.2, 1.0, 4.0], "b_s": [0.3, 0.5, 1.0]},
                               "all": {"knots_s": [1.0], "b_s": [0.4]}}}}))
        .unwrap()
    }

    fn busy(p: Option<f64>) -> BucketView {
        BucketView {
            pred_s: p,
            ..BucketView::default()
        }
    }

    #[test]
    fn bound_curve_branches() {
        let (k, b) = ([0.15, 0.7, 1.5, 5.0], [0.5, 0.5, 0.9, 1.8]);
        assert_eq!(bound_curve(&k, &b, 0.1), Some(0.5));
        assert_eq!(bound_curve(&k, &b, 0.15), Some(0.5));
        assert_eq!(bound_curve(&k, &b, 5.0), Some(1.8 * 5.0 / 5.0));
        assert_eq!(bound_curve(&k, &b, 9.0), Some(1.8 * 9.0 / 5.0));
        // Between knots 1.5 and 5.0: 0.9 + 0.9 x (4.0 - 1.5) / 3.5.
        assert_eq!(
            bound_curve(&k, &b, 4.0),
            Some(0.9 + (1.8 - 0.9) * (4.0 - 1.5) / (5.0 - 1.5))
        );
        // At an inner knot the right-hand segment starts: its left value.
        assert_eq!(bound_curve(&k, &b, 1.5), Some(0.9));
        assert_eq!(bound_curve(&[], &[], 1.0), None);
        assert_eq!(bound_curve(&[1.0, 2.0], &[1.0], 1.0), None);
    }

    #[test]
    fn error_bound_is_continuous_and_non_decreasing() {
        let t = table();
        assert_eq!(error_bound(&busy(None), &t, "0.8"), Ok(None));
        let vals: Vec<f64> = (1..2000)
            .map(|i| {
                error_bound(&busy(Some(f64::from(i) / 100.0)), &t, "0.8")
                    .unwrap()
                    .unwrap()
            })
            .collect();
        assert!(vals.windows(2).all(|w| w[1] >= w[0]));
        assert_eq!(vals[0], 0.3);
        assert!((vals[vals.len() - 1] - 19.99 / 4.0).abs() < 1e-12);
        assert!((error_bound(&busy(Some(0.6)), &t, "0.8").unwrap().unwrap() - 0.4).abs() < 1e-12);
        // The idle bucket is not in the table: the curve `all`.
        let idle = BucketView {
            idle: true,
            pred_s: Some(2.0),
            ..BucketView::default()
        };
        assert_eq!(error_bound(&idle, &t, "0.8"), Ok(Some(0.4 * 2.0 / 1.0)));
        assert!(error_bound(&busy(Some(1.0)), &t, "0.9").is_err());
        // min_bound_s floors the bound.
        let mut t2 = t.clone();
        t2.min_bound_s = Some(0.35);
        assert_eq!(error_bound(&busy(Some(0.1)), &t2, "0.8"), Ok(Some(0.35)));
    }

    #[test]
    fn bucket_schemes() {
        let mut v = BucketView {
            pred_s: Some(1.0),
            wait_s: Some(0.1),
            n_inflight: 1,
            n_queue: 3,
            ..BucketView::default()
        };
        assert_eq!(bucket_of(&v, BucketScheme::Idle), "busy");
        assert_eq!(bucket_of(&v, BucketScheme::Wait), "busy_w0");
        assert_eq!(bucket_of(&v, BucketScheme::Ahead), "busy_a1");
        v.wait_s = Some(0.5);
        v.partial = true;
        v.n_queue = 4;
        assert_eq!(bucket_of(&v, BucketScheme::Wait), "busy_w1");
        assert_eq!(bucket_of(&v, BucketScheme::Ahead), "busy_a2");
        v.wait_s = Some(0.9);
        assert_eq!(bucket_of(&v, BucketScheme::Wait), "busy_w2");
        v.pred_s = None;
        assert_eq!(bucket_of(&v, BucketScheme::Wait), "busy_w0");
        v.idle = true;
        assert_eq!(bucket_of(&v, BucketScheme::Ahead), "idle");
        assert!(serde_json::from_str::<BucketScheme>("\"other\"").is_err());
    }

    #[test]
    fn reroute_gain_charges_both_bounds() {
        assert_eq!(
            reroute_gain(2.4, 0.9, 1.1, 0.5),
            (2.4 - 1.1) - (0.9f64 * 0.9 + 0.5 * 0.5).sqrt()
        );
        assert!(reroute_gain(0.4, 0.3, 0.35, 0.3) < 0.0);
    }

    #[test]
    fn a_v1_table_does_not_parse() {
        let v1 = serde_json::json!({"scheme": "idle", "pred_edges_s": [0.5, 1.0],
                                    "levels": {"0.8": {"busy": [0.5, 0.6, 0.7], "all": [0.5, 0.6, 0.7]}}});
        assert!(serde_json::from_value::<BoundTable>(v1).is_err());
    }
}
