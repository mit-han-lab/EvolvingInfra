// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Closed-form earliest-finish forecast of one prefill (P) worker (EFT-0+).
//!
//! This is a Rust port of the Python reference
//! `eft_cheap.py` of the planner's PD calibration package
//! (`evolving_infra.planner`). The golden file `tests/fixtures/eft_cheap_golden.jsonl`
//! holds the inputs and outputs of the reference. The unit test at the end of
//! this file checks that the port gives the same values: numbers within 1e-9
//! relative, integers and strings exactly.
//!
//! The step model prices the work ahead of a request in closed form:
//!
//! ```text
//! W(u,q,h,s) = b_new u + b_pairs (u q + u (u + 1) / 2) + (b_host + b_nhost) h
//!            + (b_stor + b_nstor) s + (b_bs + b_nbs) ceil(u / C) + b_fin + b_nhr [h > 0]
//! P(X)       = ceil(sum over rows of page-rounded u / C)          (passes, b0 each)
//! D_c        = max(0, R0 + sum_S W + b0 P(S) - a) + sum_O W + b0 (P(S u O) - P(S))
//! Svc        = W(target) + b0 ceil(ceil_page(u) / C)
//! ```
//!
//! S = rows of the engine `prefill_routing` snapshot, O = rows the
//! router dispatched that the snapshot does not show, R0 = rest of the
//! in-flight pass, a = view age (poll sent to decision, plus the engine's
//! own snapshot age).
//!
//! The router's TRADE wait term (`--co-schedule-trade-wait eft`) uses
//! [`drain`] (B_w = D_c) and [`service_seconds`] (S_w = Svc). [`decide`] is
//! the complete EFT-0+ placement of the reference. The router does not call
//! it; the golden test does, so that every part of the port is checked.
//!
//! Pure and deterministic: no clock read, no I/O except [`load_model`],
//! candidates in URL order, integer token arithmetic and float operations in
//! the order of the reference.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Map, Value};

pub const SCHEMA: &str = "eft-cheap.v1";
const FAMILY: &str = "augmented";
pub const DEFAULT_PAGE: i64 = 64;
pub const DEFAULT_CHUNK: i64 = 8192;
const FEATURES: [&str; 6] = [
    "intercept",
    "new_tokens",
    "attention_pairs",
    "batch_size",
    "host_restore_tokens",
    "storage_restore_tokens",
];
const COMPOSED_FEATURES: [&str; 11] = [
    "intercept",
    "new_tokens",
    "attention_pairs",
    "batch_size",
    "host_restore_tokens",
    "storage_restore_tokens",
    "finishing_requests",
    "next_batch_size",
    "next_has_host_restore",
    "next_host_restore_tokens",
    "next_storage_restore_tokens",
];
const SERVICE_KIND: &str = "idle_launch_to_done";
/// An unidentified (0) storage coefficient becomes this multiple of the host
/// coefficient (EFTACC: pulled tokens cost 1.5-2.1 x host restore tokens).
const STORAGE_PRIOR: f64 = 2.0;
const DEFAULT_RX: (f64, f64) = (0.044, 0.0);
const DEFAULT_IDLE_DELAY: f64 = 0.03;
const READINESS: [&str; 5] = [
    "held",
    "parked",
    "pull_pending",
    "pull_inflight",
    "restore_inflight",
];

// ---------------------------------------------------------------------------
// JSON and arithmetic helpers with the semantics of the Python reference.
// ---------------------------------------------------------------------------

/// Python `valid(x)`: a finite number (JSON null and booleans are not).
fn finite(v: Option<&Value>) -> Option<f64> {
    v.and_then(Value::as_f64).filter(|x| x.is_finite())
}

/// Python `nonneg_int(x)`: a JSON integer >= 0 (a float such as 5.0 is not).
pub(crate) fn nonneg_int(v: Option<&Value>) -> Option<i64> {
    v.and_then(Value::as_u64)
        .and_then(|x| i64::try_from(x).ok())
}

/// Python `x is True`.
pub(crate) fn is_true(v: Option<&Value>) -> bool {
    matches!(v, Some(Value::Bool(true)))
}

/// Python truthiness of a JSON value.
fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|x| x != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// Python `max(a, b)`: `a` unless `b` is larger (keeps the sign of a zero).
fn pymax(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

fn flag(degraded: &mut BTreeSet<String>, reason: &str) {
    degraded.insert(reason.to_string());
}

/// ceil(a / b) for b > 0 (Python `-(-a // b)`).
fn ceil_div(a: i64, b: i64) -> i64 {
    -((-a).div_euclid(b))
}

fn ceil_page(value: i64, page: i64) -> i64 {
    (value + page - 1).div_euclid(page) * page
}

/// Page-floored cached prefix that leaves at least one page to compute
/// (the rule of `prefill_completion.target_work`).
pub fn floor_hit(tokens: i64, n: i64, page: i64) -> i64 {
    (tokens.div_euclid(page) * page).min(0.max(n - 2).div_euclid(page) * page)
}

/// P(X) = ceil(sum of page-rounded sizes / chunk tokens).
fn passes(sizes: impl IntoIterator<Item = i64>, page: i64, chunk: i64) -> i64 {
    ceil_div(sizes.into_iter().map(|u| ceil_page(u, page)).sum(), chunk)
}

/// Integer microseconds; halves round up: floor(seconds * 1e6 + 0.5).
pub fn round_us(seconds: f64) -> i64 {
    (seconds * 1e6 + 0.5).floor() as i64
}

// ---------------------------------------------------------------------------
// Frozen step-time model.
// ---------------------------------------------------------------------------

/// The step-time values that EFT-0+ reads from a frozen model
/// (`prefill_completion_fit.fit_prefill_completion_model` layout).
#[derive(Debug, Clone, PartialEq)]
pub struct EftModel {
    pub b0: f64,
    pub b_new: f64,
    pub b_pairs: f64,
    pub b_bs: f64,
    pub b_host: f64,
    pub b_stor: f64,
    pub b_fin: f64,
    pub b_nbs: f64,
    pub b_nhr: f64,
    pub b_nhost: f64,
    pub b_nstor: f64,
    pub tau: f64,
    pub rx_a: f64,
    pub rx_b: f64,
    pub idle_delay: f64,
    pub service_split_tokens: Option<i64>,
    pub service_all: Option<[f64; 6]>,
    pub service_small: Option<[f64; 6]>,
    pub service_large: Option<[f64; 6]>,
    pub sha256: String,
    pub sources: Map<String, Value>,
}

type Fit = Vec<(String, f64)>;

/// The last weight of `name` (Python `dict(zip(names, weights))`: last wins).
fn lookup(fit: &[(String, f64)], name: &str) -> Option<f64> {
    fit.iter().rev().find(|(n, _)| n == name).map(|(_, w)| *w)
}

/// Python `_named`: names (default `defaults`) and finite weights of a fit.
fn named_fit(fit: Option<&Value>, defaults: &[&str]) -> Result<Fit, String> {
    let fit = fit
        .and_then(Value::as_object)
        .ok_or_else(|| "fit missing".to_string())?;
    let names: Vec<String> = match fit.get("features") {
        Some(names) if truthy(Some(names)) => names
            .as_array()
            .ok_or_else(|| "fit feature schema".to_string())?
            .iter()
            .map(|n| {
                n.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| "fit feature schema".to_string())
            })
            .collect::<Result<_, _>>()?,
        _ => defaults.iter().map(|s| s.to_string()).collect(),
    };
    let weights = fit
        .get("coefficients")
        .and_then(Value::as_array)
        .ok_or_else(|| "fit feature schema".to_string())?;
    if names.len() != weights.len() {
        return Err("fit feature schema".into());
    }
    let weights = weights
        .iter()
        .map(|w| finite(Some(w)).ok_or_else(|| "fit weight invalid".to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(names.into_iter().zip(weights).collect())
}

/// Python `usable_fit`: named features starting with the intercept, finite weights.
fn usable_fit(fit: Option<&Value>) -> Option<Fit> {
    let fit = fit?.as_object()?;
    let names = fit.get("features")?.as_array()?;
    let weights = fit.get("coefficients")?.as_array()?;
    if names.is_empty() || names.len() != weights.len() || names[0].as_str() != Some("intercept")
    {
        return None;
    }
    names
        .iter()
        .zip(weights)
        .map(|(n, w)| Some((n.as_str()?.to_string(), finite(Some(w))?)))
        .collect()
}

fn subset(fit: &[(String, f64)], allowed: &[&str]) -> bool {
    fit.iter().all(|(n, _)| allowed.contains(&n.as_str()))
}

/// Python `_vector`: a usable fit over FEATURES as a coefficient array.
fn vector(fit: Option<&Value>) -> Option<[f64; 6]> {
    let fit = usable_fit(fit)?;
    if !subset(&fit, &FEATURES) {
        return None;
    }
    let mut out = [0.0; 6];
    for (slot, name) in out.iter_mut().zip(FEATURES) {
        *slot = lookup(&fit, name).unwrap_or(0.0);
    }
    Some(out)
}

fn holdout_mean(fits: Option<&Value>) -> Option<f64> {
    finite(
        fits?
            .get("global")?
            .get("holdout_metrics")?
            .get("actual_mean_s"),
    )
}

impl EftModel {
    /// Python `model_from_json`. `sha256` is the digest of the model file;
    /// without it, the digest of the canonical JSON (Python
    /// `json.dumps(model, sort_keys=True, separators=(",", ":"))`) is used.
    pub fn from_json(model: &Value, sha256: Option<String>) -> Result<Self, String> {
        let mut sources = Map::new();
        let kinds = model
            .get("models")
            .and_then(|m| m.get(FAMILY))
            .filter(|k| k.is_object())
            .ok_or_else(|| {
                "no pooled augmented launch interval fit: models missing".to_string()
            })?;
        let plain = named_fit(
            kinds.get("launch_interval").and_then(|k| k.get("global")),
            &FEATURES,
        )
        .map_err(|e| format!("no pooled augmented launch interval fit: {e}"))?;
        let mut values = [0.0f64; 11];
        let set = |values: &mut [f64; 11], name: &str, weight: f64| {
            if let Some(i) = COMPOSED_FEATURES.iter().position(|n| *n == name) {
                values[i] = weight;
            }
        };
        let composed = model
            .get("composed")
            .and_then(|c| c.get(FAMILY))
            .and_then(|f| f.get("global"));
        match usable_fit(composed).filter(|f| subset(f, &COMPOSED_FEATURES)) {
            Some(fit) => {
                for (name, weight) in &fit {
                    set(&mut values, name.as_str(), *weight);
                }
                sources.insert("launch_interval".into(), json!("composed"));
            }
            None if subset(&plain, &FEATURES) => {
                for (name, weight) in &plain {
                    set(&mut values, name.as_str(), *weight);
                }
                sources.insert("launch_interval".into(), json!("plain"));
            }
            None => return Err("launch interval fit has unknown features".into()),
        }
        // (storage, host) pairs: plain and next-pass.
        for (storage, host) in [(5usize, 4usize), (10, 9)] {
            let name = COMPOSED_FEATURES[storage].to_string();
            if values[storage] == 0.0 {
                values[storage] = STORAGE_PRIOR * values[host];
                sources.insert(name, json!("prior_2x_host"));
            } else {
                sources.insert(name, json!("fit"));
            }
        }
        let eft = model.get("eft").and_then(Value::as_object);
        let in_eft = |key: &str| eft.is_some_and(|e| e.contains_key(key));
        let done_mean = holdout_mean(kinds.get("last_launch_to_done"));
        let interval_mean = holdout_mean(kinds.get("launch_interval"));
        let tau = if let Some(tau) = finite(eft.and_then(|e| e.get("tau_s"))) {
            sources.insert("tau".into(), json!("explicit"));
            tau
        } else if let (Some(done), Some(interval)) = (done_mean, interval_mean) {
            sources.insert("tau".into(), json!("holdout_means"));
            done - interval
        } else {
            match named_fit(
                kinds.get("last_launch_to_done").and_then(|k| k.get("global")),
                &FEATURES,
            ) {
                Ok(done) => {
                    sources.insert("tau".into(), json!("intercepts"));
                    lookup(&done, "intercept").unwrap_or(0.0)
                        - lookup(&plain, "intercept").unwrap_or(0.0)
                }
                Err(_) => {
                    sources.insert("tau".into(), json!("none"));
                    0.0
                }
            }
        };
        let timing = model
            .get("timing")
            .and_then(|t| t.get("global"))
            .filter(|t| t.is_object());
        let rx_value = if in_eft("rx") {
            eft.and_then(|e| e.get("rx"))
        } else {
            timing.and_then(|t| t.get("rx"))
        };
        let rx = rx_value
            .and_then(Value::as_array)
            .filter(|a| a.len() == 2)
            .and_then(|a| Some((finite(Some(&a[0]))?, finite(Some(&a[1]))?)));
        let (rx_a, rx_b) = match rx {
            Some(rx) => {
                let source = if in_eft("rx") { "eft" } else { "timing" };
                sources.insert("rx".into(), json!(source));
                rx
            }
            None => {
                sources.insert("rx".into(), json!("default"));
                DEFAULT_RX
            }
        };
        let idle_value = if in_eft("idle_delay_s") {
            eft.and_then(|e| e.get("idle_delay_s"))
        } else {
            timing.and_then(|t| t.get("idle_delay"))
        };
        let idle_delay = match finite(idle_value).filter(|x| *x >= 0.0) {
            Some(idle) => {
                let source = if in_eft("idle_delay_s") { "eft" } else { "timing" };
                sources.insert("idle_delay".into(), json!(source));
                idle
            }
            None => {
                sources.insert("idle_delay".into(), json!("default"));
                DEFAULT_IDLE_DELAY
            }
        };
        let service = kinds.get(SERVICE_KIND).and_then(|k| k.get("global"));
        let service_all = vector(service);
        let (mut split, mut small, mut large) = (None, None, None);
        if service_all.is_some() {
            let classes = service.and_then(|s| s.get("classes"));
            if let Some(tokens) = nonneg_int(classes.and_then(|c| c.get("split_tokens"))) {
                split = Some(tokens);
                small = vector(classes.and_then(|c| c.get("small")));
                large = vector(classes.and_then(|c| c.get("large")));
            }
        }
        let service_source = if service_all.is_some() { "fit" } else { "none" };
        sources.insert("service".into(), json!(service_source));
        let sha256 = match sha256 {
            Some(sha) => {
                sources.insert("sha256".into(), json!("file"));
                sha
            }
            None => {
                sources.insert("sha256".into(), json!("canonical_json"));
                crate::sha256_hex(python_json(model).as_bytes())
            }
        };
        Ok(Self {
            b0: values[0],
            b_new: values[1],
            b_pairs: values[2],
            b_bs: values[3],
            b_host: values[4],
            b_stor: values[5],
            b_fin: values[6],
            b_nbs: values[7],
            b_nhr: values[8],
            b_nhost: values[9],
            b_nstor: values[10],
            tau,
            rx_a,
            rx_b,
            idle_delay,
            service_split_tokens: split,
            service_all,
            service_small: small,
            service_large: large,
            sha256,
            sources,
        })
    }

    /// The values as JSON (Python `EftModel.record()`).
    pub fn record(&self) -> Value {
        json!({
            "b0": self.b0,
            "b_new": self.b_new,
            "b_pairs": self.b_pairs,
            "b_bs": self.b_bs,
            "b_host": self.b_host,
            "b_stor": self.b_stor,
            "b_fin": self.b_fin,
            "b_nbs": self.b_nbs,
            "b_nhr": self.b_nhr,
            "b_nhost": self.b_nhost,
            "b_nstor": self.b_nstor,
            "tau": self.tau,
            "rx_a": self.rx_a,
            "rx_b": self.rx_b,
            "idle_delay": self.idle_delay,
            "service_split_tokens": self.service_split_tokens,
            "service_all": self.service_all.map(|v| v.to_vec()),
            "service_small": self.service_small.map(|v| v.to_vec()),
            "service_large": self.service_large.map(|v| v.to_vec()),
            "sha256": self.sha256,
            "sources": self.sources,
        })
    }
}

/// Read a frozen model file; the digest is that of the file bytes.
pub fn load_model(path: &std::path::Path) -> Result<EftModel, String> {
    let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let model: Value =
        serde_json::from_slice(&data).map_err(|e| format!("{}: {e}", path.display()))?;
    EftModel::from_json(&model, Some(crate::sha256_hex(&data)))
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Python `json.dumps(value, sort_keys=True, separators=(",", ":"))`
/// (ensure_ascii on, floats in `repr` form).
fn python_json(value: &Value) -> String {
    let mut out = String::new();
    write_python_json(value, &mut out);
    out
}

fn write_python_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                out.push_str(&python_float_repr(n.as_f64().unwrap_or(f64::NAN)));
            }
        }
        Value::String(s) => write_python_str(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_python_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_python_str(key, out);
                out.push(':');
                write_python_json(&map[key], out);
            }
            out.push('}');
        }
    }
}

fn write_python_str(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            ' '..='~' => out.push(ch),
            _ => {
                let mut units = [0u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

/// Python `repr(float)`: shortest round-trip digits; scientific form when the
/// decimal exponent is below -4 or at least 16 (`1e-05`, `1e+16`), else
/// fixed form with at least one fractional digit (`0.05`, `100.0`).
pub(crate) fn python_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return String::from(if x > 0.0 { "Infinity" } else { "-Infinity" });
    }
    if x == 0.0 {
        return String::from(if x.is_sign_negative() { "-0.0" } else { "0.0" });
    }
    let sci = format!("{:e}", x.abs());
    let (mantissa, exponent) = sci.split_once('e').expect("LowerExp has an exponent");
    let exponent: i64 = exponent.parse().expect("integer exponent");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let sign = if x < 0.0 { "-" } else { "" };
    if !(-4..16).contains(&exponent) {
        let mut text = digits[..1].to_string();
        if digits.len() > 1 {
            text.push('.');
            text.push_str(&digits[1..]);
        }
        let exp_sign = if exponent < 0 { '-' } else { '+' };
        return format!("{sign}{text}e{exp_sign}{:02}", exponent.abs());
    }
    if exponent < 0 {
        let zeros = "0".repeat((-exponent - 1) as usize);
        return format!("{sign}0.{zeros}{digits}");
    }
    let point = exponent as usize + 1;
    if digits.len() <= point {
        let zeros = "0".repeat(point - digits.len());
        format!("{sign}{digits}{zeros}.0")
    } else {
        format!("{sign}{}.{}", &digits[..point], &digits[point..])
    }
}

// ---------------------------------------------------------------------------
// Tuning values.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PullIssue {
    Arrival,
    Jit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PullTier {
    Max,
    Cpu,
}

/// All tuning values of EFT-0+, with the reference defaults (seconds,
/// tokens). Python `EftKnobs`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EftKnobs {
    pub gamma: f64,
    pub margin_abs_s: f64,
    pub margin_rel: f64,
    pub pull_margin_s: f64,
    pub pull_a_s: f64,
    pub pull_b_s_per_100k: f64,
    pub pull_rho_s: f64,
    pub pull_e_s: f64,
    pub pull_enabled: bool,
    pub pull_issue: PullIssue,
    pub jit_lead_requests: i64,
    pub jit_lead_tokens: i64,
    pub pull_max_wait_s: f64,
    pub pull_tier: PullTier,
    pub pull_min_tokens: i64,
    pub pull_min_incremental: i64,
    pub evict_weight: f64,
    pub evict_median_prefix_tokens: i64,
    pub evict_tau_tok_s: Option<f64>,
    pub retire_old_copy: bool,
    pub session_window_s: f64,
    pub pingpong_s: f64,
    pub pingpong_window_s: f64,
    pub gate_wait_s: f64,
    pub max_view_age_s: f64,
}

impl Default for EftKnobs {
    fn default() -> Self {
        Self {
            gamma: 0.5,
            margin_abs_s: 1.0,
            margin_rel: 0.10,
            pull_margin_s: 0.10,
            pull_a_s: 0.8,
            pull_b_s_per_100k: 0.30,
            pull_rho_s: 0.6,
            pull_e_s: 0.3,
            pull_enabled: true,
            pull_issue: PullIssue::Arrival,
            jit_lead_requests: 4,
            jit_lead_tokens: 131_072,
            pull_max_wait_s: 30.0,
            pull_tier: PullTier::Max,
            pull_min_tokens: 256,
            pull_min_incremental: 256,
            evict_weight: 0.0,
            evict_median_prefix_tokens: 65_536,
            evict_tau_tok_s: None,
            retire_old_copy: true,
            session_window_s: 600.0,
            pingpong_s: 2.0,
            pingpong_window_s: 60.0,
            gate_wait_s: 0.0,
            max_view_age_s: 1.5,
        }
    }
}

impl EftKnobs {
    fn floats(&self) -> [(&'static str, f64); 16] {
        [
            ("gamma", self.gamma),
            ("margin_abs_s", self.margin_abs_s),
            ("margin_rel", self.margin_rel),
            ("pull_margin_s", self.pull_margin_s),
            ("pull_a_s", self.pull_a_s),
            ("pull_b_s_per_100k", self.pull_b_s_per_100k),
            ("pull_rho_s", self.pull_rho_s),
            ("pull_e_s", self.pull_e_s),
            ("pull_max_wait_s", self.pull_max_wait_s),
            ("evict_weight", self.evict_weight),
            ("session_window_s", self.session_window_s),
            ("pingpong_s", self.pingpong_s),
            ("pingpong_window_s", self.pingpong_window_s),
            ("gate_wait_s", self.gate_wait_s),
            ("max_view_age_s", self.max_view_age_s),
            ("evict_tau_tok_s", self.evict_tau_tok_s.unwrap_or(0.0)),
        ]
    }

    /// Error for a value that the rule cannot use (Python `check`).
    pub fn check(&self) -> Result<(), String> {
        for (name, value) in self.floats() {
            if !(value.is_finite() && value >= 0.0) {
                return Err(format!("knob {name} must be a finite number >= 0"));
            }
        }
        for (name, value) in [
            ("jit_lead_requests", self.jit_lead_requests),
            ("jit_lead_tokens", self.jit_lead_tokens),
            ("pull_min_tokens", self.pull_min_tokens),
            ("pull_min_incremental", self.pull_min_incremental),
            ("evict_median_prefix_tokens", self.evict_median_prefix_tokens),
        ] {
            if value < 0 {
                return Err(format!("knob {name} must be a finite number >= 0"));
            }
        }
        Ok(())
    }

    /// The defaults with the changes of a JSON object (golden `knobs`).
    pub fn from_overrides(changes: &Value) -> Result<Self, String> {
        let mut k = Self::default();
        let Some(changes) = changes.as_object() else {
            return if changes.is_null() {
                Ok(k)
            } else {
                Err("knob changes must be an object".into())
            };
        };
        for (name, value) in changes {
            let float = || {
                value
                    .as_f64()
                    .ok_or_else(|| format!("knob {name} must be a number"))
            };
            let int = || {
                value
                    .as_i64()
                    .ok_or_else(|| format!("knob {name} must be an integer"))
            };
            let boolean = || {
                value
                    .as_bool()
                    .ok_or_else(|| format!("knob {name} must be True or False"))
            };
            match name.as_str() {
                "gamma" => k.gamma = float()?,
                "margin_abs_s" => k.margin_abs_s = float()?,
                "margin_rel" => k.margin_rel = float()?,
                "pull_margin_s" => k.pull_margin_s = float()?,
                "pull_a_s" => k.pull_a_s = float()?,
                "pull_b_s_per_100k" => k.pull_b_s_per_100k = float()?,
                "pull_rho_s" => k.pull_rho_s = float()?,
                "pull_e_s" => k.pull_e_s = float()?,
                "pull_enabled" => k.pull_enabled = boolean()?,
                "pull_issue" => {
                    k.pull_issue = match value.as_str() {
                        Some("arrival") => PullIssue::Arrival,
                        Some("jit") => PullIssue::Jit,
                        _ => return Err("pull_issue must be one of arrival, jit".into()),
                    }
                }
                "jit_lead_requests" => k.jit_lead_requests = int()?,
                "jit_lead_tokens" => k.jit_lead_tokens = int()?,
                "pull_max_wait_s" => k.pull_max_wait_s = float()?,
                "pull_tier" => {
                    k.pull_tier = match value.as_str() {
                        Some("max") => PullTier::Max,
                        Some("cpu") => PullTier::Cpu,
                        _ => return Err("pull_tier must be one of max, cpu".into()),
                    }
                }
                "pull_min_tokens" => k.pull_min_tokens = int()?,
                "pull_min_incremental" => k.pull_min_incremental = int()?,
                "evict_weight" => k.evict_weight = float()?,
                "evict_median_prefix_tokens" => k.evict_median_prefix_tokens = int()?,
                "evict_tau_tok_s" => {
                    k.evict_tau_tok_s = if value.is_null() {
                        None
                    } else {
                        Some(float()?)
                    }
                }
                "retire_old_copy" => k.retire_old_copy = boolean()?,
                "session_window_s" => k.session_window_s = float()?,
                "pingpong_s" => k.pingpong_s = float()?,
                "pingpong_window_s" => k.pingpong_window_s = float()?,
                "gate_wait_s" => k.gate_wait_s = float()?,
                "max_view_age_s" => k.max_view_age_s = float()?,
                other => return Err(format!("unknown knob {other}")),
            }
        }
        k.check()?;
        Ok(k)
    }

    /// The values as JSON (Python `asdict(knobs)`).
    pub fn record(&self) -> Value {
        json!({
            "gamma": self.gamma,
            "margin_abs_s": self.margin_abs_s,
            "margin_rel": self.margin_rel,
            "pull_margin_s": self.pull_margin_s,
            "pull_a_s": self.pull_a_s,
            "pull_b_s_per_100k": self.pull_b_s_per_100k,
            "pull_rho_s": self.pull_rho_s,
            "pull_e_s": self.pull_e_s,
            "pull_enabled": self.pull_enabled,
            "pull_issue": match self.pull_issue {
                PullIssue::Arrival => "arrival",
                PullIssue::Jit => "jit",
            },
            "jit_lead_requests": self.jit_lead_requests,
            "jit_lead_tokens": self.jit_lead_tokens,
            "pull_max_wait_s": self.pull_max_wait_s,
            "pull_tier": match self.pull_tier {
                PullTier::Max => "max",
                PullTier::Cpu => "cpu",
            },
            "pull_min_tokens": self.pull_min_tokens,
            "pull_min_incremental": self.pull_min_incremental,
            "evict_weight": self.evict_weight,
            "evict_median_prefix_tokens": self.evict_median_prefix_tokens,
            "evict_tau_tok_s": self.evict_tau_tok_s,
            "retire_old_copy": self.retire_old_copy,
            "session_window_s": self.session_window_s,
            "pingpong_s": self.pingpong_s,
            "pingpong_window_s": self.pingpong_window_s,
            "gate_wait_s": self.gate_wait_s,
            "max_view_age_s": self.max_view_age_s,
        })
    }
}

// ---------------------------------------------------------------------------
// Engine snapshot view.
// ---------------------------------------------------------------------------

/// One snapshot row as row work inputs: u to compute, q cached prefix at the
/// next chunk, h host restore and s storage (pulled) tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowWork {
    pub rid: String,
    pub u: i64,
    pub q: i64,
    pub h: i64,
    pub s: i64,
    pub started: bool,
}

/// A row whose pull is pending or in flight.
#[derive(Debug, Clone, PartialEq)]
pub struct PullRow {
    pub rid: Value,
    pub tokens: i64,
    pub age_s: f64,
}

/// The latest in-flight pass: its launch age and FEATURES[1..] values (raw).
#[derive(Debug, Clone, PartialEq)]
pub struct InflightPass {
    pub launch_age_s: Option<f64>,
    pub features: Vec<Value>,
}

/// The part of the EFT view that does not depend on the decision time:
/// one parsed `loads[0].prefill_routing` snapshot. The router parses
/// it once per poll, off the cost-book lock; [`View::at`] adds the age.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotView {
    pub epoch: Value,
    pub sequence: Value,
    pub engine_age_s: Option<f64>,
    pub page: i64,
    pub chunk: i64,
    pub rows: Vec<RowWork>,
    /// RIDs of the partial, waiting, bootstrap and in-flight request rows
    /// (the rid set of the reference; [`View::record`] shows it).
    pub rids: BTreeSet<String>,
    /// RIDs that the latest in-flight pass lists (`request_ids`), when R0
    /// prices that pass (its launch age is known). The rows do not show a
    /// request whose last chunk is in that pass: the engine adds it to the
    /// in-flight request queue only after the pass result. Not part of the
    /// reference view; the router leaves these RIDs out of its outstanding
    /// rows, because R0 already prices their work.
    pub pass_rids: BTreeSet<String>,
    pub pulls: Vec<PullRow>,
    pub inflight: Option<InflightPass>,
    pub gate_closed: bool,
    pub idle: bool,
    pub rows_intent_skipped: u64,
    /// Rows priced at their full input length because the engine gave no
    /// usable prefix data (`queue_compute_unknown_cold` or
    /// `queue_prefix_invalid`), and their tokens. For the router's trace
    /// only; the reference flags them in `degraded`.
    pub cold_rows: u64,
    pub cold_tokens: i64,
    pub degraded: BTreeSet<String>,
    /// The first failed usability rule other than the age rule.
    pub static_unusable_reason: Option<&'static str>,
}

struct Work {
    rid: String,
    prefix: i64,
    remaining: i64,
    host: Option<i64>,
    storage: Option<i64>,
    started: bool,
    /// Priced at the full input length: no usable prefix data.
    cold: bool,
}

/// Python `v in (True, False)` (also true for the numbers 0 and 1).
fn readiness_known(v: Option<&Value>) -> bool {
    match v {
        Some(Value::Bool(_)) => true,
        Some(Value::Number(n)) => matches!(n.as_f64(), Some(x) if x == 0.0 || x == 1.0),
        _ => false,
    }
}

/// Python `prefill_completion.queue_work`: one snapshot queue row as work,
/// or None when it is not ahead of the request.
fn queue_work(
    row: &Value,
    page: i64,
    chunk: i64,
    degraded: &mut BTreeSet<String>,
    skipped: &mut u64,
) -> Option<Work> {
    let Some(rid) = row.get("rid").and_then(Value::as_str) else {
        flag(degraded, "queue_rid_unknown");
        return None;
    };
    let state = row.get("state").filter(|s| s.is_object());
    let st = |key: &str| state.and_then(|s| s.get(key));
    if READINESS.iter().any(|f| !readiness_known(st(f))) {
        flag(degraded, "queue_readiness_unknown");
    }
    let Some(n) = nonneg_int(row.get("input_tokens")) else {
        flag(degraded, "queue_row_size_unknown");
        return None;
    };
    let mut cold = false;
    let remaining = match nonneg_int(row.get("remaining_compute_tokens")) {
        Some(remaining) => remaining,
        None => {
            // Rows received before their prefix lookup report no remaining compute.
            let device = nonneg_int(row.get("device_prefix_tokens"));
            let host = nonneg_int(row.get("host_prefix_tokens"));
            match (device, host) {
                (Some(device), Some(host)) if device + host <= n => {
                    flag(degraded, "queue_compute_from_prefixes");
                    n - device - host
                }
                _ => {
                    flag(degraded, "queue_compute_unknown_cold");
                    cold = true;
                    n
                }
            }
        }
    };
    let (mut prefix, mut remaining) = (n - remaining, remaining);
    if prefix < 0 {
        flag(degraded, "queue_prefix_invalid");
        prefix = 0;
        remaining = n;
        cold = true;
    } else if prefix % page != 0 {
        flag(degraded, "queue_prefix_unaligned");
    }
    let started = matches!(
        row.get("phase").and_then(Value::as_str),
        Some("partial" | "parked")
    );
    let host = if started {
        Some(0)
    } else {
        nonneg_int(row.get("host_prefix_tokens"))
    };
    let storage = if started {
        Some(0)
    } else {
        nonneg_int(row.get("storage_prefix_tokens"))
    };
    let gate_subject = !started && !is_true(st("bound")) && !is_true(st("launched_unbound"));
    if gate_subject && !is_true(st("intent_start_ok")) && remaining > chunk {
        // Engine start-permission rule: uncached part over one chunk is skipped.
        *skipped += 1;
        return None;
    }
    Some(Work {
        rid: rid.to_string(),
        prefix,
        remaining,
        host,
        storage,
        started,
        cold,
    })
}

/// Python `raw_is_idle`: available, in-flight passes known, and no waiting,
/// partial, bootstrapping, current or in-flight work.
fn raw_is_idle(raw: &Value) -> bool {
    if !raw.is_object()
        || !is_true(raw.get("available"))
        || !is_true(raw.get("inflight_batches_known"))
    {
        return false;
    }
    ![
        "waiting",
        "partial",
        "bootstrap",
        "current_batch",
        "inflight_batches",
    ]
    .iter()
    .any(|name| truthy(raw.get(*name)))
}

fn policy_value(
    policy: Option<&Value>,
    name: &str,
    default: i64,
    degraded: &mut BTreeSet<String>,
) -> i64 {
    match nonneg_int(policy.and_then(|p| p.get(name))).filter(|v| *v > 0) {
        Some(value) => value,
        None => {
            flag(degraded, &format!("policy_default_{name}"));
            default
        }
    }
}

/// The usability rules of the EFT view except the age rule (Python
/// `_unusable_reason`, in its order).
fn static_unusable_reason(
    raw: &Value,
    dp_size: usize,
    expected_epoch: Option<&str>,
) -> Option<&'static str> {
    let fcfs = raw
        .get("policy")
        .and_then(|p| p.get("schedule_policy"))
        .and_then(Value::as_str)
        == Some("fcfs");
    let omitted_zero = match raw.get("omitted_waiting") {
        Some(Value::Number(n)) => n.as_f64() == Some(0.0),
        Some(Value::Bool(false)) => true,
        _ => false,
    };
    let epoch_mismatch = expected_epoch
        .is_some_and(|e| raw.get("epoch").and_then(Value::as_str) != Some(e));
    [
        (!is_true(raw.get("available")), "unavailable"),
        (is_true(raw.get("stale")), "stale"),
        (is_true(raw.get("truncated")) && !omitted_zero, "truncated"),
        (!is_true(raw.get("inflight_batches_known")), "inflight_unknown"),
        (!fcfs, "policy_not_fcfs"),
        (dp_size != 1, "dp_not_1"),
        (epoch_mismatch, "epoch_mismatch"),
    ]
    .into_iter()
    .find(|(failed, _)| *failed)
    .map(|(_, reason)| reason)
}

impl SnapshotView {
    /// Python `view_from_snapshot` without the age: parse one snapshot.
    ///
    /// Rows (partial, waiting, bootstrap, in this order) go through the
    /// `queue_work` rules. A row with an unknown storage prefix uses its
    /// planned pull tokens, else 0; an unknown host prefix is 0 (both
    /// flagged). Rows with a pull pending or in flight also go to `pulls`.
    pub fn parse(raw: &Value, dp_size: usize, expected_epoch: Option<&str>) -> Self {
        let empty = Value::Object(Map::new());
        let raw = if raw.is_object() { raw } else { &empty };
        let mut degraded = BTreeSet::new();
        let policy = raw.get("policy").filter(|p| p.is_object());
        let page = policy_value(policy, "page_size", DEFAULT_PAGE, &mut degraded);
        let chunk = policy_value(policy, "chunk_tokens", DEFAULT_CHUNK, &mut degraded);
        let engine_age_s = finite(raw.get("age_s")).filter(|a| *a >= 0.0);
        let mut skipped = 0u64;
        let mut sources: Vec<&Value> = Vec::new();
        if let Some(partial) = raw.get("partial").filter(|p| p.is_object()) {
            sources.push(partial);
        }
        for name in ["waiting", "bootstrap"] {
            if let Some(rows) = raw.get(name).and_then(Value::as_array) {
                sources.extend(rows.iter().filter(|r| r.is_object()));
            }
        }
        let mut rows = Vec::new();
        let mut pulls = Vec::new();
        let mut rids = BTreeSet::new();
        let mut seen = BTreeSet::new();
        let (mut cold_rows, mut cold_tokens) = (0u64, 0i64);
        for row in sources {
            if let Some(rid) = row.get("rid").and_then(Value::as_str) {
                rids.insert(rid.to_string());
            }
            let state = row.get("state").filter(|s| s.is_object());
            let planned = nonneg_int(row.get("plan").and_then(|p| p.get("pull_tokens")));
            if is_true(state.and_then(|s| s.get("pull_pending")))
                || is_true(state.and_then(|s| s.get("pull_inflight")))
            {
                if planned.is_none() {
                    flag(&mut degraded, "pull_tokens_unknown");
                }
                pulls.push(PullRow {
                    rid: row.get("rid").cloned().unwrap_or(Value::Null),
                    tokens: planned.unwrap_or(0),
                    age_s: finite(row.get("pull_phase_age_s"))
                        .filter(|a| *a >= 0.0)
                        .unwrap_or(0.0),
                });
            }
            let Some(work) = queue_work(row, page, chunk, &mut degraded, &mut skipped) else {
                continue;
            };
            if seen.contains(&work.rid) || work.remaining == 0 {
                continue;
            }
            seen.insert(work.rid.clone());
            let storage = match work.storage {
                Some(storage) => storage,
                None => {
                    flag(&mut degraded, "row_storage_unknown");
                    planned.unwrap_or(0)
                }
            };
            let host = match work.host {
                Some(host) => host,
                None => {
                    flag(&mut degraded, "row_host_unknown");
                    0
                }
            };
            if work.cold {
                cold_rows += 1;
                cold_tokens += work.remaining;
            }
            rows.push(RowWork {
                rid: work.rid,
                u: work.remaining,
                q: work.prefix,
                h: host,
                s: storage,
                started: work.started,
            });
        }
        if let Some(inflight) = raw.get("inflight_requests").and_then(Value::as_array) {
            for row in inflight {
                if let Some(rid) = row.get("rid").and_then(Value::as_str) {
                    rids.insert(rid.to_string());
                }
            }
        }
        let last_pass = raw
            .get("inflight_batches")
            .and_then(Value::as_array)
            .and_then(|batches| batches.last())
            .filter(|last| last.is_object());
        let inflight = last_pass.map(|last| {
            let features = last.get("features").filter(|f| f.is_object());
            InflightPass {
                launch_age_s: finite(last.get("launch_age_s")).filter(|a| *a >= 0.0),
                features: FEATURES[1..]
                    .iter()
                    .map(|name| {
                        features
                            .and_then(|f| f.get(*name))
                            .cloned()
                            .unwrap_or(Value::Null)
                    })
                    .collect(),
            }
        });
        // [`drain`] prices this pass (R0) only when its launch age is known.
        let pass_rids: BTreeSet<String> = match (last_pass, &inflight) {
            (Some(last), Some(pass)) if pass.launch_age_s.is_some() => last
                .get("request_ids")
                .and_then(Value::as_array)
                .map(|ids| {
                    ids.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default(),
            _ => BTreeSet::new(),
        };
        Self {
            epoch: raw.get("epoch").cloned().unwrap_or(Value::Null),
            sequence: raw.get("sequence").cloned().unwrap_or(Value::Null),
            engine_age_s,
            page,
            chunk,
            rows,
            rids,
            pass_rids,
            pulls,
            inflight,
            gate_closed: is_true(
                raw.get("admission_resources")
                    .and_then(|r| r.get("park_gate_closed_last_scan")),
            ),
            idle: raw_is_idle(raw),
            rows_intent_skipped: skipped,
            cold_rows,
            cold_tokens,
            degraded,
            static_unusable_reason: static_unusable_reason(raw, dp_size, expected_epoch),
        }
    }

    /// A view with no engine rows, no in-flight pass and age 0, with the
    /// given page and chunk sizes. [`drain`] of it and some outstanding rows
    /// is the step-model price of those rows alone: the router's fallback
    /// wait for a P without a usable snapshot.
    pub fn empty(page: i64, chunk: i64) -> Self {
        Self {
            epoch: Value::Null,
            sequence: Value::Null,
            engine_age_s: Some(0.0),
            page,
            chunk,
            rows: Vec::new(),
            rids: BTreeSet::new(),
            pass_rids: BTreeSet::new(),
            pulls: Vec::new(),
            inflight: None,
            gate_closed: false,
            idle: true,
            rows_intent_skipped: 0,
            cold_rows: 0,
            cold_tokens: 0,
            degraded: BTreeSet::new(),
            static_unusable_reason: None,
        }
    }
}

/// A parsed snapshot at one decision: `age_s` = (decision - poll sent) +
/// engine `age_s`, an upper bound of the view age (None when the engine age
/// is unknown).
#[derive(Debug, Clone, Copy)]
pub struct View<'a> {
    pub snap: &'a SnapshotView,
    pub age_s: Option<f64>,
    pub unusable_reason: Option<&'static str>,
}

impl<'a> View<'a> {
    /// `since_poll_sent_s`: decision time minus the time the poll was sent.
    pub fn at(snap: &'a SnapshotView, since_poll_sent_s: f64) -> Self {
        let age_s = snap.engine_age_s.map(|engine| since_poll_sent_s + engine);
        let unusable_reason = snap
            .static_unusable_reason
            .or(if age_s.is_none() {
                Some("age_unknown")
            } else {
                None
            });
        Self {
            snap,
            age_s,
            unusable_reason,
        }
    }

    pub fn usable(&self) -> bool {
        self.unusable_reason.is_none()
    }

    /// The view as JSON (Python `view_from_snapshot` result).
    pub fn record(&self) -> Value {
        let s = self.snap;
        let features: Map<String, Value> = s
            .inflight
            .as_ref()
            .map(|i| {
                FEATURES[1..]
                    .iter()
                    .zip(&i.features)
                    .map(|(name, value)| (name.to_string(), value.clone()))
                    .collect()
            })
            .unwrap_or_default();
        json!({
            "key": [s.epoch, s.sequence],
            "engine_age_s": s.engine_age_s,
            "age_s": self.age_s,
            "usable": self.usable(),
            "unusable_reason": self.unusable_reason,
            "page": s.page,
            "chunk": s.chunk,
            "rows": s.rows.iter().map(|r| json!({
                "rid": r.rid, "u": r.u, "q": r.q, "h": r.h, "s": r.s, "started": r.started,
            })).collect::<Vec<_>>(),
            "rids": s.rids,
            "pulls": s.pulls.iter().map(|p| json!({
                "rid": p.rid, "tokens": p.tokens, "age_s": p.age_s,
            })).collect::<Vec<_>>(),
            "inflight": s.inflight.as_ref().map(|i| json!({
                "launch_age_s": i.launch_age_s,
                "features": features,
            })),
            "gate_closed": s.gate_closed,
            "idle": s.idle,
            "rows_intent_skipped": s.rows_intent_skipped,
            "degraded": s.degraded,
        })
    }
}

// ---------------------------------------------------------------------------
// Work, drain and forecast.
// ---------------------------------------------------------------------------

/// W(u, q, h, s): the launch-interval work of one row, without b0. The terms
/// are added in the order of the reference.
pub fn row_work(m: &EftModel, u: i64, q: i64, h: i64, s: i64, chunk: i64) -> f64 {
    let pairs = u * q + u * (u + 1) / 2;
    let mut total = m.b_new * u as f64;
    total += m.b_pairs * pairs as f64;
    total += (m.b_host + m.b_nhost) * h as f64;
    total += (m.b_stor + m.b_nstor) * s as f64;
    total += (m.b_bs + m.b_nbs) * ceil_div(u, chunk) as f64;
    total += m.b_fin;
    total += m.b_nhr * if h > 0 { 1.0 } else { 0.0 };
    total
}

/// Launch interval of an in-flight pass from its features (missing: 0).
fn interval_plain(m: &EftModel, pass: &InflightPass) -> f64 {
    let weights = [m.b_new, m.b_pairs, m.b_bs, m.b_host, m.b_stor];
    let mut total = m.b0;
    for (weight, value) in weights.iter().zip(&pass.features) {
        let value = finite(Some(value)).filter(|v| *v >= 0.0).unwrap_or(0.0);
        total += weight * value;
    }
    total
}

/// X(Q) = a_p + b_p * Q / 1e5: the serial span of one pull.
pub fn pull_span(knobs: &EftKnobs, tokens: i64) -> f64 {
    knobs.pull_a_s + knobs.pull_b_s_per_100k * tokens as f64 / 100_000.0
}

/// One router (cost-book) row as given: the inputs of Python `_outstanding`.
#[derive(Debug, Clone, PartialEq)]
pub struct OutstandingInput {
    pub rid: Value,
    /// Remaining uncached tokens; None or 0 leaves the row out.
    pub uncached_tokens: Option<i64>,
    pub expected_hit: i64,
    pub local_hit: i64,
    pub gpu_hit: i64,
    /// A pull hint was sent with the request.
    pub hinted: bool,
}

impl OutstandingInput {
    /// A golden-file row: `uncached_tokens`, `expected_hit` (null: 0),
    /// `local_hit` (null: expected), `gpu_hit` (null: 0), `hinted`.
    pub fn from_json(row: &Value) -> Self {
        let expected_hit = row.get("expected_hit").and_then(Value::as_i64).unwrap_or(0);
        Self {
            rid: row.get("rid").cloned().unwrap_or(Value::Null),
            uncached_tokens: nonneg_int(row.get("uncached_tokens")),
            expected_hit,
            local_hit: row
                .get("local_hit")
                .and_then(Value::as_i64)
                .unwrap_or(expected_hit),
            gpu_hit: row.get("gpu_hit").and_then(Value::as_i64).unwrap_or(0),
            hinted: is_true(row.get("hinted")),
        }
    }
}

/// A dispatched row that the snapshot does not show, as row work inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct OutstandingRow {
    pub rid: Value,
    pub u: i64,
    pub q: i64,
    pub h: i64,
    pub s: i64,
    pub hinted: bool,
}

/// Python `_outstanding`: rows whose rid the snapshot lists, and rows with
/// nothing to compute, are left out. h = local - gpu; s = expected - local
/// when a hint was sent, else 0.
pub fn outstanding_rows(rows: &[OutstandingInput], rids: &BTreeSet<String>) -> Vec<OutstandingRow> {
    rows.iter()
        .filter(|r| !r.rid.as_str().is_some_and(|rid| rids.contains(rid)))
        .filter_map(|r| {
            let u = r.uncached_tokens.filter(|u| *u > 0)?;
            Some(OutstandingRow {
                rid: r.rid.clone(),
                u,
                q: r.expected_hit,
                h: 0.max(r.local_hit - r.gpu_hit),
                s: if r.hinted {
                    0.max(r.expected_hit - r.local_hit)
                } else {
                    0
                },
                hinted: r.hinted,
            })
        })
        .collect()
}

/// D_c, the lead drain, the pull backlog PB_c and their parts.
#[derive(Debug, Clone, PartialEq)]
pub struct Drain {
    /// D_c: seconds until the worker has done the work ahead of the request.
    pub d: f64,
    pub d_lead: f64,
    /// PB_c: seconds of serial pull work ahead.
    pub pb: f64,
    /// Rest of the latest in-flight pass.
    pub r0: f64,
    pub work_snapshot: f64,
    pub work_outstanding: f64,
    pub passes_snapshot: i64,
    pub passes_ahead: i64,
    pub sizes_ahead: Vec<i64>,
    /// No snapshot work and no outstanding row.
    pub idle: bool,
}

/// Python `drain`:
///
/// ```text
/// D_c = max(0, R0 + sum_S W + b0 P(S) - a) + sum_O W + b0 (P(S u O) - P(S))
/// ```
///
/// R0 = max(0, I(features of the latest in-flight pass) - launch age). The
/// lead drain removes the W of the trailing rows of S u O (FCFS order) that
/// fit in the jit lead window. PB_c = sum over pulling snapshot rows of
/// max(0, X(tokens) - pull age) + sum of X(s) over hinted outstanding rows.
pub fn drain(view: &View<'_>, out: &[OutstandingRow], m: &EftModel, knobs: &EftKnobs) -> Drain {
    let snap = view.snap;
    let (page, chunk) = (snap.page, snap.chunk);
    let mut r0 = 0.0;
    if let Some(pass) = &snap.inflight {
        if let Some(launch_age) = pass.launch_age_s {
            r0 = pymax(0.0, interval_plain(m, pass) - launch_age);
        }
    }
    let mut work_s = 0.0;
    for r in &snap.rows {
        work_s += row_work(m, r.u, r.q, r.h, r.s, chunk);
    }
    let mut work_o = 0.0;
    for r in out {
        work_o += row_work(m, r.u, r.q, r.h, r.s, chunk);
    }
    let passes_s = passes(snap.rows.iter().map(|r| r.u), page, chunk);
    let sizes_ahead: Vec<i64> = snap
        .rows
        .iter()
        .map(|r| r.u)
        .chain(out.iter().map(|r| r.u))
        .collect();
    let passes_so = passes(sizes_ahead.iter().copied(), page, chunk);
    let head = pymax(
        0.0,
        r0 + work_s + m.b0 * passes_s as f64 - view.age_s.unwrap_or(0.0),
    );
    let total = head + work_o + m.b0 * (passes_so - passes_s) as f64;
    let ahead: Vec<(i64, i64, i64, i64)> = snap
        .rows
        .iter()
        .map(|r| (r.u, r.q, r.h, r.s))
        .chain(out.iter().map(|r| (r.u, r.q, r.h, r.s)))
        .collect();
    let (mut lead_work, mut count, mut tokens) = (0.0, 0i64, 0i64);
    for &(u, q, h, s) in ahead.iter().rev() {
        if count >= knobs.jit_lead_requests || tokens + u > knobs.jit_lead_tokens {
            break;
        }
        lead_work += row_work(m, u, q, h, s, chunk);
        count += 1;
        tokens += u;
    }
    let mut backlog = 0.0;
    for p in &snap.pulls {
        backlog += pymax(0.0, pull_span(knobs, p.tokens) - p.age_s);
    }
    for r in out {
        if r.hinted && r.s > 0 {
            backlog += pull_span(knobs, r.s);
        }
    }
    Drain {
        d: total,
        d_lead: pymax(0.0, total - lead_work),
        pb: backlog,
        r0,
        work_snapshot: work_s,
        work_outstanding: work_o,
        passes_snapshot: passes_s,
        passes_ahead: passes_so,
        sizes_ahead,
        idle: snap.idle && out.is_empty(),
    }
}

/// One variant of the routed request on a worker: u to compute, q cached
/// prefix, h host restore and s pulled tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub u: i64,
    pub q: i64,
    pub h: i64,
    pub s: i64,
}

impl Shape {
    /// The variant shapes of the reference (`_price_candidate`) for a
    /// request of `n` tokens with local hit `local`, GPU hit `gpu` and,
    /// optionally, a pullable hit `pull_hit` from a peer. The hits are
    /// page-floored (`floor_hit`). Returns (shape, pulled tokens); pulled is
    /// 0 for the local variant or when the pull gains no whole page.
    pub fn of(n: i64, local: i64, gpu: i64, pull_hit: Option<i64>, page: i64) -> (Self, i64) {
        let q = floor_hit(local, n, page);
        let gpu = (gpu.div_euclid(page) * page).min(q);
        let local_shape = Self {
            u: n - q,
            q,
            h: q - gpu,
            s: 0,
        };
        let Some(hit) = pull_hit else {
            return (local_shape, 0);
        };
        let q_pull = floor_hit(hit, n, page);
        let gained = q_pull - q;
        if gained <= 0 {
            return (local_shape, 0);
        }
        (
            Self {
                u: n - q_pull,
                q: q_pull,
                h: q - gpu,
                s: gained,
            },
            gained,
        )
    }
}

/// Svc = W(u, q, h, s) + b0 ceil(ceil_page(u) / C): the own service of the
/// request on the worker (the externality term of the reference).
pub fn service_seconds(m: &EftModel, shape: Shape, page: i64, chunk: i64) -> f64 {
    row_work(m, shape.u, shape.q, shape.h, shape.s, chunk)
        + m.b0 * ceil_div(ceil_page(shape.u, page), chunk) as f64
}

/// F(c, v) and its parts for one variant.
#[derive(Debug, Clone, PartialEq)]
pub struct Forecast {
    pub shape: Shape,
    pub rx: f64,
    pub gate: f64,
    pub ready: f64,
    pub w: f64,
    pub e_pull: f64,
    pub path: &'static str,
    pub pass_delta: Option<i64>,
    pub tau: f64,
    pub idle_delay: f64,
    pub service_first_s: Option<f64>,
    pub rest_s: Option<f64>,
    pub first_chunk: i64,
    pub f: f64,
    pub svc: f64,
}

fn service_fit(m: &EftModel, new_tokens: i64) -> Option<&[f64; 6]> {
    if let Some(split) = m.service_split_tokens {
        let part = if new_tokens <= split {
            &m.service_small
        } else {
            &m.service_large
        };
        if let Some(part) = part {
            return Some(part);
        }
    }
    m.service_all.as_ref()
}

/// Python `forecast`:
///
/// ```text
/// F = rx(n) + max(D_c + g_c, ready) + W_t + b0 (P(S u O u {t}) - P(S u O)) + tau + e_pull [pull]
/// ```
///
/// An idle worker with an idle service fit uses rx(n) + max(g_c, ready) +
/// idle_delay + service(first chunk) + [W(rest) + b0 P(rest) + tau if more
/// chunks follow] + e_pull [pull].
#[allow(clippy::too_many_arguments)]
pub fn forecast(
    m: &EftModel,
    knobs: &EftKnobs,
    n: i64,
    shape: Shape,
    ahead: &Drain,
    snap: &SnapshotView,
    ready: f64,
    pull: bool,
) -> Forecast {
    let (page, chunk) = (snap.page, snap.chunk);
    let Shape { u, q, h, s } = shape;
    let work = row_work(m, u, q, h, s, chunk);
    let gate = if snap.gate_closed && ceil_div(u, chunk) > 1 {
        knobs.gate_wait_s
    } else {
        0.0
    };
    let rx = m.rx_a + m.rx_b * n as f64;
    let extra = if pull { knobs.pull_e_s } else { 0.0 };
    let service = work + m.b0 * ceil_div(ceil_page(u, page), chunk) as f64;
    let first = if ceil_page(u, page) <= chunk {
        u
    } else {
        chunk.div_euclid(page) * page
    };
    let fit = if ahead.idle {
        service_fit(m, first)
    } else {
        None
    };
    let mut out = Forecast {
        shape,
        rx,
        gate,
        ready,
        w: work,
        e_pull: extra,
        path: "drain",
        pass_delta: None,
        tau: m.tau,
        idle_delay: 0.0,
        service_first_s: None,
        rest_s: None,
        first_chunk: first,
        f: 0.0,
        svc: service,
    };
    match fit {
        Some(fit) => {
            let values = [
                1.0,
                first as f64,
                (first * q + first * (first + 1) / 2) as f64,
                1.0,
                h as f64,
                s as f64,
            ];
            let mut alone = 0.0;
            for (weight, value) in fit.iter().zip(values) {
                alone += weight * value;
            }
            let alone = pymax(0.0, alone);
            let rest = u - first;
            let (mut tail, mut tau) = (0.0, 0.0);
            if rest > 0 {
                tail = row_work(m, rest, q + first, 0, 0, chunk);
                tail += m.b0 * passes([rest], page, chunk) as f64;
                tail += m.tau;
                tau = m.tau;
            }
            out.f = rx + pymax(gate, ready) + m.idle_delay + alone + tail + extra;
            out.path = "idle_service";
            out.tau = tau;
            out.idle_delay = m.idle_delay;
            out.service_first_s = Some(alone);
            out.rest_s = Some(tail);
        }
        None => {
            let delta = passes(ahead.sizes_ahead.iter().copied().chain([u]), page, chunk)
                - ahead.passes_ahead;
            let mut value = rx + pymax(ahead.d + gate, ready) + work + m.b0 * delta as f64 + m.tau;
            value += extra;
            out.f = value;
            out.pass_delta = Some(delta);
        }
    }
    out
}

impl Forecast {
    fn record(&self, score_us: i64) -> Map<String, Value> {
        let value = json!({
            "u": self.shape.u,
            "q": self.shape.q,
            "h": self.shape.h,
            "s": self.shape.s,
            "rx": self.rx,
            "gate": self.gate,
            "ready": self.ready,
            "W": self.w,
            "e_pull": self.e_pull,
            "path": self.path,
            "pass_delta": self.pass_delta,
            "tau": self.tau,
            "idle_delay": self.idle_delay,
            "service_first_s": self.service_first_s,
            "rest_s": self.rest_s,
            "first_chunk": self.first_chunk,
            "F": self.f,
            "Svc": self.svc,
            "score_us": score_us,
        });
        match value {
            Value::Object(map) => map,
            _ => unreachable!("json object"),
        }
    }
}

// ---------------------------------------------------------------------------
// Session terms and the full EFT-0+ decision (golden parity).
// ---------------------------------------------------------------------------

/// Python `SessionMap.context` result: the session inputs of one decision.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionContext {
    pub owner: Option<String>,
    pub last_len: i64,
    pub last_move_s: Option<f64>,
    /// R_x: tokens of the other recent sessions owned by x.
    pub occupancy: BTreeMap<String, i64>,
    /// K_x: host pool tokens of x (None: unknown).
    pub host_pool_tokens: BTreeMap<String, Option<i64>>,
}

impl SessionContext {
    /// None when the context is null, empty or has no session key.
    pub fn from_json(v: &Value) -> Option<Self> {
        let ctx = v.as_object().filter(|c| !c.is_empty())?;
        if ctx.get("key").is_none_or(Value::is_null) {
            return None;
        }
        let map = |name: &str| ctx.get(name).and_then(Value::as_object);
        Some(Self {
            owner: ctx.get("owner").and_then(Value::as_str).map(str::to_string),
            last_len: ctx.get("last_len").and_then(Value::as_i64).unwrap_or(0),
            last_move_s: finite(ctx.get("last_move_s")),
            occupancy: map("occupancy")
                .map(|m| {
                    m.iter()
                        .map(|(k, v)| (k.clone(), v.as_i64().unwrap_or(0)))
                        .collect()
                })
                .unwrap_or_default(),
            host_pool_tokens: map("host_pool_tokens")
                .map(|m| m.iter().map(|(k, v)| (k.clone(), nonneg_int(Some(v)))).collect())
                .unwrap_or_default(),
        })
    }
}

/// Python `session_terms`: E_c (duplication / eviction), M_c (ping-pong) and
/// their details.
fn session_terms(
    url: &str,
    n: i64,
    ctx: Option<&SessionContext>,
    m: &EftModel,
    knobs: &EftKnobs,
    decision_s: Option<f64>,
) -> (f64, f64, Value) {
    let Some(ctx) = ctx else {
        return (0.0, 0.0, json!({"session": "none"}));
    };
    let owner = ctx.owner.as_deref();
    let last = ctx.last_len;
    let added = if owner != Some(url) { n } else { 0.max(n - last) };
    let here = ctx.occupancy.get(url).copied().unwrap_or(0);
    let mut overflow = 0;
    if let Some(pool) = ctx.host_pool_tokens.get(url).copied().flatten() {
        overflow = 0.max(here + added - pool) - 0.max(here - pool);
    }
    let mut credit = 0;
    if let Some(owner) = owner.filter(|o| knobs.retire_old_copy && *o != url) {
        if let Some(old_pool) = ctx.host_pool_tokens.get(owner).copied().flatten() {
            let there = ctx.occupancy.get(owner).copied().unwrap_or(0);
            credit = 0.max(there + last - old_pool) - 0.max(there - old_pool);
        }
    }
    let tau_tok = knobs
        .evict_tau_tok_s
        .unwrap_or(m.b_new + m.b_pairs * knobs.evict_median_prefix_tokens as f64);
    let evict = knobs.evict_weight * tau_tok * 0.max(overflow - credit) as f64;
    let recent = owner.is_some_and(|o| o != url)
        && match (ctx.last_move_s, decision_s) {
            (Some(moved), Some(d)) if d.is_finite() => d - moved <= knobs.pingpong_window_s,
            _ => false,
        };
    let pingpong = if recent { knobs.pingpong_s } else { 0.0 };
    let detail = json!({"D_add": added, "dO": overflow, "Rel": credit, "tau_tok": tau_tok});
    (evict, pingpong, detail)
}

/// One healthy P worker as an input of [`decide`].
#[derive(Debug, Clone)]
pub struct DecideCandidate<'a> {
    pub worker_url: String,
    pub view: Option<View<'a>>,
    pub outstanding: Vec<OutstandingInput>,
    /// Indexer prefix hits of the request on this worker (unknown: 0).
    pub cpu_hit: i64,
    pub gpu_hit: i64,
    pub active_requests: i64,
}

struct Hit {
    url: String,
    cpu: i64,
    gpu: i64,
    max: i64,
}

/// The injector's peer pick for target `url`, and the pullable hit H_c.
fn pull_source(
    url: &str,
    local: i64,
    n: i64,
    hits: &[Hit],
    knobs: &EftKnobs,
) -> (Option<(String, i64)>, Option<&'static str>) {
    if !knobs.pull_enabled {
        return (None, Some("pull_disabled"));
    }
    let best = hits
        .iter()
        .filter(|h| h.url != url)
        .min_by(|a, b| b.max.cmp(&a.max).then_with(|| a.url.cmp(&b.url)));
    let Some(best) = best else {
        return (None, Some("no_peer"));
    };
    if best.max < knobs.pull_min_tokens {
        return (None, Some("below_min_tokens"));
    }
    if best.max - local < knobs.pull_min_incremental {
        return (None, Some("below_min_incremental"));
    }
    let pullable = match knobs.pull_tier {
        PullTier::Cpu => best.cpu,
        PullTier::Max => best.max,
    };
    (Some((best.url.clone(), n.min(pullable))), None)
}

struct Scored {
    forecast: Forecast,
    score_us: i64,
    /// Pull variant only: Q (pulled tokens) and X(Q).
    pulled: Option<(i64, f64)>,
}

impl Scored {
    fn record(&self) -> Value {
        let mut map = self.forecast.record(self.score_us);
        if let Some((q, x)) = self.pulled {
            map.insert("Q".into(), json!(q));
            map.insert("X_s".into(), json!(x));
        }
        Value::Object(map)
    }
}

struct Priced {
    url: String,
    l: i64,
    g: i64,
    active: i64,
    view_key: Value,
    a_s: Option<f64>,
    h: Option<i64>,
    pull_source: Option<String>,
    reason: Option<&'static str>,
    ahead: Option<Drain>,
    outstanding_rids: Vec<Value>,
    e: f64,
    m: f64,
    session: Value,
    pull_reason: Option<&'static str>,
    local: Option<Scored>,
    pull: Option<Scored>,
    variant: Option<&'static str>,
}

impl Priced {
    fn excluded(&self) -> bool {
        self.reason.is_some()
    }

    fn d(&self) -> Option<f64> {
        self.ahead.as_ref().map(|a| a.d)
    }

    fn chosen(&self) -> Option<&Scored> {
        match self.variant? {
            "pull" => self.pull.as_ref(),
            _ => self.local.as_ref(),
        }
    }

    fn record(&self) -> Value {
        let excluded = self.excluded();
        let ahead = self.ahead.as_ref();
        let variants = if excluded {
            json!({})
        } else {
            json!({
                "local": self.local.as_ref().map(Scored::record),
                "pull": self.pull.as_ref().map(Scored::record),
            })
        };
        json!({
            "worker_url": self.url,
            "L": self.l,
            "G": self.g,
            "active": self.active,
            "view_key": self.view_key,
            "a_s": self.a_s,
            "H": self.h,
            "pull_source": self.pull_source,
            "excluded": excluded,
            "unusable_reason": self.reason,
            "D": ahead.map(|a| a.d),
            "D_lead": ahead.map(|a| a.d_lead),
            "PB": ahead.map(|a| a.pb),
            "R0": ahead.map(|a| a.r0),
            "idle": ahead.map(|a| a.idle),
            "outstanding_rids": if excluded { Value::Null } else { json!(self.outstanding_rids) },
            "E": if excluded { Value::Null } else { json!(self.e) },
            "M": if excluded { Value::Null } else { json!(self.m) },
            "session": if excluded { Value::Null } else { self.session.clone() },
            "pull_reason": self.pull_reason,
            "variants": variants,
            "variant": self.variant,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn score(
    m: &EftModel,
    knobs: &EftKnobs,
    n: i64,
    shape: Shape,
    ahead: &Drain,
    snap: &SnapshotView,
    ready: f64,
    pull: bool,
    e: f64,
    mm: f64,
) -> Scored {
    let forecast = forecast(m, knobs, n, shape, ahead, snap, ready, pull);
    let mut total = forecast.f + knobs.gamma * forecast.svc;
    total += e;
    total += mm;
    Scored {
        forecast,
        score_us: round_us(total),
        pulled: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn price_candidate(
    c: &DecideCandidate<'_>,
    n: i64,
    hits: &[Hit],
    index: usize,
    m: &EftModel,
    knobs: &EftKnobs,
    session: Option<&SessionContext>,
    decision_s: Option<f64>,
) -> Priced {
    let url = c.worker_url.as_str();
    let hit = &hits[index];
    let (local_raw, gpu_raw) = (hit.max, hit.gpu.min(hit.max));
    let (source, no_pull) = pull_source(url, local_raw, n, hits, knobs);
    let reason = match &c.view {
        None => Some("no_view"),
        Some(view) if !view.usable() => view.unusable_reason,
        Some(view) if view.age_s.unwrap_or(f64::INFINITY) > knobs.max_view_age_s => {
            Some("view_too_old")
        }
        Some(_) => None,
    };
    let mut out = Priced {
        url: url.to_string(),
        l: local_raw,
        g: gpu_raw,
        active: c.active_requests,
        view_key: c
            .view
            .map(|v| json!([v.snap.epoch, v.snap.sequence]))
            .unwrap_or(Value::Null),
        a_s: c.view.and_then(|v| v.age_s),
        h: source.as_ref().map(|s| s.1),
        pull_source: source.as_ref().map(|s| s.0.clone()),
        reason,
        ahead: None,
        outstanding_rids: Vec::new(),
        e: 0.0,
        m: 0.0,
        session: Value::Null,
        pull_reason: None,
        local: None,
        pull: None,
        variant: None,
    };
    let (Some(view), None) = (c.view.as_ref(), reason) else {
        return out;
    };
    let snap = view.snap;
    let page = snap.page;
    let rows = outstanding_rows(&c.outstanding, &snap.rids);
    let ahead = drain(view, &rows, m, knobs);
    out.outstanding_rids = rows.iter().map(|r| r.rid.clone()).collect();
    let (evict, pingpong, detail) = session_terms(url, n, session, m, knobs, decision_s);
    out.e = evict;
    out.m = pingpong;
    out.session = detail;
    let (local, _) = Shape::of(n, local_raw, gpu_raw, None, page);
    out.local = Some(score(
        m, knobs, n, local, &ahead, snap, 0.0, false, evict, pingpong,
    ));
    let mut why = no_pull;
    if let Some((_, pullable)) = &source {
        if pullable - local_raw < knobs.pull_min_incremental {
            why = Some("below_min_incremental");
        } else {
            let q_pull = floor_hit(*pullable, n, page);
            let gained = q_pull - local.q;
            if gained <= 0 {
                why = Some("no_page_gain");
            } else if knobs.pull_issue == PullIssue::Arrival && ahead.d > knobs.pull_max_wait_s {
                why = Some("drain_over_pull_max_wait");
            } else {
                let span = pull_span(knobs, gained);
                let start = match knobs.pull_issue {
                    PullIssue::Arrival => ahead.pb,
                    PullIssue::Jit => ahead.d_lead,
                };
                let ready = start + span + knobs.pull_rho_s;
                let shape = Shape {
                    u: n - q_pull,
                    q: q_pull,
                    h: local.h,
                    s: gained,
                };
                let mut pull = score(
                    m, knobs, n, shape, &ahead, snap, ready, true, evict, pingpong,
                );
                pull.pulled = Some((gained, span));
                out.pull = Some(pull);
            }
        }
    }
    out.pull_reason = if out.pull.is_none() { why } else { None };
    out.ahead = Some(ahead);
    out
}

/// The full EFT-0+ placement of the reference (Python `decide`): choose the
/// P worker and variant of one request of `n` tokens. Returns the JSON
/// record of the reference, with every candidate and variant.
#[allow(clippy::too_many_arguments)]
pub fn decide(
    rid: &Value,
    n: i64,
    candidates: &[DecideCandidate<'_>],
    m: &EftModel,
    knobs: &EftKnobs,
    decision_s: Option<f64>,
    session: Option<&SessionContext>,
) -> Result<Value, String> {
    knobs.check()?;
    if n <= 0 {
        return Err("target input_tokens must be a positive integer".into());
    }
    let mut ordered: Vec<&DecideCandidate<'_>> = candidates.iter().collect();
    ordered.sort_by(|a, b| a.worker_url.cmp(&b.worker_url));
    let urls: BTreeSet<&str> = ordered.iter().map(|c| c.worker_url.as_str()).collect();
    if urls.len() != ordered.len() || ordered.is_empty() {
        return Err("candidate URLs must be unique and non-empty".into());
    }
    let hits: Vec<Hit> = ordered
        .iter()
        .map(|c| Hit {
            url: c.worker_url.clone(),
            cpu: n.min(c.cpu_hit),
            gpu: c.gpu_hit,
            max: n.min(c.cpu_hit.max(c.gpu_hit)),
        })
        .collect();
    let mut rows: Vec<Priced> = ordered
        .iter()
        .enumerate()
        .map(|(i, c)| price_candidate(c, n, &hits, i, m, knobs, session, decision_s))
        .collect();
    let pull_margin = round_us(knobs.pull_margin_s);
    for r in &mut rows {
        r.variant = if r.excluded() {
            None
        } else {
            let local = r.local.as_ref().map_or(0, |s| s.score_us);
            Some(match &r.pull {
                Some(pull) if pull.score_us + pull_margin < local => "pull",
                _ => "local",
            })
        };
    }
    let owner = (0..rows.len())
        .min_by(|&a, &b| {
            let (a, b) = (&rows[a], &rows[b]);
            b.l.cmp(&a.l)
                .then_with(|| b.g.cmp(&a.g))
                .then_with(|| {
                    a.d()
                        .unwrap_or(f64::INFINITY)
                        .partial_cmp(&b.d().unwrap_or(f64::INFINITY))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.active.cmp(&b.active))
                .then_with(|| a.url.cmp(&b.url))
        })
        .expect("non-empty");
    let has_owner = rows[owner].l > 0;
    let mut out = json!({
        "schema": SCHEMA,
        "rid": rid,
        "input_tokens": n,
        "model_sha256": m.sha256,
        "knobs": knobs.record(),
        "owner": rows[owner].url,
        "has_owner": has_owner,
        "chosen": null,
        "best": null,
        "margin_protected": false,
        "fallback": null,
        "eft_pred_s": null,
        "owner_pred_s": null,
        "gain_s": null,
        "candidates": [],
    });
    let usable: Vec<usize> = (0..rows.len()).filter(|&i| !rows[i].excluded()).collect();
    fn finish(mut out: Value, rows: &[Priced]) -> Value {
        out["candidates"] = Value::Array(rows.iter().map(Priced::record).collect());
        out
    }
    if has_owner && rows[owner].excluded() {
        out["fallback"] = json!("owner_view_unusable");
        return Ok(finish(out, &rows));
    }
    if usable.is_empty() {
        out["fallback"] = json!("no_usable_candidate");
        return Ok(finish(out, &rows));
    }
    let score_of = |i: usize| rows[i].chosen().map_or(0, |s| s.score_us);
    let best = usable
        .iter()
        .copied()
        .min_by(|&a, &b| {
            let (ra, rb) = (&rows[a], &rows[b]);
            score_of(a)
                .cmp(&score_of(b))
                .then_with(|| rb.l.cmp(&ra.l))
                .then_with(|| {
                    ra.d()
                        .unwrap_or(f64::INFINITY)
                        .partial_cmp(&rb.d().unwrap_or(f64::INFINITY))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| ra.active.cmp(&rb.active))
                .then_with(|| ra.url.cmp(&rb.url))
        })
        .expect("usable candidates");
    let mut chosen = best;
    out["best"] = json!({"worker_url": rows[best].url, "variant": rows[best].variant});
    if has_owner && best != owner {
        let owner_score = score_of(owner);
        let best_score = score_of(best);
        let limit = pymax(
            round_us(knobs.margin_abs_s) as f64,
            knobs.margin_rel * owner_score as f64,
        );
        if ((owner_score - best_score) as f64) <= limit {
            chosen = owner;
            out["margin_protected"] = json!(true);
        }
    }
    let picked = rows[chosen].chosen().expect("usable candidate is priced");
    out["chosen"] = json!({"worker_url": rows[chosen].url, "variant": rows[chosen].variant});
    out["eft_pred_s"] = json!(picked.forecast.f);
    if !rows[owner].excluded() {
        let held = rows[owner].chosen().expect("usable owner is priced");
        out["owner_pred_s"] = json!(held.forecast.f);
        out["gain_s"] = json!((held.score_us - picked.score_us) as f64 / 1e6);
    }
    Ok(finish(out, &rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn golden() -> Vec<Value> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/eft_cheap_golden.jsonl");
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("golden line is JSON"))
            .collect()
    }

    /// Numbers within 1e-9 relative (integers exactly); strings, booleans,
    /// nulls and key sets exactly.
    fn compare(path: &str, actual: &Value, expected: &Value, errors: &mut Vec<String>) {
        match (actual, expected) {
            (Value::Number(a), Value::Number(e)) => {
                let both_int = (a.is_i64() || a.is_u64()) && (e.is_i64() || e.is_u64());
                if both_int {
                    if a.as_i64() != e.as_i64() || a.as_u64() != e.as_u64() {
                        errors.push(format!("{path}: {a} != {e}"));
                    }
                } else {
                    let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
                    let tolerance = 1e-9 * a.abs().max(e.abs()).max(1e-300);
                    if (a - e).abs() > tolerance {
                        errors.push(format!("{path}: {a} != {e}"));
                    }
                }
            }
            (Value::Object(a), Value::Object(e)) => {
                let (ka, ke): (BTreeSet<_>, BTreeSet<_>) = (a.keys().collect(), e.keys().collect());
                if ka != ke {
                    errors.push(format!(
                        "{path}: keys {:?} != {:?}",
                        ka.difference(&ke).collect::<Vec<_>>(),
                        ke.difference(&ka).collect::<Vec<_>>()
                    ));
                }
                for (key, value) in e {
                    if let Some(actual) = a.get(key) {
                        compare(&format!("{path}.{key}"), actual, value, errors);
                    }
                }
            }
            (Value::Array(a), Value::Array(e)) => {
                if a.len() != e.len() {
                    errors.push(format!("{path}: length {} != {}", a.len(), e.len()));
                }
                for (i, (a, e)) in a.iter().zip(e).enumerate() {
                    compare(&format!("{path}[{i}]"), a, e, errors);
                }
            }
            _ if actual == expected => {}
            _ => errors.push(format!("{path}: {actual} != {expected}")),
        }
    }

    /// Run one golden case through the port: (model, views, decision).
    fn evaluate(inputs: &Value) -> (Value, Value, Value) {
        let model = EftModel::from_json(&inputs["model"], None).expect("golden model");
        let knobs = EftKnobs::from_overrides(&inputs["knobs"]).expect("golden knobs");
        let d = inputs["decision_s"].as_f64().expect("decision_s");
        let raws = inputs["candidates"].as_array().expect("candidates");
        let snaps: Vec<Option<SnapshotView>> = raws
            .iter()
            .map(|c| {
                (!c["raw"].is_null()).then(|| SnapshotView::parse(&c["raw"], 1, None))
            })
            .collect();
        let mut views = Map::new();
        let mut candidates = Vec::new();
        for (c, snap) in raws.iter().zip(&snaps) {
            let url = c["worker_url"].as_str().expect("url").to_string();
            let view = snap.as_ref().map(|s| {
                View::at(s, d - c["poll_sent_s"].as_f64().expect("poll_sent_s"))
            });
            views.insert(url.clone(), view.map_or(Value::Null, |v| v.record()));
            candidates.push(DecideCandidate {
                worker_url: url,
                view,
                outstanding: c["outstanding"]
                    .as_array()
                    .map(|rows| rows.iter().map(OutstandingInput::from_json).collect())
                    .unwrap_or_default(),
                cpu_hit: nonneg_int(c["indexer"].get("cpu")).unwrap_or(0),
                gpu_hit: nonneg_int(c["indexer"].get("gpu")).unwrap_or(0),
                active_requests: c["active_requests"].as_i64().unwrap_or(0),
            });
        }
        let session = SessionContext::from_json(&inputs["session"]);
        let n = inputs["target"]["input_tokens"].as_i64().expect("input_tokens");
        let decision = decide(
            &inputs["target"]["rid"],
            n,
            &candidates,
            &model,
            &knobs,
            Some(d),
            session.as_ref(),
        )
        .expect("golden decision");
        (model.record(), Value::Object(views), decision)
    }

    #[test]
    fn golden_file_parity_with_the_python_reference() {
        let cases = golden();
        assert_eq!(cases.len(), 64, "34 named + 30 random cases");
        let mut errors = Vec::new();
        for case in &cases {
            let name = case["name"].as_str().unwrap_or("?");
            let (model, views, decision) = evaluate(&case["inputs"]);
            let expected = &case["expected"];
            compare(&format!("{name}.model"), &model, &expected["model"], &mut errors);
            compare(&format!("{name}.views"), &views, &expected["views"], &mut errors);
            compare(
                &format!("{name}.decision"),
                &decision,
                &expected["decision"],
                &mut errors,
            );
        }
        assert!(
            errors.is_empty(),
            "{} golden mismatches; first: {:#?}",
            errors.len(),
            &errors[..errors.len().min(20)]
        );
    }

    #[test]
    fn python_float_repr_matches_python() {
        for (x, text) in [
            (0.05, "0.05"),
            (1e-5, "1e-05"),
            (1.2e-5, "1.2e-05"),
            (2e-12, "2e-12"),
            (0.0001, "0.0001"),
            (100.0, "100.0"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (-0.5, "-0.5"),
            (123.456, "123.456"),
            (0.0, "0.0"),
            (0.1 + 0.2, "0.30000000000000004"),
        ] {
            assert_eq!(python_float_repr(x), text, "{x}");
        }
        assert_eq!(
            python_json(&json!({"b": [1, 0.5, null, true], "a": "x\"y"})),
            r#"{"a":"x\"y","b":[1,0.5,null,true]}"#
        );
    }

    fn plain_model() -> EftModel {
        EftModel::from_json(
            &json!({"models": {FAMILY: {
                "launch_interval": {"global": {"coefficients": [0.05, 1e-5, 0.0, 0.004, 1e-7, 0.0]}},
            }}, "eft": {"tau_s": 0.02}}),
            Some("test".into()),
        )
        .unwrap()
    }

    fn waiting(rid: &str, n: i64) -> Value {
        json!({"rid": rid, "phase": "waiting", "input_tokens": n, "remaining_compute_tokens": n,
               "device_prefix_tokens": 0, "host_prefix_tokens": 0, "storage_prefix_tokens": 0,
               "state": {"intent_start_ok": true, "bound": false, "launched_unbound": false,
                         "held": false, "parked": false, "pull_pending": false,
                         "pull_inflight": false, "restore_inflight": false}})
    }

    fn snapshot(waiting: Vec<Value>) -> Value {
        json!({"epoch": "e1", "sequence": 1, "available": true, "age_s": 0.0, "stale": false,
               "truncated": false, "omitted_waiting": 0, "inflight_batches_known": true,
               "inflight_batches": [], "inflight_requests": [],
               "policy": {"page_size": 64, "chunk_tokens": 8192, "schedule_policy": "fcfs"},
               "waiting": waiting, "bootstrap": [],
               "admission_resources": {"park_gate_closed_last_scan": false}})
    }

    /// Five rows of 65,536 tokens: 40 passes of 8,192 tokens. D = 5 x
    /// (1e-5 x 65,536 + 0.004 x 8) + 0.05 x 40 = 5.4368 s, minus the age.
    #[test]
    fn drain_of_a_long_queue_minus_the_view_age() {
        let m = plain_model();
        let snap = SnapshotView::parse(
            &snapshot((0..5).map(|i| waiting(&format!("w{i}"), 65_536)).collect()),
            1,
            None,
        );
        let view = View::at(&snap, 0.25);
        assert!(view.usable());
        let d = drain(&view, &[], &m, &EftKnobs::default());
        assert!((d.d - (5.4368 - 0.25)).abs() < 1e-9, "{}", d.d);
        assert_eq!(d.passes_ahead, 40);
        assert!(!d.idle);
        // An outstanding row the snapshot does not show adds its full work;
        // a row it shows is not counted twice.
        let rows = outstanding_rows(
            &[
                OutstandingInput {
                    rid: json!("w0"),
                    uncached_tokens: Some(65_536),
                    expected_hit: 0,
                    local_hit: 0,
                    gpu_hit: 0,
                    hinted: false,
                },
                OutstandingInput {
                    rid: json!("new"),
                    uncached_tokens: Some(8_192),
                    expected_hit: 0,
                    local_hit: 0,
                    gpu_hit: 0,
                    hinted: false,
                },
            ],
            &snap.rids,
        );
        assert_eq!(rows.len(), 1);
        let d2 = drain(&view, &rows, &m, &EftKnobs::default());
        let extra = 1e-5 * 8_192.0 + 0.004 + 0.05;
        assert!((d2.d - d.d - extra).abs() < 1e-9, "{} {}", d2.d, d.d);
    }

    #[test]
    fn stale_and_old_views_are_unusable() {
        let mut raw = snapshot(vec![]);
        raw["stale"] = json!(true);
        let snap = SnapshotView::parse(&raw, 1, None);
        assert_eq!(View::at(&snap, 0.0).unusable_reason, Some("stale"));
        let mut raw = snapshot(vec![]);
        raw["age_s"] = Value::Null;
        let snap = SnapshotView::parse(&raw, 1, None);
        assert_eq!(View::at(&snap, 0.0).unusable_reason, Some("age_unknown"));
        let snap = SnapshotView::parse(&json!(null), 1, None);
        assert_eq!(View::at(&snap, 0.0).unusable_reason, Some("unavailable"));
        let snap = SnapshotView::parse(&snapshot(vec![]), 2, None);
        assert_eq!(View::at(&snap, 0.0).unusable_reason, Some("dp_not_1"));
        let snap = SnapshotView::parse(&snapshot(vec![]), 1, None);
        let view = View::at(&snap, 2.0);
        assert!(view.usable(), "the age limit is the caller's rule");
        assert_eq!(view.age_s, Some(2.0));
        assert!(snap.idle);
    }

    /// Svc and the variant shapes: local keeps the GPU part on device and
    /// restores the rest from host; a pull adds the gained whole pages.
    #[test]
    fn service_and_shapes() {
        let m = plain_model();
        let (local, pulled) = Shape::of(20_000, 18_000, 0, None, 64);
        assert_eq!(pulled, 0);
        assert_eq!(local, Shape { u: 2_016, q: 17_984, h: 17_984, s: 0 });
        let svc = service_seconds(&m, local, 64, 8192);
        let expected = 1e-5 * 2_016.0 + 1e-7 * 17_984.0 + 0.004 + 0.05;
        assert!((svc - expected).abs() < 1e-12, "{svc}");
        let (pull, pulled) = Shape::of(20_000, 0, 0, Some(18_000), 64);
        assert_eq!(pulled, 17_984);
        assert_eq!(pull, Shape { u: 2_016, q: 17_984, h: 0, s: 17_984 });
        // The storage prior is 2 x host.
        assert_eq!(m.b_stor, 2e-7);
        let (none, pulled) = Shape::of(20_000, 18_000, 18_000, Some(18_010), 64);
        assert_eq!(pulled, 0, "no whole page gained");
        assert_eq!(none.h, 0);
    }

    /// The RIDs of the latest in-flight pass are kept apart from the
    /// reference rid set (the recorded view does not change), and only when
    /// R0 prices that pass. A row without prefix data is priced cold and
    /// counted.
    #[test]
    fn pass_rids_and_cold_rows() {
        let cold = json!({"rid": "cold", "phase": "waiting", "input_tokens": 62_000,
            "remaining_compute_tokens": null, "device_prefix_tokens": null,
            "host_prefix_tokens": null, "storage_prefix_tokens": 0,
            "state": {"intent_start_ok": true, "bound": true, "launched_unbound": false,
                      "held": false, "parked": false, "pull_pending": false,
                      "pull_inflight": false, "restore_inflight": false}});
        let mut raw = snapshot(vec![waiting("w0", 8_192), cold]);
        raw["inflight_batches"] = json!([{"launch_age_s": 0.01,
            "request_ids": ["f0", "f1", null],
            "features": {"new_tokens": 4_000, "batch_size": 2}}]);
        let snap = SnapshotView::parse(&raw, 1, None);
        let expected: BTreeSet<String> = ["f0", "f1"].iter().map(|s| s.to_string()).collect();
        assert_eq!(snap.pass_rids, expected);
        assert!(!snap.rids.contains("f0"), "the reference rid set is unchanged");
        assert_eq!(View::at(&snap, 0.0).record()["rids"], json!(["cold", "w0"]));
        assert_eq!((snap.cold_rows, snap.cold_tokens), (1, 62_000));
        assert!(snap.degraded.contains("queue_compute_unknown_cold"));
        assert_eq!(snap.rows[1].u, 62_000);
        // Unknown launch age: R0 is 0, so the pass prices nothing.
        raw["inflight_batches"][0]["launch_age_s"] = Value::Null;
        let snap = SnapshotView::parse(&raw, 1, None);
        assert!(snap.pass_rids.is_empty());
        let m = plain_model();
        assert_eq!(drain(&View::at(&snap, 0.0), &[], &m, &EftKnobs::default()).r0, 0.0);
    }

    /// The empty view prices only the given rows: their work and passes.
    #[test]
    fn empty_view_prices_only_the_outstanding_rows() {
        let m = plain_model();
        let empty = SnapshotView::empty(64, 8192);
        let view = View::at(&empty, 0.0);
        assert!(view.usable());
        let d = drain(&view, &[], &m, &EftKnobs::default());
        assert_eq!(d.d, 0.0);
        assert!(d.idle);
        let rows = outstanding_rows(
            &[OutstandingInput {
                rid: json!("r"),
                uncached_tokens: Some(8_192),
                expected_hit: 0,
                local_hit: 0,
                gpu_hit: 0,
                hinted: false,
            }],
            &BTreeSet::new(),
        );
        let d = drain(&view, &rows, &m, &EftKnobs::default());
        let expected = 1e-5 * 8_192.0 + 0.004 + 0.05;
        assert!((d.d - expected).abs() < 1e-12, "{}", d.d);
        assert!(!d.idle);
    }
}
