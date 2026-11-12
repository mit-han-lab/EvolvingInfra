// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! The engine's `prefill_routing` snapshot of one prefill (P) worker, parsed.
//!
//! A P engine lists, in the `prefill_routing` section of `/v1/loads`, the
//! requests it holds (partial, waiting, bootstrap, in flight), its pulls and
//! its in-flight pass. [`SnapshotView`] is that section with the checks the
//! router needs before it uses it (epoch, DP size, readiness). The E3
//! forecast ([`crate::trade_e3`]) starts from it.

use std::collections::BTreeSet;

use serde_json::{Map, Value};

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

fn flag(degraded: &mut BTreeSet<String>, reason: &str) {
    degraded.insert(reason.to_string());
}

/// Page-floored cached prefix that leaves at least one page to compute
/// (the rule of `prefill_completion.target_work`).
pub fn floor_hit(tokens: i64, n: i64, page: i64) -> i64 {
    (tokens.div_euclid(page) * page).min(0.max(n - 2).div_euclid(page) * page)
}

// ---------------------------------------------------------------------------
// Frozen step-time model.
// ---------------------------------------------------------------------------

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
    let epoch_mismatch =
        expected_epoch.is_some_and(|e| raw.get("epoch").and_then(Value::as_str) != Some(e));
    [
        (!is_true(raw.get("available")), "unavailable"),
        (is_true(raw.get("stale")), "stale"),
        (is_true(raw.get("truncated")) && !omitted_zero, "truncated"),
        (
            !is_true(raw.get("inflight_batches_known")),
            "inflight_unknown",
        ),
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }
}
