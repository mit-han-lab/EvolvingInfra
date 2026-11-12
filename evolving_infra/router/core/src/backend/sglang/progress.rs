// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! Convert SGLang engine observations to backend load samples.
//!
//! Parses aggregate load and optional DP=1 per-request prefill progress from
//! `/v1/loads`, preserving absent measurements and engine snapshot timestamps.
//! HTTP retrieval and wire validation are isolated from ledger reconciliation.
use crate::backend::state::{EngineLoadSample, LoadObservation, PrefillProgress};
use serde::Deserialize;
use serde_json::Value;

/// Parse a `/v1/loads` reply into `(uncached_tokens, running, waiting)`
/// summed over DP ranks. Missing fields remain None, never a measured zero.
/// Uncached token counts must be complete across the returned ranks.
/// Read aggregate queue and running-request counts from an engine reply.
///
/// # Inputs
/// `v` is either a `/v1/loads` response containing rank entries or a single
/// load object; fields may be top-level or nested under `core`.
///
/// # Outputs
/// Returns saturated sums across ranks. Missing running/waiting counts
/// remain `None`; incomplete uncached-token counts produce an error instead
/// of a zero-load observation.
pub fn parse_loads(v: &Value) -> Result<LoadObservation, String> {
    let entries: Vec<&Value> = match v.get("loads").and_then(Value::as_array) {
        Some(arr) if !arr.is_empty() => arr.iter().collect(),
        _ => vec![v],
    };
    let field = |e: &Value, k: &str| {
        e.get(k)
            .or_else(|| e.get("core").and_then(|c| c.get(k)))
            .and_then(Value::as_u64)
    };
    let sum = |key: &str| -> Option<u64> {
        entries.iter().try_fold(0u64, |total, e| {
            field(e, key).map(|n| total.saturating_add(n))
        })
    };
    let uncached_tokens = sum("num_waiting_uncached_tokens");
    if uncached_tokens.is_none() {
        return Err("incomplete num_waiting_uncached_tokens in /v1/loads".to_string());
    }
    Ok(LoadObservation {
        uncached_tokens,
        running: sum("num_running_reqs"),
        waiting: sum("num_waiting_reqs"),
    })
}

/// For per-request work, DP ranks require independent epoch/sequence cursors.
/// This first implementation explicitly supports DP=1 rather than selecting an
/// arbitrary rank or merging unrelated sequence numbers.
/// Decode aggregate load, snapshot time and optional per-request progress.
///
/// # Inputs
/// `v` is a SGLang load response; `per_request` enables the DP=1 prefill
/// progress contract required for per-RID ledger corrections.
///
/// # Outputs
/// Returns a load sample with the minimum valid rank timestamp, or `None`
/// if any rank lacks one. Returns an error for invalid aggregate load, or
/// when requested progress has multiple ranks, a missing/invalid section,
/// or an empty engine epoch. Missing RIDs are left for ledger reconciliation.
pub fn parse_sample(v: &Value, per_request: bool) -> Result<EngineLoadSample, String> {
    parse_sample_with(v, per_request, false)
}

/// [`parse_sample`]; `optional_progress` (the unknown reconciliation without
/// per-RID accounting) reads a single-rank `prefill_work` section when it is
/// present and valid and otherwise returns the sample without progress.
pub fn parse_sample_with(
    v: &Value,
    per_request: bool,
    optional_progress: bool,
) -> Result<EngineLoadSample, String> {
    if optional_progress && !per_request {
        let mut sample = parse_sample_with(v, false, false)?;
        sample.prefill_work = parse_sample_with(v, true, false)
            .ok()
            .and_then(|s| s.prefill_work);
        return Ok(sample);
    }
    let observation = parse_loads(v)?;
    let entries: Vec<&Value> = match v.get("loads").and_then(Value::as_array) {
        Some(rows) if !rows.is_empty() => rows.iter().collect(),
        _ => vec![v],
    };
    if per_request && entries.len() != 1 {
        return Err("per-RID prefill work requires DP=1 per worker".into());
    }
    let timestamps: Option<Vec<_>> = entries
        .iter()
        .map(|row| {
            row.get("timestamp")
                .or_else(|| row.get("core").and_then(|c| c.get("timestamp")))
                .and_then(Value::as_f64)
                .filter(|t| t.is_finite() && *t > 0.0)
        })
        .collect();
    let timestamp = timestamps.and_then(|values| values.into_iter().reduce(f64::min));
    let prefill_work = if per_request {
        let raw = entries[0]
            .get("prefill_work")
            .filter(|v| !v.is_null())
            .ok_or("engine must enable --enable-prefill-work-snapshot")?;
        let progress = PrefillProgress::deserialize(raw).map_err(|e| e.to_string())?;
        if progress.epoch.is_empty() {
            return Err("prefill_work engine epoch is empty".into());
        }
        Some(progress)
    } else {
        None
    };
    Ok(EngineLoadSample {
        observation,
        timestamp,
        prefill_work,
    })
}

/// Fetch and parse one SGLang engine-load snapshot.
///
/// # Inputs
/// `client` supplies HTTP connection/timeouts; `base` is the engine URL;
/// `per_request` selects `core` alone or `core,prefill_work` fields.
///
/// # Outputs
/// Returns the parsed `EngineLoadSample`, or a string describing transport,
/// HTTP status, JSON decoding or load-contract failure. Performs one GET
/// and does not retry or mutate the backend ledger.
pub async fn fetch_loads(
    client: &reqwest::Client,
    base: &str,
    per_request: bool,
    reconcile: bool,
) -> Result<EngineLoadSample, String> {
    let include = if per_request || reconcile {
        "core,prefill_work"
    } else {
        "core"
    };
    let url = format!("{}/v1/loads?include={include}", base.trim_end_matches('/'));
    let v: Value = client
        .get(&url)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())?;
    parse_sample_with(&v, per_request, reconcile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    /// Check that optional prefill progress follows the supported wire contract.
    ///
    /// # Inputs
    /// No arguments; constructs JSON snapshots with one rank, two ranks and
    /// a missing progress section.
    ///
    /// # Outputs
    /// Returns `()` if valid progress/timestamps are preserved and unsupported
    /// or absent requested progress is rejected; assertion failure panics.
    fn progress_requires_single_rank_explicit_section_and_engine_epoch() {
        let sample = json!({"loads":[{"timestamp":10.0,"num_waiting_uncached_tokens":0,"prefill_work":{"epoch":"e","sequence":1,"remaining":{"r":8192},"truncated":true}}]});
        let parsed = parse_sample(&sample, true).unwrap();
        assert_eq!(parsed.prefill_work.unwrap().remaining["r"], 8192);
        assert_eq!(parsed.timestamp, Some(10.0));
        assert!(parse_sample(
            &json!({"loads":[sample["loads"][0].clone(),sample["loads"][0].clone()]}),
            true
        )
        .is_err());
        assert!(parse_sample(&json!({"num_waiting_uncached_tokens":0}), true).is_err());
        assert!(parse_sample(&json!({"num_waiting_uncached_tokens":0}), false).is_ok());
        // The reconciliation reads progress when present and never fails without it.
        let optional = parse_sample_with(&sample, false, true).unwrap();
        assert_eq!(optional.prefill_work.unwrap().remaining["r"], 8192);
        let none =
            parse_sample_with(&json!({"num_waiting_uncached_tokens":0}), false, true).unwrap();
        assert!(none.prefill_work.is_none());
    }
}
