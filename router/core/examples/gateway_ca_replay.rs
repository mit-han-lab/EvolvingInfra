// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Offline replay of `--policy gateway_cache_aware` against the route-time
//! observer of sgl-model-gateway
//! (`SGLANG_PREFILL_ROUTING_RECORD_PATH`, schema
//! `smg-prefill-route-observation.v1`).
//!
//! For each `decision_input` row the observer recorded, before the real
//! selection: the candidate order, each candidate's health and load
//! (`Worker::load()`), each candidate's history prefix (complete tokens of
//! the request matched in that worker's tree, read without touching the
//! LRU), the prompt length, the salt-segment length and the policy
//! parameters. The `selected` row (same `record_id`) holds the worker the
//! gateway chose. This tool applies the port's rule
//! ([`decide_from_history`]) to those inputs and reports how often the
//! gateway's choice is one the port allows.
//!
//! Exact replay is not possible from the observer alone: it records a
//! sha256 of the token ids, not the ids, so the trees cannot be rebuilt;
//! nor the creator of the deepest matched node (the gateway's pick among
//! equal-history candidates), the random tie picks, the eviction ticks, or
//! the LRU epoch. The observation and the selection are not atomic: loads
//! and the tree can change between them.
//!
//! Disagreement classes:
//! - `load_timing`: the choice is allowed with the loads of the other
//!   snapshot the observer took (candidate rows vs history rows), or with
//!   any one load moved by at most 2 (a request admitted or finished
//!   between observation and selection).
//! - `tree_timing`: another request of the same salt was selected between
//!   this observation and this selection (a concurrent tree insert), or the
//!   choice is allowed when any candidate's history is raised to the
//!   deepest match (an insert of this prefix in between).
//! - `unexplained`: none of the above (eviction timing cannot be told from
//!   a bug with the observer data).
//!
//! Usage: `gateway_ca_replay OBSERVER.jsonl [REPORT.json]`

use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader, Write};

use serde::Deserialize;
use serde_json::{json, Value};
use sgl_router::config::GatewayCacheAwareConfig;
use sgl_router::policies::gateway_cache_aware::decide_from_history;

#[derive(Deserialize)]
struct Row {
    #[serde(default)]
    sequence: u64,
    event: String,
    #[serde(default)]
    data: Value,
}

#[derive(Deserialize, Clone)]
struct HistCandidate {
    worker_url: String,
    #[serde(default)]
    history_prefix_units: Option<usize>,
    #[serde(default)]
    healthy: bool,
    #[serde(default)]
    load: usize,
}

#[derive(Deserialize, Clone)]
struct History {
    #[serde(default)]
    namespace: String,
    input_units: usize,
    #[serde(default)]
    salt_prefix_characters: usize,
    #[serde(default)]
    tree_capacity_characters: usize,
    cache_threshold: f32,
    balance_abs_threshold: usize,
    balance_rel_threshold: f32,
    #[serde(default)]
    eviction_interval_s: u64,
    candidates: Vec<HistCandidate>,
}

#[derive(Deserialize, Clone)]
struct BeginCandidate {
    worker_url: String,
    #[serde(default)]
    load: usize,
}

#[derive(Deserialize, Clone)]
struct DecisionInput {
    record_id: String,
    #[serde(default)]
    rid: Option<String>,
    #[serde(default)]
    decision_mono_s: f64,
    #[serde(default)]
    cache_salt_sha256: Option<String>,
    #[serde(default)]
    history_before_selection: Option<History>,
    #[serde(default)]
    candidates: Vec<BeginCandidate>,
}

struct Pending {
    sequence: u64,
    input: DecisionInput,
}

#[derive(Default)]
struct Report {
    decisions: u64,
    joined: u64,
    no_history: u64,
    unknown_worker: u64,
    agree_exact: u64,
    agree_tie: u64,
    branch: BTreeMap<&'static str, [u64; 3]>,
    disagree: BTreeMap<&'static str, u64>,
    examples: Vec<Value>,
    selected_per_worker: BTreeMap<String, u64>,
    params: BTreeMap<String, u64>,
    canary: u64,
}

fn config_of(h: &History) -> GatewayCacheAwareConfig {
    GatewayCacheAwareConfig {
        cache_threshold: h.cache_threshold,
        balance_abs_threshold: h.balance_abs_threshold,
        balance_rel_threshold: h.balance_rel_threshold,
        eviction_interval_secs: h.eviction_interval_s,
        max_tree_size: h.tree_capacity_characters,
        ..GatewayCacheAwareConfig::default()
    }
}

fn allows(
    loads: &[usize],
    healthy: &[bool],
    hist: &[usize],
    input_units: usize,
    cfg: &GatewayCacheAwareConfig,
    chosen: usize,
) -> bool {
    decide_from_history(loads, healthy, hist, input_units, cfg)
        .1
        .contains(&chosen)
}

fn load_timing(
    loads: &[usize],
    other: &[usize],
    healthy: &[bool],
    hist: &[usize],
    input_units: usize,
    cfg: &GatewayCacheAwareConfig,
    chosen: usize,
) -> bool {
    if other.len() == loads.len() && allows(other, healthy, hist, input_units, cfg, chosen) {
        return true;
    }
    for i in 0..loads.len() {
        for delta in [-2i64, -1, 1, 2] {
            let moved = loads[i] as i64 + delta;
            if moved < 0 {
                continue;
            }
            let mut l = loads.to_vec();
            l[i] = moved as usize;
            if allows(&l, healthy, hist, input_units, cfg, chosen) {
                return true;
            }
        }
    }
    false
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("usage: gateway_ca_replay OBSERVER.jsonl [REPORT.json]"))?;
    let reader = BufReader::with_capacity(1 << 22, std::fs::File::open(path)?);
    let mut pending: HashMap<String, Pending> = HashMap::new();
    // Per salt hash: sequence numbers of `selected` rows.
    let mut selected_by_salt: HashMap<String, Vec<u64>> = HashMap::new();
    let mut salt_of: HashMap<String, String> = HashMap::new();
    let mut joined: Vec<(Pending, String, u64)> = Vec::new();
    let mut report = Report::default();
    let mut bad_lines = 0u64;
    for line in reader.lines() {
        let line = line?;
        let row: Row = match serde_json::from_str(&line) {
            Ok(row) => row,
            Err(_) => {
                bad_lines += 1;
                continue;
            }
        };
        match row.event.as_str() {
            "decision_input" => {
                let input: DecisionInput = serde_json::from_value(row.data)?;
                report.decisions += 1;
                if let Some(salt) = &input.cache_salt_sha256 {
                    salt_of.insert(input.record_id.clone(), salt.clone());
                }
                pending.insert(
                    input.record_id.clone(),
                    Pending {
                        sequence: row.sequence,
                        input,
                    },
                );
            }
            "selected" => {
                let id = row.data["record_id"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                let url = row.data["worker_url"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                if let Some(salt) = salt_of.remove(&id) {
                    selected_by_salt.entry(salt).or_default().push(row.sequence);
                }
                if let Some(p) = pending.remove(&id) {
                    joined.push((p, url, row.sequence));
                }
            }
            _ => {}
        }
    }
    for (p, url, selected_seq) in &joined {
        report.joined += 1;
        let input = &p.input;
        if input
            .rid
            .as_deref()
            .is_some_and(|r| r.starts_with("wxu-router-canary"))
        {
            report.canary += 1;
        }
        *report.selected_per_worker.entry(url.clone()).or_default() += 1;
        let Some(h) = &input.history_before_selection else {
            report.no_history += 1;
            continue;
        };
        let cfg = config_of(h);
        *report
            .params
            .entry(format!(
                "threshold={} abs={} rel={} eviction_s={} tree_chars={} namespace={} salt_chars={}",
                h.cache_threshold,
                h.balance_abs_threshold,
                h.balance_rel_threshold,
                h.eviction_interval_s,
                h.tree_capacity_characters,
                h.namespace,
                h.salt_prefix_characters
            ))
            .or_default() += 1;
        let Some(chosen) = h.candidates.iter().position(|c| &c.worker_url == url) else {
            report.unknown_worker += 1;
            continue;
        };
        let loads: Vec<usize> = h.candidates.iter().map(|c| c.load).collect();
        let healthy: Vec<bool> = h.candidates.iter().map(|c| c.healthy).collect();
        let hist: Vec<usize> = h
            .candidates
            .iter()
            .map(|c| c.history_prefix_units.unwrap_or(0))
            .collect();
        let begin_loads: Vec<usize> = h
            .candidates
            .iter()
            .map(|c| {
                input
                    .candidates
                    .iter()
                    .find(|b| b.worker_url == c.worker_url)
                    .map_or(c.load, |b| b.load)
            })
            .collect();
        let (branch, allowed) = decide_from_history(&loads, &healthy, &hist, h.input_units, &cfg);
        let slot = branch.as_str();
        let counts = report.branch.entry(slot).or_default();
        if allowed.contains(&chosen) {
            if allowed.len() == 1 {
                report.agree_exact += 1;
                counts[0] += 1;
            } else {
                report.agree_tie += 1;
                counts[1] += 1;
            }
            continue;
        }
        counts[2] += 1;
        let concurrent_same_salt = input.cache_salt_sha256.as_ref().is_some_and(|salt| {
            selected_by_salt
                .get(salt)
                .is_some_and(|seqs| seqs.iter().any(|&s| s > p.sequence && s < *selected_seq))
        });
        let matched = hist.iter().copied().max().unwrap_or(0);
        let raised_allows = (0..hist.len()).any(|i| {
            let mut raised = hist.clone();
            raised[i] = matched.max(1).max(raised[i]);
            allows(&loads, &healthy, &raised, h.input_units, &cfg, chosen)
        });
        let class = if load_timing(
            &loads,
            &begin_loads,
            &healthy,
            &hist,
            h.input_units,
            &cfg,
            chosen,
        ) {
            "load_timing"
        } else if concurrent_same_salt || raised_allows {
            "tree_timing"
        } else {
            "unexplained"
        };
        *report.disagree.entry(class).or_default() += 1;
        if report.examples.len() < 40 {
            report.examples.push(json!({
                "rid": input.rid, "decision_mono_s": input.decision_mono_s,
                "class": class, "branch": slot, "allowed": allowed, "chosen": chosen,
                "loads": loads, "begin_loads": begin_loads, "history_units": hist,
                "input_units": h.input_units, "concurrent_same_salt": concurrent_same_salt,
            }));
        }
    }
    let replayed = report.agree_exact + report.agree_tie + report.disagree.values().sum::<u64>();
    let agree = report.agree_exact + report.agree_tie;
    let out = json!({
        "tool": "gateway_ca_replay (router gateway_cache_aware port vs sgl-model-gateway observer)",
        "observer": path,
        "decision_inputs": report.decisions,
        "joined_with_selected": report.joined,
        "unparsable_lines": bad_lines,
        "canary_requests": report.canary,
        "no_history": report.no_history,
        "selected_worker_not_a_candidate": report.unknown_worker,
        "replayed": replayed,
        "agree": agree,
        "agreement_rate": if replayed > 0 { agree as f64 / replayed as f64 } else { 0.0 },
        "agree_exact_single_choice": report.agree_exact,
        "agree_within_tie_set": report.agree_tie,
        "per_branch_exact_tie_disagree": report.branch,
        "disagreements_by_class": report.disagree,
        "selected_per_worker": report.selected_per_worker,
        "recorded_parameters": report.params,
        "examples": report.examples,
        "not_in_observer": [
            "token ids (only sha256 + length): the trees cannot be rebuilt",
            "the creator (last_tenant) of the deepest matched node: equal-history candidates form a tie set",
            "the random least-load tie picks",
            "eviction tick times and the LRU epoch counter",
            "loads and tree at the selection instant (observation is not atomic with selection)"
        ],
    });
    let text = serde_json::to_string_pretty(&out)?;
    match args.get(2) {
        Some(report_path) => std::fs::write(report_path, &text)?,
        None => std::io::stdout().write_all(text.as_bytes())?,
    }
    Ok(())
}
