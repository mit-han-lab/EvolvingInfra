// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Offline replay of a symmetric_pd cost-function trace (`router-cost.jsonl`,
//! schema `cost-balance.v1`) through a fresh [`CostBook`]: the recorded
//! reservations (input, candidates, TRADE parameters), dispatches, ends,
//! progress replies and snapshots, in trace order, as fast as possible. It
//! times every cost-book call (each takes the book's lock once, so in this
//! single-threaded replay a call's time is its lock hold) and, with
//! `--verify`, counts the differences between the incremental OLP queue and
//! the row walk at every decision.
//!
//! Exact replay is not possible from the trace alone: a progress row lists
//! only the entries whose remaining work changed (`updates`), not every
//! listed RID, so each reply is rebuilt as those updates plus the entries of
//! that P listed before and still live, up to the recorded `observed_rids`
//! (most recently updated first). Times are the replay's own (a decision's
//! clock and the poll times are `Instant::now()`), so the OLP step clock,
//! labels and gate differ from the run; `--params` can open the gate
//! (`{"warm_busy_s": 0, "warm_labels": 0, "min_pairs": 0}`).
//!
//! Usage:
//! `olp_replay TRACE.jsonl [--mode off|shadow|olp] [--incremental] [--compact]
//!  [--verify] [--params FILE] [--trace OUT.jsonl] [--async-trace]
//!  [--limit-reserves N] [--report REPORT.json]`

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use sgl_router::cost_balance::{Candidate, CostBook, PrefillProgress, Reservation};
use sgl_router::olp::{params_from_json, OlpConfig, OlpParams};
use sgl_router::scheduler::co_schedule::{TradeParams, TradeWait};

#[derive(Default)]
struct Timings {
    by_kind: HashMap<&'static str, Vec<u64>>,
}

impl Timings {
    fn add(&mut self, kind: &'static str, d: Duration) {
        self.by_kind
            .entry(kind)
            .or_default()
            .push(d.as_nanos() as u64);
    }

    fn value(&mut self) -> Value {
        let mut out = serde_json::Map::new();
        for (kind, xs) in self.by_kind.iter_mut() {
            xs.sort_unstable();
            let n = xs.len();
            let q = |p: f64| xs[((n - 1) as f64 * p).round() as usize] as f64 / 1e3;
            let total: u64 = xs.iter().sum();
            out.insert(
                kind.to_string(),
                json!({
                    "n": n,
                    "mean_us": total as f64 / n as f64 / 1e3,
                    "p50_us": q(0.5),
                    "p90_us": q(0.9),
                    "p99_us": q(0.99),
                    "max_us": xs[n - 1] as f64 / 1e3,
                    "total_ms": total as f64 / 1e6,
                }),
            );
        }
        Value::Object(out)
    }
}

struct Args {
    trace_in: String,
    mode: String,
    incremental: bool,
    compact: bool,
    verify: bool,
    params: Option<String>,
    trace_out: Option<String>,
    async_trace: bool,
    limit_reserves: Option<u64>,
    report: Option<String>,
}

fn parse_args() -> Args {
    let mut it = std::env::args().skip(1);
    let mut args = Args {
        trace_in: String::new(),
        mode: "off".into(),
        incremental: false,
        compact: false,
        verify: false,
        params: None,
        trace_out: None,
        async_trace: false,
        limit_reserves: None,
        report: None,
    };
    while let Some(a) = it.next() {
        match a.as_str() {
            "--mode" => args.mode = it.next().expect("--mode value"),
            "--incremental" => args.incremental = true,
            "--compact" => args.compact = true,
            "--verify" => args.verify = true,
            "--params" => args.params = it.next(),
            "--trace" => args.trace_out = it.next(),
            "--async-trace" => args.async_trace = true,
            "--limit-reserves" => {
                args.limit_reserves = it.next().and_then(|v| v.parse().ok());
            }
            "--report" => args.report = it.next(),
            other if args.trace_in.is_empty() && !other.starts_with("--") => {
                args.trace_in = other.to_string();
            }
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(
        !args.trace_in.is_empty(),
        "usage: olp_replay TRACE.jsonl ..."
    );
    assert!(
        matches!(args.mode.as_str(), "off" | "shadow" | "olp"),
        "--mode off|shadow|olp"
    );
    args
}

fn u64_of(v: &Value) -> u64 {
    v.as_u64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        .unwrap_or(0)
}

fn candidates_of(row: &Value) -> Vec<Candidate> {
    row["candidate_scores"]
        .as_array()
        .map(|cands| {
            cands
                .iter()
                .map(|c| Candidate {
                    worker_url: c["worker_url"].as_str().unwrap_or_default().to_string(),
                    local_hit: u64_of(&c["local_hit"]),
                    gpu_hit: c["gpu_hit"].as_u64(),
                    expected_hit: u64_of(&c["expected_hit"]),
                    hint: if c["hint"].is_null() {
                        None
                    } else {
                        Some(c["hint"].clone())
                    },
                })
                .collect()
        })
        .unwrap_or_default()
}

fn trade_params(td: &Value, wait: TradeWait) -> TradeParams {
    let d = TradeParams::default();
    let f = |key: &str, default: f64| td[key].as_f64().unwrap_or(default);
    TradeParams {
        rate_tokens_per_s: f("rate_tokens_per_s", d.rate_tokens_per_s),
        pull_fixed_ms: f("pull_fixed_ms", d.pull_fixed_ms),
        pull_ms_per_100k: f("pull_ms_per_100k", d.pull_ms_per_100k),
        copy_lambda: f("copy_lambda", d.copy_lambda),
        exclude_unlaunched: td["exclude_unlaunched"].as_bool().unwrap_or(false),
        margin_noise_k: f("margin_noise_k", d.margin_noise_k),
        eft_max_view_age_s: f("eft_max_view_age_s", d.eft_max_view_age_s),
        wait,
        ..d
    }
}

/// The P listing state of one worker: recorded reservation ID -> (RID,
/// remaining, sequence of its last update).
#[derive(Default)]
struct Listed {
    rows: HashMap<u64, (String, u64, u64)>,
}

fn main() {
    let args = parse_args();
    let params: OlpParams = match &args.params {
        Some(path) => {
            params_from_json(&std::fs::read(path).expect("read --params")).expect("valid --params")
        }
        None => OlpParams::default(),
    };
    let book = CostBook::new_uncached_with_policy(
        args.trace_out
            .as_deref()
            .map(std::path::Path::new)
            .filter(|p| p.as_os_str() != "none"),
        "symmetric_pd_cost_function",
        Some(Duration::from_secs(60)),
    )
    .expect("create the replay cost book");
    if args.mode != "off" {
        book.enable_olp(OlpConfig {
            params,
            shadow: args.mode == "shadow",
            params_sha256: None,
            incremental: args.incremental,
            compact_trace: args.compact,
            verify: args.verify,
        });
    }
    if args.async_trace {
        book.enable_async_trace().expect("async trace");
    }
    let wait = if args.mode == "olp" {
        TradeWait::Olp
    } else {
        TradeWait::Ledger
    };

    let file = std::fs::File::open(&args.trace_in).expect("open the trace");
    let reader = BufReader::with_capacity(1 << 22, file);
    let mut timings = Timings::default();
    let mut reservations: HashMap<u64, Reservation> = HashMap::new();
    let mut rid_of: HashMap<u64, String> = HashMap::new();
    let mut listed: HashMap<String, Listed> = HashMap::new();
    let mut last_sent: HashMap<String, Instant> = HashMap::new();
    let mut counts: HashMap<String, u64> = HashMap::new();
    let mut errors: HashMap<String, u64> = HashMap::new();
    let (mut same_choice, mut compared, mut reserves) = (0u64, 0u64, 0u64);
    let (mut rows_in, mut bad) = (0u64, 0u64);
    let mut olp_rows_walked = 0u64;
    let wall = Instant::now();
    for line in reader.lines() {
        let Ok(line) = line else {
            bad += 1;
            continue;
        };
        rows_in += 1;
        let Ok(row) = serde_json::from_str::<Value>(&line) else {
            bad += 1;
            continue;
        };
        let event = row["event"].as_str().unwrap_or("").to_string();
        *counts.entry(event.clone()).or_default() += 1;
        let entry = &row["entry"];
        let rec_id = entry["reservation_id"].as_u64();
        match event.as_str() {
            "reserve" => {
                if args.limit_reserves.is_some_and(|n| reserves >= n) {
                    break;
                }
                reserves += 1;
                let (Some(id), Some(rid)) = (rec_id, entry["rid"].as_str()) else {
                    continue;
                };
                rid_of.insert(id, rid.to_string());
                let td = &row["trade_decision"];
                let params = trade_params(td, wait);
                let candidates = candidates_of(&row);
                let input = u64_of(&entry["input_tokens"]);
                let t = Instant::now();
                let result = book.reserve_trade_session(
                    input,
                    Some(rid.into()),
                    candidates,
                    params,
                    None,
                    None,
                );
                timings.add("reserve", t.elapsed());
                match result {
                    Ok((selected, reservation, decision)) => {
                        if let Some(chosen) = td["chosen"].as_str() {
                            compared += 1;
                            same_choice += u64::from(selected.worker_url == chosen);
                        }
                        if let Some(cands) = decision["candidates"].as_array() {
                            olp_rows_walked += cands
                                .iter()
                                .map(|c| c["olp_rows"].as_u64().unwrap_or(0))
                                .sum::<u64>();
                        }
                        reservations.insert(id, reservation);
                    }
                    Err(e) => {
                        *errors.entry(format!("reserve: {e}")).or_default() += 1;
                    }
                }
            }
            "dispatched" => {
                if let Some(r) = rec_id.and_then(|id| reservations.get_mut(&id)) {
                    let t = Instant::now();
                    let result = r.mark_dispatched();
                    timings.add("dispatched", t.elapsed());
                    if let Err(e) = result {
                        *errors.entry(format!("dispatched: {e}")).or_default() += 1;
                    }
                }
            }
            "finished" | "unknown" => {
                let Some(id) = rec_id else { continue };
                let reason = entry["reason"].as_str().unwrap_or("replay").to_string();
                if let Some(r) = reservations.get_mut(&id) {
                    let t = Instant::now();
                    let result = if event == "finished" {
                        r.finish(&reason)
                    } else {
                        r.mark_unknown(&reason)
                    };
                    timings.add(
                        if event == "finished" {
                            "finished"
                        } else {
                            "unknown"
                        },
                        t.elapsed(),
                    );
                    if let Err(e) = result {
                        *errors.entry(format!("{event}: {e}")).or_default() += 1;
                    }
                }
                if event == "finished" {
                    if let Some(r) = reservations.remove(&id) {
                        let t = Instant::now();
                        drop(r);
                        timings.add("drop", t.elapsed());
                    }
                    for l in listed.values_mut() {
                        l.rows.remove(&id);
                    }
                }
            }
            "rolled_back" | "expired" => {
                if let Some(r) = rec_id.and_then(|id| reservations.remove(&id)) {
                    let t = Instant::now();
                    drop(r);
                    timings.add("drop", t.elapsed());
                }
                if let Some(id) = rec_id {
                    for l in listed.values_mut() {
                        l.rows.remove(&id);
                    }
                }
            }
            "prefill_progress" => {
                let url = row["worker_url"].as_str().unwrap_or_default().to_string();
                let seq = u64_of(&row["engine_sequence"]);
                let observed = u64_of(&row["observed_rids"]) as usize;
                let l = listed.entry(url.clone()).or_default();
                if let Some(updates) = row["updates"].as_array() {
                    for u in updates {
                        let id = u64_of(&u["reservation_id"]);
                        if let Some(rid) = rid_of.get(&id) {
                            l.rows
                                .insert(id, (rid.clone(), u64_of(&u["remaining_after"]), seq));
                        }
                    }
                }
                l.rows.retain(|id, _| reservations.contains_key(id));
                let mut rows: Vec<(&u64, &(String, u64, u64))> = l.rows.iter().collect();
                rows.sort_by(|a, b| b.1 .2.cmp(&a.1 .2));
                let remaining = rows
                    .into_iter()
                    .take(observed)
                    .map(|(_, (rid, rem, _))| (rid.clone(), *rem))
                    .collect();
                let progress = PrefillProgress {
                    epoch: row["engine_epoch"].as_str().unwrap_or("e").to_string(),
                    sequence: seq,
                    remaining,
                    truncated: row["snapshot_truncated"].as_bool().unwrap_or(false),
                };
                let mut sent = Instant::now();
                if let Some(last) = last_sent.get(&url) {
                    if sent <= *last {
                        sent = *last + Duration::from_nanos(1);
                    }
                }
                last_sent.insert(url.clone(), sent);
                let load = row["engine_load"].clone();
                let t = Instant::now();
                let result = book.apply_progress(&url, progress, load, sent);
                timings.add("progress", t.elapsed());
                if let Err(e) = result {
                    *errors.entry(format!("progress: {e}")).or_default() += 1;
                }
            }
            "snapshot" => {
                let t = Instant::now();
                let _ = book.snapshot();
                timings.add("snapshot", t.elapsed());
            }
            _ => {}
        }
    }
    let replay_s = wall.elapsed().as_secs_f64();
    let t = Instant::now();
    let synced = book.sync_trace().map_err(|e| e.to_string());
    let sync_ms = t.elapsed().as_secs_f64() * 1e3;
    let verify = book.olp_verify_stats().map(|v| v.value());
    let final_snapshot = book.snapshot();
    let trace_bytes = args
        .trace_out
        .as_deref()
        .filter(|p| *p != "none")
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len());
    let report = json!({
        "trace_in": args.trace_in,
        "mode": args.mode,
        "incremental": args.incremental,
        "compact": args.compact,
        "verify_on": args.verify,
        "params": args.params,
        "trace_out": args.trace_out,
        "async_trace": args.async_trace,
        "rows_in": rows_in,
        "bad_rows": bad,
        "events": counts,
        "errors": errors,
        "reserves": reserves,
        "choice_same_as_recorded": [same_choice, compared],
        "olp_rows_walked_per_decision": if reserves > 0 { olp_rows_walked as f64 / reserves as f64 } else { 0.0 },
        "timings": timings.value(),
        "replay_wall_s": replay_s,
        "sync_trace_ms": sync_ms,
        "sync_trace": synced,
        "trace_bytes": trace_bytes,
        "verify": verify,
        "final_olp": final_snapshot.get("olp"),
        "tracked_entries": final_snapshot["tracked_entries"],
    });
    let text = serde_json::to_string_pretty(&report).expect("report json");
    match &args.report {
        Some(path) => {
            let mut f = std::fs::File::create(path).expect("create the report");
            f.write_all(text.as_bytes()).expect("write the report");
        }
        None => println!("{text}"),
    }
    drop(reservations);
}
