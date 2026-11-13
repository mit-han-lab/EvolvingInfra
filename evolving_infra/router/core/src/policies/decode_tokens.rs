// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! Token-charged decode selection (`--pd-decode-policy min-inflight-tokens`
//! and `po2-tokens`).
//!
//! Round robin and the request-count policies (`po2`, `affinity-min-load`)
//! balance request counts, but a decode worker's KV pressure is the sum of
//! the contexts it holds, and agentic contexts are heavy-tailed (mean
//! ~150-220k tokens, up to ~1M). This policy keeps, per decode worker, the
//! sum of the tokens of every request dispatched to it and not yet finished:
//!
//! * charge = prompt tokens (token ids in the body, or a chars/4 estimate
//!   for text) + `min(max_tokens, cap)` (`cap` when the request has none);
//! * the charge is taken **at selection time** under one lock (so two
//!   concurrent picks see each other) and released by an RAII
//!   [`DecodeTokenCharge`] that the request routes hold beside the decode
//!   leg's `LoadGuard` — stream end, upstream error, transport error,
//!   stale-request cancel, client disconnect and a dropped handler future
//!   all release it;
//! * `min-inflight-tokens` picks the healthy decode with the fewest charged
//!   tokens (ties: fewest charged requests, then URL order);
//!   `po2-tokens` samples two distinct healthy decodes and takes the lower;
//! * `--pd-decode-tokens-normalize-capacity` compares
//!   `(charged + charge) / max_total_num_tokens` instead, when every
//!   candidate reported a capacity in `/server_info`; otherwise all decodes
//!   count as equal capacity.
//!
//! The router does not observe engine-side decode progress, eviction or
//! retraction: the charge is an upper-bound proxy for the KV a request will
//! occupy on its decode worker, held for the decode leg's whole lifetime.

use crate::config::{PdDecodePolicy, PdDecodeTokensConfig};
use crate::workers::{DecodeTokenCharge, Worker};
use parking_lot::Mutex;
use rand::Rng;
use serde_json::Value;
use std::cmp::Ordering;
use std::sync::Arc;

/// Which token rule the balancer applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenRule {
    MinInflightTokens,
    PowerOfTwoTokens,
}

impl TokenRule {
    pub fn name(self) -> &'static str {
        match self {
            TokenRule::MinInflightTokens => "min-inflight-tokens",
            TokenRule::PowerOfTwoTokens => "po2-tokens",
        }
    }
}

/// Per-decision record for the `routing decision` log line.
#[derive(Debug, Clone)]
pub struct DecodeTokenLog {
    /// `min-inflight-tokens`, `po2-tokens`, or `replay_pinned` (a
    /// `--pd-decision-replay` pin that was charged but not selected).
    pub rule: &'static str,
    /// This request's charge.
    pub charge_tokens: u64,
    /// The chosen decode's charged tokens after this charge.
    pub chosen_charged_tokens: u64,
    /// Whether the capacity-normalized comparison was used.
    pub normalized: bool,
    /// Every candidate's state after this charge, URL order:
    /// `{"<url>": [charged_tokens, charged_reqs, capacity_or_0], ...}`.
    pub loads: Value,
}

/// A decode pick with its live charge.
pub struct DecodeTokenPick {
    pub worker: Arc<Worker>,
    pub charge: DecodeTokenCharge,
    pub log: DecodeTokenLog,
}

/// Stateless apart from the selection lock: the charges live on the
/// [`Worker`]s, so a worker removed and re-added starts from 0 exactly like
/// its `active_requests` counter.
#[derive(Debug)]
pub struct DecodeTokenBalancer {
    rule: TokenRule,
    cfg: PdDecodeTokensConfig,
    /// Serializes read-min-then-charge so concurrent picks never both see
    /// the same minimum. Releases (guard drops) do not take it.
    pick: Mutex<()>,
}

#[derive(Debug, Clone, Copy)]
struct Load {
    tokens: u64,
    reqs: usize,
    capacity: Option<u64>,
}

impl DecodeTokenBalancer {
    /// `Some` exactly for the token-charged policies.
    pub fn for_policy(policy: PdDecodePolicy, cfg: PdDecodeTokensConfig) -> Option<Self> {
        let rule = match policy {
            PdDecodePolicy::MinInflightTokens => TokenRule::MinInflightTokens,
            PdDecodePolicy::PowerOfTwoTokens => TokenRule::PowerOfTwoTokens,
            _ => return None,
        };
        Some(Self::new(rule, cfg))
    }

    pub fn new(rule: TokenRule, cfg: PdDecodeTokensConfig) -> Self {
        Self {
            rule,
            cfg,
            pick: Mutex::new(()),
        }
    }

    pub fn rule(&self) -> TokenRule {
        self.rule
    }

    pub fn config(&self) -> &PdDecodeTokensConfig {
        &self.cfg
    }

    /// Charge for one request: `prompt_tokens` + the capped output budget
    /// read from the parsed body (`max_tokens`, `max_completion_tokens`, or
    /// `sampling_params.max_new_tokens`); the cap when absent or unparsed.
    pub fn request_charge(&self, prompt_tokens: u64, body: Option<&Value>) -> u64 {
        let cap = self.cfg.output_charge_cap;
        let output = body
            .and_then(requested_output_tokens)
            .map_or(cap, |n| n.min(cap));
        prompt_tokens.max(1).saturating_add(output)
    }

    /// Pick a decode from `candidates` and charge it `charge_tokens`.
    /// Healthy (closed-breaker) candidates only; when every breaker is open,
    /// all candidates (the selection stays total like the other decode
    /// selectors, and the dispatch surfaces `BreakerOpen`). `None` only for
    /// an empty slice.
    pub fn select(
        &self,
        candidates: &[Arc<Worker>],
        charge_tokens: u64,
    ) -> Option<DecodeTokenPick> {
        if candidates.is_empty() {
            return None;
        }
        let healthy: Vec<&Arc<Worker>> = candidates
            .iter()
            .filter(|w| w.breaker.would_allow())
            .collect();
        let mut pool: Vec<&Arc<Worker>> = if healthy.is_empty() {
            candidates.iter().collect()
        } else {
            healthy
        };
        // Registry order is a HashSet's; URL order makes "index" stable.
        pool.sort_by(|a, b| a.url.cmp(&b.url));

        let pick_guard = self.pick.lock();
        let loads: Vec<Load> = pool
            .iter()
            .map(|w| Load {
                tokens: w.decode_charged_tokens(),
                reqs: w.decode_charged_reqs(),
                capacity: w.kv_capacity_tokens(),
            })
            .collect();
        let normalized =
            self.cfg.normalize_by_capacity && loads.iter().all(|l| l.capacity.is_some());
        let better = |i: usize, j: usize| -> Ordering {
            compare(&loads[i], &loads[j], charge_tokens, normalized).then(i.cmp(&j))
        };
        let chosen = match (self.rule, pool.len()) {
            (_, 1) => 0,
            (TokenRule::MinInflightTokens, n) => (0..n).min_by(|&i, &j| better(i, j)).unwrap_or(0),
            (TokenRule::PowerOfTwoTokens, n) => {
                let mut rng = rand::thread_rng();
                let i = rng.gen_range(0..n);
                let mut j = rng.gen_range(0..n - 1);
                if j >= i {
                    j += 1;
                }
                if better(i, j) == Ordering::Greater {
                    j
                } else {
                    i
                }
            }
        };
        let worker = Arc::clone(pool[chosen]);
        let charge = worker.charge_decode_tokens(charge_tokens);
        let log = build_log(self.rule.name(), &pool, &worker, charge_tokens, normalized);
        drop(pick_guard);
        Some(DecodeTokenPick {
            worker,
            charge,
            log,
        })
    }

    /// Late D `d1` rule: the min-inflight-tokens pick among
    /// `candidates[set[..]]` (the eligible candidates of one bind; the
    /// caller already filtered them), charged under the selection lock.
    /// Same comparison as [`Self::select`] with the minimum rule: fewer
    /// charged tokens after this charge (or the capacity-normalized value),
    /// then fewer charged requests, then URL order. Returns the index into
    /// `candidates` and the charge; `None` for an empty set.
    pub fn select_among(
        &self,
        candidates: &[Arc<Worker>],
        set: &[usize],
        charge_tokens: u64,
    ) -> Option<(usize, DecodeTokenCharge)> {
        let _pick = self.pick.lock();
        let winner = self.d1_index_locked(candidates, set, charge_tokens)?;
        let charge = candidates[winner].charge_decode_tokens(charge_tokens);
        Some((winner, charge))
    }

    /// The selection lock: hold it to read the charges, pick and charge as
    /// one step (late D `fit_req` on observed capacity,
    /// [`crate::late_fit`]).
    pub fn lock(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.pick.lock()
    }

    /// The [`Self::select_among`] pick without the charge: the
    /// min-inflight-tokens order (fewer charged tokens, or the
    /// capacity-normalized value, then fewer charged requests, then URL).
    /// The caller holds [`Self::lock`]. `None` for an empty set.
    pub fn d1_index_locked(
        &self,
        candidates: &[Arc<Worker>],
        set: &[usize],
        charge_tokens: u64,
    ) -> Option<usize> {
        if set.is_empty() {
            return None;
        }
        let mut order: Vec<usize> = set.to_vec();
        order.sort_by(|&a, &b| candidates[a].url.cmp(&candidates[b].url));
        let loads: Vec<Load> = order
            .iter()
            .map(|&i| Load {
                tokens: candidates[i].decode_charged_tokens(),
                reqs: candidates[i].decode_charged_reqs(),
                capacity: candidates[i].kv_capacity_tokens(),
            })
            .collect();
        let normalized =
            self.cfg.normalize_by_capacity && loads.iter().all(|l| l.capacity.is_some());
        let best = (0..order.len())
            .min_by(|&i, &j| {
                compare(&loads[i], &loads[j], charge_tokens, normalized).then(i.cmp(&j))
            })
            .unwrap_or(0);
        Some(order[best])
    }

    /// Late D `fit_req` with a fixed capacity (sgl-model-gateway
    /// `select_fit_fewest_requests_and_charge`): among
    /// `candidates[set[..]]` whose charged tokens plus `charge_tokens` are
    /// at most `capacity_tokens`, the fewest charged requests, then the
    /// fewest charged tokens, then URL order; when none fits, the
    /// [`Self::select_among`] order. Charges the pick under the selection
    /// lock. The bool says whether the pick fit. `None` for an empty set.
    pub fn select_fit_fewest_requests(
        &self,
        candidates: &[Arc<Worker>],
        set: &[usize],
        charge_tokens: u64,
        capacity_tokens: u64,
    ) -> Option<(usize, DecodeTokenCharge, bool)> {
        if set.is_empty() {
            return None;
        }
        let _pick = self.pick.lock();
        // One read per candidate: releases do not take the lock.
        let loads: Vec<(usize, u64, usize)> = set
            .iter()
            .map(|&i| {
                (
                    i,
                    candidates[i].decode_charged_tokens(),
                    candidates[i].decode_charged_reqs(),
                )
            })
            .collect();
        let fitting = loads
            .iter()
            .filter(|(_, tokens, _)| tokens.saturating_add(charge_tokens) <= capacity_tokens)
            .min_by(|a, b| {
                a.2.cmp(&b.2)
                    .then(a.1.cmp(&b.1))
                    .then_with(|| candidates[a.0].url.cmp(&candidates[b.0].url))
            });
        let (winner, fit) = match fitting {
            Some(&(i, _, _)) => (i, true),
            None => (self.d1_index_locked(candidates, set, charge_tokens)?, false),
        };
        let charge = candidates[winner].charge_decode_tokens(charge_tokens);
        Some((winner, charge, fit))
    }

    /// Charge `charge_tokens` to `worker` under the selection lock (a late-D
    /// decision by another rule, or a bind with no decision).
    pub fn charge(&self, worker: &Worker, charge_tokens: u64) -> DecodeTokenCharge {
        let _pick = self.pick.lock();
        worker.charge_decode_tokens(charge_tokens)
    }

    /// Move a live charge to `to` under the selection lock: the old charge is
    /// released and the same tokens are charged to `to`.
    pub fn move_charge(&self, old: DecodeTokenCharge, to: &Worker) -> DecodeTokenCharge {
        let _pick = self.pick.lock();
        let tokens = old.tokens();
        drop(old);
        to.charge_decode_tokens(tokens)
    }

    /// Charge a decode that was pinned elsewhere (`--pd-decision-replay`),
    /// so the per-D sums still cover every dispatched request.
    pub fn charge_pinned(
        &self,
        worker: &Arc<Worker>,
        candidates: &[Arc<Worker>],
        charge_tokens: u64,
    ) -> (DecodeTokenCharge, DecodeTokenLog) {
        let _pick = self.pick.lock();
        let charge = worker.charge_decode_tokens(charge_tokens);
        let mut pool: Vec<&Arc<Worker>> = candidates.iter().collect();
        if !pool.iter().any(|w| w.url == worker.url) {
            pool.push(worker);
        }
        pool.sort_by(|a, b| a.url.cmp(&b.url));
        let log = build_log("replay_pinned", &pool, worker, charge_tokens, false);
        (charge, log)
    }
}

/// Lower post-placement load first, then fewer charged requests.
fn compare(a: &Load, b: &Load, charge: u64, normalized: bool) -> Ordering {
    let ta = a.tokens.saturating_add(charge) as u128;
    let tb = b.tokens.saturating_add(charge) as u128;
    let by_tokens = match (normalized, a.capacity, b.capacity) {
        // ta / ca vs tb / cb without floats: ta * cb vs tb * ca.
        (true, Some(ca), Some(cb)) => (ta * cb as u128).cmp(&(tb * ca as u128)),
        _ => ta.cmp(&tb),
    };
    by_tokens.then(a.reqs.cmp(&b.reqs))
}

fn build_log(
    rule: &'static str,
    pool: &[&Arc<Worker>],
    chosen: &Arc<Worker>,
    charge_tokens: u64,
    normalized: bool,
) -> DecodeTokenLog {
    let mut loads = serde_json::Map::with_capacity(pool.len());
    for w in pool {
        loads.insert(
            w.url.clone(),
            Value::Array(vec![
                w.decode_charged_tokens().into(),
                (w.decode_charged_reqs() as u64).into(),
                w.kv_capacity_tokens().unwrap_or(0).into(),
            ]),
        );
    }
    DecodeTokenLog {
        rule,
        charge_tokens,
        chosen_charged_tokens: chosen.decode_charged_tokens(),
        normalized,
        loads: Value::Object(loads),
    }
}

/// The request's output-token limit, if it states one.
pub fn requested_output_tokens(body: &Value) -> Option<u64> {
    ["max_tokens", "max_completion_tokens"]
        .iter()
        .find_map(|k| body.get(*k).and_then(Value::as_u64))
        .or_else(|| {
            body.get("sampling_params")
                .and_then(|s| s.get("max_new_tokens"))
                .and_then(Value::as_u64)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use serde_json::json;

    fn decode(url: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(url.into()),
            url: url.into(),
            mode: WorkerMode::Decode,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
        }))
    }

    fn pool(n: usize) -> Vec<Arc<Worker>> {
        // Deliberately not in URL order: selection must not depend on slice order.
        (0..n)
            .rev()
            .map(|i| decode(&format!("http://d{i}")))
            .collect()
    }

    fn by_url<'a>(ws: &'a [Arc<Worker>], url: &str) -> &'a Arc<Worker> {
        ws.iter().find(|w| w.url == url).unwrap()
    }

    fn min_balancer() -> DecodeTokenBalancer {
        DecodeTokenBalancer::new(
            TokenRule::MinInflightTokens,
            PdDecodeTokensConfig::default(),
        )
    }

    #[test]
    fn select_among_matches_select_on_the_set_and_charges() {
        let b = min_balancer();
        let ws = pool(4); // d3, d2, d1, d0
        let _c0 = by_url(&ws, "http://d0").charge_decode_tokens(100);
        let _c2 = by_url(&ws, "http://d2").charge_decode_tokens(50);
        // Set = d3 (idx 0) and d0 (idx 3): d3 holds nothing and wins.
        let (w, charge) = b.select_among(&ws, &[3, 0], 10).unwrap();
        assert_eq!((ws[w].url.as_str(), charge.tokens()), ("http://d3", 10));
        assert_eq!(by_url(&ws, "http://d3").decode_charged_tokens(), 10);
        drop(charge);
        // Equal loads: URL order (d0 before d3), whatever the set order.
        let _c3 = by_url(&ws, "http://d3").charge_decode_tokens(100);
        let (w, _charge) = b.select_among(&ws, &[0, 3], 10).unwrap();
        assert_eq!(ws[w].url, "http://d0");
        // Same tokens: fewer charged requests first.
        let ws = pool(2);
        let _a = by_url(&ws, "http://d0").charge_decode_tokens(50);
        let _b = by_url(&ws, "http://d0").charge_decode_tokens(50);
        let _c = by_url(&ws, "http://d1").charge_decode_tokens(100);
        let (w, _charge) = b.select_among(&ws, &[0, 1], 1).unwrap();
        assert_eq!(ws[w].url, "http://d1");
        assert!(b.select_among(&ws, &[], 1).is_none());
    }

    #[test]
    fn charge_and_move_charge_keep_the_sums() {
        let b = min_balancer();
        let ws = pool(2);
        let (d0, d1) = (by_url(&ws, "http://d0"), by_url(&ws, "http://d1"));
        let c = b.charge(d0, 700);
        assert_eq!(
            (d0.decode_charged_tokens(), d0.decode_charged_reqs()),
            (700, 1)
        );
        let c = b.move_charge(c, d1);
        assert_eq!(
            (d0.decode_charged_tokens(), d0.decode_charged_reqs()),
            (0, 0)
        );
        assert_eq!(
            (d1.decode_charged_tokens(), d1.decode_charged_reqs()),
            (700, 1)
        );
        assert_eq!(c.tokens(), 700);
        drop(c);
        assert_eq!(
            (d1.decode_charged_tokens(), d1.decode_charged_reqs()),
            (0, 0)
        );
    }

    #[test]
    fn for_policy_is_some_only_for_token_policies() {
        let cfg = PdDecodeTokensConfig::default;
        for p in [
            PdDecodePolicy::AffinityMinLoad,
            PdDecodePolicy::PowerOfTwo,
            PdDecodePolicy::RoundRobin,
            PdDecodePolicy::PowerOfTwoSmg,
        ] {
            assert!(DecodeTokenBalancer::for_policy(p, cfg()).is_none());
            assert!(!p.is_token_charged());
        }
        let m = DecodeTokenBalancer::for_policy(PdDecodePolicy::MinInflightTokens, cfg()).unwrap();
        assert_eq!(m.rule(), TokenRule::MinInflightTokens);
        let p = DecodeTokenBalancer::for_policy(PdDecodePolicy::PowerOfTwoTokens, cfg()).unwrap();
        assert_eq!(p.rule(), TokenRule::PowerOfTwoTokens);
        assert!(PdDecodePolicy::MinInflightTokens.is_token_charged());
        assert!(PdDecodePolicy::PowerOfTwoTokens.is_token_charged());
    }

    #[test]
    fn request_charge_adds_capped_output() {
        let b = DecodeTokenBalancer::new(
            TokenRule::MinInflightTokens,
            PdDecodeTokensConfig {
                output_charge_cap: 100,
                normalize_by_capacity: false,
            },
        );
        assert_eq!(
            b.request_charge(1000, Some(&json!({"max_tokens": 7}))),
            1007
        );
        assert_eq!(
            b.request_charge(1000, Some(&json!({"max_tokens": 1_000_000}))),
            1100
        );
        assert_eq!(
            b.request_charge(1000, Some(&json!({"max_completion_tokens": 9}))),
            1009
        );
        assert_eq!(
            b.request_charge(
                1000,
                Some(&json!({"sampling_params": {"max_new_tokens": 3}}))
            ),
            1003
        );
        // Absent / unparsed / non-integer → the cap.
        assert_eq!(b.request_charge(1000, Some(&json!({}))), 1100);
        assert_eq!(b.request_charge(1000, None), 1100);
        assert_eq!(
            b.request_charge(1000, Some(&json!({"max_tokens": "x"}))),
            1100
        );
        // A zero-length prompt still charges one token.
        assert_eq!(b.request_charge(0, Some(&json!({"max_tokens": 0}))), 1);
    }

    #[test]
    fn min_tokens_picks_least_charged_not_least_requests() {
        let b = min_balancer();
        let ws = pool(3);
        // d0: one 100k request; d1: three 1k requests; d2: one 50k request.
        let _h = by_url(&ws, "http://d0").charge_decode_tokens(100_000);
        let _l: Vec<_> = (0..3)
            .map(|_| by_url(&ws, "http://d1").charge_decode_tokens(1_000))
            .collect();
        let _m = by_url(&ws, "http://d2").charge_decode_tokens(50_000);
        let pick = b.select(&ws, 10).unwrap();
        assert_eq!(
            pick.worker.url, "http://d1",
            "fewest tokens wins over fewest requests"
        );
        assert_eq!(pick.charge.tokens(), 10);
        assert_eq!(by_url(&ws, "http://d1").decode_charged_tokens(), 3_010);
        assert_eq!(pick.log.rule, "min-inflight-tokens");
        assert_eq!(pick.log.charge_tokens, 10);
        assert_eq!(pick.log.chosen_charged_tokens, 3_010);
        assert_eq!(pick.log.loads["http://d0"], json!([100_000, 1, 0]));
        assert_eq!(pick.log.loads["http://d1"], json!([3_010, 4, 0]));
        assert_eq!(pick.log.loads["http://d2"], json!([50_000, 1, 0]));
        drop(pick);
        assert_eq!(by_url(&ws, "http://d1").decode_charged_tokens(), 3_000);
        assert_eq!(by_url(&ws, "http://d1").decode_charged_reqs(), 3);
    }

    #[test]
    fn ties_break_by_fewest_requests_then_url_order() {
        let b = min_balancer();
        let ws = pool(3);
        // All empty: URL order → d0.
        assert_eq!(b.select(&ws, 5).unwrap().worker.url, "http://d0");
        // d0 and d1 tie on tokens (2_000); d1 holds fewer requests; d2 heavier.
        let _a: Vec<_> = (0..2)
            .map(|_| by_url(&ws, "http://d0").charge_decode_tokens(1_000))
            .collect();
        let _b = by_url(&ws, "http://d1").charge_decode_tokens(2_000);
        let _c = by_url(&ws, "http://d2").charge_decode_tokens(3_000);
        assert_eq!(b.select(&ws, 5).unwrap().worker.url, "http://d1");
        // Full tie on tokens and requests between d0 and d1 → URL order.
        let _b2 = by_url(&ws, "http://d1").charge_decode_tokens(0);
        assert_eq!(b.select(&ws, 5).unwrap().worker.url, "http://d0");
    }

    #[test]
    fn heavy_sequence_spreads_by_tokens() {
        // H, L, L, L, H: count balancing would alternate; token balancing
        // keeps every light request off the first heavy one's decode and
        // puts the second heavy request on the other decode.
        let b = min_balancer();
        let ws = pool(2);
        let mut held = Vec::new();
        let mut placed = Vec::new();
        for tokens in [200_000u64, 1_000, 1_000, 1_000, 200_000] {
            let p = b.select(&ws, tokens).unwrap();
            placed.push(p.worker.url.clone());
            held.push(p);
        }
        assert_eq!(placed[0], "http://d0");
        assert!(placed[1..4].iter().all(|u| u == "http://d1"), "{placed:?}");
        assert_eq!(placed[4], "http://d1");
        let d0 = by_url(&ws, "http://d0");
        let d1 = by_url(&ws, "http://d1");
        assert_eq!(
            (d0.decode_charged_tokens(), d0.decode_charged_reqs()),
            (200_000, 1)
        );
        assert_eq!(
            (d1.decode_charged_tokens(), d1.decode_charged_reqs()),
            (203_000, 4)
        );
        drop(held);
        assert_eq!(d0.decode_charged_tokens() + d1.decode_charged_tokens(), 0);
        assert_eq!(d0.decode_charged_reqs() + d1.decode_charged_reqs(), 0);
    }

    #[test]
    fn open_breaker_is_excluded_and_all_open_is_total() {
        let b = min_balancer();
        let ws = pool(2);
        let d0 = by_url(&ws, "http://d0");
        for _ in 0..3 {
            d0.breaker.record_failure();
        }
        assert!(!d0.breaker.would_allow());
        let _heavy = by_url(&ws, "http://d1").charge_decode_tokens(1_000_000);
        // d0 is empty but breaker-open: never chosen while d1 is healthy.
        for _ in 0..10 {
            let p = b.select(&ws, 1).unwrap();
            assert_eq!(p.worker.url, "http://d1");
            assert!(p.log.loads.get("http://d0").is_none());
        }
        let d1 = by_url(&ws, "http://d1");
        for _ in 0..3 {
            d1.breaker.record_failure();
        }
        // Every breaker open: still total, min over all.
        assert_eq!(b.select(&ws, 1).unwrap().worker.url, "http://d0");
        assert!(b.select(&[], 1).is_none());
    }

    #[test]
    fn capacity_normalization_applies_only_when_every_decode_reports() {
        let cfg = PdDecodeTokensConfig {
            output_charge_cap: 0,
            normalize_by_capacity: true,
        };
        let b = DecodeTokenBalancer::new(TokenRule::MinInflightTokens, cfg);
        let ws = pool(2);
        let d0 = by_url(&ws, "http://d0");
        let d1 = by_url(&ws, "http://d1");
        let _a = d0.charge_decode_tokens(300_000);
        let _b = d1.charge_decode_tokens(200_000);
        // d1 is reported, d0 is not → raw tokens: d1 (200k) wins.
        d1.set_kv_capacity_tokens(250_000);
        let p = b.select(&ws, 10_000).unwrap();
        assert_eq!(p.worker.url, "http://d1");
        assert!(!p.log.normalized);
        drop(p);
        // Both reported: d0 at 310k/1M = 0.31 beats d1 at 210k/250k = 0.84.
        d0.set_kv_capacity_tokens(1_000_000);
        let p = b.select(&ws, 10_000).unwrap();
        assert_eq!(p.worker.url, "http://d0");
        assert!(p.log.normalized);
        assert_eq!(p.log.loads["http://d0"], json!([310_000, 2, 1_000_000]));
        drop(p);
        // Same capacities without the flag → raw tokens again.
        let raw = min_balancer();
        assert_eq!(raw.select(&ws, 10_000).unwrap().worker.url, "http://d1");
    }

    #[test]
    fn po2_tokens_takes_the_lower_of_two_samples() {
        let b =
            DecodeTokenBalancer::new(TokenRule::PowerOfTwoTokens, PdDecodeTokensConfig::default());
        // Two candidates: both sampled every time → deterministic lower.
        let ws = pool(2);
        let _h = by_url(&ws, "http://d0").charge_decode_tokens(500_000);
        let _l: Vec<_> = (0..5)
            .map(|_| by_url(&ws, "http://d1").charge_decode_tokens(1_000))
            .collect();
        for _ in 0..50 {
            let p = b.select(&ws, 1).unwrap();
            assert_eq!(p.worker.url, "http://d1");
            assert_eq!(p.log.rule, "po2-tokens");
        }
        // Three candidates: the heaviest is never chosen (it loses to
        // whichever other one is sampled with it), and both others appear.
        let ws = pool(3);
        let _x = by_url(&ws, "http://d2").charge_decode_tokens(10_000_000);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            let p = b.select(&ws, 1).unwrap();
            assert_ne!(p.worker.url, "http://d2");
            seen.insert(p.worker.url.clone());
        }
        assert_eq!(seen.len(), 2);
        // Single candidate.
        let one = pool(1);
        assert_eq!(b.select(&one, 1).unwrap().worker.url, "http://d0");
    }

    #[test]
    fn concurrent_picks_charge_every_request_and_release_to_zero() {
        let b = Arc::new(min_balancer());
        let ws = Arc::new(pool(4));
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let b = Arc::clone(&b);
                let ws = Arc::clone(&ws);
                std::thread::spawn(move || {
                    let mut held = Vec::new();
                    for i in 0..50u64 {
                        held.push(b.select(&ws, 1 + (t * 50 + i) % 7).unwrap());
                        if i % 3 == 0 {
                            drop(held.remove(0));
                        }
                    }
                    held.len()
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        // Every thread's picks were dropped when it returned.
        for w in ws.iter() {
            assert_eq!((w.decode_charged_tokens(), w.decode_charged_reqs()), (0, 0));
        }
    }

    #[test]
    fn concurrent_equal_picks_spread_evenly() {
        // 4 decodes, 400 concurrent equal charges, all held: the lock makes
        // read-min-then-charge atomic, so each decode ends with exactly 100.
        let b = Arc::new(min_balancer());
        let ws = Arc::new(pool(4));
        let picks = Arc::new(Mutex::new(Vec::new()));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let (b, ws, picks) = (Arc::clone(&b), Arc::clone(&ws), Arc::clone(&picks));
                std::thread::spawn(move || {
                    for _ in 0..50 {
                        let p = b.select(&ws, 1_000).unwrap();
                        picks.lock().push(p);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        for w in ws.iter() {
            assert_eq!(w.decode_charged_reqs(), 100, "{}", w.url);
            assert_eq!(w.decode_charged_tokens(), 100 * 1_000, "{}", w.url);
        }
        picks.lock().clear();
        for w in ws.iter() {
            assert_eq!(w.decode_charged_tokens(), 0);
        }
    }

    #[test]
    fn pinned_charge_is_counted_and_logged() {
        let b = min_balancer();
        let ws = pool(2);
        let d1 = by_url(&ws, "http://d1");
        let (charge, log) = b.charge_pinned(d1, &ws, 4_000);
        assert_eq!(log.rule, "replay_pinned");
        assert_eq!(log.chosen_charged_tokens, 4_000);
        assert_eq!(log.loads["http://d1"], json!([4_000, 1, 0]));
        // The next pick sees the pinned load.
        assert_eq!(b.select(&ws, 1).unwrap().worker.url, "http://d0");
        drop(charge);
        assert_eq!(d1.decode_charged_tokens(), 0);
    }

    #[test]
    fn requested_output_tokens_reads_known_fields() {
        assert_eq!(requested_output_tokens(&json!({"max_tokens": 5})), Some(5));
        assert_eq!(
            requested_output_tokens(&json!({"max_completion_tokens": 6})),
            Some(6)
        );
        assert_eq!(
            requested_output_tokens(&json!({"sampling_params": {"max_new_tokens": 8}})),
            Some(8)
        );
        assert_eq!(requested_output_tokens(&json!({"max_tokens": null})), None);
        assert_eq!(requested_output_tokens(&json!({"max_tokens": -1})), None);
    }
}
