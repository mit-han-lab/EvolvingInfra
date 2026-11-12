// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Global waiting queue (GWQ) of the TRADE placement
//! (`--co-schedule-global-queue sjf|hrrn`).
//!
//! The router holds P-leg requests in one queue and keeps each P engine's
//! waiting queue short. A P has room while its queued work on the router
//! ledger (the work beyond the running pass, [`queued_tokens`]) is below the
//! depth (or its progress view is stale: fail-open). Each decision step
//! walks the held requests in release order (`sjf`: smallest service on the
//! plain TRADE choice first; `hrrn`: highest (wait + service) / service
//! first; starved requests before both) and releases the first request
//! whose TRADE argmin P has room, to that P (late P binding). The requests
//! ahead of a request in the walk count as a virtual backlog of the P that
//! they go to.
//!
//! This module holds the configuration and the pure walk
//! ([`walk`]); [`crate::cost_balance::CostBook::gwq_step`] runs it under the
//! cost-book lock and [`crate::co_schedule::CoScheduler`] owns the queue.

use std::time::Duration;

use crate::scheduler::trade::{
    select_trade, trade_tie_break, TradeInput, TradeParams, TradeTieBreak, TradeTieOutcome,
};

/// Default `--co-schedule-global-queue-max-wait-ms` of `hrrn`: only a safety
/// net (for example a ledger that does not move), far above the longest
/// router hold of the GWQ replay (60-65 s for requests of 64k or more
/// uncached tokens at depth 16384), so that it does not change the order.
pub const DEFAULT_HRRN_MAX_WAIT_MS: u64 = 300_000;

/// One engine pass: the chunked-prefill size of the GB300 GLM-5.2 deployments
/// (`--chunked-prefill-size` = `--max-prefill-tokens` = 8192). It is the
/// look-ahead pass of the `auto` depth and the running pass that the queued
/// work leaves out ([`queued_tokens`]). The router has the engine chunk only
/// from the `prefill_routing` snapshot of `--trade-wait eft`,
/// which the queue does not support.
pub const CHUNK_TOKENS: u64 = 8192;

/// The dispatch delay of the `auto` depth (ms): the time from a release to
/// the request in the P waiting queue (P-leg HTTP, bootstrap, engine
/// tokenization). GB300 GLM-5.2 replays measured p50 0.21 s, p90 0.59 s
/// and p99 1.09 s for it. A GWQ replay of the same workload found the best
/// depth at 12k-16k tokens (8192 keeps 43-61 % of the gain, 32k and more
/// 62-78 %; 16384 beats 11756); with the 100 ms poll this allowance gives
/// 16292.
pub const AUTO_DISPATCH_DELAY_MS: u64 = 500;

/// A progress view is stale after this many poll intervals ...
const STALE_AFTER_POLLS: u64 = 10;
/// ... and never before this time.
const STALE_AFTER_MIN: Duration = Duration::from_secs(1);

/// Release order of the queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalQueueMode {
    /// Highest response ratio (wait + S) / S first (no knob for starvation).
    Hrrn,
}

impl GlobalQueueMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hrrn => "hrrn",
        }
    }
}

/// `--co-schedule-global-queue-room`: how the queue decides that a P has
/// room.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GlobalQueueRoom {
    /// Queued tokens on the router ledger below the depth (the first
    /// version).
    #[default]
    Tokens,
    /// The predicted drain time of the P (the E3 forecast state of
    /// `--trade-e3-bound-table`, with the ledger or the e3 wait:
    /// the wait of a cold one-chunk probe) is
    /// at most the lead = depth / R; the walk then also prices the P's wait
    /// as R x drain. A P without a usable forecast keeps the token rule.
    Predicted,
}

impl GlobalQueueRoom {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tokens => "tokens",
            Self::Predicted => "predicted",
        }
    }
}

/// The queued work of a P: the work beyond its running pass. `unlaunched`:
/// the uncached tokens routed to the P whose first prefill chunk is not in a
/// progress snapshot yet (transit, bootstrap queue, engine waiting queue).
/// `launched`: the remaining work of its launched requests (the engine
/// reports it after each chunk). One pass of `pass_tokens` of the launched
/// work is the running pass; the rest is the look-ahead batch and the rest
/// of a chunked request, which the requests released now wait for.
pub fn queued_tokens(unlaunched: u128, launched: u128, pass_tokens: u64) -> u128 {
    unlaunched.saturating_add(launched.saturating_sub(u128::from(pass_tokens)))
}

/// The two largest values of `hits` (0 when missing): h1 >= h2.
pub fn top_two(hits: &[u64]) -> (u64, u64) {
    let (mut first, mut second) = (0u64, 0u64);
    for &hit in hits {
        if hit > first {
            second = first;
            first = hit;
        } else if hit > second {
            second = hit;
        }
    }
    (first, second)
}

/// `--symmetric-pd-gwq-auto` (and `--co-schedule-global-queue-hold free` of the
/// co_schedule path): true when the request is bound to its cache owner: the
/// largest local hit over the pool exceeds the second largest by at least
/// `chunk` tokens (one engine pass), h1 - h2 >= C. Moving such a request
/// costs at least one full pass of recomputation, or a pull at the copy
/// price, so the router never holds it; TRADE decides it at arrival. A cold
/// request (no hit) or one whose prefix is on two P within one chunk is free.
/// One P (or none): bound only when its hit is at least `chunk`.
pub fn bound_to_owner(local_hits: &[u64], chunk: u64) -> bool {
    let (first, second) = top_two(local_hits);
    first - second >= chunk
}

/// The age after which a P's progress view is stale (fail-open):
/// max(1 s, 10 x the poll interval).
pub fn stale_after(load_poll_ms: u64) -> Duration {
    Duration::from_millis(load_poll_ms.saturating_mul(STALE_AFTER_POLLS)).max(STALE_AFTER_MIN)
}

/// Why a request left the queue for a P.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GwqRelease {
    /// Its TRADE argmin P had room (fresh view, queued below the depth).
    Room,
    /// Its TRADE argmin P had no fresh progress view (fail-open).
    StaleView,
    /// It was held for the max wait and its P had no room.
    Starved,
    /// `--co-schedule-global-queue-room predicted`: the P's predicted drain
    /// time was at most the lead.
    Predicted,
}

impl GwqRelease {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Room => "room",
            Self::StaleView => "stale_view",
            Self::Starved => "starved",
            Self::Predicted => "predicted",
        }
    }
}

/// One P of a decision step, in pool order.
#[derive(Debug, Clone, PartialEq)]
pub struct GwqRow {
    /// The TRADE backlog of P from the ledger (as a TRADE decision uses it).
    pub backlog: u128,
    /// The queued work of P ([`queued_tokens`]).
    pub queued: u128,
    pub depth: u64,
    /// A progress snapshot of P was applied and is not stale.
    pub fresh: bool,
    /// a_w of the HOL term (0 while it is off).
    pub arrival_rate_per_s: f64,
    /// The copy price of a pull to this P (`--trade-pull-price
    /// phi`: lambda_q x phi_P); None = the copy price of the parameters.
    pub copy_lambda: Option<f64>,
    /// `room = predicted`: the P's predicted drain time (s) from a usable
    /// forecast; None for the token rule alone (room `tokens`, or a P without
    /// a usable forecast: fail-open to the token rule).
    pub drain_s: Option<f64>,
    /// `room = predicted`: the lead (s); a P with a forecast has room while
    /// its drain is at most the lead.
    pub lead_s: f64,
}

impl GwqRow {
    /// Why P can take a request now; None without room.
    pub fn room(&self) -> Option<GwqRelease> {
        // The forecast may release earlier than the ledger, never later: a
        // drain within the lead gives room; otherwise the token rule (the
        // work-conserving floor) still decides.
        if self.drain_s.is_some_and(|d| d <= self.lead_s) {
            return Some(GwqRelease::Predicted);
        }
        if !self.fresh {
            Some(GwqRelease::StaleView)
        } else if self.queued < u128::from(self.depth) {
            Some(GwqRelease::Room)
        } else {
            None
        }
    }
}

/// The TRADE terms of one held request on one P (clamped as a TRADE
/// decision clamps them; see [`terms_of`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GwqTerms {
    pub local_hit: u64,
    /// u(P): tokens P computes after the hinted pull.
    pub uncached: u64,
    pub pulled: u64,
}

/// The terms of a candidate with `local_hit` and `expected_hit` (own hit,
/// plus the pull when a hint is sent) for a request of `input_tokens`.
pub fn terms_of(local_hit: u64, expected_hit: u64, input_tokens: u64) -> GwqTerms {
    let local_hit = local_hit.min(input_tokens);
    let expected_hit = expected_hit.min(input_tokens);
    GwqTerms {
        local_hit,
        uncached: input_tokens - expected_hit,
        pulled: expected_hit.saturating_sub(local_hit),
    }
}

/// One held request of a decision step, in queue (arrival) order.
#[derive(Debug, Clone)]
pub struct GwqHeldInput<'a> {
    /// One entry per row.
    pub terms: &'a [GwqTerms],
    /// Seconds in the queue.
    pub waited_s: f64,
    /// Held for at least the max wait.
    pub starved: bool,
}

/// The request that a step releases.
#[derive(Debug, Clone, PartialEq)]
pub struct GwqPick {
    /// Index of the request in the held list.
    pub index: usize,
    /// Row of its P: the TRADE argmin with the virtual backlog.
    pub worker: usize,
    pub release: GwqRelease,
    /// Held for at least the max wait (it went first).
    pub starved: bool,
    /// Its position in the walk order.
    pub rank: usize,
    pub key: f64,
    pub service_s: f64,
    /// Row of its plain choice (no virtual backlog).
    pub plain_worker: usize,
    /// The virtual backlog per row at this request.
    pub virtual_tokens: Vec<u128>,
    /// `--symmetric-pd-gwq-auto`: the tie-break of its TRADE choice (None
    /// without it).
    pub tie: Option<TradeTieOutcome>,
}

/// The result of one walk.
#[derive(Debug, Clone, PartialEq)]
pub struct GwqWalk {
    /// The plain choice (row) of every held request.
    pub plain: Vec<usize>,
    /// The service seconds of every held request on its plain choice.
    pub service_s: Vec<f64>,
    /// The released request; None holds every request.
    pub pick: Option<GwqPick>,
}

/// The release key of a request: (waited + S) / S with S at least `floor_s`
/// (larger first).
pub fn key(waited_s: f64, service_s: f64, floor_s: f64) -> f64 {
    let s = service_s.max(floor_s);
    (waited_s.max(0.0) + s) / s
}

fn trade_inputs<'a>(
    urls: &'a [String],
    rows: &[GwqRow],
    terms: &[GwqTerms],
    virtual_tokens: &[u128],
) -> Vec<TradeInput<'a>> {
    rows.iter()
        .zip(terms)
        .enumerate()
        .map(|(i, (row, t))| TradeInput {
            worker: urls[i].as_str(),
            backlog: row.backlog.saturating_add(virtual_tokens[i]),
            local_hit: t.local_hit,
            uncached: t.uncached,
            pulled: t.pulled,
            wait: None,
            arrival_rate_per_s: row.arrival_rate_per_s,
            copy_lambda: row.copy_lambda,
            bound_s: None,
        })
        .collect()
}

/// `--symmetric-pd-gwq-auto`: the release tie-break inputs of one step (one entry
/// per row; see [`trade_tie_break`]).
#[derive(Debug, Clone, Copy)]
pub struct GwqTie<'a> {
    /// Outstanding requests on each P (the cost book's in-flight count).
    pub outstanding: &'a [u64],
    /// The URL of the last tie-broken release.
    pub last: Option<&'a str>,
}

/// The TRADE choice of one request in a step: [`select_trade`], and with
/// `tie` its exact ties broken by [`trade_tie_break`] (room, own hit,
/// outstanding requests, rotation). Returns the row, its service (s) and,
/// with `tie`, the tie-break outcome.
fn choose(
    params: &TradeParams,
    inputs: &[TradeInput<'_>],
    rows: &[GwqRow],
    tie: Option<&GwqTie<'_>>,
) -> (usize, f64, Option<TradeTieOutcome>) {
    let selection = select_trade(params, inputs);
    let (chosen, outcome) = match tie {
        Some(tie) => {
            let room: Vec<bool> = rows.iter().map(|r| r.room().is_some()).collect();
            let breaker = TradeTieBreak {
                room: &room,
                outstanding: tie.outstanding,
                last: tie.last,
            };
            let outcome = trade_tie_break(&selection, inputs, &breaker);
            (outcome.chosen, Some(outcome))
        }
        None => (selection.chosen, None),
    };
    (chosen, selection.costs[chosen].service_s, outcome)
}

/// [`walk`] with the `--symmetric-pd-gwq-auto` tie-break (`tie`; None = the
/// plain [`select_trade`] ties, as the co_schedule queue uses them).
pub fn walk_with(
    params: &TradeParams,
    urls: &[String],
    rows: &[GwqRow],
    held: &[GwqHeldInput<'_>],
    tie: Option<&GwqTie<'_>>,
) -> GwqWalk {
    assert!(!rows.is_empty() && urls.len() == rows.len());
    if let Some(tie) = tie {
        assert_eq!(
            tie.outstanding.len(),
            rows.len(),
            "one outstanding count per P"
        );
    }
    let zero = vec![0u128; rows.len()];
    let mut plain = Vec::with_capacity(held.len());
    let mut service_s = Vec::with_capacity(held.len());
    for h in held {
        assert_eq!(h.terms.len(), rows.len(), "one term per P");
        let inputs = trade_inputs(urls, rows, h.terms, &zero);
        let (chosen, service, _) = choose(params, &inputs, rows, tie);
        plain.push(chosen);
        service_s.push(service);
    }
    let any_room = rows.iter().any(|r| r.room().is_some());
    if held.is_empty() || !(any_room || held.iter().any(|h| h.starved)) {
        return GwqWalk {
            plain,
            service_s,
            pick: None,
        };
    }
    let floor_s = 1.0 / params.rate_tokens_per_s;
    let keys: Vec<f64> = held
        .iter()
        .zip(&service_s)
        .map(|(h, s)| key(h.waited_s, *s, floor_s))
        .collect();
    let mut order: Vec<usize> = (0..held.len()).collect();
    order.sort_by(|&a, &b| {
        held[b]
            .starved
            .cmp(&held[a].starved)
            .then_with(|| {
                if held[a].starved {
                    std::cmp::Ordering::Equal
                } else {
                    keys[b].total_cmp(&keys[a])
                }
            })
            .then_with(|| a.cmp(&b))
    });
    let mut virtual_tokens = vec![0u128; rows.len()];
    for (rank, &i) in order.iter().enumerate() {
        let h = &held[i];
        let inputs = trade_inputs(urls, rows, h.terms, &virtual_tokens);
        let (chosen, _, tie_outcome) = choose(params, &inputs, rows, tie);
        let release = match (rows[chosen].room(), h.starved) {
            (Some(release), _) => Some(release),
            (None, true) => Some(GwqRelease::Starved),
            (None, false) => None,
        };
        if let Some(release) = release {
            return GwqWalk {
                pick: Some(GwqPick {
                    index: i,
                    worker: chosen,
                    release,
                    starved: h.starved,
                    rank,
                    key: keys[i],
                    service_s: service_s[i],
                    plain_worker: plain[i],
                    virtual_tokens,
                    tie: tie_outcome,
                }),
                plain,
                service_s,
            };
        }
        virtual_tokens[chosen] =
            virtual_tokens[chosen].saturating_add(u128::from(h.terms[chosen].uncached));
    }
    GwqWalk {
        plain,
        service_s,
        pick: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`walk_with`] without the tie break of `--symmetric-pd-gwq-auto`.
    fn walk(
        _mode: GlobalQueueMode,
        params: &TradeParams,
        urls: &[String],
        rows: &[GwqRow],
        held: &[GwqHeldInput<'_>],
    ) -> GwqWalk {
        walk_with(params, urls, rows, held, None)
    }

    const R: f64 = 13_500.0;

    fn params(lambda: f64) -> TradeParams {
        TradeParams {
            copy_lambda: lambda,
            ..TradeParams::default()
        }
    }

    fn urls(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("http://p{i}")).collect()
    }

    fn row(backlog: u128, queued: u128, depth: u64) -> GwqRow {
        GwqRow {
            backlog,
            queued,
            depth,
            fresh: true,
            arrival_rate_per_s: 0.0,
            copy_lambda: None,
            drain_s: None,
            lead_s: 0.0,
        }
    }

    /// A cold request of `u` uncached tokens on every P.
    fn cold(u: u64, n: usize) -> Vec<GwqTerms> {
        vec![terms_of(0, 0, u); n]
    }

    fn held(terms: &[GwqTerms], waited_s: f64) -> GwqHeldInput<'_> {
        GwqHeldInput {
            terms,
            waited_s,
            starved: false,
        }
    }

    /// Queued work: the unlaunched work plus the launched work beyond one
    /// pass (the look-ahead batch and the rest of a chunked request).
    #[test]
    fn queued_work_is_the_work_beyond_the_running_pass() {
        assert_eq!(queued_tokens(5000, 0, 8192), 5000);
        assert_eq!(queued_tokens(5000, 8192, 8192), 5000);
        assert_eq!(queued_tokens(0, 100_000, 8192), 91_808);
        assert_eq!(queued_tokens(2000, 12_000, 8192), 5808);
    }

    #[test]
    fn terms_match_the_trade_clamps() {
        assert_eq!(
            terms_of(1000, 5000, 8000),
            GwqTerms {
                local_hit: 1000,
                uncached: 3000,
                pulled: 4000
            }
        );
        // Hits above the request length are clamped to it.
        assert_eq!(
            terms_of(9000, 9500, 8000),
            GwqTerms {
                local_hit: 8000,
                uncached: 0,
                pulled: 0
            }
        );
    }

    #[test]
    fn nothing_is_released_while_no_p_has_room() {
        let urls = urls(2);
        let rows = [row(20_000, 20_000, 9542), row(15_000, 9542, 9542)];
        let a = cold(4000, 2);
        let w = walk(
            GlobalQueueMode::Hrrn,
            &params(9.2),
            &urls,
            &rows,
            &[held(&a, 1.0)],
        );
        assert!(w.pick.is_none());
        // The plain choice is still computed: the lower backlog.
        assert_eq!(w.plain, vec![1]);
        assert!((w.service_s[0] - 4000.0 / R).abs() < 1e-12);
    }

    #[test]
    fn hrrn_releases_a_long_waiting_big_request_before_a_new_small_one() {
        let urls = urls(1);
        let rows = [row(0, 0, 9542)];
        let (big, small) = (cold(50_000, 1), cold(2000, 1));
        let s_big = 50_000.0 / R;
        let s_small = 2000.0 / R;
        // Big waited 30 s: (30 + 3.7) / 3.7 = 9.1 > small (0.1 + 0.15) / 0.15 = 1.7.
        let w = walk(
            GlobalQueueMode::Hrrn,
            &params(9.2),
            &urls,
            &rows,
            &[held(&big, 30.0), held(&small, 0.1)],
        );
        let pick = w.pick.unwrap();
        assert_eq!(pick.index, 0);
        assert!((pick.key - (30.0 + s_big) / s_big).abs() < 1e-9);
        // Big waited 1 s: 1.27 < 1.67, the small request goes first.
        let w = walk(
            GlobalQueueMode::Hrrn,
            &params(9.2),
            &urls,
            &rows,
            &[held(&big, 1.0), held(&small, 0.1)],
        );
        let pick = w.pick.unwrap();
        assert_eq!(pick.index, 1);
        assert!((pick.key - (0.1 + s_small) / s_small).abs() < 1e-9);
        // A fully cached request (S = 0) uses one token of service: after
        // 10 ms its ratio is 0.01 x R + 1 = 136.
        let cached = [terms_of(2000, 2000, 2000)];
        let k = key(0.5, 0.0, 1.0 / R);
        assert!((k - (0.5 * R + 1.0)).abs() < 1e-6, "{k}");
        let w = walk(
            GlobalQueueMode::Hrrn,
            &params(9.2),
            &urls,
            &rows,
            &[held(&small, 0.1), held(&cached, 0.01)],
        );
        assert_eq!(w.pick.unwrap().index, 1);
    }

    #[test]
    fn equal_keys_release_the_older_request_first() {
        let urls = urls(1);
        let rows = [row(0, 0, 9542)];
        let a = cold(3000, 1);
        let w = walk(
            GlobalQueueMode::Hrrn,
            &params(9.2),
            &urls,
            &rows,
            &[held(&a, 2.0), held(&a, 1.0)],
        );
        assert_eq!(w.pick.unwrap().index, 0);
    }

    /// Late P binding: at arrival the TRADE choice is P0 (no room), so the
    /// request waits. When P1's backlog falls, the TRADE argmin at release is
    /// P1, which has room: the request goes to P1.
    #[test]
    fn late_binding_releases_to_the_trade_argmin_at_release_time() {
        let urls = urls(2);
        let a = cold(5000, 2);
        // P1 runs a big launched request: backlog 40k, nothing queued.
        let rows = [row(30_000, 30_000, 9542), row(40_000, 0, 9542)];
        let w = walk(
            GlobalQueueMode::Hrrn,
            &params(9.2),
            &urls,
            &rows,
            &[held(&a, 0.0)],
        );
        assert_eq!(w.plain, vec![0]);
        assert!(w.pick.is_none(), "P0 (35k) beats P1 (45k) but has no room");
        let rows = [row(30_000, 30_000, 9542), row(20_000, 0, 9542)];
        let w = walk(
            GlobalQueueMode::Hrrn,
            &params(9.2),
            &urls,
            &rows,
            &[held(&a, 0.8)],
        );
        let pick = w.pick.unwrap();
        assert_eq!((pick.worker, pick.plain_worker), (1, 1));
        assert_eq!(pick.release, GwqRelease::Room);
    }

    /// The queue never moves a request away from its TRADE argmin to a P
    /// with room: with the copy price, the cache owner without room keeps it.
    #[test]
    fn a_request_waits_for_its_trade_argmin_and_does_not_move_to_a_worse_p_with_room() {
        let urls = urls(2);
        // 62k prompt, 60k cached on P0; P1 would pull 60k.
        let x = [
            terms_of(60_000, 60_000, 62_000),
            terms_of(0, 60_000, 62_000),
        ];
        let rows = [row(20_000, 20_000, 9542), row(0, 0, 9542)];
        let w = walk(
            GlobalQueueMode::Hrrn,
            &params(18.4),
            &urls,
            &rows,
            &[held(&x, 1.0)],
        );
        assert_eq!(w.plain, vec![0]);
        assert!(
            w.pick.is_none(),
            "P1 has room, but the pull and copy cost more"
        );
        // Without a copy price, TRADE itself moves the request with a pull.
        let w = walk(
            GlobalQueueMode::Hrrn,
            &params(0.0),
            &urls,
            &rows,
            &[held(&x, 1.0)],
        );
        assert_eq!(w.pick.unwrap().worker, 1);
    }

    /// The held work ahead of a request (in release order) counts on the P
    /// it goes to. Two equal cold requests prefer P0 (no room); the second
    /// sees the first on P0 and goes to P1, which has room.
    #[test]
    fn the_virtual_backlog_spreads_held_requests() {
        let urls = urls(2);
        let a = cold(8000, 2);
        let rows = [row(10_000, 10_000, 9542), row(12_000, 0, 9542)];
        let w = walk(
            GlobalQueueMode::Hrrn,
            &params(9.2),
            &urls,
            &rows,
            &[held(&a, 1.0), held(&a, 0.5)],
        );
        assert_eq!(w.plain, vec![0, 0]);
        let pick = w.pick.unwrap();
        assert_eq!((pick.index, pick.worker, pick.rank), (1, 1, 1));
        assert_eq!(pick.virtual_tokens, vec![8000, 0]);
        assert_eq!(pick.plain_worker, 0, "late bound: plain P0, released to P1");
    }

    #[test]
    fn a_starved_request_goes_first_and_without_room() {
        let urls = urls(1);
        let (big, small) = (cold(50_000, 1), cold(1000, 1));
        let no_room = [row(30_000, 30_000, 9542)];
        let starved = GwqHeldInput {
            terms: &big,
            waited_s: 31.0,
            starved: true,
        };
        let w = walk(
            GlobalQueueMode::Hrrn,
            &params(9.2),
            &urls,
            &no_room,
            &[held(&small, 0.1), starved.clone()],
        );
        let pick = w.pick.unwrap();
        assert_eq!((pick.index, pick.rank), (1, 0));
        assert_eq!(pick.release, GwqRelease::Starved);
        assert!(pick.starved);
        // With room it still goes first, released for the room.
        let room = [row(0, 0, 9542)];
        let w = walk(
            GlobalQueueMode::Hrrn,
            &params(9.2),
            &urls,
            &room,
            &[held(&small, 0.1), starved],
        );
        let pick = w.pick.unwrap();
        assert_eq!(
            (pick.index, pick.release, pick.starved),
            (1, GwqRelease::Room, true)
        );
    }

    #[test]
    fn a_stale_or_missing_view_is_room() {
        let urls = urls(1);
        let a = cold(5000, 1);
        let mut stale = row(90_000, 90_000, 9542);
        stale.fresh = false;
        let w = walk(
            GlobalQueueMode::Hrrn,
            &params(9.2),
            &urls,
            &[stale],
            &[held(&a, 0.0)],
        );
        assert_eq!(w.pick.unwrap().release, GwqRelease::StaleView);
        assert_eq!(row(0, 9541, 9542).room(), Some(GwqRelease::Room));
        assert_eq!(row(0, 9542, 9542).room(), None);
    }

    /// `--symmetric-pd-gwq-auto` tie-break in the walk: a cold request on four
    /// idle P goes to the smallest URL without it (`walk`), to the P after
    /// the last rotation with it, to the P with the fewest outstanding
    /// requests first, and to a tied P that has room rather than one without
    /// (a held request ahead adds its virtual backlog as before).
    #[test]
    fn gwq_auto_tie_break_spreads_cold_releases_over_idle_p() {
        let urls = urls(4);
        let idle = vec![row(0, 0, 16_293); 4];
        let a = cold(5000, 4);
        let plain = walk(
            GlobalQueueMode::Hrrn,
            &params(4.6),
            &urls,
            &idle,
            &[held(&a, 0.0)],
        );
        assert_eq!(
            plain.pick.unwrap().worker,
            0,
            "plain TRADE: the smallest URL"
        );
        let pick = |rows: &[GwqRow], outstanding: &[u64], last: Option<&str>| {
            let tie = GwqTie { outstanding, last };
            walk_with(&params(4.6), &urls, rows, &[held(&a, 0.0)], Some(&tie))
                .pick
                .unwrap()
        };
        let mut last: Option<String> = None;
        let mut seen = Vec::new();
        for _ in 0..8 {
            let p = pick(&idle, &[0; 4], last.as_deref());
            assert_eq!(p.release, GwqRelease::Room);
            last = Some(urls[p.worker].clone());
            seen.push(p.worker);
        }
        assert_eq!(seen, [0, 1, 2, 3, 0, 1, 2, 3]);
        assert_eq!(
            pick(&idle, &[2, 1, 0, 1], None).worker,
            2,
            "fewest outstanding"
        );
        // Equal backlog, but p0 and p1 have no room (unlaunched work above the
        // depth): the tied P with room takes the release.
        let rows = [
            row(20_000, 20_000, 16_293),
            row(20_000, 20_000, 16_293),
            row(20_000, 11_808, 16_293),
            row(25_000, 0, 16_293),
        ];
        assert_eq!(pick(&rows, &[0; 4], None).worker, 2);
        // Two held requests, the HRRN order unchanged (b waited longer, rank
        // 0): the rotation after p0 would take p1, which has no room; b goes
        // to p0, the tied P with room.
        let b = cold(5000, 4);
        let tie = GwqTie {
            outstanding: &[0; 4],
            last: Some("http://p0"),
        };
        let full = [
            row(0, 0, 16_293),
            row(0, 20_000, 16_293),
            row(0, 20_000, 16_293),
            row(0, 20_000, 16_293),
        ];
        let w = walk_with(
            &params(4.6),
            &urls,
            &full,
            &[held(&a, 0.0), held(&b, 5.0)],
            Some(&tie),
        );
        let p = w.pick.unwrap();
        assert_eq!((p.index, p.worker, p.rank), (1, 0, 0), "{p:?}");
    }

    #[test]
    fn bound_needs_a_one_chunk_lead_of_the_owner() {
        // Cold, or the same prefix on two P: free.
        assert!(!bound_to_owner(&[0, 0, 0], 8192));
        assert!(!bound_to_owner(&[90_000, 90_000, 0], 8192));
        assert!(
            !bound_to_owner(&[90_000, 82_000, 0], 8192),
            "lead 8000 below one chunk"
        );
        // A lead of at least one chunk: bound (order does not matter).
        assert!(bound_to_owner(&[0, 90_000, 81_808], 8192));
        assert!(bound_to_owner(&[8192, 0, 0], 8192));
        assert!(!bound_to_owner(&[8191, 0, 0], 8192));
        assert!(bound_to_owner(&[50_000], 8192));
        assert!(!bound_to_owner(&[], 8192));
        assert_eq!(GlobalQueueRoom::default(), GlobalQueueRoom::Tokens);
        assert_eq!(GlobalQueueRoom::Predicted.as_str(), "predicted");
    }

    #[test]
    fn predicted_room_uses_the_drain_and_falls_back_to_the_token_rule() {
        let lead = 16_292.0 / R;
        let full = row(0, 50_000, 16_292);
        // A drain within the lead has room even with a full token ledger; a
        // longer drain leaves the decision to the token rule (the floor).
        let near = GwqRow {
            drain_s: Some(lead),
            lead_s: lead,
            ..full.clone()
        };
        assert_eq!(near.room(), Some(GwqRelease::Predicted));
        let far = GwqRow {
            drain_s: Some(lead + 0.01),
            lead_s: lead,
            ..full.clone()
        };
        assert_eq!(far.room(), None, "far and full: no room");
        let far_empty = GwqRow {
            queued: 0,
            ..far.clone()
        };
        assert_eq!(far_empty.room(), Some(GwqRelease::Room), "the token floor");
        let far_stale = GwqRow {
            fresh: false,
            ..far.clone()
        };
        assert_eq!(far_stale.room(), Some(GwqRelease::StaleView));
        // No forecast: the token rule (fail-open on a stale view).
        assert_eq!(
            GwqRow {
                lead_s: lead,
                ..full.clone()
            }
            .room(),
            None
        );
        assert_eq!(
            GwqRow {
                queued: 0,
                lead_s: lead,
                ..full.clone()
            }
            .room(),
            Some(GwqRelease::Room)
        );
        assert_eq!(
            GwqRow {
                fresh: false,
                lead_s: lead,
                ..full
            }
            .room(),
            Some(GwqRelease::StaleView)
        );
        assert_eq!(GwqRelease::Predicted.as_str(), "predicted");
    }

    #[test]
    fn predicted_walk_sends_free_work_to_the_p_that_drains_first() {
        // Three P with equal ledgers; P1 is predicted to drain first. The
        // walk prices each P's wait as its backlog, which the step sets to
        // R x drain for P with a forecast: a cold request goes to P1.
        let lead = 16_292.0 / R;
        let mk = |drain: f64| GwqRow {
            backlog: (drain * R) as u128,
            drain_s: Some(drain),
            lead_s: lead,
            ..row(0, 30_000, 16_292)
        };
        let rows = vec![mk(2.0), mk(0.4), mk(1.0)];
        let terms = cold(20_000, 3);
        let w = walk(
            GlobalQueueMode::Hrrn,
            &params(18.4),
            &urls(3),
            &rows,
            &[held(&terms, 0.5)],
        );
        let pick = w.pick.expect("P1 has room");
        assert_eq!(pick.worker, 1);
        assert_eq!(pick.release, GwqRelease::Predicted);
        // Every drain above the lead: nothing is released.
        let rows = vec![mk(2.0), mk(1.3), mk(1.5)];
        let w = walk(
            GlobalQueueMode::Hrrn,
            &params(18.4),
            &urls(3),
            &rows,
            &[held(&terms, 0.5)],
        );
        assert!(w.pick.is_none());
    }
}
