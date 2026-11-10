// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! The per-request `routing decision` line (INFO, target `routing_decision`)
//! both request routes emit before upstream dispatch. Offline analysis
//! joins it to client / engine records by `rid` and reads
//! `prefill_worker` / `decode_worker`; those names and the message text are
//! the contract. symmetric_pd placements append `placement=symmetric_pd`,
//! `transfer_id`, `local` and every score term; token-charged decode
//! policies (`--pd-decode-policy min-inflight-tokens | po2-tokens`) append
//! `decode_policy`, `decode_charge_tokens`, `decode_charged_tokens` (the
//! chosen decode's total after this charge), `decode_normalized` and
//! `decode_loads` (JSON `{"<decode url>": [charged_tokens, charged_reqs,
//! capacity_or_0]}` over the candidates); other legacy dispatches keep the
//! original field set unchanged. Late D choose (`--pd-late-decode choose`)
//! dispatches have no decode worker yet: `decode_worker` is empty and the
//! line appends `late_d` (the bind rule), `late_d_k`, `late_d_default` (the
//! arrival pick) and `bootstrap_room`; the target `late_d` line `late_d
//! bound` names the decode that took the request (`decode_worker`). On
//! symmetric_pd (symmetric workers) the line keeps every symmetric_pd field
//! (`transfer_id` is the late-D key) and appends the three `late_d*` fields
//! instead of `bootstrap_room`.

use crate::config::PolicyKind;
use crate::symmetric_pd::placement::{
    d_select_name, p_reason_name, TERM_BACKLOG, TERM_DECODE_LOAD, TERM_D_SELECT, TERM_HIT_TOKENS,
    TERM_P_REASON, TERM_REUSE_GAIN, TERM_ROLE_PEN_D, TERM_ROLE_PEN_P, TERM_SCORE, TERM_XFER_COST,
};
use crate::symmetric_pd::Decision;
use crate::policies::decode_tokens::DecodeTokenLog;
use serde_json::Value;

pub(crate) struct RoutingDecision<'a> {
    pub rid: &'a str,
    pub request_id: &'a str,
    pub policy: PolicyKind,
    pub prefill_worker: &'a str,
    /// Empty for a plain (single-worker) dispatch; for a symmetric_pd Local
    /// placement it equals `prefill_worker` (the same engine decodes).
    pub decode_worker: &'a str,
    /// Admission-queue wait (co_schedule / symmetric_pd hold queue); 0 when the
    /// request was placed inline. Excludes ingress tokenization and the
    /// indexer lookup, which `ingress_to_dispatch_ms` includes.
    pub queue_wait_ms: u64,
    /// `policy`, a co_schedule branch, or the symmetric_pd `local` / `split`.
    pub branch: &'a str,
    pub ingress_to_dispatch_ms: u64,
    pub owner_hint: Option<&'a Value>,
    pub flex: Option<&'a Decision>,
    /// Token-charged decode pick (or charged replay pin); `None` otherwise.
    pub decode_tokens: Option<&'a DecodeTokenLog>,
    /// Late D choose dispatch (the decode is chosen at the bind); `None`
    /// otherwise.
    pub late_d: Option<&'a LateDispatchLog<'a>>,
}

/// The late-D part of a choose-mode `routing decision` line.
pub(crate) struct LateDispatchLog<'a> {
    /// The bind rule (`d1`, `po2`, `keep`).
    pub rule: &'static str,
    /// Decode candidates (one leg each).
    pub k: usize,
    /// The arrival pick: the candidate the prefill admits when no decision
    /// comes in time.
    pub default_worker: &'a str,
    pub bootstrap_room: u64,
}

pub(crate) fn log_routing_decision(r: &RoutingDecision<'_>) {
    let null = Value::Null;
    let owner_hint = r.owner_hint.unwrap_or(&null);
    if let (Some(l), Some(d)) = (r.late_d, r.flex) {
        // Late D choose on symmetric_pd: every symmetric_pd placement field (the
        // placement's D is the arrival pick), no decode worker yet, then the
        // late-D fields; `transfer_id` is the late-D key.
        tracing::info!(
            target: "routing_decision",
            rid = r.rid,
            request_id = r.request_id,
            policy = ?r.policy,
            prefill_worker = %r.prefill_worker,
            decode_worker = %r.decode_worker,
            queue_wait_ms = r.queue_wait_ms,
            branch = r.branch,
            ingress_to_dispatch_ms = r.ingress_to_dispatch_ms,
            owner_hint = %owner_hint,
            placement = "symmetric_pd",
            transfer_id = l.bootstrap_room as i64,
            local = d.is_local(),
            forced = d.forced,
            prompt_len = d.prompt_len,
            hit_tokens = d.term(TERM_HIT_TOKENS),
            reuse_gain = d.term(TERM_REUSE_GAIN),
            backlog = d.term(TERM_BACKLOG),
            role_pen_p = d.term(TERM_ROLE_PEN_P),
            role_pen_d = d.term(TERM_ROLE_PEN_D),
            xfer_cost = d.term(TERM_XFER_COST),
            decode_load = d.term(TERM_DECODE_LOAD),
            p_reason = p_reason_name(d.term(TERM_P_REASON)),
            d_select = d_select_name(d.term(TERM_D_SELECT)),
            score = d.term(TERM_SCORE),
            indexer_ms = d.indexer_ms,
            late_d = l.rule,
            late_d_k = l.k,
            late_d_default = %l.default_worker,
            "routing decision",
        );
        return;
    }
    if let Some(l) = r.late_d {
        tracing::info!(
            target: "routing_decision",
            rid = r.rid,
            request_id = r.request_id,
            policy = ?r.policy,
            prefill_worker = %r.prefill_worker,
            decode_worker = %r.decode_worker,
            queue_wait_ms = r.queue_wait_ms,
            branch = r.branch,
            ingress_to_dispatch_ms = r.ingress_to_dispatch_ms,
            owner_hint = %owner_hint,
            late_d = l.rule,
            late_d_k = l.k,
            late_d_default = %l.default_worker,
            bootstrap_room = l.bootstrap_room,
            "routing decision",
        );
        return;
    }
    match (r.flex, r.decode_tokens) {
        (None, None) => tracing::info!(
            target: "routing_decision",
            rid = r.rid,
            request_id = r.request_id,
            policy = ?r.policy,
            prefill_worker = %r.prefill_worker,
            decode_worker = %r.decode_worker,
            queue_wait_ms = r.queue_wait_ms,
            branch = r.branch,
            ingress_to_dispatch_ms = r.ingress_to_dispatch_ms,
            owner_hint = %owner_hint,
            "routing decision",
        ),
        (None, Some(t)) => tracing::info!(
            target: "routing_decision",
            rid = r.rid,
            request_id = r.request_id,
            policy = ?r.policy,
            prefill_worker = %r.prefill_worker,
            decode_worker = %r.decode_worker,
            queue_wait_ms = r.queue_wait_ms,
            branch = r.branch,
            ingress_to_dispatch_ms = r.ingress_to_dispatch_ms,
            owner_hint = %owner_hint,
            decode_policy = t.rule,
            decode_charge_tokens = t.charge_tokens,
            decode_charged_tokens = t.chosen_charged_tokens,
            decode_normalized = t.normalized,
            decode_loads = %t.loads,
            "routing decision",
        ),
        (Some(d), _) => tracing::info!(
            target: "routing_decision",
            rid = r.rid,
            request_id = r.request_id,
            policy = ?r.policy,
            prefill_worker = %r.prefill_worker,
            decode_worker = %r.decode_worker,
            queue_wait_ms = r.queue_wait_ms,
            branch = r.branch,
            ingress_to_dispatch_ms = r.ingress_to_dispatch_ms,
            owner_hint = %owner_hint,
            placement = "symmetric_pd",
            transfer_id = d.placement.transfer_id.map(|t| t as i64).unwrap_or(-1),
            local = d.is_local(),
            forced = d.forced,
            prompt_len = d.prompt_len,
            hit_tokens = d.term(TERM_HIT_TOKENS),
            reuse_gain = d.term(TERM_REUSE_GAIN),
            backlog = d.term(TERM_BACKLOG),
            role_pen_p = d.term(TERM_ROLE_PEN_P),
            role_pen_d = d.term(TERM_ROLE_PEN_D),
            xfer_cost = d.term(TERM_XFER_COST),
            decode_load = d.term(TERM_DECODE_LOAD),
            p_reason = p_reason_name(d.term(TERM_P_REASON)),
            d_select = d_select_name(d.term(TERM_D_SELECT)),
            score = d.term(TERM_SCORE),
            indexer_ms = d.indexer_ms,
            "routing decision",
        ),
    }
}

/// Backlog ledger handle of whichever admission path placed the request.
/// `retire()` marks the prefill work done (idempotent on both).
#[derive(Clone)]
pub(crate) enum AdmissionLedger {
    Co(crate::co_schedule::LedgerHandle),
    Flex(crate::symmetric_pd::LedgerHandle),
}

impl AdmissionLedger {
    pub(crate) fn retire(&self) {
        match self {
            AdmissionLedger::Co(l) => l.retire(),
            AdmissionLedger::Flex(l) => l.retire(),
        }
    }
}
