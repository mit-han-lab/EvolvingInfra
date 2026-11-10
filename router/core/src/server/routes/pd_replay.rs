//! Traditional-PD per-request decision replay (`--pd-decision-replay`).
//!
//! A controlled comparison pins symmetric_pd routing to the per-request
//! (P, D) choices of a traditional-PD reference run. This module lets a
//! *traditional* PD deployment be pinned from the same table
//! (e.g. to run traditional decode engines with the symmetric engines' KV
//! pool split), so the two data planes are compared under identical
//! placement. The table format and outcome names are
//! [`crate::symmetric_pd::DecisionReplay`]'s.

use crate::discovery::ModelId;
use crate::symmetric_pd::DecisionReplay;
use crate::policies::registry::PdPoolResolver;
use crate::server::app_context::AppContext;
use crate::workers::Worker;
use axum::http::HeaderMap;
use std::sync::Arc;

pub(crate) struct PdReplayPick {
    pub prefill: Arc<Worker>,
    pub decode: Arc<Worker>,
}

/// Pin (prefill, decode) for this request from the replay table. `None`
/// (logged with the outcome) means the caller falls back to live selection:
/// no routing key, unknown key / length, exhausted session, or a pinned URL
/// that is not a healthy candidate of the right pool.
pub(crate) fn pick(
    ctx: &AppContext,
    replay: &DecisionReplay,
    headers: &HeaderMap,
    prompt_len: u64,
    prefill_candidates: &[Arc<Worker>],
    model_id: &ModelId,
    rid: Option<&str>,
) -> Option<PdReplayPick> {
    let header = ctx
        .config
        .model
        .pd_decision_replay
        .as_ref()
        .map(|c| c.header.as_str())
        .unwrap_or("X-SMG-Routing-Key");
    let routing_key = headers.get(header).and_then(|v| v.to_str().ok());
    let (pair, outcome) = replay.next(routing_key, prompt_len);
    let rid = rid.unwrap_or("");
    let Some((p_url, d_url)) = pair else {
        tracing::info!(
            target: "pd_replay",
            rid,
            replay = outcome,
            prompt_len,
            fallback = true,
            "pd_replay placement"
        );
        return None;
    };
    let prefill = prefill_candidates.iter().find(|w| w.url == p_url).cloned();
    let decode = PdPoolResolver::new(Arc::clone(&ctx.registry))
        .decode_candidates(model_id)
        .ok()
        .and_then(|ds| ds.into_iter().find(|w| w.url == d_url));
    match (prefill, decode) {
        (Some(prefill), Some(decode)) => {
            tracing::info!(
                target: "pd_replay",
                rid,
                replay = outcome,
                prompt_len,
                p_leg = %prefill.url,
                d_leg = %decode.url,
                fallback = false,
                "pd_replay placement"
            );
            Some(PdReplayPick { prefill, decode })
        }
        (p, d) => {
            tracing::warn!(
                target: "pd_replay",
                rid,
                replay = "ineligible",
                prompt_len,
                pinned_p = %p_url,
                pinned_d = %d_url,
                p_found = p.is_some(),
                d_found = d.is_some(),
                fallback = true,
                "pd_replay placement"
            );
            None
        }
    }
}
