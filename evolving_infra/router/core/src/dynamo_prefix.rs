//! Dynamo cache observations adapted to the existing cache-aware selector.
//! This adapter has no pull-hint injection or independent placement algorithm.
//!
//! Converts an external indexer reply into block-level prefix matches for the
//! cache-aware policy. Missing metadata or failed queries produce an empty signal;
//! CPU/GPU prefix overlap is counted once rather than added.
use crate::{
    kv_hint::{IndexerAnswer, KvHintInjector},
    policies::ExternalPrefixSignal,
};
use sgl_kv_indexer::{PrefixMatch, PrefixOutcome};

/// Query external prefix ownership and express hits in engine cache blocks.
///
/// # Inputs
/// `index` is the Dynamo query client; `tokens` is prepared input; `salt`
/// is the optional cache namespace; `model` identifies the model. `block_size`
/// is tokens per block; `bigram` subtracts one token when counting EAGLE blocks.
///
/// # Outputs
/// Returns an `ExternalPrefixSignal` with complete-query block count and
/// mapped worker matches. Missing/empty tokens, absent/zero block size, no
/// complete blocks or failed lookup yield an empty signal. Does not inject
/// a pull hint or choose a destination.
pub async fn query(
    index: &KvHintInjector,
    tokens: Option<&[u32]>,
    salt: Option<&str>,
    model: &str,
    block_size: Option<u32>,
    bigram: bool,
) -> ExternalPrefixSignal {
    let Some((tokens, block_size)) = tokens
        .zip(block_size)
        .filter(|(t, b)| !t.is_empty() && *b > 0)
    else {
        return ExternalPrefixSignal {
            outcome: PrefixOutcome::Empty,
            query_blocks: 0,
        };
    };
    let query_blocks = tokens.len().saturating_sub(usize::from(bigram)) / block_size as usize;
    if query_blocks == 0 {
        return ExternalPrefixSignal {
            outcome: PrefixOutcome::Empty,
            query_blocks,
        };
    }
    // EAGLE deployments point this client at the bundled hash-query adapter.
    // The adapter and event bridge preserve the engine's namespace/hash rules.
    let answer = index.prepare(tokens, salt, model).await;
    from_answer(answer, block_size, query_blocks)
}

/// Convert advisory token-prefix hits into bounded cache-block matches.
///
/// # Inputs
/// `answer` contains optional mapped indexer candidates; `block_size`
/// must be nonzero and is measured in tokens; `query_blocks` caps each hit.
///
/// # Outputs
/// Returns positive per-worker block matches and their maximum, or an
/// empty outcome when none remain. Each hit is floored to full blocks and
/// capped by the query length. Makes no network request.
fn from_answer(
    answer: Option<IndexerAnswer>,
    block_size: u32,
    query_blocks: usize,
) -> ExternalPrefixSignal {
    let matches: Vec<_> = answer
        .into_iter()
        .flat_map(|a| a.candidates)
        .filter_map(|candidate| {
            let blocks = (candidate.expected_hit_tokens / block_size).min(query_blocks as u32);
            (blocks > 0).then(|| PrefixMatch {
                worker_id: candidate.worker_url.clone(),
                address: candidate.worker_url,
                matched_prefix_blocks: blocks,
            })
        })
        .collect();
    let best_prefix_blocks = matches
        .iter()
        .map(|m| m.matched_prefix_blocks)
        .max()
        .unwrap_or(0);
    let outcome = if matches.is_empty() {
        PrefixOutcome::Empty
    } else {
        PrefixOutcome::Matched {
            matches,
            best_prefix_blocks,
        }
    };
    ExternalPrefixSignal {
        outcome,
        query_blocks,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_hint::{KvHintConfig, KvWorkerMapEntry};
    use serde_json::json;

    #[test]
    /// Verify prefix conversion counts the union as a maximum and skips unmapped workers.
    ///
    /// # Inputs
    /// No arguments. Use overlapping CPU/GPU hits plus an unmapped indexer instance.
    ///
    /// # Outputs
    /// Returns `()` when the stated invariant holds; failed assertions panic.
    fn cpu_gpu_overlap_is_max_not_sum_and_unmapped_workers_are_ignored() {
        let index = KvHintInjector::new(&KvHintConfig {
            indexer_url: "http://localhost:1".into(),
            min_tokens: 0,
            min_incremental_tokens: 0,
            local_prefix_fallback: false,
            indexer_timeout_ms: 20,
            worker_map: vec![KvWorkerMapEntry {
                id: 1,
                url: "http://p0".into(),
                servicer: String::new(),
            }],
        });
        let answer = index.answer_from_reply(
            &json!({"instances":{
            "1":{"cpu":192,"gpu":128}, "99":{"cpu":6400}}}),
            0.0,
        );
        let signal = from_answer(Some(answer), 64, 8);
        match signal.outcome {
            PrefixOutcome::Matched {
                matches,
                best_prefix_blocks,
            } => {
                assert_eq!(best_prefix_blocks, 3);
                assert_eq!(matches.len(), 1);
                assert_eq!(matches[0].address, "http://p0");
            }
            _ => panic!("missing expected prefix"),
        }
        assert!(matches!(
            from_answer(None, 64, 8).outcome,
            PrefixOutcome::Empty
        ));
    }
}
