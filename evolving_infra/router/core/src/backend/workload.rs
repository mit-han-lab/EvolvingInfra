//! Cumulative usage of prepared requests whose P and D legs both completed, as
//! reported to the external role controller (`GET /symmetric_pd/roles`). Only the
//! running sums are kept, never per-request state.
use parking_lot::Mutex;
use serde::Serialize;

/// One prepared request whose P and D legs both completed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RequestSample {
    pub input_tokens: u64,
    /// Prefix the P engine reused (its reported `cached_tokens`).
    pub cached_tokens: u64,
    pub output_tokens: u64,
    /// D engine's speculative verify steps; `None` without speculation.
    pub verify_steps: Option<u64>,
    pub vision_matmul_flops: u64,
    pub image_attention_pairs: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct WorkloadSums {
    pub requests: u64,
    pub input_tokens: u64,
    pub uncached_tokens: u64,
    /// Exact sum of causal key positions, including attention within each new chunk.
    pub causal_attention_pairs: f64,
    pub output_tokens: u64,
    /// Output and verify steps of the requests that reported verify steps.
    pub verified_output_tokens: u64,
    pub verify_steps: u64,
    pub vision_matmul_flops: f64,
    pub image_attention_pairs: f64,
}

/// Totals since the router started.
#[derive(Default)]
pub struct CompletedWorkload {
    sums: Mutex<WorkloadSums>,
}
impl CompletedWorkload {
    pub fn add(&self, s: RequestSample) {
        let cached = s.cached_tokens.min(s.input_tokens);
        let uncached = s.input_tokens - cached;
        let mut w = self.sums.lock();
        w.requests += 1;
        w.input_tokens += s.input_tokens;
        w.uncached_tokens += uncached;
        let u = uncached as f64;
        w.causal_attention_pairs += u * cached as f64 + u * (u + 1.0) / 2.0;
        w.image_attention_pairs += s.image_attention_pairs as f64;
        w.vision_matmul_flops += s.vision_matmul_flops as f64;
        w.output_tokens += s.output_tokens;
        if let Some(v) = s.verify_steps.filter(|v| *v > 0) {
            w.verified_output_tokens += s.output_tokens;
            w.verify_steps += v;
        }
    }
    pub fn totals(&self) -> WorkloadSums {
        *self.sums.lock()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_count_causal_work_and_only_reported_verify_steps() {
        let w = CompletedWorkload::default();
        w.add(RequestSample {
            input_tokens: 12,
            cached_tokens: 7,
            output_tokens: 9,
            verify_steps: Some(3),
            vision_matmul_flops: 136,
            image_attention_pairs: 6,
        });
        w.add(RequestSample {
            input_tokens: 12,
            cached_tokens: 99,
            output_tokens: 6,
            verify_steps: None,
            vision_matmul_flops: 0,
            image_attention_pairs: 0,
        });
        let sums = w.totals();
        assert_eq!(sums.requests, 2);
        assert_eq!(sums.uncached_tokens, 5);
        assert_eq!(sums.causal_attention_pairs, (5 * 7 + 5 * 6 / 2) as f64);
        assert_eq!(sums.image_attention_pairs, 6.0);
        assert_eq!(sums.vision_matmul_flops, 136.0);
        assert_eq!(sums.output_tokens, 15);
        assert_eq!((sums.verified_output_tokens, sums.verify_steps), (9, 3));
        assert_eq!(w.totals(), sums, "reading does not reset the totals");
    }
}
