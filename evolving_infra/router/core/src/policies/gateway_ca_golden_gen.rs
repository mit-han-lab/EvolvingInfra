// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! Synthetic request sequences for the gateway parity test. The same code
//! (rustfmt line breaks aside) runs in two places: the golden-vector
//! generator of the real sgl-model-gateway (as
//! `src/policies/gateway_ca_golden_gen.rs`) and this port. Both generate the
//! same steps from the same seed (SplitMix64, integers only), so the golden
//! file stores only the gateway's results and a digest of each step's
//! inputs. Std only; no crate types.

#![allow(dead_code)]

/// SplitMix64 (Steele, Lea, Flood 2014).
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (modulo bias is irrelevant here).
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }

    /// True with probability `per_mille / 1000`.
    pub fn chance(&mut self, per_mille: u64) -> bool {
        self.below(1000) < per_mille
    }
}

/// One scenario: the policy parameters and the traffic shape.
pub struct ScenarioSpec {
    pub name: &'static str,
    pub seed: u64,
    pub workers: usize,
    pub steps: usize,
    pub cache_threshold: f32,
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    pub max_tree_size: usize,
    /// An eviction step after every this many requests (0 = none).
    pub evict_every: usize,
    pub sessions: usize,
    pub max_turns: u64,
    /// Per-mille rates.
    pub salted: u64,
    pub text: u64,
    pub subset: u64,
    pub unhealthy: u64,
    pub imbalance: u64,
    pub branch: u64,
}

pub fn scenarios() -> Vec<ScenarioSpec> {
    vec![
        ScenarioSpec {
            name: "g1_params_no_eviction",
            seed: 0x6731_0001,
            workers: 3,
            steps: 1500,
            cache_threshold: 0.3,
            balance_abs_threshold: 64,
            balance_rel_threshold: 1.5,
            max_tree_size: 66_163_200,
            evict_every: 100,
            sessions: 12,
            max_turns: 15,
            salted: 700,
            text: 40,
            subset: 30,
            unhealthy: 30,
            imbalance: 150,
            branch: 100,
        },
        ScenarioSpec {
            name: "g1_params_eviction",
            seed: 0x6731_0002,
            workers: 3,
            steps: 1500,
            cache_threshold: 0.3,
            balance_abs_threshold: 64,
            balance_rel_threshold: 1.5,
            max_tree_size: 6_000,
            evict_every: 20,
            sessions: 12,
            max_turns: 15,
            salted: 700,
            text: 40,
            subset: 30,
            unhealthy: 30,
            imbalance: 150,
            branch: 100,
        },
        ScenarioSpec {
            name: "tight_thresholds",
            seed: 0x6731_0003,
            workers: 4,
            steps: 1000,
            cache_threshold: 0.5,
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            max_tree_size: 3_000,
            evict_every: 15,
            sessions: 10,
            max_turns: 10,
            salted: 300,
            text: 100,
            subset: 60,
            unhealthy: 60,
            imbalance: 300,
            branch: 200,
        },
        ScenarioSpec {
            name: "six_workers_zero_threshold",
            seed: 0x6731_0004,
            workers: 6,
            steps: 800,
            cache_threshold: 0.0,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            max_tree_size: 10_000,
            evict_every: 40,
            sessions: 16,
            max_turns: 12,
            salted: 500,
            text: 60,
            subset: 50,
            unhealthy: 40,
            imbalance: 100,
            branch: 150,
        },
    ]
}

/// Worker URL `i`.
pub fn worker_url(i: usize) -> String {
    format!("http://w{i}:18080")
}

/// One generated step.
pub enum GenStep {
    Evict,
    Request {
        /// `Some(empty)` is an empty token prompt (routes as no text).
        tokens: Option<Vec<u32>>,
        text: Option<String>,
        salt: Option<String>,
        /// Indices of the workers passed to the selection, ascending.
        candidates: Vec<usize>,
        /// Load and health per candidate.
        loads: Vec<usize>,
        healthy: Vec<bool>,
    },
}

struct Session {
    id: u64,
    context: Vec<u32>,
    salt: Option<String>,
    turns: u64,
}

fn token(rng: &mut SplitMix64) -> u32 {
    if rng.chance(50) {
        rng.next() as u32
    } else {
        rng.below(4096) as u32
    }
}

fn new_session(
    rng: &mut SplitMix64,
    spec: &ScenarioSpec,
    next_id: &mut u64,
    prefixes: &[Vec<u32>],
) -> Session {
    let id = *next_id;
    *next_id += 1;
    let salt = rng.chance(spec.salted).then(|| format!("sess-{id}"));
    let mut context = Vec::new();
    if salt.is_none() || rng.chance(500) {
        let p = rng.below(prefixes.len() as u64) as usize;
        context.extend_from_slice(&prefixes[p]);
    }
    let n = 8 + rng.below(40);
    for _ in 0..n {
        context.push(token(rng));
    }
    Session {
        id,
        context,
        salt,
        turns: 0,
    }
}

/// The steps of `spec`.
pub fn generate(spec: &ScenarioSpec) -> Vec<GenStep> {
    let mut rng = SplitMix64(spec.seed);
    let prefixes: Vec<Vec<u32>> = (0..3)
        .map(|_| (0..40).map(|_| token(&mut rng)).collect())
        .collect();
    let mut next_id = 0u64;
    let mut sessions: Vec<Session> = (0..spec.sessions)
        .map(|_| new_session(&mut rng, spec, &mut next_id, &prefixes))
        .collect();
    let mut steps = Vec::with_capacity(spec.steps + spec.steps / spec.evict_every.max(1));
    for i in 0..spec.steps {
        if spec.evict_every > 0 && i > 0 && i % spec.evict_every == 0 {
            steps.push(GenStep::Evict);
        }
        let s = rng.below(sessions.len() as u64) as usize;
        if sessions[s].turns >= spec.max_turns {
            sessions[s] = new_session(&mut rng, spec, &mut next_id, &prefixes);
        }
        let session = &mut sessions[s];
        session.turns += 1;
        let (tokens, text) = if rng.chance(spec.text) {
            let words = 1 + rng.below(6);
            let mut text = format!("session {} turn {}:", session.id, session.turns / 2);
            for _ in 0..words {
                text.push_str(&format!(" w{}", rng.below(50)));
            }
            (None, Some(text))
        } else if rng.chance(5) {
            (Some(Vec::new()), None)
        } else if rng.chance(spec.branch) && session.context.len() > 4 {
            // A side request: a prefix of the context plus new tokens.
            let cut = 1 + rng.below(session.context.len() as u64 - 1) as usize;
            let mut prompt = session.context[..cut].to_vec();
            for _ in 0..(1 + rng.below(30)) {
                prompt.push(token(&mut rng));
            }
            (Some(prompt), None)
        } else {
            for _ in 0..(4 + rng.below(60)) {
                session.context.push(token(&mut rng));
            }
            (Some(session.context.clone()), None)
        };
        let salt = session.salt.clone();
        let mut candidates: Vec<usize> = (0..spec.workers).collect();
        if spec.workers > 1 && rng.chance(spec.subset) {
            let drop = rng.below(spec.workers as u64) as usize;
            candidates.remove(drop);
        }
        let mut loads: Vec<usize> = candidates.iter().map(|_| rng.below(12) as usize).collect();
        if rng.chance(spec.imbalance) {
            let hot = rng.below(loads.len() as u64) as usize;
            loads[hot] = 60 + rng.below(100) as usize;
        } else if rng.chance(50) {
            let level = rng.below(8) as usize;
            loads.iter_mut().for_each(|l| *l = level);
        }
        let mut healthy = vec![true; candidates.len()];
        if rng.chance(spec.unhealthy) {
            let sick = rng.below(healthy.len() as u64) as usize;
            healthy[sick] = false;
        }
        steps.push(GenStep::Request {
            tokens,
            text,
            salt,
            candidates,
            loads,
            healthy,
        });
    }
    steps
}

/// FNV-1a 64.
pub fn fnv(bytes: &[u8], mut h: u64) -> u64 {
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// Digest of a step's inputs (to prove both sides generated the same step).
pub fn step_digest(step: &GenStep) -> String {
    let mut h = FNV_OFFSET;
    match step {
        GenStep::Evict => h = fnv(b"evict", h),
        GenStep::Request {
            tokens,
            text,
            salt,
            candidates,
            loads,
            healthy,
        } => {
            h = fnv(b"req", h);
            match tokens {
                Some(t) => {
                    h = fnv(b"T", h);
                    for x in t {
                        h = fnv(&x.to_le_bytes(), h);
                    }
                }
                None => h = fnv(b"N", h),
            }
            h = fnv(text.as_deref().unwrap_or("<none>").as_bytes(), h);
            h = fnv(salt.as_deref().unwrap_or("<none>").as_bytes(), h);
            for (i, c) in candidates.iter().enumerate() {
                h = fnv(&(*c as u64).to_le_bytes(), h);
                h = fnv(&(loads[i] as u64).to_le_bytes(), h);
                h = fnv(&[u8::from(healthy[i])], h);
            }
        }
    }
    format!("{h:016x}")
}

/// Digest of the per-worker tree characters (sorted `(url, chars)` of the
/// token tree, then of the text tree).
pub fn chars_digest(token_tree: &[(String, usize)], text_tree: &[(String, usize)]) -> String {
    let mut h = FNV_OFFSET;
    for (url, n) in token_tree {
        h = fnv(format!("{url}={n};").as_bytes(), h);
    }
    h = fnv(b"|", h);
    for (url, n) in text_tree {
        h = fnv(format!("{url}={n};").as_bytes(), h);
    }
    format!("{h:016x}")
}
