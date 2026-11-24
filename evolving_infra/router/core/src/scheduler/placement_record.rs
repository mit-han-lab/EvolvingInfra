// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! The router's own record of where it sent each prompt, read when the KV
//! directory lookup fails.
//!
//! Per prompt page, the record keeps the `--kv-worker-map` instances the
//! router placed it on. Pages are chained hashes of [`PAGE_TOKENS`]-token
//! pages from the prompt's start, so a held run of pages is a held prefix.
//! When a lookup fails (an error, or its deadline), placement reads the
//! record as it would the directory's answer: each instance holds its
//! longest recorded prefix of the prompt, in whole pages. So the next turn of
//! a long prompt goes to the worker that holds the previous turn even when
//! the directory does not answer in time.
//!
//! The record holds what the router sent, not what a worker still has: the
//! engine checks the pages at its own match, as for a directory hit. It keeps
//! at most [`MAX_PAGES`] pages; the pages placed least recently leave first.
//! An instance index from [`MAX_INSTANCES`] on is not recorded.

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

use parking_lot::Mutex;
use serde_json::{json, Map, Value};

/// Tokens per recorded page. A recorded prefix ends on a multiple of it, at
/// most the prompt's shared prefix.
pub const PAGE_TOKENS: usize = 256;
/// Pages the record keeps: 4,096 prompts of 64k tokens.
pub const MAX_PAGES: usize = 1 << 20;
/// Instances one page can name (the bits of its mask).
pub const MAX_INSTANCES: usize = 128;

/// The chained hash of each whole [`PAGE_TOKENS`]-token page of `tokens`.
pub fn page_hashes(tokens: &[u32]) -> Vec<u64> {
    let mut chain = 0u64;
    tokens
        .chunks_exact(PAGE_TOKENS)
        .map(|page| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            chain.hash(&mut hasher);
            page.hash(&mut hasher);
            chain = hasher.finish();
            chain
        })
        .collect()
}

#[derive(Debug, Default)]
struct Pages {
    /// Page hash -> (instance mask, stamp of its last placement).
    masks: HashMap<u64, (u128, u64)>,
    /// (page hash, stamp) in placement order; an entry whose stamp is older
    /// than the page's is skipped when the record drops pages.
    order: VecDeque<(u64, u64)>,
    stamp: u64,
}

/// See the module docs. `instances`: the instance ids of `--kv-worker-map`,
/// whose positions are the mask bits, with their worker URLs.
#[derive(Debug)]
pub struct PlacementRecord {
    instances: Vec<(u64, String)>,
    pages: Mutex<Pages>,
    max_pages: usize,
}

impl PlacementRecord {
    pub fn new(instances: Vec<(u64, String)>) -> Self {
        Self::with_capacity(instances, MAX_PAGES)
    }

    fn with_capacity(instances: Vec<(u64, String)>, max_pages: usize) -> Self {
        if instances.len() > MAX_INSTANCES {
            tracing::warn!(target: "symmetric_pd", instances = instances.len(),
                "placement record: only the first {MAX_INSTANCES} KV directory instances are recorded");
        }
        Self {
            instances,
            pages: Mutex::new(Pages::default()),
            max_pages,
        }
    }

    /// The mask bit of the first instance of worker `url` (a trailing `/` is
    /// ignored), None when the map names none or it is past [`MAX_INSTANCES`].
    pub fn worker_instance(&self, url: &str) -> Option<usize> {
        let url = url.trim_end_matches('/');
        self.instances
            .iter()
            .position(|(_, u)| u.trim_end_matches('/') == url)
            .filter(|&i| i < MAX_INSTANCES)
    }

    /// Record that the pages `hashes` ([`page_hashes`] of a prompt) went to
    /// instance `instance` (a mask bit).
    pub fn record(&self, hashes: &[u64], instance: usize) {
        if instance >= MAX_INSTANCES || hashes.is_empty() {
            return;
        }
        let bit = 1u128 << instance;
        let mut pages = self.pages.lock();
        pages.stamp += 1;
        let stamp = pages.stamp;
        for &h in hashes {
            let entry = pages.masks.entry(h).or_insert((0, 0));
            entry.0 |= bit;
            entry.1 = stamp;
            pages.order.push_back((h, stamp));
        }
        while pages.masks.len() > self.max_pages {
            let Some((h, at)) = pages.order.pop_front() else {
                break;
            };
            if pages.masks.get(&h).is_some_and(|(_, last)| *last == at) {
                pages.masks.remove(&h);
            }
        }
        // Stale order entries of re-placed pages: compact when they dominate.
        if pages.order.len() > 2 * self.max_pages.max(pages.masks.len()) {
            let Pages { masks, order, .. } = &mut *pages;
            order.retain(|(h, at)| masks.get(h).is_some_and(|(_, last)| last == at));
        }
    }

    /// Each instance's longest recorded prefix of the prompt `hashes`, in
    /// tokens, as an indexer reply (`{"instances": {"<id>": {"gpu": t}}}`),
    /// or None when no instance holds a page of it.
    pub fn reply(&self, hashes: &[u64]) -> Option<Value> {
        let pages = self.pages.lock();
        let mut alive = u128::MAX;
        let mut held = vec![0u64; self.instances.len().min(MAX_INSTANCES)];
        for (i, h) in hashes.iter().enumerate() {
            alive &= pages.masks.get(h).map_or(0, |(mask, _)| *mask);
            if alive == 0 {
                break;
            }
            for (bit, tokens) in held.iter_mut().enumerate() {
                if (alive >> bit) & 1 == 1 {
                    *tokens = ((i + 1) * PAGE_TOKENS) as u64;
                }
            }
        }
        drop(pages);
        let instances: Map<String, Value> = held
            .iter()
            .enumerate()
            .filter(|(_, tokens)| **tokens > 0)
            .map(|(bit, tokens)| (self.instances[bit].0.to_string(), json!({ "gpu": tokens })))
            .collect();
        (!instances.is_empty()).then(|| json!({ "instances": instances }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(max_pages: usize) -> PlacementRecord {
        PlacementRecord::with_capacity(
            vec![
                (20001, "http://a:30000".into()),
                (20002, "http://b:30000".into()),
                (21001, "http://a:30000".into()),
            ],
            max_pages,
        )
    }

    /// Each instance holds its longest recorded prefix, in whole pages; a
    /// prompt that diverges inside a page keeps the pages before it.
    #[test]
    fn the_record_answers_each_instance_s_longest_prefix() {
        let rec = record(MAX_PAGES);
        assert_eq!(rec.worker_instance("http://a:30000/"), Some(0));
        assert_eq!(rec.worker_instance("http://c:30000"), None);
        let turn1: Vec<u32> = (0..10 * PAGE_TOKENS as u32 + 7).collect();
        rec.record(&page_hashes(&turn1), 0);
        let mut turn2 = turn1[..4 * PAGE_TOKENS + 3].to_vec();
        turn2.extend(900_000..900_000 + 3 * PAGE_TOKENS as u32);
        rec.record(&page_hashes(&turn2), 1);
        let mut next = turn1.clone();
        next.extend(1..=PAGE_TOKENS as u32);
        let reply = rec.reply(&page_hashes(&next)).unwrap();
        assert_eq!(reply["instances"]["20001"]["gpu"], 10 * PAGE_TOKENS);
        assert_eq!(reply["instances"]["20002"]["gpu"], 4 * PAGE_TOKENS);
        assert!(reply["instances"].get("21001").is_none());
        assert!(rec
            .reply(&page_hashes(&(5_000_000..5_001_000).collect::<Vec<u32>>()))
            .is_none());
    }

    /// The record keeps at most its page limit; the least recently placed
    /// pages leave first.
    #[test]
    fn the_record_drops_the_least_recently_placed_pages() {
        let rec = record(16);
        let old: Vec<u32> = (0..8 * PAGE_TOKENS as u32).collect();
        let new: Vec<u32> = (1_000_000..1_000_000 + 12 * PAGE_TOKENS as u32).collect();
        rec.record(&page_hashes(&old), 0);
        rec.record(&page_hashes(&new), 1);
        assert!(
            rec.reply(&page_hashes(&old)).is_none(),
            "the oldest pages left"
        );
        let reply = rec.reply(&page_hashes(&new)).unwrap();
        assert_eq!(reply["instances"]["20002"]["gpu"], 12 * PAGE_TOKENS);
        // Placing the old prompt again keeps it, and drops the new prompt's first pages.
        rec.record(&page_hashes(&old), 0);
        assert!(rec.reply(&page_hashes(&old)).is_some());
        assert!(rec.reply(&page_hashes(&new)).is_none());
        assert!(rec.pages.lock().masks.len() <= 16);
    }
}
