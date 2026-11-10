// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Fixed-fleet policy matching the SMG HTTP PD fallback:
//! BLAKE3(agent key), first eight bytes little-endian, modulo P count.
//! SMG's worker-vector order is observed before measurement and supplied once
//! in EVOLVING_INFRA_EXPERIMENT_STICKY_WORKERS as a JSON array of URLs. This avoids
//! changing SMG or attributing different placement to router overhead.
//! Only the explicit experimental_agent_sticky policy reads this setting.
//! Missing configuration/key or changed membership fails closed.

use std::sync::Arc;

use crate::config::{ModelConfig, PolicyKind};
use crate::policies::{Policy, SelectionContext};
use crate::workers::Worker;
use axum::http::HeaderMap;

pub fn routing_key<'a>(model: &ModelConfig, headers: &'a HeaderMap) -> Option<&'a str> {
    let name = if model.policy == PolicyKind::ExperimentalAgentSticky {
        "x-smg-routing-key"
    } else {
        model.sticky.as_ref()?.header_name.as_str()
    };
    headers.get(name)?.to_str().ok().filter(|key| !key.is_empty())
}

pub fn sticky_slot(key: &str, count: usize) -> usize {
    let hash = blake3::hash(key.as_bytes());
    (u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap()) % count as u64) as usize
}

#[derive(Debug)]
pub struct ExperimentalAgentStickyPolicy {
    ordered_urls: Vec<String>,
}

impl ExperimentalAgentStickyPolicy {
    pub fn from_environment() -> Self {
        let raw = std::env::var("EVOLVING_INFRA_EXPERIMENT_STICKY_WORKERS")
            .expect("experimental_agent_sticky requires the observed SMG worker order");
        let urls: Vec<String> = serde_json::from_str(&raw)
            .expect("EVOLVING_INFRA_EXPERIMENT_STICKY_WORKERS must be a JSON array of URLs");
        assert!(!urls.is_empty(), "experimental worker order cannot be empty");
        assert!(urls.iter().enumerate().all(|(i, u)| !u.is_empty() && !urls[..i].contains(u)),
                "experimental worker order must contain distinct nonempty URLs");
        Self { ordered_urls: urls }
    }
}

impl Policy for ExperimentalAgentStickyPolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        let key = ctx.routing_key().filter(|key| !key.is_empty())?;
        if workers.len() != self.ordered_urls.len()
            || !self.ordered_urls.iter().all(|url| workers.iter().any(|w| &w.url == url))
        {
            return None;
        }
        let slot = sticky_slot(key, self.ordered_urls.len());
        workers.iter().find(|w| w.url == self.ordered_urls[slot]).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerSpec, WorkerMode};

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()), url: format!("http://{id}:30000"),
            mode: WorkerMode::Prefill, model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: Some(9000),
        }))
    }

    #[test]
    fn agent_sticky_is_independent_of_arrival_order_and_load() {
        let model = ModelId("tiny".into());
        let a = worker("a"); let b = worker("b");
        let first = ExperimentalAgentStickyPolicy { ordered_urls: vec![a.url.clone(), b.url.clone()] };
        let second = ExperimentalAgentStickyPolicy { ordered_urls: vec![a.url.clone(), b.url.clone()] };
        let mut expected = Vec::new();
        for i in 0..1000 {
            let key = format!("agent-{i}");
            let ctx = SelectionContext::with_routing_key(&model, None, Some(&key));
            expected.push(first.select(&[a.clone(), b.clone()], &ctx).unwrap().url.clone());
        }
        let _guards: Vec<_> = (0..100).map(|_| a.load_guard()).collect();
        for i in (0..1000).rev() {
            let key = format!("agent-{i}");
            let ctx = SelectionContext::with_routing_key(&model, None, Some(&key));
            assert_eq!(second.select(&[b.clone(), a.clone()], &ctx).unwrap().url, expected[i]);
        }
        assert!(expected.contains(&a.url) && expected.contains(&b.url));
    }

    #[test]
    fn agent_sticky_requires_key_and_candidate() {
        let model = ModelId("tiny".into());
        let policy = ExperimentalAgentStickyPolicy { ordered_urls: vec!["http://a:30000".into()] };
        assert!(policy.select(&[worker("a")], &SelectionContext::new(&model, None)).is_none());
        let ctx = SelectionContext::with_routing_key(&model, None, Some("problem"));
        assert!(policy.select(&[], &ctx).is_none());
        assert_eq!(policy.select(&[worker("a")], &ctx).unwrap().url, "http://a:30000");
        assert!(policy.select(&[worker("b")], &ctx).is_none());
    }
}
