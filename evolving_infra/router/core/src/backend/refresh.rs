// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! One outstanding load query per worker. Slow responses cannot launch duplicate
//! polls, and the backend rejects samples older than the latest applied query.
//!
//! Runs concurrent, bounded-time SGLang load polls and applies successful samples
//! to the shared Backend ledger. The task holds only a weak owner reference and
//! leaves previous observations intact when a worker cannot be queried.
use super::Backend;
use std::{
    sync::{Arc, Weak},
    time::{Duration, Instant},
};
use tokio::task::JoinSet;
use tracing::{debug, warn};

/// Start background load polling without extending the backend lifetime.
///
/// # Inputs
/// `backend` is the shared state owner; `poll_ms` requests the cadence;
/// `per_request` asks SGLang to include remaining prefill work per RID.
///
/// # Outputs
/// Returns `()` after spawning a detached task with a minimum 100 ms period.
/// If no Tokio runtime exists, logs a warning and starts no task.
pub fn spawn(backend: &Arc<Backend>, poll_ms: u64, per_request: bool, reconcile: bool) {
    if let Ok(runtime) = tokio::runtime::Handle::try_current() {
        runtime.spawn(run(
            Arc::downgrade(backend),
            Duration::from_millis(poll_ms.max(100)),
            per_request,
            reconcile,
        ));
    } else {
        warn!(target:"symmetric_pd","no Tokio runtime: backend load refresh not started");
    }
}
/// Refresh each registered worker while allowing at most one query per worker.
///
/// # Inputs
/// `weak` refers to the backend without owning it; `period` is the polling
/// cadence; `per_request` chooses aggregate or per-RID observations.
///
/// # Outputs
/// Returns `()` when the backend owner disappears. Successful queries
/// update its ledger; failures preserve old observations. Query tasks use
/// 500 ms connect/800 ms total timeouts; dropping this loop aborts its JoinSet.
/// Panics if the HTTP client cannot be constructed.
async fn run(weak: Weak<Backend>, period: Duration, per_request: bool, reconcile: bool) {
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_millis(500))
        .timeout(Duration::from_millis(800))
        .build()
        .expect("backend load client");
    let mut pending = std::collections::HashSet::new();
    let mut queries = JoinSet::new();
    let mut task_workers = std::collections::HashMap::new();
    let mut tick = tokio::time::interval(period);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            _=tick.tick()=>{
                let Some(backend)=weak.upgrade() else {return;};
                for worker in backend.registry.workers() {
                    if !pending.insert(worker.url.clone()) {continue;}
                    let backend=Arc::downgrade(&backend);let client=client.clone();
                    let worker_url=worker.url.clone();
                    let task=queries.spawn(async move {
                        let sent_at=Instant::now();
                        match super::sglang::progress::fetch_loads(&client,&worker.url,per_request,reconcile).await {
                            Ok(load)=>{if let Some(backend)=backend.upgrade() {backend.state.lock().apply_sample(&worker.url,load,sent_at);}},
                            Err(e)=>debug!(target:"symmetric_pd",worker=%worker.url,error=%e,"load poll failed; observation remains unknown or stale"),
                        }
                        worker.url.clone()
                    });
                    task_workers.insert(task.id(),worker_url);
                }
            },
            result=queries.join_next_with_id(),if !queries.is_empty()=>{
                match result {
                    Some(Ok((id,url)))=>{pending.remove(&url);task_workers.remove(&id);},
                    Some(Err(e))=>{if let Some(url)=task_workers.remove(&e.id()) {pending.remove(&url);}warn!(target:"symmetric_pd",error=%e,"backend load query task failed");},
                    None=>{},
                }
            }
        }
    }
}
