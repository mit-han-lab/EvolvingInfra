// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-worker single-flight progress queries, shared by periodic and demand triggers.
use parking_lot::Mutex;
use serde_json::Value;
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::Notify;

#[derive(Default)]
struct WorkerProgress {
    in_flight: bool,
    accepted_sent_at: Option<Instant>,
    retry_after: Option<Instant>,
    cursor: Option<(String, u64)>,
}

#[derive(Default)]
pub struct LoadRefresh {
    workers: Mutex<HashMap<String, WorkerProgress>>,
    pub changed: Notify,
}

impl LoadRefresh {
    pub fn initialized(&self, worker: &str) -> bool {
        self.workers
            .lock()
            .get(worker)
            .is_some_and(|p| p.accepted_sent_at.is_some())
    }

    pub fn claim(
        self: &Arc<Self>,
        worker: &str,
        now: Instant,
        age: Duration,
        force: bool,
    ) -> Option<Query> {
        let mut workers = self.workers.lock();
        let state = workers.entry(worker.into()).or_default();
        if state.in_flight
            || state.retry_after.is_some_and(|t| now < t)
            || (!force
                && state
                    .accepted_sent_at
                    .is_some_and(|t| now.saturating_duration_since(t) <= age))
        {
            return None;
        }
        state.in_flight = true;
        Some(Query {
            refresh: Arc::clone(self),
            worker: worker.into(),
            sent_at: now,
            retry_delay: age,
            finished: false,
        })
    }
}

/// A progress reply with the accepted epoch and sequence: the worker has not
/// published since (for example during a long prefill forward). Not applied,
/// like a stale reply, but expected; callers log it at debug level.
#[derive(Debug)]
pub struct UnchangedProgress;

impl std::fmt::Display for UnchangedProgress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("progress unchanged since the accepted reply")
    }
}

impl std::error::Error for UnchangedProgress {}

/// Dropping a cancelled/panicked task releases the query slot and applies backoff.
pub struct Query {
    refresh: Arc<LoadRefresh>,
    worker: String,
    pub sent_at: Instant,
    retry_delay: Duration,
    finished: bool,
}

impl Query {
    pub fn accepts(&self, epoch: &str, sequence: u64) -> bool {
        if epoch.is_empty() {
            return false;
        }
        let workers = self.refresh.workers.lock();
        let state = &workers[&self.worker];
        state
            .cursor
            .as_ref()
            .is_none_or(|(old_epoch, old_sequence)| epoch != old_epoch || sequence > *old_sequence)
    }

    /// `accepts`, as an error that tells an unchanged reply (the worker
    /// published nothing since the accepted one: same epoch and sequence,
    /// [`UnchangedProgress`]) from a stale one (an older sequence).
    pub fn check(&self, epoch: &str, sequence: u64) -> anyhow::Result<()> {
        if self.accepts(epoch, sequence) {
            return Ok(());
        }
        let workers = self.refresh.workers.lock();
        let unchanged =
            workers[&self.worker]
                .cursor
                .as_ref()
                .is_some_and(|(old_epoch, old_sequence)| {
                    epoch == old_epoch && sequence == *old_sequence
                });
        if unchanged {
            Err(UnchangedProgress.into())
        } else {
            Err(anyhow::anyhow!("stale progress sequence"))
        }
    }

    pub fn finish(mut self, cursor: Option<(String, u64)>, now: Instant) {
        self.finish_inner(cursor, now);
    }

    fn finish_inner(&mut self, cursor: Option<(String, u64)>, now: Instant) {
        let mut workers = self.refresh.workers.lock();
        let state = workers.get_mut(&self.worker).expect("claimed worker");
        if let Some(cursor) = cursor {
            state.cursor = Some(cursor);
            // Query start bounds data age; a late response is not fresh data.
            state.accepted_sent_at = Some(self.sent_at);
            state.retry_after = None;
        } else {
            state.retry_after = Some(now + self.retry_delay);
        }
        state.in_flight = false;
        self.finished = true;
        drop(workers);
        self.refresh.changed.notify_waiters();
    }
}

impl Drop for Query {
    fn drop(&mut self) {
        if !self.finished {
            self.finish_inner(None, Instant::now());
        }
    }
}

/// Log the outcome of one progress query of `worker` (target
/// `load_refresh`): unchanged progress and an applied reply at debug, any
/// other error at warn.
pub(crate) fn log_progress_result<T>(worker: &str, result: &anyhow::Result<T>, elapsed_ms: f64) {
    if let Err(error) = result {
        if error.is::<UnchangedProgress>() {
            tracing::debug!(target: "load_refresh", worker = %worker, elapsed_ms,
                "progress unchanged; retaining reservations");
        } else {
            tracing::warn!(target: "load_refresh", worker = %worker, %error, elapsed_ms,
                "progress query failed; retaining reservations");
        }
    } else {
        tracing::debug!(target: "load_refresh", worker = %worker, elapsed_ms,
            "progress query applied");
    }
}

/// A `/v1/loads` poll that the engine answered with HTTP 4xx or 500. An
/// engine without the `prefill_routing` section (an older
/// engine) answers `include=...,prefill_routing` this way: its load-snapshot
/// section check raises ValueError (400 or 500).
pub(crate) fn include_rejected(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<reqwest::Error>()
        .and_then(reqwest::Error::status)
        .is_some_and(|status| {
            status.is_client_error() || status == reqwest::StatusCode::INTERNAL_SERVER_ERROR
        })
}

/// One DP rank's row of a `/v1/loads` progress poll ([`fetch_progress_poll`]):
/// its `dp_rank` (None when the row has none), the prefill-work progress, the
/// rest of the load row (the trace's `engine_load`) and the raw
/// `prefill_routing` snapshot.
#[derive(Debug)]
pub(crate) struct ProgressRow {
    pub dp_rank: Option<u32>,
    pub progress: crate::prefill_work::PrefillProgress,
    pub load: Value,
    pub routing: Option<Value>,
}

/// One `/v1/loads` poll: every DP rank's row, in reply order. Each rank is a
/// scheduler with its own RID namespace and its own progress epoch and
/// sequence; the caller gives each row to its rank's slot and never merges
/// two ranks. With `routing`, each row carries its raw `prefill_routing`
/// snapshot, removed from the load row so the trace's `engine_load` keeps its
/// size.
pub(crate) async fn fetch_progress_poll(
    client: &reqwest::Client,
    base: &str,
    routing: bool,
) -> anyhow::Result<Vec<ProgressRow>> {
    let include = if routing {
        "core,queues,prefill_work,prefill_routing"
    } else {
        "core,queues,prefill_work"
    };
    let mut value: Value = client
        .get(format!("{}/v1/loads", base.trim_end_matches('/')))
        .query(&[("include", include)])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let loads = value
        .get_mut("loads")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| anyhow::anyhow!("missing loads[]"))?;
    anyhow::ensure!(!loads.is_empty(), "empty loads[]");
    loads
        .drain(..)
        .map(|mut load| {
            let row = load
                .as_object_mut()
                .ok_or_else(|| anyhow::anyhow!("a loads[] row is not an object"))?;
            let dp_rank = row
                .get("dp_rank")
                .and_then(Value::as_u64)
                .and_then(|r| u32::try_from(r).ok());
            let progress = row.remove("prefill_work").ok_or_else(|| {
                anyhow::anyhow!("engine must enable --enable-prefill-work-snapshot")
            })?;
            let snapshot = row.remove("prefill_routing").filter(|_| routing);
            Ok(ProgressRow {
                dp_rank,
                progress: serde_json::from_value(progress)?,
                load,
                routing: snapshot,
            })
        })
        .collect()
}

/// The cursor of one poll's rows for [`Query::check`]: a single row's epoch
/// and sequence; for several ranks, their epochs joined and their sequences
/// summed (each rank's own cursor is checked again where its row is applied).
pub(crate) fn rows_cursor(rows: &[ProgressRow]) -> (String, u64) {
    if let [row] = rows {
        return (row.progress.epoch.clone(), row.progress.sequence);
    }
    let epoch = rows
        .iter()
        .map(|r| r.progress.epoch.as_str())
        .collect::<Vec<_>>()
        .join("|");
    let sequence = rows
        .iter()
        .fold(0u64, |sum, r| sum.saturating_add(r.progress.sequence));
    (epoch, sequence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_triggers_share_one_query_per_worker() {
        let refresh = Arc::new(LoadRefresh::default());
        let now = Instant::now();
        let age = Duration::from_millis(100);
        let tickets: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..32)
                .map(|_| {
                    let refresh = &refresh;
                    scope.spawn(move || refresh.claim("p0", now, age, false))
                })
                .collect();
            handles
                .into_iter()
                .filter_map(|h| h.join().unwrap())
                .collect()
        });
        assert_eq!(tickets.len(), 1);
        assert!(refresh.claim("p1", now, age, false).is_some());
        assert!(refresh.claim("p0", now, age, true).is_none());
    }

    #[test]
    fn slow_response_age_uses_query_start_and_rejects_old_sequences() {
        let refresh = Arc::new(LoadRefresh::default());
        let now = Instant::now();
        let age = Duration::from_millis(100);
        let query = refresh.claim("p", now, age, false).unwrap();
        assert!(query.accepts("boot1", 10));
        query.finish(Some(("boot1".into(), 10)), now + age * 3);
        assert!(refresh.initialized("p"));
        let query = refresh.claim("p", now + age * 3, age, false).unwrap();
        assert!(!query.accepts("boot1", 10));
        assert!(!query.accepts("boot1", 9));
        assert!(query.accepts("boot2", 0));
        // The same sequence is unchanged, an older one stale.
        assert!(query
            .check("boot1", 10)
            .unwrap_err()
            .is::<UnchangedProgress>());
        assert!(!query
            .check("boot1", 9)
            .unwrap_err()
            .is::<UnchangedProgress>());
        assert!(query.check("boot1", 11).is_ok());
    }

    #[test]
    fn failed_refresh_keeps_old_initialization_but_backs_off() {
        let refresh = Arc::new(LoadRefresh::default());
        let now = Instant::now();
        let age = Duration::from_millis(100);
        refresh
            .claim("p", now, age, false)
            .unwrap()
            .finish(Some(("boot".into(), 1)), now);
        assert!(refresh.claim("p", now + age, age, false).is_none());
        refresh
            .claim("p", now + age * 2, age, false)
            .unwrap()
            .finish(None, now + age * 3);
        assert!(refresh.initialized("p"));
        assert!(refresh.claim("p", now + age * 3, age, true).is_none());
        assert!(refresh.claim("p", now + age * 4, age, false).is_some());
    }

    /// A DP-attention worker answers one `/v1/loads` row per DP rank, each with
    /// its own scheduler's prefill progress; the poll returns every row with
    /// its rank, so the cost function's work book sees each rank's progress.
    #[tokio::test]
    async fn a_dp_worker_s_progress_poll_reads_every_rank() {
        use axum::{routing::get, Json, Router};
        let rows: Vec<Value> = (0..4)
            .map(|rank| {
                serde_json::json!({
                    "dp_rank": rank, "num_waiting_uncached_tokens": 0,
                    "prefill_work": {"epoch": format!("e{rank}"), "sequence": 1, "remaining": {}},
                })
            })
            .collect();
        let app = Router::new().route(
            "/v1/loads",
            get(move || {
                let rows = rows.clone();
                async move { Json(serde_json::json!({ "loads": rows })) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = reqwest::Client::new();
        let polled = fetch_progress_poll(&client, &url, false).await;
        assert!(polled.is_ok(), "{:?}", polled.err());
        let ranks: Vec<Option<u32>> = polled.unwrap().iter().map(|row| row.dp_rank).collect();
        assert_eq!(ranks, vec![Some(0), Some(1), Some(2), Some(3)]);
    }

    #[test]
    fn cancelled_query_does_not_permanently_occupy_slot() {
        let refresh = Arc::new(LoadRefresh::default());
        let age = Duration::from_millis(100);
        drop(refresh.claim("p", Instant::now(), age, false).unwrap());
        assert!(!refresh.initialized("p"));
        assert!(refresh
            .claim("p", Instant::now() + age * 2, age, false)
            .is_some());
    }
}
