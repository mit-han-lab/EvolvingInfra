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

/// The fields of one `/v1/loads` progress poll ([`fetch_progress_poll`]):
/// the prefill-work progress, the rest of the load row (the trace's
/// `engine_load`) and the raw `prefill_routing` snapshot.
pub(crate) type ProgressPoll = (crate::prefill_work::PrefillProgress, Value, Option<Value>);

/// One `/v1/loads` poll: the prefill-work progress, the rest of the load row
/// for the trace, and with `routing` the raw `prefill_routing` snapshot. The
/// snapshot is removed from the load row, so the trace's `engine_load` keeps
/// its size.
pub(crate) async fn fetch_progress_poll(
    client: &reqwest::Client,
    base: &str,
    routing: bool,
) -> anyhow::Result<ProgressPoll> {
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
    // TP shares one scheduler RID namespace. Supporting DP requires explicit
    // per-rank cursors and routing; never silently take an arbitrary rank.
    anyhow::ensure!(
        loads.len() == 1,
        "uncached balance requires DP=1 per worker"
    );
    let mut load = loads.swap_remove(0);
    let progress = load
        .as_object_mut()
        .and_then(|row| row.remove("prefill_work"))
        .ok_or_else(|| anyhow::anyhow!("engine must enable --enable-prefill-work-snapshot"))?;
    let snapshot = load
        .as_object_mut()
        .and_then(|row| row.remove("prefill_routing"))
        .filter(|_| routing);
    Ok((serde_json::from_value(progress)?, load, snapshot))
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
