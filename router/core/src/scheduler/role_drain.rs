//! Finishing a decode->prefill role change made through external role control.
//!
//! The registry already offers the worker neither role. A drain waits for the
//! legs planned on it, aborts the D-legs still running at the deadline (their
//! tokenizers resume them on newly placed pairs after asking
//! `/symmetric_pd/migration`), and only then lets the worker take prefill legs, so its
//! remaining decodes never share it with a prefill backlog. Prefill->decode
//! changes need no drain.
//! With session retention, abort delivery is followed by terminal confirmation
//! and acknowledged local reference cleanup before exposing the new P role.
use crate::backend::{execution::Executions, sglang::request::cancel_engines};
use crate::scheduler::SymmetricPdRuntime;
use std::sync::Arc;
use tokio::time::{Duration, Instant};
use tracing::info;

const POLL: Duration = Duration::from_millis(200);

/// Start a drain for each worker that assignment `revision` moved from decode to prefill.
pub fn spawn(
    flex: &Arc<SymmetricPdRuntime>,
    executions: &Arc<Executions>,
    client: &reqwest::Client,
    workers: Vec<String>,
    revision: u64,
    timeout: Duration,
) {
    for worker in workers {
        tokio::spawn(drain_to_prefill(
            Arc::clone(flex),
            Arc::clone(executions),
            client.clone(),
            worker,
            revision,
            timeout,
        ));
    }
}

/// Drain `worker` toward prefill for the assignment `revision`.
///
/// # Inputs
/// `timeout` bounds the initial wait for its legs; `client` sends engine aborts.
/// Retention additionally waits for terminal callbacks and cleanup acknowledgements.
///
/// # Outputs
/// None. Stops without acting when a later assignment ends the drain;
/// otherwise migrates the D-legs left at the deadline, completes any required
/// retention cleanup, ends the drain and logs one record.
pub async fn drain_to_prefill(
    flex: Arc<SymmetricPdRuntime>,
    executions: Arc<Executions>,
    client: reqwest::Client,
    worker: String,
    revision: u64,
    timeout: Duration,
) {
    let started = Instant::now();
    let mut owed = executions.legs_owed(&worker);
    while owed > 0 && started.elapsed() < timeout {
        if !flex.is_role_draining(&worker, revision) {
            return;
        }
        tokio::time::sleep(POLL).await;
        owed = executions.legs_owed(&worker);
    }
    if !flex.is_role_draining(&worker, revision) {
        return;
    }
    let targets = executions.migrate_decode(&worker);
    let migrated = targets.len();
    if !targets.is_empty() {
        cancel_engines(client, targets).await;
    }
    // Abort delivery is not terminal evidence. Retention cleanup must wait for
    // the existing lifecycle callbacks (or expiry of never-claimed plans).
    // Then the same drain stays closed until versioned releases are acknowledged.
    let finished = loop {
        if !flex.is_role_draining(&worker, revision) {
            return;
        }
        owed = executions.legs_owed(&worker);
        if (flex.kv_sessions().is_none() || owed == 0) && flex.finish_role_drain(&worker, revision)
        {
            break true;
        }
        tokio::time::sleep(POLL).await;
    };
    info!(target: "symmetric_pd", worker = %worker, revision, drain_s = started.elapsed().as_secs_f64(),
          legs_owed_at_end = owed, migrated, finished, "decode->prefill drain ended");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::execution::{ExecutionEvent, Leg, LegState};
    use crate::backend::registry::RoleAssignment;
    use crate::backend::state::BackendState;
    use crate::config::SymmetricPdConfig;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::scheduler::placement::RoleWeight;
    use crate::workers::{SymmetricPdWorkerInfo, WorkerRegistry};

    /// Prefill "http://p" and decode "http://d" and "http://d2" under external
    /// control with a 30 s drain, built outside Tokio (no load poller).
    fn runtime() -> Arc<SymmetricPdRuntime> {
        runtime_with_retention(false)
    }

    fn runtime_with_retention(kv_session_retention: bool) -> Arc<SymmetricPdRuntime> {
        let registry = Arc::new(WorkerRegistry::default());
        for (id, weight) in [
            ("p", RoleWeight::PREFILL_LEANING),
            ("d", RoleWeight::DECODE_LEANING),
            ("d2", RoleWeight::DECODE_LEANING),
        ] {
            registry
                .add(WorkerSpec {
                    id: WorkerId(id.into()),
                    url: format!("http://{id}"),
                    mode: WorkerMode::Symmetric,
                    model_ids: vec![ModelId("m".into())],
                    bootstrap_port: Some(8998),
                })
                .unwrap();
            registry
                .get(&WorkerId(id.into()))
                .unwrap()
                .set_symmetric_pd(SymmetricPdWorkerInfo {
                    weight,
                    ..Default::default()
                });
        }
        let cfg = SymmetricPdConfig {
            external_role_control: true,
            role_drain_timeout: Some(Duration::from_secs(30)),
            kv_session_retention,
            ..Default::default()
        };
        std::thread::spawn(move || SymmetricPdRuntime::spawn(cfg, None, registry, ModelId("m".into())))
            .join()
            .unwrap()
    }

    /// Assign "http://d" `role`; returns the new revision.
    fn assign_d(flex: &SymmetricPdRuntime, role: RoleAssignment) -> u64 {
        let now = flex.role_assignments().unwrap();
        let mut desired = now.roles.clone();
        desired.insert("http://d".into(), role);
        let (after, _) = flex
            .replace_role_assignments(&now.epoch, now.revision, &now.roles, &desired)
            .unwrap();
        after.revision
    }

    fn weight(flex: &SymmetricPdRuntime, url: &str) -> RoleWeight {
        flex.candidates().0.workers[url].weight
    }

    /// A claimed request with its P-leg on "http://p" and D-leg on "http://d".
    fn executions() -> Arc<Executions> {
        let mut state = BackendState::new(Arc::new(tokio::sync::Notify::new()));
        let executions = Arc::new(Executions::default());
        executions.install_claimed_for_test(
            "r",
            "secret",
            ["http://p", "http://d"],
            state.reserve("http://p", 1),
        );
        executions
    }

    fn decode_completed() -> ExecutionEvent {
        ExecutionEvent {
            protocol_version: 1,
            request_id: "rid-r".into(),
            reservation_id: "r".into(),
            transfer_id: 1,
            leg: Leg::Decode,
            state: LegState::Completed,
            event_id: "r:decode:completed".into(),
            http_status: Some(200),
            terminal_response: false,
            not_submitted: false,
            usage: None,
            candidate: None,
            engine_abort: None,
        }
    }

    fn drain(
        flex: &Arc<SymmetricPdRuntime>,
        executions: &Arc<Executions>,
        revision: u64,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(drain_to_prefill(
            Arc::clone(flex),
            Arc::clone(executions),
            reqwest::Client::new(),
            "http://d".into(),
            revision,
            Duration::from_secs(30),
        ))
    }

    #[tokio::test(start_paused = true)]
    async fn a_drained_worker_takes_prefill_without_migrating() {
        let (flex, executions) = (runtime(), executions());
        let revision = assign_d(&flex, RoleAssignment::Prefill);
        assert_eq!(weight(&flex, "http://d"), RoleWeight::DRAINING);
        let task = drain(&flex, &executions, revision);
        tokio::time::sleep(Duration::from_secs(10)).await;
        assert_eq!(
            weight(&flex, "http://d"),
            RoleWeight::DRAINING,
            "its D-leg is still running"
        );
        executions.event(&decode_completed(), "secret").unwrap();
        task.await.unwrap();
        assert_eq!(weight(&flex, "http://d"), RoleWeight::PREFILL_LEANING);
        assert_eq!(executions.migration_status("r", "secret"), Ok(false));
        assert!(flex.role_drains().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn d_legs_left_at_the_deadline_migrate() {
        let (flex, executions) = (runtime(), executions());
        let revision = assign_d(&flex, RoleAssignment::Prefill);
        let started = Instant::now();
        drain(&flex, &executions, revision).await.unwrap();
        assert!(started.elapsed() >= Duration::from_secs(30));
        assert_eq!(executions.migration_status("r", "secret"), Ok(true));
        assert_eq!(weight(&flex, "http://d"), RoleWeight::PREFILL_LEANING);
    }

    #[tokio::test(start_paused = true)]
    async fn reassigning_the_worker_to_decode_ends_its_drain() {
        let (flex, executions) = (runtime(), executions());
        let revision = assign_d(&flex, RoleAssignment::Prefill);
        let task = drain(&flex, &executions, revision);
        tokio::time::sleep(Duration::from_secs(5)).await;
        assign_d(&flex, RoleAssignment::Decode);
        task.await.unwrap();
        assert_eq!(weight(&flex, "http://d"), RoleWeight::DECODE_LEANING);
        assert_eq!(
            executions.migration_status("r", "secret"),
            Ok(false),
            "nothing migrated"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn retention_waits_for_terminal_evidence_after_abort_deadline() {
        let (flex, executions) = (runtime_with_retention(true), executions());
        let revision = assign_d(&flex, RoleAssignment::Prefill);
        let task = drain(&flex, &executions, revision);
        tokio::time::sleep(Duration::from_secs(40)).await;
        assert_eq!(executions.migration_status("r", "secret"), Ok(true));
        assert_eq!(executions.legs_owed("http://d"), 1);
        assert_eq!(weight(&flex, "http://d"), RoleWeight::DRAINING);
        assert!(!task.is_finished());
        executions.event(&decode_completed(), "secret").unwrap();
        task.await.unwrap();
        assert_eq!(weight(&flex, "http://d"), RoleWeight::PREFILL_LEANING);
        assert!(flex.role_drains().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn drained_worker_stays_unavailable_until_session_releases_are_acknowledged() {
        let (flex, executions) = (runtime_with_retention(true), executions());
        let sessions = flex.kv_sessions().unwrap();
        let identity = crate::agent_identity::RequestIdentity {
            agent_id: "role-clear".into(),
            subagent_id: 0,
            request_id: 1,
        };
        sessions
            .register(&identity, "http://p", "http://d")
            .unwrap();
        let revision = assign_d(&flex, RoleAssignment::Prefill);
        executions.event(&decode_completed(), "secret").unwrap();
        let task = drain(&flex, &executions, revision);
        tokio::time::sleep(Duration::from_secs(1)).await;
        assert_eq!(executions.legs_owed("http://d"), 0);
        assert_eq!(weight(&flex, "http://d"), RoleWeight::DRAINING);
        assert!(!task.is_finished());
        assert!(sessions
            .register(&identity, "http://d", "http://d2")
            .is_err());
        assert!(!flex.finish_role_drain("http://d", revision));
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
    }

    #[test]
    fn prefill_to_decode_keeps_session_registration_and_retry_version() {
        let flex = runtime_with_retention(true);
        let revision = assign_d(&flex, RoleAssignment::Prefill);
        assert!(flex.finish_role_drain("http://d", revision));
        let sessions = flex.kv_sessions().unwrap();
        let identity = crate::agent_identity::RequestIdentity {
            agent_id: "preserved-prefill".into(),
            subagent_id: 0,
            request_id: 1,
        };
        let before = sessions
            .register(&identity, "http://d", "http://d2")
            .unwrap();
        assign_d(&flex, RoleAssignment::Decode);
        assert_eq!(weight(&flex, "http://d"), RoleWeight::DECODE_LEANING);
        assert!(flex.role_drains().is_empty());
        assert_eq!(
            sessions
                .register(&identity, "http://d", "http://d2")
                .unwrap(),
            before
        );
    }

    /// With retention, a D-leg cancelled without terminal evidence blocked the
    /// D->P drain forever (expiry off = the earlier behaviour); with the 300-s
    /// unresolved D-leg expiry the drain finishes once the leg is quiet that long.
    #[tokio::test(start_paused = true)]
    async fn retention_drain_unblocked_by_unresolved_decode_expiry() {
        for after in [None, Some(Duration::from_secs(300))] {
            let flex = runtime_with_retention(true);
            let mut state = BackendState::new(Arc::new(tokio::sync::Notify::new()));
            let executions = Arc::new(Executions::default().with_decode_unresolved_expiry(after));
            executions.install_claimed_for_test(
                "r",
                "secret",
                ["http://p", "http://d"],
                state.reserve("http://p", 1),
            );
            let mut p_done = decode_completed();
            p_done.leg = Leg::Prefill;
            p_done.event_id = "r:prefill:completed".into();
            executions.event(&p_done, "secret").unwrap();
            let mut d_cancel = decode_completed();
            d_cancel.state = LegState::Cancelled;
            d_cancel.event_id = "r:decode:cancelled".into();
            d_cancel.http_status = None;
            executions.event(&d_cancel, "secret").unwrap();
            let revision = assign_d(&flex, RoleAssignment::Prefill);
            let task = drain(&flex, &executions, revision);
            tokio::time::sleep(Duration::from_secs(60)).await;
            assert_eq!(executions.legs_owed("http://d"), 1);
            assert!(!task.is_finished());
            executions.age_for_test("r", 301);
            tokio::time::sleep(Duration::from_secs(1)).await;
            if after.is_none() {
                assert!(
                    !task.is_finished(),
                    "without the expiry the drain never ends"
                );
                assert_eq!(weight(&flex, "http://d"), RoleWeight::DRAINING);
                task.abort();
            } else {
                task.await.unwrap();
                assert_eq!(weight(&flex, "http://d"), RoleWeight::PREFILL_LEANING);
            }
        }
    }
}
