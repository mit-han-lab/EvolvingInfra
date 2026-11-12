// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! Backend capabilities are distinct from the scheduler's logical role weights.
//!
//! Filters the model-specific worker registry into usable placement candidates.
//! Capability flags describe what an engine can execute, while role weights express
//! the scheduler preference; circuit-breaker eligibility is checked separately.
use crate::scheduler::placement::{Capabilities, RoleWeight, RoleWeights};
use crate::{
    discovery::{ModelId, WorkerMode},
    workers::{Worker, WorkerRegistry},
};
use parking_lot::RwLock;
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
    time::Instant,
};

/// Exclusive logical assignment for an external P/D capacity controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoleAssignment {
    Prefill,
    Decode,
}
impl RoleAssignment {
    fn weight(self) -> RoleWeight {
        match self {
            Self::Prefill => RoleWeight::PREFILL_LEANING,
            Self::Decode => RoleWeight::DECODE_LEANING,
        }
    }
}
pub type RoleAssignments = BTreeMap<String, RoleAssignment>;
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct RoleSnapshot {
    pub epoch: String,
    pub revision: u64,
    pub roles: RoleAssignments,
}

/// A worker reassigned from decode to prefill that takes no new legs until the
/// work already planned on it has drained.
#[derive(Debug, Clone, Copy)]
struct Drain {
    /// Revision of the assignment that started it.
    revision: u64,
    since: Instant,
    /// Releases have started; reversing the role now could race their delivery.
    clearing: bool,
}

#[derive(Debug, Default)]
struct RoleState {
    revision: u64,
    draining: BTreeMap<String, Drain>,
}

pub struct BackendRegistry {
    registry: Arc<WorkerRegistry>,
    model: ModelId,
    // Placement snapshots read every weight under this lock, so a batch of role
    // changes is never observed half-applied.
    roles: RwLock<RoleState>,
    epoch: String,
}
impl BackendRegistry {
    /// Bind a backend registry view to one model.
    ///
    /// # Inputs
    /// `registry` is the shared worker catalog; `model` selects its model group.
    ///
    /// # Outputs
    /// Returns a lightweight view that retains the registry through `Arc`.
    pub fn new(registry: Arc<WorkerRegistry>, model: ModelId) -> Self {
        Self {
            registry,
            model,
            roles: RwLock::new(RoleState::default()),
            epoch: uuid::Uuid::new_v4().to_string(),
        }
    }
    /// List registered engines capable of participating in disaggregated serving.
    ///
    /// # Inputs
    /// `self` supplies the shared registry and model ID.
    ///
    /// # Outputs
    /// Returns model-matching workers except `Plain` workers. This list does not
    /// apply the circuit-breaker filter used when selecting placement candidates.
    pub fn workers(&self) -> Vec<Arc<Worker>> {
        self.registry
            .workers_for(&self.model)
            .into_iter()
            .filter(|w| w.mode() != WorkerMode::Plain)
            .collect()
    }
    /// Build aligned worker and role maps for a placement decision.
    ///
    /// # Inputs
    /// `self` supplies the model-specific registry and current worker metadata.
    ///
    /// # Outputs
    /// Returns `(roles, workers)`, both keyed by worker URL, for non-plain
    /// workers whose circuit breakers currently allow requests. No engine state
    /// is changed and no network request is made. A draining worker is offered
    /// neither role.
    pub fn candidates(&self) -> (RoleWeights, HashMap<String, Arc<Worker>>) {
        self.role_view(true)
    }
    /// [`Self::candidates`] without the circuit-breaker filter: every
    /// non-plain worker of the model with its current role (a draining
    /// worker is offered neither role). For state that must follow role
    /// membership through a breaker trip, such as the late-D links of
    /// symmetric owners and the native decode-capacity poll (the PD path
    /// covers every prefill / decode worker there too); never for placement.
    pub fn members(&self) -> (RoleWeights, HashMap<String, Arc<Worker>>) {
        self.role_view(false)
    }
    /// The role view behind [`Self::candidates`] (`breaker_filter`) and
    /// [`Self::members`].
    fn role_view(&self, breaker_filter: bool) -> (RoleWeights, HashMap<String, Arc<Worker>>) {
        let state = self.roles.read();
        let mut roles = RoleWeights::new();
        let mut workers = HashMap::new();
        for w in self.workers() {
            if breaker_filter && !w.breaker.would_allow() {
                continue;
            }
            let Some((capabilities, mut weight)) = role_of(&w) else {
                continue;
            };
            if state.draining.contains_key(&w.url) {
                weight = RoleWeight::DRAINING;
            }
            roles.insert_with_caps(w.url.clone(), weight, capabilities);
            workers.insert(w.url.clone(), w);
        }
        (roles, workers)
    }
    /// Read the full model membership, including temporarily unhealthy workers.
    /// External control rejects native/mixed-role engines instead of guessing roles.
    /// A draining worker already reports its new role.
    pub fn assignments(&self) -> Result<RoleSnapshot, String> {
        let state = self.roles.read();
        Ok(RoleSnapshot {
            epoch: self.epoch.clone(),
            revision: state.revision,
            roles: Self::assignments_of(&self.workers())?,
        })
    }

    fn assignments_of(workers: &[Arc<Worker>]) -> Result<RoleAssignments, String> {
        workers
            .iter()
            .map(|w| {
                if w.mode() != WorkerMode::Symmetric {
                    return Err(format!(
                        "external role control requires symmetric worker {}",
                        w.url
                    ));
                }
                let info = w.symmetric_pd();
                if !info.capabilities.prefill || !info.capabilities.decode {
                    return Err(format!("worker {} does not support both roles", w.url));
                }
                let role = if info.weight == RoleWeight::PREFILL_LEANING {
                    RoleAssignment::Prefill
                } else if info.weight == RoleWeight::DECODE_LEANING {
                    RoleAssignment::Decode
                } else {
                    return Err(format!("worker {} has nonexclusive role weights", w.url));
                };
                Ok((w.url.clone(), role))
            })
            .collect()
    }

    /// Compare-and-replace a complete role map under the placement snapshot lock.
    /// No request is moved or engine restarted; existing plans retain their targets.
    ///
    /// With `drain`, a worker moving from decode to prefill is offered neither
    /// role until [`Self::finish_drain`] ends its drain; at least one other
    /// prefill worker must keep taking requests meanwhile. Assigning a draining
    /// worker back to decode ends its drain until metadata cleanup has begun.
    /// Prefill to decode is otherwise never delayed.
    ///
    /// # Outputs
    /// The new snapshot and the workers whose drain it started; their drains
    /// are identified by the snapshot's revision.
    pub fn replace_assignments(
        &self,
        epoch: &str,
        expected_revision: u64,
        expected: &RoleAssignments,
        desired: &RoleAssignments,
        drain: bool,
    ) -> Result<(RoleSnapshot, Vec<String>), String> {
        let mut state = self.roles.write();
        if epoch != self.epoch || expected_revision != state.revision {
            return Err("stale controller epoch/revision; reread before deciding".into());
        }
        let workers = self.workers();
        let current = Self::assignments_of(&workers)?;
        if &current != expected {
            return Err(
                "role assignment or worker membership changed; reread before deciding".into(),
            );
        }
        if desired.keys().ne(current.keys()) {
            return Err("desired roles must include exactly all registered workers".into());
        }
        if state
            .draining
            .iter()
            .any(|(worker, d)| d.clearing && desired.get(worker) != Some(&RoleAssignment::Prefill))
        {
            return Err(
                "session cleanup is in progress; wait for the role drain before reversing it"
                    .into(),
            );
        }
        if !desired.values().any(|r| *r == RoleAssignment::Prefill)
            || !desired.values().any(|r| *r == RoleAssignment::Decode)
        {
            return Err("at least one P and one D must remain".into());
        }
        let started: Vec<String> = desired
            .iter()
            .filter(|(url, role)| {
                drain
                    && **role == RoleAssignment::Prefill
                    && current[*url] == RoleAssignment::Decode
            })
            .map(|(url, _)| url.clone())
            .collect();
        let serving_prefill = desired.iter().any(|(url, role)| {
            *role == RoleAssignment::Prefill
                && !started.contains(url)
                && !state.draining.contains_key(url)
        });
        if !serving_prefill {
            return Err("a prefill worker that is not draining must remain".into());
        }
        let revision = state.revision + 1;
        for w in workers {
            let role = desired[&w.url];
            if role == RoleAssignment::Decode {
                state.draining.remove(&w.url);
            }
            let mut info = w.symmetric_pd();
            info.weight = role.weight();
            w.set_symmetric_pd(info);
        }
        let since = Instant::now();
        for url in &started {
            state.draining.insert(
                url.clone(),
                Drain {
                    revision,
                    since,
                    clearing: false,
                },
            );
        }
        state.revision = revision;
        Ok((
            RoleSnapshot {
                epoch: self.epoch.clone(),
                revision,
                roles: desired.clone(),
            },
            started,
        ))
    }

    /// End the drain that assignment `revision` started on `worker`, so it takes
    /// prefill legs. False when a later assignment already ended that drain.
    pub fn finish_drain(&self, worker: &str, revision: u64) -> bool {
        self.finish_drain_if(worker, revision, || true)
    }

    /// Start/finish metadata cleanup while holding the same lock as role changes.
    /// `ready` performs local bookkeeping only, never engine IO. A false result
    /// keeps the worker unavailable and prevents reversal while releases arrive.
    pub(crate) fn finish_drain_if(
        &self,
        worker: &str,
        revision: u64,
        ready: impl FnOnce() -> bool,
    ) -> bool {
        let mut state = self.roles.write();
        if state
            .draining
            .get(worker)
            .is_some_and(|d| d.revision == revision)
        {
            state.draining.get_mut(worker).unwrap().clearing = true;
            if !ready() {
                return false;
            }
            state.draining.remove(worker);
            return true;
        }
        false
    }

    /// Revalidate a selected pair and install its execution under the role lock.
    /// This closes the gap between async placement and the drain's legs_owed read.
    pub(crate) fn with_current_pair<T>(
        &self,
        p: &str,
        d: &str,
        install: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let state = self.roles.read();
        if state.draining.contains_key(p) || state.draining.contains_key(d) {
            return Err("selected worker is draining; replan the request".into());
        }
        let workers = self.workers();
        for (url, prefill) in [(p, true), (d, false)] {
            let valid = workers
                .iter()
                .find(|w| w.url == url)
                .and_then(|w| role_of(w))
                .is_some_and(|(caps, weight)| {
                    if prefill {
                        caps.prefill && weight.prefill > 0.0
                    } else {
                        caps.decode && weight.decode > 0.0
                    }
                });
            if !valid {
                return Err("selected worker role changed; replan the request".into());
            }
        }
        install()
    }

    /// Whether the drain that assignment `revision` started on `worker` is still in progress.
    pub fn is_draining(&self, worker: &str, revision: u64) -> bool {
        self.roles
            .read()
            .draining
            .get(worker)
            .is_some_and(|d| d.revision == revision)
    }

    /// Draining workers, with seconds since each drain started.
    pub fn drains(&self) -> BTreeMap<String, f64> {
        self.roles
            .read()
            .draining
            .iter()
            .map(|(url, d)| (url.clone(), d.since.elapsed().as_secs_f64()))
            .collect()
    }
}

/// Translate worker mode into execution capabilities and role preference.
///
/// # Inputs
/// `worker` supplies its serving mode and, for a symmetric engine, its
/// current advertised capabilities and scheduler role weight.
///
/// # Outputs
/// Returns `None` for plain serving, fixed capability/weight pairs for
/// native P or D engines, and advertised values for symmetric engines.
pub fn role_of(worker: &Worker) -> Option<(Capabilities, RoleWeight)> {
    match worker.mode() {
        WorkerMode::Plain => None,
        WorkerMode::Prefill => Some((Capabilities::PREFILL_ONLY, RoleWeight::PREFILL_LEANING)),
        WorkerMode::Decode => Some((Capabilities::DECODE_ONLY, RoleWeight::DECODE_LEANING)),
        WorkerMode::Symmetric => {
            let info = worker.symmetric_pd();
            Some((info.capabilities, info.weight))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{WorkerId, WorkerSpec};
    use crate::workers::SymmetricPdWorkerInfo;

    fn registry_of(workers: &[(&str, RoleWeight)]) -> BackendRegistry {
        let reg = Arc::new(WorkerRegistry::default());
        for (id, weight) in workers {
            reg.add(WorkerSpec {
                id: WorkerId((*id).into()),
                url: format!("http://{id}"),
                mode: WorkerMode::Symmetric,
                model_ids: vec![ModelId("m".into())],
                bootstrap_port: Some(8998),
            })
            .unwrap();
            reg.get(&WorkerId((*id).into()))
                .unwrap()
                .set_symmetric_pd(SymmetricPdWorkerInfo {
                    weight: *weight,
                    ..Default::default()
                });
        }
        BackendRegistry::new(reg, ModelId("m".into()))
    }

    fn registry() -> BackendRegistry {
        registry_of(&[
            ("p", RoleWeight::PREFILL_LEANING),
            ("d", RoleWeight::DECODE_LEANING),
        ])
    }

    /// `snapshot` with the given workers' roles changed.
    fn with_roles(snapshot: &RoleSnapshot, changes: &[(&str, RoleAssignment)]) -> RoleAssignments {
        let mut roles = snapshot.roles.clone();
        for (url, role) in changes {
            roles.insert((*url).into(), *role);
        }
        roles
    }

    fn weight(r: &BackendRegistry, url: &str) -> RoleWeight {
        r.candidates().0.workers[url].weight
    }

    /// Replace the assignment `from` with `desired`.
    fn replace(
        r: &BackendRegistry,
        from: &RoleSnapshot,
        desired: &RoleAssignments,
        drain: bool,
    ) -> Result<(RoleSnapshot, Vec<String>), String> {
        r.replace_assignments(&from.epoch, from.revision, &from.roles, desired, drain)
    }

    #[test]
    fn external_assignment_cas_rejects_stale_missing_and_single_role() {
        let r = registry();
        let before = r.assignments().unwrap();
        let desired = before
            .roles
            .iter()
            .map(|(url, role)| {
                (
                    url.clone(),
                    match role {
                        RoleAssignment::Prefill => RoleAssignment::Decode,
                        RoleAssignment::Decode => RoleAssignment::Prefill,
                    },
                )
            })
            .collect();
        let cas = |epoch: &str, revision, expected: &RoleAssignments, desired: &RoleAssignments| {
            r.replace_assignments(epoch, revision, expected, desired, false)
                .map(|(snapshot, _)| snapshot)
        };
        assert!(cas("old-instance", 0, &before.roles, &desired).is_err());
        let after = cas(&before.epoch, before.revision, &before.roles, &desired).unwrap();
        assert_eq!(after.revision, before.revision + 1);
        assert!(cas(&before.epoch, before.revision, &before.roles, &desired).is_err());
        assert!(cas(&after.epoch, after.revision, &after.roles, &BTreeMap::new()).is_err());
        let all_p = after
            .roles
            .keys()
            .map(|url| (url.clone(), RoleAssignment::Prefill))
            .collect();
        assert!(cas(&after.epoch, after.revision, &after.roles, &all_p).is_err());
        assert_eq!(r.assignments().unwrap().roles, after.roles);
        assert_eq!(r.assignments().unwrap().revision, after.revision);
    }

    #[test]
    fn external_control_refuses_mixed_or_native_workers() {
        let r = registry();
        let p = r
            .workers()
            .into_iter()
            .find(|w| w.url == "http://p")
            .unwrap();
        let mut info = p.symmetric_pd();
        info.weight = RoleWeight {
            prefill: 0.5,
            decode: 0.5,
        };
        p.set_symmetric_pd(info);
        assert!(r.assignments().is_err());
    }

    #[test]
    fn decode_to_prefill_drains_until_its_drain_is_finished() {
        let r = registry_of(&[
            ("p", RoleWeight::PREFILL_LEANING),
            ("d1", RoleWeight::DECODE_LEANING),
            ("d2", RoleWeight::DECODE_LEANING),
        ]);
        let before = r.assignments().unwrap();
        let desired = with_roles(&before, &[("http://d1", RoleAssignment::Prefill)]);
        let (after, started) = replace(&r, &before, &desired, true).unwrap();
        assert_eq!(started, vec!["http://d1".to_string()]);
        assert_eq!(weight(&r, "http://d1"), RoleWeight::DRAINING);
        assert_eq!(weight(&r, "http://p"), RoleWeight::PREFILL_LEANING);
        assert_eq!(weight(&r, "http://d2"), RoleWeight::DECODE_LEANING);
        // The controller already sees its assignment.
        assert_eq!(r.assignments().unwrap(), after);
        assert!(r.drains().contains_key("http://d1"));
        assert!(r.is_draining("http://d1", after.revision));
        assert!(!r.finish_drain("http://d1", after.revision + 1));
        assert!(r.finish_drain("http://d1", after.revision));
        assert_eq!(weight(&r, "http://d1"), RoleWeight::PREFILL_LEANING);
        assert!(!r.finish_drain("http://d1", after.revision));
        assert!(r.drains().is_empty());
    }

    #[test]
    fn reassigning_a_draining_worker_to_decode_ends_its_drain() {
        let r = registry_of(&[
            ("p", RoleWeight::PREFILL_LEANING),
            ("d1", RoleWeight::DECODE_LEANING),
            ("d2", RoleWeight::DECODE_LEANING),
        ]);
        let before = r.assignments().unwrap();
        let desired = with_roles(&before, &[("http://d1", RoleAssignment::Prefill)]);
        let (draining, _) = replace(&r, &before, &desired, true).unwrap();
        let (_, started) = replace(&r, &draining, &before.roles, true).unwrap();
        assert!(started.is_empty());
        assert!(!r.is_draining("http://d1", draining.revision));
        assert!(!r.finish_drain("http://d1", draining.revision));
        assert_eq!(weight(&r, "http://d1"), RoleWeight::DECODE_LEANING);
    }

    #[test]
    fn prefill_to_decode_and_undrained_changes_take_effect_at_once() {
        let r = registry_of(&[
            ("p1", RoleWeight::PREFILL_LEANING),
            ("p2", RoleWeight::PREFILL_LEANING),
            ("d", RoleWeight::DECODE_LEANING),
        ]);
        let before = r.assignments().unwrap();
        let desired = with_roles(&before, &[("http://p1", RoleAssignment::Decode)]);
        let (after, started) = replace(&r, &before, &desired, true).unwrap();
        assert!(started.is_empty());
        assert_eq!(weight(&r, "http://p1"), RoleWeight::DECODE_LEANING);
        let desired = with_roles(&after, &[("http://d", RoleAssignment::Prefill)]);
        let (_, started) = replace(&r, &after, &desired, false).unwrap();
        assert!(started.is_empty());
        assert_eq!(weight(&r, "http://d"), RoleWeight::PREFILL_LEANING);
        assert!(r.drains().is_empty());
    }

    #[test]
    fn a_drain_must_leave_a_prefill_worker_taking_requests() {
        let r = registry();
        let before = r.assignments().unwrap();
        let swap = with_roles(
            &before,
            &[
                ("http://p", RoleAssignment::Decode),
                ("http://d", RoleAssignment::Prefill),
            ],
        );
        assert!(replace(&r, &before, &swap, true).is_err());
        assert_eq!(r.assignments().unwrap(), before);
        assert_eq!(weight(&r, "http://p"), RoleWeight::PREFILL_LEANING);
        assert!(replace(&r, &before, &swap, false).is_ok());
    }

    #[test]
    fn selected_pair_is_revalidated_and_install_holds_role_lock() {
        let r = registry_of(&[
            ("p", RoleWeight::PREFILL_LEANING),
            ("d1", RoleWeight::DECODE_LEANING),
            ("d2", RoleWeight::DECODE_LEANING),
        ]);
        r.with_current_pair("http://p", "http://d1", || {
            assert!(
                r.roles.try_write().is_none(),
                "role changes cannot pass installation"
            );
            Ok(())
        })
        .unwrap();
        assert!(r.roles.try_write().is_some());
        let before = r.assignments().unwrap();
        let desired = with_roles(&before, &[("http://d1", RoleAssignment::Prefill)]);
        let (after, _) = replace(&r, &before, &desired, true).unwrap();
        for (p, d) in [("http://p", "http://d1"), ("http://d1", "http://d2")] {
            assert!(r
                .with_current_pair(p, d, || -> Result<(), String> {
                    panic!("a draining pair must not install");
                })
                .is_err());
        }
        assert!(r.finish_drain("http://d1", after.revision));
        assert!(r
            .with_current_pair("http://p", "http://d1", || -> Result<(), String> {
                panic!("a stale decode placement must not install");
            })
            .is_err());
        assert!(r
            .with_current_pair("http://d1", "http://d2", || Ok(()))
            .is_ok());
    }

    #[test]
    fn cleanup_ack_gate_rejects_reversal_and_stale_drain_callbacks() {
        let r = registry_of(&[
            ("p", RoleWeight::PREFILL_LEANING),
            ("d1", RoleWeight::DECODE_LEANING),
            ("d2", RoleWeight::DECODE_LEANING),
        ]);
        let before = r.assignments().unwrap();
        let desired = with_roles(&before, &[("http://d1", RoleAssignment::Prefill)]);
        let (after, _) = replace(&r, &before, &desired, true).unwrap();
        assert!(!r.finish_drain_if("http://d1", after.revision + 1, || {
            panic!("stale drain must not clear registrations");
        }));
        assert!(!r.finish_drain_if("http://d1", after.revision, || false));
        assert_eq!(weight(&r, "http://d1"), RoleWeight::DRAINING);
        assert!(replace(&r, &after, &before.roles, true)
            .unwrap_err()
            .contains("session cleanup"));
        assert_eq!(r.assignments().unwrap(), after);
        assert!(r.finish_drain_if("http://d1", after.revision, || true));
        assert_eq!(weight(&r, "http://d1"), RoleWeight::PREFILL_LEANING);
        assert!(replace(&r, &after, &before.roles, true).is_ok());
        assert_eq!(weight(&r, "http://d1"), RoleWeight::DECODE_LEANING);
    }
}
