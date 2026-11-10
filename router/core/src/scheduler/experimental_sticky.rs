//! Explicit, default-off fixed-affinity experiment on symmetric workers.
//! Engine capabilities stay BOTH; role weights define the logical P/D groups.
use super::RoleWeights;
use crate::policies::experimental_agent_sticky::sticky_slot;

#[derive(Debug, serde::Deserialize)]
pub(super) struct ExperimentSticky {
    pub prefill: Vec<String>,
    pub decode: Vec<String>,
}

impl ExperimentSticky {
    /// Load the explicit opt-in sticky-routing experiment configuration.
    ///
    /// # Inputs
    /// No arguments; reads `EVOLVING_INFRA_EXPERIMENT_SYMMETRIC_PD_STICKY` from the process environment.
    ///
    /// # Outputs
    /// `None` if the environment value is absent or unreadable; otherwise the parsed P/D URL
    /// groups. Invalid JSON, empty/duplicate URLs, or overlapping groups panic at startup.
    pub fn from_environment() -> Option<Self> {
        let raw = std::env::var("EVOLVING_INFRA_EXPERIMENT_SYMMETRIC_PD_STICKY").ok()?;
        let value: Self = serde_json::from_str(&raw).expect("experimental flex sticky JSON");
        for urls in [&value.prefill, &value.decode] {
            assert!(!urls.is_empty(), "experimental role group cannot be empty");
            assert!(
                urls.iter()
                    .enumerate()
                    .all(|(i, u)| !u.is_empty() && !urls[..i].contains(u)),
                "duplicate experimental worker"
            );
        }
        assert!(
            value.prefill.iter().all(|p| !value.decode.contains(p)),
            "experimental role groups must be disjoint"
        );
        Some(value)
    }

    /// Map an agent routing key to a fixed P and rotate D by the supplied seed.
    ///
    /// # Inputs
    /// `self`: configured experimental URL order; `key`: nonempty agent affinity key;
    /// `weights`: current eligible roles/capabilities; `seed`: D round-robin position.
    ///
    /// # Outputs
    /// `Some((P URL, D URL))` only when both current role pools exactly match the experiment
    /// configuration; `None` for a missing key or changed pool. Worker capabilities are
    /// unchanged.
    pub fn pair(
        &self,
        key: Option<&str>,
        weights: &RoleWeights,
        seed: u64,
    ) -> Option<(String, String)> {
        let key = key.filter(|k| !k.is_empty())?;
        let p: Vec<_> = weights
            .workers
            .iter()
            .filter(|(_, r)| r.capabilities.prefill && r.weight.prefill > 0.0)
            .map(|(u, _)| u)
            .collect();
        let d: Vec<_> = weights
            .workers
            .iter()
            .filter(|(_, r)| r.capabilities.decode && r.weight.decode > 0.0)
            .map(|(u, _)| u)
            .collect();
        if p.len() != self.prefill.len()
            || d.len() != self.decode.len()
            || !self.prefill.iter().all(|u| p.contains(&u))
            || !self.decode.iter().all(|u| d.contains(&u))
        {
            return None;
        }
        Some((
            self.prefill[sticky_slot(key, self.prefill.len())].clone(),
            self.decode[(seed % self.decode.len() as u64) as usize].clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symmetric_pd::{Capabilities, RoleWeight};
    #[test]
    /// Check sticky P affinity and D round robin preserve capabilities and reject missing keys
    /// or a changed pool.
    ///
    /// # Inputs
    /// No arguments; constructs two logical P and three logical D workers with symmetric
    /// capabilities.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn symmetric_affinity_and_rr_preserve_capabilities() {
        let sticky = ExperimentSticky {
            prefill: vec!["p1".into(), "p0".into()],
            decode: vec!["d0".into(), "d1".into(), "d2".into()],
        };
        let mut roles = RoleWeights::new();
        for p in &sticky.prefill {
            roles.insert(p, RoleWeight::PREFILL_LEANING);
        }
        for d in &sticky.decode {
            roles.insert(d, RoleWeight::DECODE_LEANING);
        }
        for seed in 0..1000 {
            let key = format!("agent-{}", seed % 31);
            let pair = sticky.pair(Some(&key), &roles, seed).unwrap();
            assert_eq!(pair.0, sticky.prefill[sticky_slot(&key, 2)]);
            assert_eq!(pair.1, sticky.decode[seed as usize % 3]);
        }
        assert!(roles
            .workers
            .values()
            .all(|r| r.capabilities == Capabilities::BOTH));
        assert!(sticky.pair(None, &roles, 0).is_none());
        roles.workers.remove("p1");
        assert!(sticky.pair(Some("agent"), &roles, 0).is_none());
    }
}
