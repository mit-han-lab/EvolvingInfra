//! Compiled Rust scheduling extensions. Policies receive a borrowed snapshot;
//! the runtime owns refresh, validation, reservation and execution. Calls run
//! inside Backend's select-and-reserve lock: no I/O, blocking, fitting or second
//! ledger belongs here. Fit parameters beforehand and publish an immutable model.
use super::placement::*;

/// Replace the complete placement rule while sharing Backend state/lifecycle.
/// Return only candidates from `roles`; the runtime validates capability, role,
/// force-split and transfer identity before it reserves any work.
pub trait PlacementPolicy: std::fmt::Debug + Send + Sync {
    /// Select a placement from an authoritative borrowed backend snapshot.
    ///
    /// # Inputs
    /// `self`: compiled policy; `request`: request metadata/seed; `hits`: validated hit
    /// estimates; `loads`: current work snapshot; `roles`: allowed candidates; `config`:
    /// placement constraints.
    ///
    /// # Outputs
    /// A candidate `Placement` or `PlacementError`; the runtime validates and reserves
    /// afterward. Implementations must not perform I/O, block, fit parameters, or maintain
    /// another ledger.
    fn place(
        &self,
        request: &PlacementRequest,
        hits: &IndexerHits,
        loads: &WorkerLoads,
        roles: &RoleWeights,
        config: &PlacementConfig,
    ) -> Result<Placement, PlacementError>;
}

/// Keep existing selection, tie-breaks and D RR; replace only its cost model.
#[derive(Debug, Default)]
pub struct DefaultPlacementPolicy<C = DefaultCostModel> {
    pub cost: C,
}
impl<C: CostModel> PlacementPolicy for DefaultPlacementPolicy<C> {
    /// Apply the standard selectors using this policy's cost model.
    ///
    /// # Inputs
    /// `self.cost`: injected pure cost model; `request`, `hits`, `loads`, `roles`, `config`:
    /// current borrowed placement inputs.
    ///
    /// # Outputs
    /// The result from `place_with_cost`, including its typed eligibility/non-finite-score
    /// errors; no Backend state is mutated.
    fn place(
        &self,
        request: &PlacementRequest,
        hits: &IndexerHits,
        loads: &WorkerLoads,
        roles: &RoleWeights,
        config: &PlacementConfig,
    ) -> Result<Placement, PlacementError> {
        place_with_cost(request, hits, loads, roles, config, &self.cost)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PrefillCostInput<'a> {
    pub worker: &'a str,
    pub request: &'a PlacementRequest,
    pub load: WorkerLoad,
    pub local_hit: u64,
    pub gpu_hit: u64,
    /// Only includes remote hit after a valid pull hint has been constructed.
    pub expected_hit: u64,
}

/// Comparable costs on a single shared scale (e.g. tokens or integer us).
/// This is an estimate used to select a node, never the accounting ledger.
#[derive(Debug, Clone, Copy)]
pub struct PrefillCost {
    pub before: u128,
    pub after: u128,
}

#[derive(Debug, Clone, Copy)]
pub struct PairCostInput<'a> {
    pub request: &'a PlacementRequest,
    pub prefill: &'a str,
    pub decode: &'a str,
    pub reuse_gain: f64,
    pub backlog: f64,
    pub role_penalty_prefill: f64,
    pub role_penalty_decode: f64,
    pub transfer: f64,
    pub decode_load: f64,
}

pub trait CostModel: std::fmt::Debug + Send + Sync {
    /// Estimate comparable prefill work before and after admitting this request.
    ///
    /// # Inputs
    /// `self`: immutable cost parameters; `input`: candidate, request, current load, local/GPU
    /// hit and validated expected hit.
    ///
    /// # Outputs
    /// `PrefillCost` with before/after work on one shared scale across candidates.
    /// Called by CoSchedule when no valid replay placement pins the request, once
    /// per eligible P that has a compatible D. CacheFirst and Score do not use
    /// this method; values guide selection, not raw-token accounting.
    fn prefill_cost(&self, input: PrefillCostInput<'_>) -> PrefillCost;

    /// Convert the configured token margin to the SAME units as prefill_cost.
    /// Required even for a model in time units: never compare tokens to us.
    /// The separate owner_threshold_tokens guard always uses raw uncached tokens.
    /// Convert the configured token margin to the model's work units.
    ///
    /// # Inputs
    /// `self`: cost model; `tokens`: configured uncached-token detour margin.
    ///
    /// # Outputs
    /// A nonnegative margin on the same scale as `prefill_cost`; the separate cached-owner
    /// threshold remains in raw tokens.
    fn detour_margin(&self, tokens: u64) -> u128;

    /// Higher wins in Score mode; CacheFirst/CoSchedule still pin their P.
    /// Override for custom pair scoring; prefill-only models retain this score.
    /// Provide the default additive ranking for an eligible P/D pair.
    ///
    /// # Inputs
    /// `input`: normalized pair features and request/worker identities; `weights`: coefficients
    /// for the default score.
    ///
    /// # Outputs
    /// A floating-point score where larger is preferred; delegates to `DefaultCostModel`. The
    /// placement layer rejects non-finite values.
    fn pair_score(&self, input: PairCostInput<'_>, weights: ScoreWeights) -> f64 {
        DefaultCostModel.pair_score(input, weights)
    }
}

#[derive(Debug, Default)]
pub struct DefaultCostModel;
impl CostModel for DefaultCostModel {
    /// Use outstanding uncached tokens as the default prefill cost.
    ///
    /// # Inputs
    /// `input`: worker load, prompt length and expected usable prefix hit; `self` has no
    /// parameters.
    ///
    /// # Outputs
    /// Before = outstanding uncached tokens; after = before plus
    /// `prompt_len.saturating_sub(expected_hit)`, both widened to `u128`.
    fn prefill_cost(&self, input: PrefillCostInput<'_>) -> PrefillCost {
        let before = u128::from(input.load.backlog_uncached_tokens);
        PrefillCost {
            before,
            after: before + u128::from(input.request.prompt_len.saturating_sub(input.expected_hit)),
        }
    }
    /// Keep detour margins in the default raw-token cost units.
    ///
    /// # Inputs
    /// `tokens`: detour margin in uncached tokens; `self` is stateless.
    ///
    /// # Outputs
    /// The same numeric margin widened to `u128`.
    fn detour_margin(&self, tokens: u64) -> u128 {
        u128::from(tokens)
    }
    /// Combine normalized cache benefit and load/transfer penalties.
    ///
    /// # Inputs
    /// `input`: cache gain, backlog, role penalties, transfer cost and decode load; `w`:
    /// additive coefficients; `self` is stateless.
    ///
    /// # Outputs
    /// The original weighted floating-point score with its operation order preserved; no
    /// validation or state mutation occurs here.
    fn pair_score(&self, input: PairCostInput<'_>, w: ScoreWeights) -> f64 {
        // Preserve the original operation order, including floating-point ties.
        w.w_hit * input.reuse_gain
            - w.w_backlog * input.backlog
            - w.w_role * (input.role_penalty_prefill + input.role_penalty_decode)
            - w.w_xfer * input.transfer
            - w.w_dload * input.decode_load
    }
}

/// Validate even custom policy output before protocol construction/reservation.
/// Check a policy decision before creating marks or reserving work.
///
/// # Inputs
/// `placement`: policy output; `request`: authoritative transfer seed; `roles`: current
/// candidate capabilities/weights; `config`: force-split constraint.
///
/// # Outputs
/// `Ok(())` when both legs are eligible and transfer identity matches Local/Split; otherwise
/// `InvalidPolicyOutput` with a reason. No reservation is created.
pub fn validate_placement(
    placement: &Placement,
    request: &PlacementRequest,
    roles: &RoleWeights,
    config: &PlacementConfig,
) -> Result<(), PlacementError> {
    let invalid = |reason: &str| PlacementError::InvalidPolicyOutput(reason.into());
    let p = roles
        .workers
        .get(&placement.p_leg)
        .ok_or_else(|| invalid("P is not a candidate"))?;
    let d = roles
        .workers
        .get(&placement.d_leg)
        .ok_or_else(|| invalid("D is not a candidate"))?;
    if !p.capabilities.prefill || p.weight.prefill <= 0.0 {
        return Err(invalid("P lacks capability or positive role weight"));
    }
    if !d.capabilities.decode || d.weight.decode <= 0.0 {
        return Err(invalid("D lacks capability or positive role weight"));
    }
    if config.force_split && placement.is_local() {
        return Err(invalid("Local placement violates force_split"));
    }
    if placement.transfer_id != (!placement.is_local()).then_some(request.transfer_seed) {
        return Err(invalid(
            "transfer identity must be the runtime's seed for Split, absent for Local",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug, Default)]
    struct TimeCost {
        calls: AtomicUsize,
    }
    impl CostModel for TimeCost {
        /// Simulate a slower p0 in the custom-cost test model.
        ///
        /// # Inputs
        /// `self.calls`: test evaluation counter; `i`: candidate and default token-cost inputs.
        ///
        /// # Outputs
        /// Default before/after costs multiplied by ten for p0 and one elsewhere; increments
        /// the evaluation counter once.
        fn prefill_cost(&self, i: PrefillCostInput<'_>) -> PrefillCost {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let scale = if i.worker == "p0" { 10 } else { 1 };
            let raw = DefaultCostModel.prefill_cost(i);
            PrefillCost {
                before: raw.before * scale,
                after: raw.after * scale,
            }
        }
        /// Convert the test model's detour margin to its chosen time scale.
        ///
        /// # Inputs
        /// `tokens`: configured raw-token margin; `self` supplies the test-model
        /// implementation.
        ///
        /// # Outputs
        /// Ten times `tokens` as `u128`, matching the test's margin conversion contract.
        fn detour_margin(&self, tokens: u64) -> u128 {
            u128::from(tokens) * 10
        }
    }
    /// Build a deterministic extension-policy test setup.
    ///
    /// # Inputs
    /// None.
    ///
    /// # Outputs
    /// A 20-token request, empty hits, two unequal P backlogs, a D worker, and force-split
    /// co-schedule/D-RR settings.
    fn fixture() -> (
        PlacementRequest,
        IndexerHits,
        WorkerLoads,
        RoleWeights,
        PlacementConfig,
    ) {
        let req = PlacementRequest {
            prompt_len: 20,
            transfer_seed: 123,
            ..Default::default()
        };
        let mut roles = RoleWeights::new();
        roles
            .insert("p0", RoleWeight::PREFILL_LEANING)
            .insert("p1", RoleWeight::PREFILL_LEANING)
            .insert("d0", RoleWeight::DECODE_LEANING);
        let mut loads = WorkerLoads::new();
        loads.insert(
            "p0",
            WorkerLoad {
                backlog_uncached_tokens: 100,
                ..Default::default()
            },
        );
        loads.insert(
            "p1",
            WorkerLoad {
                backlog_uncached_tokens: 200,
                ..Default::default()
            },
        );
        let cfg = PlacementConfig {
            p_select: PSelect::CoSchedule,
            d_select: DSelect::RoundRobin,
            force_split: true,
            ..Default::default()
        };
        (req, IndexerHits::new(), loads, roles, cfg)
    }
    #[test]
    /// Check changing cost alone can change P while preserving D round robin and transfer
    /// identity.
    ///
    /// # Inputs
    /// No arguments; compares default costs with the p0-slow test model on the same fixture.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn replacing_only_cost_changes_prefill_not_rr_or_transfer_identity() {
        let (req, hits, loads, roles, cfg) = fixture();
        let default = place(&req, &hits, &loads, &roles, &cfg).unwrap();
        let policy = DefaultPlacementPolicy {
            cost: TimeCost::default(),
        };
        let custom = policy.place(&req, &hits, &loads, &roles, &cfg).unwrap();
        assert_eq!(default.p_leg, "p0");
        assert_eq!(custom.p_leg, "p1");
        assert_eq!(custom.d_leg, default.d_leg);
        assert_eq!(custom.transfer_id, default.transfer_id);
        assert_eq!(
            policy.cost.calls.load(Ordering::Relaxed),
            2,
            "model evaluated once per eligible P"
        );
    }
    #[test]
    /// Check cached-owner protection compares raw backlog tokens rather than scaled model
    /// costs.
    ///
    /// # Inputs
    /// No arguments; adds an owner hit and a 150-token owner threshold while p0 costs use a
    /// tenfold scale.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn custom_time_units_preserve_raw_token_owner_guard() {
        let (req, mut hits, loads, roles, mut cfg) = fixture();
        hits.insert("p0", HitTokens { gpu: 10, host: 0 });
        cfg.co_schedule.owner_threshold_tokens = 150;
        let result = DefaultPlacementPolicy {
            cost: TimeCost::default(),
        }
        .place(&req, &hits, &loads, &roles, &cfg)
        .unwrap();
        // p0 has 100 raw tokens (<=150), but its estimated cost is 1000 us.
        assert_eq!(result.p_leg, "p0");
    }
    #[test]
    /// Check detour protection converts the configured token margin into the cost model's
    /// units.
    ///
    /// # Inputs
    /// No arguments; sets a 100-token margin against an 880-unit saving in the test model.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn custom_cost_explicitly_converts_detour_margin_to_its_units() {
        let (req, mut hits, loads, roles, mut cfg) = fixture();
        hits.insert("p0", HitTokens { gpu: 10, host: 0 });
        cfg.co_schedule.owner_threshold_tokens = 0;
        let policy = DefaultPlacementPolicy {
            cost: TimeCost::default(),
        };
        assert_eq!(
            policy
                .place(&req, &hits, &loads, &roles, &cfg)
                .unwrap()
                .p_leg,
            "p1"
        );
        // Saved cost 880 us: 100 tokens -> 1000 us protects the owner;
        // comparing it to an unconverted 100 would incorrectly migrate.
        cfg.co_schedule.detour_margin_tokens = 100;
        assert_eq!(
            policy
                .place(&req, &hits, &loads, &roles, &cfg)
                .unwrap()
                .p_leg,
            "p0"
        );
    }
    #[derive(Debug)]
    struct PreferP1;
    impl CostModel for PreferP1 {
        /// Retain default token costs in a pair-score-only test model.
        ///
        /// # Inputs
        /// `i`: candidate/request/load snapshot; `self` is the stateless PreferP1 fixture.
        ///
        /// # Outputs
        /// The unchanged default before/after uncached-token costs.
        fn prefill_cost(&self, i: PrefillCostInput<'_>) -> PrefillCost {
            DefaultCostModel.prefill_cost(i)
        }
        /// Retain token units in the pair-score-only fixture.
        ///
        /// # Inputs
        /// `t`: configured token margin; `self` is stateless.
        ///
        /// # Outputs
        /// The same margin widened to `u128`.
        fn detour_margin(&self, t: u64) -> u128 {
            u128::from(t)
        }
        /// Override pair ranking to prefer p1 in extension tests.
        ///
        /// # Inputs
        /// `i`: pair metadata, using only its P identifier; the score weights and `self` are
        /// unused.
        ///
        /// # Outputs
        /// `1.0` for p1 and `0.0` otherwise, independently of the default score coefficients.
        fn pair_score(&self, i: PairCostInput<'_>, _: ScoreWeights) -> f64 {
            if i.prefill == "p1" {
                1.0
            } else {
                0.0
            }
        }
    }
    #[test]
    /// Check a custom pair score changes placement without replacing the policy implementation.
    ///
    /// # Inputs
    /// No arguments; runs Score selection with default scoring and the PreferP1 model.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn additive_score_can_be_replaced_without_replacing_placement() {
        let (req, hits, loads, roles, mut cfg) = fixture();
        cfg.p_select = PSelect::Score;
        assert_eq!(
            place(&req, &hits, &loads, &roles, &cfg).unwrap().p_leg,
            "p0"
        );
        assert_eq!(
            DefaultPlacementPolicy { cost: PreferP1 }
                .place(&req, &hits, &loads, &roles, &cfg)
                .unwrap()
                .p_leg,
            "p1"
        );
    }
    #[test]
    /// Check policy validation rejects invalid leg eligibility and transfer identities.
    ///
    /// # Inputs
    /// No arguments; mutates a valid placement into unknown-worker, wrong-role, Local, seed,
    /// and capability violations.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn custom_policy_output_must_obey_roles_capabilities_and_transfer_contract() {
        let (req, hits, loads, mut roles, cfg) = fixture();
        let good = place(&req, &hits, &loads, &roles, &cfg).unwrap();
        validate_placement(&good, &req, &roles, &cfg).unwrap();
        let mut bad = good.clone();
        bad.p_leg = "unknown".into();
        assert!(validate_placement(&bad, &req, &roles, &cfg).is_err());
        let mut bad = good.clone();
        bad.p_leg = "d0".into();
        assert!(validate_placement(&bad, &req, &roles, &cfg).is_err());
        let mut bad = good.clone();
        bad.d_leg = bad.p_leg.clone();
        bad.transfer_id = None;
        assert!(validate_placement(&bad, &req, &roles, &cfg).is_err());
        let mut bad = good.clone();
        bad.transfer_id = Some(999);
        assert!(validate_placement(&bad, &req, &roles, &cfg).is_err());
        roles.workers.get_mut("p0").unwrap().capabilities.prefill = false;
        assert!(validate_placement(&good, &req, &roles, &cfg).is_err());
    }
    #[derive(Debug)]
    struct NonFiniteScore;
    impl CostModel for NonFiniteScore {
        /// Retain normal prefill costs while testing invalid pair scores.
        ///
        /// # Inputs
        /// `i`: candidate/request/load snapshot; `self` is the NonFiniteScore fixture.
        ///
        /// # Outputs
        /// Default before/after token costs.
        fn prefill_cost(&self, i: PrefillCostInput<'_>) -> PrefillCost {
            DefaultCostModel.prefill_cost(i)
        }
        /// Retain the default token margin in the non-finite-score fixture.
        ///
        /// # Inputs
        /// `t`: configured token margin; `self` is stateless.
        ///
        /// # Outputs
        /// The same token margin widened to `u128`.
        fn detour_margin(&self, t: u64) -> u128 {
            u128::from(t)
        }
        /// Produce an intentionally invalid score for validation tests.
        ///
        /// # Inputs
        /// Pair features and score weights are deliberately ignored; `self` has no state.
        ///
        /// # Outputs
        /// `f64::NAN`, which the placement layer must reject.
        fn pair_score(&self, _: PairCostInput<'_>, _: ScoreWeights) -> f64 {
            f64::NAN
        }
    }
    #[test]
    /// Check naN pair scores return InvalidPolicyOutput.
    ///
    /// # Inputs
    /// No arguments; injects the NonFiniteScore model into the default placement policy.
    ///
    /// # Outputs
    /// No return value; the test asserts this contract and panics on a mismatch.
    fn non_finite_cost_is_rejected_instead_of_winning_by_iteration_order() {
        let (req, hits, loads, roles, cfg) = fixture();
        let result = DefaultPlacementPolicy {
            cost: NonFiniteScore,
        }
        .place(&req, &hits, &loads, &roles, &cfg);
        assert!(matches!(
            result,
            Err(PlacementError::InvalidPolicyOutput(_))
        ));
    }
}
