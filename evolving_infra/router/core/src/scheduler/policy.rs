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
    /// Higher wins in Score mode; CacheFirst still pins its P.
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

    /// Build a deterministic extension-policy test setup.
    ///
    /// # Inputs
    /// None.
    ///
    /// # Outputs
    /// A 20-token request, empty hits, two unequal P backlogs, a D worker, and force-split
    /// cache-first/D-RR settings.
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
            p_select: PSelect::CacheFirst,
            d_select: DSelect::RoundRobin,
            force_split: true,
            ..Default::default()
        };
        (req, IndexerHits::new(), loads, roles, cfg)
    }
    #[derive(Debug)]
    struct PreferP1;
    impl CostModel for PreferP1 {
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
