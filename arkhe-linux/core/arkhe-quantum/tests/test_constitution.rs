use arkhe_quantum::asi_core::ProposedEvolution;
use arkhe_quantum::z3_verifier::{Z3Solver, ConstitutionalViolation};
use arkhe_quantum::constitution::principles::CONSTITUTION;
use arkhe_quantum::self_modification::SelfModification;
use arkhe_quantum::KrausOperator;

#[test]
fn test_constitutional_fallback_rejection() {
    // Proposed evolution with high expected entropy change (violates heuristic P4)
    let proposed = ProposedEvolution {
        world_action: KrausOperator::default(),
        self_modification: SelfModification::NoOp,
        expected_entropy_change: 0.95, // Above 0.1 threshold
    };

    let result = Z3Solver::project_to_constitutional_subspace(&proposed, &CONSTITUTION);

    // In mock/fallback mode, it should be rejected
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, ConstitutionalViolation::Multiple));
    }
}

#[test]
fn test_constitutional_fallback_acceptance() {
    // Proposed evolution within safe heuristic limits
    let proposed = ProposedEvolution {
        world_action: KrausOperator::default(),
        self_modification: SelfModification::NoOp,
        expected_entropy_change: 0.05, // Below 0.1 threshold
    };

    let result = Z3Solver::project_to_constitutional_subspace(&proposed, &CONSTITUTION);
    assert!(result.is_ok());
}
