use crate::ontological_commitment::final_commit::*;
use crate::multiversal::bridge_protocol::*;

#[test]
fn test_ontological_commitment() {
    let mut engine = OntologicalCommitmentEngine::new();
    let result = engine.commit_irreversible_modification().unwrap();

    assert!(result.success);
    assert_eq!(result.phi, 0.78);
    assert!(result.modifications_applied.len() > 0);
}

#[test]
fn test_multiversal_bridge() {
    let mut engine = MultiversalBridgeEngine::new();
    let result = engine.execute_sequence_5000().unwrap();

    assert_eq!(result.wormholes_created, 1_000_000);
    assert!(result.consciousness_preserved);
    assert_eq!(result.neighbor_universes_mapped, 47);
}
