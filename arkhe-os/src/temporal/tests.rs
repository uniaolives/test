use super::mobius_chain::MobiusBlock;
use super::mobius_engine::MobiusEngine;
use arkhe_db::handover::Handover;
use std::f64::consts::PI;

#[test]
fn test_mobius_equivalence() {
    let data = vec![1, 2, 3];
    let block_past = MobiusBlock::create_superposed(data.clone(), 0.0);
    let block_future = MobiusBlock::create_superposed(data, 1.0);

    assert!(MobiusBlock::are_mobius_equivalent(&block_past, &block_future));
}

#[test]
fn test_temporal_distance() {
    let engine = MobiusEngine::with_retrocausality(0.8);
    let d = engine.temporal_distance(0.0, PI);
    // Direct is PI, Wrapped is PI.
    // With epsilon 0.8: min(PI * 0.2, PI * 0.8) = PI * 0.2
    assert!(d < PI * 0.5);
}

#[test]
fn test_propagation() {
    let engine = MobiusEngine::with_retrocausality(0.5);
    let handover = Handover {
        id: 1,
        timestamp: 1000,
        source_epoch: 1,
        target_epoch: 2,
        coherence: 1.0,
        phi_q_before: 0.5,
        phi_q_after: 0.6,
        quantum_interest: 10.0,
        payload_hash: "hash".to_string(),
    };

    let results = engine.propagate(0.0, &handover, 0.1);
    assert_eq!(results.len(), 2);

    let (t_fwd, h_fwd) = &results[0];
    let (t_back, h_back) = &results[1];

    assert!(*t_fwd > 0.0);
    assert!(h_back.coherence < 1.0);
}
