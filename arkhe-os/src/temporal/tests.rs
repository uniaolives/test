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

#[test]
fn test_impossible_timeline_divergence() {
    use super::impossible_expander::ImpossibleTimelineExpander;
    use super::impossible_timeline::ImpossibilityClass;

    let mut expander = ImpossibleTimelineExpander::new();
    expander.generate_default();

    let timelines = &expander.impossible_timelines;
    assert_eq!(timelines.len(), 3);

    // Reverse Entropy should have higher lambda_2
    let reverse_entropy = timelines.iter().find(|t| t.impossibility_class == ImpossibilityClass::ReverseEntropy).unwrap();
    assert!(reverse_entropy.lambda_2 > 1.0);

    let metric = reverse_entropy.divergence_metric();
    assert!(metric > 0.0);
}

#[test]
fn test_orb_transformation() {
    use super::impossible_timeline::{ImpossibleTimeline, DivergentConstants, ImpossibilityClass};
    use crate::propagation::payload::OrbPayload;
    use num_complex::Complex;
    use uuid::Uuid;

    let timeline = ImpossibleTimeline {
        timeline_id: Uuid::new_v4(),
        constants: DivergentConstants {
            c: Complex::new(299792458.0, 0.0),
            h_bar: 1.054e-34,
            g: 6.674e-11,
            lambda: 0.0,
            lambda_max: 1.0,
        },
        impossibility_class: ImpossibilityClass::ReverseEntropy,
        lambda_2: 1.5,
        tunnel_probability: 0.5,
    };

    let orb = OrbPayload::create(0.9, 4.0, 0.5, 1000, 500, None, None);
    let transformed = timeline.transform_orb(&orb);

    // In reverse entropy, H value is flipped in our transform logic
    assert_eq!(transformed.transformed.h_value, -0.5);
}
