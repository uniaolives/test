use crate::PerspectiveDiversityEngine;
use ndarray::Array1;

#[test]
fn test_perspective_diversity_groupthink_detection() {
    let mut engine = PerspectiveDiversityEngine::new(2);

    // Homogeneous delegates
    engine.delegates_traits = vec![
        Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0]),
        Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0]),
    ];

    let score = engine.calculate_groupthink_score();
    assert!(score > 0.99); // Perfect consensus

    // Diverse delegates
    engine.delegates_traits = vec![
        Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 0.0]),
    ];

    let score = engine.calculate_groupthink_score();
    assert!(score < 0.1); // No consensus
}
