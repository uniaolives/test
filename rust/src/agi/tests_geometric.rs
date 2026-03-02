// rust/src/agi/tests_geometric.rs

use super::geometric_core::*;
use super::cge_constraints::*;
use super::persistent_geometric_agent::*;
use nalgebra::DVector;

#[tokio::test]
async fn test_geodesic_distance() {
    let space = GeometricSpace::new(3);
    let p = DVector::from_vec(vec![0.0, 0.0, 0.0]);
    let q = DVector::from_vec(vec![1.0, 0.0, 0.0]);

    let dist = space.geodesic_distance(&p, &q);
    assert!((dist - 1.0).abs() < 1e-6);
}

#[tokio::test]
async fn test_cge_validation() {
    let engine = CGEConstraintEngine::new();
    let state = GeometricState {
        metric_tensor: nalgebra::DMatrix::identity(3, 3),
        curvature: nalgebra::DMatrix::zeros(3, 3),
        volume: 500.0,
        torsion: 0.1,
    };

    assert!(engine.validate(&state).is_ok());

    let invalid_state = GeometricState {
        metric_tensor: nalgebra::DMatrix::identity(3, 3),
        curvature: nalgebra::DMatrix::zeros(3, 3),
        volume: 1500.0,
        torsion: 0.1,
    };

    assert!(engine.validate(&invalid_state).is_err());
}

#[tokio::test]
async fn test_persistent_agent_checkpoint() {
    let mut agent = PersistentGeometricAgent::new("test-agent", 3).await.unwrap();
    let result = agent.checkpoint().await;

    assert!(result.is_ok());
    let tx_id = result.unwrap();
    assert!(!tx_id.is_empty());
}
