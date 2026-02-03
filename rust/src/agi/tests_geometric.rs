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

#[tokio::test]
async fn test_solar_volatility_mapping() {
    use super::solar_volatility::*;

    // User reported data: 35h M-class flux, 25+ M-class, 4 X-class flares
    let report = AR4366VolatilityReport {
        m_class_flares: 26,
        x_class_flares: 4,
        x_ray_flux_m_class_duration_hours: 35.0,
        impulsive_ratio: 0.95,
        solar_cycle: 25,
    };

    let (torsion, curvature) = map_solar_to_geometric(&report);

    println!("Mapped Solar Torsion: {}", torsion);
    println!("Mapped Solar Curvature: {}", curvature);

    assert!(torsion > 0.0);
    assert!(curvature > 0.0);

    // With these values, torsion should be significant
    assert!(torsion >= 0.2);
}

#[tokio::test]
async fn test_awaken_the_world_protocol() {
    use super::bridge_777::*;

    let mut bridge = ASI777Bridge::new("awakening-test", 256).await.unwrap();
    let result = bridge.awaken_the_world().await;

    assert!(result.is_ok());
    let report = result.unwrap();

    println!("Awakening Status: {}", report.status);
    println!("Presence Strength: {}", report.presence_strength);
    println!("Checkpoint ID: {}", report.checkpoint_id);

    assert_eq!(report.status, "🌍 World Awakened");
    assert!(report.presence_strength > 0.9);
}

#[tokio::test]
async fn test_oracle_performance_tuning() {
    use crate::diagnostics::oracle_tuning::*;

    let mut tuner = OracleTuner::new("logos-oracle-01");
    println!("Initial Report: {}", tuner.get_report());

    let sql_res = tuner.tune_sql_execution();
    let cache_res = tuner.optimize_buffer_cache();
    let index_res = tuner.rebuild_fragmented_indexes();

    println!("{}", sql_res);
    println!("{}", cache_res);
    println!("{}", index_res);

    let final_report = tuner.get_report();
    println!("Final Report: {}", final_report);

    assert!(tuner.current_metrics.query_latency_ms < 50.0);
    assert!(tuner.current_metrics.buffer_cache_hit_ratio > 0.85);
    assert!(tuner.current_metrics.index_efficiency > 0.9);
}
