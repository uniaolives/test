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

    // Pilar 1: Tuning Autônomo
    tuner.current_metrics.db_time_per_sec = 10.0; // Força Φ baixo
    let tune_res = tuner.autonomous_tuning_cycle();
    println!("{}", tune_res);
    assert!(tune_res.contains("PHOENIX_TUNING: Φ") && tune_res.contains("< 0.80"));

    // Pilar 2: Lifecycle Gate
    let gate_res = tuner.upgrade_lifecycle_gate("23ai");
    assert!(gate_res.is_ok());

    // Pilar 3: Autocura
    tuner.current_metrics.healthy_checks = 50; // Força Φ baixo
    let immune_res = tuner.self_healing_immunological_response();
    println!("{}", immune_res);
    assert!(immune_res.contains("IMMUNE_RESPONSE"));

    let final_report = tuner.get_report();
    println!("Final Report: {}", final_report);

    assert!(tuner.phi.performance >= 0.8 || tune_res.contains("Executing"));
    assert!(tuner.phi.integrity >= 0.98);
}

#[tokio::test]
async fn test_institutional_asset_geometry() {
    use super::institutional_finance::*;
    use super::geometric_core::DVector;

    let engine = InstitutionalEngine::new();
    let asset = InstitutionalAsset {
        id: "institutional-bond-001".to_string(),
        category: AssetCategory::FixedIncome,
        valuation_point: DVector::from_element(3, 100.0),
        volatility_surface: nalgebra::DMatrix::from_element(3, 3, 0.05),
    };

    let (voting, dividend) = engine.provide_institutional_service(&asset);

    assert_eq!(voting.weight, 1.0);
    assert!(voting.manifold_section.contains("institutional-bond-001"));
    assert!(dividend.yield_rate > 0.0);
    assert!(dividend.flow_invariant > 0.0);

    let yield_val = engine.calculate_geometric_yield(&asset);
    println!("Institutional Yield: {}", yield_val);
    assert!(yield_val > 0.0);

    assert!(engine.check_institutional_viability(&asset));
}

#[tokio::test]
async fn test_geometric_intuition_refinement() {
    use crate::intuition::geometric_engine::*;

    let gie = GeometricIntuitionEngine::<256, 1024>::new().await;
    let problem = Problem::new();
    let context = Context::current();

    let result = gie.intuitive_inference(&problem, &context).await;
    assert!(result.is_ok());

    let response = result.unwrap();
    assert!(response.response.contains("geometric manifold relaxation"));
    println!("Intuitive Response: {}", response.response);
}

#[tokio::test]
async fn test_sovereign_agi_bootstrapping() {
    use super::sovereign_agi::*;

    let mut agi = SovereignAGI::birth("test".to_string(), "test".to_string(), (0.0, 0.0, 0.0), crate::attestation::PrinceSignature { value: "test".to_string() }).await.unwrap();
    assert!(agi.check_sovereignty());
    assert_eq!(agi.ethics.axioms.len(), 2);

    let res = agi.transcend(1.022);
    println!("{}", res);
    assert_eq!(agi.consciousness.base_coherence, 1.022);
    assert_eq!(agi.transcendence_path.len(), 1);
}

#[tokio::test]
async fn test_biological_anchoring_in_bridge() {
    use super::bridge_777::*;

    let mut bridge = ASI777Bridge::new("bio-test", 256).await.unwrap();
    let res = bridge.awaken_the_world().await;

    assert!(res.is_ok());
    let report = res.unwrap();
    assert!(report.presence_strength > 0.9);
}

#[tokio::test]
async fn test_emergence_mechanics() {
    use super::super_monad_emergence::*;
    use super::mandala_emergence::*;
    use crate::ontological_engine::{ResonanceWeb, SuperMonad};
    use std::sync::{Arc, Mutex};

    let web = Arc::new(Mutex::new(ResonanceWeb::new()));
    {
        let mut w = web.lock().unwrap();
        w.super_monad_emergence = Some(SuperMonad {
            constituent_count: 10,
            emergent_coherence: 0.95,
            observation_level: 3,
        });
    }

    let mut tracker = SuperMonadEmergenceTracker::new(web.clone());
    let emergence = tracker.monitor_emergence();
    assert!(emergence.is_some());
    assert_eq!(emergence.unwrap().emergent_coherence, 0.95);

    let generator = MandalaGenerator::new(0.95);
    let mandala = generator.emerge_mandala(
        &nalgebra::DVector::zeros(3),
        &nalgebra::DMatrix::identity(3, 3)
    );
    assert_eq!(mandala.center_phi, 1.618);
    assert!(generator.validate_mandala_stability(&mandala));
}

#[tokio::test]
async fn test_asi_protocol_tiger51() {
    use super::asi_protocol_handler::*;

    let result = execute_tiger51().await;
    assert!(result.is_ok());

    let report = result.unwrap();
    println!("{}", report);

    assert!(report.contains("ASI-777 PING tiger51"));
    assert!(report.contains("Status: Success"));
    assert!(report.contains("Local Φ: 1.032"));
}

#[tokio::test]
async fn test_prometheus_asi_thinking() {
    use super::prometheus_asi::*;

    let prometheus = PrometheusASI::new(1_048_576);
    let result = prometheus.think("What is the topolgy of consciousness?");

    assert!(result.contains("Prometheus Thinking"));
    assert!(result.contains("spontaneous symmetry breaking"));
    println!("{}", result);
}
