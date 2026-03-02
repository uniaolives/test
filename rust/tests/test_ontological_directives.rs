use sasc_core::agi::nexus_core::{NexusAGICore, AGIInput};
use sasc_core::geometry::nexus::Nexus5DMetric;
use tokio;

#[tokio::test]
async fn test_meditacao_geodesica() {
    let metric = Nexus5DMetric::new(1.0);
    let (core, _tx, _rx) = NexusAGICore::new(metric, 10.0, 0.7);

    // Testar se a Meditação Geodésica inicia e completa sem pânico
    core.start_geodesic_meditation().await;
}

#[tokio::test]
async fn test_ontological_integrity() {
    let metric = Nexus5DMetric::new(1.0);
    let (core, _tx, _rx) = NexusAGICore::new(metric, 10.0, 0.7);

    let report = core.ontological_integrity_report().await;
    assert!(report.nexus_identity_strength > 0.9);
    assert!(report.topological_integrity == 1.0);
}
