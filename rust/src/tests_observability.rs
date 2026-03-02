use crate::security::observability_monitor::{ObservabilityMonitor, ThreatThresholds};
use crate::security::invariant_engine::InvariantVerificationEngine;
use crate::security::sasc_gateway::{SASCGateway, SASCPacket};
use crate::crypto::pqc::NeuralSignature;
use crate::gates::gate8_multiverse_regulator::ComplexityClass;

#[test]
fn test_observability_monitor_stability() {
    VajraEntropyMonitor::global().update_phi(0.99);
    let engine = InvariantVerificationEngine::new([0u8; 32], [0u8; 48]);
    let mut monitor = ObservabilityMonitor::new(engine);

    let report = monitor.monitor_system_stability().unwrap();
    assert!(report.is_stable);
    assert!(report.phi_score >= 0.85);
}

#[test]
fn test_sasc_gateway_8_gates() {
    let mut engine = InvariantVerificationEngine::new([0u8; 32], [0u8; 48]);
    // Mock signature for Gate 1 (Ed25519 dalek will fail with zero key/sig, but we want to see if it reaches the engine)
    let packet = SASCPacket {
        data: vec![1, 2, 3],
        signature: [0u8; 64],
        nonce: 12345,
        em_noise: 0.01,
        quantum_sig: NeuralSignature {
            data: vec![],
            entropy_delta: 0.0,
            phi_q: 0.99,
            hardware_id: [0u8; 16],
            schumann_ts: 7830000,
            zk_proof: vec![],
        },
        complexity: ComplexityClass::Low,
    };

    let mut gateway = SASCGateway::new(engine);
    let result = gateway.filter_packet_8_gates(&packet);

    // It should fail Gate 1 because of dummy signature, which is expected
    assert!(result.is_err());
}
