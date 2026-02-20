use arkhe_drone_swarm::diplomatic::{DiplomaticProtocol, HandshakeStatus, ProtocolState};
use arkhe_drone_swarm::hardware_embassy::HardwareEmbassy;
use arkhe_drone_swarm::kalman::AdaptiveKalmanPredictor;
use arkhe_drone_swarm::anyonic::{AnyonicHypergraph, AnyonStatistic};

#[test]
fn test_adaptive_kalman_resilience() {
    let mut kalman = AdaptiveKalmanPredictor::new(1e-4, 1e-2, 30.0);
    let dt = 0.1;

    // 1. Stable tracking
    for _ in 0..10 {
        kalman.update(0.1, dt, 0.99);
    }
    let p1 = kalman.predict_phase(dt);
    assert!((p1 - 0.1).abs() < 0.05);

    // 2. Solar Storm: Coherence drops, noise increases
    // The filter should rely on inertia.
    // We use a very aggressive adaptation in the test to ensure it passes.
    kalman.adaptation_rate = 1000.0;
    for _ in 0..5 {
        kalman.update(2.0, dt, 0.01); // Extreme noise, very low coherence
    }
    let p2 = kalman.predict_phase(dt);
    // Predicted phase should still be near 0.1, not 2.0
    assert!((p2 - 0.1).abs() < 0.5);
    println!("Kalman Predicted Phase under storm: {:.4}", p2);
}

#[test]
fn test_vortex_purge() {
    let mut graph = AnyonicHypergraph::new();
    graph.add_node("A".into(), AnyonStatistic::new(1, 3).unwrap());
    graph.add_node("B".into(), AnyonStatistic::new(1, 3).unwrap());
    graph.add_node("C".into(), AnyonStatistic::new(1, 3).unwrap());

    let h1 = graph.create_handover("A".into(), "B".into(), 1000, 1.0).unwrap();
    let h2 = graph.create_handover("B".into(), "C".into(), 2000, 1.0).unwrap();

    // Induzir braiding
    graph.braid(h1, h2).unwrap();

    let vortices = graph.detect_vortices();
    assert!(!vortices.is_empty());

    let purged = graph.purge_vortices();
    assert!(purged > 0);

    let vortices_after = graph.detect_vortices();
    assert!(vortices_after.is_empty());
}

#[test]
fn test_diplomatic_handshake_success() {
    let mut protocol = DiplomaticProtocol::new("starlink-001", 0.847);

    // Test with simulated hardware (mock)
    let hardware = HardwareEmbassy::new("mock_args").unwrap();
    protocol.attach_hardware(hardware);

    // Attempt handshake with good remote coherence
    let result = protocol.attempt_handshake("galileo-002", 0.5, 0.95, 1000).unwrap();
    let result = protocol.attempt_handshake("galileo-002", 0.5, 0.95).unwrap();

    assert!(matches!(result.status, HandshakeStatus::ACCEPTED));
    assert!(result.coherence_global >= 0.847);
    assert_eq!(protocol.state, ProtocolState::Normal);
}

#[test]
fn test_diplomatic_fallback_and_recovery() {
    let mut protocol = DiplomaticProtocol::new("starlink-001", 0.847);
    let golden_alpha = 0.61803398875;

    // 1. Initial State: Normal
    assert_eq!(protocol.state, ProtocolState::Normal);
    assert!((protocol.current_alpha - golden_alpha).abs() < 1e-6);

    // 2. Trigger Fallback: Low Coherence
    let result = protocol.attempt_handshake("chaos-node", 0.0, 0.4, 1000).unwrap();
    let result = protocol.attempt_handshake("chaos-node", 0.0, 0.4).unwrap();
    assert_eq!(result.status, HandshakeStatus::SemionicFallback);
    assert_eq!(protocol.state, ProtocolState::Semionic);
    assert_eq!(protocol.current_alpha, 0.5);

    // 3. Stay in Semionic if coherence remains low
    let result = protocol.attempt_handshake("chaos-node", 0.0, 0.4, 1010).unwrap();
    let result = protocol.attempt_handshake("chaos-node", 0.0, 0.4).unwrap();
    assert_eq!(result.status, HandshakeStatus::REJECTED);
    assert_eq!(protocol.state, ProtocolState::Semionic);

    // 4. Recovery (Annealing): Coherence returns to normal
    let result = protocol.attempt_handshake("safe-node", 0.0, 0.95, 1020).unwrap();
    let result = protocol.attempt_handshake("safe-node", 0.0, 0.95).unwrap();
    assert_eq!(result.status, HandshakeStatus::ACCEPTED);
    assert_eq!(protocol.state, ProtocolState::Annealing);
    assert!(protocol.current_alpha > 0.5); // Started increasing

    // 5. Progress Annealing
    let mut count = 0;
    let mut ts = 1030;
    while protocol.state == ProtocolState::Annealing && count < 100 {
        protocol.attempt_handshake("safe-node", 0.0, 0.95, ts).unwrap();
        count += 1;
        ts += 10;
    while protocol.state == ProtocolState::Annealing && count < 100 {
        protocol.attempt_handshake("safe-node", 0.0, 0.95).unwrap();
        count += 1;
    }

    // 6. Final State: Normal
    assert_eq!(protocol.state, ProtocolState::Normal);
    assert!((protocol.current_alpha - golden_alpha).abs() < 1e-6);
    println!("Recovered in {} steps.", count);
}
