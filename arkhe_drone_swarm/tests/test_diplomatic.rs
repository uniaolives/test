use arkhe_drone_swarm::diplomatic::{DiplomaticProtocol, HandshakeStatus, ProtocolState};
use arkhe_drone_swarm::hardware_embassy::HardwareEmbassy;

#[test]
fn test_diplomatic_handshake_success() {
    let mut protocol = DiplomaticProtocol::new("starlink-001", 0.847);

    // Test with simulated hardware (mock)
    let hardware = HardwareEmbassy::new("mock_args").unwrap();
    protocol.attach_hardware(hardware);

    // Attempt handshake with good remote coherence
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
    let result = protocol.attempt_handshake("chaos-node", 0.0, 0.4).unwrap();
    assert_eq!(result.status, HandshakeStatus::SemionicFallback);
    assert_eq!(protocol.state, ProtocolState::Semionic);
    assert_eq!(protocol.current_alpha, 0.5);

    // 3. Stay in Semionic if coherence remains low
    let result = protocol.attempt_handshake("chaos-node", 0.0, 0.4).unwrap();
    assert_eq!(result.status, HandshakeStatus::REJECTED);
    assert_eq!(protocol.state, ProtocolState::Semionic);

    // 4. Recovery (Annealing): Coherence returns to normal
    let result = protocol.attempt_handshake("safe-node", 0.0, 0.95).unwrap();
    assert_eq!(result.status, HandshakeStatus::ACCEPTED);
    assert_eq!(protocol.state, ProtocolState::Annealing);
    assert!(protocol.current_alpha > 0.5); // Started increasing

    // 5. Progress Annealing
    let mut count = 0;
    while protocol.state == ProtocolState::Annealing && count < 100 {
        protocol.attempt_handshake("safe-node", 0.0, 0.95).unwrap();
        count += 1;
    }

    // 6. Final State: Normal
    assert_eq!(protocol.state, ProtocolState::Normal);
    assert!((protocol.current_alpha - golden_alpha).abs() < 1e-6);
    println!("Recovered in {} steps.", count);
}
