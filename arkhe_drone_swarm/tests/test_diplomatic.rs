use arkhe_drone_swarm::diplomatic::{DiplomaticProtocol, HandshakeStatus};
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
}

#[test]
fn test_diplomatic_handshake_rejection() {
    let mut protocol = DiplomaticProtocol::new("starlink-001", 0.847);

    // Attempt handshake with poor remote coherence
    let result = protocol.attempt_handshake("galileo-002", 0.5, 0.5).unwrap();

    assert!(matches!(result.status, HandshakeStatus::REJECTED));
    assert!(result.coherence_global < 0.847);
}
