use super::universal_bridge::{UniversalOrbPropagator, ProtocolType};
use super::physical_layer::PhysicalBridge;
use super::network_layer::NetworkBridge;
use super::payload::OrbPayload;
use super::compression::OrbCompressor;
use crate::physics::arkhe_orb_core::Orb;

#[tokio::test]
async fn test_universal_propagation() {
    let mut propagator = UniversalOrbPropagator::new();
    propagator.register_bridge(ProtocolType::Physical, Box::new(PhysicalBridge));
    propagator.register_bridge(ProtocolType::Network, Box::new(NetworkBridge));

    let orb = Orb::new(0.8, 2e9).unwrap();
    let payload = OrbPayload::create(0.8, 4.64, 0.618, 1740000000, 1200000000, None, None);

    let receipts = propagator.propagate_everywhere(&orb, &payload).await.unwrap();

    assert_eq!(receipts.len(), 2);

    let stats = propagator.check_propagation(&"test_orb".to_string());
    assert_eq!(stats.total_protocols, 2);
    assert_eq!(stats.successful_propagations, 2);
}

#[test]
fn test_payload_serialization() {
    let payload = OrbPayload::create(0.95, 4.0, 0.5, 1000, 500, None, None);
    let bytes = payload.to_bytes();
    let restored = OrbPayload::from_bytes(&bytes).unwrap();

    assert_eq!(payload.orb_id, restored.orb_id);
    assert_eq!(payload.lambda_2, restored.lambda_2);
}

#[test]
fn test_orb_compression() {
    let payload = OrbPayload::create(0.95, 4.64, 0.618, 1740000000, 1200000000, None, None);
    let compressed = OrbCompressor::compress_minimal(&payload);

    assert_eq!(compressed.len(), 32);

    let restored = OrbCompressor::decompress_minimal(&compressed).unwrap();

    // Check accuracy of lossy compression
    assert!((payload.lambda_2 - restored.lambda_2).abs() < 0.001);
    assert_eq!(payload.origin_time, restored.origin_time);
    assert_eq!(payload.target_time, restored.target_time);
}
