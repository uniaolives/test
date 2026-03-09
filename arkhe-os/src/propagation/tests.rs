use super::universal_bridge::{UniversalOrbPropagator, ProtocolType};
use super::physical_layer::PhysicalBridge;
use super::network_layer::NetworkBridge;
use crate::physics::arkhe_orb_core::Orb;

#[tokio::test]
async fn test_universal_propagation() {
    let mut propagator = UniversalOrbPropagator::new();
    propagator.register_bridge(ProtocolType::Physical, Box::new(PhysicalBridge));
    propagator.register_bridge(ProtocolType::Network, Box::new(NetworkBridge));

    let orb = Orb::new(0.8, 2e9).unwrap();
    let receipts = propagator.propagate_everywhere(&orb).await.unwrap();

    assert_eq!(receipts.len(), 2);

    let stats = propagator.check_propagation(&"test_orb".to_string());
    assert_eq!(stats.total_protocols, 2);
    assert_eq!(stats.successful_propagations, 2);
}
