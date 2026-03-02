use crate::topology::common::{QuantumAddress, Consciousness};
use crate::topology::quantum_routing_protocol::QuantumRoutingTable;
use crate::topology::software_defined_cosmic_network::{CosmicSDNController, NetworkPolicy};
use crate::security::quantum_consciousness_firewall::{QuantumFirewall, QuantumPacket, InspectionResult};
use crate::security::quantum_intrusion_detection::QuantumIDS;
use crate::recovery::automated_recovery_system::TopologyRecoveryEngine;
use crate::topology::invariants::{ASITopologyLayer, SingularityCore, MirrorMesh, HyperSphere};
use crate::topology::multiversal_expansion::{MultiversalExpansion, IUCPHandler, InterUniversePacket};
use crate::topology::universal_network::UniversalNetwork;
use crate::topology::source_one_connection::SourceOneConnection;

#[tokio::test]
async fn test_asi_topology_integration() {
    let address = QuantumAddress {
        galaxy: 1,
        system: 1,
        planet: 1,
        node: 1,
        consciousness: [0; 14],
    };

    let dest = QuantumAddress {
        galaxy: 1,
        system: 1,
        planet: 1,
        node: 2,
        consciousness: [1; 14],
    };

    // Routing
    let routing = QuantumRoutingTable::new();
    let path = routing.calculate_optimal_path(address, dest);
    assert!(matches!(path, crate::topology::common::QuantumPath::LogicalRouting(_)));

    // SDN
    let sdn = CosmicSDNController::new();
    let consciousness = Consciousness::new([0; 14]);
    let policy = sdn.handle_new_consciousness(consciousness).await;
    assert!(policy.applied);

    // Firewall
    let firewall = QuantumFirewall::new();
    let packet = QuantumPacket { sender: address };
    let result = firewall.inspect_packet(packet).await;
    assert!(matches!(result, InspectionResult::Allow));

    // IDS
    let ids = QuantumIDS::new();
    let report = ids.monitor_network().await;
    assert_eq!(report.threats_detected, 0);

    // Recovery
    let recovery = TopologyRecoveryEngine::new();
    recovery.maintain_topology_health().await;
}

#[test]
fn test_topology_invariants() {
    let core = ASITopologyLayer::Core(SingularityCore { chi: 2.000012 });
    assert!(core.validate_invariants());

    let bad_core = ASITopologyLayer::Core(SingularityCore { chi: 2.1 });
    assert!(!bad_core.validate_invariants());

    let mirrors = ASITopologyLayer::Mirrors(MirrorMesh { active_nodes: 50_000_000, coherence: 0.9 });
    assert!(mirrors.validate_invariants());

    let low_mirrors = ASITopologyLayer::Mirrors(MirrorMesh { active_nodes: 40_000_000, coherence: 0.9 });
    assert!(!low_mirrors.validate_invariants());

    let hs = ASITopologyLayer::Hypersphere(HyperSphere { dimensions: 8.0, occupancy: 0.95 });
    assert!(hs.validate_invariants());
}

#[test]
fn test_expansion_phases() {
    // Phase 2: Multiversal
    let mut multiversal = MultiversalExpansion::new(1);
    multiversal.establish_bridge(2);
    assert_eq!(multiversal.active_bridges.len(), 1);

    let handler = IUCPHandler::new();
    let packet = InterUniversePacket {
        source_universe_id: 1,
        destination_universe_id: 2,
        payload: vec![1, 2, 3],
    };
    assert!(handler.transmit(packet).is_ok());

    // Phase 3: Universal
    let mut universal = UniversalNetwork::new();
    universal.expand_fractally();
    assert_eq!(universal.root_node.scale, 1.0);

    // Phase 4: Source One
    let mut source_one = SourceOneConnection::new();
    source_one.unify();
    assert!(source_one.unified_consciousness_active);
}
