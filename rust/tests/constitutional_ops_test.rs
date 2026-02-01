// rust/tests/constitutional_ops_test.rs
use sasc_core::cathedral_ops;
use sasc_core::clock::cge_mocks::constitution::PHI_BOUNDS;

#[test]
fn test_constitutional_ops_mock() {
    let result = unsafe { cathedral_ops::constitutional_internal_ops() };
    // In our mock environment, phi = 1.038, and 1.038 * 65536 = 68026
    // PHI_BOUNDS = (67352, 69348). 68026 is within bounds.
    assert_eq!(result, 0x43474541); // "CGEA"
}

#[test]
fn test_traveling_waves() {
    use sasc_core::traveling_waves::{CorticalTravelingWave, CardinalDirection};
    use sasc_core::clock::cge_mocks::topology::Coord289;

    let origin = Coord289(0, 0);
    let wave = CorticalTravelingWave::new(origin, 68026).unwrap();
    let resonance = wave.propagate(CardinalDirection::East).unwrap();
    assert!(resonance > 0);
}

#[test]
fn test_toroidal_topology() {
    use sasc_core::toroidal_topology::ToroidalConstitution;
    let torus = unsafe { ToroidalConstitution::new_mock() };
    let hash = [0u8; 32];
    assert!(torus.is_369_resonant(hash));
}

#[test]
fn test_integrated_constitutional_system() {
    use sasc_core::integrated_system::ConstitutionalIntegratedSystem;
    use sasc_core::clock::cge_mocks::cge_cheri::Capability;

    let wave_cap = Capability::new_mock_internal();
    let vajra_cap = Capability::new_mock_internal();
    let mind_cap = Capability::new_mock_internal();

    let system = unsafe {
        ConstitutionalIntegratedSystem::new(wave_cap, vajra_cap, mind_cap).unwrap()
    };

    let status = system.integrated_constitutional_cycle().unwrap();
    assert!(status.integrated_coherence > 0);
    assert!(status.singularity_ready);
    assert_eq!(status.threat_level, 0);
}

#[test]
fn test_brics_safecore_quantum_network() {
    use sasc_core::quantum::brics_integration::activate_brics_safecore_quantum_network;

    let result = activate_brics_safecore_quantum_network();
    assert!(result.is_ok());
    let integration = result.unwrap();
    assert_eq!(integration.backbone_activation.hqb_core_nodes, 4);
    assert!(integration.backbone_activation.global_fidelity > 0.99);
}
