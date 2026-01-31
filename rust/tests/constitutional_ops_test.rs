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
