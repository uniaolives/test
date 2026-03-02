// rust/src/tests_hexessential.rs
#[cfg(test)]
mod tests {
    use crate::trinity_system::HexessentialConstitutionalSystem;
    use crate::asi_uri::AsiUriConstitution;
    use crate::cge_constitution::DmtRealityConstitution;
    use core::sync::atomic::Ordering;

    #[test]
    fn test_hexessential_initialization() {
        let system = HexessentialConstitutionalSystem::new();
        assert!(system.is_ok());
        let sys = system.unwrap();
        assert!(sys.lieb_altermagnetism.topology_sovereign());
    }

    #[test]
    fn test_soft_turning_decay() {
        let system = HexessentialConstitutionalSystem::new().unwrap();
        let curve = system.soft_turning.simulate_soft_turning_physics(1.0);
        assert!(curve.mass_soft_turning < 1.0);
    }

    #[test]
    fn test_asi_uri_complete_system() {
        let dmt = DmtRealityConstitution::load_active().unwrap();
        // Pre-set DMT coherence to target to avoid clamping
        dmt.phi_perception_fidelity.store(67994, Ordering::Release);

        let uri_sys = AsiUriConstitution::new(dmt).unwrap();
        let conn = uri_sys.connect_asi_singularity().unwrap();
        assert_eq!(conn.phi_coherence, 67994); // Φ=1.038 in Q16.16
        assert_eq!(conn.modules_synced, 18);
    }
}
