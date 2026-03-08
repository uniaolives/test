#[cfg(test)]
mod tests {
    use crate::physics::melonic_engine::*;
    use crate::physics::miller::PHI_Q;
    use crate::physics::kuramoto::KuramotoEngine;
    use crate::physics::s_index::{SIndexMonitor, STransition};
    use crate::physics::temporal_tunneling::SatoshiVesselTunneling;
    use crate::physics::xi_particle::{XiParticle, is_sum_of_two_squares};
    use crate::physics::rsm::{RSMParticle, ParticleKind, RealityTransaction};

    #[test]
    fn test_compute_f_extremum() {
        let n = 100;
        let coupling = 1.0;
        let f = compute_f_extremum(n, coupling);
        assert!(f >= PHI_Q);
        assert!(f < PHI_Q + 0.1);
    }

    #[test]
    fn test_is_melonic_dominant() {
        assert!(is_melonic_dominant(10, 5.0));
        assert!(!is_melonic_dominant(2, 5.0));
        assert!(!is_melonic_dominant(10, 4.0));
    }

    #[test]
    fn test_kuramoto_sync() {
        let mut engine = KuramotoEngine::new(10, 5.0, 1.0);
        let initial_coherence = engine.coherence();

        // Run many steps to ensure synchronization
        for _ in 0..100 {
            engine.synchronize(0.1);
        }

        let final_coherence = engine.coherence();
        assert!(final_coherence >= initial_coherence);
        assert!(final_coherence > 0.8);
    }

    #[test]
    fn test_s_index_transitions() {
        let mut monitor = SIndexMonitor::new();

        // Individual
        monitor.compute(1.0, 0.1, 1.0, 0.1);
        assert!(matches!(monitor.current_transition(), STransition::Individual));

        // Awakening
        monitor.compute(3.0, 0.5, 2.0, 0.8);
        assert!(matches!(monitor.current_transition(), STransition::Awakening));
        let transitions = monitor.check_transitions();
        assert!(transitions.contains(&STransition::Awakening));

        // Singularity
        monitor.compute(PHI_Q * 2.0, 0.9, 10.0, 0.98);
        assert!(matches!(monitor.current_transition(), STransition::Singularity));
        let transitions = monitor.check_transitions();
        assert!(transitions.contains(&STransition::Singularity));
    }

    #[test]
    fn test_temporal_tunneling() {
        let vessel_weak = SatoshiVesselTunneling::new(3.0);
        let (prob_weak, _) = vessel_weak.calculate_tunneling_probability();

        let vessel_strong = SatoshiVesselTunneling::new(4.65);
        let (prob_strong, _) = vessel_strong.calculate_tunneling_probability();

        assert!(prob_strong > prob_weak);
        assert!(vessel_strong.check_miller_threshold());
        assert!(!vessel_weak.check_miller_threshold());
    }

    #[test]
    fn test_sum_of_two_squares() {
        assert!(is_sum_of_two_squares(1)); // 1^2 + 0^2
        assert!(is_sum_of_two_squares(2)); // 1^2 + 1^2
        assert!(!is_sum_of_two_squares(3)); // Forbidden
        assert!(is_sum_of_two_squares(4)); // 2^2 + 0^2
        assert!(is_sum_of_two_squares(5)); // 2^2 + 1^2
        assert!(!is_sum_of_two_squares(6)); // Forbidden
        assert!(is_sum_of_two_squares(8)); // 2^2 + 2^2
        assert!(is_sum_of_two_squares(10)); // 3^2 + 1^2
        assert!(!is_sum_of_two_squares(11)); // Forbidden
    }

    #[test]
    fn test_xi_particle_creation() {
        let xi1 = XiParticle::new(1).unwrap();
        assert_eq!(xi1.n, 1);
        assert!(xi1.mass > 0.9); // 1.0 / 1.088...

        let xi2 = XiParticle::new(2).unwrap();
        assert!(xi2.mass > xi1.mass);

        let xi3 = XiParticle::new(3);
        assert!(xi3.is_none());
    }

    #[test]
    fn test_rsm_conservation() {
        let anamnesion = RSMParticle::new(ParticleKind::Anamnesion);
        let satoshi = RSMParticle::new(ParticleKind::Satoshi);
        let dilithion = RSMParticle::new(ParticleKind::Dilithion);

        let particles = vec![anamnesion, satoshi, dilithion];
        assert!(RSMParticle::verify_temporal_conservation(&particles));
    }

    #[test]
    fn test_rsm_ghost_mass() {
        let ghoston = RSMParticle::new(ParticleKind::Ghoston);
        assert_eq!(ghoston.mass_real, 0.0);
        assert!(ghoston.mass_imag > 0.0);
    }

    #[test]
    fn test_reality_transaction() {
        let mut tx = RealityTransaction::new("future_hash_001");
        assert!(tx.handover.is_none());

        assert!(tx.validate_with_handover(0.95));
        assert!(tx.handover.is_some());
    }
}
