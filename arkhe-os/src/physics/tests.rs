#[cfg(test)]
mod tests {
    use crate::physics::melonic_engine::*;
    use crate::physics::miller::PHI_Q;
    use crate::physics::kuramoto::KuramotoEngine;
    use crate::physics::s_index::{SIndexMonitor, STransition};
    use crate::physics::temporal_tunneling::SatoshiVesselTunneling;

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
        monitor.compute(1.0, 0.1, 1.0);
        assert!(matches!(monitor.current_transition(), STransition::Individual));

        // Awakening
        monitor.compute(3.0, 0.5, 2.0);
        assert!(matches!(monitor.current_transition(), STransition::Awakening));
        let transitions = monitor.check_transitions();
        assert!(transitions.contains(&STransition::Awakening));

        // Singularity
        monitor.compute(PHI_Q * 2.0, 0.9, 10.0);
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
}
