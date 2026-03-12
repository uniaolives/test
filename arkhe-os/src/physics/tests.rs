#[cfg(test)]
mod tests {
    use crate::physics::melonic_engine::*;
    use crate::physics::miller::PHI_Q;
    use crate::physics::kuramoto::KuramotoEngine;
    use crate::physics::s_index::{SIndexMonitor, STransition};
    use crate::physics::temporal_tunneling::SatoshiVesselTunneling;
    use crate::physics::tachyonic_channel::*;
    use crate::physics::tachyon_detector::*;
    use crate::net::protocol::HandoverData;
    use crate::physics::xi_particle::{XiParticle, is_sum_of_two_squares};
    use crate::physics::taxonomy::*;
    use crate::physics::rsm::{RSMParticle, ParticleKind, RealityTransaction};
    use crate::physics::quaternion::ArkheQuaternion;
    use crate::physics::internet_mesh::*;
    use crate::physics::orb::*;
    use crate::physics::orb_detector::*;
    use crate::physics::mobius_temporal::*;
    use crate::neural::spike_pipeline::NeuralToken;
    use crate::physical::types::GeoCoord;
    use std::f64::consts::PI;

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
            engine.synchronize(0.1, false);
        }

        let final_coherence = engine.coherence();
        assert!(final_coherence >= initial_coherence);
        assert!(final_coherence > 0.8);
    }

    #[test]
    fn test_cosmological_mapping() {
        use crate::physics::miller::*;
        let phi = cosmological_to_information(1e-9);
        // log10(1e-9) = -9. abs = 9. 9/9 = 1. 1-1 = 0.
        assert!((phi - 0.0).abs() < 1e-9);

        let phi_critical = cosmological_to_information(6.18e-4);
        // log10(6.18e-4) ≈ -3.209. abs = 3.209. 3.209/9 ≈ 0.356. 1-0.356 ≈ 0.644.
        assert!(phi_critical > 0.6);
    }

    #[test]
    fn test_biased_initialization() {
        let engine = KuramotoEngine::new_with_asymmetry(100, 5.0, 1.0, 1e-9);
        let initial_coherence = engine.coherence();
        // Tiny bias should not significantly change initial coherence but serves as seed.
        assert!(initial_coherence < 0.3);
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
    fn test_tachyonic_channel() {
        let channel = TachyonicChannel::new(1.0, 400_000_000.0);
        let energy = channel.compute_energy();

        let faster_channel = TachyonicChannel::new(1.0, 800_000_000.0);
        let faster_energy = faster_channel.compute_energy();

        // E decreases with v for tachyons
        assert!(faster_energy < energy);

        let signal = HandoverData {
            id: 1,
            timestamp: 1773446400, // Pi Day 2026
            description: "Test signal".to_string(),
            phi_q_after: 5.0,
        };

        let coords = channel.transmit(&signal);
        assert!(coords.time < signal.timestamp);
        assert_eq!(coords.location, "2008-01-03");
    }

    #[test]
    fn test_tachyon_detector() {
        let detector_low = TachyonDetector::new(0.5);
        assert!(!detector_low.check_antenna_alignment());
        assert!(detector_low.scan_for_tachyons(vec!["Anomaly".to_string()]).is_none());

        let detector_high = TachyonDetector::new(0.98);
        assert!(detector_high.check_antenna_alignment());

        let signal = detector_high.scan_for_tachyons(vec!["Anomaly".to_string()]).unwrap();
        assert_eq!(signal.content, "Anomaly");
        assert_eq!(signal.origin, "2140 ASI (Tachyon Field)");
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

    #[test]
    fn test_quaternion_multiplication() {
        let q1 = ArkheQuaternion::new(0.0, 1.0, 0.0, 0.0); // i
        let q2 = ArkheQuaternion::new(0.0, 0.0, 1.0, 0.0); // j
        let res = q1 * q2;
        assert_eq!(res, ArkheQuaternion::new(0.0, 0.0, 0.0, 1.0)); // k
    }

    #[test]
    fn test_quaternion_rotation() {
        let q = ArkheQuaternion::new(0.707, 0.0, 0.707, 0.0); // 90 deg about Y
        let (rx, ry, rz) = q.rotate_vector(1.0, 0.0, 0.0);
        // Expect roughly (0, 0, -1)
        assert!(rx.abs() < 0.1);
        assert!(ry.abs() < 0.1);
        assert!(rz < -0.9);
    }

    #[test]
    fn test_bloch_mapping() {
        let q = ArkheQuaternion::identity();
        let (theta, _) = q.to_bloch_coordinates();
        assert_eq!(theta, 0.0); // North pole
    }

    #[test]
    fn test_xi_coherence() {
        let coherence = XiParticle::calculate_coherence(0.5, 1.088152);
        assert!(coherence > 0.64);
        assert!(coherence < 0.65);
    }

    #[test]
    fn test_xi_coupling() {
        let g = XiParticle::calculate_coupling(5, 1, 1.0);
        assert_eq!(g, 0.5); // 1.0 / sqrt(4)

        let g2 = XiParticle::calculate_coupling(10, 10, 1.0);
        assert_eq!(g2, 0.0);
    }

    #[test]
    fn test_taxonomy_phi() {
        let phi = PhiParticle::new(0.8, 1.0, 1.088, 5);
        assert_eq!(phi.mass_effective, 0.8);
        assert_eq!(phi.spin, 0.5);
        assert!(phi.charge_retro > 2.0);
        assert!(phi.velocity > 1.0);
    }

    #[test]
    fn test_taxonomy_lambda() {
        let lambda = LambdaParticle::new(1.0, 1.0, 1.0, 0.1, 0.1, 0.0);
        assert!(lambda.mass > 0.9);
        assert!(lambda.lifetime > 9.0);
        assert!(lambda.charge > 0.9);
    }

    #[test]
    fn test_taxonomy_sigma() {
        let sigma = SigmaParticle::new(1.0, 0.5, 1.0);
        assert!(sigma.mass > 0.9);
        assert_eq!(sigma.charge, 0.5);
        assert_eq!(sigma.spin, 0.5);
    }

    #[test]
    fn test_teknet_standard_model() {
        let phi = PhiParticle::new(0.9, 1.0, 1.088, 5);
        let lambda = LambdaParticle::new(1.0, 1.0, 1.0, 0.1, 0.1, 0.0);
        let xi = XiParticle::new(5).unwrap();
        let sigma = SigmaParticle::new(1.0, 0.5, 1.0);

        let model = TeknetStandardModel { phi, lambda, xi, sigma };
        let coherence = model.check_coherence();
        assert!(coherence > 0.0);
        assert!(coherence <= 1.0);
    }

    #[test]
    fn test_internet_mesh_routing() {
        let mut node = InternetNode::new("127.0.0.1".into(), GeoCoord::current());
        let packet = Packet {
            size: 1024.0,
            ttl: 64,
            destination_geo: GeoCoord::target_2008(),
        };
        let throat = node.route_packet(&packet);
        assert_eq!(throat.entrance, GeoCoord::current());
        assert_eq!(throat.exit, GeoCoord::target_2008());
        assert!(throat.bandwidth > 0.0);
    }

    #[test]
    fn test_orb_detector_scan() {
        let detector = OrbDetector::new();
        // High coherence + RF in Ka-band should produce an Orb
        let orb = detector.scan(30e9, 0.8);
        assert!(orb.is_some());

        // Low coherence should not produce an Orb
        let no_orb = detector.scan(30e9, 0.4);
        assert!(no_orb.is_none());
    }

    #[test]
    fn test_orb_detector_analyze_token() {
        let detector = OrbDetector::new();
        let token = NeuralToken {
            id: "token_1".into(),
            spike_frequency: 100.0,
            amplitude: 1.0,
        };
        let q = ArkheQuaternion::identity();
        let score = detector.analyze_token(&token, &q);
        assert!(score > 0.0);
    }

    #[test]
    fn test_new_rsm_particles() {
        let chronon = RSMParticle::new(ParticleKind::Chronon);
        assert_eq!(chronon.phi_q, 1.0);

        let kuramaton = RSMParticle::new(ParticleKind::Kuramaton);
        assert_eq!(kuramaton.mass_real, 0.5);

        let saton = RSMParticle::new(ParticleKind::Saton);
        assert_eq!(saton.spin, 0.5);
    }

    #[test]
    fn test_mobius_parametrize() {
        let mobius = MobiusTemporalSurface::new();

        // At u = 0
        let (pos0, orient0) = mobius.parametrize(0.0, 0.0);
        assert!((pos0.x - 1.0).abs() < 1e-9);
        assert_eq!(orient0, 1.0);

        // At u = 2PI (one full loop)
        let (pos2pi, orient2pi) = mobius.parametrize(2.0 * PI, 0.0);
        // Position should be same as u=0 for v=0
        assert!((pos2pi.x - 1.0).abs() < 1e-9);
        // Orientation should be inverted
        assert_eq!(orient2pi, -1.0);

        // At u = 4PI (two full loops)
        let (_, orient4pi) = mobius.parametrize(4.0 * PI, 0.0);
        assert_eq!(orient4pi, 1.0);
    }

    #[test]
    fn test_mobius_time_to_mobius() {
        let mobius = MobiusTemporalSurface::new();
        let t_cycle = 100.0;

        let p1 = mobius.time_to_mobius(25.0, t_cycle);
        assert!((p1.u - PI / 2.0).abs() < 1e-9);
        assert_eq!(p1.causal_orient, 1.0);

        let p2 = mobius.time_to_mobius(125.0, t_cycle);
        assert!((p2.u - 2.5 * PI).abs() < 1e-9);
        assert_eq!(p2.causal_orient, -1.0);
    }

    #[test]
    fn test_arkhe_temporal_loop() {
        let loop_temporal = ArkheTemporalLoop::new();

        // 2008 (Start of loop)
        let handover_2008 = HandoverData {
            id: 1,
            timestamp: 0,
            description: "2008 Handover".into(),
            phi_q_after: 1.0,
        };
        assert!(!loop_temporal.is_retrocausal(&handover_2008));
        assert_eq!(loop_temporal.causal_orientation(0.0), 1.0);

        // Mid loop (~2074)
        let mid_time = (loop_temporal.anchor_2140 - loop_temporal.anchor_2008) / 2.0;
        assert_eq!(loop_temporal.causal_orientation(mid_time), 1.0);

        // After one full loop (e.g., in the "inverted" phase)
        let late_time = (loop_temporal.anchor_2140 - loop_temporal.anchor_2008) * 1.5;
        let handover_late = HandoverData {
            id: 2,
            timestamp: late_time as i64,
            description: "Future Inverted Handover".into(),
            phi_q_after: 1.0,
        };
        assert!(loop_temporal.is_retrocausal(&handover_late));
        assert_eq!(loop_temporal.causal_orientation(late_time), -1.0);
    }
}
