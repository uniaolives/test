// rust/src/memory/dmr/validation.rs
#[cfg(test)]
mod tests {
    use crate::memory::dmr::types::*;
    use crate::memory::dmr::ring::DigitalMemoryRing;
    use std::time::Duration;

    #[test]
    fn experiment_dmr_1_tkr_accumulation() {
        // Hypothesis: Agents in stable states accumulate t_KR linearly.
        let vk_ref = KatharosVector::new(0.5, 0.5, 0.5, 0.5);
        let mut dmr = DigitalMemoryRing::new("test-dmr-1".to_string(), vk_ref.clone(), Duration::from_secs(3600));

        // Maintain ΔK < 0.30 for 24 "hours"
        for _ in 0..24 {
            let state = SystemState {
                vk: KatharosVector::new(0.51, 0.49, 0.50, 0.50), // very close to ref
                entropy: 0.1,
                events: Vec::new(),
            };
            dmr.grow_layer(state).unwrap();
        }

        let t_kr = dmr.measure_t_kr();
        assert_eq!(t_kr.as_secs(), 24 * 3600);
    }

    #[test]
    fn experiment_dmr_2_bifurcation_detection() {
        // Hypothesis: Rapid state changes trigger bifurcation markers.
        let vk_ref = KatharosVector::new(0.5, 0.5, 0.5, 0.5);
        let mut dmr = DigitalMemoryRing::new("test-dmr-2".to_string(), vk_ref.clone(), Duration::from_secs(3600));

        // Stable for 10 hours
        for _ in 0..10 {
            let state = SystemState {
                vk: KatharosVector::new(0.5, 0.5, 0.5, 0.5),
                entropy: 0.1,
                events: Vec::new(),
            };
            dmr.grow_layer(state).unwrap();
        }

        // Crisis at t=10h (ΔK = 0.85 approx)
        let crisis_state = SystemState {
            vk: KatharosVector::new(1.0, 1.0, 1.0, 1.0),
            entropy: 0.9,
            events: vec![CellularEvent { event_type: "CRISIS".to_string(), metadata: "Induced".to_string() }],
        };
        dmr.grow_layer(crisis_state).unwrap();

        assert_eq!(dmr.bifurcations.len(), 1);
        assert_eq!(dmr.bifurcations[0].bifurcation_type, BifurcationType::CrisisEntry);

        // Return to stability
        let stable_state = SystemState {
            vk: KatharosVector::new(0.5, 0.5, 0.5, 0.5),
            entropy: 0.1,
            events: Vec::new(),
        };
        dmr.grow_layer(stable_state).unwrap();
        assert_eq!(dmr.bifurcations.len(), 2);
        assert_eq!(dmr.bifurcations[1].bifurcation_type, BifurcationType::CrisisExit);
    }

    #[test]
    fn experiment_dmr_3_gemini_pattern_replication() {
        // Hypothesis: DMR intensity patterns match GEMINI fluorescence for equivalent perturbations.
        let vk_ref = KatharosVector::new(0.5, 0.5, 0.5, 0.5);
        let mut dmr = DigitalMemoryRing::new("test-dmr-3".to_string(), vk_ref.clone(), Duration::from_secs(900)); // 15 min

        // Simulate NFκB-like spike
        // Mocking some intensity values that would come from GEMINI
        let mock_gemini_intensities = vec![0.1, 0.1, 0.8, 0.9, 0.7, 0.4, 0.2, 0.1];

        let vk_sequence = vec![
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
            [0.9, 0.9, 0.9, 0.9], // spike
            [1.0, 1.0, 1.0, 1.0], // peak
            [0.8, 0.8, 0.8, 0.8], // decay
            [0.7, 0.7, 0.7, 0.7],
            [0.6, 0.6, 0.6, 0.6],
            [0.5, 0.5, 0.5, 0.5],
        ];

        for vk_comp in vk_sequence {
            let state = SystemState {
                vk: KatharosVector { components: vk_comp },
                entropy: 0.5,
                events: Vec::new(),
            };
            dmr.grow_layer(state).unwrap();
        }

        let correlation = dmr.compute_correlation(&mock_gemini_intensities);
        println!("DMR-GEMINI Correlation: {}", correlation);
        assert!(correlation > 0.85);
    }
}
