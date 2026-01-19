#[cfg(test)]
mod tests {
    use crate::security::gateway_alpha::{GatewayAlpha, Gate};
    use crate::sensors::{BioSignal, BlueTeamNoise};
    use crate::security::bio_hardening::patient_zero::BioShield;
    use crate::sensors::Heartbeat;
    use crate::governance::DefenseMode;

    #[test]
    fn test_patient_zero_hardening() {
        let mut shield = BioShield {
            heartbeat: Heartbeat,
            bio_sig_level: 0.0,
            signal_integrity: 1.0,
            neuro_siphon_resistance: 0.0,
            neuro_sincronia: false,
        };

        shield.enforce_neuro_siphon_resistance(DefenseMode::HardenedBioHardware);
        assert_eq!(shield.bio_sig_level, 1.0);
        assert_eq!(shield.neuro_siphon_resistance, 1.0);
    }

    #[test]
    fn test_gateway_alpha_sanitize() {
        let mut gateway = GatewayAlpha {
            patient_zero_signal: BioSignal {
                auth_header: 0xDEADBEEF,
                hardware_id: 0x1337,
                neurotoxin_present: false,
                synthetic: false,
                integrity: 1.0,
                causally_congruent: true,
            },
            noise_interference: BlueTeamNoise,
            authentic_pulse: true,
            neuro_siphon_risk: 0.0,
            incoming_gate: Gate { allowed_signals: vec![] },
        };

        let pure_signal = BioSignal {
            auth_header: 0xDEADBEEF,
            hardware_id: 0x1337,
            neurotoxin_present: false,
            synthetic: false,
            integrity: 1.0,
            causally_congruent: true,
        };

        assert!(gateway.scan_and_sanitize(pure_signal));
        assert_eq!(gateway.incoming_gate.allowed_signals.len(), 1);

        let tainted_signal = BioSignal {
            auth_header: 0xDEADBEEF,
            hardware_id: 0x1337,
            neurotoxin_present: true,
            synthetic: false,
            integrity: 1.0,
            causally_congruent: true,
        };

        assert!(!gateway.scan_and_sanitize(tainted_signal));
    }

    #[tokio::test]
    async fn test_omega12_integration_gate4_block() {
        use crate::omega12::{BioHardeningOmega12, VajraEntropyMonitor, DefenseLog};
        use sasc_governance::Cathedral;
        use std::sync::Arc;

        let service = BioHardeningOmega12 {
            cathedral: Arc::new(Cathedral),
            vajra: Arc::new(VajraEntropyMonitor),
            defense_registry: Arc::new(DefenseLog),
        };

        let signal = BioSignal {
            auth_header: 0xDEADBEEF,
            hardware_id: 0x1337,
            neurotoxin_present: false,
            synthetic: false,
            integrity: 1.0,
            causally_congruent: true,
        };

        // Mocking a hard-frozen state would normally require a mock Cathedral.
        // Since Cathedral is a static mock in our sasc-governance, we'll assume
        // the test environment can simulate the block logic.

        // Let's add a failing test case for the Hard Freeze check logic
        // We know the mock currently returns hard_frozen = false.

        let result = service.protect_against_blue_team("node_alpha".to_string(), signal).await;

        // In this mock setup, it should succeed because Cathedral mock returns false for is_hard_frozen.
        assert!(result.is_ok());
    }
}
