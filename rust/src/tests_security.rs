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
            omega_signal: None,
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
}
