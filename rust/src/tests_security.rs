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

    #[test]
    fn test_invariant_verification_engine_full_pass() {
        use crate::security::invariant_engine::{InvariantVerificationEngine, GateError};
        use ed25519_dalek::{SigningKey, Signer};

        let signing_key = SigningKey::from_bytes(&[1u8; 32]);
        let verifying_key = signing_key.verifying_key();

        let prince_pubkey: [u8; 32] = *verifying_key.as_bytes();
        let pcr0_invariant: [u8; 48] = [0u8; 48];

        let engine = InvariantVerificationEngine::new(prince_pubkey, pcr0_invariant);

        let doc = b"ASI_ATTESTATION_DOC_V1";
        let mut hasher = blake3::Hasher::new();
        hasher.update(doc);
        let hash = hasher.finalize();

        let signature = signing_key.sign(hash.as_bytes()).to_bytes();
        let nonce = 12345u64;

        let result = engine.verify_5_gates(doc, &signature, nonce);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invariant_verification_engine_replay_attack() {
        use crate::security::invariant_engine::{InvariantVerificationEngine, GateError};
        use ed25519_dalek::{SigningKey, Signer};

        let signing_key = SigningKey::from_bytes(&[2u8; 32]);
        let verifying_key = signing_key.verifying_key();

        let prince_pubkey: [u8; 32] = *verifying_key.as_bytes();
        let pcr0_invariant: [u8; 48] = [0u8; 48];

        let engine = InvariantVerificationEngine::new(prince_pubkey, pcr0_invariant);

        let doc = b"ASI_ATTESTATION_DOC_V1";
        let mut hasher = blake3::Hasher::new();
        hasher.update(doc);
        let hash = hasher.finalize();

        let signature = signing_key.sign(hash.as_bytes()).to_bytes();
        let nonce = 12345u64;

        // First use
        assert!(engine.verify_5_gates(doc, &signature, nonce).is_ok());

        // Replay
        let result = engine.verify_5_gates(doc, &signature, nonce);
        assert_eq!(result, Err(GateError::Gate3Failure));
    }
}
