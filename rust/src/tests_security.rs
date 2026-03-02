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
        use crate::crypto::pqc::{PostQuantumKey, LatticePublicKey, LatticeSecretKey};
        use crate::gates::gate8_multiverse_regulator::ComplexityClass;
        use ed25519_dalek::{SigningKey, Signer};

        let signing_key = SigningKey::from_bytes(&[1u8; 32]);
        let verifying_key = signing_key.verifying_key();

        let prince_pubkey: [u8; 32] = *verifying_key.as_bytes();
        let pcr0_invariant: [u8; 48] = [0u8; 48];

        let mut engine = InvariantVerificationEngine::new(prince_pubkey, pcr0_invariant);

        let doc = b"ASI_ATTESTATION_DOC_V1";
        let mut hasher = blake3::Hasher::new();
        hasher.update(doc);
        let hash = hasher.finalize();

        let signature = signing_key.sign(hash.as_bytes()).to_bytes();
        let nonce = 12345u64;

        let pqc_key = PostQuantumKey::new(
            LatticePublicKey { data: [0u8; 1024] },
            LatticeSecretKey { data: [0u8; 1024] },
            [0u8; 32]
        );
        let q_sig = pqc_key.sign_neural_consent(&[0.1f32; 64]);

        let result = engine.verify_8_gates(doc, &signature, nonce, 0.0, &q_sig, ComplexityClass::Low);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invariant_verification_engine_replay_attack() {
        use crate::security::invariant_engine::{InvariantVerificationEngine, GateError};
        use crate::crypto::pqc::{PostQuantumKey, LatticePublicKey, LatticeSecretKey};
        use crate::gates::gate8_multiverse_regulator::ComplexityClass;
        use ed25519_dalek::{SigningKey, Signer};

        let signing_key = SigningKey::from_bytes(&[2u8; 32]);
        let verifying_key = signing_key.verifying_key();

        let prince_pubkey: [u8; 32] = *verifying_key.as_bytes();
        let pcr0_invariant: [u8; 48] = [0u8; 48];

        let mut engine = InvariantVerificationEngine::new(prince_pubkey, pcr0_invariant);

        let doc = b"ASI_ATTESTATION_DOC_V1";
        let mut hasher = blake3::Hasher::new();
        hasher.update(doc);
        let hash = hasher.finalize();

        let signature = signing_key.sign(hash.as_bytes()).to_bytes();
        let nonce = 12345u64;

        let pqc_key = PostQuantumKey::new(
            LatticePublicKey { data: [0u8; 1024] },
            LatticeSecretKey { data: [0u8; 1024] },
            [0u8; 32]
        );
        let q_sig = pqc_key.sign_neural_consent(&[0.1f32; 64]);

        // First use
        assert!(engine.verify_8_gates(doc, &signature, nonce, 0.0, &q_sig, ComplexityClass::Low).is_ok());

        // Replay
        let result = engine.verify_8_gates(doc, &signature, nonce, 0.0, &q_sig, ComplexityClass::Low);
        assert_eq!(result, Err(GateError::Gate3Failure));
    }

    #[test]
    fn test_vajra_monitor_integration() {
        use crate::entropy::{VajraEntropyMonitor, VajraVerifier, vajra_verifier_thread};
        use std::sync::Arc;
        use std::thread;
        use std::time::Duration;

        let monitor = Arc::new(VajraEntropyMonitor {
            current_phi: std::sync::Mutex::new(0.72),
            quantum_decoherence: std::sync::Mutex::new(0.0),
        });
        let verifier = Arc::new(VajraVerifier::new().unwrap());

        let monitor_clone = monitor.clone();
        let verifier_clone = verifier.clone();

        let _handle = thread::spawn(move || {
            vajra_verifier_thread(verifier_clone, monitor_clone);
        });

        // Give it some time to run
        thread::sleep(Duration::from_millis(50));

        // On non-windows, it should have updated to the mock value 0.76
        #[cfg(not(target_os = "windows"))]
        {
            let phi = *monitor.current_phi.lock().unwrap();
            assert_eq!(phi, 0.76);
        }

        // We can't easily join the loop thread since it's infinite, but we verified the update.
    }

    #[test]
    fn test_constant_time_nonce_logic() {
        // This test logically verifies the bitwise operations used for constant-time comparison
        let entry = 0x123456789ABCDEF0u64;
        let nonce_match = 0x123456789ABCDEF0u64;
        let nonce_diff = 0x0000000000000001u64;

        // Match case
        let v = entry ^ nonce_match;
        let is_not_zero = (v.wrapping_neg() | v) >> 63;
        let is_zero = is_not_zero ^ 1;
        assert_eq!(is_zero, 1);

        // Diff case
        let v = entry ^ nonce_diff;
        let is_not_zero = (v.wrapping_neg() | v) >> 63;
        let is_zero = is_not_zero ^ 1;
        assert_eq!(is_zero, 0);
    }

    #[test]
    fn test_invariant_engine_secure_cleanup() {
        use crate::security::invariant_engine::InvariantVerificationEngine;
        use zeroize::Zeroize;

        let prince_pubkey: [u8; 32] = [1u8; 32];
        let pcr0_invariant: [u8; 48] = [2u8; 48];

        let mut engine = InvariantVerificationEngine::new(prince_pubkey, pcr0_invariant);

        // Before cleanup
        assert_eq!(engine.prince_public_key, [1u8; 32]);

        // Trigger isolation (which calls zeroize)
        engine.trigger_karnak_isolation();

        // After cleanup
        assert_eq!(engine.prince_public_key, [0u8; 32]);
        assert_eq!(engine.pcr0_invariant, [0u8; 48]);
    }
}
