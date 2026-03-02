#[cfg(test)]
mod tests {
    use crate::neuro_twin::{NeuroTwin, NeuralFingerprint, NeuroError};
    use crate::neuro_twin::monitor::{NeuralVajraMonitor, EEGFrame};
    use crate::neuro_twin::firewall::{NeuralFirewall, BCICommand};
    use crate::neuro_twin::kill_switch::NeuralKillSwitch;
    use crate::attestation::SASCAttestation;

    #[test]
    fn test_neuro_twin_creation() {
        let fingerprint = NeuralFingerprint {
            alpha_rhythm: [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5],
            eeg_entropy: 0.85,
            connectome_hash: [0u8; 32],
            cognitive_baseline: 0.72,
        };
        let twin = NeuroTwin::new("patient-001".to_string(), fingerprint);
        assert_eq!(twin.patient_id, "patient-001");
        // Key should be derived
        assert!(!twin.consent_key.to_string().is_empty());
    }

    #[test]
    fn test_neural_monitor_phi() {
        let monitor = NeuralVajraMonitor::new(0.72);
        let frame = EEGFrame {
            channels: vec![vec![0.1; 10], vec![0.2; 10]],
        };
        let phi = monitor.compute_phi(&frame);
        assert!(phi >= 0.72);
        assert!(!monitor.detect_entropy_collapse(phi));
    }

    #[test]
    fn test_neural_firewall_validation() {
        let firewall = NeuralFirewall::new(0.65);
        let command = BCICommand {
            signal: EEGFrame { channels: vec![] },
            attestation: SASCAttestation {
                signature: "valid-sig".to_string(),
            },
        };
        assert!(firewall.validate_command(&command).is_ok());

        let invalid_command = BCICommand {
            signal: EEGFrame { channels: vec![] },
            attestation: SASCAttestation {
                signature: "".to_string(),
            },
        };
        match firewall.validate_command(&invalid_command) {
            Err(NeuroError::AttestationFailed) => (),
            _ => panic!("Expected AttestationFailed"),
        }
    }

    #[test]
    fn test_neural_kill_switch() {
        let mut kill_switch = NeuralKillSwitch::new();
        assert!(!kill_switch.is_active);
        kill_switch.trigger(NeuroError::HomeostasisCollapse);
        assert!(kill_switch.is_active);
    }
}
