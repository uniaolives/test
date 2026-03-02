use serde::{Serialize, Deserialize};
use crate::crypto::{BLAKE3_Δ2};

pub mod monitor;
pub mod firewall;
pub mod kill_switch;

#[cfg(test)]
mod tests;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NeuralFingerprint {
    pub alpha_rhythm: [f64; 8],      // Signature of occipital alpha rhythm
    pub eeg_entropy: f64,            // Shannon entropy of basal EEG
    pub connectome_hash: [u8; 32],   // Hash of structural connectome
    pub cognitive_baseline: f64,     // Placeholder for cognitive baseline attractor metric
}

impl NeuralFingerprint {
    /// Derives the Δ2 key from the patient's unique neural biometrics
    pub fn derive_delta2_key(&self) -> BLAKE3_Δ2 {
        let mut hasher = blake3::Hasher::new();
        for val in self.alpha_rhythm.iter() {
            hasher.update(&val.to_le_bytes());
        }
        hasher.update(&self.eeg_entropy.to_le_bytes());
        hasher.update(&self.connectome_hash);
        hasher.update(&self.cognitive_baseline.to_le_bytes());

        let output = hasher.finalize();
        let mut data = [0u8; 32];
        data.copy_from_slice(output.as_bytes());
        BLAKE3_Δ2::new(data)
    }
}

pub struct NeuroTwin {
    pub patient_id: String,
    pub fingerprint: NeuralFingerprint,
    pub consent_key: BLAKE3_Δ2,
}

impl NeuroTwin {
    pub fn new(patient_id: String, fingerprint: NeuralFingerprint) -> Self {
        let consent_key = fingerprint.derive_delta2_key();
        Self {
            patient_id,
            fingerprint,
            consent_key,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum NeuroError {
    #[error("Invalid neural signal")]
    InvalidSignal,
    #[error("Attestation failed")]
    AttestationFailed,
    #[error("Homeostasis collapse detected")]
    HomeostasisCollapse,
    #[error("Unauthorized access")]
    Unauthorized,
}
