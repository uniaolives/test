use std::time::SystemTime;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioMetadata {
    pub signal_delta: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct BioMetrics {
    pub trust_level: f32,
}

impl BioMetrics {
    pub fn is_live_and_consistent(&self) -> bool {
        true // Mock
    }
}

#[derive(Debug)]
pub enum BioSecurityError {
    HardwareCompromised,
    BiometricSpoofing,
    InsufficientSocialCost,
    SybilAttackDetected,
}

#[derive(Debug)]
pub struct PatientZeroAuth {
    pub node_id: String,
    pub trust_score: f32,
    pub human_verification: bool,
    pub auth_expiry: SystemTime,
    pub metadata: BioMetadata,
}
