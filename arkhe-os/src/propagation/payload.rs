use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrbPayload {
    pub orb_id: [u8; 32],
    pub lambda_2: f64,
    pub phi_q: f64,
    pub h_value: f64,
    pub origin_time: i64,
    pub target_time: i64,
    pub timechain_hash: [u8; 32],
    pub signature: Vec<u8>,
    pub created_at: i64,
}

impl OrbPayload {
    pub fn create(
        lambda_2: f64,
        phi_q: f64,
        h_value: f64,
        origin_time: i64,
        target_time: i64,
        timechain_hash: Option<[u8; 32]>,
        signature: Option<Vec<u8>>,
    ) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let content = format!("{}{}{}{}{}{}", lambda_2, phi_q, h_value, origin_time, target_time, created_at);
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let result = hasher.finalize();
        let mut orb_id = [0u8; 32];
        orb_id.copy_from_slice(&result);

        Self {
            orb_id,
            lambda_2,
            phi_q,
            h_value,
            origin_time,
            target_time,
            timechain_hash: timechain_hash.unwrap_or([0u8; 32]),
            signature: signature.unwrap_or_else(|| b"UNSIGNED".to_vec()),
            created_at,
        }
    }

    pub fn informational_mass(&self) -> f64 {
        (self.lambda_2 * self.phi_q) / self.h_value.max(0.001)
    }

    pub fn is_retrocausal(&self) -> bool {
        self.target_time < self.origin_time
    }

    pub fn temporal_span(&self) -> i64 {
        (self.target_time - self.origin_time).abs()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        // Use bincode for internal use, but we might want a manual implementation
        // to match the exact byte layout specified in the Python version if needed for cross-language.
        // For now, let's use bincode as it's already in arkhe-os dependencies.
        bincode::serialize(self).unwrap()
    }

    pub fn from_bytes(data: &[u8]) -> bincode::Result<Self> {
        bincode::deserialize(data)
    }
}
