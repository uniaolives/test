use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use crate::physics::berry::TopologicalQubit;
use sha3::{Sha3_256, Digest};
use rand::Rng;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FutureCommitment {
    pub id: u64,
    pub created_at: DateTime<Utc>,
    pub valid_after: DateTime<Utc>,
    pub description: String,
    pub predicted_phi_q: f64,
    pub berry_phase_target: f64,
    pub signature: Option<Vec<u8>>,
    pub fulfilled: bool,
}

impl FutureCommitment {
    pub fn new_berry(
        description: &str,
        valid_after: DateTime<Utc>,
        predicted_phi_q: f64,
        berry_phase_target: f64,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let id = rng.gen();
        Self {
            id,
            created_at: Utc::now(),
            valid_after,
            description: description.to_string(),
            predicted_phi_q,
            berry_phase_target,
            signature: None,
            fulfilled: false,
        }
    }

    pub fn validate_berry(&mut self, current_berry_phase: f64, signature: Option<&[u8]>) -> bool {
        let phase_ok = (current_berry_phase - self.berry_phase_target).abs() < 0.01;

        if phase_ok && Utc::now() >= self.valid_after {
            self.fulfilled = true;
            self.signature = signature.map(|s| s.to_vec());
            true
        } else {
            false
        }
    }
}
