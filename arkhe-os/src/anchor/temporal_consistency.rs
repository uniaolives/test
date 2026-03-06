use chrono::{DateTime, Utc};
use sha3::{Sha3_256, Digest};
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnchorBlock {
    pub timestamp: DateTime<Utc>,
    pub phi_q: f64,
    pub bio_hash: String,
    pub mobile_sig: String,
    pub prev_hash: String,
    pub future_commitment: Option<String>,
    pub nonce: u64,
}

impl AnchorBlock {
    pub fn new(phi_q: f64, bio_state: &[u8], mobile_state: &[u8], prev_hash: String) -> Self {
        let mut hasher = Sha3_256::new();
        hasher.update(bio_state);
        let bio_hash = format!("{:x}", hasher.finalize_reset());

        hasher.update(mobile_state);
        let mobile_sig = format!("{:x}", hasher.finalize());

        Self {
            timestamp: Utc::now(),
            phi_q,
            bio_hash,
            mobile_sig,
            prev_hash,
            future_commitment: None,
            nonce: 0,
        }
    }
}
