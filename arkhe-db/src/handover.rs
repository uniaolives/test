use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Handover {
    pub id: u64,
    pub timestamp: i64,
    pub source_epoch: u32,
    pub target_epoch: u32,
    pub coherence: f64,
    pub phi_q_before: f64,
    pub phi_q_after: f64,
    pub quantum_interest: f64,
    pub payload_hash: String,
}
