use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KyberSession {
    pub session_id: String,
    pub peer_id: String,
    pub shared_secret_hash: String,
    pub is_verified: bool,
}

impl KyberSession {
    pub fn new_simulated(peer_id: &str) -> Self {
        Self {
            session_id: uuid::Uuid::new_v4().to_string(),
            peer_id: peer_id.to_string(),
            shared_secret_hash: "fixed_secret_for_simulation_32b".to_string(),
            is_verified: true,
        }
    }
}
