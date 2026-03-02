use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct HandshakePacket {
    pub public_key: Vec<u8>,
    pub timestamp: u64,
}

pub struct SecurityEnclave {
    pub node_id: String,
}

impl SecurityEnclave {
    pub fn new(node_id: &str) -> Self {
        SecurityEnclave {
            node_id: node_id.to_string(),
        }
    }

    pub fn encrypt(&self, data: &[u8]) -> Vec<u8> {
        // Mock AES-GCM
        data.to_vec()
    }
}
