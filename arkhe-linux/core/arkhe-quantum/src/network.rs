// arkhe-quantum/src/network.rs

use crate::crypto::{self, NodeKeys};
use anyhow::Result;

pub struct NetworkManager {
    pub node_id: String,
    pub node_keys: NodeKeys,
}

impl NetworkManager {
    pub fn new(node_id: String, node_keys: NodeKeys) -> Self {
        Self {
            node_id,
            node_keys,
        }
    }

    pub async fn send_handover(
        &self,
        receiver_id: String,
        payload: Vec<u8>,
        session_key: Option<&[u8; 32]>,
        manifold: &mut crate::manifold::GlobalManifold,
    ) -> Result<bool> {
        // P1 & biological constraint: Only Active nodes can emit handovers
        if let Some(node) = manifold.get_self_node_mut() {
            if node.state != crate::manifold::NodeState::Active {
                return Err(anyhow::anyhow!("Node is not licensed or active for emission (MCM constraint)"));
            }
            node.handover_count += 1;
        }

        let mut encrypted_payload = payload;
        if let Some(key) = session_key {
            encrypted_payload = crypto::encrypt_payload(&encrypted_payload, key)?;
        }

        let signature = crypto::sign_message(&encrypted_payload, &self.node_keys.dilithium_secret);

        tracing::info!("Sending handover to {} (signed, encrypted: {})", receiver_id, session_key.is_some());
        // gRPC call would go here

        Ok(true)
    }
}
