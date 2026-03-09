use crate::physics::arkhe_orb_core::Orb;
use super::payload::OrbPayload;
use super::universal_bridge::{ProtocolBridge, PropagationReceipt, ProtocolType, OrbId};
use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;

pub struct BlockchainBridge;

#[async_trait]
impl ProtocolBridge for BlockchainBridge {
    async fn propagate(&self, _orb: &Orb, _payload: &OrbPayload) -> Result<PropagationReceipt> {
    async fn propagate(&self, _orb: &Orb) -> Result<PropagationReceipt> {
        // Bitcoin OP_RETURN, Ethereum Contract Logs, Timechain Blocks
        Ok(PropagationReceipt {
            protocol: ProtocolType::Blockchain,
            timestamp: Utc::now().timestamp(),
        })
    }

    fn has_memory(&self, _orb_id: &OrbId) -> bool {
        true
    }
}
