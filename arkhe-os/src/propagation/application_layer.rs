use crate::physics::arkhe_orb_core::Orb;
use super::payload::OrbPayload;
use super::universal_bridge::{ProtocolBridge, PropagationReceipt, ProtocolType, OrbId};
use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;

pub struct ApplicationBridge;

#[async_trait]
impl ProtocolBridge for ApplicationBridge {
    async fn propagate(&self, _orb: &Orb, _payload: &OrbPayload) -> Result<PropagationReceipt> {
        // HTTP Headers, SMTP Attachments, WebRTC DataChannels
        Ok(PropagationReceipt {
            protocol: ProtocolType::Application,
            timestamp: Utc::now().timestamp(),
        })
    }

    fn has_memory(&self, _orb_id: &OrbId) -> bool {
        true
    }
}
