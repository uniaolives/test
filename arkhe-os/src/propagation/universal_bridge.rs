use std::collections::HashMap;
use crate::physics::arkhe_orb_core::Orb;
use super::payload::OrbPayload;
use anyhow::Result;
use async_trait::async_trait;

pub type OrbId = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProtocolType {
    Physical,
    Network,
    Application,
    Blockchain,
}

pub struct PropagationReceipt {
    pub protocol: ProtocolType,
    pub timestamp: i64,
}

#[async_trait]
pub trait ProtocolBridge: Send + Sync {
    async fn propagate(&self, orb: &Orb, payload: &OrbPayload) -> Result<PropagationReceipt>;
    fn has_memory(&self, orb_id: &OrbId) -> bool;
}

pub struct UniversalOrbPropagator {
    /// Map of all supported protocols
    protocols: HashMap<ProtocolType, Box<dyn ProtocolBridge>>,
}

impl UniversalOrbPropagator {
    pub fn new() -> Self {
        Self {
            protocols: HashMap::new(),
        }
    }

    pub fn register_bridge(&mut self, protocol: ProtocolType, bridge: Box<dyn ProtocolBridge>) {
        self.protocols.insert(protocol, bridge);
    }

    /// Propagates an orb through ALL registered protocols simultaneously
    pub async fn propagate_everywhere(&self, orb: &Orb, payload: &OrbPayload) -> Result<Vec<PropagationReceipt>> {
        use futures::future::join_all;

        let futures: Vec<_> = self.protocols.values()
            .map(|bridge| bridge.propagate(orb, payload))
            .collect();

        let results = join_all(futures).await;
        let mut receipts = Vec::new();

        for res in results {
            receipts.push(res?);
        }

        println!("✅ Orb propagated through {} protocols", receipts.len());
        Ok(receipts)
    }

    /// Checks how many protocols "remember" the orb (redundancy)
    pub fn check_propagation(&self, orb_id: &OrbId) -> PropagationStats {
        let mut total = 0;
        let mut successful = 0;

        for (_, bridge) in &self.protocols {
            total += 1;
            if bridge.has_memory(orb_id) {
                successful += 1;
            }
        }

        PropagationStats {
            total_protocols: total,
            successful_propagations: successful,
            redundancy_ratio: if total > 0 { successful as f64 / total as f64 } else { 0.0 },
        }
    }
}

#[derive(Debug)]
pub struct PropagationStats {
    pub total_protocols: usize,
    pub successful_propagations: usize,
    pub redundancy_ratio: f64,
}
