use crate::net::multimodal_anchor::{MultimodalCoherencePacket, ConnectivityLayer};
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HandoverRecord {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub phi_q_before: f64,
    pub phi_q_after: f64,
    pub zpf_signature: String,
    pub propagation_path: Vec<ConnectivityLayer>,
}

pub struct QuantumNetworkHandover {
    pub history: Vec<HandoverRecord>,
}

impl QuantumNetworkHandover {
    pub fn new() -> Self {
        Self { history: Vec::new() }
    }
}
