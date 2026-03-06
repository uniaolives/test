use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum TeknetMessage {
    Hello {
        peer_id: String,
        last_handover_id: u64
    },
    SyncRequest {
        from_id: u64,
        to_id: u64
    },
    SyncResponse {
        handovers: Vec<HandoverData>
    },
    NewHandover {
        handover: HandoverData
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HandoverData {
    pub id: u64,
    pub timestamp: i64,
    pub description: String,
    pub phi_q_after: f64,
}
