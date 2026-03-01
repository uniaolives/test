use serde::{Serialize, Deserialize};
use std::collections::VecDeque;
use std::sync::Mutex;
pub use arkhe_quantum::Handover;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HandoverPacket {
    pub id: String,
    pub target: String,
    pub payload: serde_json::Value,
    pub timestamp: u64,
    pub binary: Option<Handover>,
}

pub struct HandoverManager {
    queue: Mutex<VecDeque<HandoverPacket>>,
    history: Mutex<Vec<HandoverPacket>>,
}

impl HandoverManager {
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            history: Mutex::new(Vec::new()),
        }
    }

    pub fn enqueue(&self, packet: HandoverPacket) {
        let mut queue = self.queue.lock().unwrap();
        queue.push_back(packet);
    }

    pub fn pop_next(&self) -> Option<HandoverPacket> {
        let mut queue = self.queue.lock().unwrap();
        queue.pop_front()
    }

    pub fn record_history(&self, packet: HandoverPacket) {
        let mut history = self.history.lock().unwrap();
        history.push(packet);
    }

    pub async fn broadcast_phi(&self, _phi: f64) {
        // Implementation for phi broadcasting
    }

    pub async fn send_system_notification(&self, msg: &str) -> Result<(), anyhow::Error> {
        tracing::info!("System notification: {}", msg);
        Ok(())
    }

    pub fn get_history(&self) -> Vec<HandoverPacket> {
        self.history.lock().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handover_packet() {
        let packet = HandoverPacket {
            id: "test".to_string(),
            target: "target".to_string(),
            payload: serde_json::json!({"test": "data"}),
            timestamp: 0,
            binary: None,
        };
        assert_eq!(packet.id, "test");
    }
}
