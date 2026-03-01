use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Mutex;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HandoverPacket {
    pub id: String,
    pub target: String,
    pub payload: serde_json::Value,
    pub timestamp: u64,
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

    pub async fn process_queue(&self) -> Result<(), anyhow::Error> {
        let mut queue = self.queue.lock().unwrap();
        let mut history = self.history.lock().unwrap();
        while let Some(packet) = queue.pop_front() {
            tracing::info!("Processing handover: {}", packet.id);
            history.push(packet);
        }
        Ok(())
    }

    pub fn enqueue(&self, packet: HandoverPacket) {
        let mut queue = self.queue.lock().unwrap();
        queue.push_back(packet);
    }

    pub async fn broadcast_phi(&self, _phi: f64) {
    }

    pub async fn send_system_notification(&self, msg: &str) -> Result<(), anyhow::Error> {
        tracing::info!("System notification: {}", msg);
        Ok(())
    }

    pub fn get_history(&self) -> Vec<HandoverPacket> {
        self.history.lock().unwrap().clone()
    }
}
