use std::collections::VecDeque;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct HandoverPayload {
    pub node_id: String,
    pub timestamp: i64,
    pub image_data: Vec<u8>,
    pub entropy_estimate: f64,
}

pub struct NightVisionNode {
    pub node_id: String,
    pub last_frame: Vec<u8>,
    pub entropy_history: VecDeque<f64>,
}

impl NightVisionNode {
    pub fn new(node_id: &str) -> Self {
        Self {
            node_id: node_id.to_string(),
            last_frame: Vec::new(),
            entropy_history: VecDeque::with_capacity(100),
        }
    }

    pub fn process_handover(&mut self, payload: &HandoverPayload) -> anyhow::Result<f64> {
        self.last_frame = payload.image_data.clone();
        self.entropy_history.push_back(payload.entropy_estimate);
        if self.entropy_history.len() > 100 {
            self.entropy_history.pop_front();
        }

        let avg_entropy: f64 = self.entropy_history.iter().sum::<f64>() / self.entropy_history.len() as f64;
        let perturbation = (avg_entropy - 0.5).clamp(-0.1, 0.1);

        Ok(perturbation)
    }
}
