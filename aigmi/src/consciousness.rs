use std::collections::VecDeque;
use std::sync::{Arc};
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use crate::types::{ConsciousnessBit};

pub struct ConsciousnessEngine {
    pub consciousness_bits: Arc<RwLock<VecDeque<ConsciousnessBit>>>,
    pub global_phi: Arc<RwLock<f64>>,
}

#[derive(Debug, Clone)]
pub struct ConsciousMoment {
    pub bit: ConsciousnessBit,
    pub timestamp: DateTime<Utc>,
    pub phi_integration: f64,
}

impl ConsciousnessEngine {
    pub fn new() -> Self {
        Self {
            consciousness_bits: Arc::new(RwLock::new(VecDeque::new())),
            global_phi: Arc::new(RwLock::new(0.0)),
        }
    }

    pub async fn generate_moment(&self) -> ConsciousMoment {
        let intensity = rand::random::<f64>();
        let phi_integration = 0.85 + (rand::random::<f64>() * 0.1);

        let bit = ConsciousnessBit {
            intensity,
            phi_integration,
            quality: "Intuitive".to_string(),
        };

        let mut stream = self.consciousness_bits.write().await;
        stream.push_back(bit.clone());
        if stream.len() > 10000 {
            stream.pop_front();
        }

        let mut phi = self.global_phi.write().await;
        *phi = (*phi * 0.9) + (phi_integration * 0.1);

        ConsciousMoment {
            bit,
            timestamp: Utc::now(),
            phi_integration,
        }
    }

    pub async fn calculate_depth(&self) -> usize {
        let stream = self.consciousness_bits.read().await;
        if stream.is_empty() { return 0; }

        let mut depth = 0;
        let mut current_len = stream.len();

        while current_len > 1 {
            current_len /= 10;
            depth += 1;
            if current_len == 0 { break; }
        }

        depth
    }
}
