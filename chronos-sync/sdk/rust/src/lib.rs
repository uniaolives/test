use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub tx_id: String,
    pub events: Vec<Event>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub name: String,
    pub timestamp: f64,
}

pub struct Client {
    api_key: String,
}

impl Client {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
        }
    }

    pub async fn begin_tx(&self) -> Transaction {
        Transaction {
            tx_id: format!("orb_{}", Uuid::new_v4()),
            events: Vec::new(),
        }
    }

    pub async fn get_cluster_coherence(&self) -> f64 {
        0.99 // Mock λ₂
    }
}

impl Transaction {
    pub async fn get_global_time(&self) -> GlobalTime {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs_f64();

        GlobalTime {
            timestamp: now,
            coherence: 0.995, // Mock λ₂ at time of commit
        }
    }

    pub async fn commit(&mut self) -> Result<f64, String> {
        let global_time = self.get_global_time().await;
        println!("[Chronos] Transaction {} committed at {} (λ₂={})", self.tx_id, global_time.timestamp, global_time.coherence);
        Ok(global_time.timestamp)
    }

    pub fn record_event(&mut self, name: &str, timestamp: Option<f64>) {
        let ts = timestamp.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64()
        });
        self.events.push(Event {
            name: name.to_string(),
            timestamp: ts,
        });
        println!("[Chronos] Recorded event: {} at {}", name, ts);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalTime {
    pub timestamp: f64,
    pub coherence: f64,
}
