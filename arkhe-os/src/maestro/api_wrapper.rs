use reqwest::{Client, ClientBuilder};
use std::time::Duration;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug)]
pub struct PTPConfig {
    pub target_latency_ms: u64,
    pub timeout_ms: u64,
    pub retry_attempts: u8,
    pub circuit_breaker_threshold: u8,
}

impl Default for PTPConfig {
    fn default() -> Self {
        Self {
            target_latency_ms: 500,
            timeout_ms: 2000,
            retry_attempts: 3,
            circuit_breaker_threshold: 5,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeLatencyProfile {
    pub avg_latency_ms: f64,
    pub health_score: f64,
    pub circuit_state: CircuitState,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

pub struct PTPApiWrapper {
    client: Client,
    _config: PTPConfig,
    _circuit_failures: HashMap<String, u8>,
}

impl PTPApiWrapper {
    pub fn new(config: PTPConfig) -> Self {
        let client = ClientBuilder::new()
            .timeout(Duration::from_millis(config.timeout_ms))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            client,
            _config: config,
            _circuit_failures: HashMap::new(),
        }
    }

    pub async fn execute_raw(&self, endpoint: &str, payload: serde_json::Value) -> Result<serde_json::Value, String> {
        let response = self.client.post(endpoint)
            .json(&payload)
            .send()
            .await
            .map_err(|e| e.to_string())?;

        let body = response.json().await.map_err(|e| e.to_string())?;
        Ok(body)
    }
}
