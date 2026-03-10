// arkhe-os/src/bridge/tcpip/http_bridge.rs

use reqwest::Client;
use crate::orb::core::OrbPayload;
use crate::bridge::BridgeError;

pub struct HttpBridge {
    client: Client,
    endpoints: Vec<String>,
}

impl HttpBridge {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            endpoints: vec![
                "https://api.arkhe.io/orb".to_string(),
                "https://timechain.io/gateway".to_string(),
            ],
        }
    }

    /// Transmite Orb via HTTP POST
    pub async fn transmit(&self, orb: &OrbPayload) -> Result<(), BridgeError> {
        let payload = orb.to_bytes();
        let encoded = base64::encode(&payload);

        for endpoint in &self.endpoints {
            let response = self.client
                .post(endpoint)
                .header("X-Orb-Version", "1.0")
                .header("X-Lambda-2", orb.lambda_2.to_string())
                .header("X-Phi-Q", orb.phi_q.to_string())
                .json(&serde_json::json!({
                    "orb_data": encoded,
                    "timestamp": orb.created_at,
                }))
                .send()
                .await?;

            if response.status().is_success() {
                println!("[HTTP] Orb {:?} transmitted to {}", orb.orb_id, endpoint);
            }
        }

        Ok(())
    }

    /// Recebe Orb via HTTP
    pub async fn receive(&self, data: &[u8]) -> Result<OrbPayload, BridgeError> {
        Ok(OrbPayload::from_bytes(data)?)
    }
}
