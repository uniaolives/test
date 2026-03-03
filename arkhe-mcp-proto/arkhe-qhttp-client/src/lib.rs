use serde::{Deserialize, Serialize};
use anyhow::Result;

#[derive(Debug, Serialize, Deserialize)]
pub struct HandoverRequest {
    pub target_node: String,
    pub context_summary: String,
    pub priority: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HandoverResponse {
    pub status: String,
    pub handover_id: String,
    pub phi_remote: f64,
}

pub struct QHttpClient {
    client: reqwest::Client,
}

impl QHttpClient {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    pub async fn sync_context(&self, endpoint: &str, req: HandoverRequest) -> Result<HandoverResponse> {
        let resp = self.client.post(format!("{}/handover", endpoint))
            .json(&req)
            .send()
            .await?
            .json::<HandoverResponse>()
            .await?;
        Ok(resp)
    }
}
