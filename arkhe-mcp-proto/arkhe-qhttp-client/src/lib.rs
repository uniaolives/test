use anyhow::Result;
pub use arkhe_api_rust::objects::com::palantir::arkhe::api::{
    HandoverRequest, HandoverResponse, AgentCard, Orb, CoherenceMetric, TemporalTrajectory
};
pub use arkhe_api_rust::clients::com::palantir::arkhe::api::{
    HandoverService, AsyncHandoverService, TemporalService, AsyncTemporalService
};

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
        let resp = self.client.post(format!("{}/handover/sync", endpoint))
            .json(&req)
            .send()
            .await?
            .json::<HandoverResponse>()
            .await?;
        Ok(resp)
    }

    // New temporal emission method using Conjure types
    pub async fn emit_orb(&self, endpoint: &str, orb: Orb) -> Result<CoherenceMetric> {
        let resp = self.client.post(format!("{}/temporal/emit", endpoint))
            .json(&orb)
            .send()
            .await?
            .json::<CoherenceMetric>()
            .await?;
        Ok(resp)
    }
}
