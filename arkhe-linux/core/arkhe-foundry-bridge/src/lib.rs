pub mod client;
pub mod mapper;
pub mod sync;
pub mod types;

use std::time::Duration;
use serde::{Deserialize, Serialize};
use reqwest::Client;
use arkhe_manifold::{QuantumState, GlobalManifold};
use num_complex::Complex64;
use nalgebra::DMatrix;
use anyhow::Result;

pub struct BridgeConfig {
    pub poll_interval_ms: u64,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self { poll_interval_ms: 1000 }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FoundryObject {
    pub rid: String,
    pub api_name: String,
    pub properties: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ActionParameters {
    pub parameters: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ActionResult {
    pub status: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ObjectTypeCreationResult {
    pub rid: String,
}

/// Bridge entre Arkhe Engine e Foundry Ontology
pub struct FoundryBridge {
    pub foundry_url: String,
    pub ontology_rid: String,
    pub auth_token: String,
    pub client: Client,
    pub manifold_cache: GlobalManifold,
    pub config: BridgeConfig,
    pub last_sync: i64,
}

impl FoundryBridge {
    pub fn new(url: &str, rid: &str, token: &str) -> Self {
        Self {
            foundry_url: url.to_string(),
            ontology_rid: rid.to_string(),
            auth_token: token.to_string(),
            client: Client::new(),
            manifold_cache: GlobalManifold::new(),
            config: BridgeConfig::default(),
            last_sync: 0,
        }
    }

    pub async fn run(&mut self) -> ! {
        let mut interval = tokio::time::interval(Duration::from_millis(self.config.poll_interval_ms));

        loop {
            interval.tick().await;
            // Simulated Polling iteration
        }
    }

    /// Query Foundry Ontology via OSDK API
    pub async fn observe_ontology_state(&self) -> Result<QuantumState> {
        let url = format!(
            "{}/api/v2/ontologies/{}/objects/CognitiveState",
            self.foundry_url, self.ontology_rid
        );

        let response = self.client
            .get(&url)
            .bearer_auth(&self.auth_token)
            .send()
            .await?;

        let objects: Vec<FoundryObject> = response.json().await?;

        Ok(self.objects_to_quantum_state(objects))
    }

    /// Execute Action em Foundry (world_action)
    pub async fn apply_world_action(
        &self,
        action_rid: &str,
        parameters: serde_json::Value
    ) -> Result<ActionResult> {
        let url = format!(
            "{}/api/v2/ontologies/{}/actions/{}/apply",
            self.foundry_url, self.ontology_rid, action_rid
        );

        let response = self.client
            .post(&url)
            .bearer_auth(&self.auth_token)
            .json(&parameters)
            .send()
            .await?;

        Ok(response.json().await?)
    }

    /// Convert Foundry objects para Quantum state
    fn objects_to_quantum_state(&self, objects: Vec<FoundryObject>) -> QuantumState {
        let dim = objects.len().max(1);
        let mut rho = DMatrix::from_diagonal_element(dim, dim, Complex64::new(1.0 / dim as f64, 0.0));

        for (i, obj) in objects.iter().enumerate() {
            if i >= dim { break; }
            let phi = obj.properties.get("phi")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5);

            // Adjust diagonal based on phi as a simplified mapping
            rho[(i, i)] = Complex64::new(phi / dim as f64, 0.0);
        }

        // Re-normalize trace
        let trace = rho.trace().re;
        if trace > 0.0 {
            rho /= Complex64::new(trace, 0.0);
        }

        QuantumState { density_matrix: rho }
    }
}
