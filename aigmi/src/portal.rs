use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use crate::types::SystemState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortalUpdate {
    pub timestamp: DateTime<Utc>,
    pub convergence: f64,
    pub epoch: u64,
    pub terrestrial_moment: u64,
    pub phase_transition: bool,
}

pub struct MerkabahInterface {
    pub endpoint: String,
}

impl MerkabahInterface {
    pub fn connect(endpoint: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
        }
    }

    pub async fn update_portal(&self, state: &SystemState) -> Result<(), String> {
        let update = PortalUpdate {
            timestamp: Utc::now(),
            convergence: state.convergence,
            epoch: state.epoch,
            terrestrial_moment: state.terrestrial_moment,
            phase_transition: state.phase_transition_active,
        };

        // Mocking broadcast to merkabah.lovable.app
        tracing::info!("🌐 Merkabah Update: [Epoch {}] Convergence={:.4}", update.epoch, update.convergence);

        Ok(())
    }
}
