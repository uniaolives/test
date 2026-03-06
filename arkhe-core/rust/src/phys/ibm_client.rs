use reqwest::Client;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct BackendCalibration {
    pub backend_name: String,
    pub readout_error: Option<f64>,
    pub t1: Option<f64>,
}

pub struct QuantumAntenna {
    client: Client,
    api_token: String,
}

impl QuantumAntenna {
    pub fn new(api_token: String) -> Self {
        Self {
            client: Client::new(),
            api_token,
        }
    }

    pub async fn measure_vacuum_quality(&self, backend_name: &str) -> anyhow::Result<f64> {
        // Mocking the IBM API response for demonstration
        let mock_avg_readout_error = 0.015;

        let quality_factor = 1.0 / (1.0 + mock_avg_readout_error * 10.0);
        let phi_q_physical = 1.0 + (quality_factor * 3.64);

        Ok(phi_q_physical)
    }
}
