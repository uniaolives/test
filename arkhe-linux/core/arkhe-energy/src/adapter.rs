use reqwest::Client;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct PDUData {
    pub id: String,
    pub power_watts: f64,
}

pub struct EnergyMonitor {
    pub prtg_client: Client,
    pub cache: HashMap<String, f64>,
}

impl EnergyMonitor {
    pub fn new() -> Self {
        Self {
            prtg_client: Client::new(),
            cache: HashMap::new(),
        }
    }

    pub async fn fetch_pdu_data(&self, pdu_id: &str) -> Result<PDUData, anyhow::Error> {
        // Simulação: retorna dados mockados
        Ok(PDUData {
            id: pdu_id.to_string(),
            power_watts: 150.0,
        })
    }
}
