use serde::{Deserialize, Serialize};
use std::fs;
use anyhow::{Result, Context};

#[derive(Debug, Serialize, Deserialize)]
pub struct Constitution {
    pub version: String,
    pub dimensions: u8,
    pub invariants: Vec<String>,
    pub parameters: Parameters,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Parameters {
    pub phi_target: f64,
    pub tau_max: f64,
    pub dimensional_stability: f64,
    pub ethical_threshold: f64,
    pub evolutionary_pace: String,
}

impl Constitution {
    pub fn load(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Falha ao ler constituição: {}", path))?;

        let constitution: Self = serde_json::from_str(&content)
            .with_context(|| "Falha ao parsear constituição")?;

        Ok(constitution)
    }

    pub fn validate(&self) -> Result<()> {
        if self.dimensions != 9 {
            anyhow::bail!("Constituição deve ter 9 dimensões");
        }

        if self.parameters.phi_target <= 1.0 {
            anyhow::bail!("Φ target deve ser > 1.0");
        }

        if self.parameters.tau_max <= 1.0 {
            anyhow::bail!("τ max deve ser > 1.0");
        }

        if self.invariants.len() < 7 {
            anyhow::bail!("Pelo menos 7 invariantes constitucionais requeridos");
        }

        Ok(())
    }
}
