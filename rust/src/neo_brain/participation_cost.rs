use std::time::{SystemTime, Duration};
use serde::{Serialize, Deserialize};
use crate::neo_brain::types::BioMetadata;

/// Representa uma prova de esforço humano (não computacional)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandWorkProof {
    /// Unidades de "Moeda-de-Obra" gastas. Valor base: 1.0 = esforço médio humano.
    pub cost_units: f32,
    /// Metadados biométricos comprovando origem humana
    pub bio_metadata: BioMetadata,
    /// Timestamp do esforço (precisa ser recente)
    pub timestamp: SystemTime,
    /// Hash criptográfico do compromisso (evita replay)
    pub commitment_hash: [u8; 32],
}

impl HandWorkProof {
    /// Calcula o custo total ajustado por inflação e delta de sinal
    /// Fórmula: adjusted_cost = base_cost * (1 + inflation_rate) - signal_delta
    pub fn adjusted_cost(&self, current_inflation: f32) -> f32 {
        let base = self.cost_units;
        let signal_delta = self.bio_metadata.signal_delta.unwrap_or(0.0);

        // A inflação protege contra ataques de acumulação
        let inflated_cost = base * (1.0 + current_inflation.clamp(0.0, 0.05)); // Max 5% inflação

        // Delta negativo de sinal reduz custo (incentivo), positivo aumenta
        inflated_cost - signal_delta
    }

    /// Verifica se a prova é válida (custo suficiente e recente)
    pub fn is_valid(&self, required_cost: f32) -> bool {
        let inflation = self.calculate_current_inflation();
        let adjusted = self.adjusted_cost(inflation);

        adjusted >= required_cost &&
        self.timestamp.elapsed().unwrap_or(Duration::from_secs(999999)) < Duration::from_secs(24 * 3600)
    }

    fn calculate_current_inflation(&self) -> f32 {
        0.02 // Mock
    }
}
