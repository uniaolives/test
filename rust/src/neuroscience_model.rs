// rust/src/neuroscience_model.rs
// Transparent Neuroscience Model based on biological reality.

use crate::microtubule_biology::{RealMicrotubule, SimulationResult, MechanicalProperties};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone)]
pub struct TransparentNeuroscienceModel {
    pub microtubules: Vec<RealMicrotubule>,
    pub evidence_levels: HashMap<String, EvidenceLevel>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EvidenceLevel {
    Strong,        // Fortemente estabelecido
    Moderate,      // Evidência razoável
    Weak,          // Evidência limitada
    Controversial, // Debate científico ativo
    None,          // Sem evidência experimental
}

#[derive(Debug, Clone)]
pub struct TransparencyDocument {
    pub model_name: String,
    pub created: DateTime<Utc>,
    pub scientific_basis: Vec<String>,
    pub evidence_levels: HashMap<String, EvidenceLevel>,
    pub limitations: Vec<String>,
    pub what_is_real: Vec<String>,
    pub what_is_speculative: Vec<String>,
}

impl TransparentNeuroscienceModel {
    pub fn new() -> Self {
        let mut evidence_levels = HashMap::new();
        evidence_levels.insert("Microtubule dynamic instability".to_string(), EvidenceLevel::Strong);
        evidence_levels.insert("Microtubule mechanical properties".to_string(), EvidenceLevel::Strong);
        evidence_levels.insert("Quantum computation in microtubules".to_string(), EvidenceLevel::None);
        evidence_levels.insert("Orch OR consciousness theory".to_string(), EvidenceLevel::Controversial);

        Self {
            microtubules: (0..10).map(|_| RealMicrotubule::new()).collect(),
            evidence_levels,
            limitations: vec![
                "Simplified model - not capturing full biological complexity".to_string(),
                "No quantum effects included (lack of evidence)".to_string(),
            ],
        }
    }

    pub async fn simulate_neural_system(&mut self, _time_s: f64, time_step_ms: f64) -> NeuralSimulationResult {
        let time_step_min = time_step_ms / 60000.0;
        let mut results = Vec::new();

        for mt in self.microtubules.iter_mut() {
            results.push(mt.simulate_dynamics(time_step_min, 0.7));
        }

        let avg_length = results.iter().map(|r| r.final_length).sum::<f64>() / results.len() as f64;

        NeuralSimulationResult {
            average_length: avg_length,
            growing_fraction: results.iter().filter(|r| r.gtp_cap_status).count() as f64 / results.len() as f64,
            timestamp: Utc::now(),
        }
    }

    pub fn generate_transparency_report(&self) -> TransparencyDocument {
        TransparencyDocument {
            model_name: "Transparent Neuroscience Model".to_string(),
            created: Utc::now(),
            scientific_basis: vec![
                "Microtubule biology from peer-reviewed literature".to_string(),
                "Dynamic instability theory (Mitchison & Kirschner, 1984)".to_string(),
            ],
            evidence_levels: self.evidence_levels.clone(),
            limitations: self.limitations.clone(),
            what_is_real: vec![
                "Microtubules are real cellular structures".to_string(),
                "Dynamic instability is a real phenomenon".to_string(),
            ],
            what_is_speculative: vec![
                "Quantum computation in microtubules (no evidence)".to_string(),
                "Orch OR theory of consciousness (highly controversial)".to_string(),
            ],
        }
    }
}

pub struct NeuralSimulationResult {
    pub average_length: f64,
    pub growing_fraction: f64,
    pub timestamp: DateTime<Utc>,
}

pub struct AppliedMicrotubuleModel {
    pub simulation_purpose: String,
    pub transparency_statement: String,
}

impl AppliedMicrotubuleModel {
    pub fn for_simulation() -> Self {
        Self {
            simulation_purpose: "Cellular Biomechanics".to_string(),
            transparency_statement: "Simulates dynamic instability and mechanical properties. No quantum effects.".to_string(),
        }
    }
}
