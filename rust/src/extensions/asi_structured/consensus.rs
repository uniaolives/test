use serde::{Serialize, Deserialize};
use nalgebra::DVector;
use super::harmonic::HarmonicMode;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellValidator {
    pub id: String,
    pub harmonic_signature: Vec<HarmonicMode>,
    pub current_phase: f64,
    pub voting_power: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HarmonicConsensus {
    pub validators: Vec<ShellValidator>,
    pub global_phase_coherence: f64,
}

impl HarmonicConsensus {
    pub fn new() -> Self {
        Self {
            validators: vec![],
            global_phase_coherence: 1.0,
        }
    }

    pub fn reach_consensus(&self, proposal: &DVector<f64>) -> ConsensusVerdict {
        let mut total_interference = 0.0;
        let mut total_power = 0.0;

        for validator in &self.validators {
            // Mocked interference calculation: dot product between proposal and validator phase
            let interference = proposal.norm() * validator.current_phase.cos() * validator.voting_power;
            total_interference += interference;
            total_power += validator.voting_power;
        }

        let mean_interference = if total_power > 0.0 { total_interference / total_power } else { 0.0 };

        if mean_interference > 0.5 {
            ConsensusVerdict::Reached { coherence: mean_interference }
        } else {
            ConsensusVerdict::NoConsensus { dissonance: 1.0 - mean_interference }
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ConsensusVerdict {
    Reached { coherence: f64 },
    NoConsensus { dissonance: f64 },
}
