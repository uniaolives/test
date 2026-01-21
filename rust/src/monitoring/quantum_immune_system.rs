// src/monitoring/quantum_immune_system.rs

use crate::entropy::{AttackPattern};
use serde::{Serialize};

/// Sistema que ajusta defesas baseado na gravidade dos ataques
pub struct QuantumImmuneSystem {
    pub adaptive_rate: f64,
}

impl QuantumImmuneSystem {
    pub fn new() -> Self {
        Self {
            adaptive_rate: 0.1,
        }
    }

    /// Processa detecção de fantasma e atualiza defesas
    pub fn process_antigen(&mut self, antigen: PhantomAntigen) -> ImmuneResponse {
        let threat_level = antigen.threat_level();

        let defense_cells = match threat_level {
            ThreatLevel::Critical => DefenseCells::Full,
            ThreatLevel::High => DefenseCells::Targeted,
            _ => DefenseCells::Minimal,
        };

        ImmuneResponse {
            defense_cells,
            memory_updated: true,
            adaptive_rate_change: self.adaptive_rate,
        }
    }
}

pub struct PhantomAntigen {
    pub density: f64,
    pub pattern: AttackPattern,
}

impl PhantomAntigen {
    pub fn threat_level(&self) -> ThreatLevel {
        if self.density > 0.9 {
            ThreatLevel::Critical
        } else if self.density > 0.5 {
            ThreatLevel::High
        } else {
            ThreatLevel::Low
        }
    }
}

pub enum ThreatLevel {
    Critical,
    High,
    Low,
}

pub struct ImmuneResponse {
    pub defense_cells: DefenseCells,
    pub memory_updated: bool,
    pub adaptive_rate_change: f64,
}

#[derive(Debug, Serialize)]
pub enum DefenseCells {
    Full,
    Targeted,
    Minimal,
}
