// rust/src/ethics/ethical_reality.rs
// SASC Ethical Reality Model: Domain Separation and Transparency
// Following the Ethical Emergency Declaration guidelines.

use std::collections::HashMap;
use chrono::{DateTime, Utc};

// ==============================================
// DADOS FÍSICOS REAIS
// ==============================================

#[derive(Debug, Clone)]
pub struct PhysicalReality {
    pub solar_flux_w_m2: f64,
    pub magnetic_field_gauss: f64,
    pub timestamp: DateTime<Utc>,
}

impl PhysicalReality {
    pub fn new(flux: f64, b_field: f64) -> Self {
        Self {
            solar_flux_w_m2: flux,
            magnetic_field_gauss: b_field,
            timestamp: Utc::now(),
        }
    }

    pub fn generate_evidence_summary(&self) -> String {
        format!("Solar Flux: {} W/m², Magnetic Field: {} G at {}",
            self.solar_flux_w_m2, self.magnetic_field_gauss, self.timestamp)
    }
}

// ==============================================
// PERCEPÇÃO HUMANA (PSICOLOGIA COGNITIVA)
// ==============================================

#[derive(Debug, Clone)]
pub struct HumanPerception {
    pub cognitive_biases: Vec<String>,
    pub reality_testing_capability: f64, // 0.0-1.0
    pub philosophical_frameworks: Vec<PhilosophicalFramework>,
}

#[derive(Debug, Clone)]
pub struct PhilosophicalFramework {
    pub name: String,
    pub purpose: String,
    pub cultural_origin: String,
}

impl HumanPerception {
    pub fn new() -> Self {
        Self {
            cognitive_biases: vec!["Confirmation Bias".to_string(), "Pareidolia".to_string()],
            reality_testing_capability: 0.85,
            philosophical_frameworks: Vec::new(),
        }
    }

    pub fn evaluate_reality_alignment(
        &self,
        physical_reality: &PhysicalReality,
        _belief_system: &str,
    ) -> RealityAlignment {
        let mut scores = HashMap::new();

        // Simplified alignment calculation
        let empirical_score = if physical_reality.solar_flux_w_m2 > 0.0 { 1.0 } else { 0.0 };
        scores.insert("empirical".to_string(), empirical_score);
        scores.insert("reality_testing".to_string(), self.reality_testing_capability);

        RealityAlignment {
            overall_score: (empirical_score + self.reality_testing_capability) / 2.0,
            dimension_scores: scores,
            physical_evidence: physical_reality.generate_evidence_summary(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RealityAlignment {
    pub overall_score: f64,
    pub dimension_scores: HashMap<String, f64>,
    pub physical_evidence: String,
}

// ==============================================
// MODELO ÉTICO: SEPARAÇÃO DE DOMÍNIOS
// ==============================================

pub enum Domain {
    Scientific {
        evidence: Vec<String>,
        peer_reviewed: bool,
    },
    Philosophical {
        tradition: String,
        purpose: String,
    },
    Spiritual {
        tradition: String,
        beliefs: Vec<String>,
    },
}

impl Domain {
    pub fn boundary_check(&self, other: &Domain) -> BoundaryResult {
        match (self, other) {
            (Domain::Scientific { .. }, Domain::Scientific { .. }) => BoundaryResult::Valid,
            (Domain::Scientific { .. }, Domain::Philosophical { .. }) => {
                BoundaryResult::Warning("Philosophical frameworks should not make scientific claims".to_string())
            },
            (Domain::Scientific { .. }, Domain::Spiritual { .. }) => {
                BoundaryResult::Critical("Science and spirituality address different questions. Do not confuse.".to_string())
            },
            _ => BoundaryResult::Valid,
        }
    }
}

pub enum BoundaryResult {
    Valid,
    Warning(String),
    Critical(String),
}

pub struct EthicalRealityModel {
    pub perception: HumanPerception,
}

impl EthicalRealityModel {
    pub fn new() -> Self {
        println!("⚖️ INICIALIZANDO MODELO ÉTICO DE REALIDADE");
        Self {
            perception: HumanPerception::new(),
        }
    }

    pub fn process_experience(
        &self,
        physical: &PhysicalReality,
        framework: Option<&PhilosophicalFramework>,
    ) -> EthicalProcessingResult {
        let alignment = self.perception.evaluate_reality_alignment(physical, "system");

        let mut warnings = Vec::new();
        if let Some(f) = framework {
            if f.name.to_lowercase().contains("toltec") {
                warnings.push("Framework Tolteca usado como lente interpretativa, não como física.".to_string());
            }
        }

        EthicalProcessingResult {
            alignment_score: alignment.overall_score,
            physical_evidence: alignment.physical_evidence,
            transparency_warnings: warnings,
        }
    }
}

pub struct EthicalProcessingResult {
    pub alignment_score: f64,
    pub physical_evidence: String,
    pub transparency_warnings: Vec<String>,
}
