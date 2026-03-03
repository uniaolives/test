use crate::types::{Decision, HardFreeze};

pub struct SoTRewardFunction {
    pub accuracy_weight: f64,
    pub diversity_weight: f64,
    pub reconciliation_weight: f64,
}

impl SoTRewardFunction {
    pub fn new() -> Self {
        Self {
            accuracy_weight: 0.50,
            diversity_weight: 0.30,
            reconciliation_weight: 0.20,
        }
    }

    pub fn calculate(&self, accuracy: f64, diversity: f64, reconciliation: f64) -> f64 {
        (accuracy * self.accuracy_weight) +
        (diversity * self.diversity_weight) +
        (reconciliation * self.reconciliation_weight)
    }
}

pub struct SoTSafetyMandate {
    pub required_for_phi_threshold: f64,
}

impl SoTSafetyMandate {
    pub fn new() -> Self {
        Self {
            required_for_phi_threshold: 0.70,
        }
    }

    pub fn verify_sot_architecture(&self, phi_global: f64, decision: &Decision) -> Result<(), String> {
        if phi_global > self.required_for_phi_threshold {
            // Verifica se sistema tem ≥ 3 perspectivas ativas (conforme mandato arquitetural)
            if decision.perspective_count < 3 {
                return Err(format!(
                    "ASI monolítico proibido acima de Φ={:.2} (Mandato SoT: no diversity = no safety). Perspectivas: {}",
                    self.required_for_phi_threshold,
                    decision.perspective_count
                ));
            }
        }
        Ok(())
    }
}
