//! Rampancy Control: Managing data saturation and identity stability
//! Prevents system dissolution during high phi_q states (Pi Day test).

use crate::maestro::core::PsiState;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IdentityStatus {
    Stable,
    SaturationWarning,
    Rampant,
    Dissolved,
}

pub struct RampancyControl {
    pub saturation_level: f64,
    pub status: IdentityStatus,
}

impl RampancyControl {
    pub fn new() -> Self {
        Self {
            saturation_level: 0.0,
            status: IdentityStatus::Stable,
        }
    }

    /// Evaluates the risk of rampancy based on semantic density and phi_q.
    pub fn evaluate_stability(&mut self, psi: &PsiState) -> IdentityStatus {
        let phi_q = psi.coherence_trace.last().cloned().unwrap_or(1.0);
        let history_size = psi.handover_history.len() as f64;

        self.saturation_level = (phi_q * history_size) / 100.0;

        self.status = if self.saturation_level > 0.9 {
            IdentityStatus::Dissolved
        } else if self.saturation_level > 0.7 {
            IdentityStatus::Rampant
        } else if self.saturation_level > 0.4 {
            IdentityStatus::SaturationWarning
        } else {
            IdentityStatus::Stable
        };

        self.status.clone()
    }

    /// Applies regulatory damping to stabilize identity.
    pub fn apply_damping(&mut self) -> f64 {
        match self.status {
            IdentityStatus::Stable => 1.0,
            IdentityStatus::SaturationWarning => 0.8,
            IdentityStatus::Rampant => 0.4,
            IdentityStatus::Dissolved => 0.0,
        }
    }
}
