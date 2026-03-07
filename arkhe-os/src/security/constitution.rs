//! Constitutional Guard for thermodynamic enforcement.
//! Ensures H ≤ 1 (thermodynamic sustainability) to arrive at 2140 intact.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum ConstitutionalBreach {
    #[error("Thermodynamic Breach (H > 1.0): current_h = {current_h}")]
    ElenaViolation {
        current_h: f64,
        max_allowed: f64,
        consequence: String,
    },
    #[error("Violation of Principle P{index}: {description}")]
    PrincipleViolation {
        index: usize,
        description: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstitutionalGuard {
    /// Current H-value (consumption/production ratio)
    pub h: f64,
    /// P1-P11 compliance matrix
    pub principles: [bool; 11],
}

impl ConstitutionalGuard {
    pub fn new() -> Self {
        Self {
            h: 0.1,
            principles: [true; 11],
        }
    }

    /// Enforce H ≤ 1 (Elena Constant)
    pub fn verify_h_limit(&self) -> Result<(), ConstitutionalBreach> {
        if self.h > 1.0 {
            return Err(ConstitutionalBreach::ElenaViolation {
                current_h: self.h,
                max_allowed: 1.0,
                consequence: "Thermodynamic unsustainability. 2140 unreachable.".to_string(),
            });
        }
        Ok(())
    }

    /// Verify all 11 principles
    pub fn full_compliance(&self) -> bool {
        self.principles.iter().all(|&p| p) && self.h <= 1.0
    }

    /// Adjust H-value based on system activity
    pub fn update_h(&mut self, rate: f64, coherence: f64, interest: f64) {
        // H is roughly proportional to handover rate and inversely proportional to coherence
        // H = (rate * interest) / (coherence * capacity)
        self.h = (rate * (1.0 + interest)) / (coherence + 1e-6);

        // H is capped at 0.0 but can exceed 1.0
        if self.h < 0.0 { self.h = 0.0; }
    }
}
