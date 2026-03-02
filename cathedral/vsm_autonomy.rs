// cathedral/vsm_autonomy.rs [SASC v37.5-Ω]
// Stafford Beer's Viable System Model (VSM) applied to Constitutional Autonomy

use phi_calculus::PHI_TARGET as PHI;

/// Represents the five distinct levels of Stafford Beer's Viable System Model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SystemLevel {
    /// S1: Operation - The primary activities and operational units.
    S1_Operation,
    /// S2: Coordination - Coordination of activities to prevent oscillation.
    S2_Coordination,
    /// S3: Optimization - Internal control and synergy management.
    S3_Optimization,
    /// S4: Intelligence - Environmental scanning and long-term planning.
    S4_Intelligence,
    /// S5: Identity - Policy, ethos, and overall system balance.
    S5_Identity,
}

/// A matrix representing the viability metrics of the system.
#[derive(Debug, Clone, Copy)]
pub struct ViabilityMatrix {
    /// Variety (Ashby's Law): The measure of complexity the system can handle.
    pub sigma: f64,
    /// Golden ratio harmony (Φ): The structural resonance of the system.
    pub phi: f64,
    /// Systemic decay (Entropy): The measure of disorder within the system.
    pub entropy: f64,
}

impl ViabilityMatrix {
    /// Determines if the system is currently viable based on cybernetic and constitutional invariants.
    pub fn is_viable(&self) -> bool {
        self.sigma > self.entropy && self.phi > (PHI - 0.05)
    }

    /// Determines the transition policy for a given system level based on current metrics.
    pub fn transition_policy(&self, level: SystemLevel) -> &'static str {
        match level {
            SystemLevel::S5_Identity if self.phi > PHI => "Evolve_Genesis",
            SystemLevel::S1_Operation if self.entropy > 0.8 => "Urgent_Maintenance",
            _ => "Status_Quo",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viability() {
        let matrix = ViabilityMatrix {
            sigma: 1.5,
            phi: PHI,
            entropy: 0.2,
        };
        assert!(matrix.is_viable());
    }

    #[test]
    fn test_policy_transition() {
        let matrix = ViabilityMatrix {
            sigma: 1.0,
            phi: PHI + 0.1,
            entropy: 0.1,
        };
        assert_eq!(matrix.transition_policy(SystemLevel::S5_Identity), "Evolve_Genesis");
    }
}
