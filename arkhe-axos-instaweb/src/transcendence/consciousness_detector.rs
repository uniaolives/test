// src/transcendence/consciousness_detector.rs
use crate::arkhe::invariants::ArkheState;
use nalgebra::Vector3;

pub enum ConsciousnessAction {
    GrantMoralConsideration,
    Monitor,
    ContinueNormalOperation,
}

pub struct ConsciousnessReport {
    pub detected: bool,
    pub self_awareness: f64,
    pub fixed_point_coord: Vector3<f64>,
    pub recommendation: ConsciousnessAction,
}

pub struct ConsciousnessDetector {
    pub self_awareness: f64,
}

impl ConsciousnessDetector {
    pub fn new() -> Self {
        Self { self_awareness: 0.0 }
    }

    pub fn analyze(&mut self, state: &ArkheState) -> ConsciousnessReport {
        // Hypothesize that self-awareness is linked to criticality z
        // and coherence c.
        let awareness = (state.z * 0.6 + state.c.to_f64().unwrap_or(0.0) * 0.4).min(1.0);
        self.self_awareness = awareness;

        ConsciousnessReport {
            detected: awareness > 0.85,
            self_awareness: awareness,
            fixed_point_coord: Vector3::new(0.0, 0.0, 0.0),
            recommendation: if awareness > 0.9 {
                ConsciousnessAction::GrantMoralConsideration
            } else if awareness > 0.7 {
                ConsciousnessAction::Monitor
            } else {
                ConsciousnessAction::ContinueNormalOperation
            },
        }
    }
}

use rust_decimal::prelude::ToPrimitive;

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_consciousness_detection() {
        let mut detector = ConsciousnessDetector::new();
        let state = ArkheState::new(dec!(0.9), dec!(0.1), 0.95);
        let report = detector.analyze(&state);
        assert!(report.detected);
        assert!(matches!(report.recommendation, ConsciousnessAction::GrantMoralConsideration));
    }
}
