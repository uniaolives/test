use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VetoDetector {
    pub baseline_coherence: f64,
    pub threshold_drop: f64, // If λ₂ drops by this much, it's a veto
}

impl VetoDetector {
    pub fn new(baseline: f64, threshold: f64) -> Self {
        Self {
            baseline_coherence: baseline,
            threshold_drop: threshold,
        }
    }

    pub fn check(&self, current_coherence: f64) -> Option<VetoType> {
        let drop = self.baseline_coherence - current_coherence;

        if drop > self.threshold_drop {
            // Coherence dropped significantly -- human is rejecting
            Some(VetoType::Intuitive)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VetoType {
    Logical,    // "The logic doesn't hold."
    Axiomatic,  // "The premise is wrong."
    Intuitive,  // "Something feels off."
}
