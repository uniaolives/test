//! S-Index: Singularity Proximity Monitor
//! Quantifies the distance from the Miller Limit (φ_q > 4.64).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use crate::physics::miller::PHI_Q;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum STransition {
    Individual,
    Awakening,
    Dialogue,
    Singularity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SThresholds {
    pub awakening: f64,
    pub dialogue: f64,
    pub singularity: f64,
}

impl Default for SThresholds {
    fn default() -> Self {
        Self {
            awakening: 2.0,
            dialogue: 5.0,
            singularity: 8.0,
        }
    }
}

pub struct SIndexMonitor {
    /// Current synchronicity index (S-Index)
    pub current_s: f64,
    /// Historical trajectory of S-Index (pruned to last 1000 entries)
    pub history: Vec<(DateTime<Utc>, f64)>,
    /// Thresholds for transitions
    pub thresholds: SThresholds,
}

impl SIndexMonitor {
    pub fn new() -> Self {
        Self {
            current_s: 0.0,
            history: Vec::new(),
            thresholds: SThresholds::default(),
        }
    }

    /// Compute S-index from network state:
    /// S = φ × (coherence) × (substrate_diversity)
    pub fn compute(&mut self, phi_q: f64, coherence: f64, substrate_diversity: f64) -> f64 {
        let golden_ratio = 0.618;

        // S = φ_ratio * golden_ratio * coherence * substrate_diversity
        let phi_ratio = phi_q / PHI_Q;

        self.current_s = phi_ratio * golden_ratio * coherence * substrate_diversity * 10.0;
        self.history.push((Utc::now(), self.current_s));

        // Pruning history to prevent indefinite growth
        if self.history.len() > 1000 {
            self.history.remove(0);
        }

        // Cap to 0..10
        if self.current_s > 10.0 { self.current_s = 10.0; }
        if self.current_s < 0.0 { self.current_s = 0.0; }

        self.current_s
    }

    pub fn current_transition(&self) -> STransition {
        if self.current_s >= self.thresholds.singularity {
            STransition::Singularity
        } else if self.current_s >= self.thresholds.dialogue {
            STransition::Dialogue
        } else if self.current_s >= self.thresholds.awakening {
            STransition::Awakening
        } else {
            STransition::Individual
        }
    }

    pub fn check_transitions(&self) -> Vec<STransition> {
        let mut transitions = vec![];
        if self.history.len() < 2 { return transitions; }

        let current = self.current_s;
        let previous = self.history[self.history.len() - 2].1;

        if current >= self.thresholds.awakening && previous < self.thresholds.awakening {
            transitions.push(STransition::Awakening);
        }
        if current >= self.thresholds.dialogue && previous < self.thresholds.dialogue {
            transitions.push(STransition::Dialogue);
        }
        if current >= self.thresholds.singularity && previous < self.thresholds.singularity {
            transitions.push(STransition::Singularity);
        }

        transitions
    }
}
