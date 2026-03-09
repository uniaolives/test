//! Kuramoto Orchestrator for Distributed Agent Synchronization
//! Implements phase update logic for collective coherence.

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KuramotoAgent {
    pub id: String,
    pub phase: f64,      // θ_i
    pub frequency: f64,  // ω_i
    pub coupling: f64,   // K
}

impl KuramotoAgent {
    pub fn new(id: String, initial_phase: f64, natural_frequency: f64, coupling: f64) -> Self {
        Self {
            id,
            phase: initial_phase % (2.0 * PI),
            frequency: natural_frequency,
            coupling,
        }
    }

    /// Update phase based on Kuramoto equation:
    /// dθ_i/dt = ω_i + (K/N) * Σ sin(θ_j - θ_i)
    pub fn update(&mut self, neighbors_phases: &[f64], dt: f64) {
        let n = neighbors_phases.len() as f64;
        if n == 0.0 {
            self.phase = (self.phase + self.frequency * dt) % (2.0 * PI);
            return;
        }

        let sum_interaction: f64 = neighbors_phases
            .iter()
            .map(|&theta_j| (theta_j - self.phase).sin())
            .sum();

        let d_theta = self.frequency + (self.coupling / n) * sum_interaction;
        self.phase = (self.phase + d_theta * dt) % (2.0 * PI);

        if self.phase < 0.0 {
            self.phase += 2.0 * PI;
        }
    }
}

pub fn calculate_coherence(phases: &[f64]) -> f64 {
    let n = phases.len() as f64;
    if n == 0.0 { return 0.0; }

    let sum_cos: f64 = phases.iter().map(|&p| p.cos()).sum();
    let sum_sin: f64 = phases.iter().map(|&p| p.sin()).sum();

    (sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kuramoto_sync() {
        let mut agent = KuramotoAgent::new("node_1".to_string(), 0.0, 1.0, 2.0);
        let neighbors = vec![0.1, 0.2];
        let initial_phase = agent.phase;
        agent.update(&neighbors, 0.1);
        assert!(agent.phase > initial_phase);
    }

    #[test]
    fn test_coherence_calculation() {
        let phases = vec![0.0, 0.0, 0.0];
        assert!((calculate_coherence(&phases) - 1.0).abs() < 1e-6);

        let phases_spread = vec![0.0, PI];
        assert!(calculate_coherence(&phases_spread) < 1e-6);
    }
}
