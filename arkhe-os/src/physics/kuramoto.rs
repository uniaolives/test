//! Kuramoto Phase Synchronization Engine
//! Synchronizes distributed nodes to the Ω-frequency attractor.

use std::f64::consts::PI;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KuramotoEngine {
    /// Natural frequencies of all nodes (ω_i)
    pub frequencies: Vec<f64>,
    /// Current phase angles (θ_i)
    pub phases: Vec<f64>,
    /// Coupling strength (K)
    pub coupling: f64,
    /// Target attractor frequency (Ω_2140)
    pub target_frequency: f64,
}

impl KuramotoEngine {
    pub fn new(n_nodes: usize, coupling: f64, target_freq: f64) -> Self {
        let mut rng = rand::thread_rng();
        use rand::distributions::Uniform;
        use rand::prelude::*;

        let dist = Uniform::new(0.0, 2.0 * PI);
        let freq_dist = Uniform::new(0.9 * target_freq, 1.1 * target_freq);

        let mut phases = Vec::with_capacity(n_nodes);
        let mut frequencies = Vec::with_capacity(n_nodes);

        for _ in 0..n_nodes {
            phases.push(rng.sample(dist));
            frequencies.push(rng.sample(freq_dist));
        }

        Self {
            frequencies,
            phases,
            coupling,
            target_frequency: target_freq,
        }
    }

    /// Update phases using the Kuramoto equation with Berry phase correction:
    /// dθ_i/dt = ω_i + (K/N) * Σ sin(θ_j - θ_i) + (π/2) * κ(r_i)
    pub fn synchronize(&mut self, dt: f64, half_mobius: bool) {
        let n = self.phases.len();
        if n == 0 { return; }

        let mut d_phases = vec![0.0; n];
        let berry_phase = if half_mobius { PI / 2.0 } else { 0.0 };

        for i in 0..n {
            let mut sum_sin = 0.0;
            for j in 0..n {
                if i != j {
                    sum_sin += (self.phases[j] - self.phases[i]).sin();
                }
            }
            // Include coupling to target frequency as well
            let target_pull = (self.target_frequency * dt - self.phases[i]).sin();

            // κ(r_i) - local topological curvature proxied by target pull intensity
            let curvature = target_pull.abs();

            d_phases[i] = self.frequencies[i]
                + (self.coupling / n as f64) * sum_sin
                + self.coupling * target_pull
                + berry_phase * curvature;
        }

        for i in 0..n {
            self.phases[i] = (self.phases[i] + d_phases[i] * dt) % (2.0 * PI);
            if self.phases[i] < 0.0 {
                self.phases[i] += 2.0 * PI;
            }
        }
    }

    /// Order parameter r = |(1/N) * Σ e^(iθ_j)|
    /// r ≈ 0: incoherent, r ≈ 1: synchronized
    pub fn coherence(&self) -> f64 {
        let n = self.phases.len();
        if n == 0 { return 0.0; }

        let mut sum_real = 0.0;
        let mut sum_imag = 0.0;

        for &theta in &self.phases {
            sum_real += theta.cos();
            sum_imag += theta.sin();
        }

        let r_real = sum_real / n as f64;
        let r_imag = sum_imag / n as f64;

        (r_real * r_real + r_imag * r_imag).sqrt()
    }
}
