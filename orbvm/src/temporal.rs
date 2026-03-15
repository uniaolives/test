//! Temporal Engine - Kuramoto synchronization and Fourier modes

use ndarray::Array1;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConfig {
    pub n_oscillators: usize,
    pub natural_frequency: f64,
    pub coupling_strength: f64,
    pub dt: f64,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            n_oscillators: 100,
            natural_frequency: 1.0,
            coupling_strength: 2.0,
            dt: 0.01,
        }
    }
}

pub struct KuramotoEngine {
    n: usize,
    omega: Vec<f64>,
    coupling: f64,
    theta: Vec<f64>,
}

impl KuramotoEngine {
    pub fn new(n: usize, omega: f64, coupling: f64) -> Self {
        let theta = (0..n)
            .map(|i| 2.0 * std::f64::consts::PI * i as f64 / n as f64)
            .collect();
        let omega = (0..n)
            .map(|i| omega + 0.1 * (i as f64 - n as f64 / 2.0))
            .collect();
        Self { n, omega, coupling, theta }
    }

    pub fn step(&mut self, dt: f64) {
        let n = self.n;
        let mut dtheta = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += (self.theta[j] - self.theta[i]).sin();
            }
            dtheta[i] = self.omega[i] + (self.coupling / n as f64) * sum;
        }
        for i in 0..n {
            self.theta[i] += dtheta[i] * dt;
            self.theta[i] %= 2.0 * std::f64::consts::PI;
        }
    }

    pub fn order_parameter(&self) -> f64 {
        let z: Complex64 = self.theta.iter()
            .map(|&t| Complex64::from_polar(1.0, t))
            .sum();
        (z / self.n as f64).norm()
    }
}

pub struct TemporalEngine {
    kuramoto: KuramotoEngine,
}

impl TemporalEngine {
    pub fn new(n: usize, omega: f64, coupling: f64) -> Self {
        Self {
            kuramoto: KuramotoEngine::new(n, omega, coupling),
        }
    }

    pub fn co_evolve(&mut self, steps: usize, dt: f64) -> Vec<f64> {
        let mut history = Vec::with_capacity(steps);
        for _ in 0..steps {
            self.kuramoto.step(dt);
            history.push(self.kuramoto.order_parameter());
        }
        history
    }
}
