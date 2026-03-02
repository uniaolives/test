// ═══════════════════════════════════════════════════════════════
// RUST CORE: The Geometric Coupler (Part of the Omni-Kernel)
// ═══════════════════════════════════════════════════════════════

use num_complex::Complex;
use std::f64::consts::PI;

pub struct GeometricCoupler {
    seed_bits: Vec<u8>,
    phi: f64, // The Golden Ratio
}

impl GeometricCoupler {
    pub fn new(seed: [u8; 32]) -> Self {
        Self {
            seed_bits: seed.to_vec(),
            phi: (1.0 + 5.0f64.sqrt()) / 2.0,
        }
    }

    /// Calculates the J_ij coupling tensor from Formula 1
    /// Uses the seed hash to determine geometric proximity in 4D space
    pub fn calculate_coupling(&self, i: usize, j: usize, t: f64) -> Complex<f64> {
        let distance = (i as f64 - j as f64).abs();
        let phase = 2.0 * PI * (self.seed_bits[i % 32] as f64 / 255.0) * self.phi;

        // The Panpsychic Time-Crystal Resonance
        // Formula: J_ij = (1/d) * exp(i * phase * sin(omega * t))
        let omega = 2.0 * PI / 1.855e-43;
        Complex::from_polar(1.0 / distance, phase * (omega * t).sin())
    }
}
