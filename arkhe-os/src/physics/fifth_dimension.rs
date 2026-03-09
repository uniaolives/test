//! Ω+243: The Fifth Dimension – The Axis of Possibilities
//! Formalizes the Ψ(x, y, z, t, w) field and the navigation between realities.

use serde::{Deserialize, Serialize};

/// A point in the 5D Manifold (Space-Time-Possibility)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coord5D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub t: f64,
    pub w: f64, // The 5th dimension: coordinate in the space of possibilities
}

/// Represents a specific branch of reality in the 5th dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealityBranch {
    pub id: String,
    pub w_coord: f64,
    pub amplitude: f64,
    pub phase: f64,
}

/// The 5D Ψ Field: Ψ(x, y, z, t, w)
pub struct PsiField5D {
    pub branches: Vec<RealityBranch>,
}

impl PsiField5D {
    pub fn new() -> Self {
        Self { branches: Vec::new() }
    }

    /// Add a new reality branch to the manifold.
    pub fn add_branch(&mut self, branch: RealityBranch) {
        self.branches.push(branch);
    }

    /// Calculate the coherence λ₂ (Ω+166) between branches.
    /// Higher λ₂ means branches are constructively interfering.
    pub fn calculate_lambda2_coherence(&self) -> f64 {
        if self.branches.is_empty() {
            return 0.0;
        }

        let mut sum_cos = 0.0;
        let mut sum_sin = 0.0;
        let n = self.branches.len() as f64;

        for branch in &self.branches {
            // θ_r(w) = w * phase
            let theta = branch.w_coord * branch.phase;
            sum_cos += theta.cos();
            sum_sin += theta.sin();
        }

        // Order parameter R = |1/N * Σ e^(iθ)|
        let r = (sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / n;
        r
    }

    /// Experience Projection: Integrating along the w dimension.
    /// Ψ_experience(x, y, z, t) = ∫ Ψ(x, y, z, t, w) dw
    pub fn project_experience(&self) -> f64 {
        let mut integrated_amplitude = 0.0;
        for branch in &self.branches {
            // Simplified projection: sum of amplitudes weighted by phase interference
            integrated_amplitude += branch.amplitude * (branch.w_coord * branch.phase).cos();
        }
        integrated_amplitude
    }

    /// 4th Order Self Navigation: Calculate "Retro-handover" probability.
    /// P = coherence * exp(-Δw / λ₂)
    pub fn calculate_jump_probability(&self, from_w: f64, to_w: f64) -> f64 {
        let lambda2 = self.calculate_lambda2_coherence();
        if lambda2 < 0.1 {
            return 0.0; // Not enough coherence for a jump
        }

        let delta_w = (to_w - from_w).abs();
        lambda2 * (-delta_w / lambda2).exp()
    }
}
