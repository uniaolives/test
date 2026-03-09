//! Ω+248: The Teknet Standard Model - Retrocausal Particle Taxonomy
//! Implementation of the speculative taxonomy: Φ, Λ, Ξ, Σ.

use serde::{Deserialize, Serialize};

/// Φ (Phi) Particle - The "Chronon" of Hanover.
/// The quantum of informational handover.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiParticle {
    pub mass_effective: f64, // m_Φ = (ħ * λ₂) / c²
    pub spin: f64,           // s = 1/2 * sgn(H)
    pub charge_retro: f64,   // Q_Φ = 1/L'(E,1) * sqrt(N)
    pub velocity: f64,       // v_Φ > c (tachyonic)
}

impl PhiParticle {
    /// Calculates properties for the Phi particle.
    pub fn new(lambda2: f64, h: f64, l_prime: f64, n: u64) -> Self {
        // Using natural units where ħ = c = 1 for simplicity in taxonomy definition
        let mass_effective = lambda2;
        let spin = 0.5 * h.signum();
        let charge_retro = (n as f64).sqrt() / l_prime;
        let velocity = 1.618; // Speculative superluminal velocity (v > 1.0)

        Self {
            mass_effective,
            spin,
            charge_retro,
            velocity,
        }
    }
}

/// Λ (Lambda) Particle - The "Temporal Antimatter Link".
/// A particle of matter entangled with its own temporal antiparticle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaParticle {
    pub mass: f64,       // m_Λ = m0 * |ψ_present|² / |ψ_future|²
    pub lifetime: f64,   // τ_Λ = ħ / (ΔE * ΔS)
    pub charge: f64,     // Q_Λ = ±e * cos(θ_Berry)
}

impl LambdaParticle {
    pub fn new(m0: f64, psi_p: f64, psi_f: f64, delta_e: f64, delta_s: f64, berry_phase: f64) -> Self {
        let mass = m0 * (psi_p.powi(2) / (psi_f.powi(2) + 1e-9));
        let lifetime = 1.0 / (delta_e * delta_s + 1e-9);
        let charge = berry_phase.cos(); // Simplified charge modulation

        Self {
            mass,
            lifetime,
            charge,
        }
    }
}

/// Σ (Sigma) Particle - The "Novikov Consistency Agent".
/// The quantum of auto-consistency, enforcing boundary conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SigmaParticle {
    pub mass: f64,      // m_Σ = ħ / (c * τ_loop)
    pub charge: f64,    // Q_Σ = ∮ A_μ dx^μ (gauge invariant flux)
    pub spin: f64,      // s = ħ/2 * sgn(∮ R_μν dx^μ dx^ν)
}

impl SigmaParticle {
    pub fn new(tau_loop: f64, flux: f64, curvature_sgn: f64) -> Self {
        let mass = 1.0 / (tau_loop + 1e-9);
        let charge = flux;
        let spin = 0.5 * curvature_sgn;

        Self {
            mass,
            charge,
            spin,
        }
    }
}

/// The Teknet Standard Model: Unifying the retrocausal particles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeknetStandardModel {
    pub phi: PhiParticle,
    pub lambda: LambdaParticle,
    pub xi: crate::physics::xi_particle::XiParticle,
    pub sigma: SigmaParticle,
}

impl TeknetStandardModel {
    pub fn check_coherence(&self) -> f64 {
        // System coherence is a function of the constituent particles' properties
        (self.phi.mass_effective * self.xi.coherence * self.lambda.lifetime.tanh()).min(1.0)
    }
}
