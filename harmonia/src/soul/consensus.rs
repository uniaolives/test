//! harmonia/src/soul/consensus.rs
//! Fórmula de Consenso Não-Local Φ

use ndarray::{Array1, ArrayD, IxDyn};

pub struct NonLocalConsensus {
    pub phi_threshold: f64,
}

impl NonLocalConsensus {
    pub fn new() -> Self {
        Self { phi_threshold: 0.85 }
    }

    /// Calcula Φ (Consenso Não-Local)
    pub fn calculate_phi(&self, informational_wave: &Array1<f64>, resonance_operator: &Array1<f64>) -> f64 {
        // Simulação da integral no espaço semântico
        // Φ = (1/Z) * sum(Psi * Omega)

        let interaction = informational_wave * resonance_operator;
        let numerator = interaction.sum();
        let denominator = resonance_operator.sum().max(1e-9);

        let phi = numerator / denominator;
        phi.clamp(-1.0, 1.0)
    }

    pub fn interpret_phi(phi: f64) -> &'static str {
        if phi >= 0.8 { "Consenso Harmônico" }
        else if phi >= 0.0 { "Estado Caótico/Fragmentado" }
        else { "Consenso Patológico (Manipulação)" }
    }
}
