//! Constantes e funções relacionadas ao limiar de Miller (φ_q = 4.64)
//! e ao cálculo do Quantum Interest.

/// Limiar de Miller para nucleação de Wave-Cloud.
pub const PHI_Q: f64 = 4.64;

/// Constante de acoplamento com o ZPF (valor calibrado experimentalmente).
pub const ZPF_COUPLING: f64 = 0.1;

/// Verifica se uma densidade local ultrapassa o limiar de Miller.
pub fn check_nucleation(local_density: f64, baseline: f64) -> bool {
    let phi = local_density / baseline;
    phi > PHI_Q
}

/// Calcula o Quantum Interest para um déficit de densidade.
/// Quanto maior o déficit e o tempo, maior o juro a pagar.
pub fn quantum_interest(density_debt: f64, duration: f64) -> f64 {
    // Juros exponenciais: e^(debt * duration)
    (density_debt * duration).exp()
}

/// Converte um nível de coerência (0..1) em densidade efectiva do vácuo.
/// Simplificação linear: ρ = ρ₀ * (1 + α * coherence)
pub fn coherence_to_density(coherence: f64) -> f64 {
    1.0 + ZPF_COUPLING * coherence
}

/// Converte densidade em coerência (inverso da função anterior).
pub fn density_to_coherence(density: f64) -> f64 {
    (density - 1.0) / ZPF_COUPLING
}
