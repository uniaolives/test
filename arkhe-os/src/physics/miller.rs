//! Constantes e funções relacionadas ao limiar de Miller (φ_q = 4.64)
//! e ao cálculo do Quantum Interest.

/// Limiar de Miller para nucleação de Wave-Cloud.
pub const PHI_Q: f64 = 4.64;

/// Razão Áurea (Informational Coherence Threshold)
pub const GOLDEN_RATIO: f64 = 0.618033988749895;

/// Assimetria de Bárions (Cosmological Initialization Constant)
pub const BARYON_ASYMMETRY: f64 = 1e-9;

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

/// Mapeia assimetria física (10⁻⁹) para coerência informacional (φ).
pub fn cosmological_to_information(eta: f64) -> f64 {
    // log10(0.618) ≈ -0.209.
    // Normalized by 10⁻⁹ scale.
    let log_eta = eta.log10().abs();
    let normalized = log_eta / 9.0;

    (1.0 - normalized).clamp(0.0, 1.0)
}
