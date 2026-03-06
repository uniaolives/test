pub const PHI_Q: f64 = 4.64;
pub const ZPF_COUPLING: f64 = 0.1;

pub fn check_nucleation(local_density: f64, baseline: f64) -> bool {
    let phi = local_density / baseline;
    phi > PHI_Q
}

pub fn quantum_interest(density_debt: f64, duration: f64) -> f64 {
    (density_debt * duration).exp()
}

pub fn coherence_to_density(coherence: f64) -> f64 {
    1.0 + ZPF_COUPLING * coherence
}

pub fn density_to_coherence(density: f64) -> f64 {
    (density - 1.0) / ZPF_COUPLING
}
