pub fn calculate_aeu(complexity: f32, phi: f32) -> f32 {
    complexity * (1.0 - phi)
}

pub fn survival_probability(half_life: f32, time_elapsed: f32) -> f32 {
    (-0.693 * time_elapsed / half_life).exp()
}
