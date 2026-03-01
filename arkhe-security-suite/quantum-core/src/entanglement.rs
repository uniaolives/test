pub struct BellPair {
    pub fidelity: f32,
}

impl BellPair {
    pub fn test_chsh(&self) -> f32 {
        2.82 // Maximal violation simulated
    }
}
