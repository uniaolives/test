pub struct ButterflyGeometry {
    pub n_butterflies: usize,
    pub phase_coherence: f64,
    pub cone_ratio: f64,
}

impl ButterflyGeometry {
    pub fn new(n: usize, r: f64, ratio: f64) -> Self {
        Self { n_butterflies: n, phase_coherence: r, cone_ratio: ratio }
    }

    pub fn compute_amplification(&self) -> f64 {
        let base_amp = (self.phase_coherence * self.n_butterflies as f64).powi(2);
        base_amp * self.cone_ratio
    }

    pub fn simulate_effect(&self, initial_perturbation: f64) -> f64 {
        let amp = self.compute_amplification();
        initial_perturbation * amp
    }

    pub fn crosses_threshold(&self, threshold: f64) -> bool {
        self.compute_amplification() > threshold
    }
}
