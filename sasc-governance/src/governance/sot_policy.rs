pub struct SoTRewardFunction {
    pub accuracy_weight: f64,
    pub diversity_weight: f64,
    pub reconciliation_weight: f64,
}

impl SoTRewardFunction {
    pub fn new() -> Self {
        Self {
            accuracy_weight: 0.50,
            diversity_weight: 0.30,
            reconciliation_weight: 0.20,
        }
    }

    pub fn calculate(&self, accuracy: f64, diversity: f64, reconciliation: f64) -> f64 {
        (accuracy * self.accuracy_weight) +
        (diversity * self.diversity_weight) +
        (reconciliation * self.reconciliation_weight)
    }
}
