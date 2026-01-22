use ndarray::Array1;
use std::collections::HashMap;

pub struct PerspectiveDiversityEngine {
    pub delegates_traits: Vec<Array1<f64>>,
}

impl PerspectiveDiversityEngine {
    pub fn new(count: usize) -> Self {
        // Mock: create delegates with random traits
        let mut delegates_traits = Vec::new();
        for _ in 0..count {
            delegates_traits.push(Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.5]));
        }
        Self { delegates_traits }
    }

    /// Detects Groupthink by calculating the average cosine distance between traits
    pub fn calculate_groupthink_score(&self) -> f64 {
        if self.delegates_traits.len() < 2 {
            return 0.0;
        }

        let mut total_distance = 0.0;
        let mut pairs = 0;

        for i in 0..self.delegates_traits.len() {
            for j in (i + 1)..self.delegates_traits.len() {
                let dot = self.delegates_traits[i].dot(&self.delegates_traits[j]);
                let mag_i = self.delegates_traits[i].dot(&self.delegates_traits[i]).sqrt();
                let mag_j = self.delegates_traits[j].dot(&self.delegates_traits[j]).sqrt();

                let cosine_sim = dot / (mag_i * mag_j);
                total_distance += 1.0 - cosine_sim;
                pairs += 1;
            }
        }

        1.0 - (total_distance / pairs as f64) // 1.0 means perfect consensus (bad for SoT)
    }
}
