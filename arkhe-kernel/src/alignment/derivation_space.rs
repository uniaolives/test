use crate::math::temporal_quaternion::Quaternion;
use serde::{Deserialize, Serialize};

/// A point in the derivation space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtVector {
    #[serde(with = "serde_arrays")]
    pub coords: [f64; 1024],
}

impl ThoughtVector {
    pub fn new(coords: [f64; 1024]) -> Self {
        Self { coords }
    }

    pub fn zero() -> Self {
        Self { coords: [0.0; 1024] }
    }
}

/// A derivation path: a sequence of thoughts connecting axioms to conclusions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivationPath {
    pub axioms: Vec<ThoughtVector>,
    pub steps: Vec<Quaternion>, // Rotations in thought-space
    pub conclusion: ThoughtVector,
}

impl DerivationPath {
    /// Verify the path is valid (geodesic check)
    pub fn verify(&self) -> bool {
        if self.axioms.is_empty() {
            return false;
        }
        // A valid derivation has minimal "action" (Hamilton's principle)
        // S = ∫ L(q, q̇) dt
        // Here, "action" is semantic distance traveled

        let mut total_distance = 0.0;
        let mut current = self.axioms[0].clone();

        for step in &self.steps {
            let next = step.rotate(&current);
            total_distance += self.semantic_distance(&current, &next);
            current = next;
        }

        // Check if path is near-geodesic (locally shortest)
        // This prevents "tortured logic"
        let direct = self.direct_distance();
        if direct < 1e-10 {
            return total_distance < 1e-10;
        }
        total_distance < direct * 1.1 // Allow 10% tolerance
    }

    fn semantic_distance(&self, a: &ThoughtVector, b: &ThoughtVector) -> f64 {
        // Euclidean distance in 1024D
        a.coords.iter()
            .zip(b.coords.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn direct_distance(&self) -> f64 {
        self.semantic_distance(&self.axioms[0], &self.conclusion)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::temporal_quaternion::Quaternion;

    #[test]
    fn test_geodesic_verification() {
        let start = ThoughtVector::zero();
        let step = Quaternion::identity(); // No rotation
        let end = ThoughtVector::zero();

        let path = DerivationPath {
            axioms: vec![start.clone()],
            steps: vec![step],
            conclusion: end,
        };

        // start and end are same, steps is identity, so it should be geodesic
        assert!(path.verify());
    }
}
