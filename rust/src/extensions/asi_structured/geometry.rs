use nalgebra::{DVector};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellGeometry {
    pub ambient_dimension: usize,
    pub concentration_type: ConcentrationType,
    pub radius: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConcentrationType {
    UniformBall,
    Gaussian,
    Gravitational { energy_threshold: f64 },
}

impl ShellGeometry {
    pub fn new(ambient_dimension: usize, concentration_type: ConcentrationType) -> Self {
        let radius = match &concentration_type {
            ConcentrationType::UniformBall | ConcentrationType::Gaussian => (ambient_dimension as f64).sqrt(),
            ConcentrationType::Gravitational { energy_threshold } => energy_threshold * (ambient_dimension as f64).sqrt(),
        };

        Self {
            ambient_dimension,
            concentration_type,
            radius,
        }
    }

    pub fn project_to_shell(&self, vector: &DVector<f64>) -> DVector<f64> {
        let norm = vector.norm();
        if norm < 1e-10 {
            return vector.clone();
        }
        vector * (self.radius / norm)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicBasis {
    pub ambient_dimension: usize,
    pub modes: Vec<(i32, i32)>,
}

impl HarmonicBasis {
    pub fn new(ambient_dimension: usize, max_l: i32) -> Self {
        let mut modes = Vec::new();
        for l in 0..=max_l {
            for m in -l..=l {
                modes.push((l, m));
            }
        }
        Self {
            ambient_dimension,
            modes,
        }
    }

    pub fn project(&self, _vector: &DVector<f64>, _l: i32, _m: i32) -> nalgebra::Complex<f64> {
        // Mocked spherical harmonic projection
        nalgebra::Complex::new(0.5, 0.0)
    }

    pub fn basis_vector(&self, l: i32, _m: i32) -> DVector<f64> {
        // Mocked basis vector generation
        let mut v = DVector::zeros(self.ambient_dimension);
        if self.ambient_dimension > 0 {
            v[l as usize % self.ambient_dimension] = 1.0;
        }
        v
    }
}
