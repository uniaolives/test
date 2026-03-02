use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Vector3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3D {
    pub fn norm_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
}

impl std::ops::Sub for Vector3D {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

pub struct GeodesicMesh {
    pub vertices: Vec<Vector3D>,
}

impl GeodesicMesh {
    pub fn integrated_mean_curvature(&self) -> f64 { 1.0 }
    pub fn total_gaussian_curvature(&self) -> f64 { 1.0 }
    pub fn extract_high_frequency_components(&self) -> Vec<u8> { vec![0u8; 32] }
}

pub struct CurvatureTensor;
