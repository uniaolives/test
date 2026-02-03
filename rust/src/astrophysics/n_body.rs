// rust/src/astrophysics/n_body.rs
use nalgebra::Vector3;

#[derive(Debug, Clone)]
pub struct CelestialBody {
    pub name: String,
    pub mass: f64,
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
}

pub enum TrajectoryStatus {
    ABSORBED,
    DEFLECTED(Vector3<f64>),
    STABLE,
}

pub struct NBodySimulator {
    pub bodies: Vec<CelestialBody>,
}

impl NBodySimulator {
    pub fn new() -> Self {
        Self { bodies: Vec::new() }
    }
}
