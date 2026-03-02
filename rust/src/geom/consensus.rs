#[derive(Debug, Clone)]
pub enum AttractorType {
    TorusKnot { p: u32, q: u32 },
    Halvorsen,
    DualHelix,
    DeJong,
}

pub struct Particle;
impl Particle {
    pub async fn compute_lyapunov(&self) -> Result<f64, String> {
        Ok(0.01) // Mock
    }
}

pub struct GeometricChecker;

impl GeometricChecker {
    pub fn new() -> Self {
        Self
    }
}
