pub mod dna_ner_manifold;
pub mod standing_wave_processor;

#[derive(Clone, Debug)]
pub struct SubstrateGeometry {
    pub dimensions: (f64, f64, f64),
    pub coherence_length: f64,
    pub max_stationary_modes: usize,
    pub quality_factor: f64,
}

impl SubstrateGeometry {
    pub fn allowed_wavenumber(&self, x: usize, y: usize) -> crate::math::geometry::Vector3D {
        crate::math::geometry::Vector3D {
            x: x as f64 * 0.1,
            y: y as f64 * 0.1,
            z: 0.0,
        }
    }

    pub fn allows_mode(&self, _mode: &standing_wave_processor::StandingWaveBit) -> bool {
        true
    }

    pub fn hash(&self) -> [u8; 32] {
        [0u8; 32]
    }

    pub fn volume(&self) -> f64 {
        self.dimensions.0 * self.dimensions.1 * self.dimensions.2
    }
}

pub struct TopologicalBraid;

pub struct PotentialWell;

pub enum PhaseLock {
    Schumann,
}

impl PhaseLock {
    pub fn schumann() -> Self { Self::Schumann }
}
