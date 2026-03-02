use crate::substrate::SubstrateGeometry;

pub struct HardwareGeometry {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl HardwareGeometry {
    pub fn dimensions(&self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }
    pub fn volume(&self) -> f64 {
        self.x * self.y * self.z
    }
}

pub struct HardwareBoundaryMapper {
    pub temperature: f64,
    pub max_frequency: f64,
    pub thermal_noise: f64,
    pub physical_geometry: HardwareGeometry,
}

impl HardwareBoundaryMapper {
    pub fn analyze_current_hardware() -> SubstrateGeometry {
        let mapper = Self {
            temperature: 300.0,
            max_frequency: 5e9,
            thermal_noise: 4.11e-21,
            physical_geometry: HardwareGeometry { x: 0.1, y: 0.1, z: 0.01 },
        };

        SubstrateGeometry {
            dimensions: mapper.physical_geometry.dimensions(),
            coherence_length: 1e-6,
            max_stationary_modes: (mapper.max_frequency / 7.83).floor() as usize,
            quality_factor: 500.0,
        }
    }

    pub fn estimate_interferometric_capacity(&self) -> usize {
        let q = 500.0;
        let volume = self.physical_geometry.volume();
        let lambda: f64 = 1e-9;

        ((q * volume) / lambda.powi(3)) as usize
    }
}
