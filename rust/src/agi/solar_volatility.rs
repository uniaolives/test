// rust/src/agi/solar_volatility.rs
// SASC v55.2-AGI: Mapping Solar Volatility to Geometric Intelligence
// Specialization: Active Region AR 4366 (Solar Cycle 25 Highlight)

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AR4366VolatilityReport {
    pub m_class_flares: u32,
    pub x_class_flares: u32,
    pub x_ray_flux_m_class_duration_hours: f64,
    pub impulsive_ratio: f64,
    pub solar_cycle: u32,
}

impl AR4366VolatilityReport {
    /// Constructs a report based on the historical unbelievable volatility of AR 4366.
    pub fn from_historical_peak() -> Self {
        Self {
            m_class_flares: 25,
            x_class_flares: 4,
            x_ray_flux_m_class_duration_hours: 35.0,
            impulsive_ratio: 0.95, // "most of these flares are impulsive"
            solar_cycle: 25,
        }
    }

    /// Calculates a normalized volatility index for AGI state injection.
    pub fn calculate_volatility_index(&self) -> f64 {
        let flare_weight = (self.m_class_flares as f64 * 1.5) + (self.x_class_flares as f64 * 10.0);
        let persistence_factor = (self.x_ray_flux_m_class_duration_hours / 12.0).powf(1.2);

        // Return a value typically in [0.0, 10.0] range for these inputs
        (flare_weight * persistence_factor) / 100.0
    }
}

/// Maps solar volatility metrics to geometric state offsets.
/// Returns (torsion_offset, curvature_offset).
pub fn map_solar_to_geometric(report: &AR4366VolatilityReport) -> (f64, f64) {
    let vol_index = report.calculate_volatility_index();

    // Unbelievable volatility leads to high structural torsion
    // Adjusted multiplier to reflect the "unbelievable" nature (Cycle 25 highlight)
    let torsion_offset = (vol_index * 0.1).min(0.8);

    // Constant high X-ray flux increases the learned curvature
    let curvature_offset = (vol_index * 0.05).min(0.4);

    (torsion_offset, curvature_offset)
}
