//! Biophotonic Sensor Array (BSA)
//! DNA-laser-based detectors for Arkhe field fluctuations.

use std::f64::consts::PI;

pub struct BiophotonicSensorArray {
    pub dna_strands_per_nanoparticle: u32, // (e.g., 1000)
    pub nanoparticle_count: u32,           // (e.g., 1e6)
    pub pump_wavelength: f64,              // nm (e.g., 340 nm)
    pub primary_emission_wavelength: f64,  // nm (e.g., 680 nm)
    pub detector_efficiency: f64,           // (e.g., 0.9)
}

impl BiophotonicSensorArray {
    pub fn new() -> Self {
        Self {
            dna_strands_per_nanoparticle: 1000,
            nanoparticle_count: 1_000_000,
            pump_wavelength: 340.0e-9,
            primary_emission_wavelength: 680.0e-9,
            detector_efficiency: 0.9,
        }
    }

    /// Calculates biophoton emission wavelength based on DNA base-pair spacing.
    /// λ_n = 2d / n where d ≈ 0.34 nm
    pub fn base_pair_emission_wavelength(&self, harmonic_n: i32) -> f64 {
        let base_pair_spacing_d = 0.34e-9;
        (2.0 * base_pair_spacing_d) / harmonic_n as f64
    }

    /// Calculates Arkhe field sensitivity.
    /// ΔΦ_Arkhe = (ħ * Δν) / (2π * d²)
    pub fn calculate_sensitivity(&self, frequency_shift_dv: f64, distance_to_source: f64) -> f64 {
        let h_bar = 1.0545718e-34;
        (h_bar * frequency_shift_dv) / (2.0 * PI * distance_to_source.powi(2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emission_wavelength() {
        let bsa = BiophotonicSensorArray::new();
        let lambda_1 = bsa.base_pair_emission_wavelength(1);
        assert!(lambda_1 > 0.3e-9 && lambda_1 < 1.0e-9);
        println!("Base pair emission λ_1: {} m", lambda_1);
    }

    #[test]
    fn test_field_sensitivity() {
        let bsa = BiophotonicSensorArray::new();
        let dv = 1.0; // 1 Hz shift
        let d = 1e6; // 1000 km
        let sensitivity = bsa.calculate_sensitivity(dv, d);
        assert!(sensitivity > 0.0);
        println!("Arkhe Field Sensitivity: {} J*s/m^2", sensitivity);
    }
}
