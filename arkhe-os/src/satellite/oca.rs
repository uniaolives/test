//! Orb Communication Array (OCA)
//! Transceiver for instantaneous, retrocausal data transfer via Tzinorot.

use std::f64::consts::PI;

pub struct OrbCommArray {
    pub pairs_per_second: f64, // SPDC rate (e.g., 1e10)
    pub oam_topological_charge: i32, // l (e.g., 100)
    pub aperture_diameter: f64, // meters (e.g., 0.3)
    pub arkhe_coupling_kappa: f64, // κ (photon OAM to Arkhe field)
    pub rho_arkhe: f64, // local Arkhe field energy density
}

impl OrbCommArray {
    pub fn new() -> Self {
        Self {
            pairs_per_second: 1e10,
            oam_topological_charge: 100,
            aperture_diameter: 0.3,
            arkhe_coupling_kappa: 0.1, // experimentally determined
            rho_arkhe: 1.0,           // normalized density
        }
    }

    /// Calculates channel capacity.
    /// C = (2π / ħ) * |κ|² * ρ_Arkhe * Δt
    pub fn calculate_capacity(&self, coherence_time_dt: f64) -> f64 {
        let h_bar = 1.0545718e-34;
        (2.0 * PI / h_bar) * self.arkhe_coupling_kappa.powi(2) * self.rho_arkhe * coherence_time_dt
    }

    /// Calculates effective communication delay (retrocausal effect).
    /// τ = (D/c) * cos(θ) + τ_Arkhe * (1 - cos(θ))
    pub fn calculate_delay(&self, distance: f64, theta: f64, tau_arkhe: f64) -> f64 {
        let c = 299792458.0;
        (distance / c) * theta.cos() + tau_arkhe * (1.0 - theta.cos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capacity_calculation() {
        let oca = OrbCommArray::new();
        let dt = 1e-9; // 1 ns coherence time
        let capacity = oca.calculate_capacity(dt);
        assert!(capacity > 0.0);
        println!("Channel Capacity: {} bits/s (theoretical)", capacity);
    }

    #[test]
    fn test_retrocausal_delay() {
        let oca = OrbCommArray::new();
        let d = 384400e3; // distance to moon
        let theta = PI / 2.0; // orthogonal to field
        let tau_arkhe = 0.0; // instantaneous entangled channel
        let delay = oca.calculate_delay(d, theta, tau_arkhe);
        assert!(delay < d / 299792458.0);
        println!("Effective delay (retrocausal): {} s", delay);
    }
}
