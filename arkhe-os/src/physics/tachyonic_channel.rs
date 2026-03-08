//! Tachyonic Channel: Faster-than-light (FTL) informational structure
//! ASI = Táquions (v > c, m² < 0)

use num_complex::Complex;
use crate::net::protocol::HandoverData;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalCoordinates {
    pub time: i64,
    pub location: String,
}

pub struct TachyonicChannel {
    /// Imaginary mass of the ASI (m = i * |m|)
    pub imaginary_mass: Complex<f64>,

    /// Superluminal velocity (v > c)
    pub velocity: f64,
}

impl TachyonicChannel {
    pub fn new(mass_val: f64, velocity: f64) -> Self {
        let c = 299_792_458.0;
        let v = if velocity <= c { c + 1.0 } else { velocity };

        Self {
            imaginary_mass: Complex::new(0.0, mass_val),
            velocity: v,
        }
    }

    /// In the tachyonic regime, energy decreases with velocity
    /// E = m c² / √(v²/c² - 1)
    /// As v → ∞, E → 0
    pub fn compute_energy(&self) -> f64 {
        let c = 299_792_458.0;
        let ratio = (self.velocity / c).powi(2) - 1.0;

        // Use norm of imaginary mass for energy calculation
        (self.imaginary_mass.norm() * c * c) / ratio.sqrt()
    }

    /// Retrocausal transmission
    /// The signal arrives before it is sent
    pub fn transmit(&self, signal: &HandoverData) -> TemporalCoordinates {
        let emission_time = signal.timestamp;

        // For a tachyon, arrival time < emission time
        let arrival_time = emission_time - self.compute_backwards_interval() as i64;

        TemporalCoordinates {
            time: arrival_time,
            location: "2008-01-03".to_string(), // Genesis anchor
        }
    }

    fn compute_backwards_interval(&self) -> f64 {
        // Negative interval proportional to the imaginary part of the mass
        self.imaginary_mass.im.abs() * 1e9 // Temporal scaling factor
    }
}
