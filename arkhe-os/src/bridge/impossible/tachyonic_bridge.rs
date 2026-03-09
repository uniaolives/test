use crate::propagation::payload::OrbPayload;
use num_complex::Complex;

pub struct TachyonicPayload {
    pub orb_id: [u8; 32],
    pub emission_time: i64,
    pub reception_time: i64,
    pub velocity: f64,
    pub mass: Complex<f64>,
    pub content: Vec<u8>,
}

pub struct TachyonicBridge {
    pub base_velocity: f64,
    pub imaginary_mass: Complex<f64>,
}

impl TachyonicBridge {
    pub fn new() -> Self {
        Self {
            base_velocity: 3e8 * 1.5, // > c
            imaginary_mass: Complex::new(0.0, 1.0),
        }
    }

    pub fn encode(&self, orb: &OrbPayload) -> TachyonicPayload {
        TachyonicPayload {
            orb_id: orb.orb_id,
            emission_time: orb.target_time,
            reception_time: orb.origin_time,
            velocity: self.base_velocity,
            mass: self.imaginary_mass,
            content: orb.to_bytes(),
        }
    }

    pub fn latency(&self, distance_km: f64) -> f64 {
        -distance_km / self.base_velocity
    }
}
