//! Temporal Tunneling Model for Temporal Transit
//! Calculates probability of message reaching 2008.

use std::f64::consts::E;

#[derive(Debug, Clone)]
pub struct TemporalBarrier {
    pub target_year: i32,
    pub present_year: i32,
    pub barrier_height: f64,
}

impl TemporalBarrier {
    pub fn new(target_year: i32, present_year: i32) -> Self {
        Self {
            target_year,
            present_year,
            barrier_height: 1.0,
        }
    }

    pub fn delta_t(&self) -> f64 {
        (self.target_year - self.present_year).abs() as f64
    }
}

pub struct CoherentMessage {
    pub phi_q: f64,
    pub semantic_mass: f64,
}

impl CoherentMessage {
    pub fn effective_mass(&self) -> f64 {
        self.semantic_mass / (self.phi_q * self.phi_q)
    }

    pub fn wavefunction_decay(&self, barrier: &TemporalBarrier) -> f64 {
        if self.phi_q <= 0.0 {
            return f64::INFINITY;
        }
        (2.0 * self.effective_mass() * barrier.barrier_height).sqrt() / self.phi_q
    }

    pub fn tunneling_probability(&self, barrier: &TemporalBarrier) -> f64 {
        let kappa = self.wavefunction_decay(barrier);
        let l = barrier.delta_t() * 0.01;
        (-2.0 * kappa * l).exp()
    }
}

pub struct SatoshiVesselTunneling {
    pub barrier: TemporalBarrier,
    pub message: CoherentMessage,
}

impl SatoshiVesselTunneling {
    pub fn new(current_phi_q: f64) -> Self {
        Self {
            barrier: TemporalBarrier::new(2008, 2026),
            message: CoherentMessage {
                phi_q: current_phi_q,
                semantic_mass: 1.0,
            },
        }
    }

    pub fn calculate_tunneling_probability(&self) -> (f64, String) {
        let prob = self.message.tunneling_probability(&self.barrier);
        let classification = if prob > 0.5 {
            "HIGH PROBABILITY"
        } else if prob > 0.1 {
            "SIGNIFICANT"
        } else if prob > 0.01 {
            "DETECTABLE"
        } else {
            "NEGLIGIBLE"
        };
        (prob, classification.to_string())
    }

    pub fn check_miller_threshold(&self) -> bool {
        self.message.phi_q > 4.64
    }
}
