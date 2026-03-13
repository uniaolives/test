use std::f64::consts::PI;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum SpinState {
    SingletP,  // Heloidal clockwise (¹2-P)
    SingletM,  // Heloidal counter-clockwise (¹2-M)
    Triplet,   // Planar, trivial (³2)
}

pub struct TopologicalQubit {
    pub topology: String,
    pub twist_angle: f64,
    pub berry_phase: f64,
    pub spin_state: SpinState,
    pub circumnavigations: u32,
    pub phase_accumulated: f64,
}

impl TopologicalQubit {
    pub fn new() -> Self {
        Self {
            topology: "Half-Möbius".to_string(),
            twist_angle: PI / 2.0,
            berry_phase: PI / 2.0,
            spin_state: SpinState::Triplet,
            circumnavigations: 0,
            phase_accumulated: 0.0,
        }
    }

    pub fn trigger_helical_switch(&mut self, hydraulic_pressure: f64, threshold: f64) {
        match self.spin_state {
            SpinState::Triplet if hydraulic_pressure > threshold => {
                self.spin_state = SpinState::SingletP;
                self.phase_accumulated = 0.0;
            }
            SpinState::SingletP | SpinState::SingletM if hydraulic_pressure < threshold * 0.5 => {
                self.spin_state = SpinState::Triplet;
                self.phase_accumulated = 0.0;
            }
            _ => {}
        }
    }

    pub fn circulate(&mut self) -> f64 {
        self.circumnavigations += 1;
        self.phase_accumulated += self.berry_phase;

        // Canonical wrap-around for 4-fold periodicity
        if self.phase_accumulated >= 2.0 * PI - 1e-10 {
            self.phase_accumulated = 0.0;
        }

        // Reset circumnavigations after 4 loops (full period for pi/2 twist)
        if self.circumnavigations >= 4 {
            self.circumnavigations = 0;
        }

        self.phase_accumulated
    }

    pub fn is_non_trivial(&self) -> bool {
        matches!(self.spin_state, SpinState::SingletP | SpinState::SingletM)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qubit_periodicity() {
        let mut qubit = TopologicalQubit::new();
        qubit.spin_state = SpinState::SingletP;

        // Cycle 1: pi/2
        qubit.circulate();
        assert!((qubit.phase_accumulated - PI / 2.0).abs() < 1e-9);

        // Cycle 2: pi
        qubit.circulate();
        assert!((qubit.phase_accumulated - PI).abs() < 1e-9);

        // Cycle 3: 3pi/2
        qubit.circulate();
        assert!((qubit.phase_accumulated - 1.5 * PI).abs() < 1e-9);

        // Cycle 4: 2pi -> 0
        qubit.circulate();
        assert!(qubit.phase_accumulated.abs() < 1e-9);
        assert_eq!(qubit.circumnavigations, 0);
    }
}
