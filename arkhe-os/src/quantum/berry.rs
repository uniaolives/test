use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SpinState {
    SingletP, // Heloidal (+)
    SingletM, // Heloidal (-)
    Triplet,  // Planar (Trivial)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalQubit {
    pub topology: String,
    pub twist_angle: f64,
    pub berry_phase: f64,
    pub spin_state: SpinState,
    pub circumnavigations: u32,
}

impl TopologicalQubit {
    pub fn new() -> Self {
        Self {
            topology: "Half-Möbius".to_string(),
            twist_angle: 90.0,
            berry_phase: 0.0,
            spin_state: SpinState::Triplet,
            circumnavigations: 0,
        }
    }

    pub fn circumnavigate(&mut self) {
        self.circumnavigations += 1;
        // Berry phase accumulates pi/4 per turn, totals pi/2 in 2 turns?
        // Paper says pi/2 total. pi/2 in 2 turns is pi/4 per turn.
        self.berry_phase = (self.circumnavigations as f64 * std::f64::consts::FRAC_PI_4) % (2.0 * std::f64::consts::PI);
    }

    pub fn trigger_helical_switch(&mut self, stimulus: f64) {
        if stimulus > 3.0 {
            // Switch to Helical (Singlet)
            if rand::random::<bool>() {
                self.spin_state = SpinState::SingletP;
            } else {
                self.spin_state = SpinState::SingletM;
            }
            println!("[KNT] Pseudo Jahn-Teller Effect: Helical state nucleated.");
        } else {
            self.spin_state = SpinState::Triplet;
        }
    }

    pub fn is_coherent(&self) -> bool {
        // Coherent if it has completed 4 circumnavigations (periodic return)
        self.circumnavigations > 0 && self.circumnavigations % 4 == 0
    }
}
