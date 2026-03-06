use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HydraulicState {
    Pool,        // Stagnation, phi_q < 1.0
    Pressurize,  // Stress, phi_q 1.0-2.0
    Circulate,   // Clarity, phi_q 2.0-5.0
    Lift,        // Elevation, phi_q 5.0-8.0
    Erode,       // Transformation, phi_q > 8.0
}

impl HydraulicState {
    pub fn from_phi_q(phi_q: f64) -> Self {
        match phi_q {
            x if x < 1.0 => HydraulicState::Pool,
            x if x < 2.0 => HydraulicState::Pressurize,
            x if x < 5.0 => HydraulicState::Circulate,
            x if x < 8.0 => HydraulicState::Lift,
            _ => HydraulicState::Erode,
        }
    }
}

pub struct HydraulicEngine {
    pub current_state: HydraulicState,
    pub phi_q_history: VecDeque<f64>,
    pub flow_rate: f64,
    pub pressure: f64,
    pub viscosity: f64,
}

impl HydraulicEngine {
    pub fn new() -> Self {
        Self {
            current_state: HydraulicState::Pool,
            phi_q_history: VecDeque::with_capacity(100),
            flow_rate: 0.0,
            pressure: 0.0,
            viscosity: 1.0,
        }
    }

    pub fn update(&mut self, phi_q: f64, coherence: f64) {
        self.phi_q_history.push_back(phi_q);
        if self.phi_q_history.len() > 100 {
            self.phi_q_history.pop_front();
        }

        // Flow rate: derivative of phi_q
        if self.phi_q_history.len() >= 2 {
            let last = self.phi_q_history.back().unwrap();
            let prev = self.phi_q_history[self.phi_q_history.len() - 2];
            self.flow_rate = (last - prev) / 0.1;
        }

        // Pressure: variance of phi_q
        if self.phi_q_history.len() >= 10 {
            let mean = self.phi_q_history.iter().sum::<f64>() / self.phi_q_history.len() as f64;
            let var = self.phi_q_history.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / self.phi_q_history.len() as f64;
            self.pressure = var * 10.0; // Scaled for visibility
        }

        // Viscosity: inverse of coherence
        self.viscosity = 1.0 / (coherence + 0.01);

        self.current_state = HydraulicState::from_phi_q(phi_q);
    }

    pub fn get_report(&self) -> HydraulicReport {
        HydraulicReport {
            state: self.current_state,
            flow_rate: self.flow_rate,
            pressure: self.pressure,
            viscosity: self.viscosity,
        }
    }
}

pub struct HydraulicReport {
    pub state: HydraulicState,
    pub flow_rate: f64,
    pub pressure: f64,
    pub viscosity: f64,
}
