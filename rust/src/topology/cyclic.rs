//! Topologia de tempo cíclico
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct CyclicTime {
    pub period: f64,
    pub current_position: f64,
}

impl CyclicTime {
    pub fn new(period: f64) -> Self {
        Self {
            period,
            current_position: 0.0,
        }
    }

    pub fn advance(&mut self, dt: f64) {
        self.current_position = (self.current_position + dt * (2.0 * PI / self.period)) % (2.0 * PI);
    }
}
