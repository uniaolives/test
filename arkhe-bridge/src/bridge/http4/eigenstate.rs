use super::methods::ConfinementMode;

#[derive(Debug, Clone, PartialEq)]
pub struct TemporalEigenstate {
    pub n: u32,
    pub energy: String,
    pub probability: f64,
    pub mode: String,
}

impl TemporalEigenstate {
    pub fn new(n: u32, energy: &str, probability: f64, mode: &str) -> Self {
        Self {
            n,
            energy: energy.to_string(),
            probability,
            mode: mode.to_string(),
        }
    }
}
