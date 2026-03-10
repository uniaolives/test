use super::methods::ConfinementMode;
use anyhow::Result;

pub struct QuantumWell {
    pub barrier_height: f64,
    pub width: f64,
}

impl QuantumWell {
    pub fn configure(lambda_2: f64, mode: ConfinementMode) -> Result<Self> {
        let height = match mode {
            ConfinementMode::InfiniteWell => f64::INFINITY,
            ConfinementMode::FiniteWell => 10.0,
            ConfinementMode::Barrier => 5.0,
            ConfinementMode::Free => 0.0,
        };

        Ok(Self {
            barrier_height: height,
            width: 1.0 - lambda_2, // Width decreases as coherence increases
        })
    }
}
