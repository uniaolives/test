use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeaningField {
    pub attention: f64,
    pub intention: f64,
    pub emotion: f64,
    pub identity: f64,
    pub somatic: f64,
}

impl MeaningField {
    pub fn new() -> Self {
        Self {
            attention: 0.5,
            intention: 0.5,
            emotion: 0.5,
            identity: 1.0,
            somatic: 0.5,
        }
    }

    pub fn calculate_resonance(&self) -> f64 {
        (self.attention * self.intention * self.emotion * self.identity) * self.somatic
    }
}

pub enum LmtPhase {
    Awakening = 1,
    Alignment = 2,
    Empowerment = 3,
    Expression = 4,
    Rebirth = 5,
    Resonance = 6,
    Design = 7,
}
