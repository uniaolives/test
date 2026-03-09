use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralToken {
    pub id: String,
    pub spike_frequency: f64,
    pub amplitude: f64,
}
