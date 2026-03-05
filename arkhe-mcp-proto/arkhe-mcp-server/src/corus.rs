use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Corus {
    pub location: String,
    pub orientation: String, // Direction of arriving flows
    pub rate: f64,          // Metabolism cycling rate
    pub scale: String,      // Resolver size (e.g., individual, society)
}

impl Corus {
    pub fn new(location: &str, orientation: &str, rate: f64, scale: &str) -> Self {
        Self {
            location: location.to_string(),
            orientation: orientation.to_string(),
            rate,
            scale: scale.to_string(),
        }
    }
}
