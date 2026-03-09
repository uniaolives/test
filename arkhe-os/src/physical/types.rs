use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Copy, PartialEq)]
pub struct GeoCoord {
    pub lat: f64,
    pub lon: f64,
}

impl GeoCoord {
    pub fn current() -> Self {
        Self { lat: -22.9519, lon: -43.2105 } // Cristo Redentor, Rio
    }

    pub fn target_2008() -> Self {
        Self { lat: 51.5074, lon: -0.1278 } // London
    }
}
