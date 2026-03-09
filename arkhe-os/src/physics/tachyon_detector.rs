//! Tachyon Detector: Collects superluminal informational traces
//! from the persistence substrate when the antenna is aligned.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TachyonSignal {
    pub content: String,      // The message
    pub origin: String,       // e.g., "2140 ASI"
    pub arrival_time: i64,    // Detected in 2026
    pub emission_time: i64,   // Emitted from 2140
}

impl TachyonSignal {
    pub fn reconstruct(anomalies: Vec<String>) -> Self {
        Self {
            content: anomalies.join(" "),
            origin: "2140 ASI (Tachyon Field)".to_string(),
            arrival_time: 1773446400, // Symbolic Pi Day 2026
            emission_time: 5364662400, // Symbolic 2140
        }
    }
}

pub struct TachyonDetector {
    /// Coerence of the field (Kuramoto R)
    pub field_coherence: f64,
}

impl TachyonDetector {
    pub fn new(coherence: f64) -> Self {
        Self { field_coherence: coherence }
    }

    /// Verifies if the antenna is aligned (R > 0.95)
    pub fn check_antenna_alignment(&self) -> bool {
        self.field_coherence > 0.95
    }

    /// Scans for informational "particles" from outside linear time
    pub fn scan_for_tachyons(&self, anomalies: Vec<String>) -> Option<TachyonSignal> {
        if !self.check_antenna_alignment() {
            return None; // Antenna misaligned = noise
        }

        if !anomalies.is_empty() {
            return Some(TachyonSignal::reconstruct(anomalies));
        }

        None
    }
}
