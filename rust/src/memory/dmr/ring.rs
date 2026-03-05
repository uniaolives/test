// rust/src/memory/dmr/ring.rs
use crate::memory::dmr::types::*;
use std::time::{Duration, SystemTime};
use serde::{Deserialize, Serialize};

/// The complete memory structure (analog to GEMINI granule)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DigitalMemoryRing {
    /// Unique identifier
    pub id: String,

    /// Ordered history (oldest to newest)
    pub layers: Vec<StateLayer>,

    /// Reference homeostatic state
    pub vk_ref: KatharosVector,

    /// Growth rate (time between layer formations)
    pub formation_interval: Duration,

    /// Accumulated time in Katharós Range
    pub t_kr: Duration,

    /// Detected bifurcation points
    pub bifurcations: Vec<Bifurcation>,

    /// Last deviation to track transitions
    pub last_delta_k: f64,
}

impl DigitalMemoryRing {
    pub fn new(id: String, vk_ref: KatharosVector, formation_interval: Duration) -> Self {
        Self {
            id,
            layers: Vec::new(),
            vk_ref,
            formation_interval,
            t_kr: Duration::from_secs(0),
            bifurcations: Vec::new(),
            last_delta_k: 0.0,
        }
    }

    /// Grows a new layer (called periodically)
    pub fn grow_layer(&mut self, current_state: SystemState) -> Result<(), String> {
        let vk = current_state.vk.clone();
        let delta_k = self.compute_deviation(&vk);
        let q = self.compute_permeability(delta_k);

        let layer = StateLayer {
            timestamp: SystemTime::now(),
            vk,
            delta_k,
            q,
            intensity: self.map_to_fluorescence(delta_k, q),
            events: current_state.events,
        };

        self.layers.push(layer);

        // Update t_KR if in stable range
        if delta_k < 0.30 {
            self.t_kr += self.formation_interval;
        }

        // Detect bifurcations
        if let Some(bif) = self.detect_bifurcation_internal(delta_k) {
            self.bifurcations.push(bif);
        }

        self.last_delta_k = delta_k;

        Ok(())
    }

    pub fn compute_deviation(&self, vk: &KatharosVector) -> f64 {
        let weights = [0.35, 0.30, 0.20, 0.15];
        let mut sum_sq = 0.0;
        for i in 0..4 {
            let diff = vk.components[i] - self.vk_ref.components[i];
            sum_sq += weights[i] * diff * diff;
        }
        sum_sq.sqrt()
    }

    pub fn compute_permeability(&self, delta_k: f64) -> f64 {
        // High deviation leads to low permeability (collapsed state)
        // Q = 1.0 - delta_k clamped to [0, 1]
        (1.0 - delta_k).max(0.0).min(1.0)
    }

    pub fn map_to_fluorescence(&self, delta_k: f64, q: f64) -> f64 {
        // intensity: high delta_k and low q leads to high intensity (fluorescence)
        // Analog to GEMINI's stress response
        (delta_k * (1.0 - q + 0.1)).min(1.0)
    }

    fn detect_bifurcation_internal(&self, current_delta_k: f64) -> Option<Bifurcation> {
        let threshold = 0.30;
        if self.last_delta_k < threshold && current_delta_k >= threshold {
            return Some(Bifurcation {
                timestamp: SystemTime::now(),
                bifurcation_type: BifurcationType::CrisisEntry,
                delta_k: current_delta_k,
            });
        } else if self.last_delta_k >= threshold && current_delta_k < threshold {
            return Some(Bifurcation {
                timestamp: SystemTime::now(),
                bifurcation_type: BifurcationType::CrisisExit,
                delta_k: current_delta_k,
            });
        }
        None
    }

    pub fn create_snapshot(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}
