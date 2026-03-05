//! Digital Memory Ring – grows layers like a tree ring.

use crate::{Bifurcation, BifurcationKind, KatharosVector, StateLayer, TimeRange};
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

/// Main memory structure (analogous to GEMINI granule)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DigitalMemoryRing {
    /// Unique identifier
    pub id: String,
    /// Ordered history (oldest to newest)
    pub layers: Vec<StateLayer>,
    /// Reference homeostatic state (calibrated at birth)
    pub vk_ref: KatharosVector,
    /// Time between layer formations (e.g., 1 hour)
    pub formation_interval: Duration,
    /// Accumulated time in Katharós Range (ΔK < 0.30)
    pub t_kr: Duration,
    /// Detected bifurcations
    pub bifurcations: Vec<Bifurcation>,
}

impl DigitalMemoryRing {
    /// Creates a new empty ring.
    pub fn new(id: String, vk_ref: KatharosVector, formation_interval: Duration) -> Self {
        Self {
            id,
            layers: Vec::new(),
            vk_ref,
            formation_interval,
            t_kr: Duration::ZERO,
            bifurcations: Vec::new(),
        }
    }

    /// Grows a new layer (called periodically).
    pub fn grow_layer(&mut self, current_vk: KatharosVector, q: f64, events: Vec<String>) {
        let now = SystemTime::now();
        let delta_k = self.vk_ref.weighted_distance(&current_vk);
        let intensity = self.compute_intensity(delta_k);

        // Record layer
        let layer = StateLayer {
            timestamp: now,
            vk: current_vk,
            delta_k,
            q,
            intensity,
            events,
        };
        self.layers.push(layer);

        // Update t_KR if in stable range
        if delta_k < 0.30 {
            self.t_kr += self.formation_interval;
        }

        // Detect bifurcations (if at least two layers exist)
        if self.layers.len() >= 2 {
            self.detect_and_record_bifurcation();
        }
    }

    /// Compute fluorescence intensity as a function of ΔK.
    fn compute_intensity(&self, delta_k: f64) -> f64 {
        // Map ΔK ∈ [0, 1.0] roughly to intensity [0,1]
        // For ΔK > 0.7, intensity saturates near 1.0
        (delta_k * 1.2).min(1.0)
    }

    /// Detect a bifurcation based on the last two layers.
    fn detect_and_record_bifurcation(&mut self) {
        let n = self.layers.len();
        let prev = &self.layers[n - 2];
        let curr = &self.layers[n - 1];

        // Determine regime
        let prev_regime = if prev.delta_k < 0.30 {
            "katharos"
        } else if prev.delta_k >= 0.70 {
            "crisis"
        } else {
            "transition"
        };
        let curr_regime = if curr.delta_k < 0.30 {
            "katharos"
        } else if curr.delta_k >= 0.70 {
            "crisis"
        } else {
            "transition"
        };

        // Detect regime change
        if prev_regime != curr_regime {
            let kind = match (prev_regime, curr_regime) {
                (_, "katharos") => BifurcationKind::EntryKatharós,
                ("katharos", _) => BifurcationKind::ExitKatharós,
                (_, "crisis") => BifurcationKind::CrisisEntry,
                ("crisis", _) => BifurcationKind::CrisisExit,
                _ => return, // should not happen
            };
            self.bifurcations.push(Bifurcation {
                timestamp: curr.timestamp,
                kind,
                delta_k_before: prev.delta_k,
                delta_k_after: curr.delta_k,
            });
        }
    }

    /// Reconstruct the full VK trajectory.
    pub fn reconstruct_trajectory(&self) -> Vec<(SystemTime, KatharosVector)> {
        self.layers
            .iter()
            .map(|layer| (layer.timestamp, layer.vk.clone()))
            .collect()
    }

    /// Find contiguous periods where ΔK < 0.30.
    pub fn find_katharos_periods(&self) -> Vec<TimeRange> {
        let mut periods = Vec::new();
        let mut start: Option<usize> = None;

        for (i, layer) in self.layers.iter().enumerate() {
            if layer.delta_k < 0.30 {
                if start.is_none() {
                    start = Some(i);
                }
            } else {
                if let Some(s) = start {
                    periods.push(TimeRange {
                        start: self.layers[s].timestamp,
                        end: self.layers[i - 1].timestamp,
                    });
                    start = None;
                }
            }
        }
        // If still in a period at the end
        if let Some(s) = start {
            periods.push(TimeRange {
                start: self.layers[s].timestamp,
                end: self.layers.last().unwrap().timestamp,
            });
        }
        periods
    }

    /// Create a snapshot for Timechain anchoring (hash of current state).
    pub fn create_snapshot(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_else(|_| vec![])
    }
}
