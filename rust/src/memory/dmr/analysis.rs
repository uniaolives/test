// rust/src/memory/dmr/analysis.rs
use crate::memory::dmr::types::*;
use crate::memory::dmr::ring::DigitalMemoryRing;
use std::time::{SystemTime};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VKTrajectory {
    pub timestamps: Vec<SystemTime>,
    pub vk_history: Vec<KatharosVector>,
    pub delta_k_history: Vec<f64>,
    pub q_history: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: SystemTime,
    pub end: SystemTime,
}

impl DigitalMemoryRing {
    /// Extract VK trajectory (GEMINI readout analog)
    pub fn reconstruct_trajectory(&self) -> VKTrajectory {
        VKTrajectory {
            timestamps: self.layers.iter().map(|l| l.timestamp).collect(),
            vk_history: self.layers.iter().map(|l| l.vk.clone()).collect(),
            delta_k_history: self.layers.iter().map(|l| l.delta_k).collect(),
            q_history: self.layers.iter().map(|l| l.q).collect(),
        }
    }

    /// Measure total accumulated safety time
    pub fn measure_t_kr(&self) -> std::time::Duration {
        self.t_kr
    }

    /// Identify periods of homeostatic stability
    pub fn find_katharos_periods(&self) -> Vec<TimeRange> {
        let mut periods = Vec::new();
        let mut start = None;

        for (i, layer) in self.layers.iter().enumerate() {
            if layer.delta_k < 0.30 {
                if start.is_none() {
                    start = Some(i);
                }
            } else {
                if let Some(s) = start {
                    periods.push(TimeRange {
                        start: self.layers[s].timestamp,
                        end: self.layers[i-1].timestamp,
                    });
                    start = None;
                }
            }
        }

        // Close last period if it exists
        if let Some(s) = start {
            if !self.layers.is_empty() {
                periods.push(TimeRange {
                    start: self.layers[s].timestamp,
                    end: self.layers.last().unwrap().timestamp,
                });
            }
        }

        periods
    }

    pub fn compute_correlation(&self, other_intensities: &[f64]) -> f64 {
        let self_intensities: Vec<f64> = self.layers.iter().map(|l| l.intensity).collect();
        if self_intensities.len() != other_intensities.len() || self_intensities.is_empty() {
            return 0.0;
        }

        let n = self_intensities.len() as f64;
        let sum_x: f64 = self_intensities.iter().sum();
        let sum_y: f64 = other_intensities.iter().sum();
        let sum_x2: f64 = self_intensities.iter().map(|x| x * x).sum();
        let sum_y2: f64 = other_intensities.iter().map(|y| y * y).sum();
        let sum_xy: f64 = self_intensities.iter().zip(other_intensities.iter()).map(|(x, y)| x * y).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}
