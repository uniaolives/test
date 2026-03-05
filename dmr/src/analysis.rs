//! Analysis tools for VK trajectories.

use crate::{DigitalMemoryRing, KatharosVector};
use std::time::{SystemTime, UNIX_EPOCH};

/// Extracted trajectory with time series.
#[derive(Clone, Debug)]
pub struct VKTrajectory {
    pub timestamps: Vec<SystemTime>,
    pub vk_history: Vec<KatharosVector>,
    pub delta_k_history: Vec<f64>,
    pub q_history: Vec<f64>,
    pub intensity_history: Vec<f64>,
}

impl VKTrajectory {
    /// Build trajectory from a memory ring.
    pub fn from_ring(ring: &DigitalMemoryRing) -> Self {
        let mut timestamps = Vec::new();
        let mut vk_history = Vec::new();
        let mut delta_k_history = Vec::new();
        let mut q_history = Vec::new();
        let mut intensity_history = Vec::new();

        for layer in &ring.layers {
            timestamps.push(layer.timestamp);
            vk_history.push(layer.vk.clone());
            delta_k_history.push(layer.delta_k);
            q_history.push(layer.q);
            intensity_history.push(layer.intensity);
        }
        Self {
            timestamps,
            vk_history,
            delta_k_history,
            q_history,
            intensity_history,
        }
    }

    /// Export to CSV (for plotting).
    pub fn to_csv(&self, filename: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(filename)?;
        writeln!(file, "timestamp_secs,bio,aff,soc,cog,delta_k,q,intensity")?;

        for i in 0..self.timestamps.len() {
            let ts = self.timestamps[i]
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();
            let vk = &self.vk_history[i];
            writeln!(
                file,
                "{},{},{},{},{},{},{},{}",
                ts,
                vk.bio,
                vk.aff,
                vk.soc,
                vk.cog,
                self.delta_k_history[i],
                self.q_history[i],
                self.intensity_history[i]
            )?;
        }
        Ok(())
    }

    /// Compute Pearson correlation between two time series (for validation).
    pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }
        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_x2: f64 = x.iter().map(|v| v * v).sum();
        let sum_y2: f64 = y.iter().map(|v| v * v).sum();
        let sum_xy: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Compare DMR intensity with external GEMINI data (simulated).
    pub fn compare_with_gemini(&self, gemini_intensity: &[f64]) -> f64 {
        // Truncate to min length
        let len = self.intensity_history.len().min(gemini_intensity.len());
        let dmr = &self.intensity_history[..len];
        let gem = &gemini_intensity[..len];
        Self::pearson_correlation(dmr, gem)
    }
}
