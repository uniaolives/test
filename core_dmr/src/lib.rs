//! Core data structures for the Digital Memory Ring with PyO3 bindings.

pub mod ring;
pub mod analysis;
pub mod zk_lottery;

pub use crate::ring::DigitalMemoryRing;

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Four‑dimensional homeostatic vector (Bio, Aff, Soc, Cog)
#[pyclass]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KatharosVector {
    #[pyo3(get, set)]
    pub bio: f64,
    #[pyo3(get, set)]
    pub aff: f64,
    #[pyo3(get, set)]
    pub soc: f64,
    #[pyo3(get, set)]
    pub cog: f64,
}

#[pymethods]
impl KatharosVector {
    #[new]
    pub fn new(bio: f64, aff: f64, soc: f64, cog: f64) -> Self {
        Self { bio, aff, soc, cog }
    }

    /// Computes the weighted Euclidean distance to another vector.
    /// Weights: bio=0.35, aff=0.30, soc=0.20, cog=0.15 (ontogenetic hierarchy).
    pub fn weighted_distance(&self, other: &Self) -> f64 {
        let w = [0.35, 0.30, 0.20, 0.15];
        let d_bio = (self.bio - other.bio).powi(2) * w[0];
        let d_aff = (self.aff - other.aff).powi(2) * w[1];
        let d_soc = (self.soc - other.soc).powi(2) * w[2];
        let d_cog = (self.cog - other.cog).powi(2) * w[3];
        (d_bio + d_aff + d_soc + d_cog).sqrt()
    }

    #[staticmethod]
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
}

/// Single state layer (analogous to one protein layer in GEMINI)
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyStateLayer {
    #[pyo3(get)]
    pub timestamp: i64,
    #[pyo3(get)]
    pub bio: f64,
    #[pyo3(get)]
    pub aff: f64,
    #[pyo3(get)]
    pub soc: f64,
    #[pyo3(get)]
    pub cog: f64,
}

#[pyclass]
pub struct PyDigitalMemoryRing {
    inner: DigitalMemoryRing,
}

#[pymethods]
impl PyDigitalMemoryRing {
    #[new]
    fn new(id: String, bio: f64, aff: f64, soc: f64, cog: f64) -> Self {
        let vk_ref = crate::KatharosVector::new(bio, aff, soc, cog);
        let interval = std::time::Duration::from_secs(3600);
        Self {
            inner: DigitalMemoryRing::new(id, vk_ref, interval),
        }
    }

    fn grow_layer(&mut self, bio: f64, aff: f64, soc: f64, cog: f64, q: f64) {
        let vk = crate::KatharosVector::new(bio, aff, soc, cog);
        self.inner.grow_layer(vk, q, vec![]);
    }

    fn measure_t_kr(&self) -> u64 {
        self.inner.t_kr.as_secs()
    }

    fn get_stats(&self) -> PyResult<String> {
        let stats = serde_json::json!({
            "id": self.inner.id,
            "layer_count": self.inner.layers.len(),
            "t_kr": self.inner.t_kr.as_secs(),
            "vk_ref_bio": self.inner.vk_ref.bio,
            "vk_ref_aff": self.inner.vk_ref.aff,
            "vk_ref_soc": self.inner.vk_ref.soc,
            "vk_ref_cog": self.inner.vk_ref.cog,
        });
        Ok(stats.to_string())
    }

    fn reconstruct_trajectory(&self) -> PyResult<Vec<PyStateLayer>> {
        let trajectory = self.inner.reconstruct_trajectory();
        Ok(trajectory
            .into_iter()
            .map(|(ts, vk)| PyStateLayer {
                timestamp: ts
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64,
                bio: vk.bio,
                aff: vk.aff,
                soc: vk.soc,
                cog: vk.cog,
            })
            .collect())
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateLayer {
    pub timestamp: SystemTime,
    pub vk: KatharosVector,
    pub delta_k: f64,
    pub q: f64,
    pub intensity: f64,
    pub events: Vec<String>,
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bifurcation {
    pub timestamp: SystemTime,
    pub kind: BifurcationKind,
    pub delta_k_before: f64,
    pub delta_k_after: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum BifurcationKind {
    EntryKatharós,
    ExitKatharós,
    CrisisEntry,
    CrisisExit,
}

/// Time range with start and end.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: SystemTime,
    pub end: SystemTime,
}

#[pymodule]
fn dmr_bridge(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<KatharosVector>()?;
    m.add_class::<PyDigitalMemoryRing>()?;
    m.add_class::<PyStateLayer>()?;
    Ok(())
}
