//! Core data structures for the Digital Memory Ring.

pub mod ring;
pub mod analysis;
pub mod validation;
pub mod simple_agent;

pub use crate::ring::DigitalMemoryRing;

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Four‑dimensional homeostatic vector (Bio, Aff, Soc, Cog)
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KatharosVector {
    pub bio: f64,
    pub aff: f64,
    pub soc: f64,
    pub cog: f64,
}

impl KatharosVector {
    /// Creates a new vector with given components.
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

    /// Returns a zero vector (used as placeholder).
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
}

/// Single state layer (analogous to one protein layer in GEMINI)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateLayer {
    /// Absolute timestamp (nanoseconds since epoch)
    pub timestamp: SystemTime,
    /// Homeostatic vector at this moment
    pub vk: KatharosVector,
    /// Deviation from reference (ΔK)
    pub delta_k: f64,
    /// Qualic permeability (integration strength)
    pub q: f64,
    /// Fluorescence analog (0.0 = baseline, 1.0 = maximum activation)
    pub intensity: f64,
    /// Optional event markers (e.g., perturbations)
    pub events: Vec<String>,
}

/// Detected bifurcation (transition between regimes)
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
