// rust/src/memory/dmr/types.rs
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Basic Katharós Vector for DMR implementation
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KatharosVector {
    pub components: [f64; 4],
}

impl KatharosVector {
    pub fn new(bio: f64, aff: f64, soc: f64, cog: f64) -> Self {
        Self {
            components: [bio, aff, soc, cog],
        }
    }
}

impl Default for KatharosVector {
    fn default() -> Self {
        Self {
            components: [0.0; 4],
        }
    }
}

/// Analog to a single GEMINI protein assembly layer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateLayer {
    /// Absolute timestamp (analogous to radiometric dating)
    pub timestamp: SystemTime,

    /// Katharós Vector at this moment
    pub vk: KatharosVector,

    /// Deviation from homeostatic reference
    pub delta_k: f64,

    /// Qualic permeability (integration strength)
    pub q: f64,

    /// Fluorescence analog (0.0 = baseline, 1.0 = maximum activation)
    pub intensity: f64,

    /// Event markers (optional, like GEMINI's activity reporters)
    pub events: Vec<CellularEvent>,
}

/// Event markers (like GEMINI's activity reporters)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CellularEvent {
    pub event_type: String,
    pub metadata: String,
}

/// Detected bifurcation points
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bifurcation {
    pub timestamp: SystemTime,
    pub bifurcation_type: BifurcationType,
    pub delta_k: f64,
}

/// Types of bifurcations
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum BifurcationType {
    CrisisEntry,
    CrisisExit,
    ResonancePoint,
    PhaseTransition,
}

/// System state used for growing layers
#[derive(Clone, Debug)]
pub struct SystemState {
    pub vk: KatharosVector,
    pub entropy: f64,
    pub events: Vec<CellularEvent>,
}
