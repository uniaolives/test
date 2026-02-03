use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use super::StructureType;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASIState {
    pub phase: super::ASIPhase,
    pub composition_state: CompositionState,
    pub last_processed: Option<DateTime<Utc>>,
    pub total_processed: u64,
    pub version: String,
}

impl Default for ASIState {
    fn default() -> Self {
        Self {
            phase: super::ASIPhase::Compositional,
            composition_state: CompositionState::default(),
            last_processed: None,
            total_processed: 0,
            version: "1.0.0".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompositionState {
    pub loaded_structures: Vec<StructureType>,
    pub total_compositions: u64,
}
pub struct ASIState;
