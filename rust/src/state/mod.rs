// rust/src/state/mod.rs
mod resilient_state;
mod compression;
mod validation;

pub use resilient_state::{ResilientState, AgentMemory, DecisionRecord, KnowledgeGraph, KnowledgeNode, NodeType, TorsionMetrics};
pub use compression::StateCompressor;
pub use validation::StateValidator;

// Re-exportar tipos comuns
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone)]
pub struct StateSnapshot {
    pub full_state: ResilientState,
    pub compressed_size: usize,
    pub compression_ratio: f64,
}

pub struct StateMetadata {
    pub tx_id: String,
    pub size_bytes: usize,
    pub timestamp: u64,
    pub network: crate::wallet::Network,
    pub cost_winston: u64,
}
