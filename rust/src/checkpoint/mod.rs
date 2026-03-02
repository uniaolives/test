// rust/src/checkpoint/mod.rs
mod manager;
mod scrubber;
mod upload_strategy;

pub use manager::CheckpointManager;
pub use scrubber::StateScrubber;
pub use upload_strategy::{UploadStrategy, UploadContext};

#[derive(Debug, Clone)]
pub struct CheckpointResult {
    pub tx_id: String,
    pub size_bytes: usize,
    pub cost_winston: u64,
    pub duration_ms: u128,
    pub strategy_used: UploadStrategy,
    pub validation_passed: bool,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub enum CheckpointTrigger {
    Periodic,
    SizeBased,
    ImportantEvent,
    Manual,
    #[allow(dead_code)]
    ErrorRecovery,
}
