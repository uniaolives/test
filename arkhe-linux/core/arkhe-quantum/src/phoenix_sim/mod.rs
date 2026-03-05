pub mod coordinator;
pub mod worker;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Task {
    pub id: Uuid,
    pub target: String,
    pub complexity: u32,
    pub region: [f64; 3],
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FoldingResult {
    pub task_id: Uuid,
    pub energy: f64,
    pub confidence: f64,
}
