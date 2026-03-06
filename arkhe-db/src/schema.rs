use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Handover {
    pub id: u64,
    pub timestamp: DateTime<Utc>,
    pub source_epoch: u32,
    pub target_epoch: u32,
    pub description: String,
    pub phi_q_before: f64,
    pub phi_q_after: f64,
    pub quantum_interest: f64,
    pub status: HandoverStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HandoverStatus {
    Accepted,
    Rejected,
    Pending,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VacuumSnapshot {
    pub timestamp: DateTime<Utc>,
    pub timestamp_ns: u64,
    pub global_phi_q: f64,
    pub wave_cloud_active: bool,
    pub multi_band_metrics: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FutureCommitment {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub target_at: DateTime<Utc>,
    pub prediction_hash: String,
    pub validation_signature: Option<String>,
    pub status: CommitmentStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CommitmentStatus {
    Pending,
    Validated,
    Invalidated,
    Expired,
}
