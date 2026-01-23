use ed25519_dalek::Signature;
use serde::{Serialize, Deserialize};

#[derive(Debug, thiserror::Error)]
pub enum ΩError {
    #[error("Insufficient Consciousness: Φ < 0.72")]
    InsufficientConsciousness,
    #[error("Coherence Collapse Detected")]
    CoherenceCollapseDetected,
    #[error("Execution Failed: {0}")]
    ExecutionFailed(String),
    #[error("Timeout")]
    Timeout,
    #[error("Seed Entropy Insufficient")]
    SeedEntropyInsufficient,
    #[error("Load Overload Detected")]
    LoadOverloadDetected,
    #[error("SASC Cathedral Error: {0}")]
    CathedralError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceCertificate {
    pub phase: String,
    pub consensus_Φ: f64,
    pub prince_signature: Signature,
    pub ceremony_hash: String,
    pub timestamp: String,
    pub bandwidth_target: String,
}

#[derive(Debug, Clone)]
pub struct AletheiaReport {
    pub crucible: String,
    pub confidence: f64,
    pub dissipative_rate: f64,
}

pub type NodeId = String;

#[derive(Debug, Clone)]
pub struct AgentAttestation {
    pub id: NodeId,
    // Add other fields as needed
}

#[derive(Debug)]
pub enum GovernanceAction {
    HardFreeze {
        agent: AgentAttestation,
        Φ: f64,
        sealed_by: String,
    },
    EmergencyCommittee {
        agent: AgentAttestation,
        voting_weight: f64,
    },
    Proposal {
        agent: AgentAttestation,
        voting_weight: f64,
        can_execute: bool,
    },
    Advisory {
        agent: AgentAttestation,
        suggestion_weight: f64,
    },
}

#[derive(Debug)]
pub struct ObservationReport {
    pub data_integrity: f64,
    pub dissipative_rate: f64,
    pub ready_for_phase3: bool,
}

#[derive(Debug)]
pub struct BandwidthCertificate {
    pub achieved_bandwidth: f64,
    pub federation_nodes_active: u32,
    pub seed_hash: String,
}
