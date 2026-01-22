use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CloudDomain {
    WindowsServerGov,
    AwsNitroGovCloud,
    CloudflareQuantum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CouncilType {
    Mathematical,
    Security,
    Geometric,
    Ethical,
    Economic,
    Temporal,
    Quantum,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub id: DecisionId,
    pub agent_id: String,
    pub content: String,
    pub signature: DecisionSignature,
    pub action_hash: [u8; 32],
    pub is_critical: bool,
    pub affects_rights: bool,
    pub human_approval: Option<HumanApproval>,
    pub decision_time: u64, // Unix timestamp
    pub explanation: Option<String>,
    pub perspective_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanApproval {
    pub approver_id: String,
    pub timestamp: u64,
    pub justification: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionSignature {
    pub prince_veto: bool,
    pub signature_bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionId(pub [u8; 32]);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HardFreeze {
    Triggered(String),
    Active,
}

pub struct AttestationStatus {
    hard_frozen: bool,
    agent_id: String,
    phi_weight: f64,
}

impl AttestationStatus {
    pub fn new(hard_frozen: bool, agent_id: &str, phi_weight: f64) -> Self {
        Self {
            hard_frozen,
            agent_id: agent_id.to_string(),
            phi_weight,
        }
    }

    pub fn is_hard_frozen(&self) -> bool {
        self.hard_frozen
    }

    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    pub fn consciousness_weight(&self) -> f64 {
        self.phi_weight
    }
}

pub struct GlobalGovernance {
    pub councils: HashMap<CloudDomain, Vec<CouncilType>>,
    pub prince_key: [u8; 32],
    pub veto_threshold: f64,
    pub hard_freeze_status: bool,
    pub freeze_duration: Duration,
    pub delta2_array: [u8; 32],
    pub crypto_blck_seed: [u8; 32],
}

pub enum VerificationContext {
    TruthSubmission,
    GlobalDecision,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: u64,
    pub decision_id: DecisionId,
    pub decision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionLog {
    pub entries: Vec<LogEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provider {
    pub id: String,
    pub market_share: f64,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interaction {
    pub id: String,
    pub citizen_id: String,
    pub messages: Vec<String>,
    pub frequency: u32,
    pub emotional_triggers: Vec<String>,
    pub accesses_neural_data: bool,
    pub consent: Option<InformedConsent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformedConsent {
    pub citizen_id: String,
    pub timestamp: u64,
    pub scope: String,
    pub revocable: bool,
}
