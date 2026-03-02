#![no_std]

extern crate alloc;
use alloc::vec::Vec;
use alloc::string::String;
use serde::{Serialize, Deserialize};

// ============ CONSTITUTIONAL TYPES ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BricsAttestation {
    pub device_id: [u8; 32],
    pub phi_score: f32,
    pub location: [f64; 2],
    pub signature: [u8; 64],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceState {
    pub phi: f32,
    pub location: [f64; 2],
    pub battery: u8,
    pub storage_free: u64,
}

impl DeviceState {
    pub fn current_phi(&self) -> f32 {
        self.phi
    }

    pub fn consume_phi_gas(&self, _cost: u64) {
        // Logica de consumo de gás (Φ reduzido temporariamente)
        // Em um ambiente no_std real, isso alteraria o estado do dispositivo via syscall ou hardware register
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstitutionalStatus {
    Healthy,
    Degraded(f32),
    Violated(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstitutionalChecks {
    pub required_invariants: Vec<String>,
}

impl ConstitutionalChecks {
    pub fn verify(&self, _state: &DeviceState) -> bool {
        // Implementação simplificada para o canon
        true
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContractError {
    InsufficientPhi { required: f32, available: f32 },
    InvalidSignature,
    ConstitutionalViolation,
    SandboxError(String),
    LedgerError(String),
}

// ============ SYSTEM STATE (HEXADIMENSIONAL) ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Status {
    Active,
    Standby,
    Disabled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionState {
    pub name: String,
    pub module: String,
    pub status: Status,
    pub phi: f32,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HexadimensionalState {
    pub time: DimensionState,
    pub space: DimensionState,
    pub matter: DimensionState,
    pub cyber: DimensionState,
    pub orbital: DimensionState,
    pub mobile: DimensionState,
    pub judicial: DimensionState,
}

pub mod mobile_smart_contracts;

pub struct WasmSandbox;
impl WasmSandbox {
    pub fn new(_bytecode: &[u8]) -> Self { Self }
    pub fn set_limits(&mut self, _cycles: u64, _mem: usize) {}
    pub fn add_constitutional_api(&mut self) {}
    pub fn execute(&mut self, _input: &[u8]) -> Result<Vec<u8>, ContractError> {
        Ok(alloc::vec![0u8; 32])
    }
}

// ============ ALERT SYSTEM ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstitutionalAlert {
    pub alert_id: String,
    pub timestamp: u64,
    pub bill_id: String,
    pub violation_type: String,
    pub severity: ViolationSeverity,
    pub affected_devices: u64,
    pub required_action: String,
    pub expiration: u64,
}

// ============ IDENTITY & VOTE (EXAMPLES) ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BricsCredential {
    pub issuer: String,
    pub data_hash: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IdentityStatus {
    Verified,
    Pending,
    Transferred,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReceipt {
    pub did: [u8; 32],
    pub timestamp: u64,
    pub location: [f64; 2],
    pub phi_at_verification: f32,
    pub signature: [u8; 64],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IdentityError {
    InsufficientPhi,
    BiometricMismatch,
    CredentialsInvalid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferReceipt {
    pub old_did: [u8; 32],
    pub new_pubkey: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransferError {
    InsufficientPhi,
    OwnershipProofInvalid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoteChoice {
    Yes,
    No,
    Abstain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssetType {
    ConstitutionalValue,
    LaborCredit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteReceipt {
    pub vote_id: [u8; 32],
    pub weight: f32,
    pub confirmation_time: u64,
    pub constitutional_validation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoteError {
    NotEligible,
    OutsideTerritory,
    InsufficientPhi,
}
