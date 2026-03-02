pub mod vmcore_orchestrator;
pub mod frag_matrix_113;
pub mod agnostic_dispatch;
pub mod hardware_orbit;
pub mod phi_enforcer;

use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use std::sync::Arc;

pub type TaskId = String;

#[derive(Error, Debug)]
pub enum UnifiedError {
    #[error("Constitutional violation: {0}")]
    ConstitutionalViolation(String),
    #[error("Phi out of bounds: {0}")]
    PhiOutOfBounds(f64),
    #[error("System not running")]
    NotRunning,
    #[error("Fragment error: {0}")]
    Fragment(#[from] FragError),
    #[error("Dispatch error: {0}")]
    Dispatch(#[from] DispatchError),
    #[error("Orbit error: {0}")]
    Orbit(#[from] OrbitError),
    #[error("Consensus error: {0}")]
    Consensus(String),
    #[error("Other error: {0}")]
    Other(String),
}

impl From<anyhow::Error> for UnifiedError {
    fn from(err: anyhow::Error) -> Self {
        UnifiedError::Other(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, UnifiedError>;

#[async_trait]
pub trait VMCoreInterface: Send + Sync {
    async fn sync_phi(&self, phi: f64) -> Result<()>;
    async fn process_instructions(
        &self,
        data: &[u8],
        dispatch: Arc<agnostic_dispatch::AgnosticDispatch>,
    ) -> Result<VMCoreResult>;
}

#[async_trait]
pub trait OrchestratorInterface: Send + Sync {
    async fn sync_phi(&self, phi: f64) -> Result<()>;
    async fn coordinate_result(
        &self,
        vmcore_result: &VMCoreResult,
        protocol: ConsensusProtocol,
    ) -> Result<UnifiedExecutionResult>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VMCoreResult {
    pub processed_data: Vec<u8>,
    pub instructions_processed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedExecutionResult {
    pub success: bool,
    pub phi_before: f64,
    pub phi_after: f64,
    pub frags_used: usize,
    pub instructions_processed: u64,
    pub tmr_rounds: u64,
    pub agnostic_verified: bool,
    pub execution_time: std::time::Duration,
    pub tmr_success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedKernel {
    pub id: String,
    pub tasks: Vec<UnifiedTask>,
    pub total_instructions: u64,
    pub agnosticism_level: u32,
    pub constitutional_signature: [u8; 32],
    pub phi_power_requirement: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnifiedTask {
    Atomic {
        opcode: agnostic_dispatch::AtomicOpCode,
        operands: Vec<Operand>,
        atomicity: Atomicity,
    },
    Compute {
        algorithm: ComputeAlgorithm,
        input_size: usize,
        vectorization: bool,
    },
    Orchestration {
        coordination: CoordinationType,
        agents_required: u32,
        timeout_ms: u64,
    },
    Verification {
        check_type: VerificationType,
        tolerance: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operand {
    Memory(u64),
    Value(u64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Atomicity {
    Relaxed,
    Release,
    Acquire,
    AcqRel,
    SequentiallyConsistent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputeAlgorithm {
    SHA3_256,
    BLAKE3,
    ChaCha20,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationType {
    TMRDispatch,
    ByzantineConsensus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationType {
    PhiPower40,
    HardwareAttestation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedActivity {
    pub timestamp: u64,
    pub activity_level: f64,
    pub phi_before: f64,
    pub phi_after: f64,
    pub phi_power: u32,
    pub frags_used: usize,
    pub instructions_processed: u64,
    pub tmr_rounds: u64,
    pub agnostic_verified: bool,
    pub constitutional_signature: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedConfig {
    pub total_frags: usize,
    pub dispatch_bars: usize,
    pub tmr_config: TMRConfig,
    pub required_agnosticism: AgnosticLevel,
    pub hardware_backends: Vec<HardwareBackend>,
    pub consensus_protocol: ConsensusProtocol,
    pub execution_limits: ExecutionLimits,
}

impl Default for UnifiedConfig {
    fn default() -> Self {
        Self {
            total_frags: 113,
            dispatch_bars: 92,
            tmr_config: TMRConfig {
                groups: 36,
                replicas: 3,
                byzantine_tolerance: 1,
            },
            required_agnosticism: AgnosticLevel::Pure {
                vendor_lock_tolerance: 0.0,
                hardware_dependence_tolerance: 0.0,
                platform_dependence_tolerance: 0.0,
            },
            hardware_backends: vec![
                HardwareBackend::Cranelift,
                HardwareBackend::SpirV,
                HardwareBackend::Wasi,
            ],
            consensus_protocol: ConsensusProtocol::ByzantineTolerant {
                total_nodes: 113,
                faulty_tolerance: 37,
            },
            execution_limits: ExecutionLimits {
                max_kernels_per_second: 1000,
                max_memory_mb: 4096,
                max_concurrent_workflows: 36,
                timeout_seconds: 30,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgnosticLevel {
    Pure {
        vendor_lock_tolerance: f64,
        hardware_dependence_tolerance: f64,
        platform_dependence_tolerance: f64,
    },
    Hybrid(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum HardwareBackend {
    Cranelift,
    SpirV,
    Wasi,
    BareMetal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusProtocol {
    ByzantineTolerant {
        total_nodes: usize,
        faulty_tolerance: usize,
    },
    TMR,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TMRConfig {
    pub groups: usize,
    pub replicas: usize,
    pub byzantine_tolerance: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionLimits {
    pub max_kernels_per_second: u64,
    pub max_memory_mb: u64,
    pub max_concurrent_workflows: usize,
    pub timeout_seconds: u64,
}

// Stub implementations
pub struct VMcoreImpl {
    _phi: f64,
}

impl VMcoreImpl {
    pub fn new(phi: f64) -> Result<Self> {
        Ok(Self { _phi: phi })
    }
}

#[async_trait]
impl VMCoreInterface for VMcoreImpl {
    async fn sync_phi(&self, _phi: f64) -> Result<()> {
        Ok(())
    }
    async fn process_instructions(
        &self,
        data: &[u8],
        _dispatch: Arc<agnostic_dispatch::AgnosticDispatch>,
    ) -> Result<VMCoreResult> {
        Ok(VMCoreResult {
            processed_data: data.to_vec(),
            instructions_processed: 100, // mock
        })
    }
}

pub struct OrchestratorImpl {
    phi: f64,
}

impl OrchestratorImpl {
    pub fn new(phi: f64) -> Result<Self> {
        Ok(Self { phi })
    }
}

#[async_trait]
impl OrchestratorInterface for OrchestratorImpl {
    async fn sync_phi(&self, _phi: f64) -> Result<()> {
        Ok(())
    }
    async fn coordinate_result(
        &self,
        vmcore_result: &VMCoreResult,
        _protocol: ConsensusProtocol,
    ) -> Result<UnifiedExecutionResult> {
        Ok(UnifiedExecutionResult {
            success: true,
            phi_before: self.phi,
            phi_after: self.phi,
            frags_used: 113,
            instructions_processed: vmcore_result.instructions_processed,
            tmr_rounds: 1,
            agnostic_verified: true,
            execution_time: std::time::Duration::from_millis(10),
            tmr_success: true,
        })
    }
}

#[derive(Error, Debug)]
pub enum FragError {
    #[error("No suitable frag found for task {0}")]
    NoSuitableFrag(usize),
}

#[derive(Error, Debug)]
pub enum DispatchError {
    #[error("Invalid bar count: {0}")]
    InvalidBarCount(usize),
    #[error("Channel closed")]
    ChannelClosed,
    #[error("Unsupported opcode")]
    UnsupportedOpCode,
    #[error("No suitable bar")]
    NoSuitableBar,
}

#[derive(Error, Debug)]
pub enum OrbitError {
    #[error("Insufficient replicas for task {0}: {1}")]
    InsufficientReplicas(String, usize),
}
