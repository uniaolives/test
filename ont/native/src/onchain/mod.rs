pub mod deployer;

use crate::ast::OntologyProgram;
use crate::backends::solidity::SolidityBackend;
use crate::compiler::{CompilerError, CompilerResult, CompiledContract};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockchainTarget {
    SASC,
    EVM,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationLevel {
    None,
    Basic,
    Full,
    TMR,
    FullSASC,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentResult {
    pub contract_address: String,
    pub transaction_hash: String,
    pub block_number: u64,
    pub gas_used: u64,
    pub verification_proof: Option<String>,
}

pub struct OnChainAutomation {
    pub target: BlockchainTarget,
    pub verification: VerificationLevel,
    pub sasc_enabled: bool,
}

impl OnChainAutomation {
    pub fn new(target: BlockchainTarget, verification: VerificationLevel, sasc_enabled: bool) -> Self {
        Self { target, verification, sasc_enabled }
    }

    pub fn compile_to_smart_contract(&self, program: &OntologyProgram) -> CompilerResult<CompiledContract> {
        let backend = SolidityBackend::new(true, "0.8.24".to_string(), true);
        backend.compile(program)
    }

    pub async fn automate(&self, program: &OntologyProgram) -> Result<(), OnChainError> {
        // Mock automation logic
        if program.functions.iter().any(|f| f.name == "bad_function") {
            return Err(OnChainError::ConstraintViolation("Violation in bad_function".to_string()));
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OnChainError {
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),
    #[error("Compiler error: {0}")]
    Compiler(#[from] CompilerError),
    #[error("Deployment failed: {0}")]
    DeploymentFailed(String),
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
}
