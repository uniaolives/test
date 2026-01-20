pub mod deployer;

use crate::ast::OntologyProgram;
use crate::backends::solidity::{SolidityBackend, CompiledContract};
use crate::compiler::{CompilerError, CompilerResult};
use async_trait::async_trait;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockchainTarget {
    SASC,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationLevel {
    TMR,
    FullSASC,
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
}
