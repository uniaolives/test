pub mod sasc;
pub mod evm;

use crate::compiler::CompiledContract;
use crate::onchain::{DeploymentResult, OnChainError, VerificationLevel};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[async_trait]
pub trait DeployBackend: Send + Sync {
    /// Deploy de contrato
    async fn deploy(
        &self,
        compiled: &CompiledContract,
        constructor_args: Option<Vec<String>>,
    ) -> Result<DeploymentResult, OnChainError>;

    /// Verificar contrato implantado
    async fn verify(
        &self,
        contract_address: &str,
        source_code: &str,
    ) -> Result<(), OnChainError>;

    /// Executar função em contrato
    async fn execute(
        &self,
        contract_address: &str,
        function: &str,
        args: Vec<String>,
    ) -> Result<serde_json::Value, OnChainError>;
}

/// Configuração de deploy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeployConfig {
    pub target: DeployTarget,
    pub verification: VerificationLevel,
    pub network: String,
    pub private_key: Option<String>,
    pub rpc_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeployTarget {
    EVM(evm::EVMDeployConfig),
    SASC(sasc::SASCDeployConfig),
}

/// Fábrica de backends de deploy
pub struct DeployerFactory;

impl DeployerFactory {
    pub async fn create(config: DeployConfig) -> Result<Box<dyn DeployBackend>, OnChainError> {
        match config.target {
            DeployTarget::EVM(evm_config) => {
                let private_key = config.private_key.ok_or_else(||
                    OnChainError::DeploymentFailed("Private key necessária para EVM".to_string())
                )?;

                let deployer = evm::EVMDeployer::new(&private_key, evm_config).await?;
                Ok(Box::new(deployer))
            }
            DeployTarget::SASC(sasc_config) => {
                let deployer = sasc::SASCDeployer::new(sasc_config).await?;
                Ok(Box::new(deployer))
            }
        }
    }
}
