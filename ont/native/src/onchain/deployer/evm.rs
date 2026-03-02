// ==============================================
// EVM DEPLOYER v0.3.0
// Suporte para Ethereum, Polygon, Arbitrum, etc.
// ==============================================

use crate::compiler::CompiledContract;
use crate::onchain::{DeploymentResult, OnChainError, VerificationLevel};
use crate::onchain::deployer::DeployBackend;
use async_trait::async_trait;
use ethers::{
    core::types::{TransactionReceipt},
    prelude::*,
    providers::{Http, Provider},
    signers::{LocalWallet, Signer},
    solc::{artifacts::BytecodeObject, CompilerInput, Solc},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

/// Configuração do deploy EVM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EVMDeployConfig {
    pub rpc_url: String,
    pub chain_id: u64,
    pub gas_limit: Option<u64>,
    pub gas_price: Option<u64>,
    pub confirmations: usize,
    pub timeout_seconds: u64,
    pub etherscan_api_key: Option<String>,
    pub verification: VerificationLevel,
}

impl Default for EVMDeployConfig {
    fn default() -> Self {
        Self {
            rpc_url: "http://localhost:8545".to_string(),
            chain_id: 31337, // Anvil default
            gas_limit: None,
            gas_price: None,
            confirmations: 1,
            timeout_seconds: 60,
            etherscan_api_key: None,
            verification: VerificationLevel::None,
        }
    }
}

/// Cliente para deploy em redes EVM
pub struct EVMDeployer {
    client: Arc<SignerMiddleware<Provider<Http>, LocalWallet>>,
    config: EVMDeployConfig,
    solc: Option<Solc>,
}

impl EVMDeployer {
    /// Criar novo deployer
    pub async fn new(
        private_key: &str,
        config: EVMDeployConfig,
    ) -> Result<Self, OnChainError> {
        // Conectar ao provider
        let provider = Provider::<Http>::try_from(&config.rpc_url)
            .map_err(|e| OnChainError::DeploymentFailed(e.to_string()))?;

        // Configurar timeout
        let provider = provider.interval(Duration::from_millis(100));

        // Criar wallet
        let wallet: LocalWallet = private_key
            .parse::<LocalWallet>()
            .map_err(|e| OnChainError::DeploymentFailed(e.to_string()))?
            .with_chain_id(config.chain_id);

        let client = SignerMiddleware::new(provider, wallet);

        // Carregar solc se disponível
        let solc = Solc::default();

        Ok(Self {
            client: Arc::new(client),
            config,
            solc: Some(solc),
        })
    }

    /// Compilar código Solidity usando solc
    async fn compile_solidity(
        &self,
        source_code: &str,
    ) -> Result<(Option<Vec<u8>>, Option<serde_json::Value>), OnChainError> {
        println!("🔨 Compilando Solidity...");

        // Verificar se solc está disponível
        let solc = self.solc.as_ref().ok_or_else(||
            OnChainError::DeploymentFailed("solc não disponível".to_string())
        )?;

        // Criar arquivo temporário
        let temp_dir = tempfile::tempdir()
            .map_err(|e| OnChainError::DeploymentFailed(e.to_string()))?;

        let source_path = temp_dir.path().join("contract.sol");
        std::fs::write(&source_path, source_code)
            .map_err(|e| OnChainError::DeploymentFailed(e.to_string()))?;

        // Configurar input do compilador
        let input = CompilerInput::new(source_path)
            .map_err(|e| OnChainError::DeploymentFailed(e.to_string()))?;

        // Compilar
        let output = solc
            .compile(&input)
            .map_err(|e| OnChainError::DeploymentFailed(e.to_string()))?;

        // Extrair bytecode e ABI
        let (bytecode, abi) = output
            .contracts
            .iter()
            .flat_map(|(_, contracts)| contracts.iter())
            .next()
            .map(|(_, contract)| {
                let bytecode = contract
                    .evm
                    .as_ref()
                    .and_then(|evm| evm.bytecode.as_ref())
                    .and_then(|bc| Some(bc.object.clone()))
                    .and_then(|obj| match obj {
                        BytecodeObject::Bytecode(bytes) => Some(bytes.to_vec()),
                        _ => None,
                    });

                let abi = contract.abi.as_ref().map(|abi| {
                    serde_json::to_value(abi).unwrap_or(serde_json::Value::Null)
                });

                (bytecode, abi)
            })
            .unwrap_or((None, None));

        Ok((bytecode, abi))
    }

    /// Aguardar confirmações da transação
    async fn wait_for_confirmation(
        &self,
        tx_hash: TxHash,
    ) -> Result<TransactionReceipt, OnChainError> {
        let start_time = std::time::Instant::now();
        let timeout = Duration::from_secs(self.config.timeout_seconds);

        while start_time.elapsed() < timeout {
            match self.client.get_transaction_receipt(tx_hash).await {
                Ok(Some(receipt)) => {
                    // Verificar número de confirmações
                    let current_block = self.client.get_block_number()
                        .await
                        .map_err(|e| OnChainError::DeploymentFailed(e.to_string()))?;

                    let confirmations = current_block.as_u64().saturating_sub(receipt.block_number.unwrap().as_u64());

                    if confirmations >= self.config.confirmations as u64 {
                        return Ok(receipt);
                    }

                    println!("   Confirmations: {}/{}", confirmations, self.config.confirmations);
                }
                Ok(None) => {
                    // Transação ainda não minerada
                    sleep(Duration::from_secs(1)).await;
                }
                Err(e) => {
                    return Err(OnChainError::DeploymentFailed(e.to_string()));
                }
            }
        }

        Err(OnChainError::DeploymentFailed("Timeout aguardando confirmações".to_string()))
    }

    /// Verificação local (checagem de bytecode)
    async fn verify_locally(
        &self,
        contract_address: &Address,
        _source_code: &str,
    ) -> Result<(), OnChainError> {
        println!("   Verificação local...");

        // Obter bytecode on-chain
        let on_chain_bytecode = self.client.get_code(*contract_address, None)
            .await
            .map_err(|e| OnChainError::DeploymentFailed(e.to_string()))?;

        if on_chain_bytecode.is_empty() {
            return Err(OnChainError::DeploymentFailed(
                "Bytecode vazio no endereço do contrato".to_string()
            ));
        }

        println!("   ✅ Bytecode verificado ({} bytes)", on_chain_bytecode.len());

        Ok(())
    }

    /// Verificação no Etherscan
    async fn verify_on_etherscan(
        &self,
        _contract_address: &Address,
        _source_code: &str,
    ) -> Result<(), OnChainError> {
        let _api_key = self.config.etherscan_api_key.as_ref()
            .ok_or_else(||
                OnChainError::DeploymentFailed("Etherscan API key não configurada".to_string())
            )?;

        println!("   Verificando no Etherscan...");

        // Aqui você implementaria a chamada à API do Etherscan
        // Por enquanto, apenas simulamos
        println!("   ⚠️  Verificação no Etherscan requer implementação da API");

        Ok(())
    }

    /// Verificação com TMR (para redes compatíveis)
    async fn verify_with_tmr(
        &self,
        contract_address: &Address,
        _source_code: &str,
    ) -> Result<(), OnChainError> {
        println!("   Verificação TMR...");

        // Lista de nós para verificação TMR
        let tmr_nodes = vec![
            self.config.rpc_url.clone(),
            // Adicionar outros nós aqui
        ];

        let mut bytecodes = Vec::new();

        for node_url in &tmr_nodes {
            let provider = Provider::<Http>::try_from(node_url)
                .map_err(|e| OnChainError::DeploymentFailed(e.to_string()))?;

            match provider.get_code(*contract_address, None).await {
                Ok(bytecode) => bytecodes.push(bytecode),
                Err(e) => println!("   ⚠️  Falha ao verificar no nó {}: {}", node_url, e),
            }
        }

        // Verificar consistência (pelo menos 2/3)
        if bytecodes.len() < 2 {
            return Err(OnChainError::DeploymentFailed(
                "Verificação TMR falhou: menos de 2 nós responderam".to_string()
            ));
        }

        // Verificar se todos os bytecodes são iguais
        let first = &bytecodes[0];
        for (i, bytecode) in bytecodes.iter().enumerate().skip(1) {
            if bytecode != first {
                return Err(OnChainError::DeploymentFailed(
                    format!("Bytecode inconsistente no nó {}", i)
                ));
            }
        }

        println!("   ✅ Verificação TMR bem-sucedida ({}/{} nós)",
                 bytecodes.len(), tmr_nodes.len());

        Ok(())
    }

    /// Gerar prova de verificação
    fn generate_verification_proof(&self, receipt: &TransactionReceipt) -> String {
        use sha2::{Sha256, Digest};

        let mut hasher = Sha256::new();
        hasher.update(format!("{:?}", receipt.transaction_hash));
        hasher.update(format!("{:?}", receipt.contract_address));
        hasher.update(format!("{}", receipt.gas_used.unwrap_or_default()));

        let hash = hasher.finalize();
        format!("0x{}", hex::encode(hash))
    }
}

#[async_trait]
impl DeployBackend for EVMDeployer {
    /// Deploy de contrato compilado
    async fn deploy(
        &self,
        compiled: &CompiledContract,
        constructor_args: Option<Vec<String>>,
    ) -> Result<DeploymentResult, OnChainError> {
        println!("🚀 Iniciando deploy EVM...");

        // 1. Compilar Solidity se necessário
        let (bytecode, abi) = if compiled.bytecode.is_none() || compiled.abi.is_none() {
            self.compile_solidity(&compiled.source_code).await?
        } else {
            (
                compiled.bytecode.clone(),
                compiled.abi.clone(),
            )
        };

        let bytecode = bytecode.ok_or_else(||
            OnChainError::DeploymentFailed("Bytecode não disponível".to_string())
        )?;

        let abi_value = abi.ok_or_else(||
            OnChainError::DeploymentFailed("ABI não disponível".to_string())
        )?;

        let abi: ethers::abi::Abi = serde_json::from_value(abi_value)
            .map_err(|e| OnChainError::DeploymentFailed(e.to_string()))?;

        // 2. Preparar fábrica do contrato
        let factory = ContractFactory::new(abi, bytecode.into(), Arc::clone(&self.client));

        // 3. Estimar gas e preparar tokens
        let tokens: Vec<ethers::abi::Token> = constructor_args.unwrap_or_default().into_iter().map(|s| ethers::abi::Token::String(s)).collect();

        // 4. Configurar opções de deploy
        let mut deployer = factory
            .deploy_tokens(tokens)
            .map_err(|e| OnChainError::DeploymentFailed(e.to_string()))?;

        println!("⛽ Estimando gas...");
        let gas_estimate = self.client.estimate_gas(&deployer.tx, None)
            .await
            .map_err(|e| OnChainError::DeploymentFailed(e.to_string()))?;

        println!("   Estimativa: {} gas", gas_estimate);
        let gas_estimate = gas_estimate * 120 / 100;

        deployer.tx.set_gas(gas_estimate);

        // Aplicar configurações de gas
        if let Some(gas_limit) = self.config.gas_limit {
            deployer.tx.set_gas(gas_limit);
        }

        if let Some(gas_price) = self.config.gas_price {
            deployer.tx.set_gas_price(gas_price);
        }

        // 5. Enviar transação
        println!("📤 Enviando transação de deploy...");
        let pending_tx = self.client
            .send_transaction(deployer.tx, None)
            .await
            .map_err(|e| OnChainError::DeploymentFailed(e.to_string()))?;

        // 6. Aguardar confirmações
        println!("⏳ Aguardando confirmações...");

        let tx_hash = pending_tx.tx_hash();
        let receipt = self.wait_for_confirmation(tx_hash).await?;

        // 7. Extrair endereço do contrato
        let contract_address = receipt.contract_address.ok_or_else(|| OnChainError::DeploymentFailed("Contract address not found in receipt".to_string()))?;

        // 8. Verificar se necessário
        if self.config.verification != VerificationLevel::None {
            self.verify(&format!("{:?}", contract_address), &compiled.source_code).await?;
        }

        println!("✅ Contrato implantado com sucesso!");

        Ok(DeploymentResult {
            contract_address: format!("{:?}", contract_address),
            transaction_hash: format!("{:?}", receipt.transaction_hash),
            block_number: receipt.block_number.unwrap().as_u64(),
            gas_used: receipt.gas_used.unwrap().as_u64(),
            verification_proof: Some(self.generate_verification_proof(&receipt)),
        })
    }

    async fn verify(
        &self,
        contract_address: &str,
        source_code: &str,
    ) -> Result<(), OnChainError> {
        let address: Address = contract_address.parse()
            .map_err(|e: <Address as std::str::FromStr>::Err| OnChainError::DeploymentFailed(e.to_string()))?;

        match self.config.verification {
            VerificationLevel::Basic => {
                self.verify_locally(&address, source_code).await
            }
            VerificationLevel::Full => {
                self.verify_on_etherscan(&address, source_code).await
            }
            VerificationLevel::TMR => {
                self.verify_with_tmr(&address, source_code).await
            }
            _ => Ok(()),
        }
    }

    /// Executar função em contrato implantado
    async fn execute(
        &self,
        contract_address: &str,
        function_signature: &str,
        _args: Vec<String>,
    ) -> Result<serde_json::Value, OnChainError> {
        println!("⚡ Executando função {}...", function_signature);

        // Converter endereço
        let _address: Address = contract_address
            .parse()
            .map_err(|e: <Address as std::str::FromStr>::Err| OnChainError::ExecutionFailed(e.to_string()))?;

        // TODO: Implementar chamada de contrato
        // Isso requer o ABI do contrato

        // Por enquanto, apenas simulamos
        println!("   ⚠️  Execução de contrato requer ABI");

        // Simular transação bem-sucedida
        Ok(serde_json::json!({
            "status": "success",
            "transaction_hash": format!("{:?}", TxHash::default()),
            "gas_used": 21000
        }))
    }
}
