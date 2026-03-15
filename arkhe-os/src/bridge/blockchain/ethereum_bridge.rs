// arkhe-os/src/bridge/blockchain/ethereum_bridge.rs

use ethers::prelude::*;
use crate::orb::core::OrbPayload;
use crate::bridge::BridgeError;

pub struct EthereumBridge {
    provider: Provider<Ws>,
    wallet: LocalWallet,
    orb_contract: Address,
}

impl EthereumBridge {
    pub async fn new(ws_url: &str, private_key: &str, contract_addr: Address) -> Result<Self, BridgeError> {
        let provider = Provider::<Ws>::connect(ws_url).await
            .map_err(|e| BridgeError::Blockchain(e.to_string()))?;
        let wallet = private_key.parse::<LocalWallet>()
            .map_err(|e| BridgeError::Blockchain(e.to_string()))?;

        Ok(Self {
            provider,
            wallet,
            orb_contract: contract_addr,
        })
    }

    /// Envia Orb para smart contract
    pub async fn send_orb(&self, orb: &OrbPayload) -> Result<H256, BridgeError> {
        let orb_bytes = orb.to_bytes();

        // In a real implementation, we'd use an ABI-generated contract binding
        // Here we simulate the call
        let tx = TransactionRequest::new()
            .to(self.orb_contract)
            .data(orb_bytes) // Simplified: just raw bytes as data
            .from(self.wallet.address());

        let tx_typed: ethers::types::transaction::eip2718::TypedTransaction = tx.into();
        let pending_tx = self.provider.send_transaction(tx_typed, None).await
            .map_err(|e| BridgeError::Blockchain(e.to_string()))?;

        Ok(*pending_tx)
    }
}
