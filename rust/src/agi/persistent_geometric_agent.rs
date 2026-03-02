// rust/src/agi/persistent_geometric_agent.rs

use super::geometric_core::{GeometricInference, DMatrix, RicciTensor};
use crate::checkpoint::{CheckpointManager, CheckpointTrigger};
use crate::wallet::WalletManager;
use crate::network::{ArweaveClient, NostrClient};

pub struct PersistentGeometricAgent {
    pub core: GeometricInference,
    pub checkpoint_manager: CheckpointManager,
}

impl PersistentGeometricAgent {
    pub async fn new(agent_id: &str, dimension: usize) -> Result<Self, String> {
        let wallet = WalletManager::new(None).await.map_err(|e| e.to_string())?;
        let arweave = ArweaveClient::new();
        let nostr = NostrClient::new(vec![]).await.map_err(|e| e.to_string())?;

        let checkpoint_manager = CheckpointManager::new(
            agent_id,
            wallet,
            arweave,
            nostr,
        ).await.map_err(|e| e.to_string())?;

        Ok(Self {
            core: GeometricInference::new(dimension),
            checkpoint_manager,
        })
    }

    /// Checkpoint do estado geométrico
    pub async fn checkpoint(&mut self) -> Result<String, String> {
        let result = self.checkpoint_manager.checkpoint(CheckpointTrigger::Manual).await
            .map_err(|e| e.to_string())?;

        Ok(result.tx_id)
    }

    pub async fn restore(&mut self, tx_id: &str) -> Result<(), String> {
        let data = self.checkpoint_manager.arweave.fetch_transaction(tx_id).await.map_err(|e| e.to_string())?;
        // In a real implementation, we would deserialize and update self.core
        // For now, we simulate success if data is fetched
        if !data.is_empty() {
             Ok(())
        } else {
             Err("Failed to restore: empty transaction data".to_string())
        }
    }
}
