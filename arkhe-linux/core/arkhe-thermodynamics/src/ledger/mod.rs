use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use anyhow::Result;
use log::{info, error};

/// Um handover completo, incluindo o hash do anterior e a assinatura.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StoredHandover {
    /// Dados do handover serializados.
    pub data: Vec<u8>,
    /// Hash SHA-256 do handover anterior (32 bytes).
    pub prev_hash: [u8; 32],
    /// Timestamp do momento do armazenamento (nanossegundos).
    pub stored_at: i64,
}

/// Ledger imutável (Simulado em memória devido a restrições de ambiente,
/// mas projetado para persistência).
pub struct OmegaLedger {
    storage: Arc<Mutex<HashMap<[u8; 32], StoredHandover>>>,
    last_hash: Arc<Mutex<Option<[u8; 32]>>>,
}

impl OmegaLedger {
    pub fn open() -> Result<Self> {
        info!("Opening Omega Ledger (Volatile Memory Simulation)...");
        Ok(OmegaLedger {
            storage: Arc::new(Mutex::new(HashMap::new())),
            last_hash: Arc::new(Mutex::new(None)),
        })
    }

    /// Anexa um novo handover ao ledger.
    pub async fn append(&self, handover_data: Vec<u8>) -> Result<[u8; 32]> {
        let mut hasher = Sha256::new();
        hasher.update(&handover_data);

        let prev_hash = {
            let last = self.last_hash.lock().await;
            last.unwrap_or([0u8; 32])
        };
        hasher.update(&prev_hash);

        // Simulate timestamp
        let now: i64 = 123456789; // Fixed for simulation
        hasher.update(&now.to_le_bytes());

        let hash_result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&hash_result);

        let stored = StoredHandover {
            data: handover_data,
            prev_hash,
            stored_at: now,
        };

        let mut storage = self.storage.lock().await;
        storage.insert(hash, stored);

        let mut last = self.last_hash.lock().await;
        *last = Some(hash);

        Ok(hash)
    }

    pub async fn get(&self, hash: &[u8; 32]) -> Result<Option<StoredHandover>> {
        let storage = self.storage.lock().await;
        Ok(storage.get(hash).cloned())
    }

    pub async fn verify_chain(&self) -> bool {
        let mut current = { *self.last_hash.lock().await };
        let storage = self.storage.lock().await;

        while let Some(hash) = current {
            if let Some(stored) = storage.get(&hash) {
                let mut hasher = Sha256::new();
                hasher.update(&stored.data);
                hasher.update(&stored.prev_hash);
                hasher.update(&stored.stored_at.to_le_bytes());
                let expected_hash_result = hasher.finalize();
                let mut expected_hash = [0u8; 32];
                expected_hash.copy_from_slice(&expected_hash_result);

                if expected_hash != hash {
                    error!("Integrity break at hash {:?}", hash);
                    return false;
                }

                if stored.prev_hash == [0u8; 32] {
                    break;
                }
                current = Some(stored.prev_hash);
            } else {
                error!("Handover not found: {:?}", hash);
                return false;
            }
        }
        true
    }
}
