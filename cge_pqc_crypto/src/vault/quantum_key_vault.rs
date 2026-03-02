use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};
use crate::algorithms::CryptoError;
use crate::core::PqcConfig;

pub struct QuantumKeyVault {
    memory_store: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

impl QuantumKeyVault {
    pub async fn new(_config: &PqcConfig) -> Result<Self, CryptoError> {
        Ok(Self {
            memory_store: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    pub async fn count_keys(&self) -> Result<usize, CryptoError> {
        Ok(self.memory_store.read().map_err(|_| CryptoError::LockError)?.len())
    }
}
