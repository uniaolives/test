use crate::{ArkheError, Result};
use sha2::{Sha256, Digest};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Write};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub handover_id: u64,
    pub timestamp: u64,
    pub emitter: String,
    pub receiver: String,
    pub coherence: f64,
    pub payload_hash: [u8; 32],
    pub previous_hash: [u8; 32],
}

pub struct Ledger {
    file: File,
    entries: Vec<LedgerEntry>,
    index: HashMap<u64, usize>,
}

impl Ledger {
    pub fn create(path: &str) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| ArkheError::FfiError(e.to_string()))?;

        Ok(Self {
            file,
            entries: Vec::new(),
            index: HashMap::new(),
        })
    }

    pub fn append(
        &mut self,
        handover_id: u64,
        emitter: &str,
        receiver: &str,
        coherence: f64,
        payload: &[u8],
    ) -> Result<()> {
        let mut hasher = Sha256::new();
        hasher.update(payload);
        let payload_hash: [u8; 32] = hasher.finalize().into();

        let previous_hash = [0u8; 32]; // Simplificado

        let entry = LedgerEntry {
            handover_id,
            timestamp: 0, // Simplificado
            emitter: emitter.to_string(),
            receiver: receiver.to_string(),
            coherence,
            payload_hash,
            previous_hash,
        };

        self.index.insert(handover_id, self.entries.len());
        self.entries.push(entry.clone());

        let bytes = bincode::serialize(&entry).map_err(|_| ArkheError::LedgerCorruption)?;
        self.file.write_all(&bytes).map_err(|_| ArkheError::LedgerCorruption)?;
        self.file.flush().map_err(|_| ArkheError::LedgerCorruption)?;

        Ok(())
    }
}
