use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Handover {
    pub id: Vec<u8>,
    pub type_: u32,
    pub emitter_id: u64,
    pub receiver_id: u64,
    pub entropy_cost: f64,
    pub half_life: f64,
    pub payload: Vec<u8>,
    pub timestamp_physical: i64,
    pub timestamp_logical: u32,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Block {
    pub index: u64,
    pub handover: Handover,
    pub prev_hash: Vec<u8>,
    pub hash: Vec<u8>,
    pub timestamp: i64,
}

pub struct LedgerCore {
    blocks: Vec<Block>,
    index_map: HashMap<Vec<u8>, u64>,
    total_entropy: f64,
}

impl LedgerCore {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            index_map: HashMap::new(),
            total_entropy: 0.0,
        }
    }

    pub fn append(&mut self, handover: Handover) -> Result<(Vec<u8>, u64), String> {
        let index = self.blocks.len() as u64;
        let prev_hash = if index == 0 {
            vec![0; 32]
        } else {
            self.blocks.last().unwrap().hash.clone()
        };

        let mut hasher = Sha256::new();
        hasher.update(&prev_hash);
        hasher.update(&handover.id);
        let hash = hasher.finalize().to_vec();

        let block = Block {
            index,
            handover: handover.clone(),
            prev_hash,
            hash: hash.clone(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as i64,
        };

        self.total_entropy += block.handover.entropy_cost;
        self.index_map.insert(hash.clone(), index);
        self.blocks.push(block);
        Ok((hash, index))
    }

    pub fn range(&self, start: usize, end: usize) -> &[Block] {
        let actual_end = end.min(self.blocks.len());
        if start >= actual_end {
            &[]
        } else {
            &self.blocks[start..actual_end]
        }
    }

    pub fn len(&self) -> u64 {
        self.blocks.len() as u64
    }

    pub fn last_hash(&self) -> Option<Vec<u8>> {
        self.blocks.last().map(|b| b.hash.clone())
    }

    pub fn total_entropy(&self) -> f64 {
        self.total_entropy
    }
}
