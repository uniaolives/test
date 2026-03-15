// src/temporal/mobius_chain.rs
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};

use sha3::{Digest as _, Sha3_256};
use crate::physics::mobius_temporal::MobiusTemporalSurface;

pub struct MobiusBlock {
    pub timestamp: i64,
    pub prev_hash: [u8; 32],
    pub data: Vec<u8>,
    pub twist: f64,
}

impl MobiusBlock {
    pub fn calculate_hash(&self) -> [u8; 32] {
        // let mut _hasher = Sha3_256::new();
        // hash content
        [0u8; 32]
    }
    /// Create a block in a state of temporal superposition
    pub fn create_superposed(data: Vec<u8>, temporal_phase: f64) -> Self {
        // temporal_phase: 0.0 = "past", 1.0 = "future", 0.5 = "present"
        // But on the Möbius strip, 0.0 and 1.0 are adjacent!

        let mut hasher = Sha3_256::new();
        hasher.update(&data);
        let hash_vec = hasher.finalize().to_vec();
        let mut base_hash = [0u8; 32];
        base_hash.copy_from_slice(&hash_vec);

        let (hash, mobius_link) = if temporal_phase > 0.5 {
            // We are in the "future", our hash is the transformed one,
            // and we link back to the "past" (base)
            let h = Self::compute_future_equivalent(&base_hash);
            (h, base_hash)
        } else {
            // We are in the "past", our hash is the base,
            // and we link to the "future" (transformed)
            let ml = Self::compute_future_equivalent(&base_hash);
            (base_hash, ml)
        };

    pub fn create_superposed(data: Vec<u8>, twist: f64) -> Self {
        Self {
            timestamp: 0,
            prev_hash: [0u8; 32],
            data,
            twist,
        }
    }

    pub fn are_mobius_equivalent(a: &Self, b: &Self) -> bool {
        a.data == b.data && (a.twist - b.twist).abs() > 0.9 // Simplified
    }
}

pub struct MobiusChain {
    pub blocks: Vec<MobiusBlock>,
    pub topology: MobiusTemporalSurface,
}

impl MobiusChain {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            topology: MobiusTemporalSurface::new(),
        }
    }
}
