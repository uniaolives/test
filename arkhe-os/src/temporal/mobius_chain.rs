use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};

pub type Hash = [u8; 32];

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MobiusBlock {
    pub hash: Hash,
    pub data: Vec<u8>,
    // In a normal blockchain: prev_hash points to the past
    // In a MobiusChain: prev_hash points to... itself?
    pub mobius_link: Hash, // Points to the "other side" of the strip
}

impl MobiusBlock {
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

        Self {
            hash,
            data,
            mobius_link,
        }
    }

    /// Checks if two blocks are the "same edge" of the strip
    pub fn are_mobius_equivalent(a: &Self, b: &Self) -> bool {
        a.mobius_link == b.hash || b.mobius_link == a.hash
    }

    fn compute_past_equivalent(hash: &Hash) -> Hash {
        let mut equiv = *hash;
        // Simple mock: flip bits or apply a constant transformation for "the other side"
        for byte in equiv.iter_mut() {
            *byte ^= 0xFF;
        }
        equiv
    }

    fn compute_future_equivalent(hash: &Hash) -> Hash {
        let mut equiv = *hash;
        for byte in equiv.iter_mut() {
            *byte ^= 0xFF;
        }
        equiv
    }
}
