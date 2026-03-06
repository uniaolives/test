//! zkLottery: Verifiable fairness for the Digital Memory Ring and Arkhe(n) framework.

use bls12_381::{G1Affine, Scalar, G1Projective, G2Projective};
use group::Curve;
use serde::{Deserialize, Serialize};
use crate::DigitalMemoryRing;
use std::time::SystemTime;
use sha2::{Sha256, Digest};

/// Proof for the Verifiable Random Function (VRF).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VrfProof {
    /// The public output (randomness value)
    pub output: Vec<u8>,
    /// The proof point (Gamma in ECVRF)
    pub gamma: Vec<u8>,
    /// The proof scalar (c in ECVRF)
    pub c: Vec<u8>,
    /// The proof scalar (s in ECVRF)
    pub s: Vec<u8>,
}

impl VrfProof {
    /// Extract a float in [0..1] from the VRF output.
    pub fn extract_f64(&self) -> f64 {
        if self.output.is_empty() {
            return 0.0;
        }
        let mut bytes = [0u8; 8];
        let len = self.output.len().min(8);
        bytes[..len].copy_from_slice(&self.output[..len]);
        let val = u64::from_le_bytes(bytes);
        (val as f64) / (u64::MAX as f64)
    }
}

/// Verifiable Random Function (VRF) for zkLottery using BLS12-381.
/// Based on a simplified ECVRF construction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZkLotteryVRF {
    /// Public key (point in G2)
    pub public_key: Vec<u8>,
}

impl ZkLotteryVRF {
    pub fn new(sk_bytes: &[u8; 32]) -> (Self, Scalar) {
        let mut sk_wide = [0u8; 64];
        sk_wide[..32].copy_from_slice(sk_bytes);
        let sk = Scalar::from_bytes_wide(&sk_wide);
        let pk = G2Projective::generator() * sk;
        let pk_affine = pk.to_affine();

        (Self {
            public_key: pk_affine.to_compressed().to_vec(),
        }, sk)
    }

    /// Simplified VRF Prove: Gamma = sk * H(m), c = H(pk, Gamma, m, ...), s = ...
    /// For this demonstration, we implement the core EC operations.
    pub fn prove(&self, sk: &Scalar, input: &[u8]) -> VrfProof {
        // 1. Hash to curve (G1)
        let h = hash_to_g1(input);

        // 2. Gamma = sk * H
        let gamma = h * sk;
        let gamma_affine = gamma.to_affine();
        let gamma_bytes = gamma_affine.to_compressed().to_vec();

        // 3. Generate randomness output (Hash of Gamma)
        let mut hasher = Sha256::new();
        hasher.update(&gamma_bytes);
        let output = hasher.finalize().to_vec();

        // 4. NIZK Proof (c, s) - simplified for demo
        // In a full implementation, this would use a proper Fiat-Shamir heuristic
        let mut c_hasher = Sha256::new();
        c_hasher.update(&self.public_key);
        c_hasher.update(&gamma_bytes);
        c_hasher.update(input);
        let c_bytes = c_hasher.finalize();

        VrfProof {
            output,
            gamma: gamma_bytes,
            c: c_bytes.to_vec(),
            s: vec![0u8; 32], // Simplified: omitted full Schnorr for brevety but point math is real
        }
    }

    /// Verify proof using public key.
    pub fn verify(&self, _input: &[u8], proof: &VrfProof) -> bool {
        // 1. Check if point is in G1
        if G1Affine::from_compressed(&proof.gamma.clone().try_into().unwrap_or([0u8; 48])).is_none().into() {
            return false;
        }

        // 2. Verify output = Hash(Gamma)
        let mut hasher = Sha256::new();
        hasher.update(&proof.gamma);
        if hasher.finalize().to_vec() != proof.output {
            return false;
        }

        // 3. In a full implementation, we would verify the NIZK proof (c, s)
        // using pairings if it were a BLS-VRF, or EC-Schnorr if it were an ECVRF.
        // For now, we've validated the point structure and the deterministic output mapping.
        true
    }
}

/// Helper to hash bytes to a point in G1.
fn hash_to_g1(m: &[u8]) -> G1Projective {
    let mut hasher = Sha256::new();
    hasher.update(m);
    let h = hasher.finalize();
    // Simplified hash-to-curve for demonstration
    // In production, use a standard suite like BLS12381G1_XMD:SHA-256_SSWU_RO_
    let mut bytes = [0u8; 64];
    bytes[..32].copy_from_slice(&h);
    let s = Scalar::from_bytes_wide(&bytes);
    G1Projective::generator() * s
}

pub type ParticipantId = String;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LotteryWeight(pub f64);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LotteryResult {
    pub winner: ParticipantId,
    pub proof: VrfProof,
}

/// Lottery with weighted probabilities and verifiable fairness.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZkLottery {
    pub vrf: ZkLotteryVRF,
    pub participants: Vec<(ParticipantId, LotteryWeight)>,
}

impl ZkLottery {
    pub fn new(participants: Vec<(ParticipantId, LotteryWeight)>) -> Self {
        let (vrf, _) = ZkLotteryVRF::new(&[1u8; 32]); // Use a fixed seed for demo identity
        Self {
            vrf,
            participants,
        }
    }

    /// Conduct lottery using a secret key (simulated).
    pub fn draw_with_sk(&self, sk: &Scalar, seed: &[u8]) -> LotteryResult {
        let total_weight: f64 = self.participants.iter().map(|(_, w)| w.0).sum();

        if total_weight <= 0.0 || self.participants.is_empty() {
            return LotteryResult {
                winner: "None".to_string(),
                proof: self.vrf.prove(sk, seed),
            };
        }

        let proof = self.vrf.prove(sk, seed);
        let randomness = proof.extract_f64() * total_weight;

        let mut cumulative = 0.0;
        for (id, weight) in &self.participants {
            cumulative += weight.0;
            if randomness < cumulative {
                return LotteryResult {
                    winner: id.clone(),
                    proof,
                };
            }
        }

        LotteryResult {
            winner: self.participants.last().unwrap().0.clone(),
            proof,
        }
    }
}

pub trait ZkLotteryGrowth {
    fn lottery_growth(&self, ring: &DigitalMemoryRing) -> GrowthDecision;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GrowthDecision {
    Grow {
        timestamp: SystemTime,
        vrf_proof: VrfProof,
    },
    Wait,
}

impl ZkLotteryGrowth for ZkLottery {
    fn lottery_growth(&self, ring: &DigitalMemoryRing) -> GrowthDecision {
        let (q, delta_k) = if let Some(last) = ring.layers.last() {
            (last.q, last.delta_k)
        } else {
            (1.0, 0.0)
        };

        let weight = q * (1.0 - delta_k).max(0.0);
        let seed = format!("{}-{}-{}", ring.id, ring.layers.len(), SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos());

        // Use simulated SK
        let mut sk_bytes = [0u8; 64];
        sk_bytes[0] = 1;
        let sk = Scalar::from_bytes_wide(&sk_bytes);
        let proof = self.vrf.prove(&sk, seed.as_bytes());
        let threshold = (weight / 1.618).max(0.5);

        if proof.extract_f64() < threshold {
            GrowthDecision::Grow {
                timestamp: SystemTime::now(),
                vrf_proof: proof,
            }
        } else {
            GrowthDecision::Wait
        }
    }
}
