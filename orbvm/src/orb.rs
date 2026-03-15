use std::f64::consts::PI;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use crate::{Error, Result, GOLDEN_RATIO, PI_DAY_2026, TARGET_2140};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbPayload {
    pub id: [u8; 32],
    pub coherence: Complex64,
    pub position: [f64; 3],
    pub oam_sequence: Vec<u32>,
    pub emission_time: u64,
    pub target_time: u64,
    pub lambda_2: f64,
    pub berry_phase: f64,
    pub potential_f: Vec<Complex64>,
    pub potential_g: Vec<Complex64>,
    pub nonce: u64,
    pub signature: Option<Vec<u8>>,
}

impl Default for OrbPayload {
    fn default() -> Self {
        Self::genesis()
    }
}

impl OrbPayload {
    pub fn genesis() -> Self {
        Self {
            id: [0u8; 32],
            coherence: Complex64::new(GOLDEN_RATIO, 0.0),
            position: [-22.9068, -43.1729, 0.0],
            oam_sequence: vec![3, 1, 31],
            emission_time: PI_DAY_2026 * 1_000_000_000,
            target_time: TARGET_2140 * 1_000_000_000,
            lambda_2: GOLDEN_RATIO,
            berry_phase: PI / 2.0,
            potential_f: Vec::new(),
            potential_g: Vec::new(),
            nonce: 0,
            signature: None,
        }
    }

    pub fn new(
        position: [f64; 3],
        oam_sequence: Vec<u32>,
        coherence: Complex64,
    ) -> Self {
        Self {
            id: [0u8; 32],
            coherence,
            position,
            oam_sequence,
            emission_time: 1742241655 * 1_000_000_000,
            target_time: TARGET_2140 * 1_000_000_000,
            lambda_2: 0.0,
            berry_phase: PI / 2.0,
            potential_f: Vec::new(),
            potential_g: Vec::new(),
            nonce: 0,
            signature: None,
        }
    }

    pub fn compute_id(&mut self) {
        let mut hasher = Sha3_256::new();
        hasher.update(&self.coherence.re.to_le_bytes());
        hasher.update(&self.coherence.im.to_le_bytes());
        for p in &self.position {
            hasher.update(&p.to_le_bytes());
        }
        let result = hasher.finalize();
        self.id.copy_from_slice(&result[..32]);
    }

    pub fn is_coherent(&self) -> bool {
        self.lambda_2 > GOLDEN_RATIO
    }

    pub fn inner_product(&self, other: &OrbPayload) -> Complex64 {
        self.coherence * other.coherence.conj()
    }
}

pub struct OrbVM {
    pub config: OrbVMConfig,
    pub active_orbs: Vec<OrbPayload>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrbVMConfig {
    pub n_oscillators: usize,
    pub coherence_threshold: f64,
}

impl Default for OrbVMConfig {
    fn default() -> Self {
        Self {
            n_oscillators: 100,
            coherence_threshold: GOLDEN_RATIO,
        }
    }
}

impl OrbVM {
    pub fn new(config: OrbVMConfig) -> Self {
        Self {
            active_orbs: Vec::new(),
            config,
        }
    }
}
