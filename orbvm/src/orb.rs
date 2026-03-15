use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use crate::GOLDEN_RATIO;

const PI_DAY_2026: i64 = 1742241655;
const TARGET_2140: i64 = 4584533760;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbPayload {
    pub id: [u8; 32],
    pub coherence: Complex64,
    pub position: [f64; 3],
    pub oam_sequence: Vec<u32>,
    pub emission_time: i64,
    pub target_time: i64,
    pub lambda_2: f64,
}

impl OrbPayload {
    pub fn genesis() -> Self {
        let mut orb = Self {
            id: [0u8; 32],
            coherence: Complex64::new(GOLDEN_RATIO, 0.0),
            position: [-22.9068, -43.1729, 0.0],
            oam_sequence: vec![3, 1, 31],
            emission_time: PI_DAY_2026,
            target_time: TARGET_2140,
            lambda_2: GOLDEN_RATIO,
        };
        orb.compute_id();
        orb
    }

    pub fn new(
        position: [f64; 3],
        oam_sequence: Vec<u32>,
        coherence: Complex64,
    ) -> Self {
        let mut orb = Self {
            id: [0u8; 32],
            coherence,
            position,
            oam_sequence,
            emission_time: chrono::Utc::now().timestamp(),
            target_time: TARGET_2140,
            lambda_2: 0.0,
        };
        orb.compute_id();
        orb
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
}

pub struct OrbVM {
    pub active_orbs: Vec<OrbPayload>,
}

impl OrbVM {
    pub fn new(_config: crate::OrbVMConfig) -> Self {
        Self { active_orbs: Vec::new() }
    }

    pub fn emit(&mut self, orb: OrbPayload) -> anyhow::Result<()> {
        println!("[OrbVM] Emitting Orb: {}", hex::encode(&orb.id[..8]));
        self.active_orbs.push(orb);
        Ok(())
    }

    pub fn get_active_orbs(&self) -> &[OrbPayload] {
        &self.active_orbs
    }
}
