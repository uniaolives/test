use crate::error::OrbVMError;
use crate::orb::OrbPayload;
use crate::memory::PhaseMemory;
use ndarray::Array2;

pub struct OrbRuntime {
    pub memory: PhaseMemory,
    pub active_orbs: Vec<OrbPayload>,
}

impl OrbRuntime {
    pub fn new() -> Self {
        Self {
            memory: PhaseMemory::new(64),
            active_orbs: Vec::new(),
        }
    }

    pub fn process_orb(&mut self, mut orb: OrbPayload) -> Result<f64, OrbVMError> {
        println!("[OrbVM] Processing Orb targeting {}", orb.target_time);

        let n = 64;
        let adj = Array2::eye(n);
        orb.lambda_2 = self.memory.compute_lambda_2(&adj);

        if orb.lambda_2 < 0.01 {
             return Err(OrbVMError::Decoherence(orb.lambda_2));
        }

        println!("[OrbVM] Coherence achieved: λ₂ = {:.4}", orb.lambda_2);
        self.active_orbs.push(orb.clone());
        Ok(orb.lambda_2)
    }

    pub fn commit(&self, tx_id: &str) {
        println!("[OrbVM] Committing transaction {} to Timechain...", tx_id);
    }
}
