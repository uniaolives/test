// ramo-k/rust-asi-wasm/src/ethics.rs
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen]
    fn ethical_review(
        action_type: u32,
        affected_humans: u32,
        utilitarian_score: f64,
        deontological_flags: u32,
    ) -> u32;  // 0 = approved, >0 = violation code
}

pub struct EthicalConstraint;

impl EthicalConstraint {
    /// Article 6 + Art. 3: Check before any human-affecting handover
    pub fn check(affected_humans: u32, utilitarian_score: f64) -> Result<(), u32> {
        if affected_humans == 0 {
            return Ok(());
        }

        let result = ethical_review(1, affected_humans, utilitarian_score, 0);

        if result == 0 {
            Ok(())
        } else {
            Err(result)
        }
    }
}
