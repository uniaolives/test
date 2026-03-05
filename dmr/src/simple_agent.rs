//! Simple demonstration agent using Digital Memory Ring.

use crate::{DigitalMemoryRing, KatharosVector};
use rand::Rng;
use std::thread;
use std::time::Duration;

pub struct SimpleAgent {
    pub id: String,
    pub memory: DigitalMemoryRing,
    pub current_vk: KatharosVector,
    pub vk_ref: KatharosVector,
}

impl SimpleAgent {
    pub fn new(id: String) -> Self {
        let vk_ref = KatharosVector::new(0.5, 0.5, 0.5, 0.5);
        let memory = DigitalMemoryRing::new(
            id.clone(),
            vk_ref.clone(),
            Duration::from_secs(3600), // 1‑hour layers
        );
        Self {
            id,
            memory,
            current_vk: vk_ref.clone(),
            vk_ref,
        }
    }

    /// Update state based on internal dynamics and external perturbations.
    pub fn update(&mut self, external_stress: f64) {
        // Simple homeostatic drift: always tries to return to vk_ref.
        let mut new_vk = self.current_vk.clone();
        new_vk.bio += (self.vk_ref.bio - new_vk.bio) * 0.1;
        new_vk.aff += (self.vk_ref.aff - new_vk.aff) * 0.1;
        new_vk.soc += (self.vk_ref.soc - new_vk.soc) * 0.1;
        new_vk.cog += (self.vk_ref.cog - new_vk.cog) * 0.1;

        // Add random fluctuations and external stress.
        let mut rng = rand::thread_rng();
        new_vk.bio += rng.gen_range(-0.02..0.02) + external_stress * 0.1;
        new_vk.aff += rng.gen_range(-0.02..0.02);
        new_vk.soc += rng.gen_range(-0.02..0.02);
        new_vk.cog += rng.gen_range(-0.02..0.02);

        // Clamp to plausible range [0,1].
        new_vk.bio = new_vk.bio.clamp(0.0, 1.0);
        new_vk.aff = new_vk.aff.clamp(0.0, 1.0);
        new_vk.soc = new_vk.soc.clamp(0.0, 1.0);
        new_vk.cog = new_vk.cog.clamp(0.0, 1.0);

        self.current_vk = new_vk;

        // Compute qualic permeability (simplified).
        let delta_k = self.vk_ref.weighted_distance(&self.current_vk);
        let q = 1.0 / (1.0 + (delta_k * 5.0).exp()); // sigmoid

        // Grow memory layer.
        self.memory.grow_layer(self.current_vk.clone(), q, vec![]);
    }

    pub fn run_simulation(mut self, days: u64) -> Self {
        let hours = days * 24;
        for hour in 0..hours {
            // Simulate a stressful event at day 3.
            let stress = if hour > 72 && hour < 80 { 0.5 } else { 0.0 };
            self.update(stress);
            thread::sleep(Duration::from_millis(1)); // simulate time passing
        }
        self
    }
}
