use crate::propagation::payload::OrbPayload;

pub struct SuperposedOrb {
    pub components: Vec<OrbPayload>,
    pub combined_lambda: f64,
    pub combined_phi: f64,
    pub coherence_length: f64,
}

pub struct MacroscopicQuantumBridge {
    pub coherence_scale: f64,
}

impl MacroscopicQuantumBridge {
    pub fn new() -> Self {
        Self { coherence_scale: 1e6 } // Macroscopic scale
    }

    pub fn superpose(&self, orbs: &[OrbPayload]) -> Option<SuperposedOrb> {
        if orbs.is_empty() { return None; }

        let mut combined_lambda = 0.0;
        let mut combined_phi = 0.0;

        for orb in orbs {
            combined_lambda += orb.lambda_2;
            combined_phi += orb.phi_q;
        }

        let n = orbs.len() as f64;

        Some(SuperposedOrb {
            components: orbs.to_vec(),
            combined_lambda: combined_lambda / n,
            combined_phi: combined_phi / n,
            coherence_length: self.coherence_scale,
        })
    }
}
