use crate::propagation::payload::OrbPayload;

pub struct ReverseEntropyBridge {
    /// Negative rate of entropy change
    pub ds_dt: f64,
}

impl ReverseEntropyBridge {
    pub fn new() -> Self {
        Self { ds_dt: -0.1 }
    }

    pub fn propagate(&self, orb: &OrbPayload, dt: f64) -> OrbPayload {
        let mut evolved = orb.clone();

        // Coherence increases as time passes in reverse entropy
        evolved.lambda_2 += (-self.ds_dt) * dt * 0.001;
        evolved.lambda_2 = evolved.lambda_2.min(1.5); // Superunitary limit

        // Efficiency increases
        evolved.h_value -= 0.01 * dt;
        evolved.h_value = evolved.h_value.max(0.0);

        evolved
    }
}
