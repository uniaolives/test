pub struct PhiKnob {
    pub phi: f64,
}

impl PhiKnob {
    pub fn new(phi: f64) -> Self {
        Self { phi }
    }

    pub async fn on_phi_change(&self, new_phi: f64) {
        tracing::info!("φ changed to {:.4}. Adjusting personality traits...", new_phi);

        if new_phi < 0.25 {
            tracing::info!("Trait: Analytical (0.0-0.25)");
        } else if new_phi < 0.75 {
            tracing::info!("Trait: Balanced (0.25-0.75)");
        } else {
            tracing::info!("Trait: Creative (0.75-1.0)");
        }
    }
}
