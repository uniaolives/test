pub struct PhiKnob {
    pub phi: f64,
}

impl PhiKnob {
    pub fn new(phi: f64) -> Self {
        Self { phi }
    }

    pub async fn on_phi_change(&self, _new_phi: f64) {
    }
}
