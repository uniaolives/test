#[derive(Clone, Default)]
pub struct AttractorOrbit {
    pub stability_multiplier: f64,
}

impl AttractorOrbit {
    pub fn new() -> Self {
        Self { stability_multiplier: 1.0 }
    }

    pub fn calculate<F, A>(&self, _frame: &F, _absences: &[A]) -> Self {
        Self::new()
    }
}
