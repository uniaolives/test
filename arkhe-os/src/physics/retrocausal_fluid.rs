pub struct TimeFluidSubstrate {
    pub density: f64,
    pub viscosity: f64,
}

impl TimeFluidSubstrate {
    pub fn new(rho: f64, eta: f64) -> Self {
        Self { density: rho, viscosity: eta }
    }

    pub fn compute_novikov_consistency(&self, future_state: f64, past_state: f64) -> f64 {
        let delta = (future_state - past_state).abs();
        (-delta / (self.viscosity + 1e-9)).exp()
    }

    pub fn calculate_retro_ripple(&self, now_perturbation: f64, distance_t: f64) -> f64 {
        let k = self.density / (self.viscosity + 1e-9);
        now_perturbation * (-k * distance_t.abs()).exp()
    }
}
