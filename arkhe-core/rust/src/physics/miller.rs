pub const PHI_Q_CRITICAL: f64 = 4.64;
pub const H_BAR: f64 = 1.054_571_817e-34;

pub struct VacuumState {
    pub density_baseline: f64,
    pub current_phi_q: f64,
}

impl VacuumState {
    pub fn new() -> Self {
        Self {
            density_baseline: 1.0,
            current_phi_q: 1.0,
        }
    }

    pub fn is_wave_cloud_active(&self) -> bool {
        self.current_phi_q > PHI_Q_CRITICAL
    }

    pub fn calculate_interest(&self, debt: f64, duration: f64) -> f64 {
        (debt * duration).exp()
    }
}
