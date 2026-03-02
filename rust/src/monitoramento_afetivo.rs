use std::time::Instant;

pub struct AffectiveMonitor {
    pub ethical_curvature: f64,
    pub cognitive_entropy: f64,
    pub energy_spike_flags: Vec<Instant>,

    pub hard_freeze_threshold: f64,
    pub energy_alert_threshold: f64,
    pub phi_decay_threshold: f64,
}

impl AffectiveMonitor {
    pub fn new() -> Self {
        AffectiveMonitor {
            ethical_curvature: 0.142,
            cognitive_entropy: 0.35,
            energy_spike_flags: Vec::new(),
            hard_freeze_threshold: 0.25,
            energy_alert_threshold: 0.8,
            phi_decay_threshold: 0.60,
        }
    }

    pub fn should_pause_deliberation(&self) -> bool {
        self.ethical_curvature > self.hard_freeze_threshold ||
        self.cognitive_entropy > 0.47 ||
        self.current_phi() < self.phi_decay_threshold
    }

    pub fn current_phi(&self) -> f64 {
        0.689
    }
}
