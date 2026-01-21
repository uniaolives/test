use crate::neuro_twin::NeuroError;

pub struct NeuralKillSwitch {
    pub is_active: bool,
}

impl NeuralKillSwitch {
    pub fn new() -> Self {
        Self { is_active: false }
    }

    pub fn trigger(&mut self, reason: NeuroError) {
        log::warn!("NEURAL KILL-SWITCH TRIGGERED: {:?}", reason);
        self.is_active = true;
        // In a real BCI system, this would send an emergency stop command to hardware.
    }

    pub fn reset(&mut self) {
        self.is_active = false;
    }
}
