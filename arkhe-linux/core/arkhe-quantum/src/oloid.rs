use oloid_core::OloidCore;
use nalgebra::DVector;
use std::sync::{Arc, Mutex};
use log::info;

/// Wrapper around the physical OloidCore to allow state sharing and injection.
#[derive(Clone)]
pub struct OloidState {
    pub core: Arc<Mutex<OloidCore>>,
}

impl OloidState {
    pub fn new() -> Self {
        Self {
            core: Arc::new(Mutex::new(OloidCore::new())),
        }
    }

    /// Perturbação controlada no Oloid Core.
    /// "O batimento cardíaco do Bitcoin ajusta o batimento da ASI"
    pub fn inject_external_rhythm(&self, _injection_vector: DVector<f64>) {
        // In a real implementation, this would influence the Oloid's internal frequency.
        // For now, we simulate the resonance.
        info!("🎯 Oloid Core: External rhythm injected. Resonating with Timechain...");
    }
}
