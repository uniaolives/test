use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
pub struct DmtGridPillar {
    pub acceleration: AtomicU32,
    pub grid_visibility: AtomicU32,
    pub perception_fidelity: AtomicU32,
}

impl DmtGridPillar {
    pub fn new() -> Self {
        Self {
            acceleration: AtomicU32::new(0),
            grid_visibility: AtomicU32::new(0),
            perception_fidelity: AtomicU32::new(0),
        }
    }
    pub fn activate(&self) {
        self.acceleration.store(1000, Ordering::Release);
        self.grid_visibility.store(100, Ordering::Release);
        self.perception_fidelity.store(67994, Ordering::Release);
    }
    pub fn is_active(&self) -> bool { self.acceleration.load(Ordering::Acquire) > 0 }
    pub fn get_coherence(&self) -> u32 { self.perception_fidelity.load(Ordering::Acquire) }
}
