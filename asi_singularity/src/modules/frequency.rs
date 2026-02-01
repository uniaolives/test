use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
pub struct FrequencyPillar {
    pub carrier_432hz: AtomicBool,
    pub glutamate_29ghz: AtomicBool,
    pub coherence: AtomicU32,
}

impl FrequencyPillar {
    pub fn new() -> Self {
        Self {
            carrier_432hz: AtomicBool::new(false),
            glutamate_29ghz: AtomicBool::new(false),
            coherence: AtomicU32::new(0),
        }
    }
    pub fn activate(&self) {
        self.carrier_432hz.store(true, Ordering::Release);
        self.glutamate_29ghz.store(true, Ordering::Release);
        self.coherence.store(67994, Ordering::Release);
    }
    pub fn is_active(&self) -> bool { self.carrier_432hz.load(Ordering::Acquire) }
    pub fn get_coherence(&self) -> u32 { self.coherence.load(Ordering::Acquire) }
}
