use core::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering};
pub struct TopologyPillar {
    pub nodes_alive: AtomicU32,
    pub scars: [AtomicU32; 2], // 104, 277
    pub genus: AtomicU8,
}

impl TopologyPillar {
    pub fn new() -> Self {
        Self {
            nodes_alive: AtomicU32::new(0),
            scars: [AtomicU32::new(0), AtomicU32::new(0)],
            genus: AtomicU8::new(0),
        }
    }
    pub fn activate(&self) {
        self.nodes_alive.store(271, Ordering::Release);
        self.scars[0].store(104, Ordering::Release);
        self.scars[1].store(277, Ordering::Release);
        self.genus.store(1, Ordering::Release);
    }
    pub fn is_active(&self) -> bool { self.nodes_alive.load(Ordering::Acquire) > 0 }
    pub fn get_coherence(&self) -> u32 { 67994 }
}
