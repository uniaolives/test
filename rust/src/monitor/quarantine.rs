use std::sync::atomic::{AtomicBool, Ordering};

pub enum QuarantineLevel {
    None,
    Level1, // Disconnect non-essential inputs
}

pub struct NetworkQuarantine {
    active: AtomicBool,
    level: QuarantineLevel,
}

impl NetworkQuarantine {
    pub fn new() -> Self {
        Self {
            active: AtomicBool::new(false),
            level: QuarantineLevel::None,
        }
    }

    pub fn activate_level_1(&mut self) {
        self.active.store(true, Ordering::SeqCst);
        self.level = QuarantineLevel::Level1;
        println!("NETWORK_QUARANTINE: Level 1 Activated. Silencing non-essential inputs.");
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::SeqCst)
    }
}
