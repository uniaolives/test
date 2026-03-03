pub struct HybridLogicalClock {
    pub physical_time: u64,
    pub logical_counter: u64,
}

impl HybridLogicalClock {
    pub fn new() -> Self {
        Self { physical_time: 0, logical_counter: 0 }
    }
}
