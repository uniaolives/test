pub struct QuantumClock {
    pub physical: u64,
    pub logical: u32,
}

impl QuantumClock {
    pub fn tick(&mut self) {
        self.logical += 1;
    }
}
