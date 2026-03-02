use std::time::{SystemTime, UNIX_EPOCH};

pub struct HybridClock {
    physical: i64,
    logical: u32,
}

impl HybridClock {
    pub fn new() -> Self {
        let physical = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;
        Self { physical, logical: 0 }
    }

    pub fn tick(&mut self) -> (i64, u32) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;
        if now > self.physical {
            self.physical = now;
            self.logical = 0;
        } else {
            self.logical += 1;
        }
        (self.physical, self.logical)
    }

    pub fn update(&mut self, received_physical: i64, received_logical: u32) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;

        let old_physical = self.physical;
        self.physical = old_physical.max(now).max(received_physical);

        if self.physical == old_physical && self.physical == received_physical {
            self.logical = self.logical.max(received_logical) + 1;
        } else if self.physical == old_physical {
            self.logical += 1;
        } else if self.physical == received_physical {
            self.logical = received_logical + 1;
        } else {
            self.logical = 0;
        }
    }

    pub fn get_time(&self) -> (i64, u32) {
        (self.physical, self.logical)
    }
}
