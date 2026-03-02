use std::time::Duration;
use tokio::time::Instant;
use chrono::{DateTime, Utc};

pub struct HarmonicConcordance {
    pub cycle_duration: Duration,
    pub next_cycle: Instant,
}

pub struct CycleState {
    pub phi_coherence: f64,
    pub timestamp: DateTime<Utc>,
}

impl HarmonicConcordance {
    pub fn new() -> Self {
        Self {
            cycle_duration: Duration::from_millis(8640),
            next_cycle: Instant::now() + Duration::from_millis(8640),
        }
    }

    pub async fn wait_for_next_cycle(&mut self) {
        let now = Instant::now();
        if now < self.next_cycle {
            tokio::time::sleep(self.next_cycle - now).await;
        }
        self.next_cycle += self.cycle_duration;
    }

    pub async fn synchronize_cycle(&mut self) -> CycleState {
        self.wait_for_next_cycle().await;
        CycleState {
            phi_coherence: 0.85 + (rand::random::<f64>() * 0.05),
            timestamp: Utc::now(),
        }
    }
}
