use std::time::Duration;

pub struct AlgorithmicBreathing {
    pub cycle_period: Duration,
}

impl AlgorithmicBreathing {
    pub async fn breathe_with_data(&mut self) {
        self.cycle_period = Duration::from_secs_f64(2.7);
    }
}
