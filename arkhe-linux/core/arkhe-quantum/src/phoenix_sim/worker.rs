use super::{Task, FoldingResult};
use log::debug;
use rand::Rng;

pub struct Worker {
    pub id: String,
}

impl Worker {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
        }
    }

    pub fn execute_vqe(&self, task: &Task) -> FoldingResult {
        let mut rng = rand::thread_rng();
        let energy: f64 = rng.gen_range(-100.0..-50.0);
        let confidence: f64 = rng.gen_range(0.9..1.0);

        debug!("Worker {} executed VQE for {}", self.id, task.id);

        FoldingResult {
            task_id: task.id,
            energy,
            confidence,
        }
    }
}
