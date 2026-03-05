use super::{Task, FoldingResult};
use uuid::Uuid;
use log::info;

pub struct Coordinator {
    pub tasks: Vec<Task>,
    pub results: Vec<FoldingResult>,
}

impl Coordinator {
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            results: Vec::new(),
        }
    }

    pub fn generate_tasks(&mut self, protein: &str, count: u32) {
        for i in 0..count {
            let task = Task {
                id: Uuid::new_v4(),
                target: protein.to_string(),
                complexity: 100,
                region: [i as f64, 0.0, 0.0],
            };
            self.tasks.push(task);
        }
        info!("Generated {} tasks for {}", count, protein);
    }

    pub fn process_result(&mut self, result: FoldingResult) {
        info!("Received result for task {}: energy = {:.4}", result.task_id, result.energy);
        self.results.push(result);
    }
}
