use std::collections::BinaryHeap;
use std::cmp::Ordering;
use crate::physics::miller::{PHI_Q_CRITICAL, VacuumState};

pub struct Task {
    pub id: u64,
    pub description: String,
    pub coherence_load: f64,
    pub priority: u8,
}

impl Eq for Task {}
impl PartialEq for Task {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Ord for Task {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
            .then_with(|| self.coherence_load.partial_cmp(&other.coherence_load).unwrap_or(Ordering::Equal))
    }
}
impl PartialOrd for Task {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct CoherenceScheduler {
    queue: BinaryHeap<Task>,
    vacuum: VacuumState,
    total_interest_paid: f64,
}

impl CoherenceScheduler {
    pub fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
            vacuum: VacuumState::new(),
            total_interest_paid: 0.0,
        }
    }

    pub fn submit(&mut self, task: Task) {
        self.queue.push(task);
    }

    pub fn tick(&mut self) -> Result<String, String> {
        if let Some(task) = self.queue.pop() {
            let projected_phi = self.vacuum.current_phi_q + task.coherence_load;
            if projected_phi > PHI_Q_CRITICAL {
                return Err(format!("MILLER_LIMIT_EXCEEDED: Task {} (phi_q = {:.3})", task.id, projected_phi));
            }
            self.vacuum.current_phi_q = projected_phi;
            if task.coherence_load < 0.0 {
                let interest = self.vacuum.calculate_interest(task.coherence_load.abs(), 1.0);
                self.total_interest_paid += interest;
            }
            return Ok(format!("Task {} executed. phi_q: {:.3}", task.id, self.vacuum.current_phi_q));
        }
        Ok("Idle".to_string())
    }
}
