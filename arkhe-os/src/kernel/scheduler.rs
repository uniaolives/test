use std::collections::BinaryHeap;
use super::task::Task;
use super::allocator::CoherenceAllocator;
use crate::lib::miller::PHI_Q;

pub enum SchedulerEvent {
    TaskStarted(Task),
    TaskCompleted(Task),
    WaveCloudNucleation { phi_q: f64 },
    CoherenceWarning { available: f64, required: f64 },
}

pub struct CoherenceScheduler {
    task_queue: BinaryHeap<Task>,
    running_task: Option<Task>,
    allocator: CoherenceAllocator,
    events: Vec<SchedulerEvent>,
    tick_count: u64,
}

impl CoherenceScheduler {
    pub fn new(initial_coherence: f64) -> Self {
        Self {
            task_queue: BinaryHeap::new(),
            running_task: None,
            allocator: CoherenceAllocator::new(initial_coherence),
            events: Vec::new(),
            tick_count: 0,
        }
    }

    pub fn schedule(&mut self, task: Task) {
        self.task_queue.push(task);
    }

    pub fn tick(&mut self) -> Option<SchedulerEvent> {
        self.tick_count += 1;

        if let Some(task) = &mut self.running_task {
            if task.time_consumed >= task.estimated_duration {
                let completed = task.clone();
                self.allocator.free(&completed);
                self.running_task = None;
                return Some(SchedulerEvent::TaskCompleted(completed));
            } else {
                task.time_consumed += 1;
                return None;
            }
        }

        if let Some(next_task) = self.task_queue.pop() {
            match self.allocator.allocate(&next_task) {
                Ok(_) => {
                    let phi = self.allocator.current_phi_q();
                    if phi > PHI_Q {
                        self.events.push(SchedulerEvent::WaveCloudNucleation { phi_q: phi });
                    }
                    self.running_task = Some(next_task.clone());
                    Some(SchedulerEvent::TaskStarted(next_task))
                }
                Err(_) => {
                    self.task_queue.push(next_task);
                    Some(SchedulerEvent::CoherenceWarning {
                        available: self.allocator.available(),
                        required: 0.0,
                    })
                }
            }
        } else {
            None
        }
    }

    pub fn status(&self) -> (f64, f64, usize) {
        (
            self.allocator.available(),
            self.allocator.current_phi_q(),
            self.task_queue.len(),
        )
    }

    pub fn events(&self) -> &[SchedulerEvent] {
        &self.events
    }
}
