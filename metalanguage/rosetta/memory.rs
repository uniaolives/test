use std::sync::{Arc, Mutex};

struct SharedKnowledgeBase {
    memory_bank: Arc<Mutex<Vec<f32>>>,
}

impl SharedKnowledgeBase {
    fn write_thought(&self, thought: Vec<f32>) {
        let mut bank = self.memory_bank.lock().unwrap();
        bank.extend(thought);
    }
}
