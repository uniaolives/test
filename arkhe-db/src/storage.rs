use std::collections::HashMap;
use crate::handover::Handover;

pub struct HandoverStore {
    data: HashMap<u64, Handover>,
    phi_q_history: Vec<(i64, f64)>,
}

impl HandoverStore {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            phi_q_history: Vec::new(),
        }
    }

    pub fn record(&mut self, handover: Handover) {
        self.phi_q_history.push((handover.timestamp, handover.phi_q_after));
        self.data.insert(handover.id, handover);
    }

    pub fn get(&self, id: u64) -> Option<&Handover> {
        self.data.get(&id)
    }

    pub fn get_history(&self) -> &Vec<(i64, f64)> {
        &self.phi_q_history
    }
}
