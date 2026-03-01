use serde::{Serialize, Deserialize};
use arkhe_quantum::QuantumState;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ForensicsEntry {
    pub timestamp: u64,
    pub event_type: String,
    pub node_id: Option<[u8; 8]>,
    pub details: String,
    pub state_snapshot: Option<QuantumState>,
}

pub struct ForensicsLedger {
    entries: Vec<ForensicsEntry>,
}

impl ForensicsLedger {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn record(&mut self, entry: ForensicsEntry) {
        self.entries.push(entry);
    }

    pub fn get_entries(&self) -> &Vec<ForensicsEntry> {
        &self.entries
    }
}
