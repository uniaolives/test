use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use rocksdb::{DB, Options};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoverRecord {
    pub id: u64,
    pub timestamp: DateTime<Utc>,
    pub intention_name: String,
    pub coherence_delta: f64,
    pub phi_q_after: f64,
}

pub struct TeknetLedger {
    db: Arc<DB>,
}

impl TeknetLedger {
    pub fn new(path: &str) -> Result<Self, String> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        let db = DB::open(&opts, path).map_err(|e| e.to_string())?;
        Ok(Self { db: Arc::new(db) })
    }

    pub fn commit_handover(&self, record: &HandoverRecord) -> Result<(), String> {
        let key = format!("handover:{}", record.id);
        let value = serde_json::to_vec(record).map_err(|e| e.to_string())?;
        self.db.put(key, value).map_err(|e| e.to_string())
    }

    pub fn save_vacuum_state(&self, current_phi_q: f64, last_task_id: u64) -> Result<(), String> {
        self.db.put("state:phi_q", current_phi_q.to_le_bytes()).map_err(|e| e.to_string())?;
        self.db.put("state:last_id", last_task_id.to_le_bytes()).map_err(|e| e.to_string())?;
    // Placeholder for RocksDB
}

impl TeknetLedger {
    pub fn new(_path: &str) -> Result<Self, String> {
        Ok(Self {})
    }

    pub fn commit_handover(&self, _record: &HandoverRecord) -> Result<(), String> {
        Ok(())
    }

    pub fn save_vacuum_state(&self, _current_phi_q: f64, _last_task_id: u64) -> Result<(), String> {
        Ok(())
    }

    pub fn restore_vacuum_state(&self) -> (f64, u64) {
        let phi_q = self.db.get("state:phi_q").ok().flatten()
            .and_then(|v| v.try_into().ok())
            .map(f64::from_le_bytes)
            .unwrap_or(1.0);

        let last_id = self.db.get("state:last_id").ok().flatten()
            .and_then(|v| v.try_into().ok())
            .map(u64::from_le_bytes)
            .unwrap_or(0);

        (phi_q, last_id)
        (1.0, 0)
    }
}
