// arkhe-axos-instaweb/src/axos/kernel.rs
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Task {
    pub id: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Result {
    pub status: String,
    pub data: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LogEntry {
    pub timestamp_ns: u128,
    pub agent_id: String,
    pub task_id: String,
    pub result_status: String,
    pub determinism_hash: String,
}

pub struct AxosKernel {
    pub execution_log: Vec<LogEntry>,
}

impl AxosKernel {
    pub fn new() -> Self {
        Self {
            execution_log: Vec::new(),
        }
    }

    pub fn execute_task(&mut self, agent_id: &str, task: Task) -> Result {
        // Deterministic execution simulation
        let result = Result {
            status: "SUCCESS".to_string(),
            data: format!("Processed: {}", task.content),
        };

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        let mut hasher = Sha256::new();
        hasher.update(task.id.as_bytes());
        hasher.update(result.status.as_bytes());
        hasher.update(result.data.as_bytes());
        let hash = format!("{:x}", hasher.finalize());

        let entry = LogEntry {
            timestamp_ns: timestamp,
            agent_id: agent_id.to_string(),
            task_id: task.id.clone(),
            result_status: result.status.clone(),
            determinism_hash: hash,
        };

        self.execution_log.push(entry);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_kernel() {
        let mut kernel = AxosKernel::new();
        let task = Task { id: "1".to_string(), content: "test".to_string() };
        let res = kernel.execute_task("agent_alpha", task);
        assert_eq!(res.status, "SUCCESS");
        assert_eq!(kernel.execution_log.len(), 1);
        assert!(kernel.execution_log[0].determinism_hash.len() > 0);
    }
}
